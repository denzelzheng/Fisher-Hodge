"""MeshGraphNets (Pfaff, Fortunato, Sanchez-Gonzalez and Battaglia, "Learning Mesh-Based Simulation with Graph
Networks", ICLR 2021) and BSMS-GNN (Cao, Chai, Li and Jiang, "Efficient Learning of Mesh-Based Physical
Simulation with Bi-Stride Multi-Scale Graph Neural Network", ICML 2023), implemented from the papers.

Encode-process-decode on the mesh graph:
  * encoder: node MLP and edge MLP to a latent of size 128 (two hidden layers of 128, ReLU, LayerNorm on the output);
  * processor: 15 message-passing blocks with residual updates,
        e_ij <- e_ij + f_e([e_ij, h_i, h_j]),   h_i <- h_i + f_v([h_i, sum_j e_ji])   (separate weights per block);
  * decoder: node MLP without LayerNorm, predicting the normalised change of the state over one step.
Edge features are relative positions (x_j - x_i, |x_j - x_i|), so the model is not tied to one mesh.  Training uses a
one-step loss on normalised targets with Gaussian noise added to the input state, the target corrected for the noise
(paper Sec. 3 and App. A); at test time the model is rolled out explicitly."""
import torch
import torch.nn as nn


class MLP(nn.Module):
    def __init__(self, i, h, o, layer_norm=True):
        super().__init__()
        self.net = nn.Sequential(nn.Linear(i, h), nn.ReLU(), nn.Linear(h, h), nn.ReLU(), nn.Linear(h, o))
        self.ln = nn.LayerNorm(o) if layer_norm else None

    def forward(self, x):
        y = self.net(x)
        return self.ln(y) if self.ln is not None else y


class MeshGraphNets(nn.Module):
    def __init__(self, n_node_in, n_edge_in, latent=128, steps=15, n_out=1):
        super().__init__()
        self.enc_n = MLP(n_node_in, latent, latent)
        self.enc_e = MLP(n_edge_in, latent, latent)
        self.edge_mlp = nn.ModuleList([MLP(3 * latent, latent, latent) for _ in range(steps)])
        self.node_mlp = nn.ModuleList([MLP(2 * latent, latent, latent) for _ in range(steps)])
        self.dec = MLP(latent, latent, n_out, layer_norm=False)

    def forward(self, xn, xe, snd, rcv):
        """xn (B, V, Fn), xe (B, E, Fe) for the directed edges snd -> rcv (E,); returns (B, V, n_out)"""
        h = self.enc_n(xn)
        e = self.enc_e(xe)
        for fe, fv in zip(self.edge_mlp, self.node_mlp):
            e = e + fe(torch.cat([e, h[:, snd], h[:, rcv]], -1))
            agg = torch.zeros_like(h).index_add(1, rcv, e)
            h = h + fv(torch.cat([h, agg], -1))
        return self.dec(h)


# ------------------------------------------------------------------------------------------ BSMS-GNN
def bistride_hierarchy(edges, pts, levels, period=1.0):
    """Bi-stride pooling (Cao, Chai, Li and Jiang, ICML 2023): breadth-first search from a seed node, keep the nodes on
    every other BFS frontier (even depth), connect the kept nodes that are within two hops on the finer graph.
    Parameter-free; works on any connected graph.  Returns a list of levels, each with numpy arrays
        edges (E_l, 2) undirected edges of level l, pts (V_l, 2), keep (V_{l+1},) indices of the kept nodes in level l,
        pool (P, 2) pairs (coarse node, fine node in its closed 1-hop neighbourhood),
        unpool (Q, 2) pairs (fine node, coarse node): itself if kept, else its kept 1-hop neighbours."""
    import numpy as np
    from collections import deque
    out = []
    E = np.asarray(edges); P = np.asarray(pts)
    for _ in range(levels + 1):
        V = len(P)
        nbr = [[] for _ in range(V)]
        for a, b in E:
            nbr[a].append(b); nbr[b].append(a)
        lev = dict(edges=E, pts=P, n=V)
        out.append(lev)
        if len(out) == levels + 1:
            break
        depth = -np.ones(V, int)
        for s in range(V):                              # all components, seeds in index order
            if depth[s] >= 0:
                continue
            depth[s] = 0; q = deque([s])
            while q:
                u = q.popleft()
                for w in nbr[u]:
                    if depth[w] < 0:
                        depth[w] = depth[u] + 1; q.append(w)
        keep = np.nonzero(depth % 2 == 0)[0]
        cid = -np.ones(V, int); cid[keep] = np.arange(len(keep))
        pool = [(cid[k], k) for k in keep] + [(cid[k], w) for k in keep for w in nbr[k]]
        unpool = [(f, cid[f]) if cid[f] >= 0 else None for f in range(V)]
        unpool = [u for u in unpool if u is not None] + [(f, cid[w]) for f in range(V) if cid[f] < 0 for w in nbr[f] if cid[w] >= 0]
        ce = set()
        for k in keep:
            two = set(nbr[k])
            for w in nbr[k]:
                two.update(nbr[w])
            for m in two:
                if cid[m] >= 0 and m != k:
                    a, b = sorted((cid[k], cid[m]))
                    ce.add((a, b))
        lev['keep'] = keep; lev['pool'] = np.array(pool); lev['unpool'] = np.array(unpool)
        E = np.array(sorted(ce), dtype=np.int64).reshape(-1, 2); P = P[keep]
    return out


def _segment_mean(x, idx_out, idx_in, n_out):
    """mean over pairs: out[i] = mean_{(i, j) in pairs} x[:, j]; x (B, V_in, L)"""
    s = torch.zeros(x.shape[0], n_out, x.shape[2], dtype=x.dtype, device=x.device).index_add(1, idx_out, x[:, idx_in])
    n = torch.zeros(n_out, dtype=x.dtype, device=x.device).index_add(0, idx_out, torch.ones_like(idx_out, dtype=x.dtype))
    return s / n.clamp_min(1)[None, :, None]


class BSMSGNN(nn.Module):
    """BSMS-GNN: a U-Net of message-passing blocks over a bi-stride hierarchy (encoder / decoder / MLP blocks shared with
    MeshGraphNets).  Down path: one message-passing block per level, then parameter-free pooling (mean over the kept node's
    closed neighbourhood); bottom: two blocks; up path: parameter-free unpooling (kept nodes copy, the others average their
    kept neighbours), a skip connection and one block per level.  Coarse edges carry the relative positions of their
    end nodes; the finest edges also carry the input edge features."""
    def __init__(self, n_node_in, n_edge_in, levels=4, latent=128, n_out=1):
        super().__init__()
        self.levels = levels
        self.enc_n = MLP(n_node_in, latent, latent)
        self.enc_e = MLP(n_edge_in, latent, latent)
        self.enc_geo = nn.ModuleList([MLP(3, latent, latent) for _ in range(levels)])
        self.down = nn.ModuleList([nn.ModuleList([MLP(3 * latent, latent, latent), MLP(2 * latent, latent, latent)]) for _ in range(levels)])
        self.bottom = nn.ModuleList([nn.ModuleList([MLP(3 * latent, latent, latent), MLP(2 * latent, latent, latent)]) for _ in range(2)])
        self.up = nn.ModuleList([nn.ModuleList([MLP(3 * latent, latent, latent), MLP(2 * latent, latent, latent)]) for _ in range(levels)])
        self.dec = MLP(latent, latent, n_out, layer_norm=False)

    @staticmethod
    def _mp(blk, h, e, snd, rcv):
        fe, fv = blk
        e = e + fe(torch.cat([e, h[:, snd], h[:, rcv]], -1))
        agg = torch.zeros_like(h).index_add(1, rcv, e)
        return h + fv(torch.cat([h, agg], -1)), e

    def forward(self, xn, xe, H):
        """xn (B, V, Fn); xe (B, E0, Fe) on the directed finest edges H['snd'][0] -> H['rcv'][0]; H: tensors per level
        (snd, rcv, geo for levels >= 1, pool_o / pool_i, unpool_o / unpool_i, n)"""
        h = self.enc_n(xn)
        es, skips = [], []
        e = self.enc_e(xe)
        for l in range(self.levels):
            h, e = self._mp(self.down[l], h, e, H['snd'][l], H['rcv'][l])
            skips.append(h); es.append(e)
            h = _segment_mean(h, H['pool_o'][l], H['pool_i'][l], H['n'][l + 1])
            e = self.enc_geo[l](H['geo'][l + 1])[None].expand(h.shape[0], -1, -1)
        for blk in self.bottom:
            h, e = self._mp(blk, h, e, H['snd'][self.levels], H['rcv'][self.levels])
        for l in reversed(range(self.levels)):
            h = skips[l] + _segment_mean(h, H['unpool_o'][l], H['unpool_i'][l], H['n'][l])
            h, _ = self._mp(self.up[l], h, es[l], H['snd'][l], H['rcv'][l])
        return self.dec(h)


def hierarchy_tensors(levels, device, scale):
    """torch tensors of a bistride_hierarchy for BSMSGNN.forward; geo = (dx, dy, |d|) * scale of every directed coarse edge
    (minimum image on the unit torus)"""
    import numpy as np
    H = dict(snd=[], rcv=[], geo=[], pool_o=[], pool_i=[], unpool_o=[], unpool_i=[], n=[])
    L = lambda a: torch.as_tensor(np.asarray(a), dtype=torch.long, device=device)
    for i, lev in enumerate(levels):
        E = lev['edges']
        H['snd'].append(L(np.concatenate([E[:, 0], E[:, 1]]))); H['rcv'].append(L(np.concatenate([E[:, 1], E[:, 0]])))
        d = lev['pts'][E[:, 1]] - lev['pts'][E[:, 0]]
        d = d - np.round(d)
        d = np.concatenate([d, -d]) * scale
        H['geo'].append(torch.as_tensor(np.concatenate([d, np.linalg.norm(d, axis=1, keepdims=True)], 1), dtype=torch.float32, device=device))
        H['n'].append(lev['n'])
        if 'pool' in lev:
            H['pool_o'].append(L(lev['pool'][:, 0])); H['pool_i'].append(L(lev['pool'][:, 1]))
            H['unpool_o'].append(L(lev['unpool'][:, 0])); H['unpool_i'].append(L(lev['unpool'][:, 1]))
    return H
