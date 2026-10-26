"""Self-check of the re-implemented non-reciprocal GNN on its home task (paper Sec. 5.2, parameter set (i)):
(1) learned pair functions F2_r, F2_phi against the true ones, maps over the relative position in the frame of the
receiver at phi_j - phi_i = 0, per species pair; (2) the self terms; (3) a long re-simulation (t = 300) from a random
initial state with the learned model plus the true angular noise, against the true model: cluster statistics.
Writes runs/external/common/taskC_selfcheck.json and taskC_selfcheck.png."""
import os, sys, json, math
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
import numpy as np
import torch
from fisher_hodge.ext import nonreciprocal_gnn as ng

WORK = os.path.expanduser('~/ext_work/taskC')
OUT = os.environ.get('SELFCHECK_OUT') or os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))), 'runs', 'external', 'common')


@torch.no_grad()
def cluster_stats(X, P, C, L=20.0):
    """X (B, N, 2): fraction of particles with >= 2 neighbours within 1, fraction of their neighbours of the other
    species, mean polar order |<p>| inside clusters of radius 2, and the density coefficient of variation (16 x 16)"""
    d = ng.wrap(X[:, None] - X[:, :, None], L)
    r = d.norm(dim=-1)
    nb = (r < 1.0) & (r > 0)
    deg = nb.sum(-1)
    other = (nb & (C[:, None, :] != C[:, :, None])).sum(-1)
    p = torch.stack([torch.cos(P), torch.sin(P)], -1)
    loc = ((r < 2.0).float() @ p)                                  # sum of polarities within 2
    cnt = (r < 2.0).float().sum(-1, keepdim=True)
    pol = (loc / cnt).norm(dim=-1)
    rho = ng.bin_density(X, C)
    cv = rho.std((-1, -2)) / rho.mean((-1, -2))
    clustered = deg >= 2
    return dict(frac_clustered=float(clustered.float().mean()),
                frac_other_species_neighbours=float(other[clustered].sum() / deg[clustered].sum()),
                local_polar_order=float(pol.mean()), density_cv=float(cv.mean()))


@torch.no_grad()
def main():
    dev = 'cuda'
    truth = ng.MixedSpecies().to(dev)
    gnn = ng.NonreciprocalGNN().to(dev)
    ck = torch.load(os.path.join(WORK, os.environ.get('GNN_CKPT', 'gnn_best.pt'))); gnn.load_state_dict(ck['state']); gnn.eval()
    d = torch.load(os.path.join(WORK, 'train.pt'))
    res = dict(epoch=ck.get('epoch'), valid=ck.get('valid'))
    res['interactions'] = ng.compare_interactions(gnn, truth, d['x'], d['phi'], d['c'])
    # maps
    n = 81
    g = torch.linspace(-1, 1, n, device=dev)
    gx, gy = torch.meshgrid(g, g, indexing='ij')
    rij = torch.stack([gx.reshape(-1), gy.reshape(-1)], -1)
    inside = (rij.norm(dim=-1) < 1) & (rij.norm(dim=-1) > 0.15)
    rij = rij[inside]
    P0 = torch.zeros(len(rij), device=dev)
    maps = {}
    for ci in (0, 1):
        for cj in (0, 1):
            ct = torch.full((len(rij),), ci, device=dev); cn = torch.full((len(rij),), cj, device=dev)
            vt, wt = truth.messages(rij, P0, P0, ct, cn)
            vl, wl = gnn.pair_messages(rij, P0, P0, ct, cn)
            maps[(ci, cj)] = dict(wt=wt, wl=wl, vt=vt[:, 0], vl=vl[:, 0])
    # re-simulation from random initial states, 300 time units
    gen = torch.Generator(device=dev).manual_seed(7)
    B, N = 3, 400
    c = torch.cat([torch.zeros(200), torch.ones(200)]).long().to(dev)[None].expand(B, N).contiguous()
    x = torch.rand(B, N, 2, device=dev, generator=gen) * 20
    p = torch.rand(B, N, device=dev, generator=gen) * 2 * math.pi
    Xt, Pt = ng.simulate(truth, x, p, c, 300, generator=gen)
    Xg, Pg = ng.rollout(gnn, x, p, c, 300, sigma=truth.p['sigma'], generator=gen)
    res['resim_t300'] = dict(truth=cluster_stats(Xt[-1], Pt[-1], c), gnn=cluster_stats(Xg[-1], Pg[-1], c))
    res['resim_t300_initial'] = cluster_stats(x, p, c)
    json.dump(res, open(os.path.join(OUT, 'taskC_selfcheck.json'), 'w'), indent=1)
    print(json.dumps(res, indent=1))

    import matplotlib; matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    fig, axs = plt.subplots(3, 5, figsize=(17, 10))
    pts = rij.cpu().numpy()
    for k, (pair, mp) in enumerate(maps.items()):
        vmax = float(mp['wt'].abs().quantile(0.98))
        for row, (key, lab) in enumerate((('wt', 'true torque'), ('wl', 'learned torque'))):
            ax = axs[row, k]
            sc = ax.scatter(pts[:, 0], pts[:, 1], c=mp[key].cpu().numpy(), cmap='RdBu_r', vmin=-vmax, vmax=vmax, s=6)
            ax.set_aspect('equal'); ax.set_title(f'{lab}, receiver {pair[0]} <- neighbour {pair[1]}', fontsize=8)
            ax.arrow(0, 0, 0.3, 0, head_width=0.05, color='k')
        ax = axs[2, k]
        rr = rij.norm(dim=-1).cpu().numpy()
        ax.plot(rr, mp['vt'].cpu().numpy(), '.', ms=2, label='true v_par'); ax.plot(rr, mp['vl'].cpu().numpy(), '.', ms=2, label='learned v_par')
        ax.set_xlabel('|r_ij|'); ax.set_title(f'velocity message (along own polarity), {pair}', fontsize=8); ax.legend(fontsize=6)
    for row, (XX, CC, lab) in enumerate(((Xt[-1, 0], c[0], 'true model, t = 300'), (Xg[-1, 0], c[0], 'learned GNN + noise, t = 300'))):
        ax = axs[row, 4]
        xx = XX.cpu().numpy(); cc = CC.cpu().numpy()
        ax.scatter(xx[cc == 0, 0], xx[cc == 0, 1], s=3, c='C0', label='species 0'); ax.scatter(xx[cc == 1, 0], xx[cc == 1, 1], s=3, c='C3', label='species 1')
        ax.set_xlim(0, 20); ax.set_ylim(0, 20); ax.set_aspect('equal'); ax.set_title(lab, fontsize=8)
    axs[0, 4].legend(fontsize=6)
    axs[2, 4].axis('off')
    fig.suptitle(f'Re-implemented non-reciprocal GNN, best epoch {ck.get("epoch")}: learned vs true pair functions (phi_j = phi_i, receiver frame, polarity along +x)')
    fig.tight_layout()
    fig.savefig(os.path.join(OUT, 'taskC_selfcheck.png'), dpi=110)


if __name__ == '__main__':
    main()
