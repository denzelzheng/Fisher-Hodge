"""Baselines for the CRN benchmark.
B1 crnn : CRNN-style neural ODE (Ji & Deng 2021, minimal): known stoichiometry, mass action J = kf c^{nu+} - kb c^{nu-},
          kf, kb = softplus(MLP(oriented hyperedge feats)) independent (no detailed-balance / cycle structure),
          fixed-step Heun (no positivity control).
B2 gnn  : GNN-ODE (MeshGraphNet-lite from fisher_hodge/models.py GNNODE, 3 channels): unstructured message passing.
B3 hyperme_rev : HyperME with A = 0 (reversible / detailed balance) -> hyperme.HyperME(use_A=False)."""
import torch
import torch.nn as nn
import torch.nn.functional as Fn
from hyperme import MLP, NS, graph, edge_feats, heun_fixed


class CRNN(nn.Module):
    name = 'crnn'

    def __init__(s, H=64):
        super().__init__()
        s.kmlp = MLP(8, H, 2, bias_last=0.5)

    def prepare(s, U, dG, n):
        g = graph(n, U.device)
        sym, dU = edge_feats(U, dG, g)
        kk = Fn.softplus(s.kmlp(torch.cat([sym, dU[..., None]], -1))) + 1e-6
        return dict(g=g, kf=kk[..., 0], kb=kk[..., 1], U=U, dG=dG, n=n,
                    rp=torch.tensor(g['rp'], device=U.device), rm=torch.tensor(g['rm'], device=U.device))

    def ab(s, c, ctx):
        x = torch.cat([c.flatten(1), torch.ones_like(c[:, 0, :1])], 1)     # pad slot = 1
        rp, rm = ctx['rp'], ctx['rm']
        a = ctx['kf'] * x[:, rp[:, 0]] * x[:, rp[:, 1]]
        b = ctx['kb'] * x[:, rm[:, 0]] * x[:, rm[:, 1]]
        return a, b

    def flux(s, c, ctx):
        a, b = s.ab(c, ctx)
        return a - b

    def rhs(s, c, ctx):
        return (s.flux(c, ctx) @ ctx['g']['Smat']).view(c.shape[0], NS, -1)

    def epr(s, c, ctx):
        a, b = s.ab(c, ctx)
        return ((a - b) * (torch.log(a) - torch.log(b))).sum(-1)       # NaN if c<0 (no positivity guarantee)

    def A_cycle(s, ctx):
        return torch.log(ctx['kf'] / ctx['kb']) @ ctx['g']['P']

    def cycle_affinity_local(s, ctx):
        return torch.log(ctx['kf'] / ctx['kb']) @ ctx['g']['zloc'].T

    def advance(s, c, ctx, Dt):
        return heun_fixed(s, c, ctx, Dt)


def nbr_dirs_cl(h):
    """(B,n,n,C) -> (B,4,n,n,C) neighbours (+x,-x,+y,-y)"""
    return torch.stack([h.roll(-1, 2), h.roll(1, 2), h.roll(-1, 1), h.roll(1, 1)], 1)


class MLPLN(nn.Module):
    def __init__(s, i, h, o, ln=True, zero_last=False):
        super().__init__()
        s.net = nn.Sequential(nn.Linear(i, h), nn.SiLU(), nn.Linear(h, h), nn.SiLU(), nn.Linear(h, o))
        s.ln = nn.LayerNorm(o) if ln else None
        if zero_last:
            nn.init.zeros_(s.net[-1].weight); nn.init.zeros_(s.net[-1].bias)

    def forward(s, x):
        y = s.net(x)
        return s.ln(y) if s.ln is not None else y


class GNNODE3(nn.Module):
    """B2: encode-process-decode, 3 message-passing blocks, node inputs (c_X, c_Xp, c_E, U, dG), edge inputs (dx, dy, dU)."""
    name = 'gnn'

    def __init__(s, H=64, L=3):
        super().__init__()
        s.enc_n = MLPLN(NS + 2, H, H)
        s.enc_e = MLPLN(3, H, H)
        s.me = nn.ModuleList([MLPLN(3 * H, H, H) for _ in range(L)])
        s.mn = nn.ModuleList([MLPLN(2 * H, H, H) for _ in range(L)])
        s.dec = MLPLN(H, H, NS, ln=False, zero_last=True)
        s.register_buffer('dirs', torch.tensor([[1., 0.], [-1., 0.], [0., 1.], [0., -1.]]))

    def prepare(s, U, dG, n):
        B = U.shape[0]
        Ug = U.view(B, n, n)
        dU = torch.stack([Ug.roll(-1, 2) - Ug, Ug.roll(1, 2) - Ug, Ug.roll(-1, 1) - Ug, Ug.roll(1, 1) - Ug], 1)
        dx = s.dirs[:, 0].view(1, 4, 1, 1).expand_as(dU); dy = s.dirs[:, 1].view(1, 4, 1, 1).expand_as(dU)
        return dict(U=Ug, dG=dG, n=n, e0=s.enc_e(torch.stack([dx, dy, dU], -1)))

    def rhs(s, c, ctx):
        B, n = c.shape[0], ctx['n']
        x = torch.cat([c.view(B, NS, n, n), ctx['U'][:, None], (ctx['dG'] - 2.0).view(B, 1, 1, 1).expand(B, 1, n, n)], 1)
        h = s.enc_n(x.permute(0, 2, 3, 1))
        e = ctx['e0']
        for me, mn in zip(s.me, s.mn):
            hn = nbr_dirs_cl(h)
            e = e + me(torch.cat([e, h.unsqueeze(1).expand_as(hn), hn], -1))
            h = h + mn(torch.cat([h, e.sum(1)], -1))
        return s.dec(h).permute(0, 3, 1, 2).reshape(B, NS, n * n)

    def advance(s, c, ctx, Dt):
        return heun_fixed(s, c, ctx, Dt)
