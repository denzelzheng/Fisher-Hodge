"""Models: B1 gnn_ode, B2 euclid_gf, M fisher_hodge.  All operate on (B,N,N) node fields on the periodic grid."""
import math
import torch
import torch.nn as nn
import torch.nn.functional as Fn
from common import *

NSUB = 8          # nominal substeps per sampling interval DT for every model
MAXSUB = 4000     # safety cap for adaptive FH stepping (per DT)


def rho_scale(sysname):
    return 1.0 if sysname == 'S3' else 10.0


class MLP(nn.Module):
    def __init__(s, i, h, o, ln=False, zero_last=False):
        super().__init__()
        s.net = nn.Sequential(nn.Linear(i, h), nn.SiLU(), nn.Linear(h, h), nn.SiLU(), nn.Linear(h, o))
        s.ln = nn.LayerNorm(o) if ln else None
        if zero_last:
            nn.init.zeros_(s.net[-1].weight); nn.init.zeros_(s.net[-1].bias)

    def forward(s, x):
        y = s.net(x)
        return s.ln(y) if s.ln is not None else y


def static_edge_feats(F):
    U, P, pn = F['U'], F['dpsi'], F['psin']
    dU = grad_e(U)
    ish = torch.zeros_like(dU); ish[:, 0] = 1
    sym = torch.stack([ish, 1 - ish, mean_e(U), dU.abs(), mean_e(pn), P.abs()], -1)   # orientation-invariant
    ori = torch.stack([ish, 1 - ish, dU, P, mean_e(U), mean_e(pn)], -1)                # oriented
    return sym, ori


def node_deriv(fn, rho):
    """d/d rho_i of sum_i fn(rho)_i  (fn node-local) via autograd; differentiable when grad mode is on."""
    if torch.is_grad_enabled():
        x = rho if rho.requires_grad else rho.detach().requires_grad_(True)
        return torch.autograd.grad(fn(x).sum(), x, create_graph=True)[0]
    with torch.enable_grad():
        x = rho.detach().requires_grad_(True)
        return torch.autograd.grad(fn(x).sum(), x)[0].detach()


def logmean_e(rho):
    """Maas logarithmic mean on edges, theta(a,b)=(a-b)/(log a - log b), = a if a==b."""
    a, b = tail_e(rho), head_e(rho)
    dl = torch.log(a) - torch.log(b)
    small = dl.abs() < 1e-4
    dls = torch.where(small, torch.ones_like(dl), dl)
    return torch.where(small, 0.5 * (a + b), (a - b) / dls)


def heun_fixed(model, rho, ctx, Dt, n=NSUB):
    dt = Dt / n
    for _ in range(n):
        k1 = model.rate(rho, ctx)
        k2 = model.rate(rho + dt * k1, ctx)
        rho = rho + 0.5 * dt * (k1 + k2)
    return rho, n


# ---------------------------------------------------------------------------------------------
class GNNODE(nn.Module):
    """B1: MeshGraphNet-lite neural ODE (encode-process-decode, 3 message-passing blocks)."""
    name = 'gnn_ode'

    def __init__(s, sysname, H=64, L=3):
        super().__init__()
        s.sys, s.RS = sysname, rho_scale(sysname)
        s.enc_n = MLP(3, H, H, ln=True)
        s.enc_e = MLP(4, H, H, ln=True)
        s.me = nn.ModuleList([MLP(3 * H, H, H, ln=True) for _ in range(L)])
        s.mn = nn.ModuleList([MLP(2 * H, H, H, ln=True) for _ in range(L)])
        s.dec = MLP(H, H, 1, zero_last=True)
        s.register_buffer('dirs', torch.tensor([[1., 0.], [-1., 0.], [0., 1.], [0., -1.]]))

    def prepare(s, F):
        B = F['U'].shape[0]
        dU = edge_dirs(grad_e(F['U'])); dP = edge_dirs(F['dpsi'])
        dx = s.dirs[:, 0].view(1, 4, 1, 1).expand_as(dU); dy = s.dirs[:, 1].view(1, 4, 1, 1).expand_as(dU)
        ef = torch.stack([dx, dy, dU, dP], -1)
        return dict(F=F, e0=s.enc_e(ef))

    def rate(s, rho, ctx):
        F = ctx['F']
        h = s.enc_n(torch.stack([rho / s.RS, F['U'], F['psin']], -1))
        e = ctx['e0']
        for me, mn in zip(s.me, s.mn):
            hn = nbr_dirs_cl(h)
            e = e + me(torch.cat([e, h.unsqueeze(1).expand_as(hn), hn], -1))
            h = h + mn(torch.cat([h, e.sum(1)], -1))
        return S_SCALE * s.dec(h).squeeze(-1)

    def advance(s, rho, ctx, Dt):
        return heun_fixed(s, rho, ctx, Dt)


# ---------------------------------------------------------------------------------------------
class EuclidGF(nn.Module):
    """B2: Euclidean (flat-metric) gradient-flow GNN:
         d rho/dt = -s [ d^T W d mu + lam mu ]  (+ S2: -s d^T c(edge feats, rho_t, rho_h))  (+ S3: -R(rho,U))
       mu = dE/drho, E = sum_i 1/2 rho_i^2 + e_theta(rho_i, U_i, psi_i);  W>0 static edge weights (no mobility),
       lam>=0 L2 (non-conservative) part.  No positivity guarantee."""
    name = 'euclid_gf'

    def __init__(s, sysname, H=128):
        super().__init__()
        s.sys, s.RS = sysname, rho_scale(sysname)
        s.W = MLP(6, H, 1)
        s.e = MLP(3, H, 1, zero_last=True)
        s.lam_raw = nn.Parameter(torch.tensor(-5.0))
        if sysname == 'S2':
            s.c = MLP(8, H, 1, zero_last=True)
        if sysname == 'S3':
            s.R = MLP(2, H, 1, zero_last=True)

    def prepare(s, F):
        sym, ori = static_edge_feats(F)
        return dict(F=F, W=Fn.softplus(s.W(sym).squeeze(-1)), ori=ori)

    def energy_density(s, x, F):
        return 0.5 * x * x + s.RS ** 2 * s.e(torch.stack([x / s.RS, F['U'], F['psin']], -1)).squeeze(-1)

    def mu(s, rho, F):
        return node_deriv(lambda x: s.energy_density(x, F), rho)

    def flux_q(s, rho, ctx, mu=None):
        mu = s.mu(rho, ctx['F']) if mu is None else mu
        q = ctx['W'] * grad_e(mu)
        if s.sys == 'S2':
            fe = torch.cat([ctx['ori'], (tail_e(rho) / s.RS).unsqueeze(-1), (head_e(rho) / s.RS).unsqueeze(-1)], -1)
            q = q + s.c(fe).squeeze(-1)
        return q

    def rate(s, rho, ctx):
        F = ctx['F']
        mu = s.mu(rho, F)
        out = -S_SCALE * (div_T(s.flux_q(rho, ctx, mu)) + Fn.softplus(s.lam_raw) * mu)
        if s.sys == 'S3':
            out = out - s.R(torch.stack([rho, F['U']], -1)).squeeze(-1)
        return out

    def lyapunov(s, rho, F):
        return s.energy_density(rho, F).mean((-1, -2))

    def advance(s, rho, ctx, Dt):
        return heun_fixed(s, rho, ctx, Dt)


# ---------------------------------------------------------------------------------------------
class FisherHodge(nn.Module):
    """M: discrete Wasserstein (Otto/Maas) gradient flow + cycle (S2) + Fisher-Rao reaction (S3):
         d rho/dt = -s d^T [ kappa Theta(rho) d mu + Theta(rho) c ]  - rho * r(mu, rho, U)
       Theta = logarithmic mean, kappa>0 learned static edge coefficient, c learned edge 1-cochain,
       F = sum_i rho log rho + alpha U rho + g_theta(rho,U),  mu = dF/drho (autograd).
       Integrator: SSP-RK2 (Heun) with adaptive positivity-preserving substeps (no clamping)."""
    name = 'fisher_hodge'

    def __init__(s, sysname, H=128):
        super().__init__()
        s.sys, s.RS = sysname, rho_scale(sysname)
        s.kap = MLP(6, H, 1)
        s.g = MLP(2, H, 1, zero_last=True)
        s.alpha = nn.Parameter(torch.tensor(0.0))
        if sysname == 'S2':
            s.c = MLP(6, H, 1, zero_last=True)
        if sysname == 'S3':
            s.r = MLP(3, H, 1, zero_last=True)
        s.sub_log = []

    def prepare(s, F):
        sym, ori = static_edge_feats(F)
        ctx = dict(F=F, kap=Fn.softplus(s.kap(sym).squeeze(-1)))
        if s.sys == 'S2':
            ctx['c'] = s.c(ori).squeeze(-1)
        return ctx

    def g_density(s, x, F):
        return s.RS * s.g(torch.stack([x / s.RS, F['U']], -1)).squeeze(-1)

    def mu(s, rho, F):
        return torch.log(rho) + s.alpha * F['U'] + node_deriv(lambda x: s.g_density(x, F), rho)

    def flux_q(s, rho, ctx, mu=None):
        mu = s.mu(rho, ctx['F']) if mu is None else mu
        th = logmean_e(rho)
        q = ctx['kap'] * th * grad_e(mu)
        if s.sys == 'S2':
            q = q + th * ctx['c']
        return q

    def rate(s, rho, ctx):
        F = ctx['F']
        mu = s.mu(rho, F)
        out = -S_SCALE * div_T(s.flux_q(rho, ctx, mu))
        if s.sys == 'S3':
            out = out - rho * s.r(torch.stack([mu, rho, F['U']], -1)).squeeze(-1)
        return out

    def lyapunov(s, rho, F):
        return (rho * torch.log(rho) + s.alpha * F['U'] * rho + s.g_density(rho, F)).mean((-1, -2))

    def advance(s, rho, ctx, Dt):
        t, nsub, dt_nom = 0.0, 0, Dt / NSUB
        while Dt - t > 1e-9:
            f0 = s.rate(rho, ctx)
            with torch.no_grad():
                ratio = torch.where(f0 < 0, rho / (-f0).clamp_min(1e-30), torch.full_like(rho, float('inf')))
                dt_safe = 0.5 * torch.nan_to_num(ratio, nan=float('inf')).min().item()
            dt = min(dt_nom, Dt - t, dt_safe)
            while True:
                r1 = rho + dt * f0
                f1 = s.rate(r1, ctx)
                r2 = r1 + dt * f1
                with torch.no_grad():
                    ok = bool((torch.nan_to_num(r2, nan=1.0) > 0).all())
                if ok or dt < 1e-12:
                    break
                dt *= 0.5
            rho = 0.5 * (rho + r2)        # SSP-RK2 = convex combination of positive Euler stages
            t += dt; nsub += 1
            if nsub >= MAXSUB:
                break
        return rho, nsub


MODELS = {'gnn_ode': GNNODE, 'euclid_gf': EuclidGF, 'fisher_hodge': FisherHodge}
