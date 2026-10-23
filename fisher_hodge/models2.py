"""Stress-test additions (2026-10-19) on top of models.py.
  B3 flux_gnn        structure-preserving GNN without geometry: directed softplus rates -> edge flux -> d^T assembly
                     (exact mass conservation, exact positivity), no mobility / Otto metric / free energy.
  B4 wgf_rev         reversible Wasserstein gradient flow: Fisher-Hodge with kappa = 1 and no cycle term (the model
                     class of Rancati-Maas-Locatello), trained with the same rollout loss as FH.
  B5 jko_amort       Rancati-Maas-Locatello (arXiv 2604.11311) free-energy learning: F = <V,rho> + beta H(rho), log-mean
                     mobility, trained with the JKO first-order-optimality quadratic loss (no rollout training);
                     V amortised as V_theta(U) so that it can be tested on new potentials.
  B5' jko_inst       same, but V a free node field + beta fitted per *test* trajectory on its first T_train snapshots
                     (transductive, exactly the per-instance protocol of the paper); closed-form weighted LS.
  M2 fisher_hodge_fr Fisher-Hodge with Fisher-Rao reaction  -rho sigma_theta(rho,U) (mu_theta(rho) - mu*_theta(U)).
  M2u fisher_hodge_fr_u  same with sigma depending on U only (ablation).
"""
import math
import torch
import torch.nn as nn
import torch.nn.functional as Fn
import models
from models import MLP, static_edge_feats, node_deriv, logmean_e, GNNODE, EuclidGF, FisherHodge, NSUB, MAXSUB
from common import *


def rho_scale(sysname):
    return 1.0 if sysname in ('S3', 'S5') else 10.0


models.rho_scale = rho_scale          # S5 (exclusion) lives in [0,1]; patch for the original classes too


# ---------------------------------------------------------------------------------------------
class FluxGNN(nn.Module):
    """B3: encode-process-decode GNN (same trunk as B1) whose decoder emits non-negative *directed transition rates*
    a_{i->j} = softplus(.) for the 4 directed neighbours of every node.  Edge anti-flux (same sign convention as FH.flux_q,
    i.e. physical flux J = -q):  q_e = rho_head a_{head->tail} - rho_tail a_{tail->head};  d rho/dt = -s d^T q  (+ S3 reaction).
    => master-equation form: mass conserved exactly, positivity exact in continuous time and preserved by the same
    positivity-preserving SSP-RK2 substeps as FH.  No mobility / Otto metric / free energy / Lyapunov function."""
    name = 'flux_gnn'

    def __init__(s, sysname, H=64, L=3):
        super().__init__()
        s.sys, s.RS = sysname, rho_scale(sysname)
        s.enc_n = MLP(3, H, H, ln=True)
        s.enc_e = MLP(4, H, H, ln=True)
        s.me = nn.ModuleList([MLP(3 * H, H, H, ln=True) for _ in range(L)])
        s.mn = nn.ModuleList([MLP(2 * H, H, H, ln=True) for _ in range(L)])
        s.dec = MLP(H, H, 1)
        nn.init.zeros_(s.dec.net[-1].weight); nn.init.constant_(s.dec.net[-1].bias, math.log(math.e - 1))  # softplus -> 1: starts as exact unit diffusion
        if sysname == 'S3':
            s.r = MLP(2, H, 1, zero_last=True)
        s.register_buffer('dirs', torch.tensor([[1., 0.], [-1., 0.], [0., 1.], [0., -1.]]))
        s.sub_log = []

    prepare = GNNODE.prepare

    def rates(s, rho, ctx):
        F = ctx['F']
        h = s.enc_n(torch.stack([rho / s.RS, F['U'], F['psin']], -1))
        e = ctx['e0']
        for me, mn in zip(s.me, s.mn):
            hn = nbr_dirs_cl(h)
            e = e + me(torch.cat([e, h.unsqueeze(1).expand_as(hn), hn], -1))
            h = h + mn(torch.cat([h, e.sum(1)], -1))
        return Fn.softplus(s.dec(e).squeeze(-1))          # (B,4,n,n): rate from node towards (+x,-x,+y,-y)

    def flux_q(s, rho, ctx):
        a = s.rates(rho, ctx)
        qh = rho.roll(-1, -1) * a[:, 1].roll(-1, -1) - rho * a[:, 0]      # horizontal edge (a,b)->(a,b+1)
        qv = rho.roll(-1, -2) * a[:, 3].roll(-1, -2) - rho * a[:, 2]      # vertical   edge (a,b)->(a+1,b)
        return torch.stack([qh, qv], 1)

    def rate(s, rho, ctx):
        out = -S_SCALE * div_T(s.flux_q(rho, ctx))
        if s.sys == 'S3':
            out = out - rho * s.r(torch.stack([rho, ctx['F']['U']], -1)).squeeze(-1)
        return out

    advance = FisherHodge.advance


# ---------------------------------------------------------------------------------------------
class WGFRev(FisherHodge):
    """B4: reversible discrete Wasserstein gradient flow with learned free energy (kappa = 1, no cycle term).
    g gets (rho, U, psi_node) so that the S2 failure is structural (no stationary current), not input starvation."""
    name = 'wgf_rev'

    def __init__(s, sysname, H=128):
        nn.Module.__init__(s)
        s.sys, s.RS = sysname, rho_scale(sysname)
        s.g = MLP(3, H, 1, zero_last=True)
        s.alpha = nn.Parameter(torch.tensor(0.0))
        s.sub_log = []

    def prepare(s, F):
        return dict(F=F, kap=torch.ones((), device=F['U'].device))

    def g_density(s, x, F):
        return s.RS * s.g(torch.stack([x / s.RS, F['U'], F['psin']], -1)).squeeze(-1)

    def flux_q(s, rho, ctx, mu=None):
        mu = s.mu(rho, ctx['F']) if mu is None else mu
        return logmean_e(rho) * grad_e(mu)


# ---------------------------------------------------------------------------------------------
def solve_jko_psi(rho_prev, rho_next, tau):
    """Velocity potential of the JKO/implicit step rho_prev -> rho_next in the discrete Otto geometry:
         -s d^T Theta(rho_next) d psi = (rho_next - rho_prev)/tau   (psi has zero mean).  Dense solve, float64."""
    B, n, _ = rho_next.shape
    Nn = n * n
    dev = rho_next.device
    with torch.no_grad():
        I = torch.eye(Nn, dtype=torch.float64, device=dev).reshape(Nn, n, n)
        Dm = grad_e(I).reshape(Nn, 2 * Nn).T                      # (E, N) incidence matrix d
        th = logmean_e(rho_next.double()).reshape(B, 2 * Nn)
        L = S_SCALE * torch.einsum('ei,be,ej->bij', Dm, th, Dm)   # (B,N,N) = s d^T Theta d
        L = L + torch.ones(Nn, Nn, dtype=torch.float64, device=dev) / Nn
        rhs = ((rho_next - rho_prev).double() / tau).reshape(B, Nn)
        psi = torch.linalg.solve(L, -rhs.unsqueeze(-1)).squeeze(-1)      # (rho_next-rho_prev)/tau = -s d^T Theta d psi
        psi = psi - psi.mean(-1, keepdim=True)
        return psi.reshape(B, n, n).float()


class JKOAmort(nn.Module):
    """B5 jko_amort (Rancati-Maas-Locatello style, amortised).  F(rho) = sum_i V_theta(U_i, psi_i) rho_i + beta sum_i rho_i log rho_i,
    mu = beta log rho + V,  log-mean mobility, kappa = 1, no cycle.  Trained by the JKO first-order condition
    loss  sum_e Theta_e(rho_{k+1}) (d mu_theta(rho_{k+1}) - d psi_k)^2  (quadratic, one-step, no rollout)."""
    name = 'jko_amort'

    def __init__(s, sysname, H=64):
        super().__init__()
        s.sys, s.RS = sysname, rho_scale(sysname)
        s.V = MLP(2, H, 1, zero_last=True)
        s.beta_raw = nn.Parameter(torch.tensor(math.log(math.e - 1)))
        s.sub_log = []

    def beta(s):
        return Fn.softplus(s.beta_raw)

    def Vfield(s, F):
        return s.V(torch.stack([F['U'], F['psin']], -1)).squeeze(-1)

    def prepare(s, F):
        return dict(F=F, V=s.Vfield(F))

    def mu(s, rho, ctx):
        return s.beta() * torch.log(rho) + ctx['V']

    def flux_q(s, rho, ctx, mu=None):
        mu = s.mu(rho, ctx) if mu is None else mu
        return logmean_e(rho) * grad_e(mu)

    def rate(s, rho, ctx):
        return -S_SCALE * div_T(s.flux_q(rho, ctx))

    def lyapunov(s, rho, F):
        return (s.beta() * rho * torch.log(rho) + s.Vfield(F) * rho).mean((-1, -2))

    def jko_loss(s, rho_next, psi, F):
        ctx = s.prepare(F)
        th = logmean_e(rho_next)
        r = grad_e(s.mu(rho_next, ctx)) - grad_e(psi)
        return (th * r * r).sum((-1, -2, -3)).mean() / (th * grad_e(psi) ** 2).sum((-1, -2, -3)).mean().clamp_min(1e-12)

    advance = FisherHodge.advance


class JKOInst(JKOAmort):
    """B5' jko_inst: per-trajectory free node potential V (N free numbers) + beta, fitted in closed form (weighted LS) on
    the first T_train snapshots of each *test* trajectory (transductive).  Rollout identical to jko_amort."""
    name = 'jko_inst'

    def __init__(s, sysname, H=64):
        nn.Module.__init__(s)
        s.sys, s.RS = sysname, rho_scale(sysname)
        s.register_buffer('Vb', torch.zeros(1, N, N)); s.register_buffer('betab', torch.ones(1))
        s.sub_log = []

    def beta(s):
        return s.betab.view(-1, 1, 1)

    def Vfield(s, F):
        return s.Vb

    @torch.no_grad()
    def fit(s, traj, tau, ridge=1e-6):
        """traj (B, K+1, n, n): minimise sum_k sum_e Theta_e (dV + beta d log rho_k - d psi_k)^2 over (V, beta)."""
        B, K1, n, _ = traj.shape
        Nn = n * n; dev = traj.device
        I = torch.eye(Nn, dtype=torch.float64, device=dev).reshape(Nn, n, n)
        Dm = grad_e(I).reshape(Nn, 2 * Nn).T                       # (E,N)
        M = torch.zeros(B, Nn + 1, Nn + 1, dtype=torch.float64, device=dev)
        b = torch.zeros(B, Nn + 1, dtype=torch.float64, device=dev)
        for k in range(K1 - 1):
            rn = traj[:, k + 1]
            psi = solve_jko_psi(traj[:, k], rn, tau).double()
            th = logmean_e(rn.double()).reshape(B, 2 * Nn)
            dl = grad_e(torch.log(rn.double())).reshape(B, 2 * Nn)
            dpsi = grad_e(psi).reshape(B, 2 * Nn)
            A = torch.cat([Dm.unsqueeze(0).expand(B, -1, -1), dl.unsqueeze(-1)], -1)   # (B,E,N+1)
            M += torch.einsum('bei,be,bej->bij', A, th, A)
            b += torch.einsum('bei,be,be->bi', A, th, dpsi)
        M += ridge * torch.eye(Nn + 1, dtype=torch.float64, device=dev)
        M[:, :Nn, :Nn] += torch.ones(Nn, Nn, dtype=torch.float64, device=dev) / Nn    # gauge: mean V = 0
        x = torch.linalg.solve(M, b.unsqueeze(-1)).squeeze(-1)
        s.Vb = x[:, :Nn].reshape(B, n, n).float()
        s.betab = x[:, Nn].float().clamp_min(1e-3)
        return s.betab


# ---------------------------------------------------------------------------------------------
class FisherHodgeFR(FisherHodge):
    """M2: Fisher-Hodge with a Fisher-Rao reaction term (WFR gradient structure):
         d rho/dt|_R = - rho * sigma_theta(rho,U) * ( mu_theta(rho) - mu*_theta(U) ),
       sigma > 0 (reaction mobility, softplus), mu* = reservoir chemical potential (node MLP of U).
       Equilibrium rho*: mu_theta(rho*) = mu*(U) -> fixed by the free energy; sign of the reaction is forced
       (growth below rho*, decay above), unlike the unconstrained -rho r_theta(mu,rho,U) of the first round.
       Fisher-KPP is in this class exactly: mu = log rho, mu* = 0, sigma = r(x) (rho-1)/log rho = r(x) * logmean(rho,1)."""
    name = 'fisher_hodge_fr'
    SIG_IN = 'full'

    def __init__(s, sysname, H=128):
        super().__init__(sysname, H)
        if hasattr(s, 'r'):
            del s.r
        s.sig = MLP(3 if s.SIG_IN == 'full' else 1, H, 1)
        s.mustar = MLP(1, H, 1, zero_last=True)

    def sigma(s, rho, U):
        x = torch.stack([rho, torch.log(rho), U], -1) if s.SIG_IN == 'full' else U.unsqueeze(-1)
        return Fn.softplus(s.sig(x).squeeze(-1))

    def rate(s, rho, ctx):
        F = ctx['F']
        mu = s.mu(rho, F)
        out = -S_SCALE * div_T(s.flux_q(rho, ctx, mu))
        mus = s.mustar(F['U'].unsqueeze(-1)).squeeze(-1)
        return out - rho * s.sigma(rho, F['U']) * (mu - mus)

    def lyapunov(s, rho, F):
        mus = s.mustar(F['U'].unsqueeze(-1)).squeeze(-1)
        return (rho * torch.log(rho) + s.alpha * F['U'] * rho + s.g_density(rho, F) - mus * rho).mean((-1, -2))


class FisherHodgeFRU(FisherHodgeFR):
    name = 'fisher_hodge_fr_u'
    SIG_IN = 'U'


def logmean_s(a, b):
    dl = torch.log(a) - torch.log(b)
    small = dl.abs() < 1e-4
    return torch.where(small, 0.5 * (a + b), (a - b) / torch.where(small, torch.ones_like(dl), dl))


class FisherHodgeWFR(FisherHodge):
    """M2' (second attempt, 2026-10-19): the reaction is the Otto/Fisher-Rao flux along a *virtual reservoir edge*
    attached to every node, with the same logarithmic-mean mobility as the graph edges:
         d rho/dt|_R = - rho * kappa_R(U) * Lambda(rho, rho*(U)) * ( mu_theta(rho;U) - mu_theta(rho*(U);U) ),
       Lambda = logarithmic mean, rho*(U) = learned reservoir density (= carrying capacity), kappa_R(U) > 0.
       With mu = log rho this is *exactly* logistic growth  kappa rho (rho* - rho): the carrying capacity rho* and the
       rate kappa are identifiable from the linear and quadratic terms at small rho, i.e. from data that never reaches
       saturation (the free sigma_theta(rho,U) of M2 destroys this identifiability).  rho* is initialised at 0.5 on purpose."""
    name = 'fisher_hodge_wfr'

    def __init__(s, sysname, H=128):
        super().__init__(sysname, H)
        if hasattr(s, 'r'):
            del s.r
        s.kr = MLP(1, H, 1)
        s.lrs = MLP(1, H, 1, zero_last=True)
        nn.init.constant_(s.lrs.net[-1].bias, math.log(0.5))

    def reservoir(s, U):
        u = U.unsqueeze(-1)
        return Fn.softplus(s.kr(u).squeeze(-1)), torch.exp(s.lrs(u).squeeze(-1))

    def rate(s, rho, ctx):
        F = ctx['F']
        mu = s.mu(rho, F)
        out = -S_SCALE * div_T(s.flux_q(rho, ctx, mu))
        kr, rs = s.reservoir(F['U'])
        mus = s.mu(rs, F)
        return out - rho * kr * logmean_s(rho, rs) * (mu - mus)

    def lyapunov(s, rho, F):
        kr, rs = s.reservoir(F['U'])
        mus = s.mu(rs, F)
        return (rho * torch.log(rho) + s.alpha * F['U'] * rho + s.g_density(rho, F) - mus * rho).mean((-1, -2))


MODELS2 = dict(models.MODELS, flux_gnn=FluxGNN, wgf_rev=WGFRev, jko_amort=JKOAmort, jko_inst=JKOInst,
               fisher_hodge_fr=FisherHodgeFR, fisher_hodge_fr_u=FisherHodgeFRU, fisher_hodge_wfr=FisherHodgeWFR)
