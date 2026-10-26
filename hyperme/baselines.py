"""Baselines with the HyperME interface: prepare(drive), forward / rhs(c, drive), advance(c, drive, Dt), and rates /
flux where the model has them.  All take the same hypergraph and featurizer as HyperME.

FluxGNN        black-box rates: a hypergraph message-passing network emits a forward and a backward rate for every
               hyperedge, a_h = r+_h c^{nu+}, b_h = r-_h c^{nu-}.  Conserves every moiety and stays positive, but has no
               free energy, no symmetric kinetics and no local detailed balance (the ablation of HyperME).
GNNODE         the same trunk decodes dc/dt at every state component directly (unstructured neural ODE).
CRNN           known stoichiometry with mass action and independent forward / backward rate constants from hyperedge
               features (Ji and Deng 2021), no cycle structure.
JKOReversible  reversible Wasserstein gradient flow of F = sum_m beta c_m log c_m + V_theta c_m with logarithmic-mean
               mobility, J = -kappa Lambda(c^{nu+}, c^{nu-}) S^T mu, trained by the JKO first-order condition
               (Rancati, Maas and Locatello, arXiv:2604.11311); A = 0 by construction.
FINN           finite volume neural network (Praditia et al. 2021; Karlbauer et al. 2022).  TRANSPORT ONLY, REACTIONS AS A
               SOURCE TERM: on every transport hyperedge a learned diffusion coefficient D_theta(c) times the two-point
               stencil plus a learned upwind advection, assembled conservatively; the reaction hyperedges are ignored
               and replaced by a learned node-local source R_theta (no stoichiometry, so moieties are not conserved).
"""
import math
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as Fn
from .model import MLP, Context, default_featurizer, logmean, SOFTPLUS_INV1
from .complex import _triplets
from .integrators import positive_ssp_rk2


class MLPLN(nn.Module):
    def __init__(s, i, h, o, ln=True, last='default'):
        super().__init__()
        s.net = MLP(i, h, o, last=last)
        s.ln = nn.LayerNorm(o) if ln else None

    def forward(s, x):
        y = s.net(x)
        return s.ln(y) if s.ln is not None else y


def heun_fixed(model, c, ctx, Dt, n=8):
    dt = Dt / n
    for _ in range(n):
        k1 = model.rhs(c, ctx)
        k2 = model.rhs(c + dt * k1, ctx)
        c = c + 0.5 * dt * (k1 + k2)
    return c, n


class _Base(nn.Module):
    def __init__(s, hg, featurizer=None, rho_scale=1.0, nsub=8, scale=1.0):
        super().__init__()
        s.hg, s.featurizer = hg, featurizer or default_featurizer(hg)
        s.RS, s.nsub, s.scale = rho_scale, nsub, scale

    def prepare(s, drive):
        return Context(feats=s.featurizer(drive))

    def context(s, drive):
        return drive if isinstance(drive, Context) else s.prepare(drive)

    def forward(s, c, drive):
        return s.rhs(c, drive)

    def advance(s, c, drive, Dt):
        return heun_fixed(s, c, s.context(drive), Dt, s.nsub)

    def escape_ratio(s, c, a, b):
        D = s.hg.consumption(a, b)
        return torch.where(c > 0, D / torch.where(c > 0, c, torch.ones_like(c)), torch.zeros_like(c)).amax(-1)

    def node_inputs(s, c, ctx):
        feats = ctx['feats']; B = c.shape[0]
        t = s.hg.on(c.device, c.dtype)
        cols = [c[..., None] / s.RS] + [v[:, t['site_of']][..., None] for v in feats['node'].values()]
        if s.hg.n_species > 1:
            cols.append(Fn.one_hot(t['species_of'], s.hg.n_species).to(c.dtype).expand(B, -1, -1))
        return torch.cat(cols, -1)

    def edge_inputs(s, ctx):
        f = ctx['feats']
        return torch.cat([f['edge_sym'], f['edge_ori'][..., None]], -1)

    @staticmethod
    def n_node_in(hg, featurizer, drive_example):
        f = featurizer(drive_example)
        return 1 + len(f['node']) + (hg.n_species if hg.n_species > 1 else 0), f['edge_sym'].shape[-1] + 1


class HyperGNNTrunk(nn.Module):
    """bipartite message passing between state components (nodes) and hyperedges; reactant and product sides are
    aggregated separately, so hyperedge latents are orientation aware."""
    def __init__(s, hg, n_node_in, n_edge_in, hidden=64, layers=3):
        super().__init__()
        s.hg = hg
        s.enc_n = MLPLN(n_node_in, hidden, hidden)
        s.enc_e = MLPLN(n_edge_in, hidden, hidden)
        s.me = nn.ModuleList([MLPLN(3 * hidden, hidden, hidden) for _ in range(layers)])
        s.mn = nn.ModuleList([MLPLN(3 * hidden, hidden, hidden) for _ in range(layers)])
        for nm, N in (('p', hg.Np), ('m', hg.Nm)):
            h, m, v = _triplets(N)
            s.register_buffer(f'{nm}_h', torch.as_tensor(h)); s.register_buffer(f'{nm}_m', torch.as_tensor(m))
            s.register_buffer(f'{nm}_v', torch.as_tensor(v, dtype=torch.get_default_dtype()))

    def _gather(s, hn, side):
        h, m, v = getattr(s, f'{side}_h'), getattr(s, f'{side}_m'), getattr(s, f'{side}_v').to(hn.dtype)
        out = torch.zeros(hn.shape[0], s.hg.H, hn.shape[-1], dtype=hn.dtype, device=hn.device)
        return out.index_add(1, h, hn[:, m] * v[:, None])

    def _scatter(s, he, side):
        h, m, v = getattr(s, f'{side}_h'), getattr(s, f'{side}_m'), getattr(s, f'{side}_v').to(he.dtype)
        out = torch.zeros(he.shape[0], s.hg.M, he.shape[-1], dtype=he.dtype, device=he.device)
        return out.index_add(1, m, he[:, h] * v[:, None])

    def forward(s, xn, xe):
        hn, he = s.enc_n(xn), s.enc_e(xe)
        for me, mn in zip(s.me, s.mn):
            he = he + me(torch.cat([he, s._gather(hn, 'p'), s._gather(hn, 'm')], -1))
            hn = hn + mn(torch.cat([hn, s._scatter(he, 'p'), s._scatter(he, 'm')], -1))
        return hn, he


class FluxGNN(_Base):
    name = 'flux_gnn'

    def __init__(s, hg, n_node_in, n_edge_in, featurizer=None, hidden=64, layers=3, rho_scale=1.0, scale=1.0, step=None):
        super().__init__(hg, featurizer, rho_scale, scale=scale)
        s.trunk = HyperGNNTrunk(hg, n_node_in, n_edge_in, hidden, layers)
        s.dec = MLP(hidden, hidden, 2, last='zero', last_bias=SOFTPLUS_INV1)        # unit rates at init
        s.step = dict(dt_max=None, theta0=0.5, theta1=0.5, max_sub=4000, t_tol=1e-9); s.step.update(step or {})

    def rates(s, c, drive):
        ctx = s.context(drive)
        _, he = s.trunk(s.node_inputs(c, ctx), s.edge_inputs(ctx))
        r = s.scale * Fn.softplus(s.dec(he))
        return r[..., 0] * s.hg.mass_action(c, '+'), r[..., 1] * s.hg.mass_action(c, '-')

    def flux(s, c, drive):
        a, b = s.rates(c, drive)
        return a - b

    def rhs(s, c, drive):
        return s.hg.apply_S(s.flux(c, drive))

    def advance(s, c, drive, Dt):
        return positive_ssp_rk2(s, c, s.context(drive), Dt, **s.step)


class GNNODE(_Base):
    name = 'gnn_ode'

    def __init__(s, hg, n_node_in, n_edge_in, featurizer=None, hidden=64, layers=3, rho_scale=1.0, scale=1.0, nsub=8):
        super().__init__(hg, featurizer, rho_scale, nsub, scale)
        s.trunk = HyperGNNTrunk(hg, n_node_in, n_edge_in, hidden, layers)
        s.dec = MLP(hidden, hidden, 1, last='zero')

    def rhs(s, c, drive):
        ctx = s.context(drive)
        hn, _ = s.trunk(s.node_inputs(c, ctx), s.edge_inputs(ctx))
        return s.scale * s.dec(hn).squeeze(-1)


class CRNN(_Base):
    name = 'crnn'

    def __init__(s, hg, n_edge_in, featurizer=None, hidden=64, scale=1.0, nsub=8, oriented=True):
        super().__init__(hg, featurizer, 1.0, nsub, scale)
        s.oriented = oriented                  # False: hyperedge features only (hyper/crn2.py CRNN2), n_edge_in excludes ori
        s.kmlp = MLP(n_edge_in, hidden, 2, last_bias=0.5)

    def prepare(s, drive):
        ctx = super().prepare(drive)
        x = s.edge_inputs(ctx) if s.oriented else ctx['feats']['edge_sym']
        kk = s.scale * Fn.softplus(s.kmlp(x)) + 1e-6
        ctx['kf'], ctx['kb'] = kk[..., 0], kk[..., 1]
        return ctx

    def rates(s, c, drive):
        ctx = s.context(drive)
        return ctx['kf'] * s.hg.mass_action(c, '+'), ctx['kb'] * s.hg.mass_action(c, '-')

    def flux(s, c, drive):
        a, b = s.rates(c, drive)
        return a - b

    def rhs(s, c, drive):
        return s.hg.apply_S(s.flux(c, drive))

    def cycle_affinity(s, drive):
        ctx = s.context(drive)
        return s.hg.project_cycles(torch.log(ctx['kf'] / ctx['kb']))


class JKOReversible(_Base):
    name = 'jko_rev'

    def __init__(s, hg, node_inputs=('U',), featurizer=None, hidden=64, scale=1.0, step=None):
        super().__init__(hg, featurizer, 1.0, scale=scale)
        s.node_keys = tuple(node_inputs)
        s.V = MLP(len(s.node_keys) + (hg.n_species if hg.n_species > 1 else 0), hidden, 1, last='zero')
        s.beta_raw = nn.Parameter(torch.tensor(SOFTPLUS_INV1))
        s.step = dict(dt_max=None); s.step.update(step or {})

    def beta(s):
        return Fn.softplus(s.beta_raw)

    def prepare(s, drive):
        ctx = super().prepare(drive)
        f = ctx['feats']; t = s.hg.on(f['edge_sym'].device, f['edge_sym'].dtype)
        cols = [f['node'][k][:, t['site_of']][..., None] for k in s.node_keys]
        if s.hg.n_species > 1:
            cols.append(Fn.one_hot(t['species_of'], s.hg.n_species).to(f['edge_sym'].dtype).expand(cols[0].shape[0], -1, -1))
        ctx['V'] = s.V(torch.cat(cols, -1)).squeeze(-1)
        return ctx

    def mu(s, c, drive):
        ctx = s.context(drive)
        return s.beta() * torch.log(c) + ctx['V']

    def mobility(s, c):
        return s.scale * logmean(s.hg.mass_action(c, '+'), s.hg.mass_action(c, '-'))

    def flux(s, c, drive):
        return -s.mobility(c) * s.hg.apply_ST(s.mu(c, drive))

    def rhs(s, c, drive):
        return s.hg.apply_S(s.flux(c, drive))

    def free_energy_value(s, c, drive):
        ctx = s.context(drive)
        return (s.beta() * c * torch.log(c) + ctx['V'] * c).sum(-1)

    def advance(s, c, drive, Dt):
        from .integrators import positive_heun_rhs
        return positive_heun_rhs(s, c, s.context(drive), Dt, **s.step)

    def jko_loss(s, c_prev, c_next, tau, drive):
        """JKO first-order condition: with psi solving (c_next - c_prev)/tau = -S diag(w) S^T psi (w = mobility at
        c_next), loss = sum_h w (S^T (mu(c_next) - psi))^2 / sum_h w (S^T psi)^2."""
        with torch.no_grad():
            w = s.mobility(c_next).double()
            S = torch.as_tensor(s.hg.S, dtype=torch.float64, device=c_next.device)
            Lb = s.hg.on(c_next.device, torch.float64)['L']
            K = torch.einsum('mh,bh,nh->bmn', S, w, S) + w.mean(-1)[:, None, None] * (Lb @ Lb.T)[None]
            rhs = -((c_next - c_prev).double() / tau)
            psi = torch.linalg.solve(K, rhs.unsqueeze(-1)).squeeze(-1).to(c_next.dtype)
            w = w.to(c_next.dtype)
        r = s.hg.apply_ST(s.mu(c_next, drive) - psi)
        return (w * r * r).sum(-1).mean() / (w * s.hg.apply_ST(psi) ** 2).sum(-1).mean().clamp_min(1e-12)


class FINN(_Base):
    name = 'finn'

    def __init__(s, hg, n_edge_in, featurizer=None, hidden=64, rho_scale=1.0, scale=1.0, nsub=8, reaction=True):
        super().__init__(hg, featurizer, rho_scale, nsub, scale)
        assert hg.unimolecular or 'odd' in hg.meta
        t = hg.meta['odd'] if not hg.unimolecular else np.ones(hg.H, bool)
        tr = np.nonzero(t)[0]
        Np, Nm = hg.Np[tr].tocoo(), hg.Nm[tr].tocoo()
        tail = np.zeros(len(tr), np.int64); head = np.zeros(len(tr), np.int64)
        tail[Np.row] = Np.col; head[Nm.row] = Nm.col
        s.register_buffer('tr', torch.as_tensor(tr)); s.register_buffer('tail', torch.as_tensor(tail))
        s.register_buffer('head', torch.as_tensor(head))
        s.Dnet = MLP(n_edge_in - 1 + 1, hidden, 1, last='zero', last_bias=SOFTPLUS_INV1)   # phi_D(c_mean, edge feats)
        s.vnet = MLP(n_edge_in, hidden, 1, last='zero')                                      # advection velocity
        s.reaction = reaction and not hg.unimolecular
        if s.reaction:
            s.Rnet = MLP(hg.n_species + 1, hidden, hg.n_species, last='zero')               # phi_R(c at a site, U)

    def flux_transport(s, c, ctx):
        f = ctx['feats']
        sym = f['edge_sym'][:, s.tr]; ori = f['edge_ori'][:, s.tr]
        ci, cj = c[:, s.tail], c[:, s.head]
        D = Fn.softplus(s.Dnet(torch.cat([sym, (0.5 * (ci + cj) / s.RS)[..., None]], -1)).squeeze(-1))
        v = s.vnet(torch.cat([sym, ori[..., None]], -1)).squeeze(-1) - s.vnet(torch.cat([sym, -ori[..., None]], -1)).squeeze(-1)
        return s.scale * (D * (ci - cj) + Fn.relu(v) * ci - Fn.relu(-v) * cj)

    def rhs(s, c, drive):
        ctx = s.context(drive)
        J = torch.zeros(c.shape[0], s.hg.H, dtype=c.dtype, device=c.device).index_copy(1, s.tr, s.flux_transport(c, ctx))
        out = s.hg.apply_S(J)
        if s.reaction:
            B = c.shape[0]
            x = torch.cat([c.view(B, s.hg.n_species, s.hg.V).transpose(1, 2), ctx['feats']['node']['U'][..., None]], -1)
            out = out + s.scale * s.Rnet(x).transpose(1, 2).reshape(B, -1)
        return out
