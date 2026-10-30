"""Graph baselines that share the hypergraph of the Neural Chart and its positive integrator.

FluxGNN   black-box rates from hypergraph message passing (conservative and positive, no local detailed balance)
CRNN      mass action with independent forward / backward rate constants
"""
import torch
import torch.nn as nn
import torch.nn.functional as Fn
from ..model import MLP, Context, SOFTPLUS_INV1
from ..graphs import _triplets
from ..integrators import positive_ssp_rk2


class MLPLN(nn.Module):
    def __init__(s, i, h, o, ln=True, last='default'):
        super().__init__()
        s.net = MLP(i, h, o, last=last)
        s.ln = nn.LayerNorm(o) if ln else None

    def forward(s, x):
        y = s.net(x)
        return s.ln(y) if s.ln is not None else y


class _Base(nn.Module):
    """prepare(drive), rates(c, drive), rhs(c, drive) and advance(c, drive, Dt) with the positive SSP-RK2."""
    def __init__(s, hg, featurizer, rho_scale=1.0, scale=1.0):
        super().__init__()
        s.hg, s.featurizer = hg, featurizer
        s.RS, s.scale = rho_scale, scale

    def prepare(s, drive):
        return Context(feats=s.featurizer(drive))

    def context(s, drive):
        return drive if isinstance(drive, Context) else s.prepare(drive)

    def forward(s, c, drive):
        return s.rhs(c, drive)

    def advance(s, c, drive, Dt, **kw):
        return positive_ssp_rk2(s, c, s.context(drive), Dt, **s.step)

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
    """black-box rates: a hypergraph message-passing network emits a forward and a backward rate for every hyperedge,
    a_h = r+_h c^{nu+}, b_h = r-_h c^{nu-}.  Conserves every moiety and stays positive, but has no free energy, no
    symmetric mobility and no local detailed balance."""
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

class CRNN(_Base):
    """chemical reaction neural network (Ji and Deng 2021): the known stoichiometry with mass action and independent
    forward / backward rate constants.  chem=True: the chemostat concentrations enter as exact mass-action factors and
    the log rate constants are a table over the hyperedge types (CRNN, mass action); chem=False: the rate constants are
    an MLP of the hyperedge features and the condition features (CRNN, condition features).  No cycle structure, so
    local detailed balance is not built in."""
    name = 'crnn'

    def __init__(s, hg, n_edge_in, featurizer, hidden=64, scale=1.0, oriented=True, chem=False, table=False, step=None):
        super().__init__(hg, featurizer, 1.0, scale)
        s.step = dict(dt_max=None, theta0=0.5, theta1=0.5, max_sub=4000, t_tol=1e-9); s.step.update(step or {})
        s.oriented = oriented                  # False: hyperedge features only (n_edge_in excludes the oriented column)
        s.chem = chem                          # True: chemostat concentrations enter as exact mass-action factors
        s.table = table                        # True: log kf, log kb linear in the features (a table for one-hot types)
        if table:
            s.kmlp = nn.Linear(n_edge_in, 2, bias=False); nn.init.zeros_(s.kmlp.weight)
        else:
            s.kmlp = MLP(n_edge_in, hidden, 2, last_bias=0.5)

    def prepare(s, drive):
        ctx = super().prepare(drive)
        x = s.edge_inputs(ctx) if s.oriented else ctx['feats']['edge_sym']
        kk = s.scale * (torch.exp(s.kmlp(x)) if s.table else Fn.softplus(s.kmlp(x)) + 1e-6)
        ctx['kf'], ctx['kb'] = kk[..., 0], kk[..., 1]
        if s.chem:
            lc = ctx['feats']['chem_logc']
            Y = torch.as_tensor(s.hg.meta['chem_S'].toarray(), dtype=lc.dtype, device=lc.device)      # (H, nC V) = nu- - nu+
            nC = len(s.hg.meta['chemostats'])
            lcv = lc.repeat_interleave(s.hg.V, 1) if lc.shape[1] == nC else lc
            ctx['kf'] = ctx['kf'] * torch.exp(lcv @ (-Y).clamp_min(0).T)       # chemostats consumed forward
            ctx['kb'] = ctx['kb'] * torch.exp(lcv @ Y.clamp_min(0).T)          # chemostats consumed backward
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

    def linear_parameters(s):
        """with table=True log kf and log kb are linear in the table: the closed-form fit applies"""
        return [s.kmlp.weight] if s.table else []

    def fit_linear(s, data, dt, **kw):
        from ..fit import fit_linear
        return fit_linear(s, data, dt, **kw)
