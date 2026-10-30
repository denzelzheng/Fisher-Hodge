"""Models of the Brusselator benchmark: the Neural Chart and its ablations, the oracle, and the baselines.

Neural Chart.  mu = log c + mu0 for X and Y; the drive is the chemostat force of (a, b, d, e); log k is a table over the
six hyperedge types (R1-R4, transport of X and Y), log omega_c; there are no material descriptors and no state term.
The chemostats also enter k through their symmetric mass-action factor.  The standard potentials mu0 of X, Y and of
the chemostats are given (thermochemical data) and fixed, so the force f of every hyperedge is fixed and only k is
learned.  The gauge of mu0 is the minimum-norm solution of <nu+ - nu-, mu0> = log(k_r / eps); any other gauge is
absorbed by k.

Baselines (all see the same data):
  crnn_mass_action  CRNN with a log rate table over the hyperedge types and the chemostats as mass-action factors
  crnn_features     CRNN whose rate constants are an MLP of the hyperedge type and the z-scored (log a, log b)
  flux_gnn          Flux-GNN: hypergraph message passing emits forward and backward rates (times mass action)
  ude               UDE closure: the same hypergraph and conservative assembly, a free flux on every hyperedge
  fno, bsms, generic_fno   one-step maps of the grid fields
  persistence       u(t) = u(0)
"""
import contextlib
import math

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as Fn

from neuralchart import fast
from neuralchart.chart import NeuralChart, is_chart, chart_config
from neuralchart.model import MLP, Context, featurizer
from neuralchart.baselines import FluxGNN, CRNN
from neuralchart.graphs import grid_edge_list
from .config import STEP, KF, EPS, DIFF, H_GRID, N_TRAIN
from .data import hg_of

TASK = 'brusselator'
GRAPH_BUFFERS = ('p_h', 'p_m', 'p_v', 'm_h', 'm_m', 'm_v')


@contextlib.contextmanager
def default_dtype(dt):
    old = torch.get_default_dtype()
    torch.set_default_dtype(dt)
    try:
        yield
    finally:
        torch.set_default_dtype(old)


def chem_featurizer(hg):
    """condition dict(logab (B, 2)) -> features: the log concentrations of the chemostats A, B (D, E at 1) and the
    hyperedge type as the only feature of the mobility"""
    base, nC = featurizer(hg), len(hg.meta['chemostats'])

    def feat(F):
        lab = F['logab']
        if lab.shape[1] < nC:
            lab = torch.cat([lab, torch.zeros(lab.shape[0], nC - lab.shape[1], dtype=lab.dtype, device=lab.device)], -1)
        return base(dict(chem_logc=lab))
    feat.n_edge_sym = base.n_edge_sym
    return feat


# ------------------------------------------------------------------------------------------ the Neural Chart
def chart_model(n, cls=NeuralChart):
    """the Neural Chart class on the n x n grid (standard potentials still free, log k = 0)"""
    hg = hg_of(n)
    with default_dtype(torch.float64):
        m = cls.from_parts(hg, k_ref=1.0, T_exponent=0.0, harmonic=False, featurizer=chem_featurizer(hg), step=dict(STEP),
                           learn_mu0=True, learn_chem_mu0=True, kinetics_table=True, chem_prefactor=True)
    m = m.double()
    m.step = dict(STEP)
    return m


@torch.no_grad()
def set_true(m):
    """the reference in the Neural Chart class (exact for every a, b): reaction r has forward constant k_r and reverse
    constant eps, so log k = log sqrt(k_r eps) and <nu+ - nu-, mu0> = log(k_r / eps); transport has k = D / h^2"""
    k = torch.as_tensor(KF, dtype=torch.float64)
    x = torch.log(k / EPS)
    nu = torch.tensor([[-1., 0., 1., 0., 0., 0.],       # <nu+ - nu-> over (X, Y, A, B, D, E); R1 A -> X
                       [-1., 1., 0., 0., 0., 0.],       # R2 2X + Y -> 3X
                       [1., -1., 0., 1., -1., 0.],      # R3 B + X -> Y + D
                       [1., 0., 0., 0., 0., -1.]], dtype=torch.float64)   # R4 X -> E
    mu0 = torch.linalg.pinv(nu) @ x                     # minimum-norm solution (the two gauge directions at 0)
    assert torch.allclose(nu @ mu0, x, atol=1e-12)
    m.free_energy.mu0.copy_(mu0[:2]); m.drive.chem_mu0.copy_(mu0[2:])
    D = torch.tensor([DIFF[0] / H_GRID ** 2, DIFF[1] / H_GRID ** 2], dtype=torch.float64)
    m.kinetics.core.copy_(torch.cat([0.5 * torch.log(k * EPS), torch.log(D)]))
    return m


def true_log_k():
    """log k of the reference per hyperedge type (R1-R4, transport of X, Y)"""
    return [0.5 * math.log(k * EPS) for k in KF] + [math.log(d / H_GRID ** 2) for d in DIFF]


@torch.no_grad()
def give_mu0(m):
    """the given standard potentials: set to the reference values and frozen; log k starts at 0 and the closed-form
    step solves for k alone (a convex NNLS problem)"""
    set_true(m)
    m.kinetics.core.zero_()
    m.free_energy.mu0.requires_grad_(False); m.drive.chem_mu0.requires_grad_(False)
    m.linear_parameters = lambda: [m.kinetics.core]
    return m


def build_chart(cfg, n=N_TRAIN, norm=None):
    """the Neural Chart of a ChartConfig: log k = log omega over the six hyperedge types"""
    assert cfg.task == TASK and cfg.kappa in ('learned', 'true') and not cfg.descriptors and not cfg.state
    m = chart_model(n)
    if cfg.kappa == 'true':
        set_true(m)                                 # oracle: the true k and mu0
    elif cfg.mu0 == 'given':
        give_mu0(m)
    if cfg.drive == 'off':
        m.knobs['drive_scale'] = 0.0                # A = 0: detailed balance (the reversible ablation)
    m.drive.s_A.requires_grad_(False)               # no external force field (s_A multiplies nothing)
    m.config = cfg
    return m


# ------------------------------------------------------------------------------------------ baselines
def bb_featurizer(hg, norm):
    """inputs of the black boxes: the hyperedge type and the z-scored (log a, log b) as features"""
    ntyp = int(hg.meta['typ'].max()) + 1
    mu, sd = torch.tensor(norm['lab_mean']), torch.tensor(norm['lab_std'])

    def feat(F):
        lab = F['logab']
        z = (lab - mu.to(lab)) / sd.to(lab)
        B = lab.shape[0]
        t = hg.on(lab.device, lab.dtype)
        oh = Fn.one_hot(t['typ'], ntyp).to(lab.dtype).expand(B, hg.H, ntyp)
        sym = torch.cat([oh, z[:, None, :].expand(B, hg.H, 2)], -1)
        node = dict(za=z[:, :1].expand(B, hg.V), zb=z[:, 1:2].expand(B, hg.V))
        return dict(U=torch.zeros(B, hg.V, dtype=lab.dtype, device=lab.device), node=node, edge_sym=sym,
                    edge_ori=torch.zeros_like(sym[..., 0]), force=None, raw=F, z=z)
    feat.n_edge_sym = ntyp + 2
    return feat


def _ckpt(fn, *args):
    """activation checkpointing while training (memory only)"""
    if torch.is_grad_enabled():
        from torch.utils.checkpoint import checkpoint
        return checkpoint(fn, *args, use_reentrant=False)
    return fn(*args)


class CheckpointedFluxGNN(FluxGNN):
    """Flux-GNN with activation checkpointing of every rate evaluation (memory only)"""
    def rates(s, c, drive):
        ctx = s.context(drive)
        return _ckpt(lambda c_: FluxGNN.rates(s, c_, ctx), c)


def make_flux_gnn(n, norm):
    hg = hg_of(n)
    f = bb_featurizer(hg, norm)
    nn_in, ne_in = FluxGNN.n_node_in(hg, f, dict(logab=torch.zeros(1, 2)))
    return CheckpointedFluxGNN(hg, nn_in, ne_in, f, hidden=64, layers=3, rho_scale=norm['c_scale'], step=dict(STEP))


def make_crnn(n, norm, mass_action):
    """mass_action=True: a log rate table over the hyperedge types, the chemostats as exact mass-action factors (the
    class of the reference).  False: rate constants from an MLP of the type and the z-scored (log a, log b)."""
    hg = hg_of(n)
    f = chem_featurizer(hg) if mass_action else bb_featurizer(hg, norm)
    with default_dtype(torch.float64):
        m = CRNN(hg, f.n_edge_sym, f, oriented=False, chem=mass_action, table=mass_action, step=dict(STEP))
    return m.double()


class UDE(nn.Module):
    """UDE closure: the same hypergraph and conservative assembly dc/dt = S J with a free flux on every hyperedge,
    J_h = g(type, u_p, u_q, z) for a reaction at a site (u_p, u_q = x, y there) and J_e = g(type, u_i, u_j, z) -
    g(type, u_j, u_i, z) for transport (antisymmetric), with u = c / c_scale and z the z-scored (log a, log b).  g is an
    MLP (64 hidden units) with a zero last layer.  There are no exponential-family rates and no local detailed balance,
    so the model is integrated with error control only (neuralchart.fast, control 'err_only'); the one-way parts
    relu(J), relu(-J) are used for the diagnostics."""
    name = 'ude'
    free_flux = True

    def __init__(s, n, norm, hidden=64):
        super().__init__()
        s.norm = norm
        s.net = MLP(6 + 2 + 2, hidden, 1, last='zero')
        s.step = dict(STEP)
        s.set_n(n)

    def set_n(s, n):
        hg = hg_of(n)
        s.hg = hg
        V = hg.V
        P = torch.as_tensor(hg.rp_idx[:, 0].copy()); Q = torch.as_tensor(hg.rm_idx[:, 0].copy())
        site = torch.as_tensor(np.arange(hg.H) % V)
        tr = torch.as_tensor(hg.meta['odd'])
        P = torch.where(tr, P, site); Q = torch.where(tr, Q, V + site)       # reactions: (x, y) at the site
        s._P, s._Q, s._tr = P, Q, tr
        s._typ = torch.as_tensor(hg.meta['typ'])
        return s

    def prepare(s, drive):
        lab = drive['logab']
        z = (lab - torch.tensor(s.norm['lab_mean']).to(lab)) / torch.tensor(s.norm['lab_std']).to(lab)
        dev = lab.device
        return Context(z=z, P=s._P.to(dev), Q=s._Q.to(dev), tr=s._tr.to(dev).to(lab.dtype),
                       oh=Fn.one_hot(s._typ.to(dev), 6).to(lab.dtype))

    def context(s, drive):
        return drive if isinstance(drive, Context) else s.prepare(drive)

    def flux(s, c, drive):
        ctx = s.context(drive)

        def f(c_):
            u = c_ / s.norm['c_scale']
            B, H = u.shape[0], ctx['P'].shape[0]
            p, q = u[:, ctx['P']], u[:, ctx['Q']]
            oh = ctx['oh'][None].expand(B, H, 6); z = ctx['z'][:, None, :].expand(B, H, 2)
            g = lambda x, y: s.net(torch.cat([oh, x[..., None], y[..., None], z], -1).to(s.net.net[0].weight.dtype)).squeeze(-1).to(c_.dtype)
            return g(p, q) - ctx['tr'][None] * g(q, p)
        return _ckpt(f, c)

    def rates(s, c, drive):
        J = s.flux(c, drive)
        a = J.clamp_min(0)
        return a, a - J

    def forward(s, c, drive):
        return s.hg.apply_S(s.flux(c, drive))

    def escape_ratio(s, c, a, b):
        return (s.hg.consumption(a, b) / (c.clamp_min(0) + 0.01)).amax(-1)

    def advance(s, c, drive, Dt, **kw):
        from neuralchart.integrators import positive_ssp_rk2
        return positive_ssp_rk2(s, c, s.context(drive), Dt, **s.step)


class SpectralConv2d(nn.Module):
    def __init__(s, ci, co, m1, m2):
        super().__init__()
        s.m1, s.m2 = m1, m2
        sc = 1 / (ci * co)
        s.w1 = nn.Parameter(sc * torch.rand(ci, co, m1, m2, dtype=torch.cfloat))
        s.w2 = nn.Parameter(sc * torch.rand(ci, co, m1, m2, dtype=torch.cfloat))

    def forward(s, x):
        B, C, Hh, W = x.shape
        xf = torch.fft.rfft2(x)
        out = torch.zeros(B, s.w1.shape[1], Hh, W // 2 + 1, dtype=torch.cfloat, device=x.device)
        out[:, :, :s.m1, :s.m2] = torch.einsum('bixy,ioxy->boxy', xf[:, :, :s.m1, :s.m2], s.w1)
        out[:, :, -s.m1:, :s.m2] = torch.einsum('bixy,ioxy->boxy', xf[:, :, -s.m1:, :s.m2], s.w2)
        return torch.fft.irfft2(out, s=(Hh, W))


class FNO(nn.Module):
    """FNO (Li et al. 2021; width 32, 12 modes, 4 layers) with two state channels: the residual next-step map
    c_{t+DT} = c_t + FNO([x, y, z_a, z_b]) on any n x n grid of the training spacing"""
    name = 'fno'

    def __init__(s, norm, width=32, modes=12, layers=4):
        super().__init__()
        s.norm = norm
        s.lift = nn.Conv2d(4, width, 1)
        s.spec = nn.ModuleList([SpectralConv2d(width, width, modes, modes) for _ in range(layers)])
        s.w = nn.ModuleList([nn.Conv2d(width, width, 1) for _ in range(layers)])
        s.p1, s.p2 = nn.Conv2d(width, 128, 1), nn.Conv2d(128, 2, 1)

    def net(s, x):
        x = s.lift(x)
        for i, (sp_, w) in enumerate(zip(s.spec, s.w)):
            x = sp_(x) + w(x)
            if i < len(s.spec) - 1:
                x = Fn.gelu(x)
        return s.p2(Fn.gelu(s.p1(x)))

    def prepare(s, drive):
        lab = drive['logab']
        z = (lab - torch.tensor(s.norm['lab_mean']).to(lab)) / torch.tensor(s.norm['lab_std']).to(lab)
        return Context(z=z.float())

    def context(s, drive):
        return drive if isinstance(drive, Context) else s.prepare(drive)

    def advance(s, c, drive, Dt, **kw):
        ctx = s.context(drive)
        B = c.shape[0]
        n = int(round((c.shape[1] // 2) ** 0.5))
        u = c.reshape(B, 2, n, n).float() / s.norm['c_scale']
        x = torch.cat([u, ctx['z'][:, :, None, None].expand(B, 2, n, n)], 1)
        return ((u + s.net(x)) * s.norm['c_scale']).reshape(B, -1).to(c.dtype), 1


class BSMS(nn.Module):
    """BSMS-GNN on the grid sites: node inputs the normalised (x, y) and z, edge inputs the relative position
    (dx, dy, |d|) in grid units; predicts the normalised one-step change of (x, y)"""
    name = 'bsms'

    def __init__(s, n, norm, levels=4, latent=128):
        super().__init__()
        from neuralchart.baselines.bsms_gnn import BSMSGNN
        s.norm, s.levels = norm, levels
        s.net = BSMSGNN(4, 3, levels, latent, n_out=2)
        for k in ('c_mean', 'c_std', 'd_mean', 'd_std'):
            s.register_buffer(k, torch.zeros(2))
        s.register_buffer('noise', torch.zeros(()))
        s.set_n(n)

    def set_n(s, n):
        from neuralchart.baselines.bsms_gnn import bistride_hierarchy
        s.n = n
        E = grid_edge_list(n)
        ij = np.stack(np.meshgrid(np.arange(n), np.arange(n), indexing='ij'), -1).reshape(-1, 2)
        pts = ij[:, ::-1] / n                       # site v = i n + j at (x = j / n, y = i / n)
        s._hier = bistride_hierarchy(E, pts, s.levels)
        s._H = None
        return s

    def prepare(s, drive):
        from neuralchart.baselines.bsms_gnn import hierarchy_tensors
        lab = drive['logab']
        dev = lab.device
        if s._H is None or s._H['snd'][0].device != dev:
            s._H = hierarchy_tensors(s._hier, dev, s.n)
        z = (lab - torch.tensor(s.norm['lab_mean']).to(lab)) / torch.tensor(s.norm['lab_std']).to(lab)
        B = lab.shape[0]
        return Context(z=z.float(), H=s._H, xe=s._H['geo'][0][None].expand(B, -1, -1))

    def context(s, drive):
        return drive if isinstance(drive, Context) else s.prepare(drive)

    def nodes(s, c):
        B = c.shape[0]
        return c.reshape(B, 2, -1).transpose(1, 2).float()          # (B, V, 2)

    def delta(s, u, ctx):
        """normalised change from the node states u (B, V, 2)"""
        V = u.shape[1]
        xn = torch.cat([(u - s.c_mean) / s.c_std, ctx['z'][:, None, :].expand(-1, V, 2)], -1)
        return s.net(xn, ctx['xe'], ctx['H'])

    def advance(s, c, drive, Dt, **kw):
        ctx = s.context(drive)
        u = s.nodes(c)
        u = u + s.delta(u, ctx) * s.d_std + s.d_mean
        return u.transpose(1, 2).reshape(c.shape[0], -1).to(c.dtype), 1


class GenericFNOMap(nn.Module):
    """GENERIC-FNO (neuralchart.baselines.generic_fno) as a next-step map: the fields X, Y scaled by c_scale, the
    z-scored (log a, log b) as two constant condition channels"""
    name = 'generic_fno'

    def __init__(s, norm):
        super().__init__()
        from neuralchart.baselines.generic_fno import GenericFNO
        s.norm, s.NS, s.NC = norm, 2, 2
        s.net = GenericFNO(n_fields=s.NS, n_cond=s.NC)

    def cond_fields(s, z, n):
        return z[:, :, None, None].expand(z.shape[0], z.shape[1], n, n).contiguous()

    def prepare(s, drive):
        lab = drive['logab']
        return Context(z=((lab - torch.tensor(s.norm['lab_mean']).to(lab)) / torch.tensor(s.norm['lab_std']).to(lab)).float())

    def context(s, drive):
        return drive if isinstance(drive, Context) else s.prepare(drive)

    def advance(s, c, drive, Dt, **kw):
        ctx = s.context(drive)
        Bn = c.shape[0]
        n = int(round((c.shape[1] // s.NS) ** 0.5))
        u = c.reshape(Bn, s.NS, n, n).float() / s.norm['c_scale']
        with torch.enable_grad():                       # dE/du, dS/du by automatic differentiation
            v = s.net(u.detach(), s.cond_fields(ctx['z'], n), create_graph=False).detach()
        bad = ~torch.isfinite(v).flatten(1).all(1) | (v.flatten(1).abs().amax(1) > 1e4)
        v = torch.where(bad[:, None, None, None], torch.full_like(v, float('nan')), v)
        return (v * s.norm['c_scale']).reshape(Bn, -1).to(c.dtype), 1


class Persistence(nn.Module):
    name = 'persistence'

    def prepare(s, drive):
        return Context()

    def context(s, drive):
        return drive if isinstance(drive, Context) else s.prepare(drive)

    def advance(s, c, drive, Dt, **kw):
        return c, 1


# ------------------------------------------------------------------------------------------ by name
FLOAT32_MODELS = ('fno', 'bsms', 'flux_gnn', 'ude', 'generic_fno')      # trained and evaluated in float32


def build(name, n, norm):
    """a model by name: a Neural Chart (neuralchart.chart.CONFIGS['brusselator']) or a baseline"""
    if is_chart(TASK, name):
        return build_chart(chart_config(TASK, name), n, norm)
    if name in ('crnn_mass_action', 'crnn_features'):
        return make_crnn(n, norm, name == 'crnn_mass_action')
    if name == 'flux_gnn':
        return make_flux_gnn(n, norm)
    if name == 'ude':
        return UDE(n, norm)
    if name == 'fno':
        return FNO(norm)
    if name == 'bsms':
        return BSMS(n, norm)
    if name == 'generic_fno':
        return GenericFNOMap(norm)
    if name == 'persistence':
        return Persistence()
    raise ValueError(name)


def next_dtype(model):
    p = next(iter(model.parameters()), None)
    return p.dtype if p is not None and not p.is_complex() else torch.float64


def rebind(name, model, n, norm, device):
    """the same trained parameters on another grid (no retraining)"""
    if name in ('fno', 'generic_fno', 'persistence'):
        return model
    if name in ('ude', 'bsms'):
        kw = dict(model._fast.kw) if getattr(model, '_fast', None) is not None else None
        model.set_n(n)
        if kw is not None:                 # rebound in place: the fast operators belong to the old grid
            fast.restore(model); fast.accelerate(model, **kw)
        return model
    new = build(name, n, norm).to(device).to(next_dtype(model))
    gb = lambda k: k.split('.')[-1] in GRAPH_BUFFERS     # exact buffer names ('chem_mu0' contains 'm_m')
    sd = {k: v for k, v in model.state_dict().items() if not gb(k)}
    missing, unexpected = new.load_state_dict(sd, strict=False)
    assert not unexpected and all(gb(k) for k in missing), (missing, unexpected)
    return new.eval()
