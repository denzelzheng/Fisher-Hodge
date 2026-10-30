"""Models of the reaction-network benchmarks: the Neural Chart of each network and the baselines.

Neural Chart.  mu = log c + mu0 (+ ln gamma_theta(c) on the non-ideal Schloegl model), the drive is the chemostat force
with the given chemostat potentials, log k is one value per hyperedge type (omega_c), and the chemostats also enter k
through their mass-action factor.  The standard potentials mu0 are given (the reference model's values); the ablation
chart-learned-mu0 learns them with k.  Schloegl models use the activity form of the rates (mass action in the activity),
the phosphorylation cycle and the MAPK cascade the split form.

Baselines (inputs: the hyperedge type and the z-scored conditions; the phosphorylation cycle adds the potential):
  crnn_mass_action  CRNN with the chemostats as exact mass-action factors and a log rate table per hyperedge type
  crnn_features     CRNN whose forward / backward rate constants are an MLP of the hyperedge and condition features
  flux_gnn          Flux-GNN (neuralchart.baselines.FluxGNN)
  ude               UDE closure: a free flux on every hyperedge, integrated with error control only
  fno               FNO next-step map on the grid;  bsms: BSMS-GNN next-step map on the grid sites
  generic_fno       GENERIC-FNO next-step map with the conditions as extra input channels
  persistence       u(t) = u(0)
"""
import contextlib
import math

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as Fn

from neuralchart import fast
from neuralchart import chart
from neuralchart.model import FreeEnergy, MLP, Context, featurizer as base_featurizer
from neuralchart.fit import set_to_rates
from neuralchart.graphs import grid_edge_list
from . import config as C
from . import truth as TR


@contextlib.contextmanager
def default_dtype(dt):
    old = torch.get_default_dtype()
    torch.set_default_dtype(dt)
    try:
        yield
    finally:
        torch.set_default_dtype(old)


# ------------------------------------------------------------------------------------------ the Neural Chart
class TrueActivity(FreeEnergy):
    """mu = log x + mu0 + ln gamma(x) with the reference model's crowding ln gamma (oracle only)"""
    local_excess = True

    @property
    def c_dependent(s):
        return True

    def correction(s, c, ctx):
        return TR.lng_t(c, TR.PHI)

    def value(s, c, ctx):
        G = (-c - torch.log1p(-TR.PHI * c) / TR.PHI).sum(-1)               # int_0^x ln gamma
        return super().value(c, ctx) + G


def chem_featurizer(hg):
    """drive = dict(logab (B, k) log concentrations of the first k chemostats; the others are at 1) -> features with
    chem_logc and the hyperedge type as the only mobility feature"""
    base, nC = base_featurizer(hg), len(hg.meta['chemostats'])

    def feat(F):
        lab = F['logab']
        if lab.shape[1] < nC:
            lab = torch.cat([lab, torch.zeros(lab.shape[0], nC - lab.shape[1], dtype=lab.dtype, device=lab.device)], -1)
        return base(dict(chem_logc=lab))
    feat.n_edge_sym = base.n_edge_sym
    return feat


def pc_featurizer(hg):
    """phosphorylation cycle: drive dict(U (B, V), dG (B,)) -> potential U on X and Xp (0 on E), chemostats ATP, ADP at
    log c = +-dG / 2 (ATP ADP = 1), the hyperedge type as the only mobility feature"""
    base = base_featurizer(hg)

    def feat(F):
        U, dG = F['U'], F['dG'].reshape(-1)
        return base(dict(U=torch.cat([U, U, torch.zeros_like(U)], 1), chem_logc=torch.stack([0.5 * dG, -0.5 * dG], 1)))
    feat.n_edge_sym = base.n_edge_sym
    return feat


def _base(hg, cls, **kw):
    with default_dtype(torch.float64):
        m = cls.from_parts(hg, k_ref=1.0, T_exponent=0.0, harmonic=False, step=dict(C.STEP), learn_mu0=True,
                           learn_chem_mu0=True, kinetics_table=True, **kw)
    m = m.double()
    m.step = dict(C.STEP)
    return m


def base_model(n, activity, cls=None):
    """the Schloegl class: activity-form rates, optional learned activity correction (zero gate at the start)"""
    hg = TR.hg_of(n)
    m = _base(hg, cls or chart.NeuralChart, featurizer=chem_featurizer(hg), kinetics_split=False,
              activity_correction=activity == 'learned', activity_anchor=True)
    m.free_energy.local_excess = True                    # neuralchart.fast: the excess potential is site-local
    return m


def base_model_pc(n, cls=None):
    hg = TR.hg_of(n)
    return _base(hg, cls or chart.NeuralChart, featurizer=pc_featurizer(hg), kinetics_split=True)


def base_model_mapk(n, cls=None):
    hg = TR.hg_of(n)
    return _base(hg, cls or chart.NeuralChart, hidden=32, featurizer=TR.mapk_featurizer(hg, False))


def _pc_ref_state(hg):
    g = torch.Generator().manual_seed(3)
    c = 0.3 + torch.rand(1, hg.M, generator=g, dtype=torch.float64)
    drive = dict(U=0.5 * torch.randn(1, hg.V, generator=g, dtype=torch.float64), dG=torch.tensor([1.7], dtype=torch.float64))
    return c, drive


@torch.no_grad()
def set_true_mass_action(model, kf, kb, D, logab=None):
    """set the linear parameters of a model to an ideal mass-action reference (exact when the class contains it)"""
    hg = model.hg
    g = torch.Generator().manual_seed(1)
    c = torch.rand(1, hg.M, generator=g, dtype=torch.float64) + 0.5
    logab = torch.tensor([[0.3, -0.2]], dtype=torch.float64) if logab is None else logab
    a, b = TR.mass_action_rates(hg, logab, c, kf, kb, D)
    return set_to_rates(model, c, dict(logab=logab), a[0], b[0])


@torch.no_grad()
def set_true(m):
    """the reference model's constants in the class (exact for every condition; the activity part is set separately)"""
    if C.SYSTEM == 'mapk_cascade':
        g = torch.Generator().manual_seed(3)
        c = 0.05 + torch.rand(1, m.hg.M, generator=g, dtype=torch.float64)
        drive = dict(logc=torch.tensor([[0.4, -0.1, 0.2]], dtype=torch.float64))
        a, b = TR.true_rates(m.hg, drive, c)
        err = set_to_rates(m, c, drive, a[0], b[0])
    elif C.SYSTEM == 'phosphorylation_cycle':
        c, drive = _pc_ref_state(m.hg)
        a, b = TR.true_rates(m.hg, drive, c)
        err = set_to_rates(m, c, drive, a[0], b[0])
    else:
        err = set_true_mass_action(m, TR.KF, TR.KB, TR.DD)
    assert err < 1e-9, err
    return m


_TRUE_LOGK = {}


def true_log_k():
    """log k per hyperedge type of the reference model, in the gauge of the given standard potentials"""
    if C.SYSTEM not in _TRUE_LOGK:
        m = set_true(base_model_mapk(4) if C.SYSTEM == 'mapk_cascade' else base_model_pc(4) if C.SYSTEM == 'phosphorylation_cycle'
                     else base_model(4, 'ideal'))
        _TRUE_LOGK[C.SYSTEM] = m.kinetics.core.detach().cpu().tolist()
    return _TRUE_LOGK[C.SYSTEM]


def build_chart(cfg, n=None, norm=None):
    """the Neural Chart of a ChartConfig: log k = log omega over the hyperedge types; mu0 given (the reference values,
    frozen) or learned; activity ideal, learned (ln gamma_theta with a zero gate) or true (oracle)"""
    n = C.N_TRAIN if n is None else n
    assert cfg.task in chart.REACTION_TASKS and cfg.kappa in ('learned', 'true') and not cfg.descriptors and not cfg.state
    if cfg.task != C.SYSTEM:
        C.configure(cfg.task)
    if cfg.task == 'mapk_cascade':
        m = base_model_mapk(n)
    elif cfg.task == 'phosphorylation_cycle':
        m = base_model_pc(n)
    else:
        m = base_model(n, 'learned' if cfg.activity == 'learned' else 'ideal')
    if cfg.kappa == 'true' or cfg.mu0 == 'given':
        set_true(m)
    if cfg.kappa != 'true' and cfg.mu0 == 'given':
        with torch.no_grad():
            m.kinetics.core.zero_()
        m.free_energy.mu0.requires_grad_(False); m.drive.chem_mu0.requires_grad_(False)
        m.linear_parameters = lambda: [m.kinetics.core]
    if cfg.activity == 'true':
        m.free_energy.__class__ = TrueActivity
    if cfg.drive == 'off':
        m.knobs['drive_scale'] = 0.0
    m.drive.s_A.requires_grad_(False)                   # no external force field (s_A multiplies nothing)
    m.config = cfg
    return m


# ------------------------------------------------------------------------------------------ baseline inputs
def cond_z(drive, norm):
    """the z-scored conditions of the black boxes: (log a, log b) on the Schloegl models, dG on the phosphorylation
    cycle, log ATP on the MAPK cascade; (B, k)"""
    if C.SYSTEM == 'mapk_cascade':
        lc = drive['logc']
        return ((lc[:, :1] - norm['logATP_mean']) / norm['logATP_std'])
    if C.SYSTEM == 'phosphorylation_cycle':
        return ((drive['dG'].reshape(-1, 1) - norm['dG_mean']) / norm['dG_std'])
    lab = drive['logab']
    return (lab - torch.tensor(norm['lab_mean']).to(lab)) / torch.tensor(norm['lab_std']).to(lab)


def n_cond():
    return dict(schlogl_nonideal=2, schlogl=2, phosphorylation_cycle=1, mapk_cascade=1)[C.SYSTEM]


def edge_potential(hg, drive):
    """phosphorylation cycle: the potential difference U_tail - U_head along every hyperedge (0 on reactions) and the
    mean U, (B, H) each"""
    U = drive['U']
    Uf = torch.cat([U, U, torch.zeros_like(U)], 1)
    up, um = hg.dot(Uf, '+'), hg.dot(Uf, '-')
    odd = hg.on(U.device, U.dtype)['odd']
    return odd * (up - um), odd * 0.5 * (up + um)


def bb_featurizer(hg, norm):
    """black-box inputs: hyperedge type one-hot and the z-scored conditions; the phosphorylation cycle also the
    potential (node feature, and the oriented difference along transport edges)"""
    ntyp = int(hg.meta['typ'].max()) + 1
    pc = C.SYSTEM == 'phosphorylation_cycle'

    def feat(F):
        z = cond_z(F, norm)
        B, dev, dt = z.shape[0], z.device, z.dtype
        t = hg.on(dev, dt)
        cols = [Fn.one_hot(t['typ'], ntyp).to(dt).expand(B, hg.H, ntyp), z[:, None, :].expand(B, hg.H, z.shape[1])]
        node = {f'z{i}': z[:, i:i + 1].expand(B, hg.V) for i in range(z.shape[1])}
        ori = torch.zeros(B, hg.H, dtype=dt, device=dev)
        if pc:
            dU, Um = edge_potential(hg, F)
            cols += [dU.abs()[..., None], Um[..., None]]
            node['U'] = F['U']
            ori = dU
        sym = torch.cat(cols, -1)
        return dict(U=torch.zeros(B, hg.V, dtype=dt, device=dev), node=node, edge_sym=sym, edge_ori=ori, force=None, raw=F, z=z)
    feat.n_edge_sym = ntyp + n_cond() + (2 if pc else 0)
    return feat


def mass_action_featurizer(hg):
    """CRNN (mass action) inputs: the chemostats as exact mass-action factors and the hyperedge type; on the
    phosphorylation cycle also the oriented potential difference"""
    if C.SYSTEM == 'mapk_cascade':
        return TR.mapk_featurizer(hg, False)
    if C.SYSTEM != 'phosphorylation_cycle':
        return chem_featurizer(hg)
    base = base_featurizer(hg)

    def feat(F):
        dG = F['dG'].reshape(-1)
        f = base(dict(chem_logc=torch.stack([0.5 * dG, -0.5 * dG], 1)))
        f['dU'], _ = edge_potential(hg, F)
        return f
    feat.n_edge_sym = base.n_edge_sym
    return feat


# ------------------------------------------------------------------------------------------ baselines
def make_crnn(n, norm, mass_action):
    from neuralchart.baselines import CRNN

    class PotentialCRNN(CRNN):
        """CRNN; on the phosphorylation cycle with mass action: log k+- = table(type) + w+- dU (the oriented potential
        difference), w+- start at 0"""
        def __init__(s, *a, **k):
            super().__init__(*a, **k)
            if C.SYSTEM == 'phosphorylation_cycle' and s.chem:
                s.kdu = nn.Parameter(torch.zeros(2))

        def prepare(s, drive):
            ctx = super().prepare(drive)
            if getattr(s, 'kdu', None) is not None:
                dU = ctx['feats']['dU']
                ctx['kf'] = ctx['kf'] * torch.exp(s.kdu[0].to(dU.dtype) * dU)
                ctx['kb'] = ctx['kb'] * torch.exp(s.kdu[1].to(dU.dtype) * dU)
            return ctx
    hg = TR.hg_of(n)
    f = mass_action_featurizer(hg) if mass_action else bb_featurizer(hg, norm)
    # on the phosphorylation cycle the condition-feature CRNN also sees the oriented potential difference (as Flux-GNN)
    ori = (not mass_action) and C.SYSTEM == 'phosphorylation_cycle'
    with default_dtype(torch.float64):
        m = PotentialCRNN(hg, f.n_edge_sym + (1 if ori else 0), f, oriented=ori, chem=mass_action, table=mass_action,
                          step=C.STEP)
    m.step = C.STEP                                      # shared: the substep cap is set for training and evaluation
    return m.double()


def make_flux_gnn(n, norm):
    from neuralchart.baselines import FluxGNN
    from .data import load_set
    hg = TR.hg_of(n)
    f = bb_featurizer(hg, norm)
    ex = {k: v[:1].cpu() for k, v in load_set('train')['drive'].items()}
    if 'U' in ex:                                       # an example drive on this grid (only the shapes matter)
        ex['U'] = torch.zeros(1, hg.V, dtype=ex['U'].dtype)
    fx = f(ex)
    nn_in = 1 + len(fx['node']) + (hg.n_species if hg.n_species > 1 else 0)
    ne_in = fx['edge_sym'].shape[-1] + 1
    return FluxGNN(hg, nn_in, ne_in, f, hidden=64, layers=3, rho_scale=norm['c_scale'], step=dict(C.STEP))


class UDE(nn.Module):
    """UDE closure: the same hypergraph and conservative assembly dc/dt = S J with a free flux on every hyperedge,
    J_h = g(type, u_p, u_q, z) for a reaction at site p = q, J_e = g(type, u_i, u_j, z) - g(type, u_j, u_i, z) for
    transport (u = all state species at a site / c_scale; the phosphorylation cycle adds the potential at both ends);
    g an MLP (64 hidden) with a zero last layer.  No exponential-family rates and no local detailed balance.  It has no
    positivity structure, so it is integrated with error control only (neuralchart.fast control 'err_only')."""
    name = 'ude'
    free_flux = True

    def __init__(s, n, norm, hidden=64):
        super().__init__()
        s.norm = norm
        s.ntyp = int(TR.hg_of(n).meta['typ'].max()) + 1
        s.NS = TR.n_channels()
        s.extra = 1 if C.SYSTEM == 'phosphorylation_cycle' else 0
        s.net = MLP(s.ntyp + 2 * (s.NS + s.extra) + n_cond(), hidden, 1, last='zero')
        s.step = dict(C.STEP)
        s.set_n(n)

    def set_n(s, n):
        hg = TR.hg_of(n)
        s.hg = hg
        V = hg.V
        site = torch.as_tensor(np.arange(hg.H) % V)
        tr = torch.as_tensor(hg.meta['odd'])
        P = torch.where(tr, torch.as_tensor(hg.meta['et']), site); Q = torch.where(tr, torch.as_tensor(hg.meta['eh']), site)
        s._P, s._Q, s._tr, s._typ = P, Q, tr, torch.as_tensor(hg.meta['typ'])
        return s

    def prepare(s, drive):
        z = cond_z(drive, s.norm)
        dev = z.device
        ctx = Context(z=z, P=s._P.to(dev), Q=s._Q.to(dev), tr=s._tr.to(dev).to(z.dtype),
                      oh=Fn.one_hot(s._typ.to(dev), s.ntyp).to(z.dtype))
        if s.extra:
            ctx['U'] = drive['U']
        return ctx

    def context(s, drive):
        return drive if isinstance(drive, Context) else s.prepare(drive)

    def flux(s, c, drive):
        ctx = s.context(drive)
        B = c.shape[0]
        V = s.hg.V
        u = (c / s.norm['c_scale']).reshape(B, s.NS, V).transpose(1, 2)               # (B, V, NS)
        if s.extra:
            u = torch.cat([u, ctx['U'][..., None].to(u.dtype)], -1)
        H = ctx['P'].shape[0]
        p, q = u[:, ctx['P']], u[:, ctx['Q']]
        oh = ctx['oh'][None].expand(B, H, s.ntyp); z = ctx['z'][:, None, :].expand(B, H, ctx['z'].shape[1])
        w = s.net.net[0].weight.dtype
        g = lambda x, y: s.net(torch.cat([oh, x, y, z], -1).to(w)).squeeze(-1).to(c.dtype)
        return g(p, q) - ctx['tr'][None].to(c.dtype) * g(q, p)

    def rates(s, c, drive):
        J = s.flux(c, drive)
        a = J.clamp_min(0)
        return a, a - J                     # one-way parts with a - b = J exactly

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
        B, Cn, Hh, W = x.shape
        xf = torch.fft.rfft2(x)
        out = torch.zeros(B, s.w1.shape[1], Hh, W // 2 + 1, dtype=torch.cfloat, device=x.device)
        m1, m2 = min(s.m1, Hh // 2), min(s.m2, W // 2 + 1)
        out[:, :, :m1, :m2] = torch.einsum('bixy,ioxy->boxy', xf[:, :, :m1, :m2], s.w1[:, :, :m1, :m2])
        out[:, :, -m1:, :m2] = torch.einsum('bixy,ioxy->boxy', xf[:, :, -m1:, :m2], s.w2[:, :, :m1, :m2])
        return torch.fft.irfft2(out, s=(Hh, W))


class FNO(nn.Module):
    """standard FNO (Li et al. 2021; width 32, 4 layers, all modes of the training grid): residual next-step map on the
    state channels with the z-scored conditions (phosphorylation cycle: and U) as extra input channels"""
    name = 'fno'

    def __init__(s, norm, width=32, layers=4):
        super().__init__()
        s.norm, s.NS = norm, TR.n_channels()
        modes = C.N_TRAIN // 2
        cin = s.NS + n_cond() + (1 if C.SYSTEM == 'phosphorylation_cycle' else 0)
        s.lift = nn.Conv2d(cin, width, 1)
        s.spec = nn.ModuleList([SpectralConv2d(width, width, modes, modes + 1) for _ in range(layers)])
        s.w = nn.ModuleList([nn.Conv2d(width, width, 1) for _ in range(layers)])
        s.p1, s.p2 = nn.Conv2d(width, 128, 1), nn.Conv2d(128, s.NS, 1)

    def net(s, x):
        x = s.lift(x)
        for i, (sp_, w) in enumerate(zip(s.spec, s.w)):
            x = sp_(x) + w(x)
            if i < len(s.spec) - 1:
                x = Fn.gelu(x)
        return s.p2(Fn.gelu(s.p1(x)))

    def prepare(s, drive):
        ctx = Context(z=cond_z(drive, s.norm).float())
        if C.SYSTEM == 'phosphorylation_cycle':
            ctx['U'] = drive['U'].float()
        return ctx

    def context(s, drive):
        return drive if isinstance(drive, Context) else s.prepare(drive)

    def advance(s, c, drive, Dt, **kw):
        ctx = s.context(drive)
        B = c.shape[0]
        n = int(round((c.shape[1] // s.NS) ** 0.5))
        u = c.reshape(B, s.NS, n, n).float() / s.norm['c_scale']
        cols = [u, ctx['z'][:, :, None, None].expand(B, ctx['z'].shape[1], n, n)]
        if 'U' in ctx:
            cols.append(ctx['U'].reshape(B, 1, n, n))
        return ((u + s.net(torch.cat(cols, 1))) * s.norm['c_scale']).reshape(B, -1).to(c.dtype), 1


class BSMS(nn.Module):
    """BSMS-GNN (neuralchart.baselines.bsms_gnn) on the grid sites: node inputs the normalised state channels and the
    conditions, edge inputs the relative position; predicts the normalised one-step change"""
    name = 'bsms'

    def __init__(s, n, norm, latent=128):
        super().__init__()
        from neuralchart.baselines.bsms_gnn import BSMSGNN
        s.norm, s.NS = norm, TR.n_channels()
        s.levels = 3 if C.N_TRAIN >= 8 else 2
        s.ncond = n_cond() + (1 if C.SYSTEM == 'phosphorylation_cycle' else 0)
        s.net = BSMSGNN(s.NS + s.ncond, 3, s.levels, latent, n_out=s.NS)
        for k in ('c_mean', 'c_std', 'd_mean', 'd_std'):
            s.register_buffer(k, torch.zeros(s.NS))
        s.register_buffer('noise', torch.zeros(()))
        s.set_n(n)

    def set_n(s, n):
        from neuralchart.baselines.bsms_gnn import bistride_hierarchy
        s.n = n
        E = grid_edge_list(n)
        ij = np.stack(np.meshgrid(np.arange(n), np.arange(n), indexing='ij'), -1).reshape(-1, 2)
        s._hier = bistride_hierarchy(E, ij[:, ::-1] / n, s.levels)
        s._H = None
        return s

    def prepare(s, drive):
        from neuralchart.baselines.bsms_gnn import hierarchy_tensors
        z = cond_z(drive, s.norm).float()
        dev = z.device
        if s._H is None or s._H['snd'][0].device != dev:
            s._H = hierarchy_tensors(s._hier, dev, s.n)
        B = z.shape[0]
        V = s.n * s.n
        cnd = z[:, None, :].expand(B, V, z.shape[1])
        if C.SYSTEM == 'phosphorylation_cycle':
            cnd = torch.cat([cnd, drive['U'].float()[..., None]], -1)
        return Context(cond=cnd, H=s._H, xe=s._H['geo'][0][None].expand(B, -1, -1))

    def context(s, drive):
        return drive if isinstance(drive, Context) else s.prepare(drive)

    def nodes(s, c):
        return c.reshape(c.shape[0], s.NS, -1).transpose(1, 2).float()

    def delta(s, u, ctx):
        return s.net(torch.cat([(u - s.c_mean) / s.c_std, ctx['cond']], -1), ctx['xe'], ctx['H'])

    def advance(s, c, drive, Dt, **kw):
        ctx = s.context(drive)
        u = s.nodes(c)
        u = u + s.delta(u, ctx) * s.d_std + s.d_mean
        return u.transpose(1, 2).reshape(c.shape[0], -1).to(c.dtype), 1


class GenericFNOOp(nn.Module):
    """GENERIC-FNO (neuralchart.baselines.generic_fno) as a next-step map: every species field scaled by c_scale, the
    z-scored conditions (phosphorylation cycle: and U) as constant input channels; resolution independent"""
    name = 'generic_fno'

    def __init__(s, norm):
        super().__init__()
        from neuralchart.baselines.generic_fno import GenericFNO
        s.norm = norm
        s.NS = TR.n_channels()
        s.NC = n_cond() + (1 if C.SYSTEM == 'phosphorylation_cycle' else 0)
        s.net = GenericFNO(n_fields=s.NS, n_cond=s.NC)

    def cond_fields(s, z, U, n):
        Bn = z.shape[0]
        cols = [z[:, :, None, None].expand(Bn, z.shape[1], n, n)]
        if U is not None:
            cols.append(U.reshape(Bn, 1, n, n).float())
        return torch.cat(cols, 1).contiguous()

    def prepare(s, drive):
        ctx = Context(z=cond_z(drive, s.norm).float())
        if C.SYSTEM == 'phosphorylation_cycle':
            ctx['U'] = drive['U'].float()
        return ctx

    def context(s, drive):
        return drive if isinstance(drive, Context) else s.prepare(drive)

    def advance(s, c, drive, Dt, **kw):
        ctx = s.context(drive)
        Bn = c.shape[0]
        n = int(round((c.shape[1] // s.NS) ** 0.5))
        u = c.reshape(Bn, s.NS, n, n).float() / s.norm['c_scale']
        cf = s.cond_fields(ctx['z'], ctx.get('U'), n)
        with torch.enable_grad():                       # dE/du, dS/du by autodiff also at inference
            v = s.net(u.detach(), cf, create_graph=False).detach()
        bad = ~torch.isfinite(v).flatten(1).all(1) | (v.flatten(1).abs().amax(1) > 1e4)
        v = torch.where(bad[:, None, None, None], torch.full_like(v, float('nan')), v)
        return (v * s.norm['c_scale']).reshape(Bn, -1).to(c.dtype), 1


class Persistence(nn.Module):
    name = 'persistence'

    def prepare(s, drive):
        return Context()

    def advance(s, c, drive, Dt, **kw):
        return c, 1


FLOAT_MODELS = ('fno', 'bsms', 'flux_gnn', 'ude', 'generic_fno')          # float32 parameters; the others float64
BASELINES = ('crnn_mass_action', 'crnn_features', 'flux_gnn', 'ude', 'fno', 'bsms', 'generic_fno', 'persistence')


def build_baseline(name, n, norm):
    if name == 'crnn_mass_action':
        return make_crnn(n, norm, True)
    if name == 'crnn_features':
        return make_crnn(n, norm, False)
    if name == 'flux_gnn':
        return make_flux_gnn(n, norm)
    if name == 'ude':
        return UDE(n, norm)
    if name == 'fno':
        return FNO(norm)
    if name == 'bsms':
        return BSMS(n, norm)
    if name == 'generic_fno':
        return GenericFNOOp(norm)
    if name == 'persistence':
        return Persistence()
    raise ValueError(name)


def build(name, n, norm):
    if chart.is_chart(C.SYSTEM, name):
        return build_chart(chart.chart_config(C.SYSTEM, name), n, norm)
    return build_baseline(name, n, norm)


GRAPH_BUFFERS = ('p_h', 'p_m', 'p_v', 'm_h', 'm_m', 'm_v')


def next_dtype(model):
    p = next(iter(model.parameters()), None)
    return p.dtype if p is not None and not p.is_complex() else torch.float64


def rebind(name, model, n, norm, device):
    """the same trained parameters on another grid (no retraining)"""
    if name in ('fno', 'persistence', 'generic_fno'):
        return model
    if name in ('ude', 'bsms'):
        return model.set_n(n)
    new = build(name, n, norm).to(device).to(next_dtype(model))
    gb = lambda k: k.split('.')[-1] in GRAPH_BUFFERS
    missing, unexpected = new.load_state_dict({k: v for k, v in model.state_dict().items() if not gb(k)}, strict=False)
    assert not unexpected and all(gb(k) for k in missing), (missing, unexpected)
    return new.eval()


def accelerate(model, name, prec=None):
    """the integrator of the run (C.FAST) for every model with positive rates (and the UDE closure)"""
    if C.FAST is not None and fast.supports(model):
        fast.restore(model)
        kw = dict(C.FAST)
        if prec:
            kw['precision'] = prec
        if name == 'ude':
            kw['control'] = 'err_only'
        fast.accelerate(model, step=C.STEP, **kw)
    return model


def set_max_sub(model, k):
    C.STEP['max_sub'] = k
    for mod in model.modules() if hasattr(model, 'modules') else []:
        st = getattr(mod, 'step', None)
        if isinstance(st, dict) and 'max_sub' in st:
            st['max_sub'] = k
