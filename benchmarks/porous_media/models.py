"""Models of the porous-media benchmark: the Neural Chart and the baselines on any discretisation.

Neural Chart       EdgeMobility inside neuralchart.MasterEquation; the force (U / T, the drive) is fixed by physics
UDE closure        the same edges and descriptors with a free antisymmetric edge flux
Flux-GNN           neuralchart.baselines.FluxGNN with the black-box inputs below
FNO                residual next-step map on n x n grids
BSMS-GNN           neuralchart.baselines.bsms_gnn on the cells of any discretisation
Persistence        u(t) = u(0)
GENERIC-FNO and HelmFluid have their own training loops (benchmarks/porous_media/train.py).
Black-box node inputs (node_cond): U and amp * psi (z-scored with training-set scalars), T, amp, 6 local descriptors.
"""
import math
import os

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as Fn

from neuralchart.model import MasterEquation, FreeEnergy, Drive, MLP, Context
from neuralchart.baselines import FluxGNN
from neuralchart.baselines.bsms_gnn import MeshGraphNets, BSMSGNN, bistride_hierarchy, hierarchy_tensors
from .data import D, DT, N, KREF, line_integrals, edge_inputs

STEP = dict(dt_max=DT / 8, theta0=0.9, theta1=1.0, max_sub=20000, t_tol=1e-9)


# ------------------------------------------------------------------------------------------ the Neural Chart
class EdgeMobility(nn.Module):
    """the mobility of the Neural Chart on porous media: k_e = D T w_e exp(kappa_e), w_e the geometric weight.
    mode 'material_state': kappa = log omega + g(xi_e) + s m(log rhobar_e) (the Neural Chart);
         'material':       kappa = log omega + g(xi_e) (no state input);
         'constant':       kappa = log omega (constant mobility);
         'true':           kappa = log of the true edge mobility (feats['mtrue']);
         'true_crowding':  the true edge mobility times the true crowding factor 1 / (1 + rhobar_e / rho_s).
    log omega is the parameter `core`; g is an MLP of the 12 edge descriptors (pointwise, no message passing) whose last
    layer starts at zero; s m is the state term (s starts at zero)."""
    rho_dependent = False
    split = True
    chem_prefactor = False

    def __init__(s, mode='material_state', hidden=64, rho_s=None):
        super().__init__()
        s.mode = mode
        s.core = nn.Parameter(torch.zeros(()))
        if mode == 'true_crowding':
            # the true crowding factor on top of the true material mobility, as a fixed (parameter-free) state
            # term: log k += -log(1 + rhobar_e / rho_s), symmetric in the two cells (local detailed balance exact)
            s.rho_dependent = True
            s.rho_s = float(rho_s)
        if mode in ('material', 'material_state'):
            s.net = MLP(12, hidden, 1, last='zero')
        if mode == 'material_state':
            # state branch: kappa += s_rho * m(log rhobar_e), m a monotone (non-decreasing) MLP with positive weights, s_rho a
            # learned scale that starts at zero (the model starts exactly at the material-only law); symmetric in the two
            # endpoints, so local detailed balance and sigma >= 0 stay exact
            s.rho_dependent = True
            s.m_w = nn.ParameterList([nn.Parameter(0.5 * torch.randn(1, 16)), nn.Parameter(0.5 * torch.randn(16, 16) / 4), nn.Parameter(0.5 * torch.randn(16, 1) / 4)])
            s.m_b = nn.ParameterList([nn.Parameter(torch.zeros(16)), nn.Parameter(torch.zeros(16)), nn.Parameter(torch.zeros(1))])
            s.m_a = nn.Parameter(torch.zeros(()))         # linear part: softplus(m_a) * u, non-negative slope
            s.s_rho = nn.Parameter(torch.zeros(()))

    RHO_FLOOR = 1e-3

    def state_branch(s, logr):
        """s * (m(u) - m(u_floor)), u = log rhobar; m(u) = softplus(a) u' + m_mono(u'), u' = u / 3, with m_mono a
        positive-weight MLP with softplus activations: non-decreasing, and linear (not saturated) in both tails"""
        def m(z):
            u = z / 3.0
            h = u[..., None]
            for i, (w, b) in enumerate(zip(s.m_w, s.m_b)):
                h = h @ Fn.softplus(w).to(h.dtype) + b.to(h.dtype)
                if i < 2:
                    h = Fn.softplus(h)
            return Fn.softplus(s.m_a).to(u.dtype) * u + h[..., 0]
        if s.mode == 'true_crowding':
            return -torch.log1p(torch.exp(logr) / s.rho_s)
        return s.s_rho.to(logr.dtype) * (m(logr) - m(torch.full_like(logr[..., :1], math.log(s.RHO_FLOOR))))

    def rhobar_log(s, feats, c):
        rho = c * feats['invV'][None]
        if s.mode == 'true_crowding':                      # exactly the truth's rhobar (no floor)
            return torch.log(0.5 * (rho[:, feats['tail']] + rho[:, feats['head']]))
        return torch.log(torch.clamp(0.5 * (rho[:, feats['tail']] + rho[:, feats['head']]), min=s.RHO_FLOOR))

    def kappa(s, feats):
        x = feats['ein']
        if s.mode in ('true', 'true_crowding'):
            return torch.log(feats['mtrue'])
        k = s.core.to(x.dtype).expand(x.shape[0], x.shape[1])
        if s.mode in ('material', 'material_state'):
            k = k + s.net(x).squeeze(-1)
        return k

    def forward(s, feats, hg=None, c=None):
        if s.mode in ('material_state', 'true_crowding'):
            if 'kap_mat' not in feats:                   # the material part once per context, the state part per call
                feats['kap_mat'] = s.kappa(feats)
            kap = feats['kap_mat']
        else:
            kap = s.kappa(feats)
        if s.mode in ('material_state', 'true_crowding') and c is not None:
            kap = kap + _ckpt(lambda c_: s.state_branch(s.rhobar_log(feats, c_)), c)     # checkpointed: memory only
        return D * feats['w'][None] * feats['T'][:, None] * torch.exp(kap)


def chart_featurizer(disc):
    """the inputs of the Neural Chart on a discretisation: potential U / T - log(V_i / Vbar), the drive line integrals,
    the edge descriptors and the geometric weights"""
    w, logV, E = disc.g['w'], disc.g['logV'], disc.g['edges']

    def feat(F):
        U = F['U']; B = U.shape[0]
        T = F['T'].to(U.dtype)
        f = dict(U=U - T[:, None] * logV.to(U.device, U.dtype)[None], T=T, line=F['amp'][:, None] * F['line'], chem_logc=None, chem_mu0=None,
                 edge_sym=torch.zeros(B, disc.H, 1, dtype=U.dtype, device=U.device), edge_ori=torch.zeros(B, disc.H, dtype=U.dtype, device=U.device),
                 node={}, ein=F['ein'], pmed=F['pmed'], w=w.to(U.device, U.dtype), raw=F,
                 tail=E[:, 0].to(U.device), head=E[:, 1].to(U.device), invV=torch.exp(-logV).to(U.device, U.dtype))
        if 'mtrue' in F:
            f['mtrue'] = F['mtrue']
        return f
    feat.n_edge_sym = 1
    return feat


def build_master(mode, disc, cls=None, **kin_kw):
    """the master equation on a discretisation with an EdgeMobility of the given mode"""
    hg = disc.hg()
    return (cls or MasterEquation)(hg, FreeEnergy(hg), EdgeMobility(mode, **kin_kw), Drive(hg, harmonic=False),
                                   chart_featurizer(disc), step=dict(STEP))


def build_chart(cfg, meta=None, disc=None):
    """the Neural Chart of a ChartConfig on porous media: k = D T w_e omega exp(g(xi_e) + s m(log rhobar_e)) with one
    hyperedge class (transport).  With crowding in the data, the oracle uses the true crowding factor as well."""
    from neuralchart.chart import NeuralChart
    assert cfg.task == 'porous_media'
    if cfg.kappa == 'learned':
        mode = 'material_state' if cfg.state else 'material' if cfg.descriptors else 'constant'
    else:
        crowd = bool(meta is not None and meta.get('crowd'))
        mode = dict(true='true_crowding' if crowd else 'true', true_material='true')[cfg.kappa]
    m = build_master(mode, disc, cls=NeuralChart, **(dict(rho_s=meta['RHO_S']) if mode == 'true_crowding' else {}))
    if cfg.drive != 'learned_scale':
        m.drive.s_A.requires_grad_(False)          # the drive is known physics: A = amp int F.dl / T exactly
    m.config = cfg
    return m


# ------------------------------------------------------------------------------------------ baselines
def node_cond(F, norm):
    """node features (B, V, 10): U, amp * psi (z-scored with training scalars), T, amp, 6 microstructure descriptors"""
    U = F['U']; B, V = U.shape
    one = torch.ones_like(U)
    return torch.cat([((U - norm['U'][0]) / norm['U'][1])[..., None], ((F['amp'][:, None] * F['psin'] - norm['psi'][0]) / norm['psi'][1])[..., None],
                      (F['T'][:, None] * one)[..., None], (F['amp'][:, None] * one)[..., None], F['ndesc'].transpose(1, 2)], -1)


NC = 10


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


class FNO2d(nn.Module):
    """standard FNO (Li et al. 2021): 1x1 lift, 4 x [spectral conv + 1x1 conv, GELU except after the last], 1x1
    projection to 128 and to 1."""
    def __init__(s, cin, width=32, modes=12, layers=4):
        super().__init__()
        s.lift = nn.Conv2d(cin, width, 1)
        s.spec = nn.ModuleList([SpectralConv2d(width, width, modes, modes) for _ in range(layers)])
        s.w = nn.ModuleList([nn.Conv2d(width, width, 1) for _ in range(layers)])
        s.p1, s.p2 = nn.Conv2d(width, 128, 1), nn.Conv2d(128, 1, 1)

    def forward(s, x):
        x = s.lift(x)
        for i, (sp, w) in enumerate(zip(s.spec, s.w)):
            x = sp(x) + w(x)
            if i < len(s.spec) - 1:
                x = Fn.gelu(x)
        return s.p2(Fn.gelu(s.p1(x)))[:, 0]


class FNOOp(nn.Module):
    """residual next-step map rho_{t+DT} = rho_t + FNO([rho_t, cond]) on an n x n grid (any n)"""
    name = 'fno'

    def __init__(s, norm):
        super().__init__()
        s.norm = norm
        s.net = FNO2d(1 + NC)

    def prepare(s, drive):
        cond = node_cond(drive, s.norm)
        n = int(round(cond.shape[1] ** 0.5))
        return Context(cond=cond.transpose(1, 2).reshape(cond.shape[0], NC, n, n))

    def context(s, drive):
        return drive if isinstance(drive, Context) else s.prepare(drive)

    def advance(s, c, drive, Dt, **kw):
        ctx = s.context(drive)
        n = ctx['cond'].shape[-1]
        r = c.reshape(-1, n, n)
        return (r + s.net(torch.cat([r[:, None], ctx['cond']], 1))).reshape(c.shape), 1


class BSMSOp(nn.Module):
    """BSMS-GNN (neuralchart.baselines.bsms_gnn; arch='mgn' gives MeshGraphNets) on the cells of any discretisation.  Node inputs: the density and the 10 condition / descriptor columns of the other black boxes;
    edge inputs (both directions): relative position x_j - x_i and its length (in units of the training grid spacing),
    and the drive along the edge.  Predicts the normalised one-step change of the density; explicit rollout."""
    name = 'bsms'

    def __init__(s, norm, latent=128, steps=15, arch='bsms', levels=4):
        super().__init__()
        s.norm, s.arch, s.levels = norm, arch, levels
        s.net = MeshGraphNets(1 + NC, 4, latent, steps) if arch == 'mgn' else BSMSGNN(1 + NC, 4, levels, latent)
        for k in ('rho_mean', 'rho_std', 'd_mean', 'd_std', 'noise'):
            s.register_buffer(k, torch.zeros(()))
        s.disc = None

    def set_disc(s, disc, device=None):
        g = disc.g
        E = g['edges']
        s._snd = torch.cat([E[:, 0], E[:, 1]]).long()
        s._rcv = torch.cat([E[:, 1], E[:, 0]]).long()
        dv = torch.cat([g['dvec'], -g['dvec']]).float() * N
        s._geo = torch.cat([dv, dv.norm(dim=-1, keepdim=True)], -1)
        s._dens = g['dens'].float()
        s.disc = disc
        if s.arch == 'bsms':
            if getattr(disc, '_bsms', None) is None:
                disc._bsms = bistride_hierarchy(E.cpu().numpy(), g['pts'].cpu().numpy(), s.levels)
            s._hier = disc._bsms
        return s

    def prepare(s, drive):
        dev = drive['U'].device
        line = drive['amp'][:, None] * drive['line'] / s.norm['line']
        B = line.shape[0]
        xe = torch.cat([s._geo.to(dev)[None].expand(B, -1, -1), torch.cat([line, -line], 1)[..., None]], -1)
        ctx = Context(cond=node_cond(drive, s.norm), xe=xe, snd=s._snd.to(dev), rcv=s._rcv.to(dev), dens=s._dens.to(dev))
        if s.arch == 'bsms':
            ctx['H'] = hierarchy_tensors(s._hier, dev, N)
        return ctx

    def context(s, drive):
        return drive if isinstance(drive, Context) else s.prepare(drive)

    def delta(s, rho, ctx):
        """normalised one-step change predicted from the density rho (B, V)"""
        xn = torch.cat([((rho - s.rho_mean) / s.rho_std)[..., None], ctx['cond']], -1)
        if s.arch == 'bsms':
            return s.net(xn, ctx['xe'], ctx['H'])[..., 0]
        return s.net(xn, ctx['xe'], ctx['snd'], ctx['rcv'])[..., 0]

    def advance(s, c, drive, Dt, **kw):
        ctx = s.context(drive)
        rho = c * ctx['dens'][None]
        rho = rho + s.delta(rho, ctx) * s.d_std + s.d_mean
        return rho / ctx['dens'][None], 1


class UDEClosure(nn.Module):
    """UDE closure: the same discretisation, conservative assembly dc/dt = S J and edge descriptors as the Neural Chart,
    but a freely learned antisymmetric edge flux
        J_e = D w_e [ g(rho_i, rho_j, U_j - U_i, L_e, T, amp, desc_e) - g(rho_j, rho_i, U_i - U_j, -L_e, T, amp, desc_e) ],
    rho the cell densities, L_e = amp int_e F.dl (normalised), g an MLP of the same size as kappa_theta, zero-initialised
    last layer.  There is no exponential form of the rates and no local detailed balance.  T and amp are features.
    The closure has no positivity structure, so it is integrated with error control only (neuralchart.fast,
    control='err_only'), and negative densities are counted as violations."""
    name = 'ude'
    free_flux = True

    def __init__(s, disc, norm, hidden=64):
        super().__init__()
        s.hg = disc.hg()
        s.norm = norm
        s.net = MLP(6 + 12, hidden, 1, last='zero')
        s.featurizer = s._featurizer(disc)
        s.step = dict(STEP)

    def _featurizer(s, disc):
        w, logV, E = disc.g['w'].float(), disc.g['logV'].float(), disc.g['edges']

        def feat(F):
            U = F['U']
            return dict(w=w.to(U.device), invV=torch.exp(-logV).to(U.device), tail=E[:, 0].to(U.device), head=E[:, 1].to(U.device),
                        dU=U[:, E[:, 1]] - U[:, E[:, 0]], L=F['amp'][:, None] * F['line'] / s.norm['line'],
                        T=F['T'][:, None].expand_as(F['line']), amp=F['amp'][:, None].expand_as(F['line']), ein=F['ein'])
        return feat

    def prepare(s, drive):
        return Context(feats=s.featurizer(drive))

    def context(s, drive):
        return drive if isinstance(drive, Context) else s.prepare(drive)

    def flux(s, c, drive):
        f = s.context(drive)['feats']
        rho = c * f['invV'][None]
        ri, rj = rho[:, f['tail']], rho[:, f['head']]
        def anti(ri, rj, du, L, T, amp, ein):
            g = lambda a, b, d_, l_: s.net(torch.cat([torch.stack([a, b, d_, l_, T, amp], -1), ein], -1)).squeeze(-1)
            return g(ri, rj, du, L) - g(rj, ri, -du, -L)
        args = (ri, rj, f['dU'], f['L'], f['T'], f['amp'], f['ein'])
        out = _ckpt(anti, *args)
        return D * f['w'][None] * out

    def rates(s, c, drive):
        J = s.flux(c, drive)
        a = J.clamp_min(0)
        return a, a - J                     # one-way parts with a - b = J exactly (also in the gradient at J = 0)

    def forward(s, c, drive):
        return s.hg.apply_S(s.flux(c, drive))

    def escape_ratio(s, c, a, b):
        return (s.hg.consumption(a, b) / (c.clamp_min(0) + 0.01)).amax(-1)

    def advance(s, c, drive, Dt, **kw):
        from neuralchart.integrators import positive_ssp_rk2
        return positive_ssp_rk2(s, c, s.context(drive), Dt, **s.step)


def _ckpt(fn, *args):
    """activation checkpointing in training (memory only)"""
    if torch.is_grad_enabled():
        from torch.utils.checkpoint import checkpoint
        return checkpoint(fn, *args, use_reentrant=False)
    return fn(*args)


def graph_featurizer(norm, disc):
    """inputs of Flux-GNN: node features of node_cond, edge length and direction, and the oriented drive"""
    dv = disc.g['dvec'].float()
    L = dv.norm(dim=-1, keepdim=True)
    sym0 = torch.cat([L * N, dv ** 2 / L ** 2], -1)                       # |d| in units of the training spacing, n_x^2, n_y^2

    def feat(F):
        cond = node_cond(F, norm)
        B = cond.shape[0]
        node = {f'f{i}': cond[..., i] for i in range(NC)}
        ori = F['amp'][:, None] * F['line'] / norm['line']
        return dict(U=F['U'], node=node, edge_sym=sym0.to(cond.device)[None].expand(B, -1, -1), edge_ori=ori, raw=F)
    feat.n_edge_sym = 3
    return feat




class FluxGNNEdges(FluxGNN):
    """Flux-GNN with activation checkpointing of every rate evaluation (memory only)"""
    def rates(s, c, drive):
        ctx = s.context(drive)
        return _ckpt(lambda c_: FluxGNN.rates(s, c_, ctx), c)


class Persistence(nn.Module):
    """u(t) = u(0)"""
    name = 'persistence'

    def prepare(s, drive):
        return Context()

    def context(s, drive):
        return drive if isinstance(drive, Context) else s.prepare(drive)

    def advance(s, c, drive, Dt, **kw):
        return c, 1


def build(model, meta, disc):
    """a model by key: a Neural Chart (neuralchart.chart.CONFIGS['porous_media']) or a baseline"""
    from neuralchart import chart
    norm = meta['norm']
    if chart.is_chart('porous_media', model):
        return build_chart(chart.chart_config('porous_media', model), meta, disc)
    if model == 'fno':
        return FNOOp(norm)
    if model == 'bsms':
        return BSMSOp(norm).set_disc(disc)
    if model == 'ude':
        return UDEClosure(disc, norm)
    if model == 'persistence':
        return Persistence()
    if model == 'flux_gnn':
        return FluxGNNEdges(disc.hg(), 1 + NC, 4, graph_featurizer(norm, disc), hidden=64, layers=3, scale=KREF, step=dict(STEP))
    raise ValueError(f'unknown model {model!r}')


GRAPH_BUFFERS = ('p_h', 'p_m', 'p_v', 'm_h', 'm_m', 'm_v')


def rebind(model_name, model, meta, disc, device):
    """the same trained model on another discretisation (no retraining): rebuild the graph-dependent parts, keep the
    parameters.  Returns None when the model cannot run there."""
    if model_name == 'fno':
        return model if disc.g['kind'] == 'grid' else None
    if model_name == 'persistence':
        return model
    if model_name == 'bsms':
        return model.set_disc(disc)
    cfg = getattr(model, 'config', None)
    new = (build_chart(cfg, meta, disc) if cfg is not None else build(model_name, meta, disc)).to(device)
    gb = lambda k: any(b in k for b in GRAPH_BUFFERS)
    missing, unexpected = new.load_state_dict({k: v for k, v in model.state_dict().items() if not gb(k)}, strict=False)
    assert not unexpected and all(gb(k) for k in missing), (missing, unexpected)
    return new.eval()
