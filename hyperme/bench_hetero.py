"""H1: driven transport in heterogeneous media, across media and across conditions.

Does the wired HyperME learn a high-dimensional, transferable operator, or only fit parameters?  The kinetic
prefactor of every edge is a nonlinear function of the local microstructure, and the model has to learn that function
from 64 random media and apply it to 16 unseen ones, while the temperature and the drive move outside the training
values.

Medium.  A binary pore map on a 128 x 128 periodic fine grid (thresholded Gaussian random field, correlation length
2 to 10 fine pixels, porosity p uniform in [0.3, 0.7]; test type (d): p in [0.15, 0.3]).  Every cell of the 32 x 32
model grid keeps its 4 x 4 sub-block (16 binary features) and its porosity.

True dynamics (a master equation of the wired form, so the truth lies in the model class; the model never sees g):
    a_e = k_e rho_t exp((phi_t - phi_h + A_e)/2),  b_e = k_e rho_h exp(-(phi_t - phi_h + A_e)/2),  drho/dt = d^T (a - b)
    phi = U / T,  A_e = amp * dpsi_e / T,  k_e = k_ref T g_e,  k_ref = D n^2 (D = 0.02, the S2 scale)
    g_e = C * eps^3 / ((1 - eps)^2 + 0.3) * tau_e + 1e-4                  (Kozeny-Carman, regularised at eps -> 1)
    eps = harmonic mean of the two cell porosities
    tau_e = 0.02 + 0.58 n_c / 4 + 0.40 n_s / 4                           (tortuosity / open-face factor)
    n_c = number of the 4 pixel rows across the shared face that are pore on both sides,
    n_s = number of rows with a straight 4-pixel pore channel through the face (2 pixels deep on each side),
    C normalises the mean of g over the training edges to 1.  Edges between nearly solid cells are almost closed.
H1' (outside the model class): crowding, k_e = k_ref T g_e / (1 + rhobar_e / rho_s), rhobar_e the arithmetic mean of
the endpoint densities, rho_s = 2.

U, psi and rho_0 come from the S2 distributions (fisher_hodge/gen_data.py: POT_ID, PSI_ID, IC_ID).  Reference: RK4
in float64 with dt = DT / ceil(DT q_max), q_max the largest escape rate (|lambda| dt <= 2), saved every DT = 0.05.
The time-step error is measured by repeating the stiffest sets with half the step.

Data: train 64 media x 4 trajectories (T = 1, amp = 1, 20 steps).  Test (200 steps = 10T): 16 unseen media x 4
trajectories in five conditions with the same media, fields and rho_0: a (T 1, amp 1), b05 (T 0.5), b2 (T 2),
c2 (amp 2), c4 (amp 4); and d: 16 unseen low-porosity media x 4 (T 1, amp 1).

Models (same inputs: rho_0, U, psi, T, drive amplitude, microstructure):
  wired_k      HyperME wired, k_e = k_ref T exp(kappa_theta(x_t, x_h)), kappa a symmetric pair network on the two
               endpoint blocks in the edge frame (DeepSets sum + elementwise product of a shared embedding)
  wired_const  ablation: one log k per medium, kappa = f_theta(medium porosity) (a scalar MLP, so it transfers)
  wired_untrained  kappa = 0 (homogeneous k = k_ref T), no training; wired_oracle: kappa = log g (integrator floor)
  fno          standard FNO (ours, 4 layers, 12 modes, width 32), residual next-step map
  gnn_ode      MeshGraphNet-style GNN (hyperme.baselines.GNNODE, 3 message-passing layers) -> drho/dt, Heun x 8
  flux_gnn     the same trunk emitting non-negative edge rates (hyperme.baselines.FluxGNN), positive substeps
  gfno         GENERIC-FNO, our re-implementation (fisher_hodge/ext/generic_fno.py), drive-channel variant
  helmfluid    HelmFluid (thuml, MIT; ~/ext_baselines), condition channels (run with ~/venvs/ext)
Black boxes get U and amp * psi z-scored with training scalars (so amp x 4 is a 4x larger field), T and amp as
constant channels, the 16 block pixels and the porosity per cell.

    python -m hyperme.bench_hetero gen --device cuda
    python -m hyperme.bench_hetero train --variant H1 --model wired_k --seed 0 --device cuda
    ~/venvs/ext/bin/python -m hyperme.bench_hetero helm --variant H1 --seed 0
    python -m hyperme.bench_hetero summary ; python -m hyperme.bench_hetero figs
"""
import argparse
import json
import math
import os
import sys
import time

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as Fn

ROOT = os.environ.get('FH_ROOT') or os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
OUT = os.path.join(ROOT, 'runs', 'hetero')
N, NF, SUB = 32, 128, 4
D = 0.02
KREF = D * N * N
DT, T_TRAIN, T_TEST = 0.05, 20, 200
H = 2 * N * N
RHO_S = 2.0
KC_DELTA, G_FLOOR = 0.3, 1e-4
POT_ID = dict(K=(3, 5), a=(1.0, 3.0), s=(0.08, 0.16))
PSI_ID = dict(K=(3, 5), a=(1.5, 4.0), s=(0.08, 0.16))
IC_ID = dict(n=(1, 3), sig=(0.05, 0.10))
POR_TRAIN, POR_LOW, ELL = (0.3, 0.7), (0.15, 0.3), (2.0, 10.0)
COND = dict(a=(1.0, 1.0), b05=(0.5, 1.0), b2=(2.0, 1.0), c2=(1.0, 2.0), c4=(1.0, 4.0), d=(1.0, 1.0))   # (T, amp)
TEST_SETS = list(COND)
STEP = dict(dt_max=DT / 8, theta0=0.9, theta1=1.0, max_sub=4000, t_tol=1e-9)


# ------------------------------------------------------------------------------------------ media and fields
def sample_medium(rng, por):
    """binary pore map (NF, NF) uint8 from a thresholded Gaussian random field; returns (map, porosity, ell)."""
    p, ell = rng.uniform(*por), rng.uniform(*ELL)
    k = np.fft.fftfreq(NF)
    kx, ky = np.meshgrid(k, k)
    spec = np.exp(-0.5 * (2 * np.pi) ** 2 * (kx ** 2 + ky ** 2) * ell ** 2)
    f = np.real(np.fft.ifft2(np.fft.fft2(rng.standard_normal((NF, NF))) * np.sqrt(spec)))
    pore = (f > np.quantile(f, 1 - p)).astype(np.uint8)
    return pore, p, ell


def blocks_of(pore):
    """(..., NF, NF) -> (..., N, N, 16): the 4 x 4 sub-block of every cell, row-major (r, s)"""
    sh = pore.shape[:-2]
    return pore.reshape(*sh, N, SUB, N, SUB).swapaxes(-3, -2).reshape(*sh, N, N, SUB * SUB)


def edge_frames(blk):
    """blk (B, n, n, 16) -> tail and head blocks in the edge frame (B, H, 16) and their porosities (B, H).
    Edge order of hyperme.complex.GridOps (dir * n^2 + a * n + b).  Horizontal edge (a,b)->(a,b+1): tail as is, head
    mirrored left-right; vertical edge (a,b)->(a+1,b): both transposed, head mirrored.  In this frame the shared face
    is column 3 of both blocks and the rows are aligned, so swapping tail and head is a symmetry."""
    B, n = blk.shape[0], blk.shape[1]
    b4 = blk.reshape(B, n, n, SUB, SUB)
    por = b4.mean((-1, -2))
    Xc = torch.stack([b4, b4.transpose(-1, -2)], 1).reshape(B, 2 * n * n, SUB * SUB)
    Yc = torch.stack([b4.roll(-1, 2).flip(-1), b4.roll(-1, 1).transpose(-1, -2).flip(-1)], 1).reshape(B, 2 * n * n, SUB * SUB)
    pt = torch.stack([por, por], 1).reshape(B, -1)
    ph = torch.stack([por.roll(-1, 2), por.roll(-1, 1)], 1).reshape(B, -1)
    return Xc, Yc, pt, ph


def g_raw(Xc, Yc, pt, ph):
    """the unnormalised effective-diffusion factor (Kozeny-Carman x tortuosity) of every edge, (B, H)"""
    X = Xc.reshape(*Xc.shape[:-1], SUB, SUB); Y = Yc.reshape(*Yc.shape[:-1], SUB, SUB)
    eh = 2 * pt * ph / (pt + ph).clamp_min(1e-12)
    nc = (X[..., :, 3] * Y[..., :, 3]).sum(-1)
    ns = (X[..., :, 2] * X[..., :, 3] * Y[..., :, 3] * Y[..., :, 2]).sum(-1)
    tau = 0.02 + 0.58 * nc / 4 + 0.40 * ns / 4
    return eh ** 3 / ((1 - eh) ** 2 + KC_DELTA) * tau


def true_g(blk, C):
    Xc, Yc, pt, ph = edge_frames(blk)
    return C * g_raw(Xc, Yc, pt, ph) + G_FLOOR


def eps_edge(blk):
    _, _, pt, ph = edge_frames(blk)
    return 2 * pt * ph / (pt + ph).clamp_min(1e-12)


def pgauss(X, Y, cx, cy, s):
    d2 = (np.sin(np.pi * (X - cx)) ** 2 + np.sin(np.pi * (Y - cy)) ** 2) / np.pi ** 2
    return np.exp(-d2 / (2 * s * s))


def sample_field(rng, spec, XY):
    K = rng.integers(spec['K'][0], spec['K'][1] + 1)
    X, Y = XY
    out = np.zeros_like(X)
    for _ in range(K):
        cx, cy, a, s = rng.uniform(), rng.uniform(), rng.choice([-1, 1]) * rng.uniform(*spec['a']), rng.uniform(*spec['s'])
        out = out + a * pgauss(X, Y, cx, cy, s)
    return out


def sample_ic(rng, XY, eps=1e-3):
    n = rng.integers(IC_ID['n'][0], IC_ID['n'][1] + 1)
    w = rng.uniform(0.5, 1.5, n); w /= w.sum()
    X, Y = XY
    rho = sum(w[i] * pgauss(X, Y, rng.uniform(), rng.uniform(), rng.uniform(*IC_ID['sig'])) for i in range(n))
    rho = rho / rho.mean()
    return (rho + eps) / (1 + eps)


def head_e(u):
    return torch.stack([u.roll(-1, -1), u.roll(-1, -2)], -3)


def tail_e(u):
    return torch.stack([u, u], -3)


def div_T(f):
    return (f[..., 0, :, :].roll(1, -1) - f[..., 0, :, :]) + (f[..., 1, :, :].roll(1, -2) - f[..., 1, :, :])


def dual_dpsi(P):
    return torch.stack([P - P.roll(1, -2), -(P - P.roll(1, -1))], -3)


def corner_to_node(P):
    return 0.25 * (P + P.roll(1, -1) + P.roll(1, -2) + P.roll(1, -1).roll(1, -2))


# ------------------------------------------------------------------------------------------ reference solver
def ref_flux(rho, kk, phi, A, crowd):
    """rho (B,n,n), kk (B,2,n,n) = k_ref T g, phi = U/T, A = amp dpsi / T -> (drho/dt, J (B,2,n,n))"""
    x = 0.5 * (tail_e(phi) - head_e(phi) + A)
    k = kk / (1 + 0.5 * (tail_e(rho) + head_e(rho)) / RHO_S) if crowd else kk
    J = k * (tail_e(rho) * torch.exp(x) - head_e(rho) * torch.exp(-x))
    return div_T(J), J


def escape_max(kk, phi, A):
    x = 0.5 * (tail_e(phi) - head_e(phi) + A)
    o, i = kk * torch.exp(x), kk * torch.exp(-x)
    q = o[:, 0] + o[:, 1] + i[:, 0].roll(1, -1) + i[:, 1].roll(1, -2)
    return float(q.max())


@torch.no_grad()
def simulate(rho0, g, U, dpsi, T, amp, crowd, nsteps, ns=None):
    """RK4, float64, ns substeps per DT (default ceil(DT q_max)).  Returns (rho (B, nsteps+1, n, n), J at the end (B, H), ns)."""
    T = T.view(-1, 1, 1); amp = amp.view(-1, 1, 1, 1)
    kk = KREF * T[..., None] * g.view(-1, 2, N, N)
    phi = U / T
    A = amp * dpsi / T[..., None]
    ns = ns or int(math.ceil(DT * escape_max(kk, phi, A)))
    dt = DT / ns
    rho = rho0.clone(); out = [rho]
    f = lambda r: ref_flux(r, kk, phi, A, crowd)[0]
    for _ in range(nsteps):
        for _ in range(ns):
            k1 = f(rho); k2 = f(rho + 0.5 * dt * k1); k3 = f(rho + 0.5 * dt * k2); k4 = f(rho + dt * k3)
            rho = rho + dt / 6 * (k1 + 2 * k2 + 2 * k3 + k4)
        out.append(rho)
    J = ref_flux(rho, kk, phi, A, crowd)[1]
    return torch.stack(out, 1), J.reshape(J.shape[0], -1), ns


def make_pool(seed, n_media, n_per, por):
    """media and per-trajectory fields (numpy)."""
    rng = np.random.default_rng(seed)
    c = (np.arange(N) + 0.5) / N; k = (np.arange(N) + 1.0) / N
    XY, XYc = np.meshgrid(c, c), np.meshgrid(k, k)
    pores, pm, ells = [], [], []
    for _ in range(n_media):
        pr, p, ell = sample_medium(rng, por)
        pores.append(pr); pm.append(float(pr.mean())); ells.append(ell)
    U, P, R0, med = [], [], [], []
    for m in range(n_media):
        for _ in range(n_per):
            U.append(sample_field(rng, POT_ID, XY)); P.append(sample_field(rng, PSI_ID, XYc)); R0.append(sample_ic(rng, XY))
            med.append(m)
    return dict(pore=np.stack(pores), pmed=np.array(pm), ell=np.array(ells), U=np.stack(U), Pc=np.stack(P),
                rho0=np.stack(R0), med=np.array(med))


def pool_tensors(pool, dev):
    T64 = lambda x: torch.as_tensor(x, dtype=torch.float64, device=dev)
    blk_m = T64(blocks_of(pool['pore']))
    med = torch.as_tensor(pool['med'], device=dev)
    Pc = T64(pool['Pc'])
    return dict(blk=blk_m[med], U=T64(pool['U']), dpsi=dual_dpsi(Pc), psin=corner_to_node(Pc), rho0=T64(pool['rho0']),
                pmed=T64(pool['pmed'])[med], med=med)


def generate(device='cuda', out=OUT):
    os.makedirs(os.path.join(out, 'data'), exist_ok=True)
    t0 = time.time()
    pools = dict(train=make_pool(7001, 64, 4, POR_TRAIN), test=make_pool(7002, 16, 4, POR_TRAIN), low=make_pool(7003, 16, 4, POR_LOW))
    tens = {k: pool_tensors(v, device) for k, v in pools.items()}
    Cn = 1.0 / float(g_raw(*edge_frames(torch.as_tensor(blocks_of(pools['train']['pore']), dtype=torch.float64))).mean())
    info = dict(C=Cn, KREF=KREF, D=D, DT=DT, RHO_S=RHO_S, KC_DELTA=KC_DELTA, G_FLOOR=G_FLOOR, cond=COND)
    for k, v in tens.items():
        v['g'] = true_g(v['blk'], Cn)
        gm = v['g'][::4]
        info[f'g_{k}'] = dict(mean=float(gm.mean()), median=float(gm.median()), max=float(gm.max()),
                             frac_below_1e-2=float((gm < 1e-2).double().mean()), porosity=pools[k]['pmed'].tolist())
    sets = dict(train=('train', 1.0, 1.0, T_TRAIN))
    for s, (T, amp) in COND.items():
        sets[s] = ('low' if s == 'd' else 'test', T, amp, T_TEST)
    # normalisation scalars for the black boxes (training set): U, amp * psin, amp * dpsi
    tr = tens['train']
    norm = dict(U=(float(tr['U'].mean()), float(tr['U'].std())), psi=(float(tr['psin'].mean()), float(tr['psin'].std())),
                dpsi=float(tr['dpsi'].std()))
    sims = {}
    for variant, crowd in (('H1', False), ('H1p', True)):
        data = dict(meta=dict(info, variant=variant, crowd=crowd, norm=norm,
                              pore=dict((k, torch.as_tensor(v['pore'])) for k, v in pools.items()),
                              pmed=dict((k, v['pmed'].tolist()) for k, v in pools.items())))
        for s, (pool, T, amp, nst) in sets.items():
            v = tens[pool]
            B = v['U'].shape[0]
            Tt, At = torch.full((B,), T, dtype=torch.float64, device=device), torch.full((B,), amp, dtype=torch.float64, device=device)
            t1 = time.time()
            R, J, ns = simulate(v['rho0'], v['g'], v['U'], v['dpsi'], Tt, At, crowd, nst)
            rec = dict(ns_per_DT=ns, sim_time_s=time.time() - t1, min_rho=float(R.min()),
                       mass_drift=float((R.mean((-1, -2)) - R[:, :1].mean((-1, -2))).abs().max()))
            if s in ('b2', 'c4', 'train'):
                R2, _, _ = simulate(v['rho0'][:16], v['g'][:16], v['U'][:16], v['dpsi'][:16], Tt[:16], At[:16], crowd, nst, ns=2 * ns)
                e = (torch.linalg.vector_norm((R[:16, -1] - R2[:, -1]).flatten(1), dim=1) / torch.linalg.vector_norm(R2[:, -1].flatten(1), dim=1))
                rec['dt_halving_relerr_end_max'] = float(e.max())
            f32 = lambda x: x.float().cpu()
            data[s] = dict(rho=f32(R), U=f32(v['U']), dpsi=f32(v['dpsi']), psin=f32(v['psin']), blk=v['blk'].to(torch.uint8).cpu(),
                           pmed=f32(v['pmed']), med=v['med'].cpu(), T=f32(Tt), amp=f32(At), g=f32(v['g']), J_end=f32(J), pool=pool)
            data['meta'][f'sim_{s}'] = rec
            sims.setdefault(variant, {})[s] = rec
            print(variant, s, json.dumps(rec), flush=True)
        torch.save(data, os.path.join(out, 'data', f'{variant}.pt'))
    info['gen_time_s'] = time.time() - t0
    json.dump(dict(info, norm=norm, sims=sims), open(os.path.join(out, 'data', 'gen_info.json'), 'w'), indent=1)
    return info


def load(variant, device='cpu', out=OUT):
    """dict(set -> dict(c (B, T+1, V), drive, g (B, H), J_end), meta)"""
    data = torch.load(os.path.join(out, 'data', f'{variant}.pt'))
    res = dict(meta=data['meta'])
    for s in ['train'] + TEST_SETS:
        d = data[s]
        B, T1 = d['rho'].shape[:2]
        drive = dict(U=d['U'].reshape(B, -1), T=d['T'], amp=d['amp'], dpsi=d['dpsi'], psin=d['psin'],
                     blk=d['blk'].float(), pmed=d['pmed'])
        res[s] = dict(c=d['rho'].reshape(B, T1, -1).to(device), drive={k: v.to(device) for k, v in drive.items()},
                      g=d['g'], J_end=d['J_end'], med=d['med'])
    return res


# ------------------------------------------------------------------------------------------ HyperME (wired)
class PairKappa(nn.Module):
    """symmetric pair network: kappa(x_t, x_h) = rho([psi(x_t) + psi(x_h), psi(x_t) * psi(x_h)]), shared psi."""
    def __init__(s, nin=SUB * SUB + 1, hidden=64):
        super().__init__()
        from .model import MLP
        s.psi = MLP(nin, hidden, hidden)
        s.rho = MLP(2 * hidden, hidden, 1, last='zero')

    def forward(s, xt, xh):
        a, b = s.psi(xt), s.psi(xh)
        return s.rho(torch.cat([a + b, a * b], -1)).squeeze(-1)


class MediumKinetics(nn.Module):
    """k_e = k_ref T exp(kappa_e).  mode 'edge': kappa = core + PairKappa(endpoint blocks in the edge frame);
    'const': kappa = core + f(medium porosity) (one value per medium); 'zero': kappa = core (homogeneous);
    'oracle': kappa = log g_true (needs C)."""
    rho_dependent = False
    split = True
    chem_prefactor = False

    def __init__(s, mode='edge', hidden=64, C=None):
        super().__init__()
        from .model import MLP
        s.mode, s.C = mode, C
        s.core = nn.Parameter(torch.zeros(()))
        s.register_buffer('k_ref', torch.tensor(KREF))
        if mode == 'edge':
            s.net = PairKappa(hidden=hidden)
        elif mode == 'const':
            s.net = MLP(1, 16, 1, last='zero')

    def kappa(s, feats):
        xt, xh = feats['xt'], feats['xh']
        B = xt.shape[0]
        if s.mode == 'oracle':
            return torch.log(s.C * g_raw(*feats['frames']) + G_FLOOR)
        k = s.core.to(xt.dtype).expand(B, xt.shape[1])
        if s.mode == 'edge':
            k = k + s.net(xt, xh)
        elif s.mode == 'const':
            k = k + s.net(2 * feats['pmed'][:, None] - 1)
        return k

    def forward(s, feats, hg=None, c=None):
        return s.k_ref.to(feats['xt'].dtype) * feats['T'][:, None] * torch.exp(s.kappa(feats))


def wired_featurizer():
    def feat(F):
        U = F['U']; B = U.shape[0]
        Xc, Yc, pt, ph = edge_frames(F['blk'])
        T = F['T'].to(U.dtype)
        line = (F['amp'][:, None, None, None] * F['dpsi']).reshape(B, -1)
        return dict(U=U.reshape(B, -1), T=T, line=line, chem_logc=None, chem_mu0=None,
                    edge_sym=torch.zeros(B, H, 1, dtype=U.dtype, device=U.device), edge_ori=torch.zeros(B, H, dtype=U.dtype, device=U.device),
                    node={}, xt=torch.cat([2 * Xc - 1, (2 * pt - 1)[..., None]], -1), xh=torch.cat([2 * Yc - 1, (2 * ph - 1)[..., None]], -1),
                    frames=(Xc, Yc, pt, ph), pmed=F['pmed'], raw=F)
    feat.n_edge_sym = 1
    return feat


def build_wired(mode='edge', C=None, hidden=64):
    from .complex import Hypergraph
    from .model import HyperME, WiredFreeEnergy, WiredDrive
    hg = Hypergraph.grid_transport(N)
    return HyperME(hg, WiredFreeEnergy(hg, 'none'), MediumKinetics(mode, hidden, C), WiredDrive(hg, harmonic=True),
                   wired_featurizer(), step=dict(STEP))


# ------------------------------------------------------------------------------------------ black boxes
def bb_cond(F, norm):
    """condition channels (B, 21, n, n): U, amp * psin (z-scored with training scalars), T, amp, porosity, 16 block pixels"""
    U = F['U'].reshape(-1, N, N); B = U.shape[0]
    one = torch.ones_like(U)
    blk = F['blk'].reshape(B, N, N, SUB * SUB)
    return torch.cat([((U - norm['U'][0]) / norm['U'][1])[:, None],
                      ((F['amp'][:, None, None] * F['psin'] - norm['psi'][0]) / norm['psi'][1])[:, None],
                      (F['T'][:, None, None] * one)[:, None], (F['amp'][:, None, None] * one)[:, None],
                      blk.mean(-1)[:, None], blk.permute(0, 3, 1, 2)], 1)


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
    """standard FNO (Li et al. 2021): 1x1 lift, 4 x [spectral conv + 1x1 conv, GELU except after the last], 1x1 projection
    to 128 and to 1.  Residual next-step map rho_{t+DT} = rho_t + FNO([rho_t, cond])."""
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
    name = 'fno'

    def __init__(s, norm):
        super().__init__()
        s.norm = norm
        s.net = FNO2d(1 + 21)

    def prepare(s, drive):
        from .model import Context
        return Context(cond=bb_cond(drive, s.norm))

    def context(s, drive):
        from .model import Context
        return drive if isinstance(drive, Context) else s.prepare(drive)

    def advance(s, c, drive, Dt, **kw):
        ctx = s.context(drive)
        r = c.reshape(-1, N, N)
        return (r + s.net(torch.cat([r[:, None], ctx['cond']], 1))).reshape(c.shape), 1


def gnn_featurizer(norm):
    def feat(F):
        cond = bb_cond(F, norm)                      # (B, 21, n, n)
        B = cond.shape[0]
        node = {f'f{i}': cond[:, i].reshape(B, -1) for i in range(cond.shape[1])}
        ish = torch.zeros(B, 2, N, N, dtype=cond.dtype, device=cond.device); ish[:, 0] = 1
        sym = torch.stack([ish, 1 - ish], -1).reshape(B, H, 2)
        ori = (F['amp'][:, None, None, None] * F['dpsi']).reshape(B, H) / norm['dpsi']
        return dict(U=F['U'].reshape(B, -1), node=node, edge_sym=sym, edge_ori=ori, raw=F)
    feat.n_edge_sym = 2
    return feat


def build(model, meta):
    norm = meta['norm']
    if model.startswith('wired'):
        mode = dict(wired_k='edge', wired_const='const', wired_untrained='zero', wired_oracle='oracle')[model]
        return build_wired(mode, C=meta['C'])
    if model == 'fno':
        return FNOOp(norm)
    from .complex import Hypergraph
    from . import baselines as BL
    hg = Hypergraph.grid_transport(N)
    feat = gnn_featurizer(norm)
    nn_in, ne_in = 1 + 21, 3
    if model == 'gnn_ode':
        return BL.GNNODE(hg, nn_in, ne_in, feat, hidden=64, layers=3, scale=KREF)
    if model == 'flux_gnn':
        return BL.FluxGNN(hg, nn_in, ne_in, feat, hidden=64, layers=3, scale=KREF, step=dict(STEP))
    raise ValueError(model)


# ------------------------------------------------------------------------------------------ metrics
def relerr_traj(R, Y):
    B, T1 = Y.shape[:2]
    e = torch.linalg.vector_norm((R - Y).reshape(B, T1, -1), dim=-1) / torch.linalg.vector_norm(Y.reshape(B, T1, -1), dim=-1)
    return torch.where(torch.isfinite(e), e, torch.full_like(e, float('inf')))


def rollout_metrics(R, Y):
    """R, Y (B, T+1, V) on the CPU"""
    R, Y = R.double(), Y.double()
    e = relerr_traj(R, Y).numpy()
    B = R.shape[0]
    fin = torch.isfinite(R.reshape(B, -1)).all(1)
    m0 = Y[:, 0].mean(-1)
    drift = ((R.mean(-1) - m0[:, None]).abs() / m0[:, None])
    return dict(err_T=float(np.median(e[:, 20])), err_10T=float(np.median(e[:, -1])), err_10T_mean=float(np.mean(e[:, -1])),
                err_10T_per_traj=e[:, -1].tolist(), n_blown=int((~fin).sum()),
                n_neg_traj=int((torch.nan_to_num(R, nan=0.0) < 0).reshape(B, -1).any(1).sum()),
                min_value=float(np.nanmin(R.numpy())), mass_drift_max=float(np.nanmax(drift.numpy())))


def flux_metrics(Jm, Jt):
    Jm, Jt = Jm.double(), Jt.double()
    e = (torch.linalg.vector_norm(Jm - Jt, dim=-1) / torch.linalg.vector_norm(Jt, dim=-1)).numpy()
    cos = ((Jm * Jt).sum(-1) / (torch.linalg.vector_norm(Jm, dim=-1) * torch.linalg.vector_norm(Jt, dim=-1) + 1e-30)).numpy()
    e = np.where(np.isfinite(e), e, np.inf)
    return dict(J_relerr=float(np.median(e)), J_cos=float(np.nanmedian(cos)))


def pearson(x, y):
    x = x - x.mean(); y = y - y.mean()
    return float((x * y).sum() / (np.sqrt((x * x).sum() * (y * y).sum()) + 1e-30))


@torch.no_grad()
def k_metrics(model, sd):
    """learned vs true g = k / (k_ref T) on the distinct media of a set (first trajectory of every medium)"""
    med = sd['med'].numpy()
    first = np.unique(med, return_index=True)[1]
    idx = torch.as_tensor(first, device=sd['c'].device)
    drive = {k: v[idx] for k, v in sd['drive'].items()}
    feats = model.featurizer(drive)
    gh = torch.exp(model.kinetics.kappa(feats)).double().cpu().numpy().ravel()
    gt = sd['g'][first].double().numpy().ravel()
    eps = eps_edge(drive['blk'].double()).cpu().numpy().ravel()
    lh, lt = np.log(gh), np.log(gt)
    op = gt > 1e-2
    out = dict(pearson_logk=pearson(lh, lt), pearson_k=pearson(gh, gt), relerr_k=float(np.linalg.norm(gh - gt) / np.linalg.norm(gt)),
               median_abs_logratio=float(np.median(np.abs(lh - lt))), pearson_logk_open=pearson(lh[op], lt[op]),
               median_abs_logratio_open=float(np.median(np.abs(lh[op] - lt[op]))), frac_open=float(op.mean()),
               spearman_logk=pearson(np.argsort(np.argsort(lh)).astype(float), np.argsort(np.argsort(lt)).astype(float)))
    return out, dict(g_hat=gh.astype(np.float32), g_true=gt.astype(np.float32), eps=eps.astype(np.float32))


# ------------------------------------------------------------------------------------------ train / evaluate
def gpu_ok(device):
    if str(device).startswith('cuda'):
        from .gpu_guard import require_free
        require_free(device)


@torch.no_grad()
def eval_model(model, data, log, snap, chunk=64):
    from .train import rollout
    res = {}
    for s in TEST_SETS:
        sd = data[s]; Y = sd['c']
        t0 = time.time()
        Rs, Js, nsubs = [], [], []
        for i in range(0, Y.shape[0], chunk):
            dr = {k: v[i:i + chunk] for k, v in sd['drive'].items()}
            R, ns, _ = rollout(model, dr, Y[i:i + chunk, 0], Y.shape[1] - 1, DT)
            Rs.append(R.cpu()); nsubs.append(ns)
            if hasattr(model, 'flux'):
                cl = R[:, -1].clamp_min(1e-30) if torch.isfinite(R[:, -1]).all() else R[:, -1]
                Js.append(model.flux(cl, dr).cpu())
        R = torch.cat(Rs)
        m = rollout_metrics(R, Y.cpu())
        m['infer_time_s'] = time.time() - t0; m['substeps_per_DT'] = float(np.mean(nsubs))
        if Js:
            m.update(flux_metrics(torch.cat(Js), sd['J_end']))
        if hasattr(model, 'kinetics') and hasattr(model.kinetics, 'kappa') and s in ('a', 'd'):
            km, kd = k_metrics(model, sd)
            m['k'] = km
            snap.update({f'k_{s}_{k}': v for k, v in kd.items()})
        snap[f'R_{s}'] = R[:4, -1].numpy().astype(np.float32)
        snap[f'R_{s}_T'] = R[:4, 20].numpy().astype(np.float32)
        res[s] = m
        log(f'[{s}] T {m["err_T"]:.4g} 10T {m["err_10T"]:.4g} blown {m["n_blown"]} neg {m["n_neg_traj"]} mass {m["mass_drift_max"]:.1e}'
            + (f' | J {m["J_relerr"]:.3g}' if 'J_relerr' in m else '')
            + (f' | k r(log) {m["k"]["pearson_logk"]:.3f} r(k) {m["k"]["pearson_k"]:.3f} relerr {m["k"]["relerr_k"]:.3f}' if 'k' in m else '')
            + f' | sub {m["substeps_per_DT"]:.1f} t {m["infer_time_s"]:.0f}s')
    return res


def run_train(variant, model_name, seed, iters, device, lr=1e-3, out=OUT):
    gpu_ok(device)
    torch.manual_seed(seed); np.random.seed(seed)
    data = load(variant, device, out)
    meta = data['meta']
    odir = os.path.join(out, variant); os.makedirs(odir, exist_ok=True)
    name = f'{model_name}_s{seed}'
    logf = open(os.path.join(odir, name + '.log'), 'w')

    def log(*a):
        s = ' '.join(str(x) for x in a); print(s, flush=True); logf.write(s + '\n'); logf.flush()
    tr = dict(c=data['train']['c'], drive=data['train']['drive'])
    if model_name == 'gfno':
        return run_gfno(variant, seed, iters, device, data, odir, name, log)
    model = build(model_name, meta).to(device)
    npar = sum(p.numel() for p in model.parameters())
    log(f'{variant} {name}: params {npar} iters {iters} lr {lr} device {device}')
    info = dict(train_time_s=0.0)
    if model_name not in ('wired_untrained', 'wired_oracle') and iters > 0:
        from .train import train
        info = train(model, tr, iters, seed, DT, batch=8, lr=lr, kmin=4, kmax=8, log=log)
        torch.save(model.state_dict(), os.path.join(odir, name + '.pt'))
    model.eval()
    snap = {}
    ev = eval_model(model, data, log, snap)
    extra = {}
    if model_name.startswith('wired'):
        extra = dict(s_A=float(model.drive.s_A), harmonic_b=model.drive.b.tolist(), core=float(model.kinetics.core))
    res = dict(variant=variant, model=model_name, seed=seed, params=npar, iters=iters, lr=lr, **{k: v for k, v in info.items()}, extra=extra, sets=ev)
    json.dump(res, open(os.path.join(odir, name + '.json'), 'w'))
    np.savez_compressed(os.path.join(odir, name + '_snap.npz'), **snap)
    return res


def run_gfno(variant, seed, iters, device, data, odir, name, log):
    sys.path.insert(0, os.path.join(ROOT, 'fisher_hodge', 'ext'))
    from generic_fno import GenericFNO, train_generic, rollout, n_params
    norm = data['meta']['norm']
    tr = data['train']
    B = tr['c'].shape[0]
    X = tr['c'].reshape(B, -1, 1, N, N).float()
    cond = bb_cond(tr['drive'], norm).float()
    model = GenericFNO(n_fields=1, n_cond=cond.shape[1]).to(device)
    npar = n_params(model)
    log(f'{variant} {name}: params {npar} steps {iters} (GENERIC-FNO drive variant, paper defaults)')
    t0 = time.time()
    info = train_generic(model, X, cond, steps=iters, device=device, seed=seed, log=log)
    info['train_time_s'] = time.time() - t0
    torch.save(model.state_dict(), os.path.join(odir, name + '.pt'))
    res, snap = {}, {}
    for s in TEST_SETS:
        sd = data[s]; Y = sd['c']
        t1 = time.time()
        R = rollout(model, Y[:, 0].reshape(-1, 1, N, N).float(), Y.shape[1] - 1, bb_cond(sd['drive'], norm).float())
        R = R.reshape(R.shape[0], R.shape[1], -1)
        m = rollout_metrics(R, Y.cpu()); m['infer_time_s'] = time.time() - t1
        snap[f'R_{s}'] = R[:4, -1].numpy().astype(np.float32); snap[f'R_{s}_T'] = R[:4, 20].numpy().astype(np.float32)
        res[s] = m
        log(f'[{s}] T {m["err_T"]:.4g} 10T {m["err_10T"]:.4g} blown {m["n_blown"]} neg {m["n_neg_traj"]} mass {m["mass_drift_max"]:.1e}')
    out = dict(variant=variant, model='gfno', seed=seed, params=npar, iters=iters,
               train_time_s=info['train_time_s'], train=dict((k, v) for k, v in info.items() if k != 'loss_curve'), sets=res)
    json.dump(out, open(os.path.join(odir, name + '.json'), 'w'))
    np.savez_compressed(os.path.join(odir, name + '_snap.npz'), **snap)
    return out


def run_helm(variant, seed, steps, out=OUT):
    """HelmFluid with the defaults of scripts/Navier_Stokes_HelmFluid.sh (as runs/external/density/code/run_helmfluid.py):
    T_in 10 (history padded with rho_0), T_out 10, batch 5, lr 5e-4, their Adam (wd 1e-4), StepLR gamma 0.5 with the
    period scaled to the step budget, relative-L2 LpLoss summed over the 10 steps.  Condition channels appended."""
    from types import SimpleNamespace
    from .gpu_guard import require_free
    require_free('cuda')
    sys.path.insert(0, os.path.expanduser('~/ext_baselines/patches'))
    from helm_cond import CondHelmFluid
    from utils.utilities3 import LpLoss
    from utils.adam import Adam
    torch.manual_seed(seed); np.random.seed(seed); torch.cuda.manual_seed(seed)
    dev = 'cuda'
    T_IN, T_OUT, BATCH, LR = 10, 10, 5, 5e-4
    data = load(variant, 'cpu', out)
    norm = data['meta']['norm']
    odir = os.path.join(out, variant); os.makedirs(odir, exist_ok=True)
    name = f'helmfluid_s{seed}'
    logf = open(os.path.join(odir, name + '.log'), 'w')

    def log(*x):
        s = ' '.join(str(y) for y in x); print(s, flush=True); logf.write(s + '\n'); logf.flush()
    nc = 21
    args = SimpleNamespace(groups=4, dataset_name='NS', in_dim=T_IN + nc, out_dim=1, d_model=64, padding='0,0')
    model = CondHelmFluid(args, n_cond=nc).to(dev)
    npar = sum(q.numel() for q in model.parameters())
    log(f'{variant} {name}: params {npar} steps {steps}')

    def history(u, s):
        idx = torch.clamp(torch.arange(s - T_IN + 1, s + 1), min=0)
        return u[:, idx].permute(0, 2, 3, 1)
    tr = data['train']
    U = tr['c'].reshape(tr['c'].shape[0], -1, N, N)
    Cd = bb_cond(tr['drive'], norm).permute(0, 2, 3, 1)
    win = [(i, s) for i in range(U.shape[0]) for s in range(0, U.shape[1] - 1 - T_OUT + 1)]
    opt = Adam(model.parameters(), lr=LR, weight_decay=1e-4)
    step_size = max(1, round(steps * 100 / 501))
    sched = torch.optim.lr_scheduler.StepLR(opt, step_size=step_size, gamma=0.5)
    myloss = LpLoss(size_average=False)
    g = torch.Generator().manual_seed(seed)
    model.train(); t0 = time.time(); it = 0; hist = []
    while it < steps:
        perm = torch.randperm(len(win), generator=g)
        for j in range(0, len(perm) - BATCH + 1, BATCH):
            if it >= steps:
                break
            ii = [win[k] for k in perm[j:j + BATCH].tolist()]
            h = torch.stack([history(U[i:i + 1], s)[0] for i, s in ii]).to(dev)
            y = torch.stack([U[i, s + 1:s + 1 + T_OUT] for i, s in ii]).permute(0, 2, 3, 1).to(dev)
            c = torch.stack([Cd[i] for i, _ in ii]).to(dev)
            loss = 0
            for t in range(T_OUT):
                im, _, _ = model(torch.cat([h, c], -1))
                loss = loss + myloss(im.reshape(BATCH, -1), y[..., t:t + 1].reshape(BATCH, -1))
                h = torch.cat([h[..., 1:], im], -1)
            opt.zero_grad(); loss.backward(); opt.step(); sched.step(); it += 1
            hist.append(float(loss) / BATCH / T_OUT)
            if it % 200 == 0 or it == 1:
                log(f'it {it} loss/step {np.mean(hist[-200:]):.4e} lr {sched.get_last_lr()[0]:.2e} t {time.time() - t0:.0f}s')
    ttrain = time.time() - t0
    torch.save(model.state_dict(), os.path.join(odir, name + '.pt'))
    model.eval()
    res, snap = {}, {}
    with torch.no_grad():
        for s in TEST_SETS:
            sd = data[s]; Y = sd['c']
            B = Y.shape[0]
            t1 = time.time()
            Y2 = Y.reshape(B, -1, N, N)
            h = history(Y2[:, :1], 0).to(dev)
            c = bb_cond(sd['drive'], norm).permute(0, 2, 3, 1).to(dev)
            outs = [Y2[:, 0]]
            for t in range(Y2.shape[1] - 1):
                im, _, vel = model(torch.cat([h, c], -1))
                bad = ~torch.isfinite(im).flatten(1).all(1) | (im.flatten(1).abs().amax(1) > 1e4)
                im = torch.where(bad[:, None, None, None], torch.full_like(im, float('nan')), im)
                outs.append(im[..., 0].cpu())
                h = torch.cat([h[..., 1:], im], -1)
            R = torch.stack(outs, 1)
            m = rollout_metrics(R.reshape(B, R.shape[1], -1), Y)
            # flux from the model's own velocity at 10T (latent advection velocity; cosine is the meaningful number)
            _, _, vel = model(torch.cat([h, c], -1))
            rho = R[:, -1].double(); v = vel.double().cpu() / DT          # cells per unit time; J_e = rhobar vbar (per-edge units)
            Jh = 0.5 * (rho + rho.roll(-1, -1)) * 0.5 * (v[:, 0] + v[:, 0].roll(-1, -1))
            Jv = 0.5 * (rho + rho.roll(-1, -2)) * 0.5 * (v[:, 1] + v[:, 1].roll(-1, -2))
            m.update(flux_metrics(torch.stack([Jh, Jv], 1).reshape(B, -1), sd['J_end']))
            m['infer_time_s'] = time.time() - t1
            snap[f'R_{s}'] = R[:4, -1].reshape(4, -1).numpy().astype(np.float32)
            snap[f'R_{s}_T'] = R[:4, 20].reshape(4, -1).numpy().astype(np.float32)
            res[s] = m
            log(f'[{s}] T {m["err_T"]:.4g} 10T {m["err_10T"]:.4g} blown {m["n_blown"]} neg {m["n_neg_traj"]} mass {m["mass_drift_max"]:.1e} | J {m["J_relerr"]:.3g} cos {m["J_cos"]:.2f}')
    o = dict(variant=variant, model='helmfluid', seed=seed, params=npar, iters=steps, train_time_s=ttrain, step_size=step_size,
             final_loss=float(np.mean(hist[-200:])), sets=res)
    json.dump(o, open(os.path.join(odir, name + '.json'), 'w'))
    np.savez_compressed(os.path.join(odir, name + '_snap.npz'), **snap)
    return o


def run_persistence(variant, out=OUT):
    data = load(variant, 'cpu', out)
    res = {}
    for s in TEST_SETS:
        Y = data[s]['c']
        res[s] = rollout_metrics(Y[:, :1].expand_as(Y), Y)
    o = dict(variant=variant, model='persistence', seed=0, params=0, iters=0, train_time_s=0.0, sets=res)
    os.makedirs(os.path.join(out, variant), exist_ok=True)
    json.dump(o, open(os.path.join(out, variant, 'persistence_s0.json'), 'w'))
    return o


# ------------------------------------------------------------------------------------------ summary and figures
LABEL = dict(wired_k='HyperME wired, k = k_ref T exp(kappa_theta(microstructure))', wired_const='HyperME wired, one k per medium (ablation)',
             wired_untrained='HyperME wired, untrained (homogeneous k)', wired_oracle='HyperME wired, true k (integrator floor)',
             fno='FNO (standard, ours)', gnn_ode='GNN-ODE (MeshGraphNet-style)', flux_gnn='Flux-GNN (conservative, positive)',
             gfno='GENERIC-FNO (re-impl., drive channels)', helmfluid='HelmFluid (ICML 2024, cond. channels)',
             persistence='No change, u(t) = u(0)')
ORDER = ['wired_k', 'wired_const', 'wired_untrained', 'wired_oracle', 'fno', 'gnn_ode', 'flux_gnn', 'gfno', 'helmfluid', 'persistence']
COLS = [('a', 'a: T 1, amp 1'), ('b05', 'b: T x0.5'), ('b2', 'b: T x2'), ('c2', 'c: amp x2'), ('c4', 'c: amp x4'), ('d', 'd: low porosity')]


def collect(variant, out=OUT):
    import glob
    res = {}
    for f in sorted(glob.glob(os.path.join(out, variant, '*_s[0-9].json'))):
        r = json.load(open(f))
        res.setdefault(r['model'], []).append(r)
    return res


def fmt(xs):
    xs = [x for x in xs if x is not None]
    if not xs:
        return '–'
    f = lambda x: 'inf' if not np.isfinite(x) else (f'{x:.3g}' if x >= 1e-3 else f'{x:.2e}')
    if len(xs) == 1:
        return f(xs[0])
    return f(float(np.mean(xs))) + f' ({f(min(xs))}–{f(max(xs))})'


def table(variant, out=OUT, key='err_10T'):
    res = collect(variant, out)
    lines = ['| Method | ' + ' | '.join(c[1] for c in COLS) + ' | params | train time | seeds |', '|---' * (len(COLS) + 4) + '|']
    for m in ORDER:
        if m not in res:
            continue
        rs = res[m]
        cells = [fmt([r['sets'][c][key] if c in r['sets'] and key in r['sets'][c] else None for r in rs]) for c, _ in COLS]
        tt = np.mean([r.get('train_time_s', 0.0) or 0.0 for r in rs])
        lines.append(f'| {LABEL[m]} | ' + ' | '.join(cells) + f' | {rs[0]["params"]:,} | {tt / 60:.0f} min | {len(rs)} |')
    return '\n'.join(lines)


def side_table(variant, out=OUT):
    """negative trajectories, mass drift, J error, blow-ups (seed-mean or max)"""
    res = collect(variant, out)
    lines = ['| Method | neg. traj. (of 64) a/b05/b2/c2/c4/d | max mass drift | J(10T) rel. err a/b2/c4/d | blown |', '|---|---|---|---|---|']
    for m in ORDER:
        if m not in res:
            continue
        rs = res[m]
        neg = '/'.join(str(max(r['sets'][c]['n_neg_traj'] for r in rs)) for c, _ in COLS)
        md = max(r['sets'][c]['mass_drift_max'] for r in rs for c, _ in COLS)
        J = '/'.join(fmt([r['sets'][c].get('J_relerr') for r in rs]) for c in ('a', 'b2', 'c4', 'd'))
        bl = sum(r['sets'][c]['n_blown'] for r in rs for c, _ in COLS)
        lines.append(f'| {LABEL[m]} | {neg} | {md:.1e} | {J} | {bl} |')
    return '\n'.join(lines)


def k_table(variant, out=OUT):
    res = collect(variant, out)
    lines = ['| Model | set | Pearson log k | Spearman log k | Pearson k | rel. L2 err k | median abs log ratio | Pearson log k (g > 0.01) |', '|---|---|---|---|---|---|---|---|']
    for m in ('wired_k', 'wired_const', 'wired_untrained'):
        for r in res.get(m, []):
            for s in ('a', 'd'):
                k = r['sets'][s].get('k')
                if k:
                    lines.append(f'| {m} s{r["seed"]} | {s} | {k["pearson_logk"]:.3f} | {k["spearman_logk"]:.3f} | {k["pearson_k"]:.3f} | {k["relerr_k"]:.3f} | '
                                 f'{k["median_abs_logratio"]:.3f} | {k["pearson_logk_open"]:.3f} |')
    return '\n'.join(lines)


def summary(out=OUT):
    for v in ('H1', 'H1p'):
        if os.path.exists(os.path.join(out, 'data', f'{v}.pt')) and not os.path.exists(os.path.join(out, v, 'persistence_s0.json')):
            run_persistence(v, out)
    gi = json.load(open(os.path.join(out, 'data', 'gen_info.json')))
    s = ['# H1: driven transport in heterogeneous media (generated by hyperme/bench_hetero.py)', '',
         'Relative L2 error at 10T (200 steps), median over 64 test trajectories (16 unseen media x 4); mean over seeds, '
         '(min–max) when two seeds. Columns: a = training conditions on unseen media; b = temperature x0.5 / x2; '
         'c = drive amplitude x2 / x4; d = unseen low-porosity media (0.15–0.3, training 0.3–0.7).', '',
         '## H1 (truth inside the model class, k is an unknown function of the microstructure)', '', table('H1', out), '',
         side_table('H1', out), '', '## H1\' (crowding mobility k / (1 + rhobar / 2), outside the model class)', '', table('H1p', out), '',
         side_table('H1p', out), '', '## Learned k against the true k on unseen media (k / (k_ref T))', '', k_table('H1', out), '',
         'H1\' (no true k in the class; compared with the crowding-free g):', '', k_table('H1p', out), '',
         '## Error at T (end of the training window)', '', table('H1', out, 'err_T'), '',
         '## Data and reference', '', '```', json.dumps(gi, indent=1)[:6000], '```']
    open(os.path.join(out, 'summary.md'), 'w').write('\n'.join(s))
    print('\n'.join(s[:12]))


def figs(out=OUT, seed=0):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    # k scatter
    fig, axs = plt.subplots(1, 2, figsize=(10, 4.4), constrained_layout=True)
    sn = np.load(os.path.join(out, 'H1', f'wired_k_s{seed}_snap.npz'))
    js = json.load(open(os.path.join(out, 'H1', f'wired_k_s{seed}.json')))
    for ax, s, title in zip(axs, ('a', 'd'), ('unseen media, porosity 0.3–0.7', 'unseen low-porosity media, 0.15–0.3')):
        gt, gh, ep = sn[f'k_{s}_g_true'], sn[f'k_{s}_g_hat'], sn[f'k_{s}_eps']
        rng = np.random.default_rng(0)
        i = rng.choice(len(gt), min(len(gt), 12000), replace=False)
        sc = ax.scatter(gt[i], gh[i], c=ep[i], s=3, cmap='viridis', vmin=0, vmax=1, rasterized=True)
        lo, hi = 5e-5, max(gt.max(), gh.max()) * 2
        ax.plot([lo, hi], [lo, hi], 'k--', lw=0.8)
        ax.set_xscale('log'); ax.set_yscale('log'); ax.set_xlim(lo, hi); ax.set_ylim(lo, hi)
        k = js['sets'][s]['k']
        ax.set_title(f'{title}\nPearson log k {k["pearson_logk"]:.3f}, Pearson k {k["pearson_k"]:.3f}', fontsize=10)
        ax.set_xlabel('true k / (k_ref T)'); ax.set_ylabel('learned k / (k_ref T)')
    fig.colorbar(sc, ax=axs, label='edge porosity (harmonic mean)')
    fig.savefig(os.path.join(out, 'k_scatter.png'), dpi=150); plt.close(fig)
    # rollout panel: one unseen medium at T x 2
    data = torch.load(os.path.join(out, 'data', 'H1.pt'))
    s, j = 'b2', 0
    med = int(data[s]['med'][j])
    pore = data['meta']['pore']['test'][med].numpy()
    truth = data[s]['rho'][j, -1].numpy()
    cols = [('truth', truth)]
    for m, lab in (('wired_k', 'HyperME wired'), ('fno', 'FNO'), ('gnn_ode', 'GNN-ODE'), ('flux_gnn', 'Flux-GNN')):
        f = os.path.join(out, 'H1', f'{m}_s{seed}_snap.npz')
        if os.path.exists(f):
            cols.append((lab, np.load(f)[f'R_{s}'][j].reshape(N, N)))
    fig, axs = plt.subplots(1, len(cols) + 1, figsize=(3.0 * (len(cols) + 1), 3.4), constrained_layout=True)
    axs[0].imshow(pore, cmap='gray_r', origin='lower'); axs[0].set_title(f'microstructure (porosity {pore.mean():.2f})', fontsize=9)
    vmax = float(np.nanmax(truth)) * 1.05
    for ax, (lab, im) in zip(axs[1:], cols):
        e = np.linalg.norm(im - truth) / np.linalg.norm(truth)
        ax.imshow(im, cmap='magma', vmin=0, vmax=vmax, origin='lower')
        neg = (im < 0).sum()
        ax.set_title(f'{lab}\nrel. err {e:.3f}' + (f', {neg} cells < 0' if neg else ''), fontsize=9)
    for ax in axs:
        ax.set_xticks([]); ax.set_yticks([])
    fig.suptitle('Unseen medium, temperature x2 (training T = 1): density at 10T', fontsize=10)
    fig.savefig(os.path.join(out, 'hetero_rollout.png'), dpi=150); plt.close(fig)


if __name__ == '__main__':
    p = argparse.ArgumentParser()
    p.add_argument('cmd', choices=['gen', 'train', 'helm', 'summary', 'figs', 'persistence'])
    p.add_argument('--variant', default='H1'); p.add_argument('--model', default='wired_k')
    p.add_argument('--seed', type=int, default=0); p.add_argument('--iters', type=int, default=4000)
    p.add_argument('--device', default='cuda'); p.add_argument('--lr', type=float, default=1e-3)
    a = p.parse_args()
    torch.set_num_threads(int(os.environ.get('OMP_NUM_THREADS', '8')))
    if a.cmd == 'gen':
        gpu_ok(a.device)
        print(json.dumps(generate(a.device), indent=1, default=str))
    elif a.cmd == 'train':
        run_train(a.variant, a.model, a.seed, a.iters, a.device, a.lr)
    elif a.cmd == 'helm':
        run_helm(a.variant, a.seed, a.iters)
    elif a.cmd == 'persistence':
        run_persistence(a.variant)
    elif a.cmd == 'summary':
        summary()
    else:
        figs()
