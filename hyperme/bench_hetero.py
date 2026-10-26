"""H1: driven transport in heterogeneous media, across media, conditions and discretisations.

Question: does the wired HyperME (Fisher-Hodge) learn a transferable operator, i.e. a learned constitutive law that maps
local material descriptors (not coordinates) to the kinetic prefactor, rather than one rate per edge?  The model learns k from 64 random media on the
32 x 32 grid and is tested on 16 unseen media at new temperatures, new drive strengths, a new medium type, and on
other discretisations (64 x 64, a random point cloud) without retraining.

Physics (meshfree, defined on the 128 x 128 fine pore map, never shown to the models):
    d_t rho = div( M(x) [ T grad rho + rho grad U - amp rho F ] ),   F = (d_y psi, -d_x psi)  (divergence free)
    M(x) direction-dependent (diagonal), from a local effective-medium rule at the pore scale:
        eps(x)   = porosity in a Gaussian window of 1 fine pixel,
        tau_x(x) = fraction of horizontally connected pore pairs in the window / eps,
        M_x(x)   = C eps^3 / ((1 - eps)^2 + 0.3) * (0.05 + 0.95 tau_x^2) * (0.1 + 0.9 chi(x)) + 1e-4,
    and M_y likewise; chi is the pore indicator, C normalises the mean edge mobility of the training media to 1,
    D = 0.02 sets the time scale (the S2 diffusion constant).
H1' (crowding, outside the model class): M -> M / (1 + rho / rho_s), rho_s = 2.
Reference: finite volumes on the 128 x 128 pixels (SQRA form, harmonic mean of M on every pixel face, the drive as
the exact flux of psi through the face), RK4 in float64 with |lambda| dt <= 2, saved every DT = 0.05, then projected
(cell averages) onto the evaluation discretisation: 32 x 32 (training), 64 x 64, or the Voronoi cells of a random
point cloud of 1024 points.  U, psi and rho_0 are the S2 distributions (fisher_hodge/gen_data.py), kept as their
Gaussian parameters so that they can be evaluated at any point.

Discretisation of the models (any point set with Voronoi cells): state c_i = cell mass / mean cell volume,
    a_e = k_e c_i exp((phi_i - phi_j + A_e)/2),  b_e = k_e c_j exp(-(...)),  phi = U/T - log(V_i / Vbar),
    A_e = amp int_e F . dl / T  (3-point Gauss-Legendre line integral along the edge),
    k_e = D T w_e exp(kappa_theta(desc_e)),  w_e = |sigma_e| / (|x_j - x_i| sqrt(V_i V_j))   (the SQRA geometric
    weight; n^2 on an n x n grid).
The descriptors are mesh-independent fields on the fine pore map, Gaussian window sigma = 2 fine pixels (sigma = 4
for a second porosity): porosity, connected-pair fractions along x and y at lag 1 and 2.  For an edge with direction
n they are mixed into the edge frame (parallel / perpendicular), taken at the midpoint and averaged over the two
endpoints: 12 numbers.  kappa_theta is a plain MLP (no message passing); the geometric weight is wired, so the
edge length does not enter the network (it is constant in training and could not be identified).

Data: train 64 media x 4 trajectories (T 1, amp 1, 20 steps of DT); test (200 steps = 10T): 16 unseen media x 4
trajectories in conditions a (T 1, amp 1), b05 / b2 (T x0.5 / x2), c2 / c4 (amp x2 / x4), with the same fields and
rho_0; d: 16 unseen low-porosity media (0.15-0.3, training 0.3-0.7); e64 / ecloud: the trajectories of a projected
onto 64 x 64 and onto the point cloud (models trained on 32 x 32, evaluated without retraining).

Models (same inputs: rho_0, U, psi, T, drive amplitude and the same meshfree microstructure descriptors):
  wired_k        HyperME wired, kappa_theta MLP of the edge descriptors
  wired_const    ablation: one kappa per medium, a scalar MLP of the medium porosity
  wired_untrained / wired_oracle   kappa = 0 / kappa = log of the true M averaged harmonically along the edge
  ude            UDE closure: the same edges, descriptors and conservative assembly, a free antisymmetric edge-flux MLP
  lm_hodge       learned-metric linear Hodge operator (internal ablation): fixed coboundaries d, a learned diagonal
                 SPD metric H from local invariants, propagation d^T H d, conservative (mass-weighted divergence)
                 readout of the update; T and amp only as input features (library given by the LMH_PKG environment
                 variable, on PYTHONPATH)
  fno            standard FNO (ours), gnn_ode / flux_gnn (hyperme.baselines), gfno (our GENERIC-FNO
                 re-implementation), helmfluid (thuml, MIT, ~/ext_baselines; run with ~/venvs/ext)

    python -m hyperme.bench_hetero gen --variant H1 --device cuda
    python -m hyperme.bench_hetero train --variant H1 --model wired_k --seed 0 --device cuda
    python -m hyperme.bench_hetero summary ; python -m hyperme.bench_hetero figs
"""
import argparse
import importlib
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
N, NF = 32, 128
D = 0.02
KREF = D * N * N
DT, T_TRAIN, T_TEST = 0.05, 20, 200
RHO_S = 2.0
KC_DELTA, M_FLOOR = 0.3, 1e-4
SIG_T, SIG_M, SIG_M2 = 1.0, 2.0, 4.0              # Gaussian windows (fine pixels): truth, model descriptors
POT_ID = dict(K=(3, 5), a=(1.0, 3.0), s=(0.08, 0.16))
PSI_ID = dict(K=(3, 5), a=(1.5, 4.0), s=(0.08, 0.16))
IC_ID = dict(n=(1, 3), sig=(0.05, 0.10))
POR_TRAIN, POR_LOW, ELL = (0.3, 0.7), (0.15, 0.3), (2.0, 10.0)
COND = dict(a=(1.0, 1.0), b025=(0.25, 1.0), b05=(0.5, 1.0), b2=(2.0, 1.0), b4=(4.0, 1.0), c2=(1.0, 2.0), c4=(1.0, 4.0),
            c8=(1.0, 8.0), d=(1.0, 1.0))   # (T, amp)
TEST_SETS = ['a', 'b025', 'b05', 'b2', 'b4', 'c2', 'c4', 'c8', 'd', 'e64', 'ecloud']
DISC_OF = dict(train='g32', e64='g64', ecloud='cloud', **{k: 'g32' for k in COND})
PHYS_EVERY = 10
STEP = dict(dt_max=DT / 8, theta0=0.9, theta1=1.0, max_sub=20000, t_tol=1e-9)
CLOUD_N, CLOUD_RMIN, CLOUD_SEED, RASTER = 1024, 0.45 / 32, 4242, 512


# ------------------------------------------------------------------------------------------ fields (meshfree)
def smooth(x, sigma):
    """periodic Gaussian filter of (..., NF, NF), sigma in fine pixels"""
    f = torch.fft.fftfreq(x.shape[-1], device=x.device, dtype=x.dtype)
    g = torch.exp(-0.5 * (2 * math.pi * sigma) ** 2 * (f[:, None] ** 2 + f[None, :] ** 2))
    return torch.fft.ifft2(torch.fft.fft2(x) * g).real


def kc(e):
    return e ** 3 / ((1 - e) ** 2 + KC_DELTA)


def truth_mobility_raw(chi):
    """chi (Nm, NF, NF) pore indicator -> unnormalised pixel mobilities (Nm, 2, NF, NF) along x (columns) and y (rows)"""
    e = smooth(chi, SIG_T).clamp(1e-4, 1.0)
    tx = (smooth(chi * chi.roll(-1, -1), SIG_T) / e).clamp(0, 1)
    ty = (smooth(chi * chi.roll(-1, -2), SIG_T) / e).clamp(0, 1)
    gate = 0.1 + 0.9 * chi
    return torch.stack([kc(e) * (0.05 + 0.95 * tx ** 2) * gate, kc(e) * (0.05 + 0.95 * ty ** 2) * gate], 1)


def harm(a, b):
    return 2 * a * b / (a + b)


def fine_edge_mobility(M):
    """pixel mobilities (Nm, 2, NF, NF) -> mobility of every fine pixel face (Nm, 2, NF, NF), harmonic mean"""
    return torch.stack([harm(M[:, 0], M[:, 0].roll(-1, -1)), harm(M[:, 1], M[:, 1].roll(-1, -2))], 1)


def descriptor_fields(chi):
    """model descriptors (Nm, 6, NF, NF): porosity (sigma 2), porosity (sigma 4), connected-pair fractions x, y at lag 1
    and lag 2 (sigma 2, divided by the porosity)."""
    e = smooth(chi, SIG_M).clamp(1e-4, 1.0)
    e2 = smooth(chi, SIG_M2).clamp(0, 1)
    cx = (smooth(chi * chi.roll(-1, -1), SIG_M) / e).clamp(0, 1)
    cy = (smooth(chi * chi.roll(-1, -2), SIG_M) / e).clamp(0, 1)
    c2x = (smooth(chi * chi.roll(-1, -1) * chi.roll(-2, -1), SIG_M) / e).clamp(0, 1)
    c2y = (smooth(chi * chi.roll(-1, -2) * chi.roll(-2, -2), SIG_M) / e).clamp(0, 1)
    return torch.stack([e.clamp(0, 1), e2, cx, cy, c2x, c2y], 1)


def interp(F, pts):
    """periodic bilinear interpolation of fields F (Nm, C, NF, NF) (pixel centres at (k + 0.5) / NF) at pts (P, 2) = (x, y)
    -> (Nm, C, P)"""
    n = F.shape[-1]
    u = pts[:, 0] * n - 0.5; v = pts[:, 1] * n - 0.5
    i0 = torch.floor(u); j0 = torch.floor(v)
    fu, fv = (u - i0).to(F.dtype), (v - j0).to(F.dtype)
    i0 = i0.long() % n; j0 = j0.long() % n; i1 = (i0 + 1) % n; j1 = (j0 + 1) % n
    g = lambda j, i: F[..., j, i]
    return (g(j0, i0) * (1 - fu) * (1 - fv) + g(j0, i1) * fu * (1 - fv) + g(j1, i0) * (1 - fu) * fv + g(j1, i1) * fu * fv)


def nearest(F, pts):
    n = F.shape[-1]
    i = torch.floor(pts[:, 0] * n).long() % n; j = torch.floor(pts[:, 1] * n).long() % n
    return F[..., j, i]


def mix_dir(Dsc, n2):
    """node-frame descriptors (..., 6, P) and squared edge direction n2 (P, 2) -> edge-frame (..., 6, P):
    [eps, eps4, c_par, c_perp, c2_par, c2_perp]"""
    a, b = n2[:, 0], n2[:, 1]
    e, e2, cx, cy, c2x, c2y = Dsc.unbind(-2)
    return torch.stack([e, e2, a * cx + b * cy, b * cx + a * cy, a * c2x + b * c2y, b * c2x + a * c2y], -2)


def edge_inputs(desc, geom):
    """per-medium edge descriptors (Nm, H, 12) in [-1, 1] for a discretisation"""
    pts, E, dv = geom['pts'], geom['edges'], geom['dvec']
    n2 = dv ** 2 / (dv ** 2).sum(-1, keepdim=True)
    mid = (pts[E[:, 0]] + 0.5 * dv) % 1.0
    dm = mix_dir(interp(desc, mid), n2)
    dn = interp(desc, pts)
    de = 0.5 * (mix_dir(dn[..., E[:, 0]], n2) + mix_dir(dn[..., E[:, 1]], n2))
    return (2 * torch.cat([dm, de], 1) - 1).transpose(1, 2).contiguous()


def true_edge_mobility(Mpix, geom, ns=16):
    """the true pixel mobility along every edge segment, direction-weighted, harmonic mean over ns samples (Nm, H)"""
    pts, E, dv = geom['pts'], geom['edges'], geom['dvec']
    n2 = dv ** 2 / (dv ** 2).sum(-1, keepdim=True)
    inv = 0
    for s in (torch.arange(ns, dtype=pts.dtype, device=pts.device) + 0.5) / ns:
        x = (pts[E[:, 0]] + s * dv) % 1.0
        m = nearest(Mpix, x)                                   # (Nm, 2, H)
        inv = inv + 1.0 / (n2[:, 0] * m[:, 0] + n2[:, 1] * m[:, 1])
    return ns / inv


# U, psi, rho_0: sums of periodic Gaussians, stored as parameters (B, K, 4) = (cx, cy, a, s); a = 0 pads
def sample_gauss(rng, spec, K=5):
    k = rng.integers(spec['K'][0], spec['K'][1] + 1)
    out = np.zeros((K, 4)); out[:, 3] = 1.0
    for i in range(k):
        out[i] = (rng.uniform(), rng.uniform(), rng.choice([-1, 1]) * rng.uniform(*spec['a']), rng.uniform(*spec['s']))
    return out


def sample_ic(rng, K=3):
    n = rng.integers(IC_ID['n'][0], IC_ID['n'][1] + 1)
    w = rng.uniform(0.5, 1.5, n); w /= w.sum()
    out = np.zeros((K, 4)); out[:, 3] = 1.0
    for i in range(n):
        out[i] = (rng.uniform(), rng.uniform(), w[i], rng.uniform(*IC_ID['sig']))
    return out


def gauss_eval(par, pts, grad=False):
    """par (B, K, 4), pts (P, 2) -> values (B, P) or gradients (B, P, 2)"""
    cx, cy, a, s = (par[..., i][:, :, None] for i in range(4))
    X, Y = pts[:, 0][None, None], pts[:, 1][None, None]
    d2 = (torch.sin(math.pi * (X - cx)) ** 2 + torch.sin(math.pi * (Y - cy)) ** 2) / math.pi ** 2
    g = a * torch.exp(-d2 / (2 * s * s))
    if not grad:
        return g.sum(1)
    fac = -g / (2 * s * s)
    return torch.stack([(fac * torch.sin(2 * math.pi * (X - cx)) / math.pi).sum(1), (fac * torch.sin(2 * math.pi * (Y - cy)) / math.pi).sum(1)], -1)


def fine_centres(dev, n=NF):
    c = (torch.arange(n, dtype=torch.float64, device=dev) + 0.5) / n
    Y, X = torch.meshgrid(c, c, indexing='ij')
    return torch.stack([X.reshape(-1), Y.reshape(-1)], 1)


def line_integrals(ppar, geom, nq=3):
    """int_e F . dl, F = (d_y psi, -d_x psi), Gauss-Legendre with nq points: (B, H)"""
    x, w = np.polynomial.legendre.leggauss(nq)
    pts, E, dv = geom['pts'], geom['edges'], geom['dvec']
    out = 0
    for xi, wi in zip(x, w):
        q = (pts[E[:, 0]] + 0.5 * (xi + 1) * dv) % 1.0
        g = gauss_eval(ppar, q, grad=True)                      # (B, H, 2)
        out = out + 0.5 * wi * (g[..., 1] * dv[:, 0] - g[..., 0] * dv[:, 1])
    return out


# ------------------------------------------------------------------------------------------ discretisations
def grid_geom(n, dev='cpu'):
    from .complex import grid_edge_list
    c = (torch.arange(n, dtype=torch.float64) + 0.5) / n
    Y, X = torch.meshgrid(c, c, indexing='ij')
    pts = torch.stack([X.reshape(-1), Y.reshape(-1)], 1)
    E = torch.as_tensor(grid_edge_list(n))
    V = n * n
    dv = torch.zeros(2 * V, 2, dtype=torch.float64); dv[:V, 0] = 1.0 / n; dv[V:, 1] = 1.0 / n
    idx = np.arange(V).reshape(n, n)
    v00, v10, v11, v01 = idx.ravel(), np.roll(idx, -1, 1).ravel(), np.roll(np.roll(idx, -1, 0), -1, 1).ravel(), np.roll(idx, -1, 0).ravel()
    tris = np.concatenate([np.stack([v00, v10, v11], 1), np.stack([v00, v11, v01], 1)])      # CCW triangles of the 2-complex
    g = dict(kind='grid', n=n, pts=pts, edges=E, dvec=dv, w=torch.full((2 * V,), float(n * n), dtype=torch.float64),
             logV=torch.zeros(V, dtype=torch.float64), dens=torch.ones(V, dtype=torch.float64), faces=tris)
    return {k: (v.to(dev) if torch.is_tensor(v) else v) for k, v in g.items()}


def cloud_geom(dev='cpu', V=CLOUD_N, rmin=CLOUD_RMIN, seed=CLOUD_SEED):
    """random point cloud on the unit torus (dart throwing, minimum distance rmin), periodic Voronoi cells (areas, face
    lengths), the dual Delaunay triangles, and the raster map of RASTER^2 sub-pixels to cells for the projection."""
    from scipy.spatial import Voronoi, Delaunay, cKDTree
    rng = np.random.default_rng(seed)
    P = []
    while len(P) < V:
        q = rng.random(2)
        if P:
            d = np.abs(np.asarray(P) - q); d = np.minimum(d, 1 - d)
            if (d ** 2).sum(1).min() < rmin ** 2:
                continue
        P.append(q)
    P = np.asarray(P)
    offs = [(dx, dy) for dx in (-1, 0, 1) for dy in (-1, 0, 1)]
    tiled = np.concatenate([P + np.array(o) for o in offs])
    cen = offs.index((0, 0))
    vor = Voronoi(tiled)
    area = np.zeros(V)
    for i in range(V):
        reg = vor.regions[vor.point_region[cen * V + i]]
        poly = vor.vertices[reg]
        c0 = poly.mean(0); ang = np.arctan2(poly[:, 1] - c0[1], poly[:, 0] - c0[0]); poly = poly[np.argsort(ang)]
        area[i] = 0.5 * abs(np.dot(poly[:, 0], np.roll(poly[:, 1], -1)) - np.dot(poly[:, 1], np.roll(poly[:, 0], -1)))
    edges = {}
    for (p, q), rv in zip(vor.ridge_points, vor.ridge_vertices):
        if p // V != cen and q // V != cen:
            continue
        u, w_ = (p, q) if p // V == cen else (q, p)
        i, j = u % V, w_ % V
        disp = tiled[w_] - tiled[u]
        if i > j:
            i, j, disp = j, i, -disp
        key = (i, j)
        if key in edges or -1 in rv:
            continue
        edges[key] = (disp, np.linalg.norm(vor.vertices[rv[0]] - vor.vertices[rv[1]]))
    keys = sorted(edges)
    E = np.array(keys); dv = np.array([edges[k][0] for k in keys]); sig = np.array([edges[k][1] for k in keys])
    dist = np.linalg.norm(dv, axis=1)
    w = sig / (dist * np.sqrt(area[E[:, 0]] * area[E[:, 1]]))
    tri = Delaunay(tiled)
    S = tri.simplices
    cent = tiled[S].mean(1)
    keep = (cent >= 0).all(1) & (cent < 1).all(1)
    S = S[keep]
    a = tiled[S]
    orient = (a[:, 1, 0] - a[:, 0, 0]) * (a[:, 2, 1] - a[:, 0, 1]) - (a[:, 1, 1] - a[:, 0, 1]) * (a[:, 2, 0] - a[:, 0, 0])
    S = np.where(orient[:, None] < 0, S[:, [0, 2, 1]], S) % V
    c = (np.arange(RASTER) + 0.5) / RASTER
    Yr, Xr = np.meshgrid(c, c, indexing='ij')
    lab = cKDTree(P, boxsize=1.0).query(np.stack([Xr.ravel(), Yr.ravel()], 1))[1]
    fine_idx = ((np.arange(RASTER)[:, None] * NF // RASTER) * NF + (np.arange(RASTER)[None, :] * NF // RASTER)).ravel()
    Vr = np.bincount(lab, minlength=V) / RASTER ** 2
    g = dict(kind='cloud', n=None, pts=torch.as_tensor(P), edges=torch.as_tensor(E), dvec=torch.as_tensor(dv),
             w=torch.as_tensor(w), logV=torch.as_tensor(np.log(area * V)), dens=torch.as_tensor(1.0 / (V * Vr)),
             faces=S, lab=torch.as_tensor(lab), fine_idx=torch.as_tensor(fine_idx), area=torch.as_tensor(area),
             area_raster=torch.as_tensor(Vr))
    return {k: (v.to(dev) if torch.is_tensor(v) else v) for k, v in g.items()}


def geometry(name, dev='cpu'):
    return grid_geom(32, dev) if name == 'g32' else grid_geom(64, dev) if name == 'g64' else cloud_geom(dev)


def projector(geom):
    """fine density (B, NF, NF) -> model state on the discretisation (B, V): cell averages on grids, cell mass / mean
    cell volume on the point cloud"""
    if geom['kind'] == 'grid':
        f = NF // geom['n']
        return lambda r: r.reshape(r.shape[0], geom['n'], f, geom['n'], f).mean((2, 4)).reshape(r.shape[0], -1)
    lab, fi = geom['lab'], geom['fine_idx']
    V = geom['pts'].shape[0]

    def proj(r):
        rf = r.reshape(r.shape[0], -1)[:, fi.to(r.device)]
        m = torch.zeros(r.shape[0], V, dtype=r.dtype, device=r.device).index_add_(1, lab.to(r.device), rf)
        return m * (V / RASTER ** 2)
    return proj


# ------------------------------------------------------------------------------------------ reference solver
def head_e(u):
    return torch.stack([u.roll(-1, -1), u.roll(-1, -2)], -3)


def tail_e(u):
    return torch.stack([u, u], -3)


def div_T(f):
    return (f[..., 0, :, :].roll(1, -1) - f[..., 0, :, :]) + (f[..., 1, :, :].roll(1, -2) - f[..., 1, :, :])


def dual_dpsi(P):
    return torch.stack([P - P.roll(1, -2), -(P - P.roll(1, -1))], -3)


def ref_flux(rho, kk, phi, A, crowd):
    x = 0.5 * (tail_e(phi) - head_e(phi) + A)
    k = kk / (1 + 0.5 * (tail_e(rho) + head_e(rho)) / RHO_S) if crowd else kk
    J = k * (tail_e(rho) * torch.exp(x) - head_e(rho) * torch.exp(-x))
    return div_T(J), J


def escape_max(kk, phi, A):
    x = 0.5 * (tail_e(phi) - head_e(phi) + A)
    o, i = kk * torch.exp(x), kk * torch.exp(-x)
    return float((o[:, 0] + o[:, 1] + i[:, 0].roll(1, -1) + i[:, 1].roll(1, -2)).max())


@torch.no_grad()
def simulate(rho0, Me, U, dpsi, T, amp, crowd, nsteps, projs, ns=None):
    """fine finite volumes, RK4 float64.  rho0, U (B, NF, NF), Me (B, 2, NF, NF) face mobilities, dpsi (B, 2, NF, NF).
    Returns ([projected trajectories (B, nsteps+1, V_p)], final fine flux (B, 2, NF, NF), final fine density, ns)."""
    T = T.view(-1, 1, 1); amp = amp.view(-1, 1, 1, 1)
    kk = D * NF * NF * T[..., None] * Me
    phi = U / T
    A = amp * dpsi / T[..., None]
    ns = ns or int(math.ceil(DT * escape_max(kk, phi, A)))
    dt = DT / ns
    rho = rho0.clone()
    outs = [[p(rho)] for p in projs]
    f = lambda r: ref_flux(r, kk, phi, A, crowd)[0]
    for _ in range(nsteps):
        for _ in range(ns):
            k1 = f(rho); k2 = f(rho + 0.5 * dt * k1); k3 = f(rho + 0.5 * dt * k2); k4 = f(rho + dt * k3)
            rho = rho + dt / 6 * (k1 + 2 * k2 + 2 * k3 + k4)
        for o, p in zip(outs, projs):
            o.append(p(rho))
    return [torch.stack(o, 1) for o in outs], ref_flux(rho, kk, phi, A, crowd)[1], rho, ns


def coarse_flux32(J):
    """fine face fluxes (B, 2, NF, NF) -> fluxes through the 32 x 32 cell faces in the model's units (B, 2 * 32^2)"""
    B, f = J.shape[0], NF // N
    Jh = J[:, 0].reshape(B, N, f, N, f)[..., f - 1].sum(2) / f ** 2
    Jv = J[:, 1].reshape(B, N, f, N, f)[:, :, f - 1].sum(-1) / f ** 2
    return torch.stack([Jh, Jv], 1).reshape(B, -1)


def sample_medium(rng, por):
    p, ell = rng.uniform(*por), rng.uniform(*ELL)
    k = np.fft.fftfreq(NF)
    kx, ky = np.meshgrid(k, k)
    spec = np.exp(-0.5 * (2 * np.pi) ** 2 * (kx ** 2 + ky ** 2) * ell ** 2)
    f = np.real(np.fft.ifft2(np.fft.fft2(rng.standard_normal((NF, NF))) * np.sqrt(spec)))
    return (f > np.quantile(f, 1 - p)).astype(np.uint8)


def make_pool(seed, n_media, n_per, por):
    rng = np.random.default_rng(seed)
    pores = np.stack([sample_medium(rng, por) for _ in range(n_media)])
    U, P, R0, med = [], [], [], []
    for m in range(n_media):
        for _ in range(n_per):
            U.append(sample_gauss(rng, POT_ID)); P.append(sample_gauss(rng, PSI_ID)); R0.append(sample_ic(rng)); med.append(m)
    return dict(pore=pores, pmed=pores.mean((1, 2)), Upar=np.stack(U), Ppar=np.stack(P), Rpar=np.stack(R0), med=np.array(med))


def fine_inputs(pool, dev):
    """fine U at pixel centres, psi at pixel corners -> dual_dpsi, rho_0 at pixel centres (normalised to mean 1)"""
    T64 = lambda x: torch.as_tensor(x, dtype=torch.float64, device=dev)
    c = fine_centres(dev)
    U = gauss_eval(T64(pool['Upar']), c).reshape(-1, NF, NF)
    Pc = gauss_eval(T64(pool['Ppar']), c + 0.5 / NF).reshape(-1, NF, NF)
    r = gauss_eval(T64(pool['Rpar']), c).reshape(-1, NF, NF)
    r = r / r.mean((-1, -2), keepdim=True)
    eps = 1e-3
    return U, dual_dpsi(Pc), (r + eps) / (1 + eps)


def generate(variant, device='cuda', out=OUT):
    """reference data of one variant (H1 or H1p) on the fine grid, projected to the evaluation discretisations."""
    os.makedirs(os.path.join(out, 'data'), exist_ok=True)
    crowd = variant == 'H1p'
    t0 = time.time()
    pools = dict(train=make_pool(7001, 64, 4, POR_TRAIN), test=make_pool(7002, 16, 4, POR_TRAIN), low=make_pool(7003, 16, 4, POR_LOW))
    chi = {k: torch.as_tensor(v['pore'], dtype=torch.float64, device=device) for k, v in pools.items()}
    Mraw = {k: truth_mobility_raw(v) for k, v in chi.items()}
    C = 1.0 / float(fine_edge_mobility(Mraw['train']).mean())
    Mpix = {k: C * v + M_FLOOR for k, v in Mraw.items()}
    Mface = {k: fine_edge_mobility(v) for k, v in Mpix.items()}
    desc = {k: descriptor_fields(v) for k, v in chi.items()}
    geoms = {g: geometry(g, device) for g in ('g32', 'g64', 'cloud')}
    projs = {g: projector(geoms[g]) for g in geoms}
    info = dict(variant=variant, C=C, D=D, DT=DT, RHO_S=RHO_S, crowd=crowd, KC_DELTA=KC_DELTA, M_FLOOR=M_FLOOR,
                sig=dict(truth=SIG_T, model=SIG_M, model2=SIG_M2), cond=COND,
                cloud=dict(V=CLOUD_N, H=int(geoms['cloud']['edges'].shape[0]), rmin=CLOUD_RMIN,
                           area_cv=float(geoms['cloud']['area'].std() / geoms['cloud']['area'].mean()),
                           raster_area_relerr_max=float(((geoms['cloud']['area_raster'] - geoms['cloud']['area']).abs() / geoms['cloud']['area']).max())))
    for k in pools:
        m = Mface[k]
        info[f'M_{k}'] = dict(mean=float(m.mean()), median=float(m.median()), max=float(m.max()), frac_below_1e2=float((m < 1e-2).double().mean()),
                             porosity_min=float(pools[k]['pmed'].min()), porosity_max=float(pools[k]['pmed'].max()))
    fin = {k: fine_inputs(v, device) for k, v in pools.items()}
    sets = dict(train=('train', 1.0, 1.0, T_TRAIN, ['g32']))
    for s, (T, amp) in COND.items():
        sets[s] = ('low' if s == 'd' else 'test', T, amp, T_TEST, ['g32', 'g64', 'cloud'] if s == 'a' else ['g32'])
    data = dict(meta=dict(info, pools={k: dict(pore=torch.as_tensor(v['pore']), pmed=v['pmed'].tolist()) for k, v in pools.items()},
                          desc={k: v.float().cpu() for k, v in desc.items()}, Mpix={k: v.float().cpu() for k, v in Mpix.items()}))
    for s, (pool, T, amp, nst, discs) in sets.items():
        p = pools[pool]
        med = torch.as_tensor(p['med'], device=device)
        U, dpsi, r0 = fin[pool]
        B = U.shape[0]
        Tt = torch.full((B,), T, dtype=torch.float64, device=device); At = torch.full((B,), amp, dtype=torch.float64, device=device)
        t1 = time.time()
        Rs, J, rf, ns = simulate(r0, Mface[pool][med], U, dpsi, Tt, At, crowd, nst, [projs[g] for g in discs])
        rec = dict(ns_per_DT=ns, sim_time_s=time.time() - t1, min_rho=float(Rs[0].min()),
                   mass_drift=float((Rs[0].mean(-1) - Rs[0][:, :1].mean(-1)).abs().max()))
        if s in ('b2', 'c4', 'a'):
            R2, _, _, _ = simulate(r0[:8], Mface[pool][med[:8]], U[:8], dpsi[:8], Tt[:8], At[:8], crowd, nst, [projs['g32']], ns=2 * ns)
            rec['dt_halving_relerr_end_max'] = float((torch.linalg.vector_norm(Rs[0][:8, -1] - R2[0][:, -1], dim=-1)
                                                      / torch.linalg.vector_norm(R2[0][:, -1], dim=-1)).max())
        base = dict(Upar=torch.as_tensor(p['Upar']).float(), Ppar=torch.as_tensor(p['Ppar']).float(), med=med.cpu(), pool=pool,
                    T=Tt.float().cpu(), amp=At.float().cpu())
        for g, R in zip(discs, Rs):
            name = s if g == 'g32' else ('e64' if g == 'g64' else 'ecloud')
            data[name] = dict(base, rho=R.float().cpu(), disc=g)
            if g == 'g32':
                data[name]['J_end'] = coarse_flux32(J).float().cpu()
        data['meta'][f'sim_{s}'] = rec
        print(variant, s, json.dumps(rec), flush=True)
    data['meta']['gen_time_s'] = time.time() - t0
    torch.save(data, os.path.join(out, 'data', f'{variant}.pt'))
    gi = {k: v for k, v in data['meta'].items() if k not in ('pools', 'desc', 'Mpix')}
    json.dump(gi, open(os.path.join(out, 'data', f'gen_info_{variant}.json'), 'w'), indent=1, default=float)
    return gi


# ------------------------------------------------------------------------------------------ data loading
class Disc:
    """a discretisation with its geometry, cached per-medium descriptors and true edge mobilities"""
    def __init__(s, name, meta, device, geom=None):
        s.name, s.dev = name, device
        s.g = geom if geom is not None else geometry(name, device)
        s.V, s.H = s.g['pts'].shape[0], s.g['edges'].shape[0]
        s.ein, s.mtrue, s.ndesc = {}, {}, {}
        for pool in (meta or {}).get('desc', {}):
            dsc = meta['desc'][pool].double().to(device)
            s.ein[pool] = edge_inputs(dsc, s.g).float()
            s.ndesc[pool] = interp(dsc, s.g['pts']).float()                       # (Nm, 6, V)
            s.mtrue[pool] = true_edge_mobility(meta['Mpix'][pool].double().to(device), s.g).float()
        s._hg = None
        s._lmh = None

    def minnorm_flux(s, cdot):
        """the minimum-norm edge flux with S J = cdot: J = G L^+ cdot (G the node-to-edge difference head - tail, L = G^T G)"""
        if getattr(s, '_Lp', None) is None:
            E = s.g['edges'].cpu()
            L = torch.zeros(s.V, s.V, dtype=torch.float64)
            L.index_put_((E[:, 0], E[:, 0]), torch.ones(s.H, dtype=torch.float64), accumulate=True)
            L.index_put_((E[:, 1], E[:, 1]), torch.ones(s.H, dtype=torch.float64), accumulate=True)
            L.index_put_((E[:, 0], E[:, 1]), -torch.ones(s.H, dtype=torch.float64), accumulate=True)
            L.index_put_((E[:, 1], E[:, 0]), -torch.ones(s.H, dtype=torch.float64), accumulate=True)
            s._Lp = torch.linalg.pinv(L, hermitian=True)
        Lp = s._Lp.to(cdot.device)
        phi = cdot.double() @ Lp
        E = s.g['edges'].to(cdot.device)
        return (phi[:, E[:, 1]] - phi[:, E[:, 0]]).to(cdot.dtype)

    def hg(s):
        """transport hypergraph on the edges (GridOps edge order on grids).  The cycle basis is not used here, so it is
        set empty instead of computing the dense null space; the one moiety is the total mass."""
        if s._hg is None:
            from .complex import Hypergraph
            s._hg = Hypergraph.from_graph(s.g['edges'].cpu().numpy(), s.V, cycle_basis=np.zeros((s.H, 0)))
            s._hg._L = np.full((s.V, 1), 1.0 / math.sqrt(s.V))
        return s._hg


def load(variant, device='cpu', out=OUT, sets=None):
    """dict(set -> dict(c (B, T+1, V), drive, disc, J_end, med, pool), meta, discs)"""
    data = torch.load(os.path.join(out, 'data', f'{variant}.pt'))
    meta = data['meta']
    discs = {}
    res = dict(meta=meta, discs=discs)
    for s in sets or ['train'] + TEST_SETS:
        d = data[s]
        dn = d['disc']
        if dn not in discs:
            discs[dn] = Disc(dn, meta, device)
        G = discs[dn]
        B = d['rho'].shape[0]
        Up, Pp = d['Upar'].double().to(device), d['Ppar'].double().to(device)
        med = d['med'].to(device)
        drive = dict(U=gauss_eval(Up, G.g['pts']).float(), psin=gauss_eval(Pp, G.g['pts']).float(), line=line_integrals(Pp, G.g).float(),
                     T=d['T'].to(device), amp=d['amp'].to(device), Ppar=Pp, ein=G.ein[d['pool']][med], ndesc=G.ndesc[d['pool']][med],
                     pmed=torch.as_tensor(meta['pools'][d['pool']]['pmed'], dtype=torch.float32, device=device)[med])
        res[s] = dict(c=d['rho'].to(device), drive=drive, disc=dn, J_end=d.get('J_end'), med=d['med'], pool=d['pool'])
    return res


# ------------------------------------------------------------------------------------------ HyperME (wired)
class EdgeKinetics(nn.Module):
    """k_e = D T w_e exp(kappa_e).  mode 'edge': kappa = core + MLP(edge descriptors) (pointwise, no message passing);
    'const': kappa = core + f(medium porosity); 'zero': kappa = core; 'oracle': kappa = log of the true edge mobility
    (feats['mtrue'])."""
    rho_dependent = False
    split = True
    chem_prefactor = False

    def __init__(s, mode='edge', hidden=64):
        super().__init__()
        from .model import MLP
        s.mode = mode
        s.core = nn.Parameter(torch.zeros(()))
        if mode == 'edge':
            s.net = MLP(12, hidden, 1, last='zero')
        elif mode == 'const':
            s.net = MLP(1, 16, 1, last='zero')

    def kappa(s, feats):
        x = feats['ein']
        if s.mode == 'oracle':
            return torch.log(feats['mtrue'])
        k = s.core.to(x.dtype).expand(x.shape[0], x.shape[1])
        if s.mode == 'edge':
            k = k + s.net(x).squeeze(-1)
        elif s.mode == 'const':
            k = k + s.net(2 * feats['pmed'][:, None] - 1)
        return k

    def forward(s, feats, hg=None, c=None):
        return D * feats['w'][None] * feats['T'][:, None] * torch.exp(s.kappa(feats))


def wired_featurizer(disc):
    w, logV = disc.g['w'], disc.g['logV']

    def feat(F):
        U = F['U']; B = U.shape[0]
        T = F['T'].to(U.dtype)
        f = dict(U=U - T[:, None] * logV.to(U.device, U.dtype)[None], T=T, line=F['amp'][:, None] * F['line'], chem_logc=None, chem_mu0=None,
                 edge_sym=torch.zeros(B, disc.H, 1, dtype=U.dtype, device=U.device), edge_ori=torch.zeros(B, disc.H, dtype=U.dtype, device=U.device),
                 node={}, ein=F['ein'], pmed=F['pmed'], w=w.to(U.device, U.dtype), raw=F)
        if 'mtrue' in F:
            f['mtrue'] = F['mtrue']
        return f
    feat.n_edge_sym = 1
    return feat


def build_wired(mode, disc):
    from .model import HyperME, WiredFreeEnergy, WiredDrive
    hg = disc.hg()
    return HyperME(hg, WiredFreeEnergy(hg, 'none'), EdgeKinetics(mode), WiredDrive(hg, harmonic=False), wired_featurizer(disc), step=dict(STEP))


# ------------------------------------------------------------------------------------------ black boxes
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
    """residual next-step map rho_{t+DT} = rho_t + FNO([rho_t, cond]) on an n x n grid (any n)."""
    name = 'fno'

    def __init__(s, norm):
        super().__init__()
        s.norm = norm
        s.net = FNO2d(1 + NC)

    def prepare(s, drive):
        from .model import Context
        cond = node_cond(drive, s.norm)
        n = int(round(cond.shape[1] ** 0.5))
        return Context(cond=cond.transpose(1, 2).reshape(cond.shape[0], NC, n, n))

    def context(s, drive):
        from .model import Context
        return drive if isinstance(drive, Context) else s.prepare(drive)

    def advance(s, c, drive, Dt, **kw):
        ctx = s.context(drive)
        n = ctx['cond'].shape[-1]
        r = c.reshape(-1, n, n)
        return (r + s.net(torch.cat([r[:, None], ctx['cond']], 1))).reshape(c.shape), 1


def gnn_featurizer(norm, disc):
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


class LMHodgeOp(nn.Module):
    """learned-metric linear Hodge operator (internal ablation; the library is imported from the package named by the
    LMH_PKG, LMH_MODEL and LMH_CONFIG environment variables).  Fixed coboundaries of the periodic 2-complex (split squares on grids, the Delaunay
    triangles of a point cloud), a learned diagonal SPD metric from local invariants and the even input columns, propagation by
    d^T H d, and a mass-weighted divergence readout (triangulated 2-complex on grids and point clouds alike): rho_{t+DT} = rho_t + mdiv(b).  Vertex inputs [rho, U, amp psi |
    T, amp, 6 descriptors], edge inputs [amp int_e F.dl | 12 edge descriptors]; T and amp are features only."""
    name = 'lm_hodge'

    def __init__(s, norm, C=64, layers=4):
        super().__init__()
        s.lib = importlib.import_module(os.environ['LMH_PKG'])
        s.Net, Cfg = getattr(s.lib, os.environ['LMH_MODEL']), getattr(s.lib, os.environ['LMH_CONFIG'])
        s.norm = norm
        s.cfg = Cfg(in_dims={0: 3 + 8, 1: 1 + 12}, even_dims={0: 8, 1: 12}, C=C, n_layers=layers, metric_type='diag',
                    readout='mdiv:1', out_dim=1)
        s.disc = None
        s.net = None

    def set_disc(s, disc, device):
        """build (or reuse) the 2-complex of a discretisation; the network has no size-dependent parameters"""
        if disc._lmh is None:
            g = disc.g
            pos = g['pts'].cpu().numpy()
            K = s.lib.CochainComplex.from_triangles(pos, g['faces'], period=(1.0, 1.0), validate=False).to(device)
            e = K.cells[1].long()
            dv = torch.as_tensor(pos[e[:, 1].cpu()] - pos[e[:, 0].cpu()])
            dv = dv - torch.round(dv)
            geo = dict(pts=g['pts'], edges=e.to(g['pts'].device), dvec=dv.to(g['pts'].device))
            disc._lmh = dict(K=K, geo=geo)
        s.disc = disc
        if s.net is None:
            s.net = s.Net(s.cfg, disc._lmh['K'].geo_dims).to(device)
            s.net.record_diagnostics = False
        return s

    def prepare(s, drive):
        from .model import Context
        L = s.disc._lmh
        cond = node_cond(drive, s.norm)                                         # (B, V, 10)
        x0 = cond.permute(1, 0, 2)                                              # (V, B, 10)
        Pp = drive['Ppar']
        line = line_integrals(Pp, L['geo']).float() * drive['amp'][:, None] / s.norm['line']        # (B, H')
        ein = drive['ein_lm']                                                   # (B, H', 12)
        x1 = torch.cat([line[..., None], ein], -1).permute(1, 0, 2)
        return Context(x0=x0, x1=x1)

    def context(s, drive):
        from .model import Context
        return drive if isinstance(drive, Context) else s.prepare(drive)

    def advance(s, c, drive, Dt, **kw):
        ctx = s.context(drive)
        r = c.T[..., None]                                                      # (V, B, 1)
        x0 = torch.cat([r, ctx['x0']], -1)
        y = s.net({0: x0, 1: ctx['x1']}, s.disc._lmh['K'])                     # (V, B, 1)
        return (r + y)[..., 0].T.contiguous(), 1


def lm_drive_extras(sd, disc, meta, device):
    """per-set inputs of the LM-Hodge operator on its own edge list: Gaussian parameters of psi and edge descriptors"""
    L = disc._lmh
    pool = sd['pool']
    key = ('lm_ein', pool)
    if key not in disc.ein:
        dsc = meta['desc'][pool].double().to(device)
        disc.ein[key] = edge_inputs(dsc, L['geo']).float()
    sd['drive']['ein_lm'] = disc.ein[key][sd['med'].to(device)]


class UDEClosure(nn.Module):
    """UDE closure (internal ablation): the same discretisation, conservative assembly dc/dt = S J and meshfree edge
    descriptors as the wired HyperME, but a freely learned antisymmetric edge flux
        J_e = D w_e [ g(rho_i, rho_j, U_j - U_i, L_e, T, amp, desc_e) - g(rho_j, rho_i, U_i - U_j, -L_e, T, amp, desc_e) ],
    rho the cell densities, L_e = amp int_e F.dl (normalised), g an MLP of the same size as kappa_theta, zero-initialised
    last layer.  No exponential-family rates, no local detailed balance; T and amp are features.  Integrated with the
    same adaptive SSP-RK2 substeps (one-way parts relu(J), relu(-J); the step control uses out-flow / (c + 0.01), so
    positivity is not guaranteed)."""
    name = 'ude'

    def __init__(s, disc, norm, hidden=64):
        super().__init__()
        from .model import MLP
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
        from .model import Context
        return Context(feats=s.featurizer(drive))

    def context(s, drive):
        from .model import Context
        return drive if isinstance(drive, Context) else s.prepare(drive)

    def flux(s, c, drive):
        f = s.context(drive)['feats']
        rho = c * f['invV'][None]
        ri, rj = rho[:, f['tail']], rho[:, f['head']]
        g = lambda a, b, du, L: s.net(torch.cat([torch.stack([a, b, du, L, f['T'], f['amp']], -1), f['ein']], -1)).squeeze(-1)
        return D * f['w'][None] * (g(ri, rj, f['dU'], f['L']) - g(rj, ri, -f['dU'], -f['L']))

    def rates(s, c, drive):
        J = s.flux(c, drive)
        a = J.clamp_min(0)
        return a, a - J                     # one-way parts with a - b = J exactly (also in the gradient at J = 0)

    def forward(s, c, drive):
        return s.hg.apply_S(s.flux(c, drive))

    def escape_ratio(s, c, a, b):
        return (s.hg.consumption(a, b) / (c.clamp_min(0) + 0.01)).amax(-1)

    def advance(s, c, drive, Dt, **kw):
        from .integrators import positive_ssp_rk2
        return positive_ssp_rk2(s, c, s.context(drive), Dt, **s.step)


def build(model, meta, disc):
    norm = meta['norm']
    if model.startswith('wired'):
        mode = dict(wired_k='edge', wired_const='const', wired_untrained='zero', wired_oracle='oracle')[model]
        return build_wired(mode, disc)
    if model == 'fno':
        return FNOOp(norm)
    if model == 'lm_hodge':
        return LMHodgeOp(norm)
    if model == 'ude':
        return UDEClosure(disc, norm)
    from . import baselines as BL
    hg = disc.hg()
    feat = gnn_featurizer(norm, disc)
    if model == 'gnn_ode':
        return BL.GNNODE(hg, 1 + NC, 4, feat, hidden=64, layers=3, scale=KREF)
    if model == 'flux_gnn':
        return BL.FluxGNN(hg, 1 + NC, 4, feat, hidden=64, layers=3, scale=KREF, step=dict(STEP))
    raise ValueError(model)


def rebind(model_name, model, meta, disc, device):
    """the same trained model on another discretisation (no retraining): rebuild the graph-dependent parts, keep the
    parameters.  Returns None when the model cannot run there."""
    if model_name == 'fno':
        return model if disc.g['kind'] == 'grid' else None
    if model_name == 'lm_hodge':
        return model.set_disc(disc, device)
    new = build(model_name, meta, disc).to(device)
    sd = {k: v for k, v in model.state_dict().items() if not any(b in k for b in ('p_h', 'p_m', 'p_v', 'm_h', 'm_m', 'm_v'))}
    missing, unexpected = new.load_state_dict(sd, strict=False)
    assert not unexpected and all(any(b in k for b in ('p_h', 'p_m', 'p_v', 'm_h', 'm_m', 'm_v')) for k in missing), (missing, unexpected)
    return new.eval()


# ------------------------------------------------------------------------------------------ metrics
def relerr_traj(R, Y):
    B, T1 = Y.shape[:2]
    e = torch.linalg.vector_norm((R - Y).reshape(B, T1, -1), dim=-1) / torch.linalg.vector_norm(Y.reshape(B, T1, -1), dim=-1)
    return torch.where(torch.isfinite(e), e, torch.full_like(e, float('inf')))


def rollout_metrics(R, Y, dens=None):
    """R, Y (B, T+1, V) states on the CPU; errors on densities (state x dens), mass on states"""
    R, Y = R.double(), Y.double()
    w = 1.0 if dens is None else dens.double().cpu()[None, None]
    e = relerr_traj(R * w, Y * w).numpy()
    B = R.shape[0]
    fin = torch.isfinite(R.reshape(B, -1)).all(1)
    m0 = Y[:, 0].sum(-1)
    drift = (R.sum(-1) - m0[:, None]).abs() / m0[:, None]
    return dict(err_T=float(np.median(e[:, 20])), err_10T=float(np.median(e[:, -1])), err_10T_mean=float(np.mean(e[:, -1])),
                err_10T_per_traj=e[:, -1].tolist(), n_blown=int((~fin).sum()),
                n_neg_traj=int((torch.nan_to_num(R, nan=0.0) < 0).reshape(B, -1).any(1).sum()),
                min_value=float(np.nanmin(R.numpy())), mass_drift_max=float(np.nanmax(drift.numpy())))


def flux_metrics(Jm, Jt):
    Jm, Jt = Jm.double().cpu(), Jt.double().cpu()
    e = (torch.linalg.vector_norm(Jm - Jt, dim=-1) / torch.linalg.vector_norm(Jt, dim=-1)).numpy()
    cos = ((Jm * Jt).sum(-1) / (torch.linalg.vector_norm(Jm, dim=-1) * torch.linalg.vector_norm(Jt, dim=-1) + 1e-30)).numpy()
    return dict(J_relerr=float(np.median(np.where(np.isfinite(e), e, np.inf))), J_cos=float(np.nanmedian(cos)))


def pearson(x, y):
    x = x - x.mean(); y = y - y.mean()
    return float((x * y).sum() / (np.sqrt((x * x).sum() * (y * y).sum()) + 1e-30))


def rankdata(x):
    return np.argsort(np.argsort(x)).astype(np.float64)


@torch.no_grad()
def k_metrics(model, sd, disc):
    """learned mobility exp(kappa) against the true mobility averaged harmonically along every edge, on the distinct
    media of a set"""
    med = sd['med'].numpy()
    first = np.unique(med, return_index=True)[1]
    idx = torch.as_tensor(first, device=sd['c'].device)
    drive = {k: v[idx] for k, v in sd['drive'].items() if torch.is_tensor(v) and v.shape[0] == len(med)}
    feats = model.featurizer(drive)
    gh = torch.exp(model.kinetics.kappa(feats)).double().cpu().numpy().ravel()
    gt = disc.mtrue[sd['pool']][torch.as_tensor(med[first])].double().cpu().numpy().ravel()
    eps = (0.5 * (drive['ein'][..., 0] + 1)).double().cpu().numpy().ravel()
    lh, lt = np.log(gh), np.log(gt)
    op = gt > 1e-2
    out = dict(pearson_logk=pearson(lh, lt), spearman_logk=pearson(rankdata(lh), rankdata(lt)), pearson_k=pearson(gh, gt),
               relerr_k=float(np.linalg.norm(gh - gt) / np.linalg.norm(gt)), median_abs_logratio=float(np.median(np.abs(lh - lt))),
               pearson_logk_open=pearson(lh[op], lt[op]), frac_open=float(op.mean()),
               median_logratio=float(np.median(lh - lt)))
    return out, dict(g_hat=gh.astype(np.float32), g_true=gt.astype(np.float32), eps=eps.astype(np.float32))


# ------------------------------------------------------------------------------------------ physical consistency
def phys_force(c, F, disc):
    """local thermodynamic force of every edge under the given condition, f_e = mu_tail - mu_head + amp int_e F.dl / T,
    mu = log rho + U / T (rho the cell density); NaN where an endpoint density is not positive"""
    logV = disc.g['logV'].to(c.device, torch.float64)
    E = disc.g['edges'].to(c.device)
    rho = c.double() * torch.exp(-logV)[None]
    T = F['T'].double()[:, None]
    mu = torch.where(rho > 0, torch.log(rho.clamp_min(1e-300)), torch.full_like(rho, float('nan'))) + F['U'].double() / T
    return mu[:, E[:, 0]] - mu[:, E[:, 1]] + F['amp'].double()[:, None] * F['line'].double() / T


class PhysAcc:
    """accumulates, over trajectories and sampled times, the fraction of edges with J f < 0 (violated local detailed
    balance), the physical entropy production sigma = sum_e J_e f_e (fraction of samples with sigma < 0, its most
    negative value relative to the median |sigma|), and the fraction of negative cell values."""
    def __init__(s):
        s.nv = s.nval = s.nsneg = s.ns = s.nneg = s.ncell = 0
        s.smin, s.sigs = float('inf'), []

    def add(s, J, f, c):
        J = J.double()
        ok = torch.isfinite(f) & torch.isfinite(J)
        thr = 1e-9 * J.abs().nan_to_num(0).amax(-1, keepdim=True).clamp_min(1e-30)
        val = ok & (J.abs() > thr)
        s.nv += int(((J * f < 0) & val).sum()); s.nval += int(val.sum())
        sig = torch.where(ok, J * f, torch.zeros_like(J)).sum(-1)
        fin = torch.isfinite(sig)
        s.nsneg += int((sig[fin] < 0).sum()); s.ns += int(fin.sum())
        s.sigs.append(sig[fin].cpu())
        s.nneg += int((c < 0).sum()); s.ncell += c.numel()

    def result(s):
        sg = torch.cat(s.sigs) if s.sigs else torch.zeros(0)
        med = float(sg.abs().median()) if len(sg) else float('nan')
        return dict(ldb_violation_frac=s.nv / max(s.nval, 1), sigma_neg_frac=s.nsneg / max(s.ns, 1),
                    sigma_min_rel=(float(sg.min()) / med if len(sg) and med > 0 else float('nan')),
                    neg_cell_frac=s.nneg / max(s.ncell, 1))


def phys_metrics_rollout(R, F, disc, flux_fn=None, acc=None):
    """R (B, T+1, V) rollout states, F the drive of these trajectories.  flux_fn(c) -> J (the model's own edge flux);
    otherwise the minimum-norm flux of the step change (R[t+1] - R[t]) / DT (curl-free by construction)."""
    acc = acc or PhysAcc()
    dev = R.device
    for t in range(0, R.shape[1] - 1, PHYS_EVERY):
        c = R[:, t]
        J = flux_fn(c) if flux_fn is not None else disc.minnorm_flux((R[:, t + 1] - R[:, t]) / DT)
        acc.add(J, phys_force(c, F, disc), c)
    return acc


def local_sigma(c, J, F, disc):
    """cell map of the physical entropy production, half of J_e f_e to each endpoint (V,)"""
    f = phys_force(c[None], {k: (v[None] if torch.is_tensor(v) else v) for k, v in F.items()}, disc)[0]
    q = torch.nan_to_num(J.double() * f)
    E = disc.g['edges'].to(c.device)
    return (torch.zeros(disc.V, dtype=torch.float64, device=c.device).index_add_(0, E[:, 0], 0.5 * q).index_add_(0, E[:, 1], 0.5 * q))


# ------------------------------------------------------------------------------------------ train / evaluate
def gpu_ok(device):
    if str(device).startswith('cuda'):
        from .gpu_guard import require_free
        require_free(device)


def add_norm(meta, data):
    if 'norm' not in meta:
        dr = data['train']['drive']
        meta['norm'] = dict(U=(float(dr['U'].mean()), float(dr['U'].std())), psi=(float(dr['psin'].mean()), float(dr['psin'].std())),
                            line=float(dr['line'].std()))
    return meta


@torch.no_grad()
def eval_model(name, model, data, log, snap, device, sets=TEST_SETS, chunk=64):
    from .train import rollout
    res = {}
    meta = data['meta']
    for s in sets:
        sd = data[s]; Y = sd['c']
        disc = data['discs'][sd['disc']]
        m_ = rebind(name, model, meta, disc, device) if sd['disc'] != 'g32' else model
        if m_ is None:
            res[s] = None
            log(f'[{s}] N/A on {sd["disc"]}')
            continue
        if name == 'lm_hodge':
            lm_drive_extras(sd, disc, meta, device)
        t0 = time.time()
        Rs, Js, nsubs = [], [], []
        acc = PhysAcc()
        own = hasattr(m_, 'flux')
        for i in range(0, Y.shape[0], chunk):
            dr = {k: v[i:i + chunk] for k, v in sd['drive'].items()}
            R, ns, _ = rollout(m_, dr, Y[i:i + chunk, 0], Y.shape[1] - 1, DT)
            Rs.append(R.cpu()); nsubs.append(ns)
            ctx = m_.prepare(dr) if own else None
            phys_metrics_rollout(R, dr, disc, (lambda c: m_.flux(c, ctx)) if own else None, acc)
            if i == 0 and s in ('c8', 'b025', 'a'):
                j = 0; t = 100
                Jm = m_.flux(R[:, t], ctx)[j] if own else disc.minnorm_flux((R[:, t + 1] - R[:, t]) / DT)[j]
                snap[f'sig_{s}'] = local_sigma(R[j, t], Jm, {k: v[j] for k, v in dr.items() if torch.is_tensor(v)}, disc).cpu().numpy().astype(np.float32)
                snap[f'rho_{s}_t100'] = R[j, t].cpu().numpy().astype(np.float32)
            if hasattr(m_, 'flux') and sd['J_end'] is not None:
                Js.append(m_.flux(R[:, -1], dr).cpu() if torch.isfinite(R[:, -1]).all() and R[:, -1].min() > 0 else torch.full((R.shape[0], disc.H), float('nan')))
        R = torch.cat(Rs)
        m = rollout_metrics(R, Y.cpu(), disc.g['dens'])
        m['infer_time_s'] = time.time() - t0; m['substeps_per_DT'] = float(np.mean(nsubs))
        m['phys'] = dict(acc.result(), flux='own' if own else 'min-norm')
        if Js:
            m.update(flux_metrics(torch.cat(Js), sd['J_end']))
        if name.startswith('wired') and name != 'wired_oracle':
            km, kd = k_metrics(m_, sd, disc)
            m['k'] = km
            if s in ('a', 'd', 'ecloud'):
                snap.update({f'k_{s}_{k}': v for k, v in kd.items()})
        snap[f'R_{s}'] = R[:4, -1].numpy().astype(np.float32)
        res[s] = m
        log(f'[{s}] T {m["err_T"]:.4g} 10T {m["err_10T"]:.4g} blown {m["n_blown"]} neg {m["n_neg_traj"]} mass {m["mass_drift_max"]:.1e}'
            + (f' | J {m["J_relerr"]:.3g}' if 'J_relerr' in m else '')
            + (f' | k r(log) {m["k"]["pearson_logk"]:.3f} rho_s {m["k"]["spearman_logk"]:.3f} r(k) {m["k"]["pearson_k"]:.3f} relerr {m["k"]["relerr_k"]:.3f}' if 'k' in m else '')
            + f' | LDB viol {m["phys"]["ldb_violation_frac"]:.3g} sigma<0 {m["phys"]["sigma_neg_frac"]:.3g}'
            + f' | sub {m["substeps_per_DT"]:.1f} t {m["infer_time_s"]:.0f}s')
    return res


def oracle_drive(data):
    """the true edge mobility as a drive entry (for wired_oracle)"""
    for s, sd in data.items():
        if isinstance(sd, dict) and 'drive' in sd:
            disc = data['discs'][sd['disc']]
            sd['drive']['mtrue'] = disc.mtrue[sd['pool']][sd['med'].to(sd['c'].device)]


def run_train(variant, model_name, seed, iters, device, lr=1e-3, out=OUT):
    gpu_ok(device)
    torch.manual_seed(seed); np.random.seed(seed)
    data = load(variant, device, out)
    meta = add_norm(data['meta'], data)
    odir = os.path.join(out, variant); os.makedirs(odir, exist_ok=True)
    name = f'{model_name}_s{seed}'
    logf = open(os.path.join(odir, name + '.log'), 'w')

    def log(*a):
        s = ' '.join(str(x) for x in a); print(s, flush=True); logf.write(s + '\n'); logf.flush()
    if model_name == 'gfno':
        return run_gfno(variant, seed, iters, device, data, odir, name, log)
    if model_name == 'wired_oracle':
        oracle_drive(data)
    g32 = data['discs']['g32']
    model = build(model_name, meta, g32).to(device)
    if model_name == 'lm_hodge':
        model.set_disc(g32, device)
        lm_drive_extras(data['train'], g32, meta, device)
    npar = sum(p.numel() for p in model.parameters())
    log(f'{variant} {name}: params {npar} iters {iters} lr {lr} device {device}')
    info = dict(train_time_s=0.0)
    if model_name not in ('wired_untrained', 'wired_oracle') and iters > 0:
        from .train import train
        info = train(model, data['train'], iters, seed, DT, batch=8, lr=lr, kmin=4, kmax=8, log=log)
        torch.save(model.state_dict(), os.path.join(odir, name + '.pt'))
    model.eval()
    snap = {}
    ev = eval_model(model_name, model, data, log, snap, device)
    extra = {}
    if model_name.startswith('wired'):
        extra = dict(s_A=float(model.drive.s_A), core=float(model.kinetics.core))
    res = dict(variant=variant, model=model_name, seed=seed, params=npar, iters=iters, lr=lr, **info, extra=extra, sets=ev)
    json.dump(res, open(os.path.join(odir, name + '.json'), 'w'))
    np.savez_compressed(os.path.join(odir, name + '_snap.npz'), **snap)
    return res


def grid_cond(drive, norm):
    c = node_cond(drive, norm)
    n = int(round(c.shape[1] ** 0.5))
    return c.transpose(1, 2).reshape(c.shape[0], NC, n, n).float()


def run_gfno(variant, seed, iters, device, data, odir, name, log):
    sys.path.insert(0, os.path.join(ROOT, 'fisher_hodge', 'ext'))
    from generic_fno import GenericFNO, train_generic, rollout, n_params
    norm = data['meta']['norm']
    tr = data['train']
    B = tr['c'].shape[0]
    X = tr['c'].reshape(B, -1, 1, N, N).float()
    model = GenericFNO(n_fields=1, n_cond=NC).to(device)
    npar = n_params(model)
    log(f'{variant} {name}: params {npar} steps {iters} (GENERIC-FNO drive variant, paper defaults)')
    t0 = time.time()
    info = train_generic(model, X, grid_cond(tr['drive'], norm), steps=iters, device=device, seed=seed, log=log)
    info['train_time_s'] = time.time() - t0
    torch.save(model.state_dict(), os.path.join(odir, name + '.pt'))
    res, snap = {}, {}
    for s in TEST_SETS:
        sd = data[s]; Y = sd['c']
        if sd['disc'] == 'cloud':
            res[s] = None; log(f'[{s}] N/A'); continue
        n = int(round(Y.shape[-1] ** 0.5))
        t1 = time.time()
        R = rollout(model, Y[:, 0].reshape(-1, 1, n, n).float(), Y.shape[1] - 1, grid_cond(sd['drive'], norm))
        R = R.reshape(R.shape[0], R.shape[1], -1)
        m = rollout_metrics(R, Y.cpu()); m['infer_time_s'] = time.time() - t1
        disc = data['discs'][sd['disc']]
        m['phys'] = dict(phys_metrics_rollout(R.to(device), sd['drive'], disc).result(), flux='min-norm')
        if s in ('c8', 'b025', 'a'):
            Rd = R.to(device)
            snap[f'sig_{s}'] = local_sigma(Rd[0, 100], disc.minnorm_flux((Rd[:1, 101] - Rd[:1, 100]) / DT)[0],
                                           {k: v[0] for k, v in sd['drive'].items() if torch.is_tensor(v)}, disc).cpu().numpy().astype(np.float32)
        snap[f'R_{s}'] = R[:4, -1].numpy().astype(np.float32)
        res[s] = m
        log(f'[{s}] T {m["err_T"]:.4g} 10T {m["err_10T"]:.4g} blown {m["n_blown"]} neg {m["n_neg_traj"]} mass {m["mass_drift_max"]:.1e}')
    o = dict(variant=variant, model='gfno', seed=seed, params=npar, iters=iters, train_time_s=info['train_time_s'],
             train={k: v for k, v in info.items() if k != 'loss_curve'}, sets=res)
    json.dump(o, open(os.path.join(odir, name + '.json'), 'w'))
    np.savez_compressed(os.path.join(odir, name + '_snap.npz'), **snap)
    return o


def run_helm(variant, seed, steps, out=OUT):
    """HelmFluid with the defaults of scripts/Navier_Stokes_HelmFluid.sh (as runs/external/density/code/run_helmfluid.py):
    T_in 10 (history padded with rho_0), T_out 10, batch 5, lr 5e-4, their Adam (wd 1e-4), StepLR gamma 0.5 with the
    period scaled to the step budget, relative-L2 LpLoss summed over the 10 steps.  Condition channels appended.
    Evaluated on the 32 x 32 sets only (its padding and scales are set for the training grid)."""
    from types import SimpleNamespace
    gpu_ok('cuda')
    sys.path.insert(0, os.path.expanduser('~/ext_baselines/patches'))
    from helm_cond import CondHelmFluid
    from utils.utilities3 import LpLoss
    from utils.adam import Adam
    torch.manual_seed(seed); np.random.seed(seed); torch.cuda.manual_seed(seed)
    dev = 'cuda'
    T_IN, T_OUT, BATCH, LR = 10, 10, 5, 5e-4
    data = load(variant, 'cpu', out, sets=['train'] + [s for s in TEST_SETS if DISC_OF[s] == 'g32'])
    norm = add_norm(data['meta'], data)['norm']
    odir = os.path.join(out, variant); os.makedirs(odir, exist_ok=True)
    name = f'helmfluid_s{seed}'
    logf = open(os.path.join(odir, name + '.log'), 'w')

    def log(*x):
        s = ' '.join(str(y) for y in x); print(s, flush=True); logf.write(s + '\n'); logf.flush()
    args = SimpleNamespace(groups=4, dataset_name='NS', in_dim=T_IN + NC, out_dim=1, d_model=64, padding='0,0')
    model = CondHelmFluid(args, n_cond=NC).to(dev)
    npar = sum(q.numel() for q in model.parameters())
    log(f'{variant} {name}: params {npar} steps {steps}')

    def history(u, s):
        idx = torch.clamp(torch.arange(s - T_IN + 1, s + 1), min=0)
        return u[:, idx].permute(0, 2, 3, 1)
    tr = data['train']
    U = tr['c'].reshape(tr['c'].shape[0], -1, N, N)
    Cd = grid_cond(tr['drive'], norm).permute(0, 2, 3, 1)
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
            if DISC_OF[s] != 'g32':
                res[s] = None; continue
            sd = data[s]; Y = sd['c']
            B = Y.shape[0]
            t1 = time.time()
            Y2 = Y.reshape(B, -1, N, N)
            h = history(Y2[:, :1], 0).to(dev)
            c = grid_cond(sd['drive'], norm).permute(0, 2, 3, 1).to(dev)
            outs = [Y2[:, 0]]
            for t in range(Y2.shape[1] - 1):
                im, _, _ = model(torch.cat([h, c], -1))
                bad = ~torch.isfinite(im).flatten(1).all(1) | (im.flatten(1).abs().amax(1) > 1e4)
                im = torch.where(bad[:, None, None, None], torch.full_like(im, float('nan')), im)
                outs.append(im[..., 0].cpu())
                h = torch.cat([h[..., 1:], im], -1)
            R = torch.stack(outs, 1).reshape(B, Y2.shape[1], -1)
            m = rollout_metrics(R, Y)
            m['infer_time_s'] = time.time() - t1
            disc = data['discs'][sd['disc']]
            m['phys'] = dict(phys_metrics_rollout(R.double(), sd['drive'], disc).result(), flux='min-norm')
            snap[f'R_{s}'] = R[:4, -1].numpy().astype(np.float32)
            res[s] = m
            log(f'[{s}] T {m["err_T"]:.4g} 10T {m["err_10T"]:.4g} blown {m["n_blown"]} neg {m["n_neg_traj"]} mass {m["mass_drift_max"]:.1e}')
    o = dict(variant=variant, model='helmfluid', seed=seed, params=npar, iters=steps, train_time_s=ttrain, step_size=step_size,
             final_loss=float(np.mean(hist[-200:])), sets=res)
    json.dump(o, open(os.path.join(odir, name + '.json'), 'w'))
    np.savez_compressed(os.path.join(odir, name + '_snap.npz'), **snap)
    return o


def run_persistence(variant, out=OUT, device='cpu'):
    """the no-change row, and the reference trajectories themselves through the minimum-norm flux reconstruction (the
    level of 'violations' that the reconstruction alone produces for models without an edge flux)"""
    data = load(variant, device, out)
    res, ref = {}, {}
    for s in TEST_SETS:
        sd = data[s]; Y = sd['c']
        disc = data['discs'][sd['disc']]
        res[s] = rollout_metrics(Y[:, :1].expand_as(Y).cpu(), Y.cpu(), disc.g['dens'])
        ref[s] = dict(rollout_metrics(Y.cpu(), Y.cpu(), disc.g['dens']), phys=dict(phys_metrics_rollout(Y.double(), sd['drive'], disc).result(), flux='min-norm'))
    os.makedirs(os.path.join(out, variant), exist_ok=True)
    json.dump(dict(variant=variant, model='persistence', seed=0, params=0, iters=0, train_time_s=0.0, sets=res),
              open(os.path.join(out, variant, 'persistence_s0.json'), 'w'))
    json.dump(dict(variant=variant, model='reference_minnorm', seed=0, params=0, iters=0, train_time_s=0.0, sets=ref),
              open(os.path.join(out, variant, 'reference_minnorm_s0.json'), 'w'))
    return res


# ------------------------------------------------------------------------------------------ summary and figures
LABEL = dict(wired_k='Fisher–Hodge: learned constitutive law k = D T w exp(kappa_theta(local descriptors))',
             wired_const='Fisher–Hodge, one k per medium (ablation)',
             wired_untrained='Fisher–Hodge, untrained (k = D T w)', wired_oracle='Fisher–Hodge, true mobility on the edges (oracle)',
             reference_minnorm='Reference trajectories (min-norm flux reconstruction)',
             lm_hodge='LM-Hodge: learned-metric linear Hodge operator (internal ablation)',
             ude='UDE closure: same discretisation and descriptors, free conservative edge flux (internal ablation)',
             fno='FNO (standard, ours)', gnn_ode='GNN-ODE (MeshGraphNet-style)', flux_gnn='Flux-GNN (conservative, positive)',
             gfno='GENERIC-FNO (re-impl., drive channels)', helmfluid='HelmFluid (ICML 2024, cond. channels)',
             persistence='No change, u(t) = u(0)')
ORDER = ['wired_k', 'wired_const', 'wired_untrained', 'wired_oracle', 'ude', 'lm_hodge', 'fno', 'gnn_ode', 'flux_gnn', 'gfno', 'helmfluid',
         'persistence', 'reference_minnorm']
COLS = [('a', 'a'), ('b025', 'T x0.25'), ('b05', 'b T x0.5'), ('b2', 'b T x2'), ('b4', 'T x4'), ('c2', 'c amp x2'), ('c4', 'c amp x4'),
        ('c8', 'amp x8'), ('d', 'd low por.'), ('e64', 'e 64x64'), ('ecloud', 'e cloud')]
PCOLS = ['a', 'b025', 'b4', 'c4', 'c8', 'd', 'ecloud']


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
        return 'N/A'
    f = lambda x: 'inf' if not np.isfinite(x) else (f'{x:.3g}' if x >= 1e-3 else f'{x:.1e}')
    if len(xs) == 1:
        return f(xs[0])
    return f(float(np.mean(xs))) + f' ({f(min(xs))}–{f(max(xs))})'


def get(r, c, key):
    v = r['sets'].get(c)
    return None if v is None else v.get(key)


def table(variant, out=OUT, key='err_10T'):
    res = collect(variant, out)
    lines = ['| Method | ' + ' | '.join(c[1] for c in COLS) + ' | params | train | seeds |', '|---' * (len(COLS) + 4) + '|']
    for m in ORDER:
        if m not in res or m == 'reference_minnorm':
            continue
        rs = res[m]
        cells = [fmt([get(r, c, key) for r in rs]) for c, _ in COLS]
        tt = np.mean([r.get('train_time_s', 0.0) or 0.0 for r in rs])
        lines.append(f'| {LABEL[m]} | ' + ' | '.join(cells) + f' | {rs[0]["params"]:,} | {tt / 60:.0f} min | {len(rs)} |')
    return '\n'.join(lines)


def side_table(variant, out=OUT):
    res = collect(variant, out)
    lines = ['| Method | neg. traj. of 64 (' + '/'.join(c for c, _ in COLS) + ') | max rel. mass drift | J(10T) rel. err a/b2/c4/d | blown |', '|---|---|---|---|---|']
    for m in ORDER:
        if m not in res:
            continue
        rs = res[m]
        neg = '/'.join(('–' if any(get(r, c, 'n_neg_traj') is None for r in rs) else str(max(get(r, c, 'n_neg_traj') for r in rs))) for c, _ in COLS)
        md = max([get(r, c, 'mass_drift_max') for r in rs for c, _ in COLS if get(r, c, 'mass_drift_max') is not None] or [float('nan')])
        J = '/'.join(fmt([get(r, c, 'J_relerr') for r in rs]) for c in ('a', 'b2', 'c4', 'd'))
        bl = sum(get(r, c, 'n_blown') or 0 for r in rs for c, _ in COLS)
        lines.append(f'| {LABEL[m]} | {neg} | {md:.1e} | {J} | {bl} |')
    return '\n'.join(lines)


def phys_table(variant, out=OUT):
    res = collect(variant, out)
    pg = lambda r, c, k: (r['sets'].get(c) or {}).get('phys', {}).get(k)
    lines = ['| Method | flux | LDB-violating edges, P(J f < 0): ' + ' / '.join(PCOLS) + ' | sigma_phys < 0 (fraction of samples): ' + ' / '.join(PCOLS)
             + ' | most negative sigma / median abs sigma (amp x8) | negative cells (max over sets) |', '|---|---|---|---|---|---|']
    for m in ORDER:
        if m not in res or m == 'persistence':
            continue
        rs = res[m]
        fl = next((((r['sets'].get(c) or {}).get('phys') or {}).get('flux') for r in rs for c in PCOLS if pg(r, c, 'flux')), '–')
        v = ' / '.join(fmt([pg(r, c, 'ldb_violation_frac') for r in rs]) for c in PCOLS)
        sn = ' / '.join(fmt([pg(r, c, 'sigma_neg_frac') for r in rs]) for c in PCOLS)
        smin = fmt([pg(r, 'c8', 'sigma_min_rel') for r in rs])
        ng = max([pg(r, c, 'neg_cell_frac') or 0 for r in rs for c, _ in COLS] or [0])
        lines.append(f'| {LABEL[m]} | {fl} | {v} | {sn} | {smin} | {ng:.2g} |')
    return '\n'.join(lines)


def k_table(variant, out=OUT):
    res = collect(variant, out)
    lines = ['| Model | set | Pearson log k | Spearman log k | Pearson k | rel. L2 err k | median abs log ratio | median log ratio |', '|---|---|---|---|---|---|---|---|']
    for m in ('wired_k', 'wired_const'):
        for r in res.get(m, []):
            for s in ('a', 'b025', 'c8', 'd', 'e64', 'ecloud'):
                k = (r['sets'].get(s) or {}).get('k')
                if k:
                    lines.append(f'| {m} s{r["seed"]} | {s} | {k["pearson_logk"]:.3f} | {k["spearman_logk"]:.3f} | {k["pearson_k"]:.3f} | {k["relerr_k"]:.3f} | '
                                 f'{k["median_abs_logratio"]:.3f} | {k["median_logratio"]:+.3f} |')
    return '\n'.join(lines)


def summary(out=OUT, notes=''):
    for v in ('H1', 'H1p'):
        if os.path.exists(os.path.join(out, 'data', f'{v}.pt')) and not os.path.exists(os.path.join(out, v, 'reference_minnorm_s0.json')):
            run_persistence(v, out)
    gi = {v: json.load(open(os.path.join(out, 'data', f'gen_info_{v}.json'))) for v in ('H1', 'H1p') if os.path.exists(os.path.join(out, 'data', f'gen_info_{v}.json'))}
    s = ['# H1: driven transport in heterogeneous media (hyperme/bench_hetero.py)', '',
         'Relative L2 error at 10T (200 steps of DT = 0.05; training window 20 steps), median over 64 test trajectories '
         '(16 unseen media x 4); mean over seeds with (min–max). Columns: a = training conditions on unseen media; '
         'b = temperature x0.5 / x2; c = drive amplitude x2 / x4; d = unseen low-porosity media (0.15–0.3, training 0.3–0.7); '
         'e = the trajectories of a on a 64 x 64 grid and on a random point cloud of 1024 Voronoi cells, models trained on '
         '32 x 32 and not retrained. N/A: the model cannot run on that discretisation.', '', notes, '',
         '## H1: meshfree heterogeneous Fokker-Planck with drive', '', table('H1', out), '', side_table('H1', out), '',
         '## Physical consistency under the given conditions (H1)', '',
         'f_e = mu_i - mu_j + amp int_e F.dl / T with mu = log rho + U / T evaluated with the true condition of the test set; '
         'J_e the model\'s own edge flux where it has one, otherwise the minimum-norm flux of its step change (curl-free by '
         'construction; the row of the reference trajectories through the same reconstruction shows how much of the '
         'violation this reconstruction alone produces). Sampled every 10 steps over the 10T rollout.', '', phys_table('H1', out), '',
         "## H1': crowding mobility M / (1 + rho / 2), outside the model class", '', table('H1p', out), '', side_table('H1p', out), '',
         phys_table('H1p', out), '',
         '## Learned constitutive law against the true mobility on unseen media', '',
         'exp(kappa) of the trained model against the true pixel mobility averaged harmonically along each edge (direction-weighted).', '',
         k_table('H1', out), '', "H1' (compared with the crowding-free mobility):", '', k_table('H1p', out), '',
         '## Error at T (end of the training window)', '', table('H1', out, 'err_T'), '',
         '## Data and reference', '', '```', json.dumps(gi, indent=1, default=float)[:8000], '```']
    open(os.path.join(out, 'summary.md'), 'w').write('\n'.join(s))
    print('\n'.join(s[:14]))


def figs(out=OUT, seed=0):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    sn = np.load(os.path.join(out, 'H1', f'wired_k_s{seed}_snap.npz'))
    js = json.load(open(os.path.join(out, 'H1', f'wired_k_s{seed}.json')))
    panels = [('a', '32 x 32, unseen media (porosity 0.3–0.7)'), ('d', '32 x 32, unseen low-porosity media'), ('ecloud', 'point cloud, unseen media (no retraining)')]
    fig, axs = plt.subplots(1, 3, figsize=(14, 4.6), constrained_layout=True)
    for ax, (s, title) in zip(axs, panels):
        gt, gh, ep = sn[f'k_{s}_g_true'], sn[f'k_{s}_g_hat'], sn[f'k_{s}_eps']
        i = np.random.default_rng(0).choice(len(gt), min(len(gt), 12000), replace=False)
        sc = ax.scatter(gt[i], gh[i], c=ep[i], s=3, cmap='viridis', vmin=0, vmax=1, rasterized=True)
        lo, hi = 5e-5, max(gt.max(), gh.max()) * 2
        ax.plot([lo, hi], [lo, hi], 'k--', lw=0.8)
        ax.set_xscale('log'); ax.set_yscale('log'); ax.set_xlim(lo, hi); ax.set_ylim(lo, hi)
        k = js['sets'][s]['k']
        ax.set_title(f'{title}\nPearson log k {k["pearson_logk"]:.3f}, Spearman {k["spearman_logk"]:.3f}', fontsize=9)
        ax.set_xlabel('true edge mobility (harmonic mean along the edge)'); ax.set_ylabel('learned exp(kappa)')
    fig.colorbar(sc, ax=axs, label='local porosity at the edge (model descriptor)')
    fig.savefig(os.path.join(out, 'k_scatter.png'), dpi=150); plt.close(fig)
    data = torch.load(os.path.join(out, 'data', 'H1.pt'))
    s, j = 'b2', 0
    med = int(data[s]['med'][j])
    pore = data['meta']['pools']['test']['pore'][med].numpy()
    truth = data[s]['rho'][j, -1].reshape(N, N).numpy()
    cols = [('truth', truth)]
    for m, lab in (('wired_k', 'HyperME wired'), ('lm_hodge', 'LM-Hodge'), ('fno', 'FNO'), ('gnn_ode', 'GNN-ODE'), ('flux_gnn', 'Flux-GNN')):
        f = os.path.join(out, 'H1', f'{m}_s{seed}_snap.npz')
        if os.path.exists(f):
            cols.append((lab, np.load(f)[f'R_{s}'][j].reshape(N, N)))
    fig, axs = plt.subplots(1, len(cols) + 1, figsize=(2.9 * (len(cols) + 1), 3.4), constrained_layout=True)
    axs[0].imshow(pore, cmap='gray_r', origin='lower'); axs[0].set_title(f'microstructure (porosity {pore.mean():.2f})', fontsize=9)
    vmax = float(np.nanmax(truth)) * 1.05
    for ax, (lab, im) in zip(axs[1:], cols):
        e = np.linalg.norm(np.nan_to_num(im) - truth) / np.linalg.norm(truth)
        ax.imshow(im, cmap='magma', vmin=0, vmax=vmax, origin='lower')
        neg = int((im < 0).sum())
        ax.set_title(f'{lab}\nrel. err {e:.3f}' + (f', {neg} cells < 0' if neg else ''), fontsize=9)
    for ax in axs:
        ax.set_xticks([]); ax.set_yticks([])
    fig.suptitle('Unseen medium, temperature x2 (training T = 1): density at 10T', fontsize=10)
    fig.savefig(os.path.join(out, 'hetero_rollout.png'), dpi=150); plt.close(fig)
    double_kill(out, seed)


def double_kill(out=OUT, seed=0, variant='H1'):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    res = collect(variant, out)
    show = [('wired_k', 'Fisher–Hodge', '#1f6feb'), ('ude', 'UDE closure', '#d1242f'), ('lm_hodge', 'LM-Hodge', '#8250df'),
            ('flux_gnn', 'Flux-GNN', '#bf8700'), ('gnn_ode', 'GNN-ODE', '#57606a'), ('fno', 'FNO', '#1a7f37'),
            ('gfno', 'GENERIC-FNO', '#e16f24'), ('helmfluid', 'HelmFluid', '#6e7781')]
    axes_def = [('drive amplitude (training 1)', [1, 2, 4, 8], ['a', 'c2', 'c4', 'c8']),
                ('temperature (training 1)', [0.25, 0.5, 1, 2, 4], ['b025', 'b05', 'a', 'b2', 'b4'])]
    fig = plt.figure(figsize=(15, 8.4), constrained_layout=True)
    gs = fig.add_gridspec(2, 3, width_ratios=[1.2, 1.2, 1])
    for row, (xl, xs, sets) in enumerate(axes_def):
        ax = fig.add_subplot(gs[row, 0]); ax2 = fig.add_subplot(gs[row, 1])
        for m, lab, col in show:
            if m not in res:
                continue
            rs = res[m]
            err = [np.mean([get(r, c, 'err_10T') for r in rs]) for c in sets]
            vio = [np.mean([((r['sets'].get(c) or {}).get('phys') or {}).get('ldb_violation_frac', np.nan) for r in rs]) for c in sets]
            ax.plot(xs, np.clip(err, 1e-4, 1e2), '-o', color=col, label=lab, lw=1.6, ms=4)
            ax2.plot(xs, vio, '--s', color=col, label=lab, lw=1.4, ms=4)
        for a_, yl in ((ax, '10T relative L2 error'), (ax2, 'fraction of edges with J f < 0')):
            a_.set_xscale('log', base=2); a_.set_xticks(xs); a_.set_xticklabels([f'x{x:g}' for x in xs]); a_.set_xlabel(xl); a_.set_ylabel(yl)
            a_.axvline(1, color='0.7', lw=0.8, ls=':')
        ax.set_yscale('log')
        ax2.set_ylim(-0.02, 0.6)
        if row == 0:
            ax.set_title('accuracy on unseen media', fontsize=10); ax2.set_title('violated local detailed balance (physical force of the test condition)', fontsize=10)
            ax.legend(fontsize=7, ncol=2)
    for row, (m, lab) in enumerate((('ude', 'UDE closure'), ('wired_k', 'Fisher–Hodge'))):
        ax = fig.add_subplot(gs[row, 2])
        f = os.path.join(out, variant, f'{m}_s{seed}_snap.npz')
        if not os.path.exists(f) or 'sig_c8' not in np.load(f):
            ax.set_axis_off(); continue
        sg = np.load(f)['sig_c8'].reshape(N, N)
        lim = np.abs(sg).max()
        ax.imshow(sg, cmap='coolwarm_r', vmin=-lim, vmax=lim, origin='lower')
        neg = sg < 0
        ax.contour(neg.astype(float), levels=[0.5], colors='magenta', linewidths=0.8, origin='lower')
        ax.set_title(f'{lab}: local sigma_phys at t = 5T, drive x8\n{neg.mean() * 100:.1f} % of cells < 0 (magenta, red)', fontsize=9)
        ax.set_xticks([]); ax.set_yticks([])
    fig.savefig(os.path.join(out, 'hetero_double_kill.png'), dpi=150); plt.close(fig)


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
        print(json.dumps(generate(a.variant, a.device), indent=1, default=str)[:3000])
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
