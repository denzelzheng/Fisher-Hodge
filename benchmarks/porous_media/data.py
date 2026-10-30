"""Data of the porous-media benchmark: random media, meshfree fields, discretisations and the reference solver.

Physics.  The density rho obeys a heterogeneous Fokker-Planck equation with a divergence-free drive on the unit torus,

    d_t rho = div( M(x) [ T grad rho + rho grad U - amp rho F ] ),   F = (d_y psi, -d_x psi).

M(x) is diagonal and comes from a local effective-medium rule on a 128 x 128 pore map.  In a Gaussian window of 6 fine
pixels, eps(x) is the porosity and tau_x(x) the fraction of horizontally connected pore pairs divided by eps.  Then
M_x(x) = C eps^3 / ((1 - eps)^2 + 0.3) * (0.05 + 0.95 tau_x^2) + 1e-4, and M_y likewise.  C normalises the mean edge
mobility of the training media to 1.  D = 0.02 sets the time scale.  With crowding the mobility is
M / (1 + rhobar / rho_s), rho_s = 2, where rhobar is the mean density of the two cells of a face.

Reference.  Finite volumes on the 128 x 128 pixels (square-root approximation, harmonic mean of M on every pixel face,
the drive as the exact flux of psi through the face), RK4 in float64, saved every DT = 0.05.  The fine density is
projected (cell averages) onto the evaluation discretisations: the 32 x 32 training grid, a 64 x 64 grid and the
Voronoi cells of a random point cloud of 1024 points.  U, psi and rho_0 are sums of periodic Gaussians, stored as
their parameters so that they can be evaluated at any point.

Model inputs.  A discretisation is any point set with Voronoi cells.  The state is c_i = cell mass / mean cell
volume.  The material enters only through 12 edge descriptors computed on the pore map (porosity and connected-pair
fractions at two window sizes, mixed into the edge frame), never through coordinates.

    python -m benchmarks.porous_media.run generate --variant porous_media
"""
import json
import math
import os
import time

import numpy as np
import torch

from .test_sets import CONDITIONS, GENERATED, DISC_OF, SET_NAME

N, NF = 32, 128                          # training grid, fine (reference) grid
D = 0.02
KREF = D * N * N
DT, T_TRAIN, T_TEST = 0.05, 20, 200
T_TRAIN_DATA, N_RELAX = 100, 16          # training trajectories hold 5 time units; 16 undriven relaxation runs
RHO_S = 2.0                              # crowding density
KC_DELTA, M_FLOOR = 0.3, 1e-4
SIG_T, SIG_M, SIG_M2 = 6.0, 2.0, 6.0     # Gaussian windows (fine pixels): truth, model descriptors
POT_ID = dict(K=(3, 5), a=(1.0, 3.0), s=(0.08, 0.16))
PSI_ID = dict(K=(3, 5), a=(1.5, 4.0), s=(0.08, 0.16))
IC_ID = dict(n=(1, 3), sig=(0.05, 0.10))
POR_TRAIN, POR_LOW, ELL = (0.3, 0.7), (0.15, 0.3), (2.0, 10.0)
CLOUD_N, CLOUD_RMIN, CLOUD_SEED, RASTER = 1024, 0.45 / 32, 4242, 512
VARIANTS = ('porous_media', 'porous_media_crowding')
REPO = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


def data_dir_of(variant, data_dir=None):
    """the data directory of a variant: data_dir if given, else $NEURALCHART_DATA/<variant>, else <repo>/data/<variant>"""
    if data_dir:
        return data_dir
    root = os.environ.get('NEURALCHART_DATA') or os.path.join(REPO, 'data')
    return os.path.join(root, variant)


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
    return torch.stack([kc(e) * (0.05 + 0.95 * tx ** 2), kc(e) * (0.05 + 0.95 * ty ** 2)], 1)


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


def face_drive(ppar, geom):
    """finite-volume form of the drive per edge: the flux of F = (d_y psi, -d_x psi) through the cell face,
    psi(face end) - psi(face start), scaled to a line integral by |x_j - x_i| / |sigma_e| (exactly divergence-free in
    the discrete sense; on a grid it is the dual-edge difference of psi).  (B, H)"""
    d = geom['dvec'].norm(dim=-1)
    return (gauss_eval(ppar, geom['fb']) - gauss_eval(ppar, geom['fa'])) * (d / geom['sig'])[None]


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
    from neuralchart.graphs import grid_edge_list
    c = (torch.arange(n, dtype=torch.float64) + 0.5) / n
    Y, X = torch.meshgrid(c, c, indexing='ij')
    pts = torch.stack([X.reshape(-1), Y.reshape(-1)], 1)
    E = torch.as_tensor(grid_edge_list(n))
    V = n * n
    dv = torch.zeros(2 * V, 2, dtype=torch.float64); dv[:V, 0] = 1.0 / n; dv[V:, 1] = 1.0 / n
    idx = np.arange(V).reshape(n, n)
    v00, v10, v11, v01 = idx.ravel(), np.roll(idx, -1, 1).ravel(), np.roll(np.roll(idx, -1, 0), -1, 1).ravel(), np.roll(idx, -1, 0).ravel()
    tris = np.concatenate([np.stack([v00, v10, v11], 1), np.stack([v00, v11, v01], 1)])      # CCW triangles of the 2-complex
    t = torch.stack([-dv[:, 1], dv[:, 0]], -1)                              # face direction (n rotated by +90 degrees), length h
    mid = (pts[E[:, 0]] + 0.5 * dv) % 1.0
    g = dict(kind='grid', n=n, pts=pts, edges=E, dvec=dv, w=torch.full((2 * V,), float(n * n), dtype=torch.float64),
             fa=(mid - 0.5 * t) % 1.0, fb=(mid + 0.5 * t) % 1.0, sig=torch.full((2 * V,), 1.0 / n, dtype=torch.float64),
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
        va, vb = vor.vertices[rv[0]], vor.vertices[rv[1]]
        tdir = np.array([-disp[1], disp[0]])
        if np.dot(vb - va, tdir) < 0:
            va, vb = vb, va
        edges[key] = (disp, np.linalg.norm(vb - va), va % 1.0, vb % 1.0)
    keys = sorted(edges)
    E = np.array(keys); dv = np.array([edges[k][0] for k in keys]); sig = np.array([edges[k][1] for k in keys])
    fa = np.array([edges[k][2] for k in keys]); fb = np.array([edges[k][3] for k in keys])
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
             fa=torch.as_tensor(fa), fb=torch.as_tensor(fb), sig=torch.as_tensor(sig),
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


def generate(variant, device='cuda', data_dir=None, conds=None):
    """reference data of one variant on the fine grid, projected to the evaluation discretisations.  Every set is
    written to <data directory of the variant>/<set>.pt as soon as it is done (meta.pt first)."""
    assert variant in VARIANTS
    ddir = data_dir_of(variant, data_dir)
    os.makedirs(ddir, exist_ok=True)
    crowd = variant == 'porous_media_crowding'
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
                sig=dict(truth=SIG_T, model=SIG_M, model2=SIG_M2), cond={k: (v['T'], v['amp']) for k, v in CONDITIONS.items()},
                cloud=dict(V=CLOUD_N, H=int(geoms['cloud']['edges'].shape[0]), rmin=CLOUD_RMIN,
                           area_cv=float(geoms['cloud']['area'].std() / geoms['cloud']['area'].mean()),
                           raster_area_relerr_max=float(((geoms['cloud']['area_raster'] - geoms['cloud']['area']).abs() / geoms['cloud']['area']).max())))
    for k in pools:
        m = Mface[k]
        info[f'M_{k}'] = dict(mean=float(m.mean()), median=float(m.median()), max=float(m.max()), frac_below_1e2=float((m < 1e-2).double().mean()),
                             porosity_min=float(pools[k]['pmed'].min()), porosity_max=float(pools[k]['pmed'].max()))
    fin = {k: fine_inputs(v, device) for k, v in pools.items()}
    ntr = len(pools['train']['med'])
    sets = dict(train=('train', 1.0, 1.0, T_TRAIN_DATA, ['g32'], None),
                relax=('train', 1.0, 0.0, T_TRAIN_DATA, ['g32'], np.arange(0, ntr, ntr // N_RELAX)))   # one per 4th medium, no drive
    for s in (conds or GENERATED):
        cd = CONDITIONS[s]
        sets[s] = (cd['pool'], cd['T'], cd['amp'], T_TEST, ['g32', 'g64', 'cloud'] if s == 'training_conditions' else ['g32'], None)
    data = dict(meta=dict(info, pools={k: dict(pore=torch.as_tensor(v['pore']), pmed=v['pmed'].tolist()) for k, v in pools.items()},
                          desc={k: v.float().cpu() for k, v in desc.items()}, Mpix={k: v.float().cpu() for k, v in Mpix.items()}))
    torch.save(data['meta'], os.path.join(ddir, 'meta.pt'))
    for s, (pool, T, amp, nst, discs, idx) in sets.items():
        p = pools[pool]
        med = torch.as_tensor(p['med'], device=device)
        U, dpsi, r0 = fin[pool]
        if idx is not None:
            p = dict(p, Upar=p['Upar'][idx], Ppar=p['Ppar'][idx], med=p['med'][idx])
            ii = torch.as_tensor(idx, device=device)
            med, U, dpsi, r0 = med[ii], U[ii], dpsi[ii], r0[ii]
        B = U.shape[0]
        Tt = torch.full((B,), T, dtype=torch.float64, device=device); At = torch.full((B,), amp, dtype=torch.float64, device=device)
        t1 = time.time()
        Rs, J, rf, ns = simulate(r0, Mface[pool][med], U, dpsi, Tt, At, crowd, nst, [projs[g] for g in discs])
        rec = dict(ns_per_DT=ns, sim_time_s=time.time() - t1, min_rho=float(Rs[0].min()),
                   mass_drift=float((Rs[0].mean(-1) - Rs[0][:, :1].mean(-1)).abs().max()))
        if s == 'training_conditions':
            R2, _, _, _ = simulate(r0[:8], Mface[pool][med[:8]], U[:8], dpsi[:8], Tt[:8], At[:8], crowd, nst, [projs['g32']], ns=2 * ns)
            rec['dt_halving_relerr_end_max'] = float((torch.linalg.vector_norm(Rs[0][:8, -1] - R2[0][:, -1], dim=-1)
                                                      / torch.linalg.vector_norm(R2[0][:, -1], dim=-1)).max())
        base = dict(Upar=torch.as_tensor(p['Upar']).float(), Ppar=torch.as_tensor(p['Ppar']).float(), med=med.cpu(), pool=pool,
                    T=Tt.float().cpu(), amp=At.float().cpu())
        for g, R in zip(discs, Rs):
            name = s if g == 'g32' else SET_NAME[g]
            data[name] = dict(base, rho=R.float().cpu(), disc=g)
            if g == 'g32':
                data[name]['J_end'] = coarse_flux32(J).float().cpu()
            torch.save(data[name], os.path.join(ddir, f'{name}.tmp')); os.replace(os.path.join(ddir, f'{name}.tmp'), os.path.join(ddir, f'{name}.pt'))
        data['meta'][f'sim_{s}'] = rec
        torch.save(data['meta'], os.path.join(ddir, 'meta.pt'))
        print(variant, s, json.dumps(rec), flush=True)
    data['meta']['gen_time_s'] = time.time() - t0
    torch.save(data['meta'], os.path.join(ddir, 'meta.pt'))
    gi = {k: v for k, v in data['meta'].items() if k not in ('pools', 'desc', 'Mpix')}
    json.dump(gi, open(os.path.join(ddir, 'generation_info.json'), 'w'), indent=1, default=float)
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
        """transport hypergraph on the edges (grid_edge_list order on grids).  The cycle basis is not used here, so it is
        set empty instead of computing the dense null space; the one moiety is the total mass."""
        if s._hg is None:
            from neuralchart.graphs import Hypergraph
            s._hg = Hypergraph.from_graph(s.g['edges'].cpu().numpy(), s.V, cycle_basis=np.zeros((s.H, 0)))
            s._hg._L = np.full((s.V, 1), 1.0 / math.sqrt(s.V))
        return s._hg


def load_raw(variant, data_dir=None):
    """the stored sets of a variant: <data directory of the variant>/<set>.pt"""
    dd = data_dir_of(variant, data_dir)
    data = dict(meta=torch.load(os.path.join(dd, 'meta.pt')))
    for f in sorted(os.listdir(dd)):
        if f.endswith('.pt') and f != 'meta.pt':
            data[f[:-3]] = torch.load(os.path.join(dd, f))
    return data


def load(variant, device='cpu', sets=None, data_dir=None):
    """dict(set -> dict(c (B, T+1, V), drive, disc, J_end, med, pool), meta, discs); sets not generated are skipped"""
    from .test_sets import TEST_SETS
    data = load_raw(variant, data_dir)
    meta = data['meta']
    discs = {}
    res = dict(meta=meta, discs=discs)
    for s in [k for k in (sets or ['train', 'relax'] + list(TEST_SETS)) if k in data]:
        d = data[s]
        dn = d['disc']
        if dn not in discs:
            discs[dn] = Disc(dn, meta, device)
        G = discs[dn]
        Up, Pp = d['Upar'].double().to(device), d['Ppar'].double().to(device)
        med = d['med'].to(device)
        drive = dict(U=gauss_eval(Up, G.g['pts']).float(), psin=gauss_eval(Pp, G.g['pts']).float(), line=face_drive(Pp, G.g).float(),
                     T=d['T'].to(device), amp=d['amp'].to(device), Ppar=Pp, ein=G.ein[d['pool']][med], ndesc=G.ndesc[d['pool']][med],
                     pmed=torch.as_tensor(meta['pools'][d['pool']]['pmed'], dtype=torch.float32, device=device)[med])
        res[s] = dict(c=d['rho'].to(device), drive=drive, disc=dn, J_end=d.get('J_end'), med=d['med'], pool=d['pool'])
    return res


def add_norm(meta, data):
    """z-scoring scalars of the black-box inputs, from the training set"""
    if 'norm' not in meta:
        dr = data['train']['drive']
        meta['norm'] = dict(U=(float(dr['U'].mean()), float(dr['U'].std())), psi=(float(dr['psin'].mean()), float(dr['psin'].std())),
                            line=float(dr['line'].std()))
    return meta


def training_set(data, window, relax=False):
    """the training trajectories cut to `window` steps of DT, plus (relax) the undriven relaxation trajectories"""
    parts = [data['train']] + ([data['relax']] if relax else [])
    c = torch.cat([p['c'][:, :window + 1] for p in parts])
    drive = {k: torch.cat([p['drive'][k] for p in parts]) for k in data['train']['drive']}
    med = torch.cat([p['med'] for p in parts])
    return dict(c=c, drive=drive, med=med, pool='train', disc='g32', J_end=None)
