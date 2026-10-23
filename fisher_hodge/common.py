"""Shared grid operators / constants for the Fisher-Hodge mechanism experiments.

Grid: 32x32 periodic, node (a,b) = cell centre x=(b+1/2)h, y=(a+1/2)h  (a = row = y, b = col = x).
Edges: horizontal (a,b)->(a,b+1) [index 0], vertical (a,b)->(a+1,b) [index 1]  => E = 2*N*N = 2048.
d   : node -> edge, (d u)_e = u_head - u_tail
d^T : edge -> node (adjoint), +f at head, -f at tail.  -d^T d = graph Laplacian.
Corners (dual vertices) P[a,b] at x=(b+1)h, y=(a+1)h.
"""
import os, math
import torch

N = 32            # coarse (model) grid
NF = 64           # fine (reference) grid
D = 0.02          # diffusion
DT = 0.05         # sampling interval
T_TRAIN = 20
T_TEST = 200
HC = 1.0 / N
S_SCALE = D / HC ** 2   # common rate scale D/h^2 used by all models (time normalisation)
ROOT = os.environ.get('FH_ROOT') or os.path.expanduser('~/fisher_hodge')
RUNS = os.path.join(ROOT, 'runs')


def grad_e(u):
    """node (...,n,n) -> edge (...,2,n,n)"""
    return torch.stack([u.roll(-1, -1) - u, u.roll(-1, -2) - u], -3)


def div_T(f):
    """edge (...,2,n,n) -> node (...,n,n):  d^T f"""
    fh, fv = f[..., 0, :, :], f[..., 1, :, :]
    return (fh.roll(1, -1) - fh) + (fv.roll(1, -2) - fv)


def mean_e(u):
    return torch.stack([(u.roll(-1, -1) + u) / 2, (u.roll(-1, -2) + u) / 2], -3)


def head_e(u):
    return torch.stack([u.roll(-1, -1), u.roll(-1, -2)], -3)


def tail_e(u):
    return torch.stack([u, u], -3)


def curl_f(f):
    """edge (...,2,n,n) -> face (...,n,n), counter-clockwise circulation"""
    fh, fv = f[..., 0, :, :], f[..., 1, :, :]
    return fh + fv.roll(-1, -1) - fh.roll(-1, -2) - fv


def dual_dpsi(P):
    """corner stream function (...,n,n) -> edge (...,2,n,n): h*u_e with u = grad^perp psi, exactly div-free"""
    return torch.stack([P - P.roll(1, -2), -(P - P.roll(1, -1))], -3)


def corner_to_node(P):
    return 0.25 * (P + P.roll(1, -1) + P.roll(1, -2) + P.roll(1, -1).roll(1, -2))


def edge_dirs(f):
    """edge field (B,2,n,n) -> oriented values for the 4 directed neighbours (+x,-x,+y,-y): (B,4,n,n)"""
    fh, fv = f[:, 0], f[:, 1]
    return torch.stack([fh, -fh.roll(1, -1), fv, -fv.roll(1, -2)], 1)


def nbr_dirs_cl(h):
    """channels-last node latent (B,n,n,C) -> neighbour latents (B,4,n,n,C) in order (+x,-x,+y,-y)"""
    return torch.stack([h.roll(-1, 2), h.roll(1, 2), h.roll(-1, 1), h.roll(1, 1)], 1)


def coarsen(rho):
    B = rho.shape[:-2]
    n = rho.shape[-1] // 2
    return rho.reshape(*B, n, 2, n, 2).mean((-3, -1))


def coarsen_flux(J):
    """fine edge flux (...,2,NF,NF) -> coarse edge flux (...,2,N,N), exact coarse graining (per unit length)"""
    n = J.shape[-1] // 2
    Jh = J[..., 0, :, :].reshape(*J.shape[:-3], n, 2, n, 2)[..., :, :, :, 1].mean(-2)
    Jv = J[..., 1, :, :].reshape(*J.shape[:-3], n, 2, n, 2)[..., :, 1, :, :].mean(-1)
    return torch.stack([Jh, Jv], -3)
