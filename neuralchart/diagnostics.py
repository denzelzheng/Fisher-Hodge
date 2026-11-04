"""Physics diagnostics shared by every benchmark.

Local detailed balance.  An edge (hyperedge) violates local detailed balance when its net flux J runs against the
local thermodynamic force f, J f < 0.  Rounding must not count: with forward / backward rates a, b the net flux
J = a - b is a difference, so its rounding error scales with the gross rate a + b, not with |J|.  The rule is

    violation  <=>  J f < -LDB_TOL * scale * |f|,   scale = a + b  for a model with forward / backward rates
                                                     scale = |J|    for a model with a single edge flux (no a, b)

Entropy production sigma = sum_h J_h f_h, with f the physical force of the test condition.  Negative values of the
state are counted separately.
"""
import torch

LDB_TOL = 1e-6
LDB_RULE = 'J f < -1e-6 (a + b) |f| (models with forward / backward rates a, b); J f < -1e-6 |J| |f| (single edge flux)'


def ldb_violation(J, f, gross=None, tol=LDB_TOL):
    """boolean mask of the edges whose flux runs against the force beyond rounding.  J, f (and gross = a + b, when the
    model has forward / backward rates) of the same shape; gross None: the scale is |J| (single-flux models)."""
    scale = J.abs() if gross is None else torch.as_tensor(gross, dtype=J.dtype, device=J.device).abs()
    return J * f < -tol * scale * f.abs()


@torch.no_grad()
def activity_summary(free_energy, c, n=200):
    """the learned activity correction on the training range: the gates s_mu and, per species, the largest |ln gamma|
    = |s_mu r| over the densities of the data c (B, T + 1, M).  None without an activity correction."""
    fe = free_energy
    if not getattr(fe, 'residual', False):
        return None
    cs = fe._density(c.reshape(-1, fe.NS, fe.V).to(fe.s_mu.device).double())
    lo, hi = cs.amin((0, 2)), cs.amax((0, 2))
    x = (lo[None] + (hi - lo)[None] * torch.linspace(0, 1, n, dtype=lo.dtype, device=lo.device)[:, None])[..., None]
    g = (fe.s_mu.to(x.dtype)[None, :, None] * fe.residual_of_density(x))[..., 0]
    return dict(s_mu=fe.s_mu.detach().cpu().tolist(), max_abs_lngamma=g.abs().amax(0).cpu().tolist(),
                max_abs_lngamma_all=float(g.abs().max()), density_range=[lo.cpu().tolist(), hi.cpu().tolist()])


def relerr_traj(R, Y):
    """relative L2 error per trajectory and time, (B, T+1); a non-finite error is inf"""
    B, T1 = Y.shape[:2]
    e = torch.linalg.vector_norm((R - Y).reshape(B, T1, -1), dim=-1) / torch.linalg.vector_norm(Y.reshape(B, T1, -1), dim=-1)
    return torch.where(torch.isfinite(e), e, torch.full_like(e, float('inf')))
