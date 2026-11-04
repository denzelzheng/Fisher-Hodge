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


def relerr_traj(R, Y):
    """relative L2 error per trajectory and time, (B, T+1); a non-finite error is inf"""
    B, T1 = Y.shape[:2]
    e = torch.linalg.vector_norm((R - Y).reshape(B, T1, -1), dim=-1) / torch.linalg.vector_norm(Y.reshape(B, T1, -1), dim=-1)
    return torch.where(torch.isfinite(e), e, torch.full_like(e, float('inf')))
