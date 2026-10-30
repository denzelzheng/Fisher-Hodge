"""Positive SSP-RK2 for any model with rates(c, ctx), escape_ratio(c, a, b) and a hypergraph hg.

positive_ssp_rk2   explicit SSP-RK2 (Heun) with positivity-preserving substeps.  Each Euler stage
                   c + dt S (a - b) stays positive when dt * max_m consumption_m / c_m <= theta, because every
                   consuming term carries its reactant concentration; the RK2 update is a convex combination of two
                   such stages.  vectorized=True keeps a step size per sample and synchronises with the host once per
                   `chunk` substeps; vectorized=False uses one step size for the batch.
neuralchart.fast implements the same scheme with error control for training and evaluation on the GPU.
"""
import torch


@torch.no_grad()
def _ratio(model, c, a, b):
    return model.escape_ratio(c, a, b)


def positive_ssp_rk2(model, c, ctx, Dt, dt_max=None, theta0=0.5, theta1=0.5, max_sub=4000, t_tol=1e-9,
                     vectorized=None, chunk=None):
    """advance c (B, M) by Dt.  dt_max = nominal substep (default Dt/8).  vectorized=None picks the vectorized
    version on GPU and the loop on CPU (where there is no host sync to save).  Returns (c, substeps): substeps is a
    (B,) long tensor (vectorized) or an int (loop)."""
    dt_max = Dt / 8 if dt_max is None else dt_max
    hg = model.hg
    if vectorized is None:
        vectorized = c.is_cuda
    if not vectorized:
        t, nsub = 0.0, 0
        while Dt - t > t_tol:
            a0, b0 = model.rates(c, ctx)
            f0 = hg.apply_S(a0 - b0)
            r0 = float(_ratio(model, c, a0, b0).max())
            dt = min(dt_max, Dt - t, theta0 / r0 if r0 > 0 else float('inf'))
            while True:
                c1 = c + dt * f0
                a1, b1 = model.rates(c1, ctx)
                r1 = float(_ratio(model, c1, a1, b1).max())
                if dt * r1 <= theta1 or dt < 1e-12:
                    break
                dt *= 0.5
            c2 = c1 + dt * hg.apply_S(a1 - b1)
            c = 0.5 * (c + c2)
            t += dt; nsub += 1
            if nsub >= max_sub:
                break
        return c, nsub

    B = c.shape[0]
    dev = c.device
    t = torch.zeros(B, dtype=torch.float64, device=dev)
    cap = torch.full((B,), float('inf'), dtype=torch.float64, device=dev)
    nsub = torch.zeros(B, dtype=torch.long, device=dev)
    chunk = chunk or max(1, int(round(Dt / dt_max)))
    it = 0
    while True:
        for _ in range(chunk):
            a0, b0 = model.rates(c, ctx)
            f0 = hg.apply_S(a0 - b0)
            with torch.no_grad():
                r0 = _ratio(model, c, a0, b0).double()
                rem = Dt - t
                dt = torch.minimum(torch.minimum(rem, cap), torch.where(r0 > 0, theta0 / r0, torch.full_like(r0, float('inf'))))
                dt = torch.clamp(dt, max=dt_max)
                active = rem > t_tol
                dt = torch.where(active, dt, torch.zeros_like(dt))
            dtc = dt.to(c.dtype)[:, None]
            c1 = c + dtc * f0
            a1, b1 = model.rates(c1, ctx)
            c2 = c1 + dtc * hg.apply_S(a1 - b1)
            with torch.no_grad():
                r1 = _ratio(model, c1, a1, b1).double()
                ok = (dt * r1 <= theta1) | (dt < 1e-12)
                take = ok & active
            c = torch.where(take[:, None], 0.5 * (c + c2), c)
            with torch.no_grad():
                t = t + torch.where(take, dt, torch.zeros_like(dt))
                cap = torch.where(ok, torch.full_like(cap, float('inf')), 0.5 * dt)
                nsub = nsub + take.long()
            it += 1
        if not bool(((Dt - t) > t_tol).any()) or it >= max_sub:
            break
    return c, nsub
