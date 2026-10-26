"""Time integrators for HyperME and any model with rates(c, ctx), escape_ratio(c, a, b) and a hypergraph hg.

positive_ssp_rk2   explicit SSP-RK2 (Heun) with positivity-preserving substeps.  Each Euler stage
                   c + dt S (a - b) stays positive when dt * max_m consumption_m / c_m <= theta, because every
                   consuming term carries its reactant concentration; the RK2 update is a convex combination of two
                   such stages.  vectorized=True keeps a step size per sample as a tensor and only synchronises with
                   the host once per `chunk` substeps; vectorized=False is the original loop of the prototypes
                   (one step size for the batch, two .item() syncs per substep).
implicit_euler_linear   (I - dt L) c_new = c for a linear master equation, matrix-free CG: symmetrised CG when the
                   model is reversible (A = 0), CG on the normal equations otherwise.  Positive and mass conserving for
                   any dt (I - dt L is an M-matrix).
proximal_step      variational (JKO-type) implicit step in flux space with the Onsager coefficients frozen at c_n:
                   j* = argmin_j F(c_n + dt S j)/dt + 1/2 sum_h j_h^2 / ell_h - j.A,   c_new = c_n + dt S j*.
                   Newton-CG with Hessian-vector products and a fraction-to-boundary line search.  Exactly
                   conserving, positive, and F(c_new) + dt/2 sum j^2/ell <= F(c_n) when A = 0.
expm_step          c(t + Dt) = exp(Dt L) c for a linear master equation (dense matrix exponential).
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


def positive_heun_rhs(model, c, ctx, Dt, dt_max=None, max_sub=4000, t_tol=1e-9):
    """SSP-RK2 for models that only expose rhs(c, ctx) (the Fisher-Hodge integrator, vectorized): per-sample
    dt <= 0.5 min_m c_m / (-rhs_m) over decreasing components, a substep is rejected and halved while the second
    Euler stage has a non-positive entry."""
    dt_max = Dt / 8 if dt_max is None else dt_max
    B = c.shape[0]
    t = torch.zeros(B, dtype=torch.float64, device=c.device)
    cap = torch.full((B,), float('inf'), dtype=torch.float64, device=c.device)
    nsub = torch.zeros(B, dtype=torch.long, device=c.device)
    chunk = max(1, int(round(Dt / dt_max)))
    it = 0
    while True:
        for _ in range(chunk):
            f0 = model.rhs(c, ctx)
            with torch.no_grad():
                ratio = torch.where(f0 < 0, c / (-f0).clamp_min(1e-30), torch.full_like(c, float('inf')))
                dts = 0.5 * torch.nan_to_num(ratio, nan=float('inf')).amin(-1).double()
                rem = Dt - t
                active = rem > t_tol
                dt = torch.clamp(torch.minimum(torch.minimum(rem, cap), dts), max=dt_max)
                dt = torch.where(active, dt, torch.zeros_like(dt))
            dtc = dt.to(c.dtype)[:, None]
            c1 = c + dtc * f0
            c2 = c1 + dtc * model.rhs(c1, ctx)
            with torch.no_grad():
                ok = (torch.nan_to_num(c2, nan=1.0) > 0).all(-1) | (dt < 1e-12)
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


# ------------------------------------------------------------------------------------------ Patankar schemes
def _patankar_system(hg, a, b, sigma, c0, dt):
    """M x = r for the modified Patankar step  x = c0 + dt S (a w_fwd - b w_bwd), where each one-way channel is
    weighted by x_m / sigma_m of the one component m it consumes (channels that consume nothing stay explicit).
    M = I - dt sum_q S_q (flux_q / sigma_{m(q)}) e_{m(q)}^T is a Z-matrix whose columns sum to 1 + dt (consumption -
    production) / sigma; it is an M-matrix whenever no channel produces more state than it consumes (true for C1, C2,
    C3), and then x > 0 for every dt.  Every moiety l with l^T S = 0 is conserved exactly by construction."""
    t = hg.on(a.device, a.dtype)
    fc, bc = t['fcons'], t['bcons']
    B, M = c0.shape
    sh, sm, sv = t['s_h'], t['s_m'], t['s_val']
    dtc = dt.to(a.dtype)[:, None] if torch.is_tensor(dt) else dt
    Mf = torch.zeros(B, M * M, dtype=a.dtype, device=a.device)
    r = c0.clone()
    for flux, cons, sgn in ((a, fc, 1.0), (b, bc, -1.0)):
        col = cons[sh]                                            # consumer of the channel of each S non-zero
        imp = col >= 0
        colc = col.clamp_min(0)
        val = sgn * sv * flux[:, sh]                              # (B, nnz): this channel's contribution to dc_m/dt
        kappa = val / sigma[:, colc]
        Mf = Mf.index_add(1, (sm * M + colc)[imp], -(dtc * kappa)[:, imp])
        r = r.index_add(1, sm[~imp], (dtc * val)[:, ~imp])
    Mm = Mf.view(B, M, M) + torch.eye(M, dtype=a.dtype, device=a.device)
    return Mm, r


def mprk(model, c, ctx, Dt, nsub=1, order=2, max_split=6):
    """advance c by Dt with nsub modified Patankar steps: order=1 Patankar-Euler, order=2 MPRK22 (alpha = 1,
    Kopecz and Meister 2018).  Linearly implicit (one dense solve per stage, differentiable), positive and
    conserving for any step size on networks with one consumer per channel.  A step whose result is not positive
    (only possible when a channel produces more than it consumes) is redone as two half steps."""
    hg = model.hg
    t = hg.on(c.device, c.dtype)
    if 'fcons' not in t:
        fc, bc = hg.channel_consumers()
        t['fcons'] = torch.as_tensor(fc, device=c.device); t['bcons'] = torch.as_tensor(bc, device=c.device)

    def step(c0, dt, depth):
        a0, b0 = model.rates(c0, ctx)
        M1, r1 = _patankar_system(hg, a0, b0, c0, c0, dt)
        c1 = torch.linalg.solve(M1, r1.unsqueeze(-1)).squeeze(-1)
        cn = c1
        if order == 2:
            ok1 = bool((c1 > 0).all()) if depth < max_split else True
            if ok1:
                a1, b1 = model.rates(c1, ctx)
                M2, r2 = _patankar_system(hg, 0.5 * (a0 + a1), 0.5 * (b0 + b1), c1, c0, dt)
                cn = torch.linalg.solve(M2, r2.unsqueeze(-1)).squeeze(-1)
        if depth < max_split and not bool((cn > 0).all()):
            return step(step(c0, 0.5 * dt, depth + 1), 0.5 * dt, depth + 1)
        return cn

    dt = Dt / nsub
    for _ in range(nsub):
        c = step(c, dt, 0)
    return c, torch.full((c.shape[0],), nsub, dtype=torch.long, device=c.device)


# ------------------------------------------------------------------------------------------ linear implicit / expm
def _linear_ops(model, c_ref, ctx):
    """matrix-free L x and L^T y of the master equation frozen at c_ref (unimolecular)."""
    hg = model.hg
    t = hg.on(c_ref.device, c_ref.dtype)
    qp, qm = model.transition_rates(c_ref, ctx)
    ti, hi = t['tail'], t['head']

    def Lx(x):
        return hg.apply_S(qp * x[:, ti] - qm * x[:, hi])

    def LTy(y):
        g = hg.apply_ST(y)                                    # y_head - y_tail on every hyperedge
        out = torch.zeros_like(y).index_add(1, ti, qp * g)
        return out.index_add(1, hi, -qm * g)
    return Lx, LTy, qp, qm


def cg(Aop, b, x0=None, tol=1e-12, maxiter=1000):
    """batched conjugate gradients for SPD Aop: (B, n) -> (B, n).  Returns (x, iterations, final relative residual)."""
    x = torch.zeros_like(b) if x0 is None else x0.clone()
    r = b - Aop(x)
    p = r.clone()
    rs = (r * r).sum(-1)
    bn = (b * b).sum(-1).clamp_min(1e-300)
    k = 0
    for k in range(1, maxiter + 1):
        Ap = Aop(p)
        alpha = rs / (p * Ap).sum(-1).clamp_min(1e-300)
        x = x + alpha[:, None] * p
        r = r - alpha[:, None] * Ap
        rs_new = (r * r).sum(-1)
        if bool((rs_new <= tol ** 2 * bn).all()):
            rs = rs_new
            break
        p = r + (rs_new / rs.clamp_min(1e-300))[:, None] * p
        rs = rs_new
    return x, k, float((rs / bn).sqrt().max())


@torch.no_grad()
def implicit_euler_linear(model, c, ctx, dt, c_ref=None, tol=1e-12, maxiter=2000, method='auto'):
    """one implicit Euler step (I - dt L) c_new = c of the master equation frozen at c_ref (default c).
    method: 'sym' (symmetrised CG, needs A = 0), 'cgnr' (CG on the normal equations), 'auto'."""
    c_ref = c if c_ref is None else c_ref
    Lx, LTy, qp, qm = _linear_ops(model, c_ref, ctx)
    reversible = bool((ctx['A'].abs().max() < 1e-12).item()) if ctx.get('A') is not None else False
    if method == 'auto':
        method = 'sym' if reversible else 'cgnr'
    if method == 'sym':
        # detailed balance: pi_tail q+ = pi_head q-, pi = exp(-phi); W^{-1/2} (I - dt L) W^{1/2} is symmetric
        phi = model.phi(c_ref, ctx)
        w = torch.exp(-(phi - phi.amin(-1, keepdim=True)))
        sq = w.sqrt()
        op = lambda y: (y * sq - dt * Lx(y * sq)) / sq
        y, it, res = cg(op, c / sq, tol=tol, maxiter=maxiter)
        return y * sq, dict(iters=it, residual=res, method='sym')
    Aop = lambda x: x - dt * Lx(x)
    ATop = lambda y: y - dt * LTy(y)
    x, it, res = cg(lambda x: ATop(Aop(x)), ATop(c), x0=c, tol=tol, maxiter=maxiter)
    return x, dict(iters=it, residual=res, method='cgnr')


@torch.no_grad()
def expm_step(model, c, ctx, Dt, c_ref=None):
    """exact step c(t + Dt) = exp(Dt L) c of the master equation frozen at c_ref (exact for a linear model)."""
    L = model.generator(c if c_ref is None else c_ref, ctx)
    return (torch.linalg.matrix_exp(Dt * L) @ c.unsqueeze(-1)).squeeze(-1)


# ------------------------------------------------------------------------------------------ variational step
def proximal_step(model, c, ctx, dt, tol=1e-10, max_newton=50, cg_tol=1e-8, cg_maxiter=500, frac=0.99):
    """variational implicit step (see module docstring).  Returns (c_new, info).  c, ctx as for rates()."""
    from .model import logmean
    hg = model.hg
    with torch.no_grad():
        a, b = model.rates(c, ctx)
        ell = logmean(a, b).clamp_min(1e-300)
        A = ctx['A']
    c0 = c.detach()

    def Phi(j):
        cn = c0 + dt * hg.apply_S(j)
        return model.free_energy.value(cn, ctx) / dt + 0.5 * (j * j / ell).sum(-1) - (j * A).sum(-1)

    j = torch.zeros_like(ell)
    info = dict(newton=0, cg=0, grad_norm=None)
    for it in range(max_newton):
        with torch.enable_grad():
            jj = j.detach().requires_grad_(True)
            val = Phi(jj)
            g = torch.autograd.grad(val.sum(), jj, create_graph=True)[0]

            def Hv(v):
                return torch.autograd.grad((g * v).sum(), jj, retain_graph=True)[0]
            gn = (g.detach() ** 2 * ell).sum(-1).sqrt()
            if it == 0:
                g0 = gn.clamp_min(1e-300)
            if bool((gn <= tol * g0).all()):
                break
            p, ncg, _ = cg(lambda v: Hv(v).detach(), -g.detach(), tol=cg_tol, maxiter=cg_maxiter)
        info['cg'] += ncg
        with torch.no_grad():
            gd = g.detach()
            bad = (gd * p).sum(-1) >= 0                             # non-convex G: fall back to scaled steepest descent
            p = torch.where(bad[:, None], -gd * ell, p)
            dc = dt * hg.apply_S(p)
            cur = c0 + dt * hg.apply_S(j)
            neg = dc < 0
            amax = torch.where(neg, -frac * cur / torch.where(neg, dc, -torch.ones_like(dc)), torch.full_like(dc, float('inf'))).amin(-1)
            step = torch.clamp(amax, max=1.0)
            f0 = val.detach()
            slope = (g.detach() * p).sum(-1)
            for _ in range(40):                                 # Armijo backtracking, per sample
                jn = j + step[:, None] * p
                fn = Phi(jn)
                ok = fn <= f0 + 1e-4 * step * slope
                if bool(ok.all()):
                    break
                step = torch.where(ok, step, 0.5 * step)
            j = jn
        info['newton'] = it + 1
        info['grad_norm'] = float((gn / g0).max())
    with torch.no_grad():
        c_new = c0 + dt * hg.apply_S(j)
        info['dissipation'] = 0.5 * dt * (j * j / ell).sum(-1)
    return c_new, info
