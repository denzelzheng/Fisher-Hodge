"""Fitting the Neural Chart: the closed-form first step and the rollout refinement.

Closed-form step.  Every one-way rate factors as a_h = K+_h(theta) Phi+_h(c), b_h = K-_h(theta) Phi-_h(c), where
log K+-_h is affine in the linear parameters theta (the log-rate core, and the standard potentials when they are
learned) and Phi carries everything that physics fixes (log c, U / T, the force, the chemostat factors, the current
excess potential).  So dc/dt = S (K+ Phi+ - K- Phi-) is linear in the one-way rate constants.  Group the one-way
channels whose log K is the same function of theta into classes r and integrate over the training trajectories,

    c(t + 4 dt) - c(t) = sum_r kappa_r S int_t^{t + 4 dt} Phi_r dt           (Boole's rule),

with kappa_r >= 0 the factor by which class r has to change.  This is a non-negative least-squares problem (NNLS),
convex, so its optimum is global (fit_linear).  The optimum is mapped back to theta by least squares on log kappa.

Variable projection (fit_varpro).  G_theta and g_theta then descend the same residual, with kappa solved again at every step.  The residual
also holds the error of the quadrature, which no parameter should fit.  The integrator measures that error for the current model: the same
integral along the model's own path from the same start.  Variable projection stops when the rest of the residual is no larger than it.

Rollout refinement (train).  Every parameter is then refined on multi-step rollouts: K in [kmin, kmax] steps from a
random start inside the data, batch 8, Adam with a cosine schedule, gradient clip 1.
"""
import itertools
import math
import os
import time
import numpy as np
import scipy.linalg
import scipy.optimize
from scipy.optimize import nnls
import torch


def _take(drive, idx):
    return {k: v[idx] for k, v in drive.items()}


@torch.no_grad()
def _log_rates(model, c, drive):
    a, b = model.rates(c, model.prepare(drive))
    return torch.log(a[0]).double(), torch.log(b[0]).double()


@torch.no_grad()
def jacobian(model, c_ref, drive_ref):
    """the linear parameters and d log a / d theta, d log b / d theta (H, P each).  Checks that log a, log b are affine
    in theta."""
    params = model.linear_parameters()
    la0, lb0 = _log_rates(model, c_ref, drive_ref)
    cp, cm = [], []
    for p in params:
        flat = p.data.view(-1)
        for i in range(flat.numel()):
            flat[i] += 1.0
            la, lb = _log_rates(model, c_ref, drive_ref)
            flat[i] -= 1.0
            cp.append(la - la0); cm.append(lb - lb0)
    Mp, Mm = torch.stack(cp, 1).cpu().numpy(), torch.stack(cm, 1).cpu().numpy()
    # affine check with a random joint perturbation
    g = torch.Generator().manual_seed(0)
    dl = [0.3 * torch.randn(p.shape, generator=g, dtype=torch.float64).to(p) for p in params]
    for p, d in zip(params, dl):
        p.data += d
    la, lb = _log_rates(model, c_ref, drive_ref)
    for p, d in zip(params, dl):
        p.data -= d
    dv = torch.cat([d.reshape(-1) for d in dl]).double().cpu().numpy()
    err = max(np.abs((la - la0).cpu().numpy() - Mp @ dv).max(), np.abs((lb - lb0).cpu().numpy() - Mm @ dv).max())
    tol = 1e-5 if la0.dtype == torch.float64 and params[0].dtype == torch.float64 else 1e-3
    assert err < tol * max(1.0, np.abs(dv).max()), f'log rates are not affine in the linear parameters ({err:.2e})'
    return params, Mp, Mm


@torch.no_grad()
def set_to_rates(model, c_ref, drive_ref, a_true, b_true):
    """move the linear parameters so that the one-way rates at (c_ref, drive_ref) (batch 1) equal a_true, b_true (H,)
    as closely as the class allows (least squares on the log rates).  Returns the largest remaining log error."""
    params, Mp, Mm = jacobian(model, c_ref, drive_ref)
    la, lb = _log_rates(model, c_ref, drive_ref)
    rhs = np.concatenate([np.log(np.asarray(a_true, np.float64)) - la.cpu().numpy(), np.log(np.asarray(b_true, np.float64)) - lb.cpu().numpy()])
    A = np.concatenate([Mp, Mm])
    delta, *_ = np.linalg.lstsq(A, rhs, rcond=1e-12)
    off = 0
    for p in params:
        n = p.numel()
        p.data += torch.as_tensor(delta[off:off + n], dtype=p.dtype, device=p.device).view(p.shape)
        off += n
    return float(np.abs(A @ delta - rhs).max())


def class_structure(model, c_ref, drive_ref, decimals=6):
    """the linear parameters, the class matrix U (R, P) with log K_r = U_r theta + const, and the class of every
    forward (rp) and backward (rm) one-way channel."""
    params, Mp, Mm = jacobian(model, c_ref, drive_ref)
    H = Mp.shape[0]
    U, inv = np.unique(np.round(np.concatenate([Mp, Mm]), decimals) + 0.0, axis=0, return_inverse=True)
    inv = inv.reshape(-1)
    return params, U, inv[:H], inv[H:]


@torch.no_grad()
def _rule(T1):
    """Boole's rule over four sampling steps, or Simpson's over two when a trajectory is too short"""
    return 'boole' if T1 >= 5 else 'simpson'


def _boole(x, dt):
    return 2 * dt / 45 * (7 * x[:, :-4] + 32 * x[:, 1:-3] + 12 * x[:, 2:-2] + 32 * x[:, 3:-1] + 7 * x[:, 4:])


def _simpson(x, dt):
    return dt / 3 * (x[:, :-2] + 4 * x[:, 1:-1] + x[:, 2:])


def assemble(model, tr, dt, rule=None, batch=16):
    """the normal equations of the derivative fit in the class factors kappa (kappa = 1 is the current model):
    residual(kappa) = (yy - 2 kappa.h + kappa.G.kappa) / yy.  Returns a dict with params, U, rp, rm, fixed, G, h, yy."""
    C = tr['c']
    B, T1, M = C.shape
    assert T1 >= 3, 'need at least three samples per trajectory'
    rule = rule or _rule(T1)
    hg = model.hg
    params, U, rp, rm = class_structure(model, C[:1, 0], _take(tr['drive'], slice(0, 1)))
    R = U.shape[0]
    fixed = np.abs(U).max(1) < 1e-12              # channels whose constant does not depend on theta
    dev = C.device
    S = torch.as_tensor(hg.S, dtype=torch.float64, device=dev)
    Q = S.T @ S
    ii, jj = Q.nonzero(as_tuple=True)
    q = Q[ii, jj]
    rp_t, rm_t = torch.as_tensor(rp, device=dev), torch.as_tensor(rm, device=dev)
    kp, km = torch.as_tensor(fixed[rp], device=dev), torch.as_tensor(fixed[rm], device=dev)
    G = torch.zeros(R, R, dtype=torch.float64, device=dev)
    h = torch.zeros(R, dtype=torch.float64, device=dev)
    yy = torch.zeros((), dtype=torch.float64, device=dev)
    obs = tr.get('obs')
    if obs is not None:                            # observed flux functionals obs = J @ obs_w.T, relative error as in train()
        assert not fixed.any()
        Wo = tr['obs_w'].double().to(dev)
    for s0 in range(0, B, batch):
        idx = torch.arange(s0, min(B, s0 + batch), device=dev)
        ctx = model.prepare(_take(tr['drive'], idx))
        Cb = C[idx]
        ab = [model.rates(Cb[:, t], ctx) for t in range(T1)]
        a = torch.stack([x[0] for x in ab], 1).double(); b = torch.stack([x[1] for x in ab], 1).double()
        Cb = Cb.double()
        if obs is not None:
            yo = obs[idx].double().reshape(-1, Wo.shape[0])
            wt = 1.0 / torch.linalg.vector_norm(yo, dim=-1)
            Ap, Am = a.reshape(-1, hg.H) * wt[:, None], b.reshape(-1, hg.H) * wt[:, None]
            Xo = torch.zeros(Ap.shape[0], Wo.shape[0], R, dtype=torch.float64, device=dev)
            Xo.index_add_(2, rp_t, Ap[:, None, :] * Wo[None]); Xo.index_add_(2, rm_t, -Am[:, None, :] * Wo[None])
            Xf, Yf = Xo.reshape(-1, R), (yo * wt[:, None]).reshape(-1)
            G += Xf.T @ Xf; h += Xf.T @ Yf; yy += (Yf ** 2).sum()
        if rule == 'boole':
            quad = lambda x: _boole(x, dt)
            dC = Cb[:, 4:] - Cb[:, :-4]
        elif rule == 'simpson':
            quad = lambda x: _simpson(x, dt)
            dC = Cb[:, 2:] - Cb[:, :-2]
        elif rule == 'trapezoid':
            quad = lambda x: dt / 2 * (x[:, :-1] + x[:, 1:])
            dC = Cb[:, 1:] - Cb[:, :-1]
        else:
            raise ValueError(rule)
        Ip, Im = quad(a).reshape(-1, hg.H), quad(b).reshape(-1, hg.H)
        dC = dC.reshape(-1, M)
        om = 1.0 / torch.linalg.vector_norm(Cb[:, {'boole': 4, 'simpson': 2}.get(rule, 1):].reshape(-1, M), dim=-1)
        Ip, Im, dC = Ip * om[:, None], Im * om[:, None], dC * om[:, None]       # the relative error of the rollout loss
        if kp.any() or km.any():
            dC = dC - hg.apply_S(Ip * kp - Im * km)
            Ip, Im = Ip * ~kp, Im * ~km
        y = hg.apply_ST(dC)
        h.index_add_(0, rp_t, (y * Ip).sum(0)); h.index_add_(0, rm_t, -(y * Im).sum(0))
        yy += (dC ** 2).sum()
        for (r1, X1, s1), (r2, X2, s2) in itertools.product([(rp_t, Ip, 1.0), (rm_t, Im, -1.0)], repeat=2):
            G.index_put_((r1[ii], r2[jj]), s1 * s2 * q * (X1[:, ii] * X2[:, jj]).sum(0), accumulate=True)
    return dict(params=params, U=U, rp=rp, rm=rm, fixed=fixed, G=G.cpu().numpy(), h=h.cpu().numpy(), yy=float(yy))


def nnls_kappa(G, h, fixed):
    """the non-negative least-squares class factors (1 for classes the data do not see) and the mask of seen classes"""
    R = len(h)
    free = ~fixed & (np.diag(G) > 1e-14 * max(np.diag(G).max(), 1e-300))  # classes the data see
    Gu, hu = G[np.ix_(free, free)], h[free]
    d = 1.0 / np.sqrt(np.diag(Gu))
    Gs, hs = d[:, None] * Gu * d[None], d * hu
    Lc = np.linalg.cholesky(Gs + 1e-12 * np.eye(len(hs)))
    ks, _ = nnls(Lc.T, scipy.linalg.solve_triangular(Lc, hs, lower=True), maxiter=50 * len(hs))
    kap = np.ones(R)
    kap[free] = d * ks
    return kap, free


FLOOR = 1e-3            # the lowest rate constant of a class in the closed-form step, relative to the largest class factor


@torch.no_grad()
def fit_linear(model, data, dt, rule=None, floor=FLOOR, batch=16, log=print):
    """set the linear parameters of model to the non-negative least-squares optimum on data (a training set dict
    with c (B, T+1, M) and drive, or a dict holding it under 'train').  dt: sampling interval.  The log rate of no class starts below
    log(floor * largest fitted factor): the optimum may have a factor at zero, and a class whose rate is that small has no gradient.
    The standard potentials stay free, so a one-way channel can still be near zero.  Returns diagnostics: the relative residual of
    the derivative fit before (kappa = 1) and after, and the fitted factors."""
    t0 = time.time()
    tr = data['train'] if 'train' in data else data
    A = assemble(model, tr, dt, rule, batch)
    params, U, G, h, yy = A['params'], A['U'], A['G'], A['h'], A['yy']
    R = U.shape[0]
    kap, free = nnls_kappa(G, h, A['fixed'])
    res = lambda k: (yy - 2 * k @ h + k @ G @ k) / yy
    # back to theta: log kappa_r = U_r delta (classes not seen by the data keep their value)
    zero = 1e-6                                    # the logarithm of a factor at zero in the targets below
    tgt = np.log(np.maximum(kap, zero))
    delta, *_ = np.linalg.lstsq(U[free], tgt[free], rcond=1e-10)
    proj_err = float(np.abs(U[free] @ delta - tgt[free]).max()) if free.any() else 0.0
    log_rates = getattr(getattr(model, 'kinetics', None), 'core', None)       # the floor is on the class log rates of a chart; other models have none
    nc = 0 if log_rates is None else log_rates.numel()
    core = np.zeros(0) if log_rates is None else log_rates.detach().cpu().numpy().astype(np.float64)
    theta = torch.cat([p.detach().reshape(-1) for p in params]).cpu().numpy().astype(np.float64)
    top = (kap[free] * np.exp(U[free] @ theta)).max() if free.any() else 1.0   # the largest fitted factor against unit rates
    low = np.log(floor * top) - core if free.any() else np.full(nc, -np.inf)
    if proj_err > 1e-8 or (delta[:nc] < low).any():
        # the classes are coupled (fewer independent directions in theta than classes, e.g. a given standard
        # affinity): weight the projection by the curvature of the residual, G_rr kappa_r^2, so that classes the data
        # barely see do not dominate, then minimise the residual itself over theta
        w = np.sqrt(np.diag(G)[free] * np.maximum(kap[free], zero) ** 2) + 1e-300
        delta, *_ = np.linalg.lstsq(w[:, None] * U[free], w * tgt[free], rcond=1e-10)

        def f(dl):
            k = np.where(free, np.exp(U @ dl), 1.0)
            g = (2 * (G @ k) - 2 * h) / yy
            return res(k), U.T @ (np.where(free, k, 0.0) * g)
        bounds = [(x, None) for x in low] + [(None, None)] * (U.shape[1] - nc)
        delta = scipy.optimize.minimize(f, delta, jac=True, method='L-BFGS-B', bounds=bounds, options=dict(maxiter=2000, ftol=1e-15, gtol=1e-12)).x
        proj_err = float(np.abs(U[free] @ delta - tgt[free]).max())
    k_eff = np.where(free, np.exp(U @ delta), 1.0)
    off = 0
    for p in params:
        n = p.numel()
        p.data += torch.as_tensor(delta[off:off + n], dtype=p.dtype, device=p.device).view(p.shape)
        off += n
    info = dict(n_classes=int(R), n_free=int(free.sum()), n_params=int(U.shape[1]), rank=int(np.linalg.matrix_rank(U[free])),
                residual_init=float(res(np.ones(R))), residual_nnls=float(res(kap)), residual_theta=float(res(k_eff)),
                n_floored=int((kap[free] < zero).sum()), n_at_floor=int((delta[:nc] <= low + 1e-9).sum()), floor=float(floor * top), projection_err=proj_err, kappa=kap.tolist(), time_s=time.time() - t0)
    log(f'fit_linear: {R} classes ({info["n_free"]} seen, {info["n_floored"]} factors at zero, {info["n_at_floor"]} log rates at the floor), {U.shape[1]} parameters, '
        f'rank {info["rank"]}; relative derivative residual {info["residual_init"]:.3e} -> {info["residual_nnls"]:.3e} '
        f'(NNLS) -> {info["residual_theta"]:.3e} (theta, projection error {proj_err:.1e}); {info["time_s"]:.1f}s')
    return info


# ------------------------------------------------------------------------------------------ variable projection
VP_STEPS, VP_LR, VP_CHECK = 500, 1e-3, 25


def _projected_residual_backward(model, tr, dt, batch=16):
    """the relative derivative residual of fit_linear at the current parameters as a function of every parameter, in FP32; backpropagates it,
    batch by batch, and returns its value"""
    C = tr['c'].float()
    drive = {k: v.float() if torch.is_tensor(v) and torch.is_floating_point(v) else v for k, v in tr['drive'].items()}
    B, T1, M = C.shape
    hg = model.hg
    w, quad = (4, _boole) if _rule(T1) == 'boole' else (2, _simpson)
    yy = torch.zeros((), dtype=torch.float64, device=C.device)
    for s0 in range(0, B, batch):
        Cb = C[s0:s0 + batch]
        om = 1.0 / torch.linalg.vector_norm(Cb[:, w:], dim=-1)
        yy += (((Cb[:, w:] - Cb[:, :-w]) * om[..., None]) ** 2).sum().double()
    tot = 0.0
    for s0 in range(0, B, batch):
        idx = torch.arange(s0, min(B, s0 + batch), device=C.device)
        ctx = model.prepare(_take(drive, idx))
        Cb = C[idx]
        J = torch.stack([(lambda ab: ab[0] - ab[1])(model.rates(Cb[:, t], ctx)) for t in range(T1)], 1)
        pred = hg.apply_S(quad(J, dt).reshape(-1, hg.H)).reshape(len(idx), T1 - w, M)
        om = 1.0 / torch.linalg.vector_norm(Cb[:, w:], dim=-1)
        loss = ((((Cb[:, w:] - Cb[:, :-w]) - pred) * om[..., None]) ** 2).sum().double() / yy
        loss.backward()
        tot += float(loss.detach())
    return tot


@torch.no_grad()
def quadrature_check(model, tr, dt, batch=16):
    """the relative derivative residual and the quadrature error in it, on windows that do not overlap.  The quadrature error compares the
    integral along the data with the same integral along the model's own path from the same start, which the integrator computes."""
    C = tr['c'].float()
    drive = {k: v.float() if torch.is_tensor(v) and torch.is_floating_point(v) else v for k, v in tr['drive'].items()}
    B, T1, M = C.shape
    hg = model.hg
    w, quad = (4, _boole) if _rule(T1) == 'boole' else (2, _simpson)
    res = err = yy = 0.0
    for s0 in range(0, B, batch):
        idx = torch.arange(s0, min(B, s0 + batch), device=C.device)
        ctx = model.prepare(_take(drive, idx))
        Cb = C[idx]
        J = torch.stack([(lambda ab: ab[0] - ab[1])(model.rates(Cb[:, t], ctx)) for t in range(T1)], 1)
        I = quad(J, dt)
        for t in range(0, T1 - w, w):
            c = Cb[:, t]
            for _ in range(w):
                c, _n = model.advance(c, ctx, dt)
            pred = hg.apply_S(I[:, t])
            dC = Cb[:, t + w] - Cb[:, t]
            om = 1.0 / torch.linalg.vector_norm(Cb[:, t + w], dim=-1)[:, None]
            res += float(((pred - dC) * om).pow(2).sum())
            err += float(((pred - (c - Cb[:, t])) * om).pow(2).sum())
            yy += float((dC * om).pow(2).sum())
    return res / yy, err / yy


def fit_varpro(model, tr, dt, steps=VP_STEPS, lr=VP_LR, batch=16, log=print, trace=None):
    """G_theta and g_theta descend the derivative residual jointly, with the linear parameters at their closed-form optimum (fit_linear) at every
    step: by the envelope theorem the gradient at the optimal linear parameters is that of the projected residual.  Adam, a fixed number of
    steps at most, gradient clip 1.  The networks train in FP32, the closed-form step stays in float64.  The descent stops when the part of the
    residual beyond the quadrature error is no larger than that error, the point past which it would fit the quadrature."""
    params = list(model.free_energy.excess_parameters()) + (list(model.kinetics.net.parameters()) if model.kinetics.net is not None else [])
    params = [p for p in params if p.requires_grad]
    t0 = time.time()
    quiet = lambda *a: None
    info = fit_linear(model, tr, dt, log=quiet)
    if not params:
        return dict(linear_fit=info, steps=0, time_s=time.time() - t0)
    opt = torch.optim.Adam(params, lr=lr)
    hist = []
    start, stopped = info['residual_theta'], None
    for it in range(steps):
        fit_linear(model, tr, dt, log=quiet)
        if it % VP_CHECK == 0:
            res, err = quadrature_check(model, tr, dt, batch)
            if res <= 2 * err:
                stopped = it
                log(f'varpro it {it:4d}: residual {res:.4e}, quadrature error {err:.4e}: stops')
                break
        opt.zero_grad(set_to_none=True)
        with torch.enable_grad():
            hist.append(_projected_residual_backward(model, tr, dt, batch))
        if it % 50 == 0:
            log(f'varpro it {it:4d}: projected derivative residual {hist[-1]:.4e} t {time.time() - t0:.0f}s')
            if trace is not None:
                trace(it, hist[-1])
        torch.nn.utils.clip_grad_norm_(params, 1.0)
        opt.step()
    for p in model.parameters():
        p.grad = None
    info = fit_linear(model, tr, dt, log=log)
    done = len(hist)
    log(f'varpro: {done} steps, projected derivative residual {start:.4e} -> {info["residual_theta"]:.4e}, {time.time() - t0:.0f}s')
    return dict(linear_fit=info, steps=done, stopped_at=stopped, lr=lr, residual_start=start, residual_end=info['residual_theta'], curve=hist[::10], time_s=time.time() - t0)


# ------------------------------------------------------------------------------------------ rollout training
def take(drive, idx):
    return {k: v[idx] for k, v in drive.items()}


def _save_ckpt(path, state):
    tmp = path + '.tmp'
    torch.save(state, tmp)
    os.replace(tmp, path)                          # a killed job leaves the previous complete checkpoint


def train(model, tr, iters, seed, dt, batch=8, lr=1e-3, kmin=4, kmax=20, log=print, budget_s=12 * 3600.0, eta_min=1e-5, ckpt=None,
          resume=False, ckpt_every=100, window=None):
    """rollout training, the one trainer of every model: K in [kmin, kmax] steps from a random start, batch `batch`, relative L2 loss,
    Adam with a cosine learning rate on p = max(step / iters, elapsed / budget), gradient clip 1.  tr = dict(c (B, T + 1, M), drive).  ckpt:
    a checkpoint every ckpt_every steps; resume=True continues exactly from it."""
    rng = np.random.default_rng(10_000 + seed)
    Cs = tr['c']; ntr = Cs.shape[0]
    window = Cs.shape[1] - 1 if window is None else window
    params = [p for p in model.parameters() if p.requires_grad]
    opt = torch.optim.Adam(params, lr=lr)
    hist, subs, smax, nskip = [], [], [], 0
    t0 = time.time()
    it, p, trunc = 0, 0.0, False
    st0 = getattr(getattr(model, '_fast', None), 'stepper', None)
    if resume:
        ck = torch.load(ckpt, map_location=params[0].device, weights_only=False)
        model.load_state_dict(ck['model']); opt.load_state_dict(ck['opt'])
        rng.bit_generator.state = ck['rng']
        it, hist, subs, smax, nskip = ck['it'], ck['hist'], ck['subs'], ck['smax'], ck['nskip']
        t0 = time.time() - ck['elapsed']
        if st0 is not None:
            st0.cap_hits = ck.get('cap_hits', 0)
        log(f'resumed from {ckpt}: step {it}, elapsed {ck["elapsed"]:.0f}s')
    model.train()
    while it < iters:
        p = max(it / iters, (time.time() - t0) / budget_s)
        if p >= 1.0:
            trunc = True
            break
        for g in opt.param_groups:
            g['lr'] = eta_min + (lr - eta_min) * 0.5 * (1 + math.cos(math.pi * p))
        idx = torch.tensor(rng.choice(ntr, min(batch, ntr), replace=False), device=Cs.device)
        K = int(rng.integers(kmin, min(kmax, window) + 1))
        st = torch.tensor(rng.integers(0, window - K + 1, size=len(idx)), device=Cs.device)
        ctx = model.prepare(take(tr['drive'], idx))
        c = Cs[idx, st]
        loss = 0.0; ns = 0.0; nmax = 0
        for k in range(1, K + 1):
            c, n = model.advance(c, ctx, dt)
            ns = ns + (n.float().mean() if torch.is_tensor(n) else n)
            nmax = max(nmax, int(n.max()) if torch.is_tensor(n) else int(n))
            y = Cs[idx, st + k]
            loss = loss + (((c - y) ** 2).sum(-1) / (y ** 2).sum(-1)).mean()
        loss = loss / K
        subs.append(float(ns) / K); smax.append(nmax)
        if not torch.isfinite(loss):
            nskip += 1; opt.zero_grad(set_to_none=True); it += 1; continue
        opt.zero_grad(set_to_none=True); loss.backward()
        gn = torch.nn.utils.clip_grad_norm_(params, 1.0)
        opt.step()
        hist.append(float(loss.detach()))
        if it % 100 == 0 or it == iters - 1:
            from . import fast
            log(f'it {it:5d} loss {np.mean(hist[-100:]):.4e} gnorm {float(gn):.2e} K {K} substeps/DT {np.mean(subs[-100:]):.1f} '
                f'(max {max(smax[-100:])}) lr {opt.param_groups[0]["lr"]:.2e} p {p:.3f} skipped {nskip} t {time.time() - t0:.0f}s'
                + fast.cap_report(model))
        it += 1
        if ckpt is not None and it % ckpt_every == 0:
            _save_ckpt(ckpt, dict(model=model.state_dict(), opt=opt.state_dict(), rng=rng.bit_generator.state, it=it, hist=hist, subs=subs,
                                  smax=smax, nskip=nskip, elapsed=time.time() - t0, cap_hits=st0.cap_hits if st0 is not None else 0))
    el = time.time() - t0
    log(f'training ended: {it} steps, {el:.0f}s, truncated by the budget: {trunc}')
    if ckpt is not None and os.path.exists(ckpt):
        os.remove(ckpt)
    return dict(train_time_s=el, final_train_loss=float(np.mean(hist[-100:])) if hist else None, train_nskip=nskip,
                train_substeps_per_DT=float(np.mean(subs)) if subs else None, train_substeps_max=int(max(smax)) if smax else None,
                train_steps_done=it, truncated=trunc, budget_s=budget_s)


@torch.no_grad()
def rollout(model, drive, c0, nsteps, dt=0.05, sigma=False):
    """returns (states (B, T+1, M), mean substeps per step, min over the rollout of the per-sample sigma or None)"""
    ctx = model.prepare(drive)
    c = c0.clone(); out = [c]; nsub = []; smin = float('inf')
    for _ in range(nsteps):
        if sigma:
            a, b = model.rates(c, ctx)
            smin = min(smin, float(((a - b) * (torch.log(a) - torch.log(b))).sum(-1).min()))
        c, n = model.advance(c, ctx, dt)
        nsub.append(n.float().mean() if torch.is_tensor(n) else torch.tensor(float(n)))
        out.append(c)
    return torch.stack(out, 1), torch.stack([x.cpu() for x in nsub]).mean().item(), (smin if sigma else None)


def relerr(R, Y):
    B, T1 = Y.shape[:2]
    return (torch.linalg.vector_norm((R - Y).reshape(B, T1, -1), dim=-1) / torch.linalg.vector_norm(Y.reshape(B, T1, -1), dim=-1))


# ------------------------------------------------------------------------------------------ the Neural Chart fit
@torch.no_grad()
def chart_summary(model, tr, every=5, batch=16):
    """the largest |grad G| (the excess potential; per species and overall) and the largest |g_theta| over the states of the
    data tr"""
    fe, kin = model.free_energy, model.kinetics
    C = tr['c'][:, ::every]
    B, T, M = C.shape
    lg, gm = torch.zeros(fe.NS, dtype=torch.float64), 0.0
    for s0 in range(0, B, batch):
        idx = torch.arange(s0, min(B, s0 + batch), device=C.device)
        feats = model.prepare(_take(tr['drive'], idx))['feats']
        for t in range(T):
            c = C[idx, t]
            if fe.c_dependent:
                lg = torch.maximum(lg, fe.correction(c, None).abs().view(len(idx), fe.NS, fe.V).amax((0, 2)).double().cpu())
            if kin.net is not None:
                gm = max(gm, float(kin.state_term(feats['edge_sym'], c).abs().max()))
    return dict(max_abs_grad_G_species=lg.tolist(), max_abs_grad_G=float(lg.max()), max_abs_g=gm,
                params_G=sum(p.numel() for p in fe.excess_parameters()),
                params_g=sum(p.numel() for p in kin.net.parameters()) if kin.net is not None else 0)
