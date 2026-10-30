"""Fitting the Neural Chart: the closed-form first step and the rollout refinement.

Closed-form step.  Every one-way rate factors as a_h = K+_h(theta) Phi+_h(c), b_h = K-_h(theta) Phi-_h(c), where
log K+-_h is affine in the linear parameters theta (the log-rate core, and the standard potentials when they are
learned) and Phi carries everything that physics fixes (log c, U / T, the force, the chemostat factors, the current
activity correction).  So dc/dt = S (K+ Phi+ - K- Phi-) is linear in the one-way rate constants.  Group the one-way
channels whose log K is the same function of theta into classes r and integrate over the training trajectories,

    c(t + 2 dt) - c(t) = sum_r kappa_r S int_t^{t + 2 dt} Phi_r dt           (Simpson rule),

with kappa_r >= 0 the factor by which class r has to change.  This is a non-negative least-squares problem (NNLS),
convex, so its optimum is global (fit_linear).  The optimum is mapped back to theta by least squares on log kappa.

Variable projection (fit_varpro).  When the model has a learned state function inside mu (the activity correction
ln gamma_theta), omega is the NNLS optimum at every value of ln gamma_theta, and ln gamma_theta descends the projected
derivative residual.

Rollout refinement (train).  Every parameter is then refined on multi-step rollouts: K in [kmin, kmax] steps from a
random start inside the training window, batch 8, Adam with a cosine schedule, gradient clip 1.
"""
import itertools
import math
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
def assemble(model, tr, dt, rule='simpson', batch=16):
    """the normal equations of the derivative fit in the class factors kappa (kappa = 1 is the current model):
    residual(kappa) = (yy - 2 kappa.h + kappa.G.kappa) / yy.  Returns a dict with params, U, rp, rm, fixed, G, h, yy."""
    C = tr['c']
    B, T1, M = C.shape
    assert T1 >= 3, 'need at least three samples per trajectory'
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
        if rule == 'simpson':
            quad = lambda x: dt / 3 * (x[:, :-2] + 4 * x[:, 1:-1] + x[:, 2:])
            dC = Cb[:, 2:] - Cb[:, :-2]
        elif rule == 'trapezoid':
            quad = lambda x: dt / 2 * (x[:, :-1] + x[:, 1:])
            dC = Cb[:, 1:] - Cb[:, :-1]
        else:
            raise ValueError(rule)
        Ip, Im = quad(a).reshape(-1, hg.H), quad(b).reshape(-1, hg.H)
        dC = dC.reshape(-1, M)
        om = 1.0 / torch.linalg.vector_norm((Cb[:, 2:] if rule == 'simpson' else Cb[:, 1:]).reshape(-1, M), dim=-1)
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


@torch.no_grad()
def fit_linear(model, data, dt, rule='simpson', floor=1e-6, batch=16, log=print):
    """set the linear parameters of model to the non-negative least-squares optimum on data (a training set dict
    with c (B, T+1, M) and drive, or a dict holding it under 'train').  dt: sampling interval.  Returns diagnostics:
    the relative residual of the derivative fit before (kappa = 1) and after, and the fitted factors."""
    t0 = time.time()
    tr = data['train'] if 'train' in data else data
    A = assemble(model, tr, dt, rule, batch)
    params, U, G, h, yy = A['params'], A['U'], A['G'], A['h'], A['yy']
    R = U.shape[0]
    kap, free = nnls_kappa(G, h, A['fixed'])
    res = lambda k: (yy - 2 * k @ h + k @ G @ k) / yy
    # back to theta: log kappa_r = U_r delta (classes not seen by the data keep their value)
    tgt = np.log(np.maximum(kap, floor))
    delta, *_ = np.linalg.lstsq(U[free], tgt[free], rcond=1e-10)
    proj_err = float(np.abs(U[free] @ delta - tgt[free]).max()) if free.any() else 0.0
    if proj_err > 1e-8:
        # the classes are coupled (fewer independent directions in theta than classes, e.g. a given standard
        # affinity): weight the projection by the curvature of the residual, G_rr kappa_r^2, so that classes the data
        # barely see do not dominate, then minimise the residual itself over theta
        w = np.sqrt(np.diag(G)[free] * np.maximum(kap[free], floor) ** 2) + 1e-300
        delta, *_ = np.linalg.lstsq(w[:, None] * U[free], w * tgt[free], rcond=1e-10)

        def f(dl):
            k = np.where(free, np.exp(U @ dl), 1.0)
            g = (2 * (G @ k) - 2 * h) / yy
            return res(k), U.T @ (np.where(free, k, 0.0) * g)
        delta = scipy.optimize.minimize(f, delta, jac=True, method='L-BFGS-B', options=dict(maxiter=2000, ftol=1e-15, gtol=1e-12)).x
        proj_err = float(np.abs(U[free] @ delta - tgt[free]).max())
    k_eff = np.where(free, np.exp(U @ delta), 1.0)
    off = 0
    for p in params:
        n = p.numel()
        p.data += torch.as_tensor(delta[off:off + n], dtype=p.dtype, device=p.device).view(p.shape)
        off += n
    info = dict(n_classes=int(R), n_free=int(free.sum()), n_params=int(U.shape[1]), rank=int(np.linalg.matrix_rank(U[free])),
                residual_init=float(res(np.ones(R))), residual_nnls=float(res(kap)), residual_theta=float(res(k_eff)),
                n_floored=int((kap[free] < floor).sum()), projection_err=proj_err, kappa=kap.tolist(), time_s=time.time() - t0)
    log(f'fit_linear: {R} classes ({info["n_free"]} seen, {info["n_floored"]} at the floor), {U.shape[1]} parameters, '
        f'rank {info["rank"]}; relative derivative residual {info["residual_init"]:.3e} -> {info["residual_nnls"]:.3e} '
        f'(NNLS) -> {info["residual_theta"]:.3e} (theta, projection error {proj_err:.1e}); {info["time_s"]:.1f}s')
    return info


def _projected_residual_backward(model, tr, dt, batch=16):
    """the relative derivative residual of fit_linear (Simpson rule, the same weights) at the current parameters, as
    a differentiable function of every parameter; backpropagates it (batch by batch) and returns its value"""
    C = tr['c']
    B, T1, M = C.shape
    hg = model.hg
    yy = 0.0
    for s0 in range(0, B, batch):
        Cb = C[s0:s0 + batch].double()
        om = 1.0 / torch.linalg.vector_norm(Cb[:, 2:], dim=-1)
        yy += float((((Cb[:, 2:] - Cb[:, :-2]) * om[..., None]) ** 2).sum())
    tot = 0.0
    for s0 in range(0, B, batch):
        idx = torch.arange(s0, min(B, s0 + batch), device=C.device)
        ctx = model.prepare(_take(tr['drive'], idx))
        Cb = C[idx]
        J = torch.stack([(lambda ab: ab[0] - ab[1])(model.rates(Cb[:, t], ctx)) for t in range(T1)], 1).double()
        I = dt / 3 * (J[:, :-2] + 4 * J[:, 1:-1] + J[:, 2:])
        pred = hg.apply_S(I.reshape(-1, hg.H)).reshape(len(idx), T1 - 2, M)
        Cd = Cb.double()
        om = 1.0 / torch.linalg.vector_norm(Cd[:, 2:], dim=-1)
        loss = ((((Cd[:, 2:] - Cd[:, :-2]) - pred) * om[..., None]) ** 2).sum() / yy
        loss.backward()
        tot += float(loss.detach())
    return tot


def fit_varpro(model, data, dt, params, max_iters=2000, lr=1e-2, window=100, min_gain=0.05, batch=16, log=print):
    """variable-projected closed-form fit.  The linear parameters (the log-rate core, given or learned potentials) are
    the NNLS optimum (fit_linear) at the current value of the nonlinear state-function parameters `params` (e.g. the
    learned activity correction ln gamma_theta); `params` descend the projected derivative residual (Adam, lr, clip 1;
    by the envelope theorem the gradient at fixed optimal linear parameters is the gradient of the projected residual).
    Fixed stopping rule: the projected residual improved by less than min_gain (relative) over the last `window`
    iterations, or max_iters.  Without nonlinear parameters this is exactly fit_linear."""
    tr = data['train'] if 'train' in data else data
    t0 = time.time()
    if not params:
        return dict(linear_fit=fit_linear(model, tr, dt, log=log), iters=0, time_s=time.time() - t0)
    quiet = lambda *a: None
    opt = torch.optim.Adam(params, lr=lr)
    hist, it = [], 0
    while True:
        fit_linear(model, tr, dt, log=quiet)                      # omega: the NNLS optimum at the current state functions
        opt.zero_grad(set_to_none=True)
        with torch.enable_grad():
            hist.append(_projected_residual_backward(model, tr, dt, batch))
        if it % 50 == 0:
            log(f'varpro it {it:4d}: projected derivative residual {hist[-1]:.4e} t {time.time() - t0:.0f}s')
        stop = (it >= window and hist[-1] > (1 - min_gain) * hist[-1 - window]) or it >= max_iters or not math.isfinite(hist[-1])
        if stop:
            break
        torch.nn.utils.clip_grad_norm_(params, 1.0)
        opt.step()
        it += 1
    for p in model.parameters():
        p.grad = None
    info = fit_linear(model, tr, dt, log=log)                     # final omega at the final state functions
    out = dict(linear_fit=info, iters=it, residual_start=hist[0], residual_end=hist[-1], time_s=time.time() - t0,
               curve=hist[::10], rule=f'stop when the projected residual improves by < {min_gain:g} over {window} iterations '
                                      f'(max {max_iters}); Adam lr {lr:g}, clip 1')
    log(f'varpro: {it} iterations, projected derivative residual {hist[0]:.4e} -> {hist[-1]:.4e}, {out["time_s"]:.0f}s')
    return out


# ------------------------------------------------------------------------------------------ rollout training
def take(drive, idx):
    return {k: v[idx] for k, v in drive.items()}


def train(model, tr, iters, seed, dt, batch=8, lr=1e-3, kmin=4, kmax=8, log=print, callback=None, window=None):
    """rollout training.  If tr has 'obs' (B, T+1, k) and 'obs_w' (k, H), the observables obs = J @ obs_w.T (linear
    functionals of the net fluxes, e.g. the turnover of a reservoir) add their relative error to the loss with weight 1.
    callback(it, loss) is called after every step (loss: float, nan for a skipped step).  window: the number of sampling
    steps of every training trajectory a rollout may use (default: all of tr['c'])."""
    obs, Wo = tr.get('obs'), tr.get('obs_w')
    rng = np.random.default_rng(10_000 + seed)
    C = tr['c']; ntr = C.shape[0]
    window = C.shape[1] - 1 if window is None else window
    opt = torch.optim.Adam(model.parameters(), lr=lr)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, iters, eta_min=1e-5)
    hist, subs, nskip = [], [], 0
    t0 = time.time()
    model.train()
    for it in range(iters):
        idx = torch.tensor(rng.choice(ntr, min(batch, ntr), replace=False), device=C.device)
        K = int(rng.integers(kmin, kmax + 1))
        st = torch.tensor(rng.integers(0, window - K + 1, size=len(idx)), device=C.device)
        ctx = model.prepare(take(tr['drive'], idx))
        c = C[idx, st]
        loss = 0.0; ns = 0
        for k in range(1, K + 1):
            c, n = model.advance(c, ctx, dt)
            ns = ns + (n.float().mean() if torch.is_tensor(n) else n)
            y = C[idx, st + k]
            loss = loss + (((c - y) ** 2).sum(-1) / (y ** 2).sum(-1)).mean()
            if obs is not None:
                a, b = model.rates(c, ctx)
                yo = obs[idx, st + k]
                loss = loss + ((((a - b) @ Wo.T - yo) ** 2).sum(-1) / (yo ** 2).sum(-1)).mean()
        loss = loss / K
        subs.append(ns / K)
        if not torch.isfinite(loss):
            nskip += 1; opt.zero_grad(set_to_none=True); sched.step()
            if callback is not None:
                callback(it, float('nan'))
            continue
        opt.zero_grad(set_to_none=True); loss.backward()
        gn = torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        opt.step(); sched.step()
        hist.append(float(loss.detach()))
        if callback is not None:
            callback(it, hist[-1])
        if it % 100 == 0 or it == iters - 1:
            log(f'it {it:5d} loss {np.mean(hist[-100:]):.4e} gnorm {float(gn):.2e} K {K} '
                f'substeps/DT {float(np.mean([float(x) for x in subs[-100:]])):.1f} skipped {nskip} t {time.time() - t0:.0f}s')
    return dict(train_time_s=time.time() - t0, final_train_loss=float(np.mean(hist[-100:])) if hist else None, train_nskip=nskip,
                train_substeps_per_DT=float(np.mean([float(x) for x in subs])) if subs else None,
                loss_curve=[float(np.mean(hist[i:i + 100])) for i in range(0, len(hist), 100)])


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
def state_function_parameters(model):
    """the nonlinear parameters of the learned thermodynamic state functions (the activity correction ln gamma_theta);
    empty for every configuration without one"""
    fe = getattr(model, 'free_energy', None)
    if fe is not None and getattr(fe, 'residual', False) and hasattr(fe, 's_mu'):
        return [fe.r_alpha, fe.r_beta, fe.r_delta, fe.s_mu]
    return []


def fit_chart(model, tr, dt, iters, seed=0, log=print, trainer=None, after_linear=None, **train_kw):
    """the four-step fit of the Neural Chart: variable-projected NNLS (omega in closed form at the current state
    functions; at initialisation g = 0 and s = 0, so k is linear in omega), then the rollout refinement of every
    parameter.  With no learned state function the first step is exactly the NNLS step.
    trainer(model, tr, iters, seed, dt, log=log, **train_kw) (default: train); returns the info dict.  When the
    projection ran, its time is passed to the trainer as elapsed0 (it counts in the training time and budget)."""
    cfg = model.config
    info = {}
    if not cfg.train:                                    # oracle: nothing is fitted
        return info
    if cfg.init == 'nnls':
        assert model.linear_parameters(), 'no linear parameters for the closed-form step'
        info['linear_fit'] = fit_linear(model, dict(train=tr), dt, log=log)
        nonlin = state_function_parameters(model) if cfg.varpro else []
        if nonlin:
            info['varpro'] = fit_varpro(model, dict(train=tr), dt, nonlin, log=log)
            info['varpro'].pop('linear_fit', None)
            train_kw = dict(train_kw, elapsed0=info['varpro']['time_s'])
        if after_linear is not None:
            after_linear(model)
    if iters > 0:
        info.update((trainer or train)(model, tr, iters, seed, dt, log=log, **train_kw))
    return info
