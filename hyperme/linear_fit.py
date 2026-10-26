"""Closed-form initial value of the rate constants: the part of the model that enters linearly.

In the wired mode every one-way rate factors as a_h = K+_h(theta) Phi+_h(c), b_h = K-_h(theta) Phi-_h(c), where
log K+-_h is affine in theta = (log-rate core, standard potentials mu0, chemostat mu0) and Phi carries everything that
is wired (log c, U / T, the force line integrals, the chemostat factors, the current non-ideal correction).  So
dc/dt = S (K+ Phi+ - K- Phi-) is linear in the one-way rate constants.  Group the one-way channels whose log K is the
same function of theta into classes r (a reaction type in one direction, both directions of a transport type, or a
single hyperedge for a per-edge core) and integrate over the sampled training trajectories,

    c(t + 2 dt) - c(t) = sum_r kappa_r S int_t^{t + 2 dt} Phi_r dt           (Simpson rule, or trapezoid over dt),

with kappa_r >= 0 the factor by which class r has to change.  This is a non-negative least-squares problem, convex,
so its optimum is global.  The optimum is mapped back to theta by least squares on log kappa (exact when the classes
have independent directions in theta, as for the C2, C3 and C5 presets).  The estimator is SINDy-like regression on
the model's own dictionary; it is the default first step of fit(), which then trains with the ordinary rollout loss.
A CRNN whose log rate constants are a table is linear in the same way and gets the same initial value."""
import itertools
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
        # the classes are coupled (fewer independent directions in theta than classes, e.g. a wired standard
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


def fit(model, data, iters, dt, seed=0, linear_init=True, log=print, **train_kw):
    """the estimator: the closed-form linear step (if the model has one and linear_init) followed by the rollout
    training of hyperme.train.train on data['train'] (same optimiser, schedule, batch and loss for every model)."""
    from .train import train
    info = {}
    if linear_init and hasattr(model, 'linear_parameters') and model.linear_parameters():
        info['linear_fit'] = fit_linear(model, data, dt, log=log)
    info.update(train(model, data['train'], iters, seed, dt, log=log, **train_kw))
    return info
