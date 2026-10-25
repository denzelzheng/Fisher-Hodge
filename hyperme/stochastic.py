"""Stochastic layer: sample the chemical master equation defined by a model's (learned) rates.

System size Omega: molecule numbers n = Omega c.  Hyperedge h fires forward with propensity Omega a_h(n / Omega) and
backward with propensity Omega b_h(n / Omega) (Kurtz scaling; for mass action with distinct reactants this is the
usual k n_i n_j / Omega).  The rates come from any callable rates(c) -> (a, b), e.g. a trained HyperME with its
drive context, so the samplers use the learned free energy, kinetics and drive.

gillespie          batched direct method (independent trajectories in a batch, one event per trajectory per
                   iteration, works on CPU or GPU).  Records n at the sampling times and the integrated net
                   currents Q_h (forward minus backward firings).
chemical_langevin  Euler-Maruyama for dc = S(a - b) dt + Omega^{-1/2} S (sqrt(a) dW+ - sqrt(b) dW-), with the
                   same current bookkeeping.  Not positivity preserving: negative values are reflected at zero and
                   counted.
fluctuation_stats  occupation-number mean / variance and current mean / variance growth (Var Q_h(t) / 2t) from
                   the recorded paths.
"""
import math
import torch


def _rates_fn(model, drive, B):
    ctx = model.context(drive)
    if next(iter(v for v in ctx.values() if torch.is_tensor(v))).shape[0] == 1 and B > 1:
        ctx = ctx.expand(B)
    return lambda c: model.rates(c, ctx, zero_safe=True)


@torch.no_grad()
def gillespie(rates, hg, n0, omega, t_end, dt_rec, generator=None, max_events=10 ** 8):
    """rates: callable c (B, M) -> (a, b); n0 (B, M) integer molecule numbers; returns dict with
    times (K,), n (B, K, M) float, Q (B, K, H) integrated net currents, events (B,)."""
    dev = n0.device
    B, M, H = n0.shape[0], hg.M, hg.H
    dtype = torch.float64
    t_s = hg.on(dev, dtype)
    # signed state change of every reaction channel r in [0, 2H): forward h, then backward h
    Sd = torch.as_tensor(hg.S, dtype=dtype, device=dev)                               # (M, H)
    stoich = torch.cat([Sd.T, -Sd.T], 0)                                             # (2H, M)
    K = int(round(t_end / dt_rec)) + 1
    times = torch.arange(K, dtype=dtype, device=dev) * dt_rec
    n = n0.to(dtype).clone()
    out = torch.zeros(B, K, M, dtype=dtype, device=dev)
    Qout = torch.zeros(B, K, H, dtype=dtype, device=dev)
    Q = torch.zeros(B, H, dtype=dtype, device=dev)
    t = torch.zeros(B, dtype=dtype, device=dev)
    nxt = torch.zeros(B, dtype=torch.long, device=dev)                               # next recording index
    nev = torch.zeros(B, dtype=torch.long, device=dev)
    g = generator
    ar = torch.arange(B, device=dev)
    for _ in range(max_events):
        a, b = rates(n / omega)
        pr = omega * torch.cat([a, b], 1).clamp_min(0)                                # (B, 2H)
        a0 = pr.sum(-1)
        dead = a0 <= 0
        u = torch.rand(B, 2, dtype=dtype, device=dev, generator=g)
        tau = torch.where(dead, torch.full_like(a0, float('inf')), -torch.log(u[:, 0]) / torch.where(dead, torch.ones_like(a0), a0))
        tn = t + tau
        # record the current state at every sampling time in [t, tn)
        rec = (times[None, :] >= t[:, None]) & (times[None, :] < tn[:, None]) & (torch.arange(K, device=dev)[None, :] >= nxt[:, None])
        out = torch.where(rec[..., None], n[:, None, :], out)
        Qout = torch.where(rec[..., None], Q[:, None, :], Qout)
        nxt = nxt + rec.sum(-1)
        live = (tn <= times[-1]) & ~dead
        if not bool(live.any()):
            break
        cs = pr.cumsum(-1)
        r = torch.searchsorted(cs, (u[:, 1] * a0)[:, None]).squeeze(-1).clamp(max=2 * H - 1)
        n = n + live[:, None].to(dtype) * stoich[r]
        sgn = torch.where(r < H, 1.0, -1.0).to(dtype) * live.to(dtype)
        Q = Q.index_put((ar, r % H), Q[ar, r % H] + sgn)
        t = torch.where(live, tn, t)
        nev = nev + live.long()
    return dict(times=times, n=out, Q=Qout, events=nev, omega=omega)


@torch.no_grad()
def chemical_langevin(rates, hg, c0, omega, t_end, dt_rec, dt=None, generator=None):
    """Euler-Maruyama for the chemical Langevin equation.  Returns dict with times (K,), n (B, K, M) = Omega c,
    Q (B, K, H) integrated net currents in molecule numbers, n_reflect (number of reflected negative entries)."""
    dev, dtype = c0.device, torch.float64
    c = c0.to(dtype).clone()
    B = c.shape[0]
    dt = dt_rec / 20 if dt is None else dt
    nsub = max(1, int(round(dt_rec / dt))); dt = dt_rec / nsub
    K = int(round(t_end / dt_rec)) + 1
    out = torch.zeros(B, K, hg.M, dtype=dtype, device=dev); out[:, 0] = omega * c
    Qout = torch.zeros(B, K, hg.H, dtype=dtype, device=dev)
    Q = torch.zeros(B, hg.H, dtype=dtype, device=dev)
    nref = 0
    sq = math.sqrt(dt / omega)
    for k in range(1, K):
        for _ in range(nsub):
            a, b = rates(c)
            a, b = a.clamp_min(0), b.clamp_min(0)
            w = torch.randn(2, B, hg.H, dtype=dtype, device=dev, generator=generator)
            dq = (a - b) * dt + sq * (a.sqrt() * w[0] - b.sqrt() * w[1])
            c = c + hg.apply_S(dq)
            Q = Q + omega * dq
            neg = c < 0
            nref += int(neg.sum())
            c = c.abs()
        out[:, k] = omega * c
        Qout[:, k] = Q
    return dict(times=torch.arange(K, dtype=dtype, device=dev) * dt_rec, n=out, Q=Qout, n_reflect=nref, omega=omega)


def fluctuation_stats(res, t_burn=0.0):
    """occupation numbers after t_burn: mean and variance over trajectories and recorded times (B, K, M) -> (M,);
    currents: mean rate <Q_h>/t and the growth rate of the variance Var(Q_h(t) - Q_h(t_burn)) / (2 (t - t_burn))
    over trajectories, evaluated at the last recorded time."""
    times, n, Q = res['times'], res['n'], res['Q']
    sel = times >= t_burn
    nn_ = n[:, sel]
    occ_mean = nn_.mean((0, 1)); occ_var = nn_.var((0, 1))
    i0 = int(torch.nonzero(sel)[0])
    T = float(times[-1] - times[i0])
    dQ = Q[:, -1] - Q[:, i0]
    return dict(occ_mean=occ_mean, occ_var=occ_var, fano=occ_var / occ_mean.clamp_min(1e-12),
                current_mean=dQ.mean(0) / T, current_diffusion=dQ.var(0) / (2 * T) if dQ.shape[0] > 1 else None)


def sample_model(model, drive, c0, omega, t_end, dt_rec, method='gillespie', **kw):
    """convenience wrapper: sample the master equation of a model with its learned rates.  A drive with batch 1 is
    shared by all trajectories in c0."""
    rates = _rates_fn(model, drive, c0.shape[0])
    if method == 'gillespie':
        return gillespie(rates, model.hg, torch.round(omega * c0), omega, t_end, dt_rec, **kw)
    return chemical_langevin(rates, model.hg, c0, omega, t_end, dt_rec, **kw)
