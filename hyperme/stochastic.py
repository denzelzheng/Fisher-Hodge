"""Stochastic layer: sample the chemical master equation defined by a model's (learned) rates.

System size Omega: molecule numbers n = Omega c.  Hyperedge h fires forward with propensity Omega a_h(n / Omega) and
backward with propensity Omega b_h(n / Omega) (Kurtz scaling; for mass action with distinct reactants this is the
usual k n_i n_j / Omega).  The rates come from any callable rates(c) -> (a, b), e.g. a trained HyperME with its
drive context, so the samplers use the learned free energy, kinetics and drive.

gillespie          batched direct method (independent trajectories in a batch, one event per trajectory per
                   iteration, works on CPU or GPU).  Records n at the sampling times and, per hyperedge, the
                   forward and backward firing counts, the medium entropy of the jumps (under the sampling rates and,
                   optionally, under reference rates on the same path) and the integrated propensities.
tau_leap           Poisson tau-leaping with the same records, for large Omega.
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


class _Recorder:
    """cumulative per-hyperedge quantities, written at the sampling times.
    Nf, Nb: forward / backward firings; S: medium entropy of the jumps on each hyperedge under the sampling rates;
    Sref: the same under reference rates (e.g. the true ones) evaluated on the sampled path; Ia, Ib: time integrals of
    the propensities Omega a, Omega b after t_burn."""
    def __init__(s, B, K, M, H, dev, ref):
        z = lambda *sh: torch.zeros(*sh, dtype=torch.float64, device=dev)
        s.cur = dict(Nf=z(B, H), Nb=z(B, H), S=z(B, H))
        if ref:
            s.cur['Sref'] = z(B, H)
        s.out = {k: z(B, K, H) for k in s.cur}
        s.out['n'] = z(B, K, M)
        s.Ia, s.Ib = z(B, H), z(B, H)

    def write(s, rec, n):
        if bool(rec.any()):
            bi, ki = torch.nonzero(rec, as_tuple=True)
            s.out['n'][bi, ki] = n[bi]
            for k, v in s.cur.items():
                s.out[k][bi, ki] = v[bi]


def _safe_log(x):
    return torch.log(x.clamp_min(1e-300))


@torch.no_grad()
def gillespie(rates, hg, n0, omega, t_end, dt_rec, generator=None, max_events=10 ** 9, ref_rates=None, t_burn=0.0):
    """Exact batched direct method.  rates: callable c (B, M) -> (a, b) (zero-safe: a channel with an absent reactant
    has rate 0); n0 (B, M) molecule numbers; optional ref_rates for the reference entropy Sref.
    The medium entropy of a jump x -> x' along channel q is log w_q(x) - log w_{reverse q}(x').
    Returns dict: times (K,), n (B, K, M), Nf, Nb, S [, Sref] (B, K, H) cumulative per hyperedge, Q = Nf - Nb,
    Ia, Ib (B, H) integrated propensities over [t_burn, t_end], events (B,)."""
    dev = n0.device
    B, M, H = n0.shape[0], hg.M, hg.H
    dtype = torch.float64
    Sd = torch.as_tensor(hg.S, dtype=dtype, device=dev)                               # (M, H)
    stoich = torch.cat([Sd.T, -Sd.T], 0)                                             # (2H, M): forward h, backward h
    K = int(round(t_end / dt_rec)) + 1
    times = torch.arange(K, dtype=dtype, device=dev) * dt_rec
    n = n0.to(dtype).clone()
    R = _Recorder(B, K, M, H, dev, ref_rates is not None)
    t = torch.zeros(B, dtype=dtype, device=dev)
    nxt = torch.zeros(B, dtype=torch.long, device=dev)                               # next recording index
    nev = torch.zeros(B, dtype=torch.long, device=dev)
    ar = torch.arange(B, device=dev)
    kidx = torch.arange(K, device=dev)
    pend = None                                       # (rows, hyperedge, forward?, log w(x), log w_ref(x)) of the last jumps
    for _ in range(max_events):
        a, b = rates(n / omega)
        if ref_rates is not None:
            ar_, br_ = ref_rates(n / omega)
        if pend is not None:                          # finish the entropy of the previous jumps with the reverse rate at x'
            rows, h, fw, lw, lwr = pend
            rev = torch.where(fw, b[rows, h], a[rows, h])
            R.cur['S'][rows, h] += lw - _safe_log(omega * rev)
            if ref_rates is not None:
                rev_r = torch.where(fw, br_[rows, h], ar_[rows, h])
                R.cur['Sref'][rows, h] += lwr - _safe_log(omega * rev_r)
        pr = omega * torch.cat([a, b], 1).clamp_min(0)                                # (B, 2H)
        a0 = pr.sum(-1)
        dead = a0 <= 0
        u = torch.rand(B, 2, dtype=dtype, device=dev, generator=generator)
        tau = torch.where(dead, torch.full_like(a0, float('inf')), -torch.log(u[:, 0]) / torch.where(dead, torch.ones_like(a0), a0))
        tn = t + tau
        lo, hi = t.clamp(min=t_burn), torch.minimum(tn, times[-1].expand_as(tn))     # integrate propensities after burn-in
        w = (hi - lo).clamp_min(0)[:, None]
        R.Ia += w * pr[:, :H]; R.Ib += w * pr[:, H:]
        rec = (times[None, :] >= t[:, None]) & (times[None, :] < tn[:, None]) & (kidx[None, :] >= nxt[:, None])
        R.write(rec, n)
        nxt = nxt + rec.sum(-1)
        live = (tn <= times[-1]) & ~dead
        if not bool(live.any()):
            break
        cs = pr.cumsum(-1)
        r = torch.searchsorted(cs, (u[:, 1] * a0)[:, None]).squeeze(-1).clamp(max=2 * H - 1)
        n = n + live[:, None].to(dtype) * stoich[r]
        rows = ar[live]; rl = r[live]; h = rl % H; fw = rl < H
        R.cur['Nf'][rows[fw], h[fw]] += 1
        R.cur['Nb'][rows[~fw], h[~fw]] += 1
        lw = _safe_log(pr[rows, rl])
        lwr = None
        if ref_rates is not None:
            prr = omega * torch.cat([ar_, br_], 1)
            lwr = _safe_log(prr[rows, rl])
        pend = (rows, h, fw, lw, lwr)
        t = torch.where(live, tn, t)
        nev = nev + live.long()
    out = dict(R.out, times=times, events=nev, omega=omega, Ia=R.Ia, Ib=R.Ib)
    out['Q'] = out['Nf'] - out['Nb']
    return out


@torch.no_grad()
def tau_leap(rates, hg, n0, omega, t_end, dt_rec, tau, generator=None, ref_rates=None, t_burn=0.0):
    """Poisson tau-leaping with the same outputs as gillespie (entropy evaluated with the rates at the start of each
    leap).  A leap that would make a molecule number negative is redone for that trajectory with half the step; the
    number of such retries is returned as 'retries'."""
    dev = n0.device
    B, M, H = n0.shape[0], hg.M, hg.H
    dtype = torch.float64
    K = int(round(t_end / dt_rec)) + 1
    times = torch.arange(K, dtype=dtype, device=dev) * dt_rec
    nper = max(1, int(round(dt_rec / tau))); tau = dt_rec / nper
    n = n0.to(dtype).clone()
    R = _Recorder(B, K, M, H, dev, ref_rates is not None)
    R.write(torch.zeros(B, K, dtype=torch.bool, device=dev).index_fill(1, torch.tensor([0], device=dev), True), n)
    retries = 0
    a, b = (x.clamp_min(0) for x in rates(n / omega))
    ra, rb = ref_rates(n / omega) if ref_rates is not None else (None, None)
    for k in range(1, K):
        after_burn = float(times[k - 1]) >= t_burn - 1e-12
        for _ in range(nper):
            remaining = torch.full((B,), tau, dtype=dtype, device=dev)
            step = torch.full((B,), tau, dtype=dtype, device=dev)
            for _try in range(64):
                active = remaining > 1e-15
                if not bool(active.any()):
                    break
                st = torch.where(active, torch.minimum(step, remaining), torch.zeros_like(step))
                Ff = torch.poisson(omega * a * st[:, None], generator=generator)
                Fb = torch.poisson(omega * b * st[:, None], generator=generator)
                nn_ = n + hg.apply_S(Ff - Fb)
                ok = active & (nn_ >= 0).all(-1)
                okf = ok[:, None].to(dtype)
                n = torch.where(ok[:, None], nn_, n)
                a2, b2 = (x.clamp_min(0) for x in rates(n / omega))
                # medium entropy per firing: log w(x) - log w_reverse(x'), x' the state after the leap
                R.cur['S'] += okf * (Ff * (_safe_log(a) - _safe_log(b2)) - Fb * (_safe_log(b) - _safe_log(a2)))
                if ref_rates is not None:
                    ra2, rb2 = ref_rates(n / omega)
                    R.cur['Sref'] += okf * (Ff * (_safe_log(ra) - _safe_log(rb2)) - Fb * (_safe_log(rb) - _safe_log(ra2)))
                    ra, rb = ra2, rb2
                R.cur['Nf'] += okf * Ff; R.cur['Nb'] += okf * Fb
                if after_burn:
                    R.Ia += okf * st[:, None] * omega * a; R.Ib += okf * st[:, None] * omega * b
                a, b = a2, b2
                remaining = remaining - torch.where(ok, st, torch.zeros_like(st))
                bad = active & ~ok
                retries += int(bad.sum())
                step = torch.where(bad, 0.5 * st, step)
        R.write(torch.zeros(B, K, dtype=torch.bool, device=dev).index_fill(1, torch.tensor([k], device=dev), True), n)
    out = dict(R.out, times=times, omega=omega, Ia=R.Ia, Ib=R.Ib, retries=retries)
    out['Q'] = out['Nf'] - out['Nb']
    return out


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
                current_mean=dQ.mean(0) / T, current_diffusion=dQ.var(0) / (2 * T) if dQ.shape[0] > 1 else None,
                current_var_rate=dQ.var(0) / T if dQ.shape[0] > 1 else None)


def sample_model(model, drive, c0, omega, t_end, dt_rec, method='gillespie', **kw):
    """convenience wrapper: sample the master equation of a model with its learned rates.  A drive with batch 1 is
    shared by all trajectories in c0."""
    rates = _rates_fn(model, drive, c0.shape[0])
    if method == 'gillespie':
        return gillespie(rates, model.hg, torch.round(omega * c0), omega, t_end, dt_rec, **kw)
    return chemical_langevin(rates, model.hg, c0, omega, t_end, dt_rec, **kw)
