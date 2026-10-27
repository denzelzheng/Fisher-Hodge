"""Information-geometry checks of the Thermodynamic Neural Chart (results/info_geometry_checks.md).

The wired model is a Neural Chart: the condition fixes the force f by local detailed balance (the physical
coordinate) and the network gives the symmetric kinetics k (the material coordinate).  Five checks:
  1 path    entropy production = path KL.  C1 phosphorylation cycle at one site, Omega = 200: the log ratio of the
            probability of long Gillespie paths and of their time reversals, per unit time, against
            sum (a - b) log(a / b) and the exact master-equation value, with the true and with the learned rates.
  2 pyth    Pythagorean split of sigma for learned models (C3, S2): the Lambda-weighted Hodge projection
            (excess + housekeeping = sigma, excess = 0 at a steady state) against the Euclidean projection.
  3 proj    the convex sigma_min of results/c3_one_observable.md written edge by edge as the smallest path KL between
            the process and its time reversal over the data-consistent rates, and the bound as the feasible set
            shrinks (densities / + E turnover / + R2 current at one site).
  4 coarse  coarse graining as data processing on S2 (32 -> 16 -> 8): path-KL estimates from single-particle paths
            of a learned generator, observed through the blocks.
  5 extrap  the transfer bound of Theorem 0(iii)(b) on S2-M, an S2 variant whose true generator lies in G(S) (SQRA
            with a mobility that depends on a material field): bound against measured vector-field and 10T errors
            at drive x 1, 2, 4, 8 and temperature x 0.25 ... 4, with Flux-GNN and a UDE closure for comparison.
    python -m hyperme.info_geometry --stage c1_train | c1_path | pyth | proj | coarse | ex_data | ex_eval | plot | report
    python -m hyperme.info_geometry --stage ex_train --model wired|flux|ude
Outputs go to runs/hyperme/infogeo/, figures to assets/ig_*.png.  CPU only."""
import argparse, copy, json, os, time
import numpy as np
import scipy.linalg
import scipy.optimize
import torch
from . import presets, baselines as BL
from .presets import ROOT, default_dtype
from .complex import Hypergraph, GridOps as G
from .model import HyperME, WiredKinetics, Context, logmean
from .linear_fit import fit, assemble, nnls_kappa
from .coarse import projected_generator, grid_block_labels, model_stationary
from . import c3_anchor as CA
from . import c3_observable as C3O

OUT = os.path.join(ROOT, 'runs', 'hyperme', 'infogeo')
ASSETS = os.path.join(ROOT, 'assets')
F64 = torch.float64


def _log(name):
    os.makedirs(OUT, exist_ok=True)
    f = open(os.path.join(OUT, name + '.log'), 'w')

    def log(*a):
        s = ' '.join(str(x) for x in a); print(s, flush=True); f.write(s + '\n'); f.flush()
    return log


# ------------------------------------------------------------------------------------------ 1. C1 at one site
C1_RX = [({'X': 1, 'E': 1, 'T': 1}, {'Xp': 1, 'E': 1, 'D': 1}), ({'Xp': 1}, {'X': 1})]
C1_K = dict(kf=(2.0, 1.0), kb=(2.0, 0.02), D=())   # hyper/gen_crn.py: k1 both ways (dG in ATP / ADP), k2, k2'
C1_DT, OMEGA = 0.05, 200
C1_TESTS = (2.0, 3.5)                               # dG in the training range [1, 3] and outside it


def c1_hg():
    """X + E + ATP <-> Xp + E + ADP, Xp <-> X at one site; ATP (T) and ADP (D) chemostatted, E catalytic"""
    return Hypergraph.reaction_transport(['X', 'Xp', 'E', 'T', 'D'], C1_RX, n_sites=1, chemostats=['T', 'D'])


def c1_chart():
    """the Thermodynamic Neural Chart of C1: wired force (log c + mu0, ATP / ADP chemostats), log k a table"""
    hg = c1_hg()
    with default_dtype(F64):
        m = HyperME.build_wired(hg, k_ref=1.0, T_exponent=0.0, harmonic=False, featurizer=presets.chem_featurizer(hg),
                                step=dict(method='mprk22', nsub=8), learn_mu0=True, learn_chem_mu0=True, kinetics_table=True)
    return m.double()


def c1_drive(dG):
    """log c_ATP = dG / 2, log c_ADP = -dG / 2, so R1 runs at k1 x e^{dG/2} forward and k1 xp e^{-dG/2} backward"""
    dG = torch.as_tensor(dG, dtype=F64).reshape(-1)
    return dict(logab=torch.stack([dG / 2, -dG / 2], -1))


def c1_truth():
    m = c1_chart()
    err = presets.set_true_mass_action(m, logab=c1_drive(2.0)['logab'], **C1_K)
    assert err < 1e-10, err
    return m


@torch.no_grad()
def c1_data(seed=0, ntrain=16):
    truth = c1_truth()
    g = torch.Generator().manual_seed(seed)
    u = lambda lo, hi: lo + (hi - lo) * torch.rand(ntrain, generator=g, dtype=F64)
    tot, fr, e = u(0.6, 1.4), u(0.1, 0.9), u(0.5, 1.5)
    drive = c1_drive(u(1.0, 3.0))
    ctx = truth.prepare(drive)
    c = torch.stack([tot * fr, tot * (1 - fr), e], -1); out = [c]
    for _ in range(presets.T_TRAIN):
        c, _ = truth.advance(c, ctx, C1_DT, nsub=64); out.append(c)
    return dict(train=dict(c=torch.stack(out, 1), drive=drive))


def load_c1_chart():
    m = c1_chart(); m.load_state_dict(torch.load(os.path.join(OUT, 'c1_chart_s0.pt'))); return m.eval()


def stage_c1_train(seed=0, iters=2000, lr=1e-2):
    torch.manual_seed(seed)
    log = _log('c1_chart_s0')
    m, data = c1_chart(), c1_data(seed)
    info = fit(m, data, iters, C1_DT, seed=seed, lr=lr, log=log)
    torch.save(m.state_dict(), os.path.join(OUT, 'c1_chart_s0.pt'))
    truth = c1_truth()
    c = torch.tensor([[0.5, 0.5, 1.0]], dtype=F64)
    info['rate_ratio'] = {}
    for dG in C1_TESTS:
        with torch.no_grad():
            a, b = m.rates(c, c1_drive(dG)); at, bt = truth.rates(c, c1_drive(dG))
        info['rate_ratio'][str(dG)] = dict(a=(a / at)[0].tolist(), b=(b / bt)[0].tolist())
    log('learned / true one-way rates at x = xp = 0.5, e = 1:', json.dumps(info['rate_ratio']))
    json.dump({k: v for k, v in info.items() if k != 'loss_curve'}, open(os.path.join(OUT, 'c1_chart_s0.json'), 'w'), indent=1)


def _c1_rates(model, dG, B):
    ctx = model.prepare(c1_drive(torch.full((B,), float(dG), dtype=F64)))
    return lambda c: model.rates(c, ctx, zero_safe=True)


@torch.no_grad()
def c1_exact(model, dG, omega=OMEGA, ntot=None, ne=None):
    """one site, X + Xp = ntot molecules, E = ne fixed: the master equation is a birth-death chain in n = #Xp.
    Returns log p (stationary), the channel-resolved entropy production of the master equation per unit time
    (sum over jumps of p w_q log(w_q / w_reverse q)) and Omega sigma_det at the deterministic fixed point."""
    ntot = omega if ntot is None else ntot; ne = omega if ne is None else ne
    n = torch.arange(ntot + 1, dtype=F64)
    c = torch.stack([(ntot - n) / omega, n / omega, torch.full_like(n, ne / omega)], -1)
    a, b = _c1_rates(model, dG, len(n))(c)
    w = omega * torch.stack([a[:, 0], b[:, 0], a[:, 1], b[:, 1]], -1)   # R1+ (n -> n+1), R1-, R2+ (n -> n-1), R2-
    up, dn = w[:, 0] + w[:, 3], w[:, 1] + w[:, 2]
    logp = torch.cat([torch.zeros(1, dtype=F64), torch.cumsum(torch.log(up[:-1]) - torch.log(dn[1:]), 0)])
    logp = logp - torch.logsumexp(logp, 0); p = logp.exp()
    s = 0.0
    for q, r in ((0, 1), (3, 2)):              # raising channel at n, its reverse at n + 1
        x, y = p[:-1] * w[:-1, q], p[1:] * w[1:, r]
        s += float(((x - y) * (torch.log(w[:-1, q]) - torch.log(w[1:, r]))).sum())
    rate = _c1_rates(model, dG, 1)
    tot, e = ntot / omega, ne / omega

    def xdot(xp):
        aa, bb = rate(torch.tensor([[tot - xp, xp, e]], dtype=F64))
        J = (aa - bb)[0]
        return float(J[0] - J[1])
    xs = scipy.optimize.brentq(xdot, 1e-12 * tot, tot * (1 - 1e-12), xtol=1e-15)
    aa, bb = rate(torch.tensor([[tot - xs, xs, e]], dtype=F64))
    sdet = float(((aa - bb) * (aa.log() - bb.log())).sum())
    return logp, s, omega * sdet, xs


@torch.no_grad()
def c1_paths(model, dG, B=64, t_end=200.0, dt_rec=1.0, seed=0, omega=OMEGA):
    """path-KL estimate: per trajectory the log ratio of the path probability and that of its time reversal,
    sum over jumps of log w_q(x) / w_{reverse q}(x') plus the boundary term log p(x_0) - log p(x_t), started from the
    stationary law p of the same rates, divided by t."""
    from .stochastic import gillespie
    logp, s_me, s_det, _ = c1_exact(model, dG, omega)
    g = torch.Generator().manual_seed(seed)
    n = torch.multinomial(logp.exp(), B, replacement=True, generator=g).to(F64)
    n0 = torch.stack([omega - n, n, torch.full_like(n, omega)], -1)
    t0 = time.time()
    res = gillespie(_c1_rates(model, dG, B), model.hg, n0, omega, t_end, dt_rec, generator=g)
    nk = res['n'][..., 1].round().long()
    Stot = res['S'].sum(-1) + logp[n.long()][:, None] - logp[nk]
    ts = res['times']
    rows = []
    for k in range(1, len(ts)):
        r = Stot[:, k] / ts[k]
        rows.append(dict(t=float(ts[k]), mean=float(r.mean()), se=float(r.std() / np.sqrt(B))))
    return dict(dG=dG, sigma_det_omega=s_det, sigma_me=s_me, curve=rows, events=float(res['events'].double().mean()),
                time_s=time.time() - t0)


def stage_c1_path():
    truth, chart = c1_truth(), load_c1_chart()
    out = {}
    for name, m in (('truth', truth), ('chart', chart)):
        for dG in C1_TESTS:
            r = c1_paths(m, dG)
            r['sigma_det_omega_true'] = c1_exact(truth, dG)[2]
            out[f'{name}_{dG}'] = r
            last = r['curve'][-1]
            print(name, dG, f"path KL {last['mean']:.3f} +- {last['se']:.3f}  ME {r['sigma_me']:.3f}  "
                  f"Omega sigma_det {r['sigma_det_omega']:.3f} (true {r['sigma_det_omega_true']:.3f})  events {r['events']:.0f}", flush=True)
    om = {}
    for O in (50, 200, 1000, 5000):            # the master-equation value approaches Omega sigma_det as 1 / Omega
        _, s, sd, _ = c1_exact(truth, 2.0, O)
        om[O] = dict(me=s, det=sd)
    out['omega_scan_truth_dG2'] = om
    json.dump(out, open(os.path.join(OUT, 'c1_path.json'), 'w'), indent=1)


# ------------------------------------------------------------------------------------------ 2. Pythagorean split
def hodge_parts(S, f, ell, w, Nl=None):
    """project f onto im S^T in the w-weighted inner product (w = ell: Fisher / Onsager, w = 1: Euclidean) and
    measure both parts and f in the ell energy: (excess, housekeeping, sigma).  Nl: basis of ker S^T (gauge)."""
    L = (S * w) @ S.T
    if Nl is not None and Nl.shape[1]:
        L = L + w.mean() * Nl @ Nl.T
    rhs = S @ (w * f)
    psi = np.linalg.solve(L, rhs)
    psi += np.linalg.solve(L, rhs - L @ psi)
    g = S.T @ psi
    return float(ell @ g ** 2), float(ell @ (f - g) ** 2), float(ell @ f ** 2)


def split_row(S, a, b, Nl):
    a, b = a.double().cpu().numpy(), b.double().cpu().numpy()
    f = np.log(a) - np.log(b)
    ell = logmean(torch.as_tensor(a), torch.as_tensor(b)).numpy()
    out = {}
    for name, w in (('fisher', ell), ('euclid', np.ones_like(ell))):
        ex, hk, sig = hodge_parts(S, f, ell, w, Nl)
        out[name] = dict(ex=ex / sig, hk=hk / sig, res=abs(sig - ex - hk) / sig)
    out['sigma'] = float((a - b) @ f)
    out['cv_ell'] = float(ell.std() / ell.mean())
    out['sdot_rel'] = None
    return out


@torch.no_grad()
def pyth_c3(log, nsteps=2000, keep=(0, 5, 25, 100)):
    S = CA.hg_c3().S; Nl = scipy.linalg.null_space(S.T)
    data = CA.load(False)
    rows = []
    for run in ('c3_base_wired_s0', 'c3_v1_wired_s0'):
        m = presets.c3_wired(); m.load_state_dict(torch.load(os.path.join(C3O.OUT, run + '.pt'))); m.eval()
        for st in ('test_id', 'test_ood'):
            sd = data[st]
            ctx = m.prepare(sd['drive']); c = sd['c'][:, 0].clone()
            snap = {}
            for t in range(nsteps + 1):
                if t in keep:
                    snap[t] = c.clone()
                if t == nsteps - 50:
                    c50 = c.clone()
                if t < nsteps:
                    c, _ = m.advance(c, ctx, CA.DT)
            snap['end'] = c
            drift = ((c - c50).norm(dim=-1) / c.norm(dim=-1)).numpy()
            for key, cs in snap.items():
                a, b = m.rates(cs, ctx)
                cdot = m.hg.apply_S(a - b)
                for i in range(cs.shape[0]):
                    r = split_row(S, a[i], b[i], Nl)
                    r.update(run=run, set=st, i=i, t=key, steady=bool(key == 'end' and drift[i] < 1e-9),
                             drift=float(drift[i]) if key == 'end' else None,
                             sdot_rel=float(cdot[i].norm() / (a[i] + b[i]).norm()))
                    rows.append(r)
            log(run, st, 'steady', int((drift < 1e-9).sum()), 'of', len(drift))
    return rows


@torch.no_grad()
def load_s2_chart():
    m = presets.s2_wired().double()
    m.load_state_dict(torch.load(os.path.join(ROOT, 'runs', 'hyperme', 'ws5', 's2_wired_n16_s0.pt'), map_location='cpu'))
    return m.eval()


def _d64(drive):
    return {k: v.double() for k, v in drive.items()}


@torch.no_grad()
def pyth_s2(log, times=(0.0, 0.05, 0.5, 2.0)):
    m = load_s2_chart()
    S = m.hg.S; Nl = scipy.linalg.null_space(S.T)
    data = presets.load_s2()
    rows = []
    for st in ('test_id', 'test_oodpot'):
        sd = data[st]; drive = _d64(sd['drive']); c0 = sd['c'][:, 0].double()
        pi, Qs = model_stationary(m, drive, c0)
        states = {'end': pi}
        for t in times:
            E = torch.as_tensor(np.stack([scipy.linalg.expm(Q.T * t) for Q in Qs])) if t > 0 else None
            states[t] = c0 if t == 0 else (E @ c0[..., None]).squeeze(-1)
        ctx = m.prepare(drive)
        for key, cs in states.items():
            a, b = m.rates(cs, ctx)
            cdot = m.hg.apply_S(a - b)
            for i in range(cs.shape[0]):
                r = split_row(S, a[i], b[i], Nl)
                r.update(run='s2_wired_n16_s0', set=st, i=i, t=key, steady=key == 'end',
                         sdot_rel=float(cdot[i].norm() / (a[i] + b[i]).norm()))
                rows.append(r)
        log('s2', st, 'done')
    return rows


def stage_pyth():
    log = _log('pyth')
    rows = pyth_c3(log) + pyth_s2(log)
    json.dump(rows, open(os.path.join(OUT, 'pyth.json'), 'w'))


# ------------------------------------------------------------------------------------------ 3. I-projection bound
def c3_blocks():
    """the data blocks of C3: densities (the 48 driven training trajectories), the E turnover (sum of J_R4,
    results/c3_one_observable.md V1) and the R2 current at site 0 (Y consumed by the autocatalytic step at one
    site), each observable as (obs (B, T + 1, 1), obs_w (1, H)) from the true rates."""
    base = CA.load(False)
    v1 = C3O.load('v1')
    hg = CA.hg_c3()
    j = int(np.where(hg.meta['typ'] == 1)[0][0])
    w2 = torch.zeros(1, hg.H, dtype=F64); w2[0, j] = 1.0
    tr = base['train']
    B, T1, M = tr['c'].shape
    a, b = CA.true_rates(hg, tr['drive']['logab'].repeat_interleave(T1, 0), tr['c'].reshape(B * T1, M))
    r2 = ((a - b) @ w2.T).reshape(B, T1, 1)
    obs = dict(E=(v1['train']['obs'], v1['train']['obs_w']), Y=(r2, w2))
    return base, v1, obs, dict(site=j, r2_min_over_mean=float(r2.abs().min() / r2.abs().mean()))


def _kl_program(Pp, Pm, rp, rm, idx, nk, quads, eps, starts, floor=1e-12):
    """min over the free class factors of the path KL rate between the process (a = kappa_rp Phi+, b = kappa_rm Phi-
    on the training states) and its time reversal, sum_h D(a_h || b_h) + D(b_h || a_h) with D(x || y) =
    x log(x / y) - x + y, the relative entropy of Poisson intensities; subject to r_q(kappa) <= eps_q for every data
    block q = (G, h, yy) (normalised derivative residual).  Objective and gradient by autograd, SLSQP from several
    starts, the feasible result with the smallest objective."""
    rpt, rmt, it = torch.as_tensor(rp), torch.as_tensor(rm), torch.as_tensor(idx)
    D = lambda x, y: x * (torch.log(x) - torch.log(y)) - x + y

    def full(u):
        k = np.ones(nk); k[idx] = u
        return k
    sref = [1.0]

    def kl(u):
        uu = torch.tensor(u, dtype=F64, requires_grad=True)
        k = torch.ones(nk, dtype=F64).index_put((it,), uu)
        a, b = k[rpt] * Pp, k[rmt] * Pm
        v = (D(a, b) + D(b, a)).sum() / Pp.shape[0] / sref[0]
        v.backward()
        return float(v), uu.grad.numpy()
    sref[0] = kl(starts[0])[0]
    res = lambda q, k: (q[2] - 2 * k @ q[1] + k @ q[0] @ k) / q[2]
    cons = [dict(type='ineq', fun=lambda u, q=q, e=e: (e - res(q, full(u))) / e,
                 jac=lambda u, q=q, e=e: -((2 * (q[0] @ full(u)) - 2 * q[1]) / q[2])[idx] / e) for q, e in zip(quads, eps)]
    ok = lambda k: all(res(q, k) <= e * (1 + 1e-6) for q, e in zip(quads, eps))
    best = None
    for u0 in starts:
        r = scipy.optimize.minimize(kl, np.clip(u0, floor, 1e4), jac=True, method='SLSQP', bounds=[(floor, 1e4)] * len(idx),
                                    constraints=cons, options=dict(maxiter=3000, ftol=1e-14))
        if ok(full(r.x)) and (best is None or r.fun < best.fun):
            best = r
    best = best or r
    k = full(best.x)
    return k, dict(kl_train_min=float(best.fun * sref[0]), feasible=bool(ok(k)), converged=bool(best.success))


def path_kl_min(model, data, tol, extra_start=None):
    """the program of c3_observable.sigma_min_convex (one combined residual within (1 + tol) of its best) written
    edge by edge as a path KL (_kl_program)."""
    tr = data['train']
    A_ = assemble(model, tr, CA.DT)
    Gm, h, yy, rp, rm, fixed = A_['G'], A_['h'], A_['yy'], A_['rp'], A_['rm'], A_['fixed']
    kn, free = nnls_kappa(Gm, h, fixed)
    q = (Gm, h, yy)
    rmin = (yy - 2 * kn @ h + kn @ Gm @ kn) / yy
    Pp, Pm = CA._phi(model, tr, range(tr['c'].shape[1]))
    idx = np.where(free)[0]
    k0 = np.maximum(kn, 1e-8)
    starts = [k0[idx], np.ones(len(idx)), np.sqrt(k0[idx])] + ([extra_start[idx]] if extra_start is not None else [])
    k, info = _kl_program(Pp.detach(), Pm.detach(), rp, rm, idx, len(h), [q], [(1 + tol) * rmin], starts)
    info['test'] = lower_over_true(model, data, k, rp, rm)
    return k, info


@torch.no_grad()
def lower_over_true(model, data, k, rp, rm):
    out = {}
    for st in ('test_id', 'test_ood'):
        sd = data[st]
        T1, B = sd['c'].shape[1], sd['c'].shape[0]
        Qp, Qm = CA._phi(model, sd, range(T1))
        st_true = torch.as_tensor(sd['sigma']).double().T.reshape(-1)
        kk = torch.as_tensor(k)
        s = CA._sigma(kk[torch.as_tensor(rp)], kk[torch.as_tensor(rm)], Qp, Qm)
        out[st] = float(np.median((s.reshape(T1, B).mean(0) / st_true.reshape(T1, B).mean(0)).numpy()))
        out[st + '_frac_true_above'] = float((st_true >= s * (1 - 1e-9)).double().mean())
    return out


def block_quads(model, tr, obs):
    """normal equations (G, h, yy) of each data block on the class factors of model; an observable block is the
    difference of assemble with and without it (assemble adds its rows to those of the densities)."""
    Ad = assemble(model, tr, CA.DT)
    out = dict(d=(Ad['G'], Ad['h'], Ad['yy']))
    for key, (o, w) in obs.items():
        A = assemble(model, dict(tr, obs=o, obs_w=w), CA.DT)
        out[key] = (A['G'] - Ad['G'], A['h'] - Ad['h'], A['yy'] - Ad['yy'])
    return out, Ad


def stage_proj(check_tols=(0.1, 1.0), tols=(0.05, 0.1, 0.3, 1.0)):
    log = _log('proj')
    out = {}
    # (a) the existing bound (one combined residual, results/c3_one_observable.md) in both forms
    m = presets.c3_wired(); m.load_state_dict(torch.load(os.path.join(C3O.OUT, 'c3_base_wired_s0.pt'))); m.eval()
    base, v1, obs, meta = c3_blocks()
    out['meta'] = meta
    out['check'] = {}
    for name, data in (('dens', base), ('turnover', v1)):
        rows, prev = [], None
        for tol in check_tols:
            t0 = time.time()
            r = C3O.sigma_range(m, data, tol, extra_start=prev)
            prev = np.asarray(r['kappa_min'])
            k, info = path_kl_min(m, data, tol, extra_start=prev)
            rows.append(dict(tol=tol, sigma_form=dict(test_id=r['test_id']['lower_over_true'], test_ood=r['test_ood']['lower_over_true'],
                                                      train=r['sigma_train_min']),
                             kl_form=dict(test_id=info['test']['test_id'], test_ood=info['test']['test_ood'], train=info['kl_train_min'],
                                          feasible=info['feasible'], converged=info['converged']), n_switchable=r['n_switchable']))
            log('check', name, tol, json.dumps(rows[-1]), f'{time.time() - t0:.0f}s')
        out['check'][name] = rows
    # (b) nested feasible sets: one residual constraint per data block, so that adding a block can only shrink the set.
    # Template: the true parameters, so kappa = 1 is the truth.  eps_q = (1 + tol) r_q(kappa_hat), kappa_hat the best
    # fit to all blocks (sum of the normalised residuals), the same for every set.
    tm = presets.c3_true_wired(presets.c3_wired()).eval()
    quads, Ad = block_quads(tm, base['train'], obs)
    rp, rm, fixed = Ad['rp'], Ad['rm'], Ad['fixed']
    res = lambda q, k: float((q[2] - 2 * k @ q[1] + k @ q[0] @ k) / q[2])
    Gt = sum(q[0] / q[2] for q in quads.values()); ht = sum(q[1] / q[2] for q in quads.values())
    khat, free = nnls_kappa(Gt, ht, fixed)
    idx = np.where(free)[0]
    ones = np.ones(len(khat))
    out['blocks'] = {b: dict(r_hat=res(q, khat), r_true=res(q, ones), true_over_hat=res(q, ones) / res(q, khat)) for b, q in quads.items()}
    log('blocks', json.dumps(out['blocks']))
    Pp, Pm = CA._phi(tm, base['train'], range(base['train']['c'].shape[1]))
    out['nested'] = {}
    for name, keys in (('dens', ['d']), ('turnover', ['d', 'E']), ('turnover_y', ['d', 'E', 'Y'])):
        rows, prev = [], None
        for tol in sorted(tols):
            t0 = time.time()
            qs = [quads[b] for b in keys]; eps = [(1 + tol) * res(quads[b], khat) for b in keys]
            starts = [np.maximum(khat[idx], 1e-8), ones[idx]] + ([prev[idx]] if prev is not None else [])
            k, info = _kl_program(Pp.detach(), Pm.detach(), rp, rm, idx, len(khat), qs, eps, starts)
            prev = k
            row = dict(tol=tol, truth_inside=bool(all(res(quads[b], ones) <= e for b, e in zip(keys, eps))),
                       **lower_over_true(tm, base, k, rp, rm), train=info['kl_train_min'], feasible=info['feasible'], converged=info['converged'])
            rows.append(row)
            log('nested', name, tol, json.dumps(row), f'{time.time() - t0:.0f}s')
        out['nested'][name] = rows
    json.dump(out, open(os.path.join(OUT, 'proj.json'), 'w'), indent=1)


# ------------------------------------------------------------------------------------------ 4. coarse graining
def walk(Q, pi, labels_list, nwalk=4000, njump=5000, seed=0):
    """single-particle paths of the row generator Q, started from pi.  Path-KL estimates per unit time: fine, the
    log Q_ij / Q_ji of every jump plus the boundary term; for every partition the same with the projected generator
    Qbar on the block changes of the observed path (plus its boundary term), and the plug-in estimate
    sum (N_ab - N_ba) log(N_ab / N_ba) / t from the block-change counts."""
    rng = np.random.default_rng(seed)
    M = len(Q)
    off = Q.copy(); np.fill_diagonal(off, 0.0)
    nb = np.stack([np.nonzero(off[i] > 0)[0] for i in range(M)])
    q = off[np.arange(M)[:, None], nb]
    esc = q.sum(1); cum = np.cumsum(q / esc[:, None], 1)
    lr = np.log(q) - np.log(off[nb, np.arange(M)[:, None]])
    lev = []
    for lab in labels_list:
        pg = projected_generator(Q, pi, lab)
        Qb = pg['Qbar'].copy(); np.fill_diagonal(Qb, 0.0)
        with np.errstate(divide='ignore', invalid='ignore'):
            lrb = np.where((Qb > 0) & (Qb.T > 0), np.log(Qb) - np.log(Qb.T), 0.0)
        lev.append(dict(lab=np.asarray(lab), lrb=lrb, pib=pg['pibar'], sig=pg['sigmabar'], N=np.zeros_like(Qb)))
    x = rng.choice(M, nwalk, p=pi / pi.sum()); x0 = x.copy()
    Sf, T = np.zeros(nwalk), np.zeros(nwalk)
    Sc = [np.zeros(nwalk) for _ in lev]
    for _ in range(njump):
        T += rng.exponential(1.0 / esc[x])
        kk = np.minimum((cum[x] < rng.random(nwalk)[:, None]).sum(1), nb.shape[1] - 1)
        y = nb[x, kk]
        Sf += lr[x, kk]
        for L, s in zip(lev, Sc):
            a, b = L['lab'][x], L['lab'][y]
            ch = a != b
            s += np.where(ch, L['lrb'][a, b], 0.0)
            np.add.at(L['N'], (a[ch], b[ch]), 1.0)
        x = y

    def est(S):
        sig = S.sum() / T.sum()
        return float(sig), float(np.std(S - sig * T) / T.mean() / np.sqrt(nwalk))
    out = dict(fine=dict(zip(('path', 'se'), est(Sf + np.log(pi[x0]) - np.log(pi[x])))), jumps=nwalk * njump)
    out['fine']['exact'] = float(projected_generator(Q, pi, labels_list[0])['sigma'])
    for L, s in zip(lev, Sc):
        lab = L['lab']
        p, se = est(s + np.log(L['pib'][lab[x0]]) - np.log(L['pib'][lab[x]]))
        N = L['N']; ok = (N > 0) & (N.T > 0)
        plug = 0.5 * float(np.where(ok, (N - N.T) * (np.log(np.where(ok, N, 1)) - np.log(np.where(ok, N.T, 1))), 0).sum()) / T.sum()
        out[int(lab.max()) + 1] = dict(path=p, se=se, plugin=plug, exact=float(L['sig']))
    return out


@torch.no_grad()
def stage_coarse():
    log = _log('coarse')
    labs = [grid_block_labels(32, 2), grid_block_labels(32, 4)]
    tc = presets.load_legacy_tcme(presets.s2_tcme(), torch.load(os.path.join(ROOT, 'runs', 'tcme', 'S2_tcme_s0.pt'), map_location='cpu')).double().eval()
    models = dict(chart=load_s2_chart(), tcme=tc)
    data = presets.load_s2()
    out = []
    for mname, m in models.items():
        for st in ('test_id', 'test_oodpot'):
            sd = data[st]; drive = _d64(sd['drive'])
            pi_c, Qs = model_stationary(m, drive, sd['c'][:, 0].double())
            for i in range(len(Qs)):
                pi = pi_c[i].numpy(); pi = pi / pi.sum()
                r = walk(Qs[i], pi, labs, seed=i)
                r.update(model=mname, set=st, i=i)
                out.append(r)
                log(mname, st, i, f"fine {r['fine']['path']:.4f}+-{r['fine']['se']:.4f} (exact {r['fine']['exact']:.4f}) "
                    f"16: {r[256]['path']:.4f}+-{r[256]['se']:.4f} ({r[256]['exact']:.4f}) 8: {r[64]['path']:.4f}+-{r[64]['se']:.4f} ({r[64]['exact']:.4f})")
    json.dump(out, open(os.path.join(OUT, 'coarse.json'), 'w'), indent=1)


# ------------------------------------------------------------------------------------------ 5. transfer bound, S2-M
EX_N, EX_D, EX_DT = 32, 0.02, presets.DT
EX_DRIVES = (1.0, 2.0, 4.0, 8.0)
EX_TEMPS = (0.25, 0.5, 1.0, 2.0, 4.0)
EX_FRAMES = (0, 20, 200)


def smooth_field(rng, n, K=3):
    a, b = np.meshgrid(np.arange(n), np.arange(n), indexing='ij')
    f = np.zeros((n, n))
    for kx in range(-K, K + 1):
        for ky in range(0, K + 1):
            if (kx, ky) == (0, 0) or kx * kx + ky * ky > K * K or (ky == 0 and kx < 0):
                continue
            ph = 2 * np.pi * (kx * b + ky * a) / n
            f += rng.normal() * np.cos(ph) + rng.normal() * np.sin(ph)
    f -= f.mean()
    return f / f.std()


def ex_featurizer(hg):
    """S2-M drive = dict(U, dpsi, M (material field), T): wired features, with the edge orientation and the edge mean of
    the material field as the kinetic features (the material descriptor does not depend on the condition)."""
    def feat(F):
        U, P, Mf = F['U'], F['dpsi'], F['M']
        B = U.shape[0]
        ish = torch.zeros_like(P); ish[:, 0] = 1
        sym = torch.stack([ish, 1 - ish, G.mean_e(Mf)], -1).reshape(B, -1, 3)
        return dict(U=U.reshape(B, -1), T=F['T'].reshape(B), line=G.flat_e(P), chem_logc=None, chem_mu0=None, edge_sym=sym,
                    edge_ori=torch.zeros_like(sym[..., 0]), node={}, raw=F)
    feat.n_edge_sym = 3
    return feat


class TrueMobility(WiredKinetics):
    """the material law of S2-M: k = (D / h^2) T m, log m = 0.2 (2 ish - 1) + 0.5 tanh(Mbar) - 0.25 Mbar^2"""
    def log_k(s, feats):
        x = feats['edge_sym']
        return 0.2 * (2 * x[..., 0] - 1) + 0.5 * torch.tanh(x[..., 2]) - 0.25 * x[..., 2] ** 2


def ex_model(truth=False):
    """the Thermodynamic Neural Chart of S2-M: mu = log rho + U / T, A = line integral / T (s_A fixed at 1, no
    harmonic part: the force is wired, not learned), k = (D / h^2) T exp(core + s_k MLP(orientation, Mbar))"""
    hg = Hypergraph.grid_transport(EX_N)
    with default_dtype(F64):
        m = HyperME.build_wired(hg, hidden=32, k_ref=EX_D * EX_N ** 2, T_exponent=1.0, harmonic=False, featurizer=ex_featurizer(hg),
                                step=dict(dt_max=EX_DT / 8, theta0=0.9, theta1=1.0, max_sub=4000, t_tol=1e-9))
    m.drive.s_A.requires_grad_(False)
    if truth:
        m.kinetics.__class__ = TrueMobility
    return m.double()


def bb_featurizer(hg):
    """black-box inputs: U / T and the material field at the nodes, log T, edge orientation, edge means of M and
    U / T, |dU| / T; the drive line integral / T as the oriented edge input (the best-informed black box of G1)."""
    def feat(F):
        U, P, Mf = F['U'], F['dpsi'], F['M']
        B, V = U.shape[0], U[0].numel()
        T = F['T'].reshape(B, 1, 1)
        Ut = U / T
        ish = torch.zeros_like(P); ish[:, 0] = 1
        sym = torch.stack([ish, 1 - ish, G.mean_e(Mf), G.mean_e(Ut), G.d0(Ut).abs()], -1).reshape(B, -1, 5)
        node = dict(U=Ut.reshape(B, V), M=Mf.reshape(B, V), logT=torch.log(T).reshape(B, 1).expand(B, V))
        return dict(U=Ut.reshape(B, V), node=node, edge_sym=sym, edge_ori=G.flat_e(P / T[..., None]), force=None, raw=F)
    feat.n_edge_sym = 5
    return feat


class UDE(BL._Base):
    """universal differential equation: the known physics with a uniform mobility (the wired model untrained, SQRA,
    edited exactly in T and drive) plus a GNN closure of dc/dt from the black-box inputs (zero at initialisation)."""
    name = 'ude'

    def __init__(s, hg, n_node_in, n_edge_in, featurizer, hidden=32, layers=3, scale=1.0, nsub=8):
        super().__init__(hg, featurizer, 1.0, nsub, scale)
        s.phys = [ex_model()]                    # a list keeps its (frozen) parameters out of the optimiser
        s.trunk = BL.HyperGNNTrunk(hg, n_node_in, n_edge_in, hidden, layers)
        s.dec = BL.MLP(hidden, hidden, 1, last='zero')

    def prepare(s, drive):
        return Context(feats=s.featurizer(drive), phys=s.phys[0].prepare(drive))

    def rhs(s, c, drive):
        ctx = s.context(drive)
        hn, _ = s.trunk(s.node_inputs(c, ctx), s.edge_inputs(ctx))
        return s.phys[0].forward(c, ctx['phys']) + s.scale * s.dec(hn).squeeze(-1)


BB_HIDDEN = 32                  # width of the black-box GNNs (as the kinetic MLP of the chart); 64 is too slow on CPU


def ex_blackbox(kind):
    hg = Hypergraph.grid_transport(EX_N)
    f = bb_featurizer(hg)
    ex = dict(U=torch.zeros(1, EX_N, EX_N, dtype=F64), dpsi=torch.zeros(1, 2, EX_N, EX_N, dtype=F64),
              M=torch.zeros(1, EX_N, EX_N, dtype=F64), T=torch.ones(1, dtype=F64))
    nin, ein = BL._Base.n_node_in(hg, f, ex)
    sc = EX_D * EX_N ** 2
    with default_dtype(F64):
        if kind == 'flux':
            m = BL.FluxGNN(hg, nin, ein, featurizer=f, hidden=BB_HIDDEN, layers=3, scale=sc, step=dict(dt_max=EX_DT / 8))
        else:
            m = UDE(hg, nin, ein, f, hidden=BB_HIDDEN, scale=sc)
    return m.double()


def ex_build(kind):
    return ex_model() if kind == 'wired' else ex_blackbox(kind)


def ex_cases(split):
    s2 = presets.load_s2()
    d = s2['train' if split == 'train' else 'test_id']
    rng = np.random.default_rng(100 if split == 'train' else 200)
    Mf = torch.as_tensor(np.stack([smooth_field(rng, EX_N) for _ in range(16)]))
    return dict(U=d['drive']['U'][:16].double(), dpsi=d['drive']['dpsi'][:16].double(), M=Mf), d['c'][:16, 0].double()


def cond_drive(base, s=1.0, r=1.0):
    return dict(U=base['U'], dpsi=s * base['dpsi'], M=base['M'], T=torch.full((base['U'].shape[0],), float(r), dtype=F64))


def conditions():
    return [('drive', s, 1.0) for s in EX_DRIVES] + [('temp', 1.0, r) for r in EX_TEMPS if r != 1.0]


@torch.no_grad()
def exact_frames(truth, drive, c0, frames):
    """the true linear master equation solved exactly: c(t + dt) = exp(dt L) c(t)"""
    L = truth.generator(torch.ones_like(c0), drive)
    E = torch.linalg.matrix_exp(L * EX_DT)
    c, out = c0, {0: c0}
    for t in range(1, max(frames) + 1):
        c = (E @ c[..., None]).squeeze(-1)
        if t in frames:
            out[t] = c
    return out


def stage_ex_data():
    truth = ex_model(truth=True)
    base, c0 = ex_cases('train')
    fr = exact_frames(truth, cond_drive(base), c0, set(range(presets.T_TRAIN + 1)))
    train = dict(c=torch.stack([fr[t] for t in range(presets.T_TRAIN + 1)], 1), drive=cond_drive(base))
    tb, tc0 = ex_cases('test')
    test = {}
    for kind, s, r in conditions():
        t0 = time.time()
        test[(kind, s, r)] = exact_frames(truth, cond_drive(tb, s, r), tc0, set(EX_FRAMES))
        print(kind, s, r, f'{time.time() - t0:.0f}s', flush=True)
    torch.save(dict(train=train, test_base=tb, test_c0=tc0, test=test), os.path.join(OUT, 'ex_data.pt'))


def stage_ex_train(kind, seed=0, iters=2000, lr=1e-3):
    torch.manual_seed(seed); np.random.seed(seed)
    D = torch.load(os.path.join(OUT, 'ex_data.pt'))
    m = ex_build(kind)
    log = _log(f'ex_{kind}_s{seed}')
    log(f'ex_{kind}: params={sum(p.numel() for p in m.parameters() if p.requires_grad)} iters={iters} lr={lr}')
    info = fit(m, dict(train=D['train']), iters, EX_DT, seed=seed, lr=lr, log=log)
    torch.save(m.state_dict(), os.path.join(OUT, f'ex_{kind}_s{seed}.pt'))
    json.dump({k: v for k, v in info.items()}, open(os.path.join(OUT, f'ex_{kind}_s{seed}.json'), 'w'), indent=1)


def onsager_norm(S, x, ell, Nl):
    """sqrt(x^T (S diag(ell) S^T)^+ x) for x in im S: the smallest sqrt(sum J^2 / ell) over the currents with S J = x"""
    L = (S * ell) @ S.T + ell.mean() * Nl @ Nl.T
    return float(np.sqrt(max(x @ np.linalg.solve(L, x), 0.0)))


def relerr(a, b):
    return ((a - b).norm(dim=-1) / b.norm(dim=-1)).numpy()


@torch.no_grad()
def stage_ex_eval(seed=0):
    log = _log('ex_eval')
    D = torch.load(os.path.join(OUT, 'ex_data.pt'))
    truth = ex_model(truth=True)
    models = {}
    for kind in ('wired', 'flux', 'ude'):
        p = os.path.join(OUT, f'ex_{kind}_s{seed}.pt')
        if os.path.exists(p):
            m = ex_build(kind); m.load_state_dict(torch.load(p)); models[kind] = m.eval()
    S = truth.hg.S; Nl = scipy.linalg.null_space(S.T)
    S_norm = float(np.linalg.norm(S, 2))
    tb, c0 = D['test_base'], D['test_c0']
    # delta_k: the largest relative error of the learned k on the test edges; k does not see the condition, so it is
    # measured once (and checked to be the same at every condition)
    ctx_t, ctx_m = truth.prepare(cond_drive(tb)), models['wired'].prepare(cond_drive(tb))
    dk = float((ctx_m['k'] / ctx_t['k'] - 1).abs().max())
    out = dict(S_norm=S_norm, delta_k=dk, rows=[])
    log(f'||S||_2 = {S_norm:.4f}, delta_k = {dk:.4e}')
    for kind, s, r in conditions():
        drive = cond_drive(tb, s, r)
        fr = D['test'][(kind, s, r)]
        ct = truth.prepare(drive)
        row = dict(kind=kind, s=s, r=r, models={})
        km = models['wired'].prepare(drive)['k']
        row['delta_k_here'] = float((km / ct['k'] - 1).abs().max())
        vf = dict(bound_tight=[], bound_dk=[], ons_meas=[], ons_tight=[], ons_dk=[], cdot_true=[], sigma=[])
        meas = {k: [] for k in models}
        for t in EX_FRAMES:
            c = fr[t]
            a, b = truth.rates(c, ct)
            J = a - b; cd_t = truth.hg.apply_S(J)
            f = torch.log(a) - torch.log(b); ell = logmean(a, b)
            dJ = (km / ct['k'] - 1) * J                 # J_model - J_true at shared forces
            sig = (J * f).sum(-1)
            vf['bound_tight'] += (S_norm * dJ.norm(dim=-1)).tolist()
            vf['bound_dk'] += (S_norm * dk * J.norm(dim=-1)).tolist()
            vf['ons_tight'] += torch.sqrt((dJ ** 2 / ell).sum(-1)).tolist()
            vf['ons_dk'] += (dk * torch.sqrt(sig)).tolist()
            vf['cdot_true'] += cd_t.norm(dim=-1).tolist(); vf['sigma'] += sig.tolist()
            for k, m in models.items():
                cd = m.forward(c, m.prepare(drive)) if k == 'wired' else m.rhs(c, m.prepare(drive))
                e = cd - cd_t
                meas[k] += e.norm(dim=-1).tolist()
                if k == 'wired':
                    vf['ons_meas'] += [onsager_norm(S, e[i].numpy(), ell[i].numpy(), Nl) for i in range(len(e))]
        row['vf'] = vf
        for k, m in models.items():
            if k == 'ude':
                m.nsub = int(8 * max(1.0, r) * max(1.0, s / 2))     # explicit Heun: keep dt times the fastest rate fixed
            ctx = m.prepare(drive)
            c = c0.clone(); err = {}
            t0 = time.time()
            for t in range(1, max(EX_FRAMES) + 1):
                c, _ = m.advance(c, ctx, EX_DT)
                if t in EX_FRAMES:
                    err[t] = relerr(c, fr[t]).tolist()
            row['models'][k] = dict(vf_meas=meas[k], rollout=err, min_c=float(c.min()), time_s=time.time() - t0)
        out['rows'].append(row)
        log(kind, s, r, 'dk here', f"{row['delta_k_here']:.3e}", ' '.join(
            f"{k}: vf {np.median(v['vf_meas']):.3e} 10T {np.median(v['rollout'][200]):.3e}" for k, v in row['models'].items()),
            f"bound tight {np.median(vf['bound_tight']):.3e} dk {np.median(vf['bound_dk']):.3e}")
    json.dump(out, open(os.path.join(OUT, 'ex_eval.json'), 'w'), indent=1)


# ------------------------------------------------------------------------------------------ figures
INK, INK2, GRID = '#0b0b0b', '#52514e', '#e4e3df'
C_BLUE, C_ORANGE, C_AQUA, C_YELLOW = '#2a78d6', '#eb6834', '#1baf7a', '#eda100'


def _style(ax):
    for sp in ('top', 'right'):
        ax.spines[sp].set_visible(False)
    for sp in ('left', 'bottom'):
        ax.spines[sp].set_color(INK2)
    ax.tick_params(colors=INK2, labelsize=8)
    ax.grid(True, color=GRID, lw=0.6); ax.set_axisbelow(True)


def plot_pyth(plt):
    rows = json.load(open(os.path.join(OUT, 'pyth.json')))
    fig, axs = plt.subplots(1, 2, figsize=(9, 3.6))
    floor = 1e-17
    for ax, prob in zip(axs, ('C3', 'S2')):
        rr = [r for r in rows if r['run'].startswith('c3') == (prob == 'C3')]
        groups = [('residual,\nall states', lambda r: True, 'res'), ('excess / σ,\nsteady states', lambda r: r['steady'], 'ex')]
        pos = 0
        ticks, labs = [], []
        for gname, sel, key in groups:
            for j, (meth, col, lab) in enumerate((('fisher', C_BLUE, 'Λ-weighted'), ('euclid', C_ORANGE, 'Euclidean'))):
                v = np.array([max(r[meth][key], floor) for r in rr if sel(r)])
                x = pos + j * 0.8 + np.random.default_rng(0).uniform(-0.22, 0.22, len(v))
                ax.scatter(x, v, s=9, color=col, alpha=0.55, lw=0, label=lab if gname == groups[0][0] else None)
                ax.plot([pos + j * 0.8 - 0.3, pos + j * 0.8 + 0.3], [np.median(v)] * 2, color=INK, lw=1.5)
            ticks.append(pos + 0.4); labs.append(gname)
            pos += 2.2
        ax.set_yscale('log'); ax.set_ylim(floor / 3, 20)
        ax.set_xticks(ticks); ax.set_xticklabels(labs, fontsize=8, color=INK2)
        n_ss = sum(r['steady'] for r in rr)
        ax.set_title(f'{prob}: learned Neural Chart ({len(rr)} states, {n_ss} steady)', fontsize=9, color=INK, loc='left')
        _style(ax)
    axs[0].set_ylabel('fraction of σ (values below 1e-17 drawn at 1e-17)', fontsize=8, color=INK2)
    axs[0].legend(frameon=False, fontsize=8, loc='center left')
    fig.tight_layout(); fig.savefig(os.path.join(ASSETS, 'ig_pythagorean.png'), dpi=160); plt.close(fig)


def plot_proj(plt):
    P = json.load(open(os.path.join(OUT, 'proj.json')))
    fig, ax = plt.subplots(figsize=(5.6, 3.6))
    names = dict(dens='densities only', turnover='+ turnover of reservoir E', turnover_y='+ R2 current at one site')
    for (name, rows), col, ls, lw, ms in zip(P['nested'].items(), (C_ORANGE, C_BLUE, C_AQUA), ('-', '-', '--'), (2, 3.5, 1.8), (6, 8, 4)):
        t = [r['tol'] for r in rows]; v = [r['test_id'] for r in rows]
        ax.plot(t, v, ls, color=col, lw=lw, label=names[name])
        for r in rows:
            ax.plot(r['tol'], r['test_id'], 'o', ms=ms, mew=1.5, color=col, mfc=col if r['truth_inside'] else 'white')
    if not all(r['truth_inside'] for rr in P['nested'].values() for r in rr):
        ax.plot([], [], 'o', color=INK2, mfc='white', mew=1.5, label='true rates outside the set')
    ax.axhline(1.0, color=INK2, lw=1, ls='--'); ax.text(0.052, 1.01, 'true σ', fontsize=8, color=INK2, va='bottom')
    ax.set_xscale('log'); ax.set_xlabel('tol: every data block fits within (1 + tol) × its best-fit residual', fontsize=8, color=INK2)
    ax.set_ylabel('min path KL rate / true σ (C3, test ID)', fontsize=8, color=INK2)
    ax.set_ylim(0, 1.15)
    ax.legend(frameon=False, fontsize=7.5, loc='lower left')
    _style(ax); fig.tight_layout(); fig.savefig(os.path.join(ASSETS, 'ig_projection_bound.png'), dpi=160); plt.close(fig)


def plot_extrap(plt):
    E = json.load(open(os.path.join(OUT, 'ex_eval.json')))
    rows = E['rows']
    sel = dict(drive=sorted([r for r in rows if r['kind'] == 'drive'], key=lambda r: r['s']),
               temp=sorted([r for r in rows if r['kind'] == 'temp' or r['s'] == 1.0 and r['r'] == 1.0], key=lambda r: r['r']))
    fig, axs = plt.subplots(3, 2, figsize=(9, 8.4), sharex='col')
    lab_m = dict(wired='Neural Chart, measured', flux='Flux-GNN, measured', ude='UDE closure, measured')
    col_m = dict(wired=C_BLUE, flux=C_ORANGE, ude=C_YELLOW)
    for j, (kind, rr) in enumerate(sel.items()):
        x = [r['s'] if kind == 'drive' else r['r'] for r in rr]
        med = lambda key, src='vf': [np.median(r[src][key]) for r in rr]
        ax = axs[0, j]
        for k in ('wired', 'flux', 'ude'):
            if k in rr[0]['models']:
                ax.plot(x, [np.median(r['models'][k]['vf_meas']) for r in rr], '-o', color=col_m[k], lw=2, ms=5, mec='white', label=lab_m[k])
        ax.plot(x, med('bound_tight'), '--', color=C_BLUE, lw=1.5, label='bound ‖S‖ ‖Δk ⊙ φ‖')
        ax.plot(x, med('bound_dk'), ':', color=C_BLUE, lw=1.5, label='bound ‖S‖ δk ‖J‖')
        ax.set_yscale('log'); ax.set_ylabel('‖Δ dc/dt‖₂', fontsize=8, color=INK2)
        ax = axs[1, j]
        ax.plot(x, med('ons_meas'), '-o', color=C_BLUE, lw=2, ms=5, mec='white', label='Neural Chart, measured')
        ax.plot(x, med('ons_tight'), '--', color=C_BLUE, lw=1.5, label='bound ‖Δk ⊙ φ‖ in ℓ⁻¹')
        ax.plot(x, med('ons_dk'), ':', color=C_BLUE, lw=1.5, label='bound δk √σ')
        ax.set_yscale('log'); ax.set_ylabel('Onsager norm of Δ dc/dt', fontsize=8, color=INK2)
        ax = axs[2, j]
        for k in ('wired', 'flux', 'ude'):
            if k in rr[0]['models']:
                ax.plot(x, [np.median(r['models'][k]['rollout']['200']) for r in rr], '-o', color=col_m[k], lw=2, ms=5, mec='white', label=lab_m[k])
        ax.set_yscale('log'); ax.set_ylabel('10T relative L2 error', fontsize=8, color=INK2)
        ax.set_xscale('log', base=2)
        ax.set_xlabel('drive × s' if kind == 'drive' else 'temperature × r', fontsize=8, color=INK2)
        ax.set_xticks(x); ax.set_xticklabels([f'{v:g}' for v in x])
        for a in axs[:, j]:
            _style(a); a.axvline(1.0, color=INK2, lw=0.8, ls='-', alpha=0.4)
    axs[0, 0].set_title('drive strength (training at × 1)', fontsize=9, color=INK, loc='left')
    axs[0, 1].set_title('temperature (training at × 1)', fontsize=9, color=INK, loc='left')
    for i in range(3):
        axs[i, 1].legend(frameon=False, fontsize=7, loc='best')
    fig.tight_layout(); fig.savefig(os.path.join(ASSETS, 'ig_extrapolation_bound.png'), dpi=160); plt.close(fig)


def stage_plot():
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    plt.rcParams['font.family'] = 'DejaVu Sans'
    for f, p in (('pyth.json', plot_pyth), ('proj.json', plot_proj), ('ex_eval.json', plot_extrap)):
        if os.path.exists(os.path.join(OUT, f)):
            p(plt)


# ------------------------------------------------------------------------------------------ tables
def stage_report():
    L = []
    p = os.path.join(OUT, 'c1_path.json')
    if os.path.exists(p):
        R = json.load(open(p))
        L += ['## 1', '| rates | dG | path KL at t = 1 / 10 / 50 / 200 (± s.e.) | master equation | Ω σ_det (own rates) | Ω σ_det (true rates) |', '|---|---|---|---|---|---|']
        for k, r in R.items():
            if k.startswith('omega'):
                continue
            cv = {c['t']: c for c in r['curve']}
            L.append(f"| {k.split('_')[0]} | {r['dG']} | " + ' / '.join(f"{cv[t]['mean']:.2f} ± {cv[t]['se']:.2f}" for t in (1.0, 10.0, 50.0, 200.0))
                     + f" | {r['sigma_me']:.3f} | {r['sigma_det_omega']:.3f} | {r['sigma_det_omega_true']:.3f} |")
        L += ['', '| Ω | master equation / Ω σ_det (truth, dG = 2) |', '|---|---|']
        for O, v in R['omega_scan_truth_dG2'].items():
            L.append(f"| {O} | {v['me'] / v['det']:.5f} |")
    p = os.path.join(OUT, 'pyth.json')
    if os.path.exists(p):
        rows = json.load(open(p))
        L += ['', '## 2', '| problem | states | residual Λ: median / max | residual Euclid: median / max | steady states | excess/σ Λ: median / max | excess/σ Euclid: median / max | CV of ℓ |', '|---|---|---|---|---|---|---|---|']
        for prob in ('c3_base', 'c3_v1', 's2'):
            rr = [r for r in rows if r['run'].startswith(prob)]
            ss = [r for r in rr if r['steady']]
            q = lambda v: f'{np.median(v):.1e} / {np.max(v):.1e}' if len(v) else '-'
            L.append(f"| {prob} | {len(rr)} | {q([r['fisher']['res'] for r in rr])} | {q([r['euclid']['res'] for r in rr])} | {len(ss)} | "
                     f"{q([r['fisher']['ex'] for r in ss])} | {q([r['euclid']['ex'] for r in ss])} | {np.median([r['cv_ell'] for r in rr]):.2f} |")
    p = os.path.join(OUT, 'proj.json')
    if os.path.exists(p):
        P = json.load(open(p))
        L += ['', '## 3', f"meta {P['meta']}", '| set (combined residual) | tol | σ form ID ; OOD | path-KL form ID ; OOD | train min σ ; KL | switchable | KL feasible / converged |', '|---|---|---|---|---|---|---|']
        for name, rr in P['check'].items():
            for r in rr:
                s, kl = r['sigma_form'], r['kl_form']
                L.append(f"| {name} | {r['tol']} | {s['test_id']:.6f} ; {s['test_ood']:.6f} | {kl['test_id']:.6f} ; {kl['test_ood']:.6f} | "
                         f"{s['train']:.8g} ; {kl['train']:.8g} | {r['n_switchable']} | {kl['feasible']} / {kl['converged']} |")
        L += ['', f"blocks {json.dumps(P['blocks'])}", '| nested set | tol | truth inside | min path KL / true σ ID ; OOD | true ≥ bound (fraction ID) | feasible / converged |', '|---|---|---|---|---|---|']
        for name, rr in P['nested'].items():
            for r in rr:
                L.append(f"| {name} | {r['tol']} | {r['truth_inside']} | {r['test_id']:.4f} ; {r['test_ood']:.4f} | {r['test_id_frac_true_above']:.3f} | {r['feasible']} / {r['converged']} |")
    p = os.path.join(OUT, 'coarse.json')
    if os.path.exists(p):
        C = json.load(open(p))
        L += ['', '## 4', '| model | set | n | σ_32 path / exact | σ_16 path / exact / plug-in | σ_8 path / exact / plug-in | max abs z (path − exact) | monotone path | monotone exact |', '|---|---|---|---|---|---|---|---|---|']
        for mname in ('chart', 'tcme'):
            for st in ('test_id', 'test_oodpot'):
                rr = [r for r in C if r['model'] == mname and r['set'] == st]
                if not rr:
                    continue
                md = lambda f: np.median([f(r) for r in rr])
                z = max(max(abs(r['fine']['path'] - r['fine']['exact']) / r['fine']['se'], abs(r['256']['path'] - r['256']['exact']) / r['256']['se'],
                            abs(r['64']['path'] - r['64']['exact']) / r['64']['se']) for r in rr)
                mono = sum(r['fine']['path'] > r['256']['path'] > r['64']['path'] for r in rr)
                mono_e = sum(r['fine']['exact'] >= r['256']['exact'] >= r['64']['exact'] for r in rr)
                L.append(f"| {mname} | {st} | {len(rr)} | {md(lambda r: r['fine']['path']):.4f} / {md(lambda r: r['fine']['exact']):.4f} | "
                         f"{md(lambda r: r['256']['path']):.4f} / {md(lambda r: r['256']['exact']):.4f} / {md(lambda r: r['256']['plugin']):.4f} | "
                         f"{md(lambda r: r['64']['path']):.4f} / {md(lambda r: r['64']['exact']):.4f} / {md(lambda r: r['64']['plugin']):.4f} | {z:.2f} | {mono}/{len(rr)} | {mono_e}/{len(rr)} |")
    p = os.path.join(OUT, 'ex_eval.json')
    if os.path.exists(p):
        E = json.load(open(p))
        nb = 16                                           # vf lists: frames EX_FRAMES x 16 test cases, frame 0 first
        L += ['', '## 5', f"‖S‖₂ = {E['S_norm']:.4f}, δ_k = {E['delta_k']:.4e} (max over the test edges of |k/k_true − 1|)",
              '| condition | ‖Δċ‖ chart / bound ‖S‖‖ΔJ‖ / bound ‖S‖δk‖J‖ | max chart / bound | Onsager: chart / bound ‖ΔJ‖_ℓ⁻¹ / bound δk√σ | ‖ΔJ‖/‖J‖ chart | Flux-GNN ‖Δċ‖ | UDE ‖Δċ‖ | ‖Δċ‖/‖ċ‖ at t = 0: chart / Flux / UDE | 10T: chart / Flux / UDE | T: chart / Flux / UDE |',
              '|---|---|---|---|---|---|---|---|---|---|']
        for r in E['rows']:
            v, M_ = r['vf'], r['models']
            md = lambda x: float(np.median(x))
            g = lambda k, f: (f(M_[k]) if k in M_ else float('nan'))
            ct0 = np.array(v['cdot_true'][:nb])
            dJ = E['delta_k'] * np.array(v['bound_tight']) / np.array(v['bound_dk'])
            L.append(f"| {r['kind']} s={r['s']:g} r={r['r']:g} | {g('wired', lambda m: md(m['vf_meas'])):.3g} / {md(v['bound_tight']):.3g} / {md(v['bound_dk']):.3g} | "
                     f"{g('wired', lambda m: float(np.max(np.array(m['vf_meas']) / np.array(v['bound_tight'])))):.2f} / {g('wired', lambda m: float(np.max(np.array(m['vf_meas']) / np.array(v['bound_dk'])))):.3f} | "
                     f"{md(v['ons_meas']):.3g} / {md(v['ons_tight']):.3g} / {md(v['ons_dk']):.3g} | {md(dJ):.4f} | "
                     f"{g('flux', lambda m: md(m['vf_meas'])):.3g} | {g('ude', lambda m: md(m['vf_meas'])):.3g} | "
                     + ' / '.join(f"{g(k, lambda m: md(np.array(m['vf_meas'][:nb]) / ct0)):.3f}" for k in ('wired', 'flux', 'ude')) + ' | '
                     + ' / '.join(f"{g(k, lambda m: md(m['rollout']['200'])):.2e}" for k in ('wired', 'flux', 'ude')) + ' | '
                     + ' / '.join(f"{g(k, lambda m: md(m['rollout']['20'])):.2e}" for k in ('wired', 'flux', 'ude')) + ' |')
    txt = '\n'.join(L)
    open(os.path.join(OUT, 'tables.md'), 'w').write(txt)
    print(txt)


if __name__ == '__main__':
    p = argparse.ArgumentParser()
    p.add_argument('--stage', required=True); p.add_argument('--model', default='wired')
    p.add_argument('--seed', type=int, default=0); p.add_argument('--iters', type=int, default=2000)
    p.add_argument('--threads', type=int, default=4)
    a = p.parse_args()
    torch.set_num_threads(a.threads)
    os.makedirs(OUT, exist_ok=True)
    st = a.stage
    if st == 'c1_train':
        stage_c1_train(a.seed, a.iters)
    elif st == 'ex_train':
        stage_ex_train(a.model, a.seed, a.iters)
    else:
        dict(c1_path=stage_c1_path, pyth=stage_pyth, proj=stage_proj, coarse=stage_coarse, ex_data=stage_ex_data,
             ex_eval=stage_ex_eval, plot=stage_plot, report=stage_report)[st]()
