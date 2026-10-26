"""C3 equilibrium anchor: do a few undriven relaxation trajectories fix the entropy production of the driven Brusselator?
(results/c3_equilibrium_anchor.md)

The reverse rates of the Brusselator (1e-3 of the forward ones) barely move the driven densities, and sigma contains
log(a / b), so driven data alone leave sigma weakly determined.  In the wired model the reverse rates follow from
local detailed balance with the standard potentials mu0, which are material constants shared by every condition.
An anchor trajectory relaxes at chemostat concentrations where both cycle affinities vanish (detailed balance), so
every reverse rate is as large as its forward rate there.

    python -m hyperme.c3_anchor --stage data                              # 5 anchor trajectories, 8 x 8, DT = 0.2, 20 steps
    python -m hyperme.c3_anchor --stage train --model wired|crnn|flux [--anchor] [--seed 0]
    python -m hyperme.c3_anchor --stage bound                             # sigma lower bounds of the trained models
    python -m hyperme.c3_anchor --stage report

Protocol as in hyperme/convex_training.py: fit() (closed-form step where the model has a linear part), 2000 Adam
iterations, lr 1e-2, batch 8, K in [4, 8], MPRK22 x 8 for every model.  Anchor runs add the 5 anchor trajectories
to the 48 driven training trajectories (9 percent).

Anchor conditions (true constants k_r = 1, eps = 1e-3): pick the equilibrium x* in [0.7, 1.5] and d log-uniform in
[300, 3000]; then a = eps x*, e = x* / eps, b = eps^2 d make every reaction balanced at x = x*, y* = eps x*.  The
initial state is a random smooth field around (x*, 0.3 .. 3) and relaxes towards equilibrium."""
import argparse, glob, json, os, time
import numpy as np
import scipy.optimize
import torch
from scipy.integrate import solve_ivp
from . import presets, baselines as BL
from .presets import ROOT
from .linear_fit import fit, assemble, nnls_kappa
from .convex_training import metrics
from .integrators import mprk

OUT = os.path.join(ROOT, 'runs', 'hyperme', 'anchor')
DT, EPS = presets.C3_DT, 1e-3
KF, KB, DIFF = (1.0, 1.0, 1.0, 1.0), (EPS,) * 4, (0.4, 3.6)


def hg_c3():
    return presets.c3_wired().hg


def true_rates(hg, logab, c):
    return presets.mass_action_rates(hg, logab, c, KF, KB, DIFF)


def smooth(rng, n, std):
    a, b = np.meshgrid(np.arange(n), np.arange(n), indexing='ij')
    f = np.zeros((n, n))
    for kx in range(-3, 4):
        for ky in range(0, 4):
            if (kx, ky) == (0, 0) or kx * kx + ky * ky > 9 or (ky == 0 and kx < 0):
                continue
            ph = 2 * np.pi * (kx * b + ky * a) / n
            f += rng.normal() * np.cos(ph) + rng.normal() * np.sin(ph)
    f -= f.mean()
    return (std * f / (f.std() + 1e-12)).ravel()


def stage_data(n_traj=5, seed=0, nsteps=presets.T_TRAIN):
    hg = hg_c3()
    rng = np.random.default_rng(seed + 700)
    S = torch.as_tensor(hg.S, dtype=torch.float64)
    out = dict(c=[], logab=[], sigma=[], xstar=[])
    for i in range(n_traj):
        xs = rng.uniform(0.7, 1.5); d = np.exp(rng.uniform(np.log(300), np.log(3000)))
        lab = torch.tensor([[np.log(EPS * xs), np.log(EPS ** 2 * d), np.log(d), np.log(xs / EPS)]], dtype=torch.float64)
        c0 = np.concatenate([xs * np.exp(smooth(rng, 8, 0.4)), np.exp(rng.uniform(np.log(0.3), np.log(3.0))) * np.exp(smooth(rng, 8, 0.4))])

        def f(t, x):
            a, b = true_rates(hg, lab, torch.as_tensor(np.maximum(x, 1e-300))[None])
            return ((a - b) @ S.T)[0].numpy()
        ts = np.arange(nsteps + 1) * DT
        sol = solve_ivp(f, (0, ts[-1]), c0, method='DOP853', t_eval=ts, rtol=1e-10, atol=1e-13)
        assert sol.success and sol.y.min() > 0
        C = torch.as_tensor(sol.y.T)
        a, b = true_rates(hg, lab.expand(len(ts), -1), C)
        # local detailed balance at the equilibrium (x*, eps x*): every reaction balanced
        ce = torch.cat([torch.full((1, hg.V), xs, dtype=torch.float64), torch.full((1, hg.V), EPS * xs, dtype=torch.float64)], 1)
        ae, be = true_rates(hg, lab, ce)
        assert torch.allclose(ae, be, rtol=1e-10)
        out['c'].append(C); out['logab'].append(lab[0]); out['sigma'].append(((a - b) * (a.log() - b.log())).sum(-1)); out['xstar'].append(xs)
        print(f'anchor {i}: x* {xs:.3f} d {d:.0f} | x {C[:, :64].mean(1)[[0, -1]].tolist()} y {C[:, 64:].mean(1)[[0, -1]].tolist()} '
              f'| sigma {out["sigma"][-1][[0, -1]].tolist()}', flush=True)
    d = dict(c=torch.stack(out['c']), logab=torch.stack(out['logab']), sigma=torch.stack(out['sigma']), xstar=torch.tensor(out['xstar']))
    os.makedirs(OUT, exist_ok=True)
    torch.save(d, os.path.join(OUT, 'anchor.pt'))
    return d


def pad4(lab):
    return torch.cat([lab, torch.zeros(lab.shape[0], 4 - lab.shape[1], dtype=lab.dtype)], -1)


def load(anchor=False):
    """C3 data with drive = dict(logab (B, 4)); anchor=True appends the anchor trajectories to the training set."""
    raw = presets.load_c3()
    data = {}
    for k in ('train', 'test_id', 'test_ood'):
        d = raw[k]
        data[k] = dict(c=d['c'], drive=dict(logab=pad4(d['drive']['logab'])), sigma=d.get('sigma'), homog=d.get('homog'))
    if anchor:
        an = torch.load(os.path.join(OUT, 'anchor.pt'))
        tr = data['train']
        data['train'] = dict(c=torch.cat([tr['c'], an['c']]), drive=dict(logab=torch.cat([tr['drive']['logab'], an['logab']])))
    return data


# ------------------------------------------------------------------------------------------ black-box rates
def flux_featurizer(hg):
    """black-box inputs: hyperedge type one-hot and the four chemostat log concentrations as features (no wiring)."""
    ntyp = int(hg.meta['typ'].max()) + 1

    def feat(F):
        lab = F['logab']
        B = lab.shape[0]
        t = hg.on(lab.device, lab.dtype)
        oh = torch.nn.functional.one_hot(t['typ'], ntyp).to(lab.dtype).expand(B, hg.H, ntyp)
        sym = torch.cat([oh, lab[:, None, :].expand(B, hg.H, lab.shape[1])], -1)
        node = {f'lc{i}': lab[:, i:i + 1].expand(B, hg.V) for i in range(lab.shape[1])}
        return dict(U=torch.zeros(B, hg.V, dtype=lab.dtype, device=lab.device), node=node, edge_sym=sym,
                    edge_ori=torch.zeros_like(sym[..., 0]), force=None, raw=F)
    feat.n_edge_sym = ntyp + 4
    return feat


class FluxPatankar(BL.FluxGNN):
    """Flux-GNN (black-box forward and backward rates times mass action) with the MPRK22 x 8 time stepping"""
    def advance(s, c, drive, Dt, **kw):
        return mprk(s, c, s.context(drive), Dt, nsub=8, order=2)


def build(kind):
    if kind == 'wired':
        return presets.c3_wired()
    hg = hg_c3()
    if kind == 'crnn':
        return presets.crnn_chem(hg)
    f = flux_featurizer(hg)
    nn_in, ne_in = BL._Base.n_node_in(hg, f, dict(logab=torch.zeros(1, 4, dtype=torch.float64)))
    with presets.default_dtype(torch.float64):
        return FluxPatankar(hg, nn_in, ne_in, f, hidden=64, layers=3).double()


# ------------------------------------------------------------------------------------------ readouts
@torch.no_grad()
def rate_ratios(model, data, times=(0, 50, 100, 200)):
    """median log10(b / a) per reaction type over the test_id states, model and truth"""
    sd = data['test_id']; hg = model.hg
    typ = hg.meta['typ']
    out = {}
    lm, lt = [], []
    for t in times:
        c = sd['c'][:, t]
        a, b = model.rates(c, sd['drive'])
        at, bt = true_rates(hg, sd['drive']['logab'], c)
        lm.append(torch.log10(b / a)); lt.append(torch.log10(bt / at))
    lm, lt = torch.cat(lm), torch.cat(lt)
    for r, nm in enumerate(['R1', 'R2', 'R3', 'R4']):
        m = typ == r
        out[nm] = dict(model=float(lm[:, m].median()), true=float(lt[:, m].median()))
    return out


def run(kind, anchor, seed=0, iters=2000, lr=1e-2, out=OUT):
    torch.manual_seed(seed); np.random.seed(seed)
    model = build(kind)
    data = load(anchor)
    os.makedirs(out, exist_ok=True)
    name = f'c3_{kind}' + ('_anchor' if anchor else '') + f'_s{seed}'
    logf = open(os.path.join(out, name + '.log'), 'w')

    def log(*a):
        s = ' '.join(str(x) for x in a); print(s, flush=True); logf.write(s + '\n'); logf.flush()
    log(f'{name}: params={sum(p.numel() for p in model.parameters())} ntrain={data["train"]["c"].shape[0]} iters={iters} lr={lr}')
    info = dict(model=kind, anchor=anchor, seed=seed, iters=iters, lr=lr)
    info.update(fit(model, data, iters, DT, seed=seed, lr=lr, log=log))
    torch.save(model.state_dict(), os.path.join(out, name + '.pt'))
    model.eval()
    info['metrics'] = metrics('c3', model, data)
    info['rate_ratio'] = rate_ratios(model, data)
    log('final:', json.dumps(info['metrics']), json.dumps(info['rate_ratio']))
    json.dump(info, open(os.path.join(out, name + '.json'), 'w'), indent=1)
    return info


# ------------------------------------------------------------------------------------------ sigma lower bound
@torch.no_grad()
def _phi(model, sd, times):
    """one-way rates of the model at the given states, (n, H) each"""
    a, b = [], []
    for t in times:
        x, y = model.rates(sd['c'][:, t], sd['drive'])
        a.append(x); b.append(y)
    return torch.cat(a).double(), torch.cat(b).double()


def _sigma(kp, km, Pp, Pm):
    a, b = kp * Pp, km * Pm
    return ((a - b) * (torch.log(a) - torch.log(b))).sum(-1)


def sigma_bound(model, data, tol=0.1, floor=1e-12):
    """the smallest entropy production among the models of the class whose derivative fit on the training data is
    within a factor (1 + tol) of the best fit.  The fit residual is a convex quadratic in the class factors kappa and
    sigma = sum_h (a - b) log(a / b) is jointly convex in (a, b), which are linear in kappa, so this is a convex
    program: its optimum is global.  Returns sigma of the minimiser and of the best fit along the true test states,
    relative to the true sigma."""
    tr = data['train']
    A = assemble(model, tr, DT)
    G, h, yy, rp, rm, fixed = A['G'], A['h'], A['yy'], A['rp'], A['rm'], A['fixed']
    kbest, free = nnls_kappa(G, h, fixed)
    res = lambda k: (yy - 2 * k @ h + k @ G @ k) / yy
    rmin = res(kbest)
    bound = (1 + tol) * rmin
    Pp, Pm = _phi(model, tr, range(tr['c'].shape[1]))
    rpt, rmt = torch.as_tensor(rp), torch.as_tensor(rm)
    idx = np.where(free)[0]

    def full(z):
        k = torch.ones(len(h), dtype=torch.float64)
        k[torch.as_tensor(idx)] = torch.exp(z)
        return k

    def obj(zn):
        z = torch.tensor(zn, requires_grad=True)
        k = full(z)
        s = _sigma(k[rpt], k[rmt], Pp, Pm).mean()
        g, = torch.autograd.grad(s, z)
        return float(s.detach()), g.numpy()

    def con(zn):
        k = full(torch.tensor(zn)).numpy()
        return (bound - res(k)) / rmin

    def con_jac(zn):
        k = full(torch.tensor(zn)).numpy()
        return (-(2 * (G @ k) - 2 * h)[idx] * k[idx]) / yy / rmin
    z0 = np.log(np.maximum(kbest[idx], 1e-8))
    s0 = obj(z0)[0]
    r = scipy.optimize.minimize(obj, z0, jac=True, method='SLSQP', bounds=[(np.log(floor), np.log(1e4))] * len(idx),
                                constraints=[dict(type='ineq', fun=con, jac=con_jac)], options=dict(maxiter=500, ftol=1e-12))
    kmin = full(torch.tensor(r.x)).numpy()
    out = dict(tol=tol, residual_best=float(rmin), residual_at_min=float(res(kmin)), sigma_train_best=s0, sigma_train_min=float(r.fun),
               success=bool(r.success), message=str(r.message))
    for st in ('test_id', 'test_ood'):
        sd = data[st]
        T1 = sd['c'].shape[1]
        Qp, Qm = _phi(model, sd, range(T1))
        st_true = torch.as_tensor(sd['sigma']).double().T.reshape(-1)          # (t, b) order of _phi
        lo = _sigma(torch.as_tensor(kmin)[rpt], torch.as_tensor(kmin)[rmt], Qp, Qm)
        kb_ = torch.as_tensor(np.maximum(kbest, 1e-6))                      # the floor of fit_linear
        be = _sigma(kb_[rpt], kb_[rmt], Qp, Qm)
        B = sd['c'].shape[0]
        ratio = lambda s: float(np.median((s.reshape(T1, B).mean(0) / st_true.reshape(T1, B).mean(0)).numpy()))
        out[st] = dict(lower_over_true=ratio(lo), best_over_true=ratio(be), frac_true_above_lower=float((st_true >= lo * (1 - 1e-9)).double().mean()))
    return out


def stage_bound(out=OUT, tols=(0.1, 1.0)):
    res = {}
    for f in sorted(glob.glob(os.path.join(out, 'c3_*_s*.json'))):
        info = json.load(open(f))
        if info['model'] == 'flux':
            continue
        name = os.path.basename(f)[:-5]
        m = build(info['model']); m.load_state_dict(torch.load(f[:-5] + '.pt')); m.eval()
        data = load(info['anchor'])
        res[name] = [sigma_bound(m, data, tol) for tol in tols]
        print(name, json.dumps(res[name]), flush=True)
    json.dump(res, open(os.path.join(out, 'bounds.json'), 'w'), indent=1)
    return res


LABEL = dict(wired='HyperME wired', crnn='CRNN-chem', flux='Flux-GNN (black-box rates)')


def stage_report(out=OUT):
    bounds = json.load(open(os.path.join(out, 'bounds.json'))) if os.path.exists(os.path.join(out, 'bounds.json')) else {}
    L = ['| model | anchor | seed | ID 10T | OOD 10T | σ rel err ID ; OOD | median log10(b / a) R1 / R2 / R3 / R4, model (true) | final loss |',
         '|---|---|---|---|---|---|---|---|']
    rows = [json.load(open(f)) for f in sorted(glob.glob(os.path.join(out, 'c3_*_s*.json')))]
    for r in sorted(rows, key=lambda r: (['wired', 'crnn', 'flux'].index(r['model']), r['anchor'], r['seed'])):
        m, q = r['metrics'], r['rate_ratio']
        L.append(f"| {LABEL[r['model']]} | {'yes' if r['anchor'] else 'no'} | {r['seed']} | {m['test_id']['T10']:.4f} | {m['test_ood']['T10']:.4f} | "
                 f"{m['test_id']['sigma']:.3f} ; {m['test_ood']['sigma']:.3f} | "
                 + ' / '.join(f"{q[k]['model']:.2f} ({q[k]['true']:.2f})" for k in ('R1', 'R2', 'R3', 'R4')) + f" | {r['final_train_loss']:.2e} |")
    L += ['', '| model | tol | residual best → at min | σ_min / σ_true ID ; OOD | σ_best / σ_true ID ; OOD | true σ ≥ bound (fraction of points) ID ; OOD |',
          '|---|---|---|---|---|---|']
    for name, bl in bounds.items():
        for b in bl:
            L.append(f"| {name} | {b['tol']} | {b['residual_best']:.2e} → {b['residual_at_min']:.2e} | {b['test_id']['lower_over_true']:.3f} ; {b['test_ood']['lower_over_true']:.3f} | "
                     f"{b['test_id']['best_over_true']:.3f} ; {b['test_ood']['best_over_true']:.3f} | {b['test_id']['frac_true_above_lower']:.3f} ; {b['test_ood']['frac_true_above_lower']:.3f} |")
    txt = '\n'.join(L)
    open(os.path.join(out, 'tables.md'), 'w').write(txt)
    print(txt)


if __name__ == '__main__':
    p = argparse.ArgumentParser()
    p.add_argument('--stage', required=True); p.add_argument('--model', default='wired')
    p.add_argument('--anchor', action='store_true'); p.add_argument('--seed', type=int, default=0)
    p.add_argument('--iters', type=int, default=2000); p.add_argument('--lr', type=float, default=1e-2)
    p.add_argument('--threads', type=int, default=8)
    a = p.parse_args()
    torch.set_num_threads(a.threads)
    if a.stage == 'data':
        stage_data()
    elif a.stage == 'train':
        run(a.model, a.anchor, a.seed, a.iters, a.lr)
    elif a.stage == 'bound':
        stage_bound()
    elif a.stage == 'report':
        stage_report()
