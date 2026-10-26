"""C3: one observable separates the parallel channels R1 reverse (X -> A) and R4 forward (X -> E)
(results/c3_one_observable.md).

Both channels remove one X at a rate proportional to x under every chemostat setting, so densities cannot split them
(proposition 1': one observable per loop of indistinguishable parallel channels).  Two ways to add the missing one:
  V1  observe the turnover of one reservoir: r_E(t) = sum over sites of J_R4(t), the net production of E, for every
      training trajectory.  Its relative error enters the rollout loss with weight 1 (hyperme.train) and the
      closed-form step as extra rows (hyperme.linear_fit.assemble), for every model that has fluxes.
  V2  know the standard free energies of the reservoirs: mu0_A - mu0_E (true value) is wired into the wired model.
      No extra data.  CRNN-chem and Flux-GNN have no standard potentials, so V2 does not apply to them.

    python -m hyperme.c3_observable --stage train --variant base|v1|v2 --model wired|crnn|flux [--seed 0] [--lr 1e-2]
    python -m hyperme.c3_observable --stage bound | report

Protocol as in results/c3_equilibrium_anchor.md (48 driven training trajectories, fit() with the closed-form step,
2000 Adam iterations, lr 1e-2, MPRK22 x 8); a Flux-GNN run that collapses is repeated once at lr 1e-3."""
import argparse, glob, json, os
import numpy as np
import scipy.optimize
import torch
from . import presets
from .presets import ROOT
from .linear_fit import fit, assemble, nnls_kappa
from .convex_training import metrics
from . import c3_anchor as CA

OUT = os.path.join(ROOT, 'runs', 'hyperme', 'observable')
DT = CA.DT


def r4_weights(hg):
    """obs_w (1, H): the net flux of R4 summed over sites, the net production rate of E"""
    return torch.as_tensor(hg.meta['typ'] == 3, dtype=torch.float64)[None]


def load(variant):
    data = CA.load(False)
    if variant == 'v1':
        hg = CA.hg_c3()
        Wo = r4_weights(hg)
        for k in ('train', 'test_id', 'test_ood'):
            d = data[k]
            B, T1, M = d['c'].shape
            a, b = CA.true_rates(hg, d['drive']['logab'].repeat_interleave(T1, 0), d['c'].reshape(B * T1, M))
            d['obs'] = ((a - b) @ Wo.T).reshape(B, T1, 1)
            d['obs_w'] = Wo
    return data


def build(kind, variant):
    if variant == 'v2':
        assert kind == 'wired', 'V2 (known reservoir standard potentials) exists only for the wired model'
        return presets.c3_wired(known_cycle=presets.C3_CYCLE_AE)
    return CA.build(kind)


@torch.no_grad()
def sigma_shares(model, data, st='test_id', times=range(0, 201, 10)):
    """share of the true sigma carried by each hyperedge type, summed over the true test states: truth and model"""
    sd, hg = data[st], model.hg
    typ = torch.as_tensor(hg.meta['typ'])
    tru, mod = np.zeros(6), np.zeros(6)
    for t in times:
        c = sd['c'][:, t]
        for out, (a, b) in ((tru, CA.true_rates(hg, sd['drive']['logab'], c)), (mod, model.rates(c, sd['drive']))):
            s = (a - b) * (a.log() - b.log())
            for r in range(6):
                out[r] += float(s[:, typ == r].sum())
    tot = tru.sum()
    return dict(true=(tru / tot).tolist(), model=(mod / tot).tolist(), model_total=float(mod.sum() / tot))


def run(kind, variant, seed=0, iters=2000, lr=1e-2, out=OUT, tag=''):
    torch.manual_seed(seed); np.random.seed(seed)
    model = build(kind, variant)
    data = load(variant)
    os.makedirs(out, exist_ok=True)
    name = f'c3_{variant}_{kind}{tag}_s{seed}'
    logf = open(os.path.join(out, name + '.log'), 'w')

    def log(*a):
        s = ' '.join(str(x) for x in a); print(s, flush=True); logf.write(s + '\n'); logf.flush()
    log(f'{name}: params={sum(p.numel() for p in model.parameters())} iters={iters} lr={lr} observable={"obs" in data["train"]}')
    info = dict(model=kind, variant=variant, seed=seed, iters=iters, lr=lr, tag=tag)
    info.update(fit(model, data, iters, DT, seed=seed, lr=lr, log=log))
    torch.save(model.state_dict(), os.path.join(out, name + '.pt'))
    model.eval()
    info['metrics'] = metrics('c3', model, data)
    info['rate_ratio'] = CA.rate_ratios(model, data)
    info['shares'] = {st: sigma_shares(model, data, st) for st in ('test_id', 'test_ood')}
    log('final:', json.dumps(info['metrics']), json.dumps(info['rate_ratio']))
    json.dump(info, open(os.path.join(out, name + '.json'), 'w'), indent=1)
    if kind == 'flux' and lr > 1e-3 and (info['final_train_loss'] is None or info['final_train_loss'] > 0.05):
        log(f'collapsed (final loss {info["final_train_loss"]}): repeating once at lr 1e-3')
        return run(kind, variant, seed, iters, 1e-3, out, tag='_lr1e-3')
    return info


# ------------------------------------------------------------------------------------------ sigma range
def sigma_min_convex(model, data, tol=0.1, floor=1e-12, extra_start=None):
    """min of the training-state mean sigma over the class factors kappa (floor <= kappa <= 1e4) subject to
    residual <= (1 + tol) * best residual.  With a = x P, b = y M for the classes (x, y) = (kappa_rp, kappa_rm) of a
    hyperedge, sigma sums exactly to sum over class pairs of A x log(x/y) + B y log(y/x) + C x - D y, with A = sum P,
    B = sum M, C = sum P log(P/M), D = sum M log(P/M) over the states and hyperedges of the pair.  That is a convex
    function of a dozen variables and the residual a convex quadratic, so every KKT point is the global minimum.
    Solved with SLSQP in kappa from several starts, keeping the feasible result with the smallest sigma."""
    tr = data['train']
    A_ = assemble(model, tr, DT)
    G, h, yy, rp, rm, fixed = A_['G'], A_['h'], A_['yy'], A_['rp'], A_['rm'], A_['fixed']
    kn, free = nnls_kappa(G, h, fixed)
    res = lambda k: (yy - 2 * k @ h + k @ G @ k) / yy
    rmin = res(kn)
    bound = (1 + tol) * rmin
    Pp, Pm = CA._phi(model, tr, range(tr['c'].shape[1]))
    n = Pp.shape[0]
    L = torch.log(Pp) - torch.log(Pm)
    pairs = {}
    for hh in range(len(rp)):
        pairs.setdefault((int(rp[hh]), int(rm[hh])), []).append(hh)
    coef = [(p, m, float(Pp[:, e].sum()) / n, float(Pm[:, e].sum()) / n, float((Pp[:, e] * L[:, e]).sum()) / n,
             float((Pm[:, e] * L[:, e]).sum()) / n) for (p, m), e in ((k, torch.as_tensor(v)) for k, v in pairs.items())]
    idx = np.where(free)[0]
    sref = None

    def full(u):
        k = np.ones(len(h)); k[idx] = u
        return k

    def sig(u):
        k = full(u); g = np.zeros(len(h)); s = 0.0
        for p, m, A, B, C, D in coef:
            x, y = k[p], k[m]
            if p == m:
                s += x * (C - D); g[p] += C - D
                continue
            lxy = np.log(x / y)
            s += A * x * lxy - B * y * lxy + C * x - D * y
            g[p] += A * (lxy + 1) - B * y / x + C
            g[m] += -A * x / y + B * (1 - lxy) - D
        return s / sref, g[idx] / sref
    k0 = np.maximum(kn, 1e-8)
    sref = 1.0; sref = sig(k0[idx])[0]
    rg = lambda u: ((2 * (G @ full(u)) - 2 * h) / yy)[idx]
    cons = [dict(type='ineq', fun=lambda u: (bound - res(full(u))) / rmin, jac=lambda u: -rg(u) / rmin)]
    bnds = [(floor, 1e4)] * len(idx)
    best = None
    starts = [k0[idx], np.ones(len(idx)), np.sqrt(k0[idx])] + ([np.clip(extra_start[idx], floor, 1e4)] if extra_start is not None else [])
    for u0 in starts:
        r = scipy.optimize.minimize(sig, u0, jac=True, method='SLSQP', bounds=bnds, constraints=cons,
                                    options=dict(maxiter=2000, ftol=1e-14))
        if res(full(r.x)) <= bound * (1 + 1e-6) and (best is None or r.fun < best.fun):
            best = r
    if best is None:
        best = r
    return full(best.x), dict(residual_best=float(rmin), residual_at_min=float(res(full(best.x))), bound=float(bound),
                              rp=rp, rm=rm, converged=bool(best.success), sigma_train_min=float(best.fun * sref),
                              sigma_train_bestfit=float(sref))


def sigma_range(model, data, tol=0.1, extra_start=None):
    """the range of sigma over the models whose fit residual on the training data (densities and, in V1, the
    observable) is within (1 + tol) of the best.  Lower end: sigma_min_convex over the class factors (exact for
    independent classes; for V2, where the wired constant couples the classes, a superset of the class, so still a
    valid lower bound), evaluated along the true test states.  Upper end: infinite if a one-way class can be switched
    off within the tolerance, since log(a / b) then diverges."""
    kmin, info = sigma_min_convex(model, data, tol, extra_start=extra_start)
    rpt, rmt = torch.as_tensor(info['rp']), torch.as_tensor(info['rm'])
    out = dict(tol=tol, residual_best=info['residual_best'], residual_at_min=info['residual_at_min'],
               success=bool(info['residual_at_min'] <= info['bound'] * (1 + 1e-6) and info['converged']),
               sigma_train_min=info['sigma_train_min'], sigma_train_bestfit=info['sigma_train_bestfit'])
    for st in ('test_id', 'test_ood'):
        sd = data[st]
        T1, B = sd['c'].shape[1], sd['c'].shape[0]
        Qp, Qm = CA._phi(model, sd, range(T1))
        st_true = torch.as_tensor(sd['sigma']).double().T.reshape(-1)
        k = torch.as_tensor(kmin)
        s = CA._sigma(k[rpt], k[rmt], Qp, Qm)
        out[st] = dict(lower_over_true=float(np.median((s.reshape(T1, B).mean(0) / st_true.reshape(T1, B).mean(0)).numpy())),
                       frac_true_above_lower=float((st_true >= s * (1 - 1e-9)).double().mean()))
    A = assemble(model, data['train'], DT)
    G, h, yy, fixed = A['G'], A['h'], A['yy'], A['fixed']
    kn, free = nnls_kappa(G, h, fixed)
    res = lambda k: (yy - 2 * k @ h + k @ G @ k) / yy
    bound = (1 + tol) * res(kn)
    zero = []
    for r in np.where(free)[0]:
        f2 = free.copy(); f2[r] = False
        k2, _ = nnls_kappa(G * np.outer(f2, f2), h * f2, ~f2)
        k2[r] = 0.0
        if res(k2) <= bound:
            zero.append(int(r))
    out.update(n_switchable=len(zero), upper_finite=not zero, kappa_min=kmin.tolist())
    return out


def stage_bound(out=OUT, tols=(0.1, 1.0)):
    res = {}
    for f in sorted(glob.glob(os.path.join(out, 'c3_*_s*.json'))):
        info = json.load(open(f))
        if info['model'] == 'flux':
            continue
        name = os.path.basename(f)[:-5]
        m = build(info['model'], info['variant']); m.load_state_dict(torch.load(f[:-5] + '.pt')); m.eval()
        data = load(info['variant'])
        res[name], prev = [], None
        for tol in sorted(tols):                       # the tighter solution is feasible for the looser tolerance
            res[name].append(sigma_range(m, data, tol, extra_start=prev))
            prev = np.asarray(res[name][-1]['kappa_min'])
        print(name, json.dumps(res[name]), flush=True)
    json.dump(res, open(os.path.join(out, 'bounds.json'), 'w'), indent=1)


LABEL = dict(wired='HyperME wired', crnn='CRNN-chem', flux='Flux-GNN')


def stage_report(out=OUT):
    rows = [json.load(open(f)) for f in sorted(glob.glob(os.path.join(out, 'c3_*_s*.json')))]
    L = ['| variant | model | seed | lr | ID 10T | OOD 10T | σ rel err ID ; OOD | log10(b/a) R1 / R2 / R3 / R4 (true −3.21 / −3.63 / −2.78 / −2.79) '
         '| σ share R1 / R2 / R3 / R4, total / true (ID; true 0.138 / 0.332 / 0.382 / 0.135) | final loss |', '|---|---|---|---|---|---|---|---|---|---|']
    order = dict(base=0, v1=1, v2=2)
    for r in sorted(rows, key=lambda r: (order[r['variant']], ['wired', 'crnn', 'flux'].index(r['model']), r['seed'], r['lr'])):
        m, q, s = r['metrics'], r['rate_ratio'], r['shares']['test_id']
        L.append(f"| {r['variant']} | {LABEL[r['model']]} | {r['seed']} | {r['lr']:g} | {m['test_id']['T10']:.4f} | {m['test_ood']['T10']:.4f} | "
                 f"{m['test_id']['sigma']:.3f} ; {m['test_ood']['sigma']:.3f} | " + ' / '.join(f"{q[k]['model']:.2f}" for k in ('R1', 'R2', 'R3', 'R4'))
                 + ' | ' + ' / '.join(f'{x:.3f}' for x in s['model'][:4]) + f", {s['model_total']:.2f} | {r['final_train_loss']:.2e} |")
    p = os.path.join(out, 'bounds.json')
    if os.path.exists(p):
        L += ['', '| run | tol | best residual | σ_min / σ_true ID ; OOD | true ≥ σ_min (fraction) | one-way classes that can be switched off | upper end | SLSQP converged |',
              '|---|---|---|---|---|---|---|---|']
        for name, bl in json.load(open(p)).items():
            for b in bl:
                i, o = b['test_id'], b['test_ood']
                up = 'finite' if b['upper_finite'] else '∞'
                L.append(f"| {name} | {b['tol']} | {b['residual_best']:.2e} | {i['lower_over_true']:.3f} ; {o['lower_over_true']:.3f} | "
                         f"{i['frac_true_above_lower']:.3f} ; {o['frac_true_above_lower']:.3f} | {b['n_switchable']} | {up} | {b['success']} |")
    txt = '\n'.join(L)
    open(os.path.join(out, 'tables.md'), 'w').write(txt)
    print(txt)


if __name__ == '__main__':
    p = argparse.ArgumentParser()
    p.add_argument('--stage', required=True); p.add_argument('--variant', default='v1'); p.add_argument('--model', default='wired')
    p.add_argument('--seed', type=int, default=0); p.add_argument('--iters', type=int, default=2000)
    p.add_argument('--lr', type=float, default=1e-2); p.add_argument('--threads', type=int, default=4)
    a = p.parse_args()
    torch.set_num_threads(a.threads)
    if a.stage == 'train':
        run(a.model, a.variant, a.seed, a.iters, a.lr)
    elif a.stage == 'bound':
        stage_bound()
    elif a.stage == 'report':
        stage_report()
