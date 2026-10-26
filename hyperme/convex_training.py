"""Training from scratch without a curriculum: the closed-form linear step and zero-gated residuals, every model
under one protocol (results/convex_training.md).

    python -m hyperme.convex_training --problem c3|c2|c5 --model wired|wired_mu|crnn [--no_linear] [--out DIR]
    python -m hyperme.convex_training --truth            # reference rows with the true constants
    python -m hyperme.convex_training --report           # tables from every run in --out

Protocol (all models): seed 0, the closed-form non-negative least-squares step of hyperme.linear_fit when the model
has a linear part (skipped with --no_linear), then 2000 Adam iterations of the rollout loss (K in [4, 8], batch 8,
cosine schedule from lr 1e-2, gradient clip 1), MPRK22 with 8 steps per sampling interval.  Metrics are taken right
after the linear step (iteration 0) and after training: median relative L2 error at T and 10T on test_id and
test_ood, relative error of the entropy production rate sigma(t), negative trajectories; for C3 the OOD period and
amplitude errors; for C5 the learned activity correction against ln gamma.

Models: wired = the wired HyperME preset of the problem (C2 and C5: schlogl_wired with the gated monotone dmu residual;
C3: c3_wired), wired_mu = c3_wired with the same dmu residual, crnn = CRNN-chem with a log rate-constant table."""
import argparse, glob, json, os, time
import numpy as np
import torch
from . import presets
from .presets import ROOT
from .train import train, relerr
from .linear_fit import fit_linear
from .c3b_wired import osc_stats

OUT = os.path.join(ROOT, 'runs', 'hyperme', 'convex')
DT = dict(c2=presets.C2_DT, c3=presets.C3_DT, c5=presets.C2_DT)
PHI = 0.25                                           # hyper/gen_nonideal.py


def lngamma(x):
    return PHI * x / (1 - PHI * x)


def build(problem, kind):
    if problem == 'c3':
        if kind == 'crnn':
            return presets.crnn_chem(presets.c3_wired().hg)
        return presets.c3_wired(nonideal_residual=kind == 'wired_mu')
    if kind == 'crnn':
        return presets.crnn_chem(presets.schlogl_hg())
    assert kind == 'wired'
    return presets.schlogl_wired()


def load(problem):
    return dict(c2=presets.load_schlogl, c3=presets.load_c3, c5=presets.load_c5)[problem]()


@torch.no_grad()
def rollout_sigma(model, drive, c0, nsteps, dt):
    ctx = model.prepare(drive)
    c, out, sig = c0.clone(), [c0], []
    for _ in range(nsteps + 1):
        a, b = model.rates(c, ctx)
        sig.append(((a - b) * (torch.log(a) - torch.log(b))).sum(-1))
        if len(out) <= nsteps:
            c, _ = model.advance(c, ctx, dt)
            c = torch.where(torch.isfinite(c).all(-1, keepdim=True), c, torch.full_like(c, float('nan')))
            out.append(c)
    return torch.stack(out, 1), torch.stack(sig, 1)


@torch.no_grad()
def activity(model, lo=0.23, hi=1.92, hi_ood=2.15):
    """learned dmu(x) at a uniform state against ln gamma(x), after removing the additive constant (absorbed by mu0)"""
    def dmu(x):
        c = torch.as_tensor(x, dtype=torch.float64)[:, None].expand(-1, model.hg.M).contiguous()
        return model.free_energy.correction(c, None)[:, 0].numpy()
    x = np.linspace(lo, hi, 60); xo = np.linspace(hi, hi_ood, 12)
    g, t = dmu(x), lngamma(x)
    off = np.mean(g - t)
    return dict(rmse_id=float(np.sqrt(np.mean((g - off - t) ** 2))), rmse_ood=float(np.sqrt(np.mean((dmu(xo) - off - lngamma(xo)) ** 2))),
                slope=float(np.polyfit(t, g, 1)[0]), s_mu=float(model.free_energy.s_mu[0]))


def metrics(problem, model, data):
    res = {}
    for st in ('test_id', 'test_ood'):
        sd = data[st]; Y = sd['c']
        R, S = rollout_sigma(model, sd['drive'], Y[:, 0], Y.shape[1] - 1, DT[problem])
        e = relerr(R, Y).numpy()
        st_true = sd['sigma'].numpy()
        se = np.linalg.norm(S.numpy() - st_true, axis=1) / np.linalg.norm(st_true, axis=1)
        r = dict(T=float(np.nanmedian(e[:, 20])), T10=float(np.nanmedian(e[:, -1])), sigma=float(np.nanmedian(se)),
                 n_blown=int(np.isnan(e[:, -1]).sum()), n_neg=int((R.flatten(1) < 0).any(1).sum()))
        if problem == 'c3':
            h = np.asarray(sd['homog']).astype(bool)
            r['T10_homog'], r['T10_spatial'] = float(np.nanmedian(e[h, -1])), float(np.nanmedian(e[~h, -1]))
            Xr = R.reshape(*R.shape[:2], 2, -1)[:, :, 0].mean(-1).numpy(); Xt = Y.reshape(*Y.shape[:2], 2, -1)[:, :, 0].mean(-1).numpy()
            pe, ae = [], []
            for i in np.where(h)[0]:
                at, pt = osc_stats(Xt[i]); am, pm = osc_stats(Xr[i])
                ae.append(abs(am - at) / at if np.isfinite(am) else np.inf)
                pe.append(abs(pm - pt) / pt if np.isfinite(pm) else np.inf)
            r['period'], r['amp'] = float(np.median(pe)), float(np.median(ae))
        res[st] = r
    if problem == 'c5' and getattr(getattr(model, 'free_energy', None), 'residual', False):
        res['activity'] = activity(model)
    return res


def run(problem, kind, linear=True, iters=2000, lr=1e-2, seed=0, out=OUT):
    torch.manual_seed(seed); np.random.seed(seed)
    model = build(problem, kind)
    data = load(problem)
    os.makedirs(out, exist_ok=True)
    name = f'{problem}_{kind}' + ('' if linear else '_nolin') + f'_s{seed}'
    logf = open(os.path.join(out, name + '.log'), 'w')

    def log(*a):
        s = ' '.join(str(x) for x in a); print(s, flush=True); logf.write(s + '\n'); logf.flush()
    log(f'{name}: params={sum(p.numel() for p in model.parameters())} iters={iters} lr={lr} linear={linear}')
    info = dict(problem=problem, model=kind, linear=linear, iters=iters, lr=lr, seed=seed)
    t0 = time.time()
    # model.fit(data, iters, dt) does these two steps; they are spelled out to evaluate the model in between
    if linear:
        info['linear_fit'] = fit_linear(model, data, DT[problem], log=log)
    info['linear_time_s'] = time.time() - t0
    model.eval(); info['metrics_it0'] = metrics(problem, model, data); model.train()
    log('iteration 0:', json.dumps(info['metrics_it0']))
    info.update(train(model, data['train'], iters, seed, DT[problem], lr=lr, log=log))
    torch.save(model.state_dict(), os.path.join(out, name + '.pt'))
    model.eval(); info['metrics'] = metrics(problem, model, data)
    log('final:', json.dumps(info['metrics']))
    json.dump(info, open(os.path.join(out, name + '.json'), 'w'), indent=1)
    return info


def truth(out=OUT):
    """reference rows: C3 with the true constants (exact class), C2 with the true constants (exact class), C5 with the
    true constants of the ideal law (the mass-action limit of the non-ideal truth)."""
    res = {}
    m = presets.c3_true_wired(presets.c3_wired())
    res['c3_true'] = metrics('c3', m.eval(), load('c3'))
    for p in ('c2', 'c5'):
        m = presets.schlogl_wired(nonideal_residual=False)
        err = presets.set_true_mass_action(m, **presets.C2_TRUE)
        assert err < 1e-9, err
        res[f'{p}_true_ideal'] = metrics(p, m.eval(), load(p))
    os.makedirs(out, exist_ok=True)
    json.dump(res, open(os.path.join(out, 'truth.json'), 'w'), indent=1)
    print(json.dumps(res, indent=1))
    return res


LABEL = dict(wired='HyperME wired', wired_mu='HyperME wired + gated dmu residual', crnn='CRNN-chem (rate table)')


def f3(x):
    return '–' if x is None else f'{x:.3g}'


def report(out=OUT):
    lines = []
    tru = json.load(open(os.path.join(out, 'truth.json'))) if os.path.exists(os.path.join(out, 'truth.json')) else {}
    for p in ('c3', 'c2', 'c5'):
        runs = [json.load(open(f)) for f in sorted(glob.glob(os.path.join(out, f'{p}_*_s0.json')))]
        if not runs:
            continue
        lines += [f'### {p.upper()}', '',
                  '| model | linear step | after linear step: ID 10T / OOD 10T | ID T | ID 10T | OOD T | OOD 10T | sigma rel err ID ; OOD '
                  '| c<0 traj | train loss at it 100 / 500 / 1000 / 2000 |' + (' OOD period / amp |' if p == 'c3' else '')
                  + (' dmu vs ln gamma: RMSE ID / slope |' if p == 'c5' else ''),
                  '|' + '---|' * (10 + (p in ('c3', 'c5')))]
        for r in sorted(runs, key=lambda r: (r['model'], not r['linear'])):
            m0, m = r['metrics_it0'], r['metrics']
            lc = r['loss_curve']
            pick = [lc[i] if i < len(lc) else None for i in (0, 4, 9, len(lc) - 1)]
            row = (f"| {LABEL[r['model']]} | {'NNLS' if r['linear'] else 'none (neutral start)'} | "
                   f"{f3(m0['test_id']['T10'])} / {f3(m0['test_ood']['T10'])} | {f3(m['test_id']['T'])} | {f3(m['test_id']['T10'])} | "
                   f"{f3(m['test_ood']['T'])} | {f3(m['test_ood']['T10'])} | {m['test_id']['sigma']:.3f} ; {m['test_ood']['sigma']:.3f} | "
                   f"{m['test_id']['n_neg'] + m['test_ood']['n_neg']} | {' / '.join(f'{x:.1e}' if x is not None else '–' for x in pick)} |")
            if p == 'c3':
                row += f" {m['test_ood']['period']:.3f} / {m['test_ood']['amp']:.3f} |"
            if p == 'c5':
                a = m.get('activity')
                row += f" {a['rmse_id']:.3f} / {a['slope']:.2f} |" if a else ' – |'
            lines.append(row)
        for k, m in tru.items():
            if k.startswith(p):
                row = (f"| true constants{' (ideal law)' if 'ideal' in k else ''} | – | – | {f3(m['test_id']['T'])} | {f3(m['test_id']['T10'])} | "
                       f"{f3(m['test_ood']['T'])} | {f3(m['test_ood']['T10'])} | {m['test_id']['sigma']:.3f} ; {m['test_ood']['sigma']:.3f} | "
                       f"{m['test_id']['n_neg'] + m['test_ood']['n_neg']} | – |")
                row += f" {m['test_ood']['period']:.3f} / {m['test_ood']['amp']:.3f} |" if p == 'c3' else (' – |' if p == 'c5' else '')
                lines.append(row)
        lines.append('')
        for r in runs:
            if r['linear']:
                lf = r['linear_fit']
                lines.append(f"- {LABEL[r['model']]}: {lf['n_classes']} rate classes, {lf['n_params']} linear parameters (rank {lf['rank']}), "
                             f"{lf['n_floored']} classes at the floor; relative derivative residual {lf['residual_init']:.2e} -> "
                             f"{lf['residual_theta']:.2e}; {lf['time_s']:.1f} s.")
        lines.append('')
    txt = '\n'.join(lines)
    open(os.path.join(out, 'tables.md'), 'w').write(txt)
    print(txt)


if __name__ == '__main__':
    p = argparse.ArgumentParser()
    p.add_argument('--problem', default='c3'); p.add_argument('--model', default='wired')
    p.add_argument('--no_linear', action='store_true'); p.add_argument('--iters', type=int, default=2000)
    p.add_argument('--lr', type=float, default=1e-2); p.add_argument('--seed', type=int, default=0)
    p.add_argument('--threads', type=int, default=4); p.add_argument('--out', default=OUT)
    p.add_argument('--truth', action='store_true'); p.add_argument('--report', action='store_true')
    a = p.parse_args()
    torch.set_num_threads(a.threads)
    if a.report:
        report(a.out)
    elif a.truth:
        truth(a.out)
    else:
        run(a.problem, a.model, not a.no_linear, a.iters, a.lr, a.seed, a.out)
