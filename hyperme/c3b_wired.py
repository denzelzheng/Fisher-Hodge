"""C3b with the wired mode: the Brusselator chemostats (a, b) enter only through their chemical potential, in the force A
and in the symmetric prefactor k.  Train with `python -m hyperme.train --problem c3_wired --method mprk22 --nsub 8
--lr <lr> --out runs/hyperme/c3b` (add --no_chem_prefactor for the design without the prefactor), then

    python -m hyperme.c3b_wired [--out runs/hyperme/c3b]

prints the C3b table rows (10T, period and amplitude, sigma, learned cycle affinities) for every run in --out, in the
metrics of hyper/analyze_c23.py, and the same rows for the true parameters."""
import argparse, glob, json, os
import numpy as np
import torch
from . import presets
from .presets import ROOT
from .train import relerr

DT, N = presets.C3_DT, 8


def osc_stats(xm):
    """hyper/analyze_c23.osc_stats: amplitude and period of the site-mean X over the second half."""
    if not np.isfinite(xm).all():
        return np.nan, np.nan
    h = xm[len(xm) // 2:]
    amp = h.max() - h.min()
    dd = h - h.mean()
    up = np.where((dd[:-1] < 0) & (dd[1:] >= 0))[0]
    if len(up) < 2 or amp < 1e-3 * max(abs(h.mean()), 1e-9):
        return amp, np.nan
    tc = up + dd[up] / (dd[up] - dd[up + 1])
    return amp, float(np.mean(np.diff(tc)) * DT)


@torch.no_grad()
def rollout_sigma(model, drive, c0, nsteps):
    ctx = model.prepare(drive)
    c, out, sig = c0.clone(), [c0], []
    for _ in range(nsteps + 1):
        a, b = model.rates(c, ctx)
        sig.append(((a - b) * (torch.log(a) - torch.log(b))).sum(-1))
        if len(out) <= nsteps:
            c, _ = model.advance(c, ctx, DT)
            out.append(c)
    return torch.stack(out, 1), torch.stack(sig, 1)


def metrics(model, data):
    raw = torch.load(os.path.join(ROOT, 'hyper', 'runs', 'data', 'brus.pt'))
    res = {}
    for st in ('test_id', 'test_ood'):
        sd = data[st]; Y = sd['c']
        R, S = rollout_sigma(model, sd['drive'], Y[:, 0], Y.shape[1] - 1)
        e = relerr(R, Y).numpy()
        h = np.asarray(sd['homog']).astype(bool)
        Xr = R.reshape(*R.shape[:2], 2, -1)[:, :, 0].mean(-1).numpy(); Xt = Y.reshape(*Y.shape[:2], 2, -1)[:, :, 0].mean(-1).numpy()
        pe, ae = [], []
        for i in np.where(h)[0]:
            at, pt = osc_stats(Xt[i]); am, pm = osc_stats(Xr[i])
            ae.append(abs(am - at) / at if np.isfinite(am) else np.inf)
            pe.append(abs(pm - pt) / pt if np.isfinite(pm) else np.inf)
        st_true = raw[st]['sigma'].numpy()
        se = np.linalg.norm(S.numpy() - st_true, axis=1) / np.linalg.norm(st_true, axis=1)
        res[st] = dict(T=float(np.median(e[:, 20])), T2=float(np.median(e[:, 40])), T10=float(np.median(e[:, -1])),
                       T10_homog=float(np.median(e[h, -1])), T10_spatial=float(np.median(e[~h, -1])),
                       period=float(np.median(pe)), amp=float(np.median(ae)), sigma=float(np.median(se)),
                       min_c=float(R.min()), n_neg=int((R.flatten(1) < 0).any(1).sum()))
    # cycle affinities: R1 + R4 (A -> E) = log a + mu0_A - mu0_E, R2 + R3 (B -> D) = log b + mu0_B - mu0_D (chemostat mu0)
    m0 = model.drive.chem_mu0.detach()
    lab = data['test_id']['drive']['logab']
    res['affinity'] = [float((lab[:, 0] + m0[0] - m0[3]).mean()), float((lab[:, 1] + m0[1] - m0[2]).mean())]
    res['affinity_true'] = [float((lab[:, 0] + np.log(1 / 1e-6)).mean()), float((lab[:, 1] + np.log(1 / 1e-6)).mean())]
    return res


def row(name, r, loss=None):
    i, o = r['test_id'], r['test_ood']
    return (f"| {name} | {i['T']:.3g} | {i['T10']:.3g} ({i['T10_homog']:.3g} / {i['T10_spatial']:.3g}) | {o['T']:.3g} | {o['T2']:.3g} | "
            f"{o['T10']:.3g} ({o['T10_homog']:.3g} / {o['T10_spatial']:.3g}) | {o['period']:.3f} | {o['amp']:.3f} | {o['n_neg']} | "
            f"{i['sigma']:.3f} ; {o['sigma']:.3f} | ({r['affinity'][0]:.1f}, {r['affinity'][1]:.1f}) |"
            + (f' {loss:.2e} |' if loss is not None else ' – |'))


if __name__ == '__main__':
    p = argparse.ArgumentParser()
    p.add_argument('--out', default=os.path.join(ROOT, 'runs', 'hyperme', 'c3b'))
    a = p.parse_args()
    torch.set_num_threads(8)
    data = presets.load_c3()
    rows, allres = [], {}
    for f in sorted(glob.glob(os.path.join(a.out, '**', 'c3_wired*_s0.json'), recursive=True)):
        info = json.load(open(f))
        name = os.path.relpath(f, a.out)[:-5]
        m = presets.c3_wired(**info['preset'])
        m.load_state_dict(torch.load(f[:-5] + '.pt')); m.eval()
        r = metrics(m, data); allres[name] = dict(r, final_train_loss=info['final_train_loss'], lr=info['lr'], train_time_s=info['train_time_s'])
        rows.append(row(f"{name} (lr {info['lr']:g})", r, info['final_train_loss']))
    for method in ('ssp', 'mprk22'):
        m = presets.c3_true_wired(presets.c3_wired(method=method, nsub=8))
        r = metrics(m, data); allres[f'true_{method}'] = r
        rows.append(row(f'true parameters, {method}', r))
    print('| model | ID T | ID 10T (homog / spatial) | OOD T | OOD 2T | OOD 10T (homog / spatial) | OOD period rel err | OOD amp rel err '
          '| OOD traj c<0 | sigma rel err ID ; OOD | learned cycle affinities ID | final train loss |')
    print('|---|---|---|---|---|---|---|---|---|---|---|---|')
    print('\n'.join(rows))
    json.dump(allres, open(os.path.join(a.out, 'c3b_wired_metrics.json'), 'w'), indent=1)
