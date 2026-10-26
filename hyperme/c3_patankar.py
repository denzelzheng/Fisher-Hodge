"""C3 Brusselator: explicit positive substeps vs the linearly implicit Patankar schemes on the trained HyperME
(hyper/runs/C23/brus_hyperme_s0.pt loaded into presets.c3).  Fixed sampling step 0.2, 200 steps (10T).

    python -m hyperme.c3_patankar [--ckpt hyper/runs/C23/brus_hyperme_s0.pt] [--out runs/hyperme]
Writes c3_patankar.json and c3_patankar.md: substeps per step, wall clock, 10T error against the reference solution
(all 16 test_id trajectories, homogeneous / spatial), distance to the explicit rollout, min c, min sigma."""
import argparse, json, os, time
import numpy as np
import torch
from . import presets
from .train import relerr

METHODS = [('ssp', 0), ('mprk22', 1), ('mprk22', 2), ('mprk22', 4), ('mprk22', 8), ('mprk22', 16), ('patankar', 1),
           ('patankar', 4)]


@torch.no_grad()
def run_method(model, sd, method, nsub, dt=0.2, nsteps=200):
    model.step.pop('method', None); model.step.pop('nsub', None)
    if method != 'ssp':
        model.step.update(method=method, nsub=nsub)
    ctx = model.prepare(sd['drive'])
    c = sd['c'][:, 0].clone(); out = [c]; subs = []; smin = float('inf')
    t0 = time.time()
    for _ in range(nsteps):
        a, b = model.rates(c, ctx)
        smin = min(smin, float(((a - b) * (torch.log(a) - torch.log(b))).sum(-1).min()))
        c, n = model.advance(c, ctx, dt)
        subs.append(float(n.float().mean()) if torch.is_tensor(n) else float(n))
        out.append(c)
    return torch.stack(out, 1), dict(wall_s=time.time() - t0, substeps_per_step=float(np.mean(subs)), sigma_min=smin)


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--ckpt', default=os.path.join(presets.ROOT, 'hyper', 'runs', 'C23', 'brus_hyperme_s0.pt'))
    p.add_argument('--out', default=os.path.join(presets.ROOT, 'runs', 'hyperme'))
    p.add_argument('--threads', type=int, default=8)
    p.add_argument('--tag', default='')
    a = p.parse_args()
    torch.set_num_threads(a.threads)
    data = presets.load_c3()
    model = presets.c3()
    sd_ = torch.load(a.ckpt)
    if any(k.startswith('kmlp.') for k in sd_):
        presets.load_legacy_c3(model, sd_)
    else:
        model.load_state_dict(sd_)
    model.eval()
    te = data['test_id']
    Y = te['c']; h = np.asarray(te['homog']).astype(bool)
    res = {}; Rex = None
    for method, nsub in METHODS:
        R, info = run_method(model, te, method, nsub)
        e = relerr(R, Y).numpy()
        key = method if method == 'ssp' else f'{method}_{nsub}'
        if method == 'ssp':
            Rex = R
        d = relerr(R, Rex).numpy()
        info.update(err_10T_median=float(np.median(e[:, -1])), err_10T_homog=float(np.median(e[h, -1])),
                    err_10T_spatial=float(np.median(e[~h, -1])), err_T_median=float(np.median(e[:, 20])),
                    err_traj0_10T=float(e[0, -1]), err_traj8_10T=float(e[8, -1]),
                    dist_to_explicit_10T_median=float(np.median(d[:, -1])), min_c=float(R.min()),
                    positive=bool(R.min() > 0))
        res[key] = info
        print(key, json.dumps(info), flush=True)
    os.makedirs(a.out, exist_ok=True)
    json.dump(res, open(os.path.join(a.out, f'c3_patankar{a.tag}.json'), 'w'), indent=1)
    rows = ['| integrator | steps per 0.2 | wall s (200 steps, 16 traj) | 10T err all | homog / spatial | T err | dist. to explicit 10T | min c | min sigma |',
            '|---|---|---|---|---|---|---|---|---|']
    for k, v in res.items():
        rows.append(f'| {k} | {v["substeps_per_step"]:.1f} | {v["wall_s"]:.1f} | {v["err_10T_median"]:.4f} | {v["err_10T_homog"]:.4f} / '
                    f'{v["err_10T_spatial"]:.4f} | {v["err_T_median"]:.4f} | {v["dist_to_explicit_10T_median"]:.2e} | {v["min_c"]:.3f} | {v["sigma_min"]:.2e} |')
    open(os.path.join(a.out, f'c3_patankar{a.tag}.md'), 'w').write('\n'.join(rows) + '\n')
    print('\n'.join(rows))


if __name__ == '__main__':
    main()
