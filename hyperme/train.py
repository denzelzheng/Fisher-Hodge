"""Rollout training and evaluation for the presets (the protocol of fisher_hodge/train_tcme.py, hyper/train_hyper.py and
hyper/train_c23.py: multi-step rollout loss with K in [kmin, kmax], Adam, cosine learning rate, batch 8, gradient clip 1).

    python -m hyperme.train --problem c1 --seed 0 --iters 2000 --out runs/hyperme
    python -m hyperme.train --problem s2_tcme --seed 0 --iters 2500 --device cuda --out runs/hyperme
    python -m hyperme.train --problem c3 --seed 0 --method mprk22 --nsub 1      (Patankar integrator)
Models with a linear part (the wired presets, CRNN with a rate table) first get the closed-form least-squares fit of
hyperme.linear_fit (skip with --no_linear_init); the rollout training is the same for every model.
Writes <out>/<problem><tag>_s<seed>.json (and .pt, .log)."""
import argparse, json, os, time
import numpy as np
import torch
from .presets import PRESETS, T_TRAIN, ROOT

EVAL_SETS = dict(s2_tcme=['test_id', 'test_oodpot', 'test_oodinit'], s2_wired=['test_id', 'test_oodpot', 'test_oodinit'], c1=['test_id', 'test_oodG', 'test_oodU'],
                 c3=['test_id', 'test_ood'], c3_wired=['test_id', 'test_ood'], c2_wired=['test_id', 'test_ood'], c5_wired=['test_id', 'test_ood'])
DTS = dict(s2_tcme=0.05, s2_wired=0.05, c1=0.05, c3=0.2, c3_wired=0.2, c2_wired=0.05, c5_wired=0.05)


def take(drive, idx):
    return {k: v[idx] for k, v in drive.items()}


def train(model, tr, iters, seed, dt, batch=8, lr=1e-3, kmin=4, kmax=8, log=print, legacy_steps=False):
    """rollout training.  If tr has 'obs' (B, T+1, k) and 'obs_w' (k, H), the observables obs = J @ obs_w.T (linear
    functionals of the net fluxes, e.g. the turnover of a reservoir) add their relative error to the loss with weight 1."""
    obs, Wo = tr.get('obs'), tr.get('obs_w')
    rng = np.random.default_rng(10_000 + seed)
    C = tr['c']; ntr = C.shape[0]
    opt = torch.optim.Adam(model.parameters(), lr=lr)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, iters, eta_min=1e-5)
    hist, subs, nskip = [], [], 0
    t0 = time.time()
    model.train()
    for it in range(iters):
        idx = torch.tensor(rng.choice(ntr, min(batch, ntr), replace=False), device=C.device)
        K = int(rng.integers(kmin, kmax + 1))
        st = torch.tensor(rng.integers(0, T_TRAIN - K + 1, size=len(idx)), device=C.device)
        ctx = model.prepare(take(tr['drive'], idx))
        c = C[idx, st]
        loss = 0.0; ns = 0
        for k in range(1, K + 1):
            c, n = model.advance(c, ctx, dt, vectorized=False if legacy_steps else None)
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
            nskip += 1; opt.zero_grad(set_to_none=True); sched.step(); continue
        opt.zero_grad(set_to_none=True); loss.backward()
        gn = torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        opt.step(); sched.step()
        hist.append(float(loss.detach()))
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


@torch.no_grad()
def evaluate(problem, model, data, sets=None):
    res = {}
    hg = model.hg
    for name in sets or EVAL_SETS[problem]:
        sd = data[name]; Y = sd['c']
        t0 = time.time()
        R, ns, smin = rollout(model, sd['drive'], Y[:, 0], Y.shape[1] - 1, DTS[problem], sigma=problem[:2] in ('c2', 'c3', 'c5'))
        S = dict(infer_time_s=time.time() - t0, substeps_per_DT=ns, min_c=float(R.min()),
                 n_traj_negative=int((R.flatten(1) < 0).any(1).sum()))
        if problem == 'c1':
            V = hg.V
            e = relerr(R[..., :2 * V].double(), Y[..., :2 * V].double()).cpu().numpy()
            q = R @ hg.on(R.device, R.dtype)['L']                                   # 65 moieties
            cons = (torch.linalg.vector_norm(q - q[:, :1], dim=-1) / torch.linalg.vector_norm(q[:, :1], dim=-1)).max()
            S.update(n_conserved=int(q.shape[-1]), cons_relerr_max=float(cons))
        else:
            e = relerr(R.double(), Y.double()).cpu().numpy()
            if problem.startswith('s2'):
                S['mass_err_max'] = float((R.mean(-1) - 1.0).abs().max())
            else:
                S['sigma_min'] = smin
                if 'homog' in sd:
                    h = np.asarray(sd['homog']).astype(bool)
                    S['relerr_10T_median_homog'] = float(np.median(e[h, -1])); S['relerr_10T_median_spatial'] = float(np.median(e[~h, -1]))
        S['relerr_median'] = {str(t): float(np.median(e[:, t])) for t in (20, 40, 100, 200) if t < e.shape[1]}
        S['relerr_10T_per_traj'] = e[:, -1].tolist()
        res[name] = S
    return res


def run(problem, seed, iters, device, out, legacy_steps=False, sets=None, tag='', ntrain=0, lr=1e-3, linear_init=True, **preset_kw):
    from .gpu_guard import require_free
    require_free(device)
    build, load = PRESETS[problem]
    torch.manual_seed(seed); np.random.seed(seed)
    model = build(**preset_kw).to(device)
    data = load(device=device)
    os.makedirs(out, exist_ok=True)
    name = f'{problem}{tag}_s{seed}'
    logf = open(os.path.join(out, name + '.log'), 'w')

    def log(*a):
        s = ' '.join(str(x) for x in a); print(s, flush=True); logf.write(s + '\n'); logf.flush()
    log(f'{name}: params={sum(p.numel() for p in model.parameters())} iters={iters} lr={lr} device={device} preset={preset_kw}')
    tr = data['train'] if ntrain <= 0 else dict(c=data['train']['c'][:ntrain], drive={k: v[:ntrain] for k, v in data['train']['drive'].items()})
    from .linear_fit import fit
    info = fit(model, dict(train=tr), iters, DTS[problem], seed=seed, linear_init=linear_init, lr=lr, log=log, legacy_steps=legacy_steps)
    torch.save(model.state_dict(), os.path.join(out, name + '.pt'))
    model.eval()
    ev = evaluate(problem, model, data, sets)
    for k, v in ev.items():
        extra = (f' | cons {v["cons_relerr_max"]:.1e} ({v["n_conserved"]} moieties)' if 'cons_relerr_max' in v else
                 f' | mass {v["mass_err_max"]:.1e}' if 'mass_err_max' in v else f' | sigma min {v["sigma_min"]:.2e}')
        log(f'[{k}] relerr@T,2T,5T,10T median = ' + ', '.join(f'{x:.3e}' for x in v['relerr_median'].values())
            + f' | min c {v["min_c"]:.2e}' + extra + f' | substeps {v["substeps_per_DT"]:.1f} | infer {v["infer_time_s"]:.1f}s')
    res = dict(problem=problem, seed=seed, iters=iters, lr=lr, linear_init=linear_init, preset=preset_kw, **info, sets=ev)
    json.dump(res, open(os.path.join(out, name + '.json'), 'w'))
    return res


if __name__ == '__main__':
    p = argparse.ArgumentParser()
    p.add_argument('--problem', default='c1'); p.add_argument('--seed', type=int, default=0)
    p.add_argument('--iters', type=int, default=None); p.add_argument('--device', default='cpu')
    p.add_argument('--out', default=os.path.join(ROOT, 'runs', 'hyperme'))
    p.add_argument('--legacy_steps', action='store_true')
    p.add_argument('--method', default=None, help='c3, c3_wired: ssp | mprk22 | patankar')
    p.add_argument('--nsub', type=int, default=1)
    p.add_argument('--threads', type=int, default=4)
    p.add_argument('--ntrain', type=int, default=0, help='>0: use only the first ntrain training trajectories')
    p.add_argument('--lr', type=float, default=1e-3)
    p.add_argument('--no_chem_prefactor', action='store_true', help='c3_wired only: chemostats enter the force but not k')
    p.add_argument('--no_linear_init', action='store_true', help='skip the closed-form least-squares step of fit()')
    a = p.parse_args()
    iters = a.iters or dict(s2_tcme=2500, s2_wired=2500, c1=2000, c3=2000, c3_wired=2000, c2_wired=2000, c5_wired=2000)[a.problem]
    if a.device == 'cpu':
        torch.set_num_threads(a.threads)
    kw = dict(method=a.method, nsub=a.nsub) if a.method else {}
    if a.no_chem_prefactor:
        kw['chem_prefactor'] = False
    tag = ((f'_{a.method}{a.nsub}' if a.method else '') + (f'_n{a.ntrain}' if a.ntrain > 0 else '')
           + (f'_lr{a.lr:g}' if a.lr != 1e-3 else '') + ('_nochemk' if a.no_chem_prefactor else '') + ('_nolin' if a.no_linear_init else ''))
    run(a.problem, a.seed, iters, a.device, a.out, a.legacy_steps, tag=tag, ntrain=a.ntrain, lr=a.lr, linear_init=not a.no_linear_init, **kw)
