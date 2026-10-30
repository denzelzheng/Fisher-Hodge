"""Training on the Brusselator: the common rollout protocol with a wall-clock budget, the one-step protocols of
BSMS-GNN and GENERIC-FNO, and the driver that trains, evaluates and writes the result file of one model."""
import json
import math
import os
import time

import numpy as np
import torch

from neuralchart import fast
from neuralchart.chart import NeuralChart, fit_chart
from neuralchart.fit import take, rollout, fit_linear
from benchmarks.common import check_gpu
from .config import (DT, W_TRAIN, N_TRAIN, H_GRID, STEP, MAXSUB_TRAIN, MAXSUB_EVAL, ITERS, LR, BATCH, KMIN, KMAX, BUDGET_S,
                     RUNS)
from .data import load_set, train_norm
from .models import build, true_log_k, FLOAT32_MODELS
from .evaluate import evaluate



def set_max_sub(model, k):
    STEP['max_sub'] = k
    for mod in model.modules() if hasattr(model, 'modules') else []:
        st = getattr(mod, 'step', None)
        if isinstance(st, dict) and 'max_sub' in st:
            st['max_sub'] = k


def train_budget(model, tr, iters, seed, dt, batch=8, lr=1e-3, kmin=4, kmax=8, log=print, budget_s=BUDGET_S,
                 eta_min=1e-5, elapsed0=0.0):
    """neuralchart.fit.train (same sampling, loss, Adam, gradient clip) with a wall-clock budget: the cosine learning
    rate follows the progress p = max(step / iters, elapsed / budget_s), and training stops at p = 1.  A model that
    keeps step / iters >= elapsed / budget_s throughout gets exactly the step-count cosine.  Also records the largest
    number of substeps per sampling interval in every step."""
    rng = np.random.default_rng(10_000 + seed)
    C = tr['c']; ntr = C.shape[0]
    window = C.shape[1] - 1
    opt = torch.optim.Adam(model.parameters(), lr=lr)
    lr0 = lr
    hist, subs, smax, nskip = [], [], [], 0
    t0 = time.time() - elapsed0
    model.train()
    it, p, trunc = 0, 0.0, False
    while it < iters:
        p = max(it / iters, (time.time() - t0) / budget_s)
        if p >= 1.0:
            trunc = True
            break
        for g in opt.param_groups:
            g['lr'] = eta_min + (lr0 - eta_min) * 0.5 * (1 + math.cos(math.pi * p))
        idx = torch.tensor(rng.choice(ntr, min(batch, ntr), replace=False), device=C.device)
        K = int(rng.integers(kmin, kmax + 1))
        st = torch.tensor(rng.integers(0, window - K + 1, size=len(idx)), device=C.device)
        ctx = model.prepare(take(tr['drive'], idx))
        c = C[idx, st]
        loss = 0.0; ns = 0.0; nmax = 0
        for k in range(1, K + 1):
            c, n = model.advance(c, ctx, dt)
            ns = ns + (n.float().mean() if torch.is_tensor(n) else n)
            nmax = max(nmax, int(n.max()) if torch.is_tensor(n) else int(n))
            y = C[idx, st + k]
            loss = loss + (((c - y) ** 2).sum(-1) / (y ** 2).sum(-1)).mean()
        loss = loss / K
        subs.append(float(ns) / K); smax.append(nmax)
        if not torch.isfinite(loss):
            nskip += 1; opt.zero_grad(set_to_none=True); it += 1; continue
        opt.zero_grad(set_to_none=True); loss.backward()
        gn = torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        opt.step()
        hist.append(float(loss.detach()))
        if it % 100 == 0 or it == iters - 1:
            log(f'it {it:5d} loss {np.mean(hist[-100:]):.4e} gnorm {float(gn):.2e} K {K} substeps/DT {np.mean(subs[-100:]):.1f} '
                f'(max {max(smax[-100:])}) lr {opt.param_groups[0]["lr"]:.2e} p {p:.3f} skipped {nskip} t {time.time() - t0:.0f}s'
                + fast.cap_report(model))
        it += 1
    el = time.time() - t0
    log(f'training ended: {it} steps, {el:.0f}s, truncated by the budget: {trunc}, final loss {np.mean(hist[-100:]) if hist else float("nan"):.4e}')
    return dict(train_time_s=el, final_train_loss=float(np.mean(hist[-100:])) if hist else None, train_nskip=nskip,
                train_substeps_per_DT=float(np.mean(subs)) if subs else None, train_substeps_max=int(max(smax)) if smax else None,
                train_steps_done=it, truncated=trunc, budget_s=budget_s,
                loss_curve=[float(np.mean(hist[i:i + 100])) for i in range(0, len(hist), 100)])


def train_bsms(model, tr, iters, seed, log, noise=None, batch=8, n_val=8, tune_iters=1000):
    """the MeshGraphNets protocol for two state channels: one-step loss on normalised targets, Gaussian input noise
    with the target corrected for it, Adam with the learning rate decaying exponentially from 1e-4 to 1e-6.  The noise
    std is chosen from {1e-3, 3e-3, 1e-2} x the data std by tune_iters-step runs on the first 56 training trajectories,
    scored by the rollout error over the training window on the last n_val; the final model is trained for `iters`
    steps on all training trajectories."""
    C = tr['c']
    B, W1, _ = C.shape
    U = C.reshape(B, W1, 2, -1).transpose(2, 3).float()          # (B, W+1, V, 2)
    d = U[:, 1:] - U[:, :-1]
    model.c_mean.copy_(U.mean((0, 1, 2))); model.c_std.copy_(U.std((0, 1, 2)))
    model.d_mean.copy_(d.mean((0, 1, 2))); model.d_std.copy_(d.std((0, 1, 2)))
    W = W1 - 1
    init = {k: v.clone() for k, v in model.state_dict().items()}

    def fit(idx, n_it, sig, log_every=500):
        model.load_state_dict(init)
        opt = torch.optim.Adam(model.net.parameters(), lr=1e-4)
        sched = torch.optim.lr_scheduler.ExponentialLR(opt, (1e-6 / 1e-4) ** (1.0 / max(n_it, 1)))
        g = torch.Generator(device='cpu').manual_seed(10_000 + seed)
        hist = []
        for it in range(n_it):
            bi = idx[torch.randint(len(idx), (batch,), generator=g)].to(C.device)
            ti = torch.randint(W, (batch,), generator=g).to(C.device)
            ctx = model.prepare({k: v[bi] for k, v in tr['drive'].items()})
            x0, x1 = U[bi, ti], U[bi, ti + 1]
            xn = x0 + sig * model.c_std * torch.randn(x0.shape, generator=g).to(x0.device)
            target = ((x1 - xn) - model.d_mean) / model.d_std
            loss = ((model.delta(xn, ctx) - target) ** 2).mean()
            opt.zero_grad(set_to_none=True); loss.backward(); opt.step(); sched.step()
            hist.append(float(loss))
            if it % log_every == 0 or it == n_it - 1:
                log(f'bsms noise {sig:g} it {it} loss {np.mean(hist[-100:]):.4e}')
        return hist

    @torch.no_grad()
    def val_error(idx):
        dr = {k: v[idx] for k, v in tr['drive'].items()}
        R, _, _ = rollout(model, dr, C[idx, 0], W, DT)
        e = torch.linalg.vector_norm(R[:, -1] - C[idx, -1], dim=-1) / torch.linalg.vector_norm(C[idx, -1], dim=-1)
        return float(torch.where(torch.isfinite(e), e, torch.full_like(e, 1e9)).median())
    allidx = torch.arange(B)
    tune = {}
    if noise is None:
        for sig in (1e-3, 3e-3, 1e-2):
            fit(allidx[:B - n_val], tune_iters, sig)
            tune[sig] = val_error(allidx[B - n_val:].to(C.device))
            log(f'bsms noise {sig:g}: validation rollout error over the window {tune[sig]:.4g}')
        noise = min(tune, key=tune.get)
    model.noise.fill_(noise)
    t0 = time.time()
    hist = fit(allidx, iters, noise)
    return dict(train_time_s=time.time() - t0, final_train_loss=float(np.mean(hist[-100:])), noise=noise,
                noise_tuning={str(k): v for k, v in tune.items()})


def train_generic_fno(model, tr, iters, seed, device, log, norm):
    """GENERIC-FNO with its published protocol (neuralchart.baselines.generic_fno.train_generic), `iters` gradient
    steps, fields scaled by c_scale and the z-scored conditions as constant channels"""
    from neuralchart.baselines.generic_fno import train_generic
    N, T1 = tr['c'].shape[:2]
    n = N_TRAIN
    X = (tr['c'].float().cpu() / norm['c_scale']).reshape(N, T1, model.NS, n, n)
    ctx = model.prepare({k: v.float() for k, v in tr['drive'].items()})
    cond = model.cond_fields(ctx['z'], n).cpu()
    t0 = time.time()
    info = train_generic(model.net, X, cond, steps=iters, device=device, seed=seed, log=log)
    info['train_time_s'] = time.time() - t0
    info.pop('loss_curve', None)
    info['final_train_loss'] = info.pop('final_loss')
    return info


def default_fast(name, args_fast):
    """the integrator settings of a model: the fast path with sample_err (error control only for the UDE closure)"""
    if args_fast is None:
        return None
    kw = dict(args_fast)
    if name == 'ude':
        kw.update(control='err_only', compact=True)
    return kw


def accelerate(model, kw):
    if kw is not None and fast.supports(model):
        fast.restore(model)
        fast.accelerate(model, **kw)
    return model


def run_train(name, seed=0, iters=ITERS, device='cuda', lr=LR, fast_kw=None, data=None, out=None, ckpt_dir=None):
    """train and evaluate one model; writes <out>/<name>.json, .log and the rollout snapshots <out>/snapshots/<name>.npz
    (out defaults to runs/brusselator) and the checkpoint <ckpt_dir>/<name>.pt (default: out)"""
    out = out or RUNS
    ckpt_dir = ckpt_dir or out
    check_gpu(device)
    torch.manual_seed(seed); np.random.seed(seed)
    dkw = {} if data is None else dict(data=data)
    norm = train_norm(**dkw)
    tr = load_set('train', device, **dkw)
    if name in FLOAT32_MODELS:
        tr['c'] = tr['c'].float()
        tr['drive'] = {k: v.float() for k, v in tr['drive'].items()}
    os.makedirs(out, exist_ok=True); os.makedirs(ckpt_dir, exist_ok=True)
    logf = open(os.path.join(out, name + '.log'), 'w')

    def log(*a):
        s = ' '.join(str(x) for x in a); print(s, flush=True); logf.write(s + '\n'); logf.flush()
    model = build(name, N_TRAIN, norm).to(device)
    if name in ('flux_gnn', 'ude'):
        model = model.float()
    fkw = default_fast(name, fast_kw)
    accelerate(model, fkw)
    is_chart = isinstance(model, NeuralChart)
    npar = sum(p.numel() for p in model.parameters() if p.requires_grad)
    cap = MAXSUB_TRAIN
    log(f'brusselator {name}: parameters {npar}, steps {iters}, lr {lr}, batch {BATCH}, K [{KMIN}, {KMAX}], window {W_TRAIN}, '
        f'device {device}, grid {N_TRAIN}, h {H_GRID:.4f}, integrator {fkw}, substep cap {cap} (training) / {MAXSUB_EVAL} (evaluation)')
    set_max_sub(model, cap)
    info, trained = dict(train_time_s=0.0), True
    if name == 'bsms':
        info = train_bsms(model, tr, iters, seed, log)
    elif name == 'generic_fno':
        info = train_generic_fno(model, tr, iters, seed, device, log, norm)
    elif is_chart:
        after = (lambda m: log(f'NNLS log k {m.kinetics.core.detach().cpu().numpy().round(4).tolist()} '
                               f'true {np.round(true_log_k(), 4).tolist()}')) if model.config.mu0 == 'given' else None
        info = fit_chart(model, tr, DT, iters, seed, log=log, trainer=train_budget, after_linear=after,
                         batch=BATCH, lr=lr, kmin=KMIN, kmax=KMAX)
        trained = model.config.train
    elif name != 'persistence':
        info = {}
        if hasattr(model, 'linear_parameters') and model.linear_parameters():
            info['linear_fit'] = fit_linear(model, dict(train=tr), DT, log=log)
        info.update(train_budget(model, tr, iters, seed, DT, batch=BATCH, lr=lr, kmin=KMIN, kmax=KMAX, log=log))
    else:
        trained = False
    if trained:
        torch.save(model.state_dict(), os.path.join(ckpt_dir, name + '.pt'))
    info.pop('loss_curve', None)
    st = getattr(getattr(model, '_fast', None), 'stepper', None)
    capped_train = st.cap_hits if st is not None else None
    res = evaluate_saved(name, model, norm, device, log, fkw, **dkw)
    res.update(seed=seed, params=npar, iters=iters, lr=lr, batch=BATCH, kmin=KMIN, kmax=KMAX, window=W_TRAIN,
               max_sub_train=cap, **info)
    if res.get('fast') is not None:
        res['fast']['capped_DT'] = capped_train
    write_result(out, name, res)
    log('done')
    return res


def evaluate_saved(name, model, norm, device, log, fkw, **dkw):
    """evaluate a model (trained or loaded) in its evaluation configuration; returns the result dict"""
    if hasattr(model, 'eval'):
        model.eval()
    set_max_sub(model, MAXSUB_EVAL)
    if name in ('flux_gnn', 'ude'):
        model = model.float()
    model = accelerate(model, fkw)
    ev, snap = evaluate(name, model, norm, device, log, **dkw)
    extra = {}
    if isinstance(model, NeuralChart):
        extra = dict(log_k=model.kinetics.core.detach().cpu().tolist(), mu0=model.free_energy.mu0.detach().cpu().tolist(),
                     chem_mu0=model.drive.chem_mu0.detach().cpu().tolist(), log_k_true=true_log_k())
    st = getattr(getattr(model, '_fast', None), 'stepper', None)
    res = dict(benchmark='brusselator', model=name, eval_precision='fp32', extra=extra, sets=ev,
               fast=(dict(fkw, capped_DT_eval=st.cap_hits) if (fkw is not None and st is not None) else None))
    res['_snap'] = snap
    return res


def write_result(out, name, res):
    """<out>/<name>.json, and the rollout snapshots (for figures) in <out>/snapshots/<name>.npz"""
    snap = res.pop('_snap', {})
    os.makedirs(out, exist_ok=True)
    json.dump(res, open(os.path.join(out, name + '.json'), 'w'), default=float)
    if snap:
        os.makedirs(os.path.join(out, 'snapshots'), exist_ok=True)
        np.savez_compressed(os.path.join(out, 'snapshots', name + '.npz'), **snap)
