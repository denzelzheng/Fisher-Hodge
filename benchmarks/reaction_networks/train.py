"""Training of every model on the reaction-network benchmarks, one protocol for all (see config.py)."""
import dataclasses
import json
import math
import os
import time

import numpy as np
import torch

from neuralchart import chart, fast
from neuralchart.diagnostics import LDB_RULE, relerr_traj
from neuralchart.fit import fit_chart, fit_linear, rollout, take
from ..common import check_gpu
from . import config as C
from . import truth as TR
from .data import load_set, train_norm
from .evaluate import evaluate, lngamma_metrics, maxwell_equal_area, err_channels
from .models import FLOAT_MODELS, build, accelerate, set_max_sub, true_log_k

CKPT_EVERY = 250


def _save_ckpt(path, state):
    tmp = path + '.tmp'
    torch.save(state, tmp)
    os.replace(tmp, path)                          # a killed job leaves the previous complete checkpoint


def train_budget(model, tr, iters, seed, dt, batch=8, lr=1e-3, kmin=4, kmax=20, log=print, budget_s=None, eta_min=1e-5,
                 elapsed0=0.0, ckpt=None, resume=False, pre_info=None):
    """rollout training (sampling, loss, Adam and clip of neuralchart.fit.train) with the cosine learning rate on
    p = max(step / iters, elapsed / budget) and a stop at p = 1.  elapsed0: time already spent by the closed-form step,
    counted in the budget.  ckpt: a checkpoint written every CKPT_EVERY steps; resume=True continues exactly from it."""
    budget_s = C.BUDGET_S if budget_s is None else budget_s
    rng = np.random.default_rng(10_000 + seed)
    Cs = tr['c']; ntr = Cs.shape[0]
    opt = torch.optim.Adam([p for p in model.parameters() if p.requires_grad], lr=lr)
    lr0 = lr
    hist, subs, smax, nskip = [], [], [], 0
    t0 = time.time() - elapsed0
    it, p, trunc = 0, 0.0, False
    st0 = getattr(getattr(model, '_fast', None), 'stepper', None)
    if resume:
        ck = torch.load(ckpt, map_location=next(iter(model.parameters())).device, weights_only=False)
        model.load_state_dict(ck['model']); opt.load_state_dict(ck['opt'])
        rng.bit_generator.state = ck['rng']
        it, hist, subs, smax, nskip, elapsed0 = ck['it'], ck['hist'], ck['subs'], ck['smax'], ck['nskip'], ck['elapsed0']
        t0 = time.time() - ck['elapsed']
        if st0 is not None:
            st0.cap_hits = ck.get('cap_hits', 0)
        log(f'resumed from {ckpt}: step {it}, elapsed {ck["elapsed"]:.0f}s')
    model.train()
    while it < iters:
        p = max(it / iters, (time.time() - t0) / budget_s)
        if p >= 1.0:
            trunc = True
            break
        for g in opt.param_groups:
            g['lr'] = eta_min + (lr0 - eta_min) * 0.5 * (1 + math.cos(math.pi * p))
        idx = torch.tensor(rng.choice(ntr, min(batch, ntr), replace=False), device=Cs.device)
        K = int(rng.integers(kmin, kmax + 1))
        st = torch.tensor(rng.integers(0, C.W_TRAIN - K + 1, size=len(idx)), device=Cs.device)
        ctx = model.prepare(take(tr['drive'], idx))
        c = Cs[idx, st]
        loss = 0.0; ns = 0.0; nmax = 0
        for k in range(1, K + 1):
            c, n = model.advance(c, ctx, dt)
            ns = ns + (n.float().mean() if torch.is_tensor(n) else n)
            nmax = max(nmax, int(n.max()) if torch.is_tensor(n) else int(n))
            y = Cs[idx, st + k]
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
        if ckpt is not None and it % CKPT_EVERY == 0:
            _save_ckpt(ckpt, dict(model=model.state_dict(), opt=opt.state_dict(), rng=rng.bit_generator.state, it=it,
                                  hist=hist, subs=subs, smax=smax, nskip=nskip, elapsed=time.time() - t0, elapsed0=elapsed0,
                                  cap_hits=st0.cap_hits if st0 is not None else 0, pre_info=pre_info))
    el = time.time() - t0
    log(f'training ended: {it} steps, {el:.0f}s, truncated by the budget: {trunc}')
    return dict(train_time_s=el, final_train_loss=float(np.mean(hist[-100:])) if hist else None, train_nskip=nskip,
                train_substeps_per_DT=float(np.mean(subs)) if subs else None, train_substeps_max=int(max(smax)) if smax else None,
                train_steps_done=it, truncated=trunc, budget_s=budget_s, closed_form_time_s=elapsed0,
                loss_curve=[float(np.mean(hist[i:i + 100])) for i in range(0, len(hist), 100)])


def run_bsms(model, tr, iters, seed, log, noise=None, batch=8, n_val=8, tune_iters=1000):
    """BSMS-GNN protocol: one-step loss on normalised targets, input noise chosen from {1e-3, 3e-3, 1e-2} x the data std
    on held-out training trajectories, Adam with the learning rate decaying exponentially from 1e-4 to 1e-6"""
    Cs = tr['c'].float()
    B, W1, _ = Cs.shape
    U = Cs.reshape(B, W1, model.NS, -1).transpose(2, 3)
    d = U[:, 1:] - U[:, :-1]
    model.c_mean.copy_(U.mean((0, 1, 2))); model.c_std.copy_(U.std((0, 1, 2)))
    model.d_mean.copy_(d.mean((0, 1, 2))); model.d_std.copy_(d.std((0, 1, 2)))
    for k in ('c_std', 'd_std'):                       # a constant channel (the immobile enzyme E): no rescaling
        v = getattr(model, k)
        v.copy_(torch.where(v > 0, v, torch.ones_like(v)))
    W = W1 - 1
    init = {k: v.clone() for k, v in model.state_dict().items()}

    def fit(idx, n_it, sig, log_every=500):
        model.load_state_dict(init)
        opt = torch.optim.Adam(model.net.parameters(), lr=1e-4)
        sched = torch.optim.lr_scheduler.ExponentialLR(opt, (1e-6 / 1e-4) ** (1.0 / max(n_it, 1)))
        g = torch.Generator(device='cpu').manual_seed(10_000 + seed)
        hist = []
        for it in range(n_it):
            bi = idx[torch.randint(len(idx), (batch,), generator=g)].to(Cs.device)
            ti = torch.randint(W, (batch,), generator=g).to(Cs.device)
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
        R, _, _ = rollout(model, dr, Cs[idx, 0], W, C.DT)
        e = torch.linalg.vector_norm(R[:, -1] - Cs[idx, -1], dim=-1) / torch.linalg.vector_norm(Cs[idx, -1], dim=-1)
        return float(torch.where(torch.isfinite(e), e, torch.full_like(e, 1e9)).median())
    allidx = torch.arange(B)
    tune = {}
    if noise is None:
        for sig in (1e-3, 3e-3, 1e-2):
            fit(allidx[:B - n_val], tune_iters, sig)
            tune[sig] = val_error(allidx[B - n_val:].to(Cs.device))
            log(f'bsms noise {sig:g}: validation rollout error over the window {tune[sig]:.4g}')
        noise = min(tune, key=tune.get)
    model.noise.fill_(noise)
    t0 = time.time()
    hist = fit(allidx, iters, noise)
    return dict(train_time_s=time.time() - t0, final_train_loss=float(np.mean(hist[-100:])), noise=noise,
                noise_tuning={str(k): v for k, v in tune.items()}, train_steps_done=iters, truncated=False)


def run_generic_fno(model, tr, iters, seed, device, log, norm):
    """GENERIC-FNO with its paper defaults (AdamW lr 1e-3, batch 16, cosine, one-step MSE plus a 2-step rollout term),
    4000 gradient steps"""
    from neuralchart.baselines.generic_fno import train_generic
    n = C.N_TRAIN
    N, T1 = tr['c'].shape[:2]
    X = (tr['c'].float().cpu() / norm['c_scale']).reshape(N, T1, model.NS, n, n)
    ctx = model.prepare({k: v.float() for k, v in tr['drive'].items()})
    cond = model.cond_fields(ctx['z'], ctx.get('U'), n).cpu()
    t0 = time.time()
    info = train_generic(model.net, X, cond, steps=iters, device=device, seed=seed, log=log)
    info.pop('loss_curve', None)
    return dict(info, train_time_s=time.time() - t0)


def run_train(name, seed=0, iters=None, device='cuda', lr=None, resume=False):
    """train one model, evaluate it on every test set; writes <out>/<name>.json, .pt and .log (out: runs/<benchmark>)"""
    iters = C.ITERS if iters is None else iters
    lr = C.LR if lr is None else lr
    check_gpu(device)
    torch.manual_seed(seed); np.random.seed(seed)
    norm = train_norm()
    tr = load_set('train', device)
    float_model = name in FLOAT_MODELS
    if float_model:
        tr['c'] = tr['c'].float(); tr['drive'] = {k: v.float() for k, v in tr['drive'].items()}
    out = C.OUT
    os.makedirs(out, exist_ok=True)
    ckpt = os.path.join(out, f'{name}.ckpt.pt')
    resume = bool(resume and os.path.exists(ckpt))
    logf = open(os.path.join(out, f'{name}.log'), 'a' if resume else 'w')

    def log(*a):
        s = ' '.join(str(x) for x in a); print(s, flush=True); logf.write(s + '\n'); logf.flush()
    model = build(name, C.N_TRAIN, norm).to(device)
    if float_model:
        model = model.float()
    accelerate(model, name)
    is_chart = isinstance(model, chart.NeuralChart)
    npar = sum(p.numel() for p in model.parameters() if p.requires_grad)
    log(f'{C.SYSTEM} {name}: params {npar} iters {iters} lr {lr} batch {C.BATCH} K [{C.KMIN},{C.KMAX}] window {C.W_TRAIN} '
        f'device {device} integrator {C.FAST}')
    set_max_sub(model, C.MAXSUB_TRAIN)
    info = dict(train_time_s=0.0)
    if is_chart:
        if resume:
            fitted = dict(torch.load(ckpt, weights_only=False)['pre_info'] or {})
        else:                                     # the closed-form step (variable-projected NNLS) ...
            fitted = fit_chart(model, tr, C.DT, 0, seed, log=log)
        if model.config.train and iters > 0:      # ... then the rollout refinement
            fitted.update(train_budget(model, tr, iters, seed, C.DT, batch=C.BATCH, lr=lr, kmin=C.KMIN, kmax=C.KMAX, log=log,
                                       elapsed0=fitted.get('varpro', {}).get('time_s', 0.0), ckpt=ckpt, resume=resume,
                                       pre_info=dict(fitted)))
        info = fitted if model.config.train else info
    elif name == 'bsms':
        info = run_bsms(model, tr, iters, seed, log)
    elif name == 'generic_fno':
        info = run_generic_fno(model, tr, iters, seed, device, log, norm)
    elif name != 'persistence':
        info = {}
        if resume:
            info = dict(torch.load(ckpt, weights_only=False)['pre_info'] or {})
        elif hasattr(model, 'linear_parameters') and model.linear_parameters():
            if getattr(model, 'kdu', None) is not None:
                # phosphorylation cycle: the rate constants depend on the potential through w+-, so the closed-form step
                # does not apply; CRNN (mass action) starts at unit rates and w = 0
                info['closed_form'] = 'not applicable (potential-dependent rate constants): unit rates, w = 0'
            else:
                info['linear_fit'] = fit_linear(model, dict(train=tr), C.DT, log=log)
        info.update(train_budget(model, tr, iters, seed, C.DT, batch=C.BATCH, lr=lr, kmin=C.KMIN, kmax=C.KMAX, log=log,
                                 ckpt=ckpt, resume=resume, pre_info=dict(info)))
    if name != 'persistence' and (not is_chart or model.config.train):
        torch.save(model.state_dict(), os.path.join(out, f'{name}.pt'))
    if hasattr(model, 'eval'):
        model.eval()
    st0 = getattr(getattr(model, '_fast', None), 'stepper', None)
    train_capped = st0.cap_hits if st0 is not None else None
    set_max_sub(model, C.MAXSUB_EVAL)
    accelerate(model, name)                      # a fresh evaluation integrator (FP32, same rtol), cap counter at 0
    extra = {}
    if is_chart:
        extra = dict(log_k=model.kinetics.core.detach().cpu().tolist(), mu0=model.free_energy.mu0.detach().cpu().tolist(),
                     chem_mu0=model.drive.chem_mu0.detach().cpu().tolist(), log_k_true=true_log_k())
        if getattr(model.free_energy, 'residual', False):
            extra['s_mu'] = model.free_energy.s_mu.detach().cpu().tolist()
        if C.SYSTEM == 'schlogl_nonideal':
            high = torch.load(os.path.join(C.DATA, 'high_concentration.pt'))
            extra['lngamma'] = lngamma_metrics(model, norm, float(high['c'].max()))
    if C.SYSTEM == 'schlogl' and hasattr(model, 'flux'):
        extra['maxwell'] = dict(equal_area_model=maxwell_equal_area(model), true=TR.maxwell_b(3.0), a=3.0)
    ev, snap = evaluate(name, model, norm, device, log)
    st = getattr(getattr(model, '_fast', None), 'stepper', None)
    res = dict(model=name, benchmark=C.SYSTEM, ldb_rule=LDB_RULE, config=dataclasses.asdict(model.config) if is_chart else None,
               seed=seed, params=npar, iters=iters, lr=lr, batch=C.BATCH, kmin=C.KMIN, kmax=C.KMAX, window=C.W_TRAIN,
               max_sub_train=C.MAXSUB_TRAIN, max_sub_eval=C.MAXSUB_EVAL, grid=C.N_TRAIN,
               **{k: v for k, v in info.items() if k != 'loss_curve'}, extra=extra, sets=ev)
    if C.FAST is not None:
        res['integrator'] = dict(C.FAST, capped_DT_train=train_capped, capped_DT_eval=st.cap_hits if st else None)
    json.dump(res, open(os.path.join(out, f'{name}.json'), 'w'), default=float)
    if os.path.exists(ckpt):
        os.remove(ckpt)
    log('done')
    return res


@torch.no_grad()
def check():
    """the oracle against the reference model: one-way rates on every set, and the NNLS step from the given potentials"""
    norm = train_norm()
    for s in ['train'] + C.sets():
        d = load_set(s)
        m = build('oracle', d['n'], norm)
        hg = TR.hg_of(d['n'])
        nt = d['c'][:, ::10].shape[1]
        c, lab = d['c'][:, ::10].reshape(-1, hg.M), {k: v.repeat_interleave(nt, 0) for k, v in d['drive'].items()}
        a, b = m.rates(c, lab)
        at, bt = TR.true_rates(hg, lab, c)
        print(f'{s}: oracle one-way rates against the reference: largest relative error '
              f'{float(((a - at).abs() / at).max()):.2e} {float(((b - bt).abs() / bt).max()):.2e}', flush=True)
    m = build('chart-ideal' if C.SYSTEM == 'schlogl_nonideal' else 'chart', C.N_TRAIN, norm)
    fit_linear(m, dict(train=load_set('train')), C.DT)
    print('NNLS log k', np.round(m.kinetics.core.numpy(), 5).tolist(), 'true', np.round(true_log_k(), 5).tolist(), flush=True)


@torch.no_grad()
def calibrate(device='cuda', rtols=(1e-2, 3e-3, 1e-3, 3e-4, 1e-4, 3e-5, 1e-5)):
    """how rtol was chosen: the 10T error of the oracle (true constants in the class) with the fast integrator
    (sample_err, FP32) against the reference, per rtol and test set"""
    check_gpu(device)
    norm = train_norm()
    res = {}
    for r in rtols:
        res[r] = {}
        for s in C.sets():
            sd = load_set(s, device)
            m = build('oracle', sd['n'], norm).to(device)
            fast.accelerate(m, control='sample_err', rtol=r, precision='fp32', step=C.STEP)
            set_max_sub(m, C.MAXSUB_EVAL)
            R, ns, _ = rollout(m, sd['drive'], sd['c'][:, 0], sd['c'].shape[1] - 1, C.DT)
            ch = err_channels(sd['n'])
            e = relerr_traj(R.double().cpu()[..., ch], sd['c'].double().cpu()[..., ch]).numpy()
            res[r][s] = dict(err_10T=float(np.median(e[:, -1])), err_max=float(e.max()), substeps=float(ns))
            print(f'rtol {r:g} {s}: 10T median {res[r][s]["err_10T"]:.3e} largest {res[r][s]["err_max"]:.3e} '
                  f'substeps/DT {ns:.1f}', flush=True)
    os.makedirs(C.OUT, exist_ok=True)
    json.dump({str(k): v for k, v in res.items()}, open(os.path.join(C.OUT, 'rtol_calibration.json'), 'w'), indent=1)
    return res
