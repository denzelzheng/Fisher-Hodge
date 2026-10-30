"""Training of the porous-media benchmark.  The same protocol for every model: the 32 x 32 grid, 64 media x 4
trajectories at T = 1 and drive amplitude 1, a 5-time-unit window (100 steps of DT), 4000 gradient steps.  The
continuous-time models (Neural Chart, UDE closure, Flux-GNN) use the rollout loss with K in [4, 20] steps, batch 8,
Adam with a cosine schedule, gradient clip 1.  FNO uses the same loss.  BSMS-GNN, GENERIC-FNO and HelmFluid keep the
training protocols of their papers."""
import dataclasses
import json
import os
import time

import numpy as np
import torch

from neuralchart import fast
from neuralchart.diagnostics import LDB_RULE
from neuralchart.fit import fit_chart, train, rollout
from benchmarks.common import check_gpu
from .data import N, DT, T_TRAIN_DATA, load, add_norm, training_set
from .models import NC, build, node_cond
from .evaluate import eval_model, oracle_drive, rollout_metrics, phys_metrics_rollout, local_sigma
from .test_sets import TEST_SETS, DISC_OF


def fit_bsms(model, data, iters, seed, device, log, noise=None, batch=8, val_media=8, tune_iters=1500):
    """the MeshGraphNets training protocol: one-step loss on normalised targets, Gaussian input noise with the target corrected for it,
    Adam with the learning rate decaying exponentially from 1e-4 to 1e-6.  noise=None: the noise std is chosen once
    from {1e-3, 3e-3, 1e-2} x the data std on a validation split of the training media (the last `val_media` media),
    by the error of a rollout over the training window; the final model is then trained on all training media."""
    tr = data['train']
    C = tr['c']                                         # (B, W+1, V), training grid
    med = tr['med'].to(C.device)
    rho = C
    d = rho[:, 1:] - rho[:, :-1]
    model.rho_mean.fill_(float(rho.mean())); model.rho_std.fill_(float(rho.std()))
    model.d_mean.fill_(float(d.mean())); model.d_std.fill_(float(d.std()))
    W = C.shape[1] - 1
    init = {k: v.clone() for k, v in model.state_dict().items()}

    def fit(idx, n_it, sig, log_every=500):
        model.load_state_dict(init)
        opt = torch.optim.Adam(model.net.parameters(), lr=1e-4)
        gamma = (1e-6 / 1e-4) ** (1.0 / max(n_it, 1))
        sched = torch.optim.lr_scheduler.ExponentialLR(opt, gamma)
        g = torch.Generator(device='cpu').manual_seed(10_000 + seed)
        hist = []
        for it in range(n_it):
            bi = idx[torch.randint(len(idx), (batch,), generator=g).to(idx.device)]
            ti = torch.randint(W, (batch,), generator=g).to(idx.device)
            ctx = model.prepare({k: v[bi] for k, v in tr['drive'].items()})
            x0 = rho[bi, ti]; x1 = rho[bi, ti + 1]
            xn = x0 + sig * model.rho_std * torch.randn(x0.shape, generator=g).to(x0.device)
            target = ((x1 - xn) - model.d_mean) / model.d_std
            loss = ((model.delta(xn, ctx) - target) ** 2).mean()
            opt.zero_grad(set_to_none=True); loss.backward(); opt.step(); sched.step()
            hist.append(float(loss))
            if it % log_every == 0 or it == n_it - 1:
                log(f'noise {sig:g} it {it} loss {np.mean(hist[-100:]):.4e}')
        return hist

    @torch.no_grad()
    def val_error(idx):
        dr = {k: v[idx] for k, v in tr['drive'].items()}
        R, _, _ = rollout(model, dr, rho[idx, 0], W, DT)
        e = torch.linalg.vector_norm(R[:, -1] - rho[idx, -1], dim=-1) / torch.linalg.vector_norm(rho[idx, -1], dim=-1)
        e = torch.where(torch.isfinite(e), e, torch.full_like(e, 1e9))
        return float(e.median())
    allidx = torch.arange(C.shape[0], device=C.device)
    tune = {}
    if noise is None:
        nmed = int(med.max()) + 1
        tr_idx = allidx[med < nmed - val_media]; va_idx = allidx[med >= nmed - val_media]
        for sig in (1e-3, 3e-3, 1e-2):
            fit(tr_idx, tune_iters, sig)
            tune[sig] = val_error(va_idx)
            log(f'noise {sig:g}: validation rollout error over the window {tune[sig]:.4g}')
        noise = min(tune, key=tune.get)
    model.noise.fill_(noise)
    t0 = time.time()
    hist = fit(allidx, iters, noise)
    return dict(train_time_s=time.time() - t0, final_train_loss=float(np.mean(hist[-100:])), noise=noise,
                noise_tuning={str(k): v for k, v in tune.items()}, loss_curve=[float(np.mean(hist[i:i + 100])) for i in range(0, len(hist), 100)])


def grid_cond(drive, norm):
    c = node_cond(drive, norm)
    n = int(round(c.shape[1] ** 0.5))
    return c.transpose(1, 2).reshape(c.shape[0], NC, n, n).float()


def run_generic_fno(variant, seed, iters, device, data, odir, name, log):
    """GENERIC-FNO with the condition fields as input channels, trained with its own protocol (paper App. E)"""
    from neuralchart.baselines.generic_fno import GenericFNO, train_generic, rollout as gfno_rollout, n_params
    norm = data['meta']['norm']
    tr = data['train']
    B = tr['c'].shape[0]
    X = tr['c'].reshape(B, -1, 1, N, N).float()
    model = GenericFNO(n_fields=1, n_cond=NC).to(device)
    npar = n_params(model)
    log(f'{variant} {name}: params {npar} steps {iters} (GENERIC-FNO with condition channels)')
    t0 = time.time()
    info = train_generic(model, X, grid_cond(tr['drive'], norm), steps=iters, device=device, seed=seed, log=log)
    info['train_time_s'] = time.time() - t0
    torch.save(model.state_dict(), os.path.join(odir, name + '.pt'))
    res, snap = {}, {}
    for s in [k for k in TEST_SETS if k in data]:
        sd = data[s]; Y = sd['c']
        if sd['disc'] == 'cloud':
            res[s] = None; log(f'[{s}] N/A'); continue
        n = int(round(Y.shape[-1] ** 0.5))
        t1 = time.time()
        R = gfno_rollout(model, Y[:, 0].reshape(-1, 1, n, n).float(), Y.shape[1] - 1, grid_cond(sd['drive'], norm))
        R = R.reshape(R.shape[0], R.shape[1], -1)
        m = rollout_metrics(R, Y.cpu()); m['infer_time_s'] = time.time() - t1
        disc = data['discs'][sd['disc']]
        m['phys'] = dict(phys_metrics_rollout(R.to(device), sd['drive'], disc).result(), flux='min-norm')
        if s in ('drive_x8', 'temperature_x0.25', 'training_conditions'):
            Rd = R.to(device)
            snap[f'sig_{s}'] = local_sigma(Rd[0, 100], disc.minnorm_flux((Rd[:1, 101] - Rd[:1, 100]) / DT)[0],
                                           {k: v[0] for k, v in sd['drive'].items() if torch.is_tensor(v)}, disc).cpu().numpy().astype(np.float32)
        snap[f'R_{s}'] = R[:4, -1].numpy().astype(np.float32)
        res[s] = m
        log(f'[{s}] T {m["err_T"]:.4g} 10T {m["err_10T"]:.4g} blown {m["n_blown"]} neg {m["n_neg_traj"]} mass {m["mass_drift_max"]:.1e}')
    o = dict(variant=variant, model='generic_fno', seed=seed, params=npar, iters=iters, train_time_s=info['train_time_s'],
             train={k: v for k, v in info.items() if k != 'loss_curve'}, sets=res)
    json.dump(o, open(os.path.join(odir, name + '.json'), 'w'))
    np.savez_compressed(os.path.join(odir, name + '_snap.npz'), **snap)
    return o


HELM_T_IN, HELM_T_OUT = 10, 10


def helmfluid_model(device):
    """HelmFluid_2D_corr_potential with the settings of its scripts/Navier_Stokes_HelmFluid.sh and NC condition channels"""
    from types import SimpleNamespace
    from neuralchart.baselines.helmfluid import conditioned_helmfluid
    args = SimpleNamespace(groups=4, dataset_name='NS', in_dim=HELM_T_IN + NC, out_dim=1, d_model=64, padding='0,0')
    return conditioned_helmfluid(args, NC).to(device)


def helmfluid_history(u, s):
    """the T_in frames ending at step s (frames before t = 0 are copies of rho_0), (B, n, n, T_in)"""
    idx = torch.clamp(torch.arange(s - HELM_T_IN + 1, s + 1), min=0)
    return u[:, idx].permute(0, 2, 3, 1)


@torch.no_grad()
def helmfluid_rollout(model, Y, drive, norm, device):
    """autoregressive rollout from the initial state of Y (B, T+1, n^2) on the 32 x 32 grid; (B, T+1, n^2) on the CPU,
    NaN for a blown-up trajectory"""
    B = Y.shape[0]
    Y2 = Y.reshape(B, -1, N, N)
    h = helmfluid_history(Y2[:, :1], 0).to(device)
    c = grid_cond(drive, norm).permute(0, 2, 3, 1).to(device)
    outs = [Y2[:, 0].cpu()]
    for t in range(Y2.shape[1] - 1):
        im, _, _ = model(torch.cat([h, c], -1))
        bad = ~torch.isfinite(im).flatten(1).all(1) | (im.flatten(1).abs().amax(1) > 1e4)
        im = torch.where(bad[:, None, None, None], torch.full_like(im, float('nan')), im)
        outs.append(im[..., 0].cpu())
        h = torch.cat([h[..., 1:], im], -1)
    return torch.stack(outs, 1).reshape(B, Y2.shape[1], -1)


@torch.no_grad()
def eval_helmfluid(model, data, device, log=print):
    """metrics of HelmFluid on the 32 x 32 test sets (its padding and scales are set for the training grid)"""
    norm = data['meta']['norm']
    res, snap = {}, {}
    for s in TEST_SETS:
        if DISC_OF[s] != 'g32' or s not in data:
            res[s] = None; continue
        sd = data[s]; Y = sd['c']
        t1 = time.time()
        R = helmfluid_rollout(model, Y, sd['drive'], norm, device)
        m = rollout_metrics(R, Y.cpu())
        m['infer_time_s'] = time.time() - t1
        disc = data['discs'][sd['disc']]
        m['phys'] = dict(phys_metrics_rollout(R.double().to(Y.device), sd['drive'], disc).result(), flux='min-norm')
        snap[f'R_{s}'] = R[:4, -1].numpy().astype(np.float32)
        res[s] = m
        log(f'[{s}] T {m["err_T"]:.4g} 10T {m["err_10T"]:.4g} blown {m["n_blown"]} neg {m["n_neg_traj"]} mass {m["mass_drift_max"]:.1e}')
    return res, snap


def run_helmfluid(variant, seed, steps, out, window, data_dir=None):
    """HelmFluid with the defaults of its scripts/Navier_Stokes_HelmFluid.sh:
    T_in 10 (history padded with rho_0), T_out 10, batch 5, lr 5e-4, their Adam (wd 1e-4), StepLR gamma 0.5 with the
    period scaled to the step budget, relative-L2 LpLoss summed over the 10 steps.  Condition channels appended.
    Evaluated on the 32 x 32 sets only (its padding and scales are set for the training grid)."""
    from neuralchart.baselines.helmfluid import load_helmfluid
    check_gpu('cuda')
    _, LpLoss, Adam = load_helmfluid()
    torch.manual_seed(seed); np.random.seed(seed); torch.cuda.manual_seed(seed)
    dev = 'cuda'
    T_IN, T_OUT, BATCH, LR = HELM_T_IN, HELM_T_OUT, 5, 5e-4
    data = load(variant, 'cpu', sets=['train'] + [s for s in TEST_SETS if DISC_OF[s] == 'g32'], data_dir=data_dir)
    norm = add_norm(data['meta'], data)['norm']
    data['train'] = training_set(data, window, False)
    odir = os.path.join(out, variant); os.makedirs(odir, exist_ok=True)
    name = 'helmfluid'
    logf = open(os.path.join(odir, name + '.log'), 'w')

    def log(*x):
        s = ' '.join(str(y) for y in x); print(s, flush=True); logf.write(s + '\n'); logf.flush()
    model = helmfluid_model(dev)
    npar = sum(q.numel() for q in model.parameters())
    log(f'{variant} {name}: params {npar} steps {steps}')

    history = helmfluid_history
    tr = data['train']
    U = tr['c'].reshape(tr['c'].shape[0], -1, N, N)
    Cd = grid_cond(tr['drive'], norm).permute(0, 2, 3, 1)
    win = [(i, s) for i in range(U.shape[0]) for s in range(0, U.shape[1] - 1 - T_OUT + 1)]
    opt = Adam(model.parameters(), lr=LR, weight_decay=1e-4)
    step_size = max(1, round(steps * 100 / 501))
    sched = torch.optim.lr_scheduler.StepLR(opt, step_size=step_size, gamma=0.5)
    myloss = LpLoss(size_average=False)
    g = torch.Generator().manual_seed(seed)
    model.train(); t0 = time.time(); it = 0; hist = []
    while it < steps:
        perm = torch.randperm(len(win), generator=g)
        for j in range(0, len(perm) - BATCH + 1, BATCH):
            if it >= steps:
                break
            ii = [win[k] for k in perm[j:j + BATCH].tolist()]
            h = torch.stack([history(U[i:i + 1], s)[0] for i, s in ii]).to(dev)
            y = torch.stack([U[i, s + 1:s + 1 + T_OUT] for i, s in ii]).permute(0, 2, 3, 1).to(dev)
            c = torch.stack([Cd[i] for i, _ in ii]).to(dev)
            loss = 0
            for t in range(T_OUT):
                im, _, _ = model(torch.cat([h, c], -1))
                loss = loss + myloss(im.reshape(BATCH, -1), y[..., t:t + 1].reshape(BATCH, -1))
                h = torch.cat([h[..., 1:], im], -1)
            opt.zero_grad(); loss.backward(); opt.step(); sched.step(); it += 1
            hist.append(float(loss) / BATCH / T_OUT)
            if it % 200 == 0 or it == 1:
                log(f'it {it} loss/step {np.mean(hist[-200:]):.4e} lr {sched.get_last_lr()[0]:.2e} t {time.time() - t0:.0f}s')
    ttrain = time.time() - t0
    torch.save(model.state_dict(), os.path.join(odir, name + '.pt'))
    model.eval()
    res, snap = eval_helmfluid(model, data, dev, log)
    o = dict(variant=variant, model='helmfluid', seed=seed, params=npar, iters=steps, window=window, train_time_s=ttrain, step_size=step_size,
             final_loss=float(np.mean(hist[-200:])), sets=res)
    json.dump(o, open(os.path.join(odir, name + '.json'), 'w'))
    np.savez_compressed(os.path.join(odir, name + '_snap.npz'), **snap)
    return o


def run_persistence(variant, out, device='cpu', data_dir=None):
    """the no-change row u(t) = u(0)"""
    data = load(variant, device, data_dir=data_dir)
    res = {}
    for s in [k for k in TEST_SETS if k in data]:
        sd = data[s]; Y = sd['c']
        disc = data['discs'][sd['disc']]
        res[s] = rollout_metrics(Y[:, :1].expand_as(Y).cpu(), Y.cpu(), disc.g['dens'])
    os.makedirs(os.path.join(out, variant), exist_ok=True)
    o = dict(variant=variant, model='persistence', seed=0, params=0, iters=0, train_time_s=0.0, sets=res)
    json.dump(o, open(os.path.join(out, variant, 'persistence.json'), 'w'))
    return o


def run_train(variant, model_name, seed, iters, device, out, lr=1e-3, window=T_TRAIN_DATA, data_dir=None, kmin=4, kmax=20,
              fast_kw=None, noise=None):
    """train one model on the 32 x 32 training grid and evaluate it on every test set.  Writes <out>/<variant>/<model>
    .json, .pt and .log.  fast_kw: the options of neuralchart.fast (None: the plain integrator)."""
    from neuralchart import chart
    cfg = chart.chart_config('porous_media', model_name) if chart.is_chart('porous_media', model_name) else None
    oracle = cfg is not None and cfg.kappa != 'learned'
    check_gpu(device)
    torch.manual_seed(seed); np.random.seed(seed)
    data = load(variant, device, data_dir=data_dir)
    meta = add_norm(data['meta'], data)
    data['train'] = training_set(data, window)
    odir = os.path.join(out, variant); os.makedirs(odir, exist_ok=True)
    name = model_name
    logf = open(os.path.join(odir, name + '.log'), 'w')

    def log(*a):
        s = ' '.join(str(x) for x in a); print(s, flush=True); logf.write(s + '\n'); logf.flush()
    if model_name == 'generic_fno':
        return run_generic_fno(variant, seed, iters, device, data, odir, name, log)
    if model_name == 'persistence':
        return run_persistence(variant, out, device, data_dir)
    if oracle:
        oracle_drive(data)
    g32 = data['discs']['g32']
    model = build(model_name, meta, g32).to(device)
    if fast_kw is not None and fast.supports(model):
        fast.accelerate(model, **fast_kw)
    npar = sum(p.numel() for p in model.parameters() if p.requires_grad)
    log(f'{variant} {name}: params {npar} iters {iters} lr {lr} device {device}' + (f' fast {fast_kw}' if fast_kw else ''))
    info = dict(train_time_s=0.0)
    if model_name == 'bsms':
        info = fit_bsms(model, data, iters, seed, device, log, noise=noise)
        torch.save(model.state_dict(), os.path.join(odir, name + '.pt'))
    elif cfg is not None:
        info = dict(info, **fit_chart(model, data['train'], DT, iters, seed, log=log, batch=8, lr=lr, kmin=kmin, kmax=kmax))
        if cfg.train and iters > 0:
            torch.save(model.state_dict(), os.path.join(odir, name + '.pt'))
    elif iters > 0:
        info = train(model, data['train'], iters, seed, DT, batch=8, lr=lr, kmin=kmin, kmax=kmax, log=log)
        torch.save(model.state_dict(), os.path.join(odir, name + '.pt'))
    model.eval()
    st = getattr(getattr(model, '_fast', None), 'stepper', None)
    cap_train = st.cap_hits if st else None                  # substep-cap hits of the training integration
    log(f'LDB rule {LDB_RULE}')
    ev = eval_model(model_name, model, data, log, device)
    res = dict(benchmark=variant, model=model_name, config=dataclasses.asdict(cfg) if cfg else None, seed=seed, params=npar,
               iters=iters, lr=lr, window=window, kmin=kmin, kmax=kmax, **{k: v for k, v in info.items() if k != 'loss_curve'},
               fast=dict(fast_kw, capped_DT=cap_train) if fast_kw else None, sets=ev)
    if cfg is not None:
        res['drive_scale'] = float(model.drive.s_A.detach()); res['log_omega'] = float(model.kinetics.core.detach())
    json.dump(res, open(os.path.join(odir, name + '.json'), 'w'))
    return res


def run_evaluate(variant, model_name, checkpoint, device, out, data_dir=None, fast_kw=None):
    """evaluate a trained model (checkpoint: a state dict of the model) on every test set; writes <out>/<variant>/
    evaluated/<model>.json"""
    from neuralchart import chart
    check_gpu(device)
    data = load(variant, device, data_dir=data_dir)
    meta = add_norm(data['meta'], data)
    if model_name == 'helmfluid':
        model = helmfluid_model(device)
        model.load_state_dict(torch.load(checkpoint, map_location=device)); model.eval()
        ev, _ = eval_helmfluid(model, data, device)
        odir = os.path.join(out, variant, 'evaluated'); os.makedirs(odir, exist_ok=True)
        res = dict(benchmark=variant, model=model_name, params=sum(p.numel() for p in model.parameters()), checkpoint=checkpoint, sets=ev)
        json.dump(res, open(os.path.join(odir, model_name + '.json'), 'w'))
        return res
    if model_name in ('oracle', 'oracle-no-crowding'):
        oracle_drive(data)
    model = build(model_name, meta, data['discs']['g32']).to(device)
    if checkpoint:
        model.load_state_dict(torch.load(checkpoint, map_location=device))
    if fast_kw is not None and fast.supports(model):
        fast.accelerate(model, **fast_kw)
    model.eval()
    odir = os.path.join(out, variant, 'evaluated'); os.makedirs(odir, exist_ok=True)
    ev = eval_model(model_name, model, data, print, device)
    cfg = getattr(model, 'config', None)
    res = dict(benchmark=variant, model=model_name, config=dataclasses.asdict(cfg) if cfg else None,
               params=sum(p.numel() for p in model.parameters() if p.requires_grad), checkpoint=checkpoint, fast=fast_kw, sets=ev)
    json.dump(res, open(os.path.join(odir, model_name + '.json'), 'w'))
    return res
