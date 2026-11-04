"""The one-step baselines train with the protocol of their papers; every other model with neuralchart.fit.train."""
import time

import numpy as np
import torch

from neuralchart.fit import rollout
from . import config as C


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
    cond = model.cond_fields(ctx['z'], ctx.get('U'), n, ctx.get('dsite')).cpu()
    t0 = time.time()
    info = train_generic(model.net, X, cond, steps=iters, device=device, seed=seed, log=log)
    info.pop('loss_curve', None)
    return dict(info, train_time_s=time.time() - t0)
