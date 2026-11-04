"""BSMS-GNN, GENERIC-FNO and HelmFluid train with the protocols of their papers; every other model with neuralchart.fit.train."""
import time

import numpy as np
import torch

from neuralchart.fit import rollout
from .data import DT, N
from .models import HELM_T_IN, HELM_T_OUT, grid_cond


def fit_bsms(model, tr, iters, seed, log, noise=None, batch=8, val_media=8, tune_iters=1500):
    """the MeshGraphNets training protocol: one-step loss on normalised targets, Gaussian input noise with the target corrected for it,
    Adam with the learning rate decaying exponentially from 1e-4 to 1e-6.  noise=None: the noise std is chosen once
    from {1e-3, 3e-3, 1e-2} x the data std on a validation split of the training media (the last `val_media` media),
    by the error of a rollout over the training window; the final model is then trained on all training media."""
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
                noise_tuning={str(k): v for k, v in tune.items()})


def fit_generic_fno(model, tr, iters, seed, device, log):
    """GENERIC-FNO with the condition fields as input channels, trained with its own protocol (paper App. E)"""
    from neuralchart.baselines.generic_fno import train_generic
    B = tr['c'].shape[0]
    X = tr['c'].reshape(B, -1, 1, N, N).float()
    t0 = time.time()
    info = train_generic(model.net, X, grid_cond(tr['drive'], model.norm), steps=iters, device=device, seed=seed, log=log)
    info['train_time_s'] = time.time() - t0
    info.pop('loss_curve', None)
    info['final_train_loss'] = info.pop('final_loss')
    return info


def fit_helmfluid(model, tr, steps, seed, device, log):
    """HelmFluid with the defaults of its scripts/Navier_Stokes_HelmFluid.sh: T_in 10 (history padded with rho_0), T_out 10, batch 5, lr 5e-4,
    their Adam (wd 1e-4), StepLR gamma 0.5 with the period scaled to the step budget, relative-L2 LpLoss summed over the 10 steps.  Condition
    channels appended."""
    from neuralchart.baselines.helmfluid import load_helmfluid
    _, LpLoss, Adam = load_helmfluid()
    BATCH, LR = 5, 5e-4
    net = model.net
    U = tr['c'].reshape(tr['c'].shape[0], -1, N, N)
    Cd = grid_cond(tr['drive'], model.norm).permute(0, 2, 3, 1)

    def history(u, s):
        """the T_in frames ending at step s (frames before t = 0 are copies of rho_0), (B, n, n, T_in)"""
        idx = torch.clamp(torch.arange(s - HELM_T_IN + 1, s + 1), min=0)
        return u[:, idx].permute(0, 2, 3, 1)
    win = [(i, s) for i in range(U.shape[0]) for s in range(0, U.shape[1] - 1 - HELM_T_OUT + 1)]
    opt = Adam(net.parameters(), lr=LR, weight_decay=1e-4)
    step_size = max(1, round(steps * 100 / 501))
    sched = torch.optim.lr_scheduler.StepLR(opt, step_size=step_size, gamma=0.5)
    myloss = LpLoss(size_average=False)
    g = torch.Generator().manual_seed(seed)
    net.train(); t0 = time.time(); it = 0; hist = []
    while it < steps:
        perm = torch.randperm(len(win), generator=g)
        for j in range(0, len(perm) - BATCH + 1, BATCH):
            if it >= steps:
                break
            ii = [win[k] for k in perm[j:j + BATCH].tolist()]
            h = torch.stack([history(U[i:i + 1], s)[0] for i, s in ii]).to(device)
            y = torch.stack([U[i, s + 1:s + 1 + HELM_T_OUT] for i, s in ii]).permute(0, 2, 3, 1).to(device)
            c = torch.stack([Cd[i] for i, _ in ii]).to(device)
            loss = 0
            for t in range(HELM_T_OUT):
                im, _, _ = net(torch.cat([h, c], -1))
                loss = loss + myloss(im.reshape(BATCH, -1), y[..., t:t + 1].reshape(BATCH, -1))
                h = torch.cat([h[..., 1:], im], -1)
            opt.zero_grad(); loss.backward(); opt.step(); sched.step(); it += 1
            hist.append(float(loss) / BATCH / HELM_T_OUT)
            if it % 200 == 0 or it == 1:
                log(f'it {it} loss/step {np.mean(hist[-200:]):.4e} lr {sched.get_last_lr()[0]:.2e} t {time.time() - t0:.0f}s')
    return dict(train_time_s=time.time() - t0, step_size=step_size, final_train_loss=float(np.mean(hist[-200:])))
