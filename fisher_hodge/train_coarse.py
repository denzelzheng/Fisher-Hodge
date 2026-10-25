"""E4(c): train tcme / fisher_hodge / flux_gnn / gnn_ode FROM SCRATCH on a coarse grid (16x16 or 8x8), with the same
protocol as train_tcme.py (rollout loss K in [4,8], Adam 1e-3 cosine, batch 8, grad clip 1).  The training set is the same
64 trajectories as the 32x32 runs, 2x2- (or 4x4-) block coarse-grained: rho -> block mean, U/psin -> block mean,
dpsi -> flux-consistent coarse Peclet number (coarse_test.lump_fields, applied once per factor 2).
  python train_coarse.py --model tcme --n 16 --seed 0 --iters 2000      -> runs/rg/S2_tcme_n16_s0.{pt,json,log}"""
import argparse, json, time, os, sys
import numpy as np
import torch
from torch.utils.checkpoint import checkpoint
from common import *
import common, models, models2
from models2 import FluxGNN
from models import FisherHodge, GNNODE
from models_tcme import TCME
from coarse_test import lump_fields, set_rate_scale
from train_tcme import jclean

MODELS = dict(tcme=TCME, fisher_hodge=FisherHodge, flux_gnn=FluxGNN, gnn_ode=GNNODE)
FIELDS = ['U', 'dpsi', 'psin']
SETS = ['test_id', 'test_oodpot', 'test_oodinit']
dev = 'cuda'
ODIR = os.path.join(RUNS, 'rg')


def coarse_set(sd, n):
    """block-coarse-grain a data split from 32x32 to n x n (n = 16 or 8)."""
    F = {k: sd[k].double() for k in FIELDS}
    rho = sd['rho'].double()
    m = N // n
    while rho.shape[-1] > n:
        F = lump_fields(F); rho = coarsen(rho)
    out = {k: v.float() for k, v in F.items()}
    out['rho'] = rho.float()
    return out


def fields(setd, idx=None):
    return {k: (setd[k] if idx is None else setd[k][idx]).to(dev) for k in FIELDS}


@torch.no_grad()
def rollout(model, F, rho0, nsteps):
    ctx = model.prepare(F)
    r = rho0.clone(); out = [r.cpu()]
    for k in range(nsteps):
        r, _ = model.advance(r, ctx, DT)
        bad = (~torch.isfinite(r)).any(-1).any(-1) | (r.abs().amax((-1, -2)) > 1e4)
        if bad.any():
            r = torch.where(bad[:, None, None], torch.full_like(r, float('nan')), r)
        out.append(r.cpu())
    return torch.stack(out, 1)


if __name__ == '__main__':
    p = argparse.ArgumentParser()
    p.add_argument('--model', default='tcme'); p.add_argument('--seed', type=int, default=0)
    p.add_argument('--n', type=int, default=16)
    p.add_argument('--iters', type=int, default=2000); p.add_argument('--batch', type=int, default=8)
    p.add_argument('--lr', type=float, default=1e-3); p.add_argument('--kmin', type=int, default=4); p.add_argument('--kmax', type=int, default=8)
    p.add_argument('--smoke', action='store_true')
    args = p.parse_args()
    os.makedirs(ODIR, exist_ok=True)
    if args.smoke:
        args.iters = 10
    set_rate_scale(args.n)                       # D n^2 for fisher_hodge / flux_gnn / gnn_ode (tcme reads n from the shape)
    torch.manual_seed(args.seed); np.random.seed(args.seed)
    rng = np.random.default_rng(10_000 + args.seed)
    name = f'S2_{args.model}_n{args.n}_s{args.seed}' + ('_smoke' if args.smoke else '')
    logf = open(os.path.join(ODIR, name + '.log'), 'w')

    def log(*a):
        s = ' '.join(str(x) for x in a); print(s, flush=True); logf.write(s + '\n'); logf.flush()

    data = torch.load(os.path.join(RUNS, 'data', 'S2.pt'))
    model = MODELS[args.model]('S2').to(dev)
    nparam = sum(q.numel() for q in model.parameters())
    log(f'{name}: params={nparam}  args={vars(args)}  S_SCALE={models.S_SCALE}')
    tr = coarse_set(data['train'], args.n); RHO = tr['rho'].to(dev); ntr = RHO.shape[0]
    opt = torch.optim.Adam(model.parameters(), lr=args.lr)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, args.iters, eta_min=1e-5)
    t0 = time.time(); hist = []; nskip = 0
    model.train()
    for it in range(args.iters):
        idx = torch.tensor(rng.choice(ntr, min(args.batch, ntr), replace=False))
        K = int(rng.integers(args.kmin, args.kmax + 1))
        st = torch.tensor(rng.integers(0, T_TRAIN - K + 1, size=len(idx)), device=dev)
        F = fields(tr, idx); ctx = model.prepare(F)
        idx = idx.to(dev); r = RHO[idx, st]; loss = 0.0
        for k in range(1, K + 1):
            if args.model in ('flux_gnn', 'gnn_ode'):
                r = checkpoint(lambda rr: model.advance(rr, ctx, DT)[0], r, use_reentrant=False)
            else:
                r, _ = model.advance(r, ctx, DT)
            y = RHO[idx, st + k]
            loss = loss + (((r - y) ** 2).sum((-1, -2)) / (y ** 2).sum((-1, -2))).mean()
        loss = loss / K
        if not torch.isfinite(loss):
            nskip += 1; opt.zero_grad(set_to_none=True); sched.step(); continue
        opt.zero_grad(set_to_none=True); loss.backward()
        gn = torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        opt.step(); sched.step(); hist.append(float(loss.detach()))
        if it % 100 == 0 or it == args.iters - 1:
            log(f'it {it:5d} loss {np.mean(hist[-100:]):.4e} gnorm {float(gn):.2e} K {K} skipped {nskip} t {time.time() - t0:.0f}s')
    train_time = time.time() - t0
    torch.save(model.state_dict(), os.path.join(ODIR, name + '.pt'))
    model.eval()
    res = dict(name=name, model=args.model, n=args.n, seed=args.seed, params=nparam, args=vars(args), train_time_s=train_time,
               final_train_loss=float(np.mean(hist[-100:])) if hist else None, train_nskip=nskip, sets={})
    for setname in SETS:
        sd = coarse_set(data[setname], args.n); Y = sd['rho'].double()
        R = rollout(model, fields(sd), sd['rho'][:, 0].to(dev), Y.shape[1] - 1).double()
        B, T1 = Y.shape[:2]
        err = (torch.linalg.vector_norm((R - Y).reshape(B, T1, -1), dim=-1) / torch.linalg.vector_norm(Y.reshape(B, T1, -1), dim=-1)).numpy()
        err = np.where(np.isfinite(err), err, np.inf)
        Rn = R.numpy()
        S = dict(relerr_median=np.median(err, 0), relerr_10T_per_traj=err[:, -1],
                 mass_err_max=float(np.nanmax(np.abs(Rn.mean((2, 3)) - 1.0))), min_rho=float(np.nanmin(Rn)),
                 n_traj_with_negative=int((Rn < 0).any((1, 2, 3)).sum()), n_blown=int((~np.isfinite(Rn[:, -1])).any((1, 2)).sum()))
        res['sets'][setname] = S
        log(f'[{setname}] relerr@T,2T,5T,10T = ' + ', '.join(f'{np.median(err[:, t]):.3e}' for t in [20, 40, 100, 200]) +
            f' | mass {S["mass_err_max"]:.1e} | min rho {S["min_rho"]:.2e} | neg traj {S["n_traj_with_negative"]}')
    json.dump(jclean(res), open(os.path.join(ODIR, name + '.json'), 'w'))
    log('done', name, f'train {train_time:.0f}s')
