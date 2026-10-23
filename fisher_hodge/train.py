"""Train + evaluate one (system, model, seed).
python3 train.py --sys S1 --model fisher_hodge --seed 0 [--iters 4000] [--smoke]"""
import argparse, json, time, os, math
import numpy as np
import torch
from torch.utils.checkpoint import checkpoint
from common import *
from models import MODELS, NSUB

p = argparse.ArgumentParser()
p.add_argument('--sys', default='S1'); p.add_argument('--model', default='fisher_hodge')
p.add_argument('--seed', type=int, default=0); p.add_argument('--iters', type=int, default=4000)
p.add_argument('--batch', type=int, default=8); p.add_argument('--lr', type=float, default=1e-3)
p.add_argument('--kmin', type=int, default=4); p.add_argument('--kmax', type=int, default=8)
p.add_argument('--smoke', action='store_true'); p.add_argument('--tag', default='')
args = p.parse_args()
if args.smoke:
    args.iters = 20
dev = 'cuda'
torch.manual_seed(args.seed); np.random.seed(args.seed)
rng = np.random.default_rng(10_000 + args.seed)
name = f'{args.sys}_{args.model}_s{args.seed}{args.tag}' + ('_smoke' if args.smoke else '')
odir = os.path.join(RUNS, 'smoke' if args.smoke else 'main'); os.makedirs(odir, exist_ok=True)
logf = open(os.path.join(odir, name + '.log'), 'w')


def log(*a):
    s = ' '.join(str(x) for x in a); print(s, flush=True); logf.write(s + '\n'); logf.flush()


data = torch.load(os.path.join(RUNS, 'data', f'{args.sys}.pt'))
FIELDS = ['U', 'dpsi', 'psin']


def fields(setd, idx=None):
    return {k: (setd[k] if idx is None else setd[k][idx]).to(dev) for k in FIELDS}


model = MODELS[args.model](args.sys).to(dev)
nparam = sum(p.numel() for p in model.parameters())
log(f'{name}: params={nparam}  args={vars(args)}')
opt = torch.optim.Adam(model.parameters(), lr=args.lr)
sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, args.iters, eta_min=1e-5)
tr = data['train']; RHO = tr['rho'].to(dev)
ntr = RHO.shape[0]
use_ckpt = args.model == 'gnn_ode'

# ------------------------------------------------------------------ training
t0 = time.time(); hist = []; nskip = 0; subs = []
model.train()
for it in range(args.iters):
    idx = torch.tensor(rng.choice(ntr, args.batch, replace=False), device=dev)
    K = int(rng.integers(args.kmin, args.kmax + 1))
    st = torch.tensor(rng.integers(0, T_TRAIN - K + 1, size=args.batch), device=dev)
    F = fields(tr, idx.cpu())
    ctx = model.prepare(F)
    r = RHO[idx, st]
    loss = 0.0; ns_it = 0
    for k in range(1, K + 1):
        if use_ckpt:
            r = checkpoint(lambda rr: model.advance(rr, ctx, DT)[0], r, use_reentrant=False); ns = NSUB
        else:
            r, ns = model.advance(r, ctx, DT)
        ns_it += ns
        y = RHO[idx, st + k]
        loss = loss + (((r - y) ** 2).sum((-1, -2)) / (y ** 2).sum((-1, -2))).mean()
    loss = loss / K
    subs.append(ns_it / K)
    if not torch.isfinite(loss):
        nskip += 1; opt.zero_grad(set_to_none=True); sched.step(); continue
    opt.zero_grad(set_to_none=True); loss.backward()
    gn = torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
    opt.step(); sched.step()
    hist.append(float(loss))
    if it % 100 == 0 or it == args.iters - 1:
        log(f'it {it:5d} loss {np.mean(hist[-100:]):.4e} gnorm {float(gn):.2e} K {K} substeps/DT {np.mean(subs[-100:]):.1f} '
            f'skipped {nskip} t {time.time() - t0:.0f}s')
train_time = time.time() - t0
torch.save(model.state_dict(), os.path.join(odir, name + '.pt'))

# ------------------------------------------------------------------ evaluation
model.eval()


@torch.no_grad()
def rollout(F, rho0, nsteps):
    ctx = model.prepare(F)
    r = rho0.clone(); out = [r.cpu()]; nsub = []
    blown = torch.zeros(r.shape[0], dtype=torch.bool, device=dev)
    for k in range(nsteps):
        r, ns = model.advance(r, ctx, DT)
        nsub.append(ns)
        bad = (~torch.isfinite(r)).any(-1).any(-1) | (r.abs().amax((-1, -2)) > 1e4)
        blown |= bad
        if bad.any():
            r = torch.where(bad[:, None, None], torch.full_like(r, float('nan')), r)
        out.append(r.cpu())
    return torch.stack(out, 1), nsub, ctx


def jnum(x):
    if isinstance(x, (list, tuple)):
        return [jnum(v) for v in x]
    if isinstance(x, (np.ndarray,)):
        return jnum(x.tolist())
    if isinstance(x, float) or isinstance(x, np.floating):
        return float(x) if np.isfinite(x) else None
    return x


def frontspeed(A, nseeds, win):
    t = np.arange(len(A)) * DT
    Rr = np.sqrt(A / (np.pi * nseeds))
    if win.sum() < 3:
        return float('nan')
    return float(np.polyfit(t[win], Rr[win], 1)[0])


results = dict(name=name, sys=args.sys, model=args.model, seed=args.seed, params=nparam, args=vars(args),
               train_time_s=train_time, final_train_loss=float(np.mean(hist[-100:])) if hist else None,
               train_nskip=nskip, train_mean_substeps=float(np.mean(subs)), loss_hist=hist[::20], sets={})
snap = {}
for setname in ['test_id', 'test_oodpot', 'test_oodinit']:
    sd = data[setname]; Y = sd['rho'].numpy().astype(np.float64)
    F = fields(sd)
    torch.cuda.synchronize(); te = time.time()
    R, nsub, ctx = rollout(F, sd['rho'][:, 0].to(dev), Y.shape[1] - 1)
    torch.cuda.synchronize(); tinf = time.time() - te
    R = R.numpy().astype(np.float64)
    B, T1 = Y.shape[:2]
    err = np.linalg.norm((R - Y).reshape(B, T1, -1), axis=-1) / np.linalg.norm(Y.reshape(B, T1, -1), axis=-1)
    errinf = np.where(np.isfinite(err), err, np.inf)
    massR, massY = R.mean((2, 3)), Y.mean((2, 3))
    massE = np.abs(massR - massY) / massY if args.sys == 'S3' else np.abs(massR - 1.0)
    minR = R.min((2, 3)); negf = (R < 0).mean((2, 3))
    blown_t = ~np.isfinite(R).all((2, 3))
    S = dict(
        infer_time_s=tinf, substeps_per_DT_mean=float(np.mean(nsub)), substeps_per_DT_max=int(np.max(nsub)),
        n_traj=B, n_blown_end=int(blown_t[:, -1].sum()), first_blow_step=[int(np.argmax(b)) if b.any() else None for b in blown_t],
        relerr_median=np.median(errinf, 0), relerr_mean_finite=np.array([np.mean(e[np.isfinite(e)]) if np.isfinite(e).any() else np.nan for e in err.T]),
        mass_err_median=np.nanmedian(np.where(np.isfinite(massE), massE, np.nan), 0) if np.isfinite(massE).any() else None,
        mass_err_max=np.array([np.max(m[np.isfinite(m)]) if np.isfinite(m).any() else np.nan for m in massE.T]),
        min_rho_min=np.array([np.min(m[np.isfinite(m)]) if np.isfinite(m).any() else np.nan for m in minR.T]),
        neg_frac_mean=np.array([np.mean(m[np.isfinite(m)]) if np.isfinite(m).any() else np.nan for m in negf.T]),
        n_traj_with_negative=int(((R < 0).any((2, 3))).any(1).sum()),
        relerr_per_traj_at={str(t): errinf[:, t] for t in [20, 40, 100, 200]},
    )
    if args.sys == 'S3':
        S['overshoot_max'] = float(np.nanmax(R) - 1.0) if np.isfinite(R).any() else None
        sp_err, sps = [], []
        for b in range(B):
            AY, AR = (Y[b] > 0.5).mean((1, 2)), (R[b] > 0.5).mean((1, 2))
            win = (AY >= 0.05) & (AY <= 0.5)
            vY = frontspeed(AY, int(sd['nseeds'][b]), win); vR = frontspeed(AR, int(sd['nseeds'][b]), win)
            sps.append((vY, vR)); sp_err.append(abs(vR - vY) / abs(vY) if np.isfinite(vR) else np.inf)
        S['front_speed_true_model'] = sps; S['front_speed_relerr_median'] = float(np.median(sp_err))
        S['front_speed_relerr_per_traj'] = sp_err
    if args.sys == 'S1':
        Up = sd['U'].numpy()[:, None]
        def Ftrue(X):
            with np.errstate(all='ignore'):
                f = np.where(X > 0, X * np.log(np.where(X > 0, X, 1.0)), np.nan) + Up * X
            return f.mean((2, 3))           # (F/D) per unit area
        FY, FR = Ftrue(Y), Ftrue(R)
        mono = lambda Fv: np.array([bool(np.all(np.isfinite(f)) and np.all(np.diff(f) <= 1e-6 * (1 + np.abs(f[:-1])))) for f in Fv])
        S['true_F_monotone_frac_truth'] = float(mono(FY).mean())
        S['true_F_monotone_frac_model'] = float(mono(FR).mean())
        S['true_F_end_relerr_median'] = float(np.median(np.where(np.isfinite(FR[:, -1]), np.abs(FR[:, -1] - FY[:, -1]) / np.abs(FY[:, -1]), np.inf)))
        if hasattr(model, 'lyapunov'):
            with torch.no_grad():
                L = np.stack([model.lyapunov(torch.tensor(R[:, t], device=dev, dtype=torch.float32), F).cpu().numpy() for t in range(T1)], 1)
            S['learned_lyapunov_monotone_frac'] = float(mono(L).mean())
            S['learned_lyapunov_max_increase'] = float(np.nanmax(np.diff(L, axis=1))) if np.isfinite(L).any() else None
    if args.sys == 'S2' and hasattr(model, 'flux_q'):
        with torch.no_grad():
            rend = torch.tensor(R[:, -1], device=dev, dtype=torch.float32)
            fin = torch.isfinite(rend).all(-1).all(-1)
            if args.model == 'fisher_hodge':
                rend = torch.where(fin[:, None, None], rend, torch.ones_like(rend))
            q = model.flux_q(rend, ctx)
            Jm = (-(D / HC) * q).cpu().numpy()
        Jt = sd['J_end'].numpy()
        cm = curl_f(torch.tensor(Jm)).numpy(); ct = curl_f(torch.tensor(Jt)).numpy()
        ce = np.linalg.norm((cm - ct).reshape(B, -1), axis=1) / np.linalg.norm(ct.reshape(B, -1), axis=1)
        ce = np.where(fin.cpu().numpy() & np.isfinite(ce), ce, np.inf)
        S['curl_flux_end_relerr_median'] = float(np.median(ce)); S['curl_flux_end_relerr_per_traj'] = ce
        S['circulation_abs_true_mean'] = float(np.abs(ct).mean())
    results['sets'][setname] = S
    snap[setname] = R[:, [0, 20, 40, 100, 200]].astype(np.float32)
    log(f'[{setname}] relerr@T,2T,5T,10T median = ' + ', '.join(f'{np.median(errinf[:, t]):.3e}' for t in [20, 40, 100, 200])
        + f' | blown {S["n_blown_end"]}/{B} | max mass err {np.nanmax(S["mass_err_max"]) if np.isfinite(S["mass_err_max"]).any() else float("nan"):.2e}'
        + f' | min rho {np.nanmin(S["min_rho_min"]) if np.isfinite(S["min_rho_min"]).any() else float("nan"):.2e} | substeps {S["substeps_per_DT_mean"]:.1f} | infer {tinf:.1f}s')
results = jnum(json.loads(json.dumps(results, default=lambda o: o.tolist() if hasattr(o, 'tolist') else str(o))))
with open(os.path.join(odir, name + '.json'), 'w') as f:
    json.dump(results, f)
np.savez_compressed(os.path.join(odir, name + '_snap.npz'), **snap)
log('done', name)
