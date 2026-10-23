"""Stress-test training / evaluation (superset of train.py).  Outputs to runs/stress/main/.
python3 train2.py --sys S4 --model fisher_hodge --seed 0 [--iters 4000] [--warmup 100] [--noise 0.05] [--tag _n05]
Models: gnn_ode euclid_gf fisher_hodge (models.py) + flux_gnn wgf_rev jko_amort jko_inst fisher_hodge_fr fisher_hodge_fr_u (models2.py).
Systems: S1 S2 S3 (gen_data.py) + S4 S5 S6 (gen_data2.py).  --noise s: multiplicative Gaussian noise rho*(1+s*xi) on the
*training* snapshots only (clamped at 1e-4); test data clean."""
import argparse, json, time, os, math
import numpy as np
import torch
from torch.utils.checkpoint import checkpoint
from common import *
import models2
from models2 import MODELS2 as MODELS
from models import NSUB

p = argparse.ArgumentParser()
p.add_argument('--sys', default='S1'); p.add_argument('--model', default='fisher_hodge')
p.add_argument('--seed', type=int, default=0); p.add_argument('--iters', type=int, default=4000)
p.add_argument('--warmup', type=int, default=100)
p.add_argument('--batch', type=int, default=8); p.add_argument('--lr', type=float, default=1e-3)
p.add_argument('--kmin', type=int, default=4); p.add_argument('--kmax', type=int, default=8)
p.add_argument('--noise', type=float, default=0.0)
p.add_argument('--plateau_window', type=int, default=0, help='>0: stop when the mean training loss over the last W steps improves by less than --plateau_tol relative to the W steps before')
p.add_argument('--plateau_tol', type=float, default=0.05)
p.add_argument('--accum', type=int, default=1, help='gradient accumulation: split the batch into this many micro-batches (same optimisation, 1/accum memory)')
p.add_argument('--smoke', action='store_true'); p.add_argument('--tag', default='')
p.add_argument('--outdir', default=os.path.join(RUNS, 'stress', 'main'))
args = p.parse_args()
if args.smoke:
    args.iters = 20
dev = 'cuda'
torch.manual_seed(args.seed); np.random.seed(args.seed)
rng = np.random.default_rng(10_000 + args.seed)
name = f'{args.sys}_{args.model}_s{args.seed}{args.tag}' + ('_smoke' if args.smoke else '')
odir = args.outdir if not args.smoke else os.path.join(RUNS, 'stress', 'smoke'); os.makedirs(odir, exist_ok=True)
logf = open(os.path.join(odir, name + '.log'), 'w')


def log(*a):
    s = ' '.join(str(x) for x in a); print(s, flush=True); logf.write(s + '\n'); logf.flush()


data = torch.load(os.path.join(RUNS, 'data', f'{args.sys}.pt'))
FIELDS = ['U', 'dpsi', 'psin']
SYSCLASS = {'S1': 'S1', 'S2': 'S2', 'S3': 'S3', 'S3b': 'S3', 'S4': 'S1', 'S5': 'S1', 'S6': 'S1'}[args.sys]   # model structure class


def fields(setd, idx=None):
    return {k: (setd[k] if idx is None else setd[k][idx]).to(dev) for k in FIELDS}


model = MODELS[args.model](args.sys if args.sys in ('S1', 'S2', 'S3', 'S5') else SYSCLASS).to(dev)
if args.sys == 'S5':      # rho_scale only; structure class is S1
    model.sys = 'S1'
nparam = sum(p.numel() for p in model.parameters())
log(f'{name}: params={nparam}  args={vars(args)}')
tr = data['train']; RHO = tr['rho'].to(dev)
if args.noise > 0:
    g = torch.Generator(device=dev).manual_seed(777 + args.seed)
    RHO = (RHO * (1 + args.noise * torch.randn(RHO.shape, generator=g, device=dev))).clamp_min(1e-4)
    log(f'noise {args.noise}: train rho min {RHO.min():.2e} max {RHO.max():.3f}')
ntr = RHO.shape[0]
use_ckpt = args.model in ('gnn_ode', 'flux_gnn')   # (checkpointing does not reduce memory for the FH family: double-backward graphs of mu = dF/drho dominate)
hist = []; nskip = 0; subs = []; t0 = time.time(); early_stop = None

# ------------------------------------------------------------------ training
if args.model == 'jko_inst':
    train_time = 0.0
elif args.model == 'jko_amort':
    # precompute JKO velocity potentials for all consecutive training pairs
    PSI = torch.zeros_like(RHO[:, 1:])
    for b0 in range(0, ntr, 8):
        for k in range(T_TRAIN):
            PSI[b0:b0 + 8, k] = models2.solve_jko_psi(RHO[b0:b0 + 8, k], RHO[b0:b0 + 8, k + 1], DT)
    log(f'psi precomputed in {time.time() - t0:.0f}s')
    opt = torch.optim.Adam(model.parameters(), lr=args.lr)
    sched = torch.optim.lr_scheduler.LambdaLR(opt, lambda it: min(1.0, (it + 1) / max(1, args.warmup)) * (0.01 + 0.99 * 0.5 * (1 + math.cos(math.pi * min(1.0, it / args.iters)))))
    model.train()
    for it in range(args.iters):
        idx = torch.tensor(rng.choice(ntr, 16, replace=False), device=dev)
        st = torch.tensor(rng.integers(0, T_TRAIN, size=16), device=dev)
        F = fields(tr, idx.cpu())
        loss = model.jko_loss(RHO[idx, st + 1], PSI[idx, st], F)
        if not torch.isfinite(loss):
            nskip += 1; opt.zero_grad(set_to_none=True); sched.step(); continue
        opt.zero_grad(set_to_none=True); loss.backward()
        gn = torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0); opt.step(); sched.step(); hist.append(float(loss))
        if it % 100 == 0 or it == args.iters - 1:
            log(f'it {it:5d} jko-loss {np.mean(hist[-100:]):.4e} beta {model.beta().item():.4f} gnorm {float(gn):.2e} skipped {nskip} t {time.time() - t0:.0f}s')
    train_time = time.time() - t0
    torch.save(model.state_dict(), os.path.join(odir, name + '.pt'))
else:
    opt = torch.optim.Adam(model.parameters(), lr=args.lr)
    sched = torch.optim.lr_scheduler.LambdaLR(opt, lambda it: min(1.0, (it + 1) / max(1, args.warmup)) * (0.01 + 0.99 * 0.5 * (1 + math.cos(math.pi * min(1.0, it / args.iters)))))
    model.train()
    for it in range(args.iters):
        idx = torch.tensor(rng.choice(ntr, args.batch, replace=False), device=dev)
        K = int(rng.integers(args.kmin, args.kmax + 1))
        st = torch.tensor(rng.integers(0, T_TRAIN - K + 1, size=args.batch), device=dev)
        opt.zero_grad(set_to_none=True)
        loss_tot = 0.0; ns_it = 0; bad = False
        for ib in range(args.accum):                       # micro-batches (gradient accumulation)
            sl = slice(ib * args.batch // args.accum, (ib + 1) * args.batch // args.accum)
            idx_b, st_b = idx[sl], st[sl]
            F = fields(tr, idx_b.cpu())
            ctx = model.prepare(F)
            r = RHO[idx_b, st_b]
            loss = 0.0
            for k in range(1, K + 1):
                if use_ckpt:
                    r = checkpoint(lambda rr: model.advance(rr, ctx, DT)[0], r, use_reentrant=False); ns = NSUB
                else:
                    r, ns = model.advance(r, ctx, DT)
                ns_it += ns
                y = RHO[idx_b, st_b + k]
                loss = loss + (((r - y) ** 2).sum((-1, -2)) / (y ** 2).sum((-1, -2))).mean()
            loss = loss / K / args.accum
            if not torch.isfinite(loss):
                bad = True; break
            loss.backward()
            loss_tot += float(loss)
        subs.append(ns_it / K / args.accum)
        if bad:
            nskip += 1; opt.zero_grad(set_to_none=True); sched.step(); continue
        gn = torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        opt.step(); sched.step()
        loss = loss_tot
        hist.append(float(loss))
        if it % 100 == 0 or it == args.iters - 1:
            log(f'it {it:5d} loss {np.mean(hist[-100:]):.4e} gnorm {float(gn):.2e} lr {sched.get_last_lr()[0]:.2e} K {K} substeps/DT {np.mean(subs[-100:]):.1f} '
                f'skipped {nskip} t {time.time() - t0:.0f}s')
        W = args.plateau_window
        if W > 0 and (it + 1) % W == 0 and len(hist) >= 2 * W:
            cur, prev = np.mean(hist[-W:]), np.mean(hist[-2 * W:-W])
            log(f'plateau check at it {it + 1}: mean loss last {W} = {cur:.4e}, previous {W} = {prev:.4e}, rel. improvement {(prev - cur) / prev:.3f}')
            if (prev - cur) / prev < args.plateau_tol:
                log(f'EARLY STOP (plateau) at it {it + 1}'); early_stop = it + 1; break
    train_time = time.time() - t0
    torch.save(model.state_dict(), os.path.join(odir, name + '.pt'))
log(f'train done in {train_time:.0f}s; peak GPU mem {torch.cuda.max_memory_allocated() / 2**30:.2f} GiB')

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


mono = lambda Fv: np.array([bool(np.all(np.isfinite(f)) and np.all(np.diff(f) <= 1e-6 * (1 + np.abs(f[:-1])))) for f in Fv])
loss_curve = [float(np.mean(hist[i:i + 100])) for i in range(0, len(hist), 100)] if hist else []
results = dict(name=name, sys=args.sys, model=args.model, seed=args.seed, tag=args.tag, noise=args.noise, iters=args.iters,
               params=nparam, args=vars(args), train_time_s=train_time, early_stop_iter=early_stop, iters_done=len(hist) + nskip,
               final_train_loss=float(np.mean(hist[-100:])) if hist else None, train_nskip=nskip,
               train_mean_substeps=float(np.mean(subs)) if subs else None, loss_hist=hist[::20], loss_curve100=loss_curve, sets={})
snap = {}
for setname in ['test_id', 'test_oodpot', 'test_oodinit']:
    sd = data[setname]; Y = sd['rho'].numpy().astype(np.float64)
    F = fields(sd)
    if args.model == 'jko_inst':
        tf0 = time.time()
        beta = model.fit(sd['rho'][:, :T_TRAIN + 1].to(dev), DT)
        tfit = time.time() - tf0
        log(f'[{setname}] jko_inst fitted on first {T_TRAIN} steps in {tfit:.1f}s; beta median {beta.median():.3f} (min {beta.min():.3f} max {beta.max():.3f})')
    torch.cuda.synchronize(); te = time.time()
    R, nsub, ctx = rollout(F, sd['rho'][:, 0].to(dev), Y.shape[1] - 1)
    torch.cuda.synchronize(); tinf = time.time() - te
    R = R.numpy().astype(np.float64)
    B, T1 = Y.shape[:2]
    err = np.linalg.norm((R - Y).reshape(B, T1, -1), axis=-1) / np.linalg.norm(Y.reshape(B, T1, -1), axis=-1)
    errinf = np.where(np.isfinite(err), err, np.inf)
    massR, massY = R.mean((2, 3)), Y.mean((2, 3))
    massE = np.abs(massR - massY) / massY
    minR = R.min((2, 3)); negf = (R < 0).mean((2, 3))
    blown_t = ~np.isfinite(R).all((2, 3))
    peakY, peakR = Y.max((2, 3)), R.max((2, 3))
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
        peak_relerr_median=np.median(np.where(np.isfinite(peakR), np.abs(peakR - peakY) / peakY, np.inf), 0),
        peak_true_median=np.median(peakY, 0), peak_model_median=np.nanmedian(peakR, 0),
    )
    if SYSCLASS == 'S3':
        S['overshoot_max'] = float(np.nanmax(R) - 1.0) if np.isfinite(R).any() else None
        S['overshoot_per_traj'] = np.nanmax(R.reshape(B, -1), 1) - 1.0
        S['sat_err_10T_median'] = float(np.nanmedian(np.abs(R[:, -1] - Y[:, -1]).max((1, 2))))   # Linf error at 10T (saturation plateau)
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
            return f.mean((2, 3))
        FY, FR = Ftrue(Y), Ftrue(R)
        S['true_F_monotone_frac_truth'] = float(mono(FY).mean())
        S['true_F_monotone_frac_model'] = float(mono(FR).mean())
        S['true_F_end_relerr_median'] = float(np.median(np.where(np.isfinite(FR[:, -1]), np.abs(FR[:, -1] - FY[:, -1]) / np.abs(FY[:, -1]), np.inf)))
    if hasattr(model, 'lyapunov') and SYSCLASS != 'S3':
        with torch.no_grad():
            L = np.stack([model.lyapunov(torch.tensor(R[:, t], device=dev, dtype=torch.float32), F).cpu().numpy() for t in range(T1)], 1)
        S['learned_lyapunov_monotone_frac'] = float(mono(L).mean())
        S['learned_lyapunov_max_increase'] = float(np.nanmax(np.diff(L, axis=1))) if np.isfinite(L).any() else None
    if SYSCLASS == 'S2' and hasattr(model, 'flux_q'):
        with torch.no_grad():
            rend = torch.tensor(R[:, -1], device=dev, dtype=torch.float32)
            fin = torch.isfinite(rend).all(-1).all(-1)
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
    snap[setname + '_peak'] = peakR.astype(np.float32)
    log(f'[{setname}] relerr@T,2T,5T,10T median = ' + ', '.join(f'{np.median(errinf[:, t]):.3e}' for t in [20, 40, 100, 200])
        + f' | blown {S["n_blown_end"]}/{B} | max mass err {np.nanmax(S["mass_err_max"]) if np.isfinite(S["mass_err_max"]).any() else float("nan"):.2e}'
        + f' | min rho {np.nanmin(S["min_rho_min"]) if np.isfinite(S["min_rho_min"]).any() else float("nan"):.2e} | peak err@10T {S["peak_relerr_median"][-1]:.2e} | substeps {S["substeps_per_DT_mean"]:.1f} | infer {tinf:.1f}s')
results = jnum(json.loads(json.dumps(results, default=lambda o: o.tolist() if hasattr(o, 'tolist') else str(o))))
with open(os.path.join(odir, name + '.json'), 'w') as f:
    json.dump(results, f)
np.savez_compressed(os.path.join(odir, name + '_snap.npz'), **snap)
log('done', name)
