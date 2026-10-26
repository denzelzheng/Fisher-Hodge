"""Train + evaluate one (model, seed) on C2 (schlogl) / C3 (brus) / C3eq control (brus_eq). Same protocol as train_hyper.py:
2000 iters, batch 8, windows of K in [4,8] steps inside the 20-step training trajectories, relative-MSE loss, Adam 1e-3 cosine.
python train_c23.py --bench schlogl|brus|brus_eq --model hyperme|hyperme_rev|crnn|gnn --seed 0 [--eval_only]"""
import argparse, json, time, os, sys
import numpy as np
import torch
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from crn2 import make_model, graph, RUNS

p = argparse.ArgumentParser()
p.add_argument('--bench', default='schlogl'); p.add_argument('--model', default='hyperme'); p.add_argument('--seed', type=int, default=0)
p.add_argument('--iters', type=int, default=2000); p.add_argument('--batch', type=int, default=8); p.add_argument('--lr', type=float, default=1e-3)
p.add_argument('--kmin', type=int, default=4); p.add_argument('--kmax', type=int, default=8)
p.add_argument('--smoke', action='store_true'); p.add_argument('--eval_only', action='store_true'); p.add_argument('--tag', default=''); p.add_argument('--init', default='', help='warm start: state dict path relative to runs/C23')
args = p.parse_args()
if args.smoke:
    args.iters = 10
torch.set_num_threads(int(os.environ.get('OMP_NUM_THREADS', '8')))
dev = 'cpu'
torch.manual_seed(args.seed); np.random.seed(args.seed)
rng = np.random.default_rng(10_000 + args.seed)
name = f'{args.bench}_{args.model}{args.tag}_s{args.seed}' + ('_smoke' if args.smoke else '')
odir = os.path.join(RUNS, 'C23_smoke' if args.smoke else 'C23'); os.makedirs(odir, exist_ok=True)
logf = open(os.path.join(odir, name + ('.evallog' if args.eval_only else '.log')), 'w')


def log(*a):
    s = ' '.join(str(x) for x in a); print(s, flush=True); logf.write(s + '\n'); logf.flush()


data = torch.load(os.path.join(RUNS, 'data', f'{args.bench}.pt'))
meta = data['meta']; DT = meta['DT']; N = meta['N']; T_TRAIN = meta['T_TRAIN']
dmean, dstd = meta['drv_mean'].to(dev), meta['drv_std'].to(dev)


def ndrv(d):
    return (d.to(dev) - dmean) / dstd


model = make_model(args.model, args.bench).to(dev)
if getattr(model, 'chem', False):          # chemostat potentials enter the rate exponents: raw (log a, log b)
    ndrv = lambda d: d.to(dev)
nparam = sum(q.numel() for q in model.parameters())
log(f'{name}: params={nparam} args={vars(args)} DT={DT}')
hist, subs, nskip, train_time = [], [], 0, 0.0
if args.eval_only:
    model.load_state_dict(torch.load(os.path.join(odir, name + '.pt')))
    old = json.load(open(os.path.join(odir, name + '.json')))
    train_time, hist = old['train_time_s'], old['loss_hist']
else:
    if args.init:
        model.load_state_dict(torch.load(os.path.join(odir, args.init)))
        log(f'warm start from {args.init}')
    opt = torch.optim.Adam(model.parameters(), lr=args.lr)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, args.iters, eta_min=1e-5)
    tr = data['train']; C = tr['c'].to(dev); Dtr = ndrv(tr['drv']); ntr = C.shape[0]
    t0 = time.time()
    model.train()
    for it in range(args.iters):
        idx = torch.tensor(rng.choice(ntr, args.batch, replace=False), device=dev)
        K = int(rng.integers(args.kmin, args.kmax + 1))
        st = torch.tensor(rng.integers(0, T_TRAIN - K + 1, size=args.batch), device=dev)
        ctx = model.prepare(Dtr[idx], N)
        c = C[idx, st]
        loss = 0.0; ns_it = 0
        for k in range(1, K + 1):
            c, ns = model.advance(c, ctx, DT)
            ns_it += ns
            y = C[idx, st + k]
            loss = loss + (((c - y) ** 2).sum((-1, -2)) / (y ** 2).sum((-1, -2))).mean()
        loss = loss / K
        subs.append(ns_it / K)
        if not torch.isfinite(loss):
            nskip += 1; opt.zero_grad(set_to_none=True); sched.step(); continue
        opt.zero_grad(set_to_none=True); loss.backward()
        gn = torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        opt.step(); sched.step()
        hist.append(loss.item())
        if it % 100 == 0 or it == args.iters - 1:
            log(f'it {it:5d} loss {np.mean(hist[-100:]):.4e} gnorm {float(gn):.2e} K {K} substeps/DT {np.mean(subs[-100:]):.1f} skipped {nskip} t {time.time() - t0:.0f}s')
    train_time = time.time() - t0
    torch.save(model.state_dict(), os.path.join(odir, name + '.pt'))

# ------------------------------------------------------------------ evaluation
model.eval()
has_flux = hasattr(model, 'flux')


@torch.no_grad()
def rollout(drv, c0, nsteps, n=N):
    ctx = model.prepare(drv, n)
    c = c0.clone(); out = [c.cpu()]; nsub = []; sig = []
    for k in range(nsteps):
        if has_flux:
            sig.append(model.epr(c, ctx).cpu())
        c, ns = model.advance(c, ctx, DT); nsub.append(ns)
        bad = (~torch.isfinite(c)).flatten(1).any(1) | (c.abs().flatten(1).amax(1) > 1e4)
        if bad.any():
            c = torch.where(bad[:, None, None], torch.full_like(c, float('nan')), c)
        out.append(c.cpu())
    if has_flux:
        sig.append(model.epr(c, ctx).cpu())
    return torch.stack(out, 1).numpy(), (torch.stack(sig, 1).numpy() if has_flux else None), nsub, ctx


def relerr(R, Y):
    B, T1 = Y.shape[:2]
    e = np.linalg.norm((R - Y).reshape(B, T1, -1), axis=-1) / np.linalg.norm(Y.reshape(B, T1, -1), axis=-1)
    return np.where(np.isfinite(e), e, np.inf)


def jnum(x):
    if isinstance(x, dict):
        return {k: jnum(v) for k, v in x.items()}
    if isinstance(x, (list, tuple)):
        return [jnum(v) for v in x]
    if isinstance(x, np.ndarray):
        return jnum(x.tolist())
    if isinstance(x, (float, np.floating)):
        return float(x) if np.isfinite(x) else None
    if isinstance(x, (np.integer,)):
        return int(x)
    if isinstance(x, (np.bool_,)):
        return bool(x)
    return x


TS = [20, 40, 100, 200]
results = dict(name=name, bench=args.bench, model=args.model, seed=args.seed, params=nparam, args=vars(args), train_time_s=train_time,
               final_train_loss=float(np.mean(hist[-100:])) if hist else None, train_nskip=nskip,
               train_mean_substeps=float(np.mean(subs)) if subs else None, loss_hist=hist[::20] if not args.eval_only else hist, sets={})
save = {}
assert_fail = []
sets = [k for k in data if k.startswith('test')]
g = graph(args.bench, N, dev)
V = N * N
for setname in sets:
    sd = data[setname]; Y = sd['c'].numpy(); D = ndrv(sd['drv'])
    te = time.time()
    R, sig, nsub, ctx = rollout(D, sd['c'][:, 0].to(dev), Y.shape[1] - 1)
    tinf = time.time() - te
    B = Y.shape[0]
    e = relerr(R, Y)
    totm, tott = R[:, :, 0].sum(-1), Y[:, :, 0].sum(-1)
    etot = np.abs(totm - tott) / tott
    S = dict(infer_time_s=tinf, substeps_per_DT_mean=float(np.mean(nsub)), substeps_per_DT_max=int(np.max(nsub)),
             n_blown=int((~np.isfinite(R)).reshape(B, -1).any(1).sum()),
             relL2_median={str(t): float(np.median(e[:, t])) for t in TS}, relL2_mean={str(t): float(np.mean(e[:, t])) for t in TS},
             relL2_per_traj_10T=e[:, -1], totX_relerr_10T_median=float(np.median(np.where(np.isfinite(etot[:, -1]), etot[:, -1], np.inf))),
             min_c=float(np.nanmin(R)), n_traj_negative=int(((R < 0).reshape(B, -1)).any(1).sum()), neg_frac=float(np.nanmean(R < 0)))
    if 'homog' in sd:
        h = sd['homog'].numpy().astype(bool)
        for lab, msk in [('homog', h), ('spatial', ~h)]:
            S[f'relL2_median_{lab}'] = {str(t): float(np.median(e[msk, t])) for t in TS}
        sp = np.nanstd(R[h, -1, 0], axis=-1)
        S['homog_spatial_std_X_10T_max'] = float(np.nanmax(sp)) if np.isfinite(sp).any() else None
    sv = dict(R=R.astype(np.float32), relL2_t=e.astype(np.float32))
    if has_flux:
        st = sd['sigma'].numpy()
        se = np.linalg.norm(sig - st, axis=1) / np.linalg.norm(st, axis=1)
        se = np.where(np.isfinite(se), se, np.inf)
        S['sigma_relerr_median'] = float(np.median(se)); S['sigma_relerr_per_traj'] = se
        At = sd['A'].numpy() @ g['P'].cpu().numpy()
        Am = model.A_cycle(ctx).cpu().numpy()
        S['Acycle_relerr_median'] = float(np.median(np.linalg.norm(Am - At, axis=1) / np.linalg.norm(At, axis=1))) if np.abs(At).max() > 0 else float(np.median(np.linalg.norm(Am, axis=1)))
        aff = model.cycle_affinity_local(ctx).cpu().numpy()        # (B, ncyc, V)
        aff_t = sd['aff'].numpy().reshape(B, -1)
        S['affinity_true_mean'] = aff_t.mean(0); S['affinity_learned_sitemean'] = aff.mean(-1); S['affinity_learned_sitestd'] = aff.std(-1).mean(0)
        S['affinity_MAE'] = float(np.abs(aff - aff_t[:, :, None]).mean())
        sv.update(sigma=sig.astype(np.float32), aff=aff.mean(-1).astype(np.float32))
    if args.model.startswith('hyperme'):
        ok_p = bool(np.nanmin(R) > 0) and S['n_blown'] == 0
        ok_s = bool(np.nanmin(sig) >= 0)
        S['assert_positivity'] = ok_p; S['assert_sigma_nonneg'] = ok_s
        if not (ok_p and ok_s):
            assert_fail.append(setname)
    results['sets'][setname] = S
    save.update({f'{setname}/{k}': v for k, v in sv.items()})
    log(f'[{setname}] relL2 T {S["relL2_median"]["20"]:.3e} 2T {S["relL2_median"]["40"]:.3e} 10T {S["relL2_median"]["200"]:.3e} | totX err 10T {S["totX_relerr_10T_median"]:.3e} '
        f'| min c {S["min_c"]:.2e} neg traj {S["n_traj_negative"]} blown {S["n_blown"]} | substeps {S["substeps_per_DT_mean"]:.1f} | infer {tinf:.1f}s'
        + (f' | sigma relerr {S["sigma_relerr_median"]:.3f} | aff MAE {S["affinity_MAE"]:.3f} | Acyc relerr {S["Acycle_relerr_median"]:.3f}' if has_flux else '')
        + (f' | homog 10T {S["relL2_median_homog"]["200"]:.3e} spatial 10T {S["relL2_median_spatial"]["200"]:.3e} homog-std {S["homog_spatial_std_X_10T_max"]}' if 'homog' in sd else ''))

# ---------------- C2: single-site birth-death rate tables for the stochastic (Omega = 100) switching statistics
if args.bench == 'schlogl' and has_flux:
    OMEGA = 100
    bgrid = np.linspace(1.7, 2.3, 13)
    nmax = 6 * OMEGA
    xs = np.maximum(np.arange(nmax + 1), 1e-9) / OMEGA
    lam, mu_ = [], []
    with torch.no_grad():
        for bb in bgrid:
            drv = ndrv(torch.tensor([[np.log(3.0), np.log(bb)]]).repeat(len(xs), 1))
            ctx = model.prepare(drv, N)
            c = torch.tensor(xs, device=dev)[:, None, None].expand(len(xs), 1, V).contiguous()
            a, b = model.ab(c, ctx)
            aR1, bR1, aR2, bR2 = a[:, 0], b[:, 0], a[:, V], b[:, V]
            lam.append((OMEGA * (aR1 + bR2)).cpu().numpy()); mu_.append((OMEGA * (bR1 + aR2)).cpu().numpy())
    save['bd/lam'] = np.array(lam); save['bd/mu'] = np.array(mu_); save['bd/b'] = bgrid
    log('birth-death tables saved (a=3, b grid 1.7..2.3, Omega=100)')

# ---------------- C3 / C3eq: linearisation spectrum at the model's own end state (oscillatory modes = complex pairs)
if args.bench.startswith('brus'):
    eig = {}
    for setname in ['test_id']:
        sd = data[setname]; D = ndrv(sd['drv'])
        R = save[f'{setname}/R']
        for b in [0, 1, 8, 9]:
            cend = torch.tensor(R[b, -1], dtype=torch.float64, device=dev)
            if not torch.isfinite(cend).all():
                continue
            ctx = model.prepare(D[b:b + 1], N)
            f = lambda x: model.rhs(x.view(1, *cend.shape), ctx).flatten()
            Jm = torch.autograd.functional.jacobian(f, cend.flatten().clone()).detach().cpu().numpy()
            ev = np.linalg.eigvals(Jm)
            eig[f'{setname}_{b}'] = ev
            save[f'eig/{setname}_{b}'] = ev
            log(f'[eig {setname} traj {b}] max Re {ev.real.max():.3e} max |Im| {np.abs(ev.imag).max():.3e} '
                f'| #complex(|Im|>1e-6) {(np.abs(ev.imag) > 1e-6).sum()}')

results['assert_fail_sets'] = assert_fail
results = jnum(results)
with open(os.path.join(odir, name + '.json'), 'w') as f:
    json.dump(results, f)
np.savez_compressed(os.path.join(odir, name + '_eval.npz'), **save)
if args.model.startswith('hyperme'):
    assert not assert_fail, f'property assertions failed on {assert_fail}'
    log('property assertions (positivity, sigma>=0) PASSED on all test sets')
log('done', name, f'train {train_time:.0f}s')
