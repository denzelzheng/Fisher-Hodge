"""Train + evaluate one (model, seed) on the CRN benchmark (same protocol as fisher_hodge/train.py).
python train_hyper.py --model hyperme|hyperme_rev|crnn|gnn --seed 0 [--iters 2000] [--smoke] [--dev cuda|cpu]"""
import argparse, json, time, os, sys
import numpy as np
import torch
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from hyperme import HyperME, DT, T_TRAIN, T_TEST, RUNS, NS, graph, check_conservation
from baselines_hyper import CRNN, GNNODE3

MODELS = {'hyperme': lambda: HyperME(True), 'hyperme_rev': lambda: HyperME(False), 'crnn': CRNN, 'gnn': GNNODE3}

p = argparse.ArgumentParser()
p.add_argument('--model', default='hyperme'); p.add_argument('--seed', type=int, default=0)
p.add_argument('--iters', type=int, default=2000); p.add_argument('--batch', type=int, default=8)
p.add_argument('--lr', type=float, default=1e-3); p.add_argument('--kmin', type=int, default=4); p.add_argument('--kmax', type=int, default=8)
p.add_argument('--smoke', action='store_true'); p.add_argument('--dev', default='cuda'); p.add_argument('--data', default='')
p.add_argument('--obs', default='none', help='none|global|local4|globalR2: cycle observable added to the loss (E5)')
p.add_argument('--wobs', type=float, default=1.0); p.add_argument('--odir', default='main')
args = p.parse_args()
if args.smoke:
    args.iters = 20
dev = args.dev
torch.manual_seed(args.seed); np.random.seed(args.seed)
rng = np.random.default_rng(10_000 + args.seed)
name = f'{args.model}{args.data}' + ('' if args.obs == 'none' else f'_obs-{args.obs}') + f'_s{args.seed}' + ('_smoke' if args.smoke else '')
odir = os.path.join(RUNS, 'smoke' if args.smoke else args.odir); os.makedirs(odir, exist_ok=True)
logf = open(os.path.join(odir, name + '.log'), 'w')


def log(*a):
    s = ' '.join(str(x) for x in a); print(s, flush=True); logf.write(s + '\n'); logf.flush()


data = torch.load(os.path.join(RUNS, 'data', f'crn{args.data}.pt'))
N = 8
model = MODELS[args.model]().to(dev)
nparam = sum(q.numel() for q in model.parameters())
log(f'{name}: params={nparam} args={vars(args)}')
opt = torch.optim.Adam(model.parameters(), lr=args.lr)
sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, args.iters, eta_min=1e-5)
tr = data['train']; C = tr['c'].to(dev); Utr = tr['U'].to(dev); Gtr = tr['dG'].to(dev)
ntr = C.shape[0]
OBS_W, OBS_SITES = None, None
if args.obs != 'none':
    from obs_hyper import obs_matrix, true_flux
    OBS_W, OBS_SITES = obs_matrix(args.obs, args.seed, dev)            # (n_obs, H)
    Jtr = true_flux(tr, dev)                                             # (B, T+1, H) true fluxes on the training frames
    log(f'cycle observable {args.obs}: {OBS_W.shape[0]} rows, sites {OBS_SITES}, weight {args.wobs}')
hist_obs = []

t0 = time.time(); hist = []; nskip = 0; subs = []
model.train()
for it in range(args.iters):
    idx = torch.tensor(rng.choice(ntr, args.batch, replace=False), device=dev)
    K = int(rng.integers(args.kmin, args.kmax + 1))
    st = torch.tensor(rng.integers(0, T_TRAIN - K + 1, size=args.batch), device=dev)
    ctx = model.prepare(Utr[idx], Gtr[idx], N)
    c = C[idx, st]
    loss = 0.0; ns_it = 0; lobs = 0.0
    for k in range(1, K + 1):
        c, ns = model.advance(c, ctx, DT)
        ns_it += ns
        y = C[idx, st + k]
        loss = loss + (((c - y) ** 2).sum((-1, -2)) / (y ** 2).sum((-1, -2))).mean()
        if OBS_W is not None:
            om = model.flux(c, ctx) @ OBS_W.T; ot = Jtr[idx, st + k] @ OBS_W.T
            lobs = lobs + (((om - ot) ** 2).sum(-1) / (ot ** 2).sum(-1)).mean()
    loss = loss / K
    if OBS_W is not None:
        lobs = lobs / K; hist_obs.append(lobs.item()); loss = loss + args.wobs * lobs
    subs.append(ns_it / K)
    if not torch.isfinite(loss):
        nskip += 1; opt.zero_grad(set_to_none=True); sched.step(); continue
    opt.zero_grad(set_to_none=True); loss.backward()
    gn = torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
    opt.step(); sched.step()
    hist.append(loss.item())
    if it % 100 == 0 or it == args.iters - 1:
        log(f'it {it:5d} loss {np.mean(hist[-100:]):.4e} obs {np.mean(hist_obs[-100:]) if hist_obs else 0:.3e} gnorm {float(gn):.2e} K {K} substeps/DT {np.mean(subs[-100:]):.1f} skipped {nskip} t {time.time() - t0:.0f}s')
train_time = time.time() - t0
torch.save(model.state_dict(), os.path.join(odir, name + '.pt'))

# ------------------------------------------------------------------ evaluation
model.eval()
has_flux = hasattr(model, 'flux')


@torch.no_grad()
def rollout(U, dG, c0, nsteps, n=N):
    ctx = model.prepare(U, dG, n)
    c = c0.clone(); out = [c.cpu()]; nsub = []; sig = []
    bad_all = torch.zeros(c.shape[0], dtype=torch.bool, device=dev)
    for k in range(nsteps):
        if has_flux:
            sig.append(model.epr(c, ctx).cpu())
        c, ns = model.advance(c, ctx, DT); nsub.append(ns)
        bad = (~torch.isfinite(c)).flatten(1).any(1) | (c.abs().flatten(1).amax(1) > 1e4)
        bad_all |= bad
        if bad.any():
            c = torch.where(bad[:, None, None], torch.full_like(c, float('nan')), c)
        out.append(c.cpu())
    if has_flux:
        sig.append(model.epr(c, ctx).cpu())
    return torch.stack(out, 1).numpy(), (torch.stack(sig, 1).numpy() if has_flux else None), nsub, ctx, c


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
    return x


g8 = graph(N, dev)
P8 = g8['P'].cpu().numpy(); zloc = g8['zloc'].cpu().numpy()
results = dict(name=name, model=args.model, seed=args.seed, params=nparam, args=vars(args), train_time_s=train_time,
               final_train_loss=float(np.mean(hist[-100:])) if hist else None, train_nskip=nskip,
               train_mean_substeps=float(np.mean(subs)), obs=args.obs, obs_sites=OBS_SITES, final_obs_loss=float(np.mean(hist_obs[-100:])) if hist_obs else None, obs_hist=hist_obs[::20], loss_hist=hist[::20], sets={})
save = {}
assert_fail = []
for setname in ['test_id', 'test_oodG', 'test_oodU']:
    sd = data[setname]; Y = sd['c'].numpy(); U = sd['U'].to(dev); dG = sd['dG'].to(dev)
    torch.cuda.synchronize() if dev == 'cuda' else None; te = time.time()
    R, sig, nsub, ctx, cend = rollout(U, dG, sd['c'][:, 0].to(dev), Y.shape[1] - 1)
    tinf = time.time() - te
    B = Y.shape[0]
    eX = relerr(R[:, :, :2], Y[:, :, :2]); eA = relerr(R, Y)
    cons = check_conservation(R, g8)
    S = dict(infer_time_s=tinf, substeps_per_DT_mean=float(np.mean(nsub)), substeps_per_DT_max=int(np.max(nsub)),
             n_blown=int((~np.isfinite(R)).reshape(B, -1).any(1).sum()),
             relL2_XXp_median={str(t): float(np.median(eX[:, t])) for t in [20, 40, 100, 200]},
             relL2_XXp_mean={str(t): float(np.mean(eX[:, t])) for t in [20, 40, 100, 200]},
             relL2_all_median={str(t): float(np.median(eA[:, t])) for t in [20, 40, 100, 200]},
             relL2_XXp_per_traj_10T=eX[:, 200], cons_err=cons,
             min_c=float(np.nanmin(R)), n_traj_negative=int(((R < 0).reshape(B, -1)).any(1).sum()),
             neg_frac=float(np.nanmean(R < 0)))
    sv = dict(R_snap=R[:, [0, 20, 40, 100, 200]].astype(np.float32), relL2_XXp_t=eX.astype(np.float32))
    if has_flux:
        st = sd['sigma'].numpy()
        se = np.linalg.norm(sig - st, axis=1) / np.linalg.norm(st, axis=1)
        se = np.where(np.isfinite(se), se, np.inf)
        S['sigma_relerr_median'] = float(np.median(se)); S['sigma_relerr_per_traj'] = se
        S['sigma_end_relerr_median'] = float(np.median(np.where(np.isfinite(sig[:, -1]), np.abs(sig[:, -1] - st[:, -1]) / st[:, -1], np.inf)))
        with torch.no_grad():
            cin = torch.nan_to_num(cend, nan=1.0)
            Jm = model.flux(cin, ctx).cpu().numpy()
        Jt = sd['J'][:, -1].numpy()
        fin = np.isfinite(R[:, -1]).reshape(B, -1).all(1)
        def corr(a, b):
            return float(np.corrcoef(a, b)[0, 1]) if np.std(a) > 0 and np.std(b) > 0 else float('nan')
        cr_all = np.array([corr(Jm[b], Jt[b]) if fin[b] else np.nan for b in range(B)])
        H8 = g8['H']; V = 64
        cr_rx = np.array([corr(Jm[b, :2 * V], Jt[b, :2 * V]) if fin[b] else np.nan for b in range(B)])
        cr_tr = np.array([corr(Jm[b, 2 * V:], Jt[b, 2 * V:]) if fin[b] else np.nan for b in range(B)])
        relJ = np.linalg.norm(Jm - Jt, axis=1) / np.linalg.norm(Jt, axis=1)
        S['flux_corr_10T_median'] = float(np.nanmedian(cr_all)); S['flux_corr_10T_reactions_median'] = float(np.nanmedian(cr_rx))
        S['flux_corr_10T_transport_median'] = float(np.nanmedian(cr_tr)); S['flux_relerr_10T_median'] = float(np.nanmedian(relJ))
        Am = model.A_cycle(ctx).cpu().numpy(); At = sd['A'].numpy() @ P8
        S['Acycle_relerr_median'] = float(np.median(np.linalg.norm(Am - At, axis=1) / np.linalg.norm(At, axis=1)))
        aff = model.cycle_affinity_local(ctx).cpu().numpy()       # (B,V)
        aff_t = (sd['A'][:, g8['r1']] + sd['A'][:, g8['r2']]).numpy().mean(1)     # true dG + log(k2/k2')
        S['affinity_true'] = aff_t; S['affinity_learned_sitemean'] = aff.mean(1); S['affinity_learned_sitestd'] = aff.std(1)
        S['affinity_MAE'] = float(np.abs(aff - aff_t[:, None]).mean())
        # spurious cycles: transport + mixed part of A (true value 0)
        S['A_transport_rms'] = float(np.sqrt(((Am[:, 2 * V:] - At[:, 2 * V:]) ** 2).mean()))
        sv.update(sigma=sig.astype(np.float32), J_end=Jm.astype(np.float32), aff_local=aff.astype(np.float32),
                  A_cycle=Am.astype(np.float32))
    if args.model.startswith('hyperme'):
        ok_c = cons['ker'] < 1e-6 and cons['E'] < 1e-6 and cons['XXp'] < 1e-6
        ok_p = bool(np.nanmin(R) > 0) and S['n_blown'] == 0
        ok_s = bool(np.nanmin(sig) >= 0)
        S['assert_conservation'] = ok_c; S['assert_positivity'] = ok_p; S['assert_sigma_nonneg'] = ok_s
        if not (ok_c and ok_p and ok_s):
            assert_fail.append(setname)
    results['sets'][setname] = S
    save.update({f'{setname}/{k}': v for k, v in sv.items()})
    log(f'[{setname}] relL2(X,Xp) T {S["relL2_XXp_median"]["20"]:.3e} 10T {S["relL2_XXp_median"]["200"]:.3e} | cons ker {cons["ker"]:.1e} E {cons["E"]:.1e} XXp {cons["XXp"]:.1e} '
        f'| min c {S["min_c"]:.2e} neg traj {S["n_traj_negative"]} blown {S["n_blown"]} | substeps {S["substeps_per_DT_mean"]:.1f} | infer {tinf:.1f}s'
        + (f' | sigma relerr {S["sigma_relerr_median"]:.3f} | Jcorr {S["flux_corr_10T_median"]:.3f} (rx {S["flux_corr_10T_reactions_median"]:.3f}, tr {S["flux_corr_10T_transport_median"]:.3f}) '
           f'| aff MAE {S["affinity_MAE"]:.3f} | Acyc relerr {S["Acycle_relerr_median"]:.3f}' if has_flux else ''))

# Gillespie (noisy data) test
gd = torch.load(os.path.join(RUNS, 'data', f'crn{args.data}_gillespie.pt'))
Yg = gd['c'].numpy(); Yo = gd['c_ode'].numpy()
Rg, _, _, _, _ = rollout(gd['U'].to(dev), gd['dG'].to(dev), gd['c'][:, 0].to(dev), T_TEST)
results['gillespie'] = dict(relL2_vs_gillespie_10T=float(np.median(relerr(Rg[:, :, :2], Yg[:, :, :2])[:, -1])),
                            relL2_vs_ode_10T=float(np.median(relerr(Rg[:, :, :2], Yo[:, :, :2])[:, -1])),
                            noise_floor_ode_vs_gillespie_10T=float(np.median(relerr(Yo[:, :, :2], Yg[:, :, :2])[:, -1])))
log('[gillespie]', results['gillespie'])
results['assert_fail_sets'] = assert_fail
results = jnum(results)
with open(os.path.join(odir, name + '.json'), 'w') as f:
    json.dump(results, f)
np.savez_compressed(os.path.join(odir, name + '_eval.npz'), **save)
if args.model.startswith('hyperme'):
    assert not assert_fail, f'property assertions failed on {assert_fail}'
    log('property assertions (conservation<1e-6, positivity, sigma>=0) PASSED on all test sets')
log('done', name, f'{time.time() - t0:.0f}s')
