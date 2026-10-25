"""Train + evaluate tcme / fisher_hodge / flux_gnn on S2 with the train.py protocol (multi-step rollout loss, Adam 1e-3,
cosine LR, batch 8, K in [4,8], grad-clip 1).  Outputs to runs/tcme/.
  python3 train_tcme.py --model tcme --seed 0 --iters 2500 [--ntrain 8 --tag _n8] [--smoke]
  python3 train_tcme.py --make_ref          # fine-grid (64x64) stationary reference states for the thermo readout
Thermodynamic readout (tcme only; uses analysis/hodge_lib.py) runs after evaluation."""
import argparse, json, time, os, sys, math
import numpy as np
import torch
from torch.utils.checkpoint import checkpoint
from common import *
import models2                                   # (patches models.rho_scale; S2 unchanged)
from models2 import FluxGNN
from models import FisherHodge, NSUB
from models_tcme import TCME, stream_from_dpsi

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', 'analysis'))
import hodge_lib as HL

MODELS = dict(tcme=TCME, fisher_hodge=FisherHodge, flux_gnn=FluxGNN)
FIELDS = ['U', 'dpsi', 'psin']
SETS = ['test_id', 'test_oodpot', 'test_oodinit']
dev = 'cuda'
ODIR_DEFAULT = os.path.join(RUNS, 'tcme')


def fields(setd, idx=None):
    return {k: (setd[k] if idx is None else setd[k][idx]).to(dev) for k in FIELDS}


def model_flux_phys(model, rho, ctx):
    """physical edge flux per unit length (same units as data J_end)."""
    n = rho.shape[-1]
    if isinstance(model, TCME):
        return model.flux(rho, ctx) / n
    return -(D * n) * model.flux_q(rho, ctx)


def relerr_b(a, b):
    a = np.asarray(a, np.float64); b = np.asarray(b, np.float64)
    return np.linalg.norm((a - b).reshape(a.shape[0], -1), axis=1) / np.linalg.norm(b.reshape(b.shape[0], -1), axis=1)


# ------------------------------------------------------------------ fine-grid stationary reference
def make_ref(odir):
    import gen_data as G
    data = torch.load(os.path.join(RUNS, 'data', 'S2.pt'))
    meta = json.load(open(os.path.join(RUNS, 'data', 'S2_meta.json')))
    (fXY, fC), (cXY, cC) = G.grids(NF), G.grids(N)
    nf2 = NF * NF
    I = torch.eye(nf2, dtype=torch.float64, device=dev).reshape(nf2, NF, NF)
    for setname in SETS:
        prs = meta['sets'][setname]['params']
        out = dict(rho_ss=[], J_ss=[], sigma_ref_fine=[], sigma_ref_coarse=[], resid=[], Ucheck=[], sigma_end_ref=[])
        for b, pr in enumerate(prs):
            Uf = torch.tensor(D * G.eval_field(pr['U'], fXY), dtype=torch.float64, device=dev)
            Pf = torch.tensor(D * G.eval_field(pr['psi'], fC), dtype=torch.float64, device=dev)
            uvel = dual_dpsi(Pf) * NF
            Uc = G.eval_field(pr['U'], cXY)
            out['Ucheck'].append(float(np.abs(Uc - data[setname]['U'][b].numpy()).max()))
            M = torch.zeros(nf2, nf2, dtype=torch.float64, device=dev)
            for j0 in range(0, nf2, 1024):
                M[:, j0:j0 + 1024] = G.fine_rhs('S2', I[j0:j0 + 1024], Uf, uvel, 0.0).reshape(-1, nf2).T
            A = M.clone(); A[0] = 1.0
            rhs = torch.zeros(nf2, dtype=torch.float64, device=dev); rhs[0] = nf2      # mean density 1
            rf = torch.linalg.solve(A, rhs)
            out['resid'].append(float((M @ rf).abs().max() / (M.abs().max() * rf.abs().max())))
            rf = rf.reshape(NF, NF)
            Jf = G.fine_flux('S2', rf, Uf, uvel)
            rc, Jc = coarsen(rf), coarsen_flux(Jf)
            out['rho_ss'].append(rc.float().cpu()); out['J_ss'].append(Jc.float().cpu())
            out['sigma_ref_fine'].append(float((Jf ** 2 / (D * mean_e(rf))).sum() / NF ** 2))
            out['sigma_ref_coarse'].append(float((Jc ** 2 / (D * mean_e(rc))).sum() / N ** 2))
            Ye = data[setname]['rho'][b, -1].double(); Je = data[setname]['J_end'][b].double()
            out['sigma_end_ref'].append(float((Je ** 2 / (D * mean_e(Ye))).sum() / N ** 2))
            if b == 0:
                print(setname, 'min rho_ss', float(rf.min()), 'resid', out['resid'][-1], 'Ucheck', out['Ucheck'][-1], flush=True)
        out['rho_ss'] = torch.stack(out['rho_ss']); out['J_ss'] = torch.stack(out['J_ss'])
        torch.save(out, os.path.join(odir, f'ref_ss_{setname}.pt'))
        print(setname, 'sigma_ref fine/coarse', np.median(out['sigma_ref_fine']), np.median(out['sigma_ref_coarse']),
              'max resid', max(out['resid']), 'max Ucheck', max(out['Ucheck']), flush=True)


# ------------------------------------------------------------------ thermodynamic readout of a TC-ME model
def grid_edges(n):
    a, b = np.meshgrid(np.arange(n), np.arange(n), indexing='ij')
    tail = (a * n + b).ravel()
    hh = (a * n + (b + 1) % n).ravel(); hv = (((a + 1) % n) * n + b).ravel()
    return np.concatenate([np.stack([tail, hh], 1), np.stack([tail, hv], 1)], 0)   # order = (dir, a, b) flattening


@torch.no_grad()
def dense_Q(model, rho, ctx, E, bi):
    n = rho.shape[-1]
    qf, qb = model.rates_e(rho, ctx)
    qf = qf[bi].double().cpu().numpy().ravel(); qb = qb[bi].double().cpu().numpy().ravel()
    Q = np.zeros((n * n, n * n))
    Q[E[:, 0], E[:, 1]] = qf; Q[E[:, 1], E[:, 0]] = qb
    Q[np.arange(n * n), np.arange(n * n)] = -Q.sum(1)
    return Q


def decomp(model, rho_b, ctx, E, d0, bi):
    """sigma, excess, housekeeping (ell-weighted) and Euclidean-projection excess for density rho_b (n,n) of traj bi."""
    n = rho_b.shape[-1]
    r = torch.as_tensor(rho_b, device=dev, dtype=torch.float32)
    rr = torch.ones((ctx['k'].shape[0], n, n), device=dev); rr[bi] = r
    Q = dense_Q(model, rr, ctx, E, bi)
    p = rho_b.astype(np.float64).ravel() / (n * n)
    a, b, J, f, ell = HL.edge_quantities(p, Q, E)
    sig = float(np.sum(J * f))
    g, c = HL.hodge(f, ell, d0)
    ge, ce = HL.hodge(f, np.ones_like(f), d0)
    return dict(sigma=sig, excess=HL.energy(g, ell), hk=HL.energy(c, ell), excess_euclid=HL.energy(ge, ell),
                hk_euclid=HL.energy(ce, ell))


@torch.no_grad()
def thermo_readout(model, data, odir, R_end):
    E = grid_edges(N); d0 = HL.incidence(E, N * N)
    res = {}
    for setname in ['test_id', 'test_oodpot']:
        refp = os.path.join(odir, f'ref_ss_{setname}.pt')
        if not os.path.exists(refp):
            print('no reference', refp); continue
        ref = torch.load(refp)
        sd = data[setname]; F = fields(sd); ctx = model.prepare(F)
        B = sd['rho'].shape[0]
        Yend = sd['rho'][:, -1].to(dev)
        # --- (1) t = 10T: model flux at the reference density / at the model's own density vs J_end
        Jm_ref = (model.flux(Yend, ctx) / N).cpu().numpy(); Jt = sd['J_end'].numpy()
        Jm_mod = (model.flux(R_end[setname].to(dev), ctx) / N).cpu().numpy()
        cu = lambda x: curl_f(torch.tensor(x)).numpy()
        S = dict(J_end_relerr_at_refrho=relerr_b(Jm_ref, Jt), J_end_relerr_at_modelrho=relerr_b(Jm_mod, Jt),
                 curlJ_end_relerr_at_refrho=relerr_b(cu(Jm_ref), cu(Jt)), curlJ_end_relerr_at_modelrho=relerr_b(cu(Jm_mod), cu(Jt)))
        # --- (2) model stationary state (fixed point of pi = stationary(Q(pi)) since Q depends on rho through g)
        pis, it_used, sig_pi, ex_pi, exe_pi, sig_rss, ex_rss, exe_rss, sig_end, ex_end, exe_end = ([] for _ in range(11))
        for bi in range(B):
            rho = torch.ones((B, N, N), device=dev)
            prev = None
            for it in range(8):
                Q = dense_Q(model, rho, ctx, E, bi)
                pi = HL.stationary(Q); pi = np.clip(pi, 1e-300, None)
                dens = pi * N * N
                if prev is not None and np.abs(dens - prev).max() < 1e-6 * dens.max():
                    break
                prev = dens
                rho[bi] = torch.tensor(dens.reshape(N, N), device=dev, dtype=torch.float32)
            it_used.append(it + 1)
            pis.append(dens.reshape(N, N))
            d_pi = decomp(model, dens.reshape(N, N), ctx, E, d0, bi)
            d_rs = decomp(model, ref['rho_ss'][bi].numpy(), ctx, E, d0, bi)
            d_en = decomp(model, sd['rho'][bi, -1].numpy(), ctx, E, d0, bi)
            for L, d in [((sig_pi, ex_pi, exe_pi), d_pi), ((sig_rss, ex_rss, exe_rss), d_rs), ((sig_end, ex_end, exe_end), d_en)]:
                L[0].append(d['sigma']); L[1].append(d['excess'] / d['sigma']); L[2].append(d['excess_euclid'] / d['sigma'])
        pis = np.stack(pis)
        rpi = torch.tensor(pis, device=dev, dtype=torch.float32)
        Jpi = (model.flux(rpi, ctx) / N).cpu().numpy(); Jss = ref['J_ss'].numpy()
        Jrs = (model.flux(ref['rho_ss'].to(dev), ctx) / N).cpu().numpy()
        S.update(
            stationary_fixedpoint_iters=it_used,
            rho_ss_relerr=relerr_b(pis, ref['rho_ss'].numpy()),
            J_ss_relerr=relerr_b(Jpi, Jss), curlJ_ss_relerr=relerr_b(cu(Jpi), cu(Jss)),
            J_ss_relerr_at_refrho=relerr_b(Jrs, Jss),
            sigma_model_ss=sig_pi, sigma_ref_ss_fine=ref['sigma_ref_fine'], sigma_ref_ss_coarse=ref['sigma_ref_coarse'],
            sigma_ratio_ss_fine=list(np.array(sig_pi) / np.array(ref['sigma_ref_fine'])),
            sigma_ratio_ss_coarse=list(np.array(sig_pi) / np.array(ref['sigma_ref_coarse'])),
            sigma_model_at_refss=sig_rss, sigma_model_at_Yend=sig_end, sigma_ref_end_coarse=ref['sigma_end_ref'],
            sigma_ratio_end=list(np.array(sig_end) / np.array(ref['sigma_end_ref'])),
            excess_frac_at_pi=ex_pi, excess_frac_at_refss=ex_rss, excess_frac_at_Yend=ex_end,
            excess_euclid_frac_at_pi=exe_pi, excess_euclid_frac_at_refss=exe_rss, excess_euclid_frac_at_Yend=exe_end,
        )
        # --- (3) cycle part of A: harmonic coefficients and face affinities vs the true drift
        A = ctx['A'].cpu().numpy(); dp = sd['dpsi'].numpy()
        S.update(
            harmonic_b_per_edge=ctx['b'].cpu().numpy().tolist(),
            harmonic_b_true_per_edge=dp.mean((-1, -2)).tolist(),
            harmonic_b_over_rmsA=list(np.abs(ctx['b'].cpu().numpy()).max(1) / np.sqrt((A ** 2).mean((1, 2, 3)))),
            harmonic_v_learned=(N * ctx['b'].cpu().numpy()).tolist(),
            A_vs_dpsi_relerr=relerr_b(A, dp), curlA_vs_curl_dpsi_relerr=relerr_b(cu(A), cu(dp)),
            A_dpsi_regression_slope=[float((A[b] * dp[b]).sum() / (dp[b] ** 2).sum()) for b in range(B)],
            gam=float(model.gam), alpha=float(model.alpha), v=model.v.detach().cpu().tolist(), Wb=model.Wb.detach().cpu().tolist(),
        )
        res[setname] = S
        med = lambda k: float(np.median(S[k]))
        print(f'[thermo {setname}] J_end err(ref rho) {med("J_end_relerr_at_refrho"):.3e} curl {med("curlJ_end_relerr_at_refrho"):.3e} | '
              f'rho_ss err {med("rho_ss_relerr"):.3e} J_ss err {med("J_ss_relerr"):.3e} curl {med("curlJ_ss_relerr"):.3e} | '
              f'sigma_ss ratio fine {med("sigma_ratio_ss_fine"):.4f} coarse {med("sigma_ratio_ss_coarse"):.4f} end {med("sigma_ratio_end"):.4f} | '
              f'excess frac pi {med("excess_frac_at_pi"):.2e} refss {med("excess_frac_at_refss"):.2e} Yend {med("excess_frac_at_Yend"):.2e} '
              f'(euclid pi {med("excess_euclid_frac_at_pi"):.2e}) | b/rmsA {med("harmonic_b_over_rmsA"):.2e} curlA err {med("curlA_vs_curl_dpsi_relerr"):.3e}',
              flush=True)
    return res


def jclean(o):
    return json.loads(json.dumps(o, default=lambda x: x.tolist() if hasattr(x, 'tolist') else str(x)).replace('NaN', 'null').replace('Infinity', '1e308'))


# ------------------------------------------------------------------ main
if __name__ == '__main__':
    p = argparse.ArgumentParser()
    p.add_argument('--model', default='tcme'); p.add_argument('--seed', type=int, default=0)
    p.add_argument('--iters', type=int, default=2500); p.add_argument('--batch', type=int, default=8)
    p.add_argument('--lr', type=float, default=1e-3); p.add_argument('--kmin', type=int, default=4); p.add_argument('--kmax', type=int, default=8)
    p.add_argument('--ntrain', type=int, default=0, help='>0: use only the first ntrain training trajectories')
    p.add_argument('--smoke', action='store_true'); p.add_argument('--tag', default='')
    p.add_argument('--outdir', default=ODIR_DEFAULT); p.add_argument('--make_ref', action='store_true')
    p.add_argument('--no_thermo', action='store_true')
    args = p.parse_args()
    odir = args.outdir; os.makedirs(odir, exist_ok=True)
    if args.make_ref:
        make_ref(odir); sys.exit(0)
    if args.smoke:
        args.iters = 20
    torch.manual_seed(args.seed); np.random.seed(args.seed)
    rng = np.random.default_rng(10_000 + args.seed)
    name = f'S2_{args.model}_s{args.seed}{args.tag}' + ('_smoke' if args.smoke else '')
    logf = open(os.path.join(odir, name + '.log'), 'w')

    def log(*a):
        s = ' '.join(str(x) for x in a); print(s, flush=True); logf.write(s + '\n'); logf.flush()

    data = torch.load(os.path.join(RUNS, 'data', 'S2.pt'))
    model = MODELS[args.model]('S2').to(dev)
    nparam = sum(q.numel() for q in model.parameters())
    log(f'{name}: params={nparam}  args={vars(args)}')
    opt = torch.optim.Adam(model.parameters(), lr=args.lr)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, args.iters, eta_min=1e-5)
    tr = data['train']; RHO = tr['rho'].to(dev)
    ntr = RHO.shape[0] if args.ntrain <= 0 else args.ntrain
    trsub = {k: v[:ntr] for k, v in tr.items()}

    t0 = time.time(); hist = []; nskip = 0; subs = []
    model.train()
    for it in range(args.iters):
        idx = torch.tensor(rng.choice(ntr, min(args.batch, ntr), replace=False), device=dev)
        K = int(rng.integers(args.kmin, args.kmax + 1))
        st = torch.tensor(rng.integers(0, T_TRAIN - K + 1, size=len(idx)), device=dev)
        F = fields(trsub, idx.cpu())
        ctx = model.prepare(F)
        r = RHO[idx, st]
        loss = 0.0; ns_it = 0
        for k in range(1, K + 1):
            if args.model == 'flux_gnn':      # activation checkpointing as in train2.py (35 GB/run otherwise); same gradients
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
        hist.append(float(loss.detach()))
        if it % 100 == 0 or it == args.iters - 1:
            log(f'it {it:5d} loss {np.mean(hist[-100:]):.4e} gnorm {float(gn):.2e} K {K} substeps/DT {np.mean(subs[-100:]):.1f} '
                f'skipped {nskip} t {time.time() - t0:.0f}s')
    train_time = time.time() - t0
    torch.save(model.state_dict(), os.path.join(odir, name + '.pt'))

    # ------------------------------------------------------------------ evaluation (same metrics as train.py)
    model.eval()

    @torch.no_grad()
    def rollout(F, rho0, nsteps):
        ctx = model.prepare(F)
        r = rho0.clone(); out = [r.cpu()]; nsub = []
        for k in range(nsteps):
            r, ns = model.advance(r, ctx, DT)
            nsub.append(ns)
            bad = (~torch.isfinite(r)).any(-1).any(-1) | (r.abs().amax((-1, -2)) > 1e4)
            if bad.any():
                r = torch.where(bad[:, None, None], torch.full_like(r, float('nan')), r)
            out.append(r.cpu())
        return torch.stack(out, 1), nsub, ctx

    results = dict(name=name, model=args.model, seed=args.seed, params=nparam, args=vars(args), train_time_s=train_time,
                   final_train_loss=float(np.mean(hist[-100:])) if hist else None, train_nskip=nskip,
                   train_mean_substeps=float(np.mean(subs)), loss_hist=hist[::20], sets={})
    R_end = {}
    for setname in SETS:
        sd = data[setname]; Y = sd['rho'].numpy().astype(np.float64); F = fields(sd)
        torch.cuda.synchronize(); te = time.time()
        R, nsub, ctx = rollout(F, sd['rho'][:, 0].to(dev), Y.shape[1] - 1)
        torch.cuda.synchronize(); tinf = time.time() - te
        R_end[setname] = R[:, -1].clone()
        R = R.numpy().astype(np.float64)
        B, T1 = Y.shape[:2]
        err = np.linalg.norm((R - Y).reshape(B, T1, -1), axis=-1) / np.linalg.norm(Y.reshape(B, T1, -1), axis=-1)
        errinf = np.where(np.isfinite(err), err, np.inf)
        massE = np.abs(R.mean((2, 3)) - 1.0)
        S = dict(infer_time_s=tinf, substeps_per_DT_mean=float(np.mean(nsub)), n_traj=B,
                 n_blown_end=int((~np.isfinite(R[:, -1])).any((1, 2)).sum()),
                 relerr_median=np.median(errinf, 0), relerr_per_traj_at={str(t): errinf[:, t] for t in [20, 40, 100, 200]},
                 mass_err_max=float(np.nanmax(massE)), min_rho=float(np.nanmin(R)),
                 neg_frac_mean=float(np.nanmean(R < 0)), n_traj_with_negative=int((R < 0).any((1, 2, 3)).sum()))
        with torch.no_grad():
            rend = torch.tensor(R[:, -1], device=dev, dtype=torch.float32)
            fin = torch.isfinite(rend).all(-1).all(-1)
            rend = torch.where(fin[:, None, None], rend, torch.ones_like(rend))
            Jm = model_flux_phys(model, rend, ctx).cpu().numpy()
            refp = os.path.join(odir, f'ref_ss_{setname}.pt')
            if os.path.exists(refp):
                ref = torch.load(refp)
                Jrs = model_flux_phys(model, ref['rho_ss'].to(dev), ctx).cpu().numpy()
                S['J_ss_relerr_at_refrho'] = relerr_b(Jrs, ref['J_ss'].numpy())
                S['curlJ_ss_relerr_at_refrho'] = relerr_b(curl_f(torch.tensor(Jrs)).numpy(), curl_f(ref['J_ss']).numpy())
        Jt = sd['J_end'].numpy()
        ce = relerr_b(curl_f(torch.tensor(Jm)).numpy(), curl_f(torch.tensor(Jt)).numpy())
        S['curl_flux_end_relerr_median'] = float(np.median(np.where(fin.cpu().numpy(), ce, np.inf)))
        S['J_end_relerr_median'] = float(np.median(np.where(fin.cpu().numpy(), relerr_b(Jm, Jt), np.inf)))
        results['sets'][setname] = S
        log(f'[{setname}] relerr@T,2T,5T,10T median = ' + ', '.join(f'{np.median(errinf[:, t]):.3e}' for t in [20, 40, 100, 200])
            + f' | mass err {S["mass_err_max"]:.2e} | min rho {S["min_rho"]:.2e} | neg traj {S["n_traj_with_negative"]}'
            + f' | curlJ@10T {S["curl_flux_end_relerr_median"]:.3e} | substeps {S["substeps_per_DT_mean"]:.1f} | infer {tinf:.1f}s')
    if args.model == 'tcme' and not args.no_thermo:
        tt = time.time()
        results['thermo'] = thermo_readout(model, data, odir, R_end)
        log(f'thermo readout {time.time() - tt:.0f}s')
    with open(os.path.join(odir, name + '.json'), 'w') as f:
        json.dump(jclean(results), f)
    log('done', name)
