"""E2: coarse-graining (lumpability / Chentsov) consistency test, plus the summary of runs/tcme/.
  (a) roll out on 32x32, then 2x2 block-lump;  (b) lump the inputs, roll out the *same* trained model on 16x16 (h = 1/16).
  Lumping: density -> block mean (= block-summed mass / coarse cell area, so rho keeps mean 1); U, psin -> block mean;
  dpsi -> flux-consistent coarse edge Peclet number = dual_dpsi of the corner stream function sampled on the coarse corners
  (= sum of the two fine dpsi crossing each coarse face = h_c * block-averaged velocity / D).
  The rate scale D/h^2 of fisher_hodge / flux_gnn is a module global; it is set to D n^2 for the coarse rollout
  (no change to models.py / models2.py).  tcme reads n from the input shape.
  python3 coarse_test.py                 # E2 -> runs/tcme/E2_coarse.json, E2_commutation.png
  python3 coarse_test.py --summary       # -> runs/tcme/summary.md"""
import argparse, json, os, sys, glob, time
import numpy as np
import torch
from common import *
import common, models, models2
from train_tcme import MODELS, fields, jclean
from models_tcme import TCME, stream_from_dpsi

dev = 'cuda'
ODIR = os.path.join(RUNS, 'tcme')
TS = [20, 40, 100, 200]


def set_rate_scale(n):
    s = D * n * n
    models.S_SCALE = s; models2.S_SCALE = s; common.S_SCALE = s


def lump_fields(F):
    P = stream_from_dpsi(F['dpsi'])
    return dict(U=coarsen(F['U']), psin=coarsen(F['psin']), dpsi=dual_dpsi(P[..., 1::2, 1::2]))


@torch.no_grad()
def rollout(model, F, rho0, nsteps):
    set_rate_scale(rho0.shape[-1])
    ctx = model.prepare(F)
    r = rho0.clone(); out = [r.cpu()]
    for k in range(nsteps):
        r, _ = model.advance(r, ctx, DT)
        bad = (~torch.isfinite(r)).any(-1).any(-1) | (r.abs().amax((-1, -2)) > 1e4)
        if bad.any():
            r = torch.where(bad[:, None, None], torch.full_like(r, float('nan')), r)
        out.append(r.cpu())
    set_rate_scale(N)
    return torch.stack(out, 1).double()


def rel(a, b):
    """per (traj, time) relative L2 ||a-b|| / ||b||  -> (B, T)"""
    B, T = a.shape[:2]
    return (torch.linalg.vector_norm((a - b).reshape(B, T, -1), dim=-1) / torch.linalg.vector_norm(b.reshape(B, T, -1), dim=-1)).numpy()


def run_e2(args):
    data = torch.load(os.path.join(RUNS, 'data', 'S2.pt'))
    out = {}
    jobs = [('sqra_oracle', None)] + [(m, s) for m in ['tcme', 'fisher_hodge', 'flux_gnn'] for s in args.seeds]
    for mname, seed in jobs:
        if mname == 'sqra_oracle':        # untrained TC-ME with psi = stream function exactly: k = D/h^2, A = dpsi, phi = U
            model = TCME('S2').to(dev)
            with torch.no_grad():
                model.gam.fill_(1.0)
            key = 'sqra_oracle'
        else:
            ck = os.path.join(ODIR, f'S2_{mname}_s{seed}.pt')
            if not os.path.exists(ck):
                print('missing', ck); continue
            model = MODELS[mname]('S2').to(dev); model.load_state_dict(torch.load(ck)); model.eval()
            key = f'{mname}_s{seed}'
        out[key] = {}
        for setname in args.sets:
            t0 = time.time()
            sd = data[setname]; Y = sd['rho'].double(); F = fields(sd); Fc = lump_fields(F)
            nst = Y.shape[1] - 1
            Ra32 = rollout(model, F, sd['rho'][:, 0].to(dev), nst)
            Ra = coarsen(Ra32)
            Rb = rollout(model, Fc, coarsen(sd['rho'][:, 0]).to(dev), nst)
            Yc = coarsen(Y)
            e32, ea, eb, comm = rel(Ra32, Y), rel(Ra, Yc), rel(Rb, Yc), rel(Rb, Ra)
            mass_b = float(np.nanmax(np.abs(Rb.mean((-1, -2)).numpy() - 1)))
            S = dict(err32_med=np.nanmedian(e32, 0), errA_med=np.nanmedian(ea, 0), errB_med=np.nanmedian(eb, 0),
                     comm_med=np.nanmedian(comm, 0), comm_mean=np.nanmean(comm, 0), comm_per_traj_10T=comm[:, -1],
                     mass_err_b=mass_b, min_rho_b=float(np.nanmin(Rb.numpy())), n_nonfinite_b=int((~torch.isfinite(Rb[:, -1])).any(-1).any(-1).sum()))
            out[key][setname] = S
            print(f'{key:18s} {setname:12s} err32@10T {np.nanmedian(e32[:, -1]):.3e} | (a) {np.nanmedian(ea[:, -1]):.3e} | (b) {np.nanmedian(eb[:, -1]):.3e} | '
                  f'comm@T {np.nanmedian(comm[:, 20]):.3e} @10T {np.nanmedian(comm[:, -1]):.3e} | mass(b) {mass_b:.1e} min(b) {S["min_rho_b"]:.1e} | {time.time() - t0:.0f}s', flush=True)
    json.dump(jclean(out), open(os.path.join(ODIR, 'E2_coarse.json'), 'w'))
    plot_e2(out, args.sets)


def plot_e2(out, sets):
    import matplotlib; matplotlib.use('Agg'); import matplotlib.pyplot as plt
    cols = dict(tcme='#1b6ca8', fisher_hodge='#d1495b', flux_gnn='#3b8b5a', sqra_oracle='#777777')
    labs = dict(tcme='TC-ME', fisher_hodge='Fisher-Hodge', flux_gnn='Flux-GNN', sqra_oracle='SQRA oracle (untrained, exact drift)')
    fig, axs = plt.subplots(1, len(sets), figsize=(5.2 * len(sets), 4.0), squeeze=False)
    t = np.arange(201) * DT
    for ax, sname in zip(axs[0], sets):
        for m in ['tcme', 'fisher_hodge', 'flux_gnn', 'sqra_oracle']:
            ks = [k for k in out if (k == m or k.startswith(m + '_s')) and sname in out[k]]
            if not ks:
                continue
            Cs = np.stack([np.asarray(out[k][sname]['comm_med'], float) for k in ks])
            ax.plot(t[1:], Cs.mean(0)[1:], color=cols[m], lw=2, ls='--' if m == 'sqra_oracle' else '-', label=labs[m])
            for c in Cs:
                ax.plot(t[1:], c[1:], color=cols[m], lw=0.6, alpha=0.4)
        ax.set_yscale('log'); ax.set_xlabel('t  (T_train = 1)'); ax.set_title(sname)
        ax.grid(alpha=0.3, which='both')
    axs[0][0].set_ylabel(r'commutation error $\|\rho_a-\rho_b\|/\|\rho_a\|$ (median over traj.)')
    axs[0][0].legend(fontsize=8)
    fig.suptitle('E2: lump-then-roll-out (16x16) vs roll-out-then-lump (32x32), S2')
    fig.tight_layout(); fig.savefig(os.path.join(ODIR, 'E2_commutation.png'), dpi=150)


# ------------------------------------------------------------------ summary
def ms(v, fmt='{:.3g}'):
    v = np.array([x for x in v if x is not None], float)
    if len(v) == 0:
        return 'n/a'
    if len(v) == 1:
        return fmt.format(v[0])
    return (fmt + '±' + fmt).format(v.mean(), v.std())


def summary():
    J = {os.path.basename(f)[:-5]: json.load(open(f)) for f in glob.glob(os.path.join(ODIR, 'S2_*.json')) if 'smoke' not in f}
    L = ['# TC-ME experiments on S2 (Fokker-Planck + divergence-free drift)', '',
         'All numbers: median over 16 test trajectories, then mean±std over seeds (2 seeds). 2500 training iterations, '
         'train.py protocol (rollout loss K in [4,8], Adam 1e-3 cosine, batch 8). 10T = t = 10 (200 steps).', '']
    LAB = dict(tcme='TC-ME', fisher_hodge='Fisher-Hodge', flux_gnn='Flux-GNN')
    for title, tag in [('E1: full training set (64 trajectories)', ''), ('E3: 8 training trajectories', '_n8')]:
        L += [f'## {title}', '', '| model | params | 10T ID | 10T OOD-pot | 10T OOD-init | T ID | max mass err | min rho | traj w/ rho<0 | curl J err @10T (ID / OOD-pot) | J err at true steady rho (ID / OOD-pot) | final train loss | train time (s) |',
              '|---|---|---|---|---|---|---|---|---|---|---|---|---|']
        for m in ['tcme', 'fisher_hodge', 'flux_gnn']:
            rs = [J[k] for k in sorted(J) if k.startswith(f'S2_{m}_s') and k.endswith(tag) and (tag or '_n' not in k)]
            if not rs:
                continue
            g = lambda s, t: [r['sets'][s]['relerr_median'][t] for r in rs]
            L.append(f'| {LAB[m]} | {rs[0]["params"]} | {ms(g("test_id", 200))} | {ms(g("test_oodpot", 200))} | {ms(g("test_oodinit", 200))} | {ms(g("test_id", 20))} | '
                     f'{max(r["sets"][s]["mass_err_max"] for r in rs for s in r["sets"]):.1e} | {min(r["sets"][s]["min_rho"] for r in rs for s in r["sets"]):.1e} | '
                     f'{sum(r["sets"][s]["n_traj_with_negative"] for r in rs for s in r["sets"])}/{16 * 3 * len(rs)} | '
                     f'{ms([r["sets"]["test_id"]["curl_flux_end_relerr_median"] for r in rs])} / {ms([r["sets"]["test_oodpot"]["curl_flux_end_relerr_median"] for r in rs])} | '
                     f'{ms([np.median(r["sets"]["test_id"].get("J_ss_relerr_at_refrho", [np.nan])) for r in rs])} / {ms([np.median(r["sets"]["test_oodpot"].get("J_ss_relerr_at_refrho", [np.nan])) for r in rs])} | '
                     f'{ms([r["final_train_loss"] for r in rs])} | {ms([r["train_time_s"] for r in rs], "{:.0f}")} |')
        L.append('')
    # thermo
    rs = [J[k] for k in sorted(J) if k.startswith('S2_tcme_s') and '_n' not in k and 'thermo' in J[k]]   # E1 runs only
    if rs:
        L += ['## E1: thermodynamic readout of the learned TC-ME generator (E1 runs, 2 seeds)', '',
              'pi = stationary state of the learned Q (fixed point of pi = stat(Q(pi))); reference steady state = 64x64 finite-volume stationary solve '
              '(exact linear solve, coarse-grained to 32x32). sigma_model = sum_e J_e f_e (probability units, = h^2 sum_e J f in density units); '
              'sigma_ref = sum |J|^2/(D rho) h^2 on the 64x64 grid ("fine") or with the coarse-grained J, rho on 32x32 ("coarse"). '
              'excess fraction = sum ell (d phi)^2 / sigma with the ell-weighted Hodge projection (hodge_lib.hodge); "Euclid" = same with the unweighted projection.', '',
              '| quantity | ID | OOD-pot |', '|---|---|---|']
        rows = [('J err at t=10, true rho_end fed in', 'J_end_relerr_at_refrho'), ('J err at t=10, model rho_end', 'J_end_relerr_at_modelrho'),
                ('curl J err at t=10, true rho_end', 'curlJ_end_relerr_at_refrho'),
                ('rho_ss (model pi) rel err', 'rho_ss_relerr'), ('J_ss (model pi) rel err', 'J_ss_relerr'), ('curl J_ss rel err', 'curlJ_ss_relerr'),
                ('sigma_model(pi) / sigma_ref fine', 'sigma_ratio_ss_fine'), ('sigma_model(pi) / sigma_ref coarse', 'sigma_ratio_ss_coarse'),
                ('sigma_model(rho_end) / sigma_ref(J_end) coarse, t=10', 'sigma_ratio_end'),
                ('sigma_model(pi) absolute', 'sigma_model_ss'), ('sigma_ref fine absolute', 'sigma_ref_ss_fine'),
                ('excess frac at pi (ell-weighted)', 'excess_frac_at_pi'), ('excess frac at pi (Euclid proj.)', 'excess_euclid_frac_at_pi'),
                ('excess frac at true rho_ss (ell)', 'excess_frac_at_refss'), ('excess frac at true rho_ss (Euclid)', 'excess_euclid_frac_at_refss'),
                ('excess frac at true rho(t=10) (ell)', 'excess_frac_at_Yend'),
                ('|b| / rms(A) (harmonic part of A)', 'harmonic_b_over_rmsA'), ('A vs dpsi rel err (edge)', 'A_vs_dpsi_relerr'),
                ('curl A vs curl dpsi rel err (face cycle affinities)', 'curlA_vs_curl_dpsi_relerr'), ('regression slope A ~ dpsi', 'A_dpsi_regression_slope')]
        for lab, k in rows:
            L.append(f'| {lab} | ' + ' | '.join(ms([np.median(r['thermo'][s][k]) for r in rs], '{:.3g}') for s in ['test_id', 'test_oodpot']) + ' |')
        L.append('| learned harmonic velocity v = b/h (units D/length; truth 0) | ' + ' | '.join(
            '; '.join(f'({r["thermo"][s]["v"][0]:.2e}, {r["thermo"][s]["v"][1]:.2e})' for r in rs) for s in ['test_id', 'test_oodpot']) + ' |')
        L.append('| learned alpha / gam | ' + '; '.join(f'{r["thermo"]["test_id"]["alpha"]:.3f} / {r["thermo"]["test_id"]["gam"]:.3f}' for r in rs) + ' | |')
        L.append('')
    # E2
    p = os.path.join(ODIR, 'E2_coarse.json')
    if os.path.exists(p):
        E = json.load(open(p))
        L += ['## E2: coarse-graining consistency (32x32 -> 16x16, same trained weights)', '',
              '(a) = roll out on 32x32 then lump; (b) = lump inputs, roll out on 16x16. Errors vs the lumped reference. '
              'comm = ||rho_a - rho_b|| / ||rho_a||. SQRA oracle = untrained TC-ME with k = D/h^2, A = dpsi, phi = U/D (the exact-drift discretisation): '
              'its comm error is the floor set by discretisation alone.', '']
        for sname in [s for s in ['test_id', 'test_oodpot', 'test_oodinit'] if any(s in v for v in E.values())]:
            L += [f'### {sname}', '', '| model | err 32x32 @10T | err (a) @10T | err (b) @10T | comm @T | comm @2T | comm @5T | comm @10T | mass err (b) | min rho (b) |', '|---|---|---|---|---|---|---|---|---|---|']
            for m in ['tcme', 'fisher_hodge', 'flux_gnn', 'sqra_oracle']:
                ks = [k for k in E if (k == m or k.startswith(m + '_s')) and sname in E[k]]
                if not ks:
                    continue
                g = lambda f, t: [E[k][sname][f][t] for k in ks]
                L.append(f'| {m} | {ms(g("err32_med", 200))} | {ms(g("errA_med", 200))} | {ms(g("errB_med", 200))} | ' +
                         ' | '.join(ms(g('comm_med', t)) for t in TS) +
                         f' | {max(E[k][sname]["mass_err_b"] for k in ks):.1e} | {min(E[k][sname]["min_rho_b"] for k in ks):.1e} |')
            L.append('')
        L += ['![E2](E2_commutation.png)', '']
    open(os.path.join(ODIR, 'summary.md'), 'w').write('\n'.join(L))
    print('\n'.join(L))


if __name__ == '__main__':
    p = argparse.ArgumentParser()
    p.add_argument('--seeds', type=int, nargs='+', default=[0, 1])
    p.add_argument('--sets', nargs='+', default=['test_id', 'test_oodpot'])
    p.add_argument('--summary', action='store_true')
    args = p.parse_args()
    if args.summary:
        summary()
    else:
        run_e2(args)
