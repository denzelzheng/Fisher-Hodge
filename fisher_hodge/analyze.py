"""Aggregate runs/main/*.json -> runs/summary.json, runs/summary.md, runs/*.png"""
import json, glob, os
import numpy as np
import torch
import matplotlib; matplotlib.use('Agg')
import matplotlib.pyplot as plt
from common import RUNS, DT

MODELS = ['fisher_hodge', 'euclid_gf', 'gnn_ode']
LABEL = {'fisher_hodge': 'Fisher-Hodge (M)', 'euclid_gf': 'Euclid GF (B2)', 'gnn_ode': 'GNN-ODE (B1)'}
COL = {'fisher_hodge': '#2a78d6', 'euclid_gf': '#eb6834', 'gnn_ode': '#1baf7a'}
SETS = ['test_id', 'test_oodpot', 'test_oodinit']
SETLAB = {'test_id': 'ID (new init, same dist.)', 'test_oodpot': 'OOD potential / field', 'test_oodinit': 'OOD initial condition'}
TS = [20, 40, 100, 200]
plt.rcParams.update({'font.size': 9, 'axes.spines.top': False, 'axes.spines.right': False, 'axes.grid': True,
                     'grid.color': '#e6e6e3', 'grid.linewidth': 0.6, 'axes.edgecolor': '#8a8986'})

R = {}
for f in sorted(glob.glob(os.path.join(RUNS, 'main', '*.json'))):
    r = json.load(open(f)); R.setdefault((r['sys'], r['model']), []).append(r)


def arr(x):
    return np.array([np.inf if v is None else v for v in x], dtype=float)


def ms(v):
    v = np.asarray(v, float)
    if not np.all(np.isfinite(v)):
        return f'{np.median(v):.3g}*' if np.isfinite(np.median(v)) else 'inf'
    return f'{v.mean():.3g}±{v.std():.2g}'


summary = {}
md = ['# Fisher-Hodge mechanism experiments - summary', '',
      'Relative L2 error = trajectory-median over 16 test trajectories, then mean±std over 3 seeds '
      '(`*` = some seed non-finite, median shown; `inf` = majority of trajectories blew up). '
      'T = T_train = 20 steps (t=1.0); 10T = t=10.', '']
for sysn in ['S1', 'S2', 'S3']:
    summary[sysn] = {}
    for st in SETS:
        md += [f'## {sysn} - {SETLAB[st]}', '']
        hdr = '| model | params | T | 2T | 5T | 10T | max mass err | min rho | traj w/ rho<0 | blown@10T |'
        extra = {'S1': ' F_true monotone | learned-Lyap monotone |', 'S2': ' curl(J) err @10T |', 'S3': ' front-speed err | max overshoot |'}[sysn]
        md += [hdr + extra, '|' + '---|' * (hdr.count('|') - 1 + extra.count('|'))]
        for m in MODELS:
            runs = R.get((sysn, m), [])
            if not runs:
                continue
            S = [r['sets'][st] for r in runs]
            e = {t: [arr(s['relerr_median'])[t] for s in S] for t in TS}
            mass = [np.nanmax(arr(s['mass_err_max'])) for s in S]
            minr = [np.nanmin(arr(s['min_rho_min'])) for s in S]
            nneg = sum(s['n_traj_with_negative'] for s in S); ntr = sum(s['n_traj'] for s in S)
            nbl = sum(s['n_blown_end'] for s in S)
            d = dict(seeds=[r['seed'] for r in runs], params=runs[0]['params'],
                     relerr={str(t): e[t] for t in TS}, mass_err_max=mass, min_rho=minr, n_traj_negative=nneg,
                     n_traj_total=ntr, n_blown=nbl, train_time_s=[r['train_time_s'] for r in runs],
                     infer_time_s=[s['infer_time_s'] for s in S], substeps_per_DT_mean=[s['substeps_per_DT_mean'] for s in S],
                     substeps_per_DT_max=[s['substeps_per_DT_max'] for s in S], final_train_loss=[r['final_train_loss'] for r in runs])
            row = f'| {LABEL[m]} | {d["params"]} | ' + ' | '.join(ms(e[t]) for t in TS) + \
                  f' | {max(mass):.2e} | {min(minr):.2e} | {nneg}/{ntr} | {nbl}/{ntr} |'
            if sysn == 'S1':
                d['F_true_monotone_frac'] = [s['true_F_monotone_frac_model'] for s in S]
                d['F_truth_monotone_frac'] = [s['true_F_monotone_frac_truth'] for s in S]
                d['learned_lyap_monotone_frac'] = [s.get('learned_lyapunov_monotone_frac') for s in S]
                ly = d['learned_lyap_monotone_frac']
                row += f' {np.mean(d["F_true_monotone_frac"]):.2f} | ' + (f'{np.mean(ly):.2f} |' if ly[0] is not None else 'n/a |')
            if sysn == 'S2':
                c = [s.get('curl_flux_end_relerr_median') for s in S]
                d['curl_relerr'] = c
                row += (f' {ms(arr(c))} |' if c[0] is not None or len(c) > 1 and c[1] is not None else ' n/a |') if 'curl_flux_end_relerr_median' in S[0] else ' n/a |'
            if sysn == 'S3':
                d['front_speed_relerr'] = [s['front_speed_relerr_median'] for s in S]
                d['overshoot'] = [s['overshoot_max'] for s in S]
                row += f' {ms(arr(d["front_speed_relerr"]))} | {max(arr(d["overshoot"])):.2e} |'
            md.append(row)
            summary[sysn].setdefault(st, {})[m] = d
        md.append('')
    # timing table
    md += [f'## {sysn} - cost', '', '| model | params | train time (s) | rollout 16x200 steps (s) | substeps/DT mean (max) |', '|---|---|---|---|---|']
    for m in MODELS:
        d = summary[sysn].get('test_id', {}).get(m)
        if d:
            md.append(f'| {LABEL[m]} | {d["params"]} | {np.mean(d["train_time_s"]):.0f} | {np.mean(d["infer_time_s"]):.1f} | '
                      f'{np.mean(d["substeps_per_DT_mean"]):.1f} ({max(d["substeps_per_DT_max"])}) |')
    md.append('')

    # ---------------- rollout-error figure
    fig, ax = plt.subplots(3, 3, figsize=(12, 8.5), sharex=True)
    t = np.arange(201) * DT
    for j, st in enumerate(SETS):
        for m in MODELS:
            runs = R.get((sysn, m), [])
            if not runs:
                continue
            S = [r['sets'][st] for r in runs]
            for i, key in enumerate(['relerr_median', 'mass_err_max', 'neg_frac_mean']):
                Y = np.stack([arr(s[key]) for s in S])
                Y = np.where(np.isfinite(Y), Y, np.nan)
                if i < 2:
                    Y = np.maximum(Y, 1e-9)
                Y = Y[:, 1:]; tt = t[1:]          # skip t=0 (exact initial condition)
                med = np.nanmedian(Y, 0)
                ax[i, j].plot(tt, med, color=COL[m], lw=2, label=LABEL[m])
                ax[i, j].fill_between(tt, np.nanmin(Y, 0), np.nanmax(Y, 0), color=COL[m], alpha=0.15, lw=0)
        for i in range(3):
            ax[i, j].axvline(1.0, color='#8a8986', ls='--', lw=1)
        ax[0, j].set_yscale('log'); ax[1, j].set_yscale('log')
        ax[0, j].set_title(SETLAB[st])
        ax[2, j].set_xlabel('rollout time t  (T_train = 1, dashed)')
    ax[0, 0].set_ylabel('relative L2 error\n(median traj., band = seed range)')
    ax[1, 0].set_ylabel('|mass - mass_true| / mass_true' if sysn == 'S3' else '|sum rho - 1|  (max over traj.)')
    ax[2, 0].set_ylabel('fraction of nodes with rho < 0')
    ax[0, 0].legend(frameon=False, loc='lower right')
    fig.suptitle({'S1': 'S1 Fokker-Planck', 'S2': 'S2 non-equilibrium (drift + divergence-free flow)', 'S3': 'S3 Fisher-KPP (reaction-diffusion)'}[sysn])
    fig.tight_layout(); fig.savefig(os.path.join(RUNS, f'{sysn}_rollout_error.png'), dpi=140); plt.close(fig)

    # ---------------- snapshot figure at t = 10T (S3 also at 2T, before the domain saturates)
    data = torch.load(os.path.join(RUNS, 'data', f'{sysn}.pt'))
    for (tstep, sidx, tlab) in ([(200, 4, '10T')] + ([(40, 2, '2T')] if sysn == 'S3' else [])):
     fig, ax = plt.subplots(3, 4, figsize=(10, 8.4), constrained_layout=True)
     for i, st in enumerate(SETS):
        Yt = data[st]['rho'][0, tstep].numpy()
        vmax = Yt.max()
        cmap = plt.get_cmap('magma').copy(); cmap.set_under('#22d3ee'); cmap.set_bad('#9ca3af')
        panels = [('truth', Yt)]
        for m in MODELS:
            f = os.path.join(RUNS, 'main', f'{sysn}_{m}_s0_snap.npz')
            if os.path.exists(f):
                panels.append((LABEL[m], np.load(f)[st][0, sidx]))
        for k, (nm, X) in enumerate(panels):
            im = ax[i, k].imshow(X, origin='lower', cmap=cmap, vmin=0, vmax=vmax, extent=[0, 1, 0, 1])
            ax[i, k].set_xticks([]); ax[i, k].set_yticks([]); ax[i, k].grid(False)
            if k == 0:
                ax[i, k].set_ylabel(SETLAB[st]); ax[i, k].set_title(f'truth t={tstep*DT:g}   max {vmax:.2f}')
            else:
                fin = np.isfinite(X).all()
                err = np.linalg.norm(X - Yt) / np.linalg.norm(Yt) if fin else np.inf
                ax[i, k].set_title(f'{nm}\nrel err {err:.2e}  min {np.nanmin(X) if fin else np.nan:.1e}', fontsize=8)
        fig.colorbar(im, ax=ax[i, :].tolist(), shrink=0.8, extend='min')
     fig.suptitle(f'{sysn}: density at t = {tlab} (seed 0, test trajectory 0); cyan = negative density, grey = NaN')
     fig.savefig(os.path.join(RUNS, f'{sysn}_snapshot_t{tlab}.png'), dpi=140); plt.close(fig)

json.dump(summary, open(os.path.join(RUNS, 'summary.json'), 'w'), indent=1, default=float)
open(os.path.join(RUNS, 'summary.md'), 'w').write('\n'.join(md))
print('\n'.join(md))
