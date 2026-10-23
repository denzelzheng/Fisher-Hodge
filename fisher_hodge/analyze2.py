"""Aggregate stress-test runs (runs/stress/main/*.json + the first-round runs/main/*.json) ->
runs/stress/summary.md, summary.json, *_rollout_error.png, gnn_loss_plateau.png, S3_saturation.png.
python3 analyze2.py [--smoke]   (smoke: use runs/stress/smoke, for testing the script)"""
import json, glob, os, sys, argparse
import numpy as np
import torch
import matplotlib; matplotlib.use('Agg')
import matplotlib.pyplot as plt
from common import RUNS, DT

ap = argparse.ArgumentParser(); ap.add_argument('--smoke', action='store_true'); A = ap.parse_args()
OLD = os.path.join(RUNS, 'main'); NEW = os.path.join(RUNS, 'stress', 'smoke' if A.smoke else 'main'); OUT = os.path.join(RUNS, 'stress')
SETS = ['test_id', 'test_oodpot', 'test_oodinit']
SETLAB = {'test_id': 'ID (new init, same dist.)', 'test_oodpot': 'OOD potential / field', 'test_oodinit': 'OOD initial condition'}
TS = [20, 40, 100, 200]
plt.rcParams.update({'font.size': 9, 'axes.spines.top': False, 'axes.spines.right': False, 'axes.grid': True,
                     'grid.color': '#e6e6e3', 'grid.linewidth': 0.6, 'axes.edgecolor': '#8a8986'})

# ---------------------------------------------------------------- load
R = {}   # (src, sys, model, tag) -> [run json]
for src, d in [('old', OLD), ('new', NEW)]:
    for f in sorted(glob.glob(os.path.join(d, '*.json'))):
        r = json.load(open(f))
        tag = r.get('tag', '') if src == 'new' else ''
        if A.smoke:
            tag = tag.replace('_smoke', '')
        R.setdefault((src, r['sys'], r['model'], tag), []).append(r)

LAB = {('old', 'fisher_hodge', ''): 'Fisher-Hodge (M)', ('new', 'fisher_hodge', ''): 'Fisher-Hodge (M)',
       ('old', 'euclid_gf', ''): 'Euclid GF (B2)', ('new', 'euclid_gf', ''): 'Euclid GF (B2)',
       ('old', 'gnn_ode', ''): 'GNN-ODE (B1, 4k it)', ('new', 'gnn_ode', '_long'): 'GNN-ODE (B1, 12k it, warmup+cos)',
       ('new', 'flux_gnn', ''): 'Flux-GNN pos+mass (B3)', ('new', 'wgf_rev', ''): 'Reversible WGF (B4 = FH, no cycle, kappa=1)',
       ('new', 'jko_amort', ''): 'JKO free energy, amortised (B5, RML-style)', ('new', 'jko_inst', ''): 'JKO free energy, per-trajectory fit (B5\', RML transductive)',
       ('new', 'fisher_hodge_fr', ''): 'Fisher-Hodge + FR reaction, free sigma-MLP (M2)', ('new', 'fisher_hodge_fr_u', ''): 'M2 with sigma(U) only (ablation)',
       ('new', 'fisher_hodge_wfr', ''): 'Fisher-Hodge + reservoir-edge WFR reaction (M2\')'}
COL = {'Fisher-Hodge (M)': '#2a78d6', 'Euclid GF (B2)': '#eb6834', 'GNN-ODE (B1, 4k it)': '#9bd4bd', 'GNN-ODE (B1, 12k it, warmup+cos)': '#1baf7a',
       'Flux-GNN pos+mass (B3)': '#8e44ad', 'Reversible WGF (B4 = FH, no cycle, kappa=1)': '#7f8c8d',
       'JKO free energy, amortised (B5, RML-style)': '#c0392b', 'JKO free energy, per-trajectory fit (B5\', RML transductive)': '#e67e22',
       'Fisher-Hodge + FR reaction, free sigma-MLP (M2)': '#16a085', 'M2 with sigma(U) only (ablation)': '#2c3e50',
       'Fisher-Hodge + reservoir-edge WFR reaction (M2\')': '#d35400',
       'Fisher-Hodge (M), round-1 reaction, trained on S3b (super-saturated ICs)': '#95a5a6', 'M2 sigma-MLP reaction, trained on S3b': '#1abc9c', 'M2\' reservoir-edge reaction, trained on S3b': '#f39c12'}


def arr(x):
    return np.array([np.inf if v is None else v for v in x], dtype=float)


def ms(v):
    v = np.asarray(v, float)
    if len(v) == 1:
        return f'{v[0]:.3g}' if np.isfinite(v[0]) else 'inf'
    if not np.all(np.isfinite(v)):
        return f'{np.median(v):.3g}*' if np.isfinite(np.median(v)) else 'inf'
    return f'{v.mean():.3g}±{v.std():.2g}'


def row_label(key, noise_tag=''):
    src, m, tag = key
    base = LAB.get((src, m, tag.replace(noise_tag, '') if noise_tag else tag), f'{m}{tag}')
    return base


def tables(sysn, rows, sysclass, extra_cols=True):
    """rows: list of (label, runs).  returns md lines + dict"""
    md, summ = [], {}
    for st in SETS:
        md += [f'### {sysn} - {SETLAB[st]}', '']
        hdr = '| model | params | T | 2T | 5T | 10T | max mass err | min rho | traj w/ rho<0 | blown@10T | peak err@10T |'
        extra = {'S1': ' learned-Lyap monotone |', 'S2': ' curl(J) err @10T |', 'S3': ' front-speed err | max overshoot | Linf sat. err@10T |'}[sysclass]
        md += [hdr + extra, '|' + '---|' * (hdr.count('|') - 1 + extra.count('|'))]
        for lab, runs in rows:
            if not runs:
                continue
            S = [r['sets'][st] for r in runs]
            e = {t: [arr(s['relerr_median'])[t] for s in S] for t in TS}
            mass = [np.nanmax(arr(s['mass_err_max'])) for s in S]
            minr = [np.nanmin(arr(s['min_rho_min'])) for s in S]
            nneg = sum(s['n_traj_with_negative'] for s in S); ntr = sum(s['n_traj'] for s in S); nbl = sum(s['n_blown_end'] for s in S)
            pk = [arr(s['peak_relerr_median'])[-1] for s in S] if 'peak_relerr_median' in S[0] else None
            d = dict(seeds=[r['seed'] for r in runs], params=runs[0]['params'], relerr={str(t): e[t] for t in TS}, mass_err_max=mass,
                     min_rho=minr, n_traj_negative=nneg, n_traj_total=ntr, n_blown=nbl, train_time_s=[r['train_time_s'] for r in runs],
                     infer_time_s=[s['infer_time_s'] for s in S], final_train_loss=[r.get('final_train_loss') for r in runs],
                     iters=runs[0].get('iters', runs[0]['args'].get('iters')), peak_relerr_10T=pk)
            row = f'| {lab} | {d["params"]} | ' + ' | '.join(ms(e[t]) for t in TS) + \
                  f' | {max(mass):.2e} | {min(minr):.2e} | {nneg}/{ntr} | {nbl}/{ntr} | ' + (ms(pk) if pk is not None else 'n/a') + ' |'
            if sysclass == 'S1':
                ly = [s.get('learned_lyapunov_monotone_frac') for s in S]; d['learned_lyap_monotone_frac'] = ly
                row += (f' {np.mean([v for v in ly if v is not None]):.2f} |' if any(v is not None for v in ly) else ' n/a |')
            if sysclass == 'S2':
                c = [s.get('curl_flux_end_relerr_median') for s in S]; d['curl_relerr'] = c
                row += (f' {ms(arr(c))} |' if any(v is not None for v in c) else ' n/a |')
            if sysclass == 'S3':
                d['front_speed_relerr'] = [s['front_speed_relerr_median'] for s in S]; d['overshoot'] = [s['overshoot_max'] for s in S]
                d['sat_err'] = [s.get('sat_err_10T_median') for s in S]
                row += f' {ms(arr(d["front_speed_relerr"]))} | {max(arr(d["overshoot"])):.2e} | ' + (ms(arr(d['sat_err'])) if d['sat_err'][0] is not None else 'n/a') + ' |'
            md.append(row); summ.setdefault(st, {})[lab] = d
        md.append('')
    md += [f'### {sysn} - cost', '', '| model | params | iters | final train loss | train time (s) | rollout 16x200 steps (s) |', '|---|---|---|---|---|---|']
    for lab, runs in rows:
        d = summ.get('test_id', {}).get(lab)
        if d:
            fl = [v for v in d['final_train_loss'] if v is not None]
            md.append(f'| {lab} | {d["params"]} | {d["iters"]} | {np.mean(fl):.2e} | {np.mean(d["train_time_s"]):.0f} | {np.mean(d["infer_time_s"]):.1f} |' if fl else
                      f'| {lab} | {d["params"]} | - | - | {np.mean(d["train_time_s"]):.0f} | {np.mean(d["infer_time_s"]):.1f} |')
    md.append('')
    return md, summ


def rollout_fig(sysn, rows, fname, title, masslab):
    fig, ax = plt.subplots(3, 3, figsize=(12, 8.5), sharex=True)
    t = np.arange(201) * DT
    for j, st in enumerate(SETS):
        for lab, runs in rows:
            if not runs:
                continue
            S = [r['sets'][st] for r in runs]
            for i, key in enumerate(['relerr_median', 'mass_err_max', 'neg_frac_mean']):
                Y = np.stack([arr(s[key]) for s in S]); Y = np.where(np.isfinite(Y), Y, np.nan)
                if i < 2:
                    Y = np.maximum(Y, 1e-9)
                Y = Y[:, 1:]; tt = t[1:]
                ax[i, j].plot(tt, np.nanmedian(Y, 0), color=COL.get(lab, '#333'), lw=1.8, label=lab)
                ax[i, j].fill_between(tt, np.nanmin(Y, 0), np.nanmax(Y, 0), color=COL.get(lab, '#333'), alpha=0.12, lw=0)
        for i in range(3):
            ax[i, j].axvline(1.0, color='#8a8986', ls='--', lw=1)
        ax[0, j].set_yscale('log'); ax[1, j].set_yscale('log'); ax[0, j].set_title(SETLAB[st]); ax[2, j].set_xlabel('rollout time t  (T_train = 1, dashed)')
    ax[0, 0].set_ylabel('relative L2 error\n(median traj., band = seed range)'); ax[1, 0].set_ylabel(masslab); ax[2, 0].set_ylabel('fraction of nodes with rho < 0')
    ax[0, 0].legend(frameon=False, loc='lower right', fontsize=7)
    fig.suptitle(title); fig.tight_layout(); fig.savefig(os.path.join(OUT, fname), dpi=140); plt.close(fig)


def get(src, sysn, model, tag=''):
    return R.get((src, sysn, model, tag), [])


SYSNAME = {'S1': 'S1 Fokker-Planck', 'S2': 'S2 non-equilibrium (drift + divergence-free flow)', 'S3': 'S3 Fisher-KPP (reaction-diffusion)',
           'S4': 'S4 nonlinear mobility  m(rho) = rho^1.5/sqrt(1+rho/5)  [misspecified]', 'S5': 'S5 exclusion  m(rho) = rho(1-rho)  [misspecified]',
           'S6': 'S6 nonlocal aggregation  U - chi W*rho  [misspecified]', 'S3b': 'S3b'}
summary = {}
md = ['# Fisher-Hodge stress tests (2026-10-19) - summary', '',
      'Relative L2 error = trajectory-median over 16 test trajectories, then mean±std over seeds (`*` = some seed non-finite, median shown; '
      '`inf` = majority blew up). T = T_train = 20 steps (t=1); 10T = t=10. "peak err" = |max rho_model - max rho_true| / max rho_true at 10T. '
      'Rows marked (M)/(B2)/(B1, 4k it) on S1-S3 are the first-round runs (same data, same code path); everything else is new.', '']

# ---- 1. strong baselines on S1/S2/S3
md += ['## 1. Strong baselines on S1 / S2 / S3', '']
for sysn in ['S1', 'S2', 'S3']:
    rows = [(LAB[('old', 'fisher_hodge', '')], get('old', sysn, 'fisher_hodge'))]
    if sysn == 'S3':
        rows += [(LAB[('new', 'fisher_hodge_wfr', '')], get('new', sysn, 'fisher_hodge_wfr')), (LAB[('new', 'fisher_hodge_fr', '')], get('new', sysn, 'fisher_hodge_fr')),
                 ('Fisher-Hodge (M), round-1 reaction, trained on S3b (super-saturated ICs)', get('new', 'S3b', 'fisher_hodge')),
                 ('M2 sigma-MLP reaction, trained on S3b', get('new', 'S3b', 'fisher_hodge_fr')),
                 ('M2\' reservoir-edge reaction, trained on S3b', get('new', 'S3b', 'fisher_hodge_wfr'))]
    rows += [(LAB[('new', 'flux_gnn', '')], get('new', sysn, 'flux_gnn')), (LAB[('old', 'euclid_gf', '')], get('old', sysn, 'euclid_gf')),
             (LAB[('new', 'gnn_ode', '_long')], get('new', sysn, 'gnn_ode', '_long')), (LAB[('old', 'gnn_ode', '')], get('old', sysn, 'gnn_ode'))]
    if sysn in ('S1', 'S2'):
        rows += [(LAB[('new', 'wgf_rev', '')], get('new', sysn, 'wgf_rev')), (LAB[('new', 'jko_amort', '')], get('new', sysn, 'jko_amort')),
                 (LAB[('new', 'jko_inst', '')], get('new', sysn, 'jko_inst'))]
    m_, s_ = tables(sysn, rows, sysn); md += m_; summary[f'{sysn}_strong'] = s_
    rollout_fig(sysn, rows, f'{sysn}_strong_rollout_error.png', SYSNAME[sysn] + ' - strong baselines',
                '|mass - mass_true| / mass_true' if sysn == 'S3' else '|sum rho - 1| (max over traj.)')

# ---- 2. misspecified physics
md += ['## 2. Misspecified physics (reference dynamics outside the Fisher-Hodge hypothesis class)', '']
for sysn in ['S4', 'S5', 'S6']:
    rows = [(LAB[('new', 'fisher_hodge', '')], get('new', sysn, 'fisher_hodge')), (LAB[('new', 'flux_gnn', '')], get('new', sysn, 'flux_gnn')),
            (LAB[('new', 'euclid_gf', '')], get('new', sysn, 'euclid_gf')), (LAB[('new', 'gnn_ode', '_long')], get('new', sysn, 'gnn_ode', '_long'))]
    m_, s_ = tables(sysn, rows, 'S1'); md += m_; summary[sysn] = s_
    rollout_fig(sysn, rows, f'{sysn}_rollout_error.png', SYSNAME[sysn], '|mass - mass_true| / mass_true')

# ---- 3. noisy observations
md += ['## 3. Noisy training observations (multiplicative Gaussian noise on training snapshots only; test data clean)', '']
for sysn, tag, lvl in [('S1', '_n01', '1%'), ('S1', '_n05', '5%'), ('S2', '_n05', '5%')]:
    rows = [(f'{LAB[("new", "fisher_hodge", "")]} [clean, round 1]', get('old', sysn, 'fisher_hodge')),
            (f'{LAB[("new", "fisher_hodge", "")]} [noise {lvl}]', get('new', sysn, 'fisher_hodge', tag)),
            (f'{LAB[("new", "flux_gnn", "")]} [noise {lvl}]', get('new', sysn, 'flux_gnn', tag)),
            (f'{LAB[("new", "euclid_gf", "")]} [clean, round 1]', get('old', sysn, 'euclid_gf')),
            (f'{LAB[("new", "euclid_gf", "")]} [noise {lvl}]', get('new', sysn, 'euclid_gf', tag)),
            (f'{LAB[("new", "gnn_ode", "_long")]} [noise {lvl}]', get('new', sysn, 'gnn_ode', tag + '_long'))]
    for lab, _ in rows:
        base = lab.split(' [')[0]
        if lab not in COL:
            COL[lab] = COL.get(base, '#333') if 'noise' in lab else '#bbbbbb'
    m_, s_ = tables(f'{sysn} noise {lvl}', rows, sysn); md += m_; summary[f'{sysn}{tag}'] = s_
    rollout_fig(sysn, rows, f'{sysn}{tag}_rollout_error.png', SYSNAME[sysn] + f' - training noise {lvl}', '|sum rho - 1| (max over traj.)')

# ---- 4. GNN convergence figure
fig, ax = plt.subplots(1, 3, figsize=(12, 3.4))
for i, sysn in enumerate(['S1', 'S2', 'S3']):
    for key, lab, c in [(('old', sysn, 'gnn_ode', ''), '4k it (round 1, cosine)', '#9bd4bd'), (('new', sysn, 'gnn_ode', '_long'), '12k it (warmup + cosine)', '#1baf7a'),
                        (('new', sysn, 'flux_gnn', ''), 'Flux-GNN 4k it', '#8e44ad'), (('old', sysn, 'fisher_hodge', ''), 'Fisher-Hodge 4k it', '#2a78d6')]:
        for k, r in enumerate(R.get(key, [])):
            h = np.array(r['loss_hist'], float)   # every 20 iters
            if len(h) == 0:
                continue
            w = max(1, len(h) // 60)
            hs = np.convolve(h, np.ones(w) / w, mode='valid')
            ax[i].plot(np.arange(len(hs)) * 20 + 10 * w, hs, color=c, lw=1.2, alpha=0.9, label=lab if k == 0 else None)
    ax[i].set_yscale('log'); ax[i].set_title(sysn); ax[i].set_xlabel('iteration'); ax[i].set_ylabel('training loss (rel. L2, smoothed)')
ax[0].legend(frameon=False, fontsize=7)
fig.suptitle('Training-loss plateau check'); fig.tight_layout(); fig.savefig(os.path.join(OUT, 'gnn_loss_plateau.png'), dpi=140); plt.close(fig)

# ---- 5. S3 saturation figure: max density over time
try:
    data3 = torch.load(os.path.join(RUNS, 'data', 'S3.pt'))
    fig, ax = plt.subplots(1, 3, figsize=(12, 3.4))
    for j, st in enumerate(SETS):
        Y = data3[st]['rho'].numpy(); t = np.arange(Y.shape[1]) * DT
        ax[j].plot(t, np.median(Y.max((2, 3)), 0), color='k', lw=2, label='truth (median over traj.)')
        for key, lab in [(('new', 'S3', 'fisher_hodge_wfr', ''), LAB[('new', 'fisher_hodge_wfr', '')]), (('new', 'S3', 'fisher_hodge_fr', ''), LAB[('new', 'fisher_hodge_fr', '')]),
                         (('new', 'S3b', 'fisher_hodge_wfr', ''), 'M2\' reservoir-edge reaction, trained on S3b'), (('new', 'S3b', 'fisher_hodge_fr', ''), 'M2 sigma-MLP reaction, trained on S3b'),
                         (('new', 'S3', 'flux_gnn', ''), LAB[('new', 'flux_gnn', '')]), (('new', 'S3', 'gnn_ode', '_long'), LAB[('new', 'gnn_ode', '_long')])]:
            for k, r in enumerate(R.get(key, [])):
                f = os.path.join(NEW, r['name'] + '_snap.npz')
                if os.path.exists(f) and (st + '_peak') in np.load(f):
                    pk = np.load(f)[st + '_peak']
                    ax[j].plot(t, np.nanmedian(pk, 0), color=COL[lab], lw=1.2, alpha=0.8, label=lab if k == 0 else None)
        for k, r in enumerate(R.get(('old', 'S3', 'fisher_hodge', ''), [])):
            f = os.path.join(OLD, r['name'] + '_snap.npz')
            if os.path.exists(f):
                sn = np.load(f)[st]       # (B,5,n,n) at steps 0,20,40,100,200
                ax[j].plot(np.array([0, 20, 40, 100, 200]) * DT, np.nanmedian(sn.max((2, 3)), 0), 'o--', color=COL['Fisher-Hodge (M)'], lw=1, ms=4,
                           label='Fisher-Hodge (M), round-1 reaction' if k == 0 else None)
        ax[j].axhline(1.0, color='#8a8986', lw=0.8); ax[j].set_title(SETLAB[st]); ax[j].set_xlabel('t'); ax[j].set_ylabel('max_x rho (median over traj.)')
    ax[0].legend(frameon=False, fontsize=6.5)
    fig.suptitle('S3 saturation: carrying capacity is 1'); fig.tight_layout(); fig.savefig(os.path.join(OUT, 'S3_saturation.png'), dpi=140); plt.close(fig)
except Exception as ex:
    print('S3 saturation figure skipped:', ex)

json.dump(summary, open(os.path.join(OUT, 'summary.json'), 'w'), indent=1, default=float)
open(os.path.join(OUT, 'summary.md'), 'w').write('\n'.join(md))
print('\n'.join(md))
