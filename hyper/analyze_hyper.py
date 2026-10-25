"""Summary table + figures for the CRN benchmark.  python analyze_hyper.py"""
import os, sys, json
import numpy as np
import torch
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from hyperme import HyperME, RUNS, NS, DT, T_TEST, graph, build_graph_np
from baselines_hyper import CRNN, GNNODE3
from gen_crn import LOGK2, true_params, true_ab
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.patches import FancyArrowPatch

MD = os.path.join(RUNS, 'main')
MODELS = ['hyperme', 'hyperme_rev', 'crnn', 'gnn']
NAMES = {'hyperme': 'HyperME', 'hyperme_rev': 'B3 HyperME (A=0)', 'crnn': 'B1 CRNN', 'gnn': 'B2 GNN-ODE'}
SETS = ['test_id', 'test_oodG', 'test_oodU']
R = {}; RB = {}
for m in MODELS:
    for s in [0, 1]:
        f = os.path.join(MD, f'{m}_s{s}.json')
        if os.path.exists(f):
            R[(m, s)] = json.load(open(f))
        f = os.path.join(MD, f'{m}_k2r0.2_s{s}.json')
        if os.path.exists(f):
            RB[(m, s)] = json.load(open(f))
coarse = json.load(open(os.path.join(RUNS, 'coarse.json'))) if os.path.exists(os.path.join(RUNS, 'coarse.json')) else None
selft = json.load(open(os.path.join(RUNS, 'selftest.json'))) if os.path.exists(os.path.join(RUNS, 'selftest.json')) else None


def fmt(v, f='{:.3g}'):
    if v is None or (isinstance(v, float) and not np.isfinite(v)):
        return 'n/a'
    return f.format(v)


def both(m, getter, f='{:.3g}'):
    vals = []
    for s in [0, 1]:
        if (m, s) not in R:
            vals.append('–'); continue
        try:
            vals.append(fmt(getter(R[(m, s)]), f))
        except (KeyError, TypeError):
            vals.append('n/a')
    return ' / '.join(vals)


L = ['# HyperME on benchmark C1 (chemostat-driven spatial phosphorylation cycle, 8x8 periodic, S=3)', '',
     'Entries are `seed0 / seed1`. relL2 on the dynamic species (X, Xp), median over 16 trajectories; '
     'T = 20 steps (training window), 10T = 200 steps. ID: dG in [1,3], U std 0.7 (|k|<=2); OOD-dG: dG in [3,4]; '
     'OOD-U: U std 1.2 with |k|<=3. True local cycle affinity = dG + log(k2/k2\') = dG + %.3f.' % LOGK2, '']
L += ['## Accuracy', '', '| model | params | ID T | ID 10T | OOD-dG T | OOD-dG 10T | OOD-U T | OOD-U 10T | Gillespie: vs SSA / vs ODE (10T) | train s |',
      '|---|---|---|---|---|---|---|---|---|---|']
for m in MODELS:
    row = [NAMES[m], both(m, lambda r: r['params'], '{:d}')]
    for st in SETS:
        for t in ['20', '200']:
            row.append(both(m, lambda r: r['sets'][st]['relL2_XXp_median'][t]))
    row.append(both(m, lambda r: r['gillespie']['relL2_vs_gillespie_10T'], '{:.3f}') + ' ; ' + both(m, lambda r: r['gillespie']['relL2_vs_ode_10T'], '{:.3f}'))
    row.append(both(m, lambda r: r['train_time_s'], '{:.0f}'))
    L.append('| ' + ' | '.join(row) + ' |')
g0 = next(iter(R.values()))['gillespie']['noise_floor_ode_vs_gillespie_10T'] if R else float('nan')
L += ['', f'Gillespie (Omega=200) noise floor, ODE vs SSA at 10T: {g0:.3f}.', '']
L += ['## Structure / physics', '', '| model | max cons. err (kerS^T; E; X+Xp) over all test sets | traj with c<0 (ID/OOD-dG/OOD-U) | min c | sigma(t) rel err ID / OOD-dG | J(10T) corr all (rx; tr) ID | J(10T) rel err ID | affinity MAE ID / OOD-dG | A_cyc rel err ID |',
      '|---|---|---|---|---|---|---|---|---|']
for m in MODELS:
    def cons(r):
        return '%.1e; %.1e; %.1e' % tuple(max(r['sets'][st]['cons_err'][k] if r['sets'][st]['cons_err'][k] is not None else np.inf for st in SETS) for k in ['ker', 'E', 'XXp'])
    row = [NAMES[m]]
    row.append(' / '.join(cons(R[(m, s)]) if (m, s) in R else '–' for s in [0, 1]))
    row.append(' / '.join(','.join(str(R[(m, s)]['sets'][st]['n_traj_negative']) for st in SETS) if (m, s) in R else '–' for s in [0, 1]))
    row.append(both(m, lambda r: min(r['sets'][st]['min_c'] for st in SETS), '{:.2e}'))
    row.append(both(m, lambda r: r['sets']['test_id']['sigma_relerr_median']) + ' ; ' + both(m, lambda r: r['sets']['test_oodG']['sigma_relerr_median']))
    row.append(both(m, lambda r: r['sets']['test_id']['flux_corr_10T_median'], '{:.3f}') + ' (' + both(m, lambda r: r['sets']['test_id']['flux_corr_10T_reactions_median'], '{:.3f}') + '; ' + both(m, lambda r: r['sets']['test_id']['flux_corr_10T_transport_median'], '{:.3f}') + ')')
    row.append(both(m, lambda r: r['sets']['test_id']['flux_relerr_10T_median']))
    row.append(both(m, lambda r: r['sets']['test_id']['affinity_MAE']) + ' ; ' + both(m, lambda r: r['sets']['test_oodG']['affinity_MAE']))
    row.append(both(m, lambda r: r['sets']['test_id']['Acycle_relerr_median']))
    L.append('| ' + ' | '.join(row) + ' |')
L += ['', 'B3 has A = 0 by construction (affinity = 0, MAE = true affinity). B2 has no fluxes/EPR (n/a). '
      'J correlation is over all 384 hyperedges of each test trajectory at t = 10T (rx = R1/R2 edges, tr = transport edges).', '']
L += ['## Affinity identifiability (learned site-mean <z_v,A> vs true dG + log(k2/k2\'), pooled over the 3 test sets and both seeds)', '',
      'C1: k2\' = 0.02 (log(k2/k2\') = 3.91, reverse R2 flux ~1% of forward); control C1b: k2\' = 0.2 (log = 1.61), same everything else.', '',
      '| data | model | slope | intercept | Pearson r | MAE | sigma rel err ID | J corr ID | relL2 ID 10T | relL2 OOD-dG 10T |', '|---|---|---|---|---|---|---|---|---|---|']
for tag, RR in [('C1', R), ('C1b', RB)]:
    for m in ['hyperme', 'crnn', 'hyperme_rev']:
        ks = [(m, s) for s in [0, 1] if (m, s) in RR]
        if not ks:
            continue
        xt = np.concatenate([RR[k]['sets'][st]['affinity_true'] for k in ks for st in SETS])
        yl = np.concatenate([RR[k]['sets'][st]['affinity_learned_sitemean'] for k in ks for st in SETS])
        if np.std(yl) > 1e-9:
            sl, ic = np.polyfit(xt, yl, 1); rr = np.corrcoef(xt, yl)[0, 1]
        else:
            sl, ic, rr = 0.0, float(np.mean(yl)), float('nan')
        gs = lambda key, st='test_id': ' / '.join(fmt(RR[k]['sets'][st][key]) for k in ks)
        L.append(f'| {tag} | {NAMES[m]} | {sl:.3f} | {ic:.2f} | {fmt(rr)} | {np.mean([RR[k]["sets"][st]["affinity_MAE"] for k in ks for st in SETS]):.3f} | '
                 f'{gs("sigma_relerr_median")} | {gs("flux_corr_10T_median")} | '
                 + ' / '.join(fmt(RR[k]['sets']['test_id']['relL2_XXp_median']['200']) for k in ks) + ' | '
                 + ' / '.join(fmt(RR[k]['sets']['test_oodG']['relL2_XXp_median']['200']) for k in ks) + ' |')
L.append('')
if coarse:
    L += ['## Coarse-graining commutation (8x8 -> 4x4, same model on both graphs; X,Xp rel L2, median, ID test)', '',
          '| model | block-sum comm. T | block-sum comm. 10T | block-mean comm. T | block-mean comm. 10T | block-mean coarse-rollout vs coarsened truth 10T |', '|---|---|---|---|---|---|']
    for m in MODELS + ['truth']:
        row = [NAMES.get(m, 'true physics, naive coarse params')]
        for mode in ['sum', 'mean']:
            for t in ['20', '200']:
                if m == 'truth':
                    row.append(fmt(coarse[mode]['truth']['test_id'][t]))
                elif m in coarse[mode]:
                    row.append(' / '.join(fmt(coarse[mode][m]['test_id'][s]['comm'][t]) for s in ['s0', 's1'] if s in coarse[mode][m]['test_id']))
                else:
                    row.append('–')
        row.append('–' if m == 'truth' or m not in coarse['mean'] else ' / '.join(fmt(coarse['mean'][m]['test_id'][s]['coarse_vs_truth']['200']) for s in ['s0', 's1'] if s in coarse['mean'][m]['test_id']))
        L.append('| ' + ' | '.join(row) + ' |')
    L.append('')
if selft:
    L += ['## Property self-test (hyperme.py --selftest, random-init models, 4 traj x 200 steps)', '']
    for k, v in selft.items():
        L.append(f'- {k}: conservation rel err kerS^T {v["conservation"]["ker"]:.1e}, E {v["conservation"]["E"]:.1e}, X+Xp {v["conservation"]["XXp"]:.1e}; '
                 f'min c {v["min_c"]:.2e}; min sigma {v["sigma_min"]:.2e}; max dF/step {v["max_dF"]:.2e}')
    L.append('')
L += ['## Assertions on trained HyperME rollouts (all 3 test sets x 200 steps)', '']
for m in ['hyperme', 'hyperme_rev']:
    for s in [0, 1]:
        if (m, s) in R:
            r = R[(m, s)]
            L.append(f'- {m}_s{s}: ' + ', '.join(f'{st}: cons {r["sets"][st]["assert_conservation"]}, pos {r["sets"][st]["assert_positivity"]}, sigma>=0 {r["sets"][st]["assert_sigma_nonneg"]}' for st in SETS))
open(os.path.join(RUNS, 'summary.md'), 'w').write('\n'.join(L) + '\n')
print('\n'.join(L))

# ------------------------------------------------------------------ figures
data = torch.load(os.path.join(RUNS, 'data', 'crn.pt'))
ev = {k: np.load(os.path.join(MD, f'{k[0]}_s{k[1]}_eval.npz')) for k in R}

# affinity scatter
fig, axs = plt.subplots(2, 2, figsize=(10, 8.8))
mk = {'test_id': ('o', 'ID'), 'test_oodG': ('^', 'OOD-dG'), 'test_oodU': ('s', 'OOD-U')}
for ax, (m, RR, lk, tag) in zip(axs.ravel(), [('hyperme', R, LOGK2, 'C1'), ('crnn', R, LOGK2, 'C1'), ('hyperme', RB, np.log(5.0), 'C1b'), ('crnn', RB, np.log(5.0), 'C1b')]):
    for s, col in zip([0, 1], ['C0', 'C1']):
        if (m, s) not in RR:
            continue
        for st in SETS:
            S = RR[(m, s)]['sets'][st]
            ax.errorbar(S['affinity_true'], S['affinity_learned_sitemean'], yerr=S['affinity_learned_sitestd'], fmt=mk[st][0],
                        color=col, ms=5, alpha=0.8, lw=0.8, label=f'seed{s} {mk[st][1]}')
    lim = [1.0 + lk - 0.3, 4.0 + lk + 0.3]
    ax.plot(lim, lim, 'k--', lw=1, label='y = x')
    ax.plot(lim, np.array(lim) - lk, 'k:', lw=1, label='y = dG (R1 part only)')
    ax.set_xlabel("true cycle affinity  dG + log(k2/k2')"); ax.set_ylabel('learned <z_v, A> (site mean ± std)')
    ax.set_title(f"{NAMES[m]}, {tag} (k2'={'0.02' if tag == 'C1' else '0.2'})"); ax.legend(fontsize=6, ncol=2)
plt.tight_layout(); plt.savefig(os.path.join(RUNS, 'affinity_scatter.png'), dpi=130); plt.close()

# EPR curves
t = np.arange(T_TEST + 1) * DT
fig, axs = plt.subplots(1, 3, figsize=(15, 4))
for ax, (st, b) in zip(axs, [('test_id', 0), ('test_id', 5), ('test_oodG', 0)]):
    ax.plot(t, data[st]['sigma'][b].numpy(), 'k-', lw=2.5, label='truth')
    for m, ls in [('hyperme', '-'), ('hyperme_rev', '--'), ('crnn', '-.')]:
        if (m, 0) in ev:
            ax.plot(t, ev[(m, 0)][f'{st}/sigma'][b], ls, lw=1.5, label=NAMES[m] + ' s0')
    ax.set_title(f'{st} traj {b}: dG = {float(data[st]["dG"][b]):.2f}'); ax.set_xlabel('t'); ax.set_ylabel('entropy production sigma(t)')
    ax.set_yscale('log'); ax.legend(fontsize=7)
plt.tight_layout(); plt.savefig(os.path.join(RUNS, 'epr_curves.png'), dpi=130); plt.close()

# error curves
fig, axs = plt.subplots(1, 3, figsize=(15, 4))
for ax, st in zip(axs, SETS):
    for m in MODELS:
        for s, ls in [(0, '-'), (1, '--')]:
            if (m, s) in ev:
                e = np.median(ev[(m, s)][f'{st}/relL2_XXp_t'], 0)
                ax.semilogy(t, np.where(np.isfinite(e), e, np.nan), ls, color={'hyperme': 'C0', 'hyperme_rev': 'C4', 'crnn': 'C1', 'gnn': 'C2'}[m],
                            label=NAMES[m] if s == 0 else None)
    ax.axvline(1.0, color='gray', lw=0.8); ax.set_title(st); ax.set_xlabel('t (training window = 1)'); ax.set_ylabel('median rel L2 (X,Xp)')
    ax.legend(fontsize=7)
plt.tight_layout(); plt.savefig(os.path.join(RUNS, 'error_curves.png'), dpi=130); plt.close()

# NESS network + spatial flux map (truth vs HyperME s0) for test_id traj 0 at 10T
st, b = 'test_id', 0
g = build_graph_np(8); V = 64
U = data[st]['U'][b].numpy(); dG = float(data[st]['dG'][b]); cT = data[st]['c'][b, -1].numpy().ravel()
k, A, Us = true_params(g, U, dG)
a, bb = true_ab(cT, g, k, A, Us)
panels = [('truth', a, bb)]
if ('hyperme', 0) in R:
    m = HyperME(True); m.load_state_dict(torch.load(os.path.join(MD, 'hyperme_s0.pt'), map_location='cpu')); m.eval()
    ctx = m.prepare(torch.tensor(U[None]), torch.tensor([dG]), 8)
    cM = torch.tensor(ev[('hyperme', 0)][f'{st}/R_snap'][b, -1][None].astype(np.float64))
    with torch.no_grad():
        am, bm = m.ab(cM, ctx)
    panels.append(('HyperME (seed 0, own rollout state)', am[0].numpy(), bm[0].numpy()))


def site_epr(a, b):
    e = (a - b) * np.log(a / b)
    loc = e[g['r1']] + e[g['r2']]
    for sl in (g['tx'], g['txp']):
        np.add.at(loc, g['et'][sl], 0.5 * e[sl]); np.add.at(loc, g['eh'][sl], 0.5 * e[sl])
    return loc


eprs = [site_epr(a_, b_) for _, a_, b_ in panels]
vmin, vmax = min(e.min() for e in eprs), max(e.max() for e in eprs)
fig, axs = plt.subplots(1, len(panels), figsize=(7 * len(panels), 6.6))
axs = np.atleast_1d(axs)
cmap = plt.get_cmap('magma')
for ax, (title, a_, b_), ep in zip(axs, panels, eprs):
    J = a_ - b_
    ax.imshow(U.reshape(8, 8), cmap='Greys', alpha=0.35, origin='lower', extent=(-0.5, 7.5, -0.5, 7.5))
    jr = J[g['r1']]; jmax = np.abs(a - bb)[g['r1']].max()
    for v in range(V):
        y, x = divmod(v, 8)
        col = cmap((ep[v] - vmin) / (vmax - vmin + 1e-12))
        lw = 0.5 + 3.5 * abs(jr[v]) / jmax
        arr = FancyArrowPatch((x - 0.18, y - 0.05), (x + 0.18, y - 0.05), connectionstyle='arc3,rad=-1.3' if jr[v] > 0 else 'arc3,rad=1.3',
                              arrowstyle='-|>', mutation_scale=8 + 6 * abs(jr[v]) / jmax, lw=lw, color=col)
        ax.add_patch(arr)
    # transport net fluxes of X (blue) and Xp (red) at edge midpoints
    for sl, colr, off in [(g['tx'], 'tab:blue', -0.08), (g['txp'], 'tab:red', 0.08)]:
        Je = J[sl]; ish = np.concatenate([np.ones(V), np.zeros(V)])
        tail = g['et'][sl]; ty, tx_ = np.divmod(tail, 8)
        mx = tx_ + 0.5 * ish; my = ty + 0.5 * (1 - ish)
        keep = ~(((ish == 1) & (tx_ == 7)) | ((ish == 0) & (ty == 7)))       # skip wrap-around edges in the drawing
        ux, uy = Je * ish, Je * (1 - ish)
        ax.quiver(mx[keep] + off * (1 - ish[keep]), my[keep] + off * ish[keep], ux[keep], uy[keep], color=colr, angles='xy', scale_units='xy',
                  scale=np.abs(J[np.concatenate([g['tx'], g['txp']])]).max() / 0.4 if title == 'truth' else np.abs((a - bb)[np.concatenate([g['tx'], g['txp']])]).max() / 0.4,
                  width=0.004, alpha=0.85)
    ax.set_xlim(-0.6, 7.6); ax.set_ylim(-0.6, 7.6); ax.set_aspect('equal')
    ax.set_title(f'{title}\nR1/R2 loop: arrow width ∝ |J_R1|, ccw = net X→Xp→X turnover; color = local EPR\nblue/red arrows: net X / Xp transport; grey = U', fontsize=8)
sm = plt.cm.ScalarMappable(cmap=cmap, norm=plt.Normalize(vmin, vmax)); sm.set_array([])
fig.colorbar(sm, ax=list(axs), shrink=0.7, label='local entropy production (site reactions + ½ adjacent transport)')
corr = np.corrcoef(panels[0][1] - panels[0][2], panels[-1][1] - panels[-1][2])[0, 1] if len(panels) > 1 else np.nan
fig.suptitle(f'NESS at t = 10T, {st} traj {b} (dG = {dG:.2f});  corr(J_model, J_true) over hyperedges = {corr:.3f}', fontsize=10)
plt.savefig(os.path.join(RUNS, 'ness_flux_map.png'), dpi=130, bbox_inches='tight'); plt.close()
print('figures saved')
