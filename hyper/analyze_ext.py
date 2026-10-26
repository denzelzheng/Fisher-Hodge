"""Add-ons: (A) chemostat potentials through the exact chemical-potential channel (C3b, plus C2 OOD-a),
(B) C5 non-ideal (crowding) Schlogl.  python analyze_ext.py -> runs/C3b_summary.md, runs/C5_summary.md,
runs/C3b_ood_phase.png, runs/C5_activity.png  (notes appended from runs/C3b_notes.md, runs/C5_notes.md if present)"""
import os, sys, json
import numpy as np
import torch
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from crn2 import RUNS, make_model, graph
from analyze_c23 import osc_stats, fmt
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

CD = os.path.join(RUNS, 'C23')
NAMES = {'hyperme': 'HyperME (a,b as MLP features)', 'hyperme_chem': 'HyperME-chem (mu_A, mu_B in exponent, MLP G)', 'hyperme_chemcvx': 'HyperME-chem-cvx (+ convex separable G)',
         'crnn': 'B1 CRNN (a,b as features)', 'crnn_chem': 'B1 CRNN-chem (a,b as mass-action conc.)',
         'gnn': 'B2 GNN-ODE (features)', 'hyperme_chemcvx_warm': 'HyperME-chem-cvx, warm start from C2 fit', 'crnn_chem_warm': 'B1 CRNN-chem, warm start from C2 fit', 'hyperme_chemcvx_warm6000': 'HyperME-chem-cvx, warm start, 6000 it', 'crnn_chem_warm6000': 'B1 CRNN-chem, warm start, 6000 it', 'hyperme_chemcvx_it6000': 'HyperME-chem-cvx, 6000 it', 'hyperme_it6000': 'HyperME (features), 6000 it', 'crnn_it6000': 'B1 CRNN (features), 6000 it', 'hyperme_rev': 'B3 (A=0, features)'}
COL = {'truth': 'k', 'hyperme': '#9ecae1', 'hyperme_chem': '#6baed6', 'hyperme_chemcvx': '#08519c', 'crnn': '#fdae6b', 'crnn_chem': '#d94801', 'gnn': '#d62728', 'hyperme_chemcvx_warm': '#08306b', 'crnn_chem_warm': '#7f2704', 'hyperme_chemcvx_warm6000': '#2171b5', 'crnn_chem_warm6000': '#a63603'}


def load(bench, m, s):
    f = os.path.join(CD, f'{bench}_{m}_s{s}.json')
    if not os.path.exists(f):
        return None, None
    return json.load(open(f)), dict(np.load(os.path.join(CD, f'{bench}_{m}_s{s}_eval.npz')))


def seeds(bench, m):
    return [s for s in [0, 1] if os.path.exists(os.path.join(CD, f'{bench}_{m}_s{s}.json'))]


def cell(bench, m, get, f='{:.3g}'):
    out = []
    for s in seeds(bench, m):
        r, z = load(bench, m, s)
        try:
            out.append(fmt(get(r, z), f))
        except (KeyError, TypeError, IndexError, ValueError):
            out.append('n/a')
    return ' / '.join(out) if out else '–'


# =============================================================================================== A: C3b + C2 OOD
def part_a():
    L = ['# C3b: chemostat drive through the exact chemical-potential channel', '',
         'HyperME-chem: mu_A = log a and mu_B = log b enter the rate exponents through the chemostat stoichiometry '
         '(a_h = k exp(<nu+, mu> + <nu+_ext, mu_ext> + A/2), b_h likewise with nu-_ext). k_h, A_raw and g_theta get **no** a, b features, '
         'so the only (a, b) dependence is the exact one. B1 CRNN-chem: kf, kb constant per hyperedge type, a and b multiply as mass-action '
         'concentrations. (Note: the original B1 of C2/C3 received a, b as normalised MLP features, not as concentrations.) B2 GNN-ODE keeps '
         'normalised feature input. Same data, protocol (2000 it, batch 8) and explicit positive SSP-RK2 integrator as before '
         '(no `mprk` integrator exists in hyperme/integrators.py). New models: 1 seed (s0); old models: s0 / s1.', '']
    bench = 'brus'
    data = torch.load(os.path.join(RUNS, 'data', 'brus.pt')); DT = data['meta']['DT']
    rows = ['hyperme', 'hyperme_chem', 'hyperme_chemcvx', 'crnn', 'crnn_chem', 'gnn', 'hyperme_rev']
    L += ['## C3 Brusselator (OOD = b in [3.2, 3.7], training b in [2.3, 3.0])', '',
          '| model | ID T | ID 10T (homog / spatial) | OOD T | OOD 2T | OOD 10T (homog / spatial) | OOD period rel err | OOD amp rel err | OOD traj c<0 | sigma rel err ID / OOD | learned cycle affinities ID (true 13.8, 14.8) |',
          '|---|---|---|---|---|---|---|---|---|---|---|']

    def oscerr(r, z, st, q):
        d = data[st]; h = d['homog'].numpy().astype(bool); Y = d['c'].numpy(); R = z[f'{st}/R']; e = []
        for i in np.where(h)[0]:
            at, pt = osc_stats(Y[i, :, 0].mean(-1), DT); am, pm = osc_stats(R[i, :, 0].mean(-1), DT)
            e.append(abs(pm - pt) / pt if q == 'p' else abs(am - at) / at)
        e = np.where(np.isfinite(e), e, np.inf)
        return float(np.median(e))
    for m in rows:
        L.append(f'| {NAMES[m]} | ' + ' | '.join([
            cell(bench, m, lambda r, z: r['sets']['test_id']['relL2_median']['20']),
            cell(bench, m, lambda r, z: '{:.3g} ({:.3g} / {:.3g})'.format(r['sets']['test_id']['relL2_median']['200'], r['sets']['test_id']['relL2_median_homog']['200'], r['sets']['test_id']['relL2_median_spatial']['200']), '{}'),
            cell(bench, m, lambda r, z: r['sets']['test_ood']['relL2_median']['20']),
            cell(bench, m, lambda r, z: r['sets']['test_ood']['relL2_median']['40']),
            cell(bench, m, lambda r, z: '{:.3g} ({:.3g} / {:.3g})'.format(r['sets']['test_ood']['relL2_median']['200'], r['sets']['test_ood']['relL2_median_homog']['200'], r['sets']['test_ood']['relL2_median_spatial']['200']), '{}'),
            cell(bench, m, lambda r, z: oscerr(r, z, 'test_ood', 'p'), '{:.3f}'),
            cell(bench, m, lambda r, z: oscerr(r, z, 'test_ood', 'a'), '{:.3f}'),
            cell(bench, m, lambda r, z: r['sets']['test_ood']['n_traj_negative'], '{:d}'),
            cell(bench, m, lambda r, z: '{:.3f} ; {:.3f}'.format(r['sets']['test_id']['sigma_relerr_median'], r['sets']['test_ood']['sigma_relerr_median']), '{}') if m != 'gnn' else 'n/a',
            cell(bench, m, lambda r, z: '({:.1f}, {:.1f})'.format(*np.mean(r['sets']['test_id']['affinity_learned_sitemean'], 0)), '{}') if m != 'gnn' else 'n/a']) + ' |')
    # ---- C2 OOD-a
    L += ['', '## C2 Schlogl (OOD = a in [3.15, 3.35], b in [0.3, 1.0]; training a in [2.85, 3.10], b in [1.0, 3.2])', '',
          '| model | final train loss | ID T | ID 10T | OOD T | OOD 10T | front 10T | OOD traj c<0 | max X / true max (OOD) | sigma rel err ID / OOD | affinity MAE ID / OOD |',
          '|---|---|---|---|---|---|---|---|---|---|---|']
    d2 = torch.load(os.path.join(RUNS, 'data', 'schlogl.pt'))
    Ymax = d2['test_ood']['c'].numpy().max()
    for m in ['hyperme', 'hyperme_it6000', 'hyperme_chem', 'hyperme_chemcvx', 'hyperme_chemcvx_it6000', 'crnn', 'crnn_it6000', 'crnn_chem', 'gnn', 'hyperme_rev']:
        L.append(f'| {NAMES[m]} | ' + ' | '.join([
            cell('schlogl', m, lambda r, z: r['final_train_loss'], '{:.2e}'),
            cell('schlogl', m, lambda r, z: r['sets']['test_id']['relL2_median']['20']),
            cell('schlogl', m, lambda r, z: r['sets']['test_id']['relL2_median']['200']),
            cell('schlogl', m, lambda r, z: r['sets']['test_ood']['relL2_median']['20']),
            cell('schlogl', m, lambda r, z: r['sets']['test_ood']['relL2_median']['200']),
            cell('schlogl', m, lambda r, z: r['sets']['test_front']['relL2_median']['200']),
            cell('schlogl', m, lambda r, z: r['sets']['test_ood']['n_traj_negative'], '{:d}'),
            cell('schlogl', m, lambda r, z: np.nanmax(z['test_ood/R']) / Ymax, '{:.2f}'),
            cell('schlogl', m, lambda r, z: '{:.3f} ; {:.3f}'.format(r['sets']['test_id']['sigma_relerr_median'], r['sets']['test_ood']['sigma_relerr_median']), '{}') if m != 'gnn' else 'n/a',
            cell('schlogl', m, lambda r, z: '{:.2f} ; {:.2f}'.format(r['sets']['test_id']['affinity_MAE'], r['sets']['test_ood']['affinity_MAE']), '{}') if m != 'gnn' else 'n/a']) + ' |')
    nf = os.path.join(RUNS, 'C3b_notes.md')
    if os.path.exists(nf):
        L += ['', open(nf).read()]
    open(os.path.join(RUNS, 'C3b_summary.md'), 'w').write('\n'.join(L) + '\n')
    # ---- figure: OOD phase planes (homogeneous starts 0 and 1)
    mods = [m for m in ['hyperme', 'hyperme_chem', 'hyperme_chemcvx', 'crnn_chem', 'gnn'] if seeds(bench, m)]
    fig, ax = plt.subplots(2, len(mods), figsize=(3.4 * len(mods), 6.6), squeeze=False)
    d = data['test_ood']
    for ri, i in enumerate([0, 1]):
        Y = d['c'].numpy()[i]
        for ci, m in enumerate(mods):
            _, z = load(bench, m, 0)
            R = z['test_ood/R'][i]
            a_ = ax[ri, ci]
            a_.plot(Y[:, 0].mean(-1), Y[:, 1].mean(-1), 'k-', lw=2.4, alpha=0.55, label='truth')
            a_.plot(R[:, 0].mean(-1), R[:, 1].mean(-1), '-', color=COL[m], lw=1.2, label=m)
            a_.plot(Y[0, 0].mean(), Y[0, 1].mean(), 'ko', ms=4)
            e = np.linalg.norm(R[-1] - Y[-1]) / np.linalg.norm(Y[-1])
            a_.set_title(f'{m} | b={d["b"][i]:.2f} (OOD) | 10T err {e:.2f}', fontsize=8)
            a_.set_xlabel('mean X'); a_.legend(fontsize=7)
        ax[ri, 0].set_ylabel('mean Y')
    fig.suptitle('C3b Brusselator OOD-b (b > training range), homogeneous start, 10T = 40: drive as features vs exact chemical-potential channel (seed 0)', fontsize=9)
    fig.tight_layout(); fig.savefig(os.path.join(RUNS, 'C3b_ood_phase.png'), dpi=130); plt.close(fig)


# =============================================================================================== B: C5
def part_b():
    import gen_nonideal as GN
    bench = 'schlogl_ni'
    data = torch.load(os.path.join(RUNS, 'data', 'schlogl_ni.pt'))
    L = ['# C5: non-ideal (crowding) Schlogl model, 8x8 periodic, activity-corrected mass action', '',
         f'Truth: mu = log x + ln gamma(x), ln gamma = phi x/(1 - phi x), phi = {GN.PHI}; every hyperedge (R1, R2, transport) uses '
         'a_h = k exp(<nu+, mu> + A/2). Same constants, chemostats and sampling as C2. ID ln gamma in [0.06, 0.92]; OOD = a in [3.5, 4.0], '
         'b in [2.0, 3.5] (higher concentration, x up to 2.12, ln gamma up to 1.13). Ideal mass action with the TRUE constants misses the '
         'non-ideal truth by 23% (ID 10T relL2 median). Models: HyperME-chem (g_theta = dG/dc should learn ln gamma; no a,b features), '
         'B1 CRNN-chem (pure mass action in x, a,b as concentrations: wrong model class), B2 GNN-ODE (features). 2000 it, batch 8.', '',
         '| model | final train loss | ID T | ID 10T | OOD T | OOD 10T | traj c<0 ID / OOD | min c | sigma rel err ID / OOD | affinity MAE ID / OOD |',
         '|---|---|---|---|---|---|---|---|---|---|']
    for m in ['hyperme_chem', 'hyperme_chemcvx', 'hyperme_chemcvx_warm', 'hyperme_chemcvx_warm6000', 'crnn_chem', 'crnn_chem_warm', 'crnn_chem_warm6000', 'gnn']:
        L.append(f'| {NAMES[m]} | ' + ' | '.join([
            cell(bench, m, lambda r, z: r['final_train_loss'], '{:.2e}'),
            cell(bench, m, lambda r, z: r['sets']['test_id']['relL2_median']['20']),
            cell(bench, m, lambda r, z: r['sets']['test_id']['relL2_median']['200']),
            cell(bench, m, lambda r, z: r['sets']['test_ood']['relL2_median']['20']),
            cell(bench, m, lambda r, z: r['sets']['test_ood']['relL2_median']['200']),
            cell(bench, m, lambda r, z: f"{r['sets']['test_id']['n_traj_negative']},{r['sets']['test_ood']['n_traj_negative']}", '{}'),
            cell(bench, m, lambda r, z: min(r['sets'][q]['min_c'] for q in ['test_id', 'test_ood']), '{:.2e}'),
            cell(bench, m, lambda r, z: '{:.3f} ; {:.3f}'.format(r['sets']['test_id']['sigma_relerr_median'], r['sets']['test_ood']['sigma_relerr_median']), '{}') if m != 'gnn' else 'n/a',
            cell(bench, m, lambda r, z: '{:.2f} ; {:.2f}'.format(r['sets']['test_id']['affinity_MAE'], r['sets']['test_ood']['affinity_MAE']), '{}') if m != 'gnn' else 'n/a']) + ' |')
    # ---- learned ln gamma
    xs = np.linspace(0.2, 2.15, 80)
    lt = GN.lng(xs)
    idm = (xs >= 0.23) & (xs <= 1.92)
    G = {}
    for mn, s in [(mn, s) for mn in ['hyperme_chem', 'hyperme_chemcvx', 'hyperme_chemcvx_warm', 'hyperme_chemcvx_warm6000'] for s in seeds(bench, mn)]:
        m = make_model(mn.replace('_warm6000', '').replace('_warm', ''), bench)
        m.load_state_dict(torch.load(os.path.join(CD, f'{bench}_{mn}_s{s}.pt')))
        ctx = m.prepare(torch.zeros(len(xs), 2), 8)
        c = torch.tensor(xs)[:, None, None].expand(len(xs), 1, 64).contiguous()
        gl = m.gcorr(c, ctx)[:, 0, 0].detach().numpy()
        off = np.mean(lt[idm] - gl[idm])                    # gauge: mu is defined up to a constant (absorbed by k)
        G[(mn, s)] = gl + off
    L += ['', '## Learned activity correction', '',
          'g_theta(x) = dG_theta/dx at a uniform state, compared with the true ln gamma(x) after removing the one free additive constant '
          '(a uniform shift of mu is absorbed by k; fitted on the ID range x in [0.23, 1.92]). Ideal reference (g = 0) shown for scale.', '',
          '| model | RMSE ID range | max abs err ID range | RMSE OOD extension (x in [1.92, 2.15]) | slope of learned vs true (ID) |', '|---|---|---|---|---|']
    ood = xs > 1.92
    li = lt - np.mean(lt[idm])
    L.append(f'| ideal (g = const) | {np.sqrt(np.mean(li[idm] ** 2)):.3f} | {np.abs(li[idm]).max():.3f} | {np.sqrt(np.mean(li[ood] ** 2)):.3f} | 0 |')
    for (mn, s), gl in G.items():
        e = gl - lt
        L.append(f'| {NAMES[mn]} s{s} | {np.sqrt(np.mean(e[idm] ** 2)):.3f} | {np.abs(e[idm]).max():.3f} | {np.sqrt(np.mean(e[ood] ** 2)):.3f} | {np.polyfit(lt[idm], gl[idm], 1)[0]:.3f} |')
    nf = os.path.join(RUNS, 'C5_notes.md')
    if os.path.exists(nf):
        L += ['', open(nf).read()]
    open(os.path.join(RUNS, 'C5_summary.md'), 'w').write('\n'.join(L) + '\n')
    # ---- figure
    fig, ax = plt.subplots(1, 3, figsize=(15, 4.2))
    ax[0].axvspan(0.23, 1.92, color='0.92', label='ID concentration range')
    ax[0].plot(xs, lt, 'k-', lw=2.5, label='true ln gamma')
    ax[0].plot(xs, np.zeros_like(xs) + np.mean(lt[idm]), 'k:', lw=1, label='ideal (const)')
    for (mn, s), gl in G.items():
        ax[0].plot(xs, gl, '--' if s == 0 else ':', color=COL[mn], label=f'{mn} s{s} (gauge-aligned)')
    ax[0].set_xlabel('x'); ax[0].set_ylabel('ln gamma(x)'); ax[0].legend(fontsize=7); ax[0].set_title('learned activity coefficient vs truth', fontsize=9)
    for ai, st in enumerate(['test_id', 'test_ood']):
        d = data[st]; Y = d['c'].numpy(); t = np.arange(Y.shape[1]) * data['meta']['DT']
        for m in ['hyperme_chem', 'hyperme_chemcvx_warm6000', 'crnn_chem', 'crnn_chem_warm6000', 'gnn']:
            if not seeds(bench, m):
                continue
            _, z = load(bench, m, 0)
            e = np.linalg.norm((z[f'{st}/R'] - Y).reshape(len(Y), len(t), -1), axis=-1) / np.linalg.norm(Y.reshape(len(Y), len(t), -1), axis=-1)
            ax[ai + 1].semilogy(t, np.median(e, 0), color=COL[m], label=m + ' s0')
        ax[ai + 1].set_xlabel('t'); ax[ai + 1].set_ylabel('median rel. L2 error'); ax[ai + 1].legend(fontsize=7)
        ax[ai + 1].set_title(f'C5 {st[5:].upper()} rollout error (T = 1)', fontsize=9)
    fig.tight_layout(); fig.savefig(os.path.join(RUNS, 'C5_activity.png'), dpi=130); plt.close(fig)


if __name__ == '__main__':
    part_a()
    part_b()
    print('ext analysis done')
