"""E5 analysis: cycle observables vs identifiability.  python e5_analyze.py -> runs/E5_summary.md, runs/E5_affinity.png,
runs/E5_sigma.png, runs/E5_results.json; appends/replaces the E5 section of runs/summary.md"""
import os, sys, json
import numpy as np
import torch
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from hyperme import HyperME, RUNS, DT, T_TEST, graph
from baselines_hyper import CRNN
from obs_hyper import obs_rows, true_flux
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

torch.set_num_threads(8)
MODELS = {'hyperme': lambda: HyperME(True), 'hyperme_rev': lambda: HyperME(False), 'crnn': CRNN}
NAMES = {'hyperme': 'HyperME', 'hyperme_rev': 'B3 HyperME (A=0)', 'crnn': 'B1 CRNN'}
OBS = ['none', 'global', 'local4', 'globalR2']
OBSN = {'none': 'no observable', 'global': '(a) global r(t)=sum J_R1', 'local4': '(b) J_R1 at 4 sites', 'globalR2': '(c) sum J_R2'}
SETS = ['test_id', 'test_oodG', 'test_oodU']
LOGK2 = float(np.log(1.0 / 0.02))
data = torch.load(os.path.join(RUNS, 'data', 'crn.pt'))
sd = data['test_id']
Jtrue = true_flux(sd, 'cpu').numpy()                         # (16, 201, H)
g8 = graph(8, 'cpu')


def path(m, o, s, ext):
    return os.path.join(RUNS, 'main', f'{m}_s{s}{ext}') if o == 'none' else os.path.join(RUNS, 'E5_runs', f'{m}_obs-{o}_s{s}{ext}')


@torch.no_grad()
def obs_rollout(m, o, s):
    model = MODELS[m](); model.load_state_dict(torch.load(path(m, o, s, '.pt'), map_location='cpu')); model.eval()
    ctx = model.prepare(sd['U'], sd['dG'], 8)
    c = sd['c'][:, 0].clone(); Js = [model.flux(c, ctx)]
    for _ in range(T_TEST):
        c, _ = model.advance(c, ctx, DT); Js.append(model.flux(c, ctx))
    return torch.stack(Js, 1).numpy()                        # (16, 201, H)


res = {}
for m in MODELS:
    for o in OBS:
        for s in [0, 1]:
            f = path(m, o, s, '.json')
            if not os.path.exists(f):
                continue
            r = json.load(open(f))
            Jm = obs_rollout(m, o, s)
            ob = {}
            for kind in ['global', 'local4', 'globalR2']:
                W = obs_rows(kind, s)[0]
                om, ot = Jm @ W.T, Jtrue @ W.T
                ob[kind] = float(np.median(np.linalg.norm((om - ot).reshape(16, -1), axis=1) / np.linalg.norm(ot.reshape(16, -1), axis=1)))
            xt = np.concatenate([r['sets'][st]['affinity_true'] for st in SETS]); yl = np.concatenate([r['sets'][st]['affinity_learned_sitemean'] for st in SETS])
            dg = xt - LOGK2
            sl, ic = np.polyfit(xt, yl, 1) if np.std(yl) > 1e-9 else (0.0, float(np.mean(yl)))
            res[(m, o, s)] = dict(relL2_ID_10T=r['sets']['test_id']['relL2_XXp_median']['200'], relL2_oodG_10T=r['sets']['test_oodG']['relL2_XXp_median']['200'],
                                  relL2_oodU_10T=r['sets']['test_oodU']['relL2_XXp_median']['200'],
                                  sigma_relerr_ID=r['sets']['test_id']['sigma_relerr_median'], sigma_relerr_oodG=r['sets']['test_oodG']['sigma_relerr_median'],
                                  Jcorr_ID=r['sets']['test_id']['flux_corr_10T_median'], Jrel_ID=r['sets']['test_id']['flux_relerr_10T_median'],
                                  aff_MAE=float(np.mean([r['sets'][st]['affinity_MAE'] for st in SETS])), aff_slope=float(sl), aff_intercept=float(ic),
                                  R2_part=float(np.mean(yl - dg)), Acyc_relerr_ID=r['sets']['test_id']['Acycle_relerr_median'],
                                  obs_test_relerr=ob, final_obs_loss=r.get('final_obs_loss'), final_loss=r['final_train_loss'],
                                  aff_true=xt.tolist(), aff_learned=yl.tolist(), aff_std=np.concatenate([r['sets'][st]['affinity_learned_sitestd'] for st in SETS]).tolist(),
                                  r_model_traj0=(Jm[0] @ obs_rows('global')[0].T)[:, 0].tolist())
            print(m, o, s, {k: (round(v, 4) if isinstance(v, float) else v) for k, v in res[(m, o, s)].items() if not k.startswith(('aff_true', 'aff_learned', 'aff_std', 'r_model'))}, flush=True)
json.dump({f'{m}|{o}|s{s}': v for (m, o, s), v in res.items()}, open(os.path.join(RUNS, 'E5_results.json'), 'w'))


def cell(m, o, key, f='{:.3g}', sub=None):
    vals = []
    for s in [0, 1]:
        if (m, o, s) not in res:
            vals.append('–'); continue
        v = res[(m, o, s)][key] if sub is None else res[(m, o, s)][key][sub]
        vals.append('n/a' if v is None or not np.isfinite(v) else f.format(v))
    return ' / '.join(vals)


L = ['## E5: cycle observables and identifiability', '',
     f'Training loss = trajectory loss + w * relative squared error of the observable (w = 0.02, so both terms are the same size at initialisation: '
     f'obs ~1.4 x 0.02 vs trajectory ~2e-2). Observable truth from the true rates on the training frames. Entries `seed0 / seed1`. '
     f'"R2 part" = mean(learned affinity - dG); true value log(k2/k2\') = {LOGK2:.2f}. Observable test error = rel. L2 of the model\'s r(t)=sum_v J_R1 '
     f'along its own 10T rollout (test_id).', '',
     '| model | observable | relL2 ID 10T | relL2 OOD-dG 10T | relL2 OOD-U 10T | sigma rel err ID | J corr ID | J rel err ID | affinity MAE | slope | R2 part | A_cyc rel err | r(t) test err | final obs loss |',
     '|---|---|---|---|---|---|---|---|---|---|---|---|---|---|']
for m in MODELS:
    for o in OBS:
        if not any((m, o, s) in res for s in [0, 1]):
            continue
        L.append(f'| {NAMES[m]} | {OBSN[o]} | ' + ' | '.join([cell(m, o, 'relL2_ID_10T'), cell(m, o, 'relL2_oodG_10T'), cell(m, o, 'relL2_oodU_10T'),
                 cell(m, o, 'sigma_relerr_ID'), cell(m, o, 'Jcorr_ID', '{:.3f}'), cell(m, o, 'Jrel_ID'), cell(m, o, 'aff_MAE'),
                 cell(m, o, 'aff_slope', '{:.2f}'), cell(m, o, 'R2_part', '{:.2f}'), cell(m, o, 'Acyc_relerr_ID'),
                 cell(m, o, 'obs_test_relerr', sub='global'), cell(m, o, 'final_obs_loss', '{:.2e}')]) + ' |')
L.append('')
lin = json.load(open(os.path.join(RUNS, 'E5_linalg.json')))
L += ['### Linear-algebra check (obs_hyper.py, true C1 hypergraph)', '',
      f'- H = {lin["H"]} hyperedges (64 R1 + 64 R2 + 128 X-transport + 128 Xp-transport), 192 states, rank S = {lin["rank_S"]}, '
      f'dim ker S^T = {lin["dim_kerST"]} (E per site + X+Xp total), **dim ker S = beta_1 = {lin["dim_kerS"]} independent cycles** '
      f'= {lin["cycles_local_R1R2"]} local R1-R2 loops + {lin["cycles_transport_X"]} X-transport + {lin["cycles_transport_Xp"]} Xp-transport + {lin["cycles_mixed (transport-reaction)"]} mixed transport-reaction cycles.',
      '- Concentration data fix only S J, so ' + str(lin['flux_level']['none']['invisible_flux_dims']) + ' flux directions are invisible; fixing J completely needs 257 independent cycle observables.',
      '- Invisible flux dims ker[S; W]: ' + ', '.join(f'{k}: {v["invisible_flux_dims"]}' for k, v in lin['flux_level'].items())
      + f'. (a) and (c) are equivalent: w_R1 - w_R2 lies in im S^T (residual {lin["|w_R1 - w_R2 - proj_imST|"]:.1e}), so together they still give +1.',
      '- A-level (A = Z beta, linearised at the true NESS of test_id traj 0, k and mu known): identifiable beta directions '
      + ', '.join(f'{k}: {v["snapshot_kknown_identifiable"]}' for k, v in lin['A_level'].items()) + ' of 257, i.e. **(a) adds exactly 1, (b) adds 4**.',
      '- If per-hyperedge k is unknown as well, one snapshot identifies 0 beta directions. Over the whole trajectory (41 frames) it identifies '
      + ', '.join(f'{k}: {v["trajectory_joint_identifiable_beta"]}' for k, v in lin['A_level'].items())
      + ' (the 64 missing directions with no observable match the 64 local loops). Here (b) adds a robust +4. The gain from (a) depends on the rank tolerance: '
      '+11 / +16 / +21 at rtol 1e-6 / 1e-9 / 1e-12. The time series r(t) mixes 64 sites, so it is worth more than one number.', '']
open(os.path.join(RUNS, 'E5_summary.md'), 'w').write('\n'.join(L) + '\n')
sm = open(os.path.join(RUNS, 'summary.md')).read()
sm = sm.split('\n## E5:')[0].rstrip('\n') + '\n\n' + '\n'.join(L) + '\n'
open(os.path.join(RUNS, 'summary.md'), 'w').write(sm)
print('\n'.join(L))

# ---------------- figures
fig, axs = plt.subplots(2, 3, figsize=(14, 8.6), sharex=True)
for i, m in enumerate(['hyperme', 'crnn']):
    for j, o in enumerate(['none', 'global', 'local4']):
        ax = axs[i, j]
        for s, col in [(0, 'C0'), (1, 'C1')]:
            if (m, o, s) in res:
                R = res[(m, o, s)]
                ax.errorbar(R['aff_true'], R['aff_learned'], yerr=R['aff_std'], fmt='o', ms=4, color=col, alpha=0.75, lw=0.7,
                            label=f'seed{s}: MAE {R["aff_MAE"]:.2f}, slope {R["aff_slope"]:.2f}')
        lim = np.array([5.0, 8.0])
        ax.plot(lim, lim, 'k--', lw=1, label='y = x (truth)'); ax.plot(lim, lim - LOGK2, 'k:', lw=1, label='y = dG (R1 part only)')
        ax.set_title(f'{NAMES[m]} — {OBSN[o]}', fontsize=9); ax.legend(fontsize=7)
        if i == 1:
            ax.set_xlabel("true cycle affinity dG + log(k2/k2')")
        if j == 0:
            ax.set_ylabel('learned <z_v, A> (site mean ± std)')
fig.suptitle('E5: learned local cycle affinity, ID + OOD-dG + OOD-U test trajectories', fontsize=11)
plt.tight_layout(); plt.savefig(os.path.join(RUNS, 'E5_affinity.png'), dpi=130); plt.close()

t = np.arange(T_TEST + 1) * DT
fig, axs = plt.subplots(1, 3, figsize=(16, 4.4))
ev = lambda m, o, s: np.load(path(m, o, s, '_eval.npz'))
for ax, m in zip(axs[:2], ['hyperme', 'crnn']):
    ax.plot(t, sd['sigma'][0].numpy(), 'k-', lw=2.5, label='truth')
    for o, ls in [('none', ':'), ('global', '-'), ('local4', '--')]:
        if (m, o, 0) in res:
            ax.plot(t, ev(m, o, 0)['test_id/sigma'][0], ls, lw=1.6, label=OBSN[o])
    ax.set_yscale('log'); ax.set_title(f'{NAMES[m]} seed0: sigma(t), test_id traj 0'); ax.set_xlabel('t'); ax.legend(fontsize=7)
ax = axs[2]
rt = (Jtrue[0] @ obs_rows('global')[0].T)[:, 0]
ax.plot(t, rt, 'k-', lw=2.5, label='truth r(t)')
for m, o, ls in [('hyperme', 'none', ':'), ('hyperme', 'global', '-'), ('hyperme_rev', 'none', ':'), ('hyperme_rev', 'global', '-'), ('crnn', 'global', '-.')]:
    if (m, o, 0) in res:
        ax.plot(t, res[(m, o, 0)]['r_model_traj0'], ls, lw=1.5, label=f'{NAMES[m]}, {o}')
ax.set_title('global ATP turnover r(t) = sum_v J_R1 (test_id traj 0, seed 0)'); ax.set_xlabel('t'); ax.set_yscale('symlog', linthresh=1); ax.legend(fontsize=7)
plt.tight_layout(); plt.savefig(os.path.join(RUNS, 'E5_sigma.png'), dpi=130); plt.close()
print('figures saved')
