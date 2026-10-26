"""Task C tables (markdown) and an example figure from taskC_results_<variant>.json, taskC_selfcheck.json and
~/ext_work/taskC/taskC_examples_<variant>.pt.
python taskC_summary.py [--dir runs/external/common] [--work ~/ext_work/taskC] [--prefix taskC_results]"""
import os, sys, json, argparse
import numpy as np

FH = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
VARIANTS = [('auto', 'HyperME, fit() with automatic initial value (working row)'),
            ('fit', 'HyperME, default fit() (closed-form NNLS initial value)'),
            ('onestep', 'HyperME, one-step loss variant (sensitivity)')]
ROWS = [('oracle_mean', 'oracle: true particle model, mean of 8 re-simulations'), ('oracle_single', 'oracle: one re-simulation'),
        ('gnn_sde_mean', 'GNN (re-impl.) + true noise, mean of 8'), ('gnn_sde_single', 'GNN + true noise, one run'),
        ('gnn_ode', 'GNN, deterministic ODE'), ('persist', 'persistence rho(t0)'), ('uniform', 'uniform mean density')]


def f(x, k='median'):
    return f'{x[k]:.3f}' if isinstance(x, dict) else f'{x:.3f}'


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--dir', default=os.path.join(FH, 'runs', 'external', 'common'))
    ap.add_argument('--work', default=os.path.expanduser('~/ext_work/taskC'))
    ap.add_argument('--prefix', default='taskC_results')
    a = ap.parse_args()
    R = {v: json.load(open(os.path.join(a.dir, f'{a.prefix}_{v}.json'))) for v, _ in VARIANTS if os.path.exists(os.path.join(a.dir, f'{a.prefix}_{v}.json'))}
    base = R['auto']
    L = []
    L.append(f'GNN checkpoint: epoch {base["gnn_checkpoint"]["epoch"]}, validation loss {base["gnn_checkpoint"]["valid"]:.5f}. '
             'Medians over 27 cases per set (3 test trajectories x 9 start frames); relative L2 on the 16 x 16 x 2 density / face flux.\n')
    for st, title in (('test_id', 'in distribution (200 + 200)'), ('test_ood_ratio', 'out of distribution, species ratio 100 + 300')):
        L.append(f'#### {title}\n')
        L.append('| Method | rho @1T | rho @10T | J @1T vs data | J @1T vs expected | circulation corr. (expected) | circulation RMS ratio |')
        L.append('|---|---|---|---|---|---|---|')
        rows = [(k, lab, base[st][k]) for k, lab in ROWS[:5]]
        rows += [(f'hm_{v}', lab, R[v][st]['hyperme']) for v, lab in VARIANTS if v in R]
        rows += [(k, lab, base[st][k]) for k, lab in ROWS[5:]]
        for k, lab, r in rows:
            L.append(f'| {lab} | {f(r["rho1"])} | {f(r["rho10"])} | {f(r["J1_vs_data"])} | {f(r["J1_vs_expected"])} | '
                     f'{f(r["circ_corr_expected"])} | {r["circ_rms_ratio"]:.3f} |')
        ta = base[st]['time_averaged_flux']
        L.append(f'\nTime-averaged flux over t = 100..299 (one-step predictions from the true snapshots): data RMS {ta["data"]["rms"]:.3f}, '
                 f'circulation RMS {ta["data"]["circ_rms"]:.3f}; GNN-ODE RMS {ta["gnn_ode"]["rms"]:.3f}, circulation {ta["gnn_ode"]["circ_rms"]:.3f}, '
                 f'corr. with data {ta["gnn_ode"]["corr_vs_data"]:.2f}; HyperME (auto) RMS {ta["hyperme"]["rms"]:.3f}, circulation {ta["hyperme"]["circ_rms"]:.4f}, '
                 f'corr. with data {ta["hyperme"]["corr_vs_data"]:.2f}.\n')
        ro = base[st]['hyperme_readouts']
        L.append(f'HyperME (auto) readouts at the test snapshots (whole grid, per unit time): sigma {ro["sigma"]["median"]:.1f}, '
                 f'excess {ro["excess"]["median"]:.1f}, housekeeping {ro["housekeeping"]["median"]:.3g}.\n')
    gi = base['gnn_interactions']
    L.append('#### Only the GNN: learned pair functions vs truth (pairs of real snapshots), R^2 / relative L2\n')
    L.append('| receiver <- neighbour | velocity message R^2 | velocity rel. L2 | torque R^2 | torque rel. L2 |')
    L.append('|---|---|---|---|---|')
    for k in ('00', '01', '10', '11'):
        L.append(f'| {k[0]} <- {k[1]} | {gi[k]["velocity"]["r2"]:.3f} | {gi[k]["velocity"]["rel_l2"]:.3f} | {gi[k]["torque"]["r2"]:.3f} | {gi[k]["torque"]["rel_l2"]:.3f} |')
    sv = gi['self']
    L.append(f'\nSelf-propulsion (body frame) learned {np.round(sv["v_body"], 3).tolist()}, true (1, 0) for both species; self torque {np.round(sv["w"], 4).tolist()}, true 0.\n')
    hm = base['hyperme']
    chi = np.array(hm['chi'])
    L.append('#### Only HyperME: fitted scalars (auto)\n')
    L.append(f'log k (per species, rel. to 1/h^2) {np.round(hm["logk"], 3).tolist()}; harmonic drift b {np.round(hm["harmonic_b"], 3).tolist()}; '
             f'chi[width][s][s\'] for widths {hm["widths"]}: {np.round(chi, 3).tolist()}; '
             f'non-reciprocity chi_01 - chi_10 per width: {np.round(chi[:, 0, 1] - chi[:, 1, 0], 3).tolist()}.\n')
    sc_path = os.path.join(a.dir, 'taskC_selfcheck.json')
    if os.path.exists(sc_path):
        sc = json.load(open(sc_path))
        L.append('#### GNN self-check on its home task (re-simulation from a random state to t = 300, 3 runs)\n')
        L.append('| | fraction clustered | other-species share of neighbours | local polar order | density CV |')
        L.append('|---|---|---|---|---|')
        for k, lab in (('resim_t300_initial', 'initial state'),):
            r = sc[k]; L.append(f'| {lab} | {r["frac_clustered"]:.3f} | {r["frac_other_species_neighbours"]:.3f} | {r["local_polar_order"]:.3f} | {r["density_cv"]:.3f} |')
        for k in ('truth', 'gnn'):
            r = sc['resim_t300'][k]
            L.append(f'| {"true model" if k == "truth" else "learned GNN + true noise"}, t = 300 | {r["frac_clustered"]:.3f} | {r["frac_other_species_neighbours"]:.3f} | {r["local_polar_order"]:.3f} | {r["density_cv"]:.3f} |')
        L.append('')
    out = '\n'.join(L) + '\n'
    open(os.path.join(a.dir, 'taskC_table.md'), 'w').write(out)
    print(out)

    import torch
    ex_path = os.path.join(a.work, 'taskC_examples_auto.pt')
    if os.path.exists(ex_path):
        import matplotlib; matplotlib.use('Agg')
        import matplotlib.pyplot as plt
        ex = torch.load(ex_path)['test_id']
        names = [('truth', None), ('oracle_mean', 'oracle mean of 8'), ('gnn_sde_mean', 'GNN + noise, mean of 8'), ('gnn_ode', 'GNN ODE'),
                 ('hyperme', 'HyperME (auto)'), ('persist', 'persistence')]
        fig, axs = plt.subplots(2, len(names), figsize=(3 * len(names), 6))
        for s in range(2):
            vmax = float(ex['true10'][10][0, s].max())
            for j, (k, lab) in enumerate(names):
                img = ex['true10'][10][0, s] if k == 'truth' else ex['pred10'][k][0, s]
                axs[s, j].imshow(img.T.numpy(), origin='lower', vmin=0, vmax=vmax, cmap='viridis')
                axs[s, j].set_title(f'{"data" if k == "truth" else lab}, species {s}', fontsize=8); axs[s, j].axis('off')
        fig.suptitle('Density 10 time units after t0 = 30 (test trajectory 0)')
        fig.tight_layout(); fig.savefig(os.path.join(a.dir, 'taskC_density_example.png'), dpi=110)


if __name__ == '__main__':
    main()
