"""Result tables of the porous-media benchmark, written from results/<variant>/runs/<model>.json (or from a directory
of <model>.json files that run.py train writes, runs/<variant>/).

main       relative L2 error of the density at 10T per test set, negative-density trajectories, parameters
physics    local-detailed-balance violations, samples with sigma < 0, negative cell values
ablations  the Neural Chart variants and what each one removes
law        the learned mobility against the true mobility at the reference states (Neural Chart)
training   parameters, training time, final training loss
"""
import json
import os

from neuralchart.chart import ABLATIONS
from benchmarks.common import LABELS, OWN_FLUX, RESULTS, load_runs, write_table, fmt, param_cell
from .test_sets import COLUMNS, LABEL

MODELS = {
    'porous_media': ['chart', 'chart-no-state', 'chart-const-k', 'chart-learned-drive', 'oracle', 'ude', 'flux_gnn', 'bsms',
                     'fno', 'generic_fno', 'helmfluid', 'persistence'],
    'porous_media_crowding': ['chart', 'chart-no-state', 'oracle', 'oracle-no-crowding', 'ude', 'flux_gnn', 'bsms', 'fno',
                              'generic_fno', 'helmfluid', 'persistence'],
}
LAW_SETS = [('training_conditions', 'unseen media, 32×32 grid'), ('low_porosity', 'unseen low-porosity media'),
            ('grid64', 'unseen media, 64×64 grid'), ('point_cloud', 'unseen media, point cloud')]


def _set(r, s):
    return (r.get('sets') or {}).get(s)


def main_rows(runs, models, key='err_10T'):
    rows = []
    for m in models:
        if m not in runs:
            continue
        r = runs[m]
        cells = [(_set(r, s) or {}).get(key) if _set(r, s) is not None else None for s in COLUMNS]
        neg = [_set(r, s)['n_neg_traj'] for s in COLUMNS if _set(r, s) is not None and 'n_neg_traj' in _set(r, s)]
        rows.append([LABELS[m]] + cells + [str(max(neg)) if neg else 'N/A', param_cell(m, r)])
    return rows


def physics_rows(runs, models):
    rows = []
    for m in models:
        if m not in runs or m == 'persistence':
            continue
        r = runs[m]
        for k, lab in (('ldb_violation_frac', 'edges violating local detailed balance'),
                       ('sigma_neg_frac', 'samples with sigma < 0'), ('neg_cell_frac', 'negative cell values')):
            if m in OWN_FLUX or k == 'neg_cell_frac':
                cells = [((_set(r, s) or {}).get('phys') or {}).get(k) if _set(r, s) is not None else None for s in COLUMNS]
            else:
                cells = ['N/A'] * len(COLUMNS)
            rows.append([LABELS[m], lab] + cells)
    return rows


def write_tables(variant, models=None, runs_dir=None, out=None):
    """tables of one variant from results/<variant>/runs/ (default) or from runs_dir; written to out (default
    results/<variant>/)"""
    if runs_dir:
        runs = {f[:-5]: json.load(open(os.path.join(runs_dir, f))) for f in sorted(os.listdir(runs_dir)) if f.endswith('.json')}
    else:
        runs = load_runs(variant)
    models = models or MODELS[variant]
    out = out or os.path.join(RESULTS, variant)
    cols = [LABEL[s] for s in COLUMNS]
    write_table(os.path.join(out, 'main'), ['Method'] + cols + ['trajectories with a negative density (of 64)', 'parameters'],
                main_rows(runs, models))
    write_table(os.path.join(out, 'physics'), ['Method', 'quantity'] + cols, physics_rows(runs, models))
    abl = [m for m in models if m.startswith('chart') or m.startswith('oracle')]
    rows = []
    for m in abl:
        if m in runs:
            r = runs[m]
            rows.append([LABELS[m], 'full model' if m == 'chart' else ABLATIONS[m]] +
                        [(_set(r, s) or {}).get('err_10T') for s in COLUMNS])
    write_table(os.path.join(out, 'ablations'), ['Method', 'what it removes'] + cols, rows)
    r = runs.get('chart')
    if r:
        rows = []
        for s, lab in LAW_SETS:
            k = (_set(r, s) or {}).get('k_state')
            if k:
                rows.append([lab, k['pearson_logk'], k['spearman_logk'], k['relerr_k'], f'{k["median_logratio"]:+.2f}', k['relerr_k_flux']])
        write_table(os.path.join(out, 'law'), ['Test set', 'Pearson r of log k', 'Spearman r of log k', 'relative L2 error of k',
                                               'median log ratio (learned / true)', 'relative L2 error of k, flux-weighted'], rows)
    rows = []
    for m in models:
        if m in runs:
            r = runs[m]
            tt = r.get('train_time_s')
            rows.append([LABELS[m], param_cell(m, r), 'N/A' if not tt else f'{tt / 60:.1f}', r.get('final_train_loss')])
    write_table(os.path.join(out, 'training'), ['Method', 'parameters', 'training time (min)', 'final training loss'], rows)
    return out
