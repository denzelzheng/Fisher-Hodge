"""Tables of the reaction-network benchmarks, written from results/<benchmark>/runs/<model>.json."""
import os

import json

from ..common import RESULTS, LABELS, OWN_FLUX, excess_table, fmt, write_table
from . import config as C

MAIN = ['chart', 'chart-given-mu0', 'flux_gnn', 'crnn_mass_action', 'crnn_features', 'ude', 'fno', 'bsms', 'generic_fno', 'persistence']
ABLATIONS = ['chart', 'chart-given-mu0', 'chart-learned-chem-mu0', 'chart-const-k', 'chart-no-excess', 'chart-zero-init', 'chart-no-projection', 'chart-reversible', 'flux_gnn']


def _cols():
    return list(C.TEST_SETS[C.SYSTEM].items())


def _err(r, s):
    v = r['sets'].get(s)
    return None if v is None else v.get('err_10T')


def main_table(R, models, stem):
    cols = _cols()
    rows = []
    for m in models:
        if m not in R:
            continue
        r = R[m]
        neg = [r['sets'][s]['n_neg_traj'] for s, _ in cols if r['sets'].get(s)]
        rows.append([LABELS[m]] + [_err(r, s) for s, _ in cols] + [str(max(neg)) if neg else 'N/A', f'{r["params"]:,}'])
    return write_table(stem, ['Method'] + [lab for _, lab in cols] + ['trajectories with a negative value', 'parameters'], rows)


def physics_table(R, models, stem):
    cols = _cols()
    rows = []
    for m in models:
        if m not in R or m == 'persistence':
            continue
        S = R[m]['sets']
        for key, lab in (('ldb_violation_frac', 'hyperedges violating local detailed balance'),
                         ('sigma_neg_frac', 'samples with σ < 0'), ('neg_cell_frac', 'negative values')):
            if key == 'neg_cell_frac':
                cells = [S[s].get('neg_cell_frac') if S.get(s) else None for s, _ in cols]
            elif m in OWN_FLUX:
                cells = [(S[s].get('phys') or {}).get(key) if S.get(s) else None for s, _ in cols]
            else:
                cells = ['N/A'] * len(cols)
            rows.append([LABELS[m], lab] + cells)
    return write_table(stem, ['Method', 'quantity'] + [lab for _, lab in cols], rows)


def readout_table(R, models, stem):
    """entropy production and cycle affinity of the models with their own rates, on the reference states"""
    cols = _cols()
    rows = []
    for m in models:
        if m not in R or m not in OWN_FLUX or m == 'ude':
            continue
        S = R[m]['sets']
        rows.append([LABELS[m], 'σ, relative error (t > 0)'] + [(S[s].get('sigma') or {}).get('relerr') if S.get(s) else None for s, _ in cols])
        rows.append([LABELS[m], 'cycle affinity, model; reference'] +
                    [f'{fmt(S[s]["cycle_affinity"]["model_mean"])}; {fmt(S[s]["cycle_affinity"]["true_mean"])}'
                     if S.get(s) and 'cycle_affinity' in S[s] else 'N/A' for s, _ in cols])
    return write_table(stem, ['Method', 'quantity'] + [lab for _, lab in cols], rows)


def lngamma_table(R, stem):
    rows = []
    for m in ABLATIONS:
        g = (R.get(m, {}).get('extra') or {}).get('lngamma')
        if g:
            rows.append([LABELS[m], g['rmse_id'], g['maxerr_id'], g['rmse_ood'], g['slope_id']])
    return write_table(stem, ['Method', 'RMSE, training range', 'largest error, training range',
                              'RMSE, extension to the high-concentration range', 'slope, learned against true'], rows)


def lngamma_curve_table(R, stem, models=('chart', 'chart-given-mu0')):
    """the learned ln gamma against the true one at 12 concentrations across the training range"""
    G = [(m, (R.get(m, {}).get('extra') or {}).get('lngamma')) for m in models]
    G = [(m, g) for m, g in G if g and 'table' in g]
    if not G:
        return None
    x = G[0][1]['table']['x']
    rows = [[f'{xi:.3f}', G[0][1]['table']['true'][i]] + [g['table']['model'][i] for _, g in G] for i, xi in enumerate(x)]
    return write_table(stem, ['concentration x', 'true ln γ'] + [f'{LABELS[m]}, learned ln γ' for m, _ in G], rows)


def k_law_table(R, models, stem):
    """the learned rate constant against the true one on unseen media, per test set and hyperedge class"""
    rows = []
    for m in models:
        S = (R.get(m) or {}).get('sets', {})
        for s, lab in _cols():
            for cl, g in ((S.get(s) or {}).get('k_law') or {}).items():
                rows.append([LABELS[m], lab, cl, g['pearson'], g['spearman'], g['rmse'], f'{g["median_diff"]:+.2f}', g['std_true']])
    return write_table(stem, ['Method', 'test set', 'hyperedges', 'Pearson r of log k', 'Spearman r of log k', 'RMSE of log k',
                              'median log ratio (learned / true)', 'standard deviation of the true log k'], rows)


def excess_law_table(R, models, stem):
    """the learned excess potential against the true one on the test states, per test set and species"""
    rows = []
    for m in models:
        S = (R.get(m) or {}).get('sets', {})
        for s, lab in _cols():
            for sp, g in ((S.get(s) or {}).get('excess_law') or {}).items():
                rows.append([LABELS[m], lab, sp, g['pearson'], g['rmse'], g['slope'], g['std_true']])
    return write_table(stem, ['Method', 'test set', 'species', 'Pearson r', 'RMSE', 'slope, learned against true', 'standard deviation of the true excess potential'], rows)


def mapk_table(R, models, stem):
    cols = _cols()
    rows = []
    for m in models:
        if m not in R:
            continue
        S = R[m]['sets']
        rows.append([LABELS[m], 'moieties, largest relative drift'] + [S[s]['moiety_drift']['all'] if S.get(s) and 'moiety_drift' in S[s] else None for s, _ in cols])
        rows.append([LABELS[m], 'Kpp fraction at 10T, mean absolute error'] + [S[s].get('kpp_end_err') if S.get(s) else None for s, _ in cols])
    return write_table(stem, ['Method', 'quantity'] + [lab for _, lab in cols], rows)


def front_table(R, models, stem):
    rows = []
    for m in models:
        f = ((R.get(m) or {}).get('sets', {}).get('planar_fronts') or {}).get('front')
        if f:
            nan = lambda x: 'no zero crossing' if x is None or x != x else x
            rows.append([LABELS[m], nan(f['b_star_model']), nan(f['b_star_true']), f['speed_relerr_median']])
    return write_table(stem, ['Method', 'b where the front stops, model', 'reference', 'front speed, median relative error'], rows)


def load_runs(runs):
    """{model: run dict} from the <model>.json files of a directory"""
    return {f[:-5]: json.load(open(os.path.join(runs, f))) for f in sorted(os.listdir(runs))
            if f.endswith('.json') and not f.endswith('.ckpt.json')}


def make_tables(runs=None):
    """the tables of results/<benchmark>/ from results/<benchmark>/runs/; with runs=DIR (fresh runs, e.g.
    runs/<benchmark>/) the tables are written to DIR/tables/ instead"""
    if runs is None:
        R, out = load_runs(os.path.join(RESULTS, C.SYSTEM, 'runs')), os.path.join(RESULTS, C.SYSTEM)
    else:
        R, out = load_runs(runs), os.path.join(runs, 'tables')
    main_table(R, MAIN, os.path.join(out, 'main'))
    physics_table(R, MAIN, os.path.join(out, 'physics'))
    main_table(R, ABLATIONS, os.path.join(out, 'ablations'))
    physics_table(R, ABLATIONS, os.path.join(out, 'ablations_physics'))
    readout_table(R, MAIN[:-1] + ABLATIONS[2:-1], os.path.join(out, 'thermodynamic_readouts'))
    excess_table(R, MAIN[:2] + ABLATIONS[2:-1], os.path.join(out, 'excess'))
    if C.nonideal():
        lngamma_table(R, os.path.join(out, 'activity_coefficient'))
        lngamma_curve_table(R, os.path.join(out, 'activity_coefficient_curve'))
    if C.SYSTEM == 'mapk_cascade':
        mapk_table(R, MAIN + ABLATIONS[2:-1], os.path.join(out, 'conservation_and_output'))
    if C.is_pc() or C.SYSTEM == 'mapk_cascade':
        k_law_table(R, ['chart', 'chart-given-mu0', 'chart-learned-chem-mu0', 'chart-const-k'], os.path.join(out, 'law'))
    excess_law_table(R, ['chart', 'chart-given-mu0', 'chart-learned-chem-mu0'], os.path.join(out, 'excess_law'))
    if C.SYSTEM == 'schlogl':
        front_table(R, MAIN + ABLATIONS[2:-1], os.path.join(out, 'fronts'))
    print(open(os.path.join(out, 'main.md')).read())
