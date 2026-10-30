"""Tables of the reaction-network benchmarks, written from results/<benchmark>/runs/<model>.json."""
import os

import json

from ..common import RESULTS, LABELS, OWN_FLUX, fmt, write_table, param_cell
from . import config as C

MAIN = ['chart', 'oracle', 'crnn_mass_action', 'crnn_features', 'ude', 'flux_gnn', 'fno', 'bsms', 'generic_fno', 'persistence']
ABLATIONS = ['chart', 'chart-ideal', 'chart-no-varpro', 'chart-learned-mu0', 'chart-zero-init', 'chart-reversible', 'oracle']


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
        rows.append([LABELS[m]] + [_err(r, s) for s, _ in cols] + [str(max(neg)) if neg else 'N/A', param_cell(m, r)])
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
        rows.append([LABELS[m], 'σ, relative error'] + [(S[s].get('sigma') or {}).get('relerr') if S.get(s) else None for s, _ in cols])
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
            rows.append([LABELS[m], f['b_star_model'], f['b_star_true'], f['speed_relerr_median']])
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
    readout_table(R, MAIN[:-1] + ABLATIONS[1:], os.path.join(out, 'thermodynamic_readouts'))
    if C.SYSTEM == 'schlogl_nonideal':
        lngamma_table(R, os.path.join(out, 'activity_coefficient'))
    if C.SYSTEM == 'mapk_cascade':
        mapk_table(R, MAIN + ABLATIONS[1:], os.path.join(out, 'conservation_and_output'))
    if C.SYSTEM == 'schlogl':
        front_table(R, MAIN + ABLATIONS[1:], os.path.join(out, 'fronts'))
    print(open(os.path.join(out, 'main.md')).read())
