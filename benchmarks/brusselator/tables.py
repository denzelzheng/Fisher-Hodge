"""Tables of the Brusselator benchmark from the result files results/brusselator/runs/<model>.json.

    python -m benchmarks.brusselator.run tables [--runs DIR]
reads DIR/<model>.json (default results/brusselator/runs) and writes
results/brusselator/{main, physics, ablations, oscillation_patterns, entropy_readout, training}.{md, csv}.
"""
import json
import os

from benchmarks.common import LABELS, OWN_FLUX, write_table, fmt, param_cell
from neuralchart.chart import ABLATIONS
from .config import RESULTS
from .test_sets import TEST_SETS, LABELS as SET_LABELS

MODELS = ['chart', 'chart-learned-mu0', 'chart-zero-init', 'chart-reversible', 'oracle', 'crnn_mass_action',
          'crnn_features', 'ude', 'flux_gnn', 'bsms', 'fno', 'generic_fno', 'persistence']
MAIN = ['chart', 'oracle', 'crnn_mass_action', 'crnn_features', 'ude', 'flux_gnn', 'bsms', 'fno', 'generic_fno', 'persistence']
ABLATION_MODELS = ['chart', 'chart-learned-mu0', 'chart-zero-init', 'chart-reversible', 'oracle']


def _set(r, s):
    return (r.get('sets') or {}).get(s)


def main_table(R, models):
    header = ['Method'] + [SET_LABELS[s] for s in TEST_SETS] + ['trajectories with a negative value (max over sets)', 'parameters']
    rows = []
    for m in models:
        if m not in R:
            continue
        r = R[m]
        neg = [(_set(r, s) or {}).get('n_neg_traj') for s in TEST_SETS]
        neg = [x for x in neg if x is not None]
        rows.append([LABELS[m]] + [(_set(r, s) or {}).get('err_10T') for s in TEST_SETS]
                    + [str(max(neg)) if neg else 'N/A', param_cell(m, r)])
    return header, rows


def physics_table(R, models):
    header = ['Method', 'quantity'] + [SET_LABELS[s] for s in TEST_SETS]
    rows = []
    for m in models:
        if m not in R or m == 'persistence':
            continue
        r = R[m]
        own = m in OWN_FLUX
        for key, lab in (('ldb_violation_frac', 'hyperedges violating local detailed balance'),
                         ('sigma_neg_frac', 'samples with sigma < 0')):
            rows.append([LABELS[m], lab] + [((_set(r, s) or {}).get('phys') or {}).get(key) if own else 'N/A' for s in TEST_SETS])
        rows.append([LABELS[m], 'negative concentration values (fraction)'] + [(_set(r, s) or {}).get('neg_cell_frac') for s in TEST_SETS])
    return header, rows


def ablation_table(R):
    header = ['Variant', 'what it removes'] + [SET_LABELS[s] for s in TEST_SETS] + ['parameters']
    rows = []
    for m in ABLATION_MODELS:
        if m not in R:
            continue
        r = R[m]
        rows.append([LABELS[m], ABLATIONS.get(m, 'nothing (the method)')] + [(_set(r, s) or {}).get('err_10T') for s in TEST_SETS]
                    + [param_cell(m, r)])
    return header, rows


def oscillation_table(R, models):
    sets = [s for s in TEST_SETS if s != 'grid64']
    header = ['Method', 'quantity'] + [SET_LABELS[s] for s in sets]
    qs = [('10T error, homogeneous starts', lambda x: x.get('err_10T_homog')),
          ('period error', lambda x: (x.get('lc') or {}).get('period_err')),
          ('amplitude error', lambda x: (x.get('lc') or {}).get('amp_err')),
          ('oscillating trajectories (fraction; reference)',
           lambda x: f"{x['lc']['osc_frac']:.2f} ({x['lc']['true_osc_frac']:.2f})" if 'lc' in x else None),
          ('late amplitude, model / reference', lambda x: (x.get('lc') or {}).get('late_amp_ratio')),
          ('10T error, spatial starts', lambda x: x.get('err_10T_spatial')),
          ('dominant wavenumber matches (fraction)', lambda x: (x.get('turing') or {}).get('k_match')),
          ('pattern standard deviation, model / reference', lambda x: (x.get('turing') or {}).get('std_ratio'))]
    rows = []
    for m in models:
        if m not in R:
            continue
        for lab, f in qs:
            rows.append([LABELS[m], lab] + [f(_set(R[m], s)) if _set(R[m], s) else None for s in sets])
    return header, rows


def entropy_table(R, models):
    """sigma(t) = sum_h (a_h - b_h) log(a_h / b_h) from the model's own one-way rates on the reference states, against
    the true sigma: relative L2 error over time (median over the set) and the ratio of the time means"""
    header = ['Method', 'quantity'] + [SET_LABELS[s] for s in TEST_SETS]
    rows = []
    for m in models:
        if m not in R or not any('sigma' in (_set(R[m], s) or {}) for s in TEST_SETS):
            continue
        for key, lab in (('relerr', 'relative error of sigma(t)'), ('ratio', 'mean sigma, model / reference')):
            rows.append([LABELS[m], lab] + [((_set(R[m], s) or {}).get('sigma') or {}).get(key) for s in TEST_SETS])
    return header, rows


def training_table(R, models):
    header = ['Method', 'gradient steps', 'training time (h)', 'final training loss', 'substeps per sampling step (mean)',
              'largest substeps per sampling step']
    rows = []
    for m in models:
        if m not in R:
            continue
        r = R[m]
        if m in ('oracle', 'persistence'):
            rows.append([LABELS[m], '0 (not trained)', '–', '–', '–', '–']); continue
        one_step = m in ('fno', 'bsms', 'generic_fno')
        rows.append([LABELS[m], str(r.get('train_steps_done', r.get('iters'))), f'{(r.get("train_time_s") or 0) / 3600:.2f}',
                     r.get('final_train_loss'), '1 (one-step map)' if one_step else fmt(r.get('train_substeps_per_DT')),
                     '1' if one_step else str(r.get('train_substeps_max', 'N/A'))])
    return header, rows


def load_runs(runs=None):
    d = runs or os.path.join(RESULTS, 'runs')
    return {f[:-5]: json.load(open(os.path.join(d, f))) for f in sorted(os.listdir(d)) if f.endswith('.json')}


def make_tables(models=MODELS, out=RESULTS, runs=None):
    R = load_runs(runs)
    for stem, (h, rows) in (('main', main_table(R, [m for m in MAIN if m in models])),
                            ('physics', physics_table(R, models)),
                            ('ablations', ablation_table(R)),
                            ('oscillation_patterns', oscillation_table(R, models)),
                            ('entropy_readout', entropy_table(R, models)),
                            ('training', training_table(R, models))):
        print(write_table(os.path.join(out, stem), h, rows), '\n')
