"""Shared pieces of the benchmarks: model names and labels, GPU check, result files and table writers."""
import csv
import json
import os
import subprocess

import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
RESULTS = os.path.join(ROOT, 'results')
CHECKPOINTS = os.path.join(ROOT, 'checkpoints')

# model key -> label in tables and figures
LABELS = {
    'chart': 'Neural Chart',
    'chart-const-k': 'Neural Chart, constant mobility per class',
    'chart-given-mu0': 'Neural Chart, given standard potentials (oracle-type information)',
    'chart-zero-init': 'Neural Chart, no convex initialisation',
    'chart-reversible': 'Neural Chart, reversible (no drive)',
    'chart-no-excess': 'Neural Chart, no excess free energy',
    'chart-learned-chem-mu0': 'Neural Chart, learned chemostat standard free energies',
    'oracle': 'Oracle (true rates in the Neural Chart class)',
    'oracle-no-crowding': 'Oracle without crowding',
    'ude': 'UDE closure',
    'flux_gnn': 'No chart (Flux-GNN)',
    'crnn_mass_action': 'CRNN (mass action)',
    'crnn_features': 'CRNN (condition features)',
    'fno': 'FNO',
    'bsms': 'BSMS-GNN',
    'generic_fno': 'GENERIC-FNO',
    'helmfluid': 'HelmFluid',
    'persistence': 'Persistence (u(t) = u(0))',
}
# models with their own edge (hyperedge) flux: the local-detailed-balance and entropy columns apply to them
OWN_FLUX = ('chart', 'chart-const-k', 'chart-given-mu0', 'chart-zero-init', 'chart-reversible', 'chart-no-excess', 'chart-learned-chem-mu0', 'oracle', 'oracle-no-crowding', 'ude', 'flux_gnn',
            'crnn_mass_action', 'crnn_features')


def check_gpu(device):
    """refuse to start on a GPU that runs a process of another user (nvidia-smi + ps)"""
    if not str(device).startswith('cuda'):
        return
    import torch
    idx = torch.cuda.current_device() if ':' not in str(device) else int(str(device).split(':')[1])
    vis = os.environ.get('CUDA_VISIBLE_DEVICES')
    phys = int(vis.split(',')[idx]) if vis else idx
    try:
        q = subprocess.run(['nvidia-smi', '--query-compute-apps=gpu_uuid,pid', '--format=csv,noheader'],
                           capture_output=True, text=True, check=True).stdout
        uu = subprocess.run(['nvidia-smi', '--query-gpu=index,uuid', '--format=csv,noheader'],
                            capture_output=True, text=True, check=True).stdout
    except (OSError, subprocess.CalledProcessError):
        return
    uuid = {int(i): u.strip() for i, u in (l.split(',') for l in uu.strip().splitlines())}[phys]
    me = os.environ.get('USER') or os.getlogin()
    for line in q.strip().splitlines():
        u, pid = (x.strip() for x in line.split(','))
        if u != uuid:
            continue
        owner = subprocess.run(['ps', '-o', 'user=', '-p', pid], capture_output=True, text=True).stdout.strip()
        if owner and owner != me:
            raise RuntimeError(f'GPU {phys} is used by another user ({owner}, pid {pid}); pick a free GPU')


def param_cell(model, run):
    """the parameter count of a table row; the oracles have no trainable parameters"""
    if model.startswith('oracle'):
        return '0 (true rates)'
    return f'{run["params"]:,}'


def fmt(x):
    """a table cell: N/A, diverged, or three significant figures"""
    if x is None:
        return 'N/A'
    if isinstance(x, str):
        return x
    if not np.isfinite(x) or x > 1e3:
        return 'diverged'
    if x == 0:
        return '0'
    return f'{x:.3f}' if x >= 0.01 else f'{x:.1e}'


def write_table(path_stem, header, rows):
    """write <path_stem>.md (markdown) and <path_stem>.csv; rows are lists of cells (first cell: the row label)"""
    os.makedirs(os.path.dirname(path_stem), exist_ok=True)
    md = ['| ' + ' | '.join(header) + ' |', '|' + '---|' * len(header)]
    md += ['| ' + ' | '.join(fmt(c) if not isinstance(c, str) else c for c in r) + ' |' for r in rows]
    open(path_stem + '.md', 'w').write('\n'.join(md) + '\n')
    with open(path_stem + '.csv', 'w', newline='') as f:
        w = csv.writer(f)
        w.writerow(header)
        for r in rows:
            w.writerow([c if isinstance(c, str) else ('' if c is None else repr(float(c))) for c in r])
    return '\n'.join(md)


def excess_table(runs, models, stem):
    """the size of the learned functions of every Neural Chart variant over the states of the training data"""
    rows = []
    for m in models:
        r = runs.get(m) or {}
        a = r.get('summary') or (r.get('extra') or {}).get('summary')
        if a:
            rows.append([LABELS[m], a['max_abs_grad_G'], a['max_abs_g'], f'{a["params_G"]:,}', f'{a["params_g"]:,}'])
    return write_table(stem, ['Method', 'largest |∇G|, training states', 'largest |g_θ|, training states', 'parameters of G_θ', 'parameters of g_θ'], rows)


def load_runs(benchmark):
    """the result files results/<benchmark>/runs/<model>.json -> {model: dict}"""
    d = os.path.join(RESULTS, benchmark, 'runs')
    out = {}
    if os.path.isdir(d):
        for f in sorted(os.listdir(d)):
            if f.endswith('.json'):
                out[f[:-5]] = json.load(open(os.path.join(d, f)))
    return out
