"""README animations of the reaction-network benchmarks: reference, Neural Chart and two baselines on one test
trajectory, rolled out from the shipped or trained weights.

    python -m benchmarks.schlogl.run gifs [--weights DIR] [--device cuda]

Weights: <weights_dir>/<model>.pt (default runs/<benchmark>/, the output of `train`); a Neural Chart variant not found
there is taken from checkpoints/<benchmark>/.  The GIF is written to assets/.
"""
import argparse
import os

import numpy as np
import torch

from neuralchart import chart, fast
from neuralchart.fit import rollout
from ..common import ROOT, CHECKPOINTS, LABELS
from . import config as C
from . import truth as TR
from .data import load_set, train_norm
from .models import FLOAT_MODELS, build, set_max_sub, accelerate

# benchmark -> (test set, models after the reference, field, file name)
SCENES = {
    'schlogl': ('planar_fronts', ['chart', 'flux_gnn', 'fno'], 'X', 'schlogl_front.gif'),
    'schlogl_nonideal': ('high_concentration', ['chart', 'fno', 'flux_gnn'], 'X', 'schlogl_nonideal_high_concentration.gif'),
    'phosphorylation_cycle': ('strong_drive', ['chart', 'flux_gnn', 'fno'], 'Xp fraction', 'phosphorylation_cycle_strong_drive.gif'),
    'mapk_cascade': ('low_atp', ['chart', 'flux_gnn', 'fno'], 'state', 'mapk_cascade_low_atp.gif'),
}
TRAJ = dict(schlogl=8, schlogl_nonideal=0, phosphorylation_cycle=0, mapk_cascade=0)   # the README trajectories
SUPTITLE = {
    'schlogl': 'Schlögl model, travelling front (a = 3, b = {b:.2f}): concentration of X',
    'schlogl_nonideal': 'Non-ideal Schlögl model, a = {a:.2f}, b = {b:.2f} (outside the training range): concentration of X',
    'phosphorylation_cycle': 'Phosphorylation cycle, drive ΔG = {dG:.2f} (training 1–3): phosphorylated fraction Xp / (X + Xp)',
    'mapk_cascade': 'MAPK cascade, ATP = {atp:.2f} (training 0.5–2): 22 species (rows) at 16 sites (columns)',
}


def field(R, n):
    """the plotted scalar field (T, n, n) of states R (T, M)"""
    V = n * n
    if C.SYSTEM == 'phosphorylation_cycle':
        x, xp = R[:, :V], R[:, V:2 * V]
        f = xp / (x + xp)
    elif C.SYSTEM == 'mapk_cascade':                    # every species at every site: (T, 22, 16)
        return np.asarray(R).reshape(R.shape[0], len(TR.STATE), V)
    else:
        f = R
    return np.asarray(f).reshape(-1, n, n)


def load_model(name, norm, device, weights_dir):
    model = build(name, C.N_TRAIN, norm).to(device)
    if name in FLOAT_MODELS:
        model = model.float()
    w = os.path.join(weights_dir or C.OUT, f'{name}.pt')
    if not os.path.exists(w) and chart.is_chart(C.SYSTEM, name):
        w = os.path.join(CHECKPOINTS, C.SYSTEM, f'{name}.pt')
    model.load_state_dict(torch.load(w, map_location=device))
    model.eval()
    set_max_sub(model, C.MAXSUB_EVAL)
    accelerate(model, name)
    return model


@torch.no_grad()
def make_gif(benchmark, device='cuda', weights_dir=None, out=None, traj=None, stride=2, data=None):
    """traj: the test trajectory (default: the fastest front on schlogl, the trajectory used in the README otherwise)"""
    from scripts.gif_tools import comparison_gif
    if C.SYSTEM != benchmark or data:
        C.configure(benchmark, data)
    C.FAST = dict(control='sample_err', rtol=C.RTOL, precision='fp32', graphs=True)
    sname, models, _, fname = SCENES[benchmark]
    sd = load_set(sname, device)
    n = sd['n']
    norm = train_norm()
    Y = sd['c'].double().cpu().numpy()
    if traj is None:
        traj = TRAJ[benchmark]
    drive = {k: v[traj:traj + 1] for k, v in sd['drive'].items()}
    frames, titles = [field(Y[traj], n)], ['Reference']
    for m in models:
        model = load_model(m, norm, device, weights_dir)
        dr = {k: v.float() for k, v in drive.items()} if m in FLOAT_MODELS else drive
        c0 = sd['c'][traj:traj + 1, 0]
        c0 = c0.float() if m in FLOAT_MODELS else c0.double()
        R, _, _ = rollout(model, dr, c0, Y.shape[1] - 1, C.DT)
        frames.append(field(R[0].double().cpu().numpy(), n))
        titles.append(LABELS[m])
    lab = {k: v[traj].double().cpu() for k, v in sd['drive'].items()}
    fmt = dict(a=float(lab['logab'][0].exp()) if 'logab' in lab else 0, b=float(lab['logab'][1].exp()) if 'logab' in lab else 0,
               dG=float(lab['dG']) if 'dG' in lab else 0, atp=float(lab['logc'][0].exp()) if 'logc' in lab else 0)
    ref = frames[0]
    errs = [None] + [[f'rel. error {np.linalg.norm(np.nan_to_num(f[t]) - ref[t]) / np.linalg.norm(ref[t]):.1e}' for t in range(len(ref))]
                     for f in frames[1:]]
    frames = [f[::stride] for f in frames]
    errs = [None if e is None else e[::stride] for e in errs]
    times = np.arange(Y.shape[1])[::stride] * C.DT
    path = os.path.join(out or os.path.join(ROOT, 'assets'), fname)
    os.makedirs(os.path.dirname(path), exist_ok=True)
    vmax = float(np.nanpercentile(ref, 99.5)) * 1.05
    vmin = 0.95 * float(np.nanpercentile(ref, 0.5)) if benchmark == 'phosphorylation_cycle' else 0.0
    mb = comparison_gif(path, frames, titles, suptitle=SUPTITLE[benchmark].format(**fmt), times=times, fps=12,
                        vmin=vmin, vmax=vmax, extra_titles=errs, panel_in=2.4, dpi=80, time_label=lambda t: f't = {times[t]:.1f}')
    print(f'{path}: {mb:.2f} MB, trajectory {traj}', flush=True)
    return path, mb
