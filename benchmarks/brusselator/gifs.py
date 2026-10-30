"""Comparison animations of the Brusselator for the README.

    python -m benchmarks.brusselator.run gifs [--weights DIR]
The Neural Chart comes from checkpoints/brusselator.  Baseline weights are not shipped: train them with
`run train --model <name>` (written to runs/brusselator) or pass --weights DIR with DIR/<name>.pt.  Panels without
weights are left out.
"""
import os

import numpy as np
import torch

from benchmarks.common import LABELS
from neuralchart.fit import rollout
from .config import CHECKPOINTS, RUNS, DT, N_TRAIN, MAXSUB_EVAL, RTOL
from .data import load_set, train_norm
from .models import build
from .training import accelerate, default_fast, set_max_sub

ASSETS = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))), 'assets')
# (file, test set, trajectory, models after the reference, what the scene shows, sampling steps shown).  Trajectory 30
# of the Turing set is the spatial start whose pattern grows most (relative spatial std of X 0.40 -> 0.87); the pattern
# forms in the first 8 time units, which are shown at every sampling step.  Trajectory 30 of the a 1.3-1.5 set is the
# spatial start with the largest growth of spatial structure while it oscillates (0.40 -> 1.38).
SCENES = [
    ('brusselator_turing.gif', 'turing', 30, ['chart', 'flux_gnn', 'fno'], 'Turing regime',
     list(range(0, 40)) + list(range(40, 201, 5))),
    ('brusselator_oscillation.gif', 'a_high', 30, ['chart', 'flux_gnn', 'fno'], 'Oscillations, a and b outside training',
     list(range(0, 201, 2))),
]
TRAINING_RANGE = 'training a 0.9–1.1, b 2.3–3.0'


def weights_of(name, weights=RUNS):
    for d in (CHECKPOINTS, weights):
        f = os.path.join(d, name + '.pt')
        if os.path.exists(f):
            return f
    return None


@torch.no_grad()
def model_rollout(name, sd, i, norm, device, weights=RUNS):
    """rollout of trajectory i in the evaluation configuration of the model; None without weights"""
    f = weights_of(name, weights)
    if f is None and name not in ('oracle', 'persistence'):
        return None
    model = build(name, N_TRAIN, norm).to(device)
    if f is not None:
        model.load_state_dict(torch.load(f, map_location=device))
    model.eval()
    set_max_sub(model, MAXSUB_EVAL)
    c0, dr = sd['c'][i:i + 1, 0], {k: v[i:i + 1] for k, v in sd['drive'].items()}
    if name in ('flux_gnn', 'ude', 'fno', 'bsms', 'generic_fno'):
        model = model.float()
        c0, dr = c0.float(), {k: v.float() for k, v in dr.items()}
    accelerate(model, default_fast(name, dict(control='sample_err', rtol=RTOL, precision='fp32', graphs=True)))
    R, _, _ = rollout(model, dr, c0, sd['c'].shape[1] - 1, DT)
    return R[0].double().cpu().numpy()


def make_gifs(device='cuda', data=None, weights=RUNS, scenes=SCENES, panel_in=2.8, max_mb=2.0):
    """one GIF per scene: X on the grid at the chosen sampling steps, with the relative error of the full state
    (X and Y) against the reference in every panel"""
    from scripts.gif_tools import comparison_gif
    dkw = {} if data is None else dict(data=data)
    norm = train_norm(**dkw)
    os.makedirs(ASSETS, exist_ok=True)
    out = []
    for fname, sname, i, models, what, steps in scenes:
        idx = np.asarray(steps)
        sd = load_set(sname, device, **dkw)
        n = sd['n']; V = n * n
        a, b = np.exp(sd['logab'][i].double().cpu().numpy())
        Y = sd['c'][i].double().cpu().numpy()
        frames, titles, errs = [Y[idx, :V].reshape(-1, n, n)], ['Reference'], [None]
        for m in models:
            R = model_rollout(m, sd, i, norm, device, weights)
            if R is None:
                print(f'{fname}: no weights for {m}, panel left out', flush=True)
                continue
            frames.append(R[idx, :V].reshape(-1, n, n))
            titles.append(LABELS[m])
            e = np.linalg.norm(R[idx] - Y[idx], axis=1) / np.linalg.norm(Y[idx], axis=1)
            errs.append([f'relative error {x:.2g}' if np.isfinite(x) else 'diverged' for x in e])
        times = idx * DT
        sup = f'{what}: a = {a:.2f}, b = {b:.2f} ({TRAINING_RANGE}), concentration of X'
        mb = comparison_gif(os.path.join(ASSETS, fname), frames, titles, suptitle=sup, times=times,
                            panel_in=panel_in, extra_titles=errs, max_mb=max_mb, time_label=lambda t: f't = {times[t]:.1f}')
        std = ', '.join(f'{t_}: {F[-1].std() / abs(F[-1].mean()):.3g}' for t_, F in zip(titles, frames))
        print(f'{fname}: {mb:.2f} MB ({sname}, trajectory {i}, a {a:.3f}, b {b:.3f}); relative spatial std of X at the '
              f'last frame: {std}', flush=True)
        out.append((fname, mb))
    return out
