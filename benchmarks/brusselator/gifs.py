"""Comparison animations of the Brusselator for the README.

    python -m benchmarks.brusselator.run gifs --device cuda
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
# (file, test set, which trajectory, models after the reference, title)
SCENES = [
    ('brusselator_turing.gif', 'turing', 'spatial', ['chart', 'flux_gnn', 'fno'],
     'Turing regime (below the Hopf line): concentration of X'),
    ('brusselator_oscillation.gif', 'b_far', 'spatial', ['chart', 'flux_gnn', 'ude'],
     'Oscillations at b 4.0–4.6 (training b 2.3–3.0): concentration of X'),
]


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


def make_gifs(device='cuda', data=None, weights=RUNS, scenes=SCENES, step=2):
    from scripts.gif_tools import comparison_gif
    dkw = {} if data is None else dict(data=data)
    norm = train_norm(**dkw)
    os.makedirs(ASSETS, exist_ok=True)
    out = []
    for fname, sname, kind, models, title in scenes:
        sd = load_set(sname, device, **dkw)
        h = sd['homog'].numpy().astype(bool)
        i = int(np.where(~h)[0][0] if kind == 'spatial' else np.where(h)[0][0])
        n = sd['n']; V = n * n
        Y = sd['c'][i].double().cpu().numpy()
        frames, titles = [Y[::step, :V].reshape(-1, n, n)], ['Reference']
        for m in models:
            R = model_rollout(m, sd, i, norm, device, weights)
            if R is None:
                print(f'{fname}: no weights for {m}, panel left out', flush=True)
                continue
            frames.append(R[::step, :V].reshape(-1, n, n))
            err = np.linalg.norm(R[-1] - Y[-1]) / np.linalg.norm(Y[-1])
            titles.append(f'{LABELS[m]}\nerror at 10T {err:.2g}' if np.isfinite(err) else LABELS[m])
        times = np.arange(Y.shape[0])[::step] * DT
        mb = comparison_gif(os.path.join(ASSETS, fname), frames, titles, suptitle=title, times=times)
        print(f'{fname}: {mb:.2f} MB', flush=True)
        out.append((fname, mb))
    return out
