"""Comparison animations of the Brusselator for the README.

    python -m benchmarks.brusselator.run gifs [--weights DIR]
    python -m benchmarks.brusselator.run gifs --track [--weights DIR]     # the tracer animation (make_track_gif)
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
def model_rollout(name, sd, i, norm, device, weights=RUNS, with_model=False):
    """rollout of trajectory i in the evaluation configuration of the model; None without weights.  with_model: also
    return the model"""
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
    return (R[0].double().cpu().numpy(), model) if with_model else R[0].double().cpu().numpy()


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


# ------------------------------------------------------------------------------------------ tracer animation
# assets/brusselator_track_<test set>.gif: tracers move with the probability velocity v = J / rho of the species shown,
# from the transport flux J of every model with its own hyperedge flux (the reference: the true flux).  Trajectory 30
# of the Turing set as in brusselator_turing.gif; the pattern forms in the first 8 time units.
TRACK = dict(set='turing', traj=30, species='X', steps=list(range(0, 40, 2)) + list(range(40, 201, 10)), speed=1,
             models=['chart', 'crnn_mass_action', 'flux_gnn', 'ude'], what='Turing regime')


@torch.no_grad()
def track_panel(model, states, logab, hg, species):
    """tracer-panel inputs (scripts.gif_tools.hypergraph_track) at the states (T, M): the model's own flux, or the true
    flux when model is None, against the physical force of the condition at the same states"""
    from scripts.gif_tools import physical_force, hypergraph_track
    from .data import true_rates
    from .models import UDE, next_dtype
    c = torch.as_tensor(states).double()
    lab = logab.double().cpu().expand(c.shape[0], 2)
    at, bt = true_rates(hg, lab, c.clamp_min(1e-300))
    f = physical_force(hg, at, bt, c)
    if model is None:
        return hypergraph_track(hg, species, c, at - bt, f, at + bt)
    dev = next(iter(model.parameters())).device
    cm, dr = c.to(dev).to(next_dtype(model)), dict(logab=lab.to(dev).to(next_dtype(model)))
    if isinstance(model, UDE):
        return hypergraph_track(hg, species, c, model.flux(cm, dr).double().cpu(), f, None)
    a, b = model.rates(cm, dr)
    a, b = a.double().cpu(), b.double().cpu()
    return hypergraph_track(hg, species, c, a - b, f, a + b)


def make_track_gif(device='cuda', data=None, weights=RUNS, scene=TRACK):
    """the tracer animation of the scene; panels without weights are left out.  Returns the path, the size in MB and
    the fraction of flagged cells over the clip per panel."""
    from scripts.gif_tools import grid_disc
    from benchmarks.porous_media.gallery import track_gif
    from .data import hg_of
    dkw = {} if data is None else dict(data=data)
    norm = train_norm(**dkw)
    sd = load_set(scene['set'], device, **dkw)
    i, idx = scene['traj'], np.asarray(scene['steps'])
    n = sd['n']
    hg = hg_of(n)
    logab = sd['logab'][i:i + 1]
    panels = [('Reference', track_panel(None, sd['c'][i].double().cpu().numpy()[idx], logab, hg, scene['species']))]
    for m in scene['models']:
        out = model_rollout(m, sd, i, norm, device, weights, with_model=True)
        if out is None:
            print(f'brusselator track: no weights for {m}, panel left out', flush=True)
            continue
        panels.append((LABELS[m], track_panel(out[1], out[0][idx], logab, hg, scene['species'])))
    a, b = np.exp(logab[0].double().cpu().numpy())
    path = os.path.join(ASSETS, f'brusselator_track_{scene["set"]}.gif')
    times = idx * DT
    note = 'tracers follow v = J / rho' + (f' ({scene["speed"]}× faster than the clock)' if scene['speed'] != 1 else '')
    mb = track_gif(path, f'{scene["what"]}: a = {a:.2f}, b = {b:.2f} ({TRAINING_RANGE}): {scene["species"]}', panels,
                   grid_disc(n), times, n_part=600, width_in=12, dpi=64, colors=48, speed=scene['speed'], note=note,
                   floor=0.02 * float(np.mean(panels[0][1]['R'])))
    flagged = {p[0]: float(np.mean(p[1]['bad'])) for p in panels}
    print(f'{path}: {mb:.2f} MB ({scene["set"]}, trajectory {i}, t {times[0]:g}–{times[-1]:g}); flagged cells over the '
          f'clip: ' + ', '.join(f'{k} {100 * v:.2f} %' for k, v in flagged.items()), flush=True)
    return path, mb, flagged
