"""README animations of the reaction-network benchmarks: reference, Neural Chart and two baselines on one test
trajectory, rolled out from the shipped or trained weights.

    python -m neuralchart.run --task schlogl --cmd gifs [--weights DIR] [--device cuda]
    python -m neuralchart.run --task schlogl --cmd gifs --track       # the tracer animation (make_track_gif)

Weights: <weights_dir>/<model>.pt (default runs/<benchmark>/, the output of `train`); a Neural Chart variant not found
there is taken from checkpoints/<benchmark>/.  The GIF is written to assets/.
"""
import argparse
import os

import numpy as np
import torch

from neuralchart import NeuralChart, VARIANTS, fast
from neuralchart.fit import rollout
from ..common import ROOT, CHECKPOINTS, LABELS
from . import config as C
from . import truth as TR
from .data import load_set, train_norm
from .models import FLOAT_MODELS, build_baseline, set_max_sub, accelerate

# benchmark -> (test set, models after the reference, field, file name)
SCENES = {
    'schlogl': ('planar_fronts', ['chart', 'flux_gnn', 'fno'], 'X', 'schlogl_front.gif'),
    'schlogl_nonideal': ('high_concentration', ['chart', 'fno', 'flux_gnn'], 'X', 'schlogl_nonideal_high_concentration.gif'),
    'schlogl_nonideal_strong': ('high_a', ['chart', 'fno', 'flux_gnn'], 'X', 'schlogl_nonideal_strong_high_a.gif'),
    'phosphorylation_cycle': ('strong_drive', ['chart', 'flux_gnn', 'fno'], 'Xp fraction', 'phosphorylation_cycle_strong_drive.gif'),
    'phosphorylation_heterogeneous': ('strong_drive', ['chart', 'flux_gnn', 'fno'], 'Xp fraction', 'phosphorylation_heterogeneous_strong_drive.gif'),
    'phosphorylation_interacting': ('high_concentration', ['chart', 'flux_gnn', 'fno'], 'Xp fraction', 'phosphorylation_interacting_high_concentration.gif'),
    'phosphorylation_saturating': ('high_concentration', ['chart', 'flux_gnn', 'fno'], 'Xp fraction', 'phosphorylation_saturating_high_concentration.gif'),
    'mapk_cascade': ('low_atp', ['chart', 'flux_gnn', 'fno'], 'state', 'mapk_cascade_low_atp.gif'),
}
TRAJ = dict(schlogl=8, schlogl_nonideal=0, schlogl_nonideal_strong=0, phosphorylation_cycle=0, phosphorylation_heterogeneous=0,
            phosphorylation_interacting=0, phosphorylation_saturating=0,
            mapk_cascade=0)   # the README trajectories
SUPTITLE = {
    'schlogl': 'Schlögl model, travelling front (a = 3, b = {b:.2f}): concentration of X',
    'schlogl_nonideal': 'Non-ideal Schlögl model, a = {a:.2f}, b = {b:.2f} (outside the training range): concentration of X',
    'schlogl_nonideal_strong': 'Schlögl model with strong crowding, a = {a:.2f}, b = {b:.2f} (training a 1.5–4.5): concentration of X',
    'phosphorylation_cycle': 'Phosphorylation cycle, drive ΔG = {dG:.2f} (training 1–3): phosphorylated fraction Xp / (X + Xp)',
    'phosphorylation_heterogeneous': 'Phosphorylation cycle on an unseen medium, drive ΔG = {dG:.2f} (training 1–3): phosphorylated fraction Xp / (X + Xp)',
    'phosphorylation_interacting': 'Interacting phosphorylation cycle, X and Xp × 1.5 at the start, ΔG = {dG:.2f}: phosphorylated fraction Xp / (X + Xp)',
    'phosphorylation_saturating': 'Saturating phosphorylation cycle, X and Xp × 3 at the start, ΔG = {dG:.2f}: phosphorylated fraction Xp / (X + Xp)',
    'mapk_cascade': 'MAPK cascade, ATP = {atp:.2f} (training 0.5–2): 22 species (rows) at 16 sites (columns)',
}


def field(R, n):
    """the plotted scalar field (T, n, n) of states R (T, M)"""
    V = n * n
    if C.is_pc():
        x, xp = R[:, :V], R[:, V:2 * V]
        f = xp / (x + xp)
    elif C.SYSTEM == 'mapk_cascade':                    # every species at every site: (T, 22, 16)
        return np.asarray(R).reshape(R.shape[0], len(TR.STATE), V)
    else:
        f = R
    return np.asarray(f).reshape(-1, n, n)


def load_model(name, norm, device, weights_dir):
    """a Neural Chart from <weights_dir>/<name>.pt or checkpoints/<benchmark>/; a baseline from <weights_dir>/<name>.pt"""
    if name in VARIANTS:
        w = os.path.join(weights_dir or C.OUT, f'{name}.pt')
        if not os.path.exists(w):
            w = os.path.join(CHECKPOINTS, C.SYSTEM, f'{name}.pt')
        return NeuralChart.load(w, map_location=device).to(device).accelerate()
    model = build_baseline(name, C.N_TRAIN, norm).to(device)
    if name in FLOAT_MODELS:
        model = model.float()
    model.load_state_dict(torch.load(os.path.join(weights_dir or C.OUT, f'{name}.pt'), map_location=device))
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
    vmin = 0.95 * float(np.nanpercentile(ref, 0.5)) if benchmark.startswith('phosphorylation') else 0.0
    mb = comparison_gif(path, frames, titles, suptitle=SUPTITLE[benchmark].format(**fmt), times=times, fps=12,
                        vmin=vmin, vmax=vmax, extra_titles=errs, panel_in=2.4, dpi=80, time_label=lambda t: f't = {times[t]:.1f}')
    print(f'{path}: {mb:.2f} MB, trajectory {traj}', flush=True)
    return path, mb


# ------------------------------------------------------------------------------------------ tracer animations
# assets/<benchmark>_track_<test set>.gif: one test trajectory under a condition outside the training range.  The
# tracers of a panel move with the probability velocity v = J / rho of the species shown, from the transport flux J of
# that model (the reference: the true flux); species given as a tuple are summed.  Panels: the reference and the models
# with their own hyperedge flux.  speed > 1 advances the tracers that many times faster than the clock (the same in
# every panel), where the flow is too slow to see in the time span of the data.
TRACK = {
    'schlogl': dict(set='high_a_low_b', traj=5, species='X', label='X', steps=range(0, 201, 4), speed=1,
                    condition='a = {a:.2f}, b = {b:.2f} (training a 2.85–3.10, b 1.0–3.2)'),
    'schlogl_nonideal': dict(set='high_concentration', traj=0, species='X', label='X', steps=range(0, 61), speed=10,
                             condition='a = {a:.2f}, b = {b:.2f} (training a 2.85–3.10, b 1.0–3.2)'),
    'schlogl_nonideal_strong': dict(set='high_a', traj=0, species='X', label='X', steps=range(0, 61), speed=10,
                                    condition='a = {a:.2f}, b = {b:.2f} (training a 1.5–4.5, b 1.0–3.2)'),
    'phosphorylation_cycle': dict(set='strong_drive', traj=0, species='X', label='X', steps=range(0, 201, 4), speed=4,
                                  condition='drive ΔG = {dG:.2f} (training 1–3)'),
    'phosphorylation_heterogeneous': dict(set='strong_drive', traj=0, species='X', label='X', steps=range(0, 201, 4), speed=4,
                                          condition='unseen medium, drive ΔG = {dG:.2f} (training 1–3)'),
    'phosphorylation_interacting': dict(set='high_concentration', traj=0, species='X', label='X', steps=range(0, 201, 4), speed=4,
                                        condition='X and Xp × 1.5 at the start, ΔG = {dG:.2f}'),
    'phosphorylation_saturating': dict(set='high_concentration', traj=0, species='X', label='X', steps=range(0, 201, 4), speed=4,
                                       condition='X and Xp × 3 at the start, ΔG = {dG:.2f}'),
    'mapk_cascade': dict(set='low_atp', traj=0, species=('Kp', 'Kpp'), label='phosphorylated MAPK (Kp + Kpp)',
                         steps=range(10, 61), speed=1, max_age=8, n_part=300, condition='ATP = {atp:.2f} (training 0.5–2)'),
}
TRACK_MODELS = ['chart', 'crnn_mass_action', 'flux_gnn', 'ude']
TRACK_TITLE = dict(schlogl='Schlögl model', schlogl_nonideal='Non-ideal Schlögl model',
                   schlogl_nonideal_strong='Schlögl model with strong crowding',
                   phosphorylation_cycle='Phosphorylation cycle', phosphorylation_heterogeneous='Phosphorylation cycle',
                   phosphorylation_interacting='Interacting phosphorylation cycle', phosphorylation_saturating='Saturating phosphorylation cycle',
                   mapk_cascade='MAPK cascade')


def condition_text(benchmark, lab):
    """the condition of one trajectory, lab: its drive (CPU tensors)"""
    fmt = dict(a=float(lab['logab'][0].exp()) if 'logab' in lab else 0, b=float(lab['logab'][1].exp()) if 'logab' in lab else 0,
               dG=float(lab['dG']) if 'dG' in lab else 0, atp=float(lab['logc'][0].exp()) if 'logc' in lab else 0)
    return TRACK[benchmark]['condition'].format(**fmt)


@torch.no_grad()
def track_panel(model, name, states, drive, hg, species):
    """tracer-panel inputs (scripts.gif_tools.hypergraph_track) at the states (T, M): the model's own flux, or the
    reference flux when model is None, against the physical force of the condition at the same states"""
    from scripts.gif_tools import physical_force, hypergraph_track
    T = states.shape[0]
    lab = {k: v.double().cpu().expand(T, *v.shape[1:]) for k, v in drive.items()}
    c = torch.as_tensor(states).double()
    at, bt = TR.true_rates(hg, lab, c.clamp_min(1e-300))
    f = physical_force(hg, at, bt, c)
    if model is None:
        return hypergraph_track(hg, species, c, at - bt, f, at + bt)
    dev = next(iter(model.parameters())).device
    dt = torch.float32 if name in FLOAT_MODELS else torch.float64
    dr = {k: v.to(dev).to(dt).expand(T, *v.shape[1:]) for k, v in drive.items()}
    cm = c.to(dev).to(dt).clamp_min(1e-30)
    if getattr(model, 'free_flux', False):
        return hypergraph_track(hg, species, c, model.flux(cm, dr).double().cpu(), f, None)
    a, b = model.rates(cm, dr)
    a, b = a.double().cpu(), b.double().cpu()
    return hypergraph_track(hg, species, c, a - b, f, a + b)


@torch.no_grad()
def make_track_gif(benchmark, device='cuda', weights_dir=None, out=None, traj=None, data=None, models=TRACK_MODELS):
    """the tracer animation assets/<benchmark>_track_<test set>.gif from the weights in weights_dir (as make_gif); panels
    without weights are left out.  Returns the path, the size in MB and the fraction of flagged cells over the clip
    per panel."""
    from scripts.gif_tools import grid_disc
    from benchmarks.porous_media.gallery import track_gif
    if C.SYSTEM != benchmark or data:
        C.configure(benchmark, data)
    C.FAST = dict(control='sample_err', rtol=C.RTOL, precision='fp32', graphs=True)
    sc = TRACK[benchmark]
    traj = sc['traj'] if traj is None else traj
    steps = np.asarray(sc['steps'])
    sd = load_set(sc['set'], device)
    n = sd['n']
    hg = TR.hg_of(n)
    norm = train_norm()
    idx = torch.as_tensor(steps)
    drive = {k: v[traj:traj + 1] for k, v in sd['drive'].items()}
    panels = [('Reference', track_panel(None, None, sd['c'][traj].double().cpu()[idx], drive, hg, sc['species']))]
    for m in models:
        if m not in VARIANTS and not os.path.exists(os.path.join(weights_dir or C.OUT, f'{m}.pt')):
            print(f'{benchmark}: no weights for {m}, panel left out', flush=True)
            continue
        model = load_model(m, norm, device, weights_dir)
        dr = {k: v.float() for k, v in drive.items()} if m in FLOAT_MODELS else drive
        c0 = sd['c'][traj:traj + 1, 0]
        c0 = c0.float() if m in FLOAT_MODELS else c0.double()
        R, _, _ = rollout(model, dr, c0, int(steps[-1]), C.DT)
        panels.append((LABELS[m], track_panel(model, m, R[0].double().cpu()[idx], drive, hg, sc['species'])))
    lab = {k: v[0].double().cpu() for k, v in drive.items()}
    title = f'{TRACK_TITLE[benchmark]}, {condition_text(benchmark, lab)}: {sc["label"]}'
    path = os.path.join(out or os.path.join(ROOT, 'assets'), f'{benchmark}_track_{sc["set"]}.gif')
    os.makedirs(os.path.dirname(path), exist_ok=True)
    times = steps * C.DT
    note = 'tracers follow v = J / rho' + (f' ({sc["speed"]}× faster than the clock)' if sc['speed'] != 1 else '')
    mb = track_gif(path, title, panels, grid_disc(n), times, n_part=sc.get('n_part', 400), max_age=sc.get('max_age', 18),
                   width_in=12, dpi=64, colors=48, floor=0.02 * float(np.mean(panels[0][1]['R'])), speed=sc['speed'], note=note,
                   time_label=(lambda t: f't = {t:5.2f}') if C.DT < 0.1 else None)
    flagged = {p[0]: float(np.mean(p[1]['bad'])) for p in panels}
    print(f'{path}: {mb:.2f} MB, {sc["set"]} trajectory {traj}, t {times[0]:g}–{times[-1]:g}; flagged cells over the clip: '
          + ', '.join(f'{k} {100 * v:.2f} %' for k, v in flagged.items()), flush=True)
    return path, mb, flagged
