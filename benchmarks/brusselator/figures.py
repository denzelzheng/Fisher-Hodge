"""Static figures of the Brusselator benchmark for the README.

    python -m benchmarks.brusselator.run figures
writes assets/brusselator_phase_portraits.png (the limit cycle of homogeneous starts in every test set) and
assets/brusselator_entropy_b_far.png (local entropy production on one trajectory of "b 4.0–4.6").  The Neural Chart
variants come from checkpoints/brusselator; baseline weights from --weights DIR (default runs/brusselator).  Models
without weights are left out.
"""
import os

import numpy as np
import torch

from benchmarks.common import LABELS
from neuralchart.diagnostics import ldb_violation
from neuralchart.fit import rollout
from .config import CHECKPOINTS, RUNS, DT, N_TRAIN, MAXSUB_EVAL, RTOL
from .data import load_set, train_norm, true_rates, hg_of
from .evaluate import site_map
from .models import build, UDE
from .test_sets import TEST_SETS, LABELS as SET_LABELS
from .training import accelerate, default_fast, set_max_sub

ASSETS = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))), 'assets')
# Okabe-Ito colours (readable with colour-vision deficiency)
COLORS = {'reference': '#000000', 'chart': '#0072B2', 'chart-reversible': '#56B4E9', 'crnn_mass_action': '#009E73',
          'flux_gnn': '#E69F00', 'fno': '#CC79A7', 'ude': '#D55E00'}
PHASE_MODELS = ['chart', 'chart-reversible', 'crnn_mass_action', 'flux_gnn', 'fno', 'ude']
ENTROPY_MODELS = ['chart', 'chart-reversible', 'crnn_mass_action', 'flux_gnn', 'ude']
FLOAT32 = ('flux_gnn', 'ude', 'fno', 'bsms', 'generic_fno')
VIOL_COLOR = '#009E73'


def trained(name, norm, device, weights=RUNS):
    """the model in its evaluation configuration, or None without weights"""
    f = next((os.path.join(d, name + '.pt') for d in (CHECKPOINTS, weights) if os.path.exists(os.path.join(d, name + '.pt'))), None)
    if f is None and name not in ('oracle', 'persistence'):
        return None
    model = build(name, N_TRAIN, norm).to(device)
    if f is not None:
        model.load_state_dict(torch.load(f, map_location=device))
    model.eval()
    set_max_sub(model, MAXSUB_EVAL)
    if name in FLOAT32:
        model = model.float()
    accelerate(model, default_fast(name, dict(control='sample_err', rtol=RTOL, precision='fp32', graphs=True)))
    return model


def _inputs(name, sd, i):
    c0, dr = sd['c'][i:i + 1, 0], {k: v[i:i + 1] for k, v in sd['drive'].items()}
    if name in FLOAT32:
        c0, dr = c0.float(), {k: v.float() for k, v in dr.items()}
    return c0, dr


@torch.no_grad()
def phase_portraits(device, data_kw, weights, path):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    norm = train_norm(**data_kw)
    sets = [s for s in TEST_SETS if s != 'grid64']          # the 64 x 64 set has spatial starts only
    curves = {s: {} for s in sets}
    for s in sets:
        sd = load_set(s, device, **data_kw)
        i = int(np.where(sd['homog'].numpy())[0][0])
        V = sd['n'] ** 2
        curves[s]['reference'] = sd['c'][i].double().cpu().numpy()
        for m in PHASE_MODELS:
            model = trained(m, norm, device, weights)
            if model is None:
                print(f'phase portraits: no weights for {m}', flush=True)
                continue
            c0, dr = _inputs(m, sd, i)
            R, _, _ = rollout(model, dr, c0, sd['c'].shape[1] - 1, DT)
            curves[s][m] = R[0].double().cpu().numpy()
    fig, axs = plt.subplots(1, len(sets), figsize=(3.0 * len(sets), 3.2), constrained_layout=True)
    for ax, s in zip(axs, sets):
        for key in ['reference'] + PHASE_MODELS:
            if key not in curves[s]:
                continue
            C = curves[s][key]
            V = C.shape[1] // 2
            x, y = C[:, :V].mean(1), C[:, V:].mean(1)
            lab = 'Reference' if key == 'reference' else LABELS[key]
            kw = dict(color=COLORS[key], lw=3.2 if key == 'reference' else 1.2, label=lab,
                      ls='-' if key in ('reference', 'chart') else '--' if key in ('chart-reversible', 'crnn_mass_action') else '-',
                      zorder=3 if key == 'chart' else 2, alpha=1.0 if key in ('reference', 'chart') else 0.85)
            if s == 'turing':                             # a homogeneous start decays to the fixed point
                t = np.arange(len(x)) * DT
                ax.plot(t, x, **kw)
            else:                                         # the second half of the rollout: the limit cycle
                h = len(x) // 2
                ax.plot(x[h:], y[h:], **kw)
        if s == 'turing':
            ax.set_xlabel('time'); ax.set_ylabel('site-mean X')
            ax.set_title(f'{SET_LABELS[s]}\nhomogeneous start decays', fontsize=9)
        else:
            ax.set_xlabel('site-mean X'); ax.set_ylabel('site-mean Y')
            ax.set_title(f'{SET_LABELS[s]}\nlimit cycle, t = 20–40', fontsize=9)
    h, l = axs[0].get_legend_handles_labels()
    fig.legend(h, l, loc='lower center', ncol=4, fontsize=8, bbox_to_anchor=(0.5, -0.14), frameon=False)
    fig.savefig(path, dpi=110, bbox_inches='tight')
    plt.close(fig)


@torch.no_grad()
def entropy_maps(device, data_kw, weights, path, set_name='b_far', t=100):
    """local entropy production J_h f_h at time t (a reaction to its site, a transport edge half to each end), with
    f the true force evaluated on each model's own state; sites touched by a hyperedge with flux against the force
    are marked"""
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    from matplotlib.colors import SymLogNorm
    norm = train_norm(**data_kw)
    sd = load_set(set_name, device, **data_kw)
    i = int(np.where(~sd['homog'].numpy())[0][0])
    n = sd['n']; hg = hg_of(n)
    lab = sd['drive']['logab'][i:i + 1].double().cpu()
    c = sd['c'][i:i + 1, t].double().cpu()
    at, bt = true_rates(hg, lab, c)
    panels = [('Reference', site_map(hg, ((at - bt) * torch.log(at / bt))[0]).numpy(), None, None)]
    for m in ENTROPY_MODELS:
        model = trained(m, norm, device, weights)
        if model is None:
            print(f'entropy maps: no weights for {m}', flush=True)
            continue
        c0, dr = _inputs(m, sd, i)
        R, _, _ = rollout(model, dr, c0, t, DT)
        cm = R[:, -1]
        a, b = model.rates(cm, dr)
        a, b = a.double().cpu()[0], b.double().cpu()[0]
        J = a - b
        cmd = cm.double().cpu().clamp_min(1e-300)
        at, bt = true_rates(hg, lab, cmd)
        f = torch.log(at / bt)[0]
        viol = ldb_violation(J, f, gross=None if isinstance(model, UDE) else a + b)
        panels.append((LABELS[m], site_map(hg, torch.nan_to_num(J * f)).numpy(), site_map(hg, viol.double(), half=False).numpy() > 0,
                       float(viol.double().mean())))
    ref = panels[0][1]
    lim = float(np.abs(ref).max())
    nm = SymLogNorm(linthresh=1e-3 * lim, vmin=-lim, vmax=lim, base=10)
    ncol = 3
    nrow = (len(panels) + ncol - 1) // ncol
    fig, axs = plt.subplots(nrow, ncol, figsize=(2.6 * ncol + 0.6, 2.75 * nrow), constrained_layout=True)
    axs = np.atleast_1d(axs).ravel()
    for ax, (title, q, vs, frac) in zip(axs, panels):
        im = ax.imshow(q.reshape(n, n), cmap='PuOr_r', norm=nm, origin='lower', interpolation='nearest')
        if vs is not None and vs.any():
            yy, xx = np.nonzero(vs.reshape(n, n))
            ax.scatter(xx, yy, s=3, c=VIOL_COLOR, marker='o', linewidths=0)
        neg = float((q < 0).mean()) * 100
        sub = '' if frac is None else f'\n{100 * frac:.1f} % of hyperedges against the force'
        ax.set_title(f'{title}{sub}\n{neg:.1f} % of sites with sigma < 0', fontsize=8)
        ax.set_xticks([]); ax.set_yticks([])
    for ax in axs[len(panels):]:
        ax.set_axis_off()
    cb = fig.colorbar(im, ax=axs.tolist(), shrink=0.8)
    e = int(np.floor(np.log10(lim)))
    cb.set_ticks([-10 ** e, -10 ** (e - 2), 0, 10 ** (e - 2), 10 ** e])
    cb.ax.tick_params(labelsize=7)
    cb.set_label('local entropy production (true force)', fontsize=8)
    fig.suptitle(f'{SET_LABELS[set_name]}, t = {t * DT:g} (5T).  Green dots: sites touched by a hyperedge whose flux runs '
                 'against the force', fontsize=9)
    fig.savefig(path, dpi=110)
    plt.close(fig)


def make_figures(device='cuda', data=None, weights=RUNS):
    dkw = {} if data is None else dict(data=data)
    os.makedirs(ASSETS, exist_ok=True)
    out = []
    for fname, fn in (('brusselator_phase_portraits.png', phase_portraits), ('brusselator_entropy_b_far.png', entropy_maps)):
        p = os.path.join(ASSETS, fname)
        fn(device, dkw, weights, p)
        out.append((fname, os.path.getsize(p) / 2 ** 20))
        print(f'{fname}: {out[-1][1]:.2f} MB', flush=True)
    return out
