"""Comparison animations for the README: one row of panels (reference, Neural Chart, baselines) per frame.

    from scripts.gif_tools import comparison_gif
    comparison_gif('assets/porous_media_drive_x8.gif', frames, titles, suptitle='...', times=t)

frames: list of panels, each an array (T, ny, nx) of one scalar field over time (NaN for a diverged rollout).
All panels share one colour scale, set by the reference (first panel).  Values below zero are drawn in a separate
colour (magenta) and counted in the panel title, so that a negative concentration or density is visible at once.
The colour map is cividis (perceptually uniform and readable with colour-vision deficiency).
"""
import io

import numpy as np

NEG_COLOR = '#d81b60'          # magenta: negative values
NAN_COLOR = '#9e9e9e'          # grey: diverged


def _render(frames, titles, t, suptitle, vmin, vmax, cmap, panel_in, dpi, time_label, extra_titles=None, cells=None):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    from matplotlib.colors import ListedColormap
    n = len(frames)
    fig, axs = plt.subplots(1, n, figsize=(panel_in * n, panel_in * 1.18), dpi=dpi)
    axs = np.atleast_1d(axs)
    cm = plt.get_cmap(cmap).copy()
    cm.set_bad(NAN_COLOR)
    negmap = ListedColormap([NEG_COLOR])
    for i, (ax, F) in enumerate(zip(axs, frames)):
        f = F[t]
        ax.imshow(np.where(f >= 0, f, np.nan) if np.isfinite(f).all() else f, cmap=cm, vmin=vmin, vmax=vmax,
                  origin='lower', interpolation='nearest')
        neg = np.isfinite(f) & (f < 0)
        if neg.any():
            ax.imshow(np.where(neg, 1.0, np.nan), cmap=negmap, vmin=0, vmax=1, origin='lower', interpolation='nearest')
        title = titles[i]
        if not np.isfinite(f).all():
            title += '\ndiverged'
        elif neg.any():
            n_neg = int(neg.sum()) if cells is None else len(np.unique(cells[neg]))
            title += f'\n{n_neg} negative cells'
        elif extra_titles is not None and extra_titles[i]:
            title += '\n' + extra_titles[i][t]
        else:
            title += '\n '
        ax.set_title(title, fontsize=9)
        ax.set_xticks([]); ax.set_yticks([])
    fig.suptitle(f'{suptitle}   {time_label(t)}', fontsize=10)
    fig.tight_layout(rect=(0, 0, 1, 0.93))
    buf = io.BytesIO()
    fig.savefig(buf, format='png', dpi=dpi)
    plt.close(fig)
    buf.seek(0)
    return buf


def comparison_gif(path, frames, titles, suptitle='', times=None, step=1, fps=10, cmap='cividis', vmin=None, vmax=None,
                   panel_in=2.3, dpi=80, time_label=None, extra_titles=None, max_mb=3.0, cells=None):
    """write a GIF with one panel per entry of frames (each (T, ny, nx)); times: physical time of every frame.
    The colour scale is [vmin, vmax] (default: 0 and the 99.5th percentile of the reference).  Returns the size in MB.
    cells: optional (ny, nx) array of cell indices (an image of an unstructured mesh), so that negative values are
    counted per cell instead of per pixel.  Frames are palette-quantised; if the file exceeds max_mb the frame stride
    is doubled and it is written again."""
    from PIL import Image
    T = frames[0].shape[0]
    ref = frames[0][np.isfinite(frames[0])]
    vmin = 0.0 if vmin is None else vmin
    vmax = float(np.percentile(ref, 99.5)) if vmax is None else vmax
    times = np.arange(T) if times is None else np.asarray(times)
    time_label = time_label or (lambda t: f't = {times[t]:.2f}')
    while True:
        imgs = []
        for t in range(0, T, step):
            buf = _render(frames, titles, t, suptitle, vmin, vmax, cmap, panel_in, dpi, time_label, extra_titles, cells)
            imgs.append(Image.open(buf).convert('RGB').quantize(colors=128, method=Image.MEDIANCUT, dither=Image.NONE))
        imgs[0].save(path, save_all=True, append_images=imgs[1:], duration=int(1000 / fps), loop=0, optimize=True)
        import os
        mb = os.path.getsize(path) / 2 ** 20
        if mb <= max_mb or step >= T:
            return mb
        step *= 2


# ------------------------------------------------------------------------------------------ tracer animations on grids
def grid_disc(n):
    """the n x n periodic grid in the form used by the tracer animation (benchmarks.porous_media.gallery.track_gif):
    cells in the unit square, edges in the order of neuralchart.graphs.grid_edge_list (n^2 horizontal, then n^2
    vertical) with displacement vectors of length 1 / n"""
    import torch
    from types import SimpleNamespace
    from neuralchart.graphs import grid_edge_list
    V = n * n
    dv = torch.zeros(2 * V, 2, dtype=torch.float64)
    dv[:V, 0] = 1.0 / n
    dv[V:, 1] = 1.0 / n
    return SimpleNamespace(V=V, g=dict(kind='grid', n=n, edges=torch.as_tensor(grid_edge_list(n)), dvec=dv,
                                       dens=torch.ones(V, dtype=torch.float64)))


def physical_force(hg, at, bt, c):
    """the physical force f = log a - log b of every hyperedge (B, H) from the true one-way rates at the states c (B, M);
    NaN where a reactant or product of the hyperedge is not positive (the force is undefined there)"""
    import torch
    c = c.double().cpu()
    pos = torch.isfinite(c) & (c > 0)
    undef = (hg.dot(1.0 * ~pos, '+') + hg.dot(1.0 * ~pos, '-')) > 0
    f = torch.log(at) - torch.log(bt)
    return torch.where(undef, torch.full_like(f, float('nan')), f)


def hypergraph_track(hg, species, states, J, f, gross=None):
    """the inputs of one tracer panel on a reaction-transport hypergraph of the n x n periodic grid
    (neuralchart.graphs.Hypergraph.reaction_transport), for a sequence of states.
      states (T, M), J (T, H) the model's own hyperedge flux, f (T, H) the physical force at those states (NaN where
      undefined), gross (T, H) = a + b of a model with forward / backward rates (None: a single edge flux)
    The local entropy production of a site is J f of its reactions plus half of J f of every transport hyperedge at it
    (all species).  A site is flagged when it is negative or when a hyperedge at the site carries flux against the force
    beyond rounding (neuralchart.diagnostics.ldb_violation, among the hyperedges with |J| > 1e-9 max |J| as in the
    evaluation).  Returns dict(R (T, V) the density of `species` (a name, or a tuple of names that are summed), J (T, 2V)
    its transport flux on the grid edges, bad (T, V) flagged sites, sigma (T, V), viol (T, H))."""
    import torch
    from neuralchart.diagnostics import ldb_violation
    m = hg.meta
    V, R = hg.V, m['n_reactions']
    E = 2 * V
    species = (species,) if isinstance(species, str) else tuple(species)
    J = torch.as_tensor(J).double().cpu()
    f = torch.as_tensor(f).double().cpu()
    gross = None if gross is None else torch.as_tensor(gross).double().cpu()
    ok = torch.isfinite(f) & torch.isfinite(J)
    val = ok & (J.abs() > 1e-9 * J.abs().nan_to_num(0).amax(-1, keepdim=True).clamp_min(1e-30))
    viol = ldb_violation(J.nan_to_num(0), f.nan_to_num(0), None if gross is None else gross.nan_to_num(0)) & val
    jf = torch.where(ok, J * f, torch.zeros_like(J))
    tr = torch.as_tensor(m['odd'])
    rx = torch.nonzero(~tr)[:, 0]; te = torch.nonzero(tr)[:, 0]
    site, et, eh = (torch.as_tensor(m[k]) for k in ('site', 'et', 'eh'))
    T = J.shape[0]
    sigma = torch.zeros(T, V, dtype=torch.float64)
    sigma.index_add_(1, site[rx], jf[:, rx]).index_add_(1, et[te], 0.5 * jf[:, te]).index_add_(1, eh[te], 0.5 * jf[:, te])
    hit = torch.zeros(T, V, dtype=torch.float64)
    vd = viol.double()
    hit.index_add_(1, site[rx], vd[:, rx]).index_add_(1, et[te], vd[:, te]).index_add_(1, eh[te], vd[:, te])
    bad = (sigma < 0) | (hit > 0)
    states = torch.as_tensor(states).double().cpu()
    rho, Jt = 0.0, 0.0
    for sp in species:
        s, d = hg.species_names.index(sp), m['diffusing'].index(sp)
        rho = rho + states[:, s * V:(s + 1) * V]
        Jt = Jt + J[:, R * V + d * E:R * V + (d + 1) * E]
    return dict(R=rho.numpy(), J=Jt.numpy(), bad=bad.numpy(), sigma=sigma.numpy(), viol=viol.numpy())
