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
