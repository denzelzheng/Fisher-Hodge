"""Shared rendering helpers for the qualitative Fisher-Hodge visualisations (density panels, overlays, video IO)."""
import os, sys, numpy as np, torch
import matplotlib; matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib import colors as mcolors, font_manager
FH_ROOT = os.environ.get('FH_ROOT', os.path.expanduser('~/fisher_hodge'))
sys.path.insert(0, os.path.join(FH_ROOT, 'fisher_hodge'))
from common import RUNS, DT, D, HC

VIZ = os.path.join(FH_ROOT, 'viz'); CACHE = os.path.join(VIZ, 'cache'); OUT = os.path.join(VIZ, 'out')
os.makedirs(OUT, exist_ok=True)
FONT = 'DejaVu Sans'
plt.rcParams.update({'font.family': [FONT, 'DejaVu Sans'], 'font.size': 10, 'axes.unicode_minus': False,
                     'figure.facecolor': 'white', 'savefig.facecolor': 'white'})

NEG_SOFT = np.array(mcolors.to_rgba('#00e5ff'))   # rho < 0 (mild): cyan
NEG_HARD = np.array(mcolors.to_rgba('#ff2bd6'))   # rho strongly negative: magenta
OVER = np.array(mcolors.to_rgba('#39ff14'))       # rho above the colour range (S3 overshoot): neon green
NANC = np.array(mcolors.to_rgba('#9ca3af'))       # blown up / NaN: grey
LAB = {'truth': 'Reference', 'fisher_hodge': 'Fisher–Hodge (ours)', 'euclid_gf': 'Euclidean gradient-flow GNN', 'gnn_ode': 'GNN-ODE (unstructured)',
       'jko_amort': 'Reversible free energy (JKO)', 'fisher_hodge_wfr': 'Fisher–Hodge, Fisher–Rao reaction', 'fisher_hodge_old': 'Fisher–Hodge, free reaction MLP'}
LCOL = {'truth': '#111111', 'fisher_hodge': '#2a78d6', 'euclid_gf': '#eb6834', 'gnn_ode': '#1baf7a', 'jko_amort': '#9b59b6',
        'fisher_hodge_wfr': '#2a78d6', 'fisher_hodge_old': '#eb6834'}
TRAIN_C, FLASH_C, EXTRAP_C = '#22c55e', '#ff3b30', '#9aa0a6'


def load_truth(dsys, st):
    d = torch.load(os.path.join(RUNS, 'data', f'{dsys}.pt'))[st]
    return {k: (v.numpy() if hasattr(v, 'numpy') else v) for k, v in d.items()}


def load_cache(key):
    z = np.load(os.path.join(CACHE, key + '.npz'))
    return {k: z[k] for k in z.files}


def density_rgba(X, norm, cmap, neg_hard, over=None, over_max=1.6):
    """X (n,n) -> RGBA: magma for rho>=0, cyan/magenta for rho<0, neon green above `over`, grey for NaN."""
    fin = np.isfinite(X)
    Xc = np.where(fin, X, 0.0)
    img = cmap(norm(np.clip(Xc, 0, None)))
    neg = fin & (Xc < 0)
    img[neg & (Xc >= -neg_hard)] = NEG_SOFT
    img[neg & (Xc < -neg_hard)] = NEG_HARD
    if over is not None:
        om = fin & (Xc > over)
        if om.any():
            g = np.clip((Xc[om] - over) / (over_max - over), 0, 1)[:, None]
            img[om] = (1 - g) * np.array(mcolors.to_rgba("#c8ff9e")) + g * np.array(mcolors.to_rgba("#00b300"))
    img[~fin] = NANC
    return img


def phase(t, T=1.0, flash=0.6):
    """training-window phase for border colour"""
    if t <= T + 1e-9:
        return 'train'
    if t <= T + flash:
        return 'flash'
    return 'extrap'


def set_border(ax, t, T=1.0, k=0):
    ph = phase(t, T)
    if ph == 'flash':
        c = FLASH_C if (k // 2) % 2 == 0 else '#ffb000'
        lw = 5
    else:
        c, lw = (TRAIN_C, 4) if ph == 'train' else (EXTRAP_C, 1.5)
    for s in ax.spines.values():
        s.set_edgecolor(c); s.set_linewidth(lw); s.set_visible(True)


def stat_text(X, mass_mode='mean'):
    if not np.isfinite(X).all():
        return 'diverged (NaN)', '#ff3b30'
    m = X.mean()
    mn = X.min()
    return f'∫ρ={m:.3f}\nmin ρ={mn:+.2f}', ('#ff2bd6' if mn < 0 else 'white')


class Timeline:
    """thin timeline axis: 0..tmax, vertical marker at T (end of training window), moving cursor"""
    def __init__(s, fig, rect, tmax=10.0, T=1.0):
        s.ax = fig.add_axes(rect)
        ax = s.ax
        ax.set_xlim(0, tmax); ax.set_ylim(0, 1); ax.set_yticks([])
        ax.set_xticks([0, 1, 2, 5, 10]); ax.set_xticklabels(['0', 'T=1', '2T', '5T', '10T'], fontsize=9)
        for sp in ['left', 'right', 'top']:
            ax.spines[sp].set_visible(False)
        ax.axvspan(0, T, color=TRAIN_C, alpha=0.25, lw=0)
        ax.text(T / 2, 0.5, 'training window', ha='center', va='center', fontsize=6, color='#166534')
        ax.text((T + tmax) / 2, 0.5, 'extrapolation (t > T never seen in training)', ha='center', va='center', fontsize=8, color='#555')
        s.vl = ax.axvline(T, color=TRAIN_C, lw=2)
        s.cur = ax.axvline(0, color='#111', lw=3)
        s.T = T

    def update(s, t, k=0):
        s.cur.set_xdata([t, t])
        ph = phase(t, s.T)
        if ph == 'flash':
            s.vl.set_color(FLASH_C if (k // 2) % 2 == 0 else '#ffb000'); s.vl.set_linewidth(5)
        else:
            s.vl.set_color(TRAIN_C if ph == 'train' else '#d1d5db'); s.vl.set_linewidth(2)


def fig_to_array(fig):
    fig.canvas.draw()
    a = np.asarray(fig.canvas.buffer_rgba())[..., :3].copy()
    h, w = a.shape[:2]
    return a[:h - h % 2, :w - w % 2]


def write_video(frames, path_noext, fps=10, gif_width=None, gif_every=1):
    """MP4 (H.264) + GIF.  Returns list of written paths."""
    import imageio.v2 as iio
    from PIL import Image
    out = []
    mp4 = path_noext + '.mp4'
    w = iio.get_writer(mp4, fps=fps, codec='libx264', quality=8, pixelformat='yuv420p', macro_block_size=2)
    for f in frames:
        w.append_data(f)
    w.close(); out.append(mp4)
    gif = path_noext + '.gif'
    ims = []
    for f in frames[::gif_every]:
        im = Image.fromarray(f)
        if gif_width and im.width > gif_width:
            im = im.resize((gif_width, int(round(im.height * gif_width / im.width))), Image.LANCZOS)
        ims.append(im.convert('P', palette=Image.ADAPTIVE, colors=128))
    ims[0].save(gif, save_all=True, append_images=ims[1:], duration=int(1000 * gif_every / fps), loop=0, optimize=True)
    out.append(gif)
    for p in out:
        print(f'  wrote {p}  {os.path.getsize(p) / 1e6:.1f} MB', flush=True)
    return out


def node_vec(J):
    """edge flux (...,2,n,n) -> node-centred vector (Jx, Jy) by averaging the two adjacent edges"""
    Jh, Jv = J[..., 0, :, :], J[..., 1, :, :]
    Jx = 0.5 * (Jh + np.roll(Jh, 1, -1))
    Jy = 0.5 * (Jv + np.roll(Jv, 1, -2))
    return Jx, Jy


def legend_patches():
    from matplotlib.patches import Patch
    return [Patch(color=NEG_SOFT, label='ρ < 0'), Patch(color=NEG_HARD, label='ρ ≪ 0'), Patch(color=NANC, label='NaN')]


def wrap_label(s, width=18):
    import textwrap
    return '\n'.join(textwrap.wrap(s, width))
