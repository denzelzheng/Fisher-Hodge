"""Flow animations and the constitutive-law figure of the porous-media benchmark.

For one unseen medium and one trajectory per scene, every model with its own edge flux J is rolled out and drawn in
two ways.
  track     Tracer particles move with the probability velocity v = J / rho of each model.  The velocity of a cell is
            sum_e J_e d_e / (2 c_i) over its edges.  A particle turns magenta in a cell where the local entropy
            production is negative or where an incident edge carries flux against the physical force
            (J f < -1e-6 (a + b) |f|, neuralchart.diagnostics).
  entropy   The local entropy production sigma_i (half of J_e f_e to each end of every edge, f the physical force of the
            test condition).  Cells with sigma < 0 and edges with flux against the force are magenta.
Panels: Reference | Neural Chart | Flux-GNN | UDE closure.  The reference flux is the fine-grid flux of the reference
solution, summed over the fine pixel faces that separate two cells of the discretisation.  The reference is never
flagged.

  medium_and_law.png   an unseen medium: the pore map, the true mobility, and the learned mobility of the Neural Chart
                       on the 32 x 32 grid and on the point cloud (same parameters, no retraining).

    python -m benchmarks.porous_media.run gallery
"""
import io
import math
import os

import numpy as np
import torch

from neuralchart import fast
from neuralchart.diagnostics import ldb_violation
from neuralchart.fit import rollout
from benchmarks.common import CHECKPOINTS, ROOT, RESULTS, LABELS, check_gpu
from .data import (NF, DT, D, T_TEST, POR_TRAIN, POR_LOW, RASTER, make_pool, fine_inputs, fine_edge_mobility,
                   truth_mobility_raw, M_FLOOR, ref_flux, escape_max, projector, load, add_norm)
from .models import build, rebind
from .evaluate import phys_force, local_sigma, oracle_drive
from .test_sets import CONDITIONS, LABEL

MAGENTA = '#d81b60'                 # physics violation (the colour of negative values in the other animations)
TRACER = '#56b4e9'                  # tracers in cells without a violation (Okabe-Ito sky blue)
MODELS = ['chart', 'flux_gnn', 'ude']
STRIDE = 7                          # one frame every 7 steps of DT: 29 frames over 10 time units
SCENES = {
    'porous_media': ['training_conditions', 'drive_x8', 'temperature_x4', 'point_cloud'],
    'porous_media_crowding': ['training_conditions', 'drive_x8'],
}
TITLE = dict(porous_media='Porous media', porous_media_crowding='Porous media with crowding')
SCENE_TITLE = dict(training_conditions='training conditions', drive_x8='drive ×8', temperature_x4='temperature ×4',
                   point_cloud='point cloud of 1024 cells (no retraining)')
# the three animations shown in the README (assets/); the others go to results/porous_media/gallery/
README_PICKS = ('porous_media_track_drive_x8', 'porous_media_entropy_drive_x8', 'porous_media_track_point_cloud')
RUN_SETTINGS = {'chart': dict(control='sample_err', rtol=0.3, precision='fp32'),
                'flux_gnn': dict(control='sample_err', rtol=0.3, precision='fp32'),
                'ude': dict(control='err_only', rtol=1e-3, precision='fp32')}


# ------------------------------------------------------------------------------------------ fluxes on a discretisation
def pixel_cells(disc):
    """cell index of every fine pixel (NF, NF), rows = y"""
    g = disc.g
    if g['kind'] == 'grid':
        f = NF // g['n']
        a = np.arange(NF) // f
        return a[:, None] * g['n'] + a[None, :]
    lab = g['lab'].cpu().numpy().reshape(RASTER, RASTER)
    s = RASTER // NF
    return lab[s // 2::s, s // 2::s]


def face_to_edge(disc):
    """sparse (H, 2 NF^2) map from the fine face fluxes (horizontal, then vertical faces) to the edge fluxes of the
    discretisation in model units (dc/dt = S J): the fine faces between two cells i, j are summed, signed by the edge
    orientation, and scaled by V / NF^2 (the state is cell mass / mean cell volume)"""
    cell = pixel_cells(disc)
    E = disc.g['edges'].cpu().numpy()
    V = disc.V
    key = {}
    for e, (i, j) in enumerate(E):
        key[(int(i), int(j))] = (e, 1.0)
        key[(int(j), int(i))] = (e, -1.0)
    rows, cols, vals = [], [], []
    for d, nb in ((0, np.roll(cell, -1, 1)), (1, np.roll(cell, -1, 0))):
        a, b = cell.ravel(), nb.ravel()
        for p in np.nonzero(a != b)[0]:
            hit = key.get((int(a[p]), int(b[p])))
            if hit is not None:
                rows.append(hit[0]); cols.append(d * NF * NF + p); vals.append(hit[1] * V / NF ** 2)
    idx = torch.as_tensor(np.array([rows, cols]))
    return torch.sparse_coo_tensor(idx, torch.as_tensor(vals, dtype=torch.float64), (disc.H, 2 * NF * NF)).coalesce()


@torch.no_grad()
def reference_scene(variant, key, j, disc, meta, device, frames):
    """re-simulate trajectory j of the scene on the fine grid; returns the states (len(frames), V) and the edge fluxes
    (len(frames), H) of the discretisation at the frame steps"""
    cond = CONDITIONS['training_conditions' if key == 'point_cloud' else key]
    pool = make_pool(7003, 16, 4, POR_LOW) if cond['pool'] == 'low' else make_pool(7002, 16, 4, POR_TRAIN)
    sub = dict(pool, Upar=pool['Upar'][j:j + 1], Ppar=pool['Ppar'][j:j + 1], Rpar=pool['Rpar'][j:j + 1])
    U, dpsi, r0 = fine_inputs(sub, device)
    med = int(pool['med'][j])
    chi = torch.as_tensor(pool['pore'][med:med + 1], dtype=torch.float64, device=device)
    Me = fine_edge_mobility(meta['C'] * truth_mobility_raw(chi) + M_FLOOR)
    T = torch.full((1,), cond['T'], dtype=torch.float64, device=device)
    amp = torch.full((1,), cond['amp'], dtype=torch.float64, device=device)
    kk = D * NF ** 2 * T.view(-1, 1, 1, 1) * Me
    phi = U / T.view(-1, 1, 1)
    A = amp.view(-1, 1, 1, 1) * dpsi / T.view(-1, 1, 1, 1)
    crowd = variant == 'porous_media_crowding'
    ns = int(math.ceil(DT * escape_max(kk, phi, A)))
    dt = DT / ns
    proj = projector({k: (v.to(device) if torch.is_tensor(v) else v) for k, v in disc.g.items()})
    M = face_to_edge(disc).to(device)
    f = lambda r: ref_flux(r, kk, phi, A, crowd)
    rho = r0.clone()
    R, J = [], []
    for step in range(T_TEST + 1):
        if step > 0:
            for _ in range(ns):
                k1 = f(rho)[0]; k2 = f(rho + 0.5 * dt * k1)[0]; k3 = f(rho + 0.5 * dt * k2)[0]; k4 = f(rho + dt * k3)[0]
                rho = rho + dt / 6 * (k1 + 2 * k2 + 2 * k3 + k4)
        if step in frames:
            R.append(proj(rho)[0].cpu()); J.append((M @ f(rho)[1].reshape(-1, 1).double())[:, 0].cpu())
    return torch.stack(R).numpy(), torch.stack(J).numpy()


def _weights(variant, model, weights):
    p = os.path.join(CHECKPOINTS, variant, model + '.pt')
    return p if os.path.exists(p) else os.path.join(weights or os.path.join(ROOT, 'runs', variant), model + '.pt')


@torch.no_grad()
def model_scene(variant, name, data, key, j, device, frames, weights=None):
    """roll out model `name` on trajectory j of set `key`; returns the states, the edge fluxes, the cells flagged
    (sigma < 0 or an incident edge against the force), the local entropy production and the violating edges"""
    meta = data['meta']
    sd = data[key]
    disc = data['discs'][sd['disc']]
    m = build(name, meta, data['discs']['g32']).to(device)
    m.load_state_dict(torch.load(_weights(variant, name, weights), map_location=device))
    m.eval()
    if sd['disc'] != 'g32':
        m = rebind(name, m, meta, disc, device)
    fast.accelerate(m, **RUN_SETTINGS[name])
    dr = {k: v[j:j + 1] for k, v in sd['drive'].items()}
    R, _, _ = rollout(m, dr, sd['c'][j:j + 1, 0], T_TEST, DT)
    ctx = m.prepare(dr)
    out = dict(R=[], J=[], sigma=[], bad=[], viol=[])
    E = disc.g['edges'].to(device)
    for t in frames:
        c = R[:, t].double()
        if getattr(m, 'free_flux', False):
            J, gross = m.flux(c.float(), ctx).double(), None
        else:
            a, b = m.rates(c.float(), ctx)
            J, gross = (a - b).double(), (a + b).double()
        fphys = phys_force(c, dr, disc)
        ok = torch.isfinite(fphys) & torch.isfinite(J) & (J.abs() > 1e-9 * J.abs().nan_to_num(0).amax(-1, keepdim=True).clamp_min(1e-30))
        viol = (ldb_violation(J, fphys, gross) & ok)[0]
        sg = local_sigma(c[0], J[0], {k: v[0] for k, v in dr.items() if torch.is_tensor(v)}, disc)
        bad = sg < 0
        bad[E[viol, 0]] = True; bad[E[viol, 1]] = True
        for k, v in (('R', c[0]), ('J', J[0]), ('sigma', sg), ('bad', bad), ('viol', viol)):
            out[k].append(v.cpu().numpy())
    return {k: np.stack(v) for k, v in out.items()}


def reference_sigma(R, J, dr, disc):
    F = {k: v[0] for k, v in dr.items() if torch.is_tensor(v)}
    return np.stack([local_sigma(torch.as_tensor(R[i]), torch.as_tensor(J[i]), {k: v.cpu() for k, v in F.items()}, disc).numpy()
                     for i in range(len(R))])


# ------------------------------------------------------------------------------------------ drawing helpers
class Cells:
    """cell lookup and cell-wise fields at arbitrary points: bilinear on grids, nearest site on the point cloud"""
    def __init__(s, disc):
        g = disc.g
        s.grid = g['kind'] == 'grid'
        s.dens = g['dens'].cpu().numpy()
        if s.grid:
            s.n = g['n']
        else:
            from scipy.spatial import cKDTree
            s.tree = cKDTree(g['pts'].cpu().numpy(), boxsize=1.0)
            lab = g['lab'].cpu().numpy().reshape(RASTER, RASTER)
            f = RASTER // 128
            s.lab = lab[f // 2::f, f // 2::f]

    def cell(s, x):
        if s.grid:
            return (np.floor(x[:, 1] * s.n).astype(int) % s.n) * s.n + np.floor(x[:, 0] * s.n).astype(int) % s.n
        return s.tree.query(x % 1.0)[1]

    def sample(s, F, x):
        if not s.grid:
            return F[s.cell(x)]
        n = s.n
        u = x[:, 0] * n - 0.5; w = x[:, 1] * n - 0.5
        i0 = np.floor(u).astype(int); j0 = np.floor(w).astype(int)
        fu, fw = (u - i0)[:, None], (w - j0)[:, None]
        G = F.reshape(n, n, -1)
        g = lambda jj, ii: G[jj % n, ii % n]
        return (g(j0, i0) * (1 - fu) * (1 - fw) + g(j0, i0 + 1) * fu * (1 - fw) + g(j0 + 1, i0) * (1 - fu) * fw
                + g(j0 + 1, i0 + 1) * fu * fw)

    def image(s, v):
        return v.reshape(s.n, s.n) if s.grid else v[s.lab]


def cell_velocity(c, J, disc):
    """v_i = sum_{e at i} J_e d_e / (2 c_i), zero in nearly empty cells"""
    E = disc.g['edges'].cpu().numpy(); dv = disc.g['dvec'].cpu().numpy()
    m = np.zeros((disc.V, 2))
    np.add.at(m, E[:, 0], J[:, None] * dv); np.add.at(m, E[:, 1], J[:, None] * dv)
    v = m / (2 * np.maximum(c, 0.02))[:, None]
    v[c < 0.02] = 0.0
    return v


def seed(rng, n, w, cells):
    out = np.zeros((0, 2))
    w = np.clip(w, 0, None)
    wmax = w.max() if w.max() > 0 else 1.0
    while len(out) < n:
        x = rng.random((4 * n, 2))
        out = np.concatenate([out, x[rng.random(4 * n) < w[cells.cell(x)] / wmax]])
    return out[:n]


def _png(fig):
    from PIL import Image
    buf = io.BytesIO()
    fig.savefig(buf, format='png')
    buf.seek(0)
    return Image.open(buf).convert('RGB')


def write_gif(frames, path, fps, colors):
    from PIL import Image
    q = [f.quantize(colors=colors, method=Image.Quantize.MEDIANCUT, dither=Image.Dither.NONE) for f in frames]
    q[0].save(path, save_all=True, append_images=q[1:], duration=int(1000 / fps), loop=0, optimize=True)
    return os.path.getsize(path) / 2 ** 20


def _flag_text(p):
    return f'{100 * p["bad"].mean():.0f} % of cells flagged'


def track_gif(path, title, panels, disc, times, n_part=800, trail=3, max_age=18, substeps=8, fps=5, width_in=8.6, dpi=64):
    """panels: [(label, dict(R, J, bad) or None for a diverged model)]; the first panel is the reference"""
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    rng = np.random.default_rng(0)
    cells = Cells(disc)
    R0 = panels[0][1]['R']
    vmax = float(np.percentile(R0 * cells.dens[None], 99.5))
    x0 = seed(rng, n_part, R0[0] * cells.dens, cells)
    state = [dict(x=x0.copy(), age=rng.integers(0, max_age, n_part), hist=[x0.copy()]) for _ in panels]
    n = len(panels)
    fig, axs = plt.subplots(1, n, figsize=(width_in, width_in / n * 1.32), dpi=dpi)
    fig.subplots_adjust(left=0.005, right=0.995, top=0.74, bottom=0.01, wspace=0.03)
    frames = []
    h = (times[1] - times[0]) / substeps
    for fi in range(len(times)):
        for p, (lab, d) in enumerate(panels):
            ax = axs[p]; ax.clear(); ax.set_facecolor('black')
            st = state[p]
            c, J = d['R'][fi], d['J'][fi]
            if not (np.isfinite(c).all() and np.isfinite(J).all()):
                ax.text(0.5, 0.5, 'diverged', color='w', ha='center', va='center', transform=ax.transAxes)
            else:
                if fi > 0:
                    v = cell_velocity(c, J, disc)
                    x = st['x']
                    for _ in range(substeps):                       # midpoint rule, periodic
                        k1 = cells.sample(v, x); k2 = cells.sample(v, (x + 0.5 * h * k1) % 1.0)
                        x = (x + h * k2) % 1.0
                    st['age'] = st['age'] + 1
                    re = (st['age'] > max_age) | (c[cells.cell(x)] < 0.02)
                    if re.any():
                        x[re] = seed(rng, int(re.sum()), c * cells.dens, cells)
                        st['age'][re] = 0
                    st['x'] = x
                    st['hist'] = (st['hist'] + [x.copy()])[-trail:]
                ax.imshow(cells.image(c * cells.dens), cmap='gray', vmin=0, vmax=1.6 * vmax, origin='lower',
                          extent=(0, 1, 0, 1), interpolation='nearest')
                bad = d['bad'][fi] if 'bad' in d else np.zeros(disc.V, bool)
                flag = bad[cells.cell(st['x'])]
                col = np.where(flag[:, None], np.array(matplotlib.colors.to_rgba(MAGENTA))[None], np.array(matplotlib.colors.to_rgba(TRACER))[None])
                for k, xh in enumerate(st['hist'][:-1]):
                    ck = col.copy(); ck[:, 3] = 0.15 + 0.5 * (k + 1) / len(st['hist'])
                    ax.scatter(xh[:, 0], xh[:, 1], s=1.2, c=ck, linewidths=0)
                ax.scatter(st['x'][:, 0], st['x'][:, 1], s=5.0, c=col, linewidths=0)
                if 'bad' in d:
                    ax.text(0.02, 0.02, _flag_text(dict(bad=bad)), transform=ax.transAxes, color='w', fontsize=8,
                            va='bottom', bbox=dict(fc='black', lw=0, pad=1.5))
            ax.set_xlim(0, 1); ax.set_ylim(0, 1); ax.set_xticks([]); ax.set_yticks([])
            ax.set_title(lab, fontsize=10)
        fig.suptitle(f'{title}   t = {times[fi]:4.1f}\ntracers follow v = J / rho; magenta: sigma < 0 or flux against the '
                     f'physical force', fontsize=9)
        frames.append(_png(fig))
    plt.close(fig)
    return write_gif(frames, path, fps, colors=16)


def entropy_gif(path, title, panels, disc, times, fps=5, width_in=8.6, dpi=64):
    """panels: [(label, dict(R, sigma, viol (models only)))]"""
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    import matplotlib.colors as mcolors
    from matplotlib.collections import LineCollection
    cells = Cells(disc)
    s0 = panels[0][1]['sigma']
    hi = float(np.percentile(s0[s0 > 0], 99)) if (s0 > 0).any() else 1.0
    norm = mcolors.LogNorm(vmin=hi * 1e-3, vmax=hi)
    cm = plt.get_cmap('cividis').copy(); cm.set_bad('black'); cm.set_under('black')
    neg = mcolors.ListedColormap([MAGENTA])
    E = disc.g['edges'].cpu().numpy(); P = disc.g['pts'].cpu().numpy(); dv = disc.g['dvec'].cpu().numpy()
    n = len(panels)
    fig, axs = plt.subplots(1, n, figsize=(width_in, width_in / n * 1.32), dpi=dpi)
    fig.subplots_adjust(left=0.005, right=0.995, top=0.74, bottom=0.01, wspace=0.03)
    frames = []
    for fi in range(len(times)):
        for p, (lab, d) in enumerate(panels):
            ax = axs[p]; ax.clear()
            sg = d['sigma'][fi]
            if not np.isfinite(sg).all():
                ax.text(0.5, 0.5, 'diverged', ha='center', va='center', transform=ax.transAxes)
            else:
                img = cells.image(sg)
                ax.imshow(np.where(img > 0, img, np.nan), cmap=cm, norm=norm, origin='lower', extent=(0, 1, 0, 1),
                          interpolation='nearest')
                if 'viol' in d:
                    ax.imshow(np.where(img < 0, 1.0, np.nan), cmap=neg, vmin=0, vmax=1, origin='lower',
                              extent=(0, 1, 0, 1), interpolation='nearest')
                    vi = np.nonzero(d['viol'][fi])[0]
                    if len(vi):
                        ax.add_collection(LineCollection([[P[E[e, 0]], P[E[e, 0]] + dv[e]] for e in vi], colors=MAGENTA,
                                                         linewidths=0.8))
                    ax.text(0.02, 0.02, f'{100 * (sg < 0).mean():.0f} % of cells with sigma < 0\n'
                            f'{100 * d["viol"][fi].mean():.0f} % of edges against the force', transform=ax.transAxes,
                            color='w', fontsize=8, va='bottom', bbox=dict(fc='black', lw=0, pad=1.5))
            ax.set_xlim(0, 1); ax.set_ylim(0, 1); ax.set_xticks([]); ax.set_yticks([])
            ax.set_title(lab, fontsize=10)
        fig.suptitle(f'{title}   t = {times[fi]:4.1f}\nlocal entropy production (log scale); magenta: sigma < 0 and '
                     f'edges with flux against the physical force', fontsize=9)
        frames.append(_png(fig))
    plt.close(fig)
    return write_gif(frames, path, fps, colors=24)


# ------------------------------------------------------------------------------------------ the animations
def make_gallery(device='cuda', data_dir=None, weights=None, variants=('porous_media', 'porous_media_crowding'), j=0,
                 assets=None, gallery=None, readme_picks=None):
    """render the track and entropy animations of every scene; the ones in readme_picks go to assets/, the others to
    results/porous_media/gallery/.  Returns {path: size in MB}."""
    check_gpu(device)
    assets = assets or os.path.join(ROOT, 'assets')
    gallery = gallery or os.path.join(RESULTS, 'porous_media', 'gallery')
    picks = set(README_PICKS if readme_picks is None else readme_picks)
    os.makedirs(assets, exist_ok=True); os.makedirs(gallery, exist_ok=True)
    frames = list(range(0, T_TEST + 1, STRIDE))
    times = np.array(frames) * DT
    sizes = {}
    for variant in variants:
        for key in SCENES[variant]:
            data = load(variant, device, sets=['train', key], data_dir=data_dir)
            add_norm(data['meta'], data)
            sd = data[key]
            disc = data['discs'][sd['disc']]
            R, J = reference_scene(variant, key, j, disc, data['meta'], device, frames)
            dr = {k: v[j:j + 1] for k, v in sd['drive'].items()}
            ref = dict(R=R, J=J, sigma=reference_sigma(R, J, dr, disc))
            panels = [('Reference', ref)]
            for name in MODELS:
                panels.append((LABELS[name], model_scene(variant, name, data, key, j, device, frames, weights)))
            title = f'{TITLE[variant]}, unseen medium, {SCENE_TITLE[key]}'
            for kind, fn in (('track', track_gif), ('entropy', entropy_gif)):
                stem = f'{variant}_{kind}_{key}'
                path = os.path.join(assets if stem in picks else gallery, stem + '.gif')
                sizes[path] = fn(path, title, panels, disc, times)
                print(path, f'{sizes[path]:.2f} MB', flush=True)
    return sizes


# ------------------------------------------------------------------------------------------ the constitutive law
@torch.no_grad()
def medium_and_law(path=None, data_dir=None, j=0):
    """the pore map, the true mobility and the learned mobility of the Neural Chart on an unseen medium, on the 32 x 32
    grid and on the point cloud (no retraining).  Cell values are geometric means over the edges of a cell; the
    learned mobility is taken at the edge-mean density rhobar = 1."""
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    import matplotlib.colors as mcolors
    variant = 'porous_media'
    path = path or os.path.join(ROOT, 'assets', 'medium_and_law.png')
    data = load(variant, 'cpu', sets=['train', 'training_conditions', 'point_cloud'], data_dir=data_dir)
    meta = add_norm(data['meta'], data)
    oracle_drive(data)
    base = build('chart', meta, data['discs']['g32'])
    base.load_state_dict(torch.load(os.path.join(CHECKPOINTS, variant, 'chart.pt'), map_location='cpu'))
    base.eval()
    med = int(data['training_conditions']['med'][j])
    pore = meta['pools']['test']['pore'][med].numpy()
    Mpix = meta['Mpix']['test'][med].double().mean(0).numpy()            # mean of the x and y pixel mobilities
    norm = mcolors.LogNorm(vmin=1e-3, vmax=10)
    rows = []
    for key, lab in (('training_conditions', '32×32 grid'), ('point_cloud', 'point cloud')):
        sd = data[key]; disc = data['discs'][sd['disc']]
        m = base if sd['disc'] == 'g32' else rebind('chart', base, meta, disc, 'cpu')
        feats = m.featurizer({k: v[j:j + 1] for k, v in sd['drive'].items()})
        kap = m.kinetics.kappa(feats)[0].double() + m.kinetics.state_branch(torch.zeros(1, 1)).double()[0, 0]
        mt = sd['drive']['mtrue'][j].double()
        E = disc.g['edges']
        def per_cell(logk):
            s = torch.zeros(disc.V, dtype=torch.float64).index_add_(0, E[:, 0], logk).index_add_(0, E[:, 1], logk)
            nn = torch.zeros(disc.V, dtype=torch.float64).index_add_(0, E[:, 0], torch.ones_like(logk)).index_add_(0, E[:, 1], torch.ones_like(logk))
            return torch.exp(s / nn).numpy()
        lt, lh = torch.log(mt), kap
        r = float(np.corrcoef(lt.numpy(), lh.numpy())[0, 1])
        cells = Cells(disc)
        rows.append((lab, cells.image(per_cell(lt)), cells.image(per_cell(lh)), r, disc))
    fig, axs = plt.subplots(2, 4, figsize=(13.2, 7.0), dpi=100)
    fig.subplots_adjust(left=0.01, right=0.9, top=0.88, bottom=0.01, wspace=0.06, hspace=0.18)
    ext = (0, 1, 0, 1)
    for i, (lab, true_img, learned_img, r, disc) in enumerate(rows):
        ax = axs[i, 0]
        ax.imshow(pore, cmap='gray', origin='lower', extent=ext, interpolation='nearest')
        if disc.g['kind'] != 'grid':
            P = disc.g['pts'].numpy()
            ax.scatter(P[:, 0], P[:, 1], s=1.2, c=TRACER, linewidths=0)
            ax.set_title('pore map and the 1024 points', fontsize=10)
            lab_img = Cells(disc).lab
            edge = (lab_img != np.roll(lab_img, 1, 0)) | (lab_img != np.roll(lab_img, 1, 1))
            axs[i, 1].imshow(np.where(edge, 0.0, 1.0), cmap='gray', vmin=0, vmax=1, origin='lower', extent=ext,
                             interpolation='nearest')
            axs[i, 1].set_title('Voronoi cells of the point cloud', fontsize=10)
        else:
            ax.set_title('pore map (white: pore)', fontsize=10)
            axs[i, 1].imshow(Mpix, cmap='viridis', norm=norm, origin='lower', extent=ext, interpolation='nearest')
            axs[i, 1].set_title('true mobility at the pore scale', fontsize=10)
        axs[i, 2].imshow(true_img, cmap='viridis', norm=norm, origin='lower', extent=ext, interpolation='nearest')
        axs[i, 2].set_title(f'true mobility, {lab}', fontsize=10)
        im = axs[i, 3].imshow(learned_img, cmap='viridis', norm=norm, origin='lower', extent=ext, interpolation='nearest')
        axs[i, 3].set_title(f'learned mobility, {lab}\nPearson r of log k over edges: {r:.3f}', fontsize=10)
        for ax in axs[i]:
            ax.set_xticks([]); ax.set_yticks([])
    cax = fig.add_axes([0.915, 0.15, 0.015, 0.65])
    fig.colorbar(im, cax=cax, label='mobility (geometric mean over the edges of a cell)')
    fig.suptitle('Learned and true mobility of an unseen medium.  The Neural Chart is trained on the 32×32 grid and '
                 'used on the point cloud with the same parameters.', fontsize=11)
    fig.savefig(path)
    plt.close(fig)
    return path, os.path.getsize(path) / 2 ** 20
