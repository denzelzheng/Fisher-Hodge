"""Comparison animations of the porous-media benchmark (assets/porous_media*.gif).

Two scenes per variant, one test trajectory each:
  drive ×8      an unseen medium under eight times the training drive: Reference | Neural Chart | Flux-GNN | the black
                box with the lowest drive ×8 error (HelmFluid without crowding, GENERIC-FNO with crowding)
  point cloud   the training conditions on a random point cloud of 1024 cells: Reference | Neural Chart | Flux-GNN |
                UDE closure (FNO-type models cannot run on a point cloud)
The Neural Chart weights come from checkpoints/.  Baseline weights are not shipped.  HelmFluid needs
NEURALCHART_HELMFLUID (see THIRD_PARTY.md).  Train them with run.py train,
which writes runs/<variant>/<model>.pt (the default of --weights).
"""
import os

import numpy as np
import torch

from neuralchart import fast
from neuralchart.fit import rollout
from benchmarks.common import CHECKPOINTS, ROOT, LABELS, check_gpu
from scripts.gif_tools import comparison_gif
from .data import N, DT, load, add_norm
from .models import build, rebind
from .run import RUN_SETTINGS

# per variant: (test set, models after the reference, subtitle).  The last drive ×8 panel is the black box with the
# lowest drive ×8 error in the main table: HelmFluid without crowding, GENERIC-FNO with crowding.
SCENES = {
    'porous_media': [('drive_x8', ['chart', 'flux_gnn', 'helmfluid'], 'unseen medium, drive ×8'),
                     ('point_cloud', ['chart', 'flux_gnn', 'ude'], 'unseen medium, point cloud of 1024 cells')],
    'porous_media_crowding': [('drive_x8', ['chart', 'flux_gnn', 'generic_fno'], 'unseen medium, drive ×8'),
                              ('point_cloud', ['chart', 'flux_gnn', 'ude'], 'unseen medium, point cloud of 1024 cells')],
}
TITLE = dict(porous_media='Porous media', porous_media_crowding='Porous media with crowding')


def _weights(variant, model, weights):
    p = os.path.join(CHECKPOINTS, variant, model + '.pt')
    if os.path.exists(p):
        return p
    return os.path.join(weights or os.path.join(ROOT, 'runs', variant), model + '.pt')


@torch.no_grad()
def scene_rollout(variant, model_name, data, s, j, device, weights=None):
    """the density of trajectory j of set s, (T+1, V), for one model"""
    meta = data['meta']
    sd = data[s]
    disc = data['discs'][sd['disc']]
    dr = {k: v[j:j + 1] for k, v in sd['drive'].items()}
    c0 = sd['c'][j:j + 1, 0]
    T = sd['c'].shape[1] - 1
    if model_name == 'helmfluid':
        from .train import helmfluid_model, helmfluid_rollout
        m = helmfluid_model(device)
        m.load_state_dict(torch.load(_weights(variant, model_name, weights), map_location=device)); m.eval()
        R = helmfluid_rollout(m, sd['c'][j:j + 1], dr, meta['norm'], device)
        return R[0].double().numpy() * disc.g['dens'].cpu().numpy()[None]
    if model_name == 'generic_fno':
        from neuralchart.baselines.generic_fno import GenericFNO, rollout as gfno_rollout
        from .train import grid_cond
        from .models import NC
        m = GenericFNO(n_fields=1, n_cond=NC).to(device)
        m.load_state_dict(torch.load(_weights(variant, model_name, weights), map_location=device)); m.eval()
        n = int(round(c0.shape[-1] ** 0.5))
        R = gfno_rollout(m, c0.reshape(1, 1, n, n).float(), T, grid_cond(dr, meta['norm']))
        return R.reshape(T + 1, -1).double().numpy() * disc.g['dens'].cpu().numpy()[None]
    g32 = data['discs']['g32']
    m = build(model_name, meta, g32).to(device)
    m.load_state_dict(torch.load(_weights(variant, model_name, weights), map_location=device))
    m.eval()
    if sd['disc'] != 'g32':
        m = rebind(model_name, m, meta, disc, device)
    if model_name in RUN_SETTINGS:
        fast.accelerate(m, **RUN_SETTINGS[model_name])
    R, _, _ = rollout(m, dr, c0, T, DT)
    return (R[0].double() * disc.g['dens'].to(R.device).double()[None]).cpu().numpy()


def cell_map(disc, n_raster=128):
    """(ny, nx) cell index of every image pixel (None on grids, where a pixel is a cell)"""
    g = disc.g
    if g['kind'] == 'grid':
        return None
    from .data import RASTER
    f = RASTER // n_raster
    return g['lab'].cpu().numpy().reshape(RASTER, RASTER)[f // 2::f, f // 2::f]


def to_image(D, disc, n_raster=128):
    """(T, V) cell values -> (T, ny, nx) images: a reshape on grids, the Voronoi cells rasterised on the point cloud"""
    g = disc.g
    if g['kind'] == 'grid':
        return D.reshape(D.shape[0], g['n'], g['n'])
    from .data import RASTER
    lab = g['lab'].cpu().numpy().reshape(RASTER, RASTER)
    f = RASTER // n_raster
    lab = lab[f // 2::f, f // 2::f]
    return D[:, lab]


def make_gifs(device='cuda', data_dir=None, weights=None, out=None, j=0, variants=('porous_media', 'porous_media_crowding'),
              scenes=None):
    """data_dir and weights are the directories of one variant (pass a single variant with them)"""
    assert len(variants) == 1 or (data_dir is None and weights is None), 'data_dir / weights name the directory of one variant'
    check_gpu(device)
    out = out or os.path.join(ROOT, 'assets')
    os.makedirs(out, exist_ok=True)
    sizes = {}
    for variant in variants:
        data = load(variant, device, sets=['train'] + [x[0] for x in SCENES[variant] if scenes is None or x[0] in scenes], data_dir=data_dir)
        add_norm(data['meta'], data)
        for s, models, sub in [x for x in SCENES[variant] if scenes is None or x[0] in scenes]:
            sd = data[s]
            disc = data['discs'][sd['disc']]
            ref = (sd['c'][j].double().cpu() * disc.g['dens'].cpu().double()[None]).numpy()
            panels = [to_image(ref, disc)]
            for mname in models:
                panels.append(to_image(scene_rollout(variant, mname, data, s, j, device, weights), disc))
            titles = ['Reference'] + [LABELS[m] for m in models]
            times = np.arange(ref.shape[0]) * DT
            path = os.path.join(out, f'{variant}_{s}.gif')
            sizes[path] = comparison_gif(path, panels, titles, suptitle=f'{TITLE[variant]}, {sub}', times=times, step=2,
                                         fps=12, time_label=lambda t, times=times: f't = {times[t]:.1f}',
                                         cells=cell_map(disc))
            print(path, f'{sizes[path]:.2f} MB', flush=True)
    return sizes
