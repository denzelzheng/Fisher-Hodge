"""Comparison animations of the porous-media benchmark (assets/porous_media*.gif).

Two scenes per variant, one test trajectory each:
  drive ×8      an unseen medium under eight times the training drive: Reference | Neural Chart | Flux-GNN | the black
                box with the lowest drive ×8 error (HelmFluid without crowding, GENERIC-FNO with crowding)
  point cloud   the training conditions on a random point cloud of 1024 cells: Reference | Neural Chart | Flux-GNN |
                UDE closure (FNO-type models cannot run on a point cloud)
Weights: <weights>/<model>.pt (default runs/<variant>/, the output of neuralchart.run); a Neural Chart variant not found there comes from
checkpoints/.  Baseline weights are not shipped.  HelmFluid needs NEURALCHART_HELMFLUID (see THIRD_PARTY.md).
"""
import os

import numpy as np
import torch

from neuralchart.fit import rollout
from benchmarks.common import ROOT, LABELS, check_gpu
from scripts.gif_tools import comparison_gif
from .data import DT, load, add_norm
from .models import load_trained

# per variant: (test set, models after the reference, subtitle).  The last drive ×8 panel is the black box with the
# lowest drive ×8 error in the main table: HelmFluid without crowding, GENERIC-FNO with crowding.
SCENES = {
    'porous_media': [('drive_x8', ['chart', 'flux_gnn', 'helmfluid'], 'unseen medium, drive ×8'),
                     ('point_cloud', ['chart', 'flux_gnn', 'ude'], 'unseen medium, point cloud of 1024 cells')],
    'porous_media_crowding': [('drive_x8', ['chart', 'flux_gnn', 'generic_fno'], 'unseen medium, drive ×8'),
                              ('point_cloud', ['chart', 'flux_gnn', 'ude'], 'unseen medium, point cloud of 1024 cells')],
}
TITLE = dict(porous_media='Porous media', porous_media_crowding='Porous media with crowding')


@torch.no_grad()
def scene_rollout(variant, model_name, data, s, j, device, weights=None):
    """the density of trajectory j of set s, (T+1, V), for one model"""
    sd = data[s]
    disc = data['discs'][sd['disc']]
    dr = {k: v[j:j + 1] for k, v in sd['drive'].items()}
    m = load_trained(variant, model_name, data['meta'], data['discs']['g32'], disc, device, weights)
    R, _, _ = rollout(m, dr, sd['c'][j:j + 1, 0], sd['c'].shape[1] - 1, DT)
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
            sizes[path] = comparison_gif(path, panels, titles, suptitle=f'{TITLE[variant]}, {sub}', times=times, step=5,
                                         fps=6, dpi=72, time_label=lambda t, times=times: f't = {times[t]:.1f}',
                                         cells=cell_map(disc))
            print(path, f'{sizes[path]:.2f} MB', flush=True)
    return sizes
