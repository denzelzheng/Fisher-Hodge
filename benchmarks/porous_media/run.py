"""Porous-media benchmark: driven transport through porous media, with and without crowding.

    python -m benchmarks.porous_media.run generate --variant porous_media
    python -m benchmarks.porous_media.run train --variant porous_media --model chart --settings
    python -m benchmarks.porous_media.run evaluate --variant porous_media --model chart --settings
    python -m benchmarks.porous_media.run law --variant porous_media --model chart       # learned mobility vs truth
    python -m benchmarks.porous_media.run tables
    python -m benchmarks.porous_media.run gifs

Data: --data, default $NEURALCHART_DATA/<variant> or data/<variant>.  train writes runs/<variant>/<model>.json, .pt
and .log; evaluate and law write runs/<variant>/evaluated/.  tables reads results/<variant>/runs/ (or --runs) and
writes results/<variant>/.  gifs takes the Neural Chart from checkpoints/ and the baselines from --weights (default
runs/<variant>/).  tables and gifs run on both variants unless --variant is given.

--variant porous_media_crowding uses the crowded mobility M / (1 + rhobar / 2).  Model keys: chart, chart-no-state,
chart-const-k, chart-learned-drive, oracle, oracle-no-crowding, ude, flux_gnn, bsms, fno, generic_fno, helmfluid,
persistence.  The settings of the reported runs are in RUN_SETTINGS.
"""
import argparse
import os

from neuralchart import fast
from benchmarks.common import CHECKPOINTS, ROOT

# the integrator settings of the reported runs (neuralchart.fast); models without an entry use their own update
RUN_SETTINGS = {
    **{m: dict(control='sample_err', rtol=0.3, precision='fp32') for m in
       ('chart', 'chart-no-state', 'chart-const-k', 'chart-learned-drive', 'oracle', 'oracle-no-crowding', 'flux_gnn')},
    'ude': dict(control='err_only', rtol=1e-3, precision='fp32'),
}


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument('cmd', choices=['generate', 'train', 'evaluate', 'law', 'tables', 'gifs'])
    p.add_argument('--variant', default=None, choices=['porous_media', 'porous_media_crowding'],
                   help='default: porous_media (tables and gifs: both variants)')
    p.add_argument('--model', default='chart')
    p.add_argument('--models', default=None, help='tables: comma-separated model keys (default: the reported list)')
    p.add_argument('--seed', type=int, default=0)
    p.add_argument('--iters', type=int, default=4000)
    p.add_argument('--device', default='cuda')
    p.add_argument('--lr', type=float, default=1e-3)
    p.add_argument('--kmax', type=int, default=20)
    p.add_argument('--data', default=None, help='data directory of the variant (default $NEURALCHART_DATA/<variant> or data/<variant>)')
    p.add_argument('--out', default=os.path.join(ROOT, 'runs'), help='output root of train / evaluate / law (results go to <out>/<variant>/)')
    p.add_argument('--runs', default=None, help='tables: a directory of <model>.json files of one variant (default results/<variant>/runs/)')
    p.add_argument('--checkpoint', default=None, help='evaluate: a state dict (default checkpoints/<variant>/<model>.pt)')
    p.add_argument('--noise', type=float, default=None, help='BSMS-GNN input noise (default: tuned on held-out media)')
    p.add_argument('--settings', action='store_true', help='use the integrator settings of the reported runs')
    p.add_argument('--scene', default=None, choices=['drive_x8', 'point_cloud'], help='gifs: one scene only')
    p.add_argument('--weights', default=None, help='gifs: directory with the baseline weights <model>.pt (default runs/<variant>/)')
    fast.add_args(p, 0.3)
    a = p.parse_args(argv)
    both = ('porous_media', 'porous_media_crowding')
    variants = (a.variant,) if a.variant else both
    a.variant = a.variant or 'porous_media'
    fast_kw = fast.kwargs_from_args(a)
    if a.settings and a.model in RUN_SETTINGS:
        fast_kw = dict(RUN_SETTINGS[a.model])
    if a.cmd == 'generate':
        from benchmarks.common import check_gpu
        from .data import generate
        check_gpu(a.device)
        generate(a.variant, a.device, a.data)
    elif a.cmd == 'train':
        from .train import run_train, run_helmfluid
        if a.model == 'helmfluid':
            run_helmfluid(a.variant, a.seed, a.iters, a.out, 100, a.data)
        else:
            run_train(a.variant, a.model, a.seed, a.iters, a.device, a.out, lr=a.lr, data_dir=a.data, kmax=a.kmax,
                      fast_kw=fast_kw, noise=a.noise)
    elif a.cmd == 'evaluate':
        from .train import run_evaluate
        ck = a.checkpoint or os.path.join(CHECKPOINTS, a.variant, a.model + '.pt')
        run_evaluate(a.variant, a.model, ck if os.path.exists(ck) else None, a.device, a.out, a.data, fast_kw)
    elif a.cmd == 'law':
        import json
        from .evaluate import law_metrics
        ck = a.checkpoint or os.path.join(CHECKPOINTS, a.variant, a.model + '.pt')
        res = law_metrics(a.variant, a.model, ck, a.data)
        odir = os.path.join(a.out, a.variant, 'evaluated'); os.makedirs(odir, exist_ok=True)
        json.dump(res, open(os.path.join(odir, a.model + '_law.json'), 'w'), indent=1)
        print(json.dumps(res, indent=1))
    elif a.cmd == 'tables':
        from .tables import write_tables
        assert a.runs is None or len(variants) == 1, '--runs names the runs of one variant: give --variant'
        for v in variants:
            write_tables(v, a.models.split(',') if a.models else None, runs_dir=a.runs)
    elif a.cmd == 'gifs':
        from .gifs import make_gifs
        make_gifs(a.device, a.data, a.weights, variants=variants, scenes=[a.scene] if a.scene else None)


if __name__ == '__main__':
    main()
