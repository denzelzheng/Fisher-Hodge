"""Brusselator benchmark: one entry point.

    python -m benchmarks.brusselator.run generate                     # reference data (about 20 min on 12 CPU cores)
    python -m benchmarks.brusselator.run check                        # the oracle against the reference
    python -m benchmarks.brusselator.run train --model chart --fast   # runs/brusselator/chart.{json,pt,log}
    python -m benchmarks.brusselator.run evaluate --model chart --fast    # shipped checkpoint -> runs/brusselator/evaluated
    python -m benchmarks.brusselator.run tables                       # results/brusselator/*.md, *.csv
    python -m benchmarks.brusselator.run gifs                         # assets/brusselator_*.gif
    python -m benchmarks.brusselator.run figures                      # assets/brusselator_*.png

Data: --data, default $NEURALCHART_DATA/brusselator if set, else data/brusselator.  The device defaults to cuda.

Models: chart, chart-learned-mu0, chart-zero-init, chart-reversible, oracle (neuralchart.chart.CONFIGS['brusselator'])
and the baselines crnn_mass_action, crnn_features, flux_gnn, ude, fno, bsms, generic_fno, persistence.
--fast uses neuralchart.fast with error control (rtol 5e-5) in float32, the setting of every reported number; the
UDE closure then integrates with error control only.  Optional speed flags: --compact, --compile_rates, --fuse_net.
"""
import argparse
import os

import torch

from neuralchart import fast
from .config import RTOL, ITERS, LR, N_TRAIN, CHECKPOINTS, RESULTS, RUNS, DATA
from .test_sets import GEN, TEST_SETS


def checkpoint_of(name):
    """the shipped checkpoint of a Neural Chart variant, otherwise the checkpoint written by `train`"""
    f = os.path.join(CHECKPOINTS, name + '.pt')
    return f if os.path.exists(f) else os.path.join(RUNS, name + '.pt')


def load_trained(name, norm, device, checkpoint=None):
    """a model with its trained parameters (the oracle and persistence need none)"""
    from .models import build
    model = build(name, N_TRAIN, norm).to(device)
    path = checkpoint or checkpoint_of(name)
    if os.path.exists(path):
        model.load_state_dict(torch.load(path, map_location=device))
    else:
        assert name in ('oracle', 'persistence'), f'no checkpoint for {name}: train it or pass --checkpoint'
    return model


def cmd_evaluate(a, fkw):
    from benchmarks.common import check_gpu
    from .data import train_norm
    from .training import evaluate_saved, write_result, default_fast
    check_gpu(a.device)
    dkw = {} if a.data is None else dict(data=a.data)
    norm = train_norm(**dkw)
    model = load_trained(a.model, norm, a.device, a.checkpoint)
    out = a.out or os.path.join(RUNS, 'evaluated')
    log = print
    res = evaluate_saved(a.model, model, norm, a.device, log, default_fast(a.model, fkw), **dkw)
    write_result(out, a.model, res)
    return res


@torch.no_grad()
def cmd_check(device='cpu'):
    """the oracle (the reference in the Neural Chart class) reproduces the true one-way rates"""
    from .data import load_set, true_rates, hg_of, train_norm, describe
    from .models import build
    for s in ['train'] + TEST_SETS:
        print(f'{s}: {describe(torch.load(os.path.join(DATA, s + ".pt")))}', flush=True)
    m = build('oracle', N_TRAIN, train_norm()).to(device)
    sd = load_set('training', device)
    dr = {k: v[:2] for k, v in sd['drive'].items()}
    a, b = m.rates(sd['c'][:2, 5], dr)
    at, bt = true_rates(hg_of(N_TRAIN), dr['logab'].cpu(), sd['c'][:2, 5].cpu())
    print('oracle one-way rates against the reference: max relative error',
          float(((a.cpu() - at).abs() / at).max()), float(((b.cpu() - bt).abs() / bt).max()))


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument('cmd', choices=['generate', 'check', 'train', 'evaluate', 'tables', 'gifs', 'figures'])
    p.add_argument('--model', default='chart')
    p.add_argument('--seed', type=int, default=0)
    p.add_argument('--iters', type=int, default=ITERS)
    p.add_argument('--lr', type=float, default=LR)
    p.add_argument('--device', default='cuda')
    p.add_argument('--sets', nargs='*', default=list(GEN), help='generate: the sets to compute')
    p.add_argument('--procs', type=int, default=12, help='generate: worker processes')
    p.add_argument('--data', default=None, help='data directory (default $NEURALCHART_DATA/brusselator)')
    p.add_argument('--out', default=None, help='train: output directory (default runs/brusselator); evaluate: default runs/brusselator/evaluated')
    p.add_argument('--runs', default=None, help='tables: directory of the result files (default results/brusselator/runs)')
    p.add_argument('--weights', default=RUNS, help='gifs, figures: directory of the baseline weights <model>.pt (default runs/brusselator)')
    p.add_argument('--checkpoint', default=None, help='evaluate: the checkpoint (default checkpoints/brusselator/<model>.pt, '
                   'else runs/brusselator/<model>.pt)')
    p.add_argument('--checkpoint_dir', default=None, help='train: where the checkpoint goes (default: the output directory)')
    fast.add_args(p, RTOL)
    a = p.parse_args(argv)
    fkw = fast.kwargs_from_args(a)
    if a.cmd == 'generate':
        from .data import generate
        generate(a.sets, a.procs, **({} if a.data is None else dict(data=a.data)))
    elif a.cmd == 'check':
        cmd_check('cpu')
    elif a.cmd == 'train':
        from .training import run_train
        run_train(a.model, a.seed, a.iters, a.device, a.lr, fkw, data=a.data, out=a.out, ckpt_dir=a.checkpoint_dir)
    elif a.cmd == 'evaluate':
        cmd_evaluate(a, fkw)
    elif a.cmd == 'tables':
        from .tables import make_tables
        make_tables(runs=a.runs)
    elif a.cmd == 'gifs':
        from .gifs import make_gifs
        make_gifs(a.device, data=a.data, weights=a.weights)
    elif a.cmd == 'figures':
        from .figures import make_figures
        make_figures(a.device, data=a.data, weights=a.weights)


if __name__ == '__main__':
    main()
