"""Command line of the reaction-network benchmarks (python -m benchmarks.<name>.run <command>).

    generate   reference data (all sets, or --sets ...)
    check      the oracle against the reference model, and the closed-form step from the given potentials
    calibrate  the rtol of the fast integrator (the oracle's 10T error per rtol)
    train      train one model and evaluate it (--model chart | chart-... | oracle | crnn_mass_action | crnn_features |
               flux_gnn | ude | fno | bsms | generic_fno | persistence)
    evaluate   evaluate weights (--weights FILE; default runs/<benchmark>/<model>.pt, else the shipped
               checkpoints/<benchmark>/<model>.pt) on every test set; writes runs/<benchmark>/evaluated/<model>.json
    tables     the tables of results/<benchmark>/ from results/<benchmark>/runs/ (--runs DIR: from fresh runs)
    gifs       the README animation from the weights in --weights DIR (default runs/<benchmark>/)

Data: --data DIR, default $NEURALCHART_DATA/<benchmark> or data/<benchmark>.  Outputs of train and evaluate:
--out DIR, default runs/<benchmark>/.
"""
import argparse
import json
import os

import torch

from neuralchart import fast
from ..common import CHECKPOINTS, check_gpu
from . import config as C


def evaluate_weights(name, weights=None, device='cuda', out=None):
    """the evaluation of run_train for stored weights (no training); writes <out>/<name>.json when out is given"""
    from .data import train_norm
    from .evaluate import evaluate
    from .models import FLOAT_MODELS, build, accelerate, set_max_sub
    check_gpu(device)
    norm = train_norm()
    model = build(name, C.N_TRAIN, norm).to(device)
    if name in FLOAT_MODELS:
        model = model.float()
    trained = name != 'persistence' and getattr(getattr(model, 'config', None), 'train', True)
    if trained:
        w = weights or os.path.join(C.OUT, f'{name}.pt')
        if not os.path.exists(w) and weights is None:
            w = os.path.join(CHECKPOINTS, C.SYSTEM, f'{name}.pt')
        model.load_state_dict(torch.load(w, map_location=device))
    if hasattr(model, 'eval'):
        model.eval()
    set_max_sub(model, C.MAXSUB_EVAL)
    accelerate(model, name)
    ev, snap = evaluate(name, model, norm, device, print)
    res = dict(model=name, benchmark=C.SYSTEM, sets=ev)
    if out:
        os.makedirs(out, exist_ok=True)
        json.dump(res, open(os.path.join(out, f'{name}.json'), 'w'), default=float)
    return res, snap, model


def main(system, argv=None):
    p = argparse.ArgumentParser(prog=f'python -m benchmarks.{system}.run')
    p.add_argument('cmd', choices=['generate', 'check', 'calibrate', 'train', 'evaluate', 'tables', 'gifs'])
    p.add_argument('--model', default='chart')
    p.add_argument('--seed', type=int, default=0)
    p.add_argument('--iters', type=int, default=C.ITERS)
    p.add_argument('--device', default='cuda')
    p.add_argument('--lr', type=float, default=C.LR)
    p.add_argument('--sets', nargs='*', default=None)
    p.add_argument('--data', default=None, help='data directory (default data/<benchmark>)')
    p.add_argument('--out', default=None, help='outputs of train and evaluate (default runs/<benchmark>)')
    p.add_argument('--weights', default=None, help='evaluate: the weights file; gifs: the weights directory')
    p.add_argument('--runs', default=None, help='tables: directory of run files (default results/<benchmark>/runs)')
    p.add_argument('--resume', action='store_true', help='train: continue from the last checkpoint of the run')
    fast.add_args(p, C.RTOL)
    a = p.parse_args(argv)
    C.configure(system, a.data, a.out)
    a.fast = True                                     # every benchmark run uses the fast integrator
    C.FAST = fast.kwargs_from_args(a)
    if a.cmd == 'generate':
        from .data import generate
        generate(a.sets)
    elif a.cmd == 'check':
        from .train import check
        check()
    elif a.cmd == 'calibrate':
        from .train import calibrate
        calibrate(a.device)
    elif a.cmd == 'train':
        from .train import run_train
        run_train(a.model, a.seed, a.iters, a.device, a.lr, resume=a.resume)
    elif a.cmd == 'evaluate':
        evaluate_weights(a.model, a.weights, a.device, os.path.join(C.OUT, 'evaluated'))
    elif a.cmd == 'tables':
        from .tables import make_tables
        make_tables(a.runs)
    else:
        from .gifs import make_gif
        make_gif(system, a.device, a.weights)
