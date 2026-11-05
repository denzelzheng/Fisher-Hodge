"""One dispatch for every benchmark, model and ablation.

    python -m neuralchart.run --task schlogl --model chart                 # train and evaluate
    python -m neuralchart.run --task schlogl --model flux_gnn
    python -m neuralchart.run --task schlogl --model chart --cmd evaluate  # evaluate stored weights
    python -m neuralchart.run --task schlogl --cmd generate | calibrate | tables | gifs [--track]

A task is a row of TASKS; its Task class (benchmarks/<family>/task.py) gives the data, the reference model, the baselines and the extra
readouts.  Neural Chart variants come from neuralchart.chart.VARIANTS.  Results: <out>/<task>/<model>.json, .pt and .log (default out: runs).
"""
import argparse
import dataclasses
import importlib
import json
import os
import time

import numpy as np
import torch

from .chart import ITERS, NeuralChart, VARIANTS
from .data import Trajectories
from .diagnostics import LDB_RULE, relerr_traj
from .fit import rollout

# task -> family (the package benchmarks.<family> with its Task class)
TASKS = {
    'porous_media': 'porous_media',
    'porous_media_crowding': 'porous_media',
    'brusselator': 'brusselator',
    'schlogl': 'reaction_networks',
    'schlogl_nonideal': 'reaction_networks',
    'schlogl_nonideal_strong': 'reaction_networks',
    'phosphorylation_cycle': 'reaction_networks',
    'phosphorylation_heterogeneous': 'reaction_networks',
    'phosphorylation_interacting': 'reaction_networks',
    'phosphorylation_saturating': 'reaction_networks',
    'mapk_cascade': 'reaction_networks',
    'compose_reaction_transport': 'composition',
    'compose_cycle_cycle': 'composition',
    'compose_thermo_cross': 'composition',
    'compose_kinetic_cross': 'composition',
}


BASELINE_ITERS = 4000


def get_task(name, data=None):
    return importlib.import_module(f'benchmarks.{TASKS[name]}.task').Task(name, data)


def build_chart(task, name):
    """the Neural Chart variant `name` of a task: the same constructor, the inputs of the task, one override"""
    kw = dict(chemostats=task.chemostats(), descriptors=task.n_descriptors)
    kw.update(VARIANTS[name])
    if kw.get('mu0') == 'known':
        kw['mu0'] = task.known_mu0()
    return NeuralChart(task.hypergraph(), **kw)


def evaluate(task, name, model, device, log):
    """every test set of a task; the core metrics here, the physics and extra readouts of the task in task.metrics"""
    res = {}
    for s in task.sets():
        sd = task.load(s, device)
        m_ = model.on(task.hypergraph(sd)) if name in VARIANTS else task.rebind(name, model, sd, device)
        if m_ is None:                                   # the model cannot run on this discretisation
            res[s] = None
            continue
        if hasattr(m_, 'eval'):
            m_.eval()
        if name in VARIANTS:
            m_.accelerate()
        Y = sd['c']
        t0 = time.time()
        st = getattr(getattr(m_, '_fast', None), 'stepper', None)
        cap0 = st.cap_hits if st is not None else 0
        R, ns, _ = rollout(m_, sd['drive'], Y[:, 0], Y.shape[1] - 1, task.dt)
        R = R.double().cpu()
        Yc = Y.double().cpu()
        B = Y.shape[0]
        ch = task.err_channels(sd)
        e = relerr_traj(R[..., ch], Yc[..., ch]).numpy()
        fin = torch.isfinite(R.reshape(B, -1)).all(1)
        m = dict(err_10T=float(np.median(e[:, -1])), err_10T_mean=float(np.mean(e[:, -1])), err_T=float(np.median(e[:, min(task.t_steps, e.shape[1] - 1)])),
                 err_10T_per_traj=e[:, -1].tolist(), err_curve=np.median(e, 0)[::10].tolist(), n_blown=int((~fin).sum()),
                 n_neg_traj=int((torch.nan_to_num(R, nan=0.0) < 0).reshape(B, -1).any(1).sum()),
                 neg_cell_frac=float((torch.nan_to_num(R, nan=0.0) < 0).double().mean()), min_value=float(np.nanmin(R.numpy())),
                 infer_time_s=time.time() - t0, substeps_per_DT=float(ns), capped_DT=(st.cap_hits - cap0) if st is not None else None)
        task.metrics(name, m_, sd, R, Yc, m)
        log(f'[{s}] T {m["err_T"]:.4g} 10T {m["err_10T"]:.4g} | blown {m["n_blown"]} negative trajectories {m["n_neg_traj"]}'
            + (f' | LDB violations {m["phys"]["ldb_violation_frac"]:.3g} sigma<0 {m["phys"]["sigma_neg_frac"]:.3g}' if 'phys' in m else '')
            + f' | substeps/DT {m["substeps_per_DT"]:.1f} capped {m["capped_DT"]} {m["infer_time_s"]:.0f}s')
        res[s] = m
    return res


def run_train(task, name, iters=None, seed=0, device='cuda', out='runs', resume=False, lr=1e-3):
    if iters is None:
        iters = ITERS if name in VARIANTS or name.startswith('composed') else BASELINE_ITERS
    from benchmarks.common import check_gpu
    check_gpu(device)
    torch.manual_seed(seed); np.random.seed(seed)
    odir = os.path.join(out, task.name)
    os.makedirs(odir, exist_ok=True)
    ckpt = os.path.join(odir, f'{name}.ckpt.pt')
    resume = bool(resume and os.path.exists(ckpt))
    logf = open(os.path.join(odir, f'{name}.log'), 'a' if resume else 'w')

    def log(*a):
        s = ' '.join(str(x) for x in a); print(s, flush=True); logf.write(s + '\n'); logf.flush()
    tr = task.load('train', device)
    if name in VARIANTS:
        model = build_chart(task, name).to(device)
        npar = sum(p.numel() for p in model.parameters() if p.requires_grad)
        log(f'{task.name} {name}: params {npar} iters {iters}')
        info = model.fit(Trajectories(tr['c'], tr['drive'], dt=task.dt), iters=iters, seed=seed, log=log, checkpoint=ckpt, resume=resume, lr=lr)
        info['diagnostics'] = model.diagnostics
        model.save(os.path.join(odir, f'{name}.pt'))
    else:
        model = task.baseline(name, device)
        npar = sum(p.numel() for p in model.parameters() if p.requires_grad) if hasattr(model, 'parameters') else 0
        log(f'{task.name} {name}: params {npar} iters {iters}')
        info = task.fit_baseline(name, model, tr, iters, seed, device, log, ckpt=ckpt, resume=resume, lr=lr)
        if npar and name not in task.not_trained:
            torch.save(model.state_dict(), os.path.join(odir, f'{name}.pt'))
    sets = evaluate(task, name, model, device, log)
    res = dict(model=name, benchmark=task.name, ldb_rule=LDB_RULE, seed=seed, params=npar, iters=iters, lr=lr,
               **{k: v for k, v in info.items() if k != 'loss_curve'}, extra=task.extras(name, model), sets=sets)
    json.dump(res, open(os.path.join(odir, f'{name}.json'), 'w'), default=float)
    log('done')
    return res


def run_evaluate(task, name, weights, device, out):
    model = NeuralChart.load(weights, map_location=device).to(device) if name in VARIANTS else task.load_baseline(name, weights, device)
    sets = evaluate(task, name, model, device, print)
    odir = os.path.join(out, task.name, 'evaluated')
    os.makedirs(odir, exist_ok=True)
    json.dump(dict(model=name, benchmark=task.name, sets=sets, extra=task.extras(name, model)), open(os.path.join(odir, f'{name}.json'), 'w'), default=float)
    return sets


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument('--task', required=True, choices=list(TASKS))
    p.add_argument('--model', default='chart')
    p.add_argument('--cmd', default='train', choices=['train', 'evaluate', 'generate', 'calibrate', 'tables', 'gifs', 'figures'])
    p.add_argument('--iters', type=int, default=None, help=f'rollout steps (default: {ITERS} for the Neural Chart and composition, {BASELINE_ITERS} for the baselines)')
    p.add_argument('--seed', type=int, default=0)
    p.add_argument('--device', default='cuda')
    p.add_argument('--lr', type=float, default=1e-3)
    p.add_argument('--out', default='runs')
    p.add_argument('--data', default=None)
    p.add_argument('--weights', default=None, help='evaluate: the weights file; gifs: the directory of the baseline weights')
    p.add_argument('--runs', default=None, help='tables: directory of run files (default results/<task>/runs)')
    p.add_argument('--sets', nargs='*', default=None)
    p.add_argument('--resume', action='store_true')
    p.add_argument('--track', action='store_true', help='gifs: the tracer animation')
    a = p.parse_args(argv)
    task = get_task(a.task, a.data)
    if a.cmd == 'train':
        run_train(task, a.model, a.iters, a.seed, a.device, a.out, a.resume, a.lr)
    elif a.cmd == 'evaluate':
        run_evaluate(task, a.model, a.weights, a.device, a.out)
    elif a.cmd == 'generate':
        task.generate(a.sets)
    elif a.cmd == 'calibrate':
        task.calibrate(a.device, a.out)
    elif a.cmd == 'tables':
        task.tables(a.runs)
    elif a.cmd == 'gifs':
        task.gifs(a.device, a.weights, track=a.track)
    elif a.cmd == 'figures':
        task.figures(a.device, a.weights)


if __name__ == '__main__':
    main()
