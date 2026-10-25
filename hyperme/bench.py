"""Speed of the positivity-preserving substeps: prototype loop vs hyperme (legacy loop and vectorized).

    python -m hyperme.bench --problem s2_tcme --device cuda      (trained checkpoint runs/tcme/S2_tcme_s0.pt)
    python -m hyperme.bench --problem c1 --device cpu            (trained checkpoint hyper/runs/main/hyperme_s0.pt)
Times (a) a 200-step evaluation rollout of the 16 test trajectories and (b) one training iteration (K = 8 rollout
steps, batch 8, forward + backward).  Prints a json line."""
import argparse, importlib.util, json, os, sys, time
import numpy as np
import torch
from . import presets
from .presets import ROOT, DT


def load_module(name, path, extra_path=None):
    if extra_path and extra_path not in sys.path:
        sys.path.insert(0, extra_path)
    spec = importlib.util.spec_from_file_location(name, path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def sync(dev):
    if str(dev).startswith('cuda'):
        torch.cuda.synchronize()


def timeit(fn, dev, reps):
    fn(); sync(dev)
    ts = []
    for _ in range(reps):
        t = time.time(); fn(); sync(dev); ts.append(time.time() - t)
    return float(np.median(ts))


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--problem', default='s2_tcme'); p.add_argument('--device', default='cpu')
    p.add_argument('--reps', type=int, default=3); p.add_argument('--sparse', action='store_true')
    a = p.parse_args()
    dev = a.device
    from .gpu_guard import require_free
    require_free(dev)
    if a.problem == 's2_tcme':
        T = load_module('models_tcme', os.path.join(ROOT, 'fisher_hodge', 'models_tcme.py'), os.path.join(ROOT, 'fisher_hodge'))
        sd = torch.load(os.path.join(ROOT, 'runs', 'tcme', 'S2_tcme_s0.pt'), map_location=dev)
        old = T.TCME('S2').to(dev); old.load_state_dict(sd)
        new = presets.s2_tcme().to(dev); presets.load_legacy_tcme(new, sd)
        data = presets.load_s2(device=dev)
        shape = lambda c: c.view(c.shape[0], 32, 32)
        old_prep = lambda d: old.prepare(d)
    else:
        torch.set_num_threads(4)
        O = load_module('hyper_legacy', os.path.join(ROOT, 'hyper', 'hyperme.py'))
        sd = torch.load(os.path.join(ROOT, 'hyper', 'runs', 'main', 'hyperme_s0.pt'), map_location=dev)
        old = O.HyperME(True).to(dev); old.load_state_dict(sd)
        torch.set_default_dtype(torch.float32)
        new = presets.c1().to(dev); presets.load_legacy_c1(new, sd)
        data = presets.load_c1(device=dev)
        shape = lambda c: c.view(c.shape[0], 3, 64)
        old_prep = lambda d: old.prepare(d['U'], d['dG'], 8)
    if a.sparse:
        new.hg.dense = False; new.hg._torch = {}
    te = data['test_id']; tr = data['train']
    res = dict(problem=a.problem, device=dev, dense=new.hg.dense)

    def roll_old():
        with torch.no_grad():
            ctx = old_prep(te['drive']); c = shape(te['c'][:, 0])
            for _ in range(200):
                c, _ = old.advance(c, ctx, DT)
        return c

    def roll_new(vec):
        def f():
            with torch.no_grad():
                ctx = new.prepare(te['drive']); c = te['c'][:, 0]
                for _ in range(200):
                    c, _ = new.advance(c, ctx, DT, vectorized=vec)
            return c
        return f

    idx = torch.arange(8, device=dev)

    def step_old():
        old.zero_grad()
        d = {k: v[idx] for k, v in tr['drive'].items()}
        ctx = old_prep(d); c = shape(tr['c'][idx, 0]); loss = 0
        for k in range(1, 9):
            c, _ = old.advance(c, ctx, DT)
            y = shape(tr['c'][idx, k]); loss = loss + ((c - y) ** 2).sum() / (y ** 2).sum()
        loss.backward()

    def step_new(vec):
        def f():
            new.zero_grad()
            d = {k: v[idx] for k, v in tr['drive'].items()}
            ctx = new.prepare(d); c = tr['c'][idx, 0]; loss = 0
            for k in range(1, 9):
                c, _ = new.advance(c, ctx, DT, vectorized=vec)
                y = tr['c'][idx, k]; loss = loss + ((c - y) ** 2).sum() / (y ** 2).sum()
            loss.backward()
        return f

    ro, rl, rv = roll_old(), roll_new(False)(), roll_new(True)()
    res['rollout_maxdiff_legacy'] = float((shape(rl) - ro).abs().max())
    res['rollout_maxdiff_vectorized'] = float((shape(rv) - ro).abs().max())
    res['rollout_s'] = dict(prototype=timeit(roll_old, dev, a.reps), legacy=timeit(roll_new(False), dev, a.reps),
                            vectorized=timeit(roll_new(True), dev, a.reps))
    res['train_step_s'] = dict(prototype=timeit(step_old, dev, a.reps), legacy=timeit(step_new(False), dev, a.reps),
                               vectorized=timeit(step_new(True), dev, a.reps))
    for k in ('rollout_s', 'train_step_s'):
        res[k]['speedup_vs_prototype'] = res[k]['prototype'] / res[k]['vectorized']
    print(json.dumps(res))


if __name__ == '__main__':
    main()
