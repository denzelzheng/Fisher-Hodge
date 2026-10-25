"""Regression against the prototype numbers.

Fast: the prototype checkpoints loaded into the presets give the same rates and the same 10T errors as the stored
results.  Slow (-m slow): retraining the presets from scratch reproduces the published numbers,
  S2 TC-ME, 10T relative error on test_id, mean of seeds 0 and 1: 3.18e-3 (results/tcme_summary.md), tolerance 20%;
  C1 HyperME, 10T relative L2 of (X, Xp) on test_id, mean of seeds 0 and 1: 0.0224 (results/crn_summary.md),
  tolerance 20%, and the 65 moieties of ker S^T conserved to 1e-14 on every test set.
Set HYPERME_REUSE=1 to reuse finished runs in runs/hyperme_regression/ instead of retraining."""
import json
import os
import subprocess
import sys
import numpy as np
import pytest
import torch
from hyperme import presets
from hyperme.train import rollout, relerr
from conftest import REPO, load_legacy

TCME_10T, C1_10T, TOL = 3.18e-3, 0.0224, 0.20
OUT = os.path.join(REPO, 'runs', 'hyperme_regression')


def need(*paths):
    for p in paths:
        if not os.path.exists(os.path.join(REPO, p)):
            pytest.skip(f'missing {p}')


def test_tcme_checkpoint_equivalence(device):
    need('runs/tcme/S2_tcme_s0.pt', 'runs/tcme/S2_tcme_s0.json', 'runs/data/S2.pt')
    T = load_legacy('models_tcme', 'fisher_hodge/models_tcme.py', 'fisher_hodge')
    sd = torch.load(os.path.join(REPO, 'runs/tcme/S2_tcme_s0.pt'), map_location=device)
    old = T.TCME('S2').to(device); old.load_state_dict(sd); old.eval()
    new = presets.load_legacy_tcme(presets.s2_tcme().to(device), sd).eval()
    data = presets.load_s2(device=device)
    te = data['test_id']; B = te['c'].shape[0]
    with torch.no_grad():
        c = te['c'][:, 60]
        Jo = old.flux(c.view(B, 32, 32), old.prepare(te['drive'])).reshape(B, -1)
        Jn = new.flux(c, new.prepare(te['drive']))
        assert ((Jo - Jn).norm() / Jo.norm()) < 1e-5
        R, _ = rollout(new, te['drive'], te['c'][:, 0], 200)
    e10 = float(np.median(relerr(R.double(), te['c'].double()).cpu().numpy()[:, 200]))
    ref = json.load(open(os.path.join(REPO, 'runs/tcme/S2_tcme_s0.json')))['sets']['test_id']['relerr_median'][200]
    assert abs(e10 - ref) / ref < 0.01, (e10, ref)
    assert float((R.mean(-1) - 1).abs().max()) < 1e-6 and R.min() > 0


def test_c1_checkpoint_equivalence():
    need('hyper/runs/main/hyperme_s0.pt', 'hyper/runs/main/hyperme_s0.json', 'hyper/runs/data/crn.pt')
    O = load_legacy('hyper_legacy', 'hyper/hyperme.py')
    sd = torch.load(os.path.join(REPO, 'hyper/runs/main/hyperme_s0.pt'))
    with presets.default_dtype(torch.float64):
        old = O.HyperME(True); old.load_state_dict(sd); old.eval()
    new = presets.load_legacy_c1(presets.c1(), sd).eval()
    data = presets.load_c1()
    te = data['test_id']; B = te['c'].shape[0]
    with torch.no_grad():
        c = te['c'][:, 40]
        ao, bo = old.ab(c.view(B, 3, 64), old.prepare(te['drive']['U'], te['drive']['dG'], 8))
        an, bn = new.rates(c, new.prepare(te['drive']))
        assert ((ao - an).norm() / ao.norm()) < 1e-12 and ((bo - bn).norm() / bo.norm()) < 1e-12
        R, _ = rollout(new, te['drive'], te['c'][:, 0], 200)
    V = 64
    e10 = float(np.median(relerr(R[..., :2 * V], te['c'][..., :2 * V]).numpy()[:, 200]))
    ref = json.load(open(os.path.join(REPO, 'hyper/runs/main/hyperme_s0.json')))['sets']['test_id']['relL2_XXp_median']['200']
    assert abs(e10 - ref) / ref < 1e-6, (e10, ref)
    q = R @ new.hg.on(R.device, R.dtype)['L']
    assert q.shape[-1] == 65
    assert float(((q - q[:, :1]).norm(dim=-1) / q[:, :1].norm(dim=-1)).max()) <= 1e-14


def _train(problem, seeds, device, iters):
    os.makedirs(OUT, exist_ok=True)
    reuse = os.environ.get('HYPERME_REUSE') == '1'
    procs = []
    env = dict(os.environ, FH_ROOT=REPO, PYTHONPATH=REPO + os.pathsep + os.environ.get('PYTHONPATH', ''),
               OMP_NUM_THREADS='4', MKL_NUM_THREADS='4', OPENBLAS_NUM_THREADS='4')
    for s in seeds:
        js = os.path.join(OUT, f'{problem}_s{s}.json')
        if reuse and os.path.exists(js):
            continue
        cmd = [sys.executable, '-m', 'hyperme.train', '--problem', problem, '--seed', str(s), '--iters', str(iters),
               '--device', device, '--out', OUT]
        procs.append(subprocess.Popen(cmd, cwd=REPO, env=env, stdout=subprocess.DEVNULL, stderr=subprocess.PIPE))
    for p in procs:
        _, err = p.communicate()
        assert p.returncode == 0, err.decode()[-3000:]
    return [json.load(open(os.path.join(OUT, f'{problem}_s{s}.json'))) for s in seeds]


@pytest.mark.slow
def test_s2_tcme_retrain(device):
    need('runs/data/S2.pt')
    res = _train('s2_tcme', [0, 1], device, 2500)
    e = [r['sets']['test_id']['relerr_median']['200'] for r in res]
    assert abs(np.mean(e) - TCME_10T) / TCME_10T < TOL, e
    for r in res:
        for st in r['sets'].values():
            assert st['min_c'] > 0 and st['mass_err_max'] < 1e-5


@pytest.mark.slow
def test_c1_retrain():
    need('hyper/runs/data/crn.pt')
    res = _train('c1', [0, 1], 'cpu', 2000)
    e = [r['sets']['test_id']['relerr_median']['200'] for r in res]
    assert abs(np.mean(e) - C1_10T) / C1_10T < TOL, e
    for r in res:
        for st in r['sets'].values():
            assert st['n_conserved'] == 65 and st['cons_relerr_max'] <= 1e-14 and st['min_c'] > 0
