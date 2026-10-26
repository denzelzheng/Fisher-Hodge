"""C3 Brusselator: the Patankar integrator against the explicit positive substeps (slow, CPU).

1. On the prototype checkpoint hyper/runs/C23/brus_hyperme_s0.pt: every integrator keeps c > 0 and sigma > 0, the
   explicit rollout reproduces the prototype 10T error (0.0477), and MPRK22 converges to it as the step shrinks.
2. Retraining with MPRK22 at 8 steps per sampling interval (seed 0, 2000 iterations) gives a model whose 10T error is
   within 20% of the explicit prototype both under its own integrator and under the explicit one, i.e. it has learned
   the continuous-time rates rather than a map tied to the step size."""
import json
import os
import numpy as np
import pytest
import torch
from hyperme import presets
from hyperme.c3_patankar import run_method
from hyperme.train import relerr
from conftest import REPO
from test_regression import _train

C3_10T = 0.0477


def need(p):
    if not os.path.exists(os.path.join(REPO, p)):
        pytest.skip(f'missing {p}')


@pytest.mark.slow
def test_patankar_on_prototype_checkpoint():
    need('hyper/runs/data/brus.pt'); need('hyper/runs/C23/brus_hyperme_s0.pt')
    torch.set_num_threads(int(os.environ.get('OMP_NUM_THREADS', '8')))
    data = presets.load_c3()
    te = data['test_id']
    m = presets.load_legacy_c3(presets.c3(), torch.load(os.path.join(REPO, 'hyper/runs/C23/brus_hyperme_s0.pt'))).eval()
    Rex, info = run_method(m, te, 'ssp', 0)
    e = float(np.median(relerr(Rex, te['c']).numpy()[:, -1]))
    assert abs(e - C3_10T) / C3_10T < 0.01 and info['substeps_per_step'] > 20
    dist = []
    for ns in (4, 8, 16):
        R, inf = run_method(m, te, 'mprk22', ns)
        assert R.min() > 0 and inf['sigma_min'] > 0
        dist.append(float(np.median(relerr(R, Rex).numpy()[:, -1])))
    assert dist[0] > dist[1] > dist[2] and dist[2] < 0.01, dist
    R, inf = run_method(m, te, 'patankar', 1)
    assert R.min() > 0 and inf['sigma_min'] > 0


@pytest.mark.slow
def test_c3_retrain_with_mprk22():
    need('hyper/runs/data/brus.pt')
    res = _train('c3', [0], 'cpu', 2000, extra=['--method', 'mprk22', '--nsub', '8', '--threads', '8'], tag='_mprk228')[0]
    st = res['sets']['test_id']
    assert st['relerr_median']['200'] < 1.2 * C3_10T and st['min_c'] > 0 and st['sigma_min'] > 0
    m = presets.c3()
    m.load_state_dict(torch.load(os.path.join(REPO, 'runs', 'hyperme_regression', 'c3_mprk228_s0.pt')))
    te = presets.load_c3()['test_id']
    Rex, _ = run_method(m.eval(), te, 'ssp', 0)
    assert float(np.median(relerr(Rex, te['c']).numpy()[:, -1])) < 1.2 * C3_10T
