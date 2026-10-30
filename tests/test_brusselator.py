"""Tests of the Brusselator benchmark (CPU).  The tests that need the reference data or the shipped checkpoint skip
when these are absent."""
import math
import os

import numpy as np
import pytest
import torch

from neuralchart.chart import CONFIGS, chart_config, fit_chart
from neuralchart.fit import fit_linear, rollout
from neuralchart.diagnostics import relerr_traj
from benchmarks.brusselator import config as C
from benchmarks.brusselator.data import hg_of, true_rates, TruthNP, make_traj, load_set
from benchmarks.brusselator.models import build_chart, true_log_k

N = 4          # a small grid for the unit tests (same spacing h)


def _states(B, n, seed=0):
    g = torch.Generator().manual_seed(seed)
    return 0.2 + 3 * torch.rand(B, 2 * n * n, generator=g, dtype=torch.float64)


def _conditions(B, seed=1):
    g = torch.Generator().manual_seed(seed)
    a = 0.8 + 1.4 * torch.rand(B, generator=g, dtype=torch.float64)
    b = 2.0 + 2.5 * torch.rand(B, generator=g, dtype=torch.float64)
    return dict(logab=torch.stack([a.log(), b.log()], 1))


def test_ablations_are_one_override():
    base = CONFIGS['brusselator']['chart']
    for name, cfg in CONFIGS['brusselator'].items():
        diff = [k for k in base.__dataclass_fields__ if getattr(cfg, k) != getattr(base, k)]
        assert len(diff) <= (2 if name == 'oracle' else 1), (name, diff)        # the oracle is also not trained


@torch.no_grad()
def test_oracle_equals_reference_rates():
    m = build_chart(chart_config('brusselator', 'oracle'), n=N)
    c, dr = _states(5, N), _conditions(5)
    a, b = m.rates(c, dr)
    at, bt = true_rates(hg_of(N), dr['logab'], c)
    assert float(((a - at).abs() / at).max()) < 1e-12 and float(((b - bt).abs() / bt).max()) < 1e-12


@torch.no_grad()
def test_force_is_fixed_and_sigma_nonnegative():
    """with the given standard potentials the force log(a / b) is the reference force for every k, so local detailed
    balance and sigma >= 0 hold for any parameter value"""
    m = build_chart(chart_config('brusselator', 'chart'), n=N)
    m.kinetics.core.copy_(torch.randn(m.kinetics.core.shape, generator=torch.Generator().manual_seed(2), dtype=torch.float64))
    c, dr = _states(5, N), _conditions(5)
    a, b = m.rates(c, dr)
    at, bt = true_rates(hg_of(N), dr['logab'], c)
    f, ft = torch.log(a / b), torch.log(at / bt)
    assert torch.allclose(f, ft, rtol=0, atol=1e-10)
    J = a - b
    assert bool((J * f >= 0).all()) and bool(((J * f).sum(-1) >= 0).all())


@torch.no_grad()
def test_reversible_ablation_dissipates_free_energy():
    m = build_chart(chart_config('brusselator', 'chart-reversible'), n=N)
    out = m.readouts(_states(4, N), _conditions(4))
    assert float(out['A'].abs().max()) == 0.0
    assert bool((out['dFdt'] <= 1e-10 * out['sigma']).all())


def test_nnls_recovers_the_rate_constants():
    """noiseless trajectories of the reference, sampled finely: the closed-form step finds log k of every hyperedge
    type (the remaining error is that of the Simpson rule)"""
    from scipy.integrate import solve_ivp
    dt = 0.01
    rng = np.random.default_rng(0)
    cs, labs = [], []
    for i in range(4):
        a, b = rng.uniform(0.9, 1.1), rng.uniform(2.3, 3.0)
        c0 = np.concatenate([a * np.exp(0.4 * rng.standard_normal(N * N)), b / a * np.exp(0.4 * rng.standard_normal(N * N))])
        ts = np.arange(41) * dt
        sol = solve_ivp(TruthNP(N, a, b).rhs, (0, ts[-1]), c0, method='DOP853', t_eval=ts, rtol=1e-12, atol=1e-14)
        cs.append(sol.y.T); labs.append([math.log(a), math.log(b)])
    tr = dict(c=torch.as_tensor(np.stack(cs)), drive=dict(logab=torch.tensor(labs, dtype=torch.float64)))
    m = build_chart(chart_config('brusselator', 'chart'), n=N)
    fit_linear(m, dict(train=tr), dt, log=lambda *a: None)
    err = np.abs(m.kinetics.core.detach().numpy() - np.array(true_log_k()))
    assert err.max() < 1e-4, err


DATA_OK = os.path.exists(os.path.join(C.DATA, 'train.pt'))
CKPT = os.path.join(C.CHECKPOINTS, 'chart.pt')


@pytest.mark.slow
@pytest.mark.skipif(not DATA_OK, reason='Brusselator data not generated')
def test_generator_reproduces_stored_data():
    d = torch.load(os.path.join(C.DATA, 'train.pt'))
    Cc, lab, homog = make_traj(('train', 2, C.W_TRAIN, d['n'], 1, d['c'].shape[0]))
    assert homog == bool(d['homog'][2])
    assert torch.equal(torch.as_tensor(Cc), d['c'][2]) and torch.equal(torch.tensor(lab, dtype=torch.float64), d['logab'][2])


@pytest.mark.slow
@pytest.mark.skipif(not (DATA_OK and os.path.exists(CKPT)), reason='data or checkpoint missing')
@torch.no_grad()
def test_shipped_checkpoint():
    """the shipped Neural Chart on four test trajectories, with the reference integrator in float64"""
    m = build_chart(chart_config('brusselator', 'chart'), n=C.N_TRAIN)
    m.load_state_dict(torch.load(CKPT, map_location='cpu'))
    assert np.abs(m.kinetics.core.numpy() - np.array(true_log_k())).max() < 1e-3
    sd = load_set('training')
    idx = [0, 1, 16, 17]
    R, _, _ = rollout(m, {k: v[idx] for k, v in sd['drive'].items()}, sd['c'][idx, 0], 40, C.DT)
    e = relerr_traj(R.double(), sd['c'][idx, :41].double())[:, -1]
    assert float(e.max()) < 1e-3
