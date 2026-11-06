"""Tests of the Brusselator benchmark (CPU).  The tests that need the reference data or the shipped checkpoint skip
when these are absent."""
import math
import os

import numpy as np
import pytest
import torch

from neuralchart import NeuralChart
from neuralchart.fit import fit_linear, rollout
from neuralchart.diagnostics import relerr_traj
from benchmarks.brusselator import config as C
from benchmarks.brusselator.data import conditions, hg_of, true_rates, TruthNP, make_traj, load_set
from benchmarks.brusselator.models import known_chem_mu0, known_mu0, oracle, true_log_k

N = 4          # a small grid for the unit tests (same spacing h)


def _states(B, n, seed=0):
    g = torch.Generator().manual_seed(seed)
    return 0.2 + 3 * torch.rand(B, 2 * n * n, generator=g, dtype=torch.float64)


def _conditions(B, seed=1):
    g = torch.Generator().manual_seed(seed)
    a = 0.8 + 1.4 * torch.rand(B, generator=g, dtype=torch.float64)
    b = 2.0 + 2.5 * torch.rand(B, generator=g, dtype=torch.float64)
    return conditions(torch.stack([a.log(), b.log()], 1))


def _chart(n=N, **kw):
    return NeuralChart(hg_of(n), chemostats=known_chem_mu0(), **kw)


@torch.no_grad()
def test_oracle_equals_reference_rates():
    m = oracle(N)
    c, dr = _states(5, N), _conditions(5)
    a, b = m.rates(c, dr)
    at, bt = true_rates(hg_of(N), dr['logab'], c)
    assert float(((a - at).abs() / at).max()) < 1e-12 and float(((b - bt).abs() / bt).max()) < 1e-12


@torch.no_grad()
def test_sigma_nonnegative_for_any_parameters():
    """the Neural Chart (learned standard potentials): for random k and mu0 every hyperedge flux runs along the
    model's own force log(a / b), so sigma >= 0"""
    m = _chart()
    g = torch.Generator().manual_seed(3)
    for p in m.linear_parameters():
        p.data.copy_(torch.randn(p.shape, generator=g, dtype=torch.float64))
    a, b = m.rates(_states(5, N), _conditions(5))
    J, f = a - b, torch.log(a / b)
    assert bool((J * f >= 0).all()) and bool(((J * f).sum(-1) >= 0).all())


@torch.no_grad()
def test_given_mu0_fixes_the_force():
    """with the given standard potentials the force log(a / b) is the reference force for every k, so local detailed
    balance against the true force and sigma >= 0 hold for any parameter value"""
    m = _chart(mu0=known_mu0())
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
    m = _chart(reversible=True)
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
    tr = dict(c=torch.as_tensor(np.stack(cs)), drive=conditions(torch.tensor(labs, dtype=torch.float64)))
    m = _chart(mu0=known_mu0())
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
@pytest.mark.skipif(not (DATA_OK and os.path.exists(CKPT)), reason='data or checkpoints missing')
@pytest.mark.parametrize('name,tol', [('chart', 1e-2)])
@torch.no_grad()
def test_shipped_checkpoint(name, tol):
    """the shipped Neural Charts on four test trajectories over 40 sampling steps, with the reference integrator in
    float64"""
    m = NeuralChart.load(os.path.join(C.CHECKPOINTS, name + '.pt'))
    m.step = dict(m.step, dt_max=C.DT / 8)
    sd = load_set('training')
    idx = [0, 1, 16, 17]
    R, _, _ = rollout(m, {k: v[idx] for k, v in sd['drive'].items()}, sd['c'][idx, 0], 40, C.DT)
    e = relerr_traj(R.double(), sd['c'][idx, :41].double())[:, -1]
    assert float(e.max()) < tol
