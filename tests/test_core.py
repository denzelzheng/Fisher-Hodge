"""Unit tests of the library (CPU, a few seconds): hypergraph algebra, the structure of the master equation, the
integrators, the closed-form fit and the configurations."""
import dataclasses

import numpy as np
import pytest
import torch

from neuralchart import CONFIGS, Hypergraph, MasterEquation, fast
from neuralchart.chart import ChartConfig
from neuralchart.diagnostics import ldb_violation
from neuralchart.fit import fit_linear, rollout
from neuralchart.integrators import positive_ssp_rk2

# Schloegl-type network with chemostats A, B on a periodic 3 x 3 grid: A + 2X <-> 3X, X <-> B, X diffuses
RX = [({'A': 1, 'X': 2}, {'X': 3}), ({'X': 1}, {'B': 1})]


@pytest.fixture(autouse=True)
def float64():
    old = torch.get_default_dtype()
    torch.set_default_dtype(torch.float64)
    yield
    torch.set_default_dtype(old)


def network(n=3):
    return Hypergraph.reaction_transport(['X', 'A', 'B'], RX, grid_n=n, diffusing=['X'], chemostats=['A', 'B'])


def model(hg, activity=False, seed=0):
    torch.manual_seed(seed)
    m = MasterEquation.from_parts(hg, k_ref=1.0, T_exponent=0.0, harmonic=False, kinetics_table=True,
                                  kinetics_split=False, activity_correction=activity, activity_anchor=activity)
    with torch.no_grad():
        m.kinetics.core.copy_(0.5 * torch.randn_like(m.kinetics.core))
        if activity:
            m.free_energy.s_mu.fill_(0.7)
            m.free_energy.r_alpha.normal_()
            m.free_energy.r_beta.normal_()
    return m


def drive(B):
    g = torch.Generator().manual_seed(1)
    return dict(chem_logc=0.5 * torch.randn(B, 2, generator=g))


def state(hg, B):
    g = torch.Generator().manual_seed(2)
    return 0.2 + torch.rand(B, hg.M, generator=g)


def test_hypergraph_algebra():
    hg = network()
    S = hg.S
    assert S.shape == (hg.M, hg.H)
    assert np.abs(S @ hg.Z).max(initial=0) < 1e-10 and np.abs(S.T @ hg.L).max(initial=0) < 1e-10
    J = torch.randn(4, hg.H)
    assert torch.allclose(hg.apply_S(J), J @ torch.as_tensor(S).T)
    y = torch.randn(4, hg.M)
    assert torch.allclose(hg.apply_ST(y), y @ torch.as_tensor(S))


@pytest.mark.parametrize('activity', [False, True])
def test_local_detailed_balance_and_entropy(activity):
    hg = network()
    m = model(hg, activity)
    c, dr = state(hg, 5), drive(5)
    ctx = m.prepare(dr)
    a, b = m.rates(c, ctx)
    f = torch.log(a) - torch.log(b)
    force = -m.mu(c, ctx) @ torch.as_tensor(hg.S) + ctx['A']
    assert torch.allclose(f, force, atol=1e-10)
    assert ((a - b) * f >= 0).all()
    assert not ldb_violation(a - b, f, a + b).any()


def test_conservation_and_positivity():
    hg = Hypergraph.from_graph(np.array([[0, 1], [1, 2], [2, 0], [0, 2]]), 3)
    m = MasterEquation.from_parts(hg, harmonic=False, kinetics_table=True)
    dr = dict(U=torch.randn(2, 3), line=3.0 * torch.randn(2, hg.H))
    c = torch.tensor([[1.0, 1e-3, 0.5], [0.2, 0.3, 2.0]])
    c1, _ = positive_ssp_rk2(m, c, m.prepare(dr), 0.5)
    assert (c1 > 0).all()
    assert torch.allclose(c1.sum(-1), c.sum(-1), rtol=1e-12)


def test_activity_correction_is_monotone():
    hg = network()
    m = model(hg, activity=True)
    x = torch.linspace(1e-3, 5.0, 200)[:, None].expand(200, hg.M).contiguous()
    g = m.free_energy.correction(x, None)[:, 0]
    assert (g[1:] - g[:-1] >= -1e-12).all()
    with torch.no_grad():
        assert abs(float(m.free_energy.correction(torch.zeros(1, hg.M), None)[0, 0])) < 1e-12


def test_fast_integrator_matches_reference():
    hg = network()
    m = model(hg)
    c, dr = state(hg, 4), drive(4)
    m.step['dt_max'] = 0.05 / 4000                     # fine fixed substeps as the reference
    ref, _, _ = rollout(m, dr, c, 5, dt=0.05)
    m.step['dt_max'] = None
    fast.accelerate(m, control='sample_err', rtol=1e-9, precision='fp64', graphs=False)
    out, _, _ = rollout(m, dr, c, 5, dt=0.05)
    assert torch.allclose(out, ref, rtol=1e-6, atol=1e-9)
    assert (out > 0).all()


def test_closed_form_fit_recovers_constants():
    """NNLS on noiseless trajectories of the model itself returns its log-rate core"""
    hg = network()
    truth = model(hg, seed=3)
    c0, dr = state(hg, 8), drive(8)
    Y, _, _ = rollout(truth, dr, c0, 20, dt=0.01)
    m = model(hg, seed=4)
    with torch.no_grad():
        m.kinetics.core.zero_()
    fit_linear(m, dict(train=dict(c=Y, drive=dr)), 0.01, log=lambda *a: None)
    assert torch.allclose(m.kinetics.core, truth.kinetics.core, atol=1e-2)


def test_ldb_rule_ignores_rounding():
    a = torch.tensor([1e8, 1.0])
    b = a.clone()
    b[0] += 1e-6
    J, f = a - b, torch.tensor([1e-3, -1.0])
    assert not ldb_violation(J, f, a + b)[0]
    assert ldb_violation(torch.tensor([1.0]), torch.tensor([-1.0]))[0]


def test_every_ablation_is_one_override():
    for task, cfgs in CONFIGS.items():
        base = dataclasses.asdict(cfgs['chart'])
        for name, cfg in cfgs.items():
            assert isinstance(cfg, ChartConfig) and cfg.task == task
            diff = {k for k, v in dataclasses.asdict(cfg).items() if v != base[k]}
            if name.startswith('oracle'):
                assert 'train' in diff
            elif name == 'chart-const-k':                 # no material network, hence no state term either
                assert diff == {'descriptors', 'state'}
            elif name != 'chart':
                assert len(diff) == 1, (task, name, diff)
