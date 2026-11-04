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


def model(hg, excess=False, seed=0):
    torch.manual_seed(seed)
    m = MasterEquation.from_parts(hg, harmonic=False, excess=excess, material=excess)
    with torch.no_grad():
        m.kinetics.core.copy_(0.5 * torch.randn_like(m.kinetics.core))
        if excess:
            m.free_energy.l3.weight.normal_(0, 0.1)
            m.kinetics.net.net[-1].weight.normal_(0, 0.1)
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


def test_rate_form():
    """a, b = k eps exp(+- f / 2), eps = exp(<nu+ + nu-, y> / 2): with ln gamma = 0 the reactants enter as mass action in c"""
    hg = network()
    m = model(hg, excess=True)
    c, dr = state(hg, 4), drive(4)
    ctx = m.prepare(dr)
    a, b = m.rates(c, ctx)
    y = torch.log(c) + m.free_energy.correction(c, ctx)
    f = hg.dot(y + ctx['Ufull'], '+') - hg.dot(y + ctx['Ufull'], '-') + ctx['A']
    eps = torch.exp(0.5 * (hg.dot(y, '+') + hg.dot(y, '-')))
    k = m.kinetics(ctx['feats'], c)
    assert torch.allclose(a, k * eps * torch.exp(0.5 * f), rtol=1e-4)
    assert torch.allclose(b, k * eps * torch.exp(-0.5 * f), rtol=1e-4)


@pytest.mark.parametrize('excess', [False, True])
def test_local_detailed_balance_and_entropy(excess):
    hg = network()
    m = model(hg, excess)
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
    m = MasterEquation.from_parts(hg, harmonic=False, excess=False, material=False)
    dr = dict(U=torch.randn(2, 3), line=3.0 * torch.randn(2, hg.H))
    c = torch.tensor([[1.0, 1e-3, 0.5], [0.2, 0.3, 2.0]])
    c1, _ = positive_ssp_rk2(m, c, m.prepare(dr), 0.5)
    assert (c1 > 0).all()
    assert torch.allclose(c1.sum(-1), c.sum(-1), rtol=1e-12)


def test_excess_starts_at_zero_and_grows_without_bound():
    hg = network()
    m = MasterEquation.from_parts(hg, harmonic=False)
    c = state(hg, 2)
    assert float(m.free_energy.correction(c, None).abs().max()) == 0
    fe = model(hg, excess=True).free_energy
    x = torch.tensor([[1e1], [1e2], [1e3]]).expand(3, fe.NS).contiguous()[:, None]
    g = fe.excess_potential(x).abs().amax(-1)[:, 0]
    assert float(g[2]) > 5 * float(g[1]) > 25 * float(g[0]) > 0     # linear tails of the excess potential


def test_free_energy_is_the_integral_of_mu():
    """d F / d c = mu: the value of the excess free energy and its written-out gradient agree"""
    hg = network()
    m = model(hg, excess=True).double()
    dr = drive(3)
    ctx = m.prepare({k: v.double() for k, v in dr.items()})
    c = state(hg, 3).double().requires_grad_(True)
    g, = torch.autograd.grad(m.free_energy.value(c, ctx).sum(), c)
    assert torch.allclose(g, m.mu(c.detach(), ctx), atol=1e-9)


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
            elif name != 'chart':
                assert len(diff) == 1, (task, name, diff)
