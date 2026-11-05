"""The library interface on a small driven network: builders, fit, readouts, files, diagnostics."""
import warnings

import numpy as np
import pytest
import torch

from neuralchart import Conditions, Hypergraph, NeuralChart, Trajectories


def network(**kw):
    return Hypergraph.from_strings(['A + X <-> 2 X', 'X <-> Y', 'Y <-> B'], chemostats=['A', 'B'], **kw)


def data(hg, B=6, steps=20, dt=0.05, seed=0):
    g = torch.Generator().manual_seed(seed)
    logc = torch.stack([torch.empty(B, dtype=torch.float64).uniform_(-0.5, 0.5, generator=g), torch.zeros(B, dtype=torch.float64)], 1)
    truth = NeuralChart(hg, chemostats=[0.0, 0.0], excess=False, material=False)
    with torch.no_grad():
        truth.kinetics.core.copy_(torch.tensor([0.2, 0.5, -0.3], dtype=torch.float64))
    c = [0.3 + torch.rand(B, hg.M, generator=g, dtype=torch.float64)]
    cond = Conditions(chem_logc=logc)
    with torch.no_grad():
        for _ in range(steps):
            c.append(c[-1] + dt * truth(c[-1], cond))
    return torch.stack(c, 1), cond, dt


def test_builders():
    hg = network()
    assert hg.n_species == 2 and hg.H == 3 and hg.meta['chemostats'] == ['A', 'B']
    assert np.abs(hg.S @ hg.cycle_basis()).sum() < 1e-12 and np.abs(hg.S.T @ hg.conservation_laws()).sum() < 1e-12
    grid = network(diffusing=['X'], grid_n=3)
    assert grid.V == 9 and int(grid.meta['typ'].max()) == 3
    sm = Hypergraph.from_stoichiometry(None, S=[[-1, 1], [1, -1]])
    assert sm.H == 2 and sm.cycle_basis().shape[1] == 1 and sm.conservation_laws().shape[1] == 1
    assert Hypergraph.from_grid(3).H == 18
    pts = np.random.default_rng(0).random((30, 2))
    cloud = Hypergraph.from_points(pts)
    assert cloud.V == 30 and abs(float(np.mean(cloud.meta['site_volume'])) - 1.0) < 1e-6 and (cloud.meta['edge_weight'] > 0).all()
    mesh = Hypergraph.from_mesh(np.array([[0, 1], [1, 2]]), 3, site_volume=[1.0, 2.0, 1.0], edge_weight=[1.0, 0.5])
    assert mesh.H == 2 and mesh.meta['edge_weight'][1] == 0.5


def test_fit_rollout_save_load_readouts(tmp_path):
    hg = network()
    c, cond, dt = data(hg)
    chart = NeuralChart(hg, chemostats=[0.0, 0.0])
    info = chart.fit(Trajectories(c, cond, times=dt * np.arange(c.shape[1])), iters=5, log=lambda *a: None)
    assert info['train_steps_done'] == 5 and chart.diagnostics['design_rank'] >= 1
    R = chart.rollout(c[:, 0], cond, 3)
    assert R.shape == (6, 4, hg.M) and (R > 0).all()
    x = c[:, 0]
    assert chart.excess_potential(x).shape == x.shape and chart.mobility(x, cond).shape == (6, 3)
    assert chart.flux(x, cond).shape == (6, 3) and (chart.entropy_production(x, cond) >= 0).all()
    ca = chart.cycle_affinity(cond)
    assert ca.shape == (6, hg.cycle_basis().shape[1])
    p = tmp_path / 'chart.pt'
    chart.save(p)
    back = NeuralChart.load(p)
    assert torch.allclose(back.flux(x, cond), chart.flux(x, cond))


def test_missing_chemostat_energies_warn():
    hg = network()
    c, cond, dt = data(hg)
    with pytest.warns(UserWarning, match='standard free energies'):
        NeuralChart(hg).fit(Trajectories(c, cond, dt=dt), iters=1, log=lambda *a: None)


def test_non_uniform_times_are_refused():
    with pytest.raises(AssertionError):
        Trajectories(torch.zeros(1, 3, 2), {}, times=[0.0, 1.0, 3.0])


def test_no_class_starts_below_the_floor():
    """data with one class switched off: the factor is at zero, the log rate of the class keeps a gradient"""
    from neuralchart.fit import FLOOR, fit_linear
    hg = network()
    truth = NeuralChart(hg, chemostats=[0.0, 0.0], excess=False, material=False)
    with torch.no_grad():
        truth.kinetics.core.copy_(torch.tensor([0.2, -60.0, -0.3], dtype=torch.float64))
    c = [0.3 + torch.rand(6, hg.M, generator=torch.Generator().manual_seed(0), dtype=torch.float64)]
    cond = Conditions(chem_logc=torch.zeros(6, 2, dtype=torch.float64))
    with torch.no_grad():
        for _ in range(20):
            c.append(c[-1] + 0.05 * truth(c[-1], cond))
    chart = NeuralChart(hg, chemostats=[0.0, 0.0])
    info = fit_linear(chart, Trajectories(torch.stack(c, 1), cond, dt=0.05).as_dict(), 0.05, log=lambda *a: None)
    assert np.isclose(info['floor'], FLOOR * max(info['kappa']))
    assert float(chart.kinetics.core.min()) >= np.log(info['floor']) - 1e-9


def test_the_entropy_production_error_is_on_positive_times():
    from neuralchart.diagnostics import sigma_error, sigma_times
    t = list(sigma_times(21))
    assert t[0] > 0 and t == [5, 10, 15, 20]
    e, r = sigma_error(2 * torch.ones(3, 4), torch.ones(3, 4))
    assert torch.allclose(e, torch.ones(3)) and torch.allclose(r, 2 * torch.ones(3))


def test_variable_projection_lowers_the_residual_and_leaves_an_exact_start():
    from neuralchart.fit import fit_linear, fit_varpro
    hg = network()
    c, cond, dt = data(hg)
    d = Trajectories(c, cond, dt=dt).as_dict()
    chart = NeuralChart(hg, chemostats=[0.0, 0.0])
    start = fit_linear(chart, d, dt, log=lambda *a: None)['residual_theta']
    out = fit_varpro(chart, d, dt, steps=20, log=lambda *a: None)
    assert out['residual_end'] <= 1.01 * max(start, 1e-5)
    assert all(p.grad is None for p in chart.parameters())


def test_no_projection_is_one_override():
    from neuralchart import VARIANTS
    assert VARIANTS['chart-no-projection'] == dict(projection=False)
    hg = network()
    c, cond, dt = data(hg)
    chart = NeuralChart(hg, chemostats=[0.0, 0.0], projection=False)
    info = chart.fit(Trajectories(c, cond, dt=dt), iters=1, log=lambda *a: None)
    assert 'varpro' not in info and 'linear_fit' in info
    assert NeuralChart(hg, chemostats=[0.0, 0.0]).projection
