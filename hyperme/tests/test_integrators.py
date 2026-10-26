import torch
import pytest
from hyperme import Hypergraph, HyperME, integrators as I, presets
from helpers import case, perturb, smooth


def linear_model(reversible, n=6, seed=0, B=3):
    """single-species transport on an n x n torus with a fixed potential: a linear master equation."""
    g = torch.Generator().manual_seed(seed)
    torch.manual_seed(seed)
    with presets.default_dtype(torch.float64):
        hg = Hypergraph.grid_transport(n)
        m = HyperME.build(hg, mode="flexible", hidden=16, k_scale=4.0, drive='none' if reversible else 'project')
        m.free_energy.mlp = None                      # phi = alpha U: linear
        m = perturb(m.double(), seed=seed)
    drive = dict(U=smooth(B, n, g).reshape(B, -1), force=torch.randn(B, hg.H, generator=g, dtype=torch.float64))
    c0 = torch.rand(B, hg.M, generator=g, dtype=torch.float64) + 0.01
    assert m.is_linear
    return m, m.prepare(drive), c0


def test_vectorized_matches_loop():
    m, drive, c0 = case('c1', seed=5)
    ctx = m.prepare(drive)
    with torch.no_grad():
        a, b = c0.clone(), c0.clone()
        for _ in range(10):
            a, _ = m.advance(a, ctx, 0.05, vectorized=False)
            b, nb = m.advance(b, ctx, 0.05, vectorized=True)
    assert torch.is_tensor(nb) and nb.shape == (c0.shape[0],)
    assert ((a - b).abs().max() / a.abs().max()) < 1e-3          # per-sample vs batch step size
    assert b.min() > 0


def test_vectorized_identical_when_nominal_step_binds():
    m, drive, c0 = case('tcme', seed=6)
    ctx = m.prepare(drive)
    with torch.no_grad():
        a, _ = m.advance(c0, ctx, 0.05, vectorized=False, dt_max=1e-3)
        b, _ = m.advance(c0, ctx, 0.05, vectorized=True, dt_max=1e-3)
    assert torch.allclose(a, b, rtol=1e-12, atol=1e-14)


@pytest.mark.parametrize('reversible', [True, False])
def test_implicit_euler_linear(reversible):
    m, ctx, c0 = linear_model(reversible)
    L = m.generator(c0, ctx)
    for dt in (0.01, 100.0):
        x, info = I.implicit_euler_linear(m, c0, ctx, dt)
        ref = torch.linalg.solve(torch.eye(m.hg.M, dtype=L.dtype) - dt * L, c0.unsqueeze(-1)).squeeze(-1)
        assert info['method'] == ('sym' if reversible else 'cgnr')
        assert ((x - ref).norm() / ref.norm()) < 1e-7, (dt, info)
        assert ref.min() > 0 and x.min() > -1e-9 * x.abs().max()
        assert torch.allclose(x.sum(-1), c0.sum(-1), rtol=1e-8)


def test_expm_step_matches_explicit():
    m, ctx, c0 = linear_model(False)
    with torch.no_grad():
        e = I.expm_step(m, c0, ctx, 0.2)
        x = c0
        for _ in range(4):
            x, _ = m.advance(x, ctx, 0.05, dt_max=2e-4)
    assert ((e - x).norm() / e.norm()) < 1e-6
    assert torch.allclose(e.sum(-1), c0.sum(-1), rtol=1e-12)


@pytest.mark.parametrize('name', ['c1', 'schlogl', 'grid_nonlocal'])
def test_mprk_positive_and_conserving_at_large_steps(name):
    m, drive, c0 = case(name, seed=8)
    ctx = m.prepare(drive)
    c0 = c0.clone(); c0[:, ::5] = 1e-6
    L = m.hg.on(c0.device, c0.dtype)['L']
    for order in (1, 2):
        with torch.no_grad():
            c, _ = I.mprk(m, c0, ctx, 2.0, nsub=1, order=order)
        assert torch.isfinite(c).all() and c.min() > 0
        if L.shape[1]:
            q0, q1 = c0 @ L, c @ L
            assert ((q1 - q0).norm(dim=-1) / q0.norm(dim=-1)).max() < 1e-12


@pytest.mark.parametrize('order', [1, 2])
def test_mprk_convergence_order(order):
    m, ctx, c0 = linear_model(False)
    with torch.no_grad():
        ref = I.expm_step(m, c0, ctx, 0.4)
        err = [float((I.mprk(m, c0, ctx, 0.4, nsub=n, order=order)[0] - ref).norm() / ref.norm()) for n in (8, 16, 32)]
    rates = [err[i] / err[i + 1] for i in range(2)]
    assert all(abs(r - 2 ** order) < 0.35 * 2 ** order for r in rates), (err, rates)


def test_mprk_differentiable():
    m, drive, c0 = case('c1', seed=9)
    ctx = m.prepare(drive)
    c, _ = I.mprk(m, c0, ctx, 0.2, nsub=2)
    c.pow(2).sum().backward()
    g = [p.grad for p in m.parameters() if p.grad is not None]
    assert g and all(torch.isfinite(x).all() for x in g) and sum(float(x.abs().sum()) for x in g) > 0


@pytest.mark.parametrize('name', ['grid_nonlocal', 'c1', 'reactions'])
def test_proximal_step(name):
    m, drive, c0 = case(name, seed=7, reversible=True)
    ctx = m.prepare(drive)
    L = m.hg.on(c0.device, c0.dtype)['L']
    for dt in (0.01, 1.0):
        c1, info = I.proximal_step(m, c0, ctx, dt)
        assert c1.min() > 0
        q0, q1 = c0 @ L, c1 @ L
        assert ((q1 - q0).norm(dim=-1) / q0.norm(dim=-1)).max() < 1e-12
        with torch.no_grad():
            F0, F1 = m.free_energy_value(c0, ctx), m.free_energy_value(c1, ctx)
        assert (F1 + info['dissipation'] <= F0 + 1e-9 * F0.abs()).all()
    # small steps agree with the explicit flow to first order
    dt = 1e-3
    c1, _ = I.proximal_step(m, c0, ctx, dt)
    with torch.no_grad():
        ce, _ = m.advance(c0, ctx, dt, dt_max=dt / 20)
    assert ((c1 - ce).norm() / (ce - c0).norm()) < 0.05
