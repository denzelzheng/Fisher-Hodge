"""Samplers on a linear closed system, where the stationary law of the molecule numbers is multinomial."""
import numpy as np
import torch
from hyperme import Hypergraph, HyperME, presets, stochastic as St
from hyperme.coarse import stationary


def ring_model(n_states=4, seed=0):
    torch.manual_seed(seed)
    E = [(i, (i + 1) % n_states) for i in range(n_states)]
    with presets.default_dtype(torch.float64):
        hg = Hypergraph.from_graph(E, n_states)
        m = HyperME.build(hg, hidden=8, k_scale=1.0, force_gain_init=1.0).double()
        m.free_energy.mlp = None
    g = torch.Generator().manual_seed(seed)
    drive = dict(U=0.5 * torch.randn(1, n_states, generator=g, dtype=torch.float64),
                 force=torch.full((1, hg.H), 0.8, dtype=torch.float64))      # a net drive around the ring
    return m, drive


def test_gillespie_multinomial_and_current():
    m, drive = ring_model()
    ctx = m.prepare(drive)
    Q = m.generator(torch.ones(1, 4, dtype=torch.float64), ctx)[0].T.detach().numpy()
    pi = stationary(Q)
    omega, B = 40, 64
    big = ctx.expand(B)
    rates = lambda c: m.rates(c, big, zero_safe=True)
    n0 = torch.zeros(B, 4, dtype=torch.float64); n0[:, 0] = omega
    g = torch.Generator().manual_seed(1)
    res = St.gillespie(rates, m.hg, n0, omega, t_end=60.0, dt_rec=0.5, generator=g)
    assert (res['n'].sum(-1) == omega).all() and (res['n'] >= 0).all()
    st = St.fluctuation_stats(res, t_burn=10.0)
    mean_ref, var_ref = omega * pi, omega * pi * (1 - pi)
    assert np.allclose(st['occ_mean'].numpy(), mean_ref, rtol=0.08)
    assert np.allclose(st['occ_var'].numpy(), var_ref, rtol=0.2)
    # mean current along every ring edge = Omega * stationary probability current
    Jss = pi[0] * Q[0, 1] - pi[1] * Q[1, 0]
    assert Jss > 0
    assert np.allclose(st['current_mean'].numpy(), omega * Jss, rtol=0.1)
    assert (st['current_diffusion'] > 0).all()


def test_chemical_langevin_mean():
    m, drive = ring_model(seed=2)
    ctx = m.prepare(drive)
    B, omega = 32, 200
    big = ctx.expand(B)
    c0 =torch.full((B, 4), 0.25, dtype=torch.float64)
    g = torch.Generator().manual_seed(3)
    res = St.chemical_langevin(lambda c: m.rates(c, big, zero_safe=True), m.hg, c0, omega, t_end=20.0, dt_rec=0.5, generator=g)
    Q = m.generator(torch.ones(1, 4, dtype=torch.float64), ctx)[0].T.detach().numpy()
    pi = stationary(Q)
    st =St.fluctuation_stats(res, t_burn=5.0)
    assert np.allclose(st['occ_mean'].numpy() / omega, pi, rtol=0.05)
    assert torch.allclose(res['n'].sum(-1), torch.full((B, res['n'].shape[1]), float(omega), dtype=torch.float64), rtol=1e-9)
