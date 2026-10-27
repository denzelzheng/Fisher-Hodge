"""Identities behind the information-geometry checks (hyperme/info_geometry.py): the path KL of a jump process
against its time reversal is sum (a - b) log(a / b); the Lambda-weighted Hodge split is Pythagorean and the
Euclidean one is not; the one-site C1 chart contains the true rates and the entropy production of its master
equation equals Omega times the deterministic sigma (a first-order network); the single-particle path estimator converges to the exact (coarse) entropy production; the
transfer bound holds in the Onsager norm."""
import numpy as np
import scipy.linalg
import torch
from hyperme import info_geometry as IG
from hyperme.complex import Hypergraph
from hyperme.model import logmean


def test_poisson_path_kl_is_sigma():
    g = torch.Generator().manual_seed(0)
    a, b = torch.rand(100, generator=g, dtype=torch.float64) * 5, torch.rand(100, generator=g, dtype=torch.float64) * 1e-2
    D = lambda x, y: x * (torch.log(x) - torch.log(y)) - x + y
    assert torch.allclose(D(a, b) + D(b, a), (a - b) * (a.log() - b.log()), rtol=1e-12)


def test_fisher_split_is_pythagorean_euclid_is_not():
    rng = np.random.default_rng(0)
    S = rng.integers(-1, 2, size=(5, 12)).astype(float)
    a, b = np.exp(rng.normal(size=12) * 2), np.exp(rng.normal(size=12) * 2)
    f = np.log(a / b); ell = logmean(torch.as_tensor(a), torch.as_tensor(b)).numpy()
    Nl = scipy.linalg.null_space(S.T)
    ex, hk, sig = IG.hodge_parts(S, f, ell, ell, Nl)
    assert abs(sig - ex - hk) < 1e-12 * sig and abs(sig - (a - b) @ f) < 1e-12 * sig
    ex, hk, sig = IG.hodge_parts(S, f, ell, np.ones_like(ell), Nl)
    assert abs(sig - ex - hk) > 1e-3 * sig


def test_c1_chart_contains_truth_and_master_equation_limit():
    m = IG.c1_truth()
    c = torch.tensor([[0.3, 0.6, 1.2]], dtype=torch.float64)
    for dG in (1.0, 3.5):
        a, b = m.rates(c, IG.c1_drive(dG))
        x, xp, e = 0.3, 0.6, 1.2
        at = torch.tensor([2.0 * x * e * np.exp(dG / 2), 1.0 * xp]); bt = torch.tensor([2.0 * xp * e * np.exp(-dG / 2), 0.02 * x])
        assert torch.allclose(a[0], at.double(), rtol=1e-10) and torch.allclose(b[0], bt.double(), rtol=1e-10)
    # with E fixed the network is first order in X, Xp: the stationary law is binomial and the master-equation
    # entropy production equals Omega sigma_det at every Omega
    for O in (50, 1000):
        _, s_me, s_det, _ = IG.c1_exact(m, 2.0, O)
        assert abs(s_me / s_det - 1) < 1e-10


def test_walk_estimates_fine_and_coarse_sigma():
    Q = np.array([[0, 3.0, 0.5, 0.2], [0.4, 0, 2.5, 0.3], [2.0, 0.6, 0, 0.7], [0.5, 0.2, 0.9, 0]])
    np.fill_diagonal(Q, -Q.sum(1))
    from hyperme.coarse import stationary
    pi = stationary(Q)
    r = IG.walk(Q, pi, [np.array([0, 1, 1, 0])], nwalk=2000, njump=1000, seed=1)
    assert abs(r['fine']['path'] - r['fine']['exact']) < 5 * r['fine']['se']
    assert abs(r[2]['path'] - r[2]['exact']) < 5 * r[2]['se'] and r[2]['exact'] <= r['fine']['exact']


def test_transfer_bound_onsager_norm():
    hg = Hypergraph.grid_transport(4)
    S = hg.S; Nl = scipy.linalg.null_space(S.T)
    rng = np.random.default_rng(2)
    a, b = np.exp(rng.normal(size=hg.H)), np.exp(rng.normal(size=hg.H))
    J = a - b; ell = logmean(torch.as_tensor(a), torch.as_tensor(b)).numpy(); sig_h = J * np.log(a / b)
    dk = rng.uniform(-0.3, 0.3, hg.H)
    e = S @ (dk * J)
    assert IG.onsager_norm(S, e, ell, Nl) <= np.sqrt((dk ** 2 * sig_h).sum()) * (1 + 1e-9)
    assert IG.onsager_norm(S, e, ell, Nl) <= np.abs(dk).max() * np.sqrt(sig_h.sum())
    assert np.linalg.norm(e) <= np.linalg.norm(S, 2) * np.linalg.norm(dk * J) * (1 + 1e-12)


def test_ig2_kl_and_uniform_onsager_equality():
    hg = Hypergraph.grid_transport(4)
    S = hg.S; Nl = scipy.linalg.null_space(S.T)
    rng = np.random.default_rng(3)
    a, b = torch.as_tensor(np.exp(rng.normal(size=hg.H))), torch.as_tensor(np.exp(rng.normal(size=hg.H)))
    d = torch.as_tensor(rng.uniform(-0.5, 0.8, hg.H))
    kl = IG.poisson_kl((1 + d) * a, (1 + d) * b, a, b)
    assert torch.allclose(kl, ((a + b) * IG.phi_kl(d)).sum(), rtol=1e-12)
    J = (a - b).numpy(); ell = logmean(a, b).numpy(); cd = S @ J
    for dd in (0.2, -0.3):                    # uniform relative error: equality with sqrt(sigma_ex)
        assert abs(IG.onsager_norm(S, dd * cd, ell, Nl) - abs(dd) * IG.onsager_norm(S, cd, ell, Nl)) < 1e-10
