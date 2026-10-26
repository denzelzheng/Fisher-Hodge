"""C3 equilibrium anchor: the anchor chemostat settings are detailed-balance points of the Brusselator, and the wired
model with the true constants reproduces the generator for all four chemostats, inside and far outside the driven range."""
import numpy as np
import torch
from hyperme import presets
from hyperme import c3_anchor as CA


def _anchor_logab(xs, d):
    return torch.tensor([[np.log(CA.EPS * xs), np.log(CA.EPS ** 2 * d), np.log(d), np.log(xs / CA.EPS)]], dtype=torch.float64)


def test_anchor_settings_balance_every_reaction():
    hg = CA.hg_c3()
    for xs, d in [(0.7, 300.0), (1.1, 1000.0), (1.5, 3000.0)]:
        c = torch.cat([torch.full((1, hg.V), xs, dtype=torch.float64), torch.full((1, hg.V), CA.EPS * xs, dtype=torch.float64)], 1)
        a, b = CA.true_rates(hg, _anchor_logab(xs, d), c)
        assert torch.allclose(a, b, rtol=1e-12)


def test_true_wired_model_matches_the_generator_for_four_chemostats():
    m = presets.c3_true_wired(presets.c3_wired())
    g = torch.Generator().manual_seed(0)
    lab = torch.cat([_anchor_logab(1.2, 800.0), torch.randn(4, 4, generator=g, dtype=torch.float64)])
    c = torch.rand(5, m.hg.M, generator=g, dtype=torch.float64) * 2 + 0.01
    a, b = m.rates(c, dict(logab=lab))
    at, bt = CA.true_rates(m.hg, lab, c)
    assert torch.allclose(a, at, rtol=1e-11) and torch.allclose(b, bt, rtol=1e-11)
    # a driven drive with two columns means d = e = 1
    a2, _ = m.rates(c[:1], dict(logab=lab[1:2, :2]))
    at2, _ = CA.true_rates(m.hg, torch.cat([lab[1:2, :2], torch.zeros(1, 2, dtype=torch.float64)], 1), c[:1])
    assert torch.allclose(a2, at2, rtol=1e-11)


def test_flux_featurizer_sees_the_chemostats():
    m = CA.build('flux')
    c = torch.rand(2, m.hg.M, dtype=torch.float64) + 0.5
    lab = torch.zeros(2, 4, dtype=torch.float64); lab[1, 1] = 1.0
    with torch.no_grad():
        a, b = m.rates(c, dict(logab=lab))
    assert torch.isfinite(a).all() and (a > 0).all() and not torch.allclose(a[0], a[1])
