"""One observable per loop of indistinguishable parallel channels (proposition 1'): on a small Brusselator the
closed-form step cannot split R1 reverse (X -> A) from R4 forward (X -> E) from densities alone, and recovers both
once the net R4 flux (the turnover of reservoir E) is observed.  V2: wiring the reservoir standard affinity
mu0_A - mu0_E keeps the true generator in the class."""
import numpy as np
import torch
from hyperme import presets
from hyperme import c3_anchor as CA
from hyperme.c3_observable import r4_weights
from hyperme.linear_fit import fit_linear

DT = 0.02


@torch.no_grad()
def brusselator_data(n=2, B=8, T1=21, seed=0):
    truth = presets.c3_true_wired(presets.c3_wired(n=n))
    g = torch.Generator().manual_seed(seed)
    lab = torch.stack([torch.empty(B, dtype=torch.float64).uniform_(-0.3, 0.3, generator=g),
                       torch.empty(B, dtype=torch.float64).uniform_(0.6, 1.2, generator=g)], -1)
    drive = dict(logab=lab)
    c = torch.rand(B, truth.hg.M, generator=g, dtype=torch.float64) * 2 + 0.2
    ctx = truth.prepare(drive)
    out = [c]
    for _ in range(T1 - 1):
        c, _ = truth.advance(c, ctx, DT, nsub=32)
        out.append(c)
    C = torch.stack(out, 1)
    Wo = r4_weights(truth.hg)
    a, b = truth.rates(C.reshape(B * T1, -1), dict(logab=lab.repeat_interleave(T1, 0)))
    return truth, dict(train=dict(c=C, drive=drive)), ((a - b) @ Wo.T).reshape(B, T1, 1), Wo


@torch.no_grad()
def r1_r4_ratio(m, truth, data):
    """learned / true: R1 reverse and R4 forward one-way rates"""
    tr = data['train']; c = tr['c'][:, 3]
    a, b = m.rates(c, tr['drive']); at, bt = truth.rates(c, tr['drive'])
    typ = m.hg.meta['typ']
    return float((b[:, typ == 0] / bt[:, typ == 0]).median()), float((a[:, typ == 3] / at[:, typ == 3]).median())


def test_turnover_observable_separates_the_parallel_channels():
    truth, data, obs, Wo = brusselator_data()
    m = presets.c3_wired(n=2)
    fit_linear(m, data, DT, log=lambda *a: None)
    r1, r4 = r1_r4_ratio(m, truth, data)
    assert abs(np.log(r1)) > 3 and abs(r4 - 1) > 0.1      # densities alone: the split is wrong (R1 reverse off by 100s)
    data['train'].update(obs=obs, obs_w=Wo)
    m = presets.c3_wired(n=2)
    fit_linear(m, data, DT, log=lambda *a: None)
    r1, r4 = r1_r4_ratio(m, truth, data)
    # R4 forward exact; R1 reverse (1e-3 of the X decay) is the small remainder of the total, so within 30 percent
    assert abs(np.log(r1)) < 0.3 and abs(r4 - 1) < 1e-3, (r1, r4)


def test_wired_reservoir_affinity_contains_the_truth():
    ref = presets.c3_true_wired(presets.c3_wired())
    m = presets.c3_wired(known_cycle=presets.C3_CYCLE_AE)
    mu_true = ref.drive.chem_mu0.detach()
    assert abs(float(mu_true[0] - mu_true[3]) - presets.C3_CYCLE_AE) < 1e-10
    with torch.no_grad():
        m.free_energy.mu0.copy_(ref.free_energy.mu0); m.kinetics.core.copy_(ref.kinetics.core)
        m.drive.chem_mu0_free.copy_(m.drive.mu0_N.T @ (mu_true - m.drive.mu0_p))
    assert torch.allclose(m.drive.chem_mu0_value(), mu_true, atol=1e-12)
    c = torch.rand(3, m.hg.M, dtype=torch.float64) + 0.1
    lab = torch.randn(3, 4, dtype=torch.float64) * 0.3
    a, b = m.rates(c, dict(logab=lab)); at, bt = CA.true_rates(m.hg, lab, c)
    assert torch.allclose(a, at, rtol=1e-10) and torch.allclose(b, bt, rtol=1e-10)
    assert any(p is m.drive.chem_mu0_free for p in m.linear_parameters()) and len(m.drive.chem_mu0_free) == 3
