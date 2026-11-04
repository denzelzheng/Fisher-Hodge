"""Tests of the reaction-network benchmarks (CPU, a few seconds each)."""
import math
import os

import numpy as np
import pytest
import torch

from neuralchart import NeuralChart
from neuralchart.run import build_chart as build_variant, get_task
from neuralchart.fit import fit_linear
from benchmarks.common import CHECKPOINTS
from benchmarks.reaction_networks import config as C
from benchmarks.reaction_networks import truth as TR
from benchmarks.reaction_networks.models import known_chem_mu0, known_mu0, oracle, true_log_k

BENCHES = C.BENCHMARKS


def random_condition(bench, n, B, g):
    return TR.conditions(_raw_condition(bench, n, B, g))


def _raw_condition(bench, n, B, g):
    if bench == 'mapk_cascade':
        return dict(logc=torch.stack([torch.empty(B, dtype=torch.float64).uniform_(-2, 1.5, generator=g),
                                      torch.zeros(B, dtype=torch.float64), torch.zeros(B, dtype=torch.float64)], 1))
    if bench in ('phosphorylation_cycle', 'phosphorylation_interacting', 'phosphorylation_saturating'):
        return dict(U=0.5 * torch.randn(B, n * n, generator=g, dtype=torch.float64),
                    dG=torch.empty(B, dtype=torch.float64).uniform_(0.5, 4, generator=g))
    if bench == 'phosphorylation_heterogeneous':
        H = TR.hg_of(n).H
        return dict(U=0.5 * torch.randn(B, n * n, generator=g, dtype=torch.float64),
                    dG=torch.empty(B, dtype=torch.float64).uniform_(0.5, 4, generator=g),
                    logm=0.5 * torch.randn(B, H, generator=g, dtype=torch.float64),
                    edge=torch.randn(B, H, 3, generator=g, dtype=torch.float64),
                    dsite=torch.randn(B, n * n, 3, generator=g, dtype=torch.float64))
    return dict(logab=torch.stack([torch.empty(B, dtype=torch.float64).uniform_(1.0, 1.4, generator=g),
                                   torch.empty(B, dtype=torch.float64).uniform_(-1.5, 1.3, generator=g)], 1))


def random_state(hg, B, g, lo=0.05, hi=2.0):
    return lo + (hi - lo) * torch.rand(B, hg.M, generator=g, dtype=torch.float64)


@pytest.mark.parametrize('bench', BENCHES)
def test_oracle_reproduces_the_reference_rates(bench):
    C.configure(bench)
    m = oracle(C.N_TRAIN)
    g = torch.Generator().manual_seed(0)
    c = random_state(m.hg, 6, g, hi=2.0 if bench not in C.NONIDEAL else 3.0)
    drive = random_condition(bench, C.N_TRAIN, 6, g)
    a, b = m.rates(c, drive)
    at, bt = TR.true_rates(m.hg, drive, c)
    assert float(((a - at).abs() / at).max()) < 1e-10
    assert float(((b - bt).abs() / bt).max()) < 1e-10


@pytest.mark.parametrize('bench', BENCHES)
def test_structure_holds_for_random_parameters(bench):
    """local detailed balance with the model's own force, sigma >= 0 and conserved moieties for any parameter value"""
    C.configure(bench)
    m = build_variant(get_task(bench), 'chart')
    g = torch.Generator().manual_seed(1)
    with torch.no_grad():
        for p in m.parameters():
            p.add_(0.5 * torch.randn(p.shape, generator=g, dtype=p.dtype))
    c = random_state(m.hg, 5, g, hi=3.0 if bench not in C.NONIDEAL else 2.5)
    drive = random_condition(bench, C.N_TRAIN, 5, g)
    ctx = m.prepare(drive)
    a, b = m.rates(c, ctx)
    f = torch.log(a) - torch.log(b)
    mu = m.mu(c, ctx)
    force = -m.hg.apply_ST(mu) + ctx['A']                          # -<S_h, mu> + A_h
    assert torch.allclose(f, force, atol=1e-9, rtol=1e-9)
    assert float(((a - b) * f).sum(-1).min()) >= 0
    L = m.hg.on('cpu', torch.float64)['L']
    if L.shape[1]:
        assert float((m(c, ctx) @ L).abs().max()) < 1e-9 * max(1.0, float((a + b).max()))


def _noiseless_schlogl(n_traj=6, steps=30):
    """reference trajectories of the ideal Schloegl model (DOP853) at random conditions"""
    rng = np.random.default_rng(0)
    cs, lab = [], []
    for i in range(n_traj):
        a, b = rng.uniform(2.85, 3.1), rng.uniform(1.0, 3.2)
        x0 = rng.uniform(0.5, 3.0, 64)
        cs.append(TR.solve_schlogl(8, a, b, x0, steps)); lab.append([math.log(a), math.log(b)])
    return dict(c=torch.tensor(np.array(cs)), drive=TR.conditions(dict(logab=torch.tensor(lab, dtype=torch.float64))))


def test_closed_form_step_recovers_the_true_constants():
    """with the standard potentials given, NNLS recovers log k exactly"""
    C.configure('schlogl')
    tr = _noiseless_schlogl()
    m = build_variant(get_task('schlogl'), 'chart-given-mu0')
    info = fit_linear(m, dict(train=tr), C.DT, log=lambda *a: None)
    assert info['residual_theta'] < 1e-6
    assert np.allclose(m.kinetics.core.detach().numpy(), true_log_k(), atol=1e-3)


def test_closed_form_step_with_learned_potentials_fits_the_rates():
    """with learned standard potentials (the default chart), the closed-form step reproduces the true one-way rates"""
    C.configure('schlogl')
    tr = _noiseless_schlogl()
    m = build_variant(get_task('schlogl'), 'chart')
    info = fit_linear(m, dict(train=tr), C.DT, log=lambda *a: None)
    assert info['residual_theta'] < 1e-6
    c, drive = tr['c'][:2, 5], {k: v[:2] for k, v in tr['drive'].items()}
    a, b = m.rates(c, drive)
    at, bt = TR.true_rates(m.hg, drive, c)
    assert float(((a - at).abs() / at).max()) < 1e-3 and float(((b - bt).abs() / bt).max()) < 1e-3


def test_heterogeneous_medium():
    """the descriptors enter only through g_theta, which starts at zero, and k does not read the conditions; the medium
    is reproducible and the law is the documented one on a constant map"""
    C.configure('phosphorylation_heterogeneous')
    task = get_task('phosphorylation_heterogeneous')
    m = build_variant(task, 'chart')
    g = torch.Generator().manual_seed(2)
    dr = random_condition('phosphorylation_heterogeneous', C.N_TRAIN, 3, g)
    k = m.prepare(dr)['k']
    typ = torch.as_tensor(m.hg.meta['typ'])
    for t in range(4):
        assert float(k[:, typ == t].std()) < 1e-12                 # g = 0 at initialisation: one constant per class
    assert m.kinetics.net is not None and build_variant(task, 'chart-const-k').kinetics.net is None
    with torch.no_grad():
        for p in m.kinetics.net.parameters():
            p.add_(0.1 * torch.randn(p.shape, generator=g, dtype=p.dtype))
    c = random_state(m.hg, 3, g)
    k1 = m.kinetics(m.prepare(dr)['feats'], c)
    dr2 = dict(dr, dG=dr['dG'] + 1.0, U=dr['U'] * 0.5)
    assert torch.allclose(m.kinetics(m.prepare(dr2)['feats'], c), k1)                  # conditions do not enter k
    n, P = C.N_TRAIN, TR.PIXELS
    logm = TR.medium_log_factors(np.zeros((n * P, n * P)), TR.hg_of(n), n)
    tm = np.asarray(TR.hg_of(n).meta['typ'])
    assert abs(logm[tm == 0]).max() < 1e-12 and abs(logm[tm == 1] + 1.0).max() < 1e-12      # exp(1 - 2) at H = 0
    C.configure('schlogl')


@pytest.mark.slow
@pytest.mark.parametrize('bench', BENCHES)
def test_shipped_checkpoint_on_reference_data(bench):
    """the shipped Neural Chart weights on the first trajectories of the training-condition set (needs the data)"""
    C.configure(bench)
    w = os.path.join(CHECKPOINTS, bench, 'chart.pt')
    if not (os.path.exists(w) and os.path.exists(os.path.join(C.DATA, 'train_conditions.pt'))):
        pytest.skip('data or checkpoint not present (python -m benchmarks.<name>.run generate)')
    from neuralchart.fit import rollout
    from benchmarks.reaction_networks.data import load_set
    m = NeuralChart.load(w).accelerate()
    sd = load_set('train_conditions')
    Y = sd['c'][:2, :41]
    R, _, _ = rollout(m, {k: v[:2] for k, v in sd['drive'].items()}, Y[:, 0], 40, C.DT)
    ch = slice(0, 2 * C.N_TRAIN ** 2) if bench == 'phosphorylation_cycle' else slice(None)
    e = torch.linalg.vector_norm((R.double() - Y)[..., ch], dim=-1) / torch.linalg.vector_norm(Y[..., ch], dim=-1)
    assert float(e.max()) < 2.5e-2
    assert float(R.min()) > 0
