"""Tests of the reaction-network benchmarks (CPU, a few seconds each)."""
import math
import os

import numpy as np
import pytest
import torch

from neuralchart import chart
from neuralchart.fit import fit_linear, fit_varpro, state_function_parameters
from benchmarks.common import CHECKPOINTS
from benchmarks.reaction_networks import config as C
from benchmarks.reaction_networks import truth as TR
from benchmarks.reaction_networks.models import build_chart, true_log_k

BENCHES = C.BENCHMARKS


def random_condition(bench, n, B, g):
    if bench == 'mapk_cascade':
        return dict(logc=torch.stack([torch.empty(B, dtype=torch.float64).uniform_(-2, 1.5, generator=g),
                                      torch.zeros(B, dtype=torch.float64), torch.zeros(B, dtype=torch.float64)], 1))
    if bench == 'phosphorylation_cycle':
        return dict(U=0.5 * torch.randn(B, n * n, generator=g, dtype=torch.float64),
                    dG=torch.empty(B, dtype=torch.float64).uniform_(0.5, 4, generator=g))
    return dict(logab=torch.stack([torch.empty(B, dtype=torch.float64).uniform_(1.0, 1.4, generator=g),
                                   torch.empty(B, dtype=torch.float64).uniform_(-1.5, 1.3, generator=g)], 1))


def random_state(hg, B, g, lo=0.05, hi=2.0):
    return lo + (hi - lo) * torch.rand(B, hg.M, generator=g, dtype=torch.float64)


@pytest.mark.parametrize('bench', BENCHES)
def test_oracle_reproduces_the_reference_rates(bench):
    C.configure(bench)
    m = build_chart(chart.CONFIGS[bench]['oracle'])
    g = torch.Generator().manual_seed(0)
    c = random_state(m.hg, 6, g, hi=2.0 if bench != 'schlogl_nonideal' else 3.0)
    drive = random_condition(bench, C.N_TRAIN, 6, g)
    a, b = m.rates(c, drive)
    at, bt = TR.true_rates(m.hg, drive, c)
    assert float(((a - at).abs() / at).max()) < 1e-10
    assert float(((b - bt).abs() / bt).max()) < 1e-10


@pytest.mark.parametrize('bench', BENCHES)
def test_structure_holds_for_random_parameters(bench):
    """local detailed balance with the model's own force, sigma >= 0 and conserved moieties for any parameter value"""
    C.configure(bench)
    cfg = chart.CONFIGS[bench]['chart']
    m = build_chart(cfg)
    g = torch.Generator().manual_seed(1)
    with torch.no_grad():
        for p in m.parameters():
            p.add_(0.5 * torch.randn(p.shape, generator=g, dtype=p.dtype))
    c = random_state(m.hg, 5, g, hi=3.0 if bench != 'schlogl_nonideal' else 2.5)
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
        assert float((m(c, ctx) @ L).abs().max()) < 1e-9


def _noiseless_schlogl(n_traj=6, steps=30):
    """reference trajectories of the ideal Schloegl model (DOP853) at random conditions"""
    rng = np.random.default_rng(0)
    cs, lab = [], []
    for i in range(n_traj):
        a, b = rng.uniform(2.85, 3.1), rng.uniform(1.0, 3.2)
        x0 = rng.uniform(0.5, 3.0, 64)
        cs.append(TR.solve_schlogl(8, a, b, x0, steps)); lab.append([math.log(a), math.log(b)])
    return dict(c=torch.tensor(np.array(cs)), drive=dict(logab=torch.tensor(lab, dtype=torch.float64)))


def test_closed_form_step_recovers_the_true_constants():
    C.configure('schlogl')
    tr = _noiseless_schlogl()
    m = build_chart(chart.CONFIGS['schlogl']['chart'])
    info = fit_linear(m, dict(train=tr), C.DT, log=lambda *a: None)
    assert info['residual_theta'] < 1e-6
    assert np.allclose(m.kinetics.core.detach().numpy(), true_log_k(), atol=1e-3)


def test_variable_projection_runs():
    C.configure('schlogl')
    tr = _noiseless_schlogl(4, 20)
    C.configure('schlogl_nonideal')
    m = build_chart(chart.CONFIGS['schlogl_nonideal']['chart'])
    params = state_function_parameters(m)
    assert params
    out = fit_varpro(m, dict(train=tr), C.DT, params, max_iters=3, log=lambda *a: None)
    assert out['iters'] == 3 and math.isfinite(out['residual_end'])


@pytest.mark.slow
@pytest.mark.parametrize('bench', BENCHES)
def test_shipped_checkpoint_on_reference_data(bench):
    """the shipped Neural Chart weights on the first trajectories of the training-condition set (needs the data)"""
    C.configure(bench)
    w = os.path.join(CHECKPOINTS, bench, 'chart.pt')
    if not (os.path.exists(w) and os.path.exists(os.path.join(C.DATA, 'train_conditions.pt'))):
        pytest.skip('data or checkpoint not present (python -m benchmarks.<name>.run generate)')
    from neuralchart import fast
    from neuralchart.fit import rollout
    from benchmarks.reaction_networks.data import load_set
    from benchmarks.reaction_networks.models import set_max_sub
    m = build_chart(chart.CONFIGS[bench]['chart'])
    m.load_state_dict(torch.load(w))
    m.eval()
    fast.accelerate(m, control='sample_err', rtol=C.RTOL, precision='fp32', step=C.STEP)
    set_max_sub(m, C.MAXSUB_EVAL)
    sd = load_set('train_conditions')
    Y = sd['c'][:2, :41]
    R, _, _ = rollout(m, {k: v[:2] for k, v in sd['drive'].items()}, Y[:, 0], 40, C.DT)
    ch = slice(0, 2 * C.N_TRAIN ** 2) if bench == 'phosphorylation_cycle' else slice(None)
    e = torch.linalg.vector_norm((R.double() - Y)[..., ch], dim=-1) / torch.linalg.vector_norm(Y[..., ch], dim=-1)
    assert float(e.max()) < 2e-2
    assert float(R.min()) > 0
