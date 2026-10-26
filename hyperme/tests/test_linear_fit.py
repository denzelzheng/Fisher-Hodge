"""The closed-form linear step (hyperme.linear_fit) and the zero-gated residuals.  The wired rates are linear in the
one-way rate constants, so non-negative least squares on trajectory data recovers the generating constants from a
neutral start; the gated residuals leave the model unchanged at initialisation and only their gates get a gradient."""
import numpy as np
import pytest
import torch
from hyperme import presets, Hypergraph, HyperME
from hyperme.model import Context, site_grad
from hyperme.linear_fit import fit_linear, class_structure

DT = 0.02          # sampling interval of the synthetic data (the Simpson rule is accurate for DT * rate << 1)


@torch.no_grad()
def make_data(model, B=8, T1=21, seed=0, lo=0.3, hi=2.5, drive=None):
    g = torch.Generator().manual_seed(seed)
    c = torch.rand(B, model.hg.M, generator=g, dtype=torch.float64) * (hi - lo) + lo
    if drive is None:
        drive = dict(logab=torch.stack([torch.empty(B, dtype=torch.float64).uniform_(1.0, 1.2, generator=g),
                                        torch.empty(B, dtype=torch.float64).uniform_(-0.5, 1.0, generator=g)], -1))
    ctx = model.prepare(drive)
    out = [c]
    for _ in range(T1 - 1):
        c, _ = model.advance(c, ctx, DT, nsub=32)
        out.append(c)
    return dict(train=dict(c=torch.stack(out, 1), drive=drive))


def schlogl_truth(n=4, nonideal_residual=False):
    m = presets.schlogl_wired(n=n, nonideal_residual=nonideal_residual)
    assert presets.set_true_mass_action(m, **presets.C2_TRUE) < 1e-10
    return m


def rate_err(m, truth, data):
    tr = data['train']
    c = tr['c'][:, 5]
    a, b = m.rates(c, tr['drive']); at, bt = truth.rates(c, tr['drive'])
    return float(torch.cat([(a / at - 1).abs(), (b / bt - 1).abs()]).max())


def test_set_true_mass_action_is_exact():
    m = schlogl_truth()
    g = torch.Generator().manual_seed(3)
    lab = torch.randn(4, 2, generator=g, dtype=torch.float64) * 0.5
    c = torch.rand(4, m.hg.M, generator=g, dtype=torch.float64) + 0.1
    a, b = m.rates(c, dict(logab=lab))
    at, bt = presets.mass_action_rates(m.hg, lab, c, **presets.C2_TRUE)
    assert torch.allclose(a, at, rtol=1e-11) and torch.allclose(b, bt, rtol=1e-11)


def test_class_structure_of_schlogl():
    """5 classes (R1 and R2 in each direction, both directions of transport) and full rank in theta."""
    m = presets.schlogl_wired(n=4)
    c = torch.rand(1, m.hg.M, dtype=torch.float64) + 0.5
    params, U, rp, rm = class_structure(m, c, dict(logab=torch.zeros(1, 2, dtype=torch.float64)))
    assert U.shape[0] == 5 and np.linalg.matrix_rank(U) == 5
    tr = m.hg.meta['typ'] == 2
    assert (rp[tr] == rm[tr]).all() and len(set(rp[tr])) == 1


@pytest.mark.parametrize('kind', ['wired', 'crnn'])
def test_fit_linear_recovers_schlogl_from_neutral_start(kind):
    truth = schlogl_truth()
    data = make_data(truth)
    m = presets.schlogl_wired(n=4) if kind == 'wired' else presets.crnn_chem(presets.schlogl_hg(4))
    e0 = rate_err(m, truth, data)
    info = fit_linear(m, data, DT, log=lambda *a: None)
    e1 = rate_err(m, truth, data)
    assert e0 > 0.5 and e1 < 1e-2, (e0, e1)
    assert info['residual_theta'] < 1e-4 and info['projection_err'] < 1e-8


def test_fit_linear_per_edge_core():
    """transport with one unknown rate per edge (core='edge') and a random potential: every edge rate is recovered."""
    n = 4
    hg = Hypergraph.grid_transport(n)
    with presets.default_dtype(torch.float64):
        truth = HyperME.build_wired(hg, kinetics_core='edge', kinetics_table=True, T_exponent=0.0,
                                    step=dict(method='mprk22', nsub=8)).double()
        m = HyperME.build_wired(hg, kinetics_core='edge', kinetics_table=True, T_exponent=0.0,
                                step=dict(method='mprk22', nsub=8)).double()
    g = torch.Generator().manual_seed(1)
    with torch.no_grad():
        truth.kinetics.core.copy_(torch.randn(hg.H, generator=g, dtype=torch.float64) * 0.5 - 1.0)
    B = 8
    drive = dict(U=torch.randn(B, hg.V, generator=g, dtype=torch.float64))
    data = make_data(truth, B=B, drive=drive, lo=0.2, hi=2.0)
    fit_linear(m, data, DT, log=lambda *a: None)
    err = (m.kinetics.core - truth.kinetics.core).abs().max()
    assert err < 1e-2, float(err)


def test_gated_residuals_start_as_the_core():
    """at initialisation the gated dmu and log k residuals are switched off; a loss gives the gates a gradient and the
    residual weights none."""
    m = presets.schlogl_wired(n=4)
    fe = m.free_energy
    c = torch.rand(3, m.hg.M, dtype=torch.float64) + 0.2
    assert fe.correction(c, None).abs().max() == 0
    assert fe.residual_mu(c.view(3, 1, -1)).abs().min() > 0          # the residual itself is generic, not zero
    lab = torch.zeros(3, 2, dtype=torch.float64)
    m(c, dict(logab=lab)).pow(2).sum().backward()
    assert fe.s_mu.grad.abs().max() > 0
    for p in (fe.r_alpha, fe.r_beta, fe.r_delta):
        assert p.grad is None or p.grad.abs().max() == 0
    # log k residual on a grid with a kinetic feature (S2 wired: edge orientation)
    w = presets.s2_wired(n=8)
    kin = w.kinetics
    drive = dict(U=torch.randn(2, 8, 8), dpsi=torch.randn(2, 2, 8, 8), psin=torch.zeros(2, 8, 8))
    rho = torch.rand(2, 64) + 0.5
    w(rho, drive).pow(2).sum().backward()
    assert kin.s_k.grad.abs() > 0
    assert all(p.grad is None or p.grad.abs().max() == 0 for p in kin.residual.parameters())


def test_dmu_residual_is_the_gradient_of_a_convex_excess_free_energy():
    m = presets.schlogl_wired(n=4)
    fe = m.free_energy
    g = torch.Generator().manual_seed(2)
    with torch.no_grad():
        for p in (fe.r_alpha, fe.r_beta, fe.r_delta):
            p.add_(torch.randn(p.shape, generator=g, dtype=p.dtype) * 0.5)
        fe.s_mu.fill_(0.7)
    c = torch.rand(4, m.hg.M, generator=g, dtype=torch.float64) + 0.1
    ctx = Context(Ufull=torch.zeros_like(c))
    grad = site_grad(lambda x: fe.value(x, ctx) - (x * (torch.log(x) - 1)).sum(-1), c)
    assert torch.allclose(fe.correction(c, ctx), grad, atol=1e-10)
    x = torch.linspace(0.05, 4.0, 50, dtype=torch.float64)[:, None].expand(-1, m.hg.M).contiguous()
    d = fe.correction(x, ctx)[:, 0]
    assert (d[1:] - d[:-1] > 0).all()                                   # monotone: convex excess free energy


def test_old_kinetics_layouts_load():
    """checkpoints with kappa = mlp(features) or a bias-free table over the type one-hot load into core + gated residual"""
    m = presets.c3_wired()
    sd = {k: v for k, v in m.state_dict().items() if not k.startswith('kinetics.core')}
    lk = torch.arange(6, dtype=torch.float64)[None] * 0.1
    sd['kinetics.mlp.net.0.weight'] = lk
    m.load_state_dict(sd)
    assert torch.allclose(m.kinetics.core, lk[0])
    w = presets.s2_wired(n=8)
    ref = presets.s2_wired(n=8)
    old = {}
    for k, v in ref.state_dict().items():
        if k.startswith('kinetics.residual.'):
            old['kinetics.mlp.' + k[len('kinetics.residual.'):]] = torch.randn_like(v)
        elif k not in ('kinetics.s_k', 'kinetics.core'):
            old[k] = v
    w.load_state_dict(old)
    feats = w.featurizer(dict(U=torch.zeros(1, 8, 8), dpsi=torch.zeros(1, 2, 8, 8), psin=torch.zeros(1, 8, 8)))
    assert float(w.kinetics.s_k) == 1.0 and torch.allclose(w.kinetics.log_k(feats), w.kinetics.residual(feats['edge_sym']).squeeze(-1))
