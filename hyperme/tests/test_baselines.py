import os
import numpy as np
import pytest
import torch
from hyperme import Hypergraph, baselines as BL, presets
from hyperme.model import weighted_gradient_part
from helpers import case
from conftest import load_legacy


def c1_setup(B=3, seed=0):
    m, drive, c0 = case('c1', B=B, seed=seed)
    feat = m.featurizer
    nn_in, ne_in = BL._Base.n_node_in(m.hg, feat, drive)
    return m.hg, feat, drive, c0, nn_in, ne_in


def moieties_drift(hg, R):
    L = hg.on(R.device, R.dtype)['L']
    q = R @ L
    return float(((q - q[:, :1]).norm(dim=-1) / q[:, :1].norm(dim=-1)).max())


def roll(model, drive, c0, n=10):
    ctx = model.prepare(drive)
    out = [c0]
    c = c0
    with torch.no_grad():
        for _ in range(n):
            c, _ = model.advance(c, ctx, 0.05)
            out.append(c)
    return torch.stack(out, 1)


@pytest.mark.parametrize('kind', ['flux_gnn', 'crnn', 'gnn_ode', 'finn', 'jko_rev'])
def test_baseline_interface(kind):
    hg, feat, drive, c0, nn_in, ne_in = c1_setup()
    torch.manual_seed(0)
    with presets.default_dtype(torch.float64):
        model = dict(flux_gnn=lambda: BL.FluxGNN(hg, nn_in, ne_in, feat, hidden=16),
                     crnn=lambda: BL.CRNN(hg, ne_in, feat, hidden=16),
                     gnn_ode=lambda: BL.GNNODE(hg, nn_in, ne_in, feat, hidden=16),
                     finn=lambda: BL.FINN(hg, ne_in, feat, hidden=16),
                     jko_rev=lambda: BL.JKOReversible(hg, ('U',), feat, hidden=16))[kind]().double()
    with torch.no_grad():
        for p in model.parameters():                    # break the zero init so the dynamics is non-trivial
            p.add_(0.05 * torch.randn_like(p))
    R = roll(model, drive, c0)
    assert R.shape == (c0.shape[0], 11, hg.M) and torch.isfinite(R).all()
    if kind in ('flux_gnn', 'crnn', 'jko_rev'):
        assert moieties_drift(hg, R) < 1e-12               # S J form
    if kind in ('flux_gnn', 'jko_rev'):
        assert R.min() > 0
    if kind == 'jko_rev':
        loss = model.jko_loss(R[:, 0], R[:, 1], 0.05, drive)
        loss.backward()
        assert torch.isfinite(loss) and model.beta_raw.grad is not None


def test_finn_conserves_transport_mass():
    n = 6
    hg = Hypergraph.grid_transport(n)
    g = torch.Generator().manual_seed(0)
    with presets.default_dtype(torch.float64):
        from hyperme.model import default_featurizer
        feat = default_featurizer(hg)
        drive = dict(U=torch.randn(2, n * n, generator=g, dtype=torch.float64),
                     force=torch.randn(2, hg.H, generator=g, dtype=torch.float64))
        nn_in, ne_in = BL._Base.n_node_in(hg, feat, drive)
        model = BL.FINN(hg, ne_in, feat, hidden=16).double()
    R = roll(model, drive, torch.rand(2, n * n, generator=g, dtype=torch.float64) + 0.5)
    assert torch.allclose(R.sum(-1), R[:, :1].sum(-1).expand(-1, R.shape[1]), rtol=1e-12)


def test_weighted_hodge_matches_hodge_lib():
    HL = load_legacy('hodge_lib', 'analysis/hodge_lib.py')
    n = 5
    hg = Hypergraph.grid_transport(n)
    rng = np.random.default_rng(0)
    f = rng.normal(size=hg.H); w = rng.uniform(0.2, 2.0, size=hg.H)
    E = np.stack([hg.meta['et'], hg.meta['eh']], 1)
    g_ref, _ = HL.hodge(f, w, HL.incidence(E, n * n))
    g = weighted_gradient_part(hg, torch.tensor(f)[None], torch.tensor(w)[None])[0].numpy()
    assert np.abs(g - g_ref).max() < 1e-10
