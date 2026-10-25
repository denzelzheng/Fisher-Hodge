import numpy as np
import pytest
import torch
from hyperme import coarse as C
from test_integrators import linear_model


@pytest.mark.parametrize('reversible', [False, True])
def test_projected_generator(reversible):
    m, ctx, c0 = linear_model(reversible, n=8, seed=1)
    labels = C.grid_block_labels(8, 2)
    res = C.coarse_grain_model(m, ctx, c0, labels, method='generator')       # asserts pibar = P pi, Jbar = P J, sigmabar <= sigma
    for r in res:
        assert r['sigmabar'] <= r['sigma']
        p = r['params']
        Qb = r['Qbar']
        i, j = p['edges'][:, 0], p['edges'][:, 1]
        dphi = p['phibar'][i] - p['phibar'][j]
        assert np.allclose(p['k'] * np.exp(0.5 * (dphi + p['A'])), Qb[i, j], rtol=1e-10)
        assert np.allclose(p['k'] * np.exp(-0.5 * (dphi + p['A'])), Qb[j, i], rtol=1e-10)
        if reversible:
            assert np.abs(p['A']).max() < 1e-9 and r['sigma'] < 1e-12       # detailed balance survives lumping
        else:
            assert r['sigma'] > 1e-6
    # 8 -> 4 -> 2 chain: sigma decreases at every level
    r1 = res[0]
    r2 = C.projected_generator(r1['Qbar'], r1['pibar'], C.grid_block_labels(4, 2))
    assert r2['sigmabar'] <= r1['sigmabar'] <= r1['sigma']


def test_lagged_propagator():
    m, ctx, c0 = linear_model(False, n=8, seed=2)
    labels = C.grid_block_labels(8, 2)
    for tau in (0.05, 0.5):
        res = C.coarse_grain_model(m, ctx, c0, labels, method='lag', tau=tau)
        for r in res:
            assert r['sigmabar'] <= r['sigma']
            assert np.allclose(r['Kbar'].sum(1), 1.0)


def test_assertion_fires_on_wrong_weights():
    m, ctx, c0 = linear_model(False, n=4, seed=3)
    Q = m.generator(c0, ctx)[0].T.double().detach().numpy()
    pi = C.stationary(Q)
    with pytest.raises(AssertionError):
        C.projected_generator(Q, np.full_like(pi, 1 / len(pi)), C.grid_block_labels(4, 2))
