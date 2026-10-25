"""The five structural guarantees, checked on random (perturbed) models on four hypergraphs:
grid transport with a non-local free energy, the C1 reaction-transport hypergraph, a well-mixed network with a
stoichiometric coefficient 2, and the TC-ME grid preset (split kinetics, direct free energy, curl drive)."""
import numpy as np
import pytest
import torch
from hyperme import Hypergraph
from hyperme.complex import grid_curl_basis
from helpers import case, CASES

NSTEP = 40
DT = 0.05


def rollout(m, drive, c0, n=NSTEP):
    ctx = m.prepare(drive)
    c = c0.clone(); out = [c]
    with torch.no_grad():
        for _ in range(n):
            c, _ = m.advance(c, ctx, DT)
            out.append(c)
    return torch.stack(out, 1), ctx


@pytest.mark.parametrize('name', CASES)
def test_conservation(name):
    m, drive, c0 = case(name)
    R, _ = rollout(m, drive, c0)
    L = m.hg.on(R.device, R.dtype)['L']
    assert L.shape[1] >= 1
    q = R @ L
    err = (torch.linalg.vector_norm(q - q[:, :1], dim=-1) / torch.linalg.vector_norm(q[:, :1], dim=-1)).max()
    assert err < 1e-13, err


@pytest.mark.parametrize('name', CASES)
def test_positivity(name):
    m, drive, c0 = case(name, seed=1)
    c0 = c0.clone()
    c0[:, ::7] = 1e-8                                   # nearly empty components next to full ones
    R, _ = rollout(m, drive, c0)
    assert torch.isfinite(R).all()
    assert R.min() > 0


@pytest.mark.parametrize('name', CASES)
def test_sigma_nonneg(name):
    m, drive, c0 = case(name, seed=2)
    R, ctx = rollout(m, drive, c0, n=10)
    for t in (0, 5, 10):
        out = m.readouts(R[:, t], ctx)
        assert (out['sigma_edge'] >= 0).all()
        assert (out['sigma'] > 0).all()
        # local detailed balance and the Pythagorean split of sigma
        f_pred = -m.hg.apply_ST(out['mu']) + out['A']
        assert torch.allclose(out['f'], f_pred, atol=1e-10, rtol=1e-10)
        assert torch.allclose(out['excess'] + out['housekeeping'], out['sigma'], rtol=1e-8)
        assert torch.allclose(out['J'], out['ell'] * out['f'], rtol=1e-8, atol=1e-12)


@pytest.mark.parametrize('name', CASES)
def test_free_energy_monotone_without_drive(name):
    m, drive, c0 = case(name, seed=3, reversible=True)
    R, ctx = rollout(m, drive, c0)
    with torch.no_grad():
        F = torch.stack([m.free_energy_value(R[:, t], ctx) for t in range(R.shape[1])], 1)
        dF = F[:, 1:] - F[:, :-1]
        assert (dF <= 1e-10 * (1 + F[:, :-1].abs())).all(), dF.max()
        assert (dF < 0).any()
        out = m.readouts(R[:, 5], ctx)
        assert torch.allclose(out['dFdt'], -out['sigma'], rtol=1e-8)
        assert out['excess'].min() > 0 and (out['housekeeping'] <= 1e-10 * out['sigma']).all()


@pytest.mark.parametrize('name', CASES)
def test_drive_in_cycle_space(name):
    m, drive, c0 = case(name, seed=4)
    with torch.no_grad():
        ctx = m.prepare(drive)
    A = ctx['A']
    assert A.abs().max() > 1e-3
    SA = m.hg.apply_S(A)
    assert SA.abs().max() < 1e-10 * max(1.0, float(A.abs().max()))
    out = m.readouts(c0, ctx, hodge=False)
    Z = m.hg.on(A.device, A.dtype)['Z']
    assert torch.allclose(out['cycle_affinity'], A @ Z, atol=1e-10)


def test_curl_basis_equals_null_space():
    n = 5
    hg_c = Hypergraph.grid_transport(n, basis='curl')
    hg_s = Hypergraph.grid_transport(n, basis='svd')
    Zc, Zs = hg_c.Z, hg_s.Z
    assert Zc.shape == Zs.shape == (2 * n * n, n * n + 1)
    assert np.abs(Zc @ Zc.T - Zs @ Zs.T).max() < 1e-10
    assert np.abs(hg_c.S @ grid_curl_basis(n)).max() < 1e-12


def test_moieties_of_c1():
    hg = Hypergraph.crn_phosphorylation_grid(8)
    assert hg.L.shape[1] == 65 and hg.Z.shape[1] == 257          # E at every site + total X + Xp; beta_1 = 257
