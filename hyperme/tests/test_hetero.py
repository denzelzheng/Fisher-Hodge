"""Smoke tests of the heterogeneous-media benchmark H1 (hyperme/bench_hetero.py), CPU, a few seconds."""
import numpy as np
import pytest
import torch
from hyperme import bench_hetero as BH


@pytest.fixture(scope='module')
def tiny():
    torch.manual_seed(0)
    pool = BH.make_pool(11, 2, 2, BH.POR_TRAIN)
    t = BH.pool_tensors(pool, 'cpu')
    t['g'] = BH.true_g(t['blk'], 1.0)
    return pool, t


def drive_of(t, T=1.0, amp=1.0, dtype=torch.float64):
    B = t['U'].shape[0]
    return dict(U=t['U'].reshape(B, -1).to(dtype), T=torch.full((B,), T, dtype=dtype), amp=torch.full((B,), amp, dtype=dtype),
                dpsi=t['dpsi'].to(dtype), psin=t['psin'].to(dtype), blk=t['blk'].to(dtype), pmed=t['pmed'].to(dtype))


def test_medium(tiny):
    pool, t = tiny
    assert pool['pore'].shape == (2, BH.NF, BH.NF)
    assert np.all((pool['pmed'] > 0.25) & (pool['pmed'] < 0.75))
    blk = BH.blocks_of(pool['pore'])
    assert np.allclose(blk[0, 3, 5].reshape(4, 4), pool['pore'][0, 12:16, 20:24])
    assert t['g'].min() >= BH.G_FLOOR and torch.isfinite(t['g']).all()


def test_edge_frame_symmetry(tiny):
    """the shared face is column 3 of both canonical blocks, and g is symmetric under tail <-> head"""
    _, t = tiny
    Xc, Yc, pt, ph = BH.edge_frames(t['blk'])
    assert torch.allclose(BH.g_raw(Xc, Yc, pt, ph), BH.g_raw(Yc, Xc, ph, pt))
    b4 = t['blk'][0].reshape(BH.N, BH.N, 4, 4)
    a, b = 4, 7
    e_h, e_v = a * BH.N + b, BH.N * BH.N + a * BH.N + b
    assert torch.equal(Xc[0, e_h].reshape(4, 4)[:, 3], b4[a, b][:, 3])            # tail right column
    assert torch.equal(Yc[0, e_h].reshape(4, 4)[:, 3], b4[a, b + 1][:, 0])        # head left column
    assert torch.equal(Xc[0, e_v].reshape(4, 4)[:, 3], b4[a, b][3, :])            # tail bottom row
    assert torch.equal(Yc[0, e_v].reshape(4, 4)[:, 3], b4[a + 1, b][0, :])        # head top row


@pytest.mark.parametrize('T,amp', [(1.0, 1.0), (2.0, 4.0)])
def test_truth_in_model_class(tiny, T, amp):
    """the reference right-hand side equals the wired HyperME with the true kappa (oracle kinetics)"""
    _, t = tiny
    torch.set_default_dtype(torch.float64)
    try:
        m = BH.build_wired('oracle', C=1.0).double()
    finally:
        torch.set_default_dtype(torch.float32)
    dr = drive_of(t, T, amp)
    rho = t['rho0']
    B = rho.shape[0]
    Tv = dr['T'].view(-1, 1, 1)
    ref, J = BH.ref_flux(rho, BH.KREF * Tv[..., None] * t['g'].view(B, 2, BH.N, BH.N), t['U'] / Tv,
                         dr['amp'].view(-1, 1, 1, 1) * t['dpsi'] / Tv[..., None], crowd=False)
    out = m(rho.reshape(B, -1), dr)
    assert torch.allclose(out, ref.reshape(B, -1), rtol=1e-10, atol=1e-10)
    assert torch.allclose(m.flux(rho.reshape(B, -1), dr), J.reshape(B, -1), rtol=1e-10, atol=1e-10)


def test_reference_conserves(tiny):
    _, t = tiny
    B = t['U'].shape[0]
    one = torch.ones(B, dtype=torch.float64)
    for crowd in (False, True):
        R, J, ns = BH.simulate(t['rho0'], t['g'], t['U'], t['dpsi'], one, one, crowd, 2)
        assert ns >= 1 and R.min() > 0
        m = R.mean((-1, -2))
        assert (m - m[:, :1]).abs().max() < 1e-12


def test_pair_network_symmetric(tiny):
    _, t = tiny
    torch.manual_seed(1)
    m = BH.build_wired('edge')
    for p in m.kinetics.net.rho.parameters():
        torch.nn.init.normal_(p, std=0.1)
    f = m.featurizer(drive_of(t, dtype=torch.float32))
    k1 = m.kinetics.net(f['xt'], f['xh'])
    k2 = m.kinetics.net(f['xh'], f['xt'])
    assert torch.allclose(k1, k2, atol=1e-6) and k1.std() > 0


@pytest.mark.parametrize('name', ['wired_k', 'wired_const', 'fno', 'gnn_ode', 'flux_gnn'])
def test_models_train_step(tiny, name):
    """every model takes the common drive, runs one training iteration and keeps its interface"""
    _, t = tiny
    from hyperme.train import train, rollout
    meta = dict(C=1.0, norm=dict(U=(0.0, 1.0), psi=(0.0, 1.0), dpsi=1.0))
    torch.manual_seed(0)
    m = BH.build(name, meta)
    dr = drive_of(t, dtype=torch.float32)
    B = dr['U'].shape[0]
    with torch.no_grad():
        c = t['rho0'].reshape(B, -1).float()
        traj = [c]
        for _ in range(BH.T_TRAIN):
            traj.append(c)
    tr = dict(c=torch.stack(traj, 1), drive=dr)
    info = train(m, tr, 1, 0, BH.DT, batch=2, kmin=1, kmax=1, log=lambda *a: None)
    assert info['train_nskip'] == 0
    R = rollout(m, dr, tr['c'][:, 0], 2, BH.DT)[0]
    assert R.shape == (B, 3, BH.N * BH.N) and torch.isfinite(R).all()
    if name.startswith('wired') or name == 'flux_gnn':
        assert R.min() > 0
        assert (R.sum(-1) - R[:, :1].sum(-1)).abs().max() < 1e-2


def test_metrics():
    Y = torch.rand(4, 201, 16) + 0.5
    m = BH.rollout_metrics(Y.clone(), Y)
    assert m['err_10T'] == 0 and m['n_neg_traj'] == 0 and m['mass_drift_max'] == 0
    km = BH.pearson(np.arange(5.0), 2 * np.arange(5.0) + 1)
    assert abs(km - 1) < 1e-12
