"""Smoke tests of the heterogeneous-media benchmark H1 (hyperme/bench_hetero.py), CPU."""
import os
import numpy as np
import pytest
import torch
from hyperme import bench_hetero as BH


@pytest.fixture(scope='module')
def tiny():
    pool = BH.make_pool(11, 2, 2, BH.POR_TRAIN)
    chi = torch.as_tensor(pool['pore'], dtype=torch.float64)
    Mpix = BH.truth_mobility_raw(chi) + BH.M_FLOOR
    meta = dict(desc={'p': BH.descriptor_fields(chi).float()}, Mpix={'p': Mpix.float()},
                pools={'p': dict(pmed=pool['pmed'].tolist())})
    discs = {g: BH.Disc(g, meta, 'cpu') for g in ('g32', 'cloud')}
    return pool, chi, Mpix, meta, discs


def drive_on(pool, disc, T=1.0, amp=1.0):
    B = len(pool['med'])
    Up, Pp = torch.as_tensor(pool['Upar']), torch.as_tensor(pool['Ppar'])
    med = torch.as_tensor(pool['med'])
    return dict(U=BH.gauss_eval(Up, disc.g['pts']).float(), psin=BH.gauss_eval(Pp, disc.g['pts']).float(),
                line=BH.line_integrals(Pp, disc.g).float(), T=torch.full((B,), T), amp=torch.full((B,), amp), Ppar=Pp,
                ein=disc.ein['p'][med], ndesc=disc.ndesc['p'][med], pmed=torch.as_tensor(pool['pmed'], dtype=torch.float32)[med],
                mtrue=disc.mtrue['p'][med])


def test_geometry(tiny):
    g = BH.grid_geom(32)
    assert torch.allclose(g['w'], torch.full_like(g['w'], 1024.0))
    c = tiny[4]['cloud'].g
    assert abs(float(c['area'].sum()) - 1) < 1e-9 and (c['w'] > 0).all()
    deg = torch.bincount(c['edges'].reshape(-1), minlength=BH.CLOUD_N)
    assert deg.min() >= 3 and 2 * c['edges'].shape[0] == int(deg.sum())
    assert float((c['dvec'].norm(dim=-1)).max()) < 0.2
    assert float(((c['area_raster'] - c['area']).abs() / c['area']).median()) < 0.05


def test_projection_conserves_mass(tiny):
    r = torch.rand(3, BH.NF, BH.NF, dtype=torch.float64) + 0.1
    for g in (BH.grid_geom(32), BH.grid_geom(64), tiny[4]['cloud'].g):
        c = BH.projector(g)(r)
        mass = c.sum(-1) / c.shape[-1]                         # states are mass / mean cell volume
        assert torch.allclose(mass, r.mean((-1, -2)), rtol=1e-12)


def test_descriptors(tiny):
    _, chi, Mpix, meta, discs = tiny
    for d in discs.values():
        x = d.ein['p']
        assert x.shape == (2, d.H, 12) and x.min() >= -1 - 1e-6 and x.max() <= 1 + 1e-6
        assert (d.mtrue['p'] > 0).all()


def test_fine_truth_in_model_class(tiny):
    """on the 128 x 128 reference grid the wired HyperME with the true face mobilities is the reference operator"""
    pool, chi, Mpix, _, _ = tiny
    torch.set_default_dtype(torch.float64)
    try:
        g = BH.grid_geom(BH.NF)
        disc = BH.Disc('g128', None, 'cpu', geom=g)
        m = BH.build_wired('oracle', disc).double()
    finally:
        torch.set_default_dtype(torch.float32)
    U, dpsi, r0 = BH.fine_inputs(pool, 'cpu')
    Me = BH.fine_edge_mobility(Mpix)[torch.as_tensor(pool['med'])]
    B = U.shape[0]
    T, amp = 2.0, 3.0
    ref, J = BH.ref_flux(r0, BH.D * BH.NF ** 2 * T * Me, U / T, amp * dpsi / T, crowd=False)
    F = dict(U=U.reshape(B, -1), T=torch.full((B,), T, dtype=torch.float64), amp=torch.full((B,), amp, dtype=torch.float64),
             line=dpsi.reshape(B, -1), ein=torch.zeros(B, disc.H, 12, dtype=torch.float64), pmed=torch.zeros(B, dtype=torch.float64),
             mtrue=Me.reshape(B, -1))
    out = m(r0.reshape(B, -1), F)
    assert torch.allclose(out, ref.reshape(B, -1), rtol=1e-9, atol=1e-9)


def test_reference_conserves(tiny):
    pool, chi, Mpix, _, discs = tiny
    U, dpsi, r0 = BH.fine_inputs(pool, 'cpu')
    Me = BH.fine_edge_mobility(Mpix)[torch.as_tensor(pool['med'])]
    one = torch.ones(U.shape[0], dtype=torch.float64)
    for crowd in (False, True):
        Rs, J, rf, ns = BH.simulate(r0, Me, U, dpsi, one, one, crowd, 1, [BH.projector(discs['g32'].g)])
        assert ns >= 1 and rf.min() > 0
        assert (rf.mean((-1, -2)) - r0.mean((-1, -2))).abs().max() < 1e-12
        assert torch.allclose(Rs[0].mean(-1), r0.mean((-1, -2))[:, None])
        # coarse face fluxes reproduce the coarse rate of change
        _, Jf = BH.ref_flux(rf, BH.D * BH.NF ** 2 * Me, U, dpsi, crowd)
        dfine = BH.div_T(Jf)
        dc = dfine.reshape(-1, 32, 4, 32, 4).mean((2, 4))
        Jc = BH.coarse_flux32(Jf).reshape(-1, 2, 32, 32)
        assert torch.allclose(BH.div_T(Jc), dc, atol=1e-9)


@pytest.mark.parametrize('name', ['wired_k', 'wired_const', 'ude', 'fno', 'gnn_ode', 'flux_gnn', 'lm_hodge'])
def test_models_train_and_transfer(tiny, name):
    """one training iteration on the 32 x 32 grid, then the same parameters on the point cloud"""
    if name == 'lm_hodge' and 'LMH_PKG' not in os.environ:
        pytest.skip('LMH_PKG not set')
    pool, _, _, meta, discs = tiny
    from hyperme.train import train, rollout
    meta = dict(meta, norm=dict(U=(0.0, 1.0), psi=(0.0, 1.0), line=0.1))
    torch.manual_seed(0)
    g32 = discs['g32']
    m = BH.build(name, meta, g32)
    dr = drive_on(pool, g32)
    if name == 'lm_hodge':
        m.set_disc(g32, 'cpu')
        sd = dict(drive=dr, pool='p', med=torch.as_tensor(pool['med']))
        BH.lm_drive_extras(sd, g32, meta, 'cpu')
    B = dr['U'].shape[0]
    c0 = BH.projector(g32.g)(BH.fine_inputs(pool, 'cpu')[2]).float()
    tr = dict(c=c0[:, None].expand(B, BH.T_TRAIN + 1, -1).contiguous(), drive=dr)
    info = train(m, tr, 1, 0, BH.DT, batch=2, kmin=1, kmax=1, log=lambda *a: None)
    assert info['train_nskip'] == 0
    R = rollout(m, dr, c0, 2, BH.DT)[0]
    assert torch.isfinite(R).all()
    if name.startswith('wired'):
        ctx = m.prepare(dr)
        ph = BH.phys_metrics_rollout(R, dr, g32, lambda c: m.flux(c, ctx)).result()
        assert ph['ldb_violation_frac'] == 0 and ph['sigma_neg_frac'] == 0
    cl = discs['cloud']
    m2 = BH.rebind(name, m, meta, cl, 'cpu')
    if name == 'fno':
        assert m2 is None
        return
    d2 = drive_on(pool, cl)
    if name == 'lm_hodge':
        sd = dict(drive=d2, pool='p', med=torch.as_tensor(pool['med']))
        BH.lm_drive_extras(sd, cl, meta, 'cpu')
    c2 = BH.projector(cl.g)(BH.fine_inputs(pool, 'cpu')[2]).float()
    R2 = rollout(m2, d2, c2, 1, BH.DT)[0]
    assert R2.shape == (B, 2, BH.CLOUD_N) and torch.isfinite(R2).all()
    if name.startswith('wired') or name == 'flux_gnn':
        assert R2.min() > 0 and (R2.sum(-1) - R2[:, :1].sum(-1)).abs().max() < 1e-2 * R2[:, 0].sum(-1).max()


def test_metrics():
    Y = (torch.rand(4, 1, 16) + 0.5).expand(4, 201, 16).contiguous()
    m = BH.rollout_metrics(Y.clone(), Y)
    assert m['err_10T'] == 0 and m['n_neg_traj'] == 0 and m['mass_drift_max'] == 0
    assert abs(BH.pearson(np.arange(5.0), 2 * np.arange(5.0) + 1) - 1) < 1e-12


def test_minnorm_flux(tiny):
    for d in tiny[4].values():
        cdot = torch.randn(3, d.V, dtype=torch.float64)
        cdot = cdot - cdot.mean(-1, keepdim=True)
        J = d.minnorm_flux(cdot)
        S = torch.zeros(3, d.V, dtype=torch.float64).index_add_(1, d.g['edges'][:, 1], J).index_add_(1, d.g['edges'][:, 0], -J)
        assert torch.allclose(S, cdot, atol=1e-8)
