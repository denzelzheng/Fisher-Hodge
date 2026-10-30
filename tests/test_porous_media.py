"""Tests of the porous-media benchmark (CPU, seconds): geometry, reference solver, model class, transfer between
discretisations, physics guarantees, and a regression check of the shipped Neural Chart on the stored data."""
import json
import os

import numpy as np
import pytest
import torch

from neuralchart.chart import chart_config
from neuralchart.fit import train, rollout
from benchmarks.common import CHECKPOINTS, RESULTS
from benchmarks.porous_media import data as PD
from benchmarks.porous_media import models as PM
from benchmarks.porous_media import evaluate as PE


@pytest.fixture(scope='module')
def tiny():
    pool = PD.make_pool(11, 2, 2, PD.POR_TRAIN)
    chi = torch.as_tensor(pool['pore'], dtype=torch.float64)
    Mpix = PD.truth_mobility_raw(chi) + PD.M_FLOOR
    meta = dict(desc={'p': PD.descriptor_fields(chi).float()}, Mpix={'p': Mpix.float()},
                pools={'p': dict(pmed=pool['pmed'].tolist())}, crowd=False, RHO_S=PD.RHO_S)
    discs = {g: PD.Disc(g, meta, 'cpu') for g in ('g32', 'cloud')}
    return pool, chi, Mpix, meta, discs


def drive_on(pool, disc, T=1.0, amp=1.0):
    B = len(pool['med'])
    Up, Pp = torch.as_tensor(pool['Upar']), torch.as_tensor(pool['Ppar'])
    med = torch.as_tensor(pool['med'])
    return dict(U=PD.gauss_eval(Up, disc.g['pts']).float(), psin=PD.gauss_eval(Pp, disc.g['pts']).float(),
                line=PD.face_drive(Pp, disc.g).float(), T=torch.full((B,), T), amp=torch.full((B,), amp), Ppar=Pp,
                ein=disc.ein['p'][med], ndesc=disc.ndesc['p'][med], pmed=torch.as_tensor(pool['pmed'], dtype=torch.float32)[med],
                mtrue=disc.mtrue['p'][med])


def fine_disc():
    return PD.Disc('g128', None, 'cpu', geom=PD.grid_geom(PD.NF))


def test_geometry(tiny):
    g = PD.grid_geom(32)
    assert torch.allclose(g['w'], torch.full_like(g['w'], 1024.0))
    E = g['edges']                                            # horizontal edges first, then vertical edges
    assert int(E[0, 0]) == 0 and int(E[0, 1]) == 1 and int(E[1024, 0]) == 0 and int(E[1024, 1]) == 32
    c = tiny[4]['cloud'].g
    assert abs(float(c['area'].sum()) - 1) < 1e-9 and (c['w'] > 0).all()
    deg = torch.bincount(c['edges'].reshape(-1), minlength=PD.CLOUD_N)
    assert deg.min() >= 3 and 2 * c['edges'].shape[0] == int(deg.sum())
    assert float(((c['area_raster'] - c['area']).abs() / c['area']).median()) < 0.05


def test_projection_conserves_mass(tiny):
    r = torch.rand(3, PD.NF, PD.NF, dtype=torch.float64) + 0.1
    for g in (PD.grid_geom(32), PD.grid_geom(64), tiny[4]['cloud'].g):
        c = PD.projector(g)(r)
        assert torch.allclose(c.sum(-1) / c.shape[-1], r.mean((-1, -2)), rtol=1e-12)


def test_descriptors(tiny):
    for d in tiny[4].values():
        x = d.ein['p']
        assert x.shape == (2, d.H, 12) and x.min() >= -1 - 1e-6 and x.max() <= 1 + 1e-6
        assert (d.mtrue['p'] > 0).all()


def _fine_inputs(pool, Mpix, T, amp):
    U, dpsi, r0 = PD.fine_inputs(pool, 'cpu')
    Me = PD.fine_edge_mobility(Mpix)[torch.as_tensor(pool['med'])]
    B = U.shape[0]
    F = dict(U=U.reshape(B, -1), T=torch.full((B,), T, dtype=torch.float64), amp=torch.full((B,), amp, dtype=torch.float64),
             line=dpsi.reshape(B, -1), ein=torch.zeros(B, PD.NF ** 2 * 2, 12, dtype=torch.float64),
             pmed=torch.zeros(B, dtype=torch.float64), mtrue=Me.reshape(B, -1))
    return U, dpsi, r0, Me, F


def test_reference_operator_is_in_the_model_class(tiny):
    """on the 128 x 128 reference grid the master equation with the true face mobilities is the reference operator"""
    pool, chi, Mpix, _, _ = tiny
    torch.set_default_dtype(torch.float64)
    try:
        m = PM.build_master('true', fine_disc()).double()
    finally:
        torch.set_default_dtype(torch.float32)
    T, amp = 2.0, 3.0
    U, dpsi, r0, Me, F = _fine_inputs(pool, Mpix, T, amp)
    ref, _ = PD.ref_flux(r0, PD.D * PD.NF ** 2 * T * Me, U / T, amp * dpsi / T, crowd=False)
    B = U.shape[0]
    assert torch.allclose(m(r0.reshape(B, -1), F), ref.reshape(B, -1), rtol=1e-9, atol=1e-9)


def test_crowding_oracle_matches_reference(tiny):
    """with crowding the oracle (true mobility times 1 / (1 + rhobar / rho_s), no parameters) is the reference operator"""
    pool, chi, Mpix, _, _ = tiny
    meta = dict(crowd=True, RHO_S=PD.RHO_S)
    torch.set_default_dtype(torch.float64)
    try:
        m = PM.build_chart(chart_config('porous_media', 'oracle'), meta, fine_disc()).double()
        m0 = PM.build_chart(chart_config('porous_media', 'oracle-no-crowding'), meta, fine_disc()).double()
    finally:
        torch.set_default_dtype(torch.float32)
    assert m.kinetics.mode == 'true_crowding' and m0.kinetics.mode == 'true'
    T, amp = 2.0, 3.0
    U, dpsi, _, Me, F = _fine_inputs(pool, Mpix, T, amp)
    B = U.shape[0]
    r = torch.exp(1.5 * torch.randn(B, PD.NF, PD.NF, generator=torch.Generator().manual_seed(0), dtype=torch.float64))
    with torch.no_grad():
        out = m(r.reshape(B, -1), F)
    ref, _ = PD.ref_flux(r, PD.D * PD.NF ** 2 * T * Me, U / T, amp * dpsi / T, crowd=True)
    assert torch.allclose(out, ref.reshape(B, -1), rtol=1e-9, atol=1e-9)


def test_reference_conserves(tiny):
    pool, chi, Mpix, _, discs = tiny
    U, dpsi, r0 = PD.fine_inputs(pool, 'cpu')
    Me = PD.fine_edge_mobility(Mpix)[torch.as_tensor(pool['med'])]
    one = torch.ones(U.shape[0], dtype=torch.float64)
    for crowd in (False, True):
        Rs, J, rf, ns = PD.simulate(r0, Me, U, dpsi, one, one, crowd, 1, [PD.projector(discs['g32'].g)])
        assert ns >= 1 and rf.min() > 0
        assert (rf.mean((-1, -2)) - r0.mean((-1, -2))).abs().max() < 1e-12


def test_face_drive_is_divergence_free(tiny):
    Pp = torch.as_tensor(tiny[0]['Ppar'])
    cl = tiny[4]['cloud'].g
    A = PD.face_drive(Pp, cl) * cl['sig'] / cl['dvec'].norm(dim=-1)
    S = torch.zeros(A.shape[0], cl['pts'].shape[0], dtype=torch.float64).index_add_(1, cl['edges'][:, 1], A).index_add_(1, cl['edges'][:, 0], -A)
    assert S.abs().max() < 1e-10


@pytest.mark.parametrize('name', ['chart', 'chart-no-state', 'ude', 'fno', 'flux_gnn'])
def test_models_train_and_transfer(tiny, name):
    """one training step on the 32 x 32 grid, then the same parameters on the point cloud"""
    pool, _, _, meta, discs = tiny
    meta = dict(meta, norm=dict(U=(0.0, 1.0), psi=(0.0, 1.0), line=0.1))
    torch.manual_seed(0)
    g32 = discs['g32']
    m = PM.build(name, meta, g32)
    dr = drive_on(pool, g32)
    B = dr['U'].shape[0]
    c0 = PD.projector(g32.g)(PD.fine_inputs(pool, 'cpu')[2]).float()
    tr = dict(c=c0[:, None].expand(B, PD.T_TRAIN + 1, -1).contiguous(), drive=dr)
    info = train(m, tr, 1, 0, PD.DT, batch=2, kmin=1, kmax=1, log=lambda *a: None)
    assert info['train_nskip'] == 0
    R = rollout(m, dr, c0, 2, PD.DT)[0]
    assert torch.isfinite(R).all()
    m2 = PM.rebind(name, m, meta, discs['cloud'], 'cpu')
    if name == 'fno':
        assert m2 is None
        return
    p1 = {k: v for k, v in m.state_dict().items() if not any(b in k for b in PM.GRAPH_BUFFERS)}
    p2 = m2.state_dict()
    assert all(torch.equal(v, p2[k]) for k, v in p1.items())
    d2 = drive_on(pool, discs['cloud'])
    c2 = PD.projector(discs['cloud'].g)(PD.fine_inputs(pool, 'cpu')[2]).float()
    R2 = rollout(m2, d2, c2, 1, PD.DT)[0]
    assert R2.shape == (B, 2, PD.CLOUD_N) and torch.isfinite(R2).all()


@pytest.mark.parametrize('disc_name', ['g32', 'cloud'])
def test_chart_physics_for_random_parameters(tiny, disc_name):
    """for random parameters and random states the rates of the Neural Chart satisfy local detailed balance with the
    physical force of the condition (log a / b = f), so J f >= 0 and sigma >= 0, and S J conserves the mass"""
    pool, _, _, meta, discs = tiny
    disc = discs[disc_name]
    torch.manual_seed(1)
    torch.set_default_dtype(torch.float64)
    try:
        m = PM.build('chart', dict(meta, norm=None), disc).double()
    finally:
        torch.set_default_dtype(torch.float32)
    with torch.no_grad():
        for n, p in m.named_parameters():
            if n != 'drive.s_A':                          # the drive scale is physics (fixed at 1)
                p.add_(0.5 * torch.randn_like(p))
        dr = {k: (v.double() if torch.is_tensor(v) and v.is_floating_point() else v) for k, v in drive_on(pool, disc, T=3.0, amp=6.0).items()}
        c = torch.exp(torch.randn(dr['U'].shape[0], disc.V, dtype=torch.float64))
        a, b = m.rates(c, dr)
        f = PE.phys_force(c, dr, disc)
        assert torch.allclose(torch.log(a / b), f, atol=1e-9)
        J = a - b
        assert (J * f >= 0).all() and ((J * f).sum(-1) > 0).all()
        dc = m(c, dr)
        assert (dc.sum(-1).abs() < 1e-9 * (a + b).sum(-1)).all()            # the state is cell mass / mean cell volume


def test_chart_rollout_stays_positive(tiny):
    """a short rollout of the untrained Neural Chart under drive: positive densities, no LDB violation"""
    pool, _, _, meta, discs = tiny
    disc = discs['g32']
    m = PM.build('chart', dict(meta, norm=None), disc)
    dr = drive_on(pool, disc, T=2.0, amp=4.0)
    c0 = PD.projector(disc.g)(PD.fine_inputs(pool, 'cpu')[2]).float()
    R = rollout(m, dr, c0, 1, PD.DT)[0]
    assert R.min() > 0
    with torch.no_grad():
        ctx = m.prepare(dr)
        ph = PE.phys_metrics_rollout(R, dr, disc, rates_fn=lambda c: m.rates(c, ctx)).result()
    assert ph['ldb_violation_frac'] == 0 and ph['sigma_neg_frac'] == 0 and ph['neg_cell_frac'] == 0


def test_metrics():
    Y = (torch.rand(4, 1, 16) + 0.5).expand(4, 201, 16).contiguous()
    m = PE.rollout_metrics(Y.clone(), Y)
    assert m['err_10T'] == 0 and m['n_neg_traj'] == 0 and m['mass_drift_max'] == 0


DATA = PD.data_dir_of('porous_media')


@pytest.mark.slow
@pytest.mark.skipif(not os.path.exists(os.path.join(DATA, 'drive_x4.pt')), reason='benchmark data not generated')
def test_shipped_chart_reproduces_the_reported_error():
    """the shipped Neural Chart on the first 4 trajectories of 'drive ×4' (CPU, plain integrator): the median error is
    close to the reported one (the reported value uses all 64 trajectories and the fast integrator)"""
    data = PD.load('porous_media', 'cpu', sets=['train', 'drive_x4'])
    meta = PD.add_norm(data['meta'], data)
    m = PM.build('chart', meta, data['discs']['g32'])
    m.load_state_dict(torch.load(os.path.join(CHECKPOINTS, 'porous_media', 'chart.pt'), map_location='cpu'))
    m.eval()
    sd = data['drive_x4']
    dr = {k: v[:4] for k, v in sd['drive'].items()}
    R = rollout(m, dr, sd['c'][:4, 0], sd['c'].shape[1] - 1, PD.DT)[0]
    e = PE.rollout_metrics(R.cpu(), sd['c'][:4].cpu())['err_10T']
    rep = json.load(open(os.path.join(RESULTS, 'porous_media', 'runs', 'chart.json')))['sets']['drive_x4']
    ref = float(np.median(rep['err_10T_per_traj'][:4]))
    assert abs(e - ref) < 0.02 * ref
