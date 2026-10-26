"""The wired mode: thermodynamics wired (log c + U / T, exact force line integrals and chemostat potentials), kinetics
learned.  Untrained it is the SQRA discretisation; temperature and drive edits extrapolate without retraining."""
import os
import numpy as np
import pytest
import torch
from hyperme import presets, Hypergraph, HyperME
from hyperme.model import WiredFreeEnergy, site_grad, Context
from hyperme.train import rollout, relerr
from conftest import REPO

SQRA_10T = 4.12e-3          # results/tcme_summary.md, E2: SQRA oracle, test_id, 32 x 32, 10T


def need(name):
    if not os.path.exists(os.path.join(REPO, 'runs', 'data', f'{name}.pt')):
        pytest.skip(f'missing runs/data/{name}.pt')


def err10(model, drive, c):
    with torch.no_grad():
        R = rollout(model, drive, c[:, 0], c.shape[1] - 1)[0]
    assert R.min() > 0
    return float(np.median(relerr(R.double(), c.double()).numpy()[:, -1]))


@pytest.fixture(scope='module')
def wired():
    torch.set_num_threads(int(os.environ.get('OMP_NUM_THREADS', '8')))
    return presets.s2_wired().eval()


def test_untrained_wired_is_sqra(wired):
    need('S2')
    te = presets.load_s2()['test_id']
    e = err10(wired, te['drive'], te['c'])
    assert abs(e - SQRA_10T) / SQRA_10T < 0.03, e


@pytest.mark.parametrize('name,T,tol', [('S2_D0.01', 0.5, 0.02), ('S2_D0.04', 2.0, 0.005)])
def test_temperature_edit(wired, name, T, tol):
    """diffusion D = T D_train with the same normalised inputs: edit(T) reproduces it without retraining
    (unedited error 0.44 / 0.34)."""
    need(name)
    d = presets.load_s2(name=name)['test']
    assert err10(wired.edit(T=T), d['drive'], d['c']) < tol
    assert wired.knobs['T'] is None                      # the original model is unchanged


@pytest.mark.parametrize('name,scale,tol', [('S2_amp0', 0.0, 0.006), ('S2_amp2', 2.0, 0.015)])
def test_drive_edit(wired, name, scale, tol):
    """drift amplitude x scale with the amplitude-1 inputs; scale 0 is the counterfactual A <- 0
    (unedited error 0.26 / 0.17)."""
    need(name); need('S2_amp1')
    d = presets.load_s2(name=name)['test']
    inputs = presets.load_s2(name='S2_amp1')['test']['drive']
    assert err10(wired.edit(drive_scale=scale), inputs, d['c']) < tol


@pytest.mark.parametrize('nonideal', ['virial', 'debye_huckel'])
def test_nonideal_families(nonideal):
    """dmu is the gradient of the site-local excess free energy; zero (or 1e-6) at initialisation; virial F convex."""
    hg = Hypergraph.reaction_transport(['A', 'B'], [({'A': 1}, {'B': 1})], n_sites=3)
    with presets.default_dtype(torch.float64):
        fe = WiredFreeEnergy(hg, nonideal, charges=[1.0, 2.0]).double()
    g = torch.Generator().manual_seed(0)
    c = torch.rand(4, hg.M, generator=g, dtype=torch.float64) + 0.1
    ctx = Context(Ufull=torch.zeros_like(c))
    assert fe.correction(c, ctx).abs().max() < 1e-5                       # ideal limit at initialisation
    with torch.no_grad():
        for p in fe.parameters():
            p.copy_(torch.randn(p.shape, generator=g, dtype=p.dtype) * 0.5)
    ideal = (c * (torch.log(c) - 1)).sum(-1)
    grad = site_grad(lambda x: fe.value(x, ctx) - (x * (torch.log(x) - 1)).sum(-1), c)
    assert torch.allclose(fe.correction(c, ctx), grad, atol=1e-10)
    if nonideal == 'virial':
        Hs = torch.autograd.functional.hessian(lambda x: fe.value(x[None], ctx).sum(), c[0])
        assert torch.linalg.eigvalsh(Hs).min() > 0
    assert torch.isfinite(ideal).all()


def test_mode_switch_and_api():
    hg = Hypergraph.grid_transport(4)
    m_w, m_f = HyperME(hg), HyperME(hg, mode='flexible')
    assert m_w.mode == 'wired' and m_f.mode == 'flexible'
    n_w, n_f = (sum(p.numel() for p in m.parameters()) for m in (m_w, m_f))
    assert n_w < n_f / 10                                 # the wired model learns little
    drive = dict(U=torch.randn(2, 16), T=torch.tensor([1.0, 2.0]), line=torch.randn(2, hg.H))
    c = torch.rand(2, 16) + 0.5
    A0 = m_w.prepare(drive)['A']
    A2 = m_w.edit(drive_scale=0.0).prepare(drive)['A']
    assert A2.abs().max() == 0 and A0.abs().max() > 0
    assert torch.allclose(A0[0], drive['line'][0]) and torch.allclose(A0[1] * 2, drive['line'][1])   # A = line / T
    k0 = m_w.prepare(drive)['k']
    assert torch.allclose(k0[1], 2 * k0[0])                                                         # k = k_ref T
    out = m_w.readouts(c, drive)
    assert (out['sigma'] >= 0).all()


def test_chemostats_enter_as_mass_action_species():
    """A + ATP <-> B + ADP with mass action: a = kf A ATP, b = kb B ADP.  The wired model with k_ref = sqrt(kf kb) and
    standard potentials mu0_A - mu0_B = log(kf / kb) reproduces both one-way rates for any ATP, ADP (the chemostats
    carry the symmetric factor sqrt(ATP ADP) as well as the force log(ATP / ADP))."""
    kf, kb = 3.0, 0.2
    hg = Hypergraph.reaction_transport(['A', 'B', 'ATP', 'ADP'], [({'A': 1, 'ATP': 1}, {'B': 1, 'ADP': 1})], n_sites=1,
                                       chemostats=['ATP', 'ADP'])
    with presets.default_dtype(torch.float64):
        m = HyperME(hg, k_ref=float(np.sqrt(kf * kb)), learn_mu0=True, T_exponent=0.0).double()
    with torch.no_grad():
        m.free_energy.mu0.copy_(torch.tensor([np.log(kf / kb), 0.0]))
    g = torch.Generator().manual_seed(0)
    lc = torch.randn(5, 2, generator=g, dtype=torch.float64)
    c = torch.rand(5, 2, generator=g, dtype=torch.float64) + 0.1
    a, b = m.rates(c, dict(chem_logc=lc))
    assert torch.allclose(a[:, 0], kf * c[:, 0] * lc[:, 0].exp(), rtol=1e-12)
    assert torch.allclose(b[:, 0], kb * c[:, 1] * lc[:, 1].exp(), rtol=1e-12)
