"""Composition: the identities of docs/gluing.md on small networks."""
import warnings

import numpy as np
import pytest
import torch

from neuralchart import Conditions, Hypergraph, NeuralChart, Trajectories, compose


@pytest.fixture(autouse=True)
def double_precision():
    old = torch.get_default_dtype()
    torch.set_default_dtype(torch.float64)
    yield
    torch.set_default_dtype(old)


def rank_null(A):
    return np.asarray(A).shape[1] - np.linalg.matrix_rank(A)


def cycle_module():
    return Hypergraph.from_strings(['X + E + ATP <-> Xp + E + ADP', 'Xp <-> X'], chemostats=['ATP', 'ADP'], species=['X', 'Xp', 'E'])


def second_cycle():
    return Hypergraph.from_strings(['Y + K + ATP <-> Yp + K + ADP', 'Yp <-> Y'], chemostats=['ATP', 'ADP'], species=['Y', 'Yp', 'K'])


def chart_of(hg, seed, excess=True, material=True):
    """a chart with random nonzero networks and constants, without any fit"""
    torch.manual_seed(seed)
    n = len(hg.meta.get('chemostats', []))
    c = NeuralChart(hg, chemostats=[0.1 * i for i in range(n)] if n else None, excess=excess, material=material)
    with torch.no_grad():
        c.kinetics.core.normal_(0, 0.5)
        c.free_energy.mu0.normal_(0, 0.5)
        if excess:
            c.free_energy.l3.weight.normal_(0, 0.1)
        if material:
            list(c.kinetics.net.parameters())[-2].normal_(0, 0.05)
    c.dt.fill_(0.05)
    return c


def cond_of(hg, B=3):
    n = len(hg.meta.get('chemostats', []))
    d = dict(U=torch.randn(B, hg.V, generator=torch.Generator().manual_seed(1)) * 0.3)
    if n:
        d['chem_logc'] = torch.randn(B, n, generator=torch.Generator().manual_seed(2)) * 0.3
    return Conditions(d)


def state_of(hg, B=3):
    return 0.2 + torch.rand(B, hg.M, generator=torch.Generator().manual_seed(3))


@pytest.mark.parametrize('a,b,glue,new', [
    (lambda: Hypergraph.from_strings(['X <-> Y']), lambda: Hypergraph.from_strings(['Y <-> Z', 'Z <-> X']), {}, 1),
    (cycle_module, second_cycle, {'K': 'Xp'}, 0),
    (lambda: Hypergraph.from_strings(['X <-> Y']), lambda: Hypergraph.from_strings(['Y <-> Z']), {}, 0),
])
def test_dimension_of_the_obstruction(a, b, glue, new):
    """dim ker S - dim ker S_A - dim ker S_B = the new cycles = dim of the orthogonal of the restricted conservation laws (Theorem 1)"""
    a, b = a(), b()
    ab = compose(a, b, glue=glue)
    Z = ab.meta['new_cycles']
    assert Z.shape[1] == new
    assert rank_null(ab.S) - rank_null(a.S) - rank_null(b.S) == new
    mods = ab.meta['modules']
    y = sorted(set(mods[0]['species'].tolist()) & set(mods[1]['species'].tolist()))
    L = np.concatenate([m.conservation_laws()[[mod['species'].tolist().index(v) for v in y]] for m, mod in zip((a, b), mods)], 1)
    assert len(y) - np.linalg.matrix_rank(L) == new
    assert np.abs(ab.S @ Z).sum() < 1e-12


def test_a_lifted_module_has_one_copy_per_site():
    a, b = cycle_module(), Hypergraph.from_strings([], species=['X', 'Xp'], diffusing=['X', 'Xp'], grid_n=3)
    ab = compose(a, b)
    assert ab.V == 9 and ab.H == a.H * 9 + b.H
    n = rank_null(ab.S) - 9 * rank_null(a.S) - rank_null(b.S)
    assert ab.meta['new_cycles'].shape[1] == n and n > 0
    assert np.abs(ab.S.T @ ab.conservation_laws()).sum() < 1e-12


def test_spurious_affinity_and_its_projection():
    """a triangle: the affinity around the new cycle is the sum of the module forces, and the projection removes it"""
    a, b = Hypergraph.from_strings(['X <-> Y']), Hypergraph.from_strings(['Y <-> Z', 'Z <-> X'])
    ca, cb = NeuralChart(a, excess=False, material=False), NeuralChart(b, excess=False, material=False)
    with torch.no_grad():
        ca.free_energy.mu0.copy_(torch.tensor([0.0, 1.0]))
        cb.free_energy.mu0.copy_(torch.tensor([0.0, 0.5, 2.0]))          # Y, Z, X
    with pytest.warns(UserWarning, match='spurious affinity'):
        ab = compose(ca, cb)
    comp = ab.composition
    assert comp['n_new_cycles'] == 1
    # X -> Y in A: 1; Y -> Z in B: 0.5; Z -> X in B: 1.5
    assert np.isclose(abs(comp['spurious_affinity'][0]), 3.0)
    assert comp['force_change']['max'] > 0.5
    mu = ab.free_energy.mu0.detach().numpy()
    assert np.abs(ab.hg.cycle_basis().T @ (ab.hg.S.T @ mu)).max() < 1e-10


def test_no_new_cycle_leaves_the_forces_alone():
    with warnings.catch_warnings():
        warnings.simplefilter('error')
        ab = compose(chart_of(cycle_module(), 0), chart_of(second_cycle(), 1), glue={'K': 'Xp'})
    assert ab.composition['n_new_cycles'] == 0 and ab.composition['force_change']['max'] < 1e-10


def test_detailed_balance_and_entropy_production():
    a, b = chart_of(cycle_module(), 0), chart_of(second_cycle(), 1)
    ab = compose(a, b, glue={'K': 'Xp'})
    c, cond = state_of(ab.hg), cond_of(ab.hg)
    x, y = ab.rates(c, cond)
    f = ab.force(c, cond)
    assert torch.allclose(torch.log(x) - torch.log(y), f, atol=1e-10)
    j = x - y
    mods = ab.hg.meta['modules']
    parts = [(j[:, m['rows']] * f[:, m['rows']]).sum(-1) for m in mods]
    assert torch.allclose(parts[0] + parts[1], ab.entropy_production(c, cond))
    assert (parts[0] >= 0).all() and (parts[1] >= 0).all()


def test_ideal_modules_keep_their_rates():
    """no excess, no material term, no new cycle: the composite is the sum of the modules at the same state, and sigma = sigma_a + sigma_b"""
    a, b = chart_of(cycle_module(), 0, False, False), chart_of(second_cycle(), 1, False, False)
    ab = compose(a, b, glue={'K': 'Xp'})
    c, cond = state_of(ab.hg), cond_of(ab.hg)
    mods = ab.hg.meta['modules']
    sigma = 0.0
    for m, mod in zip((a, b), mods):
        cm = c[:, mod['species']]
        cc0 = Conditions(U=cond['U'], chem_logc=cond['chem_logc'][:, mod['chem']])
        x, y = ab.rates(c, cond)
        xm, ym = m.rates(cm, cc0)
        assert torch.allclose(x[:, mod['rows']], xm) and torch.allclose(y[:, mod['rows']], ym)
        sigma = sigma + m.entropy_production(cm, cc0)
    assert torch.allclose(ab.entropy_production(c, cond), sigma)


def test_conservation_laws_of_the_composite():
    ab = compose(cycle_module(), second_cycle(), glue={'K': 'Xp'})
    L = ab.conservation_laws()
    names = ab.species_names
    assert L.shape[1] == 3                                                # X + Xp, Y + Yp and E
    assert np.abs(ab.S.T @ L).sum() < 1e-12
    assert ab.meta['shared_species'] == ['Xp'] and 'K' not in names


def test_self_energy_is_not_counted_twice():
    """G = G_A + G_B - G_glue; with two equal modules on the shared species alone the composite G is that of one module"""
    a = chart_of(Hypergraph.from_strings(['Y <-> 2 Y']), 0, True, False)
    b = NeuralChart(a.hg, excess=True, material=False)
    b.load_state_dict(a.state_dict())
    b.dt.fill_(0.05)
    ab = compose(a, b)
    y = torch.rand(5, 1)
    assert torch.allclose(ab.free_energy.G(y), a.free_energy.G(y)) and torch.allclose(ab.free_energy.grad_G(y), a.free_energy.grad_G(y))
    assert ab.composition['self_energy_gap']['G'] < 1e-12


def test_self_energy_gap_is_reported():
    ab = compose(chart_of(cycle_module(), 0), chart_of(second_cycle(), 1), glue={'K': 'Xp'})
    assert ab.composition['self_energy_gap']['G'] > 0


def active(seed=0):
    ab = compose(chart_of(cycle_module(), 0), chart_of(second_cycle(), 1), glue={'K': 'Xp'})
    ab.kinetics.cross_on = ab.free_energy.cross_on = True
    torch.manual_seed(seed)
    with torch.no_grad():
        ab.free_energy.l3.weight.normal_(0, 0.1)
        list(ab.kinetics.cross.parameters())[-2].normal_(0, 0.1)
    return ab


def test_cross_terms_start_at_zero():
    ab = compose(chart_of(cycle_module(), 0), chart_of(second_cycle(), 1), glue={'K': 'Xp'})
    c, cond = state_of(ab.hg), cond_of(ab.hg)
    r0 = ab.rates(c, cond)
    ab.kinetics.cross_on = ab.free_energy.cross_on = True
    r1 = ab.rates(c, cond)
    assert torch.equal(r0[0], r1[0]) and torch.equal(r0[1], r1[1])


def test_cross_terms_vanish_where_the_modules_determine_the_answer():
    ab = active()
    fe, kin, mods = ab.free_energy, ab.kinetics, ab.hg.meta['modules']
    x = 0.2 + torch.rand(4, ab.hg.n_species, generator=torch.Generator().manual_seed(0))
    onlyA, onlyB = x.clone(), x.clone()
    priv_a = sorted(set(mods[0]['species'].tolist()) - set(mods[1]['species'].tolist()))
    priv_b = sorted(set(mods[1]['species'].tolist()) - set(mods[0]['species'].tolist()))
    onlyA[..., priv_b] = 0.0
    onlyB[..., priv_a] = 0.0
    h = lambda u: FreeG(fe, u)
    assert h(onlyA).abs().max() < 1e-12 and h(onlyB).abs().max() < 1e-12 and h(x).abs().max() > 1e-6
    c = 0.2 + torch.rand(4, ab.hg.M, generator=torch.Generator().manual_seed(4))
    ca, cb = c.clone(), c.clone()
    ca[..., priv_b] = 0.0
    cb[..., priv_a] = 0.0
    assert kin.cross_term(ca)[:, mods[0]['rows']].abs().max() < 1e-12
    assert kin.cross_term(cb)[:, mods[1]['rows']].abs().max() < 1e-12
    assert kin.cross_term(c).abs().max() > 1e-6


def FreeG(fe, u):
    on = fe.cross_on
    full = fe.G(u)
    fe.cross_on = False
    base = fe.G(u)
    fe.cross_on = on
    return full - base


def test_gradient_of_the_composite_free_energy():
    ab = active()
    x = (0.2 + torch.rand(3, ab.hg.n_species, generator=torch.Generator().manual_seed(5))).requires_grad_(True)
    g = torch.autograd.grad(ab.free_energy.G(x).sum(), x)[0]
    assert torch.allclose(g, ab.free_energy.grad_G(x.detach()), atol=1e-9)


def test_local_detailed_balance_holds_with_cross_terms():
    ab = active()
    c, cond = state_of(ab.hg), cond_of(ab.hg)
    x, y = ab.rates(c, cond)
    assert torch.allclose(torch.log(x) - torch.log(y), ab.force(c, cond), atol=1e-10)


def test_refine_moves_only_the_cross_terms(tmp_path):
    a, b = chart_of(cycle_module(), 0), chart_of(second_cycle(), 1)
    ab = compose(a, b, glue={'K': 'Xp'})
    frozen = {k: v.clone() for k, v in ab.state_dict().items() if not (k.startswith('kinetics.cross') or k.split('.')[0:2] in (['free_energy', 'l1'], ['free_energy', 'l2'], ['free_energy', 'l3']))}
    c = [state_of(ab.hg, 4)]
    cond = cond_of(ab.hg, 4)
    with torch.no_grad():
        for _ in range(10):
            c.append(c[-1] * (1 + 0.01 * torch.tanh(c[-1])))
    ab.refine(Trajectories(torch.stack(c, 1), cond, dt=0.05), iters=3, log=lambda *x: None)
    now = ab.state_dict()
    assert all(torch.equal(now[k], v) for k, v in frozen.items())
    assert list(ab.kinetics.cross.parameters())[-2].abs().max() > 0
    p = tmp_path / 'ab.pt'
    ab.save(p)
    back = type(ab).load(p)
    s = state_of(ab.hg)
    assert torch.allclose(back.rates(s, cond_of(ab.hg))[0], ab.rates(s, cond_of(ab.hg))[0])
