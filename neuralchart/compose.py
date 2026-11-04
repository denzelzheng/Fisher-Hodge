"""Composition of hypergraphs and of fitted Neural Charts.  The statements are in docs/gluing.md.

    ab = compose(a, b, glue={'K': 'Xp'})          # hypergraphs: union of species and hyperedges, b's K identified with a's Xp
    ab = compose(chart_a, chart_b, glue=...)      # fitted charts: no retraining; ab.composition holds the diagnostics
    ab.refine(trajectories)                       # short fit of the cross terms on data of the composite, modules frozen

Hypergraphs.  A module with one site is copied to every site of the other module.  Species with the same name, or named in glue
(name in b -> name in a), are one species; other species of b that collide get the suffix '_b'.  Chemostats are glued in the same way.

Fitted charts, with module A on species (x, y), module B on (y, z), y shared.  Each module keeps its g_theta, omega and class labels.
  mu0: the module forces come from one potential if and only if they vanish around the new cycles.  composition['spurious_affinity'] has
       one number per new cycle before the projection; mu0 of the composite is the least-squares potential of the module forces, and
       composition['force_change'] is what the projection does to them.  A spurious affinity warns.
  G:   G_A(x, y) + G_B(y, z) - G_glue(y) + G_cross(x, y, z), G_glue(y) = [G_A(0, y) + G_B(y, 0)] / 2.  composition['self_energy_gap'] is the
       difference of the two estimates of the self energy.  G_cross vanishes at x = 0 and at z = 0 and starts at zero.
  k:   each module's own; g_cross(h, x, y, z) vanishes at z = 0 on the hyperedges of A and at x = 0 on those of B and starts at zero.
The modules alone say nothing about the cross terms.
"""
import copy
import warnings

import numpy as np
import scipy.linalg
import scipy.sparse as sp
import torch
import torch.nn as nn

from .chart import ITERS, LR, NeuralChart
from .graphs import Hypergraph
from .model import FreeEnergy, MLP, Mobility, _mlp


def _names(hg):
    return list(hg.species_names)


def _lift(mat, H, Vm, V, col):
    """the coo entries of a module matrix (H, ncolm) on V sites.  col(species, site) -> new column; a one-site module is copied"""
    m = sp.coo_matrix(mat)
    if Vm == V:
        return m.row, col(m.col // Vm, m.col % Vm), m.data
    rows = np.concatenate([m.row * V + v for v in range(V)]) if len(m.row) else np.zeros(0, dtype=np.int64)
    cols = np.concatenate([col(m.col // Vm, np.full(len(m.col), v)) for v in range(V)]) if len(m.col) else np.zeros(0, dtype=np.int64)
    return rows, cols, np.tile(m.data, V)


def compose_graphs(a, b, glue=None):
    """the union hypergraph; meta['modules'] = per module its hyperedge rows, species indices and first class"""
    glue = dict(glue or {})
    V = max(a.V, b.V)
    assert a.V in (1, V) and b.V in (1, V), 'the modules must have the same sites, or one site'
    names_a, names_b = _names(a), _names(b)
    chem_a, chem_b = list(a.meta.get('chemostats', [])), list(b.meta.get('chemostats', []))
    gl = lambda n: glue.get(n, n)
    species = list(names_a)
    map_b = {}
    for n in names_b:
        t = gl(n)
        if t in names_a and (n in glue or t == n):
            map_b[n] = names_a.index(t)
        else:
            nm = t if t not in species else t + '_b'
            species.append(nm)
            map_b[n] = len(species) - 1
    chems = list(chem_a)
    for n in chem_b:
        if gl(n) not in chems:
            chems.append(gl(n))
    NS, NC = len(species), len(chems)
    sa, sb = np.arange(len(names_a)), np.array([map_b[n] for n in names_b], dtype=np.int64)
    ca, cb = np.array([chems.index(n) for n in chem_a], dtype=np.int64), np.array([chems.index(gl(n)) for n in chem_b], dtype=np.int64)
    Ha = a.H * (V // a.V)
    Hb = b.H * (V // b.V)
    H = Ha + Hb

    def assemble(fa, fb, ncol, idx_a, idx_b):
        rows, cols, vals = [], [], []
        for mat, Hm, Vm, idx, off in ((fa, a.H, a.V, idx_a, 0), (fb, b.H, b.V, idx_b, Ha)):
            if mat is None:
                continue
            col = lambda sp_, site: idx[sp_] * V + site
            r, c, x = _lift(mat, Hm, Vm, V, col)
            rows.append(r + off); cols.append(c); vals.append(x)
        return sp.coo_matrix((np.concatenate(vals), (np.concatenate(rows), np.concatenate(cols))), shape=(H, ncol)).tocsr() if rows else sp.csr_matrix((H, ncol))

    Np = assemble(a.Np, b.Np, NS * V, sa, sb)
    Nm = assemble(a.Nm, b.Nm, NS * V, sa, sb)
    na, nb = int(a.meta['typ'].max()) + 1, int(b.meta['typ'].max()) + 1

    def lift_h(x, Vm):                                   # per-hyperedge values of a module, in the row order of _lift
        return np.repeat(x, V) if Vm != V else x
    meta = dict(typ=np.concatenate([lift_h(a.meta['typ'], a.V), lift_h(b.meta['typ'], b.V) + na]),
                odd=np.concatenate([lift_h(a.meta['odd'], a.V), lift_h(b.meta['odd'], b.V)]))
    site, et, eh = [], [], []
    for m in (a, b):
        h = m.H
        s_ = np.asarray(m.meta.get('site', np.full(h, -1)))
        e_t, e_h = np.asarray(m.meta.get('et', np.full(h, -1))), np.asarray(m.meta.get('eh', np.full(h, -1)))
        if m.V == V:
            site.append(s_); et.append(e_t); eh.append(e_h)
        else:
            site.append(np.concatenate([np.full(h, v) for v in range(V)]).reshape(V, h).T.ravel()); et.append(np.repeat(e_t, V)); eh.append(np.repeat(e_h, V))
    meta.update(site=np.concatenate(site), et=np.concatenate(et), eh=np.concatenate(eh))
    w = []
    for m, hm in ((a, Ha), (b, Hb)):
        ew = m.meta.get('edge_weight')
        w.append(np.ones(hm) if ew is None else lift_h(np.asarray(ew), m.V))
    if a.meta.get('edge_weight') is not None or b.meta.get('edge_weight') is not None:
        meta['edge_weight'] = np.concatenate(w)
    for m in (a, b):
        if m.meta.get('site_volume') is not None and m.V == V:
            meta['site_volume'] = np.asarray(m.meta['site_volume'])
    if NC:
        meta['chemostats'] = chems
        for key in ('chem_S', 'chem_Np', 'chem_Nm'):
            fa, fb = a.meta.get(key) if chem_a else None, b.meta.get(key) if chem_b else None
            meta[key] = assemble(fa, fb, NC * V, ca, cb)
    if a.meta.get('grid_n') == b.meta.get('grid_n') and a.meta.get('grid_n') is not None:
        meta['grid_n'] = a.meta['grid_n']
    elif a.meta.get('grid_n') or b.meta.get('grid_n'):
        meta['grid_n'] = a.meta.get('grid_n') or b.meta.get('grid_n')
    meta['modules'] = [dict(rows=np.arange(Ha), species=sa, classes=(0, na), chem=ca), dict(rows=Ha + np.arange(Hb), species=sb, classes=(na, na + nb), chem=cb)]
    meta['shared_species'] = [species[i] for i in sb if i in sa]
    return Hypergraph(Np, Nm, n_species=NS, species_names=species, meta=meta)


def _module_cycles(m, mod, ab):
    """the cycles of a module on the hyperedges of the composite (H, n); a one-site module has one copy per site"""
    V, rows, zm = ab.V, mod['rows'], m.cycle_basis()
    if m.V == V:
        zz = np.zeros((ab.H, zm.shape[1]))
        zz[rows] = zm
        return zz
    zz = np.zeros((ab.H, zm.shape[1] * V))
    for v in range(V):
        for k in range(zm.shape[1]):
            zz[rows[0] + np.arange(m.H) * V + v, v * zm.shape[1] + k] = zm[:, k]
    return zz


def new_cycles(a, b, ab):
    """cycles of the composite that complete those of the modules to a basis of ker S (H, n), with the integer entries of the cycle basis"""
    Z = np.concatenate([_module_cycles(m, mod, ab) for m, mod in zip((a, b), ab.meta['modules'])], 1)
    all_z = ab.cycle_basis()
    n = all_z.shape[1] - Z.shape[1]
    assert n >= 0, 'the cycles of the two modules are independent in the composite'
    if not Z.shape[1]:
        return all_z
    q, _ = np.linalg.qr(Z)
    res = all_z - q @ (q.T @ all_z)
    _, _, piv = scipy.linalg.qr(res, pivoting=True, mode='economic')
    return all_z[:, np.sort(piv[:n])]


# ------------------------------------------------------------------------------------------ fitted charts
class ComposedMobility(Mobility):
    """k_h from the module of h, its omega and its g_theta on its class one-hot and the log cbar of its own species, times
    exp g_cross(h, cbar) with g_cross(h, cbar) = n(h, u) - n(h, u at the private species of the other module = 0), u = log(1 + cbar)"""

    def __init__(s, hg, kins, hidden=64):
        super().__init__(1, hg, material=False)
        mods = hg.meta['modules']
        with torch.no_grad():
            s.core.copy_(torch.cat([k.core.detach() for k in kins]).to(s.core.dtype))
        s.nets = nn.ModuleList([copy.deepcopy(k.net) if k.net is not None else nn.Identity() for k in kins])
        s.has_net = [k.net is not None for k in kins]
        keep = torch.ones(hg.H, hg.n_species)
        for i, mod in enumerate(mods):
            s.register_buffer(f'rows{i}', torch.as_tensor(mod['rows'], dtype=torch.long), persistent=False)
            s.register_buffer(f'spec{i}', torch.as_tensor(mod['species'], dtype=torch.long), persistent=False)
            s.register_buffer(f'cls{i}', s.typ[mod['rows']] - mod['classes'][0], persistent=False)
            other = sorted(set(mods[1 - i]['species'].tolist()) - set(mod['species'].tolist()))
            keep[np.ix_(mod['rows'], other)] = 0.0
        s.register_buffer('keep', keep, persistent=False)
        s.ncls = [mod['classes'][1] - mod['classes'][0] for mod in mods]
        s.n_all = sum(s.ncls)
        s.cross = MLP(s.n_all + hg.n_species, hidden, 1, last='zero')
        s.cross_on = False

    @property
    def rho_dependent(s):
        return any(s.has_net) or s.cross_on

    def cross_term(s, c):
        """g_cross (B, H) at the state c"""
        u = torch.log1p(torch.exp(s.log_cbar(c)))
        oh = torch.nn.functional.one_hot(s.typ, s.n_all).to(u.dtype).expand(u.shape[0], -1, -1)
        return (_mlp(s.cross, torch.cat([oh, u], -1)) - _mlp(s.cross, torch.cat([oh, u * s.keep.to(u.dtype)], -1))).squeeze(-1)

    def state_term(s, x, c):
        u = s.log_cbar(c)
        out = torch.zeros(u.shape[:2], dtype=u.dtype, device=u.device)
        for i, net in enumerate(s.nets):
            if not s.has_net[i]:
                continue
            rows, spec, cls = getattr(s, f'rows{i}'), getattr(s, f'spec{i}'), getattr(s, f'cls{i}')
            oh = torch.nn.functional.one_hot(cls, s.ncls[i]).to(u.dtype).expand(u.shape[0], -1, -1)
            out[:, rows] = _mlp(net, torch.cat([oh, u[:, rows][:, :, spec]], -1)).squeeze(-1)
        return out + s.cross_term(c) if s.cross_on else out


class ComposedFreeEnergy(FreeEnergy):
    """G = G_A(x, y) + G_B(y, z) - G_glue(y) + G_cross(x, y, z).  The network of the parent class is h in
    G_cross = h(x, y, z) - h(0, y, z) - h(x, y, 0) + h(0, y, 0), so G_cross vanishes at x = 0 and at z = 0 and starts at zero."""

    def __init__(s, hg, fes, mu0):
        super().__init__(hg, True, excess=True)
        with torch.no_grad():
            s.mu0.copy_(torch.as_tensor(mu0, dtype=s.mu0.dtype))
        s.parts = nn.ModuleList()
        for f in fes:                                    # own copies; the hypergraph of a module is shared
            part = FreeEnergy(f.hg, f.mu0 is not None, excess=f.excess)
            part.load_state_dict(f.state_dict())
            s.parts.append(part)
        mods = hg.meta['modules']
        sa, sb = mods[0]['species'].tolist(), mods[1]['species'].tolist()
        sh = sorted(set(sa) & set(sb))
        s.register_buffer('shared', torch.as_tensor(sh, dtype=torch.long), persistent=False)
        for i, sp_ in enumerate((sa, sb)):
            s.register_buffer(f'spec{i}', torch.as_tensor(sp_, dtype=torch.long), persistent=False)
            s.register_buffer(f'sh{i}', torch.as_tensor([sp_.index(v) for v in sh], dtype=torch.long), persistent=False)
            k = torch.ones(hg.n_species)
            k[sorted(set(sp_) - set(sh))] = 0.0
            s.register_buffer(f'keep{i}', k, persistent=False)           # zero on the private species of this module
        s.excess_on = [f.excess for f in fes]
        s.cross_on = False

    def _restrict(s, i, y):
        """the state of module i with its private species at zero and the shared species at y"""
        z = torch.zeros(*y.shape[:-1], getattr(s, f'spec{i}').numel(), dtype=y.dtype, device=y.device)
        z[..., getattr(s, f'sh{i}')] = y
        return z

    def grad_G(s, x):
        g = torch.zeros_like(x)
        for i, f in enumerate(s.parts):
            if not s.excess_on[i]:
                continue
            spec, sh = getattr(s, f'spec{i}'), getattr(s, f'sh{i}')
            g = g.index_add(-1, spec, f.grad_G(x[..., spec]))
            if len(sh):
                g = g.index_add(-1, s.shared, -0.5 * f.grad_G(s._restrict(i, x[..., s.shared]))[..., sh])
        if s.cross_on:
            h = super().grad_G
            kx, kz = s.keep0.to(x.dtype), s.keep1.to(x.dtype)            # x private to A is zeroed by keep0, z private to B by keep1
            g = g + h(x) - kx * h(x * kx) - kz * h(x * kz) + kx * kz * h(x * kx * kz)
        return g

    def G(s, x):
        g = torch.zeros_like(x[..., :1])
        for i, f in enumerate(s.parts):
            if not s.excess_on[i]:
                continue
            g = g + f.G(x[..., getattr(s, f'spec{i}')])
            if len(getattr(s, f'sh{i}')):
                g = g - 0.5 * f.G(s._restrict(i, x[..., s.shared]))
        if s.cross_on:
            h = super().G
            kx, kz = s.keep0.to(x.dtype), s.keep1.to(x.dtype)
            g = g + h(x) - h(x * kx) - h(x * kz) + h(x * kx * kz)
        return g

    @torch.no_grad()
    def self_energy_gap(s, y):
        """the difference of the two estimates of the self energy at the shared densities y (..., n_shared), and of their gradients; rms"""
        d = {}
        ests = [f.G(s._restrict(i, y)) - f.G(s._restrict(i, torch.zeros_like(y))) if s.excess_on[i] else torch.zeros_like(y[..., :1]) for i, f in enumerate(s.parts)]
        grads = [(f.excess_potential(s._restrict(i, y))[..., getattr(s, f'sh{i}')] if s.excess_on[i] else torch.zeros_like(y)) for i, f in enumerate(s.parts)]
        d['G'] = float(torch.sqrt(((ests[0] - ests[1]) ** 2).mean()))
        d['gradient'] = float(torch.sqrt(((grads[0] - grads[1]) ** 2).mean()))
        return d


def _module_mu0(m):
    mu = m.free_energy.mu0
    return np.zeros(m.hg.n_species) if mu is None else mu.detach().cpu().numpy().astype(np.float64)


def _glue_potentials(a, b, ab, Z):
    """the spurious affinity around the new cycles Z, the one potential by least squares, and what it changes in the module forces"""
    mods = ab.meta['modules']
    V, NS = ab.V, ab.n_species
    S = np.asarray(ab.S)
    f = np.zeros(ab.H)
    for m, mod in zip((a, b), mods):
        mu = np.zeros(NS)
        mu[mod['species']] = _module_mu0(m)
        f[mod['rows']] = S[:, mod['rows']].T @ np.repeat(mu, V)
    D = S.T.reshape(ab.H, NS, V).sum(-1)                 # the force of a hyperedge in a potential that is uniform per species
    mu0 = np.linalg.lstsq(D, f, rcond=1e-10)[0]
    change = D @ mu0 - f
    return mu0, Z.T @ f, change


def _shared_grid(a, b, ab, fe, n=256):
    """shared densities inside the ranges the two modules saw; every shared species between zero and the larger of the two largest values"""
    mods = ab.meta['modules']
    shared = fe.shared.tolist()
    hi = np.zeros(len(shared))
    for m, mod in zip((a, b), mods):
        top = m.diagnostics.get('density_max')
        if top is not None:
            sp_ = mod['species'].tolist()
            hi = np.maximum(hi, [top[sp_.index(v)] for v in shared])
    hi = np.where(hi > 0, hi, 1.0)
    g = torch.Generator().manual_seed(0)
    return torch.rand(n, len(shared), generator=g, dtype=torch.float64) * torch.as_tensor(hi, dtype=torch.float64)


class ComposedChart(NeuralChart):
    """the union of two fitted charts.  refine() fits only the cross terms; the modules stay frozen"""

    def refine(s, trajectories, conditions=None, times=None, iters=500, seed=0, log=print, lr=LR):
        for p in s.parameters():
            p.requires_grad_(False)
        s.kinetics.cross_on = s.free_energy.cross_on = True
        for p in list(s.kinetics.cross.parameters()) + s.free_energy.excess_parameters():
            p.requires_grad_(True)
        return super().refine(trajectories, conditions, times, iters=iters, seed=seed, log=log, lr=lr)

    def save(s, path):
        m = s.modules_fitted
        torch.save(dict(composed=True, glue=s.glue, a=dict(args=m[0].args, state=m[0].state_dict(), hypergraph=m[0].hg.to_dict(), diagnostics=m[0].diagnostics),
                        b=dict(args=m[1].args, state=m[1].state_dict(), hypergraph=m[1].hg.to_dict(), diagnostics=m[1].diagnostics),
                        state=s.state_dict(), cross_on=s.kinetics.cross_on, dt=float(s.dt)), path)

    @classmethod
    def load(cls, path, hypergraph=None, map_location='cpu'):
        d = torch.load(path, map_location=map_location, weights_only=False)
        mods = []
        for k in ('a', 'b'):
            m = NeuralChart(Hypergraph.from_dict(d[k]['hypergraph']), **d[k]['args'])
            m.load_state_dict(d[k]['state'])
            m.diagnostics = d[k]['diagnostics']
            mods.append(m)
        c = compose_charts(mods[0], mods[1], d['glue'], quiet=True)
        c.kinetics.cross_on = c.free_energy.cross_on = d['cross_on']
        c.load_state_dict(d['state'])
        return c.eval()

    def on(s, hypergraph):
        raise NotImplementedError('a composed chart lives on its composite; compose the modules on the new hypergraphs')


def compose_charts(a, b, glue=None, quiet=False):
    """the union of two fitted charts, without retraining; chart.composition holds the diagnostics"""
    assert a.args['descriptors'] == 0 and b.args['descriptors'] == 0, 'charts with descriptors cannot be composed yet'
    assert float(a.dt) == float(b.dt) or not (np.isfinite(float(a.dt)) and np.isfinite(float(b.dt))), 'the modules must have the same sampling step'
    ab = compose_graphs(a.hg, b.hg, glue)
    Z = new_cycles(a.hg, b.hg, ab)
    mu0, spurious, change = _glue_potentials(a, b, ab, Z)
    chems = ab.meta.get('chemostats', [])
    known = None
    if chems:
        vals = {}
        for m, mod in zip((a, b), ab.meta['modules']):
            if m.args['chemostats'] is not None:
                for n, v in zip(mod['chem'], m.args['chemostats']):
                    vals.setdefault(int(n), []).append(float(v))
        if len(vals) == len(chems):
            known = [float(np.mean(vals[i])) for i in range(len(chems))]
    dev = a.dt.device
    chart = ComposedChart(ab, chemostats=known, material=False, excess=False).to(dev)
    old = torch.get_default_dtype()
    torch.set_default_dtype(torch.float64)
    try:
        chart.kinetics = ComposedMobility(ab, [a.kinetics, b.kinetics]).double().to(dev)
        chart.free_energy = ComposedFreeEnergy(ab, [a.free_energy, b.free_energy], mu0).double().to(dev)
    finally:
        torch.set_default_dtype(old)
    if known is None and chems:
        with torch.no_grad():
            for m, mod in zip((a, b), ab.meta['modules']):
                if m.drive.chem_mu0 is not None:
                    chart.drive.chem_mu0[mod['chem']] = m.drive.chem_mu0.detach().to(chart.drive.chem_mu0.dtype)
    chart.dt.fill_(float(a.dt) if np.isfinite(float(a.dt)) else float(b.dt))
    chart.args, chart.glue, chart.modules_fitted = None, dict(glue or {}), (a, b)
    fe = chart.free_energy
    gap = fe.self_energy_gap(_shared_grid(a, b, ab, fe).to(dev)) if len(fe.shared) else dict(G=0.0, gradient=0.0)
    chart.composition = dict(shared_species=ab.meta['shared_species'], new_cycles=Z, n_new_cycles=int(Z.shape[1]), spurious_affinity=spurious.tolist(),
                             force_change=dict(max=float(np.abs(change).max()), rms=float(np.sqrt(np.mean(change ** 2)))),
                             self_energy_gap=gap, chem_mu0_known=known is not None)
    if not quiet and Z.shape[1] and np.abs(spurious).max() > 1e-6:
        warnings.warn(f'gluing: the module potentials leave a spurious affinity up to {np.abs(spurious).max():.3g} around {Z.shape[1]} new cycles; '
                      f'projecting to one potential changes the module forces by up to {np.abs(change).max():.3g}')
    if np.isfinite(float(chart.dt)):
        chart.accelerate()
    return chart


def compose(a, b, glue=None):
    """compose(hypergraph, hypergraph) or compose(chart, chart); see the module docstring"""
    if isinstance(a, NeuralChart):
        return compose_charts(a, b, glue)
    ab = compose_graphs(a, b, glue)
    ab.meta['new_cycles'] = new_cycles(a, b, ab)
    return ab
