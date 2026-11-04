"""Hypergraphs of reaction and transport networks.

A hypergraph is given by two non-negative integer matrices nu_plus, nu_minus of shape (H, M): hyperedge h turns the
multiset nu_plus[h] of state components into nu_minus[h].  State components are species x sites, flattened
species-major (index m = s * V + v).  From (nu_plus, nu_minus) we build

    S       = (nu_minus - nu_plus)^T      (M, H)    dc/dt = S J
    Z       orthonormal basis of ker S    (H, z)    cycle space
    L       orthonormal basis of ker S^T  (M, l)    conserved moieties, c @ L is constant in time

and the gather / scatter index arrays used by the models.  Hypergraphs with H * M <= DENSE_MAX use dense matrix
products, which are faster at that size.  Transport of one species along a graph edge i -> j is the hyperedge
{X@i} -> {X@j}.

Builders: Hypergraph.reaction_transport (a reaction list copied at every site of a spatial graph, species-selective
transport, chemostatted species as a drive) and Hypergraph.from_graph (transport along the edges of a graph).
On the periodic n x n grid the edge order is that of grid_edge_list: first the n^2 horizontal edges
(a, b) -> (a, b + 1), then the n^2 vertical edges (a, b) -> (a + 1, b).
"""
import re

import numpy as np
import scipy.linalg
import scipy.sparse as sp
import torch


def grid_edge_list(n):
    """(2 n^2, 2) array of (tail, head) node indices in the flat hyperedge order: horizontal edges, then vertical edges."""
    idx = np.arange(n * n).reshape(n, n)
    tail = np.concatenate([idx.ravel(), idx.ravel()])
    head = np.concatenate([np.roll(idx, -1, 1).ravel(), np.roll(idx, -1, 0).ravel()])
    return np.stack([tail, head], 1)


def _padded(N):
    """sparse (H, M) -> padded (H, K) index / coefficient arrays (pad index M, pad coefficient 0)."""
    N = sp.csr_matrix(N)
    H, M = N.shape
    K = max(1, int(np.diff(N.indptr).max()) if H else 1)
    idx = np.full((H, K), M, dtype=np.int64); coef = np.zeros((H, K))
    for h in range(H):
        lo, hi = N.indptr[h], N.indptr[h + 1]
        idx[h, :hi - lo] = N.indices[lo:hi]; coef[h, :hi - lo] = N.data[lo:hi]
    return idx, coef


def _triplets(N):
    """sparse (H, M) -> (h, m, val) of its non-zeros"""
    N = sp.coo_matrix(N)
    keep = N.data != 0
    return N.row[keep].astype(np.int64), N.col[keep].astype(np.int64), N.data[keep].astype(np.float64)


def _rational_null(A, tol=1e-9):
    """a basis of the null space of A by elimination (the free columns get unit vectors), rational for a rational A; (n, k)"""
    A = np.array(A, np.float64)
    m, n = A.shape
    pivots, row = [], 0
    for col in range(n):
        if row >= m:
            break
        p = row + int(np.argmax(np.abs(A[row:, col])))
        if abs(A[p, col]) < tol:
            continue
        A[[row, p]] = A[[p, row]]
        A[row] /= A[row, col]
        others = np.arange(m) != row
        A[others] -= A[others, col, None] * A[row]
        pivots.append(col)
        row += 1
    free = [c for c in range(n) if c not in pivots]
    Z = np.zeros((n, len(free)))
    for k, c in enumerate(free):
        Z[c, k] = 1.0
        for r, pc in enumerate(pivots):
            Z[pc, k] = -A[r, c]
    return Z


class Hypergraph:
    DENSE_MAX = 4_000_000

    def __init__(s, nu_plus, nu_minus, n_species=1, meta=None, cycle_basis=None, species_names=None):
        s.Np = sp.csr_matrix(nu_plus, dtype=np.float64)
        s.Nm = sp.csr_matrix(nu_minus, dtype=np.float64)
        assert s.Np.shape == s.Nm.shape
        assert s.Np.min() >= 0 and s.Nm.min() >= 0, 'stoichiometric coefficients must be non-negative'
        s.H, s.M = s.Np.shape
        assert s.M % n_species == 0
        s.n_species, s.V = n_species, s.M // n_species
        s.species_names = species_names or [f's{i}' for i in range(n_species)]
        s.meta = dict(meta or {})
        s.meta.setdefault('typ', np.zeros(s.H, dtype=np.int64))
        s.meta.setdefault('odd', np.zeros(s.H, dtype=bool))         # orientation-odd hyperedges (transport)
        s._Z = None if cycle_basis is None else np.asarray(cycle_basis, np.float64)
        s._L = None
        s._torch = {}
        s.rp_idx, s.rp_coef = _padded(s.Np)
        s.rm_idx, s.rm_coef = _padded(s.Nm)
        Sm = (s.Nm - s.Np).tocsr()
        s.s_h, s.s_m, s.s_val = _triplets(Sm)
        s.cp = _triplets(Sm.multiply(Sm < 0) * -1.0)              # net reactants of the forward direction
        s.cm = _triplets(Sm.multiply(Sm > 0))                     # net reactants of the backward direction
        s.binary = bool(np.all(np.isin(s.Np.data, [0, 1])) and np.all(np.isin(s.Nm.data, [0, 1])))
        nrp, nrm = np.diff(s.Np.indptr), np.diff(s.Nm.indptr)
        s.unimolecular = bool(s.binary and np.all(nrp == 1) and np.all(nrm == 1))
        if s.unimolecular:
            s.tail = s.rp_idx[:, 0].copy(); s.head = s.rm_idx[:, 0].copy()
            assert np.all(s.tail != s.head)
        s.site_of = np.arange(s.M) % s.V
        s.species_of = np.arange(s.M) // s.V
        s.dense = s.H * s.M <= s.DENSE_MAX           # small hypergraphs: dense matmuls beat gather / scatter

    # ---------------------------------------------------------------- linear algebra
    @property
    def S(s):
        """dense stoichiometric matrix (M, H)"""
        return (s.Nm - s.Np).T.toarray()

    @property
    def Z(s):
        """orthonormal basis of ker S (cycle space), (H, z)"""
        if s._Z is None:
            s._Z = scipy.linalg.null_space(s.S)
        return s._Z

    @property
    def L(s):
        """orthonormal basis of ker S^T (conservation laws), (M, l)"""
        if s._L is None:
            s._L = scipy.linalg.null_space(s.S.T)
        return s._L

    def cycle_basis(s):
        """a basis of ker S (H, z), the cycles of the hypergraph; rational (mostly integer) entries when it is small"""
        return s.Z if s.H * s.M > 250_000 or s._Z is not None and s._Z.shape[1] == 0 else _rational_null(s.S)

    def conservation_laws(s):
        """a basis of ker S^T (M, l), the conserved moieties; rational (mostly integer) entries when it is small"""
        return s.L if s.H * s.M > 250_000 else _rational_null(s.S.T)

    def to_dict(s):
        return dict(nu_plus=s.Np, nu_minus=s.Nm, n_species=s.n_species, species_names=s.species_names, meta=s.meta)

    @classmethod
    def from_dict(cls, d):
        return cls(d['nu_plus'], d['nu_minus'], n_species=d['n_species'], species_names=d['species_names'], meta=d['meta'])

    def set_cycle_basis(s, Z):
        Z = scipy.linalg.orth(np.asarray(Z, np.float64))
        assert np.abs(s.S @ Z).max() < 1e-9, 'cycle basis is not in ker S'
        s._Z = Z
        s._torch = {}
        return s

    def on(s, device, dtype):
        """torch tensors of the index arrays and bases on (device, dtype), cached."""
        d = s._torch.get((device, dtype))
        if d is not None:
            return d
        key = (str(device), dtype)
        if key not in s._torch:
            T = lambda x, dt=dtype: torch.as_tensor(np.asarray(x), dtype=dt, device=device)
            L = lambda x: torch.as_tensor(np.asarray(x), dtype=torch.long, device=device)
            d = dict(rp_idx=L(s.rp_idx), rp_coef=T(s.rp_coef), rm_idx=L(s.rm_idx), rm_coef=T(s.rm_coef),
                     s_h=L(s.s_h), s_m=L(s.s_m), s_val=T(s.s_val),
                     cp_h=L(s.cp[0]), cp_m=L(s.cp[1]), cp_val=T(s.cp[2]),
                     cm_h=L(s.cm[0]), cm_m=L(s.cm[1]), cm_val=T(s.cm[2]),
                     typ=L(s.meta['typ']), odd=T(s.meta['odd'].astype(np.float64)),
                     site_of=L(s.site_of), species_of=L(s.species_of))
            if s.unimolecular:
                d['tail'] = L(s.tail); d['head'] = L(s.head)
            s._torch[key] = d
        d = s._torch[key]
        if s.dense and 'NpT' not in d:
            T = lambda x: torch.as_tensor(x, dtype=dtype, device=device)
            Sm = (s.Nm - s.Np).toarray()
            d['NpT'] = T(s.Np.T.toarray()); d['NmT'] = T(s.Nm.T.toarray())
            d['ST'] = T(Sm); d['S_'] = T(Sm.T.copy())
            d['Cp'] = T(np.maximum(-Sm, 0)); d['Cm'] = T(np.maximum(Sm, 0))
        if 'Z' not in d:
            d['Z'] = torch.as_tensor(s.Z, dtype=dtype, device=device)
        if 'L' not in d:
            d['L'] = torch.as_tensor(s.L, dtype=dtype, device=device)
        s._torch[(device, dtype)] = d                    # fast path for the exact (device, dtype) objects
        return d

    # ---------------------------------------------------------------- batched sparse products (torch)
    def dot(s, x, side):
        """<nu_side[h], x> for x (B, M) -> (B, H);  side = '+' (reactants) or '-' (products)."""
        t = s.on(x.device, x.dtype)
        if s.dense:
            return x @ (t['NpT'] if side == '+' else t['NmT'])
        idx, coef = (t['rp_idx'], t['rp_coef']) if side == '+' else (t['rm_idx'], t['rm_coef'])
        xe = torch.cat([x, torch.zeros_like(x[:, :1])], 1)
        return (xe[:, idx] * coef).sum(-1)

    def mass_action(s, c, side):
        """prod_m c_m^{nu_side[h, m]} for c (B, M) -> (B, H); exact zero when a reactant is absent."""
        t = s.on(c.device, c.dtype)
        idx, coef = (t['rp_idx'], t['rp_coef']) if side == '+' else (t['rm_idx'], t['rm_coef'])
        g = torch.cat([c, torch.ones_like(c[:, :1])], 1)[:, idx]
        if not s.binary:
            g = g ** coef
        return g.prod(-1) if g.shape[-1] > 1 else g[..., 0]

    def apply_S(s, J):
        """S J for J (B, H) -> (B, M)"""
        t = s.on(J.device, J.dtype)
        if s.dense:
            return J @ t['ST']
        return torch.zeros(J.shape[0], s.M, dtype=J.dtype, device=J.device).index_add(1, t['s_m'], J[:, t['s_h']] * t['s_val'])

    def apply_ST(s, y):
        """S^T y for y (B, M) -> (B, H)"""
        t = s.on(y.device, y.dtype)
        if s.dense:
            return y @ t['S_']
        return torch.zeros(y.shape[0], s.H, dtype=y.dtype, device=y.device).index_add(1, t['s_h'], y[:, t['s_m']] * t['s_val'])

    def consumption(s, a, b):
        """net consumption of every component, sum_h [nu+ - nu-]_+ a_h + [nu- - nu+]_+ b_h (catalysts excluded), (B, M)"""
        t = s.on(a.device, a.dtype)
        if s.dense:
            return a @ t['Cp'] + b @ t['Cm']
        out = torch.zeros(a.shape[0], s.M, dtype=a.dtype, device=a.device)
        out = out.index_add(1, t['cp_m'], a[:, t['cp_h']] * t['cp_val'])
        return out.index_add(1, t['cm_m'], b[:, t['cm_h']] * t['cm_val'])

    def project_cycles(s, x):
        """orthogonal projection of a hyperedge field x (B, H) onto ker S"""
        Z = s.on(x.device, x.dtype)['Z']
        return (x @ Z) @ Z.T

    def conserved(s, c):
        """moiety coordinates c @ L, (B, l)"""
        return c @ s.on(c.device, c.dtype)['L']

    def chemostat_force(s, mu_chem):
        """drive on every hyperedge from the chemical potentials of the chemostatted species,
        force_h = <nu+_h - nu-_h, mu_Y> over the chemostats Y.  mu_chem: (B, n_chem) uniform in space or
        (B, n_chem * V) per site.  Returns (B, H)."""
        Y = s.meta.get('chem_S')
        assert Y is not None, 'no chemostatted species in this hypergraph'
        nC = len(s.meta['chemostats'])
        if mu_chem.shape[1] == nC and s.V > 1:
            mu_chem = mu_chem.repeat_interleave(s.V, 1)
        Yt = torch.as_tensor(Y.toarray(), dtype=mu_chem.dtype, device=mu_chem.device)      # (H, nC * V) = nu- - nu+
        return -mu_chem @ Yt.T

    # ---------------------------------------------------------------- builders
    @classmethod
    def reaction_transport(cls, species, reactions, edges=None, n_sites=None, diffusing=(), chemostats=(),
                           grid_n=None, **kw):
        """Reaction-transport hypergraph: a reaction network copied at every site of a spatial graph, plus transport
        of selected species along the graph edges.
          species     names of all species, including chemostatted ones
          reactions   list of (reactants, products), each a dict {species: coefficient}; many-to-many and
                      coefficients > 1 are allowed, e.g. ({'A': 1, 'X': 2}, {'X': 3})
          edges       (E, 2) directed (tail, head) site pairs; grid_n = n uses the n x n periodic grid instead
          diffusing   species that move along the edges (each one gets its own transport hyperedges)
          chemostats  species held at fixed chemical potential by a reservoir.  They are removed from the state and
                      from S, and their potentials enter as a drive through chemostat_force().
        Hyperedge order: reaction r at site v -> r * V + v, then transport of the d-th diffusing species along
        edge e -> R * V + d * E + e.  State order: the d-th non-chemostatted species at site v -> d * V + v.
        meta: typ (reaction index, or R + d for transport), odd (transport), site (reactions), et / eh (transport)."""
        if grid_n is not None:
            edges, n_sites = grid_edge_list(grid_n), grid_n * grid_n
        edges = np.zeros((0, 2), dtype=np.int64) if edges is None else np.asarray(edges, dtype=np.int64)
        V = int(n_sites if n_sites is not None else (edges.max() + 1 if len(edges) else 1))
        state = [x for x in species if x not in chemostats]
        sidx = {nm: i for i, nm in enumerate(state)}
        cidx = {nm: i for i, nm in enumerate(chemostats)}
        assert all(d in sidx for d in diffusing), 'diffusing species must be part of the state'
        R, E, Dn, NS, NC = len(reactions), len(edges), len(diffusing), len(state), len(chemostats)
        H = R * V + Dn * E
        trip = dict(p=([], [], []), m=([], [], []), y=([], [], []), yp=([], [], []), ym=([], [], []))   # (row, col, value), summed below

        def add(key, rows, cols, val):
            trip[key][0].append(rows); trip[key][1].append(cols); trip[key][2].append(np.full(len(rows), float(val)))
        typ = np.zeros(H, dtype=np.int64); site = np.full(H, -1); et = np.full(H, -1); eh = np.full(H, -1)
        v = np.arange(V)
        for r, (lhs, rhs) in enumerate(reactions):
            rows = r * V + v
            for side, key, sgn in ((lhs, 'p', -1.0), (rhs, 'm', 1.0)):
                for nm, k in side.items():
                    if nm in cidx:
                        add('y', rows, cidx[nm] * V + v, sgn * k); add('yp' if sgn < 0 else 'ym', rows, cidx[nm] * V + v, k)
                    else:
                        add(key, rows, sidx[nm] * V + v, k)
            typ[rows] = r; site[rows] = v
        for d, nm in enumerate(diffusing):
            rows = R * V + d * E + np.arange(E)
            add('p', rows, sidx[nm] * V + edges[:, 0], 1); add('m', rows, sidx[nm] * V + edges[:, 1], 1)
            typ[rows] = R + d; et[rows] = edges[:, 0]; eh[rows] = edges[:, 1]

        def mat(key, ncol):
            r, c, x = trip[key]
            if not r:
                return sp.csr_matrix((H, ncol))
            return sp.coo_matrix((np.concatenate(x), (np.concatenate(r), np.concatenate(c))), shape=(H, ncol)).tocsr()
        Np, Nm, Yc = mat('p', NS * V), mat('m', NS * V), mat('y', max(NC, 1) * V)
        Yp, Ym = mat('yp', max(NC, 1) * V), mat('ym', max(NC, 1) * V)
        meta = dict(typ=typ, odd=typ >= R, site=site, et=et, eh=eh, n_reactions=R, diffusing=list(diffusing))
        if grid_n is not None:
            meta['grid_n'] = grid_n
            # harmonic 1-cochains of the torus for every diffusing species: all its horizontal / all its vertical edges
            Hm = np.zeros((H, 2 * Dn))
            for d in range(Dn):
                Hm[R * V + d * E + np.arange(V), 2 * d] = 1.0
                Hm[R * V + d * E + V + np.arange(V), 2 * d + 1] = 1.0
            meta['harmonic'] = Hm
        if NC:
            meta['chemostats'] = list(chemostats); meta['chem_S'] = Yc
            meta['chem_Np'], meta['chem_Nm'] = Yp, Ym          # chemostat coefficients on the reactant / product side
        return cls(Np, Nm, n_species=NS, species_names=state, meta=meta, **kw)

    @classmethod
    def from_strings(cls, reactions, chemostats=(), diffusing=(), edges=None, n_sites=None, grid_n=None, species=None, **kw):
        """the reaction-transport hypergraph from reactions written as strings, 'A + 2 X <-> 3 X'; species in order of first
        appearance unless given.  Each reaction is one hyperedge class per site; each diffusing species one transport class."""
        rx, names = [], list(species or [])
        for r in reactions:
            lhs, rhs = re.split(r'<->|<=>|->|=', r)
            sides = []
            for side in (lhs, rhs):
                d = {}
                for term in side.split('+'):
                    term = term.strip()
                    if not term or term == '0':
                        continue
                    m = re.fullmatch(r'(\d+)?\s*\*?\s*(\S+)', term)
                    d[m.group(2)] = d.get(m.group(2), 0) + int(m.group(1) or 1)
                    if m.group(2) not in names:
                        names.append(m.group(2))
                sides.append(d)
            rx.append(tuple(sides))
        return cls.reaction_transport(names, rx, edges=edges, n_sites=n_sites, diffusing=diffusing, chemostats=chemostats, grid_n=grid_n, **kw)

    @classmethod
    def from_stoichiometry(cls, nu_plus, nu_minus=None, S=None, species=None, classes=None, site_volume=None, edge_weight=None, **kw):
        """one site: nu_plus, nu_minus (H, M), or S (M, H) with the reactants taken as the negative entries.  classes (H,): the
        hyperedge classes that share log omega (default: one class per hyperedge)."""
        if S is not None:
            S = np.asarray(S, np.float64)
            nu_plus, nu_minus = np.maximum(-S, 0).T, np.maximum(S, 0).T
        Np = sp.csr_matrix(nu_plus, dtype=np.float64)
        H, M = Np.shape
        meta = dict(typ=np.arange(H) if classes is None else np.asarray(classes), site=np.zeros(H, dtype=np.int64))
        if site_volume is not None:
            meta['site_volume'] = np.asarray(site_volume, np.float64)
        if edge_weight is not None:
            meta['edge_weight'] = np.asarray(edge_weight, np.float64)
        return cls(Np, nu_minus, n_species=M, species_names=list(species) if species else None, meta=meta, **kw)

    @classmethod
    def from_grid(cls, n, n_species=1, **kw):
        """transport on the n x n periodic grid"""
        h = cls.from_graph(grid_edge_list(n), n * n, n_species=n_species, **kw)
        h.meta['grid_n'] = n
        return h

    @classmethod
    def from_mesh(cls, edges, n_nodes, site_volume=None, edge_weight=None, n_species=1, **kw):
        """transport on a mesh: cell volumes (relative to their mean) and the geometric edge weights"""
        h = cls.from_graph(edges, n_nodes, n_species=n_species, **kw)
        if site_volume is not None:
            h.meta['site_volume'] = np.asarray(site_volume, np.float64)
        if edge_weight is not None:
            h.meta['edge_weight'] = np.tile(np.asarray(edge_weight, np.float64), n_species)
        return h

    @classmethod
    def from_points(cls, points, n_species=1, **kw):
        """transport between the Voronoi cells of a point cloud on the unit torus: edge weight face length / (distance sqrt(V_i V_j))"""
        from scipy.spatial import Voronoi
        P = np.asarray(points, np.float64)
        V = len(P)
        offs = [(dx, dy) for dx in (-1, 0, 1) for dy in (-1, 0, 1)]
        tiled = np.concatenate([P + np.array(o) for o in offs])
        cen = offs.index((0, 0))
        vor = Voronoi(tiled)
        area = np.zeros(V)
        for i in range(V):
            poly = vor.vertices[vor.regions[vor.point_region[cen * V + i]]]
            c0 = poly.mean(0)
            poly = poly[np.argsort(np.arctan2(poly[:, 1] - c0[1], poly[:, 0] - c0[0]))]
            area[i] = 0.5 * abs(np.dot(poly[:, 0], np.roll(poly[:, 1], -1)) - np.dot(poly[:, 1], np.roll(poly[:, 0], -1)))
        faces = {}
        for (p, q), rv in zip(vor.ridge_points, vor.ridge_vertices):
            if (p // V != cen and q // V != cen) or -1 in rv:
                continue
            u, w = (p, q) if p // V == cen else (q, p)
            key = tuple(sorted((u % V, w % V)))
            faces[key] = np.linalg.norm(vor.vertices[rv[1]] - vor.vertices[rv[0]]) / np.linalg.norm(tiled[w] - tiled[u])
        keys = sorted(faces)
        E = np.array(keys)
        w = np.array([faces[k] for k in keys]) / np.sqrt(area[E[:, 0]] * area[E[:, 1]])
        return cls.from_mesh(E, V, site_volume=area * V, edge_weight=w, n_species=n_species, **kw)

    @classmethod
    def from_reactions(cls, reactions, species, n_sites=1, chemostats=(), **kw):
        """well-mixed compartments: every reaction copied to each of n_sites sites, no transport."""
        return cls.reaction_transport(species, reactions, n_sites=n_sites, chemostats=chemostats, **kw)

    @classmethod
    def from_graph(cls, edges, n_nodes, n_species=1, **kw):
        """transport of each species along directed edges (tail, head) of a graph; hyperedge order (species, edge)."""
        E = np.asarray(edges, dtype=np.int64)
        m = len(E)
        H, M = n_species * m, n_species * n_nodes
        rows = np.arange(H)
        sp_ = np.repeat(np.arange(n_species), m)
        tail = sp_ * n_nodes + np.tile(E[:, 0], n_species)
        head = sp_ * n_nodes + np.tile(E[:, 1], n_species)
        Np = sp.csr_matrix((np.ones(H), (rows, tail)), shape=(H, M))
        Nm = sp.csr_matrix((np.ones(H), (rows, head)), shape=(H, M))
        meta = dict(typ=sp_, odd=np.ones(H, bool), et=np.tile(E[:, 0], n_species), eh=np.tile(E[:, 1], n_species))
        return cls(Np, Nm, n_species=n_species, meta=meta, **kw)
