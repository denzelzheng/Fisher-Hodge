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
