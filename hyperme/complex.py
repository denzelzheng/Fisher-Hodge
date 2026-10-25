"""Hypergraphs (stoichiometric complexes) for HyperME.

A hypergraph is given by two non-negative integer matrices nu_plus, nu_minus of shape (H, M): hyperedge h turns
the multiset nu_plus[h] of state components into nu_minus[h].  State components are species x sites, flattened
species-major (index m = s * V + v).  From (nu_plus, nu_minus) we build

    S       = (nu_minus - nu_plus)^T      (M, H)    dc/dt = S J
    Z       orthonormal basis of ker S    (H, z)    cycle space, where the drive A lives
    L       orthonormal basis of ker S^T  (M, l)    conserved moieties, c @ L is constant in time

and the gather / scatter index arrays used by the model.  Hypergraphs with H * M <= DENSE_MAX use dense matmuls
instead, which are faster at that size; larger ones never touch a dense (H, M) matrix in the forward pass.  Transport of one species along a graph edge i -> j is the hyperedge {X@i} -> {X@j}.

Builders: Hypergraph.from_reactions, Hypergraph.from_graph, Hypergraph.grid_transport (periodic grid, edge order of
fisher_hodge/common.py), Hypergraph.crn_phosphorylation_grid (benchmark C1, hyperedge order of hyper/hyperme.py).
On the periodic grid the cycle basis can also be built from the face coboundary d1^T plus the two harmonic
1-cochains ('curl' basis); the d0 / d1 operators are the roll implementations of fisher_hodge/common.py.
"""
import os
import importlib.util
import numpy as np
import scipy.linalg
import scipy.sparse as sp
import torch

_HERE = os.path.dirname(os.path.abspath(__file__))
_COMMON_PATH = os.path.join(os.path.dirname(_HERE), 'fisher_hodge', 'common.py')


def _load_common():
    spec = importlib.util.spec_from_file_location('_fh_common', _COMMON_PATH)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


common = _load_common()          # grad_e (d0), div_T (d0^T), curl_f (d1), dual_dpsi (d1^T), mean_e, ...


# ------------------------------------------------------------------------------------------ grid operators
class GridOps:
    """Cochain operators on the n x n periodic grid (roll implementation of fisher_hodge/common.py).
    Nodes (..., n, n); edges (..., 2, n, n) with [0] = (a,b)->(a,b+1), [1] = (a,b)->(a+1,b); faces (..., n, n)
    with face (a,b) the square whose lower-left node is (a,b).  Flat hyperedge index = dir * n^2 + a * n + b."""
    d0 = staticmethod(common.grad_e)          # node -> edge, (d0 u)_e = u_head - u_tail
    d0T = staticmethod(common.div_T)          # edge -> node, = S for single-species transport
    d1 = staticmethod(common.curl_f)          # edge -> face, counter-clockwise circulation
    d1T = staticmethod(common.dual_dpsi)      # face -> edge, adjoint of d1
    mean_e = staticmethod(common.mean_e)

    @staticmethod
    def node_to_corner(u):
        return 0.25 * (u + u.roll(-1, -1) + u.roll(-1, -2) + u.roll(-1, -1).roll(-1, -2))

    @staticmethod
    def stream_from_dpsi(dp):
        """invert d1T on the torus: edge (...,2,n,n) -> corner stream function (...,n,n) with mean zero."""
        dh, dv = dp[..., 0, :, :], dp[..., 1, :, :]
        z = torch.zeros_like(dv[..., :1, :1])
        row0 = torch.cat([z[..., 0, :], -torch.cumsum(dv[..., 0, 1:], -1)], -1)
        col = torch.cat([torch.zeros_like(dh[..., :1, :]), torch.cumsum(dh[..., 1:, :], -2)], -2)
        P = row0.unsqueeze(-2) + col
        return P - P.mean((-1, -2), keepdim=True)

    @staticmethod
    def flat_e(f):
        """edge (B,2,n,n) -> (B, 2 n^2)"""
        return f.reshape(*f.shape[:-3], -1)

    @staticmethod
    def grid_e(x, n):
        return x.reshape(*x.shape[:-1], 2, n, n)


def grid_edge_list(n):
    """(2 n^2, 2) array of (tail, head) node indices in the flat hyperedge order of GridOps."""
    idx = np.arange(n * n).reshape(n, n)
    tail = np.concatenate([idx.ravel(), idx.ravel()])
    head = np.concatenate([np.roll(idx, -1, 1).ravel(), np.roll(idx, -1, 0).ravel()])
    return np.stack([tail, head], 1)


def grid_curl_basis(n, dtype=torch.float64):
    """Cycle space of single-species transport on the n x n torus from d1^T (faces) and the two harmonic
    1-cochains (all horizontal edges, all vertical edges).  Returns an orthonormal (2 n^2, n^2 + 1) matrix."""
    V = n * n
    I = torch.eye(V, dtype=dtype).reshape(V, n, n)
    curl = GridOps.flat_e(GridOps.d1T(I)).T.numpy()                 # (2V, V), rank V - 1
    harm = np.zeros((2 * V, 2)); harm[:V, 0] = 1.0; harm[V:, 1] = 1.0
    return scipy.linalg.orth(np.concatenate([curl, harm], 1))


# ------------------------------------------------------------------------------------------ hypergraph
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

    # ---------------------------------------------------------------- builders
    @classmethod
    def from_reactions(cls, reactions, species, n_sites=1, **kw):
        """reactions: list of (reactants, products) dicts {species name: coefficient}, applied at one site.
        With n_sites > 1 every reaction is copied to every site (well-mixed compartments, no transport)."""
        sidx = {nm: i for i, nm in enumerate(species)}
        NS, V = len(species), n_sites
        H = len(reactions) * V
        Np = sp.lil_matrix((H, NS * V)); Nm = sp.lil_matrix((H, NS * V))
        typ = np.zeros(H, dtype=np.int64)
        for r, (lhs, rhs) in enumerate(reactions):
            for v in range(V):
                h = r * V + v
                for nm, k in lhs.items():
                    Np[h, sidx[nm] * V + v] += k
                for nm, k in rhs.items():
                    Nm[h, sidx[nm] * V + v] += k
                typ[h] = r
        return cls(Np, Nm, n_species=NS, species_names=list(species), meta=dict(typ=typ, site=np.tile(np.arange(V), len(reactions))), **kw)

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

    @classmethod
    def grid_transport(cls, n, n_species=1, basis='curl'):
        """periodic n x n grid, edge order of fisher_hodge/common.py.  basis='curl' builds ker S from d1^T plus the
        harmonic cochains (cheap for large n), basis='svd' uses scipy null_space."""
        hg = cls.from_graph(grid_edge_list(n), n * n, n_species=n_species)
        hg.meta['grid_n'] = n
        if basis == 'curl':
            Zs = grid_curl_basis(n)
            m = Zs.shape[0]
            Z = np.zeros((n_species * m, n_species * Zs.shape[1]))
            for i in range(n_species):
                Z[i * m:(i + 1) * m, i * Zs.shape[1]:(i + 1) * Zs.shape[1]] = Zs
            hg._Z = Z
            L = np.zeros((n_species * n * n, n_species))
            for i in range(n_species):
                L[i * n * n:(i + 1) * n * n, i] = 1.0 / n
            hg._L = L
        return hg

    @classmethod
    def crn_phosphorylation_grid(cls, n):
        """Benchmark C1 (hyper/gen_crn.py): species X, Xp, E on an n x n torus, hyperedges
        [R1_v: X+E -> Xp+E] (V), [R2_v: Xp -> X] (V), [TX_e: X@i -> X@j] (2V), [TXp_e: Xp@i -> Xp@j] (2V)."""
        V = n * n; E2 = 2 * V; H = 2 * V + 2 * E2; NS = 3
        Ed = grid_edge_list(n); tail, head = Ed[:, 0], Ed[:, 1]
        Np = np.zeros((H, NS * V)); Nm = np.zeros((H, NS * V))
        v = np.arange(V)
        r1 = np.arange(V); r2 = V + v; tx = 2 * V + np.arange(E2); txp = 2 * V + E2 + np.arange(E2)
        X, XP, EN = 0, 1, 2
        Np[r1, X * V + v] = 1; Np[r1, EN * V + v] = 1; Nm[r1, XP * V + v] = 1; Nm[r1, EN * V + v] = 1
        Np[r2, XP * V + v] = 1; Nm[r2, X * V + v] = 1
        Np[tx, X * V + tail] = 1; Nm[tx, X * V + head] = 1
        Np[txp, XP * V + tail] = 1; Nm[txp, XP * V + head] = 1
        typ = np.zeros(H, dtype=np.int64); typ[r2], typ[tx], typ[txp] = 1, 2, 3
        site = np.full(H, -1); site[r1] = v; site[r2] = v
        et = np.full(H, -1); eh = np.full(H, -1); et[tx] = tail; eh[tx] = head; et[txp] = tail; eh[txp] = head
        zloc = np.zeros((V, H)); zloc[v, r1] = 1; zloc[v, r2] = 1          # local R1 + R2 cycle at every site
        meta = dict(typ=typ, odd=typ >= 2, site=site, et=et, eh=eh, grid_n=n, zloc=zloc, r1=r1, r2=r2, tx=tx, txp=txp)
        return cls(Np, Nm, n_species=NS, meta=meta, species_names=['X', 'Xp', 'E'])
