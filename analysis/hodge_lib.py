import numpy as np
import scipy.linalg as sla

# ---------- graphs ----------
def connected(adj):
    n = len(adj); seen = {0}; st = [0]
    while st:
        u = st.pop()
        for v in np.nonzero(adj[u])[0]:
            if v not in seen:
                seen.add(v); st.append(v)
    return len(seen) == n

def make_adj(kind, n, rng, p=0.3):
    if kind == 'complete':
        adj = np.ones((n, n), bool); np.fill_diagonal(adj, False); return adj
    if kind == 'er':
        while True:
            U = np.triu(rng.random((n, n)) < p, 1); adj = U | U.T
            if connected(adj): return adj
    if kind == 'cycle':
        adj = np.zeros((n, n), bool)
        for i in range(n): adj[i, (i+1) % n] = adj[(i+1) % n, i] = True
        return adj
    if kind == 'tree':
        adj = np.zeros((n, n), bool)
        for i in range(1, n):
            j = rng.integers(0, i); adj[i, j] = adj[j, i] = True
        return adj
    if kind == 'grid':   # n x n/ grid of side n
        N = n*n; adj = np.zeros((N, N), bool)
        for a in range(n):
            for b in range(n):
                u = a*n+b
                if a+1 < n: adj[u, u+n] = adj[u+n, u] = True
                if b+1 < n: adj[u, u+1] = adj[u+1, u] = True
        return adj
    raise ValueError(kind)

def edges_of(adj):
    iu, ju = np.nonzero(np.triu(adj, 1))
    return np.stack([iu, ju], 1)

def incidence(E, n):
    m = len(E); d0 = np.zeros((m, n))
    d0[np.arange(m), E[:, 0]] = 1.0     # (d0 phi)_e = phi_i - phi_j,  e=(i<j)
    d0[np.arange(m), E[:, 1]] = -1.0
    return d0

def cycle_basis_mat(d0):
    """orthonormal basis of ker d0^T (Euclidean cycle space), shape (m, beta1)"""
    return sla.null_space(d0.T)

def fundamental_cycles(E, n):
    """spanning tree (BFS) -> fundamental cycles as signed edge vectors (beta1, m), orientation along e=(i<j) is +1 for i->j"""
    m = len(E); adjl = [[] for _ in range(n)]
    for k, (i, j) in enumerate(E):
        adjl[i].append((j, k, +1)); adjl[j].append((i, k, -1))
    parent = {0: None}; order = [0]; tree = set()
    for u in order:
        for v, k, s in adjl[u]:
            if v not in parent:
                parent[v] = (u, k, s); tree.add(k); order.append(v)
    # path from root: signed edge vector for walking root->v
    def path(v):
        vec = np.zeros(m)
        while parent[v] is not None:
            u, k, s = parent[v]; vec[k] += s   # walking u->v along sign s (s=+1 if edge oriented u->v)
            v = u
        return vec
    P = {v: path(v) for v in parent}
    cyc = []
    for k, (i, j) in enumerate(E):
        if k in tree: continue
        # cycle: root->i, edge i->j (+1), then j->root (reverse path)
        c = P[i].copy(); c[k] += 1; c -= P[j]
        cyc.append(c)
    return np.array(cyc).reshape(-1, m)

# ---------- Markov chain ----------
def build_Q(E, n, B, U, A, eps):
    """Q_ij = exp(-B_ij + (U_i-U_j)/2 + eps*A_ij/2); B per edge, A per edge (antisym along i<j)."""
    Q = np.zeros((n, n))
    i, j = E[:, 0], E[:, 1]
    Q[i, j] = np.exp(-B + (U[i]-U[j])/2 + eps*A/2)
    Q[j, i] = np.exp(-B + (U[j]-U[i])/2 - eps*A/2)
    Q[np.arange(n), np.arange(n)] = -Q.sum(1)
    return Q

def stationary(Q):
    n = len(Q)
    M = np.vstack([Q.T, np.ones(n)])
    b = np.zeros(n+1); b[-1] = 1
    pi = np.linalg.lstsq(M, b, rcond=None)[0]
    return pi

def sinhc(x):
    out = np.ones_like(x); m = np.abs(x) > 1e-8
    out[m] = np.sinh(x[m])/x[m]; return out

def edge_quantities(rho, Q, E):
    i, j = E[:, 0], E[:, 1]
    a = rho[i]*Q[i, j]; b = rho[j]*Q[j, i]
    f = np.log(a)-np.log(b)
    ell = np.sqrt(a*b)*sinhc(f/2)
    J = a-b
    return a, b, J, f, ell

def hodge(f, w, d0):
    n = d0.shape[1]
    Lap = d0.T @ (w[:, None]*d0)
    rhs = d0.T @ (w*f)
    Lap = Lap + (w.mean()/n)*np.ones((n, n))     # gauge fix: sum(phi)=0
    phi = np.linalg.solve(Lap, rhs)
    # one step of iterative refinement
    r = rhs - (d0.T @ (w[:, None]*d0)) @ phi
    phi += np.linalg.solve(Lap, r)
    g = d0 @ phi
    return g, f-g

def energy(x, w):
    return float(np.sum(w*x*x))

def log_ratio_cochain(Q, E):
    return np.log(Q[E[:, 0], E[:, 1]]) - np.log(Q[E[:, 1], E[:, 0]])
