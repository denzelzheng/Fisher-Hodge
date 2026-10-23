import numpy as np, json, itertools
from hodge_lib import *
rng = np.random.default_rng(2024)

def all_simple_cycles(adj, maxlen=6):
    n = len(adj); cyc = set()
    def dfs(start, path):
        u = path[-1]
        for v in np.nonzero(adj[u])[0]:
            if v == start and len(path) >= 3:
                c = tuple(path); 
                # canonical form
                k = c.index(min(c)); c = c[k:]+c[:k]
                if c[1] > c[-1]: c = (c[0],)+tuple(reversed(c[1:]))
                cyc.add(c)
            elif v > start and v not in path and len(path) < maxlen:
                dfs(start, path+[v])
    for s in range(n): dfs(s, [s])
    return cyc

def kolmogorov_all(Q, adj, cycles, tol=1e-9):
    worst = 0
    for c in cycles:
        lf = sum(np.log(Q[c[k], c[(k+1) % len(c)]]) for k in range(len(c)))
        lb = sum(np.log(Q[c[(k+1) % len(c)], c[k]]) for k in range(len(c)))
        worst = max(worst, abs(lf-lb))
    return worst

def random_Q(E, n, rng, reversible=False, sd=1.0):
    B = rng.normal(0, sd, len(E))              # log symmetric part
    c = np.zeros(len(E)) if not reversible else None
    Q = np.zeros((n, n)); i, j = E[:, 0], E[:, 1]
    if reversible:
        u = rng.normal(0, sd, n)               # reversible: log ratio = u_i-u_j
        d = u[i]-u[j]
    else:
        d = rng.normal(0, sd, len(E))
    Q[i, j] = np.exp(B + d/2); Q[j, i] = np.exp(B - d/2)
    Q[np.arange(n), np.arange(n)] = -Q.sum(1)
    return Q

cases = []
for n in range(4, 11): cases.append(('complete', n, make_adj('complete', n, rng)))
for n in (8, 12, 16): cases.append(('ER p=0.4', n, make_adj('er', n, rng, 0.4)))
cases.append(('cycle', 5, make_adj('cycle', 5, rng)))
cases.append(('cycle', 12, make_adj('cycle', 12, rng)))
for n in (6, 12, 25): cases.append(('tree', n, make_adj('tree', n, rng)))
cases.append(('grid 4x4', 16, make_adj('grid', 4, rng)))

rows = []
for name, n, adj in cases:
    E = edges_of(adj); m = len(E); d0 = incidence(E, n)
    beta1 = m - n + 1                           # connected
    rank_d0 = np.linalg.matrix_rank(d0)
    # rank of {log(Q_ij/Q_ji)} cochains modulo exact part
    K = beta1 + 12 + 5
    C = np.stack([log_ratio_cochain(random_Q(E, n, rng), E) for _ in range(K)], 1)     # m x K
    Pc = np.eye(m) - d0 @ np.linalg.pinv(d0)          # Euclidean projection onto coexact complement of im d0
    H = Pc @ C
    sv = np.linalg.svd(H, compute_uv=False)
    rank_mod_exact = int(np.sum(sv > 1e-9*max(sv.max(), 1))) if sv.max() > 1e-12 else 0
    rank_aug = np.linalg.matrix_rank(np.hstack([C, d0])) - rank_d0       # dim (span C + im d0)/im d0
    # Kolmogorov / harmonic equivalence on random-irreversible and constructed-reversible Q
    cyc = fundamental_cycles(E, n)
    allc = all_simple_cycles(adj, maxlen=min(n, 6)) if (n <= 8 and m <= 30) else None
    eq_ok = True; stats = []
    for rev in (True, False):
        for _ in range(30):
            Q = random_Q(E, n, rng, reversible=rev)
            c = log_ratio_cochain(Q, E)
            kol = float(np.max(np.abs(cyc @ c))) if beta1 > 0 else 0.0     # max |cycle log-ratio| over fundamental cycles
            harm = float(np.linalg.norm(Pc @ c))
            pi = stationary(Q); a, b, J, f, ell = edge_quantities(pi, Q, E)
            Jn = float(np.max(np.abs(J))); sig = float(np.sum(J*f))
            kol_all = kolmogorov_all(Q, adj, allc) if allc else None
            is_rev_kol = kol < 1e-9; is_rev_harm = harm < 1e-9; is_rev_J = Jn < 1e-12
            expect = rev or beta1 == 0
            ok = (is_rev_kol == is_rev_harm == is_rev_J == expect) and (kol_all is None or ((kol_all < 1e-9) == expect))
            eq_ok &= ok; stats.append((rev, kol, harm, Jn, sig, kol_all))
    rows.append(dict(graph=name, n=n, V=n, E=m, beta1=beta1, beta1_formula_complete=(n-1)*(n-2)//2 if name == 'complete' else None,
                     rank_mod_exact=rank_mod_exact, rank_aug=int(rank_aug), rank_cycle_proj=int(np.linalg.matrix_rank(Pc)),
                     fund_cycles=int(len(cyc)), n_simple_cycles=(len(allc) if allc else None),
                     equiv_all_ok=bool(eq_ok),
                     max_harm_if_rev=max([s[2] for s in stats if s[0]], default=0), min_harm_if_irrev=min([s[2] for s in stats if not s[0]], default=None) if beta1>0 else None,
                     max_J_if_rev=max([s[3] for s in stats if s[0]], default=0), min_J_if_irrev=min([s[3] for s in stats if not s[0]], default=None) if beta1>0 else None,
                     all_irrev_irreversible=(all(s[1] > 1e-9 for s in stats if not s[0]) if beta1 > 0 else True)))
    print(rows[-1]['graph'], n, m, beta1, rank_mod_exact, rank_aug, eq_ok, flush=True)

# codimension of detailed-balance manifold: Jacobian rank of Kolmogorov constraint map  Q-params(2m) -> cycle sums (beta1)
# also: C5 explicit demo
n = 5; adj = make_adj('cycle', 5, rng); E = edges_of(adj); d0 = incidence(E, n)
demo = []
for trial in range(8):
    Q = random_Q(E, n, rng, reversible=(trial % 2 == 0))
    if trial >= 6:   # make a Kolmogorov-satisfying but non-uniform Q by hand: perturb one rate and fix it back
        pass
    # 5-cycle product ratio
    fw = np.prod([Q[k, (k+1) % 5] for k in range(5)]); bw = np.prod([Q[(k+1) % 5, k] for k in range(5)])
    c = log_ratio_cochain(Q, E); Pc = np.eye(len(E)) - d0 @ np.linalg.pinv(d0)
    pi = stationary(Q); a, b, J, f, ell = edge_quantities(pi, Q, E)
    demo.append(dict(rev_construct=(trial % 2 == 0), prod_fwd_over_bwd=float(fw/bw), log_ratio=float(np.log(fw/bw)),
                     harmonic_norm=float(np.linalg.norm(Pc @ c)), harmonic_per_edge=float(np.linalg.norm(Pc @ c)/np.sqrt(5)),
                     J_ss_max=float(np.max(np.abs(J))), epr=float(np.sum(J*f))))
# analytic check on C5: harmonic comp = (A/5) * ones, |.|=|A|/sqrt(5) where A = sum of c along cycle
for d in demo: d['pred_harm_norm_|A|/sqrt5'] = abs(d['log_ratio'])/np.sqrt(5)
json.dump(dict(rows=rows, c5=demo), open('runs/e2.json', 'w'), indent=1, default=lambda x: None if x is None else float(x))
