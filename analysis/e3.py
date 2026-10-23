import numpy as np, json, sys
from hodge_lib import *

def gillespie(Q, njump, rng, x0):
    n = len(Q); rates = -np.diag(Q); P = Q.copy(); np.fill_diagonal(P, 0); P = P/rates[:, None]
    cum = np.cumsum(P, 1)
    N = np.zeros((n, n)); tau = np.zeros(n)
    u = rng.random(njump); hold = rng.exponential(size=njump)
    x = x0
    for k in range(njump):
        tau[x] += hold[k]/rates[x]
        y = int(np.searchsorted(cum[x], u[k], side='right')); y = min(y, n-1)
        N[x, y] += 1; x = y
    return N, tau

def estimators(N, tau, E, d0, pseudo=0.5):
    T = tau.sum()
    i, j = E[:, 0], E[:, 1]
    # (1) direct count (Lynn et al.): sum over ordered pairs with both counts > 0
    s = 0.0
    for (a, b) in zip(i, j):
        nab, nba = N[a, b], N[b, a]
        if nab > 0 and nba > 0: s += (nab-nba)*np.log(nab/nba)
    lynn = s/T
    # model-based: Qhat with pseudo counts, rho_hat = occupancy fraction
    n = len(tau)
    Qh = np.zeros((n, n)); Qh[i, j] = (N[i, j]+pseudo)/tau[i]; Qh[j, i] = (N[j, i]+pseudo)/tau[j]
    Qh[np.arange(n), np.arange(n)] = -Qh.sum(1)
    rho = tau/T
    a, b, J, f, ell = edge_quantities(rho, Qh, E)
    gL, cL = hodge(f, ell, d0); gE, cE = hodge(f, np.ones_like(ell), d0)
    out = dict(lynn=lynn, plug=float(np.sum(J*f)), exL=energy(gL, ell), hkL=energy(cL, ell),
               exE=energy(gE, ell), hkE=energy(cE, ell))
    # also: rho = stationary of Qhat (excess=0 by construction)
    pih = stationary(Qh)
    a2, b2, J2, f2, ell2 = edge_quantities(pih, Qh, E)
    g2, c2 = hodge(f2, ell2, d0)
    out['plug_stat'] = float(np.sum(J2*f2)); out['hkL_stat'] = energy(c2, ell2)
    gE2, cE2 = hodge(f2, np.ones_like(ell2), d0)
    out['exE_stat'] = energy(gE2, ell2); out['hkE_stat'] = energy(cE2, ell2)
    # cycle affinities via integrals of cyc parts along fundamental cycles
    cyc = fundamental_cycles(E, n)
    out['aff_L'] = (cyc @ cL).tolist(); out['aff_E'] = (cyc @ cE).tolist(); out['aff_f'] = (cyc @ f).tolist()
    # edge-wise cyclic field
    out['cL_vec'] = cL.tolist(); out['cE_vec'] = cE.tolist()
    return out

def run_system(name, kind, n, eps, seed, Ts, reps, p=0.5):
    rng = np.random.default_rng(seed)
    adj = make_adj(kind, n, rng, p); E = edges_of(adj); d0 = incidence(E, n)
    B = rng.uniform(0, 2, len(E)); U = rng.standard_normal(n)
    Z = cycle_basis_mat(d0); A = Z @ rng.standard_normal(Z.shape[1]); A /= np.sqrt(np.mean(A**2))
    Q = build_Q(E, n, B, U, A, eps); pi = stationary(Q)
    a, b, J, f, ell = edge_quantities(pi, Q, E)
    sig = float(np.sum(J*f)); gL, cL = hodge(f, ell, d0); gE, cE = hodge(f, np.ones_like(ell), d0)
    cyc = fundamental_cycles(E, n)
    truth = dict(sigma=sig, exL=energy(gL, ell), hkL=energy(cL, ell), exE=energy(gE, ell), hkE=energy(cE, ell),
                 aff=(cyc @ (eps*A)).tolist(), aff_check=(cyc @ f).tolist(), cL=cL.tolist(), cE=cE.tolist(),
                 nE=len(E), beta1=len(cyc), n=n, cv_ell=float(np.std(ell)/np.mean(ell)))
    res = {}
    for T in Ts:
        lst = []
        for r in range(reps):
            r_rng = np.random.default_rng(seed*1000+int(T)+r)
            x0 = int(r_rng.choice(n, p=pi/pi.sum()))
            N, tau = gillespie(Q, int(T), r_rng, x0)
            lst.append(estimators(N, tau, E, d0))
        res[str(int(T))] = lst
        print(name, T, flush=True)
    return dict(name=name, truth=truth, res=res)

if __name__ == '__main__':
    Ts = [1e3, 1e4, 1e5]
    out = [run_system('K6', 'complete', 6, 1.5, 11, Ts, 20),
           run_system('ER10', 'er', 10, 1.5, 12, Ts, 20, p=0.5)]
    json.dump(out, open('runs/e3.json', 'w'))
