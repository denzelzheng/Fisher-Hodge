"""Benchmark C1: chemostat-driven spatial phosphorylation cycle on an 8x8 periodic grid.
Species X, Xp, E (E immobile, catalytic).  Per site: R1 X+E <-> Xp+E (rates k1 X E e^{+dG/2}, k1 Xp E e^{-dG/2}),
R2 Xp <-> X (k2 Xp, k2' X).  Transport of X, Xp on grid edges, both feel the smooth potential U(v):
J_ij = D (c_i e^{(U_i-U_j)/2} - c_j e^{(U_j-U_i)/2}).
In HyperME form: mu = log c + U_s (U_X = U_Xp = U, U_E = 0),
  k_R1 = k1 e^{-U},  A_R1 = dG;  k_R2 = sqrt(k2 k2') e^{-U},  A_R2 = log(k2/k2');  k_T = D e^{-(U_i+U_j)/2}, A_T = 0.
Cycle affinity of the local R1+R2 loop = dG + log(k2/k2').
python gen_crn.py  -> hyper/runs/data/crn.pt, crn_gillespie.pt"""
import os, sys, time
import numpy as np
from scipy.integrate import solve_ivp
from multiprocessing import Pool
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from hyperme import build_graph_np, NS, SP_X, SP_XP, SP_E, DT, T_TRAIN, T_TEST, RUNS

N = 8
K1, K2, DIFF = 2.0, 1.0, 1.0
K2R = float(os.environ.get('CRN_K2R', '0.02'))      # control experiment C1b uses 0.2
TAG = os.environ.get('CRN_TAG', '')
LOGK2 = np.log(K2 / K2R)
OMEGA = 200


def smooth_field(rng, n, K, std):
    a, b = np.meshgrid(np.arange(n), np.arange(n), indexing='ij')
    f = np.zeros((n, n))
    for kx in range(-K, K + 1):
        for ky in range(0, K + 1):
            if (kx, ky) == (0, 0) or kx * kx + ky * ky > K * K or (ky == 0 and kx < 0):
                continue
            ph = 2 * np.pi * (kx * b + ky * a) / n
            f += rng.normal() * np.cos(ph) + rng.normal() * np.sin(ph)
    f -= f.mean()
    return (std * f / (f.std() + 1e-12)).ravel()


def true_params(g, U, dG):
    V, H = g['V'], g['H']
    k = np.zeros(H); A = np.zeros(H)
    k[g['r1']] = K1 * np.exp(-U); A[g['r1']] = dG
    k[g['r2']] = np.sqrt(K2 * K2R) * np.exp(-U); A[g['r2']] = LOGK2
    for sl in (g['tx'], g['txp']):
        k[sl] = DIFF * np.exp(-0.5 * (U[g['et'][sl]] + U[g['eh'][sl]]))
    Us = np.concatenate([U, U, np.zeros(V)])
    return k, A, Us


def true_ab(c, g, k, A, Us):
    """c (..., NS*V) -> a, b (..., H)"""
    m = np.log(c) + Us
    return k * np.exp(m @ g['Np'].T + 0.5 * A), k * np.exp(m @ g['Nm'].T - 0.5 * A)


def solve_traj(g, U, dG, c0, nsteps):
    k, A, Us = true_params(g, U, dG)
    Sm = g['Nm'] - g['Np']

    def f(t, x):
        a, b = true_ab(x, g, k, A, Us)
        return (a - b) @ Sm
    ts = np.arange(nsteps + 1) * DT
    sol = solve_ivp(f, (0, ts[-1]), c0, method='DOP853', t_eval=ts, rtol=1e-10, atol=1e-12)
    assert sol.success
    C = sol.y.T                                          # (T+1, NS*V)
    assert C.min() > 0
    a, b = true_ab(C, g, k, A, Us)
    sig = ((a - b) * (np.log(a) - np.log(b))).sum(-1)
    J = a - b
    return C, sig, J, k, A


def sample_case(rng, g, kind):
    V = g['V']
    if kind == 'oodU':
        U = smooth_field(rng, N, 3, 1.2)
    else:
        U = smooth_field(rng, N, 2, 0.7)
    dG = rng.uniform(3, 4) if kind == 'oodG' else rng.uniform(1, 3)
    X0 = 0.6 * np.exp(smooth_field(rng, N, 2, 0.4)); Xp0 = 0.4 * np.exp(smooth_field(rng, N, 2, 0.4))
    E0 = np.exp(smooth_field(rng, N, 2, 0.4))
    return U, dG, np.concatenate([X0, Xp0, E0])


def make_set(seed, nb, kind, nsteps):
    g = build_graph_np(N)
    rng = np.random.default_rng(seed)
    out = dict(c=[], U=[], dG=[], sigma=[], J=[], k=[], A=[])
    for _ in range(nb):
        U, dG, c0 = sample_case(rng, g, kind)
        C, sig, J, k, A = solve_traj(g, U, dG, c0, nsteps)
        out['c'].append(C.reshape(-1, NS, g['V'])); out['U'].append(U); out['dG'].append(dG); out['sigma'].append(sig)
        out['J'].append(J[[0, min(20, nsteps), nsteps]]); out['k'].append(k); out['A'].append(A)
    return {kk: np.array(v) for kk, v in out.items()}


# ------------------------------------------------------------------ Gillespie (direct method)
def gillespie(args):
    seed, U, dG, c0 = args
    g = build_graph_np(N)
    rng = np.random.default_rng(seed)
    k, A, Us = true_params(g, U, dG)
    # mass-action coefficients: a_h = ka_h prod_{reactants} c, b_h = kb_h prod_{products} c   (all nu entries are 0/1)
    ka = k * np.exp(0.5 * A + g['Np'] @ Us); kb = k * np.exp(-0.5 * A + g['Nm'] @ Us)
    Sm = (g['Nm'] - g['Np']).astype(np.int64)
    nvec = np.append(np.round(OMEGA * c0).astype(np.int64), OMEGA)          # last slot = Omega (constant 1)
    rp, rm = g['rp'], g['rm']
    H = g['H']
    # propensities in molecule numbers: Omega * a(n/Omega)
    cp, cm = ka, kb   # pad slot holds Omega => Omega*a(n/Omega) = ka * n_i * n_j / Omega
    stoich = np.concatenate([Sm, -Sm], 0)
    sp = [np.nonzero(stoich[r])[0] for r in range(2 * H)]
    sv = [stoich[r][sp[r]] for r in range(2 * H)]
    ts = np.arange(T_TEST + 1) * DT
    out = np.zeros((T_TEST + 1, NS * g['V'])); out[0] = nvec[:-1] / OMEGA
    t, it, nev = 0.0, 1, 0
    while it <= T_TEST:
        pr = np.concatenate([cp * nvec[rp[:, 0]] * nvec[rp[:, 1]] / OMEGA, cm * nvec[rm[:, 0]] * nvec[rm[:, 1]] / OMEGA])
        a0 = pr.sum()
        t += rng.exponential(1 / a0)
        while it <= T_TEST and ts[it] < t:
            out[it] = nvec[:-1] / OMEGA; it += 1
        r = np.searchsorted(np.cumsum(pr), rng.random() * a0)
        r = min(r, 2 * H - 1)
        nvec[sp[r]] += sv[r]; nev += 1
    return out.reshape(-1, NS, g['V']), nev


if __name__ == '__main__':
    t0 = time.time()
    odir = os.path.join(RUNS, 'data'); os.makedirs(odir, exist_ok=True)
    specs = [('train', 1, 48, 'id', T_TRAIN), ('test_id', 2, 16, 'id', T_TEST),
             ('test_oodG', 3, 16, 'oodG', T_TEST), ('test_oodU', 4, 16, 'oodU', T_TEST)]
    with Pool(4) as pool:
        sets = pool.starmap(make_set, [(s, nb, kd, ns) for _, s, nb, kd, ns in specs])
    data = {name: d for (name, *_), d in zip(specs, sets)}
    import torch
    torch.save({k: {kk: torch.tensor(vv) for kk, vv in v.items()} for k, v in data.items()}, os.path.join(odir, f'crn{TAG}.pt'))
    for k, d in data.items():
        C = d['c']; ph = C[:, :, 1].sum(-1) / C[:, :, :2].sum((-1, -2))
        chg = np.linalg.norm((C[:, -1] - C[:, min(20, C.shape[1] - 1)]).reshape(len(C), -1), axis=1) / np.linalg.norm(C[:, -1].reshape(len(C), -1), axis=1)
        print(f'{k}: c {C.shape} min {C.min():.3e} max {C.max():.2f} | Xp frac t0 {ph[:, 0].mean():.2f} end {ph[:, -1].mean():.2f} '
              f'| sigma end {d["sigma"][:, -1].mean():.2f} | rel change T->end {chg.mean():.3f} | dG [{d["dG"].min():.2f},{d["dG"].max():.2f}]', flush=True)
    print(f'ODE data done {time.time() - t0:.0f}s', flush=True)
    # Gillespie: 4 ID trajectories (Omega = 200)
    g = build_graph_np(N)
    rng = np.random.default_rng(5)
    cases = [sample_case(rng, g, 'id') for _ in range(4)]
    with Pool(4) as pool:
        res = pool.map(gillespie, [(100 + i, U, dG, c0) for i, (U, dG, c0) in enumerate(cases)])
    ode = [solve_traj(g, U, dG, r[0][0].ravel(), T_TEST)[0].reshape(-1, NS, g['V']) for (U, dG, c0), r in zip(cases, res)]
    gd = dict(c=np.stack([r[0] for r in res]), c_ode=np.stack(ode), U=np.stack([c[0] for c in cases]),
              dG=np.array([c[1] for c in cases]), nevents=np.array([r[1] for r in res]))
    torch.save({k: torch.tensor(v) for k, v in gd.items()}, os.path.join(odir, f'crn{TAG}_gillespie.pt'))
    e = np.linalg.norm((gd['c'] - gd['c_ode'])[:, :, :2].reshape(4, T_TEST + 1, -1), axis=-1) / np.linalg.norm(gd['c_ode'][:, :, :2].reshape(4, T_TEST + 1, -1), axis=-1)
    print(f'Gillespie done: events {gd["nevents"]}, rel L2 (X,Xp) Gillespie vs ODE @T {e[:, 20].mean():.3f} @10T {e[:, -1].mean():.3f}; total {time.time() - t0:.0f}s')
