"""Benchmark C3: Brusselator on an 8x8 periodic grid (driven limit cycle + Turing patterns).
Per site:  R1 A -> X (k1 a),  R2 2X + Y -> 3X (k2 x^2 y),  R3 B + X -> Y + D (k3 b x),  R4 X -> E (k4 x),
each with a tiny reverse rate eps (= 1e-3, times the chemostatted product concentration) so that the affinities
are finite.  k1..k4 = 1.  X, Y diffuse with D_X = 0.4, D_Y = 3.6 (ratio 9, Turing region).
Well-mixed fixed point (eps -> 0): x* = a, y* = b/a; Hopf at b = 1 + a^2; Turing threshold b_T = (1 + a sqrt(D_X/D_Y))^2.
HyperME form (mu = log c):  k_R1 = sqrt(k1 a eps), A_R1 = log(k1 a/eps);  k_R2 = sqrt(k2 eps), A_R2 = log(k2/eps);
k_R3 = sqrt(k3 b eps), A_R3 = log(k3 b/eps);  k_R4 = sqrt(k4 eps), A_R4 = log(k4/eps);  transport k = D_s, A = 0.
Local cycles: R1+R4 (A -> E) affinity log(k1 k4 a/eps^2), R2+R3 (B -> D) affinity log(k2 k3 b/eps^2).
--eq (control C3eq): every reverse rate constant equals the forward one (A_h = 0 for all h, detailed balance,
equilibrium x = y = 1): no oscillation is physically possible; the linearisation has real eigenvalues.
Sets: train 48 (a in [0.9,1.1], b in [2.3,3.0]; even index: homogeneous state taken from a random point of a
well-mixed burn-in + 0.5% site noise; odd: random smooth spatial field -> Turing), test_id 16 and test_ood 16
(b in [3.2,3.7]); test index < 8: exactly homogeneous off-cycle start (truth = well-mixed ODE broadcast, the exact
solution in the invariant homogeneous subspace), index >= 8: random spatial start. DT = 0.2, train 20 steps, test 200.
python gen_brusselator.py [--eq]  -> runs/data/brus.pt | brus_eq.pt"""
import os, sys, time
import numpy as np
from multiprocessing import Pool
from scipy.integrate import solve_ivp
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from crn2 import build_graph_np, solve_true, smooth_field, true_ab_np, RUNS

N = 8
K = (1.0, 1.0, 1.0, 1.0)
EPS = 1e-3
DX, DY = 0.4, 3.6
DT, T_TRAIN, T_TEST = 0.2, 20, 200
EQ = '--eq' in sys.argv
BENCH = 'brus_eq' if EQ else 'brus'


def true_params(g, a, b, eq=EQ):
    V, H = g['V'], g['H']
    k = np.zeros(H); A = np.zeros(H)
    kf = [K[0] * a, K[1], K[2] * b, K[3]]
    for r in range(4):
        if eq:
            k[r * V:(r + 1) * V] = kf[r]
        else:
            k[r * V:(r + 1) * V] = np.sqrt(kf[r] * EPS); A[r * V:(r + 1) * V] = np.log(kf[r] / EPS)
    k[4 * V:6 * V] = DX; k[6 * V:] = DY
    return k, A


def true_affinity(a, b, eq=EQ):
    if eq:
        return np.zeros(2)
    return np.array([np.log(K[0] * K[3] * a / EPS ** 2), np.log(K[1] * K[2] * b / EPS ** 2)])


def wellmixed_rhs(a, b, eq=EQ):
    k1, k2, k3, k4 = K
    if eq:
        def f(t, z):
            x, y = z
            r1 = k1 * a * (1 - x); r2 = k2 * x * x * (y - x); r3 = k3 * b * (x - y); r4 = k4 * (x - 1)
            return [r1 + r2 - r3 - r4, -r2 + r3]
    else:
        def f(t, z):
            x, y = z
            r1 = k1 * a - EPS * x; r2 = k2 * x * x * y - EPS * x ** 3; r3 = k3 * b * x - EPS * y; r4 = k4 * x - EPS
            return [r1 + r2 - r3 - r4, -r2 + r3]
    return f


def sample(rng, kind, i, split):
    a = rng.uniform(0.9, 1.1)
    b = rng.uniform(3.2, 3.7) if kind == 'ood' else rng.uniform(2.3, 3.0)
    xc, yc = (1.0, 1.0) if EQ else (a, b / a)
    sd = 0.6 if EQ else 0.4
    homog = (i % 2 == 0) if split == 'train' else (i < 8)
    if homog:
        z0 = np.array([xc * np.exp(sd * rng.standard_normal()), yc * np.exp(sd * rng.standard_normal())])
        if split == 'train':
            tb = rng.uniform(0, 8)
            if tb > 0:
                z0 = solve_ivp(wellmixed_rhs(a, b), (0, tb), z0, method='DOP853', rtol=1e-10, atol=1e-12).y[:, -1]
            noise = np.exp(0.005 * rng.standard_normal((2, N * N)))
        else:
            noise = np.ones((2, N * N))
        c0 = np.concatenate([z0[0] * noise[0], z0[1] * noise[1]])
    else:
        c0 = np.concatenate([xc * np.exp(smooth_field(rng, N, 3, sd)), yc * np.exp(smooth_field(rng, N, 3, sd))])
    return a, b, c0, homog


def make_set(seed, nb, kind, nsteps, split):
    g = build_graph_np(BENCH, N)
    rng = np.random.default_rng(seed + (100 if EQ else 0))
    out = dict(c=[], drv=[], a=[], b=[], sigma=[], J=[], k=[], A=[], homog=[], aff=[])
    for i in range(nb):
        a, b, c0, homog = sample(rng, kind, i, split)
        k, A = true_params(g, a, b)
        if homog and split != 'train':           # exact homogeneous solution (invariant subspace)
            ts = np.arange(nsteps + 1) * DT
            s = solve_ivp(wellmixed_rhs(a, b), (0, ts[-1]), c0[[0, N * N]], method='DOP853', t_eval=ts, rtol=1e-10, atol=1e-12)
            assert s.success
            C = np.repeat(s.y.T, N * N, axis=1)
            aa, bb = true_ab_np(C, g, k, A)
            sig = ((aa - bb) * (np.log(aa) - np.log(bb))).sum(-1); J = aa - bb
        else:
            C, sig, J = solve_true(g, k, A, c0, nsteps, DT)
        out['c'].append(C.reshape(-1, 2, g['V'])); out['drv'].append([np.log(a), np.log(b)]); out['a'].append(a); out['b'].append(b)
        out['sigma'].append(sig); out['J'].append(J[[0, min(20, nsteps), nsteps]]); out['k'].append(k); out['A'].append(A)
        out['homog'].append(homog); out['aff'].append(true_affinity(a, b))
    return {kk: np.array(v) for kk, v in out.items()}


if __name__ == '__main__':
    t0 = time.time()
    odir = os.path.join(RUNS, 'data'); os.makedirs(odir, exist_ok=True)
    specs = [('train', 1, 48, 'id', T_TRAIN), ('test_id', 2, 16, 'id', T_TEST), ('test_ood', 3, 16, 'ood', T_TEST)]
    with Pool(3) as pool:
        sets = pool.starmap(make_set, [(s, nb, kd, ns, name) for name, s, nb, kd, ns in specs])
    data = {name: d for (name, *_), d in zip(specs, sets)}
    dm, ds = data['train']['drv'].mean(0), data['train']['drv'].std(0)
    import torch
    out = {k: {kk: torch.tensor(vv) for kk, vv in v.items()} for k, v in data.items()}
    out['meta'] = dict(DT=DT, T_TRAIN=T_TRAIN, T_TEST=T_TEST, N=N, bench=BENCH, drv_mean=torch.tensor(dm), drv_std=torch.tensor(ds),
                       params=dict(k=K, eps=EPS, DX=DX, DY=DY, eq=EQ))
    torch.save(out, os.path.join(odir, f'{BENCH}.pt'))
    for k, d in data.items():
        C = d['c']; xm = C[:, :, 0].mean(-1)
        h = d['homog'].astype(bool)
        half = xm[:, xm.shape[1] // 2:]
        amp = half.max(1) - half.min(1)
        sp = C[:, -1, 0].std(-1)
        print(f'{k}: c {C.shape} min {C.min():.3e} max {C.max():.2f} | sigma mean {d["sigma"].mean():.1f} | homog {h.sum()} '
              f'| amp(Xbar, 2nd half) homog {amp[h].mean():.3f} spatial {amp[~h].mean():.3f} | spatial std X end homog {sp[h].mean():.2e} spatial {sp[~h].mean():.3f} '
              f'| b [{d["b"].min():.2f},{d["b"].max():.2f}]', flush=True)
    print(f'done {time.time() - t0:.0f}s')
