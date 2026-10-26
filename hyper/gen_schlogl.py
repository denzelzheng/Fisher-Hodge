"""Benchmark C2: Schlogl autocatalytic model on an 8x8 periodic grid (bistable, many-to-many hyperedges).
Per site v:  R1  A + 2X <-> 3X  (k1 a x^2, k1' x^3),   R2  X <-> B  (k2 x, k2' b);   X diffuses on grid edges (D).
A, B chemostatted (concentrations a, b = drive).  Rate law  dx/dt = k1 a x^2 - k1' x^3 - k2 x + k2' b + D lap x.
Parameters k1 = 1, k1' = 0.5, k2 = 5, k2' = 1, D = 1:  f(x) = -0.5 (x^3 - 2a x^2 + 10 x - 2b).
At a = 3: bistable for b in (1.46, 2.55), Maxwell point b_M = 2 (roots 2-sqrt2, 2, 2+sqrt2; planar front pinned).
Maxwell point for general a: middle root at the inflection x_i = 2a/3  =>  b_M(a) = (x_i^3 - 2a x_i^2 + 10 x_i)/2.
HyperME form (mu = log x): k_R1 = sqrt(k1 a k1'), A_R1 = log(k1 a/k1');  k_R2 = sqrt(k2 k2' b), A_R2 = log(k2/(k2' b));
k_T = D, A_T = 0.   Local cycle affinity R1+R2 (A -> B) = log(k1 k2 a / (k1' k2' b)).
Sets: train 48 (a in [2.85,3.10], b in [1.0,3.2], half bistable / half monostable, random high/low domains, 20 steps),
test_id 16 (same law, 200 steps), test_ood 16 (a in [3.15,3.35], b in [0.3,1.0]: both outside the training box),
test_front 13 (a = 3, b = 1.7..2.3, straight stripe of high state: planar front speed vs b across the Maxwell point).
python gen_schlogl.py -> runs/data/schlogl.pt"""
import os, sys, time
import numpy as np
from multiprocessing import Pool
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from crn2 import build_graph_np, solve_true, smooth_field, RUNS

N = 8
K1, K1R, K2, K2R, DIFF = 1.0, 0.5, 5.0, 1.0, 1.0
DT, T_TRAIN, T_TEST = 0.05, 20, 200
BENCH = 'schlogl'


def f_rate(x, a, b):
    return K1 * a * x ** 2 - K1R * x ** 3 - K2 * x + K2R * b


def roots(a, b):
    r = np.roots([-K1R, K1 * a, -K2, K2R * b])
    r = np.sort(r[np.abs(r.imag) < 1e-9].real)
    return r[r > 0]


def maxwell_b(a):
    xi = K1 * a / (3 * K1R)                       # inflection point of f
    return -(K1 * a * xi ** 2 - K1R * xi ** 3 - K2 * xi) / K2R


def threshold(a, b):
    r = roots(a, b)
    return r[1] if len(r) == 3 else K1 * a / (3 * K1R)


def true_params(g, a, b):
    V, H = g['V'], g['H']
    k = np.zeros(H); A = np.zeros(H)
    k[:V] = np.sqrt(K1 * a * K1R); A[:V] = np.log(K1 * a / K1R)
    k[V:2 * V] = np.sqrt(K2 * K2R * b); A[V:2 * V] = np.log(K2 / (K2R * b))
    k[2 * V:] = DIFF
    return k, A


def true_affinity(a, b):
    return np.log(K1 * K2 * a / (K1R * K2R * b))


def sample(rng, kind, i):
    if kind == 'ood':
        a = rng.uniform(3.15, 3.35); b = rng.uniform(0.3, 1.0)
    else:
        a = rng.uniform(2.85, 3.10)
        want = (i % 2 == 0)
        while True:
            b = rng.uniform(1.0, 3.2)
            if (len(roots(a, b)) == 3) == want:
                break
    s = smooth_field(rng, N, 2, 1.0)
    q = rng.uniform(0.3, 0.7)
    hi = s > np.quantile(s, q)
    x0 = np.where(hi, 3.4, 0.6) * np.exp(0.1 * rng.standard_normal(N * N))
    return a, b, x0


def stripe(a, b):
    r = roots(a, b)
    col = np.arange(N * N) % N
    return np.where(col < N // 2, r[-1], r[0]).astype(float)


def make_set(seed, nb, kind, nsteps):
    g = build_graph_np(BENCH, N)
    rng = np.random.default_rng(seed)
    out = dict(c=[], drv=[], a=[], b=[], sigma=[], J=[], k=[], A=[], bistable=[], bM=[], thr=[], xlo=[], xhi=[], aff=[])
    for i in range(nb):
        if kind == 'front':
            a = 3.0; b = np.linspace(1.7, 2.3, nb)[i]; x0 = stripe(a, b)
        else:
            a, b, x0 = sample(rng, kind, i)
        k, A = true_params(g, a, b)
        C, sig, J = solve_true(g, k, A, x0, nsteps, DT)
        r = roots(a, b)
        out['c'].append(C.reshape(-1, 1, g['V'])); out['drv'].append([np.log(a), np.log(b)])
        out['a'].append(a); out['b'].append(b); out['sigma'].append(sig); out['J'].append(J[[0, min(20, nsteps), nsteps]])
        out['k'].append(k); out['A'].append(A); out['bistable'].append(len(r) == 3); out['bM'].append(maxwell_b(a))
        out['thr'].append(threshold(a, b)); out['xlo'].append(r[0]); out['xhi'].append(r[-1]); out['aff'].append(true_affinity(a, b))
    return {kk: np.array(v) for kk, v in out.items()}


if __name__ == '__main__':
    t0 = time.time()
    odir = os.path.join(RUNS, 'data'); os.makedirs(odir, exist_ok=True)
    print('a=3, b=2 roots', roots(3, 2), 'f\'', [2 * K1 * 3 * x - 3 * K1R * x ** 2 - K2 for x in roots(3, 2)], 'b_M(3)', maxwell_b(3.0))
    bb = np.linspace(1.0, 3.2, 2201); bi = bb[[len(roots(3.0, b)) == 3 for b in bb]]
    print(f'a=3 bistable b-interval [{bi.min():.3f}, {bi.max():.3f}]')
    specs = [('train', 1, 48, 'id', T_TRAIN), ('test_id', 2, 16, 'id', T_TEST), ('test_ood', 3, 16, 'ood', T_TEST),
             ('test_front', 4, 13, 'front', T_TEST)]
    with Pool(4) as pool:
        sets = pool.starmap(make_set, [(s, nb, kd, ns) for _, s, nb, kd, ns in specs])
    data = {name: d for (name, *_), d in zip(specs, sets)}
    dm, ds = data['train']['drv'].mean(0), data['train']['drv'].std(0)
    import torch
    out = {k: {kk: torch.tensor(vv) for kk, vv in v.items()} for k, v in data.items()}
    out['meta'] = dict(DT=DT, T_TRAIN=T_TRAIN, T_TEST=T_TEST, N=N, bench=BENCH, drv_mean=torch.tensor(dm), drv_std=torch.tensor(ds),
                       params=dict(k1=K1, k1r=K1R, k2=K2, k2r=K2R, D=DIFF))
    torch.save(out, os.path.join(odir, 'schlogl.pt'))
    for k, d in data.items():
        C = d['c']; m = C.mean((-1, -2))
        side = np.mean(d['b'][d['bistable']] > d['bM'][d['bistable']]) if d['bistable'].any() else float('nan')
        print(f'{k}: c {C.shape} min {C.min():.3e} max {C.max():.2f} | mean X t0 {m[:, 0].mean():.2f} end {m[:, -1].mean():.2f} '
              f'| sigma end {d["sigma"][:, -1].mean():.1f} | bistable {d["bistable"].mean():.2f} (frac b>b_M among them {side:.2f}) '
              f'| a [{d["a"].min():.2f},{d["a"].max():.2f}] b [{d["b"].min():.2f},{d["b"].max():.2f}] | affinity [{d["aff"].min():.2f},{d["aff"].max():.2f}]', flush=True)
    print(f'done {time.time() - t0:.0f}s')
