"""Benchmark C5: non-ideal (crowding) Schlogl model on the 8x8 periodic grid of C2.
Same hypergraph, rate constants, chemostats and sampling as gen_schlogl.py, but every hyperedge uses
activity-corrected mass action:  a_h = k_h exp(<nu+_h, mu> + A_h/2), b_h = k_h exp(<nu-_h, mu> - A_h/2) with
    mu = log x + ln gamma(x),   ln gamma(x) = PHI x / (1 - PHI x)   (crowding, PHI = 0.25, x < 1/PHI = 4).
Transport too: J_ij = D (x_i gamma_i - x_j gamma_j)  (diffusion of activity).
Fixed points: the activity y = x gamma(x) satisfies the ideal Schlogl cubic, so x = y^{-1}(roots).
Initial states are the C2 initial states read as activities (x0 = y^{-1}(y0)): low ~0.53, high ~1.66 at y = 0.6 / 3.4.
ln gamma ranges ~0.07 .. ~0.92 in ID; OOD has a in [3.5, 4.0], b in [2.0, 3.5] (both above the training box) -> higher
high-state concentration (x up to ~2.1), stronger non-ideality (ln gamma up to ~1.1).  Train 48 x 20 steps, test_id / test_ood 16 x 200 steps (DT = 0.05).
python gen_nonideal.py -> runs/data/schlogl_ni.pt"""
import os, sys, time
import numpy as np
from multiprocessing import Pool
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from crn2 import build_graph_np, solve_true, RUNS
import gen_schlogl as GS

PHI = 0.25
N, DT, T_TRAIN, T_TEST = GS.N, GS.DT, GS.T_TRAIN, GS.T_TEST
BENCH = 'schlogl_ni'


def lng(x):
    return PHI * x / (1 - PHI * x)


def y_of_x(x):
    return x * np.exp(lng(x))


def x_of_y(y):
    y = np.asarray(y, float)
    lo, hi = np.zeros_like(y), np.full_like(y, 1 / PHI - 1e-12)
    for _ in range(200):
        mid = 0.5 * (lo + hi)
        big = y_of_x(mid) > y
        hi = np.where(big, mid, hi); lo = np.where(big, lo, mid)
    return 0.5 * (lo + hi)


def make_set(seed, nb, kind, nsteps):
    g = build_graph_np(BENCH, N)
    rng = np.random.default_rng(seed + 500)
    out = dict(c=[], drv=[], a=[], b=[], sigma=[], J=[], k=[], A=[], bistable=[], aff=[], xlo=[], xhi=[], thr=[])
    for i in range(nb):
        a, b, y0 = GS.sample(rng, 'id', i)
        if kind == 'ood':                       # higher total concentration: larger a and b than in training
            a = rng.uniform(3.5, 4.0); b = rng.uniform(2.0, 3.5)
        x0 = x_of_y(y0)
        k, A = GS.true_params(g, a, b)
        C, sig, J = solve_true(g, k, A, x0, nsteps, DT, lng=lng)
        r = GS.roots(a, b)
        out['c'].append(C.reshape(-1, 1, g['V'])); out['drv'].append([np.log(a), np.log(b)]); out['a'].append(a); out['b'].append(b)
        out['sigma'].append(sig); out['J'].append(J[[0, min(20, nsteps), nsteps]]); out['k'].append(k); out['A'].append(A)
        out['bistable'].append(len(r) == 3); out['aff'].append(GS.true_affinity(a, b))
        out['xlo'].append(x_of_y(r[0])); out['xhi'].append(x_of_y(r[-1])); out['thr'].append(x_of_y(GS.threshold(a, b)))
    return {kk: np.array(v) for kk, v in out.items()}


if __name__ == '__main__':
    t0 = time.time()
    odir = os.path.join(RUNS, 'data'); os.makedirs(odir, exist_ok=True)
    specs = [('train', 1, 48, 'id', T_TRAIN), ('test_id', 2, 16, 'id', T_TEST), ('test_ood', 3, 16, 'ood', T_TEST)]
    with Pool(3) as pool:
        sets = pool.starmap(make_set, [(s, nb, kd, ns) for _, s, nb, kd, ns in specs])
    data = {name: d for (name, *_), d in zip(specs, sets)}
    dm, ds = data['train']['drv'].mean(0), data['train']['drv'].std(0)
    import torch
    out = {k: {kk: torch.tensor(vv) for kk, vv in v.items()} for k, v in data.items()}
    out['meta'] = dict(DT=DT, T_TRAIN=T_TRAIN, T_TEST=T_TEST, N=N, bench=BENCH, drv_mean=torch.tensor(dm), drv_std=torch.tensor(ds),
                       params=dict(k1=GS.K1, k1r=GS.K1R, k2=GS.K2, k2r=GS.K2R, D=GS.DIFF, phi=PHI))
    torch.save(out, os.path.join(odir, f'{BENCH}.pt'))
    for k, d in data.items():
        C = d['c']
        print(f'{k}: c {C.shape} min {C.min():.3f} max {C.max():.3f} | ln gamma range [{lng(C.min()):.3f}, {lng(C.max()):.3f}] '
              f'| mean X end {C[:, -1].mean():.3f} | sigma end {d["sigma"][:, -1].mean():.1f} | bistable {d["bistable"].mean():.2f}', flush=True)
    # how wrong is ideal mass action with the same constants? (ID test, 10T)
    g = build_graph_np(BENCH, N)
    d = data['test_id']; e = []
    for i in range(len(d['a'])):
        Ci, _, _ = solve_true(g, d['k'][i], d['A'][i], d['c'][i, 0, 0], T_TEST, DT)
        e.append(np.linalg.norm(Ci[-1] - d['c'][i, -1, 0]) / np.linalg.norm(d['c'][i, -1, 0]))
    print(f'ideal law with true constants vs non-ideal truth, ID 10T relL2 median {np.median(e):.3f}')
    print(f'done {time.time() - t0:.0f}s')
