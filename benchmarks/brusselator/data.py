"""Reference data of the Brusselator: hypergraph, true rates, generation and loading.

    python -m benchmarks.brusselator.run generate            # every set, written to $NEURALCHART_DATA/brusselator
"""
import math
import os
import time

import numpy as np
import torch

from neuralchart.graphs import Hypergraph
from .config import DATA, DT, KF, EPS, DIFF, H_GRID, SPECIES, CHEM, REACTIONS
from .test_sets import GEN, sample_ab

_HG = {}


def hg_of(n):
    """the reaction-transport hypergraph on the n x n periodic grid (n = 1: one well-mixed site).  Hyperedge types
    0-3 are the reactions R1-R4 at every site, 4 and 5 the transport of X and Y."""
    if n not in _HG:
        kw = dict(n_sites=1) if n == 1 else dict(grid_n=n, diffusing=['X', 'Y'])
        _HG[n] = Hypergraph.reaction_transport(SPECIES, REACTIONS, chemostats=CHEM, **kw)
        if n > 1:       # the cycle and conservation bases (an SVD of S) are not needed here and do not scale
            _HG[n]._Z = np.zeros((_HG[n].H, 0)); _HG[n]._L = np.zeros((_HG[n].M, 0))
    return _HG[n]


def true_constants(hg, logab):
    """one-way rate constants (B, H) of the reference at the chemostats logab (B, 2) = (log a, log b), d = e = 1"""
    typ = torch.as_tensor(hg.meta['typ'])
    D = [d / H_GRID ** 2 for d in DIFF]
    kF = torch.tensor(list(KF) + D, dtype=torch.float64)[typ]
    kB = torch.tensor([EPS] * 4 + D, dtype=torch.float64)[typ]
    lc = torch.cat([logab.double(), torch.zeros(logab.shape[0], 2, dtype=torch.float64)], -1).repeat_interleave(hg.V, 1)
    Yp = torch.as_tensor(hg.meta['chem_Np'].toarray()); Ym = torch.as_tensor(hg.meta['chem_Nm'].toarray())
    return kF * torch.exp(lc @ Yp.T), kB * torch.exp(lc @ Ym.T)


def true_rates(hg, logab, c):
    """one-way rates (a, b) of the reference at the states c (B, M), float64"""
    kf, kb = true_constants(hg, logab)
    c = c.double()
    return kf.to(c.device) * hg.mass_action(c, '+'), kb.to(c.device) * hg.mass_action(c, '-')


class TruthNP:
    """numpy right-hand side of the reference at one condition"""
    def __init__(s, n, a, b):
        hg = hg_of(n)
        kf, kb = true_constants(hg, torch.tensor([[math.log(a), math.log(b)]]))
        s.kf, s.kb = kf[0].numpy(), kb[0].numpy()
        s.Np, s.Nm = hg.Np.tocsr(), hg.Nm.tocsr()
        s.S = (hg.Nm - hg.Np).T.tocsr()

    def rhs(s, t, x):
        L = np.log(np.maximum(x, 1e-300))
        return s.S @ (s.kf * np.exp(s.Np @ L) - s.kb * np.exp(s.Nm @ L))


def solve(n, a, b, c0, nsteps):
    from scipy.integrate import solve_ivp
    tr = TruthNP(n, a, b)
    ts = np.arange(nsteps + 1) * DT
    sol = solve_ivp(tr.rhs, (0, ts[-1]), c0, method='DOP853', t_eval=ts, rtol=1e-10, atol=1e-12)
    assert sol.success and sol.y.min() > 0, sol.message
    return sol.y.T


def smooth_field(rng, n, K, std):
    """a random periodic field of the Fourier modes |k| <= K, zero mean, standard deviation std, flattened"""
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


def make_traj(args):
    """trajectory i of set `kind`: (states (steps + 1, 2 V), (log a, log b), homogeneous start)"""
    kind, i, nsteps, n, seed, count = args
    rng = np.random.default_rng([seed, i])
    a, b = sample_ab(rng, kind)
    V = n * n
    xc, yc = a, b / a
    homog = (i % 2 == 0) if kind == 'train' else (i < count // 2 and kind != 'grid64')
    if homog:
        z0 = np.array([xc * np.exp(0.4 * rng.standard_normal()), yc * np.exp(0.4 * rng.standard_normal())])
        if kind == 'train':
            tb = rng.uniform(0, 8)
            z0 = solve(1, a, b, z0, int(round(tb / DT)))[-1] if tb > DT else z0
            noise = np.exp(0.005 * rng.standard_normal((2, V)))
            C = solve(n, a, b, np.concatenate([z0[0] * noise[0], z0[1] * noise[1]]), nsteps)
        else:
            C = np.repeat(solve(1, a, b, z0, nsteps), V, axis=1)      # exact: (x, y) copied to every site
    else:
        K = max(3, int(round(6 * n / 32)))
        c0 = np.concatenate([xc * np.exp(smooth_field(rng, n, K, 0.4)), yc * np.exp(smooth_field(rng, n, K, 0.4))])
        C = solve(n, a, b, c0, nsteps)
    return C.astype(np.float64), (math.log(a), math.log(b)), homog


def make_set(name, procs=12, idx=None):
    """the dict of set `name` (trajectories idx, default all), computed with `procs` processes"""
    count, nsteps, n, seed = GEN[name]
    idx = range(count) if idx is None else idx
    jobs = [(name, i, nsteps, n, seed, count) for i in idx]
    if procs > 1:
        # fresh worker processes, one thread each (a forked child can deadlock in a thread pool of the parent)
        from multiprocessing import get_context
        saved = {k: os.environ.get(k) for k in ('OMP_NUM_THREADS', 'MKL_NUM_THREADS')}
        os.environ.update(OMP_NUM_THREADS='1', MKL_NUM_THREADS='1')
        try:
            with get_context('spawn').Pool(procs) as pool:
                res = pool.map(make_traj, jobs)
        finally:
            for k, v in saved.items():
                if v is None:
                    os.environ.pop(k, None)
                else:
                    os.environ[k] = v
    else:
        res = [make_traj(j) for j in jobs]
    return dict(c=torch.as_tensor(np.stack([r[0] for r in res])), logab=torch.tensor([r[1] for r in res], dtype=torch.float64),
                homog=torch.tensor([r[2] for r in res]), n=n)


def generate(sets, procs=12, data=DATA):
    os.makedirs(data, exist_ok=True)
    for name in sets:
        t0 = time.time()
        d = make_set(name, procs)
        torch.save(d, os.path.join(data, f'{name}.pt'))
        print(f'{name}: {tuple(d["c"].shape)} n {d["n"]} | {describe(d)} | {time.time() - t0:.0f}s', flush=True)


def describe(d):
    """amplitude of the site-mean X over the last quarter (homogeneous starts) and spatial std of X at the end"""
    C, n = d['c'].numpy(), d['n']
    V = n * n
    xm = C[:, :, :V].mean(-1)
    q = xm[:, -C.shape[1] // 4:]
    amp = q.max(1) - q.min(1)
    sstd = C[:, -1, :V].std(-1) / C[:, -1, :V].mean(-1)
    h = d['homog'].numpy().astype(bool)
    out = []
    if h.any():
        out.append(f'homogeneous: amplitude (last quarter) {np.median(amp[h]):.3f} [{amp[h].min():.3f}, {amp[h].max():.3f}]')
    if (~h).any():
        out.append(f'spatial: relative std of X at the end {np.median(sstd[~h]):.3f} [{sstd[~h].min():.3f}, {sstd[~h].max():.3f}]')
    return ' | '.join(out) + f' | max c {C.max():.2f} min c {C.min():.2e}'


def load_set(name, device='cpu', data=DATA):
    d = torch.load(os.path.join(data, f'{name}.pt'))
    d['c'] = d['c'].to(device)
    d['drive'] = dict(logab=d['logab'].to(device))
    return d


def train_norm(data=DATA):
    """z-scoring of the conditions and the concentration scale, from the training set (inputs of the black boxes)"""
    tr = torch.load(os.path.join(data, 'train.pt'))
    lab = tr['logab']
    return dict(lab_mean=lab.mean(0).tolist(), lab_std=lab.std(0).tolist(), c_scale=float(tr['c'].mean()))
