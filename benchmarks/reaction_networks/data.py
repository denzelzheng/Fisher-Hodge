"""Data sets of the reaction-network benchmarks: reference trajectories at random conditions.

Schloegl models (DT 0.05; ranges below):
  train               48 trajectories x 100 steps: a 2.85–3.10, b 1.0–3.2, half of them bistable, random high / low domains
  train_conditions    16 x 200 steps, the training ranges
  high_a_low_b        schlogl: a 3.15–3.35, b 0.3–1.0
  high_concentration  schlogl_nonideal: a 3.5–4.0, b 2.0–3.5 (higher concentration, stronger crowding)
  planar_fronts       schlogl: a = 3, b 1.7–2.3 (13 values), a straight stripe of the high state
  grid16              8 trajectories of the training conditions on the 16 x 16 grid (same spacing)
Phosphorylation cycle (DT 0.05): train (48 x 100 steps), train_conditions (dG 1–3, potential amplitude 0.7),
  strong_drive (dG 3–4), strong_potential (amplitude 1.2 and finer: wavenumbers up to 3 instead of 2), grid16.
MAPK cascade (DT 0.5): train (48 x 20 steps), train_conditions (ATP 0.5–2), high_atp (3–5), low_atp (0.15–0.3),
  phosphorylated_start (ATP 0.5–2, most of every tier phosphorylated at t = 0); ADP = Pi = 1; 200 test steps.

    python -m benchmarks.schlogl.run generate
"""
import json
import math
import os

import numpy as np
import torch

from . import config as C
from . import truth as TR

# (count, steps, grid n, random seed) per set; the non-ideal model adds 500 to the seed
GEN = dict(train=(48, 100, 8, 1), train_conditions=(16, 200, 8, 2), high_a_low_b=(16, 200, 8, 3),
           high_concentration=(16, 200, 8, 3), planar_fronts=(13, 200, 8, 4), grid16=(8, 200, 16, 5))
GEN_PC = dict(train=(48, 100, 8, 1, 'id'), train_conditions=(16, 200, 8, 2, 'id'), strong_drive=(16, 200, 8, 3, 'drive'),
              strong_potential=(16, 200, 8, 4, 'potential'), grid16=(8, 200, 16, 5, 'id'))
# MAPK: (name, count, ATP range, initial state, steps), drawn in this order from one random stream (seed 0)
GEN_MAPK = [('train', 48, (0.5, 2.0), 'resting', 20), ('train_conditions', 16, (0.5, 2.0), 'resting', 200),
            ('high_atp', 16, (3.0, 5.0), 'resting', 200), ('low_atp', 16, (0.15, 0.3), 'resting', 200),
            ('phosphorylated_start', 16, (0.5, 2.0), 'phosphorylated', 200)]


def smooth_field(rng, n, K, std):
    """a random periodic field with wavenumbers |k| <= K, zero mean, standard deviation std (flattened)"""
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


def sample_schlogl(rng, kind, i, n=8):
    """a, b and the initial activity field: high / low domains of a smooth random field"""
    if kind == 'high_a_low_b':
        a = rng.uniform(3.15, 3.35); b = rng.uniform(0.3, 1.0)
    else:
        a = rng.uniform(2.85, 3.10)
        want = (i % 2 == 0)                          # half of the trajectories in the bistable range
        while True:
            b = rng.uniform(1.0, 3.2)
            if (len(TR.roots(a, b)) == 3) == want:
                break
    s = smooth_field(rng, n, 2 * n // 8, 1.0)
    q = rng.uniform(0.3, 0.7)
    hi = s > np.quantile(s, q)
    y0 = np.where(hi, 3.4, 0.6) * np.exp(0.1 * rng.standard_normal(n * n))
    return a, b, y0


def make_set_schlogl(name):
    count, nsteps, n, seed = GEN[name]
    rng = np.random.default_rng(seed + (500 if C.SYSTEM == 'schlogl_nonideal' else 0))
    out = dict(c=[], logab=[], a=[], b=[], bistable=[], bM=[])
    for i in range(count):
        if name == 'planar_fronts':
            a, b = 3.0, np.linspace(1.7, 2.3, count)[i]
            r = TR.roots(a, b)
            col = np.arange(n * n) % n
            y0 = np.where(col < n // 2, r[-1], r[0]).astype(float)
        else:
            a, b, y0 = sample_schlogl(rng, 'high_a_low_b' if name == 'high_a_low_b' else 'train', i, n)
            if name == 'high_concentration':          # a, b above the training box
                a = rng.uniform(3.5, 4.0); b = rng.uniform(2.0, 3.5)
        x0 = TR.x_of_y(y0)
        Cs = TR.solve_schlogl(n, a, b, x0, nsteps)
        out['c'].append(Cs); out['logab'].append([math.log(a), math.log(b)]); out['a'].append(a); out['b'].append(b)
        out['bistable'].append(len(TR.roots(a, b)) == 3); out['bM'].append(TR.maxwell_b(a))
    d = {k: torch.tensor(np.array(v)) for k, v in out.items()}
    d['n'] = n
    return d


def make_set_pc(name):
    count, nsteps, n, seed, kind = GEN_PC[name]
    rng = np.random.default_rng(seed)
    out = dict(c=[], U=[], dG=[])
    s = n // 8                                       # smooth fields scaled to the grid size
    for i in range(count):
        U = smooth_field(rng, n, 3 * s, 1.2) if kind == 'potential' else smooth_field(rng, n, 2 * s, 0.7)
        dG = rng.uniform(3, 4) if kind == 'drive' else rng.uniform(1, 3)
        X0 = 0.6 * np.exp(smooth_field(rng, n, 2 * s, 0.4)); Xp0 = 0.4 * np.exp(smooth_field(rng, n, 2 * s, 0.4))
        E0 = np.exp(smooth_field(rng, n, 2 * s, 0.4))
        out['c'].append(TR.solve_pc(n, U, dG, np.concatenate([X0, Xp0, E0]), nsteps)); out['U'].append(U); out['dG'].append(dG)
    d = {k: torch.tensor(np.array(v)) for k, v in out.items()}
    d['n'] = n
    return d


def mapk_initial_state(rng, hg, kind):
    V = hg.V
    c = {s: np.zeros(V) for s in TR.STATE}
    e1 = np.full(V, 1e-4); e1[TR.INPUT_SITES] = 0.3 * rng.uniform(0.5, 1.5, len(TR.INPUT_SITES))
    c['E1'] = e1
    for e in ('E2', 'P2', 'P3'):
        c[e] = TR.TOTALS[e] * rng.uniform(0.8, 1.2, V)
    for tier, forms in (('KKK', ['KKK', 'KKKp']), ('KK', ['KK', 'KKp', 'KKpp']), ('K', ['K', 'Kp', 'Kpp'])):
        alpha = np.array([6.0] + [1.0] * (len(forms) - 1)) if kind != 'phosphorylated' else np.array([1.0] * (len(forms) - 1) + [6.0])
        fr = rng.dirichlet(alpha, size=V)
        for j, f in enumerate(forms):
            c[f] = TR.TOTALS[tier] * rng.uniform(0.8, 1.2, V) * fr[:, j]
    for cx in ('ES1', 'ES2', 'ES3', 'ES4', 'ES5', 'ES6', 'ES7', 'ES8', 'ES9', 'ES10'):
        c[cx] = np.full(V, 1e-3)
    x = np.concatenate([c[s] for s in TR.STATE])
    return np.maximum(x, 1e-6)


def make_sets_mapk(names):
    """the MAPK sets from one random stream; sets not in names draw their random numbers but are not solved"""
    hg = TR.hg_of(TR.MAPK_N)
    rng = np.random.default_rng(0)
    data = {}
    for name, nb, (lo, hi), kind, ns in GEN_MAPK:
        out = dict(c=[], logc=[], sigma=[], turnover=[])
        for _ in range(nb):
            atp = np.exp(rng.uniform(np.log(lo), np.log(hi)))
            logc = np.log(np.array([atp, 1.0, 1.0]))
            x0 = mapk_initial_state(rng, hg, kind)
            if name not in names:
                continue
            Cs, sig, tov = TR.solve_mapk(hg, logc, x0, ns)
            out['c'].append(Cs); out['logc'].append(logc); out['sigma'].append(sig); out['turnover'].append(tov)
        if name in names:
            data[name] = {k: torch.tensor(np.array(v)) for k, v in out.items()}
            data[name]['n'] = TR.MAPK_N
    return data


def all_sets():
    return ['train'] + C.sets()


def generate(names=None, out=None):
    """write <data>/<set>.pt for every requested set (default: all)"""
    names = names or all_sets()
    out = out or C.DATA
    os.makedirs(out, exist_ok=True)
    if C.SYSTEM == 'mapk_cascade':
        res = make_sets_mapk(names)
    else:
        from multiprocessing import get_context
        fn = make_set_pc if C.SYSTEM == 'phosphorylation_cycle' else make_set_schlogl
        with get_context('fork').Pool(len(names)) as pool:
            res = dict(zip(names, pool.map(fn, names)))
    for name, d in res.items():
        torch.save(d, os.path.join(out, f'{name}.pt'))
        Cs = d['c'].numpy()
        print(f'{C.SYSTEM} {name}: {tuple(Cs.shape)} grid {d["n"]} | min {Cs.min():.3e} max {Cs.max():.3f}', flush=True)
    return res


def load_set(name, device='cpu'):
    d = torch.load(os.path.join(C.DATA, f'{name}.pt'))
    d['c'] = d['c'].to(device)
    if C.SYSTEM == 'mapk_cascade':
        d['drive'] = dict(logc=d['logc'].double().to(device))
    elif C.SYSTEM == 'phosphorylation_cycle':
        d['drive'] = dict(U=d['U'].double().to(device), dG=d['dG'].double().to(device))
    else:
        d['drive'] = dict(logab=d['logab'].double().to(device))
    return d


def train_norm():
    """training-set scalars used to normalise the inputs of the black-box baselines"""
    tr = torch.load(os.path.join(C.DATA, 'train.pt'))
    if C.SYSTEM == 'mapk_cascade':
        return dict(logATP_mean=float(tr['logc'][:, 0].mean()), logATP_std=float(tr['logc'][:, 0].std()),
                    c_scale=float(tr['c'].mean()), c_min=float(tr['c'].min()), c_max=float(tr['c'].max()))
    if C.SYSTEM == 'phosphorylation_cycle':
        V = tr['n'] ** 2
        return dict(dG_mean=float(tr['dG'].mean()), dG_std=float(tr['dG'].std()), U_std=float(tr['U'].std()),
                    c_scale=float(tr['c'].mean()), c_min=float(tr['c'].min()), c_max=float(tr['c'].max()), V=V)
    lab = tr['logab']
    return dict(lab_mean=lab.mean(0).tolist(), lab_std=lab.std(0).tolist(), c_scale=float(tr['c'].mean()),
                c_min=float(tr['c'].min()), c_max=float(tr['c'].max()))
