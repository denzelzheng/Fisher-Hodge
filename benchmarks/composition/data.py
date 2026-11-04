"""Data sets of the composition benchmarks: the data of the modules and of the coupled systems.  Seeds fixed here."""
import os

import numpy as np
import torch

from neuralchart.compose import compose_graphs
from ..reaction_networks.data import smooth_field
from . import truth as T

DT = 0.05
TINY = 1e-6
N_TRAJ, N_STEPS, N_TEST, T_TEST = 48, 100, 16, 200


def _lognormal(rng, n, shape=1, sigma=0.4):
    return shape * np.exp(sigma * rng.normal(size=n))


def conditions(dG, U):
    dG = torch.as_tensor(dG, dtype=torch.float64).reshape(-1)
    return dict(dG=dG, U=torch.as_tensor(U, dtype=torch.float64), chem_logc=torch.stack([0.5 * dG, -0.5 * dG], 1))


def _stack(rows, dG, U):
    return dict(c=torch.tensor(np.array(rows)), drive=conditions(dG, np.array(U)))


def module_reaction(seed=11):
    """the phosphorylation cycle on one site, well mixed"""
    rng = np.random.default_rng(seed)
    tr = T.Truth(T.reaction_hg(), [T.K1, T.K2], [T.K1, T.K2R])
    rows, dGs = [], []
    for _ in range(N_TRAJ):
        dG = rng.uniform(1, 3)
        c0 = np.array([_lognormal(rng, 1, 0.6)[0], _lognormal(rng, 1, 0.4)[0], _lognormal(rng, 1)[0]])
        rows.append(tr.solve(conditions([dG], np.zeros((1, 1))), c0, N_STEPS, DT)); dGs.append(dG)
    return _stack(rows, dGs, np.zeros((N_TRAJ, 1)))


def module_second(seed=12):
    """the second cycle on one site; the kinase K is constant along a trajectory, over the range Xp takes in the cascade"""
    rng = np.random.default_rng(seed)
    tr = T.Truth(T.second_hg(), [T.K1B, T.K2B], [T.K1B, T.K2RB])
    rows, dGs = [], []
    for _ in range(N_TRAJ):
        dG = rng.uniform(1, 3)
        c0 = np.array([_lognormal(rng, 1, 0.6)[0], _lognormal(rng, 1, 0.4)[0], rng.uniform(0.05, 2.0)])
        rows.append(tr.solve(conditions([dG], np.zeros((1, 1))), c0, N_STEPS, DT)); dGs.append(dG)
    return _stack(rows, dGs, np.zeros((N_TRAJ, 1)))


def module_transport(seed=13, n=8):
    """X and Xp on the grid in a potential, no reaction"""
    rng = np.random.default_rng(seed)
    tr = T.Truth(T.transport_hg(n), [T.D, T.D], [T.D, T.D], moving=('X', 'Xp'))
    rows, Us = [], []
    for _ in range(N_TRAJ):
        U = smooth_field(rng, n, 2, 0.7)
        c0 = np.concatenate([0.6 * np.exp(smooth_field(rng, n, 2, 0.4)), 0.4 * np.exp(smooth_field(rng, n, 2, 0.4))])
        rows.append(tr.solve(dict(U=torch.as_tensor(U)[None]), c0, N_STEPS, DT)); Us.append(U)
    return dict(c=torch.tensor(np.array(rows)), drive=dict(U=torch.tensor(np.array(Us))))


def cascade_truth(kind):
    hg = compose_graphs(T.reaction_hg(), T.second_hg(), {'K': 'Xp'})
    return T.Truth(hg, [T.K1, T.K2, T.K1B, T.K2B], [T.K1, T.K2R, T.K1B, T.K2RB], chi=T.CHI if kind == 'thermo' else 0.0, inhibit=kind == 'kinetic')


def cascade_set(kind, name, seed):
    """the coupled cascade (X, Xp, E, Y, Yp): train (48 x 100), train_conditions, strong_drive, high_concentration (16 x 200)"""
    count, steps = (N_TRAJ, N_STEPS) if name == 'train' else (N_TEST, T_TEST)
    rng = np.random.default_rng(seed)
    tr = cascade_truth(kind)
    rows, dGs = [], []
    for _ in range(count):
        dG = rng.uniform(3, 4) if name == 'strong_drive' else rng.uniform(1, 3)
        c0 = np.array([0.6, 0.4, 1.0, 0.6, 0.4]) * np.exp(0.4 * rng.normal(size=5))
        if name == 'high_concentration':
            c0[[0, 1, 3, 4]] *= 1.5
        rows.append(tr.solve(conditions([dG], np.zeros((1, 1))), c0, steps, DT)); dGs.append(dG)
    return _stack(rows, dGs, np.zeros((count, 1)))


CASCADE_SETS = dict(train=21, train_conditions=22, strong_drive=23, high_concentration=24)
OFFSET = dict(none=0, thermo=100, kinetic=200)


def generate(root):
    os.makedirs(root, exist_ok=True)
    for name, f in (('module_reaction', module_reaction), ('module_second', module_second), ('module_transport', module_transport)):
        p = os.path.join(root, name + '.pt')
        if not os.path.exists(p):
            torch.save(f(), p); print(name, flush=True)
    for kind, off in OFFSET.items():
        for name, seed in CASCADE_SETS.items():
            p = os.path.join(root, f'cascade_{kind}_{name}.pt')
            if not os.path.exists(p):
                torch.save(cascade_set(kind, name, seed + off), p); print(kind, name, flush=True)
