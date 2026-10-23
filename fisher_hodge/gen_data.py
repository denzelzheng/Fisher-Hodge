"""Reference data: fine 64x64 finite-volume solver (Heun, dt_ref=1e-4) coarse-grained (2x2 mass-preserving
averaging) to the 32x32 model grid.  Systems:
  S1  Fokker-Planck     d_t rho = div(rho grad U) + D lap rho
  S2  non-equilibrium   S1 - div(rho v), v = grad^perp psi (exactly div-free on the staggered grid)
  S3  Fisher-KPP (WFR)  d_t rho = D lap rho + r(x) rho (1 - rho)   (r(x)=1+a*phi(x) heterogeneous growth)
Usage: python3 gen_data.py [S1 S2 S3]
"""
import sys, json, time, math
import numpy as np
import torch
from common import *

dev = 'cuda'
DT_REF = 1e-4
PER = int(round(DT / DT_REF))

SETS = {  # name: (n_traj, n_steps)
    'train': (64, T_TRAIN), 'test_id': (16, T_TEST), 'test_oodpot': (16, T_TEST), 'test_oodinit': (16, T_TEST)}

# distribution specs ---------------------------------------------------------
POT_ID = dict(K=(3, 5), a=(1.0, 3.0), s=(0.08, 0.16))          # U/D
POT_OOD = dict(K=(6, 8), a=(2.5, 4.5), s=(0.06, 0.10))
PSI_ID = dict(K=(3, 5), a=(1.5, 4.0), s=(0.08, 0.16))          # psi/D
PSI_OOD = dict(K=(6, 8), a=(3.0, 5.0), s=(0.06, 0.10))
IC_ID = dict(n=(1, 3), sig=(0.05, 0.10))
IC_OOD = dict(n=(4, 6), sig=(0.03, 0.06))
R_ID = dict(K=(3, 5), a=(1.0, 2.0), s=(0.08, 0.16), amp=0.5)   # r = 1 + amp*tanh(field)
R_OOD = dict(K=(6, 8), a=(1.5, 3.0), s=(0.06, 0.10), amp=0.7)
KPP_ID = dict(n=(1, 2), sig=(0.04, 0.07), A=(0.5, 1.0))
KPP_OOD = dict(n=(3, 4), sig=(0.03, 0.05), A=(0.3, 0.7))


def pgauss(X, Y, cx, cy, s):
    d2 = (np.sin(np.pi * (X - cx)) ** 2 + np.sin(np.pi * (Y - cy)) ** 2) / np.pi ** 2
    return np.exp(-d2 / (2 * s * s))


def grids(n):
    c = (np.arange(n) + 0.5) / n
    k = (np.arange(n) + 1.0) / n
    return np.meshgrid(c, c), np.meshgrid(k, k)   # nodes, corners  (X[a,b]=x_b, Y[a,b]=y_a)


def sample_field(rng, spec):
    K = rng.integers(spec['K'][0], spec['K'][1] + 1)
    return [(rng.uniform(), rng.uniform(), rng.choice([-1, 1]) * rng.uniform(*spec['a']), rng.uniform(*spec['s']))
            for _ in range(K)]


def eval_field(p, XY):
    X, Y = XY
    return sum(a * pgauss(X, Y, cx, cy, s) for cx, cy, a, s in p)


def sample_ic(rng, spec):
    n = rng.integers(spec['n'][0], spec['n'][1] + 1)
    w = rng.uniform(0.5, 1.5, n); w /= w.sum()
    return [(rng.uniform(), rng.uniform(), w[i], rng.uniform(*spec['sig'])) for i in range(n)]


def eval_ic(p, XY, eps=1e-3):
    X, Y = XY
    rho = sum(w * pgauss(X, Y, cx, cy, s) for cx, cy, w, s in p)
    rho = rho / rho.mean()
    return (rho + eps) / (1 + eps)


def sample_kpp_ic(rng, spec):
    n = rng.integers(spec['n'][0], spec['n'][1] + 1)
    return [(rng.uniform(), rng.uniform(), rng.uniform(*spec['A']), rng.uniform(*spec['sig'])) for _ in range(n)]


def eval_kpp_ic(p, XY, eps=1e-4):
    X, Y = XY
    return np.minimum(sum(A * pgauss(X, Y, cx, cy, s) for cx, cy, A, s in p) + eps, 1.0)


def fine_flux(sysname, rho, U, uvel):
    h = 1.0 / NF
    if sysname == 'S3':
        return -(D / h) * grad_e(rho)
    J = -(D / h) * grad_e(rho) - mean_e(rho) * grad_e(U) / h
    if sysname == 'S2':
        J = J + mean_e(rho) * uvel
    return J


def fine_rhs(sysname, rho, U, uvel, r):
    h = 1.0 / NF
    out = div_T(fine_flux(sysname, rho, U, uvel)) / h
    if sysname == 'S3':
        out = out + r * rho * (1 - rho)
    return out


def simulate(sysname, rho0, U, uvel, r, nsave):
    rho = rho0.clone()
    out = [coarsen(rho)]
    dt = DT_REF
    for k in range(nsave):
        for _ in range(PER):
            k1 = fine_rhs(sysname, rho, U, uvel, r)
            k2 = fine_rhs(sysname, rho + dt * k1, U, uvel, r)
            rho = rho + 0.5 * dt * (k1 + k2)
        out.append(coarsen(rho))
    return torch.stack(out, 1), rho


def make_set(sysname, setname, seed):
    ntraj, nsteps = SETS[setname]
    rng = np.random.default_rng(seed)
    oodpot = setname == 'test_oodpot'
    oodinit = setname == 'test_oodinit'
    (fXY, fC), (cXY, cC) = grids(NF), grids(N)
    Uf, Uc, Pf, Pc, R0f, rf, Uc_feat, nseeds, params = [], [], [], [], [], [], [], [], []
    for b in range(ntraj):
        pr = {}
        if sysname in ('S1', 'S2'):
            pu = sample_field(rng, POT_OOD if oodpot else POT_ID)
            Uf.append(D * eval_field(pu, fXY)); Uc.append(D * eval_field(pu, cXY)); pr['U'] = pu
            if sysname == 'S2':
                pp = sample_field(rng, PSI_OOD if oodpot else PSI_ID)
                Pf.append(D * eval_field(pp, fC)); Pc.append(D * eval_field(pp, cC)); pr['psi'] = pp
            else:
                Pf.append(np.zeros((NF, NF))); Pc.append(np.zeros((N, N)))
            pic = sample_ic(rng, IC_OOD if oodinit else IC_ID)
            R0f.append(eval_ic(pic, fXY)); pr['ic'] = pic
            rf.append(np.zeros((NF, NF))); nseeds.append(len(pic))
            Uc_feat.append(Uc[-1] / D)
        else:
            spec = R_OOD if oodpot else R_ID
            prr = sample_field(rng, spec)
            rfine = 1 + spec['amp'] * np.tanh(eval_field(prr, fXY))
            rcoar = 1 + spec['amp'] * np.tanh(eval_field(prr, cXY))
            rf.append(rfine); Uc_feat.append(rcoar - 1); pr['r'] = prr
            Uf.append(np.zeros((NF, NF))); Uc.append(np.zeros((N, N)))
            Pf.append(np.zeros((NF, NF))); Pc.append(np.zeros((N, N)))
            pic = sample_kpp_ic(rng, KPP_OOD if oodinit else KPP_ID)
            R0f.append(eval_kpp_ic(pic, fXY)); pr['ic'] = pic; nseeds.append(len(pic))
        params.append(pr)
    T = lambda L: torch.tensor(np.stack(L), dtype=torch.float64, device=dev)
    Uf, Pf, R0f, rf = T(Uf), T(Pf), T(R0f), T(rf)
    uvel = dual_dpsi(Pf) * NF       # velocity = (h u)/h
    t0 = time.time()
    traj, rho_fine_end = simulate(sysname, R0f, Uf, uvel, rf, nsteps)
    tsim = time.time() - t0
    Pc_t = T(Pc)
    out = dict(
        rho=traj.float().cpu(),
        U=T(Uc_feat).float().cpu(),                       # dimensionless input field (U/D or r-1)
        dpsi=(dual_dpsi(Pc_t) / D).float().cpu(),          # dimensionless dual-edge psi difference
        psin=(corner_to_node(Pc_t) / D).float().cpu(),
        nseeds=torch.tensor(nseeds),
        J_end=coarsen_flux(fine_flux(sysname, rho_fine_end, Uf, uvel)).float().cpu(),  # exact coarse-grained flux at final time
        U_phys=T(Uc).float().cpu(),
    )
    # sanity: coarse-grained fine rate == (1/h_c) d^T J_c
    rate_c = coarsen(fine_rhs(sysname, rho_fine_end, Uf, uvel, rf))
    Jc = coarsen_flux(fine_flux(sysname, rho_fine_end, Uf, uvel))
    reac = coarsen(rf * rho_fine_end * (1 - rho_fine_end)) if sysname == 'S3' else 0.
    chk = (rate_c - (div_T(Jc) * N + reac)).abs().max().item() / (rate_c.abs().max().item() + 1e-30)
    info = dict(seed=seed, ntraj=ntraj, nsteps=nsteps, sim_time_s=tsim, fluxcheck_relerr=chk,
                min_rho=float(traj.min()), mass_t0=float(traj[:, 0].mean()), mass_end=float(traj[:, -1].mean()))
    return out, info, params


if __name__ == '__main__':
    systems = sys.argv[1:] or ['S1', 'S2', 'S3']
    os.makedirs(os.path.join(RUNS, 'data'), exist_ok=True)
    base = {'S1': 1000, 'S2': 2000, 'S3': 3000}
    off = {'train': 0, 'test_id': 1, 'test_oodpot': 2, 'test_oodinit': 3}
    for s in systems:
        data, meta = {}, {}
        for setname in SETS:
            seed = base[s] + off[setname]
            out, info, params = make_set(s, setname, seed)
            data[setname] = out; meta[setname] = dict(info=info, params=params)
            print(s, setname, json.dumps(info), flush=True)
        torch.save(data, os.path.join(RUNS, 'data', f'{s}.pt'))
        with open(os.path.join(RUNS, 'data', f'{s}_meta.json'), 'w') as f:
            json.dump(dict(DT=DT, DT_REF=DT_REF, NF=NF, N=N, D=D, sets=meta,
                           specs=dict(POT_ID=POT_ID, POT_OOD=POT_OOD, PSI_ID=PSI_ID, PSI_OOD=PSI_OOD, IC_ID=IC_ID,
                                      IC_OOD=IC_OOD, R_ID=R_ID, R_OOD=R_OOD, KPP_ID=KPP_ID, KPP_OOD=KPP_OOD)), f,
                      default=float)
