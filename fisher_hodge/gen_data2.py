"""Misspecified-physics reference data (outside the Fisher-Hodge hypothesis class).  Same grids / protocol / splits as
gen_data.py (64^2 finite volumes, Heun dt=1e-4, 2x2 coarse-graining to 32^2; 64 train traj x 20 steps, 3 x 16 test traj x 200 steps).
  S4  nonlinear mobility      d_t rho = D lap rho + div( m(rho) grad U ),  m = rho^1.5 / sqrt(1 + rho/rho_c), rho_c = 5
                              (superlinear ~rho^1.5 up to rho ~ rho_c, tempered to linear above so that OOD potentials stay bounded)
  S5  exclusion / crowding    d_t rho = D lap rho + div( rho (1-rho) grad U )                 (rho in [0,1]; IC smoothly saturated at 0.95)
  S6  nonlocal aggregation    d_t rho = D lap rho + div( rho grad(U - chi W*rho) )            (Keller-Segel type, W periodic Gaussian)
Model inputs are the same as for S1 (U/D, zero psi): the mobility law and the interaction kernel are *unknown* to every model.
  S3b super-saturated training set for S3: same test sets as S3; in half of the 64 training trajectories an extra blob with
      amplitude 0.6-1.5 and width 0.15-0.30 is added on top of the capped KPP initial condition (local rho_0 up to ~2.5), so that the decay
      branch of the reaction (rho > carrying capacity) is observed *inside* the training window.
Usage: python3 gen_data2.py [S4 S5 S6 S3b]   |   python3 gen_data2.py calib   (chi scan for S6)
"""
import sys, json, time
import numpy as np
import torch
import gen_data as G
from gen_data import *
from common import *
if os.environ.get('FH_CPU'):
    G.dev = dev = 'cpu'

CHI_OVER_D = 0.3      # S6 interaction strength chi / D   (set by calibration, see runs/data/S6_calib.json)
SIGW = 0.10           # S6 kernel width
RHO_MEAN_S5 = 0.4     # S5 IC: 0.95 (1 - exp(-0.4 rho_S1 / 0.95))  (smooth saturation, mean ~0.2)
RHOC_S4 = 5.0


def kernel_fft(n, sigw):
    (X, Y), _ = grids(n)
    W = pgauss(X, Y, 0.5 / n, 0.5 / n, sigw)          # centred on node (0,0)
    W = W / (W.sum() / n ** 2)                          # unit integral: sum W h^2 = 1
    return torch.fft.rfft2(torch.tensor(W, dtype=torch.float64, device=dev))


def conv_W(rho, Wk):
    n = rho.shape[-1]
    return torch.fft.irfft2(torch.fft.rfft2(rho) * Wk, s=(n, n)) / n ** 2


def mobility(sysname, m):
    if sysname == 'S4':
        m = m.clamp_min(0)
        return m ** 1.5 / torch.sqrt(1 + m / RHOC_S4)
    if sysname == 'S5':
        return m * (1 - m)
    return m


def fine_flux2(sysname, rho, U, Wk, chi):
    h = 1.0 / NF
    Ueff = U - chi * conv_W(rho, Wk) if sysname == 'S6' else U
    return -(D / h) * grad_e(rho) - mobility(sysname, mean_e(rho)) * grad_e(Ueff) / h


def fine_rhs2(sysname, rho, U, Wk, chi):
    return div_T(fine_flux2(sysname, rho, U, Wk, chi)) / (1.0 / NF)


def simulate2(sysname, rho0, U, Wk, chi, nsave):
    rho = rho0.clone(); out = [coarsen(rho)]; dt = DT_REF
    for k in range(nsave):
        for _ in range(PER):
            k1 = fine_rhs2(sysname, rho, U, Wk, chi)
            k2 = fine_rhs2(sysname, rho + dt * k1, U, Wk, chi)
            rho = rho + 0.5 * dt * (k1 + k2)
        out.append(coarsen(rho))
    return torch.stack(out, 1), rho


def make_set2(sysname, setname, seed, chi):
    ntraj, nsteps = SETS[setname]
    rng = np.random.default_rng(seed)
    oodpot = setname == 'test_oodpot'; oodinit = setname == 'test_oodinit'
    (fXY, _), (cXY, _) = grids(NF), grids(N)
    Uf, Uc, R0f, nseeds, params = [], [], [], [], []
    for b in range(ntraj):
        pu = sample_field(rng, POT_OOD if oodpot else POT_ID)
        Uf.append(D * eval_field(pu, fXY)); Uc.append(D * eval_field(pu, cXY))
        pic = sample_ic(rng, IC_OOD if oodinit else IC_ID)
        r0 = eval_ic(pic, fXY)
        if sysname == 'S5':
            r0 = 0.95 * (1 - np.exp(-RHO_MEAN_S5 * r0 / 0.95))
        R0f.append(r0); nseeds.append(len(pic)); params.append(dict(U=pu, ic=pic))
    T = lambda L: torch.tensor(np.stack(L), dtype=torch.float64, device=dev)
    Uf, R0f = T(Uf), T(R0f)
    Wk = kernel_fft(NF, SIGW) if sysname == 'S6' else None
    t0 = time.time()
    traj, rend = simulate2(sysname, R0f, Uf, Wk, chi, nsteps)
    tsim = time.time() - t0
    Jend = fine_flux2(sysname, rend, Uf, Wk, chi)
    z = torch.zeros(ntraj, N, N)
    out = dict(rho=traj.float().cpu(), U=(T(Uc) / D).float().cpu(), dpsi=torch.zeros(ntraj, 2, N, N), psin=z.clone(),
               nseeds=torch.tensor(nseeds), J_end=coarsen_flux(Jend).float().cpu(), U_phys=T(Uc).float().cpu())
    rate_c = coarsen(fine_rhs2(sysname, rend, Uf, Wk, chi))
    chk = (rate_c - div_T(coarsen_flux(Jend)) * N).abs().max().item() / (rate_c.abs().max().item() + 1e-30)
    info = dict(seed=seed, ntraj=ntraj, nsteps=nsteps, sim_time_s=tsim, fluxcheck_relerr=chk, min_rho=float(traj.min()),
                max_rho=float(traj.max()), mass_t0=float(traj[:, 0].mean()), mass_end=float(traj[:, -1].mean()),
                mass_drift_max=float((traj.mean((-1, -2)) - traj[:, :1].mean((-1, -2))).abs().max()))
    return out, info, params


def make_S3b(seed=3100, nextra=32):
    """S3 training set with super-saturated initial conditions in the last `nextra` trajectories."""
    ntraj, nsteps = SETS['train']
    rng = np.random.default_rng(seed)
    (fXY, _), (cXY, _) = grids(NF), grids(N)
    rf, Uc_feat, R0f, nseeds, params = [], [], [], [], []
    for b in range(ntraj):
        prr = sample_field(rng, R_ID)
        rf.append(1 + R_ID['amp'] * np.tanh(eval_field(prr, fXY))); Uc_feat.append(R_ID['amp'] * np.tanh(eval_field(prr, cXY)))
        pic = sample_kpp_ic(rng, KPP_ID)
        r0 = eval_kpp_ic(pic, fXY)
        extra = None
        if b >= ntraj - nextra:
            extra = (rng.uniform(), rng.uniform(), rng.uniform(0.6, 1.5), rng.uniform(0.15, 0.30))   # broad, so that rho>1 persists over the training window
            cx, cy, A, sg = extra
            r0 = r0 + A * pgauss(fXY[0], fXY[1], cx, cy, sg)
        R0f.append(r0); nseeds.append(len(pic)); params.append(dict(r=prr, ic=pic, extra=extra))
    T = lambda L: torch.tensor(np.stack(L), dtype=torch.float64, device=dev)
    rf, R0f = T(rf), T(R0f)
    zf = torch.zeros_like(R0f)
    t0 = time.time()
    traj, rend = G.simulate('S3', R0f, zf, zf.unsqueeze(1).expand(-1, 2, -1, -1), rf, nsteps)
    out = dict(rho=traj.float().cpu(), U=T(Uc_feat).float().cpu(), dpsi=torch.zeros(ntraj, 2, N, N), psin=torch.zeros(ntraj, N, N),
               nseeds=torch.tensor(nseeds), J_end=torch.zeros(ntraj, 2, N, N), U_phys=torch.zeros(ntraj, N, N))
    info = dict(seed=seed, ntraj=ntraj, nsteps=nsteps, nextra=nextra, sim_time_s=time.time() - t0, min_rho=float(traj.min()),
                max_rho_t0=float(traj[:, 0].max()), max_rho_end=float(traj[:, -1].max()),
                frac_nodes_above_1_t0=float((traj[:, 0] > 1.0).float().mean()), frac_nodes_above_1_end=float((traj[:, -1] > 1.0).float().mean()))
    return out, info, params


def calib():
    """chi scan for S6 on 8 ID-like trajectories to t=10: aggregation strength vs. boundedness."""
    rng = np.random.default_rng(123)
    (fXY, _), _ = grids(NF), grids(N)
    Uf, R0f = [], []
    for b in range(8):
        Uf.append(D * eval_field(sample_field(rng, POT_ID), fXY)); R0f.append(eval_ic(sample_ic(rng, IC_ID), fXY))
    T = lambda L: torch.tensor(np.stack(L), dtype=torch.float64, device=dev)
    Uf, R0f = T(Uf), T(R0f); Wk = kernel_fft(NF, SIGW)
    ref, _ = simulate2('S1x', R0f, Uf, Wk, 0.0, 200)          # chi = 0 == S1
    res = {}
    for c in [0.5, 1.0, 2.0, 3.0]:
        tr, _ = simulate2('S6', R0f, Uf, Wk, c * D, 200)
        dev_ = ((tr - ref).flatten(2).norm(dim=-1) / ref.flatten(2).norm(dim=-1))
        res[c] = dict(max_rho=float(tr.max()), max_rho_ref=float(ref.max()), min_rho=float(tr.min()),
                      reldiff_T=float(dev_[:, 20].median()), reldiff_10T=float(dev_[:, 200].median()))
        print('chi/D', c, res[c], flush=True)
    for sysn in ['S4', 'S5']:
        r0 = 0.95 * (1 - torch.exp(-RHO_MEAN_S5 * R0f / 0.95)) if sysn == "S5" else R0f
        tr, _ = simulate2(sysn, r0, Uf, None, 0.0, 200)
        rf = ref if sysn == 'S4' else simulate2('S1x', r0, Uf, None, 0.0, 200)[0]
        dev_ = ((tr - rf).flatten(2).norm(dim=-1) / rf.flatten(2).norm(dim=-1))
        res[sysn] = dict(max_rho=float(tr.max()), min_rho=float(tr.min()), reldiff_T=float(dev_[:, 20].median()),
                         reldiff_10T=float(dev_[:, 200].median()))
        print(sysn, res[sysn], flush=True)
    os.makedirs(os.path.join(RUNS, 'data'), exist_ok=True)
    json.dump(res, open(os.path.join(RUNS, 'data', 'S6_calib.json'), 'w'), indent=1)


if __name__ == '__main__':
    args = sys.argv[1:]
    if args and args[0] == 'calib':
        calib(); sys.exit()
    systems = args or ['S4', 'S5', 'S6']
    os.makedirs(os.path.join(RUNS, 'data'), exist_ok=True)
    if 'S3b' in systems:
        systems.remove('S3b')
        base3 = torch.load(os.path.join(RUNS, 'data', 'S3.pt'))
        out, info, params = make_S3b()
        print('S3b train', json.dumps(info), flush=True)
        torch.save(dict(base3, train=out), os.path.join(RUNS, 'data', 'S3b.pt'))
        json.dump(dict(note='S3 test sets + super-saturated training set', train=dict(info=info, params=params)),
                  open(os.path.join(RUNS, 'data', 'S3b_meta.json'), 'w'), default=float)
    base = {'S4': 4000, 'S5': 5000, 'S6': 6000}
    off = {'train': 0, 'test_id': 1, 'test_oodpot': 2, 'test_oodinit': 3}
    for s in systems:
        data, meta = {}, {}
        for setname in SETS:
            out, info, params = make_set2(s, setname, base[s] + off[setname], CHI_OVER_D * D)
            data[setname] = out; meta[setname] = dict(info=info, params=params)
            print(s, setname, json.dumps(info), flush=True)
        torch.save(data, os.path.join(RUNS, 'data', f'{s}.pt'))
        json.dump(dict(DT=DT, DT_REF=DT_REF, NF=NF, N=N, D=D, chi_over_D=CHI_OVER_D, sigw=SIGW, rho_mean_S5=RHO_MEAN_S5, rhoc_S4=RHOC_S4, sets=meta),
                  open(os.path.join(RUNS, 'data', f'{s}_meta.json'), 'w'), default=float)
