"""E4b: coarse-grain the lag-tau PROPAGATOR instead of the instantaneous generator (Markov-state-model style).
    Kbar(tau) = P exp(tau Q_fine) Pi ,   Kbar_ab = sum_{i in a} (pi_i/pi_a) sum_{j in b} [exp(tau Q)]_ij   (row-stochastic)
Q_fine = learned 32x32 TC-ME generator frozen at its stationary state (E4: closure and frozen agree to 3 digits).
Coarse rollout: rhobar_{n+1} = rhobar_n Kbar(tau) (row-vector convention = T^T rho in column convention).
Equivalent generator Lbar = logm(Kbar)/tau; legality check; nearest-neighbour (NN) truncation -> TC-ME form (kbar, phibar, Abar).
  python rg_lag.py           -> runs/rg/E4b.json, E4b.log, E4b_transient.png, appends the E4b section to summary.md"""
import os, sys, json, time
import numpy as np
import scipy.linalg as sla
import torch
from common import *
from models_tcme import TCME
from train_coarse import coarse_set
from train_tcme import jclean
from rg_coarse import (tcme_stationary, dense_from_rates, stat_dense, block_weights, block_sum, rg_rates, cross_sum, edge_flux,
                       sigma_of, lse_free_energy, param_form, rel, ODIR, TDIR, FIELDS, dev)

TAUS = [0.05, 0.2]
SEEDS = [0, 1]
SETS = ['test_id', 'test_oodpot']
TIDX = [20, 40, 200]          # t = T, 2T, 10T in units of DT


def lump_matrix(K, w, n, m=2):
    """K (n^2,n^2) fine row-stochastic, w in-block weights (n,n) -> Kbar (c^2,c^2)."""
    c = n // m
    lab = (np.arange(n)[:, None] // m) * c + (np.arange(n)[None, :] // m)
    Pm = np.zeros((n * n, c * c)); Pm[np.arange(n * n), lab.ravel()] = 1.0      # block-sum (fine -> coarse)
    Pi = (Pm * w.ravel()[:, None]).T                                              # (c^2, n^2): Pi_{a i} = pi_i / pi_a
    return Pi @ K @ Pm


def nn_rates_from_dense(L, c):
    """dense coarse generator -> one-way NN rates (2,c,c) [tail->head, head->tail] (grid_edges ordering)."""
    from train_tcme import grid_edges
    E = grid_edges(c)
    qf = L[E[:, 0], E[:, 1]].reshape(2, c, c); qb = L[E[:, 1], E[:, 0]].reshape(2, c, c)
    return qf, qb


def main():
    data = torch.load(os.path.join(RUNS, 'data', 'S2.pt'))
    E4 = json.load(open(os.path.join(ODIR, 'E4.json')))
    LOG = open(os.path.join(ODIR, 'E4b.log'), 'w')

    def log(*a):
        s = ' '.join(str(x) for x in a); print(s, flush=True); LOG.write(s + '\n'); LOG.flush()

    from train_tcme import grid_edges
    E16 = grid_edges(16); nnmask = np.zeros((256, 256), bool); nnmask[E16[:, 0], E16[:, 1]] = True; nnmask[E16[:, 1], E16[:, 0]] = True
    offd = ~np.eye(256, dtype=bool)
    out = {}
    for setname in SETS:
        sd = data[setname]; F32 = {k: sd[k].to(dev) for k in FIELDS}; B = sd['rho'].shape[0]
        s16 = coarse_set(sd, 16); Y16 = s16['rho'].double().numpy()
        sref16 = np.array(E4['ref'][setname]['sigma16'])
        for seed in SEEDS:
            t0 = time.time()
            fine = TCME('S2').to(dev); fine.load_state_dict(torch.load(os.path.join(TDIR, f'S2_tcme_s{seed}.pt'))); fine.eval()
            with torch.no_grad():
                ctx = fine.prepare(F32)
            pi32, qf, qb, _ = tcme_stationary(fine, ctx, B, N)
            P16 = block_sum(torch.tensor(pi32), 2).numpy() / 4
            w2 = block_weights(torch.tensor(pi32), 2).numpy()
            J32 = (edge_flux(torch.tensor(pi32), torch.tensor(qf), torch.tensor(qb))[0] - edge_flux(torch.tensor(pi32), torch.tensor(qf), torch.tensor(qb))[1])
            PJ16 = (cross_sum(J32, 2) / 4).numpy()                       # coarse native flux of the projected generator (= exact P J)
            phi32 = fine.phi(torch.tensor(pi32, device=dev, dtype=torch.float32), F32).double().detach().cpu()
            phib = lse_free_energy(phi32, 2)
            Qd = torch.tensor(np.stack([dense_from_rates(qf[b], qb[b]) for b in range(B)]), device=dev)
            # projected-generator baseline (E4 (a), frozen) rollout as discrete chain with exact exp for reference
            Qbar_f, Qbar_b = rg_rates(torch.tensor(qf), torch.tensor(qb), torch.tensor(w2), 2)
            R = {}
            for tau in TAUS:
                K = torch.linalg.matrix_exp(tau * Qd).cpu().numpy()          # (B,1024,1024) row-stochastic
                Kb = np.stack([lump_matrix(K[b], w2[b], N) for b in range(B)])
                stat_res = max(np.abs(P16[b].ravel() @ Kb[b] - P16[b].ravel()).max() / P16[b].max() for b in range(B))
                rowsum = np.abs(Kb.sum(-1) - 1).max(); kmin = Kb.min()
                # lag-tau flux consistency: pi_a Kbar_ab - pi_b Kbar_ba == sum_{i in a, j in b} (pi_i K_ij - pi_j K_ji)
                lagflux = []
                for b in range(B):
                    pf = pi32[b].ravel(); Fl = pf[:, None] * K[b]; Fl = Fl - Fl.T
                    c = N // 2; lab = ((np.arange(N)[:, None] // 2) * c + np.arange(N)[None, :] // 2).ravel()
                    Pm = np.zeros((N * N, c * c)); Pm[np.arange(N * N), lab] = 1
                    Fc = Pm.T @ Fl @ Pm
                    pc = P16[b].ravel() * 4 / (N * N) * (N * N) / 4       # density units consistent with pf (both mean-1 densities x area weight cancels)
                    Fb = (Pm.T @ pf)[:, None] * Kb[b]; Fb = Fb - Fb.T
                    lagflux.append(np.abs(Fb - Fc).max() / np.abs(Fc).max())
                # rollout
                nst = int(round(tau / DT)); steps = 200 // nst
                r = s16['rho'][:, 0].double().numpy().reshape(B, -1); traj = [r]
                for k in range(steps):
                    r = np.einsum('bi,bij->bj', r, Kb); traj.append(r)
                traj = np.stack(traj, 1).reshape(B, steps + 1, 16, 16)
                err = {str(t): rel(traj[:, t // nst], Y16[:, t]) for t in TIDX}
                # equivalent generator
                Ls, legal = [], []
                for b in range(B):
                    Lc = sla.logm(Kb[b]); imag = np.abs(Lc.imag).max() if np.iscomplexobj(Lc) else 0.0
                    Lr = np.real(Lc) / tau; Ls.append(Lr)
                    od = Lr[offd]; scale = np.abs(Lr[nnmask]).max()
                    legal.append(dict(min_offdiag_rel=float(od.min() / scale), n_neg=int((od < -1e-12 * scale).sum()),
                                      nonNN_mass_frac=float(np.abs(Lr[offd & ~nnmask]).sum() / np.abs(Lr[offd]).sum()),
                                      min_NN_rel=float(Lr[nnmask].min() / scale), imag=float(imag),
                                      stat_res=float(np.abs(P16[b].ravel() @ Lr).max() / (P16[b].max() * scale))))
                # NN truncation -> TC-ME form; check stationarity / flux / sigma, and roll it out
                nn = []
                Knn = []
                for b in range(B):
                    qfn, qbn = nn_rates_from_dense(Ls[b], 16)
                    qfn = np.clip(qfn, 1e-300, None); qbn = np.clip(qbn, 1e-300, None)
                    Ln = dense_from_rates(qfn, qbn)
                    pis = stat_dense(Ln).reshape(16, 16) * 256
                    tq = (torch.tensor(qfn)[None], torch.tensor(qbn)[None]); Pb = torch.tensor(P16[b])[None]
                    a_, b_ = edge_flux(Pb, *tq); Jn = (a_ - b_)[0].numpy()
                    k_, A_ = param_form(*tq, phib[b:b + 1])
                    rec = k_ * torch.exp(0.5 * (-grad_e(phib[b:b + 1]) + A_))
                    nn.append(dict(stat_res=float(np.abs(P16[b].ravel() @ Ln).max() / (P16[b].max() * np.abs(Ln).max())),
                                   pi_vs_Ppi=float(rel(pis[None], P16[b][None])[0]), J_vs_PJ=float(rel(Jn[None], PJ16[b][None])[0]),
                                   sigma=float(sigma_of(Pb, *tq)[0]), sigma_full=None, recon=float((rec - tq[0]).abs().max() / tq[0].abs().max()),
                                   k_over_Dn2=float(np.median(k_.numpy()) / (D * 256))))
                    Knn.append(sla.expm(DT * Ln))
                Knn = np.stack(Knn)
                r = s16['rho'][:, 0].double().numpy().reshape(B, -1); trn = [r]
                for k in range(200):
                    r = np.einsum('bi,bij->bj', r, Knn); trn.append(r)
                trn = np.stack(trn, 1).reshape(B, 201, 16, 16)
                err_nn = {str(t): rel(trn[:, t], Y16[:, t]) for t in TIDX}
                # lag-tau EPR rates (path EPR per unit time), coarse and fine; and projected-generator sigma
                sig_lag = []
                for b in range(B):
                    pb = P16[b].ravel() / 256; Fm = pb[:, None] * Kb[b]
                    msk = offd & (Fm > 0) & (Fm.T > 0)
                    sig_lag.append(float((Fm[msk] * np.log(Fm[msk] / Fm.T[msk])).sum() / tau))
                sig_lag = np.array(sig_lag)
                R[str(tau)] = dict(stat_res=float(stat_res), rowsum=float(rowsum), kmin=float(kmin), lagflux_relerr=float(max(lagflux)),
                                   err=err, err_curve=np.median(np.stack([rel(traj[:, i], Y16[:, i * nst]) for i in range(steps + 1)], 1), 0),
                                   t_curve=(np.arange(steps + 1) * tau), min_rho=float(traj.min()), mass=float(np.abs(traj.mean((2, 3)) - 1).max()),
                                   legal=legal, nn=nn, err_nn=err_nn,
                                   err_nn_curve=np.median(np.stack([rel(trn[:, i], Y16[:, i]) for i in range(201)], 1), 0),
                                   sigma_lag=sig_lag, sigma_lag_over_ref16=sig_lag / sref16,
                                   sigma_nn_over_ref16=np.array([d['sigma'] for d in nn]) / sref16)
                lg = R[str(tau)]['legal']
                log(f'[{setname} s{seed} tau={tau}] stat res {stat_res:.1e} rowsum {rowsum:.1e} min K {kmin:.1e} lagflux {max(lagflux):.1e} | '
                    'err T/2T/10T ' + ' '.join(f'{np.median(err[str(t)]):.3e}' for t in TIDX) +
                    f' | logm: min offdiag/scale {min(d["min_offdiag_rel"] for d in lg):.2e} n_neg max {max(d["n_neg"] for d in lg)} '
                    f'nonNN frac {np.median([d["nonNN_mass_frac"] for d in lg]):.3f} imag {max(d["imag"] for d in lg):.1e} stat {max(d["stat_res"] for d in lg):.1e} | '
                    f'NN-TCME: stat {max(d["stat_res"] for d in nn):.1e} pi err {np.median([d["pi_vs_Ppi"] for d in nn]):.2e} J err {np.median([d["J_vs_PJ"] for d in nn]):.2e} '
                    f'k/(Dn^2) {np.median([d["k_over_Dn2"] for d in nn]):.3f} err T/2T/10T ' + ' '.join(f'{np.median(err_nn[str(t)]):.3e}' for t in TIDX) +
                    f' | sigma_lag/ref16 {np.median(sig_lag / sref16):.3f} sigma_NN/ref16 {np.median(R[str(tau)]["sigma_nn_over_ref16"]):.3f}')
            sproj = np.array(E4['rg'][f'tcme_s{seed}'][setname]['sigma16'])
            R['proj_sigma16_over_ref16'] = sproj / sref16
            R['fine_sigma32_over_ref16'] = np.array(E4['rg'][f'tcme_s{seed}'][setname]['sigma32']) / sref16
            out.setdefault(setname, {})[f's{seed}'] = R
            log(f'[{setname} s{seed}] projected-generator sigma16/sigma_ref16 {np.median(R["proj_sigma16_over_ref16"]):.3f} | {time.time() - t0:.0f}s')
    json.dump(jclean(out), open(os.path.join(ODIR, 'E4b.json'), 'w'))
    report(out, E4)


def ms(v, fmt='{:.3g}'):
    v = np.array([x for x in v if x is not None], float)
    return (fmt + '±' + fmt).format(v.mean(), v.std()) if len(v) > 1 else fmt.format(v[0])


def report(out, E4):
    import glob
    C = {os.path.basename(f)[:-5]: json.load(open(f)) for f in glob.glob(os.path.join(ODIR, 'S2_*_n16_s*.json'))}
    E2 = json.load(open(os.path.join(TDIR, 'E2_coarse.json')))
    md = lambda x: float(np.median(np.asarray(x, float)))
    L = ['', '## E4b: lag-tau propagator coarse-graining (Markov-state-model style)', '',
         'Kbar(tau) = P exp(tau Q_fine) Pi (Pi_ia = pi_i/pi_a, P = block sum), Q_fine = trained 32x32 TC-ME generator at its stationary state. '
         'Coarse rollout rhobar_{n+1} = Kbar(tau)^T rhobar_n (tau = 0.2: one step = 4 DT). Lbar = logm(Kbar)/tau; "NN-TC-ME" = Lbar truncated to '
         'grid edges (non-NN entries dropped, negatives clipped), written in TC-ME form (kbar, phibar = lse phi, Abar) and rolled out with exp(DT Lbar_NN). '
         'sigma_ref,16 = sum j^2/(D rho) h^2 of the block-averaged true stationary state (same discrete formula on 16x16). Median over 16 traj, mean±std over 2 seeds.', '',
         '| coarse model (16x16) | T ID | 2T ID | 10T ID | T OOD | 2T OOD | 10T OOD |', '|---|---|---|---|---|---|---|']
    def row(lab, f):
        L.append(f'| {lab} | ' + ' | '.join(ms([f(s, sd, t) for sd in SEEDS]) for s in SETS for t in TIDX) + ' |')
    row('(a) projected generator (E4)', lambda s, sd, t: E4['rg'][f'tcme_s{sd}'][s]['rollout']['frozen']['relerr_median'][t])
    for tau in TAUS:
        row(f'lag propagator Kbar(tau={tau})', lambda s, sd, t, tau=tau: md(out[s][f's{sd}'][str(tau)]['err'][str(t)]))
    for tau in TAUS:
        row(f'NN-TC-ME from logm Kbar(tau={tau})', lambda s, sd, t, tau=tau: md(out[s][f's{sd}'][str(tau)]['err_nn'][str(t)]))
    row('(b) fine TC-ME rollout then lump (E2)', lambda s, sd, t: E2[f'tcme_s{sd}'][s]['errA_med'][t])
    for m, lab in [('tcme', 'TC-ME'), ('fisher_hodge', 'Fisher-Hodge'), ('flux_gnn', 'Flux-GNN')]:
        row(f'(c) {lab} trained on 16x16', lambda s, sd, t, m=m: C[f'S2_{m}_n16_s{sd}']['sets'][s]['relerr_median'][t])
    L += ['', '| check (worst over 2 seeds x 2 sets x 16 traj) | tau = 0.05 | tau = 0.2 |', '|---|---|---|']
    g = lambda tau, f: [f(out[s][f's{sd}'][str(tau)]) for s in SETS for sd in SEEDS]
    chk = [('Ppi stationary for Kbar (max |Ppi Kbar - Ppi| / max Ppi)', lambda R: R['stat_res'], max, '{:.1e}'),
           ('row sums - 1; min entry of Kbar', lambda R: (R['rowsum'], R['kmin']), None, None),
           ('lag-tau block flux == fine lag-tau flux (rel. max)', lambda R: R['lagflux_relerr'], max, '{:.1e}'),
           ('logm: most negative off-diagonal / max NN rate', lambda R: min(d['min_offdiag_rel'] for d in R['legal']), min, '{:.2e}'),
           ('logm: # negative off-diagonal entries (max per traj, of 65280)', lambda R: max(d['n_neg'] for d in R['legal']), max, '{:.0f}'),
           ('logm: share of |off-diagonal| mass on non-NN pairs (median)', lambda R: np.median([d['nonNN_mass_frac'] for d in R['legal']]), np.median, '{:.3f}'),
           ('logm: min NN rate / max NN rate', lambda R: min(d['min_NN_rel'] for d in R['legal']), min, '{:.2e}'),
           ('logm: max |imag|; Ppi Lbar = 0 residual', lambda R: (max(d['imag'] for d in R['legal']), max(d['stat_res'] for d in R['legal'])), None, None),
           ('NN-TC-ME: Ppi stationary residual (max)', lambda R: max(d['stat_res'] for d in R['nn']), max, '{:.1e}'),
           ('NN-TC-ME: stationary pi vs Ppi (median rel L2)', lambda R: np.median([d['pi_vs_Ppi'] for d in R['nn']]), np.median, '{:.2e}'),
           ('NN-TC-ME: Jbar(Ppi) vs P J_fine (median rel L2)', lambda R: np.median([d['J_vs_PJ'] for d in R['nn']]), np.median, '{:.2e}'),
           ('NN-TC-ME: kbar / (D n_16^2) (median)', lambda R: np.median([d['k_over_Dn2'] for d in R['nn']]), np.median, '{:.3f}'),
           ('NN-TC-ME: (k, phi, A) reconstructs rates', lambda R: max(d['recon'] for d in R['nn']), max, '{:.1e}')]
    for lab, f, agg, fmt in chk:
        cells = []
        for tau in TAUS:
            v = g(tau, f)
            if agg is None:
                v = np.array(v); cells.append(f'{np.abs(v[:, 0]).max():.1e}; {v[:, 1].min():.1e}' if 'row' in lab else f'{v[:, 0].max():.1e}; {v[:, 1].max():.1e}')
            else:
                cells.append(fmt.format(agg(v)))
        L.append(f'| {lab} | ' + ' | '.join(cells) + ' |')
    L += ['', '| EPR / sigma_ref,16 (median; ID / OOD) | value |', '|---|---|']
    sg = lambda f: ' / '.join(ms([md(f(out[s][f's{sd}'])) for sd in SEEDS]) for s in SETS)
    L.append(f'| fine TC-ME sigma_32 / sigma_ref,16 | {sg(lambda R: R["fine_sigma32_over_ref16"])} |')
    L.append(f'| projected generator sigma_16 / sigma_ref,16 (E4) | {sg(lambda R: R["proj_sigma16_over_ref16"])} |')
    for tau in TAUS:
        L.append(f'| lag-{tau} path EPR rate (1/tau) sum pi_a Kbar_ab log(pi_a Kbar_ab / pi_b Kbar_ba) / sigma_ref,16 | {sg(lambda R, tau=tau: R[str(tau)]["sigma_lag_over_ref16"])} |')
        L.append(f'| NN-TC-ME (tau={tau}) Schnakenberg sigma at Ppi / sigma_ref,16 | {sg(lambda R, tau=tau: R[str(tau)]["sigma_nn_over_ref16"])} |')
    for s in SETS:
        L.append(f'| scratch TC-ME sigma_16 / sigma_ref,16 ({s}) | ' + ms([md(np.array(E4['scratch'][f'tcme_s{sd}'][s]['16']['sigma']) / np.array(E4['ref'][s]['sigma16'])) for sd in SEEDS]) + ' |')
    L += ['', '![E4b](E4b_transient.png)', '']
    txt = '\n'.join(L)
    p = os.path.join(ODIR, 'summary.md'); base = open(p).read().split('\n## E4b:')[0].rstrip('\n')
    open(p, 'w').write(base + '\n' + txt)
    print(txt)
    # ---------- figure
    import matplotlib; matplotlib.use('Agg'); import matplotlib.pyplot as plt
    plt.rcParams.update({'font.size': 9, 'axes.spines.top': False, 'axes.spines.right': False})
    fig, axs = plt.subplots(1, 2, figsize=(10, 3.9))
    t201 = np.arange(201) * DT
    for ax, s in zip(axs, SETS):
        cur = lambda f: np.mean([np.asarray(f(sd), float) for sd in SEEDS], 0)
        ax.plot(t201[1:], cur(lambda sd: E4['rg'][f'tcme_s{sd}'][s]['rollout']['frozen']['relerr_median'])[1:], color='#2a78d6', lw=2, label='(a) projected generator (E4)')
        R0 = out[s]['s0']
        ax.plot(np.asarray(R0['0.05']['t_curve'])[1:], cur(lambda sd: out[s][f's{sd}']['0.05']['err_curve'])[1:], color='#6250d6', lw=2, label='lag propagator, tau = 0.05')
        ax.plot(np.asarray(R0['0.2']['t_curve'])[1:], cur(lambda sd: out[s][f's{sd}']['0.2']['err_curve'])[1:], color='#e87ba4', lw=2, marker='o', ms=4, label='lag propagator, tau = 0.2')
        ax.plot(t201[1:], cur(lambda sd: out[s][f's{sd}']['0.05']['err_nn_curve'])[1:], color='#6250d6', lw=1.5, ls=':', label='NN-TC-ME from logm Kbar(0.05)')
        ax.plot(t201[1:], cur(lambda sd: C[f'S2_tcme_n16_s{sd}']['sets'][s]['relerr_median'])[1:], color='#eb6834', lw=2, ls='--', label='(c) TC-ME trained on 16x16')
        ax.plot(t201[1:], cur(lambda sd: E2[f'tcme_s{sd}'][s]['errA_med'])[1:], color='#52514e', lw=1.5, ls='-.', label='(b) fine TC-ME, then lump')
        ax.set_yscale('log'); ax.set_xlabel('t  (T = 1)'); ax.set_ylabel('rel. L2 error vs block-averaged reference (median)')
        ax.set_title({'test_id': 'ID potentials', 'test_oodpot': 'OOD potentials'}[s], loc='left', fontsize=10); ax.grid(alpha=0.3, which='both')
    axs[0].legend(fontsize=7.5, frameon=False)
    fig.suptitle('E4b: 16x16 coarse rollouts from the trained 32x32 TC-ME (no coarse data) vs coarse training', fontsize=10, x=0.01, ha='left')
    fig.tight_layout(); fig.savefig(os.path.join(ODIR, 'E4b_transient.png'), dpi=160); plt.close(fig)


if __name__ == '__main__':
    main()
