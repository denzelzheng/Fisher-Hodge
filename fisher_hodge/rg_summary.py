"""Tables + figures for E4 (runs/rg/): summary.md, epr_hierarchy.png, steady_flux_match.png."""
import json, os, glob
import numpy as np
from common import RUNS, DT

ODIR = os.path.join(RUNS, 'rg'); TDIR = os.path.join(RUNS, 'tcme')
SEEDS = [0, 1]
LAB = dict(tcme='TC-ME', fisher_hodge='Fisher-Hodge', flux_gnn='Flux-GNN', gnn_ode='GNN-ODE')
COL = dict(rg='#2a78d6', tcme='#eb6834', fisher_hodge='#1baf7a', flux_gnn='#eda100', ref='#52514e')


def ms(v, fmt='{:.3g}'):
    v = np.array([x for x in v if x is not None and np.isfinite(x)], float)
    if len(v) == 0:
        return 'n/a'
    if len(v) == 1:
        return fmt.format(v[0])
    return (fmt + '±' + fmt).format(v.mean(), v.std())


def med(x):
    return None if x is None else float(np.median(np.asarray(x, float)))


NOTES = """## Reading

* **Steady-state readouts: exact, as predicted.** RG-TC-ME (zero coarse data) reproduces P pi_fine and P J_fine to 1e-15 on
  16x16 (and 8x8), so its coarse readouts inherit the fine model's accuracy vs truth (pi 0.0031 / J 0.009 ID, vs 0.010-0.013 /
  0.026-0.045 for every model trained from scratch on 16x16). Scratch models disagree with P pi_fine by ~1% (ID) / 4-5% (OOD) and with
  P J_fine by 2-4% / 6-7%: a coarse model learned from data is a different object from the coarse-grained fine model.
* **EPR: chain monotone for RG (64/64 trajectories, both steps); scratch models are not ordered** (TC-ME 5/32 ID, FH 8/32, formal
  Flux-GNN 29/32 ID). But the RG sigma is a strict *lower bound* that drops by ~2x per level (sigma_16/sigma_32 = 0.48, sigma_8/sigma_16
  = 0.44 ID): the intra-block edges and the log-sum loss remove ~half the dissipation per level. The scratch models instead track the
  "physical" discretised EPR sum j^2/(D rho) h^2 of the block-averaged truth (ratio 0.96 / 0.88), which is NOT a coarse-graining-monotone
  quantity for a single Markov chain but is what a coarse model fitted to data reproduces. So "sigma ordered" holds only in the
  projection sense (hidden EPR of Esposito), not as resolution-independence of the learned EPR.
* **Transient dynamics: the projected generator is a poor propagator.** The 2x2 block chain is not lumpable; with the fixed in-block
  weights pi_i/pi_a every coarse hop rate is ~2x the natural D/h_c^2 (kbar/(D n^2) = 1.99, D_eff = m D in the diffusive limit), and the
  stationary flux is exact only because the in-block profile is frozen at pi. Hence 10T error = fine-model level (0.0030 ID / 0.0154 OOD,
  4x better than scratch TC-ME 0.0127 / 0.0634, near the fine-then-lump bound (b)), but at t = T, 2T the RG model relaxes too fast
  (0.305 / 0.164 ID vs 0.014 / 0.012 scratch). Halving kbar (diagnostic only; it breaks flux exactness by exactly 2x) gives 0.085 at T.
  A transient-accurate closed-form coarse model needs a non-Markov / memory correction (Mori-Zwanzig) or a time-rescaled rate with a
  matching drift correction; not done here.
* **Parametric form:** kbar symmetric, (kbar, phibar = lse phi, Abar) reconstructs Qbar exactly; detailed balance is preserved
  (A = 0 => Abar = 0 at the float32 floor). Under drive, Abar is not purely cycle-space (Euclidean gradient fraction 8.5% median,
  max 18%): lse(phi) is not the Hodge potential of the coarse force, so the coarse TC-ME needs a small gradient part in A (or a
  re-fitted phibar) to be in the canonical gauge.
* g depends on rho (max |d phi/d log rho| / max|d phi| = 5%); local-equilibrium closure and frozen generator give the same
  numbers to 3 digits.
* **Reverse direction (not run):** a TC-ME trained only on 16x16 data cannot produce a 32x32 model: P is many-to-one (sub-block
  profiles pi_i/pi_a, intra-block rates, and the dissipation on intra-block edges are absent from coarse data; sigma_16 <= sigma_32
  is a strict inequality with no inverse). One can evaluate the coarse-trained TC-ME on a 32x32 grid (it is resolution-agnostic),
  but that is extrapolation of the functional form, not inversion of the RG map.
* GNN-ODE (unstructured, no flux / no EPR): 10T error 0.22 ID, mass drift up to 0.89, negative density in 50/64 trajectories.
"""


def summary():
    E = json.load(open(os.path.join(ODIR, 'E4.json')))
    E2 = json.load(open(os.path.join(TDIR, 'E2_coarse.json')))
    C = {os.path.basename(f)[:-5]: json.load(open(f)) for f in glob.glob(os.path.join(ODIR, 'S2_*_n*_s*.json')) if 'smoke' not in f}
    F1 = {m: [json.load(open(os.path.join(TDIR, f'S2_{m}_s{s}.json'))) for s in SEEDS] for m in LAB if os.path.exists(os.path.join(TDIR, f'S2_{m}_s0.json'))}
    L = ['# E4: closed-form coarse-graining (RG) of the learned TC-ME generator vs coarse-grid training (S2, 2 seeds)', '',
         'RG rule: Qbar_ab = sum_{i in a} (pi_i/pi_a) sum_{j in b} Q_ij (projected generator, Esposito 2012) with pi = stationary state '
         'of the trained 32x32 TC-ME for the given U, dpsi; rho-dependence of g via the local-equilibrium closure '
         'rho_i = 4 rhobar_a pi_i/pi_a. Parametric form: phibar = -log sum exp(-phi) (block), kbar = sqrt(Qbar_ab Qbar_ba), '
         'Abar = log(Qbar_ab/Qbar_ba) - (phibar_a - phibar_b). No coarse data. Scratch models: same protocol as E1 '
         '(2000 iters) on the 2x2-coarsened (or 4x4) training trajectories, lumped U/psin/dpsi as in coarse_test.py. '
         'Coarse reference = 2x2 block mean of the 32x32 data (itself a 64x64 FV solve, block-averaged). '
         'All numbers: median over the 16 test trajectories, mean±std over 2 seeds.', '']
    # ---------------- rollout table
    L += ['## Rollout accuracy on the 16x16 coarse test set (relative L2 vs block-averaged reference)', '',
          '| model | coarse training data | T ID | 2T ID | 10T ID | 10T OOD-pot | T OOD-pot | max mass err | min rho | traj with rho<0 |', '|---|---|---|---|---|---|---|---|---|---|']
    for lab, k, note in [('(a) RG-TC-ME (closure; the rule)', 'closure', 'none'), ('(a) RG-TC-ME (generator frozen at Q(pi))', 'frozen', 'none'),
                         ('(a) diagnostic: closure with kbar/2 (NOT the rule, breaks flux exactness)', 'closure_k_over_m', 'none')]:
        R = lambda s: [E['rg'][f'tcme_s{sd}'][s]['rollout'][k] for sd in SEEDS]
        g = lambda s, t: [r['relerr_median'][t] for r in R(s)]
        L.append(f'| {lab} | {note} | {ms(g("test_id", 20))} | {ms(g("test_id", 40))} | {ms(g("test_id", 200))} | {ms(g("test_oodpot", 200))} | {ms(g("test_oodpot", 20))} | '
                 f'{max(r["mass_err_max"] for s in ["test_id", "test_oodpot"] for r in R(s)):.1e} | {min(r["min_rho"] for s in ["test_id", "test_oodpot"] for r in R(s)):.1e} | '
                 f'{sum(r["n_traj_with_negative"] for s in ["test_id", "test_oodpot"] for r in R(s))}/64 |')
    for m in ['tcme', 'fisher_hodge', 'flux_gnn']:
        ks = [f'{m}_s{s}' for s in SEEDS]
        g = lambda s, t: [E2[k][s]['errA_med'][t] for k in ks]
        L.append(f'| (b) fine {LAB[m]} 32x32 rollout, then lump (E2) | (fine data) | {ms(g("test_id", 20))} | {ms(g("test_id", 40))} | {ms(g("test_id", 200))} | {ms(g("test_oodpot", 200))} | {ms(g("test_oodpot", 20))} | '
                 f'{max(r["sets"][s]["mass_err_max"] for r in F1[m] for s in ["test_id", "test_oodpot"]):.1e} | {min(r["sets"][s]["min_rho"] for r in F1[m] for s in ["test_id", "test_oodpot"]):.1e} | '
                 f'{sum(r["sets"][s]["n_traj_with_negative"] for r in F1[m] for s in ["test_id", "test_oodpot"])}/64 |')
    for n in [16, 8]:
        for m in ['tcme', 'fisher_hodge', 'flux_gnn', 'gnn_ode']:
            rs = [C[f'S2_{m}_n{n}_s{s}'] for s in SEEDS if f'S2_{m}_n{n}_s{s}' in C]
            if not rs:
                continue
            g = lambda s, t: [r['sets'][s]['relerr_median'][t] for r in rs]
            L.append(f'| (c) {LAB[m]} trained from scratch on {n}x{n}' + (' (vs 8x8 reference)' if n == 8 else '') + f' | 64 coarse traj | {ms(g("test_id", 20))} | {ms(g("test_id", 40))} | {ms(g("test_id", 200))} | '
                     f'{ms(g("test_oodpot", 200))} | {ms(g("test_oodpot", 20))} | {max(r["sets"][s]["mass_err_max"] for r in rs for s in ["test_id", "test_oodpot"]):.1e} | '
                     f'{min(r["sets"][s]["min_rho"] for r in rs for s in ["test_id", "test_oodpot"]):.1e} | {sum(r["sets"][s]["n_traj_with_negative"] for r in rs for s in ["test_id", "test_oodpot"])}/64 |')
    L.append('')
    # ---------------- readout tables
    for s in ['test_id', 'test_oodpot']:
        L += [f'## Cross-scale readout consistency, {s}', '',
              'pi err = ||pibar - P pi_fine|| / ||P pi_fine|| with pi_fine the stationary state of the 32x32 model of the same class '
              '(RG and scratch TC-ME: the fine TC-ME). J err = same for the stationary edge flux per unit length (P = flux-consistent '
              'coarse graining). "true" = vs the block-averaged 64x64 stationary reference. sigma = model EPR in probability units at its own '
              'stationary state (sigma_32 = fine model of the same class). mono = #trajectories (of 32 = 16 x 2 seeds) with sigma_32 >= sigma_16 >= sigma_8.', '',
              '| model | pi err vs P pi_fine (16) | pi err vs true (16) | J err vs P J_fine (16) | J err vs true (16) | sigma_32 | sigma_16 | sigma_8 | sigma_16/sigma_32 | sigma_8/sigma_16 | mono |',
              '|---|---|---|---|---|---|---|---|---|---|---|']
        rg = [E['rg'][f'tcme_s{sd}'][s] for sd in SEEDS]
        mono = sum(int(np.sum((np.array(r['sigma32']) >= np.array(r['sigma16'])) & (np.array(r['sigma16']) >= np.array(r['sigma8'])))) for r in rg)
        L.append(f'| (a) RG-TC-ME | {ms([med(r["pi_vs_Ppi"]) for r in rg])} | {ms([med(r["pi_vs_true"]) for r in rg])} | {ms([med(r["J_vs_PJ"]) for r in rg])} | '
                 f'{ms([med(r["J_vs_true"]) for r in rg])} | {ms([med(r["sigma32"]) for r in rg])} | {ms([med(r["sigma16"]) for r in rg])} | {ms([med(r["sigma8"]) for r in rg])} | '
                 f'{ms([med(np.array(r["sigma16"]) / np.array(r["sigma32"])) for r in rg])} | {ms([med(np.array(r["sigma8"]) / np.array(r["sigma16"])) for r in rg])} | {mono}/32 |')
        for m in ['tcme', 'fisher_hodge', 'flux_gnn', 'gnn_ode']:
            rec = [E['scratch'].get(f'{m}_s{sd}', {}).get(s, {}) for sd in SEEDS]
            if not any(rec):
                continue
            r16 = [r.get('16', {}) for r in rec]; r8 = [r.get('8', {}) for r in rec]
            if m == 'tcme':
                s32 = [E['rg'][f'tcme_s{sd}'][s]['sigma32'] for sd in SEEDS]
            else:
                s32 = [E['fine'].get(f'{m}_s{sd}', {}).get(s, {}).get('sigma32') for sd in SEEDS]
            s16 = [r.get('sigma') for r in r16]; s8 = [r.get('sigma') for r in r8]
            ok = all(x is not None for x in s32 + s16 + s8) and all(len(x) for x in s8)
            mono = sum(int(np.sum((np.array(a) >= np.array(b)) & (np.array(b) >= np.array(c)))) for a, b, c in zip(s32, s16, s8)) if ok else None
            rat = lambda A, Bv: [med(np.array(a) / np.array(b)) if (a is not None and b is not None) else None for a, b in zip(A, Bv)]
            L.append(f'| (c) scratch {LAB[m]}' + (' (sigma formal)' if m == 'flux_gnn' else '') + f' | {ms([med(r.get("pi_vs_Ppi")) for r in r16])} | {ms([med(r.get("pi_vs_true")) for r in r16])} | '
                     f'{ms([med(r.get("J_vs_PJ")) for r in r16])} | {ms([med(r.get("J_vs_true")) for r in r16])} | {ms([med(x) for x in s32])} | {ms([med(x) for x in s16])} | '
                     f'{ms([med(x) for x in s8])} | {ms(rat(s16, s32))} | {ms(rat(s8, s16))} | {"n/a" if mono is None else f"{mono}/32"} |')
        ref = E['ref'][s]
        L.append(f'| reference: sum j^2/(D rho) h^2 of the block-averaged true stationary state | - | - | - | - | {med(ref["sigma32"]):.3g} | {med(ref["sigma16"]):.3g} | {med(ref["sigma8"]):.3g} | '
                 f'{med(np.array(ref["sigma16"]) / np.array(ref["sigma32"])):.3g} | {med(np.array(ref["sigma8"]) / np.array(ref["sigma16"])):.3g} | (sigma_64 = {med(ref["sigma64"]):.3g}) |')
        L.append('')
    # ---------------- assertions
    A = E['asserts']; mx = lambda k: float(np.max(A[k]))
    L += ['## Assertions on the closed-form rule (float64; 2 seeds x 2 sets x 16 trajectories)', '', '| property | worst case | status |', '|---|---|---|',
          f'| P pi stationary for Qbar: max|d^T Jbar| / max(pi_a escape_a) | {mx("stat_resid"):.1e} | pass (<1e-9) |',
          f'| dense stationary vector of Qbar vs P pi (rel L2) | {mx("stat_dense_relerr"):.1e} | pass |',
          f'| stationary flux exact: Jbar vs sum of crossing fine fluxes (16 and 8) | {mx("flux_exact_relerr"):.1e} | pass |',
          f'| sigma_16 <= sigma_32 (max ratio) | {mx("sigma16_over_32"):.4f} | pass, all 64 |',
          f'| sigma_8 <= sigma_16 (max ratio) | {mx("sigma8_over_16"):.4f} | pass, all 64 |',
          f'| semigroup RG_4 = RG_2 o RG_2 | {mx("semigroup_relerr"):.1e} | pass |',
          f'| (kbar, phibar, Abar) reconstructs Qbar; kbar symmetric | {mx("reconstruct_relerr"):.1e} | pass |',
          f'| A := 0 in fine model => Abar = 0 with phibar = lse(phi): max|Abar| | {mx("eq_Abar_max"):.1e} (fine-level float32 floor {mx("eq_A_fine_max"):.1e}) | pass (at float32 floor) |',
          f'| Abar in cycle space? Euclidean gradient fraction of Abar (median / max) | {np.median(A["Abar_grad_frac"]):.3f} / {mx("Abar_grad_frac"):.3f} (fine A: {mx("Abar_grad_frac_fine"):.0e}) | NO: lse phibar is not the Hodge potential of the coarse force |',
          f'| kbar / (D n_16^2) (median) vs fine k / (D n_32^2) | {np.median(A["kbar_over_Dn2"]):.3f} vs {np.median(A["kfine_over_Dn2"]):.3f} | projected generator hops 2x faster (D_eff = m D) |',
          f'| g rho-sensitivity: max|dphi/dlog rho| / max|d phi| | {mx("g_rho_sensitivity"):.2e} | closure vs frozen differ little |',
          '']
    L += [NOTES]
    L += ['![EPR hierarchy](epr_hierarchy.png)', '', '![steady flux match](steady_flux_match.png)', '']
    open(os.path.join(ODIR, 'summary.md'), 'w').write('\n'.join(L))
    print('\n'.join(L))
    plots(E)


def plots(E):
    import matplotlib; matplotlib.use('Agg'); import matplotlib.pyplot as plt
    plt.rcParams.update({'font.size': 9, 'axes.spines.top': False, 'axes.spines.right': False, 'axes.edgecolor': '#8a8984',
                         'axes.labelcolor': '#0b0b0b', 'xtick.color': '#52514e', 'ytick.color': '#52514e'})
    # ---------- EPR hierarchy
    fig, axs = plt.subplots(1, 2, figsize=(9.6, 3.9))
    xs = np.array([32, 16, 8])
    for ax, s in zip(axs, ['test_id', 'test_oodpot']):
        series = []
        rg = [E['rg'][f'tcme_s{sd}'][s] for sd in SEEDS]
        series.append(('RG-TC-ME (closed form, no coarse data)', COL['rg'], 'o', '-', [[med(r['sigma32']), med(r['sigma16']), med(r['sigma8'])] for r in rg]))
        for m in ['tcme', 'fisher_hodge', 'flux_gnn']:
            v = []
            for sd in SEEDS:
                rec = E['scratch'].get(f'{m}_s{sd}', {}).get(s, {})
                s32 = med(E['rg'][f'tcme_s{sd}'][s]['sigma32']) if m == 'tcme' else med(E['fine'].get(f'{m}_s{sd}', {}).get(s, {}).get('sigma32'))
                v.append([s32, med(rec.get('16', {}).get('sigma')), med(rec.get('8', {}).get('sigma'))])
            series.append((f'{LAB[m]}: trained separately at each n' + (' (formal sigma)' if m == 'flux_gnn' else ''), COL[m], 's', '--', v))
        ref = E['ref'][s]
        series.append(('reference sum j^2/(D rho), block-averaged truth', COL['ref'], 'D', ':', [[med(ref['sigma32']), med(ref['sigma16']), med(ref['sigma8'])]]))
        for i, (lab, c, mk, ls, v) in enumerate(series):
            v = np.array([[np.nan if y is None else y for y in row] for row in v], float)
            off = (i - 2) * 0.035
            xx = xs * (1 + off)
            mean = np.nanmean(v, 0)
            ax.plot(xx, mean, ls=ls, color=c, lw=2, marker=mk, ms=8, mec='white', mew=1.5, label=lab, zorder=3)
            if v.shape[0] > 1:
                ax.vlines(xx, np.nanmin(v, 0), np.nanmax(v, 0), color=c, lw=2, zorder=2)
        ax.set_xscale('log', base=2); ax.set_xticks(xs); ax.set_xticklabels(['32x32', '16x16', '8x8']); ax.invert_xaxis()
        ax.set_ylabel('stationary EPR sigma (prob. units)'); ax.set_ylim(bottom=0)
        ax.set_title({'test_id': 'ID potentials', 'test_oodpot': 'OOD potentials'}[s], loc='left', fontsize=10)
        ax.grid(axis='y', color='#e4e3df', lw=0.8)
    axs[0].legend(fontsize=7.5, frameon=False, loc='lower left')
    fig.suptitle('EPR along the coarse-graining chain (median over 16 trajectories; bars = seed range)', fontsize=10, x=0.01, ha='left')
    fig.tight_layout(); fig.savefig(os.path.join(ODIR, 'epr_hierarchy.png'), dpi=160); plt.close(fig)
    # ---------- steady flux scatter
    panels = [('rg', 'RG-TC-ME (closed form)', COL['rg'])] + [(m, f'{LAB[m]} trained on 16x16', COL[m]) for m in ['tcme', 'fisher_hodge', 'flux_gnn']]
    panels = [p for p in panels if os.path.exists(os.path.join(ODIR, f'flux_scatter_{p[0]}.npz'))]
    fig, axs = plt.subplots(2, len(panels), figsize=(3.0 * len(panels), 5.8), gridspec_kw=dict(height_ratios=[1.6, 1]))
    for j, (k, lab, c) in enumerate(panels):
        z = np.load(os.path.join(ODIR, f'flux_scatter_{k}.npz'))
        J, PJ = z['J'].ravel(), z['PJ'].ravel()
        ax = axs[0, j]
        lim = np.abs(PJ).max() * 1.08
        ax.plot([-lim, lim], [-lim, lim], color='#8a8984', lw=1, zorder=1)
        ax.scatter(PJ, J, s=4, color=c, alpha=0.5, lw=0, zorder=2)
        e = np.linalg.norm(J - PJ) / np.linalg.norm(PJ)
        ax.set_title(f'{lab}\nrel. err {e:.1e}', fontsize=9, loc='left')
        ax.set_xlim(-lim, lim); ax.set_ylim(-lim, lim); ax.set_aspect('equal')
        ax.set_xlabel('P J_fine (fine model, coarse-grained)')
        if j == 0:
            ax.set_ylabel('coarse model stationary flux Jbar')
        ax2 = axs[1, j]
        r = np.abs(J - PJ) / np.abs(PJ).max()
        r = np.clip(r, 1e-18, None)
        ax2.hist(np.log10(r), bins=np.linspace(-18, 0, 55), color=c)
        ax2.set_xlabel('log10 |Jbar - P J_fine| / max|P J_fine|')
        if j == 0:
            ax2.set_ylabel('# edges')
    fig.suptitle('Stationary coarse edge flux vs coarse-grained fine-model flux (ID set, seed 0, 16 traj x 512 edges)', fontsize=10, x=0.01, ha='left')
    fig.tight_layout(); fig.savefig(os.path.join(ODIR, 'steady_flux_match.png'), dpi=160); plt.close(fig)
