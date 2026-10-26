"""Tables + figures for C2 (Schlogl) and C3 (Brusselator, + equilibrium control C3eq).
python analyze_c23.py  -> runs/C2_summary.md, runs/C3_summary.md, runs/C2_interface.png, C2_front_speed.png, C2_switching.png,
                          C3_phase.png, C3_turing.png, C3_sigma.png, C3eq_control.png, runs/C23_metrics.json
Notes are appended from runs/C2_notes.md / runs/C3_notes.md if present."""
import os, sys, json
import numpy as np
import torch
from scipy.signal import find_peaks
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from crn2 import RUNS, build_graph_np
import gen_schlogl as GS
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

CD = os.path.join(RUNS, 'C23')
MODELS = ['hyperme', 'hyperme_rev', 'crnn', 'gnn']
NAMES = {'hyperme': 'HyperME', 'hyperme_rev': 'B3 HyperME (A=0)', 'crnn': 'B1 CRNN', 'gnn': 'B2 GNN-ODE'}
SHORT = {'hyperme': 'HyperME', 'hyperme_rev': 'B3', 'crnn': 'B1', 'gnn': 'B2'}
COL = {'truth': 'k', 'hyperme': '#1f77b4', 'hyperme_rev': '#2ca02c', 'crnn': '#ff7f0e', 'gnn': '#d62728'}
SEEDS = [0, 1]
OUT = {}


def load(bench):
    R, Z = {}, {}
    for m in MODELS:
        for s in SEEDS:
            f = os.path.join(CD, f'{bench}_{m}_s{s}.json')
            if os.path.exists(f):
                R[(m, s)] = json.load(open(f))
                Z[(m, s)] = dict(np.load(os.path.join(CD, f'{bench}_{m}_s{s}_eval.npz')))
    return R, Z


def fmt(v, f='{:.3g}'):
    if v is None or (isinstance(v, (float, np.floating)) and not np.isfinite(v)):
        return 'n/a'
    if isinstance(v, (bool, np.bool_)):
        return str(bool(v))
    return f.format(v)


def both(R, m, getter, f='{:.3g}'):
    vals = []
    for s in SEEDS:
        if (m, s) not in R:
            vals.append('–'); continue
        try:
            vals.append(fmt(getter(R[(m, s)], s), f))
        except (KeyError, TypeError, IndexError, ValueError):
            vals.append('n/a')
    return ' / '.join(vals)


def n_extrema(x, rel=0.01):
    x = np.asarray(x, float)
    if not np.isfinite(x).all():
        return -1
    pr = rel * max(np.abs(x).mean(), 1e-12)
    return len(find_peaks(x, prominence=pr)[0]) + len(find_peaks(-x, prominence=pr)[0])


# =============================================================================================== C2 Schlogl
def c2():
    R, Z = load('schlogl')
    if not R:
        return
    data = torch.load(os.path.join(RUNS, 'data', 'schlogl.pt'))
    DT = data['meta']['DT']; N = data['meta']['N']
    M = {}
    # ---------------- interface / bistable switching on test_id + test_ood
    def phi(C, thr):
        return (C[:, :, 0] > thr[:, None, None]).mean(-1)          # (B,T+1)
    truth_phi = {}
    for st in ['test_id', 'test_ood']:
        d = data[st]; truth_phi[st] = phi(d['c'].numpy(), d['thr'].numpy())
    for key in R:
        mm = {}
        for st in ['test_id', 'test_ood']:
            d = data[st]; thr = d['thr'].numpy()
            Rm = Z[key][f'{st}/R']
            pm, pt = phi(np.nan_to_num(Rm, nan=-1.0), thr), truth_phi[st]
            cls = lambda p: np.where(p < 0.05, 0, np.where(p > 0.95, 2, 1))
            dt_, dm_ = pt[:, -1] - pt[:, 0], pm[:, -1] - pm[:, 0]
            sig = np.abs(dt_) > 0.1
            mm[st] = dict(final_class_acc=float((cls(pm[:, -1]) == cls(pt[:, -1])).mean()),
                          direction_acc=float((np.sign(dm_[sig]) == np.sign(dt_[sig])).mean()) if sig.any() else None,
                          n_dir=int(sig.sum()), phi_MAE=float(np.abs(pm - pt).mean()),
                          extrema_excess=int(sum(n_extrema(Rm[b, :, 0].mean(-1)) > n_extrema(d['c'].numpy()[b, :, 0].mean(-1)) for b in range(len(thr)))),
                          max_over_true_max=float(np.nanmax(Rm) / d['c'].numpy().max()))
        M[key] = mm
    # truth side-of-Maxwell consistency (bistable cases)
    tcons = {}
    for st in ['test_id', 'test_ood']:
        d = data[st]; bis = d['bistable'].numpy().astype(bool); pt = truth_phi[st]
        dt_ = pt[:, -1] - pt[:, 0]; side = np.sign(d['b'].numpy() - d['bM'].numpy())
        sig = bis & (np.abs(dt_) > 0.1)
        tcons[st] = dict(n=int(sig.sum()), agree=float((np.sign(dt_[sig]) == side[sig]).mean()) if sig.any() else None)
    # ---------------- planar front speed vs b (test_front)
    d = data['test_front']; bb = d['b'].numpy(); xlo, xhi = d['xlo'].numpy(), d['xhi'].numpy()

    def speed(C):
        m = ((C[:, :, 0] - xlo[:, None, None]) / (xhi - xlo)[:, None, None]).mean(-1)     # (B,T+1)
        v = np.zeros(len(m))
        t = np.arange(m.shape[1]) * DT
        for i in range(len(m)):
            if not np.isfinite(m[i]).all():
                v[i] = np.nan; continue
            out = np.where((m[i] < 0.1) | (m[i] > 0.9))[0]
            j = max(out[0] if len(out) else len(t), 3)
            v[i] = np.polyfit(t[:j], m[i, :j], 1)[0] * N / 2          # two interfaces, sites per unit time
        return v
    vt = speed(d['c'].numpy())
    VS = {key: speed(Z[key]['test_front/R']) for key in R}

    def zero_cross(v):
        s = np.sign(v)
        for i in range(len(v) - 1):
            if np.isfinite(v[i]) and np.isfinite(v[i + 1]) and s[i] < 0 and s[i + 1] > 0:
                return bb[i] - v[i] * (bb[i + 1] - bb[i]) / (v[i + 1] - v[i])
        return np.nan
    # ---------------- stochastic switching (Omega = 100), single site, exact birth-death MFPT
    OMEGA = 100
    g = build_graph_np('schlogl', N)
    bd_b = None
    for key in R:
        if 'bd/b' in Z[key]:
            bd_b = Z[key]['bd/b']; break
    BD = {}
    if bd_b is not None:
        nmax = 6 * OMEGA; xs = np.maximum(np.arange(nmax + 1), 1e-9) / OMEGA
        lt, mt = [], []
        for b in bd_b:
            a = 3.0
            lt.append(OMEGA * (GS.K1 * a * xs ** 2 + GS.K2R * b)); mt.append(OMEGA * (GS.K1R * xs ** 3 + GS.K2 * xs))
        BD['truth'] = (np.array(lt), np.array(mt))
        for key in R:
            if 'bd/lam' in Z[key]:
                BD[key] = (Z[key]['bd/lam'], Z[key]['bd/mu'])

    def mfpt(lam, mu, i, j):
        lam = np.maximum(lam, 1e-300); mu = np.maximum(mu, 1e-300)
        lp = np.concatenate([[0.0], np.cumsum(np.log(lam[:-1]) - np.log(mu[1:]))])
        if j > i:
            cs = np.logaddexp.accumulate(lp)
            k = np.arange(i, j)
            return float(np.sum(np.exp(cs[k] - np.log(lam[k]) - lp[k])))
        cs = np.logaddexp.accumulate(lp[::-1])[::-1]
        k = np.arange(j + 1, i + 1)
        return float(np.sum(np.exp(cs[k] - np.log(mu[k]) - lp[k])))
    TAU = {}
    for key, (lam, mu) in BD.items():
        up, dn = [], []
        for ib, b in enumerate(bd_b):
            r = GS.roots(3.0, b)
            nlo, nmid, nhi = (int(round(OMEGA * x)) for x in r)
            up.append(mfpt(lam[ib], mu[ib], nlo, nmid)); dn.append(mfpt(lam[ib], mu[ib], nhi, nmid))
        TAU[key] = (np.array(up), np.array(dn))

    def stoch_maxwell(up, dn):
        r = np.log(up) - np.log(dn)
        for i in range(len(r) - 1):
            if np.isfinite(r[i]) and np.isfinite(r[i + 1]) and np.sign(r[i]) != np.sign(r[i + 1]):
                return bd_b[i] - r[i] * (bd_b[i + 1] - bd_b[i]) / (r[i + 1] - r[i])
        return np.nan
    # Gillespie (SSA) of the single-site birth-death chain at the truth's grid point closest to its stochastic Maxwell point
    SSA = {}
    if 'truth' in TAU:
        bs_true = stoch_maxwell(*TAU['truth'])
        ib = int(np.nanargmin(np.abs(bd_b - bs_true))) if np.isfinite(bs_true) else 6
        r = GS.roots(3.0, bd_b[ib]); nlo, nhi = int(round(OMEGA * r[0])), int(round(OMEGA * r[-1]))
        for key in ['truth'] + [k for k in BD if k != 'truth' and k[1] == 0]:
            lam, mu = BD[key][0][ib], BD[key][1][ib]
            if max(TAU[key][0][ib], TAU[key][1][ib]) > 2e3:
                SSA[key] = None; continue
            rng = np.random.default_rng(0)
            n, t, state, tlast, dw = nlo, 0.0, 0, 0.0, {0: [], 1: []}
            tot = lam + mu
            pu = lam / np.maximum(tot, 1e-300)
            nev = 0
            while nev < 12_000_000 and len(dw[0]) + len(dw[1]) < 400:
                t += rng.exponential(1.0 / tot[n]); n = n + 1 if rng.random() < pu[n] else n - 1
                n = min(max(n, 0), len(lam) - 1); nev += 1
                if state == 0 and n >= nhi:
                    dw[0].append(t - tlast); tlast = t; state = 1
                elif state == 1 and n <= nlo:
                    dw[1].append(t - tlast); tlast = t; state = 0
            SSA[key] = dict(b=float(bd_b[ib]), dwell_low=dw[0], dwell_high=dw[1], events=nev)
    # ---------------- tables
    L = ['# C2: Schlogl autocatalytic model on an 8x8 periodic grid (bistable, many-to-many hyperedges)', '',
         'R1: A + 2X <-> 3X (k1 = 1, k1\' = 0.5), R2: X <-> B (k2 = 5, k2\' = 1), X diffuses (D = 1). A, B chemostatted (concentrations a, b).',
         'f(x) = a x^2 - 0.5 x^3 - 5x + b. At a = 3: three positive roots 2-sqrt2, 2, 2+sqrt2 at b = 2 (outer stable, f\' = -2; middle unstable, f\' = +1); '
         'bistable for b in (1.456, 2.544); Maxwell point b_M = 2. Local cycle affinity R1+R2 = log(10 a / b) in [2.2, 3.4] (ID), [3.5, 4.6] (OOD).',
         'Train: 48 traj x 20 steps (DT = 0.05, T = 1), a in [2.85, 3.10], b in [1.0, 3.2], half bistable / half monostable, random high/low domains. '
         'Test ID 16 x 200 steps (same law); OOD 16 (a in [3.15, 3.35], b in [0.3, 1.0], all bistable, both sides of their Maxwell point); '
         'front 13 (a = 3, b = 1.70 ... 2.30, straight high-state stripe). 2000 iters, batch 8, CPU float64. Entries `seed0 / seed1`, medians over trajectories.', '']
    L += ['## (a) Accuracy, total X, positivity', '',
          '| model | params | ID T | ID 10T | OOD T | OOD 10T | front 10T | sum-X rel err 10T ID / OOD | traj with c<0 (ID/OOD/front) | min c | blown | substeps/DT | train s |',
          '|---|---|---|---|---|---|---|---|---|---|---|---|---|']
    for m in MODELS:
        row = [NAMES[m], both(R, m, lambda r, s: r['params'], '{:d}')]
        for st in ['test_id', 'test_ood']:
            for t in ['20', '200']:
                row.append(both(R, m, lambda r, s: r['sets'][st]['relL2_median'][t]))
        row.append(both(R, m, lambda r, s: r['sets']['test_front']['relL2_median']['200']))
        row.append(both(R, m, lambda r, s: r['sets']['test_id']['totX_relerr_10T_median']) + ' ; ' + both(R, m, lambda r, s: r['sets']['test_ood']['totX_relerr_10T_median']))
        row.append(both(R, m, lambda r, s: ','.join(str(r['sets'][q]['n_traj_negative']) for q in ['test_id', 'test_ood', 'test_front']), '{}'))
        row.append(both(R, m, lambda r, s: min(r['sets'][q]['min_c'] for q in ['test_id', 'test_ood', 'test_front']), '{:.2e}'))
        row.append(both(R, m, lambda r, s: sum(r['sets'][q]['n_blown'] for q in ['test_id', 'test_ood', 'test_front']), '{:d}'))
        row.append(both(R, m, lambda r, s: r['sets']['test_id']['substeps_per_DT_mean'], '{:.1f}'))
        row.append(both(R, m, lambda r, s: r['train_time_s'], '{:.0f}'))
        L.append('| ' + ' | '.join(row) + ' |')
    L += ['', '## (b) Interface propagation and bistable switching', '',
          'phi(t) = fraction of sites above the true unstable root (inflection point if monostable). Final class: phi(10T) < 0.05 / > 0.95 / mixed. '
          'Direction: sign of phi(10T) - phi(0) on trajectories where the truth changes by > 0.1. '
          f'Truth check: on bistable trajectories the switching direction equals sign(b - b_M) in {tcons["test_id"]["agree"]} (ID, n={tcons["test_id"]["n"]}) and '
          f'{tcons["test_ood"]["agree"]} (OOD, n={tcons["test_ood"]["n"]}) of cases.', '',
          '| model | final-class acc ID / OOD | direction acc ID / OOD | phi MAE ID / OOD |', '|---|---|---|---|']
    for m in MODELS:
        L.append(f'| {NAMES[m]} | ' + both(M, m, lambda r, s: r['test_id']['final_class_acc'], '{:.2f}') + ' ; ' + both(M, m, lambda r, s: r['test_ood']['final_class_acc'], '{:.2f}')
                 + ' | ' + both(M, m, lambda r, s: r['test_id']['direction_acc'], '{:.2f}') + ' ; ' + both(M, m, lambda r, s: r['test_ood']['direction_acc'], '{:.2f}')
                 + ' | ' + both(M, m, lambda r, s: r['test_id']['phi_MAE'], '{:.3f}') + ' ; ' + both(M, m, lambda r, s: r['test_ood']['phi_MAE'], '{:.3f}') + ' |')
    L += ['', 'Planar front speed v (sites per unit time; > 0: high state invades) at a = 3, measured from the stripe test set. '
          'b* = zero crossing of v(b) (true Maxwell point b_M = 2).', '',
          '| b | truth | ' + ' | '.join(f'{SHORT[m]} s0 / s1' for m in MODELS) + ' |', '|---|---|' + '---|' * len(MODELS)]
    for i, b in enumerate(bb):
        L.append(f'| {b:.2f} | {vt[i]:+.3f} | ' + ' | '.join(' / '.join(fmt(VS[(m, s)][i], '{:+.3f}') if (m, s) in VS else '–' for s in SEEDS) for m in MODELS) + ' |')
    L.append(f'| b* | {fmt(zero_cross(vt), "{:.3f}")} | ' + ' | '.join(' / '.join(fmt(zero_cross(VS[(m, s)]), '{:.3f}') if (m, s) in VS else '–' for s in SEEDS) for m in MODELS) + ' |')
    vrel = {key: float(np.nanmedian(np.abs(VS[key] - vt) / np.maximum(np.abs(vt), 0.05))) for key in VS}
    L.append(f'| median rel. speed err | – | ' + ' | '.join(' / '.join(fmt(vrel.get((m, s)), '{:.2f}') for s in SEEDS) for m in MODELS) + ' |')
    L += ['', '## (c) Blow-up / spurious oscillation', '',
          'Extrema excess = number of trajectories whose site-mean X(t) has more extrema (prominence > 1% of the mean) than the truth. '
          'max ratio = max over the rollout / max of the true data.', '',
          '| model | blown (ID/OOD/front) | extrema excess ID / OOD (of 16) | max ratio ID / OOD |', '|---|---|---|---|']
    for m in MODELS:
        L.append(f'| {NAMES[m]} | ' + both(R, m, lambda r, s: ','.join(str(r['sets'][q]['n_blown']) for q in ['test_id', 'test_ood', 'test_front']), '{}')
                 + ' | ' + both(M, m, lambda r, s: f"{r['test_id']['extrema_excess']},{r['test_ood']['extrema_excess']}", '{}')
                 + ' | ' + both(M, m, lambda r, s: r['test_id']['max_over_true_max'], '{:.2f}') + ' ; ' + both(M, m, lambda r, s: r['test_ood']['max_over_true_max'], '{:.2f}') + ' |')
    L += ['', '## (d) Entropy production and cycle affinity', '',
          'sigma(t) rel. L2 error over the 10T rollout (median). Affinity = learned site-mean <z, A> of the local R1+R2 cycle vs true log(10a/b).', '',
          '| model | sigma rel err ID / OOD / front | affinity MAE ID / OOD | learned affinity (ID mean) vs true | A_cyc rel err ID |', '|---|---|---|---|---|']
    for m in MODELS:
        if m == 'gnn':
            L.append(f'| {NAMES[m]} | n/a (no fluxes) | n/a | n/a | n/a |'); continue
        L.append(f'| {NAMES[m]} | ' + both(R, m, lambda r, s: '{:.3f}/{:.3f}/{:.3f}'.format(*[r['sets'][q]['sigma_relerr_median'] for q in ['test_id', 'test_ood', 'test_front']]), '{}')
                 + ' | ' + both(R, m, lambda r, s: r['sets']['test_id']['affinity_MAE'], '{:.2f}') + ' ; ' + both(R, m, lambda r, s: r['sets']['test_ood']['affinity_MAE'], '{:.2f}')
                 + ' | ' + both(R, m, lambda r, s: np.mean(r['sets']['test_id']['affinity_learned_sitemean']), '{:.2f}') + f" vs {np.mean(next(iter(R.values()))['sets']['test_id']['affinity_true_mean']):.2f}"
                 + ' | ' + both(R, m, lambda r, s: r['sets']['test_id']['Acycle_relerr_median'], '{:.3f}') + ' |')
    if TAU:
        bs = {k: stoch_maxwell(*v) for k, v in TAU.items()}
        L += ['', '## (e) Stochastic switching, single site, Omega = 100', '',
              'Birth-death chain with propensities Omega*a_h(n/Omega) from the (learned) rates; mean first-passage times from the low root to the unstable root '
              '(tau_up) and from the high root to the unstable root (tau_down), exact formula, a = 3. b_s = stochastic Maxwell point (tau_up = tau_down). '
              'B2 has no reaction decomposition (n/a).', '',
              '| b | truth tau_up / tau_down | ' + ' | '.join(f'{SHORT[m]} s0 tau_up / tau_down' for m in ['hyperme', 'hyperme_rev', 'crnn']) + ' |', '|---|---|---|---|---|']
        for ib, b in enumerate(bd_b):
            if ib % 2:
                continue
            L.append(f'| {b:.2f} | {TAU["truth"][0][ib]:.3g} / {TAU["truth"][1][ib]:.3g} | ' + ' | '.join(
                (f'{TAU[(m, 0)][0][ib]:.3g} / {TAU[(m, 0)][1][ib]:.3g}' if (m, 0) in TAU else '–') for m in ['hyperme', 'hyperme_rev', 'crnn']) + ' |')
        L.append(f'| b_s | {fmt(bs["truth"], "{:.3f}")} | ' + ' | '.join(' / '.join(fmt(bs.get((m, s)), '{:.3f}') for s in SEEDS) for m in ['hyperme', 'hyperme_rev', 'crnn']) + ' |')
        lerr = {k: float(np.median(np.abs(np.log10(TAU[k][0]) - np.log10(TAU['truth'][0])).tolist() + np.abs(np.log10(TAU[k][1]) - np.log10(TAU['truth'][1])).tolist()))
                for k in TAU if k != 'truth'}
        L.append('| median abs log10 tau error | 0 | ' + ' | '.join(' / '.join(fmt(lerr.get((m, s)), '{:.2f}') for s in SEEDS) for m in ['hyperme', 'hyperme_rev', 'crnn']) + ' |')
        if SSA:
            L += ['', 'Gillespie SSA of the same chain at b = %.2f (grid point closest to the true b_s), seed-0 models, up to 400 switches: mean dwell time low / high (count).' % list(v for v in SSA.values() if v)[0]['b'] if any(SSA.values()) else '', '']
            for k, v in SSA.items():
                nm = 'truth' if k == 'truth' else NAMES[k[0]]
                if v is None:
                    L.append(f'- {nm}: skipped (MFPT > 2e3, too slow)')
                else:
                    L.append(f'- {nm}: low {np.mean(v["dwell_low"]):.3g} ({len(v["dwell_low"])}), high {np.mean(v["dwell_high"]):.3g} ({len(v["dwell_high"])}); events {v["events"]}')
        OUT['C2_tau'] = {str(k): [v[0].tolist(), v[1].tolist()] for k, v in TAU.items()}
        OUT['C2_bs'] = {str(k): v for k, v in bs.items()}
    L += ['', '## HyperME assertions (positivity, sigma >= 0 on every rollout)', '']
    for m in ['hyperme', 'hyperme_rev']:
        for s in SEEDS:
            if (m, s) in R:
                r = R[(m, s)]
                L.append(f'- {m}_s{s}: ' + ', '.join(f"{q}: pos {r['sets'][q]['assert_positivity']}, sigma>=0 {r['sets'][q]['assert_sigma_nonneg']}" for q in r['sets']))
    nf = os.path.join(RUNS, 'C2_notes.md')
    if os.path.exists(nf):
        L += ['', open(nf).read()]
    open(os.path.join(RUNS, 'C2_summary.md'), 'w').write('\n'.join(L) + '\n')
    OUT['C2_interface'] = {str(k): v for k, v in M.items()}; OUT['C2_front_v'] = {'truth': vt.tolist(), **{str(k): v.tolist() for k, v in VS.items()}}
    OUT['C2_truth_maxwell_consistency'] = tcons
    # ---------------- figures
    d = data['test_id']; pt = truth_phi['test_id']
    bis = d['bistable'].numpy().astype(bool)
    cand = [i for i in range(len(bis)) if bis[i] and 0.15 < abs(pt[i, 40] - pt[i, 0]) and 0.02 < pt[i, 0] < 0.98]
    if not cand:
        cand = [int(np.argmax(np.abs(pt[:, -1] - pt[:, 0])))]
    do = data['test_ood']; po = truth_phi['test_ood']
    cando = [i for i in range(len(po)) if 0.15 < abs(po[i, 40] - po[i, 0])] or [0]
    rows = ['truth', 'hyperme', 'crnn', 'gnn', 'hyperme_rev']
    tt = [0, 20, 60, 200]
    df = data['test_front']
    blocks = [(df, 'test_front', 3, 'front'), (df, 'test_front', 9, 'front'), (d, 'test_id', cand[0], 'ID'), (do, 'test_ood', cando[0], 'OOD')]
    fig, ax = plt.subplots(len(rows), len(blocks) * len(tt), figsize=(len(blocks) * len(tt) * 1.2 + 0.8, len(rows) * 1.35))
    for bi, (dd, st, i, lab) in enumerate(blocks):
        Y = dd['c'].numpy()[i, :, 0]
        vmin, vmax = 0.0, 4.0
        for ri, rw in enumerate(rows):
            X = Y if rw == 'truth' else (Z[(rw, 0)][f'{st}/R'][i, :, 0] if (rw, 0) in Z else np.full_like(Y, np.nan))
            for ci, t in enumerate(tt):
                a_ = ax[ri, bi * len(tt) + ci]
                im = a_.imshow(X[t].reshape(N, N), vmin=vmin, vmax=vmax, cmap='viridis')
                a_.set_xticks([]); a_.set_yticks([])
                if ri == 0:
                    a_.set_title(f't={t * DT:g}', fontsize=8)
                if ci == 0 and bi == 0:
                    a_.set_ylabel('truth' if rw == 'truth' else SHORT[rw], fontsize=9)
        side = 'high invades' if dd['b'][i] > dd['bM'][i] else 'low invades'
        ax[0, bi * len(tt)].text(0, -3.0, f'{lab}: a={dd["a"][i]:.2f} b={dd["b"][i]:.2f} (b_M={dd["bM"][i]:.2f}, {side})', fontsize=8)
    fig.colorbar(im, ax=ax, shrink=0.6, label='X')
    fig.suptitle('C2 Schlogl film strip (T = 1): planar fronts on both sides of the Maxwell point, an ID and an OOD case; '
                 'rows truth | HyperME | B1 CRNN | B2 GNN-ODE | B3 (A=0), seed 0', fontsize=10)
    fig.savefig(os.path.join(RUNS, 'C2_interface.png'), dpi=130, bbox_inches='tight'); plt.close(fig)
    fig, ax = plt.subplots(1, 2 if TAU else 1, figsize=(11 if TAU else 5.5, 4))
    ax = np.atleast_1d(ax)
    ax[0].axhline(0, color='0.7', lw=0.8); ax[0].axvline(2.0, color='0.7', lw=0.8, ls='--')
    ax[0].plot(bb, vt, 'k-o', lw=2, label='truth')
    for m in MODELS:
        for s in SEEDS:
            if (m, s) in VS:
                ax[0].plot(bb, VS[(m, s)], '-', marker='.^'[s], color=COL[m], alpha=0.8, label=f'{SHORT[m]} s{s}')
    ax[0].set_xlabel('b (a = 3)'); ax[0].set_ylabel('front speed (sites / time), >0: high invades'); ax[0].legend(fontsize=7, ncol=2)
    ax[0].set_title('planar front speed vs chemostat b (Maxwell b_M = 2)')
    if TAU:
        ax[1].semilogy(bd_b, TAU['truth'][0], 'k-', lw=2, label='truth tau_up'); ax[1].semilogy(bd_b, TAU['truth'][1], 'k--', lw=2, label='truth tau_down')
        for m in ['hyperme', 'hyperme_rev', 'crnn']:
            if (m, 0) in TAU:
                ax[1].semilogy(bd_b, TAU[(m, 0)][0], '-', color=COL[m], label=f'{SHORT[m]} up'); ax[1].semilogy(bd_b, TAU[(m, 0)][1], '--', color=COL[m], label=f'{SHORT[m]} down')
        ax[1].set_xlabel('b (a = 3)'); ax[1].set_ylabel('mean first-passage time'); ax[1].legend(fontsize=7, ncol=2)
        ax[1].set_title('single-site switching, Omega = 100 (seed 0)')
    fig.tight_layout(); fig.savefig(os.path.join(RUNS, 'C2_front_speed.png'), dpi=130); plt.close(fig)


# =============================================================================================== C3 Brusselator
def osc_stats(xm, DT):
    """xm (T+1,) site-mean X; second half: amplitude and period from upward mean crossings."""
    if not np.isfinite(xm).all():
        return np.nan, np.nan
    h = xm[len(xm) // 2:]
    amp = h.max() - h.min()
    dd = h - h.mean()
    up = np.where((dd[:-1] < 0) & (dd[1:] >= 0))[0]
    if len(up) < 2 or amp < 1e-3 * max(abs(h.mean()), 1e-9):
        return amp, np.nan
    tc = up + dd[up] / (dd[up] - dd[up + 1])            # linear interpolation
    return amp, float(np.mean(np.diff(tc)) * DT)


def kdom(field, N):
    F = np.abs(np.fft.fft2(field.reshape(N, N) - field.mean())) ** 2
    F[0, 0] = 0
    i, j = np.unravel_index(np.argmax(F), F.shape)
    kx, ky = min(i, N - i), min(j, N - j)
    return np.hypot(kx, ky)


def c3():
    R, Z = load('brus')
    Re, Ze = load('brus_eq')
    if not R and not Re:
        return
    data = torch.load(os.path.join(RUNS, 'data', 'brus.pt'))
    DT = data['meta']['DT']; N = data['meta']['N']
    L = ['# C3: Brusselator on an 8x8 periodic grid (driven limit cycle + Turing patterns) and equilibrium control C3eq', '',
         'R1 A -> X, R2 2X + Y -> 3X, R3 B + X -> Y + D, R4 X -> E (k = 1), every reverse rate 1e-3 (so A in ker S is finite: local cycle affinities '
         'log(a/eps^2) ~ 13.8 and log(b/eps^2) ~ 14.7). D_X = 0.4, D_Y = 3.6 (ratio 9; Turing threshold b_T ~ 1.78 < Hopf 1 + a^2 ~ 2).',
         'Train 48 traj x 20 steps (DT = 0.2, T = 4), a in [0.9, 1.1], b in [2.3, 3.0] (oscillatory, B > 1 + A^2); half near-homogeneous (point of a '
         'well-mixed burn-in + 0.5% noise), half random smooth fields. Test ID 16 / OOD 16 (b in [3.2, 3.7]) x 200 steps (10T = 40 time units, ~5 periods); '
         'index 0-7 exactly homogeneous (limit-cycle test; truth = exact homogeneous solution), 8-15 random fields (truth -> stationary Turing pattern). '
         'C3eq: same, but every reverse rate constant = forward one (A = 0, detailed balance, equilibrium x = y = 1). 2000 iters, batch 8, CPU float64. Entries `seed0 / seed1`.', '']
    M = {}
    for key in R:
        mm = {}
        for st in ['test_id', 'test_ood']:
            d = data[st]; Y = d['c'].numpy(); Rm = Z[key][f'{st}/R']; h = d['homog'].numpy().astype(bool)
            pe, ae, osc_ok = [], [], []
            for i in np.where(h)[0]:
                at, ptr = osc_stats(Y[i, :, 0].mean(-1), DT)
                am, pm = osc_stats(Rm[i, :, 0].mean(-1), DT)
                ae.append(abs(am - at) / at if np.isfinite(am) else np.inf)
                pe.append(abs(pm - ptr) / ptr if np.isfinite(pm) else np.inf)
                osc_ok.append(bool(np.isfinite(pm)))
            sp = np.where(~h)[0]
            std_ratio = [np.nanstd(Rm[i, -1, 0]) / np.std(Y[i, -1, 0]) for i in sp]
            kd = [kdom(Rm[i, -1, 0], N) == kdom(Y[i, -1, 0], N) if np.isfinite(Rm[i, -1]).all() else False for i in sp]
            fp = [float(np.abs(Rm[i, -1, 0] - d['a'].numpy()[i]).mean()) for i in np.where(h)[0]]
            mm[st] = dict(period_relerr=float(np.median(pe)), amp_relerr=float(np.median(ae)), frac_osc=float(np.mean(osc_ok)),
                          amp_model=float(np.nanmedian([osc_stats(Rm[i, :, 0].mean(-1), DT)[0] for i in np.where(h)[0]])),
                          amp_true=float(np.median([osc_stats(Y[i, :, 0].mean(-1), DT)[0] for i in np.where(h)[0]])),
                          period_true=float(np.median([osc_stats(Y[i, :, 0].mean(-1), DT)[1] for i in np.where(h)[0]])),
                          period_model=float(np.nanmedian([osc_stats(Rm[i, :, 0].mean(-1), DT)[1] for i in np.where(h)[0]])),
                          turing_std_ratio=float(np.nanmedian(std_ratio)), turing_kdom_match=float(np.mean(kd)),
                          sigma_relerr_homog=float(np.median(np.linalg.norm(Z[key][f'{st}/sigma'][h] - d['sigma'].numpy()[h], axis=1) / np.linalg.norm(d['sigma'].numpy()[h], axis=1))) if f'{st}/sigma' in Z[key] else None)
        M[key] = mm
    L += ['## (a)+(c) Accuracy, positivity, total X', '',
          '| model | params | ID T | ID 10T (homog / spatial) | OOD T | OOD 10T (homog / spatial) | sum-X rel err 10T ID / OOD | traj with c<0 ID/OOD | min c | blown | substeps/DT | homog. symmetry breaking (max spatial std at 10T) | train s |',
          '|---|---|---|---|---|---|---|---|---|---|---|---|---|']
    for m in MODELS:
        row = [NAMES[m], both(R, m, lambda r, s: r['params'], '{:d}')]
        for st in ['test_id', 'test_ood']:
            row.append(both(R, m, lambda r, s: r['sets'][st]['relL2_median']['20']))
            row.append(both(R, m, lambda r, s: '{:.3g} ({:.3g} / {:.3g})'.format(r['sets'][st]['relL2_median']['200'], r['sets'][st]['relL2_median_homog']['200'], r['sets'][st]['relL2_median_spatial']['200']), '{}'))
        row.append(both(R, m, lambda r, s: r['sets']['test_id']['totX_relerr_10T_median']) + ' ; ' + both(R, m, lambda r, s: r['sets']['test_ood']['totX_relerr_10T_median']))
        row.append(both(R, m, lambda r, s: f"{r['sets']['test_id']['n_traj_negative']},{r['sets']['test_ood']['n_traj_negative']}", '{}'))
        row.append(both(R, m, lambda r, s: min(r['sets'][q]['min_c'] for q in ['test_id', 'test_ood']), '{:.2e}'))
        row.append(both(R, m, lambda r, s: sum(r['sets'][q]['n_blown'] for q in ['test_id', 'test_ood']), '{:d}'))
        row.append(both(R, m, lambda r, s: r['sets']['test_id']['substeps_per_DT_mean'], '{:.1f}'))
        row.append(both(R, m, lambda r, s: max(r['sets'][q]['homog_spatial_std_X_10T_max'] or 0 for q in ['test_id', 'test_ood']), '{:.1e}'))
        row.append(both(R, m, lambda r, s: r['train_time_s'], '{:.0f}'))
        L.append('| ' + ' | '.join(row) + ' |')
    L += ['', '## (a)+(b) Limit cycle on the homogeneous test trajectories (site-mean X, second half of the 10T rollout)', '',
          f'True median amplitude / period: ID {M[list(M)[0]]["test_id"]["amp_true"]:.2f} / {M[list(M)[0]]["test_id"]["period_true"]:.2f}, '
          f'OOD {M[list(M)[0]]["test_ood"]["amp_true"]:.2f} / {M[list(M)[0]]["test_ood"]["period_true"]:.2f}. "osc" = fraction of the 8 trajectories with a sustained oscillation.', '',
          '| model | osc ID / OOD | amplitude (model) ID / OOD | amp rel err ID / OOD | period (model) ID / OOD | period rel err ID / OOD |', '|---|---|---|---|---|---|']
    for m in MODELS:
        L.append(f'| {NAMES[m]} | ' + ' | '.join(
            both(M, m, lambda r, s: r['test_id'][q], '{:.2f}') + ' ; ' + both(M, m, lambda r, s: r['test_ood'][q], '{:.2f}')
            for q in ['frac_osc', 'amp_model', 'amp_relerr', 'period_model', 'period_relerr']) + ' |')
    L += ['', '## (c) Turing patterns (random-field test trajectories, X at 10T)', '',
          '| model | pattern relL2 10T ID / OOD | spatial-std ratio model/truth ID / OOD | dominant wavenumber matches ID / OOD |', '|---|---|---|---|']
    for m in MODELS:
        L.append(f'| {NAMES[m]} | ' + both(R, m, lambda r, s: r['sets']['test_id']['relL2_median_spatial']['200']) + ' ; ' + both(R, m, lambda r, s: r['sets']['test_ood']['relL2_median_spatial']['200'])
                 + ' | ' + both(M, m, lambda r, s: r['test_id']['turing_std_ratio'], '{:.2f}') + ' ; ' + both(M, m, lambda r, s: r['test_ood']['turing_std_ratio'], '{:.2f}')
                 + ' | ' + both(M, m, lambda r, s: r['test_id']['turing_kdom_match'], '{:.2f}') + ' ; ' + both(M, m, lambda r, s: r['test_ood']['turing_kdom_match'], '{:.2f}') + ' |')
    L += ['', '## (d) Entropy production and affinities', '',
          '| model | sigma rel err all ID / OOD | sigma rel err homog (limit cycle) ID / OOD | learned cycle affinities (R1+R4, R2+R3), ID mean | true | A_cyc rel err ID |', '|---|---|---|---|---|---|']
    for m in MODELS:
        if m == 'gnn':
            L.append(f'| {NAMES[m]} | n/a | n/a | n/a | n/a | n/a |'); continue
        L.append(f'| {NAMES[m]} | ' + both(R, m, lambda r, s: r['sets']['test_id']['sigma_relerr_median']) + ' ; ' + both(R, m, lambda r, s: r['sets']['test_ood']['sigma_relerr_median'])
                 + ' | ' + both(M, m, lambda r, s: r['test_id']['sigma_relerr_homog']) + ' ; ' + both(M, m, lambda r, s: r['test_ood']['sigma_relerr_homog'])
                 + ' | ' + both(R, m, lambda r, s: '({:.1f}, {:.1f})'.format(*np.mean(r['sets']['test_id']['affinity_learned_sitemean'], 0)), '{}')
                 + ' | ({:.1f}, {:.1f})'.format(*next(iter(R.values()))['sets']['test_id']['affinity_true_mean'])
                 + ' | ' + both(R, m, lambda r, s: r['sets']['test_id']['Acycle_relerr_median']) + ' |')
    # ---------------- C3eq control
    if Re:
        de = torch.load(os.path.join(RUNS, 'data', 'brus_eq.pt'))
        # true Jacobian at equilibrium (finite differences of the true rhs)
        sys.argv = [sys.argv[0], '--eq']
        import importlib
        import gen_brusselator as GB
        importlib.reload(GB)
        from crn2 import true_ab_np
        g = build_graph_np('brus_eq', N)
        Sm = g['Nm'] - g['Np']
        a0, b0 = float(de['test_id']['a'][0]), float(de['test_id']['b'][0])
        k, A = GB.true_params(g, a0, b0, eq=True)
        x0 = np.ones(2 * N * N); Jt = np.zeros((len(x0), len(x0))); hh = 1e-6
        for j in range(len(x0)):
            e = np.zeros_like(x0); e[j] = hh
            fp = (lambda x: (lambda ab: (ab[0] - ab[1]) @ Sm)(true_ab_np(x, g, k, A)))
            Jt[:, j] = (fp(x0 + e) - fp(x0 - e)) / (2 * hh)
        evt = np.linalg.eigvals(Jt)
        L += ['', '## (b) Equilibrium control C3eq (A = 0 data: any oscillation is non-physical)', '',
              f'True linearisation at x = y = 1 (traj 0): max Re {evt.real.max():.3e}, max |Im| {np.abs(evt.imag).max():.1e} (real spectrum). '
              'Oscillation index on the 16 ID test rollouts: number of extrema of site-mean X(t) after t = T with prominence > 0.1% of the mean (truth: monotone relaxation); '
              'residual = max over trajectories of the second-half peak-to-peak of any site / mean. Spectrum = eigenvalues of d(rhs)/dc at the model\'s own 10T state '
              '(trajectories 0, 1, 8, 9); max |Im| and the number of complex eigenvalues (|Im| > 1e-6).', '',
              '| model | relL2 ID T / 10T | OOD 10T | traj with spurious extrema ID / OOD | max # extrema | residual osc. amplitude | max Re lambda | max abs Im lambda | # complex eig (of 128, mean) | max abs(Im)/abs(Re) | traj c<0 | sum-X err 10T |',
              '|---|---|---|---|---|---|---|---|---|---|---|---|']
        Me = {}
        for key in Re:
            mm = {}
            for st in ['test_id', 'test_ood']:
                Y = de[st]['c'].numpy(); Rm = Ze[key][f'{st}/R']
                ne = [n_extrema(Rm[i, 20:, 0].mean(-1), 1e-3) for i in range(len(Y))]
                nt = [n_extrema(Y[i, 20:, 0].mean(-1), 1e-3) for i in range(len(Y))]
                res = [np.nanmax(np.ptp(Rm[i, 100:, :], axis=0)) / np.nanmean(Rm[i, 100:]) for i in range(len(Y))]
                mm[st] = dict(spur=int(sum((a_ > b_) or a_ < 0 for a_, b_ in zip(ne, nt))), maxext=int(max(ne)), resid=float(np.nanmax(res)))
            ev = [Ze[key][k_] for k_ in Ze[key] if k_.startswith('eig/')]
            mm['eig_maxre'] = float(max(e_.real.max() for e_ in ev)) if ev else None
            mm['eig_maxim'] = float(max(np.abs(e_.imag).max() for e_ in ev)) if ev else None
            mm['eig_ncomplex'] = float(np.mean([(np.abs(e_.imag) > 1e-6).sum() for e_ in ev])) if ev else None
            mm['eig_maxQ'] = float(max((np.abs(e_.imag) / np.maximum(np.abs(e_.real), 1e-12)).max() for e_ in ev)) if ev else None
            Me[key] = mm
        for m in MODELS:
            L.append(f'| {NAMES[m]} | ' + both(Re, m, lambda r, s: '{:.3g}/{:.3g}'.format(r['sets']['test_id']['relL2_median']['20'], r['sets']['test_id']['relL2_median']['200']), '{}')
                     + ' | ' + both(Re, m, lambda r, s: r['sets']['test_ood']['relL2_median']['200'])
                     + ' | ' + both(Me, m, lambda r, s: f"{r['test_id']['spur']},{r['test_ood']['spur']}", '{}')
                     + ' | ' + both(Me, m, lambda r, s: max(r['test_id']['maxext'], r['test_ood']['maxext']), '{:d}')
                     + ' | ' + both(Me, m, lambda r, s: max(r['test_id']['resid'], r['test_ood']['resid']), '{:.1e}')
                     + ' | ' + both(Me, m, lambda r, s: r['eig_maxre'], '{:.2e}') + ' | ' + both(Me, m, lambda r, s: r['eig_maxim'], '{:.2e}')
                     + ' | ' + both(Me, m, lambda r, s: r['eig_ncomplex'], '{:.1f}')
                     + ' | ' + both(Me, m, lambda r, s: r['eig_maxQ'], '{:.2g}')
                     + ' | ' + both(Re, m, lambda r, s: f"{r['sets']['test_id']['n_traj_negative']},{r['sets']['test_ood']['n_traj_negative']}", '{}')
                     + ' | ' + both(Re, m, lambda r, s: r['sets']['test_id']['totX_relerr_10T_median']) + ' |')
        OUT['C3eq'] = {str(k): v for k, v in Me.items()}; OUT['C3eq_true_eig'] = dict(maxre=float(evt.real.max()), maxim=float(np.abs(evt.imag).max()))
    L += ['', '## HyperME assertions (positivity, sigma >= 0 on every rollout)', '']
    for bench, RR in [('brus', R), ('brus_eq', Re)]:
        for m in ['hyperme', 'hyperme_rev']:
            for s in SEEDS:
                if (m, s) in RR:
                    r = RR[(m, s)]
                    L.append(f'- {bench} {m}_s{s}: ' + ', '.join(f"{q}: pos {r['sets'][q]['assert_positivity']}, sigma>=0 {r['sets'][q]['assert_sigma_nonneg']}" for q in r['sets']))
    nf = os.path.join(RUNS, 'C3_notes.md')
    if os.path.exists(nf):
        L += ['', open(nf).read()]
    open(os.path.join(RUNS, 'C3_summary.md'), 'w').write('\n'.join(L) + '\n')
    OUT['C3'] = {str(k): v for k, v in M.items()}
    # ---------------- figures
    mods = [m for m in ['hyperme', 'hyperme_rev', 'crnn', 'gnn'] if (m, 0) in Z]
    fig, ax = plt.subplots(2, len(mods), figsize=(3.3 * len(mods), 6.4), squeeze=False)
    for ri, st in enumerate(['test_id', 'test_ood']):
        d = data[st]; i = 0
        Y = d['c'].numpy()[i]
        for ci, m in enumerate(mods):
            a_ = ax[ri, ci]
            Rm = Z[(m, 0)][f'{st}/R'][i]
            a_.plot(Y[:, 0].mean(-1), Y[:, 1].mean(-1), 'k-', lw=2.2, alpha=0.6, label='truth')
            a_.plot(Rm[:, 0].mean(-1), Rm[:, 1].mean(-1), '-', color=COL[m], lw=1.2, label=SHORT[m])
            a_.plot(Y[0, 0].mean(), Y[0, 1].mean(), 'ko', ms=4)
            a_.set_title(f'{SHORT[m]} | {st[5:].upper()} a={d["a"][i]:.2f} b={d["b"][i]:.2f}', fontsize=9)
            a_.set_xlabel('mean X'); a_.set_ylabel('mean Y') if ci == 0 else None
            a_.legend(fontsize=7)
    fig.suptitle('C3 Brusselator: X-Y phase plane, homogeneous start, 10T = 40 time units (seed 0)', fontsize=10)
    fig.tight_layout(); fig.savefig(os.path.join(RUNS, 'C3_phase.png'), dpi=130); plt.close(fig)
    fig, ax = plt.subplots(2, len(mods) + 1, figsize=(2.6 * (len(mods) + 1), 5.4), squeeze=False)
    for ri, st in enumerate(['test_id', 'test_ood']):
        d = data[st]; i = 8
        Y = d['c'].numpy()[i, -1, 0]
        vmax = Y.max() * 1.05
        im = ax[ri, 0].imshow(Y.reshape(N, N), vmin=0, vmax=vmax); ax[ri, 0].set_title(f'truth {st[5:].upper()} b={d["b"][i]:.2f}', fontsize=9)
        for ci, m in enumerate(mods):
            Rm = Z[(m, 0)][f'{st}/R'][i, -1, 0]
            ax[ri, ci + 1].imshow(Rm.reshape(N, N), vmin=0, vmax=vmax)
            ax[ri, ci + 1].set_title(f'{SHORT[m]} (min {np.nanmin(Rm):.2f})', fontsize=9)
        for a_ in ax[ri]:
            a_.set_xticks([]); a_.set_yticks([])
        fig.colorbar(im, ax=list(ax[ri]), shrink=0.8, label='X at 10T')
    fig.suptitle('C3 Brusselator: Turing pattern at 10T from a random start (seed 0)', fontsize=10)
    fig.savefig(os.path.join(RUNS, 'C3_turing.png'), dpi=130, bbox_inches='tight'); plt.close(fig)
    fig, ax = plt.subplots(1, 2, figsize=(12, 4))
    for ai, (st, i) in enumerate([('test_id', 0), ('test_id', 8)]):
        d = data[st]; t = np.arange(d['c'].shape[1]) * DT
        ax[ai].semilogy(t, d['sigma'].numpy()[i], 'k-', lw=2.2, alpha=0.7, label='truth')
        for m in mods:
            if f'{st}/sigma' in Z[(m, 0)]:
                ax[ai].semilogy(t, Z[(m, 0)][f'{st}/sigma'][i], '-', color=COL[m], lw=1.1, label=SHORT[m])
        ax[ai].set_xlabel('t'); ax[ai].set_ylabel('entropy production rate sigma(t)')
        ax[ai].set_title(('homogeneous start (limit cycle)' if i < 8 else 'random start (Turing)') + f', ID a={d["a"][i]:.2f} b={d["b"][i]:.2f}', fontsize=9)
        ax[ai].legend(fontsize=8)
    fig.tight_layout(); fig.savefig(os.path.join(RUNS, 'C3_sigma.png'), dpi=130); plt.close(fig)
    if Re:
        mods_e = [m for m in MODELS if (m, 0) in Ze]
        fig, ax = plt.subplots(1, 3, figsize=(15, 4))
        for ai, i in enumerate([0, 8]):
            Y = de['test_id']['c'].numpy()[i]; t = np.arange(Y.shape[0]) * DT
            ax[ai].plot(t, Y[:, 0].mean(-1), 'k-', lw=2.2, alpha=0.6, label='truth')
            for m in mods_e:
                ax[ai].plot(t, Ze[(m, 0)]['test_id/R'][i, :, 0].mean(-1), '-', color=COL[m], lw=1.1, label=SHORT[m])
            ax[ai].set_xlabel('t'); ax[ai].set_ylabel('mean X'); ax[ai].legend(fontsize=7)
            ax[ai].set_title(f'C3eq ({"homogeneous" if i < 8 else "random"} start)', fontsize=9)
        ax[2].plot(evt.real, evt.imag, 'k+', ms=8, label='truth (eq.)')
        for m in mods_e:
            ev = Ze[(m, 0)].get('eig/test_id_0')
            if ev is not None:
                ax[2].plot(ev.real, ev.imag, '.', color=COL[m], ms=4, label=SHORT[m])
        ax[2].set_xlabel('Re lambda'); ax[2].set_ylabel('Im lambda'); ax[2].set_title('linearisation spectrum at the 10T state (traj 0)', fontsize=9)
        ax[2].legend(fontsize=7)
        fig.tight_layout(); fig.savefig(os.path.join(RUNS, 'C3eq_control.png'), dpi=130); plt.close(fig)


if __name__ == '__main__':
    c2()
    c3()
    json.dump(OUT, open(os.path.join(RUNS, 'C23_metrics.json'), 'w'), default=lambda o: o.tolist() if hasattr(o, 'tolist') else str(o))
    print('analysis done')
