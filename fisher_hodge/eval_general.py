"""General advantages of the thermodynamic parameterisation (2026-10-25).

G1  temperature extrapolation + counterfactual on S2 (no retraining; models from runs/tcme/S2_{m}_s{seed}.pt).
    Test = the 16 S2 test_id trajectories (same physical U, psi, initial conditions; U_phys = 0.02*field, psi_phys = 0.02*field)
    re-simulated with diffusion D in {0.01, 0.02, 0.04}.  Model inputs are the training-normalised fields (U_phys / D_train, ...),
    i.e. exactly what the models saw in training; only the "temperature" D changes.
    Temperature edits (r = D_test / D_train):
      TC-ME   k -> r k, phi = alpha U + g -> phi / r (Gibbs factor e^{-U/D});
              (a) "A fixed"  : A unchanged (as specified in the task);
              (b) "A / r"    : A -> A / r as well.  In TC-ME the drift velocity is ~ k h A, so with k -> r k the drift is kept
                               temperature independent only if A -> A / r (all non-entropic forces in units of k_B T scale as 1/T,
                               mobility k ~ D ~ T by Einstein).
      Fisher-Hodge  kappa -> r kappa, (mu - log rho) -> (mu - log rho) / r, cycle term c unchanged (drift = S Theta c, not multiplied by kappa).
      Flux-GNN  (i) as is, (ii) input U -> U * D_train / D_test.
    Counterfactual "drive off": reference = S2_amp0 (psi = 0, same U and initial conditions, D = 0.02, made by amp_test.py).
      TC-ME A <- 0 (exact detailed balance), Fisher-Hodge c <- 0, and for every model "inputs dpsi = psin = 0".
G2  report only (training/eval in train_s6.py): reads runs/general/G2_*.json.
  python3 eval_general.py --gen_g1 --g1        # data + evaluation -> runs/general/G1.json
  python3 eval_general.py --report             # summary.md + figures (G1 + G2)
"""
import argparse, json, os, time, glob
import numpy as np
import torch
from common import *
import gen_data as G
import models2                                   # noqa: F401 (patches rho_scale)
from models import FisherHodge, node_deriv
from models2 import FluxGNN
from models_tcme import TCME
from train_tcme import fields, relerr_b, jclean

dev = os.environ.get('FH_DEV', 'cuda')
import train_tcme; train_tcme.dev = dev          # fields() moves tensors to train_tcme.dev
D_TRAIN = 0.02
DS = [0.01, 0.02, 0.04]
ODIR = os.path.join(RUNS, 'general')
TDIR = os.path.join(RUNS, 'tcme')


class TCMEt(TCME):
    tscale = 1.0

    def phi(s, rho, F):
        return s.tscale * super().phi(rho, F)


class FHt(FisherHodge):
    tscale = 1.0

    def mu(s, rho, F):
        return torch.log(rho) + s.tscale * (s.alpha * F['U'] + node_deriv(lambda x: s.g_density(x, F), rho))


CLS = dict(tcme=TCMEt, fisher_hodge=FHt, flux_gnn=FluxGNN)


def load(m, seed):
    model = CLS[m]('S2').to(dev)
    model.load_state_dict(torch.load(os.path.join(TDIR, f'S2_{m}_s{seed}.pt'), map_location=dev))
    return model.eval()


def dname(Dv):
    return os.path.join(RUNS, 'data', f'S2_D{Dv:g}.pt')


# ------------------------------------------------------------------ data
def fine_flux_D(rho, U, uvel, Dv, nf):
    h = 1.0 / nf
    return -(Dv / h) * grad_e(rho) - mean_e(rho) * grad_e(U) / h + mean_e(rho) * uvel


def gen_g1():
    meta = json.load(open(os.path.join(RUNS, 'data', 'S2_meta.json')))
    base = torch.load(os.path.join(RUNS, 'data', 'S2.pt'))['test_id']
    prs = meta['sets']['test_id']['params']
    info = {}
    for Dv in DS:
        t0 = time.time()
        for nf in (64, 128):
            (fXY, fC), _ = G.grids(nf), G.grids(N)
            Uf = torch.tensor(np.stack([D_TRAIN * G.eval_field(p['U'], fXY) for p in prs]), dtype=torch.float64, device=dev)
            Pf = torch.tensor(np.stack([D_TRAIN * G.eval_field(p['psi'], fC) for p in prs]), dtype=torch.float64, device=dev)
            pe = float(torch.maximum(dual_dpsi(Pf).abs().amax(), grad_e(Uf).abs().amax()) / Dv)     # fine-cell Peclet
            if pe <= 2.0:
                break
        R0 = torch.tensor(np.stack([G.eval_ic(p['ic'], fXY) for p in prs]), dtype=torch.float64, device=dev)
        uvel = dual_dpsi(Pf) * nf
        down = coarsen if nf == 64 else (lambda x: coarsen(coarsen(x)))
        rhs = lambda r: div_T(fine_flux_D(r, Uf, uvel, Dv, nf)) * nf
        dt = G.DT_REF; per = int(round(DT / dt))
        rho = R0.clone(); out = [down(rho)]
        for k in range(T_TEST):
            for _ in range(per):
                k1 = rhs(rho); k2 = rhs(rho + dt * k1)
                rho = rho + 0.5 * dt * (k1 + k2)
            out.append(down(rho))
        traj = torch.stack(out, 1)
        Jf = fine_flux_D(rho, Uf, uvel, Dv, nf)
        Jc = coarsen_flux(Jf) if nf == 64 else coarsen_flux(coarsen_flux(Jf))
        d = dict(rho=traj.float().cpu(), U=base['U'].clone(), dpsi=base['dpsi'].clone(), psin=base['psin'].clone(), J_end=Jc.float().cpu())
        drift = float((torch.linalg.vector_norm((traj[:, -1] - traj[:, -11]).flatten(1), dim=1) / torch.linalg.vector_norm(traj[:, -1].flatten(1), dim=1)).max())
        info[f'{Dv:g}'] = dict(nf=nf, fine_peclet_max=pe, min_rho=float(traj.min()), mass_dev=float((traj.mean((-1, -2)) - 1).abs().max()),
                               rel_change_last_0p5=drift, time_s=time.time() - t0,
                               relerr_vs_S2_test_id=float(relerr_b(traj[:, -1].cpu().numpy(), base['rho'][:, -1].numpy()).max()) if Dv == D_TRAIN else None)
        torch.save(dict(test=d), dname(Dv))
        print('gen D', Dv, json.dumps(info[f'{Dv:g}']), flush=True)
    os.makedirs(ODIR, exist_ok=True)
    json.dump(info, open(os.path.join(ODIR, 'G1_gen_info.json'), 'w'), indent=1)


# ------------------------------------------------------------------ evaluation helpers
@torch.no_grad()
def rollout(model, ctx, rho0, nsteps=T_TEST):
    r = rho0.clone(); out = [r.cpu()]; nsub = []
    for k in range(nsteps):
        r, ns = model.advance(r, ctx, DT); nsub.append(ns)
        bad = (~torch.isfinite(r)).any(-1).any(-1) | (r.abs().amax((-1, -2)) > 1e4)
        if bad.any():
            r = torch.where(bad[:, None, None], torch.full_like(r, float('nan')), r)
        out.append(r.cpu())
    return torch.stack(out, 1).double(), float(np.mean(nsub))


def metrics(R, Y):
    B, T1 = Y.shape[:2]
    err = (torch.linalg.vector_norm((R - Y).reshape(B, T1, -1), dim=-1) / torch.linalg.vector_norm(Y.reshape(B, T1, -1), dim=-1)).numpy()
    err = np.where(np.isfinite(err), err, np.inf)
    Rn = R.numpy()
    return dict(err10T=float(np.median(err[:, -1])), errT=float(np.median(err[:, 20])), err10T_per_traj=err[:, -1].tolist(),
                n_blown=int((~np.isfinite(Rn[:, -1])).any((1, 2)).sum()),
                mass_err=float(np.nanmax(np.abs(Rn.mean((-1, -2)) - 1))), min_rho=float(np.nanmin(Rn)),
                n_neg=int((Rn < 0).any((1, 2, 3)).sum()), neg_frac=float(np.nanmean(Rn < 0)))


def flux_phys(model, rho, ctx):
    n = rho.shape[-1]
    if model.name == 'tcme':
        return model.flux(rho, ctx) / n
    return -(D * n) * model.flux_q(rho, ctx)


def g1(mlist=('tcme', 'fisher_hodge', 'flux_gnn')):
    os.makedirs(ODIR, exist_ok=True)
    res = {}
    sets = {Dv: torch.load(dname(Dv))['test'] for Dv in DS}
    cf = torch.load(os.path.join(RUNS, 'data', 'S2_amp0.pt'))['test']
    J1 = torch.load(os.path.join(RUNS, 'data', 'S2_amp1.pt'))['test']['J_end'].numpy()
    B = J1.shape[0]
    nJ1 = np.linalg.norm(J1.reshape(B, -1), axis=1)
    res['ref_amp0_Jend_over_J1'] = float(np.median(np.linalg.norm(cf['J_end'].numpy().reshape(B, -1), axis=1) / nJ1))
    for m in mlist:
        for seed in [0, 1]:
            key = f'{m}_s{seed}'; res[key] = {}
            model = load(m, seed)
            variants = dict(tcme=['noedit', 'edit_Afixed', 'edit_Ascaled'], fisher_hodge=['noedit', 'edit'],
                            flux_gnn=['asis', 'Uscaled'])[m]
            # ---------------- temperature
            for Dv in DS:
                sd = sets[Dv]; r = Dv / D_TRAIN; Y = sd['rho'].double()
                for v in variants:
                    t0 = time.time()
                    F = fields(sd)
                    model.tscale = 1.0
                    if m == 'flux_gnn' and v == 'Uscaled':
                        F['U'] = F['U'] / r
                    with torch.no_grad():
                        ctx = model.prepare(F)
                        if m == 'tcme' and v != 'noedit':
                            model.tscale = 1.0 / r; ctx['k'] = ctx['k'] * r
                            if v == 'edit_Ascaled':
                                ctx['A'] = ctx['A'] / r
                        if m == 'fisher_hodge' and v == 'edit':
                            model.tscale = 1.0 / r; ctx['kap'] = ctx['kap'] * r
                        R, ns = rollout(model, ctx, sd['rho'][:, 0].to(dev))
                    S = metrics(R, Y); S['substeps'] = ns
                    res[key][f'D{Dv:g}_{v}'] = S
                    print(f'{key:16s} D={Dv:<5g} {v:13s} 10T {S["err10T"]:.3e} T {S["errT"]:.3e} mass {S["mass_err"]:.1e} '
                          f'neg {S["n_neg"]} blown {S["n_blown"]} sub {ns:.1f} {time.time() - t0:.0f}s', flush=True)
            model.tscale = 1.0
            # ---------------- counterfactual drive off
            Y = cf['rho'].double()
            cfv = dict(tcme=['A0', 'inputs0'], fisher_hodge=['c0', 'inputs0'], flux_gnn=['inputs0'])[m]
            for v in cfv:
                F = fields(cf) if v == 'inputs0' else fields(sets[D_TRAIN])     # A0/c0: original drive inputs, drive switched off inside the model
                with torch.no_grad():
                    ctx = model.prepare(F)
                    if v == 'A0':
                        ctx['A'] = torch.zeros_like(ctx['A'])
                    if v == 'c0':
                        ctx['c'] = torch.zeros_like(ctx['c'])
                    R, ns = rollout(model, ctx, cf['rho'][:, 0].to(dev))
                    S = metrics(R, Y)
                    fin = torch.isfinite(R[:, -1]).all(-1).all(-1)
                    rend = torch.where(fin[:, None, None], R[:, -1], torch.ones_like(R[:, -1])).float().to(dev)
                    Jm = flux_phys(model, rend, ctx).cpu().numpy()
                    Jr = flux_phys(model, cf['rho'][:, -1].to(dev), ctx).cpu().numpy()
                S['Jss_over_J1_at_model_rho'] = float(np.median(np.linalg.norm(Jm.reshape(B, -1), axis=1) / nJ1))
                S['Jss_over_J1_at_ref_rho'] = float(np.median(np.linalg.norm(Jr.reshape(B, -1), axis=1) / nJ1))
                S['curlJ_over_curlJ1_at_model_rho'] = float(np.median(
                    np.linalg.norm(curl_f(torch.tensor(Jm)).numpy().reshape(B, -1), axis=1) / np.linalg.norm(curl_f(torch.tensor(J1)).numpy().reshape(B, -1), axis=1)))
                res[key][f'cf_{v}'] = S
                print(f'{key:16s} CF {v:8s} 10T {S["err10T"]:.3e} |J|/|J1| model-rho {S["Jss_over_J1_at_model_rho"]:.2e} ref-rho {S["Jss_over_J1_at_ref_rho"]:.2e} '
                      f'curl {S["curlJ_over_curlJ1_at_model_rho"]:.2e} mass {S["mass_err"]:.1e} neg {S["n_neg"]}', flush=True)
        json.dump(jclean({k: v for k, v in res.items() if k == 'ref_amp0_Jend_over_J1' or k.startswith(m + '_s')}), open(os.path.join(ODIR, f'G1_{m}.json'), 'w'))
    print('ref amp0 |J_end|/|J1|', res['ref_amp0_Jend_over_J1'])


def from_log(path, mlist=('tcme', 'fisher_hodge')):
    """rebuild G1_{m}.json from a G1 log (the GPU run of tcme / fisher_hodge was stopped before the json was written)."""
    import re
    res = {}
    for line in open(path):
        t = line.split()
        if not t or not any(t[0] == f'{m}_s{s}' for m in mlist for s in [0, 1]):
            continue
        key = t[0]; res.setdefault(key, {})
        num = lambda name: float(re.search(name + r' ([-+0-9.e]+)', line).group(1))
        if t[1] == 'CF':
            res[key][f'cf_{t[2]}'] = dict(err10T=num('10T'), errT=None, mass_err=num('mass'), n_neg=int(num('neg')),
                                          Jss_over_J1_at_model_rho=num('model-rho'), Jss_over_J1_at_ref_rho=num('ref-rho'),
                                          curlJ_over_curlJ1_at_model_rho=num('curl'))
        else:
            Dv = t[1].split('=')[1] or t[2]
            v = t[2] if t[1] != 'D=' else t[3]
            res[key][f'D{float(Dv):g}_{v}'] = dict(err10T=num('10T'), errT=num(' T'), mass_err=num('mass'), n_neg=int(num('neg')),
                                                   n_blown=int(num('blown')), substeps=num('sub'))
    for m in mlist:
        sub = {k: v for k, v in res.items() if k.startswith(m + '_s')}
        sub['ref_amp0_Jend_over_J1'] = None
        json.dump(sub, open(os.path.join(ODIR, f'G1_{m}.json'), 'w'))
        print(m, {k: len(v) for k, v in sub.items() if isinstance(v, dict)})


# ------------------------------------------------------------------ report
def ms(v, fmt='.3g'):
    v = np.array([x for x in v if x is not None], float)
    if len(v) == 0:
        return 'n/a'
    if not np.isfinite(v).all():
        return 'inf'
    return f'{v.mean():{fmt}}' + (f' ± {v.std():.2g}' if len(v) > 1 else '')


def report():
    import matplotlib; matplotlib.use('Agg'); import matplotlib.pyplot as plt
    L = ['# General advantages of the thermodynamic parameterisation (G1, G2)', '',
         'All errors: relative L2 at t = 10 (10T = 200 steps), median over 16 test trajectories, then mean ± std over 2 seeds.', '']
    # ---------------- G1
    g1fs = [f for f in glob.glob(os.path.join(ODIR, 'G1_*.json')) if not f.endswith('gen_info.json')]
    if g1fs:
        res = {}
        for f in g1fs:
            res.update({k: v for k, v in json.load(open(f)).items() if v is not None})
        info = json.load(open(os.path.join(ODIR, 'G1_gen_info.json')))
        L += ['## G1: temperature extrapolation (S2, no retraining)', '',
              'Training D = 0.02. Test: same 16 test_id U, psi, initial conditions; D changed. r = D_test / D_train. '
              'TC-ME edit: k -> r k, phi -> phi / r; "A fixed" keeps A (task spec), "A/r" also scales A -> A / r '
              '(needed to keep the drift k h A temperature-independent). Fisher-Hodge edit: kappa -> r kappa, potential part of mu -> / r, c fixed. '
              'Flux-GNN: (i) as is, (ii) input U -> U / r.', '',
              '| D | fine grid | max fine Peclet | min rho | rel. change over last 0.5 |', '|---|---|---|---|---|']
        for Dv in DS:
            i = info[f'{Dv:g}']
            L.append(f'| {Dv:g} | {i["nf"]} | {i["fine_peclet_max"]:.2f} | {i["min_rho"]:.1e} | {i["rel_change_last_0p5"]:.1e} |')
        rows = [('tcme', 'noedit', 'TC-ME, no edit'), ('tcme', 'edit_Afixed', 'TC-ME, T-edit (A fixed)'), ('tcme', 'edit_Ascaled', 'TC-ME, T-edit (A/r)'),
                ('fisher_hodge', 'noedit', 'Fisher-Hodge, no edit'), ('fisher_hodge', 'edit', 'Fisher-Hodge, T-edit'),
                ('flux_gnn', 'asis', 'Flux-GNN (i) as is'), ('flux_gnn', 'Uscaled', 'Flux-GNN (ii) U * D_train/D_test')]
        ag = lambda m, k, f: [res[f'{m}_s{s}'][k][f] for s in [0, 1] if k in res.get(f'{m}_s{s}', {})]
        for title, f in [('10T relative L2', 'err10T'), ('T relative L2', 'errT')]:
            L += ['', f'### {title}', '', '| model | ' + ' | '.join(f'D={Dv:g}' for Dv in DS) + ' |', '|---|' + '---|' * len(DS)]
            for m, v, lab in rows:
                L.append(f'| {lab} | ' + ' | '.join(ms(ag(m, f'D{Dv:g}_{v}', f)) for Dv in DS) + ' |')
        L += ['', '### max mass error / trajectories with rho<0 (of 32) / blown-up (of 32)', '',
              '| model | ' + ' | '.join(f'D={Dv:g}' for Dv in DS) + ' |', '|---|' + '---|' * len(DS)]
        for m, v, lab in rows:
            L.append(f'| {lab} | ' + ' | '.join(f'{max(ag(m, f"D{Dv:g}_{v}", "mass_err")):.1e} / {sum(ag(m, f"D{Dv:g}_{v}", "n_neg"))} / {sum(ag(m, f"D{Dv:g}_{v}", "n_blown"))}' for Dv in DS) + ' |')
        L += ['', '### Counterfactual: drive switched off (reference: psi = 0, same U / initial conditions, D = 0.02)', '',
              f'Steady flux is reported as ||J_model(rho(10T))|| / ||J_true(psi on)||; the reference itself has {res.get("ref_amp0_Jend_over_J1", float("nan")):.1e} (residual transient at t = 10).', '',
              '| model / intervention | 10T rel. L2 | T rel. L2 | ||J|| / ||J_true(psi on)|| at model rho | at reference rho | ||curl J|| / ||curl J_true|| | mass err | traj rho<0 |',
              '|---|---|---|---|---|---|---|---|']
        for m, v, lab in [('tcme', 'A0', 'TC-ME, A <- 0'), ('tcme', 'inputs0', 'TC-ME, inputs dpsi = psin = 0'),
                          ('fisher_hodge', 'c0', 'Fisher-Hodge, c <- 0'), ('fisher_hodge', 'inputs0', 'Fisher-Hodge, inputs = 0'),
                          ('flux_gnn', 'inputs0', 'Flux-GNN, inputs dpsi = psin = 0')]:
            k = f'cf_{v}'
            L.append(f'| {lab} | {ms(ag(m, k, "err10T"))} | {ms(ag(m, k, "errT"))} | {ms(ag(m, k, "Jss_over_J1_at_model_rho"))} | '
                     f'{ms(ag(m, k, "Jss_over_J1_at_ref_rho"))} | {ms(ag(m, k, "curlJ_over_curlJ1_at_model_rho"))} | '
                     f'{max(ag(m, k, "mass_err")):.1e} | {sum(ag(m, k, "n_neg"))} |')
        L += ['', '![G1](G1_temperature.png)', '']
        cols = dict(tcme='#1b6ca8', fisher_hodge='#d1495b', flux_gnn='#3b8b5a')
        sty = dict(noedit=':', edit_Afixed='--', edit_Ascaled='-', edit='-', asis=':', Uscaled='-')
        fig, ax = plt.subplots(figsize=(6.4, 4.3))
        for m, v, lab in rows:
            y = np.array([np.mean(ag(m, f'D{Dv:g}_{v}', 'err10T')) for Dv in DS]); e = np.array([np.std(ag(m, f'D{Dv:g}_{v}', 'err10T')) for Dv in DS])
            ax.errorbar(DS, y, yerr=e, fmt='o' + sty[v], color=cols[m], lw=2 if sty[v] == '-' else 1.3, ms=4, capsize=2, label=lab)
        ax.set_xscale('log', base=2); ax.set_yscale('log'); ax.set_xticks(DS); ax.set_xticklabels([f'{d:g}' for d in DS])
        ax.axvline(D_TRAIN, color='k', lw=0.8, ls=':'); ax.set_xlabel('diffusion D ("temperature"), training D = 0.02')
        ax.set_ylabel('10T relative L2 (median over traj.)'); ax.grid(alpha=0.3, which='both'); ax.legend(fontsize=7)
        ax.set_title('G1: temperature extrapolation without retraining (S2)')
        fig.tight_layout(); fig.savefig(os.path.join(ODIR, 'G1_temperature.png'), dpi=150); plt.close(fig)
    # ---------------- G2
    fs = sorted(glob.glob(os.path.join(ODIR, 'G2_*_s[0-9].json')))
    if fs:
        res2 = {os.path.basename(f)[3:-5]: json.load(open(f)) for f in fs}
        ginfo = json.load(open(os.path.join(ODIR, 'G2_gen_info.json'))) if os.path.exists(os.path.join(ODIR, 'G2_gen_info.json')) else {}
        CH = sorted({float(c) for r in res2.values() for c in r['chi']})
        L += ['## G2: nonlocal Keller-Segel interaction (S6), stability under stronger coupling', '',
              'Train: S6 (U_eff = U - chi W*rho, chi/D = 0.3), 64 traj x 20 steps, 2500 it, train.py protocol. '
              'Both models get the same nonlocal input c = (chi/D) W*rho (coarse-grid Gaussian kernel, sigma = 0.1), computed from the current state, '
              'so chi is known to both and the test is an extrapolation in the interaction field. '
              'TC-ME-S6: phi = alpha U + g(rho/RS, U) + beta c + (chi/D) W*[p(c, U)] = dE/drho with E = sum alpha U rho + int g + (beta/2) rho c + int p dc (exact gradient form, true S6: beta = -1, g = p = 0); learned free energy F = sum rho(log rho - 1) + E. '
              'Flux-GNN-S6: Flux-GNN with c in the node encoder (state-dependent black-box non-negative rates, d^T assembly). '
              'Test: the S6 test_id initial conditions / potentials, chi/D in ' + ', '.join(f'{c:g}' for c in CH) + '.', '']
        if ginfo:
            L += ['Reference data (64x64 fine grid = same solver as the training data; 128x128 = resolution check):', '',
                  '| chi/D | min rho (64) | max rho (64) | max rho (128) | 64 vs 128 rel. L2 at 10T, median / max |', '|---|---|---|---|---|']
            for c in CH:
                i = ginfo.get(f'{c:g}', {}); j = ginfo.get(f'nf128_{c:g}', {})
                L.append(f'| {c:g} | {i.get("min_rho", float("nan")):.2e} | {i.get("max_rho", float("nan")):.1f} | {j.get("max_rho", float("nan")):.1f} | '
                         f'{j.get("relerr_64_vs_128_10T_median", float("nan")):.2e} / {j.get("relerr_64_vs_128_10T_max", float("nan")):.2e} |')
            L.append('')
        ag2 = lambda m, c, f: [res2[k]['chi'][f'{c:g}'][f] for k in res2 if k.rsplit('_s', 1)[0] == m and f'{c:g}' in res2[k]['chi'] and res2[k]['chi'][f'{c:g}'].get(f) is not None]
        M2 = [('tcme_s6', 'TC-ME-S6'), ('flux_gnn_s6', 'Flux-GNN-S6')]
        for title, f in [('10T relative L2', 'err10T'), ('T relative L2', 'errT'), ('peak error at 10T |max rho - max rho_ref| / max rho_ref', 'peak_err10T'),
                         ('10T relative L2 vs the 128x128-grid reference', 'err10T_r128'), ('peak error at 10T vs the 128x128-grid reference', 'peak_err10T_r128'),
                         ('fraction of steps where the true free energy increases (reference: see last row)', 'Ftrue_up_frac'),
                         ('fraction of steps where the learned free energy increases (TC-ME only)', 'Flearn_up_frac')]:
            L += ['', f'### {title}', '', '| model | ' + ' | '.join(f'chi/D={c:g}' for c in CH) + ' |', '|---|' + '---|' * len(CH)]
            for m, lab in M2:
                L.append(f'| {lab} | ' + ' | '.join(ms(ag2(m, c, f)) for c in CH) + ' |')
            if f == 'Ftrue_up_frac':
                L.append('| reference | ' + ' | '.join(ms(ag2('tcme_s6', c, 'Ftrue_up_frac_ref')[:1]) for c in CH) + ' |')
        L += ['', '### max mass error / traj with rho<0 / blown-up (|rho|>1e4 or NaN), of 32', '', '| model | ' + ' | '.join(f'chi/D={c:g}' for c in CH) + ' |', '|---|' + '---|' * len(CH)]
        for m, lab in M2:
            L.append(f'| {lab} | ' + ' | '.join(f'{max(ag2(m, c, "mass_err")):.1e} / {sum(ag2(m, c, "n_neg"))} / {sum(ag2(m, c, "n_blown"))}' for c in CH) + ' |')
        L += ['', '### training', '', '| run | params | final train loss | train time (s) | learned beta (true -1) | alpha |', '|---|---|---|---|---|---|']
        for k, r in res2.items():
            L.append(f'| {k} | {r["params"]} | {r["final_train_loss"]:.3e} | {r["train_time_s"]:.0f} | {r.get("beta", "-")} | {r.get("alpha", "-")} |')
        L += ['', '![G2](G2_stability.png)', '']
        fig, axs = plt.subplots(1, 2, figsize=(10.5, 4))
        cols = dict(tcme_s6='#1b6ca8', flux_gnn_s6='#3b8b5a')
        for m, lab in M2:
            for ax, f in zip(axs, ['err10T', 'peak_err10T']):
                y = [np.mean(ag2(m, c, f)) for c in CH]; e = [np.std(ag2(m, c, f)) for c in CH]
                ax.errorbar(CH, y, yerr=e, fmt='o-', color=cols[m], lw=2, capsize=2, label=lab)
        for ax, t in zip(axs, ['10T relative L2', 'peak error at 10T']):
            ax.set_yscale('log'); ax.set_xlabel('chi / D  (training: 0.3)'); ax.set_title(t); ax.axvline(0.3, color='k', lw=0.8, ls=':')
            ax.grid(alpha=0.3, which='both')
        axs[0].legend(fontsize=8); fig.suptitle('G2: nonlocal aggregation (S6), extrapolation in interaction strength')
        fig.tight_layout(); fig.savefig(os.path.join(ODIR, 'G2_stability.png'), dpi=150); plt.close(fig)
    open(os.path.join(ODIR, 'summary.md'), 'w').write('\n'.join(L))
    print('\n'.join(L))


def report_g2b(M=None, head=None, intro=None, png='G2b_stability.png', key='\n## G2b:', suptitle='G2b: constrained free energies vs. unconstrained TC-ME and Flux-GNN (S6)', extra=None,
               xlabel='chi / D  (training: 0.3)', write_md=True):
    """G2b: constrained free energies (appended to summary.md as its own section, replacing an older G2b section)."""
    import matplotlib; matplotlib.use('Agg'); import matplotlib.pyplot as plt
    fs = sorted(glob.glob(os.path.join(ODIR, 'G2_*_s[0-9].json')))
    res2 = {os.path.basename(f)[3:-5]: json.load(open(f)) for f in fs}
    CH = [0.3, 0.5, 0.8]
    NN = N * N
    M = M or [('tcme_s6', 'TC-ME-S6 (G2, unconstrained)'), ('flux_gnn_s6', 'Flux-GNN-S6 (G2)'), ('tcme_s6a', '(a) bounded p'),
         ('tcme_s6b', '(b) convex local (ICNN)'), ('tcme_s6ab', '(a)+(b)')]
    runs = lambda m: [k for k in res2 if k.rsplit('_s', 1)[0] == m]
    ag = lambda m, c, f: [res2[k]['chi'][f'{c:g}'].get(f) for k in runs(m) if res2[k]['chi'][f'{c:g}'].get(f) is not None]
    L = ['', head or '## G2b: constrained free energies (does bounding / convexifying F fix the collapse?)', ''] + ([
         'Same data, protocol and evaluation as G2; 1 seed (s0) per constrained variant, 2500 it, activation checkpointing + substep cap 64 '
         '(skipped iterations listed below). All variants keep the linear nonlocal channel beta c (true beta = -1). '
         '(a) p(c) = -a c0 log(1 + c/c0) with a, c0 > 0 learned (|p\'| <= a, sub-linear), local g unconstrained (as G2). '
         '(b) local free energy G(rho;U) = sum_k w_k(U) log cosh(s_k(U) rho/RS + b_k(U)) / s_k(U) + g0(U) rho (one-layer input-convex network, '
         'w, s >= 0), i.e. g = dG/drho monotone and bounded; p(c, U) an unconstrained MLP (as G2). (a)+(b) both. '
         'Collapse = max rho > N^2/2 = 512 at 10T (the whole mass in a few cells). G2 rows: mean over 2 seeds.'] if intro is None else [intro])
    L += ['']
    for title, f in [('10T relative L2', 'err10T'), ('10T relative L2 vs 128-grid reference', 'err10T_r128'), ('peak error at 10T', 'peak_err10T'),
                     ('max rho at 10T (reference: 46.5 / 88.2 / 153.3)', 'max_rho'), ('fraction of steps where the learned F increases', 'Flearn_up_frac'),
                     ('max relative increase of the learned F', 'Flearn_max_rel_up')]:
        L += [f'### {title}', '', '| model | ' + ' | '.join(f'chi/D={c:g}' for c in CH) + ' |', '|---|' + '---|' * len(CH)]
        for m, lab in M:
            if runs(m):
                L.append(f'| {lab} | ' + ' | '.join(ms(ag(m, c, f)) for c in CH) + ' |')
        L.append('')
    L += ['### collapsed trajectories (max rho > 512) / runs, learned parameters', '',
          '| model | runs | collapsed at chi/D=0.3 / 0.5 / 0.8 | learned beta (true -1) | alpha | p: a, c0 | skipped it. | final train loss |', '|---|---|---|---|---|---|---|---|']
    for m, lab in M:
        if not runs(m):
            continue
        col = ' / '.join(str(sum(int(v > NN / 2) for v in ag(m, c, 'max_rho'))) for c in CH)
        b = ', '.join(str(res2[k].get('beta', '-')) for k in runs(m)); a = ', '.join(str(res2[k].get('alpha', '-')) for k in runs(m))
        pa = ', '.join(f"{res2[k]['p_a']}, {res2[k]['p_c0']}" for k in runs(m) if 'p_a' in res2[k]) or '-'
        sk = ', '.join(str(res2[k].get('train_nskip', '-')) for k in runs(m)); fl = ', '.join(f"{res2[k]['final_train_loss']:.2e}" for k in runs(m))
        L.append(f'| {lab} | {len(runs(m))} | {col} | {b} | {a} | {pa} | {sk} | {fl} |')
    L += ['', '(collapse counts are runs whose max rho over all 16 trajectories exceeds 512)', ''] + (extra or []) + ['', f'![{png[:-4]}]({png})', '']
    if write_md:
        sp = os.path.join(ODIR, 'summary.md'); txt = open(sp).read() if os.path.exists(sp) else ''
        i = txt.find(key)
        if i >= 0:
            j = txt.find('\n## ', i + 5)
            txt = txt[:i] + (txt[j:] if j >= 0 else '')
        open(sp, 'w').write(txt.rstrip('\n') + '\n' + '\n'.join(L))
    print('\n'.join(L))
    cols = dict(tcme_s6='#1b6ca8', flux_gnn_s6='#3b8b5a', tcme_s6a='#d1495b', tcme_s6b='#e09f3e', tcme_s6ab='#6a4c93', tcme_s6ab_mc='#c2185b', flux_gnn_s6_mc='#00897b', tcme_s6phys='#000000')
    fig, axs = plt.subplots(1, 3, figsize=(14, 4))
    for m, lab in M:
        if not runs(m):
            continue
        for ax, f in zip(axs, ['err10T', 'peak_err10T', 'max_rho']):
            y = [np.mean(ag(m, c, f)) for c in CH]; e = [np.std(ag(m, c, f)) for c in CH]
            ax.errorbar(CH, y, yerr=e, fmt='o-' if '_mc' in m else 'o--', color=cols.get(m), lw=2, capsize=2, label=lab)
    axs[2].plot(CH, [46.53, 88.15, 153.3], 'k--', lw=1.2, label='reference')
    axs[2].axhline(NN, color='gray', lw=0.8, ls=':')
    for ax, t in zip(axs, ['10T relative L2', 'peak error at 10T', 'max rho at 10T (dotted: N^2 = full collapse)']):
        ax.set_yscale('log'); ax.set_xlabel(xlabel); ax.set_title(t); ax.axvline(0.3, color='k', lw=0.8, ls=':')
        ax.grid(alpha=0.3, which='both')
    axs[2].legend(fontsize=7); fig.suptitle(suptitle)
    fig.tight_layout(); fig.savefig(os.path.join(ODIR, png), dpi=150); plt.close(fig)


def report_g2c(write_md=True):
    """G2c: multi-chi training (chi/D in 0.15, 0.3, 0.45) vs single-chi training; beta identifiability diagnostics."""
    extra = []
    dp = os.path.join(ODIR, 'G2c_diag.json')
    if os.path.exists(dp):
        dg = json.load(open(dp))
        extra += ['### Diagnostic: where is the interaction learned? (all training snapshots)', '',
                  'Collinearity of the true interaction feature c = (chi/D) W*rho with local quantities (pooled over nodes and snapshots):', '',
                  '| training set | R^2 of c on [1, rho] | R^2 of c on [1, rho, rho^2, log rho, U] | corr(c, rho) |', '|---|---|---|---|']
        for sn, lab in [('single', 'single chi/D = 0.3 (S6 train)'), ('multi', 'multi chi/D = 0.15/0.3/0.45')]:
            if sn in dg:
                d = dg[sn]
                extra.append(f'| {lab} | {d["R2_c_on_rho_linear"]:.3f} | {d["R2_c_on_local"]:.3f} | {d["corr_c_rho"]:.3f} |')
        extra += ['', 'Regression of each learned potential component on the true interaction potential -c (slope 1 = carries all of it); '
                  '"total" = beta c + chi W*p(c) + [g(rho,U) - g(0,U)], with R^2 against -c:', '',
                  '| model | evaluated on | beta | slope: beta c | slope: chi W*p(c) | slope: local g | slope: total | R^2 total |', '|---|---|---|---|---|---|---|---|']
        for sn in ['single', 'multi']:
            for rn, v in dg.get(sn, {}).items():
                if not isinstance(v, dict) or 'total' not in v:
                    continue
                extra.append(f'| {rn} | {sn} | {v["beta"]:.3f} | {v["beta_c"]["slope_on_minus_c"]:.3f} | {v["chiWp"]["slope_on_minus_c"]:.3f} | '
                             f'{v["g_local"]["slope_on_minus_c"]:.3f} | {v["total"]["slope_on_minus_c"]:.3f} | {v["total"]["R2"]:.3f} |')
        extra.append('')
    report_g2b(M=[('tcme_s6ab', '(a)+(b), single chi (G2b)'), ('tcme_s6ab_mc', '(a)+(b), multi chi (G2c)'),
                  ('flux_gnn_s6', 'Flux-GNN, single chi (G2, 2 seeds)'), ('flux_gnn_s6_mc', 'Flux-GNN, multi chi (G2c)')],
               head='## G2c: multi-chi training (is beta identifiable, does extrapolation improve?)',
               intro='Training set: S6 generator with chi/D in {0.15, 0.3, 0.45}, 16 trajectories x 20 steps each (48 in total; '
                     'single-chi G2/G2b: 64 trajectories at 0.3). chi enters both models exactly as in G2 (through the state-dependent '
                     'input c = (chi/D) W*rho, and as the prefactor of W*p for TC-ME); architectures are unchanged. '
                     'TC-ME = constrained (a)+(b) of G2b (bounded p, convex local ICNN); 1 seed each, 2500 it, same protocol. '
                     'Test: chi/D = 0.3 (in range), 0.5 (just outside), 0.8 (extrapolation).',
               png='G2c_stability.png', key='\n## G2c:', suptitle='G2c: multi-chi vs single-chi training (S6)', extra=extra,
               xlabel='chi / D  (single-chi training: 0.3; multi-chi: 0.15, 0.3, 0.45)', write_md=write_md)


def report_g2d():
    """G2d: TC-ME with the known physical form of F (learn only s and k) next to G2 / G2b / G2c / Flux-GNN."""
    report_g2b(M=[('tcme_s6phys', 'G2d: physical F, learn s and k (single chi)'), ('tcme_s6', 'G2: TC-ME unconstrained (single chi, 2 seeds)'),
                  ('tcme_s6ab', 'G2b: (a)+(b) (single chi)'), ('tcme_s6ab_mc', 'G2c: (a)+(b) (multi chi)'),
                  ('flux_gnn_s6', 'Flux-GNN (single chi, 2 seeds)'), ('flux_gnn_s6_mc', 'Flux-GNN (multi chi)')],
               head='## G2d: known physical form of the free energy (learn only the interaction scale s and the rates k)',
               intro='TC-ME-S6 with F = sum rho log rho + sum U rho - (s chi/D / 2) rho^T (W*rho), i.e. phi = U - s c, c = (chi/D) W*rho, W known; '
                     'learned: the scalar s (true 1; reported below as beta = -s) and the edge-rate MLP k. No local g, no p, cycle part A = 0. '
                     'Single-chi training set (chi/D = 0.3), 1 seed, 2500 it, same protocol (checkpointing + substep cap 64).',
               png='G2d_stability.png', key='\n## G2d:', suptitle='G2d: physical-form free energy vs. learned free energies and Flux-GNN (S6)')


if __name__ == '__main__':
    p = argparse.ArgumentParser()
    p.add_argument('--gen_g1', action='store_true'); p.add_argument('--g1', action='store_true'); p.add_argument('--report', action='store_true')
    p.add_argument('--models', default='tcme,fisher_hodge,flux_gnn')
    p.add_argument('--from_log', default=None)
    p.add_argument('--report_g2b', action='store_true')
    p.add_argument('--report_g2c', action='store_true')
    p.add_argument('--report_g2d', action='store_true'); p.add_argument('--g2c_plot_only', action='store_true')
    a = p.parse_args()
    if a.from_log:
        from_log(a.from_log)
    if a.gen_g1:
        gen_g1()
    if a.g1:
        g1(a.models.split(','))
    if a.report:
        report()
    if a.report_g2b:
        report_g2b()
    if a.report_g2c:
        report_g2c()
    if a.g2c_plot_only:
        report_g2c(write_md=False)
    if a.report_g2d:
        report_g2d()
