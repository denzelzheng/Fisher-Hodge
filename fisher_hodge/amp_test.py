"""E6: drive-amplitude extrapolation on S2 (no retraining).
Test set = the 16 test_id trajectories of S2 (same U, same initial conditions), with the divergence-free drift
(stream function psi, hence dpsi and psin) multiplied by s in {0, 0.5, 1, 2, 4}.  Reference: the gen_data.py finite-volume
solver; if the fine-cell Peclet number max|h u|/D exceeds 2 (central advection loses positivity) the reference is run on a
128x128 grid instead of 64x64 and coarse-grained 4x4 -> 32x32.
  python3 amp_test.py --gen        # -> runs/data/S2_amp{s}.pt
  python3 amp_test.py --eval       # -> runs/tcme/E6_amplitude.{json,md,png}"""
import argparse, json, os, time
import numpy as np
import torch
from common import *
import gen_data as G
import models, models2
from models import GNNODE
from models2 import JKOAmort
from train_tcme import MODELS, fields, relerr_b, jclean

dev = 'cuda'
AMPS = [0.0, 0.5, 1.0, 2.0, 4.0]
ODIR = os.path.join(RUNS, 'tcme')
CK = {**{(m, s): os.path.join(ODIR, f'S2_{m}_s{s}.pt') for m in ['tcme', 'fisher_hodge', 'flux_gnn'] for s in [0, 1]},
      **{(m, s): os.path.join(RUNS, 'e6_ckpt', f'S2_{m}_s{s}.pt') for m in ['gnn_ode', 'jko_amort'] for s in [0, 1]}}
CLS = dict(MODELS, gnn_ode=GNNODE, jko_amort=JKOAmort)


def amp_name(s):
    return f'S2_amp{s:g}.pt'


def gen():
    meta = json.load(open(os.path.join(RUNS, 'data', 'S2_meta.json')))
    base = torch.load(os.path.join(RUNS, 'data', 'S2.pt'))['test_id']
    prs = meta['sets']['test_id']['params']
    info = {}
    for s in AMPS:
        t0 = time.time()
        nf = 64
        while True:
            (fXY, fC), (cXY, cC) = G.grids(nf), G.grids(N)
            Pf = torch.tensor(np.stack([s * D * G.eval_field(p['psi'], fC) for p in prs]), dtype=torch.float64, device=dev)
            pe = float((dual_dpsi(Pf) / D).abs().max())
            if pe <= 2.0 or nf >= 128:
                break
            nf = 128
        G.NF = nf
        Uf = torch.tensor(np.stack([D * G.eval_field(p['U'], fXY) for p in prs]), dtype=torch.float64, device=dev)
        R0 = torch.tensor(np.stack([G.eval_ic(p['ic'], fXY) for p in prs]), dtype=torch.float64, device=dev)
        uvel = dual_dpsi(Pf) * nf
        rho = R0.clone(); out = []
        down = (lambda x: coarsen(coarsen(x))) if nf == 128 else coarsen
        dt = G.DT_REF; per = int(round(DT / dt))
        out.append(down(rho))
        for k in range(T_TEST):
            for _ in range(per):
                k1 = G.fine_rhs('S2', rho, Uf, uvel, 0.0)
                k2 = G.fine_rhs('S2', rho + dt * k1, Uf, uvel, 0.0)
                rho = rho + 0.5 * dt * (k1 + k2)
            out.append(down(rho))
        traj = torch.stack(out, 1)
        Jf = G.fine_flux('S2', rho, Uf, uvel)
        Jc = coarsen_flux(Jf) if nf == 64 else coarsen_flux(coarsen_flux(Jf))
        Pc = torch.tensor(np.stack([s * D * G.eval_field(p['psi'], cC) for p in prs]), dtype=torch.float64, device=dev)
        d = dict(rho=traj.float().cpu(), U=base['U'].clone(), dpsi=(dual_dpsi(Pc) / D).float().cpu(),
                 psin=(corner_to_node(Pc) / D).float().cpu(), J_end=Jc.float().cpu())
        chk = float((d['dpsi'] - s * base['dpsi']).abs().max())
        drift = float((torch.linalg.vector_norm((traj[:, -1] - traj[:, -11]).flatten(1), dim=1) / torch.linalg.vector_norm(traj[:, -1].flatten(1), dim=1)).max())
        info[s] = dict(nf=nf, fine_peclet_max=pe, min_rho=float(traj.min()), mass_dev=float((traj.mean((-1, -2)) - 1).abs().max()),
                       dpsi_scale_check=chk, rel_change_last_0p5=drift, coarse_dpsi_max=float(d['dpsi'].abs().max()),
                       relerr_vs_base_s1=float(relerr_b(traj[:, -1].cpu().numpy(), base['rho'][:, -1].numpy()).max()) if s == 1 else None,
                       time_s=time.time() - t0)
        torch.save(dict(test=d), os.path.join(RUNS, 'data', amp_name(s)))
        print(s, json.dumps(info[s]), flush=True)
    G.NF = NF
    json.dump(info, open(os.path.join(ODIR, 'E6_gen_info.json'), 'w'), indent=1)


def flux_phys(model, rho, ctx):
    n = rho.shape[-1]
    if model.name == 'tcme':
        return model.flux(rho, ctx) / n
    if hasattr(model, 'flux_q'):
        return -(D * n) * model.flux_q(rho, ctx)
    return None


def evaluate():
    res = {}
    for (m, seed), ck in CK.items():
        model = CLS[m]('S2').to(dev); model.load_state_dict(torch.load(ck, map_location=dev)); model.eval()
        key = f'{m}_s{seed}'; res[key] = {}
        for s in AMPS:
            t0 = time.time()
            sd = torch.load(os.path.join(RUNS, 'data', amp_name(s)))['test']
            F = fields(sd); Y = sd['rho'].double()
            with torch.no_grad():
                ctx = model.prepare(F)
                r = sd['rho'][:, 0].to(dev); out = [r.cpu()]
                for k in range(T_TEST):
                    r, _ = model.advance(r, ctx, DT)
                    bad = (~torch.isfinite(r)).any(-1).any(-1) | (r.abs().amax((-1, -2)) > 1e4)
                    if bad.any():
                        r = torch.where(bad[:, None, None], torch.full_like(r, float('nan')), r)
                    out.append(r.cpu())
                R = torch.stack(out, 1).double()
                B = R.shape[0]
                err = (torch.linalg.vector_norm((R - Y).reshape(B, T_TEST + 1, -1), dim=-1) / torch.linalg.vector_norm(Y.reshape(B, T_TEST + 1, -1), dim=-1)).numpy()
                err = np.where(np.isfinite(err), err, np.inf)
                fin = torch.isfinite(R[:, -1]).all(-1).all(-1)
                S = dict(err10T=float(np.median(err[:, -1])), errT=float(np.median(err[:, 20])), n_blown=int((~fin).sum()),
                         mass_err=float(np.nanmax(np.abs(R.mean((-1, -2)).numpy() - 1))), min_rho=float(np.nanmin(R.numpy())),
                         n_neg=int((R < 0).any(-1).any(-1).any(-1).sum()))
                Jm = None
                try:
                    rend = torch.where(fin[:, None, None], R[:, -1], torch.ones_like(R[:, -1])).float().to(dev)
                    Jm = flux_phys(model, rend, ctx)
                except Exception as e:
                    print('flux fail', key, e)
                if Jm is not None:
                    Jm = Jm.cpu().numpy(); Jt = sd['J_end'].numpy()
                    cm, ct = curl_f(torch.tensor(Jm)).numpy(), curl_f(torch.tensor(Jt)).numpy()
                    if s == 0:     # no steady current: report ||J_model|| / ||J_true(s=1)||
                        J1 = torch.load(os.path.join(RUNS, 'data', amp_name(1.0)))['test']['J_end'].numpy()
                        S['J_norm_over_s1'] = float(np.median(np.linalg.norm(Jm.reshape(B, -1), axis=1) / np.linalg.norm(J1.reshape(B, -1), axis=1)))
                        S['curlJ_norm_over_s1'] = float(np.median(np.linalg.norm(cm.reshape(B, -1), axis=1) / np.linalg.norm(curl_f(torch.tensor(J1)).numpy().reshape(B, -1), axis=1)))
                    else:
                        S['J_err'] = float(np.median(np.where(fin.numpy(), relerr_b(Jm, Jt), np.inf)))
                        S['curlJ_err'] = float(np.median(np.where(fin.numpy(), relerr_b(cm, ct), np.inf)))
            res[key][s] = S
            print(f'{key:16s} s={s:<4g} 10T {S["err10T"]:.3e} T {S["errT"]:.3e} blown {S["n_blown"]} mass {S["mass_err"]:.1e} neg {S["n_neg"]} '
                  f'J {S.get("J_err", S.get("J_norm_over_s1", float("nan"))):.3e} curl {S.get("curlJ_err", S.get("curlJ_norm_over_s1", float("nan"))):.3e} {time.time() - t0:.0f}s', flush=True)
    json.dump(jclean(res), open(os.path.join(ODIR, 'E6_amplitude.json'), 'w'))
    report(res)


def report(res):
    import matplotlib; matplotlib.use('Agg'); import matplotlib.pyplot as plt
    info = json.load(open(os.path.join(ODIR, 'E6_gen_info.json')))
    ML = [('tcme', 'TC-ME'), ('fisher_hodge', 'Fisher-Hodge'), ('flux_gnn', 'Flux-GNN'), ('gnn_ode', 'GNN-ODE (4k it)'), ('jko_amort', 'JKO amortised')]
    agg = lambda m, s, f: [res[k][s].get(f) for k in res if k.startswith(m + '_s') and res[k][s].get(f) is not None]
    def ms(v):
        v = np.array(v, float)
        if len(v) == 0: return 'n/a'
        return f'{v.mean():.3g}' + (f'±{v.std():.2g}' if len(v) > 1 else '')
    L = ['# E6: drive-amplitude extrapolation (S2, no retraining)', '',
         'Test = the 16 S2 test_id trajectories (same U, same initial conditions) with the divergence-free drift scaled by s '
         '(s = 1 is the training distribution). Models trained on s = 1 only (tcme / fisher_hodge / flux_gnn: runs/tcme, 2500 it; '
         'gnn_ode: runs/main, 4000 it; jko_amort: runs/stress/main). Median over trajectories, mean±std over 2 seeds. '
         'Steady flux = flux at t = 10 (rho(10) is within ~1e-3 of steady state) vs the reference J_end; for s = 0 the true flux is ~0, '
         'so the table gives ||J_model|| / ||J_true(s=1)|| instead.', '', '## Reference data', '',
         '| s | fine grid | max fine-cell Peclet | coarse max dpsi | min rho | rel. change over last 0.5 time |', '|---|---|---|---|---|---|']
    for s in AMPS:
        i = info[str(s)] if str(s) in info else info[f'{s}']
        L.append(f'| {s:g} | {i["nf"]} | {i["fine_peclet_max"]:.2f} | {i["coarse_dpsi_max"]:.2f} | {i["min_rho"]:.1e} | {i["rel_change_last_0p5"]:.1e} |')
    for title, f in [('10T relative L2', 'err10T'), ('T relative L2', 'errT'), ('steady flux relative error (s=0: ||J||/||J_true(s=1)||)', None),
                     ('curl of steady flux relative error (s=0: ||curl J||/||curl J_true(s=1)||)', None)]:
        L += ['', f'## {title}', '', '| model | ' + ' | '.join(f's={s:g}' for s in AMPS) + ' |', '|---|' + '---|' * len(AMPS)]
        for m, lab in ML:
            cells = []
            for s in AMPS:
                ff = f if f else (('J_norm_over_s1' if s == 0 else 'J_err') if 'curl' not in title else ('curlJ_norm_over_s1' if s == 0 else 'curlJ_err'))
                cells.append(ms(agg(m, s, ff)))
            L.append(f'| {lab} | ' + ' | '.join(cells) + ' |')
    L += ['', '## Mass drift (max) / trajectories with rho<0 (of 32) / blown-up trajectories', '', '| model | ' + ' | '.join(f's={s:g}' for s in AMPS) + ' |', '|---|' + '---|' * len(AMPS)]
    for m, lab in ML:
        L.append(f'| {lab} | ' + ' | '.join(f'{max(agg(m, s, "mass_err")):.1e} / {sum(agg(m, s, "n_neg"))} / {sum(agg(m, s, "n_blown"))}' for s in AMPS) + ' |')
    L += ['', '![E6](E6_amplitude.png)', '']
    open(os.path.join(ODIR, 'E6_amplitude.md'), 'w').write('\n'.join(L))
    print('\n'.join(L))
    cols = dict(tcme='#1b6ca8', fisher_hodge='#d1495b', flux_gnn='#3b8b5a', gnn_ode='#e09f3e', jko_amort='#777777')
    fig, axs = plt.subplots(1, 2, figsize=(10.5, 4))
    for m, lab in ML:
        y = [np.mean(agg(m, s, 'err10T')) for s in AMPS]
        axs[0].plot(AMPS, y, 'o-', color=cols[m], lw=2, label=lab)
        yc = [np.mean(agg(m, s, 'curlJ_err')) if agg(m, s, 'curlJ_err') else np.nan for s in AMPS[1:]]
        if np.isfinite(yc).any():
            axs[1].plot(AMPS[1:], yc, 'o-', color=cols[m], lw=2, label=lab)
    for ax, t in zip(axs, ['10T relative L2 error', 'curl J relative error at t = 10']):
        ax.set_yscale('log'); ax.set_xlabel('drift amplitude s  (training: s = 1)'); ax.set_title(t)
        ax.axvline(1.0, color='k', lw=0.8, ls=':'); ax.grid(alpha=0.3, which='both')
    axs[0].legend(fontsize=8)
    fig.suptitle('E6: drive-amplitude extrapolation, S2 (no retraining)')
    fig.tight_layout(); fig.savefig(os.path.join(ODIR, 'E6_amplitude.png'), dpi=150)


if __name__ == '__main__':
    p = argparse.ArgumentParser(); p.add_argument('--gen', action='store_true'); p.add_argument('--eval', action='store_true')
    a = p.parse_args()
    if a.gen:
        gen()
    if a.eval:
        evaluate()
