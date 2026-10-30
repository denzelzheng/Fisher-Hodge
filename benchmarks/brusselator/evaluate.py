"""Evaluation on the Brusselator test sets.

For every set: the relative L2 error per trajectory and time (median over the set at T and 10T); the limit cycle of
the homogeneous starts (oscillating or not, period and amplitude of the site-mean X over the second half, late
amplitude over the last quarter); the patterns of the spatial starts (dominant wavenumber of X at 10T, spatial
standard deviation); negative concentrations; and, for models with their own one-way rates, local detailed balance
and the entropy production sigma = sum_h J_h f_h under the true force f = log(a_true / b_true) of the test condition.
"""
import time

import numpy as np
import torch

from neuralchart import fast
from neuralchart.diagnostics import ldb_violation, relerr_traj, LDB_RULE
from neuralchart.fit import rollout
from .config import DT, N_TRAIN, T_STEPS
from .data import hg_of, true_rates, load_set
from .models import UDE, next_dtype, rebind
from .test_sets import TEST_SETS


def osc_stats(xm, frac=0.5):
    """amplitude and period of the site-mean X over the last `frac` of the rollout (period NaN: no oscillation)"""
    if not np.isfinite(xm).all():
        return np.nan, np.nan
    h = xm[int(len(xm) * (1 - frac)):]
    amp = h.max() - h.min()
    dd = h - h.mean()
    up = np.where((dd[:-1] < 0) & (dd[1:] >= 0))[0]
    if len(up) < 2 or amp < 1e-3 * max(abs(h.mean()), 1e-9):
        return amp, np.nan
    tc = up + dd[up] / (dd[up] - dd[up + 1])
    return amp, float(np.mean(np.diff(tc)) * DT)


def dominant_k(F):
    """radial shell |k| (integer, in units of 2 pi / L) of the strongest non-constant Fourier mode of a field (n, n)"""
    P = np.abs(np.fft.fft2(F - F.mean())) ** 2
    P[0, 0] = 0
    n = F.shape[0]
    k = np.fft.fftfreq(n) * n
    kk = np.sqrt(k[:, None] ** 2 + k[None, :] ** 2)
    i = np.unravel_index(np.argmax(P), P.shape)
    return int(round(kk[i]))


def site_map(hg, q, half=True):
    """hyperedge field (H,) -> site field (V,): a reaction to its site, a transport edge half to each end (half=True)
    or fully to each end (half=False)"""
    V = hg.V
    out = torch.zeros(V, dtype=torch.float64)
    typ = torch.as_tensor(hg.meta['typ'])
    site = torch.as_tensor(hg.meta['site'])
    r = typ < 4
    out.index_add_(0, site[r], q[r].double())
    et, eh = torch.as_tensor(hg.meta['et']), torch.as_tensor(hg.meta['eh'])
    w = 0.5 if half else 1.0
    out.index_add_(0, et[~r], w * q[~r].double()); out.index_add_(0, eh[~r], w * q[~r].double())
    return out


@torch.no_grad()
def eval_set(model, sd, n, log, snap, sname):
    Y = sd['c']
    B = Y.shape[0]
    hg = hg_of(n)
    V = hg.V
    dev = Y.device
    t0 = time.time()
    Rs, nsub = [], []
    chunk = B if fast.is_fast(model) else (16 if n <= 32 else 4)      # the fast integrator runs the set in one batch
    for i in range(0, B, chunk):
        dr = {k: v[i:i + chunk] for k, v in sd['drive'].items()}
        R, ns, _ = rollout(model, dr, Y[i:i + chunk, 0], Y.shape[1] - 1, DT)
        Rs.append(R.double().cpu()); nsub.append(ns)
    R = torch.cat(Rs)
    Yc = Y.double().cpu()
    e = relerr_traj(R, Yc).numpy()
    h = sd['homog'].numpy().astype(bool)
    fin = torch.isfinite(R.reshape(B, -1)).all(1)
    m = dict(err_T=float(np.median(e[:, T_STEPS])), err_10T=float(np.median(e[:, -1])), err_10T_mean=float(np.mean(e[:, -1])),
             err_10T_per_traj=e[:, -1].tolist(), err_curve=np.median(e, 0)[::10].tolist(),
             n_blown=int((~fin).sum()), n_neg_traj=int((torch.nan_to_num(R, nan=0.0) < 0).reshape(B, -1).any(1).sum()),
             neg_cell_frac=float((torch.nan_to_num(R, nan=0.0) < 0).double().mean()), min_value=float(np.nanmin(R.numpy())),
             infer_time_s=time.time() - t0, substeps_per_DT=float(np.mean(nsub)))
    if h.any():
        m['err_10T_homog'] = float(np.median(e[h, -1]))
    if (~h).any():
        m['err_10T_spatial'] = float(np.median(e[~h, -1]))
    # limit cycle (homogeneous starts): site-mean X
    Xr, Xt = R[:, :, :V].mean(-1).numpy(), Yc[:, :, :V].mean(-1).numpy()
    if h.any():
        osc, pe, ae, ar = [], [], [], []
        for i in np.where(h)[0]:
            at, pt = osc_stats(Xt[i]); am, pm = osc_stats(Xr[i])
            atq, _ = osc_stats(Xt[i], 0.25); amq, _ = osc_stats(Xr[i], 0.25)
            osc.append(bool(np.isfinite(pm)))
            ar.append(amq / atq if np.isfinite(amq) and atq > 0 else np.inf)
            if np.isfinite(pt):
                ae.append(abs(am - at) / at if np.isfinite(am) else np.inf)
                pe.append(abs(pm - pt) / pt if np.isfinite(pm) else np.inf)
        m['lc'] = dict(osc_frac=float(np.mean(osc)), true_osc_frac=float(np.mean([np.isfinite(osc_stats(Xt[i])[1]) for i in np.where(h)[0]])),
                       amp_err=float(np.median(ae)) if ae else None, period_err=float(np.median(pe)) if pe else None,
                       late_amp_ratio=float(np.median(ar)), true_late_amp=float(np.median([osc_stats(Xt[i], 0.25)[0] for i in np.where(h)[0]])))
    # patterns (spatial starts): X at 10T
    if (~h).any():
        kmatch, sratio = [], []
        for i in np.where(~h)[0]:
            Ft, Fm = Yc[i, -1, :V].reshape(n, n).numpy(), R[i, -1, :V].reshape(n, n).numpy()
            if not np.isfinite(Fm).all():
                kmatch.append(False); sratio.append(np.inf); continue
            kmatch.append(dominant_k(Ft) == dominant_k(Fm))
            sratio.append(float(Fm.std() / Ft.std()))
        m['turing'] = dict(pattern_err=m['err_10T_spatial'], k_match=float(np.mean(kmatch)), std_ratio=float(np.median(sratio)),
                           true_k=int(np.median([dominant_k(Yc[i, -1, :V].reshape(n, n).numpy()) for i in np.where(~h)[0]])))
    # physical consistency of the model's own flux, against the true force of the test condition
    has_flux = hasattr(model, 'flux')
    lab = sd['drive']['logab'].double().cpu()
    single = isinstance(model, UDE)                         # one net flux per hyperedge: the LDB scale is |J|
    if has_flux:
        nv = nval = nsn = ns = 0
        for t in range(0, R.shape[1] - 1, 10):
            c = R[:, t]
            ok_c = torch.isfinite(c).all(1)
            AB = [model.rates(c[i:i + chunk].to(dev).to(next_dtype(model)), {k: v[i:i + chunk] for k, v in sd['drive'].items()})
                  for i in range(0, B, chunk)]
            Ar = torch.cat([x[0].double().cpu() for x in AB]); Br = torch.cat([x[1].double().cpu() for x in AB])
            J, Sab = Ar - Br, Ar + Br
            pos = (c > 0)
            at, bt = true_rates(hg, lab, c.clamp_min(1e-300))
            f = torch.log(at) - torch.log(bt)
            bad = (hg.dot(1.0 * ~pos, '+') + hg.dot(1.0 * ~pos, '-')) > 0      # f is undefined at a non-positive component
            f = torch.where(bad | ~ok_c[:, None], torch.full_like(f, float('nan')), f)
            ok = torch.isfinite(f) & torch.isfinite(J)
            thr = 1e-9 * J.abs().nan_to_num(0).amax(-1, keepdim=True).clamp_min(1e-30)
            val = ok & (J.abs() > thr)
            nv += int((ldb_violation(J, f, gross=None if single else Sab) & val).sum()); nval += int(val.sum())
            sig = torch.where(ok, J * f, torch.zeros_like(J)).sum(-1)
            good = ok_c & torch.isfinite(sig)
            nsn += int((sig[good] < 0).sum()); ns += int(good.sum())
        m['phys'] = dict(ldb_violation_frac=nv / max(nval, 1), sigma_neg_frac=nsn / max(ns, 1), ldb_rule=LDB_RULE,
                         ldb_scale='|J|' if single else 'a + b')
        # maps at t = 100 of the first spatial trajectory: local entropy production and LDB-violating hyperedges per site
        i = int(np.where(~h)[0][0]) if (~h).any() else 0
        c = R[i:i + 1, 100]
        a1, b1 = model.rates(c.to(dev).to(next_dtype(model)), {k: v[i:i + 1] for k, v in sd['drive'].items()})
        a1, b1 = a1.double().cpu()[0], b1.double().cpu()[0]
        J, Sab = a1 - b1, a1 + b1
        at, bt = true_rates(hg, lab[i:i + 1], c.clamp_min(1e-300))
        f = (torch.log(at) - torch.log(bt))[0]
        snap[f'sig_{sname}'] = site_map(hg, torch.nan_to_num(J * f)).numpy().astype(np.float32)
        snap[f'viol_{sname}'] = site_map(hg, ldb_violation(J, f, gross=None if single else Sab).double(), half=False).numpy().astype(np.float32)
        at, bt = true_rates(hg, lab[i:i + 1], Yc[i:i + 1, 100])
        snap[f'sigtrue_{sname}'] = site_map(hg, ((at - bt) * (torch.log(at) - torch.log(bt)))[0]).numpy().astype(np.float32)
    # entropy production on the reference states (models with their own one-way rates)
    if has_flux and not single:
        se, sm_, st_ = [], [], []
        for i in range(0, B, chunk):
            dr = {k: v[i:i + chunk] for k, v in sd['drive'].items()}
            ctx = model.prepare(dr)
            sgm, sgt = [], []
            for t in range(0, Yc.shape[1], 5):
                c = Yc[i:i + chunk, t]
                a, b = model.rates(c.to(dev).to(next_dtype(model)), ctx)
                a, b = a.double().cpu(), b.double().cpu()
                sgm.append(((a - b) * (torch.log(a) - torch.log(b))).sum(-1))
                at, bt = true_rates(hg, lab[i:i + chunk], c)
                sgt.append(((at - bt) * (torch.log(at) - torch.log(bt))).sum(-1))
            sgm, sgt = torch.stack(sgm, 1), torch.stack(sgt, 1)
            se.append(torch.linalg.vector_norm(sgm - sgt, dim=1) / torch.linalg.vector_norm(sgt, dim=1))
            sm_.append(sgm.mean(1)); st_.append(sgt.mean(1))
        se = torch.cat(se).numpy()
        m['sigma'] = dict(relerr=float(np.nanmedian(se)), ratio=float(np.median((torch.cat(sm_) / torch.cat(st_)).numpy())))
        # cycle affinities at site 0 (R1 + R4: A -> E, R2 + R3: B -> D), model and truth, mean over the set
        ctx = model.prepare(sd['drive'])
        a, b = model.rates(Yc[:, 0].to(dev).to(next_dtype(model)), ctx)
        la = (torch.log(a) - torch.log(b)).double().cpu()
        at, bt = true_rates(hg, lab, Yc[:, 0])
        lt = torch.log(at) - torch.log(bt)
        cyc = lambda L: [float((L[:, 0] + L[:, 3 * V]).mean()), float((L[:, V] + L[:, 2 * V]).mean())]
        m['cycle_affinity'] = dict(model=cyc(la), true=cyc(lt))
    # full rollouts of two homogeneous and two spatial trajectories (figures and animations)
    idx = (list(np.where(h)[0][:2]) + list(np.where(~h)[0][:2]))
    snap[f'R_{sname}'] = R[idx].numpy().astype(np.float32)
    snap[f'idx_{sname}'] = np.array(idx)
    log(f'[{sname}] T {m["err_T"]:.4g} 10T {m["err_10T"]:.4g}'
        + (f' | oscillating {m["lc"]["osc_frac"]:.2f} period err {m["lc"]["period_err"]} amplitude err {m["lc"]["amp_err"]}' if 'lc' in m else '')
        + (f' | k match {m["turing"]["k_match"]:.2f} std ratio {m["turing"]["std_ratio"]:.3g}' if 'turing' in m else '')
        + f' | negative trajectories {m["n_neg_traj"]}'
        + (f' | LDB violations {m["phys"]["ldb_violation_frac"]:.3g} sigma<0 {m["phys"]["sigma_neg_frac"]:.3g}' if 'phys' in m else '')
        + f' | substeps {m["substeps_per_DT"]:.1f} t {m["infer_time_s"]:.0f}s')
    return m


def evaluate(name, model, norm, device, log, sets=TEST_SETS, data=None):
    """evaluate a trained model on the test sets; models are rebound (no retraining) to other grids"""
    res, snap = {}, {}
    kw = {} if data is None else dict(data=data)
    for s in sets:
        sd = load_set(s, device, **kw)
        if next_dtype(model) == torch.float32:
            sd['c'] = sd['c'].float(); sd['drive'] = {k: v.float() for k, v in sd['drive'].items()}
        n = sd['n']
        m_ = rebind(name, model, n, norm, device) if n != N_TRAIN else model
        fast.copy_config(model, m_)         # models rebuilt on the new grid get the same fast configuration
        if hasattr(m_, 'eval'):
            m_.eval()
        res[s] = eval_set(m_, sd, n, log, snap, s)
        if n != N_TRAIN and name in ('ude', 'bsms'):
            rebind(name, model, N_TRAIN, norm, device)       # back to the training grid
    return res, snap
