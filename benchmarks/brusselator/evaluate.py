"""Evaluation on the Brusselator test sets.

For every set: the relative L2 error per trajectory and time (median over the set at T and 10T); the limit cycle of
the homogeneous starts (oscillating or not, period and amplitude of the site-mean X over the second half, late
amplitude over the last quarter); the patterns of the spatial starts (dominant wavenumber of X at 10T, spatial
standard deviation); negative concentrations; and, for models with their own one-way rates, local detailed balance
and the entropy production sigma = sum_h J_h f_h under the true force f = log(a_true / b_true) of the test condition.
"""
import numpy as np
import torch

from neuralchart import fast
from neuralchart.diagnostics import ldb_violation, relerr_traj, LDB_RULE
from .config import DT
from .data import hg_of, true_rates
from .models import UDE, next_dtype


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
def set_metrics(model, sd, R, Yc, m):
    """the readouts of one test set, added to the dict m: R, Yc (B, T + 1, M) float64 on the CPU"""
    n = sd['n']
    B = R.shape[0]
    hg = hg_of(n)
    V = hg.V
    dev = sd['c'].device
    chunk = B if fast.is_fast(model) else (16 if n <= 32 else 4)      # the fast integrator runs the set in one batch
    e = relerr_traj(R, Yc).numpy()
    h = sd['homog'].numpy().astype(bool)
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
    if not hasattr(model, 'flux'):
        return
    lab = sd['drive']['logab'].double().cpu()
    dtype = next_dtype(model)
    single = isinstance(model, UDE)                         # one net flux per hyperedge: the LDB scale is |J|
    parts = [(i, {k: v[i:i + chunk] for k, v in sd['drive'].items()}) for i in range(0, B, chunk)]
    ctxs = [model.prepare(dr) for _, dr in parts]
    nv = nval = nsn = ns = 0
    for t in range(0, R.shape[1] - 1, 10):
        c = R[:, t]
        ok_c = torch.isfinite(c).all(1)
        AB = [model.rates(c[i:i + chunk].to(dev).to(dtype), ctx) for (i, _), ctx in zip(parts, ctxs)]
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
    if single:
        return
    # entropy production on the reference states (models with their own one-way rates)
    se, sm_, st_ = [], [], []
    for (i, _), ctx in zip(parts, ctxs):
        sgm, sgt = [], []
        for t in range(0, Yc.shape[1], 5):
            c = Yc[i:i + chunk, t]
            a, b = model.rates(c.to(dev).to(dtype), ctx)
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
    a, b = model.rates(Yc[:, 0].to(dev).to(dtype), model.prepare(sd['drive']))
    la = (torch.log(a) - torch.log(b)).double().cpu()
    at, bt = true_rates(hg, lab, Yc[:, 0])
    lt = torch.log(at) - torch.log(bt)
    cyc = lambda L: [float((L[:, 0] + L[:, 3 * V]).mean()), float((L[:, V] + L[:, 2 * V]).mean())]
    m['cycle_affinity'] = dict(model=cyc(la), true=cyc(lt))
