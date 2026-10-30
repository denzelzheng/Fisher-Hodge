"""Evaluation on the test sets: the 10T error, negative values, local detailed balance and sigma against the
reference force, thermodynamic readouts, and the extra readouts of each benchmark (front speed, ln gamma, Kpp
fraction, moiety drift)."""
import math
import time

import numpy as np
import torch

from neuralchart import fast
from neuralchart.diagnostics import ldb_violation, LDB_RULE, relerr_traj
from neuralchart.fit import rollout
from . import config as C
from . import truth as TR
from .data import load_set
from .models import UDE, rebind, set_max_sub, next_dtype, accelerate


def err_channels(n):
    """the state channels of the trajectory error: X and Xp on the phosphorylation cycle (E is constant), all otherwise"""
    return slice(0, 2 * n * n) if C.SYSTEM == 'phosphorylation_cycle' else slice(None)


def lngamma_curve(model, x):
    """the model's excess chemical potential mu - log x - mu0 of X at uniform states x (1-D numpy)"""
    fe = model.free_energy
    c = torch.as_tensor(x, dtype=torch.float64)[:, None].expand(len(x), model.hg.M).contiguous()
    with torch.no_grad():
        return fe.correction(c, None)[:, 0].double().cpu().numpy() if fe.c_dependent else np.zeros(len(x))


def lngamma_metrics(model, norm, ood_max):
    xi = np.linspace(norm['c_min'], norm['c_max'], 200)
    xo = np.linspace(norm['c_max'], ood_max, 50)
    li, lo = lngamma_curve(model, xi), lngamma_curve(model, xo)
    ti, to = TR.lng_np(xi, TR.PHI), TR.lng_np(xo, TR.PHI)
    slope = float(np.polyfit(ti, li, 1)[0])
    return dict(rmse_id=float(np.sqrt(np.mean((li - ti) ** 2))), maxerr_id=float(np.abs(li - ti).max()),
                rmse_ood=float(np.sqrt(np.mean((lo - to) ** 2))), slope_id=slope, x_id=[xi[0], xi[-1]], x_ood=[xo[0], xo[-1]],
                curve_x=np.linspace(0.02, 2.6, 60).tolist(), curve_model=lngamma_curve(model, np.linspace(0.02, 2.6, 60)).tolist())


def front_speed(Cs, bb, n, a=3.0):
    """the high-state fraction m(t) of a stripe start; slope of m until it leaves [0.1, 0.9], in sites per unit time
    (two interfaces); > 0: the high state invades"""
    v = np.zeros(len(Cs))
    t = np.arange(Cs.shape[1]) * C.DT
    for i in range(len(Cs)):
        r = TR.roots(a, bb[i])
        xlo, xhi = TR.x_of_y(r[0]), TR.x_of_y(r[-1])
        mm = ((Cs[i] - xlo) / (xhi - xlo)).mean(-1)
        if not np.isfinite(mm).all():
            v[i] = np.nan; continue
        out = np.where((mm < 0.1) | (mm > 0.9))[0]
        j = max(out[0] if len(out) else len(t), 3)
        v[i] = np.polyfit(t[:j], mm[:j], 1)[0] * n / 2
    return v


def zero_cross(v, bb):
    for i in range(len(v) - 1):
        if np.isfinite(v[i]) and np.isfinite(v[i + 1]) and v[i] < 0 <= v[i + 1]:
            return float(bb[i] - v[i] * (bb[i + 1] - bb[i]) / (v[i + 1] - v[i]))
    return float('nan')


@torch.no_grad()
def maxwell_equal_area(model, a=3.0, blo=1.5, bhi=2.5):
    """the Maxwell point of a model with its own flux: the b at which the reaction rate R(x) of a uniform state has
    zero area between its outer roots"""
    hg = model.hg
    dev = next(iter(model.parameters())).device
    xs = np.linspace(0.01, 6.0, 3000)
    c = torch.as_tensor(xs, dtype=next_dtype(model), device=dev)[:, None].expand(len(xs), hg.M).contiguous()

    def area(b):
        lab = torch.tensor([[math.log(a), math.log(b)]], dtype=next_dtype(model), device=dev).repeat(len(xs), 1)
        Rx = hg.apply_S(model.flux(c, dict(logab=lab)))[:, 0].double().cpu().numpy()
        s = np.where(np.diff(np.sign(Rx)) != 0)[0]
        if len(s) < 3:
            return float('nan')
        i0, i2 = s[0] + 1, s[-1] + 1
        return float((getattr(np, 'trapezoid', None) or np.trapz)(Rx[i0:i2], xs[i0:i2]))
    fl, fh = area(blo), area(bhi)
    if not (np.isfinite(fl) and np.isfinite(fh)) or fl * fh > 0:
        return float('nan')
    for _ in range(50):
        bm = 0.5 * (blo + bhi); fm = area(bm)
        if not np.isfinite(fm):
            return float('nan')
        if fm * fl > 0:
            blo, fl = bm, fm
        else:
            bhi = bm
    return 0.5 * (blo + bhi)


@torch.no_grad()
def eval_set(name, model, sd, n, log, snap, sname):
    Y = sd['c']
    B = Y.shape[0]
    hg = TR.hg_of(n)
    V = hg.V
    dev = Y.device
    t0 = time.time()
    st = getattr(getattr(model, '_fast', None), 'stepper', None)
    cap0 = st.cap_hits if st is not None else 0
    R, ns, _ = rollout(model, sd['drive'], Y[:, 0], Y.shape[1] - 1, C.DT)
    R = R.double().cpu()
    Yc = Y.double().cpu()
    ch = err_channels(n)
    e = relerr_traj(R[..., ch], Yc[..., ch]).numpy()
    fin = torch.isfinite(R.reshape(B, -1)).all(1)
    lab = {k: v.double().cpu() for k, v in sd['drive'].items()}
    m = dict(err_T=float(np.median(e[:, C.T_STEPS])), err_10T=float(np.median(e[:, -1])), err_10T_mean=float(np.mean(e[:, -1])),
             err_10T_per_traj=e[:, -1].tolist(), err_curve=np.median(e, 0)[::10].tolist(),
             n_blown=int((~fin).sum()), n_neg_traj=int((torch.nan_to_num(R, nan=0.0) < 0).reshape(B, -1).any(1).sum()),
             neg_cell_frac=float((torch.nan_to_num(R, nan=0.0) < 0).double().mean()), min_value=float(np.nanmin(R.numpy())),
             infer_time_s=time.time() - t0, substeps_per_DT=float(ns),
             capped_DT=(st.cap_hits - cap0) if st is not None else None)
    if C.SYSTEM == 'mapk_cascade':          # moieties (L basis of the hypergraph), Kpp fraction at the end
        L = hg.on('cpu', torch.float64)['L']
        q = torch.nan_to_num(R, nan=float('inf')) @ L
        m['moiety_drift'] = dict(all=float((torch.linalg.vector_norm(q - q[:, :1], dim=-1) / torch.linalg.vector_norm(q[:, :1], dim=-1)).max()))
        m['kpp_end_err'] = float(np.nanmedian((TR.kpp_fraction(R[:, -1]) - TR.kpp_fraction(Yc[:, -1])).abs().mean(-1).numpy()))
    if C.SYSTEM == 'phosphorylation_cycle':  # moieties: E at every site, total X + Xp
        E = R[..., 2 * V:]; Xt = R[..., :2 * V].sum(-1)
        m['moiety_drift'] = dict(E=float(((E - E[:, :1]).abs() / E[:, :1]).nan_to_num(float('inf')).max()),
                                 X_total=float(((Xt - Xt[:, :1]).abs() / Xt[:, :1]).nan_to_num(float('inf')).max()))
    has_flux = hasattr(model, 'flux')
    if has_flux:
        # local detailed balance against the reference force of the condition, with the model's own flux
        nv = nval = nsn = ns_ = 0
        for t in range(0, R.shape[1], 10):
            c = R[:, t]
            ok_c = torch.isfinite(c).all(1) & (c > 0).all(1)
            a, b = model.rates(c.to(dev).to(next_dtype(model)).clamp_min(1e-30), sd['drive'])
            a, b = a.double().cpu(), b.double().cpu()
            J, Sab = a - b, a + b
            at, bt = TR.true_rates(hg, lab, c.clamp_min(1e-300))
            f = torch.log(at) - torch.log(bt)
            f = torch.where(ok_c[:, None], f, torch.full_like(f, float('nan')))
            ok = torch.isfinite(f) & torch.isfinite(J)
            val = ok & (J.abs() > 1e-9 * J.abs().nan_to_num(0).amax(-1, keepdim=True).clamp_min(1e-30))
            nv += int((ldb_violation(J, f, gross=None if isinstance(model, UDE) else Sab) & val).sum()); nval += int(val.sum())
            sig = torch.where(ok, J * f, torch.zeros_like(J)).sum(-1)
            good = ok_c & torch.isfinite(sig)
            nsn += int((sig[good] < 0).sum()); ns_ += int(good.sum())
        m['phys'] = dict(ldb_violation_frac=nv / max(nval, 1), sigma_neg_frac=nsn / max(ns_, 1), ldb_rule=LDB_RULE)
    if has_flux and not isinstance(model, UDE):
        # entropy production from the model's own one-way rates on the reference states
        sgm, sgt = [], []
        for t in range(0, Yc.shape[1], 5):
            c = Yc[:, t]
            a, b = model.rates(c.to(dev).to(next_dtype(model)), sd['drive'])
            a, b = a.double().cpu(), b.double().cpu()
            sgm.append(((a - b) * (torch.log(a) - torch.log(b))).sum(-1))
            at, bt = TR.true_rates(hg, lab, c)
            sgt.append(((at - bt) * (torch.log(at) - torch.log(bt))).sum(-1))
        sgm, sgt = torch.stack(sgm, 1), torch.stack(sgt, 1)
        se = (torch.linalg.vector_norm(sgm - sgt, dim=1) / torch.linalg.vector_norm(sgt, dim=1)).numpy()
        m['sigma'] = dict(relerr=float(np.nanmedian(se)), ratio=float(np.median((sgm.mean(1) / sgt.mean(1)).numpy())))
        a, b = model.rates(Yc[:, 0].to(dev).to(next_dtype(model)), sd['drive'])
        la = (torch.log(a) - torch.log(b)).double().cpu()
        ca = TR.model_cycle_affinity(la, V).numpy()
        at, bt = TR.true_rates(hg, lab, Yc[:, 0])
        ct = TR.model_cycle_affinity((torch.log(at) - torch.log(bt)).double(), V).numpy()
        m['cycle_affinity'] = dict(model_mean=float(ca.mean()), true_mean=float(ct.mean()), mae=float(np.abs(ca - ct).mean()))
    idx = list(range(min(4, B)))
    snap[f'R_{sname}'] = R[idx].numpy().astype(np.float32)
    if sname == 'planar_fronts':             # front speed against b across the Maxwell point (a = 3)
        bb = lab['logab'][:, 1].exp().numpy()
        vm, vt = front_speed(R.numpy(), bb, n), front_speed(Yc.numpy(), bb, n)
        ok = np.isfinite(vm) & (np.abs(vt) > 1e-3)
        m['front'] = dict(b=bb.tolist(), v_model=vm.tolist(), v_true=vt.tolist(), b_star_model=zero_cross(vm, bb),
                          b_star_true=zero_cross(vt, bb),
                          speed_relerr_median=float(np.median(np.abs(vm[ok] - vt[ok]) / np.abs(vt[ok]))) if ok.any() else None)
    log(f'[{sname}] T {m["err_T"]:.4g} 10T {m["err_10T"]:.4g} | blown {m["n_blown"]} negative trajectories {m["n_neg_traj"]}'
        + (f' | LDB violations {m["phys"]["ldb_violation_frac"]:.3g} sigma<0 {m["phys"]["sigma_neg_frac"]:.3g}' if 'phys' in m else '')
        + f' | substeps/DT {m["substeps_per_DT"]:.1f} capped {m["capped_DT"]} {m["infer_time_s"]:.0f}s')
    return m


def evaluate(name, model, norm, device, log, sets=None):
    """every test set (or the given ones); models trained on the training grid are rebound to the 16 x 16 grid"""
    res, snap = {}, {}
    for s in sets or C.sets():
        sd = load_set(s, device)
        n = sd['n']
        m_ = rebind(name, model, n, norm, device) if n != C.N_TRAIN else model
        if m_ is model and name == 'ude':
            accelerate(m_, name)                 # the fast path's operators belong to the current grid
        else:
            fast.copy_config(model, m_)
        if hasattr(m_, 'eval'):
            m_.eval()
        set_max_sub(m_, C.MAXSUB_EVAL)
        if next_dtype(model) == torch.float32:
            sd['c'] = sd['c'].float(); sd['drive'] = {k: v.float() for k, v in sd['drive'].items()}
        res[s] = eval_set(name, m_, sd, n, log, snap, s)
        if n != C.N_TRAIN and name in ('ude', 'bsms'):
            model.set_n(C.N_TRAIN)
    return res, snap
