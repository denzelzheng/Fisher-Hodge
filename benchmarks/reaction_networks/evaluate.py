"""Readouts of the reaction-network benchmarks: the error channels, ln gamma, the learned laws, front speed, Maxwell point."""
import math
import time

import numpy as np
import torch

from . import config as C
from . import truth as TR
from .models import next_dtype


def err_channels(n):
    """the state channels of the trajectory error: X and Xp on the phosphorylation cycle (E is constant), all otherwise"""
    return slice(0, 2 * n * n) if C.is_pc() else slice(None)


def lngamma_curve(model, x):
    """the model's excess chemical potential mu - log x - mu0 of X at uniform states x (1-D numpy)"""
    fe = model.free_energy
    p = next(iter(fe.parameters()), None)
    dev = p.device if p is not None else 'cpu'
    c = torch.as_tensor(x, dtype=torch.float64, device=dev)[:, None].expand(len(x), model.hg.M).contiguous()
    with torch.no_grad():
        return fe.correction(c, None)[:, 0].double().cpu().numpy() if fe.c_dependent else np.zeros(len(x))


def lngamma_metrics(model, norm, ood_max):
    """the learned ln gamma against the true one over the training range [c_min, c_max] of the data and above it up to
    ood_max; 'table': 12 points of the training range"""
    xi = np.linspace(norm['c_min'], norm['c_max'], 200)
    xo = np.linspace(norm['c_max'], ood_max, 50)
    li, lo = lngamma_curve(model, xi), lngamma_curve(model, xo)
    ti, to = TR.lng_np(xi, TR.PHI), TR.lng_np(xo, TR.PHI)
    slope = float(np.polyfit(ti, li, 1)[0])
    xt = np.linspace(norm['c_min'], norm['c_max'], 12)
    return dict(rmse_id=float(np.sqrt(np.mean((li - ti) ** 2))), maxerr_id=float(np.abs(li - ti).max()),
                rmse_ood=float(np.sqrt(np.mean((lo - to) ** 2))), slope_id=slope, x_id=[xi[0], xi[-1]], x_ood=[xo[0], xo[-1]],
                curve_x=np.linspace(0.02, 2.6, 60).tolist(), curve_model=lngamma_curve(model, np.linspace(0.02, 2.6, 60)).tolist(),
                table=dict(x=xt.tolist(), model=lngamma_curve(model, xt).tolist(), true=TR.lng_np(xt, TR.PHI).tolist()))


def _law(a, b):
    """learned a against true b (log k or an excess potential), flattened numpy"""
    from scipy.stats import spearmanr
    ok = a.std() > 1e-12 and b.std() > 1e-12
    return dict(pearson=float(np.corrcoef(a, b)[0, 1]) if ok else None, spearman=float(spearmanr(a, b).statistic) if ok else None,
                rmse=float(np.sqrt(np.mean((a - b) ** 2))), slope=float(np.polyfit(b, a, 1)[0]) if ok else None,
                median_diff=float(np.median(a - b)), std_true=float(b.std()), max_abs=float(np.abs(a).max()))


@torch.no_grad()
def k_law(model, sd, hg, every=20):
    """the learned symmetric rate constant against the true one on the reference states of a test set, over all hyperedges and per
    class on the phosphorylation networks"""
    from .models import true_log_k
    dev = next(iter(model.parameters())).device
    dt = next_dtype(model)
    drive = {k: v.to(dev).to(dt) for k, v in sd['drive'].items()}
    ctx = model.prepare(drive)
    typ = torch.as_tensor(hg.meta['typ'])
    lm, lt = [], []
    for t in range(0, sd['c'].shape[1], every):
        c = sd['c'][:, t].to(dev).to(dt)
        k = ctx['k']
        if model.kinetics.rho_dependent:
            k = k * torch.exp(model.kinetics.state_term(ctx['feats']['edge_sym'], c))
        lm.append(torch.log(k).double().cpu())
        lt.append(TR.pc_true_log_k(hg, sd['drive'], c.double().cpu()) if C.is_pc() else
                  torch.as_tensor(true_log_k(), dtype=torch.float64)[typ][None].expand(c.shape[0], -1))
    lm, lt = torch.cat(lm), torch.cat(lt)
    groups = [('all', torch.ones_like(typ, dtype=torch.bool))]
    if C.is_pc():
        groups += [('R1', typ == 0), ('R2', typ == 1), ('transport X', typ == 2), ('transport Xp', typ == 3)]
    return {n: _law(lm[:, mk].reshape(-1).numpy(), lt[:, mk].reshape(-1).numpy()) for n, mk in groups}


@torch.no_grad()
def excess_law(model, sd, every=20):
    """the learned excess potential against the true one on the reference states of a test set, overall and per species"""
    dev = next(iter(model.parameters())).device
    dt = next_dtype(model)
    fe = model.free_energy
    L, Tt = [], []
    for t in range(0, sd['c'].shape[1], every):
        c = sd['c'][:, t].to(dev).to(dt)
        L.append(fe.correction(c, None).double().cpu())
        cd = c.double().cpu()
        Tt.append(TR.pc_excess(cd) if C.interacting() else TR.lng_t(cd, TR.PHI) if C.nonideal() else torch.zeros_like(cd))
    L, Tt = torch.cat(L), torch.cat(Tt)
    out = dict(all=_law(L.reshape(-1).numpy(), Tt.reshape(-1).numpy()))
    if C.is_pc():
        for name, a, b in zip(('X', 'Xp', 'E'), L.chunk(3, -1), Tt.chunk(3, -1)):
            out[name] = _law(a.reshape(-1).numpy(), b.reshape(-1).numpy())
    return out


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
