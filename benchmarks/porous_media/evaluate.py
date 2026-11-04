"""Evaluation of the porous-media benchmark: rollout errors, physics diagnostics and the learned mobility.

Error: relative L2 error of the density over the cells, per trajectory, median over the 64 trajectories of a set.
Physics (sampled every 10 steps of the 10T rollout):
  * f_e = mu_i - mu_j + amp int_e F.dl / T, mu = log rho + U / T, is the local force under the condition of the set;
  * an edge violates local detailed balance when J f < -1e-6 (a + b) |f| (neuralchart.diagnostics; scale |J| for a
    model with a single edge flux);
  * sigma_phys = sum_e J_e f_e; the fraction of samples with sigma_phys < 0;
  * the fraction of negative cell values (every model).
The edge statistics need the model's own edge flux.  For models without one they are reported as N/A.
"""
import math

import numpy as np
import torch

from neuralchart.chart import NeuralChart
from neuralchart.diagnostics import ldb_violation, relerr_traj
from .data import D, DT, RHO_S

PHYS_EVERY = 10
K_SETS = ('training_conditions', 'low_porosity', 'grid64', 'point_cloud')     # sets of the learned-law table


def rollout_metrics(R, Y, dens=None):
    """R, Y (B, T+1, V) states on the CPU; errors on densities (state x dens), mass on states"""
    R, Y = R.double(), Y.double()
    w = 1.0 if dens is None else dens.double().cpu()[None, None]
    e = relerr_traj(R * w, Y * w).numpy()
    B = R.shape[0]
    fin = torch.isfinite(R.reshape(B, -1)).all(1)
    m0 = Y[:, 0].sum(-1)
    drift = (R.sum(-1) - m0[:, None]).abs() / m0[:, None]
    return dict(err_T=float(np.median(e[:, 20])), err_10T=float(np.median(e[:, -1])), err_10T_mean=float(np.mean(e[:, -1])),
                err_10T_per_traj=e[:, -1].tolist(), n_blown=int((~fin).sum()),
                n_neg_traj=int((torch.nan_to_num(R, nan=0.0) < 0).reshape(B, -1).any(1).sum()),
                min_value=float(np.nanmin(R.numpy())), mass_drift_max=float(np.nanmax(drift.numpy())))


def flux_metrics(Jm, Jt):
    Jm, Jt = Jm.double().cpu(), Jt.double().cpu()
    e = (torch.linalg.vector_norm(Jm - Jt, dim=-1) / torch.linalg.vector_norm(Jt, dim=-1)).numpy()
    cos = ((Jm * Jt).sum(-1) / (torch.linalg.vector_norm(Jm, dim=-1) * torch.linalg.vector_norm(Jt, dim=-1) + 1e-30)).numpy()
    return dict(J_relerr=float(np.median(np.where(np.isfinite(e), e, np.inf))), J_cos=float(np.nanmedian(cos)))


def pearson(x, y):
    x = x - x.mean(); y = y - y.mean()
    return float((x * y).sum() / (np.sqrt((x * x).sum() * (y * y).sum()) + 1e-30))


def rankdata(x):
    return np.argsort(np.argsort(x)).astype(np.float64)


@torch.no_grad()
def k_metrics(model, sd, disc):
    """learned mobility exp(kappa) against the true mobility averaged harmonically along every edge, on the distinct
    media of a set"""
    med = sd['med'].numpy()
    first = np.unique(med, return_index=True)[1]
    idx = torch.as_tensor(first, device=sd['c'].device)
    drive = {k: v[idx] for k, v in sd['drive'].items()}
    feats = model.featurizer(drive)
    gh = torch.exp(model.kinetics.core.double()[model.kinetics.typ] - math.log(D))[None].expand(feats['edge_sym'].shape[:2]).cpu().numpy().ravel()
    gt = disc.mtrue[sd['pool']][torch.as_tensor(med[first])].double().cpu().numpy().ravel()
    eps = (0.5 * (drive['ein'][..., 0] + 1)).double().cpu().numpy().ravel()
    lh, lt = np.log(gh), np.log(gt)
    op = gt > 1e-2
    out = dict(pearson_logk=pearson(lh, lt), spearman_logk=pearson(rankdata(lh), rankdata(lt)), pearson_k=pearson(gh, gt),
               relerr_k=float(np.linalg.norm(gh - gt) / np.linalg.norm(gt)), median_abs_logratio=float(np.median(np.abs(lh - lt))),
               pearson_logk_open=pearson(lh[op], lt[op]), frac_open=float(op.mean()),
               median_logratio=float(np.median(lh - lt)))
    return out, dict(g_hat=gh.astype(np.float32), g_true=gt.astype(np.float32), eps=eps.astype(np.float32))


@torch.no_grad()
def k_metrics_state(model, sd, disc, crowd, every=20):
    """learned against true edge mobility at the same states: the reference densities of every test trajectory, sampled
    every `every` steps over 10T.  Learned: omega exp(g(class, xi_e, log rhobar_e)) / D.  True: the edge-averaged mobility M_e,
    divided by (1 + rhobar_e / RHO_S) under crowding, with rhobar_e the mean density of the two cells on the same reference
    state."""
    kin = model.kinetics
    x = model.featurizer(sd['drive'])['edge_sym']
    E = disc.g['edges'].to(x.device)
    gt0 = disc.mtrue[sd['pool']][sd['med'].to(disc.mtrue[sd['pool']].device)].double().to(x.device)
    C = sd['c'][:, ::every].double().to(x.device)                             # (N, S, n)
    rho = C * torch.exp(-disc.g['logV']).to(C)[None, None]
    rbar = 0.5 * (rho[..., E[:, 0]] + rho[..., E[:, 1]])                      # (N, S, H)
    core = kin.core.double()[kin.typ] - math.log(D)
    lh = core[None, None].expand_as(rbar).clone()
    if kin.rho_dependent:
        for t in range(C.shape[1]):
            lh[:, t] = lh[:, t] + kin.state_term(x.double(), C[:, t])
    gt = gt0[:, None] / (1 + rbar / RHO_S) if crowd else gt0[:, None].expand_as(rbar)
    gh, gt = torch.exp(lh).cpu().numpy().ravel(), gt.cpu().numpy().ravel()
    eps = (0.5 * (sd['drive']['ein'][..., 0] + 1)).double()[:, None].expand_as(rbar).cpu().numpy().ravel()
    rw = rbar.cpu().numpy().ravel()                                         # flux scale: J_e ~ k_e rhobar_e
    lh, lt = np.log(gh), np.log(gt)
    out = dict(pearson_logk=pearson(lh, lt), spearman_logk=pearson(rankdata(lh), rankdata(lt)), pearson_k=pearson(gh, gt),
               relerr_k=float(np.linalg.norm(gh - gt) / np.linalg.norm(gt)), median_abs_logratio=float(np.median(np.abs(lh - lt))),
               median_logratio=float(np.median(lh - lt)), n=int(gh.size), every=every,
               relerr_k_flux=float(np.linalg.norm((gh - gt) * rw) / np.linalg.norm(gt * rw)),
               mean_logratio_flux=float(np.average(lh - lt, weights=gt * rw)))
    i = np.random.default_rng(0).choice(gh.size, min(gh.size, 20000), replace=False)
    return out, dict(g_hat=gh[i].astype(np.float32), g_true=gt[i].astype(np.float32), eps=eps[i].astype(np.float32))


def phys_force(c, F, disc):
    """local thermodynamic force of every edge under the given condition, f_e = mu_tail - mu_head + amp int_e F.dl / T,
    mu = log rho + U / T (rho the cell density); NaN where an endpoint density is not positive"""
    logV = disc.g['logV'].to(c.device, torch.float64)
    E = disc.g['edges'].to(c.device)
    rho = c.double() * torch.exp(-logV)[None]
    T = F['T'].double()[:, None]
    mu = torch.where(rho > 0, torch.log(rho.clamp_min(1e-300)), torch.full_like(rho, float('nan'))) + F['U'].double() / T
    return mu[:, E[:, 0]] - mu[:, E[:, 1]] + F['amp'].double()[:, None] * F['line'].double() / T


class PhysAcc:
    """accumulates, over trajectories and sampled times, the fraction of edges with J f < 0 (violated local detailed
    balance), the physical entropy production sigma = sum_e J_e f_e (fraction of samples with sigma < 0, its most
    negative value relative to the median |sigma|), and the fraction of negative cell values."""
    def __init__(s):
        s.nv = s.nval = s.nsneg = s.ns = s.nneg = s.ncell = 0
        s.gross = None
        s.smin, s.sigs = float('inf'), []

    def add(s, J, f, c, gross=None):
        J = J.double()
        ok = torch.isfinite(f) & torch.isfinite(J)
        thr = 1e-9 * J.abs().nan_to_num(0).amax(-1, keepdim=True).clamp_min(1e-30)
        val = ok & (J.abs() > thr)
        s.nv += int((ldb_violation(J, f, None if gross is None else gross.double()) & val).sum()); s.nval += int(val.sum())
        s.gross = gross is not None
        sig = torch.where(ok, J * f, torch.zeros_like(J)).sum(-1)
        fin = torch.isfinite(sig)
        s.nsneg += int((sig[fin] < 0).sum()); s.ns += int(fin.sum())
        s.sigs.append(sig[fin].cpu())
        s.nneg += int((c < 0).sum()); s.ncell += c.numel()

    def result(s):
        sg = torch.cat(s.sigs) if s.sigs else torch.zeros(0)
        med = float(sg.abs().median()) if len(sg) else float('nan')
        return dict(ldb_violation_frac=s.nv / max(s.nval, 1), sigma_neg_frac=s.nsneg / max(s.ns, 1),
                    sigma_min_rel=(float(sg.min()) / med if len(sg) and med > 0 else float('nan')),
                    neg_cell_frac=s.nneg / max(s.ncell, 1), ldb_rule=('J f < -1e-6 (a + b) |f|' if s.gross else 'J f < -1e-6 |J| |f|' if s.gross is not None else 'n/a'))


def phys_metrics_rollout(R, F, disc, flux_fn=None, acc=None, rates_fn=None):
    """R (B, T+1, V) rollout states, F the drive of these trajectories.  flux_fn(c) -> J (the model's own edge flux);
    otherwise the minimum-norm flux of the step change (R[t+1] - R[t]) / DT (curl-free by construction).  rates_fn(c)
    -> (a, b) for a model with forward / backward rates: J = a - b and the LDB test is scaled by a + b (neuralchart.diagnostics)."""
    acc = acc or PhysAcc()
    for t in range(0, R.shape[1] - 1, PHYS_EVERY):
        c = R[:, t]
        if rates_fn is not None:
            a, b = rates_fn(c)
            J, gross = a - b, a + b
        else:
            J = flux_fn(c) if flux_fn is not None else disc.minnorm_flux((R[:, t + 1] - R[:, t]) / DT)
            gross = None
        acc.add(J, phys_force(c, F, disc), c, gross=gross)
    return acc


def local_sigma(c, J, F, disc):
    """cell map of the physical entropy production, half of J_e f_e to each endpoint (V,)"""
    f = phys_force(c[None], {k: (v[None] if torch.is_tensor(v) else v) for k, v in F.items()}, disc)[0]
    q = torch.nan_to_num(J.double() * f)
    E = disc.g['edges'].to(c.device)
    return (torch.zeros(disc.V, dtype=torch.float64, device=c.device).index_add_(0, E[:, 0], 0.5 * q).index_add_(0, E[:, 1], 0.5 * q))



def oracle_drive(data):
    """the true edge mobility as an input of every set (for the oracles)"""
    for s, sd in data.items():
        if isinstance(sd, dict) and 'drive' in sd:
            sd['drive']['mtrue'] = true_edge_input(sd, data['discs'][sd['disc']])


def true_edge_input(sd, disc):
    """the true edge mobility of the trajectories of a set, (B, H)"""
    return disc.mtrue[sd['pool']][sd['med'].to(sd['c'].device)]


@torch.no_grad()
def set_metrics(name, model, sd, R, Yc, m, disc, crowd):
    """the readouts of one test set, added to the dict m: R, Yc (B, T + 1, V) float64 on the CPU.  Errors on the densities, mass, the
    physics of the model's own flux (the minimum-norm flux of the step change for a model without one), the flux against the reference and
    the learned law of a Neural Chart with a state-dependent mobility"""
    m.update(rollout_metrics(R, Yc, disc.g['dens']))
    if name == 'persistence':
        return
    dr = sd['drive']
    Rd = R.to(sd['c'].device, dtype=sd['c'].dtype)
    own = hasattr(model, 'flux')
    two = own and not getattr(model, 'free_flux', False)     # forward / backward rates a, b
    ctx = model.prepare(dr) if own else None
    acc = phys_metrics_rollout(Rd, dr, disc, (lambda c: model.flux(c, ctx)) if own and not two else None, rates_fn=(lambda c: model.rates(c, ctx)) if two else None)
    m['phys'] = dict(acc.result(), flux='own' if own else 'min-norm')
    if own and sd['J_end'] is not None:
        last = Rd[:, -1]
        J = model.flux(last, dr) if torch.isfinite(last).all() and last.min() > 0 else torch.full((last.shape[0], disc.H), float('nan'))
        m.update(flux_metrics(J, sd['J_end']))
    if isinstance(model, NeuralChart) and model.kinetics.net is not None:
        m['k'], _ = k_metrics(model, sd, disc)
        if sd['name'] in K_SETS:
            m['k_state'], _ = k_metrics_state(model, sd, disc, crowd)
