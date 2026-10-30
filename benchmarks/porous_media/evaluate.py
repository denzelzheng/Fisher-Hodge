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
import time

import numpy as np
import torch

from neuralchart import fast
from neuralchart.chart import NeuralChart
from neuralchart.diagnostics import ldb_violation, relerr_traj
from neuralchart.fit import rollout
from .data import DT, RHO_S
from .models import rebind
from .test_sets import TEST_SETS

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
    drive = {k: v[idx] for k, v in sd['drive'].items() if torch.is_tensor(v) and v.shape[0] == len(med)}
    feats = model.featurizer(drive)
    gh = torch.exp(model.kinetics.kappa(feats)).double().cpu().numpy().ravel()
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
    every `every` steps over 10T.  Learned: exp(kappa(xi_e) + s m(log rhobar_e)) (the material part only for a model
    without a state term).  True: the edge-averaged mobility M_e, divided by (1 + rhobar_e / RHO_S) under crowding,
    with rhobar_e the mean density of the two cells on the same reference state."""
    kin = model.kinetics
    feats = model.featurizer(sd['drive'])
    kap = kin.kappa(feats).double()                                          # (N, H)
    gt0 = disc.mtrue[sd['pool']][sd['med'].to(disc.mtrue[sd['pool']].device)].double().to(kap.device)
    C = sd['c'][:, ::every].double().to(kap.device)                           # (N, S, n)
    rho = C * feats['invV'].double()[None, None]
    rbar = 0.5 * (rho[..., feats['tail']] + rho[..., feats['head']])          # (N, S, H)
    lh = kap[:, None].expand_as(rbar).clone()
    if getattr(kin, 'mode', '') == 'material_state':
        lh = lh + kin.state_branch(torch.log(rbar.clamp_min(kin.RHO_FLOOR)))
    gt = gt0[:, None] / (1 + rbar / RHO_S) if crowd else gt0[:, None].expand_as(rbar)
    gh, gt = torch.exp(lh).cpu().numpy().ravel(), gt.cpu().numpy().ravel()
    eps = (0.5 * (feats['ein'][..., 0] + 1)).double()[:, None].expand_as(rbar).cpu().numpy().ravel()
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
            disc = data['discs'][sd['disc']]
            sd['drive']['mtrue'] = disc.mtrue[sd['pool']][sd['med'].to(sd['c'].device)]


@torch.no_grad()
def eval_model(name, model, data, log, device, sets=TEST_SETS, chunk=64, snap=None, state_k=False):
    """roll the model out on every test set (in chunks of `chunk` trajectories) and return the metrics per set.
    The model is rebuilt on the 64 x 64 grid and the point cloud with the same parameters.  snap (a dict) receives the
    first four final states of every set.  state_k: the learned-law comparison at the reference states (Neural Charts;
    the reported values come from law_metrics, which runs on the CPU)."""
    snap = {} if snap is None else snap
    res = {}
    meta = data['meta']
    for s in [k for k in sets if k in data]:
        sd = data[s]; Y = sd['c']
        disc = data['discs'][sd['disc']]
        m_ = rebind(name, model, meta, disc, device) if sd['disc'] != 'g32' else model
        if m_ is not None and m_ is not model:               # a rebuilt model: same dtype as the trained one
            pd = {p.dtype for p in model.parameters()}
            if len(pd) == 1:
                m_ = m_.to(pd.pop())
        if m_ is not None:
            fast.copy_config(model, m_)
        if m_ is None:
            res[s] = None
            log(f'[{s}] N/A on {sd["disc"]}')
            continue
        t0 = time.time()
        Rs, Js, nsubs = [], [], []
        acc = PhysAcc()
        own = hasattr(m_, 'flux')
        for i in range(0, Y.shape[0], chunk):
            dr = {k: v[i:i + chunk] for k, v in sd['drive'].items()}
            R, ns, _ = rollout(m_, dr, Y[i:i + chunk, 0], Y.shape[1] - 1, DT)
            Rs.append(R.cpu()); nsubs.append(ns)
            ctx = m_.prepare(dr) if own else None
            two = own and hasattr(m_, 'rates') and not getattr(m_, 'free_flux', False)    # forward / backward rates a, b
            phys_metrics_rollout(R, dr, disc, (lambda c: m_.flux(c, ctx)) if own else None, acc,
                                 rates_fn=(lambda c: m_.rates(c, ctx)) if two else None)
            if i == 0 and s in ('drive_x8', 'temperature_x0.25', 'training_conditions'):
                j = 0; t = 100
                Jm = m_.flux(R[:, t], ctx)[j] if own else disc.minnorm_flux((R[:, t + 1] - R[:, t]) / DT)[j]
                snap[f'sig_{s}'] = local_sigma(R[j, t], Jm, {k: v[j] for k, v in dr.items() if torch.is_tensor(v)}, disc).cpu().numpy().astype(np.float32)
                snap[f'rho_{s}_t100'] = R[j, t].cpu().numpy().astype(np.float32)
            if hasattr(m_, 'flux') and sd['J_end'] is not None:
                Js.append(m_.flux(R[:, -1], dr).cpu() if torch.isfinite(R[:, -1]).all() and R[:, -1].min() > 0 else torch.full((R.shape[0], disc.H), float('nan')))
        R = torch.cat(Rs)
        m = rollout_metrics(R, Y.cpu(), disc.g['dens'])
        m['infer_time_s'] = time.time() - t0; m['substeps_per_DT'] = float(np.mean(nsubs))
        m['phys'] = dict(acc.result(), flux='own' if own else 'min-norm')
        if Js:
            m.update(flux_metrics(torch.cat(Js), sd['J_end']))
        if isinstance(m_, NeuralChart) and m_.config.kappa == 'learned':
            km, kd = k_metrics(m_, sd, disc)
            m['k'] = km
            if state_k and s in K_SETS:
                m['k_state'], kd = k_metrics_state(m_, sd, disc, bool(meta.get('crowd')))
                snap.update({f'k_{s}_{k}': v for k, v in kd.items()})
        snap[f'R_{s}'] = R[:4, -1].numpy().astype(np.float32)
        res[s] = m
        log(f'[{s}] T {m["err_T"]:.4g} 10T {m["err_10T"]:.4g} blown {m["n_blown"]} neg {m["n_neg_traj"]} mass {m["mass_drift_max"]:.1e}'
            + (f' | J {m["J_relerr"]:.3g}' if 'J_relerr' in m else '')
            + (f' | k r(log) {m["k"]["pearson_logk"]:.3f} rho_s {m["k"]["spearman_logk"]:.3f} r(k) {m["k"]["pearson_k"]:.3f} relerr {m["k"]["relerr_k"]:.3f}' if 'k' in m else '')
            + f' | LDB viol {m["phys"]["ldb_violation_frac"]:.3g} sigma<0 {m["phys"]["sigma_neg_frac"]:.3g}'
            + f' | sub {m["substeps_per_DT"]:.1f} t {m["infer_time_s"]:.0f}s')
    return res




@torch.no_grad()
def law_metrics(variant, model_name, checkpoint, data_dir=None):
    """the learned mobility of a Neural Chart against the true mobility at the reference states (the learned-law
    table), on the CPU: {set: metrics}"""
    from .data import load, add_norm
    from .models import build
    data = load(variant, 'cpu', sets=['train'] + list(K_SETS), data_dir=data_dir)
    meta = add_norm(data['meta'], data)
    base = build(model_name, meta, data['discs']['g32'])
    base.load_state_dict(torch.load(checkpoint, map_location='cpu'))
    base.eval()
    out = {}
    for s in K_SETS:
        if s not in data:
            continue
        disc = data['discs'][data[s]['disc']]
        m_ = rebind(model_name, base, meta, disc, 'cpu') if data[s]['disc'] != 'g32' else base
        out[s], _ = k_metrics_state(m_, data[s], disc, bool(meta.get('crowd')))
    return out
