"""E5 (identifiability): cycle observables and the linear-algebra check.
Observables (rows of W, J -> W J):
  global   : r(t) = sum_v J_R1(v,t)        (global ATP turnover = net circulation of the R1-R2 cycle)
  local4   : J_R1(v_i,t) at 4 random sites (seeded by the run seed)
  globalR2 : sum_v J_R2(v,t)                (other edge of the same cycle)
python obs_hyper.py  -> runs/E5_linalg.json (+ printout)"""
import os, sys, json
import numpy as np
import torch
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from hyperme import build_graph_np, NS, RUNS

_G = build_graph_np(8)


def obs_rows(kind, seed=0, g=_G):
    H, V = g['H'], g['V']
    if kind == 'global':
        W = np.zeros((1, H)); W[0, g['r1']] = 1; sites = None
    elif kind == 'globalR2':
        W = np.zeros((1, H)); W[0, g['r2']] = 1; sites = None
    elif kind == 'local4':
        sites = sorted(np.random.default_rng(777 + seed).choice(V, 4, replace=False).tolist())
        W = np.zeros((4, H)); W[np.arange(4), g['r1'][sites]] = 1
    else:
        raise ValueError(kind)
    return W, sites


def obs_matrix(kind, seed, dev):
    W, sites = obs_rows(kind, seed)
    return torch.tensor(W, device=dev), sites


def true_flux_np(c, U, k, A, g=_G):
    """c (..., NS, V), U (V,), k, A (H,) -> J (..., H)"""
    Us = np.concatenate([U, U, np.zeros(g['V'])])
    m = np.log(c.reshape(*c.shape[:-2], -1)) + Us
    return k * np.exp(m @ g['Np'].T + 0.5 * A) - k * np.exp(m @ g['Nm'].T - 0.5 * A)


def true_flux(sd, dev):
    c, U, k, A = sd['c'].numpy(), sd['U'].numpy(), sd['k'].numpy(), sd['A'].numpy()
    J = np.stack([true_flux_np(c[b], U[b], k[b], A[b]) for b in range(len(c))])
    return torch.tensor(J, device=dev)


RTOL = float(os.environ.get('E5_RTOL', '1e-9'))


def rank(M, rtol=None):
    rtol = RTOL if rtol is None else rtol
    s = np.linalg.svd(M, compute_uv=False)
    return int((s > rtol * s[0]).sum()) if s.size and s[0] > 0 else 0


if __name__ == '__main__':
    g = _G; S = g['S']; H, V = g['H'], g['V']
    out = {}
    rS = rank(S); out['H'] = H; out['n_states'] = S.shape[0]; out['rank_S'] = rS
    out['dim_kerS'] = H - rS; out['dim_kerST'] = S.shape[0] - rS
    # cycle-space decomposition
    Z = g['Z']
    loc = g['zloc'].T                                                     # 64 local R1+R2 loops
    def tr_cycles(sl):
        M = np.zeros((H, len(sl))); M[sl, np.arange(len(sl))] = 1
        Sx = S @ M
        N = np.linalg.svd(Sx)[2][rank(Sx):].T                             # ker of S restricted to these edges
        return M @ N
    cx, cxp = tr_cycles(g['tx']), tr_cycles(g['txp'])
    out['cycles_local_R1R2'] = rank(loc); out['cycles_transport_X'] = cx.shape[1]; out['cycles_transport_Xp'] = cxp.shape[1]
    out['cycles_local+transport'] = rank(np.concatenate([loc, cx, cxp], 1))
    out['cycles_mixed (transport-reaction)'] = out['dim_kerS'] - out['cycles_local+transport']
    # J-level: invisible flux subspace = ker [S; W]   (concentrations determine S J only)
    Jlev = {}
    for name, W in [('none', np.zeros((0, H))), ('global', obs_rows('global')[0]), ('globalR2', obs_rows('globalR2')[0]),
                    ('global+globalR2', np.concatenate([obs_rows('global')[0], obs_rows('globalR2')[0]])),
                    ('local4', obs_rows('local4', 0)[0]),
                    ('all 64 local J_R1', np.eye(H)[g['r1']])]:
        M = np.concatenate([S, W])
        Jlev[name] = dict(n_obs=W.shape[0], rank=rank(M), invisible_flux_dims=H - rank(M), gain=rank(M) - rS)
    out['flux_level'] = Jlev
    # w_R1 - w_R2 in im S^T ? (=> the two global observables are equivalent modulo observed S J)
    wd = obs_rows('global')[0][0] - obs_rows('globalR2')[0][0]
    res = wd - S.T @ np.linalg.lstsq(S.T, wd, rcond=None)[0]
    out['|w_R1 - w_R2 - proj_imST|'] = float(np.linalg.norm(res))
    # A-level: linearisation at the true state of test_id trajectory 0 (A = Z beta, k and mu known)
    data = torch.load(os.path.join(RUNS, 'data', 'crn.pt'))
    sd = data['test_id']; c = sd['c'][0].numpy(); U = sd['U'][0].numpy(); k = sd['k'][0].numpy(); A = sd['A'][0].numpy()
    Us = np.concatenate([U, U, np.zeros(V)])
    def ab(ct):
        m = np.log(ct.ravel()) + Us
        return k * np.exp(m @ g['Np'].T + 0.5 * A), k * np.exp(m @ g['Nm'].T - 0.5 * A)
    Alev = {}
    for name, W in [('none', np.zeros((0, H))), ('global', obs_rows('global')[0]), ('globalR2', obs_rows('globalR2')[0]),
                    ('global+globalR2', np.concatenate([obs_rows('global')[0], obs_rows('globalR2')[0]])), ('local4', obs_rows('local4', 0)[0])]:
        O = np.concatenate([S, W])
        d = {}
        # (i) one snapshot (NESS, t = 10T), k known: identifiable beta directions = rank(O D Z)
        a, b = ab(c[-1]); D = np.diag(0.5 * (a + b))
        d['snapshot_kknown_identifiable'] = rank(O @ D @ Z)
        # (ii) one snapshot, k unknown too (log k per hyperedge): rank of O [diag(J) | D Z] restricted to the beta block
        Jd = np.diag(a - b)
        Mk = O @ Jd; Mb = O @ D @ Z
        d['snapshot_joint_identifiable_beta'] = rank(np.concatenate([Mk, Mb], 1)) - rank(Mk)
        # (iii) whole trajectory (201 frames stacked), k unknown: identifiable beta directions beyond what log k absorbs
        Mks, Mbs = [], []
        for t in range(0, c.shape[0], 5):
            a, b = ab(c[t]); Mks.append(O @ np.diag(a - b)); Mbs.append(O @ np.diag(0.5 * (a + b)) @ Z)
        Mks, Mbs = np.concatenate(Mks), np.concatenate(Mbs)
        d['trajectory_joint_identifiable_beta'] = rank(np.concatenate([Mks, Mbs], 1)) - rank(Mks)
        d['trajectory_joint_rank_total'] = rank(np.concatenate([Mks, Mbs], 1))
        Alev[name] = d
    out['A_level'] = Alev
    out['note'] = ('flux_level: concentration data fix S J only; ker S (cycle space) components of J are invisible. '
                   'A_level: Jacobian ranks of the observation map w.r.t. beta (A = Z beta, dim 257) at the true state; '
                   '"joint" = per-hyperedge log k also unknown, counting beta directions not absorbable by k.')
    print(json.dumps(out, indent=1))
    json.dump(out, open(os.path.join(RUNS, 'E5_linalg.json' if RTOL == 1e-9 else f'E5_linalg_rtol{RTOL:g}.json'), 'w'), indent=1)
