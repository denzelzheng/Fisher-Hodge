import numpy as np, json, scipy.linalg as sla, sys
from hodge_lib import *
OUT = 'runs/'
rng_master = 12345

def sample_system(kind, n, eps, seed, h=1.0):
    rng = np.random.default_rng(seed)
    adj = make_adj(kind, n, rng, 0.3); E = edges_of(adj); d0 = incidence(E, n)
    B = h*rng.uniform(0, 2, len(E)); U = h*rng.standard_normal(n)
    Z = cycle_basis_mat(d0)
    A = Z @ rng.standard_normal(Z.shape[1]); A /= np.sqrt(np.mean(A**2))
    Q = build_Q(E, n, B, U, A, eps)
    rho0 = rng.dirichlet(np.ones(n))
    return E, d0, Q, rho0, A

def analyse(rho, Q, E, d0):
    a, b, J, f, ell = edge_quantities(rho, Q, E)
    sigma = float(np.sum(J*f))
    gL, cL = hodge(f, ell, d0)
    gE, cE = hodge(f, np.ones_like(ell), d0)
    r = dict(sigma=sigma, exL=energy(gL, ell), hkL=energy(cL, ell),
             exE=energy(gE, ell), hkE=energy(cE, ell),
             crossE=2*float(np.sum(ell*gE*cE)), crossL=2*float(np.sum(ell*gL*cL)),
             # J-pairing split (always sums to sigma)
             pairG_E=float(np.sum(J*gE)), pairC_E=float(np.sum(J*cE)),
             # plain Euclid energies
             euc_g=float(np.sum(gE**2)), euc_c=float(np.sum(cE**2)), euc_f=float(np.sum(f**2)),
             cvell=float(np.std(ell)/np.mean(ell)))
    return r, (J, f, ell)

def hatano_sasa(rho, pi, Q, E):
    a, b, J, f, ell = edge_quantities(rho, Q, E)
    _, _, _, fh, _ = edge_quantities(pi, Q, E)
    return float(np.sum(J*fh)), float(np.sum(J*(f-fh)))

def run_one(kind, n, eps, seed, h=1.0, tmax=10.0, nt=201, keep=False):
    E, d0, Q, rho0, A = sample_system(kind, n, eps, seed, h)
    pi = stationary(Q)
    ts = np.linspace(0, tmax, nt); P = sla.expm(Q*(ts[1]-ts[0]))
    rho = rho0.copy(); rows = []; hs = []
    for t in ts:
        r, _ = analyse(rho, Q, E, d0); rows.append(r); hs.append(hatano_sasa(rho, pi, Q, E))
        rho = rho @ P; rho = np.abs(rho); rho /= rho.sum()
    ss, _ = analyse(pi, Q, E, d0)
    ev = np.linalg.eigvals(Q); ev = ev[np.argsort(-ev.real)]
    gap = -ev[1].real
    cur = {k: np.array([r[k] for r in rows]) for k in rows[0]}
    cur['hs_hk'] = np.array([x[0] for x in hs]); cur['hs_ex'] = np.array([x[1] for x in hs])
    return dict(ts=ts, cur=cur, ss=ss, gap=gap, E=E, n=n, nE=len(E)), ev

def summarize(res):
    c = res['cur']; sig = c['sigma']; mask = sig > 1e-9*sig.max()
    resL = np.abs(sig-c['exL']-c['hkL'])[mask]/sig[mask]
    resE = np.abs(sig-c['exE']-c['hkE'])[mask]/sig[mask]
    ss = res['ss']
    out = dict(resL_max=float(resL.max()) if mask.any() else 0.0,
               resE_max=float(resE.max()) if mask.any() else 0.0,
               resE_median=float(np.median(resE)) if mask.any() else 0.0,
               sigma_ss=ss['sigma'], gap=res['gap'], cvell_ss=ss['cvell'],
               ss_exL_frac=ss['exL']/ss['sigma'] if ss['sigma'] > 1e-12 else float('nan'),
               ss_exE_frac=ss['exE']/ss['sigma'] if ss['sigma'] > 1e-12 else float('nan'),
               ss_hkE_frac=ss['hkE']/ss['sigma'] if ss['sigma'] > 1e-12 else float('nan'),
               ss_pairG_E_frac=ss['pairG_E']/ss['sigma'] if ss['sigma'] > 1e-12 else float('nan'),
               ss_euc_g_frac=ss['euc_g']/ss['euc_f'] if ss['euc_f'] > 1e-12 else float('nan'),
               hk_conv_err=float(abs(c['hkL'][-1]-ss['sigma'])/max(ss['sigma'], 1e-30)) if ss['sigma']>1e-12 else float(abs(c['hkL'][-1])),
               hs_hk_end=float(c['hs_hk'][-1]), hk_sum_res=float(np.max(np.abs(c['hs_hk']+c['hs_ex']-sig))/sig.max()),
               hk_L_vs_HS_rel=float(np.max(np.abs(c['hkL']-c['hs_hk']))/max(sig.max(),1e-30)))
    # exponential decay rate of excess_L: fit log exL on window where exL/sigma_ss in [1e-9,1e-2]
    if ss['sigma'] > 1e-8:
        x = c['exL']/ss['sigma']; ts = res['ts']
        w = (x > 1e-9) & (x < 1e-2)
        # take the first contiguous late window
        idx = np.nonzero(w)[0]
        if len(idx) >= 5:
            slope = np.polyfit(ts[idx], np.log(x[idx]), 1)[0]
            out['decay_rate_fit'] = float(-slope); out['decay_over_2gap'] = float(-slope/(2*res['gap']))
    return out

if __name__ == '__main__':
    configs = [('er', 20), ('complete', 20), ('er', 30), ('complete', 30)]
    epss = [0, 0.5, 1, 2, 4]; nseed = 10
    table = []; keepcur = {}
    for kind, n in configs:
        for eps in epss:
            sums = []
            for s in range(nseed):
                res, ev = run_one(kind, n, eps, 1000*s+n+7)
                sums.append(summarize(res))
                if s == 0 and kind == 'er' and n == 20:
                    keepcur[str(eps)] = {k: v.tolist() for k, v in res['cur'].items() if k in
                        ('sigma','exL','hkL','exE','hkE','hs_hk','hs_ex')}
                    keepcur[str(eps)]['ts'] = res['ts'].tolist(); keepcur[str(eps)]['sigma_ss'] = res['ss']['sigma']
                    keepcur[str(eps)]['gap'] = res['gap']; keepcur[str(eps)]['ev'] = [[float(z.real), float(z.imag)] for z in ev[:6]]
            keys = sums[0].keys()
            agg = {}
            for k in keys:
                v = np.array([x.get(k, np.nan) for x in sums], float)
                agg[k] = dict(mean=float(np.nanmean(v)) if np.isfinite(v).any() else None,
                              min=float(np.nanmin(v)) if np.isfinite(v).any() else None,
                              max=float(np.nanmax(v)) if np.isfinite(v).any() else None)
            table.append(dict(kind=kind, n=n, eps=eps, nseed=nseed, agg=agg, raw=sums))
            print(kind, n, eps, 'resL_max', agg['resL_max']['max'], 'resE_med', agg['resE_median']['mean'],
                  'ss_exE_frac', agg['ss_exE_frac']['mean'], 'ss_exL', agg['ss_exL_frac']['max'], flush=True)
    # heterogeneity sweep (er, n=20, eps=2)
    het = []
    for h in [0.0, 0.25, 0.5, 1.0, 2.0]:
        sums = [summarize(run_one('er', 20, 2.0, 1000*s+27, h=h, nt=21)[0]) for s in range(10)]
        het.append(dict(h=h, cvell=float(np.mean([x['cvell_ss'] for x in sums])),
                        exE_frac=float(np.mean([x['ss_exE_frac'] for x in sums])),
                        euc_g_frac=float(np.mean([x['ss_euc_g_frac'] for x in sums])),
                        pairG_E=float(np.mean([x['ss_pairG_E_frac'] for x in sums]))))
        print('het', het[-1], flush=True)
    json.dump(dict(table=table, het=het), open(OUT+'e1_summary.json', 'w'))
    json.dump(keepcur, open(OUT+'e1_curves.json', 'w'))
