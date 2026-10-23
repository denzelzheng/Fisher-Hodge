import json, numpy as np
import matplotlib; matplotlib.use('Agg'); import matplotlib.pyplot as plt
R = 'runs/'
md = []
# ---------------- E1 ----------------
S = json.load(open(R+'e1_summary.json')); C = json.load(open(R+'e1_curves.json'))
md.append('## E1: Pythagorean decomposition (L-weighted vs Euclidean)\n')
md.append('10 random seeds per row; t in [0,10], 201 time points. resL = max_t |sigma-ex_L-hk_L|/sigma; resE = |sigma-ex_E-hk_E|/sigma, where ex_E, hk_E are the L-norm energies of the Euclidean-Hodge parts (cross term / sigma). ss_* = at the exact stationary state (energy / sigma_ss).\n')
md.append('| graph | n | eps | sigma_ss | max resL | median_t resE (mean over seeds) | max_t resE | ss ex_L/sigma | ss ex_E/sigma | ss hk_E/sigma | ss <J,g_E>/sigma | CV(ell) | decay rate/(2*gap) |')
md.append('|---|---|---|---|---|---|---|---|---|---|---|---|---|')
for r in S['table']:
    a = r['agg']; f = lambda k, fmt='%.2e': ('n/a' if a[k]['mean'] is None else fmt % a[k]['mean'])
    mx = lambda k: ('n/a' if a[k]['max'] is None else '%.1e' % a[k]['max'])
    md.append('| %s | %d | %g | %s | %s | %s | %s | %s | %s | %s | %s | %s | %s |' % (
        r['kind'], r['n'], r['eps'], f('sigma_ss', '%.3g'), mx('resL_max'), f('resE_median'), mx('resE_max'),
        mx('ss_exL_frac'), f('ss_exE_frac', '%.3f'), f('ss_hkE_frac', '%.3f'), mx('ss_pairG_E_frac'), f('cvell_ss', '%.2f'),
        f('decay_over_2gap', '%.2f') if 'decay_over_2gap' in a else 'n/a'))
md.append('\nHeterogeneity sweep (ER n=20, eps=2, B,U scaled by h; 10 seeds): \n')
md.append('| h | CV(ell_ss) | ss ex_E/sigma (L-energy) | Euclid-norm grad fraction |g_E|^2/|f|^2 |')
md.append('|---|---|---|---|')
for h in S['het']: md.append('| %g | %.2f | %.3f | %.3f |' % (h['h'], h['cvell'], h['exE_frac'], h['euc_g_frac']))
# correlation across all samples
xs = []; ys = []; es = []
for r in S['table']:
    for s in r['raw']:
        if np.isfinite(s['ss_exE_frac']): xs.append(s['cvell_ss']); ys.append(s['ss_exE_frac']); es.append(r['eps'])
xs = np.array(xs); ys = np.array(ys); es = np.array(es)
md.append('\nAcross all %d samples with eps>0: Spearman-like corr(CV(ell), ex_E/sigma)=%.2f (Pearson on logs); ' % (len(xs), np.corrcoef(np.log(xs), np.log(ys))[0, 1]))
# absolute
md.append('\nHatano-Sasa check: max_t |hk_HS + ex_HS - sigma|/sigma_max over all: %.1e; geometric hk_L differs from Hatano-Sasa hk at transient: mean over rows of max_t|hk_L-hk_HS|/sigma_max = %.3f\n' % (
    max(r['agg']['hk_sum_res']['max'] for r in S['table']), np.mean([r['agg']['hk_L_vs_HS_rel']['mean'] for r in S['table'] if r['eps'] > 0])))
md.append('Convergence of hk_L(t=10) to sigma_ss: max rel err over all seeds with eps>0 = %.1e\n' % max(r['agg']['hk_conv_err']['max'] for r in S['table'] if r['eps'] > 0))
# figure 1
fig, ax = plt.subplots(2, 4, figsize=(18, 8))
for k, eps in enumerate(['0.5', '1', '2', '4']):
    c = C[eps]; t = np.array(c['ts']); ss = c['sigma_ss']
    a = ax[0, k]
    a.plot(t, c['sigma'], 'k', label='sigma(t)')
    a.plot(t, c['exL'], 'C0', label='excess (L)'); a.plot(t, c['hkL'], 'C3', label='housekeeping (L)')
    a.plot(t, c['exE'], 'C0--', label='grad energy (Eucl)'); a.plot(t, c['hkE'], 'C3--', label='cyc energy (Eucl)')
    a.plot(t, np.array(c['exE'])+np.array(c['hkE']), 'C2:', label='Eucl sum (!= sigma)')
    a.axhline(ss, color='gray', lw=.7); a.set_title('eps=%s (ER n=20)' % eps); a.set_xlabel('t')
    if k == 0: a.legend(fontsize=7); a.set_ylabel('energy')
    b = ax[1, k]
    b.semilogy(t, np.maximum(np.array(c['exL'])/ss, 1e-18), 'C0', label='ex_L/sigma_ss')
    b.semilogy(t, np.maximum(np.array(c['exE'])/ss, 1e-18), 'C0--', label='ex_E/sigma_ss (-> const>0)')
    b.semilogy(t, np.abs(np.array(c['hkL'])-ss)/ss+1e-18, 'C3', label='|hk_L - sigma_ss|/sigma_ss')
    b.semilogy(t, np.array(c['exL'])[0]/ss*np.exp(-2*c['gap']*t), 'k:', label='exp(-2 gap t)')
    b.set_ylim(1e-16, 1e3); b.set_xlabel('t')
    if k == 0: b.legend(fontsize=7)
plt.tight_layout(); plt.savefig(R+'e1_curves.png', dpi=110); plt.close()
# figure 2
fig, ax = plt.subplots(1, 3, figsize=(16, 4.5))
for kind, n, m in [('er', 20, 'o'), ('complete', 20, 's'), ('er', 30, '^'), ('complete', 30, 'v')]:
    rows = [r for r in S['table'] if r['kind'] == kind and r['n'] == n and r['eps'] > 0]
    ax[0].semilogy([r['eps'] for r in rows], [r['agg']['resE_median']['mean'] for r in rows], m+'-', label='%s n=%d' % (kind, n))
    ax[1].plot([r['eps'] for r in rows], [r['agg']['ss_exE_frac']['mean'] for r in rows], m+'-', label='%s n=%d' % (kind, n))
ax[0].set_xlabel('eps'); ax[0].set_ylabel('median_t |sigma - ex_E - hk_E|/sigma  (Eucl, L1: ~1e-15)'); ax[0].legend(); ax[0].set_title('pseudo-Pythagorean residual')
ax[1].set_xlabel('eps'); ax[1].set_ylabel('stationary ex_E / sigma_ss  (L-weighted: 0)'); ax[1].set_title('spurious gradient energy at NESS')
sc = ax[2].scatter(xs, ys, c=es, cmap='viridis', s=14); plt.colorbar(sc, ax=ax[2], label='eps')
ax[2].set_xscale('log'); ax[2].set_yscale('log'); ax[2].set_xlabel('CV(ell) at NESS'); ax[2].set_ylabel('ex_E/sigma_ss'); ax[2].set_title('vs heterogeneity of ell')
plt.tight_layout(); plt.savefig(R+'e1_euclid_failure.png', dpi=110); plt.close()

# ---------------- E2 ----------------
D = json.load(open(R+'e2.json'))
md.append('\n## E2: topology count\n')
md.append('rank(mod exact) = rank of Euclidean-projected (out of im d0) log(Q_ij/Q_ji) cochains for 35 random Q; "aug" = rank[C|d0]-rank d0 (independent check). Equivalence test: for 30 detailed-balance-constructed and 30 generic Q per graph, Kolmogorov (fundamental cycles, and ALL simple cycles of length<=6 where enumerated) <=> exact cochain (harmonic part=0) <=> stationary current J_ss=0.\n')
md.append('| graph | V | E | beta1=E-V+1 | (n-1)(n-2)/2 | rank mod exact | rank aug | #simple cycles checked | equivalence all OK | max harm (reversible) | min harm (irrev.) | max|J_ss| (rev.) | min|J_ss| (irrev.) |')
md.append('|---|---|---|---|---|---|---|---|---|---|---|---|---|')
for r in D['rows']:
    md.append('| %s | %d | %d | %d | %s | %d | %d | %s | %s | %.1e | %s | %.1e | %s |' % (
        r['graph'], r['V'], r['E'], r['beta1'], r['beta1_formula_complete'] if r['beta1_formula_complete'] is not None else '-',
        r['rank_mod_exact'], r['rank_aug'], r['n_simple_cycles'] if r['n_simple_cycles'] else '-', r['equiv_all_ok'],
        r['max_harm_if_rev'], '%.2g' % r['min_harm_if_irrev'] if r['min_harm_if_irrev'] is not None else '-',
        r['max_J_if_rev'], '%.2g' % r['min_J_if_irrev'] if r['min_J_if_irrev'] is not None else '-'))
md.append('\n5-cycle demo (random rates; even trials built reversible): log(prod fwd / prod bwd), harmonic-component norm, predicted |A|/sqrt5, stationary current, EPR\n')
md.append('| reversible-constructed | log(prod fwd/bwd) | harmonic norm | |A|/sqrt(5) | max|J_ss| | EPR |'); md.append('|---|---|---|---|---|---|')
for d in D['c5']:
    md.append('| %s | %.3g | %.3g | %.3g | %.2g | %.3g |' % (d['rev_construct'], d['log_ratio'], d['harmonic_norm'], d['pred_harm_norm_|A|/sqrt5'], d['J_ss_max'], d['epr']))

# ---------------- E3 ----------------
try:
    O = json.load(open(R+'e3.json'))
except Exception:
    O = []
md.append('\n## E3: finite-trajectory estimation\n')
fig, axs = plt.subplots(len(O), 3, figsize=(16, 4.3*max(len(O), 1)), squeeze=False)
for si, sysd in enumerate(O):
    tr = sysd['truth']; md.append('### System %s: n=%d, |E|=%d, beta1=%d, sigma_ss=%.4f, true ex_E=%.4f, true hk_E=%.4f (E-sum %.4f), CV(ell)=%.2f\n' % (
        sysd['name'], tr['n'], tr['nE'], tr['beta1'], tr['sigma'], tr['exE'], tr['hkE'], tr['exE']+tr['hkE'], tr['cv_ell']))
    md.append('Mean (std over 20 runs) of each estimator. Truth for stationary start: sigma=%.4f; excess(L)=0; housekeeping(L)=sigma; Euclid targets: ex_E=%.4f, hk_E=%.4f.\n' % (tr['sigma'], tr['exE'], tr['hkE']))
    md.append('| jumps T | Lynn direct | plug-in (rho_hat=occupancy, = ex_L+hk_L) | plug-in (rho=stat(Qhat)) | hk_L | ex_L (true 0) | hk_E | ex_E | ex_E+hk_E (not sigma) |')
    md.append('|---|---|---|---|---|---|---|---|---|')
    keys = ['lynn', 'plug', 'plug_stat', 'hkL', 'exL', 'hkE', 'exE']
    stat = {}
    for T, lst in sysd['res'].items():
        v = {k: np.array([x[k] for x in lst]) for k in keys}; v['Esum'] = v['hkE']+v['exE']
        stat[T] = v
        md.append('| %s | %s |' % (T, ' | '.join('%.4f (%.4f)' % (v[k].mean(), v[k].std()) for k in keys+['Esum'])))
    md.append('\nBias / RMSE relative to sigma_ss (bias = mean-truth):\n')
    md.append('| T | Lynn bias | Lynn RMSE | plug bias | plug RMSE | plug_stat bias | plug_stat RMSE | hk_L bias | hk_L RMSE | ex_L bias(truth 0) | hk_E bias vs L-target sigma | hk_E bias vs E-pop target | ex_E bias vs L-target 0 | ex_E bias vs E-pop target |')
    md.append('|---|---|---|---|---|---|---|---|---|---|---|---|---|---|')
    bi = lambda x, t: (x.mean()-t)/tr['sigma']; rm = lambda x, t: np.sqrt(np.mean((x-t)**2))/tr['sigma']
    for T, v in stat.items():
        md.append('| %s | %.4f | %.4f | %.4f | %.4f | %.4f | %.4f | %.4f | %.4f | %.4f | %.4f | %.4f | %.4f | %.4f |' % (
            T, bi(v['lynn'], tr['sigma']), rm(v['lynn'], tr['sigma']), bi(v['plug'], tr['sigma']), rm(v['plug'], tr['sigma']),
            bi(v['plug_stat'], tr['sigma']), rm(v['plug_stat'], tr['sigma']), bi(v['hkL'], tr['sigma']), rm(v['hkL'], tr['sigma']),
            bi(v['exL'], 0), bi(v['hkE'], tr['sigma']), bi(v['hkE'], tr['hkE']), bi(v['exE'], 0), bi(v['exE'], tr['exE'])))
    # cycle affinity
    md.append('\nCycle affinity (fundamental cycles, %d of them): RMS error of Hodge-cyc line integral vs true sum of eps*A along cycle; per-edge cyclic-field error ||c_hat - c_true||_L2 (edgewise, relative to ||c_true||).\n' % tr['beta1'])
    md.append('| T | RMS aff err (L-cyc integral) | RMS aff err (Eucl-cyc integral) | max |L-integral - Eucl-integral| | mean sqrt(sum 1/n) heuristic scale | rel. edgewise error ||c_L-c_L*|| /||c_L*|| | rel. edgewise error ||c_E-c_E*||/||c_E*|| | affinity bias (mean signed) |')
    md.append('|---|---|---|---|---|---|---|---|')
    afft = np.array(tr['aff'])
    for T, lst in sysd['res'].items():
        aL = np.array([x['aff_L'] for x in lst]); aE = np.array([x['aff_E'] for x in lst])
        cLs = np.array([x['cL_vec'] for x in lst]); cEs = np.array([x['cE_vec'] for x in lst])
        eL = np.linalg.norm(cLs-np.array(tr['cL']), axis=1)/np.linalg.norm(tr['cL']); eE = np.linalg.norm(cEs-np.array(tr['cE']), axis=1)/np.linalg.norm(tr['cE'])
        md.append('| %s | %.4f | %.4f | %.1e | - | %.4f | %.4f | %.4f |' % (T, np.sqrt(np.mean((aL-afft)**2)), np.sqrt(np.mean((aE-afft)**2)), np.max(np.abs(aL-aE)), eL.mean(), eE.mean(), (aL-afft).mean()))
    Ts = [float(T) for T in stat]
    a = axs[si, 0]
    for k, lab in [('lynn', 'Lynn'), ('plug', 'plug-in (L total)'), ('plug_stat', 'plug-in stat(Qhat)'), ('hkL', 'hk_L'), ('hkE', 'hk_E')]:
        a.errorbar(Ts, [stat[T][k].mean()/tr['sigma'] for T in stat], [stat[T][k].std()/tr['sigma'] for T in stat], marker='o', capsize=3, label=lab)
    a.axhline(1, color='k', lw=.6); a.set_xscale('log'); a.set_title('%s: EPR estimates / sigma_ss' % sysd['name']); a.set_xlabel('jumps'); a.legend(fontsize=7)
    a = axs[si, 1]
    a.loglog(Ts, [max(stat[T]['exL'].mean(), 1e-12) for T in stat], 'o-', label='ex_L (truth 0)')
    a.loglog(Ts, [stat[T]['exE'].mean() for T in stat], 's-', label='ex_E (E-pop %.4f)' % tr['exE']); a.axhline(tr['exE'], color='C1', ls=':')
    a.set_title('excess estimates'); a.set_xlabel('jumps'); a.legend(fontsize=7)
    a = axs[si, 2]
    a.loglog(Ts, [np.sqrt(np.mean((np.array([x['aff_L'] for x in sysd['res'][T]])-afft)**2)) for T in stat], 'o-', label='cycle affinity RMS err')
    a.loglog(Ts, [np.sqrt(np.mean((stat[T]['lynn']-tr['sigma'])**2)) for T in stat], 's-', label='Lynn EPR RMSE')
    a.loglog(Ts, [np.sqrt(np.mean((stat[T]['plug_stat']-tr['sigma'])**2)) for T in stat], '^-', label='plug stat RMSE')
    a.loglog(Ts, [np.sqrt(np.mean((stat[T]['hkE']-tr['sigma'])**2)) for T in stat], 'v-', label='hk_E RMSE vs sigma')
    a.set_title('errors vs T'); a.set_xlabel('jumps'); a.legend(fontsize=7)
plt.tight_layout(); plt.savefig(R+'e3_estimation.png', dpi=110); plt.close()
open(R+'summary_tables.md', 'w').write('\n'.join(md))
print('\n'.join(md))
