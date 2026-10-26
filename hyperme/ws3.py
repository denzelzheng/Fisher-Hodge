"""WS3: do the learned rates define the right stochastic process?

Sample the chemical master equation of the true model, the learned HyperME and the baselines, all from the same
state and drive, and compare their fluctuations.
  C1 (phosphorylation cycle, 8 x 8): truth, HyperME, B1 CRNN, B3 HyperME with A = 0.
     exact Gillespie at Omega = 50 (with the medium entropy of every jump under the sampling rates and under the true
     rates on the same path), Poisson tau-leaping at Omega = 50, 200, 1000.
  C3 (Brusselator, 8 x 8, Omega = 200): truth, HyperME trained with MPRK22 x 8, B1 CRNN; tau-leaping; period
     statistics of the homogeneous limit cycle.
    python -m hyperme.ws3 --stage c1_ssa --models truth,crnn      (one process per model group)
    python -m hyperme.ws3 --stage c1_tau | c3 | analyze
Outputs go to runs/hyperme/ws3/."""
import argparse, json, os, time
import numpy as np
import torch
from . import presets, baselines as BL, stochastic as St
from .model import logmean

OUT = os.path.join(presets.ROOT, 'runs', 'hyperme', 'ws3')
C1_MAIN = os.path.join(presets.ROOT, 'hyper', 'runs', 'main')
C23 = os.path.join(presets.ROOT, 'hyper', 'runs', 'C23')


# ------------------------------------------------------------------------------------------ rate functions
def true_rates(hg, k, A, Us):
    """mass action with mu = log c + Us: a = k exp(<nu+, Us> + A/2) c^{nu+}, b = k exp(<nu-, Us> - A/2) c^{nu-}."""
    k, A, Us = (torch.as_tensor(x, dtype=torch.float64).reshape(1, -1) for x in (k, A, Us))
    pa = k * torch.exp(hg.dot(Us, '+') + 0.5 * A)
    pb = k * torch.exp(hg.dot(Us, '-') - 0.5 * A)
    return lambda c: (pa * hg.mass_action(c, '+'), pb * hg.mass_action(c, '-'))


def model_rates(model, drive, B):
    ctx = model.prepare(drive).expand(B)
    return lambda c: model.rates(c, ctx, zero_safe=True)


def c1_setup(B, traj=0):
    data = presets.load_c1()
    te = data['test_id']
    drive = {k: v[traj:traj + 1] for k, v in te['drive'].items()}
    hg_model = presets.c1()
    hg = hg_model.hg
    U = drive['U'][0].numpy()
    Us = np.concatenate([U, U, np.zeros_like(U)])
    truth = true_rates(hg, te['k'][traj].numpy(), te['A'][traj].numpy(), Us)
    c_start = te['c'][traj, -1]
    models = {'truth': lambda: truth}

    def hyperme():
        m = presets.load_legacy_c1(presets.c1(), torch.load(os.path.join(C1_MAIN, 'hyperme_s0.pt'), map_location='cpu')).eval()
        return model_rates(m, drive, B)

    def rev():
        sd = {k: v for k, v in torch.load(os.path.join(C1_MAIN, 'hyperme_rev_s0.pt'), map_location='cpu').items() if not k.startswith('amlp.')}
        m = presets.c1(use_A=False)
        mp = {'kmlp.': 'kinetics.mlp.', 'G.': 'free_energy.mlp.'}
        out = {}
        for k, v in sd.items():
            a = next((a for a in mp if k.startswith(a)), None)
            out[mp[a] + k[len(a):] if a else 'free_energy.alpha'] = v
        m.load_state_dict(out)
        return model_rates(m.eval(), drive, B)

    def crnn():
        with presets.default_dtype(torch.float64):
            m = BL.CRNN(hg, 8, presets.c1_featurizer(hg))
        m.load_state_dict(torch.load(os.path.join(C1_MAIN, 'crnn_s0.pt'), map_location='cpu'))
        ctx = m.prepare(drive)
        kf, kb = ctx['kf'], ctx['kb']
        return lambda c: (kf * hg.mass_action(c, '+'), kb * hg.mass_action(c, '-'))
    def hyperme_l4():                    # E5: HyperME trained with the R1 currents at 4 sites as an extra observable
        p = os.path.join(presets.ROOT, 'hyper', 'runs', 'E5_runs', 'hyperme_obs-local4_s0.pt')
        m = presets.load_legacy_c1(presets.c1(), torch.load(p, map_location='cpu')).eval()
        return model_rates(m, drive, B)
    models.update(hyperme=hyperme, rev=rev, crnn=crnn, hyperme_l4=hyperme_l4)
    meta = dict(dG=float(drive['dG'][0]), A_cycle_true=float(drive['dG'][0] + np.log(1.0 / 0.02)))
    return hg, models, truth, c_start, meta


def c3_setup(B, traj=0):
    data = presets.load_c3()
    te = data['test_id']
    drive = {k: v[traj:traj + 1] for k, v in te['drive'].items()}
    m0 = presets.c3()
    hg = m0.hg
    truth = true_rates(hg, te['k'][traj].numpy(), te['A'][traj].numpy(), np.zeros(hg.M))

    def hyperme():
        m = presets.c3()
        p = os.path.join(presets.ROOT, 'runs', 'hyperme_regression', 'c3_mprk228_s0.pt')
        if not os.path.exists(p):
            p = os.path.join(presets.ROOT, 'runs', 'hyperme', 'c3_mprk228_s0.pt')
        m.load_state_dict(torch.load(p, map_location='cpu'))
        return model_rates(m.eval(), drive, B)

    def crnn():
        with presets.default_dtype(torch.float64):
            m = BL.CRNN(hg, 8, presets.c3_featurizer(hg), oriented=False)
        m.load_state_dict(torch.load(os.path.join(C23, 'brus_crnn_s0.pt'), map_location='cpu'))
        ctx = m.prepare(drive)
        kf, kb = ctx['kf'], ctx['kb']
        return lambda c: (kf * hg.mass_action(c, '+'), kb * hg.mass_action(c, '-'))
    return hg, dict(truth=lambda: truth, hyperme=hyperme, crnn=crnn), truth, te['c'][traj, 0]


def c3wm_setup(B, traj=0):
    """well-mixed C3: one site, no transport, the same learned parameters and the same true rate constants."""
    data = presets.load_c3()
    te = data['test_id']
    drive = {k: v[traj:traj + 1] for k, v in te['drive'].items()}
    hg = presets.c3(well_mixed=True).hg
    Vg = 64
    kt = te['k'][traj].numpy()[[r * Vg for r in range(4)]]; At = te['A'][traj].numpy()[[r * Vg for r in range(4)]]
    truth = true_rates(hg, kt, At, np.zeros(hg.M))

    def hyperme():
        m = presets.c3(well_mixed=True)
        p = os.path.join(presets.ROOT, 'runs', 'hyperme_regression', 'c3_mprk228_s0.pt')
        if not os.path.exists(p):
            p = os.path.join(presets.ROOT, 'runs', 'hyperme', 'c3_mprk228_s0.pt')
        m.load_state_dict(torch.load(p, map_location='cpu'))
        return model_rates(m.eval(), drive, B)

    def crnn():
        with presets.default_dtype(torch.float64):
            m = BL.CRNN(hg, 8, presets.c3_featurizer(hg, ntyp=6), oriented=False)
        m.load_state_dict(torch.load(os.path.join(C23, 'brus_crnn_s0.pt'), map_location='cpu'))
        ctx = m.prepare(drive)
        kf, kb = ctx['kf'], ctx['kb']
        return lambda c: (kf * hg.mass_action(c, '+'), kb * hg.mass_action(c, '-'))
    c0 = te['c'][traj, 0][[0, Vg]]                          # homogeneous start: X, Y at any site
    return hg, dict(truth=lambda: truth, hyperme=hyperme, crnn=crnn), c0


def stage_c3wm(models_sel, B, t_end, t_burn, dt_rec, seed, omega=200):
    os.makedirs(OUT, exist_ok=True)
    hg, models, c0 = c3wm_setup(B)
    for name in models_sel:
        rates = models[name]()
        n0 = torch.round(omega * c0)[None].expand(B, -1).clone()
        t0 = time.time()
        res = St.gillespie(rates, hg, n0, omega, t_end, dt_rec, generator=torch.Generator().manual_seed(seed), t_burn=t_burn)
        out = dict(times=res['times'], X=res['n'][:, :, 0] / omega, Y=res['n'][:, :, 1] / omega, events=res['events'],
                   wall_s=time.time() - t0, t_burn=t_burn)
        torch.save(out, os.path.join(OUT, f'c3wm_ssa_{name}_om{omega}.pt'))
        print(f'c3wm {name}: {out["wall_s"]:.0f}s events {float(res["events"].float().mean()):.0f}', flush=True)


def cycle_stats(X, times, t_burn, smooth=0.5):
    """periods from upward crossings with hysteresis (enter above m + h only after being below m - h), on a moving
    average of width `smooth`; phase diffusion D = 2 pi^2 Var(T) / mean(T)^3."""
    t = times.numpy(); dt = t[1] - t[0]; w = max(1, int(round(smooth / dt)))
    per, amp = [], []
    for x in X.numpy():
        x = np.convolve(x, np.ones(w) / w, mode='same')
        sel = (t >= t_burn) & (t <= t[-1] - smooth)
        xs, ts = x[sel], t[sel]
        q10, q90 = np.quantile(xs, [0.1, 0.9]); m = 0.5 * (q10 + q90); h = 0.25 * (q90 - q10)
        armed, last = False, None
        for i in range(1, len(xs)):
            if xs[i] < m - h:
                armed = True
            if armed and xs[i - 1] < m + h <= xs[i]:
                tc = ts[i - 1] + (m + h - xs[i - 1]) / (xs[i] - xs[i - 1]) * dt
                if last is not None:
                    per.append(tc - last)
                last, armed = tc, False
        amp.append(q90 - q10)
    per = np.asarray(per)
    Tm, Tv = float(per.mean()), float(per.var())
    return dict(period_mean=Tm, period_std=float(np.sqrt(Tv)), period_cv=float(np.sqrt(Tv) / Tm), n_periods=len(per),
                phase_diffusion=float(2 * np.pi ** 2 * Tv / Tm ** 3), amplitude_q90_q10=float(np.median(amp)))


def keep(res):
    """drop what the analysis does not need, as float32 where precision allows."""
    out = {k: res[k] for k in ('times', 'Ia', 'Ib', 'omega')}
    out['n'] = res['n'].float()
    for k in ('Nf', 'Nb', 'S', 'Sref'):
        if k in res:
            out[k] = res[k]
    for k in ('events', 'retries'):
        if k in res:
            out[k] = res[k]
    return out


# ------------------------------------------------------------------------------------------ stages
def stage_c1(models_sel, method, omegas, B, t_end, t_burn, dt_rec, tau, seed):
    os.makedirs(OUT, exist_ok=True)
    hg, models, truth, c_start, meta = c1_setup(B)
    json.dump(meta, open(os.path.join(OUT, 'c1_meta.json'), 'w'))
    for name in models_sel:
        rates = models[name]()
        for om in omegas:
            g = torch.Generator().manual_seed(seed)
            n0 = torch.round(om * c_start)[None].expand(B, -1).clone()
            t0 = time.time()
            if method == 'ssa':
                res = St.gillespie(rates, hg, n0, om, t_end, dt_rec, generator=g, t_burn=t_burn,
                                   ref_rates=None if name == 'truth' else truth)
            else:
                res = St.tau_leap(rates, hg, n0, om, t_end, dt_rec, tau, generator=g, t_burn=t_burn,
                                  ref_rates=None if name == 'truth' else truth)
            res = keep(res); res['wall_s'] = time.time() - t0; res['t_burn'] = t_burn
            torch.save(res, os.path.join(OUT, f'c1_{method}_{name}_om{om}.pt'))
            print(f'c1 {method} {name} Omega {om}: {res["wall_s"]:.0f}s', ('events %.0f' % float(res['events'].float().mean())) if 'events' in res
                  else 'retries %d' % res['retries'], flush=True)


def stage_c3(models_sel, B, t_end, t_burn, dt_rec, tau, seed, omega=200):
    os.makedirs(OUT, exist_ok=True)
    hg, models, truth, c0 = c3_setup(B)
    for name in models_sel:
        rates = models[name]()
        g = torch.Generator().manual_seed(seed)
        n0 = torch.round(omega * c0)[None].expand(B, -1).clone()
        t0 = time.time()
        res = St.tau_leap(rates, hg, n0, omega, t_end, dt_rec, tau, generator=g, t_burn=t_burn)
        out = dict(times=res['times'], Xmean=res['n'][:, :, :hg.V].mean(-1) / omega, Ymean=res['n'][:, :, hg.V:].mean(-1) / omega,
                   retries=res['retries'], wall_s=time.time() - t0, t_burn=t_burn)
        torch.save(out, os.path.join(OUT, f'c3_tau_{name}_om{omega}.pt'))
        print(f'c3 {name}: {out["wall_s"]:.0f}s retries {out["retries"]}', flush=True)


# ------------------------------------------------------------------------------------------ analysis
def occupancy(res, V):
    sel = res['times'] >= res['t_burn']
    n = res['n'][:, sel, :2 * V].double()                     # X, Xp (E is immobile and conserved per site)
    m, v = n.mean((0, 1)), n.var((0, 1))
    return m.numpy(), v.numpy(), (v / m).numpy()


def channel_stats(res):
    times = res['times']; i0 = int(torch.nonzero(times >= res['t_burn'])[0]); T = float(times[-1] - times[i0])
    Q = (res['Nf'][:, -1] - res['Nb'][:, -1]) - (res['Nf'][:, i0] - res['Nb'][:, i0])
    a, b = res['Ia'].mean(0) / T, res['Ib'].mean(0) / T
    return dict(J=(Q.mean(0) / T).numpy(), var_rate=(Q.var(0) / T).numpy(), a=a.numpy(), b=b.numpy(), T=T)


def ft_slope(sig, bw=0.5, nmin=10):
    """slope (through the origin) of log P(S)/P(-S) vs S from a sample of S; bins of width bw centred on k*bw."""
    sig = np.asarray(sig)
    k = np.round(sig / bw).astype(int)
    kmax = k.max() if len(k) else 0
    xs, ys, ws = [], [], []
    for j in range(1, kmax + 1):
        np_, nm = (k == j).sum(), (k == -j).sum()
        if np_ >= nmin and nm >= nmin:
            xs.append(j * bw); ys.append(np.log(np_ / nm)); ws.append(1.0 / (1.0 / np_ + 1.0 / nm))
    if len(xs) < 2:
        return dict(slope=float('nan'), x=xs, y=ys, n=len(sig))
    xs, ys, ws = map(np.asarray, (xs, ys, ws))
    s = float((ws * xs * ys).sum() / (ws * xs * xs).sum())
    se = float(np.sqrt(1.0 / (ws * xs * xs).sum()))
    return dict(slope=s, se=se, x=xs.tolist(), y=ys.tolist(), n=len(sig), mean=float(sig.mean()))


def local_sigma(res, V, key, window, r1=0, r2=None):
    """Sigma of the local R1 + R2 loop at every site over non-overlapping windows after burn-in, pooled."""
    r2 = V if r2 is None else r2
    times = res['times']; i0 = int(torch.nonzero(times >= res['t_burn'])[0])
    dt = float(times[1] - times[0]); m = max(1, int(round(window / dt)))
    S = res[key][:, i0::m]
    loc = S[:, :, r1:r1 + V] + S[:, :, r2:r2 + V]                 # (B, W+1, V)
    d = (loc[:, 1:] - loc[:, :-1]).reshape(-1).numpy()
    return d[np.isfinite(d)]


def phase_stats(Xm, times, t_burn):
    """periods of the site-mean X oscillation from upward crossings of its running mean."""
    per, amp = [], []
    for x in Xm:
        x = x.numpy(); t = times.numpy()
        sel = t >= t_burn; x, t = x[sel], t[sel]
        xm = x.mean()
        up = np.nonzero((x[:-1] < xm) & (x[1:] >= xm))[0]
        tc = t[up] + (xm - x[up]) / (x[up + 1] - x[up]) * (t[up + 1] - t[up])
        per += list(np.diff(tc)); amp.append(x.max() - x.min())
    per = np.asarray(per)
    Tm, Tv = float(per.mean()), float(per.var())
    return dict(period_mean=Tm, period_std=float(np.sqrt(Tv)), period_cv=float(np.sqrt(Tv) / Tm), n_periods=len(per),
                phase_diffusion=float(2 * np.pi ** 2 * Tv / Tm ** 3), amplitude=float(np.median(amp)))


def analyze():
    meta = json.load(open(os.path.join(OUT, 'c1_meta.json')))
    V = 64
    hg = presets.c1().hg
    typ = hg.meta['typ']
    names = ['truth', 'hyperme', 'hyperme_l4', 'crnn', 'rev']
    out = dict(meta=meta, occupancy={}, channels={}, ft={}, c3={})
    for method, omegas in (('ssa', [50]), ('tau', [50, 200, 1000])):
        for om in omegas:
            ref = None
            for nm in names:
                p = os.path.join(OUT, f'c1_{method}_{nm}_om{om}.pt')
                if not os.path.exists(p):
                    continue
                res = torch.load(p)
                m_, v_, f_ = occupancy(res, V)
                cs = channel_stats(res)
                if nm == 'truth':
                    ref = (f_, cs)
                key = f'{method}_om{om}_{nm}'
                occ = dict(fano_median=float(np.median(f_)), fano_X=float(np.median(f_[:V])), fano_Xp=float(np.median(f_[V:])),
                           mean_median=float(np.median(m_)))
                ch = {}
                for t, lab in enumerate(['R1', 'R2', 'TX', 'TXp']):
                    sel = typ == t
                    a, b, J, vr = cs['a'][sel], cs['b'][sel], cs['J'][sel], cs['var_rate'][sel]
                    lam = logmean(torch.tensor(a), torch.tensor(b)).numpy()
                    d = dict(var_over_apb=float(np.median(vr / (a + b))), J_over_amb=float(np.median(J / (a - b))) if lab in ('R1', 'R2') else None,
                             twoLam_over_var=float(np.median(2 * lam / vr)), apb_median=float(np.median(a + b)),
                             J_median=float(np.median(J)), log_a_over_b=float(np.median(np.log(a / b))))
                    if ref is not None and nm != 'truth':
                        d['var_ratio_to_truth'] = float(np.median(vr / ref[1]['var_rate'][sel]))
                        d['apb_ratio_to_truth'] = float(np.median((a + b) / (ref[1]['a'][sel] + ref[1]['b'][sel])))
                        d['J_ratio_to_truth'] = float(np.median(J / ref[1]['J'][sel])) if lab in ('R1', 'R2') else None
                    ch[lab] = d
                if ref is not None and nm != 'truth':
                    occ['fano_relerr_median'] = float(np.median(np.abs(f_ - ref[0]) / ref[0]))
                    occ['fano_corr'] = float(np.corrcoef(f_, ref[0])[0, 1])
                    np.save(os.path.join(OUT, f'fano_{key}.npy'), np.stack([ref[0], f_]))
                if 'events' in res:
                    occ['events_per_traj'] = float(res['events'].float().mean())
                occ['wall_s'] = float(res['wall_s'])
                out['occupancy'][key] = occ; out['channels'][key] = ch
                if method == 'ssa':
                    ft = {}
                    for w in (0.05, 0.25):
                        for skey in ('S', 'Sref'):
                            if skey in res:
                                s = local_sigma(res, V, skey, w)
                                r = ft_slope(s)
                                ft[f'{skey}_w{w}'] = r
                                np.save(os.path.join(OUT, f'ftsig_{key}_{skey}_w{w}.npy'), s.astype(np.float32))
                    out['ft'][key] = ft
    out['c3_grid'] = {}
    for nm in ('truth', 'hyperme', 'crnn'):
        p = os.path.join(OUT, f'c3_tau_{nm}_om200.pt')
        if os.path.exists(p):                      # spatial C3: amplitude of the site-mean oscillation over time
            r = torch.load(p)
            t = r['times'].numpy(); X = r['Xmean'].numpy()
            out['c3_grid'][nm] = {f'ptp_{a}_{b}': float(np.median(np.ptp(X[:, (t >= a) & (t < b)], axis=1)))
                                  for a, b in ((0, 7), (7, 14), (14, 21), (21, 40), (40, 77))}
        p = os.path.join(OUT, f'c3wm_ssa_{nm}_om200.pt')
        if os.path.exists(p):                      # well-mixed C3: limit-cycle period statistics
            r = torch.load(p)
            out['c3'][nm] = cycle_stats(r['X'], r['times'], r['t_burn'])
            out['c3'][nm]['wall_s'] = float(r['wall_s'])
    json.dump(out, open(os.path.join(OUT, 'ws3.json'), 'w'), indent=1, default=float)
    print(json.dumps(out, indent=1, default=float)[:20000])


PAL = dict(surface='#fcfcfb', ink='#0b0b0b', ink2='#52514e', grid='#e4e3df', s1='#2a78d6', s2='#eb6834', s3='#1baf7a')


def _style(ax):
    ax.set_facecolor(PAL['surface'])
    for sp in ('top', 'right'):
        ax.spines[sp].set_visible(False)
    for sp in ('left', 'bottom'):
        ax.spines[sp].set_color(PAL['ink2']); ax.spines[sp].set_linewidth(0.8)
    ax.tick_params(colors=PAL['ink2'], labelsize=9)
    ax.grid(True, color=PAL['grid'], linewidth=0.6); ax.set_axisbelow(True)


def plot(assets):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    res = json.load(open(os.path.join(OUT, 'ws3.json')))
    w = 0.05
    # ---- fluctuation theorem: log P(S)/P(-S) vs S for the local R1 + R2 loop, exact SSA at Omega = 50
    fig, axs = plt.subplots(1, 4, figsize=(14, 3.7), sharey=True, facecolor=PAL['surface'])
    for ax, nm, title in zip(axs, ['truth', 'hyperme', 'hyperme_l4', 'crnn'],
                             ['Truth', 'HyperME', 'HyperME + 4 local currents', 'B1 CRNN']):
        _style(ax)
        ft = res['ft'][f'ssa_om50_{nm}']
        lim = 0
        for skey, col, lab in (('S', PAL['s1'], 'own rates'), ('Sref', PAL['s2'], 'true forces on this path')):
            r = ft.get(f'{skey}_w{w}')
            if not r or not r['x']:
                continue
            x, y = np.asarray(r['x']), np.asarray(r['y'])
            lim = max(lim, x.max())
            ax.scatter(x, y, s=22, color=col, edgecolor=PAL['surface'], linewidth=0.8, zorder=3,
                       label=f'{lab}, slope {r["slope"]:.2f}')
            xx = np.linspace(0, x.max(), 2)
            ax.plot(xx, r['slope'] * xx, color=col, linewidth=2, zorder=2)
        xx = np.linspace(0, lim, 2)
        ax.plot(xx, xx, color=PAL['ink2'], linewidth=1, linestyle=(0, (4, 3)), zorder=1, label='slope 1')
        ax.set_title(title, color=PAL['ink'], fontsize=11, loc='left')
        ax.set_xlabel(f'Σ of the local loop over {w} time units (nats)', color=PAL['ink2'], fontsize=9)
        ax.legend(frameon=False, fontsize=8, labelcolor=PAL['ink'], loc='upper left')
    axs[0].set_ylabel('log P(Σ) / P(−Σ)', color=PAL['ink2'], fontsize=9)
    fig.tight_layout()
    fig.savefig(os.path.join(assets, 'ws3_fluctuation_theorem.png'), dpi=160, facecolor=PAL['surface'])
    plt.close(fig)
    # ---- Fano factors of X and Xp at every site: model vs truth, tau-leaping at three system sizes
    fig, axs = plt.subplots(1, 3, figsize=(11, 3.6), facecolor=PAL['surface'])
    for ax, om in zip(axs, (50, 200, 1000)):
        _style(ax)
        lo, hi = np.inf, -np.inf
        for nm, col, lab in (('hyperme', PAL['s1'], 'HyperME'), ('crnn', PAL['s2'], 'B1 CRNN'), ('rev', PAL['s3'], 'B3 (A = 0)')):
            p = os.path.join(OUT, f'fano_tau_om{om}_{nm}.npy')
            if not os.path.exists(p):
                continue
            ft_, fm = np.load(p)
            lo, hi = min(lo, ft_.min(), fm.min()), max(hi, ft_.max(), fm.max())
            ax.scatter(ft_, fm, s=14, color=col, edgecolor=PAL['surface'], linewidth=0.5, alpha=0.9, label=lab, zorder=3)
        ax.plot([lo, hi], [lo, hi], color=PAL['ink2'], linewidth=1, linestyle=(0, (4, 3)), zorder=1)
        ax.set_title(f'Ω = {om}', color=PAL['ink'], fontsize=11, loc='left')
        ax.set_xlabel('Fano factor, truth', color=PAL['ink2'], fontsize=9)
    axs[0].set_ylabel('Fano factor, model', color=PAL['ink2'], fontsize=9)
    axs[0].legend(frameon=False, fontsize=8, labelcolor=PAL['ink'], loc='upper left')
    fig.tight_layout()
    fig.savefig(os.path.join(assets, 'ws3_fano.png'), dpi=160, facecolor=PAL['surface'])
    plt.close(fig)


def report(path):
    """markdown tables from ws3.json (the prose around them lives in results/ws3_fluctuations.md)."""
    r = json.load(open(os.path.join(OUT, 'ws3.json')))
    LAB = dict(truth='truth', hyperme='HyperME', hyperme_l4='HyperME + 4 local currents', crnn='B1 CRNN', rev='B3 (A = 0)')
    f = lambda x, d=3: '–' if x is None else (f'{x:.{d}f}' if isinstance(x, float) else str(x))
    L = []
    L += ['### (a) Occupation numbers of X and Xp at every site (128 per model), Fano factor', '',
          '| sampler, Omega | model | median Fano | median abs. rel. difference to truth | correlation with truth across sites |', '|---|---|---|---|---|']
    for k, v in r['occupancy'].items():
        meth, om, nm = k.split('_', 2)
        L.append(f'| {"exact SSA" if meth == "ssa" else "tau-leap"}, {om[2:]} | {LAB[nm]} | {f(v["fano_median"])} | '
                 f'{f(v.get("fano_relerr_median"))} | {f(v.get("fano_corr"), 2)} |')
    L += ['', '### (b) Net firing count of single reaction channels, exact SSA, Omega = 50 (median over the 64 sites)', '',
          '| model | channel | mean one-way rates a + b | Var(N+ - N-)/t over a + b | J over a - b | J / J_truth | Var/t over truth | log(a/b) |',
          '|---|---|---|---|---|---|---|---|']
    for nm in ('truth', 'hyperme', 'hyperme_l4', 'crnn', 'rev'):
        ch = r['channels'].get(f'ssa_om50_{nm}')
        if not ch:
            continue
        for lab in ('R1', 'R2', 'TX', 'TXp'):
            d = ch[lab]
            L.append(f'| {LAB[nm]} | {lab} | {f(d["apb_median"], 1)} | {f(d["var_over_apb"], 2)} | {f(d["J_over_amb"], 3)} | '
                     f'{f(d.get("J_ratio_to_truth"), 2)} | {f(d.get("var_ratio_to_truth"), 2)} | {f(d["log_a_over_b"], 2)} |')
    L += ['', '### (c) Fluctuation relation for the local R1 + R2 loop, exact SSA, Omega = 50', '',
          'Slope of log P(Σ)/P(−Σ) against Σ, fitted through the origin over bins of width 0.5 with at least 10 counts on both sides; '
          'Σ pooled over 64 sites x 32 trajectories x non-overlapping windows after burn-in. "own" = medium entropy under the rates that '
          'generated the path, "true forces" = under the true rates evaluated on the same path.', '',
          '| model | window | slope, own rates | slope, true forces on this path | mean Σ own / true | bins (own / true) |', '|---|---|---|---|---|---|']
    for nm in ('truth', 'hyperme', 'hyperme_l4', 'crnn', 'rev'):
        ft = r['ft'].get(f'ssa_om50_{nm}')
        if not ft:
            continue
        for w in (0.05, 0.25):
            o, t = ft.get(f'S_w{w}'), ft.get(f'Sref_w{w}')
            L.append(f'| {LAB[nm]} | {w} | {f(o["slope"], 2)} ± {f(o.get("se"), 2)} | ' + (f'{f(t["slope"], 2)} ± {f(t.get("se"), 2)}' if t else '–')
                     + f' | {f(o.get("mean"), 1)} / ' + (f(t.get('mean'), 1) if t else '–') + f' | {len(o["x"])} / ' + (str(len(t['x'])) if t else '–') + ' |')
    L += ['', '### (d) C3 limit cycle at Omega = 200', '', 'Well-mixed (one site, exact SSA, 32 trajectories x 150 time units):', '',
          '| model | mean period | std | CV | phase diffusion D = 2 pi^2 Var(T) / T^3 | periods counted |', '|---|---|---|---|---|---|']
    for nm in ('truth', 'hyperme', 'crnn'):
        d = r['c3'].get(nm)
        if d:
            L.append(f'| {LAB[nm]} | {f(d["period_mean"], 2)} | {f(d["period_std"], 2)} | {f(d["period_cv"], 3)} | {f(d["phase_diffusion"], 4)} | {d["n_periods"]} |')
    L += ['', '8 x 8 grid (tau-leaping, 32 trajectories), median peak-to-peak of the site-mean X in time windows:', '',
          '| model | t 0-7 | 7-14 | 14-21 | 21-40 | 40-77 |', '|---|---|---|---|---|---|']
    for nm in ('truth', 'hyperme', 'crnn'):
        d = r['c3_grid'].get(nm)
        if d:
            L.append(f'| {LAB[nm]} | ' + ' | '.join(f(v, 2) for v in d.values()) + ' |')
    open(path, 'w').write('\n'.join(L) + '\n')


if __name__ == '__main__':
    p = argparse.ArgumentParser()
    p.add_argument('--stage', required=True)
    p.add_argument('--models', default='truth,hyperme,crnn,rev')
    p.add_argument('--B', type=int, default=32); p.add_argument('--seed', type=int, default=0)
    p.add_argument('--threads', type=int, default=4)
    a = p.parse_args()
    torch.set_num_threads(a.threads)
    sel = a.models.split(',')
    if a.stage == 'c1_ssa':
        stage_c1(sel, 'ssa', [50], a.B, 15.0, 3.0, 0.05, None, a.seed)
    elif a.stage == 'c1_tau':
        stage_c1(sel, 'tau', [50, 200, 1000], a.B, 15.0, 3.0, 0.05, 2e-3, a.seed)
    elif a.stage == 'c3':
        stage_c3([m for m in sel if m in ('truth', 'hyperme', 'crnn')], a.B, 77.0, 7.0, 0.05, 1e-3, a.seed)
    elif a.stage == 'c3wm':
        stage_c3wm([m for m in sel if m in ('truth', 'hyperme', 'crnn')], a.B, 160.0, 10.0, 0.05, a.seed)
    elif a.stage == 'analyze':
        analyze()
    elif a.stage == 'plot':
        plot(os.path.join(presets.ROOT, 'assets'))
    elif a.stage == 'report':
        report(os.path.join(OUT, 'ws3_tables.md'))
