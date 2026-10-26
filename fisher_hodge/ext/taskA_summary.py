"""Task A tables and figure from runs/external/common/taskA_ours.json and ~/ext_work/taskA/neep/*.json.
Writes runs/external/common/taskA_table.md, taskA_neep.json (collected NEEP results), taskA_error_vs_data.png."""
import os, sys, json, glob
import numpy as np

FH = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
OUT = os.path.join(FH, 'runs', 'external', 'common')
NEEPD = os.path.expanduser('~/ext_work/taskA/neep')
MS = (1, 4, 16, 64)


def main():
    ours = json.load(open(os.path.join(OUT, 'taskA_ours.json')))
    truth = ours['truth']
    neep = [json.load(open(f)) for f in sorted(glob.glob(os.path.join(NEEPD, '*.json')))]
    for r in neep:
        r.pop('hist', None)
    json.dump(neep, open(os.path.join(OUT, 'taskA_neep.json'), 'w'), indent=1)

    def T(sysn, cond):
        t = truth[f'{sysn}_{cond}']
        return t if not isinstance(t, dict) else t['sigma_jump']

    def err(vals, t):
        e = np.abs(np.array(vals) - t) / t
        return e

    def fmt(e):
        if len(e) == 0:
            return '–'
        return f'{100 * np.mean(e):.1f} ± {100 * np.std(e):.1f}' + ('' if len(e) == 3 else f' (n={len(e)})')

    def neep_vals(sysn, cond, M, lev=None):
        return [r['best_pred_rate'] for r in neep if r['sys'] == sysn and r['cond'] == cond and r['M'] == M and r.get('level') == lev]

    o2 = json.load(open(os.path.join(OUT, 'taskA_ours_2M.json'))) if os.path.exists(os.path.join(OUT, 'taskA_ours_2M.json')) else []
    lines = []
    rows = [r for r in ours['rows'] if r['sys'] == 'ratchet']
    lines.append('### A(i) Discrete flashing ratchet (NEEP home ground), relative error of the EP per step, % (mean ± sd over 3 seeds)\n')
    lines.append(f'Truth (NEEP analytic): base {T("ratchet", "base"):.4f}, r×2 {T("ratchet", "r2"):.4f}, V×2 {T("ratchet", "V2"):.4f}. '
                 'One trajectory = 10^4 steps.\n')
    lines.append('| M traj (transitions) | NEEP | counts (plug-in) | HyperME wired | r×2: HyperME zero-shot | r×2: NEEP new data | r×2: counts new data | V×2: HyperME zero-shot | V×2: NEEP new data | V×2: counts new data |')
    lines.append('|---|---|---|---|---|---|---|---|---|---|')
    for M in MS:
        rr = [r for r in rows if r['M'] == M]
        lines.append(f'| {M} ({rr[0]["transitions"]}) | {fmt(err(neep_vals("ratchet", "base", M), T("ratchet", "base")))} | '
                     f'{fmt(err([r["counts_base"] for r in rr], T("ratchet", "base")))} | {fmt(err([r["hyperme_base"] for r in rr], T("ratchet", "base")))} | '
                     f'{fmt(err([r["hyperme_r2_zeroshot"] for r in rr], T("ratchet", "r2")))} | {fmt(err(neep_vals("ratchet", "r2", M), T("ratchet", "r2")))} | '
                     f'{fmt(err([r["counts_r2_newdata"] for r in rr], T("ratchet", "r2")))} | '
                     f'{fmt(err([r["hyperme_V2_zeroshot"] for r in rr], T("ratchet", "V2")))} | {fmt(err(neep_vals("ratchet", "V2", M), T("ratchet", "V2")))} | '
                     f'{fmt(err([r["counts_V2_newdata"] for r in rr], T("ratchet", "V2")))} |')
    if o2:
        lines.append('\nSame data budget as NEEP (our estimators on the 2M trajectories NEEP uses for training + test):\n')
        lines.append('| M | NEEP | counts (2M) | HyperME wired (2M) | r×2 zero-shot (2M) | V×2 zero-shot (2M) |')
        lines.append('|---|---|---|---|---|---|')
        for M in MS:
            rr = [r for r in o2 if r['sys'] == 'ratchet' and r['M'] == M]
            lines.append(f'| {M} | {fmt(err(neep_vals("ratchet", "base", M), T("ratchet", "base")))} | {fmt(err([r["counts_base_2M"] for r in rr], T("ratchet", "base")))} | '
                         f'{fmt(err([r["hyperme_base_2M"] for r in rr], T("ratchet", "base")))} | {fmt(err([r["hyperme_r2_zeroshot_2M"] for r in rr], T("ratchet", "r2")))} | '
                         f'{fmt(err([r["hyperme_V2_zeroshot_2M"] for r in rr], T("ratchet", "V2")))} |')
    lines.append(f'\nNaive "no change" prediction for r×2 (base truth used as the answer): {100 * abs(T("ratchet", "base") - T("ratchet", "r2")) / T("ratchet", "r2"):.1f} %; '
                 f'for V×2: {100 * abs(T("ratchet", "base") - T("ratchet", "V2")) / T("ratchet", "V2"):.1f} %.\n')
    for lev, title in (('chan', 'channel-resolved trajectories (NEEP token = (n_X, net ATP count mod 3))'),
                       ('state', 'state-only jump chain of n_X (no channel labels, no times)')):
        rows = [r for r in ours['rows'] if r['sys'] == 'c1' and r['level'] == lev]
        lines.append(f'### A(ii) C1 phosphorylation cycle, one site, Ω = 200, {title}; relative error of the EP per jump, %\n')
        lines.append(f'Truth (exact CME): base {T("c1", "base"):.4f}, ATP×2 {T("c1", "atp2"):.4f} per jump. One trajectory = 10^4 jumps.\n')
        lines.append('| M traj (transitions) | NEEP | counts | HyperME wired | ATP×2: HyperME zero-shot | ATP×2: NEEP new data | ATP×2: counts new data |')
        lines.append('|---|---|---|---|---|---|---|')
        for M in MS:
            rr = [r for r in rows if r['M'] == M]
            lines.append(f'| {M} ({rr[0]["transitions"]}) | {fmt(err(neep_vals("c1", "base", M, lev), T("c1", "base")))} | '
                         f'{fmt(err([r["counts_base"] for r in rr], T("c1", "base")))} | {fmt(err([r["hyperme_base"] for r in rr], T("c1", "base")))} | '
                         f'{fmt(err([r["hyperme_atp2_zeroshot"] for r in rr], T("c1", "atp2")))} | {fmt(err(neep_vals("c1", "atp2", M, lev), T("c1", "atp2")))} | '
                         f'{fmt(err([r["counts_atp2_newdata"] for r in rr], T("c1", "atp2")))} |')
        if o2:
            lines.append('\nSame data budget as NEEP (2M trajectories): ' + '; '.join(
                f'M={M}: counts {fmt(err([r["counts_base_2M"] for r in o2 if r["sys"] == "c1" and r["level"] == lev and r["M"] == M], T("c1", "base")))}, '
                f'HyperME {fmt(err([r["hyperme_base_2M"] for r in o2 if r["sys"] == "c1" and r["level"] == lev and r["M"] == M], T("c1", "base")))}'
                for M in MS) + ' (%).')
        lines.append('')
    open(os.path.join(OUT, 'taskA_table.md'), 'w').write('\n'.join(lines) + '\n')
    print('\n'.join(lines))

    import matplotlib; matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    fig, axs = plt.subplots(1, 3, figsize=(13, 3.8))
    panels = [('ratchet', None, 'Flashing ratchet'), ('c1', 'chan', 'C1, channel-resolved'), ('c1', 'state', 'C1, state only')]
    for ax, (sysn, lev, title) in zip(axs, panels):
        rows = [r for r in ours['rows'] if r['sys'] == sysn and r.get('level') == lev]
        tr = [np.mean([r['transitions'] for r in rows if r['M'] == M]) for M in MS]
        series = [('NEEP', [neep_vals(sysn, 'base', M, lev) for M in MS], 'C0'),
                  ('counts', [[r['counts_base'] for r in rows if r['M'] == M] for M in MS], 'C7'),
                  ('HyperME wired', [[r['hyperme_base'] for r in rows if r['M'] == M] for M in MS], 'C3')]
        zs = 'hyperme_r2_zeroshot' if sysn == 'ratchet' else 'hyperme_atp2_zeroshot'
        cz = 'r2' if sysn == 'ratchet' else 'atp2'
        series.append(('HyperME zero-shot ' + ('r×2' if sysn == 'ratchet' else 'ATP×2'), [[r[zs] for r in rows if r['M'] == M] for M in MS], 'C1'))
        series.append(('NEEP new data ' + ('r×2' if sysn == 'ratchet' else 'ATP×2'), [neep_vals(sysn, cz, M, lev) for M in MS], 'C9'))
        for k, (lab, vals, col) in enumerate(series):
            t = T(sysn, 'base') if 'zero' not in lab and 'new data' not in lab else T(sysn, cz)
            m = [100 * np.mean(err(v, t)) if len(v) else np.nan for v in vals]
            ls = '--' if ('zero' in lab or 'new' in lab) else '-'
            ax.plot(tr, m, ls, marker='o', color=col, label=lab)
        ax.set_xscale('log'); ax.set_yscale('symlog', linthresh=1)
        ax.set_xlabel('training transitions'); ax.set_ylabel('relative error of EP (%)'); ax.set_title(title)
    axs[0].legend(fontsize=7)
    fig.tight_layout()
    fig.savefig(os.path.join(OUT, 'taskA_error_vs_data.png'), dpi=130)


if __name__ == '__main__':
    main()
