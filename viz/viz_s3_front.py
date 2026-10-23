"""Item 3: S3 Fisher-KPP front propagation.  [truth | FH (old reaction) | FH-WFR | GNN-ODE] heat maps + 1D profile through the
seed centre (shape only).  Neon green = rho > 1.03 (above the true carrying capacity 1)."""
import sys, os, numpy as np
from plot_lib import *
import matplotlib.pyplot as plt
from matplotlib import colors as mcolors

st, b = sys.argv[1], int(sys.argv[2])
Y = load_truth('S3', st)['rho'][b]
P = [('truth', Y), ('fisher_hodge_old', load_cache(f'S3_fisher_hodge_{st}')['R'][b]),
     ('fisher_hodge_wfr', load_cache(f'S3_fisher_hodge_wfr_{st}')['R'][b]), ('gnn_ode', load_cache(f'S3_gnn_ode_{st}')['R'][b])]
LAB3 = dict(LAB); LAB3['fisher_hodge_old'] = 'Fisher–Hodge（旧反应项 −ρ r_θ）'; LAB3['fisher_hodge_wfr'] = 'Fisher–Hodge-WFR（新反应项）'
LC3 = dict(LCOL); LC3['fisher_hodge_old'] = '#eb6834'; LC3['fisher_hodge_wfr'] = '#2a78d6'
norm = mcolors.PowerNorm(0.4, vmin=0, vmax=1.0); cmap = plt.get_cmap('magma'); OV = 1.03
row = int(np.unravel_index(np.argmax(Y[0]), Y[0].shape)[0])
xc = (np.arange(32) + 0.5) / 32
tag = f'S3_{st}_traj{b:02d}_truth_vs_fisher_hodge_old_vs_fisher_hodge_wfr_vs_gnn_ode'
times = np.arange(0, 201, 2)


def mstat(X):
    if not np.isfinite(X).all():
        return '发散', '#ff3b30'
    return f'max ρ={X.max():.2f}\nmin ρ={X.min():+.2f}', ('#39ff14' if X.max() > OV else ('#ff2bd6' if X.min() < 0 else 'white'))


fig = plt.figure(figsize=(13.5, 7.6), dpi=100)
nc = 4; L, Rr = 0.02, 0.9; pw = (Rr - L) / nc; bot, top = 0.42, 0.88
axs, ims, txt, tt = [], [], [], []
for i, (m, X) in enumerate(P):
    ax = fig.add_axes([L + i * pw + 0.006, bot, pw - 0.012, top - bot])
    im = ax.imshow(density_rgba(X[0], norm, cmap, 0.5, over=OV), origin='lower', extent=[0, 1, 0, 1], interpolation='nearest')
    ax.axhline((row + 0.5) / 32, color='white', ls='--', lw=0.8, alpha=0.8)
    ax.set_xticks([]); ax.set_yticks([])
    ax.set_title(LAB3[m], fontsize=10.5, fontweight='bold', color=LC3[m])
    t1 = ax.text(0.97, 0.97, '', transform=ax.transAxes, ha='right', va='top', fontsize=8.5, color='white',
                 bbox=dict(boxstyle='round,pad=0.25', fc=(0, 0, 0, 0.6), ec='none'))
    t2 = ax.text(0.03, 0.97, '', transform=ax.transAxes, ha='left', va='top', fontsize=8.5, color='white',
                 bbox=dict(boxstyle='round,pad=0.25', fc=(0, 0, 0, 0.6), ec='none'))
    axs.append(ax); ims.append(im); txt.append(t1); tt.append(t2)
cax = fig.add_axes([0.915, bot, 0.012, top - bot])
cb = fig.colorbar(plt.cm.ScalarMappable(norm=norm, cmap=cmap), cax=cax); cb.set_label('ρ（真实饱和值 = 1；ρ^0.4 刻度）', fontsize=8.5)
from matplotlib.patches import Patch
fig.legend(handles=[Patch(color='#c8ff9e', label='ρ ≈ 1.05（越过饱和值）'), Patch(color='#00b300', label='ρ ≥ 1.6'), Patch(color=NEG_SOFT, label='ρ < 0'), Patch(color=NEG_HARD, label='ρ ≪ 0')],
           loc='upper right', bbox_to_anchor=(0.99, 0.405), ncol=1, fontsize=8.5, frameon=False)
pax = fig.add_axes([0.06, 0.13, 0.72, 0.22])
lines = {}
for m, X in P:
    lines[m], = pax.plot(xc, X[0, row], color=LC3[m], lw=3.0 if m == 'truth' else 2.0, ls='-' if m != 'truth' else '-',
                         alpha=0.9, label=LAB3[m], zorder=5 if m == 'truth' else 3)
pax.axhline(1.0, color='#111', ls=':', lw=1)
pax.text(0.005, 1.03, '饱和值 1', fontsize=8.5, color='#111', va='bottom')
pax.set_xlim(0, 1); pax.set_ylim(-0.15, 1.7); pax.set_yticks([0, 0.5, 1, 1.5])
pax.set_xlabel('x（沿热图中白色虚线的剖面，仅看形状）', fontsize=9); pax.set_ylabel('ρ')
for sp in ['top', 'right']:
    pax.spines[sp].set_visible(False)
pax.legend(loc='upper left', bbox_to_anchor=(1.01, 1.0), fontsize=8.5, frameon=False)
fig.suptitle('S3 Fisher–KPP 前沿传播（OOD 增长率场）：旧反应项过冲到 ~1.5，新 WFR 反应项饱和在 1', fontsize=12.5, y=0.985)
banner = fig.text(0.5, 0.925, '', ha='center', fontsize=10.5, fontweight='bold')
tl = Timeline(fig, [0.06, 0.03, 0.6, 0.035])
frames = []
for kk, k in enumerate(times):
    t = k * DT
    for i, (m, X) in enumerate(P):
        ims[i].set_data(density_rgba(X[k], norm, cmap, 0.5, over=OV))
        s, c = mstat(X[k]); txt[i].set_text(s); txt[i].set_color(c)
        tt[i].set_text(f't = {t:.1f}')
        set_border(axs[i], t, k=kk)
        lines[m].set_ydata(np.clip(X[k, row], -0.15, 1.7))
    ph = phase(t)
    banner.set_text('训练窗结束 (t = T)：此后全部为外推' if ph == 'flash' else ('训练窗内 t ≤ T' if ph == 'train' else ''))
    banner.set_color(FLASH_C if ph == 'flash' else '#166534')
    tl.update(t, kk)
    frames.append(fig_to_array(fig))
frames += [frames[-1]] * 15
plt.close(fig)
write_video(frames, os.path.join(OUT, f'{tag}_rollout'), fps=10, gif_width=1100)

TS = [(0, 't = 0'), (20, 't = T（训练窗末）'), (40, 't = 2T'), (100, 't = 5T'), (200, 't = 10T')]
fig, ax = plt.subplots(5, 5, figsize=(12, 12.2), dpi=120, gridspec_kw=dict(wspace=0.05, hspace=0.1, height_ratios=[1, 1, 1, 1, 0.8]))
for i, (m, X) in enumerate(P):
    for j, (k, lab) in enumerate(TS):
        a = ax[i, j]
        a.imshow(density_rgba(X[k], norm, cmap, 0.5, over=OV), origin='lower', extent=[0, 1, 0, 1], interpolation='nearest')
        a.axhline((row + 0.5) / 32, color='white', ls='--', lw=0.6, alpha=0.7)
        a.set_xticks([]); a.set_yticks([])
        s, c = mstat(X[k])
        a.text(0.97, 0.97, s.replace('\n', '  '), transform=a.transAxes, ha='right', va='top', fontsize=6.5, color=c,
               bbox=dict(boxstyle='round,pad=0.2', fc=(0, 0, 0, 0.6), ec='none'))
        for sp in a.spines.values():
            sp.set_edgecolor(TRAIN_C if k <= 20 else EXTRAP_C); sp.set_linewidth(2.5 if k <= 20 else 1)
        if i == 0:
            a.set_title(lab, fontsize=10)
        if j == 0:
            a.set_ylabel(LAB3[m].replace('（', '\n（'), fontsize=8.5, color=LC3[m], fontweight='bold')
for j, (k, lab) in enumerate(TS):
    a = ax[4, j]
    for m, X in P:
        a.plot(xc, np.clip(X[k, row], -0.15, 1.7), color=LC3[m], lw=2.2 if m == 'truth' else 1.4)
    a.axhline(1, color='#111', ls=':', lw=0.8); a.set_ylim(-0.15, 1.7); a.set_xlim(0, 1)
    a.set_xticks([]); a.set_yticks([0, 1, 1.5] if j == 0 else [])
    for sp in ['top', 'right']:
        a.spines[sp].set_visible(False)
    if j == 0:
        a.set_ylabel('剖面 ρ(x)', fontsize=9)
fig.legend(handles=[Patch(color='#c8ff9e', label='ρ ≈ 1.05'), Patch(color='#00b300', label='ρ ≥ 1.6'), Patch(color=NEG_SOFT, label='ρ < 0'), Patch(color=NEG_HARD, label='ρ ≪ 0')] +
           [plt.Line2D([], [], color=LC3[m], lw=2, label=LAB3[m]) for m, _ in P], loc='lower center', ncol=4, fontsize=8.5,
           frameon=False, bbox_to_anchor=(0.5, 0.045))
fig.suptitle('S3 Fisher–KPP 前沿传播胶片条（末行 = 白色虚线处剖面；绿色 = 越过饱和值 1，越深越高）', fontsize=12, y=0.92)
p = os.path.join(OUT, f'{tag}_filmstrip.png'); fig.savefig(p, bbox_inches='tight'); plt.close(fig)
print('  wrote', p)
