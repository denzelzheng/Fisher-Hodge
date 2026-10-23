"""Items 1 and 4: side-by-side density rollouts [truth | models...] for one test trajectory -> MP4/GIF + filmstrip PNG.
python viz/viz_grid.py SYS DSYS SET TRAJ model1,model2,... [title]"""
import sys, os, numpy as np
from plot_lib import *
import matplotlib.pyplot as plt
from matplotlib import colors as mcolors

sysn, dsys, st, b = sys.argv[1], sys.argv[2], sys.argv[3], int(sys.argv[4])
models = sys.argv[5].split(',')
title = sys.argv[6] if len(sys.argv) > 6 else sysn
Y = load_truth(dsys, st)['rho'][b]
P = [('truth', Y)] + [(m, load_cache(f'{sysn}_{m}_{st}')['R'][b]) for m in models]
vmax = float(Y.max())
mean0 = float(Y[0].mean())
norm = mcolors.PowerNorm(0.5, vmin=0, vmax=vmax)
cmap = plt.get_cmap('magma')
NEGH = 0.5 * mean0
tag = f'{sysn}_{st}_traj{b:02d}_' + '_vs_'.join(['truth'] + models)
nc = len(P)
times = np.arange(0, 201, 2)

# ---------------------------------------------------------------- animation
W = min(16.0, 3.3 * nc + 1.2)
fig = plt.figure(figsize=(W, 4.6), dpi=100)
L, R_, bot, top = 0.02, 0.9, 0.2, 0.86
pw = (R_ - L) / nc
axs, ims, txt, tt = [], [], [], []
for i, (m, X) in enumerate(P):
    ax = fig.add_axes([L + i * pw + 0.006, bot, pw - 0.012, top - bot])
    im = ax.imshow(density_rgba(X[0], norm, cmap, NEGH), origin='lower', extent=[0, 1, 0, 1], interpolation='nearest')
    ax.set_xticks([]); ax.set_yticks([])
    ax.set_title(LAB[m], fontsize=11, color=LCOL[m] if m != 'truth' else '#111', fontweight='bold')
    t1 = ax.text(0.97, 0.97, '', transform=ax.transAxes, ha='right', va='top', fontsize=8.5, color='white',
                 bbox=dict(boxstyle='round,pad=0.25', fc=(0, 0, 0, 0.55), ec='none'))
    t2 = ax.text(0.03, 0.97, '', transform=ax.transAxes, ha='left', va='top', fontsize=8.5, color='white',
                 bbox=dict(boxstyle='round,pad=0.25', fc=(0, 0, 0, 0.55), ec='none'))
    axs.append(ax); ims.append(im); txt.append(t1); tt.append(t2)
cax = fig.add_axes([0.915, bot, 0.012, top - bot])
sm = plt.cm.ScalarMappable(norm=norm, cmap=cmap)
cb = fig.colorbar(sm, cax=cax); cb.set_label('ρ（色标固定为真值全程最大值，√ 刻度）', fontsize=8)
fig.suptitle(title, fontsize=12.5, y=0.975)
banner = fig.text(0.5, 0.155, '', ha='center', va='center', fontsize=11, color=FLASH_C, fontweight='bold')
tl = Timeline(fig, [0.06, 0.075, 0.6, 0.05])
fig.legend(handles=legend_patches(), loc='lower right', bbox_to_anchor=(0.985, 0.03), ncol=3, fontsize=8.5, frameon=False)
frames = []
for kk, k in enumerate(times):
    t = k * DT
    for i, (m, X) in enumerate(P):
        ims[i].set_data(density_rgba(X[k], norm, cmap, NEGH))
        s, c = stat_text(X[k]); txt[i].set_text(s); txt[i].set_color(c)
        tt[i].set_text(f't = {t:.1f}')
        set_border(axs[i], t, k=kk)
    ph = phase(t)
    banner.set_text('训练窗结束 (t = T)：此后全部为外推' if ph == 'flash' else ('训练窗内 t ≤ T' if ph == 'train' else ''))
    banner.set_color(FLASH_C if ph == 'flash' else '#166534')
    tl.update(t, kk)
    frames.append(fig_to_array(fig))
frames += [frames[-1]] * 15
plt.close(fig)
write_video(frames, os.path.join(OUT, f'{tag}_rollout'), fps=10, gif_width=1100)

# ---------------------------------------------------------------- filmstrip
TS = [(0, 't = 0'), (20, 't = T（训练窗末）'), (40, 't = 2T'), (100, 't = 5T'), (200, 't = 10T')]
fig, ax = plt.subplots(nc, 5, figsize=(11.5, 2.1 * nc + 0.5), dpi=130, gridspec_kw=dict(wspace=0.04, hspace=0.08))
for i, (m, X) in enumerate(P):
    for j, (k, lab) in enumerate(TS):
        a = ax[i, j]
        a.imshow(density_rgba(X[k], norm, cmap, NEGH), origin='lower', extent=[0, 1, 0, 1], interpolation='nearest')
        a.set_xticks([]); a.set_yticks([])
        s, c = stat_text(X[k])
        a.text(0.97, 0.97, s.replace('\n', '  '), transform=a.transAxes, ha='right', va='top', fontsize=6.5, color=c,
               bbox=dict(boxstyle='round,pad=0.2', fc=(0, 0, 0, 0.55), ec='none'))
        for sp in a.spines.values():
            sp.set_edgecolor(TRAIN_C if k <= 20 else EXTRAP_C); sp.set_linewidth(2.5 if k <= 20 else 1)
        if i == 0:
            a.set_title(lab, fontsize=10)
        if j == 0:
            a.set_ylabel(LAB[m], fontsize=9.5, color=LCOL[m], fontweight='bold')
fig.colorbar(plt.cm.ScalarMappable(norm=norm, cmap=cmap), ax=ax, shrink=0.6, pad=0.01, label='ρ（√ 刻度）')
fig.legend(handles=legend_patches(), loc='lower center', ncol=3, fontsize=8.5, frameon=False, bbox_to_anchor=(0.45, 0.0))
fig.suptitle(title + '　|　绿框 = 训练窗内，灰框 = 外推', fontsize=11.5)
p = os.path.join(OUT, f'{tag}_filmstrip.png')
fig.savefig(p, bbox_inches='tight'); plt.close(fig)
print('  wrote', p, f'{os.path.getsize(p) / 1e6:.1f} MB')
