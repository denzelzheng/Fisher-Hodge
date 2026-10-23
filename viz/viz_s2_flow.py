"""Item 2: S2 stationary circulation.  (a) density at t=10T + flux streamlines J for [truth | Fisher-Hodge | JKO (reversible)];
(b) tracer particles advected (RK2, periodic) in each model's own steady velocity v = J/rho -> MP4/GIF + particle filmstrip."""
import sys, os, numpy as np
from plot_lib import *
import matplotlib.pyplot as plt
from matplotlib import colors as mcolors

st, b = sys.argv[1], int(sys.argv[2])
n = 32
T = load_truth('S2', st)
K = 200
models = ['fisher_hodge', 'jko_amort']
P = [('truth', T['rho'][b, K], T['J_end'][b])]
for m in models:
    c = load_cache(f'S2_{m}_{st}')
    P.append((m, c['R'][b, K], c['J'][b, 20]))
tag = f'S2_{st}_traj{b:02d}_truth_vs_fisher_hodge_vs_jko_amort'
vmax = max(float(T['rho'][b].max()), 1e-6)
norm = mcolors.PowerNorm(0.5, vmin=0, vmax=float(T['rho'][b, K].max()))
cmap = plt.get_cmap('magma')
xc = (np.arange(n) + 0.5) / n
Jt = np.hypot(*node_vec(P[0][2])); Jref = Jt.max()

# ------------------------------------------------------------- (a) streamlines on the 10T density
fig, ax = plt.subplots(1, 3, figsize=(13.2, 4.9), dpi=120, gridspec_kw=dict(wspace=0.04))
for i, (m, rho, J) in enumerate(P):
    a = ax[i]
    a.imshow(density_rgba(rho, norm, cmap, 0.5), origin='lower', extent=[0, 1, 0, 1], interpolation='bilinear')
    Jx, Jy = node_vec(J)
    mag = np.hypot(Jx, Jy)
    msk = mag < 0.03 * Jref
    U = np.ma.array(Jx, mask=msk); V = np.ma.array(Jy, mask=msk)
    lw = 0.35 + 2.8 * np.clip(mag / Jref, 0, 1)
    if (~msk).sum() > 3:
        a.streamplot(xc, xc, U, V, density=1.6, color='#7df9ff', linewidth=lw, arrowsize=1.0, arrowstyle='-|>', minlength=0.08)
    a.set_xlim(0, 1); a.set_ylim(0, 1); a.set_xticks([]); a.set_yticks([])
    a.set_title(LAB[m], fontsize=12, fontweight='bold', color=LCOL[m] if m != 'truth' else '#111')
    a.text(0.03, 0.03, f'max|J| = {mag.max() / Jref:.2f} × 真值', transform=a.transAxes, fontsize=9.5, color='white',
           bbox=dict(boxstyle='round,pad=0.25', fc=(0, 0, 0, 0.6), ec='none'))
    if m == 'jko_amort':
        a.text(0.5, 0.5, '稳态通量 ≈ 0\n（可逆模型无法表示环流）', transform=a.transAxes, ha='center', va='center', fontsize=12,
               color='white', fontweight='bold', bbox=dict(boxstyle='round,pad=0.4', fc=(0, 0, 0, 0.45), ec='none'))
fig.colorbar(plt.cm.ScalarMappable(norm=norm, cmap=cmap), ax=ax, shrink=0.85, pad=0.01, label='稳态密度 ρ(t=10T)（√ 刻度）')
fig.suptitle('S2 稳态 (t = 10T) 的概率通量 J = ρv 流线（线宽 ∝ |J|，|J| < 3% 真值最大值处不画）：漩涡在 / 漩涡没了', fontsize=12)
p = os.path.join(OUT, f'{tag}_flux_streamlines_t10T.png'); fig.savefig(p, bbox_inches='tight'); plt.close(fig)
print('  wrote', p)

# ------------------------------------------------------------- (b) tracer particles
def vel_field(rho, J):
    Jx, Jy = node_vec(J)
    return Jx / rho, Jy / rho


def interp(F, x, y):
    """periodic bilinear interpolation of node field F[a,b] at (x,y) (node (a,b) at ((b+.5)/n,(a+.5)/n))"""
    gx = x * n - 0.5; gy = y * n - 0.5
    i0 = np.floor(gx).astype(int); j0 = np.floor(gy).astype(int)
    fx = gx - i0; fy = gy - j0
    i0 %= n; j0 %= n; i1 = (i0 + 1) % n; j1 = (j0 + 1) % n
    return (F[j0, i0] * (1 - fx) * (1 - fy) + F[j0, i1] * fx * (1 - fy) + F[j1, i0] * (1 - fx) * fy + F[j1, i1] * fx * fy)


rng = np.random.default_rng(0)
NP = 300
w = np.sqrt(np.clip(P[0][1], 0, None)).ravel(); w /= w.sum()
cell = rng.choice(n * n, NP, p=w)
x0 = ((cell % n) + rng.uniform(0, 1, NP)) / n; y0 = ((cell // n) + rng.uniform(0, 1, NP)) / n
col = plt.get_cmap('hsv')(np.arctan2(y0 - 0.5, x0 - 0.5) / (2 * np.pi) + 0.5)
col[:, 3] = 1
S_END, DTP, FR = 10.0, 0.005, 0.1
nfr = int(round(S_END / FR)) + 1
per = int(round(FR / DTP))
tracks = []
vcap = None
for i, (m, rho, J) in enumerate(P):
    vx, vy = vel_field(rho, J)
    if vcap is None:
        vcap = 1.5 * np.quantile(np.hypot(vx, vy), 0.995)
    s = np.hypot(vx, vy); f = np.minimum(1, vcap / np.maximum(s, 1e-12)); vx, vy = vx * f, vy * f
    x, y = x0.copy(), y0.copy(); tr = [(x.copy(), y.copy())]
    for k in range(nfr - 1):
        for _ in range(per):
            u1, w1 = interp(vx, x, y), interp(vy, x, y)
            xm, ym = (x + DTP * u1) % 1, (y + DTP * w1) % 1
            u2, w2 = interp(vx, xm, ym), interp(vy, xm, ym)
            x = (x + 0.5 * DTP * (u1 + u2)) % 1; y = (y + 0.5 * DTP * (w1 + w2)) % 1
        tr.append((x.copy(), y.copy()))
    tracks.append(np.array(tr))      # (nfr, 2, P)
    print(m, 'mean particle path displacement per unit time', float(np.mean(np.hypot(*np.diff(tracks[-1], axis=0).transpose(1, 0, 2)) ) / FR))

TRAIL = 10
from matplotlib.collections import LineCollection


def segs(Tk, k):
    S, C = [], []
    for j in range(min(k, TRAIL)):
        a, b2 = Tk[k - j - 1], Tk[k - j]
        ok = (np.abs(a - b2) < 0.3).all(0)
        for p in np.nonzero(ok)[0]:
            S.append([(a[0, p], a[1, p]), (b2[0, p], b2[1, p])])
            c = col[p].copy(); c[3] = 0.85 * (1 - j / TRAIL); C.append(c)
    return S, C


fig = plt.figure(figsize=(13.2, 5.0), dpi=100)
axs, scs, lcs = [], [], []
for i, (m, rho, J) in enumerate(P):
    a = fig.add_axes([0.01 + i * 0.33, 0.1, 0.31, 0.78])
    a.set_facecolor('black')
    a.imshow(density_rgba(rho, norm, cmap, 0.5), origin='lower', extent=[0, 1, 0, 1], interpolation='bilinear', alpha=0.6)
    a.set_xlim(0, 1); a.set_ylim(0, 1); a.set_xticks([]); a.set_yticks([])
    a.set_title(LAB[m], fontsize=12, fontweight='bold', color=LCOL[m] if m != 'truth' else '#111')
    lc = LineCollection([], linewidths=1.2); a.add_collection(lc)
    sc = a.scatter(x0, y0, s=12, c=col, edgecolors='black', linewidths=0.3, zorder=3)
    axs.append(a); scs.append(sc); lcs.append(lc)
txt = fig.text(0.5, 0.035, '', ha='center', fontsize=11)
fig.suptitle(f'S2 示踪粒子：在各模型自己的稳态速度场 v = J/ρ 中推进（同一组 {NP} 个初始位置，颜色 = 初始方位）', fontsize=12)
frames = []
for k in range(nfr):
    for i in range(3):
        Tk = tracks[i]
        scs[i].set_offsets(Tk[k].T)
        S, C = segs(Tk, k); lcs[i].set_segments(S); lcs[i].set_color(C)
    txt.set_text(f'粒子时间 s = {k * FR:.1f}（流场冻结在 t = 10T 的稳态）')
    frames.append(fig_to_array(fig))
frames += [frames[-1]] * 10
plt.close(fig)
write_video(frames, os.path.join(OUT, f'{tag}_tracers'), fps=10, gif_width=1000)

# particle filmstrip
TS = [0, 1, 2, 5, 10]
fig, ax = plt.subplots(3, 5, figsize=(12, 7.6), dpi=120, gridspec_kw=dict(wspace=0.04, hspace=0.08))
for i, (m, rho, J) in enumerate(P):
    for j, s in enumerate(TS):
        k = int(round(s / FR)); a = ax[i, j]
        a.set_facecolor('black')
        a.imshow(density_rgba(rho, norm, cmap, 0.5), origin='lower', extent=[0, 1, 0, 1], interpolation='bilinear', alpha=0.6)
        S, C = segs(tracks[i], k); a.add_collection(LineCollection(S, colors=C, linewidths=0.9))
        a.scatter(*tracks[i][k], s=6, c=col, edgecolors='black', linewidths=0.2, zorder=3)
        a.set_xlim(0, 1); a.set_ylim(0, 1); a.set_xticks([]); a.set_yticks([])
        if i == 0:
            a.set_title(f's = {s}', fontsize=10)
        if j == 0:
            a.set_ylabel(LAB[m], fontsize=10, fontweight='bold', color=LCOL[m] if m != 'truth' else '#111')
fig.suptitle('S2 示踪粒子胶片条（稳态速度场 v = J/ρ，粒子时间 s；拖尾 = 最近 1 个时间单位）', fontsize=11.5)
p = os.path.join(OUT, f'{tag}_tracers_filmstrip.png'); fig.savefig(p, bbox_inches='tight'); plt.close(fig)
print('  wrote', p)
