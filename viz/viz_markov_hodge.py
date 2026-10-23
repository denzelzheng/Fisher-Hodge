"""Item 5 (CPU): Euclidean vs Onsager-Fisher weighted Hodge decomposition of the edge force f = log(rho_i Q_ij / rho_j Q_ji)
of a non-equilibrium Markov chain.  At stationarity the weighted gradient part is exactly 0 (d^T(ell f) = d^T J = 0), the
Euclidean one is not (spurious 'potential driving').  Static figure (grid 4x4 + ER n=12) + relaxation animation (grid)."""
import os, sys, numpy as np
FH_ROOT = os.environ.get('FH_ROOT', os.path.expanduser('~/fisher_hodge'))
sys.path.insert(0, os.path.join(FH_ROOT, 'analysis'))
from hodge_lib import make_adj, edges_of, incidence, cycle_basis_mat, build_Q, stationary, edge_quantities
import scipy.linalg as sla
sys.path.insert(0, os.path.join(FH_ROOT, 'viz'))
from plot_lib import OUT, write_video, fig_to_array
import matplotlib.pyplot as plt
from matplotlib import colors as mcolors
from matplotlib.patches import FancyArrowPatch


def system(kind, side_or_n, eps, seed):
    rng = np.random.default_rng(seed)
    adj = make_adj(kind, side_or_n, rng, 0.3)
    nn = len(adj); E = edges_of(adj); d0 = incidence(E, nn)
    B = rng.uniform(0, 2, len(E)); U = rng.standard_normal(nn)
    Z = cycle_basis_mat(d0); A = Z @ rng.standard_normal(Z.shape[1]); A /= np.sqrt(np.mean(A ** 2))
    Q = build_Q(E, nn, B, U, A, eps)
    rho0 = rng.dirichlet(np.ones(nn))
    if kind == 'grid':
        pos = np.array([[u % side_or_n, u // side_or_n] for u in range(nn)], float)
    else:
        th = 2 * np.pi * np.arange(nn) / nn; pos = np.stack([np.cos(th), np.sin(th)], 1) * 1.6 + 1.5
    return E, d0, Q, rho0, pos


def potential(f, w, d0):
    nn = d0.shape[1]
    Lp = d0.T @ (w[:, None] * d0) + (w.mean() / nn) * np.ones((nn, nn))
    phi = np.linalg.solve(Lp, d0.T @ (w * f))
    phi += np.linalg.solve(Lp, d0.T @ (w * f) - (d0.T @ (w[:, None] * d0)) @ phi)
    return phi - phi.mean()


def decomp(rho, Q, E, d0):
    a, b, J, f, ell = edge_quantities(rho, Q, E)
    phE = potential(f, np.ones_like(ell), d0); phL = potential(f, ell, d0)
    return dict(J=J, f=f, ell=ell, phE=phE, gE=d0 @ phE, phL=phL, gL=d0 @ phL)


def draw(ax, pos, E, edge_val, scale, node_val, nnorm, ncmap, color, title, sub, node_size=None):
    ax.set_aspect('equal'); ax.axis('off')
    for (i, j) in E:
        ax.plot(*pos[[i, j]].T, color='#d4d4d4', lw=1.0, zorder=1)
    for k, (i, j) in enumerate(E):
        v = edge_val[k]; m = abs(v) / scale
        if m < 0.02:
            continue
        s, t = (i, j) if v > 0 else (j, i)            # arrow along positive edge value (i -> j for v > 0)
        ax.add_patch(FancyArrowPatch(pos[s], pos[t], arrowstyle='-|>', mutation_scale=8 + 14 * min(m, 1), lw=0.6 + 5.5 * min(m, 1),
                                     color=color, shrinkA=11, shrinkB=11, zorder=2, alpha=0.9))
    sz = 260 if node_size is None else node_size
    ax.scatter(*pos.T, s=sz, c=ncmap(nnorm(node_val)), edgecolors='#333', linewidths=0.8, zorder=3)
    ax.set_title(title, fontsize=11, fontweight='bold')
    ax.text(0.5, -0.04, sub, transform=ax.transAxes, ha='center', va='top', fontsize=9.5)


def static_row(axr, kind, n, eps, seed, label):
    E, d0, Q, rho0, pos = system(kind, n, eps, seed)
    pi = stationary(Q)
    d = decomp(pi, Q, E, d0)
    gmax = np.abs(d['gE']).max(); pmax = np.abs(d['phE']).max()
    pnorm = mcolors.TwoSlopeNorm(0, -pmax, pmax); pc = plt.get_cmap('RdBu_r')
    psz = 140 + 900 * pi / pi.max()
    draw(axr[0], pos, E, d['J'], np.abs(d['J']).max(), pi, mcolors.Normalize(0, pi.max()), plt.get_cmap('Greens'), '#0e7490',
         f'{label}：稳态概率流 J（环流）', f'节点大小/颜色 = π，箭头 ∝ |J|\nΣ|J| = {np.abs(d["J"]).sum():.3f}', node_size=psz)
    draw(axr[1], pos, E, d['gE'], gmax, d['phE'], pnorm, pc, '#c2410c', '欧氏 Hodge 分解的梯度部分',
         f'节点色 = 势 φ_E，箭头 ∝ |dφ_E|\nmax|dφ_E| = {gmax:.3f} ≠ 0（虚假势能驱动）')
    draw(axr[2], pos, E, d['gL'], gmax, d['phL'], pnorm, pc, '#c2410c', 'Onsager–Fisher 加权分解的梯度部分',
         f'同一色标与箭头尺度\nmax|dφ_L| = {np.abs(d["gL"]).max():.1e}  ≈ 0（稳态无势能驱动）')
    return pnorm, pc


fig, ax = plt.subplots(2, 3, figsize=(15, 10), dpi=110, gridspec_kw=dict(wspace=0.05, hspace=0.3))
pn, pc = static_row(ax[0], 'grid', 4, 2.0, 7, '4×4 格子')
static_row(ax[1], 'er', 12, 2.0, 11, 'ER 图 n=12')
cax = fig.add_axes([0.92, 0.3, 0.01, 0.4])
cb = fig.colorbar(plt.cm.ScalarMappable(norm=pn, cmap=pc), cax=cax); cb.set_label('节点势 φ（上排色标）')
fig.suptitle('非平衡 Markov 链在稳态 π 处的边力 f = log(π_i Q_ij / π_j Q_ji)：欧氏 vs Onsager–Fisher 加权 Hodge 分解', fontsize=13)
p = os.path.join(OUT, 'markov_hodge_grid4x4_er12_euclid_vs_onsager_fisher_stationary.png'); fig.savefig(p, bbox_inches='tight'); plt.close(fig)
print('wrote', p)

# ---------------- relaxation animation (grid 4x4): rho(t) = rho0 exp(Qt) -> pi
E, d0, Q, rho0, pos = system('grid', 4, 2.0, 7)
pi = stationary(Q)
ev = np.sort(-np.linalg.eigvals(Q).real)[1]
ts = np.linspace(0, 25, 101)
P1 = sla.expm(Q * (ts[1] - ts[0]))
rhos = [rho0]
for _ in ts[1:]:
    r = np.abs(rhos[-1] @ P1); rhos.append(r / r.sum())
D = [decomp(r, Q, E, d0) for r in rhos]
gmax = max(np.abs(x['gE']).max() for x in D[len(D) // 2:]) * 1.0
pmax = max(np.abs(x['phE']).max() for x in D[len(D) // 2:])
pnorm = mcolors.TwoSlopeNorm(0, -pmax, pmax); pc = plt.get_cmap('RdBu_r')


def frame(axr, k, small=False):
    x = D[k]
    for a in axr:
        a.clear()
    nE, nL = np.abs(x['gE']).max(), np.abs(x['gL']).max()
    draw(axr[0], pos, E, x['gE'], gmax, np.clip(x['phE'], -pmax, pmax), pnorm, pc, '#c2410c', '欧氏分解：梯度部分',
         f'max|dφ| = {nE:.3f}' + ('' if small else '  → 收敛到非零（稳态仍有"势能驱动"）'), node_size=160 if small else None)
    draw(axr[1], pos, E, x['gL'], gmax, np.clip(x['phL'], -pmax, pmax), pnorm, pc, '#c2410c', 'Onsager–Fisher 加权分解：梯度部分',
         f'max|dφ| = {nL:.1e}' + ('' if small else '  → 0（只剩环流）'), node_size=160 if small else None)


fig, axr = plt.subplots(1, 2, figsize=(11, 5.6), dpi=100)
fig.subplots_adjust(top=0.82, bottom=0.12, wspace=0.08)
frames = []
for k in range(len(ts)):
    frame(axr, k)
    fig.suptitle(f'4×4 格子 Markov 链弛豫 ρ(t) → π：t = {ts[k]:.1f}（谱隙 λ₂ = {ev:.2f}；同一色标与箭头尺度）', fontsize=12)
    frames.append(fig_to_array(fig))
frames += [frames[-1]] * 15
plt.close(fig)
write_video(frames, os.path.join(OUT, 'markov_hodge_grid4x4_relaxation_euclid_vs_onsager_fisher'), fps=10, gif_width=1000)

TS = [0, 1, 3, 10, 25]
fig, ax = plt.subplots(2, 5, figsize=(16, 7.4), dpi=110, gridspec_kw=dict(wspace=0.05, hspace=0.3))
for j, tv in enumerate(TS):
    k = int(np.argmin(np.abs(ts - tv)))
    frame([ax[0, j], ax[1, j]], k, small=True)
    ax[0, j].set_title(f't = {tv}\n欧氏梯度部分', fontsize=10); ax[1, j].set_title('加权梯度部分', fontsize=10)
fig.suptitle('Markov 链弛豫胶片条：欧氏梯度部分收敛到非零图样，Onsager–Fisher 加权梯度部分衰减到 0', fontsize=12.5)
p = os.path.join(OUT, 'markov_hodge_grid4x4_relaxation_euclid_vs_onsager_fisher_filmstrip.png'); fig.savefig(p, bbox_inches='tight'); plt.close(fig)
print('wrote', p)
