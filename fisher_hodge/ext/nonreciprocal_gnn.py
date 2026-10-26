"""Re-implementation (from the paper) of the non-reciprocal GNN + neural-ODE baseline of
Uwamichi, Schnyder, Kobayashi, Sawai, "Integrating GNN and neural ODEs for estimating non-reciprocal two-body
interactions in mixed-species collective motion", NeurIPS 2024 (arXiv 2405.16503), and of its mixed-species
particle simulator.  Plain PyTorch, no DGL.  Written for the external-baseline comparison (common task C).

Simulator (paper Sec. 5.2, mixed-species model): N particles with position r_i in the periodic box [0, L)^2,
polarity phi_i, species c_i in {0, 1}; neighbours are the j with |r_ij| < d0, r_ij = r_j - r_i (periodic).
    dr_i   = [ v0 p_i + beta sum_j (1 - 1/|r_ij|) rhat_ij ] dt                      (excluded volume, repulsive)
    dphi_i = sum_j [ a_CF(c_i) (1 + rhat_ij . p_j) / 2 + a_Ch(c_i) J_Ch(|r_ij|) ] sin(angle(r_ij) - phi_i) dt
             + sigma dW_i
    J_Ch(d) = kappa / (2 pi) * max(K1(kappa d) - K1(cut), 0)                     (chemo-attractant field of j)
Parameter set (i): a_CF = (0.1, 0.9), a_Ch = (2.0, 0.2), v0 = beta = 1, kappa = 0.5, sigma^2 = 0.2, d0 = 1,
L = 20, 200 + 200 particles, Heun (Stratonovich) step 0.1, snapshots every 1.0 for t = 0..300.
The printed equations of the paper are not self-consistent in their signs and in whether r_ij is normalised (the
contact-following and chemotaxis terms as printed would turn particles away from a leader); the signs, the
unit-vector normalisation, the Bessel cut-off (cut = 3.5) and the 1/d excluded-volume form were taken from the
authors' public repository (read only, not copied).  See runs/external/common/ADAPTATION.md.

Learned model (paper Sec. 4, the '2Dsym' variant): in the frame of the receiving particle i,
    dr_i/dt   = R(phi_i) [ F1_r(e_r(c_i)) + sum_j F2_r(R(-phi_i) r_ij, (phi_j - phi_i) mod 2pi, e_r(c_i), e_r(c_j)) ]
    dphi_i/dt = F1_phi(e_p(c_i)) + sum_j F2_phi(R(-phi_i) r_ij, (phi_j - phi_i) mod 2pi, e_p'(c_i), e_p'(c_j))
four independent MLPs (3 hidden layers of 128, ELU) each followed by a scaling layer exp(A) x + B (A = -20, B = 0 at
initialisation, biases zero), separate 2-d species embeddings per network.  Training: from the snapshot at t0
integrate the ODE (explicit Euler, step 0.1) to t0 + 1 and compare with the snapshot,
    L = |dr|^2 / |r_true|^2 + (1 - <cos dphi>) / (1 - <cos phi_true>)   (periodic differences),
LAMB (lr 1e-3, betas (0.5, 0.4), eps 1e-6), 60 (trajectory, t0) pairs per batch, 3 training + 3 validation
trajectories, 20000 epochs, the parameters with the lowest validation loss are kept.
"""
import math
import numpy as np
import torch
import torch.nn as nn

PARAMS_I = dict(a_CF=(0.1, 0.9), a_Ch=(2.0, 0.2), v0=1.0, beta=1.0, kappa=0.5, cut=3.5, sigma=math.sqrt(0.2),
                d0=1.0, L=20.0)


# ------------------------------------------------------------------------------------------ geometry
def wrap(d, L):
    """periodic minimum image of a difference vector"""
    return d - L * torch.round(d / L)


def neighbour_pairs(x, d0, L):
    """x (B, N, 2) -> (b, i, j) index tensors of the ordered pairs with 0 < |x_j - x_i| < d0, and r_ij (P, 2)."""
    with torch.no_grad():
        d = wrap(x[:, None, :, :] - x[:, :, None, :], L)                  # d[b, i, j] = x_j - x_i
        r2 = (d * d).sum(-1)
        N = x.shape[1]
        r2.diagonal(dim1=1, dim2=2).fill_(float('inf'))
        b, i, j = torch.nonzero(r2 < d0 * d0, as_tuple=True)
    rij = wrap(x[b, j] - x[b, i], L)                                       # differentiable in x
    return b, i, j, rij


def rot(v, ang):
    """rotate 2-vectors v (..., 2) by angle ang (...)"""
    c, s = torch.cos(ang), torch.sin(ang)
    return torch.stack([c * v[..., 0] - s * v[..., 1], s * v[..., 0] + c * v[..., 1]], -1)


def scatter_sum(val, b, i, B, N):
    out = torch.zeros(B * N, *val.shape[1:], dtype=val.dtype, device=val.device)
    return out.index_add_(0, b * N + i, val).view(B, N, *val.shape[1:])


# ------------------------------------------------------------------------------------------ true model
class MixedSpecies(nn.Module):
    """true drift of the mixed-species model; state x (B, N, 2), phi (B, N), species c (B, N) long."""
    def __init__(s, **kw):
        super().__init__()
        p = dict(PARAMS_I); p.update(kw)
        s.p = p
        s.register_buffer('a_CF', torch.tensor(p['a_CF'], dtype=torch.float32))
        s.register_buffer('a_Ch', torch.tensor(p['a_Ch'], dtype=torch.float32))
        s.K1cut = float(torch.special.modified_bessel_k1(torch.tensor(p['cut'], dtype=torch.float64)))

    def J_ch(s, d):
        k = s.p['kappa']
        return k / (2 * math.pi) * torch.relu(torch.special.modified_bessel_k1(k * d) - s.K1cut)

    def messages(s, rij, phi_i, phi_j, ci, cj):
        """per-pair velocity message (lab frame) and torque message"""
        d = rij.norm(dim=-1).clamp_min(1e-9)
        rh = rij / d[:, None]
        pj = torch.stack([torch.cos(phi_j), torch.sin(phi_j)], -1)
        v = s.p['beta'] * (1 - 1 / d)[:, None] * rh
        sin_rel = rh[:, 1] * torch.cos(phi_i) - rh[:, 0] * torch.sin(phi_i)    # sin(angle(r_ij) - phi_i)
        J_cf = 0.5 * (1 + (rh * pj).sum(-1))
        w = (s.a_CF[ci] * J_cf + s.a_Ch[ci] * s.J_ch(d)) * sin_rel
        return v, w

    def drift(s, x, phi, c):
        B, N = phi.shape
        b, i, j, rij = neighbour_pairs(x, s.p['d0'], s.p['L'])
        v, w = s.messages(rij, phi[b, i], phi[b, j], c[b, i], c[b, j])
        p = torch.stack([torch.cos(phi), torch.sin(phi)], -1)
        return s.p['v0'] * p + scatter_sum(v, b, i, B, N), scatter_sum(w, b, i, B, N)


@torch.no_grad()
def simulate(model, x, phi, c, t_save, dt=0.1, sigma=None, generator=None, L=20.0):
    """Heun (Stratonovich; the noise is additive so this is also Ito) for the drift of `model` with angular noise
    sigma.  t_save: number of saved intervals of length 1.0.  Returns x (T+1, B, N, 2), phi (T+1, B, N)."""
    sig = model.p['sigma'] if sigma is None else sigma
    nsub = int(round(1.0 / dt))
    X, P = [x.clone()], [phi.clone()]
    for _ in range(t_save):
        for _ in range(nsub):
            dW = torch.randn(phi.shape, device=phi.device, generator=generator) * math.sqrt(dt) * sig
            v1, w1 = model.drift(x, phi, c)
            xp, pp = x + dt * v1, phi + dt * w1 + dW
            v2, w2 = model.drift(xp, pp, c)
            x = torch.remainder(x + 0.5 * dt * (v1 + v2), L)
            phi = torch.remainder(phi + 0.5 * dt * (w1 + w2) + dW, 2 * math.pi)
        X.append(x.clone()); P.append(phi.clone())
    return torch.stack(X), torch.stack(P)


def make_dataset(n_traj, n_per_species=(200, 200), T=300, seed=0, device='cuda', **kw):
    """uniform initial positions in [0, L)^2 and polarities; returns dict(x (T+1, M, N, 2), phi (T+1, M, N), c (M, N))."""
    g = torch.Generator(device=device).manual_seed(seed)
    m = MixedSpecies(**kw).to(device)
    L = m.p['L']
    N = sum(n_per_species)
    c = torch.cat([torch.full((n,), k, dtype=torch.long) for k, n in enumerate(n_per_species)]).to(device)
    c = c[None].expand(n_traj, N).contiguous()
    x = torch.rand(n_traj, N, 2, device=device, generator=g) * L
    phi = torch.rand(n_traj, N, device=device, generator=g) * 2 * math.pi
    X, P = simulate(m, x, phi, c, T, generator=g, L=L)
    return dict(x=X.cpu(), phi=P.cpu(), c=c.cpu(), params=m.p, n_per_species=list(n_per_species), seed=seed)


# ------------------------------------------------------------------------------------------ learned model
class Scaling(nn.Module):
    def __init__(s, n):
        super().__init__()
        s.A = nn.Parameter(torch.tensor(-20.0))
        s.B = nn.Parameter(torch.zeros(n))

    def forward(s, x):
        return torch.exp(s.A) * x + s.B


def mlp(n_in, n_out, hidden=(128, 128, 128)):
    layers, k = [], n_in
    for h in hidden:
        lin = nn.Linear(k, h); nn.init.zeros_(lin.bias)
        layers += [lin, nn.ELU()]; k = h
    lin = nn.Linear(k, n_out); nn.init.zeros_(lin.bias)
    layers += [lin, Scaling(n_out)]
    return nn.Sequential(*layers)


class NonreciprocalGNN(nn.Module):
    """F1_r, F1_phi (self terms), F2_r, F2_phi (pair terms), each with its own species embedding (dim n_emb)."""
    def __init__(s, n_species=2, n_emb=2, d0=1.0, L=20.0, hidden=(128, 128, 128)):
        super().__init__()
        s.d0, s.L = d0, L
        s.emb = nn.ModuleList([nn.Embedding(n_species, n_emb) for _ in range(4)])   # F2_r, F2_phi, F1_r, F1_phi
        s.F2r = mlp(3 + 2 * n_emb, 2, hidden)
        s.F2p = mlp(3 + 2 * n_emb, 1, hidden)
        s.F1r = mlp(n_emb, 2, hidden)
        s.F1p = mlp(n_emb, 1, hidden)

    def pair_messages(s, rij, phi_i, phi_j, ci, cj):
        """velocity message in the frame of i and torque message, for pairs"""
        rb = rot(rij, -phi_i)
        dphi = torch.remainder(phi_j - phi_i, 2 * math.pi)[:, None]
        zr = torch.cat([rb, dphi, s.emb[0](ci), s.emb[0](cj)], -1)
        zp = torch.cat([rb, dphi, s.emb[1](ci), s.emb[1](cj)], -1)
        return s.F2r(zr), s.F2p(zp).squeeze(-1)

    def self_terms(s, c):
        return s.F1r(s.emb[2](c)), s.F1p(s.emb[3](c)).squeeze(-1)

    def drift(s, x, phi, c):
        B, N = phi.shape
        b, i, j, rij = neighbour_pairs(x, s.d0, s.L)
        vb, w = s.pair_messages(rij, phi[b, i], phi[b, j], c[b, i], c[b, j])
        v0, w0 = s.self_terms(c)
        vbody = v0 + scatter_sum(vb, b, i, B, N)
        return rot(vbody, phi), w0 + scatter_sum(w, b, i, B, N)

    def ode(s, x, phi, c, t=1.0, dt=0.1):
        """explicit Euler (the training integrator); keeps the unwrapped angle"""
        for _ in range(int(round(t / dt))):
            v, w = s.drift(x, phi, c)
            x = torch.remainder(x + dt * v, s.L)
            phi = phi + dt * w
        return x, phi


def nondim_loss(xp, pp, xt, pt, L):
    dx = wrap(xp - xt, L)
    nx = wrap(xt, L)                     # periodic distance from the origin, as the normalisation
    Lr = (dx ** 2).mean() / (nx ** 2).mean()
    Lp = (1 - torch.cos(pp - pt).mean()) / (1 - torch.cos(pt).mean())
    return Lr, Lp


def train_gnn(data, train_idx, valid_idx, epochs=20000, batch=60, lr=1e-3, device='cuda', seed=0, log=None,
              ckpt=None, time_limit=None):
    """the paper's protocol; returns the best model (lowest validation loss) and the history"""
    import time
    import torch_optimizer as topt
    torch.manual_seed(seed)
    X, P, C = data['x'].to(device), data['phi'].to(device), data['c'].to(device)
    T = X.shape[0] - 1
    L = data['params']['L']
    model = NonreciprocalGNN(d0=data['params']['d0'], L=L).to(device)
    opt = topt.Lamb(model.parameters(), lr=lr, betas=(0.5, 0.4), eps=1e-6)
    pairs = lambda idx: torch.tensor([(m, t) for m in idx for t in range(T)], device=device)
    tr, va = pairs(train_idx), pairs(valid_idx)
    best, best_state, hist = float('inf'), None, []
    t0 = time.time()
    g = torch.Generator(device='cpu').manual_seed(seed)

    def batch_loss(mt):
        m, t = mt[:, 0], mt[:, 1]
        xp, pp = model.ode(X[t, m], P[t, m], C[m])
        return nondim_loss(xp, pp, X[t + 1, m], P[t + 1, m], L)
    for ep in range(epochs):
        model.train()
        perm = tr[torch.randperm(len(tr), generator=g).to(device)]
        tl = []
        for k in range(0, len(perm), batch):
            Lr, Lp = batch_loss(perm[k:k + batch])
            loss = Lr + Lp
            opt.zero_grad(); loss.backward(); opt.step()
            tl.append(loss.item())
        model.eval()
        with torch.no_grad():
            vl, n = 0.0, 0
            for k in range(0, len(va), batch):
                mt = va[k:k + batch]
                Lr, Lp = batch_loss(mt)
                vl += (Lr + Lp).item() * len(mt); n += len(mt)
            vl /= n
        hist.append((ep, float(np.mean(tl)), vl, time.time() - t0))
        if vl < best:
            best = vl
            best_state = {k: v.detach().clone() for k, v in model.state_dict().items()}
            if ckpt:
                torch.save(dict(state=best_state, epoch=ep, valid=vl), ckpt)
        if log and (ep % 50 == 0 or ep == epochs - 1):
            print(f'epoch {ep} train {np.mean(tl):.5f} valid {vl:.5f} best {best:.5f} {time.time() - t0:.0f}s', file=log, flush=True)
        if time_limit and time.time() - t0 > time_limit:
            if log:
                print(f'time limit reached at epoch {ep}', file=log, flush=True)
            break
    model.load_state_dict(best_state)
    return model, hist


@torch.no_grad()
def rollout(model, x, phi, c, t_save, dt=0.1, sigma=0.0, generator=None):
    """learned model forward in time: sigma = 0 is the deterministic ODE (Euler, the training integrator);
    sigma > 0 adds angular noise (Euler-Maruyama), as in the authors' re-simulation."""
    X, P = [x], [phi]
    nsub = int(round(1.0 / dt))
    for _ in range(t_save):
        for _ in range(nsub):
            v, w = model.drift(x, phi, c)
            x = torch.remainder(x + dt * v, model.L)
            phi = phi + dt * w
            if sigma > 0:
                phi = phi + sigma * math.sqrt(dt) * torch.randn(phi.shape, device=phi.device, generator=generator)
        X.append(x); P.append(torch.remainder(phi, 2 * math.pi))
    return torch.stack(X), torch.stack(P)


@torch.no_grad()
def compare_interactions(gnn, truth, X, P, C, n_frames=20):
    """learned vs true pair messages (velocity in the frame of i, torque) evaluated on the neighbour pairs of real
    snapshots; returns per species pair (ci, cj): relative L2 error and R^2 of both components, plus the self terms.
    The decomposition into pair and self terms is unique up to nothing here because the self terms do not depend on
    the neighbourhood and the pair terms vanish beyond d0."""
    dev = next(gnn.parameters()).device
    idx = torch.linspace(1, X.shape[0] - 1, n_frames).long()
    x = X[idx].reshape(-1, *X.shape[2:]).to(dev); phi = P[idx].reshape(-1, P.shape[2]).to(dev)
    c = C[None].expand(len(idx), *C.shape).reshape(-1, C.shape[-1]).to(dev)     # C (M, N) species per trajectory
    assert c.shape[0] == x.shape[0]
    b, i, j, rij = neighbour_pairs(x, gnn.d0, gnn.L)
    vt, wt = truth.to(dev).messages(rij, phi[b, i], phi[b, j], c[b, i], c[b, j])
    vt = rot(vt, -phi[b, i])
    vl, wl = gnn.pair_messages(rij, phi[b, i], phi[b, j], c[b, i], c[b, j])
    out = {}
    for ci in (0, 1):
        for cj in (0, 1):
            m = (c[b, i] == ci) & (c[b, j] == cj)
            def stats(a, t):
                a, t = a[m].reshape(-1), t[m].reshape(-1)
                return dict(rel_l2=float((a - t).norm() / t.norm()), r2=float(1 - ((a - t) ** 2).sum() / ((t - t.mean()) ** 2).sum()))
            out[f'{ci}{cj}'] = dict(n=int(m.sum()), velocity=stats(vl, vt), torque=stats(wl, wt))
    v0, w0 = gnn.self_terms(torch.tensor([0, 1], device=dev))
    out['self'] = dict(v_body=v0.tolist(), w=w0.tolist(), true_v_body=[[truth.p['v0'], 0.0]] * 2, true_w=[0.0, 0.0])
    return out


# ------------------------------------------------------------------------------------------ coarse graining
def bin_density(x, c, n_bins=16, L=20.0, n_species=2):
    """x (..., N, 2), c (..., N) -> number density (..., n_species, n_bins, n_bins) (particles per unit area);
    index [s, ix, iy]"""
    h = L / n_bins
    k = torch.clamp((x / h).long(), 0, n_bins - 1)
    flat = c * n_bins * n_bins + k[..., 0] * n_bins + k[..., 1]
    lead = x.shape[:-2]
    out = torch.zeros(int(np.prod(lead)) if lead else 1, n_species * n_bins * n_bins, dtype=torch.float32, device=x.device)
    out.scatter_add_(1, flat.reshape(out.shape[0], -1), torch.ones_like(flat, dtype=torch.float32).reshape(out.shape[0], -1))
    return (out / h ** 2).reshape(*lead, n_species, n_bins, n_bins)


def face_flux(x0, x1, c, n_bins=16, L=20.0, n_species=2, dt=1.0):
    """net number of particles per unit time crossing each face, from the straight displacement x0 -> x1 (minimum
    image), routed first along x then along y.  Returns Jx[s, ix, iy] (face between bin ix and ix + 1 at row iy) and
    Jy[s, ix, iy] (face between iy and iy + 1 at column ix), shape (..., n_species, n_bins, n_bins) each."""
    h = L / n_bins
    d = wrap(x1 - x0, L)
    a = torch.floor(x0 / h).long() % n_bins
    e = torch.floor((x0 + d) / h).long()                          # unwrapped end bin
    lead = x0.shape[:-2]
    Bf = int(np.prod(lead)) if lead else 1
    a, e, cc = a.reshape(Bf, -1, 2), e.reshape(Bf, -1, 2), c.reshape(Bf, -1)
    Jx = torch.zeros(Bf, n_species, n_bins, n_bins, device=x0.device)
    Jy = torch.zeros_like(Jx)
    bidx = torch.arange(Bf, device=x0.device)[:, None].expand_as(cc)
    nx = e[..., 0] - a[..., 0]; ny = e[..., 1] - a[..., 1]
    kmax = int(max(nx.abs().max().item(), ny.abs().max().item(), 0))
    for step in range(kmax):
        # x crossings at row a_y
        for sgn in (1, -1):
            m = (sgn * nx > step)
            if m.any():
                fx = (a[..., 0] + sgn * step + (0 if sgn > 0 else -1)) % n_bins
                Jx.index_put_((bidx[m], cc[m], fx[m], a[..., 1][m]), torch.full_like(fx[m], sgn, dtype=torch.float32), accumulate=True)
            m = (sgn * ny > step)
            if m.any():
                fy = (a[..., 1] + sgn * step + (0 if sgn > 0 else -1)) % n_bins
                Jy.index_put_((bidx[m], cc[m], e[..., 0][m] % n_bins, fy[m]), torch.full_like(fy[m], sgn, dtype=torch.float32), accumulate=True)
    return Jx.reshape(*lead, n_species, n_bins, n_bins) / dt, Jy.reshape(*lead, n_species, n_bins, n_bins) / dt
