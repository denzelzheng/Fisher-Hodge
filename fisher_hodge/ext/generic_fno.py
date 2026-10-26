"""GENERIC-FNO, re-implemented from the paper (Sulskis and Ravi, "GENERIC-FNO: Embedding Energy Conservation and
Entropy Production into Fourier Neural Operators", arXiv:2606.08343v8).  Written from the paper's equations; the
authors' anonymous repository (licence not stated) was only read to settle details the paper leaves open, and no
code was copied from it.  Details taken from the repository are marked [repo].

Model (paper Sec. 4):
    E[u] = h_E( mean_x g_E(u) ),  S[u] = h_S( mean_x g_S(u) )        g: FNO backbone, h: two-layer MLP head (eq. 5)
    dE/du, dS/du by reverse-mode autodiff with create_graph=True                                       (eq. 6)
    D_L(k) = i a(k),  D_M(k) = |b(k)|^2  on the lowest `band` Fourier modes, zero elsewhere            (eq. 7)
    P_w v = <w, v> w / (<w, w> + eps),  eps = 1e-12                                                    (eq. 8)
    L = (I - P_S) D_L (I - P_S),  M = (I - P_E) D_M (I - P_E)                                          (eq. 9)
    u_{t+1} = u_t + L dE/du + M dS/du                                                                  (eq. 10)
so L dS/du = 0 and M dE/du = 0 hold identically.

Defaults (paper App. E and Table 19, sizes fixed from [repo]): functional backbone width 24, 12 Fourier modes,
3 layers, each layer GELU(InstanceNorm(spectral conv + 1x1 conv)) (the paper stresses the instance normalisation,
App. B.3); lift and projection 1x1 convolutions; head 1 -> 16 -> 1 with GELU; operator band 16 [repo];
multiplier initialisation 0.3 N(0, 1) [repo].  At 128 x 128 this gives 1,001,958 parameters (complex weights
counted once), the count of paper Table 19.  Raw autodiff gradient as the variational derivative (headline setting).
Training (App. E): AdamW, lr 1e-3 [repo], weight decay 1e-4, cosine schedule, batch 16 [repo], mean-squared
one-step loss, plus a 2-step autoregressive rollout term on a random 25 % of batches after a warm-up of 25 of 120
epochs [repo], gradient clipping 1.0, 120 epochs.

Extensions used for the density benchmark (not in the paper, which treats one scalar field):
  * n_fields > 1: u has several fields (species).  The L2 inner product of the projections runs over all fields and
    sites (the paper's definition for the state u), and each field has its own diagonal multipliers a, b
    (D_L, D_M stay diagonal in (field, mode), hence skew / positive semi-definite).
  * n_cond > 0 ("drive-channel" variant): static condition fields are concatenated to u at the input of the
    functional networks, E[u; cond], S[u; cond]; derivatives are taken with respect to u only.
  * Modes are clipped to the grid as in any FNO: min(modes, H // 2) in k_x (both signs), min(modes, W // 2 + 1) in k_y.
"""
import math
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


def _band(H, W, m):
    return min(m, H // 2), min(m, W // 2 + 1)


def _apply_diag(vf, coef, H, mx, my):
    """multiply the lowest modes of the rfft2 array vf (..., H, W//2+1) by coef (..., 2, m, m): coef[..., 0] acts on
    k_x >= 0 rows, coef[..., 1] on k_x < 0 rows; all other modes are set to zero."""
    out = torch.zeros_like(vf)
    out[..., :mx, :my] = coef[..., 0, :mx, :my] * vf[..., :mx, :my]
    out[..., H - mx:, :my] = coef[..., 1, :mx, :my] * vf[..., H - mx:, :my]
    return out


class SpectralLayer(nn.Module):
    """one FNO layer: GELU(InstanceNorm(K u + W u)), K a truncated Fourier kernel, W a 1x1 convolution."""

    def __init__(self, ch, modes):
        super().__init__()
        self.modes = modes
        self.kernel = nn.Parameter(torch.randn(2, ch, ch, modes, modes, dtype=torch.cfloat) / (ch * ch))
        self.local = nn.Conv2d(ch, ch, 1)
        self.norm = nn.InstanceNorm2d(ch)

    def forward(self, x):
        H, W = x.shape[-2:]
        mx, my = _band(H, W, self.modes)
        xf = torch.fft.rfft2(x)
        yf = torch.zeros(x.shape[0], self.kernel.shape[2], H, W // 2 + 1, dtype=xf.dtype, device=x.device)
        yf[..., :mx, :my] = torch.einsum('bcxy,cdxy->bdxy', xf[..., :mx, :my], self.kernel[0, ..., :mx, :my])
        yf[..., H - mx:, :my] = torch.einsum('bcxy,cdxy->bdxy', xf[..., H - mx:, :my], self.kernel[1, ..., :mx, :my])
        return F.gelu(self.norm(torch.fft.irfft2(yf, s=(H, W)) + self.local(x)))


class ScalarFunctional(nn.Module):
    """u (B, C_in, H, W) -> scalar (B,): FNO density, spatial mean, MLP head (paper eq. 5)."""

    def __init__(self, c_in, width=24, modes=12, layers=3):
        super().__init__()
        self.lift = nn.Conv2d(c_in, width, 1)
        self.layers = nn.ModuleList([SpectralLayer(width, modes) for _ in range(layers)])
        self.proj = nn.Sequential(nn.Conv2d(width, width, 1), nn.GELU(), nn.Conv2d(width, 1, 1))
        self.head = nn.Sequential(nn.Linear(1, 16), nn.GELU(), nn.Linear(16, 1))

    def forward(self, x):
        z = self.lift(x)
        for layer in self.layers:
            z = layer(z)
        return self.head(self.proj(z).mean((-2, -1))).squeeze(-1)


class GenericFNO(nn.Module):
    def __init__(self, n_fields=1, n_cond=0, width=24, modes=12, layers=3, band=16, eps=1e-12, init_scale=0.3,
                 detach_state=False):
        super().__init__()
        self.detach_state = detach_state          # True: no gradient through the state between rollout steps [repo]
        self.n_fields, self.n_cond, self.band, self.eps = n_fields, n_cond, band, eps
        self.E = ScalarFunctional(n_fields + n_cond, width, modes, layers)
        self.S = ScalarFunctional(n_fields + n_cond, width, modes, layers)
        self.a = nn.Parameter(init_scale * torch.randn(n_fields, 2, band, band))           # D_L = i a
        self.b = nn.Parameter(init_scale * torch.randn(n_fields, 2, 2, band, band))        # b = b[:, 0] + i b[:, 1]

    # --------------------------------------------------------------- operators
    def _DL(self, v):
        H, W = v.shape[-2:]
        mx, my = _band(H, W, self.band)
        return torch.fft.irfft2(_apply_diag(torch.fft.rfft2(v), 1j * self.a, H, mx, my), s=(H, W))

    def _DM(self, v):
        H, W = v.shape[-2:]
        mx, my = _band(H, W, self.band)
        m = self.b[:, 0] ** 2 + self.b[:, 1] ** 2
        return torch.fft.irfft2(_apply_diag(torch.fft.rfft2(v), m, H, mx, my), s=(H, W))

    def _deflate(self, v, w):
        """(I - P_w) v, inner product over fields and sites, per sample."""
        dims = tuple(range(1, v.dim()))
        return v - ((v * w).sum(dims, keepdim=True) / ((w * w).sum(dims, keepdim=True) + self.eps)) * w

    def gradients(self, u, cond=None, create_graph=True):
        if self.detach_state or not u.requires_grad:
            u = u.detach().requires_grad_(True)
        x = u if cond is None else torch.cat([u, cond], 1)
        E, S = self.E(x), self.S(x)
        dE = torch.autograd.grad(E.sum(), u, create_graph=create_graph, retain_graph=True)[0]
        dS = torch.autograd.grad(S.sum(), u, create_graph=create_graph)[0]
        return E, S, dE, dS

    def split_rhs(self, dE, dS):
        rev = self._deflate(self._DL(self._deflate(dE, dS)), dS)          # L dE/du
        irr = self._deflate(self._DM(self._deflate(dS, dE)), dE)          # M dS/du
        return rev, irr

    def forward(self, u, cond=None, create_graph=None):
        """one step u -> u + L dE/du + M dS/du.  Gradients flow through u when it requires grad (rollout training)."""
        cg = self.training if create_graph is None else create_graph
        with torch.enable_grad():
            _, _, dE, dS = self.gradients(u, cond, create_graph=cg)
            rev, irr = self.split_rhs(dE, dS)
        out = u + rev + irr
        return out if cg else out.detach()

    @torch.no_grad()
    def structure_check(self, u, cond=None):
        """the identities of paper Sec. 4.3 (normalised): r_E = |<dE, du>| / (|dE| |du|), <dS, du> >= 0, degeneracy."""
        with torch.enable_grad():
            _, _, dE, dS = self.gradients(u, cond, create_graph=False)
        rev, irr = self.split_rhs(dE, dS)
        du = rev + irr
        ip = lambda x, y: (x * y).flatten(1).sum(1)
        nrm = lambda x: x.flatten(1).norm(dim=1)
        return dict(r_E=float((ip(dE, du).abs() / (nrm(dE) * nrm(du) + 1e-30)).max()),
                    entropy_prod_min=float(ip(dS, du).min()),
                    deg_L=float((ip(dS, rev).abs() / (nrm(dS) * nrm(rev) + 1e-30)).max()),
                    deg_M=float((ip(dE, irr).abs() / (nrm(dE) * nrm(irr) + 1e-30)).max()),
                    r_mech=float((-ip(u, du) / (nrm(u) * nrm(du) + 1e-30)).mean()))


def n_params(model):
    return sum(p.numel() for p in model.parameters())


def train_generic(model, X, cond=None, steps=None, epochs=120, batch=16, lr=1e-3, wd=1e-4, warm_frac=25 / 120,
                  p_roll=0.25, roll_len=2, clip=1.0, device='cuda', seed=0, log=print, log_every=None):
    """paper App. E training.  X: (N, T+1, C, H, W) training trajectories, cond: (N, n_cond, H, W) or None.
    Each batch uses one random time index; after the warm-up, a random p_roll of the batches use a roll_len-step
    autoregressive loss instead of the one-step loss.  steps (optional) sets the number of epochs to reach about
    that many gradient steps (the warm-up and the cosine schedule scale with it)."""
    g = torch.Generator().manual_seed(10_000 + seed)
    N, T1 = X.shape[:2]
    nt = T1 - 1
    nb = math.ceil(N / batch)
    if steps is not None:
        epochs = math.ceil(steps / nb)
    warm = int(round(warm_frac * epochs))
    model.to(device).train()
    opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=wd)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, epochs)
    hist, nstep = [], 0
    log_every = log_every or max(1, epochs // 12)
    for ep in range(epochs):
        perm = torch.randperm(N, generator=g)
        tot = 0.0
        for i in range(0, N, batch):
            idx = perm[i:i + batch]
            c = None if cond is None else cond[idx].to(device)
            if ep > warm and torch.rand(1, generator=g).item() < p_roll:
                t0 = int(torch.randint(0, max(1, nt - roll_len + 1), (1,), generator=g))
                L = min(roll_len, nt - t0)
                u = X[idx, t0].to(device)
                loss = 0.0
                for k in range(L):
                    u = model(u, c)
                    loss = loss + F.mse_loss(u, X[idx, t0 + k + 1].to(device))
                loss = loss / L
            else:
                t = int(torch.randint(0, nt, (1,), generator=g))
                loss = F.mse_loss(model(X[idx, t].to(device), c), X[idx, t + 1].to(device))
            opt.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), clip)
            opt.step()
            tot += float(loss); nstep += 1
        sched.step()
        hist.append(tot / nb)
        if ep % log_every == 0 or ep == epochs - 1:
            log(f'epoch {ep:4d}/{epochs} step {nstep} loss {hist[-1]:.4e}')
    model.eval()
    return dict(epochs=epochs, steps=nstep, warmup_epochs=warm, final_loss=hist[-1], loss_curve=hist)


def rollout(model, u0, nsteps, cond=None, chunk=16, blow=1e4):
    """autoregressive rollout without graph; returns (B, nsteps+1, C, H, W) on the CPU; non-finite or |u| > blow -> nan."""
    out = []
    for s in range(0, u0.shape[0], chunk):
        u = u0[s:s + chunk]
        c = None if cond is None else cond[s:s + chunk]
        traj = [u.cpu()]
        for _ in range(nsteps):
            u = model(u, c, create_graph=False)
            bad = ~torch.isfinite(u).flatten(1).all(1) | (u.flatten(1).abs().amax(1) > blow)
            if bad.any():
                u = torch.where(bad.view(-1, *[1] * (u.dim() - 1)), torch.full_like(u, float('nan')), u)
            traj.append(u.cpu())
        out.append(torch.stack(traj, 1))
    return torch.cat(out, 0)
