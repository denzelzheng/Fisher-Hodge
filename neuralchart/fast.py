"""Fast integration path (GPU and CPU) for every model with positive rates: the Neural Chart, the oracle, CRNN,
Flux-GNN, and (error control only) the UDE closure.

    from neuralchart import fast
    fast.accelerate(model, control='sample_err', rtol=1e-4, precision='fp32')   # model.advance now uses this module
    fast.restore(model)                                                          # back to neuralchart.integrators

It is the scheme and step rule of neuralchart.integrators.positive_ssp_rk2 (explicit SSP-RK2 with positivity-preserving
substeps), implemented for speed:
  * sparse operators (<nu+-, x>, S J, consumption) are padded gathers whose backward is again a padded gather over the
    transposed table (deterministic, no atomics); on the CPU they are sparse CSR products;
  * the c-independent part of the log rates (log k, A / 2, the potential) is computed once per context, so a rate
    evaluation is one gather and one exp per direction;
  * the step control runs on the device and the host only checks for completion every few substeps;
  * substeps are captured as CUDA graphs (training: one autograd node per substep, recomputed in the backward);
  * precision 'fp32' integrates in float32 while the step control (t, dt) stays in float64.

Step-size control (control=...):
  'sample_err'  one dt per trajectory: the positivity rule (dt * max_m consumption_m / c_m <= theta) plus an embedded
                error control (Euler / Heun difference, weighted RMS norm with rtol, atol = rtol).  The substep cap
                counts the accepted substeps of each trajectory per sampling interval.  A trajectory's result does not
                depend on the rest of the batch.
  'err_only'    for models without a positivity structure (the UDE closure): 'sample_err' without the positivity rule,
                i.e. plain adaptive Heun with the same error control.  Negative values are allowed and counted at the
                end of every interval (neg_report()).

Speed options (same scheme; the rounding order of batched products can change):
  compact=True        trajectories that have finished the interval leave the batch;
  compile_rates=True  the rate evaluation goes through torch.compile (needs the CPython headers);
  fuse_net=True       Flux-GNN: the edge encoder runs once per context and the gathers of one message-passing step are
                      merged.
"""
import logging
import math
import types

import numpy as np
import torch


_log = logging.getLogger('neuralchart.fast')


# ------------------------------------------------------------------------------------------ padded sparse operator
def _pad_table(rows, cols, vals, R, C):
    """entries (rows, cols, vals) of an (R, C) matrix -> padded (R, K) column indices and values, keeping the given entry
    order within a row.  A pad repeats the row's first column (column 0 for an empty row) with value 0, so no zero
    column has to be appended (the inputs are finite: log c with c > 0, fluxes)."""
    rows = np.asarray(rows, np.int64); cols = np.asarray(cols, np.int64); vals = np.asarray(vals, np.float64)
    keep = vals != 0
    rows, cols, vals = rows[keep], cols[keep], vals[keep]
    order = np.argsort(rows, kind='stable')
    rows, cols, vals = rows[order], cols[order], vals[order]
    cnt = np.bincount(rows, minlength=R)
    K = max(1, int(cnt.max()) if len(cnt) else 1)
    start = np.concatenate([[0], np.cumsum(cnt)[:-1]])
    pos = np.arange(len(rows)) - start[rows]
    idx = np.zeros((R, K), np.int64); val = np.zeros((R, K))
    idx[rows, pos] = cols; val[rows, pos] = vals
    first = idx[:, :1].copy()
    pad = np.arange(K)[None, :] >= cnt[:, None]
    idx = np.where(pad, first, idx)
    return idx, val, False, bool(np.all(vals == 1.0)) and not pad.any()


def _gather_sum(x, idx, val, pad, unit):
    """x (B, C, *F) -> (B, R, *F): out[:, r] = sum_k val[r, k] x[:, idx[r, k]]"""
    ex = (1,) * (x.dim() - 2)
    if pad:
        x = torch.cat([x, x.new_zeros((x.shape[0], 1) + tuple(x.shape[2:]))], 1)
    if idx.shape[1] == 1:
        g = x[:, idx[:, 0]]
        return g if unit else g * val[:, 0].view(-1, *ex)
    g = x[:, idx]
    return (g if unit else g * val.view(*val.shape, *ex)).sum(2)


class _SpFn(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x, fi, fv, bi, bv, fpad, funit, bpad, bunit):
        ctx.save_for_backward(fi, fv, bi, bv)
        ctx.flags = (fpad, funit, bpad, bunit)
        return _gather_sum(x, fi, fv, fpad, funit)

    @staticmethod
    def backward(ctx, g):
        fi, fv, bi, bv = ctx.saved_tensors
        fpad, funit, bpad, bunit = ctx.flags
        return (_SpFn.apply(g.contiguous(), bi, bv, fi, fv, bpad, bunit, fpad, funit),) + (None,) * 8


def _csr_apply(x, A):
    """CPU backend: A (R, C) sparse CSR times the rows of x (B, C, *F) -> (B, R, *F)"""
    B, Cn, rest = x.shape[0], x.shape[1], tuple(x.shape[2:])
    Y = torch.sparse.mm(A, x.transpose(0, 1).reshape(Cn, -1))
    return Y.reshape((A.shape[0], B) + rest).transpose(0, 1)


class _CsrFn(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x, A, AT):
        ctx.mats = (A, AT)
        return _csr_apply(x, A)

    @staticmethod
    def backward(ctx, g):
        A, AT = ctx.mats
        return _CsrFn.apply(g, AT, A), None, None


class SpMat:
    """a fixed sparse (R, C) matrix applied to the rows of x (B, C) -> (B, R); forward and backward are padded gathers."""

    def __init__(s, rows, cols, vals, R, C):
        s.R, s.C = R, C
        s.fi, s.fv, s.fpad, s.funit = _pad_table(rows, cols, vals, R, C)
        s.bi, s.bv, s.bpad, s.bunit = _pad_table(cols, rows, vals, C, R)
        s._t = {}

    def tables(s, device, dtype):
        key = (str(device), dtype)
        if key not in s._t:
            L = lambda a: torch.as_tensor(a, dtype=torch.long, device=device)
            T = lambda a: torch.as_tensor(a, dtype=dtype, device=device)
            s._t[key] = (L(s.fi), T(s.fv), L(s.bi), T(s.bv))
        return s._t[key]

    def csr(s, dtype):
        key = ('csr', dtype)
        if key not in s._t:
            mk = lambda idx, val, R, C: torch.sparse_coo_tensor(
                torch.as_tensor(np.stack([np.repeat(np.arange(R), idx.shape[1])[val.ravel() != 0], idx.ravel()[val.ravel() != 0]])),
                torch.as_tensor(val.ravel()[val.ravel() != 0], dtype=dtype), (R, C)).coalesce().to_sparse_csr()
            import warnings
            with warnings.catch_warnings():
                warnings.simplefilter('ignore')               # "sparse CSR support is in beta" and invariant-check notes
                s._t[key] = (mk(s.fi, s.fv, s.R, s.C), mk(s.bi, s.bv, s.C, s.R))
        return s._t[key]

    def __call__(s, x, tables=None):
        if not x.is_cuda:                     # CPU: sparse CSR products (MKL) are faster than gathers there
            A, AT = s.csr(x.dtype)
            return _CsrFn.apply(x, A, AT)
        fi, fv, bi, bv = tables if tables is not None else s.tables(x.device, x.dtype)
        return _SpFn.apply(x, fi, fv, bi, bv, s.fpad, s.funit, s.bpad, s.bunit)


class FastOps:
    """the sparse operators of a hypergraph: dot_p x = <nu+, x>, dot_m x = <nu-, x>, S J and the consumption of
    cat([a, b])."""

    def __init__(s, hg):
        H, M = hg.H, hg.M
        Np, Nm = hg.Np.tocoo(), hg.Nm.tocoo()
        s.dot_p = SpMat(Np.row, Np.col, Np.data, H, M)
        s.dot_m = SpMat(Nm.row, Nm.col, Nm.data, H, M)
        s.S = SpMat(hg.s_m, hg.s_h, hg.s_val, M, H)
        s.cons = SpMat(np.concatenate([hg.cp[1], hg.cm[1]]), np.concatenate([hg.cp[0], H + hg.cm[0]]),
                       np.concatenate([hg.cp[2], hg.cm[2]]), M, 2 * H)

    @staticmethod
    def of(hg):
        if getattr(hg, '_fastops', None) is None:
            hg._fastops = FastOps(hg)
        return hg._fastops


# ------------------------------------------------------------------------------------------ rate kernels
def _ratio_pos(D, c):
    """max_m D_m / c_m over c_m > 0 (MasterEquation.escape_ratio)"""
    return torch.where(c > 0, D / c, 0.0).amax(-1)


class Kernel:
    """rates(c, P) -> (a, b); ratio(c, a, b) -> (B,) escape ratio; S(J); prepare(ctx, dtype) -> P (computed once per
    context, differentiable in the parameters).  params_inside: the rates use network parameters directly, so the
    integration precision must be the parameters' dtype."""
    ude_ratio = False
    params_inside = False

    def __init__(s, model):
        s.model = model
        s.ops = FastOps.of(model.hg)

    def S(s, J):
        return s.ops.S(J)

    def ratio(s, c, a, b):
        D = s.ops.cons(torch.cat([a, b], -1))
        if s.ude_ratio:
            return (D / (c.clamp_min(0) + 0.01)).amax(-1)
        return _ratio_pos(D, c)


class ChartKernel(Kernel):
    """MasterEquation with a c-independent potential: log a = E+ + <nu+, log c>, log b = E- + <nu-, log c>, with
    E+- = log k +- A/2 + (potential terms) once per context.  Optional per-call terms: a state-dependent log k
    (kinetics.state_branch, porous media) and a site-local activity correction (free_energy.local_excess, the learned
    ln gamma of the non-ideal Schloegl model)."""

    def __init__(s, model):
        super().__init__(model)
        fe, kin = model.free_energy, model.kinetics
        s.excess = bool(fe.c_dependent and getattr(fe, 'local_excess', False))
        assert s.excess or not fe.c_dependent, 'fast path: the excess potential must not depend on c'
        s.split = bool(getattr(kin, 'split', False))
        s.state = bool(getattr(kin, 'rho_dependent', False))
        s.params_inside = s.state
        assert not s.state or hasattr(kin, 'state_branch'), 'fast path: unsupported state-dependent mobility'

    def prepare(s, ctx, dtype):
        m = s.model
        feats = ctx['feats']
        k = ctx['k'] if ctx['k'] is not None else m.kinetics(feats)          # state-dependent k: the material part only
        logk = torch.log(k)
        phi, A = ctx['Ufull'], ctx['A']
        pp, pm = s.ops.dot_p(phi), s.ops.dot_m(phi)
        if s.split:
            Ea = logk + 0.5 * A + 0.5 * (pp - pm)
            Eb = logk - 0.5 * A + 0.5 * (pm - pp)
        else:
            Ea = logk + pp + 0.5 * A
            Eb = logk + pm - 0.5 * A
        P = dict(Ea=Ea.to(dtype), Eb=Eb.to(dtype))
        if s.state:
            P.update(invV=feats['invV'].to(dtype), tail=feats['tail'], head=feats['head'])
        return P

    def rates(s, c, P):
        lc = torch.log(c)
        la = P['Ea'] + s.ops.dot_p(lc)
        lb = P['Eb'] + s.ops.dot_m(lc)
        if s.state:
            kin = s.model.kinetics
            dk = kin.state_branch(kin.rhobar_log(P, c))
            la, lb = la + dk, lb + dk
        if s.excess:
            pc = s.model.free_energy.correction(c, None)
            pp, pm = s.ops.dot_p(pc), s.ops.dot_m(pc)
            if s.split:
                la, lb = la + 0.5 * (pp - pm), lb + 0.5 * (pm - pp)
            else:
                la, lb = la + pp, lb + pm
        return torch.exp(la), torch.exp(lb)


class MassActionKernel(Kernel):
    """CRNN: a = kf c^{nu+}, b = kb c^{nu-} with per-context rate constants (log domain)."""

    def prepare(s, ctx, dtype):
        return dict(la=torch.log(ctx['kf']).to(dtype), lb=torch.log(ctx['kb']).to(dtype))

    def rates(s, c, P):
        lc = torch.log(c)
        return torch.exp(P['la'] + s.ops.dot_p(lc)), torch.exp(P['lb'] + s.ops.dot_m(lc))


class FluxGNNKernel(Kernel):
    """neuralchart.baselines.FluxGNN: the trunk with padded-gather message passing, static node / edge inputs per context,
    a = r+ c^{nu+}, b = r- c^{nu-} (mass action in the log domain, c > 0 under the positive integrator)."""

    params_inside = True

    def __init__(s, model, fuse=False):
        super().__init__(model)
        hg = model.hg
        Np, Nm = hg.Np.tocoo(), hg.Nm.tocoo()
        s.pT = SpMat(Np.col, Np.row, Np.data, hg.M, hg.H)
        s.mT = SpMat(Nm.col, Nm.row, Nm.data, hg.M, hg.H)
        # fuse=True (same values bit for bit): the edge encoder of the context-only edge inputs is evaluated once per
        # context, and the reactant / product gathers of each message-passing direction are one padded gather over the
        # stacked tables
        s.fuse = fuse
        if fuse:
            H, M = hg.H, hg.M
            s.pm = SpMat(np.concatenate([Np.row, H + Nm.row]), np.concatenate([Np.col, Nm.col]),
                         np.concatenate([Np.data, Nm.data]), 2 * H, M)
            s.pmT = SpMat(np.concatenate([Np.col, M + Nm.col]), np.concatenate([Np.row, Nm.row]),
                          np.concatenate([Np.data, Nm.data]), 2 * M, H)

    def prepare(s, ctx, dtype):
        m = s.model
        ei = m.edge_inputs(ctx)
        dummy = torch.zeros(ei.shape[0], m.hg.M, dtype=ei.dtype, device=ei.device)
        P = dict(ns=m.node_inputs(dummy, ctx)[..., 1:].to(dtype), ei=ei.to(dtype))
        if s.fuse:
            P['he0'] = m.trunk.enc_e(P['ei'])
        return P

    def rates(s, c, P):
        m = s.model
        T = m.trunk
        hn = T.enc_n(torch.cat([c[..., None] / m.RS, P['ns']], -1))
        he = P['he0'] if s.fuse else T.enc_e(P['ei'])
        L = len(T.me)
        H, M = m.hg.H, m.hg.M
        for i, (me, mn) in enumerate(zip(T.me, T.mn)):
            if s.fuse:
                g = s.pm(hn)
                he = he + me(torch.cat([he, g[:, :H], g[:, H:]], -1))
            else:
                he = he + me(torch.cat([he, s.ops.dot_p(hn), s.ops.dot_m(hn)], -1))
            if i < L - 1:                                    # the last node update does not reach the rates
                if s.fuse:
                    g = s.pmT(he)
                    hn = hn + mn(torch.cat([hn, g[:, :M], g[:, M:]], -1))
                else:
                    hn = hn + mn(torch.cat([hn, s.pT(he), s.mT(he)], -1))
        r = m.scale * torch.nn.functional.softplus(m.dec(he))
        lc = torch.log(c)
        if s.fuse:
            g = s.pm(lc)
            return r[..., 0] * torch.exp(g[:, :H]), r[..., 1] * torch.exp(g[:, H:])
        return r[..., 0] * torch.exp(s.ops.dot_p(lc)), r[..., 1] * torch.exp(s.ops.dot_m(lc))


class GenericKernel(Kernel):
    """any model with rates(c, ctx) (black boxes); the context itself is P."""

    params_inside = True

    def __init__(s, model, ude_ratio=False):
        super().__init__(model)
        s.ude_ratio = ude_ratio

    def prepare(s, ctx, dtype):
        return ctx

    def rates(s, c, P):
        return s.model.rates(c, P)


def make_kernel(model, fuse_net=False):
    from .model import MasterEquation
    from .baselines import CRNN, FluxGNN
    if isinstance(model, MasterEquation) and (not model.free_energy.c_dependent or getattr(model.free_energy, 'local_excess', False)):
        return ChartKernel(model)
    if isinstance(model, CRNN):
        return MassActionKernel(model)
    if isinstance(model, FluxGNN):
        return FluxGNNKernel(model, fuse=fuse_net)
    return GenericKernel(model, ude_ratio=getattr(model, 'free_flux', False))


# ------------------------------------------------------------------------------------------ integrator
def _compiled(fn):
    """torch.compile for a rate function: static shapes (one graph per batch shape of the compaction) and enough
    recompilations for the shapes and the gradient modes of one run"""
    import torch._dynamo as dyn
    for k in ('cache_size_limit', 'recompile_limit', 'accumulated_cache_size_limit', 'accumulated_recompile_limit'):
        if hasattr(dyn.config, k):
            setattr(dyn.config, k, max(getattr(dyn.config, k), 256))
    return torch.compile(fn, dynamic=False)


class Stepper:
    """adaptive positive SSP-RK2 over one interval Dt with device-side step control (see the module docstring)."""

    def __init__(s, kern, control='sample_err', rtol=1e-4, atol=None, compile=False, check_max=32, graphs=True,
                 compact=False, compile_rates=False):
        assert control in ('sample_err', 'err_only')
        s.kern, s.control = kern, control
        # compact=True: trajectories that have finished the interval leave the batch, so a substep costs the work of the
        # trajectories still integrating; per trajectory the result is that of compact=False up to the reduction order
        # of batched products
        s.compact = compact
        s.compact_min_iters = 8          # compact only when the slowest remaining trajectory needs at least this many
        s.cap_hits, s.cap_hit_traj, s.last_iters = 0, 0, 0
        s.rtol = rtol
        s.atol = rtol if atol is None else atol
        s.check_max = check_max
        s.compile = compile
        # compile_rates=True: the rate evaluation goes through torch.compile (inductor), inside the CUDA graphs; needs
        # the CPython headers.  The rounding order changes.
        s.compile_rates = compile_rates
        s._rates = _compiled(kern.rates) if compile_rates else kern.rates
        s.graphs = graphs
        s._fn = None
        s._runners = {}
        s.reset_neg()

    # one loop iteration (a substep attempt); pure tensor function, no host synchronisation
    def _iter(s, c, t, cap, nsub, h, P, gid, G, Dt, dt_max, theta0, theta1, max_sub, t_tol):
        kern = s.kern
        pos = s.control != 'err_only'                                   # the positivity rule
        a0, b0 = s._rates(c, P)
        f0 = kern.S(a0 - b0)
        with torch.no_grad():
            if pos:
                r0 = kern.ratio(c, a0, b0).double()
            rem = Dt - t
            if pos:
                dt = torch.minimum(torch.minimum(rem, cap), torch.where(r0 > 0, theta0 / r0, math.inf))
            else:
                dt = rem
            dt = torch.minimum(dt, h)
            dt = torch.clamp(dt, max=dt_max)
            active = rem > t_tol
            active = active & (nsub < max_sub)
            dt = torch.where(active, dt, 0.0)
            dts = dt
        dtc = dts.to(c.dtype)[:, None]
        c1 = c + dtc * f0
        a1, b1 = s._rates(c1, P)
        c2 = c1 + dtc * kern.S(a1 - b1)
        cn = 0.5 * (c + c2)
        with torch.no_grad():
            if pos:
                r1 = kern.ratio(c1, a1, b1).double()
                ok = (dt * r1 <= theta1) | (dt < 1e-12)
            else:
                ok = torch.ones_like(active)
            sc = s.atol + s.rtol * torch.maximum(c.abs(), cn.abs())
            en = ((cn - c1) / sc).square().mean(-1).sqrt().double()
            eok = (en <= 1.0) | (dt < 1e-12)
            fac = torch.clamp(0.9 * en.clamp_min(1e-12).rsqrt(), 0.2, 5.0)
            hn = dt * fac
            # accepted: next proposal from the error; a step shortened by the interval end / positivity keeps h
            h_acc = torch.where(dt < h, torch.maximum(h, hn), hn)
            h = torch.where(active & ok, torch.where(eok, h_acc, hn), h).clamp(max=dt_max)
            ok_all = ok & eok
            take = ok_all & active
            tk = take
        c = torch.where(tk[:, None], cn, c)
        with torch.no_grad():
            t = t + torch.where(take, dt, 0.0)
            cap = torch.where(ok, math.inf, 0.5 * dt)
            nsub = nsub + take.long()
            nxt = active & ((Dt - t) > t_tol)
            nxt = nxt & (nsub < max_sub)
            est = torch.where(nxt, ((Dt - t) / dt.clamp_min(1e-300)).clamp_min(1.0), 0.0)
            dteff = torch.where(tk, dts, -1.0)                              # per sample: dt if taken, -1 otherwise
        return c, t, cap, nsub, h, est, dteff

    # err_only: negative concentrations at the interval ends, counted on the device (no host synchronisation)
    def reset_neg(s):
        s._neg = None

    @torch.no_grad()
    def record_neg(s, c):
        m = c.detach().amin(-1).double()
        x = torch.stack([(m < 0).double().sum(), torch.full((), float(m.numel()), dtype=torch.float64, device=m.device),
                         m.min().clamp(max=0.0)])
        if s._neg is None:
            s._neg = x
        else:
            s._neg = torch.stack([s._neg[0] + x[0], s._neg[1] + x[1], torch.minimum(s._neg[2], x[2])])

    def neg_report(s):
        """(trajectory-intervals ending with a negative concentration, trajectory-intervals, most negative value)"""
        if s._neg is None:
            return 0, 0, 0.0
        n, N, m = s._neg.tolist()
        return int(n), int(N), m

    def _recompute(s, c, P, dteff):
        """the substep with given per-sample dt (dteff >= 0: taken, -1: kept); same values as _iter"""
        kern = s.kern
        tk = dteff >= 0
        dtc = dteff.clamp_min(0).to(c.dtype)[:, None]
        a0, b0 = s._rates(c, P)
        c1 = c + dtc * kern.S(a0 - b0)
        a1, b1 = s._rates(c1, P)
        c2 = c1 + dtc * kern.S(a1 - b1)
        return torch.where(tk[:, None], 0.5 * (c + c2), c)

    def set_kernel(s, kern):
        """a new rate kernel (the model was rebound to another grid in place): the compiled rates, the compiled
        substep and the CUDA graph runners belong to the old one; the counters (cap_hits, negative values) are kept"""
        s.kern = kern
        s._rates = _compiled(kern.rates) if s.compile_rates else kern.rates
        s._fn = None
        s._runners = {}

    def _get_fn(s):
        if s._fn is None:
            s._fn = _compiled(s._iter) if s.compile else s._iter
        return s._fn

    def advance(s, c, P, Dt, dt_max=None, theta0=0.5, theta1=0.5, max_sub=4000, t_tol=1e-9, chunk=None, h0=None, **_):
        dt_max = Dt / 8 if dt_max is None else dt_max
        B, dev = c.shape[0], c.device
        gid, G = None, B
        t = torch.zeros(G, dtype=torch.float64, device=dev)
        cap = torch.full((G,), math.inf, dtype=torch.float64, device=dev)
        nsub = torch.zeros(G, dtype=torch.long, device=dev)
        h = (h0.clone() if h0 is not None else torch.full((G,), float(dt_max), dtype=torch.float64, device=dev))
        it_stop = 8 * max_sub + 64                          # safety only (rejections are finite)
        consts = (gid, G, Dt, dt_max, theta0, theta1, max_sub, t_tol)
        it, n = 0, max(1, int(round(Dt / dt_max)))
        if s.compact:
            return s._advance_compact(c, P, Dt, dt_max, theta0, theta1, max_sub, t_tol, h, it_stop, n)
        if s.graphs and c.is_cuda:
            need_grad = torch.is_grad_enabled() and (c.requires_grad or any(x.requires_grad for x in _flatten(P, [])) or
                                                     any(p.requires_grad for p in s.params()))
            r = s._runner(c, P, consts, need_grad)
        else:
            r = None
        if r is not None:
            r.load(c, P, h0, dt_max)
            gin = [x for i, x in enumerate(_flatten(P, [])) if i in r.grad_idx] + list(r.params) if need_grad else []
            while True:
                n = min(n, it_stop - it)
                if need_grad:
                    for _ in range(n):
                        c = _GraphedStep.apply(r, c, *gin)
                else:
                    k = n
                    while k >= r.multi:
                        r.fwdn.replay(); k -= r.multi
                    for _ in range(k):
                        r.fwd.replay()
                it += n
                if it >= it_stop:
                    break
                e = float(r.est.max())
                if not e > 0.0:
                    break
                n = max(1, min(s.check_max, int(math.ceil(e - 1e-9))))
            if not need_grad:
                c = r.c.clone()
            s._cap_check(r.t, Dt, t_tol, max_sub, it)
            return c, r.nsub.clone(), r.h.clone()
        fn = s._get_fn()
        while True:
            n = min(n, it_stop - it)
            for _ in range(n):
                c, t, cap, nsub, h, est, _ = fn(c, t, cap, nsub, h, P, *consts)
                it += 1
            if it >= it_stop:
                break
            e = float(est.max())                                # the one host synchronisation per check
            if not e > 0.0:
                break
            n = max(1, min(s.check_max, int(math.ceil(e - 1e-9))))
        s._cap_check(t, Dt, t_tol, max_sub, it)
        return c, nsub, h

    def _advance_compact(s, c, P, Dt, dt_max, theta0, theta1, max_sub, t_tol, h, it_stop, n):
        """advance with batch compaction (per-trajectory semantics): at every host check the trajectories that have
        finished the interval (or reached the cap) are set aside and the loop continues on the others only.  The batch
        is compacted when at least half of it has finished (so at most log2(B) batch shapes per interval, each with its
        own CUDA graphs), and the next check is placed at the predicted finish of half of the remaining trajectories."""
        B, dev = c.shape[0], c.device
        idx = torch.arange(B, device=dev)
        t = torch.zeros(B, dtype=torch.float64, device=dev)
        cap = torch.full((B,), math.inf, dtype=torch.float64, device=dev)
        nsub = torch.zeros(B, dtype=torch.long, device=dev)
        need_grad = torch.is_grad_enabled() and (c.requires_grad or any(x.requires_grad for x in _flatten(P, [])) or
                                                 any(p.requires_grad for p in s.params()))
        use_graphs = s.graphs and c.is_cuda
        fn = s._get_fn()
        parts = []                                         # (original rows, c, t, nsub, h) of trajectories set aside
        it = 0
        while True:
            Bc = c.shape[0]
            consts = (None, Bc, Dt, dt_max, theta0, theta1, max_sub, t_tol)
            r = s._runner(c, P, consts, need_grad) if use_graphs else None
            n = min(n, it_stop - it)
            if r is not None:
                r.load_state(c, P, t, cap, nsub, h)
                gin = [x for i, x in enumerate(_flatten(P, [])) if i in r.grad_idx] + list(r.params) if need_grad else []
                if need_grad:
                    for _ in range(n):
                        c = _GraphedStep.apply(r, c, *gin)
                else:
                    k = n
                    while k >= r.multi:
                        r.fwdn.replay(); k -= r.multi
                    for _ in range(k):
                        r.fwd.replay()
                    c = r.c.clone()
                t, cap, nsub, h, est = r.t.clone(), r.cap.clone(), r.nsub.clone(), r.h.clone(), r.est.clone()
            else:
                for _ in range(n):
                    c, t, cap, nsub, h, est, _ = fn(c, t, cap, nsub, h, P, *consts)
            it += n
            if it >= it_stop:
                break
            estc = est.cpu()                               # the host synchronisation of this check
            act = estc > 0
            nact = int(act.sum())
            if nact == 0:
                break
            ea = torch.sort(estc[act]).values
            if nact <= Bc // 2 and float(ea[-1]) >= s.compact_min_iters:
                keep = torch.nonzero(act).squeeze(1).to(dev)
                drop = torch.nonzero(~act).squeeze(1).to(dev)
                parts.append((idx[drop], c.index_select(0, drop), t[drop], nsub[drop], h[drop]))
                idx, c = idx[keep], c.index_select(0, keep)
                t, cap, nsub, h, est = t[keep], cap[keep], nsub[keep], h[keep], est[keep]
                P = _rebuild(P, iter([x.index_select(0, keep) if (x.dim() > 0 and x.shape[0] == Bc) else x
                                      for x in _flatten(P, [])]))
            # next check: when half of the remaining trajectories are predicted to finish if the batch is unbalanced
            # (the slowest needs more than twice the median), otherwise when all are (as without compaction)
            med, mx = float(ea[(len(ea) - 1) // 2]), float(ea[-1])
            n = max(1, min(s.check_max, int(math.ceil((med if mx > 2 * med else mx) - 1e-9))))
        parts.append((idx, c, t, nsub, h))
        order = torch.argsort(torch.cat([p_[0] for p_ in parts]))
        cat = lambda j: torch.cat([p_[j] for p_ in parts]).index_select(0, order)
        c_out, t_out, ns_out, h_out = cat(1), cat(2), cat(3), cat(4)
        s._cap_check(t_out, Dt, t_tol, max_sub, it)
        return c_out, ns_out, h_out

    def _cap_check(s, t, Dt, t_tol, max_sub, it):
        """a DT that ends before Dt (substep cap reached) is logged and counted, never silent"""
        s.last_iters = it
        k = int(((Dt - t) > t_tol).sum())
        if k:
            s.cap_hits += 1
            s.cap_hit_traj += k
            _log.warning(f'substep cap {max_sub} reached: {k} trajectories did not complete the interval '
                         f'{Dt:g} ({s.cap_hits} capped intervals so far, control {s.control})')

    def params(s):
        m = s.kern.model
        return [p for p in m.parameters() if p.requires_grad] if hasattr(m, 'parameters') else []

    def _runner(s, c, P, consts, need_grad):
        """one GraphRunner per (shape, dtype, step constants, gradient)"""
        key = (tuple(c.shape), c.dtype, str(c.device), need_grad, consts[1:],
               tuple((tuple(x.shape), x.dtype) for x in _flatten(P, [])))
        if key in s._runners:
            return s._runners[key]
        if len(s._runners) >= 64:
            s._runners.pop(next(iter(s._runners)))
        try:
            r = GraphRunner(s, c, P, consts, s.params() if need_grad else [], need_grad)
        except Exception as e:                 # e.g. a cuBLAS path that cannot be captured: run this shape eagerly
            torch.cuda.synchronize()
            _log.warning(f'CUDA graph capture failed for shape {tuple(c.shape)} ({type(e).__name__}); eager substeps')
            r = None
        s._runners[key] = r
        return r


# ------------------------------------------------------------------------------------------ CUDA graphs
def _flatten(obj, out):
    """tensors inside nested dicts (a context), in a fixed order"""
    if torch.is_tensor(obj):
        out.append(obj)
    elif isinstance(obj, dict):
        for k in obj:
            _flatten(obj[k], out)
    return out


def _rebuild(obj, it):
    if torch.is_tensor(obj):
        return next(it)
    if isinstance(obj, dict):
        new = type(obj)() if type(obj) is not dict else {}
        for k in obj:
            new[k] = _rebuild(obj[k], it)
        return new
    return obj


class _NoCheckpoint:
    """inside a graphed substep the whole substep is recomputed in the backward, so activation checkpointing of the
    rate evaluations (the baselines' checkpointed rates) is switched off there"""
    def __enter__(s):
        import torch.utils.checkpoint as U
        s.old = U.checkpoint
        U.checkpoint = lambda fn, *a, use_reentrant=None, **k: fn(*a)

    def __exit__(s, *exc):
        import torch.utils.checkpoint as U
        U.checkpoint = s.old


class GraphRunner:
    """CUDA graphs of one substep (and of `multi` chained substeps for rollouts without gradient) for one shape.
    Training: every substep is a custom autograd node; the forward replays the substep graph and keeps only the
    input state and the per-sample (dt, taken) of the substep, the backward replays a graph that recomputes the
    substep and differentiates it (activation checkpointing at substep granularity, values identical to eager)."""

    def __init__(s, stepper, c, P, consts, params, need_grad, multi=8):
        s.st, s.consts, s.multi = stepper, consts, multi
        gid, G = consts[0], consts[1]
        dev = c.device
        s.P_struct = P
        s.Pl = [x.detach().clone() for x in _flatten(P, [])]
        s.Ps = _rebuild(P, iter(s.Pl))
        s.c = c.detach().clone()
        s.t = torch.zeros(G, dtype=torch.float64, device=dev)
        s.cap = torch.full((G,), math.inf, dtype=torch.float64, device=dev)
        s.nsub = torch.zeros(G, dtype=torch.long, device=dev)
        s.h = torch.zeros(G, dtype=torch.float64, device=dev)
        s.est = torch.zeros(G, dtype=torch.float64, device=dev)
        s.dteff = torch.zeros(c.shape[0], dtype=torch.float64, device=dev)
        s.pool = torch.cuda.graph_pool_handle()
        s.params = params
        s.fwd = s._capture(s._body_fwd)
        s.fwdn = s._capture(lambda: [s._body_fwd() for _ in range(multi)]) if not need_grad else None
        s.bwd = None
        if need_grad:
            s.gc = torch.zeros_like(s.c)
            s.gin = torch.zeros_like(s.c)
            s.grad_idx = [i for i, x in enumerate(_flatten(P, [])) if x.requires_grad and x.is_floating_point()]
            s.gP = [torch.zeros_like(s.Pl[i]) for i in s.grad_idx]
            if params:                     # keep the parameters that the substep itself uses (others keep grad None)
                with torch.enable_grad(), _NoCheckpoint():
                    s.dteff.zero_()
                    cn = stepper._recompute(s.c.detach().requires_grad_(True), s.Ps, s.dteff)
                    gs = torch.autograd.grad(cn.sum(), params, allow_unused=True)
                s.params = [p for p, g in zip(params, gs) if g is not None]
            names = {id(p): n for n, p in stepper.kern.model.named_parameters()}
            s.pnames = [(names[id(p)], p) for p in s.params]
            s.gpar = [torch.zeros_like(p) for p in s.params]
            s.bwd = s._capture(s._body_bwd)

    def _body_fwd(s):
        with torch.no_grad():
            out = s.st._iter(s.c, s.t, s.cap, s.nsub, s.h, s.Ps, *s.consts)
            for dst, src in zip((s.c, s.t, s.cap, s.nsub, s.h, s.est, s.dteff), out):
                dst.copy_(src)

    def _body_bwd(s):
        from torch.nn.utils.stateless import _reparametrize_module
        with torch.enable_grad(), _NoCheckpoint():
            c = s.c.detach().requires_grad_(True)
            Pl = [x.detach().requires_grad_(i in s.grad_idx) for i, x in enumerate(s.Pl)]
            P = _rebuild(s.P_struct, iter(Pl))
            # the parameters enter through aliases created here (leaves of the captured stream; same storage)
            alias = {n: p.detach().requires_grad_(True) for n, p in s.pnames}
            with _reparametrize_module(s.st.kern.model, alias):
                cn = s.st._recompute(c, P, s.dteff)
            ins = [c] + [Pl[i] for i in s.grad_idx] + list(alias.values())
            gs = torch.autograd.grad(cn, ins, s.gin, allow_unused=True)
        for dst, g in zip([s.gc] + s.gP + s.gpar, gs):
            if g is None:
                dst.zero_()
            else:
                dst.copy_(g)

    def _capture(s, body):
        side = torch.cuda.Stream()
        side.wait_stream(torch.cuda.current_stream())
        with torch.cuda.stream(side):
            for _ in range(2):
                body()
        torch.cuda.current_stream().wait_stream(side)
        g = torch.cuda.CUDAGraph()
        with torch.cuda.graph(g, pool=s.pool):
            body()
        return g

    def load_state(s, c, P, t, cap, nsub, h):
        """load a running state (batch compaction): c, P and the per-trajectory t, cap, nsub, h"""
        s.load(c, P, h, None)
        s.t.copy_(t); s.cap.copy_(cap); s.nsub.copy_(nsub)
        s.loaded_src = _flatten(P, [])                 # the backward of these substeps restores this context

    loaded_src = None

    def load(s, c, P, h0, dt_max):
        for dst, src in zip(s.Pl, _flatten(P, [])):
            dst.copy_(src.detach())
        s.c.copy_(c.detach())
        s.t.zero_(); s.cap.fill_(math.inf); s.nsub.zero_()
        if h0 is not None:
            s.h.copy_(h0)
        else:
            s.h.fill_(float(dt_max))


class _GraphedStep(torch.autograd.Function):
    @staticmethod
    def forward(ctx, runner, c, *grad_inputs):
        runner.fwd.replay()
        ctx.runner = runner
        ctx.P_src = runner.loaded_src          # batch compaction: the (sub-batch) context this substep ran with
        ctx.save_for_backward(c, runner.dteff.clone())
        return runner.c.clone()

    @staticmethod
    def backward(ctx, g):
        r = ctx.runner
        c, dteff = ctx.saved_tensors
        if ctx.P_src is not None and ctx.P_src is not r.loaded_src:
            # with compaction one runner (batch shape) serves different sub-batches within one forward pass: restore
            # the context of this substep before recomputing it
            for dst, src in zip(r.Pl, ctx.P_src):
                dst.copy_(src.detach())
            r.loaded_src = ctx.P_src
        r.c.copy_(c); r.dteff.copy_(dteff); r.gin.copy_(g)
        r.bwd.replay()
        return (None, r.gc.clone()) + tuple(x.clone() for x in r.gP) + tuple(x.clone() for x in r.gpar)


# ------------------------------------------------------------------------------------------ model wrapper
_DT = dict(fp32=torch.float32, fp64=torch.float64)


class FastConfig:
    def __init__(s, model, control, precision, compile, rtol, atol, step, graphs, compact=False, compile_rates=False,
                 fuse_net=False):
        s.kw = dict(control=control, precision=precision, compile=compile, rtol=rtol, atol=atol, step=step,
                    graphs=graphs, compact=compact, compile_rates=compile_rates, fuse_net=fuse_net)
        s.kern = make_kernel(model, fuse_net=fuse_net)
        s.hg = getattr(model, 'hg', None)          # the grid the kernel was built for (see _fast_advance)
        s.stepper = Stepper(s.kern, control, rtol, atol, compile, graphs=graphs, compact=compact, compile_rates=compile_rates)
        s.precision = precision
        s.step = step


def _fast_advance(self, c, drive, Dt, **kw):
    cfg = self._fast
    hg = getattr(self, 'hg', None)
    if hg is not None and hg is not cfg.hg:
        # the model was rebound to another grid in place: the sparse operators and everything built on them are
        # rebuilt for the new grid, the step control and counters stay
        cfg.kern = make_kernel(self, fuse_net=cfg.kw['fuse_net'])
        cfg.stepper.set_kernel(cfg.kern)
        cfg.hg = hg
    ctx = self.context(drive)
    st = dict(cfg.step if cfg.step is not None else self.step)
    st.update({k: v for k, v in kw.items() if k in ('dt_max', 'theta0', 'theta1', 'max_sub', 't_tol', 'chunk')})
    dtype = _DT[cfg.precision] if cfg.precision else c.dtype
    cache = ctx.__dict__.setdefault('_fast_cache', {})
    key = ('P', dtype, id(cfg), id(cfg.kern))
    if key not in cache:
        cache[key] = cfg.kern.prepare(ctx, dtype)
    hkey = ('h', id(cfg))
    c, ns, h = cfg.stepper.advance(c.to(dtype), cache[key], Dt, h0=cache.get(hkey), **st)
    cache[hkey] = h
    if cfg.stepper.control == 'err_only':
        cfg.stepper.record_neg(c)
    return c, ns


def accelerate(model, control='sample_err', precision=None, compile=False, rtol=1e-4, atol=None, step=None,
               graphs=True, compact=False, compile_rates=False, fuse_net=False):
    """route model.advance through this module.  precision None keeps the dtype of the state, 'fp32' / 'fp64' set it.
    step: the step settings (a dict read at every call, default model.step).  control='err_only' is only allowed for
    models without a positivity structure (model.free_flux = True)."""
    if control == 'err_only' and not getattr(model, 'free_flux', False):
        raise ValueError(f"control='err_only' is for models without a positivity structure (free_flux = True); "
                         f"{type(model).__name__} keeps the positivity rule (use 'sample_err')")
    cfg = FastConfig(model, control, precision, compile, rtol, atol, step, graphs, compact, compile_rates, fuse_net)
    if precision is not None and cfg.kern.params_inside:
        pd = {p.dtype for p in model.parameters() if p.is_floating_point()}
        if pd and pd != {_DT[precision]}:
            raise ValueError(f'{type(model).__name__}: precision {precision} needs the parameters in that dtype '
                             f'(model.to({_DT[precision]}) first); they are {pd}')
    model._fast = cfg
    model.advance = types.MethodType(_fast_advance, model)
    return model


def restore(model):
    for k in ('advance', '_fast'):
        model.__dict__.pop(k, None)
    return model


def supports(model):
    """models whose advance() is the positive SSP-RK2 of neuralchart.integrators"""
    from .model import MasterEquation
    from .baselines import FluxGNN, CRNN
    return isinstance(model, (MasterEquation, FluxGNN, CRNN)) or getattr(model, 'free_flux', False)


def copy_config(src, dst):
    """accelerate dst like src (e.g. a model rebound to another grid); no-op when src is not accelerated"""
    cfg = getattr(src, '_fast', None)
    if cfg is not None and dst is not src and supports(dst):
        accelerate(dst, **cfg.kw)
    return dst


def is_fast(model):
    return getattr(model, '_fast', None) is not None


def add_args(p, rtol):
    """command line options of the benchmarks"""
    p.add_argument('--fast', action='store_true', help='the fast integrator (neuralchart.fast)')
    p.add_argument('--control', default='sample_err', choices=['sample_err', 'err_only'],
                   help='step-size control of --fast (err_only: error control without the positivity rule, '
                        'for models without a positivity structure)')
    p.add_argument('--rtol', type=float, default=rtol, help='error-control tolerance (relative; atol = rtol)')
    p.add_argument('--precision', default='fp32', choices=['fp32', 'fp64'], help='integration precision of --fast')
    p.add_argument('--no_graphs', action='store_true', help='--fast without CUDA graphs')
    p.add_argument('--compact', action='store_true', help='--fast: finished trajectories leave the batch')
    p.add_argument('--compile_rates', action='store_true', help='--fast: torch.compile the rate evaluation (needs Python.h)')
    p.add_argument('--fuse_net', action='store_true', help='--fast, Flux-GNN: edge encoder once per context, merged gathers')


def kwargs_from_args(a):
    if not getattr(a, 'fast', False):
        return None
    kw = dict(control=a.control, rtol=a.rtol, precision=a.precision, graphs=not a.no_graphs)
    for k in ('compact', 'compile_rates', 'fuse_net'):
        if getattr(a, k, False):
            kw[k] = True
    return kw


def cap_report(model):
    st = getattr(getattr(model, '_fast', None), 'stepper', None)
    if st is None:
        return ''
    out = f' | capped DT {st.cap_hits}'
    if st.control == 'err_only':
        n, N, m = st.neg_report()
        out += f' | negative {n}/{N} (min {m:.3g})'
    return out
