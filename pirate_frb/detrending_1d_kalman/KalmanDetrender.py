"""
The chunked fixed-lag ("seam-free") Kalman detrender (see plans/detrend_1d_kalman.md,
and notes/detrending.tex,
section "Time detrending algorithm 2: Kalman filter").

The committed baseline at time t is

    fhat[t] = E[ f[t] | d[0 .. t+L] ],

so every output has its own right endpoint and the estimator does not depend on
where the chunk boundaries fall.  It is therefore seam-free by construction rather
than to O(exp(-L/ell)), which is the property detrend_stream() checks by requiring
bit-identical output across chunk decompositions.

Since x is Markov and the data split at t, the posterior factorizes into the forward
filter and the backward likelihood, and in information form the combination is pure
addition:

    J = J_f + J_b,   eta = eta_f + eta_b,   fhat[t] = (J^-1 eta)_0.

Contrast with detrending_1d, whose Detrender holds no state at all:

  - this class HAS state.  The forward filter runs once through the stream and
    carries (J_f, eta_f, kappa) per row across chunk boundaries, so chunks must be
    processed in order.  detrend_chunk() takes the state as an argument and returns
    a new one rather than mutating, so a caller can still checkpoint and replay.
  - there is no prepadding.  The buffer is (chunk_size + L) samples: the past lives
    in the state, and only the lookahead has to be supplied.

CALLER CONTRACT, and the one footgun the carried state creates: unmasked samples
must be finite.  Masked samples may hold anything -- NaN, Inf, uninitialized
memory from a dropped packet -- and are selected away rather than multiplied by
the mask, so they cannot reach the arithmetic.  A NaN in an *unmasked* sample is
different: it enters the forward filter, lands in the state, and from then on
every subsequent output of that row is NaN, forever.  There is no guard here,
deliberately -- one would cost a per-sample branch to defend against something
upstream already guarantees -- so if that guarantee ever weakens, the fix is an
isfinite() check on the state at chunk boundaries with a reset to
initial_state().  Contrast detrending_1d, which holds no state and where the
same poison would be confined to a single buffer.
"""

import numpy as np

from ..detrending_1d import LocalPolyFit
from .model import StateSpaceModel, tau_from_equivalent_W
from .InfoFilter import forward_step, backward_step


class KalmanState:
    """Per-row filter state carried across chunks: J_f, eta_f and the constant
    offset kappa that they are expressed relative to."""

    def __init__(self, J, eta, kappa):
        self.J, self.eta, self.kappa = J, eta, kappa

    def copy(self):
        return KalmanState(self.J.copy(), self.eta.copy(), self.kappa.copy())


def _tri_forward(L, b):
    """Solve L y = b, L lower triangular, batched over leading axes."""
    m = b.shape[-1]
    y = np.empty_like(b)
    for i in range(m):
        s = b[..., i].copy()
        for j in range(i):
            s = s - L[..., i, j] * y[..., j]
        y[..., i] = s / L[..., i, i]
    return y


def _tri_backward(L, y):
    """Solve L^T a = y."""
    m = y.shape[-1]
    a = np.empty_like(y)
    for i in reversed(range(m)):
        s = y[..., i].copy()
        for j in range(i+1, m):
            s = s - L[..., j, i] * a[..., j]
        a[..., i] = s / L[..., i, i]
    return a


class KalmanDetrender:
    def __init__(self, k, tau, L, chunk_size=2048, dtype=np.float32,
                 subtract_offset=True, eps=1e-3, mu=1e-30):
        """
        k, tau and L are all required: tau is the smoothing timescale in samples and
        L the lookahead, and both have operational meaning that a default would hide.
        See from_equivalent_W() for the natural way to pick them.

        Only k = 2 is supported, and anything else raises.  k stays an explicit
        argument rather than an implicit constant so that call sites say which order
        they mean, and so that lifting the restriction is a one-line change plus a
        test sweep.  The arithmetic below is generic in k -- in numpy, matrix
        expressions are both clearer than scalarizing and less code -- so it is this
        check, not the array shapes, that keeps untested configurations unreachable.

        eps is the mask-expansion threshold on rmin and mu is a NaN guard on the
        Cholesky pivots; neither is a regularizer.  subtract_offset=False pins
        kappa to zero, which is what the impulse-response and bit-identity tests
        need (a data-dependent kappa would shift under an impulse, and would differ
        between chunk decompositions).

        No relation is required between chunk_size and L.
        """
        if k != 2:
            raise ValueError(f'KalmanDetrender: k={k} but only k=2 is supported '
                             f'(see plans/detrend_1d_kalman.md, D8)')
        if L < 1:
            raise ValueError(f'KalmanDetrender: L={L} must be >= 1')
        if chunk_size < 1:
            raise ValueError(f'KalmanDetrender: chunk_size={chunk_size} must be >= 1')

        self.k = k
        self.L = int(L)
        self.chunk_size = int(chunk_size)
        self.dtype = np.dtype(dtype)
        self.eps = eps
        self.mu = mu
        self.subtract_offset = subtract_offset
        self.model = StateSpaceModel(k, tau, dtype=self.dtype)

        self.tau = self.model.tau
        self.ell = self.model.ell
        self.buflen = self.chunk_size + self.L

    @classmethod
    def from_equivalent_W(cls, k, W, n_ell=4.0, **kwargs):
        """
        Construct with tau matched to a local polynomial fit of half-width W (see
        model.tau_from_equivalent_W), and L = ceil(n_ell * ell).

        This is the entry point to use when comparing against detrending_1d: matching
        on tau alone is not a fair comparison, since the flux loss c_k/tau carries a
        k-dependent constant.
        """
        tau = tau_from_equivalent_W(k, W)
        ell = tau / np.sin(np.pi/(2*k))
        return cls(k, tau, int(np.ceil(n_ell*ell)), **kwargs)

    def __repr__(self):
        return (f'KalmanDetrender(k={self.k}, tau={self.tau:g}, L={self.L}, '
                f'ell={self.ell:.4g}, chunk_size={self.chunk_size}, '
                f'dtype={self.dtype.name})')

    # ------------------------------------------------------------------ state

    def initial_state(self, S):
        """
        Diffuse start: J = 0, eta = 0, kappa = 0.

        The diffuse prior is exact in information form -- there is no large-kappa
        approximation -- so the first outputs of a stream are the correct posterior
        given the data seen so far.  They are emitted rather than trimmed; the ones
        with fewer than k valid samples behind them are removed by the rmin cut, and
        the rest are simply poorly determined.
        """
        k = self.k
        return KalmanState(np.zeros((S, k, k), dtype=self.dtype),
                           np.zeros((S, k), dtype=self.dtype),
                           np.zeros(S, dtype=self.dtype))

    # ------------------------------------------------------------------ chunk

    def detrend_chunk(self, d_buf, mask_buf, state):
        """
        d_buf, mask_buf: shape (S, chunk_size + L), where S is a spectator axis
        carrying one entry per (beam,freq) pair.  There is no prepadding.

        Returns ((residual, mask_out, rmin), new_state), the first three of shape
        (S, chunk_size).

        mask_out is the expanded mask: a sample is dropped if its input sample was
        masked, or if J is too ill-conditioned to solve (rmin < eps).  Where mask_out
        is false, both residual and rmin are zero -- not just the residual -- so a
        consumer that forgets to check the mask cannot pick up a meaningless rmin.

        The expansion is a single pass: it never feeds back into J_f, J_b or the
        carried state, so fhat stays linear in d for a fixed input mask and both
        mask_out and rmin remain functions of that mask alone.
        """
        d_buf = np.asarray(d_buf)
        mask_buf = np.asarray(mask_buf)
        if d_buf.ndim != 2 or d_buf.shape != mask_buf.shape:
            raise ValueError('d_buf and mask_buf must be 2-d with the same shape')
        if d_buf.shape[1] != self.buflen:
            raise ValueError(f'expected buffer length {self.buflen}, got {d_buf.shape[1]}')

        k, L, Tc = self.k, self.L, self.chunk_size
        S_ax = d_buf.shape[0]
        d_buf = d_buf.astype(self.dtype, copy=False)
        mf = (mask_buf != 0)

        # ---- constant offset.  The fit reproduces constants exactly (degree 0 <=
        # k-1), so kappa is mathematically inert: it shifts fhat by exactly kappa and
        # leaves the residual unchanged.  It exists only so that the filter carries
        # numbers of size |d - kappa| rather than |d|, since the residual is a
        # cancelling difference of two |d|-sized quantities.
        #
        # Unlike detrending_1d, kappa cannot simply be recomputed per buffer and
        # forgotten: the state is expressed relative to it.  Shifting the data by
        # Delta shifts the trend, hence the level component of the state, so the mean
        # goes mu -> mu - Delta e_0 with J untouched, i.e. eta -> eta - Delta J e_0.
        # That rebasing is exact.  At stream start J = eta = 0 and it is a no-op, so
        # there is no initialization branch.  J_b needs nothing: it is rebuilt from
        # this buffer and is already in the new frame.
        if self.subtract_offset:
            kappa = self._masked_mean(d_buf, mf, state.kappa)
        else:
            kappa = np.zeros(S_ax, dtype=self.dtype)

        delta = (kappa - state.kappa).astype(self.dtype)
        Jf = state.J
        etaf = (state.eta - delta[:, None] * Jf[:, :, 0]).astype(self.dtype)

        dz = np.where(mf, d_buf - kappa[:, None], 0).astype(self.dtype)

        # ---- forward: one sequential pass, saving the post-measurement (J,eta) at
        # each output.  This is the only python-level loop over time.
        Jf_out = np.empty((S_ax, Tc, k, k), dtype=self.dtype)
        ef_out = np.empty((S_ax, Tc, k), dtype=self.dtype)
        for t in range(Tc):
            Jm, em, Jf, etaf = forward_step(Jf, etaf, mf[:, t], dz[:, t], self.model)
            Jf_out[:, t] = Jm
            ef_out[:, t] = em

        # ---- backward: the window [t+1, t+L] slides, and cannot be walked by
        # add/subtract (the recursion forgets exponentially, so its inverse
        # amplifies exponentially).  Instead every output runs its own recursion, all
        # of them in lockstep over the lag: at lag j, output t absorbs sample t+j.
        # That is L vectorized steps rather than L per output.
        Jb = np.zeros((S_ax, Tc, k, k), dtype=self.dtype)
        eb = np.zeros((S_ax, Tc, k), dtype=self.dtype)
        for j in range(L, 0, -1):
            Jb, eb = backward_step(Jb, eb, mf[:, j:j+Tc], dz[:, j:j+Tc], self.model)

        # ---- combine and solve
        J = Jf_out + Jb
        eta = ef_out + eb
        fhat, rmin = self._solve(J, eta)

        mask_out = mf[:, :Tc] & (rmin >= self.eps)
        resid = np.where(mask_out, dz[:, :Tc] - fhat, 0).astype(self.dtype)
        rmin = np.where(mask_out, rmin, 0).astype(self.dtype)

        return (resid, mask_out, rmin), KalmanState(Jf, etaf, kappa)

    # ----------------------------------------------------------------- stream

    def detrend_stream(self, d, mask, state=None):
        """
        d, mask: shape (S, T) with (T - L) a positive multiple of chunk_size.
        Returns (residual, mask_out, rmin) for outputs [0, T-L), i.e. each of shape
        (S, T-L).

        Because the estimator is seam-free, this must agree with a single
        whole-stream call sample by sample -- bit-identically for mask_out and rmin,
        which depend on the mask alone.
        """
        d = np.asarray(d)
        mask = np.asarray(mask)
        nout = d.shape[1] - self.L
        if nout <= 0 or nout % self.chunk_size != 0:
            raise ValueError(f'(T - L) = {nout} must be a positive multiple of '
                             f'chunk_size={self.chunk_size}')

        if state is None:
            state = self.initial_state(d.shape[0])
        cols = [[], [], []]
        for i in range(nout // self.chunk_size):
            lo = i * self.chunk_size
            outs, state = self.detrend_chunk(d[:, lo:lo+self.buflen],
                                             mask[:, lo:lo+self.buflen], state)
            for c, x in zip(cols, outs):
                c.append(x)
        return tuple(np.concatenate(c, axis=1) for c in cols)

    # ---------------------------------------------------------------- helpers

    def _solve(self, J, eta):
        """
        Solve J a = eta and return (fhat, rmin), where fhat = a_0.

        The leverage (J^-1)_00 is not computed; see the discussion in
        LocalPolyFit.solve().  Unlike the local polynomial fit, a penalized
        estimator shrinks rather than projects, so Var(r) needs a second number
        beyond the leverage anyway -- see notes/detrending.tex, section
        "Time detrending algorithm 2: Kalman filter", subsection "Outputs".

        The factorization is LocalPolyFit.cholesky() with J in place of the local
        fit's Gram matrix: same algorithm, same mu pivot guard, and the same
        conditioning statistic rmin = min_i p_i/J_ii, which is what the mask
        expansion thresholds on.  The adjugate form used by src_lib/Detrender1d.cu is
        deliberately not used here -- it is 0/0 on a degenerate window, and its
        division-free mask test relies on G_01 = 0, which has no analogue for J.
        """
        Lc, ratios = LocalPolyFit.cholesky(J, self.mu)
        rmin = ratios.min(axis=-1)

        a = _tri_backward(Lc, _tri_forward(Lc, eta))
        fhat = a[..., 0]
        return fhat, rmin

    def _masked_mean(self, d, mf, fallback):
        # Select with np.where rather than weighting by the mask: a masked sample may
        # hold anything at all, and 0*inf is NaN, which would poison kappa and hence
        # the carried state forever.  An empty buffer keeps the previous kappa rather
        # than collapsing to zero -- a stale offset is harmless (any kappa is exact),
        # whereas a zero one silently disables the subtraction.
        nv = mf.sum(axis=1)
        safe = nv > 0
        tot = np.where(mf, d, 0).sum(axis=1)
        return np.where(safe, tot / np.where(safe, nv, 1), fallback).astype(self.dtype)
