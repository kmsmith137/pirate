"""
The 1-d (frequency) regularized spline detrender.
"""

import numpy as np

from .basis import BasisTable
from .regulator import d1_banded
from .reduce import accumulate, evaluate
from .timebasis import TimeBasis
from .moments import window_moments, zone_live_counts
from .assemble import assemble, commit, bandwidth
from .solve import solve_banded, zone_slices
from .expand import expand_mask

# eta and eps are chosen JOINTLY, not independently, and the reason is a measured
# scaling law.
#
# eta is dimensionless: D_1 is defined on coefficient indices, so it is already in
# "index coordinates" and eta needs no rescaling by the knot spacing.  It also
# needs none by the time-window width -- the shrinkage bias and the residual noise
# are each W-independent.  The worst-case bias is about eta times the baseline
# amplitude, and it is UNDERSUBTRACTION, which in a rare-event search is dangerous
# at any rate however rare (see solve.py).  So eta wants to be small.
#
# What stops it is conditioning: the worst r_min over adversarial masks scales as
# eta^(4/5) at n_phi >= 2.  Measured over five configurations spanning
# nfreq = 10000 to 30000, uniform to single-knot-interval, the ratio on cutting
# eta by 3 is 0.417 to 0.421 against 0.415 predicted -- three-digit agreement.
#
# The two therefore move together.  Cutting eta from 3e-3 to 1e-3 costs a factor
# 2.4 in r_min, but eps can come down by 3.3, so every margin IMPROVES by ~1.4x
# while the median shrinkage bias roughly halves (1.06e-3 to 5.4e-4 of baseline
# amplitude).  Worst-case r_min/eps, from an offline exhaustive sweep:
#
#     config                        (3e-3, 1e-4)    (1e-3, 3e-5)
#     nfreq=10000, K=10 uniform          23.3x           32.7x
#     nfreq=10000, no interior knots      5.1x            7.1x
#     nfreq=30000, one 29900-wide         2.1x            3.0x
#     nfreq=30000, no interior knots      2.1x            3.0x
#
# LOWERING eps IS NOT FREE, which is what bounds this.  A zone surviving at
# r_min ~ eps carries a float32 error of order eps_mach/(4 r_min), and that error
# is COHERENT -- same dangerous category as shrinkage, not the harmless one.
# Measured at each setting's worst configuration it is 5.8e-6, 8.6e-6, 1.6e-5 at
# eta = 3e-3, 1e-3, 3e-4, against shrinkage biases of 1.06e-3, 5.4e-4, 1.7e-4.
# So at 1e-3 the numerical term is still ~60x below the regulator term, but by
# 3e-4 the two are within a factor of 2 of crossing.  eta = 3e-4 is the floor:
# below it one coherent error is simply traded for another.
ETA_DEFAULT = 1e-3

# eps is a threshold on the equilibrated pivot, so its natural scale is machine
# epsilon rather than anything physical -- hence one value per working precision,
# selected automatically from 'dtype' unless the caller overrides.  EPS_FLOAT64 is
# nowhere near binding (890x margin at the worst configuration above) and is left
# where it is.
EPS_FLOAT32 = 3e-5
EPS_FLOAT64 = 1e-7


def default_eps(dtype):
    return EPS_FLOAT32 if np.dtype(dtype) == np.dtype(np.float32) else EPS_FLOAT64


class SplineDetrender:
    """
    Fits a masked B-spline in frequency times a local polynomial in time, and
    subtracts it.

    Over a window of 2W+1 time samples centred on each output, the spline
    coefficients are modelled as polynomials of degree n in the offset, fitted by
    weighted least squares over the whole window and evaluated back at the centre.
    (n, W) = (0, 0) is the per-time-sample frequency fit, and is a corner of the
    same code path rather than a separate one.

    ACCUMULATE THEN SOLVE, not solve then smooth.  The distinction is invisible
    when nothing is masked and matters when something is: Gbar^-1 Ubar is not the
    average of G[t+s]^-1 U[t+s], and only the former is the least-squares
    estimator.  It also behaves better -- a time sample whose G is rank deficient
    still contributes the information it has, instead of producing a garbage
    coefficient that then gets averaged in.

    Constructor arguments:

      kv (KnotVector): the frequency basis.
      n (int): degree of the time polynomial, 0, 1 or 2.  n = 1 is exactly
        equivalent to n = 0 whenever the mask is constant over the window, and
        earns its keep only where the mask varies within the window -- see
        timebasis.py.
      W (int): window half-width; the window is 2W+1 samples.  Requires
        2W+1 >= n+1, which is the algebraic minimum (below it the time fit is
        underdetermined before any masking).  It is necessary and NOT sufficient:
        W=1, n=2 satisfies it, is exactly determined in time so the fit
        interpolates noise, and is the worst-conditioned corner measured.
        Weighing that against other considerations is the caller's business.
      eta (float): regularization strength, dimensionless, and independent of both
        n and W -- the bias and the residual noise are each W-independent, so one
        value serves every window.  A baseline that is constant in frequency within
        a zone and an arbitrary degree-n polynomial in time is removed EXACTLY at
        any eta (see regulator.py and timebasis.py).
      eps (float): conditioning threshold, defaulting to EPS_FLOAT32 or
        EPS_FLOAT64 according to 'dtype'.  A zone whose r_min falls below eps is
        masked out entirely.  A guardrail against an untrustworthy factorization,
        and deliberately NOT a test of statistical sufficiency: a zone with a
        single surviving channel is fit exactly and is not flagged.  See solve.py
        for why no such test is wanted rather than merely missing.
      dtype: float32 or float64, the working precision of the whole pipeline.
      orthogonal_time (bool): use discrete orthogonal polynomials on the window
        rather than raw monomials.  Default True and it should stay True; the
        monomial path exists to cross-check assembly, and costs a factor of up to
        5 in conditioning at n = 2 (timebasis.py).

    FOOTGUN: no offset subtraction is performed.  The constant function is exactly
    in the span, so subtracting a per-zone offset before accumulating would be
    mathematically inert, but it is what would protect float32 precision against a
    large DC level in the data.  Until that exists, feeding float32 data with a
    large offset relative to its structure will lose mantissa bits for nothing.
    """

    def __init__(self, kv, n=0, W=0, eta=ETA_DEFAULT, eps=None, dtype=np.float32,
                 orthogonal_time=True):
        n, W = int(n), int(W)
        if n not in (0, 1, 2):
            raise ValueError(f'SplineDetrender: n={n} must be 0, 1 or 2')
        if W < 0:
            raise ValueError(f'SplineDetrender: W={W} must be >= 0')
        if 2*W + 1 < n + 1:
            raise ValueError(f'SplineDetrender: a degree-{n} fit in time needs '
                             f'2W+1 >= n+1, but W={W} gives a {2*W+1}-sample window')

        self.kv = kv
        self.n, self.W = n, W
        self.eta = float(eta)
        self.dtype = np.dtype(dtype)
        self.eps = float(default_eps(self.dtype) if eps is None else eps)
        if self.eta <= 0:
            raise ValueError(f'SplineDetrender: eta={eta} must be > 0')
        if self.eps <= 0:
            raise ValueError(f'SplineDetrender: eps={eps} must be > 0')
        if self.dtype not in (np.dtype(np.float32), np.dtype(np.float64)):
            raise ValueError(f'SplineDetrender: dtype must be float32 or float64, '
                             f'got {self.dtype.name}')

        self.table = BasisTable(kv, dtype=self.dtype)
        self.D1 = d1_banded(kv, dtype=np.float64)
        self.tb = TimeBasis(n, W, orthogonal=orthogonal_time, dtype=np.float64)
        self.nband = bandwidth(kv, n)

    def __repr__(self):
        return (f'SplineDetrender(nfreq={self.kv.nfreq}, n_phi={self.kv.n_phi}, '
                f'N_phi={self.kv.N_phi}, nzone={self.kv.nzone}, n={self.n}, '
                f'W={self.W}, eta={self.eta:g}, eps={self.eps:g}, '
                f'dtype={self.dtype.name})')

    def detrend_chunk(self, d_buf, mask_buf):
        """
        d_buf, mask_buf: (M, nfreq, ntime + 2W).  'mask_buf' is read as boolean;
        d_buf is read only where it is true, so masked samples may hold anything
        including NaN.

        Returns (residual, mask_out, rmin) covering the OUTPUT region only:

          residual (M, nfreq, ntime)  in self.dtype, zero where mask_out is false
          mask_out (M, nfreq, ntime)  bool, the input mask with flagged zones cleared
          rmin     (M, ntime, nzone)  minimum relative pivot per zone, exactly 0
                                      for a zone failing the time rank test

        THE CALLER OWNS THE PADDING.  Nothing is emitted for the first or last W
        buffer samples, and supplying prepad and postpad that are consistent
        across chunks is the caller's responsibility.  That is what keeps this a
        pure function of its arguments: there is no stream-boundary policy, no
        first-chunk special case, and no carried state, so chunks may be processed
        in any order and replay is bit-reproducible.
        """
        d_buf = np.asarray(d_buf)
        mask_buf = np.asarray(mask_buf)
        if d_buf.ndim != 3:
            raise ValueError(f'SplineDetrender.detrend_chunk: expected 3-d '
                             f'(M,nfreq,ntime+2W), got shape {d_buf.shape}')
        if d_buf.shape != mask_buf.shape:
            raise ValueError(f'SplineDetrender.detrend_chunk: d.shape {d_buf.shape} '
                             f'!= mask.shape {mask_buf.shape}')
        if d_buf.shape[1] != self.kv.nfreq:
            raise ValueError(f'SplineDetrender.detrend_chunk: expected '
                             f'{self.kv.nfreq} channels, got {d_buf.shape[1]}')
        W = self.W
        ntime = d_buf.shape[2] - 2*W
        if ntime < 1:
            raise ValueError(f'SplineDetrender.detrend_chunk: buffer has '
                             f'{d_buf.shape[2]} samples, needs more than 2W={2*W}')

        G, U = accumulate(d_buf, mask_buf, self.table)
        Mcal, Vcal = window_moments(G, U, self.tb, ntime)
        live = zone_live_counts(G, self.kv, W, ntime)
        A, Ucal = assemble(Mcal, Vcal, self.kv, self.tb, self.D1, self.eta)

        alpha, rmin = solve_banded(A, Ucal, self.kv, self.n, live, self.n + 1)
        bad = rmin < self.eps
        for z, (lo, hi) in enumerate(zone_slices(self.kv)):
            Ilo, Ihi = lo*(self.n+1), hi*(self.n+1)
            alpha[..., Ilo:Ihi] = np.where(bad[..., z, None], 0, alpha[..., Ilo:Ihi])

        a = commit(alpha, self.tb, self.kv)
        model = evaluate(a, self.table)
        mask_ctr = (mask_buf[:, :, W:W+ntime] != 0)
        mask_out = expand_mask(mask_ctr, bad, self.kv)

        # Subtract only where the output mask is set.  Doing it the other way
        # round (subtract everywhere, then zero) would read d at masked channels,
        # which may be NaN.
        resid = np.where(mask_out,
                         d_buf[:, :, W:W+ntime].astype(self.dtype) - model,
                         self.dtype.type(0))
        return resid, mask_out, rmin
