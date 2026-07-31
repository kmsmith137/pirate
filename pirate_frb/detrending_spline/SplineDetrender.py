"""
The 1-d (frequency) regularized spline detrender.
"""

import numpy as np

from .basis import BasisTable
from .regulator import d1_banded
from .reduce import accumulate, evaluate
from .solve import solve_normal_equations
from .expand import expand_mask

# Defaults.  eta is dimensionless: because D_1 is defined on coefficient indices
# it is already in "index coordinates", so eta needs no rescaling by the knot
# spacing, and the worst-case shrinkage bias is about eta times the baseline
# amplitude.  It also needs no rescaling by the time-window width when the 2-d
# detrender arrives: the bias and the residual noise are both W-independent.
#
# 3e-3 rather than 1e-3: at 1e-3 the worst adversarial mask at nfreq = 30000 with
# a single very wide knot interval reaches r_min = 8.9e-5, below EPS_FLOAT32, so
# the zone expander fires on a fit that is in fact perfectly usable.  3e-3 keeps
# that case at 2.1e-4, and costs a factor 3 in shrinkage bias.
ETA_DEFAULT = 3e-3

# eps is a pure conditioning threshold on the equilibrated pivot, so its natural
# scale is machine epsilon, not anything physical -- hence one value per working
# precision rather than one value.  Selected automatically from 'dtype' unless
# the caller overrides.
EPS_FLOAT32 = 1e-4
EPS_FLOAT64 = 1e-7


def default_eps(dtype):
    return EPS_FLOAT32 if np.dtype(dtype) == np.dtype(np.float32) else EPS_FLOAT64


class SplineDetrender:
    """
    Fits a masked B-spline to each frequency spectrum and subtracts it.

    Construct with a KnotVector; call detrend() on (M, nfreq, ntime) arrays.  The
    M and T axes are pure spectators -- the fit at one (beam, time) sample depends
    on nothing else -- but they are carried through the whole pipeline because the
    2-d detrender will couple the time axis, and the sufficient statistics (G, U)
    that reduce.py produces are already in the layout it needs.

    Constructor arguments:

      kv (KnotVector): the frequency basis.
      eta (float): regularization strength, dimensionless.  Larger eta is more
        stable but shrinks the fit more; the worst-case bias left in the residual
        is roughly eta times the baseline amplitude.  A CONSTANT baseline is
        removed exactly at any eta (regulator.py), so the bias only involves
        baseline structure, not its mean level.
      eps (float): conditioning threshold, defaulting to EPS_FLOAT32 or
        EPS_FLOAT64 according to 'dtype'.  A zone whose r_min falls below eps is
        masked out entirely.  This is a guardrail against a numerically untrustworthy
        factorization; it is NOT a test of whether the fit is statistically
        meaningful, and in particular it will not fire for a zone with a single
        surviving channel (see solve.py).
      dtype: float32 or float64, the working precision of the whole pipeline.

    FOOTGUN: no offset subtraction is performed.  The constant function is exactly
    in the span, so subtracting a per-zone offset before accumulating would be
    mathematically inert, but it is what would protect float32 precision against a
    large DC level in the data.  Until that exists, feeding float32 data with a
    large offset relative to its structure will lose mantissa bits for nothing.
    """

    def __init__(self, kv, eta=ETA_DEFAULT, eps=None, dtype=np.float32):
        self.kv = kv
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

    def __repr__(self):
        return (f'SplineDetrender(nfreq={self.kv.nfreq}, n_phi={self.kv.n_phi}, '
                f'N_phi={self.kv.N_phi}, nzone={self.kv.nzone}, eta={self.eta:g}, '
                f'eps={self.eps:g}, dtype={self.dtype.name})')

    def detrend(self, d, mask):
        """
        d, mask: (M, nfreq, ntime).  'mask' is read as boolean; d is read only
        where it is true, so masked samples may hold anything including NaN.

        Returns (residual, mask_out, rmin):

          residual (M, nfreq, ntime)  in self.dtype, zero wherever mask_out is
                                      false -- masked channels carry no residual
          mask_out (M, nfreq, ntime)  bool, the input mask with flagged zones cleared
          rmin     (M, ntime, nzone)  minimum relative Cholesky pivot per zone,
                                      exactly 0 for a zone with no unmasked channel
        """
        d = np.asarray(d)
        mask = np.asarray(mask)
        if d.ndim != 3:
            raise ValueError(f'SplineDetrender.detrend: expected 3-d (M,nfreq,ntime), '
                             f'got shape {d.shape}')
        if d.shape != mask.shape:
            raise ValueError(f'SplineDetrender.detrend: d.shape {d.shape} != '
                             f'mask.shape {mask.shape}')
        if d.shape[1] != self.kv.nfreq:
            raise ValueError(f'SplineDetrender.detrend: expected {self.kv.nfreq} '
                             f'channels, got {d.shape[1]}')

        G, U = accumulate(d, mask, self.table)
        a, rmin, bad = solve_normal_equations(G, U, self.kv, self.D1,
                                              self.eta, self.eps)
        model = evaluate(a, self.table)
        mask_out = expand_mask(mask, bad, self.kv)

        # Subtract only where the output mask is set.  Doing it the other way round
        # (subtract everywhere, then zero) would read d at masked channels, which
        # may be NaN.
        resid = np.where(mask_out, d.astype(self.dtype) - model,
                         self.dtype.type(0))
        return resid, mask_out, rmin
