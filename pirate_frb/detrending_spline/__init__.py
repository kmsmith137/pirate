"""
Pure-numpy regularized spline detrender: a masked B-spline fit along the
frequency axis, subtracted per (beam, time) sample.

The fit at each sample solves the REGULARIZED normal equations

    (G + eta D_1) a = U,   G_jl = sum_f w_f phi_j(f) phi_l(f),
                           U_j  = sum_f w_f phi_j(f) d_f,

where D_1 is a first-difference penalty on the coefficients, assembled per zone.
Regularization is what makes the problem tractable rather than a refinement of it:
without it, a coefficient whose support is entirely masked leaves a zero row in G,
and cleaning up the resulting contamination costs a large fraction of the
surviving channels.  With it, the matrix is positive definite as soon as a zone
holds one unmasked channel, and the price is a bounded bias -- roughly eta times
the baseline amplitude, and exactly zero for a constant baseline.

Arrays are (M, nfreq, ntime) throughout.  The M and T axes are pure spectators
here; they exist because the 2-d detrender will couple the time axis, and the
sufficient statistics (G, U) produced by reduce.py are already in the layout it
will need.

Worth knowing before relying on the output:

  - No offset (kappa) subtraction.  DEFERRED, not decided: inert mathematically,
    but it is what would protect float32 precision against a large DC level in
    the data.
  - No residual-degrees-of-freedom (nu) cut.  DECIDED, not a gap.  r_min is a
    numerical statistic, so a zone reduced to a single unmasked channel is fit
    exactly, leaves an identically zero residual, and is not flagged.  That is
    intended: nu measures overfitting, which in a rare-event search cannot
    manufacture a false trigger -- the residual is provably under-dispersed,
    Var[r] <= sigma^2 -- whereas undersubtraction and excess variance can, at any
    rate.  solve.py has the full argument, and it should be read before anyone
    adds a nu cut.
"""

from .knots import KnotVector
from .basis import eval_basis, BasisTable
from .regulator import d1_banded, d1_dense
from .reduce import tree_sum, accumulate, evaluate, band_to_dense, dense_to_band
from .solve import (equilibrate, forward_subst, backward_subst,
                    solve_normal_equations, solve_banded, zone_slices)
from .expand import expand_mask, zone_channel_ranges
from .SplineDetrender import (SplineDetrender, ETA_DEFAULT, EPS_FLOAT32,
                              EPS_FLOAT64, default_eps)
from .reference import detrend_reference
from .timebasis import TimeBasis
from .moments import window_moments, zone_live_counts
from .assemble import assemble, commit, bandwidth
from .masks import (random_knots, random_mask, random_mask_1d, random_mask_2d,
                    adversarial_mask, MASK_TYPES, TIME_TYPES)
