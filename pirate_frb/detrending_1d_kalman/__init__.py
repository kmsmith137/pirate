"""
Pure-numpy fixed-lag ("seam-free") Kalman detrender: a masked 1-d detrender whose
committed baseline at time t is E[f[t] | d[0..t+L]], so every output has its own
right endpoint and nothing depends on where the chunk boundaries fall.

The algorithm is specified in plans/detrend_1d_kalman.md and in
notes/tree_dedispersion.tex, section "Time detrending algorithm 2: Kalman filter".  This is
a second detrender, not a replacement for detrending_1d: the two are intended to run
side by side and be compared.  Relative to the local polynomial fit it has no window
and no per-window rank deficiency (the prior regularizes everything), a monotone
1-H(omega) with no stopband ripple, and no seam; against that it needs roughly twice
the lookahead, carries state across chunks, and its guaranteed polynomial
reproduction is one degree weaker.

Only k = 2 is implemented.  The equations are written for general k throughout,
because the degree-(k-1) and degree-(2k-1) reproduction statements are what explain
the design, but k=2 already matches the shipped n=2 local fit on the asymptotic
degree 2k-1 = 3 that governs the passband, so k=3 would buy more rolloff than
anything downstream is asking for.

Deliberate divergences from any future GPU version (not bugs):

  - the backward window is recomputed per output (vectorized in lockstep over the
    lag) rather than decomposed van-Herk-style over a monoid.  The two share no
    arithmetic, which is what makes the second validatable against this one.
  - J is stored as a full (k,k) symmetric array rather than packed.
  - the forward pass is a sequential python loop, not a parallel scan.
"""

from .model import StateSpaceModel, tau_from_equivalent_W
from .InfoFilter import forward_step, backward_step
from .KalmanDetrender import KalmanDetrender, KalmanState
from .brute_force import kalman_brute_force, impulse_kernel, difference_matrix
