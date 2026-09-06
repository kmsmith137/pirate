"""
The regularized normal-equation solve, and the conditioning statistic r_min.

Three steps, in this order, and the order is the specification rather than an
implementation choice:

  1. EQUILIBRATE (G + eta*D_1) to unit diagonal.
  2. FACTOR by a banded Cholesky, recording the relative pivot of every
     coefficient.  r_min is the minimum over a zone.
  3. SOLVE, unscale, and report r_min so the caller can expand zones where it
     falls below eps.

WHY EQUILIBRATION IS LOAD-BEARING, twice over.  It is what makes eps a
scale-invariant threshold: applying eps to an un-equilibrated pivot would make the
masking decision depend on the units of the data, which test_solve() checks by
rescaling.  Less obviously, it is also what makes the problem well conditioned at
all.  The RAW matrix G + eta*D_1 has condition number O(h/eta) -- linear in the
widest knot interval and inverse in eta, so at h = 3000, eta = 1e-3 it is order
1e5, and it grows without bound as eta falls.  Equilibration removes BOTH factors:
a coefficient with no data has diagonal eta*(D_1)_jj, so dividing by sqrt of it
cancels eta exactly, and a coefficient with data has diagonal O(h), which cancels
the h.  The equilibrated condition number is then O(1) in both.  Do not "optimize"
this away, and do not threshold on the un-equilibrated pivot.

WHY r_min AND NOT THE CONDITION NUMBER.  These measure different errors, and the
choice is deliberate.

Cholesky is backward stable, so the error in the COEFFICIENTS is governed by
kappa = lambda_max/lambda_min, and equilibration pins lambda_max = O(1) (it is
bounded by n_phi+1; see the proof sketch below), leaving a relative coefficient
error of order eps_mach/lambda_min.  r_min is NOT that quantity: Cholesky pivots
are diagonal entries of Schur complements, so lambda_min <= r_min <= 1 always, and
the gap reaches 84x on real detrender matrices.  Using eps_mach/r_min as a bound on
the coefficient error fails by up to 17x.

But the coefficients are not what this module emits.  The residual is evaluated at
UNMASKED CHANNELS, and for THAT error we have no rigorous bound in terms of either
statistic -- only a measurement.  Over 523 random and adversarial configurations
spanning four decades of r_min, in float32:

    r_min in [1e-1, 1)      worst fit error 1.9e-7
    r_min in [1e-2, 1e-1)   worst fit error 1.7e-6
    r_min in [1e-3, 1e-2)   worst fit error 1.9e-5
    r_min below 1e-3        worst fit error 1.2e-4

roughly eps_mach/(4 r_min).  TREAT THIS AS AN EMPIRICAL RULE OF THUMB, NOT A BOUND.
It is a trend, not a law: the log-log slope against 1/r_min is 0.68 rather than 1,
the correlation is 0.62, and "no case exceeded eps_mach/r_min" is an observation
about the masks we generated, not a theorem.  A rigorous statement would need the
componentwise (Skeel) bound |A^-1||A||x|, which is not cheap to evaluate.

With that caveat, r_min is still the better choice of the two, and for a structural
reason rather than a lucky one: lambda_min's small modes are typically DELOCALIZED
over coefficients with no data -- the extremal one is a ramp across a dead run,
giving lambda_min ~ 1/K^2 -- and such a mode contributes nothing at an unmasked
channel.  r_min is a local statistic and does not see it.  Thresholding on
lambda_min would therefore mask zones whose fits are perfectly accurate, on account
of a direction the output never touches.

Neither statistic detects a zone that is STATISTICALLY degenerate, and THAT IS A
DECISION RATHER THAN A GAP.  The clearest case: a zone with ONE unmasked channel
has r_min ~ 1e-2, healthier than most masks -- because the regulator's null space
contains the constants, the fit passes exactly through that one point, and the
residual is identically zero with zero residual degrees of freedom
nu = M - tr(H).  There is deliberately no nu cut.  Do not add one without
revisiting the argument below, which is not recoverable by reading the code.

The argument turns on an asymmetry specific to a rare-event search.  Two failure
modes are not comparable:

  - UNDERSUBTRACTION (shrinkage, or excess variance) leaves excursions, and
    converts noise into triggers.  With large trial factors there are many 7
    sigma noise fluctuations, so a mechanism that biases by 3 sigma with
    probability 1e-5 can still flood the search and force the threshold up.
    Dangerous at ANY rate, however rare.
  - OVERFITTING (leverage) subtracts noise instead of baseline and suppresses the
    residual.  Converting a 20 sigma event to 10 sigma with probability 1e-5 does
    not move the event rate and does not change the threshold.  Only a
    significant AVERAGE effect matters.

nu is an overfitting statistic, so it lands in the second category.  Three things
then settle it:

  1. Overfitting here has the WRONG SIGN and cannot manufacture an excursion at
     all.  Restricted to the unmasked channels, H = Phi (G + eta D_1)^-1 Phi^T W
     is symmetric positive semidefinite with spectrum in [0,1], so I-H and hence
     (I-H)^2 also have spectrum in [0,1], and
         Var[r_f] = sigma^2 [(I-H)^2]_ff <= sigma^2
     identically.  (NOT sigma^2 (1-h_ff)^2, which drops the off-diagonal term
     sum_{g != f} h_fg^2; the bound is what matters and it holds either way.)
     Measured maximum 0.9968 over 101 masks; over-dispersion never occurs.  Low
     nu can only suppress the residual, never inflate it.  This is stronger than
     "small on average" -- it is "wrong direction, always".
  2. Operationally the leverage is negligible, and rigorously bounded:
     tr(H) <= N_phi(n+1), so leverage <= N_phi(n+1)/M.  Measured at
     nfreq = 4096: 0.0032 fully valid, 0.0035 at 10% flagged, 0.0064 at 50%
     flagged, each sitting exactly on that bound.  (Much larger figures appear if
     you measure over the test mask generator, whose masks have tiny M by
     construction; those are not operational numbers.)
  3. Computing nu costs a banded inverse per zone per time sample -- the
     Takahashi recursion on the factor already in hand -- for no false-positive
     protection.

CONTRAST with what does matter, since the two look similar and are not.  Poor
conditioning produces excess variance through ROUNDOFF in a near-singular solve,
which is undersubtraction's category, and that is why eps exists and why the
rank test below is not optional.  A rank-deficient matrix gives garbage
coefficients, i.e. an excursion risk, so it is a correctness guard rather than a
statistical one.

ONE CAVEAT, and it is an interface concern rather than a detrender defect: at
nu = 0 the residual is identically ZERO, not merely small.  Anything downstream
that estimates a variance from the data and divides by it will meet 0/0 on such a
zone.  If that matters, the cheap fix is not nu but the count: tr(H) <= min(M, K)
with K = N_phi(n+1), so M >= K + nu_min guarantees nu >= nu_min, and
sum_{j in zone} G_jj -- already computed just below for the dead-zone test -- lies
in [M_zone/(n_phi+1), M_zone], so changing its "> 0" to "> c*K" is a complete,
free, conservative surrogate.

NO DEFLATION.  Unlike an unregularized spline detrender, there is nothing to
deflate: with one unmasked channel in the zone, G + eta*D_1 is positive definite
(see regulator.py).  The only non-positive pivot that can arise is in a zone with
NO unmasked channels, which is detected structurally below.  The guard in
_cholesky_banded() exists to stop NaN propagating out of that case, not as a
numerical strategy.
"""

import numpy as np

from .reduce import band_to_dense


def equilibrate(A):
    """
    (..., N, nb+1) banded, symmetric, positive diagonal -> (Ahat, s) with Ahat
    having unit diagonal and s = sqrt(diag(A)).

    s is returned rather than reapplied, because the caller needs it to unscale
    the solution.  Zero diagonals (a zone with no data at all) are mapped to
    s = 1, which leaves that row of Ahat zero rather than NaN; the zone is caught
    by the structural dead-zone test in solve_normal_equations().
    """
    A = np.asarray(A)
    diag = A[..., 0]
    s = np.sqrt(np.where(diag > 0, diag, 1)).astype(A.dtype)
    out = np.empty_like(A)
    N, nb = A.shape[-2], A.shape[-1] - 1
    for b in range(nb+1):
        j = np.arange(N-b)
        out[..., j, b] = A[..., j, b] / (s[..., j] * s[..., j+b])
    # Rows with no data have s = 1 by fiat, so their diagonal is 0, not 1.
    return out, s


def _cholesky_banded(Ahat):
    """
    Banded Cholesky of an equilibrated symmetric matrix.  Returns (L, piv) with

        L[..., i, b] = L_{i, i-b}   (b = 0 is the diagonal)
        piv[..., j]  = the relative pivot of coefficient j, in [0, 1]

    A pivot that comes out non-positive can only happen in a zone with no unmasked
    channels (see the module docstring).  Such a row is replaced by e_j -- pivot
    recorded as 0, unit diagonal, no off-diagonal -- purely so that no NaN or Inf
    reaches the caller; the zone is masked out wholesale anyway.
    """
    Ahat = np.asarray(Ahat)
    N, nb = Ahat.shape[-2], Ahat.shape[-1] - 1
    dtype = Ahat.dtype
    L = np.zeros(Ahat.shape[:-1] + (nb+1,), dtype=dtype)
    piv = np.zeros(Ahat.shape[:-2] + (N,), dtype=dtype)

    for j in range(N):
        acc = Ahat[..., j, 0].astype(dtype, copy=True)
        for c in range(1, min(nb, j) + 1):
            acc = acc - L[..., j, c]**2
        good = acc > 0
        piv[..., j] = np.where(good, acc, 0)
        # Where the pivot failed, force row j = e_j.
        L[..., j, 1:] = np.where(good[..., None], L[..., j, 1:], 0)
        L[..., j, 0] = np.where(good, np.sqrt(np.where(good, acc, 1)), 1)

        for b in range(1, min(nb, N-1-j) + 1):
            acc2 = Ahat[..., j, b].astype(dtype, copy=True)
            for c in range(1, min(nb - b, j) + 1):
                acc2 = acc2 - L[..., j, c] * L[..., j+b, b+c]
            L[..., j+b, b] = np.where(good, acc2 / L[..., j, 0], 0)

    return L, piv


def forward_subst(L, b):
    """Solve L y = b for banded lower-triangular L; b, y have shape (..., N)."""
    N, nb = L.shape[-2], L.shape[-1] - 1
    y = np.zeros(b.shape, dtype=L.dtype)
    for j in range(N):
        acc = b[..., j].astype(L.dtype, copy=True)
        for c in range(1, min(nb, j) + 1):
            acc = acc - L[..., j, c] * y[..., j-c]
        y[..., j] = acc / L[..., j, 0]
    return y


def backward_subst(L, y):
    """Solve L^T x = y for banded lower-triangular L."""
    N, nb = L.shape[-2], L.shape[-1] - 1
    x = np.zeros(y.shape, dtype=L.dtype)
    for j in range(N-1, -1, -1):
        acc = y[..., j].astype(L.dtype, copy=True)
        for b in range(1, min(nb, N-1-j) + 1):
            acc = acc - L[..., j+b, b] * x[..., j+b]
        x[..., j] = acc / L[..., j, 0]
    return x


def zone_slices(kv):
    """[(lo, hi)] coefficient ranges of each zone; zones are contiguous in j."""
    out = []
    for z in range(kv.nzone):
        idx = np.flatnonzero(kv.zone_id == z)
        out.append((int(idx[0]), int(idx[-1]) + 1))
    return out


def solve_banded(A, U, kv, n, live, nmin):
    """
    A: (..., N_phi*(n+1), nb+1) banded and already regularized, U: (..., N_phi*(n+1)).
    live: (..., nzone) int, the number of window offsets at which each zone holds
    data (moments.zone_live_counts); nmin = n+1 is the number required.

    Returns (alpha, rmin), with rmin the minimum relative pivot per zone, exactly
    0 for a zone that fails the rank test.  Zeroing alpha in a flagged zone is the
    CALLER's job: eps lives with the caller, so this function reports the statistic
    and does not apply a policy to it.

    THE RANK TEST GENERALIZES THE 1-d DEAD-ZONE TEST.  At n = 0 "at least one
    unmasked channel" is exactly "live >= 1".  At n > 0 a degree-n fit in time is
    singular unless the zone carries data at n+1 distinct offsets, whatever the
    channel count at those offsets, because a nonzero degree-n polynomial
    vanishing on every populated offset is a null direction of the whole
    assembled matrix.  Structural, exact, and not inferrable from a pivot.
    """
    dtype = A.dtype
    Ahat, s_ = equilibrate(A)
    L, piv = _cholesky_banded(Ahat)
    alpha = backward_subst(L, forward_subst(L, U.astype(dtype) / s_)) / s_

    rmin = np.empty(A.shape[:-2] + (kv.nzone,), dtype=dtype)
    for z, (lo, hi) in enumerate(zone_slices(kv)):
        Ilo, Ihi = lo*(n+1), hi*(n+1)
        ok = live[..., z] >= nmin
        rmin[..., z] = np.where(ok, piv[..., Ilo:Ihi].min(axis=-1), 0)
    return alpha, rmin


def solve_normal_equations(G, U, kv, D1, eta, eps):
    """
    The (n, W) = (0, 0) path, kept as a thin wrapper: G banded (..., N_phi,
    n_phi+1), U (..., N_phi).  Returns (a, rmin, bad).
    """
    dtype = G.dtype
    A = G.astype(dtype, copy=True)
    A[..., 0] += (eta * D1[:, 0]).astype(dtype)
    if kv.n_phi >= 1:
        A[..., 1] += (eta * D1[:, 1]).astype(dtype)

    live = np.zeros(G.shape[:-2] + (kv.nzone,), dtype=np.int64)
    for z, (lo, hi) in enumerate(zone_slices(kv)):
        live[..., z] = (G[..., lo:hi, 0].sum(axis=-1) > 0).astype(np.int64)

    a, rmin = solve_banded(A, U, kv, 0, live, 1)
    bad = rmin < eps
    for z, (lo, hi) in enumerate(zone_slices(kv)):
        a[..., lo:hi] = np.where(bad[..., z, None], 0, a[..., lo:hi])
    return a, rmin, bad
