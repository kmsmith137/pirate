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

WHAT r_min IS AND IS NOT.  r_min is a NUMERICAL statistic: it says the factorization
is trustworthy, and nothing more.  It cannot detect a zone that is statistically
degenerate.  The clearest case: a zone with ONE unmasked channel has r_min ~ 1e-2,
healthier than most masks -- because the regulator's null space contains the
constants, the fit passes exactly through that one point, and the residual is
identically zero with zero degrees of freedom.  Detecting that needs the residual
degrees of freedom M - tr(H), which is deliberately not implemented yet.  Until it
is, a nearly empty zone will produce an identically zero residual and nothing here
will complain.

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


def solve_normal_equations(G, U, kv, D1, eta, eps):
    """
    G: (..., N_phi, n_phi+1) banded, U: (..., N_phi), both in the working dtype.
    D1: (N_phi, 2) banded regulator from regulator.d1_banded().

    Returns (a, rmin, bad) with

        a:    (..., N_phi)   fitted coefficients, zeroed in flagged zones
        rmin: (..., nzone)   min relative pivot in each zone, 0 if the zone is dead
        bad:  (..., nzone)   bool, rmin < eps

    A zone is DEAD (no unmasked channel anywhere in it) exactly when the sum of
    G's diagonal over that zone is zero: G_jj = sum_f w_f phi_j(f)^2 and the basis
    is a partition of unity, so that sum is sum_f w_f over the zone.  Testing it
    this way is exact and needs no separate channel count.
    """
    dtype = G.dtype
    N, n_phi = kv.N_phi, kv.n_phi

    A = G.astype(dtype, copy=True)
    A[..., 0] += (eta * D1[:, 0]).astype(dtype)
    if n_phi >= 1:
        A[..., 1] += (eta * D1[:, 1]).astype(dtype)

    Ahat, s = equilibrate(A)
    L, piv = _cholesky_banded(Ahat)

    a = backward_subst(L, forward_subst(L, U.astype(dtype) / s)) / s

    zs = zone_slices(kv)
    rmin = np.empty(G.shape[:-2] + (kv.nzone,), dtype=dtype)
    for z, (lo, hi) in enumerate(zs):
        alive = G[..., lo:hi, 0].sum(axis=-1) > 0
        rmin[..., z] = np.where(alive, piv[..., lo:hi].min(axis=-1), 0)

    bad = rmin < eps
    for z, (lo, hi) in enumerate(zs):
        a[..., lo:hi] = np.where(bad[..., z, None], 0, a[..., lo:hi])

    return a, rmin, bad
