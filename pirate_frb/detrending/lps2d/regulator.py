"""
The regulator D_1: a first-difference (P-spline) penalty on the coefficients.

The detrender solves (G + eta*D_1) a = U instead of G a = U.  Without the
regulator, a coefficient whose support is entirely masked has an exactly zero row
in G, the Cholesky has nothing to divide by, and the contamination has to be
cleaned up by expanding the mask -- expensive, and in the worst case it discards
most of the surviving channels.  With it, G + eta*D_1 is positive definite as
soon as the zone holds ONE unmasked channel, and the price is a bounded bias
instead of a structural failure.

    D_1 = sum_j (a_{j+1} - a_j)^2,   summed within a zone only.

Why this and not the textbook int (f')^2:

  - Same null space (the constants) and, measured, the same smallest nonzero
    eigenvalue to 2%, hence the same bias at matched regularization strength.
  - Half-bandwidth 1 instead of n_phi.
  - It never differentiates anything, so it cannot be broken by repeated interior
    knots.  int (f^(m))^2 with m > p - mu + 1 diverges at a knot of multiplicity
    mu, and -- the real hazard -- per-interval quadrature returns a finite matrix
    anyway, with a spurious extra null direction that silently destroys the
    positive-definiteness guarantee above.

NULL SPACE, which is the property everything rests on.  Because the basis is a
partition of unity on each zone (knots.py), the all-ones coefficient vector of a
zone is the constant function 1.  D_1 annihilates exactly that vector and nothing
else, so:

  - a constant baseline is removed EXACTLY, not shrunk (test_flat_baseline_exact);
  - 1^T G 1 = sum_f w_f = (unmasked channels in the zone), so one unmasked channel
    makes G + eta*D_1 positive definite.

ASSEMBLY IS PER ZONE, and this is not optional.  A first-difference penalty built
over the whole coefficient vector couples adjacent zones with weight 1, which
would destroy both the block structure and the exact-constant property (the null
space would drop to a single global constant rather than one per zone).  The
coefficients of a zone are contiguous, so "per zone" just means dropping the
differences that straddle a zone boundary.
"""

import numpy as np


def d1_banded(kv, dtype=np.float64):
    """
    The per-zone first-difference penalty, in the banded layout of reduce.py:
    shape (N_phi, 2), with R[j,0] = D_{jj} and R[j,1] = D_{j,j+1}.

    R[N_phi-1, 1] is zero, as is R[j,1] for any j at a zone boundary.
    """
    N = kv.N_phi
    R = np.zeros((N, 2), dtype=dtype)

    # One term per intra-zone edge (j, j+1); each contributes +1 to both diagonal
    # entries and -1 off-diagonal.
    same = kv.zone_id[:-1] == kv.zone_id[1:]
    j = np.flatnonzero(same)
    np.add.at(R[:, 0], j, 1.0)
    np.add.at(R[:, 0], j + 1, 1.0)
    R[j, 1] = -1.0
    return R


def d1_dense(kv, dtype=np.float64):
    """Dense (N_phi, N_phi) form of d1_banded(); for the reference path and tests."""
    R = d1_banded(kv, dtype)
    N = kv.N_phi
    out = np.zeros((N, N), dtype=dtype)
    out[np.arange(N), np.arange(N)] = R[:, 0]
    j = np.arange(N - 1)
    out[j, j + 1] = R[j, 1]
    out[j + 1, j] = R[j, 1]
    return out
