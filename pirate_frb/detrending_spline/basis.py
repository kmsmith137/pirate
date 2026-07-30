"""
The B-spline basis, tabulated per channel.

Only n_phi+1 basis functions are nonzero at any channel (compact support), so the
whole frequency axis is described by an (nfreq, n_phi+1) table of values plus an
(nfreq, (n_phi+1)(n_phi+2)/2) table of their pairwise products -- 9 floats per
channel at n_phi=2.  Both are independent of time and of the beam axis, so
they are built once here and the per-sample work downstream is pure FMA against
them.  The products are tabulated rather than recomputed because they are exactly
what the Gram accumulation of reduce.py consumes.

The table is built in float64 and cast, so the working dtype affects the
arithmetic that uses the basis but not the basis itself.

Evaluation is the standard triangular form of the Cox-de Boor recursion (the
NURBS book's Algorithm A2.2), which has no zero denominators at all provided the
evaluation point's span is non-empty -- which KnotVector.j0 guarantees by
construction.  That is why there is no "drop the term with a vanishing
denominator" special case here even though repeated knots are fully supported.
"""

import numpy as np


def eval_basis(knots, n_phi, x, j0):
    """
    Evaluate the n_phi+1 nonzero B-splines at each point of 'x'.

    knots: int/float array; x: float64 array of shape (nc,); j0: int array of the
    same shape, the span index with knots[j0] <= x < knots[j0+1] (the span must be
    non-empty).  Returns shape (nc, n_phi+1) in float64, with column a holding
    phi_{j0-n_phi+a}(x).
    """
    x = np.asarray(x, dtype=np.float64)
    j0 = np.asarray(j0)
    assert x.shape == j0.shape and x.ndim == 1
    k = np.asarray(knots, dtype=np.float64)

    nc = x.shape[0]
    N = np.zeros((nc, n_phi+1), dtype=np.float64)
    N[:, 0] = 1.0
    left = np.zeros((nc, n_phi+1), dtype=np.float64)
    right = np.zeros((nc, n_phi+1), dtype=np.float64)

    for p in range(1, n_phi+1):
        left[:, p] = x - k[j0 + 1 - p]
        right[:, p] = k[j0 + p] - x
        saved = np.zeros(nc, dtype=np.float64)
        for r in range(p):
            # The denominator is knots[j0+r+1] - knots[j0+r+1-p], which straddles
            # the span [knots[j0], knots[j0+1]) and is therefore strictly positive.
            temp = N[:, r] / (right[:, r+1] + left[:, p-r])
            N[:, r] = saved + right[:, r+1]*temp
            saved = left[:, p-r]*temp
        N[:, p] = saved

    return N


class BasisTable:
    """
    Per-channel tables for one KnotVector, cast to a working dtype.

    Attributes:

      kv           the KnotVector
      dtype        working dtype of phi and prod
      phi[f, a]    (nfreq, n_phi+1), value of phi_{j0[f]-n_phi+a} at channel f
      prod[f, p]   (nfreq, npair), the pairwise products phi_a*phi_b for the
                   pairs (a,b) with a <= b, in the order given by pair_a/pair_b
      pair_a[p], pair_b[p]   the pair indices, both in [0, n_phi]
      npair        (n_phi+1)(n_phi+2)/2

    'prod' is what the Gram accumulation reads: the contribution of channel f to
    G_{j,l} with j = j0[f]-n_phi+pair_a[p] and l = j0[f]-n_phi+pair_b[p] is
    w[f]*prod[f,p].
    """

    def __init__(self, kv, dtype=np.float32):
        self.kv = kv
        self.dtype = np.dtype(dtype)
        n_phi = kv.n_phi

        # Channel f carries its data at f + 1/2 (see knots.py); this is the only
        # place the half-channel offset appears.
        x = np.arange(kv.nfreq, dtype=np.float64) + 0.5
        phi64 = eval_basis(kv.knots, n_phi, x, kv.j0)

        pairs = [(a, b) for a in range(n_phi+1) for b in range(a, n_phi+1)]
        self.pair_a = np.array([a for a, _ in pairs], dtype=np.int64)
        self.pair_b = np.array([b for _, b in pairs], dtype=np.int64)
        self.npair = len(pairs)

        self.phi = phi64.astype(self.dtype)
        self.prod = (phi64[:, self.pair_a] * phi64[:, self.pair_b]).astype(self.dtype)

    def __repr__(self):
        return (f'BasisTable(nfreq={self.kv.nfreq}, n_phi={self.kv.n_phi}, '
                f'N_phi={self.kv.N_phi}, dtype={self.dtype.name})')

    def dense(self):
        """
        The full (nfreq, N_phi) basis matrix, mostly zeros.

        Used by the oracles and the tests, which want an explicit design matrix;
        the detrenders never materialize this.
        """
        kv = self.kv
        out = np.zeros((kv.nfreq, kv.N_phi), dtype=self.dtype)
        f = np.arange(kv.nfreq)
        for a in range(kv.n_phi + 1):
            out[f, kv.j0 - kv.n_phi + a] = self.phi[:, a]
        return out
