"""
The time-axis polynomial basis for local polynomial subtraction.

Over a window of 2W+1 samples the spline coefficients are modelled as
polynomials of degree n in the offset s, a_j(t+s) = sum_q alpha_jq p_q(s), and
this class holds everything about {p_q} that does not depend on the data.

ORTHOGONAL, NOT MONOMIAL, and this is the one choice here that changes a
threshold rather than a constant.  For a window-constant mask the assembled
matrix is exactly (G + eta D_1) kron T with T_qr = sum_s p_q p_r, and
equilibration of a Kronecker product is Kronecker, so the pivots multiply:

    r_min(2-d) = r_min(1-d) * r_min(T).

With raw monomials r_min(T) is 0.18 to 0.25 at n = 2 (tending to 1 - sqrt(5)/3,
worst at small W), which would multiply the 1-d conditioning margin by that
factor -- enough to push the worst adversarial mask below eps and expand zones
whose fits are perfectly accurate.  With p_q orthonormal on the window, T is the
identity and r_min(2-d) = r_min(1-d) exactly.  It costs nothing: the basis
enters only through stencil coefficients that are precomputed either way.

WHAT THE MONOMIAL BASIS DOES BUY is the Hankel structure: with p_q = s^q the
block matrix depends on (q,r) only through q+r, so there are 2n+1 distinct
moment arrays rather than (n+1)(n+2)/2.  Orthogonalizing destroys that (a
product p_q p_r is not a function of q+r).  It costs one extra stencil at n = 2
(6 rather than 5) and nothing else, because every p_q p_r still has degree <= 2n
and the number of INDEPENDENT quantities in the data pass is min(2n+1, 2W+1)
either way.

EVALUATION AT THE CENTRE.  The committed baseline is sum_j (sum_q alpha_jq
p_q(0)) phi_j, NOT sum_j alpha_j0 phi_j.  Those coincide only for monomials,
where p_q(0) = delta_q0; with an orthogonal basis every even q contributes.
Getting it wrong is silent -- it still produces a plausible baseline, just the
wrong one -- so eval0 is stored here next to the stencils rather than being
open-coded at the call site.
"""

import numpy as np


class TimeBasis:
    """
    Attributes:

      n, W          degree and window half-width; the window is 2W+1 samples
      s[k]          offsets, k = 0..2W, s[k] = k - W  (so s[W] = 0)
      P[k, q]       p_q(s[k]), shape (2W+1, n+1)
      T[q, r]       window Gram sum_k P[k,q] P[k,r]; the identity if orthogonal
      eval0[q]      p_q(0), the contraction that commits the baseline
      parity[q]     +1 if p_q is even in s, -1 if odd
      pair_q, pair_r    the (q <= r) index pairs, length npair
      npair         (n+1)(n+2)/2
      gstencil[p,k] P[k,pair_q[p]] * P[k,pair_r[p]], the G-moment stencils
      ustencil[q,k] P[k,q], the U-moment stencils
    """

    def __init__(self, n, W, orthogonal=True, dtype=np.float64):
        n, W = int(n), int(W)
        if n < 0:
            raise ValueError(f'TimeBasis: n={n} must be >= 0')
        if W < 0:
            raise ValueError(f'TimeBasis: W={W} must be >= 0')
        if 2*W + 1 < n + 1:
            raise ValueError(f'TimeBasis: a degree-{n} fit needs 2W+1 >= n+1 '
                             f'samples in the window, but W={W} gives {2*W+1}')

        self.n, self.W = n, W
        self.dtype = np.dtype(dtype)
        self.s = np.arange(-W, W+1, dtype=np.float64)

        P = np.stack([self.s**q for q in range(n+1)], axis=1)
        if orthogonal:
            # QR on a symmetric grid preserves parity, because <s^i, s^j> = 0
            # whenever i+j is odd; test_time_basis() checks that it really does.
            P, R = np.linalg.qr(P)
            P = P * np.sign(np.diag(R))[None, :]      # fix the sign convention
        self.orthogonal = bool(orthogonal)
        self.P = P.astype(self.dtype)
        self.T = (P.T @ P).astype(self.dtype)
        self.eval0 = self.P[W].copy()
        self.parity = np.array([1 if q % 2 == 0 else -1 for q in range(n+1)])

        pairs = [(q, r) for q in range(n+1) for r in range(q, n+1)]
        self.pair_q = np.array([q for q, _ in pairs], dtype=np.int64)
        self.pair_r = np.array([r for _, r in pairs], dtype=np.int64)
        self.npair = len(pairs)
        self.gstencil = (self.P[:, self.pair_q] * self.P[:, self.pair_r]).T.copy()
        self.ustencil = self.P.T.copy()
        # Parity of each stencil in s, which is what lets moments.py fold the
        # window in half.
        self.gparity = self.parity[self.pair_q] * self.parity[self.pair_r]
        self.uparity = self.parity.copy()

    def __repr__(self):
        return (f'TimeBasis(n={self.n}, W={self.W}, '
                f'orthogonal={self.orthogonal}, dtype={self.dtype.name})')
