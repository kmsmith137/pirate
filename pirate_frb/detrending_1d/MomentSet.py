"""
The 'moment monoid' used by the 1-d detrender (see notes/tree_dedispersion.tex,
section "Detrending").

A MomentSet represents, for some set S of time samples, the quantities

    nv   = sum_{u in S} m[u]                         (number of valid samples)
    c    = (sum_{u in S} m[u] u) / nv                (mask-weighted centroid)
    S_r  = sum_{u in S} m[u] x_u^r,       x_u = (u-c)/W,   0 <= r <= 2n
    U_r  = sum_{u in S} (m d)[u] x_u^r,                    0 <= r <= n

'Adaptive centering' means every set carries its own centroid c, so that the
moments are always expanded about a point inside (or at least near) the data.
This is what keeps the normal equations well conditioned when the mask leaves
only a narrow, off-center sliver of valid samples.  By construction S_1 = 0; we
store it anyway (always zero) so that the Hankel indexing G[j,l] = S[j+l] in
LocalPolyFit is trivial.  The eventual GPU implementation will omit it.

Disjoint union is an associative, commutative operation with an identity, so
these objects form a monoid and support a parallel scan (see scan.py).

Units and coordinates
---------------------
c is stored in *sample* units, relative to the start of the current buffer, NOT
in absolute stream coordinates.  This matters: absolute sample indices grow
without bound during a long run, and would destroy float32 precision in c.
Leaf construction sets c = u exactly, which is why sample units are preferred
over pre-dividing by W.
"""

import numpy as np


def _binom_table(K):
    """b[r][j] = C(r,j), for 0 <= j <= r < K.  Small python ints (exact)."""
    b = [[1]*(r+1) for r in range(K)]
    for r in range(2,K):
        for j in range(1,r):
            b[r][j] = b[r-1][j-1] + b[r-1][j]
    return b


def pascal_shift(v, delta, binom):
    """
    Change of origin for a moment vector.  If v_j are moments about c, then the
    moments about c' are given by (T_delta v)_r = sum_{j<=r} C(r,j) delta^(r-j) v_j,
    with delta = (c - c')/W.

    'v' has shape (batch..., K) and 'delta' has shape (batch...,).  Note
    T_delta T_delta' = T_(delta+delta') exactly, so composing many small shifts
    is *not* cheaper (numerically) than one large one -- see the appendix of
    notes/tree_dedispersion.tex.
    """
    K = v.shape[-1]
    dtype = v.dtype

    # powers[k] = delta**k
    powers = [np.ones_like(delta)]
    for _ in range(1, K):
        powers.append(powers[-1] * delta)

    out = np.empty_like(v)
    for r in range(K):
        acc = v[..., r].copy()
        for j in range(r):
            acc = acc + binom[r][j] * powers[r-j] * v[..., j]
        out[..., r] = acc

    assert out.dtype == dtype
    return out


class MomentSet:
    """
    Batched monoid elements.  Leading axes are batch; moments are on the last
    array axis.  Inside Detrender the batch shape is (S, nblocks, B), where S is
    the spectator axis (one entry per (beam,freq) pair).

    Fields:
       nv:  shape (batch...)         valid-sample count, stored as 'dtype'
       c:   shape (batch...)         centroid, in buffer-relative sample units
       S:   shape (batch..., 2n+1)   mask moments about c;  S[...,1] is always 0
       U:   shape (batch..., n+1)    data moments about c
    """

    def __init__(self, nv, c, S, U, n, W):
        assert S.shape[-1] == 2*n+1
        assert U.shape[-1] == n+1
        assert nv.shape == c.shape == S.shape[:-1] == U.shape[:-1]
        assert nv.dtype == c.dtype == S.dtype == U.dtype
        self.nv, self.c, self.S, self.U = nv, c, S, U
        self.n, self.W = n, W
        self.dtype = nv.dtype
        self.binom = _binom_table(2*n+1)

    @property
    def batch_shape(self):
        return self.nv.shape

    def _like(self, nv, c, S, U):
        return MomentSet(nv, c, S, U, self.n, self.W)

    def copy(self):
        return self._like(self.nv.copy(), self.c.copy(), self.S.copy(), self.U.copy())

    def slice_pos(self, sl):
        """Slice the last batch (position) axis.  Note we cannot use a plain
        ms[..., sl] since S and U carry a trailing moment axis."""
        return self._like(self.nv[..., sl], self.c[..., sl],
                          self.S[..., sl, :], self.U[..., sl, :])

    def take_batch(self, idx):
        """Index the *leading* batch axes with 'idx' (a tuple).  The trailing
        moment axis of S,U is left alone, which works because numpy applies a
        short index tuple to the leading axes."""
        return self._like(self.nv[idx], self.c[idx], self.S[idx], self.U[idx])

    def set_pos(self, sl, other):
        """In-place assignment into the last batch axis (used by the scans)."""
        self.nv[..., sl] = other.nv
        self.c[..., sl] = other.c
        self.S[..., sl, :] = other.S
        self.U[..., sl, :] = other.U

    # ---------------------------------------------------------------- builders

    @classmethod
    def leaves(cls, u, m, md, n, W, dtype):
        """
        One MomentSet per sample.  'u' is the buffer-relative sample index, 'm'
        the mask (0 or 1), 'md' the *masked* data value m*d.  All are broadcast
        to a common shape.

        A masked leaf has nv = 0 and c = u, which is finite -- see the empty-set
        rule in merge().
        """
        u, m, md = np.broadcast_arrays(u, m, md)
        u = np.ascontiguousarray(u, dtype=dtype)
        m = np.ascontiguousarray(m, dtype=dtype)
        md = np.ascontiguousarray(md, dtype=dtype)

        S = np.zeros(u.shape + (2*n+1,), dtype=dtype)
        U = np.zeros(u.shape + (n+1,), dtype=dtype)
        S[..., 0] = m        # x_u = 0 for a one-element set, so only r=0 survives
        U[..., 0] = md
        return cls(m.copy(), u.copy(), S, U, n, W)

    @classmethod
    def identity(cls, batch_shape, c_nominal, n, W, dtype):
        """Monoid identity: nv = 0, all moments zero, c finite (and arbitrary)."""
        z = np.zeros(batch_shape, dtype=dtype)
        c = np.full(batch_shape, c_nominal, dtype=dtype)
        return cls(z, c, np.zeros(batch_shape + (2*n+1,), dtype=dtype),
                   np.zeros(batch_shape + (n+1,), dtype=dtype), n, W)

    @classmethod
    def direct(cls, u, m, md, n, W, dtype, axis=-1):
        """
        Compute the moments of a whole set directly from the definition, by
        summing over 'axis'.  Used by reference.py and by the tests that merge()
        is checked against.
        """
        u = np.asarray(u, dtype=dtype)
        m = np.asarray(m, dtype=dtype)
        md = np.asarray(md, dtype=dtype)
        u, m, md = np.broadcast_arrays(u, m, md)

        nv = m.sum(axis=axis)
        safe = nv > 0
        c = np.where(safe, (m*u).sum(axis=axis) / np.where(safe, nv, 1),
                     u.mean(axis=axis)).astype(dtype)

        x = ((u - np.expand_dims(c, axis)) / W).astype(dtype)
        S = np.zeros(nv.shape + (2*n+1,), dtype=dtype)
        U = np.zeros(nv.shape + (n+1,), dtype=dtype)
        p = np.ones_like(x)
        for r in range(2*n+1):
            S[..., r] = (m*p).sum(axis=axis)
            if r <= n:
                U[..., r] = (md*p).sum(axis=axis)
            p = p * x
        S[..., 1] = 0
        return cls(nv.astype(dtype), c, S, U, n, W)


def merge(a, b):
    """
    Disjoint-union merge, a and b broadcastable.  'a' should be the earlier
    (lower-index) range and 'b' the later one; the result is mathematically
    symmetric, but keeping a consistent order keeps the rounding reproducible.

    The centroid update is written as c + f*Delta*W rather than
    (nv*c + nv'*c')/N to avoid forming the large product (Chan's parallel-mean
    update).

    Empty-set rule: c is undefined when nv == 0, and a tree scan performs many
    merges involving empty aggregates (any fully masked sub-block).  Two things
    keep this branch-free and NaN-free: leaves and identities carry a *finite*
    nominal c, and the divide below is guarded.  Merging empty with non-empty
    then gives f = 1, d2 = 0, cnew = b.c, which is correct.
    """
    assert a.n == b.n and a.W == b.W and a.dtype == b.dtype
    n, W, dtype = a.n, a.W, a.dtype

    N = a.nv + b.nv
    safe = N > 0
    f = np.where(safe, b.nv / np.where(safe, N, 1), 0).astype(dtype)
    Delta = ((b.c - a.c) / W).astype(dtype)

    cnew = (a.c + f * Delta * W).astype(dtype)
    d1 = (-f * Delta).astype(dtype)          # shift for a: (a.c - cnew)/W
    d2 = (Delta + d1).astype(dtype)          # shift for b: (b.c - cnew)/W

    S = pascal_shift(a.S, d1, a.binom) + pascal_shift(b.S, d2, b.binom)
    U = pascal_shift(a.U, d1, a.binom) + pascal_shift(b.U, d2, b.binom)

    # S_1 vanishes by construction of cnew.  Setting it rather than relying on
    # the cancellation keeps it exactly zero, which the Cholesky in
    # LocalPolyFit depends on (it is what makes p_0 = G_00 and p_1 = G_11).
    S[..., 1] = 0

    return MomentSet(N.astype(dtype), cnew, S, U, n, W)
