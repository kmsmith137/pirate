"""
The chunked, block-scan 1-d detrender (see notes/tree_dedispersion.tex, section
"Detrending").

Geometry.  The window is 2W+1 samples and the scan blocks are B = 2W samples, so
a window spans *exactly* two adjacent blocks for every alignment: writing
q = t-W for the window start and p = q mod B,

    window = Suff_b[p]  (offsets p..B-1, B-p samples)
           + Pref_{b+1}[p]  (offsets 0..p, p+1 samples)

and (B-p) + (p+1) = B+1 = 2W+1 for all p, with no special case at either end.
An odd window over even blocks is what makes this work; it also keeps the fit
exactly symmetric and puts the evaluation point on a real sample.

The block lattice is anchored at (chunk start - W), and Tc is required to be a
multiple of B, so the lattice is chunk-invariant, each chunk needs exactly
Tc/B + 1 blocks, and every output is computed exactly once from the same two
blocks regardless of which chunk it falls in.  The design is therefore seam-free
by construction.

Moment state is never carried across chunks; each chunk's scans are built from
scratch, and the constant offset is recomputed from the buffer in hand, so
detrend_chunk() is a pure function of its arguments.  A Detrender therefore holds
no mutable state at all: results are reproducible on replay, chunks may be
processed in any order, and detrend_chunk() can be tested in isolation.
"""

import numpy as np

from .MomentSet import MomentSet, merge
from .scan import tree_prefix_scan, tree_suffix_scan
from . import LocalPolyFit


class Detrender:
    def __init__(self, W, n=2, chunk_size=2048, dtype=np.float32,
                 eps=1e-3, mu=1e-30, subtract_offset=True):
        # eps is a masking threshold on rmin (see LocalPolyFit), not a
        # regularizer strength; mu is a NaN guard.
        B = 2*W
        if chunk_size % B != 0:
            raise ValueError(f'chunk_size={chunk_size} must be a multiple of B=2*W={B}')
        if n < 0 or 2*n+1 > 2*W+1:
            raise ValueError(f'bad degree n={n} for W={W}')

        self.W = W
        self.n = n
        self.B = B
        self.chunk_size = chunk_size
        self.dtype = np.dtype(dtype)
        self.eps = eps
        self.mu = mu
        self.subtract_offset = subtract_offset

        self.buflen = chunk_size + 2*W
        self.nblocks = chunk_size // B + 1
        assert self.nblocks * B == self.buflen

        # Per-output block/offset lookup.  In buffer coordinates the window for
        # local output j spans [j, j+B] inclusive, so q = j.
        j = np.arange(chunk_size)
        self.blk = j // B
        self.off = j % B

    # ------------------------------------------------------------------ chunk

    def detrend_chunk(self, d_buf, mask_buf):
        """
        d_buf, mask_buf: shape (S, chunk_size + 2W), where S is a spectator axis
        carrying one entry per (beam,freq) pair.  Every operation is elementwise
        along S; there is no coupling between spectator entries.

        Returns (residual, mask_out, leverage, rmin), each of shape (S, chunk_size).

        mask_out is the *expanded* mask: an output sample is dropped if its input
        sample was masked, or if its window is too ill-conditioned to fit
        (rmin < eps).  Where mask_out is false the residual is meaningless and is
        set to zero -- note this differs from the in-place formulation in
        notes/tree_dedispersion.tex, which leaves masked samples untouched.
        """
        d_buf = np.asarray(d_buf)
        mask_buf = np.asarray(mask_buf)
        if d_buf.ndim != 2 or d_buf.shape != mask_buf.shape:
            raise ValueError('d_buf and mask_buf must be 2-d with the same shape')
        if d_buf.shape[1] != self.buflen:
            raise ValueError(f'expected buffer length {self.buflen}, got {d_buf.shape[1]}')

        dtype, W, n = self.dtype, self.W, self.n
        S_ax, T = d_buf.shape

        d_buf = d_buf.astype(dtype, copy=False)
        m_buf = (mask_buf != 0).astype(dtype)

        # ---- constant offset.  The fit reproduces constants exactly (the basis
        # contains the constant function, and p_0 is never floored), so this
        # leaves the residual unchanged in exact arithmetic; it exists only so
        # that the accumulators sum terms of size |d-kappa| rather than |d|.
        #
        # kappa is the masked mean of *this* buffer, not of the previous chunk.
        # Any value is mathematically exact, so accuracy is not the point -- but
        # a value inherited from a previous chunk can be arbitrarily stale, and
        # in particular collapses to 0 when that chunk had no valid samples,
        # which silently disables the offset subtraction for this one.  The
        # requirement is |d - kappa| <~ 1e3 sigma, since the residual error is
        # about 3.4*eps_mach*|d - kappa|; note this bounds the *spread* of d
        # within a buffer, not its absolute level, and no single additive
        # constant can help when a large step falls inside a window.
        if self.subtract_offset:
            kappa = self._masked_mean(d_buf, m_buf)
        else:
            kappa = np.zeros(S_ax, dtype=dtype)

        dz = np.where(m_buf > 0, d_buf - kappa[:, None], 0).astype(dtype)

        # ---- leaves, reshaped to (S, nblocks, B)
        u = np.arange(T, dtype=dtype)
        leaves = self._to_blocks(
            MomentSet.leaves(np.broadcast_to(u, (S_ax, T)), m_buf, dz, n, W, dtype), S_ax)

        pref = tree_prefix_scan(leaves)
        suff = tree_suffix_scan(leaves)

        # ---- van Herk combine: one merge per output
        sl = (slice(None), self.blk, self.off)
        sr = (slice(None), self.blk + 1, self.off)
        ms = merge(suff.take_batch(sl), pref.take_batch(sr))

        u_eval = (np.arange(self.chunk_size, dtype=dtype) + W)[None, :]
        fhat, leverage, rmin = LocalPolyFit.solve(ms, u_eval, self.mu)

        # Mask expansion.  A window with rmin < eps cannot determine a degree-n
        # fit, so we drop the sample rather than shrinking the fit toward lower
        # order.  For a sample we keep, the two are the same thing: rmin >= eps
        # implies p_i >= eps*G_ii for every i, so a floor would have been inert.
        # Dropping instead means surviving samples carry no shrinkage bias at
        # all, and polynomial reproduction is exact on all of them.
        #
        # This is a single pass: the expansion applies to the output mask only,
        # and is not fed back into the moments of neighbouring windows.
        out = slice(W, W + self.chunk_size)
        d_out, m_out = d_buf[:, out], m_buf[:, out]
        mask_out = (m_out > 0) & (rmin >= self.eps)
        resid = np.where(mask_out, (d_out - kappa[:, None]) - fhat, 0).astype(dtype)
        return resid, mask_out, leverage, rmin

    # ----------------------------------------------------------------- stream

    def detrend_stream(self, d, mask):
        """
        d, mask: shape (S, T) with (T - 2W) a positive multiple of chunk_size.
        Returns (residual, mask_out, leverage, rmin) for samples [W, T-W), i.e.
        each of shape (S, T - 2W).

        This is the path that exercises chunk stitching.  Because the block
        lattice is chunk-invariant, the result should agree sample-by-sample
        with detrend_reference() applied to the whole stream at once.
        """
        d = np.asarray(d)
        mask = np.asarray(mask)
        nout = d.shape[1] - 2*self.W
        if nout <= 0 or nout % self.chunk_size != 0:
            raise ValueError(f'(T - 2W) = {nout} must be a positive multiple of '
                             f'chunk_size={self.chunk_size}')

        cols = [[], [], [], []]
        for i in range(nout // self.chunk_size):
            lo = i * self.chunk_size
            for c, x in zip(cols, self.detrend_chunk(d[:, lo:lo+self.buflen],
                                                     mask[:, lo:lo+self.buflen])):
                c.append(x)
        return tuple(np.concatenate(c, axis=1) for c in cols)

    # ---------------------------------------------------------------- helpers

    def _masked_mean(self, d, m):
        # Select with np.where rather than weighting by m: a masked sample may hold
        # anything at all, and 0*inf and 0*nan are nan, which would poison kappa and
        # hence every output.  Selecting keeps masked values strictly unread, which
        # test_masked_data_unused() checks by bit-identity under nan/inf poisoning.
        nv = m.sum(axis=1)
        safe = nv > 0
        tot = np.where(m > 0, d, 0).sum(axis=1)
        return np.where(safe, tot / np.where(safe, nv, 1), 0).astype(self.dtype)

    def _to_blocks(self, leaves, S_ax):
        shp = (S_ax, self.nblocks, self.B)
        return MomentSet(leaves.nv.reshape(shp), leaves.c.reshape(shp),
                         leaves.S.reshape(shp + (2*self.n+1,)),
                         leaves.U.reshape(shp + (self.n+1,)),
                         self.n, self.W)
