"""
The two full-resolution passes: (d,mask) -> (G,U), and coefficients -> residual.

These are the ONLY places the full-resolution arrays are touched.  Everything
between them acts on (G,U), which is about N_phi(n_phi+2) floats per (beam,time)
against nfreq for the data.  That is not an implementation convenience: (G,U) are
sufficient statistics for the whole problem at each (beam,time),

    G_jl = sum_f w[f] phi_j[f] phi_l[f],   U_j = sum_f w[f] phi_j[f] d[f],

so they are the entire interface between the frequency axis and whatever the time
axis does.  The 2-d detrender will give G and U an extra time-moment index and
change the solve; this file will not need to change.

Array layouts, which differ deliberately between the two sides of this interface:

    full resolution      (M, nfreq, ntime)     -- frequency is the strided axis
    reduced              (M, ntime, N_phi...)  -- (M,ntime) is the batch shape

The reduced layout puts the coefficient index last, which is what the batched
dense Cholesky in solve.py wants.

G is stored banded, G[..., j, b] = G_{j,j+b} for b in [0, n_phi].  Entries with
j+b >= N_phi cannot arise: channel f contributes to rows j0[f]-n_phi+a with band
b-a, so j+band = j0[f]-n_phi+b <= j0[f] <= N_phi-1.

Three rules that are not negotiable:

  - SELECT masked samples away, never weight by the mask.  A masked sample may
    hold anything, including NaN from a dropped packet, and 0*nan is nan.  This is
    what test_masked_data_unused() checks by bit-identity under poisoning.
    (Weighting by the mask is fine in the G accumulation, where the mask IS the
    data.)
  - Reduce over frequency by an explicit BINARY TREE over fixed-size channel
    blocks, not by einsum or ndarray.sum().  Two reasons, and the second is a
    correctness requirement rather than an accuracy one.  Accuracy: a zone can
    span thousands of channels and a sequential sum loses precision that a tree
    does not.  Reproducibility: numpy's own pairwise summation blocks by STRIDE,
    so its grouping over frequency silently changes with the length of the time
    axis, and the same output computed in a 1-sample chunk and a 24-sample chunk
    then differs in the last bits.  An explicit tree over a padded channel axis
    depends only on the channel count, so chunk invariance is exact -- which is
    what test_chunk_invariance() asserts.
  - Likewise reduce the n_phi+1 basis terms in evaluate() by an explicit loop, for
    the same reproducibility reason.
"""

import numpy as np

# Channels per accumulation block.  Bounds the (M, block, ntime, npair) temporary;
# only the float32 rounding depends on the value, and it must be a constant rather
# than derived from the array shape, or chunk invariance breaks.
CHANNEL_BLOCK = 512


def _spans(kv):
    """
    (span_index, lo, hi) for each non-empty knot interval, in increasing order.

    Channels in one span all have the same j0, hence contribute to the same
    n_phi+1 rows, which is what lets the accumulation be a handful of array
    products instead of a loop over channels.
    """
    j0 = kv.j0
    edges = np.flatnonzero(np.diff(j0)) + 1
    los = np.concatenate(([0], edges))
    his = np.concatenate((edges, [kv.nfreq]))
    return [(int(j0[lo]), int(lo), int(hi)) for lo, hi in zip(los, his)]


def tree_sum(x, axis):
    """
    Sum along 'axis' by an explicit binary tree: pad to a power of two, then halve.

    Every step is an elementwise add of two identically shaped arrays, so the
    result depends on the length of 'axis' and on nothing else -- in particular not
    on the shape of the other axes, which is what numpy's own pairwise summation
    does not guarantee.  See the module docstring for why that matters here.
    """
    n = x.shape[axis]
    if n == 0:
        raise ValueError('tree_sum() of an empty axis')
    p = 1 << (n-1).bit_length()
    if p != n:
        pad = [(0, 0)] * x.ndim
        pad[axis] = (0, p-n)
        x = np.pad(x, pad)
    while x.shape[axis] > 1:
        h = x.shape[axis] // 2
        lo, hi = [slice(None)]*x.ndim, [slice(None)]*x.ndim
        lo[axis], hi[axis] = slice(0, h), slice(h, None)
        x = x[tuple(lo)] + x[tuple(hi)]
    return np.squeeze(x, axis=axis)


def accumulate(d, mask, table):
    """
    d, mask: shape (M, nfreq, ntime).  Returns (G, U) with

        G: (M, ntime, N_phi, n_phi+1)  banded, see the module docstring
        U: (M, ntime, N_phi)

    both in table.dtype.  'mask' is interpreted as a boolean; d is read only where
    it is true.
    """
    kv = table.kv
    dtype = table.dtype
    d = np.asarray(d)
    mask = np.asarray(mask)
    if d.ndim != 3 or d.shape != mask.shape:
        raise ValueError('d and mask must be 3-d with the same shape')
    if d.shape[1] != kv.nfreq:
        raise ValueError(f'expected {kv.nfreq} channels, got {d.shape[1]}')

    M_ax, nfreq, ntime = d.shape
    N_phi, n_phi = kv.N_phi, kv.n_phi

    G = np.zeros((M_ax, ntime, N_phi, n_phi+1), dtype=dtype)
    U = np.zeros((M_ax, ntime, N_phi), dtype=dtype)

    mf = (mask != 0)

    for j0, lo, hi in _spans(kv):
        jlo = j0 - n_phi
        gpart, upart = [], []
        for blo in range(lo, hi, CHANNEL_BLOCK):
            bhi = min(blo + CHANNEL_BLOCK, hi)
            sel = mf[:, blo:bhi, :]
            w = sel.astype(dtype)[:, :, :, None]
            wd = np.where(sel, d[:, blo:bhi, :], 0).astype(dtype)[:, :, :, None]
            gpart.append(tree_sum(w * table.prod[blo:bhi][None, :, None, :], axis=1))
            upart.append(tree_sum(wd * table.phi[blo:bhi][None, :, None, :], axis=1))
        gsum = tree_sum(np.stack(gpart), axis=0)
        usum = tree_sum(np.stack(upart), axis=0)

        for p in range(table.npair):
            a, b = int(table.pair_a[p]), int(table.pair_b[p])
            G[:, :, jlo+a, b-a] += gsum[:, :, p]
        for a in range(n_phi+1):
            U[:, :, jlo+a] += usum[:, :, a]

    return G, U


def evaluate(a, table):
    """
    Coefficients (M, ntime, N_phi) -> model (M, nfreq, ntime), in table.dtype.

    The n_phi+1 terms are summed by an explicit loop in a fixed order; see the
    module docstring.
    """
    kv = table.kv
    a = np.asarray(a, dtype=table.dtype)
    M_ax, ntime, N_phi = a.shape
    if N_phi != kv.N_phi:
        raise ValueError(f'expected N_phi={kv.N_phi}, got {N_phi}')

    out = np.zeros((M_ax, kv.nfreq, ntime), dtype=table.dtype)
    for j0, lo, hi in _spans(kv):
        jlo = j0 - kv.n_phi
        acc = None
        for aa in range(kv.n_phi+1):
            term = table.phi[lo:hi, aa][None, :, None] * a[:, None, :, jlo+aa]
            acc = term if acc is None else acc + term
        out[:, lo:hi, :] = acc
    return out


def band_to_dense(G):
    """(..., N_phi, n_phi+1) banded -> (..., N_phi, N_phi) symmetric dense."""
    G = np.asarray(G)
    N, nb = G.shape[-2], G.shape[-1]
    out = np.zeros(G.shape[:-2] + (N, N), dtype=G.dtype)
    for b in range(nb):
        j = np.arange(N-b)
        out[..., j, j+b] = G[..., j, b]
        if b > 0:
            out[..., j+b, j] = G[..., j, b]
    return out


def dense_to_band(A, n_phi):
    """Inverse of band_to_dense(); the caller asserts A is banded to n_phi."""
    A = np.asarray(A)
    N = A.shape[-1]
    out = np.zeros(A.shape[:-2] + (N, n_phi+1), dtype=A.dtype)
    for b in range(n_phi+1):
        j = np.arange(N-b)
        out[..., j, b] = A[..., j, j+b]
    return out
