"""
The window-moment pass: per-time (G,U) -> window moments, by a fixed stencil.

This is the whole of the time axis.  Everything upstream (reduce.accumulate) and
downstream (assemble, solve) is unchanged by W.

DIRECT STENCIL, NOT VAN HERK.  The 1-d LPS detrender reduces its moving window by
a van Herk block decomposition over a moment monoid, which is O(1) per sample
independent of W.  That is not ported.  At the window sizes here a van Herk
decomposition costs the same to within a flop per full-resolution sample, and a
stencil buys: no block lattice, no prefix/suffix storage, no Pascal shift, no
carried block partials, and parallelism in t.  It is also what makes
detrend_chunk() a pure function of its arguments.

FOLDED BY PARITY.  On a symmetric window the orthogonal basis polynomials have
definite parity, so every stencil is either even or odd in s and the window can be
summed in half the multiplies:

    even:  sum_k c_k (G[t+k] + G[t-k]) + c_0 G[t]
    odd:   sum_k c_k (G[t+k] - G[t-k])

That is the small-W optimization, and it also makes the even/odd structure exact
rather than a cancellation of rounding errors -- which matters, because "n=1
reduces to n=0 for a window-constant mask" depends on odd moments being exactly
zero.

CHUNK INVARIANCE is free here and does NOT need the binary tree that reduce.py
uses over frequency.  Output t always sums the same 2W+1 buffer samples in the
same fixed order, whatever the chunk length, so the result is bit-identical
across chunkings provided the caller supplies consistent padding -- which is the
caller's contract (see SplineDetrender.detrend_chunk).
"""

import numpy as np


def _apply_stencil(X, coef, parity, W, ntime):
    """
    Correlate X (..., ntime+2W, ...) along axis 1 with a length-(2W+1) stencil.

    'coef' is indexed by k = 0..2W with s = k-W; only the s >= 0 half is read,
    since 'parity' fixes the other half.  Returns shape (..., ntime, ...).
    """
    out = coef[W] * X[:, W:W+ntime]
    for k in range(1, W+1):
        hi = X[:, W+k:W+k+ntime]
        lo = X[:, W-k:W-k+ntime]
        out = out + coef[W+k] * (hi + lo if parity > 0 else hi - lo)
    return out


def window_moments(G, U, tb, ntime):
    """
    G: (M, ntime+2W, N_phi, n_phi+1) banded, U: (M, ntime+2W, N_phi).

    Returns (Mcal, Vcal) with

        Mcal: (npair, M, ntime, N_phi, n_phi+1)   Mcal[p] = sum_s gstencil[p,s] G
        Vcal: (n+1,   M, ntime, N_phi)            Vcal[q] = sum_s ustencil[q,s] U

    in G.dtype.  npair = (n+1)(n+2)/2 covers the pairs (q <= r); the assembled
    matrix is symmetric in (q,r) so the lower pairs are never needed.
    """
    W = tb.W
    if G.shape[1] != ntime + 2*W:
        raise ValueError(f'window_moments: expected {ntime+2*W} buffer samples, '
                         f'got {G.shape[1]}')
    dt = G.dtype
    gs = tb.gstencil.astype(dt)
    us = tb.ustencil.astype(dt)

    Mcal = np.empty((tb.npair,) + (G.shape[0], ntime) + G.shape[2:], dtype=dt)
    Vcal = np.empty((tb.n+1,) + (U.shape[0], ntime) + U.shape[2:], dtype=dt)
    for p in range(tb.npair):
        Mcal[p] = _apply_stencil(G, gs[p], int(tb.gparity[p]), W, ntime)
    for q in range(tb.n+1):
        Vcal[q] = _apply_stencil(U, us[q], int(tb.uparity[q]), W, ntime)
    return Mcal, Vcal


def zone_live_counts(G, kv, W, ntime):
    """
    (M, ntime, nzone) int array: for each output sample and zone, the number of
    window offsets at which that zone holds at least one unmasked channel.

    This is the 2-d generalization of the 1-d dead-zone test.  A degree-n fit in
    time is singular unless the zone has data at >= n+1 DISTINCT offsets -- no
    nonzero degree-n polynomial may vanish at every offset that carries data --
    and that is a rank condition on the window, not a channel count.  Computed
    structurally from G rather than inferred from a pivot magnitude, for the same
    reason as the 1-d test (see solve.py).
    """
    from .solve import zone_slices
    out = np.zeros((G.shape[0], ntime, kv.nzone), dtype=np.int64)
    for z, (lo, hi) in enumerate(zone_slices(kv)):
        live = G[:, :, lo:hi, 0].sum(axis=-1) > 0          # (M, ntime+2W)
        for k in range(2*W+1):
            out[:, :, z] += live[:, k:k+ntime]
    return out
