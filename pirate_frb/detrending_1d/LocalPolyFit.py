"""
The per-window least-squares solve (see notes/tree_dedispersion.tex,
section "Time detrending algorithm 1: local polynomial subtraction", subsection "The estimator").

Given the moments of a window, we fit a degree-n polynomial in x = (u-c)/W to
the valid samples, and evaluate it back at the window center.  The normal
equations G a = U have G_{jl} = S_{j+l}, a Hankel matrix which needs no assembly
beyond indexing.  Because the moments are taken about the centroid, S_1 = 0 and
hence G_01 = 0.

There is no regularizer.  Instead the solve reports a conditioning statistic

    rmin = min_i (p_i / G_ii),   rmin = 0 if any G_ii = 0,

and Detrender masks the output sample when rmin falls below a threshold.  rmin
lies in [0,1] and is the smallest relative Cholesky pivot, so the equilibrated
condition number of G is roughly 1/rmin.

Because G_01 = 0 we get p_0 = G_00 = nv and p_1 = G_11 exactly, so the pivots
are in natural (level, slope, curvature) order and p_i/G_ii is the fraction of
order-i information surviving after the lower orders are projected out.  For
n = 2 only i = 2 can pull rmin below 1, unless nv <= 1 (where every valid sample
sits at the centroid, so G_11 = G_22 = 0 and rmin = 0).

Note p_0 is never modified for nv >= 1, which is what makes constant-offset
subtraction exact -- see Detrender.
"""

import numpy as np


def _forward(L, b):
    """Solve L y = b, L lower triangular, batched over leading axes."""
    m = b.shape[-1]
    y = np.empty_like(b)
    for i in range(m):
        s = b[..., i].copy()
        for k in range(i):
            s = s - L[..., i, k] * y[..., k]
        y[..., i] = s / L[..., i, i]
    return y


def _backward(L, y):
    """Solve L^T a = y."""
    m = y.shape[-1]
    a = np.empty_like(y)
    for i in reversed(range(m)):
        s = y[..., i].copy()
        for k in range(i+1, m):
            s = s - L[..., k, i] * a[..., k]
        a[..., i] = s / L[..., i, i]
    return a


def gram(ms):
    """G_{jl} = S_{j+l}, shape (batch..., n+1, n+1)."""
    m = ms.n + 1
    G = np.empty(ms.batch_shape + (m, m), dtype=ms.dtype)
    for j in range(m):
        for l in range(m):
            G[..., j, l] = ms.S[..., j+l]
    return G


def cholesky(G, mu):
    """
    Unregularized Cholesky, returning (L, ratios) with ratios[...,i] = p_i/G_ii
    the raw information fraction at order i.

    mu is a NaN guard, not a tuning parameter.  p_i can be zero (a rank-deficient
    window) or slightly negative (cancellation), and sqrt of either would poison
    the batch; we are computing every sample branch-free, including the ones the
    caller is about to mask.  It is inert on any sample the caller keeps: rmin
    above threshold implies p_i >= eps*G_ii, which exceeds mu = 1e-30 for any
    window with valid samples.
    """
    m = G.shape[-1]
    dtype = G.dtype
    L = np.zeros(G.shape, dtype=dtype)
    ratios = np.zeros(G.shape[:-1], dtype=dtype)

    for i in range(m):
        for j in range(i):
            s = G[..., i, j].copy()
            for k in range(j):
                s = s - L[..., i, k] * L[..., j, k]
            L[..., i, j] = s / L[..., j, j]

        praw = G[..., i, i].copy()
        for k in range(i):
            praw = praw - L[..., i, k] * L[..., i, k]

        gii = G[..., i, i]
        pos = gii > 0
        ratios[..., i] = np.where(pos, praw / np.where(pos, gii, 1), 0)

        L[..., i, i] = np.sqrt(np.maximum(praw, mu).astype(dtype))

    return L, ratios


def solve(ms, u_eval, mu):
    """
    Evaluate the local polynomial fit at buffer index 'u_eval' (broadcastable
    against the batch shape of 'ms').

    Returns (fhat, leverage, rmin).

    'leverage' = w^T G^{-1} w is simultaneously the smoother's H_tt, the variance
    of fhat in units of sigma^2, and Var(r[t]) = sigma^2 (1 - leverage[t]); for a
    fully valid window it equals 9/(8W).  With no regularizer these identities are
    exact on every sample the caller keeps.

    'rmin' is the conditioning statistic described in the module docstring.  On a
    sample whose rmin is below the caller's threshold, fhat and leverage are
    meaningless (the mu guard, not the data, set the smallest pivot).
    """
    dtype, W, n = ms.dtype, ms.W, ms.n
    m = n + 1

    x0 = ((u_eval - ms.c) / W).astype(dtype)

    w = np.empty(ms.batch_shape + (m,), dtype=dtype)
    p = np.ones_like(x0)
    for r in range(m):
        w[..., r] = p
        p = p * x0

    L, ratios = cholesky(gram(ms), mu)
    rmin = ratios.min(axis=-1)

    a = _backward(L, _forward(L, ms.U))
    fhat = (w * a).sum(axis=-1).astype(dtype)

    z = _forward(L, w)
    leverage = (z * z).sum(axis=-1).astype(dtype)

    return fhat, leverage, rmin
