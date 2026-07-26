"""
The per-window regularized least-squares solve (see notes/tree_dedispersion.tex,
section "Detrending", subsection "The estimator").

Given the moments of a window, we fit a degree-n polynomial in x = (u-c)/W to
the valid samples, and evaluate it back at the window center.  The normal
equations G a = U have G_{jl} = S_{j+l}, a Hankel matrix which needs no assembly
beyond indexing.  Because the moments are taken about the centroid, S_1 = 0 and
hence G_01 = 0.

Regularization is a multiplicative floor on the Cholesky pivots,

    p_i -> max(p_i, eps*G_ii, mu).

This is 'fall back to a lower-order fit', applied continuously and with no
branch: a rank-k G has p_i = 0 for i >= k, so flooring freezes out exactly the
orders the data cannot determine.  It is multiplicative, hence covariant under
diagonal rescaling of the basis, so it needs no calibration against sigma,
against the trend amplitude, or against the effective window width.  Crucially
it is inactive on a well-populated window (r_2 = 0.44 for a full one), so it
introduces no bias there.

Because G_01 = 0 we get p_0 = G_00 = nv and p_1 = G_11 exactly, so the pivots
are in natural (level, slope, curvature) order and r_i = p_i/G_ii in [0,1] is
the fraction of order-i information surviving after the lower orders are
projected out.  Note p_0 is never floored (for nv >= 1), which is what makes
constant-offset subtraction exact -- see Detrender.
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


def cholesky_floored(G, eps, mu):
    """
    Cholesky with multiplicatively floored pivots.  Returns (L, flagged, ratios),
    where 'flagged' says the floor was applied at some order (so the fit was
    shrunk toward lower order, and polynomial reproduction no longer holds), and
    ratios[...,i] = p_i/G_ii is the raw information fraction at order i.
    """
    m = G.shape[-1]
    dtype = G.dtype
    L = np.zeros(G.shape, dtype=dtype)
    flagged = np.zeros(G.shape[:-2], dtype=bool)
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

        pf = np.maximum(np.maximum(praw, eps * gii), mu).astype(dtype)
        flagged |= (pf > praw)
        L[..., i, i] = np.sqrt(pf)

    return L, flagged, ratios


def solve(ms, u_eval, eps, mu):
    """
    Evaluate the local polynomial fit at buffer index 'u_eval' (broadcastable
    against the batch shape of 'ms').

    Returns (fhat, leverage, flagged, ratios).

    'leverage' = w^T G_reg^{-1} w is simultaneously the smoother's H_tt, the
    variance of fhat in units of sigma^2, and the window-quality statistic; for
    a fully valid window it equals 9/(8W).  The detrended residual then has
    Var(r[t]) = sigma^2 (1 - leverage[t]), which is *not* t-independent.
    """
    dtype, W, n = ms.dtype, ms.W, ms.n
    m = n + 1

    x0 = ((u_eval - ms.c) / W).astype(dtype)

    w = np.empty(ms.batch_shape + (m,), dtype=dtype)
    p = np.ones_like(x0)
    for r in range(m):
        w[..., r] = p
        p = p * x0

    G = gram(ms)
    L, flagged, ratios = cholesky_floored(G, eps, mu)

    a = _backward(L, _forward(L, ms.U))
    fhat = (w * a).sum(axis=-1).astype(dtype)

    z = _forward(L, w)
    leverage = (z * z).sum(axis=-1).astype(dtype)

    return fhat, leverage, flagged, ratios
