"""
Dense, definitional oracle for the fixed-lag Kalman detrender.

No state space and no recursions: for each output t, the sub-problem on [0, t+L] is
assembled as an explicit matrix and solved with np.linalg.  This is the ground truth
that validates KalmanDetrender -- which is itself the python reference implementation
that a GPU production kernel would eventually be validated against, in the same
relation that detrending_1d has to src_lib/Detrender1d.cu.

The estimator is the minimizer of

    chi^2(f) = sum_t m[t] (d[t]-f[t])^2  +  rho sum_t (Delta^k f)[t]^2,   rho = tau^(2k),

restricted to [0, t+L], so the normal equations are (M + rho D_k^T D_k) fhat = M d and
everything else follows from N^-1.  It is O(T^4) and usable to T ~ 256.

The one thing it shares with the implementation is the mask criterion itself
(LocalPolyFit.cholesky), which is a definition rather than something to
cross-validate.  What is being validated is J, which is computed here from a dense
inverse and there from the two information recursions.
"""

import numpy as np

from ..detrending_1d import LocalPolyFit


def difference_matrix(n, k, dtype=np.float64):
    """The (n-k) x n matrix D_k with (D_k f)[t] = (Delta^k f)[t]."""
    from math import comb
    D = np.zeros((n-k, n), dtype=dtype)
    for i in range(k+1):
        coef = ((-1)**(k-i)) * comb(k, i)
        for t in range(n-k):
            D[t, t+i] = coef
    return D


def _state_from_samples(k, dtype=np.float64):
    """
    C with x[t] = C f[t:t+k], i.e. C[j,i] = (-1)^(j-i) C(j,i): the map from k
    consecutive samples of f to the state (f, Delta f, ..., Delta^(k-1) f).
    """
    from math import comb
    C = np.zeros((k, k), dtype=dtype)
    for j in range(k):
        for i in range(j+1):
            C[j, i] = ((-1)**(j-i)) * comb(j, i)
    return C


def kalman_brute_force(d, mask, k, tau, L, eps=1e-3, mu=1e-30, dtype=np.float64):
    """
    d, mask: shape (S, T).  Returns (residual, mask_out, rmin), each of shape
    (S, T-L), for outputs [0, T-L).

    Matches KalmanDetrender.detrend_stream()'s contract, including setting both
    residual and rmin to zero wherever mask_out is false.

    No constant offset appears anywhere: kappa is mathematically inert (the fit
    reproduces constants), so the residual is the same with or without it.
    """
    d = np.asarray(d, dtype=dtype)
    mask = np.asarray(mask)
    assert d.ndim == 2 and d.shape == mask.shape
    S_ax, T = d.shape
    nout = T - L
    assert nout > 0
    assert L >= k-1, 'need L >= k-1 to read the state covariance off the sample grid'

    rho = dtype(tau) ** (2*k)
    C = _state_from_samples(k, dtype)

    resid = np.zeros((S_ax, nout), dtype=dtype)
    mout = np.zeros((S_ax, nout), dtype=bool)
    rmn = np.zeros((S_ax, nout), dtype=dtype)

    for s in range(S_ax):
        for t in range(nout):
            n = t + L + 1                       # sub-problem is [0, t+L] inclusive
            m_sub = (mask[s, :n] != 0).astype(dtype)

            # N is positive definite iff the sub-problem holds >= k valid samples:
            # the null space of D_k is the polynomials of degree < k, and a nonzero
            # one has at most k-1 roots.  Below that there is nothing to solve.
            if m_sub.sum() < k:
                continue

            D = difference_matrix(n, k, dtype)
            N = np.diag(m_sub) + rho * (D.T @ D)
            Sigma = np.linalg.inv(N)

            fhat = (Sigma @ (m_sub * d[s, :n]))[t]

            # J[t] is the precision of the MARGINAL posterior of x[t], which is what
            # the two information recursions combine to.  Read it off Sigma: the
            # marginal covariance of (f[t], .., f[t+k-1]) is a k x k submatrix, and
            # x[t] = C f[t:t+k].
            Ssub = Sigma[t:t+k, t:t+k]
            J = np.linalg.inv(C @ Ssub @ C.T)
            _L, ratios = LocalPolyFit.cholesky(J, mu)
            rmin = float(ratios.min())

            keep = bool(mask[s, t]) and (rmin >= eps)
            if not keep:
                continue

            resid[s, t] = d[s, t] - fhat
            mout[s, t] = True
            rmn[s, t] = rmin

    return resid, mout, rmn


def impulse_kernel(det, mask_row, t_out):
    """
    Row t_out of the smoothing operator H_L, by pushing unit impulses through 'det'.

    One spectator per impulse position, so a single detrend_chunk() call gives the
    whole row.  'det' must have subtract_offset=False: a data-dependent kappa would
    shift under the impulse and the map would not be linear.

    'mask_row' has length det.buflen and is shared by every spectator (only the
    impulse position varies).  Returns kern of length det.buflen with
    kern[u] = d(fhat[t_out])/d(d[u]).
    """
    assert not det.subtract_offset, 'impulse_kernel() needs subtract_offset=False'
    Tbuf = det.buflen
    assert mask_row.shape == (Tbuf,)
    assert 0 <= t_out < det.chunk_size

    dd = np.zeros((Tbuf, Tbuf), dtype=det.dtype)
    np.fill_diagonal(dd, 1.0)
    mm = np.broadcast_to(mask_row, (Tbuf, Tbuf)).copy()

    (resid, mask_out, rmin), _ = det.detrend_chunk(dd, mm, det.initial_state(Tbuf))

    # fhat = dz - resid, and dz[u_out] is 1 only for the spectator whose impulse sits
    # on the output sample itself.
    valid = bool(mask_row[t_out])
    kern = np.array([(1.0 if (u == t_out and valid) else 0.0) - resid[u, t_out]
                     for u in range(Tbuf)])
    return kern, mask_out[:, t_out]
