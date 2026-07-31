"""
A dense, deliberately naive reference implementation.

Same mathematics as SplineDetrender, expressed with an explicit design matrix
over (channel, window offset) pairs and numpy's own dense solve, one (beam, time)
sample at a time.  It shares no code with the fast path below the level of
knots.py, basis.py and timebasis.py -- in particular it never forms window
moments, never uses the banded layout, and never touches the coefficient-major
index order -- so test_reference_agreement() is a genuine cross-check of the
moment stencil, the block assembly and the banded factorization rather than a
restatement of them.

Not for production use: it materializes a ((2W+1) nfreq, N_phi(n+1)) design
matrix per output sample.
"""

import numpy as np

from .basis import BasisTable
from .regulator import d1_dense
from .timebasis import TimeBasis
from .solve import zone_slices


def detrend_reference(d_buf, mask_buf, kv, n=0, W=0, eta=3e-3, eps=1e-7,
                      dtype=np.float64, orthogonal_time=True):
    """
    Signature and return values match SplineDetrender.detrend_chunk(); see that
    docstring.  Everything is computed in 'dtype'.
    """
    dtype = np.dtype(dtype)
    d_buf = np.asarray(d_buf)
    mask_buf = np.asarray(mask_buf) != 0
    M_ax, nfreq, nbuf = d_buf.shape
    ntime = nbuf - 2*W

    table = BasisTable(kv, dtype=dtype)
    Phi = table.dense().astype(dtype)
    R = d1_dense(kv, dtype=dtype)
    tb = TimeBasis(n, W, orthogonal=orthogonal_time, dtype=dtype)
    P = tb.P.astype(dtype)
    T = (P.T @ P).astype(dtype)
    N, K = kv.N_phi, kv.N_phi*(n+1)
    zs = zone_slices(kv)
    zone_of_channel = kv.zone_id[kv.j0]

    resid = np.zeros((M_ax, nfreq, ntime), dtype=dtype)
    mask_out = mask_buf[:, :, W:W+ntime].copy()
    rmin = np.zeros((M_ax, ntime, kv.nzone), dtype=dtype)

    # Design matrix column (j,q) at row (f,s): phi_j(f) * p_q(s).  Built once for
    # the full window and then row-selected by the mask.
    X_full = np.zeros(((2*W+1)*nfreq, K), dtype=dtype)
    for k in range(2*W+1):
        for q in range(n+1):
            X_full[k*nfreq:(k+1)*nfreq, np.arange(N)*(n+1)+q] = Phi * P[k, q]

    for m in range(M_ax):
        for t in range(ntime):
            sel = mask_buf[m, :, t:t+2*W+1].T.reshape(-1)          # (k, f) order
            y = np.where(mask_buf[m, :, t:t+2*W+1],
                         d_buf[m, :, t:t+2*W+1], 0).T.reshape(-1).astype(dtype)
            X = X_full[sel]
            A = X.T @ X + dtype.type(eta) * np.kron(R, T)
            rhs = X_full[sel].T @ y[sel]

            alpha = np.zeros(K, dtype=dtype)
            for z, (lo, hi) in enumerate(zs):
                Ilo, Ihi = lo*(n+1), hi*(n+1)
                # Rank test: the zone needs data at >= n+1 distinct offsets.
                nlive = sum(1 for k in range(2*W+1)
                            if mask_buf[m, zone_of_channel == z, t+k].any())
                if nlive < n+1:
                    rmin[m, t, z] = 0
                    continue
                blk = A[Ilo:Ihi, Ilo:Ihi]
                s = np.sqrt(np.where(np.diag(blk) > 0, np.diag(blk), 1)).astype(dtype)
                Ah = blk / np.outer(s, s)
                try:
                    L = np.linalg.cholesky(Ah)
                except np.linalg.LinAlgError:
                    rmin[m, t, z] = 0
                    continue
                rmin[m, t, z] = (np.diag(L) ** 2).min()
                if rmin[m, t, z] >= eps:
                    yv = np.linalg.solve(L, (rhs[Ilo:Ihi] / s).astype(dtype))
                    alpha[Ilo:Ihi] = np.linalg.solve(L.T, yv) / s

            for z in range(kv.nzone):
                if rmin[m, t, z] < eps:
                    mask_out[m, zone_of_channel == z, t] = False

            a = alpha.reshape(N, n+1) @ tb.eval0.astype(dtype)
            model = Phi @ a
            resid[m, :, t] = np.where(mask_out[m, :, t],
                                      d_buf[m, :, t+W].astype(dtype) - model, 0)

    return resid, mask_out, rmin
