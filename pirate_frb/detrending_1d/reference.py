"""
Float64 direct-summation reference for the 1-d detrender.

No blocks, no scans: for each output, the moments of its window are computed
directly from the definition and handed to the same LocalPolyFit.solve().  This
is O(T*W) and slow, and is the ground truth for the tests that need one.

What this does and does not check.  It is an independent implementation of the
*block decomposition and scans* only -- it shares solve(), MomentSet.direct() and
the offset convention with Detrender, so a bug in any of those would appear in
both and the comparison would not catch it.  Those are covered instead by
test_solve and test_polynomial_exactness, whose expected answers do not come from
another implementation.  The small duplication of the masked-mean below is
deliberate for the same reason: it costs five lines and keeps the offset
convention verifiable by eye against Detrender.

Note that test_polynomial_exactness() does *not* need it -- its expected answer
is known analytically -- which is why it can run before this module is trusted.
"""

import numpy as np

from .MomentSet import MomentSet
from . import LocalPolyFit


def detrend_reference(d, mask, W, n=2, eps=1e-3, mu=1e-30, dtype=np.float64,
                      subtract_offset=True, kappa=None, max_outputs_per_pass=4096):
    """
    d, mask: shape (S, T).  Returns (residual, mask_out, leverage, rmin), each of
    shape (S, T - 2W), for output samples [W, T-W).

    Unlike Detrender, this has no chunk structure at all, so it also serves as
    the check that chunking introduces nothing.
    """
    d = np.asarray(d)
    mask = np.asarray(mask)
    assert d.ndim == 2 and d.shape == mask.shape
    dtype = np.dtype(dtype)

    S_ax, T = d.shape
    nout = T - 2*W
    assert nout > 0

    d = d.astype(dtype, copy=False)
    m = (mask != 0).astype(dtype)

    if kappa is None:
        if subtract_offset:
            nv = m.sum(axis=1)
            safe = nv > 0
            kappa = np.where(safe, (m*d).sum(axis=1)/np.where(safe, nv, 1), 0).astype(dtype)
        else:
            kappa = np.zeros(S_ax, dtype=dtype)
    kappa = np.asarray(kappa, dtype=dtype)

    dz = np.where(m > 0, d - kappa[:, None], 0).astype(dtype)

    resid = np.empty((S_ax, nout), dtype=dtype)
    mout = np.empty((S_ax, nout), dtype=bool)
    lev = np.empty((S_ax, nout), dtype=dtype)
    rmn = np.empty((S_ax, nout), dtype=dtype)

    # Process outputs in passes to bound the (nout, 2W+1) gather.
    for lo in range(0, nout, max_outputs_per_pass):
        hi = min(lo + max_outputs_per_pass, nout)
        j = np.arange(lo, hi)
        idx = j[:, None] + np.arange(2*W+1)[None, :]          # (nj, 2W+1)
        u = idx.astype(dtype)[None, :, :]                     # (1, nj, 2W+1)
        mm = m[:, idx]                                        # (S, nj, 2W+1)
        dd = dz[:, idx]

        ms = MomentSet.direct(u, mm, dd, n, W, dtype, axis=-1)
        u_eval = (j + W).astype(dtype)[None, :]
        fhat, lv_, rm_ = LocalPolyFit.solve(ms, u_eval, mu)

        d_out = d[:, j+W]
        m_out = m[:, j+W]
        mo_ = (m_out > 0) & (rm_ >= eps)
        resid[:, lo:hi] = np.where(mo_, (d_out - kappa[:, None]) - fhat, 0)
        mout[:, lo:hi] = mo_
        lev[:, lo:hi] = lv_
        rmn[:, lo:hi] = rm_

    return resid, mout, lev, rmn
