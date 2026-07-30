"""
A dense, deliberately naive reference implementation.

Same mathematics as SplineDetrender, expressed with an explicit (nfreq, N_phi)
design matrix, a dense Gram, and numpy's own Cholesky, with one (beam, time)
sample handled at a time.  It shares no code with the fast path below the level of
knots.py and basis.py, so test_reference_agreement() is a genuine cross-check of
the banded accumulation, the banded factorization, and the substitutions rather
than a restatement of them.

Not for production use: it is O(nfreq * N_phi^2) per sample with no blocking, and
it materializes the dense design matrix.
"""

import numpy as np

from .basis import BasisTable
from .regulator import d1_dense
from .solve import zone_slices


def detrend_reference(d, mask, kv, eta, eps, dtype=np.float64):
    """
    Signature and return values match SplineDetrender.detrend(); see that
    docstring.  Everything is computed in 'dtype'.
    """
    dtype = np.dtype(dtype)
    d = np.asarray(d)
    mask = np.asarray(mask) != 0
    M_ax, nfreq, ntime = d.shape

    table = BasisTable(kv, dtype=dtype)
    Phi = table.dense().astype(dtype)
    R = d1_dense(kv, dtype=dtype)
    zs = zone_slices(kv)

    resid = np.zeros(d.shape, dtype=dtype)
    mask_out = mask.copy()
    rmin = np.zeros((M_ax, ntime, kv.nzone), dtype=dtype)

    # Channel range of each zone, for the expansion step.
    zone_of_channel = kv.zone_id[kv.j0]

    for m in range(M_ax):
        for t in range(ntime):
            w = mask[m, :, t].astype(dtype)
            x = np.where(mask[m, :, t], d[m, :, t], 0).astype(dtype)
            G = (Phi.T * w) @ Phi
            U = Phi.T @ (w * x)
            A = G + dtype.type(eta) * R

            s = np.sqrt(np.where(np.diag(A) > 0, np.diag(A), 1)).astype(dtype)
            Ahat = A / np.outer(s, s)

            a = np.zeros(kv.N_phi, dtype=dtype)
            for z, (lo, hi) in enumerate(zs):
                blk = Ahat[lo:hi, lo:hi]
                if not (np.diag(G)[lo:hi].sum() > 0):
                    rmin[m, t, z] = 0            # zone holds no unmasked channel
                    continue
                L = np.linalg.cholesky(blk)
                rmin[m, t, z] = (np.diag(L) ** 2).min()
                if rmin[m, t, z] >= eps:
                    y = np.linalg.solve(L, (U[lo:hi] / s[lo:hi]).astype(dtype))
                    a[lo:hi] = np.linalg.solve(L.T, y) / s[lo:hi]

            for z in range(kv.nzone):
                if rmin[m, t, z] < eps:
                    mask_out[m, zone_of_channel == z, t] = False

            model = Phi @ a
            resid[m, :, t] = np.where(mask_out[m, :, t],
                                      d[m, :, t].astype(dtype) - model, 0)

    return resid, mask_out, rmin
