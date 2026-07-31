"""
Assembly of the tensor-product normal equations, in banded form.

    Gcal[(j,q),(l,r)] = Mcal[q,r]_jl + eta * D_1[j,l] * T[q,r]
    Ucal[(j,q)]       = Vcal[q]_j

COEFFICIENT-MAJOR INDEXING IS LOAD-BEARING, not a layout preference.  With
I = j(n+1)+q the matrix is banded with half-bandwidth n_phi(n+1)+n, because
|I - I'| = |j-l|(n+1) + |q-r| and Mcal vanishes for |j-l| > n_phi.  With the
other natural order, I = q N_phi + j, the half-bandwidth is (n)N_phi + n_phi,
which grows with N_phi and is effectively dense.  Measured at n_phi = 2,
N_phi = 17: half-bandwidth 2/5/8 coefficient-major at n = 0/1/2, versus 2/19/36
time-major.  Since the solve is banded, reordering would silently turn an
O(N nb^2) factorization into an O(N^3) one.  test_bandwidth() pins it.

Zones remain contiguous blocks under this ordering (j is the major index), so
zone z occupies I in [lo*(n+1), hi*(n+1)) and the block structure that makes the
fits on either side of a zone boundary independent is preserved.
"""

import numpy as np


def bandwidth(kv, n):
    """Half-bandwidth of the assembled matrix; see the module docstring."""
    return kv.n_phi * (n + 1) + n


def assemble(Mcal, Vcal, kv, tb, D1, eta):
    """
    Mcal: (npair, M, ntime, N_phi, n_phi+1), Vcal: (n+1, M, ntime, N_phi).

    Returns (A, U) with A of shape (M, ntime, N_phi*(n+1), nb+1) in the banded
    layout of reduce.py -- A[..., I, B] = Gcal[I, I+B] -- and U of shape
    (M, ntime, N_phi*(n+1)).
    """
    n, N, np_ = tb.n, kv.N_phi, kv.n_phi
    dt = Mcal.dtype
    nb = bandwidth(kv, n)
    M_ax, ntime = Mcal.shape[1], Mcal.shape[2]

    A = np.zeros((M_ax, ntime, N*(n+1), nb+1), dtype=dt)
    U = np.zeros((M_ax, ntime, N*(n+1)), dtype=dt)

    # Look up which packed pair index holds (q,r); Mcal only stores q <= r, and
    # the matrix is symmetric in (q,r) because G is symmetric in (j,l).
    pidx = {}
    for p in range(tb.npair):
        q, r = int(tb.pair_q[p]), int(tb.pair_r[p])
        pidx[(q, r)] = p
        pidx[(r, q)] = p

    Tm = tb.T.astype(dt)
    eta_dt = np.dtype(dt).type(eta)

    # D_1 is banded to half-bandwidth 1 regardless of n_phi, so the band loop has
    # to reach b = 1 even when the data block does not (n_phi = 0).
    for b in range(max(np_, 1) + 1):
        if N - b <= 0:
            continue
        rows = np.arange(N - b)
        for q in range(n + 1):
            for r in range(n + 1):
                if b == 0 and r < q:
                    continue                     # held by the (r,q) entry instead
                B = b*(n+1) + (r - q)
                acc = None
                if b <= np_:
                    acc = Mcal[pidx[(q, r)]][:, :, rows, b]
                if b <= 1:
                    reg = eta_dt * D1[rows, b].astype(dt) * Tm[q, r]
                    acc = reg if acc is None else acc + reg
                if acc is not None:
                    A[:, :, rows*(n+1) + q, B] += acc

    for q in range(n + 1):
        U[:, :, np.arange(N)*(n+1) + q] = Vcal[q]

    return A, U


def commit(alpha, tb, kv):
    """
    (M, ntime, N_phi*(n+1)) coefficients -> (M, ntime, N_phi), the spline
    coefficients of the baseline at the window centre.

    This is the contraction sum_q alpha_jq p_q(0), NOT alpha_j0; see timebasis.py.
    """
    n, N = tb.n, kv.N_phi
    a = alpha.reshape(alpha.shape[:-1] + (N, n+1))
    return a @ tb.eval0.astype(alpha.dtype)
