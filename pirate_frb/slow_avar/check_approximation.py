"""Compare the exact vs approximate analytic peak-finding variance for a DedispersionConfig."""

import numpy as np

from .PfVariance import PfAvarExact, PfAvarApproximation


def check_approximation(config):
    """Print summary statistics comparing the exact vs approximate peak-finding variance.

    Builds PfAvarExact and PfAvarApproximation (ds_level=0) for 'config'. For each multiplet m,
    compares var_exact = exact.per_m[m] (rank r-R) to var_approx = approx.get_per_m(m) (rank
    r-L): the approximate variance is lifted to rank r-R by adding L-R spectator low DM bits,
    both are unpacked to a common 'dbits', and the fractional error
    epsilon = var_approx/var_exact - 1 is formed (shape (2^len(dbits), P)). Prints the mean of
    epsilon and its spread Delta(eps) = sqrt(<eps^2> - <eps>^2), averaged over multiplets.
    """
    exact = PfAvarExact(config)
    approx = PfAvarApproximation(config)
    M = exact.fs.M
    assert approx.fs.M == M, (approx.fs.M, M)
    lift = exact.rho - approx.rho        # (r-R) - (r-L) = L-R spectator low bits to add
    assert lift == approx.L - approx.R, (lift, approx.L, approx.R)

    # Note: eps_mean / eps2_mean accumulate the per-multiplet means, then divide by M (so they
    # are <eps> and <eps^2>). Dividing is required for Delta = sqrt(<eps^2> - <eps>^2) to be a
    # well-defined (>= 0) spread; without it the formula is not a variance.
    eps_mean = 0.0
    eps2_mean = 0.0
    for m in range(M):
        var_exact = exact.per_m[m]
        var_approx = approx.get_per_m(m)
        if not var_exact.terms or not var_approx.terms:
            raise RuntimeError(f"check_approximation: empty PfVariance at m={m} "
                               f"(exact={len(var_exact.terms)}, approx={len(var_approx.terms)} terms)")
        assert var_exact.rank == exact.rho, (m, var_exact.rank, exact.rho)
        assert var_approx.rank == approx.rho, (m, var_approx.rank, approx.rho)

        var_approx = var_approx.add_spectator_low_bits(lift)         # rank r-L -> r-R
        dbits = var_exact.get_all_dbits() | var_approx.get_all_dbits()
        a_exact = var_exact.unpack(dbits)                            # (2^len(dbits), P)
        a_approx = var_approx.unpack(dbits)
        if (a_exact <= 0.0).any() or (a_approx <= 0.0).any():
            raise RuntimeError(f"check_approximation: non-positive variance at m={m}")

        epsilon = (a_approx / a_exact) - 1.0
        eps_mean += float(np.mean(epsilon))
        eps2_mean += float(np.mean(epsilon ** 2))

    eps_mean /= M
    eps2_mean /= M
    delta = float(np.sqrt(eps2_mean - eps_mean ** 2))
    print(f"PfAvar exact-vs-approx [r={exact.r}, R={exact.R}, L={approx.L}, M={M}]:")
    print(f"  mean(epsilon)  = {eps_mean:+.6g}      (epsilon = var_approx/var_exact - 1)")
    print(f"  Delta(epsilon) = {delta:.6g}")
