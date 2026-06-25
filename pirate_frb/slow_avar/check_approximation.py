"""Compare the exact vs approximate analytic peak-finding variance for a DedispersionPlan."""

import numpy as np

from .PfVariance import PfAvarExact, PfAvarApproximation


def check_approximation(plan):
    """Print summary statistics comparing the exact vs approximate peak-finding variance, per tree.

    Builds PfAvarExact(plan) and PfAvarApproximation(plan) and compares them for every tree. For
    each tree itree and multiplet m, compares var_exact = exact.per_tm[itree][m] (rank r-R) to
    var_approx = approx.get_per_m(itree, m) (rank r-L): the approximate variance is lifted to rank
    r-R by adding L-R spectator low DM bits, both are unpacked to a common 'dbits', and the
    fractional error epsilon = var_approx/var_exact - 1 is formed (shape (2^len(dbits), P)). Prints,
    per tree, the mean of epsilon and its spread Delta(eps) = sqrt(<eps^2> - <eps>^2) over multiplets.
    """
    exact = PfAvarExact(plan, progress=True)
    approx = PfAvarApproximation(plan, progress=True)
    print("PfAvar exact-vs-approx (epsilon = var_approx/var_exact - 1):")
    for itree in range(plan.ntrees):
        _check_one_tree(exact, approx, itree)


def _check_one_tree(exact, approx, itree):
    r, R, L = int(exact.r[itree]), int(exact.R[itree]), int(approx.L[itree])
    assert (r, R) == (int(approx.r[itree]), int(approx.R[itree])), \
        (itree, r, R, int(approx.r[itree]), int(approx.R[itree]))
    rho = r - R                                  # exact PfVariance rank
    approx_rho = r - L                           # approximation PfVariance rank (r-L)
    lift = rho - approx_rho                      # (r-R) - (r-L) = L-R spectator low bits to add
    assert lift == L - R, (lift, L, R)
    M = exact.fs[itree].M
    assert approx.fs[itree].M == M, (itree, approx.fs[itree].M, M)

    # eps_mean / eps2_mean accumulate the per-multiplet means, then divide by M (so they are <eps>
    # and <eps^2>); dividing is required for Delta = sqrt(<eps^2> - <eps>^2) to be a valid spread.
    eps_mean = 0.0
    eps2_mean = 0.0
    for m in range(M):
        var_exact = exact.per_tm[itree][m]
        var_approx = approx.get_per_m(itree, m)
        if not var_exact.terms or not var_approx.terms:
            raise RuntimeError(f"check_approximation: empty PfVariance at tree={itree}, m={m} "
                               f"(exact={len(var_exact.terms)}, approx={len(var_approx.terms)} terms)")
        assert var_exact.rank == rho, (itree, m, var_exact.rank, rho)
        assert var_approx.rank == approx_rho, (itree, m, var_approx.rank, approx_rho)

        var_approx = var_approx.add_spectator_low_bits(lift)         # rank r-L -> r-R
        dbits = var_exact.get_all_dbits() | var_approx.get_all_dbits()
        a_exact = var_exact.unpack(dbits)                            # (2^len(dbits), P)
        a_approx = var_approx.unpack(dbits)
        if (a_exact <= 0.0).any() or (a_approx <= 0.0).any():
            raise RuntimeError(f"check_approximation: non-positive variance at tree={itree}, m={m}")

        epsilon = (a_approx / a_exact) - 1.0
        eps_mean += float(np.mean(epsilon))
        eps2_mean += float(np.mean(epsilon ** 2))

    eps_mean /= M
    eps2_mean /= M
    delta = float(np.sqrt(eps2_mean - eps_mean ** 2))
    print(f"  tree {itree} [r={r}, R={R}, L={L}, M={M}]: "
          f"mean(eps)={eps_mean:+.6g}, Delta(eps)={delta:.6g}")
