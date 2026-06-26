"""Compare the exact vs approximate analytic peak-finding variance for a DedispersionPlan."""

import numpy as np

from .PfVariance import PfAvarExact, PfAvarApproximation


def check_approximation(plan, freq_variances=None):
    """Print, per tree, summary statistics of the exact-vs-approximate peak-finding variance.

    The comparison is based entirely on the tree_variance[] arrays.  For each tree and multiplet m,
    a_exact = exact.tree_variance[itree][m] (the exact per-multiplet variance, rank r-R) is compared
    to a_approx = approx.tree_variance[itree][n] (the approximate per-subband variance, rank r-L,
    where n is the subband of m) replicated over the L-R spectator low DM bits up to rank r-R.
    Forms epsilon = a_approx/a_exact - 1 and prints its mean and spread
    Delta(eps) = sqrt(<eps^2> - <eps>^2) over multiplets.

    'freq_variances' is a length-nfreq array of per-channel input variances (the weights used
    when summing per-frequency variances over frequency). Defaults to all-ones (equal weighting),
    which makes the exact-vs-approx comparison independent of the weighting.
    """
    if freq_variances is None:
        freq_variances = np.ones(int(plan.nfreq))
    exact = PfAvarExact(plan, freq_variances, progress=True)
    approx = PfAvarApproximation(plan, freq_variances, progress=True)
    print("PfAvar exact-vs-approx (epsilon = var_approx/var_exact - 1):")
    for itree in range(plan.ntrees):
        _check_one_tree(exact, approx, itree)


def _check_one_tree(exact, approx, itree):
    r, R, L = int(exact.tree_r[itree]), int(exact.tree_R[itree]), int(approx.tree_L[itree])
    assert (r, R) == (int(approx.tree_r[itree]), int(approx.tree_R[itree])), \
        (itree, r, R, int(approx.tree_r[itree]), int(approx.tree_R[itree]))
    lift = L - R                                 # L-R spectator low DM bits: rank (r-L) -> (r-R)
    fs = exact.tree_fs[itree]
    M, N, P = fs.M, fs.N, int(exact.tree_P[itree])
    assert approx.tree_fs[itree].M == M, (itree, approx.tree_fs[itree].M, M)

    var_exact = exact.tree_variance[itree]       # (M, 2^(r-R), P): exact per-multiplet variance
    var_approx = approx.tree_variance[itree]     # (N, 2^(r-L), P): approx per-subband variance
    assert var_exact.shape == (M, 1 << (r - R), P), (itree, var_exact.shape)
    assert var_approx.shape == (N, 1 << (r - L), P), (itree, var_approx.shape)

    # eps_mean / eps2_mean accumulate the per-multiplet means, then divide by M (so they are <eps>
    # and <eps^2>); dividing is required for Delta = sqrt(<eps^2> - <eps>^2) to be a valid spread.
    eps_mean = 0.0
    eps2_mean = 0.0
    for m in range(M):
        a_exact = var_exact[m]                                       # (2^(r-R), P)
        # The approximation depends only on the subband n=m_to_n[m] and the high (r-L) DM bits;
        # replicate it over the L-R spectator low bits to reach the exact's (r-R) DM axis.
        a_approx = np.repeat(var_approx[fs.m_to_n[m]], 1 << lift, axis=0)   # (2^(r-R), P)
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
