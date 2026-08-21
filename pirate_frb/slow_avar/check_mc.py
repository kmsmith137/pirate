"""Monte-Carlo check of analytic peak-finding variances against a ReferenceDedisperser.

Feeds Gaussian noise (per-channel variance = freq_variances) through a ReferenceDedisperser, runs a
ReferencePfSquare over each tree's subband array (out_sb) to get a per-chunk variance estimate, and
compares that to the analytic PfAvarExact.tree_variance.  A channel (tree, coarse-DM, multiplet m,
profile p) is only compared once the *entire* chunk has reached statistical steady state for that
channel, which we obtain by coarsening the grouper's per-element steady-state boundary
(DedispersionTree.compute_steady_state_it0) to whole chunks.  Runs until KeyboardInterrupt (or
max_chunks), printing summary statistics of epsilon = mc/analytic - 1 after each chunk, over all
channels that have at least one steady-state estimate so far.

The estimate averages over EVERY time sample of the chunk, whereas the peak-finder proper evaluates
its convolutions only on a sublattice of spacing min(Dcore, 2^lambda).  Both estimate the same
per-element variance -- all output elements are equal by translation invariance -- so the central
values are the same either way; what differs is the estimator's own variance, i.e. the error bars
reported below, which are tighter than a Dcore > 1 sublattice would give.
"""

import numpy as np

from .PfVariance import PfAvarExact
from ..utils import atomic_print


def check_avar_mc(plan, sophistication=1, freq_variances=None, max_chunks=None, report_every=1):
    # Lazy imports: keep the slow_avar import pybind-light.
    from ..pirate_pybind11 import ReferenceDedisperser
    from ..kernels import ReferencePfSquare

    nfreq, nt_in, ntrees = int(plan.nfreq), int(plan.nt_in), int(plan.ntrees)

    if freq_variances is None:
        freq_variances = np.ones(nfreq)
    freq_variances = np.asarray(freq_variances, dtype=np.float64)
    assert freq_variances.shape == (nfreq,), (freq_variances.shape, nfreq)

    atomic_print("check_avar_mc: building PfAvarExact (analytic variances) ...")
    exact = PfAvarExact(plan, freq_variances, progress=True)

    atomic_print(f"check_avar_mc: building ReferenceDedisperser(sophistication={sophistication}) ...")
    rdd = ReferenceDedisperser(plan, sophistication)
    assert int(rdd.beams_per_batch) == 1 and int(rdd.nbatches) == 1, "check_avar_mc requires nbeams==1"

    # Per-tree: analytic variance aligned to (ndm_out, M, P); the PfSquare that reads out_sb and
    # its accumulator; per-coarse-DM settling table; MC accumulators (sum and sum-of-squares of
    # the per-chunk variance estimate, and per-coarse-DM steady-chunk count).
    # (PfAvarExact asserts tree_variance > 0, so no positive-prediction mask is needed.)
    analytic, pf_squares, mc_acc, settle, mc_sum, mc_sumsq, mc_count = [], [], [], [], [], [], []
    for itree in range(ntrees):
        tree = plan.trees[itree]
        r, R = int(exact.tree_r[itree]), int(exact.tree_R[itree])
        ndm_out = int(tree.ndm_out)
        if ndm_out != (1 << (r - R)):
            raise RuntimeError(f"check_avar_mc: tree {itree} has ndm_out={ndm_out} != 2^(r-R)="
                               f"{1 << (r - R)} (dm_downsampling != 2^pf_rank is unsupported)")
        a = np.ascontiguousarray(exact.tree_variance[itree].transpose(1, 0, 2))   # (ndm_out, M, P)
        analytic.append(a)
        # Every axis except time is a spectator to a PfSquare, so out_sb's (ndm_out, M) pair is
        # flattened into its row count.  Note ndm_out == 2^(r-R) is checked just above, so this
        # is the full coarse-DM axis of out_sb.
        M, P = int(tree.frequency_subbands.M), int(tree.nprofiles)
        pf_squares.append(ReferencePfSquare(int(tree.pf.max_width), 1, 1, ndm_out * M,
                                            int(tree.nt_ds)))
        mc_acc.append(np.zeros((1, ndm_out * M, P)))
        # First fully-steady chunk per coarse-DM bin: coarsen the grouper's per-element
        # steady-state boundary (compute_steady_state_it0, in output-time-bin units) to
        # whole chunks -- ichunk*nt_out >= it0 -- with a +1-chunk safety margin.
        nt_out = int(tree.nt_out)
        it0 = tree.compute_steady_state_it0(plan.config)   # (ndm_out,) int64
        settle.append((it0 + nt_out - 1) // nt_out + 1)
        mc_sum.append(np.zeros_like(a))
        mc_sumsq.append(np.zeros_like(a))
        mc_count.append(np.zeros(ndm_out, dtype=np.int64))

    # Peak-finding weights = 1.  They do not enter the estimate below (out_sb is upstream of them),
    # but it keeps out_max meaningful if anyone looks at it.
    for w in rdd.wt_arrays:
        w[...] = 1.0

    sigma = np.sqrt(freq_variances).astype(np.float32)[None, :, None]   # (1, nfreq, 1)
    in_shape = tuple(int(s) for s in rdd.input_array.shape)
    rng = np.random.default_rng()

    atomic_print(f"check_avar_mc: nfreq={nfreq} nt_in={nt_in} ntrees={ntrees}; running "
                 f"{'until Ctrl-C' if max_chunks is None else f'for {max_chunks} chunks'} ...\n\n")

    ichunk = 0
    try:
        while max_chunks is None or ichunk < max_chunks:
            rdd.input_array[...] = sigma * rng.standard_normal(in_shape, dtype=np.float32)
            rdd.dedisperse(ichunk, 0)
            for itree in range(ntrees):
                # Run the PfSquare on EVERY chunk, even one with no steady-state channels: it
                # carries 'tpad' input samples across chunk boundaries, so skipping a chunk
                # would leave the next one with stale history.
                ov = _chunk_variance(rdd, pf_squares[itree], mc_acc[itree], itree,
                                     analytic[itree].shape)   # (ndm_out, M, P)
                steady = ichunk >= settle[itree]              # (ndm_out,) bool
                if steady.any():
                    mc_sum[itree][steady] += ov[steady]
                    mc_sumsq[itree][steady] += ov[steady] ** 2
                    mc_count[itree][steady] += 1
            if ichunk % report_every == 0:
                _report(ichunk, exact, analytic, mc_sum, mc_sumsq, mc_count)
            ichunk += 1
    except KeyboardInterrupt:
        atomic_print("\ncheck_avar_mc: interrupted.")

    if ichunk > 0:
        atomic_print("check_avar_mc: final summary:")
        _report(ichunk - 1, exact, analytic, mc_sum, mc_sumsq, mc_count)


def _chunk_variance(rdd, pf_square, acc, itree, shape):
    """The per-chunk variance estimate of tree 'itree', shape (ndm_out, M, P).

    The PfSquare accumulates a sum of squares over the chunk's nt_ds time samples; dividing by
    nt_ds turns it into the per-element mean square, which for mean-zero input is the variance.
    'acc' must be zeroed here rather than accumulated across chunks: the caller wants one
    independent estimate per chunk, so that it can report a spread over chunks.
    """

    sb = np.asarray(rdd.out_sb[itree])                     # (1, ndm_out, M, nt_ds)
    nt_ds = sb.shape[3]
    acc[...] = 0.0
    pf_square.apply(acc, sb.reshape(1, -1, nt_ds), 0)
    return acc[0].reshape(shape) / nt_ds


def _spread(eps):
    """Delta(eps) = sqrt(<eps^2> - <eps>^2), guarded against tiny negative roundoff."""
    return float(np.sqrt(max(0.0, float(np.mean(eps ** 2)) - float(np.mean(eps)) ** 2)))


_MIN_COUNT_WORST = 10   # only flag the worst channel among those with >= this many steady chunks


def _report(ichunk, exact, analytic, mc_sum, mc_sumsq, mc_count):
    all_eps, lines = [], []
    worst = None   # (|a_I|, a_I, sigma_I) of the largest-|a_i| channel over all trees (n_i >= MIN)
    for itree in range(len(analytic)):
        cnt = mc_count[itree]
        ready = cnt > 0
        ndm_out = cnt.shape[0]
        r, R, M = int(exact.tree_r[itree]), int(exact.tree_R[itree]), int(exact.tree_fs[itree].M)
        if not ready.any():
            lines.append(f"  tree {itree} [r={r} R={R} M={M}]: no steady-state channels yet")
            continue
        n = cnt[ready].astype(np.float64)[:, None, None]        # (nready, 1, 1)
        a = analytic[itree][ready]                              # (nready, M, P)
        s1 = mc_sum[itree][ready]                              # sum of the estimate over steady chunks
        eps = s1 / n / a - 1.0                                 # a_i: per-channel mean of eps over chunks
        all_eps.append(eps.ravel())
        line = (f"  tree {itree} [r={r} R={R} M={M}]: dm {int(ready.sum())}/{ndm_out} steady, "
                f"{eps.size} chans, mean(eps)={float(np.mean(eps)):+.4g}, "
                f"Delta(eps)={_spread(eps.ravel()):.4g}, count {int(cnt[ready].min())}..{int(cnt[ready].max())}")

        # Worst (largest |a_i|) channel among well-sampled coarse-DMs (n_i >= _MIN_COUNT_WORST).
        # sigma_i = a_i / SE(a_i), with SE(a_i) = sqrt(v_i / n_i), v_i = Var(eps_is) over chunks.
        elig = cnt[ready] >= _MIN_COUNT_WORST
        if elig.any():
            ne = n[elig]                                        # (nel, 1, 1)
            ae, s1e, s2e = a[elig], s1[elig], mc_sumsq[itree][ready][elig]
            ai = s1e / ne / ae - 1.0
            var_ov = (s2e - s1e ** 2 / ne) / (ne - 1.0)        # unbiased Var(estimate) over chunks
            with np.errstate(divide="ignore", invalid="ignore"):
                sigma = (s1e / ne - ae) / np.sqrt(var_ov / ne)  # = a_i / sqrt(v_i / n_i)
            k = int(np.argmax(np.abs(ai)))
            absx, ax, sx = float(np.abs(ai).flat[k]), float(ai.flat[k]), float(sigma.flat[k])
            line += f", worst(eps)={ax:+.4g} ({sx:+.1f} sigma)"
            if (worst is None) or (absx > worst[0]):
                worst = (absx, ax, sx)
        lines.append(line)

    if all_eps:
        e = np.concatenate(all_eps)
        hdr = (f"[chunk {ichunk}] overall: {e.size} chans, mean(eps)={float(np.mean(e)):+.4g}, "
               f"Delta(eps)={_spread(e):.4g}")
        hdr += (f", worst(eps)={worst[1]:+.4g} ({worst[2]:+.1f} sigma)" if worst is not None
                else f", worst(eps)=n/a (need count>={_MIN_COUNT_WORST})")
        atomic_print(hdr)
    else:
        atomic_print(f"[chunk {ichunk}] overall: no steady-state channels yet")
    for line in lines:
        atomic_print(line)
