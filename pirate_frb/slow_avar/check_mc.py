"""Monte-Carlo check of analytic peak-finding variances against a ReferenceDedisperser.

Feeds Gaussian noise (per-channel variance = freq_variances) through a ReferenceDedisperser run with
enable_variances=True, and compares its per-chunk variance estimates (out_var) to the analytic
PfAvarExact.tree_variance.  A channel (tree, coarse-DM, multiplet m, profile p) is only compared
once the *entire* chunk has reached statistical steady state for that channel, which we compute
analytically from the dispersion delay (see _settle_chunks / plans/check_avar_mc.md).  Runs until
KeyboardInterrupt (or max_chunks), printing summary statistics of epsilon = mc/analytic - 1 after
each chunk, over all channels that have at least one steady-state estimate so far.
"""

import numpy as np

from .PfVariance import PfAvarExact


def _settle_chunks(tree, r, R, nt_in):
    """Length-ndm_out int array: first chunk index at which each coarse-DM bin is fully steady.

    settle = max dispersion delay (from dm_coarse) + 4*Wmax + 4*time_downsampling, in input time
    samples, then converted to chunks (ceil, +1 safety).  The delay is computed in the tree's
    downsampled samples and scaled by 2^ids; for downsampled trees (ids>0) it includes the dropped
    lower-half delay offset 2^r (the "upper-half" logic).  See plans/check_avar_mc.md.
    """
    ids = int(tree.ds_level)
    Wmax = int(tree.pf.max_width)
    Dtime = int(tree.pf.time_downsampling)
    ndm_out = int(tree.ndm_out)
    dm = np.arange(ndm_out, dtype=np.int64)
    offset_ds = (1 << r) if ids > 0 else 0                              # dropped lower-half (downsampled)
    settle_ds = offset_ds + (dm + 1) * (1 << R) + 4 * Wmax + 4 * Dtime  # downsampled samples
    settle_input = (1 << ids) * settle_ds                              # input samples
    return (settle_input + nt_in - 1) // nt_in + 1                     # chunks (ceil + 1 safety)


def check_avar_mc(plan, sophistication=1, freq_variances=None, max_chunks=None, report_every=1):
    from ..pirate_pybind11 import ReferenceDedisperser   # lazy: keep slow_avar import pybind-light

    nfreq, nt_in, ntrees = int(plan.nfreq), int(plan.nt_in), int(plan.ntrees)

    if freq_variances is None:
        freq_variances = np.ones(nfreq)
    freq_variances = np.asarray(freq_variances, dtype=np.float64)
    assert freq_variances.shape == (nfreq,), (freq_variances.shape, nfreq)

    print("check_avar_mc: building PfAvarExact (analytic variances) ...", flush=True)
    exact = PfAvarExact(plan, freq_variances, progress=True)

    print(f"check_avar_mc: building ReferenceDedisperser(sophistication={sophistication}, "
          "enable_variances=True) ...", flush=True)
    rdd = ReferenceDedisperser(plan, sophistication, enable_variances=True)
    assert int(rdd.beams_per_batch) == 1 and int(rdd.nbatches) == 1, "check_avar_mc requires nbeams==1"

    # Per-tree: analytic variance aligned to (ndm_out, M, P); per-coarse-DM settling table;
    # MC accumulators (sum of out_var and per-coarse-DM chunk count); positive-analytic mask.
    analytic, settle, mc_sum, mc_count, pos = [], [], [], [], []
    for itree in range(ntrees):
        tree = plan.trees[itree]
        r, R = int(exact.tree_r[itree]), int(exact.tree_R[itree])
        ndm_out = int(tree.ndm_out)
        if ndm_out != (1 << (r - R)):
            raise RuntimeError(f"check_avar_mc: tree {itree} has ndm_out={ndm_out} != 2^(r-R)="
                               f"{1 << (r - R)} (dm_downsampling != 2^pf_rank is unsupported)")
        a = np.ascontiguousarray(exact.tree_variance[itree].transpose(1, 0, 2))   # (ndm_out, M, P)
        analytic.append(a)
        settle.append(_settle_chunks(tree, r, R, nt_in))
        mc_sum.append(np.zeros_like(a))
        mc_count.append(np.zeros(ndm_out, dtype=np.int64))
        pos.append(a > 0.0)

    # Peak-finding weights = 1, so out_var is directly comparable to PfAvarExact (which is unweighted).
    for w in rdd.wt_arrays:
        w[...] = 1.0

    sigma = np.sqrt(freq_variances).astype(np.float32)[None, :, None]   # (1, nfreq, 1)
    in_shape = tuple(int(s) for s in rdd.input_array.shape)
    rng = np.random.default_rng()

    print(f"check_avar_mc: nfreq={nfreq} nt_in={nt_in} ntrees={ntrees}; running "
          f"{'until Ctrl-C' if max_chunks is None else f'for {max_chunks} chunks'} ...\n", flush=True)

    ichunk = 0
    try:
        while max_chunks is None or ichunk < max_chunks:
            rdd.input_array[...] = sigma * rng.standard_normal(in_shape, dtype=np.float32)
            rdd.dedisperse(ichunk, 0)
            for itree in range(ntrees):
                steady = ichunk >= settle[itree]            # (ndm_out,) bool
                if steady.any():
                    ov = np.asarray(rdd.out_var[itree])[0]  # (ndm_out, M, P)
                    mc_sum[itree][steady] += ov[steady]
                    mc_count[itree][steady] += 1
            if ichunk % report_every == 0:
                _report(ichunk, exact, analytic, mc_sum, mc_count, pos)
            ichunk += 1
    except KeyboardInterrupt:
        print("\ncheck_avar_mc: interrupted.", flush=True)

    if ichunk > 0:
        print("check_avar_mc: final summary:", flush=True)
        _report(ichunk - 1, exact, analytic, mc_sum, mc_count, pos)


def _spread(eps):
    """Delta(eps) = sqrt(<eps^2> - <eps>^2), guarded against tiny negative roundoff."""
    return float(np.sqrt(max(0.0, float(np.mean(eps ** 2)) - float(np.mean(eps)) ** 2)))


def _report(ichunk, exact, analytic, mc_sum, mc_count, pos):
    all_eps, lines = [], []
    for itree in range(len(analytic)):
        cnt = mc_count[itree]
        ready = cnt > 0
        ndm_out = cnt.shape[0]
        r, R, M = int(exact.tree_r[itree]), int(exact.tree_R[itree]), int(exact.tree_fs[itree].M)
        if not ready.any():
            lines.append(f"  tree {itree} [r={r} R={R} M={M}]: no steady-state channels yet")
            continue
        est = mc_sum[itree][ready] / cnt[ready][:, None, None]   # (nready, M, P)
        pm = pos[itree][ready]
        eps = est[pm] / analytic[itree][ready][pm] - 1.0
        all_eps.append(eps)
        lines.append(f"  tree {itree} [r={r} R={R} M={M}]: dm {int(ready.sum())}/{ndm_out} steady, "
                     f"{eps.size} chans, mean(eps)={float(np.mean(eps)):+.4g}, "
                     f"Delta(eps)={_spread(eps):.4g}, count {int(cnt[ready].min())}..{int(cnt[ready].max())}")

    if all_eps:
        e = np.concatenate(all_eps)
        print(f"[chunk {ichunk}] overall: {e.size} chans, mean(eps)={float(np.mean(e)):+.4g}, "
              f"Delta(eps)={_spread(e):.4g}", flush=True)
    else:
        print(f"[chunk {ichunk}] overall: no steady-state channels yet", flush=True)
    for line in lines:
        print(line, flush=True)
