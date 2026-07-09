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
    downsampled samples and scaled by 2^ipri; for downsampled trees (ipri>0) it includes the dropped
    lower-half delay offset 2^r (the "upper-half" logic).  See plans/check_avar_mc.md.
    """
    ipri = int(tree.primary_tree_index)
    Wmax = int(tree.pf.max_width)
    Dtime = int(tree.pf.time_downsampling)
    ndm_out = int(tree.ndm_out)
    dm = np.arange(ndm_out, dtype=np.int64)
    offset_ds = (1 << r) if ipri > 0 else 0                              # dropped lower-half (downsampled)
    settle_ds = offset_ds + (dm + 1) * (1 << R) + 4 * Wmax + 4 * Dtime  # downsampled samples
    settle_input = (1 << ipri) * settle_ds                              # input samples
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

    # Per-tree: analytic variance aligned to (ndm_out, M, P); per-coarse-DM settling table; MC
    # accumulators (sum and sum-of-squares of out_var, and per-coarse-DM steady-chunk count).
    # (PfAvarExact asserts tree_variance > 0, so no positive-prediction mask is needed.)
    analytic, settle, mc_sum, mc_sumsq, mc_count = [], [], [], [], []
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
        mc_sumsq.append(np.zeros_like(a))
        mc_count.append(np.zeros(ndm_out, dtype=np.int64))

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
                    mc_sumsq[itree][steady] += ov[steady] ** 2
                    mc_count[itree][steady] += 1
            if ichunk % report_every == 0:
                _report(ichunk, exact, analytic, mc_sum, mc_sumsq, mc_count)
            ichunk += 1
    except KeyboardInterrupt:
        print("\ncheck_avar_mc: interrupted.", flush=True)

    if ichunk > 0:
        print("check_avar_mc: final summary:", flush=True)
        _report(ichunk - 1, exact, analytic, mc_sum, mc_sumsq, mc_count)


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
        s1 = mc_sum[itree][ready]                              # sum of out_var over steady chunks
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
            var_ov = (s2e - s1e ** 2 / ne) / (ne - 1.0)        # unbiased Var(out_var) over chunks
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
        print(hdr, flush=True)
    else:
        print(f"[chunk {ichunk}] overall: no steady-state channels yet", flush=True)
    for line in lines:
        print(line, flush=True)
