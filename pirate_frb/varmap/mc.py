"""Monte-Carlo check of a stored VarianceMap against simulated noise.

Feeds Gaussian noise (per-channel variance = freq_variances) through the dedispersion pipeline,
runs a PfSquare over each tree's subband array to get one per-chunk variance estimate per
output channel, and compares that to the map's own prediction, ``vmm.apply_fine(v)``. Runs
until Ctrl-C (or 'nchunks'), printing summary statistics of

    eps = MC / map - 1

after every chunk, over all channels that have at least one steady-state estimate so far.

SIGN CONVENTION, which is the point of the tool and is easy to get backwards:

    eps > 0  <=>  MC exceeds the map  <=>  the map UNDERESTIMATES  <=>  BAD
    eps < 0  <=>  the map exceeds MC  <=>  the map OVERESTIMATES   <=>  fine (conservative)

So 'eps_max' is the worst UNDERestimate and 'eps_min' the worst OVERestimate. An admissible
map (is_admissible=True, e.g. from 'varmap df' or the LP repair) should never underestimate
with statistical significance, i.e. should have eps_max not significantly positive. That
statement is the thing this tool exists to make checkable.

WHAT THE NUMBERS ARE A STATEMENT ABOUT. eps is computed for ONE input-variance vector v; a map
is admissible for all v >= 0, and this checks one of them. The header says which.

BEWARE THE CHANNEL COUNT WHEN READING mean(eps). The channels are NOT independent -- every one
of them sees the same noise realization each chunk -- so mean(eps) is far less precise than
its channel count suggests. Measured on toy.yml at 1000 chunks and 258560 channels, mean(eps)
scattered over {-0.0013, -0.0022, +0.0007} across three runs, i.e. about +-0.0015, where naive
independence would have predicted ~1e-5. A drift of that size is not evidence of anything.

TWO GRANULARITIES THAT MUST NOT BE CONFUSED, both per tree:

  - 'nt_ds' is the subband array's time axis, and divides the PfSquare accumulator to turn a
    sum of squares into a mean square.
  - 'nt_out' is the peak-finder output's time axis, and converts compute_steady_state_it0()'s
    per-element boundary into whole chunks.

Comparison is at FINE (alpha) granularity even for a coarse-grained map: apply_fine() lifts
every stored row to its group's value, which is exactly what admissibility asserts, and it
makes fine, coarse and LP-repaired maps all one code path.
"""

import time

import numpy as np

from ..utils import atomic_print


# Only flag the worst channel among those with at least this many steady-state chunks: a
# just-settled row has a wild mean and a meaningless sigma, and would otherwise win every time.
_MIN_COUNT_WORST = 10


def run_mc(vmm, freq_variances, *, device='gpu', nchunks=None, report_every=1,
           sophistication=1):
    """Run the Monte Carlo against 'vmm' and print summaries. Returns the final stats dict.

    'vmm' is a VarianceMultiMap whose config has ALREADY been overridden by the caller (see
    varmap_mc() in pirate_frb/__main__.py): beams forced to 1, max_gpu_clag raised, dtype
    float32. This function does not touch the config, so that the overrides and their printout
    live in one place.

    A CUDA device must already be selected, on BOTH devices: DedispersionPlan allocates
    through cudaHostAlloc.
    """

    from .brute_force import _SweepGeometry

    config, detrender = vmm.config, vmm.detrender
    v = np.asarray(freq_variances, dtype=np.float64)

    geom = _SweepGeometry(config, detrender)
    ntrees = geom.ntrees
    if geom.nbeams != 1:
        raise RuntimeError(f'varmap mc: expected beams_per_batch == 1, got {geom.nbeams}')

    # The map's own prediction, per ITREE and at FINE granularity. One call, up front: it is
    # the thing every chunk is compared against, and it does not depend on the chunk.
    y = [np.asarray(a, dtype=np.float64).reshape(-1) for a in vmm.apply_fine(v)]
    if len(y) != ntrees:
        raise RuntimeError(f'varmap mc: apply_fine() returned {len(y)} trees, plan has {ntrees}')

    # Per-tree accumulators, all indexed by the FINE alpha = (d, m, p) with d over
    # Dpf = 2^(r-R) -- which is exactly the (Dpf, M, P) layout of the subband array. See
    # Dedisperser.hpp's out_sb comment: that layout does NOT depend on xdm_rank, which is why
    # this works for K > 0 and why nothing here resolves the peak-finder's m_ext convention.
    settle, mc_sum, mc_sumsq, mc_count, shapes = [], [], [], [], []
    for itree in range(ntrees):
        tree = geom.plan.trees[itree]
        D, M, P = geom.tree_D[itree], geom.tree_M[itree], geom.tree_P[itree]
        K = int(tree.xdm_rank())
        nt_out = int(tree.nt_out)

        if y[itree].size != D * M * P:
            raise RuntimeError(f'varmap mc: tree {itree}: map gives {y[itree].size} channels,'
                               f' geometry wants {D}*{M}*{P} = {D*M*P}')

        # compute_steady_state_it0() is indexed by the PEAK-FINDER's coarse DM (length
        # ndm_out), but the subband array has Dpf = ndm_out << K rows. The peak-finder reads
        # sb_out at row ((d << K) | mu) -- see PeakFindingKernel.cu, ReferencePeakFindingKernel
        # ("input DM row ((d << K) | mu) ... land at tmp_arr index em = mu*M + m") -- so full
        # row 'df' belongs to peak-finder row (df >> K), and np.repeat is the broadcast.
        #
        # GETTING THIS BACKWARDS IS SILENT AND LOOKS LIKE A PASS: warmup samples would be
        # admitted on the wrong rows, biasing variances DOWNWARD, which reads as the map
        # overestimating.
        it0 = np.asarray(geom.plan.trees[itree].compute_steady_state_it0(config), dtype=np.int64)
        if it0.size * (1 << K) != D:
            raise RuntimeError(f'varmap mc: tree {itree}: it0 has {it0.size} rows, '
                               f'K={K}, but Dpf={D}')
        it0_full = np.repeat(it0, 1 << K)

        # Coarsen the per-element boundary to WHOLE chunks -- the estimate averages over the
        # whole chunk, so a partially-warm chunk is unusable -- with a +1 safety margin.
        settle.append((it0_full + nt_out - 1) // nt_out + 1)
        mc_sum.append(np.zeros(D * M * P))
        mc_sumsq.append(np.zeros(D * M * P))
        mc_count.append(np.zeros(D, dtype=np.int64))
        shapes.append((D, M, P, K))

    runner = (_CpuRunner(geom, v, sophistication) if (device == 'cpu')
              else _GpuRunner(geom, v))

    atomic_print(f'varmap mc: {ntrees} tree(s), {sum(len(a) for a in y)} channels total,'
                 f' device={device}; running'
                 f" {'until Ctrl-C' if nchunks is None else f'for {nchunks} chunks'} ...\n")

    ichunk = 0
    try:
        while (nchunks is None) or (ichunk < nchunks):
            # EVERY chunk, including ones with no steady rows: the MegaRingbuf and the
            # PfSquare's persistent_state carry history, so skipping one leaves the next with
            # stale state. Only the accumulation below is gated.
            mc = runner.run_chunk(ichunk)

            for itree in range(ntrees):
                D, M, P, _ = shapes[itree]
                steady = ichunk >= settle[itree]                  # (D,) bool
                if not steady.any():
                    continue
                sel = np.repeat(steady, M * P)                    # (D*M*P,) bool
                mc_sum[itree][sel] += mc[itree][sel]
                mc_sumsq[itree][sel] += mc[itree][sel] ** 2
                mc_count[itree][steady] += 1

            if (ichunk % report_every) == 0:
                _report(ichunk, y, mc_sum, mc_sumsq, mc_count, shapes, geom)
            ichunk += 1
    except KeyboardInterrupt:
        atomic_print('\nvarmap mc: interrupted.')

    if ichunk > 0:
        atomic_print('varmap mc: final summary:')
        return _report(ichunk - 1, y, mc_sum, mc_sumsq, mc_count, shapes, geom)
    return {}


####################################   the two runners   ####################################


class _GpuRunner:
    """Gaussian noise -> _GpuPipeline -> one (D*M*P,) variance estimate per tree per chunk."""

    def __init__(self, geom, freq_variances):
        import cupy as cp
        from .brute_force import _GpuPipeline

        self.geom = geom
        self.pipe = _GpuPipeline(geom)
        self.pipe.allocate()
        # (1, nfreq, 1), broadcast over time when the noise is drawn.
        self.sigma = cp.asarray(np.sqrt(freq_variances).astype(np.float32)[None, :, None])
        self.rng = cp.random.default_rng()

    def run_chunk(self, ichunk):
        import cupy as cp

        g, pipe = self.geom, self.pipe

        if pipe.gpu_detrender is None:
            pipe.stream_in[...] = self.sigma * self.rng.standard_normal(
                pipe.stream_in.shape, dtype=cp.float32)
        else:
            # Fill the WHOLE detrender buffer, padding included: the detrender reads
            # (B, nfreq, nt_in + 2W) and emits [W, W+nt_in), and the padding stands in for the
            # neighbouring chunks' samples. Zero padding (which is what the sweep uses, because
            # a one-hot's response is localized) would make the edges of every chunk see an
            # unrealistic context and bias the emitted variance.
            pipe.det_data[...] = self.sigma * self.rng.standard_normal(
                pipe.det_data.shape, dtype=cp.float32)
            pipe.detrend_into_stream()

        pipe.run_chunk(ichunk)

        # Zero AFTER reading, not before running: 'acc' is accumulate-don't-overwrite, so each
        # chunk must start from zero to be an independent estimate.
        out = []
        cp.cuda.get_current_stream().synchronize()
        for itree in range(g.ntrees):
            a = cp.asnumpy(pipe.acc[itree]).reshape(-1)   # (1, D*M, P) -> (D*M*P,)
            pipe.acc[itree].fill(0.0)
            out.append(a / g.tree_nt_ds[itree])
        return out


class _CpuRunner:
    """The same, through ReferenceDedisperser + ReferencePfSquare. Orders of magnitude slower;
    the path for a config the GPU pipeline refuses (stage-2 dd_rank < 3, or a missing sbdd
    kernel)."""

    def __init__(self, geom, freq_variances, sophistication):
        from ..pirate_pybind11 import ReferenceDedisperser
        from ..kernels import ReferencePfSquare

        self.geom = geom
        self.rdd = ReferenceDedisperser(geom.plan, sophistication)
        self.rng = np.random.default_rng()
        self.sigma = np.sqrt(freq_variances).astype(np.float32)[None, :, None]
        self.in_shape = tuple(int(s) for s in self.rdd.input_array.shape)

        self.pf_squares, self.acc = [], []
        for itree in range(geom.ntrees):
            D, M, P = geom.tree_D[itree], geom.tree_M[itree], geom.tree_P[itree]
            # Every axis except time is a spectator to a PfSquare, so (Dpf, M) flattens into
            # the row count.
            self.pf_squares.append(
                ReferencePfSquare(int(geom.plan.trees[itree].pf.max_width), 1, 1,
                                  D * M, geom.tree_nt_ds[itree]))
            self.acc.append(np.zeros((1, D * M, P)))

        # Peak-finding weights are irrelevant here (out_sb is upstream of them), but setting
        # them keeps out_max meaningful if anyone looks at it.
        for w in self.rdd.wt_arrays:
            w[...] = 1.0

    def run_chunk(self, ichunk):
        g = self.geom

        x = self.sigma * self.rng.standard_normal(self.in_shape, dtype=np.float32)
        if g.spline_detrender is not None:
            x = self._detrend(x)
        self.rdd.input_array[...] = x
        self.rdd.dedisperse(ichunk, 0)

        out = []
        for itree in range(g.ntrees):
            sb = np.asarray(self.rdd.out_sb[itree])        # (1, Dpf, M, nt_ds)
            nt_ds = sb.shape[3]
            self.acc[itree][...] = 0.0
            self.pf_squares[itree].apply(self.acc[itree], sb.reshape(1, -1, nt_ds), 0)
            out.append(self.acc[itree].reshape(-1) / nt_ds)
        return out

    def _detrend(self, x):
        """Detrend one chunk with the numpy spline detrender, padding included; see the GPU
        runner's comment on why the padding is noise and not zeros."""

        g = self.geom
        W, nfreq, nt_in = g.W, g.nfreq, g.nt_in
        buf = np.zeros((1, nfreq, nt_in + 2*W), dtype=g.spline_detrender.dtype)
        buf[0, :, W:W+nt_in] = x[0]
        buf[0, :, :W] = self.sigma[0] * self.rng.standard_normal((nfreq, W), dtype=np.float32)
        buf[0, :, W+nt_in:] = self.sigma[0] * self.rng.standard_normal((nfreq, W),
                                                                       dtype=np.float32)
        mask = np.ones(buf.shape, dtype=bool)
        residual, mask_out, _ = g.spline_detrender.detrend_chunk(buf, mask)
        if not np.all(mask_out):
            raise RuntimeError('varmap mc: the Detrender2d dropped an ill-conditioned zone even'
                               ' with an all-ones input mask, so L is not the linear operator'
                               ' the variance map assumes.')
        return residual[:, :, W:W+nt_in].astype(np.float32)


####################################   reporting   ####################################


def _spread(eps):
    """Delta(eps) = sqrt(<eps^2> - <eps>^2), guarded against tiny negative roundoff.

    The spread of the per-channel means ACROSS CHANNELS. It has two contributions: the genuine
    channel-to-channel differences in the true eps, which are fixed, and the statistical noise
    in each channel's own mean, which falls as 1/sqrt(n_chunks).

    MEASURED, on a 'varmap df' map of toy.yml, the second dominates well past 4000 chunks:
    Delta(eps) went 0.0145 -> 0.0094 -> 0.0044 at 400 / 1000 / 4000 chunks, i.e. 1/sqrt(n) the
    whole way. So do NOT read a shrinking Delta(eps) as anything but the statistics settling;
    it plateaus at the genuine spread only once that spread dominates, which had not happened
    by 4000 chunks there."""
    return float(np.sqrt(max(0.0, float(np.mean(eps ** 2)) - float(np.mean(eps)) ** 2)))


def _extremes(a, sigma, cnt_ok):
    """(eps_min, sigma_min, eps_max, sigma_max) over the eligible channels, or None."""
    if not cnt_ok.any():
        return None
    ae, se = a[cnt_ok], sigma[cnt_ok]
    lo, hi = int(np.argmin(ae)), int(np.argmax(ae))
    return float(ae[lo]), float(se[lo]), float(ae[hi]), float(se[hi])


def _fmt_extremes(x):
    if x is None:
        return f', eps_min/eps_max=n/a (need count>={_MIN_COUNT_WORST})'
    lo, slo, hi, shi = x
    return f', eps_max={hi:+.4g} ({shi:+.1f} sigma), eps_min={lo:+.4g} ({slo:+.1f} sigma)'


def _report(ichunk, y, mc_sum, mc_sumsq, mc_count, shapes, geom):
    """Print the per-tree and overall summaries; return a stats dict."""

    all_a, all_x, lines = [], [], []

    for itree in range(len(y)):
        D, M, P, K = shapes[itree]
        cnt = mc_count[itree]
        ready = cnt > 0
        r, R = geom.tree_r[itree], geom.tree_R[itree]
        tag = f'tree {itree} [r={r} R={R} M={M} K={K}]'

        if not ready.any():
            lines.append(f'  {tag}: no steady-state channels yet')
            continue

        sel = np.repeat(ready, M * P)
        n = np.repeat(cnt[ready], M * P).astype(np.float64)     # per-channel count
        s1, s2 = mc_sum[itree][sel], mc_sumsq[itree][sel]
        a = s1 / n / y[itree][sel] - 1.0                        # per-channel mean of eps

        # sigma_i = a_i / SE(a_i), SE = sqrt(v_i / n_i) with v_i the UNBIASED variance of the
        # per-chunk estimate. Slightly overstated: consecutive chunks share the PfSquare's
        # 'tpad' input samples, an O(tpad/nt_ds) correlation that this ignores.
        with np.errstate(divide='ignore', invalid='ignore'):
            var_ov = (s2 - s1**2 / n) / np.maximum(n - 1.0, 1.0)
            sigma = (s1 / n - y[itree][sel]) / np.sqrt(var_ov / n)

        elig = np.repeat(cnt[ready] >= _MIN_COUNT_WORST, M * P) & np.isfinite(sigma)
        x = _extremes(a, sigma, elig)
        all_a.append(a)
        if x is not None:
            all_x.append(x)

        lines.append(f'  {tag}: dm {int(ready.sum())}/{D} steady, {a.size} chans, '
                     f'mean(eps)={float(np.mean(a)):+.4g}, Delta(eps)={_spread(a):.4g}, '
                     f'count {int(cnt[ready].min())}..{int(cnt[ready].max())}'
                     + _fmt_extremes(x))

    stats = {}
    if all_a:
        e = np.concatenate(all_a)
        # Overall extremes: the most negative eps_min and the most positive eps_max across
        # trees, each keeping its own sigma.
        ov = None
        if all_x:
            lo = min(all_x, key=lambda t: t[0])
            hi = max(all_x, key=lambda t: t[2])
            ov = (lo[0], lo[1], hi[2], hi[3])
        atomic_print(f'[chunk {ichunk}] overall: {e.size} chans, '
                     f'mean(eps)={float(np.mean(e)):+.4g}, Delta(eps)={_spread(e):.4g}'
                     + _fmt_extremes(ov))
        stats = dict(nchans=int(e.size), mean_eps=float(np.mean(e)), delta_eps=_spread(e))
        if ov is not None:
            stats.update(eps_min=ov[0], sigma_min=ov[1], eps_max=ov[2], sigma_max=ov[3])
    else:
        atomic_print(f'[chunk {ichunk}] overall: no steady-state channels yet')

    for line in lines:
        atomic_print(line)
    return stats
