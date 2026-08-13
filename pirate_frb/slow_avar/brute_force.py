"""Brute-force computation of the dense variance map A (see notes/tree_dedispersion.tex,
section "Brute-force computation of the variance map").

The variance map A of a DedispersionTree is defined by y_alpha = sum_F A[alpha,F] v_F, where
v_F is the input variance of frequency channel F, and y_alpha is the variance of peak-finding
output element alpha = (coarse DM, multiplet m, profile p). Writing the dedisperser's linear
operator as L[alpha t, F t'], the tex notes show that

    A[alpha,F] = sum_{t'} L[alpha t, F t']^2 = sum_c sum_t (L e^{(F,t_c)})_alpha[t]^2

i.e. a column of A is obtained by applying L to a "one-hot" input array, squaring, and summing
over output time. The sum over c runs over the 2^gamma polyphase components of a time-downsampled
tree (gamma = primary_tree_index), and collapses to a single pass when the part of L upstream of
the downsampler is instantaneous in time (no detrender).
"""

import numpy as np

from ..utils import atomic_print


class BruteForceVarianceMap:
    """Computes the dense variance map A of every DedispersionTree in a plan, by brute force.

    Constructed from a DedispersionPlan and an optional Detrender2dParams (see the constructor
    for requirements on both), then run once::

        bf = BruteForceVarianceMap(plan, detrender=Detrender2dParams(...))
        A = bf.run(progress=True)     # list of (2^(r-R), M, P, nfreq) float64 arrays, one per tree

    This is very slow -- one full dedispersion pass per (input channel, polyphase) pair -- but it
    treats L as a black box, so unlike slow_avar.PfAvarExact it needs no per-stage analysis, and
    is the only algorithm we have that can handle a Detrender2d.

    A[itree] has shape (2^(r-R), M, P, nfreq), i.e. coarse DM slowest, matching the (ndm_out, M,
    nprofiles) layout of ReferencePeakFindingKernel's out_var. Flattening the first three axes
    gives the matrix A of the tex notes. Note the transpose relative to PfAvarExact.tree_variance
    and VarianceMapExact.eval_tree(), which are (M, 2^(r-R), P).
    """

    def __init__(self, plan, detrender=None, detrender_dtype=np.float64):
        """Checks that 'plan' is usable, and computes the sweep geometry (see class docstring).

        Args:
          plan: a DedispersionPlan. Three requirements, each checked here:

            - Every tree must have Dcore == 1, i.e. the config must set 'time_downsampling: 1'.
              With Dcore > 1 the peak-finder evaluates its convolutions on a sublattice of output
              times, which is harmless for the variance of a single output element (all elements
              are equal by translation invariance) but silently wrong for a sum over all times.
            - beams_per_gpu == beams_per_batch, i.e. one batch. The dedisperser's beam axis is
              a pure spectator here; a GPU sweep uses it to run several passes concurrently,
              but more than one batch would interleave passes across launches. run() (the CPU
              sweep) additionally requires beams_per_batch == 1.
            - 'dm_downsampling' must be left at 0 (auto-filled to 2^R), so that ndm_out == 2^(r-R).

          detrender: a Detrender2dParams, or None for no Detrender2d in L. Its nfreq, M and T
            must match the plan, so that the same object also drives the GPU kernel.
          detrender_dtype: working precision of the numpy detrender. float64 by default (the
            tex notes recommend it, and nothing downstream is more accurate); pass float32 to
            measure the detrender's own float32 penalty, or to match the GPU kernel.
        """

        self.plan = plan
        self.config = config = plan.config
        self.nfreq = int(plan.nfreq)
        self.nt_in = int(plan.nt_in)
        self.ntrees = int(plan.ntrees)
        self.detrender = detrender

        # The beam axis is a pure spectator here, and a GPU sweep uses it to run several
        # passes at once (see slow_avar.brute_force_gpu), so B > 1 is allowed. What is not
        # allowed is more than one BATCH, since that would interleave passes across launches.
        self.nbeams = int(plan.beams_per_batch)
        if int(plan.beams_per_gpu) != self.nbeams:
            raise RuntimeError("BruteForceVarianceMap requires beams_per_gpu == beams_per_batch"
                               f" (got {int(plan.beams_per_gpu)}, {self.nbeams}); set them in the"
                               " DedispersionConfig before building the plan")

        if detrender is None:
            self.W = 0    # Detrender2d time half-width (0 = no detrender)
            self.spline_detrender = None
        else:
            from ..detrending_spline import KnotVector, SplineDetrender

            detrender.validate()
            if int(detrender.nfreq) != self.nfreq:
                raise RuntimeError(f"BruteForceVarianceMap: detrender.nfreq={int(detrender.nfreq)}"
                                   f" != plan.nfreq={self.nfreq}")
            if int(detrender.M) != self.nbeams:
                raise RuntimeError(f"BruteForceVarianceMap: detrender.M={int(detrender.M)} !="
                                   f" beams_per_batch={self.nbeams}")
            if int(detrender.T) != self.nt_in:
                raise RuntimeError(f"BruteForceVarianceMap: detrender.T={int(detrender.T)} !="
                                   f" plan.nt_in={self.nt_in}")

            self.W = int(detrender.W)
            kv = KnotVector(np.asarray(detrender.knots, dtype=np.int64),
                            int(detrender.n_phi), self.nfreq)
            self.spline_detrender = SplineDetrender(kv, n=int(detrender.n), W=self.W,
                                                    eta=float(detrender.eta),
                                                    eps=float(detrender.eps),
                                                    dtype=detrender_dtype)

        # Per-tree geometry. 'gamma' is the input time-downsampling exponent of the tree, and
        # 'ddspread' is Delta_dd from the tex notes: the largest full-band delay searched by the
        # tree, in downsampled samples. (Per-subband lags do not add to this -- the output time
        # index is the arrival time extrapolated to the top of the tree's own band.)
        self.tree_gamma = []
        self.tree_r = []
        self.tree_R = []
        self.tree_M = []
        self.tree_P = []
        self.tree_D = []        # 2^(r-R) = ndm_out
        self.tree_nt_ds = []    # peak-finding input samples per chunk (= nt_in / 2^gamma)
        self.tree_ntime = []    # input samples needed per pass, for this tree alone

        gamma_max = max(int(t.primary_tree_index) for t in plan.trees)

        for itree in range(self.ntrees):
            t = plan.trees[itree]
            r, R = int(t.total_rank()), int(t.frequency_subbands.pf_rank)
            gamma, ndm_out = int(t.primary_tree_index), int(t.ndm_out)

            if int(t.Dcore) != 1:
                raise RuntimeError(f"BruteForceVarianceMap: tree {itree} has Dcore={int(t.Dcore)},"
                                   " expected 1 (set 'time_downsampling: 1' in the config)")
            if ndm_out != (1 << (r - R)):
                raise RuntimeError(f"BruteForceVarianceMap: tree {itree} has ndm_out={ndm_out} !="
                                   f" 2^(r-R)={1 << (r-R)}; leave 'dm_downsampling' at 0 in the config")

            wmax = int(t.pf.max_width)
            ddspread = 1 << (int(config.toplevel_tree_rank) - int(t.early_trigger_level))

            self.tree_gamma.append(gamma)
            self.tree_r.append(r)
            self.tree_R.append(R)
            self.tree_M.append(int(t.frequency_subbands.M))
            self.tree_P.append(int(t.nprofiles))
            self.tree_D.append(ndm_out)
            self.tree_nt_ds.append(int(t.nt_ds))
            # Eq. (bf_ntime) of the tex, generalized to a stream shared by all trees: the one-hot
            # sits at t0 = 2W + c with 0 <= c < 2^gamma_max, and this tree spreads the response
            # over (Delta_dd + 2*Wmax) downsampled samples, i.e. 2^gamma times as many input
            # samples. The 4W covers the placement of t0 plus the detrender's forward reach and
            # trailing padding.
            self.tree_ntime.append(4*self.W + (1 << gamma_max) + (1 << gamma) * (ddspread + 2*wmax))

        self.gamma_max = gamma_max

        # Number of polyphase passes per input channel. With no detrender (W = 0) everything
        # upstream of the time-downsampler is instantaneous in time, so all 2^gamma phases give
        # the same answer and one pass suffices (tex notes; checked by test_phase_collapse).
        self.nphases = (1 << gamma_max) if (self.W > 0) else 1

        # Weight applied to the sum over phases, per tree. Running 2^gamma_max phases visits each
        # residue class mod 2^gamma exactly 2^(gamma_max-gamma) times, hence the reciprocal here;
        # running a single phase instead requires the 2^gamma of the W=0 special case.
        self.tree_phase_weight = [float(1 << g) / self.nphases for g in self.tree_gamma]

        # One pass occupies 'ntime' input samples, rounded up to 'ndata_chunks' whole chunks.
        # run() appends one more "guard" chunk whose output must be identically zero, which is
        # the cheap check that no part of the impulse response was truncated. The one-hot must
        # land in the first chunk.
        self.ntime = max(self.tree_ntime)
        self.ndata_chunks = (self.ntime + self.nt_in - 1) // self.nt_in

        # The detrended one-hot occupies input times [t0-W, t0+W] with t0 = 2W + c, and
        # _run_pass() writes all of it into the first chunk of the interval.
        if 3*self.W + self.nphases > self.nt_in:
            raise RuntimeError(f"BruteForceVarianceMap: time_samples_per_chunk={self.nt_in} is too"
                               f" small to hold the (detrended) one-hot in one chunk (3W +"
                               f" 2^gamma_max = {3*self.W + self.nphases})")

    def npasses(self):
        """Number of dedispersion passes in a sweep (= nfreq * nphases). One pass per call to
        _run_pass(), each spanning ndata_chunks (+1 guard) chunks."""
        return self.nfreq * self.nphases

    def run(self, progress=False, guard_chunk=True):
        """Computes A for every tree, and returns it as a list of ntrees float64 numpy arrays
        of shape (2^(r-R), M, P, nfreq).

        Args:
          progress: if True, print a progress line every few percent of the sweep.
          guard_chunk: if True (recommended), each pass runs one extra all-zero chunk and asserts
            that its peak-finding output is identically zero. This is the only check that the
            impulse response was fully emitted -- an undersized sweep silently UNDERESTIMATES A,
            which is the one failure mode that matters (see eq:distance_function in the notes) --
            and, since passes run as one continuous stream, the only thing that establishes that
            a pass does not leak into the next one. It costs one chunk per pass, so it is
            cheapest when time_samples_per_chunk is chosen a few times smaller than self.ntime.
        """

        A = [np.zeros((self.tree_D[i], self.tree_M[i], self.tree_P[i], self.nfreq))
             for i in range(self.ntrees)]

        rdd = self._make_dedisperser()
        npasses = self.npasses()
        report_every = max(1, npasses // 20)

        if progress:
            atomic_print(f"BruteForceVarianceMap: {npasses} passes x"
                         f" {self.ndata_chunks + (1 if guard_chunk else 0)} chunks"
                         f" (nfreq={self.nfreq}, nphases={self.nphases}, ntime={self.ntime},"
                         f" nt_in={self.nt_in})")

        ipass = 0
        for ifreq in range(self.nfreq):
            for iphase in range(self.nphases):
                acc = self._run_pass(rdd, ifreq, iphase, ipass, guard_chunk=guard_chunk)
                for itree in range(self.ntrees):
                    A[itree][:, :, :, ifreq] += self.tree_phase_weight[itree] * acc[itree]
                ipass += 1
                if progress and (ipass % report_every == 0):
                    atomic_print(f"  BruteForceVarianceMap: pass {ipass}/{npasses}")

        self.check_structure(A)
        return A

    def _make_dedisperser(self):
        """Returns a ReferenceDedisperser with unit peak-finding weights (so that its out_var is
        the unnormalized sum of squares that the variance map is defined in terms of)."""

        from ..pirate_pybind11 import ReferenceDedisperser   # lazy: keep slow_avar import pybind-light

        if self.nbeams != 1:
            raise RuntimeError(f"BruteForceVarianceMap: the CPU sweep needs beams_per_batch == 1"
                               f" (got {self.nbeams}); the beam axis buys nothing on the CPU")

        rdd = ReferenceDedisperser(self.plan, 1, enable_variances=True)
        assert int(rdd.nbatches) == 1, int(rdd.nbatches)

        for w in rdd.wt_arrays:
            w[...] = 1.0

        # out_var is a per-chunk MEAN; multiplying by samples_per_chunk recovers the raw sum of
        # squares, which is what _run_pass() accumulates. Every profile has the same count here,
        # because Dcore == 1 (checked in __init__); assert it rather than assume it, since a
        # per-profile count would silently rescale part of the answer.
        for itree in range(self.ntrees):
            spc = list(rdd.pf_kernels[itree].samples_per_chunk)
            assert spc == [self.tree_nt_ds[itree]] * len(spc), (itree, spc)

        return rdd

    def _run_pass(self, rdd, ifreq, ipass_phase, ipass, guard_chunk=True):
        """Applies L to the one-hot e^(F,t_c), and returns [sum_t y^2] per tree, as a list of
        (2^(r-R), M, P) float64 arrays.

        Passes are laid end to end in one continuous stream (pass 'ipass' occupies chunks
        [ipass*nchunks, (ipass+1)*nchunks) of 'rdd'), so no persistent state is ever reset: the
        guard chunk is what proves that one pass's response has died out before the next one's
        one-hot arrives.
        """

        nchunks = self.ndata_chunks + (1 if guard_chunk else 0)
        acc = [np.zeros((self.tree_D[i], self.tree_M[i], self.tree_P[i])) for i in range(self.ntrees)]
        t0 = 2*self.W + ipass_phase
        resp = self._one_hot_response(ifreq)

        for j in range(nchunks):
            rdd.input_array[...] = 0.0
            self._write_one_hot(rdd, resp, t0, j)
            rdd.dedisperse(ipass * nchunks + j, 0)

            for itree in range(self.ntrees):
                # out_var is a per-chunk mean square; the nt_ds factor turns it back into a raw
                # sum of squares (see the samples_per_chunk assert in _make_dedisperser).
                ov = np.asarray(rdd.out_var[itree])[0] * self.tree_nt_ds[itree]
                if j < self.ndata_chunks:
                    acc[itree] += ov
                elif np.any(ov != 0.0):
                    raise RuntimeError(
                        f"BruteForceVarianceMap: guard chunk of pass (ifreq={ifreq},"
                        f" iphase={ipass_phase}) is nonzero for tree {itree} (max"
                        f" {float(np.abs(ov).max()):.4g}): the impulse response was truncated,"
                        f" i.e. ntime={self.ntime} is too small")

        return acc

    def _one_hot_response(self, ifreq):
        """Returns what a one-hot in channel 'ifreq' contributes to the dedisperser's input
        stream, as a (nfreq, 2W+1) array covering input times [t-W, t+W] relative to the
        one-hot's own time t. With no detrender that is the one-hot itself (W = 0).

        Everything outside that window is exactly zero, so this short buffer -- rather than a
        full chunk -- is all that has to be detrended. The detrender fits a window of 2W+1
        samples centred on each output, so an output further than W from the one-hot sees
        all-zero data, and a least-squares fit to zero data has zero residual. That makes the
        detrender essentially free: 4W+1 samples per pass instead of nt_in.

        The mask is all-ones, which is what makes L a fixed LINEAR operator: the detrender's
        mask expansion is driven by r_min, which depends on the mask and the basis but not on
        the data. If a zone were ever dropped, L would not be the operator this tool assumes,
        so that is checked here rather than trusted.
        """

        W, nfreq = self.W, self.nfreq

        if self.spline_detrender is None:
            resp = np.zeros((nfreq, 1))
            resp[ifreq, 0] = 1.0
            return resp

        buf = np.zeros((1, nfreq, 4*W + 1), dtype=self.spline_detrender.dtype)
        buf[0, ifreq, 2*W] = 1.0
        mask = np.ones(buf.shape, dtype=bool)

        residual, mask_out, _ = self.spline_detrender.detrend_chunk(buf, mask)
        if not np.all(mask_out):
            raise RuntimeError("BruteForceVarianceMap: the Detrender2d dropped an ill-conditioned"
                               " zone even with an all-ones input mask, so L is not the linear"
                               " operator this tool assumes. Lower 'eps', or use more/wider zones.")

        return residual[0]

    def _write_one_hot(self, rdd, resp, t_abs, jchunk):
        """Writes the part of a _one_hot_response() placed at absolute input time 't_abs' that
        falls in chunk 'jchunk' of the stream. rdd.input_array must already be zeroed."""

        lo = t_abs - self.W - jchunk*self.nt_in     # response index 0, relative to chunk start
        a = max(0, -lo)
        b = min(resp.shape[1], self.nt_in - lo)
        if b > a:
            rdd.input_array[0, :, lo+a : lo+b] = resp[:, a:b]

    def check_structure(self, A):
        """Structural checks on a computed variance map: A >= 0 everywhere, and for a tree that
        does not search the full band, the columns of input channels below the tree's frequency
        range vanish identically."""

        for itree in range(self.ntrees):
            a = A[itree]
            if np.any(a < 0.0):
                raise RuntimeError(f"BruteForceVarianceMap: tree {itree} has a negative variance-map"
                                   f" entry ({float(a.min()):.4g}); this should be impossible")

            # Channels entirely below the tree's lowest searched frequency. A detrender spreads a
            # one-hot over every channel of its spline zone, so with one configured the cut moves
            # down to the start of the zone containing that channel.
            fmin = float(self.plan.trees[itree].trigger_frequency)
            nlo = int(np.floor(self.config.frequency_to_index(fmin)))
            if self.spline_detrender is not None:
                nlo = self._zone_lo(nlo)
            if (nlo > 0) and np.any(a[:, :, :, :nlo] != 0.0):
                raise RuntimeError(f"BruteForceVarianceMap: tree {itree} does not search below"
                                   f" {fmin:.4g} MHz (input channel {nlo}), but the columns of"
                                   f" channels 0..{nlo-1} are not identically zero (max"
                                   f" {float(np.abs(a[:,:,:,:nlo]).max()):.4g})")

    def _zone_lo(self, ifreq):
        """Returns the first channel of the detrender spline zone containing channel 'ifreq'
        (or nfreq if ifreq is out of range)."""

        from ..detrending_spline import zone_channel_ranges

        for (lo, hi) in zone_channel_ranges(self.spline_detrender.kv):
            if lo <= ifreq < hi:
                return lo
        return self.nfreq

    @staticmethod
    def test_vs_per_tfm(toplevel_tree_rank=8, subband_counts=None, num_primary_trees=1,
                        num_early_triggers=0, verbose=True):
        """Compares the brute-force variance map, element by element, against
        PfAvarExact.per_tfm (which computes the same matrix by propagating compressed sparse
        tiles, sharing no code with the dedisperser).

        This is the decisive correctness test, and doubles as the float32 measurement: brute force
        runs the float32 ReferenceTree and ReferencePeakFindingKernel, while per_tfm is float64
        throughout. Only valid with no detrender (per_tfm cannot represent one).
        """

        from ..pirate_pybind11 import DedispersionPlan
        from .PfVariance import PfAvarExact

        if subband_counts is None:
            subband_counts = [1]

        config = _make_test_config(toplevel_tree_rank, subband_counts,
                                   num_primary_trees=num_primary_trees,
                                   num_early_triggers=num_early_triggers)
        plan = DedispersionPlan(config, cdd2_kernel_required=False)
        bf = BruteForceVarianceMap(plan)
        A = bf.run()

        exact = PfAvarExact(plan, np.ones(bf.nfreq))
        worst, worst_where = 0.0, None
        eps = []

        for itree in range(bf.ntrees):
            M, D = bf.tree_M[itree], bf.tree_D[itree]
            all_dbits = (1 << (bf.tree_r[itree] - bf.tree_R[itree])) - 1
            for ifreq in range(bf.nfreq):
                # per_tfm[itree][ifreq][m] is None for multiplets this channel does not reach.
                want = np.stack([pv.unpack(all_dbits) if (pv is not None) else np.zeros((D, bf.tree_P[itree]))
                                 for pv in exact.per_tfm[itree][ifreq]])       # (M, D, P)
                want = want.transpose(1, 0, 2)                                 # (D, M, P)
                got = A[itree][:, :, :, ifreq]
                assert got.shape == want.shape, (got.shape, want.shape)

                nz = (want != 0.0)
                if np.any(got[~nz] != 0.0):
                    raise RuntimeError(f"test_vs_per_tfm: tree {itree}, channel {ifreq}: brute force"
                                       " is nonzero where per_tfm predicts an exact zero")
                if not np.any(nz):
                    continue

                e = got[nz] / want[nz] - 1.0
                eps.append(e)
                k = int(np.argmax(np.abs(e)))
                if abs(float(e[k])) > worst:
                    worst, worst_where = abs(float(e[k])), (itree, ifreq, float(e[k]))

        eps = np.concatenate(eps)
        if verbose:
            atomic_print(f"    test_vs_per_tfm(r={toplevel_tree_rank}, subbands={subband_counts},"
                         f" npri={num_primary_trees}, net={num_early_triggers}):"
                         f" {eps.size} nonzero elements, eps = A_bruteforce/A_per_tfm - 1:"
                         f" mean {float(np.mean(eps)):+.3g}, range [{float(eps.min()):+.3g},"
                         f" {float(eps.max()):+.3g}], worst |eps| {worst:.3g} at"
                         f" (tree,ifreq)={worst_where[:2]}")

        # Loose enough to pass, tight enough to catch anything but float32 roundoff: the
        # dedispersion chain is float32, so relative errors of a few times 1e-7 are expected.
        assert worst < 1.0e-5, (worst, worst_where)

    @staticmethod
    def test_phase_collapse(toplevel_tree_rank=8, verbose=True):
        """With no detrender, the 2^gamma polyphase passes of a time-downsampled tree must give
        the same result (tex notes: everything upstream of the downsampler is instantaneous in
        time). This is the sharpest available test of the polyphase logic, and of the single-pass
        shortcut that run() takes when there is no detrender.

        Agreement is not bit-exact, even though the float32 output samples themselves are: shifting
        the one-hot moves the response relative to the chunk boundaries, so the same set of squared
        samples is accumulated into out_var in a different order. The tolerance below is still six
        orders of magnitude below the float32 noise floor of the dedispersion chain.
        """

        from ..pirate_pybind11 import DedispersionPlan

        # Three primary trees => gamma = 0, 1, 2, so the phase loop has something to collapse.
        config = _make_test_config(toplevel_tree_rank, [2, 2, 1], num_primary_trees=3)
        plan = DedispersionPlan(config, cdd2_kernel_required=False)
        bf = BruteForceVarianceMap(plan)
        assert bf.gamma_max == 2, bf.gamma_max

        nphases = 1 << bf.gamma_max
        rdd = bf._make_dedisperser()
        worst = 0.0

        for ipass, ifreq in enumerate([0, bf.nfreq // 3, bf.nfreq - 1]):
            ref = None
            for iphase in range(nphases):
                acc = bf._run_pass(rdd, ifreq, iphase, ipass*nphases + iphase)
                if ref is None:
                    ref = acc
                    continue
                for itree in range(bf.ntrees):
                    # Phases c and c + 2^gamma are the same residue class mod 2^gamma, so they
                    # must agree for every tree; with W = 0 all 2^gamma_max phases do.
                    scale = float(np.abs(ref[itree]).max())
                    e = float(np.abs(acc[itree] - ref[itree]).max()) / scale if (scale > 0) else 0.0
                    if e > 1.0e-12:
                        raise RuntimeError(f"test_phase_collapse: tree {itree}, channel {ifreq}:"
                                           f" phase {iphase} differs from phase 0 (relative {e:.4g})")
                    worst = max(worst, e)

        if verbose:
            atomic_print(f"    test_phase_collapse(r={toplevel_tree_rank}): {nphases} phases agree"
                         f" for all {bf.ntrees} trees, worst relative difference {worst:.3g}")

    @staticmethod
    def test_column_norms(toplevel_tree_rank=6, subband_counts=None, num_primary_trees=1,
                          num_early_triggers=0, detrender=True, nifreq=2, verbose=True):
        """Evaluates the defining identity A[alpha,F] = sum_{t'} L[alpha t, F t']^2 LITERALLY --
        one pass per input time t', reading the output of one fixed chunk -- and compares it to
        what run() computes, which is instead a sum over output times for one input time.

        This is the test of the core math: the row-norm/column-norm exchange, the polyphase sum
        over 2^gamma phases, and the ntime/t0 sizing. It is also the only test that covers the
        Detrender2d path, since no analytic oracle can. It costs (ntime + nt_in) passes per
        channel, i.e. more than a whole sweep, so it runs at toy scale on a few channels.

        Note that the sum over t' below runs over EVERY input time, with no phase weighting: the
        polyphase decomposition of the tex notes is a way of organizing this sum, and summing it
        directly is what makes this an independent check of that organization.
        """

        from ..pirate_pybind11 import DedispersionPlan

        if subband_counts is None:
            subband_counts = [2, 1]

        config = _make_test_config(toplevel_tree_rank, subband_counts,
                                   num_primary_trees=num_primary_trees,
                                   num_early_triggers=num_early_triggers)
        plan = DedispersionPlan(config, cdd2_kernel_required=False)
        dparams = _make_test_detrender(config) if detrender else None
        bf = BruteForceVarianceMap(plan, detrender=dparams)
        A = bf.run()

        # The probe chunk sits far enough into the stream that every t' able to reach it lies in
        # [tlo, thi), and one chunk short of the end so that t' AFTER the probe chunk is covered
        # too -- the detrender is not causal, and reaches W samples back. Both extremes of the
        # range are asserted to contribute nothing, which is what makes the range wide enough.
        nchunks = bf.ndata_chunks + 3
        kprobe = nchunks - 2
        tlo = kprobe*bf.nt_in - bf.ntime
        thi = nchunks*bf.nt_in
        assert tlo >= bf.W, (tlo, bf.W)

        ifreqs = [(i * bf.nfreq) // nifreq + bf.nfreq // (2*nifreq) for i in range(nifreq)]
        worst, worst_where = 0.0, None

        for ifreq in ifreqs:
            col = [np.zeros((bf.tree_D[i], bf.tree_M[i], bf.tree_P[i])) for i in range(bf.ntrees)]
            resp = bf._one_hot_response(ifreq)

            for t_in in range(tlo, thi):
                # A fresh dedisperser per t', rather than one continuous stream: a t' near the end
                # of the interval would otherwise leak into the next one, and here correctness
                # matters more than the (toy-scale) cost.
                rdd = bf._make_dedisperser()
                edge = (t_in == tlo) or (t_in == thi-1)
                for j in range(nchunks):
                    rdd.input_array[...] = 0.0
                    bf._write_one_hot(rdd, resp, t_in, j)
                    rdd.dedisperse(j, 0)
                    if j != kprobe:
                        continue
                    for itree in range(bf.ntrees):
                        # out_var is the MEAN over the chunk's nt_ds output times, each in steady
                        # state and so each equal to the same column norm -- so summing out_var
                        # over t' gives A directly, with no nt_ds factor (unlike _run_pass(),
                        # which needs one because it sums a single response over time).
                        ov = np.asarray(rdd.out_var[itree])[0]
                        col[itree] += ov
                        if edge and np.any(ov != 0.0):
                            raise RuntimeError(f"test_column_norms: input time t'={t_in} still"
                                               f" reaches the probe chunk in tree {itree}, so the"
                                               f" sum over t' is incomplete")

            for itree in range(bf.ntrees):
                want, got = A[itree][:, :, :, ifreq], col[itree]
                scale = float(np.abs(want).max())
                if scale == 0.0:
                    assert not np.any(got != 0.0), (itree, ifreq)
                    continue
                e = float(np.abs(got - want).max()) / scale
                if e > worst:
                    worst, worst_where = e, (itree, ifreq)

        if verbose:
            atomic_print(f"    test_column_norms(r={toplevel_tree_rank}, subbands={subband_counts},"
                         f" npri={num_primary_trees}, net={num_early_triggers},"
                         f" detrender={bool(detrender)}):"
                         f" {len(ifreqs)} columns x {thi-tlo} input times, worst relative"
                         f" difference {worst:.3g} at (tree,ifreq)={worst_where}")

        # float32 dedispersion, and the two sides accumulate different numbers of terms.
        assert worst < 1.0e-5, (worst, worst_where)


    @staticmethod
    def test_detrender_fp32(toplevel_tree_rank=8, nifreq=16, verbose=True):
        """Measures the Detrender2d's own float32 penalty, by running the numpy detrender at
        float32 and float64 on the same one-hots.

        The tool itself runs the detrender at float64 (the rest of the chain is float32, so
        that is the accurate end), but the GPU Detrender2d is float32-only, so this is the
        error budget an eventual GPU sweep inherits from that stage. Reported as the signed
        relative error on the squared norm of each detrended one-hot, which is what enters A.
        """

        from ..pirate_pybind11 import DedispersionPlan

        config = _make_test_config(toplevel_tree_rank, [1])
        plan = DedispersionPlan(config, cdd2_kernel_required=False)
        dparams = _make_test_detrender(config)

        bf64 = BruteForceVarianceMap(plan, detrender=dparams, detrender_dtype=np.float64)
        bf32 = BruteForceVarianceMap(plan, detrender=dparams, detrender_dtype=np.float32)

        eps = []
        for i in range(nifreq):
            ifreq = (i * bf64.nfreq) // nifreq + bf64.nfreq // (2*nifreq)
            r64 = bf64._one_hot_response(ifreq).astype(np.float64)
            r32 = bf32._one_hot_response(ifreq).astype(np.float64)
            eps.append(np.sum(r32**2) / np.sum(r64**2) - 1.0)

        eps = np.array(eps)
        if verbose:
            atomic_print(f"    test_detrender_fp32(r={toplevel_tree_rank}): {nifreq} channels,"
                         f" eps = ||r_fp32||^2/||r_fp64||^2 - 1: mean {float(np.mean(eps)):+.3g},"
                         f" range [{float(eps.min()):+.3g}, {float(eps.max()):+.3g}]")

        assert float(np.abs(eps).max()) < 1.0e-4, float(np.abs(eps).max())


def _make_test_detrender(config, n_phi=2, n=2, W=4, nzone=2, kint=3):
    """Returns a Detrender2dParams matching 'config', for the brute-force tests."""

    from ..pirate_pybind11 import Detrender2dParams
    from ..detrending_spline.masks import zoned_knots

    nfreq = int(config.get_total_nfreq())
    kv = zoned_knots(n_phi, nfreq, nzone, kint)

    return Detrender2dParams(nfreq=nfreq, knots=[int(x) for x in kv.knots], M=1, n_phi=n_phi,
                             n=n, W=W, T=int(config.time_samples_per_chunk))


def _make_test_config(toplevel_tree_rank, subband_counts, num_primary_trees=1,
                      num_early_triggers=0, max_width=4, nfreq=None, nt_in=None):
    """Returns a small DedispersionConfig suitable for the brute-force tool: one frequency zone,
    one beam, 'time_downsampling: 1' (so that Dcore == 1) and 'dm_downsampling: 0' (auto)."""

    from ..pirate_pybind11 import DedispersionConfig, PrimaryTree

    nfreq = nfreq if (nfreq is not None) else (1 << toplevel_tree_rank)

    # Default chunk length: half the dedispersion span, but at least the config's minimum
    # (nelts_per_segment = 32, times 2^(num_primary_trees-1) for the time-downsampled trees).
    if nt_in is None:
        nt_in = max(1 << (toplevel_tree_rank - 1), 32 << (num_primary_trees - 1))

    # Weight arrays are unused here (run() sets them to 1), so downsample them as far as the
    # config allows: wt_dm_downsampling is capped by the total rank of the smallest tree, and
    # wt_time_downsampling by that tree's nt_ds.
    min_total_rank = toplevel_tree_rank - num_early_triggers - (1 if num_primary_trees > 1 else 0)

    config = DedispersionConfig()
    config.zone_nfreq = [nfreq]
    config.zone_freq_edges = [400.0, 800.0]
    config.time_sample_ms = 1.0
    config.dtype = np.float32
    config.toplevel_tree_rank = toplevel_tree_rank
    config.time_samples_per_chunk = nt_in
    config.frequency_subband_counts = subband_counts
    config.primary_trees = [
        PrimaryTree(num_early_triggers, max_width, 0, 1, 1 << min_total_rank, nt_in >> ipri)
        for ipri in range(num_primary_trees)
    ]
    config.beams_per_gpu = 1
    config.beams_per_batch = 1
    config.num_active_batches = 1
    config.validate()

    return config
