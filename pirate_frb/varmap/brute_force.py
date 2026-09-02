"""Brute-force computation of the variance map A, for every DedispersionTree of a config.

The variance map A of a DedispersionTree is defined by ``y_alpha = sum_F A[alpha,F] v_F``,
where ``v_F`` is the input variance of frequency channel F and ``y_alpha`` is the variance of
peak-finding output element alpha. Writing the dedisperser's linear operator as
``L[alpha t, F t']``, notes/variance_map.tex shows that

    A[alpha,F] = sum_{t'} L[alpha t, F t']^2 = sum_c sum_t (L e^{(F,t_c)})_alpha[t]^2

i.e. a COLUMN of A is obtained by applying L to a "one-hot" input array, squaring, and summing
over output time. The sum over c runs over the 2^gamma polyphase components of a
time-downsampled tree, and collapses to a single pass when the part of L upstream of the
downsampler is instantaneous in time (no detrender).

This treats L as a black box -- it runs the shipped dedisperser -- so unlike the analytic
route of detrender_free.py it needs no per-stage analysis, and it is the only algorithm we have
that can handle a Detrender2d. It is also very slow: one full dedispersion pass per
(input channel, polyphase) pair.

ONE COLUMN AT A TIME, WHICH IS WHAT MAKES CHORD REACHABLE
--------------------------------------------------------
Because the sweep produces A one input channel at a time, the coarse-graining max-reduction
can run INSIDE the pass loop and the column is then discarded. Pass an ``L`` to
compute_variance_multimap() and the dense A is never formed: at CHORD tree 0 the dense A is
1.2 TiB, while the coarse map is 86 GiB at L = 6 and 344 GiB at L = 4. Leaving L at None keeps
the dense fine map, which is only viable at subscale.

THE MAP IS PRODUCED BY COLUMN AND STORED BY ROW, and reconciling those cheaply is the other
half of what makes CHORD reachable. Writing each column into the natural ``(nrows, nfreq)``
layout as it arrives would touch one cache line per output row, on an array of matrix size. So
the accumulator STAGES a small block of columns and transposes each full block into the output
-- see _Accumulator, which also says why the block is small and why the transpose is tiled. The
output is the only array of matrix size the sweep holds; accumulating a whole ``(nfreq, nrows)``
transpose and turning it round at the end would need two, which is 688 GiB rather than 344 at
CHORD's L = 4.
"""

import os
import time

import numpy as np

from ..utils import atomic_print
from .VarianceMap import VarianceMap, make_plan, coarse_grain_vector, _subband_tables
from .VarianceMultiMap import VarianceMultiMap


def compute_variance_multimap(config, detrender=None, *, device='gpu', L=None,
                              guard_chunk=True, progress=False, channels=None,
                              scratch_dir=None, provenance=None,
                              detrender_dtype=np.float64):
    """Compute the variance map of every PRIMARY tree in 'config' by brute force, and return
    a VarianceMultiMap.

    Every tree is swept -- they share one dedisperser and one pass over the input channels --
    but only the (gamma, 0) maps are kept, since an early-trigger tree's map is a row subset
    of its parent's (see VarianceMultiMap). Use sweep_all_trees_dense() if you want the child
    matrices themselves.

    A CUDA device must already be selected (``ksgpu.set_cuda_device()``), even for
    ``device='cpu'``: DedispersionPlan allocates through cudaHostAlloc.

    Parameters
    ----------
    config : DedispersionConfig
        One requirement, checked up front: ``beams_per_gpu == beams_per_batch``. The
        output-array coarse-graining needs no check -- ``DedispersionTree``'s
        ``dm_downsampling`` and ``time_downsampling`` are both ``2^dd_rank1`` by construction
        and neither is a config field. The latter sets the peak-finder's Dcore, which
        neither sweep reads.
        The beam count comes from ``config.beams_per_batch``: on the GPU the beam axis is a
        pure spectator, so a batch of B beams runs B distinct passes concurrently. Measurement
        found that batching does not speed up a full sweep, so the CLI forces 1.
    detrender : Detrender2dParams, optional
        None for no Detrender2d in L. Its nfreq, M and T must match the config.
    device : {'gpu', 'cpu'}
        The CPU sweep additionally needs ``beams_per_batch == 1``; the GPU sweep needs a
        float32 config, one primary tree, and a compiled sbdd kernel per tree.
    L : None, int, or length-num_primary_trees sequence
        None: the returned maps are DENSE and FINE, with ``y_true`` set to their row sums.
        Otherwise: the sweep max-reduces each column into its groups AS THE COLUMN IS PRODUCED
        and the dense A is never formed, so the returned maps are the minimal coarse maps
        Abar. Requires ``nphases == 1`` -- with several passes summing into one column the
        max-reduction cannot run until the sum is complete -- and each map's L must satisfy
        ``R <= L <= r``, which differs per primary tree, which is why the sequence form
        exists. An entry may itself be None, leaving that one map dense and fine.

        PER PRIMARY TREE, since that is what the returned multimap holds. The early-trigger
        trees are swept (they share the dedisperser) but discarded, so they are always swept
        fine -- a child never carries a coarse-graining rank of its own.
    guard_chunk : bool
        Run one extra all-zero chunk per pass and require its peak-finding output to be
        identically zero. This is the only check that the impulse response was fully emitted;
        an undersized sweep silently UNDERESTIMATES A, which is the one failure mode that
        matters. It also establishes that one pass does not leak into the next, since passes
        run as one continuous stream.
    channels : sequence of int, optional
        Sweep only these input channels. The result is then a PARTIAL map -- every unswept
        column is zero -- so it is returned with ``y_true = None``, which is what stops
        anything downstream from scoring it. For timing a sweep before committing to it.
    scratch_dir : str, optional
        Back each tree's matrix with an on-disk memmap rather than RAM. The fallback for a
        config that does not fit; see _alloc(). The files ARE the returned maps' storage, so
        do not delete 'scratch_dir' before writing them out.
    provenance : dict, optional
        Merged into the multimap's provenance, after the sweep's own record. For the caller's
        bookkeeping -- the CLI puts its config overrides here.
    detrender_dtype : dtype
        Working precision of the NUMPY detrender, i.e. of the CPU sweep. Ignored by the GPU
        sweep, whose Detrender2d kernel is float32. Set it to float32 to compare the two
        devices at matched detrender precision, which is what makes such a comparison a test
        of the DRIVER rather than of the detrender.
    """

    t_start = time.time()

    # 'L' is per PRIMARY tree, but the accumulator coarse-grains per tree, so spread it onto
    # the (gamma, 0) trees and leave the children fine. Their matrices are discarded below.
    npri = int(config.num_primary_trees)
    if L is None:
        Ls_tree = None
    else:
        Lp = ([int(L)] * npri if np.isscalar(L)
              else [None if (x is None) else int(x) for x in L])
        if len(Lp) != npri:
            raise RuntimeError(f'compute_variance_multimap: got {len(Lp)} values of L for'
                               f' {npri} primary trees')
        Ls_tree = [None] * int(config.num_dedispersion_trees)
        for g in range(npri):
            Ls_tree[int(config.dedispersion_tree_index(g, 0))] = Lp[g]

    geom, sweep, acc = _run_sweep(config, detrender, device=device, L=Ls_tree,
                                  guard_chunk=guard_chunk, progress=progress,
                                  channels=channels, scratch_dir=scratch_dir,
                                  detrender_dtype=detrender_dtype,
                                  _caller='compute_variance_multimap')

    maps = acc.finish(device=device, guard_chunk=guard_chunk, sweep_seconds=sweep.seconds,
                      progress=progress, partial=(channels is not None))

    # The sweep produces one matrix per TREE (they share a dedisperser, so the child trees
    # cost no extra passes), but a VarianceMultiMap stores one map per PRIMARY tree: a child's
    # map is a row subset of its parent's, so keeping it would be keeping a copy. Note the
    # parent is the LAST tree of its family, not itree - e.
    primary = [maps[int(config.dedispersion_tree_index(g, 0))]
               for g in range(int(config.num_primary_trees))]

    prov = dict(algorithm='brute_force', device=device, nbeams=geom.nbeams,
                nfreq=geom.nfreq, nphases=geom.nphases, ntime=geom.ntime,
                ndata_chunks=geom.ndata_chunks, guard_chunk=bool(guard_chunk),
                npasses=sweep.npasses, partial=(channels is not None),
                sweep_seconds=sweep.seconds, reduce_seconds=acc.seconds,
                transpose_seconds=acc.transpose_seconds,
                total_seconds=time.time() - t_start,
                min_A=acc.gmin, max_A=acc.gmax,
                detrender=(detrender is not None))
    if provenance:
        prov.update(provenance)

    return VarianceMultiMap(config, primary, detrender=detrender, plan=acc.plan,
                            provenance=prov)


def _run_sweep(config, detrender=None, *, device='gpu', L=None, guard_chunk=True,
               progress=False, channels=None, scratch_dir=None,
               detrender_dtype=np.float64, _caller='_run_sweep'):
    """Configure and run one brute-force sweep; return (geometry, sweep, accumulator).

    This is the DRIVER, shared by compute_variance_multimap() and sweep_all_trees_dense(), so
    that a sweep is configured in exactly one place. It stops short of acc.finish(), which is
    where the choice of representation lives and which the two callers make differently.
    """

    geom = _SweepGeometry(config, detrender, detrender_dtype=detrender_dtype)

    Ls = [None]*geom.ntrees if (L is None) else (
        [int(L)]*geom.ntrees if np.isscalar(L)
        else [None if (x is None) else int(x) for x in L])
    if len(Ls) != geom.ntrees:
        raise RuntimeError(f'{_caller}: got {len(Ls)} values of L for {geom.ntrees} trees')

    if any(x is not None for x in Ls) and (geom.nphases != 1):
        raise RuntimeError(
            f'{_caller}: the streaming (coarse-grained) path needs nphases =='
            f' 1, but this config gives nphases = {geom.nphases}. With several polyphase'
            ' passes summing into one column, the max-reduction cannot run until the sum is'
            ' complete. Sweep with L=None and coarse_grain() afterwards, or use a config with'
            ' no detrender or no time-downsampled trees.')

    if device == 'cpu':
        sweep = _CpuSweep(geom)
    elif device == 'gpu':
        sweep = _GpuSweep(geom)
    else:
        raise RuntimeError(f"{_caller}: device={device!r}, expected 'cpu' or 'gpu'")

    acc = _Accumulator(geom, Ls, scratch_dir)
    for (ifreq, cols) in sweep.columns(channels=channels, guard_chunk=guard_chunk,
                                       progress=progress):
        acc.add(ifreq, cols)

    return geom, sweep, acc


def sweep_all_trees_dense(config, detrender=None, *, device='cpu', guard_chunk=True,
                          progress=False, scratch_dir=None, detrender_dtype=np.float64):
    """Sweep every tree and return the RAW dense matrices: a length-ntrees list of
    ``(nalpha, nfreq)`` ndarrays, indexed by itree.

    For TESTS, and specifically for comparing an early-trigger tree's matrix against its
    (primary_tree_index, 0) parent's -- which is what test_restriction_vs_sweep() does.
    compute_variance_multimap() is what production calls.

    Returns arrays rather than VarianceMaps ON PURPOSE. A VarianceMap is a PRIMARY tree's map
    by convention, and manufacturing one for an early-trigger tree would make that convention
    unenforceable: with arrays, a later change could assert in VarianceMap.__init__ that
    'itree' names an early_trigger_level == 0 tree without anything here having to be
    unpicked. It is also the honest signature -- a sweep produces numbers, and the choice of
    representation (dense or factored, fine or coarse, with y_true and history) belongs to
    compute_variance_multimap().

    Two constraints, both consequences of what it is for:

    - FINE only. There is no 'L' argument: with one set, acc.A[itree] would be coarse and have
      nbeta rows rather than nalpha, and a matrix comparison wants fine matrices anyway.
    - TEST SCALE only. It materializes every tree's dense matrix, including the early-trigger
      trees that compute_variance_multimap() does not store. At CHORD the (3,3) child alone
      is 12.0 GiB.

    Note this deliberately bypasses acc.finish(), and with it y_true, the history record and
    check_ref_covers_y_true(). That is fine for child matrices that nothing stores, but it is
    why production must keep going through compute_variance_multimap().
    """

    _, _, acc = _run_sweep(config, detrender, device=device, L=None,
                           guard_chunk=guard_chunk, progress=progress,
                           scratch_dir=scratch_dir, detrender_dtype=detrender_dtype,
                           _caller='sweep_all_trees_dense')
    acc._flush()

    out = []
    for itree in range(acc.geom.ntrees):
        A = acc.A[itree]
        if isinstance(A, np.memmap):
            A.flush()          # scratch_dir case: make the file match the mapping
        out.append(np.asarray(A))
    return out


####################################   sweep geometry   ####################################


class _SweepGeometry:
    """The part of a sweep that does not depend on the device: what the config must satisfy,
    the per-tree geometry, and how long one pass has to be.

    Held by both _CpuSweep and _GpuSweep. Split out because the two sweeps disagree about
    almost everything else, and because the config rejections all belong in one place.
    """

    def __init__(self, config, detrender=None, detrender_dtype=np.float64):
        """Builds the DedispersionPlan, checks it, and computes the sweep geometry.

        'detrender_dtype' is the working precision of the numpy detrender used for the
        one-hot responses. float64 by default (notes/variance_map.tex recommends it, and
        nothing downstream is more accurate); pass float32 to measure the detrender's own
        float32 penalty, or to match the GPU kernel.

        The CALLER'S config object is the one kept and handed to the returned maps -- not
        ``plan.config``, which is a distinct object with the same contents, and which
        VarianceMultiMap would refuse. Nothing here is allowed to write to it.
        """

        from ..pirate_pybind11 import DedispersionPlan

        self.config = config
        params = DedispersionPlan.Params(dcore_from_cdd2_registry=False)
        self.plan = plan = DedispersionPlan(config, params)
        self.nfreq = int(plan.nfreq)
        self.nt_in = int(plan.nt_in)
        self.ntrees = int(plan.ntrees)
        self.detrender = detrender

        # Collected rather than raised one at a time: no shipped config satisfies all of
        # these, so a user editing one would otherwise discover the requirements one run at a
        # time.
        errs = []

        # The beam axis is a pure spectator, and a GPU sweep uses it to run several passes at
        # once, so B > 1 is allowed. What is not allowed is more than one BATCH, since that
        # would interleave passes across launches.
        self.nbeams = int(plan.beams_per_batch)
        if int(plan.beams_per_gpu) != self.nbeams:
            errs.append(f'beams_per_gpu = {int(plan.beams_per_gpu)} != beams_per_batch ='
                        f' {self.nbeams}; set both in the DedispersionConfig before building'
                        ' the plan')

        if detrender is None:
            self.W = 0    # Detrender2d time half-width (0 = no detrender)
            self.spline_detrender = None
        else:
            from ..detrending_spline import KnotVector, SplineDetrender

            detrender.validate()
            if int(detrender.nfreq) != self.nfreq:
                errs.append(f'detrender.nfreq = {int(detrender.nfreq)} != plan.nfreq ='
                            f' {self.nfreq}')
            if int(detrender.M) != self.nbeams:
                errs.append(f'detrender.M = {int(detrender.M)} != beams_per_batch ='
                            f' {self.nbeams}')
            if int(detrender.T) != self.nt_in:
                errs.append(f'detrender.T = {int(detrender.T)} != plan.nt_in = {self.nt_in}')
            if errs:
                raise RuntimeError('_SweepGeometry: the config is not usable by this tool:'
                                   '\n  - ' + '\n  - '.join(errs))

            self.W = int(detrender.W)
            kv = KnotVector(np.asarray(detrender.knots, dtype=np.int64),
                            int(detrender.n_phi), self.nfreq)
            self.spline_detrender = SplineDetrender(kv, n=int(detrender.n), W=self.W,
                                                    eta=float(detrender.eta),
                                                    eps=float(detrender.eps),
                                                    dtype=detrender_dtype)

        # Per-tree geometry. 'gamma' is the input time-downsampling exponent of the tree, and
        # 'ddspread' is Delta_dd from notes/variance_map.tex: the largest full-band delay
        # searched by the tree, in downsampled samples. (Per-subband lags do not add to this
        # -- the output time index is the arrival time extrapolated to the top of the tree's
        # own band.)
        self.tree_gamma = []
        self.tree_r = []
        self.tree_R = []
        self.tree_M = []
        self.tree_N = []
        self.tree_P = []
        self.tree_D = []        # 2^(r-R), the number of coarse DM rows of the subband array
        self.tree_nalpha = []
        self.tree_nt_ds = []    # peak-finding input samples per chunk (= nt_in / 2^gamma)
        self.tree_ntime = []    # input samples needed per pass, for this tree alone
        self.tree_nlo = []      # first input channel this tree can reach (see check below)

        gamma_max = max(int(t.primary_tree_index) for t in plan.trees)

        for itree in range(self.ntrees):
            t = plan.trees[itree]
            fs = t.frequency_subbands
            r, R = int(t.total_rank()), int(fs.pf_rank)
            gamma = int(t.primary_tree_index)

            # Note there is deliberately NO constraint on Dcore. Both sweeps end in a
            # PfSquare, which evaluates h_p at every time sample by construction, so nothing
            # downstream of the dedisperser sees the peak-finder's Dcore sublattice at all.
            #
            # test_multimap_vs_sweep() is what makes that a CHECKED property rather than an
            # assumed one, without needing a Dcore knob to vary: detrender_free.py reads
            # neither Dcore nor time_downsampling, so it is a Dcore-blind oracle, and every
            # config it is run against has Dcore >= 2 (Dcore = time_downsampling = 2^dd_rank1
            # for a dcore_from_cdd2_registry=False plan, and dd_rank >= 1). A sweep that depended
            # on Dcore would disagree with it. Measured over 494 drawn trees: Dcore is 2 or 4,
            # never 1.
            #
            # Nor is there a constraint on xdm_rank. The subband array has 2^(r-R) coarse DM
            # rows whatever K is; K only says how many of them the PEAK-FINDER max-reduces
            # into one output row, and neither sweep runs a peak-finder.

            wmax = int(t.pf.max_width)
            ddspread = 1 << (int(config.toplevel_tree_rank) - int(t.early_trigger_level))

            self.tree_gamma.append(gamma)
            self.tree_r.append(r)
            self.tree_R.append(R)
            self.tree_M.append(int(fs.M))
            self.tree_N.append(int(fs.N))
            self.tree_P.append(int(t.nprofiles))
            self.tree_D.append(1 << (r - R))
            self.tree_nalpha.append(self.tree_D[-1] * int(fs.M) * int(t.nprofiles))
            self.tree_nt_ds.append(int(t.nt_ds))
            # Eq. (bf_ntime) of notes/variance_map.tex, generalized to a stream shared by all
            # trees: the one-hot sits at t0 = 2W + c with 0 <= c < 2^gamma_max, and this tree
            # spreads the response over (Delta_dd + 2*Wmax) downsampled samples, i.e. 2^gamma
            # times as many input samples. The 4W covers the placement of t0 plus the
            # detrender's forward reach and trailing padding.
            self.tree_ntime.append(4*self.W + (1 << gamma_max)
                                   + (1 << gamma) * (ddspread + 2*wmax))

            # Channels entirely below the tree's lowest searched frequency, whose columns must
            # vanish identically. A detrender spreads a one-hot over every channel of its
            # spline zone, so with one configured the cut moves down to the start of the zone
            # containing that channel.
            nlo = int(np.floor(config.frequency_to_index(float(t.trigger_frequency))))
            self.tree_nlo.append(self._zone_lo(nlo) if (self.spline_detrender is not None)
                                 else nlo)

        self.gamma_max = gamma_max

        # Number of polyphase passes per input channel. With no detrender (W = 0) everything
        # upstream of the time-downsampler is instantaneous in time, so all 2^gamma phases
        # give the same answer and one pass suffices (notes/variance_map.tex; checked by
        # test_sweep_phase_collapse).
        self.nphases = (1 << gamma_max) if (self.W > 0) else 1

        # Weight applied to the sum over phases, per tree. Running 2^gamma_max phases visits
        # each residue class mod 2^gamma exactly 2^(gamma_max-gamma) times, hence the
        # reciprocal here; running a single phase instead requires the 2^gamma of the W = 0
        # special case.
        self.tree_phase_weight = [float(1 << g) / self.nphases for g in self.tree_gamma]

        # One pass occupies 'ntime' input samples, rounded up to 'ndata_chunks' whole chunks.
        # A guard chunk is appended to that, and the one-hot must land in the first chunk.
        self.ntime = max(self.tree_ntime)
        self.ndata_chunks = (self.ntime + self.nt_in - 1) // self.nt_in

        # The detrended one-hot occupies input times [t0-W, t0+W] with t0 = 2W + c, and
        # _write_one_hot() puts all of it in the first chunk of the interval.
        if 3*self.W + self.nphases > self.nt_in:
            errs.append(f'time_samples_per_chunk = {self.nt_in} is too small to hold the'
                        f' (detrended) one-hot in one chunk (3W + 2^gamma_max ='
                        f' {3*self.W + self.nphases})')

        if errs:
            raise RuntimeError('_SweepGeometry: the config is not usable by this tool:\n  - '
                               + '\n  - '.join(errs))

    def resolve_channels(self, channels):
        """The input channels to sweep, validated, as a list."""

        if channels is None:
            return list(range(self.nfreq))

        ch = [int(c) for c in channels]
        if len(ch) == 0:
            raise RuntimeError('_SweepGeometry: an empty channel selection')
        if (min(ch) < 0) or (max(ch) >= self.nfreq):
            raise RuntimeError(f'_SweepGeometry: channel selection out of range [0,'
                               f' {self.nfreq})')
        if len(set(ch)) != len(ch):
            raise RuntimeError('_SweepGeometry: duplicate channels in the selection')
        return ch

    def _zone_lo(self, ifreq):
        """The first channel of the detrender spline zone containing channel 'ifreq' (or
        nfreq if ifreq is out of range)."""

        from ..detrending_spline import zone_channel_ranges

        for (lo, hi) in zone_channel_ranges(self.spline_detrender.kv):
            if lo <= ifreq < hi:
                return lo
        return self.nfreq

    def one_hot_response(self, ifreq):
        """What a one-hot in channel 'ifreq' contributes to the dedisperser's input stream, as
        a ``(nfreq, 2W+1)`` array covering input times [t-W, t+W] relative to the one-hot's own
        time t. With no detrender that is the one-hot itself (W = 0).

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
            raise RuntimeError('_SweepGeometry: the Detrender2d dropped an ill-conditioned'
                               ' zone even with an all-ones input mask, so L is not the linear'
                               ' operator this tool assumes. Lower "eps", or use more/wider'
                               ' zones.')

        return residual[0]

    def write_one_hot(self, input_array, resp, t_abs, jchunk):
        """Write the part of a one_hot_response() placed at absolute input time 't_abs' that
        falls in chunk 'jchunk' of the stream. 'input_array' must already be zeroed."""

        lo = t_abs - self.W - jchunk*self.nt_in    # response index 0, relative to chunk start
        a = max(0, -lo)
        b = min(resp.shape[1], self.nt_in - lo)
        if b > a:
            input_array[0, :, lo+a : lo+b] = resp[:, a:b]


####################################   the two sweeps   ####################################


class _SweepBase:
    """What _CpuSweep and _GpuSweep share: the progress reporting and the timing split.

    Both expose one generator, columns(), which yields ``(ifreq, cols)`` once per pass, with
    ``cols[itree]`` the phase-weighted length-nalpha column of A for that tree. That is the
    whole interface the accumulator sees, and it is why the dense and streaming paths are one
    piece of code rather than two.
    """

    def __init__(self, geom):
        self.geom = geom
        self.seconds = 0.0     # time spent producing columns; the caller's reduce time is not
        self.npasses = 0

    def _progress(self, ipass, total, t0):
        el = time.time() - t0
        eta = el * (total - ipass) / max(ipass, 1)
        atomic_print(f'  {type(self).__name__}: pass {ipass}/{total}  {el:.0f} s'
                     f' (eta {eta:.0f} s)')


class _CpuChain:
    """A ReferenceDedisperser plus the per-tree ReferencePfSquare kernels that read its
    subband arrays -- the CPU counterpart of _GpuSweep's

        GpuSbDedispersionKernel -> sb_out -> GpuPfSquare

    tail. Both objects carry persistent state that has to line up with the same chunk stream,
    so they are constructed and driven together, through dedisperse().

    Peak-finding weights are set to 1. That does not affect anything here -- the subband
    arrays are upstream of the weights -- but it keeps out_max meaningful for a caller that
    also wants to look at it.
    """

    def __init__(self, geom):
        from ..pirate_pybind11 import ReferenceDedisperser
        from ..kernels import ReferencePfSquare

        self.geom = g = geom
        self.rdd = ReferenceDedisperser(g.plan, 1)
        assert int(self.rdd.nbatches) == 1, int(self.rdd.nbatches)

        for w in self.rdd.wt_arrays:
            w[...] = 1.0

        # Every axis except time is a spectator to a PfSquare, so the subband array's
        # (Dpf, M) pair is simply flattened into its 'ndm' row count -- exactly as _GpuSweep
        # does with sb_out.
        self.pf_squares = [
            ReferencePfSquare(int(g.plan.trees[i].pf.max_width), 1, 1,
                              g.tree_D[i] * g.tree_M[i], g.tree_nt_ds[i])
            for i in range(g.ntrees)]

    @property
    def input_array(self):
        """The dedisperser's input buffer. Fill it before each dedisperse()."""
        return self.rdd.input_array

    def dedisperse(self, ichunk):
        """Run one chunk, and return that chunk's ``sum_t y^2`` per tree, as a list of
        ``(2^(r-R), M, P)`` float64 arrays.

        Chunks must be supplied in order: both the dedisperser and the PfSquare kernels carry
        state across chunk boundaries.
        """

        g = self.geom
        self.rdd.dedisperse(ichunk, 0)
        out = []

        for itree in range(g.ntrees):
            D, M, P = g.tree_D[itree], g.tree_M[itree], g.tree_P[itree]
            acc = np.zeros((1, D*M, P))
            sb = np.asarray(self.rdd.out_sb[itree])
            self.pf_squares[itree].apply(acc, sb.reshape(1, D*M, g.tree_nt_ds[itree]), 0)
            out.append(acc[0].reshape(D, M, P))

        return out


class _CpuSweep(_SweepBase):
    """The sweep on the CPU, through a _CpuChain. The pipeline is

        one-hots -> [numpy detrender] -> ReferenceDedisperser
                 -> out_sb -> ReferencePfSquare -> float64 accumulator, one per tree

    i.e. the same shape as _GpuSweep's, which is what makes test_sweep_gpu_vs_cpu a test of
    this driver rather than of a convention that reconciles two different quantities.

    Needs ``beams_per_batch == 1``: the beam axis buys nothing here, and the reference
    dedisperser is slow enough that it is a correctness reference rather than a production
    path.
    """

    def __init__(self, geom):
        super().__init__(geom)
        if geom.nbeams != 1:
            raise RuntimeError(f'_CpuSweep: needs beams_per_batch == 1 (got {geom.nbeams});'
                               ' the beam axis buys nothing on the CPU')

    def make_chain(self):
        """A _CpuChain on this sweep's geometry."""
        return _CpuChain(self.geom)

    def run_pass(self, chain, ifreq, iphase, ipass, guard_chunk=True):
        """Apply L to the one-hot e^(F,t_c), and return ``[sum_t y^2]`` per tree, as a list of
        ``(2^(r-R), M, P)`` float64 arrays.

        Passes are laid end to end in one continuous stream (pass 'ipass' occupies chunks
        ``[ipass*nchunks, (ipass+1)*nchunks)`` of 'chain'), so no persistent state is ever
        reset: the guard chunk is what proves that one pass's response has died out before the
        next one's one-hot arrives.
        """

        g = self.geom
        nchunks = g.ndata_chunks + (1 if guard_chunk else 0)
        acc = [np.zeros((g.tree_D[i], g.tree_M[i], g.tree_P[i])) for i in range(g.ntrees)]
        t0 = 2*g.W + iphase
        resp = g.one_hot_response(ifreq)

        for j in range(nchunks):
            chain.input_array[...] = 0.0
            g.write_one_hot(chain.input_array, resp, t0, j)
            sumsq = chain.dedisperse(ipass * nchunks + j)

            for itree in range(g.ntrees):
                ov = sumsq[itree]
                if j < g.ndata_chunks:
                    acc[itree] += ov
                elif np.any(ov != 0.0):
                    raise RuntimeError(
                        f'_CpuSweep: guard chunk of pass (ifreq={ifreq}, iphase={iphase}) is'
                        f' nonzero for tree {itree} (max {float(np.abs(ov).max()):.4g}): the'
                        f' impulse response was truncated, i.e. ntime={g.ntime} is too small')

        return acc

    def columns(self, *, channels=None, guard_chunk=True, progress=False):
        g = self.geom
        chans = g.resolve_channels(channels)
        self.npasses = len(chans) * g.nphases
        chain = self.make_chain()
        report_every = max(1, self.npasses // 20)

        if progress:
            atomic_print(f'_CpuSweep: {self.npasses} passes x'
                         f' {g.ndata_chunks + (1 if guard_chunk else 0)} chunks'
                         f' (nchan={len(chans)}, nphases={g.nphases}, ntime={g.ntime},'
                         f' nt_in={g.nt_in})')

        t_wall = time.time()
        ipass = 0
        for ifreq in chans:
            for iphase in range(g.nphases):
                t0 = time.time()
                acc = self.run_pass(chain, ifreq, iphase, ipass, guard_chunk=guard_chunk)
                cols = [g.tree_phase_weight[i] * acc[i].reshape(-1) for i in range(g.ntrees)]
                self.seconds += time.time() - t0
                yield ifreq, cols

                ipass += 1
                if progress and (ipass % report_every == 0):
                    self._progress(ipass, self.npasses, t_wall)


class _GpuPipeline:
    """The GPU dedispersion + PfSquare pipeline, shared by the brute-force sweep (_GpuSweep)
    and the Monte-Carlo check (varmap/mc.py). The pipeline is

        stream_in -> [Detrender2d] -> GpuTreeGriddingKernel -> stage1_buf
                  -> [GpuLaggedDownsamplingKernel -> stage1_buf.bufs[1:]]
                  -> GpuDedispersionKernel (stage 1, one per primary tree) -> MegaRingbuf
                  -> GpuSbDedispersionKernel (stage 2 + subbands)
                  -> sb_out -> GpuPfSquare -> float64 accumulator, one per tree

    one cupy stream, kernels launched synchronously from python, no GpuDedisperser, no worker
    thread, no CudaEventRingbuf. The lagged-downsampling step runs only when
    num_primary_trees > 1; at 1 there is nothing to downsample and 'lds_kernel' is None.

    THIS CLASS KNOWS NOTHING ABOUT ITS CALLER. It owns the kernels and the buffers; the caller
    fills stream_in (or fills det_data and calls detrend_into_stream()), calls run_chunk() once
    per chunk IN ORDER, and reads or zeroes 'acc' itself. What distinguishes the two callers is
    exactly that: the sweep puts one-hots in and accumulates over an interval, the MC puts
    Gaussian noise in and zeroes 'acc' every chunk.

    Two requirements beyond _SweepGeometry's: a float32 config (GpuSbDedispersionKernel is
    float32-only), and a compiled sbdd kernel for every tree's (dd_rank, subband_counts) pair.
    The constructor builds all of them, so a missing one throws here rather than mid-run.
    """

    def __init__(self, geom):
        from ..kernels import (Detrender2d, GpuDedispersionKernel,
                               GpuLaggedDownsamplingKernel, GpuPfSquare,
                               GpuSbDedispersionKernel, GpuTreeGriddingKernel)

        self.geom = geom
        plan = geom.plan
        errs = []

        if np.dtype(plan.dtype) != np.float32:
            errs.append(f'config dtype is {np.dtype(plan.dtype)}, expected float32'
                        ' (GpuSbDedispersionKernel is float32-only)')

        rb = plan.mega_ringbuf
        if int(rb.host_global_nseg) != 0:
            errs.append(f'the MegaRingbuf has {int(rb.host_global_nseg)} host segments, so the'
                        " sweep would need host<->gpu ring-buffer copies. Raise 'max_gpu_clag'"
                        ' in the config (the default, 10000, keeps the ring buffer pure-GPU).')

        for itree in range(geom.ntrees):
            dd_rank = int(plan.stage2_dd_kernel_params[itree].dd_rank)
            if dd_rank < 3:
                errs.append(f'tree {itree} has stage-2 dd_rank = {dd_rank}, and'
                            ' GpuSbDedispersionKernel needs >= 3. A tree\'s dd_rank is'
                            ' ceil((toplevel_tree_rank - early_trigger_level)/2), so either'
                            ' raise toplevel_tree_rank or use fewer early triggers.')

        if errs:
            raise RuntimeError('_GpuPipeline: the config is not usable by this tool:\n  - '
                               + '\n  - '.join(errs))

        self.npri = int(plan.num_primary_trees)
        self.tg_kernel = GpuTreeGriddingKernel(plan.tree_gridding_kernel_params)

        # One stage-1 dedispersion per primary tree, all writing into the same MegaRingbuf,
        # plus the kernel that produces the downsampled trees' inputs. Note plan.lds_params is
        # valid (and filled) even at npri == 1 -- but its output sequence has length npri-1,
        # so there is nothing for it to do there.
        self.dd1_kernels = [GpuDedispersionKernel(plan.stage1_dd_kernel_params[ipri])
                            for ipri in range(self.npri)]
        self.lds_kernel = (GpuLaggedDownsamplingKernel(plan.lds_params)
                           if (self.npri > 1) else None)

        # nelts_per_segment is a property of the plan's ring buffer, shared by every stage-1
        # kernel; reading it from primary tree 0 alone would bake in the single-tree case.
        self.rb_nelts = int(rb.gpu_global_nseg) * int(plan.nelts_per_segment)

        # Tree channels, which is NOT the input channel count: gridding rebins nfreq input
        # channels into nchan = 2^toplevel_tree_rank tree channels.
        self.nchan = int(plan.tree_gridding_kernel_params.nchan)

        self.sb_kernels, self.pf_kernels = [], []
        for itree in range(geom.ntrees):
            p2 = plan.stage2_dd_kernel_params[itree]
            self.sb_kernels.append(
                GpuSbDedispersionKernel(p2, plan.trees[itree].frequency_subbands))
            # Every axis except time is a spectator to GpuPfSquare, so sb_out's (Dpf, M) pair
            # is simply flattened into its 'ndm' row count.
            self.pf_kernels.append(GpuPfSquare(int(plan.trees[itree].pf.max_width),
                                               geom.nbeams, geom.nbeams,
                                               geom.tree_D[itree] * geom.tree_M[itree],
                                               geom.tree_nt_ds[itree]))

        # detrender.M is checked against beams_per_batch by _SweepGeometry.
        self.gpu_detrender = Detrender2d(geom.detrender) if (geom.detrender is not None) \
            else None
        self.is_allocated = False


    def allocate(self, allocator=None):
        """Allocate the kernels' persistent state and this class's GPU buffers.

        'allocator' is a BumpAllocator with af_gpu set, or None to make a dummy-mode one (each
        allocation independent), which is what a standalone sweep wants.
        """

        import cupy as cp
        from ..core import BumpAllocator
        from ..kernels import DedispersionBuffer

        if allocator is None:
            allocator = BumpAllocator('af_gpu | af_zero', -1)

        kernels = [self.tg_kernel] + self.dd1_kernels + self.sb_kernels + self.pf_kernels
        if self.lds_kernel is not None:
            kernels.append(self.lds_kernel)
        for k in kernels:
            k.allocate(allocator)

        g, B, nt_in = self.geom, self.geom.nbeams, self.geom.nt_in
        W = g.W

        # Detrender2d reads (B, nfreq, nt_in+2W) and overwrites only [W, W+nt_in); the
        # gridding kernel wants a contiguous (B, nfreq, nt_in), so the two are separate arrays.
        self.det_data, self.det_mask = None, None
        if self.gpu_detrender is not None:
            self.det_data = cp.zeros((B, g.nfreq, nt_in + 2*W), dtype=np.float32)
            self.det_mask = cp.ones((B, g.nfreq, nt_in + 2*W), dtype=np.uint8)

        self.stream_in = cp.zeros((B, g.nfreq, nt_in), dtype=np.float32)

        # The stage-1 inputs live in ONE DedispersionBuffer rather than in per-tree cupy
        # arrays: GpuLaggedDownsamplingKernel reads bufs[0] and writes bufs[1:] with a single
        # beam stride, so they must be sub-arrays of one allocation. bufs[0] is also the
        # gridding kernel's output, so nothing else needs a 'tree_in'.
        self.stage1_buf = DedispersionBuffer(g.plan.stage1_dd_buf_params)
        self.stage1_buf.allocate(allocator)

        # Per primary tree, the (B, 2^amb_rank, 2^dd_rank, ntime) view of bufs[ipri] that the
        # stage-1 kernel wants. Reshaping the INNER TWO axes only -- the beam axis is
        # non-contiguous and must not be touched.
        self.tree_in = []
        for ipri in range(self.npri):
            p1 = g.plan.stage1_dd_kernel_params[ipri]
            b = cp.asarray(self.stage1_buf.bufs[ipri])
            assert (1 << int(p1.amb_rank + p1.dd_rank)) == b.shape[1], (ipri, b.shape)
            assert int(p1.ntime) == b.shape[2], (ipri, b.shape, int(p1.ntime))
            self.tree_in.append(b.reshape(B, 1 << int(p1.amb_rank), 1 << int(p1.dd_rank),
                                          b.shape[2]))

        assert self.tree_in[0].shape[1] * self.tree_in[0].shape[2] == self.nchan
        self.ringbuf = cp.zeros(self.rb_nelts, dtype=np.float32)

        self.sb_out, self.acc = [], []
        for itree in range(g.ntrees):
            self.sb_out.append(cp.zeros((B, g.tree_D[itree], g.tree_M[itree],
                                         g.tree_nt_ds[itree]), dtype=np.float32))
            self.acc.append(cp.zeros((B, g.tree_D[itree]*g.tree_M[itree], g.tree_P[itree]),
                                     dtype=np.float64))

        self.allocator = allocator
        self.is_allocated = True
        self._mask_checked = False   # see _make_input_stream()


    def run_chunk(self, ichunk):
        """Walk the pipeline for one chunk, from the gridding kernel to the per-tree
        accumulators. The CALLER has already filled self.stream_in.

        'ichunk' is the index in one continuous stream. Every kernel with inter-chunk state
        (the lagged downsampler, the stage-1 kernels via the MegaRingbuf, and GpuPfSquare via
        its persistent_state) is keyed on it, so chunks must be run in order and none may be
        skipped.
        """

        import cupy as cp

        g, B = self.geom, self.geom.nbeams
        sptr = cp.cuda.get_current_stream().ptr

        # The gridding kernel's output is (B, nchan, ntime), which reshapes to the
        # (B, 2^amb_rank, 2^dd_rank, ntime) that stage-1 dedispersion wants. Note nchan is the
        # TREE channel count, which need not equal the input channel count nfreq.
        self.tg_kernel.launch(self.tree_in[0].reshape(B, self.nchan, g.nt_in), self.stream_in,
                              sptr)

        # Fill the downsampled trees' inputs from primary tree 0's, in place.
        if self.lds_kernel is not None:
            self.lds_kernel.launch(self.stage1_buf, ichunk, 0, sptr)

        for ipri in range(self.npri):
            self.dd1_kernels[ipri].launch(self.tree_in[ipri], self.ringbuf, ichunk, 0, sptr)

        for itree in range(g.ntrees):
            self.sb_kernels[itree].launch(self.sb_out[itree], self.ringbuf, ichunk, 0, sptr)
            self.pf_kernels[itree].launch(
                self.acc[itree],
                self.sb_out[itree].reshape(B, g.tree_D[itree]*g.tree_M[itree],
                                           g.tree_nt_ds[itree]),
                0, sptr)


    def detrend_into_stream(self):
        """Run the Detrender2d over self.det_data and copy its emitted window to stream_in.

        The CALLER has already filled det_data, shape (B, nfreq, nt_in + 2W). The detrender
        overwrites only [W, W+nt_in), which is what lands in stream_in. Callers that do not
        use a detrender fill stream_in directly and never call this.
        """

        g = self.geom
        self.det_mask.fill(1)
        self.gpu_detrender.launch(self.det_data, self.det_mask)

        # Checked once, not once per launch: the Detrender2d's mask expansion is driven by
        # r_min, which depends on the mask and the basis but not on the data, so an all-ones
        # input mask either survives every time or never. If it did not survive, L would not
        # be the linear operator the variance map assumes.
        if not self._mask_checked:
            if not bool((self.det_mask[:, :, g.W : g.W+g.nt_in] != 0).all()):
                raise RuntimeError('_GpuPipeline: the Detrender2d dropped an ill-conditioned'
                                   ' zone even with an all-ones input mask, so L is not the'
                                   ' linear operator the variance map assumes.')
            self._mask_checked = True

        self.stream_in[...] = self.det_data[:, :, g.W : g.W+g.nt_in]


class _GpuSweep(_SweepBase):
    """The brute-force sweep on the GPU: push one-hots through a _GpuPipeline, one launch
    group at a time, and read the per-tree accumulators once per interval.

    Everything about the pipeline itself -- the kernels, the buffers, the config checks, the
    per-chunk launches -- lives in _GpuPipeline, which the Monte-Carlo check
    (varmap/mc.py) shares. What is here is sweep-specific: the one-hot input, the
    pass/launch-group bookkeeping, and the interval accumulation.
    """

    def __init__(self, geom):
        super().__init__(geom)
        self.pipe = _GpuPipeline(geom)

    def allocate(self, allocator=None):
        """Allocate the pipeline's kernels and buffers; see _GpuPipeline.allocate()."""
        self.pipe.allocate(allocator)

    def columns(self, *, channels=None, guard_chunk=True, progress=False):
        """Passes are laid end to end in a single continuous stream: pass k occupies input
        samples ``[k*nchunks*nt_in, (k+1)*nchunks*nt_in)``, which is long enough that pass k's
        response has died out before pass k+1's one-hot arrives. No kernel's persistent state,
        and no part of the ring buffer, is ever reset -- only the small GpuPfSquare
        accumulator, at each interval boundary.
        """

        import cupy as cp

        if not self.pipe.is_allocated:
            self.pipe.allocate()

        g = self.geom
        chans = g.resolve_channels(channels)
        nchunks = g.ndata_chunks + (1 if guard_chunk else 0)

        # (ifreq, iphase) pairs, in launch-sized groups of nbeams. The last group may be
        # short; its unused beams are simply left with zero input, which costs a little work
        # and produces zeros rather than needing any special case.
        passes = [(f, c) for f in chans for c in range(g.nphases)]
        groups = [passes[i : i+g.nbeams] for i in range(0, len(passes), g.nbeams)]
        self.npasses = len(passes)

        if progress:
            atomic_print(f'_GpuSweep: {len(passes)} passes in {len(groups)} launches x'
                         f' {nchunks} chunks (nchan={len(chans)}, nphases={g.nphases},'
                         f' nbeams={g.nbeams}, ntime={g.ntime}, nt_in={g.nt_in})')

        report_every = max(1, len(groups) // 20)
        stream = cp.cuda.get_current_stream()
        t_wall = time.time()

        for (igroup, group) in enumerate(groups):
            t0 = time.time()
            for itree in range(g.ntrees):
                self.pipe.acc[itree].fill(0.0)

            for j in range(nchunks):
                self._run_chunk(group, igroup*nchunks + j, j)

            stream.synchronize()

            # (B, Dpf*M, P) -> (B, Dpf, M, P); beam b carried pass group[b].
            a = [cp.asnumpy(self.pipe.acc[i]).reshape(g.nbeams, g.tree_D[i], g.tree_M[i],
                                                 g.tree_P[i])
                 for i in range(g.ntrees)]
            self.seconds += time.time() - t0

            for (b, (ifreq, _)) in enumerate(group):
                yield ifreq, [g.tree_phase_weight[i] * a[i][b].reshape(-1)
                              for i in range(g.ntrees)]

            if progress and ((igroup+1) % report_every == 0):
                self._progress((igroup+1) * g.nbeams, len(passes), t_wall)

    def _run_chunk(self, group, ichunk, j):
        """Run one chunk of one launch group: fill the input stream, then walk the pipeline.
        'j' is the chunk's index within the interval (only chunk 0 carries a one-hot)."""

        self._make_input_stream(group, j)
        self.pipe.run_chunk(ichunk)

    def _make_input_stream(self, group, j):
        """Fill the pipeline's stream_in (the gridding kernel's input) with the one-hots of this
        launch group, detrending them if a Detrender2d is configured.

        Only chunk 0 of an interval carries anything: L is linear, so the all-zero chunks that
        follow map to zero, and running the detrender on them would be pure cost. The
        one-hot's detrended response spans [t0-W, t0+W] with t0 = 2W + c, so it lies strictly
        inside chunk 0's emitted region [W, W+nt_in) of the detrender buffer (checked in the
        geometry).
        """

        g, pipe = self.geom, self.pipe

        if j != 0:
            pipe.stream_in.fill(0.0)
            return

        if pipe.gpu_detrender is None:
            pipe.stream_in.fill(0.0)
            for (b, (ifreq, iphase)) in enumerate(group):
                pipe.stream_in[b, ifreq, 2*g.W + iphase] = 1.0
            return

        pipe.det_data.fill(0.0)
        for (b, (ifreq, iphase)) in enumerate(group):
            pipe.det_data[b, ifreq, g.W + (2*g.W + iphase)] = 1.0   # buffer index = W + t0

        pipe.detrend_into_stream()


####################################   accumulation   ####################################


def _alloc(shape, scratch_dir, tag):
    """The sweep's ONE allocation call for an array of matrix size. Returns (array, path).

    RAM by default: the machines this runs on have 1.5 TiB against a 344 GiB map at CHORD's
    finest legal grouping, and coarse_grain() already assumes as much, allocating its output
    with a plain np.full(). 'scratch_dir' switches to an on-disk np.memmap, which is the
    fallback for a config that does not fit -- and keeping both behind this one call is what
    makes that a one-line difference rather than a second code path.

    mode='w+' creates a sparse zero-filled file, so the memmap starts at zero as the in-RAM
    branch does.
    """

    if scratch_dir is None:
        return np.zeros(shape, dtype=np.float64), None

    path = os.path.join(scratch_dir, f'varmap_sweep_{tag}_{os.getpid()}.dat')
    return np.memmap(path, dtype=np.float64, mode='w+', shape=shape), path


def _flush_stage(stage, out, sel, n, nrows, *, tile_rows=4096):
    """``out[:, sel] = stage[:n].T``, TILED OVER ROWS.

    The tiling is not an optimization to skip. Without it the assignment walks all `nrows`
    output rows at once, and they are `nfreq*8` bytes apart (220 KiB at CHORD), so the live
    page set is the whole matrix and every element costs a TLB miss. A tile bounds it to
    `tile_rows` pages instead. Measured 2.1x on the equivalent whole-matrix transpose.

    'sel' is a slice when the staged channels are contiguous, which is every full sweep, and
    an index array only for a scattered channel subset -- where the volume is tiny anyway.
    """

    for r0 in range(0, nrows, tile_rows):
        r1 = min(r0 + tile_rows, nrows)
        out[r0:r1, sel] = stage[:n, r0:r1].T


class _Accumulator:
    """Turns the stream of columns into one VarianceMap per tree.

    THE OUTPUT IS THE ONLY ARRAY OF MATRIX SIZE THIS HOLDS, and that is the point. Each column
    is reduced into a small ``(NSTAGE, nrows)`` staging buffer as it arrives, and a full buffer
    is transposed into the output; the alternative -- accumulate a whole ``(nfreq, nrows)``
    transpose and turn it round at the end -- needs both live at once, which at CHORD's L = 4
    is 688 GiB rather than 344. Same total work, spread across the sweep instead of landing
    after it -- which is a memory win, not a time one: nothing here runs concurrently.

    Staging at all is what makes the write pattern affordable: the map is produced one COLUMN
    at a time but stored row-major, so writing each column as it arrives would touch one cache
    line per output row, on an array of matrix size.

    'nrows' is nalpha for a tree whose L is None and nbeta otherwise, and that one difference
    is the whole of the dense-versus-streaming split.
    """

    # Columns held before a flush. MEASURED TO BE IMMATERIAL to speed, which is not what one
    # would guess: it sets the length of the contiguous run each output row receives
    # (NSTAGE*8 bytes), but sweeping it over 8..512 on an 85.9 GiB output moved the flush rate
    # only between 0.73 and 0.96 GiB/s, with no trend and less spread than the run-to-run
    # noise. EVERY flush spans the whole output address range whatever its width -- a thin
    # column slice still touches all nrows rows -- so the pass is bandwidth-bound on scattered
    # writes, and widening it trades fewer passes for proportionally more bytes each.
    #
    # So this is chosen for the BUFFER it costs, NSTAGE * nrows * 8: 0.8 GiB against a 344 GiB
    # output at CHORD's L = 4, and 229 KiB on a test map.
    #
    # What staging as such costs, measured end to end at CHORD geometry: 61 s against 34 s for
    # a single tiled transpose of a whole (nfreq, nrows) accumulator, per 85.9 GiB. The
    # accumulator wins that pass because it can write each output row to completion before
    # moving on; it loses the sweep, because it is a second array of matrix size. Trading 27 s
    # per 86 GiB -- under 1% of a sweep that is hours of GPU -- for not doubling the peak is
    # not a close call.
    _NSTAGE = 64

    def __init__(self, geom, Ls, scratch_dir=None):
        self.geom = geom
        self.Ls = list(Ls)
        self.scratch_dir = scratch_dir
        self.seconds = 0.0
        self.transpose_seconds = 0.0
        self.gmin, self.gmax = np.inf, -np.inf

        self.A, self.stage, self.y, self.nrows = [], [], [], []
        self.staged = []       # input channels currently held in the buffers, in order

        # The plan the returned maps index through, which is NOT geom.plan: that one drives
        # GPU kernels, so it holds a MegaRingbuf's pinned host memory, which a map handed back
        # to a caller must not keep alive. Same geometry either way -- one constructor.
        self.plan = plan = make_plan(geom.config)
        self.trees = plan.trees   # one list build; plan.trees copies on every access

        for itree in range(geom.ntrees):
            tree = self.trees[itree]
            fs = tree.frequency_subbands
            r, R = int(tree.total_rank()), int(fs.pf_rank)

            L = self.Ls[itree]
            if L is None:
                nrows = geom.tree_nalpha[itree]
            else:
                # coarse_grain_vector() checks this too, but it is a CALLER error and a
                # sweep is hours long, so it is worth catching before the loop rather than
                # on the first column -- by which point __init__ has already allocated an
                # accumulator per tree, and possibly scratch files.
                if not (R <= L <= r):
                    raise RuntimeError(f'_Accumulator: tree {itree} was given L={L}, which is'
                                       f' out of range [R, r] = [{R}, {r}]')
                nrows = (1 << (r - L)) * int(fs.N) * int(tree.nprofiles)
                # Called for its side effect: coarse_grain_vector() assumes this tree's
                # multiplet ordering once per swept column, and this is where that assumption
                # can still be reported cheaply.
                _subband_tables(tree)
            self.nrows.append(nrows)

            # Zeroed, not -inf as coarse_grain() uses: A is a sum of squares, so the
            # max-reduction never sees a negative, and every group is occupied for
            # R <= L <= r. add() enforces the first of those on every column. An unswept
            # column of a PARTIAL sweep is therefore left at zero rather than at -inf, which
            # is what makes such a map merely incomplete instead of unreadable.
            A, _ = _alloc((nrows, geom.nfreq), scratch_dir, f'A{itree}')
            self.A.append(A)
            self.stage.append(np.zeros((self._NSTAGE, nrows)))
            self.y.append(np.zeros(geom.tree_nalpha[itree]))

    def add(self, ifreq, cols):
        """Fold one input channel's columns into the staging buffers, flushing when full.

        Channels must arrive in NON-DECREASING order, which both sweeps do: a channel's
        staging row is closed as soon as the next one opens, so revisiting it later would
        overwrite what was already flushed. With nphases > 1 the same channel arrives several
        times in a row and its passes accumulate into the row it already owns.

        The structural checks live here rather than in a pass over the finished matrix, which
        is what makes them affordable when the finished matrix is 344 GiB: the column is
        already in hand. A >= 0 and finite is a theorem (A is a sum of squares), and the
        columns of channels a tree cannot reach must vanish identically.
        """

        # The flush is timed separately, so its share is taken back out below rather than
        # counted in both places.
        t0, t_flush = time.time(), self.transpose_seconds

        fresh = (not self.staged) or (ifreq != self.staged[-1])
        if fresh:
            if self.staged and (ifreq < self.staged[-1]):
                raise RuntimeError(f'_Accumulator: input channel {ifreq} arrived after'
                                   f' {self.staged[-1]}; the sweep must yield channels in'
                                   ' non-decreasing order')
            if len(self.staged) == self._NSTAGE:
                self._flush()
            self.staged.append(ifreq)
        k = len(self.staged) - 1

        for (itree, col) in enumerate(cols):
            cmin, cmax = float(col.min()), float(col.max())

            # NaN fails both comparisons, so this catches non-finite entries too.
            if not (cmin >= 0.0) or not np.isfinite(cmax):
                raise RuntimeError(
                    f'_Accumulator: tree {itree}, input channel {ifreq}: the column of A has'
                    f' min {cmin:.4g} and max {cmax:.4g}. A is a sum of squares, so it must be'
                    ' finite and nonnegative; this should be impossible.')
            if (ifreq < self.geom.tree_nlo[itree]) and (cmax != 0.0):
                raise RuntimeError(
                    f'_Accumulator: tree {itree} does not search down to input channel'
                    f' {ifreq} (its lowest is {self.geom.tree_nlo[itree]}), but that column is'
                    f' not identically zero (max {cmax:.4g})')

            self.gmin, self.gmax = min(self.gmin, cmin), max(self.gmax, cmax)
            self.y[itree] += col

            L = self.Ls[itree]
            val = col if (L is None) else coarse_grain_vector(self.trees[itree], col, L)
            row = self.stage[itree][k]
            if fresh:
                # ASSIGN, not accumulate: the buffer is reused across flushes, so a fresh row
                # still holds the previous block's channel.
                row[:] = val
            elif L is None:
                row += val
            else:
                np.maximum(row, val, out=row)

        self.seconds += (time.time() - t0) - (self.transpose_seconds - t_flush)

    def _flush(self):
        """Write the staged columns into the output and empty the buffers."""

        if not self.staged:
            return

        t0 = time.time()
        lo, hi, n = self.staged[0], self.staged[-1], len(self.staged)
        sel = slice(lo, hi + 1) if (hi - lo + 1 == n) else np.asarray(self.staged)
        for itree in range(self.geom.ntrees):
            _flush_stage(self.stage[itree], self.A[itree], sel, n,
                         self.nrows[itree])
        self.staged = []
        self.transpose_seconds += time.time() - t0

    def finish(self, *, device, guard_chunk, sweep_seconds, progress=False, partial=False):
        """Flush what is left and wrap each matrix in a VarianceMap. Returns the length-ntrees
        list.

        y_true is the streaming per-channel sum, in BOTH the dense and the coarse case. It is
        the same quantity a dense map's row_sums() computes, in a different summation order,
        and having one code path is what lets a dense sweep and a coarse one be compared to
        the last bit.

        A PARTIAL sweep (a channel subset) instead returns maps with ``y_true = None``: y_true
        would then be a sum over the swept channels only, which is a silent lie rather than a
        missing value, and dropping it is what stops anything downstream from scoring the map.
        """

        self._flush()
        g = self.geom
        maps = []

        for itree in range(g.ntrees):
            A = self.A[itree]
            self.stage[itree] = None
            if isinstance(A, np.memmap):
                A.flush()      # so the file on disk matches the mapping, not just the pages

            L = self.Ls[itree]
            rec = dict(step='brute_force', device=device, L=L, nphases=g.nphases,
                       nbeta=self.nrows[itree], guard_chunk=bool(guard_chunk),
                       partial=bool(partial), sweep_seconds=float(sweep_seconds),
                       transpose_seconds=self.transpose_seconds)
            m = VarianceMap.from_dense(g.config, itree, A, detrender=g.detrender,
                                       y_true=(None if partial else self.y[itree]),
                                       L=L, plan=self.plan, history=[rec])

            # The one runtime guard on the property the whole scalable path assumes -- that
            # the reference does not UNDERESTIMATE A_true. Called wherever a coarse map is
            # BUILT, once, because nothing downstream can detect a failure: a map that
            # dominates a too-small reference is admissible against it and wrong against the
            # truth.
            if (L is not None) and (m.y_true is not None):
                ratio = m.check_ref_covers_y_true()
                if progress:
                    atomic_print(f'  tree {itree}: check_ref_covers_y_true() passed, worst'
                                 f' ratio {ratio:.6g}')

            maps.append(m)

        return maps
