"""Single-beam offline GPU tree-dedisperser (class OfflineDedisperser)."""

from contextlib import contextmanager

import numpy as np

import ksgpu
from .pirate_pybind11 import DedispersionPlan, GpuDedisperser
from .core import BumpAllocator, CudaStreamPool
from .kernels import GpuDequantizationKernel


class OfflineDedisperser:
    """Single-beam GPU tree-dedisperser, driven one AssembledFrame at a time.

    Eventually, we'd like the entire real-time pipeline to be runnable in
    an offline mode (for visualization, debugging, or developing RFI-flagging
    logic). The OfflineDedisperser is a work in progress that will be expanded
    later. Currently, it just processes a single beam, and is "driven" by an
    externally-provided sequence of AssembledFrames (e.g. from class Acquisition).

    Usage is simple, but note warnings and stream semantics below::

        import cupy as cp

        # 'config' has type DedispersionConfig
        od = OfflineDedisperser(config)

        # loop over time chunks in single beam
        for frame in frames:
            # 'outputs' has type GpuDedisperserOutputs.
            # WARNING: don't use output arrays outside their context manager, or
            # you'll get a silent race condition!! (The output arrays are views
            # into a shared GPU memory ring buffer, which is reused soon after
            # context manager exit.)
            with od.dedisperse(frame) as outputs:
                snr = max(float(cp.asarray(outputs.out_max[t]).max())
                          for t in range(od.ntrees))
                print(f'max snr in chunk = {snr}')
    
    The first call to dedisperse() does a lot of initialization, including
    initializing attributes config / nfreq / nt_in / ntrees / trees / plan / dd.
    """

    def __init__(self, config, cuda_device_id=0):
        self.original_config = config
        self.config = None
        self.cuda_device_id = cuda_device_id
        self.dtype = config.dtype   # not metadata-dependent; known immediately

        # Pipeline objects: built lazily on the first dedisperse() (see _initialize).
        self.plan = None
        self.dd = None
        self.dqk = None
        self.stream_pool = None
        self.nfreq = None
        self.nt_in = None
        self.ntrees = None
        self.trees = None

        # Metadata captured from the first frame + checked against later frames.
        self.zone_nfreq = None
        self.zone_freq_edges = None
        self.dt_ns_per_seq = None
        self.seq_per_frb_time_sample = None
        self.time_sample_ms = None
        self.noise_variance = None
        self.unix_ns_at_seq_0 = None
        self.beamset = None

        # Per-frame invariants captured from the first frame (see dedisperse()).
        self._beam_id = None        # all frames must have this beam_id
        self._initial_tci = None    # time_chunk_index of the first frame

        self._seq_id = 0            # next chunk's seq_id (== chunk index)

        # Set (to the exception) when a dedisperse() call raises; all later
        # dedisperse() calls then fail fast -- see the latch in dedisperse().
        self._failed = None

    def _initialize(self, md):
        """Build the plan + dedisperser from the first frame's XEngineMetadata 'md'.

        The metadata is authoritative for the frequency-zone structure and time
        sampling, so those config fields are overwritten before the plan is built.
        """
        self.config = self.original_config.clone()
        self.config.beams_per_gpu = 1
        self.config.beams_per_batch = 1
        self.config.num_active_batches = 1
        self.config.zone_nfreq = list(md.zone_nfreq)
        self.config.zone_freq_edges = list(md.zone_freq_edges)
        # FRB time-sample length (ms) implied by the metadata's seq timing.
        self.config.time_sample_ms = md.dt_ns_per_seq * md.seq_per_frb_time_sample / 1.0e6
        self.config.validate()

        self.plan = DedispersionPlan(self.config)
        self.nfreq = self.plan.nfreq
        self.nt_in = self.plan.nt_in
        self.ntrees = self.plan.ntrees
        self.trees = self.plan.trees

        self.stream_pool = CudaStreamPool(self.config.num_active_batches)
        self.dd = GpuDedisperser(
            self.plan, self.stream_pool,
            cuda_device_id=self.cuda_device_id,
            num_consumers=1,
            nbatches_out=self.config.num_active_batches,   # 1: strictly sequential
            nbatches_wt=self.config.num_active_batches,    # 1: one weight slot
            initial_chunk=0,
        )
        # Dummy-mode allocators (capacity < 0): each internal array gets its own
        # allocation. Simplest correct choice for an offline one-shot run.
        self._gpu_alloc = BumpAllocator(ksgpu.af_gpu | ksgpu.af_zero, -1)
        self._host_alloc = BumpAllocator(ksgpu.af_rhost | ksgpu.af_zero, -1)
        self.dd.allocate(self._gpu_alloc, self._host_alloc)

        # int4 -> float16 dequantizer (single beam).
        self.dqk = GpuDequantizationKernel(self.dtype, 1, self.nfreq, self.nt_in)

        # Analytic peak-finding weights from the per-channel noise variances, so
        # out_max comes out as an SNR.
        fv = np.ascontiguousarray(md.get_channel_variances(), dtype=np.float64)
        assert fv.shape == (self.nfreq,), (fv.shape, self.nfreq)
        self.dd.fill_analytic_weights(fv)

        # Capture the metadata that later frames must match.
        self.zone_nfreq = list(md.zone_nfreq)
        self.zone_freq_edges = list(md.zone_freq_edges)
        self.dt_ns_per_seq = md.dt_ns_per_seq
        self.seq_per_frb_time_sample = md.seq_per_frb_time_sample
        self.time_sample_ms = self.config.time_sample_ms
        self.noise_variance = list(md.noise_variance)
        self.unix_ns_at_seq_0 = md.unix_ns_at_seq_0
        self.beamset = md.beamset

    def _check_metadata(self, md):
        """Check a subsequent frame's metadata against the first frame's."""
        assert list(md.zone_nfreq) == self.zone_nfreq, \
            (list(md.zone_nfreq), self.zone_nfreq)
        assert list(md.zone_freq_edges) == self.zone_freq_edges, \
            (list(md.zone_freq_edges), self.zone_freq_edges)
        # dt_ns_per_seq + seq_per_frb_time_sample determine time_sample_ms; checking
        # the two integer fields is stricter than checking their product.
        assert md.dt_ns_per_seq == self.dt_ns_per_seq, \
            (md.dt_ns_per_seq, self.dt_ns_per_seq)
        assert md.seq_per_frb_time_sample == self.seq_per_frb_time_sample, \
            (md.seq_per_frb_time_sample, self.seq_per_frb_time_sample)
        assert list(md.noise_variance) == self.noise_variance, \
            (list(md.noise_variance), self.noise_variance)
        # Same acquisition + beamset for the whole stream.
        assert md.unix_ns_at_seq_0 == self.unix_ns_at_seq_0, \
            (md.unix_ns_at_seq_0, self.unix_ns_at_seq_0)
        assert md.beamset == self.beamset, (md.beamset, self.beamset)

    @contextmanager
    def dedisperse(self, frame):
        """Dedisperse one time chunk; yield its GpuDedisperserOutputs as a context manager.
        
        WARNING: don't use output arrays outside their context manager, or
        you'll get a silent race condition!! (The output arrays are views
        into a shared GPU memory ring buffer, which is reused soon after
        context manager exit.)

        Stream semantics: dedisperse() issues all GPU work (upload, dequantization,
        dedispersion kernels, output acquire/release) on the cupy stream that is
        current when it is called, and the yielded outputs are tied to that stream.

        With "normal" cupy usage you shouldn't need to worry about race conditions,
        but if you use multiple streams or asynchronous copies, then you may need to
        synchronize. For example, cupy's arr.get() / cp.asnumpy() are fine, but if
        you call arr.get(blocking=False) then you must synchronize the stream before
        exiting the context manager (or reading the numpy array).

        The first call to dedisperse() does a lot of initialization, including
        initializing attributes config / nfreq / nt_in / ntrees / trees / plan / dd.

        If a dedisperse() call raises -- from the GPU pipeline, from the caller's
        with-block, or from a frame-sequencing assert -- the OfflineDedisperser
        must be DISCARDED: internal state (including the seq_id cursors shared
        with the C++ GpuDedisperser) may be out of sync, so every later call
        raises RuntimeError pointing back at the original error. Construct a
        fresh OfflineDedisperser to continue.
        """
        import cupy as cp

        # A previous dedisperse() call raised: internal state (the python/C++
        # seq_id cursors, or a partially-built pipeline) may be inconsistent, so
        # fail fast with the root cause instead of tripping a confusing
        # cursor-mismatch assert deeper in (see the except-latch below).
        if self._failed is not None:
            raise RuntimeError(
                "OfflineDedisperser: a previous dedisperse() call raised "
                f"({self._failed!r}); internal state may be out of sync -- "
                "construct a fresh OfflineDedisperser") from self._failed

        md = frame.metadata
        assert md is not None, "OfflineDedisperser: frame.metadata is None"
        # A single-beam ASDF frame projects its metadata to a length-1 beam list.
        assert list(md.beam_ids) == [frame.beam_id], \
            (f"OfflineDedisperser: frame.metadata.beam_ids {list(md.beam_ids)} "
             f"!= [frame.beam_id] ([{frame.beam_id}])")

        try:
            if self.dd is None:
                self._beam_id = frame.beam_id
                self._initial_tci = frame.time_chunk_index
                self._initialize(md)
            else:
                # Consecutive frames must be the same beam and adjacent in time (the
                # dedisperser is a streaming engine, so a gap or reordering would splice
                # non-adjacent chunks). Independent of how the frames were enumerated.
                assert frame.beam_id == self._beam_id, \
                    f"OfflineDedisperser: beam_id changed mid-stream ({frame.beam_id} != {self._beam_id})"
                expected_tci = self._initial_tci + self._seq_id
                assert frame.time_chunk_index == expected_tci, \
                    (f"OfflineDedisperser: non-consecutive time_chunk_index "
                     f"(got {frame.time_chunk_index}, expected {expected_tci})")
                self._check_metadata(md)

            assert frame.nfreq == self.nfreq, (frame.nfreq, self.nfreq)
            assert frame.ntime == self.nt_in, (frame.ntime, self.nt_in)

            seq = self._seq_id
            stream = cp.cuda.get_current_stream()

            # Upload this chunk's quantized data to the GPU (add a length-1 beam axis)
            # and dequantize it directly into the dedisperser's input buffer. Leaving
            # the get_input() block launches the dedispersion kernels for this seq_id.
            with self.dd.get_input(seq, stream=stream) as arr:               # (1, nfreq, nt_in), dtype
                data_gpu = cp.asarray(np.asarray(frame.data))[None]          # (1, nfreq, nt_in/2) uint8
                scoff_gpu = cp.asarray(np.asarray(frame.scales_offsets))[None]  # (1, nfreq, mpc, 2) float16
                self.dqk.launch(arr, scoff_gpu, data_gpu, stream=stream)

            # Input committed; advance the seq counter now (before yielding) so that a
            # caller exception inside the output block cannot desync us from the
            # dedisperser's internal counters, which already advanced above.
            self._seq_id += 1

            # get_output() acquires this chunk's output on entry and releases it on
            # exit. release_output is stream-ordered, so no explicit host sync is
            # needed: the caller consumes 'outputs' on the same stream inside the block.
            with self.dd.get_output(seq, stream=stream) as outputs:
                yield outputs
        except BaseException as e:
            # Anything that raises in here can leave internal state out of
            # lockstep: an exception inside the get_input block advances the C++
            # input cursor (the context manager's finally still releases and
            # launches) but skips the _seq_id += 1; a partial _initialize()
            # leaves self.dd built but not allocated/weighted; a sequencing
            # assert means the frame stream itself is broken. Every such failure
            # is loud, but a RETRY would trip a confusing C++ cursor xassert
            # (which also stops the GpuDedisperser) -- so latch the failure and
            # make later calls fail fast with a clear message instead (see the
            # _failed check above).
            self._failed = e
            raise
