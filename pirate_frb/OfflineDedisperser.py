"""Single-beam offline GPU tree-dedisperser (class OfflineDedisperser)."""

from contextlib import contextmanager

import numpy as np

import ksgpu
from .pirate_pybind11 import DedispersionPlan, GpuDedisperser
from .core import BumpAllocator, CudaStreamPool
from .kernels import GpuDequantizationKernel


class OfflineDedisperser:
    """Single-beam GPU tree-dedisperser, driven one AssembledFrame at a time.

    Constructed from a DedispersionConfig (which is coerced to a single-beam
    geometry: beams_per_gpu = beams_per_batch = num_active_batches = 1).

    Construction of the DedispersionPlan and GpuDedisperser is DEFERRED until the
    first dedisperse(): the frequency-zone structure (zone_nfreq / zone_freq_edges),
    the time sampling (time_sample_ms), and the per-channel noise variances are
    taken from the first frame's XEngineMetadata (the data on disk is
    authoritative). The analytic peak-finding weights are then filled from those
    variances, so out_max comes out as an SNR.

    dedisperse(frame) is a context manager. On entry it validates the frame:
      - frame.metadata is present, and its beam_ids == [frame.beam_id];
      - frames are the same beam, and consecutive in time_chunk_index (the
        dedisperser is a streaming engine, so a gap/reorder would splice
        non-adjacent chunks);
      - every subsequent frame's metadata matches the first frame's:
        zone_nfreq, zone_freq_edges, dt_ns_per_seq, seq_per_frb_time_sample,
        noise_variance, unix_ns_at_seq_0, beamset.

    It then dequantizes the frame on the GPU, feeds it to the dedisperser, and
    yields that chunk's GpuDedisperserOutputs (per-tree out_max / out_argmax GPU
    arrays). Those arrays are views into a small output ring buffer, valid ONLY
    inside the 'with' block -- consume them (e.g. host-copy the peak) before the
    block exits. Frames must be supplied in increasing time-chunk order.

    Example::

        od = OfflineDedisperser(config)
        for frame in frames:                       # in increasing time-chunk order
            with od.dedisperse(frame) as outputs:
                snr = max(float(cp.asarray(outputs.out_max[t]).max())
                          for t in range(od.ntrees))
                print(snr)

    Attributes nfreq / nt_in / ntrees / trees / plan / dd are None until the first
    dedisperse() has initialized the pipeline (dtype is known immediately from the
    config).
    """

    def __init__(self, config, cuda_device_id=0):
        # Work on a copy: coercing to single-beam geometry (and overwriting the
        # zone/timing fields from metadata, in _initialize) must not mutate the
        # caller's config. num_active_batches = 1 => one CUDA compute stream, one
        # beam-batch per chunk, seq_id == chunk index.
        config = config.clone()
        config.beams_per_gpu = 1
        config.beams_per_batch = 1
        config.num_active_batches = 1

        self.config = config
        self.cuda_device_id = cuda_device_id
        self.dtype = config.dtype   # not metadata-dependent; known immediately

        # Pipeline objects: built lazily on the first run() (see _initialize).
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

        # Per-frame invariants captured from the first frame (see run()).
        self._beam_id = None        # all frames must have this beam_id
        self._initial_tci = None    # time_chunk_index of the first frame

        self._seq_id = 0            # next chunk's seq_id (== chunk index)

    def _initialize(self, md):
        """Build the plan + dedisperser from the first frame's XEngineMetadata 'md'.

        The metadata is authoritative for the frequency-zone structure and time
        sampling, so those config fields are overwritten before the plan is built.
        """
        config = self.config
        config.zone_nfreq = list(md.zone_nfreq)
        config.zone_freq_edges = list(md.zone_freq_edges)
        # FRB time-sample length (ms) implied by the metadata's seq timing.
        config.time_sample_ms = md.dt_ns_per_seq * md.seq_per_frb_time_sample / 1.0e6
        config.validate()

        self.plan = DedispersionPlan(config)
        self.nfreq = self.plan.nfreq
        self.nt_in = self.plan.nt_in
        self.ntrees = self.plan.ntrees
        self.trees = self.plan.trees

        self.stream_pool = CudaStreamPool(config.num_active_batches)
        self.dd = GpuDedisperser(
            self.plan, self.stream_pool,
            cuda_device_id=self.cuda_device_id,
            num_consumers=1,
            nbatches_out=config.num_active_batches,   # 1: strictly sequential
            nbatches_wt=config.num_active_batches,    # 1: one weight slot
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
        self.time_sample_ms = config.time_sample_ms
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
        """Dedisperse one time chunk; yield its GpuDedisperserOutputs.

        A context manager. The first entry initializes the pipeline from
        frame.metadata; later entries check their metadata is consistent. The
        yielded arrays are views into the dedisperser's output ring buffer and are
        valid ONLY inside the 'with' block -- consume them before it exits. See the
        class docstring.
        """
        import cupy as cp

        md = frame.metadata
        assert md is not None, "OfflineDedisperser: frame.metadata is None"
        # A single-beam ASDF frame projects its metadata to a length-1 beam list.
        assert list(md.beam_ids) == [frame.beam_id], \
            (f"OfflineDedisperser: frame.metadata.beam_ids {list(md.beam_ids)} "
             f"!= [frame.beam_id] ([{frame.beam_id}])")

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
