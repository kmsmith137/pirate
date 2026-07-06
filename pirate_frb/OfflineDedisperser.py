"""Single-beam offline GPU tree-dedisperser (class OfflineDedisperser)."""

import numpy as np

import ksgpu
from .pirate_pybind11 import DedispersionPlan, GpuDedisperser
from .core import BumpAllocator, CudaStreamPool
from .kernels import GpuDequantizationKernel


class OfflineDedisperser:
    """Single-beam GPU tree-dedisperser driven one AssembledFrame at a time.

    Constructed from a DedispersionConfig (which is coerced to a single-beam
    geometry: beams_per_gpu = beams_per_batch = num_active_batches = 1). The
    peak-finding weights should be filled once, via fill_analytic_weights(),
    before the first run().

    run(frame) dequantizes the frame on the GPU, feeds it to the dedisperser, and
    returns the GpuDedisperserOutputs for that chunk (per-tree out_max / out_argmax
    GPU arrays). Those arrays are views into a small output ring buffer and are only
    valid until the NEXT run() (or close()) -- so the caller must consume them (e.g.
    host-copy the peak) before calling run() again. Frames must be supplied in
    increasing time-chunk order (the dedisperser is a streaming engine).
    """

    def __init__(self, config, cuda_device_id=0):
        # Work on a copy: coercing to single-beam geometry below must not mutate
        # the caller's config. num_active_batches = 1 => one CUDA compute stream,
        # one beam-batch per chunk, seq_id == chunk index.
        config = config.clone()
        config.beams_per_gpu = 1
        config.beams_per_batch = 1
        config.num_active_batches = 1
        config.validate()

        self.config = config
        self.dtype = config.dtype
        self.plan = DedispersionPlan(config)
        self.nfreq = self.plan.nfreq
        self.nt_in = self.plan.nt_in
        self.ntrees = self.plan.ntrees
        self.trees = self.plan.trees

        self.stream_pool = CudaStreamPool(config.num_active_batches)
        self.dd = GpuDedisperser(
            self.plan, self.stream_pool,
            cuda_device_id=cuda_device_id,
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

        self._seq_id = 0            # next chunk's seq_id (== chunk index)
        self._acquired_seq = None   # seq_id of the currently-acquired output, if any

    def fill_analytic_weights(self, freq_variances):
        """Install non-random analytic peak-finding weights (so out_max is an SNR).

        'freq_variances' is the per-channel noise variance, length nfreq (e.g.
        XEngineMetadata.get_channel_variances()).
        """
        fv = np.ascontiguousarray(freq_variances, dtype=np.float64)
        assert fv.shape == (self.nfreq,), (fv.shape, self.nfreq)
        self.dd.fill_analytic_weights(fv)

    def run(self, frame):
        """Dedisperse one time chunk; return its GpuDedisperserOutputs.

        See the class docstring for the output-lifetime contract.
        """
        import cupy as cp

        assert frame.nfreq == self.nfreq, (frame.nfreq, self.nfreq)
        assert frame.ntime == self.nt_in, (frame.ntime, self.nt_in)

        stream = cp.cuda.get_current_stream()

        # Release the previous chunk's output slot (its arrays are now stale). The
        # caller has already consumed it (host-copied the peak) before this call.
        if self._acquired_seq is not None:
            stream.synchronize()
            self.dd.release_output(self._acquired_seq, stream=stream)
            self._acquired_seq = None

        # Upload this chunk's quantized data to the GPU, adding a length-1 beam axis.
        data_gpu = cp.asarray(np.asarray(frame.data))[None]              # (1, nfreq, nt_in/2) uint8
        scoff_gpu = cp.asarray(np.asarray(frame.scales_offsets))[None]   # (1, nfreq, mpc, 2) float16

        seq = self._seq_id
        # Dequantize directly into the dedisperser's input buffer, then launch the
        # dedispersion kernels; finally acquire this chunk's output.
        arr = self.dd.acquire_input(seq, stream=stream)                  # (1, nfreq, nt_in), dtype
        self.dqk.launch(arr, scoff_gpu, data_gpu, stream=stream)
        self.dd.release_input_and_launch_dd_kernels(seq, stream=stream)
        outputs = self.dd.acquire_output(seq, stream=stream)

        self._acquired_seq = seq
        self._seq_id += 1
        return outputs

    def close(self):
        """Release the final acquired output slot (call after the last run())."""
        if self._acquired_seq is not None:
            import cupy as cp
            stream = cp.cuda.get_current_stream()
            stream.synchronize()
            self.dd.release_output(self._acquired_seq, stream=stream)
            self._acquired_seq = None
