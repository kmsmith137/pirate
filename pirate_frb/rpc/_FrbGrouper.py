"""
FrbGrouper method injections (context-manager usage + get_output).

Split out from pirate_frb/pybind11_injections.py and kept here, alongside the
RPC clients, because FrbGrouper is the consumer side of the RPC interface.
Applied as a side effect of importing pirate_frb.rpc.
"""

from contextlib import contextmanager, ExitStack

import numpy as np

import ksgpu
from .. import pirate_pybind11
from .FrbSifterClient import FrbSifterEvents


@ksgpu.inject_methods(pirate_pybind11.FrbGrouper)
class FrbGrouperInjections:
    """Python extensions for FrbGrouper (context-manager usage + get_output)."""

    def __enter__(self):
        import cupy as cp
        from ..Hardware import Hardware
        from ..utils import ThreadAffinity

        # Blocks until the client connects and the handshake is processed.
        self.open()

        # The handshake yaml strings are not YAML::Node-wrapped in pybind; parse
        # the wire strings into Python objects and attach them as attributes
        # (py::dynamic_attr() on the C++ class enables setting these). The strings
        # are only populated after the handshake, so this must follow self.open().
        import yaml
        self.xengine_yaml = yaml.safe_load(self.xengine_metadata_yaml_string)
        self.dedispersion_config_yaml = yaml.safe_load(self.dedispersion_config_yaml_string)
        self.dedispersion_plan_yaml = yaml.safe_load(self.dedispersion_plan_yaml_string)

        # The IPC-mapped output arrays live on cuda_device_id (known after the
        # handshake). For the duration of the 'with' block, pin this thread to
        # the vcpus local to that GPU and select the device, so the consumer's
        # cupy work runs on the right device with good CPU locality. Both are
        # entered via an ExitStack and undone in __exit__.
        vcpu_list = Hardware().vcpu_list_from_gpu(self.cuda_device_id)
        print(f"FrbGrouper: pinning thread to vcpu_list={vcpu_list} and selecting "
              f"cuda_device_id={self.cuda_device_id}", flush=True)
        self._exit_stack = ExitStack()
        self._exit_stack.enter_context(ThreadAffinity(vcpu_list))
        self._exit_stack.enter_context(cp.cuda.Device(self.cuda_device_id))
        return self

    def __exit__(self, *exc):
        # Undo the ThreadAffinity / cuda.Device contexts entered in __enter__,
        # then close the grouper. try/finally so close() always runs.
        try:
            es = getattr(self, "_exit_stack", None)
            if es is not None:
                self._exit_stack = None
                es.close()
        finally:
            self.close()
        return False

    @contextmanager
    def get_output(self, ichunk, ibatch):
        """Acquire one beam-batch's outputs; on exit synchronize the GPU, then release.

        Parameters
        ----------
        ichunk : int
            Zero-based time-chunk index (must be >= 0).
        ibatch : int
            Beam-batch index within the chunk (must satisfy
            0 <= ibatch < self.nbatches).

        The producer sequence id is ``seq_id = ichunk * nbatches + ibatch``.

        On exit this calls ``cupy.cuda.get_current_stream().synchronize()``
        BEFORE ``release_output(seq_id)``, so all GPU reads the body queued on
        the current cupy stream complete before CONSUMED is sent to the
        producer. This is required because there is no IPC-event fence: once
        CONSUMED is sent, the producer may overwrite the ring-buffer slot (see
        plans/grouper_server.md). The body must therefore do its GPU work on the
        current cupy stream (the default; FrbGrouper's __enter__ has already
        selected the right device).

        Yields
        ------
        _GpuDedisperserOutputs
            Per-batch slice with .out_max / .out_argmax (lists of ksgpu Arrays,
            convertible to cupy via DLPack).
        """
        if ichunk < 0:
            raise ValueError(f"FrbGrouper.get_output: ichunk must be >= 0 (got {ichunk})")
        if not (0 <= ibatch < self.nbatches):
            raise ValueError(f"FrbGrouper.get_output: ibatch must be in "
                             f"[0, {self.nbatches}) (got {ibatch})")
        import cupy as cp
        seq_id = ichunk * self.nbatches + ibatch
        outputs = self.acquire_output(seq_id)
        try:
            yield outputs
        finally:
            cp.cuda.get_current_stream().synchronize()
            self.release_output(seq_id)

    def create_events(self, ichunk, itrees, ibeams, idm, itime, snr):
        """Build a FrbSifterEvents from GPU arrays of (tree, beam, dm, time) indices.

        Converts the array indices of a set of events -- the dedispersion-tree
        index, global beam index, and the tree's (dm, time) output-axis indices --
        plus their SNRs into physical units (beam id, absolute FPGA timestamp, DM),
        using this grouper's X-engine metadata and dedispersion plan (parsed in
        __enter__). The result is a host-side FrbSifterEvents, ready to pass to
        FrbSifterClient.send_events.

        Parameters
        ----------
        ichunk : int
            Zero-based time-chunk index (>= 0); sets the absolute FPGA timing.
        itrees, ibeams, idm, itime, snr : cupy.ndarray
            Equal-shaped GPU arrays (one event per element). itrees selects the
            dedispersion tree; ibeams is a global beam index; idm/itime index that
            tree's output (dm, time) axes; snr is the event SNR. Index values are
            assumed in range (not bounds-checked, to avoid a device->host sync).

        Returns
        -------
        FrbSifterEvents
            chunk_fpga_count = ichunk * (seq_per_frb_time_sample * nt_in). dm_error
            and rfi_prob are set to zero (not computed by the toy grouper).
        """
        import cupy as cp

        if not isinstance(ichunk, (int, np.integer)):
            raise TypeError(f"FrbGrouper.create_events: ichunk must be an int, "
                            f"got {type(ichunk).__name__}")
        if ichunk < 0:
            raise ValueError(f"FrbGrouper.create_events: ichunk must be >= 0 (got {ichunk})")

        named = dict(itrees=itrees, ibeams=ibeams, idm=idm, itime=itime, snr=snr)
        for name, a in named.items():
            if not isinstance(a, cp.ndarray):
                raise TypeError(f"FrbGrouper.create_events: {name!r} must be a cupy array, "
                                f"got {type(a).__name__}")
        shapes = {name: a.shape for name, a in named.items()}
        if len(set(shapes.values())) != 1:
            raise ValueError(f"FrbGrouper.create_events: itrees/ibeams/idm/itime/snr must all "
                             f"have the same shape, got {shapes}")

        # Scalar FPGA count at the start of this chunk.
        seq_per_sample = self.xengine_yaml['seq_per_frb_time_sample']
        fpga_per_chunk = int(seq_per_sample) * int(self.nt_in)
        chunk_fpga = int(ichunk) * fpga_per_chunk

        # Per-tree lookup tables (small host lists -> device arrays), indexed by itrees;
        # plus the global-beam-index -> X-engine beam-id table.
        trees = self.dedispersion_plan_yaml['trees']
        dm_min  = cp.asarray([t['dm_min']  for t in trees], dtype=cp.float64)
        dm_max  = cp.asarray([t['dm_max']  for t in trees], dtype=cp.float64)
        ndm_out = cp.asarray([t['ndm_out'] for t in trees], dtype=cp.float64)
        nt_out  = cp.asarray([t['nt_out']  for t in trees], dtype=cp.int64)
        beam_id_lut = cp.asarray(self.xengine_yaml['beam_ids'], dtype=cp.int64)

        it = itrees.astype(cp.int64)   # integer indices for the lookup tables
        ib = ibeams.astype(cp.int64)

        # Index -> physical-unit conversions, all on the GPU.
        dms = dm_min[it] + (dm_max[it] - dm_min[it]) * (idm / ndm_out[it] + 0.5)
        # Each tree's nt_out output samples span the whole chunk, so one output
        # sample is fpga_per_chunk // nt_out FPGA counts.
        fpga_per_out_sample = fpga_per_chunk // nt_out[it]
        fpga_timestamps = chunk_fpga + itime.astype(cp.int64) * fpga_per_out_sample
        beam_ids = beam_id_lut[ib]

        # One bulk device->host copy per array (FrbSifterEvents casts dtypes and
        # validates shapes). dm_error / rfi_prob are zero for the toy grouper.
        return FrbSifterEvents(
            beam_ids = beam_ids.get(),
            fpga_timestamps = fpga_timestamps.get(),
            dms = dms.get(),
            dm_errors = np.zeros(snr.shape, dtype=np.float32),
            snrs = snr.get(),
            rfi_probs = np.zeros(snr.shape, dtype=np.float32),
            chunk_fpga_count = chunk_fpga,
        )


