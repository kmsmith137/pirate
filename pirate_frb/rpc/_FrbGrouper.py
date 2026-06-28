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
        plus their SNRs into physical units (beam id, absolute FPGA timestamp, DM).
        The conversion follows notes/tree_dedispersion.tex, section "Dedispersion
        output arrays": the per-tree geometry (ndm_out, nt_out, d_lo, d_hi) is
        computed from (T_in, r_top, ids, delta, T_ds, D_ds) read from the
        dedispersion plan -- NOT from the plan's dm_min/dm_max -- and the full-band
        delay is converted to a DM via pirate::constants::k_dm plus the band edges
        and sample length from the dedispersion config.

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
            chunk_fpga_count = ichunk * T_in * seq_per_frb_time_sample. dm_error and
            rfi_prob are set to zero (not computed by the toy grouper). For an
            early-trigger tree (delta>0), the reported arrival time is corrected from
            the trigger frequency to the lowest full-band frequency (Eq. idm_it_early).
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

        # Global plan params (tree_dedispersion.tex, "Parameter definitions").
        plan  = self.dedispersion_plan_yaml
        trees = plan['trees']
        T_in  = plan['nt_in']           # input time samples per chunk  (T_in)
        r_top = plan['toplevel_rank']   # tree rank                     (r_top)

        # DM per unit full-band delay, reconstructed from constants::k_dm and the
        # config's band edges + sample length (matches DedispersionConfig::dm_per_unit_delay):
        #   delay_ms = k_dm * DM * (f_lo^{-2} - f_hi^{-2}),  with delay_ms = d * time_sample_ms.
        cfg   = self.dedispersion_config_yaml
        edges = cfg['zone_freq_edges']
        f_lo, f_hi = edges[0], edges[-1]
        k_dm  = pirate_pybind11.constants.k_dm
        dm_per_unit_delay = cfg['time_sample_ms'] / (k_dm * (1.0/f_lo**2 - 1.0/f_hi**2))

        # Per-tree output geometry + full-band delay range, computed from
        # (ids, delta, T_ds, D_ds) via the tex equations -- NOT read from the plan's
        # ndm_out/nt_out/dm_min/dm_max. Those plan values are read back only to assert
        # that our formulas agree with the plan (a guard against code/tex/plan drift).
        ndm_l, nt_l, dlo_l, dhi_l, delta_l = [], [], [], [], []
        for i, tr in enumerate(trees):
            ids, delta = tr['ds_level'], tr['delta_rank']
            T_ds, D_ds = tr['time_downsampling'], tr['dm_downsampling']
            ndm = (2**(r_top - delta) if ids == 0 else 2**(r_top - delta - 1)) // D_ds  # Eq.(ndm_out)
            nt  = T_in // (2**ids * T_ds)                                               # Eq.(nt_out)
            dlo = 0 if ids == 0 else 2**(r_top + ids - 1)                               # Eq.(dlo_dhi)
            dhi = 2**(r_top + ids)                                                      # Eq.(dlo_dhi)
            if (ndm != tr['ndm_out']) or (nt != tr['nt_out']):
                raise RuntimeError(f"FrbGrouper.create_events: tex-derived (ndm_out, nt_out) = "
                                   f"({ndm}, {nt}) disagree with the plan ({tr['ndm_out']}, "
                                   f"{tr['nt_out']}) for tree {i}")
            # dm_min/dm_max are the DMs of full-band delays d_lo/d_hi (DM = d * dm_per_unit_delay).
            for label, got, exp in (("dm_min", dlo * dm_per_unit_delay, tr['dm_min']),
                                    ("dm_max", dhi * dm_per_unit_delay, tr['dm_max'])):
                if abs(got - exp) > 1e-9 * max(1.0, abs(exp)):
                    raise RuntimeError(f"FrbGrouper.create_events: tex-derived {label} = {got} "
                                       f"disagrees with the plan ({exp}) for tree {i}")
            ndm_l.append(ndm); nt_l.append(nt); dlo_l.append(dlo); dhi_l.append(dhi)
            delta_l.append(delta)

        # Per-tree lookup tables (small host lists -> device arrays), indexed by itrees.
        ndm_out = cp.asarray(ndm_l,   dtype=cp.float64)
        nt_out  = cp.asarray(nt_l,    dtype=cp.float64)
        d_lo    = cp.asarray(dlo_l,   dtype=cp.float64)
        d_hi    = cp.asarray(dhi_l,   dtype=cp.float64)
        delta_a = cp.asarray(delta_l, dtype=cp.float64)

        # FPGA timing (X-engine metadata) and global-beam-index -> beam-id table.
        seq_per_sample = self.xengine_yaml['seq_per_frb_time_sample']  # fpga counts / input sample
        chunk_fpga = int(ichunk) * int(T_in) * int(seq_per_sample)
        beam_id_lut = cp.asarray(self.xengine_yaml['beam_ids'], dtype=cp.int64)

        it = itrees.astype(cp.int64)   # integer indices for the lookup tables
        ib = ibeams.astype(cp.int64)

        # Index -> physical units (all on the GPU), following the tex:
        #   d  = full-band delay (input-sample units)         Eq.(idm_it)
        #   t  = arrival time at the trigger frequency         Eq.(idm_it)
        #   t' = arrival time at the lowest full-band freq     Eq.(idm_it_early)
        d  = d_lo[it] + (d_hi[it] - d_lo[it]) * (idm + 0.5) / ndm_out[it]
        t  = T_in * (itime + 0.5) / nt_out[it]
        t_full = t + (1.0 - 2.0**(-delta_a[it])) * d

        dms = d * dm_per_unit_delay
        beam_ids = beam_id_lut[ib]
        # Absolute fpga timestamp = chunk start + within-chunk offset (t' in input
        # samples * fpga-counts-per-sample), rounded to the nearest integer fpga count.
        offset = cp.rint(t_full * seq_per_sample).astype(cp.int64)
        fpga_timestamps = chunk_fpga + offset

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


