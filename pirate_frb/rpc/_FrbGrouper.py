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

        # Everything below runs after open() but before we return; if it raises, the
        # caller's 'with' will NOT run __exit__, so we undo it here and re-raise.
        try:
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

            # Precompute the (host/numpy) per-tree lookup tables used by create_events(),
            # once, rather than on every create_events() call.
            self._precompute_event_tables()
        except BaseException:
            # Mirror __exit__: ensure close() runs even if the ExitStack unwind raises.
            try:
                es = getattr(self, "_exit_stack", None)
                if es is not None:
                    self._exit_stack = None
                    es.close()
            finally:
                self.close()
            raise
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

    def _precompute_event_tables(self):
        """Precompute the per-tree lookup tables used by create_events().

        Called once from __enter__ (after the handshake yamls are parsed). Following
        notes/tree_dedispersion.tex, section "Dedispersion output arrays", the
        per-tree output geometry (ndm_out, nt_out, d_lo, d_hi) is computed from
        (T_in, r_top, ids, delta, T_ds, D_ds) and cross-checked against the plan's
        stored ndm_out/nt_out/dm_min/dm_max (a one-time guard against code/tex/plan
        drift). Sets the members used by create_events():

          - dm_per_unit_delay: a full-band delay of 'd' input samples is DM =
            d * dm_per_unit_delay (matches DedispersionConfig::dm_per_unit_delay).
          - _nt_in, _seq_per_sample: T_in, and fpga counts per input time sample.
          - _beam_id_lut: global beam index -> X-engine beam id (numpy array).
          - _tree_{ndm_out,nt_out,d_lo,d_hi,delta}: per-tree numpy arrays, indexed
            by tree index.

        The tables are small numpy (host) arrays: create_events() indexes them on the
        CPU, since for the tiny event arrays a GPU kernel launch would cost more than
        the CPU compute.
        """
        plan  = self.dedispersion_plan_yaml
        trees = plan['trees']
        T_in  = plan['nt_in']           # input time samples per chunk  (T_in)
        r_top = plan['toplevel_rank']   # tree rank                     (r_top)

        # DM per unit full-band delay, from constants::k_dm and the config's band
        # edges + sample length:  delay_ms = k_dm * DM * (f_lo^{-2} - f_hi^{-2}),
        # with delay_ms = d * time_sample_ms.
        cfg   = self.dedispersion_config_yaml
        edges = cfg['zone_freq_edges']
        f_lo, f_hi = edges[0], edges[-1]
        k_dm  = pirate_pybind11.constants.k_dm
        self.dm_per_unit_delay = cfg['time_sample_ms'] / (k_dm * (1.0/f_lo**2 - 1.0/f_hi**2))

        # Per-tree geometry from (ids, delta, T_ds, D_ds) via the tex equations,
        # cross-checked against the plan's stored ndm_out/nt_out/dm_min/dm_max.
        ndm_l, nt_l, dlo_l, dhi_l, delta_l = [], [], [], [], []
        for i, tr in enumerate(trees):
            ids, delta = tr['ds_level'], tr['delta_rank']
            T_ds, D_ds = tr['time_downsampling'], tr['dm_downsampling']
            ndm = (2**(r_top - delta) if ids == 0 else 2**(r_top - delta - 1)) // D_ds  # Eq.(ndm_out)
            nt  = T_in // (2**ids * T_ds)                                               # Eq.(nt_out)
            dlo = 0 if ids == 0 else 2**(r_top + ids - 1)                               # Eq.(dlo_dhi)
            dhi = 2**(r_top + ids)                                                      # Eq.(dlo_dhi)
            if (ndm != tr['ndm_out']) or (nt != tr['nt_out']):
                raise RuntimeError(f"FrbGrouper: tex-derived (ndm_out, nt_out) = ({ndm}, {nt}) "
                                   f"disagree with the plan ({tr['ndm_out']}, {tr['nt_out']}) "
                                   f"for tree {i}")
            # dm_min/dm_max are the DMs of full-band delays d_lo/d_hi.
            for label, got, exp in (("dm_min", dlo * self.dm_per_unit_delay, tr['dm_min']),
                                    ("dm_max", dhi * self.dm_per_unit_delay, tr['dm_max'])):
                if abs(got - exp) > 1e-9 * max(1.0, abs(exp)):
                    raise RuntimeError(f"FrbGrouper: tex-derived {label} = {got} disagrees with "
                                       f"the plan ({exp}) for tree {i}")
            ndm_l.append(ndm); nt_l.append(nt); dlo_l.append(dlo); dhi_l.append(dhi)
            delta_l.append(delta)

        # Per-tree lookup tables (numpy/host arrays), indexed by tree index.
        self._tree_ndm_out = np.asarray(ndm_l,   dtype=np.float64)
        self._tree_nt_out  = np.asarray(nt_l,    dtype=np.float64)
        self._tree_d_lo    = np.asarray(dlo_l,   dtype=np.float64)
        self._tree_d_hi    = np.asarray(dhi_l,   dtype=np.float64)
        self._tree_delta   = np.asarray(delta_l, dtype=np.float64)

        # Beam-id lookup table + timing scalars.
        self._beam_id_lut    = np.asarray(self.xengine_yaml['beam_ids'], dtype=np.int64)
        self._nt_in          = int(T_in)
        self._seq_per_sample = int(self.xengine_yaml['seq_per_frb_time_sample'])

    def create_events(self, ichunk, itrees, ibeams, idm, itime, snr):
        """Build a FrbSifterEvents from GPU arrays of (tree, beam, dm, time) indices.

        Converts the array indices of a set of events -- the dedispersion-tree index,
        global beam index, and the tree's (dm, time) output-axis indices -- plus their
        SNRs into physical units (beam id, absolute FPGA timestamp, DM), following
        notes/tree_dedispersion.tex, section "Dedispersion output arrays" (Eqs. idm_it,
        idm_it_early). The per-tree geometry tables and dm_per_unit_delay are
        precomputed once in __enter__ (see _precompute_event_tables); this method does
        only the per-event arithmetic.

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

        # Copy the (tiny) index arrays to the host in one stacked device->host copy,
        # plus snr. All subsequent math is done on the CPU with numpy: these arrays
        # are tiny, so a GPU kernel launch would cost more than the CPU compute.
        itree_h, ibeam_h, idm_h, itime_h = cp.stack([itrees, ibeams, idm, itime]).get().astype(np.int64)
        snr_h = snr.get()

        # Index -> physical units (on the CPU), following the tex. The per-tree tables
        # (_tree_*) and dm_per_unit_delay were precomputed in __enter__.
        #   d  = full-band delay (input-sample units)        Eq.(idm_it)
        #   t  = arrival time at the trigger frequency        Eq.(idm_it)
        #   t' = arrival time at the lowest full-band freq    Eq.(idm_it_early)
        d_lo, d_hi = self._tree_d_lo[itree_h], self._tree_d_hi[itree_h]
        d  = d_lo + (d_hi - d_lo) * (idm_h + 0.5) / self._tree_ndm_out[itree_h]
        t  = self._nt_in * (itime_h + 0.5) / self._tree_nt_out[itree_h]
        t_full = t + (1.0 - 2.0**(-self._tree_delta[itree_h])) * d

        dms = d * self.dm_per_unit_delay
        beam_ids = self._beam_id_lut[ibeam_h]
        # Absolute fpga timestamp = chunk start + within-chunk offset (t' in input
        # samples * fpga-counts-per-sample), rounded to the nearest integer fpga count.
        chunk_fpga = int(ichunk) * self._nt_in * self._seq_per_sample
        offset = np.rint(t_full * self._seq_per_sample).astype(np.int64)
        fpga_timestamps = chunk_fpga + offset

        # FrbSifterEvents casts dtypes and validates shapes. dm_error / rfi_prob are
        # zero for the toy grouper.
        return FrbSifterEvents(
            beam_ids = beam_ids,
            fpga_timestamps = fpga_timestamps,
            dms = dms,
            dm_errors = np.zeros(snr_h.shape, dtype=np.float32),
            snrs = snr_h,
            rfi_probs = np.zeros(snr_h.shape, dtype=np.float32),
            chunk_fpga_count = chunk_fpga,
        )


