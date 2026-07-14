"""
FrbGrouper method injections (context-manager usage + get_output).

Split out from pirate_frb/pybind11_injections.py and kept here, alongside the
RPC clients, because FrbGrouper is the consumer side of the RPC interface.
Applied as a side effect of importing pirate_frb.rpc.
"""

from contextlib import contextmanager, ExitStack

import numpy as np

import ksgpu
from ..pirate_pybind11 import FrbGrouper
from .FrbSifterClient import FrbSifterEvents


@ksgpu.inject_methods(FrbGrouper)
class FrbGrouperInjections:
    """The FrbGrouper manages pirate-grouper communication (from the grouper side).
    
    This is a complex class, and the docstrings just summarize syntax.
    For a lot more info, see the grouper-specific parts of the sphinx docs.

    Constructor::

        g = pirate_frb.rpc.FrbGrouper(ip_addr)

    where ``ip_addr`` (str) is the grouper's own ``ip:port`` listen address (e.g.
    '127.0.0.1:7000'); the producer (an FrbServer running pirate) connects to it and
    hands off the GpuDedisperser output ring buffer over CUDA IPC. Use the object as a
    context manager (see below).

    Usage summary is straightforward, but note warnings below::

        with pirate_frb.rpc.FrbGrouper('127.0.0.1:7000') as g:
            for ichunk in itertools.count():       # outer loop over time chunks
                for ibatch in range(g.nbatches):   # inner loop over beam batches
                   with g.get_output(ichunk, ibatch) as outputs:
                       pass
                events = g.create_events(...)

    WARNING 1: don't use output arrays outside their context manager -- otherwise
    you'll get a silent race condition!! (The output arrays are views into a GPU
    memory ring buffer, and will be overwritten soon after context manager exit.)

    WARNING 2: the context manager synchronizes the current cupy stream on exit,
    then 'releases' the output arrays for the dedisperser to overwrite. The caller
    is responsible for ensuring that synchronizing the cupy stream is sufficient
    for buffer reuse. In normal cupy usage, you shouldn't need to worry about
    race conditions, but if you're using multiple streams then you may need
    additional synchronization logic.

    Attributes (all read-only):

    - ``is_stopped`` (bool) -- whether the grouper is in the stopped state.
    - ``cuda_device_id`` (int) -- CUDA device where the IPC-mapped outputs live.
    - ``dtype`` (ksgpu.Dtype) -- data type of the out_max arrays.
    - ``nt_in`` (int) -- input time samples per time chunk.
    - ``total_beams`` (int) -- total beams per chunk (= beams_per_gpu).
    - ``beams_per_batch`` (int) -- beams per output batch.
    - ``nbatches`` (int) -- beam-batches per chunk (= total_beams / beams_per_batch); producer ``seq_id = ichunk*nbatches + ibatch``.
    - ``num_batch_slots`` (int) -- output ring-buffer depth; leading beam axis = ``num_batch_slots*beams_per_batch`` (<= total_beams).
    - ``initial_chunk`` (int) -- chunk index of the producer's first output vs FPGA seq 0 (sets GpuDedisperserOutputs.ichunk_fpga_based).
    - ``ntrees`` (int) -- number of dedispersion trees.
    - ``ndm_out`` (list of int) -- per-tree output DM-channel counts (length ntrees).
    - ``nt_out`` (list of int) -- per-tree output time-sample counts (length ntrees).
    - ``dedispersion_config`` (DedispersionConfig) -- producer's dedispersion config (from the handshake).
    - ``xengine_metadata`` (XEngineMetadata) -- X-engine metadata (from the handshake).
    - ``xengine_metadata_yaml_string`` (str) -- X-engine metadata as a YAML string.
    - ``dedispersion_config_yaml_string`` (str) -- dedispersion config as a YAML string.
    - ``dedispersion_plan_yaml_string`` (str) -- dedispersion plan as a YAML string.
    - ``grouper_ip_addr`` (str) -- the grouper's own listen address (``ip:port``), set at construction.
    - ``search_ip_addr`` (str) -- producer FrbServer's FrbSearch RPC endpoint (``ip:port``), from the handshake.

    The Python context manager also attaches (available inside the ``with`` block):

    - ``xengine_yaml`` (dict) -- parsed X-engine metadata (from xengine_metadata_yaml_string).
    - ``dedispersion_config_yaml`` (dict) -- parsed dedispersion config.
    - ``dedispersion_plan_yaml`` (dict) -- parsed dedispersion plan.
    - ``steady_state_it0`` (list of cupy int arrays, one per tree). A dedispersion
      output array element (ichunk, beam, itree, idm, it) is "steady-state", i.e.
      unaffected by initial zero-padding, if:
      ``ichunk*nt_out + it >= steady_state_it0[itree][idm]``. Deliberately
      GPU-resident: it is consumed on the GPU once per tree per chunk (see
      ``steady_state_mask()``), and keeping it there avoids per-call host->GPU
      copies. Call ``.get()`` on the elements for host copies.
    - ``full_steady_ichunk`` (int) -- the smallest ichunk at/above which EVERY
      element of EVERY tree's output is steady-state (so the whole chunk is free of
      zero-padding artifacts). For ``ichunk >= full_steady_ichunk`` the entire
      ``steady_state_mask()`` is True.
    """
    # This class docstring (above) is the FrbGrouper docstring: the pybind11 binding
    # deliberately sets none, and ksgpu.inject_methods copies this one onto the class
    # (option 2 in notes/docstrings.md). It's kept here, next to the context-manager
    # / get_output() code that defines FrbGrouper's primary Python interface.

    def __enter__(self):
        import cupy as cp
        from ..Hardware import Hardware
        from ..utils import ThreadAffinity

        # Blocks until the client connects and the handshake is processed.
        self._open()

        # Everything below runs after _open() but before we return; if it raises, the
        # caller's 'with' will NOT run __exit__, so we undo it here and re-raise.
        try:
            # The handshake yaml strings are not YAML::Node-wrapped in pybind; parse
            # the wire strings into Python objects and attach them as attributes
            # (py::dynamic_attr() on the C++ class enables setting these). The strings
            # are only populated after the handshake, so this must follow self._open().
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

            # Precompute (once) the lookup tables derived from the handshake metadata,
            # used by create_events() and steady_state_mask().
            self._precompute_derived_tables()
        except BaseException:
            # Mirror __exit__: ensure close() runs even if the ExitStack unwind raises.
            try:
                es = getattr(self, "_exit_stack", None)
                if es is not None:
                    self._exit_stack = None
                    es.close()
            finally:
                self._close()
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
            self._close()
        return False

    @contextmanager
    def get_output(self, ichunk, ibatch):
        """Acquire one beam-batch's outputs, as a GpuDedisperserOutputs object.

        Usage summary (see grouper-specific parts of sphinx docs for more info)::

            # Warning: output arrays are only valid inside the context manager!!
            # Therefore, each batch must be fully processed, before seeing the next batch.
            with grouper.get_output(ichunk, ibatch) as outputs:
                # Loop over dedispersion trees.
                for itree, tree_out in enumerate(outputs.out_max):
                    # 'tree_out' has shape (beams_per_batch, coarse_ndm, coarse_ntime)
        
        WARNING 1: don't use output arrays outside their context manager -- otherwise
        you'll get a silent race condition!! (The output arrays are views into a GPU
        memory ring buffer, and will be overwritten soon after context manager exit.)
        
        WARNING 2: the context manager synchronizes the current cupy stream on exit,
        then 'releases' the output arrays for the dedisperser to overwrite. The caller
        is responsible for ensuring that synchronizing the cupy stream is sufficient
        for buffer reuse. In normal cupy usage, you shouldn't need to worry about
        race conditions, but if you're using multiple streams then you may need
        additional synchronization logic. For example:
        
        - Any GPU read of 'outputs' that has not COMPLETED by block exit races
          with the producer. The exit sync covers only work enqueued on the
          current cupy stream: enqueue all reads there, and don't leave a
          different stream current at exit. Work on any other stream must be
          synchronized manually before the block exits.

        - A GPU->host copy on the current cupy stream is covered by the exit
          sync, so its host buffer is valid after the block. On any other
          stream, it is not.

        Parameters
        ----------
        ichunk : int
            Zero-based time-chunk index (i.e. first call to get_output() should
            specify ichunk=0, and ignore the value of FrbGrouper.initial_chunk).
        ibatch : int
            Beam-batch index within the chunk (must satisfy
            0 <= ibatch < self.nbatches).

        Yields
        ------
        GpuDedisperserOutputs
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
        outputs = self._acquire_output(seq_id)
        try:
            yield outputs
        finally:
            cp.cuda.get_current_stream().synchronize()
            self._release_output(seq_id)

    def _precompute_derived_tables(self):
        """Precompute the small lookup tables derived from the handshake metadata,
        used by create_events() and steady_state_mask().

        Called once from __enter__ (after the handshake). All quantities come from the
        pybind-wrapped handshake objects (dedispersion_config, xengine_metadata) or are
        computed in C++ (_compute_steady_state_it0) -- nothing is re-derived in python.
        Sets the members used by create_events():

          - _seq_per_sample: fpga counts per input time sample.
          - _time_sample_ms: input time sample length (converts decoded widths to ms).
          - _beam_id_lut: global beam index -> X-engine beam id (numpy array).

        and the members used by steady_state_mask() (the per-tree steady-state
        boundary steady_state_it0 and the chunk-level threshold full_steady_ichunk,
        both documented in the class docstring).

        The tables are small numpy (host) arrays / scalars -- create_events() indexes
        them on the CPU, since for the tiny event arrays a GPU kernel launch would
        cost more than the CPU compute -- EXCEPT steady_state_it0, which is
        deliberately GPU-resident (see below).
        """
        import cupy as cp

        cfg = self.dedispersion_config
        xmd = self.xengine_metadata

        self._time_sample_ms   = float(cfg.time_sample_ms)
        self._beam_id_lut      = np.asarray(xmd.beam_ids, dtype=np.int64)
        self._seq_per_sample   = int(xmd.seq_per_frb_time_sample)

        # Per-tree steady-state boundary (see class docstring), computed in C++
        # (_compute_steady_state_it0() forwards DedispersionPlan.compute_steady_state_it0()
        # on the producer's plan from the handshake), then moved to the GPU: it is
        # consumed on the GPU once per tree per chunk by steady_state_mask(), and
        # keeping it GPU-resident avoids a host->GPU copy on every call. (__enter__
        # has already selected the grouper's CUDA device at this point.)
        # (ichunk*nt_out + it) >= steady_state_it0[itree][idm]  ==>  steady-state.
        it0 = [self._compute_steady_state_it0(i) for i in range(self.ntrees)]  # host (numpy)
        self.steady_state_it0 = [cp.asarray(a) for a in it0]

        # Chunk-level threshold: the smallest ichunk at/above which every element of
        # every tree is steady-state. For tree i, the whole chunk is steady once
        # ichunk*nt_out[i] >= max(it0[i]) (worst element is idm=argmax(it0), it=0), so
        # ichunk >= ceil(max(it0[i]) / nt_out[i]); take the max over trees.
        self.full_steady_ichunk = max(
            ((int(a.max()) + self.nt_out[i] - 1) // self.nt_out[i] for i, a in enumerate(it0)),
            default=0)

    def steady_state_mask(self, itree, ichunk):
        """Return a cupy bool mask of shape (ndm_out, nt_out) for tree 'itree': True
        where output element (ichunk, idm, it) is steady-state, i.e. unaffected by
        the zero-padding before the start of the acquisition (see the
        'steady_state_it0' bullet in the class docstring; the mask is independent
        of the beam axis).

        Computed entirely on the GPU (steady_state_it0 is GPU-resident), with no
        host<->GPU copies or syncs -- cheap enough to call once per tree per chunk.
        Intended for masking per-chunk statistics, e.g.
        GpuGrouperHistogram.add_tree(tree_out, itree, ..., mask=grouper.steady_state_mask(...)).
        (For ichunk >= full_steady_ichunk the mask is all True; callers may pass
        mask=None in that case to skip the redundant mask + copy.)
        """
        import cupy as cp
        min_it = self.steady_state_it0[itree] - ichunk * self.nt_out[itree]
        return cp.arange(self.nt_out[itree])[None, :] >= min_it[:, None]

    def create_events(self, ichunk, itrees, ibeams, idm, itime, snr, argmax):
        """Build a FrbSifterEvents from GPU arrays of (tree, beam, dm, time, token) data.

        The grouper finds peaks at index quadruples (itree, ibeam, idm, itime), and
        needs to translate these indices into physical quantities (DM, time, etc)
        before sending events to the sifter.

        The create_events() method performs this translation by decoding the
        out_argmax tokens (via DedispersionPlan.decode_argmax / decode_argmax2, using
        the producer's plan from the handshake): the per-event dm, arrival time,
        intrinsic width, and frequency subband are the fine-grained "winning" trial
        parameters, not coarse-pixel centers. This includes subtleties like early
        triggers: arrival times are extrapolated to the lowest frequency of the full
        band, so they can lie outside the chunk's time window -- in the future (early
        triggers), or slightly before the chunk start (finite peak-finder kernel widths
        shift the estimated pulse-center time earlier than the detection sample).
        For more info, see the grouper-specific parts of the sphinx docs, and/or the
        tex notes.

        The per-event arrays (itrees, ibeams, idm, itime, snr, argmax) must all be
        1-d cupy arrays of the same length -- one event per element. The 'argmax'
        values are the out_argmax tokens of the selected peaks, which the caller must
        gather (on the GPU) while the output arrays are valid, i.e. inside the
        get_output() context manager. Indices and tokens are bounds-checked on the
        host as a side effect of decoding.

        Parameters
        ----------
        ichunk : int
            Time-chunk index, using the same zero-based indexing as get_output().
            (Note: when create_events() computes the arrival time, it adds the value
            of 'initial_chunk', which was sent in the pirate-grouper handshake.)
        itrees : cupy.ndarray
            Per-event dedispersion-tree index.
        ibeams : cupy.ndarray
            Per-event global beam index.
        idm : cupy.ndarray
            Per-event index along the selected tree's output DM axis.
        itime : cupy.ndarray
            Per-event index along the selected tree's output time axis.
        snr : cupy.ndarray
            Per-event SNR.
        argmax : cupy.ndarray
            Per-event out_argmax token (uint32), gathered by the caller from
            outputs.out_argmax[itree][ibeam - beam0, idm, itime] inside the
            get_output() context manager.

        Returns
        -------
        FrbSifterEvents
            The events translated into physical units (beam id, absolute FPGA
            timestamp, DM, SNR, width, frequency subband), ready to pass to
            FrbSifterClient.send_events(). Also carries the chunk's absolute FPGA
            window (``chunk_fpga_start``, ``chunk_fpga_end``). Per-event timestamps
            may lie before or after that window (see above), but negative absolute
            timestamps (possible in the earliest chunks, where events are warmup
            artifacts anyway) are clamped to zero. rfi_prob is a placeholder (= 0,
            not measured by the grouper).
        """
        import cupy as cp

        if not isinstance(ichunk, (int, np.integer)):
            raise TypeError(f"FrbGrouper.create_events: ichunk must be an int, "
                            f"got {type(ichunk).__name__}")
        if ichunk < 0:
            raise ValueError(f"FrbGrouper.create_events: ichunk must be >= 0 (got {ichunk})")

        named = dict(itrees=itrees, ibeams=ibeams, idm=idm, itime=itime, snr=snr, argmax=argmax)
        for name, a in named.items():
            if not isinstance(a, cp.ndarray):
                raise TypeError(f"FrbGrouper.create_events: {name!r} must be a cupy array, "
                                f"got {type(a).__name__}")
            if a.ndim != 1:
                raise ValueError(f"FrbGrouper.create_events: {name!r} must be 1-d "
                                 f"(one event per element), got shape {a.shape}")
        shapes = {name: a.shape for name, a in named.items()}
        if len(set(shapes.values())) != 1:
            raise ValueError(f"FrbGrouper.create_events: itrees/ibeams/idm/itime/snr/argmax "
                             f"must all have the same shape, got {shapes}")

        # Absolute (FPGA-seq-0-based) timing of this chunk. 'ichunk' is zero-based (relative
        # to the producer's first output chunk); adding initial_chunk offsets it to FPGA
        # seq 0 (cf. GpuDedisperserOutputs.ichunk_fpga_based = ichunk_zero_based + initial_chunk).
        # fpga_per_chunk = nt_in input samples * fpga-counts-per-sample.
        fpga_per_chunk = self.nt_in * self._seq_per_sample
        chunk_fpga_start = (int(ichunk) + self.initial_chunk) * fpga_per_chunk
        chunk_fpga_end = chunk_fpga_start + fpga_per_chunk

        snr_h = snr.get()
        n = int(itrees.size)

        if n > 0:
            # Copy the (tiny) per-event arrays to the host in one stacked device->host
            # copy, plus snr. All subsequent math is done on the CPU: these arrays are
            # tiny, so a GPU kernel launch would cost more than the CPU compute.
            itree_h, ibeam_h, idm_h, itime_h, tok_h = \
                cp.stack([itrees, ibeams, idm, itime, argmax.astype(cp.int64)]).get().astype(np.int64)

            # Decode the out_argmax tokens into the fine-grained "winning" trial params:
            # frequency subband, DM (from the winning fine-grained delay slope), arrival
            # timestamp (chunk-relative toplevel samples, extrapolated to the lowest
            # full-band frequency), and peak-finder width (toplevel samples).
            fmins, fmaxs, tlos, this_, ps = self.decode_argmax_batch(
                tok_h.astype(np.uint32), itree_h, idm_h, itime_h)
            freqs_lo, freqs_hi, dms, ts_samp, widths_samp = self.decode_argmax2_batch(
                itree_h, fmins, fmaxs, tlos, this_, ps)

            beam_ids = self._beam_id_lut[ibeam_h]
            widths_ms = widths_samp * self._time_sample_ms

            # Per-event absolute timestamp = chunk start + chunk-relative offset (decoded
            # arrival time in input samples * fpga-counts-per-sample), rounded to the
            # nearest integer fpga count. Timestamps may fall before or after the chunk
            # window (high-DM events reach into earlier chunks, and finite peak-finder
            # kernel widths shift near-chunk-start events slightly earlier; early
            # triggers extrapolate into the future); negative ABSOLUTE timestamps are
            # clamped.
            offset = np.rint(ts_samp * self._seq_per_sample).astype(np.int64)
            fpga_timestamps = np.maximum(chunk_fpga_start + offset, 0)
        else:
            # Zero events: the batch decoders require nonempty arrays, so build the
            # (empty) per-event arrays directly. (FrbSifterEvents casts dtypes.)
            beam_ids = np.zeros(0, dtype=np.int64)
            fpga_timestamps = np.zeros(0, dtype=np.int64)
            dms = freqs_lo = freqs_hi = widths_ms = np.zeros(0, dtype=np.float64)

        # FrbSifterEvents casts dtypes and validates shapes. rfi_prob is the one remaining
        # placeholder (the grouper doesn't measure it).
        return FrbSifterEvents(
            beam_ids = beam_ids,
            fpga_timestamps = fpga_timestamps,
            dms = dms,
            snrs = snr_h,
            rfi_probs = np.zeros(snr_h.shape, dtype=np.float64),
            widths_ms = widths_ms,
            subband_freqs_lo_MHz = freqs_lo,
            subband_freqs_hi_MHz = freqs_hi,
            chunk_fpga_start = chunk_fpga_start,
            chunk_fpga_end = chunk_fpga_end,
        )


