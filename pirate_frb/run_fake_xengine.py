"""Implementation of 'pirate_frb run_fake_xengine' subcommand."""

import time
import threading

from datetime import datetime, timezone

import numpy as np

from .Hardware import Hardware
from .utils import ThreadAffinity, extract_ip, check_mtu
from .core import (
    AssembledFrameAllocator,
    FakeXEngine,
    FrequencySubbands,
    SimulatedFrameFactory,
    SlabAllocator,
    XEngineMetadata,
)
from .rpc import FrbSearchClient, FrbSifterClient


class RunFakeXEngineHelper:
    """Encapsulates state and logic for 'pirate_frb run_fake_xengine'.

    Constructed once per invocation; call .run() to drive the full
    lifecycle (Phase 1: per-receiver GetConfig + FakeXEngine; Phase 2:
    spawn controllers; wait for Ctrl-C; cleanup).
    """

    def __init__(self, rpc_addrs, nworkers=128, paced=True, normalized=True,
                 gaussian=True, send_junk=False, simulate_frbs=False, sifter_addr=None,
                 frb_gap_sec=0.0):
        # Strings are iterable, so a caller who passes a bare string would
        # silently iterate character-by-character. Short-circuit with a clear
        # error.
        if isinstance(rpc_addrs, str):
            raise RuntimeError(
                f"run_fake_xengine: rpc_addrs must be a list of strings, "
                f"not a single string ({rpc_addrs!r})"
            )
        if not rpc_addrs:
            raise RuntimeError("run_fake_xengine: rpc_addrs is empty")
        if len(set(rpc_addrs)) != len(rpc_addrs):
            raise RuntimeError(
                f"run_fake_xengine: duplicate rpc_addrs in {rpc_addrs!r}"
            )

        self.rpc_addrs = rpc_addrs
        self.nworkers = nworkers
        self.paced = paced
        # normalized: when True (default), the SimulatedFrameFactory calibrates
        # each frame's scales/offsets to xmd's per-zone noise variance; when
        # False the normalization is arbitrary. See AssembledFrame.randomize().
        self.normalized = normalized
        # gaussian: when True (default), the factory fills the int4 data with
        # simulated Gaussian noise clamped to [-7,+7]; when False it fills with
        # uniform int4 over [-8,+7]. See AssembledFrame.randomize().
        self.gaussian = gaussian
        # send_junk: randomize+send only the FIRST chunk (SEND_MINICHUNK),
        # then send all-zero SEND_JUNK for every subsequent chunk -- a cheap
        # load mode that skips per-chunk randomization.
        self.send_junk = send_junk
        # simulate_frbs: if True, the SimulatedFrameFactory injects simulated FRBs,
        # with parameters derived from each receiver's GetConfig (see _frb_kwargs).
        self.simulate_frbs = simulate_frbs
        # frb_gap_sec: extra padding (seconds) between consecutive FRBs on a beam,
        # passed through to the SimulatedFrameFactory (see _frb_kwargs). 0 = none.
        self.frb_gap_sec = frb_gap_sec
        # sifter_addr: if not None, an 'ip:port'; each receiver sends its simulated FRB
        # events (from_simulator=True) to an FrbSifter there. Requires simulate_frbs.
        self.sifter_addr = sifter_addr
        if sifter_addr is not None and not simulate_frbs:
            raise RuntimeError("run_fake_xengine: a sifter address (-s) requires FRB "
                               "simulation (-f); there are no events to send otherwise")
        self.hw = Hardware()
        # Running base for globally-unique beam_ids: each receiver gets a contiguous
        # block [base, base+nbeams), advanced in _build_one so the blocks are disjoint
        # across servers (downstream consumers generally assume beam_ids are unique).
        self._beam_id_base = 0
        self.fake_xengines = []   # parallel to rpc_addrs (Phase 1 fills this)
        self.allocators = []      # parallel: AssembledFrameAllocator per receiver
        self.factories = []       # parallel: SimulatedFrameFactory per receiver
        self.sifters = []         # parallel: FrbSifterClient (or None) per receiver
        self.fxe_vcpus = []       # parallel
        self.controllers = []     # parallel (Phase 2 fills this)
        self.exc_list = []        # list[(rpc_addr, BaseException)]
        self.exc_lock = threading.Lock()

    def run(self):
        """Top-level lifecycle: Phase 1 build, Phase 2 spawn, wait, cleanup.

        If any controller errors out, the cascade in _controller_wrapper
        stops every FakeXEngine; the main thread then surfaces the first
        non-cascade exception.
        """
        try:
            self._build_all()
            self._spawn_all_controllers()
            print(f"\nAll {len(self.rpc_addrs)} FakeXEngine(s) running. "
                  f"Press Ctrl-C to stop.")
            self._wait_for_controllers()
            self._stop_and_join_all()
            self._surface_real_exceptions()
        finally:
            # Defensive cleanup -- harmless if the main path already ran it.
            self._stop_all()
            # Close the sifter channels after the controllers have been stopped/joined (above),
            # so close()'s brief drain covers the last in-flight async sends.
            self._close_sifters()
            print("All FakeXEngine(s) stopped.")

    def _build_all(self):
        # Doing all per-receiver builds before spawning any controllers means
        # a misconfigured/unreachable receiver fails fast, with no half-built
        # state running in the background.
        for beamset, rpc_addr in enumerate(self.rpc_addrs):
            self._build_one(beamset, rpc_addr)

    def _build_one(self, beamset, rpc_addr):
        """Per-receiver Phase 1: GetConfig -> XMD -> NIC checks -> FakeXEngine.

        'beamset' is this receiver's index in rpc_addrs, stamped into the
        XEngineMetadata so that, with multiple servers, each one advertises a
        distinct beam-set id.
        """
        print(f"\n[{rpc_addr}] Connecting ...")
        with FrbSearchClient(rpc_addr) as c:
            cfg = c.config
        if not list(cfg.data_ip_addrs):
            raise RuntimeError(f"[{rpc_addr}] reported empty data_ip_addrs")

        # Synthesize XEngineMetadata from the receiver's prefilled config.
        # Assign this receiver a contiguous block of globally-unique beam_ids,
        # [base, base+nbeams); the running base makes the blocks disjoint across
        # servers (robust even if fake_nbeams differs per receiver), since
        # downstream consumers generally assume beam_ids are unique. The receiver
        # records whatever we send, so there's no cross-X-engine consensus to satisfy.
        base = self._beam_id_base
        beam_ids = list(range(base, base + cfg.fake_nbeams))
        self._beam_id_base += cfg.fake_nbeams
        xmd = XEngineMetadata.make_fiducial(
            list(cfg.fake_zone_nfreq),
            list(cfg.fake_zone_freq_edges),
            beam_ids,
            cfg.fake_time_sample_ms,
        )
        # make_fiducial() defaults beamset=0; override it with this receiver's
        # index so that, with multiple servers, each one advertises a distinct
        # beam-set id. (Both the beamset and the disjoint beam_ids block above
        # identify a receiver's beams downstream.)
        xmd.beamset = beamset
        # Anchor the X-engine clock (unix-timestamp-at-fpga-seq-0) to the
        # current UTC wall-clock time.
        now = datetime.now(timezone.utc)
        xmd.unix_ns_at_seq_0 = int(now.timestamp() * 1e9)

        self._print_receiver_details(rpc_addr, cfg, xmd)
        vcpu_list = self._verify_nics_and_mtu(rpc_addr, cfg)

        # Size a real (non-dummy) slab pool large enough for a handful of
        # AssembledFrameSets. The pool must hold, at once: the factory's output
        # queue (frame_set_queue_size), the set the factory is currently
        # randomizing (1), and the sets the send loop pins (~2-3, held by the
        # FakeXEngine command queue until their minichunks are sent). Sizing it
        # above that keeps the queue bound -- not the pool -- as the effective
        # lookahead limit. A set's slabs return to the pool when its last
        # reference drops.
        frame_set_queue_size = 4
        n_sets = frame_set_queue_size + 4
        capacity = n_sets * self._frame_set_nbytes(cfg, xmd)

        # Size the SimulatedFrameFactory's randomizer-thread pool: min(nbeams, num_vcpus/2).
        # num_vcpus is the data-NIC vcpu affinity (vcpu_list) that the factory's threads
        # inherit; the /2 leaves headroom for the FakeXEngine worker threads sharing those
        # vcpus. The factory requires >= 1 randomizer thread.
        num_randomizer_threads = min(cfg.fake_nbeams, len(vcpu_list) // 2)
        if num_randomizer_threads < 1:
            raise RuntimeError(
                f"[{rpc_addr}] num_randomizer_threads = {num_randomizer_threads} ="
                f" min(nbeams={cfg.fake_nbeams}, num_vcpus/2={len(vcpu_list) // 2});"
                f" requires nbeams >= 1 and a data-NIC vcpu affinity of size >= 2")

        with ThreadAffinity(vcpu_list):
            fxe = FakeXEngine(
                xmd, list(cfg.data_ip_addrs), self.nworkers,
                time_samples_per_chunk=cfg.time_samples_per_chunk,
                paced=self.paced, rpc_address=rpc_addr)

            # SlabAllocator + AssembledFrameAllocator to source the
            # AssembledFrameSets, then a SimulatedFrameFactory that pre-randomizes
            # them on a producer + randomizer-thread pool. All built inside the
            # ThreadAffinity context so the backing allocation, the allocator's
            # pre-init thread, and the factory's producer/randomizer threads all
            # land on the data-NIC CPU. One consumer: this receiver's controller
            # thread (which pulls via factory.get_frame_set()).
            slab_allocator = SlabAllocator("af_rhost", capacity)
            allocator = AssembledFrameAllocator(
                slab_allocator,
                num_consumers=1,
                time_samples_per_chunk=cfg.time_samples_per_chunk,
            )
            allocator.initialize_metadata(xmd)
            allocator.initialize_initial_chunk(0)

            # The factory is the sole consumer of 'allocator' and propagates
            # stop() to it. Must be constructed after the allocator is
            # initialized (above), and inside this ThreadAffinity context.
            factory = SimulatedFrameFactory(
                allocator,
                num_randomizer_threads=num_randomizer_threads,
                normalized=self.normalized,
                gaussian=self.gaussian,
                frame_set_queue_size=frame_set_queue_size,
                **self._frb_kwargs(cfg, xmd, num_randomizer_threads),
            )

        # Per-receiver sifter client (one gRPC channel per beamset). The channel is created
        # here but not connected; the first send (in the controller) is what actually reaches
        # the sifter. None when -s was not given.
        sifter = FrbSifterClient(self.sifter_addr) if (self.sifter_addr is not None) else None

        self.fake_xengines.append(fxe)
        self.allocators.append(allocator)
        self.factories.append(factory)
        self.sifters.append(sifter)
        self.fxe_vcpus.append(vcpu_list)
        paced_note = "paced" if self.paced else "unpaced"
        norm_note = "normalized" if self.normalized else "unnormalized"
        data_note = "gaussian" if self.gaussian else "uniform"
        frb_note = ""
        if self.simulate_frbs:
            gap_note = f", gap={factory.frb_gap_sec:g} s" if (factory.frb_gap_sec > 0) else ""
            frb_note = (f", FRBs (max_dm={factory.frb_max_dm:.0f}, "
                        f"max_width={factory.frb_max_width_ms:.2f} ms, "
                        f"N_subbands={len(factory.frb_subband_fmin_MHz)}{gap_note})")
        sifter_note = f", sifter={self.sifter_addr}" if (sifter is not None) else ""
        print(f"[{rpc_addr}] FakeXEngine started ({self.nworkers} workers, "
              f"{paced_note}, {norm_note}, {data_note}{frb_note}{sifter_note}).")

    def _frb_kwargs(self, cfg, xmd, num_randomizer_threads):
        """SimulatedFrameFactory FRB-injection kwargs, or {} when --frbs is off.

        Derived from the receiver's GetConfig: the DM reach (max_dm_of_all_trees),
        the base-tree peak-finding width (max_width_of_base_tree, in frame samples),
        and the frequency subbands (frequency_subband_counts + the band edges).
        """
        if not self.simulate_frbs:
            return {}

        # Actual frame time-sample length in ms. make_fiducial() rounds
        # seq_per_frb_time_sample to an integer, so this differs slightly from
        # cfg.fake_time_sample_ms; max_width_of_base_tree counts frame samples, so
        # the actual per-sample duration is the right conversion factor.
        dt_ms = xmd.dt_ns_per_seq * xmd.seq_per_frb_time_sample / 1.0e6

        # Frequency subbands: build a FrequencySubbands over the full band, then map
        # each subband's coarse-freq index pair to physical (fmin, fmax). f_to_freq
        # is DECREASING (index 0 = fmax = high freq), and n_to_flo < n_to_fhi are
        # coarse indices, so the smaller index (n_to_flo) is the HIGHER freq -> fmax,
        # and the larger index (n_to_fhi) is the LOWER freq -> fmin (the lo/hi swap).
        band_fmin = cfg.fake_zone_freq_edges[0]     # zone_freq_edges is increasing
        band_fmax = cfg.fake_zone_freq_edges[-1]
        fs = FrequencySubbands(list(cfg.frequency_subband_counts), band_fmin, band_fmax)
        fmin_list = [fs.f_to_freq[fs.n_to_fhi[n]] for n in range(fs.N)]
        fmax_list = [fs.f_to_freq[fs.n_to_flo[n]] for n in range(fs.N)]

        return dict(
            simulate_frbs=True,
            frb_dm0=50.0,
            frb_max_dm=cfg.max_dm_of_all_trees,
            frb_max_width_ms=cfg.max_width_of_base_tree * dt_ms,
            frb_snr=30.0,
            frb_subband_fmin_MHz=fmin_list,
            frb_subband_fmax_MHz=fmax_list,
            frb_gap_sec=self.frb_gap_sec,
            num_frb_simulator_threads=num_randomizer_threads,
            single_pulse_queue_size=cfg.fake_nbeams,
        )

    @staticmethod
    def _frame_set_nbytes(cfg, xmd):
        """Backing bytes for one AssembledFrameSet (= nbeams frames).

        Delegates the per-frame slab arithmetic to AssembledFrameAllocator's
        static slab_nbytes() so this stays in lockstep with
        AssembledFrameAllocator::_create_frame_set (single source of truth).
        """
        per_frame = AssembledFrameAllocator.slab_nbytes(
            xmd.get_total_nfreq(), cfg.time_samples_per_chunk)
        return cfg.fake_nbeams * per_frame

    def _print_receiver_details(self, rpc_addr, cfg, xmd):
        ip_addrs = list(cfg.data_ip_addrs)
        actual_time_sample_ms = (xmd.dt_ns_per_seq * xmd.seq_per_frb_time_sample) / 1.0e6
        print(f"[{rpc_addr}]   data_ip_addrs = {ip_addrs}")
        print(f"[{rpc_addr}]   time_samples_per_chunk = {cfg.time_samples_per_chunk}")
        print(f"[{rpc_addr}]   min_data_mtu = {cfg.min_data_mtu}")
        print(f"[{rpc_addr}]   beamset = {xmd.beamset}, nbeams = {cfg.fake_nbeams}, total_nfreq = {xmd.get_total_nfreq()}")
        print(f"[{rpc_addr}]   time_sample_ms = {actual_time_sample_ms}")
        print(f"[{rpc_addr}]   dt_ns_per_seq = {xmd.dt_ns_per_seq}")
        print(f"[{rpc_addr}]   seq_per_frb_time_sample = {xmd.seq_per_frb_time_sample}")

    def _verify_nics_and_mtu(self, rpc_addr, cfg):
        """Check MTU on each data NIC and verify all NICs are on one CPU.
        Returns the (consistent) vcpu_list for use with ThreadAffinity.
        """
        ip_addrs = list(cfg.data_ip_addrs)
        vcpu_list = None
        first_cpu = None
        for addr in ip_addrs:
            ip = extract_ip(addr)
            vl = self.hw.vcpu_list_from_ip_addr(ip, is_dst_addr=True)
            check_mtu(self.hw, f"[{rpc_addr}] FakeXEngine -> {addr}", ip,
                      cfg.min_data_mtu, 'min_data_mtu', is_dst_addr=True)
            cpu = self.hw.cpu_from_vcpu_list(vl)
            if vcpu_list is None:
                vcpu_list = vl
                first_cpu = cpu
            elif cpu != first_cpu:
                raise RuntimeError(
                    f"[{rpc_addr}] destination IPs {ip_addrs} route through "
                    f"NICs on different CPUs (need all on one CPU)"
                )
        return vcpu_list

    def _spawn_all_controllers(self):
        # Spawning all controllers after Phase 1 completes means the cascade
        # in _controller_wrapper sees the complete self.fake_xengines list
        # the moment the first controller runs.
        for rpc_addr, fxe, factory, sifter, vcpu_list in zip(
                self.rpc_addrs, self.fake_xengines, self.factories,
                self.sifters, self.fxe_vcpus):
            with ThreadAffinity(vcpu_list):
                t = threading.Thread(
                    target=self._controller_wrapper,
                    args=(rpc_addr, fxe, factory, sifter),
                    daemon=True,
                )
                t.start()
            self.controllers.append(t)

    def _controller_wrapper(self, rpc_addr, fxe, factory, sifter):
        """Wraps _controller_main. On exit (normal or exceptional),
        cascade-stops every FakeXEngine, SimulatedFrameFactory, and allocator so
        sibling controllers exit promptly (via "called on stopped instance").
        Captures any exception (paired with its rpc_addr) under self.exc_lock for
        the main thread to surface.
        """
        try:
            self._controller_main(rpc_addr, fxe, factory, sifter)
        except BaseException as e:
            with self.exc_lock:
                self.exc_list.append((rpc_addr, e))
        finally:
            self._stop_all()

    def _controller_main(self, rpc_addr, fxe, factory, sifter):
        """Drive one FakeXEngine in 'send-random-frames-forever' mode.

        A SimulatedFrameFactory (its producer + randomizer-thread pool)
        pre-randomizes whole AssembledFrameSets ("frames" = time chunks, each
        mpc = minichunks_per_chunk wire minichunks) ahead of this send loop; the
        loop pulls them via factory.get_frame_set(). How far ahead is bounded by
        the factory's output queue and the slab pool. This send loop works in
        minichunks: for minichunk n it stays up to 2 minichunks ahead of every
        worker (the "minichunk N waits for (N-2)" serialization the C++
        FakeXEngine used to enforce with its barrier), sending real randomized
        data rather than all-zero junk.

        At each chunk boundary (n % mpc == 0) it pulls the next randomized frame
        from the factory. It deliberately keeps no reference to a frame beyond
        the chunk it is currently sending -- the C++ command queue holds the
        frame alive until its minichunks are processed, after which the slabs
        return to the pool.

        send-junk mode (self.send_junk): only the first chunk (chunk 0) carries
        real randomized data via SEND_MINICHUNK; every subsequent chunk is sent
        as all-zero SEND_JUNK (no frame set needed). The loop therefore pulls
        from the factory only at the chunk-0 boundary.

        Runs until fxe.stop() or factory.stop() is called (from anywhere), at
        which point the next factory.get_frame_set() / wait_until_processed() /
        enqueue_send_minichunk() raises "called on stopped instance", ending the
        loop (filtered as benign by _surface_real_exceptions).
        """
        nworkers = fxe.nworkers
        mpc = fxe.minichunks_per_chunk
        send_junk = self.send_junk

        fset = None
        beam_set_id = None      # set (with the sifter ConfigMessage) on the first chunk
        n = 0
        while True:
            chunk = n // mpc
            # Pull the next pre-randomized frame at chunk boundaries. In
            # send-junk mode only chunk 0 carries real data, so we pull (and
            # block on the factory) only there. On shutdown get_frame_set()
            # raises "called on stopped instance", which ends the loop.
            if (n % mpc) == 0 and not (send_junk and chunk >= 1):
                fset = factory.get_frame_set()

                # Per-chunk summary line, printed whether or not FRBs are simulated
                # (with -f, one line per injected FRB follows below). The FPGA window
                # is this chunk's own [tci*seq_per_chunk, (tci+1)*seq_per_chunk).
                seq_per_chunk = fset.ntime * fset.metadata.seq_per_frb_time_sample
                tci = fset.time_chunk_index
                fpga_start, fpga_end = tci * seq_per_chunk, (tci + 1) * seq_per_chunk
                print(f"fake_xengine: beamset={fset.metadata.beamset}, ichunk={tci}, "
                      f"fpga=[{fpga_start}:{fpga_end}]", flush=True)

                # With -f, drain and print this chunk's simulated-FRB injection events (the
                # C++ producer records them; the printing lives here in Python), and with -s
                # forward them to the sifter. The factory runs ahead of this send loop, so
                # pop_events() may return events injected into a slightly later chunk than
                # 'fset'; the window passed is fset's own chunk.
                if self.simulate_frbs:
                    # On the first chunk, send the one-time sifter ConfigMessage (only xengine_yaml
                    # is meaningful; pirate/plan/grouper/search are empty for simulator events).
                    if sifter is not None and beam_set_id is None:
                        sifter.send_configuration(
                            pirate_yaml="", xengine_yaml=fset.metadata.to_yaml_string(),
                            dedispersion_plan_yaml="", grouper_yaml="", search_ip_addr="")
                        beam_set_id = fset.metadata.beamset
                        print(f"[{rpc_addr}] sent ConfigMessage to sifter at "
                              f"{sifter.server_address}", flush=True)

                    events = factory.pop_events(fpga_start, fpga_end)
                    self._print_events(events)

                    if sifter is not None:
                        # coarsegrain_snr: per-beam max event SNR this chunk (0 for beams with no
                        # events), indexed by LOCAL beam index in [0, nbeams). This receiver's
                        # beam_ids are a contiguous globally-unique block, so the local index is
                        # beam_id - base (base = this receiver's first beam_id).
                        base = int(fset.metadata.beam_ids[0])
                        cg_snr = np.zeros(fset.nbeams, dtype=np.float32)
                        np.maximum.at(cg_snr, events.beam_ids - base, events.snrs)
                        sifter.send_events(beam_set_id, events, cg_snr, from_simulator=True)

            # Stay <= 2 minichunks ahead of every worker. For n < 2 the target
            # is negative, so wait_until_processed returns immediately
            # (per-worker last_processed_minichunk is -1).
            for w in range(nworkers):
                fxe.wait_until_processed(w, n - 2)

            # Send minichunk n on every worker. For SEND_MINICHUNK all workers
            # reference the same chunk fset (each gathers its own freq-channel
            # subset); the C++ command queue holds a reference to fset until its
            # minichunks are processed, so it stays alive across the chunk even
            # after our reference is gone. In send-junk mode, chunks >= 1 send
            # all-zero junk instead.
            use_junk = send_junk and (chunk >= 1)
            for w in range(nworkers):
                if use_junk:
                    fxe.enqueue_send_junk(w, n)
                else:
                    fxe.enqueue_send_minichunk(w, n, fset)
            n += 1

    @staticmethod
    def _print_events(events):
        """Print one indented line per injected FRB, below the per-chunk summary line.

        (This printing was moved from C++ to Python.) 'events' is the FrbSifterEvents
        returned by SimulatedFrameFactory.pop_events(); the summary line above it is
        printed by the controller (whether or not FRBs are simulated).
        """
        for i in range(len(events)):
            print(f"    injected FRB: beam_id={int(events.beam_ids[i])}, "
                  f"dm={float(events.dms[i]):.4g}, "
                  f"fpga_timestamp={int(events.fpga_timestamps[i])}, "
                  f"width={float(events.widths_ms[i]):.4g} ms, "
                  f"subband=[{float(events.subband_freqs_lo_MHz[i]):.1f},"
                  f"{float(events.subband_freqs_hi_MHz[i]):.1f}] MHz, "
                  f"snr={float(events.snrs[i]):.4g}", flush=True)

    def _wait_for_controllers(self):
        try:
            while any(t.is_alive() for t in self.controllers):
                time.sleep(0.2)
        except KeyboardInterrupt:
            print("\nStopping...")

    def _stop_and_join_all(self):
        self._stop_all()
        for t in self.controllers:
            t.join(timeout=5.0)

    def _surface_real_exceptions(self):
        with self.exc_lock:
            # A "called on stopped instance" RuntimeError is the controller's
            # natural teardown artefact (or cascade artefact across receivers
            # when one of them errors); re-raise only "real" exceptions.
            real = [(addr, e) for (addr, e) in self.exc_list
                    if "called on stopped instance" not in str(e)]
            if real:
                raise real[0][1]

    def _stop_all(self):
        """Idempotent: stop every FakeXEngine, SimulatedFrameFactory, and
        AssembledFrameAllocator.

        Stopping the factories is what unblocks a controller parked in
        factory.get_frame_set(), and a factory producer parked in the allocator's
        get_frame_set() on an exhausted pool (factory.stop() also stops its
        allocator). We also stop the allocators directly for defense in depth
        (idempotent). Called from each controller's finally and defensively from
        run()'s finally.
        """
        for fxe in self.fake_xengines:
            try:
                fxe.stop()
            except Exception:
                pass
        for factory in self.factories:
            try:
                factory.stop()
            except Exception:
                pass
        for allocator in self.allocators:
            try:
                allocator.stop()
            except Exception:
                pass

    def _close_sifters(self):
        """Close every FrbSifterClient channel (idempotent). Called once from run()'s finally."""
        for i, sifter in enumerate(self.sifters):
            if sifter is None:
                continue
            try:
                sifter.close()
            except Exception:
                pass
            self.sifters[i] = None   # don't double-close on a second pass


def run_fake_xengine(rpc_addrs, nworkers=128, paced=True, normalized=True,
                     gaussian=True, send_junk=False, simulate_frbs=False, sifter_addr=None,
                     frb_gap_sec=0.0):
    """Main entry point for 'pirate_frb run_fake_xengine'.

    For each rpc_addr in rpc_addrs, sends a GetConfig RPC, synthesizes an
    XEngineMetadata, runs the MTU + NIC-CPU consistency checks, and spawns
    one FakeXEngine + SimulatedFrameFactory + controller thread pinned to that
    receiver's data-NIC CPU. Blocks until Ctrl-C or every controller exits.

    Args:
        rpc_addrs: non-empty list[str] of "ip:port" strings (one per
            receiver). Duplicates are rejected, and a bare string is
            rejected (use [addr], not addr).
        nworkers: worker threads per FakeXEngine (not aggregate across
            receivers).
        paced: if True (default), each FakeXEngine spawns a pacing
            thread that subscribes to MonitorRingbuf and gates each
            worker's sends to stay <=5 chunks ahead of server-side
            rb_processed. If False, the sender runs unthrottled.
        normalized: if True (default), the factory calibrates each frame's
            scales/offsets to xmd's per-zone noise variance so the
            dequantized data is "normalized". If False, the normalization
            is arbitrary (scales/offsets are uniform junk).
        gaussian: if True (default), the factory fills the int4 data with
            simulated Gaussian noise clamped to [-7,+7]; if False, with
            uniform int4 over [-8,+7].
        send_junk: if True, only the first chunk is randomized and sent as
            real data (SEND_MINICHUNK); every subsequent chunk is sent as
            all-zero junk (SEND_JUNK), skipping per-chunk randomization. If
            False (default), every chunk is randomized.
        simulate_frbs: if True, the SimulatedFrameFactory injects simulated
            FRBs, with parameters derived from each receiver's GetConfig (DM
            reach, base-tree width, and frequency subbands -- see
            RunFakeXEngineHelper._frb_kwargs). Requires normalized and gaussian
            (the caller rejects the incompatible flags). Default False.
        sifter_addr: if not None, an 'ip:port' string; each receiver sends its
            simulated FRB events (from_simulator=True) to an FrbSifter there
            (with a one-time ConfigMessage carrying only xengine_yaml). Requires
            simulate_frbs. Default None.
        frb_gap_sec: extra padding (seconds, >= 0) between consecutive FRBs on a
            beam, passed to the SimulatedFrameFactory (rounded to whole time
            samples). Only meaningful with simulate_frbs. Default 0 (no extra gap).
    """
    helper = RunFakeXEngineHelper(rpc_addrs, nworkers, paced=paced,
                                  normalized=normalized, gaussian=gaussian,
                                  send_junk=send_junk, simulate_frbs=simulate_frbs,
                                  sifter_addr=sifter_addr, frb_gap_sec=frb_gap_sec)
    helper.run()
