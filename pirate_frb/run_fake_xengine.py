"""Implementation of 'pirate_frb run_fake_xengine' subcommand."""

import queue
import time
import threading

from .Hardware import Hardware
from .utils import ThreadAffinity, extract_ip, check_mtu
from .core import (
    AssembledFrameAllocator,
    FakeXEngine,
    SlabAllocator,
    XEngineMetadata,
)
from .rpc import FrbClient


class RunFakeXEngineHelper:
    """Encapsulates state and logic for 'pirate_frb run_fake_xengine'.

    Constructed once per invocation; call .run() to drive the full
    lifecycle (Phase 1: per-receiver GetConfig + FakeXEngine; Phase 2:
    spawn controllers; wait for Ctrl-C; cleanup).
    """

    def __init__(self, rpc_addrs, nworkers=128, paced=True, send_junk=False,
                 now=False):
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
        # send_junk: randomize+send only the FIRST chunk (SEND_MINICHUNK),
        # then send all-zero SEND_JUNK for every subsequent chunk -- a cheap
        # load mode that skips per-chunk randomization.
        self.send_junk = send_junk
        self.hw = Hardware()
        self.fake_xengines = []   # parallel to rpc_addrs (Phase 1 fills this)
        self.allocators = []      # parallel: AssembledFrameAllocator per receiver
        self.fxe_vcpus = []       # parallel
        self.controllers = []     # parallel (Phase 2 fills this)
        self.exc_list = []        # list[(rpc_addr, BaseException)]
        self.exc_lock = threading.Lock()
        self.now = now

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
            print("All FakeXEngine(s) stopped.")

    def _build_all(self):
        # Doing all per-receiver builds before spawning any controllers means
        # a misconfigured/unreachable receiver fails fast, with no half-built
        # state running in the background.
        for rpc_addr in self.rpc_addrs:
            self._build_one(rpc_addr)

    def _build_one(self, rpc_addr):
        """Per-receiver Phase 1: GetConfig -> XMD -> NIC checks -> FakeXEngine."""
        print(f"\n[{rpc_addr}] Connecting ...")
        with FrbClient(rpc_addr) as c:
            cfg = c.get_config()
        if not list(cfg.data_ip_addrs):
            raise RuntimeError(f"[{rpc_addr}] reported empty data_ip_addrs")

        # Synthesize XEngineMetadata from the receiver's prefilled config.
        # beam_ids = {0, 1, ..., nbeams-1} -- the receiver records whatever
        # we send, so no cross-X-engine consensus to satisfy.
        beam_ids = list(range(cfg.fake_nbeams))
        xmd = XEngineMetadata.make_fiducial(
            list(cfg.fake_zone_nfreq),
            list(cfg.fake_zone_freq_edges),
            beam_ids,
            cfg.fake_time_sample_ms,
        )
        if self.now:
            from datetime import datetime
            now = datetime.utcnow()
            xmd.unix_ns_at_seq_0 = int(now.timestamp() * 1e9)

        self._print_receiver_details(rpc_addr, cfg, xmd)
        vcpu_list = self._verify_nics_and_mtu(rpc_addr, cfg)

        # Size a real (non-dummy) slab pool large enough for a handful of
        # AssembledFrameSets. The pool depth is what bounds how far the frame
        # provider can run ahead of the send loop: get_frame_set() blocks once
        # the pool is exhausted, and a set's slabs return to the pool when its
        # last reference drops (after its minichunks have been sent). A few
        # sets is ample -- the send loop pins at most ~2-3 at once.
        n_sets = 5
        capacity = n_sets * self._frame_set_nbytes(cfg, xmd)

        with ThreadAffinity(vcpu_list):
            fxe = FakeXEngine(
                xmd, list(cfg.data_ip_addrs), self.nworkers,
                time_samples_per_chunk=cfg.time_samples_per_chunk,
                paced=self.paced, rpc_address=rpc_addr)

            # SlabAllocator + AssembledFrameAllocator to source the
            # AssembledFrameSets we randomize and send. Built inside the
            # ThreadAffinity context so the backing allocation, the allocator's
            # pre-init thread, and FakeXEngine's randomizer threads all land on
            # the data-NIC CPU. One consumer: this receiver's controller thread.
            slab_allocator = SlabAllocator("af_rhost", capacity)
            allocator = AssembledFrameAllocator(
                slab_allocator,
                num_consumers=1,
                time_samples_per_chunk=cfg.time_samples_per_chunk,
            )
            allocator.initialize_metadata(xmd)
            allocator.initialize_initial_chunk(0)

        self.fake_xengines.append(fxe)
        self.allocators.append(allocator)
        self.fxe_vcpus.append(vcpu_list)
        paced_note = "paced" if self.paced else "unpaced"
        print(f"[{rpc_addr}] FakeXEngine started ({self.nworkers} workers, {paced_note}).")

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
        print(f"[{rpc_addr}]   nbeams = {cfg.fake_nbeams}, total_nfreq = {xmd.get_total_nfreq()}")
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
        for rpc_addr, fxe, allocator, vcpu_list in zip(
                self.rpc_addrs, self.fake_xengines, self.allocators,
                self.fxe_vcpus):
            with ThreadAffinity(vcpu_list):
                t = threading.Thread(
                    target=self._controller_wrapper,
                    args=(rpc_addr, fxe, allocator),
                    daemon=True,
                )
                t.start()
            self.controllers.append(t)

    def _controller_wrapper(self, rpc_addr, fxe, allocator):
        """Wraps _controller_main. On exit (normal or exceptional),
        cascade-stops every FakeXEngine AND allocator so sibling controllers
        exit promptly (via "called on stopped instance", or via their
        provider's None sentinel once get_frame_set() is unblocked). Captures
        any exception (paired with its rpc_addr) under self.exc_lock for the
        main thread to surface.
        """
        try:
            self._controller_main(fxe, allocator)
        except BaseException as e:
            with self.exc_lock:
                self.exc_list.append((rpc_addr, e))
        finally:
            self._stop_all()

    def _frame_provider_main(self, fxe, allocator, ready, stop_event,
                             exc_holder, max_sets=None):
        """Dedicated thread that hands the send loop pre-randomized frames.

        Pulls AssembledFrameSets from the allocator (chunk 0, 1, 2, ... in
        order), randomizes each via FakeXEngine.randomize_frames(), and hands
        them to the send loop through the 'ready' queue. Both get_frame_set()
        and randomize_frames() release the GIL, so this thread's allocation +
        randomization run concurrently with -- and do not stall -- the send
        loop.

        If max_sets is not None, it produces at most that many sets and then
        exits (used in send-junk mode, max_sets=1, where only the first chunk
        carries real data).

        No explicit producer/consumer coordination: backpressure comes from
        the bounded slab pool. get_frame_set() blocks once the pool is
        exhausted, and a set's slabs return to the pool when its last
        reference drops (after the send loop and the C++ command queue are
        done with it). So this thread naturally runs at most ~pool-depth
        chunks ahead.

        On shutdown (stop_event set; allocator.stop() unblocking a blocked
        get_frame_set(); or randomize_frames() raising because fxe stopped)
        it pushes a None sentinel so a sender blocked on ready.get() wakes up.
        """
        try:
            produced = 0
            while not stop_event.is_set():
                if max_sets is not None and produced >= max_sets:
                    break
                # Blocks here when the slab pool is full (backpressure);
                # allocator.stop() unblocks it with an exception on shutdown.
                fset = allocator.get_frame_set(consumer_id=0)
                fxe.randomize_frames(fset)   # GIL released during the fill
                ready.put(fset)              # unbounded; pool bounds liveness
                produced += 1
        except Exception as e:
            # allocator/fxe stopped (get_frame_set / randomize_frames raise
            # "called on stopped instance") or a genuine error. Stash it so the
            # send loop can re-raise on the None sentinel; _surface_real_exceptions
            # filters out the benign "called on stopped instance" case.
            exc_holder.append(e)
        finally:
            stop_event.set()
            ready.put(None)   # wake a sender blocked on ready.get()

    def _controller_main(self, fxe, allocator):
        """Drive one FakeXEngine in 'send-random-frames-forever' mode.

        A dedicated provider thread (_frame_provider_main) randomizes whole
        AssembledFrameSets ("frames" = time chunks, each mpc =
        minichunks_per_chunk wire minichunks) ahead of the send loop and hands
        them over via the 'ready' queue. How far ahead is bounded by the slab
        pool, not by explicit coordination. This send loop works in minichunks:
        for minichunk n it stays up to 2 minichunks ahead of every worker (the
        "minichunk N waits for (N-2)" serialization the C++ FakeXEngine used to
        enforce with its barrier), now sending real randomized data rather than
        all-zero junk.

        At each chunk boundary (n % mpc == 0) it pulls the next randomized
        frame from the provider. It deliberately keeps no reference to a frame
        beyond the chunk it is currently sending -- the C++ command queue holds
        the frame alive until its minichunks are processed, after which the
        slabs return to the pool.

        send-junk mode (self.send_junk): only the first chunk (chunk 0) carries
        real randomized data via SEND_MINICHUNK; every subsequent chunk is sent
        as all-zero SEND_JUNK (no frame set needed). The provider therefore
        produces exactly one frame, and the loop only pulls from 'ready' at the
        chunk-0 boundary.

        Runs until fxe.stop() is called (from anywhere), at which point the next
        wait_until_processed() / enqueue_send_minichunk() raises RuntimeError,
        or the provider's None sentinel ends the loop.
        """
        nworkers = fxe.nworkers
        mpc = fxe.minichunks_per_chunk
        send_junk = self.send_junk

        ready = queue.Queue()                  # provider -> sender handoff
        stop_event = threading.Event()
        exc_holder = []                        # provider's exception, if any
        # In send-junk mode we only need the first chunk randomized.
        max_sets = 1 if send_junk else None
        provider = threading.Thread(
            target=self._frame_provider_main,
            args=(fxe, allocator, ready, stop_event, exc_holder, max_sets),
            daemon=True,
        )
        # Inherits this (already pinned) thread's vcpu affinity.
        provider.start()

        try:
            fset = None
            n = 0
            while True:
                chunk = n // mpc
                # Pull the next pre-randomized frame at chunk boundaries. In
                # send-junk mode only chunk 0 carries real data, so we pull
                # (and block on the provider) only there. A None sentinel means
                # the provider has stopped -> end the loop.
                if (n % mpc) == 0 and not (send_junk and chunk >= 1):
                    fset = ready.get()
                    if fset is None:
                        # Provider exited. Re-raise a genuine provider error;
                        # a benign "called on stopped instance" is filtered by
                        # _surface_real_exceptions.
                        if exc_holder:
                            raise exc_holder[0]
                        return

                # Stay <= 2 minichunks ahead of every worker. For n < 2 the
                # target is negative, so wait_until_processed returns
                # immediately (per-worker last_processed_minichunk is -1).
                for w in range(nworkers):
                    fxe.wait_until_processed(w, n - 2)

                # Send minichunk n on every worker. For SEND_MINICHUNK all
                # workers reference the same chunk fset (each gathers its own
                # freq-channel subset); the C++ command queue holds a reference
                # to fset until its minichunks are processed, so it stays alive
                # across the chunk even after the provider's reference is gone.
                # In send-junk mode, chunks >= 1 send all-zero junk instead.
                use_junk = send_junk and (chunk >= 1)
                for w in range(nworkers):
                    if use_junk:
                        fxe.enqueue_send_junk(w, n)
                    else:
                        fxe.enqueue_send_minichunk(w, n, fset)
                n += 1
        finally:
            # Stop the provider. allocator.stop() is what unblocks a provider
            # parked in get_frame_set() on an exhausted pool (fxe.stop() alone
            # would not). Note the helper-level cleanup also stops the allocator
            # -- needed because this controller can itself be parked in
            # ready.get() waiting on such a provider, in which case it never
            # reaches this finally until the allocator is stopped from outside.
            stop_event.set()
            try:
                allocator.stop()
            except Exception:
                pass
            provider.join(timeout=5.0)

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
        """Idempotent: stop every FakeXEngine and every AssembledFrameAllocator.

        Stopping the allocators (not just the fxes) is essential for clean
        shutdown: a controller can be parked in ready.get() waiting on a
        provider parked in get_frame_set() on an exhausted pool, and only
        allocator.stop() unblocks that provider (which then feeds the
        controller its None sentinel). Also called defensively from `finally:`.
        """
        for fxe in self.fake_xengines:
            try:
                fxe.stop()
            except Exception:
                pass
        for allocator in self.allocators:
            try:
                allocator.stop()
            except Exception:
                pass


def run_fake_xengine(rpc_addrs, nworkers=128, paced=True, send_junk=False, now=False):
    """Main entry point for 'pirate_frb run_fake_xengine'.

    For each rpc_addr in rpc_addrs, sends a GetConfig RPC, synthesizes an
    XEngineMetadata, runs the MTU + NIC-CPU consistency checks, and spawns
    one FakeXEngine + controller thread pinned to that receiver's data-NIC
    CPU. Blocks until Ctrl-C or every controller exits.

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
        send_junk: if True, only the first chunk is randomized and sent as
            real data (SEND_MINICHUNK); every subsequent chunk is sent as
            all-zero junk (SEND_JUNK), skipping per-chunk randomization. If
            False (default), every chunk is randomized.
    """
    helper = RunFakeXEngineHelper(rpc_addrs, nworkers, paced=paced,
                                  send_junk=send_junk, now=now)
    helper.run()
