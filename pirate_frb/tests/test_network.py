"""
Network test: FakeXEngine -> FrbServer over 127.0.0.1 loopback.

Constructs an FrbServer (dummy-mode allocator) and a FakeXEngine
(debug=True) in a single Python process, runs a randomized 1000-turn
send loop that produces ragged per-worker progress, and verifies that
none of the real-time debug-mode asserts trigger. Random subscale
parameters per call (see NetworkTester._random_params).

Goals of the network test:

 - end-to-end test of receive -> write -> read code, by sending random data
      to the server, writing to disk, reading it back, and checking that
      contents are as expected.

 - stress-test data-dropping/eviction logic, in situations where some clients
     are ahead of others.

 - test handling of "skipped" minichunks (allowed by network protocol, see
     NOTE 1 in notes/network_protocol.md).

 - test (initial sequence number != 0), see NOTE 2.

 - test short-read logic (via Socket.misbehaving_reads).

 - test disconnect/reconnect logic and "missing" clients.

Run via: python -m pirate_frb test --net
"""

import os
import time
import random
import secrets
import shutil

import grpc
import ksgpu
import numpy as np

from ..core import (
    AssembledFrame,
    AssembledFrameAllocator,
    BumpAllocator,
    FakeXEngine,
    FileWriter,
    Receiver,
    SlabAllocator,
    XEngineMetadata,
)
from ..pirate_pybind11 import FrbServer, constants
from ..rpc import FrbSearchClient
from ..rpc.grpc import frb_search_pb2


def _acq_filename(acqdir, beam_id, chunk):
    """Client-side mirror of the server's fixed naming scheme
    (make_acq_relpath in src_lib/FileWriter.cpp):
    {acqdir}/frame_b{beam_id}_t{chunk}.asdf, unpadded decimal."""
    return f"{acqdir}/frame_b{beam_id}_t{chunk}.asdf"
from .utils import make_random_subscale_config, pick_receiver_worker_counts


class NetworkTester:
    """Drives one FakeXEngine -> FrbServer loopback test.

    Use as a context manager: __init__ picks random params (no side
    effects), __enter__ creates /dev/shm subdirs and starts the server +
    client threads, __exit__ stops everything and rmtree's. run()
    sequences the test's post-build phases (send loop, sync, drain,
    summary, readback verification).
    """

    # ---- Construction + lifecycle ----

    @staticmethod
    def _random_params():
        """Return one random subscale config (a plain dict).

        The DedispersionConfig is built first, and the rest of the test params
        are derived from it -- see tests/utils.py (make_random_subscale_config,
        pick_receiver_worker_counts) for the config-first rationale and the
        FakeXEngine worker-count constraints.
        """
        config = make_random_subscale_config()
        total_nfreq = sum(config.zone_nfreq)
        num_receivers, nworkers = pick_receiver_worker_counts(total_nfreq)
        # Exercise the FakeXEngine pacing path 20% of the time. Default
        # paced=False because pacing throttles the sender, which masks
        # the real-time eviction races and FLAG_ACK ack-prediction
        # checks that this test was designed for.
        paced = (random.random() < 0.2)
        # Exercise the FrbServer's --no-dedispersion processing path 80% of
        # the time. In this mode the processing thread skips ALL GPU work
        # (no host->device copies, no dequant/dedispersion kernels), but
        # still consumes each assembled frame so rb_processed advances, and
        # the receive/assemble/ringbuf/reaper/file-writing pipeline -- which
        # is what this test actually verifies -- runs in full. Implies no
        # grouper, which holds trivially since the test never sets one.
        no_dedispersion = (random.random() < 0.8)
        return dict(
            config                 = config,
            num_receivers          = num_receivers,
            nworkers               = nworkers,
            num_ssd_threads        = random.randint(1, 5),
            num_nfs_threads        = random.randint(1, 5),
            time_samples_per_chunk = config.time_samples_per_chunk,
            nbeams                 = config.beams_per_gpu,
            total_nfreq            = total_nfreq,
            base_beam_id           = random.randint(0, 10000),
            data_base_port         = 5000,
            rpc_port               = 6000,
            ringbuf_nchunks        = random.randint(50, 100),
            paced                  = paced,
            no_dedispersion        = no_dedispersion,
        )

    def __init__(self):
        # Random parameters + derived paths. No side effects yet (no
        # threads, no directories) -- those happen in __enter__.
        self.p = NetworkTester._random_params()
        self.run_id = secrets.token_hex(8)
        self.ssd_dir = f"/dev/shm/pirate_test_network_ssd_{self.run_id}"
        self.nfs_dir = f"/dev/shm/pirate_test_network_nfs_{self.run_id}"

        # Filled by __enter__'s builders. Listed here so __exit__ can
        # blindly None-check even if __enter__ failed partway through.
        # Server side.
        self.allocator   = None
        self.file_writer = None
        self.receivers   = None
        self.server      = None
        # Client side.
        self.xmd              = None
        self.fxe              = None
        self.client_allocator = None
        self.nworkers         = None
        self.mpc              = None
        self.beam_ids         = None
        self.ipos             = None
        self.wpos             = None
        self.dstate           = None
        self.skipped          = None
        self.framesets        = None
        self.fspos            = None
        # RPC + tracking.
        self.rpc_client             = None
        self.file_sub               = None
        self.stream_sub             = None
        self.safe_written_set       = None
        self.unsafe_written_set     = None
        self.unsafe_not_written_set = None
        self.filename_meta          = None
        # Streams (StartStream RPC) exercises.
        self.stream_group = None    # dict, set once by _maybe_start_streams
        self.stream_must  = None    # set of (stream_name, filename), from _compute_stream_expectations
        self.stream_must_chunks = None   # list of chunk indices underlying stream_must
        self.expected_deactivations = 0  # python-side mirror of ShowStreams.num_deactivated_streams

    def __enter__(self):
        os.makedirs(self.ssd_dir, exist_ok=True)
        os.makedirs(self.nfs_dir, exist_ok=True)
        try:
            self._build_server()
            self._build_client()
            self._build_rpc()
        except BaseException:
            # Make sure any partially-constructed threads / directories
            # are torn down before propagating.
            self.__exit__(None, None, None)
            raise
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        try:
            if self.file_sub is not None:
                self.file_sub.close()
            if self.stream_sub is not None:
                self.stream_sub.close()
            if self.rpc_client is not None:
                self.rpc_client.close()
            if self.fxe is not None:
                self.fxe.stop()
            if self.server is not None:
                self.server.stop()
        finally:
            shutil.rmtree(self.ssd_dir, ignore_errors=True)
            shutil.rmtree(self.nfs_dir, ignore_errors=True)

    # ---- Top-level driver ----

    def run(self):
        """Execute the test's post-build phases in order."""
        self._send_loop()
        self._post_loop_sync()
        # Compute stream MUST-sets BEFORE _test_stream_misc (whose final
        # cancel_all stops future stream captures; already-transitioned
        # chunks have their writes queued regardless).
        self._compute_stream_expectations()
        self._test_stream_misc()
        self._print_debug_counters()
        self._drain_filesub()
        self._drain_streamsub()
        # Ring-history phases, in a deliberate order: the INACTIVE poll for
        # stream_a/b/c must run while they are still in the inactive ring
        # (deactivation budget: 4 entries so far, capacity >= 5), so the
        # re-registration test (+1 entry) and the eviction subtest (which
        # floods the ring) come after.
        self._poll_streams_inactive()
        self._test_stream_reregister()
        self._test_stream_eviction()
        self._print_summary()
        self._verify_files()
        self._verify_stream_files()

    # ---- Private builders ----

    def _build_server(self):
        p = self.p

        # Dummy-mode SlabAllocator (capacity=-1): FrbServer skips its
        # reaper thread, frames are allocated lazily on demand.
        slab_allocator = SlabAllocator("af_rhost", -1)

        self.allocator = AssembledFrameAllocator(
            slab_allocator,
            num_consumers          = p['num_receivers'],
            time_samples_per_chunk = p['time_samples_per_chunk'],
        )

        self.file_writer = FileWriter(
            ssd_root        = self.ssd_dir,
            nfs_root        = self.nfs_dir,
            num_ssd_threads = p['num_ssd_threads'],
            num_nfs_threads = p['num_nfs_threads'],
        )

        self.receivers = [
            Receiver(
                address     = f"127.0.0.1:{p['data_base_port'] + j}",
                allocator   = self.allocator,
                consumer_id = j,
                # Per-Socket short-read misbehavior on every accepted
                # peer socket. Strengthens the test by exercising the
                # incremental-parse path against pathological short
                # reads (1 byte at a time, etc.) that production
                # traffic rarely produces.
                misbehaving_reads = True,
            )
            for j in range(p['num_receivers'])
        ]

        # Dummy-mode BumpAllocators for the FrbServer's dedispersion host /
        # GPU buffers. With capacity=-1 each allocate_array() call allocates
        # its own backing memory via af_alloc, which is what
        # GpuDedisperser::allocate exclusively uses.
        host_alloc = BumpAllocator(ksgpu.af_rhost | ksgpu.af_zero, -1)
        gpu_alloc  = BumpAllocator(ksgpu.af_gpu   | ksgpu.af_zero, -1)

        self.server = FrbServer(p['config'], self.receivers, self.file_writer,
                                f"127.0.0.1:{p['rpc_port']}",
                                p['ringbuf_nchunks'],
                                min_data_mtu=1500,
                                host_allocator=host_alloc,
                                gpu_allocator=gpu_alloc,
                                cuda_device_id=0,
                                no_dedispersion=p['no_dedispersion'],
                                # Suppress the per-chunk "FrbServer: beamset=..." line;
                                # this test assembles hundreds of chunks.
                                quiet=True)
        self.server.start()

    def _build_client(self):
        p = self.p
        self.nworkers = p['nworkers']
        self.mpc      = p['time_samples_per_chunk'] // 256
        self.beam_ids = list(range(p['base_beam_id'],
                                   p['base_beam_id'] + p['nbeams']))

        self.xmd = XEngineMetadata.make_fiducial(
            p['config'].zone_nfreq, p['config'].zone_freq_edges, self.beam_ids,
            1.0,
        )

        ip_addrs = [f"127.0.0.1:{p['data_base_port'] + j}"
                    for j in range(p['num_receivers'])]

        # Pacing is randomized in _random_params (20% chance True).
        # Most iterations run paced=False so the test continues to
        # exercise real-time eviction races and FLAG_ACK ack-prediction
        # at full throughput; the occasional paced=True iteration
        # exercises the FakeXEngine pacing-thread path against a live
        # MonitorRingbuf stream from the in-process FrbServer.
        self.fxe = FakeXEngine(
            self.xmd, ip_addrs, self.nworkers,
            time_samples_per_chunk = p['time_samples_per_chunk'],
            debug = True,
            paced = p['paced'],
            rpc_address = f"127.0.0.1:{p['rpc_port']}",
        )

        # Per-worker positions (minichunk indices).
        ipos0 = np.random.randint(10**10)
        self.ipos = np.random.randint(ipos0, ipos0 + 10,
                                      size=self.nworkers, dtype=np.int64)
        self.wpos = np.copy(self.ipos)

        # Per-worker "dstate": workers can be in a temporary
        # "disconnected" state.
        self.dstate = np.random.random(self.nworkers) < np.random.uniform(0, 1)

        # Keeps track of which (worker_id, minichunk_index) pairs have been skipped.
        self.skipped = set()

        # Client-side AssembledFrameSets, allocated lazily as workers
        # advance into new chunks. self.fspos is the next un-allocated
        # chunk index.
        self.framesets = dict()
        self.fspos     = ipos0 // self.mpc

        # Client-side allocator (distinct from server-side).
        client_slab_allocator = SlabAllocator("af_rhost", -1)
        self.client_allocator = AssembledFrameAllocator(
            client_slab_allocator,
            num_consumers = 1,
            time_samples_per_chunk = p['time_samples_per_chunk']
        )
        self.client_allocator.initialize_metadata(self.xmd)
        self.client_allocator.initialize_initial_chunk(self.fspos)

    def _build_rpc(self):
        p = self.p

        # The subscriber must be opened BEFORE any write_files call to
        # capture every notification (FileSubscriber's constructor
        # blocks on the ready sentinel). We drain the subscription in
        # the main thread AFTER the iouter loop; gRPC's HTTP/2 stream
        # buffers the notifications in the meantime, which is fine at
        # our scale (<= ~90 notifications * ~40 bytes is well below
        # the 64 KB INITIAL_WINDOW_SIZE).
        self.rpc_client = FrbSearchClient(f"127.0.0.1:{p['rpc_port']}")
        self.file_sub   = self.rpc_client.subscribe_files()

        # Second subscription with subscribe_streams=True: receives BOTH
        # WriteFiles-triggered and stream-triggered notifications (drained in
        # _drain_streamsub, which ignores the stream_name=="" WriteFiles ones).
        # Also implicitly tests that file_sub (subscribe_streams=False, the
        # default) never sees stream files -- _drain_filesub asserts
        # stream_name == "" on everything it receives.
        self.stream_sub = self.rpc_client.subscribe_files(subscribe_streams=True)

        # Filenames tracked across all iouter turns. Three disjoint sets:
        #   safe_written_set       -- requested chunk was in [safe_lower, safe_upper];
        #                             server MUST schedule it (and we wait for notif).
        #   unsafe_written_set     -- requested chunk was outside the safe range,
        #                             server scheduled it anyway (we wait for notif).
        #   unsafe_not_written_set -- requested chunk was outside the safe range,
        #                             server did not schedule it (no notif expected).
        self.safe_written_set       = set()
        self.unsafe_written_set     = set()
        self.unsafe_not_written_set = set()

        # Per-filename (chunk_idx, beam_id) for the readback verification.
        self.filename_meta = {}

    # ---- Send loop ----

    def _send_loop(self):
        """1000-turn randomized send loop.

        Each turn picks a random worker, occasionally synchronizes it,
        and enqueues a Poisson-sized batch of SEND_MINICHUNK (or SKIP)
        commands. The Poisson mean is (1 + 0.1 * lag), where
        lag = max(wpos) - wpos[worker] -- so workers that have fallen
        behind catch up faster. This produces ragged per-worker
        progress (good coverage for the ambiguous band of the
        ack-prediction check).
        """
        fxe = self.fxe

        for iouter in range(1000):
            worker_id = random.randrange(self.nworkers)
            skip = self.dstate[worker_id] or (random.random() < 0.1)

            # Ocassionally synchronize, to prevent workers from getting too out-of-sync.
            if random.random() < 0.1:
                fxe.synchronize(worker_id)

            # n = Number of minichunks to advance.
            # Computed in a way that biases workers toward catching up with the leader.
            lag = int(np.max(self.wpos) - self.wpos[worker_id])
            n = int(np.random.poisson(1.0 + 0.1 * lag))

            # Advance (either skip or send) by n minichunks.
            for k in range(n):
                imc    = int(self.wpos[worker_id]) + k   # minichunk index
                ichunk = imc // self.mpc                 # chunk index

                while self.fspos <= ichunk:
                    self.framesets[self.fspos] = self.client_allocator.get_frame_set(consumer_id=0)
                    # Serial randomization. This transport/assembly test doesn't
                    # check data normalization, so normalize=False (arbitrary
                    # scales/offsets) and uniform int4 are fine.
                    self.framesets[self.fspos].randomize(normalize=False, gaussian=False)
                    assert self.framesets[self.fspos].time_chunk_index == self.fspos
                    self.fspos += 1

                if skip:
                    fxe.enqueue_skip_minichunk(worker_id, imc)
                    self.skipped.add((worker_id, imc))
                else:
                    fxe.enqueue_send_minichunk(worker_id, imc, self.framesets[ichunk])

            self.wpos[worker_id] += n

            # Randomly toggle dstate.
            if self.dstate[worker_id]:
                self.dstate[worker_id] = (random.random() < 0.8)   # 20% reconnection probability
            elif (random.random() < 0.01):                          # 1% disconnection probability
                fxe.enqueue_disconnect(worker_id)
                self.dstate[worker_id] = True

            # Issue rpc write with 1% probability.
            if random.random() < 0.01:
                self._maybe_issue_write(iouter)

            # Start the stream group once, a quarter of the way in (retries
            # until the server is rb_initialized; no-op after success).
            self._maybe_start_streams(iouter)

    def _maybe_issue_write(self, iouter):
        """Compute the safe chunk range, pick a request rectangle (with
        unsafe widening), issue write_files, and update tracking sets.
        Early-returns if no chunk is requestable this turn.
        """
        p = self.p
        status = self.rpc_client.get_status()
        rb_start     = status.rb_start
        rb_processed = status.rb_processed
        rb_assembled = status.rb_assembled
        rb_end       = status.rb_end

        # Compute "safe" chunk range. Two bounds:
        #
        # UPPER (rb_processed): chunks must be FULLY GPU-PROCESSED
        # (every beam in the chunk satisfies frame_id < rb_processed).
        # A chunk in [rb_processed, rb_assembled) is fully assembled
        # but the GPU may still be modifying frames there, so it is
        # NOT rpc-writeable. A chunk in [rb_assembled, rb_end) is in
        # the ringbuf but still being assembled by some receivers.
        #
        # LOWER (rb_start, plus a future-bound): chunks must
        # be in the ringbuf at SERVER-processing time, not
        # just snapshot time. The FrbServer worker advances
        # rb_start as new chunks arrive. We bound the
        # worst-case rb_start at server time using:
        #   max_future_rb_end <= (max_wpos // mpc - 1) * nbeams
        #   max_future_rb_start <= max_future_rb_end - rb_size
        rb_size = p['ringbuf_nchunks'] * p['nbeams']

        max_wpos = int(np.max(self.wpos))
        if max_wpos > 0:
            highest_enqueued_chunk = (max_wpos - 1) // self.mpc
            rb_end_upper = max(rb_end, (highest_enqueued_chunk - 1) * p['nbeams'])
        else:
            rb_end_upper = rb_end
        rb_start_upper = max(rb_start, max(0, rb_end_upper - rb_size))

        safe_lower = (rb_start_upper + p['nbeams'] - 1) // p['nbeams']  # ceil
        safe_upper = (rb_processed   // p['nbeams']) - 1                # fully GPU-processed

        # Widen the requested chunk range to also exercise "unsafe"
        # chunks (which may or may not still be in the ringbuf at
        # server processing time). The safe sub-range remains
        # guaranteed-writable.
        lower_bound = max(0, safe_lower - 2)
        upper_bound = safe_upper + 2
        if lower_bound > upper_bound:
            return   # nothing to request this turn

        # Pick a contiguous range of 1-3 chunks in [lower_bound, upper_bound].
        max_nchunks = min(3, upper_bound - lower_bound + 1)
        selected_nchunks = random.randint(1, max_nchunks)
        chunk_min = random.randint(lower_bound, upper_bound - selected_nchunks + 1)
        chunk_max = chunk_min + selected_nchunks - 1

        # Pick 1-3 random beams.
        all_beam_ids = list(range(p['base_beam_id'], p['base_beam_id'] + p['nbeams']))
        selected_nbeams = random.randint(1, min(3, p['nbeams']))
        selected_beams = random.sample(all_beam_ids, selected_nbeams)

        # Include iouter in the acqdir so that filenames are unique across
        # iouter turns (same beam+chunk may be requested twice across
        # iterations; the per-turn uniqueness lives in the directory now).
        acqdir = f"test_{iouter}"

        # Compute the expected filenames client-side (via _acq_filename,
        # which mirrors the server's make_acq_relpath). Tag each filename
        # as safe or unsafe based on whether its chunk falls in
        # [safe_lower, safe_upper].
        expanded = {}   # filename -> is_safe
        for c in range(chunk_min, chunk_max + 1):
            for b in selected_beams:
                fn = _acq_filename(acqdir, b, c)
                expanded[fn] = (safe_lower <= c <= safe_upper)
                self.filename_meta[fn] = (c, b)

        # write_files takes an fpga-seq range (half-open). Convert our inclusive
        # chunk range [chunk_min, chunk_max] to [chunk_min*spc, (chunk_max+1)*spc),
        # where spc = seq_per_chunk = time_samples_per_chunk * seq_per_frb_time_sample
        # (same value the server uses, so it recovers exactly [chunk_min, chunk_max]).
        spc = p['time_samples_per_chunk'] * self.xmd.seq_per_frb_time_sample
        filenames = self.rpc_client.write_files(
            beams          = selected_beams,
            fpga_seq_start = chunk_min * spc,
            fpga_seq_end   = (chunk_max + 1) * spc,
            acqdir         = acqdir,
        )

        returned = set(filenames)

        # Safety check: every safe filename in the requested rectangle
        # MUST be returned. Unsafe filenames may or may not be returned
        # (both outcomes are accepted).
        safe_this_call = {fn for fn, is_safe in expanded.items() if is_safe}
        missing_safe = safe_this_call - returned
        if missing_safe:
            raise RuntimeError(
                f"write_files at iouter={iouter}: missing safe filenames {missing_safe}, "
                f"chunks=[{chunk_min}, {chunk_max}], beams={selected_beams}, "
                f"safe range=[{safe_lower}, {safe_upper}], "
                f"rb=(start={rb_start}, processed={rb_processed}, assembled={rb_assembled}, end={rb_end}), "
                f"rb_start_upper={rb_start_upper}, max_wpos={max_wpos}"
            )

        # Bookkeeping for the three running totals.
        self.safe_written_set.update(safe_this_call)
        for fn, is_safe in expanded.items():
            if is_safe:
                continue
            if fn in returned:
                self.unsafe_written_set.add(fn)
            else:
                self.unsafe_not_written_set.add(fn)

    # ---- Stream (StartStream RPC) exercises ----

    def _maybe_start_streams(self, iouter):
        """Start the test's stream group (once, a quarter of the way through
        the send loop, so plenty of chunks transition while it is active):

          - stream_a and stream_b share ONE acqdir -> every captured frame gets
            duplicate save_paths, exercising the FileWriter NFS dup-skip
            branch (both streams must still get their own success
            notifications, file written once);
          - stream_c uses a distinct MULTI-LEVEL acqdir -> exercises lazy
            (two-deep) directory creation in the FileWriter threads, and
            gives a second NFS name that must be a HARDLINK of the
            shared-acqdir file (verified via st_ino in
            _verify_stream_files).

        All three streams: same beams, fpga range [0, end) with end chosen
        K chunks past the highest chunk queued so far on the sender side.
        Retries every turn until the server is rb_initialized.
        """
        if (self.stream_group is not None) or (iouter < 250):
            return

        p = self.p
        K = 8   # chunks of stream lifetime beyond the highest queued chunk
        spc = p['time_samples_per_chunk'] * self.xmd.seq_per_frb_time_sample
        sel_beams = self.beam_ids[:min(2, len(self.beam_ids))]
        acqdir_shared = "stream_shared"
        acqdir_c      = "stream_c/sub"

        # Highest chunk enqueued to any FakeXEngine worker (wpos[w] = next
        # minichunk to enqueue). The streams' chunk range is [0, cq + K].
        #
        # Basing the range on the sender's queued position (rather than the
        # server's current position, via GetStatus) makes the server-side
        # "entirely in the past" rejection impossible instead of merely
        # unlikely: the ring buffer only advances when new data (or skips)
        # arrive, so rb_processed/nbeams <= cq + 1 <= cq + K -- and we are
        # the thread that enqueues, so cq cannot grow before the RPC lands.
        cq = (int(np.max(self.wpos)) - 1) // self.mpc
        end_fpga = (cq + K + 1) * spc

        try:
            for stream_name, acqdir in [("stream_a", acqdir_shared),
                                ("stream_b", acqdir_shared),
                                ("stream_c", acqdir_c)]:
                self.rpc_client.start_stream(
                    stream_name=stream_name, acqdir=acqdir, beam_ids=sel_beams,
                    fpga_seq_start=0, fpga_seq_end=end_fpga)
        except grpc.RpcError as e:
            # Server hasn't locked onto the X-engine stream yet; retry on a
            # later turn. Anything else is a real failure.
            if "initial fpga chunk" in e.details():
                return
            raise

        # Registration boundary for the capture guarantee: chunks strictly
        # greater than c0 transition entirely AFTER all three streams were
        # registered, so they MUST be captured (chunk c0 itself is
        # ambiguous -- it may have partially transitioned mid-registration).
        c0 = self.rpc_client.get_status().rb_processed // p['nbeams']

        self.stream_group = dict(
            sel_beams=sel_beams, spc=spc, c0=c0, chunk_last=(cq + K),
            acqdir_shared=acqdir_shared, acqdir_c=acqdir_c)

    def _compute_stream_expectations(self):
        """Compute the MUST-be-written set of (stream_name, filename) pairs, and
        sanity-check the ShowStreams queued-counters.

        Guarantee tested: every matching frame that TRANSITIONS
        (assembled -> processed) while a stream is active must be written.
        Chunks in [c0 + 1, min(chunk_last, c_end - 1)] provably transitioned
        after registration and before now, so they are exactly the
        guaranteed set (boundary chunk c0 and chunks still in flight are
        ambiguous and merely tolerated by _drain_streamsub).
        """
        self.stream_must = set()
        self.stream_must_chunks = []

        if self.stream_group is None:
            return
        g = self.stream_group

        c_end = self.rpc_client.get_status().rb_processed // self.p['nbeams']
        lo = g['c0'] + 1
        hi = min(g['chunk_last'], c_end - 1)
        self.stream_must_chunks = list(range(lo, hi + 1))

        for c in self.stream_must_chunks:
            for b in g['sel_beams']:
                fn_shared = _acq_filename(g['acqdir_shared'], b, c)
                fn_c      = _acq_filename(g['acqdir_c'], b, c)
                self.stream_must.add(("stream_a", fn_shared))
                self.stream_must.add(("stream_b", fn_shared))
                self.stream_must.add(("stream_c", fn_c))

        # ShowStreams counter sanity. Streams may already be deactivated
        # (expired) and listed from the inactive-ring history; either way,
        # every stream listed at this point is one of stream_a/b/c. The
        # R1/R2 increment-ordering rules guarantee written + errored <=
        # queued at every instant.
        must_files_per_stream = len(self.stream_must_chunks) * len(g['sel_beams'])
        max_files_per_stream = (g['chunk_last'] + 1) * len(g['sel_beams'])

        ss = self.rpc_client.show_streams()
        assert list(ss.beam_ids) == self.beam_ids
        for info in ss.streams:
            assert info.args.stream_name in ("stream_a", "stream_b", "stream_c")
            assert list(info.args.beam_ids) == g['sel_beams']
            assert info.num_files_written + info.num_files_errored <= info.num_files_queued
            assert must_files_per_stream <= info.num_files_queued <= max_files_per_stream

    def _expect_rpc_error(self, fn, substr):
        """Run fn(), assert it raises grpc.RpcError whose details contain substr."""
        try:
            fn()
        except grpc.RpcError as e:
            assert substr in e.details(), \
                f"expected RPC error containing {substr!r}, got: {e.details()!r}"
        else:
            raise RuntimeError(f"expected RPC error containing {substr!r}, got success")

    def _start_never_matching_stream(self, stream_name):
        """Register a stream whose fpga range is in the far future, so it
        never matches (no files, no notifications, zero counters). Used by
        the cancel / ring-history subtests."""
        p = self.p
        rpc = self.rpc_client
        spc = p['time_samples_per_chunk'] * self.xmd.seq_per_frb_time_sample
        c_now = rpc.get_status().rb_processed // p['nbeams']
        rpc.start_stream([self.beam_ids[0]], stream_name=stream_name,
                         acqdir=stream_name,
                         fpga_seq_start=(c_now + 10**6) * spc,
                         fpga_seq_end=(c_now + 10**6 + 1) * spc)

    def _test_stream_misc(self):
        """StartStream/CancelStream validation failures + cancel semantics.
        Runs post-loop (so the server is rb_initialized and well past chunk
        0). The valid stream used for cancel tests lives in the far future,
        so it never matches (no files, no notifications).

        Ring-history budget: this phase adds exactly 1 + (3 if the stream
        group ran) entries to the inactive ring (misc_x's cancel, plus
        stream_a/b/c via expiry or cancel_all -- each stream deactivates
        exactly once). Keeping the running total at 4 < CAP until after
        _poll_streams_inactive is what guarantees the poll finds stream_a/b/c
        still visible in the ring."""
        rpc = self.rpc_client
        p = self.p

        CAP = constants.inactive_file_stream_capacity
        assert CAP >= 5, (
            "test's stream-history budget assumes inactive_file_stream_capacity >= 5; "
            "if the capacity was decreased, revisit _test_stream_misc / _test_stream_eviction")

        spc = p['time_samples_per_chunk'] * self.xmd.seq_per_frb_time_sample
        beam0 = self.beam_ids[0]
        c_now = rpc.get_status().rb_processed // p['nbeams']
        ACTIVE   = frb_search_pb2.STREAM_STATUS_ACTIVE
        INACTIVE = frb_search_pb2.STREAM_STATUS_INACTIVE

        def start(**kw):
            kw.setdefault('stream_name', 'misc_x')
            kw.setdefault('acqdir', 'misc_dir')
            kw.setdefault('beam_ids', [beam0])
            # Default range: far future (never matches).
            kw.setdefault('fpga_seq_start', (c_now + 10**6) * spc)
            kw.setdefault('fpga_seq_end', (c_now + 10**6 + 1) * spc)
            return lambda: rpc.start_stream(**kw)

        # Validation failures.
        self._expect_rpc_error(start(stream_name=''), "stream_name must be a nonempty")
        self._expect_rpc_error(start(beam_ids=[]), "beam_ids must be nonempty")
        self._expect_rpc_error(start(beam_ids=[999999]), "unknown beam_id")
        self._expect_rpc_error(start(beam_ids=[beam0, beam0]), "appears more than once")
        self._expect_rpc_error(start(acqdir=''), "acqdir must be a nonempty")
        self._expect_rpc_error(start(acqdir='/abs/path'), "invalid acqdir")
        self._expect_rpc_error(start(acqdir='../evil'), "invalid acqdir")
        self._expect_rpc_error(start(acqdir='foo//bar'), "invalid acqdir")
        self._expect_rpc_error(start(acqdir='foo/'), "invalid acqdir")
        self._expect_rpc_error(start(acqdir='.'), "invalid acqdir")
        self._expect_rpc_error(start(fpga_seq_start=100, fpga_seq_end=50), "invalid fpga_seq range")
        if c_now >= 1:
            # Range [0, spc) covers only chunk 0, long since processed.
            self._expect_rpc_error(start(fpga_seq_start=0, fpga_seq_end=spc),
                                   "entirely in the past")

        # Cancel semantics: register a (never-matching) stream, then cancel it.
        start()()   # succeeds
        infos = [i for i in rpc.show_streams().streams if i.args.stream_name == 'misc_x']
        assert len(infos) == 1 and infos[0].status == ACTIVE
        self._expect_rpc_error(start(), "already in use")           # duplicate stream_name
        assert rpc.cancel_stream(stream_name='misc_x') == 1
        self.expected_deactivations += 1

        # The cancelled stream remains VISIBLE, now in the inactive-ring
        # history: it never matched anything, so its counters are all zero
        # and it is INACTIVE (nothing to drain) immediately, cancelled=true.
        infos = [i for i in rpc.show_streams().streams if i.args.stream_name == 'misc_x']
        assert len(infos) == 1
        info = infos[0]
        assert info.status == INACTIVE
        assert info.cancelled
        assert info.num_files_queued == 0
        assert info.num_files_written == 0
        assert info.num_files_errored == 0
        assert info.started_at_unix_ns > 0
        assert info.deactivated_at_unix_ns >= info.started_at_unix_ns

        # Cancelling an already-inactive stream is an error (with a clearer
        # message than the never-heard-of case).
        self._expect_rpc_error(lambda: rpc.cancel_stream(stream_name='misc_x'),
                               "already inactive")
        self._expect_rpc_error(lambda: rpc.cancel_stream(stream_name='no_such_stream'),
                               "no active stream")

        # cancel_all: 0 to 3 streams may remain active (stream_a/b/c, if not
        # yet expired). Their MUST files (computed in
        # _compute_stream_expectations, before this) already transitioned,
        # so their writes are queued regardless -- cancellation never
        # retracts queued work. Either way, all of a/b/c have been
        # deactivated (expiry or cancel) once this returns.
        n = rpc.cancel_stream(cancel_all=True)
        assert 0 <= n <= 3
        if self.stream_group is not None:
            self.expected_deactivations += 3

        ss = rpc.show_streams()
        assert all(i.status != ACTIVE for i in ss.streams)
        assert ss.num_deactivated_streams == self.expected_deactivations

    def _poll_streams_inactive(self):
        """Post-drain strong check: poll ShowStreams until stream_a/b/c all
        report INACTIVE (deactivated + fully drained), then verify their
        final counters: errored == 0 and written == queued. No timeout,
        matching the drain philosophy: all three are already deactivated
        (the misc phase ended with cancel_all), and every queued write
        eventually completes. Visibility is guaranteed by the ring budget:
        only 4 deactivations have happened, and CAP >= 5."""
        if self.stream_group is None:
            return
        INACTIVE = frb_search_pb2.STREAM_STATUS_INACTIVE
        stream_names = ("stream_a", "stream_b", "stream_c")

        while True:
            ss = self.rpc_client.show_streams()
            infos = [i for i in ss.streams if i.args.stream_name in stream_names]
            assert len(infos) == 3, \
                f"stream_a/b/c not all visible in stream history: {[i.args.stream_name for i in infos]}"
            if all(i.status == INACTIVE for i in infos):
                break
            time.sleep(0.05)

        for i in infos:
            # 'cancelled' may be True (the misc-phase cancel_all got there
            # first) or False (expired naturally) -- both are valid.
            assert i.num_files_errored == 0
            assert i.num_files_written == i.num_files_queued
            assert i.started_at_unix_ns > 0
            assert i.deactivated_at_unix_ns >= i.started_at_unix_ns

    def _test_stream_reregister(self):
        """stream_name uniqueness is enforced against ACTIVE streams only:
        re-registering an stream_name that still sits in the inactive-ring
        history succeeds. (Run AFTER _poll_streams_inactive: this adds a
        5th ring entry, eating the a/b/c visibility margin.)"""
        rpc = self.rpc_client
        assert any(i.args.stream_name == 'misc_x' for i in rpc.show_streams().streams)
        self._start_never_matching_stream('misc_x')   # succeeds despite the ring entry
        assert rpc.cancel_stream(stream_name='misc_x') == 1
        self.expected_deactivations += 1
        assert rpc.show_streams().num_deactivated_streams == self.expected_deactivations

    def _test_stream_eviction(self):
        """Ring eviction + the mass-cancellation property: a single
        cancel_all over MORE than CAP streams cancels ALL of them
        (num_cancelled counts every one), while the display history retains
        only the newest CAP. Runs LAST among the stream subtests: it evicts
        stream_a/b/c and misc_x from the ring (their checks are done)."""
        rpc = self.rpc_client
        CAP = constants.inactive_file_stream_capacity
        ACTIVE   = frb_search_pb2.STREAM_STATUS_ACTIVE
        INACTIVE = frb_search_pb2.STREAM_STATUS_INACTIVE

        names = [f"evict_{i}" for i in range(CAP + 1)]
        for name in names:
            self._start_never_matching_stream(name)

        n = rpc.cancel_stream(cancel_all=True)
        assert n == CAP + 1
        self.expected_deactivations += CAP + 1

        ss = rpc.show_streams()
        assert all(i.status != ACTIVE for i in ss.streams)

        # cancel_all deactivates in registration order, so evict_0 is the
        # oldest ring entry and the one evicted; the ring lists
        # evict_1 .. evict_CAP, oldest to newest.
        inactive = [i for i in ss.streams if i.status != ACTIVE]
        assert len(inactive) == CAP
        assert [i.args.stream_name for i in inactive] == names[1:]
        for i in inactive:
            assert i.status == INACTIVE   # never matched -> nothing to drain
            assert i.cancelled
            assert i.num_files_queued == 0
            assert i.num_files_written == 0
            assert i.num_files_errored == 0

        # The monotone total counts evict_0 too -- this is how a client
        # detects that history was dropped.
        assert ss.num_deactivated_streams == self.expected_deactivations

    def _drain_streamsub(self):
        """Wait until every MUST (stream_name, filename) pair has arrived via
        the subscribe_streams=True subscription. This subscription also
        receives WriteFiles-triggered notifications (stream_name == ""), which
        are skipped; stream notifications beyond the MUST set (boundary
        chunk c0, chunks still transitioning during the drain) are
        tolerated. No timeout, matching _drain_filesub."""
        received = set()
        while not received.issuperset(self.stream_must):
            filename, error_message, stream_name = next(self.stream_sub)
            if stream_name == "":
                continue    # WriteFiles-triggered; drained/checked in _drain_filesub
            if error_message:
                raise RuntimeError(
                    f"stream_sub: write failed for {filename!r} "
                    f"(stream_name={stream_name!r}): {error_message}")
            assert stream_name in ("stream_a", "stream_b", "stream_c"), \
                f"unexpected stream_name {stream_name!r} (misc_x streams never match)"
            received.add((stream_name, filename))

    def _verify_stream_files(self):
        """Byte-verify the MUST stream files, and check the dedup/hardlink
        behavior: for each captured (chunk, beam), stream_a/stream_b's shared
        acqdir and stream_c's distinct acqdir must be ONE inode on NFS
        (bytes written once; second name is a hardlink; duplicate second
        entry for the shared acqdir skipped)."""
        if self.stream_group is None:
            print("    streams: not exercised (server was not rb_initialized by iouter=250)")
            return
        g = self.stream_group

        for c in self.stream_must_chunks:
            for b in g['sel_beams']:
                fn_shared = _acq_filename(g['acqdir_shared'], b, c)
                fn_c      = _acq_filename(g['acqdir_c'], b, c)

                # Full content check on one name; inode identity makes it
                # cover the other.
                self._verify_one_file(fn_c, c, b)

                st_shared = os.stat(os.path.join(self.nfs_dir, fn_shared))
                st_c      = os.stat(os.path.join(self.nfs_dir, fn_c))
                assert st_shared.st_ino == st_c.st_ino, \
                    f"expected hardlink: {fn_shared!r} and {fn_c!r} have different inodes"
                assert st_shared.st_nlink >= 2

    # ---- Post-loop phases ----

    def _post_loop_sync(self):
        """Block on each worker's command queue and assert that every
        enqueued minichunk reached a terminal status.
        """
        for worker_id in range(self.nworkers):
            # Block until worker thread has processed all commands,
            # and received all acks.
            self.fxe.synchronize(worker_id)

            # Check status for each minichunk.
            for imc in range(self.ipos[worker_id], self.wpos[worker_id]):
                status = self.fxe.get_minichunk_status(worker_id, imc)
                if (worker_id, imc) in self.skipped:
                    assert status == FakeXEngine.STATUS_SKIPPED
                else:
                    assert (status == FakeXEngine.STATUS_DROPPED) or (status == FakeXEngine.STATUS_ASSEMBLED)

    def _print_debug_counters(self):
        # All acks drained by _post_loop_sync; the counters are now a
        # stable snapshot.
        counters = self.fxe.get_debug_counters()
        labels = [
            "unambiguous, DROPPED",
            "unambiguous, ASSEMBLED",
            "ambiguous,   DROPPED",
            "ambiguous,   ASSEMBLED",
        ]
        for label, count in zip(labels, counters):
            print(f"    {label}: {count}")

    def _drain_filesub(self):
        """Wait until every scheduled filename has arrived via the
        FileSubscriber stream. No timeout (future prompt will add one);
        a stuck write will manifest as the test blocking forever here.
        """
        scheduled = self.safe_written_set | self.unsafe_written_set
        received_filenames = set()
        while not received_filenames.issuperset(scheduled):
            filename, error_message, stream_name = next(self.file_sub)
            # file_sub was opened with the default subscribe_streams=False,
            # so stream-triggered files (nonempty stream_name) must never
            # appear here.
            assert stream_name == "", \
                f"subscribe_streams=False subscriber received stream file {filename!r} (stream_name={stream_name!r})"
            if error_message:
                raise RuntimeError(
                    f"FileSubscriber: write failed for {filename!r}: "
                    f"{error_message}"
                )
            received_filenames.add(filename)

    def _print_summary(self):
        print(f"    safe, written:       {len(self.safe_written_set)}")
        print(f"    unsafe, written:     {len(self.unsafe_written_set)}")
        print(f"    unsafe, not written: {len(self.unsafe_not_written_set)}")
        print(f"    stream, must-write:  {len(self.stream_must)}")

    def _verify_files(self):
        """Read every scheduled file back from disk and byte-compare its
        contents to an expected buffer reconstructed from client-side
        state. _drain_filesub ensures every file is on disk; minichunk
        statuses are terminal after _post_loop_sync.
        """
        scheduled = self.safe_written_set | self.unsafe_written_set
        for filename in sorted(scheduled):
            chunk_idx, beam_id = self.filename_meta[filename]
            self._verify_one_file(filename, chunk_idx, beam_id)

    def _verify_one_file(self, filename, chunk_idx, beam_id):
        """Read one written file back from NFS and byte-compare its contents
        to the expected buffer reconstructed from client-side state. Shared
        by _verify_files (WriteFiles) and _verify_stream_files (streams).
        """
        expected_data, expected_so = self._compute_expected_data(chunk_idx, beam_id)
        path = os.path.join(self.nfs_dir, filename)
        frame = AssembledFrame.from_asdf(path)

        assert frame.beam_id          == beam_id
        assert frame.time_chunk_index == chunk_idx
        assert frame.nfreq            == self.p['total_nfreq']
        assert frame.ntime            == self.p['time_samples_per_chunk']

        actual_data = np.asarray(frame.data)
        if not np.array_equal(actual_data, expected_data):
            mismatch = np.argwhere(actual_data != expected_data)
            first = tuple(mismatch[0])
            raise RuntimeError(
                f"data mismatch for {filename!r} "
                f"(chunk={chunk_idx}, beam={beam_id}): "
                f"{len(mismatch)} mismatching bytes, first at "
                f"index {first}: "
                f"actual=0x{actual_data[first]:02x}, "
                f"expected=0x{expected_data[first]:02x}"
            )

        # scales_offsets: byte-exact compare via uint8 view (avoids
        # any float-equality ambiguity from possibly-NaN bit patterns).
        actual_so = np.asarray(frame.scales_offsets).view(np.uint8)
        if not np.array_equal(actual_so, expected_so):
            mismatch = np.argwhere(actual_so != expected_so)
            first = tuple(mismatch[0])
            raise RuntimeError(
                f"scales_offsets mismatch for {filename!r} "
                f"(chunk={chunk_idx}, beam={beam_id}): "
                f"{len(mismatch)} mismatching bytes, first at "
                f"index {first}: "
                f"actual=0x{actual_so[first]:02x}, "
                f"expected=0x{expected_so[first]:02x}"
            )

    def _compute_expected_data(self, chunk_idx, beam_id):
        """Reconstruct the byte-exact contents of a written frame file.

        Returns (expected_data, expected_scales_offsets) where:
          - expected_data is uint8 shape (total_nfreq, time_samples_per_chunk // 2),
            filled with 0x88 except for (worker_freq_channels, 128-byte time slice)
            regions whose (worker, global_minichunk) is in STATUS_ASSEMBLED;
          - expected_scales_offsets is uint8 shape (total_nfreq, mpc, 4)
            (= float16 (total_nfreq, mpc, 2) viewed as bytes), filled with 0x00
            except for analogous ASSEMBLED slots.

        Caller must ensure statuses are terminal (post-synchronize); STATUS_DROPPED
        and STATUS_SKIPPED both leave the default mask in place. Minichunks outside
        a worker's submitted range [ipos[w], wpos[w]) also leave the mask.
        """
        p = self.p
        total_nfreq = p['total_nfreq']
        tspc        = p['time_samples_per_chunk']
        mpc         = self.mpc

        expected_data = np.full((total_nfreq, tspc // 2), 0x88, dtype=np.uint8)
        expected_so   = np.zeros((total_nfreq, mpc, 4),    dtype=np.uint8)

        # framesets[chunk_idx] is guaranteed populated for any chunk
        # the server wrote: the receiver only pushes a chunk into the
        # ringbuf in response to an incoming minichunk from a higher
        # chunk, and that minichunk's enqueue_send_minichunk call
        # already advanced fspos past chunk_idx.
        frame_set = self.framesets[chunk_idx]
        b_idx = beam_id - p['base_beam_id']
        source_data = np.asarray(frame_set.frames[b_idx].data)
        # scales_offsets is float16 (nfreq, mpc, 2); view as uint8
        # (nfreq, mpc, 4) for byte-exact assembly.
        source_so = np.asarray(frame_set.frames[b_idx].scales_offsets).view(np.uint8)
        assert source_so.shape == (total_nfreq, mpc, 4)

        for imc in range(mpc):
            global_mc = chunk_idx * mpc + imc
            t0 = imc * 128
            t1 = t0 + 128
            for w in range(self.nworkers):
                if (global_mc < self.ipos[w]) or (global_mc >= self.wpos[w]):
                    continue
                if self.fxe.get_minichunk_status(w, global_mc) != FakeXEngine.STATUS_ASSEMBLED:
                    continue
                for f in self.fxe.get_worker_freq_channels(w):
                    expected_data[f, t0:t1] = source_data[f, t0:t1]
                    expected_so[f, imc, :]  = source_so[f, imc, :]

        return expected_data, expected_so


def test_network():
    """One iteration of the FakeXEngine <-> FrbServer loopback test."""
    print("  test_network()...")
    with NetworkTester() as t:
        # The DedispersionConfig is verbose; print other params and a
        # one-line config summary separately.
        params_no_config = {k: v for k, v in t.p.items() if k != 'config'}
        print(f"    params: {params_no_config}")
        c = t.p['config']
        print(f"    config: toplevel_tree_rank={c.toplevel_tree_rank}, num_primary_trees={c.num_primary_trees},"
              f" beams_per_batch={c.beams_per_batch}, num_active_batches={c.num_active_batches},"
              f" dtype={c.dtype}")
        t.run()
    print("    PASSED")
