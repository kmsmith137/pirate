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

 - test WriteFiles "future writes": requests extending past the current
     processing threshold (rb_streamed) return the promised filenames
     immediately, every promised file is written EXACTLY once (delivered
     via SubscribeFiles with empty stream_name), and the future part is
     truncated at future_write_max_samples (rounded up to a chunk).

NOTE: "stream" (StartStream/ShowStreams/CancelStream RPC) testing was
temporarily removed from this test on 2026-07-10: it had race conditions
that were a recurring source of spurious failures. Recover the removed
code from git history when reinstating (it was introduced around commit
9b0fd91, and removed in the commit that added this note).

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

from ..pirate_pybind11 import FrbServer
from ..rpc import FrbSearchClient

from .utils import make_random_subscale_config, pick_receiver_worker_counts


def _acq_filename(acqdir, beam_id, chunk):
    """Client-side mirror of the server's fixed naming scheme
    (make_acq_relpath in src_lib/FileWriter.cpp):
    {acqdir}/frame_b{beam_id}_t{chunk}.asdf, unpadded decimal."""
    return f"{acqdir}/frame_b{beam_id}_t{chunk}.asdf"


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

        # Future-writes geometry: the server's truncation horizon in chunks
        # (future_write_max_samples rounded up to whole chunks, mirroring
        # the WriteFiles handler). Zero => future writes disabled.
        tspc = self.p['time_samples_per_chunk']
        fwms = self.p['config'].future_write_max_samples
        self.n_fut = (fwms + tspc - 1) // tspc
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
        self.safe_written_set       = None
        self.unsafe_written_set     = None
        self.unsafe_not_written_set = None
        self.filename_meta          = None

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
            if self.rpc_client is not None:
                self.rpc_client.close()
            if self.fxe is not None:
                self.fxe.stop()
            if self.server is not None:
                self.server.stop()
                # If the server error-stopped during the test, the stop()
                # above was a no-op (first-stop-wins) and this rethrows the
                # server's saved root-cause exception, chaining onto any
                # in-flight test exception. After a clean stop it's a no-op.
                self.server.poll_from_python(timeout_ms=0)
        finally:
            shutil.rmtree(self.ssd_dir, ignore_errors=True)
            shutil.rmtree(self.nfs_dir, ignore_errors=True)

    # ---- Top-level driver ----

    def run(self):
        """Execute the test's post-build phases in order."""
        self._send_loop()
        self._post_loop_sync()
        self._flush_promises()
        self._print_debug_counters()
        self._drain_filesub()
        self._print_summary()
        self._verify_files()

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

        # Filenames tracked across all iouter turns. Three disjoint sets:
        #   safe_written_set       -- requested chunk was in the GUARANTEED range
        #                             [safe_lower, guaranteed_upper] (fully processed
        #                             past, or within the future-write guarantee);
        #                             server MUST schedule it (and we wait for notif).
        #   unsafe_written_set     -- requested chunk was outside the guaranteed range,
        #                             server scheduled it anyway (we wait for notif).
        #   unsafe_not_written_set -- requested chunk was outside the guaranteed range,
        #                             server did not schedule it (no notif expected).
        self.safe_written_set       = set()
        self.unsafe_written_set     = set()
        self.unsafe_not_written_set = set()

        # Per-filename (chunk_idx, beam_id) for the readback verification.
        self.filename_meta = {}

        # Highest chunk index over all returned (scheduled) filenames; the
        # flush tail (_flush_promises) must push processing past it so that
        # every promised future file actually gets written. -1 = none yet.
        self.max_promised_chunk = -1

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

    def _maybe_issue_write(self, iouter):
        """Compute the guaranteed chunk range, pick a request rectangle
        (widened into the past and the future), issue write_files, and
        update tracking sets. Early-returns if no chunk is requestable
        this turn.
        """
        p = self.p
        nbeams = p['nbeams']
        n_fut = self.n_fut

        status = self.rpc_client.get_status()   # "S0" (pre-request snapshot)
        rb_start     = status.rb_start
        rb_processed = status.rb_processed
        rb_streamed  = status.rb_streamed
        rb_assembled = status.rb_assembled
        rb_end       = status.rb_end

        # Compute the GUARANTEED-returned chunk range. Three bounds:
        #
        # PAST UPPER (rb_processed): chunks must be FULLY GPU-PROCESSED
        # (every beam in the chunk satisfies frame_id < rb_processed).
        # A chunk in [rb_processed, rb_assembled) is fully assembled
        # but the GPU may still be modifying frames there, so it is
        # NOT rpc-writeable via the past path.
        #
        # LOWER (rb_start, plus a future-bound): chunks must
        # be in the ringbuf at SERVER-processing time, not
        # just snapshot time. The FrbServer worker advances
        # rb_start as new chunks arrive. We bound the
        # worst-case rb_start at server time using:
        #   max_future_rb_end <= (max_wpos // mpc - 1) * nbeams
        #   max_future_rb_start <= max_future_rb_end - rb_size
        #
        # FUTURE UPPER (rb_streamed + n_fut): the server's past/future
        # boundary chunk at handling time (its rb_streamed // nbeams) is
        # >= lb_chunk, since rb_streamed is monotone and S0 was read
        # before the RPC was handled. So every requested chunk in
        # [lb_chunk, lb_chunk + n_fut - 1] is guaranteed returned:
        # promised by the future path if it lands at/above the boundary,
        # written by the past path otherwise. (The past sub-case needs the
        # chunk to still be in the ringbuf -- covered by safe_lower -- and
        # relies on this test running NO reaper (dummy-mode allocator), so
        # the past path's reaped-skip can never drop a filename.)
        rb_size = p['ringbuf_nchunks'] * nbeams

        max_wpos = int(np.max(self.wpos))
        if max_wpos > 0:
            highest_enqueued_chunk = (max_wpos - 1) // self.mpc
            rb_end_upper = max(rb_end, (highest_enqueued_chunk - 1) * nbeams)
        else:
            rb_end_upper = rb_end
        rb_start_upper = max(rb_start, max(0, rb_end_upper - rb_size))

        safe_lower = (rb_start_upper + nbeams - 1) // nbeams  # ceil
        safe_upper = (rb_processed   // nbeams) - 1           # fully GPU-processed
        lb_chunk   = rb_streamed // nbeams                    # lower bound on server boundary chunk

        guaranteed_upper = max(safe_upper, lb_chunk + n_fut - 1)

        # Widen the requested chunk range beyond the guaranteed range, to
        # also exercise "unsafe" past chunks (which may or may not still be
        # in the ringbuf at server processing time) and future chunks beyond
        # the truncation horizon. The guaranteed sub-range remains
        # guaranteed-returned.
        lower_bound = max(0, safe_lower - 2)
        upper_bound = safe_upper + n_fut + 3
        if lower_bound > upper_bound:
            return   # nothing to request this turn

        # Pick a contiguous range of 1-3 chunks in [lower_bound, upper_bound].
        max_nchunks = min(3, upper_bound - lower_bound + 1)
        selected_nchunks = random.randint(1, max_nchunks)
        chunk_min = random.randint(lower_bound, upper_bound - selected_nchunks + 1)
        chunk_max = chunk_min + selected_nchunks - 1

        # Pick 1-3 random beams (random.sample: no duplicates, matching the
        # server's duplicate-beam rejection).
        all_beam_ids = list(range(p['base_beam_id'], p['base_beam_id'] + p['nbeams']))
        selected_nbeams = random.randint(1, min(3, p['nbeams']))
        selected_beams = random.sample(all_beam_ids, selected_nbeams)

        # Occasionally check that duplicate beam_ids are rejected.
        if random.random() < 0.05:
            self._check_duplicate_beam_rejection(iouter, selected_beams, chunk_min, chunk_max)

        # Include iouter in the acqdir so that filenames are unique across
        # iouter turns (same beam+chunk may be requested twice across
        # iterations; the per-turn uniqueness lives in the directory now).
        acqdir = f"test_{iouter}"

        # Compute the expected filenames client-side (via _acq_filename,
        # which mirrors the server's make_acq_relpath).
        expanded = {}   # filename -> chunk index
        for c in range(chunk_min, chunk_max + 1):
            for b in selected_beams:
                fn = _acq_filename(acqdir, b, c)
                expanded[fn] = c
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

        # "S1" (post-request snapshot): rb_streamed is monotone and S1 is
        # read after the RPC was handled, so the server's boundary chunk at
        # handling time was <= ub_chunk.
        ub_chunk = self.rpc_client.get_status().rb_streamed // nbeams

        returned = set(filenames)

        # Every returned filename must be in the requested rectangle, and
        # the list must be sorted ascending by (chunk, beam order in the
        # request) -- the normalization promised by the RPC.
        assert returned <= set(expanded), \
            f"write_files at iouter={iouter}: unrequested filenames {returned - set(expanded)}"
        order = [(self.filename_meta[fn][0], selected_beams.index(self.filename_meta[fn][1]))
                 for fn in filenames]
        assert order == sorted(order), \
            f"write_files at iouter={iouter}: response not sorted: {filenames}"

        # Truncation check: nothing may be returned beyond the largest
        # admissible horizon (sharpest when n_fut == 0, where nothing beyond
        # the capture-boundary chunk may ever be returned; the max() covers
        # that case, where the past path can still return the boundary
        # chunk's already-captured beams but nothing above it).
        never_lo = ub_chunk + max(n_fut - 1, 0) + 1
        overpromised = {fn for fn in returned if expanded[fn] >= never_lo}
        if overpromised:
            raise RuntimeError(
                f"write_files at iouter={iouter}: filenames beyond the truncation horizon "
                f"{overpromised}, chunks=[{chunk_min}, {chunk_max}], beams={selected_beams}, "
                f"n_fut={n_fut}, ub_chunk={ub_chunk}, "
                f"rb=(start={rb_start}, processed={rb_processed}, streamed={rb_streamed}, "
                f"assembled={rb_assembled}, end={rb_end})"
            )

        # Guarantee check: every filename in the guaranteed range MUST be
        # returned. Others may or may not be (both outcomes are accepted).
        guaranteed_this_call = {fn for fn, c in expanded.items()
                                if safe_lower <= c <= guaranteed_upper}
        missing_guaranteed = guaranteed_this_call - returned
        if missing_guaranteed:
            raise RuntimeError(
                f"write_files at iouter={iouter}: missing guaranteed filenames "
                f"{missing_guaranteed}, chunks=[{chunk_min}, {chunk_max}], beams={selected_beams}, "
                f"guaranteed range=[{safe_lower}, {guaranteed_upper}] "
                f"(safe_upper={safe_upper}, lb_chunk={lb_chunk}, n_fut={n_fut}), "
                f"rb=(start={rb_start}, processed={rb_processed}, streamed={rb_streamed}, "
                f"assembled={rb_assembled}, end={rb_end}), "
                f"rb_start_upper={rb_start_upper}, max_wpos={max_wpos}"
            )

        # Bookkeeping for the three running totals, and the flush tail.
        self.safe_written_set.update(guaranteed_this_call)
        for fn, c in expanded.items():
            if fn in guaranteed_this_call:
                continue
            if fn in returned:
                self.unsafe_written_set.add(fn)
            else:
                self.unsafe_not_written_set.add(fn)
        if returned:
            self.max_promised_chunk = max(self.max_promised_chunk,
                                          max(expanded[fn] for fn in returned))

    def _check_duplicate_beam_rejection(self, iouter, selected_beams, chunk_min, chunk_max):
        """write_files with a duplicated beam_id must fail (a validation
        error, rejected before any server state is touched)."""
        spc = self.p['time_samples_per_chunk'] * self.xmd.seq_per_frb_time_sample
        try:
            self.rpc_client.write_files(
                beams          = selected_beams + [selected_beams[0]],
                fpga_seq_start = chunk_min * spc,
                fpga_seq_end   = (chunk_max + 1) * spc,
                acqdir         = f"dup_{iouter}",
            )
        except grpc.RpcError as e:
            assert "more than once" in e.details(), e.details()
        else:
            raise RuntimeError(
                f"write_files with duplicate beam_ids unexpectedly succeeded (iouter={iouter})")

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

    def _flush_promises(self):
        """Future-write promises can extend past the last chunk the send
        loop delivered; without more data those frames would never be
        processed, and _drain_filesub() would block forever. Send a "flush
        tail" -- every worker, real sends, deliberately ignoring the dstate
        fiction (enqueue_send after a disconnect reconnects, as in the main
        loop) -- through the FIRST minichunk of chunk (max_promised_chunk +
        2). The receiver's 2-chunk assembly window advances on the chunk
        index of INCOMING data (first data of chunk c finalizes chunk c-2),
        so that one minichunk is exactly what finalizes max_promised_chunk
        itself; sending only through the end of chunk (max_promised_chunk +
        1) was observed to stall rb_processed one chunk short, at exactly
        max_promised_chunk * nbeams. Then wait (bounded) for the server to
        process past every promised chunk.
        """
        if self.max_promised_chunk < 0:
            return

        # Last minichunk sent: the FIRST of chunk (max_promised_chunk + 2),
        # which finalizes chunk max_promised_chunk (see docstring).
        target_mc = (self.max_promised_chunk + 2) * self.mpc + 1

        for w in range(self.nworkers):
            for imc in range(int(self.wpos[w]), target_mc):
                ichunk = imc // self.mpc
                while self.fspos <= ichunk:
                    self.framesets[self.fspos] = self.client_allocator.get_frame_set(consumer_id=0)
                    self.framesets[self.fspos].randomize(normalize=False, gaussian=False)
                    assert self.framesets[self.fspos].time_chunk_index == self.fspos
                    self.fspos += 1
                self.fxe.enqueue_send_minichunk(w, imc, self.framesets[ichunk])
            if self.wpos[w] < target_mc:
                self.wpos[w] = target_mc

        for w in range(self.nworkers):
            self.fxe.synchronize(w)

        # All flush sends are acked (statuses terminal, as _verify_files
        # requires); now wait for the server-side pipeline to process past
        # every promised chunk, so that every promised file has been queued
        # to the FileWriter before _drain_filesub() starts waiting.
        deadline = time.time() + 60.0
        nbeams = self.p['nbeams']
        while True:
            status = self.rpc_client.get_status()
            if status.rb_processed // nbeams > self.max_promised_chunk:
                return
            if time.time() > deadline:
                raise RuntimeError(
                    f"_flush_promises: rb_processed={status.rb_processed} did not pass "
                    f"max_promised_chunk={self.max_promised_chunk} within 60s"
                )
            time.sleep(0.05)

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
        FileSubscriber stream, checking EXACTLY-ONCE delivery: each
        scheduled filename gets exactly one notification (a duplicate would
        mean a frame was written twice, e.g. by both the WriteFiles past
        path and an anonymous future-write stream -- the bug class the
        rb_streamed boundary excludes), and no unscheduled filename is ever
        written (zero-times and never-promised writes are both bugs).
        _flush_promises() has already pushed rb_processed past every
        promised chunk, so this terminates. No timeout: a stuck write
        manifests as the test blocking forever here.
        """
        scheduled = self.safe_written_set | self.unsafe_written_set
        seen = set()
        remaining = set(scheduled)
        while remaining:
            filename, error_message, stream_name = next(self.file_sub)
            # All writes in this test are WriteFiles-triggered. Anonymous
            # future-write streams notify with stream_name == "" as well --
            # and must arrive WITHOUT subscribe_streams, which this
            # subscription never sets.
            assert stream_name == "", \
                f"unexpected stream-triggered notification {filename!r} (stream_name={stream_name!r})"
            if error_message:
                raise RuntimeError(
                    f"FileSubscriber: write failed for {filename!r}: "
                    f"{error_message}"
                )
            assert filename in scheduled, f"unscheduled file written: {filename!r}"
            assert filename not in seen, f"duplicate write notification for {filename!r}"
            seen.add(filename)
            remaining.discard(filename)

    def _print_summary(self):
        print(f"    guaranteed, written:  {len(self.safe_written_set)}")
        print(f"    unsafe, written:      {len(self.unsafe_written_set)}")
        print(f"    unsafe, not written:  {len(self.unsafe_not_written_set)}")

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
        to the expected buffer reconstructed from client-side state.
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
              f" dtype={c.dtype}, future_write_max_samples={c.future_write_max_samples} (n_fut={t.n_fut})")
        t.run()
    print("    PASSED")
