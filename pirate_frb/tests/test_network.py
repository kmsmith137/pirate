"""
Network test: FakeXEngine -> FrbServer over 127.0.0.1 loopback.

Constructs an FrbServer (dummy-mode allocator) and a FakeXEngine
(debug=True) in a single Python process, runs a randomized 1000-turn
send loop that produces ragged per-worker progress, and verifies that
none of the real-time debug-mode asserts trigger. Random subscale
parameters per call (see _random_params).

Run via: python -m pirate_frb test --net
"""

import os
import random
import secrets
import shutil

import numpy as np

from ..core import (
    AssembledFrameAllocator,
    FakeXEngine,
    FileWriter,
    Receiver,
    SlabAllocator,
    XEngineMetadata,
)
from ..pirate_pybind11 import FrbServer
from ..rpc import FrbClient


def _random_params():
    """Return one random subscale config (a plain dict)."""
    num_receivers = random.randint(1, 5)
    nworkers      = num_receivers * random.randint(1, 5)
    # total_nfreq must be >= nworkers (FakeXEngine ctor) since
    # frequency channels are assigned round-robin to workers.
    total_nfreq   = max(nworkers, random.randint(8, 32))
    return dict(
        num_receivers          = num_receivers,
        nworkers               = nworkers,
        num_ssd_threads        = random.randint(1, 5),
        num_nfs_threads        = random.randint(1, 5),
        time_samples_per_chunk = 256 * random.randint(1, 5),
        nbeams                 = random.randint(1, 4),
        total_nfreq            = total_nfreq,
        base_beam_id           = random.randint(0, 10000),
        data_base_port         = 5000,
        rpc_port               = 6000,
    )


def test_network():
    """One iteration of the FakeXEngine <-> FrbServer loopback test."""
    print("  test_network()...")

    p = _random_params()
    print(f"    params: {p}")

    # Per-run dirs so concurrent test invocations don't collide and we
    # leave a clean /dev/shm after teardown.
    run_id  = secrets.token_hex(8)
    ssd_dir = f"/dev/shm/pirate_test_network_ssd_{run_id}"
    nfs_dir = f"/dev/shm/pirate_test_network_nfs_{run_id}"
    os.makedirs(ssd_dir, exist_ok=True)
    os.makedirs(nfs_dir, exist_ok=True)

    try:
        # ---- Server side ----
        # Dummy-mode SlabAllocator (capacity=-1): FrbServer skips its
        # reaper thread, frames are allocated lazily on demand.
        slab_allocator = SlabAllocator("af_rhost", -1)

        allocator = AssembledFrameAllocator(
            slab_allocator,
            num_consumers          = p['num_receivers'],
            time_samples_per_chunk = p['time_samples_per_chunk'],
        )

        file_writer = FileWriter(
            ssd_root        = ssd_dir,
            nfs_root        = nfs_dir,
            num_ssd_threads = p['num_ssd_threads'],
            num_nfs_threads = p['num_nfs_threads'],
        )

        receivers = [
            Receiver(
                address     = f"127.0.0.1:{p['data_base_port'] + j}",
                allocator   = allocator,
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

        server = FrbServer(receivers, file_writer,
                           f"127.0.0.1:{p['rpc_port']}")
        server.start()

        # ---- Client side (FakeXEngine, debug=True) ----
        beam_ids = list(range(p['base_beam_id'],
                              p['base_beam_id'] + p['nbeams']))
        xmd = XEngineMetadata.make_test_instance(
            [p['total_nfreq']], [400.0, 800.0], beam_ids,
        )

        ip_addrs = [f"127.0.0.1:{p['data_base_port'] + j}"
                    for j in range(p['num_receivers'])]

        fxe = FakeXEngine(
            xmd, ip_addrs, p['nworkers'],
            time_samples_per_chunk = p['time_samples_per_chunk'],
            debug = True,
        )

        # Per-worker positions (minichunk indices)
        nworkers = p['nworkers']
        ipos0 = np.random.randint(10**10)
        ipos = np.random.randint(ipos0, ipos0+10, size=nworkers, dtype=np.int64)
        wpos = np.copy(ipos)

        # Per-worker "dstate": workers can can be in a temporary "disconnected" state.
        dstate = np.random.random(nworkers) < np.random.uniform(0,1)

        # Keeps track of which (worker_id, minichunk_index) pairs have been skipped.
        skipped = set()

        # Keeps track of which filenames have been scheduled for writing.
        files_written = set()

        # Client-side AssembledFrameSets
        mpc = p['time_samples_per_chunk'] // 256   # minichunks per chunk
        framesets = dict()     # ichunk -> AssembledFrameSet
        fspos = ipos0 // mpc   # chunk index

        # Client-side allocators (distinct from server)
        client_slab_allocator = SlabAllocator("af_rhost", -1)
        client_allocator = AssembledFrameAllocator(
            client_slab_allocator,
            num_consumers = 1,
            time_samples_per_chunk = p['time_samples_per_chunk']
        )

        client_allocator.initialize_metadata(xmd)
        client_allocator.initialize_initial_chunk(fspos)

        # Set up rpc client + file subscription. The subscriber must
        # be opened BEFORE any write_files call to capture every
        # notification (FileSubscriber's constructor blocks on the
        # ready sentinel). We drain the subscription in the main
        # thread AFTER the iouter loop; gRPC's HTTP/2 stream buffers
        # the notifications in the meantime, which is fine at our
        # scale (<= ~90 notifications * ~40 bytes is well below the
        # 64 KB INITIAL_WINDOW_SIZE).
        rpc_client = FrbClient(f"127.0.0.1:{p['rpc_port']}")
        file_sub = rpc_client.subscribe_files()

        # Filenames returned by write_files RPCs across all iouter turns.
        expected_filenames = set()

        # Track chunks we've already written via write_files. The SSD
        # thread reaps frame->data after writing, and the WriteFiles
        # RPC handler silently skips frames with data.size==0 -- so a
        # repeat write_files for an already-written chunk returns
        # fewer filenames than expected, tripping our sanity check.
        # We avoid that by never re-requesting a used chunk.
        used_chunks = set()

        try:
            # Randomized send loop: 1000 turns, each turn picks a
            # random worker, occasionally synchronizes it, and
            # enqueues a Poisson-sized batch of SEND_JUNK commands.
            # The Poisson mean is (1 + 0.1 * lag), where
            # lag = max(wpos) - wpos[worker] -- so workers that
            # have fallen behind catch up faster. This produces
            # ragged per-worker progress (good coverage for the
            # ambiguous band of the ack-prediction check).
            
            for iouter in range(1000):
                worker_id = random.randrange(nworkers)
                skip = dstate[worker_id] or (random.random() < 0.1)

                # Ocassionally synchronize, to prevent workers from getting too out-of-snyc
                if random.random() < 0.1:
                    fxe.synchronize(worker_id)

                # n = Number of minichunks to advance.
                # Computed in a way that biases workers toward catching up with the leader.
                lag = int(np.max(wpos) - wpos[worker_id])
                n = int(np.random.poisson(1.0 + 0.1 * lag))

                # Advance (either skip or send) by n minichunks.
                for k in range(n):
                    imc = int(wpos[worker_id]) + k   # minichunk index
                    ichunk = imc // mpc              # chunk index
                    
                    while fspos <= ichunk:
                        framesets[fspos] = client_allocator.get_frame_set(consumer_id=0)
                        framesets[fspos].randomize()
                        
                        assert framesets[fspos].time_chunk_index == fspos
                        fspos += 1

                    if skip:
                        fxe.enqueue_skip_minichunk(worker_id, imc)
                        skipped.add((worker_id, imc))
                        
                    else:
                        fxe.enqueue_send_minichunk(worker_id, imc, framesets[ichunk])
                
                wpos[worker_id] += n

                # Randomly toggle dstate.
                if dstate[worker_id]:
                    dstate[worker_id] = (random.random() < 0.8)   # 20% reconnection probability
                elif (random.random() < 0.01):                    # 1% disconnection probability
                    fxe.enqueue_disconnect(worker_id)
                    dstate[worker_id] = True

                # Issue rpc write with 1% probability.
                if random.random() > 0.01:
                    continue

                status = rpc_client.get_status()
                rb_start     = status.rb_start
                rb_finalized = status.rb_finalized
                rb_end       = status.rb_end

                # Compute "safe" chunk range. Two bounds:
                #
                # UPPER (rb_finalized): chunks must be FULLY FINALIZED
                # (received from every receiver). Why not just "in
                # ringbuf": frame data is shared across receivers, so
                # a chunk in [rb_finalized, rb_end) is in the ringbuf
                # but still being written by the receivers that haven't
                # delivered yet.
                #
                # LOWER (rb_start, plus a future-bound): chunks must
                # be in the ringbuf at SERVER-processing time, not
                # just snapshot time. The FrbServer worker advances
                # rb_start as new chunks arrive. We bound the
                # worst-case rb_start at server time using:
                #   max_future_rb_end <= (max_wpos // mpc - 1) * nbeams
                #   max_future_rb_start <= max_future_rb_end - rb_size
                RB_NCHUNKS = 512   # = FrbServer::ringbuf_nchunks (constant in FrbServer.hpp)
                rb_size = RB_NCHUNKS * p['nbeams']

                max_wpos = int(np.max(wpos))
                if max_wpos > 0:
                    highest_enqueued_chunk = (max_wpos - 1) // mpc
                    rb_end_upper = max(rb_end, (highest_enqueued_chunk - 1) * p['nbeams'])
                else:
                    rb_end_upper = rb_end
                rb_start_upper = max(rb_start, max(0, rb_end_upper - rb_size))

                safe_lower = (rb_start_upper + p['nbeams'] - 1) // p['nbeams']   # ceil
                safe_upper = (rb_finalized // p['nbeams']) - 1                   # fully-finalized

                if safe_lower > safe_upper:
                    continue   # no safe chunks this turn

                # Pick a contiguous range of 1-3 safe chunks.
                max_nchunks = min(3, safe_upper - safe_lower + 1)
                selected_nchunks = random.randint(1, max_nchunks)
                chunk_min = random.randint(safe_lower, safe_upper - selected_nchunks + 1)
                chunk_max = chunk_min + selected_nchunks - 1

                # If any chunk in the range has already been written via
                # write_files, skip this turn (see used_chunks doc above).
                if any(c in used_chunks for c in range(chunk_min, chunk_max + 1)):
                    continue

                # Pick 1-3 random beams.
                all_beam_ids = list(range(p['base_beam_id'], p['base_beam_id'] + p['nbeams']))
                selected_nbeams = random.randint(1, min(3, p['nbeams']))
                selected_beams = random.sample(all_beam_ids, selected_nbeams)

                # Include iouter in the filename pattern to avoid
                # collisions across rpc writes (same beam+chunk may
                # be requested twice across iterations).
                filename_pattern = f"test_{iouter}_(BEAM)_(CHUNK).asdf"
                filenames = rpc_client.write_files(
                    beams                = selected_beams,
                    min_time_chunk_index = chunk_min,
                    max_time_chunk_index = chunk_max,
                    filename_pattern     = filename_pattern,
                )

                # write_files writes every (chunk, beam) frame in the
                # requested rectangle that's currently in the ringbuf.
                # Our safety bound guarantees all of them are present,
                # so a shortfall means our safety logic was wrong or
                # we hit a race we missed.
                expected_count = selected_nchunks * selected_nbeams
                if len(filenames) != expected_count:
                    raise RuntimeError(
                        f"write_files at iouter={iouter}: got {len(filenames)} filenames, "
                        f"expected {expected_count}. chunks=[{chunk_min}, {chunk_max}], "
                        f"beams={selected_beams}, "
                        f"rb=(start={rb_start}, finalized={rb_finalized}, end={rb_end}), "
                        f"rb_start_upper={rb_start_upper}, max_wpos={max_wpos}"
                    )

                expected_filenames.update(filenames)
                used_chunks.update(range(chunk_min, chunk_max + 1))

            for worker_id in range(nworkers):
                # Block until worker thread has processed all commands,
                # and received all acks.
                fxe.synchronize(worker_id)

                # Check status for each minichunk.
                for imc in range(ipos[worker_id], wpos[worker_id]):
                    status = fxe.get_minichunk_status(worker_id, imc)
                    if (worker_id,imc) in skipped:
                        assert status == FakeXEngine.STATUS_SKIPPED
                    else:
                        assert (status == FakeXEngine.STATUS_DROPPED) or (status == FakeXEngine.STATUS_ASSEMBLED)

            # All acks drained; the counters are now a stable snapshot.
            counters = fxe.get_debug_counters()
            labels = [
                "unambiguous, DROPPED",
                "unambiguous, ASSEMBLED",
                "ambiguous,   DROPPED",
                "ambiguous,   ASSEMBLED",
            ]
            for label, count in zip(labels, counters):
                print(f"    {label}: {count}")

            # Drain the FileSubscriber stream until we've observed
            # a notification for every filename we requested. No
            # timeout (future prompt will add one); a stuck write
            # will manifest as the test blocking forever here --
            # detect via external timeout / Ctrl-C.
            received_filenames = set()
            while not received_filenames.issuperset(expected_filenames):
                filename, error_message = next(file_sub)
                if error_message:
                    raise RuntimeError(
                        f"FileSubscriber: write failed for {filename!r}: "
                        f"{error_message}"
                    )
                received_filenames.add(filename)

            print(f"    files written: {len(expected_filenames)}")
        finally:
            file_sub.close()
            rpc_client.close()
            fxe.stop()
            server.stop()
    finally:
        shutil.rmtree(ssd_dir, ignore_errors=True)
        shutil.rmtree(nfs_dir, ignore_errors=True)

    print("    PASSED")
