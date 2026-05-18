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

        try:
            # Randomized send loop: 1000 turns, each turn picks a
            # random worker, occasionally synchronizes it, and
            # enqueues a Poisson-sized batch of SEND_JUNK commands.
            # The Poisson mean is (1 + 0.1 * lag), where
            # lag = max(wpos) - wpos[worker] -- so workers that
            # have fallen behind catch up faster. This produces
            # ragged per-worker progress (good coverage for the
            # ambiguous band of the ack-prediction check).

            nworkers = p['nworkers']
            ipos0 = np.random.randint(10**10)
            ipos = np.random.randint(ipos0, ipos0+10, size=nworkers, dtype=np.int64)
            wpos = np.copy(ipos)

            # Workers can be in a temporary "disconnected" state (dstate).
            dstate = np.random.random(nworkers) < np.random.uniform(0,1)
            
            for _ in range(1000):
                worker_id = random.randrange(nworkers)
                skip = dstate[worker_id] or (random.random() < 0.1)
                
                if random.random() < 0.1:
                    fxe.synchronize(worker_id)
                    
                lag = int(np.max(wpos) - wpos[worker_id])
                n = int(np.random.poisson(1.0 + 0.1 * lag))
                
                for k in range(n):
                    if skip:
                        fxe.enqueue_skip_minichunk(worker_id, int(wpos[worker_id]) + k)
                    else:
                        fxe.enqueue_send_junk(worker_id, int(wpos[worker_id]) + k)
                
                wpos[worker_id] += n

                if dstate[worker_id]:
                    dstate[worker_id] = (random.random() < 0.8)   # 20% reconnection probability
                elif (random.random() < 0.01):                    # 1% disconnection probability
                    fxe.enqueue_disconnect(worker_id)
                    dstate[worker_id] = False

            # synchronize(w) blocks until worker w's command queue is
            # empty; in debug=True mode it also enqueues WAIT_FOR_ACKS
            # first, so the wait covers all outstanding FLAG_ACK acks.
            # If a debug-mode assertion inside _read_acks fired, the
            # worker latched that error -- synchronize() rethrows it
            # to this thread.
            for w in range(nworkers):
                fxe.synchronize(w)

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
        finally:
            fxe.stop()
            server.stop()
    finally:
        shutil.rmtree(ssd_dir, ignore_errors=True)
        shutil.rmtree(nfs_dir, ignore_errors=True)

    print("    PASSED")
