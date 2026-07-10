"""
End-to-end server test: FakeXEngine -> FrbServer -> GpuDedisperser -> FrbGrouper.

Like 'test --dd', this compares GPU dedispersion outputs against a CPU reference,
but end-to-end: random AssembledFrames are sent over the loopback network to an
FrbServer (which dequantizes + dedisperses on the GPU and hands its outputs to an
FrbGrouper over CUDA IPC), while the same frames are fed through a
ReferenceDequantizationKernel + ReferenceDedisperser in the test process. The
per-(chunk, batch, tree) outputs (out_max, out_argmax) are compared the same way
GpuDedisperser::test_one() compares its GPU/CPU outputs.

Process layout: CUDA IPC does not allow opening a memory handle in the process
that exported it (cudaIpcOpenMemHandle fails on a same-process handle), so the
grouper consumer runs in a CHILD process (multiprocessing 'spawn' -- fork is
unsafe once the parent has touched CUDA). The child just loops
FrbGrouper.get_output(), host-copies the outputs, and ships them to the parent
over a multiprocessing.Queue; the parent does everything else, single-threaded.

Sequencing subtleties (see also the Receiver's 2-chunk assembly window,
Receiver.hpp): chunk c only becomes fully assembled -- and hence dedispersed --
once data from chunk c+2 arrives at each Receiver. The parent therefore
interleaves "send chunk c" with "compare chunk c-2", and finishes by sending 2
junk chunks to flush the last 2 real chunks. Peak-finding weights are
randomized ONCE, in the window after the FrbServer publishes its dedisperser
(which happens once chunk 0's data has arrived) but before anything is
assembled (which requires chunk 2's data) -- so the fill cannot race the
dedispersion kernels.

Run via: python -m pirate_frb test --serv
"""

import math
import os
import queue as _queue
import random
import secrets
import shutil
import time
import multiprocessing

import ksgpu
import numpy as np

from ..core import (
    AssembledFrameAllocator,
    BumpAllocator,
    CudaStreamPool,
    FakeXEngine,
    FileWriter,
    Receiver,
    SimulatedFrameFactory,
    SlabAllocator,
    XEngineMetadata,
)
from ..pirate_pybind11 import (
    DedispersionPlan,
    FrbServer,
    FrbGrouperClient,
    GpuDedisperser,
    ReferenceDedisperser,
    ReferenceDequantizationKernel,
)
from ..Hardware import Hardware
from ..utils import ThreadAffinity
from .utils import make_random_subscale_config, pick_receiver_worker_counts


def _grouper_child_main(grouper_addr, nchunks, out_queue, shutdown_event):
    """Child-process main: the ad hoc grouper consumer.

    Blocks in FrbGrouper.__enter__ until the producer (FrbServer) completes the
    handshake, then consumes every (ichunk, ibatch) output in order, host-copies
    the per-tree arrays, and ships them to the parent. Runs in a separate
    process because CUDA IPC cannot map a same-process handle (see module
    docstring). All messages are tuples whose first element is a tag:
    ('handshake', ...), ('out', ...), ('done',), or ('error', traceback_str).
    """
    # Ignore SIGINT: on Ctrl-C the whole process group gets it, but the parent
    # must orchestrate this child's shutdown (stop the server first, then release
    # us via shutdown_event -- see ServerTester.__exit__). Tearing ourselves down
    # on SIGINT would close the grouper stream while the server is still running,
    # which the server would report as an unexpected disconnect.
    import signal
    signal.signal(signal.SIGINT, signal.SIG_IGN)
    try:
        import cupy as cp
        from pirate_frb.rpc import FrbGrouper

        with FrbGrouper(grouper_addr) as g:
            out_queue.put(('handshake', g.nbatches, g.initial_chunk, g.ntrees))
            for ichunk in range(nchunks):
                for ibatch in range(g.nbatches):
                    with g.get_output(ichunk, ibatch) as out:
                        assert out.ichunk_zero_based == ichunk
                        # Host-copy INSIDE the context manager -- the arrays are
                        # raw views of the IPC-mapped ring buffer, only valid here.
                        gpu_max = [cp.asnumpy(a) for a in out.out_max]
                        gpu_tok = [cp.asnumpy(a) for a in out.out_argmax]
                    out_queue.put(('out', ichunk, ibatch, gpu_max, gpu_tok))
            out_queue.put(('done',))
            # Hold the grouper Session open until the parent has stopped the
            # server. Then the server's receive thread is unblocked by its own
            # stop() (is_stopped=True -> quiet) rather than by us disconnecting
            # first (which it would report as an unexpected stream close).
            shutdown_event.wait()
    except BaseException:
        import traceback
        out_queue.put(('error', traceback.format_exc()))


class ServerTester:
    """Drives one end-to-end FakeXEngine -> FrbServer -> FrbGrouper test.

    Use as a context manager: __init__ picks random params (no side effects),
    __enter__ creates /dev/shm subdirs and builds/starts everything, __exit__
    stops everything and rmtree's. run() executes the send/compare phases.
    """

    # ---- Construction + lifecycle ----

    @staticmethod
    def _random_params():
        """Return one random subscale config (a plain dict). Config-first, like
        test --net (see tests/utils.py); no_dedispersion/pacing are never used
        here, and a grouper is always configured.

        Extra rejection constraint vs test --net: a grouper-enabled FrbServer
        builds its dedisperser with nbatches_out = 2*num_active_batches, and
        FrbGrouper requires the output ring to fit within one chunk
        (num_batch_slots * beams_per_batch <= total_beams; FrbGrouper.cpp) --
        so require 2 * num_active_batches * beams_per_batch <= beams_per_gpu.
        """
        for _ in range(200):
            config = make_random_subscale_config()
            if 2 * config.num_active_batches * config.beams_per_batch <= config.beams_per_gpu:
                break
        else:
            raise RuntimeError(
                "test_server: failed to generate a random DedispersionConfig with "
                "2*num_active_batches*beams_per_batch <= beams_per_gpu in 200 attempts"
            )
        total_nfreq = sum(config.zone_nfreq)
        num_receivers, nworkers = pick_receiver_worker_counts(total_nfreq)
        nab = config.num_active_batches

        return dict(
            config                 = config,
            num_receivers          = num_receivers,
            nworkers               = nworkers,
            time_samples_per_chunk = config.time_samples_per_chunk,
            nbeams                 = config.beams_per_gpu,
            total_nfreq            = total_nfreq,
            base_beam_id           = random.randint(0, 10000),
            # Weight-ring depth, randomized as in 'test --dd': uniform in
            # [nstreams, 2*nbatches_out) with nbatches_out = 2*nab (FrbServer-fixed).
            nbatches_wt            = random.randint(nab, 4*nab - 1),
            # Number of real (verified) data chunks. >= 5 guarantees the weight
            # ring wraps (nchunks*nbatches > nbatches_wt, since nbatches >= nab).
            nchunks                = random.randint(6, 12),
            # Initial time-chunk index (chunk-aligned start; workers all begin
            # at minichunk c0*mpc, so FrbServer's initial_chunk == c0).
            c0                     = random.randint(0, 1000),
            # Distinct port ranges from test --net, in case both run in one process.
            data_base_port         = 5100,
            rpc_port               = 6100,
            grouper_port           = 7100,
            ringbuf_nchunks        = random.randint(16, 32),
            num_ssd_threads        = random.randint(1, 3),
            num_nfs_threads        = random.randint(1, 3),
        )

    def __init__(self):
        self.p = ServerTester._random_params()
        self.run_id = secrets.token_hex(8)
        self.ssd_dir = f"/dev/shm/pirate_test_server_ssd_{self.run_id}"
        self.nfs_dir = f"/dev/shm/pirate_test_server_nfs_{self.run_id}"

        # Derived geometry (used throughout).
        p = self.p
        self.B        = p['config'].beams_per_batch
        self.nbatches = p['nbeams'] // self.B
        self.mpc      = p['time_samples_per_chunk'] // 256
        self.c0       = p['c0']
        self.nbw      = p['nbatches_wt']
        self.nchunks  = p['nchunks']

        # Filled by __enter__ / run(); listed so __exit__ can blindly None-check.
        self.receivers   = None
        self.file_writer = None
        self.server      = None
        self.grouper_client = None
        self.child       = None
        self.queue       = None
        self.shutdown_event = None
        self.xmd         = None
        self.fxe         = None
        self.factory     = None
        self.framesets   = {}     # absolute chunk index -> AssembledFrameSet
        # Reference-side state (built in _setup_reference).
        self.wt          = None
        self.rdd         = None
        self.rdqk        = None
        self.pf_kernels  = None
        self.pf_tmp      = None
        self.dq_out      = None
        self.eps         = None
        self.ntrees      = None

    def __enter__(self):
        os.makedirs(self.ssd_dir, exist_ok=True)
        os.makedirs(self.nfs_dir, exist_ok=True)
        try:
            self._build_server()
            self._spawn_grouper_child()
            # Ping the grouper before start(): this both fails fast if it's not
            # coming up and (unlike run_server, where the grouper is already
            # running) WAITS for the freshly-spawned child to import cupy + bind.
            # A generous timeout, since start()'s reconnect is only bounded at 2s.
            self.grouper_client.ping(timeout_ms=60000)
            self.server.start()
            # With a grouper, FrbServer starts its Receivers only after the
            # grouper connection is READY (grouper_send_thread). FakeXEngine's
            # lazy connect throws on ECONNREFUSED, so wait here until every
            # Receiver is accepting before any minichunk can be sent.
            for r in self.receivers:
                self._wait_until_listening(r)
            self._build_client()
        except BaseException:
            self.__exit__(None, None, None)
            raise
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        try:
            if self.fxe is not None:
                self.fxe.stop()
            if self.server is not None:
                # Stop the server FIRST (before releasing the child): this sets
                # is_stopped and cancels the grouper Session, so the server's
                # grouper receive thread exits quietly instead of reporting the
                # child's disconnect as unexpected. Also wakes a child still
                # blocked in get_output() on the failure paths.
                self.server.stop()
            if self.shutdown_event is not None:
                # Release the child from its post-'done' wait, now that the
                # server is stopped, so it disconnects cleanly.
                self.shutdown_event.set()
            if self.child is not None:
                # Clean shutdown returns promptly; the terminate() is a fallback
                # for a child wedged in a C++ wait (e.g. grouper never connected).
                self.child.join(timeout=10)
                if self.child.is_alive():
                    self.child.terminate()
                    self.child.join(timeout=10)
            if self.factory is not None:
                self.factory.stop()
        finally:
            shutil.rmtree(self.ssd_dir, ignore_errors=True)
            shutil.rmtree(self.nfs_dir, ignore_errors=True)

    # ---- Private builders ----

    def _compute_gpu_nbytes(self):
        """Exact GPU BumpAllocator capacity for the FrbServer.

        The grouper handshake exports the gpu_allocator's base over CUDA IPC, so
        (unlike test --net) the allocator must be real (capacity > 0). Since the
        test builds the DedispersionConfig first and derives the XMD from it,
        config_postfilled == config, and the footprint can be computed exactly:
        a throwaway unallocated GpuDedisperser with the same Params gives the
        gmem footprint, plus the processing thread's per-stream scratch
        (int4_data_gpu + scales_offsets_gpu; see FrbServer.cpp) and a little
        alignment slop (BumpAllocator aligns each allocation to 128 bytes).
        """
        p = self.p
        config = p['config']
        nab = config.num_active_batches
        S, B, F, T = nab, self.B, p['total_nfreq'], p['time_samples_per_chunk']

        plan_tmp = DedispersionPlan(config)
        sp_tmp = CudaStreamPool(nab)
        dd_tmp = GpuDedisperser(plan_tmp, sp_tmp, cuda_device_id=0, num_consumers=1,
                                nbatches_out=2*nab, nbatches_wt=self.nbw)
        gmem = dd_tmp.resource_tracker.get_gmem_footprint()

        scratch = S * B * F * (T // 2)                  # int4_data_gpu
        scratch += S * B * F * (T // 256) * 2 * 2       # scales_offsets_gpu (fp16)
        return gmem + scratch + (1 << 16)               # alignment slop

    def _build_server(self):
        p = self.p

        # Dummy-mode SlabAllocator (capacity=-1): FrbServer skips its reaper
        # thread, frames are allocated lazily on demand.
        slab_allocator = SlabAllocator("af_rhost", -1)
        allocator = AssembledFrameAllocator(
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
                allocator   = allocator,
                consumer_id = j,
                misbehaving_reads = True,   # stress the parser; doesn't alter data
            )
            for j in range(p['num_receivers'])
        ]

        # Host allocator: dummy mode (as in test --net). GPU allocator: real,
        # exactly sized (the grouper handshake needs a real IPC-able base, and
        # the dedisperser's output_ringbuf must be its first allocation --
        # hence a fresh, dedicated allocator).
        host_alloc = BumpAllocator(ksgpu.af_rhost | ksgpu.af_zero, -1)
        gpu_alloc  = BumpAllocator(ksgpu.af_gpu | ksgpu.af_zero,
                                   self._compute_gpu_nbytes(), cuda_device=0)

        # Producer-side grouper connection (constructed now, pinged after the
        # grouper child is spawned; see __enter__). FrbServer opens the real
        # connection + Handshake later, from its grouper send thread.
        self.grouper_client = FrbGrouperClient(f"127.0.0.1:{p['grouper_port']}")

        self.server = FrbServer(p['config'], self.receivers, self.file_writer,
                                f"127.0.0.1:{p['rpc_port']}",
                                p['ringbuf_nchunks'],
                                min_data_mtu=1500,
                                host_allocator=host_alloc,
                                gpu_allocator=gpu_alloc,
                                cuda_device_id=0,
                                grouper_client=self.grouper_client,
                                nbatches_wt=p['nbatches_wt'],
                                quiet=True)

    def _spawn_grouper_child(self):
        # 'spawn' (not fork): the parent has already initialized CUDA, and a
        # forked child would inherit a broken CUDA context.
        ctx = multiprocessing.get_context('spawn')
        self.queue = ctx.Queue()
        # Set (by __exit__, after server.stop()) to release the child from its
        # post-'done' wait so it disconnects cleanly. See _grouper_child_main.
        self.shutdown_event = ctx.Event()
        # daemon=True: if the parent dies unexpectedly, the child is auto-killed
        # rather than left orphaned (blocked in the grouper).
        self.child = ctx.Process(
            target=_grouper_child_main,
            args=(f"127.0.0.1:{self.p['grouper_port']}", self.nchunks,
                  self.queue, self.shutdown_event),
            daemon=True,
        )
        self.child.start()

    def _build_client(self):
        p = self.p
        beam_ids = list(range(p['base_beam_id'], p['base_beam_id'] + p['nbeams']))

        # make_fiducial defaults noise_variance to 1.0 per zone, so the
        # normalized factory frames have unit variance (keeps the fp16
        # dedispersion in a good dynamic range).
        self.xmd = XEngineMetadata.make_fiducial(
            p['config'].zone_nfreq, p['config'].zone_freq_edges, beam_ids, 1.0)

        ip_addrs = [f"127.0.0.1:{p['data_base_port'] + j}"
                    for j in range(p['num_receivers'])]

        self.fxe = FakeXEngine(
            self.xmd, ip_addrs, p['nworkers'],
            time_samples_per_chunk = p['time_samples_per_chunk'],
            debug = True,
            paced = False,
            rpc_address = f"127.0.0.1:{p['rpc_port']}",
        )

        # Client-side frame source: dummy-mode allocator (lazily allocated
        # frames, freed on refcount-zero -- the test only retains a ~3-chunk
        # window) + a SimulatedFrameFactory producing normalized, non-gaussian
        # frames with no injected pulses. The factory must be constructed
        # inside a ThreadAffinity context (its threads inherit the affinity).
        client_slab = SlabAllocator("af_rhost", -1)
        client_alloc = AssembledFrameAllocator(
            client_slab, num_consumers=1,
            time_samples_per_chunk=p['time_samples_per_chunk'])
        client_alloc.initialize_metadata(self.xmd)
        client_alloc.initialize_initial_chunk(self.c0)

        vcpu_list = Hardware().vcpu_list_from_cpu(0)
        with ThreadAffinity(vcpu_list):
            self.factory = SimulatedFrameFactory(
                client_alloc,
                num_randomizer_threads=2,
                normalized=True,
                gaussian=False,
            )

    # ---- Blocking-wait helpers (poll so the parent stays interruptible) ----

    def _check_child_alive(self):
        """Raise if the grouper child died. The parent's waits below all depend
        on the child being alive (the server's Receivers only start once the
        child's grouper connects), so a dead child would otherwise hang us."""
        if self.child is not None and not self.child.is_alive():
            raise RuntimeError(
                f"test_server: grouper child exited unexpectedly "
                f"(exitcode={self.child.exitcode})")

    def _wait_until_listening(self, r, timeout=60.0):
        """Wait for Receiver r to start listening, polling with a finite C++
        timeout so the parent processes signals (Ctrl-C) and can detect a dead
        child instead of blocking forever in the GIL-released C++ call."""
        t0 = time.monotonic()
        while not r.wait_until_listening(timeout_sec=0.5):
            self._check_child_alive()
            if time.monotonic() - t0 > timeout:
                raise RuntimeError("test_server: timed out waiting for a Receiver "
                                   "to start listening (grouper never connected?)")

    # ---- Send helpers ----

    def _send_chunk(self, c):
        """Fetch the frame set for absolute chunk c from the factory, enqueue all
        its minichunks on every worker, and barrier on the acks. The per-chunk
        barrier guarantees zero drops: a minichunk can only be DROPPED if its
        worker lags >= 2 chunks behind the receiver's leading edge, which the
        barrier makes impossible. (Acks are per-minichunk and immediate, so this
        does not wait on the 2-chunk assembly flush -- no deadlock.)"""
        fs = self.factory.get_frame_set()
        assert fs.time_chunk_index == c
        self.framesets[c] = fs
        for w in range(self.p['nworkers']):
            for imc in range(c * self.mpc, (c + 1) * self.mpc):
                self.fxe.enqueue_send_minichunk(w, imc, fs)
        for w in range(self.p['nworkers']):
            self.fxe.synchronize(w)

    def _send_junk_chunk(self, c):
        """Send junk minichunks for the whole of chunk c from every worker.
        The first data of chunk c is what advances each Receiver's 2-chunk
        assembly window past chunk c-2; the junk contents are never verified
        (and chunk c itself never assembles, since c+2 is never sent). All
        mpc minichunks are sent because FakeXEngine enforces strict +1
        minichunk-index monotonicity per worker."""
        for w in range(self.p['nworkers']):
            for imc in range(c * self.mpc, (c + 1) * self.mpc):
                self.fxe.enqueue_send_junk(w, imc)
        for w in range(self.p['nworkers']):
            self.fxe.synchronize(w)

    # ---- Reference-side setup ----

    def _wait_for_dedisperser(self, timeout=120.0):
        """Poll until the FrbServer's processing thread publishes its
        GpuDedisperser (which happens once chunk 0's data has arrived)."""
        t0 = time.monotonic()
        while True:
            dd = self.server.dedisperser
            if dd is not None:
                return dd
            if self.server.is_stopped:
                raise RuntimeError("test_server: FrbServer stopped while waiting "
                                   "for its dedisperser")
            self._check_child_alive()
            if time.monotonic() - t0 > timeout:
                raise RuntimeError("test_server: timed out waiting for the "
                                   "FrbServer's dedisperser")
            time.sleep(0.1)

    def _setup_reference(self):
        """Fill random peak-finding weights (once), and build the reference
        dequantizer + dedisperser.

        Called after chunk 0 was sent (so the server's dedisperser exists) but
        before chunk 1 is sent. Nothing can be assembled until chunk 2's data
        arrives (2-chunk Receiver window), so the dedispersion kernels cannot
        yet be running and the weight fill is race-free -- it simply overwrites
        the analytic weights the processing thread installed at startup.
        """
        p = self.p
        dd = self._wait_for_dedisperser()
        plan = self.server.plan
        trees = dd.trees
        self.ntrees = len(trees)

        # Random weights, one fill per tree covering all nbatches_wt slots.
        # Pre-round to fp16-representable values: for fp16 configs the GPU
        # weight layout stores fp16, while the reference uses fp32 -- rounding
        # here makes both sides use bit-identical weight values. (Harmless for
        # fp32 configs: both sides then use the same fp32 values either way.)
        rng = np.random.default_rng()
        self.wt = []
        for t, tree in enumerate(trees):
            shape = (self.nbw, self.B, tree.ndm_wt, tree.nt_wt,
                     tree.nprofiles, tree.frequency_subbands.N)
            w = rng.random(size=shape, dtype=np.float32)
            w = w.astype(np.float16).astype(np.float32)
            dd.fill_all_weights(t, w)
            self.wt.append(w)

        # Reference chain. Dcore from the GPU kernels makes the reference
        # peak-finder mimic the GPU exactly (same convention as test_one).
        self.rdd = ReferenceDedisperser(plan, sophistication=2, Dcore=dd.Dcore)
        self.rdqk = ReferenceDequantizationKernel(self.B, p['total_nfreq'],
                                                  p['time_samples_per_chunk'])
        self.pf_kernels = self.rdd.pf_kernels
        self.pf_tmp = [np.zeros((self.B, tr.ndm_out, tr.nt_out), dtype=np.float32)
                       for tr in trees]
        self.dq_out = np.zeros((self.B, p['total_nfreq'], p['time_samples_per_chunk']),
                               dtype=np.float32)

        # Comparison threshold, per tree -- same formula as test_one, but with
        # coefficient 5 instead of 3:
        #   eps = 5 * dtype.precision * sqrt(n+2),
        #   n = ds_level + amb_rank + early_dd_rank.
        # The 3-sigma-style bound was flaky in this end-to-end test (~1-in-3
        # runs failed on a single element, with |delta| up to ~1.45x the
        # threshold, at a random position with a random config each time).
        # Real bugs produce order-unity errors, so the looser bound loses
        # essentially no detection power.
        prec = p['config'].dtype.precision
        self.eps = [5.0 * prec * math.sqrt(tr.primary_tree_index + tr.total_rank() + 2)

        # First child message: the handshake echo (arrives once the producer's
        # handshake completes; the queue orders it before any outputs).
        msg = self._queue_get()
        assert msg[0] == 'handshake', f"expected handshake message, got {msg[0]!r}"
        _, g_nbatches, g_initial_chunk, g_ntrees = msg
        assert g_nbatches == self.nbatches
        assert g_initial_chunk == self.c0
        assert g_ntrees == self.ntrees

    # ---- Compare ----

    def _queue_get(self, timeout=180.0):
        # Poll with a short block so the parent stays interruptible (Ctrl-C) and
        # notices if the child died without sending anything (e.g. SIGKILL).
        t0 = time.monotonic()
        while True:
            try:
                msg = self.queue.get(timeout=1.0)
                break
            except _queue.Empty:
                self._check_child_alive()
                if time.monotonic() - t0 > timeout:
                    raise RuntimeError("test_server: timed out waiting for a "
                                       "grouper-child message (pipeline wedged?)") from None
        if msg[0] == 'error':
            raise RuntimeError(f"test_server: grouper child failed:\n{msg[1]}")
        return msg

    def _compare_chunk(self, z):
        """Compare GPU vs reference outputs for zero-based chunk z.

        Runs the reference (dequantize + dedisperse) for each batch of the
        chunk, pops the child's corresponding output message, and compares
        out_max directly and out_argmax via eval_tokens -- exactly the
        GPU-vs-reference comparison in GpuDedisperser::test_one().
        Returns the max |gpu - ref| out_max deviation over the chunk.
        """
        fs = self.framesets.pop(self.c0 + z)
        maxdiff = 0.0

        for ibatch in range(self.nbatches):
            seq = z * self.nbatches + ibatch

            # Reference input: dequantize the same bytes the server assembled.
            frames = [fs.frames[ibatch * self.B + k] for k in range(self.B)]
            data  = np.stack([np.asarray(fr.data)           for fr in frames])
            scoff = np.stack([np.asarray(fr.scales_offsets) for fr in frames])
            self.rdqk.apply(self.dq_out, scoff, data)
            np.asarray(self.rdd.input_array)[...] = self.dq_out

            # Reference weights: mirror the GPU's slot choice (iwt = seq % nbatches_wt).
            for t in range(self.ntrees):
                np.asarray(self.rdd.wt_arrays[t])[...] = self.wt[t][seq % self.nbw]

            self.rdd.dedisperse(z, ibatch)

            # GPU outputs from the child (strictly ordered).
            msg = self._queue_get()
            assert msg[0] == 'out', f"expected output message, got {msg[0]!r}"
            _, zi, bi, gpu_max, gpu_tok = msg
            assert (zi, bi) == (z, ibatch), \
                f"out-of-order grouper output: got {(zi, bi)}, expected {(z, ibatch)}"

            for t in range(self.ntrees):
                wt_ref = np.asarray(self.rdd.wt_arrays[t])
                d = ksgpu.assert_arrays_equal(
                    self.rdd.out_max[t], gpu_max[t], "ref_max", "gpu_max",
                    ["beam", "pfdm", "pft"], epsabs=self.eps[t], epsrel=self.eps[t])
                maxdiff = max(maxdiff, d)

                # out_argmax: evaluate the GPU tokens with the reference kernel +
                # weights; must reproduce the reference max (within eps).
                self.pf_kernels[t].eval_tokens(self.pf_tmp[t], gpu_tok[t], wt_ref)
                d = ksgpu.assert_arrays_equal(
                    self.rdd.out_max[t], self.pf_tmp[t], "ref_max", "gpu_tokens",
                    ["beam", "pfdm", "pft"], epsabs=self.eps[t], epsrel=self.eps[t])
                maxdiff = max(maxdiff, d)

        return maxdiff

    # ---- Top-level driver ----

    def run(self):
        p = self.p

        # Chunk 0 first: its data triggers metadata -> plan -> dedisperser on
        # the server. Then fill the weights + build the reference in the
        # race-free window before chunk 1 (see _setup_reference).
        self._send_chunk(self.c0)
        self._setup_reference()

        # Steady state: send chunk c, then compare chunk c-2 (just flushed by
        # c's data). The blocking queue.get inside _compare_chunk provides the
        # backpressure that keeps only ~3 chunks in flight.
        for c in range(self.c0 + 1, self.c0 + self.nchunks):
            self._send_chunk(c)
            z = c - self.c0 - 2
            if z >= 0:
                maxdiff = self._compare_chunk(z)
                print(f"    chunk {z}/{self.nchunks}: OK (max |gpu-ref| = {maxdiff:.3g})")

        # Tail: two junk chunks flush the last two real chunks (and nothing
        # more -- junk chunk N would itself need chunk N+2 to assemble).
        for j in range(2):
            self._send_junk_chunk(self.c0 + self.nchunks + j)
            z = self.nchunks - 2 + j
            maxdiff = self._compare_chunk(z)
            print(f"    chunk {z}/{self.nchunks}: OK (max |gpu-ref| = {maxdiff:.3g})")

        # Child exits its with-block after nchunks chunks and reports 'done'.
        msg = self._queue_get()
        assert msg[0] == 'done', f"expected done message, got {msg[0]!r}"

        # Belt-and-braces: every real minichunk must have been assembled
        # (a drop would have already shown up as a data mismatch above).
        for w in range(p['nworkers']):
            for imc in range(self.c0 * self.mpc, (self.c0 + self.nchunks) * self.mpc):
                status = self.fxe.get_minichunk_status(w, imc)
                assert status == FakeXEngine.STATUS_ASSEMBLED, \
                    f"worker {w}, minichunk {imc}: status {status} != ASSEMBLED"


def test_server():
    """One iteration of the end-to-end FakeXEngine -> FrbServer -> FrbGrouper test."""
    print("  test_server()...")
    with ServerTester() as t:
        params_no_config = {k: v for k, v in t.p.items() if k != 'config'}
        print(f"    params: {params_no_config}")
        c = t.p['config']
        print(f"    config: toplevel_tree_rank={c.toplevel_tree_rank}, num_primary_trees={c.num_primary_trees},"
              f" beams_per_batch={c.beams_per_batch}, num_active_batches={c.num_active_batches},"
              f" dtype={c.dtype}")
        t.run()
    print("    PASSED")
