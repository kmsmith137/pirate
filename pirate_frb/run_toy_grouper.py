"""Toy FrbGrouper consumer(s): print the per-chunk global max of 'out_max'."""

import itertools
import subprocess
import sys
import time


def _run_toy_grouper(grouper_addr, grouper, stream):
    """Main loop (factored out of run_toy_grouper to reduce nesting).

    For each time chunk, accumulate the global max over all beam-batches, trees,
    DMs and time samples of 'out_max', and print one line per chunk. 'stream' is
    the current cupy stream; we synchronize it before each release so all GPU
    reads of a batch complete before CONSUMED is sent (there is no IPC-event
    fence -- see plans/grouper_server.md).
    """
    import cupy as cp

    for ichunk in itertools.count():            # loop over time chunks
        running_max = cp.full((1,), -cp.inf, dtype=cp.float32)

        for ibatch in range(grouper.nbatches):  # loop over beam batches
            seq_id = ichunk * grouper.nbatches + ibatch
            with grouper.get_output(seq_id) as outputs:
                # outputs.out_max: list (length ntrees) of cupy arrays (views
                # into the IPC-mapped memory via DLPack).
                for tree_out in outputs.out_max:        # loop over trees
                    cp.maximum(running_max, tree_out.max(), out=running_max)
                # Finish GPU reads before __exit__ -> release_output() (CONSUMED).
                stream.synchronize()

        # float() does a D2H copy (+ sync); one print per chunk.
        print(f'{grouper_addr}: ichunk={ichunk}: '
              f'global out_max = {float(running_max[0])}', flush=True)


def run_toy_grouper(grouper_addr):
    """Run a toy FrbGrouper consumer at 'grouper_addr' (e.g. '127.0.0.1:7000').

    Acts as the downstream consumer of an FrbServer producer over CUDA IPC.
    Blocks (in FrbGrouper.open(), via __enter__) until the producer connects,
    then prints the per-chunk global 'out_max' until the producer disconnects or
    Ctrl-C.
    """
    # Heavy / GPU imports are deferred so 'import pirate_frb' stays light and
    # does not require cupy.
    import cupy as cp
    from .rpc import FrbGrouper

    with FrbGrouper(grouper_addr) as grouper:
        # The IPC-mapped output arrays live on grouper.cuda_device_id. Run our
        # cupy reductions on a dedicated stream on that device.
        with cp.cuda.Device(grouper.cuda_device_id):
            stream = cp.cuda.Stream(non_blocking=True)
            with stream:
                try:
                    _run_toy_grouper(grouper_addr, grouper, stream)
                except KeyboardInterrupt:
                    print(f'{grouper_addr}: interrupted; shutting down', flush=True)
                except RuntimeError as e:
                    # Most likely the producer disconnected (the grouper stops and
                    # acquire_output rethrows). Report cleanly; re-raise anything
                    # that is not a stop (e.g. a genuine usage/assert bug).
                    if grouper.is_stopped:
                        print(f'{grouper_addr}: producer disconnected ({e}); exiting', flush=True)
                    else:
                        raise
        # FrbGrouper.__exit__ (close()) runs on the way out on every path.


def run_toy_groupers(grouper_addrs):
    """Run one or more toy grouper consumers.

    With a single address, runs in this process. With more than one, runs each
    grouper in its own child subprocess (re-invoking
    'pirate_frb run_toy_grouper <addr>'); if ANY child exits -- for any reason --
    the parent terminates the remaining children and exits. This makes the group
    fail-fast: one grouper going down brings the whole set down.
    """
    if len(grouper_addrs) == 1:
        run_toy_grouper(grouper_addrs[0])
        return

    procs = []   # list of (addr, Popen)
    rc = 0
    try:
        for addr in grouper_addrs:
            # Re-invoke the CLI with a single address -> the child runs the
            # grouper in-process (len==1 branch above). A fresh process (not
            # fork) avoids CUDA-after-fork hazards. stdout is inherited, so each
            # child's messages (which carry its own addr) appear interleaved.
            procs.append((addr, subprocess.Popen(
                [sys.executable, '-m', 'pirate_frb', 'run_toy_grouper', addr])))
        rc = _monitor_children(procs)
    except KeyboardInterrupt:
        print('run_toy_grouper: interrupted; stopping all groupers', flush=True)
    finally:
        _terminate_children(procs)

    if rc:
        sys.exit(rc)


def _monitor_children(procs):
    """Block until any child exits; return 0 if all dead children exited cleanly,
    else 1. (The caller then terminates the survivors.)"""
    while True:
        dead = [(addr, p) for addr, p in procs if p.poll() is not None]
        if dead:
            for addr, p in dead:
                print(f'run_toy_grouper: child for {addr} exited (code {p.returncode}); '
                      f'stopping the other groupers', flush=True)
            return 0 if all(p.returncode == 0 for _, p in dead) else 1
        time.sleep(0.2)


def _terminate_children(procs, grace_sec=5.0):
    """SIGTERM all still-running children, then SIGKILL any that don't exit
    within grace_sec."""
    for _, p in procs:
        if p.poll() is None:
            p.terminate()
    deadline = time.monotonic() + grace_sec
    for _, p in procs:
        try:
            p.wait(timeout=max(0.0, deadline - time.monotonic()))
        except subprocess.TimeoutExpired:
            p.kill()
            p.wait()
