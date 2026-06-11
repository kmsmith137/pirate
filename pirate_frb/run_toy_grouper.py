"""Toy FrbGrouper consumer(s): print the per-chunk global max of 'out_max'."""

import itertools
import subprocess
import sys
import time


def _run_toy_grouper(grouper_addr, grouper, delay=0.0):
    """Main loop (factored out of run_toy_grouper to reduce nesting).

    For each time chunk, accumulate the global max over all beam-batches, trees,
    DMs and time samples of 'out_max', and print one line per chunk. The cupy
    work runs on the current/default stream (FrbGrouper.__enter__ already
    selected the right device); get_output's __exit__ synchronizes that stream
    before releasing each batch.

    If 'delay' > 0, sleep that many seconds at the end of each chunk -- an
    artificial slowdown for testing how the producer behaves when the consumer
    lags.
    """
    import cupy as cp

    for ichunk in itertools.count():            # loop over time chunks
        running_max = cp.full((1,), -cp.inf, dtype=cp.float32)

        for ibatch in range(grouper.nbatches):  # loop over beam batches
            seq_id = ichunk * grouper.nbatches + ibatch
            with grouper.get_output(seq_id) as outputs:
                # outputs.out_max: list (length ntrees) of cupy arrays (views
                # into the IPC-mapped memory via DLPack). get_output's __exit__
                # synchronizes the current stream before releasing the batch.
                for tree_out in outputs.out_max:        # loop over trees
                    cp.maximum(running_max, tree_out.max(), out=running_max)

        # float() does a D2H copy (+ sync); one print per chunk.
        print(f'{grouper_addr}: ichunk={ichunk}: '
              f'global out_max = {float(running_max[0])}', flush=True)

        if delay > 0:
            time.sleep(delay)


def run_toy_grouper(grouper_addr, delay=0.0):
    """Run a toy FrbGrouper consumer at 'grouper_addr' (e.g. '127.0.0.1:7000').

    Acts as the downstream consumer of an FrbServer producer over CUDA IPC.
    Blocks (in FrbGrouper.open(), via __enter__) until the producer connects,
    then prints the per-chunk global 'out_max' until the producer disconnects or
    Ctrl-C.

    'delay' (seconds) inserts an artificial per-chunk slowdown into the loop;
    see _run_toy_grouper.
    """
    # Imported here (not at module top) so 'import pirate_frb' stays light.
    from .rpc import FrbGrouper

    # FrbGrouper.__enter__ blocks until the producer connects, then pins this
    # thread to the GPU's vcpus and selects the CUDA device (printing a message);
    # __exit__ restores them and closes the grouper.
    with FrbGrouper(grouper_addr) as grouper:
        try:
            _run_toy_grouper(grouper_addr, grouper, delay)
        except KeyboardInterrupt:
            print(f'{grouper_addr}: interrupted; shutting down', flush=True)
        except RuntimeError as e:
            # Most likely the producer disconnected (the grouper stops and
            # acquire_output rethrows). Report cleanly; re-raise anything that is
            # not a stop (e.g. a genuine usage/assert bug).
            if grouper.is_stopped:
                print(f'{grouper_addr}: producer disconnected ({e}); exiting', flush=True)
            else:
                raise
        # FrbGrouper.__exit__ restores affinity/device + closes on every path.

def run_toy_groupers(grouper_addrs, delay=0.0):
    """
    'delay' (seconds) is forwarded to each grouper as an artificial per-chunk
    slowdown (see run_toy_grouper).
    """
    run_groupers(run_toy_grouper, grouper_addrs, (), dict(delay=delay),
                 [sys.executable, '-m', 'pirate_frb', 'run_toy_grouper',
                 '--delay', str(delay)])

def run_groupers(grouper_func, grouper_addrs, args, kwargs, pycommand):
    """Run one or more grouper consumers.

    With a single address, runs "grouper_func" in this process. With more than one, runs each
    grouper in its own child subprocess (re-invoking
    'pycommand' with the <addr> appended, eg, 'pirate_frb run_toy_grouper <addr>');
    if ANY child exits -- for any reason --
    the parent terminates the remaining children and exits. This makes the group
    fail-fast: one grouper going down brings the whole set down.

    """
    if len(grouper_addrs) == 1:
        grouper_func(grouper_addrs[0], *args, **kwargs)
        return

    procs = []   # list of (addr, Popen)
    rc = 0
    try:
        for addr in grouper_addrs:
            # Re-invoke the CLI with a single address -> the child runs the
            # grouper in-process (len==1 branch above). A fresh process (not
            # fork) avoids CUDA-after-fork hazards. stdout is inherited, so each
            # child's messages (which carry its own addr) appear interleaved.
            procs.append((addr, subprocess.Popen(pycommand + [addr])))
        rc = _monitor_children(procs)
    except KeyboardInterrupt:
        print('run_groupers: interrupted; stopping all groupers', flush=True)
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
