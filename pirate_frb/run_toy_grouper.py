"""Toy FrbGrouper consumer(s): per-chunk peak SNR + argmax, optionally reported to a sifter."""

import contextlib
import itertools
import subprocess
import sys
import time


def _run_toy_grouper(grouper, sifter=None, delay=0.0):
    """Main loop (factored out of run_toy_grouper to reduce nesting).

    For each time chunk, scan all beam-batches/trees/DMs/time-samples of
    'out_max' and compute, ENTIRELY ON THE GPU (no device->host copies in the
    inner loops):

      - the single highest-SNR "event", and which (tree, ibeam, idm, itime) it
        came from -- all ARRAY INDICES (not physical units). 'ibeam' is a GLOBAL
        beam index (0 <= ibeam < total_beams), formed by adding
        ibatch*beams_per_batch to the per-batch beam index; idm/itime index the
        winning tree's output dm/time axes.
      - the per-beam maximum SNR (a length-total_beams array; no per-beam argmax).

    Only after the chunk's GPU scan finishes do we copy the winning event's
    indices/value to the host (one small copy) to print it and -- if a sifter is
    configured -- send it.

    'sifter' is an already-connected FrbSifterClient (constructed by the caller),
    or None. If not None, we send one check_configuration message up front, then
    one FrbEvents message per chunk: a single event (the chunk's peak), with
    coarsegrain_snr set to the per-beam max. grouper.create_events() converts the
    peak's array indices into a FrbSifterEvents (physical fields + chunk_fpga_count)
    using the X-engine metadata (timing, beam_ids, beamset) and the winning tree's
    dm_min/dm_max in the dedispersion plan.

    The cupy work runs on the current/default stream (FrbGrouper.__enter__ already
    selected the right device); get_output's __exit__ synchronizes that stream
    before releasing each batch.

    If 'delay' > 0, sleep that many seconds at the end of each chunk -- an
    artificial slowdown for testing how the producer behaves when the consumer
    lags.
    """
    import cupy as cp

    nbeams_tot = grouper.total_beams
    nbatches = grouper.nbatches
    beams_per_batch = nbeams_tot // nbatches

    if sifter is not None:
        # Metadata for the FrbEvents messages. The yaml strings are parsed once by
        # FrbGrouper.__enter__; the per-event index -> physical-unit conversion is
        # done by grouper.create_events() below. We still need beam_set_id and
        # fpga_per_chunk here (for the coarsegrain time window).
        xengine = grouper.xengine_yaml
        beam_set_id = xengine['beamset']
        fpga_per_chunk = xengine['seq_per_frb_time_sample'] * grouper.nt_in

        sifter.check_configuration(
            pirate_yaml = grouper.dedispersion_config_yaml_string,
            xengine_yaml = grouper.xengine_metadata_yaml_string,
            dedispersion_plan_yaml = grouper.dedispersion_plan_yaml_string,
            grouper_yaml = {'toy_grouper': True})
        print(f'{grouper.grouper_ip_addr}: connected to sifter at {sifter.server_address}, '
              f'sent check_configuration', flush=True)

    for ichunk in itertools.count():            # loop over time chunks
        # GPU-resident accumulators, updated entirely on the GPU below (no
        # device->host copies until the chunk's scan is complete). gmax is the
        # running peak SNR; the g_* 0-d arrays hold its (tree, ibeam, idm, itime)
        # array indices.
        gmax    = cp.full((), -cp.inf, dtype=cp.float32)
        g_tree  = cp.full((), -1, dtype=cp.int64)
        g_ibeam = cp.full((), -1, dtype=cp.int64)
        g_idm   = cp.full((), -1, dtype=cp.int64)
        g_itime = cp.full((), -1, dtype=cp.int64)
        per_beam_max = cp.full((nbeams_tot,), -cp.inf, dtype=cp.float32)

        for ibatch in range(nbatches):          # loop over beam batches
            beam0 = ibatch * beams_per_batch    # global index of this batch's first beam
            with grouper.get_output(ichunk, ibatch) as outputs:
                # outputs.out_max: list (length ntrees) of cupy arrays, each
                # shape (beams_per_batch, ndm, nt), viewing IPC-mapped GPU
                # memory. get_output's __exit__ synchronizes the current stream
                # before releasing the batch.
                for itree, tree_out in enumerate(outputs.out_max):   # loop over trees
                    t = cp.asarray(tree_out)    # cupy view (no-op if already cupy)

                    # Per-beam max (over dm,time) for this batch's beams.
                    sl = per_beam_max[beam0 : beam0 + beams_per_batch]
                    cp.maximum(sl, t.max(axis=(1, 2)), out=sl)

                    # Global peak + argmax candidate from this tree (all GPU).
                    # unravel gives (local beam, dm, time) ARRAY INDICES.
                    m = t.max()
                    lb, idm, itime = cp.unravel_index(t.argmax(), t.shape)
                    upd = m > gmax
                    gmax    = cp.where(upd, m, gmax)
                    g_tree  = cp.where(upd, itree, g_tree)
                    g_ibeam = cp.where(upd, beam0 + lb, g_ibeam)
                    g_idm   = cp.where(upd, idm, g_idm)
                    g_itime = cp.where(upd, itime, g_itime)

        # Chunk scan done. One small device->host copy of the peak's indices/value.
        tree, ibeam, idm, itime = (int(v) for v in
                                   cp.stack([g_tree, g_ibeam, g_idm, g_itime]).get())
        snr = float(gmax.get())

        print(f'{grouper.grouper_ip_addr}: ichunk={ichunk}: max snr={snr:.3g} '
              f'at (tree={tree}, ibeam={ibeam}, idm={idm}, itime={itime})', flush=True)

        if sifter is not None:
            # Convert the chunk's peak (still GPU 0-d index arrays) to a one-event
            # FrbSifterEvents -- create_events does the index -> physical-unit math
            # and sets chunk_fpga_count from ichunk. coarsegrain_snr = the per-beam
            # max (cupy array; the client copies it host-side in one shot). FrbEvent
            # has no 'tree' field (printed only).
            events = grouper.create_events(ichunk, g_tree, g_ibeam, g_idm, g_itime, gmax)
            chunk_fpga = events.chunk_fpga_count
            sifter.send_events(
                has_injections = False,
                beam_set_id = beam_set_id,
                events = events,
                coarsegrain_start_fpga_count = chunk_fpga,
                coarsegrain_end_fpga_count = chunk_fpga + fpga_per_chunk,
                coarsegrain_snr = per_beam_max)

        if delay > 0:
            time.sleep(delay)


def run_toy_grouper(grouper_addr, sifter_addr=None, delay=0.0):
    """Run a toy FrbGrouper consumer at 'grouper_addr' (e.g. '127.0.0.1:7000').

    Acts as the downstream consumer of an FrbServer producer over CUDA IPC.
    Blocks (in FrbGrouper.open(), via __enter__) until the producer connects, then
    per chunk finds the peak SNR event (and the per-beam max) and prints it; if
    'sifter_addr' is given, also reports to that FrbSifter. Runs until the producer
    disconnects or Ctrl-C.

    'sifter_addr' is an 'ip:port' string, or None to run without a sifter.
    'delay' (seconds) inserts an artificial per-chunk slowdown into the loop;
    see _run_toy_grouper.
    """
    # Imported here (not at module top) so 'import pirate_frb' stays light.
    from .rpc import FrbGrouper, FrbSifterClient

    # Construct the sifter client (opens a gRPC channel; the RPCs themselves are
    # issued in _run_toy_grouper, which has the grouper metadata). It's used as a
    # context manager so it's closed on every exit path; with no sifter,
    # nullcontext() yields None (which _run_toy_grouper accepts).
    sifter_cm = FrbSifterClient(sifter_addr) if (sifter_addr is not None) else contextlib.nullcontext()

    # FrbGrouper.__enter__ blocks until the producer connects, then pins this thread
    # to the GPU's vcpus and selects the CUDA device (printing a message); __exit__
    # restores them and closes the grouper. The sifter (if any) is closed when its
    # 'with' exits (after the grouper's).
    with sifter_cm as sifter, FrbGrouper(grouper_addr) as grouper:
        try:
            _run_toy_grouper(grouper, sifter, delay)
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

def run_toy_groupers(grouper_addrs, sifter_addr=None, delay=0.0):
    """
    'sifter_addr' (an 'ip:port' string, or None for no sifter) and 'delay' (an
    artificial per-chunk slowdown, in seconds) are forwarded to each grouper.
    """
    # When fanning out to child subprocesses, re-pass exactly one of the
    # (mutually-exclusive, required) sifter flags.
    sifter_flag = ['--sifter', sifter_addr] if (sifter_addr is not None) else ['--no-sifter']
    run_groupers(run_toy_grouper, grouper_addrs, (), dict(sifter_addr=sifter_addr, delay=delay),
                 [sys.executable, '-m', 'pirate_frb', 'run_toy_grouper',
                  *sifter_flag, '--delay', str(delay)])

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
