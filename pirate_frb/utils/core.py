"""Core utilities for pirate_frb (integer/bit helpers + subprocess-group orchestration)."""

import subprocess
import time


def integer_log2(n):
    """Return log2(n) as an int, where n must be a positive power of two.

    Raises ValueError otherwise. Python analog of C++ pirate::integer_log2().
    """
    n = int(n)
    if n <= 0 or (n & (n - 1)) != 0:
        raise ValueError(f"integer_log2: argument {n} is not a positive power of two")
    return n.bit_length() - 1


def run_processes(multi_args):
    """Run several commands as child processes in parallel, fail-fast.

    'multi_args' is a list of argv lists (each a list of strings, as passed to
    subprocess.Popen). Each command is launched as its own child process; this
    function then blocks until ANY child exits -- for any reason -- at which point
    it terminates the remaining children and returns. This makes the group
    fail-fast: one process going down brings the whole set down.

    Returns 0 if every child that had exited did so cleanly (exit code 0), else 1.
    In status messages, each child is labelled by its command string.
    """
    procs = []   # list of (label, Popen)
    rc = 0
    try:
        for argv in multi_args:
            # A fresh process (not fork) avoids CUDA-after-fork hazards. stdout is
            # inherited, so the children's messages appear interleaved on our stdout.
            procs.append((" ".join(argv), subprocess.Popen(argv)))
        rc = _monitor_children(procs)
    except KeyboardInterrupt:
        print("run_processes: interrupted; stopping all processes", flush=True)
    finally:
        _terminate_children(procs)
    return rc


def _monitor_children(procs):
    """Block until any child exits; return 0 if all dead children exited cleanly,
    else 1. (The caller then terminates the survivors.) 'procs' is a list of
    (label, Popen) pairs."""
    while True:
        dead = [(label, p) for label, p in procs if p.poll() is not None]
        if dead:
            for label, p in dead:
                print(f"run_processes: child [{label}] exited (code {p.returncode}); "
                      f"stopping the other processes", flush=True)
            return 0 if all(p.returncode == 0 for _, p in dead) else 1
        time.sleep(0.2)


def _terminate_children(procs, grace_sec=5.0):
    """SIGTERM all still-running children, then SIGKILL any that don't exit within
    grace_sec. 'procs' is a list of (label, Popen) pairs."""
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
