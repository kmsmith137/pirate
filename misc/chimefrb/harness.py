"""Shared helpers for the chimefrb spot tests.

A spot test compares pirate against the old CHIME pipeline.  The old side is a
standalone C++ program with one job:

    driver <input.npy> <output.npy> [key=value ...]

It links only old libraries, prints nothing on success, and signals failure by exit
status.  It does not know pirate exists, what it is being compared against, or what
counts as agreement.  Everything else -- generating the input, running pirate,
comparing, deciding the tolerance -- belongs to the test's own test.py, which uses
this module.

Nothing is committed to git except source.  Inputs are generated from a seed the test
records, the driver is compiled on demand, and its output is regenerated every run, so
a spot test always needs a built oldpipe (misc/chimefrb/build_oldpipe.sh).  Run
artifacts land under oldpipe/spot_tests/<name>/, which is gitignored and is discarded
whenever the old pipeline is rebuilt -- correct, since a driver links against it.
"""

import json
import os
import subprocess
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
OLDPIPE = os.path.join(HERE, "oldpipe")


def _die(msg):
    print("ERROR: " + msg, file=sys.stderr)
    sys.exit(1)


def driver_flags():
    """Compiler flags for building a driver against the old pipeline."""

    path = os.path.join(OLDPIPE, "driver_flags.json")
    if not os.path.exists(path):
        _die("no old-pipeline build found at %s.\n"
             "       Run misc/chimefrb/build_oldpipe.sh first (about a minute)." % OLDPIPE)

    with open(path) as f:
        return json.load(f)


def build_manifest():
    """One-line provenance for the old build, or None if it was not recorded."""

    path = os.path.join(OLDPIPE, "BUILD_MANIFEST.txt")
    if not os.path.exists(path):
        return None

    with open(path) as f:
        return f.read()


def workdir(test_dir):
    """Scratch directory for one test: compiled driver, input and output arrays."""

    d = os.path.join(OLDPIPE, "spot_tests", os.path.basename(os.path.abspath(test_dir)))
    os.makedirs(d, exist_ok=True)
    return d


def build_driver(test_dir):
    """Compile <test_dir>/driver.cpp, and return the path to the executable.

    Recompiles when driver.cpp or npy.hpp is newer than the binary.  Everything else
    the driver depends on lives in oldpipe, which takes the whole scratch directory
    with it when it is rebuilt.
    """

    test_dir = os.path.abspath(test_dir)
    src = os.path.join(test_dir, "driver.cpp")
    if not os.path.exists(src):
        _die("no driver.cpp in %s" % test_dir)

    exe = os.path.join(workdir(test_dir), "driver")
    deps = [src, os.path.join(HERE, "npy.hpp")]

    if os.path.exists(exe) and all(os.path.getmtime(d) <= os.path.getmtime(exe) for d in deps):
        return exe

    f = driver_flags()
    cmd = ([f["cxx"]] + f["cxxflags"].split() + f["incflags"].split()
           + ["-o", exe, src] + f["libflags"].split())

    p = subprocess.run(cmd, cwd=test_dir, capture_output=True, text=True)
    if p.returncode != 0:
        _die("failed to compile %s:\n%s" % (src, p.stderr.strip()))

    return exe


def run_driver(test_dir, in_array, params=None, dtype=np.float64):
    """Write in_array, run the test's driver on it, and return what it wrote back.

    'params' is a dict of scalars, passed to the driver as key=value arguments.
    """

    exe = build_driver(test_dir)
    wd = workdir(test_dir)
    in_path = os.path.join(wd, "in.npy")
    out_path = os.path.join(wd, "out.npy")

    np.save(in_path, np.ascontiguousarray(in_array, dtype=dtype))
    if os.path.exists(out_path):
        os.remove(out_path)

    cmd = [exe, in_path, out_path]
    for k, v in sorted((params or {}).items()):
        cmd.append("%s=%s" % (k, v))

    p = subprocess.run(cmd, capture_output=True, text=True)
    if p.returncode != 0:
        _die("driver failed (exit %d)\n  %s\n%s"
             % (p.returncode, " ".join(cmd), (p.stderr or p.stdout).strip()))
    if not os.path.exists(out_path):
        _die("driver exited 0 but wrote no output:\n  %s" % " ".join(cmd))

    return np.load(out_path)


class Test:
    """Accumulates the checks one spot test makes, and reports at the end.

    Usage:

        t = harness.Test("dispersion_delay")
        t.check_allclose("delay", got, want, rtol=1e-6, why="...")
        sys.exit(t.done())
    """

    def __init__(self, name):
        self.name = name
        self.failures = []
        print("[%s]" % name)

    def note(self, msg):
        print("  %s" % msg)

    def check_allclose(self, label, got, want, rtol, atol=0.0, why=None):
        """Compare two arrays, and say what the observed disagreement actually was.

        'why' should justify the tolerance.  A spot test that just picks a number and
        does not say why cannot be reviewed later, and the tolerances here are rarely
        arbitrary -- they usually encode a specific known difference between the two
        codes.
        """

        got = np.asarray(got, dtype=np.float64)
        want = np.asarray(want, dtype=np.float64)

        if got.shape != want.shape:
            self.failures.append("%s: shape %s vs %s" % (label, got.shape, want.shape))
            print("  FAIL %s: shape %s vs %s" % (label, got.shape, want.shape))
            return False

        denom = np.where(want != 0.0, np.abs(want), 1.0)
        rel = np.abs(got - want) / denom
        worst = float(rel.max()) if rel.size else 0.0

        ok = bool(np.allclose(got, want, rtol=rtol, atol=atol))
        print("  %s %-22s max rel diff %.3g   (tolerance %.3g)"
              % ("ok  " if ok else "FAIL", label, worst, rtol))
        if why:
            print("       %s" % why)

        if not ok:
            i = int(np.argmax(rel))
            print("       worst at flat index %d: got %.17g, want %.17g"
                  % (i, got.flat[i], want.flat[i]))
            self.failures.append("%s: max rel diff %.3g exceeds %.3g" % (label, worst, rtol))

        return ok

    def done(self):
        if self.failures:
            print("  %d check(s) FAILED" % len(self.failures))
            return 1
        print("  ok")
        return 0
