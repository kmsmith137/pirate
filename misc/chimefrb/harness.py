"""Shared helpers for the chimefrb spot tests.

A spot test compares pirate against the old CHIME pipeline.  The old side is a
standalone C++ program with one job:

    driver <input.npy> <output.npy> [key=value ...]

A transform that operates on an (intensity, weights) PAIR stacks the two along a
leading length-2 axis, since only one array travels each way; see
misc/chimefrb/rfi_wi_downsample/driver.cpp.

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


def _driver_libs(src):
    """Libraries a driver asks for, via a '// LIBS: -lfoo -lbar' line near the top of it.

    driver_flags() carries search paths but no -l flags, because which libraries a driver
    needs is a property of the driver.  (The dispersion_delay example needs none: the
    bonsai function it calls is header-only.)  Keeping the list in driver.cpp keeps a spot
    check to two files.
    """

    with open(src) as f:
        for line in f:
            if line.startswith("// LIBS:"):
                return line.split(":", 1)[1].split()
            if not line.startswith("//") and line.strip():
                break   # past the header comment
    return []


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
           + ["-o", exe, src] + f["libflags"].split() + _driver_libs(src))

    p = subprocess.run(cmd, cwd=test_dir, capture_output=True, text=True)
    if p.returncode != 0:
        _die("failed to compile %s:\n%s" % (src, p.stderr.strip()))

    return exe


def run_driver(test_dir, in_array, params=None, dtype=None):
    """Write in_array, run the test's driver on it, and return what it wrote back.

    'params' is a dict of scalars, passed to the driver as key=value arguments.
    'dtype' force-casts the input array; the default (None) preserves what the caller gave.
    """

    exe = build_driver(test_dir)
    wd = workdir(test_dir)
    in_path = os.path.join(wd, "in.npy")
    out_path = os.path.join(wd, "out.npy")

    # dtype=None preserves the caller's dtype.  Passing one force-casts, which is
    # occasionally what you want and is otherwise a silent-upcast footgun: a caller
    # who passes float32 or uint8 and forgets to say so gets a comparison that
    # quietly measures something else.
    arr = (np.ascontiguousarray(in_array, dtype=dtype) if dtype is not None
           else np.ascontiguousarray(in_array))
    np.save(in_path, arr)
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

    def check_sandwich(self, label, got, lo, hi, why=None):
        """Check lo <= got <= hi elementwise, and say how far outside anything fell.

        A bracket rather than a tolerance. Several of the RFI transforms make a hard
        decision on a floating-point statistic -- clip or do not clip, variance valid or
        not -- and two implementations that sum in different orders will occasionally
        land on opposite sides of it. Running the reference twice with the threshold
        perturbed either way turns "the two disagree" into the reviewable statement
        "the fast code's answer lies between what the reference gives at thresholds
        either side of the real one", which is what a correct implementation must do and
        an incorrect one generally will not.

        This is the tool for that; check_allclose() cannot express it. Booleans work:
        pass 0/1 arrays to bracket a decision rather than a value.
        """

        got = np.asarray(got, dtype=np.float64)
        lo = np.asarray(lo, dtype=np.float64)
        hi = np.asarray(hi, dtype=np.float64)

        if not (got.shape == lo.shape == hi.shape):
            self.failures.append("%s: shape %s vs [%s, %s]"
                                 % (label, got.shape, lo.shape, hi.shape))
            print("  FAIL %s: shape %s vs [%s, %s]"
                  % (label, got.shape, lo.shape, hi.shape))
            return False

        excursion = np.maximum(lo - got, got - hi)
        worst = float(excursion.max()) if excursion.size else 0.0
        nbad = int(np.sum(excursion > 0.0))
        ok = (nbad == 0)

        print("  %s %-22s %d/%d outside bracket   (worst excursion %.3g)"
              % ("ok  " if ok else "FAIL", label, nbad, excursion.size, worst))
        if why:
            print("       %s" % why)

        if not ok:
            i = int(np.argmax(excursion))
            print("       worst at flat index %d: got %.17g, bracket [%.17g, %.17g]"
                  % (i, got.flat[i], lo.flat[i], hi.flat[i]))
            self.failures.append("%s: %d element(s) outside bracket, worst %.3g"
                                 % (label, nbad, worst))

        return ok

    def done(self):
        if self.failures:
            print("  %d check(s) FAILED" % len(self.failures))
            return 1
        print("  ok")
        return 0
