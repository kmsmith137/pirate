#!/usr/bin/env python3
"""Spot test: dispersion delay, pirate vs bonsai.

Both codes convert a dispersion measure and a radio frequency into a time delay:

    bonsai::dispersion_delay()      4.148806e3 * DM / freq_MHz^2          seconds
    pirate_frb.simpulse             1e-3 * k_dm * DM / freq_MHz^2         seconds
                                    with k_dm = 4.148808e6

They do not use the same dispersion constant, so they do not agree exactly.  That is
the point of choosing this as the first spot test: it is small enough to read in one
sitting, and it still forces the question every later test has to answer -- what
counts as agreement, and why.

This test is also the worked example for how to add another one.  A spot test is two
files in a directory of its own:

    driver.cpp   the OLD side.  Reads one .npy, writes one .npy, links only old
                 libraries, silent on success.  Knows nothing about pirate.
    test.py      this file.  Owns the input, the pirate side, the comparison, and
                 the tolerance.

Nothing binary is committed.  The input is drawn from the fixed seed below, and the
driver is compiled and run on demand, so the whole comparison regenerates from source
given a built oldpipe.

Run me directly, or through misc/chimefrb/run_spot_tests.py.
"""

import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import harness

from pirate_frb import simpulse

HERE = os.path.dirname(os.path.abspath(__file__))

# Any fixed value works; it is written down so the comparison is reproducible without
# committing the array it generates.
SEED = 137
NSAMP = 256

# The two dispersion constants differ by (4.148808 - 4.148806) / 4.148808 = 4.8e-7, and
# the delay is linear in the constant, so every sample should miss by almost exactly
# that.  1e-6 is that difference with a little room, and nothing more: a disagreement
# any larger than the known constant mismatch is a real result and should fail.
RTOL = 1.0e-6


def make_input(rng):
    """(N,2) array of (DM, freq_MHz) pairs.

    DMs span the range a CHIME/CHORD search cares about, and frequencies span the CHIME
    band.  Both are strictly positive: pirate asserts freq > 0 (it is a divisor) and
    DM >= 0, and a spot test should compare the codes where both are defined rather
    than probe each one's error handling.
    """

    dm = rng.uniform(1.0, 3000.0, size=NSAMP)
    freq_MHz = rng.uniform(400.0, 800.0, size=NSAMP)
    return np.column_stack([dm, freq_MHz])


def main():
    t = harness.Test("dispersion_delay")

    manifest = harness.build_manifest()
    if manifest:
        for line in manifest.splitlines():
            if line.startswith("date:") or line.startswith("arch:"):
                t.note(line.strip())

    x = make_input(np.random.default_rng(SEED))
    t.note("%d (DM, freq_MHz) pairs from seed %d" % (NSAMP, SEED))

    old = harness.run_driver(HERE, x)
    new = np.array([simpulse.dispersion_delay(dm, f) for dm, f in x])

    t.check_allclose("delay (seconds)", new, old, rtol=RTOL,
                     why="bonsai uses 4.148806e3, pirate uses k_dm=4.148808e6 "
                         "(constants.hpp:49): a 4.8e-7 relative offset by construction")

    return t.done()


if __name__ == "__main__":
    sys.exit(main())
