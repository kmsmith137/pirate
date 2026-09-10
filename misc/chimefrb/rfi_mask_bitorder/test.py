#!/usr/bin/env python3
"""Spot test: the RFI mask's bit-packing convention, pirate vs rf_kernels.

pirate reads the rfi_mask in a chimefrb data file as bit-packed LSB-first -- bit i of
byte j is time sample 8*j+i -- with a SET bit meaning GOOD data.  That reading comes from
rf_kernels::mask_counter_data::slow_reference_mask_count(), which is the code that packed
every mask in every real file:

    for (int ibit = 0; ibit < 8; ibit++)
        if (in[ifreq*istride + 8*ibyte + ibit] > 0.0f)
            byte |= (1 << ibit);

It cannot be confirmed from the data files themselves.  Both bit orders produce an
identical transition-rate spread there, because the masks carry their own 8-sample-periodic
structure from the RFI chain's time downsampling.  So this test runs the packer itself.

The input deliberately uses RUNS of good and bad samples rather than uniform noise: against
noise, an off-by-one in the byte or bit index is invisible; against runs it is obvious.

Run me directly, or through misc/chimefrb/run_spot_tests.py.
"""

import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import harness

HERE = os.path.dirname(os.path.abspath(__file__))

SEED = 137
NFREQ = 16
NT = 1024


def make_input(rng):
    """(NFREQ, NT) float32 weights, in runs, with both all-good and all-bad rows."""

    w = np.zeros((NFREQ, NT), dtype=np.float32)
    for i in range(NFREQ):
        t, val = 0, int(rng.integers(2))
        while t < NT:
            n = int(rng.integers(1, 40))
            w[i, t:t+n] = float(val)
            t += n
            val ^= 1
    w[0, :] = 1.0      # a fully unmasked row
    w[1, :] = 0.0      # a fully masked row
    return w


def main():
    t = harness.Test("rfi_mask_bitorder")

    manifest = harness.build_manifest()
    if manifest:
        for line in manifest.splitlines():
            if line.startswith("date:") or line.startswith("arch:"):
                t.note(line.strip())

    w = make_input(np.random.default_rng(SEED))
    t.note("%d x %d float32 weights in runs, from seed %d" % (NFREQ, NT, SEED))

    # dtype=np.float32 is not optional: run_driver() would otherwise hand the driver
    # whatever the caller happened to build, and npy::read<float> would reject it.
    old = harness.run_driver(HERE, w, dtype=np.float32)

    # pirate's reading of the same bytes.
    new = np.packbits((w > 0).astype(np.uint8), axis=1, bitorder='little')

    t.check_allclose("packed mask bytes", new.astype(np.float64), old.astype(np.float64),
                     rtol=0.0, atol=0.0,
                     why="a bit layout is a convention, not a computation: any disagreement "
                         "at all means the LSB-first / set-means-good reading is wrong")

    return t.done()


if __name__ == "__main__":
    sys.exit(main())
