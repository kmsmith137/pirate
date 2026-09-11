#!/usr/bin/env python3
"""Spot test: the (Df,Dt) downsampler, pirate vs rf_kernels.

Compares pirate_frb.chimefrb.ReferenceWiDownsampler against
rf_kernels::wi_downsampler, the kernel it was transcribed from.

WHY THIS TEST EXISTS. The routine unit test ('pirate_frb test --cfrb') compares the
CUDA kernel GpuWiDownsampler against ReferenceWiDownsampler, so it establishes that
the two pirate implementations agree -- but both were written from one reading of the
old code, and a misreading would pass it. This is the test that reads the old code by
running it.

The trap it is really here to catch is the weight normalization: rf_kernels sums the
cell's weights, while the python helper rf_pipelines.utils.wi_downsample() averages
them, so the two conventions differ by a factor (Df*Dt). Picking the wrong one would
leave every clipper downstream weighting its statistics wrong, and nothing in the
pirate-only test suite would notice. Df=16 is in the list below because at (Df,Dt) =
(1,1) sum and mean coincide, so that configuration cannot catch it.

ONE KNOWN DIVERGENCE, in a don't-care slot. Where a cell's weights sum to zero, the
downsampled intensity is undefined -- every consumer multiplies it by that zero weight
-- and the two codes fill it differently. rf_kernels' general path writes 0 (its
guarded divide yields 0/1), but it short-circuits (Df,Dt)=(1,1) to a plain memcpy
(downsample_internals.hpp, "Special case (Df,Dt)=(1,1)"), which passes the intensity
through untouched. pirate writes 0 in both cases: one rule instead of two, and a
consumer that forgets to check the weight sees zeros rather than stale intensity. So
the intensity comparison below runs where out_w > 0, which is where it is defined, and
the masked cells are checked separately against pirate's stronger guarantee. Where a
(1,1) cell has weight, both codes copy the intensity through exactly -- rf_kernels by
its memcpy, pirate by design -- so there the comparison is exact.

NOT COVERED HERE: 'transpose', which the old kernel does not have. The transposed path
is pinned instead by a structural check in pirate_frb/chimefrb/test_wi_downsampler.py
-- the two orientations must be bitwise swapaxes of each other. So the division of
labour is: this test fixes the semantics, that one fixes the layout.

Needs a built oldpipe (misc/chimefrb/build_oldpipe.sh). Run me directly, or through
misc/chimefrb/run_spot_tests.py.
"""

import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import harness

from pirate_frb.chimefrb import ReferenceWiDownsampler

HERE = os.path.dirname(os.path.abspath(__file__))

# Any fixed value works; it is written down so the comparison is reproducible without
# committing the array it generates.
SEED = 137

# Small on purpose: nothing is cached between runs, and a bigger array buys no extra
# coverage here. F and T are divisible by every Df and Dt below, and every T//Dt is a
# multiple of 8, which rf_kernels requires.
NFREQ = 256
NT = 256

# The (Df, Dt) pairs the old search's production RFI config uses. (2,16) is the one
# with a nontrivial time reduction; (16,1) is the one that pins the weight
# normalization; (1,1) is the degenerate case, cheap to include.
CONFIGS = [(16, 1), (2, 16), (1, 1)]

# The two implementations sum a (Df*Dt)-element block in different orders, and in
# different precisions (rf_kernels in float32 with AVX2, the reference in float64), so
# exact agreement is the wrong expectation. Float32 roundoff accumulated over at most
# 32 terms is ~sqrt(32)*1.2e-7 = 7e-7, so 1e-5 has a comfortable margin over that and
# is still far tighter than any semantic error could hide in.
RTOL = 1.0e-5


def make_input(rng):
    """A (2, F, T) float32 array: arr[0] = intensity, arr[1] = weights.

    Two deliberate choices. The weights are drawn so that some (Df,Dt) cells come out
    FULLY masked: 'out_w <= 0' is the only branch in the kernel, and uniform random
    weights would never reach it. And the intensity is positive and O(100), like real
    CHIME intensity data, so that the weighted mean stays away from zero -- a mean
    near zero would make the relative comparison below meaningless. Cancellation in
    the mean is exercised by the unit test instead, which can afford a signed
    tolerance.
    """

    intensity = rng.normal(100.0, 10.0, size=(NFREQ, NT))

    # About 30% of samples masked at random, plus 16 whole channels and 16 whole time
    # blocks knocked out, which is what guarantees some fully-masked cells at every
    # (Df, Dt) in CONFIGS.
    weights = (rng.uniform(size=(NFREQ, NT)) < 0.7).astype(np.float64)
    weights *= rng.uniform(0.5, 1.5, size=(NFREQ, NT))
    weights[rng.choice(NFREQ, size=16, replace=False), :] = 0.0
    weights[:, rng.choice(NT, size=16, replace=False)] = 0.0

    return np.stack([intensity, weights]).astype(np.float32)


def main():
    t = harness.Test("rfi_wi_downsample")

    manifest = harness.build_manifest()
    if manifest:
        for line in manifest.splitlines():
            if line.startswith("date:") or line.startswith("arch:"):
                t.note(line.strip())

    x = make_input(np.random.default_rng(SEED))
    t.note("(2, %d, %d) intensity/weights from seed %d" % (NFREQ, NT, SEED))

    for (Df, Dt) in CONFIGS:
        old = harness.run_driver(HERE, x, params={"Df": Df, "Dt": Dt}, dtype=np.float32)

        ref = ReferenceWiDownsampler(Df, Dt, transpose=False)
        (new_i, new_w) = ref.apply(x[0][None, :, :], x[1][None, :, :])

        # The intensity is only defined where the cell has weight; see the docstring.
        ok = (old[1] > 0.0)
        nmasked = int(np.sum(~ok))
        t.note("Df=%d Dt=%d: %d of %d output cells fully masked"
               % (Df, Dt, nmasked, old[1].size))

        t.check_allclose("out_i (Df=%d, Dt=%d)" % (Df, Dt), new_i[0][ok], old[0][ok], rtol=RTOL,
                         why="weighted mean over each cell, compared where out_w > 0; both"
                             " sum the same terms in different orders and precisions")

        if (Df, Dt) == (1, 1):
            t.check_allclose("out_i exact (1,1)", new_i[0][ok], old[0][ok], rtol=0.0,
                             why="a (1,1) cell with weight is copied through bit for bit by"
                                 " both codes; the clippers' AXIS_FREQ path relies on it")

        # pirate zeroes the masked cells; the old code does so too except at (1,1),
        # where its memcpy shortcut leaves the raw intensity there. Checking pirate's
        # side keeps the stronger guarantee honest, and the note records what the old
        # code actually did, so a reader does not have to rediscover the shortcut.
        if nmasked > 0:
            stale = int(np.sum(old[0][~ok] != 0.0))
            t.note("        masked cells: pirate writes 0; rf_kernels left %d of %d nonzero"
                   % (stale, nmasked))
            assert np.all(new_i[0][~ok] == 0.0), "pirate left a masked cell nonzero"

        t.check_allclose("out_w (Df=%d, Dt=%d)" % (Df, Dt), new_w[0], old[1], rtol=RTOL,
                         why="SUM of the cell's weights, not the mean -- the two"
                             " conventions differ by Df*Dt = %d" % (Df*Dt))

    return t.done()


if __name__ == "__main__":
    sys.exit(main())
