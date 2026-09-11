#!/usr/bin/env python3
"""Spot test: the intensity clipper, pirate vs rf_kernels.

Compares pirate_frb.chimefrb.ReferenceIntensityClipper against
rf_kernels::intensity_clipper, the principal RFI flagger in the old CHIME FRB search's
transform chain (48 of the production config's 120 nodes).

WHY THIS TEST EXISTS. The routine unit test ('pirate_frb test --cfrb') compares the CUDA
class GpuIntensityClipper against this reference, so it establishes that the two pirate
implementations agree -- but both were written from one reading of the old code. This is
the test that reads the old code by running it.

Most of the reference is already pinned down elsewhere: it is built out of
ReferenceWiDownsampler and ReferenceWrms, each with its own spot test. What is new, and
what this test is really about, is intensity_clip() -- the final clip and the upsample of
the cell mask back to full resolution.

THE PRIMARY CHECK TAKES THE OLD KERNEL'S OWN (mean, rms). That is not a shortcut; it is
what rf_kernels' own unit test does (test-intensity-clipper.cpp), and for a good reason.
The production clipper runs nine rounds of refinement, and after nine rounds a single
sample that lands on the other side of a threshold in float32 than in float64 has moved
everything downstream. Comparing two full refinement chains is a coin flip, not a test.
So the driver is run twice -- once for the clipped weights, once for the statistic -- and
the reference supplies only the final clip, bracketed at sigma*(1 +/- 1e-4). Both sides
then start from bit-identical statistics and there is nothing to amplify.

The secondary check runs the whole reference end to end, but only at niter=1, where there
are no refinements and nothing can amplify. That is what validates the downsample, the
statistic and the axis reshaping as a unit.

At niter=9 the end-to-end disagreement is REPORTED, not asserted. That number is the
amplification rate on this data, and it is worth knowing rather than designing around
blindly.

Needs a built oldpipe (misc/chimefrb/build_oldpipe.sh). Run me directly, or through
misc/chimefrb/run_spot_tests.py.
"""

import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import harness

from pirate_frb.chimefrb import (AXIS_FREQ, AXIS_TIME, AXIS_NONE,
                                 ReferenceIntensityClipper, ReferenceWiDownsampler,
                                 intensity_clip)

HERE = os.path.dirname(os.path.abspath(__file__))

SEED = 137

# Small on purpose: nothing is cached between runs. These also satisfy pirate's own
# divisibility rules (F % (32*Df) == 0, T % (32*Dt) == 0) at every config below, so the
# geometry used here is one GpuIntensityClipper could run too.
NFREQ = 128
NT = 512

# The four distinct intensity_clippers in the production config
# (misc/chimefrb/configs/21-03-07-low-latency-uniform-badchannel-mask-noplot.json), each
# of which appears there with both two_pass values. sigma is 5 throughout; note that
# iter_sigma is 3, not 5, for the (2,16) pair -- the two thresholds are different numbers.
#
# (axis, Df, Dt, sigma, iter_sigma) -- all with niter=9 in production.
CONFIGS = [
    (AXIS_FREQ, 1, 1,  5.0, 5.0),
    (AXIS_TIME, 1, 1,  5.0, 5.0),
    (AXIS_NONE, 2, 16, 5.0, 3.0),
    (AXIS_FREQ, 2, 16, 5.0, 3.0),
]

AXIS_NAMES = {AXIS_FREQ: "FREQ", AXIS_TIME: "TIME", AXIS_NONE: "NONE"}

# The bracket on the final clip, in relative units of sigma. This is not a fudge factor:
# it is the amount of roundoff we are prepared to tolerate in |I_ds - mean| vs sigma*rms,
# and it is the number the old code's own unit test chose for the same comparison.
SIGMA_BRACKET = 1.0e-4

# The bracket on the variance-validity cutoffs, as a multiplier on eps_2 and eps_3. A row
# whose variance is rejected loses ALL of its weights, so this decision has a large blast
# radius and cannot be left to a tolerance.
EPS_LO, EPS_HI = 0.5, 1.5


def make_input(rng):
    """A (2, F, T) float32 array: arr[0] = intensity, arr[1] = weights.

    Built to reach the corner cases rather than just the common one:

      - most channels are ordinary: Gaussian, with a large positive offset, since the
        eps_2*mean variance cutoff only means anything when the mean is far from zero;
      - some channels and some time samples carry 20-sigma outliers, so the clip actually
        fires (on clean Gaussian data a 5-sigma clip masks essentially nothing, and the
        test would be comparing two arrays of untouched weights);
      - a few channels are fully masked (no weight at all), a few are constant (variance
        exactly zero, so the whole row is clipped), and a few are nearly constant at a
        large mean, which is the case that lands on the epsilon cutoff.

    Deliberately NOT included: samples placed AT a threshold. Sampling the ambiguous
    region on purpose would make the brackets straddle at a controlled rate rather than a
    negligible one, which is worse, not better.
    """

    offset = 100.0
    scale = 10.0
    intensity = rng.normal(offset, scale, size=(NFREQ, NT))
    weights = (rng.uniform(size=(NFREQ, NT)) < 0.8) * rng.uniform(0.5, 1.5, size=(NFREQ, NT))

    # Whole bad channels and whole bad time samples: the two RFI shapes the AXIS_TIME and
    # AXIS_FREQ clippers respectively exist to catch.
    for f in rng.choice(NFREQ, size=4, replace=False):
        intensity[f, :] = offset + 20.0 * scale
    for t in rng.choice(NT, size=8, replace=False):
        intensity[:, t] = offset + 20.0 * scale

    # Isolated spikes, which AXIS_NONE catches and the other two mostly do not.
    for _ in range(20):
        f = rng.integers(0, NFREQ)
        t = rng.integers(0, NT)
        intensity[f, t] = offset + 25.0 * scale * rng.choice([-1.0, 1.0])

    intensity[8:12, :] = offset            # constant: variance exactly zero
    weights[12:16, :] = 0.0                # no weight at all
    intensity[16:20, :] = offset * (1.0 + 1.0e-6 * rng.normal(size=(4, NT)))  # on the cutoff

    return np.stack([intensity, weights]).astype(np.float32)


def run_old_weights(x, axis, Df, Dt, sigma, niter, iter_sigma, two_pass):
    """Clipped weights, shape (F, T), from rf_kernels::intensity_clipper."""

    return harness.run_driver(HERE, x, dtype=np.float32, params={
        "output": "weights", "axis": int(axis), "sigma": sigma, "Df": Df, "Dt": Dt,
        "niter": niter, "iter_sigma": iter_sigma, "two_pass": int(two_pass)})


def run_old_wrms(x, axis, Df, Dt, niter, iter_sigma, two_pass):
    """(mean, var) from rf_kernels::weighted_mean_rms. The driver reports rms, so square it."""

    out = harness.run_driver(HERE, x, dtype=np.float32, params={
        "output": "wrms", "axis": int(axis), "Df": Df, "Dt": Dt,
        "niter": niter, "iter_sigma": iter_sigma, "two_pass": int(two_pass)})

    return (out[0].astype(np.float64), out[1].astype(np.float64)**2)


def check_final_clip(t, label, x, axis, Df, Dt, sigma, niter, iter_sigma, two_pass):
    """Primary check: our final clip, fed the OLD kernel's statistic, vs the old clipper.

    Both sides see the same (mean, var), so the only thing under test is the clip itself
    and the upsample of the cell mask -- which is exactly the part of the reference that
    is new code rather than already-validated pieces.
    """

    w_old = run_old_weights(x, axis, Df, Dt, sigma, niter, iter_sigma, two_pass)
    (mean, var) = run_old_wrms(x, axis, Df, Dt, niter, iter_sigma, two_pass)

    I = x[0].astype(np.float64)[None, :, :]      # (1, F, T): the reference takes a beam axis
    W = x[1].astype(np.float64)[None, :, :]
    (i_ds, _) = ReferenceWiDownsampler(Df, Dt, transpose=False).apply(I, W)

    # Smaller sigma clips more, so it is the lower bound on the surviving weights.
    lo = intensity_clip(i_ds, W, mean, var, sigma * (1.0 - SIGMA_BRACKET), axis, Df, Dt)
    hi = intensity_clip(i_ds, W, mean, var, sigma * (1.0 + SIGMA_BRACKET), axis, Df, Dt)

    t.check_sandwich("%s final clip" % label, w_old[None, :, :], lo, hi,
                     why="our clip at sigma*(1 -/+ %g), given the OLD kernel's own"
                         " (mean, rms); weights are only ever zeroed, so this is an"
                         " elementwise bracket on a {0, w_in}-valued array" % SIGMA_BRACKET)

    nmasked = int(np.sum((w_old == 0) & (x[1] != 0)))
    t.note("        %d of %d weights newly zeroed by the old clipper"
           % (nmasked, w_old.size))
    if nmasked == 0:
        t.note("        WARNING: the clip fired on nothing, so this check is vacuous")


def check_end_to_end(t, label, x, axis, Df, Dt, sigma, niter, iter_sigma, two_pass,
                     assert_it):
    """Secondary check: the whole reference, end to end, against the old clipper.

    Bracketed on BOTH sharp decisions: sigma for the final clip, and eps_multiplier for
    the variance-validity cutoffs (a rejected variance zeroes the whole row, so the two
    effects compose in the same direction -- clip harder and reject more variances to get
    the lower bound).

    'assert_it' is False at niter > 1, where the two codes run nine independent
    refinement chains and are allowed to disagree; there the excursion is reported so
    that the amplification rate on this data is on the record.
    """

    w_old = run_old_weights(x, axis, Df, Dt, sigma, niter, iter_sigma, two_pass)

    I = x[0].astype(np.float64)[None, :, :]
    W = x[1].astype(np.float64)[None, :, :]

    lo = ReferenceIntensityClipper(axis, sigma * (1.0 - SIGMA_BRACKET), Df, Dt, niter,
                                   iter_sigma, two_pass,
                                   eps_multiplier=EPS_HI).apply(I, W)
    hi = ReferenceIntensityClipper(axis, sigma * (1.0 + SIGMA_BRACKET), Df, Dt, niter,
                                   iter_sigma, two_pass,
                                   eps_multiplier=EPS_LO).apply(I, W)

    if assert_it:
        t.check_sandwich("%s end-to-end" % label, w_old[None, :, :], lo, hi,
                         why="the whole reference at sigma*(1 -/+ %g) and eps_multiplier"
                             " %g/%g; at niter=1 there are no refinements, so nothing can"
                             " amplify" % (SIGMA_BRACKET, EPS_HI, EPS_LO))
        return

    excursion = np.maximum(lo[0] - w_old, w_old - hi[0])
    nbad = int(np.sum(excursion > 0.0))
    t.note("  info %-22s %d/%d outside bracket end-to-end (not asserted: %d independent"
           " refinement chains on each side)" % (label, nbad, w_old.size, niter))


def main():
    t = harness.Test("rfi_intensity_clipper")

    manifest = harness.build_manifest()
    if manifest:
        for line in manifest.splitlines():
            if line.startswith("date:") or line.startswith("arch:"):
                t.note(line.strip())

    x = make_input(np.random.default_rng(SEED))
    t.note("(2, %d, %d) intensity/weights from seed %d" % (NFREQ, NT, SEED))

    for (axis, Df, Dt, sigma, iter_sigma) in CONFIGS:
        for two_pass in (True, False):
            tag = "%s (%d,%d) tp=%d" % (AXIS_NAMES[axis], Df, Dt, int(two_pass))

            # niter=1: no refinements, so the whole reference can be compared directly.
            check_end_to_end(t, tag + " n1", x, axis, Df, Dt, sigma, 1, iter_sigma,
                             two_pass, assert_it=True)

            # niter=9: the production setting. The statistic comes from the old kernel.
            check_final_clip(t, tag + " n9", x, axis, Df, Dt, sigma, 9, iter_sigma, two_pass)
            check_end_to_end(t, tag + " n9", x, axis, Df, Dt, sigma, 9, iter_sigma,
                             two_pass, assert_it=False)

    return t.done()


if __name__ == "__main__":
    sys.exit(main())
