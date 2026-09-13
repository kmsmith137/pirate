#!/usr/bin/env python3
"""Spot test: peak-finding profiles, pirate vs bonsai.

Both codes finish dedispersion by convolving each dedispersed time series with a small bank
of profiles -- short, non-negative, symmetric kernels that approximate matched filters for a
range of pulse widths -- and then maximizing. This test asks whether the profiles themselves
are the same.

WHAT IS COMPARED, AND WHAT IS NOT. Only the finest scale is common ground. bonsai's "ab4"
peak finder has exactly four profiles per tree, and covers wider pulses with a ladder of
time-downsampled trees; pirate instead repeats three shapes at every octave inside one tree,
so its bank has 3*log2(max_kernel_width)+1 profiles. At max_kernel_width=2 pirate's bank is
four profiles built from single samples, which is precisely bonsai's set. That is the
configuration used here. Pirate's higher levels are the same three shapes with every tap
replaced by a block of 2^lambda equal samples, which is checked in-tree rather than here
('pirate_frb test --pfsq' and '--varmap').

The old side is bonsai::reference_peak_finder(), which returns both of the things its
production kernel computes: 'tm', a weighted max over profiles, and 'tv', a per-profile mean
square. Both are used, for two independent comparisons:

  1. SHAPES, from 'tm'. With one unit impulse in the input, weights one-hot on a single
     profile, and no time coarse-graining, tm[i] is that profile's coefficients, read off
     sample by sample. This is compared against pirate's bank, and also against
     peak_finder_params::pf_coeffs -- a second copy of the coefficients, written down
     separately in the old code for its pulse-simulation path.

  2. THE QUADRATIC FORM, from 'tv'. Sum_i (h_p * x)[i]^2 for a random x, against pirate's
     ReferencePfSquare (the streaming float32 convolver that the GPU kernel mirrors) and
     against PfVarianceConvolver (the analytic autocorrelation table that the variance map
     is built from). This reaches pirate's implementations, not just its kernel bank.

NORMALIZATION. Both codes defer normalization to the weights, so a profile is only defined
up to overall scale: bonsai's are [1], [1,1], [1,3,1], [1,2,2,1] and pirate's are the same
divided by their largest entry. The scale is measured from bonsai's own output rather than
hardcoded here, so constants::pf3_a and pf4_b appear nowhere in this file.

ALIGNMENT. bonsai anchors its convolution at the kernel's leading edge and pirate at its
trailing edge, and the offset between them differs per profile, so the two codes' evaluation
windows cannot be aligned for all four profiles at once. Zero-padding the input removes the
problem: every sum below then runs over the full convolution, whatever the window.

Needs a built oldpipe (misc/chimefrb/build_oldpipe.sh). No GPU: everything on the pirate
side is host code. Run me directly, or through run_spot_tests.py.
"""

import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import harness

from pirate_frb.varmap import PfVarianceConvolver
from pirate_frb.kernels import ReferencePfSquare

HERE = os.path.dirname(os.path.abspath(__file__))

# Any fixed value works; it is written down so the comparison is reproducible without
# committing the array it generates.
SEED = 137

# bonsai's "ab4" peak finder: P profiles, and npad samples read past the end of the range
# the caller asks for. Both are properties of the old code (peak_finder_params), restated
# here because the test shapes its arrays before it can call the driver. The driver
# validates them.
PF_NAME = "ab4"
P = 4
NPAD = 3

# The pirate bank with one profile per bonsai profile. See "WHAT IS COMPARED" above.
MAX_KERNEL_WIDTH = 2

# Part 2's time series: a short random signal inside a generous zero pad. PADL exceeds
# ReferencePfSquare's warm-up (tpad = max(2*max_kernel_width, 32) = 32) and PADR exceeds the
# longest profile, so both codes see the whole convolution. 200 float32 samples.
NSIG, PADL, PADR = 128, 64, 8

# Part 2 is a float32-accumulation comparison, nothing more. bonsai sums tv in float32 over
# N ~ 200 terms (~sqrt(N) * 6e-8 ~ 8e-7 relative) and forms each pf_p in float32 (a few more
# eps, doubled by the square); pirate's ReferencePfSquare convolves in float32 but
# accumulates in float64, and PfVarianceConvolver is float64 throughout. A few times 1e-6 is
# the expectation; 1e-5 is ~10x above what is measured (see MEASURED below).
RTOL_SSQ = 1.0e-5

# MEASURED, at pf_xi3 = 1/3 and pf_xi4 = 1/2: part 1 agrees exactly (all three sources
# bit-identical), and part 2's worst relative disagreement is 1.9e-7 against
# PfVarianceConvolver and 2.1e-7 against ReferencePfSquare.
#
# NEGATIVE CONTROL: substituting the pf_xi3 = 1/2 bank that pirate used before it adopted
# bonsai's value, and holding this same bonsai output fixed, part 1 reports a relative
# difference of 0.33 on profile 2 (pirate 0.5 against bonsai's 0.33333334) and part 2 reports
# 0.178 on the same profile -- 1.8e4 times RTOL_SSQ. The other three profiles do not move
# (2e-7), since pf_xi4 and the two boxcars are untouched. So neither half of this test is
# passing by being loose. (Measured by swapping the bank on the pirate side, not by
# rebuilding pirate; the streaming and analytic paths agree to 2e-7, so the figure applies
# to both.)


def run(data, wt, N):
    """One driver call. Returns (tm, tv, pf_coeffs), with tv flattened over its U axis."""

    (tm, tv, coeffs) = harness.run_driver(
        HERE, [data, wt], dtype=np.float32, noutputs=3,
        params={"pf_name": PF_NAME, "M": 1, "N": N, "U": 1})

    return (tm, tv[:, 0], coeffs)


def bonsai_profiles():
    """bonsai's profiles, read out of the kernel its dedisperser applies.

    Returns a (P, NPAD+1) float32 array, each row zero-padded to NPAD+1 like pf_coeffs.

    One unit impulse at index NPAD, weights one-hot on profile p, no time coarse-graining:
    then pf_p[i] = c[p, NPAD-i], so tm[0:NPAD+1] reversed is the profile. The impulse is
    what makes this work -- the driver zero-initializes tm and the old code max()es into
    it, so a profile with a negative sample would read back clipped.
    """

    N = 2 * (NPAD + 1)          # anything past NPAD is zero, and confirms the tail is
    rows = []                   # really zero rather than off the end of the array

    for p in range(P):
        data = np.zeros(N + NPAD, dtype=np.float32)
        data[NPAD] = 1.0
        wt = np.zeros(P, dtype=np.float32)
        wt[p] = 1.0

        (tm, _, coeffs) = run(data, wt, N)

        if np.any(tm[NPAD+1:] != 0.0):
            raise RuntimeError("peak_finding: impulse response extends past NPAD, which "
                               "contradicts peak_finder_params::npad = %d" % NPAD)
        rows.append(tm[:NPAD+1][::-1])

    return (np.array(rows, dtype=np.float32), coeffs)


def check_shapes(t):
    """Part 1: bonsai's profile coefficients against pirate's bank, exactly.

    Returns bonsai's profiles, which part 2 needs for their normalization."""

    (applied, declared) = bonsai_profiles()
    pirate = PfVarianceConvolver.peak_finding_kernels(MAX_KERNEL_WIDTH)

    if len(pirate) != P:
        raise RuntimeError("peak_finding: pirate has %d profiles at max_kernel_width=%d, "
                           "bonsai's %s has %d" % (len(pirate), MAX_KERNEL_WIDTH, PF_NAME, P))

    # bonsai's two copies of its own coefficients: the ones reference_peak_finder() applies,
    # and the pf_coeffs table its pulse simulation uses. Equal, or the old code disagrees
    # with itself and neither is "bonsai's profile".
    t.check_allclose("bonsai applied vs declared", applied, declared, rtol=0.0, atol=0.0,
                     why="bonsai writes its coefficients down twice (reference_peak_finder"
                         " and peak_finder_params::pf_coeffs); they must agree exactly")

    # Rescale each bonsai profile to a peak of 1, which is pirate's convention. Done in
    # float32, so 3.0f -> 1.0f/3.0f rounds exactly as constants::pf_xi3 does, and the
    # comparison below is exact rather than approximate.
    #
    # Pirate's profile is zero-padded out to bonsai's row length rather than bonsai's row
    # being truncated to pirate's: a bonsai profile with one more tap than pirate's has
    # would otherwise be compared only where the two happen to overlap, and pass.
    for p in range(P):
        h = np.zeros(NPAD + 1)
        h[:len(pirate[p])] = pirate[p]
        got = applied[p] / applied[p].max()

        t.check_allclose("profile %d shape" % p, got, h, rtol=0.0, atol=0.0,
                         why="pirate's profile is bonsai's divided by its largest"
                             " coefficient; equality is exact because both sides round"
                             " the same float32 quotient")

    t.note("bonsai profiles (peak-normalized): %s"
           % [[float(v) for v in applied[p] / applied[p].max()] for p in range(P)])

    return applied


def check_quadratic_form(t, rng, applied):
    """Part 2: sum_i (h_p * x)[i]^2, bonsai against two pirate implementations."""

    sig = rng.standard_normal(NSIG).astype(np.float32)
    z = np.zeros(PADL + NSIG + PADR, dtype=np.float32)
    z[PADL:PADL+NSIG] = sig

    N = len(z) - NPAD
    (_, tv, _) = run(z, np.ones(P, dtype=np.float32), N)

    # tv[p] = (1/N) sum_i pf_p[i]^2, and pf_p is pirate's h_p times the profile's largest
    # coefficient (part 1), so undo both to get sum_i (h_p * sig)[i]^2.
    scale = applied.max(axis=1).astype(np.float64)
    old = tv.astype(np.float64) * N / scale**2

    # Pirate's analytic path: the autocorrelation table the variance map is built from.
    pfv = PfVarianceConvolver()
    new_analytic = pfv.variance(sig.astype(np.float64), P)

    # Pirate's streaming path: the float32 convolver that GpuPfSquare mirrors. One chunk,
    # zero history, so the zero pad is genuinely zero on both sides.
    ker = ReferencePfSquare(max_kernel_width=MAX_KERNEL_WIDTH, total_beams=1,
                            beams_per_batch=1, ndm=1, nt_in=len(z))
    acc = np.zeros((1, 1, P), dtype=np.float64)
    ker.apply(acc, z.reshape(1, 1, len(z)), 0)
    new_streaming = acc[0, 0]

    t.note("sum_i (h_p * x)[i]^2: bonsai %s" % np.array2string(old, precision=6))

    t.check_allclose("ssq vs PfVarianceConvolver", old, new_analytic, rtol=RTOL_SSQ,
                     why="bonsai accumulates tv in float32 over %d terms; pirate's analytic"
                         " path is float64 throughout (see RTOL_SSQ)" % N)

    t.check_allclose("ssq vs ReferencePfSquare", old, new_streaming, rtol=RTOL_SSQ,
                     why="both convolve in float32; bonsai also accumulates in float32,"
                         " pirate in float64 (see RTOL_SSQ)")


def main():
    t = harness.Test("peak_finding")

    manifest = harness.build_manifest()
    if manifest:
        for line in manifest.splitlines():
            if line.startswith("date:") or line.startswith("arch:"):
                t.note(line.strip())

    rng = np.random.default_rng(SEED)
    t.note("peak finder %s, max_kernel_width %d, %d signal samples from seed %d"
           % (PF_NAME, MAX_KERNEL_WIDTH, NSIG, SEED))

    applied = check_shapes(t)
    check_quadratic_form(t, rng, applied)

    return t.done()


if __name__ == "__main__":
    sys.exit(main())
