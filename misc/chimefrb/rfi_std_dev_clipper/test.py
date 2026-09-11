#!/usr/bin/env python3
"""Spot test: the std_dev_clipper, pirate vs rf_kernels.

Compares pirate_frb.chimefrb.ReferenceStdDevClipper against rf_kernels::std_dev_clipper, the
most numerous transform in the old CHIME FRB search's RFI chain (60 of its 120 nodes).

WHY THIS TEST MATTERS MORE THAN MOST. Stage 2 of this transform, std_dev_clipper::_clip_1d(),
has no reference implementation anywhere in the old code: its own comment says it was never
unit-tested, and the old test of the transform is circular (it calls the real _clip_1d). So
our clip_1d() is the first independent check it has ever had, and a disagreement found here
could be a bug in the OLD code. That is why the first thing this test does is drive the real
_clip_1d() directly (the driver's output=clip1d mode), on vectors we construct -- including
the degenerate ones random data never reaches.

The full-transform checks follow the intensity clipper's: the primary one feeds the OLD
kernel's own stage-1 variances into our stage 2 and apply, so only stage 2 and the apply are
under test; the secondary one runs our whole reference, conditioned on the old kernel's
stage-1 decisions. Conditioning rather than bracketing, because a stage-1 decision changes the
population stage 2 sees, so bracketing it does not bracket stage 2 (plan 9.2b).

Needs a built oldpipe (misc/chimefrb/build_oldpipe.sh). Run me directly, or through
misc/chimefrb/run_spot_tests.py.
"""

import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import harness

from pirate_frb.chimefrb import (AXIS_FREQ, AXIS_TIME, ReferenceStdDevClipper, clip_1d,
                                 std_dev_apply)
from pirate_frb.chimefrb.test_std_dev_clipper import (stage2_bracket, end_to_end_bracket,
                                                      ill_conditioned, keep_bracket)

HERE = os.path.dirname(os.path.abspath(__file__))
SEED = 137

# Small on purpose, and legal for both codes: rf_kernels needs nfreq % 8 == 0 and
# nt_chunk % 8 == 0, pirate needs F and T divisible by 32.
NFREQ = 128
NT = 512
SIGMA = 3.0          # the production value
AXIS_NAMES = {AXIS_FREQ: "FREQ", AXIS_TIME: "TIME"}


# -------------------------------------------------------------------------------------------------
#
# _clip_1d in isolation


def clip1d_cases(rng):
    """(label, V) pairs, V a (K, n) float32 batch of variance vectors, n a multiple of 8."""

    def gaussian(K, n, frac_invalid, noutliers):
        V = np.abs(1.0 + 0.03 * rng.normal(size=(K, n)))
        V[rng.uniform(size=(K, n)) < frac_invalid] = 0.0
        for k in range(K):
            cols = rng.choice(n, size=noutliers, replace=False)
            V[k, cols] = rng.choice([0.5, 1.5], size=noutliers)
        return V

    def nvalid(K, n, m):
        V = np.zeros((K, n))
        for k in range(K):
            cols = rng.choice(n, size=m, replace=False)
            V[k, cols] = rng.uniform(0.5, 2.0, size=m)
        return V

    cases = [
        ("gaussian n=1024", gaussian(16, 1024, 0.05, 3)),
        ("gaussian n=4096", gaussian(4, 4096, 0.05, 6)),
        ("mostly invalid n=256", gaussian(8, 256, 0.70, 2)),
        ("n_valid=0", np.zeros((2, 64))),
        ("n_valid=1", nvalid(4, 64, 1)),
        ("n_valid=2", nvalid(4, 64, 2)),
        ("n_valid=3", nvalid(4, 64, 3)),
        ("small n=8", np.abs(1.0 + 0.3 * rng.normal(size=(16, 8)))),
        ("wide range n=512", 10.0 ** rng.uniform(-3.0, 3.0, size=(8, 512))),
    ]
    return [(label, V.astype(np.float32)) for (label, V) in cases]


def check_clip1d(t, label, V, sigma):
    """The real _clip_1d against our clip_1d(), on the same variances."""

    old = harness.run_driver(HERE, V, dtype=np.float32,
                             params={"output": "clip1d", "sigma": sigma})

    lo = np.zeros(V.shape, dtype=bool)
    hi = np.zeros(V.shape, dtype=bool)
    for k in range(V.shape[0]):
        assert not ill_conditioned(V[k:k+1])[0], "test input must not be ill-conditioned"
        d = stage2_bracket(V[k:k+1], sigma)
        (lo[k], hi[k]) = keep_bracket(V[k], sigma, d)

    kept = (old != 0)
    ok = t.check_sandwich("clip1d %s s=%g" % (label, sigma), kept, lo, hi,
                          why="which variances survive; our float64 sums vs the old float32"
                              " sequential ones, bracketed on sigma per stage2_bracket()")
    ok &= t.check_allclose("clip1d %s s=%g kept" % (label, sigma), old[kept], V[kept],
                           rtol=0.0, why="a surviving variance is passed through untouched")
    t.note("        %d of %d entries clipped (of %d valid)"
           % (int(np.sum((V > 0) & ~kept)), V.size, int(np.sum(V > 0))))
    return ok


def report_all_equal(t, rng):
    """The ill-conditioned corner: every valid variance equal. REPORTED, NOT ASSERTED.

    Exact arithmetic clips every row (s = 0). The old float32 loop usually does not come back
    to exactly x for its mean, and then clips nothing -- how often depends on n and x. Our
    float64 clip_1d() sums exactly here and clips everything. Both are 'right'; see
    plans/chimefrb_std_dev_clipper.md 2.4.
    """

    t.note("  info all-equal valid variances (ill-conditioned, not asserted), sigma=%g:" % SIGMA)
    for m in (2, 3, 64, 1024):
        K = 100
        V = np.zeros((K, 1024), dtype=np.float32)
        for k in range(K):
            V[k, :m] = np.float32(rng.uniform(1.0, 1000.0))
        old = harness.run_driver(HERE, V, dtype=np.float32,
                                 params={"output": "clip1d", "sigma": SIGMA})
        ours = clip_1d(V.astype(np.float64), SIGMA)
        t.note("         n_valid=%4d: old clips everything in %3d of %d, ours in %3d of %d"
               % (m, int(np.sum(np.all(old == 0, axis=1))), K,
                  int(np.sum(np.all(ours == 0, axis=1))), K))


# -------------------------------------------------------------------------------------------------
#
# The whole transform


def make_input(rng):
    """A (2, F, T) float32 array: arr[0] = intensity, arr[1] = weights.

    Designed around stage 2, which sees VARIANCES: the injected RFI is rows with the wrong
    noise level -- a few channels and a few time samples with 5x the noise -- rather than
    bright samples, which a clean variance estimate barely notices. Kept to a few percent,
    since outliers inflate the very spread they are measured against. Plus the three ways to
    make a variance invalid (constant, fully masked, nearly constant at a large mean), in
    both directions, so each axis sees some.
    """

    offset, scale = 100.0, 10.0
    intensity = rng.normal(offset, scale, size=(NFREQ, NT))
    weights = (rng.uniform(size=(NFREQ, NT)) < 0.8) * rng.uniform(0.5, 1.5, size=(NFREQ, NT))

    for f in rng.choice(NFREQ, size=4, replace=False):
        intensity[f, :] = rng.normal(offset, 5.0 * scale, size=NT)
    for s in rng.choice(NT, size=8, replace=False):
        intensity[:, s] = rng.normal(offset, 5.0 * scale, size=NFREQ)

    intensity[20:22, :] = offset                                            # constant
    weights[22:24, :] = 0.0                                                 # masked
    intensity[24:26, :] = offset * (1.0 + 1.0e-6 * rng.normal(size=(2, NT)))  # on the cutoff
    intensity[:, 100:102] = offset
    weights[:, 102:104] = 0.0

    return np.stack([intensity, weights]).astype(np.float32)


def make_one_valid(rng, axis):
    """An input in which exactly ONE row has any weight, so the whole-beam branch runs."""

    x = make_input(rng)
    x[1] = 0.0
    if axis == AXIS_TIME:
        x[1, 40, :] = 1.0
    else:
        x[1, :, 300] = 1.0
    return x


def run_old(x, output, axis, two_pass):
    return harness.run_driver(HERE, x, dtype=np.float32, params={
        "output": output, "axis": axis, "sigma": SIGMA, "Df": 1, "Dt": 1,
        "two_pass": int(two_pass)})


def check_transform(t, label, x, axis, two_pass):
    w_old = run_old(x, "weights", axis, two_pass)
    vclip_old = run_old(x, "clipped_var", axis, two_pass).astype(np.float64)
    wrms = run_old(x, "wrms", axis, two_pass)
    v_old = wrms[1].astype(np.float64) ** 2         # the driver reports rms

    W = x[1].astype(np.float64)[None]               # (1, F, T): the reference has a beam axis
    I = x[0].astype(np.float64)[None]
    v1 = v_old[None]                                # (1, nrows)

    nkill = int(np.sum((v_old > 0) & (vclip_old == 0)))
    t.note("    %s: old stage 1 valid in %d of %d rows; old stage 2 clips %d of them"
           % (label, int(np.sum(v_old > 0)), v_old.size, nkill))

    if ill_conditioned(v1)[0]:
        t.note("        ill-conditioned (all valid variances equal); skipped")
        return

    # Primary: the OLD stage-1 variances, through our stage 2 and apply. rms^2 is within an
    # ulp or two of the variance the clipper used, and exactly zero where that is.
    d = stage2_bracket(v1, SIGMA)
    (klo, khi) = keep_bracket(v1, SIGMA, d)
    t.check_sandwich("%s stage 2 rows" % label, (vclip_old[None] != 0), klo, khi,
                     why="rows surviving stage 2, given the old stage-1 variances;"
                         " sigma bracket %.2g" % d)
    wlo = std_dev_apply(W, klo.astype(np.float64), axis, 1, 1)
    whi = std_dev_apply(W, khi.astype(np.float64), axis, 1, 1)
    t.check_sandwich("%s weights" % label, w_old[None], wlo, whi,
                     why="the old clipper's weights, against our stage 2 + apply on its"
                         " stage-1 variances")

    # Secondary: our whole reference, conditioned on the old stage-1 decisions.
    (_, v_c) = ReferenceStdDevClipper(axis, SIGMA, 1, 1, two_pass, eps_multiplier=1.5).variances(I, W)
    (m_p, v_p) = ReferenceStdDevClipper(axis, SIGMA, 1, 1, two_pass, eps_multiplier=0.5).variances(I, W)
    t.check_sandwich("%s stage 1 validity" % label, (v1 > 0), (v_c > 0), (v_p > 0),
                     why="a variance is valid or not; eps_multiplier bracket 1.5 / 0.5")

    v_cond = np.where(v1 > 0, v_p, 0.0)
    L = NT if (axis == AXIS_TIME) else NFREQ
    dB = end_to_end_bracket(v_cond, m_p, L, two_pass, SIGMA)
    (klo, khi) = keep_bracket(v_cond, SIGMA, dB)
    wlo = std_dev_apply(W, klo.astype(np.float64), axis, 1, 1)
    whi = std_dev_apply(W, khi.astype(np.float64), axis, 1, 1)
    t.check_sandwich("%s end-to-end" % label, w_old[None], wlo, whi,
                     why="our reference's variances, conditioned on the old stage-1 valid"
                         " set; sigma bracket %.2g" % dB)


def main():
    t = harness.Test("rfi_std_dev_clipper")

    manifest = harness.build_manifest()
    if manifest:
        for line in manifest.splitlines():
            if line.startswith("date:") or line.startswith("arch:"):
                t.note(line.strip())

    rng = np.random.default_rng(SEED)

    # _clip_1d first: it is the function with no reference anywhere in the old code.
    for (label, V) in clip1d_cases(rng):
        for sigma in (1.5, SIGMA):
            check_clip1d(t, label, V, sigma)
    report_all_equal(t, rng)

    x = make_input(rng)
    t.note("(2, %d, %d) intensity/weights from seed %d" % (NFREQ, NT, SEED))

    for axis in (AXIS_TIME, AXIS_FREQ):
        for two_pass in (True, False):
            tag = "%s tp=%d" % (AXIS_NAMES[axis], int(two_pass))
            check_transform(t, tag, x, axis, two_pass)
            check_transform(t, tag + " 1-valid", make_one_valid(rng, axis), axis, two_pass)

    return t.done()


if __name__ == "__main__":
    sys.exit(main())
