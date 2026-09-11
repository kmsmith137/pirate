#!/usr/bin/env python3
"""Spot test: std_dev_clipper::_clip_1d() on its own, pirate vs rf_kernels.

_clip_1d() is stage 2 of rf_kernels::std_dev_clipper: given one variance per row, it zeroes
the outliers among them. This test compares pirate_frb.chimefrb.clip_1d() against the real
function, which the driver calls directly.

WHY THIS TEST MATTERS MORE THAN MOST. _clip_1d() has no reference implementation anywhere
in the old code: its own comment says it was never unit-tested, and the old test of the
transform is circular (it calls the real _clip_1d). So our clip_1d() is the first
independent check it has ever had, and a disagreement found here could be a bug in the OLD
code. Driving the function directly, on variance vectors we construct, reaches the
degenerate cases random data never does: 0 to 3 valid entries, Samuelson's bound at small
n, six decades of dynamic range, and every valid variance equal (reported, not asserted;
see report_all_equal()). misc/chimefrb/rfi_std_dev_clipper/ checks the whole transform.

Needs a built oldpipe (misc/chimefrb/build_oldpipe.sh). Run me directly, or through
misc/chimefrb/run_spot_tests.py.
"""

import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import harness

from pirate_frb.chimefrb import clip_1d
from pirate_frb.chimefrb.test_std_dev_clipper import stage2_bracket, ill_conditioned, keep_bracket

HERE = os.path.dirname(os.path.abspath(__file__))
SEED = 137
SIGMA = 3.0          # the production value


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
                             params={"sigma": sigma})

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
    float64 clip_1d() sums exactly here and clips everything. Both are 'right'.
    """

    t.note("  info all-equal valid variances (ill-conditioned, not asserted), sigma=%g:" % SIGMA)
    for m in (2, 3, 64, 1024):
        K = 100
        V = np.zeros((K, 1024), dtype=np.float32)
        for k in range(K):
            V[k, :m] = np.float32(rng.uniform(1.0, 1000.0))
        old = harness.run_driver(HERE, V, dtype=np.float32,
                                 params={"sigma": SIGMA})
        ours = clip_1d(V.astype(np.float64), SIGMA)
        t.note("         n_valid=%4d: old clips everything in %3d of %d, ours in %3d of %d"
               % (m, int(np.sum(np.all(old == 0, axis=1))), K,
                  int(np.sum(np.all(ours == 0, axis=1))), K))


def main():
    t = harness.Test("rfi_std_dev_clip_1d")

    manifest = harness.build_manifest()
    if manifest:
        for line in manifest.splitlines():
            if line.startswith("date:") or line.startswith("arch:"):
                t.note(line.strip())

    rng = np.random.default_rng(SEED)

    for (label, V) in clip1d_cases(rng):
        for sigma in (1.5, SIGMA):
            check_clip1d(t, label, V, sigma)
    report_all_equal(t, rng)

    return t.done()


if __name__ == "__main__":
    sys.exit(main())
