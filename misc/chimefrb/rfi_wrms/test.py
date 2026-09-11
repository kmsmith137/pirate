#!/usr/bin/env python3
"""Spot test: the weighted mean and variance, pirate vs rf_kernels.

Compares pirate_frb.chimefrb.ReferenceWrms against rf_kernels::weighted_mean_rms, the
kernel it was transcribed from. This is the statistic both chimefrb clippers are built
on, and it is the subtlest numerics in the port, so it is the reference most worth
pinning against the real thing.

WHY THIS TEST EXISTS. The routine unit test ('pirate_frb test --cfrb') compares the CUDA
kernel GpuWrms against ReferenceWrms, so it establishes that the two pirate
implementations agree -- but both were written from one reading of the old code. This is
the test that reads the old code by running it.

TWO THINGS MAKE IT MORE THAN A CALL AND AN allclose().

First, the variance-validity cutoffs. A row whose variance falls below (eps_2*mean)^2 or
eps_3*mean^2 is declared dead and its variance set to exactly zero, and our reference
does not evaluate those cutoffs in quite the same arithmetic as the old kernel (see
ReferenceWrms's docstring: the two-pass 'finalize' has no '- dmean^2' term and applies
only one of the two cutoffs, where a second iterate step applies both). So the
comparison brackets the decision: run the reference with eps_multiplier 0.5 and 1.5, and
require the old kernel's validity to lie between the two. Only then are the numbers
compared, and only on rows both call valid.

Second, the refinements amplify. After nine rounds of discarding outliers, a single
sample that lands on the other side of a threshold in float32 than in float64 moves the
variance by percent, not by roundoff -- so comparing nine float64 refinements against
nine float32 ones is not a test, it is a coin flip. The fix is induction, which is what
the old code's own unit test does: run the old kernel at niter-1 AND at niter, take ONE
reference step from the old kernel's own niter-1 state, and compare. Both sides then
start from bit-identical state and there is nothing to amplify. A pleasant side effect
is that this validates iclip() too, since the survivor set we build with it has to match
the one the old kernel built internally.

Needs a built oldpipe (misc/chimefrb/build_oldpipe.sh). Run me directly, or through
misc/chimefrb/run_spot_tests.py.
"""

import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import harness

from pirate_frb.chimefrb import ReferenceWrms, wrms_iterate, iclip

HERE = os.path.dirname(os.path.abspath(__file__))

SEED = 137

# Small on purpose: nothing is cached between runs. L is a multiple of 8, which
# rf_kernels requires, and R is large enough that the degenerate rows below are a small
# minority rather than the whole test.
NROW = 96
NSAMP = 512

# (niter, iter_sigma, two_pass). The old kernel refuses sigma < 1 when niter > 1, so the
# refining rows use the production values. niter=1 is the base case, where there are no
# refinements and the comparison is direct.
CONFIGS = [
    (1, 0.0, True),
    (1, 0.0, False),
    (9, 5.0, True),
    (9, 3.0, False),
]

# Both means and variances are sums over L=512 terms; the old kernel does them in float32
# with AVX2, our reference in float64, so the expected disagreement is float32 roundoff
# accumulated over the row, ~sqrt(512)*1.2e-7 = 3e-6. 1e-4 leaves margin for the
# cancellation in var = <I^2> - mean^2 on the single-pass path, and is still far tighter
# than any semantic error could hide in.
RTOL = 1.0e-4


def make_input(rng):
    """A (2, R, L) float32 array: arr[0] = intensity, arr[1] = weights.

    Built to reach the corner cases rather than just the common one:

      - most rows are ordinary: Gaussian, with a large positive offset, since the
        eps_2*mean cutoff only means anything when the mean is far from zero;
      - a few rows carry 20-sigma outliers, so the refinements actually discard
        something (at sigma=5 on clean Gaussian data they would discard nothing);
      - a few rows are fully masked (no weight at all), a few are constant (variance
        exactly zero), and a few are nearly constant at a large mean, which is the case
        that lands on the epsilon cutoff and is the whole reason for the bracketing.

    Deliberately NOT included: samples placed AT a threshold. Sampling the ambiguous
    region on purpose would make the brackets straddle at a controlled rate rather than
    a negligible one, which is worse, not better.
    """

    offset = 100.0
    intensity = rng.normal(offset, 10.0, size=(NROW, NSAMP))
    weights = (rng.uniform(size=(NROW, NSAMP)) < 0.8) * rng.uniform(0.5, 1.5, size=(NROW, NSAMP))

    # Rows 0-7: outliers at 20 sigma, well beyond any threshold in CONFIGS.
    for r in range(8):
        cols = rng.choice(NSAMP, size=5, replace=False)
        intensity[r, cols] = offset + 200.0

    intensity[8:12, :] = offset          # constant: variance exactly zero
    weights[12:16, :] = 0.0              # no weight at all
    intensity[16:20, :] = offset * (1.0 + 1.0e-6 * rng.normal(size=(4, NSAMP)))   # on the cutoff

    return np.stack([intensity, weights]).astype(np.float32)


def run_old(x, niter, iter_sigma, two_pass):
    """(mean, var) from rf_kernels. The driver reports rms, so square it."""

    out = harness.run_driver(HERE, x, dtype=np.float32,
                             params={"niter": niter, "iter_sigma": iter_sigma,
                                     "two_pass": int(two_pass)})
    return (out[0].astype(np.float64), out[1].astype(np.float64)**2)


def compare(t, label, x, niter, iter_sigma, two_pass, old_mean, old_var,
            ref_mean_lo, ref_var_lo, ref_mean_hi, ref_var_hi):
    """Bracket the validity decision, then compare the numbers where both agree it holds.

    'lo' is the permissive reference (eps_multiplier=0.5, rejects fewer variances) and
    'hi' the conservative one (1.5, rejects more), so the old kernel's set of valid rows
    must lie between the conservative set and the permissive set.
    """

    ok = t.check_sandwich("%s validity" % label,
                          (old_var > 0), (ref_var_hi > 0), (ref_var_lo > 0),
                          why="a row's variance is valid or not; the two codes evaluate"
                              " the eps_2/eps_3 cutoffs in slightly different arithmetic,"
                              " so the decision is bracketed at eps_multiplier 1.5 and 0.5")

    both = (old_var > 0) & (ref_var_lo > 0)
    t.note("        %d of %d rows valid in both" % (int(np.sum(both)), both.size))

    if not np.any(both):
        return ok

    # Compare against the unbracketed reference (eps_multiplier = 1), which is the real
    # algorithm; the bracketed pair above exists only to judge the validity decision.
    ok &= t.check_allclose("%s mean" % label, ref_mean_lo[both], old_mean[both], rtol=RTOL,
                           why="sums over L=%d in float32 (old) vs float64 (reference)" % NSAMP)
    ok &= t.check_allclose("%s var" % label, ref_var_lo[both], old_var[both], rtol=RTOL,
                           why="likewise; the old kernel reports rms, squared here")
    return ok


def main():
    t = harness.Test("rfi_wrms")

    manifest = harness.build_manifest()
    if manifest:
        for line in manifest.splitlines():
            if line.startswith("date:") or line.startswith("arch:"):
                t.note(line.strip())

    x = make_input(np.random.default_rng(SEED))
    t.note("(2, %d, %d) intensity/weights from seed %d" % (NROW, NSAMP, SEED))

    I = x[0].astype(np.float64)
    W = x[1].astype(np.float64)

    for (niter, iter_sigma, two_pass) in CONFIGS:
        tag = "niter=%d tp=%d" % (niter, int(two_pass))
        (old_mean, old_var) = run_old(x, niter, iter_sigma, two_pass)

        if niter == 1:
            # Base case: no refinements, so a direct comparison is safe.
            refs = {}
            for (name, eps) in (("lo", 0.5), ("hi", 1.5)):
                refs[name] = ReferenceWrms(niter, iter_sigma, two_pass,
                                           eps_multiplier=eps).apply(I, W)
            compare(t, tag, x, niter, iter_sigma, two_pass, old_mean, old_var,
                    refs["lo"][0], refs["lo"][1], refs["hi"][0], refs["hi"][1])
            continue

        # Inductive step: one reference refinement from the OLD kernel's own niter-1
        # state, against the old kernel at niter. Both sides start from bit-identical
        # state, so a survivor that sits near the threshold cannot amplify.
        (prev_mean, prev_var) = run_old(x, niter-1, iter_sigma, two_pass)
        t.note("    %s: one reference step from the old kernel's niter=%d state"
               % (tag, niter-1))

        Wk = iclip(prev_mean, iter_sigma * np.sqrt(prev_var), I, W)
        step = {}
        for (name, eps) in (("lo", 0.5), ("hi", 1.5)):
            step[name] = wrms_iterate(prev_mean, I, Wk, eps_multiplier=eps)

        compare(t, tag, x, niter, iter_sigma, two_pass, old_mean, old_var,
                step["lo"][0], step["lo"][1], step["hi"][0], step["hi"][1])

    return t.done()


if __name__ == "__main__":
    sys.exit(main())
