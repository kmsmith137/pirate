#!/usr/bin/env python3
"""Spot test: the frequency-direction spline detrender, pirate vs rf_kernels.

Compares pirate_frb.chimefrb.ReferenceSplineDetrender -- and GpuSplineDetrender, when
cupy is available -- against rf_kernels::spline_detrender, at the two shapes the old
search's production RFI chain runs it at.

WHY THIS TEST EXISTS. The routine unit test ('pirate_frb test --cfrb') compares the CUDA
kernel against the numpy reference, so it establishes that the two pirate
implementations agree -- but both were written from one reading of the old code, and a
misreading would pass it. This is the test that reads the old code by running it. The
traps it is here to catch are the ones a transcription gets wrong silently: the bin
edges (fractional positions, rounded), the Hermite basis and its coefficient order, and
above all the regularization strength epsilon * (sum of weights) / nbins, which SCALES
WITH THE TOTAL WEIGHT. A fixed-strength reading would agree on a fully valid sample and
disagree on every heavily flagged one, which is why the weights below are flagged as
the pipeline flags them, dead runs included.

WHAT "AGREE" MEANS. The old kernel accumulates in float32, sequentially over bins of up
to 2731 channels, and factors its normal equations in float32 with no rescaling;
pirate's reference is float64 throughout and its kernel equilibrates. The comparison is
therefore made on the fitted BASELINE (intensity minus output), which is O(100) so a
relative tolerance means something, at the old code's float32 roundoff: sqrt(2731)
terms of size 100 at 6e-8 each is a few 1e-6 in the sums, and the unequilibrated solve
multiplies that by its condition number, of order 1e2 to 1e3. Measured, the old code
lands 9e-6 to 3.8e-5 from both pirate implementations at weighted channels, and 6e-5 to
2.1e-4 at zero-weight ones; the output reports the two separately, since a zero-weight
channel's value is set by the penalty alone, in the directions where an unequilibrated
solve is least accurate. The two pirate implementations agree with each other far more
closely than either does with the old code.

KNOWN DEPARTURE, not exercised here: a zero-weight channel holding NaN poisons the old
kernel's whole time sample (it multiplies weight by intensity) and contributes exactly
zero in pirate. No NaNs are fed here; the unit test pins pirate's side of it.

Needs a built oldpipe (misc/chimefrb/build_oldpipe.sh). Run me directly, or through
misc/chimefrb/run_spot_tests.py.
"""

import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import harness

from pirate_frb.chimefrb import ReferenceSplineDetrender
from pirate_frb.chimefrb.test_spline_detrender import random_intensity, random_weights

HERE = os.path.dirname(os.path.abspath(__file__))

# Any fixed value works; it is written down so the comparison is reproducible without
# committing the array it generates.
SEED = 137

# The production shapes, as (nfreq, nbins, epsilon, weight kind). 32 time samples: a
# multiple of 8 for the old kernel, of 32 for the GPU one, and enough columns for the
# mixture of weight patterns make_input() draws.
CONFIGS = [(1024, 6, 3.0e-4, 'counts'), (16384, 6, 3.0e-4, 'binary')]
NT = 32

# See "what agree means" in the docstring. MEASURED on the weights this test draws now
# (the unit test's generators -- see make_input): the old code's baseline differs from the
# float64 reference by 6.0e-5 (1024 channels) and 2.1e-4 (16384 channels) at worst, and
# from the GPU by 8.3e-5 and 1.8e-4. 5e-4 leaves a margin of 2.3x over the worst of those
# while staying far below anything a semantic error (a wrong bin edge, coefficient order,
# or regulator strength) could hide in.
RTOL = 5.0e-4


def make_input(rng, nfreq, nbins, kind):
    """A (2, F, T) float32 array: arr[0] = intensity, arr[1] = weights.

    Both halves come from the unit test's own generators
    (pirate_frb/chimefrb/test_spline_detrender.py), so the two tests exercise one set of
    weight patterns rather than two that can drift apart. At a production kind the columns
    are Binomial(16, p) counts or Bernoulli {0,1}, with a contiguous dead run cut into half
    of them and half of those runs covering whole spline bins -- the shape a bad-channel
    mask leaves behind, and what pins down the weight-scaled regulator strength. The
    intensity is a baseline in the spline's span, of order 100, so that a relative
    comparison means something.

    Those generators take a leading beam axis, and this test compares one beam.
    """
    intensity = random_intensity(rng, 1, nfreq, nbins, NT)[0]
    weights = random_weights(rng, 1, nfreq, nbins, NT, kind)[0]
    return np.stack([intensity, weights]).astype(np.float32)


def main():
    t = harness.Test("spline_detrender")

    manifest = harness.build_manifest()
    if manifest:
        for line in manifest.splitlines():
            if line.startswith("date:") or line.startswith("arch:"):
                t.note(line.strip())

    try:
        import cupy as cp
        from pirate_frb.chimefrb import GpuSplineDetrender
    except ImportError:
        cp = None
        t.note("cupy not available: comparing the numpy reference only")

    rng = np.random.default_rng(SEED)

    for (nfreq, nbins, epsilon, kind) in CONFIGS:
        x = make_input(rng, nfreq, nbins, kind)
        t.note("(2, %d, %d) intensity/%s weights from seed %d; nbins=%d epsilon=%g"
               % (nfreq, NT, kind, SEED, nbins, epsilon))

        old = harness.run_driver(HERE, x, params={"nbins": nbins, "epsilon": epsilon},
                                 dtype=np.float32)

        ref = ReferenceSplineDetrender(nfreq, nbins, epsilon)
        new = ref.apply(x[0][None, :, :], x[1][None, :, :])[0]

        # Compare the fitted baselines, not the residuals (see the docstring). Both codes
        # subtract at every channel, weighted or not, so every channel is compared.
        base_old = x[0].astype(np.float64) - old
        base_new = x[0].astype(np.float64) - new

        nzero = int(np.sum(x[1] == 0))
        t.note("        %d of %d weights are zero; %d columns have a whole dead bin"
               % (nzero, x[1].size, NT // 4))

        def split_note(label, base):
            rel = np.abs(base - base_old) / np.abs(base_old)
            wz = (x[1] == 0)
            t.note("        %s: max rel diff %.3g at weighted channels, %.3g at zero-weight ones"
                   % (label, rel[~wz].max(), rel[wz].max() if wz.any() else 0.0))

        split_note("reference vs old", base_new)
        t.check_allclose("baseline, reference (%d, %d)" % (nfreq, nbins), base_new, base_old,
                         rtol=RTOL,
                         why="same estimator; rf_kernels sums and factors in float32 without"
                             " rescaling, the reference is float64")

        if cp is not None:
            det = GpuSplineDetrender(1, nfreq, NT, nbins, epsilon)
            gi = cp.asarray(x[0][None, :, :])
            det.launch(gi, cp.asarray(x[1][None, :, :]), None)
            cp.cuda.get_current_stream().synchronize()
            base_gpu = x[0].astype(np.float64) - cp.asnumpy(gi)[0]

            split_note("GPU vs old", base_gpu)
            t.check_allclose("baseline, GPU (%d, %d)" % (nfreq, nbins), base_gpu, base_old,
                             rtol=RTOL,
                             why="same estimator; both float32, the GPU with an equilibrated"
                                 " solve and a blocked accumulation")

    return t.done()


if __name__ == "__main__":
    sys.exit(main())
