#!/usr/bin/env python3
"""Spot test: the weight upsampler, pirate vs rf_kernels.

Compares pirate_frb.chimefrb.ReferenceWeightUpsampler -- and GpuWeightUpsampler, when cupy
is available -- against rf_kernels::weight_upsampler, the kernel it was transcribed from.

WHY THIS TEST EXISTS. The routine unit test ('pirate_frb test --cfrb') compares the CUDA
kernel against the numpy reference, so it establishes that the two pirate implementations
agree -- but both were written from one reading of the old code, and a misreading would pass
it. This is the test that reads the old code by running it.

There is no arithmetic here, only a comparison and a store, so EVERY comparison below is
exact, made on the raw bits (a uint32 view, which is also how NaN compares equal to itself
and how +0.0 is told from -0.0). The traps it exists to catch are all in the comparison:

  - it is strict, so a low-resolution weight exactly equal to the cutoff masks its cell;
  - it is made in float32 against float32(w_cutoff), which matters for a cutoff float32
    cannot represent, such as 0.1;
  - it is false for NaN, so a NaN low-resolution weight masks its cell;
  - a kept full-resolution weight keeps its exact bits (the old code ANDs an all-ones mask
    into it), including NaN and -0.0, and a masked one becomes +0.0 whatever it held.

NOT COVERED: denormal weights. pirate builds with --use_fast_math, so a denormal may compare
as zero on the GPU, while numpy compares by IEEE rules and the old code depends on its
process's FTZ/DAZ state. Real weights are counts, never denormal, so no draw makes one.

Needs a built oldpipe (misc/chimefrb/build_oldpipe.sh). Run me directly, or through
misc/chimefrb/run_spot_tests.py.
"""

import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import harness

from pirate_frb.chimefrb import ReferenceWeightUpsampler

HERE = os.path.dirname(os.path.abspath(__file__))

# Any fixed value works; it is written down so the comparison is reproducible without
# committing the arrays it generates.
SEED = 137

# The production call: the wi_sub_pipeline upsample, 1024 -> 16384 channels at native time
# resolution, with w_cutoff = 0 and low-resolution weights that are counts 0..16.
PRODUCTION = (16, 1, 0.0)

# What the old kernel accepts: Df and Dt in {1, 2, 4} or a multiple of 8, and nt_lo % 8 == 0.
DF_CHOICES = [1, 2, 4, 8, 16, 24]
DT_CHOICES = [1, 2, 4, 8, 16]

NDRAW = 30


def make_lores(rng, nfreq_lo, nt_lo, Df, Dt, w_cutoff):
    """Low-resolution weights: counts, with the corner cases of the docstring planted in."""

    ncell = Df * Dt
    w = rng.integers(0, ncell + 1, size=(nfreq_lo, nt_lo)).astype(np.float32)

    # A few percent of cells of each awkward kind. The ulp-adjacent pair is only drawn for a
    # positive cutoff: one ulp below zero is denormal, which no draw may contain.
    kinds = [np.float32(w_cutoff), np.float32(-0.0), np.float32(-1.5), np.float32(np.nan),
             np.float32(np.inf)]
    if w_cutoff > 0:
        c = np.float32(w_cutoff)
        kinds += [np.nextafter(c, np.float32(0)), np.nextafter(c, np.float32(np.inf))]

    flat = w.reshape(-1)
    for x in kinds:
        n = max(1, flat.size // 50)
        flat[rng.choice(flat.size, size=n, replace=False)] = x

    return w


def make_hires(rng, nfreq_hi, nt_hi):
    """Full-resolution weights: {0,1}-like counts, with non-finite values and -0.0 planted in
    kept and masked cells alike -- a kept weight must come back with its exact bits."""

    w = rng.uniform(0.5, 1.5, size=(nfreq_hi, nt_hi)).astype(np.float32)
    w *= (rng.uniform(size=(nfreq_hi, nt_hi)) < 0.8)

    flat = w.reshape(-1)
    for x in (np.float32(np.nan), np.float32(np.inf), np.float32(-np.inf), np.float32(-0.0)):
        n = max(1, flat.size // 100)
        flat[rng.choice(flat.size, size=n, replace=False)] = x

    return w


def bits(a):
    """The raw bits of a float32 array, as float64 so that harness can compare them."""
    return np.ascontiguousarray(a, dtype=np.float32).view(np.uint32).astype(np.float64)


def check(t, label, cp, hi, lo, Df, Dt, w_cutoff):
    old = harness.run_driver(HERE, [hi, lo], dtype=np.float32,
                             params={"Df": Df, "Dt": Dt, "w_cutoff": repr(float(w_cutoff))})

    ref = ReferenceWeightUpsampler(Df, Dt, w_cutoff).apply(hi[None], lo[None])[0]

    ok = t.check_allclose(label, bits(ref), bits(old), rtol=0.0,
                          why="bit patterns of the upsampled weights; no arithmetic is done,"
                              " so the two must agree exactly")

    nmasked = int(np.sum(old == 0.0)) - int(np.sum(hi == 0.0))
    t.note("        %d of %d full-resolution weights newly zeroed" % (nmasked, old.size))

    if cp is not None:
        from pirate_frb.chimefrb import GpuWeightUpsampler
        g_hi = cp.asarray(hi[None])
        GpuWeightUpsampler(Df, Dt, w_cutoff).launch(g_hi, cp.asarray(lo[None]))
        cp.cuda.get_current_stream().synchronize()
        ok &= t.check_allclose(label + " (gpu)", bits(cp.asnumpy(g_hi)[0]), bits(old), rtol=0.0,
                               why="the CUDA kernel against the old code, same comparison")

    return ok


def main():
    t = harness.Test("rfi_weight_upsample")

    manifest = harness.build_manifest()
    if manifest:
        for line in manifest.splitlines():
            if line.startswith("date:") or line.startswith("arch:"):
                t.note(line.strip())

    try:
        import cupy as cp
        cp.cuda.runtime.getDeviceCount()
    except Exception:
        cp = None
        t.note("cupy not available: comparing the numpy reference only")

    rng = np.random.default_rng(SEED)

    # The production configuration, on a few channels' worth of the real shape.
    (Df, Dt, w_cutoff) = PRODUCTION
    lo = make_lores(rng, 64, 64, Df, Dt, w_cutoff)
    hi = make_hires(rng, 64*Df, 64*Dt)
    check(t, "production (16,1) cutoff 0", cp, hi, lo, Df, Dt, w_cutoff)

    # Random draws over what the old kernel accepts.
    for i in range(NDRAW):
        Df = int(rng.choice(DF_CHOICES))
        Dt = int(rng.choice(DT_CHOICES))
        while Df * Dt > 64:
            Df = int(rng.choice(DF_CHOICES))
            Dt = int(rng.choice(DT_CHOICES))

        nfreq_lo = int(rng.integers(1, 17))
        nt_lo = 8 * int(rng.integers(1, 9))          # the old kernel needs nt_lo % 8 == 0
        w_cutoff = float(rng.choice([0.0, 0.0, 0.1, float(rng.uniform(0.5, 3.0))]))

        lo = make_lores(rng, nfreq_lo, nt_lo, Df, Dt, w_cutoff)
        hi = make_hires(rng, nfreq_lo*Df, nt_lo*Dt)
        check(t, "Df=%d Dt=%d cutoff=%g" % (Df, Dt, w_cutoff), cp, hi, lo, Df, Dt, w_cutoff)

    return t.done()


if __name__ == "__main__":
    sys.exit(main())
