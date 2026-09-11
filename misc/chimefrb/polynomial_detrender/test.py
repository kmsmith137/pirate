#!/usr/bin/env python3
"""Spot test: the polynomial detrender, pirate vs rf_kernels.

Compares pirate_frb.chimefrb.ReferencePolynomialDetrender -- and GpuPolynomialDetrender,
when it is built and cupy is available -- against rf_kernels::polynomial_detrender, at
the production configuration (degree 4, epsilon 0.01, 1024-sample chunks, along time)
and on a sweep of degrees along both axes.

WHY THIS TEST MATTERS MORE THAN MOST. The transform's main effect on the data stream is
its conditioning GATE: rows whose Cholesky pivots fall below epsilon have all their
weights zeroed. That gate has no reference implementation anywhere in the old code --
its own unit test only checks that rows with fewer than polydeg+1 weighted samples get
masked -- and the old python transform has no gate at all. So the reference's gate is
its first transcription, and this is the one place it is checked against the old
binary. The rows below are built to straddle it: contiguous dead runs of 30-90% of the
chunk, which at degree 4 cross the threshold at about half the chunk.

WHAT "AGREE" MEANS. The gate decision is bracketed, not compared: a row the float64
reference puts more than GATE_BAND (2%) from the threshold must land on that side in the
old code too, and rows inside the band may go either way (both codes are float32 near a
cancellation there). The weights are compared EXACTLY: the old code never writes them
except to zero a masked row. The fitted polynomial is compared on rows both codes let
through, at a relative tolerance the old code's float32 arithmetic sets: ~1024-term
sums (about 2e-6 relative) through an unequilibrated solve whose condition the gate
bounds (pivots above 0.01, so at most ~1e2 amplification).

KNOWN DEPARTURE, not exercised here: a zero-weight sample holding NaN turns the old
kernel's whole row to NaN (it multiplies weight by intensity) and contributes exactly
zero in pirate. No NaNs are fed here; the unit test pins pirate's side of it.

Needs a built oldpipe (misc/chimefrb/build_oldpipe.sh). Run me directly, or through
misc/chimefrb/run_spot_tests.py.
"""

import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import harness

from pirate_frb.chimefrb.ReferencePolynomialDetrender import (ReferencePolynomialDetrender,
                                                              AXIS_FREQ, AXIS_TIME)
from pirate_frb.chimefrb.test_polynomial_detrender import (gate_bracket, random_weights,
                                                           random_intensity)

HERE = os.path.dirname(os.path.abspath(__file__))
SEED = 137

# (axis, polydeg, epsilon, nt_chunk, F, T, weight kind). Along time nfreq is a spectator
# (rows are independent), so the production configuration is represented at full chunk
# length with a few dozen rows; the degree sweep uses shorter chunks. Along frequency the
# rows are time samples and F is the fit length. 'mix' draws every pattern of
# test_polynomial_detrender.weight_row(); the production rows pin their own kind but are
# given dead runs too, since that is what the chain produces.
EPSILON = 0.01
CONFIGS = ([(AXIS_TIME, 4, EPSILON, 1024, 64, 2048, 'counts'),
            (AXIS_TIME, 4, EPSILON, 1024, 64, 2048, 'binary'),
            (AXIS_TIME, 4, EPSILON, 1024, 64, 2048, None)]
           + [(AXIS_TIME, d, EPSILON, 256, 32, 512, None) for d in range(0, 9)]
           + [(AXIS_FREQ, d, EPSILON, 0, 512, 32, None) for d in (0, 4, 8)])

# See "what agree means". Both relative to each row's max |fit|, and per axis. MEASURED
# (seed 137): along time, weighted samples differ from the float64 reference by at most
# 5.4e-5 at the production configuration and at every degree 0..8 (nt_chunk 256), and
# extrapolated (zero-weight) samples by up to 6e-3 at production and 1.4e-2 at degree 7.
# Along frequency, where the old kernel sums the 512 channels sequentially in float32,
# weighted samples differ by up to 2.9e-4 (degree 4) and 1.1e-3 (degree 8), and
# extrapolated ones by up to 0.23 at degree 8 -- the old unequilibrated solve's own error,
# magnified by extrapolation -- so the zero-weight comparison is not made along frequency:
# it would say nothing about either code. A semantic error (a wrong grid, basis or
# coefficient order) shows up at weighted samples at 1e-1 and beyond.
RTOL_WEIGHTED = {AXIS_TIME: 2.0e-4, AXIS_FREQ: 4.0e-3}
RTOL_UNWEIGHTED = {AXIS_TIME: 5.0e-2}

AXIS_NAMES = {AXIS_FREQ: "FREQ", AXIS_TIME: "TIME"}


def make_input(rng, ref, F, T, kind):
    """A (2, F, T) float32 array: arr[0] = intensity, arr[1] = weights.

    The production kinds ('counts', 'binary') get a dead run on two rows in three, so
    that the gate is exercised on the weights the chain really produces; the mixed
    kind already includes dead runs.
    """
    shape = (1, F, T)
    intensity = random_intensity(rng, ref, shape)[0]
    if kind is None:
        weights = random_weights(rng, ref, shape)[0]
    else:
        weights = random_weights(rng, ref, shape, kind)[0]
        runs = random_weights(rng, ref, shape, 'dead_run')[0]
        pick = (rng.uniform(size=ref._row_shape(shape)) < 2.0 / 3.0)
        pick = np.broadcast_to(ref._row_mask_expand(pick, shape), shape)[0]
        weights = np.where(pick, np.where(runs != 0, weights, 0.0), weights).astype(np.float32)
    return np.stack([intensity, weights])


def fit_check(t, label, ref, x, base_got, base_want, both):
    """Compare two fitted polynomials on the rows 'both' (per-row boolean) let through,
    relative to each ROW's scale (the max |want| over the row): the fit crosses zero, so a
    pointwise relative difference would be dominated by the samples nearest the
    crossings. Weighted and zero-weight samples are checked separately."""
    got = ref._rows(base_got[None])
    want = ref._rows(base_want[None])
    w = ref._rows(x[1][None])
    rows = both.reshape(-1)
    if not rows.any():
        t.note("        %s: no row fitted by both; nothing to compare" % label)
        return
    scale = np.abs(want[rows]).max(axis=1, keepdims=True)
    scale = np.where(scale > 0, scale, 1.0)
    (g, m, w) = (got[rows] / scale, want[rows] / scale, w[rows])
    d = np.abs(g - m)
    t.note("        %s: fit differs by %.3g (weighted) / %.3g (zero-weight) x the row's max |fit|"
           % (label, d[w != 0].max() if (w != 0).any() else 0.0,
              d[w == 0].max() if (w == 0).any() else 0.0))
    t.check_allclose("fit at weighted, %s" % label, g[w != 0], m[w != 0], rtol=0.0,
                     atol=RTOL_WEIGHTED[ref.axis],
                     why="relative to the row's max |fit|; same estimator, rf_kernels sums and"
                         " factors in float32 without rescaling")
    if (ref.axis in RTOL_UNWEIGHTED) and (w == 0).any():
        t.check_allclose("fit at zero-weight, %s" % label, g[w == 0], m[w == 0], rtol=0.0,
                         atol=RTOL_UNWEIGHTED[ref.axis],
                         why="extrapolation beyond the data, where an unequilibrated float32"
                             " solve is least accurate; looser on purpose")


def compare(t, label, x, out_i, out_w, ref, model):
    """The three comparisons of one implementation's (out_i, out_w) against the reference:
    the gate decision (bracketed), the weights (exact), the fit (per row scale)."""
    (F, T) = x.shape[1:]
    rows_w = ref._rows(out_w[None])
    rows_w0 = ref._rows(x[1][None])
    rshape = ref._row_shape((1, F, T))
    masked = (rows_w == 0).all(axis=1).reshape(rshape)

    (lo, surely_passed, in_band) = gate_bracket(ref, x[1][None])
    hi = ~surely_passed
    nband = int(in_band.sum())
    t.note("        %s: %d of %d rows masked; %d rows inside the gate band"
           % (label, int(masked.sum()), masked.size, nband))
    t.check_sandwich("gate, %s" % label, masked.astype(float), lo.astype(float), hi.astype(float),
                     why="rows the float64 reference puts outside the roundoff band must land"
                         " on their side; both codes are float32 near a cancellation inside it")

    # Weights: exact. Unmasked rows untouched, masked rows zero (which 'masked' asserts).
    unmasked_rows = ~masked.reshape(-1)
    if unmasked_rows.any():
        t.check_allclose("weights kept, %s" % label, rows_w[unmasked_rows], rows_w0[unmasked_rows],
                         rtol=0.0, why="the old code does not write the weights of a row it fits")

    both = ~masked & ~ref.masked(x[1][None])
    fit_check(t, label, ref, x, x[0].astype(np.float64) - out_i, model[0], both)
    return masked


def main():
    t = harness.Test("polynomial_detrender")

    manifest = harness.build_manifest()
    if manifest:
        for line in manifest.splitlines():
            if line.startswith("date:") or line.startswith("arch:"):
                t.note(line.strip())

    gpu = None
    try:
        import cupy as cp
        from pirate_frb.chimefrb import GpuPolynomialDetrender
        gpu = (cp, GpuPolynomialDetrender)
    except ImportError:
        t.note("GpuPolynomialDetrender or cupy not available: comparing the numpy reference only")

    rng = np.random.default_rng(SEED)

    for (axis, polydeg, epsilon, nt_chunk, F, T, kind) in CONFIGS:
        ref = ReferencePolynomialDetrender(polydeg, epsilon, nt_chunk, axis)
        x = make_input(rng, ref, F, T, kind)
        t.note("axis=%s polydeg=%d epsilon=%g nt_chunk=%d (F, T)=(%d, %d) weights=%s, seed %d"
               % (AXIS_NAMES[axis], polydeg, epsilon, nt_chunk, F, T, kind or 'mix', SEED))

        old = harness.run_driver(HERE, x, params={"polydeg": polydeg, "epsilon": epsilon,
                                                  "axis": int(axis), "nt_chunk": nt_chunk},
                                 dtype=np.float32)

        model = ref.fit(x[0][None], x[1][None])
        old_masked = compare(t, "old vs reference", x, old[0], old[1], ref, model)

        if (gpu is not None) and (axis == AXIS_TIME):
            (cp, Gpu) = gpu
            det = Gpu(polydeg, epsilon, nt_chunk)
            gi = cp.asarray(x[0][None])
            gw = cp.asarray(x[1][None])
            det.launch(gi, gw)
            cp.cuda.get_current_stream().synchronize()
            (gi, gw) = (cp.asnumpy(gi)[0], cp.asnumpy(gw)[0])
            gpu_masked = compare(t, "GPU vs reference", x, gi, gw, ref, model)
            # And the two float32 codes directly against each other, where both fit.
            fit_check(t, "GPU vs old", ref, x, x[0].astype(np.float64) - gi,
                      x[0].astype(np.float64) - old[0], ~old_masked & ~gpu_masked)

    return t.done()


if __name__ == "__main__":
    sys.exit(main())
