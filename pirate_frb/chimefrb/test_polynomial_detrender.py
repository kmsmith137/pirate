"""Randomized unit tests for GpuPolynomialDetrender, and the data generators they share
with misc/chimefrb/polynomial_detrender/.

Dispatched from ``python -m pirate_frb test --cfrb``.

The GPU port is checked against ReferencePolynomialDetrender, a numpy transcription of
the old kernel, on the weight patterns the real pipeline produces. The transform's main
behaviour is its conditioning GATE (see the reference's class docstring): at the
production setting it zeroes the weights of any channel whose weighted samples form a
contiguous run shorter than about half the chunk, and leaves that channel's intensity
alone. So the weights drawn here are dominated by contiguous dead runs of 30-90% of the
chunk, which straddle the gate for every degree, and the gate decision is checked
first, as a sandwich: a row the float64 reference puts clearly on one side must land
there on the GPU, and only rows within a roundoff band of the threshold may go either
way (GATE_BAND). The fitted polynomial is then compared on rows both sides let through.

Beyond the float comparison, several properties hold BITWISE and are asserted as such:
run-to-run reproducibility; ``warps_per_block`` is a pure occupancy knob; rows are
independent of each other and of which chunk they sit in; NaN at a zero-weight sample
changes nothing elsewhere; and scaling every weight by 4 changes nothing at all (the
estimator and the gate are exactly weight-scale invariant, and with correctly rounded
sqrt and reciprocal so is the float32 arithmetic).

WHAT THIS FILE CANNOT ESTABLISH: a misreading of the old kernel would pass every check
here, since the reference and the port were written from the same reading -- and the
gate has no reference implementation in the old code at all. The spot check
misc/chimefrb/polynomial_detrender/ runs the old code itself.
"""

import numpy as np

from .ReferencePolynomialDetrender import (ReferencePolynomialDetrender, AXIS_TIME,
                                           legendre, z_grid)
from ..utils import atomic_print
from .testutils import default_rng as _default_rng


# The production chain's three instances are one configuration, (polydeg, epsilon,
# nt_chunk), seen with count-valued weights (the 16x-downsampled sub-pipelines) and with
# {0,1} weights (the full-band instance). Drawn often, so a run spends its iterations
# where the port will.
PRODUCTION_CONFIGS = [(4, 0.01, 1024, 'counts'), (4, 0.01, 1024, 'binary')]

# Per-row weight patterns; see weight_row().
WEIGHT_KINDS = ['ones', 'binary', 'counts', 'continuous', 'dead_run', 'gap', 'sparse', 'zero']

WARP_COUNTS = [4, 8, 16]

# The gate's roundoff band. The old gate fails a row at the first pivot j whose ratio
# pivot_j is not > epsilon (ReferencePolynomialDetrender.gate_pivots). A float32
# implementation is held to that decision only where the float64 reference puts the pivot
# clearly on one side: with, per pivot,
#
#     band_j = GATE_BAND * epsilon  +  GATE_BAND_MACH * eps_mach * sum_{k < j} 1 / pivot_k,
#
# a row with SOME pivot_j < epsilon - band_j must be masked by the kernel, a row with EVERY
# pivot_j > epsilon + band_j must not be, and any other row may go either way. Two terms
# because float32 error in a pivot has two sources. The pivot is the difference
# 1 - sum_k L_jk^2, which near the threshold is a small fraction of its terms, so the
# roundoff in the accumulated normal matrix (about sqrt(n)*eps_mach relative) is amplified
# by the cancellation -- to a few 1e-4 relative at n = 1024 and epsilon = 0.01, an order of
# magnitude under the first term. And each pivot is computed THROUGH the ones before it,
# dividing by L_kk = sqrt(pivot_k) at every step, so a run of small pivots compounds the
# error roughly as the sum of their inverses; at degree 7 with pivots of 1e-3 that reaches
# 1e-3 absolute, far beyond any relative band on a small epsilon. MEASURED on the two rows
# that first exposed this: error / (eps_mach * sum) of 1.6 and 0.2; GATE_BAND_MACH = 8 is
# 5x above the larger. At the production configuration the second term is ~1e-5 against a
# first term of 2e-4, so a row lands in the band only when a dead run's length is within a
# couple of samples of the crossing; the coverage report tracks how often. Same
# construction as stage2_bracket() in test_std_dev_clipper.py: a bracket, because the old
# code is float32 too.
GATE_BAND = 0.02
GATE_BAND_MACH = 8.0


def gate_bracket(ref, weights):
    """Per row of the reference's axis: (surely_masked, surely_passed, in_band) booleans
    (see GATE_BAND). A row with NaN weights counts as surely masked, which is what both
    codes do with it.
    """
    piv = ref.gate_pivots(weights)                                  # (..., N)
    with np.errstate(divide='ignore', invalid='ignore', over='ignore'):
        inv = np.where(piv > 0, 1.0 / np.where(piv > 0, piv, 1.0), 0.0)
        S = np.cumsum(inv, axis=-1) - inv                           # sum over k < j
        band = GATE_BAND * ref.epsilon + GATE_BAND_MACH * np.finfo(np.float32).eps * S
        surely_masked = np.any(~(piv >= ref.epsilon - band), axis=-1)
        surely_passed = np.all(piv > ref.epsilon + band, axis=-1)
    return (surely_masked, surely_passed, ~surely_masked & ~surely_passed)


# Tolerances for the float comparison of the fitted polynomial, per row, with 'scale' the
# max |intensity| over the row's weighted samples and (rmin, lmin) from
# ReferencePolynomialDetrender.conditioning():
#
#     at weighted samples:  |gpu - reference|  <=  TOL_WEIGHTED * eps_mach * scale / rmin
#     at every sample:      |gpu - reference|  <=  TOL_ALL      * eps_mach * scale / lmin
#
# The same two-regime form as test_spline_detrender.py: at a weighted sample the float32
# error is the accumulation plus an equilibrated solve whose error is of order
# eps_mach / rmin -- and here the gate GUARANTEES rmin > epsilon on every row compared, so
# the first bound is at worst eps_mach * scale / epsilon. At a sample without weight the
# polynomial is extrapolated across a gap or beyond a dead run, in the directions the
# pivots see least, where the coefficient error is of order eps_mach / lmin.
#
# MEASURED over 400 random draws, production configuration included, in these units:
#
#     weighted (/ rmin):   median 4.2   p90 8.1   max 26
#     all      (/ lmin):   median 3.9   p90 6.3   max 12
#
# with every weighted-sample tail case at degree 7 or 8, where the coefficient error
# compounds through the small pivots exactly as the gate statistic does (see GATE_BAND) and
# so outruns the single 1/rmin factor; at the production configuration the worst draw was
# 9.6 on the all-samples bound and under 8 on the weighted one. The tolerances sit 5x above
# the observed maxima; a basis, grid or gate error shows up at 1e3 and beyond, so the margin
# costs no detection power.
TOL_WEIGHTED = 128.0
TOL_ALL = 64.0


def random_config(rng):
    """Draw (polydeg, epsilon, nt_chunk, weight kind, warps_per_block).

    Production values 40% of the time; otherwise polydeg 0..8, epsilon log-uniform over
    1e-4..0.3, nt_chunk a multiple of 64 from 64 to 1024 (log-uniform, so short chunks --
    where a warp's lanes hold two or four samples each -- are common). A weight kind of
    None means "mix the patterns per row" (random_weights()); the production draws pin
    their own.
    """
    W = int(rng.choice(WARP_COUNTS))
    if rng.uniform() < 0.4:
        (polydeg, epsilon, nt_chunk, kind) = PRODUCTION_CONFIGS[rng.integers(len(PRODUCTION_CONFIGS))]
        return (polydeg, epsilon, nt_chunk, kind, W)

    polydeg = int(rng.integers(0, 9))
    epsilon = float(10.0 ** rng.uniform(-4.0, np.log10(0.3)))
    nt_chunk = 64 * int(np.exp(rng.uniform(0.0, np.log(16.0))) + 0.5)
    nt_chunk = int(np.clip(nt_chunk, 64, 1024))
    return (polydeg, epsilon, nt_chunk, None, W)


def random_geometry(rng):
    """Draw (M, nfreq, nchunk): rows are independent, so nfreq stays small."""
    M = int(rng.integers(1, 4))
    nfreq = int(rng.integers(1, 49))
    nchunk = int(rng.integers(1, 4))
    return (M, nfreq, nchunk)


def weight_row(rng, n, N, kind, base=None):
    """One row of n weights, float64, of the given kind. N = polydeg + 1.

      ones         every weight 1
      binary       Bernoulli {0,1} at a per-row rate in [0.3, 1] (a clipper's mask)
      counts       integers 0..16, Binomial(16, p) (16x-downsampled clipper output)
      continuous   uniform in [0, 2] (nothing in the pipeline makes these; they pin the
                   weighted fit)
      dead_run     one of the above with all weight OUTSIDE a contiguous run of length
                   0.3-0.9 n set to zero: the production mask geometry, and the pattern
                   that straddles the gate (a run of about half the chunk at degree 4)
      gap          ones or binary with an interior zero gap of 0.3-0.8 n and data on both
                   sides: usually passes the gate, and makes the fit extrapolate
      sparse       k isolated weighted samples, k in 1..N+3: exactly singular for k <= N-1
      zero         no weight: pivot 0 fails, the row is left untouched

    'base' names the pattern under a dead run or gap ('ones', 'binary', 'counts' or
    'continuous'); by default one is drawn at random.
    """
    if kind == 'ones':
        return np.ones(n)
    if kind == 'zero':
        return np.zeros(n)
    if kind == 'sparse':
        k = int(rng.integers(1, min(N + 3, n) + 1))
        c = np.zeros(n)
        c[rng.choice(n, k, replace=False)] = rng.uniform(0.5, 16.0, size=k)
        return c

    p = rng.uniform(0.3, 1.0)
    if kind in ('binary', 'counts', 'continuous'):
        base = kind
    elif base is None:
        base = ['ones', 'binary', 'counts', 'continuous'][int(rng.integers(4))]
    if base == 'ones':
        c = np.ones(n)
    elif base == 'binary':
        c = (rng.uniform(size=n) < p).astype(np.float64)
    elif base == 'counts':
        c = rng.binomial(16, p, size=n).astype(np.float64)
    else:
        c = rng.uniform(0.0, 2.0, size=n)

    if kind == 'dead_run':
        L = max(1, int(round(rng.uniform(0.3, 0.9) * n)))
        lo = int(rng.integers(0, n - L + 1))
        keep = np.zeros(n, dtype=bool)
        keep[lo:lo+L] = True
        c[~keep] = 0.0
    elif kind == 'gap':
        g = int(round(rng.uniform(0.3, 0.8) * n))
        g = int(np.clip(g, 1, max(1, n - 2)))
        lo = int(rng.integers(1, max(2, n - g)))
        c[lo:lo+g] = 0.0
    return c


def random_weights(rng, ref, shape, kind=None):
    """(M, F, T) float32 weights, one pattern per row of the reference's axis.

    kind=None mixes the families with the probabilities below. A fixed kind (the
    production draws) uses that family, with a dead run cut into one row in three: the
    chain's clippers flag in runs, and without them the production configuration would
    never reach the gate (a Bernoulli or Binomial mask alone does not trip it).
    """
    probs = {'ones': 0.10, 'binary': 0.15, 'counts': 0.15, 'continuous': 0.10,
             'dead_run': 0.25, 'gap': 0.10, 'sparse': 0.10, 'zero': 0.05}
    kinds = list(probs.keys())
    pk = np.array(list(probs.values()))

    rows = ref._rows(np.zeros(shape))
    (K, n) = rows.shape
    w = np.empty((K, n))
    for k in range(K):
        if kind is None:
            w[k] = weight_row(rng, n, ref.N, kinds[rng.choice(len(kinds), p=pk)])
        elif rng.uniform() < 1.0 / 3.0:
            w[k] = weight_row(rng, n, ref.N, 'dead_run', base=kind)
        else:
            w[k] = weight_row(rng, n, ref.N, kind)
    return ref._unrows(w, shape).astype(np.float32)


def random_intensity(rng, ref, shape, noise=1.0):
    """(M, F, T) float32: per row, a polynomial in the fit's span plus Gaussian noise.

    Legendre coefficients of order 100 (constant term) and 30 (the rest), so the fitted
    values are O(100) and a relative comparison means something. With noise = 0 the
    reference comparison becomes a nulling test.
    """
    rows = ref._rows(np.zeros(shape))
    (K, n) = rows.shape
    coeffs = 30.0 * rng.standard_normal((K, ref.N))
    coeffs[:, 0] += 100.0
    base = coeffs @ legendre(ref.N, z_grid(n))
    base += noise * rng.standard_normal(base.shape)
    return ref._unrows(base, shape).astype(np.float32)


def inject_nans(rng, ref, intensity, weights):
    """Copies of (intensity, weights) with NaNs planted, and where.

    Returns ``(intensity, weights, zw_nan, rows_wnan, rows_nanw)``: ``zw_nan`` is a
    boolean array over samples, the zero-weight samples given a NaN intensity (the case
    the port handles differently from the old code, which is why it is planted often);
    ``rows_wnan`` and ``rows_nanw`` are per-row booleans (reference _row_shape) marking
    the rows given a NaN intensity at a WEIGHTED sample, or a NaN weight -- one row each,
    rarely.
    """
    intensity = intensity.copy()
    weights = weights.copy()
    zw_nan = np.zeros(intensity.shape, dtype=bool)
    rshape = ref._row_shape(intensity.shape)
    rows_wnan = np.zeros(rshape, dtype=bool)
    rows_nanw = np.zeros(rshape, dtype=bool)

    if rng.uniform() < 0.15:
        zero = (weights == 0)
        if zero.any():
            pick = zero & (rng.uniform(size=zero.shape) < 0.3)
            intensity[pick] = np.nan
            zw_nan = pick

    def one_row():
        return tuple(int(rng.integers(s)) for s in rshape)

    if rng.uniform() < 0.05:
        r = one_row()
        rows_i = ref._rows(intensity).reshape(rshape + (-1,))
        rows_w = ref._rows(weights).reshape(rshape + (-1,))
        cand = np.nonzero(rows_w[r] != 0)[0]
        if cand.size:
            rows_i[r + (int(rng.choice(cand)),)] = np.nan
            rows_wnan[r] = True
            intensity = ref._unrows(rows_i.reshape(-1, rows_i.shape[-1]), intensity.shape).astype(np.float32)

    if rng.uniform() < 0.05:
        r = one_row()
        if not rows_wnan[r]:
            rows_w = ref._rows(weights).reshape(rshape + (-1,))
            rows_w[r + (int(rng.integers(rows_w.shape[-1])),)] = np.nan
            rows_nanw[r] = True
            weights = ref._unrows(rows_w.reshape(-1, rows_w.shape[-1]), weights.shape).astype(np.float32)

    return (intensity, weights, zw_nan, rows_wnan, rows_nanw)


def _same(a, b):
    """Elementwise bitwise-style equality, with NaN equal to NaN."""
    return (a == b) | (np.isnan(a) & np.isnan(b))


def _run_gpu(cp, det, intensity, weights):
    """Run one GpuPolynomialDetrender on numpy inputs; returns (intensity, weights) after."""
    gi = cp.asarray(intensity)          # copies: the kernel works in place on both
    gw = cp.asarray(weights)
    det.launch(gi, gw)
    cp.cuda.get_current_stream().synchronize()
    return (cp.asnumpy(gi), cp.asnumpy(gw))


def test_polynomial_detrender(iteration=0, rng=None, verbose=False):
    """One randomized comparison of GpuPolynomialDetrender against ReferencePolynomialDetrender."""

    try:
        import cupy as cp
    except ImportError:
        if verbose:
            atomic_print('    test_polynomial_detrender: cupy not available, skipped')
        return

    from . import GpuPolynomialDetrender

    rng = _default_rng(rng)

    (polydeg, epsilon, nt_chunk, kind, W) = random_config(rng)
    (M, nfreq, nchunk) = random_geometry(rng)
    T = nchunk * nt_chunk
    shape = (M, nfreq, T)
    ref = ReferencePolynomialDetrender(polydeg, epsilon, nt_chunk)
    weights = random_weights(rng, ref, shape, kind)
    noise = 0.0 if (rng.uniform() < 0.1) else 1.0
    intensity = random_intensity(rng, ref, shape, noise)
    (intensity, weights, zw_nan, rows_wnan, rows_nanw) = inject_nans(rng, ref, intensity, weights)
    tag = (f'(polydeg={polydeg}, epsilon={epsilon:.3g}, nt_chunk={nt_chunk}, kind={kind}, W={W},'
           f' M={M}, nfreq={nfreq}, nchunk={nchunk})')

    det = GpuPolynomialDetrender(polydeg, epsilon, nt_chunk, W)
    (gi, gw) = _run_gpu(cp, det, intensity, weights)

    def rows(a):
        return a.reshape(M, nfreq, nchunk, nt_chunk)

    # ---- The gate, as a sandwich (see GATE_BAND).
    (rmin, lmin) = ref.conditioning(weights)                                # (M, nfreq, nchunk)
    (surely_masked, surely_passed, in_band) = gate_bracket(ref, weights)

    gpu_zeroed = (rows(gw) == 0).all(axis=3)
    gpu_w_same = _same(rows(gw), rows(weights)).all(axis=3)
    gpu_i_same = _same(rows(gi), rows(intensity)).all(axis=3)
    gpu_masked = gpu_zeroed & gpu_i_same

    assert (gpu_masked | gpu_w_same).all(), f'a row is neither cleanly masked nor left with its weights {tag}'
    assert gpu_masked[surely_masked].all(), f'a row the gate must fail was not masked {tag}'
    assert gpu_w_same[surely_passed].all(), f'a row the gate must pass had its weights changed {tag}'
    assert (gpu_zeroed[surely_passed] == (rows(weights)[surely_passed] == 0).all(axis=-1)).all(), \
        f'a row the gate must pass was zeroed {tag}'

    # ---- The fit, on rows both sides let through, excluding NaN-poisoned ones.
    model = ref.fit(intensity, weights)
    resid = intensity.astype(np.float64) - model
    compare = surely_passed & ~rows_wnan & ~rows_nanw
    eps32 = np.finfo(np.float32).eps
    err_w = err_a = 0.0
    if compare.any():
        diff = np.abs(rows(gi).astype(np.float64) - rows(resid))                # (M, nfreq, nchunk, n)
        valid = (rows(weights) > 0)
        absi = np.where(valid, np.abs(rows(intensity)), 0.0)
        absi = np.where(np.isnan(absi), 0.0, absi)
        scale = absi.max(axis=3)                                                # per row
        scale = np.where(scale > 0, scale, 1.0)
        bw = (eps32 * scale / np.where(rmin > 0, rmin, 1.0))[..., None]
        ba = (eps32 * scale / np.where(lmin > 0, lmin, 1.0))[..., None]
        sel = compare[..., None] & ~zw_nan.reshape(rows(gi).shape)
        ew = np.where(sel & valid, diff / bw, 0.0)
        ea = np.where(sel, diff / ba, 0.0)
        err_w = float(np.nanmax(ew))
        err_a = float(np.nanmax(ea))
        assert np.isfinite(rows(gi)[sel]).all(), f'non-finite output on a clean row {tag}'
        assert err_w <= TOL_WEIGHTED, f'weighted samples: {err_w:.3g} x eps_mach*scale/rmin {tag}'
        assert err_a <= TOL_ALL, f'all samples: {err_a:.3g} x eps_mach*scale/lmin {tag}'

    # ---- NaN at a weighted sample: if the row passes the gate (which never sees the
    # intensity), the whole row comes out NaN with its weights unchanged, as in the old
    # code; if it fails, it is masked like any other, which the sandwich above covered. A
    # NaN weight masks the row (also implied by the sandwich; asserted here by name).
    sel = rows_wnan & surely_passed
    if sel.any():
        assert np.isnan(rows(gi)[sel]).all() and gpu_w_same[sel].all(), \
            f'NaN at a weighted sample did not give an all-NaN row with weights unchanged {tag}'
    if rows_nanw.any():
        assert gpu_masked[rows_nanw].all(), f'a NaN weight did not mask its row {tag}'

    # ---- Run-to-run bit identity, and warps_per_block as a pure occupancy knob.
    (gi2, gw2) = _run_gpu(cp, det, intensity, weights)
    assert _same(gi2, gi).all() and _same(gw2, gw).all(), f'not reproducible {tag}'
    W2 = WARP_COUNTS[(WARP_COUNTS.index(W) + 1 + int(rng.integers(3))) % len(WARP_COUNTS)]
    det2 = GpuPolynomialDetrender(polydeg, epsilon, nt_chunk, W2)
    (gi2, gw2) = _run_gpu(cp, det2, intensity, weights)
    assert _same(gi2, gi).all() and _same(gw2, gw).all(), f'warps_per_block {W} vs {W2} differ {tag}'

    # ---- Rows are spectators of one another: permute the (beam, channel) rows, and the
    # beam axis is just more rows.
    perm = rng.permutation(M * nfreq)
    flat_i = intensity.reshape(M * nfreq, T)[perm][None]
    flat_w = weights.reshape(M * nfreq, T)[perm][None]
    (gi2, gw2) = _run_gpu(cp, det, np.ascontiguousarray(flat_i), np.ascontiguousarray(flat_w))
    assert _same(gi2[0], gi.reshape(M * nfreq, T)[perm]).all() and \
        _same(gw2[0], gw.reshape(M * nfreq, T)[perm]).all(), f'rows are not independent {tag}'

    # ---- Chunks are independent: the first chunk alone gives the same answer.
    if nchunk > 1:
        (gi2, gw2) = _run_gpu(cp, det, np.ascontiguousarray(intensity[:, :, :nt_chunk]),
                              np.ascontiguousarray(weights[:, :, :nt_chunk]))
        assert _same(gi2, gi[:, :, :nt_chunk]).all() and _same(gw2, gw[:, :, :nt_chunk]).all(), \
            f'chunk 0 alone differs from chunk 0 of the full launch {tag}'

    # ---- NaN at a zero-weight sample changes nothing anywhere else (the deliberate
    # departure from the old code): replace those NaNs by zero and compare everywhere
    # except at those samples themselves.
    if zw_nan.any():
        clean = intensity.copy()
        clean[zw_nan] = 0.0
        (gi2, gw2) = _run_gpu(cp, det, clean, weights)
        assert _same(gi2, gi)[~zw_nan].all() and _same(gw2, gw).all(), \
            f'NaN at a zero-weight sample leaked {tag}'

    # ---- Scaling every weight by a power of two is a bitwise no-op (see the module
    # docstring). 4 rather than 2, so that counts of 16 become 64.
    (gi2, gw2) = _run_gpu(cp, det, intensity, 4.0 * weights)
    assert _same(gi2, gi).all() and _same(gw2, 4.0 * gw).all(), \
        f'weight scaling by 4 changed the output {tag}'

    if verbose:
        atomic_print(f'    test_polynomial_detrender{tag}: {int(surely_masked.sum())} rows masked,'
                     f' {int(in_band.sum())} in band, {int(surely_passed.sum())} passed; err'
                     f' {err_w:.3g} (weighted, /rmin) / {err_a:.3g} (all, /lmin) x eps_mach*scale')
    return (err_w, err_a, int(in_band.sum()))
