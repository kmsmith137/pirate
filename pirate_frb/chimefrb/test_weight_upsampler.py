"""Randomized unit tests for GpuWeightUpsampler.

Dispatched from ``python -m pirate_frb test --cfrb``.

The kernel does no arithmetic -- one float32 comparison per cell, and a store -- so it is
compared with ReferenceWeightUpsampler BITWISE, with no tolerance and no bracket. What can go
wrong is the cell geometry (which full-resolution block a cell owns) and the comparison
itself (strict, in float32, false for NaN), and the draws are built to reach both. Because
the comparison is bitwise over the whole array, it subsumes the usual determinism,
beam-spectator and cell-uniformity checks, which are therefore not written separately.

A second check composes three built classes: an intensity clipper at (Df, Dt) must equal
"downsample, clip at (1,1), upsample the mask", BITWISE. That identity is the old code's own
(rf_pipelines' test-cpp-python-equivalence.py argues correctness with it), and it is the only
check that ties this kernel's cell geometry to the clippers'.

WHAT THIS FILE CANNOT ESTABLISH: that the numpy reference matches the old code. Both pirate
implementations were written from one reading of it, so a misreading would pass here. That is
what misc/chimefrb/rfi_weight_upsample/ is for.
"""

import numpy as np

from . import (GpuIntensityClipper, GpuWeightUpsampler, GpuWiDownsampler,
               ReferenceWeightUpsampler)
from . import test_intensity_clipper as ict
from ..utils import atomic_print
from .testutils import default_rng as _default_rng


WARP_COUNTS = [4, 8, 16, 32]

# The chain's one instance: the wi_sub_pipeline upsample, 16384 channels from 1024, at native
# time resolution, keeping a cell iff its weight is positive.
PRODUCTION_CONFIG = (16, 1)

# Cutoffs that float32 cannot represent exactly. The comparison is made in float32 (the old
# code takes a float), so a weight equal to float32(0.1) masks where a float64 comparison
# would keep it -- these draws are what would catch a reference that skipped the cast.
INEXACT_CUTOFFS = [0.1, 0.3, 0.7]

# The corner cases planted in the low-resolution weights; see random_lores().
LORES_KINDS = ('at the cutoff', 'negative zero', 'negative', 'NaN', 'inf',
               'one ulp below the cutoff', 'one ulp above the cutoff')


def random_config(rng):
    """Draw (Df, Dt, w_cutoff, warps_per_block)."""

    if rng.uniform() < 0.33:
        (Df, Dt) = PRODUCTION_CONFIG
    else:
        # Nothing restricts (Df, Dt) to powers of two -- they are runtime kernel arguments.
        Df = int(rng.choice([1, 2, 3, 4, 5, 8, 16]))
        Dt = int(rng.choice([1, 1, 2, 3, 4, 8, 16]))

    x = rng.uniform()
    if x < 0.6:
        w_cutoff = 0.0                                   # production
    elif x < 0.8:
        w_cutoff = float(rng.choice(INEXACT_CUTOFFS))
    else:
        w_cutoff = float(rng.uniform(0.25, 4.0))

    return (Df, Dt, w_cutoff, int(rng.choice(WARP_COUNTS)))


def random_geometry(rng, Df, Dt):
    """Draw (B, F_lo, T_lo), with the full-resolution array kept small.

    T_lo is often not a multiple of 32, and sometimes smaller than 32. This kernel has no
    divisibility rule -- the row loop's own bound is its tail predicate -- unlike every other
    chimefrb kernel except GpuBadChannelMask's, and these draws are what keeps that true.
    """

    B = int(rng.integers(1, 5))
    F_lo = 1 if (rng.uniform() < 0.1) else int(rng.integers(1, 65))
    T_lo = int(rng.integers(1, 32)) if (rng.uniform() < 0.25) else int(rng.integers(32, 257))

    # Shrink the time axis if the full-resolution array would be large. Done on T_lo rather
    # than by redrawing, so that the "not a multiple of 32" property survives.
    tmax = max(1, 2**21 // (B * F_lo * Df * Dt))
    return (B, F_lo, min(T_lo, tmax))


def random_lores(rng, B, F_lo, T_lo, Df, Dt, w_cutoff):
    """Low-resolution weights (B, F_lo, T_lo), and the corner-case kinds planted in them.

    Mostly what the chain produces: integer counts 0..Df*Dt, with a cell having any weight
    with probability 'p'. 'p' is drawn so that it is sometimes exactly 0 or 1, which is how
    the "everything masked" and "nothing masked" cases occur a few percent of the time
    (notes/unit_tests.md item 6). Those two draws are left clean -- the corner cases below are
    planted only in the mixed ones -- so that each case stays reachable.
    """

    p = float(np.clip(rng.uniform(-0.1, 1.1), 0.0, 1.0))
    ncell = Df * Dt

    w = rng.integers(1, ncell + 1, size=(B, F_lo, T_lo)).astype(np.float32)
    w *= (rng.uniform(size=(B, F_lo, T_lo)) < p)

    if (p <= 0.0) or (p >= 1.0):
        return (w, ())

    c = np.float32(w_cutoff)
    specials = [('at the cutoff', c),
                ('negative zero', np.float32(-0.0)),
                ('negative', np.float32(-2.0)),
                ('NaN', np.float32(np.nan)),
                ('inf', np.float32(np.inf))]

    # One ulp either side of the cutoff, which pins the comparison down to the last bit. Only
    # for a positive cutoff: one ulp below zero is denormal, and denormals are out of scope
    # (they may compare as zero on the GPU, which is built with --use_fast_math).
    if w_cutoff > 0.0:
        specials += [('one ulp below the cutoff', np.nextafter(c, np.float32(0.0))),
                     ('one ulp above the cutoff', np.nextafter(c, np.float32(np.inf)))]

    flat = w.reshape(-1)
    kinds = []

    for (label, x) in specials:
        if rng.uniform() < 0.5:
            n = max(1, flat.size // 50)
            flat[rng.integers(0, flat.size, size=n)] = x
            kinds.append(label)

    return (w, tuple(kinds))


def random_hires(rng, B, F_hi, T_hi):
    """Full-resolution weights, with non-finite values and -0.0 planted throughout.

    The kernel must never read these: a kept weight has to come back with its exact bits, and
    a masked one has to become +0.0 whatever it held.
    """

    w = rng.uniform(0.5, 1.5, size=(B, F_hi, T_hi)).astype(np.float32)
    w *= (rng.uniform(size=(B, F_hi, T_hi)) < 0.9)

    flat = w.reshape(-1)
    for x in (np.nan, np.inf, -np.inf, -0.0):
        flat[rng.integers(0, flat.size, size=max(1, flat.size // 200))] = np.float32(x)

    return w


def _check_kernel(cp, rng, verbose):
    """GpuWeightUpsampler against ReferenceWeightUpsampler, bitwise."""

    (Df, Dt, w_cutoff, warps) = random_config(rng)
    (B, F_lo, T_lo) = random_geometry(rng, Df, Dt)
    (lo, kinds) = random_lores(rng, B, F_lo, T_lo, Df, Dt, w_cutoff)
    hi = random_hires(rng, B, F_lo*Df, T_lo*Dt)

    ups = GpuWeightUpsampler(Df, Dt, w_cutoff, warps)
    assert (ups.Df, ups.Dt, ups.warps_per_block) == (Df, Dt, warps)
    assert ups.w_cutoff == w_cutoff

    g_hi = cp.asarray(hi)
    ups.launch(g_hi, cp.asarray(lo))
    cp.cuda.get_current_stream().synchronize()

    got = cp.asnumpy(g_hi)
    want = ReferenceWeightUpsampler(Df, Dt, w_cutoff).apply(hi, lo)

    # Bitwise, through a uint32 view: that is also how NaN compares equal to itself, and how
    # +0.0 is told from -0.0.
    bad = (got.view(np.uint32) != want.view(np.uint32))
    if bad.any():
        (b, f, t) = np.argwhere(bad)[0]
        raise AssertionError(
            f'test_weight_upsampler: GPU and reference differ at {int(bad.sum())} of'
            f' {bad.size} weights; first at (b,f,t) = ({b},{f},{t}), with'
            f' w_lores = {lo[b, f//Df, t//Dt]!r} and w_cutoff = {w_cutoff!r}:'
            f' got {got[b,f,t]!r}, want {want[b,f,t]!r}')

    if verbose:
        nmasked = int(np.sum(~(lo > np.float32(w_cutoff))))
        atomic_print(f'    test_weight_upsampler: (Df,Dt)=({Df},{Dt}),'
                     f' (B,F_lo,T_lo)=({B},{F_lo},{T_lo}), w_cutoff={w_cutoff},'
                     f' warps_per_block={warps}, {nmasked} of {lo.size} cells masked,'
                     f' planted {kinds}: ok')


def _check_composition(cp, rng, verbose):
    """A clipper at (Df,Dt) equals downsample, clip at (1,1), upsample the mask -- bitwise.

    The two paths run the same kernels with the same parameters on the same values, so they
    make the same decisions: the (Df,Dt) clipper downsamples with GpuWiDownsampler and then
    thresholds, and the composed path hands that same downsampled pair to a (1,1) clipper.
    Where the first zeroes a cell's Df-by-Dt block of weights, the second zeroes the cell's
    low-resolution weight and this class zeroes the block. The composed path also zeroes the
    blocks of cells that had no weight to begin with, but those blocks are already +0.0.
    """

    while True:
        (axis, Df, Dt, niter, sigma, iter_sigma, two_pass, _) = ict.random_config(rng)
        if (Df, Dt) != (1, 1):
            break

    (B, F, T) = ict.random_geometry(rng, axis, Df, Dt)
    (I, W) = ict.random_arrays(rng, B, F, T)
    (F_ds, T_ds) = (F // Df, T // Dt)

    # The transform itself.
    g_w = cp.asarray(W)
    GpuIntensityClipper(B, F, T, T, axis, sigma, Df, Dt, niter, iter_sigma,
                        two_pass).launch(cp.asarray(I), g_w, None)
    cp.cuda.get_current_stream().synchronize()
    direct = cp.asnumpy(g_w)

    # The same thing, composed out of three classes.
    ds_i = cp.empty((B, F_ds, T_ds), dtype=cp.float32)
    ds_w = cp.empty((B, F_ds, T_ds), dtype=cp.float32)
    GpuWiDownsampler(Df, Dt, False).launch(ds_i, ds_w, cp.asarray(I), cp.asarray(W))

    GpuIntensityClipper(B, F_ds, T_ds, T_ds, axis, sigma, 1, 1, niter, iter_sigma,
                        two_pass).launch(ds_i, ds_w, None)

    g_w2 = cp.asarray(W)
    GpuWeightUpsampler(Df, Dt, 0.0).launch(g_w2, ds_w)
    cp.cuda.get_current_stream().synchronize()
    composed = cp.asnumpy(g_w2)

    bad = (direct.view(np.uint32) != composed.view(np.uint32))
    if bad.any():
        (b, f, t) = np.argwhere(bad)[0]
        raise AssertionError(
            f'test_weight_upsampler: the clipper at (Df,Dt)=({Df},{Dt}) and its'
            f' downsample/clip/upsample composition differ at {int(bad.sum())} of {bad.size}'
            f' weights; first at (b,f,t) = ({b},{f},{t}): {direct[b,f,t]!r} vs'
            f' {composed[b,f,t]!r}')

    if verbose:
        atomic_print(f'    test_weight_upsampler: downsampling reduction, axis={axis},'
                     f' (Df,Dt)=({Df},{Dt}), (B,F,T)=({B},{F},{T}), niter={niter}: ok')


def _check_arguments(cp):
    """Every argument error raises. Runs on iteration 0 only."""

    def expect_raise(label, f):
        try:
            f()
        except (RuntimeError, ValueError, TypeError):
            return
        raise AssertionError(f'test_weight_upsampler: expected an exception: {label}')

    expect_raise('Df = 0', lambda: GpuWeightUpsampler(0, 1))
    expect_raise('Dt = 0', lambda: GpuWeightUpsampler(1, 0))
    expect_raise('w_cutoff < 0', lambda: GpuWeightUpsampler(1, 1, -1.0))
    expect_raise('w_cutoff = NaN', lambda: GpuWeightUpsampler(1, 1, float('nan')))
    expect_raise('warps_per_block = 5', lambda: GpuWeightUpsampler(1, 1, 0.0, 5))

    ups = GpuWeightUpsampler(2, 2)
    lo = cp.ones((1, 4, 4), dtype=cp.float32)
    hi = cp.ones((1, 8, 8), dtype=cp.float32)

    expect_raise('w_hires has the wrong shape',
                 lambda: ups.launch(cp.ones((1, 8, 4), dtype=cp.float32), lo))
    expect_raise('w_lores is 2-d', lambda: ups.launch(hi, cp.ones((4, 4), dtype=cp.float32)))
    expect_raise('host arrays', lambda: ups.launch(np.ones((1, 8, 8), dtype=np.float32),
                                                   np.ones((1, 4, 4), dtype=np.float32)))
    expect_raise('w_hires is w_lores', lambda: GpuWeightUpsampler(1, 1).launch(lo, lo))
    expect_raise('w_hires not contiguous',
                 lambda: ups.launch(cp.ones((1, 8, 16), dtype=cp.float32)[:, :, ::2], lo))


def test_weight_upsampler(iteration=0, rng=None, verbose=False):
    import cupy as cp

    rng = _default_rng(rng)

    if iteration == 0:
        _check_arguments(cp)

    _check_kernel(cp, rng, verbose)

    # The composition check runs a whole intensity clipper twice, so it is not run every
    # iteration; a third of the time is plenty to keep it exercised.
    if rng.uniform() < 0.34:
        _check_composition(cp, rng, verbose)
