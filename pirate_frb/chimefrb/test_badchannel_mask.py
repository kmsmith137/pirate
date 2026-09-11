"""Randomized unit tests for GpuBadChannelMask and badchannel_keep(), and the random draws
they share with misc/chimefrb/rfi_badchannel_mask/.

Dispatched from ``python -m pirate_frb test --cfrb``.

Two independent things are tested. The kernel does no arithmetic, so it is compared with
ReferenceBadChannelMask BITWISE: no tolerance, no bracket. badchannel_keep(), which turns MHz
ranges into channels, is all arithmetic but integer-valued, so it is compared EXACTLY with
keep_by_rule(), a second statement of the same rule written a different way.

WHAT THIS FILE CANNOT ESTABLISH: that badchannel_keep() matches the old code. Both it and
keep_by_rule() come from one reading of rf_pipelines' _bind_transform(), so a misreading
would pass here. That is what misc/chimefrb/rfi_badchannel_mask/ is for.
"""

import numpy as np

from . import GpuBadChannelMask, ReferenceBadChannelMask, badchannel_keep
from .ReferenceBadChannelMask import FUDGE
from ..utils import atomic_print
from .testutils import default_rng as _default_rng


# The production config's mask ranges, in MHz: the badchannel_mask node of
# misc/chimefrb/configs/21-03-07-low-latency-uniform-badchannel-mask-noplot.json (the spot
# check re-reads the file to confirm this copy). Every endpoint is a 16K channel edge.
PRODUCTION_MASK_RANGES = [
    (440.2099609375, 440.4052734375),
    (417.1630859375, 417.626953125),
    (418.310546875, 419.2138671875),
    (781.8115234375, 786.6943359375),
    (627.1728515625, 641.40625),
    (449.9755859375, 450.4150390625),
    (453.2470703125, 453.3203125),
    (730.078125, 755.859375),
]

# What those ranges mask in the production chain, where badchannel_mask runs at 1024 channels
# over 400-800 MHz: 129 channels, as half-open index runs, one per range. Confirmed against
# rf_pipelines::badchannel_mask by the spot check.
PRODUCTION_MASKED_RUNS_1024 = [
    (920, 922), (978, 981), (974, 978), (34, 47),
    (406, 443), (894, 897), (887, 888), (113, 179),
]

WARP_COUNTS = [4, 8, 16, 32]

# See random_range().
RANGE_KINDS = ('inside', 'aligned', 'straddle_bottom', 'straddle_top',
               'touch_bottom', 'touch_top', 'edge_inside', 'grid')


# -------------------------------------------------------------------------------------------------
#
# The kernel.


def random_geometry(rng):
    """Draw (B, F, T).

    The kernel has no divisibility rule, so F and T are drawn freely -- including values that
    are not multiples of 32, T < 32 and F = 1, which no other chimefrb kernel accepts -- with
    the production (F, T) some of the time.
    """

    if rng.uniform() < 0.1:
        return (1, 1024, 4096)

    B = int(rng.integers(1, 5))

    while True:
        F = int(rng.integers(1, 32)) if (rng.uniform() < 0.2) else int(rng.integers(32, 2049))
        T = int(rng.integers(1, 32)) if (rng.uniform() < 0.2) else int(rng.integers(32, 601))
        if rng.uniform() < 0.3:
            F = 32 * max(F // 32, 1)
        if B*F*T <= 2**21:
            return (B, F, T)


def random_keep(rng, F):
    """Draw a bool 'keep' array of length F, and a label for coverage.

    A mixture: all kept (launch() then returns without launching), all masked, channels
    masked independently at a random rate, and a few random contiguous runs -- the shape of
    the production mask.
    """

    x = rng.uniform()

    if x < 0.05:
        return (np.ones(F, dtype=bool), 'all kept')
    if x < 0.10:
        return (np.zeros(F, dtype=bool), 'all masked')
    if x < 0.55:
        return (rng.uniform(size=F) >= rng.uniform(), 'independent')

    keep = np.ones(F, dtype=bool)
    for _ in range(int(rng.integers(1, 9))):
        start = int(rng.integers(F))
        keep[start:start + 1 + int(rng.exponential(F/16))] = False
    return (keep, 'runs')


def random_kernel_case(rng):
    """Draw one kernel test case, as a dict.

    Holds the geometry, the keep array and the form it is passed in, warps_per_block, and
    where to plant non-finite weights. random_weights() makes the array itself, so that
    coverage can draw cases without allocating it.
    """

    (B, F, T) = random_geometry(rng)
    (keep, kind) = random_keep(rng, F)

    # How 'keep' reaches the constructor: as uint8 with arbitrary nonzero values, which the
    # C++ normalizes, sometimes as a strided view, which it must accept; or as bool, which the
    # python __init__ converts.
    if rng.uniform() < 0.5:
        keep_arg = np.where(keep, rng.integers(1, 256, size=F), 0).astype(np.uint8)
        if rng.uniform() < 0.3:
            big = np.zeros(2*F, dtype=np.uint8)
            big[::2] = keep_arg
            keep_arg = big[::2]
    else:
        keep_arg = keep.copy()

    # A few non-finite weights, in kept and masked rows alike: a kept row must come back
    # bit-identical, and a masked one must become +0.0 whatever it held.
    plants = []
    for _ in range(int(rng.integers(0, 4))):
        (b, f, t) = (int(rng.integers(B)), int(rng.integers(F)), int(rng.integers(T)))
        plants.append((b, f, t, float(rng.choice([np.nan, np.inf, -np.inf]))))

    return dict(B=B, F=F, T=T, keep=keep, keep_kind=kind, keep_arg=keep_arg,
                warps=int(rng.choice(WARP_COUNTS)), plants=plants)


def random_weights(rng, case):
    """The (B, F, T) float32 weights for random_kernel_case()'s 'case'.

    Arbitrary signs and a few negative zeros, which real weights never have: the kernel must
    not look at the values at all, and a bitwise comparison is how that gets checked.
    """

    (B, F, T) = (case['B'], case['F'], case['T'])
    w = rng.standard_normal(size=(B, F, T), dtype=np.float32) * np.float32(rng.uniform(0.1, 10.0))

    flat = w.reshape(-1)
    flat[rng.integers(0, flat.size, size=min(8, flat.size))] = -0.0

    for (b, f, t, x) in case['plants']:
        w[b, f, t] = x

    return w


def _check_kernel(cp, rng, verbose):
    """GpuBadChannelMask against ReferenceBadChannelMask, bitwise."""

    c = random_kernel_case(rng)
    w = random_weights(rng, c)

    g = GpuBadChannelMask(c['keep_arg'], c['warps'])
    assert g.F == c['F']
    assert g.nmasked == int((~c['keep']).sum())
    assert np.array_equal(cp.asnumpy(g.keep), c['keep'].astype(np.uint8)), \
        'GpuBadChannelMask.keep is not the normalized keep array'

    w_gpu = cp.asarray(w)
    g.launch(w_gpu)
    got = cp.asnumpy(w_gpu)
    want = ReferenceBadChannelMask(c['keep']).apply(w)

    # Bitwise, through a uint32 view: that is also how NaN compares equal to itself, and how
    # +0.0 is told from -0.0.
    bad = (got.view(np.uint32) != want.view(np.uint32))
    if bad.any():
        (b, f, t) = np.argwhere(bad)[0]
        raise AssertionError(
            f'test_badchannel_mask: GPU and reference differ at {int(bad.sum())} of {bad.size}'
            f' elements; first at (b,f,t) = ({b},{f},{t}), keep[f] = {bool(c["keep"][f])}:'
            f' got {got[b,f,t]!r}, want {want[b,f,t]!r}')

    if verbose:
        atomic_print(f'    test_badchannel_mask: kernel (B,F,T)=({c["B"]},{c["F"]},{c["T"]}),'
                     f' keep={c["keep_kind"]} ({g.nmasked} masked), keep dtype'
                     f' {c["keep_arg"].dtype}, warps_per_block={c["warps"]}: ok')


# -------------------------------------------------------------------------------------------------
#
# badchannel_keep().


def random_nfreq(rng):
    x = rng.uniform()
    if x < 0.3:
        return 1024
    if x < 0.5:
        return 16384
    return int(rng.integers(1, 5001))


def random_band(rng):
    if rng.uniform() < 0.7:
        return (400.0, 800.0)
    lo = float(rng.uniform(100.0, 1000.0))
    return (lo, lo + float(rng.uniform(1.0, 1000.0)))


def random_range(rng, kind, nfreq, freq_lo_MHz, freq_hi_MHz):
    """One (lo, hi) MHz pair of the given kind. Every kind is one that badchannel_keep()
    accepts; the refused ones are tested separately.

      'inside'           in the band, anywhere from 1/100 of a channel to 1/4 of the band wide
      'aligned'          both ends on channel edges (up to roundoff), which is what FUDGE is for
      'straddle_bottom'  crosses the bottom of the band, and is clipped to it
      'straddle_top'     crosses the top of the band
      'touch_bottom'     ends exactly at the bottom: clips to zero width, masks channel nfreq-1
      'touch_top'        starts exactly at the top: clips to zero width, masks nothing
      'edge_inside'      inside, with one end exactly on a band edge
      'grid'             both ends on 16K channel edges, where the production ranges sit
    """

    (flo, fhi) = (freq_lo_MHz, freq_hi_MHz)
    bw = fhi - flo
    chan = bw / nfreq

    if kind == 'inside':
        w = float(np.exp(rng.uniform(np.log(0.01*chan), np.log(max(0.25*bw, 0.02*chan)))))
        lo = float(rng.uniform(flo, fhi - w))
        return (lo, lo + w)
    if kind == 'aligned':
        (k1, k2) = sorted(int(k) for k in rng.choice(nfreq + 1, size=2, replace=False))
        return (fhi - k2*chan, fhi - k1*chan)
    if kind == 'straddle_bottom':
        return (flo - float(rng.uniform(0.001, 0.3))*bw, flo + float(rng.uniform(0.001, 0.3))*bw)
    if kind == 'straddle_top':
        return (fhi - float(rng.uniform(0.001, 0.3))*bw, fhi + float(rng.uniform(0.001, 0.3))*bw)
    if kind == 'touch_bottom':
        return (flo - float(rng.uniform(0.001, 0.3))*bw, flo)
    if kind == 'touch_top':
        return (fhi, fhi + float(rng.uniform(0.001, 0.3))*bw)
    if kind == 'edge_inside':
        w = float(rng.uniform(0.001, 0.3))*bw
        return (flo, flo + w) if (rng.uniform() < 0.5) else (fhi - w, fhi)
    if kind == 'grid':
        g = bw / 16384
        (k1, k2) = sorted(int(k) for k in rng.choice(16385, size=2, replace=False))
        return (flo + k1*g, flo + k2*g)

    raise ValueError(f'random_range: unknown kind {kind!r}')


def random_range_case(rng, nmax=10):
    """Draw (mask_ranges, kinds, nfreq, freq_lo_MHz, freq_hi_MHz), with 0 to nmax ranges."""

    nfreq = random_nfreq(rng)
    (flo, fhi) = random_band(rng)
    kinds = [RANGE_KINDS[int(rng.integers(len(RANGE_KINDS)))]
             for _ in range(int(rng.integers(0, nmax + 1)))]
    ranges = [random_range(rng, k, nfreq, flo, fhi) for k in kinds]
    return (ranges, kinds, nfreq, flo, fhi)


def keep_by_rule(mask_ranges, nfreq, freq_lo_MHz, freq_hi_MHz, bottom_clamp=True):
    """badchannel_keep(), restated per channel: channel i is masked by a range iff

        u(lo) > i + FUDGE   and   (u(hi) + FUDGE < i + 1   or   i == nfreq - 1)

    with the range clipped to the band by max/min, and u(x) the channel coordinate of
    frequency x. A second formulation of the same rule -- per channel rather than per range,
    comparisons rather than floor() and ceil(), max/min rather than three branches -- so that
    a slip in either is caught.

    Valid only for ranges that badchannel_keep() accepts (random_range_case() draws no others),
    and only away from the roundoff boundaries that near_fudge_boundary() flags.
    'bottom_clamp=False' drops the 'i == nfreq - 1' clause, which lets coverage measure how
    often that quirk decides the answer.
    """

    (flo, fhi) = (float(freq_lo_MHz), float(freq_hi_MHz))
    scale = nfreq / (fhi - flo)
    factor = scale * fhi

    i = np.arange(nfreq, dtype=np.float64)
    last = (i == nfreq - 1) if bottom_clamp else np.zeros(nfreq, dtype=bool)
    masked = np.zeros(nfreq, dtype=bool)

    for (lo, hi) in mask_ranges:
        (lo, hi) = (max(lo, flo), min(hi, fhi))
        (u_lo, u_hi) = (factor - lo*scale, factor - hi*scale)
        masked |= (u_lo > i + FUDGE) & ((u_hi + FUDGE < i + 1) | last)

    return (~masked).astype(np.uint8)


def near_fudge_boundary(mask_ranges, nfreq, freq_lo_MHz, freq_hi_MHz, tol=1.0e-9):
    """True if some range end lands within 'tol' channels of an integer +/- FUDGE.

    There, badchannel_keep()'s floor()/ceil() and keep_by_rule()'s comparisons round
    differently, and so may the old code, which is built with -ffast-math and may evaluate
    'factor - x*scale' as a fused multiply-add. The disagreement would be roundoff, not a bug,
    so such draws are skipped -- as the clippers' tests never put an outlier at the threshold.
    It takes an end within about 1e-13 channels to matter, so 1e-9 is generous, and a draw
    this flags should essentially never happen (coverage reports the rate).
    """

    (flo, fhi) = (float(freq_lo_MHz), float(freq_hi_MHz))
    scale = nfreq / (fhi - flo)
    factor = scale * fhi

    for (lo, hi) in mask_ranges:
        (lo, hi) = (max(lo, flo), min(hi, fhi))
        for v in ((factor - hi*scale) + FUDGE, (factor - lo*scale) - FUDGE):
            if abs(v - round(v)) < tol:
                return True

    return False


def _check_keep(rng, verbose):
    """badchannel_keep() against keep_by_rule(), exactly."""

    (ranges, kinds, nfreq, flo, fhi) = random_range_case(rng)

    if near_fudge_boundary(ranges, nfreq, flo, fhi):
        atomic_print('    test_badchannel_mask: a range end within 1e-9 of a fudge boundary, skipped')
        return

    got = badchannel_keep(ranges, nfreq, flo, fhi)
    want = keep_by_rule(ranges, nfreq, flo, fhi)

    assert (got.dtype == np.uint8) and (got.shape == (nfreq,))
    if not np.array_equal(got, want):
        f = int(np.flatnonzero(got != want)[0])
        raise AssertionError(
            f'test_badchannel_mask: badchannel_keep() and keep_by_rule() disagree at'
            f' {int((got != want).sum())} channel(s), first at {f} (got keep={got[f]}); nfreq={nfreq},'
            f' band=({flo!r}, {fhi!r}), ranges={ranges!r}, kinds={kinds}')

    if verbose:
        atomic_print(f'    test_badchannel_mask: badchannel_keep nfreq={nfreq},'
                     f' band=({flo:.6g}, {fhi:.6g}), {len(ranges)} range(s): ok')


# -------------------------------------------------------------------------------------------------
#
# Deterministic checks, run on iteration 0 only.


def _expect_raise(exc, f, *args, **kwargs):
    try:
        f(*args, **kwargs)
    except exc:
        return
    raise AssertionError(f'test_badchannel_mask: expected {exc.__name__} from {f.__name__}{args!r}')


def _check_production():
    """The production mask, pinned, and the documented quirks, as named cases."""

    expected = np.ones(1024, dtype=np.uint8)
    for (a, b) in PRODUCTION_MASKED_RUNS_1024:
        expected[a:b] = 0

    keep = badchannel_keep(PRODUCTION_MASK_RANGES, 1024, 400.0, 800.0)
    assert np.array_equal(keep, expected), 'the production mask at 1024 channels has changed'
    assert int((keep == 0).sum()) == 129

    g = GpuBadChannelMask.from_mask_ranges(PRODUCTION_MASK_RANGES, 1024, 400.0, 800.0)
    assert (g.F == 1024) and (g.nmasked == 129)

    def masked(ranges):
        return list(np.flatnonzero(badchannel_keep(ranges, 1024, 400.0, 800.0) == 0))

    assert masked([(390.0, 400.0)]) == [1023], 'a range ending at the bottom of the band'
    assert masked([(800.0, 810.0)]) == [], 'a range starting at the top of the band'
    assert masked([(400.0, 800.0)]) == list(range(1024)), 'the whole band, written as "inside"'
    assert masked([(800.0 - 7*400.0/1024, 800.0)]) == list(range(7)), 'a channel-aligned range'


def _check_raises():
    """The inputs the old code refuses, plus the two it does not check."""

    for ranges in ([(500.0, 500.0)], [(501.0, 500.0)], [(300.0, 350.0)], [(850.0, 900.0)],
                   [(390.0, 810.0)], [(410.0, 420.0), (300.0, 350.0)], [(410.0,)]):
        _expect_raise(ValueError, badchannel_keep, ranges, 1024, 400.0, 800.0)

    _expect_raise(ValueError, badchannel_keep, [], 0, 400.0, 800.0)
    _expect_raise(ValueError, badchannel_keep, [], 1024, 800.0, 400.0)


def _check_arguments(cp):
    keep = np.array([True, False, True, True])

    _expect_raise(ValueError, GpuBadChannelMask, np.ones((2, 3), dtype=bool))
    _expect_raise(TypeError, GpuBadChannelMask, np.ones(4, dtype=np.float32))
    _expect_raise(TypeError, GpuBadChannelMask, cp.ones(4, dtype=bool))
    _expect_raise(RuntimeError, GpuBadChannelMask, np.ones(0, dtype=bool))
    _expect_raise(RuntimeError, GpuBadChannelMask, keep, 5)

    g = GpuBadChannelMask(keep)
    _expect_raise(RuntimeError, g.launch, cp.zeros((2, 5, 8), dtype=cp.float32))    # wrong F
    _expect_raise(RuntimeError, g.launch, cp.zeros((4, 8), dtype=cp.float32))       # 2-d
    _expect_raise(RuntimeError, g.launch, cp.zeros((2, 4, 16), dtype=cp.float32)[:, :, ::2])
    _expect_raise(RuntimeError, g.launch, cp.zeros((0, 4, 8), dtype=cp.float32))    # empty


# -------------------------------------------------------------------------------------------------


def test_badchannel_mask(iteration=0, rng=None, verbose=False):
    """One randomized kernel comparison and one randomized badchannel_keep() comparison; on
    iteration 0, also the production mask, the refused inputs and the argument checks."""

    try:
        import cupy as cp
    except ImportError:
        if verbose:
            atomic_print('    test_badchannel_mask: cupy not available, skipped')
        return

    rng = _default_rng(rng)

    _check_kernel(cp, rng, verbose)
    _check_keep(rng, verbose)

    if iteration == 0:
        _check_production()
        _check_raises()
        _check_arguments(cp)
