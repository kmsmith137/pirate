"""Randomized unit tests for GpuWiDownsampler.

Dispatched from ``python -m pirate_frb test --cfrb``.

The GPU kernel is checked against ReferenceWiDownsampler, a numpy transcription of the
old code's own scalar reference. There are no sharp thresholds anywhere in this kernel --
the only branch is ``out_w > 0``, and a sum of nonnegative floats is zero if and only if
every term is zero -- so this is a plain float32-tolerance comparison, with none of the
threshold-bracketing machinery the clippers will need.

WHAT THIS FILE CANNOT ESTABLISH: the numpy reference is a transcription, so a misreading
of the old kernel's semantics would pass every test below. The weight normalization is
the trap (sum over the cell, not mean). That is what a spot check under misc/chimefrb/
is for; see notes/chimefrb.md.
"""

import numpy as np

from . import GpuWiDownsampler, ReferenceWiDownsampler
from ..utils import atomic_print


# The (Df, Dt, transpose) configurations used by the old search's production RFI config.
# Drawn most of the time, so that a run spends its iterations where the port will.
PRODUCTION_CONFIGS = [(16, 1, False), (1, 1, True), (2, 16, True), (2, 16, False)]

WARP_COUNTS = [4, 8, 16, 32]


def random_config(rng):
    """Draw (Df, Dt, transpose). Mostly production values, sometimes an off-menu pair.

    Off-menu draws are worth having because nothing restricts (Df, Dt) to a compiled
    list -- they are runtime kernel arguments -- so an odd factor or a large Dt is a
    legal call that no production config would ever make.
    """

    if rng.uniform() < 0.7:
        return PRODUCTION_CONFIGS[rng.integers(len(PRODUCTION_CONFIGS))]

    Df = int(rng.integers(1, 6))
    Dt = int(rng.integers(1, 6))
    transpose = bool(rng.integers(2))

    if (Df == 1) and (Dt == 1) and not transpose:
        transpose = True   # the identity, which the constructor rejects

    return (Df, Dt, transpose)


def random_geometry(rng, Df, Dt):
    """Draw (B, F, T).

    The kernel has no edge predication, so F and T must be multiples of 32*Df and 32*Dt.
    Drawing the tile counts rather than F and T makes that hold by construction, instead
    of by rejection.
    """

    B = int(rng.integers(1, 4))
    F = 32 * Df * int(rng.integers(1, 5))
    T = 32 * Dt * int(rng.integers(1, 5))
    return (B, F, T)


def random_arrays(rng, B, F, T):
    """Random (intensity, weights), designed to reach the out_w <= 0 branch.

    'p' is the probability that a sample is unmasked, and is drawn so that it is
    sometimes exactly 0 or 1: at low p, whole (Df,Dt) cells come out fully masked,
    which is the only way to exercise the guarded divide. The intensity carries a
    large random offset so that the wisum/wsum division is tested where cancellation
    matters, not only near zero.
    """

    p = np.clip(rng.uniform(-0.1, 1.1), 0.0, 1.0)
    w = (rng.uniform(size=(B, F, T)) < p).astype(np.float32)

    # Make the weights genuinely non-binary on part of the array: the downsampled
    # weights that the clippers consume look like this, not like a 0/1 mask.
    scale = rng.uniform(0.5, 1.5, size=(B, F, T)).astype(np.float32)
    w *= np.where(rng.uniform(size=(B, F, T)) < 0.5, scale, np.float32(1))

    offset = np.float32(rng.uniform(-1.0e3, 1.0e3))
    i = (rng.normal(size=(B, F, T)) + offset).astype(np.float32)

    return (i, w)


def _run_gpu(cp, ds, in_i, in_w):
    """Run one GpuWiDownsampler on numpy inputs; returns numpy (out_i, out_w).

    The output arrays are pre-filled with NaN, and the caller checks that none
    survives: that is what catches a grid-dimension error which silently skips a tile.
    """

    (B, F, T) = in_i.shape
    (F_ds, T_ds) = (F // ds.Df, T // ds.Dt)
    oshape = (B, T_ds, F_ds) if ds.transpose else (B, F_ds, T_ds)

    g_out_i = cp.full(oshape, cp.nan, dtype=cp.float32)
    g_out_w = cp.full(oshape, cp.nan, dtype=cp.float32)

    ds.launch(g_out_i, g_out_w, cp.asarray(in_i), cp.asarray(in_w))
    cp.cuda.get_current_stream().synchronize()

    return (cp.asnumpy(g_out_i), cp.asnumpy(g_out_w))


def test_wi_downsampler(iteration=0, rng=None, verbose=False):
    """One randomized comparison of GpuWiDownsampler against ReferenceWiDownsampler."""

    try:
        import cupy as cp
    except ImportError:
        if verbose:
            atomic_print('    test_wi_downsampler: cupy not available, skipped')
        return

    rng = np.random.default_rng() if (rng is None) else rng

    (Df, Dt, transpose) = random_config(rng)
    W = WARP_COUNTS[rng.integers(len(WARP_COUNTS))]

    (B, F, T) = random_geometry(rng, Df, Dt)
    (in_i, in_w) = random_arrays(rng, B, F, T)

    ds = GpuWiDownsampler(Df, Dt, transpose, W)
    (gpu_i, gpu_w) = _run_gpu(cp, ds, in_i, in_w)

    assert not np.isnan(gpu_i).any(), 'GpuWiDownsampler left part of out_i unwritten'
    assert not np.isnan(gpu_w).any(), 'GpuWiDownsampler left part of out_w unwritten'

    (ref_i, ref_w) = ReferenceWiDownsampler(Df, Dt, transpose).apply(in_i, in_w)

    # Tolerance: the two sum a (Df,Dt) block in different orders, and the reference sums
    # in float64. The relative error is at the float32 roundoff level times sqrt(Df*Dt),
    # so 1e-5 has several orders of margin at the block sizes we use, and is still tight
    # enough that a real indexing error cannot slip through.
    scale = max(float(np.max(np.abs(ref_i))), 1.0)
    err_i = float(np.max(np.abs(gpu_i - ref_i))) / scale
    err_w = float(np.max(np.abs(gpu_w - ref_w))) / max(float(np.max(ref_w)), 1.0)

    assert err_i < 1.0e-5, f'out_i: max rel err {err_i:.3e} (Df={Df} Dt={Dt} transpose={transpose} W={W})'
    assert err_w < 1.0e-5, f'out_w: max rel err {err_w:.3e} (Df={Df} Dt={Dt} transpose={transpose} W={W})'

    # Structural check 1: warps_per_block is a performance knob and must not change the
    # answer. Compared bitwise -- every warps_per_block sums each cell in the same order.
    for W2 in WARP_COUNTS:
        if W2 == W:
            continue
        (alt_i, alt_w) = _run_gpu(cp, GpuWiDownsampler(Df, Dt, transpose, W2), in_i, in_w)
        assert np.array_equal(alt_i, gpu_i), f'out_i differs between warps_per_block {W} and {W2}'
        assert np.array_equal(alt_w, gpu_w), f'out_w differs between warps_per_block {W} and {W2}'

    # Structural check 2: transposing the output is a layout choice, not an arithmetic
    # one. The two orientations run the same reduction in the same order, so this is
    # bitwise, and it pins down the transpose independently of the reference -- which
    # matters because the transpose is the one feature the old kernel does not have, and
    # so is the one thing a spot check against the old code cannot cover.
    if (Df, Dt) != (1, 1):
        (flip_i, flip_w) = _run_gpu(cp, GpuWiDownsampler(Df, Dt, not transpose, W), in_i, in_w)
        assert np.array_equal(np.swapaxes(flip_i, 1, 2), gpu_i), 'transpose changes out_i'
        assert np.array_equal(np.swapaxes(flip_w, 1, 2), gpu_w), 'transpose changes out_w'

    # Structural check 3: the beam axis is a pure spectator. This axis is ours, not the
    # old code's -- chimefrb ran one beam per pipeline -- so nothing else checks it.
    if B > 1:
        for b in range(B):
            (one_i, one_w) = _run_gpu(cp, ds, in_i[b:b+1], in_w[b:b+1])
            assert np.array_equal(one_i[0], gpu_i[b]), f'beam {b} of out_i is not a spectator'
            assert np.array_equal(one_w[0], gpu_w[b]), f'beam {b} of out_w is not a spectator'

    if verbose:
        atomic_print(f'    test_wi_downsampler(Df={Df}, Dt={Dt}, transpose={transpose},'
                     f' warps_per_block={W}, B={B}, F={F}, T={T}):'
                     f' max rel err {max(err_i, err_w):.3e}')
