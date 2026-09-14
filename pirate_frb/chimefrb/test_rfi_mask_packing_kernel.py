"""Randomized unit tests for RfiMaskPackingKernel.

Dispatched from ``python -m pirate_frb test --cfrb``.

The oracle is ``np.packbits(w > 0, axis=1, bitorder='little')``, and the comparison is EXACT
-- a bit layout is a convention, not a computation, so any disagreement at all means the
convention is wrong. That numpy expression is not merely plausible:
misc/chimefrb/spot_checks/rfi_mask_bitorder runs the real rf_kernels packer, the code that
wrote every mask in every data file, and asserts the two agree.

The draws are built from RUNS of good and bad samples rather than from noise alone. Against
noise an off-by-one in the bit or the byte index is invisible -- both orders give an
indistinguishable-looking pattern -- and against runs it is obvious. The same reasoning is
spelled out in the spot check.

DENORMALS ARE EXCLUDED, and that is the one real gap. pirate builds with --use_fast_math, so
the GPU comparison flushes a positive denormal weight (below 1.18e-38) to zero and calls the
sample bad, where the CPU calls it good. Measured, not assumed: 5.9e-39 and 1.4e-45 disagree,
while NaN, +/-0.0, +/-inf and FLT_MIN itself all agree. Chain weights are 0, 1, or averages
of those, so a denormal cannot occur in production.
"""

import numpy as np

from . import RfiMaskPackingKernel
from ..utils import atomic_print
from .test_assembled_chunk import _expect_raise
from .testutils import default_rng as _default_rng


WARP_COUNTS = [4, 8, 16, 32]

# Everything except denormals (see the module docstring). FLT_MIN is the smallest value that
# must still count as good, so it is the edge of the supported range and belongs here.
SPECIAL_VALUES = np.array([0.0, -0.0, np.nan, np.inf, -np.inf, 1.0, -1.0, 1e-30, -1e-30,
                           np.finfo(np.float32).min, np.finfo(np.float32).max,
                           np.finfo(np.float32).tiny, -np.finfo(np.float32).tiny],
                          dtype=np.float32)


def make_weights(rng, nfreq, nt):
    """An (nfreq, nt) float32 weights array, in runs, with special values scattered in it.

    Three ingredients, and each is load-bearing:
      - RUNS of zero and nonzero, so that an off-by-one in the bit or byte index shows up
        (see the module docstring);
      - continuous noise on the nonzero samples, so the kernel is not only ever asked about
        exact ones;
      - SPECIAL_VALUES at random positions, which is what pins '>' against '>=' (the exact
        zeros), and NaN and the infinities.
    """

    w = np.zeros((nfreq, nt), dtype=np.float32)

    for i in range(nfreq):
        t, good = 0, bool(rng.integers(2))
        while t < nt:
            n = int(rng.integers(1, 40))
            if good:
                w[i, t:t+n] = rng.uniform(0.25, 2.0, size=len(w[i, t:t+n]))
            t += n
            good = not good

    if nfreq >= 2:
        w[0, :] = 1.0      # a fully unmasked row
        w[1, :] = 0.0      # a fully masked row

    # A few percent of the samples, so that a special value lands in every byte position.
    nspecial = max(1, (nfreq * nt) // 32)
    fs = rng.integers(0, nfreq, size=nspecial)
    ts = rng.integers(0, nt, size=nspecial)
    w[fs, ts] = SPECIAL_VALUES[rng.integers(0, len(SPECIAL_VALUES), size=nspecial)]

    return w


def reference_mask(w):
    """The oracle: (nfreq, nt/8) uint8, bit-packed LSB-first, a set bit meaning good."""

    return np.packbits(w > 0, axis=1, bitorder='little')


def _tag(nfreq, nt, warps, pad, off):
    return f'(nfreq,nt)=({nfreq},{nt}) warps={warps} pad={pad} off={off}'


def _compare(got, want, tag):
    bad = (got != want)
    if bad.any():
        (f, j) = np.argwhere(bad)[0]
        raise AssertionError(
            f'test_rfi_mask_packing_kernel: GPU and numpy masks differ at'
            f' {int(bad.sum())} of {bad.size} bytes; first at (f,byte) = ({f},{j}):'
            f' got 0x{int(got[f,j]):02x}, want 0x{int(want[f,j]):02x}'
            f' (time samples {8*int(j)}..{8*int(j)+7}). [{tag}]')


def _check_kernel(cp, rng, nfreq, nt, verbose):
    """One draw: pack it on the GPU, from a strided view, and check the bytes and the guard.

    The input is always a column window of a wider array -- the production call shape, since
    the chain runs on blocks and the mask is written per chunk -- with a random offset. A
    random (not multiple-of-32) offset is wanted: it is what exercises the misaligned-row load
    path, which is the general case the kernel promises to accept.

    The output is required to be fully contiguous, so it has no padding to check for
    overruns. Instead it is carved out of the FRONT of a longer flat buffer, whose tail is
    prefilled and has to come back untouched.
    """

    warps = int(rng.choice(WARP_COUNTS))
    pad = int(rng.integers(0, 65))
    off = int(rng.integers(0, pad + 1))

    w = make_weights(rng, nfreq, nt)

    full = cp.zeros((nfreq, nt + pad), dtype=cp.float32)
    full[:, off:off+nt] = cp.asarray(w)
    weights = full[:, off:off+nt]

    nguard = 64
    buf = cp.full(nfreq * (nt // 8) + nguard, 0xAA, dtype=cp.uint8)
    rfi_mask = buf[:nfreq * (nt // 8)].reshape(nfreq, nt // 8)
    assert rfi_mask.flags.c_contiguous

    k = RfiMaskPackingKernel(nfreq, nt, warps)
    assert (k.nfreq, k.nt, k.warps_per_block) == (nfreq, nt, warps)

    k.launch(rfi_mask, weights)
    cp.cuda.get_current_stream().synchronize()

    tag = _tag(nfreq, nt, warps, pad, off)
    _compare(cp.asnumpy(rfi_mask), reference_mask(w), tag)

    guard = cp.asnumpy(buf[nfreq * (nt // 8):])
    if np.any(guard != 0xAA):
        raise AssertionError(
            f'test_rfi_mask_packing_kernel: the kernel wrote past the end of the mask, at'
            f' {int(np.sum(guard != 0xAA))} of {guard.size} guard bytes. [{tag}]')

    if verbose:
        atomic_print(f'    test_rfi_mask_packing_kernel: {tag}: ok')


def _check_arguments(cp):
    """Every argument error raises. Runs on iteration 0 only."""

    # The constructor's geometry rules.
    _expect_raise('nfreq = 0', lambda: RfiMaskPackingKernel(0, 1024))
    _expect_raise('nt = 0', lambda: RfiMaskPackingKernel(4, 0))
    _expect_raise('nt = 512 (not a multiple of 1024)',
                  lambda: RfiMaskPackingKernel(4, 512))
    _expect_raise('nt = 1536 (not a multiple of 1024)',
                  lambda: RfiMaskPackingKernel(4, 1536))
    _expect_raise('warps_per_block = 5', lambda: RfiMaskPackingKernel(4, 1024, 5))

    (nfreq, nt) = (4, 1024)
    k = RfiMaskPackingKernel(nfreq, nt)

    weights = cp.zeros((nfreq, nt), dtype=cp.float32)
    rfi_mask = cp.zeros((nfreq, nt//8), dtype=cp.uint8)

    def launch(**kw):
        args = dict(rfi_mask=rfi_mask, weights=weights)
        args.update(kw)
        return lambda: k.launch(**args)

    _expect_raise('mask has the wrong shape',
                  launch(rfi_mask=cp.zeros((nfreq, nt//8 + 1), dtype=cp.uint8)))
    _expect_raise('mask is not contiguous',
                  launch(rfi_mask=cp.zeros((nfreq, nt//4), dtype=cp.uint8)[:, ::2]))
    _expect_raise('host mask', launch(rfi_mask=np.zeros((nfreq, nt//8), dtype=np.uint8)))
    _expect_raise('weights have the wrong shape',
                  launch(weights=cp.zeros((nfreq, nt//2), dtype=cp.float32)))
    _expect_raise('the weights time axis is not contiguous',
                  launch(weights=cp.zeros((nfreq, 2*nt), dtype=cp.float32)[:, ::2]))
    _expect_raise('transposed weights (time stride != 1)',
                  launch(weights=cp.zeros((nt, nfreq), dtype=cp.float32).T))
    _expect_raise('host weights', launch(weights=np.zeros((nfreq, nt), dtype=np.float32)))

    # Overlapping rows: a (1, nt) row broadcast to nfreq rows has frequency stride 0. Note
    # that this one does NOT reach the kernel's own stride check -- ksgpu::Array refuses an
    # overlapping array when it is constructed (Array.cpp:172). Kept because the precondition
    # is real and should stay tested wherever it is enforced.
    _expect_raise('weights rows overlap (frequency stride 0)',
                  launch(weights=cp.broadcast_to(cp.zeros((1, nt), dtype=cp.float32),
                                                 (nfreq, nt))))


def _check_bit_order_by_hand(cp):
    """One hand-computed case, independent of np.packbits. Runs on iteration 0 only.

    np.packbits is the oracle everywhere else, so this is the one check that does not rely on
    it: a single good sample at time t must set bit (t%8) of byte (t//8) and nothing else.
    """

    (nfreq, nt) = (1, 1024)
    k = RfiMaskPackingKernel(nfreq, nt)
    rfi_mask = cp.zeros((nfreq, nt//8), dtype=cp.uint8)

    for t in (0, 1, 7, 8, 9, 31, 32, 33, 255, 256, 1022, 1023):
        weights = cp.zeros((nfreq, nt), dtype=cp.float32)
        weights[0, t] = 1.0
        k.launch(rfi_mask, weights)
        cp.cuda.get_current_stream().synchronize()

        got = cp.asnumpy(rfi_mask)[0]
        want = np.zeros(nt//8, dtype=np.uint8)
        want[t // 8] = 1 << (t % 8)

        if not np.array_equal(got, want):
            nz = np.flatnonzero(got)
            raise AssertionError(
                f'test_rfi_mask_packing_kernel: a single good sample at t={t} should set'
                f' bit {t%8} of byte {t//8} and nothing else; got nonzero bytes'
                f' {nz.tolist()} = {[hex(int(got[j])) for j in nz]}')


def test_rfi_mask_packing_kernel(iteration=0, rng=None, verbose=False):
    import cupy as cp

    rng = _default_rng(rng)

    if iteration == 0:
        _check_arguments(cp)
        _check_bit_order_by_hand(cp)

        # The production geometry, once. The old chain ran its mask counter inside a
        # wi_sub_pipeline with Df=16, so nfreq is 16384/16 = 1024 (which is why real files
        # have nrfifreq = 1024), at nt_chunk = 1024. Costs 4 MB of GPU input; the random
        # draws below are much smaller.
        _check_kernel(cp, rng, 1024, 1024, verbose)
        atomic_print('    test_rfi_mask_packing_kernel: production (1024 x 1024) draw passed')

    nfreq = int(rng.integers(1, 41))
    nt = 1024 * int(rng.integers(1, 5))
    _check_kernel(cp, rng, nfreq, nt, verbose)
