"""Randomized unit tests for ChimeDequantizationKernel.

Dispatched from ``python -m pirate_frb test --cfrb``.

The oracle is ``AssembledChunk.decode_intensity()`` / ``decode_weights()`` -- the CPU
implementation of the same operation -- and the comparison is BITWISE, with no tolerance.
That is not optimism: the intensity is one fused multiply-add on both sides (AVX2's
``_mm256_fmadd_ps`` and the GPU's FFMA both round once), and the weights are comparisons
against 0 and 255. Two things the bitwise comparison is there to catch, which a tolerance
would hide:

- masking must be a SELECT, not a multiply. The reference ANDs the float with an
  all-ones/all-zeros mask, so a masked sample is +0.0 whatever it held; multiplying by 0.0f
  gives -0.0 for a negative intensity, and NaN for an infinite one.
- the saturation sentinels (data 0 and 255) zero the WEIGHT only. The intensity still gets
  the affine transform.

The one way bit-exactness could fail legitimately is a denormal intensity: the GPU build
uses --use_fast_math, which flushes denormals to zero, and the CPU does not. It takes
|scale*data + offset| < 1.2e-38 from inputs of order 100, which these draws do not produce.

WHAT THIS FILE CANNOT ESTABLISH: that the CPU decode itself is right. That is
misc/chimefrb/spot_checks/assembled_chunk_decode, which runs the real ch_frb_io decode, and
test_assembled_chunk.py, which compares the CPU decode with an independent numpy reference
every iteration. This test inherits both.
"""

import numpy as np

from . import AssembledChunk, ChimeDequantizationKernel
from ..utils import atomic_print
from .test_assembled_chunk import (TempChunkFile, _expect_raise, make_random_chunk)
from .testutils import default_rng as _default_rng


WARP_COUNTS = [4, 8, 16, 32]


def _read_chunk(chunk, rng):
    """Write 'chunk' to a scratch file and read it back as an AssembledChunk.

    The round trip through the file is what supplies the oracle: the AssembledChunk carries
    both the arrays the kernel consumes and the decode methods it is compared against.
    """

    with TempChunkFile(chunk.to_msgpack(rng=rng)) as fn:
        return AssembledChunk.from_msgpack(fn)


def _gpu_inputs(cp, c):
    """The kernel's four input arrays, on the GPU. 'rfi_mask' is None if the file has none."""

    mask = cp.asarray(np.asarray(c.rfi_mask)) if c.has_rfi_mask else None
    return (cp.asarray(np.asarray(c.scales)),
            cp.asarray(np.asarray(c.offsets)),
            cp.asarray(np.asarray(c.data)),
            mask)


def _tag(c, apply_rfimask, warps):
    return (f'(nfreq,nt)=({c.nfreq},{c.nt}) nupfreq={c.nupfreq} nt_coarse={c.nt_coarse}'
            f' nrfifreq={c.nrfifreq} apply_rfimask={apply_rfimask} warps={warps}')


def _compare(got, want, name, tag):
    """Bitwise, through a uint32 view -- which is also how +0.0 is told from -0.0."""

    bad = (got.view(np.uint32) != want.view(np.uint32))
    if bad.any():
        (f, t) = np.argwhere(bad)[0]
        raise AssertionError(
            f'test_chime_dequantization_kernel: GPU and CPU {name} differ at'
            f' {int(bad.sum())} of {bad.size} samples; first at (f,t) = ({f},{t}):'
            f' got {got[f,t]!r}, want {want[f,t]!r}. [{tag}]')


def _launch(cp, c, inputs, apply_rfimask, warps, out=None):
    """Run the kernel on 'c', returning (intensity, weights) as numpy arrays.

    'out' is an optional (intensity, weights) pair of cupy arrays to write into -- which is
    how _check_freq_stride() hands it a strided view. Otherwise a contiguous pair is
    allocated here.
    """

    (scales, offsets, data, mask) = inputs

    if out is None:
        out = (cp.empty((c.nfreq, c.nt), dtype=cp.float32),
               cp.empty((c.nfreq, c.nt), dtype=cp.float32))

    k = ChimeDequantizationKernel(c.nfreq, c.nt, c.nfreq_coarse, c.nt_coarse, warps)
    assert (k.nfreq, k.nt, k.nfreq_coarse, k.nt_coarse) == (c.nfreq, c.nt, c.nfreq_coarse,
                                                            c.nt_coarse)
    assert (k.nupfreq, k.warps_per_block) == (c.nupfreq, warps)

    k.launch(out[0], out[1], scales, offsets, data, mask, apply_rfimask)
    cp.cuda.get_current_stream().synchronize()
    return (cp.asnumpy(out[0]), cp.asnumpy(out[1]))


def _check_kernel(cp, rng, c, inputs, verbose):
    """The GPU kernel against the CPU decode, bitwise, for every applicable apply_rfimask."""

    warps = int(rng.choice(WARP_COUNTS))
    mask_choices = (False, True) if c.has_rfi_mask else (False,)

    for apply_rfimask in mask_choices:
        (got_i, got_w) = _launch(cp, c, inputs, apply_rfimask, warps)
        tag = _tag(c, apply_rfimask, warps)
        _compare(got_i, np.asarray(c.decode_intensity(apply_rfimask=apply_rfimask)),
                 'intensity', tag)
        _compare(got_w, np.asarray(c.decode_weights(apply_rfimask=apply_rfimask)),
                 'weights', tag)

        if verbose:
            atomic_print(f'    test_chime_dequantization_kernel: {tag}: ok')


def _check_freq_stride(cp, rng, c, inputs, verbose):
    """The outputs written into a column window of a wider array.

    This is the production call shape -- a chunk is a quarter of a pipeline block's time
    axis -- and it is the only check that exercises a frequency stride other than nt. As
    well as the values, it checks that NOTHING OUTSIDE THE WINDOW IS TOUCHED: the two output
    arrays are prefilled with NaN, and every element outside the window has to come back
    with those exact bits.
    """

    pad = int(rng.integers(0, 65))
    off = int(rng.integers(0, pad + 1))
    apply_rfimask = bool(c.has_rfi_mask and (rng.uniform() < 0.5))
    warps = int(rng.choice(WARP_COUNTS))

    # Independent strides for the two outputs, since the kernel takes them separately: a
    # kernel that used the intensity's stride for both would pass with equal padding.
    pad_w = int(rng.integers(0, 65))
    off_w = int(rng.integers(0, pad_w + 1))

    full_i = cp.full((c.nfreq, c.nt + pad), np.nan, dtype=cp.float32)
    full_w = cp.full((c.nfreq, c.nt + pad_w), np.nan, dtype=cp.float32)
    out = (full_i[:, off:off+c.nt], full_w[:, off_w:off_w+c.nt])

    (got_i, got_w) = _launch(cp, c, inputs, apply_rfimask, warps, out=out)
    tag = _tag(c, apply_rfimask, warps) + f' pad=({pad},{pad_w}) off=({off},{off_w})'

    _compare(got_i, np.asarray(c.decode_intensity(apply_rfimask=apply_rfimask)),
             'intensity', tag)
    _compare(got_w, np.asarray(c.decode_weights(apply_rfimask=apply_rfimask)),
             'weights', tag)

    # The padding, bitwise: a NaN payload survives a copy but not arithmetic, and comparing
    # bits is also the only way to see a NaN that stayed put.
    nan_bits = np.float32(np.nan).view(np.uint32)
    for (name, full, o) in (('intensity', full_i, off), ('weights', full_w, off_w)):
        h = cp.asnumpy(full).view(np.uint32)
        touched = np.concatenate([h[:, :o].reshape(-1), h[:, o+c.nt:].reshape(-1)])
        if touched.size and np.any(touched != nan_bits):
            raise AssertionError(
                f'test_chime_dequantization_kernel: the kernel wrote outside the {name}'
                f' window at {int(np.sum(touched != nan_bits))} of {touched.size} padding'
                f' elements. [{tag}]')

    if verbose:
        atomic_print(f'    test_chime_dequantization_kernel: strided outputs, {tag}: ok')


def _check_arguments(cp):
    """Every argument error raises. Runs on iteration 0 only."""

    # The constructor's geometry rules.
    _expect_raise('nfreq = 0', lambda: ChimeDequantizationKernel(0, 16, 1, 1))
    _expect_raise('nt_coarse = 0', lambda: ChimeDequantizationKernel(4, 16, 2, 0))
    _expect_raise('nfreq_coarse does not divide nfreq',
                  lambda: ChimeDequantizationKernel(4, 16, 3, 1))
    _expect_raise('nt != 16*nt_coarse (nt_per_packet = 32)',
                  lambda: ChimeDequantizationKernel(4, 32, 2, 1))
    _expect_raise('warps_per_block = 5', lambda: ChimeDequantizationKernel(4, 16, 2, 1, 5))

    # A 4-channel, 32-sample geometry: nupfreq = 2, nt_coarse = 2, and a mask with 2 rows.
    (nfreq, nt, nfreq_coarse, nt_coarse) = (4, 32, 2, 2)
    k = ChimeDequantizationKernel(nfreq, nt, nfreq_coarse, nt_coarse)

    sc = cp.zeros((nfreq_coarse, nt_coarse), dtype=cp.float32)
    of = cp.zeros((nfreq_coarse, nt_coarse), dtype=cp.float32)
    data = cp.zeros((nfreq, nt), dtype=cp.uint8)
    mask = cp.zeros((2, nt//8), dtype=cp.uint8)
    out_i = cp.zeros((nfreq, nt), dtype=cp.float32)
    out_w = cp.zeros((nfreq, nt), dtype=cp.float32)

    def launch(**kw):
        args = dict(intensity=out_i, weights=out_w, scales=sc, offsets=of, data=data,
                    rfi_mask=mask, apply_rfimask=False)
        args.update(kw)
        return lambda: k.launch(**args)

    _expect_raise('intensity has the wrong shape',
                  launch(intensity=cp.zeros((nfreq, nt//2), dtype=cp.float32)))
    _expect_raise('the weights time axis is not contiguous',
                  launch(weights=cp.zeros((nfreq, 2*nt), dtype=cp.float32)[:, ::2]))
    _expect_raise('transposed output (time stride != 1)',
                  launch(intensity=cp.zeros((nt, nfreq), dtype=cp.float32).T))
    _expect_raise('intensity is weights', launch(weights=out_i))
    _expect_raise('host output', launch(intensity=np.zeros((nfreq, nt), dtype=np.float32)))
    _expect_raise('host input', launch(data=np.zeros((nfreq, nt), dtype=np.uint8)))
    _expect_raise('scales has the wrong shape',
                  launch(scales=cp.zeros((nfreq_coarse, nt_coarse+1), dtype=cp.float32)))
    _expect_raise('data is not contiguous',
                  launch(data=cp.zeros((nfreq, 2*nt), dtype=cp.uint8)[:, ::2]))

    # The mask is only checked when it is used.
    _expect_raise('apply_rfimask with no mask',
                  launch(rfi_mask=None, apply_rfimask=True))
    _expect_raise('mask rows do not divide nfreq',
                  launch(rfi_mask=cp.zeros((3, nt//8), dtype=cp.uint8), apply_rfimask=True))
    _expect_raise('mask has the wrong byte count',
                  launch(rfi_mask=cp.zeros((2, nt//8 + 1), dtype=cp.uint8), apply_rfimask=True))

    # ... and an unusable mask passes unnoticed when it is not.
    k.launch(out_i, out_w, sc, of, data, cp.zeros((3, 1), dtype=cp.uint8), False)
    cp.cuda.get_current_stream().synchronize()


def test_chime_dequantization_kernel(iteration=0, rng=None, verbose=False):
    import cupy as cp

    rng = _default_rng(rng)

    if iteration == 0:
        _check_arguments(cp)

        # The production CHIME geometry (16384 x 1024, nupfreq = 16), once: the random draws
        # below are small, and this is the shape the kernel will actually run on. Costs a
        # 17 MB file and 134 MB of GPU output, the same reason test_assembled_chunk() gates
        # its own full-size draw on iteration 0.
        big = _read_chunk(make_random_chunk(full_size=True), rng)
        _check_kernel(cp, rng, big, _gpu_inputs(cp, big), verbose)
        atomic_print('    test_chime_dequantization_kernel: full-size (16384 x 1024) draw passed')

    # force_ntpp16: the kernel requires nt_per_packet == 16, as the CPU decode does, so any
    # other draw would have neither a legal kernel nor an oracle.
    c = _read_chunk(make_random_chunk(force_ntpp16=True), rng)
    inputs = _gpu_inputs(cp, c)

    _check_kernel(cp, rng, c, inputs, verbose)
    _check_freq_stride(cp, rng, c, inputs, verbose)
