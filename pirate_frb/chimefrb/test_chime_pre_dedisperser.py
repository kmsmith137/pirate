"""Randomized unit tests for ChimePreDedisperser.

Dispatched from ``python -m pirate_frb test --cfrb``.

The oracle is the same chain run WITHOUT the driver: every chunk of a block decoded on the
CPU (with the same prescale), concatenated along time, run through a second copy of the
pipeline in one launch with a fresh extractor planted on a fresh array, and sliced per
chunk. What the driver adds on top of that -- the staged copies, the decode into a column
window, the block-by-block handshake, the per-chunk copy out -- must change nothing, so the
comparison is BITWISE. The chunks come from test_assembled_chunk.py's writer, in sequences
with contiguous fpga_begin.
"""

import threading

import numpy as np

from . import (AssembledChunk, AssembledChunkReader, ChimePreDedisperser, ExamplePythonTransform,
               Pipeline, RfiMaskExtractor, RfiMaskPipeline)
from ..core import SlabAllocator
from ..utils import atomic_print
from .test_assembled_chunk import Chunk
from .test_assembled_chunk_reader import TempChunkFiles
from .testutils import default_rng as _default_rng


def _expect_raise(exc, f, *args, **kwds):
    try:
        f(*args, **kwds)
    except exc:
        return
    raise AssertionError(f'test_chime_pre_dedisperser: expected {exc.__name__} from {f}')


# -------------------------------------------------------------------------------------------------
#
# Chunk sequences


def make_chunk_sequence(rng, nchunks, nfreq_coarse, nupfreq, nt_coarse, ichunk0, beam_id=7,
                        fpga_counts_per_sample=384):
    """'nchunks' chunks with identical parameters, contiguous in time, the first at absolute
    chunk index 'ichunk0'. nt_per_packet is 16 (the decode kernels' requirement) and the RFI
    mask, which the driver ignores, is random."""

    nt = 16 * nt_coarse
    nfreq = nfreq_coarse * nupfreq
    chunks = []
    for c in range(nchunks):
        scales = rng.uniform(0.1, 900.0, size=(nfreq_coarse, nt_coarse)).astype(np.float32)
        offsets = rng.uniform(-7300.0, 7300.0, size=(nfreq_coarse, nt_coarse)).astype(np.float32)
        zero = rng.uniform(size=scales.shape) < 0.2
        scales[zero] = 0.0
        offsets[zero] = 0.0

        data = rng.integers(0, 256, size=(nfreq, nt), dtype=np.uint8)
        data = data.reshape(nfreq_coarse, nupfreq, nt_coarse, 16)
        data[zero[:, None, :, None].repeat(nupfreq, 1).repeat(16, 3)] = 0
        data = data.reshape(nfreq, nt)
        hit = rng.uniform(size=data.shape) < 0.1
        data[hit] = np.where(rng.uniform(size=int(hit.sum())) < 0.5, 0, 255).astype(np.uint8)

        rfi_mask = rng.integers(0, 256, size=(nfreq_coarse, nt // 8), dtype=np.uint8)
        chunks.append(Chunk(beam_id=beam_id, nupfreq=nupfreq, nt_per_packet=16,
                            fpga_counts_per_sample=fpga_counts_per_sample, binning=1,
                            nfreq_coarse=nfreq_coarse, nt_coarse=nt_coarse,
                            fpga_begin=(ichunk0 + c) * nt * fpga_counts_per_sample,
                            frame0_nano=123456789, nrfifreq=nfreq_coarse, has_rfi_mask=True,
                            scales=scales, offsets=offsets, data=data, rfi_mask=rfi_mask))
    return chunks


def random_chain(rng, nfreq, ntime):
    """A chain with one extractor: a plain Pipeline, or one wrapping an RfiMaskPipeline
    (Df=2, Dt=1) when the geometry allows it."""

    if (nfreq % 64 == 0) and (rng.uniform() < 0.5):
        inner = [ExamplePythonTransform(1, nfreq // 2, ntime, sigma=3.0), RfiMaskExtractor(1, nfreq // 2, ntime)]
        return Pipeline([RfiMaskPipeline(inner, 2, 1, 0.0)])
    return Pipeline([ExamplePythonTransform(1, nfreq, ntime, sigma=3.0), RfiMaskExtractor(1, nfreq, ntime)])


# -------------------------------------------------------------------------------------------------
#
# The oracle


def expected_masks(cp, pipeline, chunks, prescale):
    """The per-chunk masks of one BLOCK of chunks, from a fresh copy of the chain run in one
    launch on the CPU-decoded, concatenated data."""

    cpb = len(chunks)
    p2 = Pipeline.from_yaml_dict(pipeline.to_yaml_dict(), pipeline.nbeams, pipeline.nfreq, pipeline.ntime)
    ext = p2.get_mask_extractor()
    nt_mask = ext.ntime // cpb

    intensity = np.concatenate([np.asarray(c.decode_intensity(apply_rfimask=False, scale=prescale))
                                for c in chunks], axis=1)[None]
    weights = np.concatenate([np.asarray(c.decode_weights(apply_rfimask=False)) for c in chunks], axis=1)[None]

    dst = cp.full((cpb, 1, ext.nfreq, nt_mask // 8), 0xAA, dtype=cp.uint8)
    ext.set_rfi_mask(dst)
    p2.launch(cp.asarray(intensity), cp.asarray(weights), None)
    cp.cuda.get_current_stream().synchronize()
    return [cp.asnumpy(dst[j, 0]) for j in range(cpb)]


def _compare(got, want, tag):
    assert got.shape == want.shape and got.dtype == np.uint8, f'{tag}: got {got.shape} {got.dtype}'
    bad = (got != want)
    assert not bad.any(), f'{tag}: {int(bad.sum())} of {bad.size} mask bytes differ'


# -------------------------------------------------------------------------------------------------
#
# Runs


def run_single_threaded(cp, cpd, filenames, verbose, tag):
    """put a block, get its masks, repeat; then finish() and the two trailing Nones."""

    cpb = None
    got = []
    i = 0
    for fn in filenames:
        cpd.put_chunk(AssembledChunk.from_msgpack(fn))
        i += 1
        cpb = cpd.chunks_per_block
        if i % cpb == 0:
            for _ in range(cpb):
                got.append(cpd.get_rfimask())
    cpd.finish()
    assert cpd.get_rfimask() is None and cpd.get_rfimask() is None, f'{tag}: expected None after finish()'
    assert (cpd.nchunks_put, cpd.nchunks_got) == (len(filenames), len(filenames))
    return got


def run_two_threads(cp, cpd, filenames, nthreads, allocator, verbose, tag):
    """A producer thread feeding from an AssembledChunkReader; the caller's thread consumes,
    alternating between a caller-supplied 'out' and the pinned default."""

    failure = []

    def producer():
        try:
            with AssembledChunkReader(filenames, nthreads, allocator) as reader:
                for chunk in reader:
                    cpd.put_chunk(chunk)
            cpd.finish()
        except BaseException as e:      # reported through the latch; recorded here for the assert
            failure.append(e)

    t = threading.Thread(target=producer, name='test_cpd_producer')
    t.start()
    got = []
    while True:
        if len(got) % 2:
            out = np.zeros((cpd.nrfifreq, cpd.nt_mask // 8), dtype=np.uint8) if cpd.nt_mask else None
            m = cpd.get_rfimask(out=out) if (out is not None) else cpd.get_rfimask()
            if (m is not None) and (out is not None):
                assert m is out, f'{tag}: out= was not written in place'
        else:
            m = cpd.get_rfimask()
        if m is None:
            break
        got.append(m)
    t.join()
    assert not failure, f'{tag}: the producer raised {failure[0]!r}'
    assert len(got) == len(filenames), f'{tag}: got {len(got)} masks for {len(filenames)} files'
    return got


def check_run(cp, rng, verbose):
    """One random stream through one random chain, single-threaded or two-threaded, against
    the oracle."""

    nfreq_coarse = int(rng.choice([2, 4]))
    nupfreq = int(rng.choice([1, 2, 16]))
    nfreq = nfreq_coarse * nupfreq
    nt_coarse = 64 * int(rng.integers(1, 3))            # nt = 1024 or 2048
    nt = 16 * nt_coarse
    cpb = int(rng.integers(1, 5))
    nblocks = int(rng.integers(1, 4))
    ichunk0 = cpb * int(rng.integers(0, 1000))
    prescale = float(rng.choice([1.0, 1.0e-4, rng.uniform(0.01, 100.0)]))
    two_threads = bool(rng.uniform() < 0.5)
    pinned = bool(rng.uniform() < 0.5)

    pipeline = random_chain(rng, nfreq, cpb * nt)
    chunks = make_chunk_sequence(rng, cpb * nblocks, nfreq_coarse, nupfreq, nt_coarse, ichunk0)
    tag = (f'nfreq={nfreq} nt={nt} cpb={cpb} nblocks={nblocks} prescale={prescale:g}'
           f' two_threads={two_threads} pinned={pinned} chain={pipeline!r}')

    with TempChunkFiles(chunks, rng) as filenames:
        cpd = ChimePreDedisperser(pipeline, prescale)
        assert (cpd.nfreq, cpd.ntime, cpd.nt, cpd.chunks_per_block) == (nfreq, cpb * nt, None, None)

        if two_threads:
            allocator = SlabAllocator('af_rhost') if pinned else None
            got = run_two_threads(cp, cpd, filenames, int(rng.integers(1, 4)), allocator, verbose, tag)
        else:
            got = run_single_threaded(cp, cpd, filenames, verbose, tag)

        assert (cpd.nt, cpd.chunks_per_block, cpd.nt_mask) == (nt, cpb, nt), tag
        for k in range(nblocks):
            block = [AssembledChunk.from_msgpack(fn) for fn in filenames[k*cpb:(k+1)*cpb]]
            for (j, want) in enumerate(expected_masks(cp, pipeline, block, prescale)):
                _compare(got[k*cpb + j], want, f'{tag} block {k} chunk {j}')

    if verbose:
        atomic_print(f'    test_chime_pre_dedisperser: {tag}: ok')


def check_errors(cp, rng):
    """Construction errors, the chunk checks, the partial block, the latch, and stop()."""

    (nfreq_coarse, nupfreq, nt_coarse, cpb) = (2, 2, 64, 2)
    nfreq = nfreq_coarse * nupfreq
    nt = 16 * nt_coarse
    chain = lambda: Pipeline([RfiMaskExtractor(1, nfreq, cpb * nt)])

    # Construction.
    _expect_raise(ValueError, ChimePreDedisperser, Pipeline([ExamplePythonTransform(1, nfreq, cpb * nt)]), 1.0)  # no extractor
    _expect_raise(ValueError, ChimePreDedisperser, Pipeline([RfiMaskExtractor(2, nfreq, cpb * nt)]), 1.0)     # nbeams
    _expect_raise(ValueError, ChimePreDedisperser, chain(), 0.0)                                              # prescale
    _expect_raise(TypeError, ChimePreDedisperser, 'not a chain', 1.0)

    chunks = make_chunk_sequence(rng, 3 * cpb, nfreq_coarse, nupfreq, nt_coarse, ichunk0=4 * cpb)
    other = make_chunk_sequence(rng, 1, nfreq_coarse, nupfreq, nt_coarse, ichunk0=4 * cpb + 1, beam_id=8)[0]
    wide = make_chunk_sequence(rng, 1, nfreq_coarse, nupfreq + 1, nt_coarse, ichunk0=4 * cpb)[0]

    with TempChunkFiles(chunks + [other, wide], rng) as names:
        read = lambda i: AssembledChunk.from_msgpack(names[i])

        # A gap, then the latch: every later call reports the first error.
        cpd = ChimePreDedisperser(chain(), 1.0)
        cpd.put_chunk(read(0))
        _expect_raise(ValueError, cpd.put_chunk, read(2))
        assert cpd.is_stopped
        _expect_raise(RuntimeError, cpd.put_chunk, read(1))
        _expect_raise(RuntimeError, cpd.get_rfimask)
        _expect_raise(RuntimeError, cpd.finish)

        # Misaligned first chunk, unless the check is off.
        _expect_raise(ValueError, ChimePreDedisperser(chain(), 1.0).put_chunk, read(1))
        cpd = ChimePreDedisperser(chain(), 1.0, check_alignment=False)
        cpd.put_chunk(read(1))
        cpd.put_chunk(read(2))
        assert cpd.get_rfimask().shape == (nfreq, nt // 8)

        # Another beam; other parameters; the wrong channel count; metadata only; a partial block.
        _expect_raise(ValueError, ChimePreDedisperser(chain(), 1.0).put_chunk, read(len(chunks) + 1))
        cpd = ChimePreDedisperser(chain(), 1.0)
        cpd.put_chunk(read(0))
        _expect_raise(ValueError, cpd.put_chunk, read(len(chunks)))          # beam 8 where 7 is expected
        cpd = ChimePreDedisperser(chain(), 1.0)
        _expect_raise(ValueError, cpd.put_chunk, AssembledChunk.from_msgpack(names[0], metadata_only=True))
        cpd = ChimePreDedisperser(chain(), 1.0)
        cpd.put_chunk(read(0))
        _expect_raise(ValueError, cpd.finish)
        _expect_raise(RuntimeError, cpd.get_rfimask)                          # latched

        # A bad 'out'.
        cpd = ChimePreDedisperser(chain(), 1.0)
        cpd.put_chunk(read(0))
        cpd.put_chunk(read(1))
        _expect_raise(ValueError, cpd.get_rfimask, out=np.zeros((nfreq, nt // 8 + 1), dtype=np.uint8))
        _expect_raise(RuntimeError, cpd.get_rfimask)                          # latched by the bad out

        # stop() wakes a consumer parked in get_rfimask().
        cpd = ChimePreDedisperser(chain(), 1.0)
        outcome = []

        def consumer():
            try:
                cpd.get_rfimask()
                outcome.append('returned')
            except RuntimeError:
                outcome.append('raised')

        t = threading.Thread(target=consumer, name='test_cpd_consumer')
        t.start()
        t.join(timeout=0.2)
        assert t.is_alive(), 'the consumer should be blocked with nothing launched'
        cpd.stop()
        t.join(timeout=5.0)
        assert outcome == ['raised'], f'stop() did not wake the consumer: {outcome}'
        assert cpd.is_stopped

        # The context manager stops on exit.
        with ChimePreDedisperser(chain(), 1.0) as cpd:
            cpd.put_chunk(read(0))
        assert cpd.is_stopped
        _expect_raise(RuntimeError, cpd.put_chunk, read(1))


# -------------------------------------------------------------------------------------------------


def test_chime_pre_dedisperser(iteration=0, rng=None, verbose=False):
    """One random stream against the oracle in the module docstring; on iteration 0, also
    every error path."""

    try:
        import cupy as cp
    except ImportError:
        if verbose:
            atomic_print('    test_chime_pre_dedisperser: cupy not available, skipped')
        return

    rng = _default_rng(rng)
    check_run(cp, rng, verbose)

    if iteration == 0:
        check_errors(cp, rng)
