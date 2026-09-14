"""Randomized unit tests for pirate_frb.chimefrb.AssembledChunkReader.

Dispatched from ``python -m pirate_frb test --cfrb``. Needs no GPU.

WHAT THESE TESTS ARE FOR. The reader's contract is "a serial loop over
AssembledChunk.from_msgpack(), only faster", so every check below compares it against that
loop: same chunks, same order, same exceptions at the same point in the sequence. The one
thing that cannot be checked by comparison is that it does not HANG, which is what the
teardown and stop() cases are for -- a threading bug here shows up as a deadlock, not as a
wrong answer, so run the suite under a timeout.

The input files come from test_assembled_chunk.py's own writer (make_random_chunk() +
Chunk.to_msgpack()), so the reader is exercised on the same corner cases the parser is.
"""

import os

import numpy as np

from . import AssembledChunk, AssembledChunkReader
from ..core import SlabAllocator
from ..utils import atomic_print
from .test_assembled_chunk import TempChunkFile, make_random_chunk, _assert_metadata_equal
from .testutils import default_rng as _default_rng


class TempChunkFiles:
    """Context manager: writes a list of chunks to scratch files, removes them all on exit.

    Wraps TempChunkFile (one file, /dev/shm, pid-tagged) so a test can make several at once
    and still clean up on a failure.
    """

    def __init__(self, chunks, rng):
        self.files = [TempChunkFile(c.to_msgpack(rng=rng)) for c in chunks]

    def __enter__(self):
        self.names = [f.__enter__() for f in self.files]
        return self.names

    def __exit__(self, *exc):
        for f in self.files:
            f.__exit__(*exc)
        return False


def _compare(chunk, want, tag):
    """One chunk against the same file read serially: metadata, then the four raw arrays
    BIT-EXACT (they are memcpy paths, so any tolerance would hide a bug)."""

    assert chunk is not None, f'{tag}: reader returned None early'
    _assert_metadata_equal(chunk, want, tag)

    for field in ('data', 'scales', 'offsets'):
        g, w = np.asarray(getattr(chunk, field)), np.asarray(getattr(want, field))
        assert np.array_equal(g, w), f'{tag}: {field} differs'

    if want.rfi_mask is None:
        assert chunk.rfi_mask is None, f'{tag}: expected rfi_mask None'
    else:
        assert np.array_equal(np.asarray(chunk.rfi_mask), np.asarray(want.rfi_mask)), \
            f'{tag}: rfi_mask differs'


def _expect_raise(exc, f, *args, **kwds):
    try:
        f(*args, **kwds)
    except exc:
        return
    raise AssertionError(f'test_assembled_chunk_reader: expected {exc.__name__} from {f}')


def check_order(filenames, nthreads, allocator=None):
    """The central test: the reader's chunks are the serial loop's chunks, in the same order.

    Also covers the end of the list -- get_chunk() returns None there and on every later
    call -- and, since 'nthreads' is drawn by the caller, the degenerate cases (one thread,
    more threads than files, an empty list).
    """

    want = [AssembledChunk.from_msgpack(fn) for fn in filenames]

    reader = AssembledChunkReader(filenames, nthreads, allocator)
    tag = f'nfiles={len(filenames)} nthreads={nthreads} allocator={allocator is not None}'

    assert reader.nfiles == len(filenames), tag
    assert reader.nthreads == min(nthreads, len(filenames)), f'{tag}: nthreads not clamped'

    for (i, w) in enumerate(want):
        _compare(reader.get_chunk(), w, f'{tag} file {i}')

    # Past the end: None, and it stays None (no exception, and not a stopped instance).
    for _ in range(3):
        assert reader.get_chunk() is None, f'{tag}: expected None past the end'
    assert not reader.is_stopped, f'{tag}: the end of the list must not stop the reader'


def check_iteration(filenames, nthreads):
    """The python sugar: 'for chunk in reader', and the context manager."""

    got = list(AssembledChunkReader(filenames, nthreads))
    assert len(got) == len(filenames), f'iteration gave {len(got)} of {len(filenames)} chunks'

    # zip(filenames, reader) is the idiom the order guarantee is there to support.
    with AssembledChunkReader(filenames, nthreads) as reader:
        pairs = list(zip(filenames, reader))
    assert len(pairs) == len(filenames)
    assert reader.is_stopped, 'leaving the with-block must stop the reader'


def check_read_error(chunks, rng, nthreads, ibad):
    """A file that cannot be read does not cost us the files before it.

    The bad file is truncated to a few bytes, which from_msgpack() rejects while parsing the
    header -- before it allocates anything. Files 0..ibad-1 must still arrive, in order, and
    the ibad'th get_chunk() must raise, with the filename in the message.
    """

    with TempChunkFiles(chunks, rng) as filenames:
        want = [AssembledChunk.from_msgpack(fn) for fn in filenames[:ibad]]

        with open(filenames[ibad], 'r+b') as f:
            f.truncate(8)

        reader = AssembledChunkReader(filenames, nthreads)
        tag = f'nfiles={len(filenames)} nthreads={nthreads} ibad={ibad}'

        for (i, w) in enumerate(want):
            _compare(reader.get_chunk(), w, f'{tag} file {i}')

        try:
            reader.get_chunk()
            raise AssertionError(f'{tag}: expected the bad file to raise')
        except RuntimeError as e:
            assert filenames[ibad] in str(e), f'{tag}: message does not name the file: {e}'

        # Rethrowing stopped the reader, so it keeps rethrowing the same error.
        assert reader.is_stopped, f'{tag}: a read error must stop the reader'
        _expect_raise(RuntimeError, reader.get_chunk)


def check_teardown(filenames):
    """Two ways of walking away early, neither of which may hang.

    Dropping the reader after one chunk is the interesting one: with more files than
    threads, the workers are parked on the window cv, and only stop() wakes them. If the
    destructor could not do that, this test would not fail -- it would never return.
    """

    reader = AssembledChunkReader(filenames, nthreads=2)
    assert reader.get_chunk() is not None
    del reader                                  # destructor: stop(), then join

    reader = AssembledChunkReader(filenames, nthreads=2)
    assert reader.get_chunk() is not None
    reader.stop()
    assert reader.is_stopped
    assert reader.get_chunk() is None, 'a clean stop must return None, even with chunks read'


def check_allocator(chunk, rng, nthreads):
    """The optional SlabAllocator: the same chunks come back, and the allocator is
    single-use afterwards.

    Dummy mode (a fresh af_alloc() per slab) on purpose: a bump-backed pool smaller than
    'nthreads' DEADLOCKS the reader -- see the constructor comment in
    AssembledChunkReader.hpp -- and a test has no business sailing near that.
    """

    # Every file must have identical parameters, since a SlabAllocator serves one slab
    # size. The same chunk written to several files is the simplest way to guarantee it.
    with TempChunkFiles([chunk] * 3, rng) as filenames:
        allocator = SlabAllocator('af_rhost')
        check_order(filenames, nthreads, allocator)

        # The reader stopped the allocator when it stopped (the cascade, which is what lets
        # it wake a worker parked in get_slab()). So the allocator is single-use: a second
        # reader on the same one fails. That is the documented ownership rule, pinned here.
        _expect_raise(RuntimeError, AssembledChunkReader(filenames, 1, allocator).get_chunk)


def check_allocator_size_mismatch(rng):
    """A file whose parameters differ from the first one's cannot be served by the same
    allocator, and the reader reports it as that file's read error.

    The two geometries are a small random draw and the full-size CHIME one, because what
    the allocator compares is the SLAB SIZE, not the parameters: two chunks can differ in
    nupfreq or nrfifreq and still want the same number of bytes (which is how an earlier
    version of this check failed, ~1 iteration in 100). A few KB against 17 MB cannot
    collide. Called on iteration 0 only -- the full-size chunk is ~50x the cost of a small
    one, the same reason test_assembled_chunk() gates its own full-size draw.
    """

    small = make_random_chunk()
    big = make_random_chunk(full_size=True)

    # nthreads=1, so that "the first file" -- the one that fixes the slab size -- is
    # deterministically file 0.
    with TempChunkFiles([small, big], rng) as filenames:
        reader = AssembledChunkReader(filenames, 1, SlabAllocator('af_rhost'))
        assert reader.get_chunk() is not None, 'the first file should have been served'
        try:
            reader.get_chunk()
            raise AssertionError('expected the slab-size mismatch to raise')
        except RuntimeError as e:
            assert 'slab' in str(e), f'unexpected message: {e}'


def test_assembled_chunk_reader(iteration=0, rng=None, verbose=False):
    """One iteration, dispatched from 'python -m pirate_frb test --cfrb'.

    Draws the file count and thread count per iteration, so the degenerate shapes (no
    files, one thread, more threads than files) come up on their own rather than being
    hardcoded.
    """

    rng = _default_rng(rng)

    nfiles = int(rng.integers(0, 9))
    nthreads = int(rng.integers(1, 7))
    chunks = [make_random_chunk() for _ in range(max(nfiles, 1))]

    with TempChunkFiles(chunks[:nfiles], rng) as filenames:
        check_order(filenames, nthreads)
        if nfiles > 0:
            check_iteration(filenames, nthreads)
        if nfiles >= 3:
            check_teardown(filenames)

    if nfiles > 0:
        check_read_error(chunks[:nfiles], rng, nthreads, int(rng.integers(0, nfiles)))
        check_allocator(chunks[0], rng, nthreads)

    if iteration == 0:
        check_allocator_size_mismatch(rng)

    if verbose:
        atomic_print(f'    test_assembled_chunk_reader: nfiles={nfiles}, nthreads={nthreads}: ok')
