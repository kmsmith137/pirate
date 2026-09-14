"""AssembledChunkReader: the python side of the C++ class of the same name.

The class itself is C++ (include/pirate/chimefrb/AssembledChunkReader.hpp), bound with
pybind11. This module adds the python interface a caller actually uses -- iteration and the
context-manager pair -- with ksgpu.inject_methods, and re-exports the class. The class
docstring lives here too (option 2 in notes/docstrings.md), next to that code.

Unlike the transforms and kernels, whose injections share one file (cpp_transforms.py),
this class is the step BEFORE the transforms: it turns a list of filenames into
AssembledChunks, and has nothing to do with the transform interface.
"""

import ksgpu

from ..pirate_pybind11 import AssembledChunkReader


@ksgpu.inject_methods(AssembledChunkReader)
class AssembledChunkReaderInjections:
    """Reads a list of chimefrb data files with a pool of threads, and hands them to the
    caller one at a time, IN FILENAME ORDER.

    Equivalent to looping over :meth:`AssembledChunk.from_msgpack`, with the same order and
    the same exceptions, except that up to ``nthreads`` files are read ahead of the caller.
    Use it as an iterator, and as a context manager if you want the threads to stop at a
    definite point rather than when the reader is garbage-collected::

        with AssembledChunkReader(sorted(glob.glob('*.msg')), nthreads=8) as reader:
            for chunk in reader:
                intensity = chunk.decode_intensity(apply_rfimask=False)

    The order guarantee is what makes the obvious thing work::

        for (filename, chunk) in zip(filename_list, AssembledChunkReader(filename_list)):
            ...

    MEMORY. At most ``nthreads`` chunks exist inside the reader at once. One chunk owns a
    ~17 MB buffer at the production CHIME geometry, so the default costs about 68 MB.
    Keeping the chunks you take defeats that bound -- a chunk, or any numpy view of one,
    keeps its buffer alive.

    READ ERRORS ARRIVE IN ORDER. A file that fails to read does not interrupt the files
    before it: :meth:`get_chunk` delivers those first and then raises, exactly where a
    serial loop would. Raising stops the reader, so nothing is read past the failure.

    ``allocator`` (optional) is a :class:`SlabAllocator` to take the chunk buffers from,
    instead of allocating per file. Without one, a chunk's buffer is UNPINNED host memory,
    so copying its arrays to the GPU is staged by the CUDA runtime; the simplest way to get
    pinned memory, and DMA copies, is a dummy-mode allocator, ``SlabAllocator('af_rhost')``,
    which hands out fresh page-locked memory per file and never blocks (a few milliseconds
    per file to register the buffer). Three conditions come with a POOLED allocator, one
    built on a :class:`BumpAllocator`: all the files must have
    identical parameters (a SlabAllocator serves one slab size); the pool must hold at least
    ``nthreads`` slabs, or the reader DEADLOCKS (a dummy-mode allocator, which never blocks,
    is always safe); and the reader stops the allocator on teardown, so it must not be
    shared with anything that outlives the reader. The C++ constructor comment
    (``include/pirate/chimefrb/AssembledChunkReader.hpp``) spells all three out.

    Attributes (read-only):

    - ``filenames`` (list of str) -- what the reader was constructed with.
    - ``nfiles`` (int) -- ``len(filenames)``.
    - ``nthreads`` (int) -- worker threads, and the read-ahead depth. Clamped to ``nfiles``,
      so it can be less than what you asked for (and 0 for an empty list).
    - ``allocator`` -- the SlabAllocator given to the constructor, or None.
    - ``is_stopped`` (bool).
    """

    def __iter__(self):
        # A generator on purpose. ksgpu.inject_methods only injects dunders on a whitelist,
        # and __next__ is NOT on it (__iter__ is). A generator __iter__ needs no __next__ --
        # do not "tidy" this into an __iter__/__next__ pair, whose __next__ would be
        # silently dropped, leaving an iterator that raises TypeError.
        while True:
            chunk = self.get_chunk()
            if chunk is None:
                return
            yield chunk

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        self.stop()
        return False    # do not suppress an exception from the body

    def __repr__(self):
        stopped = ", stopped" if self.is_stopped else ""
        return f'AssembledChunkReader({self.nfiles} file(s), nthreads={self.nthreads}{stopped})'
