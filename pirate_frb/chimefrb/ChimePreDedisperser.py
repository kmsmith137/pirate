"""ChimePreDedisperser: runs a chimefrb transform chain on a stream of AssembledChunks and
hands back, one chunk at a time, the RFI mask the chain's RfiMaskExtractor produces.

The driver that turns the pieces of this subpackage into the front half of the old CHIME
search: chunks in (AssembledChunkReader), decode (ChimeDequantizationKernel), the chain
(Pipeline), the mask out (RfiMaskExtractor). See the class docstring for the contract, and
'pirate_frb cfrb reproduce_rfimask' for the comparison against the masks the telescope saved.
"""

import threading

import numpy as np

from .cpp_transforms import GpuTransform
from .ChimeDequantizationKernel import ChimeDequantizationKernel


class ChimePreDedisperser:
    """Runs a chimefrb transform chain on a stream of AssembledChunks and hands back, one
    chunk at a time, the RFI mask the chain's :class:`RfiMaskExtractor` produces.

    The caller sees host arrays only: chunks go in with :meth:`put_chunk`, masks come out with
    :meth:`get_rfimask`, each covering one chunk's ``nt`` samples. Inside, ``chunks_per_block =
    pipeline.ntime // nt`` consecutive chunks are decoded into one GPU block, the chain runs on
    it, and the extractor packs each chunk's window of the block's mask into its own slice of
    one GPU array, from which :meth:`get_rfimask` copies. So chunks must arrive in time order,
    without gaps, and by default the first chunk of every block must be aligned the way the
    CHIME L1 server aligned its 4096-sample blocks (``ichunk % chunks_per_block == 0``, with
    ``ichunk = fpga_begin // (fpga_counts_per_sample * nt)``) -- all of which is checked, so
    that a mistake is an exception rather than a silently shifted mask.

    Meant for one producer thread calling ``put_chunk()`` and one consumer thread calling
    ``get_rfimask()``. The two alternate block by block: the ``put_chunk()`` call that
    launches a block's chain first waits until every mask of the previous block has been
    taken. So a SINGLE thread can drive it -- put ``chunks_per_block`` chunks, then get that
    many masks -- but must keep that order: putting a second block before taking the first
    block's masks waits for a consumer that never comes. Errors latch: after any failure every
    later call raises the same error, and :meth:`stop` wakes a thread blocked in either call.
    Use it as a context manager to have ``stop()`` called on the way out.

    MEMORY. A chunk read the default way lives in unpinned host memory, and its copy to the
    GPU is then staged by the CUDA runtime. Pinned chunks make the copies plain DMAs; the
    simplest way to get them is a dummy-mode allocator handed to the reader,
    ``AssembledChunkReader(paths, allocator=SlabAllocator('af_rhost'))``. Likewise on the way
    out: ``get_rfimask(out=None)`` returns a pinned array (``cupyx.empty_pinned``), and a
    caller allocating its own ``out`` can do the same. Neither is required.

    On the GPU: one (1, nfreq, ntime) block of intensity and weights, the chain's scratch,
    staging arrays for one chunk, and the mask array. Nothing is ring-buffered, which is what
    the block-by-block alternation above buys. Everything runs on one stream of the class's
    own, so the GPU work of one block is queued in order behind the previous block's.

    Attributes (read-only):

    - ``pipeline`` -- the chain, and ``extractor`` -- its RfiMaskExtractor.
    - ``nbeams`` (always 1), ``nfreq``, ``ntime`` -- the block shape, from the chain.
    - ``prescale`` (float) -- multiplies every intensity at decode time; the L1 server used
      1e-4 (its ``intensity_prescale``).
    - ``nt``, ``chunks_per_block`` -- samples per chunk and chunks per block; None until the
      first chunk, which fixes them (and the chunk parameters every later one must match).
    - ``nrfifreq``, ``nt_mask`` -- the mask's shape per chunk is ``(nrfifreq, nt_mask // 8)``:
      the extractor's channel count, and ``nt`` at the extractor's time resolution (``nt`` for
      the production chain). ``nt_mask`` is None until the first chunk.
    - ``nchunks_put``, ``nchunks_got``, ``is_stopped``.
    """

    def __init__(self, pipeline, prescale, check_alignment=True, cuda_device_id=0):
        """Create a ChimePreDedisperser.

        Parameters
        ----------
        pipeline : GpuTransform
            The chain: a :class:`Pipeline` (or any transform) with ``nbeams == 1`` whose
            ``get_mask_extractor()`` finds exactly one :class:`RfiMaskExtractor`. Its ``ntime``
            is the block length, and must be a multiple of the chunk length.
        prescale : float
            Multiplies every intensity at decode time (:class:`ChimeDequantizationKernel`'s
            ``scale``), the way the CHIME L1 server's ``intensity_prescale`` did. No default:
            pass 1e-4 to reproduce the real-time masks, 1 to see whether they reproduce
            without it.
        check_alignment : bool, optional
            Require the first chunk of every block to have ``ichunk % chunks_per_block == 0``
            (see the class docstring). On by default; turn off for data that was not
            produced by an L1 server with a matching ``nt_align``.
        cuda_device_id : int, optional

        Raises
        ------
        ValueError
            If the chain has ``nbeams != 1``, or no extractor, or ``prescale`` is not a
            positive number.
        """
        import cupy as cp

        if not isinstance(pipeline, GpuTransform):
            raise TypeError(f'ChimePreDedisperser: expected a transform (a GpuTransform subclass),'
                            f' got {type(pipeline).__name__}')
        if pipeline.nbeams != 1:
            raise ValueError(f'ChimePreDedisperser: expected a chain with nbeams == 1 (this version'
                             f' drives one beam), got nbeams={pipeline.nbeams}')

        extractor = pipeline.get_mask_extractor()
        if extractor is None:
            raise ValueError('ChimePreDedisperser: the chain contains no RfiMaskExtractor, so there'
                             ' is no mask to hand back; add one at the point where the mask is'
                             ' to be taken (see RfiMaskExtractor)')
        if pipeline.ntime % extractor.ntime != 0:
            raise ValueError(f'ChimePreDedisperser: the chain\'s ntime ({pipeline.ntime}) is not a'
                             f' multiple of its extractor\'s ({extractor.ntime})')

        prescale = float(prescale)
        if not (prescale > 0.0) or not np.isfinite(prescale):
            raise ValueError(f'ChimePreDedisperser: expected a positive finite prescale, got {prescale!r}')

        self.pipeline = pipeline
        self.extractor = extractor
        self.nbeams = 1
        self.nfreq = pipeline.nfreq
        self.ntime = pipeline.ntime
        self.prescale = prescale
        self.nrfifreq = extractor.nfreq
        self.check_alignment = bool(check_alignment)
        self.cuda_device_id = int(cuda_device_id)

        # Fixed by the first chunk (see _initialize()).
        self.nt = None
        self.chunks_per_block = None
        self.nt_mask = None
        self._first = None             # the parameters every later chunk must match
        self._next_fpga_begin = None   # where the next chunk must start
        self._dqk = None
        self._g_scales = self._g_offsets = self._g_data = None
        self._g_mask = None

        # The producer/consumer handshake. Everything below the condition is protected by it.
        # Waiters and predicates:
        #   put_chunk(), before launching block k:  _blocks_consumed >= k, or stopped
        #   get_rfimask(), for chunk i:  chunks launched > i, or (finished and i >= _nput), or stopped
        # Signaled with notify_all on every change, since the two predicates differ.
        self._cond = threading.Condition()
        self._nput = 0
        self._ngot = 0
        self._blocks_launched = 0
        self._blocks_consumed = 0
        self._finished = False
        self._stopped = False
        self._error = None

        with cp.cuda.Device(self.cuda_device_id):
            self._stream = cp.cuda.Stream(non_blocking=True)
            self._intensity = cp.empty((1, self.nfreq, self.ntime), dtype=cp.float32)
            self._weights = cp.empty((1, self.nfreq, self.ntime), dtype=cp.float32)
            self._scratch = cp.empty(int(pipeline.scratch_nelts), dtype=cp.float32)
            # Blocking-sync events sleep a waiting thread instead of spinning it.
            self._event = cp.cuda.Event(block=True, disable_timing=True)
            self._h2d_event = cp.cuda.Event(block=True, disable_timing=True)

    # ---------------------------------------------------------------------------------
    #
    # Chunk validation

    @staticmethod
    def _ichunk(chunk):
        return chunk.fpga_begin // (chunk.fpga_counts_per_sample * chunk.nt)

    def _initialize(self, chunk):
        """Fix the chunk geometry from the first chunk, and allocate what depends on it."""
        import cupy as cp

        who = 'ChimePreDedisperser'
        if chunk.nfreq != self.nfreq:
            raise ValueError(f'{who}: the chunk has {chunk.nfreq} channels, the chain {self.nfreq}')
        if self.ntime % chunk.nt != 0:
            raise ValueError(f'{who}: the chain\'s ntime ({self.ntime}) is not a multiple of the'
                             f' chunk length ({chunk.nt})')

        nt = chunk.nt
        cpb = self.ntime // nt
        if self.extractor.ntime % cpb != 0:
            raise ValueError(f'{who}: the extractor\'s ntime ({self.extractor.ntime}) does not split'
                             f' into {cpb} windows, one per chunk')
        nt_mask = self.extractor.ntime // cpb
        if nt_mask % 1024 != 0:
            raise ValueError(f'{who}: one chunk is {nt_mask} samples at the extractor\'s resolution,'
                             f' which is not a multiple of 1024 (the tiling of RfiMaskPackingKernel)')

        # The kernel's constructor checks nt_per_packet == 16 and the coarse geometry.
        self._dqk = ChimeDequantizationKernel(self.nfreq, nt, chunk.nfreq_coarse, chunk.nt_coarse)

        with cp.cuda.Device(self.cuda_device_id):
            self._g_scales = cp.empty((chunk.nfreq_coarse, chunk.nt_coarse), dtype=cp.float32)
            self._g_offsets = cp.empty((chunk.nfreq_coarse, chunk.nt_coarse), dtype=cp.float32)
            self._g_data = cp.empty((self.nfreq, nt), dtype=cp.uint8)
            self._g_mask = cp.empty((cpb, 1, self.nrfifreq, nt_mask // 8), dtype=cp.uint8)

        self.extractor.set_rfi_mask(self._g_mask)
        self.nt = nt
        self.chunks_per_block = cpb
        self.nt_mask = nt_mask
        self._first = dict(beam_id=chunk.beam_id, nupfreq=chunk.nupfreq, nt_per_packet=chunk.nt_per_packet,
                           fpga_counts_per_sample=chunk.fpga_counts_per_sample, nt=chunk.nt,
                           nfreq_coarse=chunk.nfreq_coarse, binning=chunk.binning)

    def _validate(self, chunk, j):
        """Every chunk: the arrays are present, the parameters match the first chunk's, it
        starts where the previous one ended, and (at a block start) it is aligned."""

        who = 'ChimePreDedisperser.put_chunk()'
        if chunk.metadata_only:
            raise ValueError(f'{who}: {chunk.filename!r} was read with metadata_only=True and has no arrays')
        if chunk.binning != 1:
            raise ValueError(f'{who}: {chunk.filename!r} has binning={chunk.binning}; only'
                             f' undownsampled chunks (binning 1) can be run through the chain')

        if self._first is None:
            self._initialize(chunk)
        else:
            for (key, want) in self._first.items():
                got = getattr(chunk, key)
                if got != want:
                    raise ValueError(f'{who}: {chunk.filename!r} has {key}={got}, but the first chunk'
                                     f' had {key}={want}; every chunk must come from the same'
                                     f' beam with the same parameters')

        if (self._next_fpga_begin is not None) and (chunk.fpga_begin != self._next_fpga_begin):
            gap = (int(chunk.fpga_begin) - int(self._next_fpga_begin)) / float(chunk.fpga_counts_per_sample * chunk.nt)
            raise ValueError(f'{who}: {chunk.filename!r} starts at fpga_begin={chunk.fpga_begin}, but the'
                             f' previous chunk ended at {self._next_fpga_begin} (a gap of {gap:g}'
                             f' chunks); chunks must be consecutive')

        if self.check_alignment and (j == 0):
            ichunk = self._ichunk(chunk)
            if ichunk % self.chunks_per_block != 0:
                raise ValueError(f'{who}: {chunk.filename!r} (ichunk={ichunk}) starts a block, but'
                                 f' ichunk % chunks_per_block = {ichunk % self.chunks_per_block} != 0:'
                                 f' the L1 server aligned its blocks to ichunk % {self.chunks_per_block}'
                                 f' == 0, so the masks would not correspond. Start the stream at an'
                                 f' aligned chunk, or pass check_alignment=False')

    # ---------------------------------------------------------------------------------
    #
    # Stop / error handling. _raise_if_unusable() is called with the condition held.

    def _raise_if_unusable(self, where):
        if self._error is not None:
            raise RuntimeError(f'ChimePreDedisperser.{where}(): a previous call raised'
                               f' ({self._error!r}); the GPU state may be out of sync, so construct'
                               f' a fresh ChimePreDedisperser') from self._error
        if self._stopped:
            raise RuntimeError(f'ChimePreDedisperser.{where}(): called on a stopped instance')

    def stop(self, e=None):
        """Put the instance in the stopped state and wake any thread blocked in
        :meth:`put_chunk` or :meth:`get_rfimask`, which then raises. With ``e``, an exception,
        that is what later calls report; without, they report a plain stop."""
        with self._cond:
            if (e is not None) and (self._error is None):
                self._error = e
            self._stopped = True
            self._cond.notify_all()

    @property
    def is_stopped(self):
        with self._cond:
            return self._stopped

    @property
    def nchunks_put(self):
        with self._cond:
            return self._nput

    @property
    def nchunks_got(self):
        with self._cond:
            return self._ngot

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        self.stop()
        return False

    # ---------------------------------------------------------------------------------
    #
    # The two entry points

    def put_chunk(self, chunk):
        """Feed the next chunk (producer side).

        Copies the chunk's arrays to the GPU and decodes them into the current block; every
        ``chunks_per_block``-th call also launches the chain -- after waiting until the
        consumer has taken every mask of the previous block. Returns once the chunk's arrays
        are no longer needed (the copies to the GPU are complete), so the caller may drop the
        chunk.

        Parameters
        ----------
        chunk : AssembledChunk
            Fully read (not ``metadata_only``), with the same parameters as the first chunk,
            starting where the previous one ended.

        Raises
        ------
        ValueError
            If the chunk fails the checks in the class docstring.
        RuntimeError
            If the instance is stopped, or a previous call raised.
        """
        import cupy as cp

        with self._cond:
            self._raise_if_unusable('put_chunk')
            if self._finished:
                raise RuntimeError('ChimePreDedisperser.put_chunk(): called after finish()')
            i = self._nput

        try:
            with cp.cuda.Device(self.cuda_device_id):
                # _validate() also initializes from the first chunk, which fixes chunks_per_block.
                j = 0 if (self.chunks_per_block is None) else (i % self.chunks_per_block)
                self._validate(chunk, j)
                (k, j) = divmod(i, self.chunks_per_block)
                s = self._stream
                (t0, t1) = (j * self.nt, (j + 1) * self.nt)

                self._g_scales.set(np.asarray(chunk.scales), stream=s)
                self._g_offsets.set(np.asarray(chunk.offsets), stream=s)
                self._g_data.set(np.asarray(chunk.data), stream=s)
                self._dqk.launch(self._intensity[0, :, t0:t1], self._weights[0, :, t0:t1],
                                 self._g_scales, self._g_offsets, self._g_data,
                                 scale=self.prescale, stream=s)
                self._h2d_event.record(s)

                if j == self.chunks_per_block - 1:
                    # The chain is about to overwrite the mask array: wait until the consumer
                    # has copied every mask of the previous block out of it.
                    with self._cond:
                        while self._blocks_consumed < k:
                            self._raise_if_unusable('put_chunk')
                            self._cond.wait()
                    self.pipeline.launch(self._intensity, self._weights, self._scratch, s)
                    self._event.record(s)
                    with self._cond:
                        self._blocks_launched += 1
                        self._cond.notify_all()

                self._next_fpga_begin = chunk.fpga_end
                with self._cond:
                    self._nput += 1
                    self._cond.notify_all()

                # The event was recorded before the chain was launched, so this waits for the
                # copies only -- which on one stream includes the previous block's chain, the
                # "room in the GPU buffer" this class needs before overwriting the block.
                self._h2d_event.synchronize()
        except BaseException as e:
            self.stop(e)
            raise

    def get_rfimask(self, out=None):
        """Take the next chunk's mask (consumer side).

        Blocks until the chain has run on the block containing the next chunk, then copies
        that chunk's slice of the mask out with one plain device-to-host copy.

        Parameters
        ----------
        out : numpy.ndarray or None, optional
            A uint8, C-contiguous array of shape ``(nrfifreq, nt_mask // 8)`` to write into;
            None allocates a pinned one (``cupyx.empty_pinned``).

        Returns
        -------
        numpy.ndarray or None
            The mask, bit-packed LSB-first with a SET bit meaning GOOD data (the layout of a
            data file's ``rfi_mask``): ``out``, or the new array. None once :meth:`finish` has
            been called and every mask has been taken.

        Raises
        ------
        RuntimeError
            If the instance is stopped, or a previous call raised.
        """
        import cupy as cp
        import cupyx

        with self._cond:
            self._raise_if_unusable('get_rfimask')
            i = self._ngot
            while True:
                self._raise_if_unusable('get_rfimask')
                cpb = self.chunks_per_block
                if (cpb is not None) and (self._blocks_launched * cpb > i):
                    break
                if self._finished and (i >= self._nput):
                    return None
                self._cond.wait()
            (k, j) = divmod(i, cpb)

        try:
            shape = (self.nrfifreq, self.nt_mask // 8)
            if out is None:
                out = cupyx.empty_pinned(shape, dtype=np.uint8)
            elif (not isinstance(out, np.ndarray)) or (out.dtype != np.uint8) \
                    or (tuple(out.shape) != shape) or (not out.flags.c_contiguous):
                raise ValueError(f'ChimePreDedisperser.get_rfimask(): expected out to be a C-contiguous'
                                 f' uint8 numpy array of shape {shape}, got {type(out).__name__}'
                                 + (f' {out.dtype} {tuple(out.shape)}' if isinstance(out, np.ndarray) else ''))

            with cp.cuda.Device(self.cuda_device_id):
                self._event.synchronize()             # the chain that wrote the mask array is done
                self._g_mask[j, 0].get(out=out)       # one contiguous slice, synchronous copy

            with self._cond:
                self._ngot += 1
                if j == cpb - 1:
                    self._blocks_consumed += 1
                self._cond.notify_all()
            return out
        except BaseException as e:
            self.stop(e)
            raise

    def finish(self):
        """Declare the end of the stream (producer side): :meth:`get_rfimask` returns None once
        every mask has been taken. Raises ``ValueError`` if a partial block is pending, since
        the chain runs on whole blocks only: feed whole blocks, or drop the trailing chunks
        (the rule script 01 applies by trimming the file list)."""
        with self._cond:
            self._raise_if_unusable('finish')
            cpb = self.chunks_per_block
            pending = (self._nput % cpb) if cpb else 0
            if pending:
                e = ValueError(f'ChimePreDedisperser.finish(): {pending} chunk(s) of an incomplete block'
                               f' are pending (put_chunk() was called {self._nput} times,'
                               f' chunks_per_block is {cpb}); feed whole blocks, or drop the'
                               f' trailing chunks')
                self._error = e
                self._stopped = True
                self._cond.notify_all()
                raise e
            self._finished = True
            self._cond.notify_all()

    def __repr__(self):
        return (f'ChimePreDedisperser(nfreq={self.nfreq}, ntime={self.ntime}, nt={self.nt},'
                f' prescale={self.prescale:g}, put={self.nchunks_put}, got={self.nchunks_got})')
