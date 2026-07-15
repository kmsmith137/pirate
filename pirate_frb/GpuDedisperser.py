"""GpuDedisperser method injections (+ re-export of the pybind11 class)."""

from contextlib import contextmanager

import ksgpu
from .pirate_pybind11 import GpuDedisperser


@ksgpu.inject_methods(GpuDedisperser)
class GpuDedisperserInjections:
    """Low-level C++ GPU dedisperser class, with a python context-manager interface.

    This is probably not the class you want -- you probably want OfflineDedisperser!
    The OfflineDedisperser is a wrapper which handles details like memory allocation
    and parsing AssembledFrames.

    GpuDedisperser has two interfaces: a high-level context manager interface::

        self.get_input(seq_id) -> (context manager)
        self.get_output(seq_id) -> (context manager)

    and a lower-level interface which is closer to C++::

        self._acquire_input(seq_id, stream_ptr) -> (cupy array)
        self._release_input_and_launch_dd_kernels(seq_id, stream_ptr) -> None
        self._acquire_output(consumer_id, seq_id, stream_ptr) -> GpuDedisperserOutputs
        self._release_output(consumer_id, seq_id, stream_ptr) -> None

    Processing is one seq_id at a time (seq_id = ichunk*nbatches + ibatch).

    WARNING: don't use input/output arrays outside their context managers -- otherwise
    you'll get a silent race condition!! (The input/output arrays are views into a
    GPU memory ring buffer, and will be overwritten soon after context manager exit.)

    Example code::

        dd = GpuDedisperser(plan, stream_pool, cuda_device_id=0, num_consumers=1)
        dd.allocate(gpu_allocator, host_allocator)
        dd.fill_analytic_weights(freq_variances)   # so out_max comes out as an SNR

        for seq_id in range(nchunks * dd.nbatches):
            # Write this beam-batch's input; leaving the block launches the
            # dedispersion kernels. in_arr has shape (beams_per_batch, nfreq, nt_in).
            with dd.get_input(seq_id) as in_arr:
                in_arr[:] = chunk_data             # e.g. a cupy array

            # Read this seq_id's outputs (only valid inside the block).
            with dd.get_output(seq_id) as outputs:
                for itree in range(dd.ntrees):
                    out_max = outputs.out_max[itree]        # (beams_per_batch, ndm, nt)
                    out_argmax = outputs.out_argmax[itree]
                    ...

    Stream semantics. All synchronization is via CUDA events on the stream you pass
    to get_input()/get_output() (default: cupy's current stream, captured at context
    manager entry). The context managers never synchronize the host with the GPU:

    - get_input(): entry makes 'stream' wait until the input slot is free; exit
      records an event on 'stream' which the dedispersion kernels wait on. So all
      writes to the input array must be ENQUEUED ON 'stream', inside the block.
      They need only be enqueued (not finished) by exit. The buffer holds stale
      data from an earlier chunk -- overwrite it completely.

    - get_output(): entry makes 'stream' wait until the outputs for seq_id are
      ready. (The host may block until the producing kernel has been *launched*,
      but never waits for it to *finish* -- so at entry the outputs are visible
      only to work enqueued on 'stream', not to the host or other streams.) Exit
      records an event on 'stream'; the ring slot is reused only after that event,
      so reads also need only be enqueued (not finished) by exit.

    Gotchas which follow from the above:

    - GPU work submitted on any OTHER stream inside a block is unordered in both
      directions: it can read outputs before the kernels finish, and the ring slot
      can be recycled while it is still running. With default cupy usage, just
      don't change the current stream inside the block.

    - A GPU->host copy of the outputs enqueued on the context manager's stream is
      safe with no host sync (slot reuse is stream-ordered behind it), but the
      HOST buffer is not valid until the copy actually finishes. Blocking copies
      (cupy's arr.get() / cp.asnumpy(), blocking=True by default) handle this;
      after arr.get(blocking=False) you must stream.synchronize() before reading
      the numpy array.

    - Back-pressure: exiting the get_input(seq_id) block blocks the host until
      output (seq_id - nbatches_out) has been released. A single thread driving
      both inputs and outputs (as in the example below) must therefore not let
      output releases lag input submissions by more than nbatches_out batches,
      or it will deadlock.
    """
    # This class docstring (above) is the GpuDedisperser docstring: the pybind11
    # binding deliberately sets none, and inject_methods copies this one onto the
    # class (option 2 in notes/docstrings.md).

    # The get_input()/get_output() context managers below are the Python interface.
    # They wrap the low-level C++ acquire/release methods, which are bound (with a
    # leading underscore, to mark them internal) in pirate_pybind11.cpp:
    #     self._acquire_input(seq_id, stream_ptr)
    #     self._release_input_and_launch_dd_kernels(seq_id, stream_ptr)
    #     self._acquire_output(consumer_id, seq_id, stream_ptr)
    #     self._release_output(consumer_id, seq_id, stream_ptr)
    # These take a raw cudaStream_t pointer (not a cupy stream); call them directly
    # only if you need low-level acquire/release control outside the context managers.

    @contextmanager
    def get_input(self, seq_id, stream=None):
        """Context manager for acquiring and releasing input buffer.

        Acquires the input buffer on entry and releases it on exit. Entry makes
        'stream' wait until the slot is free; exit records an event on 'stream'
        and launches the dedispersion kernels behind it. All writes to the buffer
        must be enqueued on 'stream' inside the block (enqueued suffices -- they
        need not have completed by exit). The buffer holds stale data from an
        earlier chunk: overwrite it completely. See the class docstring ("Stream
        semantics") for gotchas.

        Note: the release-and-launch happens in a 'finally', so leaving the block
        by RAISING an exception still launches the dedispersion kernels on whatever
        is in the input buffer (partially-written or stale). This keeps the internal
        seq_id cursors consistent (the alternative -- skipping the launch -- would
        desync them), at the cost of producing garbage outputs for this seq_id
        rather than an error. So don't rely on an exception here to abort the chunk.

        Parameters
        ----------
        seq_id : int
            Global batch index 0, 1, 2, ... (= ichunk*nbatches + ibatch).
        stream : cupy.cuda.Stream or None, optional
            CUDA stream to use. If None, uses current cupy stream.

        Yields
        ------
        ksgpu.Array
            Input buffer array that can be used as a cupy array.

        Examples
        --------
        >>> g = GpuDedisperser(plan, stream_pool, cuda_device_id=0)
        >>> g.allocate(gpu_alloc, host_alloc)
        >>> with g.get_input(seq_id=0) as arr:
        ...     arr[:] = input_data  # write to the buffer
        """
        import cupy as cp
        if stream is None:
            stream = cp.cuda.get_current_stream()
        arr = self._acquire_input(seq_id, stream.ptr)
        try:
            yield arr
        finally:
            self._release_input_and_launch_dd_kernels(seq_id, stream.ptr)

    @contextmanager
    def get_output(self, seq_id, stream=None, consumer_id=0):
        """Context manager for acquiring and releasing output buffer.

        Acquires the output buffer on entry and yields the Outputs object;
        releases the buffer on exit. Entry makes 'stream' wait until the outputs
        are ready (the host does not wait for the producing kernel to finish, so
        at entry the outputs are visible only to work enqueued on 'stream'). Exit
        records an event on 'stream'; the ring slot is reused only after that
        event, so reads need only be enqueued (not completed) by exit. A GPU->host
        copy enqueued on 'stream' is therefore race-free, but its host buffer is
        only valid once the copy finishes (cupy's arr.get() blocks by default;
        after a non-blocking copy, synchronize the stream before reading it). See
        the class docstring ("Stream semantics") for gotchas.

        Parameters
        ----------
        seq_id : int
            Global batch index 0, 1, 2, ... (= ichunk*nbatches + ibatch).
        stream : cupy.cuda.Stream or None, optional
            CUDA stream to use. If None, uses current cupy stream.
        consumer_id : int, optional
            Output consumer id in [0, num_consumers). Defaults to 0, which
            is correct for the typical num_consumers=1 case.

        Yields
        ------
        GpuDedisperserOutputs
            Object with out_max and out_argmax attributes (see docstring)

        Examples
        --------
        >>> g = GpuDedisperser(plan, stream_pool, cuda_device_id=0, num_consumers=1)
        >>> g.allocate(gpu_alloc, host_alloc)
        >>> with g.get_output(seq_id=0) as outputs:
        ...     for itree in range(g.ntrees):
        ...         process_tree(outputs.out_max[itree], outputs.out_argmax[itree])
        """
        import cupy as cp
        if stream is None:
            stream = cp.cuda.get_current_stream()
        outputs = self._acquire_output(consumer_id, seq_id, stream.ptr)
        try:
            yield outputs
        finally:
            self._release_output(consumer_id, seq_id, stream.ptr)
