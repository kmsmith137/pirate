"""BumpAllocator method injections (+ re-export of the pybind11 class)."""

import ksgpu
from ..pirate_pybind11 import BumpAllocator, constants


@ksgpu.inject_methods(BumpAllocator)
class BumpAllocatorInjections:
    # No class docstring here: BumpAllocator's docstring lives in the pybind11
    # binding (option 1 in notes/docstrings.md). inject_methods would otherwise
    # copy a docstring written here onto the class, overriding the pybind11 one.

    # Save original C++ constructor
    _cpp_init = BumpAllocator.__init__

    def __init__(self, aflags, capacity, is_async=False, nthreads=0, cuda_device=-1):
        """
        Create a BumpAllocator.

        Parameters
        ----------
        aflags : int, str, or ksgpu flags
            Memory allocation flags. Can be:
            - int: raw flags (e.g., af_gpu | af_zero)
            - str: 'gpu', 'rhost', 'uhost', etc.
            - Result of ksgpu.parse_aflags()
        capacity : int
            Capacity in bytes.
            - If >= 0: pre-allocates this many bytes
            - If < 0: dummy mode (each array gets independent allocation)
        is_async : bool
            If True, the constructor returns immediately; zeroing and (for
            af_rhost) the chunked cudaHostRegister run on worker threads.
            Public methods (e.g. allocate_array) block until init completes;
            async-init failures rethrow. Requires capacity >= 0 (no dummy
            mode). Any aflags combination is allowed; combinations with no
            async work to do (e.g. af_uhost or af_gpu without af_zero)
            initialize instantly.
        nthreads : int
            Async worker threads. Required >= 2 for async af_rhost + af_zero
            (1 registrar + >= 1 zero worker), >= 1 for async af_uhost +
            af_zero; otherwise ignored.
        cuda_device : int
            CUDA device id. Required >= 0 whenever af_gpu is set with
            capacity > 0 (sync or async), and for af_rhost in async mode;
            otherwise ignored. (For sync af_rhost, the chunked register
            runs against the caller's current CUDA device.)

        Examples
        --------
        >>> # GPU allocator with 1 GB capacity (sync)
        >>> alloc = BumpAllocator('gpu', 1024**3, cuda_device=0)
        >>>
        >>> # Async host allocator with hugepages
        >>> alloc = BumpAllocator(ksgpu.af_mmap_huge | ksgpu.af_rhost | ksgpu.af_zero,
        ...                       100 * 1024**3, is_async=True, nthreads=8, cuda_device=0)
        >>> alloc.wait_until_initialized()
        """
        aflags = ksgpu.parse_aflags(aflags)
        self._cpp_init(aflags, capacity, is_async, nthreads, cuda_device)

    # Save the raw C++ binding (same pattern as _cpp_init above); the
    # injected wait_until_initialized() below shadows it.
    _cpp_wait_until_initialized = BumpAllocator.wait_until_initialized

    def wait_until_initialized(self, timeout_ms=-1):
        """
        Block until async init completes, fails, or the timeout elapses.

        Returns True once the allocator is initialized, or False if
        timeout_ms elapsed first. If async init failed (or the allocator was
        stopped), the stored error is re-raised. In sync mode, returns True
        immediately. timeout_ms < 0 (the default) waits indefinitely;
        timeout_ms == 0 is a non-blocking poll.

        The wait is driven in constants.default_poll_cadence_ms steps: the
        GIL is released during each step and reacquired between steps, so
        Ctrl-C stays responsive even during multi-minute inits. (The raw C++
        binding, saved as _cpp_wait_until_initialized, blocks signal
        delivery for its whole duration.)
        """
        step = constants.default_poll_cadence_ms
        if timeout_ms < 0:
            while not self._cpp_wait_until_initialized(step):
                pass    # KeyboardInterrupt is delivered here, between steps
            return True
        remaining = int(timeout_ms)
        while True:
            chunk = min(step, remaining)
            if self._cpp_wait_until_initialized(chunk):
                return True
            remaining -= chunk
            if remaining <= 0:
                return False

    def allocate_array(self, dtype, shape):
        """
        Allocate an array from this allocator.

        The array's lifetime is tied to the allocator's base memory region.
        All allocations are aligned to 128-byte cache lines.

        Parameters
        ----------
        dtype : str, numpy.dtype, cupy.dtype, or ksgpu.Dtype
            Data type. Examples: 'float32', np.int64, cp.complex64
        shape : list or tuple of int
            Array dimensions

        Returns
        -------
        ksgpu.Array
            Allocated array backed by this allocator's memory

        Raises
        ------
        RuntimeError
            If allocation would exceed capacity (in normal mode). Per the
            strict stoppable-class policy, any allocate_array() failure
            also puts the allocator in the stopped state.

        Examples
        --------
        >>> alloc = BumpAllocator('gpu', 1024**3, cuda_device=0)
        >>> arr1 = alloc.allocate_array('float32', (1024, 1024))
        >>> arr2 = alloc.allocate_array(np.int16, (512, 512, 4))
        >>> print(alloc.nbytes_allocated)  # Shows total allocated
        """
        dtype = ksgpu.Dtype(dtype)
        return self._allocate_array_raw(dtype, list(shape))
