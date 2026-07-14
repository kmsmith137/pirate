"""BumpAllocator method injections (+ re-export of the pybind11 class)."""

import ksgpu
from ..pirate_pybind11 import BumpAllocator


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
