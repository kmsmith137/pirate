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
            If True, constructor returns immediately; allocation/zeroing
            happens on worker threads. Public methods (allocate_array,
            get_base) block until init complete; failures rethrow.
            Async mode supports exactly these aflag combinations:
              - af_mmap_huge | af_rhost | af_zero (mmap + chunked register)
              - af_rhost | af_zero (cudaHostAlloc + parallel zero)
              - af_gpu | af_zero (cudaMalloc + cudaMemset)
            cuda_device is required (>= 0). nthreads >= 2 for cases 1 and 2;
            ignored for case 3 (af_gpu).
        nthreads : int
            Number of worker threads (async mode only).
        cuda_device : int
            CUDA device id (async mode only; required >= 0).

        Examples
        --------
        >>> # GPU allocator with 1 GB capacity (sync)
        >>> alloc = BumpAllocator('gpu', 1024**3)
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
            If allocation would exceed capacity (in normal mode)

        Examples
        --------
        >>> alloc = BumpAllocator('gpu', 1024**3)
        >>> arr1 = alloc.allocate_array('float32', (1024, 1024))
        >>> arr2 = alloc.allocate_array(np.int16, (512, 512, 4))
        >>> print(alloc.nbytes_allocated)  # Shows total allocated
        """
        dtype = ksgpu.Dtype(dtype)
        return self._allocate_array_raw(dtype, list(shape))
