"""
Method injections for pybind11-wrapped pirate classes.
Extends C++ classes with Python convenience methods.
"""

import ksgpu
from . import pirate_pybind11


@ksgpu.inject_methods(pirate_pybind11.BumpAllocator)
class BumpAllocatorInjections:
    """Python extensions for BumpAllocator."""
    
    # Save original C++ constructor
    _cpp_init = pirate_pybind11.BumpAllocator.__init__
    
    def __init__(self, aflags, capacity):
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
        
        Examples
        --------
        >>> # GPU allocator with 1 GB capacity
        >>> alloc = BumpAllocator('gpu', 1024**3)
        >>> 
        >>> # Host allocator (registered) with 100 MB
        >>> alloc = BumpAllocator('rhost', 100 * 1024**2)
        >>> 
        >>> # Dummy mode
        >>> alloc = BumpAllocator('gpu', -1)
        """
        aflags = ksgpu.parse_aflags(aflags)
        self._cpp_init(aflags, capacity)
    
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

