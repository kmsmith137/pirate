"""SlabAllocator method injections (+ re-export of the pybind11 class)."""

import ksgpu
from ..pirate_pybind11 import SlabAllocator, BumpAllocator


@ksgpu.inject_methods(SlabAllocator)
class SlabAllocatorInjections:
    # No class docstring here: SlabAllocator's docstring lives in the pybind11
    # binding (option 1 in notes/docstrings.md). inject_methods would otherwise
    # copy a docstring written here onto the class, overriding the pybind11 one.

    # Save original C++ constructor
    _cpp_init = SlabAllocator.__init__

    def __init__(self, aflags_or_bump_allocator, nbytes):
        """
        Create a SlabAllocator.

        Parameters
        ----------
        aflags_or_bump_allocator : int, str, ksgpu flags, or BumpAllocator
            Either memory allocation flags or a BumpAllocator to get memory from.
            If aflags:
            - int: raw flags (e.g., af_gpu | af_zero)
            - str: 'gpu', 'rhost', 'uhost', etc.
            - Result of ksgpu.parse_aflags()
            If BumpAllocator: memory is allocated from the BumpAllocator.
        nbytes : int
            Capacity in bytes.
            - If aflags and > 0: pre-allocates this many bytes, subdivided into slabs
            - If aflags and < 0: dummy mode (each get_slab() allocates fresh memory)
            - If aflags and == 0: rejected (raises)
            - If BumpAllocator: must be positive

        Examples
        --------
        >>> # Host allocator with 100 MB capacity
        >>> alloc = SlabAllocator('rhost', 100 * 1024**2)
        >>>
        >>> # GPU allocator with 1 GB capacity
        >>> alloc = SlabAllocator('gpu', 1024**3)
        >>>
        >>> # Dummy mode (no pre-allocation)
        >>> alloc = SlabAllocator('rhost', -1)
        >>>
        >>> # From a BumpAllocator
        >>> bump = BumpAllocator('rhost', 1024**3)
        >>> slab = SlabAllocator(bump, 100 * 1024**2)
        """
        if isinstance(aflags_or_bump_allocator, BumpAllocator):
            self._cpp_init(aflags_or_bump_allocator, nbytes)
        else:
            aflags = ksgpu.parse_aflags(aflags_or_bump_allocator)
            self._cpp_init(aflags, nbytes)
