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

    def __init__(self, aflags_or_bump_allocator):
        """
        Create a SlabAllocator.

        Parameters
        ----------
        aflags_or_bump_allocator : BumpAllocator, int, str, or ksgpu flags
            Either a BumpAllocator to carve slabs from (normal mode: slabs
            are carved on demand, one per get_slab() call, until the
            BumpAllocator is exhausted), or memory allocation flags (dummy
            mode: each get_slab() allocates fresh memory). Flags can be:
            - int: raw flags (e.g., af_rhost | af_zero)
            - str: 'af_gpu', 'af_rhost', 'af_rhost | af_zero', etc.
            - Result of ksgpu.parse_aflags()

        Examples
        --------
        >>> # Slab pool drawing from a BumpAllocator
        >>> bump = BumpAllocator('af_rhost', 1024**3)
        >>> slab = SlabAllocator(bump)
        >>>
        >>> # Dummy mode (no pre-allocation)
        >>> alloc = SlabAllocator('af_rhost')
        """
        if isinstance(aflags_or_bump_allocator, BumpAllocator):
            self._cpp_init(aflags_or_bump_allocator)
        else:
            aflags = ksgpu.parse_aflags(aflags_or_bump_allocator)
            self._cpp_init(aflags)
