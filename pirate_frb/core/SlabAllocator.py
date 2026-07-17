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

    def __init__(self, aflags_or_bump_allocator, nbytes=None):
        """
        Create a SlabAllocator.

        Parameters
        ----------
        aflags_or_bump_allocator : BumpAllocator, int, str, or ksgpu flags
            Either a BumpAllocator to carve memory from (normal mode), or
            memory allocation flags (dummy mode: each get_slab() allocates
            fresh memory). Flags can be:
            - int: raw flags (e.g., af_rhost | af_zero)
            - str: 'af_gpu', 'af_rhost', 'af_rhost | af_zero', etc.
            - Result of ksgpu.parse_aflags()
        nbytes : int, optional
            Bytes to carve from the BumpAllocator (must be positive).
            Required in normal mode; must be omitted in dummy mode.

        Examples
        --------
        >>> # Slab pool carved from a BumpAllocator
        >>> bump = BumpAllocator('af_rhost', 1024**3)
        >>> slab = SlabAllocator(bump, 100 * 1024**2)
        >>>
        >>> # Dummy mode (no pre-allocation)
        >>> alloc = SlabAllocator('af_rhost')
        """
        if isinstance(aflags_or_bump_allocator, BumpAllocator):
            if nbytes is None:
                raise TypeError("SlabAllocator: 'nbytes' is required when constructing from a BumpAllocator")
            self._cpp_init(aflags_or_bump_allocator, nbytes)
        else:
            if nbytes is not None:
                raise TypeError("SlabAllocator: 'nbytes' must be omitted in dummy mode (constructed from"
                                " aflags alone); to create a slab pool, pass a BumpAllocator instead")
            aflags = ksgpu.parse_aflags(aflags_or_bump_allocator)
            self._cpp_init(aflags)
