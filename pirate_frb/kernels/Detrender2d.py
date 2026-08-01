"""Detrender2d method injections (+ re-export of the pybind11 class)."""

import ksgpu
from ..pirate_pybind11 import Detrender2d


@ksgpu.inject_methods(Detrender2d)
class Detrender2dInjections:
    # No class docstring here: Detrender2d's docstring lives in the pybind11
    # binding (option 1 in notes/docstrings.md); this injector adds a stream
    # argument for launch().

    # Save reference to C++ method
    _cpp_launch = Detrender2d.launch

    def launch(self, data, mask, stream=None):
        """GPU kernel launch (async, does not sync stream).

        Parameters
        ----------
        data : ksgpu.Array
            Shape (M, nfreq, nbuf), dtype float32, fully contiguous, on GPU.
            Modified in place: buffer samples [W, W+T) of each row are
            replaced by the detrended residual, and the 2W padding samples
            are left untouched.
        mask : ksgpu.Array
            Shape (M, nfreq, nbuf), dtype uint8, fully contiguous, on GPU,
            {0,1}-valued. Modified in place over the same range, and the
            output mask is the authoritative one (see the class docstring).
        stream : cupy.cuda.Stream or None, optional
            CUDA stream to use. If None, uses current cupy stream.
        """
        import cupy as cp

        if stream is None:
            stream = cp.cuda.get_current_stream()

        self._cpp_launch(data, mask, stream.ptr)
