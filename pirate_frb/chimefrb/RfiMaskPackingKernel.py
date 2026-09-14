"""Python method injections for RfiMaskPackingKernel (a C++ class, see
src_lib/chimefrb/RfiMaskPackingKernel.cu).

In a file of its own, rather than in cpp_transforms.py with the other kernels, because this
is the step AFTER the transforms -- it turns the weights a chain leaves behind into the
packed mask a data file carries -- rather than part of the transform interface. Same
reasoning as ChimeDequantizationKernel.py, which is the step before.
"""

import ksgpu

from ..pirate_pybind11 import RfiMaskPackingKernel


@ksgpu.inject_methods(RfiMaskPackingKernel)
class RfiMaskPackingKernelInjections:
    # No class docstring here: RfiMaskPackingKernel's docstring lives in the pybind11
    # binding (option 1 in notes/docstrings.md); this injector only adds a stream default
    # to launch().

    # Save reference to C++ method
    _cpp_launch = RfiMaskPackingKernel.launch

    def launch(self, rfi_mask, weights, stream=None):
        """GPU kernel launch (async, does not sync stream).

        Parameters
        ----------
        rfi_mask : cupy.ndarray
            Shape ``(nfreq, nt/8)``, uint8, FULLY CONTIGUOUS, on GPU. Fully overwritten,
            bit-packed LSB-first with a SET bit meaning GOOD data.
        weights : cupy.ndarray
            Shape ``(nfreq, nt)``, float32, on GPU. Read only, and must not be the same array
            as ``rfi_mask``. PARTIALLY CONTIGUOUS: the time stride must be 1, but the
            frequency stride is free (``>= nt``), so this may be a column slice of a larger
            block.
        stream : cupy.cuda.Stream or None, optional
            CUDA stream to use. If None, uses current cupy stream.
        """
        import cupy as cp

        if stream is None:
            stream = cp.cuda.get_current_stream()

        self._cpp_launch(rfi_mask, weights, stream.ptr)
