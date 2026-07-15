"""GpuDequantizationKernel method injections (+ re-export of the pybind11 class)."""

import ksgpu
from ..pirate_pybind11 import GpuDequantizationKernel


@ksgpu.inject_methods(GpuDequantizationKernel)
class GpuDequantizationKernelInjections:
    # No class docstring here: GpuDequantizationKernel's docstring lives in the
    # pybind11 binding (option 1 in notes/docstrings.md); this injector adds a
    # stream argument for launch().

    # Save reference to C++ method
    _cpp_launch = GpuDequantizationKernel.launch

    def launch(self, out, scales_offsets, data_uint8, stream=None):
        """GPU kernel launch (async, does not sync stream).

        Parameters
        ----------
        out : ksgpu.Array
            Output array, shape (nbeams, nfreq, ntime), dtype matches
            kernel's dtype (float32 or float16), fully contiguous, on GPU.
        scales_offsets : ksgpu.Array
            Shape (nbeams, nfreq, ntime//256, 2), dtype float16, fully
            contiguous, on GPU. Last axis is (scale, offset); one pair is
            applied to every int4 sample in the matching (beam, freq,
            minichunk) slice of data_uint8.
        data_uint8 : ksgpu.Array
            Shape (nbeams, nfreq, ntime//2), dtype uint8, fully contiguous,
            on GPU. Reinterpreted as int4 with shape (nbeams, nfreq, ntime).
        stream : cupy.cuda.Stream or None, optional
            CUDA stream to use. If None, uses current cupy stream.

        Note
        ----
        The data array is passed as uint8 because numpy/cupy don't support int4
        (all dtypes must be at least 8 bits). Each uint8 element contains two
        int4 values: low nibble = even index, high nibble = odd index.

        data == -8 is mapped to 0 in the output regardless of scale and
        offset; see the __init__ docstring above.
        """
        import cupy as cp

        if stream is None:
            stream = cp.cuda.get_current_stream()

        self._cpp_launch(out, scales_offsets, data_uint8, stream.ptr)
