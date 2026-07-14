"""GpuDequantizationKernel method injections (+ re-export of the pybind11 class)."""

import ksgpu
from ..pirate_pybind11 import GpuDequantizationKernel


@ksgpu.inject_methods(GpuDequantizationKernel)
class GpuDequantizationKernelInjections:
    # No class docstring here: GpuDequantizationKernel's docstring lives in the
    # pybind11 binding (option 1 in notes/docstrings.md); this injector adds dtype
    # conversion for the constructor and a stream argument for launch().

    # Save references to C++ methods
    _cpp_init = GpuDequantizationKernel.__init__
    _cpp_launch = GpuDequantizationKernel.launch

    def __init__(self, dtype, nbeams, nfreq, ntime):
        """Create a GpuDequantizationKernel.

        Applies the per-(beam, freq, minichunk) affine transform

            out[b,f,t] = 0                                       if data[b,f,t] == -8
            out[b,f,t] = scales_offsets[b,f,t//256,0] * data[b,f,t]
                       + scales_offsets[b,f,t//256,1]            otherwise

        during int4 -> float32/float16 conversion. data == -8 is the
        "missing sample" sentinel (matching the convention of
        AssembledFrame.data) and is always mapped to 0 in the output,
        regardless of scale and offset.

        Parameters
        ----------
        dtype : str, numpy.dtype, cupy.dtype, or ksgpu.Dtype
            Output dtype (must be float32 or float16)
        nbeams : int
            Number of beams
        nfreq : int
            Number of frequency channels
        ntime : int
            Number of time samples (must be divisible by 256)
        """
        dtype = ksgpu.Dtype(dtype)
        self._cpp_init(dtype, nbeams, nfreq, ntime)

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
