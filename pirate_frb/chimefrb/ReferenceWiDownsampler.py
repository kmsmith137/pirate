"""Numpy reference for GpuWiDownsampler, plus that class's method injections.

The reference is transcribed from reference_wi_downsample() in
../../extern/rf_kernels/test-downsample.cpp, which is the old code's own scalar
reference for rf_kernels::wi_downsampler.
"""

import numpy as np

import ksgpu
from ..pirate_pybind11 import GpuWiDownsampler


@ksgpu.inject_methods(GpuWiDownsampler)
class GpuWiDownsamplerInjections:
    # No class docstring here: GpuWiDownsampler's docstring lives in the pybind11
    # binding (option 1 in notes/docstrings.md); this injector adds a stream
    # argument for launch().

    # Save reference to C++ method
    _cpp_launch = GpuWiDownsampler.launch

    def launch(self, out_i, out_w, in_i, in_w, stream=None):
        """GPU kernel launch (async, does not sync stream).

        Parameters
        ----------
        out_i, out_w : cupy.ndarray
            Shape (B, F//Df, T//Dt), or (B, T//Dt, F//Df) if ``transpose``.
            Float32, fully contiguous, on GPU. Fully overwritten.
        in_i, in_w : cupy.ndarray
            Shape (B, F, T), float32, fully contiguous, on GPU. Read only, and
            must not alias the output arrays. ``in_w`` must be >= 0.
        stream : cupy.cuda.Stream or None, optional
            CUDA stream to use. If None, uses current cupy stream.
        """
        import cupy as cp

        if stream is None:
            stream = cp.cuda.get_current_stream()

        self._cpp_launch(out_i, out_w, in_i, in_w, stream.ptr)


class ReferenceWiDownsampler:
    """Numpy reference for GpuWiDownsampler (src_lib/chimefrb/WiDownsampler.cu).

    Same semantics, same argument names. Two things to know:

    Note the weight normalization, which is the one place the old code has two
    conventions: out_w is the SUM of the cell's weights, not its mean.

    And where a cell's weights sum to zero, the downsampled intensity is undefined --
    every consumer multiplies it by that zero weight. We write 0 there. rf_kernels
    writes 0 too, except at (Df,Dt) = (1,1), where its memcpy short-circuit passes the
    raw intensity through instead. misc/chimefrb/rfi_wi_downsample/ measures this.
    """

    def __init__(self, Df, Dt, transpose):
        assert Df >= 1 and Dt >= 1
        self.Df = int(Df)
        self.Dt = int(Dt)
        self.transpose = bool(transpose)

    def apply(self, in_i, in_w):
        """Returns (out_i, out_w), given (B,F,T) arrays. Does not modify its arguments."""

        assert in_i.shape == in_w.shape
        assert in_i.ndim == 3

        (B, F, T) = in_i.shape
        assert (F % self.Df == 0) and (T % self.Dt == 0)
        (F_ds, T_ds) = (F // self.Df, T // self.Dt)

        # Reshape splits each axis into (coarse, fine), and the sum is over the two
        # fine axes. Done in float64 so that the reference is more accurate than the
        # kernel it is checking, not equally inaccurate.
        w = in_w.astype(np.float64).reshape(B, F_ds, self.Df, T_ds, self.Dt)
        wi = w * in_i.astype(np.float64).reshape(B, F_ds, self.Df, T_ds, self.Dt)

        out_w = w.sum(axis=(2, 4))
        out_i = np.where(out_w > 0, wi.sum(axis=(2, 4)) / np.where(out_w > 0, out_w, 1), 0.0)

        if self.transpose:
            out_i = np.ascontiguousarray(np.swapaxes(out_i, 1, 2))
            out_w = np.ascontiguousarray(np.swapaxes(out_w, 1, 2))

        return (out_i, out_w)
