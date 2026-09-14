"""Numpy reference for GpuWiDownsamplingKernel.

The reference is transcribed from reference_wi_downsample() in
../../extern/rf_kernels/test-downsample.cpp, which is the old code's own scalar
reference for rf_kernels::wi_downsampler.
"""

import numpy as np


class ReferenceWiDownsamplingKernel:
    """Numpy reference for GpuWiDownsamplingKernel (src_lib/chimefrb/WiDownsamplingKernel.cu).

    Same semantics, same argument names. Two things to know:

    Note the weight normalization, which is the one place the old code has two
    conventions: out_w is the SUM of the cell's weights, not its mean.

    And where a cell's weights sum to zero, the downsampled intensity is undefined --
    every consumer multiplies it by that zero weight. We write 0 there. rf_kernels
    writes 0 too, except at (Df,Dt) = (1,1), where its memcpy short-circuit passes the
    raw intensity through instead. misc/chimefrb/spot_checks/rfi_wi_downsample/ measures this.

    Where a (1,1) cell has weight, its intensity is copied through exactly, not recomputed
    as (w*i)/w; see GpuWiDownsamplingKernel.
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

        if (self.Df, self.Dt) == (1, 1):
            # A (1,1) cell is one sample, copied through exactly (see the class docstring).
            out_w = in_w.astype(np.float64)
            out_i = np.where(out_w > 0, in_i.astype(np.float64), 0.0)
        else:
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
