"""Numpy reference for GpuWtUpsamplingKernel.

The reference is transcribed from reference_weight_upsample() in
../../extern/rf_kernels/test-upsample.cpp, the old code's own scalar reference for
rf_kernels::weight_upsampler, which the old unit test compares with the AVX2 kernel exactly.
"""

import numpy as np


class ReferenceWtUpsamplingKernel:
    """Numpy reference for GpuWtUpsamplingKernel (src_lib/chimefrb/WtUpsamplingKernel.cu): zeroes
    every full-resolution weight whose ``(Df, Dt)`` cell has a low-resolution weight
    ``w_lo <= w_cutoff``, and leaves every other weight bit-identical.

    Three things a reader might take for bugs, all of them the old code's behaviour:

    - The comparison is strict: a low-resolution weight equal to the cutoff masks its cell.
    - A NaN low-resolution weight masks its cell, since ``NaN > w_cutoff`` is false.
    - The comparison is in float32, against ``float32(w_cutoff)``, since the old code takes
      the cutoff as a float. The cast matters for a cutoff float32 cannot represent, such as
      0.1: a weight equal to ``float32(0.1)`` masks, where a float64 comparison would keep it.
    """

    def __init__(self, Df, Dt, w_cutoff=0.0):
        assert (Df >= 1) and (Dt >= 1) and (w_cutoff >= 0)
        self.Df = int(Df)
        self.Dt = int(Dt)
        self.w_cutoff = float(w_cutoff)

    def apply(self, w_hires, w_lores):
        """Returns a copy of 'w_hires', shape (B, F_lo*Df, T_lo*Dt), with every masked cell
        set to +0.0. Does not modify its arguments."""

        w_lores = np.asarray(w_lores, dtype=np.float32)
        assert w_lores.ndim == 3
        (B, F_lo, T_lo) = w_lores.shape
        assert np.shape(w_hires) == (B, F_lo * self.Df, T_lo * self.Dt)

        # The cast is what makes this a float32 comparison, as the old code's is. Do not
        # drop it: on numpy >= 2 a bare python float happens to give the same answer (NEP
        # 50 makes it a weak type, so the array is not upcast) and no test would catch the
        # removal, but against a float64 cutoff it changes which cells are masked.
        keep = w_lores > np.float32(self.w_cutoff)
        keep = np.repeat(np.repeat(keep, self.Df, axis=1), self.Dt, axis=2)

        # np.where copies the kept elements without arithmetic, so NaN payloads and -0.0
        # survive bit for bit, and the masked ones become +0.0.
        return np.where(keep, w_hires, np.float32(0.0))
