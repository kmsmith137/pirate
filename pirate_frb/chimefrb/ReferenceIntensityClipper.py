"""Numpy reference for GpuIntensityClipper.

The intensity clipper is the old CHIME FRB search's principal RFI flagger: it computes a
weighted mean and variance over an axis, then zeroes the weights of every sample more
than 'sigma' standard deviations away.

Almost nothing here is transcribed from the old C++, because the pieces it is built from
already are: ReferenceWiDownsamplingKernel and ReferenceWrmsKernel, each validated against
rf_kernels by its own spot test. What is new is intensity_clip(), the final clip and mask
upsample, which is checked against rf_kernels::intensity_clipper in
misc/chimefrb/spot_checks/rfi_intensity_clipper/.
"""

import numpy as np

from .ReferenceWiDownsamplingKernel import ReferenceWiDownsamplingKernel
from .ReferenceWrmsKernel import ReferenceWrmsKernel


# The axis is one of the three strings 'freq', 'time', 'none' -- the same spelling the GPU
# classes take, a saved yaml file carries, and C++ uses (axis_to_string() in
# include/pirate/chimefrb/ClipperAxis.hpp). The old code spells them 'freq' and so on,
# and numbers them 0, 1, 2; that numbering survives only where a spot-check driver casts an
# integer to rf_kernels::axis_type, and the old spelling only in
# chimefrb.utils.axis_from_json().


def wrms_view(arr, axis):
    """Reshape a downsampled (B, F_ds, T_ds) array to the (R, L) form GpuWrmsKernel wants.

    The clippers reduce along three axes, but by the time the statistic runs there is only
    one shape: one output per row of a contiguous 2-D array. 'none' works because a
    contiguous (B, F_ds, T_ds) array IS a (B, F_ds*T_ds) array -- reducing a whole plane is
    reducing one long row.

    Note that 'freq' is the one that moves data (the frequency axis has to become the
    fast one). GpuIntensityClipper does the same thing with GpuWiDownsamplingKernel(1,1,True).
    """

    (B, F_ds, T_ds) = arr.shape
    if axis == 'time':
        return arr.reshape(B * F_ds, T_ds)
    if axis == 'freq':
        return np.ascontiguousarray(np.swapaxes(arr, 1, 2)).reshape(B * T_ds, F_ds)
    if axis == 'none':
        return arr.reshape(B, F_ds * T_ds)

    raise RuntimeError(f'bad axis {axis!r}')


def intensity_clip(i_ds, weights, mean, var, sigma, axis, Df, Dt):
    """Zero the weights of every downsampled cell with |i_ds - mean| >= sigma*sqrt(var).

    Returns a new weights array; the argument is not modified. This is the numpy twin of
    the CUDA kernel intensity_clip_kernel() in src_lib/chimefrb/IntensityClipper.cu, and
    the two must agree.

    Parameters
    ----------
    i_ds : ndarray
        Shape (B, F_ds, T_ds). The DOWNSAMPLED intensity: a cell is masked or not as a
        unit, whatever its individual samples look like.
    weights : ndarray
        Shape (B, F, T) with F = F_ds*Df and T = T_ds*Dt. Not modified.
    mean, var : ndarray
        Shape (R,), indexed by beam and by the axis being reduced (see wrms_view()).
    sigma : float
        The FINAL clip threshold, in units of the row's rms. This is NOT the statistic's
        iter_sigma -- in the production chain they are 5 and 3 -- and confusing the two
        produces a plausible-looking wrong answer.
    axis : int
        'freq', 'time' or 'none'.

    Notes
    -----
    The survivor test is a strict '<', so a row whose variance was rejected (var == 0) has
    thresh == 0 and loses ALL of its weights, including samples exactly at the mean. That
    is intentional in the original: no usable statistic means no usable data.

    Arithmetic follows the inputs' dtype. Passing float32 arrays reproduces the CUDA
    kernel's arithmetic exactly, which is what the unit test's GPU-supplied-(mean,var)
    comparison relies on; passing float64 gives the reference's own.
    """

    i_ds = np.asarray(i_ds)
    weights = np.asarray(weights)
    var = np.asarray(var)

    (B, F_ds, T_ds) = i_ds.shape
    thresh = np.asarray(sigma, dtype=var.dtype) * np.sqrt(var)

    # Broadcast (mean, thresh) back against the downsampled cell grid. The reshape is the
    # inverse of wrms_view()'s, and is the only place the axis matters.
    ax = axis
    if ax == 'time':
        shape = (B, F_ds, 1)
    elif ax == 'freq':
        shape = (B, 1, T_ds)
    elif ax == 'none':
        shape = (B, 1, 1)
    else:
        raise RuntimeError(f'bad axis {axis}')

    survive = np.abs(i_ds - np.asarray(mean).reshape(shape)) < thresh.reshape(shape)

    # Upsample the cell mask back to full resolution, then AND it into the weights.
    mask = np.repeat(np.repeat(survive, Df, axis=1), Dt, axis=2)
    return np.where(mask, weights, 0)


class ReferenceIntensityClipper:
    """Numpy reference for GpuIntensityClipper (src_lib/chimefrb/IntensityClipper.cu).

    Same semantics and same argument names, in float64. Three steps: downsample by
    (Df, Dt), compute the weighted mean and variance over ``axis``, then clip.

    Two things are worth knowing before calling it:

    ``sigma`` and ``iter_sigma`` are different numbers. ``sigma`` is the final clip;
    ``iter_sigma`` is used inside the statistic's refinements. In the production chain
    they are 5 and 3.

    This class implements ``T = N*nt_chunk``, which GpuIntensityClipper does not (it
    requires exactly one chunk). The definition is the one that removes all ambiguity:
    ``apply()`` on a (B, F, N*nt_chunk) array returns exactly what N separate calls on
    the (B, F, nt_chunk) sub-arrays would return, concatenated along the time axis. That
    is what the old pipeline does -- rf_pipelines hands intensity_clipper::clip() one
    nt_chunk block at a time -- and the sub-chunks are completely independent, so there
    is no ordering or carry-over question to resolve.

    Two consequences of that, since a reader will wonder:

    * For ``axis = 'freq'`` the answer does not depend on ``nt_chunk`` at all. Each
      time column is its own statistic, so chunking the time axis changes nothing. Only
      'time' and 'none' see the boundaries.
    * ``nt_chunk`` must be a multiple of ``Dt``, so that a downsampled cell cannot
      straddle a boundary, and ``T`` must be a multiple of ``nt_chunk``.

    Unlike ReferenceWrmsKernel there is no ``dtype`` argument: this reference is float64
    throughout, because its job is to be more accurate than the kernel it checks. The
    free function intensity_clip() is dtype-agnostic, which is what a float32 comparison
    against the kernel's own (mean, var) needs.
    """

    def __init__(self, axis, sigma, Df, Dt, niter, iter_sigma, two_pass,
                 nt_chunk=None, eps_multiplier=1.0):
        assert axis in ('freq', 'time', 'none')
        assert Df >= 1 and Dt >= 1
        assert niter >= 1
        assert sigma >= 0.0 and iter_sigma >= 0.0

        self.axis = axis
        self.sigma = float(sigma)
        self.Df = int(Df)
        self.Dt = int(Dt)
        self.niter = int(niter)
        self.iter_sigma = float(iter_sigma)
        self.two_pass = bool(two_pass)
        self.nt_chunk = None if (nt_chunk is None) else int(nt_chunk)
        self.eps_multiplier = float(eps_multiplier)

    def apply(self, in_i, in_w):
        """Returns the clipped weights, shape (B, F, T) float64. Arguments not modified."""

        assert in_i.shape == in_w.shape
        assert in_i.ndim == 3

        (B, F, T) = in_i.shape
        nt = T if (self.nt_chunk is None) else self.nt_chunk

        assert T % nt == 0, 'T must be a multiple of nt_chunk'
        assert F % self.Df == 0
        assert nt % self.Dt == 0, 'a downsampled cell must not straddle a chunk boundary'

        out = np.empty((B, F, T), dtype=np.float64)

        for s in range(T // nt):
            sl = slice(s * nt, (s+1) * nt)
            out[:, :, sl] = self._apply_chunk(in_i[:, :, sl], in_w[:, :, sl])

        return out

    def _apply_chunk(self, in_i, in_w):
        """One nt_chunk block: downsample, compute the statistic, clip."""

        I = np.asarray(in_i, dtype=np.float64)
        W = np.asarray(in_w, dtype=np.float64)

        (i_ds, w_ds) = ReferenceWiDownsamplingKernel(self.Df, self.Dt, transpose=False).apply(I, W)

        wrms = ReferenceWrmsKernel(self.niter, self.iter_sigma, self.two_pass,
                                   eps_multiplier=self.eps_multiplier)
        (mean, var) = wrms.apply(wrms_view(i_ds, self.axis), wrms_view(w_ds, self.axis))

        # Note sigma here, not iter_sigma: the refinements clipped at iter_sigma, and this
        # final clip uses a different threshold. The old code comments on this at each of
        # its six call sites, which is a fair measure of how easy it is to get wrong.
        return intensity_clip(i_ds, W, mean, var, self.sigma, self.axis, self.Df, self.Dt)
