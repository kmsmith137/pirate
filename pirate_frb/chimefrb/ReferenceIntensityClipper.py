"""Numpy reference for GpuIntensityClipper.

The intensity clipper is the old CHIME FRB search's principal RFI flagger: it computes a
weighted mean and variance over an axis, then zeroes the weights of every sample more
than 'sigma' standard deviations away.

Almost nothing here is transcribed from the old C++, because the pieces it is built from
already are: ReferenceWiDownsampler and ReferenceWrms, each validated against rf_kernels
by its own spot test. What is new is intensity_clip(), the final clip and mask upsample,
which is checked against rf_kernels::intensity_clipper in misc/chimefrb/rfi_intensity_clipper/.
"""

import numpy as np

import ksgpu
from ..pirate_pybind11 import ClipperAxis, GpuIntensityClipper
from .ReferenceWiDownsampler import ReferenceWiDownsampler
from .ReferenceWrms import ReferenceWrms


# Axis codes. These three numbers are shared by rf_kernels::axis_type (core.hpp),
# pirate::chimefrb::ClipperAxis, and this module, deliberately: a spot-check driver casts
# an integer straight to the old enum, and int(ClipperAxis.FREQ) == AXIS_FREQ here.
AXIS_FREQ = 0    # one statistic per downsampled time sample, reducing over frequency
AXIS_TIME = 1    # one statistic per downsampled frequency, reducing over time
AXIS_NONE = 2    # one statistic per beam, reducing over the whole plane

# Checked here rather than trusted, since a silent drift would make the spot-check driver
# run a different transform from the one the test thinks it asked for.
assert (int(ClipperAxis.FREQ), int(ClipperAxis.TIME), int(ClipperAxis.NONE)) == \
    (AXIS_FREQ, AXIS_TIME, AXIS_NONE)


@ksgpu.inject_methods(GpuIntensityClipper)
class GpuIntensityClipperInjections:
    # No class docstring here: GpuIntensityClipper's docstring lives in the pybind11
    # binding (option 1 in notes/docstrings.md); this injector adds a stream argument
    # for launch(), and lets the caller omit the scratch array.

    # Save reference to C++ method
    _cpp_launch = GpuIntensityClipper.launch

    def launch(self, intensity, weights, scratch=None, stream=None):
        """GPU kernel launch (async, does not sync stream).

        Parameters
        ----------
        intensity : cupy.ndarray
            Shape (B, F, nt_chunk), float32, fully contiguous, on GPU. Read only.
        weights : cupy.ndarray
            Same shape and dtype. MODIFIED IN PLACE: zeroed where the clip fires, and
            bit-identical everywhere else. Must be >= 0 on entry.
        scratch : cupy.ndarray or None, optional
            Shape ``(self.scratch_nelts,)``, float32, on GPU. If None, one is allocated
            here -- convenient for tests, wasteful in a loop, since the whole point of
            the argument is to reuse one allocation across chunks and across the many
            clippers in a chain.
        stream : cupy.cuda.Stream or None, optional
            CUDA stream to use. If None, uses current cupy stream.
        """
        import cupy as cp

        if stream is None:
            stream = cp.cuda.get_current_stream()
        if scratch is None:
            scratch = cp.empty(self.scratch_nelts, dtype=cp.float32)

        self._cpp_launch(intensity, weights, scratch, stream.ptr)


def wrms_view(arr, axis):
    """Reshape a downsampled (B, F_ds, T_ds) array to the (R, L) form GpuWrms wants.

    The clippers reduce along three axes, but by the time the statistic runs there is only
    one shape: one output per row of a contiguous 2-D array. AXIS_NONE works because a
    contiguous (B, F_ds, T_ds) array IS a (B, F_ds*T_ds) array -- reducing a whole plane is
    reducing one long row.

    Note that AXIS_FREQ is the one that moves data (the frequency axis has to become the
    fast one). GpuIntensityClipper does the same thing with GpuWiDownsampler(1,1,True).
    """

    (B, F_ds, T_ds) = arr.shape
    ax = int(axis)

    if ax == AXIS_TIME:
        return arr.reshape(B * F_ds, T_ds)
    if ax == AXIS_FREQ:
        return np.ascontiguousarray(np.swapaxes(arr, 1, 2)).reshape(B * T_ds, F_ds)
    if ax == AXIS_NONE:
        return arr.reshape(B, F_ds * T_ds)

    raise RuntimeError(f'bad axis {axis}')


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
        AXIS_FREQ, AXIS_TIME or AXIS_NONE.

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
    ax = int(axis)
    if ax == AXIS_TIME:
        shape = (B, F_ds, 1)
    elif ax == AXIS_FREQ:
        shape = (B, 1, T_ds)
    elif ax == AXIS_NONE:
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

    * For ``axis = AXIS_FREQ`` the answer does not depend on ``nt_chunk`` at all. Each
      time column is its own statistic, so chunking the time axis changes nothing. Only
      AXIS_TIME and AXIS_NONE see the boundaries.
    * ``nt_chunk`` must be a multiple of ``Dt``, so that a downsampled cell cannot
      straddle a boundary, and ``T`` must be a multiple of ``nt_chunk``.

    Unlike ReferenceWrms there is no ``dtype`` argument: this reference is float64
    throughout, because its job is to be more accurate than the kernel it checks. The
    free function intensity_clip() is dtype-agnostic, which is what a float32 comparison
    against the kernel's own (mean, var) needs.
    """

    def __init__(self, axis, sigma, Df, Dt, niter, iter_sigma, two_pass,
                 nt_chunk=None, eps_multiplier=1.0):
        assert int(axis) in (AXIS_FREQ, AXIS_TIME, AXIS_NONE)
        assert Df >= 1 and Dt >= 1
        assert niter >= 1
        assert sigma >= 0.0 and iter_sigma >= 0.0

        self.axis = int(axis)
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

        (i_ds, w_ds) = ReferenceWiDownsampler(self.Df, self.Dt, transpose=False).apply(I, W)

        wrms = ReferenceWrms(self.niter, self.iter_sigma, self.two_pass,
                             eps_multiplier=self.eps_multiplier)
        (mean, var) = wrms.apply(wrms_view(i_ds, self.axis), wrms_view(w_ds, self.axis))

        # Note sigma here, not iter_sigma: the refinements clipped at iter_sigma, and this
        # final clip uses a different threshold. The old code comments on this at each of
        # its six call sites, which is a fair measure of how easy it is to get wrong.
        return intensity_clip(i_ds, W, mean, var, self.sigma, self.axis, self.Df, self.Dt)
