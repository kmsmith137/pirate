"""Numpy reference for GpuStdDevClipper.

The std_dev_clipper flags whole channels (AXIS_TIME) or whole time samples (AXIS_FREQ) whose
NOISE LEVEL is anomalous: it computes one variance per row, then clips outliers in that
array of variances.

Stage 1 -- one variance per row -- is ReferenceWrms at niter=1, already validated against
rf_kernels. Stage 2 is clip_1d() below, transcribed from std_dev_clipper.cpp::_clip_1d() in
../../extern/rf_kernels. That function has no reference implementation anywhere in the old
code (its own comment says it was never unit-tested), so clip_1d() is a genuinely new test of
it, and misc/chimefrb/rfi_std_dev_clipper/ drives the real _clip_1d() directly to compare.
"""

import numpy as np

import ksgpu
from ..pirate_pybind11 import GpuStdDevClipper
from .ReferenceIntensityClipper import AXIS_FREQ, AXIS_TIME, AXIS_NONE, wrms_view
from .ReferenceWiDownsampler import ReferenceWiDownsampler
from .ReferenceWrms import ReferenceWrms


@ksgpu.inject_methods(GpuStdDevClipper)
class GpuStdDevClipperInjections:
    # No class docstring here: GpuStdDevClipper's docstring lives in the pybind11 binding
    # (option 1 in notes/docstrings.md); this injector adds a stream argument for
    # launch(), and lets the caller omit the scratch array.

    # Save reference to C++ method
    _cpp_launch = GpuStdDevClipper.launch

    def launch(self, intensity, weights, scratch=None, stream=None):
        """GPU kernel launch (async, does not sync stream).

        Parameters
        ----------
        intensity : cupy.ndarray
            Shape (B, F, nt_chunk), float32, fully contiguous, on GPU. Read only.
        weights : cupy.ndarray
            Same shape and dtype. MODIFIED IN PLACE: whole rows are zeroed where the clip
            fires, and every other weight is left bit-identical. Must be >= 0 on entry.
        scratch : cupy.ndarray or None, optional
            Shape ``(self.scratch_nelts,)``, float32, on GPU. If None, one is allocated
            here -- convenient for tests, wasteful in a loop, since the point of the
            argument is to share one allocation across chunks and across a chain's clippers.
        stream : cupy.cuda.Stream or None, optional
            CUDA stream to use. If None, uses current cupy stream.
        """
        import cupy as cp

        if stream is None:
            stream = cp.cuda.get_current_stream()
        if scratch is None:
            scratch = cp.empty(self.scratch_nelts, dtype=cp.float32)

        self._cpp_launch(intensity, weights, scratch, stream.ptr)


def clip_1d(v, sigma):
    """Stage 2 of the std_dev_clipper: rf_kernels std_dev_clipper::_clip_1d(), per beam.

    'v' is (B, nrows) -- one beam's row variances per row of the array -- or (nrows,) for a
    single beam. Returns a new array of the same shape; 'v' is not modified. Transcribed line
    by line from the production source, which is the only source there is. Five details,
    each easy to get wrong:

      - the mean vbar and standard deviation s are over the entries with v > 0 only, and s
        divides by their count n, NOT n-1;
      - if n < 2 (the old code's "acc0 < 1.5"), EVERY entry is zeroed -- which zeroes the
        whole beam once the apply runs;
      - vbar and s are computed before anything is clipped: one pass, not an iteration;
      - the clip is |v - vbar| >= sigma*s, applied to every entry (a no-op on the zeros);
      - so an entry survives iff |v - vbar| < sigma*s, strictly, and s = 0 clips everything.

    That last point has an ill-conditioned corner. If every valid v is exactly equal, then
    s = 0 in exact arithmetic and everything is clipped, but a float32 mean of n copies of x
    usually does not come back to exactly x, and then nothing is. The old code, this function
    and the GPU all round differently there; see plans/chimefrb_std_dev_clipper.md 2.4.

    Arithmetic follows the input dtype: float32 input gives float32 sums (in numpy's pairwise
    order, not the old sequential one), float64 gives float64.
    """

    v = np.asarray(v)
    single = (v.ndim == 1)
    v2 = np.atleast_2d(v)

    valid = (v2 > 0)
    n = valid.sum(axis=1)
    nn = np.maximum(n, 1).astype(v2.dtype)

    vbar = np.where(valid, v2, 0).sum(axis=1) / nn
    d = v2 - vbar[:, None]
    s = np.sqrt(np.where(valid, d * d, 0).sum(axis=1) / nn)
    thresh = np.asarray(sigma, dtype=v2.dtype) * s

    out = np.where(np.abs(d) >= thresh[:, None], 0, v2)
    out[n < 2, :] = 0

    return out[0] if single else out


def std_dev_apply(weights, v, axis, Df, Dt):
    """Step 4: zero the full-resolution weights of every row whose variance is zero.

    'weights' is (B, F, T); 'v' is (B, nrows), or anything reshapeable to it, with
    nrows = F/Df for AXIS_TIME and T/Dt for AXIS_FREQ. Returns a new array. A killed row is
    Df whole channels (TIME) or Dt whole time samples across every frequency (FREQ).
    """

    weights = np.asarray(weights)
    (B, F, T) = weights.shape
    ax = int(axis)

    if ax == AXIS_TIME:
        kill = (np.asarray(v).reshape(B, F // Df) == 0)
        kill = np.repeat(kill, Df, axis=1)[:, :, None]      # (B, F, 1)
    elif ax == AXIS_FREQ:
        kill = (np.asarray(v).reshape(B, T // Dt) == 0)
        kill = np.repeat(kill, Dt, axis=1)[:, None, :]      # (B, 1, T)
    else:
        raise RuntimeError('std_dev_apply: the std_dev_clipper does not implement AXIS_NONE')

    return np.where(kill, 0, weights)


class ReferenceStdDevClipper:
    """Numpy reference for GpuStdDevClipper (src_lib/chimefrb/StdDevClipper.cu).

    Same semantics and argument names, in float64. Stage 1 is ReferenceWiDownsampler, then
    ReferenceWrms at niter=1 on the wrms_view() of the downsampled pair; stage 2 is clip_1d()
    per beam; then std_dev_apply(). AXIS_NONE is rejected, as in the old code.

    Like ReferenceIntensityClipper, this implements T = N*nt_chunk, which the GPU class does
    not: apply() on a (B, F, N*nt_chunk) array returns exactly what N separate calls on the
    (B, F, nt_chunk) sub-arrays would, concatenated along time. Here it is not a no-op for
    either axis: stage 2 pools one beam's rows within one chunk, so it runs per (beam, chunk).

    'eps_multiplier' scales stage 1's variance-validity cutoffs, for the bracketing in the
    tests. Note that it does not bracket stage 2 monotonically -- a row that stage 1 admits
    changes the population stage 2 sees -- which is why the tests condition on the GPU's own
    stage-1 decisions instead; see plans/chimefrb_std_dev_clipper.md 9.2(b).
    """

    def __init__(self, axis, sigma, Df, Dt, two_pass, nt_chunk=None, eps_multiplier=1.0):
        if int(axis) == AXIS_NONE:
            raise RuntimeError('ReferenceStdDevClipper: AXIS_NONE is not supported'
                               ' (rf_kernels::std_dev_clipper does not implement it)')
        assert int(axis) in (AXIS_FREQ, AXIS_TIME)
        assert Df >= 1 and Dt >= 1
        assert sigma >= 0.0

        self.axis = int(axis)
        self.sigma = float(sigma)
        self.Df = int(Df)
        self.Dt = int(Dt)
        self.two_pass = bool(two_pass)
        self.nt_chunk = None if (nt_chunk is None) else int(nt_chunk)
        self.eps_multiplier = float(eps_multiplier)

    def variances(self, in_i, in_w):
        """Stage 1 only, on ONE chunk: returns (mean, var), each (B, nrows) float64.

        Exposed because the tests compare stage 2 on conditioned inputs, and need the
        reference's stage-1 values to condition.
        """

        I = np.asarray(in_i, dtype=np.float64)
        W = np.asarray(in_w, dtype=np.float64)
        (B, F, T) = I.shape
        assert (self.nt_chunk is None) or (T == self.nt_chunk)

        (i_ds, w_ds) = ReferenceWiDownsampler(self.Df, self.Dt, transpose=False).apply(I, W)
        wrms = ReferenceWrms(1, 0.0, self.two_pass, eps_multiplier=self.eps_multiplier)
        (mean, var) = wrms.apply(wrms_view(i_ds, self.axis), wrms_view(w_ds, self.axis))
        return (mean.reshape(B, -1), var.reshape(B, -1))

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

        for c in range(T // nt):
            sl = slice(c * nt, (c+1) * nt)
            (_, var) = self.variances(in_i[:, :, sl], in_w[:, :, sl])
            v = clip_1d(var, self.sigma)
            W = np.asarray(in_w[:, :, sl], dtype=np.float64)
            out[:, :, sl] = std_dev_apply(W, v, self.axis, self.Df, self.Dt)

        return out
