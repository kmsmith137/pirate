"""Numpy reference for GpuWrms, plus that class's method injections.

The primitive wrms_iterate() is transcribed from _ref_wrms_iterate() in
../../extern/rf_kernels/test-intensity-clipper.cpp, which is the old code's own scalar
reference for rf_kernels::weighted_mean_rms. ReferenceWrms then assembles a whole wrms
from it the same way reference_wrms_compute() does.
"""

import numpy as np

import ksgpu
from ..pirate_pybind11 import GpuWrms


# The variance-validity cutoffs, as FLOAT32 constants even though this reference runs in
# float64. They are part of the algorithm as chimefrb defines it -- simd_helpers hardcodes
# machine_epsilon<float>() = 1.19e-07 -- not a property of the arithmetic evaluating it.
EPS_MACH_F32 = 1.19e-07
EPS_2 = 1.0e2 * EPS_MACH_F32
EPS_3 = 1.0e3 * EPS_MACH_F32


@ksgpu.inject_methods(GpuWrms)
class GpuWrmsInjections:
    # No class docstring here: GpuWrms's docstring lives in the pybind11 binding
    # (option 1 in notes/docstrings.md); this injector adds a stream argument for
    # launch(), and lets the caller omit the scratch array.

    # Save reference to C++ method
    _cpp_launch = GpuWrms.launch

    def launch(self, mean, var, in_i, in_w, scratch=None, stream=None):
        """GPU kernel launch (async, does not sync stream).

        Parameters
        ----------
        mean, var : cupy.ndarray
            Shape (R,), float32, contiguous, on GPU. Fully overwritten.
        in_i, in_w : cupy.ndarray
            Shape (R, L), float32, fully contiguous, on GPU. Read only, and must not
            alias the outputs. ``in_w`` must be >= 0.
        scratch : cupy.ndarray or None, optional
            Shape ``(self.scratch_nelts(R),)``, float32, on GPU. If None, one is
            allocated here -- convenient for tests, wasteful in a loop, since the
            whole point of the argument is to reuse one allocation across chunks.
        stream : cupy.cuda.Stream or None, optional
            CUDA stream to use. If None, uses current cupy stream.
        """
        import cupy as cp

        if stream is None:
            stream = cp.cuda.get_current_stream()
        if scratch is None:
            scratch = cp.empty(self.scratch_nelts(in_i.shape[0]), dtype=cp.float32)

        self._cpp_launch(mean, var, in_i, in_w, scratch, stream.ptr)


def wrms_iterate(mean_in, I, W, eps_multiplier=1.0):
    """One UN-THRESHOLDED wrms step from a given mean. Returns (mean_out, var_out).

    All of I, W are (R, L); mean_in and both outputs are (R,). Transcribed from
    _ref_wrms_iterate(). Note that the sums are taken about 'mean_in' rather than about
    zero, which is what keeps them stable when the mean is far from zero, and that a
    rejected variance becomes exactly zero while the mean is updated anyway.

    'eps_multiplier' scales both variance-validity cutoffs. It exists for the bracketing
    in test_wrms.py: running the reference at 0.5 and 1.5 brackets the cutoff decision, so
    that a kernel which lands on the other side of it near roundoff is not called wrong.
    """

    mean_in = np.asarray(mean_in)
    d = I - mean_in[:, None]

    wsum = W.sum(axis=1)
    wisum = (W * d).sum(axis=1)
    wiisum = (W * d * d).sum(axis=1)

    pos = (wsum > 0)
    den = np.where(pos, wsum, 1)
    dmean = np.where(pos, wisum / den, 0)
    var = np.where(pos, wiisum / den - dmean * dmean, 0)

    # Threshold the variance at (eps_2 * mean_in)^2 and at eps_3 * dmean^2. Note that the
    # first uses the OLD mean and the second the INCREMENT, not the new mean. Note also
    # that at mean_in = 0 the first cutoff is 0, which is what clamps a slightly negative
    # single-pass variance to zero.
    e2 = EPS_2 * eps_multiplier
    e3 = EPS_3 * eps_multiplier
    var = np.where(var < (e2 * mean_in)**2, 0, var)
    var = np.where(var < e3 * dmean * dmean, 0, var)

    return (mean_in + dmean, var)


def iclip(mean, thresh, I, W):
    """The survivor weights of one refinement: W, zeroed where |I - mean| >= thresh.

    Returns a new array; W is not modified. 'mean' and 'thresh' are (R,), I and W are
    (R, L). Note the strict '<' for a survivor, which means thresh = 0 kills the whole
    row -- that is how a rejected variance propagates.

    This is also what the intensity_clipper applies to the weights at the end of its run,
    which is the identity the induction in test_wrms.py rests on.
    """

    survive = np.abs(I - np.asarray(mean)[:, None]) < np.asarray(thresh)[:, None]
    return np.where(survive, W, 0)


class ReferenceWrms:
    """Numpy reference for GpuWrms (src_lib/chimefrb/Wrms.cu).

    Same semantics, same argument names, one output per row of an (R, L) array.

    DELIBERATELY NOT THE SAME FORMULATION as the GPU kernel. The kernel implements
    rf_kernels/mean_rms_internals.hpp directly, which is what we are porting; this
    assembles the same statistic out of repeated wrms_iterate() calls, which is what the
    old code's own reference does. The two agree except for rows whose variance sits
    within roundoff of a validity cutoff, because the kernel's two-pass 'finalize' has no
    '- dmean^2' term and applies only the eps_2 cutoff where a second iterate applies
    both. Bracketing with eps_multiplier is what covers that corner; see test_wrms.py.
    """

    def __init__(self, niter, iter_sigma, two_pass, eps_multiplier=1.0, dtype=np.float64):
        assert niter >= 1
        assert iter_sigma >= 0.0

        self.niter = int(niter)
        self.iter_sigma = float(iter_sigma)
        self.two_pass = bool(two_pass)
        self.eps_multiplier = float(eps_multiplier)
        self.dtype = dtype

    def apply(self, in_i, in_w):
        """Returns (mean, var), given (R, L) arrays. Does not modify its arguments."""

        assert in_i.shape == in_w.shape
        assert in_i.ndim == 2

        I = np.asarray(in_i, dtype=self.dtype)
        W = np.asarray(in_w, dtype=self.dtype)
        (R, _) = I.shape

        # First pass. One un-thresholded step from zero gives the single-pass form; a
        # second gives the two-pass form, since the increment is zero in exact arithmetic
        # and the sums are then taken about the mean. (This identity is why the reference
        # needs only one primitive; see the class docstring for where it is not exact.)
        mean = np.zeros(R, dtype=self.dtype)
        (mean, var) = wrms_iterate(mean, I, W, self.eps_multiplier)
        if self.two_pass:
            (mean, var) = wrms_iterate(mean, I, W, self.eps_multiplier)

        # Refinements. Each one re-derives its survivor set from the ORIGINAL weights and
        # the current (mean, var) -- the masking is not cumulative, because the kernel
        # reads the weights fresh from the buffer every time.
        for _ in range(self.niter - 1):
            thresh = self.iter_sigma * np.sqrt(var)
            (mean, var) = wrms_iterate(mean, I, iclip(mean, thresh, I, W), self.eps_multiplier)

        return (mean, var)
