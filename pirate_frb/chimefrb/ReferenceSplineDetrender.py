"""Numpy reference for GpuSplineDetrender, plus that class's method injections.

The reference is transcribed from reference_spline_detrender in
../../extern/rf_kernels/test-spline-detrender.cpp, which is the old code's own scalar
reference for rf_kernels::spline_detrender, with two deliberate departures (see the
class docstring).
"""

import numpy as np

import ksgpu
from ..pirate_pybind11 import GpuSplineDetrender


@ksgpu.inject_methods(GpuSplineDetrender)
class GpuSplineDetrenderInjections:
    # No class docstring here: GpuSplineDetrender's docstring lives in the pybind11
    # binding (option 1 in notes/docstrings.md); this injector adds a stream argument
    # for launch().

    # Save reference to C++ method
    _cpp_launch = GpuSplineDetrender.launch

    def launch(self, intensity, weights, stream=None):
        """GPU kernel launch (async, does not sync stream).

        Parameters
        ----------
        intensity : cupy.ndarray
            Shape (M, nfreq, T), float32, fully contiguous, on GPU. The fitted baseline
            is subtracted in place at every channel.
        weights : cupy.ndarray
            Same shape and dtype. Must be >= 0. Read only.
        stream : cupy.cuda.Stream or None, optional
            CUDA stream to use. If None, uses current cupy stream.
        """
        import cupy as cp

        if stream is None:
            stream = cp.cuda.get_current_stream()

        self._cpp_launch(intensity, weights, stream.ptr)


# Q[a,c] = int_0^1 h_a'(x) h_c'(x) dx for the cubic Hermite basis (h00, h10, h01, h11) on
# [0,1]: the 'reg_matrix' the old code hard-codes, written here as the fractions it is.
HERMITE_SLOPE_PENALTY = np.array([
    [ 6/5,   1/10, -6/5,   1/10],
    [ 1/10,  2/15, -1/10, -1/30],
    [-6/5,  -1/10,  6/5,  -1/10],
    [ 1/10, -1/30, -1/10,  2/15]])


def bin_edges(nfreq, nbins):
    """The old code's bin edges: b*nfreq/nbins rounded to the nearest channel, b = 0..nbins.

    The expression is evaluated in the same order as rf_kernels' _spline_detrender_init()
    and as GpuSplineDetrender's constructor, so the three agree even at an exact
    half-integer.
    """
    return np.array([int(b / nbins * nfreq + 0.5) for b in range(nbins + 1)], dtype=np.int64)


def hermite_basis(nfreq, nbins):
    """Per-channel bin index and Hermite basis values.

    Returns (bin, H): ``bin`` (nfreq,) int, the bin of each channel; ``H`` (nfreq, 4)
    float64, the values of (h00, h10, h01, h11) at the channel's fractional position
    within its bin, x = nbins*(f+1/2)/nfreq - bin. Coefficient 2*bin+a multiplies
    H[f, a], so the four coefficients of a bin are (value, slope) at its lower edge then
    (value, slope) at its upper edge, and consecutive bins share two.
    """
    edges = bin_edges(nfreq, nbins)
    f = np.arange(nfreq)
    b = np.searchsorted(edges, f, side='right') - 1
    x = nbins * (f + 0.5) / nfreq - b
    H = np.stack([(1-x)*(1-x)*(1+2*x), (1-x)*(1-x)*x, x*x*(3-2*x), x*x*(x-1)], axis=1)
    return (b, H)


class ReferenceSplineDetrender:
    """Numpy reference for GpuSplineDetrender (src_lib/chimefrb/SplineDetrender.cu).

    Same semantics and argument names. Per (beam, time sample), independently, the
    2*(nbins+1) coefficients of a piecewise-cubic spline (C^1 at the ``nbins`` equal bin
    edges) minimize::

        sum_f w_f (d_f - b(f))^2  +  epsilon * (sum_f w_f) / nbins * sum_bins int_0^1 (db/dx)^2 dx

    and b(f) is subtracted from every channel. A sample whose weights are all zero is
    left untouched. Note the penalty scales with the sample's total weight.

    Two departures from the transcription source, both deliberate and both shared with
    the GPU kernel: everything is float64, so the reference is more accurate than the
    kernel it checks rather than equally inaccurate; and a channel with zero weight
    contributes exactly zero to the fit even if its intensity is NaN. The old code
    multiplies weight by intensity, so 0*NaN would poison the whole time sample.

    Attributes (read-only):

    - ``nfreq``, ``nbins``, ``epsilon`` -- the constructor arguments.
    - ``N_phi`` (int) -- number of coefficients, 2*(nbins+1).
    - ``edges`` (int array, nbins+1) -- bin edges as channel indices, from 0 to nfreq.
    """

    def __init__(self, nfreq, nbins, epsilon):
        nfreq, nbins = int(nfreq), int(nbins)
        assert nbins >= 1
        assert nfreq >= nbins, 'every bin must hold at least one channel'
        assert epsilon > 0

        self.nfreq = nfreq
        self.nbins = nbins
        self.epsilon = float(epsilon)
        self.N_phi = 2*(nbins+1)
        self.edges = bin_edges(nfreq, nbins)
        (self.bin, self.H) = hermite_basis(nfreq, nbins)

        # The slope penalty on the whole coefficient vector: the 4x4 block of every bin,
        # accumulating where adjacent bins share an edge.
        R = np.zeros((self.N_phi, self.N_phi))
        for b in range(nbins):
            R[2*b:2*b+4, 2*b:2*b+4] += HERMITE_SLOPE_PENALTY
        self.R = R

    def _normal_equations(self, intensity, weights):
        """(A, U, wsum): the regularized normal equations of every (beam, time) sample.

        A is (M, T, N, N), U is (M, T, N), wsum is (M, T). Each bin contributes its
        weighted Gram of the four Hermite functions and its four data moments; then the
        weight-scaled penalty is added.
        """
        d = np.asarray(intensity, dtype=np.float64)
        w = np.asarray(weights, dtype=np.float64)
        assert d.ndim == 3 and d.shape == w.shape
        assert d.shape[1] == self.nfreq
        (M, _, T) = d.shape
        N = self.N_phi

        # SELECT, do not multiply: 0*nan is nan (see the class docstring).
        wd = np.where(w != 0, w*d, 0.0)

        A = np.zeros((M, T, N, N))
        U = np.zeros((M, T, N))
        for b in range(self.nbins):
            sl = slice(self.edges[b], self.edges[b+1])
            Hb = self.H[sl]
            A[:, :, 2*b:2*b+4, 2*b:2*b+4] += np.einsum('mft,fa,fc->mtac', w[:, sl, :], Hb, Hb)
            U[:, :, 2*b:2*b+4] += np.einsum('mft,fa->mta', wd[:, sl, :], Hb)

        wsum = w.sum(axis=1)                                  # (M, T)
        strength = self.epsilon * wsum / self.nbins
        A += strength[:, :, None, None] * self.R
        return (A, U, wsum)

    def fit(self, intensity, weights):
        """The fitted baseline, shape (M, nfreq, T) float64, given (M, nfreq, T) arrays.

        Zero wherever the sample's weights are all zero. Does not modify its arguments.
        """
        (A, U, wsum) = self._normal_equations(intensity, weights)
        (M, T, N) = U.shape

        coeffs = np.zeros((M, T, N))
        ok = (wsum > 0)
        if ok.any():
            coeffs[ok] = np.linalg.solve(A[ok], U[ok][..., None])[..., 0]

        model = np.zeros((M, self.nfreq, T))
        for b in range(self.nbins):
            sl = slice(self.edges[b], self.edges[b+1])
            model[:, sl, :] = np.einsum('fa,mta->mft', self.H[sl], coeffs[:, :, 2*b:2*b+4])
        return model

    def conditioning(self, weights):
        """Two conditioning statistics of every (beam, time) sample's fit, each (M, T).

        Returns ``(rmin, lmin)``, both of the normal equations after rescaling to unit
        diagonal (equilibration, as the GPU kernel does before factoring), both in
        (0, 1], and both exactly 0 for a sample with no weight:

        - ``rmin``, the smallest relative Cholesky pivot: the statistic the pirate
          detrenders threshold on (notes/detrending.tex). It governs the float32 accuracy
          of the kernel's solve AT WEIGHTED CHANNELS, where the coefficient error is of
          order eps_mach / rmin.
        - ``lmin``, the smallest eigenvalue, lmin <= rmin. It governs the accuracy at
          channels WITHOUT weight, whose fitted values are set by the penalty alone and
          live in the near-null directions that pivots do not see; there the error is of
          order eps_mach / lmin, and at small epsilon lmin can be far below rmin.

        GpuSplineDetrender computes neither -- the old code has no such statistics -- so
        these exist to calibrate tests. They depend on the weights alone.
        """
        w = np.asarray(weights, dtype=np.float64)
        (A, _, wsum) = self._normal_equations(np.zeros_like(w), w)
        (M, T) = wsum.shape

        rmin = np.zeros((M, T))
        lmin = np.zeros((M, T))
        ok = (wsum > 0)
        if ok.any():
            # With any weight at all the penalty makes the diagonal strictly positive.
            Aok = A[ok]
            s = 1.0 / np.sqrt(np.einsum('kii->ki', Aok))
            Ahat = Aok * s[:, :, None] * s[:, None, :]
            L = np.linalg.cholesky(Ahat)
            rmin[ok] = (np.einsum('kii->ki', L) ** 2).min(axis=-1)
            lmin[ok] = np.maximum(np.linalg.eigvalsh(Ahat)[:, 0], np.finfo(np.float64).tiny)
        return (rmin, lmin)

    def apply(self, intensity, weights):
        """The detrended intensity, shape (M, nfreq, T) float64: intensity minus fit()."""
        return np.asarray(intensity, dtype=np.float64) - self.fit(intensity, weights)
