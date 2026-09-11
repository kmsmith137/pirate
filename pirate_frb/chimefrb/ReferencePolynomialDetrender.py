"""Numpy reference for GpuPolynomialDetrender, plus that class's method injections.

The reference is transcribed from the old code's KERNEL -- _kernel_detrend_t() and
_kernel_detrend_f() in ../../extern/rf_kernels/rf_kernels/polynomial_detrender_internals.hpp,
and the simd_trimatrix::cholesky_in_place_checked() they call -- NOT from the old python
transform in rf_pipelines/retirement_home/polynomial_detrender.py, which samples a
different grid, has no conditioning gate and ignores epsilon. The gate has no reference
implementation anywhere in the old code; this is its first transcription, and the spot
check misc/chimefrb/polynomial_detrender/ is what pins it to the old binary.
"""

import numpy as np

import ksgpu
from ..pirate_pybind11 import GpuPolynomialDetrender
from .ReferenceIntensityClipper import AXIS_FREQ, AXIS_TIME


@ksgpu.inject_methods(GpuPolynomialDetrender)
class GpuPolynomialDetrenderInjections:
    # No class docstring here: GpuPolynomialDetrender's docstring lives in the pybind11
    # binding (option 1 in notes/docstrings.md); this injector adds a stream argument
    # for launch().

    # Save reference to C++ method
    _cpp_launch = GpuPolynomialDetrender.launch

    def launch(self, intensity, weights, stream=None):
        """GPU kernel launch (async, does not sync stream).

        Parameters
        ----------
        intensity : cupy.ndarray
            Shape (M, nfreq, T), float32, fully contiguous, on GPU, T a multiple of
            nt_chunk. Detrended in place on rows that pass the gate.
        weights : cupy.ndarray
            Same shape and dtype. Must be >= 0. MODIFIED IN PLACE: zeroed on rows that
            fail the gate, untouched elsewhere.
        stream : cupy.cuda.Stream or None, optional
            CUDA stream to use. If None, uses current cupy stream.
        """
        import cupy as cp

        if stream is None:
            stream = cp.cuda.get_current_stream()

        self._cpp_launch(intensity, weights, stream.ptr)


def z_grid(n):
    """The old code's sample coordinate: the midpoints of n equal cells of [-1, 1].

    z_i = (2 i + 1 - n) / n, i = 0..n-1, so z never reaches +-1.
    """
    return (2.0 * np.arange(n) + 1.0 - n) / n


def legendre(N, z):
    """Legendre polynomials P_0(z) .. P_{N-1}(z), shape (N, len(z)), by the old code's
    recurrence P_m = ((2m-1) z P_{m-1} - (m-1) P_{m-2}) / m."""
    z = np.asarray(z, dtype=np.float64)
    P = np.zeros((N,) + z.shape)
    P[0] = 1.0
    if N > 1:
        P[1] = z
    for m in range(2, N):
        P[m] = ((2*m - 1) * z * P[m-1] - (m - 1) * P[m-2]) / m
    return P


def pivot_ratios(Ahat):
    """The old gate's statistic for a batch of unit-diagonal matrices ``Ahat`` (K, N, N).

    Returns (K, N): at each Cholesky pivot j, in order, the Schur complement
    u_j = Ahat_jj - sum_{k<j} L_jk^2, which on a unit diagonal is the old code's
    u_j / A_jj (the gate is invariant under the rescaling to unit diagonal). A pivot
    that is not positive is reported as it is (<= 0) and the factorization continues with
    L_jj = 1, as the old code does, so later pivots are finite; only the minimum over
    pivots is meaningful for such a matrix, and it is <= 0. A diagonal entry that is
    exactly zero (no weight, or weight only where the basis function vanishes) gives a
    ratio of 0 at that pivot.
    """
    Ahat = np.asarray(Ahat, dtype=np.float64)
    (K, N, _) = Ahat.shape
    L = np.zeros((K, N, N))
    r = np.zeros((K, N))
    for j in range(N):
        s = Ahat[:, j, j] - np.einsum('kj,kj->k', L[:, j, :j], L[:, j, :j])
        r[:, j] = np.where(Ahat[:, j, j] > 0, s, 0.0)
        pos = (s > 0)
        Ljj = np.where(pos, np.sqrt(np.where(pos, s, 1.0)), 1.0)
        L[:, j, j] = Ljj
        for i in range(j+1, N):
            L[:, i, j] = (Ahat[:, i, j] - np.einsum('kj,kj->k', L[:, i, :j], L[:, j, :j])) / Ljj
    return r


class ReferencePolynomialDetrender:
    """Numpy reference for GpuPolynomialDetrender (src_lib/chimefrb/PolynomialDetrender.cu),
    and for the old code's AXIS_FREQ variant, which the GPU class does not implement.

    Per row -- one (beam, channel, chunk of ``nt_chunk`` samples) along time, or one
    (beam, time sample) across all channels along frequency -- fit a polynomial of degree
    ``polydeg`` to the intensity by weighted least squares, in the Legendre basis on the
    midpoint grid z_grid(n), and subtract it at every sample of the row. There is no
    regularization. Before solving, the CONDITIONING GATE decides whether the row is fit
    at all: the Cholesky factorization of the normal matrix is taken pivot by pivot, and
    the row passes only if at every pivot the Schur complement exceeds ``epsilon`` times
    the diagonal entry (see pivot_ratios()). A row that fails has ALL its weights set to
    zero and its intensity left untouched. That is the transform's only effect on the
    weights, and at the production setting (degree 4, epsilon 0.01, 1024-sample chunks)
    it fires on any channel whose weighted samples form a contiguous run shorter than
    about half the chunk.

    Everything is float64, so the reference is more accurate than the kernel it checks
    rather than equally inaccurate; and a sample with zero weight contributes exactly
    zero to the fit even if its intensity is NaN. The old code multiplies weight by
    intensity, so a NaN at a masked sample turned the whole row to NaN. A NaN at a
    WEIGHTED sample does that in both codes; a NaN weight fails the gate in both.

    Attributes (read-only):

    - ``polydeg``, ``epsilon``, ``nt_chunk``, ``axis`` -- the constructor arguments.
    - ``N`` (int) -- number of coefficients, polydeg + 1.
    """

    def __init__(self, polydeg, epsilon, nt_chunk, axis=AXIS_TIME):
        polydeg, nt_chunk = int(polydeg), int(nt_chunk)
        assert polydeg >= 0
        assert epsilon > 0
        assert axis in (AXIS_FREQ, AXIS_TIME)
        if axis == AXIS_TIME:
            assert nt_chunk >= 1

        self.polydeg = polydeg
        self.epsilon = float(epsilon)
        self.nt_chunk = nt_chunk
        self.axis = axis
        self.N = polydeg + 1

    # ---- Rows. Along time, an (M, F, T) array is (M*F*nchunk) rows of nt_chunk samples,
    # contiguous; along frequency it is (M*T) rows of F samples, which needs a transpose.
    # _rows() returns a view or a copy; _unrows() maps a row-shaped result back.

    def _rows(self, arr):
        arr = np.asarray(arr, dtype=np.float64)
        assert arr.ndim == 3
        (M, F, T) = arr.shape
        if self.axis == AXIS_TIME:
            assert T % self.nt_chunk == 0, 'T must be a multiple of nt_chunk'
            return arr.reshape(M * F * (T // self.nt_chunk), self.nt_chunk)
        return np.ascontiguousarray(arr.transpose(0, 2, 1)).reshape(M * T, F)

    def _unrows(self, rows, shape):
        (M, F, T) = shape
        if self.axis == AXIS_TIME:
            return rows.reshape(M, F, T)
        return rows.reshape(M, T, F).transpose(0, 2, 1)

    def _row_shape(self, shape):
        """Shape of a per-row statistic: (M, F, nchunk) along time, (M, T) along frequency."""
        (M, F, T) = shape
        return (M, F, T // self.nt_chunk) if (self.axis == AXIS_TIME) else (M, T)

    def _normal_equations(self, d_rows, w_rows):
        """(A, v, wsum) of every row: A (K, N, N), v (K, N), wsum (K,)."""
        n = d_rows.shape[1]
        P = legendre(self.N, z_grid(n))                        # (N, n)
        # SELECT, do not multiply: 0*nan is nan (see the class docstring).
        wd = np.where(w_rows != 0, w_rows * d_rows, 0.0)
        A = np.einsum('ki,ai,bi->kab', w_rows, P, P)
        v = np.einsum('ki,ai->ka', wd, P)
        return (A, v, w_rows.sum(axis=1))

    def _equilibrate(self, A):
        """(Ahat, s): Ahat = diag(s) A diag(s) has unit diagonal where A's is positive.

        Where a diagonal entry is zero (no weight in the row, or weight only where that
        basis function vanishes) s is 1, so the entry stays zero and pivot_ratios()
        reports 0 there.
        """
        diag = np.einsum('kii->ki', A)
        s = np.where(diag > 0, 1.0 / np.sqrt(np.where(diag > 0, diag, 1.0)), 1.0)
        return (A * s[:, :, None] * s[:, None, :], s)

    def _gate_matrices(self, weights):
        """(Ahat, ok, finite): the equilibrated normal matrix of every row (K, N, N), which
        rows have any weight and finite entries ('ok'), and which have finite entries at all.
        The gate and both conditioning statistics are functions of these."""
        w = self._rows(weights)
        (A, _, wsum) = self._normal_equations(np.zeros_like(w), w)
        finite = np.isfinite(A).all(axis=(1, 2))
        ok = finite & (wsum > 0)
        (Ahat, _) = self._equilibrate(A)
        return (Ahat, ok, finite)

    def gate_pivots(self, weights):
        """The old gate's statistic at every pivot of every row: pivot_ratios() of the
        equilibrated normal matrix, shaped _row_shape() + (N,).

        Zero at every pivot for a row with no weight, NaN for a row whose weights are not
        finite. The row is masked iff NOT min(pivots) > epsilon. Exposed so that a test can
        size the roundoff band of the gate decision per row: float32 error in pivot j is
        amplified by the pivots before it, roughly in proportion to sum_{k<j} 1/pivot_k.
        """
        (Ahat, ok, finite) = self._gate_matrices(weights)
        K = Ahat.shape[0]
        piv = np.where(finite[:, None], 0.0, np.nan) * np.ones((K, self.N))
        if ok.any():
            piv[ok] = pivot_ratios(Ahat[ok])
        return piv.reshape(self._row_shape(np.shape(weights)) + (self.N,))

    def conditioning(self, weights):
        """Two conditioning statistics of every row's fit, each shaped like _row_shape().

        Returns ``(rmin, lmin)``, both of the normal equations after rescaling to unit
        diagonal, both in (0, 1] for a well-posed row, both exactly 0 for a row with no
        weight, and both NaN for a row whose weights contain a NaN or an infinity:

        - ``rmin``, the smallest sequential Cholesky pivot (min over gate_pivots()): the
          old gate's statistic. The row is masked iff NOT rmin > epsilon. It also governs
          the float32 accuracy of the kernel's solve at WEIGHTED samples, where the
          coefficient error is of order eps_mach / rmin -- and the gate guarantees
          rmin > epsilon on every row it lets through.
        - ``lmin``, the smallest eigenvalue, lmin <= rmin. It governs the accuracy at
          samples WITHOUT weight (extrapolation across a gap), whose fitted values live
          in the directions the pivots see least; the error there is of order
          eps_mach / lmin.

        Both depend on the weights alone.
        """
        (Ahat, ok, finite) = self._gate_matrices(weights)
        rmin = np.where(finite, 0.0, np.nan)
        lmin = np.where(finite, 0.0, np.nan)
        if ok.any():
            rmin[ok] = pivot_ratios(Ahat[ok]).min(axis=1)
            lmin[ok] = np.maximum(np.linalg.eigvalsh(Ahat[ok])[:, 0], np.finfo(np.float64).tiny)

        shape = self._row_shape(np.shape(weights))
        return (rmin.reshape(shape), lmin.reshape(shape))

    def masked(self, weights):
        """Boolean, shaped like _row_shape(): the rows the gate fails (weights to be
        zeroed, intensity to be left alone). NaN statistics count as masked."""
        (rmin, _) = self.conditioning(weights)
        return ~(rmin > self.epsilon)

    def fit(self, intensity, weights):
        """The fitted polynomial, shape (M, F, T) float64, given (M, F, T) arrays.

        Exactly zero on every row the gate masks. Does not modify its arguments.
        """
        shape = np.shape(intensity)
        assert np.shape(weights) == shape
        d = self._rows(intensity)
        w = self._rows(weights)
        (A, v, wsum) = self._normal_equations(d, w)
        (K, N) = v.shape
        n = d.shape[1]

        # The gate, on the same normal equations, then the solve on the survivors.
        ok = np.isfinite(A).all(axis=(1, 2)) & (wsum > 0)
        coeffs = np.zeros((K, N))
        if ok.any():
            (Ahat, s) = self._equilibrate(A[ok])
            passed = pivot_ratios(Ahat).min(axis=1) > self.epsilon
            idx = np.nonzero(ok)[0][passed]
            if idx.size:
                y = np.linalg.solve(Ahat[passed], (s[passed] * v[idx])[..., None])[..., 0]
                coeffs[idx] = s[passed] * y

        model = coeffs @ legendre(N, z_grid(n))                # (K, n)
        return self._unrows(model, shape)

    def apply(self, intensity, weights):
        """(intensity_out, weights_out), both (M, F, T) float64: the old transform's effect.

        On a row that passes the gate, intensity - fit and unchanged weights; on a row that
        fails, unchanged intensity and all weights zero.
        """
        shape = np.shape(intensity)
        model = self.fit(intensity, weights)
        masked = self.masked(weights)
        masked_rows = np.broadcast_to(self._row_mask_expand(masked, shape), shape)

        out_i = np.where(masked_rows, np.asarray(intensity, dtype=np.float64),
                         np.asarray(intensity, dtype=np.float64) - model)
        out_w = np.where(masked_rows, 0.0, np.asarray(weights, dtype=np.float64))
        return (out_i, out_w)

    def _row_mask_expand(self, per_row, shape):
        """A per-row boolean (see _row_shape()) broadcast to the (M, F, T) array shape."""
        (M, F, T) = shape
        if self.axis == AXIS_TIME:
            return np.repeat(per_row, self.nt_chunk, axis=2).reshape(M, F, T)
        return per_row[:, None, :]
