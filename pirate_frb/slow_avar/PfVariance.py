"""Pure-Python reference: output variance of the peak-finding kernels.

Background (see the "Peak-finding" section of notes/tree_dedispersion.tex):
the peak-finder convolves each time series with P short kernels h_p, multiplies
by weights, and reduces by max.  This module answers a normalization question:
if the peak-finder is fed noise of the form y = x * g (white unit-variance noise
g, smeared by a short kernel x), what is the output variance Var(z_p) of each
profile p?

Since z_p = h_p * y = h_p * (x * g) = (h_p * x) * g, and white unit-variance
noise through a kernel k has variance ||k||^2,

    Var(z_p) = ||h_p * x||^2 = sum_delta R_x[delta] R_{h_p}[delta],

where R_a[delta] = sum_t a[t] a[t+delta] is the (auto)correlation.  Only lags
|delta| < min(T, 2*Wmax) contribute: R_x vanishes past T = len(x), and R_{h_p}
vanishes past the longest kernel (length 2*Wmax).  We precompute the kernel
autocorrelations once into a table of width Tmax = 2*Wmax -- this captures every
kernel's autocorrelation in full, with no assumption about len(x) -- after which
each call is a small matmul.
"""

import numpy as np


class PfVarianceConvolver:
    """Computes Var(h_p * x) for each peak-finding profile p, given a short kernel x.

    Construct once for a given max_kernel_width Wmax (which fixes the kernel bank
    and the number of profiles P = 3*log2(Wmax) + 1), then call variance() many
    times.  variance() maps an input of shape (..., T) to output (..., P), with
    the leading axes treated as spectators.
    """

    def __init__(self, Wmax):
        self.Wmax = int(Wmax)
        self.kernels, self.labels = self.peak_finding_kernels(self.Wmax)
        self.P = len(self.kernels)
        self.Tmax = 2 * self.Wmax     # covers the longest kernel (len 2*Wmax) in full
        self.A = self._autocorr_table(self.kernels, self.Tmax)   # (P, Tmax); A[p,0] = ||h_p||^2

    @staticmethod
    def peak_finding_kernels(Wmax):
        """Materialize the P peak-finding kernels as float64 arrays.

        Returns (kernels, labels), where kernels[p] is the 1-d kernel h_p and
        labels[p] = (l, q).  The ordering is the code convention p = 3*l + q with
        the special profile p=0 = (l=0, q=0) (matching ReferencePeakFindingKernel
        and notes/tree_dedispersion.tex).  Each kernel is built from adjacent
        width-2^l boxcars: q=1 is the width-2^(l+1) boxcar, and q=2,q=3 are
        trapezoids whose end taps are half-weighted.
        """
        Wmax = int(Wmax)
        assert Wmax >= 1 and (Wmax & (Wmax - 1)) == 0, "Wmax must be a power of two >= 1"
        Lq = Wmax.bit_length() - 1    # = log2(Wmax) = number of levels carrying q=1,2,3 profiles

        kernels, labels = [], []
        kernels.append(np.ones(1));  labels.append((0, 0))   # p=0: finest single sample
        for l in range(Lq):
            w = 1 << l                                       # 2^l
            kernels.append(np.ones(2 * w))
            labels.append((l, 1))
            kernels.append(np.concatenate([0.5 * np.ones(w), np.ones(w),     0.5 * np.ones(w)]))
            labels.append((l, 2))
            kernels.append(np.concatenate([0.5 * np.ones(w), np.ones(2 * w), 0.5 * np.ones(w)]))
            labels.append((l, 3))

        assert len(kernels) == 3 * Lq + 1
        return kernels, labels

    @staticmethod
    def _autocorr_table(kernels, Tmax):
        """Table A[p, delta] = sum_t h_p[t] h_p[t+delta], for delta = 0 .. Tmax-1."""
        A = np.zeros((len(kernels), Tmax))
        for p, h in enumerate(kernels):
            n = len(h)
            for delta in range(min(Tmax, n)):    # min == n here (Tmax >= longest kernel)
                A[p, delta] = float(h[:n - delta] @ h[delta:])
        return A

    def variance(self, x):
        """Var(h_p * x) for each profile p.  x: shape (..., T) -> out: shape (..., P)."""
        x = np.asarray(x, dtype=np.float64)
        assert x.ndim >= 1
        T = x.shape[-1]
        assert T >= 1
        d = min(T, self.Tmax)        # number of lags that can be nonzero

        # One-sided autocorrelation of x, lags 0..d-1, over the last axis
        # (leading/spectator axes broadcast through the sum).
        rho = np.stack([(x[..., :T - k] * x[..., k:]).sum(axis=-1) for k in range(d)], axis=-1)  # (..., d)
        rho[..., 1:] *= 2.0          # +/- delta symmetry of R_x

        return rho @ self.A[:, :d].T  # (..., d) @ (d, P) -> (..., P)

    # ---------------------------------------------------------------------------
    # Tests (dispatched from pirate_frb/__main__.py via 'test --avar').

    @staticmethod
    def test_random_variance():
        """Compare variance() to brute-force ||h_p * x||^2, with random spectators/T/Wmax."""
        Wmax = 1 << np.random.randint(0, 6)              # one of 1,2,4,8,16,32
        pfv = PfVarianceConvolver(Wmax)

        shape = tuple(int(s) for s in np.random.randint(1, 4, size=np.random.randint(1, 4)))
        T = int(np.random.randint(1, 13))               # includes T > 2*Wmax for small Wmax
        x = np.random.standard_normal(shape + (T,))

        got = pfv.variance(x)
        want = np.empty(shape + (pfv.P,))
        for idx in np.ndindex(*shape):
            for p, h in enumerate(pfv.kernels):
                k = np.convolve(h, x[idx])
                want[idx + (p,)] = float((k * k).sum())

        assert got.shape == want.shape, (got.shape, want.shape)
        assert np.allclose(got, want, rtol=1e-9, atol=1e-12), \
            (Wmax, shape, T, float(np.abs(got - want).max()))

    @staticmethod
    def test_reduces_to_norms():
        """x = [1] (T=1) must reproduce ||h_p||^2 = {1, 2, 3/2, 5/2} * 2^l per profile."""
        for Wmax in [1, 2, 4, 8, 16, 32]:
            pfv = PfVarianceConvolver(Wmax)
            var = pfv.variance(np.array([1.0]))          # shape (P,) == A[:, 0] == ||h_p||^2
            for p, (l, q) in enumerate(pfv.labels):
                w = 1 << l
                want = {0: 1.0, 1: 2.0 * w, 2: 1.5 * w, 3: 2.5 * w}[q]
                assert abs(var[p] - want) < 1e-9, (Wmax, p, l, q, var[p], want)

    @staticmethod
    def test_kernels_match_reference():
        """Check our kernels h_p against the ones ReferencePeakFindingKernel actually uses.

        The reference peak-finder does not expose its kernels: apply() coalesces
        (convolve with h_p) + (multiply by weights) + (max-reduce).  But with the weights
        set to 1, eval_tokens() returns a single profile's value w*y == (h_p * in) at a
        fixed reference time -- a linear functional of the input.  So we feed unit impulses
        (one per DM row) and read eval_tokens() for each profile p; the readout sweeps out
        h_p, recovered up to a time shift and a reversal.  That is exactly the equivalence
        class that leaves Var = ||h_p * x||^2 unchanged, so it is the right thing to pin: if
        the reference's kernel coefficients/shapes/profile-ordering change, this test fails.

        Deterministic (no randomness) -- intended to run once, not every iteration.
        """
        from ..pirate_pybind11 import ReferencePeakFindingKernel

        nt_in, Dout, Dcore = 512, 4, 1     # validated params (see plans/); reads land mid-array
        nt_out = nt_in // Dout
        tout = nt_out // 2                  # middle output bin -> reference time interior, big margins
        J = nt_in                          # impulse-position axis == DM axis (a power of two)

        for Wmax in [1, 2, 4, 8, 16, 32]:
            ker = ReferencePeakFindingKernel(
                subband_counts=[1], max_kernel_width=Wmax,
                beams_per_batch=1, total_beams=1, ndm_out=J, ndm_wt=1,
                nt_out=nt_out, nt_in=nt_in, nt_wt=1, Dcore=Dcore)
            P = ker.P

            # "Identity" of impulses: DM row j carries a unit impulse at time j.
            in_ = np.zeros((1, J, 1, nt_in), dtype=np.float32)
            in_[0, np.arange(J), 0, np.arange(J)] = 1.0
            wt = np.ones((1, 1, 1, P, 1), dtype=np.float32)          # weights = 1 -> read raw y
            out_max = np.zeros((1, J, nt_out), dtype=np.float32)
            out_arg = np.zeros((1, J, nt_out), dtype=np.uint32)
            ker.apply(out_max, out_arg, in_, wt, 0)                  # one apply builds all temp arrays

            kernels, labels = PfVarianceConvolver.peak_finding_kernels(Wmax)
            assert len(kernels) == P, (Wmax, len(kernels), P)

            for p in range(P):
                in_tok = np.zeros((1, J, nt_out), dtype=np.uint32)
                in_tok[0, :, tout] = (p << 8)                       # token = t | (p<<8) | (m<<16), m=t=0
                out = np.zeros((1, J, nt_out), dtype=np.float32)
                ker.eval_tokens(out, in_tok, wt)
                c = out[0, :, tout]                                 # c[j] = h_p[t_ref - j]

                nz = np.nonzero(c)[0]
                assert len(nz) > 0, (Wmax, p, "extracted an all-zero kernel")
                assert nz[0] > 0 and nz[-1] < J - 1, \
                    (Wmax, p, "kernel support reached the array edge; increase nt_in")
                ctrim = c[nz[0]:nz[-1] + 1]                         # trim -> h_p up to shift/reversal
                hp = kernels[p]
                ok = (ctrim.shape == hp.shape) and \
                     (np.allclose(ctrim, hp) or np.allclose(ctrim, hp[::-1]))
                assert ok, (Wmax, p, labels[p], list(ctrim), list(hp))
