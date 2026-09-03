"""PfVarianceConvolver: time series -> peak-finding output variances.

The peak-finding kernels h_p are fixed by the config's max_kernel_width, and the variance of
(h_p * x) for an uncorrelated x is a fixed quadratic form in x -- so it can be read off a
precomputed autocorrelation table rather than by convolving.  That table, and the evaluation,
are all this class is.  See notes/variance_map.tex, "Variance from autocorrelations".

Used by varmap/detrender_free.py (SdPlan._emit) and, on the C++ side, by both SdPlan and
PeakFindingKernel.cu's bare-kernel weights.  The C++ twin is in src_lib/varmap.cpp and
pirate_frb/fast_varmap compares the two.
"""
import numpy as np

from ..utils import integer_log2


###################################   class PfVarianceConvolver   ##################################


class PfVarianceConvolver:
    """Convert time series to variances, for the peak-finding kernels.

    The PfVarianceConvolver has one purpose in life: to convert time series to variances,
    after convolving with the first P pirate peak-finding kernels. See variance() below.
    """
    
    def __init__(self):
        from ..pirate_pybind11 import constants    # lazy: keep this module's top level pybind-free
        self.kernels = self.peak_finding_kernels(int(constants.max_pf_width))
        self.Pmax = len(self.kernels)              # = 3*log2(max_pf_width)+1
        self.Tmax = np.array([len(h) for h in self.kernels], dtype=np.int64)  # per-profile autocorr extent
        assert np.all(np.diff(self.Tmax) >= 0)     # non-decreasing -> Tmax[P-1] == max(Tmax[:P])

        # Autocorrelation table A[p, delta] = sum_t h_p[t] h_p[t+delta] for delta = 0..Tmax-1, where
        # Tmax = self.Tmax[-1] is the longest kernel.  Lags >= len(h_p) vanish (no self-overlap), so
        # row p is the kernel's one-sided autocorrelation, zero-padded out to Tmax.
        self.A = np.zeros((self.Pmax, int(self.Tmax[-1])))   # (Pmax, 2*max_pf_width)
        for p, h in enumerate(self.kernels):
            self.A[p, :len(h)] = self._autocorr(h, len(h))

    @staticmethod
    def peak_finding_kernels(Wmax):
        """Returns a length-Pmax list of 1-d arrays, containing peak-finding kernels."""
        
        Lq = integer_log2(Wmax)       # = log2(Wmax) = number of levels carrying q=1,2,3 profiles
        kernels = [np.ones(1)]        # p=0: finest single sample (l=0, q=0)
        
        for l in range(Lq):           # level l adds the three profiles p = 3l+q (q = 1, 2, 3)
            w = 1 << l
            kernels.append(np.ones(2 * w))
            kernels.append(np.concatenate([0.5 * np.ones(w), np.ones(w),     0.5 * np.ones(w)]))
            kernels.append(np.concatenate([0.5 * np.ones(w), np.ones(2 * w), 0.5 * np.ones(w)]))

        assert len(kernels) == 3 * Lq + 1
        return kernels

    @staticmethod
    def _autocorr(a, maxlag):
        """One-sided autocorrelation over the last axis.

        Computes sum_t a[..., t] a[..., t+k] for lags k = 0..maxlag-1.
        Acts on the last axis (leading axes are spectators); needs 1 <= maxlag <= a.shape[-1].
        Returns shape a.shape[:-1] + (maxlag,).
        """
        
        a = np.asarray(a, dtype=np.float64)
        T = a.shape[-1]
        assert 1 <= maxlag <= T, (maxlag, T)
        return np.stack([(a[..., :T - k] * a[..., k:]).sum(axis=-1) for k in range(maxlag)], axis=-1)

    def variance(self, x, P):
        """Return per-profile variances for a time series.

        Setup: x[..., T] is an array whose last index is time (other indices are spectators).
        Returns an array of variances x[..., P] for the first P peak-finding kernels.

        Formal definition (streamlining notation by removing spectator indices):
          - x is a 1-d time series defined for 0 <= t < T
          - convolve x with a unit Gaussian time series g, defined for -infty < t < infinity
          - convolve with each peak-finding kernel h_p
          - the resulting time series y_p = (x * g * h_p) is statistically time-translation
             invariant; let V[p] be its variance (which is equal for each sample)
          - this function computes x[t] -> V[p]
        """
        x = np.asarray(x, dtype=np.float64)
        assert x.ndim >= 1
        assert 1 <= P <= self.Pmax, (P, self.Pmax)
        T = x.shape[-1]
        assert T >= 1
        d = min(T, int(self.Tmax[P - 1]))   # longest kernel among the first P profiles

        rho = self._autocorr(x, d)           # (..., d)
        rho[..., 1:] *= 2.0                  # +/- delta symmetry of R_x

        return rho @ self.A[:P, :d].T        # (..., d) @ (d, P) -> (..., P)

    # ---------------------------------------------------------------------------
    # Tests (dispatched from pirate_frb/__main__.py via 'test --avar').

    @staticmethod
    def test_random_variance():
        """Compare variance(x, P) to brute-force ||h_p * x||^2, with random spectators/T/P."""
        pfv = PfVarianceConvolver()
        P = int(np.random.randint(1, pfv.Pmax + 1))      # 1..Pmax

        shape = tuple(int(s) for s in np.random.randint(1, 4, size=np.random.randint(1, 4)))
        T = int(np.random.randint(1, 13))               # spans T < and >= Tmax[P-1]
        x = np.random.standard_normal(shape + (T,))

        got = pfv.variance(x, P)
        want = np.empty(shape + (P,))
        for idx in np.ndindex(*shape):
            for p in range(P):
                k = np.convolve(pfv.kernels[p], x[idx])
                want[idx + (p,)] = float((k * k).sum())

        assert got.shape == want.shape, (got.shape, want.shape)
        assert np.allclose(got, want, rtol=1e-9, atol=1e-12), \
            (P, shape, T, float(np.abs(got - want).max()))

    @staticmethod
    def test_reduces_to_norms():
        """x = [1] (T=1) must reproduce ||h_p||^2 = {1, 2, 3/2, 5/2} * 2^l per profile."""
        pfv = PfVarianceConvolver()
        var = pfv.variance(np.array([1.0]), pfv.Pmax)    # (Pmax,) == A[:, 0] == ||h_p||^2
        for p in range(pfv.Pmax):
            l, q = (0, 0) if p == 0 else ((p - 1) // 3, (p - 1) % 3 + 1)   # invert p = 3l+q
            w = 1 << l
            want = {0: 1.0, 1: 2.0 * w, 2: 1.5 * w, 3: 2.5 * w}[q]
            assert abs(var[p] - want) < 1e-9, (p, l, q, var[p], want)
        # P-slicing: variance(x, P) is the length-P prefix of variance(x, Pmax).
        for P in [1, 4, 7, 13, pfv.Pmax]:
            assert np.allclose(pfv.variance(np.array([1.0]), P), var[:P]), P

    @staticmethod
    def test_unimodality():
        """Check that each kernel autocorrelation A_p[delta] is non-negative, and non-increasing in delta.

        This property ("unimodality" of the kernel autocorrelations) is load-bearing, not a
        curiosity: it is one of the two hypotheses of the variance-map monotonicity result in
        notes/variance_map.tex, appendix "Monotonicity of the variance map in the DM bits
        (no detrender)". It holds because every current profile is a co-centered, non-negative
        sum of boxcars. A profile which is not -- say a matched filter for a scattered or
        multi-component pulse -- would break that result itself, not merely its proof, so this
        test exists to fail loudly (with an explanation) if the kernel bank ever changes that way.

        Deterministic -- intended to run once, not every iteration.
        """

        pfv = PfVarianceConvolver()

        # Where to send the reader of a failure. (Section titles, not numbers: the appendix has
        # moved once already, and the lemma numbering is by hand.)
        appendix = ('notes/variance_map.tex, appendix "Monotonicity of the variance map in\n'
                    '  the DM bits (no detrender)"')

        def fmt(h):
            """Format a kernel for an error message (they can be thousands of samples long)."""
            v = [float(x) for x in h]
            if len(v) <= 12:
                return str(v)
            return f"(length {len(v)}) {v[:6]}...{v[-6:]}".replace("]...[", ", ..., ")

        for p in range(pfv.Pmax):
            # Row p of the autocorrelation table: A[p,delta] = sum_t h_p[t] h_p[t+delta], one-sided
            # (A_p is even in delta), and zero-padded past len(h_p). The bank for any smaller
            # max_kernel_width is a prefix of this one (see peak_finding_kernels()), so looping over
            # all Pmax profiles covers every config.
            a = pfv.A[p]

            # The current kernel coefficients (halves and ones) are exactly representable, so their
            # autocorrelations are exact. A future kernel with rounded coefficients could show O(eps)
            # non-monotonicity, which is not the failure this test is looking for.
            eps = 1e-12 * a[0]

            rise = np.nonzero(np.diff(a) > eps)[0]
            if len(rise) > 0:
                d = int(rise[0])
                raise RuntimeError(
                    f"PfVarianceConvolver.test_unimodality: the autocorrelation of peak-finding\n"
                    f"profile p={p} is not monotone: it decreases to A_p[{d}]={float(a[d])}, then rises\n"
                    f"to A_p[{d+1}]={float(a[d+1])}. The kernel is h_{p} = {fmt(pfv.kernels[p])}.\n"
                    f"\n"
                    f"This breaks a hypothesis that downstream results depend on. See\n"
                    f"  {appendix},\n"
                    f"which proves that the variance map never increases when a bit of the DM index is\n"
                    f"set (in particular, that the first entry of every aligned dyadic DM block is the\n"
                    f"largest). Lemma 6 there is exactly the property this test checks, and the\n"
                    f"'Sharpness' subsection gives an explicit two-hump counterexample: a kernel whose\n"
                    f"autocorrelation rises again -- as this one does -- makes the conclusion FALSE, not\n"
                    f"merely unproved, for ordinary gridding weights.\n"
                    f"\n"
                    f"So this is a design decision, not a test to relax: either keep the peak-finding\n"
                    f"kernels co-centered non-negative sums of boxcars (see peak_finding_kernels()), or\n"
                    f"revisit every place which assumes the variance map is DM-bit monotone.")

            if np.min(a) < -eps:
                d = int(np.argmin(a))
                raise RuntimeError(
                    f"PfVarianceConvolver.test_unimodality: the autocorrelation of peak-finding\n"
                    f"profile p={p} is negative at lag {d}: A_p[{d}]={float(a[d])}. The kernel is\n"
                    f"h_{p} = {fmt(pfv.kernels[p])}.\n"
                    f"\n"
                    f"Non-negativity of A_p is implied by the monotonicity checked above (A_p vanishes\n"
                    f"past len(h_p)), so reaching this means the table itself is malformed, not just the\n"
                    f"kernel shape. Either way it breaks the variance-map monotonicity result of\n"
                    f"  {appendix}.")

    @staticmethod
    def test_kernels_match_reference():
        """Check our kernels h_p against the ones ReferencePeakFindingKernel actually uses.

        The reference doesn't expose its kernels (apply() fuses convolve + weight + max-reduce).
        But with weights == 1, eval_tokens() returns the linear functional (h_p * in) at a fixed
        reference time.  Feeding unit impulses (one per DM row) and reading eval_tokens() for each
        profile p sweeps out h_p, up to a time shift and a reversal -- exactly the equivalence
        class that leaves Var = ||h_p * x||^2 unchanged.  So this fails if the reference's kernel
        coefficients/shapes/profile-ordering change.

        Deterministic -- intended to run once, not every iteration.
        """
        from ..pirate_pybind11 import ReferencePeakFindingKernel

        nt_in, Dout, Dcore = 512, 4, 1     # validated params; reads land mid-array
        nt_out = nt_in // Dout
        tout = nt_out // 2                  # middle output bin -> reference time interior, big margins
        J = nt_in                          # impulse-position axis == DM axis (a power of two)

        for Wmax in [1, 2, 4, 8, 16, 32]:
            ker = ReferencePeakFindingKernel(
                subband_counts=[1], max_kernel_width=Wmax,
                beams_per_batch=1, total_beams=1, dm_downsampling=1, time_downsampling=Dout,
                ndm_out=J, ndm_wt=1, nt_out=nt_out, nt_wt=1, Dcore=Dcore)
            P = ker.P

            # "Identity" of impulses: DM row j carries a unit impulse at time j.
            in_ = np.zeros((1, J, 1, nt_in), dtype=np.float32)
            in_[0, np.arange(J), 0, np.arange(J)] = 1.0
            wt = np.ones((1, 1, 1, P, 1), dtype=np.float32)          # weights = 1 -> read raw y
            out_max = np.zeros((1, J, nt_out), dtype=np.float32)
            out_arg = np.zeros((1, J, nt_out), dtype=np.uint32)
            ker.apply(out_max, out_arg, in_, wt, 0)                  # one apply builds all temp arrays

            kernels = PfVarianceConvolver.peak_finding_kernels(Wmax)
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
                assert ok, (Wmax, p, list(ctrim), list(hp))
