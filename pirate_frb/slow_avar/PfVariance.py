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

from .SparseTile import SparseTile, SparseTileTriple, SparseTilePerM


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


class PfVariance:
    """Compressed peak-finding output variance var[d, p], for delay 0 <= d < 2^rank and
    peak-finding profile 0 <= p < P (P is fixed by the convolver).

    The variance is stored as a small sum of terms, each of which depends on the delay d
    through only a few of its bits.  A term is keyed by an integer bitmask 'dbits' (bit b set
    means the term depends on digit d_b) and holds an array of shape (2^popcount(dbits), P)
    indexed by (compacted selected-bit pattern of d, profile).  The full variance is

        var[d, p] = sum over terms (dbits, arr) of arr[sel(d, dbits), p],

    where sel(d, dbits) = SparseTile._selected_bits_index (highest selected bit is the most
    significant) -- the same delay-axis convention SparseTile.data uses.  Keeping the terms
    factored (rather than expanding to a dense (2^rank, P) array) is the compression.

    Members
    -------
      rank:      number of delay bits (delays run over 0 <= d < 2^rank).
      convolver: PfVarianceConvolver defining the P peak-finding profiles.
      terms:     dict mapping dbits (int bitmask) -> ndarray of shape (2^popcount(dbits), P).
    """

    def __init__(self, rank, convolver):
        self.rank = int(rank)
        self.convolver = convolver
        self.terms = {}      # dbits (int) -> (2^popcount(dbits), P) float64

    @staticmethod
    def from_tile(tile, convolver):
        """Build a PfVariance from a single singleton SparseTile.

        Shorthand for PfVariance(tile.k, convolver) followed by add_tile(tile): the result
        has rank == tile.k and one term (keyed by tile.dbits). 'convolver' must be supplied --
        the peak-finding profiles can't be inferred from the tile.
        """
        pfvar = PfVariance(tile.k, convolver)
        pfvar.add_tile(tile)
        return pfvar

    def get_all_dbits(self):
        """Bitwise-OR of all term keys (the union of bits any term depends on)."""
        all_dbits = 0
        for dbits in self.terms:
            all_dbits |= dbits
        return all_dbits

    def unpack(self, dbits):
        """Expand every term to shape (2^popcount(dbits), P) and return their sum.

        'dbits' must be a superset of every term's dbits (otherwise it cannot represent some
        term's delay-dependence); a ValueError is raised if it is too small.
        """
        dbits = int(dbits)
        missing = self.get_all_dbits() & ~dbits
        if missing:
            raise ValueError(f"PfVariance.unpack: dbits={dbits:#x} is too small; a term "
                             f"depends on bit(s) {missing:#x} not present in dbits")
        m = dbits.bit_count()
        out = np.zeros((1 << m, self.convolver.P), dtype=np.float64)
        # Representative delay for each output row (its 'dbits' bits encode the row index).
        d = SparseTile._representative_delay(np.arange(1 << m), dbits)
        for term_dbits, term_arr in self.terms.items():
            out += term_arr[SparseTile._selected_bits_index(d, term_dbits)]
        return out

    def add_tile(self, tile):
        """Convolve a singleton SparseTile (tile.k == self.rank) into a new term.

        tile.data has shape (1, 2^popcount(tile.dbits), nt); the convolver maps its length-nt
        time axis to the P profiles, giving a term (tile.dbits, (2^popcount, P)).  tile.tshifts
        and tile.t0 are irrelevant (the variance is time-shift invariant) and are ignored.
        """
        assert tile.nf == 1, "PfVariance.add_tile: tile must be a singleton (nf == 1)"
        assert tile.k == self.rank, (tile.k, self.rank)
        tile_var = self.convolver.variance(tile.data)[0]   # drop the nf==1 axis -> (2^popcount, P)
        self._accumulate(tile.dbits, tile_var)

    def add(self, pfvar):
        """Accumulate all terms of another PfVariance into self."""
        assert pfvar.rank == self.rank, (pfvar.rank, self.rank)
        for dbits, arr in pfvar.terms.items():
            self._accumulate(dbits, arr)

    def scaled(self, c):
        """Return a new PfVariance (same rank/convolver) with every term multiplied by c."""
        out = PfVariance(self.rank, self.convolver)
        for dbits, arr in self.terms.items():
            out._accumulate(dbits, c * arr)
        return out

    def add_spectator_low_bits(self, nbits):
        """Return a new PfVariance of rank (self.rank + nbits) that is independent of the new
        low 'nbits' delay bits (spectators): each term keeps its array, but its dbits shift up
        by nbits (old bit j -> bit j+nbits). Used to raise a coarser variance to a finer rank
        so it can be compared against a finer one."""
        out = PfVariance(self.rank + nbits, self.convolver)
        for dbits, arr in self.terms.items():
            out._accumulate(dbits << nbits, arr)
        return out

    def _accumulate(self, dbits, arr):
        # Add the term (dbits -> arr), summing into any existing same-dbits term.
        dbits = int(dbits)
        assert 0 <= dbits < (1 << self.rank), (dbits, self.rank)
        assert arr.shape == (1 << dbits.bit_count(), self.convolver.P), \
            (arr.shape, dbits, self.convolver.P)
        if dbits in self.terms:
            self.terms[dbits] = self.terms[dbits] + arr
        else:
            self.terms[dbits] = np.array(arr, dtype=np.float64)   # copy (avoid aliasing)


class PfAvarExact:
    """Exact analytic peak-finding variances for a DedispersionPlan (all DedispersionTrees).

    For each DedispersionTree (one per (downsampling level, early trigger)), computes the
    peak-finding output variance of every (input frequency channel, multiplet) pair, plus the
    frequency-summed variance per multiplet.  Each tree is processed "from scratch" (no reuse of
    computations across trees).  Every PfVariance for tree itree has rank r[itree] - R[itree].

    Members
    -------
      ntrees:      number of DedispersionTrees (= plan.ntrees).
      r:           length-ntrees int array; r[itree] = config.tree_rank - delta - (ids>0 ? 1 : 0),
                   the tree's kept-output rank (delta = early-trigger reduction, ids = ds_level).
                   It already includes the early-trigger -delta and the downsampled -1, so
                   r == config.tree_rank only for tree 0.
      R:           length-ntrees int array; R[itree] = the tree's pf_rank (NOT the config pf_rank).
      Wmax:        length-ntrees int array; Wmax[itree] = the tree's max peak-finding kernel width.
      convolvers:  length-ntrees list of PfVarianceConvolver (one per tree).
      fs:          length-ntrees list of FrequencySubbands (= tree.frequency_subbands).
      plan, nfreq: the DedispersionPlan and the number of input frequency channels.
      per_tfm:     (ntrees, nfreq, M) ragged; per_tfm[itree][ifreq] is None (ifreq below the tree's
                   truncated band) or a length-M list of (PfVariance of rank r[itree]-R[itree], or
                   None where multiplet m doesn't overlap ifreq).
      per_tm:      (ntrees, M) ragged; per_tm[itree][m] is the frequency-summed PfVariance for tree
                   itree, multiplet m (also rank r[itree]-R[itree]).
    """

    def __init__(self, plan, progress=False):
        # If progress is set, print one line per tree and one '.' per 1000 input freq channels
        # (this constructor is the slow part of 'pirate_frb check_avar_approximation').
        cfg = plan.config
        full_cm = np.asarray(cfg.make_channel_map(), dtype=np.float64)
        self.plan, self.nfreq, self.ntrees = plan, int(plan.nfreq), int(plan.ntrees)
        # r[itree] = config.tree_rank - delta - (ids>0 ? 1 : 0)
        self.r = np.array([cfg.tree_rank - (t.pri_dd_rank - t.early_dd_rank) - (t.ds_level > 0)
                           for t in plan.trees])
        self.R = np.array([t.frequency_subbands.pf_rank for t in plan.trees])
        self.Wmax = np.array([t.pf.max_width for t in plan.trees])
        self.fs = [t.frequency_subbands for t in plan.trees]
        self.convolvers = [PfVarianceConvolver(int(w)) for w in self.Wmax]   # one per tree (lightweight)

        self.per_tfm = []
        for itree, tree in enumerate(plan.trees):
            if progress:
                print(f"  PfAvarExact tree {itree}/{self.ntrees}: ", end="", flush=True)
            self.per_tfm.append(self._make_per_fm(itree, tree, full_cm, progress))
            if progress:
                print(flush=True)
        self.per_tm = [self._make_per_m(itree) for itree in range(self.ntrees)]

    def _make_per_fm(self, itree, tree, full_cm, progress=False):
        # per_fm[ifreq]: length-M list of (PfVariance or None) for tree itree, or None (no overlap).
        fs, conv, ids = self.fs[itree], self.convolvers[itree], tree.ds_level
        rho_cm = self.plan.config.tree_rank - (tree.pri_dd_rank - tree.early_dd_rank)   # = r + (ids>0)
        cm = np.ascontiguousarray(full_cm[: (1 << rho_cm) + 1])    # truncate to first 2^rho_cm channels
        per_fm = []
        for ifreq in range(self.nfreq):
            if progress and (ifreq + 1) % 1000 == 0:
                print(".", end="", flush=True)
            if not (ifreq < cm[0] and ifreq + 1 > cm[-1]):        # ifreq below the truncated band
                per_fm.append(None)
                continue
            ssa = SparseTilePerM.make_dedispersion_output(cm, ifreq, fs, upper_half_only=(ids > 0))
            per_fm.append([None if t is None else PfVariance.from_tile(t, conv) for t in ssa.per_m])
        return per_fm

    def _make_per_m(self, itree):
        # per_m[m]: frequency-summed PfVariance for tree itree, multiplet m.
        rho = int(self.r[itree]) - int(self.R[itree])
        conv, per_fm = self.convolvers[itree], self.per_tfm[itree]
        per_m = []
        for m in range(self.fs[itree].M):
            acc = PfVariance(rho, conv)
            for row in per_fm:
                if row is not None and row[m] is not None:
                    acc.add(row[m])
            per_m.append(acc)
        return per_m


class PfAvarApproximation:
    """Approximate analytic peak-finding variances for a DedispersionPlan (tree 0 only, so far).

    Like PfAvarExact, but compressed and approximated, and consistent with PfAvarExact at tree 0
    (it reads all per-tree quantities from plan.trees[0], so r/R/Wmax match PfAvarExact.{r,R,Wmax}[0]).
    Variances are stored per coarse-freq channel (the 2^R level-0 subbands, tree-freq range
    2^(r-R) f <= tree-freq < 2^(r-R) (f+1)), at the WEIGHTS' DM resolution 2^(r-L): each PfVariance
    has rank r-L.

    The approximation: rather than coherently dedispersing each coarse-freq's full 2^(r-R)-channel
    block, we iterate only to level k = r-L (giving 2^L finer "sub-block" coarse-freqs f', each a
    2^(r-L)-channel dedispersion with 2^(r-L) DMs) and take, into the coarse f = f' >> (L-R), the
    MEAN of its 2^(L-R) sub-block variances. This drops the inter-sub-block cross-covariances and
    reads each sub-block's own 2^(r-L) DMs as the coarse-DM axis. The mean (factor 2^-(L-R) on the
    sub-block-variance sum) is the 1/sqrt(2)-per-level DD normalization: because the full-block
    dedispersion runs L-R more 1/sqrt(2) levels than a sub-block, its variance equals the sub-block
    MEAN, not the sum -- analogous to the 1/(ihi-ilo) factor get_per_m applies across coarse-freqs.
    (When L == R this reduces to the plain per-coarse-freq dedispersion.)

    Per-multiplet variances are reconstructed on demand (get_per_fm / get_per_m): a multiplet m
    spans a coarse-freq range [ilo, ihi) (= fs.m_to_ilo(m)..m_to_ihi(m)); its variance is the
    mean of the per-coarse-freq variances over that range -- i.e. (1/(ihi-ilo)) * sum -- where
    the 1/(ihi-ilo) factor is the inter-coarse-freq DD normalization (2^-l in variance for a
    width-2^l subband) that the exact subband dedispersion would apply.

    Members
    -------
      r, R, L:       tree 0's rank (= config.tree_rank), pf_rank (= fs.pf_rank), and
                     log2(wt_dm_downsampling), from plan.trees[0]. The PfVariance rank rho = r - L
                     is recomputed locally (as a plain variable) where needed.
      fs:            FrequencySubbands (subband scheme) of plan.trees[0].
      Wmax:          tree 0's max peak-finding kernel width.
      convolver:     PfVarianceConvolver(Wmax), shared by all PfVariance objects.
      nfreq:         number of input frequency channels.
      per_ff:        (nfreq, 2^R) list-of-lists; per_ff[ifreq][f] is the single-channel PfVariance
                     (rank r-L) of coarse-freq f for input channel ifreq, or None (no overlap).
      per_f:         length-2^R list of PfVariance; per_f[f] = sum over ifreq of per_ff[ifreq][f].
    """

    def __init__(self, plan, progress=False):
        # If progress is set, print one '.' per 1000 input freq channels.
        tree0 = plan.trees[0]                            # base tree (ids=0, delta=0)
        self.r = int(plan.config.tree_rank)              # = tree0's kept-output rank (ids=0, delta=0)
        self.fs = tree0.frequency_subbands
        self.R = self.fs.pf_rank
        wt_dd = int(tree0.pf.wt_dm_downsampling)
        self.L = wt_dd.bit_length() - 1                  # log2(wt_dm_downsampling)
        assert wt_dd == (1 << self.L), "wt_dm_downsampling must be a power of two"
        assert self.R <= self.L <= self.r, (self.R, self.L, self.r)
        self.Wmax = int(tree0.pf.max_width)
        self.nfreq = int(plan.nfreq)
        self.convolver = PfVarianceConvolver(self.Wmax)

        channel_map = np.asarray(plan.config.make_channel_map(), dtype=np.float64)
        if progress:
            print("  PfAvarApproximation: ", end="", flush=True)
        self.per_ff = self._make_per_ff(channel_map, progress)
        if progress:
            print(flush=True)
        self.per_f = self._make_per_f()

    def _make_per_ff(self, channel_map, progress=False):
        # per_ff[ifreq][f]: PfVariance (rank r-L) for coarse-freq f, or None (no overlap). Built by
        # iterating to level r-L (2^L sub-block coarse-freqs f', each rank-(r-L)) and taking, into
        # coarse f = f' >> (L-R), the MEAN of its 2^(L-R) sub-block variances (the 2^-(L-R) factor
        # is the DD 1/sqrt(2)-per-level normalization). See the class docstring.
        R, L = int(self.R), int(self.L)
        rho = self.r - L                                  # PfVariance rank
        norm = 2.0 ** (-(L - R))                          # DD normalization: sub-block-variance mean
        per_ff = []
        for ifreq in range(self.nfreq):
            if progress and (ifreq + 1) % 1000 == 0:
                print(".", end="", flush=True)
            sarr = SparseTileTriple.make_tree_gridding_output(channel_map, ifreq)
            for _ in range(self.r - L):
                sarr = sarr.iterate()            # now k == r-L: 2^L coarse-freqs, 2^(r-L) DMs
            row = [None] * (1 << R)
            for fp in range(1 << L):                       # fine "sub-block" coarse-freq f'
                tile = sarr.get_singleton(fp, allow_none=True)   # rank-(r-L) singleton, or None
                if tile is None:
                    continue
                f = fp >> (L - R)                          # coarsify the f-index by 2^(L-R)
                if row[f] is None:
                    row[f] = PfVariance(rho, self.convolver)
                row[f].add_tile(tile)                      # sum the sub-block variances ...
            row = [None if pv is None else pv.scaled(norm) for pv in row]   # ... then mean (2^-(L-R))
            per_ff.append(row)
        return per_ff

    def _make_per_f(self):
        # per_f[f]: frequency-summed PfVariance for coarse-freq f (sum over per_ff[:, f]).
        rho = self.r - self.L
        per_f = []
        for f in range(1 << self.R):
            acc = PfVariance(rho, self.convolver)
            for ifreq in range(self.nfreq):
                pv = self.per_ff[ifreq][f]
                if pv is not None:
                    acc.add(pv)
            per_f.append(acc)                    # always a PfVariance (empty if f overlaps nothing)
        return per_f

    def get_per_fm(self, ifreq, m):
        """Approximate single-channel variance for (ifreq, multiplet m): the mean of per_ff over
        the multiplet's coarse-freq range. Returns None if no coarse-freq in the range overlaps."""
        rho = self.r - self.L
        ilo, ihi = self.fs.m_to_ilo(m), self.fs.m_to_ihi(m)
        acc = None
        for f in range(ilo, ihi):
            pv = self.per_ff[ifreq][f]
            if pv is not None:
                if acc is None:
                    acc = PfVariance(rho, self.convolver)
                acc.add(pv)
        return None if acc is None else acc.scaled(1.0 / (ihi - ilo))

    def get_per_m(self, m):
        """Approximate frequency-summed variance for multiplet m: the mean of per_f over the
        multiplet's coarse-freq range. Always returns a PfVariance."""
        rho = self.r - self.L
        ilo, ihi = self.fs.m_to_ilo(m), self.fs.m_to_ihi(m)
        acc = PfVariance(rho, self.convolver)
        for f in range(ilo, ihi):
            acc.add(self.per_f[f])
        return acc.scaled(1.0 / (ihi - ilo))
