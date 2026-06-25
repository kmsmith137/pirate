import numpy as np

from .SparseTile import SparseTile, SparseTileTriple, SparseTilePerM
from ..utils import integer_log2


###################################   class PfVarianceConvolver   ##################################


class PfVarianceConvolver:
    """
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
        """
        One-sided autocorrelation sum_t a[..., t] a[..., t+k] for lags k = 0..maxlag-1.
        Acts on the last axis (leading axes are spectators); needs 1 <= maxlag <= a.shape[-1].
        Returns shape a.shape[:-1] + (maxlag,).
        """
        
        a = np.asarray(a, dtype=np.float64)
        T = a.shape[-1]
        assert 1 <= maxlag <= T, (maxlag, T)
        return np.stack([(a[..., :T - k] * a[..., k:]).sum(axis=-1) for k in range(maxlag)], axis=-1)

    def variance(self, x, P):
        """
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


#######################################   class PfVariance   #######################################


class PfVariance:
    """
    Represents a variance array var[d, p], for delay 0 <= d < 2^rank and peak-finding
    profile 0 <= p < P (P is fixed by the convolver). Variance arrays with more indices
    can be represented by building larger data structures that contain PfVariance objects.

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
      P:         number of peak-finding profiles (each term's last axis has length P).
      terms:     dict mapping dbits (int bitmask) -> ndarray of shape (2^popcount(dbits), P).
    """

    def __init__(self, rank, P):
        assert rank >= 0
        assert P >= 1
        
        self.rank = int(rank)
        self.P = int(P)
        self.terms = {}      # dbits (int) -> (2^popcount(dbits), P) float64

    
    def get_all_dbits(self):
        """Bitwise-OR of all term keys (the union of bits any term depends on)."""
        all_dbits = 0
        for dbits in self.terms:
            all_dbits |= dbits
        return all_dbits

    
    def unpack(self, dbits):
        """
        Expand every term to shape (2^popcount(dbits), P) and return their sum.
        'dbits' must be a superset of every term's dbits.
        """
        
        dbits = int(dbits)
        missing = self.get_all_dbits() & ~dbits
        if missing:
            raise ValueError(f"PfVariance.unpack: dbits={dbits:#x} is too small; a term "
                             f"depends on bit(s) {missing:#x} not present in dbits")
        
        m = dbits.bit_count()
        out = np.zeros((1 << m, self.P), dtype=np.float64)
        
        # Representative delay for each output row (its 'dbits' bits encode the row index).
        d = SparseTile._representative_delay(np.arange(1 << m), dbits)
        for term_dbits, term_arr in self.terms.items():
            out += term_arr[SparseTile._selected_bits_index(d, term_dbits)]
        
        return out

    
    def add_tile(self, tile, convolver):
        """
        Given a singleton SparseTile representing dedisperser output, compute the variance
        and accumulate it into (self.terms).
        """
        assert tile.nf == 1, "PfVariance.add_tile: tile must be a singleton (nf == 1)"
        assert tile.k == self.rank, (tile.k, self.rank)
        tile_var = convolver.variance(tile.data, self.P)[0]   # drop nf==1 axis -> (2^popcount, P)
        self._accumulate(tile.dbits, tile_var, steal=True)    # tile_var is a fresh, unshared temporary

    
    def add(self, pfvar, upper_half=False, scale=1.0):
        """
        Accumulate (scale * pfvar) into self.
        
        If upper_half=True, accumulate the logical upper half of pfvar's delay axis,
        i.e. fix the delay bit to 1 and drop it.

        We require pfvar.rank = self.rank + (upper_half ? 1 : 0).
        We require (self.P <= pfvar.P); extra profiles in pfvar are discarded.
        """
        
        assert self is not pfvar, "PfVariance.add: cannot add a PfVariance to itself"
        assert pfvar.rank == self.rank + (1 if upper_half else 0), (pfvar.rank, self.rank)
        assert pfvar.P >= self.P
        
        topbit = 1 << self.rank
        
        for dbits, arr in pfvar.terms.items():
            if (not upper_half) or (dbits & topbit) == 0:
                self._accumulate(dbits, arr[:, :self.P], scale=scale)   # steal=False: arr is borrowed
            else:
                self._accumulate(dbits & ~topbit, arr[arr.shape[0] // 2:, :self.P], scale=scale)  # upper half


    @staticmethod
    def from_tile(tile, P, convolver):
        """
        Given a singleton SparseTile representing dedisperser output, compute the variance
        and return it as a new PfVariance object.
        """
        pfvar = PfVariance(tile.k, P)
        pfvar.add_tile(tile, convolver)
        return pfvar

    
    def _accumulate(self, dbits, arr, *, scale=1.0, steal=False):
        """
        Accumulate (dbits, scale*arr) into self.terms.
        The 'steal' arg indicates whether it's okay to steal ownership of 'arr'.
        """
        
        dbits = int(dbits)
        assert 0 <= dbits < (1 << self.rank), (dbits, self.rank)
        assert arr.shape == (1 << dbits.bit_count(), self.P), (arr.shape, dbits, self.P)
        
        arr = np.asarray(arr, dtype=np.float64)
        
        if dbits in self.terms:
            self.terms[dbits] += arr if (scale == 1.0) else (scale * arr)
        elif not steal:
            self.terms[dbits] = np.array(arr) if scale == 1.0 else (scale * arr)
        else:
            if (scale != 1.0):
                arr *= scale
            self.terms[dbits] = arr  # steal it

    
    # ---------------------------------------------------------------------------
    # Code below is only called on testing/checking paths.

    
    def add_spectator_low_bits(self, nbits):
        """Returna a new PfVariance of rank (self.rank + nbits)."""
        out = PfVariance(self.rank + nbits, self.P)
        for dbits, arr in self.terms.items():
            out._accumulate(dbits << nbits, arr)
        return out

    
    @staticmethod
    def test_add_truncate_upper_half():
        """add(): p-truncation (self.P <= pfvar.P) and upper_half match the per-tile path.

        A PfVariance built from a raw singleton at the larger P_shared, accumulated with
        truncation (and optionally upper_half), must equal building at the smaller P directly from
        the raw tile -- resp. from its specialize_dbits(1, 1, low=False) upper half. This is the
        hoist PfAvarApproximation relies on (per-sub-block variance built once, shared across trees
        that then truncate P and take the upper DM half in add())."""
        conv = PfVarianceConvolver()
        k = int(np.random.randint(1, 6))                  # k >= 1 for the upper-half case
        tile = SparseTile.make_random(k, k, 0, 1)         # singleton, rank k
        P1 = int(np.random.randint(1, conv.Pmax + 1))     # a tree's P
        P2 = int(np.random.randint(P1, conv.Pmax + 1))    # shared P >= P1
        pv = PfVariance.from_tile(tile, P2, conv)          # shared, built at P2

        # ids=0: add with p-truncation P2 -> P1, vs add_tile at P1.
        a_new = PfVariance(k, P1); a_new.add(pv)
        a_old = PfVariance(k, P1); a_old.add_tile(tile, conv)
        assert np.allclose(a_new.unpack(tile.dbits), a_old.unpack(tile.dbits)), (k, P1, P2)

        # ids>0: add(upper_half) + truncation, vs specialize_dbits(1, 1, low=False) + add_tile.
        spec = tile.specialize_dbits(1, 1, low=False)
        b_new = PfVariance(k - 1, P1); b_new.add(pv, upper_half=True)
        b_old = PfVariance(k - 1, P1); b_old.add_tile(spec, conv)
        assert np.allclose(b_new.unpack(spec.dbits), b_old.unpack(spec.dbits)), (k, P1, P2)

        # scale: add(scale=c) accumulates c * pfvar (both ids branches).
        c = float(np.random.uniform(0.5, 2.0))
        for base, ph in ((a_new, False), (b_new, True)):
            s = PfVariance(base.rank, P1); s.add(pv, upper_half=ph, scale=c)
            assert np.allclose(s.unpack(base.get_all_dbits()),
                               c * base.unpack(base.get_all_dbits())), (k, P1, P2, c, ph)


#######################################   class PfAvarExact   ######################################


class PfAvarExact:
    """Exact analytic peak-finding variances for a DedispersionPlan (all DedispersionTrees).

    For each DedispersionTree (one per (downsampling level, early trigger)), computes the
    peak-finding output variance of every (input frequency channel, multiplet) pair, plus the
    frequency-summed variance per multiplet.  Each tree is processed "from scratch" (no reuse of
    computations across trees).  Every PfVariance for tree itree has rank tree_r[itree] - tree_R[itree].

    Members
    -------
      ntrees:      number of DedispersionTrees (= plan.ntrees).
      tree_r:      length-ntrees int array; tree_r[itree] = amb_rank + early_dd_rank, the tree's
                   kept-output rank (equivalently config.tree_rank - delta - (ids>0 ? 1 : 0), where
                   delta = early-trigger reduction and ids = ds_level; so tree_r == config.tree_rank
                   only for tree 0).
      tree_R:      length-ntrees int array; tree_R[itree] = the tree's pf_rank (NOT the config pf_rank).
      tree_P:      length-ntrees int array; tree_P[itree] = the tree's nprofiles (= 1 + 3 log2(Wmax)).
      convolver:   a single shared PfVarianceConvolver (full kernel bank to constants.max_pf_width;
                   every tree's tree_P[itree] <= convolver.Pmax).
      tree_fs:     length-ntrees list of FrequencySubbands (= tree.frequency_subbands).
      plan, nfreq: the DedispersionPlan and the number of input frequency channels.
      per_tfm:     (ntrees, nfreq, M) ragged; per_tfm[itree][ifreq] is None (ifreq below the tree's
                   truncated band) or a length-M list of (PfVariance of rank tree_r-tree_R, or
                   None where multiplet m doesn't overlap ifreq).
      per_tm:      (ntrees, M) ragged; per_tm[itree][m] is the frequency-summed PfVariance for tree
                   itree, multiplet m (also rank tree_r-tree_R).
    """

    def __init__(self, plan, progress=False):
        # If progress is set, print one line per tree and one '.' per 1000 input freq channels.
        self.plan, self.nfreq, self.ntrees = plan, int(plan.nfreq), int(plan.ntrees)
        self.convolver = PfVarianceConvolver()   # shared full kernel bank; sliced per-tree by P

        # First line is equivalent to: tree_r[itree] = config.tree_rank - delta - (ids > 0).
        self.tree_r = np.array([t.amb_rank + t.early_dd_rank for t in plan.trees])
        self.tree_R = np.array([t.frequency_subbands.pf_rank for t in plan.trees])
        self.tree_P = np.array([t.nprofiles for t in plan.trees])
        self.tree_fs = [t.frequency_subbands for t in plan.trees]
        self._tree_ids = np.array([t.ds_level for t in plan.trees])

        full_cm = np.asarray(plan.config.make_channel_map(), dtype=np.float64)

        # per_tfm: (ntrees,nfreq,M) -> (length-M list of (PfVariance rank r-R, or None), or None)
        # per_tm:  (ntrees,M) -> (PfVariance rank r-R)   [accumulated alongside per_tfm]
        self.per_tfm = [ None ] * self.ntrees
        self.per_tm = [ None ] * self.ntrees

        for itree in range(self.ntrees):
            rho = int(self.tree_r[itree] - self.tree_R[itree])
            P, M = int(self.tree_P[itree]), self.tree_fs[itree].M
            self.per_tfm[itree] = [ None ] * self.nfreq
            self.per_tm[itree] = [ PfVariance(rho, P) for _ in range(M) ]

        for itree in range(self.ntrees):
            if progress:
                print(f"  PfAvarExact tree {itree}/{self.ntrees}: ", end="", flush=True)
            self._process_tree(itree, full_cm, progress)
            if progress:
                print(flush=True)

    def _process_tree(self, itree, full_cm, progress):
        # Fill per_tfm[itree][ifreq] (length-M list of PfVariance/None, or None where ifreq is below
        # the tree's truncated band) and accumulate per_tm[itree][m] += per_tfm[itree][ifreq][m].
        fs, P, ids = self.tree_fs[itree], int(self.tree_P[itree]), int(self._tree_ids[itree])
        rho_cm = int(self.tree_r[itree]) + (ids > 0)   # = tree_r + (ids>0) = config.tree_rank - delta
        cm = np.ascontiguousarray(full_cm[: (1 << rho_cm) + 1])    # truncate to first 2^rho_cm channels

        for ifreq in range(self.nfreq):
            if progress and (ifreq + 1) % 1000 == 0:
                print(".", end="", flush=True)
            if not (ifreq < cm[0] and ifreq + 1 > cm[-1]):        # ifreq below the truncated band
                continue
            ssa = SparseTilePerM.make_dedispersion_output(cm, ifreq, fs, upper_half_only=(ids > 0))
            row = [None if t is None else PfVariance.from_tile(t, P, self.convolver) for t in ssa.per_m]
            self.per_tfm[itree][ifreq] = row

            for m, pv in enumerate(row):
                if pv is not None:
                    self.per_tm[itree][m].add(pv)


###################################   class PfAvarApproximation   ##################################


class PfAvarApproximation:
    """Approximate analytic peak-finding variances for a DedispersionPlan (all DedispersionTrees).

    Like PfAvarExact, but compressed and approximated. Every PfVariance for tree itree has rank
    r[itree] - L[itree] (the WEIGHTS' DM resolution 2^(r-L)) and is stored per coarse-freq channel
    (the 2^R[itree] level-0 subbands). Writing rho_cm = r + (ids>0) = config.tree_rank - delta for
    the tree's pre-upper-half DD rank (it processes the top 2^rho_cm tree-channels):

    The approximation (sub-block-mean): rather than coherently dedispersing each coarse-freq's full
    2^(rho_cm-R)-channel block, we iterate only to level klevel = rho_cm-L (giving 2^L finer
    "sub-block" coarse-freqs f', each a 2^(rho_cm-L)-channel dedispersion) and take, into the coarse
    f = f' >> (L-R), the MEAN of its 2^(L-R) sub-block variances. This drops the inter-sub-block
    cross-covariances and reads each sub-block's own DMs as the coarse-DM axis. The mean (factor
    2^-(L-R) on the sub-block-variance sum) is the 1/sqrt(2)-per-level DD normalization: the
    full-block dedispersion runs L-R more 1/sqrt(2) levels than a sub-block, so its variance equals
    the sub-block MEAN, not the sum -- analogous to the 1/(ihi-ilo) factor get_per_m applies across
    coarse-freqs. (When L == R this reduces to the plain per-coarse-freq dedispersion.)

    A single full-band (rank config.tree_rank) gridding SparseTileTriple is iterated ONCE per input
    freq channel and shared across all trees: the iteration is rank-agnostic and footprint-local,
    and a tree's truncated channel map is a prefix of the full one, so the first 2^L sub-blocks at
    level klevel reproduce the truncated tree's sub-blocks exactly. At each level the RAW sub-block
    singletons are converted to PfVariance ONCE (at the max P over the trees at that level) and
    shared across those trees; each tree then coarsify-means them into its 2^R coarse-freqs via
    PfVariance.add(upper_half=ids>0), which truncates the p-axis to the tree's P and (for ids>0)
    keeps the upper coarse-DM half (rank rho_cm-L -> r-L) -- equivalent to specializing each tile,
    but with the expensive variance evaluated once per level instead of once per tree.

    Per-multiplet variances are reconstructed on demand (get_per_fm / get_per_m): a multiplet m
    spans a coarse-freq range [ilo, ihi) (= fs.m_to_ilo(m)..m_to_ihi(m)); its variance is the
    mean of the per-coarse-freq variances over that range -- i.e. (1/(ihi-ilo)) * sum -- where
    the 1/(ihi-ilo) factor is the inter-coarse-freq DD normalization (2^-l in variance for a
    width-2^l subband) that the exact subband dedispersion would apply.

    Members
    -------
      ntrees:      number of DedispersionTrees (= plan.ntrees).
      tree_r:      length-ntrees int array; tree_r[itree] = amb_rank + early_dd_rank (matches
                   PfAvarExact.tree_r). The PfVariance rank is tree_r[itree] - tree_L[itree].
      tree_R:      length-ntrees int array; tree_R[itree] = the tree's pf_rank (NOT the config pf_rank).
      tree_L:      length-ntrees int array; tree_L[itree] = log2(tree's wt_dm_downsampling). Read per
                   tree -- no assumption that early-trigger trees share L/R with their siblings;
                   only the structural 0 <= R <= L <= r is required (asserted per tree).
      tree_P:      length-ntrees int array; tree_P[itree] = the tree's nprofiles (= 1 + 3 log2(Wmax)).
      convolver:   a single shared PfVarianceConvolver (full kernel bank to constants.max_pf_width;
                   every tree's tree_P[itree] <= convolver.Pmax).
      tree_fs:     length-ntrees list of FrequencySubbands (= tree.frequency_subbands).
      plan, nfreq: the DedispersionPlan and the number of input frequency channels.
      per_tff:     (ntrees, nfreq, 2^R[itree]) ragged; per_tff[itree][ifreq][f] is the single-
                   channel PfVariance (rank r-L) of coarse-freq f, or None (no overlap).
      per_tf:      (ntrees, 2^R[itree]) ragged; per_tf[itree][f] = sum over ifreq of
                   per_tff[itree][ifreq][f].
    """

    def __init__(self, plan, progress=False):
        # If progress is set, print one '.' per 1000 input freq channels.
        self.plan, self.nfreq, self.ntrees = plan, int(plan.nfreq), int(plan.ntrees)
        self.convolver = PfVarianceConvolver()   # shared full kernel bank; sliced per-tree by P
        
        # First line is equivalent to: tree_r[itree] = config.tree_rank - delta - (ids > 0).
        self.tree_r = np.array([t.amb_rank + t.early_dd_rank for t in plan.trees])
        self.tree_R = np.array([t.frequency_subbands.pf_rank for t in plan.trees])
        self.tree_L = np.array([integer_log2(int(t.pf.wt_dm_downsampling)) for t in plan.trees])
        self.tree_P = np.array([t.nprofiles for t in plan.trees])
        self.tree_fs = [t.frequency_subbands for t in plan.trees]
        assert np.all((self.tree_R >= 0) & (self.tree_R <= self.tree_L) & (self.tree_L <= self.tree_r))

        # Per-tree validation + derived ids flag and iterate level klevel = rho_cm - L = (r+ids) - L.
        self._tree_ids = np.array([t.ds_level for t in plan.trees])
        self._tree_klevel = self.tree_r - self.tree_L + (self._tree_ids > 0)
        self._max_klevel = np.max(self._tree_klevel)

        # max P (or L) among all trees with a given klevel.
        self._klevel_Pmax = np.full(self._max_klevel+1, -1, dtype=int)
        self._klevel_Lmax = np.full(self._max_klevel+1, -1, dtype=int)

        for itree,k in enumerate(self._tree_klevel):
            self._klevel_Pmax[k] = max(self._klevel_Pmax[k], self.tree_P[itree])
            self._klevel_Lmax[k] = max(self._klevel_Lmax[k], self.tree_L[itree])

        full_cm = np.asarray(plan.config.make_channel_map(), dtype=np.float64)

        # per_tff: (ntrees,nfreq,2^R) -> (PfVariance rank r-L, or None)
        # per_tf:  (ntrees,2^R) -> (PfVariance rank r-L)
        self.per_tff = [ None ] * self.ntrees
        self.per_tf = [ None ] * self.ntrees

        for itree in range(self.ntrees):
            r, R, L, P = int(self.tree_r[itree]), int(self.tree_R[itree]), int(self.tree_L[itree]), int(self.tree_P[itree])
            self.per_tff[itree] = [ [None]*(1<<R) for _ in range(self.nfreq) ]
            self.per_tf[itree] = [ PfVariance(r-L,P) for _ in range(1 << R) ]

        for ifreq in range(self.nfreq):
            if progress and (ifreq + 1) % 1000 == 0:
                print(".", end="", flush=True)
                
            sarr = SparseTileTriple.make_tree_gridding_output(full_cm, ifreq)   # rank config.tree_rank, level 0
            
            for k in range(0, self._max_klevel + 1):
                self._process_klevel(sarr, k, ifreq)
                if k < self._max_klevel:
                    sarr = sarr.iterate()


    def _process_klevel(self, sarr, k, ifreq):
        if self._klevel_Lmax[k] < 0:
            return   # no trees at this klevel

        for fp in range(1 << self._klevel_Lmax[k]):
            tile = sarr.get_singleton(fp, allow_none=True)
            if tile is None:
                continue

            pv = PfVariance.from_tile(tile, self._klevel_Pmax[k], self.convolver)

            for itree in range(self.ntrees):
                if self._tree_klevel[itree] != k:
                    continue
                
                r, R, L, P = int(self.tree_r[itree]), int(self.tree_R[itree]), int(self.tree_L[itree]), int(self.tree_P[itree])
                upper_half = (int(self._tree_ids[itree]) > 0)
                norm = 2.0 ** (-(L - R))

                if fp >= (1 << L):
                    continue                                   # sub-block fp is outside this tree

                f = fp >> (L - R)                              # coarsify f-index by 2^(L-R)

                if self.per_tff[itree][ifreq][f] is None:
                    self.per_tff[itree][ifreq][f] = PfVariance(r - L, P)

                self.per_tff[itree][ifreq][f].add(pv, upper_half=upper_half, scale=norm)
                self.per_tf[itree][f].add(pv, upper_half=upper_half, scale=norm)

    
    def get_per_fm(self, itree, ifreq, m):
        """Approximate single-channel variance for tree itree, (ifreq, multiplet m): the mean of
        per_tff over the multiplet's coarse-freq range. Returns None if no coarse-freq overlaps."""
        rho = int(self.tree_r[itree]) - int(self.tree_L[itree])
        fs = self.tree_fs[itree]
        ilo, ihi = fs.m_to_ilo(m), fs.m_to_ihi(m)
        scale = 1.0 / (ihi - ilo)            # mean over the coarse-freq range
        acc = None
        for f in range(ilo, ihi):
            pv = self.per_tff[itree][ifreq][f]
            if pv is not None:
                if acc is None:
                    acc = PfVariance(rho, int(self.tree_P[itree]))
                acc.add(pv, scale=scale)
        return acc

    
    def get_per_m(self, itree, m):
        """Approximate frequency-summed variance for tree itree, multiplet m: the mean of per_tf
        over the multiplet's coarse-freq range. Always returns a PfVariance."""
        rho = int(self.tree_r[itree]) - int(self.tree_L[itree])
        fs = self.tree_fs[itree]
        ilo, ihi = fs.m_to_ilo(m), fs.m_to_ihi(m)
        scale = 1.0 / (ihi - ilo)            # mean over the coarse-freq range
        acc = PfVariance(rho, int(self.tree_P[itree]))
        for f in range(ilo, ihi):
            acc.add(self.per_tf[itree][f], scale=scale)
        return acc
