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

where R_a[delta] = sum_t a[t] a[t+delta] is the (auto)correlation.  For profile p only lags
|delta| < min(T, len(h_p)) contribute: R_x vanishes past T = len(x), and R_{h_p} past the
kernel length.  We build the full kernel bank (out to constants.max_pf_width) and precompute
every kernel's autocorrelation once into a table -- capturing each in full, with no assumption
about len(x) -- after which variance(x, P) is a small matmul over the first P profiles.
"""

import numpy as np

from .SparseTile import SparseTile, SparseTileTriple, SparseTilePerM
from ..utils import integer_log2


###################################   class PfVarianceConvolver   ##################################


class PfVarianceConvolver:
    """Computes Var(h_p * x) for the peak-finding profiles p, given a short kernel x.

    Constructs the full kernel bank out to constants.max_pf_width (Pmax = 3*log2(max_pf_width)+1
    profiles); call variance(x, P) for the first P profiles.  The bank is nested in P (each kernel
    depends only on its (level, shape), not the max width), so the first P profiles are exactly the
    bank of a peak-finder with P = 3*log2(Wmax)+1 profiles.  variance() maps an input of shape
    (..., T) to output (..., P), with the leading axes treated as spectators.
    """

    def __init__(self):
        from ..pirate_pybind11 import constants    # lazy: keep this module's top level pybind-free
        self.kernels, self.labels = self.peak_finding_kernels(int(constants.max_pf_width))
        self.Pmax = len(self.kernels)              # = 3*log2(max_pf_width)+1
        self.Tmax = np.array([len(h) for h in self.kernels], dtype=np.int64)  # per-profile autocorr extent
        assert np.all(np.diff(self.Tmax) >= 0)     # non-decreasing -> Tmax[P-1] == max(Tmax[:P])
        self.A = self._autocorr_table(self.kernels, int(self.Tmax[-1]))   # (Pmax, 2*max_pf_width)

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
        Lq = integer_log2(Wmax)       # = log2(Wmax) = number of levels carrying q=1,2,3 profiles

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

    def variance(self, x, P):
        """Var(h_p * x) for the first P profiles.  x: shape (..., T) -> out: shape (..., P)."""
        x = np.asarray(x, dtype=np.float64)
        assert x.ndim >= 1
        assert 1 <= P <= self.Pmax, (P, self.Pmax)
        T = x.shape[-1]
        assert T >= 1
        d = min(T, int(self.Tmax[P - 1]))   # longest kernel among the first P profiles

        # One-sided autocorrelation of x, lags 0..d-1, over the last axis
        # (leading/spectator axes broadcast through the sum).
        rho = np.stack([(x[..., :T - k] * x[..., k:]).sum(axis=-1) for k in range(d)], axis=-1)  # (..., d)
        rho[..., 1:] *= 2.0          # +/- delta symmetry of R_x

        return rho @ self.A[:P, :d].T  # (..., d) @ (d, P) -> (..., P)

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
        for p, (l, q) in enumerate(pfv.labels):
            w = 1 << l
            want = {0: 1.0, 1: 2.0 * w, 2: 1.5 * w, 3: 2.5 * w}[q]
            assert abs(var[p] - want) < 1e-9, (p, l, q, var[p], want)
        # P-slicing: variance(x, P) is the length-P prefix of variance(x, Pmax).
        for P in [1, 4, 7, 13, pfv.Pmax]:
            assert np.allclose(pfv.variance(np.array([1.0]), P), var[:P]), P

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


#######################################   class PfVariance   #######################################


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
      P:         number of peak-finding profiles (each term's last axis has length P).
      terms:     dict mapping dbits (int bitmask) -> ndarray of shape (2^popcount(dbits), P).
    """

    def __init__(self, rank, P):
        self.rank = int(rank)
        self.P = int(P)
        self.terms = {}      # dbits (int) -> (2^popcount(dbits), P) float64

    @staticmethod
    def from_tile(tile, P, convolver):
        """Build a PfVariance from a single singleton SparseTile.

        Shorthand for PfVariance(tile.k, P) followed by add_tile(tile, convolver): the result
        has rank == tile.k and one term (keyed by tile.dbits). P (profile count) and the
        convolver must both be supplied -- neither can be inferred from the tile.
        """
        pfvar = PfVariance(tile.k, P)
        pfvar.add_tile(tile, convolver)
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
        out = np.zeros((1 << m, self.P), dtype=np.float64)
        # Representative delay for each output row (its 'dbits' bits encode the row index).
        d = SparseTile._representative_delay(np.arange(1 << m), dbits)
        for term_dbits, term_arr in self.terms.items():
            out += term_arr[SparseTile._selected_bits_index(d, term_dbits)]
        return out

    def add_tile(self, tile, convolver):
        """Convolve a singleton SparseTile (tile.k == self.rank) into a new term.

        tile.data has shape (1, 2^popcount(tile.dbits), nt); 'convolver' maps its length-nt time
        axis to self.P profiles, giving a term (tile.dbits, (2^popcount, P)).  tile.tshifts and
        tile.t0 are irrelevant (the variance is time-shift invariant) and are ignored.
        """
        assert tile.nf == 1, "PfVariance.add_tile: tile must be a singleton (nf == 1)"
        assert tile.k == self.rank, (tile.k, self.rank)
        tile_var = convolver.variance(tile.data, self.P)[0]   # drop nf==1 axis -> (2^popcount, P)
        self._accumulate(tile.dbits, tile_var)

    def add(self, pfvar, upper_half=False):
        """Accumulate another PfVariance into self.

        Requires self.P <= pfvar.P; if pfvar has more profiles its p-axis is truncated to the first
        self.P (the kernel bank is nested in P, so the truncation is the lower-P variance exactly).

        If upper_half (which requires pfvar.rank == self.rank + 1), accumulate the logical UPPER
        HALF of pfvar's delay axis: fix pfvar's top delay bit (bit self.rank) to 1 and drop it. A
        term whose dbits lacks that bit is independent of it and added as-is; a term that depends on
        it keeps only the upper half of its (highest-bit-first) array. This equals adding from a
        tile specialized via specialize_dbits(1, 1, low=False), because variance is computed
        independently per delay row.
        """
        assert self.P <= pfvar.P, (self.P, pfvar.P)
        if not upper_half:
            assert pfvar.rank == self.rank, (pfvar.rank, self.rank)
            for dbits, arr in pfvar.terms.items():
                self._accumulate(dbits, arr[:, :self.P])
            return
        assert pfvar.rank == self.rank + 1, (pfvar.rank, self.rank)
        topbit = 1 << self.rank                          # pfvar's extra (top) delay bit
        for dbits, arr in pfvar.terms.items():
            arr = arr[:, :self.P]
            if dbits & topbit:                           # depends on the top bit (its array MSB):
                self._accumulate(dbits & ~topbit, arr[arr.shape[0] // 2:])   # drop it, keep upper half
            else:
                self._accumulate(dbits, arr)             # independent of the top bit

    def scaled(self, c):
        """Return a new PfVariance (same rank/P) with every term multiplied by c."""
        out = PfVariance(self.rank, self.P)
        for dbits, arr in self.terms.items():
            out._accumulate(dbits, c * arr)
        return out

    def add_spectator_low_bits(self, nbits):
        """Return a new PfVariance of rank (self.rank + nbits) that is independent of the new
        low 'nbits' delay bits (spectators): each term keeps its array, but its dbits shift up
        by nbits (old bit j -> bit j+nbits). Used to raise a coarser variance to a finer rank
        so it can be compared against a finer one."""
        out = PfVariance(self.rank + nbits, self.P)
        for dbits, arr in self.terms.items():
            out._accumulate(dbits << nbits, arr)
        return out

    def _accumulate(self, dbits, arr):
        # Add the term (dbits -> arr), summing into any existing same-dbits term.
        dbits = int(dbits)
        assert 0 <= dbits < (1 << self.rank), (dbits, self.rank)
        assert arr.shape == (1 << dbits.bit_count(), self.P), \
            (arr.shape, dbits, self.P)
        if dbits in self.terms:
            self.terms[dbits] = self.terms[dbits] + arr
        else:
            self.terms[dbits] = np.array(arr, dtype=np.float64)   # copy (avoid aliasing)

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

        full_cm = np.asarray(plan.config.make_channel_map(), dtype=np.float64)

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
        fs, P, ids = self.tree_fs[itree], int(self.tree_P[itree]), tree.ds_level
        rho_cm = int(self.tree_r[itree]) + (ids > 0)   # = tree_r + (ids>0) = config.tree_rank - delta
        cm = np.ascontiguousarray(full_cm[: (1 << rho_cm) + 1])    # truncate to first 2^rho_cm channels
        per_fm = []
        for ifreq in range(self.nfreq):
            if progress and (ifreq + 1) % 1000 == 0:
                print(".", end="", flush=True)
            if not (ifreq < cm[0] and ifreq + 1 > cm[-1]):        # ifreq below the truncated band
                per_fm.append(None)
                continue
            ssa = SparseTilePerM.make_dedispersion_output(cm, ifreq, fs, upper_half_only=(ids > 0))
            per_fm.append([None if t is None else PfVariance.from_tile(t, P, self.convolver)
                           for t in ssa.per_m])
        return per_fm

    def _make_per_m(self, itree):
        # per_m[m]: frequency-summed PfVariance for tree itree, multiplet m.
        rho = int(self.tree_r[itree]) - int(self.tree_R[itree])
        P, per_fm = int(self.tree_P[itree]), self.per_tfm[itree]
        per_m = []
        for m in range(self.tree_fs[itree].M):
            acc = PfVariance(rho, P)
            for row in per_fm:
                if row is not None and row[m] is not None:
                    acc.add(row[m])
            per_m.append(acc)
        return per_m


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
        self._tree_klevel = self.tree_r + (self._tree_ids > 0) - self.tree_L
        
        max_k = int(max(self._tree_klevel))
        trees_at = {}                                     # level k -> [itree, ...]
        for itree, k in enumerate(self._tree_klevel):
            trees_at.setdefault(int(k), []).append(itree)

        full_cm = np.asarray(plan.config.make_channel_map(), dtype=np.float64)

        # per_tff[itree][ifreq]: length-2^R list (PfVariance rank r-L, or None).
        # per_tf[itree][f]: length-2^R list (PfVariance rank r-L)
        self.per_tff = [[None] * self.nfreq for _ in range(self.ntrees)]
        self.per_tf = [[PfVariance(int(r) - int(L), int(P)) for _ in range(1 << int(R))]
                       for r, R, L, P in zip(self.tree_r, self.tree_R, self.tree_L, self.tree_P)]

        for ifreq in range(self.nfreq):
            if progress and (ifreq + 1) % 1000 == 0:
                print(".", end="", flush=True)
                
            sarr = SparseTileTriple.make_tree_gridding_output(full_cm, ifreq)   # rank config.tree_rank, level 0
            
            for k in range(0, max_k + 1):
                trees_k = trees_at.get(k, ())
                
                if trees_k:
                    pv_fp = self._sub_block_variances(sarr, trees_k)   # shared across trees_k
                    for itree in trees_k:
                        pv = self._coarsify_row(pv_fp, itree)          # list of 2^R (PfVariance or None)
                        self.per_tff[itree][ifreq] = pv
                        for f, x in enumerate(pv):
                            if x is not None:
                                self.per_tf[itree][f].add(x)
                
                if k < max_k:
                    sarr = sarr.iterate()

    
    def _sub_block_variances(self, sarr, trees_k):
        # Convert each RAW sub-block singleton at the current level to a PfVariance ONCE, shared
        # across all trees in trees_k (all have klevel == this level). Built at the max P over
        # trees_k and without any upper-half specialization; each tree's _coarsify_row truncates the
        # p-axis to its own P and (ids>0) takes the upper DM half via PfVariance.add(). pv_fp[fp] is
        # None for sub-blocks outside the gridding footprint.
        P_shared = int(max(self.tree_P[itree] for itree in trees_k))
        nfp = 1 << int(max(self.tree_L[itree] for itree in trees_k))
        pv_fp = [None] * nfp
        for fp in range(nfp):
            tile = sarr.get_singleton(fp, allow_none=True)
            if tile is not None:
                pv_fp[fp] = PfVariance.from_tile(tile, P_shared, self.convolver)
        return pv_fp

    
    def _coarsify_row(self, pv_fp, itree):
        # Coarsify the shared per-sub-block PfVariances into tree itree's 2^R coarse-freqs:
        # f = f' >> (L-R), mean over its 2^(L-R) sub-blocks (scaled by 2^-(L-R)). add() truncates
        # the p-axis to this tree's P and (ids>0) keeps the upper DM half (rank rho_cm-L -> r-L).
        R, L, ids = int(self.tree_R[itree]), int(self.tree_L[itree]), self._tree_ids[itree]
        P = int(self.tree_P[itree])
        rho = int(self.tree_r[itree]) - L                       # PfVariance rank = r - L  (== klevel - ids)
        norm = 2.0 ** (-(L - R))                            # DD normalization: sub-block-variance mean
        row = [None] * (1 << R)
        for fp in range(1 << L):                            # sub-block coarse-freq f'
            pv = pv_fp[fp]
            if pv is None:
                continue
            f = fp >> (L - R)                               # coarsify the f-index by 2^(L-R)
            if row[f] is None:
                row[f] = PfVariance(rho, P)
            row[f].add(pv, upper_half=(ids > 0))            # sum sub-block variances (upper half if ids)
        return [None if pv is None else pv.scaled(norm) for pv in row]   # ... then mean (2^-(L-R))

    
    def get_per_fm(self, itree, ifreq, m):
        """Approximate single-channel variance for tree itree, (ifreq, multiplet m): the mean of
        per_tff over the multiplet's coarse-freq range. Returns None if no coarse-freq overlaps."""
        rho = int(self.tree_r[itree]) - int(self.tree_L[itree])
        fs = self.tree_fs[itree]
        ilo, ihi = fs.m_to_ilo(m), fs.m_to_ihi(m)
        acc = None
        for f in range(ilo, ihi):
            pv = self.per_tff[itree][ifreq][f]
            if pv is not None:
                if acc is None:
                    acc = PfVariance(rho, int(self.tree_P[itree]))
                acc.add(pv)
        return None if acc is None else acc.scaled(1.0 / (ihi - ilo))

    
    def get_per_m(self, itree, m):
        """Approximate frequency-summed variance for tree itree, multiplet m: the mean of per_tf
        over the multiplet's coarse-freq range. Always returns a PfVariance."""
        rho = int(self.tree_r[itree]) - int(self.tree_L[itree])
        fs = self.tree_fs[itree]
        ilo, ihi = fs.m_to_ilo(m), fs.m_to_ihi(m)
        acc = PfVariance(rho, int(self.tree_P[itree]))
        for f in range(ilo, ihi):
            acc.add(self.per_tf[itree][f])
        return acc.scaled(1.0 / (ihi - ilo))
