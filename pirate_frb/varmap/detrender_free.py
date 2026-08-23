"""Analytic variance maps of a DedispersionConfig's trees, with no detrender.

compute_detrender_free_base_map(config) returns the variance map A of the tree with
(primary_tree_index, early_trigger_level) == (0, 0), computed analytically -- no dedisperser
run, no DedispersionPlan, no GPU -- and returned in FACTORED form, A = Q W^T. The map is
numerically low-rank because variances vary smoothly with input channel; see
notes/variance_map.tex, subsection "Per-group SVD and stacking".

Only the base tree is needed: with no detrender, every other tree's map is a ROW RESTRICTION
of the base tree's (notes/variance_map.tex, appendix "Variance maps of a config's trees are
row-restrictions of one another"). compute_detrender_free_multi_map() cashes that in,
slicing one base map into a VarianceMultiMap that covers every primary tree.

The algorithm, in outline:

  1. Per input channel, work out which frequency subbands see it, and over what tree-freq
     range. Group those (channel, subband-set) entries by a key 'sdbits' which packs the
     delay-bit mask and the subband set, so that ONE tile computation and ONE convolution
     serve every subband seeing the same range. Measured, that is 2.9x to 5.7x fewer
     convolutions than the per-multiplet route of slow_avar.PfAvarExact.
  2. Build one dense (capacity, D*P) matrix per group and SVD it, dropping small singular
     values.
  3. Lift each group's factors into a global (nbeta, Ktot) / (nfreq, Ktot) pair.

THE COST IS ALL IN STEP 3. Steps 1 and 2 run at full CHORD scale (chord_sb2_et.yml, nalpha =
5.96e6) in 14 seconds and hold 17 MB of rows -- a scale at which PfAvarExact cannot be built
at all. The lift then materializes a dense (nbeta, Ktot) Q, which for a FINE map (L = None)
is 0.36 GiB on toy.yml but 63.8 GiB at CHORD.

THERE ARE TWO WAYS OVER THAT WALL, and which one applies depends on what the caller needs.

THE FIRST IS 'L', and it is nearly free here. With no detrender the coarse-grained map is not
a max at all but a SLICE -- keep the bottom of each dyadic DM block -- so the coarse map is
built directly rather than by building the fine one and reducing it. At chord_sb2_et.yml the
lifted Q goes 63.8 GiB (fine) -> 17.5 (L = R = 4) -> 4.3 (L = 6) -> 0.96 (L = 8). Almost all
of that is the row count: Ktot only moves 1435 -> 1399 at L = 6, so what L buys is 14.6x
SHORTER columns of Q, not more compressible groups. Steps 1 and 2 are unchanged by L by
design -- it is the lift that goes away.

THE SECOND IS TO SKIP STEP 3 ENTIRELY, which a caller who needs only A v can do: one variance
vector per tree, for given input-channel variances, is what production actually consumes, and
it comes from putting the weight v_F on step 1's accumulator and lifting THAT -- no Q, no
dense group matrices, no SVD. That is compute_detrender_free_varfine(), and at
chord_sb2_et.yml it delivers all ten trees in 16 seconds inside 0.49 GiB, against 63.8 GiB of
Q for the base tree alone. The two functions share one code path (class SdPlan) and share it
EXACTLY: compute_detrender_free_varfine(config, ones)[itree0].reshape(-1) is BITWISE equal to
compute_detrender_free_base_map(config).y_true.

The remaining scaling fix for the MAP is a second, global SVD round that runs BEFORE the
lift; the per-group factors that round needs are what SdPlan leaves behind.

TWO THINGS HERE ARE NON-OBVIOUS, and getting either wrong gives a wrong ANSWER rather than a
crash. Both are derived at the code that implements them.

  - Every row is stored in a common "level-r" normalization, and the lift undoes it per
    subband with a factor 2^(R-l). See SdPlan._emit() and the lift. Measured: omitting it
    moves apply() by 75% on toy.yml.
  - Half-aligned subbands whose footprint straddles the subband midpoint need their own
    branch. See SdPlan._tile_pass(). Measured: omitting it moves apply() by 1.6e-3 on
    toy.yml -- rare (1 row of 645) but not negligible.

A THIRD, for the L path: y_true is defined at FINE granularity whatever the map's
coarse-graining rank (VarianceMap's class docstring is explicit about this), so it cannot be
read off the coarse factors. It is accumulated separately, in the full delay-bit basis, in
'sd_vectors' -- see class SdPlan.
"""

import time

import numpy as np

from .VarianceMap import VarianceMap, make_tree
from .VarianceMultiMap import VarianceMultiMap, expand_fine_vectors
from ..slow_avar.SparseTile import SparseTile, SparseTileTriple
from ..slow_avar.PfVariance import PfVarianceConvolver
from ..utils import atomic_print


# The sdbits key is (dbits << _SBITS_WIDTH) | sbits. The split is chosen so the key would
# still fit a uint64 in a future C++ port: N <= 42 subbands (notes/dedispersion.tex, section
# "Subbanded dedispersion", at R <= constants::max_peak_finding_rank == 4) and r <= 16
# (constants::max_tree_rank), so 42 + 16 < 64. SdPlan asserts both bounds up front, because
# raising either would silently corrupt keys rather than fail.
_SBITS_WIDTH = 42
_SBITS_MASK = (1 << _SBITS_WIDTH) - 1


def _iter_bits(bits):
    """The set bit positions of 'bits', ascending.

    DELIBERATELY KNOWS NOTHING ABOUT THE sdbits PACKING. A caller iterating a group's subbands
    passes 'sdbits & _SBITS_MASK' explicitly, so that the mask reads as the load-bearing step
    it is: without it the delay bits would arrive as subband indices. Other callers pass a
    plain subband mask, and a name promising to unpack an sdbits would be wrong for them.
    """
    out = []
    while bits:
        b = bits & (-bits)
        out.append(b.bit_length() - 1)
        bits ^= b
    return out


def _coarsen_sdbits(sdbits, L):
    """The group key after coarse-graining at rank L: clear the dbits below L, keep sbits.

    L is None for a fine map, in which case this is the identity -- so the L = None path keys,
    sizes and pools exactly as it did before L existed.

    Both boundaries behave: L = 0 gives an empty mask (right, since coarse-graining at R = 0
    removes no delay bits), and L = r clears every delay bit, leaving D == 1.
    """
    if L is None:
        return sdbits
    mask = (1 << (L + _SBITS_WIDTH)) - (1 << _SBITS_WIDTH)    # dbits 0..L-1, shifted into place
    return sdbits & ~mask


####################################   class SdMatrix   ####################################


class SdMatrix:
    """One term of the map: every (input channel, subband set) entry sharing one 'sdbits'.

    The rows are input channels and the columns are (delay, profile) pairs, so the stored
    matrix is the TRANSPOSE of the corresponding block of A. See factorize() for the
    factorization convention, which is easy to get backwards.

    Members
    -------
      sdbits        (dbits << 42) | sbits, the group key.
                    sbits = bit n set iff subband n is served by this term.
                    dbits = the delay-bit mask, IN THE LEVEL-r LABELLING (see the module
                    docstring). A python int, not a numpy integer: numpy's fixed-width types
                    would only add overflow warnings at these widths.
      D             1 << popcount(dbits), the length of the delay axis. NOT the same quantity
                    as 'ndm' = 2^(r-R), the alpha convention's coarse-DM count.
      P             number of peak-finding profiles.
      capacity      final row count, known before any row is filled.
      F             rows filled so far.
      freq_indices  (capacity,) int64: the input channel of each row. DISTINCT within a
                    matrix, which is what makes the W lift a plain scatter.
      dense_matrix  (capacity, D*P) float64, the rows.
      epsilon       the relative singular-value threshold factorize() actually used.
      is_factored, factor_rank, Q_factor (D*P, K), W_factor (F, K)

    Note there is no row-sum member here. y_true is FINE whatever the coarse-graining rank,
    so its terms are accumulated in the FULL delay-bit basis, in SdPlan's 'sd_vectors' --
    which this matrix's columns are a slice of once Lmat is set.
    """

    def __init__(self, sdbits, capacity, D, P):
        self.sdbits = int(sdbits)
        self.capacity = int(capacity)
        self.D, self.P = int(D), int(P)
        assert self.capacity >= 1 and self.D >= 1 and self.P >= 1

        self.F = 0
        self.freq_indices = np.zeros(self.capacity, dtype=np.int64)
        self.dense_matrix = np.zeros((self.capacity, self.D * self.P), dtype=np.float64)

        self.epsilon = None
        self.is_factored = False
        self.factor_rank = None
        self.Q_factor = None
        self.W_factor = None

    def __repr__(self):
        return (f'SdMatrix(dbits={self.sdbits >> _SBITS_WIDTH:#x},'
                f' sbits={self.sdbits & _SBITS_MASK:#x}, shape=({self.capacity},'
                f' {self.D * self.P}), K={self.factor_rank})')

    @staticmethod
    def default_epsilon(nrow, ncol):
        """Relative singular-value threshold for an (nrow, ncol) matrix.

        Truncating at eps*S_max perturbs the reconstruction at the eps level, so eps is an
        accuracy knob -- but it is only meaningful above the float64 noise floor on singular
        values, which numpy's matrix_rank estimates as max(shape) * eps_f64 * S_max. At the
        variance maps' sizes that floor can exceed the 1e-11 which is safe for smaller
        matrices, so we take whichever is larger. See notes/variance_map.tex, subsection
        "Truncation threshold".
        """
        return max(1.0e-11, 16.0 * max(int(nrow), int(ncol)) * float(np.finfo(np.float64).eps))

    def factorize(self, epsilon=None):
        """Truncated SVD of dense_matrix, in the TRANSPOSED convention

            dense_matrix.T ~= Q_factor @ W_factor.T

        i.e. Q_factor is indexed by (delay*P + profile) and W_factor by input channel. That
        is the orientation the lift wants, and a transpose slip here is SILENT whenever
        D*P == F, since the shapes still match -- so it is checked by a test.

        The square root of each singular value is split between the two factors. Neither
        factor is semiorthogonal afterwards, which costs nothing: the globally stacked
        factors are not orthogonal anyway (different groups overlap in their rows of A).
        """

        assert self.F == self.capacity, (self.F, self.capacity)
        assert not self.is_factored

        eps = (self.default_epsilon(*self.dense_matrix.shape) if (epsilon is None)
               else float(epsilon))
        assert eps > 0.0, eps
        self.epsilon = eps          # recorded: with epsilon=None it varies per group

        u, s, vh = np.linalg.svd(self.dense_matrix, full_matrices=False)
        K = int(np.sum(s > eps * s[0])) if ((s.size > 0) and (s[0] > 0.0)) else 0

        rs = np.sqrt(s[:K])
        self.Q_factor = vh[:K].T * rs                # (D*P, K)
        self.W_factor = u[:, :K] * rs                # (F,   K)
        self.factor_rank = K
        self.is_factored = True


####################################   the algorithm   ####################################


class SdPlan:
    """Everything except the lift, for the base tree of 'config'.

    The constructor runs the four passes -- planning, sizing, tiles, factorization -- and
    leaves two parallel accumulators. THE CONSTRUCTOR DOES NO LIFTING; lift_sd_vectors() is a
    separate, opt-in step. That split is the reason this is a class rather than a function
    call inside compute_detrender_free_base_map(): the per-group factors, not the lifted Q,
    are what a second (global) SVD round and a coarse-graining pass would consume, and those
    callers would otherwise be handed an O(nalpha) vector to discard.

    THE TWO ACCUMULATORS DO NOT SHARE A KEY SPACE, and that is the one thing to get right
    here:

      sd_matrices   COARSE sdbits -> SdMatrix, each (capacity, D_coarse * P). None when
                    init_sd_matrices is False.
      sd_vectors    FULL   sdbits -> (D_full * P,) float64, the summed rows.

    'Lmat' is spelled differently from the callers' 'L' to keep that asymmetry visible: the
    rank governs sd_matrices AND NOTHING ELSE -- it selects their coarse key space and slices
    the rows written into them -- while sd_vectors is accumulated in the full delay-bit basis
    whatever Lmat is, because y_true is FINE whatever the map's coarse-graining rank. At
    Lmat = None the two key spaces COINCIDE, so indexing one with the other's key passes every
    fine test. It is also a QUIET bug with Lmat set: a coarse key is a syntactically valid
    full key whenever the full dbits happen to have no bits below Lmat, so the lookup can
    succeed and return another group's data. Hence every use site names which key it is using,
    and the sizing pass asserts len(sd_vectors) >= len(sd_matrices) -- coarsening can only
    merge keys, never split them.

    Parameters
    ----------
    Lmat : int or None
        Coarse-graining rank of sd_matrices, R <= Lmat <= r, or None. This is the callers'
        'L'; see above for why it is named differently here.
    epsilon : float or None
        Relative singular-value threshold, per group. None uses SdMatrix.default_epsilon().
    freq_variances : array or None
        Length-nfreq input-channel variances, or None for all-ones. THE WEIGHT REACHES
        sd_vectors ONLY, NEVER THE SdMatrix ROWS -- see _emit(). With weights set,
        lift_sd_vectors() returns ``A v`` rather than y_true, which is
        compute_detrender_free_varfine()'s whole algorithm.
    init_sd_matrices : bool
        False skips the capacity counting, the SdMatrix allocation, the row writes and the
        SVD, leaving sd_matrices None. sd_vectors is unaffected. Lmat and epsilon are then
        meaningless and are rejected rather than ignored.
    debug : bool
        Turn on the O(subbands) planning-pass cross-checks: that every subband of an entry
        predicts the same dbits, and that no input channel contributes two entries with the
        same key. Both are statements the shared-row pooling depends on, and both are too
        expensive to leave on at production scale.

    Members
    -------
      config, itree0, tree0:
          which tree this is a plan for. NOTE itree0 IS NOT ALWAYS ZERO: early_trigger_level
          DESCENDS within a primary-tree family, so the e = 0 tree is the LAST of its family.
          It is 0 for every shipped config -- which is exactly what would make an assertion
          to that effect a trap -- and 1 for _make_test_config(7, [2,2,1],
          num_early_triggers=1).

      r, R, N, M, P, nfreq, ndm, nalpha:
          the base tree's geometry, with ndm = 2^(r-R) and nalpha = ndm*M*P.

      lev, c, I_lo, I_hi, I_mid, case1, mbase:
          the length-N per-subband tables; see _subband_geometry().

      Lmat, epsilon, freq_variances, debug:
          the constructor's arguments, kept. Read self.Lmat rather than stats['Lmat']: it is
          the int()-ed, range-checked value, and stats mirrors it only for the record.

      cmap, convolver, footprint:
          the (nfreq+1,) input-channel edges in tree-freq units, one shared
          PfVarianceConvolver for the whole run, and each channel's UNCLIPPED [f0, f1) as
          (nfreq, 2) int64. See _tile_pass() for why the footprint is worth caching.

      unstraddled_plan, straddled_plan:
          the two plans; see _plan_pass().

      sd_matrices, sd_vectors:
          the two accumulators above.

      stats:
          dict, which the caller folds into VarianceMap.history.
    """

    def __init__(self, config, *, Lmat=None, epsilon=None, freq_variances=None,
                 init_sd_matrices=True, progress=False, debug=False):

        from ..pirate_pybind11 import constants

        self.config = config
        self.itree0 = int(config.dedispersion_tree_index(0, 0))
        self.tree0 = make_tree(config, self.itree0)
        fs = self.tree0.frequency_subbands

        self.r, self.R = int(self.tree0.total_rank()), int(fs.pf_rank)
        self.N, self.M, self.P = int(fs.N), int(fs.M), int(self.tree0.nprofiles)
        self.nfreq = int(config.get_total_nfreq())
        self.ndm = 1 << (self.r - self.R)
        self.nalpha = self.ndm * self.M * self.P

        # The sdbits packing has exactly these two headrooms; see _SBITS_WIDTH.
        assert self.N <= _SBITS_WIDTH, (self.N, _SBITS_WIDTH)
        assert self.r <= constants.max_tree_rank, (self.r, constants.max_tree_rank)

        if Lmat is not None:
            Lmat = int(Lmat)
            # Same bounds and the same wording as VarianceMap.coarse_grain(), so the two read
            # the same. Below R is meaningless because beta's definition uses dc = d >>
            # (Lmat-R); above r is impossible because there are only 2^(r-R) coarse DMs to
            # merge. The message names the caller's spelling as well as this class's: nobody
            # outside SdPlan types 'Lmat'.
            if not (self.R <= Lmat <= self.r):
                raise RuntimeError(f"SdPlan: Lmat (the caller's L) = {Lmat} is out of range"
                                   f" [R, r] = [{self.R}, {self.r}] for this config's base"
                                   " tree")

        if (not init_sd_matrices) and ((Lmat is not None) or (epsilon is not None)):
            raise RuntimeError('SdPlan: init_sd_matrices=False leaves nothing for Lmat or'
                               ' epsilon to act on -- Lmat selects the coarse key space of'
                               ' sd_matrices and slices the rows written into them, epsilon'
                               ' is their SVD threshold, and sd_vectors is FULL and'
                               ' untruncated whatever either of them is. Accepting them'
                               ' silently would hand back a fine, untruncated result to a'
                               ' caller who asked for a coarse or truncated one.')

        self.Lmat, self.epsilon = Lmat, epsilon
        self.debug = bool(debug)

        self.freq_variances = None
        if freq_variances is not None:
            self.freq_variances = np.asarray(freq_variances, dtype=np.float64)
            if self.freq_variances.shape != (self.nfreq,):
                raise RuntimeError(f'SdPlan: expected freq_variances of shape'
                                   f' ({self.nfreq},), got'
                                   f' {self.freq_variances.shape}')

        # The alpha convention assumes 2^R coarse DM channels per multiplet, which is what an
        # unset (auto) dm_downsampling gives. validate() already requires it, so this is a
        # tripwire rather than a check.
        dmds = int(config.primary_trees[0].dm_downsampling)
        if dmds != 0:
            raise RuntimeError(f'SdPlan: primary tree 0 has dm_downsampling={dmds}, but the'
                               " variance map's index convention needs the auto value 0")

        self._subband_geometry()

        self.cmap = np.asarray(config.make_channel_map(), dtype=np.float64)
        self.convolver = PfVarianceConvolver()     # ONE shared instance for the whole run

        self.unstraddled_plan, self.straddled_plan = [], []
        self.footprint = np.zeros((self.nfreq, 2), dtype=np.int64)
        self.sd_matrices, self.sd_vectors = None, {}
        self.stats = {}
        self._n_sliced = 0                         # folded into stats by _tile_pass()

        self._plan_pass()
        self._size_pass(init_sd_matrices, progress)
        self._tile_pass()
        if init_sd_matrices:
            self._factorize(progress)


    def __repr__(self):
        what = ('sd_matrices=None' if (self.sd_matrices is None)
                else f'{len(self.sd_matrices)} matrices')
        return (f'SdPlan(itree0={self.itree0}, r={self.r}, R={self.R}, N={self.N},'
                f' M={self.M}, P={self.P}, nfreq={self.nfreq},'
                f' n_entries={self.stats["n_entries"]}, {what})')


    def _subband_geometry(self):
        """Set the length-N per-subband tables of the algorithm, as array members.

        All in TOPLEVEL TREE-FREQ units: a coarse channel is 2^(r-R) tree-freqs wide, so
        subband n occupies [I_lo[n], I_hi[n]), of width 2^c[n] with c[n] = r-R+l[n] its own
        tree depth.
        """

        fs = self.tree0.frequency_subbands
        r, R = self.r, self.R

        self.lev = np.asarray(fs.n_to_level, dtype=np.int64)
        self.c = (r - R) + self.lev
        self.mbase = np.asarray(fs.n_to_mbase, dtype=np.int64)

        flo = np.asarray(fs.n_to_flo, dtype=np.int64)
        fhi = np.asarray(fs.n_to_fhi, dtype=np.int64)
        self.I_lo = flo << (r - R)
        self.I_hi = fhi << (r - R)

        # Case 1 (aligned): I_n is a node of the toplevel tree at level c, so ordinary aligned
        # iteration reproduces the subband's merges. Case 2 (half-aligned, l > 0 and odd
        # index): I_n starts at an odd multiple of 2^(c-1) and is NOT a node of the tree. See
        # notes/dedispersion.tex, section "Subbanded dedispersion".
        self.case1 = (flo & ((1 << self.lev) - 1)) == 0

        # Exact: I_hi - I_lo = 2^c, and the only branch that reads I_mid has l >= 1 hence
        # c >= 1, so I_mid = I_lo + 2^(c-1). Note the midpoint is generic -- a case-1
        # subband's top merge joins its two halves at the same point -- but for case 1 the
        # halves are the ALIGNED pair, which SparseTileTriple.iterate() already merges
        # correctly, so there is nothing to detect and I_mid is never consulted there.
        self.I_mid = (self.I_lo + self.I_hi) // 2


    # ---------------- the four passes ----------------

    def intersect(self, j0, j1, n):
        """This channel's footprint [j0, j1) intersected with subband n. Empty iff lo >= hi.

        Shared by the planning pass and the tile pass so the two cannot drift.
        """
        return max(j0, int(self.I_lo[n])), min(j1, int(self.I_hi[n]))


    def predict_dbits_r(self, lo, hi, n):
        """The level-r dbits of the range [lo, hi), for any subband n of its entry.

        _predict_dbits() is called in SUBBAND-LOCAL f-coordinates (f - I_lo[n]), run for the
        subband's own depth c[n], and then shifted into the level-r labelling.

        Local coordinates are the right ones on BOTH branches, because I_lo[n] is a multiple
        of 2^(c-1) in every case (a multiple of 2^c in case 1). Subtracting it preserves the
        block structure at every level up to c-1, and the subband's own top merge -- the
        aligned merge of local blocks 0 and 1 -- is the aligned merge _predict_dbits()
        assumes. In other words THE SUBBAND'S DEDISPERSION TREE IS THE ORDINARY ALIGNED TREE
        IN LOCAL COORDINATES, for case 2 as much as for case 1. That is also why the straddle
        branch can use SparseTile.iterate_singletons(), which reads only the relative order of
        its two arguments and not their absolute f-index.
        """
        cc = int(self.c[n])
        return SparseTile._predict_dbits(cc, lo - int(self.I_lo[n]), hi - lo) << (self.r - cc)


    def _plan_pass(self):
        """Per input channel, which subbands see it, over what range, and with what dbits.

        Two plans rather than one plan with a nullable discriminant. 'ifreq' is a member of
        both tuples, not just the loop variable: the tile pass walks the flat plans and needs
        it to rebuild the channel's gridding triple and to write freq_indices.

          unstraddled_plan: (ifreq, lo, hi, sdbits)       the common path
          straddled_plan:   (ifreq, straddle_n, sdbits)   case-2 midpoint straddles

        The straddled triple carries no (lo, hi) because (ifreq, straddle_n) already
        determines them, via the same intersect() the tile pass calls.
        """

        r, R, N = self.r, self.R, self.N
        I_mid, case1, lev = self.I_mid, self.case1, self.lev
        seen_subbands = 0

        for ifreq in range(self.nfreq):
            tri = SparseTileTriple.make_tree_gridding_output(self.cmap, ifreq)
            j0, j1 = int(tri.f0), int(tri.f0 + tri.nf)
            self.footprint[ifreq] = (j0, j1)

            # A LIST OF [lo, hi, sbits], NOT A DICT KEYED BY (lo, hi). It holds at most one
            # entry per unstraddled subband seeing this channel, and measured it is almost
            # always length 1 and never longer than 3 (at chord_sb2_et.yml, N = 25), so a
            # linear scan beats hashing a composite key. Appending also makes the entry order
            # explicit rather than a property of the container: entries come out in
            # ascending-n order, which is the order they reach unstraddled_plan and therefore
            # the row order of every dense_matrix.
            local_unstraddled_plan = []
            u0, s0 = len(self.unstraddled_plan), len(self.straddled_plan)

            for n in range(N):
                lo, hi = self.intersect(j0, j1, n)
                if lo >= hi:
                    continue                       # this subband does not see this channel
                seen_subbands |= (1 << n)
                if (not case1[n]) and (lo < I_mid[n] < hi):
                    dbits = self.predict_dbits_r(lo, hi, n)
                    self.straddled_plan.append((ifreq, n, (dbits << _SBITS_WIDTH) | (1 << n)))
                else:
                    for e in local_unstraddled_plan:
                        if (e[0] == lo) and (e[1] == hi):
                            e[2] |= (1 << n)
                            break
                    else:
                        local_unstraddled_plan.append([lo, hi, 1 << n])

            for (lo, hi, sbits) in local_unstraddled_plan:
                n0 = (sbits & -sbits).bit_length() - 1    # any subband of the entry will do
                dbits = self.predict_dbits_r(lo, hi, n0)

                # The rule agrees with the simpler GLOBAL form on this branch: [lo,hi) lies
                # inside a single 2^(c-1)-aligned block, so neither f0's low bits nor the XOR
                # of the two endpoints is changed by the shift of origin. This is the cheapest
                # available check that the entry was classified as unstraddled correctly.
                assert dbits == SparseTile._predict_dbits(r, lo, hi - lo), (ifreq, lo, hi, n0)

                if self.debug:
                    # One shared row is legitimate only because every subband of the entry
                    # predicts the same dbits -- they may differ in l and in I_lo, and they
                    # all agree because they all agree with the global form above.
                    for n in _iter_bits(sbits):
                        assert self.predict_dbits_r(lo, hi, n) == dbits, (ifreq, lo, hi, n, n0)

                self.unstraddled_plan.append((ifreq, lo, hi, (dbits << _SBITS_WIDTH) | sbits))

            if self.debug:
                # NO SdMatrix EVER GETS TWO ROWS FROM ONE INPUT CHANNEL, which is what makes
                # freq_indices distinct and the W lift a plain scatter (see the assignment in
                # _emit()). The reason is a property of the plan, checkable here: within one
                # ifreq, distinct entries carry DISJOINT sbits -- each seen subband is
                # assigned to exactly one entry of exactly one plan -- so their full sdbits
                # are distinct, and since _coarsen_sdbits() clears only dbits and preserves
                # sbits, their COARSE keys are distinct too, for EVERY Lmat.
                #
                # Checking it here rather than against a filling SdMatrix makes the statement
                # Lmat-free, cheap (a few entries per channel, not an O(F) scan), and true in
                # the init_sd_matrices=False mode as well -- where a duplicate would silently
                # DOUBLE-COUNT into sd_vectors rather than merely overwrite a row.
                keys = [s for (_, _, _, s) in self.unstraddled_plan[u0:]]
                keys += [s for (_, _, s) in self.straddled_plan[s0:]]
                assert len(set(keys)) == len(keys), (ifreq, keys)

        # A subband seeing no channel at all would give identically-zero rows of A, which
        # breaks y_true and hence get_distance(). This is the analogue of PfAvarExact's
        # m_cnt >= 1 assert.
        assert seen_subbands == ((1 << N) - 1), \
            f'subband(s) {_iter_bits(((1 << N) - 1) & ~seen_subbands)} see no input channel'

        # Every subband of an entry must have R - l[n] zero low bits in the entry's dbits,
        # since the lift's virtual level-r delay index (d << R) | (e << (R-l)) has none there.
        # The argument that this holds is EXACTLY the straddle discriminant, which is why the
        # assertion cannot fire for any other reason:
        #
        #   Write [lo,hi) for the intersection and use the closed form of _predict_dbits().
        #   The leading run occupies bit positions [r-j1, r-1], so it dips below R-l iff
        #   j1 > c; the isolated bit sits at r-1-h, so it dips iff h >= c. Now j1 <=
        #   bit_length(hi-lo-1) <= c always, since [lo,hi) is contained in I_n and I_n has
        #   width 2^c -- the run never dips. And h is the highest bit at which lo and hi-1
        #   differ. In case 1, I_n is 2^c-aligned, so lo and hi-1 agree above bit c-1 and
        #   h <= c-1. In case 2, I_n spans the two 2^(c-1)-aligned blocks either side of its
        #   midpoint: if [lo,hi) lies in one of them, again h <= c-2; if it STRADDLES, then
        #   lo >> (c-1) and (hi-1) >> (c-1) differ, so h >= c and the assertion fails.
        #
        # So assertion failures and midpoint straddles are the same set. With the straddle
        # branch taken explicitly the assertion now holds everywhere -- trivially so on that
        # branch, where the << (r - c) supplies R - l[n] zero low bits. Keep it: it is the
        # statement that makes the lift's index formula well defined.
        for (_, _, _, sdbits) in self.unstraddled_plan:
            dbits = sdbits >> _SBITS_WIDTH
            for n in _iter_bits(sdbits & _SBITS_MASK):
                assert (dbits & ((1 << (R - int(lev[n]))) - 1)) == 0, (dbits, n, int(lev[n]))


    def _count(self, sdbits, sd_capacities):
        """One plan entry's contribution to the two accumulators' sizes.

        'sd_capacities' is None when init_sd_matrices is off; sd_vectors is preallocated
        either way, since it is FULL and untruncated in both modes.
        """

        if sd_capacities is not None:
            # Keying sd_matrices by the COARSE sdbits is where the extra pooling comes from:
            # entries whose full dbits differ only below Lmat now share a matrix. Measured,
            # that is worth 0.2% of Ktot at Lmat = R and 26% at Lmat = R+6 -- small at the
            # interesting end, but free, and it is the natural key.
            key = _coarsen_sdbits(sdbits, self.Lmat)
            sd_capacities[key] = sd_capacities.get(key, 0) + 1

        if sdbits not in self.sd_vectors:
            D_full = 1 << (sdbits >> _SBITS_WIDTH).bit_count()
            self.sd_vectors[sdbits] = np.zeros(D_full * self.P)


    def _size_pass(self, init_sd_matrices, progress):
        """Allocate both accumulators. ONE capacity dict, fed by both plans: nothing stops a
        straddled entry's sdbits from coinciding with an unstraddled one's (measured, it never
        does, but two dicts merged with update() would silently drop a group).
        """

        sd_capacities = {} if init_sd_matrices else None       # COARSE sdbits -> row count

        for (_, _, _, sdbits) in self.unstraddled_plan:
            self._count(sdbits, sd_capacities)
        for (_, n, sdbits) in self.straddled_plan:
            # A straddling subband always gets a row to itself, so the two fields of its
            # sdbits are redundant with each other -- and checkable.
            assert (sdbits & _SBITS_MASK) == (1 << n), (sdbits, n)
            self._count(sdbits, sd_capacities)

        if init_sd_matrices:
            self.sd_matrices = {}
            for (sdbits, capacity) in sd_capacities.items():
                dbits = sdbits >> _SBITS_WIDTH
                self.sd_matrices[sdbits] = SdMatrix(sdbits, capacity,
                                                    1 << dbits.bit_count(), self.P)

            # Coarsening can only merge keys, never split them, so this holds always -- and
            # fails loudly if the two dicts are ever indexed with each other's key. See the
            # class docstring.
            assert len(self.sd_vectors) >= len(self.sd_matrices), \
                (len(self.sd_vectors), len(self.sd_matrices))

        n_entries = len(self.unstraddled_plan) + len(self.straddled_plan)
        self.stats.update(n_entries=n_entries, n_straddled=len(self.straddled_plan),
                          n_groups=len(self.sd_vectors), Lmat=self.Lmat)

        if progress:
            atomic_print(f'  SdPlan: r={self.r} R={self.R} N={self.N} M={self.M} P={self.P}'
                         f' nfreq={self.nfreq}: {n_entries} entries'
                         f' ({len(self.straddled_plan)} straddled) -> {len(self.sd_vectors)}'
                         f' y_true groups, {n_entries/self.nfreq:.2f} rows per input channel')


    def _emit(self, ifreq, tile, klev, sdbits):
        """The level-r normalization, then one term of each accumulator.

        A subband's tile lives at its own level c = r-R+l, but rows from subbands of DIFFERENT
        levels share a matrix, so they must be stored in a COMMON labelling -- and level r is
        it. Lifting a tile from level c to level r is r-c = R-l single-leg dedispersion steps
        (the other half of each merge is absent, since no subband sees a footprint wider than
        itself), and a single-leg step leaves 'data' untouched, shifts dbits left by one, and
        multiplies 'scale' by 1/sqrt(2). Time shifts change, but the variance does not see
        them. So

            dbits(level r) = dbits(level c) << (R-l)
            var  (level r) = var  (level c) *  2^-(R-l)

        The lift undoes the second line per subband, with a factor 2^(R-l[n]). It CANNOT be
        folded into the stored row: two subbands sharing a row have different l.

        THE FOUR STEPS BELOW HAVE AN ORDER THAT MATTERS, and it is the reason this is one
        method with a branch rather than a row generator two consumers each accumulate from.
        sd_vectors must see the FULL, un-sliced, WEIGHTED row; the SdMatrix must see the
        possibly-sliced, UNWEIGHTED one.
        """

        # This is the assert that closes the loop between the planning pass's closed form and
        # the actual iteration, on both plans. It reads the FULL sdbits, as does every other
        # assertion here -- plan entries carry the full mask whatever Lmat is, and the coarse
        # key is derived only where it is needed. Everything downstream -- the sizing, the
        # shared rows, the lift -- is built on that prediction being right.
        assert (tile.dbits << (self.r - klev)) == (sdbits >> _SBITS_WIDTH), \
            (ifreq, klev, tile.dbits, sdbits >> _SBITS_WIDTH)

        # scale**2 because variance is quadratic (this is what PfVariance.add_tile() does);
        # the [0] drops the length-1 nf axis. Omitting scale**2 is silently wrong wherever an
        # edge tile deferred its 1/sqrt(2).
        P = self.P
        var = (tile.scale ** 2) * self.convolver.variance(tile.data, P)[0]    # (D_full, P)
        var *= 2.0 ** (-(self.r - klev))

        # y_true is FINE whatever Lmat is, so its terms accumulate in the FULL dbits basis,
        # under the FULL key, BEFORE the slice below.
        #
        # THE WEIGHT GOES HERE AND NOWHERE ELSE. sd_matrices factorizes A itself, whose W
        # factor is indexed by input channel, so folding v into its rows would corrupt the
        # factorization -- and it would be INVISIBLE under the default, where w is exactly
        # 1.0. Scaling 'var' in place before the slice would hit both, which is the mistake
        # this ordering is written to prevent. (w * var with w == 1.0 is bitwise the identity,
        # which is what makes compute_detrender_free_varfine(config, ones) bitwise equal to
        # y_true.)
        w = 1.0 if (self.freq_variances is None) else self.freq_variances[ifreq]
        self.sd_vectors[sdbits] += w * var.reshape(-1)

        if self.sd_matrices is None:
            return

        if self.Lmat is not None:
            # COARSE-GRAINING IS A SLICE, NOT A MAX, and this is where that is cashed in.
            #
            # Group beta = (dc, n, p) contains the fine rows whose subband DM index d_m runs
            # over an aligned dyadic block of size 2^(Lmat-R+l): d = dc*2^(Lmat-R) +
            # [0, 2^(Lmat-R)), e = [0, 2^l). With NO DETRENDER the variance is antitone in the
            # bits of d_m, so the max over an aligned dyadic block is attained at its BOTTOM --
            # see notes/variance_map.tex, appendix "Monotonicity of the variance map in the DM
            # bits (no detrender)", eq:dyadic_property, which says in as many words that a map
            # resolving only the top DM bits "can simply be evaluated at the bottom of each
            # dyadic DM block: no maximization over the block is needed, and the resulting
            # bound is tight".
            #
            # THE NO-DETRENDER HYPOTHESIS IS LOAD-BEARING. The same appendix shows the
            # property is false with a Detrender2d in front, so the day someone adds a
            # detrender path is the day this slice silently becomes wrong.
            #
            # In the level-r labelling the block bottom is "all bits below Lmat zero", since
            # d_full = d_m << (R-l) turns d_m's low (Lmat-R+l) bits into d_full's low Lmat.
            # And _remap_d() packs selected bits in order, so those bits occupy the LOWEST
            # n_rm packed positions, contiguously -- hence a reshape and a slice. Taking
            # [:, -1, :] instead of [:, 0, :] would give a min-envelope, which is why the test
            # set checks n_sliced > 0 and calls check_ref_covers_y_true().
            #
            # The max commutes with the group structure per column: at fixed F every alpha in
            # beta gets its value from ONE plan entry (they share the subband, and each
            # subband is assigned to exactly one entry per channel), so this may be done
            # row-by-row, before any summation over channels and before the SVD.
            n_rm = ((sdbits >> _SBITS_WIDTH) & ((1 << self.Lmat) - 1)).bit_count()
            self._n_sliced += int(n_rm > 0)
            var = var.reshape(-1, 1 << n_rm, P)[:, 0, :]           # (D_coarse, P)

        sdm = self.sd_matrices[_coarsen_sdbits(sdbits, self.Lmat)]     # COARSE key
        assert sdm.F < sdm.capacity, (sdm.F, sdm.capacity)
        sdm.dense_matrix[sdm.F, :] = var.reshape(-1)
        # DISTINCT within a matrix, which is what makes the W lift a plain scatter. That is a
        # property of the plan, and _plan_pass()'s debug branch is where it is checked.
        sdm.freq_indices[sdm.F] = ifreq
        sdm.F += 1


    def _tile_pass(self):
        """Build the tiles and fill the accumulators. Two straight-line loops, one per plan.

        Two calls to make_tree_gridding_output() are in play and they are NOT the same call:
        the planning pass needed the UNCLIPPED footprint once per channel (cached in
        'footprint'), while these need a triple CLIPPED to the entry's own [lo, hi), which
        cannot be hoisted. Its repeated searchsorted over cmap is negligible at these sizes.
        """

        for (ifreq, lo, hi, sdbits) in self.unstraddled_plan:
            tri = SparseTileTriple.make_tree_gridding_output(self.cmap, ifreq, flo=lo, fhi=hi)
            for _ in range(self.r):
                tri = tri.iterate()
            assert tri.nf == 1 and len(tri.tiles) == 1, (tri.nf, len(tri.tiles))
            self._emit(ifreq, tri.tiles[0], self.r, sdbits)

        for (ifreq, n, sdbits) in self.straddled_plan:
            j0, j1 = int(self.footprint[ifreq, 0]), int(self.footprint[ifreq, 1])
            lo, hi = self.intersect(j0, j1, n)
            cc = int(self.c[n])
            tri = SparseTileTriple.make_tree_gridding_output(self.cmap, ifreq, flo=lo, fhi=hi)
            for _ in range(cc - 1):
                tri = tri.iterate()

            # The subband's top merge combines the level-(cc-1) blocks either side of its
            # midpoint, which is NOT an aligned pair -- aligned pairs are (2F, 2F+1). Ordinary
            # iteration would merge the lower block with its (absent) left neighbour and the
            # upper with its (absent) right one, producing two tiles that never combine the
            # way the dedisperser combines them. Indexing the pair off the midpoint is the
            # same pair as the "2f+1, 2f+2" of notes/dedispersion.tex Case 2, without needing
            # that section's 'f' convention. Both halves are present by the definition of
            # "straddle", which the default allow_none=False makes an implicit assertion.
            ublk = int(self.I_mid[n]) >> (cc - 1)
            lower = tri.get_singleton(ublk - 1)
            upper = tri.get_singleton(ublk)
            self._emit(ifreq, SparseTile.iterate_singletons(lower, upper), cc, sdbits)

        if self.sd_matrices is None:
            return

        # Exact allocation: there is no growth and no reallocation, so 'capacity' is never an
        # upper bound that a row count falls short of.
        for sdm in self.sd_matrices.values():
            assert sdm.F == sdm.capacity, (sdm.sdbits, sdm.F, sdm.capacity)
        self.stats['n_sliced'] = self._n_sliced


    def _factorize(self, progress):
        """Per-group truncated SVD.

        Note that "rows are in channel order" is NOT true: the tile pass runs its two loops in
        sequence, so a matrix fed by both gets its straddled rows after all its unstraddled
        ones. Nothing depends on row order -- the SVD does not, the W lift scatters by
        freq_indices, and y_true sums over rows -- but it is an easy thing to assume.

        NO COLUMN PRECONDITIONER. slow_avar.TmpVmapExact divides by a per-channel scale before
        its SVD, because it stacks CHANNELS AS COLUMNS of a per-multiplet matrix, where a
        barely-overlapping edge channel contributes an anomalously small column that a
        relative threshold would delete. Here channels are ROWS of a much more finely divided
        set of groups, and it is measured to make no difference. On toy.yml and chime_sb2.yml,
        preconditioning by each row's mean leaves Ktot bit-identical (361 and 836) and moves
        the worst per-row relative reconstruction error from 2.68e-11 to 2.71e-11 and from
        6.36e-14 to 5.67e-14 -- i.e. one gets slightly worse and one slightly better.
        """

        for sdm in self.sd_matrices.values():
            sdm.factorize(self.epsilon)

        Ftot = sum(sdm.F for sdm in self.sd_matrices.values())
        Ktot = sum(sdm.factor_rank for sdm in self.sd_matrices.values())
        eps_max = max(sdm.epsilon for sdm in self.sd_matrices.values())
        nbeta = (((1 << (self.r - self.R)) * self.M * self.P) if (self.Lmat is None)
                 else ((1 << (self.r - self.Lmat)) * self.N * self.P))

        self.stats.update(Ftot=Ftot, Ktot=Ktot, n_matrices=len(self.sd_matrices),
                          nbeta=nbeta, eps_max=eps_max)

        if progress:
            atomic_print(f'  SdPlan: Lmat={self.Lmat}, {len(self.sd_matrices)} SdMatrices,'
                         f' Ftot={Ftot} rows -> Ktot={Ktot}'
                         f' ({Ftot/max(Ktot,1):.1f}x compression), nbeta={nbeta},'
                         f' eps_max={eps_max:.3g}, {self._n_sliced} rows sliced')


    # ---------------- the lift ----------------

    def lift_sd_vectors(self):
        """The FINE (ndm, M, P) vector that sd_vectors' per-group terms add up to.

        With freq_variances None that is y_true, the base tree's untruncated row sums; with
        freq_variances set it is ``A v``. Either way it is UNTRUNCATED -- no SVD is involved --
        and it is FINE whatever Lmat is, which is why it runs over sd_vectors (keyed by the
        FULL sdbits) rather than over sd_matrices.

        An explicit method rather than part of the constructor: a caller who wants only the
        per-group factors would otherwise be handed an O(nalpha) array to discard.
        """

        r, R, ndm, P = self.r, self.R, self.ndm, self.P
        out = np.zeros((ndm, self.M, P))

        # ACCUMULATE, not assign: one alpha receives contributions from many groups. The body
        # is the fine Q lift's index arithmetic, over one vector per group instead of one
        # factor block.
        #
        # Do NOT hoist the 'dfull' construction out of the subband loop. It is cacheable by l,
        # of which there are at most R+1 <= 5 values, and it looks like free money -- measured
        # at chord_sb2_et.yml it is worth 2% of this lift, i.e. 0.3% of
        # compute_detrender_free_varfine().
        for (sdbits, yg) in self.sd_vectors.items():
            dbits = sdbits >> _SBITS_WIDTH                          # FULL key
            yg = yg.reshape(1 << dbits.bit_count(), P)
            for n in _iter_bits(sdbits & _SBITS_MASK):
                ll, mb = int(self.lev[n]), int(self.mbase[n])
                dfull = ((np.arange(ndm)[:, None] << R)
                         | (np.arange(1 << ll)[None, :] << (R - ll)))
                idx = SparseTile._remap_d(dfull, (1 << r) - 1, dbits)         # (ndm, 2^ll)
                out[:, mb:mb + (1 << ll), :] += (2.0 ** (R - ll)) * yg[idx]

        return out


def compute_detrender_free_base_map(config, *, L=None, epsilon=None, max_bytes=None,
                                    progress=False, debug=False):
    """Analytic variance map of 'config''s base tree (ipri = et_level = 0), factored.

    No detrender, no DedispersionPlan and no GPU. See the module docstring for the algorithm
    and for the scaling limit.

    Parameters
    ----------
    L : int or None
        Coarse-graining rank, R <= L <= r, or None for a FINE map. The result equals
        ``compute_detrender_free_base_map(config).coarse_grain(L)`` up to the SVD truncation
        (measured, 8e-14 relative), and is very much cheaper: at chord_sb2_et.yml the lifted
        Q goes from 63.8 GiB to 17.5 at L = R = 4, 4.3 at L = 6 and 0.96 at L = 8. See
        VarianceMap's module docstring for the beta convention, and note L = R is NOT the
        same as fine -- it leaves the DM axis alone but still merges M -> N, which is 3.6x at
        chord_sb2_et.yml.

        The whole optimization rests on the no-detrender monotonicity theorem
        (notes/variance_map.tex, appendix "Monotonicity of the variance map in the DM bits"),
        which makes the max-envelope a SLICE. See SdPlan._emit().
    epsilon : float or None
        Relative singular-value threshold, per group. None uses
        SdMatrix.default_epsilon().
    max_bytes : int or None
        Ceiling on the lifted Q, which is the only large allocation here (63.8 GiB for a fine
        map at chord_sb2_et.yml scale). None means NO LIMIT; the size is reported before the
        allocation either way, so a run that is about to fail says why rather than being
        killed by the OOM reaper.
    debug : bool
        Turn on SdPlan's planning-pass cross-checks.

    Notes
    -----
    y_true IS FINE WHATEVER L IS -- a length-nalpha vector, per VarianceMap's convention --
    so it is not the row sums of the returned (possibly coarse) matrix, and it cannot be read
    off the coarse factors. It is accumulated separately; see SdPlan.

    It is also the PRE-TRUNCATION row sum, i.e. the TRUE map's rather than the stored map's.
    row_sums() will therefore differ from it at the 1e-11 level, and get_distance() will
    return ~1 + 1e-11 rather than exactly 1. That is normal and is exactly what the distance
    machinery is built to measure -- it is not a bug to fix.

    is_admissible is True, for both values of L. Before truncation the fine map IS A_true and
    the coarse map IS Abar (the max-envelope dominates every member by construction), so both
    qualify. What is left is the SVD truncation, which is SIGNED: admissibility can fail by
    O(epsilon) in max_diff, where epsilon is the per-group threshold -- measured between 1.0
    and 7.6 times epsilon across four decades. Tighten epsilon if a stricter guarantee is
    needed. Do not read max_r for this: it is a per-element relative measure with no floor,
    so on toy.yml it reports 2.0e-9 for the same 2.7e-11 error, amplified ~200x by the
    matrix's dynamic range.

    pinned_columns stays empty; a caller who needs a nonnegative W column for the LP
    machinery calls pin_column() afterwards.
    """

    t0 = time.time()

    # Cheap (const, microseconds) and worth doing before the tile pass and the lift. Note
    # compute_detrender_free_multi_map() validates too and reaches this function afterwards,
    # so the call runs twice on that path; that costs nothing and is better than either
    # caller assuming the other did it.
    config.validate()

    plan = SdPlan(config, Lmat=L, epsilon=epsilon, progress=progress, debug=debug)

    r, R, N, M, P = plan.r, plan.R, plan.N, plan.M, plan.P
    nfreq, ndm, nalpha = plan.nfreq, plan.ndm, plan.nalpha
    lev, mbase = plan.lev, plan.mbase
    stats = plan.stats
    Ktot, nbeta = stats['Ktot'], stats['nbeta']
    L = plan.Lmat                              # int()-ed and range-checked by SdPlan

    nbytes = 8 * nbeta * Ktot
    # Unconditional, not under 'progress': this is the number that makes an OOM diagnosable.
    what = 'nalpha' if (L is None) else f'nbeta at L={L}'
    atomic_print(f'compute_detrender_free_base_map: lifting to Q ({nbeta} x {Ktot}),'
                 f' {nbytes/(1<<30):.2f} GiB ({what})')

    if (max_bytes is not None) and (nbytes > max_bytes):
        raise RuntimeError(f'compute_detrender_free_base_map: the lifted Q is'
                           f' {nbytes/(1<<30):.1f} GiB ({what}={nbeta}, Ktot={Ktot}), over'
                           f' the caller-supplied max_bytes={max_bytes/(1<<30):.1f} GiB.'
                           + ('' if (L is not None) else
                              ' Passing L would shrink it by 2^(L-R) * M/N.'))

    Wtot = np.zeros((nfreq, Ktot))

    # ---- the Q lift. Two branches; W is unchanged by L (coarse-graining acts on ROWS of A).

    if L is None:
        # A 4-d view of Q, so the per-subband writes are natural. Reshaped to (nalpha, Ktot)
        # at the end, which is a no-op: alpha = (d*M + m)*P + p is exactly this axis order.
        Qtot = np.zeros((ndm, M, P, Ktot))
        k0 = 0
        for sdm in plan.sd_matrices.values():
            K = sdm.factor_rank
            Qg = sdm.Q_factor.reshape(sdm.D, P, K)
            dbits = sdm.sdbits >> _SBITS_WIDTH

            for n in _iter_bits(sdm.sdbits & _SBITS_MASK):
                ll, mb = int(lev[n]), int(mbase[n])

                # Undo the level-r normalization for THIS subband (see SdPlan._emit()).
                # Sanity: at l == R (a full-band subband) the factor is 2^0 = 1 and
                # d_full = (d << R) | e is the honest r-bit index; at l == 0 the factor is
                # 2^R, matching the fact that a level-0 subband sums 2^(r-R) tree-freqs with
                # normalization 2^-((r-R)/2) while the level-r tile carries 2^-(r/2).
                fac = 2.0 ** (R - ll)

                # The virtual level-r delay index of multiplet (n, e) at coarse DM d. Its low
                # R-l bits are zero, which is why dbits is required to have none there.
                dfull = ((np.arange(ndm)[:, None] << R)
                         | (np.arange(1 << ll)[None, :] << (R - ll)))
                idx = SparseTile._remap_d(dfull, (1 << r) - 1, dbits)     # (ndm, 2^ll)

                # ASSIGN: each group owns a disjoint column block, and within a group the
                # subbands own disjoint multiplet ranges.
                Qtot[:, mb:mb + (1 << ll), :, k0:k0+K] = fac * Qg[idx]

            Wtot[sdm.freq_indices, k0:k0+K] = sdm.W_factor
            k0 += K
    else:
        # The coarse lift is a simpler sibling: beta = (dc*N + n)*P + p has no fine-DM axis,
        # and the delay index is dc << L -- the dyadic block's bottom (see SdPlan._emit()),
        # which is l-INDEPENDENT. So 'idx' hoists out of the subband loop, which it cannot do
        # above.
        Qtot = np.zeros((1 << (r - L), N, P, Ktot))
        dc_full = np.arange(1 << (r - L), dtype=np.int64) << L
        k0 = 0
        for sdm in plan.sd_matrices.values():
            K = sdm.factor_rank
            Qg = sdm.Q_factor.reshape(sdm.D, P, K)
            dbits = sdm.sdbits >> _SBITS_WIDTH                 # COARSE: no bits below L
            idx = SparseTile._remap_d(dc_full, (1 << r) - 1, dbits)       # (2^(r-L),)

            # The 2^(R-l[n]) factor is still per subband, and coarse-graining does not touch
            # it -- it undoes the stored row's level-r normalization, which is orthogonal.
            for n in _iter_bits(sdm.sdbits & _SBITS_MASK):
                Qtot[:, n, :, k0:k0+K] = (2.0 ** (R - int(lev[n]))) * Qg[idx]

            Wtot[sdm.freq_indices, k0:k0+K] = sdm.W_factor
            k0 += K

    assert k0 == Ktot, (k0, Ktot)

    # ---- the y_true lift, which is FINE for both branches. A tripwire, not a check: y_true
    # is BY DEFINITION the untruncated row sums, so a weighted plan would silently return
    # A v here instead. That is compute_detrender_free_varfine()'s job, not this one's.
    assert plan.freq_variances is None
    ytrue = plan.lift_sd_vectors()

    dt = time.time() - t0
    if progress:
        atomic_print(f'  compute_detrender_free_base_map: done in {dt:.2f} seconds')

    return VarianceMap.from_factors(
        config, plan.itree0, Qtot.reshape(nbeta, Ktot), Wtot, detrender=None, L=L,
        y_true=ytrue.reshape(nalpha), tree=plan.tree0, is_admissible=True,
        history=[dict(step='compute_detrender_free_base_map', time=dt, L=L, nalpha=nalpha,
                      nbeta=nbeta, Ktot=Ktot, Q_nbytes=nbytes,
                      n_matrices=stats['n_matrices'], n_entries=stats['n_entries'],
                      n_straddled=stats['n_straddled'], n_sliced=stats['n_sliced'],
                      Ftot=stats['Ftot'], epsilon=epsilon, eps_max=stats['eps_max'])])


####################################   the multimap   ####################################


def compute_detrender_free_multi_map(config, *, L=None, epsilon=None, max_bytes=None,
                                     progress=False, debug=False):
    """Analytic variance map of EVERY primary tree, as a VarianceMultiMap. No detrender.

    Computes the base tree once with compute_detrender_free_base_map() and SLICES it for
    every other primary tree, so the whole multimap costs one base-map computation plus some
    array copies -- no second tile pass, no second SVD. Compare compute_variance_multimap(),
    which runs one dedispersion pass per input channel.

    A caller who needs only ``A v`` per tree, rather than the maps, wants
    compute_detrender_free_varfine() instead: it never forms Q at all.

    Parameters are compute_detrender_free_base_map()'s, and mean the same thing, with two
    differences:

    L : int or None
        The legal range is [R, r-1] when num_primary_trees > 1, since a downsampled primary
        tree has rank r-1. Checked BEFORE the base map is computed.
    max_bytes : int or None
        Still bounds the BASE map's Q, which is roughly half of what this function
        allocates. The multimap total is
        ``(1 + 0.5 * sum over gamma>0 of P_gamma/P_0)`` times the base Q -- 1.94x at every
        shipped CHORD/CHIME config, 1.65x at toy.yml -- and is reported before any slice is
        taken.

    Notes
    -----
    THE SLICE IS PROPOSITION 2 of notes/variance_map.tex, appendix "Variance maps of a
    config's trees are row-restrictions of one another" (\\label{ssec:restriction_ds}), read
    as an array operation. Every primary tree's map is the same slice of the base tree's on
    three axes -- the upper half of the coarse-DM axis, every multiplet, and the first
    P_gamma profiles -- with W and mid untouched, since the restriction acts on ROWS of A.

    Early-trigger trees are not stored: they are Proposition 1, which
    VarianceMultiMap.apply_fine() derives from the parent. See that class's docstring.
    """

    from ..pirate_pybind11 import DedispersionTree

    t0 = time.time()

    # VALIDATE BEFORE COMPUTING: the base map can take minutes and tens of GiB, and raising
    # afterwards for a reason knowable from the config alone is pure waste. validate() is
    # const and costs microseconds, and make_tree() needs no plan, no GPU and no map.
    config.validate()

    npri = int(config.num_primary_trees)
    itree0 = int(config.dedispersion_tree_index(0, 0))
    tree0 = make_tree(config, itree0)
    r0, R = int(tree0.total_rank()), int(tree0.frequency_subbands.pf_rank)

    # THE LEGAL RANGE FOR 'L' IS THE DOWNSAMPLED TREES', NOT THE BASE TREE'S. Left unchecked
    # this still fails, but loudly and confusingly: VarianceMap.__init__ raises "L=6 is out of
    # range [R, r] = [2, 5]" from the CHILD's constructor, naming a rank the caller never
    # asked for.
    if L is not None:
        hi = r0 if (npri == 1) else (r0 - 1)   # a downsampled primary tree has rank r0 - 1
        if not (R <= int(L) <= hi):
            raise RuntimeError(
                f'compute_detrender_free_multi_map: L={L} is out of range [R, r] ='
                f' [{R}, {hi}].'
                + (f' The upper bound is {r0-1} rather than the base tree\'s own {r0}'
                   f' because this config has num_primary_trees={npri}, and a downsampled'
                   f' primary tree has rank {r0-1}.' if (npri > 1) else ''))

    itrees = [int(config.dedispersion_tree_index(g, 0)) for g in range(1, npri)]
    trees = [make_tree(config, i) for i in itrees]
    Ps = [int(t.nprofiles) for t in trees]

    # THE ONE PRECONDITION OF THE SLICE THAT WOULD FAIL SILENTLY. Proposition 2 needs three
    # facts, and the other two are already enforced where they would fail LOUDLY: r_gamma =
    # r_0 - 1 with R unchanged, which VarianceMap.__init__'s shape check catches because the
    # slice would then have the wrong row count; and P_gamma <= P_0, which is
    # config.validate()'s non-increasing max_width rule above. This one is different -- if a
    # primary tree carried the base tree's multiplet index over PERMUTED rather than
    # unchanged, every shape would still match and the slice would simply select the wrong
    # rows.
    #
    # It cannot happen today, and not because of anything validate() does: both trees get
    # FrequencySubbands(restrict_subband_counts(counts, 0), fmin, fmax), and no argument
    # there depends on the primary tree index (DedispersionTree.cpp). So this is a tripwire
    # on that C++ rather than a check on the caller -- which is why it is an assert, and why
    # it is the only one of the three left. Note test_subband_property() checks the weaker
    # statement: m_index_mapping() in both argument orders gives set EQUALITY, not identity.
    M0 = int(tree0.frequency_subbands.M)
    for (g, t) in enumerate(trees):
        m_map = DedispersionTree.m_index_mapping(tree0, t)
        assert np.array_equal(m_map, np.arange(M0)), (g + 1, np.asarray(m_map))

    prov = dict(algorithm='detrender_free', L=L, epsilon=epsilon,
                num_primary_trees=npri)

    base = compute_detrender_free_base_map(config, L=L, epsilon=epsilon,
                                           max_bytes=max_bytes, progress=progress,
                                           debug=debug)
    if npri == 1:
        prov['total_seconds'] = time.time() - t0
        return VarianceMultiMap(config, [base], detrender=None, provenance=prov)

    N, M, P0 = base.nsubbands, base.nmultiplets, base.nprofiles
    K = base.factor_rank
    D0 = 1 << (r0 - R)

    # Rows of Q are alpha = (d*M + m)*P + p when fine, and beta = (dc*N + n)*P + p when
    # coarse, so the same 4-d view serves both with (D0, M) or (2^(r0-L), N) as the first two
    # axes. y_true is FINE WHATEVER L IS (VarianceMap's convention), so its view is always the
    # (D0, M, P0) one -- the single easiest thing to get wrong here.
    nrow0, ax1 = ((1 << (r0 - L)), N) if (L is not None) else (D0, M)
    Q4 = np.asarray(base.Q).reshape(nrow0, ax1, P0, K)
    y3 = np.asarray(base.y_true).reshape(D0, M, P0)

    # The coarse groups line up because dc_0 = d_0 >> (L-R) and d_0 = d_gamma + D0/2 give
    # dc_0 = dc_gamma + 2^(r0-L-1) exactly, provided 2^(L-R) divides D0/2 = 2^(r0-R-1), i.e.
    # L <= r0-1. That is not an extra assumption: it is exactly the L range validated above.

    nbytes = 8 * K * (base.nbeta + sum((base.nbeta // 2) * P // P0 for P in Ps))
    atomic_print(f'compute_detrender_free_multi_map: {npri} primary trees, P='
                 f'{[P0] + Ps}, total Q {nbytes/(1<<30):.2f} GiB'
                 f' ({1 + 0.5*sum(P/P0 for P in Ps):.3f}x the base map)')

    maps = [base]
    for gamma in range(1, npri):
        itree, tree, Pg = itrees[gamma - 1], trees[gamma - 1], Ps[gamma - 1]

        # THE NO-DETRENDER HYPOTHESIS IS THE WHOLE ARGUMENT, not a formality. Proposition 1
        # (early triggers) holds whatever the upstream chain; Proposition 2 does NOT, and the
        # appendix says so twice. Measured on _make_test_config(6, [2,2,1], num_primary_trees=2)
        # against the brute-force sweep: 4.9e-7 without a detrender, and 2.1 WITH one -- a
        # factor of two wrong, not a rounding difference. If a detrender path is ever added to
        # this module, this function must not be part of it.
        #
        # np.ascontiguousarray COPIES ONLY WHEN Pg < P0, which is when the sliced view is
        # non-contiguous; at Pg == P0 it hands back a view of base.Q. That is safe here and
        # only here, because VarianceMap stores its arrays read-only, so the sharing cannot be
        # observed. compute_detrender_free_varfine() takes the same slice of plain writeable
        # arrays and has to use an unconditional .copy() instead.
        maps.append(VarianceMap.from_factors(
            config, itree,
            np.ascontiguousarray(Q4[nrow0//2:, :, :Pg, :].reshape(-1, K)),
            base.W, mid=base.mid, detrender=None, L=L, tree=tree,
            y_true=np.ascontiguousarray(y3[D0//2:, :, :Pg].reshape(-1)),
            # A row subset of a map that dominates A_true elementwise also dominates it, so
            # whatever the base earned, each gamma earns. W and mid are SHARED objects rather
            # than copies: they are identical across trees and VarianceMap stores them
            # read-only, which saves npri copies of (nfreq, K).
            is_admissible=base.is_admissible,
            history=list(base.history) + [dict(step='restrict_to_primary_tree', gamma=gamma,
                                               itree=itree, P=Pg, D=D0//2)]))

    dt = time.time() - t0
    prov['total_seconds'] = dt
    if progress:
        atomic_print(f'  compute_detrender_free_multi_map: {npri} maps in {dt:.2f} seconds')

    return VarianceMultiMap(config, maps, detrender=None, provenance=prov)


####################################   A v, without A   ####################################


def compute_detrender_free_varfine(config, freq_variances, *, progress=False, debug=False):
    """``A v`` at FINE granularity for EVERY tree of 'config'. No detrender, no map formed.

    Returns a length-ntrees list of ``(2^(r-R), M, P)`` float64 arrays INDEXED BY ITREE, each
    in its own tree's geometry -- exactly VarianceMultiMap.apply_fine()'s contract, since that
    is the definition::

        vmm = compute_detrender_free_multi_map(config, L=None)
        vmm.apply_fine(freq_variances)

    The point is to compute that WITHOUT forming a VarianceMap, a lifted Q or an SdMatrix. The
    weight v_F rides along on the per-group accumulator of the shared tile pass (see SdPlan),
    and the result is lifted straight to fine granularity. At chord_sb2_et.yml that is 16
    seconds and 0.49 GiB peak RSS for all ten trees, where the defining expression needs 63.8
    GiB of Q for the base tree alone and cannot be run at all.

    The arrays are fresh and writeable, and no two of them share storage.

    Parameters
    ----------
    freq_variances : array
        Length-nfreq input-channel variances. NOT required to be positive: this is defined
        against VarianceMap.apply(), which does not require it either. The length and the
        dtype are checked, nothing more.
    debug : bool
        Turn on SdPlan's planning-pass cross-checks -- which is ALL of them here, since there
        is no SdMatrix to cross-check.

    Notes
    -----
    THIS IS MORE ACCURATE THAN THE EXPRESSION THAT DEFINES IT. No SVD is involved anywhere, so
    the result is the exact map applied to v; the defining expression holds only to the
    reference path's truncation (measured 3.1e-13 relative on toy.yml). In particular
    ``compute_detrender_free_varfine(config, ones)[itree0].reshape(-1)`` is BITWISE equal to
    ``compute_detrender_free_base_map(config).y_true``, which is documented as the
    pre-truncation row sum.

    THE NO-DETRENDER HYPOTHESIS IS LOAD-BEARING, for the same reason it is in
    compute_detrender_free_multi_map(): the step from the base tree to the other PRIMARY trees
    is Proposition 2, which is false with a Detrender2d in front (measured against the
    brute-force sweep: 4.9e-7 without one, 2.1 with). A future detrender path must not be
    routed through this function.

    There is deliberately no 'L'. The result is fine by definition; a caller who wants it
    coarse-grained applies ``coarse_grain_vector(tree, y, L)`` to it, and a caller who wants a
    coarse MAP wants compute_detrender_free_base_map(config, L=...). Those are not the same
    thing in general -- a coarse map's apply() is ``sum_F max_alpha A[alpha,F] v_F``, which
    dominates ``max_alpha (A v)[alpha]``. There is no 'epsilon' or 'max_bytes' either: no SVD,
    and no allocation worth a ceiling.
    """

    from ..pirate_pybind11 import DedispersionTree

    t0 = time.time()

    # Validate BEFORE building the plan: the tile pass is 13.5 seconds at CHORD, and both a
    # bad config and a length mismatch are knowable without it.
    config.validate()

    nfreq = int(config.get_total_nfreq())
    v = np.asarray(freq_variances, dtype=np.float64)
    if v.shape != (nfreq,):
        raise RuntimeError(f'compute_detrender_free_varfine: expected freq_variances of'
                           f' shape ({nfreq},), got {v.shape}')

    # Same argument for the restriction geometry: O(N) per tree, and it decides whether the
    # slice below is legitimate at all. SdPlan builds tree0 again a few lines down; that is
    # microseconds, and building it here is what keeps the geometry ahead of the tile pass.
    npri = int(config.num_primary_trees)
    tree0 = make_tree(config, int(config.dedispersion_tree_index(0, 0)))
    trees = [make_tree(config, int(config.dedispersion_tree_index(g, 0)))
             for g in range(1, npri)]
    Ps = [int(t.nprofiles) for t in trees]

    # The slice's one silently-failing precondition; see compute_detrender_free_multi_map(),
    # which explains why this is the only one of Proposition 2's three facts left as a check.
    M0 = int(tree0.frequency_subbands.M)
    for (g, t) in enumerate(trees):
        m_map = DedispersionTree.m_index_mapping(tree0, t)
        assert np.array_equal(m_map, np.arange(M0)), (g + 1, np.asarray(m_map))

    plan = SdPlan(config, freq_variances=v, init_sd_matrices=False,
                  progress=progress, debug=debug)
    y0 = plan.lift_sd_vectors()                    # (ndm, M, P0) == A_base @ v
    D0 = plan.ndm

    # Proposition 2 as an array operation, and it is the same slice
    # compute_detrender_free_multi_map() takes of the base map's y_true -- see the comment
    # there, which records what makes it legitimate.
    #
    # .copy() rather than np.ascontiguousarray(): the slice is ALREADY contiguous whenever
    # Pg == P0, so ascontiguousarray would hand back a view of y0 and the returned list would
    # alias itself -- a caller mutating one tree's result would silently corrupt another's.
    per_primary = [y0.reshape(-1)]
    per_primary += [y0[D0//2:, :, :Pg].copy().reshape(-1) for Pg in Ps]

    # Proposition 1 (early triggers), which is the same expansion apply_fine() does.
    out = expand_fine_vectors(config, per_primary)

    if progress:
        atomic_print(f'  compute_detrender_free_varfine: {len(out)} trees in'
                     f' {time.time() - t0:.2f} seconds')
    return out
