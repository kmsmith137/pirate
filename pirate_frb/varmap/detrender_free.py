"""Analytic variance map of a config's BASE tree, with no detrender and no coarse-graining.

compute_detrender_free_base_map(config) returns the variance map A of the tree with
(primary_tree_index, early_trigger_level) == (0, 0), computed analytically -- no dedisperser
run, no DedispersionPlan, no GPU -- and returned in FACTORED form, A = Q W^T. The map is
numerically low-rank because variances vary smoothly with input channel; see
notes/variance_map.tex, subsection "Per-group SVD and stacking".

Only the base tree is needed: with no detrender, every other tree's map is a ROW RESTRICTION
of the base tree's (notes/variance_map.tex, appendix "Variance maps of a config's trees are
row-restrictions of one another"). Assembling a VarianceMultiMap from this map is a separate
piece of work -- the cross-primary-tree restriction it needs does not exist yet.

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

'L' IS WHAT GETS OVER THAT WALL, and it is nearly free here. With no detrender the
coarse-grained map is not a max at all but a SLICE -- keep the bottom of each dyadic DM block
-- so the coarse map is built directly rather than by building the fine one and reducing it.
At chord_sb2_et.yml the lifted Q goes 63.8 GiB (fine) -> 17.5 (L = R = 4) -> 4.3 (L = 6) ->
0.96 (L = 8). Almost all of that is the row count: Ktot only moves 1435 -> 1399 at L = 6, so
what L buys is 14.6x SHORTER columns of Q, not more compressible groups. Steps 1 and 2 are
unchanged by L by design -- it is the lift that goes away.

The remaining scaling fix is a second, global SVD round that runs BEFORE the lift; the
per-group factors that round needs are what build_sd_matrices() returns.

TWO THINGS HERE ARE NON-OBVIOUS, and getting either wrong gives a wrong ANSWER rather than a
crash. Both are derived at the code that implements them.

  - Every row is stored in a common "level-r" normalization, and the lift undoes it per
    subband with a factor 2^(R-l). See emit() and the lift. Measured: omitting it moves
    apply() by 75% on toy.yml.
  - Half-aligned subbands whose footprint straddles the subband midpoint need their own
    branch. See the straddle loop. Measured: omitting it moves apply() by 1.6e-3 on toy.yml
    -- rare (1 row of 645) but not negligible.

A THIRD, for the L path: y_true is defined at FINE granularity whatever the map's
coarse-graining rank (VarianceMap's class docstring is explicit about this), so it cannot be
read off the coarse factors. It is accumulated separately, in the full delay-bit basis, in
'sd_vectors' -- see build_sd_matrices().
"""

import time

import numpy as np

from .VarianceMap import VarianceMap, make_tree
from .VarianceMultiMap import VarianceMultiMap
from ..slow_avar.SparseTile import SparseTile, SparseTileTriple
from ..slow_avar.PfVariance import PfVarianceConvolver
from ..utils import atomic_print


# The sdbits key is (dbits << _SBITS_WIDTH) | sbits. The split is chosen so the key would
# still fit a uint64 in a future C++ port: N <= 42 subbands (notes/dedispersion.tex, section
# "Subbanded dedispersion", at R <= constants::max_peak_finding_rank == 4) and r <= 16
# (constants::max_tree_rank), so 42 + 16 < 64. build_sd_matrices() asserts both bounds up
# front, because raising either would silently corrupt keys rather than fail.
_SBITS_WIDTH = 42
_SBITS_MASK = (1 << _SBITS_WIDTH) - 1


def _iter_subbands(sbits):
    """The set bits of 'sbits', ascending."""
    out = []
    while sbits:
        b = sbits & (-sbits)
        out.append(b.bit_length() - 1)
        sbits ^= b
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
    so its terms are accumulated in the FULL delay-bit basis, in build_sd_matrices()'s
    'sd_vectors' -- which this matrix's columns are a slice of once L is set.
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


def _subband_geometry(tree):
    """The length-N per-subband tables of the algorithm, as a dict of int64 arrays.

    All in TOPLEVEL TREE-FREQ units: a coarse channel is 2^(r-R) tree-freqs wide, so subband
    n occupies [I_lo[n], I_hi[n]), of width 2^c[n] with c[n] = r-R+l[n] its own tree depth.
    """

    fs = tree.frequency_subbands
    r, R = int(tree.total_rank()), int(fs.pf_rank)

    lev = np.asarray(fs.n_to_level, dtype=np.int64)
    flo = np.asarray(fs.n_to_flo, dtype=np.int64)
    fhi = np.asarray(fs.n_to_fhi, dtype=np.int64)

    I_lo = flo << (r - R)
    I_hi = fhi << (r - R)

    # Case 1 (aligned): I_n is a node of the toplevel tree at level c, so ordinary aligned
    # iteration reproduces the subband's merges. Case 2 (half-aligned, l > 0 and odd index):
    # I_n starts at an odd multiple of 2^(c-1) and is NOT a node of the tree. See
    # notes/dedispersion.tex, section "Subbanded dedispersion".
    case1 = (flo & ((1 << lev) - 1)) == 0

    # Exact: I_hi - I_lo = 2^c, and the only branch that reads I_mid has l >= 1 hence c >= 1,
    # so I_mid = I_lo + 2^(c-1). Note the midpoint is generic -- a case-1 subband's top merge
    # joins its two halves at the same point -- but for case 1 the halves are the ALIGNED
    # pair, which SparseTileTriple.iterate() already merges correctly, so there is nothing to
    # detect and I_mid is never consulted there.
    I_mid = (I_lo + I_hi) // 2

    return dict(l=lev, c=(r - R) + lev, I_lo=I_lo, I_hi=I_hi, I_mid=I_mid, case1=case1,
                mbase=np.asarray(fs.n_to_mbase, dtype=np.int64))


def build_sd_matrices(config, *, L=None, epsilon=None, progress=False, debug=False):
    """Everything except the lift, for the base tree of 'config'.

    Returns (tree, itree0, sd_matrices, sd_vectors, stats). THIS FUNCTION DOES NO LIFTING --
    that is what its name promises, and y_true is a lift. It returns two parallel
    accumulators and lets compute_detrender_free_base_map() consume both.

    THE TWO DICTS DO NOT SHARE A KEY SPACE, and that is the one thing to get right here:

      sd_matrices   COARSE sdbits -> SdMatrix, each (capacity, D_coarse * P)
      sd_vectors    FULL   sdbits -> (D_full * P,) float64, the summed rows

    At L = None the two key spaces COINCIDE, so indexing one with the other's key passes
    every fine test. It is also a QUIET bug with L set: a coarse key is a syntactically valid
    full key whenever the full dbits happened to have no bits below L, so the lookup can
    succeed and return another group's data. Hence every use site names which key it is
    using, and the sizing pass asserts len(sd_vectors) >= len(sd_matrices) -- coarsening can
    only merge keys, never split them.

    'stats' carries n_entries, n_straddled, n_sliced, Ftot, Ktot, n_matrices, L, nbeta and
    eps_max. n_straddled and n_sliced are there because a test cannot see from the returned
    VarianceMap whether the straddle branch or the coarse slice actually did anything.

    'itree0' is in the return because IT IS NOT ALWAYS ZERO: early_trigger_level DESCENDS
    within a primary-tree family, so the e = 0 tree is the LAST of its family. It is 0 for
    every shipped config -- which is exactly what would make an assertion to that effect a
    trap -- and 1 for _make_test_config(7, [2,2,1], num_early_triggers=1). DedispersionTree
    does not carry its own index, so 'tree' cannot supply it, and having the caller recompute
    config.dedispersion_tree_index(0, 0) would admit a silent inconsistency:
    VarianceMap.from_factors(config, itree, ..., tree=tree) takes both at face value.

    'debug' turns on the O(F) and O(subbands) cross-checks: that no SdMatrix receives two rows
    from one input channel, and that every subband of an entry predicts the same dbits. Both
    are statements the shared-row pooling depends on, and both are too expensive to leave on
    at production scale.

    This is split out from compute_detrender_free_base_map() because the per-group factors,
    not the lifted Q, are what a second (global) SVD round and a coarse-graining pass would
    consume -- and those callers would otherwise be handed a y_true they have to discard.
    """

    from ..pirate_pybind11 import constants

    itree0 = int(config.dedispersion_tree_index(0, 0))
    tree = make_tree(config, itree0)
    fs = tree.frequency_subbands

    r, R = int(tree.total_rank()), int(fs.pf_rank)
    N, M, P = int(fs.N), int(fs.M), int(tree.nprofiles)
    nfreq = int(config.get_total_nfreq())

    # The sdbits packing has exactly these two headrooms; see _SBITS_WIDTH.
    assert N <= _SBITS_WIDTH, (N, _SBITS_WIDTH)
    assert r <= constants.max_tree_rank, (r, constants.max_tree_rank)

    if L is not None:
        L = int(L)
        # Same bounds and the same wording as VarianceMap.coarse_grain(), so the two read the
        # same. L < R is meaningless because beta's definition uses dc = d >> (L-R); L > r is
        # impossible because there are only 2^(r-R) coarse DMs to merge.
        if not (R <= L <= r):
            raise RuntimeError(f'build_sd_matrices: L={L} is out of range [R, r] ='
                               f' [{R}, {r}] for this config\'s base tree')

    # The alpha convention assumes 2^R coarse DM channels per multiplet, which is what an
    # unset (auto) dm_downsampling gives. validate() already requires it, so this is a
    # tripwire rather than a check.
    dmds = int(config.primary_trees[0].dm_downsampling)
    if dmds != 0:
        raise RuntimeError(f'build_sd_matrices: primary tree 0 has dm_downsampling={dmds},'
                           " but the variance map's index convention needs the auto value 0")

    g = _subband_geometry(tree)
    I_lo, I_mid, c, lev = g['I_lo'], g['I_mid'], g['c'], g['l']
    case1 = g['case1']
    cmap = np.asarray(config.make_channel_map(), dtype=np.float64)
    convolver = PfVarianceConvolver()          # ONE shared instance for the whole run

    def intersect(j0, j1, n):
        """This channel's footprint [j0, j1) intersected with subband n. Empty iff lo >= hi.

        Shared by the planning loop and the tile loop so the two cannot drift.
        """
        return max(j0, int(I_lo[n])), min(j1, int(g['I_hi'][n]))

    def predict_dbits_r(lo, hi, n):
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
        cc = int(c[n])
        return SparseTile._predict_dbits(cc, lo - int(I_lo[n]), hi - lo) << (r - cc)

    # Two plans rather than one plan with a nullable discriminant. 'ifreq' is a member of both
    # tuples, not just the loop variable: the tile pass walks the flat plans and needs it to
    # rebuild the channel's gridding triple and to write freq_indices.
    #
    #   unstraddled_plan: (ifreq, lo, hi, sdbits)          the common path
    #   straddled_plan:   (ifreq, straddle_n, sdbits)      case-2 midpoint straddles
    #
    # The straddled triple carries no (lo, hi) because (ifreq, straddle_n) already determines
    # them, via the same intersect() the tile pass calls.
    unstraddled_plan = []
    straddled_plan = []
    footprint = np.zeros((nfreq, 2), dtype=np.int64)
    seen_subbands = 0

    for ifreq in range(nfreq):
        tri = SparseTileTriple.make_tree_gridding_output(cmap, ifreq)
        j0, j1 = int(tri.f0), int(tri.f0 + tri.nf)
        footprint[ifreq] = (j0, j1)

        local = {}                             # (lo, hi) -> sbits, unstraddled subbands only

        for n in range(N):
            lo, hi = intersect(j0, j1, n)
            if lo >= hi:
                continue                       # this subband does not see this channel
            seen_subbands |= (1 << n)
            if (not case1[n]) and (lo < I_mid[n] < hi):
                dbits = predict_dbits_r(lo, hi, n)
                straddled_plan.append((ifreq, n, (dbits << _SBITS_WIDTH) | (1 << n)))
            else:
                local[(lo, hi)] = local.get((lo, hi), 0) | (1 << n)

        for ((lo, hi), sbits) in local.items():
            n0 = (sbits & -sbits).bit_length() - 1        # any subband of the entry will do
            dbits = predict_dbits_r(lo, hi, n0)

            # The rule agrees with the simpler GLOBAL form on this branch: [lo,hi) lies inside
            # a single 2^(c-1)-aligned block, so neither f0's low bits nor the XOR of the two
            # endpoints is changed by the shift of origin. This is the cheapest available
            # check that the entry was classified as unstraddled correctly.
            assert dbits == SparseTile._predict_dbits(r, lo, hi - lo), (ifreq, lo, hi, n0)

            if debug:
                # One shared row is legitimate only because every subband of the entry
                # predicts the same dbits -- they may differ in l and in I_lo, and they all
                # agree because they all agree with the global form above.
                for n in _iter_subbands(sbits):
                    assert predict_dbits_r(lo, hi, n) == dbits, (ifreq, lo, hi, n, n0)

            unstraddled_plan.append((ifreq, lo, hi, (dbits << _SBITS_WIDTH) | sbits))

    # A subband seeing no channel at all would give identically-zero rows of A, which breaks
    # y_true and hence get_distance(). This is the analogue of PfAvarExact's m_cnt >= 1
    # assert.
    assert seen_subbands == ((1 << N) - 1), \
        f'subband(s) {_iter_subbands(((1 << N) - 1) & ~seen_subbands)} see no input channel'

    # Every subband of an entry must have R - l[n] zero low bits in the entry's dbits, since
    # the lift's virtual level-r delay index (d << R) | (e << (R-l)) has none there. The
    # argument that this holds is EXACTLY the straddle discriminant, which is why the
    # assertion cannot fire for any other reason:
    #
    #   Write [lo,hi) for the intersection and use the closed form of _predict_dbits(r,.,.).
    #   The leading run occupies bit positions [r-j1, r-1], so it dips below R-l iff j1 > c;
    #   the isolated bit sits at r-1-h, so it dips iff h >= c. Now j1 <= bit_length(hi-lo-1)
    #   <= c always, since [lo,hi) is contained in I_n and I_n has width 2^c -- the run never
    #   dips. And h is the highest bit at which lo and hi-1 differ. In case 1, I_n is
    #   2^c-aligned, so lo and hi-1 agree above bit c-1 and h <= c-1. In case 2, I_n spans the
    #   two 2^(c-1)-aligned blocks either side of its midpoint: if [lo,hi) lies in one of
    #   them, again h <= c-2; if it STRADDLES, then lo >> (c-1) and (hi-1) >> (c-1) differ, so
    #   h >= c and the assertion fails.
    #
    # So assertion failures and midpoint straddles are the same set. With the straddle branch
    # taken explicitly the assertion now holds everywhere -- trivially so on that branch,
    # where the << (r - c) supplies R - l[n] zero low bits. Keep it: it is the statement that
    # makes the lift's index formula well defined.
    for (_, _, _, sdbits) in unstraddled_plan:
        dbits = sdbits >> _SBITS_WIDTH
        for n in _iter_subbands(sdbits & _SBITS_MASK):
            assert (dbits & ((1 << (R - int(lev[n]))) - 1)) == 0, (dbits, n, int(lev[n]))

    # ---- allocate. ONE dict, fed by both plans: nothing stops a straddled entry's sdbits
    # from coinciding with an unstraddled one's (measured, it never does, but two dicts merged
    # with update() would silently drop a group).

    sd_capacities = {}      # COARSE sdbits -> row count
    sd_vectors = {}         # FULL   sdbits -> (D_full * P,) accumulator for y_true's terms

    def count(sdbits):
        # Keying sd_matrices by the COARSE sdbits is where the extra pooling comes from:
        # entries whose full dbits differ only below L now share a matrix. Measured, that is
        # worth 0.2% of Ktot at L = R and 26% at L = R+6 -- small at the interesting end, but
        # free, and it is the natural key.
        key = _coarsen_sdbits(sdbits, L)
        sd_capacities[key] = sd_capacities.get(key, 0) + 1
        if sdbits not in sd_vectors:
            D_full = 1 << (sdbits >> _SBITS_WIDTH).bit_count()
            sd_vectors[sdbits] = np.zeros(D_full * P)

    for (_, _, _, sdbits) in unstraddled_plan:
        count(sdbits)
    for (_, n, sdbits) in straddled_plan:
        # A straddling subband always gets a row to itself, so the two fields of its sdbits
        # are redundant with each other -- and checkable.
        assert (sdbits & _SBITS_MASK) == (1 << n), (sdbits, n)
        count(sdbits)

    sd_matrices = {}
    for (sdbits, capacity) in sd_capacities.items():
        dbits = sdbits >> _SBITS_WIDTH
        sd_matrices[sdbits] = SdMatrix(sdbits, capacity, 1 << dbits.bit_count(), P)

    # Coarsening can only merge keys, never split them, so this holds always -- and fails
    # loudly if the two dicts are ever indexed with each other's key. See the docstring.
    assert len(sd_vectors) >= len(sd_matrices), (len(sd_vectors), len(sd_matrices))

    n_entries = len(unstraddled_plan) + len(straddled_plan)
    if progress:
        atomic_print(f'  build_sd_matrices: r={r} R={R} N={N} M={M} P={P} nfreq={nfreq}'
                     f' L={L}: {n_entries} entries ({len(straddled_plan)} straddled) ->'
                     f' {len(sd_matrices)} SdMatrices ({len(sd_vectors)} y_true groups),'
                     f' {n_entries/nfreq:.2f} rows per input channel')

    # ---- build the tiles and fill the rows. Two straight-line loops, one per plan.
    #
    # Two calls to make_tree_gridding_output() are in play and they are NOT the same call: the
    # planning loop above needed the UNCLIPPED footprint once per channel (cached in
    # 'footprint'), while these need a triple CLIPPED to the entry's own [lo, hi), which
    # cannot be hoisted. Its repeated searchsorted over cmap is negligible at these sizes.

    n_sliced = 0        # rows whose coarse slice actually removed a delay bit (n_rm > 0)

    def emit(ifreq, tile, klev, sdbits):
        """The level-r normalization, then one row.

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

        With L set, the row is then COARSE-GRAINED, which is a slice rather than a max -- see
        the comment at the slice below.
        """

        nonlocal n_sliced

        # This is the assert that closes the loop between the planning pass's closed form and
        # the actual iteration, on both plans. It reads the FULL sdbits, as does every other
        # assertion here -- plan entries carry the full mask whatever L is, and the coarse key
        # is derived only where it is needed. Everything downstream -- the sizing, the shared
        # rows, the lift -- is built on that prediction being right.
        assert (tile.dbits << (r - klev)) == (sdbits >> _SBITS_WIDTH), \
            (ifreq, klev, tile.dbits, sdbits >> _SBITS_WIDTH)

        # scale**2 because variance is quadratic (this is what PfVariance.add_tile() does);
        # the [0] drops the length-1 nf axis. Omitting scale**2 is silently wrong wherever an
        # edge tile deferred its 1/sqrt(2).
        var = (tile.scale ** 2) * convolver.variance(tile.data, P)[0]      # (D_full, P)
        var *= 2.0 ** (-(r - klev))

        # y_true is FINE whatever L is, so its terms accumulate in the FULL dbits basis, under
        # the FULL key, BEFORE the slice below.
        sd_vectors[sdbits] += var.reshape(-1)

        if L is not None:
            # COARSE-GRAINING IS A SLICE, NOT A MAX, and this is where that is cashed in.
            #
            # Group beta = (dc, n, p) contains the fine rows whose subband DM index d_m runs
            # over an aligned dyadic block of size 2^(L-R+l): d = dc*2^(L-R) + [0, 2^(L-R)),
            # e = [0, 2^l). With NO DETRENDER the variance is antitone in the bits of d_m, so
            # the max over an aligned dyadic block is attained at its BOTTOM -- see
            # notes/variance_map.tex, appendix "Monotonicity of the variance map in the DM
            # bits (no detrender)", eq:dyadic_property, which says in as many words that a map
            # resolving only the top DM bits "can simply be evaluated at the bottom of each
            # dyadic DM block: no maximization over the block is needed, and the resulting
            # bound is tight".
            #
            # THE NO-DETRENDER HYPOTHESIS IS LOAD-BEARING. The same appendix shows the
            # property is false with a Detrender2d in front, so the day someone adds a
            # detrender path is the day this slice silently becomes wrong.
            #
            # In the level-r labelling the block bottom is "all bits below L zero", since
            # d_full = d_m << (R-l) turns d_m's low (L-R+l) bits into d_full's low L. And
            # _remap_d() packs selected bits in order, so those bits occupy the LOWEST n_rm
            # packed positions, contiguously -- hence a reshape and a slice. Taking [:, -1, :]
            # instead of [:, 0, :] would give a min-envelope, which is why the test set checks
            # n_sliced > 0 and calls check_ref_covers_y_true().
            #
            # The max commutes with the group structure per column: at fixed F every alpha in
            # beta gets its value from ONE plan entry (they share the subband, and each
            # subband is assigned to exactly one entry per channel), so this may be done
            # row-by-row, before any summation over channels and before the SVD.
            n_rm = ((sdbits >> _SBITS_WIDTH) & ((1 << L) - 1)).bit_count()
            n_sliced += int(n_rm > 0)
            var = var.reshape(-1, 1 << n_rm, P)[:, 0, :]           # (D_coarse, P)

        sdm = sd_matrices[_coarsen_sdbits(sdbits, L)]              # COARSE key
        assert sdm.F < sdm.capacity, (sdm.F, sdm.capacity)
        if debug:
            # Within one ifreq, distinct entries carry DISJOINT sbits -- each subband is
            # assigned to exactly one entry of exactly one plan -- hence distinct sdbits. So
            # no SdMatrix ever gets two rows from one channel, which is what makes
            # freq_indices distinct and the W lift a plain scatter. Note this holds ACROSS the
            # two plans, not just within one.
            assert ifreq not in sdm.freq_indices[:sdm.F], (ifreq, sdbits)

        sdm.dense_matrix[sdm.F, :] = var.reshape(-1)
        sdm.freq_indices[sdm.F] = ifreq
        sdm.F += 1

    for (ifreq, lo, hi, sdbits) in unstraddled_plan:
        tri = SparseTileTriple.make_tree_gridding_output(cmap, ifreq, flo=lo, fhi=hi)
        for _ in range(r):
            tri = tri.iterate()
        assert tri.nf == 1 and len(tri.tiles) == 1, (tri.nf, len(tri.tiles))
        emit(ifreq, tri.tiles[0], r, sdbits)

    for (ifreq, n, sdbits) in straddled_plan:
        j0, j1 = int(footprint[ifreq, 0]), int(footprint[ifreq, 1])
        lo, hi = intersect(j0, j1, n)
        cc = int(c[n])
        tri = SparseTileTriple.make_tree_gridding_output(cmap, ifreq, flo=lo, fhi=hi)
        for _ in range(cc - 1):
            tri = tri.iterate()

        # The subband's top merge combines the level-(cc-1) blocks either side of its
        # midpoint, which is NOT an aligned pair -- aligned pairs are (2F, 2F+1). Ordinary
        # iteration would merge the lower block with its (absent) left neighbour and the upper
        # with its (absent) right one, producing two tiles that never combine the way the
        # dedisperser combines them. Indexing the pair off the midpoint is the same pair as
        # the "2f+1, 2f+2" of notes/dedispersion.tex Case 2, without needing that section's
        # 'f' convention. Both halves are present by the definition of "straddle", which the
        # default allow_none=False makes an implicit assertion.
        ublk = int(I_mid[n]) >> (cc - 1)
        lower = tri.get_singleton(ublk - 1)
        upper = tri.get_singleton(ublk)
        emit(ifreq, SparseTile.iterate_singletons(lower, upper), cc, sdbits)

    # Exact allocation: there is no growth and no reallocation, so 'capacity' is never an
    # upper bound that a row count falls short of.
    for sdm in sd_matrices.values():
        assert sdm.F == sdm.capacity, (sdm.sdbits, sdm.F, sdm.capacity)

    # ---- factorize. Note that "rows are in channel order" is NOT true: the two loops above
    # run in sequence, so a matrix fed by both gets its straddled rows after all its
    # unstraddled ones. Nothing depends on row order -- the SVD does not, the W lift scatters
    # by freq_indices, and y_true sums over rows -- but it is an easy thing to assume.
    #
    # NO COLUMN PRECONDITIONER. slow_avar.TmpVmapExact divides by a per-channel scale before
    # its SVD, because it stacks CHANNELS AS COLUMNS of a per-multiplet matrix, where a
    # barely-overlapping edge channel contributes an anomalously small column that a relative
    # threshold would delete. Here channels are ROWS of a much more finely divided set of
    # groups, and it is measured to make no difference. On toy.yml and chime_sb2.yml,
    # preconditioning by each row's mean leaves Ktot bit-identical (361 and 836) and moves the
    # worst per-row relative reconstruction error from 2.68e-11 to 2.71e-11 and from 6.36e-14
    # to 5.67e-14 -- i.e. one gets slightly worse and one slightly better.

    for sdm in sd_matrices.values():
        sdm.factorize(epsilon)

    Ftot = sum(sdm.F for sdm in sd_matrices.values())
    Ktot = sum(sdm.factor_rank for sdm in sd_matrices.values())
    eps_max = max(sdm.epsilon for sdm in sd_matrices.values())
    nbeta = ((1 << (r - R)) * M * P) if (L is None) else ((1 << (r - L)) * N * P)

    if progress:
        atomic_print(f'  build_sd_matrices: Ftot={Ftot} rows -> Ktot={Ktot}'
                     f' ({Ftot/max(Ktot,1):.1f}x compression), nbeta={nbeta},'
                     f' eps_max={eps_max:.3g}, {n_sliced} rows sliced')

    stats = dict(n_entries=n_entries, n_straddled=len(straddled_plan), n_sliced=n_sliced,
                 Ftot=Ftot, Ktot=Ktot, n_matrices=len(sd_matrices), L=L, nbeta=nbeta,
                 eps_max=eps_max)
    return tree, itree0, sd_matrices, sd_vectors, stats


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
        which makes the max-envelope a SLICE. See emit().
    epsilon : float or None
        Relative singular-value threshold, per group. None uses
        SdMatrix.default_epsilon().
    max_bytes : int or None
        Ceiling on the lifted Q, which is the only large allocation here (63.8 GiB for a fine
        map at chord_sb2_et.yml scale). None means NO LIMIT; the size is reported before the
        allocation either way, so a run that is about to fail says why rather than being
        killed by the OOM reaper.
    debug : bool
        Turn on build_sd_matrices()'s O(F) cross-checks.

    Notes
    -----
    y_true IS FINE WHATEVER L IS -- a length-nalpha vector, per VarianceMap's convention --
    so it is not the row sums of the returned (possibly coarse) matrix, and it cannot be read
    off the coarse factors. It is accumulated separately; see build_sd_matrices().

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

    tree, itree0, sd_matrices, sd_vectors, stats = build_sd_matrices(
        config, L=L, epsilon=epsilon, progress=progress, debug=debug)

    fs = tree.frequency_subbands
    r, R = int(tree.total_rank()), int(fs.pf_rank)
    N, M, P = int(fs.N), int(fs.M), int(tree.nprofiles)
    nfreq = int(config.get_total_nfreq())
    ndm = 1 << (r - R)
    nalpha = ndm * M * P
    Ktot, nbeta = stats['Ktot'], stats['nbeta']
    L = stats['L']                             # int()-ed and range-checked by the callee

    g = _subband_geometry(tree)
    lev, mbase = g['l'], g['mbase']

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
        for sdm in sd_matrices.values():
            K = sdm.factor_rank
            Qg = sdm.Q_factor.reshape(sdm.D, P, K)
            dbits = sdm.sdbits >> _SBITS_WIDTH

            for n in _iter_subbands(sdm.sdbits & _SBITS_MASK):
                ll, mb = int(lev[n]), int(mbase[n])

                # Undo the level-r normalization for THIS subband (see emit()). Sanity: at
                # l == R (a full-band subband) the factor is 2^0 = 1 and d_full = (d << R) | e
                # is the honest r-bit index; at l == 0 the factor is 2^R, matching the fact
                # that a level-0 subband sums 2^(r-R) tree-freqs with normalization
                # 2^-((r-R)/2) while the level-r tile carries 2^-(r/2).
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
        # and the delay index is dc << L -- the dyadic block's bottom (see emit()), which is
        # l-INDEPENDENT. So 'idx' hoists out of the subband loop, which it cannot do above.
        Qtot = np.zeros((1 << (r - L), N, P, Ktot))
        dc_full = np.arange(1 << (r - L), dtype=np.int64) << L
        k0 = 0
        for sdm in sd_matrices.values():
            K = sdm.factor_rank
            Qg = sdm.Q_factor.reshape(sdm.D, P, K)
            dbits = sdm.sdbits >> _SBITS_WIDTH                 # COARSE: no bits below L
            idx = SparseTile._remap_d(dc_full, (1 << r) - 1, dbits)       # (2^(r-L),)

            # The 2^(R-l[n]) factor is still per subband, and coarse-graining does not touch
            # it -- it undoes the stored row's level-r normalization, which is orthogonal.
            for n in _iter_subbands(sdm.sdbits & _SBITS_MASK):
                Qtot[:, n, :, k0:k0+K] = (2.0 ** (R - int(lev[n]))) * Qg[idx]

            Wtot[sdm.freq_indices, k0:k0+K] = sdm.W_factor
            k0 += K

    assert k0 == Ktot, (k0, Ktot)

    # ---- the y_true lift, which is FINE for both branches and therefore runs over the OTHER
    # dict: sd_vectors, keyed by the FULL sdbits and holding untruncated row sums in the full
    # delay-bit basis. Its body is the fine Q lift's index arithmetic, over one vector per
    # group instead of one factor block. ACCUMULATE, not assign: one alpha receives
    # contributions from many groups.

    ytrue = np.zeros((ndm, M, P))
    for (sdbits, yg) in sd_vectors.items():
        dbits = sdbits >> _SBITS_WIDTH                          # FULL key
        yg = yg.reshape(1 << dbits.bit_count(), P)
        for n in _iter_subbands(sdbits & _SBITS_MASK):
            ll, mb = int(lev[n]), int(mbase[n])
            dfull = ((np.arange(ndm)[:, None] << R)
                     | (np.arange(1 << ll)[None, :] << (R - ll)))
            idx = SparseTile._remap_d(dfull, (1 << r) - 1, dbits)         # (ndm, 2^ll)
            ytrue[:, mb:mb + (1 << ll), :] += (2.0 ** (R - ll)) * yg[idx]

    dt = time.time() - t0
    if progress:
        atomic_print(f'  compute_detrender_free_base_map: done in {dt:.2f} seconds')

    return VarianceMap.from_factors(
        config, itree0, Qtot.reshape(nbeta, Ktot), Wtot, detrender=None, L=L,
        y_true=ytrue.reshape(nalpha), tree=tree, is_admissible=True,
        history=[dict(step='compute_detrender_free_base_map', time=dt, L=L, nalpha=nalpha,
                      nbeta=nbeta, Ktot=Ktot, Q_nbytes=nbytes,
                      n_matrices=stats['n_matrices'], n_entries=stats['n_entries'],
                      n_straddled=stats['n_straddled'], n_sliced=stats['n_sliced'],
                      Ftot=stats['Ftot'], epsilon=epsilon, eps_max=stats['eps_max'])])


####################################   the multimap   ####################################


def _validate_multi_map_L(L, npri, r0, R):
    """The legal range for 'L' is the DOWNSAMPLED trees', not the base tree's.

    Left unchecked this still fails, but loudly and confusingly: VarianceMap.__init__ raises
    "L=6 is out of range [R, r] = [2, 5]" from the CHILD's constructor, naming a rank the
    caller never asked for -- and only AFTER the base map has been built, which can be
    minutes and tens of GiB.
    """

    if L is None:
        return
    hi = r0 if (npri == 1) else (r0 - 1)      # a downsampled primary tree has rank r0 - 1
    if not (R <= int(L) <= hi):
        raise RuntimeError(
            f'compute_detrender_free_multi_map: L={L} is out of range [R, r] = [{R}, {hi}].'
            + (f' The upper bound is {r0-1} rather than the base tree\'s own {r0} because'
               f' this config has num_primary_trees={npri}, and a downsampled primary tree'
               f' has rank {r0-1}.' if (npri > 1) else ''))


def _check_restriction(base, tree, gamma):
    """The geometry Proposition 2 rests on, checked on the actual trees rather than assumed.

    Cheap (O(N) per tree, once per gamma) and worth it: two of the three facts are properties
    of how DedispersionTree derives its subbands rather than of config parameters, so
    DedispersionConfig::validate() -- which runs before any tree exists -- is the wrong place
    for them, and they are otherwise only covered by test_subband_property() in another file.
    """

    from ..pirate_pybind11 import DedispersionTree

    fs = tree.frequency_subbands
    M = int(base.tree.frequency_subbands.M)

    # (F2) plus Observation (a): time downsampling does not change WHICH bands are searched,
    # so the multiplet index is carried over UNCHANGED. m_index_mapping() raises unless every
    # band of the second tree is a band of the first AT THE SAME LEVEL -- which a set
    # comparison would not see -- so this is the containment check and the identity check in
    # one. Measured: it is the identity for every gamma of toy, chime_sb2 and chord_sb2_et.
    m_map = np.asarray(DedispersionTree.m_index_mapping(base.tree, tree), dtype=np.int64)
    if not np.array_equal(m_map, np.arange(M)):
        raise RuntimeError(f'compute_detrender_free_multi_map: primary tree {gamma} does not'
                           f' carry the base tree\'s multiplet index over unchanged'
                           f' (m_index_mapping is not the identity), so the slice below would'
                           f' select the wrong rows.')

    # (F1) and Observation (b): r_gamma = r_0 - 1 with R unchanged, hence D_gamma = D_0/2.
    # Observation (c): P_gamma <= P_0, which validate()'s non-increasing max_width rule gives.
    r_g, R_g, P_g = int(tree.total_rank()), int(fs.pf_rank), int(tree.nprofiles)
    if (R_g, r_g) != (base.pf_rank, base.tree_rank - 1):
        raise RuntimeError(f'compute_detrender_free_multi_map: primary tree {gamma} has'
                           f' (r, R) = ({r_g}, {R_g}), expected'
                           f' ({base.tree_rank - 1}, {base.pf_rank}) -- a downsampled primary'
                           ' tree drops one tree rank and no subband levels.')
    if P_g > base.nprofiles:
        raise RuntimeError(f'compute_detrender_free_multi_map: primary tree {gamma} has'
                           f' nprofiles={P_g}, more than the base tree\'s'
                           f' {base.nprofiles}. DedispersionConfig::validate() requires'
                           ' max_width to be non-increasing across primary trees, which is'
                           ' what makes the profile axes nest.')
    return P_g


def compute_detrender_free_multi_map(config, *, L=None, epsilon=None, max_bytes=None,
                                     progress=False, debug=False):
    """Analytic variance map of EVERY primary tree, as a VarianceMultiMap. No detrender.

    Computes the base tree once with compute_detrender_free_base_map() and SLICES it for
    every other primary tree, so the whole multimap costs one base-map computation plus some
    array copies -- no second tile pass, no second SVD. Compare compute_variance_multimap(),
    which runs one dedispersion pass per input channel.

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

    t0 = time.time()

    # VALIDATE BEFORE COMPUTING: the base map can take minutes and tens of GiB, and raising
    # afterwards for a reason knowable from the config alone is pure waste. make_tree() needs
    # no plan, no GPU and no map.
    npri = int(config.num_primary_trees)
    itree0 = int(config.dedispersion_tree_index(0, 0))
    tree0 = make_tree(config, itree0)
    r0, R = int(tree0.total_rank()), int(tree0.frequency_subbands.pf_rank)
    _validate_multi_map_L(L, npri, r0, R)

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

    Ps = [_check_restriction(base, make_tree(config, int(config.dedispersion_tree_index(g, 0))), g)
          for g in range(1, npri)]

    nbytes = 8 * K * (base.nbeta + sum((base.nbeta // 2) * P // P0 for P in Ps))
    atomic_print(f'compute_detrender_free_multi_map: {npri} primary trees, P='
                 f'{[P0] + Ps}, total Q {nbytes/(1<<30):.2f} GiB'
                 f' ({1 + 0.5*sum(P/P0 for P in Ps):.3f}x the base map)')

    maps = [base]
    for gamma in range(1, npri):
        itree = int(config.dedispersion_tree_index(gamma, 0))
        tree = make_tree(config, itree)
        Pg = Ps[gamma - 1]

        # THE NO-DETRENDER HYPOTHESIS IS THE WHOLE ARGUMENT, not a formality. Proposition 1
        # (early triggers) holds whatever the upstream chain; Proposition 2 does NOT, and the
        # appendix says so twice. Measured on _make_test_config(6, [2,2,1], num_primary_trees=2)
        # against the brute-force sweep: 4.9e-7 without a detrender, and 2.1 WITH one -- a
        # factor of two wrong, not a rounding difference. If a detrender path is ever added to
        # this module, this function must not be part of it.
        #
        # np.ascontiguousarray because the sliced view is non-contiguous whenever Pg < P0;
        # saying so marks the copy as intended rather than incidental.
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
