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
  3. Lift each group's factors into a global (nalpha, Ktot) / (nfreq, Ktot) pair.

THE COST IS ALL IN STEP 3. Steps 1 and 2 run at full CHORD scale (chord_sb2_et.yml, nalpha =
5.96e6) in 14 seconds and hold 17 MB of rows -- a scale at which PfAvarExact cannot be built
at all. The lift then materializes a dense (nalpha, Ktot) Q, which is 0.36 GiB on
toy.yml but 63.8 GiB at CHORD. compute_detrender_free_base_map() imposes no limit by
default, but it reports the size before allocating, and takes an optional 'max_bytes'
ceiling. The fix is a second, global SVD round that runs BEFORE the lift; the per-group
factors that round needs are what build_sd_matrices() returns.

TWO THINGS HERE ARE NON-OBVIOUS, and getting either wrong gives a wrong ANSWER rather than a
crash. Both are derived at the code that implements them.

  - Every row is stored in a common "level-r" normalization, and the lift undoes it per
    subband with a factor 2^(R-l). See emit() and the lift. Measured: omitting it moves
    apply() by 75% on toy.yml.
  - Half-aligned subbands whose footprint straddles the subband midpoint need their own
    branch. See the straddle loop. Measured: omitting it moves apply() by 1.6e-3 on toy.yml
    -- rare (1 row of 645) but not negligible.
"""

import time

import numpy as np

from .VarianceMap import VarianceMap, make_tree
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
      col_sums      (D*P,) float64, recorded by factorize() BEFORE truncation.
      is_factored, factor_rank, Q_factor (D*P, K), W_factor (F, K)
    """

    def __init__(self, sdbits, capacity, D, P):
        self.sdbits = int(sdbits)
        self.capacity = int(capacity)
        self.D, self.P = int(D), int(P)
        assert self.capacity >= 1 and self.D >= 1 and self.P >= 1

        self.F = 0
        self.freq_indices = np.zeros(self.capacity, dtype=np.int64)
        self.dense_matrix = np.zeros((self.capacity, self.D * self.P), dtype=np.float64)

        self.col_sums = None
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

        # BEFORE truncating: these are the true map's row sums, which the lift turns into
        # y_true. Taking them here (rather than at lift time) is also what would let
        # dense_matrix be released, if its memory were ever wanted back.
        self.col_sums = self.dense_matrix.sum(axis=0)

        eps = (self.default_epsilon(*self.dense_matrix.shape) if (epsilon is None)
               else float(epsilon))
        assert eps > 0.0, eps

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


def build_sd_matrices(config, *, epsilon=None, progress=False, debug=False):
    """Everything except the lift, for the base tree of 'config'.

    Returns (tree, itree0, sd_matrices, stats), where 'sd_matrices' is a dict keyed by sdbits
    and 'stats' carries n_entries, n_straddled, Ftot and Ktot. n_straddled is there because a
    test cannot see from the returned VarianceMap whether the straddle branch ran at all.

    'debug' turns on the O(F) and O(subbands) cross-checks: that no SdMatrix receives two rows
    from one input channel, and that every subband of an entry predicts the same dbits. Both
    are statements the shared-row pooling depends on, and both are too expensive to leave on
    at production scale.

    This is split out from compute_detrender_free_base_map() because the per-group factors,
    not the lifted Q, are what a second (global) SVD round and a coarse-graining pass would
    consume.
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

    sd_capacities = {}
    for (_, _, _, sdbits) in unstraddled_plan:
        sd_capacities[sdbits] = sd_capacities.get(sdbits, 0) + 1
    for (_, n, sdbits) in straddled_plan:
        # A straddling subband always gets a row to itself, so the two fields of its sdbits
        # are redundant with each other -- and checkable.
        assert (sdbits & _SBITS_MASK) == (1 << n), (sdbits, n)
        sd_capacities[sdbits] = sd_capacities.get(sdbits, 0) + 1

    sd_matrices = {}
    for (sdbits, capacity) in sd_capacities.items():
        dbits = sdbits >> _SBITS_WIDTH
        sd_matrices[sdbits] = SdMatrix(sdbits, capacity, 1 << dbits.bit_count(), P)

    n_entries = len(unstraddled_plan) + len(straddled_plan)
    if progress:
        atomic_print(f'  build_sd_matrices: r={r} R={R} N={N} M={M} P={P} nfreq={nfreq}:'
                     f' {n_entries} entries ({len(straddled_plan)} straddled) ->'
                     f' {len(sd_matrices)} SdMatrices,'
                     f' {n_entries/nfreq:.2f} rows per input channel')

    # ---- build the tiles and fill the rows. Two straight-line loops, one per plan.
    #
    # Two calls to make_tree_gridding_output() are in play and they are NOT the same call: the
    # planning loop above needed the UNCLIPPED footprint once per channel (cached in
    # 'footprint'), while these need a triple CLIPPED to the entry's own [lo, hi), which
    # cannot be hoisted. Its repeated searchsorted over cmap is negligible at these sizes.

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
        """

        # This is the assert that closes the loop between the planning pass's closed form and
        # the actual iteration, on both plans. Everything downstream -- the sizing, the shared
        # rows, the lift -- is built on that prediction being right.
        assert (tile.dbits << (r - klev)) == (sdbits >> _SBITS_WIDTH), \
            (ifreq, klev, tile.dbits, sdbits >> _SBITS_WIDTH)

        # scale**2 because variance is quadratic (this is what PfVariance.add_tile() does);
        # the [0] drops the length-1 nf axis. Omitting scale**2 is silently wrong wherever an
        # edge tile deferred its 1/sqrt(2).
        var = (tile.scale ** 2) * convolver.variance(tile.data, P)[0]      # (D, P)
        var *= 2.0 ** (-(r - klev))

        sdm = sd_matrices[sdbits]
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

    if progress:
        atomic_print(f'  build_sd_matrices: Ftot={Ftot} rows -> Ktot={Ktot}'
                     f' ({Ftot/max(Ktot,1):.1f}x compression)')

    stats = dict(n_entries=n_entries, n_straddled=len(straddled_plan), Ftot=Ftot, Ktot=Ktot,
                 n_matrices=len(sd_matrices))
    return tree, itree0, sd_matrices, stats


def compute_detrender_free_base_map(config, *, epsilon=None, max_bytes=None,
                                    progress=False, debug=False):
    """Analytic variance map of 'config''s base tree (ipri = et_level = 0), factored.

    No detrender, no coarse-graining, no DedispersionPlan and no GPU. See the module
    docstring for the algorithm and for the scaling limit.

    Parameters
    ----------
    epsilon : float or None
        Relative singular-value threshold, per group. None uses
        SdMatrix.default_epsilon().
    max_bytes : int or None
        Ceiling on the lifted Q, which is the only large allocation here (63.8 GiB at
        chord_sb2_et.yml scale). None means NO LIMIT; the size is reported before the
        allocation either way, so a run that is about to fail says why rather than being
        killed by the OOM reaper.
    debug : bool
        Turn on build_sd_matrices()'s O(F) cross-checks.

    Notes
    -----
    The returned map's y_true is the PRE-TRUNCATION row sum, i.e. it is the TRUE map's rather
    than the stored map's. row_sums() will therefore differ from it at the 1e-11 level, and
    get_distance() will return ~1 + 1e-11 rather than exactly 1. That is normal and is exactly
    what the distance machinery is built to measure -- it is not a bug to fix.

    is_admissible stays False (a truncated SVD guarantees nothing elementwise) and
    pinned_columns stays empty; a caller who needs a nonnegative W column for the LP machinery
    calls pin_column() afterwards.
    """

    t0 = time.time()

    tree, itree0, sd_matrices, stats = build_sd_matrices(
        config, epsilon=epsilon, progress=progress, debug=debug)

    fs = tree.frequency_subbands
    r, R = int(tree.total_rank()), int(fs.pf_rank)
    M, P = int(fs.M), int(tree.nprofiles)
    nfreq = int(config.get_total_nfreq())
    ndm = 1 << (r - R)
    nalpha = ndm * M * P
    Ktot = stats['Ktot']

    g = _subband_geometry(tree)
    lev, mbase = g['l'], g['mbase']

    nbytes = 8 * nalpha * Ktot
    # Unconditional, not under 'progress': this is the number that makes an OOM diagnosable.
    atomic_print(f'compute_detrender_free_base_map: lifting to Q ({nalpha} x {Ktot}),'
                 f' {nbytes/(1<<30):.2f} GiB')

    if (max_bytes is not None) and (nbytes > max_bytes):
        raise RuntimeError(f'compute_detrender_free_base_map: the lifted Q is'
                           f' {nbytes/(1<<30):.1f} GiB (nalpha={nalpha}, Ktot={Ktot}), over'
                           f' the caller-supplied max_bytes={max_bytes/(1<<30):.1f} GiB.')

    # A 4-d view of Q, so the per-subband writes are natural. Reshaped to (nalpha, Ktot) at
    # the end, which is a no-op: alpha = (d*M + m)*P + p is exactly this axis order.
    Qtot = np.zeros((ndm, M, P, Ktot))
    Wtot = np.zeros((nfreq, Ktot))
    ytrue = np.zeros((ndm, M, P))

    k0 = 0
    for sdm in sd_matrices.values():
        K = sdm.factor_rank
        Qg = sdm.Q_factor.reshape(sdm.D, P, K)
        yg = sdm.col_sums.reshape(sdm.D, P)
        dbits = sdm.sdbits >> _SBITS_WIDTH

        for n in _iter_subbands(sdm.sdbits & _SBITS_MASK):
            ll, mb = int(lev[n]), int(mbase[n])

            # Undo the level-r normalization for THIS subband (see emit()). Sanity: at l == R
            # (a full-band subband) the factor is 2^0 = 1 and d_full = (d << R) | e is the
            # honest r-bit index; at l == 0 the factor is 2^R, matching the fact that a
            # level-0 subband sums 2^(r-R) tree-freqs with normalization 2^-((r-R)/2) while
            # the level-r tile carries 2^-(r/2).
            fac = 2.0 ** (R - ll)

            # The virtual level-r delay index of multiplet (n, e) at coarse DM d. Its low R-l
            # bits are zero, which is why dbits is required to have none there.
            dfull = ((np.arange(ndm)[:, None] << R)
                     | (np.arange(1 << ll)[None, :] << (R - ll)))
            idx = SparseTile._remap_d(dfull, (1 << r) - 1, dbits)     # (ndm, 2^ll)

            # ASSIGN into Q (each group owns a disjoint column block, and within a group the
            # subbands own disjoint multiplet ranges), ACCUMULATE into y_true (one alpha
            # receives contributions from many groups).
            Qtot[:, mb:mb + (1 << ll), :, k0:k0+K] = fac * Qg[idx]
            ytrue[:, mb:mb + (1 << ll), :] += fac * yg[idx]

        Wtot[sdm.freq_indices, k0:k0+K] = sdm.W_factor
        k0 += K

    assert k0 == Ktot, (k0, Ktot)

    dt = time.time() - t0
    if progress:
        atomic_print(f'  compute_detrender_free_base_map: done in {dt:.2f} seconds')

    return VarianceMap.from_factors(
        config, itree0, Qtot.reshape(nalpha, Ktot), Wtot, detrender=None,
        y_true=ytrue.reshape(nalpha), tree=tree,
        history=[dict(step='compute_detrender_free_base_map', time=dt, nalpha=nalpha,
                      Ktot=Ktot, Q_nbytes=nbytes, n_matrices=stats['n_matrices'],
                      n_entries=stats['n_entries'], n_straddled=stats['n_straddled'],
                      Ftot=stats['Ftot'], epsilon=epsilon)])
