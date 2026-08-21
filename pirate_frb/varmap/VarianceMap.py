"""class VarianceMap: the variance map A of a single DedispersionTree.

See notes/variance_map.tex (section "The variance map") for what A is, and the class
docstring below for the representation.

Index conventions, which nothing else in this package restates
--------------------------------------------------------------
Per tree, with tree_rank = r, pf_rank = R, nmultiplets = M, nsubbands = N, nprofiles = P:

  - A multiplet 0 <= m < M is a (frequency subband, fine DM) pair. Subband n sits at level l
    and owns 2^l consecutive multiplets, so M = sum_l C_l 2^l and N = sum_l C_l, where
    C_l = subband_counts[l].
  - FINE index:   alpha = (d*M + m)*P + p,   0 <= d < 2^(r-R), 0 <= m < M, 0 <= p < P.
    So nalpha = 2^(r-R) * M * P, DM slowest and profile fastest. This is exactly the
    brute-force sweep's (2^(r-R), M, P, nfreq) array reshaped to (nalpha, nfreq), so no
    transpose is introduced anywhere.
  - COARSE index: beta = (dc*N + n)*P + p,   dc = d >> (L-R) and n = m_to_n[m], for a
    coarse-graining rank R <= L <= r. So nbeta = 2^(r-L) * N * P.

THERE ARE EXACTLY TWO SHAPES, AND NOTHING ELSE IS SUPPORTED: a map is fine, or coarse at L.
There is no general grouping descriptor, no explicit label array, and no way to merge along
any other axis. This is a measured restriction, not a shortcut -- merging frequency subbands
costs 489x to 3174x in D, merging peak-finding profiles 11x to 175x, and non-uniform DM cuts
up to 10.4x, while merging fine DM within a multiplet is nearly free (1.0x to 3.9x) and is
therefore done unconditionally. The coarse DM axis is the one that is neither: collapsing it
ENTIRELY costs 2.9x to 10.5x, but coarse-graining it by a practical factor costs only 1.0x to
1.55x with no detrender and 1.10x to 2.80x with one -- which is why L is a useful knob rather
than just a more expensive way of merging subbands. The two shapes are those results written into the type, where
nobody can set them wrong.

Two consequences worth stating explicitly:

  - Groups are NOT all the same size. Group beta contains 2^(L-R) * 2^l(n) fine rows, and l
    varies across subbands. D is a mean over alpha, so it weights groups by size.
  - "Coarse at L = R" is generally NOT the same as "fine". Setting L = R leaves the DM axis
    exact but still merges fine DM within each multiplet (M -> N). The two coincide exactly
    when M == N, i.e. when every subband sits at level 0 -- which is every single-subband
    config, so it is not an exotic corner.

Note that the production peak-finding weight array uses shape (ndm_wt, nt_wt, P, N) -- profile
OUTER, subband INNER -- which is the transpose of beta's last two axes. Exporting to
production weights is a separate piece of work, and the transpose belongs in that export path
rather than in this representation.
"""

import time

import numpy as np

from .distance import YTRUE_FLOOR, f, AdmissibilityResult, DistanceEstimate


####################################   geometry   ####################################

# The per-tree geometry comes straight from the C++ DedispersionTree, which is constructible
# with no DedispersionPlan and no GPU. That matters because a DedispersionPlan cannot be
# constructed without a CUDA device (its MegaRingbuf allocates page-locked host memory), and
# archived variance maps must be analyzable anywhere.
#
# Nothing here re-derives the geometry: a second implementation in python would drift from
# the C++, and the ordering conventions it encodes are what every archived map is interpreted
# through.
#
# DO NOT EXTRACT THE INDEX ARITHMETIC INTO A GEOMETRY MODULE OR VALUE OBJECT. It is the
# obvious tidy-up and it was considered and rejected: the fields such an object would carry
# (r, R, M, N, P, subband_counts, nalpha, m_to_n) are ALREADY members of VarianceMap, so it
# would be a second copy of data this class holds anyway, with the attendant risk of the two
# disagreeing; and alpha_to_beta_block() / group_sizes() are two lines of arithmetic each
# once the grouping is one integer, which belong next to the data they index.


def make_tree(config, itree):
    """The DedispersionTree for tree 'itree' of 'config'. Needs no plan and no GPU.

    Dcore is deliberately NOT taken from the cdd2 kernel registry: varmap never reads it, and
    requiring the registry would make an archived map unreadable on any build whose compiled
    cdd2 kernel set does not cover that config. So a tree obtained here (and hence 'Dcore' in
    any variance-map file) carries the placeholder pf.time_downsampling.
    """
    from ..pirate_pybind11 import DedispersionTree
    return DedispersionTree(config, int(itree), Dcore_from_cdd2_registry=False)


def _subband_tables(tree):
    """(m_to_n, n_level, n_to_mbase) for a tree, from its FrequencySubbands.

    m_to_n is the frequency subband of each multiplet: the only part of the alpha -> beta map
    that is not arithmetic. The other two are derived from it, and are what group_sizes() and
    group_members() index through.
    """

    fs = tree.frequency_subbands
    m_to_n = np.asarray(fs.m_to_n, dtype=np.int64)

    # Subband n owns 2^level(n) consecutive multiplets, so its level and multiplet base fall
    # out of a count and a cumulative sum.
    counts = np.bincount(m_to_n, minlength=int(fs.N))
    n_level = np.round(np.log2(counts)).astype(np.int64)
    n_to_mbase = np.concatenate([[0], np.cumsum(counts)[:-1]]).astype(np.int64)

    if not np.array_equal(counts, 1 << n_level):
        raise RuntimeError(f'VarianceMap: per-subband multiplet counts {counts} are not all'
                           ' powers of two, which the index convention requires')
    return m_to_n, n_level, n_to_mbase


def _readonly(a):
    """A read-only VIEW of 'a', without touching the writeability of 'a' itself.

    Flipping the flag in place would flip it on whatever the CALLER handed us -- np.asarray()
    returns its argument unchanged when the dtype already matches -- so this always takes a
    view first.

    Used for the stored arrays and for every accessor that may hand back a window into them.
    A VarianceMap is immutable, and that is what makes its flags trustworthy; without this,
    an in-place edit through such a window rewrites the map behind the flags' back, and for a
    memmapped read it rewrites the FILE. The two read paths used to differ on this -- a
    memmapped matrix refused the write, an eagerly-read one accepted it silently -- which is
    the worse of the two failures on the mode people develop against.
    """

    if not a.flags.writeable:
        return a          # idempotent, so replace() does not build a chain of views

    v = a.view()
    v.flags.writeable = False
    return v


####################################   class VarianceMap   ####################################


class VarianceMap:
    """The variance map A of a single DedispersionTree, in one of several representations.

    A is the matrix of notes/variance_map.tex (section "The variance map"):
    ``y_alpha = sum_F A[alpha,F]``, where v is the vector of per-channel input variances and y
    is the vector of peak-finding output variances. The module docstring pins down the index
    conventions for alpha and beta, and nothing else in this package restates them.

    An instance is IMMUTABLE. Every method that changes the map returns a new VarianceMap;
    this is what makes the flags below trustworthy, since an algorithm cannot invalidate
    ``is_admissible`` by writing into an array behind our back. (Consequence for callers:
    rows() and cols() may return VIEWS into the stored matrix -- do not modify what they
    hand back.)

    Three independent axes describe which representation this is: coarse-grained or not, and
    certified (``is_admissible``) or not, and dense or factored. In all cases the stored
    matrix has shape ``(nbeta, nfreq)``, where ``nbeta == nalpha`` when ``is_coarse_grained``
    is False -- so code that only wants "the matrix" can ignore the coarse-graining axis.

    Metadata (enough to re-run the brute-force sweep that produced this map):

    - ``config`` -- the DedispersionConfig. NOT a DedispersionPlan: plan construction calls
      cudaHostAlloc, so a plan cannot be built on a machine with no working GPU, and we want
      variance maps to be analyzable anywhere. Use plan() when a plan is genuinely needed.
    - ``detrender`` -- the Detrender2dParams used, or None for "no Detrender2d".
    - ``itree`` (int) -- index of this tree in the plan.
    - ``tree`` -- the ``DedispersionTree`` this map's geometry comes from, verbatim from the
      C++. Constructed with no plan and no GPU. Its ``Dcore`` is a placeholder and is not
      meaningful here -- see make_tree().

    Geometry (all read off ``tree`` at construction, with no plan and no GPU):

    - ``nfreq`` (int) -- number of input frequency channels.
    - ``tree_rank`` (int) -- r, the tree's total rank.
    - ``pf_rank`` (int) -- R, the tree's peak-finding rank.
    - ``nmultiplets`` (int) -- M, number of multiplets (subband x fine DM).
    - ``nsubbands`` (int) -- N, number of frequency subbands.
    - ``nprofiles`` (int) -- P, number of peak-finding profiles.
    - ``subband_counts`` (tuple) -- the tree's RESTRICTED subband vector, length R+1. This is
      what decomposes m into (subband, fine DM); the config's toplevel vector is a different
      thing.
    - ``gamma`` (int) -- the tree's time-downsampling exponent (its input samples are
      downsampled by ``2^gamma``).
    - ``early_trigger_level`` (int) -- the tree's early-trigger level.
    - ``nalpha`` (int) -- ``2^(r-R) * M * P``, the number of FINE output indices.
    - ``m_to_n`` (ndarray) -- (M,) int64, the frequency subband of each multiplet.

    Coarse-graining:

    - ``is_coarse_grained`` (bool) -- False means the rows ARE the fine outputs alpha; True
      means they are the coarse groups beta.
    - ``L`` (int) -- the coarse-graining rank, ``R <= L <= r``, or None when not
      coarse-grained. This is the whole of the grouping. The production name for ``2^L`` is
      ``wt_dm_downsampling``.
    - ``nbeta`` (int) -- ``2^(r-L) * N * P`` when coarse-grained, else ``nalpha``.

    What we know about this A:

    - ``is_admissible`` (bool) -- True iff we GUARANTEE ``A_stored >= A_true`` elementwise, in
      the lifted sense ``A_stored[beta(alpha),F] >= A_true[alpha,F]``. False means NO
      GUARANTEE; it does not mean "known to violate". A truncated SVD (usually inadmissible)
      and a map that simply has not been checked are both False, because nothing acts
      differently on the two -- the measured fact lives in the result of
      measure_admissibility(), which reports ``max_r``.

    Matrix storage:

    - ``is_factored`` (bool) -- False for a dense ``(nbeta, nfreq)`` matrix ``A``, True for
      ``A = Q @ mid @ W.T``. A dense fine map is 64 GiB at CHIME tree-0 and 1.2 TiB at CHORD,
      so nothing here may materialize a temporary of its size, and every consumer walks it in
      row blocks (see rows() and default_block_rows()).
    - ``A`` (ndarray) -- the ``(nbeta, nfreq)`` matrix when ``is_factored`` is False, else
      None. May be a numpy memmap, which the row-blocked access makes transparent.
    - ``Q`` (ndarray) -- ``(nbeta, K)`` when factored, else None. Sign-free, and ALWAYS DENSE:
      there is deliberately no second storage layout, and a Q whose support has been capped is
      stored dense with zeros in it. Sizing, so the decision can be revisited on evidence: a
      dense Q is 8 MiB at the CHIME tree-0 point (nbeta = 8192, K = 128), and W at
      ``(nfreq, K)`` is the same order. Revisit only if a smaller L at CHORD makes nbeta large
      enough to matter.
    - ``mid`` (ndarray) -- ``(K, K)`` when factored, else None; the identity when unused. It
      lets Q and W BOTH be semiorthogonal at once (an SVD is ``U diag(s) V^T``, and folding
      s into either factor destroys one of the two properties), and makes rescaling O(K^2)
      instead of O(nbeta*K).
    - ``W`` (ndarray) -- ``(nfreq, K)`` when factored, else None. Sign-free; its columns are
      frequency "atoms".
    - ``factor_rank`` (int) -- K, the number of columns of Q and W; None when dense.
    - ``Q_is_semiorthogonal`` / ``W_is_semiorthogonal`` (bool) -- what the code that built the
      factors knows, not a measurement. svd() and reorthogonalize() are the only things that
      set one True, truncate() is the only thing that reads one, and everywhere else
      replace()'s conservative False applies.
    - ``pinned_columns`` (ndarray) -- int64 column indices of W held fixed by the steps; empty
      by default, and None when dense. A pin exists to guarantee that W has a NONNEGATIVE
      column, which the additive repair, the one-hot seed and the LP's feasibility certificate
      all need. pin_column() checks the column it is given, and every method that drops or
      reorders columns remaps the set rather than copying it.

    ONLY THE STRUCTURE OF A FACTORIZATION IS ENFORCED AT CONSTRUCTION -- shapes, a consistent
    K, dtypes, and column indices in range. Whether ``Q^T Q`` really is the identity, whether
    ``mid`` is diagonal, whether a pinned column is nonnegative: none of that is re-checked
    here, because the methods that establish those invariants are the cheap place to know
    them, and an O(nbeta*K^2) verification on every construction would be paid by every step
    of an alternation. So a hand-built map can claim anything; one built through the methods
    below cannot.

    How this map was produced:

    - ``history`` (tuple) -- one record per transformation, appended to (never rewritten) as
      the map passes through coarse_grain() and the rest. Each record carries at least the
      step name, the wall time, and whatever the step measured. This is why there is no
      schedule driver: the per-step log a driver would have returned is carried by the map
      itself, so it survives being written to a file and read back on another machine.
      Free-form on purpose -- new algorithms will want to record things nobody has thought
      of, and the reader must never need a schema change to load a file.

    The true row sums:

    - ``y_true`` (ndarray) -- (nalpha,) float64, or None. IMPORTANT AND EASY TO GET WRONG:
      this is always ``y_true[alpha] = sum_F A_true[alpha,F]`` -- derived from the TRUE map,
      at FINE granularity -- regardless of this instance's flags, and it is carried along
      unchanged by every transformation. It is kept because it is the only part of A_true
      that survives coarse-graining and low-rank approximation, and D cannot be computed
      without it. None means "unavailable", and get_distance() raises rather than guessing.

    Note that this class is deliberately NOT built on the ``PfAvar*`` / ``TmpVmap*`` classes
    of pirate_frb.slow_avar, which were a first pass at the same idea.
    """

    # Rows per block for the 1-d walks over alpha (y_true, row labels, per-row distances).
    # These are length-nalpha vectors, not matrices, so the budget is much larger than
    # default_block_rows()'.
    _ALPHA_BLOCK = 1 << 20

    # How many entries of 'worst_rows' measure_admissibility() reports.
    _N_WORST_ROWS = 16

    # The factorization rank above which svd()'s unit-sum SHAPE matrix is the better W-matrix,
    # and below which the raw matrix is. A measured crossover, not a round number.
    _SHAPE_NORMALIZE_RANK = 32

    # The randomized SVD's sampling defaults, and THEY ARE NOT THE TEXTBOOK ONES. The textbook
    # setting (one power iteration, ten extra samples) is chosen to get the RESIDUAL right, and
    # it does: it lands within 2.5% of the optimal rank-K residual and its singular values agree
    # to 0.4%. The basis it produces nevertheless delivers a D 1.32x-1.45x worse than the exact
    # SVD's, because D is paid on each group's WORST channel while a residual is an RMS over the
    # whole matrix. So THE RESIDUAL IS USELESS AS A STOPPING CRITERION HERE -- it saturates at
    # 1.000x optimal several settings before D does, and anyone tuning this against it will
    # conclude, wrongly, that the textbook defaults are fine.
    #
    # Measured on r2_nf3200 L6 (12800 x 3200), D as a ratio to the exact SVD's. Power
    # iterations at ten extra samples, which is how the textbook spends its budget:
    #
    #        power iterations:     1       2       3       4
    #             K = 16        1.4014  1.0613  1.0045  0.9999
    #             K = 32        1.2554          1.0108
    #             K = 128       1.3249  1.0929  1.0389
    #
    # and then OVERSAMPLING at two power iterations, which is how this spends it:
    #
    #        extra samples:       16      24      32      48      64      96     128
    #             K = 16       1.0312          1.0017  1.0002  1.0000
    #             K = 32       1.0272  1.0130  1.0024  1.0005  1.0001
    #             K = 128                      1.0276  1.0093  1.0020  1.0002  1.0000
    #
    # Two things fall out. OVERSAMPLING IS THE KNOB THAT PAYS: 48 extra samples at two
    # iterations beats four iterations at ten, in five passes rather than nine -- which is the
    # right ordering for production, since a pass over the CHORD map is 344 GiB of reads while
    # extra samples only widen a GEMM. And THE REQUIRED OVERSAMPLE GROWS SUBLINEARLY IN K: 48
    # suffices at both 16 and 32, and 96 at 128, so a rule proportional to K overpays at high
    # rank exactly where a pass is most expensive.
    #
    # Hence max(48, K): every entry it selects is measured at 1.0005x or better, and at K = 128
    # it asks for ell = 256 rather than the 384 a 2K rule would. All of it is one map, so treat
    # the ROW SHAPE as the finding and the constants as fitted to it.
    _SVD_OVERSAMPLE_MULT = 1        # oversample = max(_SVD_OVERSAMPLE_MIN, MULT * factor_rank)
    _SVD_OVERSAMPLE_MIN = 48
    _SVD_POWER_ITERS = 2


    def __init__(self, config, itree, detrender=None, *, A=None,
                 Q=None, mid=None, W=None, y_true=None,
                 L=None, is_admissible=False, pinned_columns=None,
                 Q_is_semiorthogonal=False, W_is_semiorthogonal=False,
                 history=None, tree=None):
        """Low-level constructor: validates every shape against the geometry of tree 'itree'
        of 'config', and every flag against the arrays.

        Prefer from_dense(), or the classmethods of VarianceMultiMap, over calling this
        directly. Exactly one of {A} and {Q, W} must be given.

        'tree' is the DedispersionTree supplying the geometry. It defaults to
        make_tree(config, itree); pass it explicitly to reuse one across transformations, or
        to supply a tree read from a file rather than re-derived from the config.
        """

        factored = (Q is not None) or (W is not None) or (mid is not None)

        if factored and (A is not None):
            raise RuntimeError('VarianceMap: got both a dense matrix A and factors; exactly'
                               ' one of {A} and {Q, W} may be given.')
        if factored and ((Q is None) or (W is None)):
            missing = 'Q' if (Q is None) else 'W'
            raise RuntimeError(f'VarianceMap: the factored representation needs BOTH Q and W'
                               f' ({missing} is missing). Only "mid" is optional, and it'
                               ' defaults to the identity.')
        if (not factored) and (A is None):
            raise RuntimeError('VarianceMap: expected either a dense matrix A, or factors'
                               ' Q and W')
        if (not factored) and (pinned_columns is not None):
            raise RuntimeError('VarianceMap: pinned_columns indexes the columns of W, so it'
                               ' is meaningless for a dense map.')
        if (not factored) and (Q_is_semiorthogonal or W_is_semiorthogonal):
            raise RuntimeError('VarianceMap: the semiorthogonality flags describe Q and W,'
                               ' so they are meaningless for a dense map.')

        itree = int(itree)
        ntrees = int(config.num_dedispersion_trees)
        if not (0 <= itree < ntrees):
            raise RuntimeError(f'VarianceMap: itree={itree} is out of range for this config,'
                               f' which has {ntrees} dedispersion trees')

        tree = make_tree(config, itree) if (tree is None) else tree
        fs = tree.frequency_subbands
        m_to_n, n_level, n_to_mbase = _subband_tables(tree)

        set_ = lambda k, v: object.__setattr__(self, k, v)
        set_('config', config)
        set_('itree', itree)
        set_('detrender', detrender)
        set_('tree', tree)

        set_('nfreq', int(config.get_total_nfreq()))
        set_('tree_rank', int(tree.total_rank()))
        set_('pf_rank', int(fs.pf_rank))
        set_('nmultiplets', int(fs.M))
        set_('nsubbands', int(fs.N))
        set_('nprofiles', int(tree.nprofiles))
        set_('gamma', int(tree.primary_tree_index))
        set_('early_trigger_level', int(tree.early_trigger_level))
        set_('subband_counts', tuple(int(c) for c in fs.subband_counts))
        # Read-only: a silent edit to any of these reinterprets every index computation.
        set_('m_to_n', _readonly(m_to_n))
        set_('_n_level', _readonly(n_level))
        set_('_n_to_mbase', _readonly(n_to_mbase))

        r, R = self.tree_rank, self.pf_rank

        # The whole index convention rests on ndm_out == 2^(r-R), which holds iff the config
        # leaves dm_downsampling at 0 (auto-filled to 2^R). Any other value gives a map whose
        # rows are not the alpha of the module docstring, so it is refused up front rather
        # than silently reinterpreted.
        if int(tree.ndm_out) != (1 << (r - R)):
            raise RuntimeError(
                f'VarianceMap: tree {itree} has ndm_out={tree.ndm_out} != 2^(r-R)'
                f' = {1 << (r-R)} (r={r}, R={R}). This means the config sets'
                " 'dm_downsampling' explicitly; leave it at 0 (auto) for variance maps.")

        set_('nalpha', (1 << (r - R)) * self.nmultiplets * self.nprofiles)

        # ---- coarse-graining ----

        if L is not None:
            L = int(L)
            if not (R <= L <= r):
                raise RuntimeError(f'VarianceMap: L={L} is out of range [R, r] = [{R}, {r}]')
            nbeta = (1 << (r - L)) * self.nsubbands * self.nprofiles
        else:
            nbeta = self.nalpha

        set_('L', L)
        set_('nbeta', nbeta)

        # ---- the matrix ----

        def _check_dtype(x, name):
            if x.dtype not in (np.float32, np.float64):
                raise RuntimeError(f'VarianceMap: {name} has dtype {x.dtype}, expected'
                                   ' float32/float64')

        if not factored:
            A = A if hasattr(A, 'shape') else np.asarray(A)
            if A.ndim != 2:
                raise RuntimeError(f'VarianceMap: expected a 2-d matrix A, got shape'
                                   f' {A.shape}')
            if A.shape != (nbeta, self.nfreq):
                expect = 'coarse' if (L is not None) else 'fine'
                raise RuntimeError(f'VarianceMap: A has shape {A.shape}, expected'
                                   f' ({nbeta}, {self.nfreq}) for a {expect} map'
                                   + (f' at L={L}' if (L is not None) else ''))
            _check_dtype(A, 'A')
            set_('A', _readonly(A) if isinstance(A, np.ndarray) else A)
            for k in ('Q', 'mid', 'W', 'pinned_columns'):
                set_(k, None)
            set_('Q_is_semiorthogonal', False)
            set_('W_is_semiorthogonal', False)
        else:
            # ONLY STRUCTURE IS CHECKED HERE: shapes, a consistent K, dtypes, and column
            # indices in range. Whether Q^T Q is actually the identity, whether 'mid' is
            # diagonal, whether a pinned column is nonnegative -- none of that is asserted.
            # Those invariants belong to the steps that establish them, and a flag this
            # class cannot check is one it should carry rather than enforce.
            Q = Q if hasattr(Q, 'shape') else np.asarray(Q)
            W = W if hasattr(W, 'shape') else np.asarray(W)
            if (Q.ndim != 2) or (W.ndim != 2):
                raise RuntimeError(f'VarianceMap: expected 2-d factors, got Q{Q.shape} and'
                                   f' W{W.shape}')
            if Q.shape[1] != W.shape[1]:
                raise RuntimeError(f'VarianceMap: Q has {Q.shape[1]} columns and W has'
                                   f' {W.shape[1]}; both are the factorization rank K, so'
                                   ' they must agree')

            K = int(Q.shape[1])
            if Q.shape[0] != nbeta:
                expect = 'coarse' if (L is not None) else 'fine'
                raise RuntimeError(f'VarianceMap: Q has shape {Q.shape}, expected'
                                   f' ({nbeta}, {K}) for a {expect} map'
                                   + (f' at L={L}' if (L is not None) else ''))
            if W.shape[0] != self.nfreq:
                raise RuntimeError(f'VarianceMap: W has shape {W.shape}, expected'
                                   f' ({self.nfreq}, {K}) -- its rows are frequency channels')

            mid = np.eye(K) if (mid is None) else (mid if hasattr(mid, 'shape')
                                                  else np.asarray(mid))
            if mid.shape != (K, K):
                raise RuntimeError(f'VarianceMap: mid has shape {mid.shape}, expected'
                                   f' ({K}, {K})')

            _check_dtype(Q, 'Q')
            _check_dtype(mid, 'mid')
            _check_dtype(W, 'W')

            pc = (np.zeros(0, dtype=np.int64) if (pinned_columns is None)
                  else np.asarray(pinned_columns, dtype=np.int64).reshape(-1))
            if pc.size and ((pc.min() < 0) or (pc.max() >= K)):
                raise RuntimeError(f'VarianceMap: pinned_columns {list(pc)} indexes the'
                                   f' columns of W, so every entry must lie in [0, {K})')
            if np.unique(pc).size != pc.size:
                raise RuntimeError(f'VarianceMap: pinned_columns {list(pc)} contains'
                                   ' duplicates')

            set_('A', None)
            set_('Q', _readonly(Q) if isinstance(Q, np.ndarray) else Q)
            set_('mid', _readonly(mid) if isinstance(mid, np.ndarray) else mid)
            set_('W', _readonly(W) if isinstance(W, np.ndarray) else W)
            set_('pinned_columns', _readonly(pc))
            set_('Q_is_semiorthogonal', bool(Q_is_semiorthogonal))
            set_('W_is_semiorthogonal', bool(W_is_semiorthogonal))

        # ---- y_true ----

        if y_true is not None:
            y_true = np.asarray(y_true, dtype=np.float64)
            if y_true.shape != (self.nalpha,):
                raise RuntimeError(f'VarianceMap: y_true has shape {y_true.shape}, expected'
                                   f' ({self.nalpha},) -- it is always at FINE granularity,'
                                   ' whether or not the map is coarse-grained')
        set_('y_true', _readonly(y_true) if (y_true is not None) else None)

        set_('is_admissible', bool(is_admissible))
        set_('history', tuple(history) if (history is not None) else ())

        set_('_row_sums', None)
        set_('_plan', None)


    def __setattr__(self, k, v):
        raise AttributeError(f'VarianceMap is immutable (tried to set {k!r}); use replace()')


    def __repr__(self):
        what = f'coarse at L={self.L}' if self.is_coarse_grained else 'fine'
        adm = 'admissible' if self.is_admissible else 'uncertified'
        rep = f'factored K={self.factor_rank}' if self.is_factored else 'dense'
        return (f'VarianceMap(itree={self.itree}, r={self.tree_rank}, R={self.pf_rank},'
                f' M={self.nmultiplets}, N={self.nsubbands}, P={self.nprofiles},'
                f' shape={self.shape}, {what}, {rep}, {adm})')


    # ---------------- construction ----------------

    @classmethod
    def from_dense(cls, config, itree, A, *, detrender=None, y_true=None,
                   L=None, is_admissible=False, history=None):
        """Wrap an existing (nbeta, nfreq) array.

        Parameters
        ----------
        y_true : ndarray or 'row_sums' or None
            Normally passed explicitly. The sentinel ``'row_sums'`` derives it from A
            instead, which is ONLY correct when A is the true FINE map, and is therefore an
            assertion the caller is making rather than something this function can check.
            It raises on a coarse-grained map, where the row sums of Abar are not y_true (a
            max-envelope's row sum overestimates every member's).
        """

        if isinstance(y_true, str):
            if y_true != 'row_sums':
                raise RuntimeError(f"VarianceMap.from_dense: y_true={y_true!r} is not a"
                                   " recognized sentinel (did you mean 'row_sums'?)")
            if L is not None:
                raise RuntimeError(
                    "VarianceMap.from_dense: y_true='row_sums' is not valid for a"
                    ' coarse-grained map. The row sums of Abar are not y_true -- a'
                    " max-envelope's row sum overestimates every member's. Pass the fine"
                    ' y_true explicitly (it is carried unchanged by coarse_grain()).')
            y_true = np.asarray(A, dtype=np.float64).sum(axis=1)

        return cls(config, itree, detrender, A=A, y_true=y_true, L=L,
                   is_admissible=is_admissible, history=history)


    @classmethod
    def from_factors(cls, config, itree, Q, W, *, mid=None, detrender=None, **kwargs):
        """Wrap an existing factorization ``A = Q @ mid @ W.T``.

        'mid' defaults to the identity. It exists so that Q and W can BOTH be semiorthogonal
        at once (an SVD is ``U diag(s) V^T``, and folding s into either factor destroys one
        of the two properties), and so that rescaling costs O(K^2) rather than O(nbeta*K).

        Only structure is validated -- see the constructor. In particular this does NOT
        check the semiorthogonality flags or anything about the pinned columns; it carries
        what it is told.
        """
        return cls(config, itree, detrender, Q=Q, W=W, mid=mid, **kwargs)


    def replace(self, *, history_record=None, **kwargs):
        """Return a copy with the named members replaced, re-validating.

        This is how every transformation is implemented, and how a caller makes a one-off
        variant without reaching into the object.

        It does NOT invent a history record: the transformations pass their own via
        ``history_record``, and a caller doing something they do not cover should pass one
        too rather than leaving a silent gap. A bare replace() copies history across
        unchanged, which is right for a relabelling and wrong for a change to the matrix --
        so if you change A (or Q, mid, W), say so.

        PASSING A MATRIX SWITCHES THE REPRESENTATION. ``replace(A=...)`` on a factored map
        drops the factors, which is what makes coarse_grain() work on one: a max-envelope is
        nonlinear, so it comes back dense. ``replace(Q=...)`` on a dense map does the
        reverse. The alternative -- carrying both and erroring -- would make every such
        transformation spell out what to clear.

        Replacing Q or W CLEARS the matching semiorthogonality flag unless the call restates
        it. That is not a semantic check (this class never verifies the flags); it is the
        same conservative move inflated() makes with is_admissible, since False means "no
        guarantee" and is always safe to assert.
        """

        # The tree is carried across rather than rebuilt: it is a pure function of
        # (config, itree), and replace() is on the hot path of an alternation schedule.
        args = dict(config=self.config, itree=self.itree, detrender=self.detrender,
                    y_true=self.y_true, L=self.L,
                    is_admissible=self.is_admissible, history=self.history, tree=self.tree)

        rep = dict(A=self.A, Q=self.Q, mid=self.mid, W=self.W,
                   pinned_columns=self.pinned_columns,
                   Q_is_semiorthogonal=self.Q_is_semiorthogonal,
                   W_is_semiorthogonal=self.W_is_semiorthogonal)

        if 'A' in kwargs:
            rep = dict(A=None, Q=None, mid=None, W=None, pinned_columns=None,
                       Q_is_semiorthogonal=False, W_is_semiorthogonal=False)
        elif any(k in kwargs for k in ('Q', 'mid', 'W')):
            rep['A'] = None
            if ('Q' in kwargs) and ('Q_is_semiorthogonal' not in kwargs):
                rep['Q_is_semiorthogonal'] = False
            if ('W' in kwargs) and ('W_is_semiorthogonal' not in kwargs):
                rep['W_is_semiorthogonal'] = False

        args.update(rep)
        args.update(kwargs)

        if history_record is not None:
            args['history'] = tuple(args['history']) + (dict(history_record),)

        config = args.pop('config')
        itree = args.pop('itree')
        detrender = args.pop('detrender')
        return type(self)(config, itree, detrender, **args)


    # ---------------- basic accessors ----------------

    @property
    def is_coarse_grained(self):
        """True iff the rows are the coarse groups beta rather than the fine outputs alpha.

        Equivalent to ``self.L is not None``: the two are one fact, and L is the whole of the
        grouping. Neither is derivable from ``nbeta``, though -- ``nbeta == nalpha`` also
        holds for a coarse map with M == N -- so do not infer either from the shape.
        """
        return self.L is not None

    @property
    def is_factored(self):
        """True iff the matrix is stored as ``A = Q @ mid @ W.T`` rather than densely.

        Equivalent to ``self.Q is not None``: the two are one fact, and deriving it means
        they cannot disagree.
        """
        return self.Q is not None

    @property
    def factor_rank(self):
        """K, the number of columns of Q and W -- None for a dense map.

        This is the rank of the FACTORIZATION, which is an upper bound on the numerical rank
        of the product and deliberately not the same thing. It is what gets reported, because
        an approximation is a factorization we intend to evaluate as one, so the column count
        is the honest cost.
        """
        return None if (self.Q is None) else int(self.Q.shape[1])

    @property
    def shape(self):
        """(nbeta, nfreq)."""
        return (self.nbeta, self.nfreq)

    @property
    def nscored(self):
        """Number of outputs that contribute to get_distance(): the count of alpha with
        ``y_true[alpha] >= YTRUE_FLOOR``.

        A property of the problem instance, not of any particular approximation, since
        y_true is carried unchanged by every transformation -- which is why reporting it
        per-distance would be misleading.
        """
        if self.y_true is None:
            raise RuntimeError('VarianceMap.nscored: y_true is unavailable for this map')
        return int(np.count_nonzero(self.y_true >= YTRUE_FLOOR))


    def rows(self, start, stop):
        """Return rows [start, stop) of the stored (nbeta, nfreq) matrix, as a dense float64
        array.

        This is THE accessor: dense and factored maps differ only here, and every consumer
        goes through it, so that nothing ever needs the dense product of a factored map.
        Blocked callers should size their blocks with default_block_rows().

        The result is READ-ONLY. For a dense map it is usually a view into the stored matrix
        rather than a copy of it; for a factored map it is freshly computed. Copy it if you
        need to modify it -- see _readonly().
        """

        start, stop = int(start), int(stop)
        if not self.is_factored:
            return _readonly(np.asarray(self.A[start:stop], dtype=np.float64))

        Q = np.asarray(self.Q[start:stop], dtype=np.float64)
        return _readonly((Q @ np.asarray(self.mid, dtype=np.float64))
                         @ np.asarray(self.W, dtype=np.float64).T)


    def cols(self, start, stop):
        """Return columns [start, stop) of the stored matrix, as a dense (nbeta, ncol) float64
        array.

        Note the cost asymmetry runs the OPPOSITE way from rows(): a dense map is stored
        C-order (nbeta, nfreq), so a column block is a strided gather. Size column blocks with
        default_block_cols(), which accounts for that.

        The result is READ-ONLY. For a dense map it is usually a view into the stored matrix
        rather than a copy of it; for a factored map it is freshly computed. Copy it if you
        need to modify it -- see _readonly().
        """

        start, stop = int(start), int(stop)
        if not self.is_factored:
            return _readonly(np.asarray(self.A[:, start:stop], dtype=np.float64))

        QM = np.asarray(self.Q, dtype=np.float64) @ np.asarray(self.mid, dtype=np.float64)
        return _readonly(QM @ np.asarray(self.W[start:stop], dtype=np.float64).T)


    def dense(self, *, force=False, max_bytes=1 << 31):
        """The full (nbeta, nfreq) matrix, as float64, READ-ONLY (it is rows() over the whole
        matrix, so the same view-not-copy caveat applies).

        Raises if it would exceed 'max_bytes', since forming it is almost always a mistake at
        production scale; force=True overrides. For tests and small maps.
        """
        nbytes = 8 * self.nbeta * self.nfreq
        if (nbytes > max_bytes) and not force:
            raise RuntimeError(f'VarianceMap.dense(): the matrix is {nbytes/(1<<30):.1f} GiB,'
                               f' over the {max_bytes/(1<<30):.1f} GiB limit. Walk it in row'
                               ' blocks with rows(), or pass force=True.')
        return self.rows(0, self.nbeta)


    def default_block_rows(self, target_bytes=32 << 20):
        """Rows per block for a blocked walk over rows(), sized to 'target_bytes'.

        Purely a memory budget. A row block of a dense map is contiguous, so nothing else
        constrains it.
        """
        return max(1, min(self.nbeta, int(target_bytes) // (8 * self.nfreq)))


    def default_block_cols(self, target_bytes=32 << 20):
        """Columns per block for a blocked walk over cols().

        NOT the mirror image of default_block_rows(), because a column block of a dense
        C-order map has a LOWER bound as well as an upper one: each row contributes a
        contiguous run of ``ncol * 8`` bytes at stride ``nfreq * 8``, so a block narrower
        than a 4 KiB page reads whole pages to use a fraction of each. The returned value is
        therefore ``max(memory budget, 512)`` columns, capped at nfreq.

        At our geometries the two constraints happen to agree; at a large nbeta the floor
        wins and the block exceeds target_bytes, which is the right trade -- it is still only
        a few hundred MiB, and the alternative is reading the file many times over.

        For a FACTORED map neither consideration applies -- columns are computed, not read --
        so the memory budget alone is used, with no floor.
        """
        budget = int(target_bytes) // (8 * self.nbeta)
        if self.is_factored:
            return max(1, min(self.nfreq, budget))
        return max(1, min(self.nfreq, max(budget, 512)))


    def apply(self, freq_variances):
        """Evaluate ``y = A v`` for a length-nfreq input vector, returning a length-nbeta
        array. This is the operation production actually performs."""

        v = np.asarray(freq_variances, dtype=np.float64)
        if v.shape != (self.nfreq,):
            raise RuntimeError(f'VarianceMap.apply: expected a length-{self.nfreq} vector,'
                               f' got shape {v.shape}')

        if self.is_factored:
            # Q @ (mid @ (W.T @ v)) -- three small products, and A is never formed. This is
            # the operation production actually performs, and the reason rank is the figure
            # of merit rather than element count.
            return (np.asarray(self.Q, dtype=np.float64)
                    @ (np.asarray(self.mid, dtype=np.float64)
                       @ (np.asarray(self.W, dtype=np.float64).T @ v)))

        out = np.empty(self.nbeta)
        nb = self.default_block_rows()
        for start in range(0, self.nbeta, nb):
            stop = min(start + nb, self.nbeta)
            out[start:stop] = self.rows(start, stop) @ v
        return out


    def row_sums(self):
        """The length-nbeta vector ``sum_F A[.,F]``, i.e. ``apply(ones(nfreq))``. Cached.

        This is y_approx, and it is the ONLY thing get_distance() needs from the matrix --
        which is why get_distance() takes no reference argument.

        THE REPRESENTATION IS A SOURCE OF LAST-BIT DIFFERENCE HERE, and it is worth knowing
        before chasing one. A factored map contracts K vectors; a dense map sums each row over
        nfreq. Those group their additions differently, so a factored map and the dense map it
        stands for return row sums differing by ~1e-13 relative -- and D inherits that. It is
        the same order as the blocked-versus-unblocked difference the distance already has, so
        D is reproducible to ~1e-13 and not bitwise, across representations as well as across
        block sizes.
        """
        if self._row_sums is None:
            if self.is_factored:
                # One K-vector contraction, versus a pass over the whole matrix.
                out = np.array(self.apply(np.ones(self.nfreq)))
            else:
                out = np.empty(self.nbeta)
                nb = self.default_block_rows()
                for start in range(0, self.nbeta, nb):
                    stop = min(start + nb, self.nbeta)
                    out[start:stop] = self.rows(start, stop).sum(axis=1)
            out.flags.writeable = False
            object.__setattr__(self, '_row_sums', out)
        return self._row_sums


    def apply_cost(self):
        """Multiply-adds needed by apply(): ``nbeta * nfreq`` dense, or
        ``factor_rank * nfreq + factor_rank^2 + nnz(Q)`` factored.

        DESCRIPTIVE ONLY. The agreed figure of merit is RANK, not apply cost -- that was a
        deliberate decision, and it is why nothing here trades D away for a cheaper apply.
        Report this alongside D; do not optimize against it without saying so.

        NOT FREE, and not cached: the factored branch counts the nonzeros of Q, which is an
        O(nbeta * K) pass -- 2e8 elements at CHORD's L = 4 with K = 128. That is nothing once
        per cell of a results table, which is what this is for, and it is real money inside a
        loop. Hoist it if you find yourself calling it per iteration.
        """
        if not self.is_factored:
            return self.nbeta * self.nfreq

        K = self.factor_rank
        return K * self.nfreq + K * K + int(np.count_nonzero(self.Q))


    def nbytes(self):
        """Bytes of matrix storage (excluding y_true)."""
        if not self.is_factored:
            return int(self.A.dtype.itemsize) * self.nbeta * self.nfreq
        return int(self.Q.nbytes) + int(self.mid.nbytes) + int(self.W.nbytes)


    # ---------------- geometry helpers ----------------

    def plan(self, **kwargs):
        """Build (and cache) a DedispersionPlan from self.config.

        Needs a working CUDA device; everything else in this class does not.
        """
        if self._plan is None:
            from ..pirate_pybind11 import DedispersionPlan
            object.__setattr__(self, '_plan', DedispersionPlan(self.config, **kwargs))
        return self._plan


    def alpha_to_beta_block(self, start, stop, L=None):
        """The group index beta for each FINE index alpha in [start, stop), as an int array of
        length (stop-start).

        Two lines of arithmetic on the index conventions in the module docstring, plus the
        small m_to_n table. The blockwise form is what the streaming sweep needs, since it
        sees one column at a time and never has all of alpha in hand.

        'L' defaults to self.L. On a map with no coarse-graining the default is the IDENTITY,
        since the rows are the fine outputs themselves; pass an explicit L to compute coarse
        labels for a fine map.
        """

        start, stop = int(start), int(stop)
        L = self.L if (L is None) else int(L)

        if L is None:
            return np.arange(start, stop, dtype=np.int64)
        if not (self.pf_rank <= L <= self.tree_rank):
            raise RuntimeError(f'VarianceMap.alpha_to_beta_block: L={L} is out of range'
                               f' [R, r] = [{self.pf_rank}, {self.tree_rank}]')

        M, N, P = self.nmultiplets, self.nsubbands, self.nprofiles
        alpha = np.arange(start, stop, dtype=np.int64)
        p = alpha % P
        m = (alpha // P) % M
        d = alpha // (P * M)
        return ((d >> (L - self.pf_rank)) * N + self.m_to_n[m]) * P + p


    def _row_labels_block(self, start, stop, L):
        """The group index at coarse-graining rank L for each STORED row in [start, stop).

        Handles both sources: a fine map (rows are alpha) and an already-coarse one (rows are
        beta at self.L, and L > self.L). Coarsening a coarse map is well-defined because the
        families are nested -- at L' > L each new group is an exact union of old ones, since
        the subband and profile axes are untouched and the DM axis is a dyadic split.
        """

        if not self.is_coarse_grained:
            return self.alpha_to_beta_block(start, stop, L)

        N, P = self.nsubbands, self.nprofiles
        beta = np.arange(int(start), int(stop), dtype=np.int64)
        p = beta % P
        n = (beta // P) % N
        dc = beta // (P * N)
        return ((dc >> (L - self.L)) * N + n) * P + p


    def group_sizes(self, L=None):
        """Length-nbeta int array: how many fine alpha are in each group.

        Groups are NOT all the same size -- a subband at level l contributes
        ``2^(L-R) * 2^l`` -- and D is a mean over alpha, so this weight matters whenever
        groups are being aggregated or sampled.
        """

        L = self.L if (L is None) else int(L)
        if L is None:
            return np.ones(self.nalpha, dtype=np.int64)
        if not (self.pf_rank <= L <= self.tree_rank):
            raise RuntimeError(f'VarianceMap.group_sizes: L={L} is out of range [R, r] ='
                               f' [{self.pf_rank}, {self.tree_rank}]')

        per_subband = (1 << (L - self.pf_rank)) * (1 << self._n_level)
        return np.tile(np.repeat(per_subband, self.nprofiles), 1 << (self.tree_rank - L))


    def group_members(self, beta, L=None):
        """The fine indices alpha belonging to coarse group 'beta', as an int array.

        The inverse of alpha_to_beta_block() for one group. Cheap and closed-form: group
        (dc, n, p) is the set of ``alpha = (d*M + m)*P + p`` with d in the dyadic block dc and
        m in subband n's multiplet range.
        """

        L = self.L if (L is None) else int(L)
        if L is None:
            raise RuntimeError('VarianceMap.group_members: this map has no coarse-graining,'
                               ' so each group is the single row beta; pass an explicit L')

        M, N, P = self.nmultiplets, self.nsubbands, self.nprofiles
        beta = int(beta)
        p = beta % P
        n = (beta // P) % N
        dc = beta // (P * N)

        d = np.arange(dc << (L - self.pf_rank), (dc+1) << (L - self.pf_rank), dtype=np.int64)
        m = np.arange(self._n_to_mbase[n], self._n_to_mbase[n] + (1 << self._n_level[n]),
                      dtype=np.int64)
        return ((d[:,None] * M + m[None,:]) * P + p).ravel()


    # ---------------- transformations ----------------

    def coarse_grain(self, L):
        """Return the coarse-grained map ``Abar[beta,F] = max over alpha in beta of
        A[alpha,F]``, at coarse-graining rank L. Requires ``R <= L <= r``.

        Accepts a FINE map (the usual case) or an already-coarse one, in which case L must be
        strictly greater than self.L. That second case is not a convenience: a brute-force
        sweep costs O(nfreq) passes and is INDEPENDENT of L, while the LP costs O(nbeta) and
        ``nbeta = 2^(r-L) * N * P``, so every level you LOWER L doubles the LP work while
        leaving the sweep unchanged. The scalable workflow is therefore "sweep once at the
        finest L you can store, then coarsen down", and without coarse-to-coarser every level
        would need its own sweep.

        y_true is carried through unchanged, and the result inherits self.is_admissible (a
        max-envelope of an admissible map is admissible; a max-envelope of an uncertified one
        is uncertified).

        Applied to the TRUE map this gives the Abar of notes/variance_map.tex: the MINIMAL
        coarse map, and by the pivot identity -- any q with ``W q >= Abar[beta,:]`` is
        admissible for every fine row in the group -- the right reference for everything
        downstream. Applied to an approximation it is still well-defined and still
        admissible, just looser.
        """

        L = int(L)
        r, R = self.tree_rank, self.pf_rank
        if not (R <= L <= r):
            raise RuntimeError(f'VarianceMap.coarse_grain: L={L} is out of range [R, r] ='
                               f' [{R}, {r}]')
        if self.is_coarse_grained and (L <= self.L):
            raise RuntimeError(f'VarianceMap.coarse_grain: this map is already coarse at'
                               f' L={self.L}, so it can only be coarsened FURTHER (L > {self.L}),'
                               f' not to L={L}. Coarse-graining is not invertible.')

        nbeta_out = (1 << (r - L)) * self.nsubbands * self.nprofiles

        # -inf rather than zero, so that this is correct for a map with negative entries (a
        # factored ref can have them) and so that an unfilled group is detectable. Every group
        # is occupied for R <= L <= r, so the check below is a tripwire on the index
        # arithmetic rather than a real case.
        Abar = np.full((nbeta_out, self.nfreq), -np.inf)

        t0 = time.time()
        nb = self.default_block_rows()
        for start in range(0, self.nbeta, nb):
            stop = min(start + nb, self.nbeta)
            rows = self.rows(start, stop)
            lab = self._row_labels_block(start, stop, L)

            # Sort-and-reduceat rather than np.maximum.at: same answer (max is exact, so the
            # reduction order cannot matter), much faster, and this operation dominates a
            # large reduction.
            order = np.argsort(lab, kind='stable')
            ls = lab[order]
            uniq, first = np.unique(ls, return_index=True)
            seg = np.maximum.reduceat(rows[order], first, axis=0)
            Abar[uniq] = np.maximum(Abar[uniq], seg)

        empty = np.flatnonzero(np.all(Abar == -np.inf, axis=1))
        if empty.size > 0:
            raise RuntimeError(f'VarianceMap.coarse_grain: {empty.size} of {nbeta_out} groups'
                               f' received no rows (first: beta={empty[0]}). Every group is'
                               ' occupied for R <= L <= r, so this is a bug in the index'
                               ' arithmetic.')

        rec = dict(step='coarse_grain', L=L, L_from=self.L, nbeta=nbeta_out,
                   seconds=time.time() - t0)
        return self.replace(A=Abar, L=L, history_record=rec)


    def inflated(self, factor):
        """Return self scaled by 'factor'.

        Used by measure_admissibility(inflate=True) to turn an inadmissible approximation into
        an admissible one so that it can be reported on the same axis as an admissible one.

        ``is_admissible`` is preserved only when ``factor >= 1``: scaling DOWN can break the
        covering property, and a flag that survived it would be a lie.

        O(K^2) for a factored map, which stays factored: the scale goes into ``mid``, so
        neither factor is touched and both semiorthogonality claims survive. O(nbeta*nfreq)
        for a dense one.
        """

        factor = float(factor)
        rec = dict(step='inflated', factor=factor)
        adm = self.is_admissible and (factor >= 1.0)

        if self.is_factored:
            return self.replace(mid=np.asarray(self.mid, dtype=np.float64) * factor,
                                Q_is_semiorthogonal=self.Q_is_semiorthogonal,
                                W_is_semiorthogonal=self.W_is_semiorthogonal,
                                is_admissible=adm, history_record=rec)

        return self.replace(A=self.rows(0, self.nbeta) * factor,
                            is_admissible=adm, history_record=rec)


    def lift(self, *, max_bytes=1 << 31):
        """Return the equivalent NON-coarse-grained map, with each coarse row duplicated
        across its group.

        Conceptually useful and used by tests; for a DENSE map at production scale this is a
        memory disaster, so it refuses above 'max_bytes'. Prefer to keep maps coarse and let
        get_distance() do the lifting implicitly.

        A FACTORED map lifts cheaply and stays factored: only the rows of Q are duplicated,
        so the cost is ``nalpha * K`` rather than ``nalpha * nfreq``, and W and mid are
        untouched.
        """

        if not self.is_coarse_grained:
            return self

        rec = dict(step='lift', L_from=self.L)

        if self.is_factored:
            nbytes = 8 * self.nalpha * self.factor_rank
            if nbytes > max_bytes:
                raise RuntimeError(f'VarianceMap.lift(): the lifted Q would be'
                                   f' {nbytes/(1<<30):.1f} GiB, over the'
                                   f' {max_bytes/(1<<30):.1f} GiB limit.')
            Q = np.asarray(self.Q)[self.alpha_to_beta_block(0, self.nalpha)]
            return self.replace(Q=Q, L=None,
                                Q_is_semiorthogonal=False,
                                W_is_semiorthogonal=self.W_is_semiorthogonal,
                                history_record=rec)

        nbytes = 8 * self.nalpha * self.nfreq
        if nbytes > max_bytes:
            raise RuntimeError(f'VarianceMap.lift(): the lifted matrix would be'
                               f' {nbytes/(1<<30):.1f} GiB, over the'
                               f' {max_bytes/(1<<30):.1f} GiB limit. get_distance() and'
                               ' measure_admissibility() work on the coarse map directly.')

        out = np.empty((self.nalpha, self.nfreq))
        for start in range(0, self.nalpha, self._ALPHA_BLOCK):
            stop = min(start + self._ALPHA_BLOCK, self.nalpha)
            out[start:stop] = self.A[self.alpha_to_beta_block(start, stop)]

        return self.replace(A=out, L=None, history_record=rec)


    # ---------------- factorizations ----------------
    #
    # Everything above MOVES data; these CHOOSE or reshape a factorization, and they are what
    # give the two carried claims their meaning:
    #
    #   - THE SEMIORTHOGONALITY FLAGS. svd() and reorthogonalize() are the only things that set
    #     one True, truncate() is the only thing that reads one, and everywhere else replace()'s
    #     conservative False applies. Nothing here verifies a flag numerically: the point of the
    #     flags is to record what the code that built the factors knows, which is cheaper and
    #     more reliable than an O(nbeta*K^2) re-check.
    #   - pinned_columns. A pinned column is one the steps hold fixed, and it exists so that a
    #     NONNEGATIVE column is guaranteed to be present: the additive repair, the one-hot seed
    #     and the LP's feasibility certificate all need one, and a raw SVD basis has none
    #     (numpy's per-mode sign is arbitrary, and a real subbanded map came back with mode 0
    #     all-negative). So pin_column() checks the column it is handed, and every method that
    #     drops or reorders columns REMAPS the set rather than copying it -- a stale index is
    #     silent right up until a repair goes looking for its nonnegative column and finds an
    #     arbitrary one.

    def _require_factored(self, what):
        if not self.is_factored:
            raise RuntimeError(f'VarianceMap.{what}: this map is dense, and this operates on the'
                               ' factorization A = Q mid W.T. Build one with with_basis() or'
                               ' svd() first.')

    def _mid_is_identity(self):
        """True iff 'mid' is exactly the identity. O(K^2), and the answer is what decides
        whether a caller has to fold mid into Q."""
        return np.array_equal(np.asarray(self.mid), np.eye(self.factor_rank))

    def _QM(self):
        """``Q @ mid``, i.e. Q in the mid-free convention that varmap.lp works in.

        Returns the stored Q itself when mid is the identity, so the common case costs a K^2
        comparison and no matmul.
        """
        Q = np.asarray(self.Q, dtype=np.float64)
        return Q if self._mid_is_identity() else (Q @ np.asarray(self.mid, dtype=np.float64))

    @staticmethod
    def _qr_posdiag(X):
        """Thin QR with a NONNEGATIVE diagonal on R, which numpy does not guarantee.

        The sign convention is what makes an ordered QR useful: with R[0,0] > 0, the first
        column of the orthonormal factor is a POSITIVE multiple of the first input column, so a
        nonnegative first column stays nonnegative.
        """
        Qf, R = np.linalg.qr(np.asarray(X, dtype=np.float64), mode='reduced')
        sg = np.sign(np.diag(R))
        sg = np.where(sg == 0.0, 1.0, sg)
        # In place on both: np.linalg.qr always returns fresh arrays, and Qf is the (nbeta, ell)
        # one -- 3.1 GiB at CHORD with rank 128, which is worth not copying a third time.
        Qf *= sg[None, :]
        R *= sg[:, None]
        return Qf, R

    def _pin_order(self, keep_pinned):
        """(npin, order): the column permutation that puts the pinned columns first.

        npin is 0 when there is nothing to preserve, in which case 'order' is the identity and
        every caller's pinned path collapses to the plain one.
        """
        K = self.factor_rank
        pin = np.asarray(self.pinned_columns, dtype=np.int64)
        if (not keep_pinned) or (pin.size == 0):
            return 0, np.arange(K, dtype=np.int64)
        rest = np.setdiff1d(np.arange(K, dtype=np.int64), pin)
        return int(pin.size), np.concatenate([pin, rest])

    @staticmethod
    def _svd_nkeep(s, factor_rank, eps):
        """How many leading modes to keep: at most 'factor_rank', and none with
        ``s < eps * s[0]``."""
        n = s.size if (factor_rank is None) else min(int(factor_rank), s.size)
        if (eps is not None) and (s.size > 0):
            n = min(n, int(np.count_nonzero(s >= float(eps) * s[0])))
        if n < 1:
            raise RuntimeError(f'VarianceMap.svd: every mode was dropped (factor_rank='
                               f'{factor_rank}, eps={eps}, {s.size} singular values, largest'
                               f' {s[0] if s.size else 0.0:.6g}). A rank-0 map is not'
                               ' representable.')
        return n

    def _shape_scale(self, use):
        """The per-row divisor for 'shape_normalize', or None.

        Rows that sum to zero are left alone rather than divided: their shape is undefined, and
        a zero row is a zero row in either normalization.
        """
        if not use:
            return None
        rs = np.asarray(self.row_sums(), dtype=np.float64)
        return np.where(rs > 0.0, rs, 1.0)

    def _blocked_AM(self, M, rs):
        """``diag(1/rs) A @ M`` for a DENSE self, in row blocks (rs=None means no scaling)."""
        M = np.asarray(M, dtype=np.float64)
        out = np.empty((self.nbeta, M.shape[1]))
        nb = self.default_block_rows()
        for start in range(0, self.nbeta, nb):
            stop = min(start + nb, self.nbeta)
            blk = self.rows(start, stop)
            out[start:stop] = blk @ M
            if rs is not None:
                out[start:stop] /= rs[start:stop, None]
        return out

    def _blocked_AtY(self, Y, rs):
        """``(diag(1/rs) A).T @ Y`` for a DENSE self, in row blocks."""
        Y = np.asarray(Y, dtype=np.float64)
        out = np.zeros((self.nfreq, Y.shape[1]))
        nb = self.default_block_rows()
        for start in range(0, self.nbeta, nb):
            stop = min(start + nb, self.nbeta)
            y = Y[start:stop] if (rs is None) else (Y[start:stop] / rs[start:stop, None])
            out += self.rows(start, stop).T @ y
        return out

    def _svd_randomized(self, factor_rank, rs, rng, oversample, power_iters):
        """(U, s, V) of a dense self by a randomized range finder, in ``1 + 2*power_iters``
        blocked passes with nothing of matrix size in memory.

        This is what makes an SVD basis reachable at a scale where the matrix is a file. It is
        APPROXIMATE and it depends on the draw, so pass an explicit 'rng' for anything that has
        to be reproducible -- and note that reproducibility does not survive a numpy version
        change, which is why the campaign's own dictionaries were cached rather than rebuilt.

        HOW ACCURATE IT HAS TO BE IS NOT THE TEXTBOOK QUESTION, and this is the one place where
        taking the textbook defaults is wrong. Measured on a real coarse map at rank 16, the
        standard setting (one power iteration, 10 extra samples) lands within 2.5% of the
        optimal rank-K residual -- textbook-good -- and yet delivers D **1.40x worse** than the
        exact SVD once the basis is handed to a Q-step. There is no contradiction: the residual
        is an RMS over the whole matrix, while D is paid on each group's WORST channel, so a
        few percent of misplaced Frobenius mass is a large covering error. The defaults set in
        svd() are chosen against D, not against the residual.

        BUDGET IT AS A HANDFUL OF ``nbeta * (factor_rank + oversample) * 8`` BYTE ARRAYS and
        nothing of matrix size; the peak is np.linalg.qr (which copies its input and carries
        LAPACK workspace) plus the previous iterate still being live. A memory bar expressed as
        a FRACTION OF THE MATRIX is the wrong denominator and silently becomes a different test
        at a different rank -- 19 GiB at rank 128 is 5.5% of the matrix but 6.1x one work
        array, and only the second number means anything.
        """
        if factor_rank is None:
            raise RuntimeError("VarianceMap.svd: method='randomized' needs an explicit"
                               ' factor_rank -- it approximates a fixed number of modes, so'
                               ' there is no full spectrum for eps to threshold.')

        ell = int(min(int(factor_rank) + int(oversample), self.nfreq, self.nbeta))
        rng = np.random.default_rng(0) if (rng is None) else rng

        Qy = np.linalg.qr(self._blocked_AM(rng.standard_normal((self.nfreq, ell)), rs),
                          mode='reduced')[0]
        Z = R = None
        for _ in range(max(1, int(power_iters))):
            Z, _ = self._qr_posdiag(self._blocked_AtY(Qy, rs))
            Qy, R = self._qr_posdiag(self._blocked_AM(Z, rs))

        # A Z = Qy R and A ~ (A Z) Z.T, so the small SVD of R carries both factors.
        Ub, s, Vbt = np.linalg.svd(R, full_matrices=False)
        return Qy @ Ub, s, Z @ Vbt.T

    def _svd_dense(self, factor_rank, eps, shape_normalize, method, rng, oversample,
                   power_iters):
        """(Q, mid, W, Q_is_semiorthogonal) for a truncated SVD of a DENSE self. 'method' is
        already resolved to 'exact' or 'randomized'."""

        rs = self._shape_scale(shape_normalize)

        if method == 'exact':
            B = self.rows(0, self.nbeta)
            B = B if (rs is None) else (B / rs[:, None])
            U, s, Vt = np.linalg.svd(B, full_matrices=False)
            V = Vt.T
        elif method == 'randomized':
            U, s, V = self._svd_randomized(factor_rank, rs, rng, oversample, power_iters)
        else:
            raise RuntimeError(f'VarianceMap.svd: method={method!r} is not one of'
                               " 'auto'/'exact'/'randomized'")

        n = self._svd_nkeep(s, factor_rank, eps)
        Q = np.ascontiguousarray(U[:, :n])
        if rs is not None:
            Q *= rs[:, None]
        return Q, np.diag(s[:n]), np.ascontiguousarray(V[:, :n]), (rs is None)

    def _svd_factored(self, factor_rank, eps, shape_normalize, keep_pinned):
        """(Q, mid, W, Q_is_semiorthogonal, npin) for a truncated SVD of a FACTORED self, with
        no dense product anywhere: two thin QRs and one K-by-K SVD.

        This is the rank-reduction path -- take an accurate high-rank factorization, drop modes,
        and let a Q-step restore admissibility -- so its cost must depend on K and not on nfreq
        times nbeta.
        """

        rs = self._shape_scale(shape_normalize)
        Qm = self._QM()
        Qm = Qm if (rs is None) else (Qm / rs[:, None])
        W = np.asarray(self.W, dtype=np.float64)

        npin, order = self._pin_order(keep_pinned)
        if npin:
            # An ordered QR of W with the pinned columns first, then SVD only what is left. The
            # pinned columns come through as an orthonormal basis of their own span -- so the
            # FIRST one is still a positive multiple of itself, and hence still nonnegative --
            # while a plain SVD would rotate them together with everything else and destroy it.
            Wq, R = self._qr_posdiag(W[:, order])
            if R[0, 0] <= 0.0:
                raise RuntimeError('VarianceMap.svd: the first pinned column of W is zero, so'
                                   ' there is no nonnegative column to preserve. Re-pin with'
                                   ' pin_column(), or pass keep_pinned=False.')
            G = Qm[:, order] @ R.T
            base_Q, base_W = G[:, :npin], Wq[:, :npin]
            res_Q, res_W = G[:, npin:], Wq[:, npin:]
        else:
            base_Q = base_W = None
            res_Q, res_W = Qm, W

        # SVD of res_Q @ res_W.T through two thin QRs and one small SVD. In the pinned case
        # res_W is already orthonormal AND orthogonal to base_W, so its rotation below stays
        # orthogonal to the preserved columns and the result is semiorthogonal as a whole.
        Qg, Rg = self._qr_posdiag(res_Q)
        Wg, Rw = self._qr_posdiag(res_W)
        Um, s, Vmt = np.linalg.svd(Rg @ Rw.T, full_matrices=False)

        want = None if (factor_rank is None) else max(0, int(factor_rank) - npin)
        if (want is not None) and (want < 1) and (s.size > 0):
            raise RuntimeError(f'VarianceMap.svd: factor_rank={factor_rank} leaves no room for'
                               f' any mode beside the {npin} pinned column(s). Ask for a rank'
                               ' above the pin count, or pass keep_pinned=False.')
        n = self._svd_nkeep(s, want, eps)

        Q = Qg @ Um[:, :n]
        W = Wg @ Vmt[:n].T
        sing = s[:n]
        if npin:
            Q = np.ascontiguousarray(np.hstack([base_Q, Q]))
            W = np.ascontiguousarray(np.hstack([base_W, W]))
            # The preserved columns carry their coefficients in Q, so their entry in the
            # diagonal middle matrix is 1 rather than a singular value.
            sing = np.concatenate([np.ones(npin), sing])
        if rs is not None:
            Q = Q * rs[:, None]
        # Q is orthonormal only when nothing was prepended and no row scale was folded back.
        return (np.ascontiguousarray(Q), np.diag(sing), np.ascontiguousarray(W),
                (npin == 0) and (rs is None), npin)

    def svd(self, factor_rank=None, *, eps=None, shape_normalize=None, keep_pinned=True,
            method='auto', max_bytes=1 << 31, rng=None, oversample=None,
            power_iters=None):
        """Return a factored VarianceMap holding a truncated SVD of self: ``Q = U``,
        ``mid = diag(s)``, ``W = V``.

        Keeps at most 'factor_rank' modes and drops any with ``s < eps * s[0]``; at least one of
        the two must be given. Very small singular values are not helping the approximation and
        are better dropped than carried.

        THE RESULT HAS ``is_admissible = False``, AND THAT IS NOT A TECHNICALITY. A truncated SVD
        used directly as an approximation has an admissibility cliff -- below K ~ 32 it puts a
        non-positive value on a positive entry of the reference, which no rescaling repairs, so
        D is infinite. The SAME truncation used as a W-MATRIX, with a Q-step free to choose the
        coefficients, has no cliff at all. Conflating the two is the single most expensive
        confusion available here, so this method is honest about which it produces: a
        Frobenius-optimal truncation, usable as an approximation only when it happens to be
        admissible and usable as a basis always. qstep() is what makes something admissible.

        CALL canonicalize_signs() ON THE RESULT. numpy's per-mode sign is arbitrary, so a raw SVD
        basis typically has ZERO nonnegative columns, and the additive repair, the one-hot seed
        and the LP's feasibility certificate all need one.

        Parameters
        ----------
        shape_normalize : bool or None
            Decompose the unit-sum SHAPE matrix ``S[beta,F] = A[beta,F] / sum_F A[beta,F]``
            instead of A, folding the row sums back into Q. The shape SVD is the better
            W-matrix at rank >= 32 and A itself below it, so None means "choose by rank" at
            that measured crossover. Because the row sums go into Q, the result is NOT
            semiorthogonal on the Q side and truncate() will refuse it -- ask svd() for the
            rank you want instead.
        keep_pinned : bool
            Only meaningful for a factored self that has pinned columns, where a plain SVD would
            rotate all columns together and destroy the nonnegative column the seed and the
            additive repair depend on. True preserves the pinned columns' span, with the same
            guarantee and the same caveats as reorthogonalize(): the FIRST pinned column comes
            through as a positive multiple of itself, the later ones are orthogonalized against
            it and can go negative, and all of them are rescaled to unit norm. False drops the
            pinned set outright, which is how a pinned-versus-not comparison is run.
        method : str
            'exact' is one ``np.linalg.svd`` of the whole matrix; 'randomized' is a range finder
            with one power iteration, three blocked passes and nothing of matrix size in memory,
            which is the only one of the two available once the matrix is a file. 'auto' picks
            exact only when the matrix fits in 'max_bytes' AND ``min(nbeta, nfreq)`` is within
            2x of ``factor_rank + oversample`` -- i.e. on predicted flops, not on size alone,
            because exact is superlinear in the small dimension and randomized is not. An
            eps-only truncation always takes exact, since randomized has no rank to work to.
            Ignored for a factored self, which is always exact and cheap.
        rng, oversample, power_iters
            The randomized path's draw, its extra sample count and its number of power
            iterations -- ``1 + 2*power_iters`` blocked passes. The defaults are measured
            against D rather than against the residual, and are NOT the textbook ones; the
            table is on ``_SVD_OVERSAMPLE_MULT``, and the short version is that oversampling
            buys basis quality far more cheaply than power iterations do. Lower them only with
            a measurement in hand.
        """

        if (factor_rank is None) and (eps is None):
            raise RuntimeError('VarianceMap.svd: give factor_rank, or eps, or both -- with'
                               ' neither there is nothing to truncate to.')
        if (factor_rank is not None) and (int(factor_rank) < 1):
            raise RuntimeError(f'VarianceMap.svd: factor_rank={factor_rank} must be >= 1')
        if shape_normalize is None:
            shape_normalize = ((factor_rank is not None)
                               and (int(factor_rank) >= self._SHAPE_NORMALIZE_RANK))
        if oversample is None:
            oversample = max(self._SVD_OVERSAMPLE_MIN,
                             self._SVD_OVERSAMPLE_MULT * int(factor_rank or 0))
        power_iters = self._SVD_POWER_ITERS if (power_iters is None) else int(power_iters)

        # Resolved here rather than inside the dense path, so that the history says which
        # algorithm actually ran -- 'auto' in a record is exactly the thing nobody can
        # reconstruct later.
        if self.is_factored:
            method = 'factored'
        elif method == 'auto':
            # CHOOSE ON FLOPS, NOT ON BYTES. Exact costs O(nbeta * nfreq * min(nbeta, nfreq));
            # randomized costs O(nbeta * nfreq * ell) with ell = factor_rank + oversample. So
            # exact wins only while min(nbeta, nfreq) is comparable to ell -- and a pure memory
            # test picks exact PRECISELY WHERE IT IS WORST, just under the byte bound where the
            # matrix is largest. Measured at nbeta = 12800, K = 16, single-threaded: exact took
            # 174 s at nfreq = 6400 and over 800 s (killed) at nfreq = 12800, against 5.0 s for
            # randomized at nfreq = 28160. That is a 35x inversion, and it cost an agent a cell
            # of a cost model before anyone noticed.
            #
            # The memory bound stays authoritative, and is tested FIRST: exact materializes the
            # whole matrix, and when it does not fit there is no choice to make. Keeping that
            # branch first also preserves the informative error for an eps-only truncation of a
            # matrix that does not fit -- randomized raises there, saying it needs an explicit
            # factor_rank, which is more useful than an out-of-memory kill from exact.
            if 8 * self.nbeta * self.nfreq > int(max_bytes):
                method = 'randomized'
            elif factor_rank is None:
                method = 'exact'      # randomized approximates a fixed number of modes; eps
                                      # thresholds a full spectrum, and has no rank to give it.
            else:
                ell = int(factor_rank) + int(oversample)
                method = ('exact' if (min(self.nbeta, self.nfreq) <= 2 * ell)
                          else 'randomized')

        t0 = time.time()
        if self.is_factored:
            Q, mid, W, qflag, npin = self._svd_factored(factor_rank, eps, shape_normalize,
                                                        keep_pinned)
        else:
            Q, mid, W, qflag = self._svd_dense(factor_rank, eps, shape_normalize, method,
                                               rng, oversample, power_iters)
            npin = 0

        rec = dict(step='svd', factor_rank=int(W.shape[1]), eps=eps, method=method,
                   shape_normalize=bool(shape_normalize), n_pinned=int(npin),
                   # Only meaningful for the randomized path, and recorded because it is what
                   # nobody can reconstruct from the result: two runs at different sampling
                   # give bases that look identical by every cheap measure and differ by 1.4x
                   # in delivered D.
                   oversample=(int(oversample) if (method == 'randomized') else None),
                   power_iters=(int(power_iters) if (method == 'randomized') else None),
                   n_pinned_dropped=int(self.pinned_columns.size - npin
                                        if self.is_factored else 0),
                   seconds=time.time() - t0)
        return self.replace(Q=Q, mid=mid, W=W, pinned_columns=np.arange(npin),
                            Q_is_semiorthogonal=bool(qflag), W_is_semiorthogonal=True,
                            is_admissible=False, history_record=rec)

    def truncate(self, factor_rank):
        """Drop all but the leading 'factor_rank' modes, and set ``is_admissible = False``.

        Only meaningful straight after svd() or reorthogonalize(), i.e. while both factors are
        semiorthogonal and mid is diagonal -- otherwise "leading" is not a property of the
        column order and the truncation is not the Frobenius-optimal one. That is CHECKED, and
        select_columns() is the unchecked primitive for keeping an arbitrary subset.

        Dropping a pinned column raises rather than silently unpinning it.
        """

        self._require_factored('truncate')
        K, K0 = int(factor_rank), self.factor_rank
        if not (1 <= K <= K0):
            raise RuntimeError(f'VarianceMap.truncate: factor_rank={K} is out of range'
                               f' [1, {K0}]')
        if not (self.Q_is_semiorthogonal and self.W_is_semiorthogonal):
            raise RuntimeError(
                'VarianceMap.truncate: this factorization does not claim semiorthogonal Q and'
                ' W, so its columns are not ordered by singular value and keeping a prefix is'
                ' not a truncated SVD. Use svd() to build one at the rank you want, or'
                ' select_columns() to keep a subset with no such claim.')
        mid = np.asarray(self.mid, dtype=np.float64)
        if not np.array_equal(mid, np.diag(np.diag(mid))):
            raise RuntimeError("VarianceMap.truncate: 'mid' is not diagonal, so the modes are"
                               ' mixed and a prefix is not a truncation. See select_columns().')
        pin = np.asarray(self.pinned_columns, dtype=np.int64)
        if pin.size and (int(pin.max()) >= K):
            raise RuntimeError(f'VarianceMap.truncate: pinned column {int(pin.max())} is'
                               f' outside the kept prefix [0, {K}). Dropping a pinned column'
                               ' loses the nonnegative column the additive repair needs, so'
                               ' say so explicitly with select_columns() if that is intended.')

        rec = dict(step='truncate', factor_rank=K, factor_rank_from=K0)
        # A column subset of a semiorthogonal matrix is still semiorthogonal, so both claims
        # survive verbatim.
        return self.replace(Q=np.ascontiguousarray(np.asarray(self.Q)[:, :K]),
                            mid=np.ascontiguousarray(mid[:K, :K]),
                            W=np.ascontiguousarray(np.asarray(self.W)[:, :K]),
                            Q_is_semiorthogonal=True, W_is_semiorthogonal=True,
                            is_admissible=False, history_record=rec)

    def reorthogonalize(self, *, keep_pinned=True):
        """Re-express A as ``Q mid W.T`` with W semiorthogonal, at the same rank and with the
        SAME matrix A -- exact, not an approximation. Nothing changes but the factorization and
        the flags.

        HOW THE PINNED COLUMNS SURVIVE, and why it is worth the trouble. A plain SVD-based
        reorthogonalization rotates every column together, which destroys the nonnegative column
        the seed and the additive repair depend on: measured at 1.769x in D, with no choice of
        'mid' recovering it. So this reorthogonalizes by an ORDERED QR with the pinned columns
        first, ``W = W' R`` with R upper triangular, and folds R into the other factor so the
        product is exact. Because R is upper triangular with a positive diagonal, ``W'[:,0]`` is
        a POSITIVE multiple of the first pinned column, so it is still nonnegative and the
        feasibility certificate survives with ``q = c e_0`` for some c > 0.

        Two consequences to know before relying on it:

        - Only the FIRST pinned column is guaranteed to stay nonnegative. Later ones are
          orthogonalized against the earlier ones and can go negative. With the single envelope
          column that is the recommended pinning this is no limitation; with several it is.
        - The preserved column is rescaled to unit norm, so it is no longer literally the
          envelope over groups. The certificate is unaffected; an equality test against the
          envelope column is not.

        With ``keep_pinned=False`` the QR runs in the columns' own order and the pinned set is
        DROPPED, since the columns it named no longer exist as such. (Whether any nonnegative
        column happens to survive is then an accident of ordering -- column 0 always does --
        which is exactly the point: the guarantee comes from the ordering, not from the QR.)
        That is the right thing when there are no pinned columns, and it is also how the
        pinned-versus-not comparison is run -- the one variant never measured, and the
        experiment to run before concluding anything about reorthogonalization at all. As an
        intervention it has not paid so far: it is provably a no-op on D by construction, and as
        a preconditioner it helped 1.17x at rank 64 and HURT 3.08x at 128, in measurements taken
        without pinning, where the rotation was destroying the nonnegative column at the same
        time as it was changing the conditioning.
        """

        self._require_factored('reorthogonalize')
        npin, order = self._pin_order(keep_pinned)
        Qm = self._QM()[:, order]
        Wq, R = self._qr_posdiag(np.asarray(self.W, dtype=np.float64)[:, order])
        if npin and (R[0, 0] <= 0.0):
            raise RuntimeError('VarianceMap.reorthogonalize: the first pinned column of W is'
                               ' zero, so there is no nonnegative column to preserve. Re-pin'
                               ' with pin_column(), or pass keep_pinned=False.')

        ndropped = int(np.asarray(self.pinned_columns).size - npin)
        rec = dict(step='reorthogonalize', n_pinned=int(npin), n_pinned_dropped=ndropped)
        return self.replace(Q=np.ascontiguousarray(Qm @ R.T), mid=None, W=Wq,
                            pinned_columns=np.arange(npin),
                            Q_is_semiorthogonal=False, W_is_semiorthogonal=True,
                            history_record=rec)

    def with_basis(self, W, *, mid=None, pinned_columns=None):
        """Return a factored map with the given W and an UNSET Q (all zero), ready for a
        qstep(). ``is_admissible`` is False, since a zero Q covers nothing.

        This is how a W-matrix built elsewhere enters the pipeline -- a different tree, a
        different config, a random matrix, a column-subset selection. It is a first-class entry
        point rather than a curiosity: a transplanted W starts out 3x-19x worse than a
        purpose-built one, but ONE W-step brings it to within 3-9%, which is the cheapest known
        route to a good basis at a new config. The one axis where transfer does not work is
        channel count; treat a foreign nfreq as a seed rather than a transfer.
        """

        W = np.asarray(W, dtype=np.float64)
        if (W.ndim != 2) or (W.shape[0] != self.nfreq):
            raise RuntimeError(f'VarianceMap.with_basis: W has shape {W.shape}, expected'
                               f' ({self.nfreq}, K) -- its rows are frequency channels')

        K = int(W.shape[1])
        rec = dict(step='with_basis', factor_rank=K,
                   n_pinned=(0 if pinned_columns is None else len(list(pinned_columns))))
        return self.replace(Q=np.zeros((self.nbeta, K)), mid=mid, W=W,
                            pinned_columns=pinned_columns,
                            Q_is_semiorthogonal=False, W_is_semiorthogonal=False,
                            is_admissible=False, history_record=rec)

    # ---------------- the column algebra ----------------

    def canonicalize_signs(self):
        """Flip each FREE column of W so that its entries sum to >= 0, compensating exactly in
        the other factors so that A is bitwise unchanged.

        Cheap, exactly invariant, and there is no reason not to call it on any freshly built
        basis: the LP is invariant under a per-column sign flip when q is sign-free, but the
        one-hot seed and the additive repair both search for a NONNEGATIVE column and numpy's
        SVD sign convention is arbitrary. Without this a raw SVD basis has zero nonnegative
        columns and the seed fails outright.

        Pinned columns are left alone -- they are nonnegative by construction, and flipping one
        would destroy the very certificate it was pinned for.
        """

        self._require_factored('canonicalize_signs')
        W = np.asarray(self.W, dtype=np.float64)
        sgn = np.where(W.sum(axis=0) < 0.0, -1.0, 1.0)
        sgn[np.asarray(self.pinned_columns, dtype=np.int64)] = 1.0

        nflip = int(np.count_nonzero(sgn < 0.0))
        rec = dict(step='canonicalize_signs', n_flipped=nflip)
        if nflip == 0:
            return self.replace(history_record=rec)

        # The symmetric conjugation Q -> Q D, W -> W D, mid -> D mid D with D = diag(+-1) leaves
        # the product invariant to the last bit (a sign flip is exact), keeps mid diagonal with
        # its diagonal untouched, and preserves both semiorthogonality claims.
        return self.replace(Q=np.asarray(self.Q, dtype=np.float64) * sgn[None, :],
                            mid=sgn[:, None] * np.asarray(self.mid, dtype=np.float64)
                                * sgn[None, :],
                            W=W * sgn[None, :],
                            Q_is_semiorthogonal=self.Q_is_semiorthogonal,
                            W_is_semiorthogonal=self.W_is_semiorthogonal,
                            history_record=rec)

    def n_nonneg_cols(self):
        """How many columns of W are nonnegative and not identically zero.

        The additive repair, the one-hot seed and the LP's feasibility certificate all need at
        least one, and a raw SVD basis has ZERO until canonicalize_signs() has run. Cheap, and
        worth asserting before a step that depends on it rather than discovering it inside a
        repair.
        """
        self._require_factored('n_nonneg_cols')
        W = np.asarray(self.W)
        return int(((W.min(axis=0) >= 0.0) & (W.max(axis=0) > 0.0)).sum())

    def rescale_columns(self, mode='unit'):
        """Rescale the columns of W to unit 2-norm, absorbing the reciprocal into 'mid'.

        WORTH UP TO 1.49x IN D, FOR REASONS NOBODY UNDERSTANDS, and that is why this is a step
        with a name rather than a keyword on svd(). The transformation is provably inert: it
        leaves the product unchanged and cannot change the feasible set or the objective of
        either LP, since ``(W[:,c], q_c) -> (lambda W[:,c], q_c/lambda)`` is an exact symmetry
        of both. It is nevertheless measured to be worth up to 1.49x at high rank.

        The obvious explanation -- that the columns sit far from the solver's ABSOLUTE
        feasibility tolerance -- was tested directly and FALSIFIED: a 64x change in the tolerance
        moves D in the eighth digit, while the scale itself is worth 34%. So the mechanism is
        unknown, and this may be a symptom of a bug elsewhere (in our own repair or
        admissibility logic as easily as in the solver) rather than a feature. Anyone who
        finds out should delete this paragraph and replace it with the reason.

        Unit column norm is the best setting measured and every alternative placement of the
        scale is worse, including the general middle-matrix form, so 'mode' is a single scalar
        convention and 'unit' is the only value implemented.
        """

        self._require_factored('rescale_columns')
        if mode != 'unit':
            raise RuntimeError(f'VarianceMap.rescale_columns: mode={mode!r} is not implemented.'
                               " 'unit' (unit column 2-norm) is the best setting measured and"
                               ' every alternative placement of the scale was worse, so there is'
                               ' nothing to check another convention against.')

        W = np.asarray(self.W, dtype=np.float64)
        lam = np.linalg.norm(W, axis=0)
        lam = np.where(lam > 0.0, lam, 1.0)         # an all-zero column has no scale to fix

        rec = dict(step='rescale_columns', mode=mode, scale_min=float(lam.min()),
                   scale_max=float(lam.max()))
        # Into 'mid', not into Q: that is what mid is for, and it makes this O(K^2) rather than
        # O(nbeta*K). Q is untouched, so its semiorthogonality claim survives; W's does not.
        return self.replace(mid=np.asarray(self.mid, dtype=np.float64) * lam[None, :],
                            W=W / lam[None, :],
                            Q_is_semiorthogonal=self.Q_is_semiorthogonal,
                            W_is_semiorthogonal=False, history_record=rec)

    def pin_column(self, w, *, replace_last=True):
        """Add 'w' (typically basis_envelope_column(ref)) to W as a PINNED column, at index 0.

        Index 0 is not cosmetic: the Q-step's prefix rescue re-solves a failed group on a PREFIX
        of W, so a pin at index 0 is in every prefix and every rescue LP is feasible by the same
        certificate as the full one.

        'w' must be nonnegative and not identically zero, and that IS checked -- the whole
        purpose of a pin is to guarantee a nonnegative column exists, and a pin that does not
        buys nothing while making n_nonneg_cols() look answered. Pass an arbitrary held-fixed
        column through with_basis(pinned_columns=...) if that is really what you want.

        With replace_last (the default) the new column takes the place of the last FREE column,
        so factor_rank is unchanged -- which is what makes "pinned versus not" a fair comparison
        at equal rank. The dropped column's contribution is lost, so ``is_admissible`` becomes
        False; with ``replace_last=False`` the rank grows by one, the product is bitwise
        unchanged and the flag survives.
        """

        self._require_factored('pin_column')
        w = np.asarray(w, dtype=np.float64).reshape(-1)
        if w.shape != (self.nfreq,):
            raise RuntimeError(f'VarianceMap.pin_column: w has {w.size} entries, expected'
                               f' nfreq = {self.nfreq}')
        if (w.min() < 0.0) or (w.max() <= 0.0):
            raise RuntimeError('VarianceMap.pin_column: w must be nonnegative and not'
                               ' identically zero. A pinned column exists to guarantee that the'
                               ' additive repair, the one-hot seed and the LP certificate have'
                               ' a nonnegative column to use.')

        K = self.factor_rank
        pin = np.asarray(self.pinned_columns, dtype=np.int64)
        keep = np.arange(K, dtype=np.int64)
        if replace_last:
            free = np.setdiff1d(keep, pin)
            if free.size == 0:
                raise RuntimeError('VarianceMap.pin_column: every column of W is already'
                                   ' pinned, so there is no free column to replace. Pass'
                                   ' replace_last=False to grow the rank instead.')
            keep = keep[keep != free[-1]]

        Q, mid, W = (np.asarray(self.Q, dtype=np.float64),
                     np.asarray(self.mid, dtype=np.float64),
                     np.asarray(self.W, dtype=np.float64))
        Knew = keep.size + 1
        midn = np.zeros((Knew, Knew))
        midn[0, 0] = 1.0
        midn[1:, 1:] = mid[np.ix_(keep, keep)]

        pos = {int(c): (i + 1) for i, c in enumerate(keep)}
        rec = dict(step='pin_column', factor_rank=Knew, factor_rank_from=K,
                   replace_last=bool(replace_last))
        return self.replace(
            Q=np.ascontiguousarray(np.hstack([np.zeros((self.nbeta, 1)), Q[:, keep]])),
            mid=midn,
            W=np.ascontiguousarray(np.hstack([w[:, None], W[:, keep]])),
            pinned_columns=np.array([0] + [pos[int(c)] for c in pin], dtype=np.int64),
            Q_is_semiorthogonal=False, W_is_semiorthogonal=False,
            is_admissible=(self.is_admissible and not replace_last),
            history_record=rec)

    def select_columns(self, idx):
        """Return a map keeping only these columns of W (and the matching columns of Q and rows
        and columns of mid), at the reduced factor_rank. ``is_admissible`` becomes False.

        pinned_columns is REMAPPED, not carried: it holds column INDICES, so dropping a column
        shifts every index above it, and a naive copy leaves the pinned set pointing at the
        wrong columns. That failure is silent -- the map stays admissible right up until the
        additive repair looks for its nonnegative column and finds an arbitrary one -- so
        dropping a pinned column raises instead.

        This is the primitive for RANK REDUCTION BY PRUNING: drop one column, re-run a Q-step,
        keep the drop that costs least D, repeat. That has never been measured against
        rebuilding at the target rank, which is exactly why the primitive should exist rather
        than the experiment being blocked on writing it.
        """

        self._require_factored('select_columns')
        K = self.factor_rank
        idx = np.asarray(idx, dtype=np.int64).reshape(-1)
        if idx.size == 0:
            raise RuntimeError('VarianceMap.select_columns: an empty column set leaves a rank-0'
                               ' map, which is not representable')
        if (idx.min() < 0) or (idx.max() >= K):
            raise RuntimeError(f'VarianceMap.select_columns: column indices must lie in'
                               f' [0, {K})')
        if np.unique(idx).size != idx.size:
            raise RuntimeError('VarianceMap.select_columns: duplicate column indices')

        pin = np.asarray(self.pinned_columns, dtype=np.int64)
        lost = np.setdiff1d(pin, idx)
        if lost.size:
            raise RuntimeError(f'VarianceMap.select_columns: this would drop pinned column(s)'
                               f' {list(lost)}. A pinned column is what guarantees W has a'
                               ' nonnegative one, so unpin it explicitly (replace(pinned_'
                               'columns=...)) if the drop is intended.')

        pos = {int(c): i for i, c in enumerate(idx)}
        rec = dict(step='select_columns', factor_rank=int(idx.size), factor_rank_from=K)
        # Selecting (or reordering) columns of a semiorthogonal matrix leaves it semiorthogonal,
        # so both claims survive.
        return self.replace(
            Q=np.ascontiguousarray(np.asarray(self.Q)[:, idx]),
            mid=np.ascontiguousarray(np.asarray(self.mid)[np.ix_(idx, idx)]),
            W=np.ascontiguousarray(np.asarray(self.W)[:, idx]),
            pinned_columns=np.array([pos[int(c)] for c in pin], dtype=np.int64),
            Q_is_semiorthogonal=self.Q_is_semiorthogonal,
            W_is_semiorthogonal=self.W_is_semiorthogonal,
            is_admissible=False, history_record=rec)

    def augment_basis(self, W_extra):
        """Append columns to W, with zero coefficients in Q so that the product is bitwise
        unchanged and ``is_admissible`` survives.

        The counterpart of select_columns(), and the primitive for GREEDY FORWARD SELECTION of a
        basis and for growing an existing approximation to a higher rank -- both posed and
        neither measured. pinned_columns needs no remapping here, since appending shifts nothing.
        """

        self._require_factored('augment_basis')
        We = np.asarray(W_extra, dtype=np.float64)
        if We.ndim == 1:
            We = We[:, None]
        if (We.ndim != 2) or (We.shape[0] != self.nfreq):
            raise RuntimeError(f'VarianceMap.augment_basis: W_extra has shape {We.shape},'
                               f' expected ({self.nfreq}, E)')

        K, E = self.factor_rank, int(We.shape[1])
        midn = np.eye(K + E)
        midn[:K, :K] = np.asarray(self.mid, dtype=np.float64)
        rec = dict(step='augment_basis', factor_rank=K + E, factor_rank_from=K)
        return self.replace(
            Q=np.ascontiguousarray(np.hstack([np.asarray(self.Q, dtype=np.float64),
                                              np.zeros((self.nbeta, E))])),
            mid=midn,
            W=np.ascontiguousarray(np.hstack([np.asarray(self.W, dtype=np.float64), We])),
            Q_is_semiorthogonal=False, W_is_semiorthogonal=False, history_record=rec)

    # ---------------- the steps ----------------
    #
    # There is deliberately NO run_schedule() / alternate() driver. A schedule is a sequence of
    # these calls, and a caller writing them out is both clearer and strictly more flexible than
    # any argument list a driver could grow:
    #
    #     m = ref.svd(factor_rank=K).canonicalize_signs().qstep(ref)   # the frozen arm
    #     m = m.wstep(ref).qstep(ref)                                  # one alternation
    #     m = m.wstep(ref).repair(ref)                                 # ... ending cheaply
    #
    # Each step appends a record to the returned map's 'history' (name, config, D, max_r, wall
    # time, solver info), so the per-step log a driver would have returned is carried by the map
    # itself and survives being written to a file.
    #
    # All three are WRAPPERS over varmap.lp, which owns the numerics and takes bare arrays. Four
    # things belong here rather than there, and they are the reason the wrappers are not
    # one-liners: the reference matrix and the group labels come from the geometry; 'mid' has to
    # be folded into Q before an additive repair, which is defined on the columns of W;
    # pinned_columns is what wstep() holds fixed; and is_admissible is inherited from 'ref'
    # rather than asserted, since admissibility is transitive and an uncertified ref certifies
    # nothing.

    def _lp_reference(self, ref, what):
        """The (nbeta, nfreq) float64 matrix of 'ref', after checking it against self.

        A dense ref is handed over as a VIEW: the reference at production scale is a memmapped
        coarse map, and copying it is what fails first. A FACTORED ref is materialized, which is
        allowed and is a research direction in its own right -- a Q-step whose reference is
        itself a high-rank factorization is how rank reduction is posed -- but it is the
        non-default path, and a factored ref can be NEGATIVE where a streamed max-envelope
        cannot, which every subproblem then feels: without a nonnegative column of W the LP is
        infeasible everywhere.
        """

        if self.shape != ref.shape:
            raise RuntimeError(f'VarianceMap.{what}: shape mismatch between self {self.shape}'
                               f' and ref {ref.shape}')
        if (self.is_coarse_grained != ref.is_coarse_grained) or (self.L != ref.L):
            raise RuntimeError(
                f'VarianceMap.{what}: self and ref must have the SAME coarse-graining (self:'
                f' L={self.L}, ref: L={ref.L}). The covering constraint is elementwise between'
                ' the two matrices, so mixing granularities is not a valid step.')
        if ref.is_factored:
            return np.asarray(ref.dense(force=True))
        return np.asarray(ref.A, dtype=np.float64)

    def _fine_labels(self):
        """The group index of every fine alpha, as one (nalpha,) array.

        Assembled blockwise to bound the temporaries. The W-step's majorization needs it because
        its objective is a sum over FINE rows with Q row-duplicated.
        """
        out = np.empty(self.nalpha, dtype=np.int64)
        for start in range(0, self.nalpha, self._ALPHA_BLOCK):
            stop = min(start + self._ALPHA_BLOCK, self.nalpha)
            out[start:stop] = self.alpha_to_beta_block(start, stop)
        return out

    @staticmethod
    def _cfg_repairs(cfg, axis):
        """True iff 'cfg' selects a repair stage that actually enforces domination.

        A step's is_admissible cannot come from the ``repair=`` kwarg alone: the three repair
        stages are config FIELDS, so a config with none of them selected repairs nothing however
        the kwarg is set, and ``single_shot_repair`` produces knowingly inadmissible output by
        design. The alternative -- trusting the kwarg -- would hand back a map claiming to
        dominate the reference after doing nothing to it, which is the one error the one-sided
        distance exists to prevent.
        """
        return bool((cfg.additive_first or cfg.additive_last
                     or (cfg.resolved_rescale(axis) != 'none'))
                    and not cfg.single_shot_repair)

    def _step_result(self, name, ref, cfg, Q, mid, W, info, t0, repair, qflag, wflag):
        """The new map a step returns, with its history record.

        D is recorded when it can be computed at all, because the per-step distance is the whole
        point of the log and it costs one contraction against the row sums that the caller is
        about to pay for anyway.
        """

        elapsed = time.time() - t0
        adm = bool(ref.is_admissible) if repair else False

        # A repair that RAN is not a repair that SUCCEEDED, and admissibility here was inferred
        # from the first. lp's steps now report 'n_neg_after', a post-repair count of negative
        # product entries; any of those is inadmissible outright, since Abar >= 0 makes P < 0 a
        # violation even where Abar == 0. A repair triple with no additive stage -- which is
        # what for_wstep() selects -- cannot clear one, because a positive row scale cannot
        # change a sign. Without this the map would carry is_admissible = True and
        # get_distance() would return a finite number for a map that underestimates the
        # variance somewhere, which is the single failure the one-sided distance exists to
        # prevent. Refusing the flag makes get_distance() raise instead, with its own message
        # saying how to fix it.
        if adm and int(info.get('n_neg_after', 0)) > 0:
            adm = False
        out = self.replace(Q=Q, mid=mid, W=W, pinned_columns=self.pinned_columns,
                           Q_is_semiorthogonal=bool(qflag), W_is_semiorthogonal=bool(wflag),
                           is_admissible=adm)

        D = float(out.get_distance()) if (adm and (out.y_true is not None)) else None
        # The solver's own figures go in first and the wrapper's names win, because the two
        # levels use different vocabularies for the same two keys: lp calls this step 'Q' and
        # times only its own solve-and-repair. 'Q_raw' is dropped outright -- cfg.stash_raw puts
        # the pre-repair LP point there, and an (nbeta, K) array in the history would be written
        # into the FILE; the caller who asked for it already has it.
        rec = {k: v for (k, v) in info.items() if k not in ('step', 'Q_raw', 'seconds')}
        rec.update(step=name, config=cfg, D=D, seconds=elapsed,
                   lp_step_seconds=info.get('seconds'), repaired=bool(repair))
        return out.replace(history_record=rec)

    def seed_onehot(self, ref, *, block_rows=None):
        """Return a map whose Q is the best ADMISSIBLE ONE-HOT choice for this W: per group, the
        single NONNEGATIVE column of W that covers it most cheaply, scaled just enough to
        dominate. ``is_admissible = ref.is_admissible``, by construction rather than by
        measurement.

        Two jobs, and the second is the one that is easy to skip. It is a decent starting point
        in its own right -- a rescaled envelope, which is the baseline the whole low-rank family
        is trying to beat. And it is the FALLBACK a Q-step returns for any subproblem whose LP
        fails: with_basis() leaves Q at zero, and a zero row survives a repair as a zero row, so
        a step seeded from one turns a solver failure into a silently inadmissible group. One
        failure per ~450 groups costs more D than an entire doubling of the rank.

        Requires at least one nonnegative column of W -- see n_nonneg_cols() -- and raises if
        some group is covered by no column, since a seed that is not feasible is not a seed.
        """

        self._require_factored('seed_onehot')
        Abar = self._lp_reference(ref, 'seed_onehot')
        # The atoms are the columns of W folded through mid, since that is what a one-hot Q
        # actually multiplies. With the usual identity or positive-diagonal mid these are W's
        # own columns and this agrees with n_nonneg_cols().
        W = np.asarray(self.W, dtype=np.float64) @ np.asarray(self.mid, dtype=np.float64).T
        cols = np.flatnonzero((W.min(axis=0) >= 0.0) & (W.max(axis=0) > 0.0))
        if cols.size == 0:
            raise RuntimeError('VarianceMap.seed_onehot: W has no nonnegative column, so no'
                               ' single atom covers anything. Run canonicalize_signs(), or'
                               ' pin_column(basis_envelope_column(ref)).')

        t0 = time.time()
        # The search forms an (nrow, ncol, nfreq) temporary, which is the one array here that is
        # not bounded by the map's own size -- hence the row blocking.
        Wt = np.ascontiguousarray(W[:, cols].T)
        cost_of = W[:, cols].sum(axis=0)
        nb = (max(1, (32 << 20) // (8 * max(1, cols.size * self.nfreq)))
              if (block_rows is None) else int(block_rows))
        Q = np.zeros((self.nbeta, self.factor_rank))

        for start in range(0, self.nbeta, nb):
            stop = min(start + nb, self.nbeta)
            a = Abar[start:stop][:, None, :]
            with np.errstate(divide='ignore', invalid='ignore'):
                # Where a column is zero and the reference is not, the ratio is +inf: that
                # column cannot cover this group at any scale, which is what makes the argmin
                # below a feasibility test as well as a choice.
                r = np.where(a > 0, a / np.where(Wt[None] > 0.0, Wt[None], 0.0), 0.0).max(axis=2)
            cost = np.where(np.isfinite(r), r * cost_of[None, :], np.inf)
            c = np.argmin(cost, axis=1)
            rows = np.arange(stop - start)
            if not np.all(np.isfinite(cost[rows, c])):
                bad = int(np.count_nonzero(~np.isfinite(cost[rows, c])))
                raise RuntimeError(f'VarianceMap.seed_onehot: {bad} groups (first beta='
                                   f'{start + int(np.argmin(np.isfinite(cost[rows, c])))}) are'
                                   ' covered by no nonnegative column of W, so there is no'
                                   ' feasible one-hot point. pin_column() an envelope column,'
                                   ' which covers every group by construction.')
            # The margin is what makes the seed DOMINATE rather than merely equal the reference
            # in the channel that set the ratio, which is the difference between admissible and
            # admissible-up-to-rounding.
            Q[start + rows, cols[c]] = r[rows, c] * (1.0 + 1.0e-12)

        rec = dict(step='seed_onehot', n_nonneg_cols=int(cols.size),
                   seconds=time.time() - t0)
        return self.replace(Q=Q, Q_is_semiorthogonal=False,
                            W_is_semiorthogonal=self.W_is_semiorthogonal,
                            is_admissible=bool(ref.is_admissible), history_record=rec)

    def qstep(self, ref, *, cfg=None, repair=True, solve_fn=None, groups=None, q_lower=None,
              workers=None, progress=False):
        """One Q-step: hold W fixed and solve one covering LP per group for the rows of Q.

        EXACT, not a heuristic: f is strictly increasing, so minimizing D over a group's
        coefficients is exactly minimizing that group's row sum, and the groups are independent.
        Given W, no better Q exists -- and it follows that the step is insensitive to the precise
        shape of f.

        'ref' supplies the right-hand sides, one row per group; it must have the same geometry
        and coarse-graining as self, and may be dense or factored. A LOOSE ref is safe rather
        than wrong: it makes the covering constraint stricter, so the result is admissible but
        suboptimal.

        The result has the same W, the same pinned columns, a new Q, an identity 'mid' (the LP
        chooses the coefficients outright, so a middle matrix has nothing left to say), and

            is_admissible = ref.is_admissible

        -- NOT unconditionally True, and only when a repair really ran. The covering constraint
        is exactly the statement ``self >= ref`` and the repair enforces it exactly, but
        admissibility is transitive: ``self >= ref`` and ``ref >= A_true`` give the conclusion,
        and an uncertified ref certifies nothing. With the usual streamed reference that is True
        and the distinction never shows. The second condition is the config's, not the kwarg's:
        the three repair stages are LpConfig fields, so a config selecting none of them repairs
        nothing however ``repair=`` is set.

        Parameters
        ----------
        cfg : LpConfig or None
            None means LpConfig.for_qstep(), which is the research code's shipped settings and
            NOT the best values known; LpConfig.recommended('q') is those. The W-step's config is
            a genuinely different config, not this one with a flag.
        repair : bool
            Apply the admissibility repair before returning. True is the right default. False
            returns the RAW LP point with ``is_admissible=False``, which is what to store when
            several repairs may be tried on one expensive solve -- a Q-step at production scale
            is hundreds of core-hours and a repair is a single blocked pass.
        solve_fn : callable or None
            An alternative to lp.solve_covering_lps with the same signature: the extension point
            for a different solver, a heuristic or a warm start, without reimplementing the
            plumbing that assembles the LPs, applies the repair and rebuilds the map.
        groups : ndarray or None
            Solve only this subset of beta, keeping the rest of Q. The rows of Q are independent
            given W, so slices run in separate processes and combined with replace(Q=...) give
            exactly the Q one process would have produced. Requires repair=False: the repair
            must run AFTER merging, since per slice it loses the step's own violation accounting.
        """

        from . import lp
        self._require_factored('qstep')
        Abar = self._lp_reference(ref, 'qstep')
        cfg = lp.LpConfig.for_qstep() if (cfg is None) else cfg

        t0 = time.time()
        Q, W, info = lp.q_step(Abar, np.asarray(self.W, dtype=np.float64), cfg, Q0=self._QM(),
                               q_lower=q_lower, workers=workers, progress=progress,
                               repair=repair, solve_fn=solve_fn, groups=groups)
        return self._step_result('qstep', ref, cfg, Q, None, W, info, t0,
                                 repair and self._cfg_repairs(cfg, 'rows'),
                                 False, self.W_is_semiorthogonal)

    def wstep(self, ref, *, cfg=None, repair=True, solve_fn=None, channels=None, workers=None,
              progress=False):
        """One W-step: hold Q fixed and solve one majorize-minimize LP per channel for the rows
        of W.

        Needs a majorization because the objective depends on W through column sums while the
        constraint depends on it elementwise, so it does not decouple as written. f is concave,
        so its tangent at the current iterate is a global UPPER bound; minimizing the tangent
        cannot increase the true objective, and the tangent IS linear in W and does decouple over
        channels. The tangent is taken at the FINE rows, which is why ref.y_true is required.

        Columns listed in ``self.pinned_columns`` are excluded from the LP and left unchanged.

        The result has the same Q (unless a repair arm charges the violation to it), a new W, an
        identity 'mid', and ``is_admissible`` from ref and from whether the config repairs at
        all -- see qstep(). 'channels' parallelizes exactly as qstep()'s 'groups' does, with the
        same repair=False requirement.

        cfg defaults to LpConfig.for_wstep(), which is NOT the Q-step's config: the relative
        constraint floor is 400x worse in this direction, the prefix rescue helps nothing here,
        and the repair's damage scales with group count rather than with rank.
        """

        from . import lp
        self._require_factored('wstep')
        Abar = self._lp_reference(ref, 'wstep')
        if ref.y_true is None:
            raise RuntimeError('VarianceMap.wstep: ref has no y_true, so the majorization has'
                               ' nothing to linearize about. y_true is carried by every'
                               ' transformation, so a reference without it came from somewhere'
                               ' that never had it.')
        cfg = lp.LpConfig.for_wstep() if (cfg is None) else cfg

        midI = self._mid_is_identity()
        Qm = self._QM()
        t0 = time.time()
        Q, W, info = lp.w_step(Abar, Qm, np.asarray(ref.y_true, dtype=np.float64),
                               self._fine_labels(), np.asarray(self.W, dtype=np.float64), cfg,
                               pinned=self.pinned_columns, workers=workers, progress=progress,
                               repair=repair, solve_fn=solve_fn, channels=channels)

        # This direction's repair normally scales W, but the 'rows' arm charges the violation to
        # Q instead -- so Q's claim survives only when the array really did come back untouched.
        qflag = self.Q_is_semiorthogonal and midI and np.array_equal(Q, Qm)
        return self._step_result('wstep', ref, cfg, Q, None, W, info, t0,
                                 repair and self._cfg_repairs(cfg, 'cols'), qflag, False)

    def repair(self, ref, *, cfg=None, axis='rows'):
        """Raise self until it dominates 'ref', with no LP at all. Returns a map with the same
        factor_rank and ``is_admissible = ref.is_admissible``.

        Two uses, and the second is the one worth knowing. First, it is the cheap replacement for
        a TERMINAL Q-step: after a W-step the incumbent Q is stale but nearly right, and
        repairing it against the new W costs 100x-2000x less than re-solving for 0-0.2% in D --
        occasionally better, since the Q-step optimizes the LP objective, which bounds D rather
        than being D. Second, it makes "solve once, repair several ways" a real workflow:
        ``qstep(repair=False)`` stores the raw point, and each candidate repair is one blocked
        pass over the product rather than another solve. Both matter because a surprising amount
        of the achievable D lives in the repair -- up to 2.5x by itself -- and which repair wins
        depends on the direction and the scale.

        Parameters
        ----------
        cfg : LpConfig or None
            Defaults to for_qstep() or for_wstep() according to 'axis'. ONLY its repair fields
            are read, and they mean exactly what they mean inside qstep() and wstep(); the three
            stages and why both additive ones are there are documented once, on LpConfig.
        axis : str
            Which factor the multiplicative stage scales when ``cfg.rescale`` is 'auto': 'rows'
            scales rows of Q (the Q-step's axis), 'cols' scales rows of W. An argument rather
            than a config field because a standalone repair has no step to infer the axis from.
            It also selects WHICH additive routine runs -- the two directions genuinely differ
            there, and 'rows' is the only one that lifts to a noise floor as well as to ref.

        Requires a nonnegative column of W when either additive stage is on, and RAISES rather
        than falling back to the multiplicative-only path, because a silent fallback looks
        exactly like the additive repair simply not helping. See n_nonneg_cols(). It also
        refuses a config that selects no repair stage at all, since returning the same map with
        ``is_admissible`` newly set is the one thing this must never do -- a step's
        ``repair=False`` is how you ask for the raw point.

        There is no separate pass against the FINE map. For the usual reference -- the true map's
        max-envelope -- the pivot identity makes one redundant: a row dominating ``ref[beta,:]``
        dominates every fine row in the group, which is what makes scoring possible at CHORD
        scale at all.
        """

        from . import lp
        self._require_factored('repair')
        if axis not in ('rows', 'cols'):
            raise RuntimeError(f"VarianceMap.repair: axis must be 'rows' or 'cols', got"
                               f' {axis!r}')
        Abar = self._lp_reference(ref, 'repair')
        cfg = ((lp.LpConfig.for_qstep() if (axis == 'rows') else lp.LpConfig.for_wstep())
               if (cfg is None) else cfg)

        if not self._cfg_repairs(cfg, axis):
            raise RuntimeError(
                f'VarianceMap.repair: cfg selects no repair stage ({cfg.repair_label!r}), so'
                ' this would return the same map while claiming it dominates the reference.'
                " If the raw point is what you want, that is what a step's repair=False"
                ' returns.')

        additive = bool(cfg.additive_first or cfg.additive_last)
        if additive and (self.n_nonneg_cols() < 1):
            raise RuntimeError(
                'VarianceMap.repair: the additive stage adds a multiple of a NONNEGATIVE column'
                ' of W and this map has none, so it has nothing to lift with. Run'
                ' canonicalize_signs(), or pin_column(basis_envelope_column(ref)), or turn both'
                ' additive stages off.')

        Q, W = np.asarray(self.Q, dtype=np.float64), np.asarray(self.W, dtype=np.float64)
        mid = None if self._mid_is_identity() else np.asarray(self.mid, dtype=np.float64)
        folded = additive and (mid is not None)
        if folded:
            # The additive lift is defined on the COLUMNS of W, so it only raises the product
            # when mid is the identity; the multiplicative stage handles mid exactly, since a row
            # scale commutes with it.
            Q, mid = Q @ mid, None

        t0 = time.time()
        Qn, Wn, info = lp.apply_repair(Q, W, mid, Abar, cfg, axis=axis)
        qflag = self.Q_is_semiorthogonal and (not folded) and np.array_equal(Qn, Q)
        wflag = self.W_is_semiorthogonal and np.array_equal(Wn, W)
        return self._step_result('repair', ref, cfg, Qn, mid, Wn, info, t0, True, qflag, wflag)

    # ---------------- scoring ----------------

    def _check_scorable(self, what):
        if self.y_true is None:
            raise RuntimeError(f'VarianceMap.{what}: y_true is unavailable for this map, so'
                               ' D cannot be computed. y_true is carried by every'
                               ' transformation, so a map without it came from somewhere that'
                               ' never had it (e.g. a W-matrix transplanted from another'
                               ' config).')
        if not self.is_admissible:
            raise RuntimeError(
                f'VarianceMap.{what}: this map is not certified admissible, so it has not'
                ' earned a finite distance. Either build it with a Q-step, or establish the'
                ' flag by measurement: m = m.measure_admissibility(ref).vmap.')


    def get_distance(self):
        """Compute ``D(A_true, self)`` exactly, and return it AS A FLOAT.

        Takes no reference matrix: it needs only self.row_sums() and self.y_true. The
        elementwise half of D's definition -- the ``D = infinity`` test -- is what
        measure_admissibility() is for, and it is redundant for anything built by a Q-step,
        where admissibility is a theorem rather than a measurement.

        Requires ``self.is_admissible`` and ``self.y_true``, and RAISES otherwise, with a
        message saying how to fix it. This is deliberately strict rather than silently
        returning the second branch of the definition: reporting a finite distance for a map
        that underestimates the variance somewhere is the one error the whole one-sided
        distance function exists to prevent.

        The coarse and fine cases differ only in which row alpha maps to, which
        alpha_to_beta_block() answers, so there is one implementation rather than two.
        Accumulates in blocks; nothing of size nalpha is materialized.

        Outputs with y_true below ``YTRUE_FLOOR`` are skipped, and the mean is over the
        remaining ``self.nscored`` rows.

        THE LAST FEW BITS DEPEND ON THE BLOCK SIZE, by a few ulp: a different block size is a
        different pairwise summation grouping. So D is reproducible to ~1e-13 relative, not
        bitwise, and a bit-identity harness comparing two implementations has to hold the
        block size fixed. (get_row_distances(), which does one division per row and no
        summation, IS bitwise reproducible.)

        DO NOT CHANGE THE DEFINITION OF D SILENTLY -- see the module docstring of
        varmap/distance.py.
        """

        self._check_scorable('get_distance')

        s = self.row_sums()
        fsum, nscored = 0.0, 0

        for start in range(0, self.nalpha, self._ALPHA_BLOCK):
            stop = min(start + self._ALPHA_BLOCK, self.nalpha)
            yt = self.y_true[start:stop]
            mask = (yt >= YTRUE_FLOOR)
            if not mask.any():
                continue
            ya = s[self.alpha_to_beta_block(start, stop)]
            fsum += float(np.sum(f(ya[mask] / yt[mask])))
            nscored += int(np.count_nonzero(mask))

        if nscored == 0:
            raise RuntimeError(
                f'VarianceMap.get_distance: all {self.nalpha} outputs have no variance'
                f' (y_true below {YTRUE_FLOOR}), so no output could be scored. A few such'
                ' outputs are expected (a W=0 Detrender2d annihilates the DM=0 output), but a'
                ' map where every output has zero variance means a broken sweep or config.')

        return fsum / nscored


    def get_row_distances(self):
        """The (nalpha,) array of per-row ``f(y_approx/y_true)``: which rows the distance is
        being paid on.

        A row with no variance comes back as NAN, not 0: it is excluded from the mean rather
        than contributing zero to it, and a 0 here would understate D. Use ``np.nanmean`` /
        ``np.nanargmax``.

        Unlike get_distance() this does NOT require ``is_admissible``: it is a diagnostic, not
        a reported score, and the per-row numbers are meaningful whether or not the map is
        admissible overall.

        O(nalpha) memory -- 48 MB at CHORD -- so it is an analysis tool, not something to call
        in a loop.
        """

        if self.y_true is None:
            raise RuntimeError('VarianceMap.get_row_distances: y_true is unavailable for this'
                               ' map')

        s = self.row_sums()
        out = np.full(self.nalpha, np.nan)

        for start in range(0, self.nalpha, self._ALPHA_BLOCK):
            stop = min(start + self._ALPHA_BLOCK, self.nalpha)
            yt = self.y_true[start:stop]
            mask = (yt >= YTRUE_FLOOR)
            if not mask.any():
                continue
            ya = s[self.alpha_to_beta_block(start, stop)]
            out[start:stop][mask] = f(ya[mask] / yt[mask])

        return out


    def estimate_distance(self, *, groups=None, frac=None, rng=None):
        """A SUBSAMPLED estimate of get_distance(), over a subset of rows.

        Returns a DistanceEstimate, and is named for what it is, because an estimate and an
        exact value must never be confused in a table that exists to be compared across
        experiments.

        Not a nicety: a single high-rank cell at CHORD scale is thousands of core-hours, and
        several such cells were settled by sampling 1-32% of groups at a standard error of
        0.5-2%. An honest error bar beats a missing cell.

        Samples coarse groups on a coarse map and fine rows on a fine one; 'groups' indexes
        whichever the map has, and passing it explicitly is what makes PAIRED comparisons
        possible -- see DistanceEstimate.

        Two things the estimator has to get right, both easy to get wrong: groups are NOT all
        the same size while D is a mean over FINE rows (so a plain mean over sampled groups
        would be biased, by 2^l across subbands), and rows below the y_true floor are dropped,
        so the effective denominator is the sampled analogue of nscored. Both fall out of
        scoring each sampled group's fine members explicitly and forming a RATIO of the two
        sampled totals.

        Parameters
        ----------
        groups : ndarray, optional
            Explicit row indices to sample. Overrides 'frac'.
        frac : float, optional
            Fraction of rows to sample uniformly without replacement (default 0.01).
        rng : np.random.Generator, optional
        """

        self._check_scorable('estimate_distance')

        nrows = self.nbeta
        if groups is not None:
            groups = np.unique(np.asarray(groups, dtype=np.int64))
            if groups.size == 0:
                raise RuntimeError('VarianceMap.estimate_distance: empty group list')
            if (groups[0] < 0) or (groups[-1] >= nrows):
                raise RuntimeError(f'VarianceMap.estimate_distance: group indices must be in'
                                   f' [0, {nrows})')
        else:
            frac = 0.01 if (frac is None) else float(frac)
            if not (0.0 < frac <= 1.0):
                raise RuntimeError(f'VarianceMap.estimate_distance: frac={frac} must be in'
                                   ' (0, 1]')
            k = max(1, min(nrows, int(np.ceil(frac * nrows))))
            rng = np.random.default_rng() if (rng is None) else rng
            groups = np.sort(rng.choice(nrows, size=k, replace=False))

        s = self.row_sums()
        k = groups.size

        # Per-group totals: C_i = sum of f() over the group's SCORED fine rows, n_i = how many
        # of them there were. D is the ratio of the two population totals, so the estimator is
        # a ratio estimator rather than a sample mean -- which is exactly what removes the
        # group-size bias, since the sizes enter both sums.
        C = np.zeros(k)
        n = np.zeros(k)

        for i, b in enumerate(groups):
            alpha = (np.array([b], dtype=np.int64) if not self.is_coarse_grained
                     else self.group_members(b))
            yt = self.y_true[alpha]
            mask = (yt >= YTRUE_FLOOR)
            if not mask.any():
                continue
            C[i] = float(np.sum(f(s[b] / yt[mask])))
            n[i] = int(np.count_nonzero(mask))

        nsum = float(n.sum())
        if nsum == 0.0:
            raise RuntimeError('VarianceMap.estimate_distance: no sampled row had any output'
                               f' with variance above {YTRUE_FLOOR}; sample more rows')

        D = float(C.sum()) / nsum

        # Standard error of a ratio estimator under simple random sampling without
        # replacement, with the finite-population correction (1 - k/nrows): exact when
        # k == nrows, which is what makes the frac=1 case agree with get_distance() at
        # stderr 0.
        if k > 1:
            e = C - D * n
            nbar = nsum / k
            var = (1.0 - k/nrows) * float(np.sum(e*e)) / (k * (k-1) * nbar * nbar)
            stderr = float(np.sqrt(max(var, 0.0)))
        else:
            stderr = float('nan')

        return DistanceEstimate(D=D, stderr=stderr, nsampled=k, frac_sampled=k/nrows,
                                nscored=int(nsum), groups=groups)


    def measure_admissibility(self, ref, *, block_rows=None, inflate=False, viol_tol=1.0e-12):
        """Test ``self >= ref`` elementwise, and summarize by
        ``max_r = max over (row,F) of ref/self``. Returns an AdmissibilityResult.

        This is the expensive, elementwise half of D. It is separated from get_distance()
        because for anything built by a Q-step it is redundant -- admissibility is the
        covering constraint the Q-step solves, enforced exactly by its repair step -- but it
        stays first-class, because it is how you validate a solver that is known to lie, how
        you score something that did not come from our pipeline, and how the inflation path
        gets its number.

        'ref' is the stand-in for A_true and must have the same geometry AND the same
        coarse-graining as self. In the coarse case the test is on the COARSE matrices, which
        by the pivot identity -- any row dominating ``Abar[beta,:]`` dominates every fine row
        in the group -- implies admissibility for the fine map. That implication is what makes
        scoring possible at CHORD scale.

        SIGNS. The elementwise test is always well defined; the ratio is not. It is a
        meaningful "factor to inflate self by" only where both sides are positive, so:

          | ``self <= 0 < ref``: ratio = +inf. A real underestimate that NO rescaling repairs.
          | ``ref <= 0``: ratio = 0, so such an element can never become the argmax
            (0/0 included).

        Reporting ``max_r = inf`` rather than raising on a negative matrix element is
        deliberate: scoring a SIGNED candidate is the main reason this method exists, and
        clipping it to zero first would be a modeling choice that changes the matrix.

        NO YTRUE_FLOOR HERE, AND THAT IS DELIBERATE. get_distance() skips outputs whose
        y_true falls below YTRUE_FLOOR; this method never looks at y_true at all. The two
        rules belong in different places: the floor exists because y_approx/y_true is
        UNDEFINED rather than large for an output with no variance, which is a property of
        the distance, whereas admissibility is elementwise domination, where nothing divides
        by y_true and a floor would only weaken the guarantee.

        The consequence is a real difference from the superseded slow_avar.VarMapDistance,
        which skipped a row below the floor for max_r as well as for D0. The two disagree on
        exactly one case: a ref row that is positive but sums below the floor, where self is
        zero. That needs an approximation that is exactly zero where ref is not, which a
        covering LP does not produce -- and empirically does not: rescoring all 26 published
        cells reproduced max_r bit-identically, on maps that do contain such rows. Do not
        "fix" this by adding a floor here.

        WHAT 'ref' HAS TO BE, AND WHY THAT IS NOT CHECKED. The useful ref is the true map
        (fine case) or ``true_map.coarse_grain(L)`` (coarse case). A LOOSE ref -- admissible
        but not minimal -- is SAFE: self must then dominate something larger than A_true, so
        the test can only refuse an approximation that is really admissible, never accept one
        that is secretly underestimating. The dangerous case is a ref that UNDERESTIMATES
        A_true (a buggy reduction, a mean where a max was intended), which is not detectable
        from ref alone; the cheap runtime guard for it is check_ref_covers_y_true().

        Walks both matrices in row blocks, allocating no matrix-sized temporaries, so a
        memmapped ref is fine.

        Parameters
        ----------
        inflate : bool
            Also report the distance of ``self.inflated(max_r * (1 + 1e-12))``, which is
            admissible by construction and tends to D as ``max_r -> 1``. Without it,
            ``D = infinity`` does not distinguish "max_r = 1.02, a 2% rescale fixes it" from
            "max_r = 50, hopeless". Costs a second pass.
        viol_tol : float
            Relative tolerance for the ``nviol`` / ``viol_rows`` counts. ``admissible`` itself
            is the exact test and does not use it.
        """

        t0 = time.time()

        if self.shape != ref.shape:
            raise RuntimeError(f'VarianceMap.measure_admissibility: shape mismatch between'
                               f' self {self.shape} and ref {ref.shape}')
        if (self.is_coarse_grained != ref.is_coarse_grained) or (self.L != ref.L):
            raise RuntimeError(
                f'VarianceMap.measure_admissibility: self and ref must have the SAME'
                f' coarse-graining (self: L={self.L}, ref: L={ref.L}). Mixing a coarse ref'
                ' with a fine self is not a valid test.')

        nb = self.default_block_rows() if (block_rows is None) else int(block_rows)
        nb = max(1, min(nb, self.nbeta))

        max_r, argmax_r = 0.0, (0, 0)
        nviol, viol_rows, nneg_self = 0, 0, 0
        row_max_r = np.zeros(self.nbeta)
        admissible = True

        for start in range(0, self.nbeta, nb):
            stop = min(start + nb, self.nbeta)
            S = self.rows(start, stop)
            T = ref.rows(start, stop)

            admissible = admissible and bool(np.all(S >= T))
            nneg_self += int(np.count_nonzero(S < 0))

            # Sign conventions, spelled out above. Note the 'ref > 0' guard has to come first:
            # with a signed self, ref/self is negative rather than large where self < 0 < ref,
            # so the ratio cannot be formed and then patched.
            ratio = np.where(T > 0,
                             np.where(S > 0, T / np.where(S > 0, S, 1.0), np.inf),
                             0.0)

            rmax = ratio.max(axis=1)
            row_max_r[start:stop] = rmax

            # First strict maximum wins, matching the tie-breaking of the reference
            # implementation, so that argmax_r is reproducible across block sizes.
            i = int(np.argmax(rmax))
            if rmax[i] > max_r:
                max_r = float(rmax[i])
                argmax_r = (start + i, int(np.argmax(ratio[i])))

            # Relative to |ref|, so that the test behaves the same way for a signed ref as for
            # a nonnegative one (scaling S by (1+tol) would loosen in the wrong direction
            # wherever S < 0).
            bad = (T - S) > viol_tol * np.abs(T)
            nviol += int(np.count_nonzero(bad))
            viol_rows += int(np.count_nonzero(bad.any(axis=1)))

        total_elements = self.nbeta * self.nfreq
        nworst = min(self._N_WORST_ROWS, self.nbeta)
        part = np.argpartition(row_max_r, -nworst)[-nworst:]
        worst = part[np.argsort(row_max_r[part])[::-1]]

        vmap = self.replace(is_admissible=admissible,
                            history_record=dict(step='measure_admissibility',
                                                admissible=admissible, max_r=max_r,
                                                nviol=nviol, seconds=time.time()-t0))

        inflation, D_inflated = None, None
        if inflate:
            inflation = float(max_r * (1.0 + 1.0e-12)) if (max_r > 1.0) else 1.0
            if np.isfinite(inflation):
                # The inflated map dominates ref by construction, so it is admissible in the
                # same sense the '.vmap' field is: with ref taken as the stand-in for A_true.
                D_inflated = self.inflated(inflation).replace(is_admissible=True).get_distance()
            else:
                # max_r = inf: a zero (or negative) self where ref is positive, which no
                # rescaling repairs. Must not build the inflated map -- inf * 0 is nan, and
                # the array would be discarded anyway.
                D_inflated = np.inf

        return AdmissibilityResult(
            admissible=admissible, max_r=max_r, argmax_r=argmax_r, nviol=nviol,
            viol_frac=(nviol / total_elements) if total_elements else 0.0,
            viol_rows=viol_rows, worst_rows=worst, total_elements=total_elements,
            nneg_self=nneg_self, vmap=vmap, inflation=inflation, D_inflated=D_inflated,
            seconds=time.time() - t0)


    def check_ref_covers_y_true(self):
        """Sanity-check that this coarse map is a max-envelope and not, say, a mean.

        For a coarse ref, ``sum_F ref[beta,F]`` must dominate ``max over alpha in beta of
        y_true[alpha]``: a max-envelope dominates every member elementwise, hence every
        member's row sum. A violation PROVES the reduction is broken.

        This is the one runtime check available on the property the whole scalable path rests
        on -- that ref does not UNDERESTIMATE A_true -- and it costs one pass over the row
        sums and y_true. It is NECESSARY, NOT SUFFICIENT: the reduction can be wrong in ways
        this does not see, which is why coarse_grain() is also tested against a dense
        reduction on a config small enough to form one.

        WHO SHOULD CALL IT: whatever BUILDS a reference, once, at the point it is built. Not
        the steps -- they run in a loop, and this is a property of ref rather than of any step,
        so a call inside qstep() would pay for it per iteration and still not cover a reference
        that never reaches a step. A loose reference is safe by construction (it only makes the
        covering constraint stricter); one that underestimates is the failure this exists to
        catch, and nothing downstream can detect it.

        Raises on failure; returns the worst (smallest) ratio otherwise.
        """

        if not self.is_coarse_grained:
            raise RuntimeError('VarianceMap.check_ref_covers_y_true: only meaningful for a'
                               ' coarse-grained map')
        if self.y_true is None:
            raise RuntimeError('VarianceMap.check_ref_covers_y_true: y_true is unavailable')

        s = self.row_sums()
        ymax = np.zeros(self.nbeta)
        for start in range(0, self.nalpha, self._ALPHA_BLOCK):
            stop = min(start + self._ALPHA_BLOCK, self.nalpha)
            lab = self.alpha_to_beta_block(start, stop)
            order = np.argsort(lab, kind='stable')
            uniq, first = np.unique(lab[order], return_index=True)
            seg = np.maximum.reduceat(self.y_true[start:stop][order], first)
            ymax[uniq] = np.maximum(ymax[uniq], seg)

        with np.errstate(divide='ignore', invalid='ignore'):
            ratio = np.where(ymax > 0, s / ymax, np.inf)
        i = int(np.argmin(ratio))

        # The exact-arithmetic invariant is ratio >= 1, so the tolerance is not slack in the
        # property -- it is roundoff in how the two sides are summed. sum_F ref[beta,F] adds
        # nfreq terms in one order; y_true[alpha] added them in another, during the sweep that
        # produced it. On the real CHORD map (nfreq = 28160) the tightest group comes out at
        # 1 - 2.9e-14, so a bare '< 1.0' test would reject a correct map. Keep the margin: it
        # is ~5 decades below the smallest real violation this is meant to catch, since a mean
        # where a max was intended is off by a factor, not by ulps.
        if ratio[i] < 1.0 - 1.0e-9:
            raise RuntimeError(
                f'VarianceMap.check_ref_covers_y_true: group beta={i} has row sum {s[i]:.6g},'
                f' below the largest member row sum y_true = {ymax[i]:.6g} (ratio'
                f' {ratio[i]:.6g}). A max-envelope cannot do this, so the coarse-graining that'
                ' produced this map is wrong -- a mean where a max was intended, or a broken'
                ' index convention.')

        return float(ratio[i])


    # ---------------- I/O ----------------
    #
    # The format itself lives in varmap/asdf_io.py, which these forward to. It is imported
    # inside the methods rather than at module scope because it imports this module back.

    def write_asdf(self, filename, *, provenance=None):
        """Write this map to 'filename', as a variance-map file holding one tree.

        Same format as VarianceMultiMap.write_asdf()'s, with a one-element tree list. Read
        it back with ``VarianceMap.from_asdf(filename, itree)``; it is also a complete
        multimap file when the config has a single dedispersion tree, and not otherwise.
        """
        from .asdf_io import write_map
        write_map(self, filename, provenance=provenance)


    @classmethod
    def from_asdf(cls, filename, itree=0):
        """Read one tree out of a variance-map file, eagerly.

        Unlike VarianceMultiMap.from_asdf() this does not require the file to cover every
        tree of its config.
        """
        from .asdf_io import read_map
        return read_map(filename, itree)
