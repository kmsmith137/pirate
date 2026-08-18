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
therefore done unconditionally. The two shapes are those results written into the type, where
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
    - ``Q`` (ndarray) -- ``(nbeta, K)`` when factored, else None. Sign-free.
    - ``mid`` (ndarray) -- ``(K, K)`` when factored, else None; the identity when unused. It
      lets Q and W BOTH be semiorthogonal at once (an SVD is ``U diag(s) V^T``, and folding
      s into either factor destroys one of the two properties), and makes rescaling O(K^2)
      instead of O(nbeta*K).
    - ``W`` (ndarray) -- ``(nfreq, K)`` when factored, else None. Sign-free; its columns are
      frequency "atoms".
    - ``factor_rank`` (int) -- K, the number of columns of Q and W; None when dense.
    - ``Q_is_semiorthogonal`` / ``W_is_semiorthogonal`` (bool) -- carried, NOT verified. This
      class never checks them, and never sets either True on its own.
    - ``pinned_columns`` (ndarray) -- int64 column indices of W held fixed by the steps;
      empty by default, and None when dense. Likewise carried rather than interpreted.

    ONLY THE STRUCTURE OF A FACTORIZATION IS ENFORCED HERE -- shapes, a consistent K, dtypes,
    and column indices in range. Whether ``Q^T Q`` really is the identity, whether ``mid`` is
    diagonal, whether a pinned column is nonnegative: none of that is checked, because those
    invariants belong to the steps that establish them. A flag this class cannot check is one
    it carries rather than enforces.

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
            # The inflated map dominates ref by construction, so it is admissible in the same
            # sense the '.vmap' field is: with ref taken as the stand-in for A_true.
            infl = self.inflated(inflation).replace(is_admissible=True)
            D_inflated = infl.get_distance() if np.isfinite(inflation) else np.inf

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
