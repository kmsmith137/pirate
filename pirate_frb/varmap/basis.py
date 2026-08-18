"""Ways of building a W-matrix, and the one-call initialization that composes them.

A truncated SVD of a map is naturally a METHOD, and it is ``VarianceMap.svd()``. Every other
way of building a basis is a module-level FUNCTION returning a bare ``(nfreq, K)`` array, fed
to ``VarianceMap.with_basis()``. That keeps the class small and makes "try a different basis"
a one-line change rather than a subclass.

WHICH ONE TO USE, in one line each, from the campaign written up in notes/variance_map.tex:

  basis_svd               the recommended default
  basis_greedy_envelope   still WINS below rank ~8; the incumbent everywhere else until the
                          SVD superseded it. O(nbeta^2.4), which is why it lost
  basis_envelope_column   one column, nonnegative by construction. This is what pinning uses,
                          and it is on the critical path rather than being an extra
  basis_pivoted_qr        a column-subset selection: cheap, nonnegative for free, NEVER MEASURED
  basis_random            the control. 1.2x-2.0x worse than a good basis, NOT within 20% -- so
                          the choice of basis genuinely matters, and this is the number that
                          proves it

TRANSFER IS A FIRST-CLASS USE OF with_basis(), not an afterthought. A W built for one tree,
config or subband count and transplanted to another starts out 3x-19x worse -- but ONE W-step
brings it to within 3-9% of a purpose-built one, which is the cheapest known route to a good
basis at a new config. The one axis where transfer does not work is CHANNEL COUNT: two
interpolation schemes and a principled preconditioner all landed in the same place, so treat a
foreign nfreq as a seed, not a transfer.

TWO CONSTRUCTORS THAT ARE NAMED IN THE DESIGN AND ARE NOT HERE, so that their absence is not
mistaken for an oversight:

  - the ANALYTIC no-detrender basis, i.e. the exact low-rank factorization that exists once the
    Detrender2d is removed. It is the best cheap starting point known (1.0x-1.8x against a
    purpose-built basis with no clustering at all), and it needs the analytic-variance
    machinery of notes/variance_map.tex evaluated with the detrender off, which nothing in this
    package can do yet. Its natural home is next to the sweep.
  - a CUR / leverage-score selection. Proposed, never measured, and unlike basis_pivoted_qr it
    is a family rather than a definite algorithm, so there is nothing to check a choice of
    variant against.
"""

import time

import numpy as np

from .distance import YTRUE_FLOOR, f


####################################   the constructors   ####################################


def basis_svd(ref, factor_rank, *, shape_normalize=None, **kwargs):
    """The leading 'factor_rank' right singular vectors of 'ref', as an ``(nfreq, K)`` array.

    A thin wrapper over ``ref.svd()``, which is where the real work and the options are; it
    exists so that "which basis" reads the same way for every member of this family.

    'shape_normalize' defaults to "choose by rank": the unit-sum SHAPE matrix above rank 32 and
    the raw matrix below it, which is the measured crossover. Column scaling is NOT a parameter
    here -- it is ``rescale_columns()``, a step, for reasons its docstring gives.
    """
    return np.ascontiguousarray(
        np.asarray(ref.svd(factor_rank, shape_normalize=shape_normalize, **kwargs).W))


def basis_envelope_column(ref, *, block_rows=None):
    """The single column ``w[F] = max over rows of ref[.,F]``, as an ``(nfreq,)`` array.

    Nonnegative by construction whenever ref is, positive exactly where some group needs it,
    one streaming pass, and the TIGHTEST single vector that dominates every group. That last
    property is what makes it a certificate: ``q = c e_0`` is feasible for EVERY group whatever
    the other columns of W are, so an LP that would otherwise come back infeasible has a point
    to return, and the additive repair has a column to lift with.

    This is what pin_column() is normally given.
    """
    nb = ref.default_block_rows() if (block_rows is None) else int(block_rows)
    nb = max(1, min(nb, ref.nbeta))
    out = np.full(ref.nfreq, -np.inf)
    for start in range(0, ref.nbeta, nb):
        stop = min(start + nb, ref.nbeta)
        np.fmax(out, ref.rows(start, stop).max(axis=0), out=out)
    return np.ascontiguousarray(out)


def basis_random(ref, factor_rank, *, rng, kind='smooth'):
    """A random ``(nfreq, K)`` basis: the CONTROL, and worth having as one.

    It is 1.2x-2.0x worse than a good basis rather than within 20%, which is the measurement
    that says the choice of basis genuinely matters and the problem does not collapse to "pick
    any spanning set". Report it next to a new constructor, not instead of one.

    'kind' is 'smooth' (nonnegative Gaussian bumps at random centres and widths -- a
    random-feature model, and the control the campaign actually ran) or 'iid'
    (``|N(0,1)|`` entries). Both are nonnegative, so either can seed a Q-step on its own.
    """
    K = int(factor_rank)
    if K < 1:
        raise RuntimeError(f'varmap.basis basis_random: factor_rank={factor_rank} must be >= 1')
    if kind == 'iid':
        return np.ascontiguousarray(np.abs(rng.standard_normal((ref.nfreq, K))))
    if kind != 'smooth':
        raise RuntimeError(f"varmap.basis basis_random: kind={kind!r} is not 'smooth' or 'iid'")

    x = (np.arange(ref.nfreq) + 0.5) / ref.nfreq
    centre = rng.random(K)
    width = 10.0 ** rng.uniform(-1.5, 0.0, size=K)
    return np.ascontiguousarray(
        np.exp(-0.5 * ((x[:, None] - centre[None, :]) / width[None, :])**2))


def basis_pivoted_qr(ref, factor_rank, *, max_bytes=1 << 31):
    """A column-subset selection: the 'factor_rank' rows of 'ref' that a column-pivoted QR
    picks out, as an ``(nfreq, K)`` array.

    Cheap, and it inherits two properties from the data for free -- every atom is an actual
    group's row, so the basis is NONNEGATIVE (hence usable by the additive repair and the
    one-hot seed with no canonicalization) and every atom is in the range of the map. Whether
    that is worth anything against an SVD is UNMEASURED; it is here so that the experiment is a
    call rather than a piece of work.

    Needs the dense matrix, so it refuses above 'max_bytes'.
    """
    from scipy.linalg import qr

    K = int(factor_rank)
    if not (1 <= K <= ref.nbeta):
        raise RuntimeError(f'varmap.basis basis_pivoted_qr: factor_rank={K} is out of range'
                           f' [1, {ref.nbeta}]')
    A = ref.dense(max_bytes=max_bytes)
    # Pivoting on A.T selects COLUMNS of A.T, i.e. rows of A, which is what a W column is.
    piv = qr(np.ascontiguousarray(A.T), pivoting=True, mode='economic')[2]
    return np.ascontiguousarray(np.asarray(A)[piv[:K]].T)


####################################   the greedy envelope   ####################################


class _AgglomerativeEnvelope:
    """Greedy agglomerative clustering of a map's rows under the distance function itself.

    Start with one cluster per row, repeatedly merge the pair whose merge costs least, and read
    off the cluster max-envelopes at any rank. The objective merged on is D's own kernel f, not
    a Euclidean distance, which is what made this the incumbent basis for most of the campaign.

    Cost is O(K0^2 * nfreq) time and O(K0^2) memory in the number of starting rows, and that --
    not quality -- is why the SVD superseded it. Kept because it still wins below rank ~8.

    'y' and 'labels' are what tie the objective to the FINE rows: a cluster's cost is summed
    over the fine outputs inside it, so groups of different sizes are weighted correctly.
    """

    def __init__(self, Abar, y, labels, verbose=False):
        Abar = np.ascontiguousarray(np.asarray(Abar, dtype=np.float64))
        y = np.asarray(y, dtype=np.float64)
        labels = np.asarray(labels, dtype=np.int64)
        K0 = Abar.shape[0]
        if (labels.ndim != 1) or (labels.size != y.size):
            raise RuntimeError('varmap.basis: y and labels must have the same length (they are'
                               ' both indexed by fine alpha)')
        if (labels.size == 0) or (labels.min() < 0) or (int(labels.max()) >= K0):
            raise RuntimeError('varmap.basis: labels do not index the rows of the map')

        self.Abar = Abar
        self.K0 = K0
        self.nalpha = y.size
        self.y = y

        self._W = Abar.copy()                       # (K0, nfreq), one envelope per cluster
        self._S = self._W.sum(axis=1)
        self._clu = labels.copy()
        self._member_y = [y[self._clu == c] for c in range(K0)]
        self._cost = np.array([f(self._S[c] / self._member_y[c]).sum() for c in range(K0)])
        self._active = np.ones(K0, dtype=bool)

        # Dlt[i,j] = the increase in sum_c cost[c] from merging i and j. float32 halves the
        # memory and is plenty for RANKING candidate merges; the objective itself is recomputed
        # in float64 whenever a merge is taken.
        self._Dlt = np.full((K0, K0), np.inf, dtype=np.float32)
        self._best_val = np.full(K0, np.inf)
        self._best_idx = np.zeros(K0, dtype=np.int64)

        act = np.arange(K0)
        for i in range(K0):
            d = self._row_deltas(i, act)
            self._Dlt[i, act] = d
            self._best_val[i] = d.min()
            self._best_idx[i] = act[np.argmin(d)]

        self.merges = []
        self.objective = {K0: float(self._cost.sum() / self.nalpha)}
        self._run(verbose)

    def _row_deltas(self, i, act):
        """Delta-cost of merging cluster i with each active cluster (+inf for i itself)."""

        Smax = np.maximum(self._W[i], self._W[act]).sum(axis=1)         # (nact,)

        # What i's own members cost after the merge.
        yi = self._member_y[i]
        ci = f(Smax[:, None] / yi[None, :]).sum(axis=1)

        # What each candidate j's members cost after the merge. That is ragged over j, so it is
        # accumulated over ROWS instead: every fine row belongs to exactly one active cluster,
        # so a single bincount gives every j at once.
        pos = np.empty(self.K0, dtype=np.int64)
        pos[act] = np.arange(act.size)
        prow = pos[self._clu]
        cj = np.bincount(prow, weights=f(Smax[prow] / self.y), minlength=act.size)

        d = ci + cj - self._cost[i] - self._cost[act]
        d[pos[i]] = np.inf
        return d

    def _run(self, verbose):
        t0 = time.time()
        nact = self.K0

        while nact > 1:
            act = np.flatnonzero(self._active)
            i = act[np.argmin(self._best_val[act])]
            j = int(self._best_idx[i])
            if (j == i) or (not self._active[j]):
                raise RuntimeError('varmap.basis: internal error, stale best_idx')

            self._W[i] = np.maximum(self._W[i], self._W[j])
            self._S[i] = self._W[i].sum()
            self._member_y[i] = np.concatenate([self._member_y[i], self._member_y[j]])
            self._member_y[j] = None
            self._clu[self._clu == j] = i
            self._cost[i] = f(self._S[i] / self._member_y[i]).sum()
            self._active[j] = False
            self._best_val[j] = np.inf
            self.merges.append((int(i), int(j)))
            nact -= 1

            act = np.flatnonzero(self._active)
            self.objective[nact] = float(self._cost[act].sum() / self.nalpha)
            if nact == 1:
                break

            d = self._row_deltas(i, act)
            self._Dlt[i, act] = d
            self._Dlt[act, i] = d
            self._best_val[i] = d.min()
            self._best_idx[i] = act[np.argmin(d)]

            # Only rows whose cached best pointed at i or j are stale; for everyone else the one
            # new candidate is i, a single comparison. A rescan reads the cached Dlt row, so it
            # costs O(nact) rather than O(nact*nfreq) -- which is what keeps this at K0^2.4
            # rather than K0^3.
            others = act[act != i]
            stale = others[(self._best_idx[others] == i) | (self._best_idx[others] == j)]
            for k in stale:
                row = self._Dlt[k, act]
                self._best_val[k] = row.min()
                self._best_idx[k] = act[np.argmin(row)]

            fresh = others[self._best_idx[others] != i]
            dv = self._Dlt[fresh, i]
            better = dv < self._best_val[fresh]
            self._best_val[fresh[better]] = dv[better]
            self._best_idx[fresh[better]] = i

            if verbose and (nact % 500 == 0):
                print(f'  varmap.basis: {nact} clusters, J={self.objective[nact]:.6g},'
                      f' {time.time()-t0:.1f} s', flush=True)

        self.seconds = time.time() - t0

    def roots(self, K):
        """(K0,) array giving each starting row's cluster at K clusters, by replaying the merge
        sequence."""
        K = int(K)
        parent = np.arange(self.K0)
        n = self.K0
        for (i, j) in self.merges:
            if n <= K:
                break
            parent[j] = i
            n -= 1
        root = np.arange(self.K0)
        for c in range(self.K0):
            r = c
            while parent[r] != r:
                r = parent[r]
            root[c] = r
        return root

    def basis(self, K):
        """The ``(nfreq, K)`` matrix of cluster max-envelopes at K clusters.

        Rebuilt from the stored matrix rather than from the incrementally maintained envelopes,
        so it is an independent check on the merge bookkeeping.
        """
        root = self.roots(K)
        keep = np.unique(root)
        pos = np.zeros(int(root.max()) + 1, dtype=np.int64)
        pos[keep] = np.arange(keep.size)
        W = np.zeros((self.Abar.shape[1], keep.size))
        np.maximum.at(W.T, pos[root], self.Abar)
        return np.ascontiguousarray(W)


def basis_greedy_envelope(ref, factor_rank, *, on_shapes=True, tree=None, max_rows=32768,
                          verbose=False):
    """Agglomerative max-envelope clustering of 'ref', as an ``(nfreq, K)`` array. Nonnegative
    whenever ref is.

    Superseded by the SVD in general, but it still WINS at rank <= 8, so it is not a historical
    curiosity. Cost is what lost it: O(nbeta^2.4) against the SVD's seconds, and O(nbeta^2)
    memory, so it refuses above 'max_rows' rows rather than trying.

    'on_shapes' clusters the unit-sum SHAPES of ref's rows rather than the rows themselves, and
    it is the default because the Q-step is exactly scale-invariant in each atom -- it takes the
    best nonnegative COMBINATION of atoms, so a greedy that merges on raw rows is optimizing an
    objective the step does not have. Worth 1.13x-1.40x at rank <= 64 on every detrended map.

    A rank SWEEP should not call this per rank: the clustering does not depend on the rank, so
    build the merge tree once with greedy_envelope_tree() and pass it as 'tree' (or call
    ``tree.basis(K)`` directly).
    """
    if tree is None:
        tree = greedy_envelope_tree(ref, on_shapes=on_shapes, max_rows=max_rows,
                                    verbose=verbose)
    elif bool(tree.on_shapes) != bool(on_shapes):
        # A reused tree carries the space it was clustered in, and the two give genuinely
        # different atoms. Silently honouring the tree would make the argument a no-op exactly
        # when a sweep is comparing the two.
        raise RuntimeError(f'varmap.basis basis_greedy_envelope: this tree was built with'
                           f' on_shapes={tree.on_shapes}, not {bool(on_shapes)}. Build one per'
                           ' space.')
    K = int(factor_rank)
    if not (1 <= K <= ref.nbeta):
        raise RuntimeError(f'varmap.basis basis_greedy_envelope: factor_rank={K} is out of'
                           f' range [1, {ref.nbeta}]')
    return tree.basis(K)


def greedy_envelope_tree(ref, *, on_shapes=True, max_rows=32768, verbose=False):
    """The merge tree behind basis_greedy_envelope(), built once and reusable at every rank.

    The O(nbeta^2) work is here, and pulling a basis out of the result is O(nbeta * nfreq), so a
    rank sweep should build this once and call ``tree.basis(K)`` per rank.
    """
    if ref.nbeta > int(max_rows):
        raise RuntimeError(
            f'varmap.basis greedy_envelope_tree: this map has {ref.nbeta} rows, over the'
            f' {max_rows} limit. The clustering is O(nbeta^2.4) in time and O(nbeta^2) in'
            ' memory, which is why the SVD superseded it; coarse-grain further, or use'
            ' basis_svd().')
    if ref.y_true is None:
        raise RuntimeError('varmap.basis greedy_envelope_tree: ref has no y_true, and the merge'
                           ' objective is a sum over the FINE rows inside each cluster')

    A = np.asarray(ref.dense(force=True), dtype=np.float64)
    if on_shapes:
        s = A.sum(axis=1)
        A = A / np.where(s > 0.0, s, 1.0)[:, None]
        # With unit-sum shapes the row sums carry no information, so every fine row is weighted
        # the same and only the group SIZES enter -- which they still do, through 'labels'.
        y = np.ones(ref.nalpha)
    else:
        y = np.asarray(ref.y_true, dtype=np.float64)

    labels = np.empty(ref.nalpha, dtype=np.int64)
    for lo in range(0, ref.nalpha, ref._ALPHA_BLOCK):
        hi = min(lo + ref._ALPHA_BLOCK, ref.nalpha)
        labels[lo:hi] = ref.alpha_to_beta_block(lo, hi)

    # Rows with no variance are dropped, for the same reason get_distance() skips them: they do
    # not contribute to D, so a merge must not be charged for them -- and 1/y_true is ~1e14
    # there rather than large, so they would own the objective outright. A group all of whose
    # fine rows are unscored survives as a cluster with zero cost, which is correct: merging it
    # into another costs only what raising the OTHER group's envelope costs.
    scored = np.asarray(ref.y_true, dtype=np.float64) >= YTRUE_FLOOR
    tree = _AgglomerativeEnvelope(A, y[scored], labels[scored], verbose=verbose)
    tree.on_shapes = bool(on_shapes)
    return tree


####################################   the initialization   ####################################


def svd_init(ref, factor_rank, *, shape_normalize=None, pin_envelope=False,
             rescale_columns=True, cfg=None, **step_kwargs):
    """The campaign-2 initialization in one call:

        ref.svd(K).canonicalize_signs().rescale_columns().qstep(ref)

    and the result is an ADMISSIBLE rank-K map, which is what makes it worth a name: the SVD on
    its own is not.

    ``canonicalize_signs()`` is in there and is NOT optional. numpy's per-mode sign is
    arbitrary, so a raw SVD basis has zero nonnegative columns, and the additive repair, the
    one-hot seed and the LP's feasibility certificate all need one.

    Parameters
    ----------
    rescale_columns : bool
        Unit column norm, the best setting measured (up to 1.49x, for reasons nobody
        understands -- see VarianceMap.rescale_columns). On by default here because this
        wrapper is for the common case; pass False for the bare truncation.
    pin_envelope : bool
        Reserve one column for basis_envelope_column(ref), which makes a nonnegative column
        guaranteed rather than merely likely, and exclude it from later W-steps. It replaces
        the last free column, so the rank is unchanged and a pinned-versus-not comparison is
        fair. Costs a median +0.23 columns of rank, which is why it is not the default.
    shape_normalize, cfg, **step_kwargs
        Passed to svd() and qstep(). cfg defaults to LpConfig.for_qstep().

    NOT in the chain, deliberately: ``seed_onehot(ref)``. The Q-step returns the incumbent Q for
    any subproblem whose LP fails, and the SVD's own Q is not admissible, so a run that expects
    failures should insert it -- it is one call and it makes the fallback certified. It is left
    out here because this wrapper is the documented chain above and nothing more.

    Everything here is three lines at a call site, and writing them out is better whenever the
    schedule is anything but the common case -- there is deliberately no driver.
    """
    init = ref.svd(factor_rank, shape_normalize=shape_normalize).canonicalize_signs()
    if pin_envelope:
        init = init.pin_column(basis_envelope_column(ref))
    if rescale_columns:
        init = init.rescale_columns()
    return init.qstep(ref, cfg=cfg, **step_kwargs)


####################################   cheap predictors   ####################################


def spectrum_effective_rank(vmap, threshold=1.0e-2, *, max_bytes=1 << 31):
    """The number of singular values of 'vmap' above 'threshold' times the largest.

    USE IT TO ORDER CONFIGS BY ACHIEVABLE D, NOT TO PREDICT THE LEVEL. It correlates with
    achieved D at Spearman +0.97 to +0.99 across 29 maps, but says nothing about the value,
    because D is paid on a group's WORST channel while a spectrum measures RMS.

    The reason to prefer this over the obvious alternative is worth stating, because the obvious
    alternative is free: the coarse-graining FLOOR -- the best any coarse-assigned method could
    achieve at a given L -- is a BOUND, not a ranking. It orders detrender half-width perfectly
    and the detrender knot vector not at all: one config with a 19x better floor came out 1.3x
    WORSE in achieved D. So report the floor as a bound and rank with this.
    """
    A = np.asarray(vmap.dense(max_bytes=max_bytes), dtype=np.float64)
    s = np.linalg.svd(A, compute_uv=False)
    if (s.size == 0) or (s[0] <= 0.0):
        return 0
    return int(np.count_nonzero(s >= float(threshold) * s[0]))


def shape_cover_statistic(vmap, ref, *, block=None, max_bytes=1 << 31):
    """Per-row cost of covering each of ref's groups from above with a SINGLE row of 'vmap',
    both taken as unit-sum shapes. Returns a length-``ref.nbeta`` array, every entry >= 1.

    This is exactly the rank-1 Q-step, so it measures "how far outside vmap's shape repertoire
    each of ref's groups falls" in the campaign's own objective, at zero LP cost -- seconds,
    against the hours a frontier takes. It reproduced the ranking of nine dictionary-transfer
    questions that had been settled by ~230 LP runs.

    An entry of 1 means that group is already in the repertoire; the summary statistic to quote
    is the median or a high quantile, not the mean, since a single unreachable group sends the
    mean to infinity while costing the LP one extra atom.

    Both maps must have the same channel count -- a dictionary does not transfer across nfreq
    anyway (two interpolation schemes and a principled preconditioner all landed in the same
    place), so resampling one onto the other would be measuring the resampler.
    """
    if vmap.nfreq != ref.nfreq:
        raise RuntimeError(f'varmap.basis shape_cover_statistic: nfreq mismatch'
                           f' ({vmap.nfreq} vs {ref.nfreq}). A basis does not transfer across'
                           ' channel count, so there is nothing this number would mean.')

    def _shapes(m):
        X = np.asarray(m.dense(max_bytes=max_bytes), dtype=np.float64)
        s = X.sum(axis=1)
        return X / np.where(s > 0.0, s, 1.0)[:, None]

    U, W = _shapes(ref), _shapes(vmap)
    # A zero source channel can cover nothing there, so it costs +inf rather than dividing by 0.
    Wsafe = np.where(W > 0.0, W, np.inf)
    n = U.shape[0]
    if block is None:
        block = max(1, int(int(max_bytes) // 8 // max(1, W.shape[0] * W.shape[1])))
    out = np.empty(n)
    for lo in range(0, n, block):
        hi = min(lo + block, n)
        out[lo:hi] = (U[lo:hi, None, :] / Wsafe[None, :, :]).max(axis=2).min(axis=1)
    return out
