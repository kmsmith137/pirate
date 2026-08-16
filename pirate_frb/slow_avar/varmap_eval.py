"""Evaluation harness for low-rank approximations to a variance map.

Every experiment which approximates a variance map should report its numbers through this
module, so that results from different experiments are comparable. All distances are computed
by :class:`~pirate_frb.slow_avar.VarMapDistance` -- nothing here reimplements the distance
function D.

The distance is one-sided: D = infinity if ``A_approx[alpha,F] < A_true[alpha,F]`` anywhere,
and otherwise ``D = D0 = mean_alpha f(y_approx/y_true)`` with ``f(x) = (x-1)/(1+x/10)`` and y
the row sum. See the ``VarMapDistance`` docstring, and ``notes/variance_map.tex``
(section "Distance function").

**The low-rank approximation problem is not settled.** We are still exploring algorithms, and
this module is deliberately separate from all of them: it fixes how results are *reported*, not
how they are *computed*, and it makes no assumption about how an approximation was obtained.
Anyone working on this -- human or agent -- should feel free to explore new algorithms, and to
propose changes to this module, to the distance function, or to the recommended algorithm.
Record such proposals in ``research/HIGH_LEVEL.md`` (see the ``ch-research`` skill); do not
change the distance function silently, since that would make new numbers incomparable with old
ones, which is the one thing this module exists to prevent.

Quick start
-----------

::

    from pirate_frb.slow_avar import varmap_eval as ve

    # An approximation is a factored object, not a dense matrix.
    approx = ve.ClusterEnvelope(labels, W)          # or ve.FactoredApprox(Q, W)
    r = ve.evaluate(A_true, approx, name='env-K64')
    print(ve.format_row(r))

    # (rank, distance) frontier for an algorithm, one row per rank:
    rows = ve.frontier(A_true, lambda A, K: my_algo(A, K), [1,2,4,8,16], name='my_algo')
    print(ve.format_table(rows))
    ve.save_json(rows, 'my_algo.json')

:func:`evaluate` returns a plain dict (not an object) so that a frontier is a list of dicts,
which prints as a table and serializes to json with no extra machinery. Keys:

===============  ===========================================================================
name             caller-supplied label, or the approximation's own name
rank             k, the number of factor columns -- taken from the FACTORED form, never
                 from a numerically computed rank of the dense product. None if unknown.
nalpha, nfreq    shape of A_true
nscored          rows that contributed to D0. Less than nalpha when A_true has outputs with
                 no variance, which ``VarMapDistance`` ignores -- see its docstring.
D, D0, max_r     as defined by ``VarMapDistance``
argmax_r         (alpha, F) of the worst underestimate, in GLOBAL row indices
admissible       True iff max_r <= 1, i.e. iff D is finite
D_inflated       D of the approximation rescaled to be admissible; see below
inflation        the scale factor used (1.0 when the approximation was already admissible)
eval_seconds     wall-clock time in this function
===============  ===========================================================================

Inflation
---------

An inadmissible approximation has D = infinity, which does not distinguish "max_r = 1.02, so a
2% rescaling fixes it" from "max_r = 50, hopeless". :func:`evaluate` therefore also reports
``D_inflated``: the distance of ``approx.scaled(max_r * (1 + 1e-12))``, which is admissible by
construction and tends to D0 as max_r -> 1. (The 1e-12 matters: scaling by exactly max_r lands
on the D = infinity boundary, where rounding can leave a residual underestimate.) This puts an
inadmissible algorithm on the same axis as an admissible one, at the cost of a second pass.
Rescaling multiplies Q by a constant, so ``rank`` is unchanged and ``D_inflated`` is an honest
number for the same rank. ``D_inflated`` is infinite only if max_r is (i.e. A_approx has a zero
where A_true does not), which no amount of rescaling can fix.

Memory, and scoring without the dense matrix
--------------------------------------------

Nothing here materializes an (nalpha, nfreq) approximation. :func:`evaluate` walks A_true in
row blocks and asks the approximation for the matching block of rows, so it works on a lazily
memmapped A_true (``read_variance_map(..., lazy=True)``) and on approximations whose dense form
would not fit.

That still requires A_true itself. For a *coarse-assigned* approximation -- one where all
outputs in a group share a coefficient vector -- :func:`evaluate_reduced` scores from the
coarse-grained map alone, so a map far too large to form can still be scored. See its docstring
for exactly what it does and does not check.
"""

import json
import os
import time

import numpy as np

from .VarMapDistance import VarMapDistance, YTRUE_FLOOR
from ..utils import atomic_print


####################################   approximations   ####################################


class LowRankApprox:
    """A_approx = Q @ W.T, with Q of shape (nalpha, k) and W of shape (nfreq, k).

    Subclasses supply rows on demand; the dense product is never needed and is usually not
    formed. Subclass contract: ``rows(start, stop)`` returns a float64 (stop-start, nfreq)
    array, ``scaled(c)`` returns the same approximation multiplied by a scalar, and
    ``factors()`` returns (Q, W) -- which may be expensive, and is not used by the harness.

    ``rank`` is k, the number of factor columns. It is an upper bound on the algebraic rank of
    the product, and is what gets reported: an approximation is a factorization we intend to
    evaluate as a factorization, so the column count is the honest cost, not the rank of the
    dense product.
    """

    def __init__(self, nalpha, nfreq, rank, name=None):
        self.nalpha = int(nalpha)
        self.nfreq = int(nfreq)
        self.rank = int(rank) if (rank is not None) else None
        self.name = name

    @property
    def shape(self):
        return (self.nalpha, self.nfreq)

    def rows(self, start, stop):
        raise NotImplementedError

    def dense(self):
        """The full (nalpha, nfreq) matrix. Only for small cases and for testing."""
        return self.rows(0, self.nalpha)

    def scaled(self, c):
        raise NotImplementedError

    def factors(self):
        raise NotImplementedError

    def __repr__(self):
        return (f'{type(self).__name__}(name={self.name!r}, rank={self.rank},'
                f' shape={self.shape})')


class FactoredApprox(LowRankApprox):
    """The general case: explicit Q (nalpha, k) and W (nfreq, k), with A_approx = Q @ W.T."""

    def __init__(self, Q, W, name=None):
        Q = np.asarray(Q, dtype=np.float64)
        W = np.asarray(W, dtype=np.float64)
        if Q.ndim != 2 or W.ndim != 2:
            raise RuntimeError(f'FactoredApprox: expected 2-d Q,W, got {Q.shape}, {W.shape}')
        if Q.shape[1] != W.shape[1]:
            raise RuntimeError(f'FactoredApprox: Q {Q.shape} and W {W.shape} disagree on k')
        super().__init__(Q.shape[0], W.shape[0], Q.shape[1], name)
        self.Q = Q
        self.W = W

    def rows(self, start, stop):
        return self.Q[start:stop] @ self.W.T

    def scaled(self, c):
        return FactoredApprox(self.Q * float(c), self.W, self.name)

    def factors(self):
        return self.Q, self.W


class ClusterEnvelope(LowRankApprox):
    """A max-envelope approximation: row alpha is ``scale[alpha] * W[:, labels[alpha]]``.

    This is the factored form Q @ W.T with a one-hot (times a scalar) Q, stored as an index
    array so that Q is never materialized.

    The construction is admissible by design when ``W[:,c] = max over the cluster`` of the rows
    assigned to c: a max over a set containing the row itself cannot underestimate it. That
    argument belongs to whatever built the envelope, not to this class -- this class will
    happily represent an inadmissible one, and :func:`evaluate` is what checks.

    Empty clusters are dropped and the labels relabeled at construction, so ``rank`` is the
    number of clusters actually used -- an empty cluster would be a zero column of Q, and
    reporting it would overstate the rank.
    """

    def __init__(self, labels, W, scale=None, name=None):
        labels = np.asarray(labels)
        W = np.asarray(W, dtype=np.float64)
        if labels.ndim != 1 or W.ndim != 2:
            raise RuntimeError(f'ClusterEnvelope: expected 1-d labels and 2-d W,'
                               f' got {labels.shape}, {W.shape}')
        if labels.size == 0:
            raise RuntimeError('ClusterEnvelope: empty labels')
        if labels.min() < 0 or labels.max() >= W.shape[1]:
            raise RuntimeError(f'ClusterEnvelope: labels out of range [0,{W.shape[1]})')

        used, labels = np.unique(labels, return_inverse=True)
        W = W[:, used]

        super().__init__(labels.size, W.shape[0], W.shape[1], name)
        self.labels = np.ascontiguousarray(labels, dtype=np.int32)
        self.W = W
        self.scale = None if (scale is None) else np.asarray(scale, dtype=np.float64)
        if (self.scale is not None) and (self.scale.shape != (self.nalpha,)):
            raise RuntimeError(f'ClusterEnvelope: scale has shape {self.scale.shape},'
                               f' expected ({self.nalpha},)')

        # Row extraction gathers whole rows, so keep a C-contiguous (k, nfreq) copy. W itself
        # is (nfreq, k) because that is the convention of the variance map tex notes and of factors().
        self._Wt = np.ascontiguousarray(W.T)

    def rows(self, start, stop):
        out = self._Wt[self.labels[start:stop]]
        if self.scale is not None:
            out = out * self.scale[start:stop, np.newaxis]
        return out

    def scaled(self, c):
        s = np.full(self.nalpha, float(c)) if (self.scale is None) else (self.scale * float(c))
        return ClusterEnvelope(self.labels, self.W, s, self.name)

    def factors(self):
        """(Q, W) with Q the (nalpha, k) one-hot indicator. Q is dense: only for small cases."""
        Q = np.zeros((self.nalpha, self.rank))
        Q[np.arange(self.nalpha), self.labels] = 1.0 if (self.scale is None) else self.scale
        return Q, self.W

    def cluster_sizes(self):
        return np.bincount(self.labels, minlength=self.rank)


class DenseApprox(LowRankApprox):
    """Wraps an already-dense (nalpha, nfreq) matrix, e.g. a reconstructed SVD truncation.

    ``rank`` must be supplied by the caller if it is to be reported: this class does not
    compute a numerical rank, because a matrix that arrived dense has no factored form to
    be honest about.
    """

    def __init__(self, A, rank=None, name=None):
        A = np.asarray(A, dtype=np.float64)
        if A.ndim != 2:
            raise RuntimeError(f'DenseApprox: expected 2-d array, got shape {A.shape}')
        super().__init__(A.shape[0], A.shape[1], rank, name)
        self.A = A

    def rows(self, start, stop):
        return self.A[start:stop]

    def dense(self):
        return self.A

    def scaled(self, c):
        return DenseApprox(self.A * float(c), self.rank, self.name)


def as_approx(approx, rank=None, name=None):
    """Coerces a LowRankApprox or a bare (nalpha, nfreq) array to a LowRankApprox."""
    if isinstance(approx, LowRankApprox):
        return approx
    return DenseApprox(approx, rank=rank, name=name)


####################################   evaluation   ####################################


# A row block of A_true plus one of A_approx, in float64. 32 MiB each is small enough to stay
# in cache-friendly territory and large enough that the per-call VarMapDistance overhead and
# the per-block memmap read are both negligible.
_BLOCK_BYTES = 32 << 20


def _block_rows(nfreq, block_rows):
    if block_rows is not None:
        return max(1, int(block_rows))
    return max(1, _BLOCK_BYTES // (8 * max(1, int(nfreq))))


def _distance(A_true, approx, block_rows):
    """(D, D0, max_r, argmax_r, nscored), computed by VarMapDistance one row block at a time.

    Combining blocks is exact: D0 is a mean over SCORED rows, so it is re-weighted by each
    block's scored count (not by the block size -- VarMapDistance ignores rows of A_true with
    no variance, see its docstring), and max_r is a max. Tie-breaking for argmax_r matches the
    single-call case because both this loop and VarMapDistance keep the first strict maximum.

    Blocks are evaluated with allow_empty=True, since a block may legitimately contain only
    ignored rows; the "nothing was scored" check is then applied to the whole matrix here.
    """

    nalpha, nfreq = A_true.shape
    nb = _block_rows(nfreq, block_rows)

    fsum = 0.0
    nscored = 0
    max_r = 0.0
    argmax_r = (0, 0)

    for start in range(0, nalpha, nb):
        stop = min(start + nb, nalpha)
        rows_true = np.asarray(A_true[start:stop])
        rows_approx = approx.rows(start, stop)
        try:
            d = VarMapDistance(rows_true, rows_approx, allow_empty=True)
        except RuntimeError as e:
            # VarMapDistance reports a row index within the block it was handed; make the
            # index global, since that is the one the caller can act on.
            raise RuntimeError(f'{e} [rows {start}:{stop} of the full matrix; add {start} to'
                               ' any row index in the message above]') from None
        if d.nscored > 0:
            fsum += d.D0 * d.nscored
            nscored += d.nscored
        if d.max_r > max_r:
            max_r = d.max_r
            argmax_r = (d.argmax_r[0] + start, d.argmax_r[1])

    if nscored == 0:
        raise RuntimeError(
            f'evaluate: all {nalpha} rows of A_true have no variance (row sums below'
            f' {YTRUE_FLOOR}), so no row could be scored. A few such rows are expected'
            ' (a W=0 Detrender2d annihilates the DM=0 output), but a map where every output'
            ' has zero variance means a broken sweep or config.')

    D0 = fsum / nscored
    return (D0 if (max_r <= 1.0) else np.inf), D0, max_r, argmax_r, nscored


def _inflate(distance_fn, approx, max_r):
    """(inflation, D_inflated) for an inadmissible approximation; see the module docstring.

    'distance_fn' maps a scaled approximation to its D.
    """

    if not np.isfinite(max_r):
        # A zero in A_approx where A_true is positive. No rescaling can repair it.
        return np.inf, np.inf

    # Scale by slightly more than max_r. The loop is paranoia about the "slightly": one step
    # is always enough in practice.
    for fudge in (1.0e-12, 1.0e-9, 1.0e-6):
        inflation = max_r * (1.0 + fudge)
        D_inflated = distance_fn(approx.scaled(inflation))
        if np.isfinite(D_inflated):
            return inflation, D_inflated

    return inflation, D_inflated


def evaluate(A_true, approx, *, name=None, rank=None, inflate=True, block_rows=None,
             extra=None):
    """Scores an approximation against A_true. Returns the dict documented in the module
    docstring.

    Args:
      A_true: (nalpha, nfreq) array-like, sliceable by rows. A lazily memmapped map is fine.
      approx: a LowRankApprox, or a dense array (wrapped by as_approx()).
      name: label for the report; defaults to approx.name.
      rank: only used if `approx` is a bare array with no factored form.
      inflate: compute D_inflated. Costs a second pass, and is a no-op if already admissible.
      block_rows: rows per VarMapDistance call. The default targets 32 MiB per block.
      extra: dict merged into the result, for experiment bookkeeping (config name, timings).
    """

    approx = as_approx(approx, rank=rank, name=name)
    A_true = A_true if hasattr(A_true, 'shape') else np.asarray(A_true)

    if len(A_true.shape) != 2:
        raise RuntimeError(f'evaluate: expected 2-d A_true, got shape {A_true.shape}')
    if tuple(A_true.shape) != approx.shape:
        raise RuntimeError(f'evaluate: shape mismatch, A_true {tuple(A_true.shape)} vs'
                           f' approx {approx.shape}')

    t0 = time.time()
    D, D0, max_r, argmax_r, nscored = _distance(A_true, approx, block_rows)

    inflation, D_inflated = 1.0, D
    if (max_r > 1.0) and inflate:
        inflation, D_inflated = _inflate(
            lambda a: _distance(A_true, a, block_rows)[0], approx, max_r)

    r = dict(name=(name if (name is not None) else approx.name),
             rank=approx.rank,
             nalpha=int(A_true.shape[0]),
             nscored=int(nscored),
             nfreq=int(A_true.shape[1]),
             D=float(D),
             D0=float(D0),
             max_r=float(max_r),
             argmax_r=(int(argmax_r[0]), int(argmax_r[1])),
             admissible=bool(max_r <= 1.0),
             D_inflated=float(D_inflated),
             inflation=float(inflation),
             eval_seconds=time.time() - t0)

    if extra:
        r.update(extra)
    return r


####################################   reduced evaluation   ####################################


def _reduced_distance(Abar, y, labels, approx, block_rows):
    """(D, D0, max_r, argmax_r) for a coarse-assigned approximation; see evaluate_reduced()."""

    nbeta, nfreq = Abar.shape

    # Part 1: admissibility, on the REDUCED matrix. This is an ordinary VarMapDistance call
    # comparing (nbeta, nfreq) against (nbeta, nfreq); we keep its max_r and discard its D0,
    # which is a mean over groups rather than over outputs.
    max_r = -np.inf
    argmax_r = (0, 0)
    nb = _block_rows(nfreq, block_rows)

    for start in range(0, nbeta, nb):
        stop = min(start + nb, nbeta)
        rows_true = np.asarray(Abar[start:stop])
        rows_approx = approx.rows(start, stop)
        try:
            d = VarMapDistance(rows_true, rows_approx, allow_empty=True)
        except RuntimeError as e:
            raise RuntimeError(f'{e} [groups {start}:{stop} of Abar; add {start} to any row'
                               ' index in the message above]') from None
        if d.max_r > max_r:
            max_r = d.max_r
            argmax_r = (d.argmax_r[0] + start, d.argmax_r[1])

    # Part 2: D0, over the FINE outputs. The approximate row sum of every output in group beta
    # is s[beta], so D0 = mean_alpha f(s[labels[alpha]] / y[alpha]). Rather than reimplement f,
    # hand VarMapDistance a pair of (nblock, 1) arrays whose row sums are exactly y and s: its
    # D0 is then the quantity we want. (Its max_r for that call compares row sums, not matrix
    # elements, so it is meaningless here and is discarded.)
    s = np.empty(nbeta)
    for start in range(0, nbeta, nb):
        stop = min(start + nb, nbeta)
        s[start:stop] = approx.rows(start, stop).sum(axis=1)

    nalpha = labels.size
    nb2 = _block_rows(1, block_rows)
    fsum = 0.0
    nscored = 0

    for start in range(0, nalpha, nb2):
        stop = min(start + nb2, nalpha)
        yt = np.asarray(y[start:stop], dtype=np.float64).reshape(-1, 1)
        ya = s[labels[start:stop]].reshape(-1, 1)
        try:
            d = VarMapDistance(yt, ya, allow_empty=True)
        except RuntimeError as e:
            raise RuntimeError(f'{e} [outputs {start}:{stop}; the "matrix" here is the column'
                               ' of true row sums y, so a "row" is one output alpha]') from None
        if d.nscored > 0:
            fsum += d.D0 * d.nscored
            nscored += d.nscored

    if nscored == 0:
        raise RuntimeError(
            f'evaluate_reduced: all {nalpha} outputs have no variance (y below {YTRUE_FLOOR}),'
            ' so no output could be scored. A few such outputs are expected (a W=0 Detrender2d'
            ' annihilates the DM=0 output), but a map where every output has zero variance'
            ' means a broken sweep or config.')

    D0 = fsum / nscored
    return (D0 if (max_r <= 1.0) else np.inf), D0, max_r, argmax_r, nscored


def evaluate_reduced(Abar, y, labels, approx, *, name=None, rank=None, inflate=True,
                     block_rows=None, extra=None):
    """Scores a COARSE-ASSIGNED approximation without ever forming the dense variance map.

    A coarse-assigned approximation partitions the outputs alpha into groups beta and gives
    every output in a group the same approximate row, i.e. ``A_approx[alpha,:] =
    approx.rows(beta, beta+1)`` where ``beta = labels[alpha]``. This is the construction of
    ``notes/variance_map.tex`` (section "Proposed algorithm and initial results"), and it
    is also the shape of the production peak-finding weight array.

    For such an approximation the distance is computable from the coarse-grained map alone,
    which matters because the dense map is what the algorithm was designed never to form: at
    CHORD scale ``Abar`` is a few GiB where A is over a terabyte.

    Args:
      Abar: (nbeta, nfreq) coarse-grained map, ``Abar[beta,F] = max over alpha in beta of
        A[alpha,F]``. Row-sliceable; a memmap is fine.
      y: (nalpha,) TRUE row sums of the fine map, ``y[alpha] = sum_F A[alpha,F]``. Accumulated
        in the same streaming pass that builds Abar.
      labels: (nalpha,) int array mapping each output to its group.
      approx: a LowRankApprox with nalpha == nbeta, i.e. one row per GROUP.

    Returns the same dict as :func:`evaluate`, with ``nalpha`` the number of fine outputs, plus
    ``nbeta`` and ``reduced=True``. Note ``argmax_r`` is a **(beta, F)** pair, indexing a group
    rather than an output.

    What this checks, and what it does not
    --------------------------------------

    It checks ``approx >= Abar`` elementwise, and computes D0 over the fine outputs using their
    true row sums. Given a correct Abar, that is exactly equivalent to :func:`evaluate` on the
    lifted fine approximation, because ``max over alpha in beta`` of A equals Abar by
    construction -- the self-test asserts the two agree.

    It does **not** verify that Abar was correctly computed from A. If Abar underestimates
    A[alpha,F] anywhere -- a bug in the reduction, a truncated sweep, a mean where a max was
    intended -- then an approximation this function calls admissible may underestimate the true
    variance, and the guarantee is void. Abar is trusted input. The reduction that produces it
    should be tested separately, against a dense reference on a config small enough to form one.
    """

    Abar = Abar if hasattr(Abar, 'shape') else np.asarray(Abar)
    y = np.asarray(y, dtype=np.float64)
    labels = np.asarray(labels)
    approx = as_approx(approx, rank=rank, name=name)

    if len(Abar.shape) != 2:
        raise RuntimeError(f'evaluate_reduced: expected 2-d Abar, got shape {Abar.shape}')
    if tuple(Abar.shape) != approx.shape:
        raise RuntimeError(f'evaluate_reduced: shape mismatch, Abar {tuple(Abar.shape)} vs'
                           f' approx {approx.shape} (approx must have ONE ROW PER GROUP)')
    if labels.ndim != 1 or y.ndim != 1:
        raise RuntimeError(f'evaluate_reduced: expected 1-d y and labels, got {y.shape},'
                           f' {labels.shape}')
    if labels.size != y.size:
        raise RuntimeError(f'evaluate_reduced: y has {y.size} entries but labels has'
                           f' {labels.size}; both are indexed by output alpha')
    if labels.size == 0:
        raise RuntimeError('evaluate_reduced: empty labels')
    if labels.min() < 0 or labels.max() >= Abar.shape[0]:
        raise RuntimeError(f'evaluate_reduced: labels out of range [0,{Abar.shape[0]})')

    t0 = time.time()
    D, D0, max_r, argmax_r, nscored = _reduced_distance(Abar, y, labels, approx, block_rows)

    inflation, D_inflated = 1.0, D
    if (max_r > 1.0) and inflate:
        inflation, D_inflated = _inflate(
            lambda a: _reduced_distance(Abar, y, labels, a, block_rows)[0], approx, max_r)

    r = dict(name=(name if (name is not None) else approx.name),
             rank=approx.rank,
             nalpha=int(labels.size),
             nscored=int(nscored),
             nbeta=int(Abar.shape[0]),
             nfreq=int(Abar.shape[1]),
             reduced=True,
             D=float(D),
             D0=float(D0),
             max_r=float(max_r),
             argmax_r=(int(argmax_r[0]), int(argmax_r[1])),
             admissible=bool(max_r <= 1.0),
             D_inflated=float(D_inflated),
             inflation=float(inflation),
             eval_seconds=time.time() - t0)

    if extra:
        r.update(extra)
    return r


def reduce_map(A, labels, nbeta=None):
    """(Abar, y) for a fine map A and a grouping: the input :func:`evaluate_reduced` wants.

    ``Abar[beta,F] = max over alpha in beta of A[alpha,F]``, and ``y[alpha] = sum_F A[alpha,F]``.

    This forms Abar from a dense A, so it is for tests and for small maps. At scale the point
    of the reduction is that it can be done one input channel at a time, inside the sweep that
    computes A, without ever holding A.
    """

    A = A if hasattr(A, 'shape') else np.asarray(A)
    labels = np.asarray(labels)
    nalpha, nfreq = A.shape

    if labels.shape != (nalpha,):
        raise RuntimeError(f'reduce_map: labels has shape {labels.shape}, expected ({nalpha},)')

    nbeta = int(labels.max()) + 1 if (nbeta is None) else int(nbeta)
    Abar = np.zeros((nbeta, nfreq))
    y = np.zeros(nalpha)

    nb = _block_rows(nfreq, None)
    for start in range(0, nalpha, nb):
        stop = min(start + nb, nalpha)
        rows = np.asarray(A[start:stop], dtype=np.float64)
        y[start:stop] = rows.sum(axis=1)
        # np.maximum.at is correct but slow; sort-and-reduceat is bit-identical and much
        # faster, and this is the operation that dominates a large reduction.
        lab = labels[start:stop]
        order = np.argsort(lab, kind='stable')
        ls = lab[order]
        bounds = np.searchsorted(ls, np.arange(nbeta + 1))
        for b in range(nbeta):
            lo, hi = bounds[b], bounds[b+1]
            if hi > lo:
                np.maximum(Abar[b], rows[order[lo:hi]].max(axis=0), out=Abar[b])

    return Abar, y


def frontier(A_true, algorithm, ranks, *, name=None, **kwargs):
    """(rank, distance) frontier: calls ``algorithm(A_true, K)`` for each K in `ranks` and
    evaluates the result. Returns a list of dicts, one per K, with two extra keys: 'K' (the
    requested rank, which may exceed the achieved `rank` if the algorithm merged or dropped
    clusters) and 'algo_seconds'.

    An algorithm that naturally produces all ranks in one run (agglomerative merging, say)
    should precompute and pass a closure ``lambda A, K: precomputed.approx(K)``; this function
    makes no assumption about how the algorithm is implemented.

    Extra kwargs go to evaluate().
    """

    rows = []
    for K in ranks:
        t0 = time.time()
        approx = algorithm(A_true, K)
        dt = time.time() - t0
        r = evaluate(A_true, approx, name=name, **kwargs)
        r['K'] = int(K)
        r['algo_seconds'] = dt
        rows.append(r)
    return rows


def row_distances(A_true, approx, block_rows=None):
    """Per-row f(y_approx/y_true), as an (nalpha,) array: which rows the distance is paid on.

    This is the summand of D0, so ``np.nanmean(row_distances(...))`` is D0. Computed by calling
    VarMapDistance once per row, which costs a few microseconds per row of python overhead --
    fine for analysis, not for an inner loop.

    A row with no variance (see the ``VarMapDistance`` docstring) has no distance to report and
    comes back as **nan**, not 0: it is excluded from D0 rather than contributing zero to it,
    and a 0 here would understate the mean. Use ``np.nanmean`` / ``np.nanargmax``.
    """

    approx = as_approx(approx)
    nalpha, nfreq = A_true.shape
    nb = _block_rows(nfreq, block_rows)
    out = np.zeros(nalpha)

    for start in range(0, nalpha, nb):
        stop = min(start + nb, nalpha)
        rows_true = np.asarray(A_true[start:stop])
        rows_approx = approx.rows(start, stop)
        for i in range(stop - start):
            # allow_empty: a single ignored row is a call in which nothing was scored, which
            # is a legitimate per-row outcome rather than the degenerate-map error.
            out[start+i] = VarMapDistance(rows_true[i:i+1], rows_approx[i:i+1],
                                          allow_empty=True).D0

    return out


####################################   reporting   ####################################


_COLUMNS = ('name', 'K', 'rank', 'D', 'D0', 'max_r', 'D_inflated', 'algo_seconds')


def format_table(rows, columns=None):
    """A markdown table of the dicts returned by evaluate()/frontier()."""

    if len(rows) == 0:
        return '(no rows)'
    columns = columns if (columns is not None) else [c for c in _COLUMNS if c in rows[0]]

    def fmt(v):
        if isinstance(v, float):
            return 'inf' if np.isinf(v) else f'{v:.6g}'
        return str(v)

    cells = [[fmt(r.get(c, '')) for c in columns] for r in rows]
    width = [max(len(str(c)), max(len(row[i]) for row in cells))
             for i, c in enumerate(columns)]

    out = ['| ' + ' | '.join(str(c).ljust(width[i]) for i, c in enumerate(columns)) + ' |',
           '|' + '|'.join('-' * (width[i]+2) for i in range(len(columns))) + '|']
    out += ['| ' + ' | '.join(row[i].ljust(width[i]) for i in range(len(columns))) + ' |'
            for row in cells]
    return '\n'.join(out)


def format_row(r):
    """One-line summary of an evaluate() result."""
    a, f = r['argmax_r']
    return (f"{r.get('name')}: rank={r['rank']} D={r['D']:.6g} D0={r['D0']:.6g}"
            f" max_r={r['max_r']:.6g} argmax=(alpha={a},F={f})"
            f" D_inflated={r['D_inflated']:.6g}")


def save_json(rows, path):
    """Writes evaluate()/frontier() results to json (infinities become the string 'inf')."""
    def clean(v):
        if isinstance(v, float) and not np.isfinite(v):
            return 'inf' if (v > 0) else '-inf'
        if isinstance(v, tuple):
            return list(v)
        return v
    with open(path, 'w') as fp:
        json.dump([{k: clean(v) for k, v in r.items()} for r in rows], fp, indent=2)


def load_json(path):
    """Inverse of save_json(): the string infinities become floats again."""
    def restore(v):
        return {'inf': np.inf, '-inf': -np.inf}.get(v, v) if isinstance(v, str) else v
    with open(path) as fp:
        return [{k: restore(v) for k, v in r.items()} for r in json.load(fp)]


####################################   self-test   ####################################


def _reference_matrix(rng, nalpha=201, nfreq=37):
    """A small positive matrix with the features that break things: exact zeros, a large
    dynamic range between rows, and duplicated rows."""

    A = rng.uniform(0.5, 2.0, size=(nalpha, nfreq))
    A *= np.exp(rng.normal(0, 3.0, size=(nalpha, 1)))     # dynamic range ~ e^{+-9}
    A[:, :3] = 0.0                                        # dead channels: zero column block
    A[7, 5:] = 0.0                                        # a row that is nearly all zeros
    A[20:25] = A[19]                                      # exact duplicate rows
    return A


def self_test(verbose=True):
    """Checks the harness against direct VarMapDistance calls, and exercises the edge cases
    that occur in real maps: zeros in A_true, large dynamic range, and a row count that is
    not a multiple of the block size."""

    rng = np.random.default_rng(1234)
    A = _reference_matrix(rng)
    nalpha, nfreq = A.shape

    # --- blockwise evaluation reproduces a single whole-matrix VarMapDistance call ---
    # Three cases: strictly admissible, one planted underestimate, and an approximation with
    # a zero where A_true is positive (max_r = inf).
    for case in ('admissible', 'planted', 'zero'):
        B = A * rng.uniform(1.0, 1.4, size=A.shape)
        if case == 'planted':
            B[123, 11] = A[123, 11] * 0.9
        elif case == 'zero':
            B[5, 30] = 0.0
        ref = VarMapDistance(A, B)
        for nb in (1, 7, nalpha - 1, nalpha, nalpha + 5):     # nalpha % nb != 0 on purpose
            r = evaluate(A, DenseApprox(B, rank=7), block_rows=nb, inflate=False)
            assert abs(r['D0'] - ref.D0) < 1.0e-12 * max(1.0, abs(ref.D0)), (case, nb, r)
            assert r['max_r'] == ref.max_r, (case, nb, r['max_r'], ref.max_r)
            assert r['argmax_r'] == ref.argmax_r, (case, nb, r['argmax_r'], ref.argmax_r)
            assert np.isinf(r['D']) == np.isinf(ref.D), (case, nb, r['D'], ref.D)
            assert np.isinf(r['D']) or (r['D'] == r['D0']), (case, nb, r)
            assert r['rank'] == 7, r
            # 0/0 must not become the argmax: the zero column block is never selected.
            assert r['argmax_r'][1] >= 3, r

    # --- inflation ---
    B = A * rng.uniform(1.0, 1.4, size=A.shape)
    B[123, 11] = A[123, 11] * 0.5
    r = evaluate(A, DenseApprox(B))
    assert not r['admissible'] and np.isinf(r['D']), r
    assert np.isfinite(r['D_inflated']) and r['D_inflated'] > r['D0'], r
    assert abs(r['inflation'] / r['max_r'] - 1.0) < 1.0e-9, r
    # An already-admissible approximation is not touched.
    r2 = evaluate(A, DenseApprox(A * 1.5))
    assert r2['inflation'] == 1.0 and r2['D_inflated'] == r2['D'], r2
    # A zero in A_approx where A_true is positive cannot be repaired by rescaling.
    B2 = A * 1.5
    B2[5, 30] = 0.0
    r3 = evaluate(A, DenseApprox(B2))
    assert np.isinf(r3['max_r']) and np.isinf(r3['D_inflated']), r3

    # --- factored forms: rows(), dense() and factors() agree, and rank comes from k ---
    Q = rng.uniform(0, 1, size=(nalpha, 5))
    W = rng.uniform(0, 1, size=(nfreq, 5))
    fa = FactoredApprox(Q, W, name='fa')
    assert fa.rank == 5 and fa.shape == (nalpha, nfreq)
    assert np.allclose(fa.dense(), Q @ W.T)
    assert np.allclose(fa.rows(3, 9), (Q @ W.T)[3:9])
    assert np.allclose(fa.scaled(2.0).dense(), 2.0 * (Q @ W.T))

    labels = rng.integers(0, 4, size=nalpha)
    Wc = rng.uniform(1, 2, size=(nfreq, 4))
    ce = ClusterEnvelope(labels, Wc, name='ce')
    assert ce.rank == 4 and np.allclose(ce.dense(), Wc.T[labels])
    Qc, Wc2 = ce.factors()
    assert np.allclose(Qc @ Wc2.T, ce.dense())
    scale = rng.uniform(0.1, 1.0, size=nalpha)
    ces = ClusterEnvelope(labels, Wc, scale)
    assert np.allclose(ces.dense(), Wc.T[labels] * scale[:, None])
    assert np.allclose(ces.scaled(3.0).dense(), 3.0 * ces.dense())
    # An unused cluster is dropped rather than counted as a factor column.
    lab2 = np.where(labels == 3, 2, labels)
    assert ClusterEnvelope(lab2, Wc).rank == 3

    # --- an exact representation has D = 0, and a scalar overestimate has a predictable D ---
    r = evaluate(A, DenseApprox(A))
    assert r['D'] == 0.0 and r['max_r'] == 1.0, r
    c = 1.25
    r = evaluate(A, DenseApprox(A * c))
    assert abs(r['D'] - (c-1)/(1+c/10)) < 1.0e-12, r

    # --- row_distances sums to D0, and finds the row that pays ---
    B = A * 1.1
    B[42] = A[42] * 4.0
    rd = row_distances(A, DenseApprox(B))
    r = evaluate(A, DenseApprox(B))
    assert abs(np.nanmean(rd) - r['D0']) < 1.0e-12, (np.nanmean(rd), r['D0'])
    assert int(np.nanargmax(rd)) == 42, np.nanargmax(rd)

    # --- outputs with no variance are ignored, and the block weighting stays exact ---
    # A W=0 Detrender2d annihilates the DM=0 output, so these rows occur in real maps. They
    # must drop out of BOTH the D0 average and its denominator, and the blockwise combination
    # must weight by scored rows rather than by block size -- otherwise the answer depends on
    # the block size, which is the bug this checks for.
    Az = A.copy()
    Az[150] = 0.0                  # exactly zero
    Az[151] = 1.0e-16              # float32 roundoff of zero, which is how they really arrive
    ref = evaluate(A[[i for i in range(nalpha) if i not in (150, 151)]],
                   DenseApprox(A[[i for i in range(nalpha) if i not in (150, 151)]] * 1.3))
    for nb in (7, 64, 150, nalpha):
        r = evaluate(Az, DenseApprox(Az * 1.3), block_rows=nb)
        assert r['nscored'] == nalpha - 2, (nb, r['nscored'])
        assert r['nalpha'] == nalpha, (nb, r)
        assert abs(r['D0'] - ref['D0']) < 1.0e-12 * max(1.0, abs(ref['D0'])), (nb, r, ref)
        assert np.isfinite(r['D']), (nb, r)
        assert r['argmax_r'][0] not in (150, 151), (nb, r)

    # row_distances reports nan there, and nanmean still reproduces D0.
    rd = row_distances(Az, DenseApprox(Az * 1.3))
    assert np.isnan(rd[150]) and np.isnan(rd[151]), rd[150:152]
    assert abs(np.nanmean(rd) - ref['D0']) < 1.0e-12, (np.nanmean(rd), ref['D0'])

    # A map in which EVERY output has no variance is an error, not a distance of zero.
    Aall = np.full((32, nfreq), 1.0e-16)
    try:
        evaluate(Aall, DenseApprox(Aall * 2.0), block_rows=8)
        raise AssertionError('evaluate accepted a wholly degenerate map')
    except RuntimeError as e:
        assert 'no row could be scored' in str(e), str(e)

    # --- shape mismatch is caught here, not deep inside VarMapDistance ---
    try:
        evaluate(A, DenseApprox(A[:, :5]))
        raise AssertionError('evaluate accepted a shape mismatch')
    except RuntimeError as e:
        assert 'shape mismatch' in str(e), str(e)

    # --- a memmapped A_true gives identical numbers, and is never materialized ---
    # (This is how a real map arrives: read_variance_map(..., lazy=True) hands back memmaps.)
    import tempfile
    with tempfile.TemporaryDirectory() as d:
        path = os.path.join(d, 'A.npy')
        np.save(path, A)
        Am = np.load(path, mmap_mode='r')
        assert isinstance(Am, np.memmap)
        env = ClusterEnvelope(rng.integers(0, 8, size=nalpha), rng.uniform(5, 9, (nfreq, 8)))
        r_mem = evaluate(A, env, block_rows=32)
        r_mmap = evaluate(Am, env, block_rows=32)
        assert r_mem['D0'] == r_mmap['D0'] and r_mem['max_r'] == r_mmap['max_r'], \
            (r_mem, r_mmap)

    # --- frontier() and the formatters run ---
    rows = frontier(A, lambda M, K: DenseApprox(M * (1.0 + 1.0/K), rank=K), [1, 2, 4],
                    name='scalar')
    assert [r['K'] for r in rows] == [1, 2, 4]
    assert rows[0]['D'] > rows[-1]['D'] > 0
    assert len(format_table(rows).splitlines()) == 5
    assert isinstance(format_row(rows[0]), str)

    _self_test_reduced(rng, A)

    if verbose:
        atomic_print('    varmap_eval.self_test: pass')


def _self_test_reduced(rng, A):
    """evaluate_reduced() against evaluate() on the lifted fine approximation.

    This is the contract that makes reduced scoring worth trusting: given a correct Abar, the
    two agree exactly, so a result scored without the dense map is the same number as one
    scored with it.
    """

    nalpha, nfreq = A.shape

    # Groups of contiguous outputs, plus a singleton and an empty-in-the-middle case, since
    # both occur in real groupings.
    labels = np.sort(rng.integers(0, 12, size=nalpha))
    labels[0] = 0
    nbeta = int(labels.max()) + 1
    Abar, y = reduce_map(A, labels, nbeta)

    assert Abar.shape == (nbeta, nfreq) and y.shape == (nalpha,)
    assert np.allclose(y, A.sum(axis=1))
    # Abar dominates every row of its group -- the property the whole scheme rests on.
    assert np.all(Abar[labels] >= A - 1.0e-12), 'reduce_map: Abar does not dominate A'

    for case in ('admissible', 'planted'):
        # An approximation with ONE ROW PER GROUP.
        G = Abar * rng.uniform(1.0, 1.5, size=Abar.shape)
        if case == 'planted':
            G[3, 20] = Abar[3, 20] * 0.8      # an underestimate, in group space
        g_approx = DenseApprox(G, rank=5)
        lifted = DenseApprox(G[labels], rank=5)

        r_red = evaluate_reduced(Abar, y, labels, g_approx)
        r_full = evaluate(A, lifted)

        assert abs(r_red['D0'] - r_full['D0']) < 1.0e-12 * max(1.0, abs(r_full['D0'])), \
            (case, r_red['D0'], r_full['D0'])
        assert r_red['admissible'] == r_full['admissible'], (case, r_red, r_full)
        assert np.isinf(r_red['D']) == np.isinf(r_full['D']), (case, r_red, r_full)
        assert abs(r_red['max_r'] - r_full['max_r']) < 1.0e-12 * r_full['max_r'], \
            (case, r_red['max_r'], r_full['max_r'])
        assert np.isclose(r_red['D_inflated'], r_full['D_inflated'], rtol=1.0e-9), \
            (case, r_red['D_inflated'], r_full['D_inflated'])
        # argmax_r is in GROUP space, and must point at the group containing the fine argmax.
        assert r_red['argmax_r'][0] == labels[r_full['argmax_r'][0]], (case, r_red, r_full)
        assert r_red['argmax_r'][1] == r_full['argmax_r'][1], (case, r_red, r_full)
        assert r_red['nalpha'] == nalpha and r_red['nbeta'] == nbeta and r_red['reduced']

    # --- argument checking ---
    G = Abar * 1.2
    for bad, msg in [
            (lambda: evaluate_reduced(Abar, y[:-1], labels, DenseApprox(G)), 'both are indexed'),
            (lambda: evaluate_reduced(Abar, y, labels + nbeta, DenseApprox(G)), 'out of range'),
            (lambda: evaluate_reduced(Abar, y, labels, DenseApprox(G[:-1])), 'ONE ROW PER GROUP')]:
        try:
            bad()
            raise AssertionError(f'evaluate_reduced accepted bad input ({msg})')
        except RuntimeError as e:
            assert msg in str(e), (msg, str(e))


if __name__ == '__main__':
    self_test()
