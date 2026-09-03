"""The distance function D on variance maps, and the value objects that VarianceMap's
scoring methods return.

D is defined in notes/variance_map.tex (section "Distance function"). It is one-sided:
underestimating any single matrix element is catastrophic (false positives in the far tail of
a ~1e15-trial search), so it is scored as infinite; overestimating is merely a lost event rate,
and is scored by an average which saturates, so that extreme overestimation on a few outputs
cannot dominate. With row sums y_alpha = sum_F A[alpha,F],

    D = infinity                                    if A_approx[alpha,F] < A_true[alpha,F]
                                                       for any (alpha,F)
      = (1/nscored) sum_alpha f(y_approx / y_true)   otherwise

The two branches have wildly different costs, and VarianceMap splits them accordingly:
get_distance() computes the second from row sums alone, while measure_admissibility()
is the elementwise scan that establishes the first.

DO NOT CHANGE THE DEFINITION OF D SILENTLY. Its whole purpose is that numbers from different
experiments are comparable; a change makes new numbers incomparable with every number recorded
in notes/variance_map.tex. Propose changes in the research journal instead. varmap/tests.py's
test_distance_oracles() pins the VALUE of D against oracles written out by hand, so an
accidental change fails there rather than quietly producing new numbers.

In particular, if a number recorded before this package disagrees with get_distance(), suspect
a known CONVENTION difference before suspecting a bug here. An early evaluation harness
combined row blocks weighted by BLOCK SIZE with denominator nalpha, while the distance class it
called already returned a mean over that block's SCORED rows. The two agree except on a map
with degenerate rows (see YTRUE_FLOOR below), where they do not, and get_distance() implements
the latter -- the same convention every later result used. Such a disagreement is not a reason
to change D.
"""

import numpy as np


# Rows of A_true whose sum falls below this are IGNORED: they are outputs with genuinely zero
# variance, for which y_approx/y_true is undefined rather than large. A Detrender2d with time
# half-width W = 0 removes the frequency-constant mode exactly, and the DM = 0 dedispersion
# output is precisely the unlagged sum over all channels, so that output has identically zero
# variance. In floating point such a row sums to ~1e-14 rather than 0, which is why the test is
# a floor rather than a comparison against zero.
#
# The value is chosen from a wide empirical gap, measured over 24 variance maps spanning tree
# rank 8-16, 400-16384 channels, and 1-6 frequency subbands: the largest degenerate row sum was
# 9.3e-14, and the smallest healthy one 5.1e-05. So this floor sits ~3 decades above the noise
# and ~5.7 decades below any real output. Both scalings work in production's favour -- the
# smallest healthy row sum falls ~2x per tree rank but grows ~linearly with nfreq, and
# production has far more channels than the subscale maps that set the 5.1e-05 figure.
#
# Revisit this if A is ever stored in different units or a different dtype: the MARGIN above,
# not the absolute number, is the justification.
YTRUE_FLOOR = 1.0e-10


def f(x):
    """The distance function's kernel f(x) = (x-1)/(1 + x/10), elementwise.

    Strictly increasing and concave. The concavity is what makes the W-step a
    majorize-minimize step (its tangent is a global upper bound); the monotonicity is what
    makes the Q-step exact.
    """
    return (x - 1.0) / (1.0 + x/10.0)


def fprime(x):
    """The derivative f'(x) = 1.1 / (1 + x/10)^2, elementwise.

    Exposed because the W-step's majorization weights are built from it and nothing should
    re-derive them.
    """
    return 1.1 / (1.0 + x/10.0)**2


####################################   value objects   ####################################


class AdmissibilityResult:
    """The result of VarianceMap.measure_admissibility(). A plain value object.

    The (``max_r``, ``argmax_r``) pair is much more informative than a bare pass/fail, which
    is why it is reported rather than folded into a boolean: underestimates cluster (typically
    at low DM), so knowing WHERE the worst one is usually determines the remedy.

    Attributes (read-only):

    - ``admissible`` (bool) -- the elementwise test ``self >= ref``. THIS is the load-bearing
      answer, and it is a separate member rather than ``max_r <= 1`` because the two can only
      be identified when both matrices are nonnegative.
    - ``max_r`` (float) -- max over (row, F) of ``ref/self``, with the sign conventions of
      measure_admissibility(). THIS IS THE INFLATION FACTOR, not an accuracy measure: it is
      exactly the number ``self`` must be scaled by for ``self >= ref`` to hold, which is why
      inflate=True uses it and why it is infinite when ``self`` is non-positive somewhere
      ``ref`` is positive (a real defect no rescaling can repair).

      DO NOT READ IT AS "the error in self". It is a PER-ELEMENT RELATIVE quantity with no
      floor, so on a matrix with wide dynamic range it is set by the smallest element
      compared, not by how well the matrix is approximated. A map accurate to the float64
      truncation floor in every absolute sense can report max_r - 1 = 2e-9 simply because the
      worst element is two decades below the matrix maximum. ``max_diff`` is the number to
      read for accuracy.
    - ``max_diff`` (float) -- the sup-norm error, normalized by the sup-norm of the reference:
      ``max over (row,F) of |ref - self|`` divided by ``max over (row,F) of |ref|``. Zero for
      an exact map. Unlike ``max_r`` this is an ACCURACY measure and nothing more -- it says
      nothing about admissibility, and in particular a map that is negative where ref is
      positive (max_r = inf, unrepairable) can still have a small max_diff. Read the two
      together: max_r for whether it can be used, max_diff for how good it is.
    - ``argmax_r`` (tuple) -- the (row, F) pair attaining ``max_r``. In the coarse case
      ``row`` is a beta, not an alpha -- the caller needs to know which. Note the two
      statistics generally attain their maxima at DIFFERENT elements: ``max_r`` at a small
      one, ``max_diff`` at a large one.
    - ``nviol`` (int) -- number of (row, F) pairs violating ``self >= ref`` by more than a
      relative tolerance, and
    - ``viol_frac`` (float) -- that count as a fraction of the elements compared.
      ``max_r`` reports the WORST violation; these report HOW MANY, and the two answer
      different questions -- one bad element and a systematic 1% shortfall can share a
      ``max_r``, and they want different responses. This is not a nicety, because an LP solver
      returns points violating their own constraints WHILE REPORTING SUCCESS, at roughly one
      decade per rank doubling. A SOLVER STATUS HISTOGRAM IS NOT EVIDENCE OF ADMISSIBILITY AND
      MUST NEVER BE USED AS ONE. These members are.
    - ``viol_rows`` (int) -- number of DISTINCT rows containing a violation. "One row is badly
      wrong" and "every row is slightly wrong" are different diagnoses, and on a coarse map a
      single bad group is repairable by rescaling that group alone while a spread-out
      violation is not.
    - ``worst_rows`` (ndarray) -- indices of the few worst rows, for drilling in. Small and
      bounded; the caller usually wants to look at them, and recomputing means a second full
      pass.
    - ``total_elements`` (int) -- how many (row, F) pairs were compared, so that ``viol_frac``
      can be turned back into a count and so that two results are comparable only when it
      agrees.
    - ``nneg_self`` (int) -- count of elements where ``self < 0``. Zero for anything out of a
      Q-step (the covering constraint on every channel forces the product nonnegative),
      nonzero for a raw truncated SVD -- so this doubles as a cheap detector of an
      under-constrained Q-step.
    - ``vmap`` (VarianceMap) -- a copy of ``self`` with ``is_admissible`` set to the measured
      answer. This is the supported way to turn an uncertified map into one that
      get_distance() will accept.
    - ``inflation``, ``D_inflated`` -- set when ``inflate=True``, else None.
    - ``seconds`` (float) -- wall clock.
    """

    def __init__(self, *, admissible, max_r, argmax_r, nviol, viol_frac, viol_rows,
                 worst_rows, total_elements, nneg_self, vmap, max_diff=0.0, inflation=None,
                 D_inflated=None, seconds=0.0):
        self.admissible = bool(admissible)
        self.max_r = float(max_r)
        self.max_diff = float(max_diff)
        self.argmax_r = tuple(int(i) for i in argmax_r)
        self.nviol = int(nviol)
        self.viol_frac = float(viol_frac)
        self.viol_rows = int(viol_rows)
        self.worst_rows = np.asarray(worst_rows, dtype=np.int64)
        self.total_elements = int(total_elements)
        self.nneg_self = int(nneg_self)
        self.vmap = vmap
        self.inflation = inflation
        self.D_inflated = D_inflated
        self.seconds = float(seconds)

    def __repr__(self):
        row, F = self.argmax_r
        inf = '' if (self.D_inflated is None) else f', D_inflated={self.D_inflated:.6g}'
        return (f'AdmissibilityResult(admissible={self.admissible}, max_r={self.max_r:.6g},'
                f' max_diff={self.max_diff:.6g},'
                f' argmax_r=(row={row}, F={F}), nviol={self.nviol}'
                f' ({self.viol_frac:.3g} of {self.total_elements}),'
                f' viol_rows={self.viol_rows}, nneg_self={self.nneg_self}{inf})')


class DistanceEstimate:
    """The result of VarianceMap.estimate_distance(): a SUBSAMPLED estimate of D.

    Named for what it is, because an estimate and an exact value must never be confused in a
    table that exists to be compared across experiments.

    Attributes (read-only):

    - ``D`` (float) -- the estimate of get_distance().
    - ``stderr`` (float) -- its standard error, from the finite-population ratio estimator.
      D is a ratio of two sampled totals (the summed f-contributions over the summed scored
      row counts), not a sample mean, which is why this is not simply ``std/sqrt(n)``.
    - ``nsampled`` (int) -- number of rows of the map that were sampled.
    - ``frac_sampled`` (float) -- ``nsampled / nbeta``.
    - ``nscored`` (int) -- fine outputs that contributed, within the sample.
    - ``groups`` (ndarray) -- the sampled row indices. WHEN COMPARING ANYTHING, SHARE THE
      SUBSET: evaluating two arms on the SAME subset makes their ratio far better determined
      than either value (measured: ratios to 1-3% where the individual cells carried ~10%).
      Pass this back as estimate_distance(groups=...) for the paired second call.
    """

    def __init__(self, *, D, stderr, nsampled, frac_sampled, nscored, groups):
        self.D = float(D)
        self.stderr = float(stderr)
        self.nsampled = int(nsampled)
        self.frac_sampled = float(frac_sampled)
        self.nscored = int(nscored)
        self.groups = np.asarray(groups, dtype=np.int64)

    def __repr__(self):
        return (f'DistanceEstimate(D={self.D:.6g} +/- {self.stderr:.3g},'
                f' nsampled={self.nsampled} ({100*self.frac_sampled:.3g}%),'
                f' nscored={self.nscored})')
