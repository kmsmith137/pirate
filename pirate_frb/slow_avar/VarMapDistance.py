import numpy as np

from ..utils import atomic_print


# Rows of A_true whose sum falls below this are IGNORED (see the VarMapDistance docstring).
# They are outputs with no variance, for which y_approx/y_true is undefined rather than large.
#
# The value is chosen from a wide empirical gap, measured over 24 variance maps spanning tree
# rank 8-16, 400-16384 channels, and 1-6 frequency subbands: the largest degenerate row sum was
# 9.3e-14, and the smallest healthy one 5.1e-05. So this floor sits ~3 decades above the noise
# and ~5.7 decades below any real output. Both scalings work in production's favour -- the
# smallest healthy row sum falls ~2x per tree rank but grows ~linearly with nfreq, and
# production has far more channels than the subscale maps that set the 5.1e-05 figure.
#
# Revisit this if A is ever stored in different units or a different dtype: the margin above,
# not the absolute number, is the justification.
YTRUE_FLOOR = 1.0e-10


####################################   class VarMapDistance   ####################################


class VarMapDistance:
    """Distance between a true variance map and a low-rank approximation to it.

    Implements the distance function D(A_true, A_approx) of notes/variance_map.tex
    (section "Distance function"). Both arguments are (nalpha, nfreq) arrays, where alpha
    is the flattened (dm, multiplet, profile) output index and F is an input frequency
    channel.

    The distance is deliberately asymmetric. Underestimating any single matrix element is
    catastrophic (it produces false positives in the far tail of a ~1e15-trial search), so
    it is scored as infinite; overestimating is merely a lost event rate, and is scored by
    an average which saturates, so that extreme overestimation on a few outputs cannot
    dominate. Concretely, with row sums y_alpha = sum_F A[alpha,F],

        D = infinity                                  if A_approx[alpha,F] < A_true[alpha,F]
                                                         for any (alpha,F)
          = (1/nalpha) sum_alpha f(y_approx / y_true)  otherwise

        f(x) = (x-1) / (1 + x/10)

    Members
    -------
      D:          the distance function above (may be np.inf).
      D0:         the second branch, (1/nscored) sum_alpha f(...), computed unconditionally,
                  i.e. without the infinity test.
      max_r:      max over (alpha,F) of A_true[alpha,F] / A_approx[alpha,F]. Equivalently,
                  D is infinite iff max_r > 1.
      argmax_r:   the (alpha, F) pair at which max_r is attained.
      nalpha:     number of output indices.
      nscored:    number of rows that contributed (nalpha minus the ignored rows, below).
      nfreq:      number of input frequency channels.

    Outputs with no variance are ignored
    ------------------------------------

    A row of A_true whose sum is below YTRUE_FLOOR is skipped entirely: it contributes to
    neither D0 nor max_r, and D0 averages over 'nscored' rows rather than over nalpha.

    Such a row is not a defect. A Detrender2d with time half-width W = 0 removes the
    frequency-constant mode exactly (a clamped B-spline basis is a partition of unity, and the
    roughness regulator does not penalize a constant), and the DM = 0 dedispersion output is
    precisely the unlagged sum over all channels -- so that output has identically zero
    variance. CHIME currently runs W = 0. The affected rows are the P profiles of the single
    multiplet spanning the whole band; with frequency subbands the other subbands' DM = 0
    outputs are unaffected, since a subband's channel sum is not the full-band constant.

    For such a row y_approx/y_true is undefined, not merely large, so there is no number to
    report; in floating point it shows up as a row summing to ~1e-14 instead of 0, which is
    why the test is a floor rather than a comparison against zero.

    Ignoring them for max_r as well as for D0 is safe, not a loosening: A_true >= 0 is checked,
    so a row sum below the floor implies every element of the row is below the floor. An
    ignored row cannot be hiding a large matrix element.

    If NO row is scored, the constructor raises: a wholly degenerate map means a broken sweep
    or config, not a distance of zero. Pass allow_empty=True to get D0 = nan instead, which is
    what a caller evaluating in row blocks needs, since one block may legitimately contain only
    ignored rows (see varmap_eval).

    The pair (max_r, D0) is more informative than D alone, and is why they are computed
    even when D is infinite. If max_r is a little larger than 1, we know that we just need
    to inflate A_approx a little to get a distance which is close to D0: scaling A_approx
    by max_r makes it >= A_true everywhere (so D becomes finite), and multiplies every
    ratio y_approx/y_true by max_r, so the resulting distance tends to D0 as max_r -> 1.
    A rejected approximation with max_r = 1.02 is therefore nearly usable, whereas one with
    max_r = 50 is not, and D = infinity does not distinguish them. (In floating point,
    scale by slightly more than max_r: scaling by exactly max_r lands on the D = infinity
    boundary, where rounding can leave a residual underestimate.)

    'argmax_r' says where the worst underestimate is, which is usually more useful than the
    fact that there is one -- underestimates tend to cluster (e.g. at low DM), and the
    remedy differs depending on where they are.

    The constructor loops over rows of the two matrices, and allocates no matrix-sized
    temporaries, so it is usable on maps too large to duplicate in memory, and on the lazily
    loaded arrays returned by read_variance_map(..., lazy=True).
    """

    def __init__(self, A_true, A_approx, allow_empty=False):
        """The two args are (nalpha, nfreq) arrays; see the class docstring.

        allow_empty: if True, a call in which every row is ignored returns D0 = nan rather
        than raising. Only row-blocked callers should need it."""

        A_true = np.asanyarray(A_true)
        A_approx = np.asanyarray(A_approx)

        if A_true.ndim != 2:
            raise RuntimeError(f'VarMapDistance: expected 2-d A_true, got shape {A_true.shape}')
        if A_true.shape != A_approx.shape:
            raise RuntimeError('VarMapDistance: shape mismatch between A_true'
                               f' {A_true.shape} and A_approx {A_approx.shape}')

        self.nalpha, self.nfreq = A_true.shape

        if self.nalpha == 0 or self.nfreq == 0:
            raise RuntimeError(f'VarMapDistance: empty matrix (shape {A_true.shape})')

        max_r = 0.0
        argmax_r = (0, 0)
        fsum = 0.0
        nscored = 0

        for alpha in range(self.nalpha):
            # One row at a time: these are the only temporaries, and they are (nfreq,) not
            # (nalpha,nfreq). Cast to float64 -- the maps are often stored as float32, and
            # the row sums below are reductions over ~1e4 channels.
            row_true = np.asarray(A_true[alpha], dtype=np.float64)
            row_approx = np.asarray(A_approx[alpha], dtype=np.float64)

            if not (np.all(np.isfinite(row_true)) and np.all(np.isfinite(row_approx))):
                raise RuntimeError(f'VarMapDistance: non-finite matrix element in row {alpha}')
            if np.any(row_true < 0) or np.any(row_approx < 0):
                raise RuntimeError(f'VarMapDistance: negative matrix element in row {alpha}'
                                   ' (matrix elements are variances)')

            y_true = row_true.sum()

            # An output with no variance: skip it entirely, for max_r as well as for D0.
            # See "Outputs with no variance are ignored" in the class docstring.
            if y_true < YTRUE_FLOOR:
                continue

            # Ratio conventions at zero: (positive / 0) = inf is a real underestimate, but
            # (0 / 0) is not, and must not be allowed to become the argmax, so it maps to 0.
            with np.errstate(divide='ignore', invalid='ignore'):
                ratio = np.where(row_true > 0, row_true / row_approx, 0.0)

            f = int(np.argmax(ratio))
            if ratio[f] > max_r:
                max_r = float(ratio[f])
                argmax_r = (alpha, f)

            x = row_approx.sum() / y_true
            fsum += (x - 1.0) / (1.0 + x/10.0)
            nscored += 1

        if (nscored == 0) and not allow_empty:
            raise RuntimeError(
                f'VarMapDistance: all {self.nalpha} rows of A_true sum to less than'
                f' {YTRUE_FLOOR}, so no row could be scored. A few such rows are expected'
                ' (a W=0 Detrender2d annihilates the DM=0 output), but a map where every'
                ' output has zero variance means a broken sweep or config.')

        self.nscored = nscored
        self.max_r = max_r
        self.argmax_r = argmax_r
        self.D0 = (fsum / nscored) if (nscored > 0) else np.nan
        self.D = self.D0 if (max_r <= 1.0) else np.inf


    def __str__(self):
        a, f = self.argmax_r
        ign = '' if (self.nscored == self.nalpha) else \
            f', ignored={self.nalpha - self.nscored}'
        return (f'VarMapDistance(D={self.D:.6g}, D0={self.D0:.6g}, max_r={self.max_r:.6g},'
                f' argmax_r=(alpha={a}, F={f}), shape=({self.nalpha},{self.nfreq}){ign})')


    @staticmethod
    def test_random(nalpha=97, nfreq=53, niter=10):
        """Checks D against a dense (non-row-looped) evaluation, and checks the inflation
        property in the class docstring: scaling A_approx by max_r always makes D finite,
        and drives it to D0 as max_r -> 1."""

        rng = np.random.default_rng()

        for _ in range(niter):
            A_true = rng.uniform(0.1, 2.0, size=(nalpha,nfreq))

            # Half the iterations are strict overestimates (D finite); half plant a single
            # underestimate, so that the D = infinity branch is exercised even at small
            # matrix sizes, with max_r near 1.
            A_approx = A_true * rng.uniform(1.0, 1.5, size=(nalpha,nfreq))
            if rng.uniform() < 0.5:
                a, f = rng.integers(nalpha), rng.integers(nfreq)
                A_approx[a,f] = A_true[a,f] * rng.uniform(0.9, 0.999)

            d = VarMapDistance(A_true, A_approx)

            # Dense reference for D0 and max_r.
            x = A_approx.sum(axis=1) / A_true.sum(axis=1)
            d0_ref = np.mean((x-1) / (1 + x/10))
            r = A_true / A_approx
            max_r_ref = r.max()

            assert d.nscored == nalpha, d
            assert abs(d.D0 - d0_ref) < 1.0e-12 * max(1.0, abs(d0_ref)), (d.D0, d0_ref)
            assert abs(d.max_r - max_r_ref) < 1.0e-12 * max_r_ref, (d.max_r, max_r_ref)
            assert r[d.argmax_r] == max_r_ref, (d.argmax_r, r[d.argmax_r], max_r_ref)

            # The defining contract: D is D0, except that a single underestimate anywhere
            # sends it to infinity.
            if d.max_r > 1.0:
                assert d.D == np.inf, d
            else:
                assert d.D == d.D0, d

            # Inflation property, for the rejected (max_r > 1) case: scaling by a hair more
            # than max_r removes every underestimate, and the resulting distance approaches
            # D0 from above as max_r -> 1. The 'hair more' matters -- scaling by exactly
            # max_r lands on the D = infinity boundary, where rounding can leave a residual
            # underestimate.
            if d.max_r > 1.0:
                d2 = VarMapDistance(A_true, A_approx * d.max_r * (1 + 1.0e-12))
                assert np.isfinite(d2.D), d2
                assert d2.D >= d.D0 - 1.0e-12, (d2.D, d.D0)
                assert d2.D <= d.D0 + 2.0 * (d.max_r - 1.0), (d2.D, d.D0, d.max_r)

        # --- rows with no variance are ignored, not scored and not raised on ---
        # A W=0 Detrender2d annihilates the DM=0 output, which reaches us as a row summing to
        # ~1e-14 rather than to 0. Such a row must not contribute to D0 (its ratio is
        # undefined), and must not contribute to max_r either, even when A_approx is zero
        # there -- which it legitimately can be, since A_true is zero.
        A_true = rng.uniform(0.1, 2.0, size=(nalpha, nfreq))
        A_approx = A_true * 1.5
        ref = VarMapDistance(A_true, A_approx)

        A_true[3] = 1.0e-16                   # ~ float32 roundoff of an exactly-zero row
        A_true[9] = 0.0                       # and an exactly-zero one
        A_approx = A_true * 1.5               # so A_approx is zero on the zero row
        d = VarMapDistance(A_true, A_approx)

        assert d.nscored == nalpha - 2, (d.nscored, nalpha)
        assert np.isfinite(d.D) and d.max_r <= 1.0, d
        # The scored rows are unchanged, so D0 must match the all-rows reference exactly:
        # the ignored rows are removed from both the sum and the denominator.
        assert abs(d.D0 - ref.D0) < 1.0e-12 * max(1.0, abs(ref.D0)), (d.D0, ref.D0)
        assert d.argmax_r[0] not in (3, 9), d

        # An ignored row cannot hide a large matrix element, because A_true >= 0 means a row
        # sum below the floor bounds every element of the row.
        assert A_true[3].max() < 1.0e-10 and A_true[9].max() == 0.0

        # A wholly degenerate map is an error, unless the caller is working in row blocks.
        Az = np.full((5, nfreq), 1.0e-16)
        try:
            VarMapDistance(Az, Az * 2.0)
            raise AssertionError('VarMapDistance accepted a wholly degenerate map')
        except RuntimeError as e:
            assert 'no row could be scored' in str(e), str(e)
        de = VarMapDistance(Az, Az * 2.0, allow_empty=True)
        assert de.nscored == 0 and np.isnan(de.D0) and np.isnan(de.D), de

        atomic_print(f'    test_random(nalpha={nalpha}, nfreq={nfreq}): pass')
