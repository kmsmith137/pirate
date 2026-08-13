import numpy as np

from ..utils import atomic_print


####################################   class VarMapDistance   ####################################


class VarMapDistance:
    """Distance between a true variance map and a low-rank approximation to it.

    Implements the distance function D(A_true, A_approx) of notes/tree_dedispersion.tex
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
      D0:         the second branch, (1/nalpha) sum_alpha f(...), computed unconditionally,
                  i.e. without the infinity test.
      max_r:      max over (alpha,F) of A_true[alpha,F] / A_approx[alpha,F]. Equivalently,
                  D is infinite iff max_r > 1.
      argmax_r:   the (alpha, F) pair at which max_r is attained.
      nalpha:     number of output indices.
      nfreq:      number of input frequency channels.

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

    def __init__(self, A_true, A_approx):
        """The two args are (nalpha, nfreq) arrays; see the class docstring."""

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

        max_r = -np.inf
        argmax_r = (0, 0)
        fsum = 0.0

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

            # Ratio conventions at zero: (positive / 0) = inf is a real underestimate, but
            # (0 / 0) is not, and must not be allowed to become the argmax, so it maps to 0.
            with np.errstate(divide='ignore', invalid='ignore'):
                ratio = np.where(row_true > 0, row_true / row_approx, 0.0)

            f = int(np.argmax(ratio))
            if ratio[f] > max_r:
                max_r = float(ratio[f])
                argmax_r = (alpha, f)

            y_true = row_true.sum()
            y_approx = row_approx.sum()

            if y_true <= 0:
                raise RuntimeError(f'VarMapDistance: row {alpha} of A_true sums to {y_true};'
                                   ' every output must have nonzero variance')

            x = y_approx / y_true
            fsum += (x - 1.0) / (1.0 + x/10.0)

        self.max_r = max_r
        self.argmax_r = argmax_r
        self.D0 = fsum / self.nalpha
        self.D = self.D0 if (max_r <= 1.0) else np.inf


    def __str__(self):
        a, f = self.argmax_r
        return (f'VarMapDistance(D={self.D:.6g}, D0={self.D0:.6g}, max_r={self.max_r:.6g},'
                f' argmax_r=(alpha={a}, F={f}), shape=({self.nalpha},{self.nfreq}))')


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

        atomic_print(f'    test_random(nalpha={nalpha}, nfreq={nfreq}): pass')
