"""
Helpers shared by the three detrending test suites (--dt1d, --dt1k, --dts).

Everything here was duplicated two or three times before, in copies that had
already started to drift apart.  A helper belongs here when all three suites want
the SAME behaviour and a divergence between copies would be a bug rather than a
choice; where the suites legitimately differ -- what their test_masked_data_unused
checks, how they key their expansion tallies -- the difference is expressed as an
argument, not as a fork of the code.

The suites are pure numpy on purpose, and everything below draws from a numpy
Generator passed in by the caller.  That is what lets run_all() reproduce one suite
from its printed entropy alone; see default_rng() below.
"""

import numpy as np


####################################################################################################
#
# Random draws.


def default_rng(rng):
    """A fresh numpy Generator for one test, seeded from the master --seed.

    NEVER call numpy's zero-argument default_rng() here.  It seeds itself from OS
    ENTROPY, which puts every draw in these suites outside __main__.seed_rngs(): a
    failing draw then cannot be replayed, and 'test' prints a seed that does not
    cover them.  Seeded instead from numpy's global RandomState, which seed_rngs()
    pins -- so successive calls still differ (a long run explores different data)
    while the whole sequence replays from one integer.  This is the same rule, for
    the same reason, as varmap/tests.py's _rng().

    Each suite's run_all() prints the entropy of the generator it makes, which
    reproduces THAT SUITE on its own without re-running anything else: pass
    np.random.default_rng(<entropy>) back in as 'rng'.  That finer replay holds only
    while every draw in the suite comes from this Generator, which is why the shape
    helpers below take one instead of reaching for a global.
    """
    return np.random.default_rng(np.random.randint(0, 1 << 32)) if rng is None else rng


def random_spectator_shape(rng, budget, first_max=4):
    """Draw (first, second) with first*second <= 'budget' and both >= 1.

    The two axes a detrending test has to choose besides the one under test: a
    batch/spectator count and a time extent (or a chunk count).  Hardcoding them --
    which is what every one of these sites used to do -- fixes the cost but also
    fixes the SHAPE, and the shapes that break loop bounds are the degenerate ones:
    a single spectator row, a single time sample, a single chunk.  A budget on the
    product keeps the cost where it was while letting the draw reach them.

    'budget' should be about the product the site used to hardcode, so the run does
    not get slower.  'first_max' caps the first axis separately, since the spectator
    count is the cheaper of the two to grow and a draw that spent the whole budget
    on it would stop exercising time at all.
    """
    budget, first_max = int(budget), int(first_max)
    if budget < 1 or first_max < 1:
        raise ValueError(f'random_spectator_shape: need budget >= 1 and first_max >= 1, '
                         f'got budget={budget}, first_max={first_max}')
    first = int(rng.integers(1, min(first_max, budget) + 1))
    return first, int(rng.integers(1, budget // first + 1))


def random_stream_geometry(rng, nsamp, s_max, lag, block=1):
    """Draw (chunk_size, nchunk, S_ax, T) for one streaming-detrender test.

    A streaming test needs four numbers -- how many spectator rows, how many chunks,
    how long a chunk, and hence how long the buffer -- and every site in these suites
    used to hardcode all four.  That fixes the cost, but it also fixes the SHAPE, and
    the shapes that break a streaming implementation are the degenerate ones: a
    single spectator row, a single chunk (nothing carried across a seam), and the
    shortest chunk the detrender allows.  None of those is reachable from a tuple
    that was chosen once to be "a reasonable size".

    'nsamp' is the total sample count to spend, which is what the cost scales with;
    pass about the product the site used to hardcode and the run stays as fast as it
    was.  'lag' is the padding the detrender needs beyond the chunks (2W for the
    local polynomial fit, L for the Kalman one) and 'block' the granularity
    chunk_size must be a multiple of (2W, or 1 where the detrender does not care).
    """
    S_ax, nchunk = random_spectator_shape(rng, 4*s_max, first_max=s_max)
    chunk_size = max(1, (int(nsamp) // (S_ax * nchunk)) // int(block)) * int(block)
    return chunk_size, nchunk, S_ax, nchunk*chunk_size + int(lag)


def random_polynomial(rng, S_ax, T, deg, scale, dtype):
    """P(t) of degree 'deg', normalized to |P| <= 1 across the buffer.

    't' is measured in units of 'scale' -- the fit's own length scale (W for the
    local polynomial fit, tau for the Kalman one) -- so the coefficients are O(1)
    in the units the detrender works in, whatever the buffer length.  Built in
    float64 and cast at the end, so the test data is the same to the last bit at
    both precisions and a float32 comparison measures the detrender rather than
    the input.
    """
    x = (np.arange(T, dtype=np.float64) - T/2) / max(float(scale), 1.0)
    coef = rng.normal(size=(S_ax, deg+1))
    P = np.zeros((S_ax, T), dtype=np.float64)
    for j in range(deg+1):
        P += coef[:, j:j+1] * (x**j)[None, :]
    P /= max(float(np.max(np.abs(P))), 1e-300)
    return P.astype(dtype)


def poison_masked(rng, clean, mask):
    """A copy of 'clean' with every masked sample replaced by garbage.

    The garbage is nan, +-inf and huge finite values in roughly equal parts, because
    a masked sample may hold literally anything -- a dropped packet leaves whatever
    was in the buffer.  That mixture is the strong form of the test: any arithmetic
    that WEIGHTS by the mask rather than SELECTING on it fails immediately, since
    0*inf and 0*nan are nan, while a merely huge finite value catches a leak that
    survives an isfinite() guard.

    Callers compare a run on this against a run on 'clean' and require the outputs
    to be bit-identical.  Bit-identity rather than "the tolerances still hold": a
    leak small enough to hide inside a tolerance is still a leak.
    """
    junk = rng.uniform(-1e10, 1e10, size=clean.shape)
    pick = rng.integers(0, 4, size=clean.shape)
    junk = np.where(pick == 1, np.inf, junk)
    junk = np.where(pick == 2, -np.inf, junk)
    junk = np.where(pick == 3, np.nan, junk)
    return np.where(mask, clean, junk).astype(clean.dtype)


####################################################################################################
#
# Reporting.


def maxdiff(a, b):
    """max |a-b| in float64, or 0.0 if the arrays are empty."""
    a = np.asarray(a, dtype=np.float64)
    b = np.asarray(b, dtype=np.float64)
    return float(np.max(np.abs(a-b))) if a.size else 0.0


class ExpansionTally:
    """Cumulative mask-expansion accounting, for visual inspection at end of run.

    Deliberately never reset: under 'test -n N' each iteration draws fresh masks,
    and the running total across iterations is what says whether expansion fires at
    a sane rate.  A single iteration is far too small a sample to judge from.

    'key' separates populations that expand at different rates BY CONSTRUCTION --
    the polynomial degree, the spline degree, the eps of the precision under test.
    Pooling those would make the printed rate uninterpretable rather than merely
    coarse, since one of the populations often cannot expand at all.  Suites keep
    SEPARATE tallies (one instance each) for the same reason.  A suite with nothing
    to split on passes a 'fmt_key' returning the empty string and gets one bucket.
    """

    def __init__(self, fmt_key=str):
        self._d = {}
        self._fmt_key = fmt_key

    def note(self, key, mask_in, mask_out):
        """'mask_in' restricted to the output range, and the detrender's mask_out."""
        d = self._d.setdefault(key, {'valid_in': 0, 'expanded': 0})
        d['valid_in'] += int(mask_in.sum())
        d['expanded'] += int((mask_in & ~mask_out).sum())

    def __str__(self):
        if not self._d:
            return 'none recorded'
        return '  '.join(
            f"{self._fmt_key(k) + ': ' if self._fmt_key(k) else ''}"
            f"{d['expanded']}/{d['valid_in']} = "
            f"{(d['expanded']/d['valid_in'] if d['valid_in'] else 0.0):.3e}"
            for k, d in sorted(self._d.items()))
