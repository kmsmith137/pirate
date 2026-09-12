"""Helpers shared by the chimefrb port's randomized tests ('pirate_frb test --cfrb') and its
section of 'pirate_frb dev coverage'."""

import collections

import numpy as np


def default_rng(rng=None):
    """A fresh numpy Generator for one test, seeded from the master --seed; or 'rng' itself,
    if one is given.

    NEVER call numpy's zero-argument default_rng() in these tests. It seeds itself from OS
    ENTROPY, which puts every draw outside __main__.seed_rngs(): a failing draw then cannot be
    replayed, and 'pirate_frb test' prints a seed that does not cover it. Seeded instead from
    numpy's global RandomState, which seed_rngs() pins -- so successive calls still differ (a
    long run explores different data) while the whole run replays from one integer. This is
    the same rule as varmap/tests.py's _rng() and detrending.testutils.default_rng().

    Passing a Generator in is how one draw is replayed on its own, without re-running the
    tests before it: e.g. test_std_dev_clipper(1, rng=np.random.default_rng(235)).
    """
    return np.random.default_rng(np.random.randint(0, 1 << 32)) if rng is None else rng


def draw_within_budget(rng, candidates, cost, budget):
    """Draw one of 'candidates' uniformly, among those whose 'cost' fits in 'budget'.

    Several shape draws here have a cost that is a PRODUCT of things drawn separately: the
    spline detrender costs nfreq*M*T, with nfreq from random_config() and (M, T) from
    random_geometry(). Drawing the factors independently lets a rare expensive corner run
    away with the whole suite's runtime, which is what notes/unit_tests.md item 4 means by
    bounding a machine-independent proxy for running time.

    The candidate shapes are ENUMERATED and filtered rather than drawn and retried: the set
    is small, a filter cannot loop, and a budget that excludes everything still yields the
    CHEAPEST candidate instead of raising. That last part is what keeps the degenerate
    shapes -- one beam, the shortest chunk -- reachable at every size, and those are the
    ones that break loop bounds.

    'cost' is called on each candidate and compared against 'budget', both in whatever
    units the caller finds natural.
    """
    affordable = [c for c in candidates if cost(c) <= budget]
    if not affordable:
        return min(candidates, key=cost)
    return affordable[int(rng.integers(len(affordable)))]


WiDraw = collections.namedtuple('WiDraw', 'intensity weights offset scale')


def random_wi_pair(rng, shape, offset_max=1.0e3, scale_range=(0.5, 20.0)):
    """The (intensity, weights) pair that the three statistic-bearing tests start from:
    float64 arrays of 'shape', plus the two scalars the draw used.

    The intensity is Gaussian about a large random OFFSET, and the offset is the point.
    The variance-validity cutoff these transforms apply is proportional to the mean, so it
    is inert when the mean sits near zero; and a large mean is also what makes the
    single-pass variance cancel, which is the whole reason `two_pass` exists.

    The weights are a Bernoulli mask times a scale, with the mask probability drawn as
    clip(uniform(-0.1, 1.1), 0, 1) -- the idiom of notes/unit_tests.md item 6 -- so that a
    few percent of draws come out all-masked and a few percent all-unmasked.

    'offset' and 'scale' are returned because the callers need them: outlier amplitudes are
    multiples of the scale, and plant_degenerate_rows() plants rows AT the offset.
    """
    if len(scale_range) != 2 or not (0 < scale_range[0] <= scale_range[1]):
        raise ValueError(f'random_wi_pair: bad scale_range {scale_range!r}')

    offset = rng.uniform(-offset_max, offset_max)
    scale = rng.uniform(*scale_range)
    intensity = rng.normal(offset, scale, size=shape)

    p = np.clip(rng.uniform(-0.1, 1.1), 0.0, 1.0)
    weights = (rng.uniform(size=shape) < p) * rng.uniform(0.5, 1.5, size=shape)

    return WiDraw(intensity, weights, offset, scale)


def plant_degenerate_rows(rng, draw, nrows, sel, rate=0.05):
    """Plant the three kinds of degenerate row into 'draw', in place, each kind with
    probability 'rate' per row.

    THESE THREE AND NO OTHERS, because they are the three ways a weighted variance comes
    out INVALID, and a transform that rejects a row's variance zeroes every weight in that
    row -- a decision with a blast radius no tolerance can absorb, so a test that never
    reaches it is not testing the branch that matters most:

      constant                    the variance is exactly zero
      no weight at all            there is no variance to compute
      near-constant, large mean   the eps_2*mean cutoff is the operative one

    'sel' maps a row index to a numpy index tuple, because the callers mean different
    things by "row": a row of an (R, L) array, a whole channel of one beam, or a block of
    Df channels. A caller working beam by beam passes a selector closing over the beam.
    """
    for r in np.flatnonzero(rng.uniform(size=nrows) < rate):
        draw.intensity[sel(r)] = draw.offset
    for r in np.flatnonzero(rng.uniform(size=nrows) < rate):
        draw.weights[sel(r)] = 0.0
    for r in np.flatnonzero(rng.uniform(size=nrows) < rate):
        block = draw.intensity[sel(r)]
        draw.intensity[sel(r)] = draw.offset * (1.0 + 1.0e-6 * rng.normal(size=block.shape))


# The families random_weight_base() knows. A caller that lets a user name a "kind" which
# may be either a family or a structural pattern (a run, a gap) should test membership here
# before passing it on as a base.
BASE_KINDS = ('ones', 'binary', 'counts', 'continuous')


def random_weight_base(rng, n, kind):
    """One length-n float64 weight vector of the named family, before any structural
    mutation (a dead run, a gap) that the caller cuts into it.

    The families the two detrender tests draw from, and what makes each in the real chain:

      ones         every weight 1
      binary       Bernoulli {0,1} at a per-vector rate p in [0.3, 1] -- a clipper's mask
      counts       integers 0..16, Binomial(16, p) -- 16x-downsampled clipper output
      continuous   uniform in [0, 2] -- nothing in the pipeline makes these, but the
                   kernels accept them, and they are what pins the WEIGHTED fit down

    'p' is drawn only for the families that use it.
    """
    if kind == 'ones':
        return np.ones(n)
    p = rng.uniform(0.3, 1.0)
    if kind == 'binary':
        return (rng.uniform(size=n) < p).astype(np.float64)
    if kind == 'counts':
        return rng.binomial(16, p, size=n).astype(np.float64)
    if kind == 'continuous':
        return rng.uniform(0.0, 2.0, size=n)
    raise ValueError(f'random_weight_base: unknown kind {kind!r}')
