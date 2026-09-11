"""Helpers shared by the chimefrb port's randomized tests ('pirate_frb test --cfrb') and its
section of 'pirate_frb dev coverage'."""

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
