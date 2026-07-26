"""
Randomized mask generation for the detrender tests.

random_mask() draws a shape (M,T) boolean mask by choosing a *type* independently
for each of the M rows, then randomizing that type's parameters.  Rows are
therefore independent, which matches the real data layout (every (beam,freq) pair
has its own RFI mask) and exercises the spectator axis rather than replicating one
pattern across it.

All-valid is deliberately given 50% of the probability mass: it is by far the most
common case in real data, and it is also the case where the estimator has exact
analytic properties (symmetric window => S_odd = 0 => checkerboard Gram), so it is
worth hitting often.  The remaining 50% is spread over the geometries that broke
earlier candidate algorithms -- long gaps, one-sided windows, narrow off-center
clusters, and fully masked scan blocks (see the appendix of
notes/tree_dedispersion.tex).

Note that a single cluster in a length-T array automatically sweeps the whole
range of within-window offsets as the window slides past it, so the cluster
*position* within the window does not need to be randomized separately.
"""

import numpy as np


def _loguniform_int(rng, lo, hi):
    """Integer in [lo,hi], log-uniform, so that both small and large values of a
    length or stride parameter are sampled."""
    lo, hi = max(1, int(lo)), max(1, int(hi))
    if hi <= lo:
        return lo
    return int(np.clip(round(np.exp(rng.uniform(np.log(lo), np.log(hi)))), lo, hi))


# ---------------------------------------------------------------- type builders
# Each returns a 1-d bool array of length T.  W is the window half-width, so the
# scan block length is B = 2W.

def _all_valid(T, W, rng):
    return np.ones(T, dtype=bool)


def _all_masked(T, W, rng):
    return np.zeros(T, dtype=bool)


def _bernoulli(T, W, rng):
    """iid Bernoulli, p ~ U(0,1) so the whole sparsity range is swept."""
    return rng.random(T) < rng.uniform(0.0, 1.0)


def _gap(T, W, rng):
    """One contiguous masked run.  Length is log-uniform up to 4W, so this covers
    both short dropouts and gaps wider than a full window."""
    m = np.ones(T, dtype=bool)
    L = _loguniform_int(rng, 1, 4*W)
    lo = int(rng.integers(0, T))
    m[lo:lo+L] = False
    return m


def _one_sided(T, W, rng):
    """Everything masked on one side of a random boundary.  Windows straddling it
    see valid samples on one side only, which is the case that makes the fit an
    extrapolation.  The side is randomized (the old fixed zoo only masked left)."""
    b = int(rng.integers(0, T+1))
    m = np.ones(T, dtype=bool)
    if rng.random() < 0.5:
        m[:b] = False
    else:
        m[b:] = False
    return m


def _periodic(T, W, rng):
    """Periodic dropouts.  The period is drawn from a set that deliberately
    includes values commensurate with the block length B = 2W, plus a random one,
    since commensurability with the scan geometry is what we want to stress."""
    cands = [max(2, W//2), max(2, W), max(2, 2*W), max(2, 4*W),
             int(rng.integers(2, max(3, 4*W)))]
    period = int(cands[rng.integers(len(cands))])
    duty = int(rng.integers(1, max(2, period)))     # masked samples per period
    phase = int(rng.integers(0, period))
    return ((np.arange(T) + phase) % period) >= duty


def _cluster(T, W, rng):
    """A single narrow run of valid samples, everything else masked.  As the
    window slides past it the cluster's offset from the window center sweeps the
    full range, which is the degenerate extrapolation geometry."""
    m = np.zeros(T, dtype=bool)
    hw = int(rng.integers(0, max(1, W//8) + 1))
    c = int(rng.integers(0, T))
    m[max(0, c-hw):c+hw+1] = True
    return m


def _bimodal(T, W, rng):
    """Two narrow clusters.  When both fall inside one window, G_ii > 0 for every
    i and yet the curvature is barely determined -- the case that motivates the
    pivot floor acting on pivots rather than on the diagonal."""
    m = np.zeros(T, dtype=bool)
    c0 = int(rng.integers(0, T))
    sep = int(rng.integers(0, max(1, 3*W)))
    for c in (c0, c0 + sep):
        if 0 <= c < T:
            hw = int(rng.integers(0, max(1, W//8) + 1))
            m[max(0, c-hw):c+hw+1] = True
    return m


def _masked_blocks(T, W, rng):
    """One or more whole scan blocks masked -- the NaN trap for the empty-set rule
    in MomentSet.merge.  Absolute block boundaries in the stream fall at multiples
    of B (the lattice is anchored at chunk_start - W and Tc is a multiple of B),
    so half the time we align to one and half the time we deliberately do not."""
    B = 2*W
    m = np.ones(T, dtype=bool)
    k = int(rng.integers(1, 4))
    if rng.random() < 0.5:
        lo = int(rng.integers(0, max(1, T // B))) * B
    else:
        lo = int(rng.integers(0, T))
    m[lo:lo+k*B] = False
    return m


def _sparse_lattice(T, W, rng):
    """A regular lattice of isolated valid samples, so that the valid count per
    window is small and nearly uniform (nv down to 0 or 1)."""
    stride = _loguniform_int(rng, 2, 4*W)
    phase = int(rng.integers(0, stride))
    m = np.zeros(T, dtype=bool)
    m[phase::stride] = True
    return m


# name, probability, builder
MASK_TYPES = (
    ('all-valid',      0.50, _all_valid),
    ('bernoulli',      0.15, _bernoulli),
    ('gap',            0.05, _gap),
    ('one-sided',      0.05, _one_sided),
    ('masked-blocks',  0.05, _masked_blocks),
    ('periodic',       0.04, _periodic),
    ('cluster',        0.04, _cluster),
    ('bimodal',        0.04, _bimodal),
    ('sparse-lattice', 0.04, _sparse_lattice),
    ('all-masked',     0.04, _all_masked),
)

assert abs(sum(t[1] for t in MASK_TYPES) - 1.0) < 1e-12

_PROBS = np.array([t[1] for t in MASK_TYPES])


def random_mask(M, T, W, rng):
    """
    Returns (mask, labels): mask of shape (M,T) dtype bool, and a length-M list
    of the type name used for each row, so that a failing test can report which
    geometry produced it.
    """
    picks = rng.choice(len(MASK_TYPES), size=M, p=_PROBS)
    mask = np.empty((M, T), dtype=bool)
    labels = []
    for i, k in enumerate(picks):
        name, _, builder = MASK_TYPES[k]
        mask[i] = builder(T, W, rng)
        labels.append(name)
    return mask, labels
