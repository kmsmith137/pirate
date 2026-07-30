"""
Random knot vectors and random masks for the test suite.

THE CENTRAL FACT, which is why this file is as long as it is: RANDOM MASKS DO NOT
FIND THE HARD CASES.  Measured at n_phi=2, K=5, h=3000, eta=1e-3, five thousand
Bernoulli masks at log-uniform density reach a worst r_min of 9.6e-3, while a
constructed adversarial mask reaches 4.1e-4 -- a factor of 23.  A suite built on
Bernoulli masks would report an enormous margin over eps no matter how wrong the
conditioning handling was.

Intuition points the wrong way too.  A single unmasked channel is one of the
SAFEST masks (r_min about 1.6e-2, roughly 40x better than the worst), because the
regulator's null space contains the constants and the fit simply interpolates that
one point.  The dangerous mask is a single contiguous run of a few hundred
channels sitting at a particular fractional offset inside one knot interval, with
everything else masked -- a configuration no one would write down by hand.

So random_mask() is a mixture of three things:

  1. adversarial_mask(): the exact worst-case construction (see below).  This is
     the one that matters.
  2. Structured families drawn from the measured extremal geometry, and from the
     corner cases of the code paths (dead zones, trailing versus leading dead runs,
     M = 0, 1, 2).
  3. Plain Bernoulli, for breadth.  Cheap, and it exercises code paths the other
     two do not, but it must never be the only ingredient.

The knot vector matters as much as the mask, because conditioning depends on the
WIDEST knot interval h_max rather than on nfreq or on the number of intervals.
With non-uniform knots h_max can be many times nfreq/K, so random_knots() targets
h_max directly instead of hoping a random draw produces a wide interval.
"""

import numpy as np

from .knots import KnotVector
from .basis import BasisTable
from .regulator import d1_dense

MASK_TYPES = ['adversarial', 'one_run', 'replicated_runs', 'dead_run', 'tiny',
              'all_unmasked', 'knot_blocks', 'comb', 'bernoulli']


# ---------------------------------------------------------------- knot vectors

def random_knots(rng, n_phi=2, nfreq=None, kind=None):
    """
    A random KnotVector.  'kind' selects the width profile; None draws one.

    The profiles exist to cover h_max, which is what conditioning turns on:
    'one_wide' is the family that came closest to eps in the parameter study, and
    it is not reachable by drawing cut points uniformly at random.
    """
    if nfreq is None:
        nfreq = int(rng.integers(64, 4096))
    if kind is None:
        kind = str(rng.choice(['uniform', 'graded', 'one_wide', 'random']))

    kmax = max(1, min(10, nfreq // (n_phi + 2)))
    K = int(rng.integers(1, kmax + 1))

    if K == 1:
        widths = np.array([nfreq])
    elif kind == 'uniform':
        widths = np.full(K, nfreq // K)
        widths[-1] += nfreq - widths.sum()
    elif kind == 'graded':
        g = np.geomspace(1.0, float(rng.uniform(2, 8)), K)
        widths = np.maximum(1, (g / g.sum() * nfreq).astype(np.int64))
        widths[-1] += nfreq - widths.sum()
    elif kind == 'one_wide':
        narrow = max(1, nfreq // (8 * K))
        widths = np.full(K, narrow)
        widths[int(rng.integers(0, K))] += nfreq - widths.sum()
    else:
        cuts = np.sort(rng.choice(np.arange(1, nfreq), size=K-1, replace=False))
        widths = np.diff(np.concatenate(([0], cuts, [nfreq])))

    if widths.min() < 1 or widths.sum() != nfreq:
        widths = np.full(K, nfreq // K)
        widths[-1] += nfreq - widths.sum()

    cuts = np.cumsum(widths)[:-1]

    # Interior multiplicity: 1 usually, sometimes 2..n_phi (a reduced-continuity
    # knot), sometimes n_phi+1 (a zone boundary, so multi-zone vectors get tested).
    knots = [0] * (n_phi + 1)
    for c in cuts:
        r = rng.random()
        if r < 0.70 or n_phi == 0:
            mult = 1
        elif r < 0.88:
            mult = int(rng.integers(1, n_phi + 1))
        else:
            mult = n_phi + 1
        knots.extend([int(c)] * mult)
    knots.extend([nfreq] * (n_phi + 1))
    return KnotVector(np.array(knots, dtype=np.int64), n_phi, nfreq)


# ---------------------------------------------------------------- adversarial

def adversarial_mask(kv, rng, eta, v=None, niter=25):
    """
    The exact worst-case mask construction.

    For a FIXED coefficient vector v, the equilibrated Rayleigh quotient

        R(w) = (sum_f w_f a_f + A0) / (sum_f w_f b_f + B0),
        a_f = (Phi v)_f^2,  b_f = sum_j phi_j(f)^2 v_j^2,
        A0 = eta v^T D_1 v,  B0 = eta sum_j (D_1)_jj v_j^2,

    is LINEAR-FRACTIONAL in the mask, so its minimizer over w in {0,1}^nfreq has a
    closed form: at level t it is exactly {f : a_f/b_f < t}, and the Dinkelbach
    iteration t <- R(w) converges to the optimum.  Alternating that with
    v <- (minimizing generalized eigenvector) is a genuine descent on the smallest
    equilibrated eigenvalue, not a random search.

    Two consequences worth knowing.  Because a linear-fractional function on the
    box [0,1]^nfreq attains its minimum at a vertex, BINARY masks are optimal --
    real-valued weights can never be worse.  And because the construction depends
    on v only through a_f/b_f, seeding it with structured v (alternating signs, a
    ramp) reaches different extremal families than a random seed.

    Returns a boolean mask of shape (nfreq,) with at least one channel set.
    """
    table = BasisTable(kv, dtype=np.float64)
    Phi = table.dense().astype(np.float64)
    R = d1_dense(kv, dtype=np.float64)
    dR = np.diag(R).copy()
    N = kv.N_phi

    if v is None:
        pick = rng.integers(0, 3)
        if pick == 0:
            v = rng.standard_normal(N)
        elif pick == 1:
            v = np.array([(-1.0) ** j for j in range(N)])
        else:
            v = np.arange(N, dtype=np.float64) - rng.uniform(0, N)
    v = np.asarray(v, dtype=np.float64)

    best = None
    for _ in range(niter):
        a = (Phi @ v) ** 2
        b = (Phi ** 2) @ (v ** 2)
        A0 = eta * float(v @ R @ v)
        B0 = eta * float(dR @ v ** 2)
        ratio = a / np.maximum(b, 1e-300)
        first = int(np.argmin(ratio))

        t = A0 / max(B0, 1e-300)
        sel = None
        for _ in range(40):
            sel = ratio < t
            if not sel.any():
                sel = np.zeros(kv.nfreq, dtype=bool)
                sel[first] = True
            tn = (a[sel].sum() + A0) / (b[sel].sum() + B0)
            if abs(tn - t) <= 1e-14 * abs(t):
                break
            t = tn

        w = sel.astype(np.float64)
        A = (Phi.T * w) @ Phi + eta * R
        dg = np.diag(A)
        if not np.all(dg > 0):
            break
        s = np.sqrt(dg)
        Ah = A / np.outer(s, s)
        ev, V = np.linalg.eigh(0.5 * (Ah + Ah.T))
        if best is None or ev[0] < best[0]:
            best = (ev[0], sel.copy())
        v = V[:, 0] / s

    return best[1] if best is not None else sel


# ---------------------------------------------------------------- structured

def _interval_bounds(kv):
    """(lo, hi) of each non-empty knot interval, in channels."""
    tau = np.unique(kv.knots)
    return [(int(a), int(b)) for a, b in zip(tau[:-1], tau[1:]) if b > a]


def _run_in_interval(kv, rng, q=None, u=None, m=None):
    """
    One contiguous unmasked run inside knot interval q, parameterized by fractional
    offset u and length m rather than by (start, stop).

    That parameterization is the point: the measured worst masks sit at a specific
    fractional position within an interval, with length a fixed fraction of the
    interval width (m ~ 0.06..0.10 h at n_phi=2), and a sampler over raw (start,
    stop) hits that region only by luck.
    """
    bounds = _interval_bounds(kv)
    if q is None:
        q = int(rng.integers(0, len(bounds)))
    lo, hi = bounds[q % len(bounds)]
    h = hi - lo
    if m is None:
        if rng.random() < 0.4:
            m = max(1, int(round(h * rng.uniform(0.04, 0.15))))   # the measured band
        else:
            m = max(1, int(round(np.exp(rng.uniform(0, np.log(max(2, h)))))))
    m = int(min(max(1, m), h))
    if u is None:
        u = float(rng.choice([0.5, 0.5, rng.random(), rng.random()]))
    start = lo + int(round(u * h - m / 2.0))
    start = int(np.clip(start, lo, hi - m))
    w = np.zeros(kv.nfreq, dtype=bool)
    w[start:start+m] = True
    return w


def random_mask_1d(kv, rng, eta, kind=None):
    """One (nfreq,) boolean mask.  'kind' from MASK_TYPES; None draws one."""
    nfreq = kv.nfreq
    if kind is None:
        p = np.array([0.22, 0.16, 0.10, 0.12, 0.08, 0.03, 0.09, 0.06, 0.14])
        kind = str(rng.choice(MASK_TYPES, p=p / p.sum()))

    if kind == 'adversarial':
        return adversarial_mask(kv, rng, eta)

    if kind == 'one_run':
        return _run_in_interval(kv, rng)

    if kind == 'replicated_runs':
        # The same (u, m) in every interval: the "bulk" extremal family.
        u = float(rng.choice([0.5, rng.random()]))
        frac = float(rng.uniform(0.04, 0.15))
        w = np.zeros(nfreq, dtype=bool)
        for q in range(len(_interval_bounds(kv))):
            lo, hi = _interval_bounds(kv)[q]
            w |= _run_in_interval(kv, rng, q=q, u=u,
                                  m=max(1, int(round((hi-lo) * frac))))
        return w

    if kind == 'dead_run':
        # A run of fully masked knot intervals, placed leading / interior /
        # trailing.  Only a TRAILING dead run degrades the Cholesky pivots (the
        # factorization is ordered low channel to high), so all three placements
        # must be generated or the suite would silently depend on that order.
        bounds = _interval_bounds(kv)
        nb = len(bounds)
        w = np.ones(nfreq, dtype=bool)
        L = int(rng.integers(1, nb + 1))
        where = str(rng.choice(['leading', 'interior', 'trailing']))
        if where == 'leading':
            q0 = 0
        elif where == 'trailing':
            q0 = nb - L
        else:
            q0 = int(rng.integers(0, max(1, nb - L + 1)))
        for q in range(q0, min(nb, q0 + L)):
            lo, hi = bounds[q]
            w[lo:hi] = False
        return w

    if kind == 'tiny':
        # M = 0, 1 or 2.  M = 0 is the only case where the matrix is genuinely
        # singular; M = 1 is the case r_min cannot see (see solve.py).
        w = np.zeros(nfreq, dtype=bool)
        nkeep = int(rng.integers(0, 3))
        if nkeep:
            w[rng.choice(nfreq, size=nkeep, replace=False)] = True
        return w

    if kind == 'all_unmasked':
        return np.ones(nfreq, dtype=bool)

    if kind == 'knot_blocks':
        # Masked blocks whose edges are knot-aligned or deliberately offset from
        # the knots, which is where coefficients start to lose support.
        w = np.ones(nfreq, dtype=bool)
        bounds = _interval_bounds(kv)
        for _ in range(int(rng.integers(1, 4))):
            q = int(rng.integers(0, len(bounds)))
            lo, hi = bounds[q]
            span = int(rng.integers(1, kv.n_phi + 3))
            hi2 = bounds[min(len(bounds)-1, q + span - 1)][1]
            off = 0 if rng.random() < 0.5 else int(rng.integers(-(hi-lo), hi-lo+1))
            a = int(np.clip(lo + off, 0, nfreq))
            b = int(np.clip(hi2 + off, 0, nfreq))
            w[a:b] = False
        return w

    if kind == 'comb':
        step = int(rng.integers(2, max(3, min(nfreq, 4096))))
        off = int(rng.integers(0, step))
        w = np.zeros(nfreq, dtype=bool)
        w[off::step] = True
        return w

    # bernoulli
    return rng.random(nfreq) < 10.0 ** rng.uniform(-4, 0)


def random_mask(shape, kv, rng, eta, kind=None):
    """
    (M, nfreq, ntime) boolean mask; each (beam, time) column drawn independently
    from random_mask_1d(), so one array covers many families at once.
    """
    M_ax, nfreq, ntime = shape
    if nfreq != kv.nfreq:
        raise ValueError(f'random_mask: shape has {nfreq} channels, kv has {kv.nfreq}')
    out = np.zeros(shape, dtype=bool)
    for m in range(M_ax):
        for t in range(ntime):
            out[m, :, t] = random_mask_1d(kv, rng, eta, kind=kind)
    return out
