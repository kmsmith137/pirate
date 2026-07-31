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

    The profiles exist to cover h_max, which is what conditioning turns on --
    NOT nfreq and not the interval count.  Neither 'one_wide' nor 'no_interior'
    is reachable by drawing cut points uniformly at random, and they are the two
    that approach eps.

    'no_interior' (K = 1, the whole band one knot interval, h_max = nfreq) is the
    extreme.  Measured at nfreq = 10000, eta = 3e-3, against adversarially
    constructed masks: r_min = 5.9e-4, 5.1e-4, 5.1e-4 at n_phi = 1, 2, 3, i.e.
    about 5x eps, and exactly 1 at n_phi = 0 where the zone is a single
    coefficient.  The h^(-4/5) law that governs this regime predicts a further
    factor 2.06 going from nfreq 4096 to 10000 against 2.02 measured, so it
    extrapolates: r_min would not reach eps until nfreq ~ 7.6e4 with no interior
    knots.  Note lambda_min DOES dip below eps there (8.6e-5 at n_phi = 2) while
    r_min does not -- see solve.py on why r_min is the statistic we threshold.
    """
    if nfreq is None:
        nfreq = int(rng.integers(64, 10001))
    if kind is None:
        kind = str(rng.choice(['uniform', 'graded', 'one_wide', 'random',
                               'no_interior']))

    kmax = max(1, min(10, nfreq // (n_phi + 2)))
    K = 1 if kind == 'no_interior' else int(rng.integers(1, kmax + 1))

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


# ================================================================ 2-d masks
#
# Two stages, drawn independently per spectator: a BASE TYPE, then a RECTANGLE
# PERTURBATION.  The perturbation is the part that serves "find problems we did
# not think of", and it is also what keeps the masks NON-SEPARABLE.
#
# Why non-separability is the central concern.  A separable mask m[f,t] =
# m_f[f] & m_t[t] gives an identical G at every live time, so a whole
# (M, nfreq, ntime) draw yields ONE distinct conditioning test rather than
# ntime of them -- most of the apparent sample size is illusory.  It also makes
# the tensor factorization Gcal = (G + eta D_1) kron T exact, which is precisely
# the regime where n=1 degenerates to n=0 and r_min(2d) = r_min(1d).  All the
# genuinely 2-d behaviour lives in the non-separable case.  Separable masks are
# still drawn deliberately (they are the analytically special case that several
# tests assert properties of), just not by accident.
#
# The base type is drawn INDEPENDENTLY PER ZONE.  Zones are already independent
# in the mathematics -- a zone boundary makes G and D_1 exactly block diagonal --
# so one band-wide base type would yield one zone-level test instead of nzone.
# The perturbation is deliberately band-wide and zone-agnostic, so rectangles
# straddle zone boundaries by chance, which is a distinct and realistic case
# better reached naturally than special-cased.
#
# Spectator slices are independent.  Real beams do see correlated RFI, but
# correlation can only reduce the variety per draw, and a bug in the spectator
# axis is better caught by slices that disagree than by slices that agree.
#
# MEASURED OUTPUT DISTRIBUTION.  The weights in MASK_TYPES_2D are provisional and
# are NOT the specification: the rectangle perturbation erodes the base mix
# substantially, and drawing the base per zone makes "fully valid" rarer as
# nzone^-1 in the exponent.  What the tests actually see, at n_phi = 2, n = 2,
# W = 2, ntime = 9, over 300 draws per nfreq (mask_distribution() reproduces it):
#
#     nfreq   all-valid  all-masked  separable  window-const  distinct cols
#       256      0.083      0.023      0.207       0.220        4 of 9
#      1024      0.057      0.030      0.243       0.247        4 of 9
#     10000      0.073      0.043      0.227       0.260        4 of 9
#
#   unmasked fraction, quartiles: about (0.15, 0.49, 0.87) at every nfreq
#   live offsets per zone (n=2, W=2, so 5 offsets, 3 required):
#       0: 2.2%   1: 3.8%   2: 3.8%   3: 6.8%   4: 12.2%   5: 71.2%
#       -> 9.8% of zone-samples fall BELOW the rank threshold, so the >= n+1
#          condition is straddled rather than approached from one side.
#
# The median of 4 distinct columns out of 9 is the number that matters most: the
# previous generator produced 1 or 2, i.e. most of the apparent sample size was
# illusory.  For comparison, an all-separable generator would score exactly 1.
#
# Goal (c) is served by random_mask_1d's explicit kinds rather than by this
# distribution -- test_conditioning cycles them deliberately, and reaches the same
# worst r_min as an offline exhaustive sweep to four significant figures.  Drawn
# at random from the table above the 2-d generator gets within about 17% of that
# in 150 draws, which is adequate for the 2-d tests but is not what pins the
# conditioning margin.

STRIPE_PROB = 0.05


def _powerlaw_extent(rng, T):
    """
    Scale-free stamp length: with |u| uniform on (T^(-1/3), 1) and
    n = floor(1/|u|^3), P(n > x) = (x^(-1/3) - T^(-1/3)) / (1 - T^(-1/3)).

    The point of the clamp is that the distribution stays scale-free: the median
    stamp is 4-6 samples at every T while the mean grows as 9.5, 23, 86 at
    T = 64, 256, 2048.  Small stamps therefore stay common as the array grows,
    which is what keeps fine structure represented at large nfreq.
    """
    if T <= 1:
        return 1
    u = rng.uniform(-1.0, -(T ** (-1.0/3.0)))
    return int(np.clip(np.floor(1.0/abs(u)**3), 1, T))


def _knot_extent(rng, kv):
    """
    A frequency extent expressed as a fraction of a knot interval.

    Drawn alongside the channel-unit power law, because the conditioning extrema
    sit at run lengths that are a fixed FRACTION of a knot interval (measured
    0.06 to 0.10 h), not a fixed number of channels.  At nfreq = 10000 with one
    knot interval the channel-unit power law has median extent 4-6 against an
    extremal length near 900, so on its own it would essentially never land
    there.
    """
    widths = np.diff(np.unique(kv.knots))
    h = int(widths[rng.integers(0, len(widths))]) if len(widths) else kv.nfreq
    frac = float(np.exp(rng.uniform(np.log(0.01), np.log(2.0))))
    return int(np.clip(round(frac*h), 1, kv.nfreq))


def _perturb_rectangles(m, kv, rng):
    """
    Stamp N random rectangles over the base mask, each set entirely masked or
    entirely valid, applied in sequence so later ones overwrite earlier ones.
    That produces nested structure -- an island of valid samples inside a dead
    region -- and makes the result non-separable.

        N = uniform_int(0, 20),  p = uniform(0,1) drawn ONCE per spectator

    p is drawn once so a spectator tends to be mostly-masking or
    mostly-unmasking rather than a wash.

    The stripe probability is not a tuning knob.  The power law's tail approaches
    the full axis but reaches it with probability exactly zero, and a full-width
    stripe -- a channel dead for all time, or all channels dead for a time range
    -- is qualitatively different from a rectangle covering 90% of the axis, not
    merely a larger one.  The stripe flag is drawn once and THEN an axis chosen,
    rather than tested per axis: independent draws would cover the whole plane
    with probability STRIPE_PROB^2 per stamp, which over 20 stamps means a ~5%
    chance per spectator of wiping out the base type entirely.
    """
    nf, nt = m.shape
    # Stamp count is a MIXTURE.  A single uniform(0,20) erodes the base far too
    # hard for goal (a): measured, it left only 0.7% of draws fully valid and a
    # median unmasked fraction of 0.35, where real data is mostly valid.  The
    # light component keeps typical cases typical; the heavy one is what serves
    # goal (d).  Survival of an all-valid base under the heavy component alone is
    # E[(1-p)^N] = H_21/21 = 17.4%, which is where the erosion comes from.
    N = int(rng.integers(0, 4)) if rng.random() < 0.45 else int(rng.integers(0, 21))
    p = rng.uniform(0.0, 1.0)
    for _ in range(N):
        val = bool(rng.random() >= p)          # True = stamp valid, False = mask
        if rng.random() < 0.5:
            df = _powerlaw_extent(rng, nf)
        else:
            df = _knot_extent(rng, kv)
        dt = _powerlaw_extent(rng, nt)
        if rng.random() < STRIPE_PROB:
            if rng.random() < 0.5:
                df = nf
            else:
                dt = nt
        f0 = int(rng.integers(0, max(1, nf - df + 1)))
        t0 = int(rng.integers(0, max(1, nt - dt + 1)))
        m[f0:f0+df, t0:t0+dt] = val
    return m


# ---------------------------------------------------------------- base builders
# Each returns (nfreq, ntime) bool.  All take the same signature so the zoo can
# be a plain table.

def _b_frozen(kv, rng, eta, nt, n, W):
    """
    A frequency mask from the 1-d zoo, held constant in time.

    This is the WINDOW-CONSTANT case, and the only one with exact analytic
    structure: G[t+s] = G for every offset, so the assembled matrix is exactly
    (G + eta D_1) kron T, r_min(2d) = r_min(1d), and n=1 reduces to n=0.  The
    tests that assert those properties must ask for this kind by name and switch
    the perturbation off, since a rectangle would destroy the very property being
    asserted.
    """
    return np.repeat(random_mask_1d(kv, rng, eta)[:, None], nt, axis=1)


def _b_separable(kv, rng, eta, nt, n, W):
    """
    m[f,t] = m_f[f] & m_t[t], the frequency half from this package's zoo and the
    TIME half from detrending_1d's zoo -- which already knows about scan-block
    alignment, long gaps, one-sided windows and narrow off-centre clusters.

    NOT the same as window-constant, and the difference matters.  Here
    G[t] = m_t[t] * G_f, so the DATA block is G_f kron T_w with
    T_w[q,r] = sum_s m_t[s] p_q(s) p_r(s) -- but the regulator contributes
    eta D_1 kron T with the FULL window Gram T, so the sum is a clean Kronecker
    product only when T_w = T, i.e. only when m_t is all-valid across the window.
    That case is _b_frozen(), and it is the one with the exact factorization,
    r_min(2d) = r_min(1d), and n=1 degenerate to n=0.

    Separable is still special -- G_f is the same at every live time, so the draw
    yields few distinct conditioning tests -- which is why it is drawn
    deliberately at a modest weight rather than arising by accident.
    """
    from ..detrending_1d.masks import random_mask as _rm1d
    mf = random_mask_1d(kv, rng, eta)
    mt = _rm1d(1, nt, max(W, 1), rng)[0][0]
    return mf[:, None] & mt[None, :]


def _b_swept(kv, rng, eta, nt, n, W):
    """A masked band whose centre drifts linearly in time (radar)."""
    nf = kv.nfreq
    m = np.ones((nf, nt), dtype=bool)
    width = int(rng.integers(1, max(2, nf//6)))
    c0 = rng.uniform(0, nf)
    slope = rng.uniform(-nf/max(nt, 1), nf/max(nt, 1))
    for t in range(nt):
        c = int(c0 + slope*t)
        m[max(0, c):max(0, c)+width, t] = False
    return m


def _b_widening(kv, rng, eta, nt, n, W):
    """
    A masked band whose WIDTH grows or shrinks with time.

    Unlike a swept band this changes the number of dead coefficients as the
    window slides, so the live-offset count varies within a single window -- the
    shape that exercises the >= n+1 distinct offsets rank condition without
    being constructed for it.
    """
    nf = kv.nfreq
    m = np.ones((nf, nt), dtype=bool)
    c = int(rng.integers(0, nf))
    w0 = rng.uniform(0, nf/4)
    w1 = rng.uniform(0, nf/2)
    for t in range(nt):
        half = int(w0 + (w1-w0)*t/max(nt-1, 1)) // 2
        m[max(0, c-half):c+half+1, t] = False
    return m


def _b_narrowband(kv, rng, eta, nt, n, W):
    """
    A few channels dead for a run of times straddling the window length: does the
    window see anything at all, and does it see the same thing at every offset.
    """
    nf = kv.nfreq
    m = np.ones((nf, nt), dtype=bool)
    cands = [max(1, W), max(1, 2*W), max(1, 2*W+1), max(1, 2*W+2), nt]
    dur = int(cands[rng.integers(len(cands))])
    nch = int(rng.integers(1, kv.n_phi+4))
    f0, t0 = int(rng.integers(0, nf)), int(rng.integers(0, nt))
    m[f0:f0+nch, t0:t0+dur] = False
    return m


def _b_impulse(kv, rng, eta, nt, n, W):
    """All channels dead for a few consecutive samples (a dropped packet)."""
    m = np.ones((kv.nfreq, nt), dtype=bool)
    for _ in range(int(rng.integers(1, 4))):
        t0 = int(rng.integers(0, nt))
        m[:, t0:t0+int(rng.integers(1, max(2, 2*W+2)))] = False
    return m


def _b_n_live(kv, rng, eta, nt, n, W):
    """
    Exactly k live time offsets in a window, k swept over n-1 .. n+2.

    Targets the rank condition directly: below n+1 the assembled matrix is
    exactly singular whatever the channel count at the live offsets, so this is
    the family that must straddle the boundary rather than approach it.
    """
    nf, nwin = kv.nfreq, 2*W+1
    m = np.zeros((nf, nt), dtype=bool)
    k = int(np.clip(n + int(rng.integers(-1, 3)), 0, nwin))
    t0 = int(rng.integers(0, max(1, nt - nwin + 1)))
    if k:
        live = rng.choice(min(nwin, nt - t0), size=min(k, max(1, nt-t0)),
                          replace=False)
        base = random_mask_1d(kv, rng, eta)
        for kk in live:
            m[:, t0+int(kk)] = base
    return m


def _b_survivor_cluster(kv, rng, eta, nt, n, W):
    """A wide masked band with c ADJACENT surviving channels, c swept across the
    rank boundary c = n_phi+1 where the restricted Gram stops gaining rank."""
    nf = kv.nfreq
    m = np.ones((nf, nt), dtype=bool)
    lo = int(rng.integers(0, max(1, nf//2)))
    hi = min(nf, lo + int(rng.integers(max(2, nf//4), max(3, nf))))
    m[lo:hi] = False
    c = int(rng.integers(1, kv.n_phi+3))
    if hi - lo > c:
        s = int(rng.integers(lo, hi-c+1))
        m[s:s+c] = True
    return m


def _b_band_edge(kv, rng, eta, nt, n, W):
    """Rolloff at both ends, where the clamped edge basis functions have reduced
    support and therefore see the least data."""
    nf = kv.nfreq
    m = np.ones((nf, nt), dtype=bool)
    m[:int(rng.integers(0, max(1, nf//4)))] = False
    m[nf-int(rng.integers(0, max(1, nf//4))):] = False
    return m


def _b_dead_zone(kv, rng, eta, nt, n, W):
    """One whole zone masked, for a random run of times."""
    m = np.ones((kv.nfreq, nt), dtype=bool)
    z = int(rng.integers(0, kv.nzone))
    zoc = kv.zone_id[kv.j0]
    t0 = int(rng.integers(0, nt))
    dt = int(rng.integers(1, nt - t0 + 1))
    m[np.flatnonzero(zoc == z)[:, None], np.arange(t0, t0+dt)[None, :]] = False
    return m


def _b_bernoulli(kv, rng, eta, nt, n, W):
    return rng.random((kv.nfreq, nt)) < rng.uniform(0.0, 1.0)


def _b_all_valid(kv, rng, eta, nt, n, W):
    return np.ones((kv.nfreq, nt), dtype=bool)


def _b_all_masked(kv, rng, eta, nt, n, W):
    return np.zeros((kv.nfreq, nt), dtype=bool)


def adversarial_mask_2d(kv, rng, eta, n, W, nt, constrained=True, niter=20):
    """
    The exact worst-case construction, extended from one time sample to the whole
    window.  Returns (nfreq, nt) bool.

    Writing b(f,s) = sum_jq alpha_jq phi_j(f) p_q(s), the equilibrated Rayleigh
    quotient of the assembled 2-d matrix is

        R(w) = (sum_fs w_fs a_fs + A0) / (sum_fs w_fs c_fs + B0)
        a_fs = b(f,s)^2,  c_fs = sum_jq alpha_jq^2 phi_j(f)^2 p_q(s)^2

    with A0, B0 the regulator terms, which are mask-INDEPENDENT because T is the
    window Gram of the time basis and never sees the mask.  Both sums are linear
    in w over the (f,s) grid, so the quotient is linear-fractional there, the
    minimizer at level t is exactly {(f,s) : a_fs < t c_fs}, and Dinkelbach
    converges.  Verified: the decomposition reproduces the true Rayleigh quotient
    to 6.3e-16 over 200 random (n,W,mask,alpha), and no single (f,s) flip
    improves the result in 40/40 configurations.

    TWO MODES, because unconstrained this converges to EXACT singularity and that
    is the true optimum rather than a bug -- the optimizer is free to empty whole
    time offsets, and once a zone drops below n+1 live offsets the matrix is
    singular and the objective is 0.

      constrained=True   reject any iterate that leaves a zone below n+1 live
                         offsets.  The hardest mask that is still a well-posed
                         problem; this is the conditioning stressor.
      constrained=False  let it run.  An excellent generator of rank-deficient
                         configurations, i.e. a corner-case generator rather than
                         a conditioning one.

    The adversarial block covers one window (2W+1 samples) and is placed at a
    random time offset; the remaining samples get a frozen 1-d mask, so at least
    one output window sees the exact adversarial configuration.
    """
    from .basis import BasisTable
    from .regulator import d1_dense
    from .timebasis import TimeBasis
    from .solve import zone_slices

    nwin = min(2*W + 1, nt)
    Phi = BasisTable(kv, np.float64).dense()
    R = d1_dense(kv)
    tb = TimeBasis(n, (nwin - 1)//2, orthogonal=True)
    P = tb.P[:nwin]
    N, K = kv.N_phi, kv.N_phi*(n+1)
    zoc = kv.zone_id[kv.j0]
    TT = P.T @ P

    def live_ok(w):
        return all(sum(1 for k in range(nwin) if w[zoc == z, k].any()) >= n+1
                   for z in range(kv.nzone))

    alpha = rng.standard_normal(K)
    best = None
    for _ in range(niter):
        A = alpha.reshape(N, n+1)
        a = (Phi @ A @ P.T) ** 2
        c = (Phi**2) @ (A**2) @ (P.T**2)
        A0 = eta * float(alpha @ np.kron(R, TT) @ alpha)
        B0 = eta * float(np.sum(A**2 * np.outer(np.diag(R), np.diag(TT))))

        fa, fc = a.ravel(), c.ravel()
        t = A0/max(B0, 1e-300)
        first = int(np.argmin(fa/np.maximum(fc, 1e-300)))
        sel = None
        for _ in range(40):
            sel = fa < t*fc
            if not sel.any():
                sel = np.zeros(fa.size, bool); sel[first] = True
            tn = (fa[sel].sum() + A0)/(fc[sel].sum() + B0)
            if abs(tn - t) <= 1e-14*abs(t):
                break
            t = tn
        w = sel.reshape(a.shape)

        if constrained and not live_ok(w):
            break                      # keep the last admissible iterate
        # Assemble and take the minimizing direction for the next round.
        Gc = np.zeros((K, K))
        wf = w.astype(float)
        for k in range(nwin):
            g = (Phi.T * wf[:, k]) @ Phi
            for q in range(n+1):
                for r in range(n+1):
                    Gc[q::n+1, r::n+1] += P[k, q]*P[k, r]*g
        Gc += eta*np.kron(R, TT)
        dg = np.diag(Gc)
        if not np.all(dg > 0):
            best = w.copy() if best is None else best
            break
        s_ = np.sqrt(dg)
        Gh = Gc/np.outer(s_, s_)
        ev, V = np.linalg.eigh(0.5*(Gh + Gh.T))
        best = w.copy()
        alpha = V[:, 0]/s_

    if best is None:
        best = np.ones((kv.nfreq, nwin), dtype=bool)

    out = np.repeat(random_mask_1d(kv, rng, eta)[:, None], nt, axis=1)
    t0 = int(rng.integers(0, max(1, nt - nwin + 1)))
    out[:, t0:t0+nwin] = best
    return out


def _b_adv(kv, rng, eta, nt, n, W):
    return adversarial_mask_2d(kv, rng, eta, n, W, nt, constrained=True)


def _b_adv_singular(kv, rng, eta, nt, n, W):
    return adversarial_mask_2d(kv, rng, eta, n, W, nt, constrained=False)


# name, provisional weight, builder.  The weights are PROVISIONAL: the output
# distribution is what the tests see, and the perturbation erodes the base mix
# substantially, so the numbers to trust are the measured ones in
# mask_distribution() below and quoted in the module docstring -- not these.
MASK_TYPES_2D = (
    ('all_valid',           0.20, _b_all_valid),
    ('frozen',              0.16, _b_frozen),
    ('separable',           0.09, _b_separable),
    ('adversarial',         0.11, _b_adv),
    ('swept',               0.05, _b_swept),
    ('widening',            0.05, _b_widening),
    ('narrowband',          0.05, _b_narrowband),
    ('n_live_offsets',      0.06, _b_n_live),
    ('survivor_cluster',    0.05, _b_survivor_cluster),
    ('adversarial_singular', 0.05, _b_adv_singular),
    ('impulse',             0.04, _b_impulse),
    ('band_edge',           0.03, _b_band_edge),
    ('dead_zone',           0.03, _b_dead_zone),
    ('bernoulli',           0.02, _b_bernoulli),
    ('all_masked',          0.01, _b_all_masked),
)
assert abs(sum(t[1] for t in MASK_TYPES_2D) - 1.0) < 1e-12
TIME_TYPES = [t[0] for t in MASK_TYPES_2D]
_PROBS_2D = np.array([t[1] for t in MASK_TYPES_2D])


def random_mask_2d(shape, kv, rng, eta, n=0, W=0, time_kind=None, perturb=True):
    """
    (M, nfreq, ntime) boolean mask.

    Base type drawn independently PER ZONE (so one draw gives nzone distinct
    zone-level tests), then a band-wide rectangle perturbation that is
    deliberately zone-agnostic.  Spectator slices are independent.

    'time_kind' pins the base type by name for the tests that need a specific
    regime -- 'separable' for the exact-Kronecker properties, 'n_live_offsets'
    for the rank condition.  'perturb=False' suppresses stage 2, which those same
    tests need, since a rectangle can destroy the property being asserted.
    """
    M_ax, nfreq, ntime = shape
    if nfreq != kv.nfreq:
        raise ValueError(f'random_mask_2d: shape has {nfreq} channels, kv has {kv.nfreq}')
    builders = {t[0]: t[2] for t in MASK_TYPES_2D}
    zoc = kv.zone_id[kv.j0]

    out = np.zeros(shape, dtype=bool)
    for m in range(M_ax):
        for z in range(kv.nzone):
            name = time_kind if time_kind is not None else \
                str(rng.choice(TIME_TYPES, p=_PROBS_2D))
            blk = builders[name](kv, rng, eta, ntime, n, W)
            sel = zoc == z
            out[m][sel] = blk[sel]
        if perturb:
            _perturb_rectangles(out[m], kv, rng)
    return out


def mask_distribution(kv, rng, eta, n=0, W=0, ntime=9, ndraw=400):
    """
    Measure what random_mask_2d actually EMITS, as opposed to what the weights
    above ask for.  Returns a dict; see the module docstring for the numbers this
    produced and why they, rather than the weights, are the specification.
    """
    sep = frac = fullvalid = fullmask = 0
    ncols = []
    for _ in range(ndraw):
        m = random_mask_2d((1, kv.nfreq, ntime), kv, rng, eta, n=n, W=W)[0]
        ncols.append(len({m[:, t].tobytes() for t in range(ntime)}))
        ra, ca = m.any(1), m.any(0)
        sep += int(np.array_equal(m, ra[:, None] & ca[None, :]))
        frac += m.mean()
        fullvalid += int(m.all())
        fullmask += int(not m.any())
    return dict(separable=sep/ndraw, unmasked_fraction=frac/ndraw,
                all_valid=fullvalid/ndraw, all_masked=fullmask/ndraw,
                distinct_cols_median=float(np.median(ncols)),
                distinct_cols_min=int(min(ncols)), ntime=ntime)
