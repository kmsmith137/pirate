"""
Unit tests for the regularized spline detrender.

Run with 'python -m pirate_frb test --dts'.  All tests are pure numpy.

Two of them carry more weight than the rest and are worth knowing about before
changing anything here:

  - test_flat_baseline_exact() asserts that a CONSTANT baseline is removed to
    machine precision, for every mask.  That is not an approximation that happens
    to be good; it is exact, because the basis is a partition of unity on each zone
    and D_1 annihilates the corresponding all-ones vector.  It fails loudly if the
    regulator is ever assembled across a zone boundary, or if the end multiplicity
    rule in knots.py is relaxed.
  - test_conditioning() is what justifies the choice of eps.  It reports the
    smallest r_min it can construct rather than asserting a tight bound, because
    the interesting number is the MARGIN, and a tight assertion would turn a
    change in that margin into an opaque failure.
"""

import numpy as np

from . import masks as msk
from .basis import BasisTable
from .knots import KnotVector
from .reduce import accumulate, evaluate, band_to_dense, tree_sum, CHANNEL_BLOCK
from .reference import detrend_reference
from .regulator import d1_banded, d1_dense
from .solve import solve_normal_equations, zone_slices
from .expand import zone_channel_ranges
from .SplineDetrender import SplineDetrender, ETA_DEFAULT, EPS_FLOAT32, EPS_FLOAT64
from .timebasis import TimeBasis
from .assemble import bandwidth
from .reduce import band_to_dense as _b2d

NW_CASES = [(0, 0), (0, 1), (0, 3), (1, 1), (1, 2), (2, 1), (2, 2), (2, 4)]

# Spline degree under test.  Rebound by run_all() rather than fixed, so that a
# multi-iteration run ('test --dts -n 8') covers 0..3; every test reads it, in
# the same style as the module-level counter below.  Degree 0 is not a rounding
# error in the coverage: there, multiplicity n_phi+1 is 1, so EVERY interior knot
# is a zone boundary, the median zone count goes from 1 to 5, and D_1 is
# identically zero (a zone is a single coefficient, so there are no intra-zone
# differences to penalize) -- the unregularized limit, reached legitimately.

N_PHI = 2

# test_dtype_agreement()'s bound on the float32-vs-float64 disagreement in r_min,
# in units of float32 machine epsilon.  A named constant because the number is
# calibrated rather than chosen, and the calibration is worth recording.
#
# r_min is a Cholesky pivot, i.e. a difference, so its error is roundoff-limited in
# ABSOLUTE terms whatever r_min itself is.  The TYPICAL value is small and the TAIL
# is heavy: measured over 120 seeds, median 13, p90 55, p99 153, max 274.  The
# previous bound of 100 was therefore exceeded on 3.3% of draws, which under the
# default 'test --dts -n 100' is a spurious failure in essentially every run.
#
# 1000 is set from the mechanism rather than from the sample maximum: errors
# propagate through a banded factorization of dimension N with half-bandwidth n_b,
# so O(N n_b) eps_mach ~ 600 is the scale to expect at n = n_phi = 2, and a bound
# below that would be policing the tail of a distribution rather than testing
# anything.  Note this is 4x eps itself (eps = 3e-5 = 252 eps_mach): a disagreement
# this large genuinely flips a near-threshold masking decision, which is assertion
# (c)'s job, not this one's.  This assertion only checks that r_min is
# roundoff-limited rather than garbage; the printed value is what tracks drift.
RMIN_ABS_TOL_EPS = 1000

# Smallest r_min observed across a run of test_conditioning(), so that run_all()
# can print the measured margin over eps.
_worst_rmin = [np.inf]

# Cumulative mask-expansion accounting for test_2d_dtype_agreement(), in the
# style of detrending_1d.tests.  Deliberately module-level and never reset:
# under 'test --dts -n N' each iteration draws fresh knots and masks, and the
# running total across iterations is what tells us whether expansion is firing at
# a sane rate -- a single iteration is far too small a sample to judge from.
#
# Keyed by (n_phi, dtype-eps), because the two precisions use different eps
# (1e-4 vs 1e-7) and therefore expand at different rates BY CONSTRUCTION; pooling
# them would make the number uninterpretable.  n_phi is in the key because at
# n_phi = 0 the regulator is identically zero and r_min is exactly 1, so
# expansion can never fire there and a pooled rate would be diluted by it.
_EXPANSION_2D = {}


def _note_expansion_2d(key, mask_in, mask_out):
    """mask_in restricted to the output range, and the detrender's mask_out."""
    d = _EXPANSION_2D.setdefault(key, {'valid_in': 0, 'expanded': 0})
    d['valid_in'] += int(mask_in.sum())
    d['expanded'] += int((mask_in & ~mask_out).sum())


def _expansion_2d_str():
    if not _EXPANSION_2D:
        return 'none recorded'
    return '  '.join(
        f"n_phi={k[0]},{k[1]}: {d['expanded']}/{d['valid_in']} = "
        f"{(d['expanded']/d['valid_in'] if d['valid_in'] else 0.0):.3e}"
        for k, d in sorted(_EXPANSION_2D.items()))


def _default_rng(rng):
    return np.random.default_rng() if rng is None else rng


def _smooth_baseline(kv, rng, M_ax, ntime, dtype=np.float64):
    """
    A baseline that lies exactly in the spline space, of O(1) amplitude.

    Coefficients are drawn directly rather than by projecting a function, so the
    residual of an unregularized fit would be exactly zero and the only thing left
    is the shrinkage bias.  Amplitude is O(1) deliberately: without offset
    subtraction, a large DC level would consume float32 mantissa for nothing (see
    the SplineDetrender docstring).
    """
    a = rng.standard_normal((M_ax, ntime, kv.N_phi)) * 0.3
    a += rng.standard_normal((M_ax, ntime, 1))
    table = BasisTable(kv, dtype=dtype)
    return evaluate(a.astype(dtype), table), a


# ---------------------------------------------------------------- T1, T2, T3

def test_knots(rng, verbose=True):
    for _ in range(60):
        kv = msk.random_knots(rng, n_phi=N_PHI)
        assert kv.N_phi == len(kv.knots) - kv.n_phi - 1
        assert np.all(np.diff(kv.knots) >= 0)
        assert kv.knots[0] == 0 and kv.knots[-1] == kv.nfreq
        assert np.all(kv.j0 >= kv.n_phi) and np.all(kv.j0 <= kv.N_phi - 1)
        # supp_lo/supp_hi must agree with the evaluated basis.
        table = BasisTable(kv, dtype=np.float64)
        dense = table.dense()
        for j in range(kv.N_phi):
            nz = np.flatnonzero(dense[:, j] != 0)
            if nz.size:
                assert kv.supp_lo[j] <= nz[0] and nz[-1] < kv.supp_hi[j]
        # Zones are contiguous in j, and channel ranges tile [0, nfreq).
        assert np.all(np.diff(kv.zone_id) >= 0)
        rngs = zone_channel_ranges(kv)
        assert rngs[0][0] == 0 and rngs[-1][1] == kv.nfreq
        assert all(rngs[i][1] == rngs[i+1][0] for i in range(len(rngs)-1))

    for bad, msgpart in (
            (dict(knots=np.array([0, 0, 0, 8, 8]), n_phi=2, nfreq=8), 'multiplicity'),
            (dict(knots=np.array([0, 0, 0, 4, 8, 8, 8]), n_phi=2, nfreq=9), 'run from 0'),
            (dict(knots=np.array([0., 0., 0., 8., 8., 8.]), n_phi=2, nfreq=8), 'integer'),
            (dict(knots=np.array([0, 0, 0, 5, 4, 8, 8, 8]), n_phi=2, nfreq=8), 'non-decreasing')):
        try:
            KnotVector(**bad)
            raise AssertionError(f'KnotVector accepted invalid input: {bad}')
        except ValueError as e:
            assert msgpart in str(e), f'unexpected message {e!r}, wanted {msgpart!r}'

    if verbose:
        print('    test_knots: pass')


def test_basis(rng, verbose=True):
    for _ in range(40):
        kv = msk.random_knots(rng, n_phi=N_PHI)
        table = BasisTable(kv, dtype=np.float64)
        dense = table.dense()

        # Compact support: exactly n_phi+1 nonzero basis functions per channel.
        assert np.all((dense != 0).sum(axis=1) <= kv.n_phi + 1)

        # Partition of unity, PER ZONE and exactly.  This is the property the
        # whole regularization scheme rests on: it is what makes the all-ones
        # coefficient vector of a zone equal to the constant function.
        zone_of_channel = kv.zone_id[kv.j0]
        for z in range(kv.nzone):
            sel = kv.zone_id == z
            s = dense[:, sel].sum(axis=1)
            inside = zone_of_channel == z
            assert np.abs(s[inside] - 1).max() < 1e-13, np.abs(s[inside] - 1).max()
            if (~inside).any():
                assert np.abs(s[~inside]).max() == 0.0
        assert np.abs(dense.sum(axis=1) - 1).max() < 1e-13

        # Nonnegativity, and prod really is the pairwise product table.
        assert dense.min() >= 0
        pr = table.phi[:, table.pair_a] * table.phi[:, table.pair_b]
        assert np.array_equal(pr, table.prod)

    # Against the closed form for uniform quadratic B-splines on an interior span.
    h, K = 16, 6
    kv = KnotVector(np.array([0]*3 + list(np.arange(1, K)*h) + [K*h]*3), 2, K*h)
    table = BasisTable(kv, dtype=np.float64)
    q = 3
    f = np.arange(q*h, (q+1)*h)
    u = (f + 0.5 - q*h) / h
    want = np.stack([(1-u)**2/2, (1 + 2*u - 2*u**2)/2, u**2/2], axis=1)
    assert np.abs(table.phi[f] - want).max() < 1e-13

    if verbose:
        print('    test_basis: pass')


def test_regulator(rng, verbose=True):
    for _ in range(40):
        kv = msk.random_knots(rng, n_phi=N_PHI)
        R = d1_dense(kv, dtype=np.float64)
        Rb = d1_banded(kv, dtype=np.float64)

        assert np.allclose(R, R.T)
        # Half-bandwidth 1, regardless of n_phi.
        idx = np.argwhere(np.abs(R) > 0)
        if idx.size:
            assert np.abs(idx[:, 0] - idx[:, 1]).max() <= 1

        # Exactly zero across a zone boundary, and one constant per zone in the
        # null space (not one global constant).
        for z in range(kv.nzone):
            sel = kv.zone_id == z
            other = ~sel
            if other.any():
                assert np.abs(R[np.ix_(sel, other)]).max() == 0.0
            v = sel.astype(np.float64)
            assert np.abs(R @ v).max() == 0.0
        ev = np.linalg.eigvalsh(R)
        nnull = int(np.sum(ev < 1e-12 * max(ev.max(), 1.0)))
        assert nnull == kv.nzone, (nnull, kv.nzone)

        # Banded and dense forms agree.
        N = kv.N_phi
        chk = np.zeros((N, N))
        chk[np.arange(N), np.arange(N)] = Rb[:, 0]
        j = np.arange(N-1)
        chk[j, j+1] = Rb[j, 1]
        chk[j+1, j] = Rb[j, 1]
        assert np.array_equal(chk, R)

    if verbose:
        print('    test_regulator: pass')


# ---------------------------------------------------------------- T4, T5

def test_reduce(rng, verbose=True):
    for _ in range(20):
        kv = msk.random_knots(rng, n_phi=N_PHI, nfreq=int(rng.integers(64, 600)))
        table = BasisTable(kv, dtype=np.float64)
        Phi = table.dense()
        M_ax, ntime = int(rng.integers(1, 3)), int(rng.integers(1, 4))
        mask = msk.random_mask((M_ax, kv.nfreq, ntime), kv, rng, ETA_DEFAULT)
        d = rng.standard_normal((M_ax, kv.nfreq, ntime))

        G, U = accumulate(d, mask, table)
        Gd = band_to_dense(G)
        for m in range(M_ax):
            for t in range(ntime):
                w = mask[m, :, t].astype(np.float64)
                want_G = (Phi.T * w) @ Phi
                want_U = Phi.T @ (w * d[m, :, t])
                assert np.abs(Gd[m, t] - want_G).max() < 1e-9 * max(1, np.abs(want_G).max())
                assert np.abs(U[m, t] - want_U).max() < 1e-9 * max(1, np.abs(want_U).max())

        a = rng.standard_normal((M_ax, ntime, kv.N_phi))
        model = evaluate(a, table)
        for m in range(M_ax):
            for t in range(ntime):
                assert np.abs(model[m, :, t] - Phi @ a[m, t]).max() < 1e-10

    # tree_sum is shape-independent along the other axes, which numpy's own
    # pairwise summation is not.
    x = rng.standard_normal((37, 5)).astype(np.float32)
    for k in (1, 2, 3, 5):
        y = np.repeat(x[:, :, None], k, axis=2)
        assert np.array_equal(tree_sum(y, axis=0)[:, 0], tree_sum(x, axis=0))

    if verbose:
        print('    test_reduce: pass')


def test_chunk_invariance(rng, verbose=True):
    """
    Bit-identical results across time chunking and across spectator batching.

    The frequency reduction is the thing at risk: numpy blocks its pairwise sum by
    stride, so the grouping over frequency would otherwise change with ntime.
    """
    for _ in range(6):
        kv = msk.random_knots(rng, n_phi=N_PHI, nfreq=int(rng.integers(600, 2500)))
        det = SplineDetrender(kv, dtype=np.float32)
        M_ax, ntime = 2, 12
        mask = msk.random_mask((M_ax, kv.nfreq, ntime), kv, rng, det.eta)
        d = rng.standard_normal((M_ax, kv.nfreq, ntime)).astype(np.float32)

        r0, m0, p0 = det.detrend_chunk(d, mask)
        for chunk in (1, 3, 5, 12):
            rs, ms, ps = [], [], []
            for lo in range(0, ntime, chunk):
                hi = min(lo + chunk, ntime)
                r, mm, p = det.detrend_chunk(d[:, :, lo:hi], mask[:, :, lo:hi])
                rs.append(r); ms.append(mm); ps.append(p)
            assert np.array_equal(np.concatenate(rs, axis=2), r0)
            assert np.array_equal(np.concatenate(ms, axis=2), m0)
            assert np.array_equal(np.concatenate(ps, axis=1), p0)

        # Splitting the spectator axis must be inert too.
        r1, _, _ = det.detrend_chunk(d[:1], mask[:1])
        assert np.array_equal(r1, r0[:1])

    assert kv.nfreq > CHANNEL_BLOCK, 'test never exercised multi-block accumulation'
    if verbose:
        print('    test_chunk_invariance: pass')


# ---------------------------------------------------------------- T6

def test_solve(rng, verbose=True):
    for _ in range(25):
        kv = msk.random_knots(rng, n_phi=N_PHI, nfreq=int(rng.integers(64, 800)))
        table = BasisTable(kv, dtype=np.float64)
        D1 = d1_banded(kv)
        Rd = d1_dense(kv)
        eta, eps = ETA_DEFAULT, EPS_FLOAT64
        M_ax, ntime = 1, 6
        mask = msk.random_mask((M_ax, kv.nfreq, ntime), kv, rng, eta)
        d = rng.standard_normal((M_ax, kv.nfreq, ntime))

        G, U = accumulate(d, mask, table)
        a, rmin, bad = solve_normal_equations(G, U, kv, D1, eta, eps)
        Gd = band_to_dense(G)

        for t in range(ntime):
            A = Gd[0, t] + eta * Rd
            for z, (lo, hi) in enumerate(zone_slices(kv)):
                blk = A[lo:hi, lo:hi]
                if not (np.diag(Gd[0, t])[lo:hi].sum() > 0):
                    assert rmin[0, t, z] == 0.0
                    continue
                s = np.sqrt(np.diag(blk))
                Ah = blk / np.outer(s, s)
                want = float((np.diag(np.linalg.cholesky(Ah)) ** 2).min())
                assert abs(rmin[0, t, z] - want) < 1e-9 * max(want, 1e-12), \
                    (rmin[0, t, z], want)
                # The Cholesky pivot always dominates the smallest eigenvalue.
                assert rmin[0, t, z] >= np.linalg.eigvalsh(Ah)[0] - 1e-12
                if not bad[0, t, z]:
                    want_a = np.linalg.solve(blk, U[0, t, lo:hi])
                    scale = max(1.0, np.abs(want_a).max())
                    assert np.abs(a[0, t, lo:hi] - want_a).max() < 1e-7 * scale

    # r_min and the flag must be invariant under rescaling the DATA (the matrix
    # scale changes but the equilibrated pivot does not).  Rescaling eta would of
    # course change things: it is a physical parameter, not a unit.
    kv = msk.random_knots(rng, n_phi=N_PHI, nfreq=512)
    det = SplineDetrender(kv, dtype=np.float64, eps=EPS_FLOAT64)
    mask = msk.random_mask((1, kv.nfreq, 8), kv, rng, det.eta)
    d = rng.standard_normal((1, kv.nfreq, 8))
    _, m0, p0 = det.detrend_chunk(d, mask)
    for scale in (2.0**40, 2.0**-40):
        _, m1, p1 = det.detrend_chunk(d * scale, mask)
        assert np.array_equal(m0, m1)
        assert np.allclose(p0, p1, rtol=1e-12, atol=0)

    if verbose:
        print('    test_solve: pass')


# ---------------------------------------------------------------- T7, T8

def test_flat_baseline_exact(rng, verbose=True):
    """
    A constant baseline is removed EXACTLY, for every mask and every eta.

    Exact, not approximate: the all-ones coefficient vector of a zone is the
    constant function (partition of unity) and lies in the null space of D_1, so
    the regulator applies no penalty to it, A @ 1 = G @ 1 = U/level, and the fit
    reproduces the constant with no shrinkage at any eta.

    The assertion is therefore not "small" but "roundoff": the tolerance is
    derived from the measured conditioning, eps_mach/r_min, rather than picked.
    A fixed tolerance would either be too loose to catch a real regression or --
    since the mask generator deliberately produces badly conditioned cases -- fail
    at random when it happened to draw one.
    """
    worst = 0.0
    for _ in range(25):
        kv = msk.random_knots(rng, n_phi=N_PHI, nfreq=int(rng.integers(64, 10001)),
                              kind='no_interior' if rng.random() < 0.3 else None)
        for dtype in (np.float64, np.float32):
            det = SplineDetrender(kv, dtype=dtype, eps=EPS_FLOAT64)
            M_ax, ntime = 1, 8
            mask = msk.random_mask((M_ax, kv.nfreq, ntime), kv, rng, det.eta)
            level = rng.uniform(0.5, 2.0)
            d = np.full((M_ax, kv.nfreq, ntime), level, dtype=dtype)
            r, mo, p = det.detrend_chunk(d, mask)
            live = p[p > 0]
            rmin = float(live.min()) if live.size else 1.0
            tol = 50 * np.finfo(dtype).eps / rmin
            worst = max(worst, float(np.abs(r).max()) / (level * tol))
            assert np.abs(r).max() < tol * level, \
                (dtype, float(np.abs(r).max()), rmin, tol * level)

        # Also exact for a different constant per (beam, time) sample.
        det = SplineDetrender(kv, dtype=np.float64, eps=EPS_FLOAT64)
        lv = rng.uniform(0.5, 2.0, size=(1, 1, 8))
        d = np.broadcast_to(lv, (1, kv.nfreq, 8)).copy()
        mask = msk.random_mask((1, kv.nfreq, 8), kv, rng, det.eta)
        r, _, p = det.detrend_chunk(d, mask)
        live = p[p > 0]
        assert np.abs(r).max() < 50 * np.finfo(np.float64).eps / \
            (float(live.min()) if live.size else 1.0) * float(lv.max())

    if verbose:
        print(f'    test_flat_baseline_exact: pass  '
              f'[worst residual {worst:.2f} x the roundoff bound]')


def test_shrinkage_bias_bounded(rng, verbose=True):
    """
    For a baseline exactly in the spline space, the leftover residual is bounded
    by a small multiple of eta times the baseline amplitude.

    Deliberately NOT an exactness test.  The regulator biases the fit by design;
    an unregularized detrender would leave exactly zero here.  The bound below is
    loose because the constant depends on how rough the baseline is; what matters
    is that it scales with eta, which the second half checks.
    """
    worst = 0.0
    for _ in range(20):
        kv = msk.random_knots(rng, n_phi=N_PHI, nfreq=int(rng.integers(64, 700)))
        det = SplineDetrender(kv, dtype=np.float64, eps=EPS_FLOAT64)
        d, _ = _smooth_baseline(kv, rng, 1, 6)
        mask = msk.random_mask((1, kv.nfreq, 6), kv, rng, det.eta)
        r, mo, _ = det.detrend_chunk(d, mask)
        amp = np.abs(d).max()
        rel = np.abs(r).max() / amp
        worst = max(worst, rel)
        assert rel < 60 * det.eta, rel

    # Lowering eta must lower the bias, and on a NON-DEGENERATE mask it does so
    # proportionally.  The mask here is fully unmasked deliberately.  Where G has
    # null directions the bias does NOT vanish as eta -> 0: on those directions
    # (G + eta D_1)^-1 -> (eta D_1)^-1, so the eta prefactor cancels and the
    # coefficient is set by the regulator alone at any strength.  Some of that
    # leaks to unmasked channels, so on a masked configuration the bias saturates
    # and a proportionality assertion would flake -- measured, a 16x reduction in
    # eta bought only 2.1x on one random mask.
    kv = msk.random_knots(rng, n_phi=N_PHI, nfreq=512)
    d, _ = _smooth_baseline(kv, rng, 1, 4)
    mask = np.ones((1, kv.nfreq, 4), dtype=bool)
    biases = []
    for eta in (ETA_DEFAULT, ETA_DEFAULT/4, ETA_DEFAULT/16):
        det = SplineDetrender(kv, dtype=np.float64, eta=eta, eps=EPS_FLOAT64)
        r, _, _ = det.detrend_chunk(d, mask)
        biases.append(np.abs(r).max())
    if np.abs(d1_dense(kv)).max() == 0:
        # Degree 0: every interior knot has multiplicity n_phi+1 = 1, so every
        # coefficient is its own zone, D_1 has no intra-zone difference left to
        # penalize, and the regulator is identically zero.  The detrender is then
        # exactly the unregularized block-mean fit, there is no shrinkage bias at
        # any eta, and asserting monotonicity would be asserting that roundoff is
        # monotone.  Assert the stronger thing instead.
        assert max(biases) < 1e-12 * np.abs(d).max(), biases
    else:
        assert biases[0] > biases[1] > biases[2], biases
        assert biases[0] / max(biases[2], 1e-300) > 4, biases

    if verbose:
        print(f'    test_shrinkage_bias_bounded: pass  '
              f'[worst bias/amplitude {worst:.2e}, eta {ETA_DEFAULT:g}]')


# ---------------------------------------------------------------- T9, T10

def test_masked_data_unused(rng, verbose=True):
    for _ in range(15):
        kv = msk.random_knots(rng, n_phi=N_PHI, nfreq=int(rng.integers(64, 700)))
        for dtype in (np.float32, np.float64):
            det = SplineDetrender(kv, dtype=dtype, eps=EPS_FLOAT64)
            M_ax, ntime = 2, 5
            mask = msk.random_mask((M_ax, kv.nfreq, ntime), kv, rng, det.eta)
            d = rng.standard_normal((M_ax, kv.nfreq, ntime)).astype(dtype)
            r0, m0, p0 = det.detrend_chunk(d, mask)

            poison = d.copy()
            junk = np.array([np.inf, -np.inf, np.nan, 1e30, -1e30], dtype=dtype)
            poison[~mask] = junk[rng.integers(0, len(junk), size=int((~mask).sum()))]
            r1, m1, p1 = det.detrend_chunk(poison, mask)

            assert np.array_equal(m0, m1)
            assert np.array_equal(p0, p1)
            assert np.array_equal(r0, r1), 'masked data leaked into the residual'
            assert np.all(np.isfinite(r1))

    if verbose:
        print('    test_masked_data_unused: pass')


def test_spectator_axes(rng, verbose=True):
    """At (n,W) = (0,0) both M and T are spectators: no coupling, either direction."""
    for _ in range(8):
        kv = msk.random_knots(rng, n_phi=N_PHI, nfreq=int(rng.integers(64, 600)))
        det = SplineDetrender(kv, dtype=np.float64, eps=EPS_FLOAT64)
        M_ax, ntime = 3, 7
        mask = msk.random_mask((M_ax, kv.nfreq, ntime), kv, rng, det.eta)
        d = rng.standard_normal((M_ax, kv.nfreq, ntime))
        r, mo, p = det.detrend_chunk(d, mask)

        # Every (m,t) slice equals the (1, nfreq, 1) run of that slice alone.
        for m in range(M_ax):
            for t in range(ntime):
                r1, m1, p1 = det.detrend_chunk(d[m:m+1, :, t:t+1], mask[m:m+1, :, t:t+1])
                assert np.array_equal(r1[0, :, 0], r[m, :, t])
                assert np.array_equal(m1[0, :, 0], mo[m, :, t])
                assert np.array_equal(p1[0, 0], p[m, t])

        # Permuting the spectator axes permutes the output and nothing else.
        pm = rng.permutation(M_ax)
        pt = rng.permutation(ntime)
        r2, m2, p2 = det.detrend_chunk(d[pm][:, :, pt], mask[pm][:, :, pt])
        assert np.array_equal(r2, r[pm][:, :, pt])
        assert np.array_equal(m2, mo[pm][:, :, pt])
        assert np.array_equal(p2, p[pm][:, pt])

    if verbose:
        print('    test_spectator_axes: pass')


# ---------------------------------------------------------------- T11, T12

def test_conditioning(rng, verbose=True, heavy=False):
    """
    Adversarial conditioning sweep: how close does r_min get to eps?

    Asserts only a loose floor, and REPORTS the minimum.  The point of the test is
    to track the margin: a tight assertion here would convert a change in the
    margin into an opaque failure, whereas the number itself is the thing we want
    to see when eta, eps or n_phi change.

    nfreq = 30000 IS RUN BY DEFAULT, and deliberately so: 3.0x is the worst margin
    over eps anywhere in the parameter study, it is the number the constants block
    in SplineDetrender.py quotes to justify eta and eps, and until it was promoted
    here nothing verified it -- an offline sweep is not a test.  It costs about
    0.3 s, so there was never a real reason for it to sit behind 'heavy'.

    'heavy' still adds the remaining large-F configurations from the parameter
    study.  They probe the same regime and are redundant with the pinned pair, so
    they are not run by default.
    """
    eta, eps = ETA_DEFAULT, EPS_FLOAT32
    worst, worst_cfg = np.inf, None

    cfgs = [(int(rng.integers(200, 10001)), None) for _ in range(10)]
    # The two profiles that approach eps are not reachable by chance, so pin them
    # at the top of the nfreq range rather than hoping the draw produces them.
    # h_max, not nfreq, is what conditioning turns on, so the 30000-channel entries
    # are the single-wide-interval profiles rather than 'uniform'.
    cfgs += [(10000, 'no_interior'), (10000, 'one_wide'), (10000, 'no_interior'),
             (4096, 'no_interior'), (30000, 'no_interior'), (30000, 'one_wide')]
    if heavy:
        cfgs += [(30000, 'uniform'), (15000, 'one_wide')]

    kvs = [msk.random_knots(rng, n_phi=N_PHI, nfreq=nfreq, kind=kind)
           for nfreq, kind in cfgs]

    # One fixed geometry on top of the random draws: 30000 channels, four zones,
    # three interior knots each.  It is far from the extremum (h_max is 1875, not
    # 30000, so the margin is ~70x rather than 3x) and that is the point -- it is
    # an operationally plausible configuration, so a regression that made ordinary
    # knot vectors ill-conditioned would show up here rather than only in a corner
    # nobody runs.
    kvs.append(msk.zoned_knots(N_PHI, 30000, 4, 3))

    for kv in kvs:
        det = SplineDetrender(kv, dtype=np.float64, eta=eta, eps=eps)
        table = det.table
        D1 = d1_banded(kv)
        # Cycle the two extremal families explicitly rather than drawing kinds at
        # random.  'adversarial' (Dinkelbach) and 'one_run' (a contiguous run at a
        # measured fractional offset) reach DIFFERENT extrema -- the alternating
        # minimization in the former is a local method and does not reliably find
        # the single-run optimum -- and leaving the choice to chance made the
        # reported margin about 2x optimistic at large h_max.
        for kk in ('adversarial', 'one_run', 'one_run', 'adversarial',
                   'one_run', None):
            w = msk.random_mask_1d(kv, rng, eta, kind=kk)
            mask = w[None, :, None]
            d = rng.standard_normal((1, kv.nfreq, 1))
            G, U = accumulate(d, mask, table)
            a, rmin, bad = solve_normal_equations(G, U, kv, D1, eta, eps)

            for z, (lo, hi) in enumerate(zone_slices(kv)):
                alive = np.diag(band_to_dense(G)[0, 0])[lo:hi].sum() > 0
                if not alive:
                    assert rmin[0, 0, z] == 0.0 and bad[0, 0, z]
                    continue
                # A live zone must have a strictly positive pivot: the regulator
                # guarantees positive definiteness with one unmasked channel.
                assert rmin[0, 0, z] > 0, 'live zone with a non-positive pivot'
                if rmin[0, 0, z] < worst:
                    worst = float(rmin[0, 0, z])
                    worst_cfg = (kv.nfreq, kv.nzone, int(np.diff(np.unique(kv.knots)).max()))
                assert rmin[0, 0, z] > 1e-6, \
                    f'r_min {rmin[0,0,z]:.3e} far below eps={eps:g}; ' \
                    f'nfreq={kv.nfreq}, h_max={np.diff(np.unique(kv.knots)).max()}'

    _worst_rmin[0] = min(_worst_rmin[0], worst)
    if verbose:
        print(f'    test_conditioning: pass  [worst r_min {worst:.3e} = '
              f'{worst/eps:.1f} x eps; nfreq,nzone,h_max = {worst_cfg}]')


def test_zone_expansion(rng, verbose=True):
    """Flagging is per zone, all-or-nothing, and never touches a neighbour."""
    for _ in range(20):
        kv = msk.random_knots(rng, n_phi=N_PHI, nfreq=int(rng.integers(128, 900)))
        # A large eps makes the flag fire often, which is the only practical way to
        # exercise this path: at the production eps it essentially never fires.
        det = SplineDetrender(kv, dtype=np.float64, eps=0.2)
        M_ax, ntime = 1, 8
        mask = msk.random_mask((M_ax, kv.nfreq, ntime), kv, rng, det.eta)
        d = rng.standard_normal((M_ax, kv.nfreq, ntime))
        r, mo, p = det.detrend_chunk(d, mask)

        ranges = zone_channel_ranges(kv)
        for t in range(ntime):
            for z, (lo, hi) in enumerate(ranges):
                if p[0, t, z] < det.eps:
                    assert not mo[0, lo:hi, t].any(), 'flagged zone not fully masked'
                    assert np.all(r[0, lo:hi, t] == 0)
                else:
                    assert np.array_equal(mo[0, lo:hi, t], mask[0, lo:hi, t]), \
                        'unflagged zone was modified'

        # A fully masked zone: r_min exactly zero, output empty.
        mask2 = mask.copy()
        lo, hi = ranges[0]
        mask2[:, lo:hi, :] = False
        r2, m2, p2 = det.detrend_chunk(d, mask2)
        assert np.all(p2[:, :, 0] == 0)
        assert not m2[:, lo:hi, :].any()
        assert np.all(r2[:, lo:hi, :] == 0)
        if len(ranges) > 1:
            lo2, hi2 = ranges[1]
            assert np.array_equal(p2[:, :, 1], p[:, :, 1]), 'zones are not independent'

    # Fully masked input: everything zero, nothing raised.
    kv = msk.random_knots(rng, n_phi=N_PHI, nfreq=256)
    det = SplineDetrender(kv, dtype=np.float32)
    d = rng.standard_normal((1, kv.nfreq, 3)).astype(np.float32)
    r, mo, p = det.detrend_chunk(d, np.zeros((1, kv.nfreq, 3), dtype=bool))
    assert np.all(p == 0) and not mo.any() and np.all(r == 0)

    if verbose:
        print('    test_zone_expansion: pass')


# ---------------------------------------------------------------- T13

def test_reference_agreement(rng, verbose=True):
    for _ in range(10):
        kv = msk.random_knots(rng, n_phi=N_PHI, nfreq=int(rng.integers(64, 500)))
        eta, eps = ETA_DEFAULT, EPS_FLOAT32
        det = SplineDetrender(kv, dtype=np.float64, eta=eta, eps=eps)
        M_ax, ntime = 2, 5
        mask = msk.random_mask((M_ax, kv.nfreq, ntime), kv, rng, eta)
        base, _ = _smooth_baseline(kv, rng, M_ax, ntime)
        d = base + 0.05 * rng.standard_normal(base.shape)

        r0, m0, p0 = det.detrend_chunk(d, mask)
        r1, m1, p1 = detrend_reference(d, mask, kv, n=0, W=0, eta=eta, eps=eps,
                                       dtype=np.float64)

        assert np.array_equal(m0, m1), 'expansion decisions differ from the reference'
        assert np.abs(p0 - p1).max() < 1e-9 * max(1.0, np.abs(p1).max())
        scale = max(1.0, np.abs(d).max())
        assert np.abs(r0 - r1).max() < 1e-8 * scale, np.abs(r0 - r1).max()

    if verbose:
        print('    test_reference_agreement: pass')


# ---------------------------------------------------------------- T14

def test_dtype_agreement(rng, verbose=True):
    """
    float32 at (eta, eps) = (3e-3, 1e-4) against float64 at (3e-3, 1e-7).

    The two runs differ in BOTH dtype and eps, so the assertions separate those
    effects rather than lumping them:

      (a) eps alone: float64 at the two eps values must be bit-identical wherever
          both keep the channel, which isolates the threshold from the arithmetic;
      (b) monotonicity: raising eps can only shrink the kept set;
      (c) decisions: the float32 run uses the looser eps, so any zone it keeps
          should be kept by float64 too -- except for zones whose r_min sits within
          a factor of a few of eps, where float32 rounding of r_min can legitimately
          flip the comparison;
      (d) residuals agree on the channels both keep;
      (e) r_min itself agrees, since it is now an output in its own right.

    Baselines are O(1) with no large offset: without offset subtraction a large DC
    level would consume float32 mantissa and this test would be measuring that
    instead of the algorithm.

    This is where nfreq = 30000 is reached with a RANDOM knot vector, because the
    float32 vs float64 comparison is what would actually catch a large-h_max
    problem.  It does NOT measure the conditioning margin there -- see the note by
    the print; test_conditioning() pins that, and test_production_geometry() pins
    one fixed 30000-channel geometry end to end.
    """
    n_flip, n_zone, worst_rel, worst_resid = 0, 0, 0.0, 0.0
    worst_abs_p = 0.0
    worst_p, worst_cfg = np.inf, None

    for _ in range(12):
        # nfreq reaches 30000 HERE AND NOWHERE ELSE.  Conditioning turns on h_max,
        # and the worst configurations are a nearly-band-wide knot interval at the
        # top of the range -- where the float32/float64 comparison is the thing
        # that would actually catch a problem.  Confined to this test because a
        # 30000-channel accumulate is not worth paying for in the other twenty.
        kv = msk.random_knots(rng, n_phi=N_PHI, nfreq=int(rng.integers(128, 30001)),
                              kind='no_interior' if rng.random() < 0.4 else None)
        M_ax, ntime = 1, 10
        # Cycle the extremal kinds explicitly rather than letting random_mask draw
        # them.  At large h_max the conditioning extremum is reached only by
        # 'adversarial' and 'one_run', and a random draw finds neither reliably --
        # measured, it reported a margin about 2.6x better than the truth, which
        # is exactly the direction that matters.
        kinds = ('adversarial', 'one_run', 'one_run', 'adversarial',
                 'one_run', None, 'knot_blocks', 'bernoulli', 'dead_run', None)
        mask = np.stack([msk.random_mask_1d(kv, rng, ETA_DEFAULT, kind=kinds[t])
                         for t in range(ntime)], axis=1)[None, :, :]
        base, _ = _smooth_baseline(kv, rng, M_ax, ntime)
        d = base + 0.05 * rng.standard_normal(base.shape)
        d32 = d.astype(np.float32)

        det32 = SplineDetrender(kv, dtype=np.float32, eps=EPS_FLOAT32)
        det64 = SplineDetrender(kv, dtype=np.float64, eps=EPS_FLOAT64)
        det64_loose = SplineDetrender(kv, dtype=np.float64, eps=EPS_FLOAT32)

        r32, m32, p32 = det32.detrend_chunk(d32, mask)
        r64, m64, p64 = det64.detrend_chunk(d, mask)
        r64l, m64l, p64l = det64_loose.detrend_chunk(d, mask)

        # (a) eps changes decisions but not arithmetic.
        assert np.array_equal(p64, p64l), 'r_min depends on eps'
        both = m64 & m64l
        assert np.array_equal(np.where(both, r64, 0), np.where(both, r64l, 0))

        # (b) monotone in eps.
        for eps_a, eps_b in ((EPS_FLOAT64, 1e-5), (1e-5, EPS_FLOAT32), (EPS_FLOAT32, 1e-3)):
            ma = SplineDetrender(kv, dtype=np.float64, eps=eps_a).detrend_chunk(d, mask)[1]
            mb = SplineDetrender(kv, dtype=np.float64, eps=eps_b).detrend_chunk(d, mask)[1]
            assert np.all(mb <= ma), 'kept set grew when eps increased'

        # (c) decisions agree away from the threshold band.
        band = (p64 > EPS_FLOAT32 / 3) & (p64 < 3 * EPS_FLOAT32)
        keep32 = ~(p32 < EPS_FLOAT32)
        keep64l = ~(p64l < EPS_FLOAT32)
        disagree = (keep32 != keep64l) & ~band
        n_flip += int(disagree.sum())
        n_zone += int(disagree.size)

        # (d) residuals on the intersection.
        inter = m32 & m64
        if inter.any():
            scale = float(np.abs(d).max())
            worst_resid = max(worst_resid,
                              float(np.abs(r32.astype(np.float64) - r64)[inter].max() / scale))

        live = p64[p64 > 0]
        if live.size and float(live.min()) < worst_p:
            worst_p = float(live.min())
            worst_cfg = (kv.nfreq, kv.nzone, int(np.diff(np.unique(kv.knots)).max()))

        # (e) r_min agreement, asserted in ABSOLUTE terms.  r_min is a Cholesky
        # pivot, computed as a difference, so its error is roundoff-limited in
        # absolute terms and the RELATIVE error therefore scales as
        # eps_mach/r_min -- large exactly where r_min is small, i.e. where the
        # zone is masked anyway and the value is irrelevant.  Measured, the
        # absolute error sits at about 13*eps_mach across four decades of r_min
        # while the relative error spans 1.8e-7 to 8.0e-3.  Asserting on the
        # relative figure would need recalibration every time eps moves, which is
        # exactly how this test broke when eps went from 1e-4 to 3e-5.
        ok = p64 > 0
        if ok.any():
            ae = np.abs(p32.astype(np.float64) - p64)[ok]
            worst_abs_p = max(worst_abs_p, float(ae.max()))
            worst_rel = max(worst_rel, float((ae / p64[ok]).max()))

    assert n_flip <= 1e-3 * max(n_zone, 1), \
        f'{n_flip}/{n_zone} zone decisions flipped outside the threshold band'
    assert worst_resid < 1e-3, f'float32 vs float64 residual {worst_resid:.3e}'
    assert worst_abs_p < RMIN_ABS_TOL_EPS*np.finfo(np.float32).eps, \
        f'float32 vs float64 r_min absolute error {worst_abs_p:.3e} ' \
        f'= {worst_abs_p/np.finfo(np.float32).eps:.0f} x eps_mach'

    if verbose:
        print(f'    test_dtype_agreement: pass  [resid {worst_resid:.2e}, '
              f'r_min abs {worst_abs_p:.2e} (rel {worst_rel:.2e}), '
              f'flips {n_flip}/{n_zone}]')
        # Reported, not asserted, and NOT the conditioning margin.  Sampling 120
        # masks cannot reach the adversarial extremum -- measured, this number is
        # eta-independent while the true worst r_min scales as eta^(4/5), because
        # what the sampler finds is data-dominated rather than regulator-limited.
        # The authoritative margin comes from an offline exhaustive sweep; this is
        # only the smallest r_min that happened to be exercised.
        print(f'      smallest r_min exercised {worst_p:.3e} = '
              f'{worst_p/EPS_FLOAT32:.1f} x eps32 (not the margin; see the code) '
              f'at nfreq,nzone,h_max = {worst_cfg}')



# ================================================================ 2-d (time) tests

def test_time_basis(rng, verbose=True):
    """Orthogonality, parity, and the evaluation vector -- everything downstream
    of timebasis.py assumes all three."""
    for n in (0, 1, 2):
        for W in range(max(1, (n+1+1)//2), 9):
            if 2*W+1 < n+1:
                continue
            tb = TimeBasis(n, W, orthogonal=True)
            # T = I exactly, which is what makes r_min(2-d) = r_min(1-d).
            assert np.abs(tb.T - np.eye(n+1)).max() < 1e-12, np.abs(tb.T-np.eye(n+1)).max()
            assert abs(np.linalg.eigvalsh(tb.T)[0] - 1.0) < 1e-12
            # Definite parity in s: p_q(-s) = (-1)^q p_q(s).  moments.py folds the
            # window on this, and the n=1 degeneracy depends on it being exact.
            for q in range(n+1):
                assert np.abs(tb.P[::-1, q] - tb.parity[q]*tb.P[:, q]).max() < 1e-13
            # eval0 is p_q(0), not delta_q0 -- see the timebasis docstring.
            assert np.abs(tb.eval0 - tb.P[W]).max() == 0.0
            # Monomials: T is Hankel (depends on q+r), orthogonal ones are not.
            tm = TimeBasis(n, W, orthogonal=False)
            for q in range(n+1):
                for r in range(n+1):
                    for q2 in range(n+1):
                        for r2 in range(n+1):
                            if q+r == q2+r2:
                                assert abs(tm.T[q, r] - tm.T[q2, r2]) < 1e-9*max(1, abs(tm.T[q, r]))
            assert np.abs(tm.eval0 - np.eye(n+1)[0]).max() < 1e-13
    if verbose:
        print('    test_time_basis: pass')


def test_bandwidth(rng, verbose=True):
    """
    The assembled matrix is banded only in coefficient-major order, and the
    half-bandwidth is max(n_phi,1)(n+1)+n -- the max() because the regulator has
    half-bandwidth 1 in j regardless of n_phi, so at n_phi = 0 it is the WIDER of
    the two contributions.  Getting that wrong writes out of bounds rather than
    producing a wrong answer, but only at n_phi = 0.
    """
    for _ in range(6):
        kv = msk.random_knots(rng, n_phi=N_PHI, nfreq=int(rng.integers(200, 600)))
        for n, W in NW_CASES:
            det = SplineDetrender(kv, n=n, W=W, dtype=np.float64)
            d = rng.standard_normal((1, kv.nfreq, 3 + 2*W))
            m = msk.random_mask_2d((1, kv.nfreq, 3 + 2*W), kv, rng, det.eta,
                                   time_kind='bernoulli')
            G, U = accumulate(d, m, det.table)
            from .moments import window_moments
            from .assemble import assemble
            Mcal, Vcal = window_moments(G, U, det.tb, 3)
            A, _ = assemble(Mcal, Vcal, kv, det.tb, det.D1, det.eta)
            assert A.shape[-1] == bandwidth(kv, n) + 1
            dense = _b2d(A)[0, 0]
            idx = np.argwhere(np.abs(dense) > 1e-13*max(np.abs(dense).max(), 1e-30))
            if idx.size:
                got = int(np.abs(idx[:, 0] - idx[:, 1]).max())
                assert got <= bandwidth(kv, n), (got, bandwidth(kv, n), n, W)
    if verbose:
        print('    test_bandwidth: pass')


def test_2d_reference_agreement(rng, verbose=True):
    worst = 0.0
    for _ in range(4):
        kv = msk.random_knots(rng, n_phi=N_PHI, nfreq=int(rng.integers(64, 300)))
        for n, W in NW_CASES:
            det = SplineDetrender(kv, n=n, W=W, dtype=np.float64, eps=EPS_FLOAT64)
            T = 4
            m = msk.random_mask_2d((1, kv.nfreq, T + 2*W), kv, rng, det.eta, n=n, W=W)
            base, _ = _smooth_baseline(kv, rng, 1, T + 2*W)
            d = base + 0.05*rng.standard_normal(base.shape)
            r0, m0, p0 = det.detrend_chunk(d, m)
            r1, m1, p1 = detrend_reference(d, m, kv, n=n, W=W, eta=det.eta,
                                           eps=det.eps, dtype=np.float64)
            assert np.array_equal(m0, m1), (n, W)
            assert np.abs(p0 - p1).max() < 1e-9*max(1.0, np.abs(p1).max()), (n, W)
            e = np.abs(r0 - r1).max() / max(1.0, np.abs(d).max())
            worst = max(worst, e)
            assert e < 1e-8, (n, W, e)
    if verbose:
        print(f'    test_2d_reference_agreement: pass  [worst {worst:.2e}]')


def test_2d_flat_baseline_exact(rng, verbose=True):
    """
    The 2-d generalization of test_flat_baseline_exact, and strictly stronger:
    a baseline constant in frequency within a zone and an arbitrary polynomial of
    degree <= n in TIME is removed to roundoff, at any eta.  Both halves matter --
    the frequency half is D_1's null space, the time half is that the model can
    represent any degree-n polynomial exactly.
    """
    worst = 0.0
    for _ in range(6):
        kv = msk.random_knots(rng, n_phi=N_PHI, nfreq=int(rng.integers(64, 500)))
        for n, W in NW_CASES:
            det = SplineDetrender(kv, n=n, W=W, dtype=np.float64, eps=EPS_FLOAT64)
            T = 6
            nbuf = T + 2*W
            tt = np.arange(nbuf, dtype=np.float64) - (nbuf-1)/2.0
            coef = rng.standard_normal(n+1)
            level = sum(coef[q]*tt**q for q in range(n+1))       # degree-n in time
            d = np.broadcast_to(level[None, None, :], (1, kv.nfreq, nbuf)).copy()
            m = msk.random_mask_2d((1, kv.nfreq, nbuf), kv, rng, det.eta, n=n, W=W)
            r, mo, p = det.detrend_chunk(d, m)
            live = p[p > 0]
            rmin = float(live.min()) if live.size else 1.0
            tol = 200*np.finfo(np.float64).eps/rmin*max(1.0, np.abs(d).max())
            worst = max(worst, float(np.abs(r).max())/max(tol, 1e-300))
            assert np.abs(r).max() < tol, (n, W, float(np.abs(r).max()), tol)
    if verbose:
        print(f'    test_2d_flat_baseline_exact: pass  '
              f'[worst {worst:.2f} x the roundoff bound]')


def test_n1_degeneracy(rng, verbose=True):
    """
    n=1 is EXACTLY n=0 when the mask is window-constant, and differs when it is
    not.  Both halves are asserted: the first pins the odd moments vanishing
    exactly, the second is the entire reason n=1 is implemented at all.
    """
    same, diff = 0.0, 0.0
    for _ in range(8):
        kv = msk.random_knots(rng, n_phi=N_PHI, nfreq=int(rng.integers(64, 400)))
        W, T = int(rng.integers(1, 5)), 4
        d0 = SplineDetrender(kv, n=0, W=W, dtype=np.float64, eps=EPS_FLOAT64)
        d1 = SplineDetrender(kv, n=1, W=W, dtype=np.float64, eps=EPS_FLOAT64)
        base, _ = _smooth_baseline(kv, rng, 1, T + 2*W)
        d = base + 0.05*rng.standard_normal(base.shape)

        mc = msk.random_mask_2d((1, kv.nfreq, T+2*W), kv, rng, d0.eta, n=1, W=W,
                                time_kind='frozen', perturb=False)
        r0 = d0.detrend_chunk(d, mc)[0]
        r1 = d1.detrend_chunk(d, mc)[0]
        same = max(same, float(np.abs(r0-r1).max()))
        assert np.abs(r0-r1).max() < 1e-9*max(1.0, np.abs(d).max()), 'n=1 != n=0'

        mv = msk.random_mask_2d((1, kv.nfreq, T+2*W), kv, rng, d0.eta, n=1, W=W,
                                time_kind='bernoulli')
        rr0 = d0.detrend_chunk(d, mv)[0]
        rr1 = d1.detrend_chunk(d, mv)[0]
        both = (d0.detrend_chunk(d, mv)[1] & d1.detrend_chunk(d, mv)[1])
        if both.any():
            diff = max(diff, float(np.abs(rr0-rr1)[both].max()))
    assert diff > 1e-6, f'n=1 never differed from n=0 ({diff:.2e}); odd moments dropped?'
    if verbose:
        print(f'    test_n1_degeneracy: pass  [window-constant {same:.2e}, '
              f'time-varying {diff:.2e}]')


def test_time_rank_deficiency(rng, verbose=True):
    """
    A zone needs data at >= n+1 DISTINCT window offsets.  Below that the assembled
    matrix is exactly singular no matter how many channels survive at the offsets
    that do carry data, and the zone must be flagged.
    """
    for _ in range(8):
        kv = msk.random_knots(rng, n_phi=N_PHI, nfreq=int(rng.integers(64, 400)))
        for n, W in ((1, 2), (2, 2), (2, 3)):
            det = SplineDetrender(kv, n=n, W=W, dtype=np.float64, eps=EPS_FLOAT64)
            nbuf = 1 + 2*W
            d = rng.standard_normal((1, kv.nfreq, nbuf))
            for nlive in range(0, min(n+3, nbuf)+1):
                m = np.zeros((1, kv.nfreq, nbuf), dtype=bool)
                # 'nlive' fully-unmasked offsets, the rest fully masked.
                for k in rng.choice(nbuf, size=nlive, replace=False):
                    m[0, :, k] = True
                r, mo, p = det.detrend_chunk(d, m)
                if nlive < n+1:
                    assert np.all(p == 0), (n, W, nlive, p)
                    assert not mo.any() and np.all(r == 0)
                else:
                    assert np.all(p > 0), (n, W, nlive, p.min())
    if verbose:
        print('    test_time_rank_deficiency: pass')


def test_2d_chunk_invariance(rng, verbose=True):
    """
    Bit-identical across time chunking, given the caller supplies the halo.
    The stencil sums the same 2W+1 buffer samples in the same order for every
    output regardless of the chunk length, so this holds by construction; the
    test is here to catch anyone replacing it with something shape-dependent.
    """
    for _ in range(5):
        kv = msk.random_knots(rng, n_phi=N_PHI, nfreq=int(rng.integers(600, 1600)))
        n, W = NW_CASES[int(rng.integers(0, len(NW_CASES)))]
        det = SplineDetrender(kv, n=n, W=W, dtype=np.float32)
        T = 12
        nbuf = T + 2*W
        m = msk.random_mask_2d((2, kv.nfreq, nbuf), kv, rng, det.eta, n=n, W=W)
        d = rng.standard_normal((2, kv.nfreq, nbuf)).astype(np.float32)
        r0, m0, p0 = det.detrend_chunk(d, m)
        for chunk in (1, 2, 5, T):
            rs, ms, ps = [], [], []
            for lo in range(0, T, chunk):
                hi = min(lo+chunk, T)
                r, mm, p = det.detrend_chunk(d[:, :, lo:hi+2*W], m[:, :, lo:hi+2*W])
                rs.append(r); ms.append(mm); ps.append(p)
            assert np.array_equal(np.concatenate(rs, axis=2), r0), (n, W, chunk)
            assert np.array_equal(np.concatenate(ms, axis=2), m0)
            assert np.array_equal(np.concatenate(ps, axis=1), p0)
        # M remains a pure spectator even though T no longer is.
        assert np.array_equal(det.detrend_chunk(d[:1], m[:1])[0], r0[:1])
    if verbose:
        print('    test_2d_chunk_invariance: pass')


def test_2d_conditioning(rng, verbose=True):
    """
    r_min must not degrade going from 1-d to 2-d.  For a window-constant mask it
    should be EQUAL: the Cholesky factor of A kron T is L_A kron L_T so the pivots
    multiply, and an orthogonal time basis makes T = I.  This is the test that
    justifies keeping eps unchanged from the 1-d detrender.
    """
    worst_ratio, worst_abs = np.inf, np.inf
    for _ in range(8):
        kv = msk.random_knots(rng, n_phi=N_PHI, nfreq=int(rng.integers(200, 800)))
        for n, W in NW_CASES:
            if W == 0:
                continue
            det2 = SplineDetrender(kv, n=n, W=W, dtype=np.float64, eps=EPS_FLOAT64)
            det1 = SplineDetrender(kv, n=0, W=0, dtype=np.float64, eps=EPS_FLOAT64)
            nbuf = 1 + 2*W
            d = rng.standard_normal((1, kv.nfreq, nbuf))
            m = msk.random_mask_2d((1, kv.nfreq, nbuf), kv, rng, det2.eta,
                                   time_kind='frozen', perturb=False)
            p2 = det2.detrend_chunk(d, m)[2]
            p1 = det1.detrend_chunk(d[:, :, W:W+1], m[:, :, W:W+1])[2]
            live = p1 > 0
            if not live.any():
                continue
            assert np.abs(p2[live] - p1[live]).max() < 1e-10, \
                f'window-constant r_min not preserved at n={n}, W={W}'
            # Time-varying: no factorization, so only require it stays sane.
            mv = msk.random_mask_2d((1, kv.nfreq, nbuf), kv, rng, det2.eta,
                                    time_kind='bernoulli')
            pv = det2.detrend_chunk(d, mv)[2]
            if (pv > 0).any():
                worst_abs = min(worst_abs, float(pv[pv > 0].min()))
    assert worst_abs > 1e-6, worst_abs
    if verbose:
        print(f'    test_2d_conditioning: pass  [window-constant r_min preserved '
              f'exactly; worst time-varying r_min {worst_abs:.3e}]')


def test_2d_dtype_agreement(rng, verbose=True):
    """
    float32 at EPS_FLOAT32 vs float64 at EPS_FLOAT64, swept over (n, W).

    Also the only place mask expansion is tallied.  The point of the tally is not
    a pass/fail threshold but a number to look at: a sudden change in the rate is
    a signal even when every assertion still passes.

    THE TWO PRECISIONS NORMALLY REPORT THE SAME COUNT, and that is the expected
    state rather than a broken tally.  Expansion here is essentially all
    RANK-driven, not eps-driven: measured over these configurations, 14.1% of
    zone-samples had r_min exactly 0 (a zone failing the >= n+1 distinct live
    offsets condition, which is structural and precision-independent) and 0 of
    320 fell in 0 < r_min < eps.  Since r_min = 0 is below both thresholds, the
    f32 and f64 runs flag the same zones despite their eps differing by 1000x.
    They are still counted separately because a DIVERGENCE between them is
    exactly the signal that eps has begun to bite, which is the thing worth
    noticing.
    """
    worst = 0.0
    iter_counts = {}
    for _ in range(4):
        kv = msk.random_knots(rng, n_phi=N_PHI, nfreq=int(rng.integers(128, 800)))
        for n, W in NW_CASES:
            T = 5
            nbuf = T + 2*W
            m = msk.random_mask_2d((1, kv.nfreq, nbuf), kv, rng, ETA_DEFAULT, n=n, W=W)
            base, _ = _smooth_baseline(kv, rng, 1, nbuf)
            d = base + 0.05*rng.standard_normal(base.shape)
            a32 = SplineDetrender(kv, n=n, W=W, dtype=np.float32)
            a64 = SplineDetrender(kv, n=n, W=W, dtype=np.float64)
            r32, m32, _ = a32.detrend_chunk(d.astype(np.float32), m)
            r64, m64, _ = a64.detrend_chunk(d, m)

            # The input mask restricted to the output range is what expansion is
            # measured against; the halo is never emitted.
            m_in = (m[:, :, W:W+T] != 0)
            for tag, mo in (('f32', m32), ('f64', m64)):
                _note_expansion_2d((N_PHI, tag), m_in, mo)
                c = iter_counts.setdefault(tag, [0, 0])
                c[0] += int((m_in & ~mo).sum())
                c[1] += int(m_in.sum())

            inter = m32 & m64
            if inter.any():
                e = float(np.abs(r32.astype(np.float64) - r64)[inter].max()
                          / max(1.0, np.abs(d).max()))
                worst = max(worst, e)
    assert worst < 1e-3, worst
    if verbose:
        this = '  '.join(f'{t}: {c[0]}/{c[1]} = '
                         f'{(c[0]/c[1] if c[1] else 0.0):.3e}'
                         for t, c in sorted(iter_counts.items()))
        print(f'    test_2d_dtype_agreement: pass  [resid {worst:.2e}]')
        print(f'      mask expansion this iteration: {this}')
        print(f'      cumulative:                    {_expansion_2d_str()}')


def test_production_geometry(rng, verbose=True):
    """
    One pinned end-to-end run at the geometry a GPU kernel would be compiled for:
    n_phi = 2, nfreq = 30000, four zones of three interior knots each, n = 2,
    W = 4, M = 2.

    Everything else in this file sweeps; this one deliberately does not.  It pins
    n_phi, n and W rather than reading N_PHI and NW_CASES, because the point is to
    exercise the one configuration that gets compiled, and to leave measured
    numbers behind -- worst r_min, and the float32-vs-float64 residual -- that a
    kernel's tolerances can be set from.

    Two things here are not covered elsewhere.  The knot intervals are 1875
    channels wide, about 2x what the random draws reach at K up to 10, so this is
    the widest float32 frequency accumulation the suite performs -- and the
    frequency reduction is exactly what a port is most likely to get subtly wrong,
    since reduce.py's binary tree exists for that reason and is easy to replace
    with something that looks equivalent and is not.  And M = 2 with four zones
    means the (beam, zone) index pair is non-degenerate in both slots at once,
    which the single-beam tests cannot check.
    """
    n_phi, n, W, M_ax, T = 2, 2, 4, 2, 8
    nbuf = T + 2*W
    kv = msk.zoned_knots(n_phi, 30000, 4, 3)
    assert (kv.nzone, kv.N_phi) == (4, 24), (kv.nzone, kv.N_phi)

    det32 = SplineDetrender(kv, n=n, W=W, dtype=np.float32)
    det64 = SplineDetrender(kv, n=n, W=W, dtype=np.float64)

    m = msk.random_mask_2d((M_ax, kv.nfreq, nbuf), kv, rng, ETA_DEFAULT, n=n, W=W)
    base, _ = _smooth_baseline(kv, rng, M_ax, nbuf)
    d = base + 0.05*rng.standard_normal(base.shape)

    r32, m32, p32 = det32.detrend_chunk(d.astype(np.float32), m)
    r64, m64, p64 = det64.detrend_chunk(d, m)

    # The float32-vs-float64 residual is scaled against eps_mach/r_min rather than
    # compared to a constant, and r_min here is the worst over the zone-samples the
    # comparison actually sees -- those BOTH runs kept.  A flat threshold does not
    # work: the mask generator routinely produces a zone sitting just above eps,
    # which survives the cut and then carries an error of order eps_mach/(4 r_min)
    # by design (solve.py), so the residual moves by more than an order of
    # magnitude from draw to draw while nothing is wrong.  Measured over 24 draws
    # at this geometry the ratio below runs 0.005 to 3.1, median 0.13.
    kept = (p32 >= EPS_FLOAT32) & (p64 > 0)
    rmin = float(p32[kept].min()) if kept.any() else 1.0
    inter = m32 & m64
    resid = float(np.abs(r32.astype(np.float64) - r64)[inter].max()) if inter.any() else 0.0
    tol = np.finfo(np.float32).eps / rmin * max(1.0, float(np.abs(d).max()))
    assert resid < 50*tol, (resid, tol)

    # The exact-reproduction property at this geometry, in float32: constant in
    # frequency within each zone, degree-n in time.  This is the assertion that
    # would catch a regulator assembled across a zone boundary, and doing it in
    # float32 at 30000 channels is what makes it a statement about the working
    # precision rather than about float64.
    tt = np.arange(nbuf, dtype=np.float64) - (nbuf-1)/2.0
    coef = rng.standard_normal(n+1)
    level = sum(coef[q]*tt**q for q in range(n+1))
    dflat = np.broadcast_to(level[None, None, :], (M_ax, kv.nfreq, nbuf)).astype(np.float32)
    rf, mf, pf = det32.detrend_chunk(dflat, m)
    livef = pf[pf > 0]
    rminf = float(livef.min()) if livef.size else 1.0
    tolf = np.finfo(np.float32).eps/rminf*max(1.0, float(np.abs(dflat).max()))
    flat = float(np.abs(rf).max())
    assert flat < 200*tolf, (flat, tolf)

    # Both errors are reported in the same unit, eps_mach/r_min times the data
    # scale, so the two lines are directly comparable and neither depends on the
    # particular mask draw the way a raw magnitude would.
    if verbose:
        exp = int(((m[:, :, W:W+T] != 0) & ~m32).sum())
        print(f'    test_production_geometry: pass  [f32-vs-f64 resid {resid:.2e} '
              f'= {resid/tol:.3f} x eps_mach/r_min at worst kept r_min {rmin:.2e} '
              f'({rmin/EPS_FLOAT32:.1f} x eps); flat baseline {flat/tolf:.1e} x the '
              f'same bound; {exp} channel-samples expanded]')


def test_gpu_kernel(rng=None, verbose=True, nfreq=1024, M_ax=2):
    """
    Compare pirate.Detrender2d (the GPU kernel) to SplineDetrender.

    Structured like detrending_1d.tests.test_gpu_kernel(): the GPU runs in float32 and
    the reference in float64, so the two do not produce identical masks and residuals
    are compared on the intersection.  Data is generated in fp64, cast to fp32, then
    cast back for the reference, so both see bit-identical inputs.

    Everything the kernel does not fix is swept: the knot vector, nfreq, the mask kind,
    the window half-width W, the time-polynomial degree n, and the chunk length T.  n_phi
    is the one compile-time parameter, so the test covers every value the kernel was
    compiled for, read back from configs().

Two parts, and what separates them is HOW MANY KNOT VECTORS THEY USE, not what kind
    of property they check:

      - PART 1 runs every check, at ONE fixed knot vector, once per (n,W) in FIXED_CASES.
        Most of these are exact rather than approximate, and those are the ones that would
        catch an indexing or ordering bug anywhere: the 2W padding samples must be
        bit-identical to the input; two runs on one input must be bit-identical (the
        cheapest race detector there is, and it only works because the reduction is
        deterministic -- no atomics, see the "chunk invariance" comment in
        Detrender2d.cu); the same output sample computed at two chunk lengths must be
        bit-identical; and poisoning the masked samples with nan/inf must not change the
        output at all.  None of those get stronger by varying the knot vector, which is
        why one is enough here.  (Two of the eight, the residual comparison and the
        flat-baseline bound, ARE numerical -- they ride along because the run is already
        set up.)
      - PART 1b pins the degenerate mask endpoints (all-masked, all-valid, one live
        channel, dead zones, ...) that the sweep leaves to chance -- and two of which it
        actively discards, since a draw with no live zone is skipped.  Checked for
        non-finite output as well as agreement.
      - PART 2 runs only the agreement check, but sweeps 8 knot vectors x 6 mask kinds.
        Its job is to reach small r_min, which is where the GPU and the reference can
        actually disagree: conditioning turns on h_max, and a fixed benign vector never
        leaves the well-conditioned regime.  Sized to be comparable to
        test_dtype_agreement's 120 draws, allowing for the GPU oracle being several times
        more expensive per draw (T >= 32 against that test's ntime = 10).

    TWO DEGENERATE CORNERS ARE PINNED rather than left to the sweep.  W = 0 means there
    is no padding at all (nbuf == T), the parity fold has no k >= 1 term, and the rank
    test degenerates to the 1-d dead-zone test.  n_phi = 0 means every interior knot has
    multiplicity 1 and is therefore a zone boundary, so every zone is a single
    coefficient, D_1 is identically zero, and the regulator disappears -- the
    unregularized limit, reached legitimately.  Their intersection,
    (n_phi, n, W) = (0, 0, 0), is a 1x1 solve per zone with no padding and no regulator,
    and is where any surviving assumption about buffer geometry or matrix shape would
    break.

    THE TOLERANCE IS SCALED BY eps_mach/r_min, NOT A CONSTANT.  The mask generator
    routinely leaves a zone sitting just above eps; it survives the cut and then carries
    an error of order eps_mach/(4 r_min) BY DESIGN (see solve.py), so a flat threshold
    either fails on a normal draw or is too loose to test anything.

    Skips itself with a message if no kernel is compiled, or if cupy is missing.
    """
    rng = _default_rng(rng)

    try:
        import cupy as cp
    except ImportError:
        if verbose:
            print('    test_gpu_kernel: cupy not available, skipped')
        return

    from ..kernels import Detrender2d      # local import: this package is otherwise numpy-only

    cfgs = Detrender2d.configs()
    if not cfgs:
        if verbose:
            print('    test_gpu_kernel: no kernel compiled, skipped')
        return

    n_phis = sorted(cfgs)

    # T is a RUNTIME argument, not a compiled configuration, so each half picks the chunk
    # length that suits it.  That matters most for the sweep, whose cost is dominated by
    # the numpy oracle: at T = 64 the reference is ~8x cheaper than at 512, which is what
    # lets the sweep reach nfreq = 30000.
    # Tsweep is the smallest legal chunk (T must be a positive multiple of 32).  The
    # sweep's cost is dominated by the numpy oracle, which scales with nfreq*(T+2W), so
    # the smallest T is what buys the draw count below.
    T, Tbig, Tsweep = 512, 2048, 32
    epsm = np.finfo(np.float32).eps
    band = RMIN_ABS_TOL_EPS * epsm

    # (n_phi, n, W) for part 1: every compiled n_phi at the production (n, W), plus the
    # two degenerate corners described above.
    FIXED_CASES = ([(p, 2, 4) for p in n_phis]
                   + [(n_phis[-1], 0, 0), (n_phis[0], 0, 0)])

    def run(det, dd, mm):
        gd, gm = cp.asarray(dd.copy()), cp.asarray(mm.astype(np.uint8))
        det.launch(gd, gm)
        cp.cuda.get_current_stream().synchronize()
        return cp.asnumpy(gd), cp.asnumpy(gm)

    def check_expansion(kvx, m_in, m_gpu, p64, eps, Wx, Tx, tag):
        """
        The kernel emits no r_min, so expansion is checked against the reference's.
        Zone-samples whose r_min sits within a band of eps are exempt: the
        float32-vs-float64 disagreement in r_min reaches a few hundred machine epsilons
        (see RMIN_ABS_TOL_EPS), which is comparable to eps itself.
        """
        expect_bad = (p64 < eps)
        ambiguous = np.abs(p64 - eps) < band
        for z, (lo, hi) in enumerate(zone_channel_ranges(kvx)):
            want = (m_in[:, lo:hi, Wx:Wx+Tx] != 0) & ~expect_bad[:, None, :, z]
            got = m_gpu[:, lo:hi, :]
            free = np.broadcast_to(ambiguous[:, None, :, z], got.shape)
            assert np.array_equal(got[~free], want[~free]), \
                f'{tag}: mask expansion differs in zone {z}'

    # ------------------------------------------ part 1: every check, at one knot vector

    report = []
    for (n_phi, n, W) in FIXED_CASES:
        tag = f'(n_phi,n,W)=({n_phi},{n},{W})'
        nbuf = T + 2*W

        kv = msk.zoned_knots(n_phi, nfreq, 4, 3)
        knots = [int(x) for x in kv.knots]
        det = Detrender2d(nfreq=nfreq, knots=knots, M=M_ax, n_phi=n_phi, n=n, W=W, T=T)

        m_in = msk.random_mask_2d((M_ax, nfreq, nbuf), kv, rng, det.eta, n=n, W=W)
        base, _ = _smooth_baseline(kv, rng, M_ax, nbuf)
        d32 = (base + 0.05*rng.standard_normal(base.shape)).astype(np.float32)
        d64 = d32.astype(np.float64)               # bit-identical inputs

        ref = SplineDetrender(kv, n=n, W=W, dtype=np.float64, eta=det.eta, eps=det.eps)
        r64, m64, p64 = ref.detrend_chunk(d64, m_in)

        out_d, out_m = run(det, d32, m_in)

        # (a) the padding is read but never written.  At W = 0 both ranges are empty,
        # which is the correct behaviour rather than a skipped check: there is no padding.
        for lo, hi, what in ((0, W, 'prepadding'), (W+T, nbuf, 'postpadding')):
            assert np.array_equal(out_d[:, :, lo:hi], d32[:, :, lo:hi]), \
                f'{tag}: {what} data modified'
            assert np.array_equal(out_m[:, :, lo:hi], m_in[:, :, lo:hi].astype(np.uint8)), \
                f'{tag}: {what} mask modified'

        r_gpu, m_gpu = out_d[:, :, W:W+T], out_m[:, :, W:W+T].astype(bool)

        # (b) residuals, on the channels both runs keep.
        inter = m_gpu & m64
        kept = (p64 > 0)
        rmin = float(p64[kept].min()) if kept.any() else 1.0
        tol = epsm/rmin*max(1.0, float(np.abs(d32).max()))
        resid = float(np.abs(r_gpu.astype(np.float64) - r64)[inter].max()) if inter.any() else 0.0
        assert resid < 50*tol, (tag, resid, tol)

        # (c) mask expansion.
        check_expansion(kv, m_in, m_gpu, p64, det.eps, W, T, tag)

        # (d) run-to-run bit identity.
        again = run(det, d32, m_in)
        assert np.array_equal(out_d, again[0]) and np.array_equal(out_m, again[1]), \
            f'{tag}: two runs on the same input are not bit-identical'

        # (e) bit-identical across chunk lengths, given consistent padding.  Needs no
        # reference, so it runs on the largest T: the numpy oracle is what makes the
        # checks above expensive, not the kernel.
        nchunk = 0
        nb2 = Tbig + 2*W
        m2 = msk.random_mask_2d((M_ax, nfreq, nb2), kv, rng, det.eta, n=n, W=W)
        b2, _ = _smooth_baseline(kv, rng, M_ax, nb2)
        d2 = (b2 + 0.05*rng.standard_normal(b2.shape)).astype(np.float32)
        big = Detrender2d(nfreq=nfreq, knots=knots, M=M_ax, n_phi=n_phi, n=n, W=W, T=Tbig)
        # Bit-exactness across chunk lengths holds only if the two instances derive the
        # same channels_per_range, since that is part of the frequency summation order.
        # The derivation depends on T, but at nfreq=1024 both T's land on the CPR_MIN
        # clamp, so they agree.  Asserted rather than assumed: at a large enough nfreq
        # they would diverge, and the comparisons below would then have to be
        # tolerance-based (there is no numpy reference here to set a tolerance from).
        assert big.channels_per_range == det.channels_per_range, \
            (f'{tag}: T={T} and T={Tbig} derive different channels_per_range '
             f'({det.channels_per_range} vs {big.channels_per_range}), so chunk '
             f'invariance is no longer bit-exact at nfreq={nfreq}')
        big_d, big_m = run(big, d2, m2)
        for c in range(Tbig // T):
            sub_d = np.ascontiguousarray(d2[:, :, T*c:T*c + T + 2*W])
            sub_m = np.ascontiguousarray(m2[:, :, T*c:T*c + T + 2*W])
            o_d, o_m = run(det, sub_d, sub_m)
            assert np.array_equal(o_d[:, :, W:W+T], big_d[:, :, W+T*c:W+T*(c+1)]), \
                f'{tag}: chunk invariance T={T} vs T={Tbig} differs in data (chunk {c})'
            assert np.array_equal(o_m[:, :, W:W+T], big_m[:, :, W+T*c:W+T*(c+1)]), \
                f'{tag}: chunk invariance T={T} vs T={Tbig} differs in mask (chunk {c})'
            nchunk += 1

        # (f) masked samples are never read.  Only the OUTPUT region is compared: the
        # padding is not written, so it still holds the poison.
        for poison in (np.nan, np.inf, -1.0e30):
            dp = d32.copy()
            dp[m_in == 0] = poison
            p_d, p_m = run(det, dp, m_in)
            assert np.array_equal(p_d[:, :, W:W+T], out_d[:, :, W:W+T]), \
                f'{tag}: output changed when masked samples were poisoned with {poison}'
            assert np.array_equal(p_m[:, :, W:W+T], out_m[:, :, W:W+T]), \
                f'{tag}: mask changed when masked samples were poisoned with {poison}'

        # (g) exact reproduction: constant in frequency within a zone, degree-n in time.
        # An absolute-zero assertion, and the reason it is worth keeping alongside (b):
        # (b)'s tolerance is set by the WORST r_min in the draw, so an O(eta) error in the
        # regulator hides inside it, while this one is immune.  At n = 0 it degenerates to
        # "a constant is removed exactly", which is the 1-d statement.
        tt = (np.arange(nbuf, dtype=np.float64) - (nbuf-1)/2.0) / max(W, 1)
        coef = rng.standard_normal(n+1)
        level = sum(coef[q]*tt**q for q in range(n+1))
        level = level / np.abs(level).max()
        dflat = np.broadcast_to(level[None, None, :], (M_ax, nfreq, nbuf)).astype(np.float32)
        f_d, f_m = run(det, dflat, m_in)
        pf = ref.detrend_chunk(dflat.astype(np.float64), m_in)[2]
        rminf = float(pf[pf > 0].min()) if (pf > 0).any() else 1.0
        tolf = epsm/rminf*max(1.0, float(np.abs(dflat).max()))
        flat = float(np.abs(f_d[:, :, W:W+T])[f_m[:, :, W:W+T].astype(bool)].max())
        assert flat < 200*tolf, (tag, flat, tolf)

        # (h) the rank test: a zone carrying data at fewer than n+1 distinct window offsets
        # must be flagged, whatever its channel count.  Constructed rather than drawn -- the
        # generator produces it often, but not reliably enough to rely on here, and a
        # silently non-firing rank test yields garbage coefficients.  At n = 0 the condition
        # reduces to "the zone holds no unmasked channel", i.e. the 1-d dead-zone test.
        m_rank = msk.random_mask_2d((M_ax, nfreq, nbuf), kv, rng, det.eta, n=n, W=W,
                                    time_kind='n_live_offsets', perturb=False)
        rr_d, rr_m = run(det, d32, m_rank)
        p_rank = ref.detrend_chunk(d64, m_rank)[2]
        dead = (p_rank == 0.0)
        nrank = int(dead.sum())
        if dead.any():
            for z, (lo, hi) in enumerate(zone_channel_ranges(kv)):
                sel = dead[:, None, :, z] & np.ones((1, hi-lo, 1), dtype=bool)
                assert not rr_m[:, lo:hi, W:W+T][sel].any(), \
                    f'{tag}: rank-deficient zone {z} was not flagged'

        # NOT CHECKED HERE: that channels_per_range is a pure tuning knob, i.e. that the
        # frequency-summation grouping changes the answer only by roundoff and never
        # changes the mask.  Constructing two instances that differ only in that value is
        # no longer possible -- it is derived from (nfreq, knots, T) -- so nothing tests
        # it.  If it ever starts affecting the answer materially, no test will notice.

        report.append((n_phi, n, W, resid/tol, flat/tolf, nchunk, nrank))

    # ------------------------------------------ part 1b: pinned mask corner cases
    #
    # random_mask_2d draws its base type per zone from a weighted mixture, which covers
    # the extremal FAMILIES well but leaves the degenerate ENDPOINTS to chance -- and two
    # of them the sweep actively discards, since a draw with no live zone is skipped.
    # These are constructed instead, and checked for non-finite output as well as
    # agreement: every one of them drives a guard (a zero pivot, an empty zone, a
    # single-point exact fit), and the failure mode of a guard is Inf or NaN escaping,
    # not a wrong number.
    cnfreq, cT, cW, cn = 2048, 128, 4, 2
    cnbuf = cT + 2*cW
    ckv = msk.zoned_knots(n_phis[-1], cnfreq, 4, 3)
    cdet = Detrender2d(nfreq=cnfreq, knots=[int(x) for x in ckv.knots], M=1,
                       n_phi=n_phis[-1], n=cn, W=cW, T=cT)
    cref = SplineDetrender(ckv, n=cn, W=cW, dtype=np.float64, eta=cdet.eta, eps=cdet.eps)
    cbase, _ = _smooth_baseline(ckv, rng, 1, cnbuf)
    cd = (cbase + 0.05*rng.standard_normal(cbase.shape)).astype(np.float32)

    ones = np.ones((1, cnfreq, cnbuf), dtype=bool)
    corner = {}
    corner['all masked'] = np.zeros((1, cnfreq, cnbuf), dtype=bool)
    corner['all valid'] = ones.copy()
    mm = np.zeros((1, cnfreq, cnbuf), dtype=bool); mm[:, 7, :] = True
    corner['one live channel'] = mm                      # must SURVIVE: fit is exact there
    mm = ones.copy(); mm[:, 0, :] = False; mm[:, -1, :] = False
    corner['band edges'] = mm                            # the clamped ends of the basis
    mm = ones.copy(); mm[:, :, cW] = False
    corner['one dead time sample'] = mm
    mm = ones.copy(); mm[:, :, :cW] = False; mm[:, :, cW+cT:] = False
    corner['padding only'] = mm                          # the halo is masked, the output is not
    mm = np.zeros((1, cnfreq, cnbuf), dtype=bool); mm[:, ::2, :] = True
    corner['alternating channels'] = mm
    mm = ones.copy(); mm[:, cnfreq//4:3*cnfreq//4, :] = False
    corner['whole zones dead'] = mm

    for cname, cmask in corner.items():
        cr64, cm64, cp64 = cref.detrend_chunk(cd.astype(np.float64), cmask)
        co_d, co_m = run(cdet, cd, cmask)
        cout = co_d[:, :, cW:cW+cT]
        assert np.isfinite(cout).all(), f'corner case "{cname}": non-finite output'
        cmg = co_m[:, :, cW:cW+cT].astype(bool)
        assert np.array_equal(cmg, cm64), \
            f'corner case "{cname}": mask differs from the reference'
        clive = cp64[cp64 > 0]
        cii = cmg & cm64
        if cii.any() and clive.size:
            ce = float(np.abs(cout.astype(np.float64) - cr64)[cii].max())
            ctol = epsm/float(clive.min())*max(1.0, float(np.abs(cd).max()))
            assert ce < 50*ctol, f'corner case "{cname}": resid {ce:.3e} vs tol {ctol:.3e}'

    # ------------------------------------------ part 2: agreement, over many knot vectors
    #
    # Part 1 runs on ONE fixed, benign knot vector, which is right for the checks that
    # dominate it -- they are about indexing and ordering, and a second knot vector would
    # not make them stronger.  It is wrong for the agreement comparison:
    # conditioning turns on h_max, the widest knot interval, and the fixed vector has
    # h_max = nfreq/16, so the eps / r_min regime the whole design exists for is never
    # touched and the residual ratio sits orders of magnitude inside its tolerance.
    #
    # So sweep the knot vector, nfreq, W and n together.  The knot profiles are the ones
    # test_conditioning pins, for the same reason: 'no_interior' and 'one_wide' put h_max
    # at or near nfreq and are not reachable by drawing cut points at random.  Mask kinds
    # are cycled explicitly rather than drawn, because a random draw under-samples the
    # extremes -- masks.py measures a factor of 23 between Bernoulli and adversarial masks.
    #
    # Running at T = Tsweep is what makes nfreq = 30000 affordable: the numpy oracle costs
    # ~0.9 s per call at (30000, T=512) and an eighth of that at T = 64.  30000 with
    # 'no_interior' reaches the worst margin over eps anywhere in the parameter study
    # (3.0x), so the GPU sees it rather than only the reference.
    #
    # (n,W) = (0,0) appears here too, at a wide knot interval, so the degenerate corner is
    # exercised against small r_min and not only in part 1's well-conditioned geometry.
    # Two pinned extremes plus six random draws, each a (nfreq, knot profile, W, n, n_phi)
    # tuple.  nfreq is drawn log-uniformly so most draws are cheap and a few are wide; the
    # pinned pair guarantees the wide-h_max regime is visited every call regardless.
    sweep = [(30000, 'no_interior', 4, 2, n_phis[-1]),
             (int(rng.integers(8000, 20001)), 'one_wide', 1, 2, n_phis[-1])]
    for _ in range(6):
        ns = int(rng.integers(0, 3))
        # 2W+1 >= n+1 is the algebraic minimum, so W = 0 is legal only at n = 0.
        Wok = [w for w in (0, 1, 2, 4, 8) if 2*w + 1 >= ns + 1]
        sweep.append((int(round(np.exp(rng.uniform(np.log(128), np.log(30000))))),
                      None if rng.random() < 0.6 else 'no_interior',
                      int(rng.choice(Wok)),
                      ns,
                      int(rng.choice(n_phis))))

    # Cycled explicitly rather than drawn: a random draw under-samples the extremes, and
    # masks.py measures a factor of 23 between Bernoulli and adversarial masks.  The first
    # three are the conditioning extremes; the next two drive mask expansion; None adds
    # breadth from the generator's own weighted mixture.
    kinds = ('adversarial', 'adversarial_singular', 'narrowband',
             'n_live_offsets', 'dead_zone', None)

    worst_ratio, worst_rmin, ndraw, nflag = 0.0, np.inf, 0, 0
    nsecond = [0]         # draws that tripped the screen and needed the float32 reference
    for nf, kind, Ws, ns, ps in sweep:
        kvs = msk.random_knots(rng, n_phi=ps, nfreq=nf, kind=kind)
        dets = Detrender2d(nfreq=kvs.nfreq, knots=[int(x) for x in kvs.knots], M=1,
                           n_phi=ps, n=ns, W=Ws, T=Tsweep)
        refs = SplineDetrender(kvs, n=ns, W=Ws, dtype=np.float64, eta=dets.eta, eps=dets.eps)
        nbs = Tsweep + 2*Ws

        for kind_t in kinds:
            tag = (f'nfreq={kvs.nfreq}, (n_phi,n,W)=({ps},{ns},{Ws}), '
                   f'kind={kind}/{kind_t}')
            ms = msk.random_mask_2d((1, kvs.nfreq, nbs), kvs, rng, dets.eta,
                                    n=ns, W=Ws, time_kind=kind_t)
            bs, _ = _smooth_baseline(kvs, rng, 1, nbs)
            ds = (bs + 0.05*rng.standard_normal(bs.shape)).astype(np.float32)
            rs64, ms64, ps64 = refs.detrend_chunk(ds.astype(np.float64), ms)
            os_d, os_m = run(dets, ds, ms)
            rs_g = os_d[:, :, Ws:Ws+Tsweep]
            ms_g = os_m[:, :, Ws:Ws+Tsweep].astype(bool)

            live = ps64[ps64 > 0]
            if not live.size:
                continue
            rmin_s = float(live.min())
            worst_rmin = min(worst_rmin, rmin_s)
            tol_s = epsm/rmin_s*max(1.0, float(np.abs(ds).max()))

            ii = ms_g & ms64
            if ii.any():
                es = float(np.abs(rs_g.astype(np.float64) - rs64)[ii].max())
                worst_ratio = max(worst_ratio, es/tol_s)

                # TWO-STAGE, and the second stage is the one that means something.
                #
                # eps_mach/r_min is a rule of thumb rather than a bound -- solve.py says
                # so outright: log-log slope 0.68 against 1, correlation 0.62.  Measured
                # over 28 seeds of this sweep the ratio runs median 3.1, p90 4.3, but one
                # draw in 28 reaches 79, at a configuration that is not even
                # ill-conditioned.  A constant large enough never to trip on those is
                # large enough to be meaningless.
                #
                # So when the cheap screen trips, ask the question the test actually cares
                # about: is the GPU worse than the numpy reference AT THE SAME PRECISION?
                # A hard configuration makes both float32 paths equally bad, which is not
                # a GPU bug; only the GPU being an order of magnitude worse is.  Measured
                # over the configurations that trip the screen, the ratio of the two
                # float32 errors runs 0.2 to 3.0, so 10x is a wide margin.  The second
                # reference costs a float32 detrend_chunk, paid on ~4% of draws.
                if es >= 50*tol_s:
                    r32, m32ref, _ = SplineDetrender(
                        kvs, n=ns, W=Ws, dtype=np.float32, eta=dets.eta,
                        eps=dets.eps).detrend_chunk(ds, ms)
                    jj = m32ref[:, :, :] & ms64
                    enp = float(np.abs(r32.astype(np.float64) - rs64)[jj].max()) if jj.any() else 0.0
                    nsecond[0] += 1
                    assert es < 10*enp, \
                        f'{tag}: GPU resid {es:.3e} vs numpy-float32 {enp:.3e} ' \
                        f'(r_min {rmin_s:.2e}, eps_mach/r_min bound {tol_s:.3e})'

            nflag += int((ps64 < dets.eps).sum())
            check_expansion(kvs, ms, ms_g, ps64, dets.eps, Ws, Tsweep, tag)
            ndraw += 1

    if verbose:
        for (pp, n, W, rr, ff, nc, nk) in report:
            print(f'    test_gpu_kernel (n_phi,n,W)=({pp},{n},{W}): pass  '
                  f'[resid {rr:.3f} x eps_mach/r_min, flat baseline {ff:.1e} x the same '
                  f'bound, {nc} chunk-invariance comparisons bit-identical, '
                  f'{nk} rank-deficient zone-samples]')
        print(f'      sweep: {ndraw} draws over (nfreq, knots, n_phi, n, W, mask kind), '
              f'worst resid {worst_ratio:.3f} x eps_mach/r_min, '
              f'worst r_min {worst_rmin:.2e}, {nflag} zone-samples flagged, '
              f'{nsecond[0]} checked against numpy-float32')


# ----------------------------------------------------------------

def test_params_yaml(rng=None, verbose=True):
    """
    Detrender2dParams round-trips through yaml: from_yaml_string(to_yaml_string(p)) == p,
    plus one case through a real file to cover from_yaml().

    Skips itself if the extension is not built.  The knot geometries are drawn rather
    than pinned, since the knot vector is the only field whose yaml representation is
    non-trivial, and its validity rules (end multiplicity, monotonicity, span) are
    exactly what a serialization bug would violate.
    """
    rng = _default_rng(rng)

    try:
        from ..kernels import Detrender2dParams
    except ImportError:
        if verbose:
            print('    test_params_yaml: extension not built, skipped')
        return

    def _check_equal(p, q):
        for field in ('nfreq', 'M', 'n_phi', 'n', 'W', 'T', 'eta', 'eps'):
            assert getattr(p, field) == getattr(q, field), \
                (field, getattr(p, field), getattr(q, field))
        assert list(p.knots) == list(q.knots), (list(p.knots), list(q.knots))

    ncase = 0

    for n_phi in (0, 1, 2, 3):
        for (nzone, kint) in ((1, 0), (1, 3), (4, 2)):
            nfreq = 64 * int(rng.integers(1, 8))
            kv = msk.zoned_knots(n_phi, nfreq, nzone, kint)
            knots = [int(x) for x in kv.knots]
            W = int(rng.integers(0, 9))
            n = int(rng.integers(0, min(2, 2*W) + 1))

            # explicit_tuning=False exercises the optional eta/eps defaults; the verbose
            # form is round-tripped too, since its comments must not confuse the reader.
            for explicit_tuning in (False, True):
                kw = {}
                if explicit_tuning:
                    kw = dict(eta=float(rng.uniform(1e-4, 1e-2)),
                              eps=float(rng.uniform(1e-6, 1e-4)))
                p = Detrender2dParams(nfreq=nfreq, knots=knots, M=int(rng.integers(1, 4)),
                                      n_phi=n_phi, n=n, W=W, T=32*int(rng.integers(1, 8)), **kw)

                _check_equal(p, Detrender2dParams.from_yaml_string(
                    p.to_yaml_string(verbose=explicit_tuning)))
                ncase += 1

    # The file path is a thin wrapper over the string path, so one case covers it.
    import os
    import tempfile

    with tempfile.NamedTemporaryFile('w', suffix='.yml', delete=False) as fh:
        fh.write(p.to_yaml_string())
        path = fh.name
    try:
        _check_equal(p, Detrender2dParams.from_yaml(path))
    finally:
        os.unlink(path)

    # Retired key: yaml written against the old interface must be rejected, not silently
    # reinterpreted (channels_per_range used to be a constructor argument). An unknown key
    # is an error too, via YamlFile.check_for_invalid_keys().
    for (extra, want) in ((f'\nchannels_per_range: 256\n', 'channels_per_range'),
                          (f'\nnot_a_real_key: 7\n', '')):
        try:
            Detrender2dParams.from_yaml_string(p.to_yaml_string() + extra)
            raise AssertionError(f'from_yaml_string accepted {extra.strip()!r}')
        except RuntimeError as e:
            assert want in str(e), (want, str(e))

    if verbose:
        print(f'    test_params_yaml: pass  [{ncase} round trips via from_yaml_string, plus '
              f'the file path, retired-key and unknown-key rejection]')


def run_all(verbose=True, rng=None, heavy=False, n_phi=None, gpu=False):
    """
    All tests share one generator, so printing its entropy makes the whole run
    reproducible: pass np.random.default_rng(<entropy>) back in as 'rng'.

    The spline degree is drawn from {0,1,2,3} per call rather than fixed, so a
    multi-iteration run covers all four; pass n_phi explicitly to pin it.  This
    matters more than it looks: an assembly bug confined to n_phi = 0 (D_1 has
    half-bandwidth 1 regardless of n_phi, so its off-diagonal band outlives the
    data bands) is invisible at n_phi = 2.

    'heavy' turns on some extra large-nfreq conditioning configurations.  It is no
    longer where the eps margin is probed -- the configurations that reach 3x eps
    are pinned and run by default (see test_conditioning) -- so leaving it off does
    not weaken the suite.
    """
    global N_PHI
    rng = _default_rng(rng)
    if n_phi is None:
        n_phi = int(rng.integers(0, 4))
    N_PHI = n_phi
    ent = rng.bit_generator.seed_seq.entropy
    print(f'  detrending_spline tests (n_phi={N_PHI}, eta={ETA_DEFAULT:g}, '
          f'eps={EPS_FLOAT32:g}, rng entropy {ent})')
    test_knots(rng, verbose=verbose)
    test_basis(rng, verbose=verbose)
    test_regulator(rng, verbose=verbose)
    test_reduce(rng, verbose=verbose)
    test_chunk_invariance(rng, verbose=verbose)
    test_solve(rng, verbose=verbose)
    test_flat_baseline_exact(rng, verbose=verbose)
    test_shrinkage_bias_bounded(rng, verbose=verbose)
    test_masked_data_unused(rng, verbose=verbose)
    test_spectator_axes(rng, verbose=verbose)
    test_conditioning(rng, verbose=verbose, heavy=heavy)
    test_zone_expansion(rng, verbose=verbose)
    test_reference_agreement(rng, verbose=verbose)
    test_dtype_agreement(rng, verbose=verbose)
    test_time_basis(rng, verbose=verbose)
    test_bandwidth(rng, verbose=verbose)
    test_2d_reference_agreement(rng, verbose=verbose)
    test_2d_flat_baseline_exact(rng, verbose=verbose)
    test_n1_degeneracy(rng, verbose=verbose)
    test_time_rank_deficiency(rng, verbose=verbose)
    test_2d_chunk_invariance(rng, verbose=verbose)
    test_2d_conditioning(rng, verbose=verbose)
    test_2d_dtype_agreement(rng, verbose=verbose)
    test_production_geometry(rng, verbose=verbose)
    test_params_yaml(rng, verbose=verbose)
    if gpu:
        test_gpu_kernel(rng, verbose=verbose)
    print(f'  detrending_spline tests passed   '
          f'[cumulative worst r_min: {_worst_rmin[0]:.3e}, '
          f'{_worst_rmin[0]/EPS_FLOAT32:.1f} x eps]')
