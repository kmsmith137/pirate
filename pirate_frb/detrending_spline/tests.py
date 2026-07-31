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

# Smallest r_min observed across a run of test_conditioning(), so that run_all()
# can print the measured margin over eps.
_worst_rmin = [np.inf]


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
        kv = msk.random_knots(rng, n_phi=N_PHI, nfreq=int(rng.integers(64, 900)))
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

    'heavy' adds the large-F, wide-knot-interval configurations from the parameter
    study.  They are slow and are not run by default.
    """
    eta, eps = ETA_DEFAULT, EPS_FLOAT32
    worst, worst_cfg = np.inf, None

    cfgs = [(int(rng.integers(200, 2000)), None) for _ in range(14)]
    if heavy:
        cfgs += [(30000, 'uniform'), (30000, 'one_wide'), (15000, 'one_wide')]

    for nfreq, kind in cfgs:
        kv = msk.random_knots(rng, n_phi=N_PHI, nfreq=nfreq, kind=kind)
        det = SplineDetrender(kv, dtype=np.float64, eta=eta, eps=eps)
        table = det.table
        D1 = d1_banded(kv)
        for _ in range(6 if nfreq < 5000 else 2):
            w = msk.random_mask_1d(kv, rng, eta,
                                   kind='adversarial' if rng.random() < 0.6 else None)
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
    """
    n_flip, n_zone, worst_rel, worst_resid = 0, 0, 0.0, 0.0

    for _ in range(12):
        kv = msk.random_knots(rng, n_phi=N_PHI, nfreq=int(rng.integers(128, 1500)))
        M_ax, ntime = 1, 10
        mask = msk.random_mask((M_ax, kv.nfreq, ntime), kv, rng, ETA_DEFAULT)
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

        # (e) r_min agreement away from the band.
        ok = (~band) & (p64 > 0)
        if ok.any():
            rel = np.abs(p32.astype(np.float64) - p64)[ok] / p64[ok]
            worst_rel = max(worst_rel, float(rel.max()))

    assert n_flip <= 1e-3 * max(n_zone, 1), \
        f'{n_flip}/{n_zone} zone decisions flipped outside the threshold band'
    assert worst_resid < 1e-3, f'float32 vs float64 residual {worst_resid:.3e}'
    assert worst_rel < 1e-2, f'float32 vs float64 r_min relative error {worst_rel:.3e}'

    if verbose:
        print(f'    test_dtype_agreement: pass  [resid {worst_resid:.2e}, '
              f'r_min rel {worst_rel:.2e}, flips {n_flip}/{n_zone}]')



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
                                   time_kind='independent')
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
            m = msk.random_mask_2d((1, kv.nfreq, T + 2*W), kv, rng, det.eta)
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
            m = msk.random_mask_2d((1, kv.nfreq, nbuf), kv, rng, det.eta)
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

        mc = msk.random_mask_2d((1, kv.nfreq, T+2*W), kv, rng, d0.eta,
                                time_kind='persistent')
        r0 = d0.detrend_chunk(d, mc)[0]
        r1 = d1.detrend_chunk(d, mc)[0]
        same = max(same, float(np.abs(r0-r1).max()))
        assert np.abs(r0-r1).max() < 1e-9*max(1.0, np.abs(d).max()), 'n=1 != n=0'

        mv = msk.random_mask_2d((1, kv.nfreq, T+2*W), kv, rng, d0.eta,
                                time_kind='independent')
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
        m = msk.random_mask_2d((2, kv.nfreq, nbuf), kv, rng, det.eta)
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
                                   time_kind='persistent')
            p2 = det2.detrend_chunk(d, m)[2]
            p1 = det1.detrend_chunk(d[:, :, W:W+1], m[:, :, W:W+1])[2]
            live = p1 > 0
            if not live.any():
                continue
            assert np.abs(p2[live] - p1[live]).max() < 1e-10, \
                f'window-constant r_min not preserved at n={n}, W={W}'
            # Time-varying: no factorization, so only require it stays sane.
            mv = msk.random_mask_2d((1, kv.nfreq, nbuf), kv, rng, det2.eta,
                                    time_kind='independent')
            pv = det2.detrend_chunk(d, mv)[2]
            if (pv > 0).any():
                worst_abs = min(worst_abs, float(pv[pv > 0].min()))
    assert worst_abs > 1e-6, worst_abs
    if verbose:
        print(f'    test_2d_conditioning: pass  [window-constant r_min preserved '
              f'exactly; worst time-varying r_min {worst_abs:.3e}]')


def test_2d_dtype_agreement(rng, verbose=True):
    """float32 at EPS_FLOAT32 vs float64 at EPS_FLOAT64, swept over (n, W)."""
    worst = 0.0
    for _ in range(4):
        kv = msk.random_knots(rng, n_phi=N_PHI, nfreq=int(rng.integers(128, 800)))
        for n, W in NW_CASES:
            T = 5
            nbuf = T + 2*W
            m = msk.random_mask_2d((1, kv.nfreq, nbuf), kv, rng, ETA_DEFAULT)
            base, _ = _smooth_baseline(kv, rng, 1, nbuf)
            d = base + 0.05*rng.standard_normal(base.shape)
            a32 = SplineDetrender(kv, n=n, W=W, dtype=np.float32)
            a64 = SplineDetrender(kv, n=n, W=W, dtype=np.float64)
            r32, m32, _ = a32.detrend_chunk(d.astype(np.float32), m)
            r64, m64, _ = a64.detrend_chunk(d, m)
            inter = m32 & m64
            if inter.any():
                e = float(np.abs(r32.astype(np.float64) - r64)[inter].max()
                          / max(1.0, np.abs(d).max()))
                worst = max(worst, e)
    assert worst < 1e-3, worst
    if verbose:
        print(f'    test_2d_dtype_agreement: pass  [resid {worst:.2e}]')


# ----------------------------------------------------------------

def run_all(verbose=True, rng=None, heavy=False, n_phi=None):
    """
    All tests share one generator, so printing its entropy makes the whole run
    reproducible: pass np.random.default_rng(<entropy>) back in as 'rng'.

    The spline degree is drawn from {0,1,2,3} per call rather than fixed, so a
    multi-iteration run covers all four; pass n_phi explicitly to pin it.  This
    matters more than it looks: an assembly bug confined to n_phi = 0 (D_1 has
    half-bandwidth 1 regardless of n_phi, so its off-diagonal band outlives the
    data bands) is invisible at n_phi = 2.

    'heavy' turns on the large-nfreq conditioning configurations, which are slow;
    they are the ones that probe the eps margin most closely.
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
    print(f'  detrending_spline tests passed   '
          f'[cumulative worst r_min: {_worst_rmin[0]:.3e}, '
          f'{_worst_rmin[0]/EPS_FLOAT32:.1f} x eps]')
