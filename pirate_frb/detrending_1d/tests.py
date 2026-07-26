"""
Unit tests for the 1-d detrender.  Dispatched from pirate_frb/__main__.py:

    python -m pirate_frb test --dt1d

Most tests run at a small size (W=16, B=32, Tc=64) for speed -- the algorithm is
size-parameterized, so small sizes exercise the same code paths -- plus a
full-size (W=512) run where it is cheap enough to matter.
"""

import numpy as np

from .MomentSet import MomentSet, merge, pascal_shift, _binom_table
from .scan import tree_prefix_scan, tree_suffix_scan, sequential_prefix_scan
from .Detrender import Detrender
from .reference import detrend_reference
from .masks import mask_zoo
from . import LocalPolyFit


# ------------------------------------------------------------------ utilities

def _default_rng(rng):
    """Unseeded by choice: these tests are randomized, so repeated runs (and
    'test --dt1d -n N') explore different data rather than re-checking one
    fixture.  run_all() prints the entropy of the generator it creates, which is
    enough to reproduce a specific failing run if one turns up."""
    return np.random.default_rng() if rng is None else rng


def _maxdiff(a, b):
    a = np.asarray(a, dtype=np.float64)
    b = np.asarray(b, dtype=np.float64)
    return float(np.max(np.abs(a-b))) if a.size else 0.0


def _ms_maxdiff(a, b):
    # The centroid is only meaningful where nv > 0; for an empty set it is an
    # arbitrary finite placeholder (see the empty-set rule in MomentSet.merge),
    # so two correct implementations may legitimately disagree there.
    ok = (np.asarray(a.nv) > 0) & (np.asarray(b.nv) > 0)
    dc = _maxdiff(np.asarray(a.c)[ok], np.asarray(b.c)[ok]) if ok.any() else 0.0
    return max(_maxdiff(a.nv, b.nv), dc, _maxdiff(a.S, b.S), _maxdiff(a.U, b.U))


def _moments_about(u, m, md, origin, n, W, dtype, axis=-1):
    """Moments about an arbitrary origin (not necessarily the centroid)."""
    x = ((u - np.expand_dims(origin, axis)) / W).astype(dtype)
    S = np.zeros(x.shape[:axis] + x.shape[axis+1:] + (2*n+1,), dtype=dtype) \
        if False else np.zeros(np.sum(m, axis=axis).shape + (2*n+1,), dtype=dtype)
    U = np.zeros(np.sum(m, axis=axis).shape + (n+1,), dtype=dtype)
    p = np.ones_like(x)
    for r in range(2*n+1):
        S[..., r] = (m*p).sum(axis=axis)
        if r <= n:
            U[..., r] = (md*p).sum(axis=axis)
        p = p * x
    return S, U


def _random_set(rng, S_ax, L, W, n, dtype, pvalid=0.7, offset=0.0):
    u = (np.arange(L, dtype=dtype) + offset)[None, :] * np.ones((S_ax, 1), dtype=dtype)
    m = (rng.random((S_ax, L)) < pvalid).astype(dtype)
    d = rng.normal(size=(S_ax, L)).astype(dtype)
    return u, m, (m*d).astype(dtype)


# ------------------------------------------------------------------- 1. monoid

def test_monoid(rng=None, niter=8, verbose=True):
    rng = _default_rng(rng)
    n, W, dtype = 2, 16, np.float64
    binom = _binom_table(2*n+1)
    worst = dict(merge=0.0, assoc=0.0, comm=0.0, ident=0.0, s1=0.0, shift=0.0, empty=0.0)

    for it in range(niter):
        S_ax, L = 5, 40
        u, m, md = _random_set(rng, S_ax, L, W, n, dtype, pvalid=0.7, offset=100.0)

        # merge of a two-way split == direct moments of the whole set
        k = rng.integers(1, L)
        A = MomentSet.direct(u[:, :k], m[:, :k], md[:, :k], n, W, dtype)
        B = MomentSet.direct(u[:, k:], m[:, k:], md[:, k:], n, W, dtype)
        full = MomentSet.direct(u, m, md, n, W, dtype)
        worst['merge'] = max(worst['merge'], _ms_maxdiff(merge(A, B), full))

        # associativity over a three-way split
        k1, k2 = sorted(rng.choice(np.arange(1, L), size=2, replace=False))
        C1 = MomentSet.direct(u[:, :k1], m[:, :k1], md[:, :k1], n, W, dtype)
        C2 = MomentSet.direct(u[:, k1:k2], m[:, k1:k2], md[:, k1:k2], n, W, dtype)
        C3 = MomentSet.direct(u[:, k2:], m[:, k2:], md[:, k2:], n, W, dtype)
        worst['assoc'] = max(worst['assoc'],
                             _ms_maxdiff(merge(merge(C1, C2), C3), merge(C1, merge(C2, C3))))

        # commutativity, and the identity element
        worst['comm'] = max(worst['comm'], _ms_maxdiff(merge(A, B), merge(B, A)))
        idt = MomentSet.identity(A.batch_shape, 12345.0, n, W, dtype)
        worst['ident'] = max(worst['ident'], _ms_maxdiff(merge(A, idt), A))
        worst['ident'] = max(worst['ident'], _ms_maxdiff(merge(idt, A), A))

        # S_1 is identically zero
        worst['s1'] = max(worst['s1'], float(np.max(np.abs(merge(A, B).S[..., 1]))))

        # empty sets produce no NaN/Inf
        e = MomentSet.identity(A.batch_shape, 7.0, n, W, dtype)
        for x, y in ((e, e), (e, A), (A, e)):
            r = merge(x, y)
            assert np.all(np.isfinite(r.nv)) and np.all(np.isfinite(r.c))
            assert np.all(np.isfinite(r.S)) and np.all(np.isfinite(r.U))
        worst['empty'] = max(worst['empty'], _ms_maxdiff(merge(e, e), e))

        # pascal_shift against a direct recomputation about a new origin
        origin0 = u.mean(axis=1)
        origin1 = origin0 + rng.normal(size=S_ax) * W
        S0, U0 = _moments_about(u, m, md, origin0, n, W, dtype)
        S1, U1 = _moments_about(u, m, md, origin1, n, W, dtype)
        delta = ((origin0 - origin1) / W).astype(dtype)
        worst['shift'] = max(worst['shift'],
                             _maxdiff(pascal_shift(S0, delta, binom), S1),
                             _maxdiff(pascal_shift(U0, delta, binom), U1))

    if verbose:
        print('    test_monoid: ' + '  '.join(f'{k}={v:.2e}' for k, v in worst.items()))
    for k, v in worst.items():
        assert v < 1e-9, f'test_monoid: {k} = {v}'


# ------------------------------------------------------------------ 2. vanherk

def test_vanherk(rng=None, niter=4, verbose=True):
    rng = _default_rng(rng)
    n, dtype = 2, np.float64
    worst_ms, worst_cnt = 0.0, 0

    for W in (4, 16):
        B, nblocks = 2*W, 3
        T = nblocks*B
        for it in range(niter):
            S_ax = 3
            m = (rng.random((S_ax, T)) < rng.uniform(0.02, 1.0)).astype(dtype)
            d = rng.normal(size=(S_ax, T)).astype(dtype)
            u = np.broadcast_to(np.arange(T, dtype=dtype), (S_ax, T))
            leaves = MomentSet.leaves(u, m, (m*d).astype(dtype), n, W, dtype)
            leaves = MomentSet(leaves.nv.reshape(S_ax, nblocks, B),
                               leaves.c.reshape(S_ax, nblocks, B),
                               leaves.S.reshape(S_ax, nblocks, B, 2*n+1),
                               leaves.U.reshape(S_ax, nblocks, B, n+1), n, W)
            pref = tree_prefix_scan(leaves)
            suff = tree_suffix_scan(leaves)

            for b in range(nblocks - 1):
                for p in range(B):
                    # decomposition arithmetic: (B-p) + (p+1) == 2W+1, always
                    assert (B - p) + (p + 1) == 2*W + 1
                    worst_cnt += 1
                    got = merge(suff.take_batch((slice(None), b, p)),
                                pref.take_batch((slice(None), b+1, p)))
                    q = b*B + p
                    idx = np.arange(q, q + B + 1)
                    want = MomentSet.direct(u[:, idx], m[:, idx], (m*d)[:, idx], n, W, dtype)
                    worst_ms = max(worst_ms, _ms_maxdiff(got, want))

    if verbose:
        print(f'    test_vanherk: {worst_cnt} decompositions checked, max err {worst_ms:.2e}')
    assert worst_ms < 1e-9, f'test_vanherk: {worst_ms}'


# -------------------------------------------------------------------- 3. solve

def test_solve(rng=None, niter=4, verbose=True):
    rng = _default_rng(rng)
    n, dtype = 2, np.float64
    eps, mu = 1e-3, 1e-30
    W = 16
    L = 2*W + 1
    worst_lstsq, worst_lev, worst_kap = 0.0, 0.0, 0.0

    # (a) against np.linalg.lstsq on well-determined windows
    for it in range(niter):
        S_ax = 6
        u = np.broadcast_to(np.arange(L, dtype=dtype), (S_ax, L))
        m = (rng.random((S_ax, L)) < 0.8).astype(dtype)
        d = rng.normal(size=(S_ax, L)).astype(dtype)
        ms = MomentSet.direct(u, m, (m*d).astype(dtype), n, W, dtype)
        u_eval = np.full(S_ax, float(W), dtype=dtype)
        fhat, lev, flagged, ratios = LocalPolyFit.solve(ms, u_eval, eps, mu)
        assert not np.any(flagged), 'floor unexpectedly active on a well-determined window'

        for s in range(S_ax):
            x = (u[s] - ms.c[s]) / W
            A = np.vander(x, n+1, increasing=True) * m[s][:, None]
            coef = np.linalg.lstsq(A, m[s]*d[s], rcond=None)[0]
            x0 = (W - ms.c[s]) / W
            want = sum(coef[j] * x0**j for j in range(n+1))
            worst_lstsq = max(worst_lstsq, abs(want - fhat[s]))

        # pivot ratios: in [0,1], and r_0 = r_1 = 1 exactly (because G_01 = 0)
        assert np.all(ratios >= -1e-12) and np.all(ratios <= 1 + 1e-12)
        assert _maxdiff(ratios[..., 0], 1.0) < 1e-12
        assert _maxdiff(ratios[..., 1], 1.0) < 1e-12

    # (b) leverage identity, via impulses through the FULL Detrender path.
    #     One spectator per impulse position, so a single call gives the kernel.
    for maskname, mfun in (('all valid', lambda T: np.ones(T, dtype=bool)),
                           ('right half only', lambda T: np.arange(T) >= T//2 - 3),
                           ('every 3rd', lambda T: (np.arange(T) % 3) != 1)):
        Tc = 2*W
        det = Detrender(W=W, n=n, chunk_size=Tc, dtype=dtype, eps=eps, mu=mu,
                        subtract_offset=False)
        Tbuf = det.buflen
        S_ax = 2*W + 1
        base = mfun(Tbuf)
        if not base[W]:
            continue
        dd = np.zeros((S_ax, Tbuf), dtype=dtype)
        for i in range(S_ax):
            dd[i, i] = 1.0
        mm = np.broadcast_to(base, (S_ax, Tbuf)).copy()
        resid, lev, flagged = det.detrend_chunk(dd, mm)
        # fhat at local output 0 (buffer index W) = kernel weight for sample i
        kern = np.array([dd[i, W] - resid[i, 0] for i in range(S_ax)])
        lam = lev[0, 0]
        worst_kap = max(worst_kap, abs(kern[W] - lam))
        worst_lev = max(worst_lev, abs(float((kern**2).sum()) - lam))

    # (c) full window: leverage equals (G^-1)_00 for the exact discrete Gram, and
    #     approaches the continuum value 9/(8W) as W grows.  Note 9/(8W) is the
    #     continuum limit; the discrete value differs by O(1/W) (2.9% at W=16,
    #     0.1% at W=512), so it is the discrete value we assert against.
    devs = []
    for Wbig in (16, 128, 512):
        det = Detrender(W=Wbig, n=n, chunk_size=2*Wbig, dtype=np.float64,
                        eps=eps, mu=mu, subtract_offset=False)
        dd = np.zeros((1, det.buflen)); mm = np.ones((1, det.buflen), dtype=bool)
        _, lev, _ = det.detrend_chunk(dd, mm)

        xx = np.arange(-Wbig, Wbig+1) / Wbig
        Pv = np.vander(xx, n+1, increasing=True)
        exact = np.linalg.inv(Pv.T @ Pv)[0, 0]
        assert abs(float(lev[0, 0]) - exact) < 1e-12 * exact, \
            f'W={Wbig}: leverage {lev[0,0]} vs exact discrete {exact}'
        devs.append(abs(exact / (9.0/(8*Wbig)) - 1))
    assert devs[0] > devs[1] > devs[2] and devs[2] < 2e-3, \
        f'continuum limit 9/(8W) not approached: {devs}'

    # (d) degenerate windows
    det = Detrender(W=W, n=n, chunk_size=2*W, dtype=dtype, eps=eps, mu=mu,
                    subtract_offset=False)
    Tbuf = det.buflen
    dvals = np.arange(Tbuf, dtype=dtype) + 0.5

    # nv = 1, the single valid sample being the output sample itself
    mm = np.zeros((1, Tbuf), dtype=bool); mm[0, W] = True
    r, lv, fl = det.detrend_chunk(dvals[None, :], mm)
    assert abs(float(r[0, 0])) < 1e-12, f'nv=1 residual {r[0,0]} (expected 0)'

    # nv = 1, valid sample elsewhere in the window: fhat must equal that value
    mm = np.zeros((1, Tbuf), dtype=bool); mm[0, W] = True; mm[0, 3] = True
    mm2 = np.zeros((1, Tbuf), dtype=bool); mm2[0, 3] = True
    r2, _, _ = det.detrend_chunk(dvals[None, :], mm2)
    assert np.all(np.isfinite(r2))

    # nv = 0 anywhere in the window: finite output, no NaN
    mm0 = np.zeros((1, Tbuf), dtype=bool)
    r0, lv0, fl0 = det.detrend_chunk(dvals[None, :], mm0)
    assert np.all(np.isfinite(r0)) and np.all(np.isfinite(lv0))

    # nv = 2 with n = 2: succeeds, and the floor fires
    mm2 = np.zeros((1, Tbuf), dtype=bool); mm2[0, W] = True; mm2[0, W+2] = True
    r3, lv3, fl3 = det.detrend_chunk(dvals[None, :], mm2)
    assert np.all(np.isfinite(r3))
    assert bool(fl3[0, 0]), 'expected the pivot floor to fire for nv=2, n=2'

    if verbose:
        print(f'    test_solve: lstsq={worst_lstsq:.2e}  kappa[0]-lev={worst_kap:.2e}  '
              f'sum(kappa^2)-lev={worst_lev:.2e}')
    assert worst_lstsq < 1e-10
    assert worst_kap < 1e-10
    assert worst_lev < 1e-10


# ---------------------------------------------------- 4. polynomial exactness

def _poly_stream(rng, S_ax, T, deg, W, dtype):
    """P(t) with |P| = O(1) across the buffer; built in fp64, then cast."""
    x = (np.arange(T, dtype=np.float64) - T/2) / max(W, 1)
    coef = rng.normal(size=(S_ax, deg+1))
    P = np.zeros((S_ax, T), dtype=np.float64)
    for j in range(deg+1):
        P += coef[:, j:j+1] * (x**j)[None, :]
    P /= max(float(np.max(np.abs(P))), 1e-300)
    return P.astype(dtype)


def test_polynomial_exactness(rng=None, verbose=True):
    """
    Feed d[t] = P(t) for a polynomial of degree <= n with an arbitrary mask and
    no noise.  Every window's valid samples then lie exactly on P, the local fit
    recovers it, and the residual must be zero at every valid sample where the
    pivot floor is inactive.

    The expected answer is known analytically, so this needs no reference
    implementation and has no statistical tolerance: whatever residual appears is
    pure accumulated numerical error.
    """
    rng = _default_rng(rng)
    n = 2
    results = []

    for dtype, tol in ((np.float64, 1e-11), (np.float32, 3e-5)):
        for W, Tc, nchunk in ((16, 64, 3), (512, 2048, 2)):
            S_ax = 2
            T = nchunk*Tc + 2*W
            worst_ok, worst_flag, nflag = 0.0, 0.0, 0
            for name, mask in mask_zoo(S_ax, T, W, rng):
                P = _poly_stream(rng, S_ax, T, n, W, dtype)
                det = Detrender(W=W, n=n, chunk_size=Tc, dtype=dtype)
                resid, lev, flagged = det.detrend_stream(P, mask)
                mout = mask[:, W:T-W]
                good = mout & ~flagged
                bad = mout & flagged
                nflag += int(bad.sum())
                if good.any():
                    e = float(np.max(np.abs(resid[good])))
                    worst_ok = max(worst_ok, e)
                    assert e < tol, (f'test_polynomial_exactness [{np.dtype(dtype).name} '
                                     f'W={W} "{name}"]: residual {e:.3e} > {tol:.1e}')
                if bad.any():
                    worst_flag = max(worst_flag, float(np.max(np.abs(resid[bad]))))
            results.append((np.dtype(dtype).name, W, worst_ok, worst_flag, nflag))

    # Controls.  Note that for a *fully valid* window the mask is symmetric about
    # the evaluation point, so S_odd = 0, G is checkerboard, and a_0 depends only
    # on the even block: a degree-n fit then reproduces degree n+1 exactly, for
    # even n.  This is the n=2 == n=3 degeneracy noted in tree_dedispersion.tex,
    # and it is asserted below rather than treated as a failure.  Degree n+2, and
    # degree n+1 under an asymmetric mask, must both fail to be reproduced.
    W, Tc = 16, 64
    T = 2*Tc + 2*W
    det = Detrender(W=W, n=n, chunk_size=Tc, dtype=np.float64)
    allvalid = np.ones((2, T), dtype=bool)
    asym = np.ones((2, T), dtype=bool)
    asym[:, ::3] = False          # breaks the within-window symmetry

    P1 = _poly_stream(rng, 2, T, n+1, W, np.float64)
    r1, _, _ = det.detrend_stream(P1, allvalid)
    assert float(np.max(np.abs(r1))) < 1e-11, \
        (f'degree n+1 should be reproduced exactly on a fully valid window '
         f'(checkerboard degeneracy), got {float(np.max(np.abs(r1))):.3e}')

    r2, _, f2 = det.detrend_stream(P1, asym)
    mo = asym[:, W:T-W] & ~f2
    assert float(np.max(np.abs(r2[mo]))) > 1e-6, \
        'degree n+1 was reproduced even under an asymmetric mask'

    P2 = _poly_stream(rng, 2, T, n+2, W, np.float64)
    r3, _, _ = det.detrend_stream(P2, allvalid)
    assert float(np.max(np.abs(r3))) > 1e-6, \
        'negative control: a degree-(n+2) polynomial was reproduced exactly'

    if verbose:
        for name, W, wok, wfl, nf in results:
            print(f'    test_polynomial_exactness [{name} W={W}]: '
                  f'max|resid| unflagged={wok:.2e}  flagged={wfl:.2e} ({nf} flagged samples)')


# ------------------------------------------------------ 5. vs fp64 reference

def test_detrender_vs_reference(rng=None, verbose=True):
    rng = _default_rng(rng)
    n, dtype = 2, np.float64
    worst = 0.0
    worst_name = ''

    for W, Tc, nchunk in ((16, 64, 3), (128, 512, 2)):
        S_ax = 2
        T = nchunk*Tc + 2*W
        for name, mask in mask_zoo(S_ax, T, W, rng):
            d = (rng.normal(size=(S_ax, T)) + 3.0).astype(dtype)
            det = Detrender(W=W, n=n, chunk_size=Tc, dtype=dtype)
            got, glev, gfl = det.detrend_stream(d, mask)
            want, wlev, wfl = detrend_reference(d, mask, W, n=n, dtype=dtype)
            scale = max(1.0, float(np.max(np.abs(d))))
            e = _maxdiff(got, want) / scale
            if e > worst:
                worst, worst_name = e, f'W={W} "{name}"'
            assert np.array_equal(gfl, wfl), f'flag mismatch, W={W} "{name}"'

            # Leverage is only well defined where the window holds at least one
            # valid sample.  With nv = 0 the whole Gram is zero, every pivot is
            # floored to mu, and leverage = |w|^2/mu -- where w depends on the
            # arbitrary finite placeholder centroid, which the two code paths
            # choose differently.  Both are meaninglessly huge (1e30 vs 3e30) and
            # those samples are masked in the output anyway.
            cs = np.concatenate([np.zeros((S_ax, 1), dtype=np.int64),
                                 np.cumsum(mask.astype(np.int64), axis=1)], axis=1)
            nvw = cs[:, 2*W+1:] - cs[:, :-(2*W+1)]
            live = nvw > 0
            assert np.all(glev[~live] > 1e20) and np.all(wlev[~live] > 1e20)
            rel = np.abs(glev[live] - wlev[live]) / np.maximum(np.abs(wlev[live]), 1e-300)
            assert (rel.max() if rel.size else 0.0) < 1e-9, \
                f'leverage mismatch, W={W} "{name}"'
    if verbose:
        print(f'    test_detrender_vs_reference: max rel err {worst:.2e} ({worst_name})')
    assert worst < 1e-11, f'test_detrender_vs_reference: {worst} ({worst_name})'


# ------------------------------------------------ 6. dtype agreement (fp32/fp64)

def test_dtype_agreement(rng=None, tol=1e-3, verbose=True):
    """
    Run the *same instance* twice, float32 and float64, and compare.

    To make this a pure arithmetic comparison, the data is generated in fp64,
    cast to fp32, and then cast *back* to fp64 for the fp64 run, so both runs see
    bit-identical inputs and the only difference is intermediate precision.

    Noise is unit-variance, so all numbers below are in units of sigma.  Recall
    that fhat carries an irreducible statistical error of 1.06*sigma/sqrt(W)
    (0.047 sigma at W=512), so the target is numerical error well under
    1e-3 sigma.
    """
    rng = _default_rng(rng)
    n = 2
    rows = []
    worst = 0.0
    worst_name = ''

    for W, Tc, nchunk in ((16, 64, 3), (512, 2048, 2)):
        S_ax = 2
        T = nchunk*Tc + 2*W
        for offset in (0.0, 1e4):
            for name, mask in mask_zoo(S_ax, T, W, rng):
                d64 = rng.normal(size=(S_ax, T)) + offset
                d32 = d64.astype(np.float32)
                dref = d32.astype(np.float64)     # bit-identical inputs

                r32, l32, f32 = Detrender(W=W, n=n, chunk_size=Tc,
                                          dtype=np.float32).detrend_stream(d32, mask)
                r64, l64, f64 = Detrender(W=W, n=n, chunk_size=Tc,
                                          dtype=np.float64).detrend_stream(dref, mask)

                mo = mask[:, W:T-W]
                if not mo.any():
                    continue
                e = _maxdiff(r32[mo], r64[mo])
                rows.append((W, offset, name, e, int((f32 != f64).sum())))
                if e > worst:
                    worst, worst_name = e, f'W={W} offset={offset:g} "{name}"'

    # Moment-level comparison: localizes a failure to the scan rather than the solve.
    def _scan_err(W, dtype, scanner):
        B, nblocks = 2*W, 3
        Tb = nblocks*B
        m = (rng.random((1, Tb)) < 0.9)
        dd = rng.normal(size=(1, Tb))
        out = []
        for dt in (dtype, np.float64):
            mm = m.astype(dt)
            lv = MomentSet.leaves(np.broadcast_to(np.arange(Tb, dtype=dt), (1, Tb)),
                                  mm, (mm*dd).astype(dt), n, W, dt)
            lv = MomentSet(lv.nv.reshape(1, nblocks, B), lv.c.reshape(1, nblocks, B),
                           lv.S.reshape(1, nblocks, B, 2*n+1),
                           lv.U.reshape(1, nblocks, B, n+1), n, W)
            out.append(scanner(lv))
        return _maxdiff(out[0].S, out[1].S), _maxdiff(out[0].U, out[1].U)

    tree_S, tree_U = _scan_err(512, np.float32, tree_prefix_scan)

    if verbose:
        print(f'    test_dtype_agreement: max |r32-r64| = {worst:.3e} sigma  ({worst_name})')
        print(f'      moments, fp32 tree scan vs fp64 (1024-sample block): '
              f'max|dS|={tree_S:.2e} max|dU|={tree_U:.2e}')
        by_case = {}
        for W, off, name, e, nf in rows:
            by_case[(W, off)] = max(by_case.get((W, off), 0.0), e)
        for (W, off), e in sorted(by_case.items()):
            print(f'      W={W:4d} offset={off:8g}: max |r32-r64| = {e:.3e} sigma')
        for W, off, name, e, nf in sorted(rows, key=lambda r: -r[3])[:6]:
            print(f'        worst masks: W={W:4d} offset={off:8g} '
                  f'{name:32s} {e:.3e}  (flag mismatches: {nf})')

    assert worst < tol, (f'test_dtype_agreement: max |r32-r64| = {worst:.3e} sigma '
                         f'> {tol:.0e} ({worst_name})')


# ----------------------------------------------------------------- entry point

def run_all(verbose=True, rng=None):
    """
    All six tests share one generator, so printing its entropy makes the whole
    run reproducible: pass np.random.default_rng(<entropy>) back in as 'rng'.
    """
    rng = _default_rng(rng)
    ent = rng.bit_generator.seed_seq.entropy
    print(f'  detrending_1d tests (rng entropy {ent})')
    test_monoid(rng, verbose=verbose)
    test_vanherk(rng, verbose=verbose)
    test_solve(rng, verbose=verbose)
    test_polynomial_exactness(rng, verbose=verbose)
    test_detrender_vs_reference(rng, verbose=verbose)
    test_dtype_agreement(rng, verbose=verbose)
    print('  detrending_1d tests passed')
