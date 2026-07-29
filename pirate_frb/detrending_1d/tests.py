"""
Unit tests for the 1-d detrender.  Dispatched from pirate_frb/__main__.py:

    python -m pirate_frb test --dt1d

Most tests run at a small size (W=16, B=32, Tc=64) for speed -- the algorithm is
size-parameterized, so small sizes exercise the same code paths -- plus a
full-size (W=512) run where it is cheap enough to matter.

The polynomial degree n is drawn per call by run_all() rather than fixed, so a
multi-iteration run covers both degrees we build kernels for.  n=1 and n=2 are
not cosmetic variants of each other: at n=1 the Gram matrix is diagonal, so
rmin collapses to a valid-sample count and several of the controls below flip
sign.  Those places are marked.
"""

import numpy as np

from .MomentSet import MomentSet, merge, pascal_shift, _binom_table
from . import scan as scan_mod
from .scan import tree_prefix_scan, tree_suffix_scan
from .Detrender import Detrender
from .reference import detrend_reference
from .masks import random_mask
from . import LocalPolyFit


# ------------------------------------------------------------------ utilities

# Cumulative mask-expansion accounting, for visual inspection.  Deliberately
# module-level and never reset: under 'test --dt1d -n N' each iteration draws
# fresh masks, and the running total across iterations is what tells us whether
# expansion is firing at a sane rate.
# Keyed by polynomial degree, because the two degrees expand at rates that differ
# by about an order of magnitude (at n=1 the only windows rmin can reject are
# nv <= 1), and pooling them would make the number uninterpretable.
_EXPANSION = {}


def _note_expansion(n, mask_in, mask_out):
    """mask_in restricted to the output range, and the detrender's mask_out."""
    d = _EXPANSION.setdefault(n, {'valid_in': 0, 'expanded': 0})
    d['valid_in'] += int(mask_in.sum())
    d['expanded'] += int((mask_in & ~mask_out).sum())


def _expansion_str():
    return '  '.join(f"n={n}: {d['expanded']}/{d['valid_in']} = "
                     f"{(d['expanded']/d['valid_in'] if d['valid_in'] else 0.0):.3e}"
                     for n, d in sorted(_EXPANSION.items()))


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
    batch_shape = np.sum(m, axis=axis).shape
    S = np.zeros(batch_shape + (2*n+1,), dtype=dtype)
    U = np.zeros(batch_shape + (n+1,), dtype=dtype)
    p = np.ones_like(x)
    for r in range(2*n+1):
        S[..., r] = (m*p).sum(axis=axis)
        if r <= n:
            U[..., r] = (md*p).sum(axis=axis)
        p = p * x
    return S, U


def _random_set(rng, S_ax, L, W, dtype, offset=0.0):
    """A random (u, mask, mask*data) triple, mask drawn per row by random_mask()."""
    u = (np.arange(L, dtype=dtype) + offset)[None, :] * np.ones((S_ax, 1), dtype=dtype)
    m = random_mask(S_ax, L, W, rng)[0].astype(dtype)
    d = rng.normal(size=(S_ax, L)).astype(dtype)
    return u, m, (m*d).astype(dtype)


# ------------------------------------------------------------------- 1. monoid

def test_monoid(rng=None, n=2, niter=8, verbose=True):
    rng = _default_rng(rng)
    W, dtype = 16, np.float64
    binom = _binom_table(2*n+1)
    worst = dict(merge=0.0, assoc=0.0, comm=0.0, ident=0.0, s1=0.0, shift=0.0, empty=0.0)

    for _ in range(niter):
        S_ax, L = 8, 40
        u, m, md = _random_set(rng, S_ax, L, W, dtype, offset=100.0)

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

def test_vanherk(rng=None, n=2, niter=4, verbose=True):
    rng = _default_rng(rng)
    dtype = np.float64
    worst_ms, worst_cnt = 0.0, 0

    for W in (4, 16):
        B, nblocks = 2*W, 3
        T = nblocks*B
        for _ in range(niter):
            S_ax = 8
            m = random_mask(S_ax, T, W, rng)[0].astype(dtype)
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

    # The scans must be trees.  Nothing else in the suite would notice if they were
    # replaced by a sequential scan: that costs about 10x in moment accuracy on a
    # full block, which is still inside every tolerance we assert.  Check the
    # structural property directly, by counting merge() calls -- a Hillis-Steele
    # scan makes log2(B) vectorized calls, a sequential one makes B-1 -- rather
    # than by comparing accuracy, whose ratio depends on the mask and turns out to
    # be a weak signal (under 2x at a 90% valid fraction).
    Wt, Bt = 512, 1024
    mm = (rng.random((1, Bt)) < 0.9).astype(np.float64)
    dd = rng.normal(size=(1, Bt))
    lv = MomentSet.leaves(np.broadcast_to(np.arange(Bt, dtype=np.float64), (1, Bt)),
                          mm, (mm*dd).astype(np.float64), n, Wt, np.float64)
    ncalls = [0]
    orig = scan_mod.merge

    def _counting_merge(a, b):
        ncalls[0] += 1
        return orig(a, b)

    scan_mod.merge = _counting_merge
    try:
        tree_prefix_scan(lv)
    finally:
        scan_mod.merge = orig
    depth = int(np.ceil(np.log2(Bt)))

    if verbose:
        print(f'    test_vanherk: {worst_cnt} decompositions checked, max err {worst_ms:.2e}; '
              f'prefix scan over B={Bt} used {ncalls[0]} merges (tree depth {depth})')
    assert worst_ms < 1e-9, f'test_vanherk: {worst_ms}'
    assert ncalls[0] <= 2*depth, (f'the prefix scan is not a tree: {ncalls[0]} merge calls over '
                                  f'B={Bt}, expected about log2(B)={depth} '
                                  f'(a sequential scan would use {Bt-1})')


# -------------------------------------------------------------------- 3. solve

def test_solve(rng=None, n=2, niter=4, verbose=True):
    rng = _default_rng(rng)
    dtype = np.float64
    eps, mu = 1e-3, 1e-30
    W = 16
    L = 2*W + 1
    worst_lstsq, worst_lev, worst_kap = 0.0, 0.0, 0.0

    # (a) against np.linalg.lstsq on well-determined windows
    for _ in range(niter):
        S_ax = 16
        u = np.broadcast_to(np.arange(L, dtype=dtype), (S_ax, L))
        m = random_mask(S_ax, L, W, rng)[0].astype(dtype)
        d = rng.normal(size=(S_ax, L)).astype(dtype)
        ms = MomentSet.direct(u, m, (m*d).astype(dtype), n, W, dtype)
        u_eval = np.full(S_ax, float(W), dtype=dtype)
        _Lc, ratios = LocalPolyFit.cholesky(LocalPolyFit.gram(ms), mu)
        fhat, lev, rmin = LocalPolyFit.solve(ms, u_eval, mu)

        # With randomized masks some rows are genuinely degenerate; agreement with
        # lstsq is only meaningful on rows the detrender would keep.
        for s in np.flatnonzero(rmin >= eps):
            x = (u[s] - ms.c[s]) / W
            A = np.vander(x, n+1, increasing=True) * m[s][:, None]
            coef = np.linalg.lstsq(A, m[s]*d[s], rcond=None)[0]
            x0 = (W - ms.c[s]) / W
            want = sum(coef[j] * x0**j for j in range(n+1))
            worst_lstsq = max(worst_lstsq, abs(want - fhat[s]))

        # Pivot ratios lie in [0,1], and r_0 = r_1 = 1 exactly -- the latter
        # because G_01 = 0 under adaptive centering, so p_0 = G_00 and p_1 = G_11.
        # That only says anything where the corresponding G_ii > 0: for nv <= 1
        # every valid sample sits at the centroid, so G_11 = S_2 = 0 and the ratio
        # is reported as 0 ("no information at this order") rather than 1.
        assert np.all(ratios >= -1e-12) and np.all(ratios <= 1 + 1e-12)
        for i, gii in ((0, ms.S[..., 0]), (1, ms.S[..., 2])):
            live = gii > 0
            if live.any():
                assert _maxdiff(ratios[..., i][live], 1.0) < 1e-12, \
                    f'r_{i} != 1 where G_{i}{i} > 0'
            assert np.all(ratios[..., i][~live] == 0)

    # (b) leverage identity, via impulses through the FULL Detrender path.
    #     One spectator per impulse position, so a single call gives the kernel.
    #     Here the mask must be identical across spectators (we vary only the
    #     impulse position), and the centre sample must be valid for
    #     kappa[0] == leverage, so we draw one random row and force m[W] = True.
    Tc = 2*W
    det = Detrender(W=W, n=n, chunk_size=Tc, dtype=dtype, eps=eps, mu=mu,
                    subtract_offset=False)
    Tbuf = det.buflen
    S_ax = 2*W + 1
    nchecked = 0
    for _ in range(3*niter):
        base = random_mask(1, Tbuf, W, rng)[0][0]
        base[W] = True
        dd = np.zeros((S_ax, Tbuf), dtype=dtype)
        for i in range(S_ax):
            dd[i, i] = 1.0
        mm = np.broadcast_to(base, (S_ax, Tbuf)).copy()
        resid, mko, lev, _ = det.detrend_chunk(dd, mm)
        if not bool(mko[0, 0]):
            continue          # the sample was mask-expanded away
        # fhat at local output 0 (buffer index W) = kernel weight for sample i
        kern = np.array([dd[i, W] - resid[i, 0] for i in range(S_ax)])
        lam = lev[0, 0]
        worst_kap = max(worst_kap, abs(kern[W] - lam))
        worst_lev = max(worst_lev, abs(float((kern**2).sum()) - lam))
        nchecked += 1
    assert nchecked > 0, "leverage identity was never checked"

    # (c) full window: leverage equals (G^-1)_00 for the exact discrete Gram, and
    #     approaches the continuum value C_n/(2W) as W grows, with C_1 = 1 and
    #     C_2 = 9/4.  (At n=1 the fit on a symmetric window is just the boxcar
    #     mean, so the discrete value is exactly 1/(2W+1).)  Note C_n/(2W) is the
    #     continuum limit; the discrete value differs by O(1/W) (about 3% at
    #     W=16, 0.1% at W=512), so it is the discrete value we assert against.
    cont_c = {1: 1.0, 2: 2.25}[n]
    devs = []
    for Wbig in (16, 128, 512):
        det = Detrender(W=Wbig, n=n, chunk_size=2*Wbig, dtype=np.float64,
                        eps=eps, mu=mu, subtract_offset=False)
        dd = np.zeros((1, det.buflen)); mm = np.ones((1, det.buflen), dtype=bool)
        _, _, lev, _ = det.detrend_chunk(dd, mm)

        xx = np.arange(-Wbig, Wbig+1) / Wbig
        Pv = np.vander(xx, n+1, increasing=True)
        exact = np.linalg.inv(Pv.T @ Pv)[0, 0]
        assert abs(float(lev[0, 0]) - exact) < 1e-12 * exact, \
            f'W={Wbig}: leverage {lev[0,0]} vs exact discrete {exact}'
        devs.append(abs(exact / (cont_c/(2*Wbig)) - 1))
    assert devs[0] > devs[1] > devs[2] and devs[2] < 2e-3, \
        f'continuum limit {cont_c}/(2W) not approached: {devs}'

    # (d) degenerate windows
    det = Detrender(W=W, n=n, chunk_size=2*W, dtype=dtype, eps=eps, mu=mu,
                    subtract_offset=False)
    Tbuf = det.buflen
    dvals = np.arange(Tbuf, dtype=dtype) + 0.5

    # nv <= n cannot determine a degree-n fit, so all of these must be masked
    # away rather than producing a degenerate estimate.  nv=1 gives G_ii = 0 for
    # every i >= 1 (the single valid sample sits at the centroid), so rmin = 0.
    degenerate = [('nv=0', []), ('nv=1 at the output sample', [W]),
                  ('nv=1 elsewhere', [3])]
    if n >= 2:
        # nv=2 is the case that shows why rmin has to look at the pivots and not
        # at the diagonal: G_ii > 0 for every i, and yet p_2 = 0 exactly, since G
        # has rank 2.  At n=1 the same window has full rank and must SURVIVE, so
        # there it is the positive control below rather than a degenerate case.
        degenerate.append(('nv=2', [W, W+2]))

    for lbl, idx in degenerate:
        mm = np.zeros((1, Tbuf), dtype=bool)
        for i in idx:
            mm[0, i] = True
        r, mko, _, rmin = det.detrend_chunk(dvals[None, :], mm)
        assert np.all(np.isfinite(r)) and np.all(np.isfinite(rmin)), f'{lbl}: non-finite'
        assert not bool(mko[0, 0]), f'{lbl}: expected mask expansion, got mask_out=True'
        assert float(r[0, 0]) == 0.0, f'{lbl}: masked residual should be 0'

    # nv = n+1 spread across the window determines the fit exactly, so it must
    # survive -- rmin is a conditioning test, not a sample count.  The output
    # sample itself must be one of them, or mask_out is false for the trivial
    # reason.
    mm = np.zeros((1, Tbuf), dtype=bool)
    for i in [W, 1, 2*W-1][:n+1]:
        mm[0, i] = True
    r, mko, _, rmin = det.detrend_chunk(dvals[None, :], mm)
    assert bool(mko[0, 0]), f'nv={n+1} spread should survive (rmin={rmin[0,0]:.2e})'

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


def test_polynomial_exactness(rng=None, n=2, verbose=True):
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
    results = []

    for dtype, tol in ((np.float64, 1e-11), (np.float32, 3e-5)):
        for W, Tc, nchunk, S_ax, ndraw in ((16, 64, 3, 16, 2), (512, 2048, 2, 6, 1)):
            T = nchunk*Tc + 2*W
            worst_ok, nflag, worst_lbl = 0.0, 0, '-'
            for _ in range(ndraw):
                mask, labels = random_mask(S_ax, T, W, rng)
                P = _poly_stream(rng, S_ax, T, n, W, dtype)
                det = Detrender(W=W, n=n, chunk_size=Tc, dtype=dtype)
                resid, mko, _, _ = det.detrend_stream(P, mask)
                min_ = mask[:, W:T-W]
                _note_expansion(n, min_, mko)
                nflag += int((min_ & ~mko).sum())
                # No regularizer means no shrinkage bias, so exactness holds at
                # *every* surviving sample -- there is no carve-out to make.
                for s in range(S_ax):
                    if not mko[s].any():
                        continue
                    e = float(np.max(np.abs(resid[s][mko[s]])))
                    if e > worst_ok:
                        worst_ok, worst_lbl = e, labels[s]
                    assert e < tol, (f'test_polynomial_exactness [{np.dtype(dtype).name} '
                                     f'W={W} mask={labels[s]}]: residual {e:.3e} > {tol:.1e}')
            results.append((np.dtype(dtype).name, W, worst_ok, nflag, worst_lbl))

    # Controls.  For a *fully valid* window the mask is symmetric about the
    # evaluation point, so S_odd = 0 and G is checkerboard: the even and odd
    # coefficients decouple, and fhat (which is a_0, since x_0 = 0) is determined
    # by the even part of the data alone.  A degree-D polynomial is therefore
    # reproduced exactly iff its even part is spanned by the even basis functions
    # {1, x^2, ..., x^(2*(n//2))}, i.e. iff D//2 <= n//2.
    #
    # For even n that admits D = n+1 -- the n=2 == n=3 degeneracy noted in
    # tree_dedispersion.tex, asserted below rather than treated as a failure.
    # For odd n it does not (at n=1 the even basis is just the constant), so
    # degree n+1 joins degree n+2 as a negative control.  Degree n+1 under an
    # *asymmetric* mask must fail at every n, since the decoupling is what the
    # symmetry was buying.
    W, Tc = 16, 64
    T = 2*Tc + 2*W
    det = Detrender(W=W, n=n, chunk_size=Tc, dtype=np.float64)
    allvalid = np.ones((2, T), dtype=bool)
    asym = np.ones((2, T), dtype=bool)
    asym[:, ::3] = False          # breaks the within-window symmetry

    P1 = _poly_stream(rng, 2, T, n+1, W, np.float64)
    r1, _, _, _ = det.detrend_stream(P1, allvalid)
    w1 = float(np.max(np.abs(r1)))
    if (n+1)//2 <= n//2:
        assert w1 < 1e-11, \
            (f'degree n+1 should be reproduced exactly on a fully valid window '
             f'(checkerboard degeneracy, n={n}), got {w1:.3e}')
    else:
        assert w1 > 1e-6, \
            (f'degree n+1 was reproduced on a fully valid window at n={n}, where '
             f'the even basis functions cannot span its even part')

    r2, mk2, _, _ = det.detrend_stream(P1, asym)
    mo = mk2
    assert float(np.max(np.abs(r2[mo]))) > 1e-6, \
        'degree n+1 was reproduced even under an asymmetric mask'

    P2 = _poly_stream(rng, 2, T, n+2, W, np.float64)
    r3, _, _, _ = det.detrend_stream(P2, allvalid)
    assert float(np.max(np.abs(r3))) > 1e-6, \
        'negative control: a degree-(n+2) polynomial was reproduced exactly'

    if verbose:
        for name, W, wok, nf, lbl in results:
            print(f'    test_polynomial_exactness [{name} W={W}]: '
                  f'max|resid| = {wok:.2e} ({lbl})  ({nf} samples mask-expanded)')


# ------------------------------------------------------ 5. vs fp64 reference

def test_detrender_vs_reference(rng=None, n=2, verbose=True):
    rng = _default_rng(rng)
    dtype = np.float64
    worst = 0.0
    worst_name = ''

    for W, Tc, nchunk, S_ax, ndraw in ((16, 64, 3, 16, 2), (128, 512, 2, 8, 1)):
        T = nchunk*Tc + 2*W
        for _ in range(ndraw):
            mask, labels = random_mask(S_ax, T, W, rng)
            d = (rng.normal(size=(S_ax, T)) + 3.0).astype(dtype)
            det = Detrender(W=W, n=n, chunk_size=Tc, dtype=dtype)
            got, gmk, glev, grm = det.detrend_stream(d, mask)
            want, wmk, wlev, wrm = detrend_reference(d, mask, W, n=n, dtype=dtype)
            _note_expansion(n, mask[:, W:T-W], gmk)
            scale = max(1.0, float(np.max(np.abs(d))))
            for s in range(S_ax):
                e = _maxdiff(got[s], want[s]) / scale
                if e > worst:
                    worst, worst_name = e, f'W={W} {labels[s]}'
            assert np.array_equal(gmk, wmk), f'mask_out mismatch, W={W} {labels}'
            assert _maxdiff(grm, wrm) < 1e-11, f'rmin mismatch, W={W} {labels}'

            # Leverage is only well defined where the window holds at least one
            # valid sample.  With nv = 0 the whole Gram is zero, every pivot is
            # floored to mu, and leverage = |w|^2/mu -- where w depends on the
            # arbitrary finite placeholder centroid, which the two code paths
            # choose differently.  Both are meaninglessly huge (1e30 vs 3e30) and
            # those samples are masked in the output anyway.
            # Leverage is only meaningful on surviving samples; elsewhere the mu
            # guard, not the data, set the smallest pivot.
            rel = (np.abs(glev[gmk] - wlev[gmk]) / np.maximum(np.abs(wlev[gmk]), 1e-300)
                   if gmk.any() else np.zeros(0))
            assert (rel.max() if rel.size else 0.0) < 1e-9, \
                f'leverage mismatch, W={W} {labels}'
    if verbose:
        print(f'    test_detrender_vs_reference: max rel err {worst:.2e} ({worst_name})')
    assert worst < 1e-11, f'test_detrender_vs_reference: {worst} ({worst_name})'


# ------------------------------------------------ 6. dtype agreement (fp32/fp64)

def test_dtype_agreement(rng=None, n=2, tol=1e-3, verbose=True):
    """
    Run the same instance twice, float32 and float64, and compare.

    The data is generated in fp64, cast to fp32, then cast *back* to fp64 for the
    fp64 run, so both runs see bit-identical inputs and the only difference is
    intermediate precision.  Noise is unit-variance, so all numbers are in units
    of sigma.

    The two runs use *different* masking thresholds, eps32 = 1e-3 and
    eps64 = 1e-6, so they do not produce the same output mask.  Residuals are
    therefore compared on the intersection.  eps32 > eps64 makes the float32 mask
    close to a superset of the float64 one, but they are not exactly nested since
    rmin32 != rmin64.

    A constant offset ~ U(0, 1e3) is added to the data on each draw.  This is what
    catches a broken constant-offset subtraction, which shows up as ~5e-3 sigma
    against ~1e-7 sigma when it is working -- the failure mode that a stale kappa
    inherited from a previous chunk used to produce.  1e3 rather than a larger
    value because the residual error scales as 3.4*eps_mach*|d - kappa|, so the
    requirement is |d - kappa| <~ 1e3 sigma; at that level float32's ulp is
    6.1e-5 sigma, so input quantization stays mild.
    """
    rng = _default_rng(rng)
    eps32, eps64 = 1e-3, 1e-6
    rows = []
    worst = 0.0
    worst_name = ''
    worst_rmin = 0.0

    for W, Tc, nchunk, S_ax, ndraw in ((16, 64, 3, 16, 4), (512, 2048, 2, 6, 2)):
        T = nchunk*Tc + 2*W
        for _ in range(ndraw):
            mask, labels = random_mask(S_ax, T, W, rng)
            offset = rng.uniform(0.0, 1e3)
            d64 = rng.normal(size=(S_ax, T)) + offset
            d32 = d64.astype(np.float32)
            dref = d32.astype(np.float64)     # bit-identical inputs

            r32, m32, _, q32 = Detrender(W=W, n=n, chunk_size=Tc, eps=eps32,
                                           dtype=np.float32).detrend_stream(d32, mask)
            r64, m64, _, q64 = Detrender(W=W, n=n, chunk_size=Tc, eps=eps64,
                                           dtype=np.float64).detrend_stream(dref, mask)
            _note_expansion(n, mask[:, W:T-W], m32)

            both = m32 & m64
            worst_rmin = max(worst_rmin, _maxdiff(q32, q64))
            for s in range(S_ax):
                if not both[s].any():
                    continue
                e = _maxdiff(r32[s][both[s]], r64[s][both[s]])
                vin = mask[s, W:T-W]
                rows.append((W, labels[s], e, int((vin & ~m32[s]).sum()), int(vin.sum()), offset))
                if e > worst:
                    worst, worst_name = e, f'W={W} {labels[s]} offset={offset:.3g}'

    if verbose:
        print(f'    test_dtype_agreement: max |r32-r64| = {worst:.3e} sigma  ({worst_name})')
        print(f'      max |rmin32-rmin64| = {worst_rmin:.3e}   '
              f'(eps32={eps32:.0e}, eps64={eps64:.0e})')
        by_W = {}
        for W, name, e, nx, nv, off in rows:
            by_W[W] = max(by_W.get(W, 0.0), e)
        for W, e in sorted(by_W.items()):
            print(f'      W={W:4d}: max |r32-r64| = {e:.3e} sigma')
        for W, name, e, nx, nv, off in sorted(rows, key=lambda r: -r[2])[:4]:
            print(f'        worst masks: W={W:4d} {name:16s} {e:.3e}  '
                  f'(expanded {nx}/{nv}, offset {off:.3g})')

    assert worst < tol, (f'test_dtype_agreement: max |r32-r64| = {worst:.3e} sigma '
                         f'> {tol:.0e} ({worst_name})')


# ------------------------------------------------- 7. masked data must be unread

def test_masked_data_unused(rng=None, n=2, verbose=True):
    """
    The detrender must never read a masked sample.  Checked by poisoning the
    masked entries and requiring every output to be *bit-identical* to a run on
    clean data.

    Bit-identity rather than "the tolerances still hold": a leak small enough to
    stay inside a tolerance would otherwise go unnoticed, and the whole point is
    that masked values must have exactly zero influence, not merely a small one.

    The poison includes nan and inf as well as huge finite values, since a masked
    sample may hold literally anything (a dropped packet can leave uninitialized
    memory behind).  That is the strong form: it fails for any arithmetic that
    weights by the mask rather than selecting on it, because 0*inf and 0*nan are
    nan.
    """
    rng = _default_rng(rng)
    checked = 0

    for W, Tc, nchunk, S_ax in ((16, 64, 3, 16), (512, 2048, 2, 4)):
        T = nchunk*Tc + 2*W
        mask, _ = random_mask(S_ax, T, W, rng)
        clean = rng.normal(size=(S_ax, T)) + rng.uniform(0.0, 1e3)
        junk = rng.uniform(-1e10, 1e10, size=(S_ax, T))
        pick = rng.integers(0, 4, size=(S_ax, T))
        junk = np.where(pick == 1, np.inf, junk)
        junk = np.where(pick == 2, -np.inf, junk)
        junk = np.where(pick == 3, np.nan, junk)
        poison = np.where(mask, clean, junk)

        for dtype in (np.float32, np.float64):
            det = lambda d: Detrender(W=W, n=n, chunk_size=Tc,
                                      dtype=dtype).detrend_stream(d.astype(dtype), mask)
            for nm, x, y in zip(('residual', 'mask_out', 'leverage', 'rmin'),
                                det(clean), det(poison)):
                assert np.array_equal(x, y), \
                    f'{nm} changed when masked samples were poisoned ' \
                    f'({np.dtype(dtype).name}, W={W})'
                checked += 1

        for nm, x, y in zip(('residual', 'mask_out', 'leverage', 'rmin'),
                            detrend_reference(clean, mask, W, n=n),
                            detrend_reference(poison, mask, W, n=n)):
            assert np.array_equal(x, y), \
                f'detrend_reference: {nm} changed when masked samples were poisoned (W={W})'
            checked += 1

    if verbose:
        print(f'    test_masked_data_unused: {checked} output arrays bit-identical '
              f'under nan/inf/+-1e10 poisoning of masked samples')

# ------------------------------------------------------------- 8. GPU kernel

def test_gpu_kernel(rng=None, n=2, tol=1e-3, rmin_tol=1e-4, verbose=True):
    """
    Compare pirate.Detrender1d (the GPU kernel) to detrend_reference().

    Structured like test_dtype_agreement(): the GPU runs in float32 at
    eps = 1e-3 and the reference in float64 at eps = 1e-6, so the two do not
    produce the same output mask and residuals are compared on the
    intersection.  Data is generated in fp64, cast to fp32, then cast back for
    the reference, so both see bit-identical inputs.  A constant offset
    ~ U(0, 1e3) is added on each draw, which is what catches a broken
    constant-offset subtraction (the kernel uses a per-BLOCK kappa rather than
    the reference's per-buffer one, which is exact in principle but has its own
    ways to go wrong -- see the comments in src_lib/Detrender1d.cu).

    The float32 CPU Detrender is run on the same data at the same parameters
    and reported alongside, so the two implementations' numerical stability can
    be compared directly.  Do not expect them to agree closely: they sum in
    completely different orders.

    The kernel emits no rmin, so mask expansion is checked against the
    reference's: the GPU mask must equal mask_in & (rmin64 >= eps32), except on
    samples whose rmin sits within 'rmin_tol' of the threshold, where the two
    are free to disagree.  rmin_tol = 1e-4 is an order of magnitude above the
    measured float32-vs-float64 disagreement in rmin, and an order of magnitude
    below eps itself.

    Unlike the rest of this module, W, T and eps are not free parameters: they
    are compile-time constants of the kernel and are read back from it.  Every
    compiled configuration of the requested degree is tested; if there is none
    (the kernel does not have to be built for every degree the numpy reference
    supports) the test is a no-op and says so.
    """
    from ..kernels import Detrender1d   # local import: this package is otherwise numpy-only
    import cupy as cp

    rng = _default_rng(rng)
    cfgs = [c for c in Detrender1d.configs() if c[0] == n]
    if not cfgs:
        if verbose:
            print(f'    test_gpu_kernel: no kernel compiled at n={n}, skipped '
                  f'(have {Detrender1d.configs()})')
        return

    eps32, eps64 = Detrender1d.eps, 1e-6
    ndraw = 2

    for (_n, W, T) in cfgs:
        _test_gpu_kernel_1(rng, Detrender1d(n=n, W=W, T=T), cp, eps32, eps64,
                           ndraw, tol, rmin_tol, verbose)


def _test_gpu_kernel_1(rng, det, cp, eps32, eps64, ndraw, tol, rmin_tol, verbose):
    """One configuration of test_gpu_kernel(); see its docstring."""
    n, W, T, nbuf = det.n, det.W, det.T, det.nbuf
    S_ax = 8

    worst_gpu, worst_cpu, worst_name = 0.0, 0.0, ''
    ndisagree, worst_slack = 0, 0.0

    for _ in range(ndraw):
        mask, labels = random_mask(S_ax, nbuf, W, rng)
        offset = rng.uniform(0.0, 1e3)
        d32 = (rng.normal(size=(S_ax, nbuf)) + offset).astype(np.float32)
        dref = d32.astype(np.float64)      # bit-identical inputs

        gpu_d = cp.asarray(d32)
        gpu_m = cp.asarray(mask.astype(np.uint8))
        det.launch(gpu_d, gpu_m)
        cp.cuda.get_current_stream().synchronize()
        out_d, out_m = cp.asnumpy(gpu_d), cp.asnumpy(gpu_m)

        # The kernel writes samples [W, W+T) in place and must leave the padding alone.
        # This is the check that catches an off-by-one in the output range, or a write
        # that races ahead of a read of the same buffer.
        for lo, hi, what in ((0, W, 'prepadding'), (W+T, nbuf, 'postpadding')):
            assert np.array_equal(out_d[:, lo:hi], d32[:, lo:hi]), f'{what} data modified'
            assert np.array_equal(out_m[:, lo:hi], mask[:, lo:hi]), f'{what} mask modified'

        r_gpu, m_gpu = out_d[:, W:W+T], out_m[:, W:W+T].astype(bool)
        r_cpu, m_cpu, _, _ = Detrender(W=W, n=n, chunk_size=T, eps=eps32,
                                       dtype=np.float32).detrend_stream(d32, mask)
        r_ref, m_ref, _, rmin = detrend_reference(dref, mask, W, n=n, eps=eps64,
                                                  dtype=np.float64,
                                                  max_outputs_per_pass=512)
        _note_expansion(n, mask[:, W:W+T], m_gpu)

        bad = (m_gpu != (mask[:, W:W+T] & (rmin >= eps32)))
        if bad.any():
            ndisagree += int(bad.sum())
            worst_slack = max(worst_slack, float(np.abs(rmin[bad] - eps32).max()))

        for s in range(S_ax):
            for m_other, r_other, which in ((m_gpu, r_gpu, 'gpu'), (m_cpu, r_cpu, 'cpu')):
                both = m_other[s] & m_ref[s]
                if not both.any():
                    continue
                e = _maxdiff(r_other[s][both], r_ref[s][both])
                if which == 'gpu' and e > worst_gpu:
                    worst_gpu, worst_name = e, f'{labels[s]} offset={offset:.3g}'
                elif which == 'cpu':
                    worst_cpu = max(worst_cpu, e)

    if verbose:
        print(f'    test_gpu_kernel [n={n} W={W} T={T}]: max |r_gpu-r_ref| = {worst_gpu:.3e} sigma  ({worst_name})')
        print(f'      same data through the float32 CPU Detrender: {worst_cpu:.3e} sigma')
        print(f'      mask expansion disagreements: {ndisagree} '
              f'(worst |rmin-eps| = {worst_slack:.3e}, tolerated below {rmin_tol:.0e})')

    assert worst_slack < rmin_tol, (f'test_gpu_kernel: {ndisagree} mask disagreements, worst '
                                    f'|rmin-eps| = {worst_slack:.3e} > {rmin_tol:.0e}')
    assert worst_gpu < tol, (f'test_gpu_kernel: max |r_gpu-r_ref| = {worst_gpu:.3e} sigma '
                             f'> {tol:.0e} ({worst_name})')


# ----------------------------------------------------------------- entry point

def run_all(verbose=True, rng=None, n=None):
    """
    All eight tests share one generator, so printing its entropy makes the whole
    run reproducible: pass np.random.default_rng(<entropy>) back in as 'rng'.

    The polynomial degree is drawn from {1, 2} per call rather than fixed, so a
    multi-iteration run ('test --dt1d -n 100') covers both.  Pass n explicitly to
    pin it.  test_gpu_kernel follows the same degree, and skips itself if no
    kernel is compiled for it; it is also the only test that needs a GPU (and the
    compiled extension), the other seven being pure numpy.
    """
    rng = _default_rng(rng)
    ent = rng.bit_generator.seed_seq.entropy
    if n is None:
        n = int(rng.integers(1, 3))
    print(f'  detrending_1d tests (n={n}, rng entropy {ent})')
    test_monoid(rng, n=n, verbose=verbose)
    test_vanherk(rng, n=n, verbose=verbose)
    test_solve(rng, n=n, verbose=verbose)
    test_polynomial_exactness(rng, n=n, verbose=verbose)
    test_detrender_vs_reference(rng, n=n, verbose=verbose)
    test_dtype_agreement(rng, n=n, verbose=verbose)
    test_masked_data_unused(rng, n=n, verbose=verbose)
    test_gpu_kernel(rng, n=n, verbose=verbose)
    print(f'  detrending_1d tests passed   [cumulative mask expansion: {_expansion_str()}]')
