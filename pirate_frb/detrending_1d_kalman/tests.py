"""
Unit tests for the fixed-lag Kalman detrender.  Dispatched from pirate_frb/__main__.py:

    python -m pirate_frb test --dt1k

T1-T8 are debugging tests: each has an oracle or an analytic answer, so a failure
points at a specific line.  T9 (test_dtype_agreement) is different in kind -- it is a
test of whether we understand where this estimator's float32 error lives and whether
rmin is the right mask criterion -- so it is run last and reported separately.  A T9
failure is evidence about the design, not a bug to chase.

Sizes are small (tau=4, L=32, Tc=64) except where noted; the algorithm is
size-parameterized, so small sizes exercise the same code paths.
"""

import numpy as np

from ..detrending_1d.masks import random_mask
from .model import StateSpaceModel, tau_from_equivalent_W
from .InfoFilter import forward_step, backward_step
from .KalmanDetrender import KalmanDetrender
from .brute_force import (kalman_brute_force, impulse_kernel, difference_matrix,
                          _state_from_samples)


# Cumulative mask-expansion accounting, kept separate from detrending_1d's: the two
# detrenders expand for different reasons and at wildly different rates, so a pooled
# number would describe neither.
_EXPANSION = {'valid_in': 0, 'expanded': 0}


def _note_expansion(mask_in, mask_out):
    _EXPANSION['valid_in'] += int(mask_in.sum())
    _EXPANSION['expanded'] += int((mask_in & ~mask_out).sum())


def _expansion_str():
    v, e = _EXPANSION['valid_in'], _EXPANSION['expanded']
    return f'{e}/{v} = {(e/v if v else 0.0):.3e}'


def _default_rng(rng):
    """A fresh numpy Generator for one test, seeded from the master --seed.

    NEVER call numpy's zero-argument default_rng() here.  It seeds itself from OS
    ENTROPY, which puts every draw in this file outside __main__.seed_rngs(): a
    failing draw then cannot be replayed, and 'test' prints a seed that does not
    cover these suites.  Seeded instead from numpy's global RandomState, which
    seed_rngs() pins -- so successive calls still differ (a long run explores
    different data) while the whole sequence replays from one integer.  This is the
    same rule, for the same reason, as varmap/tests.py's _rng().

    run_all() prints the entropy of the generator it makes, which reproduces THIS
    SUITE on its own without re-running anything else: pass
    np.random.default_rng(<entropy>) back in as 'rng'.
    """
    return np.random.default_rng(np.random.randint(0, 1 << 32)) if rng is None else rng


def _maxdiff(a, b):
    a = np.asarray(a, dtype=np.float64)
    b = np.asarray(b, dtype=np.float64)
    return float(np.max(np.abs(a-b))) if a.size else 0.0


# Default small geometry.  L/ell = 5.7 here, so exp(-L/ell) = 3e-3.
K, TAU, LAG, TC = 2, 4.0, 32, 64


def _det(**kw):
    kw.setdefault('k', K)
    kw.setdefault('tau', TAU)
    kw.setdefault('L', LAG)
    kw.setdefault('chunk_size', TC)
    kw.setdefault('dtype', np.float64)
    return KalmanDetrender(**kw)


# --------------------------------------------------------------------- T1. model

def test_model_algebra(rng=None, verbose=True):
    rng = _default_rng(rng)
    worst = 0.0

    for k in (2, 3, 4):
        for tau in (2.0, 4.0, 37.5):
            mo = StateSpaceModel(k, tau, dtype=np.float64)

            # A and A^-1 are exact integer matrices (no rescaling), so these are
            # array_equal rather than allclose.
            N = np.diag(np.ones(k-1), 1)
            assert np.array_equal(mo.A, np.eye(k) + N), f'A != I+N at k={k}'
            assert np.array_equal(mo.A @ mo.Ainv, np.eye(k)), f'A Ainv != I at k={k}'
            assert np.array_equal(mo.Ainv @ mo.A, np.eye(k)), f'Ainv A != I at k={k}'

            # A^s is the Pascal matrix; check it against repeated multiplication.
            P = np.eye(k)
            for s in range(0, 5):
                assert np.array_equal(P, mo.A_pow(s)), f'A^{s} != Pascal at k={k}'
                P = P @ mo.A

            # parameter round-trip
            worst = max(worst, abs(mo.rho - tau**(2*k))/mo.rho)
            worst = max(worst, abs(mo.q*mo.rho - 1.0))
            worst = max(worst, abs(mo.invq - mo.rho)/mo.rho)
            s_ = np.sin(np.pi/(2*k))
            worst = max(worst, abs(mo.ell - tau/s_)/mo.ell)
            worst = max(worst, abs(mo.c_k - 1.0/(2*k*s_))*2*k)
            worst = max(worst, abs(mo.ell - 2*k*mo.c_k*tau)/mo.ell)

    # tau matched to a local polynomial fit of half-width W
    assert abs(tau_from_equivalent_W(2, 256)/256 - 0.3143) < 1e-3

    # k != 2 must be rejected by the detrender even though the model supports it.
    for bad in (1, 3):
        try:
            _det(k=bad)
        except ValueError:
            pass
        else:
            raise AssertionError(f'KalmanDetrender accepted k={bad}')

    if verbose:
        print(f'    T1 test_model_algebra: A, A^-1, A^s exact; params round-trip to {worst:.2e}')
    assert worst < 1e-14


# ------------------------------------------------------- T2. recursions vs dense

def _dense_state_posterior(d_row, m_row, n_obs, k, tau):
    """
    Dense posterior of f for the sub-problem on [0, n-1] where only the first 'n_obs'
    samples are observed.  Returns (Sigma, fhat_all) with Sigma = N^-1, or None if
    the sub-problem holds fewer than k valid samples, where N is singular and there
    is nothing to compare against.

    The trailing n - n_obs samples are unobserved on purpose: the forward filter's
    state x[t] involves f[t..t+k-1], so the dense analogue has to extend past the
    last observation.
    """
    n = len(m_row)
    m_sub = m_row.astype(np.float64).copy()
    m_sub[n_obs:] = 0.0
    if m_sub.sum() < k:
        return None
    D = difference_matrix(n, k)
    N = np.diag(m_sub) + (tau ** (2*k))*(D.T @ D)
    Sigma = np.linalg.inv(N)
    return Sigma, Sigma @ (m_sub * d_row)


def _run_recursions(d_row, m_row, t, k, tau, L):
    """Forward filter to t, backward over [t+1,t+L], and their sum -- one row, plain
    python, so the test does not depend on KalmanDetrender's chunking."""
    mo = StateSpaceModel(k, tau, dtype=np.float64)
    J = np.zeros((k, k))
    eta = np.zeros(k)
    for u in range(t+1):
        Jm, em, J, eta = forward_step(J, eta, np.array(m_row[u]), np.array(d_row[u]), mo)
    Jf, ef = Jm, em

    Jb = np.zeros((k, k))
    eb = np.zeros(k)
    for u in range(t+L, t, -1):
        Jb, eb = backward_step(Jb, eb, np.array(m_row[u]), np.array(d_row[u]), mo)
    return (Jf, ef), (Jb, eb), (Jf+Jb, ef+eb)


def test_recursions_vs_dense(rng=None, niter=3, verbose=True):
    """
    The test that pins the signs and the A vs A^-T placement.

    Two independent dense comparisons: the forward filter alone against the posterior
    of [0,t], and the combination against the posterior of [0,t+L].  Those two pin the
    BACKWARD recursion as well, since J_b = J - J_f is then determined by difference
    -- which is why there is no separate dense backward oracle here (the backward
    factor is a likelihood, not a distribution, so it has no clean dense analogue).
    """
    rng = _default_rng(rng)
    k, tau, L = 2, 4.0, 12
    C = _state_from_samples(k)
    worst_f, worst_c = 0.0, 0.0
    nchecked = 0

    for _ in range(niter):
        T = 40
        mask = random_mask(6, T, L, rng)[0]
        d = rng.normal(size=(6, T))
        for s in range(6):
            for t in (5, 12, 20):
                m_row, d_row = mask[s].astype(float), d[s]
                (Jf, ef), _, (J, e) = _run_recursions(d_row, m_row, t, k, tau, L)

                # forward alone: sub-problem on [0, t+k-1], observed only to t.
                n = t + k
                got = _dense_state_posterior(d_row[:n], m_row[:n], t+1, k, tau)
                if got is not None:
                    Sig, fh = got
                    Jd = np.linalg.inv(C @ Sig[t:t+k, t:t+k] @ C.T)
                    ed = Jd @ (C @ fh[t:t+k])
                    worst_f = max(worst_f, _maxdiff(Jf, Jd)/max(1.0, np.abs(Jd).max()))
                    worst_f = max(worst_f, _maxdiff(ef, ed)/max(1.0, np.abs(ed).max()))

                # combined: sub-problem on [0, t+L], all observed.
                n = t + L + 1
                got = _dense_state_posterior(d_row[:n], m_row[:n], n, k, tau)
                if got is None:
                    continue
                Sig, fh = got
                Jd = np.linalg.inv(C @ Sig[t:t+k, t:t+k] @ C.T)
                ed = Jd @ (C @ fh[t:t+k])
                worst_c = max(worst_c, _maxdiff(J, Jd)/max(1.0, np.abs(Jd).max()))
                worst_c = max(worst_c, _maxdiff(e, ed)/max(1.0, np.abs(ed).max()))
                nchecked += 1

    if verbose:
        print(f'    T2 test_recursions_vs_dense: {nchecked} points; forward {worst_f:.2e}, '
              f'combined {worst_c:.2e}')
    assert nchecked > 0
    assert worst_f < 1e-8, f'forward filter vs dense: {worst_f}'
    assert worst_c < 1e-8, f'combined vs dense: {worst_c}'


# --------------------------------------------------- T3. polynomial exactness

def _poly_stream(rng, S_ax, T, deg, scale, dtype):
    """P(t) with O(1) coefficients in units where the trend moves by ~1 per 'scale'."""
    x = (np.arange(T, dtype=np.float64) - T/2) / scale
    coef = rng.normal(size=(S_ax, deg+1))
    P = np.zeros((S_ax, T), dtype=np.float64)
    for j in range(deg+1):
        P += coef[:, j:j+1] * (x**j)[None, :]
    P /= max(float(np.max(np.abs(P))), 1e-300)
    return P.astype(dtype)


def test_polynomial_exactness(rng=None, verbose=True):
    """
    Degree <= k-1 is annihilated exactly, for any mask and any position, because
    f = P zeroes BOTH terms of chi^2.  That is the analytic, tolerance-free test.

    Degrees k .. 2k-1 are annihilated only on the infinite array: the obstruction is
    an array endpoint, not a mask edge, and the fixed-lag estimator always has one L
    away.  So degree 2k-1 is asserted two-sided -- O(1) near the stream start where
    the left endpoint is close, and O(exp(-L/ell)) in the interior -- which is what
    keeps either half from silently regressing.
    """
    rng = _default_rng(rng)
    k = K
    results = []

    for dtype, tol in ((np.float64, 1e-10), (np.float32, 1e-4)):
        for tau, L, Tc, nchunk, S_ax in ((4.0, 32, 64, 3, 8), (16.0, 128, 256, 2, 4)):
            T = nchunk*Tc + L
            worst, lbl = 0.0, '-'
            mask, labels = random_mask(S_ax, T, L, rng)
            P = _poly_stream(rng, S_ax, T, k-1, tau, dtype)
            det = KalmanDetrender(k=k, tau=tau, L=L, chunk_size=Tc, dtype=dtype)
            resid, mko, _ = det.detrend_stream(P, mask)
            _note_expansion(mask[:, :T-L], mko)
            for s in range(S_ax):
                if not mko[s].any():
                    continue
                e = float(np.max(np.abs(resid[s][mko[s]])))
                if e > worst:
                    worst, lbl = e, labels[s]
                assert e < tol, (f'T3 [{np.dtype(dtype).name} tau={tau} mask={labels[s]}]: '
                                 f'deg<=k-1 residual {e:.3e} > {tol:.1e}')
            results.append((np.dtype(dtype).name, tau, worst, lbl))

    # (E2) two-sided at deg = 2k-1, on a fully valid mask so the only endpoint is the
    # array's.  L/ell = 8 here, so the interior residual should be ~1e-3 of the
    # start-of-stream one.
    tau, nl = 8.0, 8.0
    ell = tau/np.sin(np.pi/(2*k))
    L = int(round(nl*ell))
    Tc = 4*L
    T = 2*Tc + L
    det = KalmanDetrender(k=k, tau=tau, L=L, chunk_size=Tc, dtype=np.float64)
    P = _poly_stream(rng, 1, T, 2*k-1, tau, np.float64)
    r3, _, _ = det.detrend_stream(P, np.ones((1, T), dtype=bool))
    head = float(np.max(np.abs(r3[0, :int(2*ell)])))
    interior = float(np.max(np.abs(r3[0, int(8*ell):])))

    # The suppression in the interior is the exp(-L/ell) law, so tie the assertion to
    # it rather than to an arbitrary constant.  head is only required to sit well
    # clear of the fp64 floor: the polynomial is normalized over the whole stream, so
    # its absolute size at the head is a property of the test geometry and not of the
    # estimator, whereas the RATIO is the thing the law predicts.
    decay = float(np.exp(-nl))
    ratio = interior/max(head, 1e-300)

    if verbose:
        for name, tau_, w, lbl in results:
            print(f'    T3 test_polynomial_exactness [{name} tau={tau_}]: '
                  f'max|resid| = {w:.2e} ({lbl})')
        print(f'      deg={2*k-1} two-sided: head {head:.2e}, interior {interior:.2e}, '
              f'ratio {ratio:.2e} vs exp(-L/ell) = {decay:.2e}')
    assert head > 1e-6, f'deg 2k-1 should NOT be reproduced near the stream start: {head:.2e}'
    assert ratio < 30*decay, (f'deg 2k-1 not suppressed as exp(-L/ell) in the interior: '
                              f'ratio {ratio:.2e} vs decay {decay:.2e}')


# ------------------------------------------------------------- T4. seam freedom

def test_seam_free(rng=None, verbose=True):
    """
    The headline test.  Because J_f and J_b are driven by the input mask alone,
    mask_out and rmin are bit-identical across chunk decompositions
    unconditionally.  The residual is bit-identical only when kappa is held fixed:
    with a per-buffer kappa the decompositions see different buffers, so the residual
    differs by pure rounding.  Asserting bit-identity there would be asserting
    something false.
    """
    rng = _default_rng(rng)
    k, tau, L = K, TAU, LAG
    nout, S_ax = 192, 8
    T = nout + L
    mask, _ = random_mask(S_ax, T, L, rng)
    d = rng.normal(size=(S_ax, T)) + 5.0
    sizes = (nout, nout//2, nout//4, nout//8)

    worst_r = 0.0
    for offs in (False, True):
        ref = None
        for Tc in sizes:
            det = KalmanDetrender(k=k, tau=tau, L=L, chunk_size=Tc,
                                  dtype=np.float64, subtract_offset=offs)
            out = det.detrend_stream(d, mask)
            if ref is None:
                ref = out
                _note_expansion(mask[:, :nout], out[1])
                continue
            for j, nm in ((1, 'mask_out'), (2, 'rmin')):
                assert np.array_equal(out[j], ref[j]), \
                    f'T4: {nm} not bit-identical at chunk_size={Tc} (subtract_offset={offs})'
            if not offs:
                assert np.array_equal(out[0], ref[0]), \
                    f'T4: residual not bit-identical at chunk_size={Tc} with fixed kappa'
            else:
                worst_r = max(worst_r, _maxdiff(out[0], ref[0]))

    if verbose:
        print(f'    T4 test_seam_free: mask_out/rmin bit-identical over '
              f'chunk sizes {sizes}; residual bit-identical at fixed kappa, '
              f'{worst_r:.2e} with per-buffer kappa')
    assert worst_r < 1e-12


# --------------------------------------------------------- T5. vs the dense oracle

def test_vs_brute_force(rng=None, verbose=True):
    rng = _default_rng(rng)
    k, tau, L, Tc = K, TAU, 16, 32
    T = 2*Tc + L
    worst_r, worst_l, name = 0.0, 0.0, ''

    for _ in range(2):
        S_ax = 6
        mask, labels = random_mask(S_ax, T, L, rng)
        d = rng.normal(size=(S_ax, T)) + 2.0
        det = KalmanDetrender(k=k, tau=tau, L=L, chunk_size=Tc, dtype=np.float64)
        r, mk, rmn = det.detrend_stream(d, mask)
        br, bmk, brmn = kalman_brute_force(d, mask, k, tau, L, eps=det.eps)
        _note_expansion(mask[:, :T-L], mk)

        assert np.array_equal(mk, bmk), (
            f'T5: mask_out differs from the oracle on '
            f'{int((mk != bmk).sum())} samples, masks {labels}')
        if mk.any():
            e = _maxdiff(r[mk], br[mk])
            if e > worst_r:
                worst_r, name = e, str(labels)
            worst_l = max(worst_l, _maxdiff(rmn[mk], brmn[mk]))

    if verbose:
        print(f'    T5 test_vs_brute_force: max|r-r_ref| = {worst_r:.2e}, '
              f'max|rmin diff| = {worst_l:.2e}')
    assert worst_r < 1e-9, f'residual vs oracle: {worst_r} ({name})'
    assert worst_l < 1e-9, f'rmin vs oracle: {worst_l}'


# ----------------------------------------------------- T6. steady-state response

def test_kernel_response(rng=None, verbose=True):
    """
    On a full mask, the equivalent kernel at zero lag must approach the closed form
    h[0] = c_k/tau of notes/detrending.tex, section "Time detrending
    algorithm 2: Kalman filter", subsection "Response and numerics".

    This is the strongest check in the file: the expected answer is analytic, so it
    needs neither the dense oracle nor a statistical tolerance, and it exercises the
    forward recursion, the backward recursion and the combine together.

    c_k/tau is the continuum limit, approached from above as O(tau^-2) (measured
    3.0e-2, 7.7e-3, 2.0e-3 at tau = 2, 4, 8), so what is asserted is the rate of
    approach rather than a fixed tolerance.  Both the history behind the output
    sample and the lag ahead of it must be several ell = tau/sin(pi/2k), or the
    forward filter is still spinning up and the deviation is dominated by that
    instead -- hence Tc and L scale with tau below.
    """
    k = K
    c_k = 1.0 / (2*k*np.sin(np.pi/(2*k)))
    devs = []

    for tau in (2.0, 4.0, 8.0):
        # ell = sqrt(2) tau at k=2; 64 samples is 11 ell at tau=8, 23 at tau=4.
        n = int(64 * max(1.0, tau/4.0))
        det = KalmanDetrender(k=k, tau=tau, L=n, chunk_size=n, dtype=np.float64,
                              subtract_offset=False)
        base = np.ones(det.buflen, dtype=bool)
        t_out = n - 1                    # deepest into the chunk, so J_f is in steady state
        kern, mk = impulse_kernel(det, base, t_out)
        assert bool(mk[0]), 'T6: full-mask output was mask-expanded away'
        h0 = float(kern[t_out])
        assert 0.0 <= h0 <= 1.0, f'T6: h[0] outside [0,1]: {h0}'
        devs.append(abs(h0/(c_k/tau) - 1.0))

    if verbose:
        print(f'    T6 test_kernel_response: |h[0]/(c_k/tau) - 1| = '
              + ', '.join(f'{d:.2e}' for d in devs) + ' at tau = 2, 4, 8')
    assert devs[0] > devs[1] > devs[2], \
        f'continuum limit c_k/tau not approached: {devs}'
    # O(tau^-2), so each doubling should gain a factor near 4; 3.0 leaves margin.
    assert devs[0]/devs[1] > 3.0 and devs[1]/devs[2] > 3.0, \
        f'approach to c_k/tau not O(tau^-2): {devs}'
    assert devs[2] < 3e-3, f'h[0] vs c_k/tau at tau=8: {devs[2]}'


# ------------------------------------------------------------ T7. PSD and finite

def test_psd_and_finite(rng=None, verbose=True):
    """
    The recursions must stay symmetric, PSD and finite under every mask in the zoo,
    including all-masked rows and rows with a single valid sample, and the only
    divide must never approach zero: beta >= 1/q always, by positive
    semidefiniteness of J.
    """
    rng = _default_rng(rng)
    k, tau, L = K, TAU, LAG
    worst_asym, min_beta_ratio = 0.0, np.inf
    nstep = 0

    # PSD is exact in exact arithmetic, so the tolerance is pure roundoff and has to
    # be per-dtype: a single loose threshold would hide a real fp64 failure behind
    # fp32's noise.  Measured worst case is ~1e-17 (fp64) and ~1e-8 (fp32).
    worst_neg = {np.dtype(np.float64): 0.0, np.dtype(np.float32): 0.0}
    neg_tol = {np.dtype(np.float64): 1e-13, np.dtype(np.float32): 1e-6}

    for dtype in (np.float64, np.float32):
        mo = StateSpaceModel(k, tau, dtype=dtype)
        T = 96
        mask, _ = random_mask(24, T, L, rng)
        d = (rng.normal(size=(24, T)) + 3.0).astype(dtype)
        m = mask.astype(dtype)

        for name, step in (('forward', forward_step), ('backward', backward_step)):
            J = np.zeros((24, k, k), dtype=dtype)
            eta = np.zeros((24, k), dtype=dtype)
            for u in range(T):
                beta = J[:, k-1, k-1] + mo.invq
                min_beta_ratio = min(min_beta_ratio, float((beta/mo.invq).min()))
                if name == 'forward':
                    _Jm, _em, J, eta = step(J, eta, m[:, u], d[:, u], mo)
                else:
                    J, eta = step(J, eta, m[:, u], d[:, u], mo)
                assert np.all(np.isfinite(J)) and np.all(np.isfinite(eta)), \
                    f'T7: non-finite in {name} at u={u} ({np.dtype(dtype).name})'
                worst_asym = max(worst_asym, _maxdiff(J, np.swapaxes(J, -1, -2)))
                ev = np.linalg.eigvalsh(J.astype(np.float64))
                scale = np.maximum(np.abs(ev).max(axis=-1), 1e-300)
                worst_neg[np.dtype(dtype)] = max(worst_neg[np.dtype(dtype)],
                                                 float((-ev.min(axis=-1)/scale).max()))
                nstep += 1

    # The full path, over the zoo, must also be finite everywhere.
    for dtype in (np.float64, np.float32):
        det = _det(dtype=dtype, chunk_size=64)
        T = 2*64 + LAG
        mask, _ = random_mask(24, T, LAG, rng)
        d = (rng.normal(size=(24, T)) + 3.0).astype(dtype)
        outs = det.detrend_stream(d, mask)
        _note_expansion(mask[:, :T-LAG], outs[1])
        for nm, x in zip(('residual', 'mask_out', 'rmin'), outs):
            assert np.all(np.isfinite(np.asarray(x, dtype=np.float64))), \
                f'T7: non-finite {nm} ({np.dtype(dtype).name})'

    if verbose:
        neg = '  '.join(f'{dt.name} {v:.2e}' for dt, v in worst_neg.items())
        print(f'    T7 test_psd_and_finite: {nstep} recursion steps; symmetry '
              f'{worst_asym:.2e}, min beta/(1/q) = {min_beta_ratio:.4f}')
        print(f'      worst relative negative eigenvalue: {neg}')
    assert worst_asym < 1e-12
    for dt, v in worst_neg.items():
        assert v < neg_tol[dt], f'J went indefinite in {dt.name}: {v}'
    assert min_beta_ratio >= 1.0 - 1e-12, f'beta fell below 1/q: {min_beta_ratio}'


# ----------------------------------------------------- T8. masked data unread

def test_masked_data_unused(rng=None, verbose=True):
    """
    Masked samples must never be read.  Checked by poisoning them and requiring every
    output to be BIT-IDENTICAL -- and, unlike detrending_1d, the carried state too: a
    NaN reaching the state would destroy every subsequent output of that row forever,
    which is the one failure mode this estimator has and the local fit does not.
    """
    rng = _default_rng(rng)
    checked = 0

    for dtype in (np.float32, np.float64):
        for tau, L, Tc, nchunk in ((4.0, 32, 64, 3), (16.0, 64, 128, 2)):
            T = nchunk*Tc + L
            S_ax = 12
            mask, _ = random_mask(S_ax, T, L, rng)
            clean = rng.normal(size=(S_ax, T)) + rng.uniform(0.0, 1e3)
            junk = rng.uniform(-1e10, 1e10, size=(S_ax, T))
            pick = rng.integers(0, 4, size=(S_ax, T))
            junk = np.where(pick == 1, np.inf, junk)
            junk = np.where(pick == 2, -np.inf, junk)
            junk = np.where(pick == 3, np.nan, junk)
            poison = np.where(mask, clean, junk)

            def run(x):
                det = KalmanDetrender(k=K, tau=tau, L=L, chunk_size=Tc, dtype=dtype)
                st = det.initial_state(S_ax)
                outs = None
                for i in range(nchunk):
                    lo = i*Tc
                    o, st = det.detrend_chunk(x.astype(dtype)[:, lo:lo+det.buflen],
                                              mask[:, lo:lo+det.buflen], st)
                    outs = o if outs is None else tuple(
                        np.concatenate([a, b], axis=1) for a, b in zip(outs, o))
                return outs, st

            (oc, sc), (op, sp) = run(clean), run(poison)
            for nm, x, y in zip(('residual', 'mask_out', 'rmin'), oc, op):
                assert np.array_equal(x, y), \
                    f'T8: {nm} changed under poisoning ({np.dtype(dtype).name}, tau={tau})'
                checked += 1
            for nm, x, y in ((f'state.J', sc.J, sp.J), ('state.eta', sc.eta, sp.eta),
                             ('state.kappa', sc.kappa, sp.kappa)):
                assert np.array_equal(x, y), \
                    f'T8: {nm} changed under poisoning ({np.dtype(dtype).name}, tau={tau})'
                checked += 1

    if verbose:
        print(f'    T8 test_masked_data_unused: {checked} arrays bit-identical under '
              f'nan/inf/+-1e10 poisoning (outputs and carried state)')


# ------------------------------------------- T9. dtype agreement (gated: see above)

def test_dtype_agreement(rng=None, tol=1e-3, verbose=True):
    """
    fp32 against fp64 through the whole pipeline, with a constant offset ~ U(0,1e3)
    to exercise the kappa path, plus a long stream to check that the discrepancy does
    NOT grow with time -- which is the direct test of whether the forward filter's
    exponential forgetting really bounds the accumulated state error.

    The two runs use different eps (1e-3 fp32, 1e-6 fp64) so that the test exercises
    samples whose rmin roundoff could move them across the threshold; residuals are
    compared on the intersection of the two masks.

    Also reports max|rmin32 - rmin64|, which is the quantity eps has to clear for the
    masking decision to be well defined.
    """
    rng = _default_rng(rng)
    eps32, eps64 = 1e-3, 1e-6
    worst, worst_name, worst_rmin = 0.0, '', 0.0
    drift = []

    for tau, L, Tc, nchunk, S_ax in ((4.0, 32, 64, 4, 12), (16.0, 128, 256, 3, 4)):
        T = nchunk*Tc + L
        for _ in range(2):
            mask, labels = random_mask(S_ax, T, L, rng)
            offset = rng.uniform(0.0, 1e3)
            d64 = rng.normal(size=(S_ax, T)) + offset
            d32 = d64.astype(np.float32)
            dref = d32.astype(np.float64)          # bit-identical inputs

            r32, m32, q32 = KalmanDetrender(k=K, tau=tau, L=L, chunk_size=Tc,
                                            dtype=np.float32, eps=eps32
                                            ).detrend_stream(d32, mask)
            r64, m64, q64 = KalmanDetrender(k=K, tau=tau, L=L, chunk_size=Tc,
                                            dtype=np.float64, eps=eps64
                                            ).detrend_stream(dref, mask)
            _note_expansion(mask[:, :T-L], m32)
            both = m32 & m64
            worst_rmin = max(worst_rmin, _maxdiff(q32, q64))
            for s in range(S_ax):
                if not both[s].any():
                    continue
                e = _maxdiff(r32[s][both[s]], r64[s][both[s]])
                if e > worst:
                    worst, worst_name = e, f'tau={tau} {labels[s]} offset={offset:.3g}'

    # Does the error grow with time?  Split a long stream into thirds and compare the
    # worst discrepancy in each.
    tau, L, Tc, nchunk, S_ax = 4.0, 32, 64, 24, 8
    T = nchunk*Tc + L
    mask = np.ones((S_ax, T), dtype=bool)
    d64 = rng.normal(size=(S_ax, T)) + rng.uniform(0.0, 1e3)
    d32 = d64.astype(np.float32)
    r32, m32, _ = KalmanDetrender(k=K, tau=tau, L=L, chunk_size=Tc,
                                  dtype=np.float32).detrend_stream(d32, mask)
    r64, m64, _ = KalmanDetrender(k=K, tau=tau, L=L, chunk_size=Tc,
                                  dtype=np.float64).detrend_stream(
                                      d32.astype(np.float64), mask)
    nout = T - L
    for a, b in ((0, nout//3), (nout//3, 2*nout//3), (2*nout//3, nout)):
        drift.append(_maxdiff(r32[:, a:b], r64[:, a:b]))

    if verbose:
        print(f'    T9 test_dtype_agreement: max |r32-r64| = {worst:.3e} sigma ({worst_name})')
        print(f'      max |rmin32-rmin64| = {worst_rmin:.3e}  (eps32={eps32:.0e})')
        print(f'      drift over a {nout}-sample stream, by third: '
              f'{" ".join(f"{x:.2e}" for x in drift)}')
    assert worst < tol, f'T9: max |r32-r64| = {worst:.3e} > {tol:.0e} ({worst_name})'
    assert worst_rmin < 0.1*eps32, \
        f'T9: rmin fp32 noise {worst_rmin:.3e} is not clear of eps={eps32:.0e}'
    assert drift[2] < 10*max(drift[0], 1e-12), \
        f'T9: fp32-vs-fp64 error grows with time: {drift}'


# ----------------------------------------------------------------- entry point

_STAGE1 = [test_model_algebra, test_recursions_vs_dense, test_polynomial_exactness,
           test_seam_free, test_vs_brute_force, test_kernel_response, test_psd_and_finite,
           test_masked_data_unused]

# T1 and T6 have no parameters: T1 enumerates a fixed (k, tau) grid and T6 a fixed sweep, and
# neither draws anything -- so a second call says exactly what the first one did, and
# notes/unit_tests.md point 11 puts them on iteration 0. Skipping them does not perturb the
# shared generator, since neither touches it.
_EXHAUSTIVE = (test_model_algebra, test_kernel_response)


def run_all(verbose=True, rng=None, iteration=0):
    """
    T1-T8 first, then T9.  All share one generator, so printing its entropy makes the
    whole run reproducible: pass np.random.default_rng(<entropy>) back in as 'rng'.

    'iteration' is the index of the caller's test loop; the two parameterless tests run
    only at 0.  See _EXHAUSTIVE.
    """
    rng = _default_rng(rng)
    print(f'  detrending_1d_kalman tests (rng entropy {rng.bit_generator.seed_seq.entropy})')
    for fn in _STAGE1:
        if (iteration == 0) or (fn not in _EXHAUSTIVE):
            fn(rng, verbose=verbose)
    test_dtype_agreement(rng, verbose=verbose)
    print(f'  detrending_1d_kalman tests passed   '
          f'[cumulative mask expansion: {_expansion_str()}]')
