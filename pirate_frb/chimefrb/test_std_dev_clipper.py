"""Randomized unit tests for GpuStdDevClipper, and the tolerance helpers they share with
misc/chimefrb/rfi_std_dev_clipper/.

Dispatched from ``python -m pirate_frb test --cfrb``.
"""

import numpy as np

from .test_wrms import EPS32, MARGIN


# Never bracket stage 2 more tightly than this, relative to sigma. It is the number the old
# code's own clipper tests chose for the same kind of comparison.
SIGMA_FLOOR = 1.0e-4


def _beam_stats(row):
    """(n, vbar, s) of one beam's valid variances, in float64 -- stage 2's own statistic."""

    row = np.asarray(row, dtype=np.float64)
    ok = (row > 0)
    n = int(ok.sum())
    if n < 2:
        return (n, 0.0, 0.0)
    vbar = float(row[ok].mean())
    s = float(np.sqrt(np.mean((row[ok] - vbar)**2)))
    return (n, vbar, s)


def ill_conditioned(v):
    """Per beam: True if at least two variances are valid and all valid ones are EQUAL.

    That is the corner of plans/chimefrb_std_dev_clipper.md 2.4 where the answer is decided
    by float32 roundoff (s = 0 exactly, but a rounded mean usually is not), so no bracket can
    make a comparison meaningful. The randomized tests must not produce it.
    """

    v = np.atleast_2d(np.asarray(v, dtype=np.float64))
    out = []
    for row in v:
        (n, _, s) = _beam_stats(row)
        out.append((n >= 2) and (s == 0.0))
    return np.array(out)


def stage2_bracket(v, sigma):
    """d_A: how far apart two implementations' stage-2 thresholds can legitimately be,
    GIVEN THE SAME stage-1 variances 'v' (B, nrows). Returned relative to sigma.

    Stage 2 compares |v_i - vbar| against sigma*s, where vbar and s are sums over the beam
    that each implementation rounds differently (float32 in a block reduction or a sequential
    loop, float64 in numpy). Each is off by float32 roundoff times its own scale, so the
    threshold moves by about eps*s and the compared quantity by about eps*vbar. Relative to
    sigma*s that is eps*(1 + (vbar + s)/(sigma*s)) -- and the second term is NOT small,
    because in noise the variances are all similar: s/vbar ~ sqrt(2/L), about 0.02 at
    L = 4096, so vbar/(sigma*s) is around 15 and a flat 1e-4 would be marginal. The budget
    uses test_wrms.py's MARGIN and EPS32, so the port has one error model.

    Beams with fewer than two valid rows have no threshold (their outcome is an exact count),
    and ill-conditioned beams are skipped; callers must exclude those (ill_conditioned()).
    """

    d = SIGMA_FLOOR
    for row in np.atleast_2d(v):
        (n, vbar, s) = _beam_stats(row)
        if (n < 2) or (s == 0.0):
            continue
        d = max(d, MARGIN * EPS32 * (1.0 + (vbar + s) / (sigma * s)))
    return d


def end_to_end_bracket(v, mean, L, two_pass, sigma):
    """d_B: d_A plus the effect of the stage-1 variances THEMSELVES differing, float32 against
    float64. 'v' and 'mean' are the float64 reference's stage-1 outputs, (B, nrows).

    test_wrms.py budgets the rms error as eps_r = MARGIN*EPS32*rms, plus
    MARGIN*EPS32*sqrt(L)*|mean| on the single-pass path, where var = <I^2> - mean^2 cancels.
    As a variance that is e_i = 2*rms_i*eps_r_i. In stage 2, |v_i - vbar| then moves by up to
    e_i + mean(e) and s by up to about max(e), so relative to sigma*s the bracket grows by
    (max(e) + mean(e))/(sigma*s) + 2*max(e)/s. On single-pass draws with a mean far from zero
    this can reach several percent: the end-to-end check is weak there, and the check that
    takes the variances from the GPU carries the weight.
    """

    v = np.atleast_2d(np.asarray(v, dtype=np.float64))
    mean = np.atleast_2d(np.asarray(mean, dtype=np.float64))

    d = stage2_bracket(v, sigma)
    for (row, mrow) in zip(v, mean):
        ok = (row > 0)
        (n, _, s) = _beam_stats(row)
        if (n < 2) or (s == 0.0):
            continue
        rms = np.sqrt(row[ok])
        eps_r = MARGIN * EPS32 * rms
        if not two_pass:
            eps_r = eps_r + MARGIN * EPS32 * np.sqrt(L) * np.abs(mrow[ok])
        e = 2.0 * rms * eps_r
        d = max(d, stage2_bracket(row, sigma)
                + (e.max() + e.mean()) / (sigma * s) + 2.0 * e.max() / s)
    return d


def keep_bracket(v, sigma, d):
    """The survivors of clip_1d() at sigma*(1-d) and sigma*(1+d), as 0/1 arrays.

    Weights and variances are only ever zeroed, so 'survives at the smaller sigma' implies
    'survives at the larger' and the two bound any correct implementation's survivors.
    """

    from .ReferenceStdDevClipper import clip_1d

    v = np.asarray(v, dtype=np.float64)
    lo = (clip_1d(v, sigma * (1.0 - d)) != 0)
    hi = (clip_1d(v, sigma * (1.0 + d)) != 0)
    return (lo, hi)


# -------------------------------------------------------------------------------------------------
#
# The randomized GPU test.
#
# THE VARIANCES COME FROM THE GPU, AND WHAT FOLLOWS THEM FROM NUMPY -- the intensity clipper's
# central move, from rf_kernels' own clipper tests. Rebuilding steps 1-2 from
# GpuWiDownsampler and GpuWrms (deliberately not through GpuClipperBase, so that the check is
# independent of the plumbing it checks) gives the exact float32 variances the clipper used;
# numpy's clip_1d() and std_dev_apply() on those, bracketed on sigma, must sandwich the
# kernel's weights. That tests stage 2 and the apply with nothing upstream in the way.
#
# The end-to-end comparison against the float64 reference cannot use the intensity
# clipper's form. There, a smaller sigma and a larger eps_multiplier both clip more, so two
# references sandwich the kernel. Here the epsilon bracket is NOT monotone: a stage-1 row
# near its cutoff changes the POPULATION stage 2 sees -- one near-zero variance counted as
# valid drags vbar down and inflates s, which spares rows that would otherwise be clipped --
# so admitting fewer rows in stage 1 can clip fewer in stage 2. Instead, the stage-1
# decision is bracketed on its own (exactly as test_wrms.py does), the GPU's decision is then
# imposed on the reference, and only stage 2 is bracketed on sigma, where it is monotone.
#
# Nothing here compares two GPU runs on a sharp decision; see test_wrms.py's docstring.
# Bitwise GPU-vs-GPU checks are used only where nothing sharp differs between the runs.

from .ReferenceStdDevClipper import clip_1d as _clip_1d, std_dev_apply
from . import (ClipperAxis, GpuStdDevClipper, GpuWiDownsampler, GpuWrms,
               ReferenceStdDevClipper)
from ..utils import atomic_print


WARP_COUNTS = [4, 8, 16, 32]

# The production chain's std_dev_clippers: 36 AXIS_TIME and 24 AXIS_FREQ, all at (1,1).
PRODUCTION_CONFIGS = [(ClipperAxis.TIME, 1, 1), (ClipperAxis.FREQ, 1, 1)]


def random_config(rng):
    """Draw (axis, Df, Dt, sigma, two_pass, warps_per_block)."""

    if rng.uniform() < 0.6:
        (axis, Df, Dt) = PRODUCTION_CONFIGS[int(rng.integers(len(PRODUCTION_CONFIGS)))]
    else:
        axis = [ClipperAxis.TIME, ClipperAxis.FREQ][int(rng.integers(2))]
        # Nothing requires powers of two.
        Df = int(rng.choice([1, 1, 2, 3, 4]))
        Dt = int(rng.choice([1, 1, 2, 3, 8, 16]))

    # Mostly near the production 3. Sometimes low (1.5-2.5), where stage 2 clips a large
    # fraction of rows and the apply is exercised hard; sometimes high (3.5-5).
    u = rng.uniform()
    if u < 0.6:
        sigma = 3.0 * float(rng.uniform(0.85, 1.15))
    elif u < 0.8:
        sigma = float(rng.uniform(1.5, 2.5))
    else:
        sigma = float(rng.uniform(3.5, 5.0))

    two_pass = bool(rng.integers(2))
    warps = int(rng.choice(WARP_COUNTS))
    return (axis, Df, Dt, sigma, two_pass, warps)


def random_geometry(rng, axis, Df, Dt):
    """Draw (B, F, nt_chunk) with F % (32*Df) == 0 and nt_chunk % (32*Dt) == 0.

    Stage 2 pools one beam's rows -- F_ds of them for TIME, T_ds for FREQ -- and by
    Samuelson's inequality it can clip nothing unless nrows > sigma^2 + 1. The 32-rule already
    guarantees nrows >= 32, enough for sigma < 5.5; here nrows is usually in the tens to a few
    hundred, so the clip has room to fire.
    """

    B = int(rng.integers(1, 4))
    rows = int(rng.choice([1, 2, 2, 4, 4, 8]))      # nrows = 32*rows
    other = int(rng.choice([1, 2, 4]))              # the statistic's L = 32*other

    (a, b) = (rows, other) if (axis == ClipperAxis.TIME) else (other, rows)
    return (B, 32*a*Df, 32*b*Dt)


def random_arrays(rng, B, F, T, axis, Df, Dt):
    """Random (intensity, weights), float32, shaped (B, F, T).

    Designed around stage 2, which sees VARIANCES:

      - the injected RFI is rows with the wrong NOISE LEVEL (noise scaled by 3-10), since a
        bright but steady row barely moves a variance. Kept to a few percent of rows: they
        inflate the very spread they are measured against;
      - a large random offset, which makes the single-pass variance cancel and the
        epsilon cutoffs matter;
      - a few percent of rows constant, fully masked, or nearly constant at a large mean --
        the three ways to make a stage-1 variance invalid;
      - a few percent of beams with at most one valid row, so the whole-beam branch runs.

    Deliberately NOT produced: every valid variance in a beam bit-identical (ill-conditioned,
    plans/chimefrb_std_dev_clipper.md 2.4), and rows placed at the stage-2 threshold.
    """

    offset = rng.uniform(-1.0e3, 1.0e3)
    scale = rng.uniform(0.5, 20.0)
    intensity = rng.normal(offset, scale, size=(B, F, T))

    p = np.clip(rng.uniform(-0.1, 1.1), 0.0, 1.0)
    weights = (rng.uniform(size=(B, F, T)) < p) * rng.uniform(0.5, 1.5, size=(B, F, T))

    # Rows, in the reduced sense: a block of Df channels (TIME) or Dt time samples (FREQ).
    time_axis = (axis == ClipperAxis.TIME)
    nrows = (F // Df) if time_axis else (T // Dt)
    D = Df if time_axis else Dt

    def rowslice(r):
        return (slice(None), slice(r*D, (r+1)*D)) if time_axis else (slice(None), slice(None), slice(r*D, (r+1)*D))

    for b in range(B):
        def sel(r):
            s = rowslice(r)
            return (b,) + s[1:]

        for r in np.flatnonzero(rng.uniform(size=nrows) < 0.03):
            blk = intensity[sel(r)]
            intensity[sel(r)] = offset + (blk - offset) * rng.uniform(3.0, 10.0)
        for r in np.flatnonzero(rng.uniform(size=nrows) < 0.03):
            intensity[sel(r)] = offset                                        # variance zero
        for r in np.flatnonzero(rng.uniform(size=nrows) < 0.03):
            weights[sel(r)] = 0.0                                             # no weight
        for r in np.flatnonzero(rng.uniform(size=nrows) < 0.03):
            blk = intensity[sel(r)]
            intensity[sel(r)] = offset * (1.0 + 1.0e-6 * rng.normal(size=blk.shape))   # on the cutoff

        if rng.uniform() < 0.05:
            keep = int(rng.integers(nrows))
            w = np.zeros_like(weights[b])
            s = rowslice(keep)[1:]
            w[s] = weights[b][s]
            weights[b] = w

    return (intensity.astype(np.float32), weights.astype(np.float32))


def _run_gpu(cp, sd, in_i, in_w):
    """Run one GpuStdDevClipper on numpy inputs; returns the clipped weights."""

    g_i = cp.asarray(in_i)
    g_w = cp.asarray(in_w)
    sd.launch(g_i, g_w)
    cp.cuda.get_current_stream().synchronize()

    assert np.array_equal(cp.asnumpy(g_i), in_i), 'GpuStdDevClipper modified the intensity'
    return cp.asnumpy(g_w)


def _run_statistic(cp, sd, in_i, in_w):
    """Steps 1-2 rebuilt from GpuWiDownsampler and GpuWrms: the float32 variances the
    clipper used, shaped (B, nrows). Bit-identical to the clipper's, since these are the
    same kernels with the same parameters on the same input."""

    g_i = cp.asarray(in_i)
    g_w = cp.asarray(in_w)

    if (sd.Df, sd.Dt) != (1, 1):
        ds_i = cp.empty((sd.B, sd.F_ds, sd.T_ds), dtype=cp.float32)
        ds_w = cp.empty((sd.B, sd.F_ds, sd.T_ds), dtype=cp.float32)
        GpuWiDownsampler(sd.Df, sd.Dt, False).launch(ds_i, ds_w, g_i, g_w)
    else:
        (ds_i, ds_w) = (g_i, g_w)

    if sd.axis == ClipperAxis.FREQ:
        t_i = cp.empty((sd.B, sd.T_ds, sd.F_ds), dtype=cp.float32)
        t_w = cp.empty((sd.B, sd.T_ds, sd.F_ds), dtype=cp.float32)
        GpuWiDownsampler(1, 1, True).launch(t_i, t_w, ds_i, ds_w)
        (ds_i, ds_w) = (t_i, t_w)

    mean = cp.empty(sd.wrms_R, dtype=cp.float32)
    var = cp.empty(sd.wrms_R, dtype=cp.float32)
    GpuWrms(sd.wrms_L, 1, 0.0, sd.two_pass).launch(
        mean, var, ds_i.reshape(sd.wrms_R, sd.wrms_L), ds_w.reshape(sd.wrms_R, sd.wrms_L))

    cp.cuda.get_current_stream().synchronize()
    return cp.asnumpy(var).reshape(sd.B, -1)


def _sandwich(label, got, lo, hi):
    bad = (got < lo) | (got > hi)
    if bad.any():
        excursion = float(np.max(np.maximum(lo - got, got - hi)))
        raise AssertionError(f'{label}: {int(bad.sum())} of {got.size} elements outside the '
                             f'bracket, worst excursion {excursion:.3g}')


def _weights_bracket(W, klo, khi, axis, Df, Dt):
    """Weights after the apply, from the two survivor sets of keep_bracket()."""
    return (std_dev_apply(W, klo, axis, Df, Dt), std_dev_apply(W, khi, axis, Df, Dt))


def test_std_dev_clipper(iteration=0, rng=None, verbose=False):
    """One randomized comparison of GpuStdDevClipper against its numpy reference."""

    try:
        import cupy as cp
    except ImportError:
        if verbose:
            atomic_print('    test_std_dev_clipper: cupy not available, skipped')
        return

    rng = np.random.default_rng() if (rng is None) else rng

    (axis, Df, Dt, sigma, two_pass, warps) = random_config(rng)
    (B, F, T) = random_geometry(rng, axis, Df, Dt)
    (in_i, in_w) = random_arrays(rng, B, F, T, axis, Df, Dt)
    W64 = in_w.astype(np.float64)

    sd = GpuStdDevClipper(B, F, T, axis, sigma, Df, Dt, two_pass, warps)
    w_gpu = _run_gpu(cp, sd, in_i, in_w)
    v_gpu = _run_statistic(cp, sd, in_i, in_w)

    if ill_conditioned(v_gpu).any():
        # The one corner no bracket can cover (plans/chimefrb_std_dev_clipper.md 2.4).
        # random_arrays() is designed never to produce it, and coverage reports how often it
        # does; skip rather than fail, since the kernel is not wrong there.
        atomic_print('    test_std_dev_clipper: ill-conditioned draw (all valid variances'
                     ' equal in some beam), skipped')
        return

    # Check 1: the variances from the GPU, stage 2 and the apply from numpy.
    d_A = stage2_bracket(v_gpu, sigma)
    (klo, khi) = keep_bracket(v_gpu, sigma, d_A)
    (wlo, whi) = _weights_bracket(W64, klo, khi, axis, Df, Dt)
    _sandwich('stage 2 + apply, on the GPU variances', w_gpu, wlo, whi)

    # Check 2: end to end against the float64 reference, conditioned on the GPU's stage-1
    # decisions. First bracket that decision, then impose it on the reference -- using the
    # permissive reference's values, so that every row the GPU calls valid has one (at
    # niter=1 eps_multiplier only decides whether a variance is zeroed, never its value).
    I64 = in_i.astype(np.float64)
    (_, v_c) = ReferenceStdDevClipper(axis, sigma, Df, Dt, two_pass, eps_multiplier=1.5).variances(I64, W64)
    (m_p, v_p) = ReferenceStdDevClipper(axis, sigma, Df, Dt, two_pass, eps_multiplier=0.5).variances(I64, W64)

    valid = (v_gpu > 0)
    assert not (valid & ~(v_p > 0)).any(), 'stage 1: GPU calls valid a row the permissive reference rejects'
    assert not (~valid & (v_c > 0)).any(), 'stage 1: GPU rejects a row the conservative reference keeps'

    v_cond = np.where(valid, v_p, 0.0)
    d_B = end_to_end_bracket(v_cond, m_p, sd.wrms_L, two_pass, sigma)
    (klo, khi) = keep_bracket(v_cond, sigma, d_B)
    (wlo, whi) = _weights_bracket(W64, klo, khi, axis, Df, Dt)
    _sandwich('end-to-end, conditioned on the GPU stage-1 decisions', w_gpu, wlo, whi)

    # Structural check 1: AXIS_FREQ equals AXIS_TIME on the transposed input, BITWISE, at
    # (1,1): the FREQ path transposes and reduces rows, then stage 2 sees the same values in
    # the same order. That needs GpuWiDownsampler's (1,1) transpose to copy the intensity
    # through exactly; (w*i)/w would move the variances by roundoff, and a stage-2 decision
    # within roundoff of its threshold would then flip.
    if (Df, Dt) == (1, 1) and (axis == ClipperAxis.FREQ):
        sd_t = GpuStdDevClipper(B, T, F, ClipperAxis.TIME, sigma, 1, 1, two_pass, warps)
        w_t = _run_gpu(cp, sd_t, np.ascontiguousarray(np.swapaxes(in_i, 1, 2)),
                       np.ascontiguousarray(np.swapaxes(in_w, 1, 2)))
        assert np.array_equal(np.swapaxes(w_t, 1, 2), w_gpu), \
            'AXIS_FREQ disagrees with AXIS_TIME on the transposed input'

    # Structural check 2: the mask is constant along the reduced axis. Every row -- a block
    # of Df channels (TIME) or Dt time samples (FREQ) -- is either entirely zero or
    # untouched. Exact, and the direct test of sd_apply_kernel's tile loop.
    if axis == ClipperAxis.TIME:
        wg = w_gpu.reshape(B, F // Df, Df * T)
        wi = in_w.reshape(B, F // Df, Df * T)
    else:
        wg = np.swapaxes(w_gpu.reshape(B, F, T // Dt, Dt), 1, 2).reshape(B, T // Dt, F * Dt)
        wi = np.swapaxes(in_w.reshape(B, F, T // Dt, Dt), 1, 2).reshape(B, T // Dt, F * Dt)
    row_zero = (wg == 0).all(axis=2)
    row_same = (wg == wi).all(axis=2)
    assert (row_zero | row_same).all(), 'a row was partially masked; the clip must zero rows whole'

    # Structural check 3: a whole beam is zeroed iff at most one of its rows has a usable
    # variance. One direction is the n < 2 branch. The other is Samuelson's inequality: some
    # |v - vbar| is always <= s, so for sigma > 1 at least one valid row survives, and a
    # valid row has nonzero weight.
    nvalid = valid.sum(axis=1)
    beam_zero = (w_gpu.reshape(B, -1) == 0).all(axis=1)
    assert np.array_equal(beam_zero, nvalid <= 1), 'a beam was zeroed whole iff n <= 1 failed'

    # Structural check 4: warps_per_block is a performance knob. Bitwise, because it touches
    # only sd_apply, which involves no statistic.
    for w2 in WARP_COUNTS:
        if w2 != warps:
            sd2 = GpuStdDevClipper(B, F, T, axis, sigma, Df, Dt, two_pass, w2)
            assert np.array_equal(_run_gpu(cp, sd2, in_i, in_w), w_gpu), \
                f'warps_per_block {warps} vs {w2}: different result'

    # Structural check 5: determinism.
    assert np.array_equal(_run_gpu(cp, sd, in_i, in_w), w_gpu), 'GpuStdDevClipper is not deterministic'

    # Structural check 6: weights only ever decrease, and a zero weight stays zero.
    assert np.all(w_gpu <= in_w), 'a weight increased'
    assert np.all(w_gpu[in_w == 0] == 0), 'a zero weight became nonzero'

    # Structural check 7: masked data is unused. Poison with large FINITE values, not NaN.
    poisoned = np.where(in_w == 0, rng.uniform(-1e6, 1e6, size=in_i.shape), in_i).astype(np.float32)
    assert np.array_equal(_run_gpu(cp, sd, poisoned, in_w), w_gpu), 'masked intensity changed the result'

    # Structural check 8: B is a spectator.
    if B > 1:
        sd1 = GpuStdDevClipper(1, F, T, axis, sigma, Df, Dt, two_pass, warps)
        for b in range(B):
            assert np.array_equal(_run_gpu(cp, sd1, in_i[b:b+1], in_w[b:b+1])[0], w_gpu[b]), \
                f'beam {b} depends on the other beams'

    # Structural check 9: the reference's T = N*nt_chunk is N independent calls. Checked
    # reference against reference, which is exact: comparing a chunked reference against
    # separate GPU calls would need the stage-1 conditioning of check 2 per chunk, and the
    # GPU side of each chunk is already covered above. A fraction of draws, for cost.
    if rng.uniform() < 0.25:
        (j_i, j_w) = random_arrays(rng, B, F, T, axis, Df, Dt)
        big_i = np.concatenate([in_i, j_i], axis=2).astype(np.float64)
        big_w = np.concatenate([in_w, j_w], axis=2).astype(np.float64)
        ref = ReferenceStdDevClipper(axis, sigma, Df, Dt, two_pass, nt_chunk=T)
        piecewise = np.concatenate([ReferenceStdDevClipper(axis, sigma, Df, Dt, two_pass).apply(x, w)
                                    for (x, w) in ((I64, W64), (j_i, j_w))], axis=2)
        assert np.array_equal(ref.apply(big_i, big_w), piecewise), \
            "the reference's T = N*nt_chunk is not N independent calls"

    if verbose:
        nkill = int(np.sum(valid & (_clip_1d(v_gpu.astype(np.float64), sigma) == 0)))
        atomic_print(f'    test_std_dev_clipper(axis={sd.axis}, (Df,Dt)=({Df},{Dt}),'
                     f' (B,F,T)=({B},{F},{T}), sigma={sigma:.2f}, two_pass={two_pass},'
                     f' warps={warps}): {int(valid.sum())} valid rows, stage 2 clips {nkill};'
                     f' d_A={d_A:.2g}, d_B={d_B:.2g}')
