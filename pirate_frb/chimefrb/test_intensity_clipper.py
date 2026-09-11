"""Randomized unit tests for GpuIntensityClipper.

Dispatched from ``python -m pirate_frb test --cfrb``.

THE STATISTIC COMES FROM THE GPU, THE CLIP COMES FROM NUMPY. That is the central move
here, and it is borrowed from the old code's own unit test
(rf_kernels/test-intensity-clipper.cpp) rather than invented. That test does NOT compare
against an independent end-to-end reference: it runs the production weighted_mean_rms
kernel -- the same one the clipper uses internally -- and references only the final clip,
twice, with sigma perturbed either way.

The reason is that the production clipper runs nine rounds of refinement. After nine
rounds, a single sample that lands on the other side of a threshold in float32 than in
float64 has moved everything downstream: comparing two full refinement chains is a coin
flip, not a test (test_wrms.py's docstring has the measured numbers). Feeding both sides
the same (mean, var) removes the amplification entirely, and the statistic is not left
untested -- test_wrms.py tests it, with its own induction.

Rebuilding the clipper's internal pipeline from public pieces is also a real test in its
own right: it asserts that the class wires GpuWiDownsampler -> GpuWiDownsampler(transpose)
-> GpuWrms together the way its header says it does, which is where an axis or transpose
bug would live.

The end-to-end comparison against the float64 ReferenceIntensityClipper is confined to
niter=1, where there are no refinements and nothing can amplify. That is what checks the
downsample, the statistic and the axis reshaping as a unit rather than only as a claim.

WHAT THIS FILE CANNOT ESTABLISH: the numpy reference and the CUDA kernel were written
from one reading of the old code, so a shared misreading would pass everything here. That
is what misc/chimefrb/rfi_intensity_clipper/ is for.
"""

import numpy as np

from . import (AXIS_FREQ, AXIS_TIME, AXIS_NONE, ClipperAxis, GpuIntensityClipper,
               GpuWiDownsampler, GpuWrms, ReferenceIntensityClipper, intensity_clip)
from .test_wrms import EPS32, MARGIN
from ..utils import atomic_print
from .testutils import default_rng as _default_rng


WARP_COUNTS = [4, 8, 16, 32]

# The four distinct intensity_clippers in the production RFI config. Note that iter_sigma
# is 3, not 5, for the (2,16) pair: the two thresholds are different numbers, and a test
# that always drew them equal could not see a kernel that confused them.
PRODUCTION_CONFIGS = [
    (ClipperAxis.FREQ, 1, 1),
    (ClipperAxis.TIME, 1, 1),
    (ClipperAxis.NONE, 2, 16),
    (ClipperAxis.FREQ, 2, 16),
]

# The bracket on the final clip, in relative units of sigma: how much roundoff we accept
# in |I_ds - mean| vs sigma*rms before calling a disagreement real. This is the number the
# old code's unit test chose for the same comparison, and it is the right one when both
# sides are handed the SAME (mean, var), which is check 1 below. The end-to-end comparison
# needs a wider and data-dependent bracket -- see _sigma_bracket().
SIGMA_BRACKET = 1.0e-4

# The bracket on the variance-validity cutoffs. A row whose variance is rejected loses ALL
# of its weights, so the decision has a large blast radius and cannot be left to a
# tolerance. 1.5 rejects more variances (clips more), 0.5 fewer.
EPS_HI, EPS_LO = 1.5, 0.5

# GpuWrms switches from its shared-memory kernel to its global-memory one when a row stops
# fitting in shared memory. Asked for rather than recomputed, so that the test cannot drift
# from the kernel's own idea of where the boundary is.
L_SHARED_MAX = GpuWrms.max_shared_L()


def random_config(rng):
    """Draw (axis, Df, Dt, niter, sigma, iter_sigma, two_pass, warps_per_block)."""

    if rng.uniform() < 0.5:
        (axis, Df, Dt) = PRODUCTION_CONFIGS[int(rng.integers(len(PRODUCTION_CONFIGS)))]
    else:
        axis = [ClipperAxis.FREQ, ClipperAxis.TIME, ClipperAxis.NONE][int(rng.integers(3))]
        # Nothing requires Df or Dt to be a power of two, so draw some that are not.
        Df = int(rng.choice([1, 1, 2, 3, 4]))
        Dt = int(rng.choice([1, 1, 2, 3, 8, 16]))

    # niter=1 is the base case, where the end-to-end comparison runs; 9 is production. The
    # small values in between keep the (Df,Dt) and axis plumbing honest at more than one
    # refinement count without costing much.
    niter = int(rng.choice([1, 1, 2, 3, 9]))

    # Drawn independently, so that sigma < iter_sigma and sigma > iter_sigma both occur.
    sigma = float(rng.uniform(2.0, 6.0))
    iter_sigma = float(rng.uniform(2.0, 6.0))

    two_pass = bool(rng.integers(2))
    warps = int(rng.choice(WARP_COUNTS))

    return (axis, Df, Dt, niter, sigma, iter_sigma, two_pass, warps)


def random_geometry(rng, axis, Df, Dt):
    """Draw (B, F, nt_chunk) satisfying F % (32*Df) == 0 and nt_chunk % (32*Dt) == 0.

    Writing F_ds = 32*a and T_ds = 32*b, the statistic's row length L is 32*b for
    AXIS_TIME, 32*a for AXIS_FREQ, and 1024*a*b for AXIS_NONE. Only the last can reach
    GpuWrms's global-memory kernel at any geometry small enough to test, so AXIS_NONE
    draws sometimes put a*b either side of the threshold on purpose: a shared-memory
    budget that forgets an allocation fails to launch only in a narrow band, which a
    uniform draw hides for a long time (that is not hypothetical -- it is what happened
    to GpuWrms).
    """

    B = int(rng.integers(1, 4))

    if (axis == ClipperAxis.NONE) and (rng.uniform() < 0.25):
        ab0 = max(1, L_SHARED_MAX // 1024)      # largest a*b that still fits on-chip
        ab = int(rng.integers(max(1, ab0 - 1), ab0 + 3))
        (a, b) = (1, ab) if rng.integers(2) else (ab, 1)
    else:
        a = int(rng.integers(1, 5))
        b = int(rng.integers(1, 9))

    return (B, 32*a*Df, 32*b*Dt)


def random_arrays(rng, B, F, T):
    """Random (intensity, weights), float32, shaped (B, F, T).

    Designed around the code paths rather than around realism:

      - a large per-beam offset, because the eps_2*mean variance cutoff is inert when the
        mean is near zero, and because it is what makes the single-pass variance cancel;
      - whole bad channels and whole bad time samples, which are the two RFI shapes
        AXIS_TIME and AXIS_FREQ respectively exist to catch, plus isolated spikes for
        AXIS_NONE. Without them a 2-6 sigma clip on clean Gaussian data masks nothing and
        the whole test is a comparison of two untouched weight arrays;
      - a few percent of channels fully masked, constant, or near-constant at a large
        mean, which are the three ways to land on a variance-validity cutoff (and a
        rejected variance zeroes an entire row).

    Deliberately NOT included: samples placed AT a threshold. Sampling the ambiguous
    region on purpose would make the brackets straddle at a controlled rate rather than a
    negligible one, which is worse, not better.
    """

    offset = rng.uniform(-1.0e3, 1.0e3)
    scale = rng.uniform(0.5, 20.0)
    intensity = rng.normal(offset, scale, size=(B, F, T))

    p = np.clip(rng.uniform(-0.1, 1.1), 0.0, 1.0)
    weights = (rng.uniform(size=(B, F, T)) < p) * rng.uniform(0.5, 1.5, size=(B, F, T))

    for b in range(B):
        # The bad rows are kept to about 1.5% and the spikes are made much larger than
        # them, for a reason specific to AXIS_NONE: its statistic is the whole plane, so
        # bad rows inflate the very rms the clip is measured against. A fraction f of
        # samples at A sigma raises the plane's rms by sqrt(1 + f*A^2), and at f = 3% and
        # A = 20 that factor is 3.5 -- enough that a 20-sigma sample is no longer an
        # outlier by the plane's own standard, and the clip fires on nothing.
        for f in rng.choice(F, size=max(1, F // 64), replace=False):
            intensity[b, f, :] = offset + 20.0 * scale
        for t in rng.choice(T, size=max(1, T // 64), replace=False):
            intensity[b, :, t] = offset + 20.0 * scale
        for _ in range(max(1, (F*T) // 500)):
            intensity[b, rng.integers(F), rng.integers(T)] = \
                offset + 40.0 * scale * rng.choice([-1.0, 1.0])

        # Degenerate channels, a few percent each.
        for f in np.flatnonzero(rng.uniform(size=F) < 0.05):
            intensity[b, f, :] = offset                     # variance exactly zero
        for f in np.flatnonzero(rng.uniform(size=F) < 0.05):
            weights[b, f, :] = 0.0                          # no weight at all
        for f in np.flatnonzero(rng.uniform(size=F) < 0.05):
            intensity[b, f, :] = offset * (1.0 + 1.0e-6 * rng.normal(size=T))  # on the cutoff

    return (intensity.astype(np.float32), weights.astype(np.float32))


def _run_gpu(cp, ic, in_i, in_w):
    """Run one GpuIntensityClipper on numpy inputs; returns the clipped weights."""

    g_i = cp.asarray(in_i)
    g_w = cp.asarray(in_w)

    ic.launch(g_i, g_w)
    cp.cuda.get_current_stream().synchronize()

    out = cp.asnumpy(g_w)
    assert np.array_equal(cp.asnumpy(g_i), in_i), 'GpuIntensityClipper modified the intensity'
    return out


def _run_pipeline(cp, ic, in_i, in_w):
    """Rebuild the clipper's internal pipeline from public pieces.

    Returns (i_ds, mean, var) as numpy float32: the downsampled intensity the final clip
    compares against, and the statistic it thresholds with. These are bit-identical to
    what the clipper computed internally, because they are the same kernels with the same
    parameters on the same input -- which is what makes the comparison in test_intensity_clipper()
    free of the refinement amplification described in this file's docstring.
    """

    g_i = cp.asarray(in_i)
    g_w = cp.asarray(in_w)

    if (ic.Df, ic.Dt) != (1, 1):
        ds_i = cp.empty((ic.B, ic.F_ds, ic.T_ds), dtype=cp.float32)
        ds_w = cp.empty((ic.B, ic.F_ds, ic.T_ds), dtype=cp.float32)
        GpuWiDownsampler(ic.Df, ic.Dt, False).launch(ds_i, ds_w, g_i, g_w)
    else:
        (ds_i, ds_w) = (g_i, g_w)

    if ic.axis == ClipperAxis.FREQ:
        t_i = cp.empty((ic.B, ic.T_ds, ic.F_ds), dtype=cp.float32)
        t_w = cp.empty((ic.B, ic.T_ds, ic.F_ds), dtype=cp.float32)
        GpuWiDownsampler(1, 1, True).launch(t_i, t_w, ds_i, ds_w)
        (st_i, st_w) = (t_i, t_w)
    else:
        (st_i, st_w) = (ds_i, ds_w)

    mean = cp.empty(ic.wrms_R, dtype=cp.float32)
    var = cp.empty(ic.wrms_R, dtype=cp.float32)

    wrms = GpuWrms(ic.wrms_L, ic.niter, ic.iter_sigma, ic.two_pass)
    wrms.launch(mean, var, st_i.reshape(ic.wrms_R, ic.wrms_L),
                st_w.reshape(ic.wrms_R, ic.wrms_L))

    cp.cuda.get_current_stream().synchronize()
    return (cp.asnumpy(ds_i), cp.asnumpy(mean), cp.asnumpy(var))


def _sandwich(label, got, lo, hi):
    """Require lo <= got <= hi elementwise, and say how far outside anything fell."""

    bad = (got < lo) | (got > hi)
    if bad.any():
        excursion = float(np.max(np.maximum(lo - got, got - hi)))
        raise AssertionError(f'{label}: {int(bad.sum())} of {got.size} weights outside the '
                             f'bracket, worst excursion {excursion:.3g}')


def _sigma_bracket(L, two_pass, sigma, mean, var):
    """How far apart the float32 and float64 clip thresholds can legitimately be.

    Returned as a relative perturbation of sigma, for the end-to-end comparison. A fixed
    1e-4 is NOT good enough there, and the reason is the single-pass variance.

    The mask is |I - mean| < sigma*sqrt(var), and the two sides compute (mean, var) in
    different precisions. With two_pass the disagreement is plain float32 roundoff. Without
    it, var = <I^2> - mean^2 cancels log10(mean^2/var) of float32's seven digits, and the
    error grows as sqrt(L) with the accumulated sum -- so a row with a mean 60x its rms
    has a threshold good to about three digits, not seven, and a 1e-4 bracket is an order
    of magnitude too tight. (Measured: it fails on roughly one draw in 250, always on the
    single-pass path, always with |mean|/rms in the tens.)

    The budget is test_wrms's, imported rather than restated so that the port has one
    error model rather than two. Converting it to a sigma perturbation: the threshold moves
    by sigma*eps_r, and the mean moves the compared quantity by eps_m, so the relative
    perturbation that covers both is eps_r/rms + eps_m/(sigma*rms). The max over rows is
    taken, which is bounded in practice because the variance-validity cutoff itself keeps
    |mean|/rms below about 1/sqrt(eps_3) = 92.
    """

    ok = (var > 0)
    if not ok.any():
        return SIGMA_BRACKET

    rms = np.sqrt(var[ok])
    mn = np.abs(mean[ok])

    eps_r = MARGIN * EPS32 * rms
    if not two_pass:
        eps_r = eps_r + MARGIN * EPS32 * np.sqrt(L) * mn
    eps_m = MARGIN * EPS32 * (mn + rms)

    rel = eps_r / rms + eps_m / (sigma * rms)
    return max(SIGMA_BRACKET, float(rel.max()))


def _reference_bracket(axis, sigma, Df, Dt, niter, iter_sigma, two_pass, I, W,
                       sigma_bracket, nt_chunk=None):
    """The float64 reference, clipping as hard as possible and as softly as possible.

    Both sharp decisions move in the same direction: a smaller sigma clips more, and a
    larger eps_multiplier rejects more variances (and a rejected variance zeroes a whole
    row). So the strict bound takes both, and the loose bound neither.
    """

    def run(s, eps):
        return ReferenceIntensityClipper(axis, s, Df, Dt, niter, iter_sigma, two_pass,
                                         nt_chunk=nt_chunk, eps_multiplier=eps).apply(I, W)

    return (run(sigma * (1.0 - sigma_bracket), EPS_HI),
            run(sigma * (1.0 + sigma_bracket), EPS_LO))


def test_intensity_clipper(iteration=0, rng=None, verbose=False):
    """One randomized comparison of GpuIntensityClipper against its numpy reference."""

    try:
        import cupy as cp
    except ImportError:
        if verbose:
            atomic_print('    test_intensity_clipper: cupy not available, skipped')
        return

    rng = _default_rng(rng)

    (axis, Df, Dt, niter, sigma, iter_sigma, two_pass, warps) = random_config(rng)
    (B, F, T) = random_geometry(rng, axis, Df, Dt)
    (in_i, in_w) = random_arrays(rng, B, F, T)

    I64 = in_i.astype(np.float64)
    W64 = in_w.astype(np.float64)

    ic = GpuIntensityClipper(B, F, T, axis, sigma, Df, Dt, niter, iter_sigma, two_pass, warps)
    w_gpu = _run_gpu(cp, ic, in_i, in_w)

    # Check 1 (the main one): the statistic comes from the GPU, the clip from numpy.
    # Both sides start from a bit-identical (mean, var), so this works at every niter --
    # see this file's docstring for why that matters.
    (i_ds, mean, var) = _run_pipeline(cp, ic, in_i, in_w)

    lo = intensity_clip(i_ds, in_w, mean, var, sigma * (1.0 - SIGMA_BRACKET), axis, Df, Dt)
    hi = intensity_clip(i_ds, in_w, mean, var, sigma * (1.0 + SIGMA_BRACKET), axis, Df, Dt)
    _sandwich(f'final clip (niter={niter})', w_gpu, lo, hi)

    # Check 2: the whole transform against the float64 reference, at niter=1 only. This is
    # what validates the downsample, the statistic and the axis reshaping as a unit; it is
    # confined to niter=1 because bracketing does not propagate through refinements.
    if niter == 1:
        (ic1, w1, m1, v1) = (ic, w_gpu, mean, var)
    else:
        ic1 = GpuIntensityClipper(B, F, T, axis, sigma, Df, Dt, 1, iter_sigma, two_pass, warps)
        w1 = _run_gpu(cp, ic1, in_i, in_w)
        (_, m1, v1) = _run_pipeline(cp, ic1, in_i, in_w)

    sb = _sigma_bracket(ic1.wrms_L, two_pass, sigma, m1, v1)
    (r_lo, r_hi) = _reference_bracket(axis, sigma, Df, Dt, 1, iter_sigma, two_pass,
                                      I64, W64, sb)
    _sandwich('end-to-end (niter=1)', w1, r_lo, r_hi)

    # Structural check 1: AXIS_FREQ equals AXIS_TIME on the transposed input, BITWISE.
    # The FREQ path transposes and then reduces rows, which is the same kernel on the same
    # values in the same order as the TIME path on pre-transposed input, so there is no
    # roundoff to allow for -- provided GpuWiDownsampler's (1,1) transpose copies the
    # intensity through exactly, which it does. This is the cheapest strong check on the
    # transpose plumbing, and it needs no reference at all.
    if (Df == 1) and (Dt == 1) and (axis == ClipperAxis.FREQ):
        ic_t = GpuIntensityClipper(B, T, F, ClipperAxis.TIME, sigma, 1, 1, niter,
                                   iter_sigma, two_pass, warps)
        w_t = _run_gpu(cp, ic_t, np.ascontiguousarray(np.swapaxes(in_i, 1, 2)),
                       np.ascontiguousarray(np.swapaxes(in_w, 1, 2)))
        assert np.array_equal(np.swapaxes(w_t, 1, 2), w_gpu), \
            'AXIS_FREQ disagrees with AXIS_TIME on the transposed input'

    # Structural check 2: the mask is constant within a (Df, Dt) cell. Either all of a
    # cell's weights are zero, or none of them changed. This is the direct test of the
    # kernel's Df/Dt write loop -- an off-by-one in it fails here immediately.
    wg = w_gpu.reshape(B, F//Df, Df, T//Dt, Dt)
    wi = in_w.reshape(B, F//Df, Df, T//Dt, Dt)
    cell_zero = (wg == 0).all(axis=(2, 4))
    cell_same = (wg == wi).all(axis=(2, 4))
    assert (cell_zero | cell_same).all(), \
        'a (Df,Dt) cell was partially masked; the clip must zero a cell as a unit'

    # Structural check 3: warps_per_block is a performance knob, and must not change the
    # answer. Bitwise, and safe at any niter -- unlike GpuWrms's threads_per_block, this
    # knob changes nothing inside the statistic.
    for w2 in WARP_COUNTS:
        if w2 == warps:
            continue
        ic2 = GpuIntensityClipper(B, F, T, axis, sigma, Df, Dt, niter, iter_sigma, two_pass, w2)
        assert np.array_equal(_run_gpu(cp, ic2, in_i, in_w), w_gpu), \
            f'warps_per_block {warps} vs {w2}: different result'

    # Structural check 4: determinism. Catches races and uninitialized scratch.
    assert np.array_equal(_run_gpu(cp, ic, in_i, in_w), w_gpu), \
        'GpuIntensityClipper is not deterministic'

    # Structural check 5: weights only ever decrease, and a zero weight stays zero.
    assert np.all(w_gpu <= in_w), 'a weight increased'
    assert np.all(w_gpu[in_w == 0] == 0), 'a zero weight became nonzero'

    # Structural check 6: masked data is unused. Poison with large FINITE values, not
    # NaN/inf: every kernel here multiplies by a zero weight rather than branching, so
    # 0*NaN would propagate in the original C++ too.
    poisoned = np.where(in_w == 0, rng.uniform(-1e6, 1e6, size=in_i.shape), in_i).astype(np.float32)
    assert np.array_equal(_run_gpu(cp, ic, poisoned, in_w), w_gpu), \
        'masked intensity changed the result'

    # Structural check 7: B is a spectator. The only axis we added ourselves, and the
    # check is nearly free.
    if B > 1:
        ic1b = GpuIntensityClipper(1, F, T, axis, sigma, Df, Dt, niter, iter_sigma,
                                   two_pass, warps)
        for b in range(B):
            w_b = _run_gpu(cp, ic1b, in_i[b:b+1], in_w[b:b+1])
            assert np.array_equal(w_b[0], w_gpu[b]), f'beam {b} depends on the other beams'

    # Structural check 8: the reference's T = N*nt_chunk really is N independent calls.
    # This is what pins down the semantics of the generalization GpuIntensityClipper has
    # not implemented (see the chunking note in IntensityClipper.hpp), so that extending
    # the GPU side later means matching a tested reference rather than re-deriving the
    # answer. Run on a fraction of draws: it costs two more float64 reference runs on
    # twice the data.
    if rng.uniform() < 0.25:
        (j_i, j_w) = random_arrays(rng, B, F, T)
        big_i = np.concatenate([in_i, j_i], axis=2)
        big_w = np.concatenate([in_w, j_w], axis=2)

        w2 = _run_gpu(cp, ic1, j_i, j_w)
        (_, m2, v2) = _run_pipeline(cp, ic1, j_i, j_w)
        sb2 = max(sb, _sigma_bracket(ic1.wrms_L, two_pass, sigma, m2, v2))

        (c_lo, c_hi) = _reference_bracket(axis, sigma, Df, Dt, 1, iter_sigma, two_pass,
                                          big_i.astype(np.float64), big_w.astype(np.float64),
                                          sb2, nt_chunk=T)
        _sandwich('chunked reference vs 2 GPU calls', np.concatenate([w1, w2], axis=2),
                  c_lo, c_hi)

    if verbose:
        nmask = int(np.sum((w_gpu == 0) & (in_w != 0)))
        path = 'shared' if (ic.wrms_L <= L_SHARED_MAX) else 'global'
        atomic_print(f'    test_intensity_clipper(axis={ic.axis}, (Df,Dt)=({Df},{Dt}),'
                     f' (B,F,T)=({B},{F},{T}), niter={niter}, two_pass={two_pass},'
                     f' warps={warps}, {path}): {nmask} of {w_gpu.size} weights clipped')
