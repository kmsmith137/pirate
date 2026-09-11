"""Randomized unit tests for GpuWrms.

Dispatched from ``python -m pirate_frb test --cfrb``.

Two things make this more than a call and an allclose, and both are borrowed from the old
code's own unit test (rf_kernels/test-intensity-clipper.cpp) rather than invented.

BRACKETING, for the variance-validity cutoffs. A row whose variance falls below
(eps_2*mean)^2 or eps_3*mean^2 is declared dead and its variance set to exactly zero. The
kernel and the reference do not evaluate those cutoffs in identical arithmetic -- see
ReferenceWrms's docstring -- so instead of a tolerance we bracket the decision: run the
reference at eps_multiplier 0.5 and 1.5, and require the kernel's set of valid rows to lie
between the conservative and permissive ones. Only then are the numbers compared, and only
on rows both call valid.

INDUCTION, for niter > 1. Bracketing does not propagate through refinements: one sample
that lands on the other side of a threshold in float32 than in float64 changes the mean,
which changes the next threshold, and after nine rounds the two answers are unrelated. So
the refinements are never compared end to end. Instead, the kernel is run at niter-1 AND
at niter, and ONE reference step is taken from the kernel's own niter-1 state. Both sides
then start from bit-identical state and there is nothing to amplify.

How badly do the refinements amplify? Measurably. On the single-pass path with a mean of
420 and an rms of 12, the first pass's variance is only good to about four digits in
float32 (var = <I^2> - mean^2 cancels three of the seven), so the first refinement's
threshold is uncertain at 5e-5 relative, which moves roughly one sample of 6000 in or out
of the survivor set, which moves the mean by 1.5e-5 relative -- 40x the budget the
un-refined comparison is held to. That is not a bug in anything; it is why two_pass exists.
It is also why nothing in this file compares two full refinement chains to each other.

That induction has a hole, and it is closed elsewhere on purpose: the survivor set it
feeds the reference comes from our own iclip(), so a wrong comparison in the kernel's
refinement would cancel out. What closes it is that the kernel's refinement and (later)
the intensity_clipper's final clip share one __device__ predicate, wrms_survives() in
include/pirate/chimefrb/Wrms.hpp, and the final clip IS tested against an independent
reference. The spot check in misc/chimefrb/rfi_wrms/ closes it too, by running the old
kernel's own internal masking.

A NOTE ON GPU-VS-GPU COMPARISONS. Nothing here compares two GPU runs and asserts that
they agree on a sharp decision. Two block sizes sum in a different order, so a row whose
variance sits within roundoff of a validity cutoff lands on either side depending on the
block size, and requiring them to agree is requiring something no two correct
implementations must satisfy. Every such comparison goes through the bracket instead.
(Bitwise comparisons are fine where nothing sharp is involved -- determinism, row
independence, and the masked-data check below are all bitwise.)

WHAT THIS FILE CANNOT ESTABLISH: the numpy reference is a transcription, so a shared
misreading would pass everything here. That is what the spot check is for.
"""

import numpy as np

from . import GpuWrms, ReferenceWrms, wrms_iterate, iclip
from ..utils import atomic_print


THREAD_COUNTS = [128, 256, 512, 1024]

# The kernel switches from the shared-memory path to the global-memory one when a row
# stops fitting in shared memory. Asked for rather than recomputed: the budget has to
# leave room for the block-reduction buffer as well as the row, and a test that worked
# that out for itself would be free to get it wrong in the same way the kernel once did.
L_SHARED_MAX = GpuWrms.max_shared_L()


def random_config(rng):
    """Draw (L, R, niter, iter_sigma, two_pass, threads_per_block).

    L straddles the shared-memory threshold, because the two paths are different kernels
    and only a draw on each side exercises both. R is then bounded so that R*L stays
    manageable -- the large-L path has few rows by nature (its caller is the AXIS_NONE
    clipper, whose 'row' is a whole plane), which is exactly the regime where a per-row
    kernel would starve, so it is worth sampling honestly rather than making R large.
    """

    u = rng.uniform()
    if u < 0.1:
        # Right at the boundary between the two paths, a few percent of the time. This
        # band is where a shared-memory budget that forgets an allocation shows up -- as a
        # launch failure rather than a wrong answer -- and it is narrow enough to hide for
        # a long time behind a uniform draw. It has caught one such bug already.
        L = int(rng.integers(L_SHARED_MAX - 16, L_SHARED_MAX + 17))
    elif u < 0.7:
        L = int(rng.integers(32, L_SHARED_MAX + 1))         # shared-memory path
    else:
        L = int(rng.integers(L_SHARED_MAX + 1, 40000))      # global-memory path

    R = int(rng.integers(1, max(2, min(600, 2_000_000 // L))))

    # niter=1 is the base case and the std_dev_clipper's setting; 9 is the production
    # intensity_clipper. The small values in between keep the induction honest at its
    # boundary, where niter-1 is itself the base case.
    niter = int(rng.choice([1, 2, 3, 9]))
    iter_sigma = float(rng.uniform(2.0, 6.0))
    two_pass = bool(rng.integers(2))
    tpb = int(rng.choice(THREAD_COUNTS))

    return (L, R, niter, iter_sigma, two_pass, tpb)


def random_arrays(rng, R, L):
    """Random (intensity, weights), float32, shaped (R, L).

    Designed around the code paths rather than around realism:

      - a large per-row offset, because the eps_2*mean cutoff is inert when the mean is
        near zero, and because it is what makes the single-pass variance actually
        cancel -- which is the whole reason two_pass exists;
      - 20-sigma outliers in some rows, so the refinements discard something (on clean
        Gaussian data a 5-sigma clip discards nothing and niter is untested);
      - a few percent of rows fully masked, constant, or near-constant at a large mean,
        which are the three ways to land on a validity cutoff.

    Deliberately NOT included: samples placed AT a threshold. Sampling the ambiguous
    region on purpose would make the brackets straddle at a controlled rate rather than a
    negligible one, which is worse, not better.
    """

    offset = rng.uniform(-1.0e3, 1.0e3)
    scale = rng.uniform(0.5, 20.0)
    intensity = rng.normal(offset, scale, size=(R, L))

    p = np.clip(rng.uniform(-0.1, 1.1), 0.0, 1.0)
    weights = (rng.uniform(size=(R, L)) < p) * rng.uniform(0.5, 1.5, size=(R, L))

    # Outliers, in a random subset of rows.
    for r in np.flatnonzero(rng.uniform(size=R) < 0.3):
        cols = rng.integers(0, L, size=max(1, L // 200))
        intensity[r, cols] = offset + 20.0 * scale * rng.choice([-1.0, 1.0])

    # Degenerate rows, a few percent each.
    for r in np.flatnonzero(rng.uniform(size=R) < 0.05):
        intensity[r, :] = offset                       # variance exactly zero
    for r in np.flatnonzero(rng.uniform(size=R) < 0.05):
        weights[r, :] = 0.0                            # no weight at all
    for r in np.flatnonzero(rng.uniform(size=R) < 0.05):
        intensity[r, :] = offset * (1.0 + 1.0e-6 * rng.normal(size=L))   # on the cutoff

    return (intensity.astype(np.float32), weights.astype(np.float32))


def _run_gpu(cp, wrms, in_i, in_w):
    """Run one GpuWrms on numpy inputs; returns numpy (mean, var).

    The outputs are pre-filled with NaN, and the caller checks that none survives: that
    is what catches a row the kernel never wrote.
    """

    R = in_i.shape[0]
    g_mean = cp.full(R, cp.nan, dtype=cp.float32)
    g_var = cp.full(R, cp.nan, dtype=cp.float32)

    wrms.launch(g_mean, g_var, cp.asarray(in_i), cp.asarray(in_w))
    cp.cuda.get_current_stream().synchronize()

    (mean, var) = (cp.asnumpy(g_mean), cp.asnumpy(g_var))
    assert not np.isnan(mean).any(), 'GpuWrms left part of mean unwritten'
    assert not np.isnan(var).any(), 'GpuWrms left part of var unwritten'
    assert np.all(var >= 0.0), 'GpuWrms returned a negative variance'
    return (mean, var)


# float32 roundoff, and the safety factor the budgets below carry over the disagreement
# actually measured between the kernel and the float64 reference.
EPS32 = 1.2e-7
MARGIN = 50.0


def _budgets(L, two_pass, mean, rms):
    """Absolute error budgets for (mean, rms), given the reference's own values.

    Derived from the arithmetic rather than guessed, and then checked against a 40-draw
    sweep of the parameter space; the measured worst cases sit 15x to 65x inside these.

      - The mean is a weighted average, so its error is float32 roundoff times its own
        scale. Measured worst: 3.9e-7 of (|mean| + rms).

      - The rms carries the same term, plus -- on the SINGLE-PASS path only -- the
        cancellation in var = <I^2> - mean^2, which loses log10(mean^2/var) digits out of
        float32's seven. That term grows as sqrt(L) with the accumulated sum, and it is
        the entire reason two_pass exists. Measured worst: 1.4e-3 of rms on that path
        (versus 2.4e-7 with two_pass), or 1.85e-5 of |mean|, against a predicted
        sqrt(40000)*EPS32 = 2.4e-5.

    The old unit test also carries a '1e-2 * sqrt(|mean_hint - mean|)' term, for the last
    refinement's variance being poorly determined when the mean moves. We do not: their
    comparison was float32 against float32 from different starting states, ours is float32
    against float64 from bit-identical state (see the induction in this file's docstring),
    and the sweep showed no such effect. If a failure ever turns up on a row whose mean
    moved a long way, that is the term to reinstate.
    """

    scale = np.abs(mean) + rms
    eps_m = MARGIN * EPS32 * scale
    eps_r = MARGIN * EPS32 * rms
    if not two_pass:
        eps_r = eps_r + MARGIN * EPS32 * np.sqrt(L) * np.abs(mean)
    return (eps_m, eps_r)


def _check(label, L, two_pass, got_mean, got_var, ref):
    """Bracket the validity decision, then compare the numbers where both say valid.

    'ref' is {eps_multiplier: (mean, var)} for 0.5 (permissive, rejects fewer variances),
    1.0 (the real algorithm) and 1.5 (conservative, rejects more). Returns the two errors
    as fractions of their budgets, so that a regression shows up as the number moving
    rather than only as a failure.
    """

    (_, lo_var) = ref[0.5]
    (mid_mean, mid_var) = ref[1.0]
    (_, hi_var) = ref[1.5]

    valid = (got_var > 0)
    bad = valid & ~(lo_var > 0)
    bad |= (~valid) & (hi_var > 0)
    assert not bad.any(), (f'{label}: {int(bad.sum())} row(s) whose validity is outside '
                           f'the eps_multiplier [0.5, 1.5] bracket')

    both = valid & (mid_var > 0)
    if not both.any():
        return (0.0, 0.0, 0)

    got_rms = np.sqrt(got_var[both])
    ref_rms = np.sqrt(mid_var[both])
    (eps_m, eps_r) = _budgets(L, two_pass, mid_mean[both], ref_rms)

    rm = float(np.max(np.abs(got_mean[both] - mid_mean[both]) / eps_m))
    rr = float(np.max(np.abs(got_rms - ref_rms) / eps_r))

    assert rm <= 1.0, f'{label}: mean is {rm:.3g} x its error budget'
    assert rr <= 1.0, f'{label}: rms is {rr:.3g} x its error budget'
    return (rm, rr, int(both.sum()))


def _reference_triple(niter, iter_sigma, two_pass, I, W):
    """The reference at eps_multiplier 0.5, 1.0 and 1.5."""

    return {e: ReferenceWrms(niter, iter_sigma, two_pass, eps_multiplier=e).apply(I, W)
            for e in (0.5, 1.0, 1.5)}


def test_wrms(iteration=0, rng=None, verbose=False):
    """One randomized comparison of GpuWrms against ReferenceWrms."""

    try:
        import cupy as cp
    except ImportError:
        if verbose:
            atomic_print('    test_wrms: cupy not available, skipped')
        return

    rng = np.random.default_rng() if (rng is None) else rng

    (L, R, niter, iter_sigma, two_pass, tpb) = random_config(rng)
    (in_i, in_w) = random_arrays(rng, R, L)

    I = in_i.astype(np.float64)
    W = in_w.astype(np.float64)

    wrms = GpuWrms(L, niter, iter_sigma, two_pass, tpb)
    (gpu_mean, gpu_var) = _run_gpu(cp, wrms, in_i, in_w)

    if niter == 1:
        # Base case: no refinements, so a direct comparison is safe. mean_hint is the
        # mean the last step started from -- zero on the single-pass path, and the mean
        # itself on the two-pass path, whose second sweep starts from it.
        ref = _reference_triple(1, iter_sigma, two_pass, I, W)
        (rm, rr, nboth) = _check('niter=1', L, two_pass, gpu_mean, gpu_var, ref)
    else:
        # Inductive step: one reference refinement from the KERNEL's own niter-1 state.
        prev = GpuWrms(L, niter-1, iter_sigma, two_pass, tpb)
        (pm, pv) = _run_gpu(cp, prev, in_i, in_w)

        Wk = iclip(pm, iter_sigma * np.sqrt(pv), I, W)
        ref = {e: wrms_iterate(pm, I, Wk, eps_multiplier=e) for e in (0.5, 1.0, 1.5)}
        # A refinement's sums are taken about the mean, so there is no cancellation to
        # budget for even on the single-pass path: pass two_pass=True here.
        (rm, rr, nboth) = _check(f'niter={niter}', L, True, gpu_mean, gpu_var, ref)

    # Structural check 1: threads_per_block is a performance knob. Every value on the menu
    # is held to the reference, with the same bracketing and the same budgets as the main
    # comparison above.
    #
    # AT niter=1 ONLY, and deliberately so. Comparing two full refinement chains is the
    # comparison the induction above exists to avoid: two correct float32 runs genuinely
    # diverge, because the first pass's variance sets the first refinement's threshold, and
    # a threshold that differs by roundoff moves a sample in or out of the survivor set.
    # On the single-pass path with a mean far from zero this is not a subtle effect --
    # measured at 1.5e-5 on the mean, 40x its budget, from a single sample flipping in a
    # row of 6000 -- and it is the whole reason two_pass exists. niter=1 is where the block
    # reduction, which is the only thing threads_per_block changes, is exercised cleanly.
    #
    # Note what is NOT asserted: that two block sizes produce the same SET of valid rows.
    # The validity cutoff is a hard decision on a float32 variance, and two block sizes sum
    # in a different order, so a row whose variance sits within roundoff of the cutoff can
    # legitimately land on either side. That is not rare and not hypothetical -- it happens
    # on about 1% of draws, always on the single-pass path, always within 0.2% of the
    # cutoff, because random_arrays() deliberately makes some rows nearly constant at a
    # large mean in order to reach exactly this corner. Bracketing each block size against
    # the reference is both correct and strictly stronger than requiring them to agree with
    # each other.
    ref1 = ref if (niter == 1) else _reference_triple(1, iter_sigma, two_pass, I, W)

    for tpb2 in THREAD_COUNTS:
        (m2, v2) = _run_gpu(cp, GpuWrms(L, 1, iter_sigma, two_pass, tpb2), in_i, in_w)
        _check(f'niter=1, tpb={tpb2}', L, two_pass, m2, v2, ref1)

    # Structural check 2: rows are independent. Catches a scratch-indexing error on the
    # global path, where blocks from different rows share one scratch array.
    if R > 1:
        half = R // 2
        (m_h, v_h) = _run_gpu(cp, wrms, in_i[:half], in_w[:half])
        assert np.array_equal(m_h, gpu_mean[:half]), 'a row depends on the rows after it'
        assert np.array_equal(v_h, gpu_var[:half]), 'a row depends on the rows after it'

    # Structural check 3: masked data is unused. Poison with large FINITE values, not
    # NaN/inf: the kernel multiplies by a zero weight rather than branching, so 0*NaN
    # would propagate in the original C++ too.
    poisoned = np.where(in_w == 0, rng.uniform(-1e6, 1e6, size=in_i.shape), in_i).astype(np.float32)
    (m_p, v_p) = _run_gpu(cp, wrms, poisoned, in_w)
    assert np.array_equal(m_p, gpu_mean), 'masked intensity changed the mean'
    assert np.array_equal(v_p, gpu_var), 'masked intensity changed the variance'

    # Structural check 4: determinism. Catches races and uninitialized scratch.
    (m_d, v_d) = _run_gpu(cp, wrms, in_i, in_w)
    assert np.array_equal(m_d, gpu_mean) and np.array_equal(v_d, gpu_var), \
        'GpuWrms is not deterministic'

    # Structural check 5: a dead row stays dead. A row with no weight has no statistic at
    # any niter, and a rejected variance must not come back to life in a later refinement
    # (its threshold is zero, so nothing survives).
    dead = (in_w.sum(axis=1) == 0)
    if dead.any():
        assert np.all(gpu_var[dead] == 0) and np.all(gpu_mean[dead] == 0), \
            'a row with no weight produced a statistic'

    if verbose:
        path = 'shared' if wrms.is_shared_memory_path else 'global'
        atomic_print(f'    test_wrms(L={L}, R={R}, niter={niter}, two_pass={two_pass},'
                     f' tpb={tpb}, {path}): {nboth} rows compared,'
                     f' mean {rm:.2f}x budget, rms {rr:.2f}x budget')
