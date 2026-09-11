"""Randomized unit tests for GpuSplineDetrender.

Dispatched from ``python -m pirate_frb test --cfrb``.

The GPU port is checked against ReferenceSplineDetrender, a numpy transcription of the
old code's own reference, on the weight patterns the real pipeline produces -- and NOT
on adversarially constructed masks. The goal is to reproduce the old estimator, not to
characterize it (see the class comment in include/pirate/chimefrb/SplineDetrender.hpp),
and the patterns below are the ones a CHIME RFI chain emits: clipper flagging, bad-channel
runs, 16x-downsampled counts, and the degenerate columns those produce.

Beyond the float comparison, six properties hold BITWISE and are asserted as such: the
result is reproducible run to run; the beam axis is a spectator; permuting the time
columns permutes the output (no time sample influences another); NaN in a zero-weight
channel leaves every weighted channel's output unchanged; a sample with no weight is
left untouched; and scaling every weight by 4 changes nothing at all. The last is the
estimator's exact invariance under w -> c*w -- data term and penalty both scale by c --
and for a power of two every float32 product and sum scales exactly too, so it holds to
the bit. It is the direct test that {0,1} weights and counts are one algorithm.

WHAT THIS FILE CANNOT ESTABLISH: a misreading of the old kernel would pass every check
here, since the reference and the port were written from the same reading. The spot
check misc/chimefrb/spline_detrender/ runs the old code itself.
"""

import numpy as np

from . import GpuSplineDetrender, ReferenceSplineDetrender
from .ReferenceSplineDetrender import bin_edges, hermite_basis
from ..utils import atomic_print
from .testutils import default_rng as _default_rng


# The two shapes the old search's production RFI chain runs the detrender at, as
# (nfreq, nbins, epsilon, weight kind): the 16x-downsampled sub-pipelines see count-valued
# weights, the full-band instance sees {0,1}. Drawn often, so a run spends its iterations
# where the port will.
PRODUCTION_CONFIGS = [(1024, 6, 3.0e-4, 'counts'), (16384, 6, 3.0e-4, 'binary')]

# Per-column weight patterns; see random_weights().
WEIGHT_KINDS = ['ones', 'binary', 'counts', 'continuous', 'dead_run', 'single', 'zero']

# Tolerances for the float comparison, per (beam, time) sample, with 'scale' the larger
# of max |intensity| and max |baseline| and the conditioning statistics from
# ReferenceSplineDetrender.conditioning():
#
#     at weighted channels:    |gpu - reference|  <=  TOL_WEIGHTED * eps_mach * scale / r_min
#     at every channel:        |gpu - reference|  <=  TOL_ALL      * eps_mach * scale / l_min
#
# Two bounds because the kernel's float32 error has two regimes. At a weighted channel
# it is the accumulation over a freq-range (~10 eps_mach, see detrender_kernels.hpp) plus
# an equilibrated solve whose error is of order eps_mach / r_min; the 1/r_min form is the
# one pirate_frb.detrending.lps2d.tests.test_gpu_kernel() uses for these kernels. At a
# channel WITHOUT weight the fitted value is set by the penalty alone, in the near-null
# directions of the normal equations that the pivots do not see, and there the error is of
# order eps_mach / l_min -- which at epsilon ~ 1e-6 can be a hundred times worse. These
# values are unused downstream (their weight is zero) and the old code's own float32
# solve is inaccurate there too, but they are the same estimator and are checked as such.
#
# MEASURED over 400 random draws, production shapes included, in these units:
#
#     weighted (/ r_min):   median 2.2   p90 4.7   max 17
#     all      (/ l_min):   median 2.9   p90 6.8   max 27
#
# with every tail case at epsilon ~ 1e-6, or a single-bin fit, or both -- the poorly
# conditioned corner the 1/r_min and 1/l_min factors exist for. The tolerances sit 4-5x
# above the observed maxima; an indexing, basis or regulator error shows up at 1e3 and
# beyond, so the margin costs no detection power.
TOL_WEIGHTED = 64.0
TOL_ALL = 128.0


def random_config(rng):
    """Draw (nfreq, nbins, epsilon, weight kind).

    Production values about a third of the time; otherwise nbins in 1..8, nfreq
    log-uniform from a few channels per bin to the full CHIME band, and epsilon
    log-uniform over four decades. A weight kind of None means "mix the patterns per
    column" (random_weights()); the production draws pin their own.
    """
    if rng.uniform() < 0.35:
        return PRODUCTION_CONFIGS[rng.integers(len(PRODUCTION_CONFIGS))]

    nbins = int(rng.integers(1, 9))
    nfreq = int(np.exp(rng.uniform(np.log(max(nbins, 8)), np.log(16384))))
    epsilon = float(10.0 ** rng.uniform(-6.0, -2.0))
    return (nfreq, nbins, epsilon, None)


def random_geometry(rng):
    """Draw (M, T). Small: the reference solves an (M*T)-batch of dense systems."""
    M = int(rng.integers(1, 4))
    T = int(rng.choice([32, 64, 128]))
    return (M, T)


def _weight_column(rng, nfreq, nbins, edges, kind):
    """One (nfreq,) float64 weight column of the given kind."""
    if kind == 'ones':
        return np.ones(nfreq)
    if kind == 'zero':
        return np.zeros(nfreq)
    if kind == 'single':
        c = np.zeros(nfreq)
        c[rng.integers(nfreq)] = rng.uniform(0.5, 16.0)
        return c

    p = rng.uniform(0.3, 1.0)
    if kind == 'binary':
        c = (rng.uniform(size=nfreq) < p).astype(np.float64)
    elif kind == 'counts':
        c = rng.binomial(16, p, size=nfreq).astype(np.float64)
    elif kind == 'continuous':
        c = rng.uniform(0.0, 2.0, size=nfreq)
    else:
        c = np.ones(nfreq)                                    # dead_run

    # A contiguous dead run -- what a bad-channel mask looks like -- always for
    # 'dead_run', half the time for the clipper-like kinds. Half of the runs are aligned
    # to whole bins, so that a bin with no data at all is common: its coefficients are
    # then set by the regulator alone.
    if (kind == 'dead_run') or (kind in ('binary', 'counts') and rng.uniform() < 0.5):
        if rng.uniform() < 0.5:
            b0 = int(rng.integers(0, nbins))
            b1 = int(rng.integers(b0 + 1, nbins + 1))
            (lo, hi) = (int(edges[b0]), int(edges[b1]))
        else:
            L = int(rng.integers(1, max(2, nfreq // 2 + 1)))
            lo = int(rng.integers(0, nfreq - L + 1))
            hi = lo + L
        c[lo:hi] = 0.0
    return c


def random_weights(rng, M, nfreq, nbins, T, kind=None):
    """(M, nfreq, T) float32 weights, one pattern per (beam, time) column.

    kind=None mixes the families the pipeline produces; a fixed kind uses that family
    only (the production draws). The families:

      ones         every weight 1
      binary       Bernoulli {0,1} at a per-column rate (a clipper's output mask)
      counts       integers 0..16, Binomial(16, p) (16x-downsampled clipper output)
      continuous   uniform in [0, 2] (nothing in the pipeline makes these; the kernel
                   accepts them, and they are what pins the weighted fit)
      dead_run     ones with one contiguous zero run, often covering whole bins
      single       one nonzero channel: the fit passes through it exactly
      zero         no weight: the sample must be left untouched

    'binary' and 'counts' also get a dead run half the time.
    """
    edges = bin_edges(nfreq, nbins)
    probs = {'ones': 0.15, 'binary': 0.25, 'counts': 0.20, 'continuous': 0.10,
             'dead_run': 0.20, 'single': 0.05, 'zero': 0.05}
    kinds = list(probs.keys())
    pk = np.array(list(probs.values()))

    w = np.empty((M, nfreq, T), dtype=np.float32)
    for m in range(M):
        for t in range(T):
            k = kind if (kind is not None) else kinds[rng.choice(len(kinds), p=pk)]
            w[m, :, t] = _weight_column(rng, nfreq, nbins, edges, k)
    return w


def random_intensity(rng, M, nfreq, nbins, T):
    """(M, nfreq, T) float32: a baseline in the spline's span plus unit Gaussian noise.

    The baseline has a random value (of order 100, spread 30) and slope (of order 30 per
    bin) at every bin edge, so with no noise both codes would fit it exactly up to
    shrinkage. There is deliberately no large DC offset: neither code subtracts one, and
    both would lose float32 precision to it.
    """
    (_, H) = hermite_basis(nfreq, nbins)
    edges = bin_edges(nfreq, nbins)
    coeffs = np.zeros((M, T, 2*(nbins+1)))
    coeffs[:, :, 0::2] = 100.0 + 30.0*rng.standard_normal((M, T, nbins+1))
    coeffs[:, :, 1::2] = 30.0*rng.standard_normal((M, T, nbins+1))

    base = np.empty((M, nfreq, T))
    for b in range(nbins):
        sl = slice(edges[b], edges[b+1])
        base[:, sl, :] = np.einsum('fa,mta->mft', H[sl], coeffs[:, :, 2*b:2*b+4])
    return (base + rng.standard_normal(base.shape)).astype(np.float32)


def _run_gpu(cp, det, intensity, weights):
    """Run one GpuSplineDetrender on numpy inputs; returns the detrended intensity (numpy)."""
    gi = cp.asarray(intensity)          # a copy: the kernel works in place
    gw = cp.asarray(weights)
    det.launch(gi, gw)
    cp.cuda.get_current_stream().synchronize()
    return cp.asnumpy(gi)


def test_spline_detrender(iteration=0, rng=None, verbose=False):
    """One randomized comparison of GpuSplineDetrender against ReferenceSplineDetrender."""

    try:
        import cupy as cp
    except ImportError:
        if verbose:
            atomic_print('    test_spline_detrender: cupy not available, skipped')
        return

    rng = _default_rng(rng)

    (nfreq, nbins, epsilon, kind) = random_config(rng)
    (M, T) = random_geometry(rng)
    weights = random_weights(rng, M, nfreq, nbins, T, kind)
    intensity = random_intensity(rng, M, nfreq, nbins, T)
    tag = f'(nfreq={nfreq}, nbins={nbins}, epsilon={epsilon:.3g}, kind={kind}, M={M}, T={T})'

    det = GpuSplineDetrender(nfreq, nbins, epsilon, M, T)
    ref = ReferenceSplineDetrender(nfreq, nbins, epsilon)
    assert list(det.bin_edges()) == list(ref.edges), f'bin edges differ {tag}'

    gpu = _run_gpu(cp, det, intensity, weights)
    assert np.isfinite(gpu).all(), f'non-finite output {tag}'

    # ---- Against the reference. Both subtract the baseline at every channel, so the
    # comparison covers every channel, with two bounds (see TOL_WEIGHTED / TOL_ALL above).
    # A sample with no weight has r_min = l_min = 0 and is checked bitwise further down.
    model = ref.fit(intensity, weights)
    resid = intensity.astype(np.float64) - model
    scale = max(float(np.abs(intensity).max()), float(np.abs(model).max()))
    (rmin, lmin) = ref.conditioning(weights)                                 # (M, T) each
    eps32 = np.finfo(np.float32).eps
    diff = np.abs(gpu.astype(np.float64) - resid)

    valid = (weights > 0)
    live = np.broadcast_to((rmin > 0)[:, None, :], diff.shape)
    err_w = diff / (eps32 * scale / np.where(rmin > 0, rmin, 1.0))[:, None, :]
    err_a = diff / (eps32 * scale / np.where(lmin > 0, lmin, 1.0))[:, None, :]
    err_valid = float(err_w[valid & live].max()) if (valid & live).any() else 0.0
    err_all = float(err_a[live].max()) if live.any() else 0.0
    assert err_valid <= TOL_WEIGHTED, f'weighted channels: {err_valid:.3g} x eps_mach*scale/r_min {tag}'
    assert err_all <= TOL_ALL, f'all channels: {err_all:.3g} x eps_mach*scale/l_min {tag}'

    # ---- Run-to-run bit identity.
    assert np.array_equal(_run_gpu(cp, det, intensity, weights), gpu), f'not reproducible {tag}'

    # ---- The beam axis is a spectator (the old code ran one beam per pipeline, so this
    # axis is ours and nothing else checks it).
    if M > 1:
        det1 = GpuSplineDetrender(nfreq, nbins, epsilon, 1, T)
        for m in range(M):
            one = _run_gpu(cp, det1, intensity[m:m+1], weights[m:m+1])
            assert np.array_equal(one[0], gpu[m]), f'beam {m} is not a spectator {tag}'

    # ---- Permuting the time columns permutes the output: no sample sees another, and
    # the freq-range partition (part of the summation order) depends on T alone.
    perm = rng.permutation(T)
    pg = _run_gpu(cp, det, np.ascontiguousarray(intensity[:, :, perm]),
                  np.ascontiguousarray(weights[:, :, perm]))
    assert np.array_equal(pg, gpu[:, :, perm]), f'time columns are not independent {tag}'

    # ---- NaN in a zero-weight channel changes nothing at the weighted channels. This is
    # the deliberate departure from the old code, which would propagate it.
    if (~valid).any():
        poisoned = intensity.copy()
        poisoned[~valid] = np.nan
        pn = _run_gpu(cp, det, poisoned, weights)
        assert np.array_equal(pn[valid], gpu[valid]), f'NaN at a zero-weight channel leaked {tag}'

    # ---- A sample with no weight is left untouched.
    dead = (weights.sum(axis=1) == 0)                          # (M, T)
    if dead.any():
        for (m, t) in zip(*np.nonzero(dead)):
            assert np.array_equal(gpu[m, :, t], intensity[m, :, t]), \
                f'an all-zero-weight sample was modified {tag}'

    # ---- Scaling every weight by a power of two is a bitwise no-op (see the module
    # docstring). 4 rather than 2, so that counts of 16 become 64 and the total weight of
    # a full 16384-channel column still sits far below float32 overflow.
    assert np.array_equal(_run_gpu(cp, det, intensity, 4.0*weights), gpu), \
        f'weight scaling by 4 changed the output {tag}'

    if verbose:
        atomic_print(f'    test_spline_detrender{tag}: err {err_valid:.3g} (weighted, /r_min) / '
                     f'{err_all:.3g} (all, /l_min) x eps_mach*scale; min r_min '
                     f'{float(rmin[rmin > 0].min()) if (rmin > 0).any() else 0:.3g}, min l_min '
                     f'{float(lmin[lmin > 0].min()) if (lmin > 0).any() else 0:.3g}; '
                     f'{int(dead.sum())} dead samples')
    return (err_valid, err_all)
