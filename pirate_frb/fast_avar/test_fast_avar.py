"""Tests comparing the fast_avar C++ classes against the slow_avar python reference.

Dispatched from pirate_frb/__main__.py via 'test --avar'. We compare a few key methods (not every
method), per the design: the convolver, the gridding+iterate sweep (exercises SparseTile), a direct
PfVariance check, and the end-to-end PfAvarApproximation.tree_variance.

Tolerances: cross-language float results are NOT bit-exact (numpy uses pairwise/vectorized
summation; the C++ uses sequential loops), so we use np.allclose tolerances rather than epsabs=0.
"""
import numpy as np

from .. import slow_avar
from . import (SparseTile, SparseTileTriple, PfVarianceConvolver,
               PfVariance, PfAvarApproximation)


def _allclose(a, b, rtol=1e-9, atol=1e-12):
    a = np.asarray(a, dtype=np.float64)
    b = np.asarray(b, dtype=np.float64)
    return (a.shape == b.shape) and np.allclose(a, b, rtol=rtol, atol=atol)


def test_cpp_convolver():
    """C++ PfVarianceConvolver vs python: Pmax, Tmax, A table, and variance(x, P)."""
    py = slow_avar.PfVarianceConvolver()
    cpp = PfVarianceConvolver()

    assert cpp.Pmax == py.Pmax, (cpp.Pmax, py.Pmax)
    assert np.array_equal(np.asarray(cpp.Tmax), py.Tmax), (list(cpp.Tmax), list(py.Tmax))
    assert _allclose(cpp.A, py.A), "convolver A table mismatch"

    # variance(x, P) on random 2-D x (S, nt) and random P. The python version accepts arbitrary
    # spectator dims; we compare on the 2-D shape the C++ specializes to.
    for _ in range(8):
        P = int(np.random.randint(1, py.Pmax + 1))
        S = int(np.random.randint(1, 6))
        T = int(np.random.randint(1, 13))          # spans T < and >= Tmax[P-1]
        x = np.ascontiguousarray(np.random.standard_normal((S, T)))
        got = np.asarray(cpp.variance(x, P))
        want = py.variance(x, P)
        assert got.shape == (S, P), (got.shape, S, P)
        assert _allclose(got, want, rtol=1e-11, atol=1e-13), \
            (P, S, T, float(np.abs(got - want).max()))


def _sweep_ntime(r):
    # ntime comfortably larger than the largest (delay + shift) after iterating a rank-r tree.
    return (((3 << r) + 128) // 32 + 1) * 32


def test_cpp_sparse_tile_triple():
    """C++ SparseTileTriple gridding + iterate vs python (compares unpack() after a random #steps).

    Exercises make_tree_gridding_output, slice, iterate_aligned, iterate_singletons (+lower/upper),
    remap_d, eval_tshifts transitively.
    """
    cm, ifreq = slow_avar.SparseTileTriple.random_channel_map()
    cm = np.ascontiguousarray(cm, dtype=np.float64)

    py = slow_avar.SparseTileTriple.make_tree_gridding_output(cm, ifreq)
    cpp = SparseTileTriple.make_tree_gridding_output(cm, ifreq)
    r = py.r
    nsteps = int(np.random.randint(0, r + 1))
    for _ in range(nsteps):
        py = py.iterate()
        cpp = cpp.iterate()

    assert cpp.r == py.r and cpp.k == py.k and cpp.f0 == py.f0 and cpp.nf == py.nf, \
        ((cpp.r, cpp.k, cpp.f0, cpp.nf), (py.r, py.k, py.f0, py.nf))

    ntime = _sweep_ntime(r)
    got = np.asarray(cpp.unpack(ntime))
    want = py.unpack(ntime)
    assert _allclose(got, want, rtol=1e-11, atol=1e-13), \
        (cm.shape, ifreq, nsteps, float(np.abs(got - want).max()))


def _make_cpp_tile(t):
    # Build a C++ SparseTile equivalent to a python SparseTile 't'.
    return SparseTile(int(t.r), int(t.k), int(t.f0), int(t.nf), int(t.nt), int(t.dbits),
                      np.ascontiguousarray(t.data, dtype=np.float64),
                      np.ascontiguousarray(t.tshifts, dtype=np.int64),
                      int(t.t0), float(t.scale))


def test_cpp_pf_variance():
    """C++ PfVariance vs python: from_tile + unpack, and add() with scale / upper_half."""
    py_conv = slow_avar.PfVarianceConvolver()
    cpp_conv = PfVarianceConvolver()
    Pmax = py_conv.Pmax

    k = int(np.random.randint(1, 6))                  # k >= 1 for the upper-half case
    t = slow_avar.SparseTile.make_random(k, k, 0, 1)  # singleton, rank k
    ct = _make_cpp_tile(t)
    P = int(np.random.randint(1, Pmax + 1))

    py_pv = slow_avar.PfVariance.from_tile(t, P, py_conv)
    cpp_pv = PfVariance.from_tile(ct, P, cpp_conv)
    assert _allclose(cpp_pv.unpack(int(t.dbits)), py_pv.unpack(int(t.dbits))), "from_tile mismatch"

    # add() with profile-truncation + scale (no upper_half).
    P1 = int(np.random.randint(1, P + 1))
    c = float(np.random.uniform(0.5, 2.0))
    py_a = slow_avar.PfVariance(k, P1); py_a.add(py_pv, scale=c)
    cpp_a = PfVariance(k, P1);          cpp_a.add(cpp_pv, scale=c)
    db = py_a.get_all_dbits()
    assert _allclose(cpp_a.unpack(db), py_a.unpack(db)), "add(scale) mismatch"

    # add() with upper_half: rank k -> k-1.
    py_b = slow_avar.PfVariance(k - 1, P1); py_b.add(py_pv, upper_half=True, scale=c)
    cpp_b = PfVariance(k - 1, P1);          cpp_b.add(cpp_pv, upper_half=True, scale=c)
    db = py_b.get_all_dbits()
    assert _allclose(cpp_b.unpack(db), py_b.unpack(db)), "add(upper_half) mismatch"


def test_cpp_pf_avar_approximation():
    """End-to-end: C++ PfAvarApproximation.tree_variance vs python, on a small random plan.

    Run once (it builds a plan and runs the full python reference sweep), not every iteration.
    """
    from ..pirate_pybind11 import DedispersionConfig, DedispersionPlan

    # gpu_valid=False: PfAvarApproximation is pure-CPU (tree structure + channel map only), so we
    # don't need a GPU-runnable config (which would require a precompiled cdd2 kernel for the rank).
    config = DedispersionConfig.make_random(max_toplevel_rank=5, max_early_triggers=2, gpu_valid=False)
    config.validate()
    plan = DedispersionPlan(config)
    fv = np.asarray(config.make_random_freq_variances(noisy=True), dtype=np.float64)

    py = slow_avar.PfAvarApproximation(plan, fv)
    cpp = PfAvarApproximation(plan, fv)

    assert cpp.ntrees == plan.ntrees, (cpp.ntrees, plan.ntrees)
    for itree in range(plan.ntrees):
        got = np.asarray(cpp.tree_variance[itree])
        want = py.tree_variance[itree]
        assert got.shape == want.shape, (itree, got.shape, want.shape)
        assert _allclose(got, want, rtol=1e-9, atol=1e-12), \
            (itree, got.shape, float(np.abs(got - want).max()))
