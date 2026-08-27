"""Tests comparing the fast_avar C++ ports against their python references.

TWO REFERENCES, because the C++ in src_lib/varmap.cpp ports two things -- and they are dispatched
from DIFFERENT test flags, each one next to the python it guards:

  - pirate_frb.slow_avar, under 'test --avar' (see pirate_frb/__main__.py): the convolver, the
    gridding+iterate sweep (which exercises SparseTile), a direct PfVariance check, and the
    end-to-end PfAvarApproximation.tree_variance. A few key methods, not every method, per the
    design.
  - pirate_frb.varmap.detrender_free, under 'test --varmap' (see varmap/tests.py run_all(), which
    is where the call lives): compute_detrender_free_varfine() and _varcoarse(). The python is
    fully tested in its own right by test_varfine(), so the ONLY thing needed on this side is that
    the two agree -- and whoever edits detrender_free.py, the reference, runs '--varmap'.

Tolerances: cross-language float results are NOT bit-exact (numpy uses pairwise/vectorized
summation, and BLAS for matmuls; the C++ uses sequential loops), so we use np.allclose tolerances
rather than epsabs=0. See test_cpp_detrender_free() for where the difference actually enters and
what it measures.
"""
import numpy as np

from .. import slow_avar
from . import (SparseTile, SparseTileTriple, PfVarianceConvolver,
               PfVariance, PfAvarApproximation,
               compute_detrender_free_varfine, compute_detrender_free_varcoarse)


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


def test_cpp_predict_dbits():
    """C++ SparseTile::predict_dbits() vs the python SparseTile._predict_dbits().

    Two sweeps, both exact (integers in and out, so no tolerance). Neither side allocates or
    iterates -- these are two closed forms -- so a case costs ~1us and we can afford both.

      1. Exhaustive over a small grid. Complete agreement there, not a sample.
      2. Random draws with much wider bounds. The C++ shifts left by up to 'kmax', and the
         exhaustive grid only reaches kmax=6, so half the input domain would otherwise never
         see the C++ at all. These bounds are far outside anything _predict_dbits_slow() could
         build tiles for, which is exactly why this test can use them and the python-side test
         cannot.

    This test is only about the port: that the prediction agrees with the real iteration is
    checked on the python side, by SparseTile.test_random_predict_dbits().
    """
    ncases = 0
    for kmax in range(0, 7):
        for f0 in range(0, 1 << 7):
            for nf in range(1, (1 << 7) - f0 + 1):
                got = SparseTile.predict_dbits(kmax, f0, nf)     # bound under the C++ name
                want = slow_avar.SparseTile._predict_dbits(kmax, f0, nf)
                assert got == want, (kmax, f0, nf, got, want)
                ncases += 1
    assert ncases == 57792, ncases     # tripwire: the sweep must not silently shrink

    for _ in range(10000):
        kmax, f0, nf = slow_avar.SparseTile._random_predict_dbits_args(nf_max=1 << 40,
                                                                      sum_max=1 << 41)
        got = SparseTile.predict_dbits(kmax, f0, nf)
        want = slow_avar.SparseTile._predict_dbits(kmax, f0, nf)
        assert got == want, (kmax, f0, nf, got, want)


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

    # gpu_valid=False / cdd2_kernel_required=False: PfAvarApproximation is pure-CPU (tree
    # structure + channel map only), so neither the config nor the plan needs a precompiled
    # cdd2 kernel for the rank.
    config = DedispersionConfig.make_random(max_toplevel_rank=5, max_early_triggers=2, gpu_valid=False)
    config.validate()
    plan = DedispersionPlan(config, cdd2_kernel_required=False)
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


def test_cpp_detrender_free(verbose=False):
    """C++ compute_detrender_free_{varfine,varcoarse}() vs python, on a random config.

    BOTH LEVELS ARE COMPARED, not just varcoarse. varcoarse is varfine followed by a per-tree
    max-reduction, and the extra C++ call is nearly free, so checking both turns "the answer is
    wrong" into "wrong before the reduction" or "wrong after it" -- very different bugs.

    Runs every iteration: measured, the PYTHON reference costs a median of 0.06 s and a max of
    0.13 s on a _random_config() draw, and the C++ side is negligible. No work-proxy cap is needed
    at that scale (notes/unit_tests.md item 4). If the draw is ever widened past
    max_toplevel_rank=7, revisit -- cost is roughly nfreq*r for the tile pass plus ndm*M*P for the
    lift, and an uncapped draw at CHORD scale is 15 seconds of python.

    THE TOLERANCE IS NOT epsabs=0, and it cannot be. The accumulation order is identical by
    construction (SdPlan's sd_vectors is a vector in insertion order, which is python's dict
    order), so the only difference is inside PfVarianceConvolver::variance(): python ends it with
    'rho @ A[:P,:d].T', a numpy matmul that goes through BLAS, against the C++'s sequential 'for
    lag' accumulation. That would be exact only if d == 1, and measured the tiles reaching emit()
    have nt in [3, 18] with d = min(nt, 32). Measured worst relative difference: 1.09e-14 over 200
    random draws, and at the shipped configs 5.0e-15 (chord_sb2_et.yml), 4.7e-15 (chime_sb2.yml),
    4.2e-15 (toy.yml). rtol=1e-11 is ~900x the observed worst case; a failure at that level is a
    bug, not a tolerance problem.
    """

    from ..pirate_pybind11 import DedispersionConfig
    from ..varmap import (compute_detrender_free_varfine as py_varfine,
                          compute_detrender_free_varcoarse as py_varcoarse)
    # _random_config() is what every randomized varmap test draws from, and pirate_frb/tests/
    # coverage.py reports its rates. Drawing here from a fresh make_random() instead would let this
    # test's population drift away from the one the coverage rows describe.
    from ..varmap.tests import _random_config

    config = _random_config()
    nfreq = int(config.get_total_nfreq())

    # NOT required to be positive -- this is defined against VarianceMap.apply(), which does not
    # require it either -- so let make_random_freq_variances() do whatever it does.
    v = np.asarray(config.make_random_freq_variances(noisy=True), dtype=np.float64)

    for (label, py_f, cpp_f) in [('varfine', py_varfine, compute_detrender_free_varfine),
                                 ('varcoarse', py_varcoarse, compute_detrender_free_varcoarse)]:
        want = py_f(config, v)
        got = cpp_f(config, v)
        assert len(got) == len(want), (label, len(got), len(want))
        for itree in range(len(want)):
            g, w = np.asarray(got[itree]), np.asarray(want[itree])
            assert g.shape == w.shape, (label, itree, g.shape, w.shape)
            assert _allclose(g, w, rtol=1e-11, atol=1e-13), \
                (label, itree, g.shape, float(np.abs(g - w).max()))

    if verbose:
        from ..utils import atomic_print
        atomic_print(f'    test_cpp_detrender_free(nfreq={nfreq},'
                     f' {int(config.num_dedispersion_trees)} trees): ok\n')
