"""Unit tests for pirate_frb.varmap.

TWO ENTRY POINTS, and the split is deliberate. run_all() is 'python -m pirate_frb test
--varmap': it needs no DedispersionPlan and no CUDA device, which is the same property that
lets an archived map be analyzed anywhere, and it runs in seconds. run_sweep_tests() is
'--vmbf': the brute-force sweep of varmap/brute_force.py, which is the one part of this
package that does need a device, and which runs a full sweep over every input channel per
test.

Three of these are load-bearing beyond the usual sense, because they are the only checks on
properties the scalable path assumes and cannot verify at runtime:

  - test_coarse_grain() compares the blockwise reduction against a dense one on a map small
    enough to form. At production scale the dense map is never built, so this is the only
    place the reduction is checked against something obviously correct.
  - test_sweep_streaming_coarse() does the same for the sweep's own reduction, which is a
    different algorithm again -- label-free, exploiting the two fixed shapes -- and is
    required to agree with coarse_grain() to the last bit.
  - test_index_arithmetic() compares alpha_to_beta_block() -- which indexes through the C++
    m_to_n -- against a label array derived independently in python. That is the tripwire on
    the multiplet ordering convention, which lives in C++ and would silently reinterpret
    every archived map if it changed.

Note there is deliberately NO test of the per-tree geometry itself. It comes verbatim from
the C++ DedispersionTree, which the DedispersionPlan constructor also uses, so any such test
would compare the C++ to itself. The tree constructor and its yaml round-trip are tested in
pirate_frb/tests/test_decode_argmax.py, where the plan-level yaml round-trip already lives.

Several tests cross-check against pirate_frb.slow_avar (varmap_eval, VarMapDistance), which
varmap supersedes. Those comparisons are deliberately temporary: they are what licenses
deleting the old code, and they go away with it.

The test_factored_* tests establish that ``A = Q @ mid @ W.T`` is what the map reports
through every accessor, and that the representation survives a file. They say NOTHING about
whether a factorization is any good: a hand-built map may claim to be semiorthogonal or to have
a pinned column and be neither, because the constructor carries those claims rather than
verifying them.

Whether a factorization is any good is what the test_svd / test_column_algebra /
test_reorthogonalize / test_basis_constructors / test_map_steps group is for, and there the
claims ARE checked -- numerically, against the matrices they describe, rather than read back
out of the object that set them. test_map_steps additionally pins the map-level steps to
varmap.lp bitwise, which is what makes lp.py's one-time equivalence gate a gate on the wrappers
too.

The test_lp_* tests check varmap.lp against things OTHER than itself wherever they can: the
Q-step's optimality against brute-force vertex enumeration, the free LP against the bounded
one, the majorization against the same sum accumulated the other way round, and a blocked
repair against an unblocked one over deliberately ragged tails. What they cannot check is
that the port computes the same numbers as the research code it came from -- that is a
one-time equivalence test living outside this repo, and it is what the defaults exist for.
"""

import contextlib
import dataclasses

import numpy as np

from .distance import YTRUE_FLOOR
from .VarianceMap import VarianceMap, make_tree
from .VarianceMultiMap import VarianceMultiMap
from ..utils import atomic_print


####################################   helpers   ####################################


def _make_test_config(toplevel_tree_rank, subband_counts, num_primary_trees=1,
                      num_early_triggers=0, max_width=4, nfreq=None):
    """A small DedispersionConfig with 'dm_downsampling: 0' (auto), which is what makes
    ndm_out == 2^(r-R) and hence makes the index convention apply."""

    from ..pirate_pybind11 import DedispersionConfig, PrimaryTree

    nfreq = nfreq if (nfreq is not None) else (1 << toplevel_tree_rank)
    nt_in = max(1 << (toplevel_tree_rank - 1), 32 << (num_primary_trees - 1))
    min_total_rank = toplevel_tree_rank - num_early_triggers - (1 if num_primary_trees > 1 else 0)

    config = DedispersionConfig()
    config.zone_nfreq = [nfreq]
    config.zone_freq_edges = [400.0, 800.0]
    config.time_sample_ms = 1.0
    config.dtype = np.float32
    config.toplevel_tree_rank = toplevel_tree_rank
    config.time_samples_per_chunk = nt_in
    config.frequency_subband_counts = subband_counts
    config.primary_trees = [
        PrimaryTree(num_early_triggers, max_width, 0, 1, 1 << min_total_rank, nt_in >> ipri)
        for ipri in range(num_primary_trees)
    ]
    config.beams_per_gpu = 1
    config.beams_per_batch = 1
    config.num_active_batches = 1
    config.validate()
    return config


def _obvious_labels(m, L):
    """The alpha -> beta map, computed the long way round: through the FULL-RESOLUTION DM of
    each output rather than through the two lines of arithmetic VarianceMap uses.

    A row's physical DM is a function of BOTH the coarse DM index d and the multiplet m,

        dm_full = d * 2^R + fine_dm(m) * 2^(R - level(subband(m)))

    and the notes' coarse-graining groups a contiguous range of dm_full. That is a genuinely
    different route to the same answer -- it mixes d and m, where alpha_to_beta_block() shifts
    d alone -- so an off-by-one in the multiplet decomposition shows up here and nowhere else.
    """

    M, N, P, R = m.nmultiplets, m.nsubbands, m.nprofiles, m.pf_rank

    # m -> (subband, fine DM), derived independently of the C++ FrequencySubbands.
    m_to_n, m_to_d, n_level = [], [], []
    for level, count in enumerate(m.subband_counts):
        for _ in range(int(count)):
            n = len(n_level)
            for d in range(1 << level):
                m_to_n.append(n)
                m_to_d.append(d)
            n_level.append(level)
    m_to_n = np.array(m_to_n, dtype=np.int64)
    m_to_d = np.array(m_to_d, dtype=np.int64)
    n_level = np.array(n_level, dtype=np.int64)

    alpha = np.arange(m.nalpha, dtype=np.int64)
    p = alpha % P
    mi = (alpha // P) % M
    d = alpha // (P * M)
    n = m_to_n[mi]
    dm_full = (d << R) + (m_to_d[mi] << (R - n_level[n]))

    return (dm_full >> L, n, p)


def _obvious_beta(m, L):
    """The composite label beta, from _obvious_labels().

    Tests that need a label array use THIS rather than alpha_to_beta_block(), so that they
    stay end-to-end checks: taking the labels from the code under test would make a
    transposed or shifted beta self-consistent, and invisible.
    """
    dm_key, n, p = _obvious_labels(m, L)
    return (dm_key * m.nsubbands + n) * m.nprofiles + p


def _random_map(config, itree, rng, *, nzero=0, dtype=np.float64):
    """A fine VarianceMap with a random nonnegative matrix, standing in for A_true.

    'nzero' rows are set to zero, so that the YTRUE_FLOOR path (outputs with no variance) is
    exercised -- it is not an edge case in practice, since a W=0 Detrender2d annihilates the
    DM=0 output.
    """

    tree = make_tree(config, itree)
    nalpha = tree.ndm_out * tree.frequency_subbands.M * tree.nprofiles
    A = rng.uniform(0.1, 2.0, size=(nalpha, config.get_total_nfreq())).astype(dtype)
    if nzero > 0:
        A[rng.choice(nalpha, size=nzero, replace=False)] = 0.0
    return VarianceMap.from_dense(config, itree, A, y_true='row_sums', is_admissible=True)


####################################   tests   ####################################


def test_index_arithmetic(r=8, subband_counts=(2,2,1), num_early_triggers=1):
    """alpha_to_beta_block() and group_sizes(), against label arrays built the long way."""

    config = _make_test_config(r, list(subband_counts),
                               num_early_triggers=num_early_triggers)
    rng = np.random.default_rng(1)

    for itree in range(config.num_dedispersion_trees):
        m = _random_map(config, itree, rng)
        R, N, P = m.pf_rank, m.nsubbands, m.nprofiles

        for L in range(R, m.tree_rank + 1):
            dm_key, n, p = _obvious_labels(m, L)

            # Ragged block boundaries on purpose: a latent bug in a block splitter once
            # survived a verification precisely because every test case divided exactly.
            beta = np.empty(m.nalpha, dtype=np.int64)
            start = 0
            for nb in (1, 7, 13, 1000, m.nalpha):
                while start < m.nalpha:
                    stop = min(start + nb, m.nalpha)
                    beta[start:stop] = m.alpha_to_beta_block(start, stop, L)
                    start = stop
                    if nb < 1000:
                        break
                if start >= m.nalpha:
                    break

            # The two derivations of the coarse DM index: dm_full >> L (which mixes d and m)
            # versus d >> (L-R). They agree for L >= R, and that identity is the whole reason
            # the grouping collapses to one integer.
            expect = (dm_key * N + n) * P + p
            assert np.array_equal(beta, expect), (itree, L)

            nbeta = (1 << (m.tree_rank - L)) * N * P
            assert beta.min() == 0 and beta.max() == nbeta - 1, (itree, L)

            sizes = m.group_sizes(L)
            assert sizes.shape == (nbeta,)
            assert np.array_equal(sizes, np.bincount(beta, minlength=nbeta)), (itree, L)
            assert sizes.sum() == m.nalpha

            # group_members() is the inverse, and estimate_distance() depends on it.
            for b in rng.choice(nbeta, size=min(8, nbeta), replace=False):
                members = m.group_members(int(b), L)
                assert np.array_equal(np.sort(members), np.flatnonzero(beta == b)), (L, b)

        # A fine map's default is the identity, not "coarse at L = R".
        assert np.array_equal(m.alpha_to_beta_block(0, m.nalpha), np.arange(m.nalpha))

    atomic_print(f'    test_index_arithmetic(r={r}, subbands={list(subband_counts)}): pass')



def test_constructor_validation(r=7, subband_counts=(2,1)):
    """The constructor's shape and flag checks, and immutability."""

    config = _make_test_config(r, list(subband_counts))
    rng = np.random.default_rng(2)
    m = _random_map(config, 0, rng)
    R = m.pf_rank

    def expect_raise(fn, needle):
        try:
            fn()
        except RuntimeError as e:
            assert needle in str(e), (needle, str(e))
            return
        raise AssertionError(f'expected a RuntimeError mentioning {needle!r}')

    # L out of range, and L set on an array that is not the coarse shape.
    Abar = m.coarse_grain(R).A
    expect_raise(lambda: VarianceMap.from_dense(config, 0, Abar, L=R-1), 'out of range')
    expect_raise(lambda: VarianceMap.from_dense(config, 0, Abar, L=m.tree_rank+1),
                 'out of range')
    expect_raise(lambda: VarianceMap.from_dense(config, 0, m.A, L=R), 'expected')
    expect_raise(lambda: VarianceMap.from_dense(config, 0, Abar), 'expected')
    expect_raise(lambda: VarianceMap.from_dense(config, 0, m.A, y_true=np.zeros(3)),
                 'FINE granularity')
    expect_raise(lambda: VarianceMap.from_dense(config, 0, Abar, L=R, y_true='row_sums'),
                 'not valid for a')
    expect_raise(lambda: VarianceMap.from_dense(config, 99, m.A), 'out of range')

    # Immutability is what makes is_admissible trustworthy, so it is enforced, not documented.
    try:
        m.is_admissible = False
        raise AssertionError('VarianceMap allowed an attribute assignment')
    except AttributeError:
        pass

    # ... and that extends to the arrays, which the accessors hand back as VIEWS. An in-place
    # edit through one of these rewrites the map (or, for a memmapped read, the file) behind
    # the flags' back, so they are read-only rather than merely documented as such. The Q-step
    # applies a relative floor to the rows it is given, which is exactly the shape of code
    # that would otherwise corrupt its own reference matrix.
    for (what, arr) in [('rows', m.rows(0, 4)), ('cols', m.cols(0, 4)), ('dense', m.dense()),
                        ('A', m.A), ('y_true', m.y_true), ('m_to_n', m.m_to_n),
                        ('row_sums', m.row_sums())]:
        try:
            arr[0] = 0
            raise AssertionError(f'VarianceMap.{what} is writable')
        except ValueError:
            pass

    # The read-only wrapper must be idempotent, or replace() builds a chain of views and
    # "the matrix was not copied" stops being checkable.
    assert m.replace().A is m.A

    # is_coarse_grained and L are one fact; neither is derivable from nbeta.
    assert (m.L is None) and (not m.is_coarse_grained) and (m.nbeta == m.nalpha)
    c = m.coarse_grain(R)
    assert c.is_coarse_grained and (c.L == R)
    if m.nmultiplets == m.nsubbands:
        assert c.nbeta == m.nalpha, 'M == N, so coarse-at-R and fine have the same nbeta'

    atomic_print(f'    test_constructor_validation(r={r}): pass')


def test_coarse_grain(r=8, subband_counts=(2,2,1), num_early_triggers=1):
    """coarse_grain() against a dense reduction, and coarse-to-coarser against fine-to-coarse.

    This is the one property the scalable path trusts and cannot check at runtime: at
    production scale the dense map is never formed.
    """

    from ..slow_avar import varmap_eval as ve      # temporary cross-check; see module docstring

    config = _make_test_config(r, list(subband_counts),
                               num_early_triggers=num_early_triggers)
    rng = np.random.default_rng(3)

    for itree in range(config.num_dedispersion_trees):
        m = _random_map(config, itree, rng, nzero=2)
        R = m.pf_rank

        for L in range(R, m.tree_rank + 1):
            labels = _obvious_beta(m, L)
            nbeta = (1 << (m.tree_rank - L)) * m.nsubbands * m.nprofiles

            # Dense reference, built the obvious way.
            ref = np.zeros((nbeta, m.nfreq))
            np.maximum.at(ref, labels, np.asarray(m.A, dtype=np.float64))

            c = m.coarse_grain(L)
            assert c.shape == (nbeta, m.nfreq)
            assert np.array_equal(c.A, ref), (itree, L)

            # ... and against the old streaming reducer, which is what the published numbers
            # were computed from.
            old_Abar, old_y = ve.reduce_map(np.asarray(m.A, dtype=np.float64), labels, nbeta)
            assert np.array_equal(c.A, old_Abar), (itree, L)
            assert np.array_equal(c.y_true, old_y), (itree, L)

            # y_true is the TRUE row sums at FINE granularity, carried unchanged.
            assert np.array_equal(c.y_true, m.y_true)
            assert c.is_admissible == m.is_admissible
            assert c.check_ref_covers_y_true() >= 1.0

            # Coarsening is nested, so coarse-to-coarser must agree with fine-to-coarse. This
            # is what lets a sweep run once at the finest L that fits and coarsen down.
            for L2 in range(L+1, m.tree_rank + 1):
                assert np.array_equal(c.coarse_grain(L2).A, m.coarse_grain(L2).A), (L, L2)

            # ... and cannot go the other way.
            try:
                c.coarse_grain(L)
                raise AssertionError('coarse_grain() accepted L <= self.L')
            except RuntimeError as e:
                assert 'already coarse' in str(e)

            # lift() is the inverse of the row duplication, not of the max.
            lifted = c.lift()
            assert (not lifted.is_coarse_grained) and lifted.shape == (m.nalpha, m.nfreq)
            assert np.array_equal(lifted.A, c.A[labels])

    atomic_print(f'    test_coarse_grain(r={r}, subbands={list(subband_counts)}): pass')


def test_distance(r=8, subband_counts=(2,2,1)):
    """get_distance() against slow_avar.varmap_eval, in both the fine and the coarse case."""

    from ..slow_avar import varmap_eval as ve      # temporary cross-check; see module docstring

    config = _make_test_config(r, list(subband_counts))
    rng = np.random.default_rng(4)

    for nzero in (0, 3):
        true = _random_map(config, 0, rng, nzero=nzero)
        A_true = np.asarray(true.A, dtype=np.float64)

        assert true.nscored == true.nalpha - nzero

        # --- fine case: an admissible dense approximation ---
        A_approx = A_true * rng.uniform(1.0, 1.6, size=A_true.shape)
        approx = VarianceMap.from_dense(config, 0, A_approx, y_true=true.y_true,
                                        is_admissible=True)
        D = approx.get_distance()
        D_old = ve.evaluate(A_true, ve.DenseApprox(A_approx), inflate=False)['D']
        assert abs(D - D_old) <= 1.0e-12 * max(1.0, abs(D_old)), (D, D_old)

        # The summand of D, materialized. Rows with no variance come back as nan, not 0: a 0
        # would understate the mean, and this is what pins that convention.
        #
        # get_distance() accumulates in blocks while this materializes an nalpha array, so the
        # two really are separate implementations. They agree bitwise in practice, but the
        # tolerance is what is contractual -- two different summation orders are not required
        # to round identically.
        rd = approx.get_row_distances()
        assert np.count_nonzero(np.isnan(rd)) == nzero
        assert abs(np.nanmean(rd) - D) <= 1.0e-13 * max(1.0, abs(D)), (np.nanmean(rd), D)

        # --- coarse case: a coarse-assigned approximation ---
        L = true.pf_rank + 1
        ref = true.coarse_grain(L)
        capprox = ref.inflated(1.5)
        Dc = capprox.get_distance()

        labels = _obvious_beta(true, L)
        Dc_old = ve.evaluate_reduced(np.asarray(ref.A), true.y_true, labels,
                                     ve.DenseApprox(np.asarray(capprox.A)),
                                     inflate=False)['D']
        assert abs(Dc - Dc_old) <= 1.0e-12 * max(1.0, abs(Dc_old)), (Dc, Dc_old)

        # Scoring the coarse map and scoring its lift are the same computation, and D is what
        # makes that true: y_approx is constant across a group.
        assert capprox.lift().get_distance() == Dc

        # The block size must not change any of the three answers, at a RAGGED tail as well as
        # a dividing one. A latent bug in a block splitter once survived a verification
        # precisely because every test case happened to divide exactly.
        assert np.array_equal(capprox.alpha_to_beta_block(0, true.nalpha), labels)

        # The per-row values are exact regardless of blocking (each is one division and one
        # f()); only their SUM depends on the block size, and only at the few-ulp level, since
        # a different block size is a different pairwise summation grouping.
        rd_ref = capprox.get_row_distances()
        saved = VarianceMap._ALPHA_BLOCK
        try:
            for blk in (7, 64, 997, 4096, true.nalpha):
                VarianceMap._ALPHA_BLOCK = blk
                assert abs(capprox.get_distance() - Dc) <= 1.0e-13 * abs(Dc), blk
                assert np.array_equal(capprox.get_row_distances(), rd_ref,
                                      equal_nan=True), blk
        finally:
            VarianceMap._ALPHA_BLOCK = saved

        # --- get_distance() is strict about what it will score ---
        try:
            capprox.replace(is_admissible=False).get_distance()
            raise AssertionError('get_distance() scored an uncertified map')
        except RuntimeError as e:
            assert 'not certified admissible' in str(e)

        try:
            VarianceMap.from_dense(config, 0, A_approx, is_admissible=True).get_distance()
            raise AssertionError('get_distance() scored a map with no y_true')
        except RuntimeError as e:
            assert 'y_true is unavailable' in str(e)

    atomic_print(f'    test_distance(r={r}, subbands={list(subband_counts)}): pass')


def test_admissibility(r=8, subband_counts=(2,1)):
    """measure_admissibility(): the coarse/fine theorem, the sign conventions, and inflation."""

    config = _make_test_config(r, list(subband_counts))
    rng = np.random.default_rng(5)

    true = _random_map(config, 0, rng, nzero=2)
    L = true.pf_rank + 1
    ref = true.coarse_grain(L)

    # A coarse-assigned approximation. Measuring it against the COARSE ref and measuring its
    # lift against the FINE true map must agree -- that is the pivot identity, and it is what
    # makes scoring possible at CHORD scale, where the fine map does not exist.
    capprox = ref.inflated(1.5)
    rc = capprox.measure_admissibility(ref)
    rf = capprox.lift().measure_admissibility(true)
    assert rc.admissible and rf.admissible
    assert rc.max_r == rf.max_r, (rc.max_r, rf.max_r)
    assert rc.nviol == 0 and rf.nviol == 0
    assert rc.vmap.is_admissible and (rc.vmap.A is capprox.A)

    # A single planted underestimate: infinite D, but a small inflation fixes it, and that is
    # the number that distinguishes "nearly usable" from "hopeless".
    bad = np.array(capprox.A)
    bad[3, 7] = ref.A[3, 7] * 0.9
    bad = capprox.replace(A=bad, is_admissible=False, history_record=dict(step='test'))
    rb = bad.measure_admissibility(ref, inflate=True)
    assert (not rb.admissible) and (rb.max_r > 1.0) and np.isfinite(rb.max_r)
    assert rb.argmax_r == (3, 7), rb.argmax_r
    assert rb.nviol == 1 and rb.viol_rows == 1 and (3 in rb.worst_rows)
    assert np.isfinite(rb.D_inflated) and (rb.D_inflated >= capprox.get_distance())
    assert not rb.vmap.is_admissible

    # SIGNS. A non-positive entry where ref is positive is an underestimate NO rescaling
    # repairs, and is reported as max_r = inf rather than raised on: scoring a signed
    # candidate is the main reason this method exists.
    neg = np.array(capprox.A)
    neg[5, 2] = -1.0
    neg = capprox.replace(A=neg, is_admissible=False, history_record=dict(step='test'))
    rn = neg.measure_admissibility(ref)
    assert np.isinf(rn.max_r) and (rn.argmax_r == (5, 2)) and (rn.nneg_self == 1)

    # ref <= 0 maps to ratio 0, so such an element can never become the argmax (0/0 included).
    zref = np.array(ref.A)
    zref[1, :] = 0.0
    zref = ref.replace(A=zref, history_record=dict(step='test'))
    zapp = capprox.replace(A=np.where(np.arange(capprox.nbeta)[:,None] == 1, 0.0,
                                      np.asarray(capprox.A)),
                           history_record=dict(step='test'))
    rz = zapp.measure_admissibility(zref)
    assert rz.admissible and (rz.argmax_r[0] != 1)

    # Block size must not change the answer, including at a ragged tail.
    for nb in (1, 3, 7, ref.nbeta):
        r2 = capprox.measure_admissibility(ref, block_rows=nb)
        assert (r2.max_r, r2.argmax_r, r2.nviol) == (rc.max_r, rc.argmax_r, rc.nviol), nb

    # Mixing granularities is not a valid test, and says so.
    try:
        capprox.measure_admissibility(true)
        raise AssertionError('measure_admissibility() compared a coarse self with a fine ref')
    except RuntimeError as e:
        assert 'coarse-graining' in str(e) or 'shape mismatch' in str(e)

    atomic_print(f'    test_admissibility(r={r}, subbands={list(subband_counts)}): pass')


def test_estimate_distance(r=8, subband_counts=(2,2,1)):
    """estimate_distance(): exact at frac=1, and unbiased (not group-weighted) below it."""

    config = _make_test_config(r, list(subband_counts))
    rng = np.random.default_rng(6)

    true = _random_map(config, 0, rng, nzero=2)
    L = true.pf_rank + 1
    approx = true.coarse_grain(L).inflated(1.4)
    D = approx.get_distance()

    full = approx.estimate_distance(frac=1.0)
    assert full.nsampled == approx.nbeta and full.frac_sampled == 1.0
    assert full.stderr == 0.0, full.stderr
    assert abs(full.D - D) <= 1.0e-12 * abs(D), (full.D, D)
    assert full.nscored == true.nscored

    # Groups are NOT all the same size (a subband at level l contributes 2^(L-R) * 2^l), and D
    # is a mean over FINE rows -- so a plain mean over sampled groups would be biased. Check
    # that the sizes really do vary here, or the test would not be testing anything.
    sizes = approx.group_sizes()
    assert sizes.min() != sizes.max(), 'this config does not exercise the group-size weighting'

    # A subsample lands within a few standard errors, over many draws.
    nbad = 0
    for _ in range(40):
        e = approx.estimate_distance(frac=0.2, rng=rng)
        if abs(e.D - D) > 4.0 * e.stderr + 1.0e-12:
            nbad += 1
    assert nbad <= 2, f'{nbad}/40 subsamples were more than 4 sigma from the exact D'

    # Passing 'groups' back is what makes a PAIRED comparison possible: two arms on the same
    # subset have a far better determined ratio than either value.
    e1 = approx.estimate_distance(frac=0.1, rng=rng)
    e2 = approx.inflated(1.01).replace(is_admissible=True).estimate_distance(groups=e1.groups)
    assert np.array_equal(e1.groups, e2.groups) and (e2.D > e1.D)

    atomic_print(f'    test_estimate_distance(r={r}): pass')


def test_check_ref_covers_y_true(r=7, subband_counts=(2,1)):
    """The one runtime check on the property the scalable path rests on."""

    config = _make_test_config(r, list(subband_counts))
    rng = np.random.default_rng(7)
    true = _random_map(config, 0, rng)
    ref = true.coarse_grain(true.pf_rank)

    assert ref.check_ref_covers_y_true() >= 1.0

    # A MEAN where a max was intended is the bug class this catches outright.
    labels = _obvious_beta(true, ref.L)
    mean = np.zeros_like(np.asarray(ref.A))
    np.add.at(mean, labels, np.asarray(true.A, dtype=np.float64))
    mean /= ref.group_sizes()[:,None]
    broken = ref.replace(A=mean, history_record=dict(step='test'))

    try:
        broken.check_ref_covers_y_true()
        raise AssertionError('check_ref_covers_y_true() accepted a mean-reduced map')
    except RuntimeError as e:
        assert 'max-envelope cannot do this' in str(e), str(e)

    atomic_print(f'    test_check_ref_covers_y_true(r={r}): pass')


def test_multimap(r=8, subband_counts=(2,1), num_primary_trees=2, num_early_triggers=1):
    """VarianceMultiMap: one map per tree, for every tree, sharing one config object."""

    config = _make_test_config(r, list(subband_counts),
                               num_primary_trees=num_primary_trees,
                               num_early_triggers=num_early_triggers)
    rng = np.random.default_rng(8)
    ntrees = int(config.num_dedispersion_trees)
    assert ntrees == num_primary_trees * (num_early_triggers + 1)

    maps = [_random_map(config, i, rng) for i in range(ntrees)]
    vmm = VarianceMultiMap(config, maps, provenance=dict(algorithm='test'))

    assert (len(vmm) == ntrees) and (vmm[0] is maps[0])
    assert [m.itree for m in vmm] == list(range(ntrees))

    # The legal range R <= L <= r differs per tree, so L may be a per-tree sequence.
    Ls = [m.pf_rank for m in vmm]
    cvmm = vmm.coarse_grain(Ls)
    for (i, m) in enumerate(cvmm):
        assert m.L == Ls[i] and np.array_equal(m.A, maps[i].coarse_grain(Ls[i]).A)

    v = rng.uniform(0.5, 1.5, size=config.get_total_nfreq())
    ys = vmm.apply(v)
    assert len(ys) == ntrees
    assert np.allclose(ys[0], np.asarray(maps[0].A, dtype=np.float64) @ v)

    res = cvmm.measure_admissibility(cvmm)
    assert all(x.admissible for x in res) and (max(x.max_r for x in res) <= 1.0)

    # A short list is a bug in whatever assembled it, not a subset.
    try:
        VarianceMultiMap(config, maps[:-1])
        raise AssertionError('VarianceMultiMap accepted a partial tree list')
    except RuntimeError as e:
        assert 'one map per tree' in str(e)

    atomic_print(f'    test_multimap(r={r}, ntrees={ntrees}): pass')


def test_dense_float32(r=7, subband_counts=(1,)):
    """A float32 stored matrix: rows() promotes, and nothing downstream sees the difference."""

    config = _make_test_config(r, list(subband_counts))
    rng = np.random.default_rng(9)
    m32 = _random_map(config, 0, rng, dtype=np.float32)
    m64 = m32.replace(A=np.asarray(m32.A, dtype=np.float64))

    assert m32.A.dtype == np.float32 and m32.rows(0, 4).dtype == np.float64
    assert m32.nbytes() == m64.nbytes() // 2
    assert np.array_equal(m32.coarse_grain(m32.pf_rank).A, m64.coarse_grain(m64.pf_rank).A)

    atomic_print(f'    test_dense_float32(r={r}): pass')


class _eager_ctx:
    """Wraps an already-read object as a no-op context manager, so that the eager and
    memmapped readers can be driven by the same test body (open_asdf() is a context manager;
    from_asdf() is not)."""

    def __init__(self, obj):
        self.obj = obj

    def __enter__(self):
        return self.obj

    def __exit__(self, *exc):
        return False


def _factored_map(config, itree, rng, K=5, *, L=None, mid='full', nbeta=None, **kwargs):
    """A factored VarianceMap with random factors, and the dense product it stands for.

    Signed on purpose: campaign 2 dropped the nonnegativity constraint, so a factored map is
    routinely signed and nothing here may assume otherwise.
    """

    tree = make_tree(config, itree)
    fs = tree.frequency_subbands
    if nbeta is None:
        nbeta = tree.ndm_out * fs.M * tree.nprofiles
    nfreq = config.get_total_nfreq()

    Q = rng.normal(size=(nbeta, K))
    W = rng.normal(size=(nfreq, K))
    M = np.eye(K) if (mid == 'identity') else rng.normal(size=(K, K))
    m = VarianceMap.from_factors(config, itree, Q, W, mid=M, L=L, **kwargs)
    return m, Q @ M @ W.T


def test_factored_algebra(r=7, subband_counts=(2,1), K=5):
    """The product identity, and every accessor that has a factored branch.

    Nothing here is about whether a factorization is any GOOD -- only that
    ``A = Q @ mid @ W.T`` is what the map reports through every route a consumer has.
    """

    config = _make_test_config(r, list(subband_counts))
    rng = np.random.default_rng(20)

    for mid_kind in ('identity', 'full'):
        m, ref = _factored_map(config, 0, rng, K=K, mid=mid_kind)
        assert m.is_factored and (m.factor_rank == K)
        assert m.A is None and (m.shape == ref.shape)

        # Ragged block boundaries on purpose, in both directions.
        for nb in (1, 3, 7, 997, m.nbeta):
            for start in range(0, m.nbeta, nb):
                stop = min(start + nb, m.nbeta)
                assert np.allclose(m.rows(start, stop), ref[start:stop]), (mid_kind, nb)
        for nc in (1, 5, 13, m.nfreq):
            for start in range(0, m.nfreq, nc):
                stop = min(start + nc, m.nfreq)
                assert np.allclose(m.cols(start, stop), ref[:, start:stop]), (mid_kind, nc)

        assert np.allclose(m.dense(), ref)

        # apply() must never form A, and row_sums() is one K-vector contraction. Both are
        # different summation orders from the dense route, so the bar is a tolerance -- the
        # same lesson as get_distance()'s block-size sensitivity.
        #
        # Scale the bar by the size of the TERMS, not of the result. A factored map is
        # routinely signed (campaign 2 dropped the nonnegativity constraint), so a row sum
        # can cancel to near zero, and a relative bar would then demand agreement to a
        # precision neither route has.
        scale = float(np.abs(ref).sum(axis=1).max())
        v = rng.normal(size=m.nfreq)
        assert np.max(np.abs(m.apply(v) - ref @ v)) <= 1e-12 * scale * np.abs(v).max()
        assert np.max(np.abs(m.row_sums() - ref.sum(axis=1))) <= 1e-12 * scale

        # Descriptive cost figures, and the block sizer that has no page-alignment floor
        # because factored columns are computed rather than read.
        assert m.nbytes() == 8 * (m.nbeta*K + K*K + m.nfreq*K)
        assert m.apply_cost() == K*m.nfreq + K*K + np.count_nonzero(np.asarray(m.Q))
        assert m.apply_cost() < m.nbeta * m.nfreq, 'K << nfreq should be cheaper than dense'
        assert m.default_block_cols(1 << 10) == max(1, (1 << 10) // (8 * m.nbeta))

    atomic_print(f'    test_factored_algebra(r={r}, K={K}): pass')


def test_factored_equivalence(r=7, subband_counts=(2,1), K=4):
    """A dense map and a factored map that densify to the SAME matrix must agree everywhere.

    This is the cheapest way to catch a code path that still reaches for ``self.A``: every
    consumer below goes through rows() / row_sums(), so any that did not would diverge here.
    The factored map is genuinely rank-deficient, so the two representations are not
    trivially the same object.
    """

    config = _make_test_config(r, list(subband_counts))
    rng = np.random.default_rng(21)

    # Nonnegative factors and a nonnegative 'mid', so the product is positive: the scoring
    # paths below are about VARIANCES, and get_distance() is only meaningful on one. (The
    # signed case is covered by test_factored_algebra, which touches no scoring.)
    tree = make_tree(config, 0)
    fs = tree.frequency_subbands
    nbeta = tree.ndm_out * fs.M * tree.nprofiles
    Q = rng.uniform(0.5, 1.5, size=(nbeta, K))
    W = rng.uniform(0.5, 1.5, size=(config.get_total_nfreq(), K))
    mid = np.eye(K) + rng.uniform(0.0, 0.2, size=(K, K))
    ref = Q @ mid @ W.T
    assert np.all(ref > 0), 'this test needs a positive matrix'

    y = ref.sum(axis=1)
    fac = VarianceMap.from_factors(config, 0, Q, W, mid=mid, y_true=y, is_admissible=True)
    den = VarianceMap.from_dense(config, 0, ref, y_true=y, is_admissible=True)

    assert fac.is_factored and (not den.is_factored)
    assert np.allclose(fac.row_sums(), den.row_sums())
    assert abs(fac.get_distance() - den.get_distance()) <= 1e-12

    # Scoring and admissibility, factored-vs-dense in both roles.
    for (a, b) in ((fac, den), (den, fac), (fac, fac)):
        res = a.inflated(1.5).measure_admissibility(b)
        assert res.admissible and (abs(res.max_r - 1/1.5) < 1e-9), res.max_r

    # coarse_grain() of a factored map: a max-envelope is NONLINEAR, so the result is dense
    # by construction and must equal the envelope of the densified original.
    for L in range(fac.pf_rank, fac.tree_rank + 1):
        cf, cd = fac.coarse_grain(L), den.coarse_grain(L)
        assert not cf.is_factored, 'a max-envelope cannot stay factored'
        assert np.allclose(np.asarray(cf.A), np.asarray(cd.A)), L

    atomic_print(f'    test_factored_equivalence(r={r}, K={K}): pass')


def test_factored_transformations(r=7, subband_counts=(2,1), K=4):
    """inflated() and lift() keep the factorization; both agree with the dense answer."""

    config = _make_test_config(r, list(subband_counts))
    rng = np.random.default_rng(22)

    m, ref = _factored_map(config, 0, rng, K=K, Q_is_semiorthogonal=True,
                           W_is_semiorthogonal=True, pinned_columns=[1])

    # inflated() scales 'mid', so neither factor is touched and both flags survive.
    inf = m.inflated(2.5)
    assert inf.is_factored and (inf.factor_rank == K)
    assert np.allclose(inf.dense(), 2.5 * ref)
    assert np.array_equal(np.asarray(inf.Q), np.asarray(m.Q))
    assert np.array_equal(np.asarray(inf.W), np.asarray(m.W))
    assert inf.Q_is_semiorthogonal and inf.W_is_semiorthogonal
    assert np.array_equal(inf.pinned_columns, m.pinned_columns)

    # lift() duplicates rows of Q: nalpha*K rather than nalpha*nfreq, and W is untouched.
    L = m.pf_rank + 1
    nb = (1 << (m.tree_rank - L)) * m.nsubbands * m.nprofiles
    c, _ = _factored_map(config, 0, rng, K=K, L=L, nbeta=nb)
    lifted = c.lift()
    assert lifted.is_factored and (not lifted.is_coarse_grained)
    assert np.allclose(lifted.dense(), c.dense()[c.alpha_to_beta_block(0, c.nalpha)])
    assert np.array_equal(np.asarray(lifted.W), np.asarray(c.W))
    # Duplicating rows destroys any column orthogonality Q had, so the flag must not survive.
    assert not lifted.Q_is_semiorthogonal

    # replace() switches representation, which is what lets coarse_grain() return a dense map
    # from a factored one; and a replaced factor drops its unrestated semiorthogonality claim.
    d = m.replace(A=m.dense())
    assert (not d.is_factored) and (d.Q is None) and (d.pinned_columns is None)
    assert np.allclose(np.asarray(d.A), ref)
    assert d.replace(Q=np.asarray(m.Q), W=np.asarray(m.W), mid=np.asarray(m.mid)).is_factored

    q2 = m.replace(Q=np.asarray(m.Q) * 2.0)
    assert (not q2.Q_is_semiorthogonal) and q2.W_is_semiorthogonal
    assert m.replace(Q=np.asarray(m.Q), Q_is_semiorthogonal=True).Q_is_semiorthogonal

    atomic_print(f'    test_factored_transformations(r={r}, K={K}): pass')


def test_factored_validation(r=7, subband_counts=(2,1), K=4):
    """Constructor rejections, and read-only enforcement on the factors.

    Only STRUCTURE is enforced -- shapes, a consistent K, dtypes, indices in range. The
    semiorthogonality flags and the pinned set are carried, not verified, so a map that
    claims them falsely is accepted here and is the steps' problem.
    """

    config = _make_test_config(r, list(subband_counts))
    rng = np.random.default_rng(23)
    m, _ = _factored_map(config, 0, rng, K=K)
    Q, mid, W = np.asarray(m.Q), np.asarray(m.mid), np.asarray(m.W)
    nb, nf = m.nbeta, m.nfreq

    def expect_raise(fn, needle):
        try:
            fn()
        except RuntimeError as e:
            assert needle in str(e), (needle, str(e))
            return
        raise AssertionError(f'expected a RuntimeError mentioning {needle!r}')

    F = lambda **kw: VarianceMap(config, 0, **kw)
    expect_raise(lambda: F(A=m.dense(), Q=Q, W=W), 'exactly')
    expect_raise(lambda: F(Q=Q), 'BOTH Q and W')
    expect_raise(lambda: F(W=W), 'BOTH Q and W')
    expect_raise(lambda: F(), 'either a dense matrix A')
    expect_raise(lambda: F(Q=Q, W=W[:, :K-1]), 'both are the factorization rank')
    expect_raise(lambda: F(Q=Q[:-1], W=W), 'Q has shape')
    expect_raise(lambda: F(Q=Q, W=W[:-1]), 'W has shape')
    expect_raise(lambda: F(Q=Q, W=W, mid=np.eye(K+1)), 'mid has shape')
    expect_raise(lambda: F(Q=Q.astype(np.float16), W=W), 'dtype')
    expect_raise(lambda: F(Q=Q, W=W, pinned_columns=[K]), 'must lie in')
    expect_raise(lambda: F(Q=Q, W=W, pinned_columns=[-1]), 'must lie in')
    expect_raise(lambda: F(Q=Q, W=W, pinned_columns=[1, 1]), 'duplicates')
    expect_raise(lambda: F(A=m.dense(), pinned_columns=[0]), 'meaningless for a dense map')
    expect_raise(lambda: F(A=m.dense(), Q_is_semiorthogonal=True), 'meaningless for a dense')

    # 'mid' defaults to the identity, and an empty pinned set is the default.
    plain = VarianceMap.from_factors(config, 0, Q, W)
    assert np.array_equal(np.asarray(plain.mid), np.eye(K))
    assert plain.pinned_columns.size == 0

    # The factors are read-only, for the same reason A / y_true / m_to_n are: they may be
    # views, and a write through one corrupts the map (or the mapped file).
    for (what, arr) in [('Q', m.Q), ('mid', m.mid), ('W', m.W),
                        ('pinned_columns', m.pinned_columns), ('rows', m.rows(0, 2))]:
        try:
            arr[0] = 0
            raise AssertionError(f'VarianceMap.{what} is writable')
        except ValueError:
            pass

    # Flags and pinned columns are DATA: a false claim is accepted, because verifying it is
    # the business of the steps that establish it.
    lying = VarianceMap.from_factors(config, 0, Q, W, Q_is_semiorthogonal=True,
                                     pinned_columns=[0, 2])
    assert lying.Q_is_semiorthogonal and (list(lying.pinned_columns) == [0, 2])

    atomic_print(f'    test_factored_validation(r={r}, K={K}): pass')


@contextlib.contextmanager
def _open_one(path, itree):
    """One tree of a memmapped read. VarianceMap has no open_asdf() of its own -- the scoped
    opener lives on VarianceMultiMap -- and these test configs are single-tree, so going
    through it is the same thing."""

    with VarianceMultiMap.open_asdf(path) as vmm:
        yield vmm[itree]


def _corrupt(path, out, fn):
    """Copy the variance-map file at 'path' to 'out', with fn(root) applied to its
    'variance_multimap' block first.

    This is how the reader's tripwires are tested: each one guards against a file written
    by something that got a convention wrong, and the only way to produce such a file here
    is to break a good one.
    """

    import asdf
    from .asdf_io import ROOT_KEY

    with asdf.open(path, lazy_load=False, memmap=False) as af:
        root = dict(af[ROOT_KEY])
        root['trees'] = [dict(d) for d in root['trees']]
        fn(root)
        asdf.AsdfFile({ROOT_KEY: root}).write_to(out)


def test_asdf_io(r=8, subband_counts=(2,2,1), num_primary_trees=2, num_early_triggers=1):
    """The file format: every representation round-trips, and every tripwire fires.

    The tripwires matter more than the round-trip. The archived library is hundreds of GiB
    that cannot be regenerated cheaply, so a reader that silently reinterprets a file --
    because the multiplet ordering convention drifted, or because a flag and the array it
    describes disagree -- is the expensive failure mode, and each check below is one of
    those turned into an exception.
    """

    import dataclasses
    import os
    import shutil
    import tempfile

    from .asdf_io import FORMAT_VERSION

    config = _make_test_config(r, list(subband_counts),
                               num_primary_trees=num_primary_trees,
                               num_early_triggers=num_early_triggers)
    rng = np.random.default_rng(10)
    ntrees = int(config.num_dedispersion_trees)
    tmp = tempfile.mkdtemp()

    def expect_raise(fn, needle):
        try:
            fn()
        except RuntimeError as e:
            assert needle in str(e), (needle, str(e))
            return
        raise AssertionError(f'expected a RuntimeError mentioning {needle!r}')

    try:
        path = os.path.join(tmp, 'vm.asdf')

        # Every representation the dense path can produce: fine/coarse, certified or not,
        # y_true present or absent, float64 or float32. (The factored half of that product
        # is not reachable yet; the reader's refusal of it is checked below.)
        maps = [_random_map(config, i, rng, nzero=2) for i in range(ntrees)]
        maps[1] = maps[1].coarse_grain(maps[1].pf_rank + 1)
        maps[2] = maps[2].replace(is_admissible=False)
        maps[3] = maps[3].replace(y_true=None, A=np.asarray(maps[3].A, dtype=np.float32))

        prov = dict(algorithm='test', ntime=np.int64(1024), overrides=['a', 'b'],
                    nested=dict(host='here', seconds=1.5))
        vmm = VarianceMultiMap(config, maps, provenance=prov)
        vmm.write_asdf(path)

        for eager in (True, False):
            ctx = (_eager_ctx(VarianceMultiMap.from_asdf(path)) if eager
                   else VarianceMultiMap.open_asdf(path))
            with ctx as v2:
                assert len(v2) == ntrees
                assert v2.provenance == {'algorithm': 'test', 'ntime': 1024,
                                         'overrides': ['a', 'b'],
                                         'nested': {'host': 'here', 'seconds': 1.5}}

                # The inputs survive as yaml and re-parse into one shared object per file.
                assert int(v2.config.toplevel_tree_rank) == r
                assert all(m.config is v2.config for m in v2)
                assert v2.detrender is None

                for (i, m) in enumerate(v2):
                    w = maps[i]
                    assert m.itree == i and m.L == w.L
                    assert m.is_coarse_grained == w.is_coarse_grained
                    assert m.is_admissible == w.is_admissible
                    assert m.shape == w.shape and m.nalpha == w.nalpha
                    assert np.asarray(m.A).dtype == np.asarray(w.A).dtype, i
                    assert np.array_equal(np.asarray(m.A), np.asarray(w.A)), i
                    assert (m.y_true is None) == (w.y_true is None)
                    if w.y_true is not None:
                        assert np.array_equal(m.y_true, w.y_true), i
                    assert m.history == w.history, i

                    # The geometry comes off the file's tree yaml, not from re-deriving it.
                    assert np.array_equal(m.m_to_n, w.m_to_n)
                    assert (m.tree_rank, m.pf_rank, m.nmultiplets, m.nsubbands,
                            m.nprofiles, m.gamma, m.early_trigger_level) == \
                        (w.tree_rank, w.pf_rank, w.nmultiplets, w.nsubbands,
                         w.nprofiles, w.gamma, w.early_trigger_level)

                # Scoring works on a map that has been through the file, which is the point
                # of storing y_true at fine granularity.
                assert abs(v2[1].get_distance() - maps[1].get_distance()) <= 1.0e-14

                if not eager:
                    # Uncompressed blocks are what make this possible, and memmapping is the
                    # scale path -- an asdf upgrade that changed its defaults would silently
                    # turn every large read into a full materialization.
                    chain, a = [], np.asarray(v2[0].A)
                    while a is not None:
                        chain.append(a)
                        a = getattr(a, 'base', None)
                    assert any(isinstance(x, np.memmap) for x in chain), \
                        [type(x) for x in chain]

        # A frozen dataclass in a history record -- which is how a step will carry the
        # LpConfig it ran under. asdf cannot represent one, and the write it breaks is the
        # write of the WHOLE map, so without the conversion a long run cannot save its
        # result. The round trip is asymmetric on purpose: a dataclass out, a dict back.
        @dataclasses.dataclass(frozen=True)
        class _Cfg:
            clip_rel: float = 1.0e-8
            cuts: tuple = (1, 2, 3)

        dpath = os.path.join(tmp, 'dc.asdf')
        rec = dict(step='qstep', cfg=_Cfg(), D=np.float64(0.25), rows=np.int64(7))
        maps[0].replace(history_record=rec).write_asdf(dpath)
        got = VarianceMap.from_asdf(dpath, 0).history[-1]
        assert got == {'step': 'qstep', 'cfg': {'clip_rel': 1.0e-8, 'cuts': [1, 2, 3]},
                       'D': 0.25, 'rows': 7}, got

        # A memmap-backed matrix. This is what a large write actually looks like -- a map
        # accumulated on disk, or one read back from open_asdf() -- and asdf refuses ndarray
        # SUBCLASSES outright, so without a base-class view the scale path fails at the one
        # size where it matters and nowhere else.
        npy = os.path.join(tmp, 'A.npy')
        mm = np.lib.format.open_memmap(npy, mode='w+', dtype=np.float64, shape=maps[0].shape)
        mm[:] = np.asarray(maps[0].A)
        mm.flush()

        mmapped = maps[0].replace(A=np.lib.format.open_memmap(npy, mode='r'))
        assert isinstance(mmapped.A, np.memmap), type(mmapped.A)

        mpath = os.path.join(tmp, 'mm.asdf')
        mmapped.write_asdf(mpath)
        assert np.array_equal(np.asarray(VarianceMap.from_asdf(mpath, 0).A),
                              np.asarray(maps[0].A))

        # A single-tree file: readable by VarianceMap.from_asdf(), and refused by the
        # multimap reader, which covers every tree by definition.
        one = os.path.join(tmp, 'one.asdf')
        maps[1].write_asdf(one, provenance=dict(note='single'))
        m1 = VarianceMap.from_asdf(one, 1)
        assert m1.itree == 1 and np.array_equal(np.asarray(m1.A), np.asarray(maps[1].A))
        expect_raise(lambda: VarianceMultiMap.from_asdf(one), 'covers EVERY tree')
        expect_raise(lambda: VarianceMap.from_asdf(one, 0), 'trees present: [1]')

        # ---- the tripwires ----

        bad = os.path.join(tmp, 'bad.asdf')

        # m_to_n is the one field with no independent witness in the file.
        def break_m_to_n(root):
            mn = np.array(root['trees'][0]['m_to_n'])
            mn[-1] = 0 if (mn[-1] != 0) else 1
            root['trees'][0]['m_to_n'] = mn
        _corrupt(path, bad, break_m_to_n)
        expect_raise(lambda: VarianceMultiMap.from_asdf(bad), 'multiplet ordering convention')

        # is_coarse_grained and L are one fact.
        _corrupt(path, bad, lambda root: root['trees'][0].__setitem__('is_coarse_grained',
                                                                      True))
        expect_raise(lambda: VarianceMultiMap.from_asdf(bad), 'are one fact')

        # nbeta against the array it describes.
        _corrupt(path, bad, lambda root: root['trees'][1].__setitem__('nbeta', 3))
        expect_raise(lambda: VarianceMultiMap.from_asdf(bad), 'stored nbeta')

        # itree against the tree yaml's own (primary_tree_index, early_trigger_level), which
        # is what stops a mislabelled entry from being read as a different tree.
        _corrupt(path, bad, lambda root: root['trees'][0].__setitem__('itree', 1))
        expect_raise(lambda: VarianceMultiMap.from_asdf(bad), "'itree' field claims")

        # A tree yaml describing a different instrument: check_consistency() names the member.
        _corrupt(path, bad, lambda root: root['trees'][0].__setitem__(
            'tree_yaml', root['trees'][0]['tree_yaml'].replace('nprofiles: ', 'nprofiles: 1')))
        expect_raise(lambda: VarianceMultiMap.from_asdf(bad), 'nprofiles')

        # is_factored is checked AGAINST the arrays, never believed: a dense block that
        # claims to be factored is refused rather than reinterpreted. (The factored round
        # trip itself is test_asdf_factored().)
        _corrupt(path, bad, lambda root: root['trees'][0].__setitem__('is_factored', True))
        expect_raise(lambda: VarianceMultiMap.from_asdf(bad), 'carries no')

        # Version and identity, so the next format change is an error and not a KeyError.
        _corrupt(path, bad, lambda root: root.__setitem__('format_version',
                                                          FORMAT_VERSION + 1))
        expect_raise(lambda: VarianceMultiMap.from_asdf(bad), 'format_version is')

        _corrupt(path, bad, lambda root: root.pop('format_version'))
        expect_raise(lambda: VarianceMultiMap.from_asdf(bad), 'predates this format')

        import asdf
        asdf.AsdfFile({'something_else': 1}).write_to(bad)
        expect_raise(lambda: VarianceMultiMap.from_asdf(bad), 'not a variance-map file')

        # An OLD-format file is named as such, since that is the one wrong file a user is
        # likely to have in hand.
        old = os.path.join(tmp, 'old.asdf')
        asdf.AsdfFile({'variance_map': {'trees': []}}).write_to(old)
        expect_raise(lambda: VarianceMultiMap.from_asdf(old), 'old-format file')

        nbytes = os.path.getsize(path)
    finally:
        shutil.rmtree(tmp, ignore_errors=True)

    atomic_print(f'    test_asdf_io(r={r}, ntrees={ntrees}): {nbytes/2**20:.1f} MiB file,'
                 ' eager + memmapped reads and every reader check exercised')


def test_asdf_factored(r=7, subband_counts=(2,1), K=4):
    """The factored half of the round trip, which completes the representation matrix:
    factored x {fine, coarse} x {admissible, uncertified} x {y_true present, absent}, through
    both readers.

    Also the reader's flag-versus-arrays checks. is_factored is never BELIEVED -- a block
    that carries both array groups or neither is refused by name, because that is exactly the
    case where trusting the flag silently reinterprets a matrix.
    """

    import os
    import shutil
    import tempfile

    from .asdf_io import ROOT_KEY

    config = _make_test_config(r, list(subband_counts))
    rng = np.random.default_rng(24)
    tmp = tempfile.mkdtemp()

    def expect_raise(fn, needle):
        try:
            fn()
        except RuntimeError as e:
            assert needle in str(e), (needle, str(e))
            return
        raise AssertionError(f'expected a RuntimeError mentioning {needle!r}')

    try:
        path = os.path.join(tmp, 'f.asdf')

        tree = make_tree(config, 0)
        fs = tree.frequency_subbands
        Lc = int(fs.pf_rank) + 1
        nb_coarse = (1 << (int(tree.total_rank()) - Lc)) * int(fs.N) * int(tree.nprofiles)

        fine, _ = _factored_map(config, 0, rng, K=K)
        coarse, _ = _factored_map(config, 0, rng, K=K, L=Lc, nbeta=nb_coarse)

        # The remaining two axes are folded onto the two cases: fine + admissible + y_true +
        # pinned columns + both flags set, and coarse + uncertified + no y_true + no pins.
        y = np.abs(rng.normal(size=fine.nalpha)) + 1.0
        cases = [fine.replace(y_true=y, is_admissible=True,
                              Q_is_semiorthogonal=True, W_is_semiorthogonal=True,
                              pinned_columns=[0, 2]),
                 coarse.replace(y_true=None, is_admissible=False)]

        for (i, m) in enumerate(cases):
            m.write_asdf(path, provenance=dict(case=i))

            for eager in (True, False):
                ctx = (_eager_ctx(VarianceMap.from_asdf(path, 0)) if eager
                       else _open_one(path, 0))
                with ctx as g:
                    assert g.is_factored and (g.factor_rank == K), i
                    assert g.A is None
                    assert np.array_equal(np.asarray(g.Q), np.asarray(m.Q)), i
                    assert np.array_equal(np.asarray(g.mid), np.asarray(m.mid)), i
                    assert np.array_equal(np.asarray(g.W), np.asarray(m.W)), i
                    assert np.array_equal(g.pinned_columns, m.pinned_columns), i
                    assert g.Q_is_semiorthogonal == m.Q_is_semiorthogonal, i
                    assert g.W_is_semiorthogonal == m.W_is_semiorthogonal, i
                    assert (g.L == m.L) and (g.is_admissible == m.is_admissible), i
                    assert (g.y_true is None) == (m.y_true is None), i
                    assert np.array_equal(g.dense(), m.dense()), i

            # No dense matrix is written for a factored map.
            import asdf
            with asdf.open(path, lazy_load=False, memmap=False) as af:
                blk = af[ROOT_KEY]['trees'][0]
                assert blk.get('A') is None and (blk['is_factored'] is True), i
                assert int(blk['factor_rank']) == K, i

        # ---- the reader checks the flag against the arrays, in both directions ----
        bad = os.path.join(tmp, 'bad.asdf')

        _corrupt(path, bad, lambda root: root['trees'][0].__setitem__('is_factored', False))
        expect_raise(lambda: VarianceMap.from_asdf(bad, 0), 'carries Q/mid/W')

        def add_dense(root):
            t = root['trees'][0]
            t['A'] = np.zeros((int(t['nbeta']), config.get_total_nfreq()))
        _corrupt(path, bad, add_dense)
        expect_raise(lambda: VarianceMap.from_asdf(bad, 0), 'BOTH')

        def drop_all(root):
            t = root['trees'][0]
            for k in ('Q', 'mid', 'W'):
                t[k] = None
        _corrupt(path, bad, drop_all)
        expect_raise(lambda: VarianceMap.from_asdf(bad, 0), 'no matrix at all')

        _corrupt(path, bad, lambda root: root['trees'][0].__setitem__('factor_rank', K + 1))
        expect_raise(lambda: VarianceMap.from_asdf(bad, 0), 'stored factor_rank')

        _corrupt(path, bad, lambda root: root['trees'][0].__setitem__('nbeta', 3))
        expect_raise(lambda: VarianceMap.from_asdf(bad, 0), 'stored nbeta')

        # A memmap-backed Q, the same regression the dense path has: asdf refuses ndarray
        # subclasses, and an accumulated-on-disk Q is the obvious large case.
        npy = os.path.join(tmp, 'Q.npy')
        m = cases[1]
        mm = np.lib.format.open_memmap(npy, mode='w+', dtype=np.float64,
                                       shape=np.asarray(m.Q).shape)
        mm[:] = np.asarray(m.Q)
        mm.flush()
        mmapped = m.replace(Q=np.lib.format.open_memmap(npy, mode='r'))
        assert isinstance(mmapped.Q, np.memmap), type(mmapped.Q)
        mpath = os.path.join(tmp, 'mm.asdf')
        mmapped.write_asdf(mpath)
        assert np.array_equal(np.asarray(VarianceMap.from_asdf(mpath, 0).Q), np.asarray(m.Q))

        nbytes = os.path.getsize(path)
    finally:
        shutil.rmtree(tmp, ignore_errors=True)

    atomic_print(f'    test_asdf_factored(r={r}, K={K}): {nbytes/2**10:.1f} KiB file,'
                 ' fine + coarse round-tripped through both readers, flag-vs-arrays checked')


####################################   the LP (varmap/lp.py)   ####################################


def _lp_cell(r=6, subband_counts=(1, 1), L=None, K=5, seed=11, nzero=1):
    """A small but REAL-geometry LP cell: (Abar, y, labels, W, config, coarse map).

    Real geometry rather than a random matrix, because the label arithmetic and the
    coarse-graining are half of what the steps have to get right.
    """

    config = _make_test_config(r, list(subband_counts))
    rng = np.random.default_rng(seed)
    fine = _random_map(config, 0, rng, nzero=nzero)
    L = fine.pf_rank + 1 if (L is None) else L
    coarse = fine.coarse_grain(L)

    Abar = np.ascontiguousarray(coarse.dense(force=True))
    scale = float(2.0 ** np.ceil(np.log2(float(Abar.max()))))    # exact in binary
    Abar = np.ascontiguousarray(Abar / scale)
    y = np.asarray(fine.y_true, dtype=np.float64) / scale
    labels = coarse.alpha_to_beta_block(0, coarse.nalpha)

    # A signed dictionary whose first column is nonnegative, which is what the additive
    # repairs need and what an SVD basis does not provide on its own.
    W = rng.normal(size=(coarse.nfreq, K))
    W[:, 0] = np.abs(W[:, 0]) + Abar.max(axis=0)
    return Abar, y, labels, np.ascontiguousarray(W), config, coarse


def _dominates(Q, W, Abar):
    """The elementwise admissibility test, done densely -- the tests here are small enough."""
    return bool(np.all((Q @ W.T) >= Abar))


def _max_ratio(Q, W, Abar):
    """max over the entries with Abar > 0 of Abar/(Q W^T).

    NOT the same test as _dominates(), and the difference is the whole reason the additive
    repair exists: where the product is NEGATIVE this ratio is negative, so it loses to the
    row's maximum and the multiplicative repair cannot see the violation at all.
    """
    with np.errstate(divide='ignore', invalid='ignore'):
        return float(np.nanmax(np.where(Abar > 0, Abar / (Q @ W.T), 0.0)))


def test_lp_config():
    """LpConfig: the presets, the repair label, and the fields that refuse rather than guess."""

    from .lp import LpConfig

    q, w = LpConfig.for_qstep(), LpConfig.for_wstep()

    # The three families that are genuinely per-direction. Sharing them would make the two
    # directions one config with a flag, which they are not.
    assert (q.clip_rel, q.rescue, q.rescale) == (1.0e-8, 'prefix', 'rows')
    assert (w.clip_rel, w.rescue, w.rescale) == (0.0, None, 'cols')
    assert q.resolved_rescale('rows') == 'rows'
    assert LpConfig().resolved_rescale('cols') == 'cols'          # 'auto' follows the axis

    # The four-way choose-one repair knob maps onto the three fields without loss.
    for tag, triple in (('cols', (False, False, 'cols')), ('rows', (False, False, 'rows')),
                        ('additive', (False, True, 'none')), ('none', (False, False, 'none'))):
        c = LpConfig.for_wstep(**dict(zip(('additive_first', 'additive_last', 'rescale'),
                                          triple)))
        assert (c.additive_first, c.additive_last, c.rescale) == triple, tag

    assert LpConfig(additive_first=True, additive_last=True).repair_label == 'additive_first'
    assert LpConfig(additive_last=True).repair_label == 'shipped'
    assert LpConfig(rescale='none').repair_label == 'raw'
    assert LpConfig(additive_first=True).repair_label not in ('additive_first', 'shipped', 'raw')

    for d in ('q', 'w'):
        rec = LpConfig.recommended(d)
        assert rec.cuts and (rec.cuts_pool == 8192) and rec.cuts_agg
        assert rec.additive_last
    assert LpConfig.recommended('w').rescale == 'none'
    assert LpConfig.recommended('q', threads=4).threads == 4

    # Frozen, and serializable: a step stashes one of these in a map's history, and the file
    # format can only write plain data.
    try:
        q.nonneg = True
        raise AssertionError('LpConfig should be frozen')
    except dataclasses.FrozenInstanceError:
        pass
    assert LpConfig(**dataclasses.asdict(q)) == q
    # ... including after a file round trip, which turns the tuple field into a list. A step
    # stashes its config in the map's history, and reading it back has to give the config the
    # step ran under, not one that merely looks like it.
    assert LpConfig(**dict(dataclasses.asdict(q), rescue_ladder=[64, 32, 16, 8])) == q

    # Named but not implemented: these must raise rather than silently doing something else.
    def expect_raise(fn, needle):
        try:
            fn()
        except RuntimeError as e:
            assert needle in str(e), (needle, str(e))
            return
        raise AssertionError(f'expected a RuntimeError mentioning {needle!r}')

    for kw, needle in ((dict(equilibrate=False), 'unequilibrated'),
                       (dict(slack=0.1), 'slack'), (dict(nnz_cap=2), 'nnz_cap')):
        expect_raise(LpConfig(**kw)._check_implemented, needle)
    expect_raise(lambda: LpConfig.recommended('x'), "'q' or 'w'")

    atomic_print(f'    test_lp_config: {len(dataclasses.fields(q))} fields, both presets,'
                 ' the four-way repair mapping, and the three refusals')


def test_lp_primitive():
    """solve_covering_lps() on problems whose answer is checkable by other means."""

    from .lp import LpConfig, solve_covering_lps, solve_cover_lp

    rng = np.random.default_rng(4)

    # The free LP can never be worse than the bounded one: dropping x >= 0 enlarges the
    # feasible set, and the covering constraint is on the PRODUCT, not on the coefficients.
    M = np.abs(rng.random((60, 4))) + 0.2
    b = np.abs(rng.random((60, 5)))
    cost = M.sum(axis=0)
    Xp, ip = solve_covering_lps(M, b, cost, LpConfig(nonneg=True))
    Xf, iff = solve_covering_lps(M, b, cost, LpConfig(nonneg=False))
    assert (ip['n_failed'] == 0) and (iff['n_failed'] == 0)
    assert Xp.shape == (5, 4) and Xf.shape == (5, 4)
    for j in range(5):
        assert np.all((M @ Xp[j]) >= b[:, j] - 1e-7)
        assert np.all((M @ Xf[j]) >= b[:, j] - 1e-7)
        assert (cost @ Xf[j]) <= (cost @ Xp[j]) + 1e-9

    # A zero right-hand side is NOT automatically satisfied once the matrix and the variable
    # may be signed: the product can go negative exactly where Abar is zero. This is the one
    # property the zero-rhs block exists for, so it is checked directly.
    Ms = np.abs(rng.random((40, 5))) + 0.1
    Ms[:, 3] *= -1.0
    bs = np.abs(rng.random((40, 3)))
    bs[::3] = 0.0
    Xs, _ = solve_covering_lps(Ms, bs, Ms.sum(axis=0), LpConfig(nonneg=False))
    for j in range(3):
        v = Ms @ Xs[j]
        assert np.all(v >= bs[:, j] - 1e-7), 'positive-rhs rows violated'
        assert v.min() >= 0.0, f'the zero-rhs rows let the product go negative: {v.min()}'

    # A column that appears in no constraint is dropped rather than repriced: with a free
    # variable a positive price on it would send the LP to minus infinity.
    Md = np.hstack([Ms, np.zeros((40, 1))])
    Xd, idd = solve_covering_lps(Md, bs, np.append(Ms.sum(axis=0), 1.0), LpConfig(nonneg=False))
    assert idd['dead_cols'] == 1
    assert np.all(Xd[:, -1] == 0.0)

    # Parallel execution changes no answer: the subproblems are independent.
    Xw, _ = solve_covering_lps(Ms, bs, Ms.sum(axis=0), LpConfig(nonneg=False), workers=3)
    assert np.array_equal(Xs, Xw), 'the worker count changed the answer'

    # A failure with no fallback raises rather than returning a silent zero.
    try:
        solve_cover_lp(np.array([-1.0]), np.ones((2, 1)), np.array([1.0, 1.0]),
                       LpConfig(nonneg=False))
        raise AssertionError('an unbounded LP with no fallback should raise')
    except RuntimeError as e:
        assert 'no fallback' in str(e), str(e)

    atomic_print('    test_lp_primitive: free <= bounded, zero-rhs rows keep the product'
                 ' nonnegative, dead columns dropped, workers inert')


def test_lp_optimality(K=3, nfreq=7):
    """The Q-step is EXACT: its point is the LP optimum, checked by brute-force enumeration.

    An optimum of ``min s.q  s.t.  W q >= b`` with no bounds sits at a vertex where K of the
    constraints are active, so enumerating the K-subsets and keeping the feasible ones gives
    the optimum outright. That is a genuinely independent answer rather than a second run of
    the same solver.
    """

    import itertools

    from .lp import LpConfig, q_step

    rng = np.random.default_rng(7)
    W = np.abs(rng.random((nfreq, K))) + 0.25
    W[:, 1] *= -1.0                                  # signed, so the vertex argument is used
    Abar = np.abs(rng.random((6, nfreq))) + 0.05     # every rhs positive: no zero-rhs block
    s = W.sum(axis=0)

    Q, _, _ = q_step(Abar, W, LpConfig.for_qstep(nonneg=False, clip_rel=0.0), repair=False,
                     Q0=np.zeros((6, K)))
    worst = 0.0
    for i in range(Abar.shape[0]):
        b = Abar[i]
        best = np.inf
        for rows in itertools.combinations(range(nfreq), K):
            A = W[list(rows)]
            if abs(np.linalg.det(A)) < 1e-12:
                continue
            q = np.linalg.solve(A, b[list(rows)])
            if np.all(W @ q >= b - 1e-9 * max(1.0, np.abs(b).max())):
                best = min(best, float(s @ q))
        assert np.isfinite(best), 'the brute force found no feasible vertex'
        worst = max(worst, abs(float(s @ Q[i]) - best) / max(abs(best), 1e-300))
    assert worst < 1e-7, worst

    atomic_print(f'    test_lp_optimality(K={K}, nfreq={nfreq}): the raw Q-step point matches'
                 f' brute-force enumeration to {worst:.2g} relative')


def test_lp_repairs():
    """Every repair, against what it actually guarantees -- which is not the same for all four."""

    from .lp import (LpConfig, repair_rows, repair_cols, repair_additive, fix_nonneg,
                     violation_stats, check_nonneg, blocking_is_exact)

    Abar, y, labels, W, config, coarse = _lp_cell(K=5)
    nbeta, nfreq = Abar.shape
    cfg = LpConfig.for_qstep()

    # A well-conditioned inadmissible point: the feasible one-hot seed on W's nonnegative
    # column, shrunk. The product is positive everywhere, so every repair has work to do and
    # all four can finish it.
    seed = (Abar / np.maximum(W[:, 0], 1e-300)[None, :]).max(axis=1)
    Q = np.zeros((nbeta, W.shape[1]))
    Q[:, 0] = 0.6 * seed
    assert not _dominates(Q, W, Abar)
    st0 = violation_stats(Q, W, None, Abar)
    assert (st0['n_viol'] > 0) and (st0['max_ratio'] > 1.0)

    for name, fn, factor in (('repair_rows', repair_rows, 'Q'),
                             ('repair_cols', repair_cols, 'W'),
                             ('repair_additive', repair_additive, 'Q'),
                             ('fix_nonneg', fix_nonneg, 'Q')):
        out, st = fn(Q, W, None, Abar, cfg)
        Qn, Wn = (out, W) if (factor == 'Q') else (Q, out)
        assert _dominates(Qn, Wn, Abar), name
        assert Qn.shape == Q.shape and Wn.shape == W.shape, name    # rank-preserving
        assert not np.shares_memory(out, Q if factor == 'Q' else W), f'{name} wrote in place'
        assert check_nonneg(Qn, Wn)[0] == 0, name

    # THE REASON THERE ARE TWO KINDS OF REPAIR. On a signed dictionary the product can go
    # NEGATIVE where Abar is positive; a positive row scale makes such an entry more negative,
    # and the ratio it works from cannot even see it. So the multiplicative repair reports
    # success -- correctly, by its own test -- while the map is still inadmissible, and only
    # an additive lift on a nonnegative column fixes it.
    rng = np.random.default_rng(31)
    Qbad = np.abs(rng.random((nbeta, W.shape[1]))) * 0.4
    Qbad[:, 0] += 0.3
    Qm, _ = repair_rows(Qbad, W, None, Abar, cfg)
    assert _max_ratio(Qm, W, Abar) <= 1.0, 'the multiplicative repair failed its own test'
    n_neg = check_nonneg(Qm, W)[0]
    assert n_neg > 0 and not _dominates(Qm, W, Abar), \
        'this case is meant to exercise the sign blind spot and no longer does'
    Qa, _ = repair_additive(Qbad, W, None, Abar, cfg)
    assert _dominates(Qa, W, Abar) and (check_nonneg(Qa, W)[0] == 0)

    # The additive lift is defined on the COLUMNS of W, so it refuses a non-identity mid
    # rather than quietly raising the wrong thing.
    try:
        fix_nonneg(Q, W, np.eye(W.shape[1]), Abar, cfg)
        raise AssertionError('fix_nonneg should refuse a non-identity mid')
    except RuntimeError as e:
        assert 'mid' in str(e), str(e)

    # Blocking is bit-identical, with a RAGGED tail forced: a splitter that bounds the block
    # size but not the tail ends on a short block, and numpy changes gemm kernel there.
    assert blocking_is_exact(nfreq), nfreq
    # 15 and 37 leave a tail SHORTER than the 8-row floor at nbeta = 112, so they are what
    # exercise the tail merge rather than just the block size.
    base, _ = repair_rows(Qbad, W, None, Abar, cfg)
    for block_rows in (8, 13, 15, 37):
        c = LpConfig.for_qstep(block_bytes=int(block_rows * 1.5 * 8 * nfreq))
        got, _ = repair_rows(Qbad, W, None, Abar, c)
        assert np.array_equal(base, got), f'blocking changed the repair at {block_rows} rows'

    # A subset repair touches only its rows -- which is what makes a partial step's repair
    # meaningful at all.
    rows = np.arange(0, nbeta, 3)
    sub, _ = repair_rows(Q, W, None, Abar, cfg, rows=rows)
    keep = np.setdiff1d(np.arange(nbeta), rows)
    assert np.array_equal(sub[keep], Q[keep])
    assert np.all((sub[rows] @ W.T) >= Abar[rows])

    atomic_print(f'    test_lp_repairs(nbeta={nbeta}, nfreq={nfreq}): four repairs dominate,'
                 f' the sign blind spot reproduced ({n_neg} entries) and fixed additively,'
                 ' blocking exact over ragged tails')


def test_lp_steps():
    """q_step and w_step end to end on a real-geometry cell, and their contracts."""

    from .lp import LpConfig, q_step, w_step, violation_stats, f as lp_f

    Abar, y, labels, W0, config, coarse = _lp_cell(K=5)
    nbeta, nfreq = Abar.shape
    K = W0.shape[1]
    Q0 = np.zeros((nbeta, K))
    # The one-hot seed on the nonnegative column: feasible for every group by construction.
    Q0[:, 0] = (Abar / np.maximum(W0[:, 0], 1e-300)[None, :]).max(axis=1) * (1 + 1e-12)
    assert _dominates(Q0, W0, Abar)

    cfgq = LpConfig.for_qstep(nonneg=False)
    Q, W, iq = q_step(Abar, W0, cfgq, Q0=Q0)
    assert np.array_equal(W, W0), 'a Q-step must not touch W'
    assert _dominates(Q, W0, Abar), 'the repaired Q-step is not admissible'
    assert iq['step'] == 'Q' and iq['n_lp'] == nbeta
    # Exact given W, so it cannot be worse than the feasible seed it was handed.
    s = W0.sum(axis=0)
    assert float((Q @ s).sum()) <= float((Q0 @ s).sum()) * (1 + 1e-9)

    # repair=False returns the RAW point. It is not usable as an approximation until
    # repaired, which is exactly why it is worth storing rather than re-solving.
    Qraw, _, ir = q_step(Abar, W0, cfgq, Q0=Q0, repair=False)
    assert ir['repair_label'] == 'raw'
    assert float((Qraw @ s).sum()) <= float((Q @ s).sum()) * (1 + 1e-9)

    # groups= slices merge to exactly what one process would have produced.
    lo = np.arange(0, nbeta // 2)
    hi = np.arange(nbeta // 2, nbeta)
    Qa, _, _ = q_step(Abar, W0, cfgq, Q0=Q0, repair=False, groups=lo)
    Qb, _, _ = q_step(Abar, W0, cfgq, Q0=Q0, repair=False, groups=hi)
    merged = np.vstack([Qa[lo], Qb[hi]])
    assert np.array_equal(merged, Qraw), 'a sliced Q-step did not merge to the whole one'
    try:
        q_step(Abar, W0, cfgq, Q0=Q0, groups=lo)
        raise AssertionError('groups= with repair=True should raise')
    except RuntimeError as e:
        assert 'after the slices are merged' in str(e), str(e)

    # SOLVE ONCE, REPAIR SEVERAL WAYS. This is the workflow the repair is exposed for: a
    # Q-step at production scale is hundreds of core-hours and a repair is one blocked pass,
    # so trying a second repair must never mean re-solving.
    from .lp import apply_repair
    arms = {}
    for label, kw in (('rows', dict(rescale='rows')),
                      ('additive_first', dict(rescale='rows', additive_first=True,
                                              additive_last=True)),
                      ('additive_only', dict(rescale='none', additive_last=True))):
        c = LpConfig.for_qstep(nonneg=False, **kw)
        Qr, Wr, ri = apply_repair(Qraw, W0, None, Abar, c, axis='rows')
        assert np.array_equal(Wr, W0), (label, 'a rows repair must not touch W')
        assert _dominates(Qr, W0, Abar), (label, 'repaired but not admissible')
        arms[label] = float((Qr @ s).sum())
        assert ri['repair_label'] == c.repair_label, label
    # Every arm starts from the SAME stored point, so they are comparable by construction --
    # which is the entire reason for storing it. (On this cell the raw point is already
    # nearly admissible, so the arms agree to ~1e-12; asserting that they DIFFER would be
    # asserting a property of the cell rather than of the code.)
    assert len(arms) == 3 and all(np.isfinite(v) for v in arms.values()), arms

    # THE LOAD-BEARING ONE: repairing the stored raw point with the step's own settings
    # reproduces the step's own output exactly. That is what makes "solve once, repair many
    # ways" a valid comparison rather than an approximation of one.
    Qsame, _, _ = apply_repair(Qraw, W0, None, Abar, cfgq, axis='rows')
    assert np.array_equal(Qsame, Q), 'the standalone repair is not the step\'s own repair'

    # The W-step, and the majorize-minimize guarantee its objective must satisfy.
    cfgw = LpConfig.for_wstep(nonneg=False)
    Qw, W2, iw = w_step(Abar, Q, y, labels, W0, cfgw)
    assert iw['step'] == 'W' and iw['n_lp'] == nfreq
    assert _dominates(Qw, W2, Abar), 'the repaired W-step is not admissible'
    assert iw['w_obj_raw'] <= iw['w_obj_before'] * (1 + 1e-9), \
        'a correctly solved W-step cannot increase its own linear objective'

    # A pinned column is excluded from the LP and can only be scaled UP by the 'cols' repair,
    # so it stays nonnegative and still dominates every group -- which is what makes the
    # additive repair's certificate survive a W-step.
    Qp, Wp, ip = w_step(Abar, Q, y, labels, W0, cfgw, pinned=[0])
    assert ip['n_pinned'] == 1
    assert np.all(Wp[:, 0] >= W0[:, 0] - 1e-12) and (ip['pin_drift'] >= 1.0 - 1e-12)

    # Monotonicity of an alternation, which is what the whole majorization is for. When this
    # goes the wrong way it has twice been a real defect rather than a bad schedule.
    scored = y >= YTRUE_FLOOR          # exactly the rows get_distance() scores

    def D0(Q_, W_):
        x = ((Q_ @ W_.sum(axis=0))[labels])[scored] / y[scored]
        return float(np.mean(lp_f(x)))

    d = [D0(Q, W0)]
    Qi, Wi = Q, W0
    for _ in range(2):
        Qi, Wi, _ = w_step(Abar, Qi, y, labels, Wi, cfgw)
        Qi, Wi, _ = q_step(Abar, Wi, cfgq, Q0=Qi)
        d.append(D0(Qi, Wi))
    assert all(d[i+1] <= d[i] * (1 + 1e-9) for i in range(len(d)-1)), d

    atomic_print(f'    test_lp_steps(nbeta={nbeta}, nfreq={nfreq}, K={K}): both steps'
                 f' admissible, slices merge exactly, D0 {d[0]:.6g} -> {d[-1]:.6g} over'
                 ' two alternations')


def test_lp_rescue():
    """The prefix rescue, including the branch that ACCEPTS a rescued row.

    When a subproblem fails, the solver returns the caller's seed, and the step re-solves that
    subproblem on a PREFIX of the same dictionary -- fewer columns, the same admissibility
    argument, the rank unchanged -- keeping the result only if it is admissible AND its
    objective beats the incumbent. It matters out of proportion to how often it fires: one
    failure per ~450 groups costs more D than an entire doubling of the rank.

    The failures are INJECTED through solve_fn rather than provoked. Provoking them needs a
    numerically degenerate cell, and on every such cell found so far the prefix solves fail
    too -- the rescue runs and accepts nothing, which exercises the machinery but never the
    accept path. Injecting them leaves the rescue itself entirely real: it re-solves through
    the ordinary solver, judges by its own admissibility and objective tests, and writes back
    through the same code. It also exercises solve_fn, which is the documented extension point
    for an agent swapping in a different solver.
    """

    from .lp import LpConfig, q_step, solve_covering_lps

    Abar, y, labels, W, config, coarse = _lp_cell(K=12)
    nbeta, K = Abar.shape[0], W.shape[1]
    seed = np.zeros((nbeta, K))
    seed[:, 0] = (Abar / np.maximum(W[:, 0], 1e-300)[None, :]).max(axis=1) * (1 + 1e-12)
    assert _dominates(seed, W, Abar), 'the seed must be feasible for the incumbent to mean something'

    hurt = np.array([0, 3, 7, 11])

    def flaky(M, B, cost, cfg, **kw):
        """solve_covering_lps, with a few subproblems reported as failed -- which is exactly
        what the solver does when it cannot solve one: it returns the seed and says so."""
        X, info = solve_covering_lps(M, B, cost, cfg, **kw)
        X[hurt] = kw['x_seed'][hurt]
        return X, dict(info, failed=[int(i) for i in hurt], n_failed=int(hurt.size))

    cfg = LpConfig.for_qstep(nonneg=False)
    Q, _, info = q_step(Abar, W, cfg, Q0=seed, repair=False, solve_fn=flaky)

    assert info['rescue_rows'] == hurt.size, info
    assert info['rescue_improved'] >= 1, ('the rescue accepted nothing, so its accept branch'
                                          ' is still untested', info)

    s = W.sum(axis=0)
    changed = np.flatnonzero(np.any(Q[hurt] != seed[hurt], axis=1))
    for j in changed:
        i = hurt[j]
        # Accepted rows are re-solved on a PREFIX, so the rank is unchanged and the columns
        # past that prefix are exactly zero.
        nz = np.flatnonzero(Q[i] != 0.0)
        assert nz.size == 0 or (nz.max() + 1) in cfg.rescue_ladder, (i, nz)
        # ... and accepted only because the objective strictly improved.
        assert (s @ Q[i]) < (s @ seed[i]), (i, s @ Q[i], s @ seed[i])
        # ... and only if the rescued row is itself admissible.
        assert np.all((Q[i] @ W.T) >= Abar[i] - 1e-12), i

    # Rows that were never touched are untouched.
    keep = np.setdiff1d(np.arange(nbeta), hurt)
    Qref, _, _ = q_step(Abar, W, cfg, Q0=seed, repair=False)
    assert np.array_equal(Q[keep], Qref[keep])

    # With the rescue off, a failed row keeps the seed and nothing re-solves it.
    Qoff, _, ioff = q_step(Abar, W, LpConfig.for_qstep(nonneg=False, rescue=None), Q0=seed,
                           repair=False, solve_fn=flaky)
    assert 'rescue_rows' not in ioff
    assert np.array_equal(Qoff[hurt], seed[hurt])

    atomic_print(f'    test_lp_rescue(nbeta={nbeta}, K={K}): {info["rescue_improved"]} of'
                 f' {info["rescue_rows"]} injected failures rescued on a prefix, rank'
                 ' preserved, and none accepted without improving')


def test_lp_negative_rhs():
    """The right-hand sides go NEGATIVE, which only a factored reference can make happen.

    A streamed Abar is a max over nonnegative entries, so its right-hand sides are >= 0 by
    construction and nothing about this case arises. A FACTORED reference with a signed Q or
    W -- which after campaign 2 is the normal case -- is negative wherever the true map is
    zero or small, and it is the only thing that exercises the branch.

    Three properties, and the second is the one worth knowing:

      - the solve does not divide by a negative right-hand side (which would flip the
        inequality). Rows with b <= 0 are normalized to unit max-abs instead, zero and
        negative alike;
      - so the constraint imposed where b < 0 is that the product be POSITIVE, which is
        STRICTER than b. That is safe (it implies the true constraint) and it is useful: it
        stops the product going negative in exactly the channels a later additive repair
        would otherwise have to lift;
      - and it has a sharp price. A dictionary with no nonnegative column cannot make the
        product positive everywhere, so on a signed reference EVERY LP comes back infeasible.
        That is what sign-canonicalizing a basis, or pinning a nonnegative column, is for.
    """

    from .lp import LpConfig, q_step, w_step, covering_lp_data, _clip_rhs

    config = _make_test_config(6, [1, 1])
    rng = np.random.default_rng(23)
    tree = make_tree(config, 0)
    r, R = tree.total_rank(), tree.frequency_subbands.pf_rank
    L = R + 1
    N, P = tree.frequency_subbands.N, tree.nprofiles
    nbeta = (1 << (r - L)) * N * P
    nfreq = config.get_total_nfreq()

    ref = VarianceMap.from_factors(config, 0, rng.normal(size=(nbeta, 4)),
                                   rng.normal(size=(nfreq, 4)), L=L)
    Abar = np.array(ref.dense(), copy=True)
    Abar /= np.abs(Abar).max()
    frac_neg = float(np.mean(Abar < 0))
    assert frac_neg > 0.3, ('this reference is meant to be genuinely signed and is not',
                            frac_neg)

    # A dictionary with a STRICTLY POSITIVE column, so that a feasible point exists by
    # construction whatever the other columns do.
    W = rng.normal(size=(nfreq, 5))
    W[:, 0] = np.abs(W[:, 0]) + np.clip(Abar.max(axis=0), 0.0, None) + 0.1
    seed = np.zeros((nbeta, 5))
    seed[:, 0] = np.clip((Abar / W[:, 0][None, :]).max(axis=1), 0.0, None) * 1.1 + 1.0

    # The building block hands back the negative right-hand sides unchanged.
    _, _, b = covering_lp_data(ref.replace(Q=ref.Q, W=W[:, :4], mid=np.eye(4)), ref, 0)
    assert np.any(b < 0), 'the reference row has no negative entries'

    # THE Q DIRECTION DOES NOT REACH THIS CASE AT ITS DEFAULT: its relative constraint floor
    # is 1e-8, and every negative entry is below that, so they are clipped to zero before the
    # solver sees them. The W direction ships a floor of 0, so there the case is live.
    assert np.all(_clip_rhs(np.array(Abar[0], copy=True), 1.0e-8) >= 0.0)
    assert np.any(_clip_rhs(np.array(Abar[0], copy=True), 0.0) < 0.0)

    Q, _, iq = q_step(Abar, W, LpConfig.for_qstep(nonneg=False, clip_rel=0.0), Q0=seed)
    prod = Q @ W.T
    assert iq['n_failed'] == 0, iq['status']
    assert np.all(prod >= Abar), 'the result does not dominate the signed reference'
    # The property the strict treatment buys: positive even where the reference is negative,
    # so no additive repair is needed to rescue those channels.
    assert prod.min() > 0.0, prod.min()

    # The W-step reaches the same case at its OWN default, since its floor is 0.
    labels = ref.alpha_to_beta_block(0, ref.nalpha)
    y = np.abs(rng.random(ref.nalpha)) + 0.05
    _, W2, iw = w_step(Abar, Q, y, labels, W, LpConfig.for_wstep(nonneg=False))
    assert iw['n_failed'] == 0, iw['status']
    assert np.all((Q @ W2.T) >= Abar)

    # THE PRICE. With no nonnegative column the product cannot be made positive everywhere,
    # so a signed reference makes every subproblem infeasible. Asserted rather than merely
    # noted, because it is the tripwire on anybody "relaxing" the zero-rhs treatment: if this
    # ever starts succeeding, the constraint that keeps the product nonnegative has been lost.
    Wsigned = rng.normal(size=(nfreq, 5))
    assert not np.any((Wsigned.min(axis=0) >= 0) & (Wsigned.max(axis=0) > 0))
    _, _, ibad = q_step(Abar, Wsigned, LpConfig.for_qstep(nonneg=False, clip_rel=0.0),
                        Q0=np.zeros((nbeta, 5)), repair=False)
    assert ibad['n_failed'] == nbeta, ibad['status']
    assert ibad['status'].get('infeasible') == nbeta, ibad['status']

    atomic_print(f'    test_lp_negative_rhs(nbeta={nbeta}, nfreq={nfreq}):'
                 f' {frac_neg:.0%} of the reference is negative; the product stays positive'
                 f' there, and a dictionary with no nonnegative column is infeasible'
                 f' {nbeta}/{nbeta}')


def test_lp_building_blocks():
    """covering_lp_data() and majorizer_weights() against the direct computation."""

    from .lp import LpConfig, covering_lp_data, majorizer_weights, fprime as lp_fprime

    Abar, y, labels, W, config, coarse = _lp_cell(K=4)
    rng = np.random.default_rng(17)
    K = W.shape[1]
    Q = np.abs(rng.random((coarse.nbeta, K))) + 0.1

    vmap = VarianceMap.from_factors(config, 0, Q, W, L=coarse.L)
    # _lp_cell rescales Abar and y by the same power of two, so the reference map has to
    # carry the SCALED y_true: majorizer_weights reads it from the map, not from the caller.
    ref = coarse.replace(A=Abar, y_true=y, history_record=dict(step='test'))

    for ibeta in (0, 3, coarse.nbeta - 1):
        cost, M, b = covering_lp_data(vmap, ref, ibeta)
        assert np.array_equal(cost, W.sum(axis=0))
        assert M is not None and np.array_equal(np.asarray(M), W)
        assert np.array_equal(b, Abar[ibeta])
    # The clip is applied when a config is given, and never in place on the reference.
    _, _, bc = covering_lp_data(vmap, ref, 0, LpConfig.for_qstep(clip_rel=0.5))
    assert np.any(bc == 0.0) and np.array_equal(ref.rows(0, 1)[0], Abar[0])

    # The majorization weights are a sum over FINE alpha with Q row-duplicated. Getting the
    # per-group accumulation wrong silently weights every group equally, so the reference
    # here is built the other way round: fine first, group second. Rows the distance does not
    # score carry weight ZERO -- unfloored, an output with no variance has weight ~1e14 and
    # would own the objective outright.
    g = majorizer_weights(vmap, ref)
    y_app = (Q @ W.sum(axis=0))[labels]
    scored = y >= YTRUE_FLOOR
    w = np.zeros_like(y)
    w[scored] = lp_fprime(y_app[scored] / y[scored]) / y[scored]
    g_ref = Q.T @ np.bincount(labels, weights=w, minlength=coarse.nbeta)
    assert not np.all(scored), 'this cell no longer exercises the y_true floor'
    assert np.allclose(g, g_ref, rtol=1e-14, atol=0), np.abs(g - g_ref).max()
    # ... and it is NOT the same as weighting every group equally, so the test has teeth.
    g_flat = Q.T @ np.bincount(labels, minlength=coarse.nbeta).astype(float)
    assert not np.allclose(g, g_flat, rtol=1e-3)

    atomic_print(f'    test_lp_building_blocks: (cost, M, b) for {coarse.nbeta} groups, and'
                 ' the majorizer accumulated per group')


####################################   factorizations (section 8)   ####################################
#
# These check the methods that CHOOSE a factorization. Two things they are careful about:
#
#   - the semiorthogonality flags are CLAIMS, so the tests verify them numerically rather than
#     reading them back. A flag the class sets and the tests only echo would be worth nothing.
#   - the pinned column is checked for the property it exists for (it is nonnegative, and it
#     dominates every group) rather than for its index, since every failure mode that has
#     actually been seen is a lost or rotated column rather than a lost integer.


def _basis_cell(r=6, subband_counts=(1, 1), L=None, seed=17, nzero=1):
    """(coarse ref, fine map, rng) at a small but REAL geometry -- the label arithmetic and the
    coarse-graining are half of what the steps have to get right."""

    config = _make_test_config(r, list(subband_counts))
    rng = np.random.default_rng(seed)
    fine = _random_map(config, 0, rng, nzero=nzero)
    L = (fine.pf_rank + 1) if (L is None) else L
    return fine.coarse_grain(L), fine, rng


def _decaying_map(r=6, subband_counts=(1, 1), K=8, seed=19, rate=0.5):
    """A coarse map whose spectrum DECAYS, built as a nonnegative low-rank product plus noise.

    _random_map()'s iid matrix has a nearly flat spectrum, which is the worst case for any
    low-rank method and tells a randomized range finder nothing. A real variance map is not
    like that, so anything comparing an approximate SVD against an exact one needs this instead.

    'rate' is the geometric ratio between successive modes. A real variance map decays SLOWLY,
    and that is the regime where a randomized range finder is hard and where its sampling
    settings show up -- so a test about those settings has to ask for one.
    """

    config = _make_test_config(r, list(subband_counts))
    rng = np.random.default_rng(seed)
    fine = _random_map(config, 0, rng)
    nbeta, nfreq = fine.coarse_grain(fine.pf_rank + 1).shape

    s = float(rate) ** np.arange(K)
    A = (rng.uniform(0.2, 1.0, size=(nbeta, K)) * s) @ rng.uniform(0.2, 1.0, size=(K, nfreq))
    A += 1e-6 * rng.uniform(0.0, 1.0, size=A.shape)
    coarse = fine.coarse_grain(fine.pf_rank + 1)
    return coarse.replace(A=A, history_record=dict(step='synthetic'))


def test_svd(r=6, subband_counts=(1, 1), K=5):
    """svd() and truncate(): the dense path against numpy, the factored path against the dense
    one, and the flags against the matrices they describe."""

    ref, fine, rng = _basis_cell(r, subband_counts)
    A = np.asarray(ref.dense(), dtype=np.float64)
    U, s, Vt = np.linalg.svd(A, full_matrices=False)

    m = ref.svd(K)
    assert m.is_factored and (m.factor_rank == K)
    assert not m.is_admissible, 'a truncated SVD has an admissibility cliff and must say so'
    assert np.allclose(m.dense(), U[:, :K] @ np.diag(s[:K]) @ Vt[:K])
    # The flags are claims; verify them rather than echoing them.
    assert m.Q_is_semiorthogonal and m.W_is_semiorthogonal
    for X in (np.asarray(m.Q), np.asarray(m.W)):
        assert np.max(np.abs(X.T @ X - np.eye(K))) < 1e-12
    assert np.allclose(np.asarray(m.mid), np.diag(np.diag(np.asarray(m.mid))))

    # 'eps' drops modes below a relative floor, and factor_rank and eps compose.
    frac = s[K-1] / s[0]
    assert ref.svd(K, eps=0.5 * frac).factor_rank == K
    assert ref.svd(K, eps=1.5 * frac).factor_rank < K
    assert ref.svd(eps=1.0e-13).factor_rank <= min(A.shape)
    for bad in (lambda: ref.svd(), lambda: ref.svd(0)):
        try:
            bad()
            raise AssertionError('svd() should refuse this')
        except RuntimeError:
            pass

    # THE FACTORED PATH IS A DIFFERENT ALGORITHM -- two thin QRs and a K-by-K SVD, with no dense
    # product anywhere -- so agreeing with the dense one is a real check and not a tautology.
    # This is the rank-reduction path, and it is the reason svd() is a method at all.
    hi = ref.svd(3 * K).canonicalize_signs()
    dense_hi = hi.replace(A=hi.dense(), history_record=dict(step='densify'))
    lo_f, lo_d = hi.svd(K), dense_hi.svd(K)
    scale = float(np.abs(np.asarray(dense_hi.A)).max())
    assert np.max(np.abs(lo_f.dense() - lo_d.dense())) < 1e-12 * scale
    assert np.max(np.abs(np.asarray(lo_f.W).T @ np.asarray(lo_f.W) - np.eye(K))) < 1e-12

    # shape_normalize decomposes the unit-sum SHAPES and folds the row sums back into Q, which
    # is exactly why Q is then NOT semiorthogonal and truncate() refuses the result.
    S = A / ref.row_sums()[:, None]
    Us, ss, Vst = np.linalg.svd(S, full_matrices=False)
    sn = ref.svd(K, shape_normalize=True)
    assert np.allclose(sn.dense(), ref.row_sums()[:, None] * (Us[:, :K] @ np.diag(ss[:K])
                                                             @ Vst[:K]))
    assert (not sn.Q_is_semiorthogonal) and sn.W_is_semiorthogonal
    # The default is 'choose by rank', at the measured crossover.
    assert ref.svd(VarianceMap._SHAPE_NORMALIZE_RANK).history[-1]['shape_normalize']
    assert not ref.svd(K).history[-1]['shape_normalize']

    # The randomized range finder: 1 + 2*power_iters blocked passes, nothing of matrix size in
    # memory. Checked on a SLOWLY decaying spectrum, which is what a real variance map has and
    # the only regime where the sampling settings matter at all.
    dec = _decaying_map(r, subband_counts, K=24, rate=0.85)
    ex = dec.svd(K, method='exact', shape_normalize=False)
    scale = float(np.abs(np.asarray(ex.dense())).max())

    def err_of(m):
        """(RMS, WORST-CASE) error against the exact truncation. The second is the one that
        matters: a basis at the textbook sampling passes an RMS test comfortably and still
        costs 1.4x in delivered D, because D is paid on each group's worst channel."""
        d = np.asarray(m.dense()) - np.asarray(ex.dense())
        return (float(np.linalg.norm(d)) / float(np.linalg.norm(np.asarray(ex.dense()))),
                float(np.max(np.abs(d))) / scale)

    rd = dec.svd(K, method='randomized', shape_normalize=False, rng=np.random.default_rng(5))
    err, emax = err_of(rd)
    assert err < 1e-3, err
    assert np.max(np.abs(np.asarray(rd.W).T @ np.asarray(rd.W) - np.eye(K))) < 1e-10

    # THE DEFAULTS ARE NOT THE TEXTBOOK ONES, deliberately, and this is what says so: the
    # shipped sampling must beat one power iteration with ten extra samples by a clear margin
    # on the worst-case bar. Without this, a well-meaning revert to the standard settings
    # passes every other test in the suite and costs 1.4x in D on a real map.
    tb_err, tb_emax = err_of(dec.svd(K, method='randomized', shape_normalize=False,
                                     oversample=10, power_iters=1,
                                     rng=np.random.default_rng(5)))
    assert emax < tb_emax / 3.0, (emax, tb_emax)

    # truncate() is the SVD's own prefix, so it must agree with asking svd() for that rank.
    t = m.truncate(2)
    assert (t.factor_rank == 2) and (not t.is_admissible)
    assert np.allclose(t.dense(), ref.svd(2, shape_normalize=False).dense())
    assert t.Q_is_semiorthogonal and t.W_is_semiorthogonal

    # ... and it refuses whenever "leading" is not a property of the column order, rather than
    # quietly returning a prefix of an arbitrary basis.
    for bad, needle in ((sn, 'semiorthogonal'),
                        (m.replace(mid=np.asarray(m.mid) + 0.1,
                                   Q_is_semiorthogonal=True, W_is_semiorthogonal=True),
                         'not diagonal'),
                        (m.replace(pinned_columns=[K-1], Q_is_semiorthogonal=True,
                                   W_is_semiorthogonal=True), 'pinned column')):
        try:
            bad.truncate(2)
            raise AssertionError(f'truncate() should have refused ({needle})')
        except RuntimeError as e:
            assert needle in str(e), str(e)

    try:
        ref.svd(K).replace(A=ref.dense()).truncate(2)
        raise AssertionError('truncate() should refuse a dense map')
    except RuntimeError as e:
        assert 'dense' in str(e), str(e)

    atomic_print(f'    test_svd(r={r}, K={K}): dense path matches numpy, factored path matches'
                 f' the dense one to {np.max(np.abs(lo_f.dense() - lo_d.dense()))/scale:.2g},'
                 f' randomized to {err:.2g} relative ({emax:.2g} worst-case)')


def test_column_algebra(r=6, subband_counts=(1, 1), K=5):
    """The column helpers, each against the property it exists for."""

    from .basis import basis_envelope_column

    ref, fine, rng = _basis_cell(r, subband_counts)
    raw = ref.svd(K)
    A0 = np.array(raw.dense())

    # THE MEASURED FAILURE MODE: numpy's per-mode sign is arbitrary, so a raw SVD basis has zero
    # nonnegative columns and everything that needs one fails. Canonicalization is exactly
    # invariant -- a sign flip is exact in floating point -- so there is never a reason to skip
    # it, and this asserts the invariance bitwise rather than to a tolerance.
    can = raw.canonicalize_signs()
    assert np.array_equal(can.dense(), A0), 'canonicalize_signs() is not bitwise invariant'
    assert np.all(np.asarray(can.W).sum(axis=0) >= 0.0)
    assert can.n_nonneg_cols() >= 1 and (raw.n_nonneg_cols() == 0)
    assert can.Q_is_semiorthogonal and can.W_is_semiorthogonal
    assert can.canonicalize_signs().history[-1]['n_flipped'] == 0, 'not idempotent'

    # rescale_columns() is provably inert and is measured to be worth up to 1.49x anyway; what a
    # test can check is the inertness and where the scale went.
    rs = can.rescale_columns()
    assert np.allclose(np.linalg.norm(np.asarray(rs.W), axis=0), 1.0)
    assert np.max(np.abs(rs.dense() - A0)) < 1e-12 * np.abs(A0).max()
    assert np.array_equal(np.asarray(rs.Q), np.asarray(can.Q)), 'the scale belongs in mid'
    assert rs.Q_is_semiorthogonal and (not rs.W_is_semiorthogonal)
    try:
        can.rescale_columns(mode='sum')
        raise AssertionError('rescale_columns() should refuse an unmeasured convention')
    except RuntimeError as e:
        assert 'not implemented' in str(e), str(e)

    # A pinned column is a certificate: nonnegative, and dominating every group. Check the
    # property, not the bookkeeping.
    w = basis_envelope_column(ref)
    assert np.all(np.asarray(ref.dense()) <= w[None, :] + 1e-15)
    assert np.allclose(w, np.asarray(ref.dense()).max(axis=0))

    pin = can.pin_column(w)
    assert (pin.factor_rank == K) and (list(pin.pinned_columns) == [0])
    assert np.array_equal(np.asarray(pin.W)[:, 0], w) and (pin.n_nonneg_cols() >= 1)
    assert not pin.is_admissible, 'replacing a column changes the product'
    grow = can.pin_column(w, replace_last=False)
    assert (grow.factor_rank == K + 1) and np.array_equal(grow.dense(), A0), \
        'appending a column with a zero coefficient must be bitwise inert'

    for bad in (-np.abs(w), np.zeros_like(w), w[:-1]):
        try:
            can.pin_column(bad)
            raise AssertionError('pin_column() should refuse this w')
        except RuntimeError:
            pass

    # Pinned indices are REMAPPED, never carried: dropping a column shifts every index above it,
    # and the resulting mis-pin is silent until a repair goes looking for its nonnegative column.
    two = pin.pin_column(w, replace_last=False)              # pins at 0 and 1
    assert list(two.pinned_columns) == [0, 1]
    sel = two.select_columns([3, 1, 0])
    assert (sel.factor_rank == 3) and (list(sel.pinned_columns) == [2, 1])
    assert np.allclose(sel.dense(), np.asarray(two.Q)[:, [3, 1, 0]]
                       @ np.asarray(two.mid)[np.ix_([3, 1, 0], [3, 1, 0])]
                       @ np.asarray(two.W)[:, [3, 1, 0]].T)
    assert not sel.is_admissible
    try:
        two.select_columns([1, 2, 3])
        raise AssertionError('select_columns() should refuse to drop a pinned column')
    except RuntimeError as e:
        assert 'pinned column' in str(e), str(e)

    # augment_basis() appends with zero coefficients, so the product is bitwise unchanged and
    # an admissible map stays admissible; pinned indices need no remap, since appending shifts
    # nothing.
    base = can.replace(is_admissible=True).pin_column(w, replace_last=False)
    assert base.is_admissible, 'a zero-coefficient append cannot break admissibility'
    aug = base.augment_basis(np.asarray(can.W)[:, :2])
    assert aug.factor_rank == base.factor_rank + 2
    assert np.array_equal(aug.dense(), base.dense())
    assert aug.is_admissible and (list(aug.pinned_columns) == [0])

    # with_basis() takes a W from anywhere and hands back something ready for a qstep().
    wb = ref.with_basis(np.asarray(can.W), pinned_columns=[0])
    assert wb.is_factored and (not wb.is_admissible)
    assert np.count_nonzero(np.asarray(wb.Q)) == 0
    assert np.array_equal(np.asarray(wb.W), np.asarray(can.W))

    atomic_print(f'    test_column_algebra(r={r}, K={K}): sign canonicalization bitwise inert'
                 f' ({raw.n_nonneg_cols()} -> {can.n_nonneg_cols()} nonnegative columns),'
                 ' column scaling inert, pinned indices remapped')


def test_reorthogonalize(r=6, subband_counts=(1, 1), K=6):
    """reorthogonalize(): the same matrix, a semiorthogonal W, and the pinned column intact.

    The last is the whole reason for the ordered QR. A plain rotation destroys the nonnegative
    column that the seed and the additive repair depend on, which is measured at 1.769x in D and
    is reproduced here on purpose.
    """

    from .basis import basis_envelope_column

    ref, fine, rng = _basis_cell(r, subband_counts)
    w = basis_envelope_column(ref)
    m = ref.svd(K).canonicalize_signs().pin_column(w).replace(is_admissible=True)
    A0 = np.array(m.dense())
    scale = float(np.abs(A0).max())

    ro = m.reorthogonalize()
    assert ro.factor_rank == K
    assert np.max(np.abs(ro.dense() - A0)) < 1e-12 * scale, 'the re-expression must be exact'
    assert ro.W_is_semiorthogonal and (not ro.Q_is_semiorthogonal)
    Wr = np.asarray(ro.W)
    assert np.max(np.abs(Wr.T @ Wr - np.eye(K))) < 1e-12
    assert ro.is_admissible, 'nothing changed but the factorization'

    # The first pinned column comes through as a POSITIVE multiple of itself -- so it is still
    # nonnegative and q = c e_0 still certifies every group -- but it is rescaled to unit norm,
    # so an equality test against the envelope column would fail and a direction test is right.
    c0 = Wr[:, 0]
    assert list(ro.pinned_columns) == [0]
    assert np.max(np.abs(c0 / np.linalg.norm(c0) - w / np.linalg.norm(w))) < 1e-12
    assert (c0.min() >= 0.0) and (ro.n_nonneg_cols() >= 1)

    # The guarantee is about the ORDERING, so it has to be tested with the pin somewhere other
    # than column 0, where a plain QR would preserve it by accident.
    moved = m.select_columns([1, 2, 0, 3, 4, 5])
    assert list(moved.pinned_columns) == [2]
    kept = np.asarray(moved.reorthogonalize().W)[:, 0]
    assert np.max(np.abs(kept/np.linalg.norm(kept) - w/np.linalg.norm(w))) < 1e-12

    # keep_pinned=False is a plain rotation in the columns' own order. It DROPS the pinned set,
    # and the pinned column does not survive it in any position -- which is the arm of the
    # comparison that has never been measured, and the reason the flag exists at all.
    plain = moved.reorthogonalize(keep_pinned=False)
    assert plain.pinned_columns.size == 0
    assert np.max(np.abs(plain.dense() - moved.dense())) < 1e-12 * scale
    Wp = np.asarray(plain.W)
    wn = w / np.linalg.norm(w)
    assert not any(np.max(np.abs(Wp[:, c]/np.linalg.norm(Wp[:, c]) - wn)) < 1e-8
                   for c in range(Wp.shape[1])), 'the plain rotation kept the pinned column'

    atomic_print(f'    test_reorthogonalize(r={r}, K={K}): exact to'
                 f' {np.max(np.abs(ro.dense()-A0))/scale:.2g}, pinned column preserved up to a'
                 ' positive scale, and lost by the plain rotation')


def test_basis_constructors(r=6, subband_counts=(1, 1), K=4):
    """Every module-level basis constructor, through the one thing they are all for: a Q-step
    against it produces an admissible map."""

    from . import basis as vb

    ref, fine, rng = _basis_cell(r, subband_counts)
    A = np.asarray(ref.dense())

    W_svd = vb.basis_svd(ref, K)
    assert np.array_equal(W_svd, np.asarray(ref.svd(K).W))

    tree = vb.greedy_envelope_tree(ref)
    W_greedy = vb.basis_greedy_envelope(ref, K, tree=tree)
    assert np.array_equal(W_greedy, vb.basis_greedy_envelope(ref, K)), 'tree reuse changed it'
    # Every atom is a max-envelope of a set of the map's own rows, hence nonnegative and >= each
    # of its members. At rank 1 there is only one cluster, so the atom is the envelope of the
    # whole map -- of the unit-sum SHAPES by default, which is the difference 'on_shapes' makes
    # and the reason it is the default (the Q-step is exactly scale-invariant in each atom).
    assert W_greedy.min() >= 0.0
    S = A / A.sum(axis=1)[:, None]
    assert np.allclose(vb.basis_greedy_envelope(ref, 1, tree=tree)[:, 0], S.max(axis=0))
    assert np.allclose(vb.basis_greedy_envelope(ref, 1, on_shapes=False)[:, 0], A.max(axis=0))

    # THE REDUCED FORM'S WHOLE TRICK: the merge objective is summed over the FINE rows inside
    # each cluster, so groups enter weighted by their SIZE. Getting that wrong -- one unit of
    # weight per group -- is silent, since it still returns a plausible basis, so it is checked
    # against the same clustering run group-blind. Group sizes differ by 2^l across subbands
    # here, which is what makes the two answers differ at all.
    A_sh = A / A.sum(axis=1)[:, None]
    lab = ref.alpha_to_beta_block(0, ref.nalpha)
    keep = np.asarray(ref.y_true) >= YTRUE_FLOOR
    flat = vb._AgglomerativeEnvelope(A_sh, np.ones(ref.nbeta),
                                     np.arange(ref.nbeta, dtype=np.int64))
    sized = vb._AgglomerativeEnvelope(A_sh, np.ones(int(keep.sum())), lab[keep])
    assert np.allclose(sized.basis(K), W_greedy), 'the default tree is not the sized one'
    assert not np.allclose(flat.basis(K), W_greedy), \
        'this cell no longer distinguishes size-weighted merging from group-blind merging'

    W_qr = vb.basis_pivoted_qr(ref, K)
    # Its atoms are literally rows of the map, which is where the nonnegativity comes from.
    for c in range(K):
        assert np.any(np.all(np.abs(A - W_qr[:, c][None, :]) < 1e-15, axis=1)), c

    W_rand = vb.basis_random(ref, K, rng=np.random.default_rng(7))
    assert (W_rand.shape == (ref.nfreq, K)) and (W_rand.min() >= 0.0)

    D = {}
    for name, W in (('svd', W_svd), ('greedy', W_greedy), ('pivoted_qr', W_qr),
                    ('random', W_rand)):
        m = ref.with_basis(W).canonicalize_signs().qstep(ref, workers=1)
        assert m.is_admissible, name
        assert m.measure_admissibility(ref).admissible, name
        D[name] = m.get_distance()

    # svd_init() is exactly the chain it documents, so it must reproduce it call for call.
    chain = ref.svd(K).canonicalize_signs().rescale_columns().qstep(ref, workers=1)
    made = vb.svd_init(ref, K, workers=1)
    assert np.array_equal(np.asarray(made.Q), np.asarray(chain.Q))
    assert np.array_equal(np.asarray(made.W), np.asarray(chain.W))
    pinned = vb.svd_init(ref, K, pin_envelope=True, workers=1)
    assert (pinned.factor_rank == K) and (list(pinned.pinned_columns) == [0])
    assert pinned.is_admissible

    # The cheap predictors, against their own definitions rather than against an LP.
    assert vb.spectrum_effective_rank(ref, threshold=1.0) == 1
    assert vb.spectrum_effective_rank(ref, threshold=0.0) >= 1
    cover = vb.shape_cover_statistic(ref, ref)
    # Every group covers itself exactly, so the self-cover is 1 and the statistic is calibrated.
    assert np.allclose(cover, 1.0), cover.max()

    atomic_print('    test_basis_constructors(K=%d): four bases, all admissible after a Q-step;'
                 ' D = %s' % (K, ', '.join(f'{k} {v:.4g}' for k, v in D.items())))


def test_map_steps(r=6, subband_counts=(1, 1), K=5):
    """qstep() / wstep() / repair(): the wrappers against the array level they wrap.

    The numerics are varmap.lp's and are tested there. What is tested here is everything the
    wrapper adds and could get wrong: the reference matrix, the labels, folding 'mid', the
    pinned set, and where is_admissible comes from.
    """

    from . import lp
    from .basis import basis_envelope_column

    ref, fine, rng = _basis_cell(r, subband_counts)
    Abar = np.asarray(ref.dense(), dtype=np.float64)
    init = (ref.svd(K).canonicalize_signs().pin_column(basis_envelope_column(ref)))
    W0, Q0 = np.asarray(init.W, dtype=np.float64), np.asarray(init.Q, dtype=np.float64)

    # The one-hot seed is admissible BY CONSTRUCTION, which is what makes it a usable fallback
    # for a failed subproblem: one nonnegative atom per group, scaled until it dominates. It
    # must not depend on where the scale is kept, so a non-identity mid gives the same map.
    sd = init.seed_onehot(ref)
    assert sd.is_admissible and sd.measure_admissibility(ref).admissible
    assert np.all(np.count_nonzero(np.asarray(sd.Q), axis=1) == 1), 'not one-hot'
    assert abs(init.rescale_columns().seed_onehot(ref).get_distance()
               - sd.get_distance()) < 1e-12 * sd.get_distance()
    try:
        ref.svd(K).seed_onehot(ref)
        raise AssertionError('seed_onehot() needs a nonnegative column')
    except RuntimeError as e:
        assert 'no nonnegative column' in str(e), str(e)

    # BIT-IDENTICAL to the array level, which is what makes lp.py's equivalence gate a gate on
    # this too. Anything the wrapper assembled wrongly -- the reference, the seed, the config --
    # shows up here as a difference rather than as a plausible number.
    cq = lp.LpConfig.for_qstep(nonneg=False)
    m = init.qstep(ref, cfg=cq, workers=1)
    Q, W, _ = lp.q_step(Abar, W0, cq, Q0=Q0, workers=1)
    assert np.array_equal(np.asarray(m.Q), Q) and np.array_equal(np.asarray(m.W), W)
    assert m.is_admissible and (m.factor_rank == K)
    assert m._mid_is_identity(), 'the LP chooses the coefficients outright'
    assert list(m.pinned_columns) == [0]
    assert (not m.Q_is_semiorthogonal) and m.measure_admissibility(ref).admissible

    rec = m.history[-1]
    assert (rec['step'] == 'qstep') and (rec['config'] == cq)
    assert abs(rec['D'] - m.get_distance()) < 1e-15

    # 'mid' is folded into Q on the way down, since lp works in the mid-free convention. A map
    # and its folded equivalent must therefore give the identical step.
    scaled = init.rescale_columns()
    assert not scaled._mid_is_identity()
    folded = scaled.replace(Q=scaled._QM(), mid=None, W=np.asarray(scaled.W),
                            history_record=dict(step='fold'))
    assert np.array_equal(np.asarray(scaled.qstep(ref, cfg=cq, workers=1).Q),
                          np.asarray(folded.qstep(ref, cfg=cq, workers=1).Q))

    # is_admissible is INHERITED, not asserted: admissibility is transitive, so an uncertified
    # reference certifies nothing however well the LP solved.
    loose = ref.replace(is_admissible=False, history_record=dict(step='uncertify'))
    assert not init.qstep(loose, cfg=cq, workers=1).is_admissible
    assert not init.qstep(ref, cfg=cq, repair=False, workers=1).is_admissible

    # ... and it must come from the CONFIG too, not from the repair= kwarg alone: the three
    # repair stages are config fields, so a config selecting none of them repairs nothing
    # however the kwarg is set, and a map claiming to dominate the reference after nothing was
    # done to it is the one error the one-sided distance exists to prevent.
    craw = lp.LpConfig.for_qstep(nonneg=False, rescale='none')
    assert not init.qstep(ref, cfg=craw, workers=1).is_admissible
    assert not init.qstep(ref, cfg=lp.LpConfig.for_qstep(nonneg=False,
                                                        single_shot_repair=True),
                          workers=1).is_admissible
    try:
        init.qstep(ref, cfg=cq, repair=False, workers=1).repair(ref, cfg=craw)
        raise AssertionError('repair() should refuse a config that repairs nothing')
    except RuntimeError as e:
        assert 'no repair stage' in str(e), str(e)

    # SOLVE ONCE, REPAIR SEVERAL WAYS, at map level: repairing the stored raw point with the
    # step's own config reproduces the step's own output exactly.
    raw = init.qstep(ref, cfg=cq, repair=False, workers=1)
    assert np.array_equal(np.asarray(raw.repair(ref, cfg=cq).Q), np.asarray(m.Q))
    assert raw.repair(ref, cfg=cq).is_admissible

    # The additive stages are defined on the COLUMNS of W, so a wrapper holding a non-identity
    # mid has to FOLD it; the multiplicative stage takes mid directly and leaves it in place.
    # Both are exercised from the same inadmissible point, which carries a non-identity mid.
    ca = lp.LpConfig.for_qstep(nonneg=False, additive_first=True, additive_last=True)
    shrunk = m.rescale_columns().inflated(0.6)
    assert not (shrunk._mid_is_identity() or shrunk.is_admissible)

    add = shrunk.repair(ref, cfg=ca)
    assert add._mid_is_identity(), 'the additive lift needs mid folded into Q'
    assert add.is_admissible and add.measure_admissibility(ref).admissible

    mul = shrunk.repair(ref, cfg=lp.LpConfig.for_qstep(nonneg=False))
    assert not mul._mid_is_identity(), 'a multiplicative row scale commutes with mid'
    # A multiplicative repair passes its OWN test -- every positive entry of the reference is
    # dominated -- and is structurally blind to a product entry that went negative where the
    # reference is zero. That is why the additive stage exists, and why only the arm above is
    # asserted to be admissible elementwise.
    P = np.asarray(mul.dense())
    with np.errstate(divide='ignore', invalid='ignore'):
        assert float(np.nanmax(np.where(Abar > 0, Abar / P, 0.0))) <= 1.0

    # ... and it raises rather than silently falling back, which would look exactly like the
    # additive repair not helping.
    try:
        ref.svd(K).replace(is_admissible=False).repair(ref, cfg=ca)
        raise AssertionError('repair() should refuse an additive stage with no nonneg column')
    except RuntimeError as e:
        assert 'NONNEGATIVE column' in str(e), str(e)

    # The W-step: same bit-identity, the pinned column held fixed, and the majorize-minimize
    # guarantee its own objective has to satisfy.
    cw = lp.LpConfig.for_wstep(nonneg=False)
    labels = ref.alpha_to_beta_block(0, ref.nalpha)
    m2 = m.wstep(ref, cfg=cw, workers=1)
    Q2, W2, iw = lp.w_step(Abar, np.asarray(m.Q), np.asarray(ref.y_true), labels,
                           np.asarray(m.W), cw, pinned=[0], workers=1)
    assert np.array_equal(np.asarray(m2.W), W2) and np.array_equal(np.asarray(m2.Q), Q2)
    assert m2.history[-1]['w_obj_raw'] <= m2.history[-1]['w_obj_before'] * (1 + 1e-9)
    assert m2.is_admissible and (not m2.W_is_semiorthogonal)
    # The 'cols' repair scales whole channels UP, so a pinned column stays nonnegative and still
    # dominates every group: everything the pin buys survives a W-step.
    assert np.all(np.asarray(m2.W)[:, 0] >= np.asarray(m.W)[:, 0] - 1e-12)

    try:
        m.wstep(ref.replace(y_true=None, history_record=dict(step='drop y')), cfg=cw, workers=1)
        raise AssertionError('wstep() needs y_true')
    except RuntimeError as e:
        assert 'y_true' in str(e), str(e)

    # An alternation written out at the call site, which is what there is instead of a driver.
    # The Q-step is exactly optimal given W, so D cannot rise across one.
    seq = [m.get_distance()]
    cur = m
    for _ in range(2):
        cur = cur.wstep(ref, cfg=cw, workers=1).qstep(ref, cfg=cq, workers=1)
        seq.append(cur.get_distance())
    assert all(seq[i+1] <= seq[i] * (1 + 1e-9) for i in range(len(seq)-1)), seq

    # Geometry mismatches are refused by name rather than being broadcast into nonsense.
    for bad, needle in ((fine, 'shape mismatch'), (ref.coarse_grain(ref.L + 1), 'shape')):
        try:
            m.qstep(bad, cfg=cq, workers=1)
            raise AssertionError('qstep() should have refused this ref')
        except RuntimeError as e:
            assert needle in str(e), str(e)
    try:
        ref.qstep(ref, cfg=cq, workers=1)
        raise AssertionError('qstep() should refuse a dense self')
    except RuntimeError as e:
        assert 'dense' in str(e), str(e)

    atomic_print(f'    test_map_steps(r={r}, K={K}): wrappers bit-identical to varmap.lp, mid'
                 f' folded for the additive repair, D {seq[0]:.6g} -> {seq[-1]:.6g} over two'
                 ' alternations')


def test_report(r=6, subband_counts=(1, 1), K=4):
    """varmap/report.py: the record is assembled from the map, and survives a json round trip.

    The property worth testing is not the formatting -- it is that a record says what the map
    says, since a results table that drifts from the map it describes is worse than no table.
    """

    import json
    import os
    import tempfile

    from . import basis as vb
    from . import report as vr

    ref, fine, rng = _basis_cell(r, subband_counts)
    m = vb.svd_init(ref, K, workers=1)

    D = m.get_distance()
    rec = vr.row_dict(m, D, name='svd_init')
    for key, want in (('name', 'svd_init'), ('factor_rank', K), ('is_factored', True),
                      ('nalpha', m.nalpha), ('nbeta', m.nbeta), ('nfreq', m.nfreq),
                      ('is_coarse_grained', True), ('L', m.L), ('nscored', m.nscored),
                      ('itree', m.itree), ('admissible', True),
                      ('apply_cost', m.apply_cost())):
        assert rec[key] == want, (key, rec[key], want)
    assert abs(rec['D'] - D) < 1e-15
    # Without a measurement, 'admissible' is the map's own FLAG and the elementwise fields are
    # absent rather than guessed at.
    assert 'max_r' not in rec

    # With one, the measurement wins -- including when it CONTRADICTS the flag, which is the
    # case the distinction exists for.
    adm = m.measure_admissibility(ref, inflate=True)
    rec2 = vr.row_dict(m, D, adm=adm)
    assert rec2['admissible'] and (abs(rec2['max_r'] - adm.max_r) < 1e-15)
    assert (len(rec2['argmax_r']) == 2) and (rec2['inflation'] is not None)

    lying = m.inflated(0.5).replace(is_admissible=True,
                                    history_record=dict(step='lie'))
    bad = lying.measure_admissibility(ref, inflate=True)
    liar = vr.row_dict(lying, np.inf, adm=bad)
    assert liar['admissible'] is False, 'the measurement must override the flag'
    assert (liar['max_r'] > 1.0) and np.isfinite(liar['D_inflated'])
    assert liar['D_inflated'] > D, 'inflating to admissibility cannot improve D'

    # 'extra' is where an experiment puts what nobody anticipated, and it takes numpy scalars
    # straight from a step's info dict -- which is exactly what json.dump() refuses.
    info = m.history[-1]
    rec3 = vr.row_dict(m, D, extra=dict(max_r_raw=np.float64(info['max_r_raw']),
                                        n_lp=np.int64(info['n_lp']), tag='x'))
    assert isinstance(rec3['max_r_raw'], float) and isinstance(rec3['n_lp'], int)

    # frontier(): one record per rank, K is what was ASKED for, and D falls with rank.
    ranks = [2, 4, 6]
    rows = vr.frontier(ref, lambda rf, K_: vb.svd_init(rf, K_, workers=1), ranks,
                       name='svd', measure=True, inflate=True)
    assert [r_['K'] for r_ in rows] == ranks
    assert all(r_['factor_rank'] == r_['K'] for r_ in rows)
    assert all(r_['admissible'] and (r_['algo_seconds'] > 0.0) for r_ in rows)
    assert rows[-1]['D'] < rows[0]['D'], [r_['D'] for r_ in rows]

    # An INADMISSIBLE map is reported with D = inf rather than raising, and 'inflate' is what
    # tells "a rescale fixes this" from "hopeless". A raw SVD is exactly that case.
    raw = vr.frontier(ref, lambda rf, K_: rf.svd(K_), [2], name='svd-raw',
                      measure=True, inflate=True)[0]
    assert (not raw['admissible']) and np.isinf(raw['D'])
    assert raw['max_r'] > 1.0 and (raw['inflation'] > 1.0)

    # The table prints every row, and the json round trip preserves the record INCLUDING the
    # infinities, which is the one thing plain json.dump gets wrong.
    tbl = vr.format_table(rows + [raw])
    assert len(tbl.split('\n')) == len(rows) + 3, tbl      # 2 header lines + one per record
    assert 'inf' in vr.format_table([raw])
    assert 'D=inf' in vr.format_row(raw)
    assert 'not measured' in vr.format_row(rec), vr.format_row(rec)

    with tempfile.TemporaryDirectory() as d:
        path = os.path.join(d, 'rows.json')
        vr.save_json(rows + [raw, rec], path)
        # PORTABLE json, which is the whole reason save_json() exists rather than json.dump():
        # a bare dump writes 'Infinity', which python reads back and nothing else does. So the
        # test is on the file's TEXT, not on whether python can reload it.
        text = open(path).read()
        assert 'Infinity' not in text, 'save_json wrote non-portable infinities'
        assert '"inf"' in text
        json.load(open(path))
        back = vr.load_json(path)
    assert len(back) == len(rows) + 2
    assert np.isinf(back[-2]['D']) and (back[-2]['D'] > 0)
    assert back[0] == rows[0], 'the round trip changed a record'

    atomic_print(f'    test_report(r={r}, K={K}): record matches the map, measurement overrides'
                 f' the flag, D {rows[0]["D"]:.4g} -> {rows[-1]["D"]:.4g} over ranks {ranks},'
                 ' json round trip exact')


####################################   the brute-force sweep   ###################################
#
# These need a DedispersionPlan and (for the GPU sweep) a device, which the rest of this file
# deliberately does not -- so they are dispatched separately, by run_sweep_tests(). They are
# also minutes rather than seconds: each one runs at least one full sweep over every input
# channel.


def _make_test_detrender(config, n_phi=2, n=2, W=4, nzone=2, kint=3):
    """A Detrender2dParams matching 'config', for the sweep tests."""

    from ..pirate_pybind11 import Detrender2dParams
    from ..detrending_spline.masks import zoned_knots

    nfreq = int(config.get_total_nfreq())
    kv = zoned_knots(n_phi, nfreq, nzone, kint)

    return Detrender2dParams(nfreq=nfreq, knots=[int(x) for x in kv.knots], M=1, n_phi=n_phi,
                             n=n, W=W, T=int(config.time_samples_per_chunk))


def _abcd(m):
    """A map's matrix as the (2^(r-R), M, P, nfreq) array the analytic references are indexed
    by. The sweep's own (nalpha, nfreq) layout is that array with its first three axes
    flattened, so this is a reshape and not a transpose."""

    D = 1 << (m.tree_rank - m.pf_rank)
    return np.asarray(m.A).reshape(D, m.nmultiplets, m.nprofiles, m.nfreq)


def test_sweep_vs_per_tfm(r=7, subband_counts=None, num_primary_trees=1, num_early_triggers=0,
                          verbose=True):
    """The sweep, element by element, against PfAvarExact.per_tfm, which computes the same
    matrix by propagating compressed sparse tiles and shares no code with the dedisperser.

    This is the decisive correctness test, and it doubles as the float32 measurement: the
    sweep runs the float32 ReferenceTree and ReferencePeakFindingKernel, while per_tfm is
    float64 throughout. Only valid with no detrender (per_tfm cannot represent one).
    """

    from ..pirate_pybind11 import DedispersionPlan
    from ..slow_avar.PfVariance import PfAvarExact
    from .brute_force import compute_variance_multimap

    subband_counts = [1] if (subband_counts is None) else subband_counts
    config = _make_test_config(r, subband_counts, num_primary_trees=num_primary_trees,
                               num_early_triggers=num_early_triggers)
    vmm = compute_variance_multimap(config, device='cpu')

    exact = PfAvarExact(DedispersionPlan(config, cdd2_kernel_required=False),
                        np.ones(int(config.get_total_nfreq())))
    worst, worst_where, eps = 0.0, None, []

    for (itree, m) in enumerate(vmm):
        A = _abcd(m)
        D, P = A.shape[0], m.nprofiles
        all_dbits = (1 << (m.tree_rank - m.pf_rank)) - 1
        for ifreq in range(m.nfreq):
            # per_tfm[itree][ifreq][mu] is None for multiplets this channel does not reach.
            want = np.stack([pv.unpack(all_dbits) if (pv is not None) else np.zeros((D, P))
                             for pv in exact.per_tfm[itree][ifreq]])       # (M, D, P)
            want = want.transpose(1, 0, 2)                                 # (D, M, P)
            got = A[:, :, :, ifreq]
            assert got.shape == want.shape, (got.shape, want.shape)

            nz = (want != 0.0)
            if np.any(got[~nz] != 0.0):
                raise RuntimeError(f'test_sweep_vs_per_tfm: tree {itree}, channel {ifreq}: the'
                                   ' sweep is nonzero where per_tfm predicts an exact zero')
            if not np.any(nz):
                continue

            e = got[nz] / want[nz] - 1.0
            eps.append(e)
            k = int(np.argmax(np.abs(e)))
            if abs(float(e[k])) > worst:
                worst, worst_where = abs(float(e[k])), (itree, ifreq, float(e[k]))

    eps = np.concatenate(eps)
    if verbose:
        atomic_print(f'    test_sweep_vs_per_tfm(r={r}, subbands={subband_counts},'
                     f' npri={num_primary_trees}, net={num_early_triggers}): {eps.size} nonzero'
                     f' elements, eps = A_sweep/A_per_tfm - 1: mean {float(np.mean(eps)):+.3g},'
                     f' range [{float(eps.min()):+.3g}, {float(eps.max()):+.3g}], worst |eps|'
                     f' {worst:.3g} at (tree,ifreq)={worst_where[:2]}')

    # Loose enough to pass, tight enough to catch anything but float32 roundoff: the
    # dedispersion chain is float32, so relative errors of a few times 1e-7 are expected.
    assert worst < 1.0e-5, (worst, worst_where)


def test_sweep_phase_collapse(r=7, verbose=True):
    """With no detrender, the 2^gamma polyphase passes of a time-downsampled tree must give
    the same result (notes/variance_map.tex: everything upstream of the downsampler is
    instantaneous in time). This is the sharpest available test of the polyphase logic, and of
    the single-pass shortcut the sweep takes when there is no detrender.

    Agreement is not bit-exact, even though the float32 output samples themselves are:
    shifting the one-hot moves the response relative to the chunk boundaries, so the same set
    of squared samples is accumulated into out_var in a different order. The tolerance below is
    still six orders of magnitude below the float32 noise floor of the dedispersion chain.
    """

    from .brute_force import _CpuSweep, _SweepGeometry

    # Three primary trees => gamma = 0, 1, 2, so the phase loop has something to collapse.
    geom = _SweepGeometry(_make_test_config(r, [2, 2, 1], num_primary_trees=3))
    assert geom.gamma_max == 2, geom.gamma_max

    sweep = _CpuSweep(geom)
    nphases = 1 << geom.gamma_max
    rdd = sweep.make_dedisperser()
    worst = 0.0

    for (ipass, ifreq) in enumerate([0, geom.nfreq // 3, geom.nfreq - 1]):
        ref = None
        for iphase in range(nphases):
            acc = sweep.run_pass(rdd, ifreq, iphase, ipass*nphases + iphase)
            if ref is None:
                ref = acc
                continue
            for itree in range(geom.ntrees):
                # Phases c and c + 2^gamma are the same residue class mod 2^gamma, so they
                # must agree for every tree; with W = 0 all 2^gamma_max phases do.
                scale = float(np.abs(ref[itree]).max())
                e = float(np.abs(acc[itree] - ref[itree]).max()) / scale if (scale > 0) else 0.0
                if e > 1.0e-12:
                    raise RuntimeError(f'test_sweep_phase_collapse: tree {itree}, channel'
                                       f' {ifreq}: phase {iphase} differs from phase 0'
                                       f' (relative {e:.4g})')
                worst = max(worst, e)

    if verbose:
        atomic_print(f'    test_sweep_phase_collapse(r={r}): {nphases} phases agree for all'
                     f' {geom.ntrees} trees, worst relative difference {worst:.3g}')


def test_sweep_column_norms(r=6, subband_counts=None, num_primary_trees=1,
                            num_early_triggers=0, detrender=True, nifreq=2, verbose=True):
    """Evaluates the defining identity ``A[alpha,F] = sum_{t'} L[alpha t, F t']^2`` LITERALLY
    -- one pass per input time t', reading the output of one fixed chunk -- and compares it to
    what the sweep computes, which is instead a sum over output times for one input time.

    This is the test of the core math: the row-norm/column-norm exchange, the polyphase sum
    over 2^gamma phases, and the ntime/t0 sizing. It is also the only test that covers the
    Detrender2d path, since no analytic oracle can. It costs (ntime + nt_in) passes per
    channel, i.e. more than a whole sweep, so it runs at toy scale on a few channels.

    Note that the sum over t' below runs over EVERY input time, with no phase weighting: the
    polyphase decomposition is a way of organizing this sum, and summing it directly is what
    makes this an independent check of that organization.
    """

    from .brute_force import _CpuSweep, _SweepGeometry, compute_variance_multimap

    subband_counts = [2, 1] if (subband_counts is None) else subband_counts
    config = _make_test_config(r, subband_counts, num_primary_trees=num_primary_trees,
                               num_early_triggers=num_early_triggers)
    dparams = _make_test_detrender(config) if detrender else None

    vmm = compute_variance_multimap(config, detrender=dparams, device='cpu')
    A = [_abcd(m) for m in vmm]

    geom = _SweepGeometry(config, detrender=dparams)
    sweep = _CpuSweep(geom)

    # The probe chunk sits far enough into the stream that every t' able to reach it lies in
    # [tlo, thi), and one chunk short of the end so that t' AFTER the probe chunk is covered
    # too -- the detrender is not causal, and reaches W samples back. Both extremes of the
    # range are asserted to contribute nothing, which is what makes the range wide enough.
    nchunks = geom.ndata_chunks + 3
    kprobe = nchunks - 2
    tlo = kprobe*geom.nt_in - geom.ntime
    thi = nchunks*geom.nt_in
    assert tlo >= geom.W, (tlo, geom.W)

    ifreqs = [(i * geom.nfreq) // nifreq + geom.nfreq // (2*nifreq) for i in range(nifreq)]
    worst, worst_where = 0.0, None

    for ifreq in ifreqs:
        col = [np.zeros((geom.tree_D[i], geom.tree_M[i], geom.tree_P[i]))
               for i in range(geom.ntrees)]
        resp = geom.one_hot_response(ifreq)

        for t_in in range(tlo, thi):
            # A fresh dedisperser per t', rather than one continuous stream: a t' near the end
            # of the interval would otherwise leak into the next one, and here correctness
            # matters more than the (toy-scale) cost.
            rdd = sweep.make_dedisperser()
            edge = (t_in == tlo) or (t_in == thi-1)
            for j in range(nchunks):
                rdd.input_array[...] = 0.0
                geom.write_one_hot(rdd.input_array, resp, t_in, j)
                rdd.dedisperse(j, 0)
                if j != kprobe:
                    continue
                for itree in range(geom.ntrees):
                    # out_var is the MEAN over the chunk's nt_ds output times, each in steady
                    # state and so each equal to the same column norm -- so summing out_var
                    # over t' gives A directly, with no nt_ds factor (unlike run_pass(), which
                    # needs one because it sums a single response over time).
                    ov = np.asarray(rdd.out_var[itree])[0]
                    col[itree] += ov
                    if edge and np.any(ov != 0.0):
                        raise RuntimeError(f"test_sweep_column_norms: input time t'={t_in}"
                                           f' still reaches the probe chunk in tree {itree},'
                                           " so the sum over t' is incomplete")

        for itree in range(geom.ntrees):
            want, got = A[itree][:, :, :, ifreq], col[itree]
            scale = float(np.abs(want).max())
            if scale == 0.0:
                assert not np.any(got != 0.0), (itree, ifreq)
                continue
            e = float(np.abs(got - want).max()) / scale
            if e > worst:
                worst, worst_where = e, (itree, ifreq)

    if verbose:
        atomic_print(f'    test_sweep_column_norms(r={r}, subbands={subband_counts},'
                     f' npri={num_primary_trees}, net={num_early_triggers},'
                     f' detrender={bool(detrender)}): {len(ifreqs)} columns x {thi-tlo} input'
                     f' times, worst relative difference {worst:.3g} at'
                     f' (tree,ifreq)={worst_where}')

    # float32 dedispersion, and the two sides accumulate different numbers of terms.
    assert worst < 1.0e-5, (worst, worst_where)


def test_sweep_detrender_fp32(r=8, nifreq=16, verbose=True):
    """Measures the Detrender2d's own float32 penalty, by running the numpy detrender at
    float32 and float64 on the same one-hots.

    The sweep itself runs the detrender at float64 (the rest of the chain is float32, so that
    is the accurate end), but the GPU Detrender2d is float32-only, so this is the error budget
    the GPU sweep inherits from that stage. Reported as the signed relative error on the
    squared norm of each detrended one-hot, which is what enters A.
    """

    from .brute_force import _SweepGeometry

    config = _make_test_config(r, [1])
    dparams = _make_test_detrender(config)
    g64 = _SweepGeometry(config, detrender=dparams, detrender_dtype=np.float64)
    g32 = _SweepGeometry(config, detrender=dparams, detrender_dtype=np.float32)

    eps = []
    for i in range(nifreq):
        ifreq = (i * g64.nfreq) // nifreq + g64.nfreq // (2*nifreq)
        r64 = g64.one_hot_response(ifreq).astype(np.float64)
        r32 = g32.one_hot_response(ifreq).astype(np.float64)
        eps.append(np.sum(r32**2) / np.sum(r64**2) - 1.0)

    eps = np.array(eps)
    if verbose:
        atomic_print(f'    test_sweep_detrender_fp32(r={r}): {nifreq} channels, eps ='
                     f' ||r_fp32||^2/||r_fp64||^2 - 1: mean {float(np.mean(eps)):+.3g}, range'
                     f' [{float(eps.min()):+.3g}, {float(eps.max()):+.3g}]')

    assert float(np.abs(eps).max()) < 1.0e-4, float(np.abs(eps).max())


def test_sweep_gpu_vs_cpu(r=8, subband_counts=None, num_early_triggers=0, detrender=False,
                          nbeams=1, nfreq=None, verbose=True):
    """The GPU sweep against the CPU one, element by element, on the same config.

    Both GPU kernels are separately validated against their reference implementations
    ('pirate_frb test --sbdd' and '--pfsq'), so a discrepancy here points at the driver rather
    than at a kernel. 'nfreq' defaults to 2^r, but is worth varying: that is the case where
    the input channel count and the TREE channel count differ, and the buffers the GPU driver
    allocates are sized by one or the other.
    """

    from .brute_force import compute_variance_multimap

    subband_counts = [2, 2, 1] if (subband_counts is None) else subband_counts
    config = _make_test_config(r, subband_counts, nfreq=nfreq,
                               num_early_triggers=num_early_triggers)
    dparams = _make_test_detrender(config) if detrender else None

    cpu = compute_variance_multimap(config, detrender=dparams, device='cpu')

    config.beams_per_gpu = config.beams_per_batch = nbeams
    if dparams is not None:
        dparams.M = nbeams
    gpu = compute_variance_multimap(config, detrender=dparams, device='gpu')

    worst, worst_where = 0.0, None
    for itree in range(len(cpu)):
        want, got = np.asarray(cpu[itree].A), np.asarray(gpu[itree].A)
        assert got.shape == want.shape, (got.shape, want.shape)
        scale = float(np.abs(want).max())
        e = float(np.abs(got - want).max()) / scale if (scale > 0) else 0.0
        if e > worst:
            worst, worst_where = e, itree

    if verbose:
        atomic_print(f'    test_sweep_gpu_vs_cpu(r={r}, subbands={subband_counts},'
                     f' net={num_early_triggers}, detrender={bool(detrender)},'
                     f' nbeams={nbeams}, nfreq={cpu[0].nfreq}): worst relative difference'
                     f' {worst:.3g} at tree {worst_where}')

    # Both sides are float32 pipelines, but they are different float32 pipelines (the GPU tree
    # is not the reference tree), so this is a float32-roundoff comparison.
    assert worst < 1.0e-4, (worst, worst_where)


def test_sweep_streaming_coarse(r=6, subband_counts=None, num_early_triggers=0,
                                detrender=False, verbose=True):
    """The streaming coarse-graining inside the sweep, against coarse_grain() of the dense map
    -- at EVERY legal L, and required to be bit-identical.

    This is the property the whole scalable path rests on and the one thing the runtime cannot
    check for itself: above test scale the dense A is never formed, so there is nothing to
    compare the streaming reduction against. The two are deliberately different algorithms --
    coarse_grain() sorts a label array and reduces segments, while the sweep exploits the two
    fixed shapes to reduce with no labels at all -- and max being exact is what makes bit
    identity the right bar rather than a tolerance.

    Also checks y_true, which the two paths accumulate identically (channel by channel), and
    that a partial sweep declines to report one at all.
    """

    from .brute_force import compute_variance_multimap

    subband_counts = [2, 1] if (subband_counts is None) else subband_counts
    config = _make_test_config(r, subband_counts, num_early_triggers=num_early_triggers)
    dparams = _make_test_detrender(config) if detrender else None
    dense = compute_variance_multimap(config, detrender=dparams, device='cpu')

    nL = 0
    for (itree, m) in enumerate(dense):
        R, rr = m.pf_rank, m.tree_rank
        for L in range(R, rr + 1):
            Ls = [None] * len(dense)
            Ls[itree] = L
            got = compute_variance_multimap(config, detrender=dparams,
                                           device='cpu', L=Ls)[itree]
            want = m.coarse_grain(L)
            assert got.nbeta == want.nbeta, (itree, L, got.nbeta, want.nbeta)
            nd = int(np.count_nonzero(np.asarray(got.A) != np.asarray(want.A)))
            if nd != 0:
                raise RuntimeError(f'test_sweep_streaming_coarse: tree {itree}, L={L}:'
                                   f' {nd} of {got.nbeta * got.nfreq} entries differ from'
                                   ' coarse_grain() of the dense map')
            assert np.array_equal(got.y_true, m.y_true), (itree, L)
            nL += 1

    # A partial sweep is the one case where y_true would be a sum over the swept channels
    # only, so it is dropped rather than reported.
    chans = [0, dense[0].nfreq // 2, dense[0].nfreq - 1]
    part = compute_variance_multimap(config, detrender=dparams, device='cpu', channels=chans)
    assert part.provenance['partial'] is True
    assert part[0].y_true is None, 'a partial sweep must not claim a y_true'
    Ap, Af = np.asarray(part[0].A), np.asarray(dense[0].A)
    assert np.array_equal(Ap[:, chans], Af[:, chans]), 'swept columns must match a full sweep'
    unswept = [c for c in range(dense[0].nfreq) if c not in chans]
    assert not np.any(Ap[:, unswept]), 'unswept columns must be identically zero'

    if verbose:
        atomic_print(f'    test_sweep_streaming_coarse(r={r}, subbands={subband_counts},'
                     f' net={num_early_triggers}, detrender={bool(detrender)}): the streaming'
                     f' reduction is bit-identical to coarse_grain() at all {nL} legal (tree,'
                     f' L) pairs; a {len(chans)}-channel partial sweep reports no y_true')


####################################   entry point   ####################################


def run_all():
    """Everything, in dependency order: the index arithmetic first, since the rest is built
    on it."""

    test_index_arithmetic()
    test_constructor_validation()
    test_coarse_grain()
    test_distance()
    test_admissibility()
    test_estimate_distance()
    test_check_ref_covers_y_true()
    test_multimap()
    test_dense_float32()
    test_asdf_io()
    test_factored_algebra()
    test_factored_equivalence()
    test_factored_transformations()
    test_factored_validation()
    test_asdf_factored()
    test_lp_config()
    test_lp_primitive()
    test_lp_optimality()
    test_lp_repairs()
    test_lp_steps()
    test_lp_rescue()
    test_lp_negative_rhs()
    test_lp_building_blocks()
    test_svd()
    test_column_algebra()
    test_reorthogonalize()
    test_basis_constructors()
    test_map_steps()
    test_report()


def run_sweep_tests():
    """The brute-force sweep (varmap/brute_force.py). Separate from run_all() because these
    need a DedispersionPlan and a CUDA device, and take minutes rather than seconds."""

    test_sweep_vs_per_tfm(7, [1])
    test_sweep_vs_per_tfm(7, [2, 2, 1], num_early_triggers=1)
    test_sweep_phase_collapse(7)
    test_sweep_detrender_fp32(7)
    # The Detrender2d path has no analytic oracle, so test_sweep_column_norms is what covers
    # it -- and the polyphase sum, which is where it interacts with a time-downsampled tree.
    test_sweep_column_norms(6, [2, 1], detrender=False)
    test_sweep_column_norms(6, [2, 1], detrender=True)
    test_sweep_column_norms(6, [2, 1], num_primary_trees=2, nifreq=1)
    test_sweep_column_norms(6, [1], num_early_triggers=1, nifreq=1)
    # The streaming reduction against the dense one, which is the only check on the property
    # the scalable path assumes. Two subband layouts, because a ragged one (levels 1 and 0
    # mixed) is where the multiplet decomposition can go wrong.
    test_sweep_streaming_coarse(6, [2, 1])
    test_sweep_streaming_coarse(6, [1], num_early_triggers=1)
    test_sweep_streaming_coarse(6, [2, 1], detrender=True)
    # The GPU sweep against the CPU one. Both GPU kernels are validated against their
    # reference implementations by --sbdd and --pfsq, so this covers the python driver rather
    # than the kernels.
    test_sweep_gpu_vs_cpu(8, [2, 2, 1], nbeams=4)
    test_sweep_gpu_vs_cpu(8, [2, 2, 1], num_early_triggers=1, detrender=True, nfreq=200)
