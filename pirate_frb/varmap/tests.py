"""Unit tests for pirate_frb.varmap. Run with 'python -m pirate_frb test --varmap'.

Two of these are load-bearing beyond the usual sense, because they are the only checks on
properties the scalable path assumes and cannot verify at runtime:

  - test_coarse_grain() compares the blockwise reduction against a dense one on a map small
    enough to form. At production scale the dense map is never built, so this is the only
    place the reduction is checked against something obviously correct.
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
whether a factorization is any good, and they deliberately do not check that
``Q_is_semiorthogonal`` or ``pinned_columns`` are truthful -- VarianceMap carries those
rather than enforcing them, since only the steps that establish them can verify them.
"""

import contextlib

import numpy as np

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
