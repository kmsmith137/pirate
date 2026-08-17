"""Unit tests for pirate_frb.varmap. Run with 'python -m pirate_frb test --varmap'.

Two of these are load-bearing beyond the usual sense, because they are the only checks on
properties the scalable path assumes and cannot verify at runtime:

  - test_geometry() compares the pure-python geometry derivation against a real
    DedispersionPlan. Everything else in varmap is built on those numbers, and they are a
    transcription of C++ that no other test would notice drifting.
  - test_coarse_grain() compares the blockwise reduction against a dense one on a map small
    enough to form. At production scale the dense map is never built, so this is the only
    place the reduction is checked against something obviously correct.

Several tests cross-check against pirate_frb.slow_avar (varmap_eval, VarMapDistance), which
varmap supersedes. Those comparisons are deliberately temporary: they are what licenses
deleting the old code, and they go away with it.
"""

import numpy as np

from .VarianceMap import VarianceMap, _tree_geometries, _subband_tables
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

    # m -> (subband, fine DM), derived independently of _subband_tables().
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

    g = _tree_geometries(config)[itree]
    nalpha = (1 << (g['tree_rank'] - g['pf_rank'])) * g['nmultiplets'] * g['nprofiles']
    A = rng.uniform(0.1, 2.0, size=(nalpha, g['nfreq'])).astype(dtype)
    if nzero > 0:
        A[rng.choice(nalpha, size=nzero, replace=False)] = 0.0
    return VarianceMap.from_dense(config, itree, A, y_true='row_sums', is_admissible=True)


####################################   tests   ####################################


def test_geometry(niter=30):
    """The pure-python geometry derivation against a real DedispersionPlan.

    varmap derives every per-tree number from the DedispersionConfig alone, so that a map can
    be constructed and scored on a machine with no GPU. That derivation is a transcription of
    the DedispersionPlan constructor and of the FrequencySubbands constructor; this is what
    keeps the transcription honest.
    """

    from ..pirate_pybind11 import DedispersionConfig, DedispersionPlan

    nchecked = 0
    for _ in range(niter):
        config = DedispersionConfig.make_random()
        plan = DedispersionPlan(config, False)   # cdd2_kernel_required=False: no registry
        geoms = _tree_geometries(config)

        assert len(geoms) == len(plan.trees), (len(geoms), len(plan.trees))

        for (g, t) in zip(geoms, plan.trees):
            fs = t.frequency_subbands
            assert g['gamma'] == t.primary_tree_index, (g, t.primary_tree_index)
            assert g['early_trigger_level'] == t.early_trigger_level
            assert g['tree_rank'] == t.total_rank()
            assert g['pf_rank'] == fs.pf_rank
            assert g['subband_counts'] == tuple(fs.subband_counts)
            assert g['nmultiplets'] == fs.M, (g['nmultiplets'], fs.M)
            assert g['nsubbands'] == fs.N
            assert g['nprofiles'] == t.nprofiles
            assert g['ndm_out'] == t.ndm_out
            assert g['nfreq'] == config.get_total_nfreq()

            # The multiplet ordering: an ORDERING CONVENTION that lives in C++ and could
            # drift. Every archived map would be silently reinterpreted if it did, which is
            # why m_to_n is also written to the map file and re-checked on read.
            assert np.array_equal(g['m_to_n'], np.asarray(fs.m_to_n))
            nchecked += 1

    atomic_print(f'    test_geometry(niter={niter}): pass ({nchecked} trees)')


def test_index_arithmetic(r=8, subband_counts=(2,2,1), num_early_triggers=1):
    """alpha_to_beta_block() and group_sizes(), against label arrays built the long way."""

    config = _make_test_config(r, list(subband_counts),
                               num_early_triggers=num_early_triggers)
    rng = np.random.default_rng(1)

    for itree in range(len(_tree_geometries(config))):
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


def test_subband_tables():
    """_subband_tables() against the C++ FrequencySubbands, over the legal count vectors."""

    from ..pirate_pybind11 import FrequencySubbands

    ncases = 0
    for pf_rank in range(0, 5):
        for counts in _enumerate_subband_counts(pf_rank):
            fs = FrequencySubbands(list(counts))
            m_to_n, n_level, n_to_mbase = _subband_tables(counts)
            assert np.array_equal(m_to_n, np.asarray(fs.m_to_n)), counts
            assert m_to_n.size == fs.M and n_level.size == fs.N, counts
            assert np.array_equal(n_to_mbase, np.asarray(fs.n_to_mbase)), counts
            ncases += 1

    atomic_print(f'    test_subband_tables(): pass ({ncases} count vectors)')


def _enumerate_subband_counts(pf_rank, max_count=3):
    """Legal subband_counts vectors of length pf_rank+1 (last element must be 1)."""

    if pf_rank == 0:
        yield (1,)
        return

    from ..pirate_pybind11 import FrequencySubbands

    def rec(level, acc):
        if level == pf_rank:
            counts = tuple(acc) + (1,)
            try:
                FrequencySubbands.validate_subband_counts(list(counts))
            except Exception:
                return
            yield counts
            return
        for c in range(0, max_count + 1):
            yield from rec(level + 1, acc + [c])

    yield from rec(0, [])


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

    for itree in range(len(_tree_geometries(config))):
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
    ntrees = len(_tree_geometries(config))
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


####################################   entry point   ####################################


def run_all(niter_geometry=30):
    """Everything, in dependency order: geometry first, since the rest is built on it."""

    test_subband_tables()
    test_geometry(niter_geometry)
    test_index_arithmetic()
    test_constructor_validation()
    test_coarse_grain()
    test_distance()
    test_admissibility()
    test_estimate_distance()
    test_check_ref_covers_y_true()
    test_multimap()
    test_dense_float32()
