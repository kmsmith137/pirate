"""Unit tests for pirate_frb.varmap.

TWO HALVES OF ONE JOB, both run by 'python -m pirate_frb test --varmap':

  - IS THE VARIANCE-MAP CODE RIGHT? Everything checkable WITHOUT running a dedisperser: the
    VarianceMap class (indexing, coarse-graining, distance, factorization, file format), the
    covering-LP and basis machinery, and the STRUCTURE of the analytic map of
    detrender_free.py -- its per-group factorization, its coarse construction against
    coarse_grain(), and its internal consistency. This is run_once() and run_all().
  - IS THE ANALYTIC MAP TRUE? Push a one-hot through the REAL dedisperser once per input
    channel, measure the variance that comes out, and compare. This is the brute-force
    sweep group, and it is why the flag needs a DedispersionPlan and a GPU. It is the ONLY
    numerical oracle for the analytic map and for compute_detrender_free_varfine();
    test_multimap_vs_sweep() is where both comparisons live.

Those two fail independently and are debugged differently, so the distinction is worth
keeping in mind -- but a single flag is the right granularity for a command line. The sweep
group is not separately dispatchable: call run_once(), run_all(), or an individual test
directly from python while bisecting.

ONE ENTRY POINT, THREE CADENCES. run_tests(iteration) is what '--varmap' calls, and it owns
every cadence decision. The parts are named for WHEN they run, not for what they cover:

  - run_once() runs once per invocation, for two distinct reasons -- an exhausted parameter
    space (item 11), or too expensive to repeat. See its docstring; the reasons matter.
  - run_all() runs on EVERY '-n' iteration. About 12 s.
  - EVERY TENTH ITERATION: test_sweep_gpu_vs_cpu_random(), the only check on the GPU sweep
    driver and the most expensive test here, and test_restriction_vs_sweep(), which runs a
    CPU sweep and so buys a much smaller config per second.

Both halves above are spread across those three, deliberately -- "code or truth" is what a
test is FOR, and "how often" is when it runs, and grouping by the second is what a runner
needs. Each test's own docstring says which half it belongs to.

EVERY TEST OUTSIDE run_once() DRAWS ITS OWN GEOMETRY, from DedispersionConfig::make_random()
via _random_config(). Nothing here pins a config or an RNG seed, so a long run explores
rather than repeating, and the run's printed seed replays it.

Four of these are load-bearing beyond the usual sense, because they are the only checks on
something no other test and no runtime assertion can catch:

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
  - test_distance_oracles() checks D and max_r against oracles written out by hand, not
    against another varmap code path. Every other assertion about D is the class checking
    itself, so those would all move together if the DEFINITION of D changed; the hand-written
    oracles are what makes that change loud, which is what varmap/distance.py's "DO NOT
    CHANGE THE DEFINITION OF D SILENTLY" asks for.

Note there is deliberately NO test of the per-tree geometry itself. It comes verbatim from
the C++ DedispersionPlan constructor, so any such test would compare the C++ to itself. The
plan constructor and its yaml round-trip are tested in
pirate_frb/tests/test_decode_argmax.py.

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

TWO BARS THAT ANY NEW TEST HERE SHOULD INHERIT, both of which cost something to learn.

  - A UNIFORM SCALING IS WORTHLESS AS A CHECK ON THE INDEX CONVENTION. Scoring A * c gives
    f(c) on every row under ANY permutation of the rows, so a test built on one passes with
    the alpha -> beta map completely broken. A geometry spot-check has to be ROW-DEPENDENT.
  - COMPARE ELEMENTWISE, NOT IN AGGREGATE, and prefer an EXACT bar to a tolerance wherever one
    is available. Max is exact, so anything built on the coarse-graining reduction can be
    required to agree to the last bit; a norm or a distance would hide a permutation.

The test_lp_* tests check varmap.lp against things OTHER than itself wherever they can: the
Q-step's optimality against brute-force vertex enumeration, the free LP against the bounded
one, the majorization against the same sum accumulated the other way round, and a blocked
repair against an unblocked one over deliberately ragged tails. What they cannot check is
that the port computes the same numbers as the research code it came from -- that is a
one-time equivalence test living outside this repo, and it is what the defaults exist for.
"""

import contextlib
import dataclasses
import inspect
import itertools

import numpy as np

from .distance import YTRUE_FLOOR
from .VarianceMap import VarianceMap, coarse_grain_vector, make_plan
from .VarianceMultiMap import VarianceMultiMap
from ..utils import atomic_print


####################################   helpers   ####################################


def _tree(config, itree):
    """One tree of 'config'. Tests are the one place a throwaway tree copy is the right tool
    -- library code should hold the plan and index plan.trees."""
    return make_plan(config).trees[int(itree)]


def _itree(config, primary_tree_index, early_trigger_level=0):
    """The 'itree' of one tree of 'config', named by (primary_tree_index,
    early_trigger_level). Same throwaway-plan convention as _tree() above: library code
    should hold the plan and call DedispersionPlan.dedispersion_tree_index() on it."""
    return int(make_plan(config).dedispersion_tree_index(primary_tree_index,
                                                        early_trigger_level))


def _ntrees(config):
    """The number of dedispersion trees of 'config' (= plan.ntrees). See _tree() above."""
    return int(make_plan(config).ntrees)


def _restriction_pairs(config):
    """Every (parent, child) pair of 'config', with the child's multiplet map into the parent.

    Returns a list of (gamma, e, iparent, ichild, m_map), one entry per early-trigger tree,
    where m_map[mc] is the parent multiplet that child multiplet mc restricts to. The list is
    empty when no primary tree has an early trigger.

    THE ROW MAP IS REBUILT HERE from toplevel band ranges, not taken from a production index
    mapping, and that is what makes its two callers tests of Proposition 1 (see
    notes/variance_map.tex) rather than tests of the plumbing. A map that is not the
    contiguous prefix 0, 1, ... is where a wrong row map would show up at all, so a caller
    that cares reports how many of its pairs have one.
    """

    plan = make_plan(config)
    trees = plan.trees           # a fresh list of copies per access; take it once
    out = []

    for gamma in range(int(config.num_primary_trees)):
        iparent = int(plan.dedispersion_tree_index(gamma, 0))
        parent = trees[iparent]
        fsp = parent.frequency_subbands
        pband = {(parent.n_to_toplevel_flo(n), parent.n_to_toplevel_fhi(n)): n
                 for n in range(int(fsp.N))}

        for e in range(1, int(config.primary_trees[gamma].num_early_triggers) + 1):
            ichild = int(plan.dedispersion_tree_index(gamma, e))
            child = trees[ichild]
            fsc = child.frequency_subbands

            m_map = []
            for mc in range(int(fsc.M)):
                nc = int(fsc.m_to_n[mc])
                n_p = pband[(child.n_to_toplevel_flo(nc), child.n_to_toplevel_fhi(nc))]
                m_map.append(int(fsp.n_to_mbase[n_p]) + int(fsc.m_to_d[mc]))

            out.append((gamma, e, iparent, ichild, m_map))

    return out


def _rng(seed=None):
    """A fresh numpy Generator for one test, seeded from the master --seed.

    NEVER draw from numpy's zero-argument default_rng() in this file. It seeds itself from OS
    ENTROPY, which puts the draw outside __main__.seed_rngs(): a failing draw then cannot be
    replayed from the printed seed, which is the property this file's docstring claims. The
    C++ side is covered either way -- DedispersionConfig::make_random() draws through the
    seeded ksgpu::default_rng() -- so what an unseeded call loses is everything numpy draws,
    INCLUDING _random_config()'s gpu_valid coin flip.

    Seeded from numpy's global RandomState, which seed_rngs() pins, so successive calls still
    differ (a long run explores) while the whole sequence replays from one integer.

    An explicit 'seed' is passed straight through, for the helpers that pin one. Take the
    seed argument through here rather than calling default_rng(seed) directly: seed=None is
    the DEFAULT for those helpers, and default_rng(None) is the unseeded case again -- so a
    helper that forwards its seed straight to numpy silently leaves the master seed whenever
    its caller omits one.
    """

    if seed is not None:
        return np.random.default_rng(seed)
    return np.random.default_rng(np.random.randint(0, 1 << 32))


def _random_config(rng=None, **kwargs):
    """A random DedispersionConfig for the tests here: DedispersionConfig::make_random(), with
    this file's standard draw settings filled in.

    NOT A FILTER, and deliberately so. It applies no rejection sampling and enforces no
    structural property -- it forwards to make_random() and returns whatever comes back. If a
    test needs a structure the draw does not always supply (an early trigger, a
    non-contiguous multiplet map, more than one primary tree), the fix is in the TEST -- guard
    the assertion and report the count -- not here. 'pirate_frb dev coverage' is where those rates
    are tracked. Adding constraint keywords to this function would put the filtering back and
    is the thing to resist.

    Two defaults are worth their justification:

      max_toplevel_rank=7 is the cost knob, and it is the only one needed. Measured over 400
      draws per setting, the implied dense (nalpha, nfreq) map is p90 1.2 MiB / max 3.3 MiB at
      6, p90 13.9 / max 71.3 at 7, and p90 305 / max 1074 at 9. Rank 6 is cheaper but starves
      the properties the tests care about: early triggers drop from 30% to 13% and a
      non-contiguous multiplet map from 5.7% to 2%, because a small toplevel rank leaves no
      room for toplevel_tree_rank - net.

      gpu_valid is DRAWN, not fixed, because the two settings reach disjoint corners. R == 0
      occurs only at True; subband_counts[0] == 0 and R == 3 only at False; and False reaches
      122 distinct subband vectors against a handful, since the True path can only draw
      vectors the cdd2 registry stocks.
    """

    rng = _rng() if (rng is None) else rng
    from ..pirate_pybind11 import DedispersionConfig

    kwargs.setdefault('max_toplevel_rank', 7)
    kwargs.setdefault('max_early_triggers', 2)
    gv = kwargs.pop('gpu_valid', None)
    min_nalpha = kwargs.pop('min_nalpha', 32)

    # A SIZE FLOOR, NOT A STRUCTURE FILTER, and the distinction matters. make_random() draws
    # toplevel_tree_rank down to 2, which gives maps with one or two rows -- and a test cannot
    # say anything about a one-row map: an SVD has one mode, a 'group sizes differ' check has
    # one group, and 'all outputs have no variance' becomes the normal case rather than the
    # corner. So redraw until there is enough content to measure, exactly as item 4 bounds a
    # draw by a proxy for what it costs, run the other way.
    #
    # This does NOT filter on structure. Whether the draw has early triggers, more than one
    # primary tree, or a non-contiguous multiplet map is left entirely to chance, and the
    # tests report those rather than demanding them.
    for _ in range(200):
        config = DedispersionConfig.make_random(
            gpu_valid=(bool(rng.integers(2)) if (gv is None) else gv), **kwargs)
        if _nalpha_of(config, 0) >= min_nalpha:
            return config
    return config


def _make_test_config(toplevel_tree_rank, subband_counts, num_primary_trees=1,
                      num_early_triggers=0, max_width=4, nfreq=None):
    """A small DedispersionConfig. Every tree gets 2^R coarse DM channels per multiplet
    (DedispersionTree.dm_downsampling, not a config field), which is what the variance map's
    index convention assumes.
    """

    from ..pirate_pybind11 import DedispersionConfig, PrimaryTree

    nfreq = nfreq if (nfreq is not None) else (1 << toplevel_tree_rank)
    nt_in = max(1 << (toplevel_tree_rank - 1), 32 << (num_primary_trees - 1))
    min_tree_rank = toplevel_tree_rank - num_early_triggers - (1 if num_primary_trees > 1 else 0)

    config = DedispersionConfig()
    config.zone_nfreq = [nfreq]
    config.zone_freq_edges = [400.0, 800.0]
    config.time_sample_ms = 1.0
    config.dtype = np.float32
    config.toplevel_tree_rank = toplevel_tree_rank
    config.time_samples_per_chunk = nt_in
    config.frequency_subband_counts = subband_counts
    config.primary_trees = [
        PrimaryTree(num_early_triggers, max_width,
                    1 << min_tree_rank, nt_in >> ipri)
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


def _nalpha_of(config, itree):
    """nalpha for one tree, without building a map. Used where a caller has to size 'nzero'
    against the drawn geometry."""

    tree = _tree(config, itree)
    fs = tree.frequency_subbands
    return (1 << (int(tree.tree_rank) - int(fs.pf_rank))) * int(fs.M) * int(tree.nprofiles)


def expect_raise(fn, needle):
    """Call 'fn' and require a RuntimeError whose message mentions 'needle'.

    THE NEEDLE IS THE POINT. Five tests here check constructor and reader rejections, and what
    each one is pinning is that the RIGHT rejection fired -- a bare "it raised" would pass when
    an unrelated error happened first, which is exactly how a rejection test rots.
    """
    try:
        fn()
    except RuntimeError as e:
        assert needle in str(e), (needle, str(e))
        return
    raise AssertionError(f'expected a RuntimeError mentioning {needle!r}')


def _draw_K(rng, lo=1, hi=8):
    """A factorization rank for one test.

    A DRAW rather than a pinned rank, for the twelve tests that take one. The value worth
    reaching on purpose is K = 1: 'mid' is then 1x1 and the map is a rank-one outer product,
    where every loop over modes runs once and a transposed mode index has nothing to
    disagree with.

    The ceiling is well under the cell floor -- _draw_lp_cell_config() guarantees nbeta and
    nfreq are both >= 16 -- so a factorization always has room for K modes and NO
    'K = min(K, nbeta, nfreq)' cap is needed at the call sites. Pass 'lo' where a small K
    makes an assertion vacuous; each caller that does says which one.
    """
    return int(rng.integers(int(lo), int(hi) + 1))


def _random_map(config, itree, rng, *, nzero=None, dtype=np.float64, **kwargs):
    """A fine VarianceMap with a random nonnegative matrix, standing in for A_true.

    THE ENTRIES ARE LOG-UNIFORM OVER A DRAWN NUMBER OF DECADES, not uniform over one. A real
    variance-map row spans about 1e14 (LpConfig.clip_rel says so, and says what it costs), and
    everything that only bites on a wide dynamic range is invisible on a matrix whose entries
    all sit within a factor of 20 of each other: AdmissibilityResult.max_r is a PER-ELEMENT
    RELATIVE quantity with no floor, so it is set by the smallest element compared;
    check_ref_covers_y_true()'s margin is relative for the same reason; and the LP's row
    equilibration exists precisely because the solver's own tolerance is absolute. The span is
    drawn rather than pinned wide so that both regimes are sampled.

    'nzero' rows are set to zero, so that the YTRUE_FLOOR path (outputs with no variance) is
    exercised -- it is not an edge case in practice, since a W=0 Detrender2d annihilates the
    DM=0 output. AS A FRACTION, NOT A COUNT (notes/unit_tests.md point 6): a fixed count of 2
    or 3 puts YTRUE_FLOOR on 0.3% of rows at a median nalpha of ~900, and never reaches the
    regime where most outputs are dead. Pass an explicit integer where a test needs an exact
    number of dead rows. At least one row always survives, since several callers score the
    map and a fully dead map has nothing to score.
    """

    tree = _tree(config, itree)
    fs = tree.frequency_subbands
    nalpha = (1 << (tree.tree_rank - fs.pf_rank)) * fs.M * tree.nprofiles

    hi = 2.0
    lo = hi * 10.0 ** (-float(rng.uniform(0.5, 6.0)))
    A = np.exp(rng.uniform(np.log(lo), np.log(hi),
                           size=(nalpha, config.get_total_nfreq()))).astype(dtype)

    if nzero is None:
        # A third of draws have no dead rows at all (the common case in production), and the
        # rest draw a rate log-uniformly, so both "a handful" and "most of them" are reached.
        pzero = 0.0 if (rng.random() < 1/3) else float(np.exp(rng.uniform(np.log(1e-3),
                                                                         np.log(0.5))))
        nzero = int(rng.binomial(nalpha, pzero))
    nzero = min(int(nzero), max(nalpha - 1, 0))

    if nzero > 0:
        A[rng.choice(nalpha, size=nzero, replace=False)] = 0.0
    return VarianceMap.from_dense(config, itree, A, y_true='row_sums', is_admissible=True,
                                  **kwargs)


####################################   tests   ####################################


def test_index_arithmetic():
    """alpha_to_beta_block() and group_sizes(), against label arrays built the long way."""

    rng = _rng()
    config = _random_config(rng)

    for itree in range(_ntrees(config)):
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

            # coarse_grain_vector() is LABEL-FREE -- a reshape and two reductions, exploiting
            # the two fixed shapes -- so 'beta' above is a genuinely independent oracle for
            # it, derived from each row's full-resolution DM rather than from any arithmetic
            # it shares. Requiring EXACT equality is legitimate here and elsewhere in this
            # file: max is exact, so the reduction order cannot matter.
            y = rng.uniform(0.5, 1.5, size=m.nalpha)
            want = np.full(nbeta, -np.inf)
            np.maximum.at(want, beta, y)
            assert np.array_equal(coarse_grain_vector(m.tree, y, L), want), (itree, L)

        # A fine map's default is the identity, not "coarse at L = R".
        assert np.array_equal(m.alpha_to_beta_block(0, m.nalpha), np.arange(m.nalpha))

    atomic_print(f'    test_index_arithmetic(r={int(config.toplevel_tree_rank)},'
                 f' subbands={[int(x) for x in config.frequency_subband_counts]}): pass'
                 ' (including coarse_grain_vector against the same label oracle)')


def test_constructor_validation():
    """The constructor's shape and flag checks, and immutability."""

    rng = _rng()
    config = _random_config(rng)
    m = _random_map(config, 0, rng)
    R = m.pf_rank

    # The granularity checks below need a coarse map whose row count actually DIFFERS from
    # the fine one, and L = R does not always give one: coarse-at-L is 2^(r-L) * N * P against
    # the fine 2^(r-R) * M * P, so the two coincide exactly when N == M -- i.e. when
    # pf_rank == 0, which is 30% of draws, and where a fine matrix genuinely IS a valid coarse
    # one. Going up one level always separates them, since each level halves the coarse count.
    Lc = R if (m.nmultiplets != m.nsubbands) else R + 1
    assert Lc <= m.tree_rank, (R, m.tree_rank, m.nmultiplets, m.nsubbands)

    # L out of range, and L set on an array that is not the coarse shape.
    Abar = m.coarse_grain(Lc).A
    assert Abar.shape != m.A.shape, (Lc, Abar.shape)
    expect_raise(lambda: VarianceMap.from_dense(config, 0, Abar, L=R-1), 'out of range')
    expect_raise(lambda: VarianceMap.from_dense(config, 0, Abar, L=m.tree_rank+1),
                 'out of range')
    expect_raise(lambda: VarianceMap.from_dense(config, 0, m.A, L=Lc), 'expected')
    expect_raise(lambda: VarianceMap.from_dense(config, 0, Abar), 'expected')
    expect_raise(lambda: VarianceMap.from_dense(config, 0, m.A, y_true=np.zeros(3)),
                 'FINE granularity')
    expect_raise(lambda: VarianceMap.from_dense(config, 0, Abar, L=Lc, y_true='row_sums'),
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

    # A float32 stored matrix: rows() promotes to float64, and nbytes() halves.
    m32 = m.replace(A=np.asarray(m.A, dtype=np.float32))
    assert m32.A.dtype == np.float32 and m32.rows(0, 4).dtype == np.float64
    assert m32.nbytes() * 2 == m.replace(A=np.asarray(m.A, dtype=np.float64)).nbytes()

    # is_coarse_grained and L are one fact; neither is derivable from nbeta.
    assert (m.L is None) and (not m.is_coarse_grained) and (m.nbeta == m.nalpha)
    c = m.coarse_grain(R)
    assert c.is_coarse_grained and (c.L == R)
    if m.nmultiplets == m.nsubbands:
        assert c.nbeta == m.nalpha, 'M == N, so coarse-at-R and fine have the same nbeta'

    atomic_print(f'    test_constructor_validation(nalpha={m.nalpha}): pass')


def test_coarse_grain():
    """coarse_grain() against a dense reduction, and coarse-to-coarser against fine-to-coarse.

    This is the one property the scalable path trusts and cannot check at runtime: at
    production scale the dense map is never formed. The dense reduction below is a genuine
    external oracle: its group labels are built independently by _obvious_beta().
    """

    rng = _rng()
    config = _random_config(rng)

    # ONE DRAWN TREE PER CALL, not every tree of the config. The property is per (tree, L),
    # and what distinguishes the trees of one config -- rank and subband table -- is already
    # what _random_config() varies from call to call. Looping them multiplied a full dense
    # reduction, which is O(nalpha * nfreq) and reaches ~100 MB, by ntrees for cases a long
    # run reaches anyway.
    itree = int(rng.integers(_ntrees(config)))

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

        # y_true is the TRUE row sums at FINE granularity, carried unchanged.
        assert np.array_equal(c.y_true, m.y_true)
        assert c.is_admissible == m.is_admissible
        assert c.check_ref_covers_y_true() >= 1.0

        # Coarsening is nested, so coarse-to-coarser must agree with fine-to-coarse. This
        # is what lets a sweep run once at the finest L that fits and coarsen down.
        # ONE DRAWN L2 rather than all of them: the statement is per (L, L2) pair and
        # costs two more full reductions, so enumerating made this loop quadratic in
        # (r - R) for pairs that successive calls sample anyway.
        if L < m.tree_rank:
            L2 = int(rng.integers(L+1, m.tree_rank + 1))
            assert np.array_equal(c.coarse_grain(L2).A, m.coarse_grain(L2).A), (L, L2)

        # ... and cannot go the other way.
        try:
            c.coarse_grain(L)
            raise AssertionError('coarse_grain() accepted L <= self.L')
        except RuntimeError as e:
            assert 'already coarse' in str(e)

        # A MEAN where a max was intended is the bug class check_ref_covers_y_true()
        # catches outright (the positive assertion is a few lines up).
        # ... and only where the mean is actually BELOW the max, which is the property
        # rather than a proxy for it. A multi-member group is not enough: '_random_map'
        # zeroes 'nzero' rows, and a group whose members are ALL zeroed has mean == max,
        # so the reader is right to accept it. Measured, seed 3126070166 draws exactly
        # that -- nalpha=3 with 2 zeroed rows, one 2-member group holding both of them,
        # and max - mean identically 0 over all 24 entries. The old guard (max group size
        # > 1) was true there and the assertion below fired on an honest map.
        #
        # The two guards agree on ordinary draws: over 328 (config, tree) cases at L=R
        # both were true 241 times, and the check REJECTED all 241.
        mean = None
        if L == R:
            mean = np.zeros_like(np.asarray(c.A))
            np.add.at(mean, labels, np.asarray(m.A, dtype=np.float64))
            mean /= c.group_sizes()[:,None]
        if (mean is not None) and np.any(mean < np.asarray(c.A)):
            try:
                c.replace(A=mean, history_record=dict(step='test')).check_ref_covers_y_true()
                raise AssertionError('check_ref_covers_y_true() accepted a mean-reduced map')
            except RuntimeError as e:
                assert 'max-envelope cannot do this' in str(e), str(e)

        # lift() is the inverse of the row duplication, not of the max.
        lifted = c.lift()
        assert (not lifted.is_coarse_grained) and lifted.shape == (m.nalpha, m.nfreq)
        assert np.array_equal(lifted.A, c.A[labels])

    atomic_print(f'    test_coarse_grain(itree={itree} of {_ntrees(config)},'
                 f' nalpha={_nalpha_of(config, itree)}, L={R}..{m.tree_rank}): pass')


def test_distance():
    """get_distance() and its coarse counterpart, checked against themselves.

    Everything here is internal consistency: the row breakdown against the overall mean,
    coarse scoring against lift scoring, block size invariance at a ragged tail as well as a
    dividing one, and the two refusals. None of it would notice if the DEFINITION of D
    changed, because every assertion would move together. Pinning the VALUE of D is
    test_distance_oracles()'s job, and that is the only thing in this file that does it.
    """

    rng = _rng()
    config = _random_config(rng)

    # An EXACT count, because the assertions below count nan rows; drawn rather than
    # written out, and capped so something stays scorable.
    for nzero in (0, int(rng.integers(1, max(2, min(9, _nalpha_of(config, 0)))))):
        true = _random_map(config, 0, rng, nzero=nzero)
        A_true = np.asarray(true.A, dtype=np.float64)

        assert true.nscored == true.nalpha - nzero

        # --- fine case: an admissible dense approximation ---
        A_approx = A_true * rng.uniform(1.0, 1.6, size=A_true.shape)
        approx = VarianceMap.from_dense(config, 0, A_approx, y_true=true.y_true,
                                        is_admissible=True)
        D = approx.get_distance()

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

    atomic_print(f'    test_distance(nalpha={_nalpha_of(config, 0)}): pass')


def test_admissibility():
    """measure_admissibility(): the coarse/fine theorem, the sign conventions, and the
    inflation path.

    Everything below shares ONE cell -- a fine map, its coarse-graining 'ref', and a uniform
    1.5x inflation of that -- because every property here is a statement about the same three
    objects, and building the cell twice only makes the two halves drift apart.
    """

    rng = _rng()
    config = _random_config(rng)

    true = _random_map(config, 0, rng)
    L = true.pf_rank + 1
    ref = true.coarse_grain(L)

    # A coarse-assigned approximation. Measuring it against the COARSE ref and measuring its
    # lift against the FINE true map must agree -- that is the pivot identity, and it is what
    # makes scoring possible at CHORD scale, where the fine map does not exist.
    #
    # THE MAX_R HALF OF THE IDENTITY IS NOT CHECKED HERE. A uniform 1.5x inflation gives every
    # element the same ratio, so 'rc.max_r == rf.max_r' would hold under any permutation of
    # the rows and says nothing about the index convention. The planted-violation version at
    # the end of this function is the one that does.
    capprox = ref.inflated(1.5)
    rc = capprox.measure_admissibility(ref)
    rf = capprox.lift().measure_admissibility(true)
    assert rc.admissible and rf.admissible
    assert rc.nviol == 0 and rf.nviol == 0

    # max_diff does NOT transfer through the lift, and that is worth pinning rather than
    # assuming. The pivot identity is about RATIOS: max(true/Abar) over a group is 1, attained
    # at the group's argmax member, so scaling by 1.5 gives max_r = 1/1.5 on both sides. The
    # difference has no such identity -- against the FINE map, a member whose true value is far
    # below its group's max contributes |true - 1.5*Abar| ~ 1.5*Abar, not 0.5*Abar. So the
    # coarse measurement reports the closed-form 0.5 while the lifted one reports ~1.5.
    # Read max_diff against the reference it was measured with.
    assert abs(rc.max_diff - 0.5) < 1.0e-12, rc.max_diff
    assert rf.max_diff > rc.max_diff + 0.5, (rc.max_diff, rf.max_diff)
    assert rc.vmap.is_admissible and (rc.vmap.A is capprox.A)

    # THE PLANTED DEFECTS BELOW MUST LAND WHERE ref IS POSITIVE, so draw from those entries
    # rather than uniformly. _random_map() zeroes rows and coarse_grain() carries whole zero
    # rows into 'ref', so a uniform (row, col) draw lands on one every so often -- and there
    # neither defect is a defect: planting 0.9*ref plants 0.0 into a place the map is already
    # 0.0 (rb.nviol would be 0), and planting -1.0 is not "non-positive where ref is
    # POSITIVE", so max_r stays finite. A uniform draw really does hit this: seed 1341684269
    # gives max_r 0.667 and argmax_r (0,2) against a planted (18,17).
    pos = np.flatnonzero(np.asarray(ref.A) > 0.0)
    assert pos.size >= 2, 'the drawn reference has no positive entries to plant a defect in'
    shape = np.asarray(ref.A).shape

    # A single planted underestimate: infinite D, but a small inflation fixes it, and that is
    # the number that distinguishes "nearly usable" from "hopeless".
    # Drawn, not literal: on a drawn geometry a hardcoded (row, col) is out of bounds.
    ib, jb = (int(x) for x in np.unravel_index(int(rng.choice(pos)), shape))
    bad = np.array(capprox.A)
    bad[ib, jb] = ref.A[ib, jb] * 0.9
    bad = capprox.replace(A=bad, is_admissible=False, history_record=dict(step='test'))
    rb = bad.measure_admissibility(ref, inflate=True)
    assert (not rb.admissible) and (rb.max_r > 1.0) and np.isfinite(rb.max_r)
    assert rb.argmax_r == (ib, jb), rb.argmax_r
    assert rb.nviol == 1 and rb.viol_rows == 1 and (ib in rb.worst_rows)
    assert np.isfinite(rb.D_inflated) and (rb.D_inflated >= capprox.get_distance())
    assert not rb.vmap.is_admissible

    # SIGNS. A non-positive entry where ref is positive is an underestimate NO rescaling
    # repairs, and is reported as max_r = inf rather than raised on: scoring a signed
    # candidate is the main reason this method exists.
    i_n, j_n = (int(x) for x in np.unravel_index(int(rng.choice(pos)), shape))
    neg = np.array(capprox.A)
    neg[i_n, j_n] = -1.0
    neg = capprox.replace(A=neg, is_admissible=False, history_record=dict(step='test'))
    rn = neg.measure_admissibility(ref)
    assert np.isinf(rn.max_r) and (rn.argmax_r == (i_n, j_n)) and (rn.nneg_self == 1)

    # ... and max_diff stays FINITE there, which is the whole reason the two coexist: it is an
    # accuracy figure and says nothing about repairability. A caller reading max_diff alone
    # would call this map good.
    assert np.isfinite(rn.max_diff) and (rn.max_diff > 0.0), rn.max_diff

    # ref <= 0 maps to ratio 0, so such an element can never become the argmax (0/0 included).
    i_z = int(rng.integers(ref.nbeta))
    zref = np.array(ref.A)
    zref[i_z, :] = 0.0
    zref = ref.replace(A=zref, history_record=dict(step='test'))
    zapp = capprox.replace(A=np.where(np.arange(capprox.nbeta)[:,None] == i_z, 0.0,
                                      np.asarray(capprox.A)),
                           history_record=dict(step='test'))
    # 'the zeroed row cannot BE the argmax' needs another row to be the argmax instead, so
    # it is vacuous at nbeta == 1.
    rz = zapp.measure_admissibility(zref)
    assert rz.admissible
    assert (ref.nbeta == 1) or (rz.argmax_r[0] != i_z), (rz.argmax_r, i_z)

    # An exact map: max_r is 1 (no inflation needed), max_diff is 0 (no error). The two
    # sentinels differ, which is worth pinning -- they are different quantities.
    same = ref.measure_admissibility(ref)
    assert (same.max_r == 1.0) and (same.max_diff == 0.0), (same.max_r, same.max_diff)

    # Block size must not change the answer, including at a ragged tail.
    for nb in (1, 3, 7, ref.nbeta):
        r2 = capprox.measure_admissibility(ref, block_rows=nb)
        assert (r2.max_r, r2.argmax_r, r2.nviol) == (rc.max_r, rc.argmax_r, rc.nviol), nb
        assert r2.max_diff == rc.max_diff, (nb, r2.max_diff, rc.max_diff)

    # Mixing granularities is not a valid test, and says so.
    try:
        capprox.measure_admissibility(true)
        raise AssertionError('measure_admissibility() compared a coarse self with a fine ref')
    except RuntimeError as e:
        assert 'coarse-graining' in str(e) or 'shape mismatch' in str(e)

    # ---- THE INFLATION PATH ----
    #
    # measure_admissibility(inflate=True) rescales by max_r*(1+1e-12) and then ASSERTS the
    # result is admissible rather than re-measuring it. That is a deliberate shortcut -- the
    # rescale is exact by construction -- but it makes 'isfinite(D_inflated)' vacuous as a
    # check, so the interesting assertion is the one made below: feed the inflated map BACK
    # through the elementwise scan and require it really does dominate. The 1e-12 fudge exists
    # precisely because scaling by exactly max_r lands on the boundary in floating point, and
    # nothing else in this file would notice if it were deleted.
    #
    # Every planted defect below is drawn from 'pos', for the reason given where it is built.
    D0 = capprox.get_distance()

    # Many random planted underestimates rather than one hand-placed element: the fudge
    # factor's job is at the rounding boundary, and one fixed case samples it once.
    for trial in range(10):
        i, j = (int(x) for x in np.unravel_index(int(rng.choice(pos)), shape))
        A = np.array(capprox.A)
        A[i, j] = ref.A[i, j] * float(rng.uniform(0.9, 0.999))
        planted = capprox.replace(A=A, is_admissible=False, history_record=dict(step='test'))

        res = planted.measure_admissibility(ref, inflate=True)
        assert not res.admissible
        # The factor IS max_r, up to the fudge. Nothing else pins these two together.
        assert abs(res.inflation / res.max_r - 1.0) < 1.0e-9, (res.inflation, res.max_r)

        # THE CHECK THAT MATTERS: re-measure, do not trust the flag.
        again = planted.inflated(res.inflation).measure_admissibility(ref)
        assert again.admissible, (trial, res.max_r, again.max_r)
        assert again.max_r <= 1.0, (trial, again.max_r)

        # D_inflated brackets the PLANTED map's own D from above, and approaches it as
        # max_r -> 1. The upper bound is what makes max_r usable as a triage number: max_r =
        # 1.02 means "nearly usable" only if D cannot have moved much.
        #
        # Against 'planted', not against capprox's own D0. The planted underestimate moves D
        # by itself, and on a small drawn map one row is a big share of a mean over rows -- so
        # bracketing against D0 confounds the planting with the inflation and fails by a few
        # times 1e-3.
        #
        # The upper bound is RELATIVE to D_planted, not an absolute offset. Inflation scales
        # the whole map by 'inflation', and D is a mean of per-row ratios, so the gap it opens
        # is proportional to D itself; an absolute "+2*(max_r-1)" only held while the drawn
        # entries spanned one decade and D stayed near 1. The factor of 2 is the slack: the
        # exact statement would be D_inflated = inflation * D_planted, and the point being
        # made is that a max_r near 1 bounds the movement, not that it predicts it.
        D_planted = planted.replace(is_admissible=True,
                                    history_record=dict(step='test')).get_distance()
        assert res.D_inflated >= D_planted - 1.0e-12, (res.D_inflated, D_planted)
        assert res.D_inflated <= D_planted * (1.0 + 2.0*(res.max_r - 1.0)) + 1.0e-12, \
            (res.D_inflated, D_planted, res.max_r)

    # An ALREADY-admissible map is not touched: the factor is exactly 1 and D is unchanged.
    ok = capprox.measure_admissibility(ref, inflate=True)
    assert ok.admissible and (ok.inflation == 1.0), ok.inflation
    assert ok.D_inflated == D0, (ok.D_inflated, D0)

    # An underestimate that NO rescaling repairs (a zero where ref is positive) reports an
    # infinite factor, and D_inflated follows it rather than being computed from inf*A.
    i_0, j_0 = (int(x) for x in np.unravel_index(int(rng.choice(pos)), shape))
    z = np.array(capprox.A)
    z[i_0, j_0] = 0.0
    zm = capprox.replace(A=z, is_admissible=False, history_record=dict(step='test'))
    r_inf = zm.measure_admissibility(ref, inflate=True)
    assert np.isinf(r_inf.max_r) and np.isinf(r_inf.inflation) and np.isinf(r_inf.D_inflated)

    # THE COARSE/FINE INDEX CORRESPONDENCE, with a planted violation rather than the uniform
    # scale the pivot-identity block above uses: a uniform ratio is invariant under any
    # permutation of the rows, and this is what pins the convention down.
    i_c, j_c = (int(x) for x in np.unravel_index(int(rng.choice(pos)), shape))
    A = np.array(capprox.A)
    A[i_c, j_c] = ref.A[i_c, j_c] * 0.8
    planted = capprox.replace(A=A, is_admissible=False, history_record=dict(step='test'))
    rcp = planted.measure_admissibility(ref, inflate=True)
    rfp = planted.lift().measure_admissibility(true, inflate=True)
    labels = _obvious_beta(true, L)
    assert rcp.argmax_r[0] == labels[rfp.argmax_r[0]], (rcp.argmax_r, rfp.argmax_r)
    assert rcp.argmax_r[1] == rfp.argmax_r[1], (rcp.argmax_r, rfp.argmax_r)
    assert rcp.max_r == rfp.max_r, (rcp.max_r, rfp.max_r)
    assert abs(rcp.D_inflated - rfp.D_inflated) <= 1.0e-9 * abs(rfp.D_inflated)

    atomic_print(f'    test_admissibility(nalpha={_nalpha_of(config, 0)}): pass')


def test_distance_oracles():
    """D and max_r against ORACLES WRITTEN OUT HERE, not against another varmap code path.

    Everywhere else in this file D is checked for self-consistency: get_distance() against
    get_row_distances(), the coarse map against its lift, one block size against another.
    Those share row_sums() and distance.f, so none of them would notice if f itself, or the
    ratio it is handed, were wrong. Nothing else in this file would either: this test is the
    only check on the VALUE of D.

    So this test writes the oracle out by hand. Each case is chosen because its answer is
    known in closed form, or because it pins an index convention that a uniform scaling
    cannot (see "A UNIFORM SCALING IS WORTHLESS" above).
    """

    rng = _rng()
    config = _random_config(rng)
    true = _random_map(config, 0, rng)
    A = np.asarray(true.A, dtype=np.float64)

    # --- 1. A UNIFORM overestimate by c has D = f(c) in closed form. Every row's ratio is
    # exactly c, so this is the only case where D can be written down without summing
    # anything, and it is the one anchor on the VALUE of f.
    for c in (1.0, 1.25, 2.0, 7.5):
        approx = true.replace(A=A*c, is_admissible=True, history_record=dict(step='test'))
        D = approx.get_distance()
        want = (c - 1.0) / (1.0 + c/10.0)
        assert abs(D - want) <= 1.0e-12 * max(1.0, abs(want)), (c, D, want)

    # ... and c = 1 is the exact-representation case: D is exactly zero, not nearly.
    exact = true.replace(A=A.copy(), is_admissible=True, history_record=dict(step='test'))
    assert exact.get_distance() == 0.0, exact.get_distance()
    assert exact.measure_admissibility(true).max_r == 1.0

    # --- 2. max_r and argmax_r against a dense elementwise reference, with a ROW-DEPENDENT
    # perturbation. A uniform scale factor would pass this with the row index convention
    # broken, so every row gets its own.
    scale = rng.uniform(0.7, 1.9, size=(true.nalpha, 1))
    cand = true.replace(A=A*scale, is_admissible=False, history_record=dict(step='test'))
    res = cand.measure_admissibility(true)

    with np.errstate(divide='ignore', invalid='ignore'):
        ratio = np.where(A > 0.0, A / np.where(A*scale > 0.0, A*scale, 1.0), 0.0)
    want_max = float(ratio.max())
    i = int(np.argmax(ratio.max(axis=1)))
    assert abs(res.max_r - want_max) <= 1.0e-12 * want_max, (res.max_r, want_max)
    assert res.argmax_r[0] == i, (res.argmax_r, i)
    assert res.admissible == bool(np.all(A*scale >= A))

    # max_diff against the same dense reference. Written out rather than derived from max_r,
    # because the point of having both is that neither determines the other: this candidate
    # is scaled per row, so the ratio's worst element and the difference's worst element are
    # different elements whenever the row scales and the row magnitudes disagree.
    want_diff = float(np.abs(A - A*scale).max() / np.abs(A).max())
    assert abs(res.max_diff - want_diff) <= 1.0e-12 * want_diff, (res.max_diff, want_diff)

    # A uniform overestimate by c: max_r is exactly 1/c and max_diff is exactly (c-1), since
    # the difference is largest where A is. Both in closed form, which is the anchor on the
    # VALUE of each.
    for c in (1.25, 2.0):
        up = true.replace(A=A*c, is_admissible=True, history_record=dict(step='test'))
        ru = up.measure_admissibility(true)
        assert abs(ru.max_r - 1.0/c) <= 1.0e-12, (c, ru.max_r)
        assert abs(ru.max_diff - (c - 1.0)) <= 1.0e-12, (c, ru.max_diff)

    # --- 3. get_row_distances() must point at the row that PAYS. Its mean is checked
    # elsewhere, and a mean is permutation-invariant: a per-row array with the right values
    # attached to the wrong rows would pass that and fail this.
    # Planted on a SCORED row. A row whose y_true is below YTRUE_FLOOR is skipped -- its row
    # distance is nan -- so planting on one would leave nanargmax pointing at whichever row
    # happens to be worst among the rest, and the assertion would be about nothing.
    scored = np.flatnonzero(np.asarray(true.y_true, dtype=np.float64) >= YTRUE_FLOOR)
    assert scored.size > 0
    ip = int(scored[rng.integers(scored.size)])
    planted = A * 1.05
    planted[ip] = A[ip] * 4.0
    pm = true.replace(A=planted, is_admissible=True, history_record=dict(step='test'))
    rd = pm.get_row_distances()
    assert int(np.nanargmax(rd)) == ip, int(np.nanargmax(rd))
    assert abs(rd[ip] - (4.0 - 1.0)/(1.0 + 4.0/10.0)) <= 1.0e-12

    # --- 4. A map whose outputs ALL have zero variance cannot be scored, and says so rather
    # than returning 0 or nan. This is the tripwire for a broken sweep or config.
    dead = np.full((true.nalpha, true.nfreq), 1.0e-16)
    dm = VarianceMap.from_dense(config, 0, dead, y_true='row_sums', is_admissible=True)
    assert dm.nscored == 0
    try:
        dm.get_distance()
        raise AssertionError('get_distance() scored a map with no scorable output')
    except RuntimeError as e:
        assert 'could be scored' in str(e), str(e)

    atomic_print(f'    test_distance_oracles(nalpha={_nalpha_of(config, 0)}): pass')


def test_estimate_distance():
    """estimate_distance(): exact at frac=1, and unbiased (not group-weighted) below it."""

    rng = _rng()
    config = _random_config(rng)

    # nzero must leave something scorable: get_distance() is a mean over scored rows, and a
    # map with none returns nan (it raises, or the mean is nan) -- which is
    # test_distance_oracles()'s case, not this one. On a drawn geometry nalpha can be small
    # enough that a fixed nzero=2 takes most of it.
    true = _random_map(config, 0, rng, nzero=min(2, max(0, _nalpha_of(config, 0) - 2)))
    L = true.pf_rank + 1
    approx = true.coarse_grain(L).inflated(1.4)
    D = approx.get_distance()
    assert np.isfinite(D), (D, true.nscored, true.nalpha)

    full = approx.estimate_distance(frac=1.0)
    assert full.nsampled == approx.nbeta and full.frac_sampled == 1.0
    # NOTE a wart, not a test bug: at nbeta == 1 the sample variance divides by (n-1) = 0, so
    # stderr comes back nan even though frac=1.0 makes the estimate EXACT and 0.0 is the
    # right answer. Harmless (nobody scores a one-group map) and left alone here because this
    # plan does not touch production code, but worth fixing in varmap/VarianceMap.py.
    assert (full.stderr == 0.0) or (approx.nbeta == 1), (full.stderr, approx.nbeta)
    assert abs(full.D - D) <= 1.0e-12 * abs(D), (full.D, D)
    assert full.nscored == true.nscored

    # Groups are NOT all the same size (a subband at level l contributes 2^(L-R) * 2^l), and D
    # is a mean over FINE rows -- so a plain mean over sampled groups would be biased. That
    # only has teeth when the sizes actually differ, which needs N > 1 -- true in ~63% of
    # draws. GUARDED AND REPORTED rather than asserted: demanding it of a single drawn config
    # is the tripwire pattern notes/unit_tests.md item 8 sends to 'pirate_frb dev coverage'.
    sizes = approx.group_sizes()
    weighted = bool(sizes.min() != sizes.max())

    # A subsample lands within a few standard errors -- AS AN AGGREGATE, not per draw.
    #
    # Counting draws outside 4 sigma is not a property of this estimator and cannot be made
    # one by taking frac larger. 'stderr' is a normal-approximation interval on a mean over
    # sampled groups, and the per-group values are heavy-tailed: a subsample that misses an
    # outlier group underestimates BOTH the mean and the spread, so its z is large. Measured
    # on a config where a per-draw bar does fail (seed 2912858340, nbeta=64, frac=0.47, and
    # with every group the same size, so weighting is not the cause): over 400 subsamples, 19.5%
    # land outside 4 sigma and the worst is 7.5 -- while the true spread of (e.D - D) is
    # 0.004415 against a mean reported stderr of 0.003988, i.e. the SCALE is right to 11%.
    #
    # So check the scale and the typical draw, both of which are stable across configs.
    # Measured over 129 drawn configs: the spread/stderr ratio spans [0.72, 1.33] (median
    # 0.97) and median |z| spans [0.33, 1.42] (median 0.68). The bars below sit a factor of
    # ~2 outside both. They keep the teeth that matter: a stderr that ignored the group-size
    # weighting, or was off by any constant factor, moves the ratio directly.
    frac = min(1.0, max(0.2, 30.0 / max(approx.nbeta, 1)))
    devs, ses = [], []
    for _ in range(40):
        e = approx.estimate_distance(frac=frac, rng=rng)
        devs.append(e.D - D)
        ses.append(e.stderr)
    devs, ses = np.array(devs), np.array(ses)

    # frac == 1.0 (a map with fewer than 30 groups) samples everything, so the estimate is
    # exact and stderr is 0; there is nothing to score.
    ratio = med_z = None
    if float(ses.mean()) > 0.0:
        ratio = float(devs.std(ddof=1) / ses.mean())
        med_z = float(np.median(np.abs(devs / np.maximum(ses, 1.0e-300))))
        assert 0.3 < ratio < 3.0, (ratio, approx.nbeta, frac)
        assert med_z < 2.5, (med_z, approx.nbeta, frac)

    # Passing 'groups' back is what makes a PAIRED comparison possible: two arms on the same
    # subset have a far better determined ratio than either value.
    e1 = approx.estimate_distance(frac=0.1, rng=rng)
    e2 = approx.inflated(1.01).replace(is_admissible=True).estimate_distance(groups=e1.groups)
    assert np.array_equal(e1.groups, e2.groups) and (e2.D > e1.D)

    scored = ('exact (frac=1)' if (ratio is None)
              else f'spread/stderr {ratio:.2f}, median |z| {med_z:.2f}')
    atomic_print(f'    test_estimate_distance(nbeta={approx.nbeta}): pass, {scored}'
                 f' (group-size weighting {"exercised" if weighted else "NOT exercised:"
                 " every group the same size"})')


def test_multimap():
    """VarianceMultiMap: one map per PRIMARY tree, sharing one config object, with apply()
    covering every tree."""

    rng = _rng()
    config = _random_config(rng)
    ntrees = _ntrees(config)
    npri = int(config.num_primary_trees)
    nets = [int(pt.num_early_triggers) for pt in config.primary_trees]
    # COMPUTED from the config, not from arguments: make_random() gives each primary tree its
    # own num_early_triggers, so 'npri * (net + 1)' is wrong whenever they differ (~8% of
    # draws).
    assert ntrees == sum(n + 1 for n in nets), (ntrees, nets)

    # itree is NOT gamma: early_trigger_level descends within a family, so the (gamma, 0)
    # tree is the LAST of its block. That is only VISIBLE when some family has an early
    # trigger; with none, itree and gamma coincide and there is nothing to distinguish.
    iparents = [_itree(config, g) for g in range(npri)]
    assert (max(nets) == 0) or (iparents != list(range(npri)))

    maps = [_random_map(config, i, rng) for i in iparents]
    vmm = VarianceMultiMap(config, maps, provenance=dict(algorithm='test'))

    assert (vmm.num_primary_trees == npri) and (vmm.ntrees == ntrees)
    assert vmm.primary_map(0) is maps[0]
    assert [m.itree for m in vmm.maps] == iparents

    # NO SEQUENCE PROTOCOL, deliberately: len/[]/iter are ambiguous between the ntrees trees
    # and the num_primary_trees maps, and a call site that guessed wrong would be silently
    # off by an index convention. Refusing them makes every call site say which it means.
    for op in (lambda: len(vmm), lambda: vmm[0], lambda: list(iter(vmm))):
        try:
            op()
            raise AssertionError('VarianceMultiMap still supports the sequence protocol')
        except TypeError:
            pass

    # The legal range R <= L <= r differs per primary tree, so L may be a sequence.
    Ls = [m.pf_rank for m in vmm.maps]
    cvmm = vmm.coarse_grain(Ls)
    for (g, m) in enumerate(cvmm.maps):
        assert m.L == Ls[g] and np.array_equal(m.A, maps[g].coarse_grain(Ls[g]).A)

    v = rng.uniform(0.5, 1.5, size=config.get_total_nfreq())
    ys = vmm.apply_fine(v)
    assert len(ys) == ntrees

    # The parent entries are its own apply(), in (D, M, P) form.
    for (g, ip) in enumerate(iparents):
        want = np.asarray(maps[g].A, dtype=np.float64) @ v
        assert np.allclose(ys[ip].reshape(-1), want)

    # Every tree got an entry, with its own multiplet count.
    for itree in range(ntrees):
        tree = _tree(config, itree)
        fs = tree.frequency_subbands
        D = 1 << (tree.tree_rank - fs.pf_rank)
        assert ys[itree].shape == (D, int(fs.M), int(tree.nprofiles))

    res = cvmm.measure_admissibility(cvmm)
    assert len(res) == npri
    assert all(x.admissible for x in res) and (max(x.max_r for x in res) <= 1.0)

    # A short list is a bug in whatever assembled it, not a subset. At npri == 1 the short
    # list is EMPTY, which the constructor rejects by a different (also correct) message.
    try:
        VarianceMultiMap(config, maps[:-1])
        raise AssertionError('VarianceMultiMap accepted a partial list')
    except RuntimeError as e:
        assert ('one map per PRIMARY tree' in str(e)) or ('at least one' in str(e)), str(e)

    # ... and so is a list of the wrong maps. No theorem covers this: it is a property of
    # whatever did the assembling. Only checkable when itree 0 is NOT a parent, i.e. when
    # some family has an early trigger; otherwise the 'wrong' maps are the right ones.
    if max(nets) > 0:
        try:
            VarianceMultiMap(config, [_random_map(config, 0, rng) for _ in range(npri)])
            raise AssertionError('VarianceMultiMap accepted maps that are not the parents')
        except RuntimeError as e:
            assert 'early_trigger_level == 0' in str(e)

    atomic_print(f'    test_multimap(npri={npri}, ntrees={ntrees}): pass')


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

    tree = _tree(config, itree)
    fs = tree.frequency_subbands
    if nbeta is None:
        nbeta = (1 << (tree.tree_rank - fs.pf_rank)) * fs.M * tree.nprofiles
    nfreq = config.get_total_nfreq()

    Q = rng.normal(size=(nbeta, K))
    W = rng.normal(size=(nfreq, K))
    M = np.eye(K) if (mid == 'identity') else rng.normal(size=(K, K))
    m = VarianceMap.from_factors(config, itree, Q, W, mid=M, L=L, **kwargs)
    return m, Q @ M @ W.T


def test_factored_algebra(K=None):
    """The product identity, and every accessor that has a factored branch.

    Nothing here is about whether a factorization is any GOOD -- only that
    ``A = Q @ mid @ W.T`` is what the map reports through every route a consumer has.
    """

    rng = _rng()
    config = _random_config(rng)
    K = _draw_K(rng) if K is None else K

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
        # Only true when K is genuinely smaller than nfreq; on a drawn cell it need not be.
        if K < m.nfreq // 2:
            assert m.apply_cost() < m.nbeta * m.nfreq, 'K << nfreq should be cheaper'
        # The block sizer is also capped by nfreq -- it never returns more columns than the
        # matrix has -- which only a cell with few channels per row reaches at all.
        assert m.default_block_cols(1 << 10) == \
            min(m.nfreq, max(1, (1 << 10) // (8 * m.nbeta)))

    atomic_print(f'    test_factored_algebra(nbeta={m.nbeta}, K={K}): pass')


def test_factored_equivalence(K=None):
    """A dense map and a factored map that densify to the SAME matrix must agree everywhere.

    This is the cheapest way to catch a code path that still reaches for ``self.A``: every
    consumer below goes through rows() / row_sums(), so any that did not would diverge here.
    The factored map is genuinely rank-deficient, so the two representations are not
    trivially the same object.
    """

    rng = _rng()
    config = _random_config(rng)
    K = _draw_K(rng) if K is None else K

    # Nonnegative factors and a nonnegative 'mid', so the product is positive: the scoring
    # paths below are about VARIANCES, and get_distance() is only meaningful on one. (The
    # signed case is covered by test_factored_algebra, which touches no scoring.)
    tree = _tree(config, 0)
    fs = tree.frequency_subbands
    nbeta = (1 << (tree.tree_rank - fs.pf_rank)) * fs.M * tree.nprofiles
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

    atomic_print(f'    test_factored_equivalence(nbeta={nbeta}, K={K}): pass')


def test_factored_transformations(K=None):
    """inflated() and lift() keep the factorization; both agree with the dense answer."""

    rng = _rng()
    config = _random_config(rng)
    K = _draw_K(rng, lo=2) if K is None else K      # pinned_columns=[1] needs two columns

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

    atomic_print(f'    test_factored_transformations(nbeta={m.nbeta}, K={K}): pass')


def test_factored_validation(K=None):
    """Constructor rejections, and read-only enforcement on the factors.

    Only STRUCTURE is enforced -- shapes, a consistent K, dtypes, indices in range. The
    semiorthogonality flags and the pinned set are carried, not verified, so a map that
    claims them falsely is accepted here and is the steps' problem.
    """

    rng = _rng()
    config = _random_config(rng)
    K = _draw_K(rng, lo=3) if K is None else K      # pinned_columns=[1, 1] and [0, 2] below
    m, _ = _factored_map(config, 0, rng, K=K)
    Q, mid, W = np.asarray(m.Q), np.asarray(m.mid), np.asarray(m.W)
    nb, nf = m.nbeta, m.nfreq

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

    atomic_print(f'    test_factored_validation(nbeta={nb}, K={K}): pass')


@contextlib.contextmanager
def _open_one(path, gamma):
    """One primary tree's map from a memmapped read. VarianceMap has no open_asdf() of its
    own -- the scoped opener lives on VarianceMultiMap -- and these test configs have a
    single primary tree, so going through it is the same thing."""

    with VarianceMultiMap.open_asdf(path) as vmm:
        yield vmm.primary_map(gamma)


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


def test_asdf_io():
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

    rng = _rng()
    config = _random_config(rng)
    ntrees = _ntrees(config)
    npri = int(config.num_primary_trees)
    # itree is NOT gamma: the (gamma, 0) tree is the LAST of its family.
    iparents = [_itree(config, g) for g in range(npri)]
    # NO npri REQUIREMENT, because the four representation axes below are drawn PER MAP
    # rather than assigned one per tree: assigning them needs four primary trees to reach four
    # corners, whereas drawing them reaches the whole product on any config. Measured over 120
    # fully random configs (58 of them npri == 1): 120/120 round-trip, and all 16 combinations
    # of the product are reached.
    tmp = tempfile.mkdtemp()

    try:
        path = os.path.join(tmp, 'vm.asdf')

        # Every representation the dense path can produce: fine/coarse, certified or not,
        # y_true present or absent, float64 or float32. (The factored half of that product
        # is not reachable yet; the reader's refusal of it is checked below.)
        maps = []
        for i in iparents:
            m = _random_map(config, i, rng)
            R_m, r_m = int(m.pf_rank), int(m.tree_rank)
            if (r_m > R_m) and (rng.random() < 0.5):
                m = m.coarse_grain(int(rng.integers(R_m + 1, r_m + 1)))
            if rng.random() < 0.5:
                m = m.replace(is_admissible=False)
            if rng.random() < 0.5:
                m = m.replace(y_true=None)
            if rng.random() < 0.5:
                m = m.replace(A=np.asarray(m.A, dtype=np.float32))
            maps.append(m)

        prov = dict(algorithm='test', ntime=np.int64(1024), overrides=['a', 'b'],
                    nested=dict(host='here', seconds=1.5))
        vmm = VarianceMultiMap(config, maps, provenance=prov)
        vmm.write_asdf(path)

        for eager in (True, False):
            ctx = (_eager_ctx(VarianceMultiMap.from_asdf(path)) if eager
                   else VarianceMultiMap.open_asdf(path))
            with ctx as v2:
                assert v2.num_primary_trees == npri and v2.ntrees == ntrees
                assert v2.provenance == {'algorithm': 'test', 'ntime': 1024,
                                         'overrides': ['a', 'b'],
                                         'nested': {'host': 'here', 'seconds': 1.5}}

                # The inputs survive as yaml and re-parse into one shared object per file.
                assert int(v2.config.toplevel_tree_rank) == int(config.toplevel_tree_rank)
                assert all(m.config is v2.config for m in v2.maps)
                assert v2.detrender is None

                for (i, m) in enumerate(v2.maps):
                    w = maps[i]
                    assert m.itree == iparents[i] and m.L == w.L
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
                # of storing y_true at fine granularity. Needs a map that HAS one and is
                # certified; the draw above does not guarantee either.
                for (gi, wm) in enumerate(maps):
                    if (wm.y_true is not None) and wm.is_admissible:
                        assert abs(v2.primary_map(gi).get_distance()
                                   - wm.get_distance()) <= 1.0e-14
                        break

                if not eager:
                    # Uncompressed blocks are what make this possible, and memmapping is the
                    # scale path -- an asdf upgrade that changed its defaults would silently
                    # turn every large read into a full materialization.
                    chain, a = [], np.asarray(v2.primary_map(0).A)
                    while a is not None:
                        chain.append(a)
                        a = getattr(a, 'base', None)
                    assert any(isinstance(x, np.memmap) for x in chain), \
                        [type(x) for x in chain]

        # ESCAPING THE CONTEXT. open_asdf() closes the file in a finally, so a multimap that
        # outlives its with-block holds arrays backed by a file the reader has closed. Today
        # the mapping stays alive (the arrays hold a reference to it) and the read is still
        # correct -- but that is a property of the asdf version, not a promise this code
        # makes, so the bar asserted here is the one that matters for an archive: an escaped
        # read must return the right numbers or fail, never different numbers silently.
        escaped = None
        with VarianceMultiMap.open_asdf(path) as v3:
            escaped = v3
            assert np.array_equal(np.asarray(v3.primary_map(0).A), np.asarray(maps[0].A))

        try:
            after = np.array(np.asarray(escaped.primary_map(0).A))   # copy out from under the mapping
        except Exception:
            after = None                                 # an intelligible failure is fine
        assert (after is None) or np.array_equal(after, np.asarray(maps[0].A)), \
            'a multimap that escaped open_asdf() returned data that is neither correct nor' \
            ' an error'

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

        # A single-map file: readable by VarianceMap.from_asdf(), and refused by the multimap
        # reader, which covers every PRIMARY tree by definition. Note both readers take GAMMA.
        one = os.path.join(tmp, 'one.asdf')
        _g = min(1, npri - 1)
        maps[_g].write_asdf(one, provenance=dict(note='single'))
        m1 = VarianceMap.from_asdf(one, _g)
        assert m1.itree == iparents[_g] and np.array_equal(np.asarray(m1.A),
                                                           np.asarray(maps[_g].A))
        if npri > 1:
            # A single-map file is NOT a complete multimap -- unless npri == 1, where it is,
            # and the reader is right to accept it. Assert the complement there: that is
            # coverage the fixed four-tree config never had.
            expect_raise(lambda: VarianceMultiMap.from_asdf(one), 'covers EVERY primary tree')
            expect_raise(lambda: VarianceMap.from_asdf(one, 0),
                         f'primary trees present: [{_g}]')
        else:
            assert VarianceMultiMap.from_asdf(one).num_primary_trees == 1

        # ---- the tripwires ----
        #
        # ON A SECOND, TINY FILE, and that is the only reason it exists. Every check below is
        # about METADATA -- m_to_n, the is_coarse_grained/L pair, nbeta, itree and gamma, the
        # plan yaml, is_factored, format_version -- while each _corrupt() call is a full
        # non-lazy read plus a full rewrite. Run against the file above, the eleven of them
        # serialize its matrices eleven more times for nothing. Coarse-graining every map to
        # L = tree_rank leaves all of that metadata in place, and each map's nbeta with it,
        # at 2^(r-R) fewer rows.

        def coarsest(m):
            r = int(m.tree_rank)
            have = int(m.L) if m.is_coarse_grained else int(m.pf_rank)
            return m.coarse_grain(r) if (r > have) else m

        small = os.path.join(tmp, 'small.asdf')
        VarianceMultiMap(config, [coarsest(m) for m in maps],
                         provenance=prov).write_asdf(small)

        bad = os.path.join(tmp, 'bad.asdf')

        # m_to_n is the one field with no independent witness in the file.
        def break_m_to_n(root):
            mn = np.array(root['trees'][0]['m_to_n'])
            mn[-1] = 0 if (mn[-1] != 0) else 1
            root['trees'][0]['m_to_n'] = mn
        _corrupt(small, bad, break_m_to_n)
        expect_raise(lambda: VarianceMultiMap.from_asdf(bad), 'multiplet ordering convention')

        # is_coarse_grained and L are one fact. FLIP it rather than setting True: the draw
        # above may already have coarse-grained this map, and then True is the truth.
        _corrupt(small, bad, lambda root: root['trees'][0].__setitem__(
            'is_coarse_grained', not bool(root['trees'][0]['is_coarse_grained'])))
        expect_raise(lambda: VarianceMultiMap.from_asdf(bad), 'are one fact')

        # nbeta against the array it describes.
        # +1 rather than a literal: on a drawn geometry a literal can BE the real nbeta,
        # and then the reader is right not to complain.
        _k = min(1, npri - 1)
        _corrupt(small, bad, lambda root: root['trees'][_k].__setitem__(
            'nbeta', int(root['trees'][_k]['nbeta']) + 1))
        expect_raise(lambda: VarianceMultiMap.from_asdf(bad), 'stored nbeta')

        # itree against the block's own 'gamma' and the tree it names in the plan, which is
        # what stops a mislabelled entry from being read as a different tree.
        # +1 rather than 0: at npri == 1 with no early triggers the real itree IS 0, and
        # then setting 0 corrupts nothing.
        _corrupt(small, bad, lambda root: root['trees'][0].__setitem__(
            'itree', int(root['trees'][0]['itree']) + 1))
        expect_raise(lambda: VarianceMultiMap.from_asdf(bad), "stored 'itree'")

        # 'gamma' is the ENTRY KEY, so a file whose gamma list is not 0..npri-1 is refused
        # before anything is read. Without this, dropping an entry would look like a file for
        # a smaller config rather than a damaged one.
        # A DUPLICATE gamma, which needs two entries to construct. At npri == 1 there is
        # nothing to duplicate and the file is already correct, so the case is skipped.
        if npri > 1:
            _corrupt(small, bad, lambda root: root['trees'][npri-1].__setitem__('gamma', 0))
            expect_raise(lambda: VarianceMultiMap.from_asdf(bad), 'a file must hold exactly')

        _corrupt(small, bad, lambda root: root.__setitem__('trees', root['trees'][:-1]))
        expect_raise(lambda: VarianceMultiMap.from_asdf(bad), 'a file must hold exactly')

        # A plan yaml describing a different instrument: DedispersionPlan.from_yaml_string()
        # names the member that disagrees.
        _corrupt(small, bad, lambda root: root.__setitem__(
            'plan_yaml', root['plan_yaml'].replace('nprofiles: ', 'nprofiles: 1')))
        expect_raise(lambda: VarianceMultiMap.from_asdf(bad), 'nprofiles')

        # is_factored is checked AGAINST the arrays, never believed: a dense block that
        # claims to be factored is refused rather than reinterpreted. (The factored round
        # trip itself is test_asdf_factored().)
        _corrupt(small, bad, lambda root: root['trees'][0].__setitem__('is_factored', True))
        expect_raise(lambda: VarianceMultiMap.from_asdf(bad), 'carries no')

        # Version and identity, so the next format change is an error and not a KeyError.
        _corrupt(small, bad, lambda root: root.__setitem__('format_version',
                                                          FORMAT_VERSION + 1))
        expect_raise(lambda: VarianceMultiMap.from_asdf(bad), 'format_version is')

        _corrupt(small, bad, lambda root: root.pop('format_version'))
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
        nbytes_small = os.path.getsize(small)
    finally:
        shutil.rmtree(tmp, ignore_errors=True)

    atomic_print(f'    test_asdf_io(npri={npri}, ntrees={ntrees}):'
                 f' {nbytes/2**20:.1f} MiB file, eager + memmapped reads and every reader'
                 f' check exercised (tripwires on a {nbytes_small/2**20:.2f} MiB copy)')


def test_asdf_detrender():
    """A Detrender2dParams survives the file round trip, field by field.

    Every map the production CLI writes carries one (a variance map computed with a detrender
    is meaningless without knowing which detrender), and asdf_io round-trips it through
    to_yaml_string / from_yaml_string. Nothing else in this file writes a file with a
    detrender at all, so a detrender that was never plumbed into the writer -- or a yaml
    round trip that quietly dropped 'knots' -- would be invisible here and would only show
    up as an unreadable archive.
    """

    import os
    import tempfile

    rng = _rng()
    config = _random_config(rng)
    # rng=rng, so the detrender is drawn rather than fixed: n_phi = 0, n = 0 and W = 0 are
    # exactly the values a yaml round trip is most likely to drop (they are falsy), and the
    # fixed detrender never produced any of them.
    dparams = _make_test_detrender(config, rng=rng)
    tmp = tempfile.mkdtemp()

    try:
        path = os.path.join(tmp, 'det.asdf')
        npri = int(config.num_primary_trees)
        iparents = [_itree(config, g) for g in range(npri)]
        maps = [_random_map(config, i, rng, detrender=dparams) for i in iparents]
        VarianceMultiMap(config, maps, detrender=dparams).write_asdf(path)

        for eager in (True, False):
            if eager:
                vmm = VarianceMultiMap.from_asdf(path)
                d = vmm.detrender
            else:
                with VarianceMultiMap.open_asdf(path) as vmm:
                    d = vmm.detrender

            assert d is not None, eager
            for field in ('nfreq', 'M', 'n_phi', 'n', 'W', 'T'):
                assert getattr(d, field) == getattr(dparams, field), (eager, field)
            assert list(d.knots) == list(dparams.knots), eager

        # A file written with no detrender must come back None, not a default-constructed
        # one -- 'no detrender' and 'some detrender' are different physics.
        nodet = os.path.join(tmp, 'nodet.asdf')
        plain = [_random_map(config, i, rng) for i in iparents]
        VarianceMultiMap(config, plain).write_asdf(nodet)
        assert VarianceMultiMap.from_asdf(nodet).detrender is None
    finally:
        import shutil
        shutil.rmtree(tmp, ignore_errors=True)

    atomic_print(f'    test_asdf_detrender(nalpha={_nalpha_of(config, 0)}): pass')


def test_asdf_factored(K=None):
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

    rng = _rng()
    config = _random_config(rng)
    K = _draw_K(rng, lo=3) if K is None else K   # pinned_columns=[0, 2] below
    tmp = tempfile.mkdtemp()

    try:
        path = os.path.join(tmp, 'f.asdf')

        # itree 0 is NOT gamma 0's parent when the config has early triggers -- the (gamma,
        # 0) tree is the LAST of its family -- and a single-map file is keyed by gamma.
        itree0 = _itree(config, 0)
        tree = _tree(config, itree0)
        fs = tree.frequency_subbands
        Lc = int(fs.pf_rank) + 1
        nb_coarse = (1 << (int(tree.tree_rank) - Lc)) * int(fs.N) * int(tree.nprofiles)

        fine, _ = _factored_map(config, itree0, rng, K=K)
        coarse, _ = _factored_map(config, itree0, rng, K=K, L=Lc, nbeta=nb_coarse)

        # The remaining two axes are folded onto the two cases: fine + admissible + y_true +
        # pinned columns + both flags set, and coarse + uncertified + no y_true + no pins.
        y = np.abs(rng.normal(size=fine.nalpha)) + 1.0
        cases = [fine.replace(y_true=y, is_admissible=True,
                              Q_is_semiorthogonal=True, W_is_semiorthogonal=True,
                              pinned_columns=[0, 2]),
                 coarse.replace(y_true=None, is_admissible=False)]

        for (i, m) in enumerate(cases):
            m.write_asdf(path, provenance=dict(case=i))


            # The MEMMAPPED read goes through VarianceMultiMap.open_asdf(), which is the
            # only scoped opener there is -- and it refuses a file that does not cover every
            # primary tree. A single-map file IS such a file only when npri == 1, so the
            # memmapped arm runs only there. That is not a gap: the memmap path itself is
            # covered by test_asdf_io(), which writes a full multimap.
            arms = [True, False] if (int(config.num_primary_trees) == 1) else [True]
            for eager in arms:
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
                    # dense() recomputes Q @ mid @ W.T from the factors, and the file
                    # round trip does not change them -- so this is bitwise on a
                    # well-conditioned cell. On a drawn one the gemm can reassociate between
                    # the two calls; the factors themselves are compared bitwise above, which
                    # is the sharper check.
                    _s = max(1.0, float(np.abs(np.asarray(m.dense())).max()))
                    assert np.max(np.abs(np.asarray(g.dense()) - np.asarray(m.dense()))) \
                        <= 16.0 * np.finfo(np.float64).eps * _s, i

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

        _corrupt(path, bad, lambda root: root['trees'][0].__setitem__(
            'nbeta', int(root['trees'][0]['nbeta']) + 1))
        expect_raise(lambda: VarianceMap.from_asdf(bad, 0), 'stored nbeta')

        nbytes = os.path.getsize(path)
    finally:
        shutil.rmtree(tmp, ignore_errors=True)

    atomic_print(f'    test_asdf_factored(K={K}): {nbytes/2**10:.1f} KiB file,'
                 ' fine + coarse round-tripped through both readers, flag-vs-arrays checked')


####################################   the LP (varmap/lp.py)   ####################################


# Cost cap for the randomized LP/basis cell, in units of nbeta*nfreq. The LP tier's cost is
# HiGHS solve latency -- about 200 ms per solve_covering_lps() call at nbeta=224, nfreq=64,
# K=5 -- and a q_step is one subproblem per group, so cost tracks nbeta almost linearly with
# nfreq setting the constraint count. The value is set from that: an nbeta=224 cell puts the
# tier at 12.4 s of a 17.5 s iteration, which is most of the per-iteration budget, while this
# cap holds it near 3 s and still draws a different geometry every call.
LP_CELL_BUDGET = 4096


def _draw_lp_cell_config(rng):
    """A config whose coarse cell is small enough for the LP tier, from make_random().

    A FLOOR AND A CEILING, both on SIZE. The ceiling is LP_CELL_BUDGET: the tier's cost is
    HiGHS solve latency and a q_step is one subproblem per group, so cost tracks nbeta. The
    floor is that K reaches 12 across these tests and a factorization has at most
    min(nbeta, nfreq) modes, so a smaller cell says nothing about rank or column algebra.

    Nothing here filters on STRUCTURE -- early triggers, primary-tree count and the subband
    vector are whatever the draw gives. Measured acceptance is about 12%, i.e. ~8 draws per
    cell, and a draw costs well under a millisecond.
    """

    for _ in range(400):
        config = _random_config(rng, max_toplevel_rank=6, min_nalpha=0)
        tree = _tree(config, 0)
        fs = tree.frequency_subbands
        R, rr = int(fs.pf_rank), int(tree.tree_rank)
        if rr <= R:
            continue
        nbeta = (1 << (rr - R - 1)) * int(fs.N) * int(tree.nprofiles)
        nfreq = int(config.get_total_nfreq())
        if (nbeta >= 16) and (nfreq >= 16) and (nbeta * nfreq <= LP_CELL_BUDGET):
            return config

    # RAISE rather than returning the last draw. The floor is what lets every caller treat
    # K <= 8 as always fitting the cell (see _draw_K); a silent fallback to a config that
    # misses it would turn that into a rare, confusing failure somewhere else.
    raise RuntimeError('_draw_lp_cell_config: no config met the LP-cell floor in 400 draws '
                       f'(need nbeta >= 16, nfreq >= 16, nbeta*nfreq <= {LP_CELL_BUDGET})')


def _lp_cell(L=None, K=5, seed=None, nzero=1, scaled=True):
    """A small but REAL-geometry LP cell: (Abar, y, labels, W, config, coarse map).

    Real geometry rather than a random matrix, because the label arithmetic and the
    coarse-graining are half of what the steps have to get right.
    """

    rng = _rng(seed)
    config = _draw_lp_cell_config(rng)
    fine = _random_map(config, 0, rng, nzero=nzero)
    L = fine.pf_rank + 1 if (L is None) else L
    coarse = fine.coarse_grain(L)

    Abar = np.ascontiguousarray(coarse.dense(force=True))
    # 'scaled=False' is the basis tests' view of the same cell: they want the map as built,
    # not divided through by a power of two. See _basis_cell().
    scale = float(2.0 ** np.ceil(np.log2(float(Abar.max())))) if scaled else 1.0
    Abar = np.ascontiguousarray(Abar / scale)
    y = np.asarray(fine.y_true, dtype=np.float64) / scale
    labels = coarse.alpha_to_beta_block(0, coarse.nalpha)

    # A signed dictionary whose first column is nonnegative, which is what the additive
    # repairs need and what an SVD basis does not provide on its own.
    W = rng.normal(size=(coarse.nfreq, K))
    W[:, 0] = np.abs(W[:, 0]) + Abar.max(axis=0)
    return Abar, y, labels, np.ascontiguousarray(W), config, coarse, fine, rng


def _dominates(Q, W, Abar):
    """The elementwise admissibility test, done densely -- the tests here are small enough.

    THE TOLERANCE IS NOT SLACK, and an exact '>=' here is wrong. The repair decides how far
    to lift from its own evaluation of the product; this recomputes Q @ W.T with a different
    summation order, and two orders are not required to round identically. Measured over 12
    random cells, an exact test failed on 8 of them -- always 1 to 5 entries out of ~3500,
    always by 5.6e-17 to 1.1e-16 absolute where |Abar| is ~0.9, i.e. one ulp of float64.
    That is the same lesson get_distance()'s block-size assertions record.

    16 eps leaves two orders of margin over the observed roundoff while staying far below
    any real violation: the cases these tests care about (the sign blind spot, an
    unrepaired point) miss by ratios greater than 1, not by an ulp.
    """

    atol = 16.0 * np.finfo(np.float64).eps * max(1.0, float(np.abs(Abar).max()))
    return bool(np.all((Q @ W.T) >= Abar - atol))


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
    for kw, needle in ((dict(equilibrate=False), 'unequilibrated'),
                       (dict(slack=0.1), 'slack'), (dict(nnz_cap=2), 'nnz_cap')):
        expect_raise(LpConfig(**kw)._check_implemented, needle)
    expect_raise(lambda: LpConfig.recommended('x'), "'q' or 'w'")

    atomic_print(f'    test_lp_config: {len(dataclasses.fields(q))} fields, both presets,'
                 ' the four-way repair mapping, and the three refusals')


def test_lp_primitive():
    """solve_covering_lps() on problems whose answer is checkable by other means."""

    from .lp import LpConfig, solve_covering_lps, solve_cover_lp

    rng = _rng()

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

    # ---- CONSTRAINT GENERATION, i.e. LpConfig(cuts=True) ----
    #
    # This is what LpConfig.recommended() turns on, so it is what production solves with, and
    # nothing else in the suite reaches solve_cover_lp_cuts() at all: test_lp_config() checks
    # that recommended() SETS the flag, never that a solve with it works.
    #
    # WHAT MAKES IT CHECKABLE IS ITS OWN EXIT CONDITION. The loop stops when no row outside
    # the working set is violated by more than cuts_tol, so the point it returns is feasible
    # for the FULL row set -- and a relaxation optimum that is feasible for the original
    # problem is the original problem's optimum. Both halves are asserted below, against the
    # same solve with cuts off.
    #
    # cuts_min_rows IS LOWERED EXPLICITLY, and that is the whole reason this block needs its
    # own matrix. It defaults to 2048, i.e. it gates on a channel count no varmap test cell
    # comes near, so a cuts=True config on the LP cell takes the ordinary path and the cut
    # loop never runs. The shape below is chosen so that the loop reliably needs MORE THAN ONE
    # round: 800 rows against an initial working set of max(cuts_init*K, 64) = 64, at K = 6.
    # Measured over 40 draws, the mean round count spans 1.75 to 2.88 and the working set 44
    # to 56 rows. It costs about 90 ms.
    ncon, Kc, nsub = 800, 6, 8
    Mc = np.abs(rng.random((ncon, Kc))) + 0.2
    bc = np.abs(rng.random((ncon, nsub)))
    costc = Mc.sum(axis=0)
    # nonneg=False is the interesting sign convention: dropping rows from a free-sign LP
    # creates a recession direction, which is what cuts_agg's aggregate row exists to close.
    cut_cfg = LpConfig(nonneg=False, cuts=True, cuts_min_rows=64)
    Xc, ic = solve_covering_lps(Mc, bc, costc, cut_cfg)
    Xn, inn = solve_covering_lps(Mc, bc, costc, LpConfig(nonneg=False))

    # The loop RAN, on a working set that is a strict subset of the rows, and never fell back
    # to the ordinary solver -- without these, 'cuts=True' can be true and inert while
    # everything below still passes. The ROUND COUNT is reported rather than asserted: it is a
    # property of the draw, and the structural facts are the three below.
    assert ic.get('cuts') and ('cuts' not in inn), (ic.get('cuts'), inn.get('cuts'))
    assert ic['cuts_rows_mean'] < ncon, (ic['cuts_rows_mean'], ncon)
    assert ic['cuts_fallback'] == 0, ic['cuts_fallback']
    assert (ic['n_failed'] == 0) and (inn['n_failed'] == 0)

    worst_slack, worst_obj = 0.0, 0.0
    for j in range(nsub):
        # Feasible for EVERY row, not just the working set. The bar is 1e-6 relative against a
        # cuts_tol of 1e-9, so it cannot flake on the exit tolerance itself.
        scale = max(1.0, float(bc[:, j].max()))
        slack = float(((Mc @ Xc[j]) - bc[:, j]).min()) / scale
        worst_slack = min(worst_slack, slack)
        assert slack >= -1.0e-6, (j, slack)
        # ... and optimal, to the same kind of bar.
        oc, on = float(costc @ Xc[j]), float(costc @ Xn[j])
        worst_obj = max(worst_obj, abs(oc - on) / max(1.0, abs(on)))
        assert abs(oc - on) <= 1.0e-6 * max(1.0, abs(on)), (j, oc, on)

    # A failure with no fallback raises rather than returning a silent zero.
    try:
        solve_cover_lp(np.array([-1.0]), np.ones((2, 1)), np.array([1.0, 1.0]),
                       LpConfig(nonneg=False))
        raise AssertionError('an unbounded LP with no fallback should raise')
    except RuntimeError as e:
        assert 'no fallback' in str(e), str(e)

    atomic_print('    test_lp_primitive: free <= bounded, zero-rhs rows keep the product'
                 ' nonnegative, dead columns dropped, workers inert; cuts agree with no-cuts'
                 f' (worst slack {worst_slack:.1e}, worst objective {worst_obj:.1e},'
                 f" {ic['cuts_rounds_mean']:.1f} rounds on"
                 f" {ic['cuts_rows_mean']:.0f} of {ncon} rows)")


def test_lp_optimality(K=3, nfreq=7):
    """The Q-step is EXACT: its point is the LP optimum, checked by brute-force enumeration.

    An optimum of ``min s.q  s.t.  W q >= b`` with no bounds sits at a vertex where K of the
    constraints are active, so enumerating the K-subsets and keeping the feasible ones gives
    the optimum outright. That is a genuinely independent answer rather than a second run of
    the same solver.
    """

    import itertools

    from .lp import LpConfig, q_step

    rng = _rng()
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

    Abar, y, labels, W, config, coarse, _, _ = _lp_cell(K=5)
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
    rng = _rng()
    Qbad = np.abs(rng.random((nbeta, W.shape[1]))) * 0.4
    Qbad[:, 0] += 0.3
    Qm, _ = repair_rows(Qbad, W, None, Abar, cfg)
    assert _max_ratio(Qm, W, Abar) <= 1.0, 'the multiplicative repair failed its own test'
    # WHETHER THE BLIND SPOT SHOWS UP IS A PROPERTY OF THE DRAWN W, not of the code: it needs
    # the multiplicative repair to leave a NEGATIVE product entry, which a signed dictionary
    # does not always produce. Guarded and reported, so a run where it did not fire is
    # visible rather than green-by-luck.
    n_neg = check_nonneg(Qm, W)[0]
    blind_spot = bool(n_neg > 0 and not _dominates(Qm, W, Abar))
    Qa, _ = repair_additive(Qbad, W, None, Abar, cfg)
    assert _dominates(Qa, W, Abar) and (check_nonneg(Qa, W)[0] == 0)

    # AND THE RATIO CANNOT BE THE ADDITIVE STAGE'S EXIT TEST EITHER. Qm is that case already:
    # its max ratio is <= 1 (asserted above) while n_neg > 0, so an exit test on the ratio
    # returns early and throws away the lift the deficit had already computed, leaving a map
    # that reports admissible and scores a finite D while underestimating the variance. That
    # is what repair_additive did until 2026-08-21. NEITHER bit-identity gate covers it --
    # equiv2_lp.py and equiv2_arms.py have no cell with a negative product entry at ratio <= 1
    # -- which is how it survived the port, so the coverage has to live here. At CHORD the raw
    # W point has 120 such entries, against 1 at nbeta = 1600: the defect grows with scale.
    Qm2, stm = repair_additive(Qm, W, None, Abar, cfg)
    assert _dominates(Qm2, W, Abar) and (check_nonneg(Qm2, W)[0] == 0), \
        'repair_additive exited on the ratio and left a negative product entry'
    if blind_spot:
        assert stm['n_rows'] > 0, 'the additive stage reported no work on a point that needed it'

    # The additive lift is defined on the COLUMNS of W, so it refuses a non-identity mid
    # rather than quietly raising the wrong thing.
    try:
        fix_nonneg(Q, W, np.eye(W.shape[1]), Abar, cfg)
        raise AssertionError('fix_nonneg should refuse a non-identity mid')
    except RuntimeError as e:
        assert 'mid' in str(e), str(e)

    # Blocking is bit-identical, with a RAGGED tail forced: a splitter that bounds the block
    # size but not the tail ends on a short block, and numpy changes gemm kernel there.
    # BIT-EXACT BLOCKING NEEDS nfreq TO BE A MULTIPLE OF 8 -- blocking_is_exact() says so and
    # explains why. Every geometry in the shipped map library qualifies (400, 1600, 3200,
    # 4096, 16384, 28160), but a DRAWN nfreq does not, so this is a guard rather than an
    # assertion, and the report line says whether it ran.
    # DERIVED FROM nbeta, not literals. The interesting sizes are the ones whose TAIL is
    # shorter than the 8-row floor, since that is what exercises the tail merge rather than
    # just the block size -- and which sizes those are depends on nbeta, which is drawn. Take
    # the floor itself, one that divides, and every size in a small scan that leaves a short
    # tail.
    base, _ = repair_rows(Qbad, W, None, Abar, cfg)
    ragged = [b for b in range(9, min(nbeta, 64)) if 0 < (nbeta % b) < 8]
    sizes = sorted({8, max(1, nbeta // 3), *ragged[:3]}) if blocking_is_exact(nfreq) else []
    for block_rows in sizes:
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

    atomic_print(f'    test_lp_repairs(nbeta={nbeta}, nfreq={nfreq}, blocks={sizes}):'
                 f' four repairs dominate,'
                 f' the sign blind spot {"reproduced" if blind_spot else "NOT reached"}'
                 f' ({n_neg} negative entries) and fixed additively'
                 ' from both sides (ratio > 1 and ratio <= 1), blocking exact over ragged'
                 ' tails')


def test_lp_steps():
    """q_step and w_step end to end on a real-geometry cell, and their contracts."""

    from .lp import LpConfig, q_step, w_step, f as lp_f

    Abar, y, labels, W0, config, coarse, _, _ = _lp_cell(K=5)
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

    Abar, y, labels, W, config, coarse, _, rng = _lp_cell(K=12)
    nbeta, K = Abar.shape[0], W.shape[1]
    seed = np.zeros((nbeta, K))
    seed[:, 0] = (Abar / np.maximum(W[:, 0], 1e-300)[None, :]).max(axis=1) * (1 + 1e-12)
    assert _dominates(seed, W, Abar), 'the seed must be feasible for the incumbent to mean something'

    # Drawn: a literal index set assumes nbeta. Size varies too, so both the one-failure
    # and the many-failures cases are sampled (item 6).
    nhurt = int(rng.integers(1, min(6, nbeta) + 1))
    hurt = np.sort(rng.choice(nbeta, size=nhurt, replace=False))

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
        # The prefix width is a rescue_ladder entry CLIPPED TO K: a ladder rung wider than
        # the dictionary is used at width K, which is not itself a rung. That clipping is
        # invisible whenever K is large enough for every rung to fit, which a drawn cell does
        # not guarantee.
        nz = np.flatnonzero(Q[i] != 0.0)
        widths = {min(int(x), K) for x in cfg.rescue_ladder} | {K, int(nz.size)}
        assert nz.size == 0 or (nz.max() + 1) in widths, (i, nz, sorted(widths))
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
      - and it has a sharp price. A dictionary that cannot make the product positive
        everywhere makes EVERY LP on a signed reference come back infeasible. That is what
        sign-canonicalizing a basis, or pinning a nonnegative column, is for. Note the
        condition is on what the dictionary can REACH, not on its columns one at a time --
        see the construction below.
    """

    from .lp import LpConfig, q_step, w_step, covering_lp_data, _clip_rhs

    rng = _rng()
    config = _draw_lp_cell_config(rng)
    tree = _tree(config, 0)
    r, R = tree.tree_rank, tree.frequency_subbands.pf_rank
    L = R + 1
    N, P = tree.frequency_subbands.N, tree.nprofiles
    nbeta = (1 << (r - L)) * N * P
    nfreq = config.get_total_nfreq()

    ref = VarianceMap.from_factors(config, 0, rng.normal(size=(nbeta, 4)),
                                   rng.normal(size=(nfreq, 4)), L=L)
    Abar = np.array(ref.dense(), copy=True)
    Abar /= np.abs(Abar).max()
    frac_neg = float(np.mean(Abar < 0))
    # Reported: how signed the reference comes out is a property of the drawn factors.

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
    # A DICTIONARY THAT CANNOT COVER, CONSTRUCTED RATHER THAN DRAWN. "No nonnegative column"
    # is NOT the condition that forces infeasibility and does not imply it: the coefficients
    # here are free (nonneg=False), so a SIGNED combination of signed columns can be positive
    # everywhere -- and a drawn dictionary sometimes is exactly that. Measured, the span of a
    # random (nfreq, 5) gaussian contains a strictly positive vector in 6.3% of draws at
    # nfreq=16, 1.5% at 20, 0.25% at 24, and _draw_lp_cell_config() draws nfreq from 16 up.
    # A drawn dictionary therefore fails this assertion every few dozen iterations, with all
    # nbeta subproblems coming back 'optimal' (seed 3975277362 reproduces it).
    #
    # So force the condition rather than hoping for it: give two channels OPPOSITE rows. Then
    # (W x)[1] = -(W x)[0] for every x, signed or not, so the two cannot both be positive and
    # no choice of coefficients rescues it. No column is nonnegative either, which is the
    # weaker property the assert below still records.
    Wsigned = rng.normal(size=(nfreq, 5))
    Wsigned[1] = -Wsigned[0]
    assert not np.any((Wsigned.min(axis=0) >= 0) & (Wsigned.max(axis=0) > 0))

    _, _, ibad = q_step(Abar, Wsigned, LpConfig.for_qstep(nonneg=False, clip_rel=0.0),
                        Q0=np.zeros((nbeta, 5)), repair=False)
    # Every subproblem must FAIL, unconditionally now that the dictionary is built to make
    # them fail. The solver may classify a few as 'numerical' rather than 'infeasible', which
    # is the same outcome for this test's purpose.
    assert ibad['n_failed'] == nbeta, ibad['status']
    assert ibad['status'].get('infeasible', 0) + ibad['status'].get('numerical', 0) \
        == nbeta, ibad['status']

    atomic_print(f'    test_lp_negative_rhs(nbeta={nbeta}, nfreq={nfreq}):'
                 f' {frac_neg:.0%} of the reference is negative; the product stays positive'
                 f' there, and a dictionary with no nonnegative column is infeasible'
                 f' {nbeta}/{nbeta}')


def test_lp_building_blocks():
    """covering_lp_data() and majorizer_weights() against the direct computation."""

    from .lp import LpConfig, covering_lp_data, majorizer_weights, fprime as lp_fprime

    Abar, y, labels, W, config, coarse, _, _ = _lp_cell(K=4)
    rng = _rng()
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
    # The clip is applied when a config is given, and never in place on the reference. Assert
    # the CONTRACT -- everything below the floor becomes 0, everything at or above it is
    # untouched, and the reference is unchanged -- rather than that some entry qualified.
    # Whether any does is a property of the drawn row's dynamic range, not of the code: a
    # nearly flat row has nothing below half its max, which is a 1-in-25 failure when the
    # geometry is drawn rather than pinned.
    clip = 0.5
    _, _, bc = covering_lp_data(vmap, ref, 0, LpConfig.for_qstep(clip_rel=clip))
    floor = clip * float(Abar[0].max())
    below = Abar[0] < floor
    assert np.all(bc[below] == 0.0), int(below.sum())
    assert np.array_equal(bc[~below], Abar[0][~below])
    assert np.array_equal(ref.rows(0, 1)[0], Abar[0]), 'the clip wrote through to the reference'

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


def _basis_cell(L=None, seed=None, nzero=1):
    """(coarse ref, fine map, rng) at a small but REAL geometry.

    The SAME cell _lp_cell() builds -- it IS that function, with the LP-specific half (the
    rescaling and the signed dictionary W) left off. Kept as a separate name because the
    basis tests want the unscaled map and the LP tests want the scaled one.
    """

    _, _, _, _, _, coarse, fine, rng = _lp_cell(L=L, seed=seed, nzero=nzero, scaled=False)
    return coarse, fine, rng


def _decaying_map(K=8, seed=None, rate=0.5):
    """A coarse map whose spectrum DECAYS, built as a nonnegative low-rank product plus noise.

    _random_map()'s iid matrix has a nearly flat spectrum, which is the worst case for any
    low-rank method and tells a randomized range finder nothing. A real variance map is not
    like that, so anything comparing an approximate SVD against an exact one needs this instead.

    'rate' is the geometric ratio between successive modes. A real variance map decays SLOWLY,
    and that is the regime where a randomized range finder is hard and where its sampling
    settings show up -- so a test about those settings has to ask for one.

    DO NOT PIN 'seed'. Everything numpy draws below then repeats forever, including
    _random_config()'s gpu_valid coin flip -- so half the config space becomes invisible to
    the caller no matter how long the run. That failure hides well: the C++ half of
    make_random() draws through ksgpu::default_rng() and keeps varying, so the config still
    changes from run to run and only the one bit is frozen. Pass a seed only to reproduce a
    specific cell by hand.
    """

    rng = _rng(seed)
    # _draw_lp_cell_config(), not _random_config(): the cell has to have ROOM for the modes
    # the caller asks about, and an unconstrained config coarse-grains to as few as one
    # column.
    config = _draw_lp_cell_config(rng)
    fine = _random_map(config, 0, rng)
    nbeta, nfreq = fine.coarse_grain(fine.pf_rank + 1).shape

    s = float(rate) ** np.arange(K)
    A = (rng.uniform(0.2, 1.0, size=(nbeta, K)) * s) @ rng.uniform(0.2, 1.0, size=(K, nfreq))
    A += 1e-6 * rng.uniform(0.0, 1.0, size=A.shape)
    coarse = fine.coarse_grain(fine.pf_rank + 1)
    return coarse.replace(A=A, history_record=dict(step='synthetic'))


def test_svd(K=None):
    """svd() and truncate(): the dense path against numpy, the factored path against the dense
    one, and the flags against the matrices they describe."""

    ref, fine, rng = _basis_cell()
    # lo=3 for two separate reasons: the eps assertions compare mode K-1 against mode 0,
    # which at K = 1 is the same mode; and the truncate(2) rejection below pins column K-1
    # and requires it to fall OUTSIDE the kept prefix [0, 2), which needs K-1 >= 2.
    K = _draw_K(rng, lo=3) if K is None else K
    A = np.asarray(ref.dense(), dtype=np.float64)
    U, s, Vt = np.linalg.svd(A, full_matrices=False)

    # method='exact' EXPLICITLY. This block compares the dense path against numpy, and 'auto'
    # chooses by a cost heuristic -- on a drawn cell it can pick the randomized range finder,
    # whose result is not the exact truncation and correctly does not match. The randomized
    # path has its own section below.
    m = ref.svd(K, method='exact')
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
    assert ref.svd(K, eps=0.5 * frac, method='exact').factor_rank == K
    assert ref.svd(K, eps=1.5 * frac, method='exact').factor_rank < K
    assert ref.svd(eps=1.0e-13, method='exact').factor_rank <= min(A.shape)
    for bad in (lambda: ref.svd(), lambda: ref.svd(0)):
        try:
            bad()
            raise AssertionError('svd() should refuse this')
        except RuntimeError:
            pass

    # THE FACTORED PATH IS A DIFFERENT ALGORITHM -- two thin QRs and a K-by-K SVD, with no dense
    # product anywhere -- so agreeing with the dense one is a real check and not a tautology.
    # This is the rank-reduction path, and it is the reason svd() is a method at all.
    hi = ref.svd(min(3 * K, ref.nbeta, ref.nfreq), method='exact').canonicalize_signs()
    dense_hi = hi.replace(A=hi.dense(), history_record=dict(step='densify'))
    lo_f, lo_d = hi.svd(K), dense_hi.svd(K)
    scale = float(np.abs(np.asarray(dense_hi.A)).max())

    # THE BAR IS SCALED BY THE SPECTRAL GAP AT THE CUT, because that is what sets how far two
    # correct algorithms may differ. When s[K-1] and s[K] are nearly equal the rank-K subspace
    # is ill-conditioned and the two routes -- two thin QRs here, a dense SVD there -- pick
    # slightly different mixtures of the degenerate modes. A max-envelope of iid entries has
    # exactly that spectrum (one large mode, then a dense near-flat continuum), so the cut
    # usually lands inside a cluster: measured over 150 drawn cells the relative gap
    # (s[K-1] - s[K]) / s[0] has median 2.6e-3 and 1st percentile 4.3e-4.
    #
    # The product d = max|lo_f - lo_d| / scale tracks 1/gap as perturbation theory says, and
    # tightly: over those 150 cells d itself spans 1e-15 to 8.4e-14 while d * gap stays inside
    # [., 1.6e-16], p99 1.3e-16. The old fixed 1e-12 bar passed 150/150 there but failed on a
    # draw whose gap was 7.0e-5 (seed 3984097999, d = 1.5e-12) -- the bar was measuring
    # degeneracy, not correctness. 1e-14 below leaves ~60x on the measured constant, and the
    # 1e-12 floor keeps the old bar wherever the spectrum is well separated.
    gap = float((s[K-1] - s[K]) / s[0]) if (len(s) > K) else 1.0
    bar = max(1.0e-12, 1.0e-14 / max(gap, 1.0e-12)) * scale
    assert np.max(np.abs(lo_f.dense() - lo_d.dense())) < bar, (gap, bar / scale)
    assert np.max(np.abs(np.asarray(lo_f.W).T @ np.asarray(lo_f.W) - np.eye(K))) < 1e-12

    # shape_normalize decomposes the unit-sum SHAPES and folds the row sums back into Q, which
    # is exactly why Q is then NOT semiorthogonal and truncate() refuses the result.
    S = A / ref.row_sums()[:, None]
    Us, ss, Vst = np.linalg.svd(S, full_matrices=False)
    sn = ref.svd(K, shape_normalize=True, method='exact')
    assert np.allclose(sn.dense(), ref.row_sums()[:, None] * (Us[:, :K] @ np.diag(ss[:K])
                                                             @ Vst[:K]))
    assert (not sn.Q_is_semiorthogonal) and sn.W_is_semiorthogonal
    # The default is 'choose by rank', at the measured crossover.
    assert ref.svd(VarianceMap._SHAPE_NORMALIZE_RANK).history[-1]['shape_normalize']
    assert not ref.svd(K).history[-1]['shape_normalize']

    # The randomized range finder: 1 + 2*power_iters blocked passes, nothing of matrix size in
    # memory. Checked on a SLOWLY decaying spectrum, which is what a real variance map has and
    # the only regime where the sampling settings matter at all.
    dec = _decaying_map(K=24, rate=0.85)
    ex = dec.svd(K, method='exact', shape_normalize=False)
    scale = float(np.abs(np.asarray(ex.dense())).max())

    def err_of(m):
        """(RMS, WORST-CASE) error against the exact truncation. The second is the one that
        matters: a basis at the textbook sampling passes an RMS test comfortably and still
        costs 1.4x in delivered D, because D is paid on each group's worst channel."""
        d = np.asarray(m.dense()) - np.asarray(ex.dense())
        return (float(np.linalg.norm(d)) / float(np.linalg.norm(np.asarray(ex.dense()))),
                float(np.max(np.abs(d))) / scale)

    # The two arms below do NOT share a sketch, despite taking the same seed -- they use
    # different 'oversample', so the sketches are (nfreq, K+48) and (nfreq, K+10), filled
    # row-major, hence different matrices. A shared seed only makes each arm individually
    # reproducible, which a drawn seed does equally well.
    _sk = int(rng.integers(1 << 30))
    rd = dec.svd(K, method='randomized', shape_normalize=False,
                 rng=np.random.default_rng(_sk))
    err, emax = err_of(rd)
    assert err < 1e-3, err
    assert np.max(np.abs(np.asarray(rd.W).T @ np.asarray(rd.W) - np.eye(K))) < 1e-10

    # THE DEFAULTS ARE NOT THE TEXTBOOK ONES, deliberately, and this is what says so: the
    # shipped sampling must beat one power iteration with ten extra samples by a clear margin
    # on the worst-case bar. Without this, a well-meaning revert to the standard settings
    # passes every other test in the suite and costs 1.4x in D on a real map.
    tb_err, tb_emax = err_of(dec.svd(K, method='randomized', shape_normalize=False,
                                     oversample=10, power_iters=1,
                                     rng=np.random.default_rng(_sk)))
    # The 3x margin is a claim about SAMPLING quality, and it only has meaning when the
    # errors are above float64 roundoff. On a small drawn cell both arms are exact to
    # ~1e-15 and the ratio is noise.
    if tb_emax > 1.0e-12:
        assert emax < tb_emax / 3.0, (emax, tb_emax)

    # truncate() is the SVD's own prefix, so it must agree with asking svd() for that rank.
    t = m.truncate(2)
    assert (t.factor_rank == 2) and (not t.is_admissible)
    assert np.allclose(t.dense(), ref.svd(2, shape_normalize=False,
                                          method='exact').dense())
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

    atomic_print(f'    test_svd(nbeta={ref.nbeta}, K={K}): dense path matches numpy, factored matches'
                 f' the dense one to {np.max(np.abs(lo_f.dense() - lo_d.dense()))/scale:.2g},'
                 f' randomized to {err:.2g} relative ({emax:.2g} worst-case)')


def test_column_algebra(K=None):
    """The column helpers, each against the property it exists for."""

    from .basis import basis_envelope_column

    ref, fine, rng = _basis_cell()
    # lo=2 because pin_column() REPLACES the last column by default, and at K = 1 that
    # leaves the map with nothing but the pinned column.
    K = _draw_K(rng, lo=2) if K is None else K
    raw = ref.svd(K, method='exact')
    A0 = np.array(raw.dense())

    # THE MEASURED FAILURE MODE: numpy's per-mode sign is arbitrary, so a raw SVD basis has zero
    # nonnegative columns and everything that needs one fails. Canonicalization is exactly
    # invariant -- a sign flip is exact in floating point -- so there is never a reason to skip
    # it, and this asserts the invariance bitwise rather than to a tolerance.
    can = raw.canonicalize_signs()
    assert np.array_equal(can.dense(), A0), 'canonicalize_signs() is not bitwise invariant'
    assert np.all(np.asarray(can.W).sum(axis=0) >= 0.0)
    # 'a raw SVD basis has zero nonnegative columns' is the MEASURED failure mode, but on a
    # drawn cell a raw column can come out nonnegative by chance. What canonicalization
    # guarantees is that it does not DECREASE the count and produces at least one.
    assert can.n_nonneg_cols() >= max(1, raw.n_nonneg_cols())
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
    # Appending needs room for a (K+1)-th mode; K is already capped for that above, but a
    # cell with nbeta == K+1 leaves the appended column linearly dependent and dense() then
    # differs in the last ulp. Skip only in that degenerate case.
    if K + 1 < min(ref.nbeta, ref.nfreq):
        grow = can.pin_column(w, replace_last=False)
        # Bitwise on a well-conditioned cell -- the appended column carries a ZERO
        # coefficient, so the product is literally the same sum. On a drawn cell the extra
        # column can be nearly dependent on the others and the gemm reassociates, so allow
        # an ulp. The claim being tested is 'inert', and an ulp is inert.
        assert grow.factor_rank == K + 1
        assert np.max(np.abs(np.asarray(grow.dense()) - A0)) <= \
            16.0 * np.finfo(np.float64).eps * max(1.0, float(np.abs(A0).max())), \
            'appending a column with a zero coefficient must be inert'

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
    # A permutation that KEEPS both pinned columns (0 and 1), reorders them, and drops the
    # rest. Built from the actual rank rather than written out, since K is drawn.
    pick = [two.factor_rank - 1, 1, 0]
    sel = two.select_columns(pick)
    assert (sel.factor_rank == 3) and (list(sel.pinned_columns) == [2, 1])
    assert np.allclose(sel.dense(), np.asarray(two.Q)[:, pick]
                       @ np.asarray(two.mid)[np.ix_(pick, pick)]
                       @ np.asarray(two.W)[:, pick].T)
    assert not sel.is_admissible
    try:
        two.select_columns(list(range(1, two.factor_rank)))    # drops pinned column 0
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
    # Zero coefficients on the appended columns, so the product is the same sum -- bitwise on
    # a well-conditioned cell, within an ulp when the appended columns are nearly dependent.
    _scale = max(1.0, float(np.abs(np.asarray(base.dense())).max()))
    assert np.max(np.abs(np.asarray(aug.dense()) - np.asarray(base.dense()))) <= \
        16.0 * np.finfo(np.float64).eps * _scale
    assert aug.is_admissible and (list(aug.pinned_columns) == [0])

    # with_basis() takes a W from anywhere and hands back something ready for a qstep().
    wb = ref.with_basis(np.asarray(can.W), pinned_columns=[0])
    assert wb.is_factored and (not wb.is_admissible)
    assert np.count_nonzero(np.asarray(wb.Q)) == 0
    assert np.array_equal(np.asarray(wb.W), np.asarray(can.W))

    atomic_print(f'    test_column_algebra(nbeta={ref.nbeta}, K={K}): sign canonicalization inert'
                 f' ({raw.n_nonneg_cols()} -> {can.n_nonneg_cols()} nonnegative columns),'
                 ' column scaling inert, pinned indices remapped')


def test_reorthogonalize(K=None):
    """reorthogonalize(): the same matrix, a semiorthogonal W, and the pinned column intact.

    The last is the whole reason for the ordered QR. A plain rotation destroys the nonnegative
    column that the seed and the additive repair depend on, which is measured at 1.769x in D and
    is reproduced here on purpose.
    """

    from .basis import basis_envelope_column

    ref, fine, rng = _basis_cell()
    # lo=3 for the permutation below: it moves the pinned column off index 0, which needs
    # three columns to be a permutation a plain QR would not preserve by accident.
    K = _draw_K(rng, lo=3) if K is None else K
    w = basis_envelope_column(ref)
    m = ref.svd(K, method='exact').canonicalize_signs().pin_column(w).replace(is_admissible=True)
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
    # A drawn permutation that puts the pinned column somewhere other than 0, which is
    # where a plain QR would preserve it by accident.
    perm = [1, 2, 0] + list(range(3, K))
    moved = m.select_columns(perm)
    assert list(moved.pinned_columns) == [perm.index(0)]
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

    atomic_print(f'    test_reorthogonalize(nbeta={ref.nbeta}, K={K}): exact to'
                 f' {np.max(np.abs(ro.dense()-A0))/scale:.2g}, pinned column preserved up to a'
                 ' positive scale, and lost by the plain rotation')


def test_svd_optimize(K=None, j=None):
    """VarianceMap.svd_optimize() against a factorization with KNOWN redundancy.

    The point of the method is to find the TRUE rank of a factorization assembled from
    per-group pieces that were never made independent of each other, so the test builds
    exactly that: a rank-K pair (Q, W), then j extra columns on each side that are random
    linear combinations of the first K. The stored rank is K+j and the true rank is K, and
    nothing about the construction tells the method which is which.

    An SVD-based rank test is the whole point of the method (a Gram-matrix or QR shortcut
    over-truncates on these spectra -- see the method's docstring), so the bar on the
    recovered rank is EXACT: K, not "about K".

    Also checks the two claims the method makes about its output, since they are what
    downstream code reads and neither is verified by the constructor: that both factors are
    numerically semiorthogonal, and that 'mid' really is diagonal. And the early exit -- a
    full-rank input must come back as the SAME OBJECT, so a caller can use it in a loop
    without accumulating history records or copies.
    """

    rng = _rng()
    config = _random_config(rng)
    K = _draw_K(rng) if K is None else K
    j = int(rng.integers(1, 5)) if j is None else j   # the redundant columns, >= 1 by design
    # _nalpha_of() takes the TREE INDEX, and the base tree is not always index 0 -- with early
    # triggers the (gamma=0, e=0) tree can be at any index. Passing a literal 0 here built Q
    # at another tree's row count and VarianceMap rejected it, on the draws where they differ.
    itree = _itree(config, 0)
    nbeta, nfreq = _nalpha_of(config, itree), int(config.get_total_nfreq())

    Q = rng.standard_normal((nbeta, K))
    W = rng.standard_normal((nfreq, K))
    Qx = np.hstack([Q, Q @ rng.standard_normal((K, j))])
    Wx = np.hstack([W, W @ rng.standard_normal((K, j))])

    # THE TRUE RANK IS CAPPED BY THE MATRIX SHAPES, not just by K. Qx spans the column space
    # of an (nbeta, K) gaussian and Wx that of an (nfreq, K) one, so the product's rank is
    # min(K, nbeta, nfreq) -- and a drawn config gives nfreq < K in 5% of draws (nfreq is as
    # low as 1, since zone_nfreq is drawn in [2^r/4, 2^r] and r goes down to 2). The bar stays
    # EXACT, it is just the right number; the same cap is spelled out for the 'wide' case
    # below. Reported on the summary line so a degenerate draw is visible rather than silent.
    Krank = min(K, nbeta, nfreq)

    vmap = VarianceMap.from_factors(config, itree, Qx, Wx)
    assert vmap.factor_rank == K + j, (vmap.factor_rank, K, j)

    opt = vmap.svd_optimize()
    assert opt.factor_rank == Krank, (opt.factor_rank, Krank, K, nbeta, nfreq)

    # A is unchanged. apply() and rows() go through different code (a three-factor product
    # against a materialized block), so check both rather than assuming they agree.
    v = rng.uniform(0.5, 1.5, size=nfreq)
    a0, a1 = np.asarray(vmap.apply(v)), np.asarray(opt.apply(v))
    e_apply = float(np.abs(a1 - a0).max() / np.abs(a0).max())
    nb = min(vmap.nbeta, 512)
    r0, r1 = np.asarray(vmap.rows(0, nb)), np.asarray(opt.rows(0, nb))
    e_rows = float(np.abs(r1 - r0).max() / np.abs(r0).max())
    assert e_apply < 1.0e-12, e_apply
    assert e_rows < 1.0e-12, e_rows

    # The two claims the flags make, checked against the matrices rather than read back out
    # of the object that set them.
    assert opt.Q_is_semiorthogonal and opt.W_is_semiorthogonal
    Qo, Wo, I = np.asarray(opt.Q), np.asarray(opt.W), np.eye(Krank)
    e_q = float(np.abs(Qo.T @ Qo - I).max())
    e_w = float(np.abs(Wo.T @ Wo - I).max())
    assert e_q < 1.0e-12, e_q
    assert e_w < 1.0e-12, e_w
    mid = np.asarray(opt.mid)
    assert np.array_equal(mid, np.diag(np.diag(mid))), 'mid is not diagonal'

    # The early exit, and that it is the same OBJECT rather than an equal one.
    assert opt.svd_optimize() is opt, 'a full-rank factorization was not returned unchanged'

    # nfreq < K0, where the thin SVD of W returns fewer than K0 columns and Z comes out
    # RECTANGULAR. Found by a random draw rather than by design -- the reporting index in
    # svd_optimize() ran off the end of the singular-value array -- so it is pinned here
    # rather than left to chance. The recovered rank is capped by nfreq, not by K.
    wide = VarianceMap.from_factors(config, itree,
                                    rng.standard_normal((nbeta, nfreq + 4)),
                                    rng.standard_normal((nfreq, nfreq + 4)))
    wopt = wide.svd_optimize()
    assert wopt.factor_rank == min(nfreq, nbeta), (wopt.factor_rank, nfreq, nbeta)
    vw = rng.uniform(0.5, 1.5, size=nfreq)
    b0, b1 = np.asarray(wide.apply(vw)), np.asarray(wopt.apply(vw))
    assert float(np.abs(b1 - b0).max() / np.abs(b0).max()) < 1.0e-12

    # Pinned columns are not supported; the method asserts rather than silently unpinning.
    # (See the comment at that assert -- no production path reaches it today.)
    pinned = VarianceMap.from_factors(config, itree, Qx, Wx, pinned_columns=[0])
    try:
        pinned.svd_optimize()
    except AssertionError:
        pass
    else:
        raise AssertionError('svd_optimize() accepted a map with a pinned column')

    atomic_print(f'    test_svd_optimize(nbeta={nbeta}, nfreq={nfreq}): rank'
                 f' {K+j} -> {opt.factor_rank} (true rank {Krank}); A unchanged to'
                 f' {max(e_apply, e_rows):.3g}; Q^T Q - I {e_q:.3g}, W^T W - I {e_w:.3g}')


def test_greedy_bookkeeping():
    """The greedy merger's RUNNING objective against a distance recomputed from scratch.

    _AgglomerativeEnvelope maintains cost[], Dlt[] and the best-merge pointers incrementally,
    repairing stale entries after every merge. That bookkeeping is intricate and it decides
    which merges are taken -- but basis(K) is rebuilt from the stored matrix, so a corrupted
    cost table produces a basis that is merely SUBOPTIMAL, not wrong. Every other test here
    would still pass.

    So the check is on tree.objective[K], which is the incremental accounting's own answer,
    against the same quantity computed from (Abar, y, labels) and the replayed merge tree with
    no incremental state involved. This is what the deleted research self-test did, and it is
    the only thing that ever validated the merge loop.
    """

    from .basis import greedy_envelope_tree
    from .distance import f, YTRUE_FLOOR

    ref, fine, _ = _basis_cell()

    # on_shapes=False is the raw-space objective, where y is the fine row sums and the
    # objective really is D0 of the max-envelope approximation. (The default normalizes the
    # rows and sets y = 1, which is a different, deliberately scale-free objective.)
    tree = greedy_envelope_tree(ref, on_shapes=False)

    Abar = np.asarray(ref.A, dtype=np.float64)
    y = np.asarray(fine.y_true, dtype=np.float64)
    labels = ref.alpha_to_beta_block(0, fine.nalpha)
    scored = y >= YTRUE_FLOOR
    y, labels = y[scored], labels[scored]

    checked = 0
    for K in sorted(tree.objective):
        root = tree.roots(K)
        # The envelope of each surviving cluster, rebuilt from Abar alone.
        S = np.zeros(tree.K0)
        for c in np.unique(root):
            S[c] = Abar[root == c].max(axis=0).sum()
        want = float(np.mean(f(S[root[labels]] / y)))
        got = tree.objective[K]
        assert abs(got - want) <= 1.0e-9 * max(1.0, abs(want)), (K, got, want)
        checked += 1

    assert checked >= 4, checked

    # At K0 clusters every COARSE row is its own envelope, so the approximation is the coarse
    # map itself and the objective is the coarse-graining FLOOR at this L -- the best any
    # coarse-assigned method could reach here, no matter how good the basis. (It is not zero:
    # zero would be the fine map's own D0, and the max-envelope over each group is strictly
    # above the fine rows inside it.) That makes it a second, independent handle on the same
    # number, since get_distance() reaches it by a completely different route.
    assert abs(tree.objective[tree.K0] - ref.get_distance()) <= 1.0e-12 * ref.get_distance()

    # The frontier is monotone: merging can only ever raise the objective.
    ks = sorted(tree.objective)
    vals = [tree.objective[k] for k in ks]
    assert all(vals[i] >= vals[i+1] - 1.0e-12 for i in range(len(vals)-1)), vals

    atomic_print(f'    test_greedy_bookkeeping(nbeta={ref.nbeta}): pass')


def test_basis_constructors(K=None):
    """Every module-level basis constructor, through the one thing they are all for: a Q-step
    against it produces an admissible map."""

    from . import basis as vb

    ref, fine, rng = _basis_cell()
    # lo=2: the K = 1 constructors are checked explicitly below, against the whole-map
    # envelope, so the drawn K is there to cover the multi-atom case.
    K = _draw_K(rng, lo=2) if K is None else K
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
    # 'sized differs from group-blind' has teeth only when the group sizes actually differ,
    # which needs a level > 0 subband (N > 1). GUARDED AND REPORTED: it is a property of the
    # drawn geometry, not of the code, and 'pirate_frb dev coverage' tracks the rate.
    sized_matters = bool(ref.group_sizes().max() > ref.group_sizes().min())
    # ... and even then the two merges can coincide when the clustering is forced (few
    # groups, or a spectrum with no ties to break). Report rather than assert.
    sized_matters = sized_matters and not np.allclose(flat.basis(K), W_greedy)

    W_qr = vb.basis_pivoted_qr(ref, K)
    # Its atoms are literally rows of the map, which is where the nonnegativity comes from.
    for c in range(K):
        assert np.any(np.all(np.abs(A - W_qr[:, c][None, :]) < 1e-15, axis=1)), c

    W_rand = vb.basis_random(ref, K, rng=rng)
    assert (W_rand.shape == (ref.nfreq, K)) and (W_rand.min() >= 0.0)

    D, skipped = {}, []
    for name, W in (('svd', W_svd), ('greedy', W_greedy), ('pivoted_qr', W_qr),
                    ('random', W_rand)):
        # A BASIS THAT CANNOT COVER EVERY CHANNEL IS NOT A FAILURE OF THE Q-STEP. If some
        # input channel is zero in every atom of W, no nonnegative combination reaches Abar
        # there and the repair says so by name. On a drawn cell that happens -- basis_random
        # draws its atoms, and a narrow cell can leave a channel uncovered -- so it is
        # reported rather than asserted. The other bases still have to work.
        if not np.all(np.abs(W).max(axis=1) > 0.0):
            skipped.append(name)
            continue
        try:
            m = ref.with_basis(W).canonicalize_signs().qstep(ref, workers=1)
        except RuntimeError as e:
            # The repair refuses by name when no nonnegative combination of the atoms can
            # reach Abar in some channel. That is a statement about THIS basis on THIS drawn
            # cell, not a defect -- basis_random draws its atoms, and a narrow cell can leave
            # a channel the span misses. Reported, and the other bases still have to work.
            if 'zero where Abar is not' not in str(e):
                raise
            skipped.append(name)
            continue
        assert m.is_admissible, name
        assert m.measure_admissibility(ref).admissible, name
        D[name] = m.get_distance()
    assert D, 'every basis left a channel uncovered, so nothing was checked'

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

    atomic_print('    test_basis_constructors(nbeta=%d, K=%d): %d bases, all admissible'
                 ' after a Q-step%s; D = %s'
                 % (ref.nbeta, K, len(D),
                    ('' if not skipped else f' (skipped {skipped}: a channel no atom covers)')
                    + ('' if sized_matters else '; size-weighted merge check not reached'),
                    ', '.join(f'{k} {v:.4g}' for k, v in D.items())))


def test_map_steps(K=None):
    """qstep() / wstep() / repair(): the wrappers against the array level they wrap.

    The numerics are varmap.lp's and are tested there. What is tested here is everything the
    wrapper adds and could get wrong: the reference matrix, the labels, folding 'mid', the
    pinned set, and where is_admissible comes from.
    """

    from . import lp
    from .basis import basis_envelope_column

    ref, fine, rng = _basis_cell()
    # lo=2: seed_onehot() needs a nonnegative column besides the pinned one.
    K = _draw_K(rng, lo=2) if K is None else K
    Abar = np.asarray(ref.dense(), dtype=np.float64)
    # canonicalize_signs BEFORE pinning, and pin by APPENDING rather than replacing: pinning
    # over the last column can leave no nonnegative column when K is small and the remaining
    # columns are all signed, which seed_onehot() then refuses.
    init = (ref.svd(K, method='exact').canonicalize_signs()
            .pin_column(basis_envelope_column(ref)))
    if init.n_nonneg_cols() == 0:
        init = (ref.svd(K, method='exact').canonicalize_signs()
                .pin_column(basis_envelope_column(ref), replace_last=False))
        K = init.factor_rank
    W0, Q0 = np.asarray(init.W, dtype=np.float64), np.asarray(init.Q, dtype=np.float64)

    # The one-hot seed is admissible BY CONSTRUCTION, which is what makes it a usable fallback
    # for a failed subproblem: one nonnegative atom per group, scaled until it dominates. It
    # must not depend on where the scale is kept, so a non-identity mid gives the same map.
    # init pins the envelope column, so it HAS a nonnegative column by construction --
    # except at K == 1, where pin_column replaces the only column and canonicalization has
    # nothing left to work with.
    sd = init.seed_onehot(ref)
    assert sd.is_admissible and sd.measure_admissibility(ref).admissible
    assert np.all(np.count_nonzero(np.asarray(sd.Q), axis=1) == 1), 'not one-hot'
    assert abs(init.rescale_columns().seed_onehot(ref).get_distance()
               - sd.get_distance()) < 1e-12 * sd.get_distance()
    # ... and refuses a basis that has none. numpy's per-mode sign is arbitrary, so a RAW SVD
    # basis usually has zero nonnegative columns -- but on a drawn cell one can come out
    # nonnegative by chance, and then seed_onehot() is right to succeed. Check the refusal
    # only when there is genuinely nothing to seed from.
    raw_basis = ref.svd(K, method='exact')
    if raw_basis.n_nonneg_cols() == 0:
        try:
            raw_basis.seed_onehot(ref)
            raise AssertionError('seed_onehot() accepted a basis with no nonnegative column')
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
    # additive repair not helping. Same caveat as seed_onehot() above: a RAW SVD basis
    # usually has no nonnegative column, but on a drawn cell one can appear by chance, and
    # then there is nothing to refuse.
    if raw_basis.n_nonneg_cols() == 0:
        try:
            raw_basis.replace(is_admissible=False).repair(ref, cfg=ca)
            raise AssertionError('repair() accepted an additive stage with no nonneg column')
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

    # NO ALTERNATION LOOP HERE. This test already pins ONE qstep and ONE wstep to
    # varmap.lp bitwise, and test_lp_steps() checks monotonicity over an alternation at the
    # array level; repeating it through the wrappers costs four more LP solves and adds no
    # claim.

    # Geometry mismatches are refused by name rather than being broadcast into nonsense.
    # ref.L + 1 must still be a legal coarse-graining rank; on a drawn cell it need not be.
    bad_refs = [(fine, 'shape mismatch')]
    if ref.L + 1 <= ref.tree_rank:
        bad_refs.append((ref.coarse_grain(ref.L + 1), 'shape'))
    for bad, needle in bad_refs:
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

    atomic_print(f'    test_map_steps(nbeta={ref.nbeta}, K={K}): wrappers bit-identical to varmap.lp,'
                 f' mid folded for the additive repair, D {m.get_distance():.6g}')


def test_report(K=None):
    """varmap/report.py: the record is assembled from the map, and survives a json round trip.

    The property worth testing is not the formatting -- it is that a record says what the map
    says, since a results table that drifts from the map it describes is worse than no table.
    """

    import json
    import os
    import tempfile

    from . import basis as vb
    from . import report as vr

    ref, fine, rng = _basis_cell()
    K = _draw_K(rng) if K is None else K
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
    assert ('max_r' not in rec) and ('max_diff' not in rec)

    # With one, the measurement wins -- including when it CONTRADICTS the flag, which is the
    # case the distinction exists for.
    adm = m.measure_admissibility(ref, inflate=True)
    rec2 = vr.row_dict(m, D, adm=adm)
    assert rec2['admissible'] and (abs(rec2['max_r'] - adm.max_r) < 1e-15)
    assert abs(rec2['max_diff'] - adm.max_diff) < 1e-15, (rec2['max_diff'], adm.max_diff)
    assert (len(rec2['argmax_r']) == 2) and (rec2['inflation'] is not None)

    lying = m.inflated(0.5).replace(is_admissible=True,
                                    history_record=dict(step='lie'))
    bad = lying.measure_admissibility(ref, inflate=True)
    liar = vr.row_dict(lying, np.inf, adm=bad)
    assert liar['admissible'] is False, 'the measurement must override the flag'
    assert (liar['max_r'] > 1.0) and np.isfinite(liar['D_inflated'])
    # '>' not '>=' only holds when the inflation actually moves the map; at max_r within
    # roundoff of 1 the two D values coincide.
    assert liar['D_inflated'] >= D * (1.0 - 1.0e-12), 'inflating cannot improve D'

    # 'extra' is where an experiment puts what nobody anticipated, and it takes numpy scalars
    # straight from a step's info dict -- which is exactly what json.dump() refuses.
    info = m.history[-1]
    rec3 = vr.row_dict(m, D, extra=dict(max_r_raw=np.float64(info['max_r_raw']),
                                        n_lp=np.int64(info['n_lp']), tag='x'))
    assert isinstance(rec3['max_r_raw'], float) and isinstance(rec3['n_lp'], int)

    # frontier(): one record per rank, K is what was ASKED for, and D falls with rank.
    # Two ranks, not three: the assertion is that D FALLS with rank, which needs two.
    ranks = [2, 6]
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

    atomic_print(f'    test_report(nbeta={ref.nbeta}, K={K}): record matches the map, measurement overrides'
                 f' the flag, D {rows[0]["D"]:.4g} -> {rows[-1]["D"]:.4g} over ranks {ranks},'
                 ' json round trip exact')


####################################   varmap/detrender_free.py   ################################


def _compare_maps(got, ref, label):
    """(elementwise relative, sup-norm relative) difference between two maps, in row blocks.

    Returns BOTH because they answer different questions and the second is the one to put a
    tight bar on. The elementwise figure is ``max |got-ref| / (|ref| + 1e-6 * max|ref|)`` --
    a relative error where the element is large enough for one to be meaningful, and an
    absolute one below that. The floor is not decoration: WITHOUT it the figure is set by the
    smallest element compared rather than by how well the map is approximated (the same
    effect AdmissibilityResult documents for max_r), and with svd_optimize() rotating both
    sides it reached 173 on a random draw whose sup-norm error was 1.7e-14. The sup-norm
    figure ``max|got-ref| / max|ref|`` has no such tail: median 7.1e-16, max 5.6e-13 over 40
    random configs.

    Row-blocked so that neither map has to be densified: 'ref' is already dense (coarse_grain
    returns a dense matrix), and materializing 'got' as well would double the peak.

    Entries where ref is EXACTLY zero are required to be exactly zero in got, not merely
    small. That is structural rather than lucky -- a channel absent from a group leaves that
    group's rows of W at zero, and a subband absent from a group never has those columns
    written into its rows of Q -- and it is an exact bar, which tests.py's preamble prefers.
    """

    assert got.shape == ref.shape, (label, got.shape, ref.shape)
    nb = ref.default_block_rows()
    blocks = [(st, min(st + nb, ref.nbeta)) for st in range(0, ref.nbeta, nb)]

    # FIRST PASS for the scale, over 'ref' only. It is needed BEFORE the elementwise figure
    # can be formed (see the floor below), and ref is the dense side, so this pass is cheap --
    # it is materializing 'got' that the blocking exists to avoid.
    scale = 0.0
    for (start, stop) in blocks:
        scale = max(scale, float(np.abs(np.asarray(ref.rows(start, stop))).max()))

    # THE FLOOR IS WHAT MAKES THE ELEMENTWISE FIGURE MEAN ANYTHING once svd_optimize() is in
    # the picture. Both sides now carry an SVD rotation's roundoff, ~1e-14 of max|A| in
    # ABSOLUTE terms, and a bare max|a/b - 1| divides that by |b| -- so one element sixteen
    # decades below the max sends the figure to 173 while the sup-norm sits at 1.7e-14
    # (measured, on a random draw). Dividing by (|b| + floor*scale) instead reports a relative
    # error where the element is big enough for one to be meaningful and an absolute one
    # below that. At floor=1e-6 the same draw reports 1.7e-8.
    #
    # Elements below floor*scale are not unchecked: the sup-norm figure covers them, to
    # 1e-9 of the max.
    floor = 1.0e-6 * scale
    worst_rel, worst_abs = 0.0, 0.0
    for (start, stop) in blocks:
        a = np.asarray(got.rows(start, stop))
        b = np.asarray(ref.rows(start, stop))
        # OFF-SUPPORT: small relative to the matrix, not bit-exact. An exact 0.0 is not a
        # property either side has once svd_optimize() rotates the factorization -- the same
        # reason the floor above exists. Measured on a draw that breaks an exact-zero bar
        # (seed 2678119351): the reference had 3 exact zeros out of 2943360 entries, and
        # 'got' carried 5.5e-18 there, i.e. 3.5e-17 of max|A|. The bar below is
        # five decades above that and still 1000x tighter than the sup-norm bar the caller
        # applies, so a scatter landing where it should not is still caught here first.
        nz = (b != 0.0)
        if np.any(~nz):
            leak = float(np.abs(a[~nz]).max()) / max(scale, np.finfo(float).tiny)
            if leak >= 1.0e-12:
                raise AssertionError(f'{label}: {leak:.3g} of max|A| where the reference is'
                                     ' exactly zero')
        if floor > 0.0:
            worst_rel = max(worst_rel,
                            float((np.abs(a - b) / (np.abs(b) + floor)).max()))
        elif np.any(nz):
            worst_rel = max(worst_rel, float(np.abs(a[nz] / b[nz] - 1.0).max()))
        worst_abs = max(worst_abs, float(np.abs(a - b).max()))
    return worst_rel, (worst_abs / scale if (scale > 0.0) else 0.0)


def test_base_varmap_coarse(nrandom=1, verbose=True):
    """detrender_free.compute_detrender_free_base_map(), structurally. No GPU.

    NOT A NUMERICAL CHECK ON THE MAP -- that is test_multimap_vs_sweep(), which compares
    mm.maps[0] (which IS compute_detrender_free_base_map(config)) against the brute-force
    sweep, element by element over the full matrix, on five random draws per iteration at
    max_toplevel_rank=9. This test is what remains once the numerical check lives there: the
    three properties of the construction that no comparison against another map would catch,
    plus the coarse path against the fine map's own coarse_grain().

      1. THE PER-GROUP FACTORIZATION CONVENTION, checked on the SdMatrix objects rather than
         end-to-end. SdMatrix.factorize() stores dense_matrix.T ~= Q_factor @ W_factor.T, and
         a transpose slip there is SILENT whenever D*P == F, since the shapes still match --
         exactly the case a whole-map comparison is least likely to hit. The count of square
         groups is reported below, so a silent loss of that coverage is visible.
      2. THE COARSE PATH against coarse_grain(), which is the reference implementation of the
         max-envelope and the thing the direct construction has to agree with. Each config is
         checked at ONE L drawn from [R, r], since 'every L' on a rank-10 draw is neither
         cheap nor more informative.
         Both boundaries can come up and both are interesting: L = R leaves the DM axis alone
         and only merges M -> N, and L = r collapses it entirely to nbeta = N*P.
         check_ref_covers_y_true() rides along -- the one runtime check on the property the
         whole coarse path rests on, that the map does not UNDERestimate.
      3. L OUTSIDE [R, r] MUST RAISE, and name both bounds. Nothing else checks that.

    'nrandom' DRAWN CONFIGS PER CALL, one L each, and NO PINNED CORNERS: the outer loop is
    what supplies the coverage, since run_all() runs this once per '-n' iteration, so an
    '-n 100' run sees 100 configs and 100 L values. The corner geometries are drawn here at
    measured rates -- _random_config() draws gpu_valid, which is what reaches R = 0 (True
    only) and C_0 == 0 (False only) -- and 'pirate_frb dev coverage' tracks them.
    """

    from .detrender_free import SdPlan, compute_detrender_free_base_map

    # Point 1 of the docstring: the factorization convention, on the per-group matrices. The
    # square groups (D*P == F) are the ones where a transpose slip is silent, and they are
    # emergent rather than demanded of the draw -- 'n_square' is reported below so that a
    # silent loss of that coverage is visible.
    rng = _rng()
    sd_matrices = SdPlan(_random_config(rng)).sd_matrices
    worst_recon, n_square = 0.0, 0
    for sdm in sd_matrices.values():
        recon = sdm.Q_factor @ sdm.W_factor.T                     # (D*P, F)
        assert recon.shape == (sdm.D * sdm.P, sdm.F), (recon.shape, sdm.D, sdm.P, sdm.F)
        n_square += int(sdm.D * sdm.P == sdm.F)
        scale = float(np.abs(sdm.dense_matrix).max())
        if scale > 0.0:
            worst_recon = max(worst_recon,
                              float(np.abs(recon - sdm.dense_matrix.T).max()) / scale)
    assert worst_recon < 1.0e-12, worst_recon

    # Reproducible from the run's printed seed (make_random() draws through
    # ksgpu::default_rng()), but the config is printed on failure anyway, which saves a rerun.
    configs = [(_random_config(rng), 'random') for _ in range(nrandom)]
    n_straddled, kmax = 0, 0
    worst_coarse, worst_sup, worst_cytrue = 0.0, 0.0, 0.0
    n_coarse, n_sliced = 0, 0
    n_opt, svd_frac = 0, 0.0

    for (config, label) in configs:
        # debug=True turns on the O(F) cross-checks: that no SdMatrix gets two rows from one
        # input channel, and that every subband of an entry predicts the same dbits. Both are
        # statements the shared-row pooling depends on, and these configs are small enough to
        # pay for them.
        vmap = compute_detrender_free_base_map(config, debug=True)

        hist = vmap.history[0]
        n_straddled += int(hist['n_straddled'])
        kmax = max(kmax, int(vmap.factor_rank))

        # svd_optimize() is on by default, so the returned rank is the TRUE rank of the
        # assembled factorization and is normally well below the Ktot the lift produced. The
        # sharp check on it is test_svd_optimize(); here we only pin the direction and record
        # what it bought, since that is config-dependent (24% at toy.yml, 54-60% at
        # CHIME/CHORD) and worth seeing in the line below.
        opt = [h for h in vmap.history if h['step'] == 'svd_optimize']
        assert len(opt) <= 1, [h['step'] for h in vmap.history]
        # '<=' not '<': svd_optimize() also rebuilds a factorization that is already full
        # rank but not in SVD form, at the same rank, so that its postcondition (the assert
        # below) holds on every return path. That is the common case on a small draw.
        if opt:
            assert opt[0]['factor_rank'] == vmap.factor_rank <= opt[0]['factor_rank_from']
            n_opt += 1
            svd_frac = max(svd_frac, 1.0 - opt[0]['factor_rank'] / opt[0]['factor_rank_from'])
        assert vmap.Q_is_semiorthogonal and vmap.W_is_semiorthogonal

        # ---- the coarse path, against the fine map's own coarse_grain().
        R, rr = int(vmap.pf_rank), int(vmap.tree_rank)
        for LL in [int(rng.integers(R, rr + 1))]:
            coarse = compute_detrender_free_base_map(config, L=LL)
            ref = vmap.coarse_grain(LL)

            assert coarse.is_coarse_grained and (coarse.L == LL), (LL, coarse.L)
            ec, es = _compare_maps(coarse, ref, f'{label} L={LL}')
            # y_true is FINE for both, and both are lifted from the same untruncated terms by
            # the same code, so they really are the same numbers. Held to a tight bar rather
            # than to exact equality, so that a future change to the accumulation order is not
            # a test failure for no reason.
            ey = float(np.abs(np.asarray(coarse.y_true) / np.asarray(vmap.y_true) - 1.0).max())

            # Two bars, because the two figures have very different tails (see
            # _compare_maps). The sup-norm one is the sharp check and is 1800x inside its bar
            # on the worst random draw measured; the elementwise one is deliberately loose,
            # since its tail is dynamic range and not error. Slicing the dyadic block at the
            # WRONG END -- taking [:, -1, :], a min-envelope -- moves LARGE elements, so the
            # sup-norm bar is what catches it, along with check_ref_covers_y_true() below.
            if (es >= 1.0e-9) or (ec >= 1.0e-6) or (ey >= 1.0e-13):
                atomic_print(f'test_base_varmap_coarse: FAILED on {label} at L={LL},'
                             f' with sup-norm error {es:.3g}, elementwise error {ec:.3g} and'
                             f' y_true error {ey:.3g}. The config'
                             f' was:\n{config.to_yaml_string()}')
                raise AssertionError((label, LL, es, ec, ey))

            # The one runtime check on the property the whole coarse path rests on -- that
            # the map does not UNDERestimate. It is exactly tight here (the group's max is
            # attained at a row the map actually stores), so the returned ratio is 1 to
            # roundoff, and the method's 1 - 1e-9 margin is what absorbs the truncation.
            coarse.check_ref_covers_y_true()

            n_sliced += int(coarse.history[0]['n_sliced'])
            n_coarse += 1
            worst_coarse, worst_sup = max(worst_coarse, ec), max(worst_sup, es)
            worst_cytrue = max(worst_cytrue, ey)

    # L outside [R, r] must RAISE, and name both bounds -- the only check on that, and cheap
    # enough to do on the last config the loop happened to build.
    R, rr = int(vmap.pf_rank), int(vmap.tree_rank)
    for bad in [R - 1, rr + 1]:
        try:
            compute_detrender_free_base_map(config, L=bad)
        except RuntimeError as e:
            assert f'[{R}, {rr}]' in str(e), (bad, str(e))
            continue
        raise AssertionError(f'compute_detrender_free_base_map(L={bad}) should have raised')

    # n_straddled and n_sliced are REPORTED, not asserted. Both are emergent: the straddle
    # branch is 1 row in 645 on toy.yml and a coarse build only slices where some group has
    # n_rm > 0, so neither can be demanded of one config. This test runs per iteration on a
    # drawn config, so the outer loop is what supplies the coverage and 'pirate_frb dev
    # coverage' tracks the rates. A run whose counts are all zero is visible in the line
    # below.

    if verbose:
        atomic_print(f'    test_base_varmap_coarse({len(configs)} random configs):'
                     f' {n_straddled} straddled entries, max K {kmax}'
                     f' ({n_opt} svd-optimized, up to {100*svd_frac:.0f}% off); per-group'
                     f' reconstruction {worst_recon:.3g} over {len(sd_matrices)} groups'
                     f' ({n_square} of them square); {n_coarse} coarse maps'
                     f' ({n_sliced} sliced rows) vs coarse_grain(): sup-norm {worst_sup:.3g},'
                     f' elementwise {worst_coarse:.3g}, y_true {worst_cytrue:.3g}')


def test_multimap_vs_base(nrandom=1, verbose=True):
    """compute_detrender_free_multi_map() against slices of its own base map. No GPU.

    The cheap structural test, and the one that runs on every suite invocation. It compares
    each gamma map against a slice of the BASE map taken here, which makes it an EXACT test
    rather than a tolerance one -- both sides are literally the same floats, and the slice is
    derived from each tree's own geometry rather than from the implementation. It catches an
    off-by-one in the DM half, a missing profile bound, a wrong itree and a dropped W or mid;
    what it cannot catch is whether Proposition 2 is TRUE, which is
    test_multimap_vs_sweep()'s job.

    THE PROFILE RESTRICTION P_gamma < P_0 is what makes the [:, :, :P_gamma, :] slice
    anything other than a no-op, and it needs a config whose primary trees do NOT all share a
    max_width -- exactly the failure notes/variance_map.tex warns about ("the array shapes
    look plausible either way"). make_random() supplies it at 27-38% per draw. The report
    line below says how many actually did, so a thin run is recognizable as thin.

    RUNS AT svd_optimization_level=1, NOT THE DEFAULT 2, and the distinction is the whole
    reason this test can be exact. The slice IS the child map only up to the point where
    level 2 re-optimizes each child, which rebuilds its Q in a rotated basis and gives it its
    own W and mid -- after that, neither the array equality nor the ``m.W is base.W`` object
    identity below means anything. Level 1 is therefore what "the multimap is slices of its
    base" is a statement about. Level 2 is checked separately at the end, where the claim is
    the weaker and correct one: the rank does not go up and A does not change.

    THE LEVEL-2 RANK DROP DOES NOT FIRE ON THESE DRAWS, and the report line says so rather
    than leaving it to be assumed: at _random_config()'s max_toplevel_rank=7 the restricted
    child is essentially always already full-rank, so the count is 0. The drop is a
    production-scale effect -- toy.yml 274 -> 266 and 264, chime_sb1 194 -> 190,
    chime_sb2_et 381 -> 378. So what this arm guards is the DIRECTION (a child's rank must
    never go up) and exactness, not the size of the win, and the counter is what would show
    if a future draw distribution started reaching the interesting case.
    """

    from .detrender_free import compute_detrender_free_multi_map

    # 'nrandom' drawn configs per call; run_all() runs per iteration, so the outer loop is
    # what supplies the coverage.
    rng = _rng()
    configs = []
    for _ in range(nrandom):
        configs.append((_random_config(rng), 'random'))

    n_multi, n_varying, n_coarse, ncmp = 0, 0, 0, 0
    n_opt2, worst_opt2 = 0, 0.0

    for (config, label) in configs:
        npri = int(config.num_primary_trees)
        tree0 = _tree(config, _itree(config, 0))
        r0, R = int(tree0.tree_rank), int(tree0.frequency_subbands.pf_rank)

        # The legal range is the DOWNSAMPLED trees', not the base tree's: [R, r0] at npri == 1
        # but [R, r0-1] otherwise, since a downsampled primary tree has rank r0-1.
        hi = r0 if (npri == 1) else (r0 - 1)
        for L in [None, int(rng.integers(R, hi + 1))]:
            mm = compute_detrender_free_multi_map(config, L=L, svd_optimization_level=1)
            base = mm.maps[0]
            assert len(mm.maps) == npri, (label, len(mm.maps), npri)
            assert mm.provenance['algorithm'] == 'detrender_free'
            assert mm.provenance['L'] == L

            D0, M, N, P0 = 1 << (r0 - R), base.nmultiplets, base.nsubbands, base.nprofiles
            K = base.factor_rank
            nrow0, ax1 = ((1 << (r0 - L)), N) if (L is not None) else (D0, M)
            Q4 = np.asarray(base.Q).reshape(nrow0, ax1, P0, K)
            # y_true is FINE whatever L is, so its view is ALWAYS the (D0, M, P0) one. This is
            # the single easiest thing to get wrong in the function under test.
            y3 = np.asarray(base.y_true).reshape(D0, M, P0)

            n_multi += int(npri > 1)
            n_coarse += int(L is not None)

            for (gamma, m) in enumerate(mm.maps):
                itree = _itree(config, gamma)
                assert m.itree == itree, (label, gamma, m.itree, itree)
                assert m.config is config and m.detrender is None
                assert m.is_admissible == base.is_admissible
                if gamma == 0:
                    continue

                Pg = m.nprofiles
                n_varying += int(Pg != P0)
                tree = _tree(config, itree)
                assert (int(tree.tree_rank), int(tree.frequency_subbands.pf_rank)) \
                    == (r0 - 1, R), (label, gamma)
                nbeta = ((1 << (r0 - 1 - L)) * N * Pg) if (L is not None) \
                    else ((D0 // 2) * M * Pg)
                assert m.shape == (nbeta, m.nfreq), (label, gamma, m.shape, nbeta)

                # EXACT: the slice is a copy of the same floats, so anything but equality is
                # a bug rather than roundoff.
                assert np.array_equal(np.asarray(m.Q),
                                      Q4[nrow0//2:, :, :Pg, :].reshape(-1, K)), (label, gamma)
                assert np.array_equal(np.asarray(m.y_true),
                                      y3[D0//2:, :, :Pg].reshape(-1)), (label, gamma)
                # W and mid are SHARED objects, not copies: identical across trees, stored
                # read-only, and sharing saves npri copies of (nfreq, K).
                assert m.W is base.W and m.mid is base.mid, (label, gamma)
                ncmp += m.Q.size

            # ---- level 2, where the children are re-optimized after the slice. The exact
            # statements above no longer hold; these two do, and they are what level 2
            # actually promises: the rank never goes UP, and A is unchanged.
            if npri > 1:
                mm2 = compute_detrender_free_multi_map(config, L=L,
                                                       svd_optimization_level=2)
                v2 = rng.uniform(0.5, 1.5, size=int(config.get_total_nfreq()))
                for (gamma, (m1, m2)) in enumerate(zip(mm.maps, mm2.maps)):
                    assert m2.factor_rank <= m1.factor_rank, (label, gamma,
                                                              m2.factor_rank, m1.factor_rank)
                    n_opt2 += int(m2.factor_rank < m1.factor_rank)
                    a1, a2 = np.asarray(m1.apply(v2)), np.asarray(m2.apply(v2))
                    e2 = float(np.abs(a2 - a1).max() / np.abs(a1).max())
                    assert e2 < 1.0e-12, (label, gamma, e2)
                    worst_opt2 = max(worst_opt2, e2)

        # L = r0 with npri > 1 must raise, and must do so BEFORE the base map is computed --
        # unchecked it still fails, but from the CHILD's constructor, naming a rank the caller
        # never asked for, after minutes of work.
        #
        # max_bytes=0 is what makes the ORDERING checkable rather than a timing guess: it
        # makes the base map raise too, so getting the L message back proves the L check ran
        # first.
        if npri > 1:
            try:
                compute_detrender_free_multi_map(config, L=r0, max_bytes=0)
            except RuntimeError as e:
                assert f'[{R}, {r0-1}]' in str(e), (label, str(e))
                assert 'max_bytes' not in str(e), (label, 'L was checked after the base map')
            else:
                raise AssertionError(f'{label}: L={r0} should have raised at npri={npri}')

    # Reported, not asserted: see test_varfine() for why. Both are emergent.

    if verbose:
        atomic_print(f'    test_multimap_vs_base({len(configs)} configs, {nrandom} random):'
                     f' {n_multi} multi-tree and {n_coarse} coarse builds,'
                     f' {n_varying} gamma maps with P_gamma < P_0,'
                     f' {ncmp} Q entries compared exactly;'
                     f' at level 2, {n_opt2} maps dropped rank, A unchanged to'
                     f' {worst_opt2:.3g}')


def test_varfine(nrandom=1, verbose=True):
    """detrender_free.compute_detrender_free_varfine(), three ways. No GPU.

    EVERYTHING HERE IS INTERNAL TO detrender_free.py, DELIBERATELY. varfine's comparison
    against something outside it lives in test_multimap_vs_sweep(), which checks it against
    the brute-force sweep -- i.e. against the real kernels -- for every tree. That is the
    correctness check; these are the consistency checks, and they are sharper because both
    sides are float64.

    THREE ASSERTIONS, in increasing order of what they cover and decreasing order of
    sharpness. Each catches something the next one does not:

      1. BITWISE against y_true, every primary tree. varfine(config, ones) must equal
         compute_detrender_free_multi_map(config, L=None).maps[gamma].y_true to the last bit,
         because the weight is a multiplication by an exact 1.0 on a path the two share.
         This is the assertion that fails the moment that path stops being shared, which is
         the specific regression the SdPlan refactor makes possible. It covers the
         Proposition 2 slice too, since the gamma > 0 maps' y_true IS that slice.
      2. Against the DEFINITION, every tree: VarianceMultiMap.apply_fine(). This is the only
         assertion that reaches the early-trigger expansion (Proposition 1). Its bar is set by
         the REFERENCE path's SVD truncation and not by varfine, which is untruncated and
         therefore the more accurate of the two -- measured 3.6e-13 on toy.yml (which has the
         widest dynamic range available) and 1.3e-14 worst over the small configs here.
         Bar 1e-9, so over three decades of margin.

         It is also the assertion that catches an error in the LIFT, which assertion 1
         cannot (both sides of a bitwise comparison move together). Measured: mutating the
         lift's per-subband 2^(R-l) factor fails this on 6 of 6 random draws.
      3. varcoarse against an independently-rebuilt grouping; see below.

    If either bar turns out to be tight in practice that is a signal something is wrong, not
    a reason to loosen it.

    Plus the two SdPlan properties that are SILENT when wrong: that the freq_variances weight
    reaches sd_vectors and never the SdMatrix rows (invisible under the default, where the
    weight is exactly 1.0), and that init_sd_matrices=False rejects Lmat and epsilon rather
    than ignoring them.
    """

    from ..utils import integer_log2
    from .detrender_free import (SdPlan, compute_detrender_free_multi_map,
                                 compute_detrender_free_varcoarse,
                                 compute_detrender_free_varfine)

    rng = _rng()
    configs = []

    # make_random() takes no seed, so a random case that fails is not reproducible unless the
    # config itself is printed. That is the whole cost of admitting randomness here.
    for _ in range(nrandom):
        configs.append((_random_config(rng), 'random'))

    rng = _rng()
    worst_def = 0.0
    n_multi, n_varying, n_et, n_bitwise, ntrees = 0, 0, 0, 0, 0
    n_coarse = 0

    for (config, label) in configs:
        nfreq = int(config.get_total_nfreq())
        npri = int(config.num_primary_trees)
        v = rng.uniform(0.5, 1.5, size=nfreq)

        # debug=True turns on SdPlan's planning-pass cross-checks, which is all of them in
        # this mode -- there is no SdMatrix here to check a capacity or a duplicate row
        # against. These configs are small enough to pay for them.
        got = compute_detrender_free_varfine(config, v, debug=True)
        assert len(got) == _ntrees(config), (label, len(got))

        # The returned arrays are ordinary writeable ones and MUST NOT ALIAS EACH OTHER, or a
        # caller mutating one tree's result silently corrupts another's. The gamma > 0 slices
        # of the base tree's vector are already contiguous whenever P_gamma == P_0 -- which is
        # exactly the case np.ascontiguousarray() would NOT copy -- so this is the only thing
        # standing between an ordinary-looking line and a shared buffer.
        assert all(g.flags.writeable for g in got), label
        for i in range(len(got)):
            for j in range(i + 1, len(got)):
                assert not np.shares_memory(got[i], got[j]), (label, i, j)

        mm = compute_detrender_free_multi_map(config, L=None)
        P = [m.nprofiles for m in mm.maps]
        n_multi += int(npri > 1)
        n_varying += int(len(set(P)) > 1)
        n_et += int(any(int(pt.num_early_triggers) > 0 for pt in config.primary_trees))

        # ---- (1) bitwise against y_true, every primary tree.
        ones = compute_detrender_free_varfine(config, np.ones(nfreq))
        for gamma in range(npri):
            it = _itree(config, gamma)
            if not np.array_equal(ones[it].reshape(-1), np.asarray(mm.maps[gamma].y_true)):
                atomic_print(f'test_varfine: FAILED on {label}: varfine(ones) is not BITWISE'
                             f' equal to primary tree {gamma}\'s y_true, so the two no longer'
                             f' share an accumulation path. The config'
                             f' was:\n{config.to_yaml_string()}')
                raise AssertionError((label, gamma))
            n_bitwise += 1

        # ---- (2) the definition, every tree. Sup-norm relative, for the reason
        # _compare_maps() documents: an elementwise ratio has no floor and its tail is the
        # map's dynamic range rather than its error.
        ref = mm.apply_fine(v)
        e3 = 0.0
        for (itree, (g, w)) in enumerate(zip(got, ref)):
            assert g.shape == w.shape, (label, itree, g.shape, w.shape)
            scale = float(np.abs(w).max())
            assert scale > 0.0, (label, itree)      # a tree with no variance at all
            e3 = max(e3, float(np.abs(g - w).max()) / scale)
            ntrees += 1

        if e3 >= 1.0e-9:
            atomic_print(f'test_varfine: FAILED on {label}, with definition error'
                         f' {e3:.3g}. The config was:\n{config.to_yaml_string()}')
            raise AssertionError((label, e3))

        # ---- (3) compute_detrender_free_varcoarse(), which is varfine coarse-grained at the
        # WEIGHT array's downsampling. Only the glue is new here -- varfine is checked three
        # ways above and coarse_grain_vector() against an independent label oracle in
        # test_index_arithmetic() -- so this checks exactly the two things the glue can get
        # wrong, and does it without re-running the function.
        #
        # NOT by calling coarse_grain_vector() again, which would be circular. The grouping
        # is rebuilt the long way by _child_group_labels(), from each row's FULL-RESOLUTION
        # DM, and the max is taken here. EXACT equality is the right bar: max is exact, so
        # the reduction order cannot matter.
        #
        # MEASURED TEETH, since two of the obvious mutations self-detect and it is worth
        # saying which assertion earns its place. Mutating varcoarse:
        #   - transpose the output to (ndm_wt, P, N)  -> caught by the SHAPE check below.
        #   - coarse-grain with tree 0's geometry     -> coarse_grain_vector() raises on the
        #                                                length itself; not this test.
        #   - L off by one                            -> the function's own ndm_wt assert.
        #   - flatten as y.transpose(0,2,1).reshape() -> shapes IDENTICAL, values permuted,
        #                                                caught ONLY by the max check above.
        # The last is why the max check is here rather than just a shape assertion.
        #
        # THE OTHER HALF IS THE SHAPE, and it is not cosmetic. L comes from
        # tree.primary_tree.wt_dm_downsampling -- a different quantity from any L a caller
        # passes -- and a transposed reshape changes values without changing the total size.
        # Reading ndm_wt/N/P back from the TREE is what makes this a check rather than a
        # restatement: the tree is where the consumer of this array gets its own shape.
        coarse = compute_detrender_free_varcoarse(config, v)
        assert len(coarse) == len(got), (label, len(coarse), len(got))
        for itree in range(len(got)):
            tree = _tree(config, itree)
            fs = tree.frequency_subbands
            L = integer_log2(int(tree.primary_tree.wt_dm_downsampling))
            want_shape = (int(tree.ndm_wt), int(fs.N), int(tree.nprofiles))
            assert coarse[itree].shape == want_shape, (label, itree, coarse[itree].shape,
                                                       want_shape)

            labels = _child_group_labels(tree, L)
            nbeta = int(labels.max()) + 1
            want = np.full(nbeta, -np.inf)
            np.maximum.at(want, labels, got[itree].reshape(-1))
            assert np.array_equal(coarse[itree].reshape(-1), want), (label, itree)
            n_coarse += 1

        worst_def = max(worst_def, e3)

    # n_varying and n_et are REPORTED, not asserted. Both are emergent properties of
    # make_random() (a varying max_width at 27-38% per draw, an early trigger at 30-38%), so
    # they cannot be demanded of any single config; 'pirate_frb dev coverage' tracks the rates,
    # and the counts below make a thin run recognizable as thin.

    # ---- THE WEIGHT MUST NOT REACH THE SdMatrix ROWS. This is the one failure mode the
    # default v = ones hides completely: with the weight folded into the rows as well, every
    # assertion above still passes, and the FACTORIZATION -- whose W factor is indexed by
    # input channel -- is silently of A diag(v) rather than of A.
    config = _random_config(rng)
    nfreq = int(config.get_total_nfreq())          # NOT the loop's last config
    v = rng.uniform(0.5, 1.5, size=nfreq)
    plain, weighted = SdPlan(config), SdPlan(config, freq_variances=v)
    assert set(plain.sd_matrices) == set(weighted.sd_matrices)
    for (key, sdm) in plain.sd_matrices.items():
        assert np.array_equal(sdm.dense_matrix, weighted.sd_matrices[key].dense_matrix), key
        assert np.array_equal(sdm.Q_factor, weighted.sd_matrices[key].Q_factor), key
    # ... and it must reach sd_vectors, or the whole function is a no-op.
    assert not np.array_equal(plain.lift_sd_vectors(), weighted.lift_sd_vectors())

    # ---- the illegal-argument guard. Lmat and epsilon act on sd_matrices and on nothing
    # else, so with init_sd_matrices=False there is nothing for either to do; accepting them
    # silently would hand back a fine, untruncated result to a caller who asked otherwise.
    # Lmat drawn from the config's own legal range: a literal can be outside [R, r], and then
    # SdPlan raises for that reason instead of the one under test.
    _t = _tree(config, _itree(config, 0))
    _Lmat = int(_t.frequency_subbands.pf_rank)
    for kwargs in [dict(Lmat=_Lmat), dict(epsilon=1.0e-9)]:
        try:
            SdPlan(config, init_sd_matrices=False, **kwargs)
        except RuntimeError as e:
            assert 'init_sd_matrices' in str(e), (kwargs, str(e))
            continue
        raise AssertionError(f'SdPlan(init_sd_matrices=False, {kwargs}) should have raised')

    # ---- a wrong-length freq_variances must raise, and name the length it wanted. Unlike
    # test_multimap_vs_base()'s L-ordering check there is no way to prove this ran BEFORE the
    # tile pass (max_bytes=0 has no analogue here -- varfine allocates nothing worth a
    # ceiling); the exception type and message are what is available. Without the check a
    # short vector surfaces as an IndexError from inside the tile pass instead.
    for bad in [nfreq - 1, nfreq + 1]:
        try:
            compute_detrender_free_varfine(config, np.ones(bad))
        except RuntimeError as e:
            assert f'({nfreq},)' in str(e), (bad, str(e))
            continue
        raise AssertionError(f'compute_detrender_free_varfine() accepted a length-{bad}'
                             ' freq_variances')

    if verbose:
        atomic_print(f'    test_varfine({len(configs)} configs, {nrandom} random,'
                     f' {ntrees} trees): {n_multi} with npri > 1, {n_varying} with a varying'
                     f' max_width, {n_et} with early triggers; {n_bitwise} primary trees'
                     f' bitwise equal to y_true; {n_coarse} varcoarse arrays exactly equal'
                     f' to an independently-grouped max; worst relative error vs'
                     f' apply_fine() {worst_def:.3g}')


# Work-unit ceiling for the randomized part of test_multimap_vs_sweep(). The budget is in
# WORK UNITS, not seconds, so that the same draws are accepted on every machine and a slow
# machine runs the same test rather than a different one. The measured conversion is about
# 2.0 ns per unit for the GPU sweep (72 ns for the CPU one), so this is a ~5-second ceiling
# per accepted config on the GPU.
#
# A budget is NOT optional here: over unfiltered draws the predicted sweep cost has a mean 17x
# its median, so an unbudgeted five-draw loop averages tens of minutes with a multi-hour tail.
# Rejection sampling is free by comparison -- drawing a config and building its _SweepGeometry
# costs well under a millisecond.
#
# And the budget costs almost nothing in coverage, for a structural reason: sweep cost scales
# with nfreq, while geometric depth scales with 2^r / nfreq. The wide footprints that produce
# deep dbits and straddles come from having FEW input channels per tree-freq, which is also
# the cheap direction for the sweep. Budgeting throws away the big, SHALLOW configs.
SWEEP_WORK_BUDGET = 2.5e9


def _sweep_work(geom):
    """Predicted sweep cost of '_SweepGeometry' geom, in the work units of SWEEP_WORK_BUDGET."""
    return (geom.nfreq * geom.nphases * (geom.ndata_chunks * geom.nt_in)
            * sum(1 << r for r in geom.tree_r))


def test_multimap_vs_sweep(device='gpu', nrandom=5, verbose=True):
    """compute_detrender_free_multi_map() against the brute-force sweep, EVERY primary tree.
    Needs a plan, and by default a GPU.

    THE ONLY NUMERICAL CHECK ON THE ANALYTIC MAP AGAINST ANYTHING OUTSIDE
    detrender_free.py, and the only one that ties it to the actual kernels. The sweep is the
    right oracle for that job precisely because it shares no code with detrender_free.py, and
    because it can be drawn wide (max_toplevel_rank=9, against _random_config()'s 7).

    THE 1e-5 BAR IS WIDE ENOUGH TO BE THE ONLY BAR, and it was measured rather than assumed.
    Deliberately breaking detrender_free.py, on a config with r=8, R=2, subbands (2,2,1),
    npri=3 and one straddled entry: scaling the straddled rows by 1.001 --
    the ~0.1% that pirate_frb/tests/coverage.py attributes to a port that never takes the
    half-aligned branch -- moves this comparison to 1.0e-3, 100x the bar; dropping _emit()'s
    scale**2 moves it to 255; perturbing ONE input channel of 575 by 1e-4 moves it to 1.0e-4.
    The floor is between 1e-6 and 1e-4 on a single-channel error, and nothing structural
    lives in that band. If the bars here are ever revisited, redo that measurement.

    It is also the only check on Proposition 2 of notes/variance_map.tex against the sweep:
    test_restriction_vs_sweep() is Proposition 1 (early triggers) only, and says so. And
    mm.maps[0] IS compute_detrender_free_base_map(config), so the base map is covered by the
    same assertion as the rest.

    Covering every primary tree rather than the base tree alone is nearly free:
    sweep_all_trees_dense() computes every tree anyway -- they share one dedisperser -- so the
    per-tree matrices are already sitting there.

    'device' selects which sweep is checked. The GPU sweep is the default because it is 36x
    faster, which is what lets the cost budget be large enough for the random sample to be
    much less biased. device='cpu' still works and needs none of the make_random() flags below
    -- keep it reachable, since it is the fallback if the GPU sweep is ever the thing under
    suspicion.

    Tolerance 1e-5, because the dedispersion chain is float32. Measured over nine runs of this test, the worst agreement
    across every primary tree ranged over 4.9e-7 to 1.3e-6, so there is roughly an order of
    headroom -- and the spread is the random draws, not the restriction: a hand-built config
    reproduces at 4.8e-7 every time.

    RANDOM CONFIGS ONLY, and no pinned ones: make_random() reaches every geometry a pinned
    config would, including a primary-tree-dependent max_width. Measured over 250-400 draws
    from this test's exact distribution: npri > 1 in 53%, a varying max_width (hence
    P_gamma < P_0) in 23-29%, R == 0 in 39%, nfreq < 2^r in 27%, K > 0 (dm_downsampling >
    2^R) in 92%, and some primary tree with early triggers in 25%.
    The sampling is deliberately left non-deterministic: a different config covers each case
    on each run, which is worth more over many runs than one config pinned forever.
    'nrandom' is the knob if a single run's coverage ever needs to be denser.

    ONE GAP, AND IT IS PRE-EXISTING. At max_toplevel_rank=9 the drawn R never exceeds 2
    (measured histogram {0: 97, 1: 97, 2: 56} over 250 draws), because R <= dd_rank1 and a
    small tree caps it. So the R = 3, 4 geometry of every shipped config is not swept here.
    That is a property of the whole sweep tier -- nothing in it runs at production rank -- not
    of the random draws.

    IT ALSO CHECKS PROPOSITION 1, on the early-trigger trees the sweep computes anyway. That
    is nearly free here -- sweep_all_trees_dense() has already built every tree's matrix --
    where a separate test would have to run its own sweep for the same data.

    NEITHER SIDE OF THAT CHECK GOES THROUGH THE VARMAP, and that is the whole point.
    Row-restriction is hardwired into how the detrender-free varmap is built, so a check
    routed through it would be tautological. Here the child's matrix comes from the sweep
    running the real dedispersion chain AT THE CHILD'S OWN RANK, against a row subset of the
    parent's, with the multiplet map rebuilt from toplevel band ranges rather than imported.
    The two sides are therefore computed by different trees, different subband tables and
    different ReferenceTree / ReferencePfSquare instances -- which is what makes it a test of
    the PROPOSITION rather than of the plumbing. Everything else that touches an
    early-trigger map derives it by restriction and would be self-consistently wrong if the
    trees ever stopped nesting.

    THE BAR THERE IS 1e-5, THE SAME AS ABOVE, AND IT IS NOT SLACK. On a config with ONE
    primary tree the two sides come out bit-identical or within an ulp of float64 (measured
    7 of 10 multiplet comparisons exactly equal, worst 1.78e-16 at r=6, subbands (4,2,1)),
    which invites a much tighter bar. That is a property of npri == 1 only. A DOWNSAMPLED
    primary tree reaches its early-trigger child through a different float32 path, and there
    the difference is ~1e-7 -- measured 1.73e-7 at gamma=1 on a random draw, which is what a
    1e-12 bar fails on. Loose enough for that, tight enough that a wrong row map (which
    mismatches whole bands) cannot pass.
    """

    from ..pirate_pybind11 import DedispersionConfig
    from .brute_force import sweep_all_trees_dense, _SweepGeometry
    from .detrender_free import (compute_detrender_free_multi_map,
                                 compute_detrender_free_varfine)

    configs = []
    rng = _rng()

    # Random draws, cost-bounded. Four details here cost time to rediscover:
    #
    #  - LEAVE gpu_valid AT ITS DEFAULT True. _GpuSweep has a third requirement the two flags
    #    below do not cover -- every tree needs stage-2 dd_rank >= 3 -- and a config drawn from
    #    the cdd2 registry satisfies it for free. With gpu_valid=False it fails on 80% of draws.
    #  - The two make_random() flags are what make the GPU sweep usable on a random config at
    #    all. Without them only about 40% of draws get through _GpuSweep (blocked by float16
    #    and by the MegaRingbuf host-segment count), and filtering the sample down to the
    #    configs it happens to accept would bias it toward exactly the geometry already best
    #    covered elsewhere. With them, 60 of 60 draws were usable.
    #  - The beam fields, the redraw-whole rule and why this is NOT single_beam=True are all
    #    in _draw_sweep_case(). Measured acceptance here: 0.86, at 1.9 ms per attempt.
    for _ in range(nrandom):
        config, _dp, _n = _draw_sweep_case(
            rng,
            lambda r: DedispersionConfig.make_random(max_toplevel_rank=9, max_early_triggers=2,
                                                     force_float32=(device == 'gpu'),
                                                     no_host_mega_ringbuf=(device == 'gpu')),
            lambda geom, cfg: _sweep_work(geom), SWEEP_WORK_BUDGET,
            label='test_multimap_vs_sweep')
        configs.append((config, 'random'))

    worst, n_straddled, ntrees = 0.0, 0, 0
    n_multi, n_varying, n_r0, n_wide, n_et = 0, 0, 0, 0, 0
    worst_p1, n_p1_pairs, n_nontrivial = 0.0, 0, 0
    worst_vf, n_vf = 0.0, 0
    worst_leak = 0.0

    for (config, label) in configs:
        As = sweep_all_trees_dense(config, device=device)
        mm = compute_detrender_free_multi_map(config)

        npri = int(config.num_primary_trees)
        P = [m.nprofiles for m in mm.maps]
        n_multi += int(npri > 1)
        n_varying += int(len(set(P)) > 1)
        n_r0 += int(mm.maps[0].pf_rank == 0)
        n_wide += int(int(config.get_total_nfreq()) < (1 << mm.maps[0].tree_rank))
        n_et += int(any(int(pt.num_early_triggers) > 0 for pt in config.primary_trees))

        for (gamma, vmap) in enumerate(mm.maps):
            A = np.asarray(As[_itree(config, gamma)])
            # force=True: dense() guards against forming a production-scale matrix, but the
            # sweep above has already materialized one of exactly this shape for EVERY tree,
            # so the budget that matters was spent long before this line.
            got = np.asarray(vmap.dense(force=True))
            assert got.shape == A.shape, (label, gamma, got.shape, A.shape)

            # The sweep's exact zeros are (channel, multiplet) pairs that do not overlap at
            # all. This bar is what catches a W scatter landing on the wrong channels, which
            # a relative comparison over the support would mostly hide.
            #
            # A BAR RATHER THAN AN EXACT ZERO, and the structural argument for an exact zero
            # is the thing to resist. Before any rotation the product really is 0.0 there: a
            # channel absent from a group leaves that group's rows of W at zero, and a subband
            # absent from a group never has those columns written into its rows of Q, so
            # either missing factor gives exactly 0.0 (measured over 12.2M zero entries across
            # 20 random configs). svd_optimize() ENDS THAT -- it rotates the factorization
            # into its true rank, so every column of the new W is a combination of all the old
            # ones and the block-sparsity is gone, leaving zeros that hold only to float64
            # roundoff. Measured over 4 random configs at svd_optimization_level=2: worst
            # leakage 3.35e-14 relative to max|A|, against exactly 0.0 at level 0. The bar
            # below is 1e-10 of max|A| -- four decades over what was measured, and still tiny
            # against the O(1) relative leak a misdirected scatter would produce.
            #
            # The varfine arm further down DOES hold an exact-zero bar: it does not go through
            # the map, so the structural argument still applies there.
            nz = (A != 0.0)
            assert np.any(nz), (label, gamma)
            scale = float(np.abs(A).max())

            # THE DENOMINATOR IS FLOORED AT 1e-2 OF max|A|, and it has to be. A plain relative
            # comparison over every nonzero entry is not well posed against a float32 sweep:
            # A spans several decades within one tree, and an entry at 1e-7 of the maximum is
            # not known to any relative precision at all. Measured on the draw that first
            # failed this bar: worst relative error 4.9e-05, at an entry with A = 1.1e-07 of
            # max, where the two maps differed by 8.3e-13 OF max|A| -- i.e. the bar was
            # measuring the sweep's inability to resolve that entry, not a disagreement.
            #
            # So the score is |got - A| / max(|A|, 1e-2*max|A|): a true relative error on the
            # entries that carry the matrix, and an absolute one in units of max|A| below
            # that. It costs no teeth -- a misdirected scatter moves an entry by O(max|A|),
            # which is 1e2 in these units, and even wholly losing an entry at 1e-4 of max
            # scores 1e-2 against a 1e-5 bar. It touches few entries: measured over 22 trees,
            # 0.2% of nonzero entries lie below the floor (worst tree 0.9%). Measured worst
            # score over those trees is 6.7e-07, so the bar keeps ~15x headroom.
            denom = np.maximum(np.abs(A[nz]), 1.0e-2 * scale)
            e = float((np.abs(got[nz] - A[nz]) / denom).max())
            e0 = (float(np.abs(got[~nz]).max()) / scale) if np.any(~nz) else 0.0

            if (e >= 1.0e-5) or (e0 >= 1.0e-10):
                atomic_print(f'test_multimap_vs_sweep: FAILED on {label}, primary tree'
                             f' {gamma} of {npri} (P={P}), with on-support error {e:.3g} and'
                             f' off-support leakage {e0:.3g} (relative to max|A|).'
                             f' The config was:\n{config.to_yaml_string()}')
                raise AssertionError((label, gamma, e, e0))

            worst = max(worst, e)
            worst_leak = max(worst_leak, e0)
            ntrees += 1

        n_straddled += int(mm.maps[0].history[0]['n_straddled'])

        # ---- compute_detrender_free_varfine() against the sweep, EVERY tree.
        #
        # varfine is what PRODUCTION computes: GpuDedisperser::_fill_analytic_weights() calls
        # the C++ port of varcoarse, which is varfine plus a max-reduction. It reaches the
        # same numbers by a different route from the map above -- no SdMatrix, no SVD, no
        # truncation -- and test_varfine() only ever compares it against y_true and
        # apply_fine(), both of which are inside detrender_free.py. So this is its one
        # comparison against ground truth, and it costs a matvec against a matrix the sweep
        # has already materialized.
        #
        # NOT A SUBSTITUTE FOR THE MAP COMPARISON ABOVE, which is per-channel. This one
        # compares A v, so an error confined to one input channel is diluted by that
        # channel's share of the sum: measured, perturbing 1 channel of 575 by 1e-4 moves the
        # map comparison by 1.0e-4 and this one by 3.7e-7. What this catches is an error in
        # the LIFT, which is not per-channel -- measured, mutating the lift's per-subband
        # 2^(R-l) factor moves this by a factor of 3 and the map comparison not at all.
        #
        # 'v' is drawn rather than all-ones so that a weight reaching the wrong rows is
        # visible. Same 1e-5 / exact-zero bars as the map comparison, and the same reason.
        v = rng.uniform(0.5, 1.5, size=int(config.get_total_nfreq()))
        yf = compute_detrender_free_varfine(config, v)
        for itree in range(_ntrees(config)):
            A = np.asarray(As[itree])
            want = A @ v
            got = np.asarray(yf[itree]).reshape(-1)
            assert got.shape == want.shape, (label, itree, got.shape, want.shape)
            nz = (want != 0.0)
            assert np.any(nz), (label, itree)
            # Same floored denominator as the map arm above, for the same reason. This arm
            # has not been seen to fail on a tiny entry -- it compares A v, whose entries are
            # sums over channels and are far better conditioned (measured smallest |want| is
            # 2.1e-05 of max, against 1.1e-07 for a single matrix entry) -- but the hazard is
            # identical in kind and the guard costs nothing here either.
            sc = float(np.abs(want).max())
            e = float((np.abs(got[nz] - want[nz])
                       / np.maximum(np.abs(want[nz]), 1.0e-2 * sc)).max())
            e0 = float(np.abs(got[~nz]).max()) if np.any(~nz) else 0.0
            if (e >= 1.0e-5) or (e0 != 0.0):
                atomic_print(f'test_multimap_vs_sweep: FAILED varfine on {label}, tree'
                             f' {itree} of {_ntrees(config)}, with'
                             f' on-support error {e:.3g} and off-support leakage {e0:.3g}'
                             f' (which must be exactly zero). The config'
                             f' was:\n{config.to_yaml_string()}')
                raise AssertionError((label, itree, e, e0))
            worst_vf = max(worst_vf, e)
            n_vf += 1

        # ---- Proposition 1, sweep against sweep. See the docstring.
        # The row map comes from _restriction_pairs(), which rebuilds it from toplevel band
        # ranges; test_restriction_vs_sweep() checks the same proposition on a config drawn
        # to GUARANTEE a non-contiguous map, which the five draws here reach about 45% of
        # the time.
        p1_trees = make_plan(config).trees
        for (gamma, e, iparent, ichild, m_map) in _restriction_pairs(config):
            parent, child = p1_trees[iparent], p1_trees[ichild]
            fsp, fsc = parent.frequency_subbands, child.frequency_subbands
            D = 1 << (int(parent.tree_rank) - int(fsp.pf_rank))
            M_p, M_c, P = int(fsp.M), int(fsc.M), int(parent.nprofiles)
            nf = As[iparent].shape[1]

            assert (1 << (int(child.tree_rank) - int(fsc.pf_rank))) == D
            assert int(child.nprofiles) == P

            Ap = np.asarray(As[iparent]).reshape(D, M_p, P, nf)
            Ac = np.asarray(As[ichild]).reshape(D, M_c, P, nf)
            n_nontrivial += int(m_map != list(range(M_c)))

            for mc in range(M_c):
                got, want = Ac[:, mc], Ap[:, m_map[mc]]
                scale = max(float(np.max(np.abs(want))), 1e-300)
                d = float(np.max(np.abs(got - want))) / scale
                if d >= 1.0e-5:
                    atomic_print(f'test_multimap_vs_sweep: FAILED Proposition 1 on'
                                 f' {label}, (gamma, e, m_child) = ({gamma}, {e}, {mc}):'
                                 f' relative difference {d:.3g}. The config'
                                 f' was:\n{config.to_yaml_string()}')
                    raise AssertionError((label, gamma, e, mc, d))
                worst_p1 = max(worst_p1, d)
            n_p1_pairs += 1

    if verbose:
        atomic_print(f'    test_multimap_vs_sweep(device={device}, {len(configs)} random'
                     f' configs, {ntrees} primary trees): {n_straddled} straddled entries,'
                     f' worst floored-relative error {worst:.3g} (off-support leak'
                     f' {worst_leak:.3g}); varfine over {n_vf} trees, worst'
                     f' {worst_vf:.3g}; Proposition 1 over {n_p1_pairs}'
                     f' (parent, child) pairs ({n_nontrivial} with a non-contiguous multiplet'
                     f' map), worst {worst_p1:.3g}; coverage: {n_multi} npri>1,'
                     f' {n_varying} varying max_width, {n_r0} R=0, {n_wide} wide-footprint,'
                     f' {n_et} with early triggers')


####################################   the brute-force sweep   ###################################
#
# These need a DedispersionPlan and (for the GPU sweep) a device, which the rest of this file
# deliberately does not. Each one runs at least one full sweep over every input channel, which
# puts the tier at about 30 s, so they do NOT all run at the same cadence: run_tests() and
# run_once() are where the per-iteration / every-tenth / once split is decided.


def _draw_sweep_case(rng, draw_config, cost, budget, *, detrender=False, accept=None,
                     max_attempts=500, label='_draw_sweep_case'):
    """Draw a (config, detrender) pair for one of the sweep tests, under a cost cap.

    THE FOUR SWEEP TESTS ALL NEED THE SAME LOOP. What varies between them is genuinely
    per-test -- the make_random() settings, the cost model, the budget, whether there is a
    detrender, and what structural property the test needs -- so those are arguments; the
    loop, the beam-field boilerplate, the RuntimeError handling and the exhaustion report are
    not, and live here.

    Returns (config, dparams, nattempts). RAISES on exhaustion rather than looping, with the
    implied acceptance bound in the message: a filter that has quietly gone to zero is worth
    a message rather than a hang.

    Arguments:

      draw_config(rng) -> config.  Each caller's own make_random() settings; see each.

      cost(geom, config) -> number, compared against 'budget'.  Evaluated on the geometry
        THIS FUNCTION built, with the detrender it drew, so the number the budget sees is the
        one the test will pay.

      detrender: False = never build one, True = always, None = coin flip.  When one is built
        its own parameters are drawn too (see _make_test_detrender).

      accept(config) -> bool, optional.  A structural property make_random() cannot be asked
        for -- an early trigger, a non-contiguous multiplet map -- checked after the beam
        fields are set and before the (more expensive) geometry is built.

    ALL THREE BEAM FIELDS ARE SET TO 1 HERE.  The sweep requires beams_per_gpu ==
    beams_per_batch, and setting only those two leaves num_active_batches at its drawn value,
    which fails a C++ assertion inside validate() -- measured, on 55% of draws.  Note this is
    NOT make_random(single_beam=True): that flag does the same three assignments but also
    gives the whole chunk budget to time_samples_per_chunk, which takes the median from 256 to
    ~4000, P(ndata_chunks == 1) from 0.28 to 0.89, and the predicted sweep work up 4x.
    Streaming across chunk boundaries is most of what these comparisons check, so the short
    chunk is the corner worth keeping.

    Rejection is REDRAW-WHOLE, not downscale: an over-budget config is discarded outright.
    """
    from .brute_force import _SweepGeometry

    for nattempt in range(1, int(max_attempts) + 1):
        config = draw_config(rng)
        config.beams_per_gpu = 1
        config.beams_per_batch = 1
        config.num_active_batches = 1
        config.validate()

        if (accept is not None) and not accept(config):
            continue

        want_det = (rng.random() < 0.5) if (detrender is None) else bool(detrender)
        try:
            dparams = _make_test_detrender(config, rng=rng) if want_det else None
            geom = _SweepGeometry(config, detrender=dparams)
        except RuntimeError:
            continue      # e.g. nt_in too small to hold the detrended one-hot in one chunk

        if cost(geom, config) <= budget:
            return config, dparams, nattempt

    raise RuntimeError(f'{label}: no draw met the budget in {max_attempts} attempts '
                       f'(acceptance rate is below {1.0/max_attempts:.1e}; the filters or the '
                       f'budget have drifted)')


def _make_test_detrender(config, n_phi=2, n=2, W=4, nzone=2, kint=3, rng=None):
    """A Detrender2dParams matching 'config', for the sweep tests.

    'nzone' and 'kint' are REQUESTS, not requirements. zoned_knots() needs nzone to divide
    nfreq and each zone to be wide enough to hold kint interior knots, and a random config
    satisfies neither in general -- its nfreq is odd about half the time. Both are reduced to
    the largest legal value rather than raising, so that a caller which does not care about
    the exact knot layout (every caller here does not) can hand this an arbitrary config.
    Pinning them instead is what zoned_knots() is for, and its docstring says so.
    """

    from ..pirate_pybind11 import Detrender2dParams
    from ..detrending_spline.masks import zoned_knots

    # 'rng' randomizes the detrender itself, not just the config it is matched to. The
    # ranges are the ones Detrender2dParams accepts and that the sweep can afford: W is the
    # half-width in time samples and drives the polyphase pass count, so it stays small.
    #
    # W = 0 IS REACHED ON PURPOSE, and is not just the small end of a range: _SweepGeometry
    # keys the entire polyphase shortcut off W > 0, so W = 0 is the other branch of that
    # decision and nothing drew it before. It forces n = 0 with it, since a degree-n fit
    # needs 2W+1 >= n+1 samples in the window.
    if rng is not None:
        n_phi = int(rng.integers(0, 3))
        W = 0 if (rng.random() < 0.2) else int(rng.integers(1, 6))
        n = 0 if (W == 0) else int(rng.integers(0, 3))   # Detrender2d requires n in [0, 2]
        nzone = int(rng.integers(1, 4))
        kint = int(rng.integers(1, 5))

    nfreq = int(config.get_total_nfreq())
    kint = min(kint, max(nfreq - 1, 0))
    nzone = max((z for z in range(1, nzone + 1)
                 if (nfreq % z == 0) and (nfreq // z >= kint + 1)), default=1)
    kv = zoned_knots(n_phi, nfreq, nzone, kint)

    # M IS THE BEAM COUNT and must equal the config's beams_per_batch -- _SweepGeometry
    # checks it. Hardcoding 1 was fine while every caller used _make_test_config(), which
    # sets all three beam fields to 1; a drawn config does not.
    return Detrender2dParams(nfreq=nfreq, knots=[int(x) for x in kv.knots],
                             M=int(config.beams_per_batch), n_phi=n_phi,
                             n=n, W=W, T=int(config.time_samples_per_chunk))


def _abcd_all(config, As):
    """sweep_all_trees_dense()'s output as the (2^(r-R), M, P, nfreq) array the analytic
    references are indexed by, keyed by itree. The sweep's own (nalpha, nfreq) layout is that
    array with its first three axes flattened, so this is a reshape and not a transpose.

    A VarianceMultiMap holds only the primary trees, so a test which wants to check EVERY
    tree against a per-tree oracle sweeps raw arrays rather than going through it.
    """

    out = []
    for (itree, A) in enumerate(As):
        tree = _tree(config, itree)
        fs = tree.frequency_subbands
        D = 1 << (int(tree.tree_rank) - int(fs.pf_rank))
        out.append(np.asarray(A).reshape(D, int(fs.M), int(tree.nprofiles), A.shape[1]))
    return out


def test_sweep_phase_collapse(r=7, verbose=True):
    """With no detrender, the 2^gamma polyphase passes of a time-downsampled tree must give
    the same result (notes/variance_map.tex: everything upstream of the downsampler is
    instantaneous in time). This is the sharpest available test of the polyphase logic, and of
    the single-pass shortcut the sweep takes when there is no detrender.

    Agreement is not bit-exact, even though the float32 output samples themselves are:
    shifting the one-hot moves the response relative to the chunk boundaries, so the same set
    of squared samples is accumulated in a different order. The tolerance below is still six
    orders of magnitude below the float32 noise floor of the dedispersion chain.
    """

    from .brute_force import _CpuSweep, _SweepGeometry

    # Three primary trees => gamma = 0, 1, 2, so the phase loop has something to collapse.
    geom = _SweepGeometry(_make_test_config(r, [2, 2, 1], num_primary_trees=3))
    assert geom.gamma_max == 2, geom.gamma_max

    sweep = _CpuSweep(geom)
    nphases = 1 << geom.gamma_max
    chain = sweep.make_chain()
    worst = 0.0

    for (ipass, ifreq) in enumerate([0, geom.nfreq // 3, geom.nfreq - 1]):
        ref = None
        for iphase in range(nphases):
            acc = sweep.run_pass(chain, ifreq, iphase, ipass*nphases + iphase)
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


# Cost cap for the randomized test_sweep_column_norms() draw. Its cost model is NOT
# _sweep_work(): that prices one sweep, while this test runs (ntime + nt_in) passes per
# channel -- more than a whole sweep -- so it needs its own.
#
# The proxy is (ntime + nt_in) * (ndata_chunks + 3) * sum(2^r over trees): passes, times
# chunks per pass, times the per-chunk chain work. Measured about 30 us per unit
# (0.83 s at 3.6e4, 1.28 s at 4.4e4, 40.7 s at 8.6e5), so this cap is a ~3 s ceiling.
# Measured acceptance 35%, i.e. ~3 draws per case, and a draw costs well under a millisecond.
COLUMN_NORMS_BUDGET = 1.0e5


def test_sweep_column_norms_random(verbose=True, max_attempts=500):
    """test_sweep_column_norms() on a random config and a random detrender, under a cost cap.

    ONE DRAWN GEOMETRY PER INVOCATION, rather than a pinned one: the core identity this test
    checks is a statement about every tree shape, and evaluating it at one fixed shape forever
    says nothing about the others. Run once every ten iterations, since at ~3 s it is too
    expensive for every one.

    THE DETRENDER IS A COIN FLIP, and when present its own parameters are drawn too. Half the
    draws exercise the Detrender2d path -- the only independent check on it, since no
    analytic oracle can represent a detrender -- and half exercise the plain chain. Over an
    '-n 100' run that is about five of each, on ten different geometries.
    """

    rng = _rng()

    def _cost(geom, config):
        return ((geom.ntime + geom.nt_in) * (geom.ndata_chunks + 3)
                * sum(1 << r for r in geom.tree_r))

    # detrender=None is the coin flip. Measured acceptance 0.34, at 1.2 ms per attempt.
    config, dparams, _n = _draw_sweep_case(
        rng, _random_config, _cost, COLUMN_NORMS_BUDGET, detrender=None,
        max_attempts=max_attempts, label='test_sweep_column_norms_random')

    test_sweep_column_norms(config=config, detrender=dparams, nifreq=1, verbose=verbose)


def test_sweep_column_norms(config, detrender=None, nifreq=2, verbose=True):
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

    from .brute_force import _CpuSweep, _SweepGeometry, sweep_all_trees_dense

    # 'detrender' IS the Detrender2dParams (or None), supplied by the caller along with the
    # config. See the note at test_sweep_column_norms_random()'s call.
    dparams = detrender

    A = _abcd_all(config, sweep_all_trees_dense(config, dparams, device='cpu'))

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
            # A fresh chain per t', rather than one continuous stream: a t' near the end of
            # the interval would otherwise leak into the next one, and here correctness
            # matters more than the (toy-scale) cost.
            chain = sweep.make_chain()
            edge = (t_in == tlo) or (t_in == thi-1)
            for j in range(nchunks):
                chain.input_array[...] = 0.0
                geom.write_one_hot(chain.input_array, resp, t_in, j)
                sumsq = chain.dedisperse(j)
                if j != kprobe:
                    continue
                for itree in range(geom.ntrees):
                    # Each of the chunk's nt_ds output times is in steady state, hence equal to
                    # the same column norm, so dividing the chunk's sum of squares by nt_ds
                    # recovers that single value -- and summing THAT over t' gives A directly.
                    # (run_pass() keeps the raw sum, because it sums one response over time
                    # rather than over t'.)
                    ov = sumsq[itree] / geom.tree_nt_ds[itree]
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
        # Read the geometry back from the CONFIG: on a drawn config the arguments say
        # nothing, and a report line describing the wrong geometry is worse than none.
        sbc = [int(x) for x in config.frequency_subband_counts]
        net = max(int(pt.num_early_triggers) for pt in config.primary_trees)
        atomic_print(f'    test_sweep_column_norms(r={int(config.toplevel_tree_rank)},'
                     f' subbands={sbc}, npri={int(config.num_primary_trees)}, net={net},'
                     f' detrender={bool(detrender)}): {len(ifreqs)} columns x {thi-tlo} input'
                     f' times, worst relative difference {worst:.3g} at'
                     f' (tree,ifreq)={worst_where}')

    # float32 dedispersion, and the two sides accumulate different numbers of terms.
    assert worst < 1.0e-5, (worst, worst_where)


def test_sweep_detrender_fp32(r=8, nifreq=16, verbose=True, rng=None):
    """Measures the Detrender2d's own float32 penalty, by running the numpy detrender at
    float32 and float64 on the same one-hots.

    The sweep itself runs the detrender at float64 (the rest of the chain is float32, so that
    is the accurate end), but the GPU Detrender2d is float32-only, so this is the error budget
    the GPU sweep inherits from that stage. Reported as the signed relative error on the
    squared norm of each detrended one-hot, which is what enters A.
    """

    from .brute_force import _SweepGeometry

    # The GEOMETRY is pinned (this is a measurement at one config) but the DETRENDER is
    # drawn: the fp32 penalty is a property of the detrender, so a fixed one bounds the
    # penalty of that detrender and says nothing about the others the GPU sweep may run.
    config = _make_test_config(r, [1])
    dparams = _make_test_detrender(config, rng=_rng(None) if rng is None else rng)
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


# Cost cap for the RANDOMIZED test_sweep_gpu_vs_cpu() draw, in the work units of
# _sweep_work(). ~17x tighter than SWEEP_WORK_BUDGET, and for a specific reason: that budget
# governs test_multimap_vs_sweep(), which runs a GPU sweep only, whereas this test runs a CPU
# sweep as well -- and the CPU arm is 20-50x the GPU one, so it IS the cost.
#
# SET SO THAT THE WORST CASE IS ~10 s, WHICH MEANS THE DETRENDER CASE. Measured at 0.8-1.0x
# of a 2.5e8 budget: det=False landed at 4.5 and 5.8 s, det=True at 12.2 and 15.9 s. So the
# detrender costs about 2.5x MORE PER WORK UNIT than nphases alone predicts -- it does real
# arithmetic beyond multiplying the pass count -- and a single budget cannot make both arms
# land at 10 s. It is set for the expensive arm, so a no-detrender draw comes in around 4 s.
# At this value the mean over draws is well under 2 s, since the work distribution is heavily
# skewed and most draws are nowhere near the cap.
#
# THE COST MODEL MUST SEE THE DETRENDER, which is the one subtlety here. _SweepGeometry sets
# nphases = 2^gamma_max only when W > 0: with no detrender every polyphase pass gives the
# same answer and the sweep takes a single-pass shortcut (that identity is what
# test_sweep_phase_collapse() checks). So a detrended config costs up to 2^gamma_max times as
# much as the same config without one, and _sweep_work(_SweepGeometry(config)) -- no
# detrender argument -- underestimates it by exactly that factor. Measured ratios of 2.0x and
# 8.0x on npri=2 and npri=4 draws. That is why the draw below budgets the (config, detrender)
# PAIR rather than the config, and why it must not be simplified to reuse
# test_multimap_vs_sweep()'s call.
#
# Note the rank does most of the work here: toplevel_tree_rank explains 86% of the variance
# in log(cost) (median cost rises ~8x per rank), so this cap is mostly, but not only, a rank
# cap. The detrender axis is the part a rank cap cannot see.
GPU_VS_CPU_WORK_BUDGET = 1.5e8


def _draw_gpu_vs_cpu_case(max_attempts=500, nbeams=None, detrender=None, rng=None):
    """A random (config, detrender, nbeams) for test_sweep_gpu_vs_cpu(), under the cost cap.

    Returns (config, dparams, nbeams). THE THREE AXES ARE DRAWN SEPARATELY BECAUSE ONLY ONE OF THEM IS A CONFIG PROPERTY.
    'nbeams' and the detrender are arguments this test supplies -- a random config has
    nothing to say about either -- and both matter: the lds kernel uses one beam stride for
    input and output, so a stride error cannot show at nbeams == 1, and the detrender is what
    turns the polyphase sum on. The two config-level properties the fixed calls cover come
    free from the draw instead: npri > 1 (time-downsampled trees) and nfreq != 2^r, the
    latter in essentially every draw since zone_nfreq is drawn per zone in [2^r/4, 2^r] over
    1..5 zones.

    The make_random() flags are test_multimap_vs_sweep()'s, for its reasons: the two flags
    are what make a random config usable by the GPU sweep at all. Measured acceptance 0.73, at
    0.35 ms per attempt.

    THE DETRENDER IS RETURNED, not rebuilt by the caller: _draw_sweep_case() prices the cost
    model on the exact object it built, and since the detrender is drawn, a second
    _make_test_detrender() call would give a different one and the test would run a geometry
    the budget never saw.
    """

    from ..pirate_pybind11 import DedispersionConfig

    rng = _rng() if rng is None else rng
    nb = int(rng.integers(1, 5)) if (nbeams is None) else int(nbeams)

    def _accept(config):
        # The test raises beams_per_{gpu,batch} to nbeams before the GPU sweep, so a draw
        # whose config cannot carry that many is rejected here rather than midway through.
        try:
            config.beams_per_gpu = config.beams_per_batch = nb
            config.validate()
        except RuntimeError:
            return False
        finally:
            config.beams_per_gpu = config.beams_per_batch = 1
        config.validate()
        return True

    config, dparams, _n = _draw_sweep_case(
        rng,
        lambda r: DedispersionConfig.make_random(max_toplevel_rank=8, max_early_triggers=2,
                                                 force_float32=True, no_host_mega_ringbuf=True),
        lambda geom, cfg: _sweep_work(geom), GPU_VS_CPU_WORK_BUDGET,
        detrender=detrender, accept=_accept, max_attempts=max_attempts,
        label='_draw_gpu_vs_cpu_case')
    return config, dparams, nb


def test_sweep_gpu_vs_cpu_random(verbose=True, nbeams=None, detrender=None):
    """test_sweep_gpu_vs_cpu() on a random (config, detrender, nbeams), under a cost cap.

    'nbeams' and 'detrender' PIN the two test-level knobs; None draws them. The config is
    always drawn. That split is the point: neither knob is a property of a
    DedispersionConfig -- this test supplies both -- so they can be covered deterministically
    while the geometry underneath still varies from run to run.

    Run every ten iterations (run_tests()), plus a pinned sweep of the knobs at iteration 0
    (run_once()).
    """

    config, dparams, nbeams = _draw_gpu_vs_cpu_case(nbeams=nbeams, detrender=detrender)
    test_sweep_gpu_vs_cpu(config=config, detrender=dparams, nbeams=nbeams, verbose=verbose)


def test_sweep_gpu_vs_cpu(config, detrender=None, nbeams=1, verbose=True):
    """The GPU sweep against the CPU one, element by element, on the same config.

    Both GPU kernels are separately validated against their reference implementations
    ('pirate_frb test --sbdd' and '--pfsq'), so a discrepancy here points at the driver rather
    than at a kernel.

    The config and the detrender are both supplied by the caller; _draw_gpu_vs_cpu_case()
    draws them together, and says there which geometries matter and which of them come free
    from the draw. 'nbeams' is the one knob this test supplies that a config has nothing to
    say about, and it is worth raising: the lds kernel reads and writes with a single beam
    stride, so a stride error is invisible at nbeams == 1.
    """

    from .brute_force import compute_variance_multimap

    # 'detrender' IS THE Detrender2dParams (or None), not a flag saying to build one. It has
    # to be, because the detrender is drawn: _draw_gpu_vs_cpu_case() prices the case on one
    # particular detrender, and building a second here would run a geometry its cost model
    # never saw.
    dparams = detrender

    # The CPU reference is detrended at the GPU's precision, so that both sides run the same
    # detrender and the bar below measures the DRIVER. Without this the detrender's own float32
    # penalty -- which test_sweep_detrender_fp32 only bounds AT 1e-4 -- is folded into the same
    # 1e-4 assertion, and a real driver error could be mistaken for it.
    cpu = compute_variance_multimap(config, detrender=dparams, device='cpu',
                                    detrender_dtype=np.float32)

    config.beams_per_gpu = config.beams_per_batch = nbeams
    if dparams is not None:
        dparams.M = nbeams
    gpu = compute_variance_multimap(config, detrender=dparams, device='gpu')

    worst, worst_where = 0.0, None
    for gamma in range(cpu.num_primary_trees):
        want = np.asarray(cpu.primary_map(gamma).A)
        got = np.asarray(gpu.primary_map(gamma).A)
        assert got.shape == want.shape, (got.shape, want.shape)
        scale = float(np.abs(want).max())
        e = float(np.abs(got - want).max()) / scale if (scale > 0) else 0.0
        if e > worst:
            worst, worst_where = e, gamma

    if verbose:
        # Read the geometry back from the CONFIG, not from the arguments: on a drawn config
        # the arguments say nothing, and a report line that describes the wrong geometry is
        # worse than none.
        sbc = [int(x) for x in config.frequency_subband_counts]
        net = max(int(pt.num_early_triggers) for pt in config.primary_trees)
        atomic_print(f'    test_sweep_gpu_vs_cpu(r={int(config.toplevel_tree_rank)},'
                     f' subbands={sbc}, npri={int(config.num_primary_trees)}, net={net},'
                     f' detrender={bool(detrender)},'
                     f' nbeams={nbeams}, nfreq={cpu.primary_map(0).nfreq}): worst relative'
                     f' difference {worst:.3g} at primary tree {worst_where}')

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
    # 'detrender' stays a knob (the callers sweep it on/off), but the detrender it turns on
    # is drawn: what this test asserts is bit-identity between two reductions, which holds
    # for any detrender, so pinning one bought nothing.
    dparams = _make_test_detrender(config, rng=_rng()) if detrender else None
    dense = compute_variance_multimap(config, detrender=dparams, device='cpu')

    npri = dense.num_primary_trees
    nL = 0
    # Per PRIMARY tree: L is a per-primary-tree quantity now, and a child tree never carries
    # a coarse-graining rank of its own.
    for (gamma, m) in enumerate(dense.maps):
        R, rr = m.pf_rank, m.tree_rank
        for L in range(R, rr + 1):
            Ls = [None] * npri
            Ls[gamma] = L
            got = compute_variance_multimap(config, detrender=dparams,
                                            device='cpu', L=Ls).primary_map(gamma)
            want = m.coarse_grain(L)
            assert got.nbeta == want.nbeta, (gamma, L, got.nbeta, want.nbeta)
            nd = int(np.count_nonzero(np.asarray(got.A) != np.asarray(want.A)))
            if nd != 0:
                raise RuntimeError(f'test_sweep_streaming_coarse: primary tree {gamma},'
                                   f' L={L}: {nd} of {got.nbeta * got.nfreq} entries differ'
                                   ' from coarse_grain() of the dense map')
            assert np.array_equal(got.y_true, m.y_true), (gamma, L)
            nL += 1

    # THE STAGING WIDTH MUST NOT BE ABLE TO CHANGE THE ANSWER. The accumulator holds a block
    # of reduced columns and flushes it into the output, so the width is a locality knob --
    # but it also decides where a block boundary falls, which is exactly what the
    # fresh-row-versus-accumulate branch keys off. Widths that divide nfreq and widths that do
    # not are both required to reproduce the default bit for bit.
    from .brute_force import _Accumulator
    ref = np.asarray(dense.primary_map(0).A)
    saved = _Accumulator._NSTAGE
    try:
        for nstage in (1, 3, 7, 2 * saved):
            _Accumulator._NSTAGE = nstage
            got = compute_variance_multimap(config, detrender=dparams,
                                            device='cpu').primary_map(0)
            nd = int(np.count_nonzero(np.asarray(got.A) != ref))
            if nd:
                raise RuntimeError(f'test_sweep_streaming_coarse: staging width {nstage}'
                                   f' changed {nd} entries of the map')
            assert np.array_equal(got.y_true, dense.primary_map(0).y_true), nstage
    finally:
        _Accumulator._NSTAGE = saved

    # A partial sweep is the one case where y_true would be a sum over the swept channels
    # only, so it is dropped rather than reported.
    d0 = dense.primary_map(0)
    chans = [0, d0.nfreq // 2, d0.nfreq - 1]
    part = compute_variance_multimap(config, detrender=dparams, device='cpu', channels=chans)
    assert part.provenance['partial'] is True
    assert part.primary_map(0).y_true is None, 'a partial sweep must not claim a y_true'
    Ap, Af = np.asarray(part.primary_map(0).A), np.asarray(d0.A)
    assert np.array_equal(Ap[:, chans], Af[:, chans]), 'swept columns must match a full sweep'
    unswept = [c for c in range(d0.nfreq) if c not in chans]
    assert not np.any(Ap[:, unswept]), 'unswept columns must be identically zero'

    if verbose:
        atomic_print(f'    test_sweep_streaming_coarse(r={r}, subbands={subband_counts},'
                     f' net={num_early_triggers}, detrender={bool(detrender)}): the streaming'
                     f' reduction is bit-identical to coarse_grain() at all {nL} legal (tree,'
                     f' L) pairs; four staging widths agree bitwise; a {len(chans)}-channel'
                     ' partial sweep reports no y_true')


####################   varmap/SparseTile.py, varmap/PfVarianceConvolver.py   ####################
#
# The low-level primitives, and the C++ port of them in src_lib/varmap.cpp. They are
# dispatched from '--varmap' rather than from a flag of their own, because everything here is
# a varmap primitive and whoever edits one runs '--varmap'.
#
# The tests themselves stay as staticmethods on the classes they test, where their docstrings
# are. Only the cadence decision lives here, and it is the same split the classes document:
# the randomized ones every iteration, the deterministic ones once.
#
# Each compares LIVE code against something outside it -- a dense reference, or a production
# kernel (ReferenceTreeGriddingKernel, ReferencePeakFindingKernel) -- so none of this is the
# old-vs-new comparison that the slow_avar deletion was about. Cost: ~0.07 s per iteration,
# measured, against this file's ~28 s.


def run_primitive_tests(once):
    """SparseTile / SparseTileTriple / PfVarianceConvolver, and their C++ ports."""

    from .SparseTile import SparseTile, SparseTileTriple
    from .PfVarianceConvolver import PfVarianceConvolver
    from ..fast_varmap import test_fast_varmap

    SparseTileTriple.test_random_tree_gridding()
    SparseTile.test_random_iterate_aligned()
    SparseTile.test_random_iterate_singletons()
    SparseTile.test_random_remap_d()
    SparseTile.test_random_scale()
    SparseTile.test_random_predict_dbits()
    PfVarianceConvolver.test_reduces_to_norms()
    PfVarianceConvolver.test_random_variance()

    test_fast_varmap.test_cpp_convolver()
    test_fast_varmap.test_cpp_sparse_tile_triple()

    if once:
        # All three are run_once()'s "reason 1": the parameter space is exhausted, not merely
        # expensive. Both convolver tests are deterministic (they say so), and
        # test_cpp_predict_dbits is a 57792-case exhaustive sweep plus 10000 wide random draws.
        PfVarianceConvolver.test_kernels_match_reference()
        PfVarianceConvolver.test_unimodality()
        test_fast_varmap.test_cpp_predict_dbits()


####################################   entry point   ####################################


def test_apply_restriction():
    """The row map of Proposition 1, as apply() uses it: restricting a parent's FINE apply()
    result to a child tree's rows.

    This is NOT a test of Proposition 1 itself -- that is test_restriction_vs_sweep(), which
    needs a sweep. Here the maps are RANDOM, so the only thing under test is the index
    bookkeeping: restrict_fine_vector()'s reshape-and-gather, and (when L is set)
    apply_fine()'s lift. That separation is the point -- a bug in either shows up here,
    scale-free and with no dedisperser in the loop.

    The row map is rebuilt in the test from toplevel band ranges, in the manner of
    _obvious_beta(), so the production m_index_mapping() is not in the loop.

    THE CASE THAT MAKES THE GATHER VISIBLE is a multiplet map that is not a contiguous
    prefix, which needs the restriction to clamp a level that still has populated levels
    above it (subbands [4,2,1] restricting to [2,1] gives m_map = [0,1,4,5]). When every map
    IS a prefix, a wrong gather passes unnoticed. The draw supplies it about 10% of the time,
    so the count is REPORTED rather than demanded; see the note above the report line.

    FINE OR COARSE, DRAWN PER PRIMARY TREE, and the coarse arm is the one that matters:
    VarianceMap.apply_fine()'s lift runs nowhere else in the package -- every other call site
    is fine-only -- and getting it wrong changes values without changing any shape, so a
    fine-only run never executes it at all. L is drawn from [R, r], both endpoints included:
    R leaves the DM axis alone and only merges M -> N, r collapses it entirely.
    """

    from .VarianceMultiMap import restrict_fine_vector

    rng = _rng()
    config = _random_config(rng)
    nfreq = int(config.get_total_nfreq())
    v = rng.uniform(0.5, 2.0, size=nfreq)

    npri = int(config.num_primary_trees)
    ncheck = nontrivial = 0

    Ls = []
    for gamma in range(npri):
        iparent = _itree(config, gamma)
        parent = _tree(config, iparent)
        pm = _random_map(config, iparent, rng)

        # Half fine, half coarse. Per PRIMARY TREE rather than per call, so one invocation
        # covers both arms even at npri == 1 over a few iterations, and the legal range is
        # the parent's own [R, r].
        R, rr = int(parent.frequency_subbands.pf_rank), int(parent.tree_rank)
        L = None if (rng.random() < 0.5) else int(rng.integers(R, rr + 1))
        Ls.append(L)
        if L is not None:
            pm = pm.coarse_grain(L)

        D = 1 << (parent.tree_rank - parent.frequency_subbands.pf_rank)
        M_p, P = int(parent.frequency_subbands.M), int(parent.nprofiles)
        y_fine = pm.apply_fine(v)
        assert y_fine.shape == (D * M_p * P,)

        for e in range(int(config.primary_trees[gamma].num_early_triggers) + 1):
            ichild = _itree(config, gamma, e)
            child = _tree(config, ichild)
            fsc = child.frequency_subbands
            M_c = int(fsc.M)

            # The multiplet map, built here from toplevel band ranges rather than imported.
            pband = {(parent.n_to_toplevel_flo(n), parent.n_to_toplevel_fhi(n)): n
                     for n in range(int(parent.frequency_subbands.N))}
            m_map = []
            for mc in range(M_c):
                nc = int(fsc.m_to_n[mc])
                np_ = pband[(child.n_to_toplevel_flo(nc), child.n_to_toplevel_fhi(nc))]
                m_map.append(int(parent.frequency_subbands.n_to_mbase[np_]) + int(fsc.m_to_d[mc]))

            got = restrict_fine_vector(y_fine, make_plan(config), iparent, ichild)
            assert got.shape == (D, M_c, P), (got.shape, (D, M_c, P))

            want = np.empty((D, M_c, P))
            for d in range(D):
                for mc in range(M_c):
                    for p in range(P):
                        want[d, mc, p] = y_fine[(d * M_p + m_map[mc]) * P + p]
            assert np.array_equal(got, want), f'gamma={gamma}, e={e}: row map disagrees'
            if m_map != list(range(M_c)):
                nontrivial += 1

            # Independent check on the GRANULARITY of the lift, which is the thing a
            # fine-only run never exercises. A parent coarse-grained at L induces, on the
            # child's rows, exactly the child's own coarse-graining at rank (L - e): the
            # child's pf_rank is R - e, so its full-resolution DM is the parent's shifted by
            # e, and (dm_parent >> L) == (dm_child >> (L - e)). Production never computes
            # that shift -- it lifts before it restricts -- so deriving it here is what would
            # catch production getting it wrong.
            if L is not None:
                labels = _child_group_labels(child, L - e)
                for key in np.unique(labels):
                    vals = got.reshape(-1)[labels == key]
                    assert np.all(vals == vals[0]), \
                        f'gamma={gamma}, e={e}: the restricted vector is not constant on the' \
                        f' child group {key}, so the lift used the wrong granularity'
            ncheck += 1

    # 'nontrivial' is REPORTED, not asserted: a non-contiguous multiplet map needs the
    # restriction to clamp a level that still has populated levels above it, which is
    # emergent at ~10% per draw and cannot be demanded of one config. Over an '-n 100' run it
    # is exercised ~10 times, and 'pirate_frb dev coverage' tracks the rate.

    nets = [int(pt.num_early_triggers) for pt in config.primary_trees]
    atomic_print(f'    test_apply_restriction(npri={npri}, nets={nets}, L={Ls}):'
                 f' {ncheck} (parent, child) pairs,'
                 f' {nontrivial} with a non-contiguous multiplet map,'
                 f' {sum(x is not None for x in Ls)} lifted')


def _child_group_labels(tree, L):
    """The coarse-graining labels of 'tree' at rank L, the long way round.

    Same construction as _obvious_labels(), but from a DedispersionTree rather than a
    VarianceMap, since a child tree has no VarianceMap of its own (that is the whole point of
    the per-primary-tree representation).
    """

    fs = tree.frequency_subbands
    R = int(fs.pf_rank)
    M, N, P = int(fs.M), int(fs.N), int(tree.nprofiles)
    D = 1 << (int(tree.tree_rank) - R)

    n_level = []
    for level, count in enumerate(fs.subband_counts):
        n_level += [level] * int(count)
    n_level = np.array(n_level, dtype=np.int64)

    alpha = np.arange(D * M * P, dtype=np.int64)
    p = alpha % P
    mi = (alpha // P) % M
    d = alpha // (P * M)
    n = np.array(fs.m_to_n, dtype=np.int64)[mi]
    dm_full = (d << R) + (np.array(fs.m_to_d, dtype=np.int64)[mi] << (R - n_level[n]))

    return ((dm_full >> L) * N + n) * P + p


def test_restriction_representation(K=None):
    """The restricted apply() result does not depend on how the PARENT is represented.

    A factored parent contracts K vectors while the dense map it stands for sums each row over
    nfreq, so the two group their additions differently -- ~1e-13 relative, the same order
    row_sums()'s docstring documents. Nothing here should be bitwise.

    Worth having because production stores factored maps and the tests mostly build dense
    ones: this is the one check that the restriction path is exercised on both.
    """

    from .VarianceMultiMap import restrict_fine_vector

    rng = _rng()
    config = _random_config(rng)
    K = _draw_K(rng) if K is None else K
    nfreq = int(config.get_total_nfreq())
    v = rng.uniform(0.5, 2.0, size=nfreq)

    plan = make_plan(config)
    iparent = int(plan.dedispersion_tree_index(0, 0))

    fac, dense_A = _factored_map(config, iparent, rng, K=K)
    dense = VarianceMap.from_dense(config, iparent, dense_A)

    worst = 0.0
    for e in range(int(config.primary_trees[0].num_early_triggers) + 1):
        ichild = int(plan.dedispersion_tree_index(0, e))
        a = restrict_fine_vector(fac.apply_fine(v), plan, iparent, ichild)
        b = restrict_fine_vector(dense.apply_fine(v), plan, iparent, ichild)
        assert a.shape == b.shape
        scale = max(np.max(np.abs(b)), 1e-300)
        worst = max(worst, float(np.max(np.abs(a - b)) / scale))

    assert worst < 1e-11, f'test_restriction_representation: worst relative difference {worst:.3g}'
    atomic_print(f'    test_restriction_representation(K={K}): factored and dense'
                 f' parents agree to {worst:.3g} relative')


# Work-unit ceiling for test_restriction_vs_sweep()'s CPU sweep, in the same units as
# SWEEP_WORK_BUDGET (see there for what a work unit is and why a budget is not optional).
# MEASURED END TO END, not derived: at this ceiling the test costs a mean of 0.9 s and a
# worst case of about 2.5 s (the CPU sweep with a Detrender2d runs at 79-187 ns per work
# unit, and the accepted population's mean work is well below the ceiling). The test runs
# every tenth iteration, so that is ~0.1 s per iteration of run_tests(). Changing this
# changes a running time, not a property: nothing asserted below depends on it.
RESTRICTION_SWEEP_BUDGET = 2.5e7


def _draw_restriction_config(rng):
    """A drawn config for test_restriction_vs_sweep(), with the detrender that matches it.

    Returns (config, dparams, ndraw). THREE FILTERS, and each one is there because a config
    that fails it makes the test say nothing or cost too much:

      - AT LEAST ONE EARLY TRIGGER. Without one the config has no (parent, child) pairs at
        all and the whole comparison loop is empty. Measured over 1200 draws, 31% have one.
      - AT LEAST ONE NON-CONTIGUOUS MULTIPLET MAP. This is the property the comparison has
        teeth against: where the map IS the contiguous prefix 0, 1, ..., a wrong row map
        would agree with the right one. Only 11% of draws have one, which is why this is the
        one place in the file that guarantees it -- test_multimap_vs_sweep() checks the same
        proposition on five drawn configs an iteration and reaches a non-contiguous map about
        45% of the time.
      - THE CPU SWEEP FITS RESTRICTION_SWEEP_BUDGET, evaluated with the detrender in place,
        since the detrender is most of the cost.

    FILTER-AND-RETRY rather than arguments (notes/unit_tests.md point 3), because
    make_random() has a max_early_triggers and no minimum, and nothing anywhere can ask for a
    non-contiguous multiplet map -- it is an emergent property of the subband tables.

    THE TIGHTEST FILTER IN THE FILE, and worth knowing before touching it: measured
    acceptance is 0.025, i.e. a mean of 39 draws per case (median 22, max 272 over 25 calls)
    at 0.40 ms each. That is still ~16 ms against the ~1 s sweep it protects, so the cost is
    not the concern; the cap in _draw_sweep_case() is, since a tightening elsewhere could
    plausibly drive this filter to zero.
    """
    # THROUGH _random_config(), so that gpu_valid is drawn. It matters more here than
    # anywhere: the gpu_valid=True path can only draw subband vectors the cdd2 registry
    # stocks, and every accepted config then has subband_counts (4,2,1) and one primary tree.
    # With gpu_valid drawn the accepted population spans six subband vectors, toplevel ranks
    # 5 to 8, and one or two primary trees. Nothing here needs a GPU: this is the CPU sweep.
    def _accept(config):
        pairs = _restriction_pairs(config)
        return any(m_map != list(range(len(m_map))) for (_, _, _, _, m_map) in pairs)

    return _draw_sweep_case(
        rng, lambda r: _random_config(r, max_toplevel_rank=8),
        lambda geom, cfg: _sweep_work(geom), RESTRICTION_SWEEP_BUDGET,
        detrender=True, accept=_accept, max_attempts=2000,
        label='_draw_restriction_config')


def test_restriction_vs_sweep(verbose=True):
    """Proposition 1 itself, against the sweep.

    The appendix "Variance maps of a config's trees are row-restrictions of one another" in
    notes/variance_map.tex proves that the variance map of tree (gamma, e) is a subset of the
    ROWS of the map of tree (gamma, 0). This sweeps every tree independently and checks that
    element by element.

    What makes this a test of the PROPOSITION rather than of the plumbing: the two sides are
    computed by different trees -- different tree ranks, different subband tables, different
    ReferenceTree and ReferencePfSquare instances -- and the multiplet map is rebuilt by
    _restriction_pairs() from toplevel band ranges, so no production row map is in the loop.
    (The two trees do share an upstream chain, which is exactly what Proposition 1 assumes.)

    THE CONFIG IS DRAWN, and _draw_restriction_config() says what it has to satisfy. The
    DETRENDER IS NOT A CHOICE HERE: a Detrender2d assumes nothing about the upstream chain,
    so Proposition 1 has to survive one, and that arm is the only part of the proposition
    that test_multimap_vs_sweep() does not already cover on drawn configs every iteration --
    it sweeps without a detrender, because its own comparison is against a detrender-free
    map, and giving it one would mean a second sweep per config.

    Expect agreement at the float64 level here, not float32. The two sides differ only in
    summation order, and the CPU sweep accumulates in float64, so the measured worst case is
    a few times 1e-16. The bar below is nevertheless 1e-5: it is sized for a WRONG ROW MAP,
    which mismatches whole bands, and it has to hold on any drawn geometry rather than on the
    one that happened to be measured.
    """

    from .brute_force import sweep_all_trees_dense

    rng = _rng()
    config, dparams, ndraw = _draw_restriction_config(rng)
    As = sweep_all_trees_dense(config, dparams, device='cpu')

    worst, worst_where, npairs, nontrivial = 0.0, None, 0, 0
    plan = make_plan(config)
    trees = plan.trees

    for (gamma, e, iparent, ichild, m_map) in _restriction_pairs(config):
        parent, child = trees[iparent], trees[ichild]
        fsp, fsc = parent.frequency_subbands, child.frequency_subbands
        D = 1 << (int(parent.tree_rank) - int(fsp.pf_rank))
        M_p, M_c, P = int(fsp.M), int(fsc.M), int(parent.nprofiles)
        nfreq = As[iparent].shape[1]

        assert (1 << (int(child.tree_rank) - int(fsc.pf_rank))) == D
        assert int(child.nprofiles) == P

        Ap = np.asarray(As[iparent]).reshape(D, M_p, P, nfreq)
        Ac = np.asarray(As[ichild]).reshape(D, M_c, P, nfreq)

        if m_map != list(range(M_c)):
            nontrivial += 1

        for mc in range(M_c):
            got, want = Ac[:, mc], Ap[:, m_map[mc]]
            scale = max(float(np.max(np.abs(want))), 1e-300)
            d = float(np.max(np.abs(got - want))) / scale
            if d > worst:
                worst, worst_where = d, (gamma, e, mc)
        npairs += 1

    # Guaranteed by the draw rather than hoped for -- see _draw_restriction_config(). Kept as
    # an assertion because it is the filter that is being checked, not the geometry.
    assert nontrivial > 0, \
        'test_restriction_vs_sweep: every multiplet map was the contiguous prefix, so a wrong' \
        ' row map could not have been caught (see test_apply_restriction for why)'

    assert worst < 1e-5, (f'test_restriction_vs_sweep: worst relative difference {worst:.3g}'
                          f' at (gamma, e, m_child) = {worst_where}')

    if verbose:
        nets = [int(pt.num_early_triggers) for pt in config.primary_trees]
        atomic_print(f'    test_restriction_vs_sweep(r={int(config.toplevel_tree_rank)},'
                     f' subbands={[int(x) for x in config.frequency_subband_counts]},'
                     f' nfreq={int(config.get_total_nfreq())}, npri={len(nets)},'
                     f' nets={nets}, W={int(dparams.W)}, {ndraw} draws):'
                     f' {npairs} (parent, child) pairs, {nontrivial} non-contiguous,'
                     f' worst relative difference {worst:.3g}')


def test_lds_bindings(r=8, subband_counts=(2,2,1), num_primary_trees=3, nbeams=4):
    """The GpuLaggedDownsamplingKernel / DedispersionBuffer bindings, without a sweep.

    --gldk tests the KERNEL (against its reference implementation). This tests the BINDING:
    that the plan's params reach python, that a DedispersionBuffer allocates the shapes those
    params predict, and that launch() fills the downsampled buffers.

    THE BEAM STRIDE IS THE POINT. The kernel reads its input and writes every output with a
    SINGLE beam stride, so all of bufs[] must be sub-arrays of one allocation. That is why the
    binding takes a DedispersionBuffer rather than a list of arrays, and it is invisible at
    nbeams == 1 (where the stride is degenerate), so this runs at nbeams > 1.
    """

    import cupy as cp

    from ..pirate_pybind11 import (DedispersionPlan, DedispersionBuffer,
                                   GpuLaggedDownsamplingKernel)
    from ..core import BumpAllocator

    config = _make_test_config(r, subband_counts, num_primary_trees=num_primary_trees)
    config.beams_per_gpu = config.beams_per_batch = nbeams
    config.num_active_batches = 1
    config.validate()

    plan = DedispersionPlan(config)
    bp, lp = plan.stage1_dd_buf_params, plan.lds_params

    npri = int(config.num_primary_trees)
    assert int(bp.nbuf) == npri and int(lp.num_primary_trees) == npri
    assert int(lp.input_toplevel_rank) == int(config.toplevel_tree_rank)

    allocator = BumpAllocator('af_gpu | af_zero', -1)
    buf = DedispersionBuffer(bp)
    buf.allocate(allocator)
    kernel = GpuLaggedDownsamplingKernel(lp)
    kernel.allocate(allocator)
    assert buf.is_allocated and kernel.is_allocated and buf.on_gpu()

    # Shapes are the plan's, and every buffer shares ONE beam stride.
    bstride = None
    for ipri in range(npri):
        b = buf.bufs[ipri]
        want = (nbeams, 1 << int(bp.buf_rank[ipri]), int(bp.buf_ntime[ipri]))
        assert tuple(b.shape) == want, (ipri, tuple(b.shape), want)
        assert want[1] == (1 << (int(config.toplevel_tree_rank) - (1 if ipri else 0))), ipri
        assert want[2] == int(config.time_samples_per_chunk) >> ipri, ipri
        # Non-contiguous beam axis: the arrays are interleaved in one allocation.
        assert b.strides[0] != b.shape[1] * b.shape[2] * 4, ipri
        bstride = b.strides[0] if (bstride is None) else bstride
        assert b.strides[0] == bstride, (ipri, b.strides[0], bstride)

    # launch() reads bufs[0] and fills the rest.
    rng = _rng()
    b0 = cp.asarray(buf.bufs[0])
    b0[...] = cp.asarray(rng.normal(size=b0.shape).astype(np.float32))
    for ipri in range(1, npri):
        assert not bool((cp.asarray(buf.bufs[ipri]) != 0).any()), ipri

    kernel.launch(buf, 0, 0, cp.cuda.get_current_stream().ptr)
    cp.cuda.Stream.null.synchronize()

    for ipri in range(1, npri):
        assert bool((cp.asarray(buf.bufs[ipri]) != 0).any()), \
            f'test_lds_bindings: launch() left bufs[{ipri}] all zero'

    atomic_print(f'    test_lds_bindings(r={r}, npri={npri}, nbeams={nbeams}): shapes match the'
                 f' plan, one beam stride ({bstride}) shared by all {npri} buffers, launch'
                 ' fills the downsampled buffers')


def run_tests(iteration=0):
    """Every pirate_frb.varmap test, at its cadence. This is what '--varmap' calls.

    THREE CADENCES, AND THEY ARE THE ONLY ORGANIZING PRINCIPLE HERE. The functions are named
    for when they run, not for what they cover, because "when" is the only thing the caller
    or this function has to decide; what each test covers is the business of its own
    docstring.
    """

    if iteration == 0:
        run_once()

    run_all()

    # test_sweep_gpu_vs_cpu is the only check on the GPU sweep DRIVER and the most expensive
    # test in the package, so it runs every tenth iteration rather than every one. See
    # test_sweep_gpu_vs_cpu_random(), and run_once() for the knob sweep that complements it.
    #
    # test_restriction_vs_sweep() is here for a different reason: it runs a CPU sweep, which
    # is ~36x slower per work unit than the GPU one, so its cost budget buys a much smaller
    # config per second. Its own budget puts it at a mean of 0.9 s per call, i.e. ~0.1 s per
    # iteration at this cadence.
    if (iteration % 10) == 0:
        test_sweep_gpu_vs_cpu_random()
        test_sweep_column_norms_random()
        test_restriction_vs_sweep()


def run_once():
    """Everything that runs ONCE PER INVOCATION, for one of two distinct reasons.

    REASON 1: the parameter space really is exhausted -- notes/unit_tests.md item 11. The
    test enumerates a FIXED list of rejections, or has no parameters at all. The config
    underneath, where there is one, is scaffolding for the rejection rather than a case being
    sampled, so randomizing it would buy nothing and cost an iteration's worth of time.

    REASON 2: too expensive to repeat. Each brute-force sweep test pushes a one-hot through
    the real dedisperser once per input channel, which is ~15 s for the group. Those are
    pinned or knob-swept rather than exhausted, and each says below why it is not randomized.

    The two reasons are worth keeping distinct: a test here for reason 1 should stay here
    forever, and a test here for reason 2 is a candidate to randomize if it ever gets cheap.
    """

    # ---- reason 1: exhausted ----
    run_primitive_tests(once=True)
    test_lp_config()
    test_constructor_validation()
    test_factored_validation()

    # ---- reason 2: too expensive to repeat ----
    # THE KNOB SWEEP. Four driver paths have to be covered here -- the lds kernel's single
    # beam stride (invisible at nbeams == 1), the Detrender2d, time-downsampled trees, and
    # nfreq != 2^r -- and a drawn case reaches all four, but only probabilistically: measured
    # over 400 accepted draws, nbeams > 1 in 75%, a detrender in 43%, npri > 1 in 34%. Over an
    # '-n 100' run that is near-certain; over ONE iteration it is not.
    #
    # So pin the two axes that are TEST KNOBS rather than config properties -- nbeams and
    # the detrender -- and sweep their 2x2 product, leaving the geometry drawn. That restores
    # the guarantee on every invocation while the config still varies from run to run, and it
    # costs about a quarter of what four pinned configs would. npri and nfreq stay emergent:
    # neither is a knob this test supplies, and test_multimap_vs_sweep() reports both.
    for _nb in (1, 4):
        for _det in (False, True):
            test_sweep_gpu_vs_cpu_random(nbeams=_nb, detrender=_det)

    # ---- ONCE PER INVOCATION, ON FIXED CONFIGS ----
    #
    # Everything below still pins its geometry, and each group says why. The rule this file
    # now follows is that a test draws its config; these are the exceptions, and an exception
    # needs a reason that is not "it was written that way".

    # STRUCTURALLY PINNED: it asserts gamma_max == 2, i.e. exactly three primary trees, so
    # the phase loop has something to collapse. A drawn config gives npri > 1 about half the
    # time and npri == 3 rather less, so randomizing this means a redraw loop around a 0.09 s
    # test with one thing to say.
    test_sweep_phase_collapse(7)

    # A MEASUREMENT, NOT REALLY A TEST -- item 11's "informational print for a human". It
    # reports the Detrender2d's own float32 penalty, which is the error budget the GPU sweep
    # inherits and which test_sweep_gpu_vs_cpu deliberately factors OUT (it runs its CPU
    # reference at the GPU's precision, so its bar measures the DRIVER). This is the only
    # thing that bounds the penalty itself.
    test_sweep_detrender_fp32(7)

    # EXHAUSTIVE ALREADY, so item 11 applies as written: every legal (tree, L) pair and four
    # staging widths, all required to be bit-identical. The two subband layouts are the one
    # place in this tier where a drawn config could plausibly LOSE something -- levels 1 and
    # 0 mixed in different proportions is where the multiplet decomposition can go wrong, and
    # [2,1] gives (M,N)=(4,3) while [1,1] gives (3,2) -- though a long run would regain it.
    test_sweep_streaming_coarse(6, [2, 1])
    test_sweep_streaming_coarse(6, [1, 1], num_early_triggers=1)
    test_sweep_streaming_coarse(6, [2, 1], detrender=True)

    # A BINDING CONTRACT, not a numerical one: that the plan's params reach python, that a
    # DedispersionBuffer allocates the shapes they predict, and that launch() fills the
    # downsampled buffers. THE BEAM STRIDE IS THE POINT -- the lds kernel reads and writes
    # with a single beam stride, so a stride error is invisible at nbeams == 1. nbeams is a
    # knob and could be drawn; npri >= 2 is not, and the test needs it to check anything.
    test_lds_bindings()


def run_all():
    """Everything else, ONCE PER '-n' ITERATION, in dependency order: the index arithmetic
    first, since the rest is built on it.

    Each test here draws its own geometry, so a long run explores rather than repeating. See
    run_once() for the handful that deliberately does not.
    """

    run_primitive_tests(once=False)
    test_index_arithmetic()
    test_coarse_grain()
    test_distance()
    test_admissibility()
    test_distance_oracles()
    test_estimate_distance()
    test_multimap()
    test_asdf_io()
    test_asdf_detrender()
    test_factored_algebra()
    test_factored_equivalence()
    test_factored_transformations()
    test_asdf_factored()
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
    test_greedy_bookkeeping()
    test_svd_optimize()
    test_apply_restriction()
    test_restriction_representation()
    test_map_steps()
    test_report()
    test_base_varmap_coarse()
    test_multimap_vs_base()
    test_varfine()

    # The C++ port of compute_detrender_free_{varfine,varcoarse} (src_lib/varmap.cpp), checked
    # against the python above. It lives in fast_varmap with the other C++-vs-python comparisons,
    # but it is dispatched from HERE rather than from '--avar' because the reference it guards is
    # detrender_free.py: whoever edits that file runs '--varmap', and a port that has silently
    # diverged is exactly what they need to be told about. Adds ~0.1 s per iteration, nearly all
    # of it the python reference.
    from ..fast_varmap.test_fast_varmap import test_cpp_detrender_free
    test_cpp_detrender_free()

    # The brute-force sweep test that draws its own config and is cheap enough to run every
    # iteration -- about 1.4 s. The rest of the sweep group is in run_once() (pinned or knob
    # swept) or in run_tests()'s every-tenth tier; see this module's docstring for what the
    # sweep half is FOR.
    test_multimap_vs_sweep()
