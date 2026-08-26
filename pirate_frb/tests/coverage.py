"""Coverage analysis of the randomization utilities the unit tests draw from.

NOT A TEST: nothing here asserts. It is a diagnostic, reached as 'pirate_frb coverage', and
it answers one question -- how often does a randomized unit test actually get the structure
it needs? Several tests in pirate_frb.varmap draw configs from
DedispersionConfig::make_random() and REPORT rather than assert what they covered, precisely
because a fair draw misses some cases; this is where you find out what those rates are.

EVERY LINE PRINTED HERE HAS A CONSUMER. If you add one, say which test needs the structure;
if a line no longer has a test behind it, delete it.
"""

import numpy as np

from ..pirate_pybind11 import DedispersionConfig, DedispersionTree
from ..utils import atomic_print, print_separator


def report_coverage():
    """How often does a randomized varmap test get the structure it needs?

    IT DRAWS FROM varmap.tests._random_config(), NOT FROM make_random() DIRECTLY, and that is
    the whole point. Those tests draw at max_toplevel_rank=7 with gpu_valid drawn per config;
    reporting make_random()'s unconditioned distribution instead would describe a population
    no test samples, which is what this file did until the settings drifted apart. Calling
    the same helper keeps the two in step by construction.

    EVERY ROW BELOW NAMES THE TEST THAT NEEDS IT. Each was an assertion until the tests were
    randomized: a test that asserts it was handed a structure is asserting a property of the
    draw, not of the code (notes/unit_tests.md item 8), so they became reported counts and
    this is where the rates live. If a row's consumer disappears, delete the row.

      max_width chain not flat        test_multimap_vs_base, test_varfine. Makes the profile
                                      restriction P_gamma < P_0 something other than a no-op.
                                      On the gpu_valid path it needs the cdd2 registry to
                                      stock two Wmax for the DOWNSAMPLED tree's key, so it is
                                      a property of which kernels this build compiled.
      early triggers                  test_varfine, test_apply_restriction. Without one,
                                      nothing exercises Proposition 1: the early-trigger
                                      trees, restrict_fine_vector(), apply_fine()'s expansion.
      non-contiguous multiplet map    test_apply_restriction, test_multimap_vs_sweep. The
                                      only case where a wrong gather is visible at all --
                                      with a prefix map an off-by-one moves nothing. Needs
                                      restrict_subband_counts() to clamp a level that still
                                      has populated levels ABOVE it; see point 3 of the long
                                      comment in makefile_helper.py. Was 0% on the gpu_valid
                                      path until (4,2,1) was stocked there.
      a straddled entry               test_base_varmap_vs_analytic. The half-aligned branch
                                      in SdPlan, 1 row in 645 on toy.yml.
      xdm_rank > 0 in some tree       test_base_varmap_vs_analytic. Nothing in varmap READS
                                      K, but K > 0 is precisely where 2^(r-R) and ndm_out
                                      diverge, so it is the only case where a row count taken
                                      from the wrong one is visible.
      R == 0                          test_base_varmap_vs_analytic. Degenerate subband
                                      geometry: N = M = 1, no coarse DM axis to speak of.
      nfreq < 2^r                     test_base_varmap_vs_analytic. The production-like
                                      regime -- chord_sb2_et.yml grids 28160 channels onto
                                      65536 tree-freqs -- and the one that widens footprints.
      N > 1 (group sizes differ)      test_estimate_distance, test_basis_constructors. D is a
                                      mean over FINE rows, so a plain mean over groups is
                                      biased only when the sizes differ; and the greedy
                                      merge's size weighting differs from group-blind only
                                      then.
      nfreq a multiple of 8           test_lp_repairs. blocking_is_exact(): a row-blocked
                                      pass is bit-identical to an unblocked one only there,
                                      so the blocking check runs only on those draws.

    Two structural diagnostics that carry no assertion but explain a rate when it moves: the
    npri histogram, and the number of DISTINCT subband_counts drawn. The second is the
    tripwire on the cdd2 registry -- it was 3 before (4,2,1) was stocked, which is why the
    multiplet-map rate was zero.

    Reported per dtype because the two halves of the registry are stocked differently.
    """

    from ..varmap.tests import _random_config

    # NCONFIG is per dtype; dtype is not an argument to make_random() (on the gpu_valid path
    # it comes from the randomly chosen cdd2 registry key), so we draw 2*NCONFIG unfiltered
    # and BIN by what we got. That reports the float32 fraction for free, which is the sanity
    # check on the pool itself.
    #
    # The straddle row needs an SdPlan per config, which is much more expensive than a draw,
    # so it is measured on a SUBSAMPLE and reported against that count.
    NCONFIG, NSTRADDLE = 400, 120

    atomic_print(f'coverage: {NCONFIG} configs per dtype, drawn by'
                 f' varmap.tests._random_config() -- the same helper the tests use, so these'
                 f' rates describe the draws they actually make.\n')

    rng = np.random.default_rng()
    rows = {}
    for _ in range(NCONFIG * 2):
        config = _random_config(rng)
        rows.setdefault(str(np.dtype(config.dtype)), []).append(config)

    for dtype in sorted(rows):
        configs = rows[dtype]
        n = len(configs)
        npri = [int(c.num_primary_trees) for c in configs]
        varying = sum(1 for c in configs
                      if len(set(int(pt.max_width) for pt in c.primary_trees)) > 1)

        # (n_early, n_noncontiguous, n_pairs): the last two are only defined on configs that
        # HAVE an early trigger, so the non-contiguous rate is reported against n_early
        # rather than against n. A low absolute rate then reads as "few early triggers" or
        # "few of them clamp", which are different problems with different fixes.
        n_early, n_nc, n_pairs, n_nc_pairs = 0, 0, 0, 0
        n_r0, n_wide, n_multi_sub, n_blk, n_xdm = 0, 0, 0, 0, 0
        for config in configs:
            nets = [int(pt.num_early_triggers) for pt in config.primary_trees]
            t0 = _tree(config, config.dedispersion_tree_index(0, 0))
            fs = t0.frequency_subbands
            R, r = int(fs.pf_rank), int(t0.total_rank())
            nfreq = int(config.get_total_nfreq())
            n_r0 += int(R == 0)
            n_wide += int(nfreq < (1 << r))
            n_multi_sub += int(int(fs.N) > 1)
            n_blk += int(nfreq % 8 == 0)
            n_xdm += int(any(int(_tree(config, i).xdm_rank()) > 0
                             for i in range(int(config.num_dedispersion_trees))))
            if max(nets) == 0:
                continue
            n_early += 1
            found = False
            for g in range(int(config.num_primary_trees)):
                parent = _tree(config, config.dedispersion_tree_index(g, 0))
                for e in range(1, nets[g] + 1):
                    child = _tree(config, config.dedispersion_tree_index(g, e))
                    m_map = np.asarray(DedispersionTree.m_index_mapping(parent, child))
                    n_pairs += 1
                    if not np.array_equal(m_map, np.arange(m_map.size)):
                        n_nc_pairs += 1
                        found = True
            n_nc += int(found)

        sbc = set(tuple(int(x) for x in c.frequency_subband_counts) for c in configs)
        ranks = sorted(set(int(c.toplevel_tree_rank) for c in configs))
        n_str, n_str_tot = _straddle_rate(configs[:NSTRADDLE])

        print_separator(f'{dtype}  ({n} of {2*NCONFIG} draws)')
        _row('Max_width is primary-tree-dependent (implies npri > 1)', varying, n)
        _row('Some primary tree has early triggers', n_early, n)
        _row('... of those, a non-contiguous multiplet map', n_nc, n_early)
        atomic_print(f'    (over {n_pairs} (parent, child) pairs, {n_nc_pairs} non-contiguous)')
        _row('A straddled (channel, subband) entry', n_str, n_str_tot)
        _row('xdm_rank > 0 in some tree', n_xdm, n)
        _row('R == 0 (N = M = 1)', n_r0, n)
        _row('nfreq < 2^r (wide footprints)', n_wide, n)
        _row('N > 1 (coarse-graining group sizes differ)', n_multi_sub, n)
        _row('nfreq a multiple of 8 (blocking is bit-exact)', n_blk, n)
        atomic_print(f'    npri histogram: {_hist(npri)}')
        atomic_print(f'    toplevel_tree_rank range: {ranks[0]}..{ranks[-1]};'
                     f' {len(sbc)} distinct subband_counts drawn\n')


def _straddle_rate(configs):
    """(n_with_a_straddle, n_examined) over 'configs'.

    Its own function because it is the one statistic here that costs real work: it builds an
    SdPlan per config, which is orders of magnitude more than a draw, so the caller passes a
    subsample and the count is reported against that.
    """

    import contextlib
    import io

    from ..varmap.detrender_free import SdPlan

    n = 0
    for config in configs:
        with contextlib.redirect_stdout(io.StringIO()):
            n += int(int(SdPlan(config, init_sd_matrices=False).stats['n_straddled']) > 0)
    return n, len(configs)


def _tree(config, itree):
    """A DedispersionTree, with Dcore NOT taken from the cdd2 registry -- the multiplet map
    does not depend on it, and requiring the registry would make this report depend on which
    kernels the build compiled in a second, unrelated way."""

    return DedispersionTree(config, int(itree), Dcore_from_cdd2_registry=False)


def _row(label, count, n):
    pct = f'{100.0*count/n:5.1f}%' if (n > 0) else '   --'
    atomic_print(f'    {label + ":":<58s} {pct}   ({count}/{n})')


def _hist(values):
    from collections import Counter
    return {k: v for (k, v) in sorted(Counter(values).items())}
