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
    """How often does make_random() produce the structures a unit test needs?

    Three rates, each with a named consumer, and two structural diagnostics that exist to
    explain a rate when it moves.

      - A max_width chain that is NOT flat. Several tests want a config whose profile count
        nprofiles = 1 + 3*log2(max_width) differs across primary trees, so that the profile
        restriction P_gamma < P_0 is not a no-op. On the gpu_valid path this can only happen
        when the cdd2 registry stocks two Wmax values for the DOWNSAMPLED tree's (dtype,
        dd_rank, subband_counts) -- see DedispersionConfig::make_random(). It is therefore a
        property of which kernels this build compiled, not of the random draw.

      - A primary tree with early triggers. Without one, nothing exercises Proposition 1 of
        notes/variance_map.tex: the early-trigger trees, restrict_fine_vector(), and
        VarianceMultiMap.apply_fine()'s expansion.

      - An early-trigger tree whose MULTIPLET MAP IS NON-CONTIGUOUS, i.e. not the prefix
        [0..M_c). This is the only case in which a wrong gather in restrict_fine_vector() is
        visible at all -- with a prefix map, an off-by-one moves nothing. It needs
        restrict_subband_counts() to clamp a level that still has populated levels ABOVE it;
        see point 3 of the long comment in makefile_helper.py, which is where the subband
        vectors that can do this are stocked. This rate was 0% on the gpu_valid path until
        (4,2,1) was added there.

    Reported per dtype because the two halves of the registry are stocked differently.

    DedispersionConfig::make_random() is the only randomization utility covered so far; this
    module is named for the general question because that is where a second one would go.
    """

    # NO FLAGS: the numbers only mean anything at one setting, and this is it. GPU_VALID is
    # the load-bearing one -- it restricts the draw to configs whose cdd2 kernels this build
    # actually compiled, which is what makes the max_width and multiplet-map rows properties
    # of the REGISTRY rather than of the random draw. 1000 per dtype puts the sampling error
    # on each percentage at a point or two, which is the resolution the numbers are read at.
    # They are echoed below so a pasted report says what produced it.
    NCONFIG, MAX_TOPLEVEL_RANK, MAX_EARLY_TRIGGERS, GPU_VALID = 1000, 10, 5, True

    atomic_print(f'coverage: {NCONFIG} configs per dtype, gpu_valid={GPU_VALID},'
                 f' max_toplevel_rank={MAX_TOPLEVEL_RANK},'
                 f' max_early_triggers={MAX_EARLY_TRIGGERS}\n')

    # dtype is not an argument to make_random(): on the gpu_valid path it comes from the
    # randomly chosen cdd2 registry key. force_float32 filters the key pool, and there is no
    # 'force_float16', so we draw unfiltered and BIN by the dtype we got. That also reports
    # the float32 fraction for free, which is the sanity check on the pool itself.
    rows = {}
    for _ in range(NCONFIG * 2):
        config = DedispersionConfig.make_random(max_toplevel_rank=MAX_TOPLEVEL_RANK,
                                                max_early_triggers=MAX_EARLY_TRIGGERS,
                                                gpu_valid=GPU_VALID)
        dtype = str(np.dtype(config.dtype))
        rows.setdefault(dtype, []).append(config)

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
        for config in configs:
            nets = [int(pt.num_early_triggers) for pt in config.primary_trees]
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

        print_separator(f'{dtype}  ({n} of {2*NCONFIG} draws)')
        _row('Max_width is primary-tree-dependent (implies npri > 1)', varying, n)
        _row('Some primary tree has early triggers', n_early, n)
        _row('... of those, a non-contiguous multiplet map', n_nc, n_early)
        atomic_print(f'    (over {n_pairs} (parent, child) pairs, {n_nc_pairs} non-contiguous)')
        atomic_print(f'    npri histogram: {_hist(npri)}')
        atomic_print(f'    toplevel_tree_rank range: {ranks[0]}..{ranks[-1]};'
                     f' {len(sbc)} distinct subband_counts drawn\n')


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
