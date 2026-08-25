"""Coverage analysis of the randomization utilities the unit tests draw from.

NOT A TEST: nothing here asserts. It is a diagnostic, reached as 'pirate_frb coverage', and
it answers one question -- how often does a randomized unit test actually get the structure
it needs? Several tests in pirate_frb.varmap draw configs from
DedispersionConfig::make_random() and REPORT rather than assert what they covered, precisely
because a fair draw misses some cases; this is where you find out what those rates are.
"""

import numpy as np

from ..pirate_pybind11 import DedispersionConfig
from ..utils import atomic_print, print_separator


def report_coverage():
    """How often does make_random() produce the structures a unit test needs?

    The number this exists for is the SECOND row: a max_width chain that is not flat. Several
    tests want a config whose profile count nprofiles = 1 + 3*log2(max_width) differs across
    primary trees, and on the gpu_valid path that can only happen when the cdd2 registry
    stocks two Wmax values for the DOWNSAMPLED tree's (dtype, dd_rank, subband_counts) --
    see DedispersionConfig::make_random(). It is therefore a property of which kernels this
    build compiled, not of the random draw, and it moves when makefile_helper.py changes.

    Reported per dtype because the two halves of the registry are stocked differently.

    DedispersionConfig::make_random() is the only randomization utility covered so far; this
    module is named for the general question because that is where a second one would go.
    """

    # NO FLAGS: the numbers only mean anything at one setting, and this is it. GPU_VALID is
    # the load-bearing one -- it restricts the draw to configs whose cdd2 kernels this build
    # actually compiled, which is what makes the max_width row a property of the REGISTRY
    # rather than of the random draw. 1000 per dtype puts the sampling error on each
    # percentage at a point or two, which is the resolution the numbers are read at. They
    # are echoed below so a pasted report says what produced it.
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
        widths = [[int(pt.max_width) for pt in c.primary_trees] for c in configs]
        nets = [max(int(pt.num_early_triggers) for pt in c.primary_trees) for c in configs]

        multi = sum(1 for w in widths if len(w) > 1)
        varying = sum(1 for w in widths if len(set(w)) > 1)
        et = sum(1 for x in nets if x > 0)
        ranks = sorted(set(int(c.toplevel_tree_rank) for c in configs))

        print_separator(f'{dtype}  ({n} of {2*NCONFIG} draws)')
        _row('Multiple primary trees (npri > 1)', multi, n)
        _row('Max_width is primary-tree-dependent (implies npri > 1)', varying, n)
        _row('Some primary tree has early triggers', et, n)
        atomic_print(f'    toplevel_tree_rank range: {ranks[0]}..{ranks[-1]}')
        atomic_print(f'    npri histogram: {_hist([len(w) for w in widths])}')
        atomic_print(f'    max_width chains seen: {_chains(widths)}\n')


def _row(label, count, n):
    atomic_print(f'    {label + ":":<58s} {100.0*count/n:5.1f}%   ({count}/{n})')


def _hist(values):
    from collections import Counter
    return {k: v for (k, v) in sorted(Counter(values).items())}


def _chains(widths, nshow=6):
    """The most common max_width chains, longest-first, so a flat registry is visible."""
    from collections import Counter
    c = Counter(tuple(w) for w in widths if len(w) > 1)
    if not c:
        return 'none (every config had one primary tree)'
    top = c.most_common(nshow)
    out = ', '.join(f'{list(k)} x{v}' for (k, v) in top)
    return out + (f', ... ({len(c)} distinct)' if len(c) > nshow else '')
