"""
Tests of DedispersionConfig::validate() and ::make_random() that random configs cannot
exercise on their own (run via 'test --dd').

  - test_max_width_monotone(): a downsampled primary tree's max_width must equal its
    predecessor's or be half of it. The rule and its justification are in validate()'s
    exception text; briefly, max_width is in the primary tree's OWN time samples, so those are
    the two ways of scaling the search window deliberately (holding it fixed after downsampling
    or before), and the resulting monotonicity of nprofiles = 1 + 3*log2(max_width) is what
    makes every downsampled tree's profile set a subset of tree 0's -- see notes/variance_map.tex,
    appendix "Variance maps of a config's trees are row-restrictions of one another".

    The POSITIVE side of this rule is covered elsewhere and is not repeated here: make_random()
    builds a legal chain (so every config the --dd and --amax tests generate is a positive case),
    and test_subband_property() in test_subbands.py loads every shipped config, all nine of which
    are halving chains. What is left, and what this file is for, is the negative side: a config
    that breaks the rule must throw, and say which two primary trees disagree.

  - test_random_args_flags(): make_random()'s force_float32 / no_host_mega_ringbuf really do
    constrain what they promise. Both exist so that a random config is usable by the GPU
    brute-force variance-map sweep, which lives two packages away and is dispatched by a
    different flag -- so without this, a change to the key-selection logic could silently stop
    honouring them and only --vmbf would notice.
"""

import numpy as np

from ..pirate_pybind11 import DedispersionConfig
from ..utils import atomic_print


def _config_with_widths(widths):
    """A random config with len(widths) primary trees, whose max_width chain is 'widths'.

    Built from make_random() rather than a shipped config: configs/ is not packaged with
    pirate_frb, and a negative test should not be best-effort. gpu_valid=False because no
    kernels are launched, and because the chain we install would not match a registry key
    anyway.
    """

    for _ in range(400):
        config = DedispersionConfig.make_random(max_toplevel_rank=8, max_early_triggers=2,
                                                gpu_valid=False)
        if len(config.primary_trees) != len(widths):
            continue

        # config.primary_trees converts to a fresh python list, so mutate and assign back.
        pts = list(config.primary_trees)
        for (pt, w) in zip(pts, widths):
            pt.max_width = int(w)
        config.primary_trees = pts
        return config

    raise RuntimeError(f"test_dedispersion_config: make_random() produced no config with"
                       f" {len(widths)} primary trees in 400 attempts")


def test_max_width_monotone():
    """A max_width chain that is neither flat nor halving must be rejected by validate()."""

    # Legal chains: flat, halving, and a mixture. The last two entries of the mixed chain also
    # cover max_width=1, whose only legal successor is 1 (halving would give 0, which the
    # per-primary-tree loop in validate() rejects).
    for widths in [[16, 16], [16, 8], [32, 16, 8, 4], [4, 4, 2, 1], [1, 1]]:
        _config_with_widths(widths).validate()

    # Illegal chains. Each is (widths, ipri of the first offending step).
    bad = [([8, 16], 1),          # increasing
           ([16, 4], 1),          # decreasing, but not by a factor of two
           ([1, 2], 1),           # increasing off the max_width=1 boundary
           ([8, 8, 32], 2),       # legal first step, illegal second
           ([4, 2, 4], 2)]        # halves, then increases back

    for (widths, ipri) in bad:
        config = _config_with_widths(widths)
        try:
            config.validate()
        except RuntimeError as e:
            # The message must be enough to find the typo without opening the source: it names
            # both primary trees and both max_width values.
            msg = str(e)
            for s in [f"primary tree {ipri} has max_width={widths[ipri]}",
                      f"primary tree {ipri-1} has max_width={widths[ipri-1]}"]:
                assert s in msg, (widths, s, msg)
            continue
        raise AssertionError(f"DedispersionConfig.validate() should have thrown"
                             f" (max_width chain {widths})")

    atomic_print(f"test_max_width_monotone: 5 legal and {len(bad)} illegal max_width chains")


def test_random_args_flags(ndraw=4):
    """make_random()'s force_float32 and no_host_mega_ringbuf, on both draw paths."""

    for gpu_valid in [True, False]:
        for _ in range(ndraw):
            config = DedispersionConfig.make_random(max_toplevel_rank=8, max_early_triggers=2,
                                                    gpu_valid=gpu_valid, force_float32=True,
                                                    no_host_mega_ringbuf=True)
            config.validate()

            # force_float32 FILTERS the candidate cdd2 keys rather than patching ret.dtype
            # afterwards, because later code re-derives quantities from the (key, dtype) pair.
            # So a float16 config here means the filter was bypassed, not that a patch was
            # missed -- and the validate() above is what would catch the desynchronization a
            # patch would cause.
            assert np.dtype(config.dtype) == np.float32, (gpu_valid, np.dtype(config.dtype))

            # 10000 is the member's own default, i.e. "no limit, pure-GPU ring buffer".
            assert config.max_gpu_clag == 10000, (gpu_valid, config.max_gpu_clag)

    atomic_print(f'test_random_args_flags: {2*ndraw} draws, all float32 with'
                 ' max_gpu_clag=10000')
