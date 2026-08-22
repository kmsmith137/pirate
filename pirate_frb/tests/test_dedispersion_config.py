"""
Tests of DedispersionConfig::validate() rules that random configs cannot exercise (run via
'test --dd').

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
"""

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
