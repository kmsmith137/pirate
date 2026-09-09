"""
Tests of DedispersionConfig::validate() and ::make_random(), run via 'test --dd'.

Covers the cases that random configs cannot exercise on their own.

  - test_primary_tree_chains(): the four per-primary-tree fields that validate() CHAINS across
    primary trees. max_width, wt_dm_downsampling and wt_time_downsampling must each equal their
    predecessor's value or be half of it; num_early_triggers must equal its predecessor's or be
    one more. Each rule and its justification are in validate()'s exception text; briefly, all
    four are quantities in a primary tree's OWN (2^i-downsampled) units, so the two legal steps
    are the two ways of scaling the underlying physical quantity deliberately.

    The POSITIVE side of these rules is covered elsewhere and is not repeated here beyond a
    handful of sanity chains: make_random() builds four legal chains on every draw (so every
    config the --dd and --amax tests generate is a positive case), and test_subband_property()
    in test_subbands.py loads every shipped config, all nine of which satisfy all four. What is
    left, and what this file is for, is the negative side: a config that breaks a rule must
    throw, and say which two primary trees disagree.

  - test_random_args_flags(): make_random()'s force_float32 / no_host_mega_ringbuf really do
    constrain what they promise. Both exist so that a random config is usable by the GPU
    brute-force variance-map sweep, which lives two packages away and is dispatched by a
    different flag -- so without this, a change to the key-selection logic could silently stop
    honouring them and only the varmap sweep tests would notice.
"""

import numpy as np

from ..pirate_pybind11 import DedispersionConfig
from ..utils import atomic_print


def _random_base(npri):
    """A random config with at least 'npri' primary trees, for the max_width chains.

    max_width is the one chained field with no per-tree bound to collide with (validate() asks
    only for a power of two in (0, max_pf_width]), so an arbitrary chain can be installed on an
    arbitrary config and the only rule left to trip is the chain rule. The other three fields
    have per-tree bounds that depend on toplevel_tree_rank -- at toplevel_tree_rank=2 a
    downsampled tree's wt_dm_downsampling is pinned to a single value -- so they use
    _chord_base() instead.

    Built from make_random() rather than a shipped config: configs/ is not packaged with
    pirate_frb, and a negative test should not be best-effort.  gpu_valid=False because no
    kernels are launched, and because the chain we install would not match a registry key
    anyway.

    ASKED FOR AS A MINIMUM AND THEN TRUNCATED by the caller, rather than drawn and rejected.
    make_random() honours min_primary_trees by construction, so this returns on the first draw;
    the old rejection loop wanted an exact count, which the unconstrained draw supplies about
    16% of the time for npri == 3. Truncation is safe because validate() only gets EASIER to
    satisfy as trees are dropped -- its 'min_nt' divisor falls by a factor of two per dropped
    tree, dropping to a single tree relaxes the pf_rank bound rather than tightening it, and a
    prefix of a legal chain is a legal chain.
    """

    return DedispersionConfig.make_random(max_toplevel_rank=8, max_early_triggers=2,
                                          gpu_valid=False, min_primary_trees=npri)


def _chord_base(npri):
    """A production-shaped config with at least 'npri' primary trees, for the other three chains.

    make_mini_chord() rather than make_random(), because those three fields DO have per-tree
    bounds and this test has to be sure it is tripping the chain rule and not one of them. At
    toplevel_tree_rank=16 the wt_* window is [16, 32768] at every primary tree, which leaves
    room for an illegal step in either direction; a random config can draw
    toplevel_tree_rank=2, where the window is a single point and no illegal chain is even
    expressible.

    The subband counts are replaced because make_mini_chord() has none (frequency_subband_counts
    = [0,0,0,0,1]), and validate() rejects ANY early trigger against a zero count at the level
    it truncates to -- which would mask the num_early_triggers chain rule behind a different
    exception on every legal chain.
    """

    assert npri <= 4
    config = DedispersionConfig.make_mini_chord(np.float32)
    config.frequency_subband_counts = [5, 9, 7, 3, 1]   # chord_sb2_et.yml's, pf_rank 4
    return config


def _config_with_chain(base, field, values):
    """'base', truncated to len(values) primary trees, with 'field' set to 'values'."""

    config = base(len(values))

    # config.primary_trees converts to a fresh python list, so mutate and assign back.
    pts = list(config.primary_trees)[:len(values)]
    for (pt, v) in zip(pts, values):
        setattr(pt, field, int(v))
    config.primary_trees = pts
    return config


# (field, base, legal chains, illegal chains as (values, ipri of the first offending step)).
#
# The max_width row's legal chains cover max_width=1, whose only legal successor is 1 (halving
# would give 0, which the per-primary-tree loop in validate() rejects), and its illegal chains
# cover stepping off that boundary.
_CHAIN_CASES = [
    ('max_width', _random_base,
     [[16, 16], [16, 8], [32, 16, 8, 4], [4, 4, 2, 1], [1, 1]],
     [([8, 16], 1),            # increasing
      ([16, 4], 1),            # decreasing, but not by a factor of two
      ([1, 2], 1),             # increasing off the max_width=1 boundary
      ([8, 8, 32], 2),         # legal first step, illegal second
      ([4, 2, 4], 2)]),        # halves, then increases back

    ('wt_dm_downsampling', _chord_base,
     [[64, 64], [64, 32], [128, 64, 32, 32], [32, 16]],
     [([64, 128], 1),
      ([64, 16], 1),
      ([64, 64, 16], 2),
      ([64, 32, 64], 2)]),

    ('wt_time_downsampling', _chord_base,
     [[64, 64], [64, 32], [128, 64, 32, 32], [32, 16]],
     [([64, 128], 1),
      ([64, 16], 1),
      ([64, 64, 16], 2),
      ([64, 32, 64], 2)]),

    # num_early_triggers steps UP by one rather than halving, so its illegal chains are the
    # mirror image: any decrease, and any increase by more than one.
    ('num_early_triggers', _chord_base,
     [[0, 0], [0, 1], [0, 1, 2, 3], [1, 1, 2, 2], [4, 4]],
     [([0, 2], 1),
      ([1, 0], 1),
      ([2, 0], 1),
      ([0, 1, 3], 2),
      ([0, 1, 1, 3], 3)]),
]


def test_primary_tree_chains():
    """A per-primary-tree chain that takes an illegal step must be rejected by validate()."""

    nlegal = nbad = 0

    for (field, base, legal, bad) in _CHAIN_CASES:
        for values in legal:
            _config_with_chain(base, field, values).validate()
        nlegal += len(legal)

        for (values, ipri) in bad:
            config = _config_with_chain(base, field, values)
            try:
                config.validate()
            except RuntimeError as e:
                # The message must be enough to find the typo without opening the source: it
                # names both primary trees and both values.
                msg = str(e)
                for s in [f"primary tree {ipri} has {field}={values[ipri]}",
                          f"primary tree {ipri-1} has {field}={values[ipri-1]}"]:
                    assert s in msg, (field, values, s, msg)
                continue
            raise AssertionError(f"DedispersionConfig.validate() should have thrown"
                                 f" ({field} chain {values})")
        nbad += len(bad)

    atomic_print(f"test_primary_tree_chains: {nlegal} legal and {nbad} illegal chains"
                 f" over {len(_CHAIN_CASES)} fields")


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
