"""
Tests of the frequency-subband restriction rule (run via 'test --sb').

Two tests, both cheap and GPU-free:

  - test_restrict_parity(): the C++ FrequencySubbands.restrict_subband_counts() and its
    python twin (pirate_frb.cuda_generator.FrequencySubbands) must agree, along with
    can_early_trigger(). They are two transcriptions of one rule: makefile_helper.py uses
    the python one to decide which kernels to COMPILE, while DedispersionTree uses the C++
    one to decide which kernel to ASK FOR. A divergence is silent at build time and
    surfaces much later as "Kernel not found in registry" for some config.

  - test_subband_property(): for every tree of a config, the set of frequency subbands
    searched must satisfy

        bands(ipri, iet)  <=  bands(ipri, 0)  ==  bands(0, 0)

    i.e. time-downsampling neither adds nor removes subbands, and an early trigger can
    remove subbands but never add one. Subbands are compared as TOPLEVEL tree-freq ranges
    (DedispersionTree.n_to_toplevel_{flo,fhi}), which is what makes trees of different rank
    comparable at all.

    The "never add" half is the one with teeth: it is why an early-trigger tree's own full
    band has to be a band the config already declares, which is what
    FrequencySubbands.can_early_trigger() and DedispersionConfig.validate() enforce.
"""

import os
import glob
import random
import itertools

from ..pirate_pybind11 import DedispersionConfig, DedispersionTree, FrequencySubbands
from ..cuda_generator import FrequencySubbands as PyFrequencySubbands
from ..utils import atomic_print


####################################################################################################
#
# C++ / python parity of the restriction rule.


def _max_bands(pf_rank, level):
    # Level 0 is special (non-overlapping bands).
    return (2**(pf_rank+1-level) - 1) if (level > 0) else 2**pf_rank


def _all_counts(pf_rank):
    """Yields every valid subband_counts vector with this pf_rank."""

    ranges = [ range(_max_bands(pf_rank, level) + 1) for level in range(pf_rank) ]
    for combo in itertools.product(*ranges):
        yield list(combo) + [1]


def _random_counts(pf_rank):
    """One random valid subband_counts vector with this pf_rank."""

    return [ random.randint(0, _max_bands(pf_rank, level)) for level in range(pf_rank) ] + [1]


def _compare(subband_counts, et_level):
    """Compares the two implementations for one (subband_counts, et_level) pair."""

    cpp_ok = FrequencySubbands.can_early_trigger(subband_counts, et_level)
    py_ok = PyFrequencySubbands.can_early_trigger(subband_counts, et_level)

    assert cpp_ok == py_ok, \
        f"can_early_trigger({subband_counts}, {et_level}): C++ says {cpp_ok}, python says {py_ok}"

    if not cpp_ok:
        # Both must refuse. (A silent difference here would be as bad as a different
        # result: it decides which early-trigger levels a config may declare.)
        for name, fn in [('C++', FrequencySubbands), ('python', PyFrequencySubbands)]:
            try:
                fn.restrict_subband_counts(subband_counts, et_level)
            except RuntimeError:
                continue
            raise AssertionError(f"{name} restrict_subband_counts({subband_counts}, {et_level}) "
                                 f"should have thrown (can_early_trigger is False)")
        return

    cpp = list(FrequencySubbands.restrict_subband_counts(subband_counts, et_level))
    py = list(PyFrequencySubbands.restrict_subband_counts(subband_counts, et_level))

    assert cpp == py, \
        f"restrict_subband_counts({subband_counts}, {et_level}): C++ gives {cpp}, python gives {py}"

    # Two invariants of the rule itself, cheap to check here and independent of the two
    # implementations agreeing with each other.
    assert len(cpp) == len(subband_counts) - et_level, \
        f"restrict_subband_counts({subband_counts}, {et_level}) = {cpp}: wrong pf_rank"

    for level, count in enumerate(cpp):
        assert count <= subband_counts[level], \
            f"restrict_subband_counts({subband_counts}, {et_level}) = {cpp}: level {level} " \
            f"gained bands, which would ADD a subband to the search"


def test_restrict_parity(nrandom=300):
    """C++ vs python restrict_subband_counts() / can_early_trigger()."""

    npairs = 0

    # Exhaustive at pf_rank <= 3 (312 counts vectors).
    for pf_rank in range(4):
        for subband_counts in _all_counts(pf_rank):
            # et_level up to pf_rank+1, so the out-of-range case is covered too.
            for et_level in range(pf_rank + 2):
                _compare(subband_counts, et_level)
                npairs += 1

    # Sampled at pf_rank == 4 (8704 counts vectors, too many to sweep every iteration).
    for _ in range(nrandom):
        subband_counts = _random_counts(4)
        for et_level in range(6):
            _compare(subband_counts, et_level)
            npairs += 1

    atomic_print(f'test_restrict_parity: {npairs} (subband_counts, et_level) pairs agree')


####################################################################################################
#
# Property (*): bands(ipri,iet) <= bands(ipri,0) == bands(0,0).


def _tree_bands(tree):
    """Set of (flo, fhi) toplevel tree-freq ranges searched by 'tree' (fhi EXCLUSIVE)."""

    fs = tree.frequency_subbands
    return { (tree.n_to_toplevel_flo(n), tree.n_to_toplevel_fhi(n)) for n in range(fs.N) }


def _check_subband_property(config, label):
    bands = {}

    for itree in range(config.num_dedispersion_trees):
        # Dcore_from_cdd2_registry=False: subband geometry is pure arithmetic, and we do not
        # want to require that every tree's cdd2 kernel is compiled into this build.
        tree = DedispersionTree(config, itree, Dcore_from_cdd2_registry=False)
        bands[(tree.primary_tree_index, tree.early_trigger_level)] = _tree_bands(tree)

    base = bands[(0,0)]

    for (ipri, iet), b in sorted(bands.items()):
        if iet == 0:
            assert b == base, \
                f"{label}: tree (ipri={ipri}, iet=0) searches {sorted(b)}, but tree (0,0) " \
                f"searches {sorted(base)}. Time-downsampling must not change the subband set."
        else:
            main = bands[(ipri,0)]
            assert b <= main, \
                f"{label}: tree (ipri={ipri}, iet={iet}) searches {sorted(b - main)}, which " \
                f"tree (ipri={ipri}, iet=0) does not. An early trigger may remove subbands, " \
                f"but must never add one."


def test_subband_property(nrandom=8):
    """Property (*) on random configs, plus the shipped configs if they are reachable."""

    # gpu_valid=False: this test needs no kernels, and the !gpu_valid path generates deeper
    # early-trigger ladders (it is not filtered by which cdd2 kernels happen to be compiled),
    # which is exactly where the property has content.
    for i in range(nrandom):
        config = DedispersionConfig.make_random(max_toplevel_rank=12, gpu_valid=False)
        _check_subband_property(config, f'make_random() #{i}')

    # The shipped configs, when the test is run from a source checkout. (pirate_frb does not
    # package configs/, so this is best-effort extra coverage rather than a requirement.)
    paths = sorted(glob.glob('configs/dedispersion/*.yml'))

    for path in paths:
        config = DedispersionConfig.from_yaml(path)
        _check_subband_property(config, os.path.basename(path))

    atomic_print(f'test_subband_property: {nrandom} random configs + {len(paths)} shipped configs')


def run_all():
    test_restrict_parity()
    test_subband_property()
