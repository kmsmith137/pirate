#!/usr/bin/env python3
"""Spot test: tree gridding, pirate vs bonsai.

Both codes begin dedispersion by rebinning frequency channels onto "tree" channels equally
spaced in freq^(-2), so that a dispersed pulse becomes a straight line. The tree-channel
edges are identical in the two codes. The one place the rebinnings could differ is how a
frequency channel that straddles a tree-channel edge is split between the two tree
channels: bonsai splits it in proportion to extent in freq^(-2), and pirate's
DedispersionConfig::make_channel_map() places each tree edge inside its frequency channel
by that same rule, so the two operations should agree to float32 roundoff. This test is
what says they do.

Compares pirate_frb.kernels.ReferenceTreeGriddingKernel, on the channel map that
DedispersionConfig.make_channel_map() produces, against bonsai::gridding's production
(AVX2) path, constructed with beta_fid = 0 and nups = 1 as in every CHIME production
config. The pirate GPU kernel is not run here: 'pirate_frb test --gtgk' ties it to the
reference kernel on random channel maps, including CHORD-scale ones.

The driver also returns bonsai's own unoptimized reference path, and the test compares
that against the production path with the SAME tolerance. That is the calibration: it
shows how far two implementations of the same weights sit apart, which is what "within
roundoff" has to mean for pirate vs bonsai.

NEGATIVE CONTROL. If make_channel_map() instead placed each tree edge linearly in
FREQUENCY within its channel (splitting a straddling channel in proportion to bandwidth),
this test fails on the CHIME geometry with a worst disagreement of 1.7e-5 x max|in|
against the 2e-6 x max|in| tolerance (measured), and by 4e-3 to 3e-2 on the random
geometries, whose channels are far wider. That is the split-rule discrepancy -- per
weight, up to (3/8) x (fractional channel width) = 2.3e-5 at 400 MHz -- so the
tolerance below can tell the two rules apart. Bonsai's own reference-vs-production
figure is 1e-7 x max|in| either way.

Needs a built oldpipe (misc/chimefrb/build_oldpipe.sh), and a CUDA device:
make_channel_map() allocates through cudaHostAlloc even though no GPU kernel runs. Run me
directly, or through misc/chimefrb/spot_checks/run_spot_tests.py.
"""

import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import harness

from pirate_frb import DedispersionConfig
from pirate_frb.kernels import ReferenceTreeGriddingKernel

HERE = os.path.dirname(os.path.abspath(__file__))
PIRATE = os.path.abspath(os.path.join(HERE, "..", "..", "..", ".."))     # the repo root
CHIME_CONFIG = os.path.join(PIRATE, "configs", "dedispersion", "chime.yml")

# Any fixed value works; it is written down so the comparison is reproducible without
# committing the arrays it generates.
SEED = 137

# A multiple of 16 for bonsai (constants::floats_per_cache_line) and of 32 for pirate's
# fp32 reference kernel (ntime % (1024/nbits) == 0). The CHIME input is then
# (16384, 32) float32, 2 MB.
NTIME = 32

# Random single-zone geometries, beyond the CHIME one. Kept to cases where a tree channel
# meets at most a few frequency channels, which is what the tolerance argument assumes:
# the widest tree channel (at the top of the band) is rho*(rho+1)/2 * nfreq/tree_size
# frequency channels wide, rho = freq_hi/freq_lo, so rho <= 3 and tree_size >= 2*nfreq
# keeps that at or below 3. Bonsai cannot do zones, so single-zone is the whole common
# domain.
NDRAW = 4

# The tolerance is ABSOLUTE and scaled to the input, with no relative part: a relative
# comparison is meaningless where an output happens to be near zero.
#
# Both codes form each weight in double and round it to float32. The two doubles are
# the same real number computed two ways (pirate: a difference of two channel_map
# entries in fractional-channel units; bonsai: a ratio of differences in tree-index
# units), agreeing to ~1e-11, far inside a float32 ulp of 6e-8; so the float32 weights
# are identical except when a double lies within that of a rounding boundary, and then
# differ by one ulp. Each output element is then a float32 sum of at most three or four
# weighted inputs, accumulated in opposite channel orders by the two codes: a few ulps
# of max|in|, i.e. a few times 6e-8 x max|in|. 2e-6 is ~10x above that, and ~10x below
# the split-rule discrepancy described in the module docstring.
ATOL_PER_MAX = 2.0e-6


def chime_geometry():
    """The CHIME production geometry: 16384 channels over 400-800 MHz, 32768 tree channels.

    Read from configs/dedispersion/chime.yml (toplevel_tree_rank = 15), which matches the
    top tree of every CHIME production bonsai config (ch_frb_l1/bonsai_configs). This
    geometry exercises both regimes in one array: at 800 MHz a tree channel is 1.5
    frequency channels wide, at 400 MHz it is 0.19.
    """
    config = DedispersionConfig.from_yaml(CHIME_CONFIG)
    return ("chime", config)


def random_geometry(rng, i):
    """A bare DedispersionConfig with only the three fields make_channel_map() reads."""
    rank = int(rng.integers(5, 10))                       # tree_size = 32 .. 512
    nfreq = int(rng.integers(8, (1 << rank) // 2 + 1))    # tree_size >= 2 * nfreq
    freq_lo = float(rng.uniform(300.0, 800.0))
    freq_hi = freq_lo * float(rng.uniform(1.5, 3.0))      # rho = freq_hi/freq_lo <= 3

    config = DedispersionConfig()
    config.zone_nfreq = [nfreq]
    config.zone_freq_edges = [freq_lo, freq_hi]
    config.toplevel_tree_rank = rank
    return ("random %d" % i, config)


def check(t, rng, label, config):
    nfreq = int(config.get_total_nfreq())
    tree_size = 1 << int(config.toplevel_tree_rank)
    freq_lo = float(config.zone_freq_edges[0])
    freq_hi = float(config.zone_freq_edges[-1])
    label = "%s (nfreq=%d, tree_size=%d, %.0f-%.0f MHz)" % (label, nfreq, tree_size, freq_lo, freq_hi)

    # Pirate's channel order: row 0 is the BOTTOM of the band.
    x = rng.standard_normal((nfreq, NTIME)).astype(np.float32)
    atol = ATOL_PER_MAX * float(np.max(np.abs(x)))

    # The pirate side. The tree channels come out in the same order in both codes (index 0
    # at the top of the band), so the outputs compare directly.
    cm = config.make_channel_map()
    kernel = ReferenceTreeGriddingKernel(nfreq=nfreq, nchan=tree_size, ntime=NTIME,
                                         beams_per_batch=1, channel_map=cm)
    new = np.asarray(kernel.apply(x[None]))[0]

    # The old side, with the channel-order flip: bonsai's row 0 is the TOP of the band.
    (old_prod, old_ref) = harness.run_driver(
        HERE, x[::-1], dtype=np.float32, noutputs=2,
        params={"freq_lo": repr(freq_lo), "freq_hi": repr(freq_hi), "tree_size": tree_size})

    # check_allclose() reports the disagreement relative to |want|, which says little where
    # an output is near zero. The number that matters -- and the one the module docstring
    # records for the negative control -- is the worst absolute disagreement in units of
    # max|in|.
    def scaled(a, b):
        return float(np.max(np.abs(a - b))) / float(np.max(np.abs(x)))

    t.note("%s: worst |pirate - bonsai| = %.3g x max|in|, worst |bonsai ref - prod| = %.3g x max|in|"
           % (label, scaled(new, old_prod), scaled(old_ref, old_prod)))

    ok = t.check_allclose(label + " pirate vs bonsai", new, old_prod, rtol=0.0, atol=atol,
                          why="same double weights rounded to float32, summed in float32 in"
                              " opposite channel orders: a few ulps of max|in| (see ATOL_PER_MAX)")

    ok &= t.check_allclose(label + " bonsai ref vs prod", old_ref, old_prod, rtol=0.0, atol=atol,
                           why="calibration: bonsai's two implementations of the same weights,"
                               " same tolerance")
    return ok


def main():
    t = harness.Test("tree_gridding")

    manifest = harness.build_manifest()
    if manifest:
        for line in manifest.splitlines():
            if line.startswith("date:") or line.startswith("arch:"):
                t.note(line.strip())

    rng = np.random.default_rng(SEED)
    t.note("ntime = %d, inputs Gaussian from seed %d" % (NTIME, SEED))

    (label, config) = chime_geometry()
    check(t, rng, label, config)

    for i in range(NDRAW):
        (label, config) = random_geometry(rng, i)
        check(t, rng, label, config)

    return t.done()


if __name__ == "__main__":
    sys.exit(main())
