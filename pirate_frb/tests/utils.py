"""Shared helpers for the loopback tests (test --net, test --serv)."""

import random

from ..pirate_pybind11 import DedispersionConfig


def make_random_subscale_config(min_batch_slots=1):
    """Return one random "subscale" DedispersionConfig for the loopback tests.

    "Subscale" is a size cap plus two protocol constraints, and this is the one
    place they are written down; test --net and test --serv both come here.

      - max_toplevel_rank=6.  These are loopback tests, so the config only has to
        be buildable.  The FLOOR is 5, not 1: make_random() needs
        max_stage2_rank = (max_toplevel_rank+1)/2 to reach 3, the smallest dd_rank
        the precompiled cdd2 registry stocks.
      - time_samples_per_chunk a multiple of 256, the network protocol's cadence.
      - beams_per_gpu <= 8, to keep the frame count manageable.

    'min_batch_slots' is the one knob a caller varies: pass 2 for a
    grouper-enabled FrbServer, which needs
    beams_per_gpu >= 2 * num_active_batches * beams_per_batch.

    All four are ARGUMENTS TO THE DRAW, not conditions checked afterwards, and
    that distinction matters more than it looks: they correlate with
    num_primary_trees and dtype, so a caller that drew and retried would silently
    end up testing a narrow slice of configs.  See RandomArgs in
    include/pirate/DedispersionConfig.hpp for the numbers.

    The config is built FIRST and the rest of the test params derived from it, so
    that the four metadata-dependent members (zone_nfreq, zone_freq_edges,
    time_sample_ms, beams_per_gpu) that the FrbServer's processing thread
    overwrites with XMD-derived values land on values matching the random config
    -- so config_postfilled.validate() trivially succeeds.
    """
    return DedispersionConfig.make_random(max_toplevel_rank = 6,
                                          tspc_multiple     = 256,
                                          max_beams_per_gpu = 8,
                                          min_batch_slots   = min_batch_slots)


def pick_receiver_worker_counts(total_nfreq):
    """Return random (num_receivers, nworkers) for a FakeXEngine test setup.

    FakeXEngine imposes two constraints on nworkers:
      - nworkers <= total_nfreq        (freq channels are assigned round-robin)
      - nworkers % num_receivers == 0  (ip_addrs distributed evenly across workers)
    Pick num_receivers and workers_per_receiver so both hold by construction,
    rather than clamping nworkers after-the-fact (which can break divisibility).
    """
    num_receivers        = random.randint(1, min(5, total_nfreq))
    workers_per_receiver = random.randint(1, min(5, total_nfreq // num_receivers))
    return num_receivers, workers_per_receiver * num_receivers
