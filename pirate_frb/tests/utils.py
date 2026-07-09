"""Shared helpers for the loopback tests (test --net, test --serv)."""

import random

from ..pirate_pybind11 import DedispersionConfig


def make_random_subscale_config(max_toplevel_rank=6, max_beams=8):
    """Return one random "subscale" DedispersionConfig for the loopback tests.

    The DedispersionConfig is built FIRST (via make_random + rejection
    sampling for test-friendly constraints), and the rest of the test
    params are derived from it. This way, the four metadata-dependent
    members of the config (zone_nfreq, zone_freq_edges, time_sample_ms,
    beams_per_gpu) that the FrbServer's processing thread overwrites
    with XMD-derived values land on values that match the random
    config -- so config_postfilled.validate() trivially succeeds.

    Rejection-sample constraints:
      - time_samples_per_chunk % 256 == 0 (network protocol cadence)
      - beams_per_gpu <= max_beams (keep frame count manageable for the test)

    max_toplevel_rank=6 is the smallest value compatible with the precompiled
    cdd2 kernel registry (which has dd_rank in {3,4,5}, requiring
    max_stage2_rank = (max_toplevel_rank+1)/2 >= 3, i.e. max_toplevel_rank >= 5).
    """
    for _ in range(200):
        config = DedispersionConfig.make_random(max_toplevel_rank=max_toplevel_rank)
        if config.time_samples_per_chunk % 256 != 0:
            continue
        if config.beams_per_gpu > max_beams:
            continue
        return config
    raise RuntimeError(
        f"make_random_subscale_config: failed to generate a random DedispersionConfig "
        f"satisfying (tsc%256==0 and beams_per_gpu<={max_beams}) in 200 attempts"
    )


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
