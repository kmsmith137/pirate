"""Explicit producer geometry for newly generated benchmark inputs.

Saved acquisitions must use their recorded producer metadata instead. This helper
selects kernels for NEW synthetic benchmark inputs, outside measured intervals.
"""

from dataclasses import dataclass
from typing import Any

ARGMAX_ENCODING = "pirate-1.5:t8-p8-m8-mu8"


@dataclass(frozen=True)
class ProducerPlan:
    config: Any
    plan: Any
    config_yaml: str
    plan_yaml: str
    dcores: tuple[int, ...]
    argmax_encoding: str


def build_producer_plan(config, *, cuda_device_id=None):
    """Select producer kernels and reconstruct the corresponding consumer plan.

    No acquisition buffers are allocated and no dedispersion is run. Dcores are
    read from the constructed GPU producer, never inferred from serialized YAML.
    The caller's current GPU is used unless a device is explicitly supplied.
    """
    import cupy as cp
    from pirate_frb import DedispersionPlan, GpuDedisperser
    from pirate_frb.core import CudaStreamPool
    from pirate_frb.ArgmaxMetadata import ARGMAX_ENCODING as supported_encoding, read_argmax_metadata

    if ARGMAX_ENCODING != supported_encoding:
        raise ValueError("benchmark and runtime argmax encodings disagree")
    device = cp.cuda.runtime.getDevice() if cuda_device_id is None else cuda_device_id
    with cp.cuda.Device(device):
        producer_plan = DedispersionPlan(config)
        streams = CudaStreamPool(int(config.num_active_batches))
        producer = GpuDedisperser(producer_plan, streams, cuda_device_id=device,
                                  num_consumers=1)
        dcores = read_argmax_metadata(
            {"argmax_encoding": ARGMAX_ENCODING, "dcores": list(producer.Dcores)},
            ntrees=int(producer_plan.ntrees),
            douts=[int(t.nt_ds) // int(t.nt_out) for t in producer_plan.trees],
        )
        config_yaml, plan_yaml = config.to_yaml_string(), producer_plan.to_yaml_string()
        consumer = DedispersionPlan.from_yaml_string(config, plan_yaml)
    return ProducerPlan(config, consumer, config_yaml, plan_yaml, tuple(dcores), ARGMAX_ENCODING)
