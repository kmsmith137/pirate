"""Kernel-selected Dcores retained across benchmark plan reconstruction."""
from . import benchmark_peakfinder_batch_timing as benchmark


def test_benchmark_records_actual_gpu_producer(monkeypatch):
    import cupy as cp
    import pirate_frb
    from pirate_frb import DedispersionConfig
    from pirate_frb.ArgmaxMetadata import ARGMAX_ENCODING
    from .producer_metadata import build_producer_plan

    original = pirate_frb.GpuDedisperser
    observed = []
    def capture(*args, **kwargs):
        producer = original(*args, **kwargs)
        observed.append(tuple(producer.Dcores))
        return producer
    monkeypatch.setattr(pirate_frb, "GpuDedisperser", capture)
    with cp.cuda.Device(0):
        bundle = benchmark.load_authoritative_plan(benchmark.DEFAULT_CONFIG)
        assert observed == [bundle.dcores]
        assert bundle.argmax_encoding == ARGMAX_ENCODING
        assert tuple(spec.shape for spec in bundle.specs) == benchmark.EXPECTED_SHAPES
        assert tuple(spec.dcore for spec in bundle.specs) == bundle.dcores
        assert all(spec.token_extra_dm >= 1 for spec in bundle.specs)
        inputs = benchmark.generate_clean_inputs(bundle.specs, 1, 137)
        decoded = benchmark.validate_tokens_with_plan(bundle.plan, inputs)
        assert len(decoded) == len(bundle.specs) == 10
        geometries, _ = benchmark.build_geometries(cp, bundle.plan, bundle.specs, (1,), 1)
        assert len(geometries[1]) == 10
        config = DedispersionConfig.from_yaml(str(benchmark.DEFAULT_CONFIG))
        descriptor = build_producer_plan(config)
        assert descriptor.dcores == observed[-1]
        assert descriptor.argmax_encoding == ARGMAX_ENCODING
        assert descriptor.plan_yaml == bundle.producer_plan_yaml
