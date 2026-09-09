
"""Small real entry-point checks for 1.5 benchmark producer metadata and parity."""
import csv
import hashlib
import json
from pathlib import Path

import pytest
import yaml


def _cuda():
    cp = pytest.importorskip("cupy")
    if cp.cuda.runtime.getDeviceCount() < 1:
        pytest.skip("no CUDA device")
    cp.cuda.Device(0).use()
    return cp


def _csv(path):
    with path.open(newline="") as stream:
        return list(csv.DictReader(stream))


def _notebook_cells(path, identifiers):
    notebook = json.loads(path.read_text())
    if identifiers is None:
        return [c for c in notebook["cells"] if c["cell_type"] == "code"][:1]
    return [c for c in notebook["cells"] if c.get("id") in identifiers]


def _check_notebook(module, results, identifiers, result_variable, monkeypatch):
    monkeypatch.setenv("MPLBACKEND", "Agg")
    monkeypatch.setenv(result_variable, str(results))
    monkeypatch.setenv("PIRATE_REPOSITORY_ROOT", str(Path(module.__file__).resolve().parents[1]))
    path = Path(module.__file__).with_name(Path(module.__file__).name.replace("benchmark_", "analyze_")).with_suffix(".ipynb")
    namespace = {"__name__": "__benchmark_notebook_check__"}
    for cell in _notebook_cells(path, identifiers):
        exec(compile("".join(cell["source"]), str(path), "exec"), namespace)
    return namespace


@pytest.mark.parametrize("variant", ["cpu", "gpu_representative"])
def test_gaussian_variant_actual_producer_metadata_and_reference_parity(variant, tmp_path, monkeypatch):
    _cuda()
    if variant == "cpu":
        from . import benchmark_gaussian_corruption_cpu_grouper_timing as benchmark
    else:
        from . import benchmark_gaussian_corruption_gpu_representative_grouper_timing as benchmark
    argv = [
        "--total-beams", "2", "--beam-batch-size", "2", "--trials", "1",
        "--corruption-percentages", "0", "0.001", "--dm-reach", "1",
        "--base-seed", "137", "--device", "0", "--results-dir", str(tmp_path),
    ]
    if variant == "cpu":
        argv.append("--verify-production-reference")
    benchmark.run_benchmark(benchmark._make_arg_parser().parse_args(argv))
    metadata = yaml.safe_load((tmp_path / "metadata.yaml").read_text())
    rows = _csv(tmp_path / "trials.csv")
    assert metadata["schema_version"] == benchmark.SCHEMA_VERSION == 2
    assert metadata["campaign_completeness"]["complete"]
    assert len(rows) == 2 and all(row["status"] == "completed" for row in rows)
    plan = metadata["plan"]
    payload = metadata["campaign_signature_payload"]
    assert plan["argmax_encoding"] == payload["argmax_encoding"] == "pirate-1.5:t8-p8-m8-mu8"
    assert plan["dcores"] == payload["dcores"] == [tree["dcore"] for tree in payload["token_geometry"]]
    assert len(plan["dcores"]) == 10
    assert hashlib.sha256(plan["producer_plan_yaml"].encode()).hexdigest() == payload["producer_plan_sha256"]
    assert int(rows[0]["total_peakfinder_candidates"]) == 0
    assert int(rows[1]["total_peakfinder_candidates"]) > 0
    parity_field = "production_reference_verified" if variant == "cpu" else "exact_parity_verified"
    for row in rows:
        assert int(row["total_decoded_candidates"]) == int(row["total_peakfinder_candidates"])
        assert int(row[parity_field]) == 1
    benchmark.validate_resumed_trial_rows(
        rows, benchmark.validate_arguments(benchmark._make_arg_parser().parse_args(argv)),
        metadata["campaign_id"], plan["chunk_duration_ms"],
    )
    _check_notebook(
        benchmark, tmp_path,
        {"load-and-validate", "validate-cpu-results"} if variant == "cpu" else None,
        "PIRATE_GAUSSIAN_CPU_GROUPER_RESULTS" if variant == "cpu" else "PIRATE_GPU_REPRESENTATIVE_GROUPER_RESULTS",
        monkeypatch,
    )


def test_concentrated_variant_exact_counts_metadata_and_parity(tmp_path, monkeypatch):
    _cuda()
    from . import benchmark_concentrated_map_grouper_timing as benchmark
    args = benchmark.build_parser().parse_args([
        "--layouts", "single_map", "single_family", "--hot-pixel-counts", "4",
        "--selected-tree", "2", "--repeats", "1", "--dm-reach", "1",
        "--base-seed", "137", "--results-dir", str(tmp_path),
    ])
    benchmark.run_benchmark(args)
    metadata = yaml.safe_load((tmp_path / "metadata.yaml").read_text())
    rows = _csv(tmp_path / "trials.csv")
    assert len(rows) == 2 and metadata["schema_version"] == 2
    producer = metadata["producer"]
    assert producer["argmax_encoding"] == "pirate-1.5:t8-p8-m8-mu8"
    assert producer["dcores"] == [tree["dcore"] for tree in producer["trees"]]
    assert hashlib.sha256(producer["plan_yaml"].encode()).hexdigest() == metadata["config"]["producer_plan_sha256"]
    for row in rows:
        assert int(row["raw_candidate_count"]) == int(row["decoded_candidate_count"]) == 4
        assert int(row["gpu_complete"]) == int(row["exact_parity_verified"]) == 1
        assert row["cpu_status"] == "completed"
    _check_notebook(benchmark, tmp_path, {"load"}, "PIRATE_CONCENTRATED_MAP_GROUPER_RESULTS", monkeypatch)


def test_kernel_comparison_records_producer_and_preserves_filter_equality(tmp_path):
    _cuda()
    from . import benchmark_full_band_peakfinder_kernel as benchmark
    result = benchmark.run(benchmark.build_parser().parse_args([
        "--total-beams", "1", "--beam-batch-size", "1", "--dm-reach", "1",
        "--warmup", "1", "--iterations", "1", "--base-seed", "137",
        "--results", str(tmp_path / "comparison.json"),
    ]))
    assert result["schema_version"] == 3
    assert result["all_filter_outputs_equal"] is True
    producer = result["producer"]
    assert producer["argmax_encoding"] == "pirate-1.5:t8-p8-m8-mu8"
    assert len(producer["dcores"]) == len(producer["trees"]) == 10
    assert producer["dcores"] == [tree["dcore"] for tree in producer["trees"]]
