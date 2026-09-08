"""Focused tests for the benchmark-only CPU candidate grouper.

The host grouping algorithm is exercised without CuPy wherever possible.  One
small optional integration test compares every reconstructed GPU grouping field
against the active production grouper when PIRATE and a CUDA device are
available; it is deliberately not a benchmark campaign.
"""

from __future__ import annotations

import csv
from dataclasses import fields
import json
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest
import yaml

from . import benchmark_gaussian_corruption_cpu_grouper_timing as benchmark
from . import benchmark_gaussian_corruption_timing as gaussian
from .cpu_candidate_grouper import (
    HOST_COLUMN_NAMES,
    CpuGroupingDiagnostics,
    assert_gpu_grouping_results_equal,
    group_candidates_on_cpu,
    group_host_columns,
    host_processing_order,
)


_HOST_DTYPES = {
    "beam_id": np.int32,
    "primary_tree_index": np.int32,
    "tree": np.int32,
    "source_chunk_index": np.int64,
    "idm": np.int32,
    "itime": np.int32,
    "argmax_token": np.uint32,
    "snr": np.float64,
    "dm": np.float64,
    "toa_sample_abs": np.float64,
    "dm_step": np.float64,
    "time_step_samples": np.float64,
}
TEST_CHUNK_DURATION_MS = 2048.0


def _host_columns(rows):
    """Build exact host grouping columns from compact dictionaries."""

    defaults = {
        "beam_id": 7,
        "primary_tree_index": 0,
        "tree": 0,
        "source_chunk_index": 0,
        "idm": 0,
        "itime": 0,
        "argmax_token": 0,
        "snr": 20.0,
        "dm": 100.0,
        "toa_sample_abs": 1000.0,
        "dm_step": 1.0,
        "time_step_samples": 1.0,
    }
    return {
        name: np.asarray(
            [row.get(name, defaults[name]) for row in rows], dtype=dtype
        )
        for name, dtype in _HOST_DTYPES.items()
    }


def _host_geometry():
    return SimpleNamespace(
        residual_slope_lo_samples_per_dm=0.0,
        residual_slope_hi_samples_per_dm=2.0,
    )


def _host_config():
    return SimpleNamespace(dm_tolerance_bins=1.5, time_padding_bins=1.0)


def test_host_grouping_empty_singleton_and_multi_member_semantics():
    empty = group_host_columns(
        _host_columns([]), _host_geometry(), _host_config()
    )
    assert empty.processing_order.dtype == np.int64
    assert empty.candidate_event_id.tolist() == []
    assert empty.representatives.tolist() == []
    assert empty.member_count.dtype == np.int32
    assert empty.member_order.tolist() == []
    assert empty.partition_count == empty.largest_partition == 0

    singleton = group_host_columns(
        _host_columns([{"tree": 2, "snr": 30.0}]),
        _host_geometry(),
        _host_config(),
    )
    assert singleton.processing_order.tolist() == [0]
    assert singleton.candidate_event_id.tolist() == [0]
    assert singleton.representatives.tolist() == [0]
    assert singleton.member_count.tolist() == [1]
    assert singleton.member_order.tolist() == [0]
    assert singleton.partition_count == singleton.largest_partition == 1

    # Two equal-S/N candidates from tree 1 are both compatible with the seed.
    # Exact priority selects smaller idm (row 2); row 1 must seed a later event.
    columns = _host_columns([
        {"tree": 0, "snr": 30.0},
        {"tree": 1, "snr": 25.0, "idm": 5, "dm": 100.2},
        {"tree": 1, "snr": 25.0, "idm": 4, "dm": 100.1},
        {"tree": 3, "snr": 23.0, "dm": 99.9},
    ])
    grouped = group_host_columns(columns, _host_geometry(), _host_config())
    assert grouped.processing_order.tolist() == [0, 2, 1, 3]
    assert grouped.candidate_event_id.tolist() == [0, 1, 0, 0]
    assert grouped.representatives.tolist() == [0, 1]
    assert grouped.member_count.tolist() == [3, 1]
    assert grouped.member_order.tolist() == [0, 2, 3, 1]
    assert grouped.partition_count == 1
    assert grouped.largest_partition == 4


def test_host_grouping_isolates_beam_and_primary_tree_family():
    columns = _host_columns([
        {"beam_id": 1, "primary_tree_index": 0, "tree": 0, "snr": 30.0},
        {"beam_id": 2, "primary_tree_index": 0, "tree": 1, "snr": 29.0},
        {"beam_id": 1, "primary_tree_index": 1, "tree": 2, "snr": 28.0},
        {"beam_id": 1, "primary_tree_index": 0, "tree": 1, "snr": 27.0},
    ])
    grouped = group_host_columns(columns, _host_geometry(), _host_config())
    assert grouped.candidate_event_id.tolist() == [0, 1, 2, 0]
    assert grouped.representatives.tolist() == [0, 1, 2]
    assert grouped.member_count.tolist() == [2, 1, 1]
    assert grouped.member_order.tolist() == [0, 3, 1, 2]
    assert grouped.partition_count == 3
    assert grouped.largest_partition == 2


def test_host_processing_order_matches_every_priority_key_and_stable_tie():
    common = {
        "snr": 20.0,
        "beam_id": 1,
        "primary_tree_index": 0,
        "source_chunk_index": 0,
        "tree": 0,
        "idm": 0,
        "itime": 0,
        "argmax_token": 0,
    }
    rows = [
        dict(common),
        dict(common),  # exact duplicate: original input order is final tie-breaker
        {**common, "argmax_token": np.uint32(1)},
        {**common, "itime": 1},
        {**common, "idm": 1},
        {**common, "tree": 1},
        {**common, "source_chunk_index": 2**53 + 7},
        {**common, "primary_tree_index": 1},
        {**common, "beam_id": 2},
        {**common, "snr": 21.0, "beam_id": 99},
        {**common, "snr": 19.0, "beam_id": 0},
    ]
    columns = _host_columns(rows)
    expected = [9, 0, 1, 2, 3, 4, 5, 6, 7, 8, 10]
    assert host_processing_order(columns).tolist() == expected
    assert host_processing_order(columns).tolist() == expected

    wrong_dtype = dict(columns)
    wrong_dtype["argmax_token"] = wrong_dtype["argmax_token"].astype(np.int64)
    with pytest.raises(TypeError, match="argmax_token"):
        host_processing_order(wrong_dtype)


def test_transfer_formulas_and_diagnostic_stage_sum_are_exact():
    columns = _host_columns([
        {"tree": 0, "snr": 30.0},
        {"tree": 1, "snr": 25.0},
        {"tree": 0, "beam_id": 8, "snr": 20.0},
        {"tree": 1, "beam_id": 8, "snr": 15.0},
    ])
    decisions = group_host_columns(columns, _host_geometry(), _host_config())
    n = len(columns["snr"])
    nevents = len(decisions.representatives)
    assert tuple(columns) == HOST_COLUMN_NAMES
    assert sum(array.nbytes for array in columns.values()) == 72 * n
    return_bytes = sum((
        decisions.candidate_event_id.nbytes,
        decisions.representatives.nbytes,
        decisions.member_count.nbytes,
        decisions.member_order.nbytes,
    ))
    assert return_bytes == 16 * n + 12 * nevents

    diagnostics = CpuGroupingDiagnostics(
        candidate_normalization_ms=0.1,
        gpu_to_cpu_transfer_ms=0.2,
        cpu_grouping_ms=0.3,
        cpu_to_gpu_result_ms=0.4,
        gpu_to_cpu_bytes=72 * n,
        cpu_to_gpu_bytes=return_bytes,
        partition_count=decisions.partition_count,
        largest_partition=decisions.largest_partition,
    )
    assert diagnostics.stage_sum_ms == pytest.approx(1.0)


def _benchmark_args(*, trials=2, percentages=(0.01,)):
    argv = [
        "--trials", str(trials),
        "--corruption-percentages", *(str(value) for value in percentages),
    ]
    return benchmark.validate_arguments(
        benchmark._make_arg_parser().parse_args(argv)
    )


def _completed_trial_row(args, *, campaign_id="cpu-test", trial=0,
                         cpu_wall_ms=1.25):
    n = 4
    nevents = 2
    decoder_ms = 0.25
    stage_values = (0.1, 0.2, 0.3, 0.4)
    stage_sum = sum(stage_values)
    combined = decoder_ms + cpu_wall_ms
    percentage = float(args.percentages[0])
    target_per_beam = gaussian.target_count_from_percentage(percentage)
    target_total = (
        target_per_beam * args.total_beams
    )
    calibration_json = json.dumps([
        {
            "beam_id": beam_id,
            "target_count": target_per_beam,
            "achieved_count": target_per_beam,
            "pixel_count_error": 0,
            "scale": 1.0,
            "calibration_evaluations": 1,
        }
        for beam_id in range(args.total_beams)
    ], separators=(",", ":"))
    row = {field: "" for field in benchmark.TRIAL_FIELDS}
    row.update({
        "schema_version": benchmark.SCHEMA_VERSION,
        "campaign_id": campaign_id,
        "trial": trial,
        "seed": gaussian.derive_seed(args.base_seed, trial, 0, "trial"),
        "total_beams": args.total_beams,
        "beam_batch_size": args.beam_batch_size,
        "threshold": args.threshold,
        "method": args.method,
        "dm_reach": args.dm_reach,
        "waist_bins": args.waist_bins,
        "target_corruption_percent": percentage,
        "achieved_corruption_percent": (
            100.0 * target_total
            / (args.total_beams * gaussian.PIXELS_PER_BEAM)
        ),
        "target_pixels_above_threshold": target_total,
        "achieved_pixels_above_threshold": target_total,
        "pixel_count_error": 0,
        "max_abs_beam_pixel_count_error": 0,
        "per_beam_calibration_json": calibration_json,
        "total_gaussians": 60,
        "gaussians_per_beam_min": 1,
        "gaussians_per_beam_median": 1.0,
        "gaussians_per_beam_max": 1,
        "gaussian_dm_width_bins_min": 2.0,
        "gaussian_dm_width_bins_median": 3.0,
        "gaussian_dm_width_bins_max": 4.0,
        "gaussian_time_width_bins_min": 1.0,
        "gaussian_time_width_bins_median": 2.0,
        "gaussian_time_width_bins_max": 3.0,
        "gaussian_correlation_min": -0.25,
        "gaussian_correlation_median": 0.0,
        "gaussian_correlation_max": 0.25,
        "gaussian_relative_amplitude_min": 0.5,
        "gaussian_relative_amplitude_median": 1.0,
        "gaussian_relative_amplitude_max": 2.0,
        "template_attempt_per_beam_min": 0,
        "template_attempt_per_beam_median": 0.0,
        "template_attempt_per_beam_max": 0,
        "total_pixels_above_threshold": target_total,
        "pixels_above_threshold_per_beam_min": target_total // args.total_beams,
        "pixels_above_threshold_per_beam_median": (
            target_total / args.total_beams
        ),
        "pixels_above_threshold_per_beam_max": target_total // args.total_beams,
        "total_peakfinder_candidates": n,
        "peakfinder_candidates_per_beam_min": 0,
        "peakfinder_candidates_per_beam_median": 0.0,
        "peakfinder_candidates_per_beam_max": n,
        "peakfinder_survival_fraction": n / target_total,
        "decoder_wall_ms": decoder_ms,
        "decoder_candidates_per_second": 1000.0 * n / decoder_ms,
        "total_decoded_candidates": n,
        "candidate_normalization_ms": stage_values[0],
        "gpu_to_cpu_transfer_ms": stage_values[1],
        "cpu_grouping_ms": stage_values[2],
        "cpu_to_gpu_result_ms": stage_values[3],
        "cpu_grouping_diagnostic_stage_sum_ms": stage_sum,
        "gpu_to_cpu_bytes": 72 * n,
        "cpu_to_gpu_bytes": 16 * n + 12 * nevents,
        "cpu_grouper_wall_ms": cpu_wall_ms,
        "cpu_grouper_candidates_per_second": 1000.0 * n / cpu_wall_ms,
        "cpu_partition_count": 2,
        "cpu_largest_partition": 2,
        "total_grouped_events": nevents,
        "grouped_events_per_beam_min": 0,
        "grouped_events_per_beam_median": 0.0,
        "grouped_events_per_beam_max": nevents,
        "decoder_plus_cpu_grouper_wall_ms": combined,
        "post_peakfinder_cpu_load_fraction": (
            combined / TEST_CHUNK_DURATION_MS
        ),
        "production_reference_verified": 0,
        "status": "completed",
        "failure_reason": "",
    })
    assert tuple(row) == benchmark.TRIAL_FIELDS
    return row


def test_cpu_csv_summary_resume_and_overwrite_contract(tmp_path):
    args = _benchmark_args()
    rows = [
        _completed_trial_row(args, trial=0, cpu_wall_ms=1.25),
        _completed_trial_row(args, trial=1, cpu_wall_ms=2.25),
    ]
    benchmark.validate_resumed_trial_rows(
        rows, args, "cpu-test", TEST_CHUNK_DURATION_MS
    )

    summary = benchmark.summarize_trials(
        rows, args.percentages, "cpu-test"
    )[0]
    assert tuple(summary) == benchmark.SUMMARY_FIELDS
    assert summary["recorded_configurations"] == 2
    assert summary["completed_trials"] == 2
    assert summary["failed_trials"] == 0
    assert summary["cpu_grouper_wall_ms_median"] == pytest.approx(1.75)
    assert summary["cpu_grouper_wall_ms_minimum"] == pytest.approx(1.25)
    assert summary["cpu_grouper_wall_ms_maximum"] == pytest.approx(2.25)
    assert summary["cpu_grouper_wall_ms_q25"] == pytest.approx(1.5)
    assert summary["cpu_grouper_wall_ms_q75"] == pytest.approx(2.0)
    assert summary["cpu_grouper_wall_ms_iqr"] == pytest.approx(0.5)

    payload = {"trials": 2, "percentages": list(args.percentages)}
    signature = gaussian.campaign_signature(payload)
    metadata = {
        "schema_name": benchmark.SCHEMA_NAME,
        "schema_version": benchmark.SCHEMA_VERSION,
        "campaign_id": "cpu-test",
        "campaign_signature": signature,
        "campaign_signature_payload": payload,
        "created_utc": "2026-08-26T00:00:00+00:00",
    }
    paths = benchmark.result_paths(tmp_path / "cpu-results")
    benchmark.write_checkpoint(
        paths, rows, args.percentages, "cpu-test", metadata
    )
    with paths["trials"].open(newline="", encoding="utf-8") as stream:
        reader = csv.DictReader(stream)
        assert tuple(reader.fieldnames or ()) == benchmark.TRIAL_FIELDS
        assert len(list(reader)) == 2
    with paths["summary"].open(newline="", encoding="utf-8") as stream:
        reader = csv.DictReader(stream)
        assert tuple(reader.fieldnames or ()) == benchmark.SUMMARY_FIELDS
        assert len(list(reader)) == 1
    with paths["metadata"].open(encoding="utf-8") as stream:
        assert yaml.safe_load(stream) == metadata

    with pytest.raises(FileExistsError, match="--resume or --overwrite"):
        benchmark.prepare_result_state(
            paths, resume=False, overwrite=False, signature=signature,
            campaign_id="cpu-test", trials=2,
            percentages=args.percentages,
        )
    resumed, created = benchmark.prepare_result_state(
        paths, resume=True, overwrite=False, signature=signature,
        campaign_id="cpu-test", trials=2, percentages=args.percentages,
    )
    assert len(resumed) == 2
    assert created == metadata["created_utc"]
    benchmark.validate_resumed_trial_rows(
        resumed, args, "cpu-test", TEST_CHUNK_DURATION_MS
    )

    fresh, fresh_created = benchmark.prepare_result_state(
        paths, resume=False, overwrite=True, signature=signature,
        campaign_id="cpu-test", trials=2, percentages=args.percentages,
    )
    assert fresh == []
    assert isinstance(fresh_created, str) and fresh_created
    with pytest.raises(ValueError, match="incompatible"):
        benchmark.prepare_result_state(
            paths, resume=True, overwrite=False, signature="0" * 64,
            campaign_id="cpu-test", trials=2,
            percentages=args.percentages,
        )

    invalid = dict(rows[0])
    invalid["gpu_to_cpu_bytes"] += 1
    with pytest.raises(ValueError, match="transfer byte count"):
        benchmark.validate_resumed_trial_rows(
            [invalid], args, "cpu-test", TEST_CHUNK_DURATION_MS
        )


def _metadata_specs():
    primary = (0, 1, 1, 2, 2, 2, 3, 3, 3, 3)
    early = (0, 1, 0, 2, 1, 0, 3, 2, 1, 0)
    result = []
    for tree_index, (shape, family, level) in enumerate(zip(
            gaussian.EXPECTED_SHAPES, primary, early)):
        ndm, ntime = shape
        result.append(SimpleNamespace(
            tree_index=tree_index,
            primary_tree_index=family,
            early_trigger_level=level,
            ndm=ndm,
            ntime=ntime,
            shape=shape,
            pixels_per_beam=ndm * ntime,
            multiplets=1,
            profiles=1,
            token_dout=1,
        ))
    return tuple(result)


def test_cpu_metadata_records_transfer_timing_and_gpu_handoff_contracts():
    args = _benchmark_args()
    specs = _metadata_specs()
    bundle = SimpleNamespace(
        config_document={"unit_test": True},
        producer_plan_yaml="unit-test-plan",
        specs=specs,
        chunk_duration_ms=TEST_CHUNK_DURATION_MS,
    )
    gaussian_config = gaussian.GaussianConfig()
    token_policy = tuple({
        "tree_index": spec.tree_index,
        "multiplet": 0,
        "profile": 0,
        "fine_time": 0,
        "token_uint32": 0,
    } for spec in specs)
    token_decodes = tuple((0, 0, 0, 0, 0) for _ in specs)
    geometries = tuple(
        SimpleNamespace(time_radius=1, ntime=spec.ntime) for spec in specs
    )
    payload = {"unit_test": "cpu-grouper-metadata"}
    signature = gaussian.campaign_signature(payload)
    metadata = benchmark.build_metadata(
        args=args,
        bundle=bundle,
        gaussian_config=gaussian_config,
        token_policy=token_policy,
        token_decodes=token_decodes,
        geometries=geometries,
        gpu={"model": "test GPU"},
        software={"python": "test", "numpy": "test", "cupy": "test"},
        git={"commit": "deadbeef", "dirty": True},
        signature=signature,
        campaign_id="cpu-test",
        created_utc="2026-08-26T00:00:00+00:00",
        rows=(),
        signature_payload_=payload,
    )
    assert metadata["schema_name"] == benchmark.SCHEMA_NAME
    assert metadata["schema_version"] == benchmark.SCHEMA_VERSION
    cpu = metadata["cpu_grouper"]
    assert cpu["implementation_scope"].startswith("benchmark-only")
    assert cpu["partition_key"] == ["beam_id", "primary_tree_index"]
    assert cpu["host_columns"] == list(HOST_COLUMN_NAMES)
    assert cpu["host_bytes_per_candidate"] == 72
    assert cpu["return_bytes_formula"] == "16*Ncandidate + 12*Nevent"
    assert "stay in VRAM" in cpu["classification_handoff"]
    timing = metadata["cpu_grouper_timing"]
    assert timing["authoritative_metric"] == "cpu_grouper_wall_ms"
    assert "GPU-to-CPU" in timing["boundary"]
    assert "rebuild the GPU-resident" in timing["boundary"]
    assert "separately synchronized" in timing["combined_metric"]
    assert metadata["csv_schemas"]["trials.csv"] == list(
        benchmark.TRIAL_FIELDS
    )
    assert metadata["csv_schemas"]["summary.csv"] == list(
        benchmark.SUMMARY_FIELDS
    )
    assert "cpu" in metadata["environment"]
    assert metadata["campaign_completeness"]["complete"] is False


def test_analysis_notebook_is_cleared_compilable_and_has_exact_four_plots():
    path = Path(__file__).with_name(
        "analyze_gaussian_corruption_cpu_grouper_timing.ipynb"
    )
    notebook = json.loads(path.read_text(encoding="utf-8"))
    assert notebook["nbformat"] == 4
    code_cells = [
        cell for cell in notebook["cells"] if cell["cell_type"] == "code"
    ]
    assert code_cells

    sources = {}
    for index, cell in enumerate(code_cells):
        assert cell.get("execution_count") is None
        assert cell.get("outputs") == []
        cell_id = cell.get("id")
        assert cell_id and cell_id not in sources
        source = "".join(cell.get("source", ()))
        compile(source, f"{path.name}:cell-{index}:{cell_id}", "exec")
        sources[cell_id] = source

    validation = sources["load-and-validate"] + sources["validate-cpu-results"]
    assert "PIRATE_REPOSITORY_ROOT" in validation
    assert "sys.path.insert" in validation
    assert validation.index("sys.path.insert") < validation.index(
        "from peakfinder_tests"
    )
    for filename in ("trials.csv", "summary.csv", "metadata.yaml"):
        assert filename in validation
    for concept in (
        "EXPECTED_SHAPES", "983_040", "gaussian_mixture",
        "campaign signature does not match its payload",
        "validate_resumed_trial_rows", "campaign is incomplete",
        "cpu_grouper_wall_ms", "host_bytes_per_candidate",
        "total_peakfinder_candidates", "total_decoded_candidates",
    ):
        assert concept in validation

    plot_ids = {
        "cpu-wall-vs-candidates",
        "cpu-breakdown-vs-candidates",
        "cpu-wall-vs-corruption",
        "post-peakfinder-cpu-vs-corruption",
    }
    show_ids = {
        cell_id for cell_id, source in sources.items() if "plt.show()" in source
    }
    assert show_ids == plot_ids
    for cell_id in plot_ids:
        source = sources[cell_id]
        assert source.count("plt.show()") == 1
        assert ".scatter(" in source
    helpers = sources["plot-helpers"]
    for concept in (
        "np.min(values)", "np.quantile(values, 0.25)",
        "np.median(values)", "np.quantile(values, 0.75)",
        "np.max(values)", "fill_between", "errorbar",
    ):
        assert concept in helpers

    overlay = sources["optional-gpu-overlay"]
    for safeguard in (
        "percentage grid", "base seed", "plan shapes", "config hash",
        "producer plan hash", "Peakfinders", "GpuArgmaxDecoder",
        "OfflineCandidateGrouper", "total_peakfinder_candidates",
        "per_beam_calibration_json",
    ):
        assert safeguard in overlay
    markdown = "".join(
        "".join(cell.get("source", ())) for cell in notebook["cells"]
        if cell["cell_type"] == "markdown"
    )
    assert "not the number of peaks" in markdown
    assert "downward timing point" in markdown


def _cuda_or_skip():
    cp = pytest.importorskip("cupy")
    try:
        if cp.cuda.runtime.getDeviceCount() < 1:
            pytest.skip("no CUDA device is available")
        cp.cuda.Device(0).use()
        cp.cuda.get_current_stream().synchronize()
        from pirate_frb.OfflineCandidateGrouper import (
            GpuDecodedCandidates,
            GroupingConfig,
            GroupingGeometry,
            group_candidates,
        )
    except (ImportError, OSError) as exc:  # pragma: no cover - environment dependent
        pytest.skip(f"production PIRATE GPU modules are unavailable: {exc}")
    except Exception as exc:  # pragma: no cover - environment dependent
        pytest.skip(f"CUDA is unavailable: {exc}")
    return (
        cp, GpuDecodedCandidates, GroupingConfig, GroupingGeometry,
        group_candidates,
    )


def _gpu_geometry(cp, GroupingGeometry):
    return GroupingGeometry(
        ntrees=4,
        primary_tree_index_by_tree=cp.asarray([0, 0, 1, 0], cp.int32),
        dm_step_by_tree=cp.asarray([1.0, 2.0, 1.0, 4.0], cp.float64),
        time_step_samples_by_tree=cp.asarray(
            [1.0, 2.0, 1.0, 4.0], cp.float64
        ),
        reference_freq_MHz=300.0,
        full_band_freq_lo_MHz=300.0,
        full_band_freq_hi_MHz=1500.0,
        time_sample_ms=1.0,
        residual_slope_lo_samples_per_dm=0.0,
        residual_slope_hi_samples_per_dm=2.0,
    )


def _gpu_candidates(cp, GpuDecodedCandidates, rows):
    host = _host_columns(rows)
    tree = host["tree"]
    return GpuDecodedCandidates(
        beam_id=cp.asarray(host["beam_id"]),
        source_chunk_index=cp.asarray(host["source_chunk_index"]),
        tree=cp.asarray(tree),
        idm=cp.asarray(host["idm"]),
        itime=cp.asarray(host["itime"]),
        snr=cp.asarray(host["snr"]),
        argmax_token=cp.asarray(host["argmax_token"]),
        edge_flags=cp.asarray(
            [row.get("edge_flags", 0) for row in rows], dtype=cp.uint8
        ),
        dm=cp.asarray(host["dm"]),
        toa_sample_abs=cp.asarray(host["toa_sample_abs"]),
        width_samp=cp.asarray(
            [row.get("width_samp", 2.0) for row in rows], dtype=cp.float64
        ),
        width_ms=cp.asarray(
            [row.get("width_ms", 2.0) for row in rows], dtype=cp.float64
        ),
        freq_lo_MHz=cp.asarray(
            [row.get("freq_lo_MHz", 300.0) for row in rows], dtype=cp.float64
        ),
        freq_hi_MHz=cp.asarray(
            [row.get("freq_hi_MHz", 1500.0) for row in rows], dtype=cp.float64
        ),
        primary_tree_index=cp.asarray(host["primary_tree_index"]),
        dm_step=cp.asarray(host["dm_step"]),
        time_step_samples=cp.asarray(host["time_step_samples"]),
    )


def test_cuda_cpu_grouping_is_exact_and_reconstructs_gpu_resident_result():
    (
        cp, GpuDecodedCandidates, GroupingConfig, GroupingGeometry,
        production_group_candidates,
    ) = _cuda_or_skip()
    geometry = _gpu_geometry(cp, GroupingGeometry)
    config = GroupingConfig(dm_tolerance_bins=1.5, time_padding_bins=1.0)
    rows = [
        {
            "beam_id": 1, "primary_tree_index": 0, "tree": 0,
            "snr": 30.0, "dm": 100.0, "toa_sample_abs": 1000.0,
            "dm_step": 1.0, "time_step_samples": 1.0,
        },
        {
            "beam_id": 1, "primary_tree_index": 0, "tree": 1,
            "snr": 25.0, "dm": 100.5, "toa_sample_abs": 1001.0,
            "dm_step": 2.0, "time_step_samples": 2.0,
            "edge_flags": 1,
        },
        {
            "beam_id": 1, "primary_tree_index": 0, "tree": 1,
            "snr": 24.0, "dm": 100.4, "toa_sample_abs": 1000.5,
            "dm_step": 2.0, "time_step_samples": 2.0,
        },
        {
            "beam_id": 1, "primary_tree_index": 0, "tree": 3,
            "snr": 23.0, "dm": 100.2, "toa_sample_abs": 999.5,
            "dm_step": 4.0, "time_step_samples": 4.0,
        },
        {
            "beam_id": 2, "primary_tree_index": 0, "tree": 1,
            "snr": 22.0, "dm": 100.0, "toa_sample_abs": 1000.0,
            "dm_step": 2.0, "time_step_samples": 2.0,
        },
        {
            "beam_id": 1, "primary_tree_index": 1, "tree": 2,
            "snr": 21.0, "dm": 100.0, "toa_sample_abs": 1000.0,
            "dm_step": 1.0, "time_step_samples": 1.0,
        },
        {
            "beam_id": 1, "primary_tree_index": 0, "tree": 1,
            "source_chunk_index": 1, "snr": 20.0,
            "dm": 42.0, "toa_sample_abs": 1024.0,
            "dm_step": 2.0, "time_step_samples": 2.0,
        },
        {
            "beam_id": 1, "primary_tree_index": 0, "tree": 0,
            "source_chunk_index": 0, "snr": 19.0,
            "dm": 42.1, "toa_sample_abs": 1023.75,
            "dm_step": 1.0, "time_step_samples": 1.0,
        },
    ]
    candidates = _gpu_candidates(cp, GpuDecodedCandidates, rows)
    production = production_group_candidates(
        candidates, geometry, config=config
    )
    cpu_result, diagnostics = group_candidates_on_cpu(
        candidates, geometry, config=config, cp_module=cp
    )
    cp.cuda.get_current_stream().synchronize()
    assert_gpu_grouping_results_equal(cp, production, cpu_result)
    assert diagnostics.gpu_to_cpu_bytes == 72 * len(rows)
    assert diagnostics.cpu_to_gpu_bytes == (
        16 * len(rows) + 12 * len(cpu_result.events)
    )
    assert diagnostics.stage_sum_ms >= 0.0

    device = int(candidates.beam_id.device.id)
    assert int(cpu_result.candidate_event_id.device.id) == device
    for table_name in ("events", "members"):
        table = getattr(cpu_result, table_name)
        for field in fields(table):
            value = getattr(table, field.name)
            assert isinstance(value, cp.ndarray)
            assert int(value.device.id) == device

    empty = _gpu_candidates(cp, GpuDecodedCandidates, [])
    empty_production = production_group_candidates(
        empty, geometry, config=config
    )
    empty_cpu, empty_diagnostics = group_candidates_on_cpu(
        empty, geometry, config=config, cp_module=cp
    )
    assert_gpu_grouping_results_equal(cp, empty_production, empty_cpu)
    assert empty_diagnostics.gpu_to_cpu_bytes == 0
    assert empty_diagnostics.cpu_to_gpu_bytes == 0


def test_cuda_randomized_cpu_grouping_matches_production_exactly():
    (
        cp, GpuDecodedCandidates, GroupingConfig, GroupingGeometry,
        production_group_candidates,
    ) = _cuda_or_skip()
    geometry = _gpu_geometry(cp, GroupingGeometry)
    config = GroupingConfig(dm_tolerance_bins=1.5, time_padding_bins=1.0)
    rng = np.random.default_rng(20260826)
    family_by_tree = (0, 0, 1, 0)
    dm_step_by_tree = (1.0, 2.0, 1.0, 4.0)
    time_step_by_tree = (1.0, 2.0, 1.0, 4.0)

    for case in range(16):
        n = int(rng.integers(1, 33))
        rows = []
        for _ in range(n):
            tree = int(rng.integers(0, 4))
            cluster = int(rng.integers(0, 5))
            dm_centre = 50.0 + 20.0 * cluster
            toa_centre = 1_000.0 + 50.0 * cluster
            rows.append({
                "beam_id": int(rng.integers(0, 4)),
                "primary_tree_index": family_by_tree[tree],
                "tree": tree,
                "source_chunk_index": int(rng.integers(0, 3)),
                "idm": int(rng.integers(0, 64)),
                "itime": int(rng.integers(0, 64)),
                "argmax_token": int(rng.integers(
                    0, 2**32, dtype=np.uint64
                )),
                # A discrete S/N grid deliberately creates deterministic ties.
                "snr": float(rng.choice((15.0, 20.0, 25.0, 30.0))),
                "dm": dm_centre + float(rng.normal(0.0, 0.35)),
                "toa_sample_abs": (
                    toa_centre + float(rng.normal(0.0, 0.75))
                ),
                "dm_step": dm_step_by_tree[tree],
                "time_step_samples": time_step_by_tree[tree],
                "edge_flags": int(rng.integers(0, 4)),
            })
        candidates = _gpu_candidates(cp, GpuDecodedCandidates, rows)
        production = production_group_candidates(
            candidates, geometry, config=config
        )
        cpu_result, diagnostics = group_candidates_on_cpu(
            candidates, geometry, config=config, cp_module=cp
        )
        cp.cuda.get_current_stream().synchronize()
        assert_gpu_grouping_results_equal(cp, production, cpu_result)
        assert diagnostics.gpu_to_cpu_bytes == 72 * n
        assert diagnostics.cpu_to_gpu_bytes == (
            16 * n + 12 * len(cpu_result.events)
        )
