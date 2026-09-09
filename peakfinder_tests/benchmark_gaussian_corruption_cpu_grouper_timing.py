#!/usr/bin/env python3
"""Time an exact CPU grouper between PIRATE's GPU decoder and GPU results.

Synthetic-map generation, calibration, peakfinding, and decoding reuse the
existing Gaussian-corruption benchmark and active production APIs.  Only the
grouping implementation is replaced by the benchmark-side host-partitioned
alternative in :mod:`peakfinder_tests.cpu_candidate_grouper`.
"""

from __future__ import annotations

import argparse
import math
import os
from pathlib import Path
import platform
from typing import Any, Iterable, Mapping, Sequence

import numpy as np
import yaml

from peakfinder_tests import benchmark_gaussian_corruption_timing as gaussian
from peakfinder_tests import benchmark_peakfinder_batch_timing as batch_benchmark
from peakfinder_tests.cpu_candidate_grouper import (
    HOST_COLUMN_NAMES,
    CpuGroupingDiagnostics,
    assert_gpu_grouping_results_equal,
    group_candidates_on_cpu,
    synchronized_cpu_grouping_wall_time,
)


SCHEMA_NAME = "pirate-gaussian-corruption-cpu-grouper-timing"
SCHEMA_VERSION = 2
REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_RESULTS_DIR = (
    Path(__file__).resolve().parent
    / "results_gaussian_corruption_cpu_grouper_timing_pirate15"
)
NOTEBOOK_PATH = (
    Path(__file__).resolve().parent
    / "analyze_gaussian_corruption_cpu_grouper_timing.ipynb"
)

_WORKLOAD_FIELDS = gaussian.TRIAL_FIELDS[
    :gaussian.TRIAL_FIELDS.index("grouper_wall_ms")
]
TRIAL_FIELDS = (
    *_WORKLOAD_FIELDS,
    "candidate_normalization_ms",
    "gpu_to_cpu_transfer_ms",
    "cpu_grouping_ms",
    "cpu_to_gpu_result_ms",
    "cpu_grouping_diagnostic_stage_sum_ms",
    "gpu_to_cpu_bytes",
    "cpu_to_gpu_bytes",
    "cpu_grouper_wall_ms",
    "cpu_grouper_candidates_per_second",
    "cpu_partition_count",
    "cpu_largest_partition",
    "total_grouped_events",
    "grouped_events_per_beam_min",
    "grouped_events_per_beam_median",
    "grouped_events_per_beam_max",
    "decoder_plus_cpu_grouper_wall_ms",
    "post_peakfinder_cpu_load_fraction",
    "production_reference_verified",
    "status",
    "failure_reason",
)

SUMMARY_METRICS = (
    "total_pixels_above_threshold",
    "total_peakfinder_candidates",
    "peakfinder_survival_fraction",
    "decoder_wall_ms",
    "total_decoded_candidates",
    "candidate_normalization_ms",
    "gpu_to_cpu_transfer_ms",
    "cpu_grouping_ms",
    "cpu_to_gpu_result_ms",
    "cpu_grouping_diagnostic_stage_sum_ms",
    "gpu_to_cpu_bytes",
    "cpu_to_gpu_bytes",
    "cpu_grouper_wall_ms",
    "total_grouped_events",
    "decoder_plus_cpu_grouper_wall_ms",
    "post_peakfinder_cpu_load_fraction",
)
SUMMARY_STATISTICS = gaussian.SUMMARY_STATISTICS
SUMMARY_FIELDS = (
    "schema_version",
    "campaign_id",
    "target_corruption_percent",
    "recorded_configurations",
    "completed_trials",
    "failed_trials",
    *tuple(
        f"{metric}_{statistic}"
        for metric in SUMMARY_METRICS
        for statistic in SUMMARY_STATISTICS
    ),
)


def _validate_rows(
        rows: Iterable[Mapping[str, Any]], fields_: Sequence[str],
        ) -> list[dict[str, Any]]:
    return gaussian._validate_rows(rows, fields_)


def result_paths(results_dir: Path | str) -> dict[str, Path]:
    directory = Path(results_dir).resolve()
    return {
        "directory": directory,
        "trials": directory / "trials.csv",
        "summary": directory / "summary.csv",
        "metadata": directory / "metadata.yaml",
    }


def summarize_trials(
        rows: Sequence[Mapping[str, Any]], percentages: Sequence[float],
        campaign_id: str,
        ) -> list[dict[str, Any]]:
    result = []
    for percentage in percentages:
        matching = [
            row for row in rows
            if gaussian.canonical_percentage(row["target_corruption_percent"])
            == gaussian.canonical_percentage(percentage)
        ]
        completed = [row for row in matching if row["status"] == "completed"]
        summary: dict[str, Any] = {
            "schema_version": SCHEMA_VERSION,
            "campaign_id": campaign_id,
            "target_corruption_percent": percentage,
            "recorded_configurations": len(matching),
            "completed_trials": len(completed),
            "failed_trials": sum(
                str(row["status"]).startswith("failed_") for row in matching
            ),
        }
        for metric in SUMMARY_METRICS:
            values = np.asarray([
                float(row[metric]) for row in completed if row[metric] != ""
            ], dtype=np.float64)
            if values.size:
                q25 = gaussian._quantile(values, 0.25)
                q75 = gaussian._quantile(values, 0.75)
                statistics = {
                    "median": float(np.median(values)),
                    "minimum": float(np.min(values)),
                    "maximum": float(np.max(values)),
                    "q25": q25,
                    "q75": q75,
                    "iqr": q75 - q25,
                }
            else:
                statistics = {name: "" for name in SUMMARY_STATISTICS}
            for statistic, value in statistics.items():
                summary[f"{metric}_{statistic}"] = value
        result.append(summary)
    return _validate_rows(result, SUMMARY_FIELDS)


def build_trial_row(
        *, campaign_id: str, args: argparse.Namespace,
        morphology: gaussian.TrialMorphology, percentage: float,
        calibrations: Sequence[gaussian.BeamCalibration],
        pixel_counts: np.ndarray, raw_counts: np.ndarray,
        decoder_wall_ms: float, decoded_count: int,
        cpu_grouper_wall_ms: float, diagnostics: CpuGroupingDiagnostics,
        event_counts: np.ndarray, production_reference_verified: bool,
        chunk_duration_ms: float,
        ) -> dict[str, Any]:
    # Reuse the reviewed workload/count construction.  Its generic grouper
    # slots are used only as an intermediate and renamed below.
    base_row = gaussian.build_trial_row(
        campaign_id=campaign_id,
        args=args,
        morphology=morphology,
        percentage=percentage,
        calibrations=calibrations,
        pixel_counts=pixel_counts,
        raw_counts=raw_counts,
        decoder_wall_ms=decoder_wall_ms,
        decoded_count=decoded_count,
        grouper_wall_ms=cpu_grouper_wall_ms,
        event_counts=event_counts,
        status="completed",
        chunk_duration_ms=chunk_duration_ms,
    )
    row = {name: base_row[name] for name in _WORKLOAD_FIELDS}
    row.update({
        "candidate_normalization_ms": diagnostics.candidate_normalization_ms,
        "gpu_to_cpu_transfer_ms": diagnostics.gpu_to_cpu_transfer_ms,
        "cpu_grouping_ms": diagnostics.cpu_grouping_ms,
        "cpu_to_gpu_result_ms": diagnostics.cpu_to_gpu_result_ms,
        "cpu_grouping_diagnostic_stage_sum_ms": diagnostics.stage_sum_ms,
        "gpu_to_cpu_bytes": diagnostics.gpu_to_cpu_bytes,
        "cpu_to_gpu_bytes": diagnostics.cpu_to_gpu_bytes,
        "cpu_grouper_wall_ms": cpu_grouper_wall_ms,
        "cpu_grouper_candidates_per_second": (
            base_row["grouper_candidates_per_second"]
        ),
        "cpu_partition_count": diagnostics.partition_count,
        "cpu_largest_partition": diagnostics.largest_partition,
        "total_grouped_events": base_row["total_grouped_events"],
        "grouped_events_per_beam_min": (
            base_row["grouped_events_per_beam_min"]
        ),
        "grouped_events_per_beam_median": (
            base_row["grouped_events_per_beam_median"]
        ),
        "grouped_events_per_beam_max": (
            base_row["grouped_events_per_beam_max"]
        ),
        "decoder_plus_cpu_grouper_wall_ms": (
            base_row["decoder_plus_grouper_wall_ms"]
        ),
        "post_peakfinder_cpu_load_fraction": (
            base_row["post_peakfinder_load_fraction"]
        ),
        "production_reference_verified": int(production_reference_verified),
        "status": "completed",
        "failure_reason": "",
    })
    return _validate_rows([row], TRIAL_FIELDS)[0]


def blank_failed_row(
        *, campaign_id: str, args: argparse.Namespace, trial: int,
        percentage: float, status: str, reason: str,
        ) -> dict[str, Any]:
    base_row = gaussian._blank_failed_row(
        campaign_id=campaign_id,
        args=args,
        trial=trial,
        percentage=percentage,
        status=status,
        reason=reason,
    )
    row = {field: "" for field in TRIAL_FIELDS}
    row.update({name: base_row[name] for name in _WORKLOAD_FIELDS})
    row.update({
        "production_reference_verified": 0,
        "status": status,
        "failure_reason": reason,
    })
    return _validate_rows([row], TRIAL_FIELDS)[0]


def validate_resumed_trial_rows(
        rows: Sequence[Mapping[str, Any]], args: argparse.Namespace,
        campaign_id: str, chunk_duration_ms: float,
        ) -> None:
    # Reconstruct the parent benchmark's schema so its reviewed scientific,
    # calibration, count-conservation, and shared timing checks remain the
    # single source of truth for the workload portion of every resumed row.
    gaussian_rows = []
    for row in rows:
        shared = {field: "" for field in gaussian.TRIAL_FIELDS}
        shared.update({field: row[field] for field in _WORKLOAD_FIELDS})
        shared.update({
            "grouper_wall_ms": row["cpu_grouper_wall_ms"],
            "grouper_candidates_per_second": (
                row["cpu_grouper_candidates_per_second"]
            ),
            "total_grouped_events": row["total_grouped_events"],
            "grouped_events_per_beam_min": (
                row["grouped_events_per_beam_min"]
            ),
            "grouped_events_per_beam_median": (
                row["grouped_events_per_beam_median"]
            ),
            "grouped_events_per_beam_max": (
                row["grouped_events_per_beam_max"]
            ),
            "decoder_plus_grouper_wall_ms": (
                row["decoder_plus_cpu_grouper_wall_ms"]
            ),
            "post_peakfinder_load_fraction": (
                row["post_peakfinder_cpu_load_fraction"]
            ),
            "candidate_safety_limit": args.candidate_safety_limit,
            "status": row["status"],
            "failure_reason": row["failure_reason"],
        })
        gaussian_rows.append(shared)
    gaussian.validate_resumed_trial_rows(
        gaussian_rows, args, campaign_id, chunk_duration_ms
    )

    allowed = {
        "completed", "failed_template_unattainable", "failed_calibration"
    }
    for index, row in enumerate(rows):
        context = f"resumed trials.csv row {index + 2}"
        if str(row["campaign_id"]) != campaign_id:
            raise ValueError(f"{context}: incompatible campaign_id")
        for name, expected in (
                ("schema_version", SCHEMA_VERSION),
                ("total_beams", args.total_beams),
                ("beam_batch_size", args.beam_batch_size),
                ("dm_reach", args.dm_reach),
                ("waist_bins", args.waist_bins)):
            if int(row[name]) != int(expected):
                raise ValueError(f"{context}: incompatible {name}")
        if str(row["method"]) != args.method or not math.isclose(
                float(row["threshold"]), args.threshold,
                rel_tol=0.0, abs_tol=1.0e-12):
            raise ValueError(f"{context}: incompatible peakfinder configuration")
        trial = int(row["trial"])
        if int(row["seed"]) != gaussian.derive_seed(
                args.base_seed, trial, 0, "trial"):
            raise ValueError(f"{context}: incompatible seed")
        status = str(row["status"])
        if status not in allowed:
            raise ValueError(f"{context}: unsupported status {status!r}")
        if status.startswith("failed_"):
            if not str(row["failure_reason"]).strip():
                raise ValueError(f"{context}: failed row lacks a reason")
            continue

        candidates = int(row["total_peakfinder_candidates"])
        decoded = int(row["total_decoded_candidates"])
        events = int(row["total_grouped_events"])
        if candidates != decoded or not 0 <= events <= decoded:
            raise ValueError(f"{context}: candidate/event counts are inconsistent")
        timings = [
            float(row[name]) for name in (
                "decoder_wall_ms", "candidate_normalization_ms",
                "gpu_to_cpu_transfer_ms", "cpu_grouping_ms",
                "cpu_to_gpu_result_ms",
                "cpu_grouping_diagnostic_stage_sum_ms",
                "cpu_grouper_wall_ms",
                "decoder_plus_cpu_grouper_wall_ms",
                "post_peakfinder_cpu_load_fraction",
            )
        ]
        if not all(math.isfinite(value) and value >= 0.0 for value in timings):
            raise ValueError(f"{context}: timing is not finite/non-negative")
        stage_sum = sum(timings[1:5])
        if not math.isclose(
                timings[5], stage_sum, rel_tol=1.0e-12, abs_tol=1.0e-12):
            raise ValueError(f"{context}: diagnostic stage sum is inconsistent")
        if timings[5] > timings[6] + 1.0e-9:
            raise ValueError(f"{context}: diagnostic stages exceed wall time")
        if not math.isclose(
                timings[7], timings[0] + timings[6],
                rel_tol=1.0e-12, abs_tol=1.0e-12):
            raise ValueError(f"{context}: combined timing is inconsistent")
        if not math.isclose(
                timings[8], timings[7] / chunk_duration_ms,
                rel_tol=1.0e-12, abs_tol=1.0e-12):
            raise ValueError(f"{context}: real-time load is inconsistent")
        d2h = int(row["gpu_to_cpu_bytes"])
        h2d = int(row["cpu_to_gpu_bytes"])
        if d2h != 72 * decoded or h2d != 16 * decoded + 12 * events:
            raise ValueError(f"{context}: transfer byte count is inconsistent")
        verified = int(row["production_reference_verified"])
        expected_verified = int(
            args.verify_production_reference
            and (
                decoded <= args.candidate_safety_limit
                or args.allow_unsafe_grouping
            )
        )
        if verified != expected_verified:
            raise ValueError(
                f"{context}: production-reference verification is inconsistent"
            )
        partitions = int(row["cpu_partition_count"])
        largest = int(row["cpu_largest_partition"])
        if decoded == 0:
            valid_partitions = partitions == 0 and largest == 0
        else:
            valid_partitions = (
                1 <= partitions <= events
                and 1 <= largest <= decoded
            )
        if not valid_partitions:
            raise ValueError(f"{context}: partition diagnostics are inconsistent")


def write_checkpoint(
        paths: Mapping[str, Path], rows: Sequence[Mapping[str, Any]],
        percentages: Sequence[float], campaign_id: str,
        metadata: Mapping[str, Any],
        ) -> None:
    gaussian._atomic_write_csv(paths["trials"], TRIAL_FIELDS, rows)
    gaussian._atomic_write_csv(
        paths["summary"], SUMMARY_FIELDS,
        summarize_trials(rows, percentages, campaign_id),
    )
    gaussian._atomic_write_yaml(paths["metadata"], metadata)


def prepare_result_state(
        paths: Mapping[str, Path], *, resume: bool, overwrite: bool,
        signature: str, campaign_id: str, trials: int,
        percentages: Sequence[float],
        ) -> tuple[list[dict[str, Any]], str]:
    if resume and overwrite:
        raise ValueError("resume and overwrite are mutually exclusive")
    files = [paths[name] for name in ("trials", "summary", "metadata")]
    existing = [path for path in files if path.exists()]
    if overwrite:
        return [], gaussian._utc_now()
    if existing and not resume:
        raise FileExistsError(
            "result files already exist; use --resume or --overwrite: "
            + ", ".join(str(path) for path in existing)
        )
    if not existing:
        return [], gaussian._utc_now()
    if len(existing) != len(files):
        raise ValueError("resume requires all three result files")
    with paths["metadata"].open("r", encoding="utf-8") as stream:
        metadata = yaml.safe_load(stream)
    if not isinstance(metadata, dict):
        raise ValueError("metadata.yaml must contain a mapping")
    payload = metadata.get("campaign_signature_payload")
    try:
        stored_signature = (
            gaussian.campaign_signature(payload)
            if isinstance(payload, dict) else None
        )
    except (TypeError, ValueError):
        stored_signature = None
    if (metadata.get("schema_name") != SCHEMA_NAME
            or int(metadata.get("schema_version", -1)) != SCHEMA_VERSION
            or metadata.get("campaign_signature") != signature
            or stored_signature != signature
            or metadata.get("campaign_id") != campaign_id):
        raise ValueError("resume metadata is incompatible with this campaign")
    rows = [
        dict(row)
        for row in gaussian._read_csv_exact(paths["trials"], TRIAL_FIELDS)
    ]
    gaussian._read_csv_exact(paths["summary"], SUMMARY_FIELDS)
    gaussian.validate_trial_prefix(rows, trials, percentages, campaign_id)
    return rows, str(metadata.get("created_utc") or gaussian._utc_now())


def cpu_information() -> dict[str, Any]:
    model = platform.processor() or "unknown"
    cpuinfo = Path("/proc/cpuinfo")
    if cpuinfo.exists():
        for line in cpuinfo.read_text(encoding="utf-8", errors="replace").splitlines():
            if line.lower().startswith("model name") and ":" in line:
                model = line.split(":", 1)[1].strip()
                break
    return {
        "model": model,
        "logical_cpu_count": os.cpu_count(),
        "machine": platform.machine(),
    }


def benchmark_source_information() -> dict[str, Any]:
    paths = {
        "cpu_grouper_benchmark": Path(__file__).resolve(),
        "cpu_grouper_helper": (
            REPOSITORY_ROOT / "peakfinder_tests/cpu_candidate_grouper.py"
        ).resolve(),
        "gaussian_corruption_helpers": Path(gaussian.__file__).resolve(),
        "peakfinder_batch_helpers": Path(batch_benchmark.__file__).resolve(),
        "producer_metadata": REPOSITORY_ROOT / "peakfinder_tests/producer_metadata.py",
    }
    return {
        name: {"path": str(path), "sha256": gaussian._sha256_path(path)}
        for name, path in paths.items()
    }


def signature_payload(
        args: argparse.Namespace, bundle: Any,
        gaussian_config: gaussian.GaussianConfig,
        gpu: Mapping[str, Any], software: Mapping[str, Any],
        ) -> dict[str, Any]:
    payload = gaussian._signature_payload(
        args, bundle, gaussian_config, gpu, software
    )
    payload.update({
        "schema_name": SCHEMA_NAME,
        "schema_version": SCHEMA_VERSION,
        "cpu_grouping_policy": (
            "exact representative-seeded grouping partitioned by "
            "(beam_id, primary_tree_index), then globally renumbered"
        ),
        "host_columns": list(HOST_COLUMN_NAMES),
        "verify_production_reference": bool(args.verify_production_reference),
        "benchmark_sources": benchmark_source_information(),
    })
    return payload


def build_metadata(
        *, args: argparse.Namespace, bundle: Any,
        gaussian_config: gaussian.GaussianConfig,
        token_policy: Sequence[Mapping[str, int]],
        token_decodes: Sequence[Sequence[int]], geometries: Sequence[Any],
        gpu: Mapping[str, Any], software: Mapping[str, Any],
        git: Mapping[str, Any], signature: str, campaign_id: str,
        created_utc: str, rows: Sequence[Mapping[str, Any]],
        signature_payload_: Mapping[str, Any],
        ) -> dict[str, Any]:
    metadata = gaussian.build_metadata(
        args=args,
        bundle=bundle,
        gaussian_config=gaussian_config,
        token_policy=token_policy,
        token_decodes=token_decodes,
        geometries=geometries,
        gpu=gpu,
        software=software,
        git=git,
        signature=signature,
        campaign_id=campaign_id,
        created_utc=created_utc,
        rows=rows,
        signature_payload=signature_payload_,
    )
    metadata["schema_name"] = SCHEMA_NAME
    metadata["schema_version"] = SCHEMA_VERSION
    metadata.pop("grouper_timing", None)
    metadata.pop("safety_limit", None)
    metadata["cpu_grouper"] = {
        "implementation_scope": (
            "benchmark-only; production OfflineCandidateGrouper.py and "
            "FrbOfflineGrouper.py are unchanged"
        ),
        "semantics": (
            "production stable priority and representative-seeded greedy "
            "membership, exact one-row-per-tree selection, representative-derived "
            "events, and stable member ordering"
        ),
        "partition_key": ["beam_id", "primary_tree_index"],
        "partition_equivalence": (
            "production compatibility forbids cross-key membership; local events "
            "are globally renumbered by representative rank"
        ),
        "host_columns": list(HOST_COLUMN_NAMES),
        "host_bytes_per_candidate": 72,
        "return_columns": {
            "candidate_event_id": "int64[N]",
            "representative_candidate_index": "int64[E]",
            "member_count": "int32[E]",
            "member_order": "int64[N]",
        },
        "return_bytes_formula": "16*Ncandidate + 12*Nevent",
        "classification_handoff": (
            "decoded feature columns stay in VRAM; returned indices rebuild the "
            "production GPU event/member tables, so a GPU classifier can gather "
            "features without re-uploading decoded physical columns"
        ),
        "dm_tolerance_bins": args.dm_tolerance_bins,
        "time_padding_bins": args.time_padding_bins,
        "complexity": (
            "worst-case quadratic only within an independent beam/family partition"
        ),
    }
    metadata["cpu_grouper_timing"] = {
        "authoritative_metric": "cpu_grouper_wall_ms",
        "boundary": (
            "synchronize current CUDA stream; perf_counter; production normalize/"
            "validate decoded candidates; copy grouping columns GPU-to-CPU; execute "
            "partitioned CPU grouping; upload compact integer results; rebuild the "
            "GPU-resident grouping result; synchronize; elapsed wall time"
        ),
        "diagnostics": [
            "candidate_normalization_ms", "gpu_to_cpu_transfer_ms",
            "cpu_grouping_ms", "cpu_to_gpu_result_ms",
            "cpu_grouping_diagnostic_stage_sum_ms",
        ],
        "combined_metric": (
            "decoder_plus_cpu_grouper_wall_ms is the sum of two separately "
            "synchronized measurements, not a fused interval"
        ),
        "excluded": (
            "map generation, calibration, upload, peakfinding, correctness counts, "
            "optional production-reference comparison, and output writing"
        ),
    }
    metadata["production_reference_verification"] = {
        "enabled": bool(args.verify_production_reference),
        "candidate_limit": args.candidate_safety_limit,
        "unsafe_override": bool(args.allow_unsafe_grouping),
        "policy": (
            "optional and outside all authoritative timers; configurations above "
            "the limit are not reference-grouped unless --allow-unsafe-grouping"
        ),
    }
    metadata["environment"]["cpu"] = cpu_information()
    metadata["environment"]["benchmark_sources"] = (
        benchmark_source_information()
    )
    metadata["csv_schemas"] = {
        "trials.csv": list(TRIAL_FIELDS),
        "summary.csv": list(SUMMARY_FIELDS),
        "summary_metrics": list(SUMMARY_METRICS),
        "summary_statistics": list(SUMMARY_STATISTICS),
        "zero_denominator_peakfinder_survival_fraction": 0.0,
    }
    metadata["outputs"] = {
        "trials": "trials.csv",
        "summary": "summary.csv",
        "metadata": "metadata.yaml",
        "analysis_notebook": str(NOTEBOOK_PATH.resolve()),
    }
    return metadata


def _make_arg_parser() -> argparse.ArgumentParser:
    parser = gaussian._make_arg_parser()
    parser.description = (
        "Monte Carlo Gaussian-corruption timing of an exact benchmark-side CPU "
        "grouper, including its complete GPU round trip"
    )
    parser.set_defaults(
        results_dir=str(DEFAULT_RESULTS_DIR),
        candidate_safety_limit=512,
    )
    for action in parser._actions:
        if action.dest == "candidate_safety_limit":
            action.help = (
                "maximum candidates for optional slow production-reference "
                "verification"
            )
        elif action.dest == "allow_unsafe_grouping":
            action.help = (
                "allow optional production-reference verification above its "
                "candidate limit"
            )
    parser.add_argument(
        "--verify-production-reference", action="store_true",
        help=(
            "outside timers, compare CPU and production GPU grouping for batches "
            "at or below the candidate limit"
        ),
    )
    return parser


def validate_arguments(args: argparse.Namespace) -> argparse.Namespace:
    return gaussian.validate_arguments(args)


def _print_percentage_summary(
        rows: Sequence[Mapping[str, Any]], percentage: float,
        ) -> None:
    completed = [
        row for row in rows
        if row["status"] == "completed"
        and gaussian.canonical_percentage(row["target_corruption_percent"])
        == gaussian.canonical_percentage(percentage)
    ]
    if not completed:
        print(f"corruption={percentage:g}% completed=0", flush=True)
        return
    candidates = np.asarray([
        float(row["total_peakfinder_candidates"]) for row in completed
    ])
    cpu_ms = np.asarray([
        float(row["cpu_grouper_wall_ms"]) for row in completed
    ])
    load = np.asarray([
        float(row["post_peakfinder_cpu_load_fraction"]) for row in completed
    ])
    print(
        f"corruption={percentage:g}% completed={len(completed)} "
        f"candidates_median={np.median(candidates):.0f} "
        f"cpu_grouper_median={np.median(cpu_ms):.3f}ms "
        f"post_peakfinder_load_median={100.0*np.median(load):.3f}%",
        flush=True,
    )


def run_benchmark(args: argparse.Namespace) -> None:
    args = validate_arguments(args)
    gaussian_config = gaussian._make_gaussian_config(args)
    paths = result_paths(args.results_dir)

    import cupy as cp
    from pirate_frb.GpuArgmaxDecoder import GpuArgmaxDecoder
    from pirate_frb.OfflineCandidateGrouper import (
        GroupingConfig,
        GroupingGeometry,
        group_candidates,
    )
    from pirate_frb.Peakfinders import (
        GpuRawCandidates,
        concatenate_raw_candidates,
    )

    with cp.cuda.Device(args.device):
        stream = cp.cuda.get_current_stream()
        active_sources = batch_benchmark.active_repository_sources()
        bundle = batch_benchmark.load_authoritative_plan(args.config)
        if int(bundle.config.beams_per_gpu) != gaussian.DEFAULT_TOTAL_BEAMS:
            raise ValueError("authoritative config no longer declares 60 beams per GPU")
        if tuple(spec.shape for spec in bundle.specs) != gaussian.EXPECTED_SHAPES:
            raise AssertionError("authoritative plan shapes changed")

        geometries_by_reach, _ = batch_benchmark.build_geometries(
            cp, bundle.plan, bundle.specs, (args.dm_reach,), args.waist_bins
        )
        peak_geometries = geometries_by_reach[args.dm_reach]
        decoder = GpuArgmaxDecoder(bundle.plan, cuda_device_id=args.device, dcores=bundle.dcores)
        grouping_geometry = GroupingGeometry.from_plan(bundle.plan)
        grouping_config = GroupingConfig(
            dm_tolerance_bins=args.dm_tolerance_bins,
            time_padding_bins=args.time_padding_bins,
        )

        argmax_host, token_policy = gaussian.generate_valid_argmax_maps(
            bundle.specs, args.total_beams, args.base_seed
        )
        token_decodes = gaussian.validate_tokens_with_plan(
            bundle.plan, bundle.specs, token_policy
        )
        context_host = gaussian.generate_clean_context_maps(
            bundle.specs, args.total_beams, args.base_seed, args.threshold
        )
        argmax_gpu = tuple(cp.asarray(array) for array in argmax_host)
        context_gpu = tuple(cp.asarray(array) for array in context_host)
        del argmax_host, context_host
        stream.synchronize()

        gpu = batch_benchmark.gpu_information(cp, args.device)
        software = batch_benchmark.software_information(cp)
        software["active_repository_sources"] = active_sources
        software["python"] = platform.python_version()
        git = batch_benchmark.git_information()
        payload = signature_payload(
            args, bundle, gaussian_config, gpu, software
        )
        signature = gaussian.campaign_signature(payload)
        campaign_id = signature[:16]
        rows, created_utc = prepare_result_state(
            paths,
            resume=bool(args.resume),
            overwrite=bool(args.overwrite),
            signature=signature,
            campaign_id=campaign_id,
            trials=args.trials,
            percentages=args.percentages,
        )
        gaussian.validate_trial_prefix(
            rows, args.trials, args.percentages, campaign_id
        )
        validate_resumed_trial_rows(
            rows, args, campaign_id, bundle.chunk_duration_ms
        )

        def checkpoint() -> None:
            metadata = build_metadata(
                args=args,
                bundle=bundle,
                gaussian_config=gaussian_config,
                token_policy=token_policy,
                token_decodes=token_decodes,
                geometries=peak_geometries,
                gpu=gpu,
                software=software,
                git=git,
                signature=signature,
                campaign_id=campaign_id,
                created_utc=created_utc,
                rows=rows,
                signature_payload_=payload,
            )
            write_checkpoint(
                paths, rows, args.percentages, campaign_id, metadata
            )

        checkpoint()
        expected_keys = gaussian.expected_trial_keys(
            args.trials, args.percentages
        )
        if len(rows) == len(expected_keys):
            print("Campaign is already complete; nothing to resume.", flush=True)
            for percentage in args.percentages:
                _print_percentage_summary(rows, percentage)
            return

        # Exercise empty schemas and allocations outside authoritative timers.
        empty_decoded = decoder.decode(GpuRawCandidates.empty())
        group_candidates_on_cpu(
            empty_decoded, grouping_geometry, config=grouping_config,
            cp_module=cp,
        )
        stream.synchronize()
        nonempty_warmed = False
        required_targets = tuple(
            gaussian.target_count_from_percentage(percentage)
            for percentage in args.percentages
        )
        maximum_target = max(required_targets)

        for trial in range(args.trials):
            present = {
                (int(row["trial"]), gaussian.canonical_percentage(
                    row["target_corruption_percent"]
                ))
                for row in rows
            }
            pending = [
                percentage for percentage in args.percentages
                if (trial, gaussian.canonical_percentage(percentage)) not in present
            ]
            if not pending:
                continue
            try:
                morphology = gaussian.generate_trial_morphology(
                    bundle.specs,
                    base_seed=args.base_seed,
                    trial=trial,
                    total_beams=args.total_beams,
                    gaussian_config=gaussian_config,
                    threshold=args.threshold,
                    maximum_target_count=maximum_target,
                    attempt_limit=args.template_attempt_limit,
                    calibration_pixel_tolerance=(
                        args.calibration_pixel_tolerance
                    ),
                    calibration_max_iterations=(
                        args.calibration_max_iterations
                    ),
                    required_target_counts=required_targets,
                )
            except gaussian.MorphologyGenerationError as exc:
                for percentage in pending:
                    rows.append(blank_failed_row(
                        campaign_id=campaign_id,
                        args=args,
                        trial=trial,
                        percentage=percentage,
                        status="failed_template_unattainable",
                        reason=str(exc),
                    ))
                    gaussian.validate_trial_prefix(
                        rows, args.trials, args.percentages, campaign_id
                    )
                    checkpoint()
                continue

            for percentage in args.percentages:
                key = (trial, gaussian.canonical_percentage(percentage))
                if key in present:
                    continue
                target_count = gaussian.target_count_from_percentage(percentage)
                unavailable = [
                    beam.beam_id for beam in morphology.beams
                    if target_count not in beam.validated_target_counts
                ]
                if unavailable:
                    preview = ",".join(map(str, unavailable[:12]))
                    suffix = "..." if len(unavailable) > 12 else ""
                    rows.append(blank_failed_row(
                        campaign_id=campaign_id,
                        args=args,
                        trial=trial,
                        percentage=percentage,
                        status="failed_template_unattainable",
                        reason=(
                            f"rounded per-beam target {target_count} did not "
                            f"calibrate within {args.template_attempt_limit} "
                            f"deterministic attempts for beam(s) {preview}{suffix}"
                        ),
                    ))
                    gaussian.validate_trial_prefix(
                        rows, args.trials, args.percentages, campaign_id
                    )
                    checkpoint()
                    continue
                try:
                    target_host, calibrations = (
                        gaussian.calibrate_trial_morphology(
                            morphology,
                            bundle.specs,
                            percentage,
                            args.threshold,
                            pixel_tolerance=args.calibration_pixel_tolerance,
                            max_iterations=args.calibration_max_iterations,
                        )
                    )
                    pixel_counts = gaussian.validate_target_maps(
                        target_host, bundle.specs, args.total_beams,
                        args.threshold,
                    )
                except (ValueError, AssertionError) as exc:
                    rows.append(blank_failed_row(
                        campaign_id=campaign_id,
                        args=args,
                        trial=trial,
                        percentage=percentage,
                        status="failed_calibration",
                        reason=str(exc),
                    ))
                    gaussian.validate_trial_prefix(
                        rows, args.trials, args.percentages, campaign_id
                    )
                    checkpoint()
                    continue

                target_gpu = tuple(cp.asarray(array) for array in target_host)
                del target_host
                stream.synchronize()
                raw, raw_counts = gaussian.extract_complete_target_candidates(
                    cp,
                    peak_geometries,
                    target_gpu,
                    context_gpu,
                    argmax_gpu,
                    tuple(range(args.total_beams)),
                    args.threshold,
                    concatenate_raw_candidates,
                )
                del target_gpu
                total_raw = len(raw)

                if total_raw and not nonempty_warmed:
                    warm_raw = gaussian._raw_prefix(raw, min(32, total_raw))
                    warm_decoded = decoder.decode(warm_raw)
                    group_candidates_on_cpu(
                        warm_decoded,
                        grouping_geometry,
                        config=grouping_config,
                        cp_module=cp,
                    )
                    stream.synchronize()
                    nonempty_warmed = True

                decoder_wall_ms, decoded = gaussian.synchronized_wall_time(
                    stream, lambda: decoder.decode(raw)
                )
                decoded_count = len(decoded)
                if decoded_count != total_raw:
                    raise AssertionError(
                        "production decoder silently changed candidate count"
                    )

                cpu_wall_ms, grouped, diagnostics = (
                    synchronized_cpu_grouping_wall_time(
                        stream,
                        decoded,
                        grouping_geometry,
                        config=grouping_config,
                        cp_module=cp,
                    )
                )
                if (len(grouped.candidates) != decoded_count
                        or len(grouped.members) != decoded_count):
                    raise AssertionError("CPU grouping lost candidate members")
                event_counts = gaussian.gpu_per_beam_counts(
                    cp, grouped.events.beam_id, args.total_beams
                )
                if int(np.sum(event_counts)) != len(grouped.events):
                    raise AssertionError("per-beam event counts do not sum to total")

                verified = False
                if args.verify_production_reference:
                    within_limit = decoded_count <= args.candidate_safety_limit
                    if within_limit or args.allow_unsafe_grouping:
                        production = group_candidates(
                            decoded, grouping_geometry, config=grouping_config
                        )
                        stream.synchronize()
                        assert_gpu_grouping_results_equal(cp, production, grouped)
                        verified = True
                        del production
                    else:
                        print(
                            f"reference verification skipped for {decoded_count} "
                            f"candidates above limit {args.candidate_safety_limit}",
                            flush=True,
                        )

                row = build_trial_row(
                    campaign_id=campaign_id,
                    args=args,
                    morphology=morphology,
                    percentage=percentage,
                    calibrations=calibrations,
                    pixel_counts=pixel_counts,
                    raw_counts=raw_counts,
                    decoder_wall_ms=decoder_wall_ms,
                    decoded_count=decoded_count,
                    cpu_grouper_wall_ms=cpu_wall_ms,
                    diagnostics=diagnostics,
                    event_counts=event_counts,
                    production_reference_verified=verified,
                    chunk_duration_ms=bundle.chunk_duration_ms,
                )
                rows.append(row)
                gaussian.validate_trial_prefix(
                    rows, args.trials, args.percentages, campaign_id
                )
                checkpoint()
                print(
                    f"checkpoint trial={trial} corruption={percentage:g}% "
                    f"pixels={int(np.sum(pixel_counts))} candidates={total_raw} "
                    f"decoder={decoder_wall_ms:.3f}ms "
                    f"cpu_grouper={cpu_wall_ms:.3f}ms "
                    f"load={100.0*float(row['post_peakfinder_cpu_load_fraction']):.3f}%",
                    flush=True,
                )
                del raw, decoded, grouped, calibrations
                stream.synchronize()
            del morphology

        checkpoint()
        print("aggregate summary", flush=True)
        for percentage in args.percentages:
            _print_percentage_summary(rows, percentage)
        print(f"Results: {paths['directory']}", flush=True)


def main(argv: Sequence[str] | None = None) -> int:
    parser = _make_arg_parser()
    args = parser.parse_args(argv)
    try:
        validate_arguments(args)
    except (TypeError, ValueError) as exc:
        parser.error(str(exc))
    try:
        run_benchmark(args)
    except FileExistsError as exc:
        parser.error(str(exc))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
