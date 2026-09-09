#!/usr/bin/env python3
"""Compare production and persistent-GPU representative grouping side by side.

The scientific workload, calibration, streaming bowtie extraction, and decoder
are inherited unchanged from :mod:`benchmark_gaussian_corruption_timing`.  For
each trial/corruption point, one decoded GPU batch is passed unchanged to both
the production representative-seeded grouper and the benchmark-only persistent
GPU prototype.  Complete public calls are timed between current-stream
synchronizations, then every returned field is compared exactly outside the
authoritative timers.

CuPy, production PIRATE modules, and the prototype remain imported only inside
``run_benchmark`` so parser, schema, summary, and resume tests work on CPU-only
hosts.
"""

from __future__ import annotations

import argparse
import json
import math
import platform
import subprocess
import sys
from dataclasses import fields
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

import numpy as np
import yaml

from peakfinder_tests import benchmark_gaussian_corruption_timing as gaussian
from peakfinder_tests import benchmark_peakfinder_batch_timing as batch_benchmark


SCHEMA_NAME = "pirate-gaussian-corruption-gpu-representative-grouper-timing"
SCHEMA_VERSION = 2
REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_RESULTS_DIR = (
    Path(__file__).resolve().parent
    / "results_gaussian_corruption_gpu_representative_grouper_timing_pirate15"
)
NOTEBOOK_PATH = (
    Path(__file__).resolve().parent
    / "analyze_gaussian_corruption_gpu_representative_grouper_timing.ipynb"
)
TIMING_ORDER_POLICY = (
    "alternate by trial-major campaign ordinal, production first at even "
    "ordinals and experimental first at odd ordinals"
)


_WORKLOAD_FIELDS = gaussian.TRIAL_FIELDS[
    :gaussian.TRIAL_FIELDS.index("grouper_wall_ms")
]
TRIAL_FIELDS = (
    *_WORKLOAD_FIELDS,
    "partition_count",
    "largest_partition",
    "total_grouped_events",
    "grouped_events_per_beam_min",
    "grouped_events_per_beam_median",
    "grouped_events_per_beam_max",
    "production_grouper_wall_ms",
    "experimental_grouper_wall_ms",
    "production_candidates_per_second",
    "experimental_candidates_per_second",
    "production_over_experimental_speedup",
    "decoder_plus_production_grouper_wall_ms",
    "decoder_plus_experimental_grouper_wall_ms",
    "post_peakfinder_production_load_fraction",
    "post_peakfinder_experimental_load_fraction",
    "exact_parity_verified",
    "timing_order",
    "candidate_safety_limit",
    "status",
    "failure_reason",
    "failure_diagnostic_json",
)

SUMMARY_METRICS = (
    "total_pixels_above_threshold",
    "total_peakfinder_candidates",
    "peakfinder_survival_fraction",
    "decoder_wall_ms",
    "total_decoded_candidates",
    "partition_count",
    "largest_partition",
    "total_grouped_events",
    "production_grouper_wall_ms",
    "experimental_grouper_wall_ms",
    "production_candidates_per_second",
    "experimental_candidates_per_second",
    "production_over_experimental_speedup",
    "decoder_plus_production_grouper_wall_ms",
    "decoder_plus_experimental_grouper_wall_ms",
    "post_peakfinder_production_load_fraction",
    "post_peakfinder_experimental_load_fraction",
)
SUMMARY_STATISTICS = (
    "count", "minimum", "q25", "median", "q75", "maximum", "iqr"
)
SUMMARY_FIELDS = (
    "schema_version",
    "campaign_id",
    "target_corruption_percent",
    "recorded_configurations",
    "completed_trials",
    "skipped_trials",
    "parity_failed_trials",
    "failed_trials",
    *tuple(
        f"{metric}_{statistic}"
        for metric in SUMMARY_METRICS
        for statistic in SUMMARY_STATISTICS
    ),
)

_TIMED_FIELDS = (
    "production_grouper_wall_ms",
    "experimental_grouper_wall_ms",
    "production_candidates_per_second",
    "experimental_candidates_per_second",
    "production_over_experimental_speedup",
    "decoder_plus_production_grouper_wall_ms",
    "decoder_plus_experimental_grouper_wall_ms",
    "post_peakfinder_production_load_fraction",
    "post_peakfinder_experimental_load_fraction",
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
        "failures": directory / "failure_diagnostics.yaml",
    }


def _timing_order(trial: int, percentage_index: int,
                  percentage_count: int) -> str:
    """Return a resume-stable alternating order for one campaign point."""

    ordinal = int(trial) * int(percentage_count) + int(percentage_index)
    return (
        "production_then_experimental"
        if ordinal % 2 == 0 else "experimental_then_production"
    )


def _percentage_index(percentage: Any,
                      percentages: Sequence[float]) -> int:
    target = gaussian.canonical_percentage(percentage)
    matching = [
        index for index, value in enumerate(percentages)
        if gaussian.canonical_percentage(value) == target
    ]
    if len(matching) != 1:
        raise ValueError(f"percentage {percentage!r} is not unique in campaign grid")
    return matching[0]


def _rate(candidate_count: int, wall_ms: float | None) -> Any:
    if wall_ms is None:
        return ""
    if candidate_count == 0 or wall_ms == 0.0:
        return 0.0
    return 1_000.0 * candidate_count / wall_ms


def _post_metrics(
        decoder_wall_ms: float, grouper_wall_ms: float | None,
        decoded_count: int, chunk_duration_ms: float,
        ) -> tuple[Any, Any]:
    if grouper_wall_ms is None:
        return "", ""
    combined = decoder_wall_ms + grouper_wall_ms
    return combined, combined / chunk_duration_ms


def build_trial_row(
        *, campaign_id: str, args: argparse.Namespace,
        morphology: gaussian.TrialMorphology, percentage: float,
        calibrations: Sequence[gaussian.BeamCalibration],
        pixel_counts: np.ndarray, raw_counts: np.ndarray,
        decoder_wall_ms: float, decoded_count: int,
        partition_count: int, largest_partition: int,
        event_counts: np.ndarray | None,
        production_wall_ms: float | None,
        experimental_wall_ms: float | None,
        exact_parity_verified: bool, timing_order: str, status: str,
        chunk_duration_ms: float, failure_reason: str = "",
        failure_diagnostic: Mapping[str, Any] | None = None,
        ) -> dict[str, Any]:
    """Build one strict side-by-side row from the reviewed workload schema."""

    base = gaussian.build_trial_row(
        campaign_id=campaign_id,
        args=args,
        morphology=morphology,
        percentage=percentage,
        calibrations=calibrations,
        pixel_counts=pixel_counts,
        raw_counts=raw_counts,
        decoder_wall_ms=decoder_wall_ms,
        decoded_count=decoded_count,
        grouper_wall_ms=None,
        event_counts=None,
        status=status,
        chunk_duration_ms=chunk_duration_ms,
        failure_reason=failure_reason,
    )
    row = {name: base[name] for name in _WORKLOAD_FIELDS}
    if event_counts is None:
        total_events: Any = ""
        event_summary = {"min": "", "median": "", "max": ""}
    else:
        total_events = int(np.sum(event_counts))
        event_summary = gaussian.summarize_counts(event_counts)
    production_combined, production_load = _post_metrics(
        decoder_wall_ms, production_wall_ms, decoded_count, chunk_duration_ms
    )
    experimental_combined, experimental_load = _post_metrics(
        decoder_wall_ms, experimental_wall_ms, decoded_count, chunk_duration_ms
    )
    speedup: Any = ""
    if exact_parity_verified and experimental_wall_ms is not None:
        speedup = (
            0.0 if production_wall_ms is None
            else (
                math.inf if experimental_wall_ms == 0.0
                else production_wall_ms / experimental_wall_ms
            )
        )
    diagnostic_json = (
        "" if failure_diagnostic is None else json.dumps(
            dict(failure_diagnostic), sort_keys=True, separators=(",", ":"),
            allow_nan=False,
        )
    )
    row.update({
        "partition_count": partition_count,
        "largest_partition": largest_partition,
        "total_grouped_events": total_events,
        "grouped_events_per_beam_min": event_summary["min"],
        "grouped_events_per_beam_median": event_summary["median"],
        "grouped_events_per_beam_max": event_summary["max"],
        "production_grouper_wall_ms": (
            "" if production_wall_ms is None else production_wall_ms
        ),
        "experimental_grouper_wall_ms": (
            "" if experimental_wall_ms is None else experimental_wall_ms
        ),
        "production_candidates_per_second": _rate(
            decoded_count, production_wall_ms
        ),
        "experimental_candidates_per_second": _rate(
            decoded_count, experimental_wall_ms
        ),
        "production_over_experimental_speedup": speedup,
        "decoder_plus_production_grouper_wall_ms": production_combined,
        "decoder_plus_experimental_grouper_wall_ms": experimental_combined,
        "post_peakfinder_production_load_fraction": production_load,
        "post_peakfinder_experimental_load_fraction": experimental_load,
        "exact_parity_verified": int(exact_parity_verified),
        "timing_order": timing_order,
        "candidate_safety_limit": args.candidate_safety_limit,
        "status": status,
        "failure_reason": failure_reason,
        "failure_diagnostic_json": diagnostic_json,
    })
    return _validate_rows([row], TRIAL_FIELDS)[0]


def blank_failed_row(
        *, campaign_id: str, args: argparse.Namespace, trial: int,
        percentage: float, percentage_index: int, status: str, reason: str,
        failure_diagnostic: Mapping[str, Any] | None = None,
        ) -> dict[str, Any]:
    base = gaussian._blank_failed_row(
        campaign_id=campaign_id,
        args=args,
        trial=trial,
        percentage=percentage,
        status=status,
        reason=reason,
    )
    row = {field: "" for field in TRIAL_FIELDS}
    row.update({name: base[name] for name in _WORKLOAD_FIELDS})
    row.update({
        "timing_order": _timing_order(
            trial, percentage_index, len(args.percentages)
        ),
        "candidate_safety_limit": args.candidate_safety_limit,
        "status": status,
        "failure_reason": reason,
        "failure_diagnostic_json": (
            "" if failure_diagnostic is None else json.dumps(
                dict(failure_diagnostic), sort_keys=True,
                separators=(",", ":"), allow_nan=False,
            )
        ),
    })
    return _validate_rows([row], TRIAL_FIELDS)[0]


def summarize_trials(
        rows: Sequence[Mapping[str, Any]], percentages: Sequence[float],
        campaign_id: str,
        ) -> list[dict[str, Any]]:
    """Summarize exact-parity completed rows and retain all status counts."""

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
            "skipped_trials": sum(
                row["status"] == "skipped_candidate_safety_limit"
                for row in matching
            ),
            "parity_failed_trials": sum(
                row["status"] == "failed_parity" for row in matching
            ),
            "failed_trials": sum(
                str(row["status"]).startswith("failed_")
                and row["status"] != "failed_parity"
                for row in matching
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
                    "count": int(values.size),
                    "minimum": float(np.min(values)),
                    "q25": q25,
                    "median": float(np.median(values)),
                    "q75": q75,
                    "maximum": float(np.max(values)),
                    "iqr": q75 - q25,
                }
            else:
                statistics = {
                    "count": 0,
                    **{name: "" for name in SUMMARY_STATISTICS if name != "count"},
                }
            for statistic, value in statistics.items():
                summary[f"{metric}_{statistic}"] = value
        result.append(summary)
    return _validate_rows(result, SUMMARY_FIELDS)


def _failure_records(
        rows: Sequence[Mapping[str, Any]],
        ) -> list[dict[str, Any]]:
    records = []
    for row in rows:
        status = str(row["status"])
        if not status.startswith("failed_"):
            continue
        encoded = str(row.get("failure_diagnostic_json", "")).strip()
        diagnostic = json.loads(encoded) if encoded else None
        records.append({
            "trial": int(row["trial"]),
            "target_corruption_percent": gaussian.canonical_percentage(
                row["target_corruption_percent"]
            ),
            "status": status,
            "failure_reason": str(row["failure_reason"]),
            "diagnostic": diagnostic,
        })
    return records


def failure_document(
        rows: Sequence[Mapping[str, Any]], campaign_id: str,
        ) -> dict[str, Any]:
    return {
        "schema_name": SCHEMA_NAME,
        "schema_version": SCHEMA_VERSION,
        "campaign_id": campaign_id,
        "diagnostics": _failure_records(rows),
    }


def write_checkpoint(
        paths: Mapping[str, Path], rows: Sequence[Mapping[str, Any]],
        percentages: Sequence[float], campaign_id: str,
        metadata: Mapping[str, Any],
        ) -> None:
    # Metadata is the final commit marker; every preceding derived file can be
    # checked exactly against trials.csv during resume.
    gaussian._atomic_write_csv(paths["trials"], TRIAL_FIELDS, rows)
    gaussian._atomic_write_csv(
        paths["summary"], SUMMARY_FIELDS,
        summarize_trials(rows, percentages, campaign_id),
    )
    gaussian._atomic_write_yaml(
        paths["failures"], failure_document(rows, campaign_id)
    )
    gaussian._atomic_write_yaml(paths["metadata"], metadata)


def _csv_rows_equal(
        observed: Sequence[Mapping[str, Any]],
        expected: Sequence[Mapping[str, Any]], fields_: Sequence[str],
        ) -> bool:
    return len(observed) == len(expected) and all(
        all(str(left[field]) == str(right[field]) for field in fields_)
        for left, right in zip(observed, expected)
    )


def prepare_result_state(
        paths: Mapping[str, Path], *, resume: bool, overwrite: bool,
        signature: str, campaign_id: str, trials: int,
        percentages: Sequence[float],
        ) -> tuple[list[dict[str, Any]], str]:
    """Validate a complete four-file checkpoint or authorize a fresh one."""

    if resume and overwrite:
        raise ValueError("resume and overwrite are mutually exclusive")
    files = [
        paths[name] for name in ("trials", "summary", "failures", "metadata")
    ]
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
        raise ValueError("resume requires all four result files")

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
    gaussian.validate_trial_prefix(rows, trials, percentages, campaign_id)
    observed_summary = gaussian._read_csv_exact(
        paths["summary"], SUMMARY_FIELDS
    )
    expected_summary = summarize_trials(rows, percentages, campaign_id)
    if not _csv_rows_equal(observed_summary, expected_summary, SUMMARY_FIELDS):
        raise ValueError("summary.csv is inconsistent with trials.csv")

    with paths["failures"].open("r", encoding="utf-8") as stream:
        failures = yaml.safe_load(stream)
    if failures != failure_document(rows, campaign_id):
        raise ValueError(
            "failure_diagnostics.yaml is inconsistent with trials.csv"
        )
    return rows, str(metadata.get("created_utc") or gaussian._utc_now())


def _optional_float(value: Any, context: str) -> float | None:
    if value is None or not str(value).strip():
        return None
    result = float(value)
    if not math.isfinite(result):
        raise ValueError(f"{context} is not finite")
    return result


def _close(actual: Any, expected: float, context: str) -> None:
    value = float(actual)
    if (not math.isfinite(value)
            or not math.isclose(
                value, expected, rel_tol=1.0e-12, abs_tol=1.0e-12
            )):
        raise ValueError(f"{context} is inconsistent")


def validate_resumed_trial_rows(
        rows: Sequence[Mapping[str, Any]], args: argparse.Namespace,
        campaign_id: str, chunk_duration_ms: float,
        ) -> None:
    """Validate inherited science and all side-by-side timing/status rules."""

    gaussian_rows = []
    inherited_failures = {
        "failed_template_unattainable", "failed_calibration"
    }
    for row in rows:
        status = str(row["status"])
        shared = {field: "" for field in gaussian.TRIAL_FIELDS}
        shared.update({field: row[field] for field in _WORKLOAD_FIELDS})
        shared["candidate_safety_limit"] = args.candidate_safety_limit
        shared["failure_reason"] = row["failure_reason"]
        if status in inherited_failures:
            shared["status"] = status
        elif status == "skipped_candidate_safety_limit":
            shared["status"] = status
        else:
            # Reuse the parent validator for every completed scientific
            # workload.  Failed comparison rows may lack one timing, so a zero
            # surrogate is used only for this inherited consistency check;
            # persisted side-by-side timings are checked explicitly below.
            production_ms = _optional_float(
                row["production_grouper_wall_ms"], "production timing"
            )
            production_ms = 0.0 if production_ms is None else production_ms
            decoded = int(row["total_decoded_candidates"])
            events = (
                0 if not str(row["total_grouped_events"]).strip()
                else int(row["total_grouped_events"])
            )
            combined = float(row["decoder_wall_ms"]) + production_ms
            shared.update({
                "grouper_wall_ms": production_ms,
                "grouper_candidates_per_second": (
                    0.0 if decoded == 0 or production_ms == 0.0
                    else 1_000.0 * decoded / production_ms
                ),
                "total_grouped_events": events,
                "grouped_events_per_beam_min": (
                    row["grouped_events_per_beam_min"] or 0
                ),
                "grouped_events_per_beam_median": (
                    row["grouped_events_per_beam_median"] or 0
                ),
                "grouped_events_per_beam_max": (
                    row["grouped_events_per_beam_max"] or 0
                ),
                "decoder_plus_grouper_wall_ms": combined,
                "post_peakfinder_load_fraction": (
                    combined / chunk_duration_ms
                ),
                "status": "completed",
                "failure_reason": "",
            })
        gaussian_rows.append(shared)
    gaussian.validate_resumed_trial_rows(
        gaussian_rows, args, campaign_id, chunk_duration_ms
    )

    allowed = {
        "completed",
        "skipped_candidate_safety_limit",
        "failed_template_unattainable",
        "failed_calibration",
        "failed_grouping",
        "failed_parity",
    }
    for row_index, row in enumerate(rows):
        context = f"resumed trials.csv row {row_index + 2}"
        status = str(row["status"])
        if status not in allowed:
            raise ValueError(f"{context}: unsupported status {status!r}")
        trial = int(row["trial"])
        percentage_index = _percentage_index(
            row["target_corruption_percent"], args.percentages
        )
        expected_order = _timing_order(
            trial, percentage_index, len(args.percentages)
        )
        if str(row["timing_order"]) != expected_order:
            raise ValueError(f"{context}: timing order is inconsistent")
        if int(row["candidate_safety_limit"]) != args.candidate_safety_limit:
            raise ValueError(f"{context}: candidate safety limit is inconsistent")

        if status in inherited_failures:
            if not str(row["failure_reason"]).strip():
                raise ValueError(f"{context}: failed row lacks a reason")
            continue

        decoded = int(row["total_decoded_candidates"])
        partitions = int(row["partition_count"])
        largest = int(row["largest_partition"])
        if decoded == 0:
            valid_partition_shape = partitions == 0 and largest == 0
        else:
            valid_partition_shape = (
                1 <= partitions <= decoded and 1 <= largest <= decoded
            )
        if not valid_partition_shape:
            raise ValueError(f"{context}: partition diagnostics are inconsistent")

        if status == "skipped_candidate_safety_limit":
            if decoded <= args.candidate_safety_limit:
                raise ValueError(f"{context}: safety skip is not above the limit")
            for field_name in (
                    "total_grouped_events",
                    "grouped_events_per_beam_min",
                    "grouped_events_per_beam_median",
                    "grouped_events_per_beam_max",
                    *_TIMED_FIELDS,
                    "failure_diagnostic_json"):
                if str(row[field_name]).strip():
                    raise ValueError(
                        f"{context}: safety-skipped {field_name} must be empty"
                    )
            if int(row["exact_parity_verified"]) != 0:
                raise ValueError(f"{context}: safety skip cannot claim parity")
            continue

        if status.startswith("failed_") and not str(row["failure_reason"]).strip():
            raise ValueError(f"{context}: failed row lacks a reason")

        production_ms = _optional_float(
            row["production_grouper_wall_ms"], f"{context}: production timing"
        )
        experimental_ms = _optional_float(
            row["experimental_grouper_wall_ms"],
            f"{context}: experimental timing",
        )
        if status in {"completed", "failed_parity"}:
            if (production_ms is None or experimental_ms is None
                    or production_ms <= 0.0 or experimental_ms <= 0.0):
                raise ValueError(f"{context}: comparison timings must be positive")
            encoded_events = str(row["total_grouped_events"]).strip()
            if status == "completed" and not encoded_events:
                raise ValueError(f"{context}: completed event count is missing")
            if encoded_events:
                events = int(encoded_events)
                if not (0 <= events <= decoded) or (decoded and events == 0):
                    raise ValueError(
                        f"{context}: grouped-event count is inconsistent"
                    )
            _close(
                row["production_candidates_per_second"],
                0.0 if decoded == 0 else 1_000.0 * decoded / production_ms,
                f"{context}: production throughput",
            )
            _close(
                row["experimental_candidates_per_second"],
                0.0 if decoded == 0 else 1_000.0 * decoded / experimental_ms,
                f"{context}: experimental throughput",
            )
            decoder_ms = float(row["decoder_wall_ms"])
            for prefix, wall_ms in (
                    ("production", production_ms),
                    ("experimental", experimental_ms)):
                combined = decoder_ms + wall_ms
                _close(
                    row[f"decoder_plus_{prefix}_grouper_wall_ms"], combined,
                    f"{context}: decoder-plus-{prefix}",
                )
                _close(
                    row[f"post_peakfinder_{prefix}_load_fraction"],
                    combined / chunk_duration_ms,
                    f"{context}: {prefix} load",
                )

        parity = int(row["exact_parity_verified"])
        diagnostic = str(row["failure_diagnostic_json"]).strip()
        if status == "completed":
            if parity != 1 or diagnostic or str(row["failure_reason"]).strip():
                raise ValueError(f"{context}: completed parity state is inconsistent")
            _close(
                row["production_over_experimental_speedup"],
                production_ms / experimental_ms,
                f"{context}: speedup",
            )
        elif status == "failed_parity":
            if parity != 0 or not diagnostic:
                raise ValueError(f"{context}: parity-failure diagnostic is missing")
            if str(row["production_over_experimental_speedup"]).strip():
                raise ValueError(f"{context}: parity failure must not report speedup")
            try:
                parsed = json.loads(diagnostic)
            except json.JSONDecodeError as exc:
                raise ValueError(
                    f"{context}: parity diagnostic is invalid JSON"
                ) from exc
            if parsed.get("kind") != "exact_parity_mismatch":
                raise ValueError(f"{context}: parity diagnostic kind is invalid")
        elif status == "failed_grouping":
            if parity != 0 or not diagnostic:
                raise ValueError(f"{context}: grouping-failure diagnostic is missing")
            if str(row["production_over_experimental_speedup"]).strip():
                raise ValueError(f"{context}: grouping failure must not report speedup")


def _json_safe(value: Any) -> Any:
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, Mapping):
        return {str(key): _json_safe(item) for key, item in value.items()}
    if isinstance(value, (tuple, list)):
        return [_json_safe(item) for item in value]
    return value


def _benchmark_source_information() -> dict[str, Any]:
    paths = {
        "side_by_side_benchmark": Path(__file__).resolve(),
        "gpu_representative_grouper": (
            REPOSITORY_ROOT / "peakfinder_tests/gpu_representative_grouper.py"
        ).resolve(),
        "gaussian_corruption_helpers": Path(gaussian.__file__).resolve(),
        "peakfinder_batch_helpers": Path(batch_benchmark.__file__).resolve(),
        "producer_metadata": REPOSITORY_ROOT / "peakfinder_tests/producer_metadata.py",
    }
    return {
        name: {"path": str(path), "sha256": gaussian._sha256_path(path)}
        for name, path in paths.items()
    }


def _git_information() -> dict[str, Any]:
    information = dict(batch_benchmark.git_information())
    try:
        branch = subprocess.check_output(
            ("git", "branch", "--show-current"), cwd=REPOSITORY_ROOT,
            text=True, stderr=subprocess.DEVNULL,
        ).strip()
    except (OSError, subprocess.CalledProcessError):
        branch = ""
    information["branch"] = branch or None
    return information


def _gpu_information(cp: Any, device: int) -> dict[str, Any]:
    information = dict(batch_benchmark.gpu_information(cp, device))
    properties = cp.cuda.runtime.getDeviceProperties(device)

    def property_value(name: str, default: int = 0) -> int:
        return int(properties.get(name, properties.get(name.encode(), default)))

    major = property_value("major")
    minor = property_value("minor")
    information.update({
        "compute_capability": f"{major}.{minor}",
        "compute_capability_major": major,
        "compute_capability_minor": minor,
    })
    return information


def signature_payload(
        args: argparse.Namespace, bundle: Any,
        gaussian_config: gaussian.GaussianConfig,
        gpu: Mapping[str, Any], software: Mapping[str, Any],
        git: Mapping[str, Any], prototype: Mapping[str, Any],
        ) -> dict[str, Any]:
    payload = gaussian._signature_payload(
        args, bundle, gaussian_config, gpu, software
    )
    payload.update({
        "schema_name": SCHEMA_NAME,
        "schema_version": SCHEMA_VERSION,
        "comparison_policy": (
            "same decoded GPU object; complete synchronized production and "
            "persistent-GPU public calls; exact all-field parity outside timers"
        ),
        "timing_order_policy": TIMING_ORDER_POLICY,
        "prototype": dict(prototype),
        "gpu": dict(gpu),
        "git_commit": git.get("commit"),
        "git_branch": git.get("branch"),
        "benchmark_sources": _benchmark_source_information(),
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
        prototype: Mapping[str, Any],
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
    metadata["side_by_side_grouping"] = {
        "production_implementation": (
            "pirate_frb.OfflineCandidateGrouper.group_candidates"
        ),
        "experimental_implementation": prototype["implementation_name"],
        "identical_input_policy": (
            "maps are generated/calibrated once, bowtie peakfinding runs once, "
            "decoding runs once, and the identical decoded GPU batch is passed "
            "unchanged to both complete public calls"
        ),
        "partition_key": ["beam_id", "primary_tree_index"],
        "partition_diagnostics": (
            "computed outside authoritative timers after both runnable calls "
            "(immediately only for safety skips), with aggregate scalar transfers "
            "only and no candidate-sized host copy"
        ),
        "prototype_kernel": dict(prototype),
    }
    metadata["side_by_side_timing"] = {
        "authoritative_metrics": [
            "production_grouper_wall_ms", "experimental_grouper_wall_ms"
        ],
        "clock": "time.perf_counter",
        "boundary": (
            "synchronize current CuPy stream; call the complete public grouper; "
            "synchronize the same stream; elapsed synchronized wall time"
        ),
        "included": [
            "candidate normalization and validation",
            "exact processing-order construction",
            "partition construction in the prototype",
            "persistent RawKernel execution",
            "event-ID compaction/remapping",
            "event and member table construction",
            "fixed aggregate synchronizations required by each public call",
        ],
        "excluded": [
            "map generation, calibration, upload, peakfinding, decoding",
            "partition diagnostics", "parity host checks", "output writing",
            "plotting and diagnostic formatting",
        ],
        "warmup": (
            "kernel compilation plus empty paths and the first safe non-empty "
            "prefix for both implementations occur before authoritative timing"
        ),
        "ordering_bias_control": signature_payload_.get(
            "timing_order_policy", TIMING_ORDER_POLICY
        ),
        "combined_metrics": (
            "decoder-plus-grouper values sum separately synchronized decoder and "
            "grouper measurements; they are not fused intervals"
        ),
    }
    metadata["exact_parity"] = {
        "required_for_completed_status": True,
        "comparison_scope": (
            "candidate_event_id and every candidate/event/member field, including "
            "shape, dtype, order, and values; NaNs compare equal"
        ),
        "timed": False,
        "failure_status": "failed_parity",
        "speedup_policy": (
            "production/experimental speedup is emitted only after exact parity"
        ),
        "diagnostic_fields": [
            "table", "field", "first_differing_index", "expected_dtype",
            "actual_dtype", "expected_shape", "actual_shape", "expected_value",
            "actual_value", "candidate_count", "partition_count",
            "largest_partition",
        ],
        "schema_difference_index_convention": (
            "first_differing_index=[] means the mismatch occurred before a "
            "common array value coordinate existed; value mismatches use the "
            "non-empty multidimensional integer coordinate"
        ),
    }
    metadata["safety_limit"] = {
        "candidate_limit": args.candidate_safety_limit,
        "unsafe_override": bool(args.allow_unsafe_grouping),
        "policy": (
            "decode and aggregate partition diagnostics are retained; above the "
            "limit both grouping comparisons are skipped with "
            "status=skipped_candidate_safety_limit unless explicitly unsafe"
        ),
    }
    metadata["cli_arguments"] = _json_safe(vars(args))
    metadata["environment"]["gpu"] = dict(gpu)
    metadata["environment"]["git"] = dict(git)
    metadata["environment"]["benchmark_sources"] = (
        _benchmark_source_information()
    )
    metadata["csv_schemas"] = {
        "trials.csv": list(TRIAL_FIELDS),
        "summary.csv": list(SUMMARY_FIELDS),
        "summary_metrics": list(SUMMARY_METRICS),
        "summary_statistics": list(SUMMARY_STATISTICS),
        "zero_denominator_peakfinder_survival_fraction": 0.0,
    }
    metadata["failure_diagnostics"] = {
        "path": "failure_diagnostics.yaml",
        "recorded_failures": len(_failure_records(rows)),
        "policy": (
            "one structured record per failed trial/configuration; parity records "
            "include the first exact field difference and partition context"
        ),
    }
    metadata["outputs"] = {
        "trials": "trials.csv",
        "summary": "summary.csv",
        "metadata": "metadata.yaml",
        "failure_diagnostics": "failure_diagnostics.yaml",
        "analysis_notebook": str(NOTEBOOK_PATH.resolve()),
    }
    return metadata


def _make_arg_parser() -> argparse.ArgumentParser:
    parser = gaussian._make_arg_parser()
    parser.description = (
        "Side-by-side Monte Carlo Gaussian-corruption timing of production and "
        "persistent-GPU representative-seeded groupers"
    )
    parser.set_defaults(results_dir=str(DEFAULT_RESULTS_DIR))
    return parser


def validate_arguments(args: argparse.Namespace) -> argparse.Namespace:
    return gaussian.validate_arguments(args)


def _scalar_value(value: Any) -> Any:
    result = value.item() if hasattr(value, "item") else value
    if isinstance(result, np.generic):
        result = result.item()
    if isinstance(result, float) and not math.isfinite(result):
        if math.isnan(result):
            return "NaN"
        return "Infinity" if result > 0.0 else "-Infinity"
    if isinstance(result, complex):
        return {
            "real": _scalar_value(result.real),
            "imaginary": _scalar_value(result.imag),
        }
    if result is None or isinstance(result, (bool, int, float, str)):
        return result
    return repr(result)


def _schema_difference(
        *, table: str, field: str, expected: Any, actual: Any,
        mismatch_kind: str, diagnostic_error: str | None = None,
        ) -> dict[str, Any]:
    """Describe a mismatch that occurs before a common value index exists.

    An empty first_differing_index list is the explicit convention for a
    missing result/table/field, malformed array, shape mismatch, dtype
    mismatch, or comparator failure. Non-empty integer lists are reserved for
    real array value coordinates.
    """

    def descriptor(value: Any) -> tuple[Any, Any]:
        if value is None:
            return None, None
        try:
            shape = getattr(value, "shape", None)
            normalized_shape = None if shape is None else list(shape)
        except Exception:
            normalized_shape = None
        try:
            dtype = getattr(value, "dtype", None)
        except Exception:
            dtype = None
        return normalized_shape, None if dtype is None else str(dtype)

    expected_shape, expected_dtype = descriptor(expected)
    actual_shape, actual_dtype = descriptor(actual)
    result = {
        "table": table,
        "field": field,
        "mismatch_kind": mismatch_kind,
        "first_differing_index": [],
        "expected_dtype": expected_dtype,
        "actual_dtype": actual_dtype,
        "expected_shape": expected_shape,
        "actual_shape": actual_shape,
        "expected_value": "missing" if expected is None else "present",
        "actual_value": "missing" if actual is None else "present",
    }
    if diagnostic_error is not None:
        result["diagnostic_error"] = diagnostic_error
    return result


def _attribute_or_difference(
        owner: Any, name: str, *, table: str, field: str, side: str,
        ) -> tuple[Any, dict[str, Any] | None]:
    """Read one result attribute without allowing diagnostics to abort."""

    try:
        value = getattr(owner, name)
    except AttributeError:
        value = None
    except Exception as exc:
        return None, _schema_difference(
            table=table, field=field,
            expected=owner if side == "expected" else None,
            actual=owner if side == "actual" else None,
            mismatch_kind="attribute_access_exception",
            diagnostic_error=f"{side}: {type(exc).__name__}: {exc}",
        )
    return value, None


def _array_difference(
        cp: Any, *, table: str, field: str, expected: Any, actual: Any,
        ) -> dict[str, Any] | None:
    """Compare one required one-dimensional GPU field exactly and totally."""

    if expected is None or actual is None:
        return _schema_difference(
            table=table, field=field, expected=expected, actual=actual,
            mismatch_kind="missing_field",
        )
    try:
        expected_shape = tuple(int(value) for value in expected.shape)
        actual_shape = tuple(int(value) for value in actual.shape)
        expected_dtype = expected.dtype
        actual_dtype = actual.dtype
    except Exception as exc:
        return _schema_difference(
            table=table, field=field, expected=expected, actual=actual,
            mismatch_kind="malformed_array",
            diagnostic_error=f"{type(exc).__name__}: {exc}",
        )
    if len(expected_shape) != 1 or len(actual_shape) != 1:
        return _schema_difference(
            table=table, field=field, expected=expected, actual=actual,
            mismatch_kind="malformed_shape",
        )
    if expected_shape != actual_shape:
        return _schema_difference(
            table=table, field=field, expected=expected, actual=actual,
            mismatch_kind="shape_mismatch",
        )
    if expected_dtype != actual_dtype:
        return _schema_difference(
            table=table, field=field, expected=expected, actual=actual,
            mismatch_kind="dtype_mismatch",
        )
    try:
        equal = expected == actual
        if expected_dtype.kind in "fc":
            equal = equal | (cp.isnan(expected) & cp.isnan(actual))
        if bool(_scalar_value(cp.all(equal))):
            return None
        different = cp.flatnonzero(~equal.ravel())
        flat_index = int(_scalar_value(different[0]))
        index = [int(value) for value in np.unravel_index(
            flat_index, expected_shape
        )]
        return {
            "table": table,
            "field": field,
            "mismatch_kind": "value",
            "first_differing_index": index,
            "expected_dtype": str(expected_dtype),
            "actual_dtype": str(actual_dtype),
            "expected_shape": list(expected_shape),
            "actual_shape": list(actual_shape),
            "expected_value": _scalar_value(expected.ravel()[flat_index]),
            "actual_value": _scalar_value(actual.ravel()[flat_index]),
        }
    except Exception as exc:
        return _schema_difference(
            table=table, field=field, expected=expected, actual=actual,
            mismatch_kind="comparator_exception",
            diagnostic_error=f"{type(exc).__name__}: {exc}",
        )


def first_grouping_difference(cp: Any, expected: Any,
                              actual: Any) -> dict[str, Any] | None:
    """Return the first exact difference, including malformed schemas."""

    if expected is None or actual is None:
        return _schema_difference(
            table="result", field="<result>", expected=expected, actual=actual,
            mismatch_kind="missing_result",
        )

    expected_assignment, error = _attribute_or_difference(
        expected, "candidate_event_id", table="result",
        field="candidate_event_id", side="expected",
    )
    if error is not None:
        return error
    actual_assignment, error = _attribute_or_difference(
        actual, "candidate_event_id", table="result",
        field="candidate_event_id", side="actual",
    )
    if error is not None:
        return error
    mismatch = _array_difference(
        cp, table="result", field="candidate_event_id",
        expected=expected_assignment, actual=actual_assignment,
    )
    if mismatch is not None:
        return mismatch

    for table_name in ("candidates", "events", "members"):
        expected_table, error = _attribute_or_difference(
            expected, table_name, table=table_name, field="<table>",
            side="expected",
        )
        if error is not None:
            return error
        actual_table, error = _attribute_or_difference(
            actual, table_name, table=table_name, field="<table>",
            side="actual",
        )
        if error is not None:
            return error
        if expected_table is None or actual_table is None:
            return _schema_difference(
                table=table_name, field="<table>", expected=expected_table,
                actual=actual_table, mismatch_kind="missing_table",
            )
        try:
            expected_fields = tuple(field.name for field in fields(expected_table))
            actual_fields = tuple(field.name for field in fields(actual_table))
        except Exception as exc:
            return _schema_difference(
                table=table_name, field="<table>", expected=expected_table,
                actual=actual_table, mismatch_kind="malformed_table",
                diagnostic_error=f"{type(exc).__name__}: {exc}",
            )
        if expected_fields != actual_fields:
            first_field = next(
                (
                    name for index, name in enumerate(expected_fields)
                    if index >= len(actual_fields)
                    or actual_fields[index] != name
                ),
                actual_fields[len(expected_fields)]
                if len(actual_fields) > len(expected_fields) else "<fields>",
            )
            return _schema_difference(
                table=table_name, field=first_field,
                expected=expected_table, actual=actual_table,
                mismatch_kind="field_schema_mismatch",
            )
        for field_name in expected_fields:
            expected_value, error = _attribute_or_difference(
                expected_table, field_name, table=table_name,
                field=field_name, side="expected",
            )
            if error is not None:
                return error
            actual_value, error = _attribute_or_difference(
                actual_table, field_name, table=table_name,
                field=field_name, side="actual",
            )
            if error is not None:
                return error
            mismatch = _array_difference(
                cp, table=table_name, field=field_name,
                expected=expected_value, actual=actual_value,
            )
            if mismatch is not None:
                return mismatch
    return None


def _comparison_diagnostic(
        *, mismatch: Mapping[str, Any], candidate_count: int,
        partition_count: int, largest_partition: int,
        ) -> dict[str, Any]:
    return {
        "kind": "exact_parity_mismatch",
        **dict(mismatch),
        "candidate_count": candidate_count,
        "partition_count": partition_count,
        "largest_partition": largest_partition,
    }


def _exception_diagnostic(
        errors: Mapping[str, BaseException], *, candidate_count: int,
        partition_count: int, largest_partition: int,
        ) -> dict[str, Any]:
    return {
        "kind": "grouping_exception",
        "errors": {
            name: {
                "exception_type": type(error).__name__,
                "message": str(error),
            }
            for name, error in errors.items()
        },
        "candidate_count": candidate_count,
        "partition_count": partition_count,
        "largest_partition": largest_partition,
    }


def _print_percentage_summary(
        rows: Sequence[Mapping[str, Any]], percentage: float,
        ) -> None:
    matching = [
        row for row in rows
        if gaussian.canonical_percentage(row["target_corruption_percent"])
        == gaussian.canonical_percentage(percentage)
    ]
    completed = [row for row in matching if row["status"] == "completed"]
    if not completed:
        print(
            f"corruption={percentage:g}% recorded={len(matching)} completed=0",
            flush=True,
        )
        return
    candidates = np.asarray([
        float(row["total_decoded_candidates"]) for row in completed
    ])
    production = np.asarray([
        float(row["production_grouper_wall_ms"]) for row in completed
    ])
    experimental = np.asarray([
        float(row["experimental_grouper_wall_ms"]) for row in completed
    ])
    speedup = np.asarray([
        float(row["production_over_experimental_speedup"])
        for row in completed
    ])
    print(
        f"corruption={percentage:g}% completed={len(completed)} "
        f"candidates_median={np.median(candidates):.0f} "
        f"production_median={np.median(production):.3f}ms "
        f"experimental_median={np.median(experimental):.3f}ms "
        f"speedup_median={np.median(speedup):.3f}x",
        flush=True,
    )


def run_benchmark(args: argparse.Namespace) -> None:
    args = validate_arguments(args)
    gaussian_config = gaussian._make_gaussian_config(args)
    paths = result_paths(args.results_dir)

    import cupy as cp
    from pirate_frb.GpuArgmaxDecoder import GpuArgmaxDecoder
    from pirate_frb.OfflineCandidateGrouper import (
        GpuDecodedCandidates,
        GroupingConfig,
        GroupingGeometry,
        group_candidates,
    )
    from pirate_frb.Peakfinders import (
        GpuRawCandidates,
        concatenate_raw_candidates,
    )
    from peakfinder_tests.gpu_representative_grouper import (
        PROTOTYPE_IMPLEMENTATION_NAME,
        RAW_MODULE_BACKEND,
        RAW_MODULE_COMPILE_OPTIONS,
        SHARED_MEMORY_CANDIDATE_TILING,
        THREADS_PER_BLOCK,
        compile_gpu_representative_kernels,
        gpu_partition_statistics,
        group_candidates_gpu_representative,
    )

    prototype = {
        "implementation_name": PROTOTYPE_IMPLEMENTATION_NAME,
        "module": "peakfinder_tests.gpu_representative_grouper",
        "public_entry_point": "group_candidates_gpu_representative",
        "raw_module_backend": RAW_MODULE_BACKEND,
        "raw_module_compile_options": list(RAW_MODULE_COMPILE_OPTIONS),
        "threads_per_block": THREADS_PER_BLOCK,
        "shared_memory_candidate_tiling": bool(
            SHARED_MEMORY_CANDIDATE_TILING
        ),
    }

    with cp.cuda.Device(args.device):
        stream = cp.cuda.get_current_stream()
        # Compile/load NVRTC before any authoritative measurement, even if the
        # first scientific point is the empty zero-corruption path.
        compile_gpu_representative_kernels()
        stream.synchronize()

        active_sources = batch_benchmark.active_repository_sources()
        bundle = batch_benchmark.load_authoritative_plan(args.config)
        if int(bundle.config.beams_per_gpu) != gaussian.DEFAULT_TOTAL_BEAMS:
            raise ValueError("authoritative config no longer declares 60 beams per GPU")
        if tuple(spec.shape for spec in bundle.specs) != gaussian.EXPECTED_SHAPES:
            raise AssertionError("authoritative plan shapes changed")
        if args.total_beams == gaussian.DEFAULT_TOTAL_BEAMS:
            if args.total_beams * gaussian.PIXELS_PER_BEAM != 58_982_400:
                raise AssertionError("60-beam pixel total changed")

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

        gpu = _gpu_information(cp, args.device)
        software = batch_benchmark.software_information(cp)
        software["active_repository_sources"] = active_sources
        software["python"] = platform.python_version()
        git = _git_information()
        payload = signature_payload(
            args, bundle, gaussian_config, gpu, software, git, prototype
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
                prototype=prototype,
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

        if args.allow_unsafe_grouping:
            print(
                "WARNING: --allow-unsafe-grouping is active. The production "
                "grouper can be quadratic and may take a very long time or "
                "exhaust resources.",
                file=sys.stderr,
                flush=True,
            )

        # Empty schemas, allocations, and the already-compiled prototype path
        # are exercised outside every authoritative interval.
        empty_decoded = decoder.decode(GpuRawCandidates.empty())
        group_candidates(
            empty_decoded, grouping_geometry, config=grouping_config
        )
        group_candidates_gpu_representative(
            empty_decoded, grouping_geometry, config=grouping_config
        )
        stream.synchronize()
        decoder_nonempty_warmed = False
        production_nonempty_warmed = False
        experimental_nonempty_warmed = False
        required_target_counts = tuple(
            gaussian.target_count_from_percentage(percentage)
            for percentage in args.percentages
        )
        maximum_target = max(required_target_counts)

        for trial in range(args.trials):
            trial_keys = {
                (
                    int(row["trial"]),
                    gaussian.canonical_percentage(
                        row["target_corruption_percent"]
                    ),
                )
                for row in rows
            }
            pending = [
                (index, percentage)
                for index, percentage in enumerate(args.percentages)
                if (
                    trial, gaussian.canonical_percentage(percentage)
                ) not in trial_keys
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
                    required_target_counts=required_target_counts,
                )
            except gaussian.MorphologyGenerationError as exc:
                for percentage_index, percentage in pending:
                    rows.append(blank_failed_row(
                        campaign_id=campaign_id,
                        args=args,
                        trial=trial,
                        percentage=percentage,
                        percentage_index=percentage_index,
                        status="failed_template_unattainable",
                        reason=str(exc),
                    ))
                    gaussian.validate_trial_prefix(
                        rows, args.trials, args.percentages, campaign_id
                    )
                    checkpoint()
                continue

            for percentage_index, percentage in enumerate(args.percentages):
                key = (trial, gaussian.canonical_percentage(percentage))
                if key in trial_keys:
                    continue
                timing_order = _timing_order(
                    trial, percentage_index, len(args.percentages)
                )
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
                        percentage_index=percentage_index,
                        status="failed_template_unattainable",
                        reason=(
                            f"rounded per-beam target {target_count} did not "
                            f"calibrate within {args.template_attempt_limit} "
                            f"deterministic attempts for beam(s) "
                            f"{preview}{suffix}"
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
                        percentage_index=percentage_index,
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
                predecode_safety = gaussian.grouping_safety_status(
                    total_raw,
                    args.candidate_safety_limit,
                    args.allow_unsafe_grouping,
                )

                needs_warmup = (
                    not decoder_nonempty_warmed
                    or (
                        predecode_safety == "run"
                        and (
                            not production_nonempty_warmed
                            or not experimental_nonempty_warmed
                        )
                    )
                )
                if total_raw and needs_warmup:
                    warm_raw = gaussian._raw_prefix(
                        raw, min(32, args.candidate_safety_limit)
                    )
                    warm_decoded = decoder.decode(warm_raw)
                    decoder_nonempty_warmed = True
                    if predecode_safety == "run":
                        if not production_nonempty_warmed:
                            group_candidates(
                                warm_decoded, grouping_geometry,
                                config=grouping_config,
                            )
                            production_nonempty_warmed = True
                        if not experimental_nonempty_warmed:
                            group_candidates_gpu_representative(
                                warm_decoded, grouping_geometry,
                                config=grouping_config,
                            )
                            experimental_nonempty_warmed = True
                    stream.synchronize()

                decoder_wall_ms, decoded = gaussian.synchronized_wall_time(
                    stream, lambda: decoder.decode(raw)
                )
                decoded_count = len(decoded)
                if decoded_count != total_raw:
                    raise AssertionError(
                        "production decoder silently changed candidate count"
                    )
                safety = gaussian.grouping_safety_status(
                    decoded_count,
                    args.candidate_safety_limit,
                    args.allow_unsafe_grouping,
                )
                if safety != predecode_safety:
                    raise AssertionError(
                        "decoder count changed the grouping safety decision"
                    )

                if safety == "skipped_candidate_safety_limit":
                    # No grouper is timed for a safety skip, so aggregate
                    # partition diagnostics can be computed immediately.
                    normalized = GpuDecodedCandidates.from_decoder_result(
                        decoded, grouping_geometry
                    )
                    partition_count, largest_partition = (
                        gpu_partition_statistics(normalized)
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
                        partition_count=partition_count,
                        largest_partition=largest_partition,
                        event_counts=None,
                        production_wall_ms=None,
                        experimental_wall_ms=None,
                        exact_parity_verified=False,
                        timing_order=timing_order,
                        status=safety,
                        chunk_duration_ms=bundle.chunk_duration_ms,
                        failure_reason=(
                            f"{decoded_count} candidates exceed safety limit "
                            f"{args.candidate_safety_limit}; both comparisons "
                            "were skipped"
                        ),
                    )
                    rows.append(row)
                else:
                    if (decoded_count > args.candidate_safety_limit
                            and args.allow_unsafe_grouping):
                        print(
                            f"WARNING: trial={trial} "
                            f"corruption={percentage:g}% is comparing "
                            f"{decoded_count} candidates above safety limit "
                            f"{args.candidate_safety_limit}",
                            file=sys.stderr,
                            flush=True,
                        )

                    sequence = (
                        ("production", "experimental")
                        if timing_order == "production_then_experimental"
                        else ("experimental", "production")
                    )
                    timed: dict[str, tuple[float, Any]] = {}
                    errors: dict[str, BaseException] = {}
                    operations = {
                        "production": lambda: group_candidates(
                            decoded, grouping_geometry, config=grouping_config
                        ),
                        "experimental": lambda: (
                            group_candidates_gpu_representative(
                                decoded, grouping_geometry,
                                config=grouping_config,
                            )
                        ),
                    }
                    for implementation in sequence:
                        try:
                            timed[implementation] = (
                                gaussian.synchronized_wall_time(
                                    stream, operations[implementation]
                                )
                            )
                        except Exception as exc:  # preserve a failed trial record
                            errors[implementation] = exc
                            try:
                                stream.synchronize()
                            except Exception as synchronization_error:
                                errors[f"{implementation}_synchronize"] = (
                                    synchronization_error
                                )

                    production_wall_ms = (
                        timed.get("production", (None, None))[0]
                    )
                    experimental_wall_ms = (
                        timed.get("experimental", (None, None))[0]
                    )
                    production_result = timed.get(
                        "production", (None, None)
                    )[1]
                    experimental_result = timed.get(
                        "experimental", (None, None)
                    )[1]
                    # Run aggregate partition diagnostics only after both
                    # authoritative calls so their full-array sorts cannot
                    # precondition either implementation. Prefer the already
                    # normalized production candidate table to avoid retaining
                    # a third duplicate candidate schema.
                    normalized = getattr(
                        production_result, "candidates", None
                    )
                    if not isinstance(normalized, GpuDecodedCandidates):
                        normalized = getattr(
                            experimental_result, "candidates", None
                        )
                    if not isinstance(normalized, GpuDecodedCandidates):
                        normalized = GpuDecodedCandidates.from_decoder_result(
                            decoded, grouping_geometry
                        )
                    try:
                        partition_count, largest_partition = (
                            gpu_partition_statistics(normalized)
                        )
                    except (AttributeError, IndexError, TypeError, ValueError):
                        # A malformed returned candidate table belongs in the
                        # exact-parity diagnostic, not as an aborted campaign.
                        # Recover partition context from the original decoded
                        # object only on this exceptional path.
                        normalized = GpuDecodedCandidates.from_decoder_result(
                            decoded, grouping_geometry
                        )
                        partition_count, largest_partition = (
                            gpu_partition_statistics(normalized)
                        )
                    reference_result = (
                        production_result
                        if production_result is not None
                        else experimental_result
                    )
                    event_counts = None
                    diagnostic: dict[str, Any] | None = None
                    if errors:
                        status = "failed_grouping"
                        diagnostic = _exception_diagnostic(
                            errors,
                            candidate_count=decoded_count,
                            partition_count=partition_count,
                            largest_partition=largest_partition,
                        )
                        reason = (
                            "grouping call failed: "
                            + "; ".join(
                                f"{name}={type(error).__name__}: {error}"
                                for name, error in errors.items()
                            )
                        )
                        parity = False
                    else:
                        mismatch = first_grouping_difference(
                            cp, production_result, experimental_result
                        )
                        if mismatch is None:
                            status = "completed"
                            reason = ""
                            parity = True
                        else:
                            status = "failed_parity"
                            diagnostic = _comparison_diagnostic(
                                mismatch=mismatch,
                                candidate_count=decoded_count,
                                partition_count=partition_count,
                                largest_partition=largest_partition,
                            )
                            reason = (
                                f"exact parity mismatch at "
                                f"{mismatch['table']}.{mismatch['field']} "
                                f"index={mismatch['first_differing_index']}"
                            )
                            parity = False

                    try:
                        if reference_result is not None:
                            events = getattr(reference_result, "events")
                            event_beam_id = getattr(events, "beam_id")
                            event_counts = gaussian.gpu_per_beam_counts(
                                cp, event_beam_id, args.total_beams
                            )
                            if int(np.sum(event_counts)) != len(events):
                                raise AssertionError(
                                    "per-beam event counts do not sum to total"
                                )
                    except Exception as exc:
                        event_counts = None
                        if status == "completed":
                            mismatch = _schema_difference(
                                table="events", field="beam_id",
                                expected=getattr(
                                    getattr(production_result, "events", None),
                                    "beam_id", None,
                                ),
                                actual=getattr(
                                    getattr(experimental_result, "events", None),
                                    "beam_id", None,
                                ),
                                mismatch_kind="event_diagnostic_exception",
                                diagnostic_error=(
                                    f"{type(exc).__name__}: {exc}"
                                ),
                            )
                            diagnostic = _comparison_diagnostic(
                                mismatch=mismatch,
                                candidate_count=decoded_count,
                                partition_count=partition_count,
                                largest_partition=largest_partition,
                            )
                            status = "failed_parity"
                            reason = (
                                "exact parity result could not be audited: "
                                f"{type(exc).__name__}: {exc}"
                            )
                            parity = False
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
                        partition_count=partition_count,
                        largest_partition=largest_partition,
                        event_counts=event_counts,
                        production_wall_ms=production_wall_ms,
                        experimental_wall_ms=experimental_wall_ms,
                        exact_parity_verified=parity,
                        timing_order=timing_order,
                        status=status,
                        chunk_duration_ms=bundle.chunk_duration_ms,
                        failure_reason=reason,
                        failure_diagnostic=diagnostic,
                    )
                    rows.append(row)
                    del production_result, experimental_result, reference_result

                gaussian.validate_trial_prefix(
                    rows, args.trials, args.percentages, campaign_id
                )
                validate_resumed_trial_rows(
                    rows, args, campaign_id, bundle.chunk_duration_ms
                )
                checkpoint()
                latest = rows[-1]
                print(
                    f"checkpoint trial={trial} corruption={percentage:g}% "
                    f"pixels={int(np.sum(pixel_counts))} candidates={total_raw} "
                    f"partitions={partition_count} largest={largest_partition} "
                    f"production={latest['production_grouper_wall_ms'] or 'skipped'} "
                    f"experimental="
                    f"{latest['experimental_grouper_wall_ms'] or 'skipped'} "
                    f"status={latest['status']}",
                    flush=True,
                )
                del raw, decoded, normalized, calibrations
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
