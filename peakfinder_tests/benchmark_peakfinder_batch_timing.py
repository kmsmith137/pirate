#!/usr/bin/env python3
"""Benchmark the clean steady-state production peakfinder over one chunk.

The authoritative measurement is a synchronized wall-clock interval around
all production ``OfflinePeakExtractor.process_chunk()`` calls required to
cover ``total_beams`` for all ten ragged trees.  Each beam batch owns its own
persistent extractors.  Input generation/upload, geometry/extractor setup,
halo priming, warm-up, aggregation, and file I/O are outside the timer.

This module deliberately keeps CuPy and PIRATE imports inside GPU-facing
functions so its campaign, input-generation, resume, and CSV helpers can be
tested on a CPU-only host.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import importlib
import json
import math
import os
import platform
import subprocess
import sys
import tempfile
import time
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

import numpy as np
import yaml


SCHEMA_NAME = "pirate-peakfinder-batch-timing"
SCHEMA_VERSION = 1
METHOD = "full_band"
NOISE_MODEL = "gaussian_white_noise_no_corruption"
SNR_DTYPE = np.dtype(np.float16)
ARGMAX_DTYPE = np.dtype(np.uint32)

REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_CONFIG = REPOSITORY_ROOT / "configs/dedispersion/chord_sb2_et.yml"
DEFAULT_RESULTS_DIR = Path(__file__).resolve().parent / "results_peakfinder_batch_timing"
DEFAULT_TOTAL_BEAMS = 60
DEFAULT_BEAM_BATCH_SIZES = (1, 2, 3, 4, 5, 6, 10, 12, 15, 20, 30, 60)
DEFAULT_DM_REACHES = (1, 2, 4, 8, 16, 32)
DEFAULT_THRESHOLD = 10.0
DEFAULT_WAIST_BINS = 1
DEFAULT_WARMUP = 1
DEFAULT_ITERATIONS = 10
DEFAULT_DEVICE = 0
DEFAULT_BASE_SEED = 20260825
SOURCE_CHUNK_ORIGIN = 1_000_000

EXPECTED_SHAPES = (
    (4096, 128),
    (1024, 64),
    (2048, 64),
    (1024, 64),
    (1024, 32),
    (2048, 32),
    (512, 32),
    (1024, 32),
    (1024, 16),
    (2048, 16),
)
EXPECTED_PIXELS_PER_BEAM = 983_040

TIMING_FIELDS = (
    "schema_version",
    "campaign_id",
    "iteration",
    "base_seed",
    "device",
    "method",
    "total_beams",
    "beam_batch_size",
    "n_batches",
    "actual_batch_sizes",
    "final_batch_size",
    "threshold",
    "dm_reach",
    "waist_bins",
    "warmup",
    "total_pixels_per_beam",
    "total_input_pixels",
    "total_candidates",
    "chunk_duration_ms",
    "total_beams_peakfinding_wall_ms",
    "peakfinding_ms_per_beam",
    "equivalent_ms_per_batch",
    "beam_chunks_per_second",
    "realtime_beam_capacity",
    "load_fraction_for_total_beams",
)

SUMMARY_METRICS = (
    "total_beams_peakfinding_wall_ms",
    "peakfinding_ms_per_beam",
    "equivalent_ms_per_batch",
    "beam_chunks_per_second",
    "realtime_beam_capacity",
    "load_fraction_for_total_beams",
)
SUMMARY_STATISTICS = ("median", "minimum", "maximum", "q25", "q75", "iqr")
SUMMARY_FIELDS = (
    "schema_version",
    "campaign_id",
    "dm_reach",
    "beam_batch_size",
    "completed_iterations",
    *tuple(
        f"{metric}_{statistic}"
        for metric in SUMMARY_METRICS
        for statistic in SUMMARY_STATISTICS
    ),
)

TREE_DIAGNOSTIC_FIELDS = (
    "schema_version",
    "campaign_id",
    "dm_reach",
    "beam_batch_size",
    "n_batches",
    "final_batch_size",
    "tree_index",
    "primary_tree_index",
    "early_trigger_level",
    "ndm",
    "ntime",
    "pixels_per_beam",
    "dm_radius",
    "active_footprint_cells",
    "left_time_radius",
    "right_time_radius",
    "time_radius",
    "halo_columns",
    "required_priming_calls",
    "regular_batch_size",
    "regular_work_shape",
    "final_work_shape",
)


@dataclass(frozen=True)
class TreePlanSpec:
    """CPU metadata needed to generate one exact ragged input map."""

    tree_index: int
    primary_tree_index: int
    early_trigger_level: int
    ndm: int
    ntime: int
    multiplets: int
    profiles: int
    token_dout: int

    @property
    def shape(self) -> tuple[int, int]:
        return (self.ndm, self.ntime)

    @property
    def pixels_per_beam(self) -> int:
        return self.ndm * self.ntime


@dataclass(frozen=True)
class BeamBatch:
    """One contiguous beam-ID partition and its full-array slice."""

    beam_ids: tuple[int, ...]
    start: int
    stop: int

    @property
    def size(self) -> int:
        return self.stop - self.start


@dataclass(frozen=True)
class CleanTreeInput:
    """Reusable host maps for one tree."""

    spec: TreePlanSpec
    snr: np.ndarray
    argmax: np.ndarray
    noise_seed: int
    token_multiplet: int
    token: int


@dataclass(frozen=True)
class PlanBundle:
    config: Any
    plan: Any
    producer_plan_yaml: str
    config_document: Mapping[str, Any]
    specs: tuple[TreePlanSpec, ...]
    chunk_duration_ms: float


@dataclass(frozen=True)
class GeometryDiagnostic:
    tree_index: int
    dm_reach: int
    dm_radius: int
    active_footprint_cells: int
    left_time_radius: int
    right_time_radius: int
    time_radius: int
    halo_columns: int


@dataclass
class BatchState:
    batch: BeamBatch
    extractors: tuple[Any, ...]
    snr_maps: tuple[Any, ...]
    argmax_maps: tuple[Any, ...]


def _exact_positive_integer(value: Any, name: str) -> int:
    if isinstance(value, (bool, np.bool_)):
        raise ValueError(f"{name} must be a positive integer")
    try:
        integer = int(value)
    except (TypeError, ValueError, OverflowError) as exc:
        raise ValueError(f"{name} must be a positive integer") from exc
    if integer != value or integer <= 0:
        raise ValueError(f"{name} must be a positive integer")
    return integer


def _exact_nonnegative_integer(value: Any, name: str) -> int:
    if isinstance(value, (bool, np.bool_)):
        raise ValueError(f"{name} must be a non-negative integer")
    try:
        integer = int(value)
    except (TypeError, ValueError, OverflowError) as exc:
        raise ValueError(f"{name} must be a non-negative integer") from exc
    if integer != value or integer < 0:
        raise ValueError(f"{name} must be a non-negative integer")
    return integer


def _unique_positive_grid(values: Sequence[Any], name: str) -> tuple[int, ...]:
    result = tuple(_exact_positive_integer(value, name) for value in values)
    if len(set(result)) != len(result):
        raise ValueError(f"{name} values must be unique")
    if not result:
        raise ValueError(f"{name} must not be empty")
    return result


def expected_tree_specs(plan: Any | None = None) -> tuple[TreePlanSpec, ...]:
    """Return expected shapes, or validate and return specs from ``plan``."""

    if plan is not None:
        return assert_expected_plan(plan)
    return tuple(
        TreePlanSpec(i, -1, -1, ndm, ntime, 1, 1, 1)
        for i, (ndm, ntime) in enumerate(EXPECTED_SHAPES)
    )


def tree_specs_from_plan(plan: Any) -> tuple[TreePlanSpec, ...]:
    specs = []
    for tree_index, tree in enumerate(plan.trees):
        nt_ds = int(tree.nt_ds)
        ntime = int(tree.nt_out)
        if ntime <= 0 or nt_ds <= 0 or nt_ds % ntime:
            raise ValueError(f"tree {tree_index} has inconsistent time dimensions")
        specs.append(TreePlanSpec(
            tree_index=tree_index,
            primary_tree_index=int(tree.primary_tree_index),
            early_trigger_level=int(tree.early_trigger_level),
            ndm=int(tree.ndm_out),
            ntime=ntime,
            multiplets=int(tree.frequency_subbands.M),
            profiles=int(tree.nprofiles),
            token_dout=nt_ds // ntime,
        ))
    return tuple(specs)


def assert_expected_plan(plan: Any) -> tuple[TreePlanSpec, ...]:
    specs = tree_specs_from_plan(plan)
    shapes = tuple(spec.shape for spec in specs)
    if int(plan.ntrees) != 10 or len(specs) != 10:
        raise ValueError(f"expected exactly 10 output trees, found {len(specs)}")
    if shapes != EXPECTED_SHAPES:
        raise ValueError(f"unexpected plan tree shapes: {shapes!r}")
    pixels = sum(spec.pixels_per_beam for spec in specs)
    if pixels != EXPECTED_PIXELS_PER_BEAM:
        raise ValueError(
            f"expected {EXPECTED_PIXELS_PER_BEAM} pixels per beam, found {pixels}"
        )
    for spec in specs:
        if not (1 <= spec.multiplets <= 65_536):
            raise ValueError(f"tree {spec.tree_index} has invalid multiplet count")
        if not (1 <= spec.profiles <= 256):
            raise ValueError(f"tree {spec.tree_index} has invalid profile count")
        if not (1 <= spec.token_dout <= 256):
            raise ValueError(f"tree {spec.tree_index} has invalid token time range")
    return specs


def load_authoritative_plan(config_path: Path | str) -> PlanBundle:
    """Build the registered producer plan and reconstruct its offline consumer."""

    from pirate_frb import DedispersionConfig, DedispersionPlan

    path = Path(config_path).resolve()
    with path.open("r", encoding="utf-8") as stream:
        document = yaml.safe_load(stream)
    if not isinstance(document, dict):
        raise ValueError("dedispersion config must contain a YAML mapping")
    config = DedispersionConfig.from_yaml(str(path))
    # Match the active offline consumer: construct the authoritative producer
    # plan, then reconstruct its serialized consumer plan so producer-selected
    # Dcore/token metadata are retained exactly.
    producer_plan = DedispersionPlan(config, gpu_runnable=True)
    producer_plan_yaml = producer_plan.to_yaml_string()
    plan = DedispersionPlan.make_incomplete_plan_from_yaml(
        config.to_yaml_string(), producer_plan_yaml
    )
    specs = assert_expected_plan(plan)
    if str(document.get("dtype")) != "float16":
        raise ValueError("authoritative configuration must use float16 S/N maps")
    chunk_duration_ms = float(config.time_sample_ms) * int(plan.nt_in)
    if not math.isfinite(chunk_duration_ms) or chunk_duration_ms <= 0.0:
        raise ValueError(f"invalid chunk duration {chunk_duration_ms} ms")
    configured_samples = int(document.get("time_samples_per_chunk", -1))
    if configured_samples != int(plan.nt_in):
        raise ValueError(
            "configuration time_samples_per_chunk disagrees with plan.nt_in"
        )
    return PlanBundle(
        config, plan, producer_plan_yaml, document, specs, chunk_duration_ms
    )


def active_repository_sources() -> dict[str, str]:
    """Require production imports to resolve to this repository checkout."""

    package = importlib.import_module("pirate_frb")
    peakfinders = importlib.import_module("pirate_frb.Peakfinders")
    expected_peakfinders = (REPOSITORY_ROOT / "pirate_frb/Peakfinders.py").resolve()
    actual_peakfinders = Path(peakfinders.__file__).resolve()
    if actual_peakfinders != expected_peakfinders:
        raise RuntimeError(
            "pirate_frb.Peakfinders resolved outside the active repository: "
            f"{actual_peakfinders}; expected {expected_peakfinders}. Run the "
            "benchmark as `python -m peakfinder_tests.benchmark_peakfinder_batch_timing` "
            "from the repository root."
        )
    extension = getattr(package, "pirate_pybind11", None)
    return {
        "pirate_frb_package": str(Path(package.__file__).resolve()),
        "peakfinders_module": str(actual_peakfinders),
        "pirate_pybind11_extension": str(Path(extension.__file__).resolve()),
    }


def partition_beams(total_beams: Any, beam_batch_size: Any) -> tuple[BeamBatch, ...]:
    """Partition beam IDs ``0..total_beams-1`` exactly once and contiguously."""

    total = _exact_positive_integer(total_beams, "total_beams")
    requested = _exact_positive_integer(beam_batch_size, "beam_batch_size")
    batches = tuple(
        BeamBatch(tuple(range(start, min(start + requested, total))),
                  start, min(start + requested, total))
        for start in range(0, total, requested)
    )
    flattened = tuple(beam for batch in batches for beam in batch.beam_ids)
    if flattened != tuple(range(total)):
        raise AssertionError("internal beam partitioning error")
    return batches


def halo_fill_calls(time_radius: Any, ntime: Any) -> int:
    """Minimum calls whose retained columns can fill the exact ``2*h`` halo."""

    radius = _exact_nonnegative_integer(time_radius, "time_radius")
    width = _exact_positive_integer(ntime, "ntime")
    return math.ceil((2 * radius) / width)


def required_priming_calls(time_radius: Any, ntime: Any) -> int:
    """Prior calls that fill the halo and put the next call past the left edge."""

    radius = _exact_nonnegative_integer(time_radius, "time_radius")
    width = _exact_positive_integer(ntime, "ntime")
    # Strictly more than 2*h columns makes the next work array's absolute
    # start positive. This proves that the timed call is not treated as the
    # physical acquisition-left boundary, including exact-divisibility cases.
    return max(1, (2 * radius) // width + 1)


def tree_seed(base_seed: Any, tree_index: Any, purpose: str = "snr") -> int:
    """Derive a stable independent uint64 RNG seed for a tree and purpose."""

    base = _exact_nonnegative_integer(base_seed, "base_seed")
    tree = _exact_nonnegative_integer(tree_index, "tree_index")
    payload = json.dumps(
        [SCHEMA_NAME, base, tree, str(purpose)], separators=(",", ":")
    ).encode("utf-8")
    return int.from_bytes(hashlib.sha256(payload).digest()[:8], "little")


def _valid_tree_token(spec: TreePlanSpec, base_seed: int) -> tuple[int, int]:
    multiplet = tree_seed(base_seed, spec.tree_index, "token") % spec.multiplets
    # Current ABI: profile bits 8..15 and fine time bits 0..7 are both zero;
    # the tree-specific multiplet occupies bits 16..31.
    token = int(np.uint32(multiplet) << np.uint32(16))
    return int(multiplet), token


def validate_argmax_tokens(
        specs: Sequence[TreePlanSpec], argmax_maps: Sequence[np.ndarray]) -> None:
    if len(specs) != len(argmax_maps):
        raise ValueError("one argmax map is required for every tree")
    for spec, tokens in zip(specs, argmax_maps):
        values = np.asarray(tokens)
        if values.dtype != ARGMAX_DTYPE:
            raise ValueError(f"tree {spec.tree_index} argmax dtype is not uint32")
        if values.ndim != 3 or values.shape[1:] != spec.shape:
            raise ValueError(f"tree {spec.tree_index} argmax shape is invalid")
        if np.any(values == np.uint32(0xFFFFFFFF)):
            raise ValueError(f"tree {spec.tree_index} uses the invalid sentinel")
        multiplet = values >> np.uint32(16)
        profile = (values >> np.uint32(8)) & np.uint32(0xFF)
        fine_time = values & np.uint32(0xFF)
        if (np.any(multiplet >= spec.multiplets)
                or np.any(profile >= spec.profiles)
                or np.any(fine_time >= spec.token_dout)):
            raise ValueError(f"tree {spec.tree_index} contains invalid tokens")


def validate_tokens_with_plan(
        plan: Any, inputs: Sequence[CleanTreeInput]) -> tuple[tuple[int, ...], ...]:
    """Decode every constant token once with the authoritative current plan."""

    decoded = []
    for item in inputs:
        values = tuple(int(value) for value in plan.decode_argmax(
            item.token, item.spec.tree_index, 0, 0
        ))
        if len(values) != 5 or values[-1] != 0:
            raise ValueError(
                f"tree {item.spec.tree_index} token did not decode to profile 0"
            )
        decoded.append(values)
    return tuple(decoded)


def generate_clean_inputs(
        specs: Sequence[TreePlanSpec], total_beams: Any, base_seed: Any,
        threshold: float = DEFAULT_THRESHOLD) -> tuple[CleanTreeInput, ...]:
    """Generate deterministic Gaussian maps and valid constant token maps."""

    beams = _exact_positive_integer(total_beams, "total_beams")
    base = _exact_nonnegative_integer(base_seed, "base_seed")
    threshold = float(threshold)
    if not math.isfinite(threshold):
        raise ValueError("threshold must be finite")
    inputs = []
    for spec in specs:
        seed = tree_seed(base, spec.tree_index, "snr")
        rng = np.random.default_rng(seed)
        shape = (beams, spec.ndm, spec.ntime)
        snr = rng.standard_normal(shape, dtype=np.float32).astype(SNR_DTYPE)
        if snr.dtype != SNR_DTYPE or not np.all(np.isfinite(snr)):
            raise AssertionError(f"tree {spec.tree_index} generated invalid S/N values")
        if np.any(snr >= threshold):
            maximum = float(np.max(snr))
            raise ValueError(
                f"threshold {threshold} is not above tree {spec.tree_index} "
                f"noise maximum {maximum}"
            )
        multiplet, token = _valid_tree_token(spec, base)
        argmax = np.full(shape, np.uint32(token), dtype=ARGMAX_DTYPE)
        inputs.append(CleanTreeInput(spec, snr, argmax, seed, multiplet, token))
    validate_argmax_tokens(specs, tuple(item.argmax for item in inputs))
    return tuple(inputs)


def derive_timing_metrics(
        wall_ms: float, total_beams: Any, n_batches: Any,
        chunk_duration_ms: float) -> dict[str, float]:
    """Calculate the request's exact throughput and real-time definitions."""

    wall = float(wall_ms)
    total = _exact_positive_integer(total_beams, "total_beams")
    batches = _exact_positive_integer(n_batches, "n_batches")
    duration = float(chunk_duration_ms)
    if not math.isfinite(wall) or wall <= 0.0:
        raise ValueError("wall_ms must be finite and strictly positive")
    if not math.isfinite(duration) or duration <= 0.0:
        raise ValueError("chunk_duration_ms must be finite and strictly positive")
    return {
        "peakfinding_ms_per_beam": wall / total,
        "equivalent_ms_per_batch": wall / batches,
        "beam_chunks_per_second": 1000.0 * total / wall,
        "realtime_beam_capacity": duration * total / wall,
        "load_fraction_for_total_beams": wall / duration,
    }


def _validate_rows(rows: Iterable[Mapping[str, Any]], fields: Sequence[str]) -> list[dict[str, Any]]:
    expected = set(fields)
    normalized = []
    for row_index, row in enumerate(rows):
        if set(row) != expected:
            missing = sorted(expected - set(row))
            extra = sorted(set(row) - expected)
            raise ValueError(
                f"row {row_index} has an incompatible schema; "
                f"missing={missing}, extra={extra}"
            )
        normalized.append(dict(row))
    return normalized


def _atomic_write_csv(
        path: Path, fields: Sequence[str], rows: Iterable[Mapping[str, Any]]) -> None:
    rows = _validate_rows(rows, fields)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary_name = None
    try:
        with tempfile.NamedTemporaryFile(
                mode="w", encoding="utf-8", newline="", prefix=f".{path.name}.",
                suffix=".tmp", dir=path.parent, delete=False) as stream:
            temporary_name = stream.name
            writer = csv.DictWriter(stream, fieldnames=fields, lineterminator="\n")
            writer.writeheader()
            writer.writerows(rows)
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary_name, path)
        temporary_name = None
    finally:
        if temporary_name is not None:
            try:
                os.unlink(temporary_name)
            except FileNotFoundError:
                pass


def _atomic_write_yaml(path: Path, document: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary_name = None
    try:
        with tempfile.NamedTemporaryFile(
                mode="w", encoding="utf-8", prefix=f".{path.name}.",
                suffix=".tmp", dir=path.parent, delete=False) as stream:
            temporary_name = stream.name
            yaml.safe_dump(dict(document), stream, sort_keys=False)
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary_name, path)
        temporary_name = None
    finally:
        if temporary_name is not None:
            try:
                os.unlink(temporary_name)
            except FileNotFoundError:
                pass


def _read_csv_exact(path: Path, fields: Sequence[str]) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8", newline="") as stream:
        reader = csv.DictReader(stream)
        if tuple(reader.fieldnames or ()) != tuple(fields):
            raise ValueError(
                f"{path} has incompatible columns {reader.fieldnames!r}; "
                f"expected {tuple(fields)!r}"
            )
        rows = list(reader)
    _validate_rows(rows, fields)
    return rows


def _quantile(values: np.ndarray, fraction: float) -> float:
    try:
        return float(np.quantile(values, fraction, method="linear"))
    except TypeError:  # NumPy < 1.22
        return float(np.quantile(values, fraction, interpolation="linear"))


def summarize_timings(rows: Sequence[Mapping[str, Any]]) -> list[dict[str, Any]]:
    """Return one wide median/min/max/IQR row per completed configuration."""

    groups: dict[tuple[int, int], list[Mapping[str, Any]]] = {}
    for row in rows:
        key = (int(row["dm_reach"]), int(row["beam_batch_size"]))
        groups.setdefault(key, []).append(row)
    summary = []
    for key in sorted(groups):
        group = groups[key]
        campaign_ids = {str(row["campaign_id"]) for row in group}
        if len(campaign_ids) != 1:
            raise ValueError(f"configuration {key} mixes campaign IDs")
        output: dict[str, Any] = {
            "schema_version": SCHEMA_VERSION,
            "campaign_id": campaign_ids.pop(),
            "dm_reach": key[0],
            "beam_batch_size": key[1],
            "completed_iterations": len(group),
        }
        for metric in SUMMARY_METRICS:
            values = np.asarray([float(row[metric]) for row in group], dtype=np.float64)
            if values.size == 0 or not np.all(np.isfinite(values)) or np.any(values < 0):
                raise ValueError(f"configuration {key} has invalid {metric}")
            q25 = _quantile(values, 0.25)
            q75 = _quantile(values, 0.75)
            output.update({
                f"{metric}_median": float(np.median(values)),
                f"{metric}_minimum": float(np.min(values)),
                f"{metric}_maximum": float(np.max(values)),
                f"{metric}_q25": q25,
                f"{metric}_q75": q75,
                f"{metric}_iqr": q75 - q25,
            })
        summary.append(output)
    return _validate_rows(summary, SUMMARY_FIELDS)


def result_paths(results_dir: Path | str) -> dict[str, Path]:
    directory = Path(results_dir).resolve()
    return {
        "directory": directory,
        "timings": directory / "timings.csv",
        "summary": directory / "summary.csv",
        "tree_diagnostics": directory / "tree_diagnostics.csv",
        "metadata": directory / "metadata.yaml",
    }


def write_result_bundle(
        paths: Mapping[str, Path], timing_rows: Sequence[Mapping[str, Any]],
        tree_rows: Sequence[Mapping[str, Any]], metadata: Mapping[str, Any]) -> None:
    """Atomically checkpoint all four result files, raw timings first."""

    _atomic_write_csv(paths["timings"], TIMING_FIELDS, timing_rows)
    _atomic_write_csv(paths["summary"], SUMMARY_FIELDS, summarize_timings(timing_rows))
    _atomic_write_csv(
        paths["tree_diagnostics"], TREE_DIAGNOSTIC_FIELDS, tree_rows
    )
    _atomic_write_yaml(paths["metadata"], metadata)


def file_identity(path: Path | str) -> dict[str, Any]:
    resolved = Path(path).resolve()
    digest = hashlib.sha256()
    with resolved.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return {
        "path": str(resolved),
        "sha256": digest.hexdigest(),
        "size_bytes": resolved.stat().st_size,
    }


def git_information(repository: Path | str = REPOSITORY_ROOT) -> dict[str, Any]:
    root = Path(repository).resolve()

    def run(*arguments: str) -> str:
        return subprocess.check_output(
            ("git", *arguments), cwd=root, text=True, stderr=subprocess.DEVNULL
        ).strip()

    try:
        commit = run("rev-parse", "HEAD")
        status = run("status", "--porcelain=v1", "--untracked-files=all")
    except (OSError, subprocess.CalledProcessError):
        return {"commit": None, "dirty": None, "status_porcelain": None}
    return {
        "commit": commit,
        "dirty": bool(status),
        "status_porcelain": status.splitlines(),
    }


def campaign_signature(payload: Mapping[str, Any]) -> str:
    encoded = json.dumps(
        payload, sort_keys=True, separators=(",", ":"), allow_nan=False
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def expected_timing_keys(
        dm_reaches: Sequence[int], batch_sizes: Sequence[int], iterations: int
        ) -> tuple[tuple[int, int, int], ...]:
    return tuple(
        (int(reach), int(batch_size), iteration)
        for reach in dm_reaches
        for batch_size in batch_sizes
        for iteration in range(iterations)
    )


def validate_timing_prefix(
        rows: Sequence[Mapping[str, Any]], dm_reaches: Sequence[int],
        batch_sizes: Sequence[int], iterations: int,
        campaign_id: str | None = None) -> tuple[tuple[int, int, int], ...]:
    """Require checkpoint rows to be an exact, duplicate-free campaign prefix."""

    expected = expected_timing_keys(dm_reaches, batch_sizes, iterations)
    if len(rows) > len(expected):
        raise ValueError("timings contain more rows than the requested campaign")
    actual = tuple(
        (int(row["dm_reach"]), int(row["beam_batch_size"]), int(row["iteration"]))
        for row in rows
    )
    if actual != expected[:len(actual)]:
        raise ValueError("timings are not an exact ordered campaign prefix")
    if campaign_id is not None and any(
            str(row["campaign_id"]) != campaign_id for row in rows):
        raise ValueError("timings contain a different campaign ID")
    return actual


def validate_resumed_timing_rows(
        rows: Sequence[Mapping[str, Any]], *, campaign_id: str,
        dm_reaches: Sequence[int], batch_sizes: Sequence[int], iterations: int,
        total_beams: int, base_seed: int, device: int, method: str,
        threshold: float, waist_bins: int, warmup: int,
        chunk_duration_ms: float) -> None:
    """Validate every persisted scientific value before replaying a checkpoint."""

    validate_timing_prefix(
        rows, dm_reaches, batch_sizes, iterations, campaign_id
    )

    def exact_float(actual: Any, expected: float, label: str) -> None:
        value = float(actual)
        if (not math.isfinite(value)
                or not math.isclose(value, expected, rel_tol=1.0e-12, abs_tol=1.0e-12)):
            raise ValueError(f"resumed timing row has incompatible {label}")

    for row_index, row in enumerate(rows):
        batch_size = int(row["beam_batch_size"])
        partitions = partition_beams(total_beams, batch_size)
        expected_sizes = [batch.size for batch in partitions]
        try:
            actual_sizes = json.loads(str(row["actual_batch_sizes"]))
        except (TypeError, ValueError, json.JSONDecodeError) as exc:
            raise ValueError(
                f"resumed timing row {row_index} has invalid batch-size JSON"
            ) from exc
        expected_exact = {
            "schema_version": SCHEMA_VERSION,
            "base_seed": base_seed,
            "device": device,
            "total_beams": total_beams,
            "n_batches": len(partitions),
            "final_batch_size": partitions[-1].size,
            "waist_bins": waist_bins,
            "warmup": warmup,
            "total_pixels_per_beam": EXPECTED_PIXELS_PER_BEAM,
            "total_input_pixels": EXPECTED_PIXELS_PER_BEAM * total_beams,
            "total_candidates": 0,
        }
        for field, expected in expected_exact.items():
            if int(row[field]) != expected:
                raise ValueError(
                    f"resumed timing row {row_index} has incompatible {field}"
                )
        if str(row["method"]) != method:
            raise ValueError(f"resumed timing row {row_index} has incompatible method")
        if actual_sizes != expected_sizes:
            raise ValueError(
                f"resumed timing row {row_index} has incompatible beam partition"
            )
        exact_float(row["threshold"], float(threshold), "threshold")
        exact_float(
            row["chunk_duration_ms"], float(chunk_duration_ms),
            "chunk_duration_ms"
        )
        wall_ms = float(row["total_beams_peakfinding_wall_ms"])
        if not math.isfinite(wall_ms) or wall_ms <= 0.0:
            raise ValueError(
                f"resumed timing row {row_index} has invalid authoritative wall time"
            )
        derived = derive_timing_metrics(
            wall_ms, total_beams, len(partitions), chunk_duration_ms
        )
        for field, expected in derived.items():
            exact_float(row[field], expected, field)


def _tree_rows_equal(
        existing: Sequence[Mapping[str, Any]], expected: Sequence[Mapping[str, Any]]) -> bool:
    if len(existing) != len(expected):
        return False
    return all(
        all(str(old[field]) == str(new[field]) for field in TREE_DIAGNOSTIC_FIELDS)
        for old, new in zip(existing, expected)
    )


def prepare_result_state(
        paths: Mapping[str, Path], *, resume: bool, overwrite: bool,
        signature: str, campaign_id: str,
        tree_rows: Sequence[Mapping[str, Any]], dm_reaches: Sequence[int],
        batch_sizes: Sequence[int], iterations: int,
        ) -> tuple[list[dict[str, Any]], Mapping[str, Any] | None]:
    """Validate a resumable bundle or authorize a new/overwritten campaign."""

    result_files = tuple(paths[name] for name in (
        "timings", "summary", "tree_diagnostics", "metadata"
    ))
    existing_files = tuple(path for path in result_files if path.exists())
    if resume and overwrite:
        raise ValueError("--resume and --overwrite are mutually exclusive")
    if not resume:
        if existing_files and not overwrite:
            joined = ", ".join(str(path) for path in existing_files)
            raise FileExistsError(
                f"result files already exist ({joined}); use --resume or --overwrite"
            )
        return [], None
    if len(existing_files) != len(result_files):
        missing = [str(path) for path in result_files if not path.exists()]
        raise FileNotFoundError(
            "resume requires a complete checkpoint bundle; missing " + ", ".join(missing)
        )
    with paths["metadata"].open("r", encoding="utf-8") as stream:
        metadata = yaml.safe_load(stream)
    if not isinstance(metadata, dict):
        raise ValueError("metadata.yaml must contain a mapping")
    if metadata.get("schema_name") != SCHEMA_NAME or int(
            metadata.get("schema_version", -1)) != SCHEMA_VERSION:
        raise ValueError("metadata schema is incompatible")
    if metadata.get("campaign_signature") != signature:
        raise ValueError("existing results belong to an incompatible campaign")
    if metadata.get("campaign_id") != campaign_id:
        raise ValueError("existing metadata campaign ID is inconsistent")
    timings = _read_csv_exact(paths["timings"], TIMING_FIELDS)
    validate_timing_prefix(
        timings, dm_reaches, batch_sizes, iterations, campaign_id
    )
    existing_tree_rows = _read_csv_exact(
        paths["tree_diagnostics"], TREE_DIAGNOSTIC_FIELDS
    )
    if not _tree_rows_equal(existing_tree_rows, tree_rows):
        raise ValueError("existing tree diagnostics are incompatible")
    # Validate summary syntax; it is regenerated at the next checkpoint.
    _read_csv_exact(paths["summary"], SUMMARY_FIELDS)
    return [dict(row) for row in timings], metadata


def gpu_information(cp: Any, device: int) -> dict[str, Any]:
    properties = cp.cuda.runtime.getDeviceProperties(device)
    name = properties.get("name", properties.get(b"name", "unknown"))
    if isinstance(name, bytes):
        name = name.decode("utf-8", errors="replace")
    total_memory = int(properties.get(
        "totalGlobalMem", properties.get(b"totalGlobalMem", 0)
    ))
    return {
        "device": int(device),
        "model": str(name),
        "total_memory_bytes": total_memory,
        "total_memory_gib": total_memory / (1024.0 ** 3),
        "cuda_runtime_version": int(cp.cuda.runtime.runtimeGetVersion()),
        "cuda_driver_version": int(cp.cuda.runtime.driverGetVersion()),
    }


def software_information(cp: Any) -> dict[str, Any]:
    try:
        from importlib.metadata import PackageNotFoundError, version
        try:
            pirate_version = version("pirate_frb")
        except PackageNotFoundError:
            pirate_version = None
    except ImportError:  # pragma: no cover - Python >=3.8 in supported environments
        pirate_version = None
    return {
        "python": platform.python_version(),
        "python_executable": sys.executable,
        "numpy": np.__version__,
        "cupy": cp.__version__,
        "pirate_frb_distribution": pirate_version,
        "platform": platform.platform(),
    }


def build_geometries(
        cp: Any, plan: Any, specs: Sequence[TreePlanSpec],
        dm_reaches: Sequence[int], waist_bins: int,
        ) -> tuple[dict[int, tuple[Any, ...]], dict[tuple[int, int], GeometryDiagnostic]]:
    """Construct each current production tree/reach geometry outside timing."""

    from pirate_frb.Peakfinders import PeakFinderGeometry

    geometries: dict[int, tuple[Any, ...]] = {}
    diagnostics: dict[tuple[int, int], GeometryDiagnostic] = {}
    for reach in dm_reaches:
        by_tree = []
        for spec in specs:
            geometry = PeakFinderGeometry.from_plan(
                plan, spec.tree_index, dm_reach=reach,
                waist_bins=waist_bins,
            )
            if (int(geometry.tree) != spec.tree_index
                    or (int(geometry.ndm), int(geometry.ntime)) != spec.shape):
                raise AssertionError("production geometry does not match the plan")
            footprint = cp.asnumpy(geometry.full_band_bowtie)
            if footprint.dtype != np.bool_ or footprint.ndim != 2:
                raise AssertionError("production full-band footprint is invalid")
            active_columns = np.flatnonzero(np.any(footprint, axis=0))
            if active_columns.size == 0:
                raise AssertionError("production full-band footprint is empty")
            centre = footprint.shape[1] // 2
            left_radius = centre - int(active_columns[0])
            right_radius = int(active_columns[-1]) - centre
            diagnostic = GeometryDiagnostic(
                tree_index=spec.tree_index,
                dm_reach=int(reach),
                dm_radius=int(geometry.dm_radius),
                active_footprint_cells=int(np.count_nonzero(footprint)),
                left_time_radius=left_radius,
                right_time_radius=right_radius,
                time_radius=int(geometry.time_radius),
                halo_columns=2 * int(geometry.time_radius),
            )
            diagnostics[(int(reach), spec.tree_index)] = diagnostic
            by_tree.append(geometry)
        geometries[int(reach)] = tuple(by_tree)
    cp.cuda.get_current_stream().synchronize()
    return geometries, diagnostics


def build_tree_diagnostic_rows(
        campaign_id: str, specs: Sequence[TreePlanSpec],
        diagnostics: Mapping[tuple[int, int], GeometryDiagnostic],
        dm_reaches: Sequence[int], batch_sizes: Sequence[int], total_beams: int,
        ) -> list[dict[str, Any]]:
    rows = []
    for reach in dm_reaches:
        for requested_batch_size in batch_sizes:
            partitions = partition_beams(total_beams, requested_batch_size)
            regular_size = partitions[0].size
            final_size = partitions[-1].size
            for spec in specs:
                diagnostic = diagnostics[(int(reach), spec.tree_index)]
                work_time = spec.ntime + diagnostic.halo_columns
                rows.append({
                    "schema_version": SCHEMA_VERSION,
                    "campaign_id": campaign_id,
                    "dm_reach": int(reach),
                    "beam_batch_size": int(requested_batch_size),
                    "n_batches": len(partitions),
                    "final_batch_size": final_size,
                    "tree_index": spec.tree_index,
                    "primary_tree_index": spec.primary_tree_index,
                    "early_trigger_level": spec.early_trigger_level,
                    "ndm": spec.ndm,
                    "ntime": spec.ntime,
                    "pixels_per_beam": spec.pixels_per_beam,
                    "dm_radius": diagnostic.dm_radius,
                    "active_footprint_cells": diagnostic.active_footprint_cells,
                    "left_time_radius": diagnostic.left_time_radius,
                    "right_time_radius": diagnostic.right_time_radius,
                    "time_radius": diagnostic.time_radius,
                    "halo_columns": diagnostic.halo_columns,
                    "required_priming_calls": required_priming_calls(
                        diagnostic.time_radius, spec.ntime
                    ),
                    "regular_batch_size": regular_size,
                    "regular_work_shape": json.dumps(
                        [regular_size, spec.ndm, work_time], separators=(",", ":")
                    ),
                    "final_work_shape": json.dumps(
                        [final_size, spec.ndm, work_time], separators=(",", ":")
                    ),
                })
    return _validate_rows(rows, TREE_DIAGNOSTIC_FIELDS)


def upload_inputs(cp: Any, inputs: Sequence[CleanTreeInput]) -> tuple[tuple[Any, ...], tuple[Any, ...]]:
    snr_maps = tuple(cp.asarray(item.snr) for item in inputs)
    argmax_maps = tuple(cp.asarray(item.argmax) for item in inputs)
    cp.cuda.get_current_stream().synchronize()
    return snr_maps, argmax_maps


def _assert_zero_candidates(candidate_count: int, context: str) -> None:
    if int(candidate_count) != 0:
        raise AssertionError(f"{context} produced {candidate_count} candidates, expected zero")


def assert_filled_streaming_halo(
        extractor: Any, *, expected_last_chunk: int | None = None) -> None:
    """Prove the current private streaming tail is full and no longer at startup.

    ``OfflinePeakExtractor`` currently has no public halo-state accessor, so
    this focused benchmark inspects the production object's actual persistent
    arrays and counters.  Metadata records that current-API deviation.
    """

    geometry = extractor.geometry
    radius = int(geometry.time_radius)
    expected_halo = 2 * radius
    tail_snr = extractor._tail_snr
    if tail_snr is None:
        raise AssertionError("streaming extractor was not initialized")
    expected_snr_shape = (
        len(extractor.beam_ids), int(geometry.ndm), expected_halo
    )
    if tuple(tail_snr.shape) != expected_snr_shape:
        raise AssertionError(
            f"S/N halo shape {tail_snr.shape} != {expected_snr_shape}"
        )
    if tuple(extractor._tail_argmax.shape) != expected_snr_shape:
        raise AssertionError("argmax halo shape is inconsistent")
    if tuple(extractor._tail_valid.shape) != (int(geometry.ndm), expected_halo):
        raise AssertionError("startup-validity halo shape is inconsistent")
    if tuple(extractor._tail_chunk.shape) != (expected_halo,):
        raise AssertionError("chunk-owner halo shape is inconsistent")
    if tuple(extractor._tail_itime.shape) != (expected_halo,):
        raise AssertionError("time-owner halo shape is inconsistent")
    if int(extractor._total_columns) <= expected_halo:
        raise AssertionError("stream has not advanced beyond its physical left edge")
    if int(extractor._next_emit) != int(extractor._total_columns) - radius:
        raise AssertionError("stream emission frontier is inconsistent")
    if expected_last_chunk is not None and int(extractor._last_chunk) != int(expected_last_chunk):
        raise AssertionError("stream ended on an unexpected source chunk")
    if bool(extractor._flushed):
        raise AssertionError("authoritative benchmark must never flush an extractor")


def construct_and_prime_batch_states(
        geometries: Sequence[Any], partitions: Sequence[BeamBatch],
        full_snr_maps: Sequence[Any], full_argmax_maps: Sequence[Any],
        threshold: float, stream: Any,
        ) -> tuple[BatchState, ...]:
    """Build independent per-batch extractors and fill every actual halo."""

    from pirate_frb.Peakfinders import OfflinePeakExtractor

    states = []
    for batch in partitions:
        snr_maps = tuple(source[batch.start:batch.stop] for source in full_snr_maps)
        argmax_maps = tuple(source[batch.start:batch.stop] for source in full_argmax_maps)
        extractors = tuple(
            OfflinePeakExtractor(
                geometry, threshold=threshold, beam_ids=batch.beam_ids,
                assume_steady_state=True
            )
            for geometry in geometries
        )
        state = BatchState(batch, extractors, snr_maps, argmax_maps)
        for tree_index, extractor in enumerate(extractors):
            calls = required_priming_calls(
                extractor.geometry.time_radius, extractor.geometry.ntime
            )
            for source_chunk in range(SOURCE_CHUNK_ORIGIN - calls, SOURCE_CHUNK_ORIGIN):
                part = extractor.process_chunk(
                    snr_maps[tree_index], argmax_maps[tree_index], source_chunk
                )
                _assert_zero_candidates(
                    len(part), f"priming batch {batch.beam_ids} tree {tree_index}"
                )
            assert_filled_streaming_halo(
                extractor, expected_last_chunk=SOURCE_CHUNK_ORIGIN - 1
            )
        states.append(state)
    stream.synchronize()
    return tuple(states)


def process_complete_workload(
        states: Sequence[BatchState], source_chunk_index: int,
        concatenate_raw_candidates: Any) -> int:
    """Run all batches/trees and production concatenation once per batch."""

    total_candidates = 0
    for state in states:
        parts = tuple(
            extractor.process_chunk(
                state.snr_maps[tree_index], state.argmax_maps[tree_index],
                source_chunk_index
            )
            for tree_index, extractor in enumerate(state.extractors)
        )
        combined = concatenate_raw_candidates(parts)
        count = len(combined)
        _assert_zero_candidates(
            count, f"batch {state.batch.beam_ids} chunk {source_chunk_index}"
        )
        total_candidates += count
    _assert_zero_candidates(total_candidates, "complete workload")
    return total_candidates


def synchronized_wall_time(stream: Any, function: Any) -> tuple[float, Any]:
    stream.synchronize()
    start = time.perf_counter()
    result = function()
    stream.synchronize()
    elapsed_ms = (time.perf_counter() - start) * 1000.0
    if not math.isfinite(elapsed_ms) or elapsed_ms < 0.0:
        raise AssertionError("synchronized wall-clock timing is invalid")
    return elapsed_ms, result


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _metadata_document(
        *, campaign_id: str, signature: str,
        signature_payload: Mapping[str, Any], bundle: PlanBundle,
        config_path: Path, plan_yaml: str, inputs_summary: Sequence[Mapping[str, Any]],
        diagnostics: Mapping[tuple[int, int], GeometryDiagnostic],
        dm_reaches: Sequence[int], batch_sizes: Sequence[int],
        total_beams: int, base_seed: int, threshold: float, waist_bins: int,
        warmup: int, iterations: int, device: int,
        gpu: Mapping[str, Any], software: Mapping[str, Any],
        git: Mapping[str, Any], warning: str | None,
        created_utc: str | None = None) -> dict[str, Any]:
    partitions = {
        str(batch_size): {
            "n_batches": len(partition_beams(total_beams, batch_size)),
            "actual_batch_sizes": [
                batch.size for batch in partition_beams(total_beams, batch_size)
            ],
            "beam_ids": [
                list(batch.beam_ids)
                for batch in partition_beams(total_beams, batch_size)
            ],
        }
        for batch_size in batch_sizes
    }
    geometry_records = [
        asdict(diagnostics[(reach, spec.tree_index)])
        for reach in dm_reaches
        for spec in bundle.specs
    ]
    plan_records = [asdict(spec) | {"pixels_per_beam": spec.pixels_per_beam}
                    for spec in bundle.specs]
    expected_rows = len(dm_reaches) * len(batch_sizes) * iterations
    return {
        "schema_name": SCHEMA_NAME,
        "schema_version": SCHEMA_VERSION,
        "campaign_id": campaign_id,
        "campaign_signature": signature,
        "campaign_signature_payload": dict(signature_payload),
        "created_utc": created_utc or _utc_now(),
        "last_updated_utc": _utc_now(),
        "campaign_complete": False,
        "warning": warning,
        "authoritative_config": {
            **file_identity(config_path),
            "document": dict(bundle.config_document),
        },
        "authoritative_plan": {
            "construction": (
                "DedispersionConfig.from_yaml; DedispersionPlan(config, "
                "gpu_runnable=True); serialized reconstruction with "
                "DedispersionPlan.make_incomplete_plan_from_yaml"
            ),
            "plan_yaml": plan_yaml,
            "plan_yaml_sha256": hashlib.sha256(plan_yaml.encode("utf-8")).hexdigest(),
            "ntrees": len(bundle.specs),
            "tree_shapes_ndm_ntime": [list(spec.shape) for spec in bundle.specs],
            "pixels_per_beam": EXPECTED_PIXELS_PER_BEAM,
            "chunk_duration_ms": bundle.chunk_duration_ms,
            "trees": plan_records,
        },
        "inputs": {
            "noise_model": NOISE_MODEL,
            "corruption": "none",
            "distribution": "independent standard normal N(0,1)",
            "generation_dtype": "float32",
            "production_snr_dtype": SNR_DTYPE.name,
            "base_seed": base_seed,
            "tree_seed_policy": (
                "uint64 little-endian prefix of SHA256([schema, base_seed, "
                "tree_index, purpose])"
            ),
            "threshold_validation": "all finite and every float16 value < threshold",
            "per_tree_realized": list(inputs_summary),
            "token_policy": (
                "constant valid uint32 per tree; deterministic tree-specific "
                "multiplet in bits 16..31, profile 0, fine time 0; no sentinel; "
                "each token is decoded once by the authoritative plan before timing"
            ),
            "reuse_policy": (
                "same ten full-total-beam GPU arrays are sliced and reused for "
                "every geometry, partition, priming call, warm-up, and iteration"
            ),
        },
        "parameters": {
            "method": METHOD,
            "threshold": threshold,
            "dm_reaches": list(dm_reaches),
            "waist_bins": waist_bins,
            "total_beams": total_beams,
            "beam_batch_sizes": list(batch_sizes),
            "warmup": warmup,
            "iterations": iterations,
            "device": device,
        },
        "batching": {
            "policy": (
                "contiguous IDs 0..total_beams-1 exactly once; ceil division; "
                "smaller final batch allowed; independent persistent ten-extractor "
                "set per beam group; batches execute sequentially on one stream"
            ),
            "partitions": partitions,
        },
        "peakfinder_geometry": geometry_records,
        "steady_state": {
            "assume_steady_state": True,
            "source_chunk_origin": SOURCE_CHUNK_ORIGIN,
            "source_label_policy": {
                "priming": (
                    "for each tree with P required calls: consecutive labels "
                    "source_chunk_origin-P through source_chunk_origin-1"
                ),
                "warmup": "source_chunk_origin + warmup_index",
                "timed_iteration": (
                    "source_chunk_origin + warmup + iteration"
                ),
                "resume_replay": (
                    "rebuild and prime, repeat all warm-ups, then replay every "
                    "persisted iteration untimed at its original source label"
                ),
            },
            "startup_validity_scope": (
                "disables startup censoring only; it is not used as proof of a filled halo"
            ),
            "halo_columns": "exactly 2 * production geometry.time_radius",
            "priming_calls": (
                "max(1, floor(2*time_radius/ntime)+1), strictly enough that "
                "the next work array begins after the physical left edge"
            ),
            "priming_input": "repeat the same clean below-threshold chunk",
            "validation": (
                "inspect actual production tail shapes, total columns, emission "
                "frontier, last source chunk, and unflushed state before timing"
            ),
            "current_api_deviation": (
                "OfflinePeakExtractor has no public halo-filled accessor; this "
                "benchmark therefore inspects its current private persistent state"
            ),
            "flush_policy": "never call flush in priming, warm-up, replay, or timing",
        },
        "timing": {
            "authoritative_metric": "total_beams_peakfinding_wall_ms",
            "clock": "time.perf_counter synchronized before and after on current stream",
            "boundary": (
                "within one synchronized interval: sequentially process every beam "
                "batch; for each batch call production process_chunk for all ten "
                "trees, collect parts, and call production concatenate/sort helper"
            ),
            "included": [
                "finite-value and token validation",
                "full-band maximum filtering",
                "candidate mask and compaction",
                "streaming extractor bookkeeping",
                "all ten tree calls for every beam batch",
                "per-batch production candidate concatenation and deterministic sorting",
                "Python loop over all batches",
            ],
            "excluded": [
                "plan/geometry/extractor construction",
                "streaming halo priming and warm-up",
                "synthetic data generation and CPU-to-GPU upload",
                "flush",
                "decoder and grouper",
                "aggregation, disk I/O, CSV/YAML writing, plotting",
            ],
            "synchronization_inside": "none after individual trees or batches",
            "equivalent_ms_per_batch": "derived average, not synchronized batch latency",
            "optional_per_tree_cuda_diagnostics": "not implemented",
            "derived_formulas": {
                "peakfinding_ms_per_beam": "wall_ms / total_beams",
                "equivalent_ms_per_batch": "wall_ms / n_batches",
                "beam_chunks_per_second": "1000 * total_beams / wall_ms",
                "realtime_beam_capacity": "chunk_duration_ms * total_beams / wall_ms",
                "load_fraction_for_total_beams": "wall_ms / chunk_duration_ms",
            },
        },
        "gpu": dict(gpu),
        "software": dict(software),
        "git": dict(git),
        "outputs": {
            "timings.csv": list(TIMING_FIELDS),
            "summary.csv": list(SUMMARY_FIELDS),
            "tree_diagnostics.csv": list(TREE_DIAGNOSTIC_FIELDS),
            "metadata.yaml": "this document",
        },
        "progress": {
            "expected_timing_rows": expected_rows,
            "completed_timing_rows": 0,
        },
    }


def _signature_payload(
        *, args: argparse.Namespace, config_path: Path, plan_yaml: str,
        specs: Sequence[TreePlanSpec], diagnostics: Mapping[tuple[int, int], GeometryDiagnostic],
        gpu: Mapping[str, Any], software: Mapping[str, Any], git: Mapping[str, Any],
        ) -> dict[str, Any]:
    source_paths = (
        Path(__file__).resolve(),
        REPOSITORY_ROOT / "pirate_frb/Peakfinders.py",
    )
    return {
        "schema_name": SCHEMA_NAME,
        "schema_version": SCHEMA_VERSION,
        "sources": [file_identity(path) for path in source_paths],
        "config": file_identity(config_path),
        "plan_yaml_sha256": hashlib.sha256(plan_yaml.encode("utf-8")).hexdigest(),
        "tree_specs": [asdict(spec) for spec in specs],
        "geometry": [
            asdict(diagnostics[(reach, spec.tree_index)])
            for reach in args.dm_reaches for spec in specs
        ],
        "method": args.method,
        "total_beams": args.total_beams,
        "beam_batch_sizes": list(args.beam_batch_sizes),
        "dm_reaches": list(args.dm_reaches),
        "threshold": args.threshold,
        "waist_bins": args.waist_bins,
        "warmup": args.warmup,
        "iterations": args.iterations,
        "device": args.device,
        "base_seed": args.base_seed,
        "gpu": dict(gpu),
        "software": {
            key: software[key]
            for key in ("python", "numpy", "cupy", "pirate_frb_distribution")
        },
        "active_repository_sources": software["active_repository_sources"],
        "git_commit": git.get("commit"),
    }


def _timing_row(
        *, campaign_id: str, args: argparse.Namespace, iteration: int,
        partitions: Sequence[BeamBatch], wall_ms: float,
        total_candidates: int, chunk_duration_ms: float) -> dict[str, Any]:
    metrics = derive_timing_metrics(
        wall_ms, args.total_beams, len(partitions), chunk_duration_ms
    )
    row = {
        "schema_version": SCHEMA_VERSION,
        "campaign_id": campaign_id,
        "iteration": iteration,
        "base_seed": args.base_seed,
        "device": args.device,
        "method": args.method,
        "total_beams": args.total_beams,
        "beam_batch_size": args.current_batch_size,
        "n_batches": len(partitions),
        "actual_batch_sizes": json.dumps(
            [batch.size for batch in partitions], separators=(",", ":")
        ),
        "final_batch_size": partitions[-1].size,
        "threshold": args.threshold,
        "dm_reach": args.current_dm_reach,
        "waist_bins": args.waist_bins,
        "warmup": args.warmup,
        "total_pixels_per_beam": EXPECTED_PIXELS_PER_BEAM,
        "total_input_pixels": EXPECTED_PIXELS_PER_BEAM * args.total_beams,
        "total_candidates": total_candidates,
        "chunk_duration_ms": chunk_duration_ms,
        "total_beams_peakfinding_wall_ms": wall_ms,
        **metrics,
    }
    return _validate_rows([row], TIMING_FIELDS)[0]


def _print_configuration_summary(
        timing_rows: Sequence[Mapping[str, Any]], reach: int, batch_size: int) -> None:
    rows = [row for row in timing_rows
            if int(row["dm_reach"]) == reach
            and int(row["beam_batch_size"]) == batch_size]
    wall = np.asarray([
        float(row["total_beams_peakfinding_wall_ms"]) for row in rows
    ])
    load = np.asarray([float(row["load_fraction_for_total_beams"]) for row in rows])
    capacity = np.asarray([float(row["realtime_beam_capacity"]) for row in rows])
    print(
        f"dm_reach={reach:>2} batch={batch_size:>2} n={len(rows)} "
        f"wall={np.median(wall):.3f} ms [{np.min(wall):.3f}, {np.max(wall):.3f}] "
        f"load={100*np.median(load):.3f}% capacity={np.median(capacity):.1f} beams",
        flush=True,
    )


def _print_aggregate(timing_rows: Sequence[Mapping[str, Any]], dm_reaches: Sequence[int]) -> None:
    print("Aggregate fastest batch per dm_reach (median synchronized wall time):")
    summaries = summarize_timings(timing_rows)
    for reach in dm_reaches:
        matching = [row for row in summaries if int(row["dm_reach"]) == reach]
        if not matching:
            continue
        fastest = min(
            matching,
            key=lambda row: float(row["total_beams_peakfinding_wall_ms_median"]),
        )
        print(
            f"  dm_reach={reach:>2}: batch={int(fastest['beam_batch_size']):>2}, "
            f"wall={float(fastest['total_beams_peakfinding_wall_ms_median']):.3f} ms, "
            f"load={100*float(fastest['load_fraction_for_total_beams_median']):.3f}%, "
            f"capacity={float(fastest['realtime_beam_capacity_median']):.1f} beams"
        )


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--results-dir", type=Path, default=DEFAULT_RESULTS_DIR)
    parser.add_argument("--total-beams", type=int, default=DEFAULT_TOTAL_BEAMS)
    parser.add_argument(
        "--beam-batch-sizes", type=int, nargs="+",
        default=DEFAULT_BEAM_BATCH_SIZES
    )
    parser.add_argument(
        "--dm-reach", "--dm-reaches", dest="dm_reaches", type=int, nargs="+",
        default=DEFAULT_DM_REACHES
    )
    parser.add_argument("--threshold", type=float, default=DEFAULT_THRESHOLD)
    parser.add_argument("--waist-bins", type=int, default=DEFAULT_WAIST_BINS)
    parser.add_argument("--warmup", type=int, default=DEFAULT_WARMUP)
    parser.add_argument("--iterations", type=int, default=DEFAULT_ITERATIONS)
    parser.add_argument("--device", type=int, default=DEFAULT_DEVICE)
    parser.add_argument("--base-seed", type=int, default=DEFAULT_BASE_SEED)
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--overwrite", action="store_true")
    return parser


def validate_arguments(args: argparse.Namespace) -> argparse.Namespace:
    # Peakfinder choice is intentionally not a CLI surface: production has one
    # full-band implementation.  Retain the constant only in result metadata.
    args.method = METHOD
    args.total_beams = _exact_positive_integer(args.total_beams, "total_beams")
    args.beam_batch_sizes = _unique_positive_grid(
        args.beam_batch_sizes, "beam_batch_size"
    )
    args.dm_reaches = _unique_positive_grid(args.dm_reaches, "dm_reach")
    args.waist_bins = _exact_nonnegative_integer(args.waist_bins, "waist_bins")
    args.warmup = _exact_nonnegative_integer(args.warmup, "warmup")
    args.iterations = _exact_positive_integer(args.iterations, "iterations")
    args.device = _exact_nonnegative_integer(args.device, "device")
    args.base_seed = _exact_nonnegative_integer(args.base_seed, "base_seed")
    args.threshold = float(args.threshold)
    if not math.isfinite(args.threshold):
        raise ValueError("threshold must be finite")
    if args.resume and args.overwrite:
        raise ValueError("--resume and --overwrite are mutually exclusive")
    args.config = Path(args.config).resolve()
    args.results_dir = Path(args.results_dir).resolve()
    return args


def run_benchmark(args: argparse.Namespace) -> None:
    args = validate_arguments(args)
    paths = result_paths(args.results_dir)
    existing_files = [
        paths[name] for name in ("timings", "summary", "tree_diagnostics", "metadata")
        if paths[name].exists()
    ]
    if existing_files and not (args.resume or args.overwrite):
        raise FileExistsError(
            "result files already exist; use --resume or --overwrite: "
            + ", ".join(str(path) for path in existing_files)
        )

    import cupy as cp

    with cp.cuda.Device(args.device):
        stream = cp.cuda.get_current_stream()
        active_sources = active_repository_sources()
        bundle = load_authoritative_plan(args.config)
        # Resolve the production concatenation helper before any possible
        # authoritative call, including campaigns configured with warmup=0.
        from pirate_frb.Peakfinders import concatenate_raw_candidates
        if int(bundle.config.beams_per_gpu) != DEFAULT_TOTAL_BEAMS:
            raise ValueError(
                "authoritative config no longer declares 60 beams per GPU"
            )

        # Generate and validate once on the host, then upload once. Neither
        # operation is inside any warm-up or authoritative timed interval.
        clean_inputs = generate_clean_inputs(
            bundle.specs, args.total_beams, args.base_seed, args.threshold
        )
        decoded_tokens = validate_tokens_with_plan(bundle.plan, clean_inputs)
        inputs_summary = [
            {
                "tree_index": item.spec.tree_index,
                "shape": list(item.snr.shape),
                "noise_seed": item.noise_seed,
                "minimum_snr": float(np.min(item.snr)),
                "maximum_snr": float(np.max(item.snr)),
                "finite": bool(np.all(np.isfinite(item.snr))),
                "above_or_equal_threshold": int(np.count_nonzero(item.snr >= args.threshold)),
                "token_multiplet": item.token_multiplet,
                "token_uint32": item.token,
                "plan_decode_at_idm0_itime0": list(decoded_token),
            }
            for item, decoded_token in zip(clean_inputs, decoded_tokens)
        ]
        full_snr_maps, full_argmax_maps = upload_inputs(cp, clean_inputs)
        del clean_inputs

        geometries_by_reach, diagnostics = build_geometries(
            cp, bundle.plan, bundle.specs, args.dm_reaches, args.waist_bins
        )
        gpu = gpu_information(cp, args.device)
        software = software_information(cp)
        software["active_repository_sources"] = active_sources
        git = git_information()
        warning = None
        if "NVIDIA A40" not in str(gpu["model"]):
            warning = (
                f"Campaign GPU is {gpu['model']!r}, not an NVIDIA A40; "
                "retain this warning when interpreting portability."
            )
            print(f"WARNING: {warning}", file=sys.stderr, flush=True)

        # The active offline consumer plan is intentionally incomplete and
        # cannot be serialized. Retain the exact producer serialization used
        # to reconstruct it in load_authoritative_plan().
        plan_yaml = bundle.producer_plan_yaml
        signature_payload = _signature_payload(
            args=args, config_path=args.config, plan_yaml=plan_yaml,
            specs=bundle.specs, diagnostics=diagnostics, gpu=gpu,
            software=software, git=git
        )
        signature = campaign_signature(signature_payload)
        campaign_id = signature[:16]
        tree_rows = build_tree_diagnostic_rows(
            campaign_id, bundle.specs, diagnostics, args.dm_reaches,
            args.beam_batch_sizes, args.total_beams
        )
        timing_rows, existing_metadata = prepare_result_state(
            paths, resume=args.resume, overwrite=args.overwrite,
            signature=signature, campaign_id=campaign_id,
            tree_rows=tree_rows, dm_reaches=args.dm_reaches,
            batch_sizes=args.beam_batch_sizes, iterations=args.iterations
        )
        validate_resumed_timing_rows(
            timing_rows, campaign_id=campaign_id,
            dm_reaches=args.dm_reaches, batch_sizes=args.beam_batch_sizes,
            iterations=args.iterations, total_beams=args.total_beams,
            base_seed=args.base_seed, device=args.device, method=args.method,
            threshold=args.threshold, waist_bins=args.waist_bins,
            warmup=args.warmup, chunk_duration_ms=bundle.chunk_duration_ms,
        )
        metadata = _metadata_document(
            campaign_id=campaign_id, signature=signature,
            signature_payload=signature_payload, bundle=bundle,
            config_path=args.config, plan_yaml=plan_yaml,
            inputs_summary=inputs_summary, diagnostics=diagnostics,
            dm_reaches=args.dm_reaches, batch_sizes=args.beam_batch_sizes,
            total_beams=args.total_beams, base_seed=args.base_seed,
            threshold=args.threshold, waist_bins=args.waist_bins,
            warmup=args.warmup, iterations=args.iterations, device=args.device,
            gpu=gpu, software=software, git=git, warning=warning,
            created_utc=(
                str(existing_metadata["created_utc"])
                if existing_metadata is not None else None
            ),
        )
        if existing_metadata is not None:
            metadata["resumed_utc"] = _utc_now()
        metadata["progress"]["completed_timing_rows"] = len(timing_rows)
        metadata["campaign_complete"] = (
            len(timing_rows)
            == len(expected_timing_keys(
                args.dm_reaches, args.beam_batch_sizes, args.iterations
            ))
        )
        write_result_bundle(paths, timing_rows, tree_rows, metadata)

        if metadata["campaign_complete"]:
            print("Campaign checkpoint is already complete; nothing to resume.")
            _print_aggregate(timing_rows, args.dm_reaches)
            return

        for reach in args.dm_reaches:
            for batch_size in args.beam_batch_sizes:
                completed = sum(
                    int(row["dm_reach"]) == reach
                    and int(row["beam_batch_size"]) == batch_size
                    for row in timing_rows
                )
                if completed == args.iterations:
                    _print_configuration_summary(timing_rows, reach, batch_size)
                    continue

                partitions = partition_beams(args.total_beams, batch_size)
                states = construct_and_prime_batch_states(
                    geometries_by_reach[reach], partitions,
                    full_snr_maps, full_argmax_maps, args.threshold, stream
                )

                # Warm-up and resume replay both advance the exact persistent
                # streams with consecutive source labels, entirely untimed.
                for warmup_index in range(args.warmup):
                    process_complete_workload(
                        states, SOURCE_CHUNK_ORIGIN + warmup_index,
                        concatenate_raw_candidates
                    )
                for iteration in range(completed):
                    process_complete_workload(
                        states, SOURCE_CHUNK_ORIGIN + args.warmup + iteration,
                        concatenate_raw_candidates
                    )
                stream.synchronize()
                last_untimed_chunk = (
                    SOURCE_CHUNK_ORIGIN + args.warmup + completed - 1
                    if args.warmup + completed
                    else SOURCE_CHUNK_ORIGIN - 1
                )
                for state in states:
                    for extractor in state.extractors:
                        assert_filled_streaming_halo(
                            extractor, expected_last_chunk=last_untimed_chunk
                        )

                for iteration in range(completed, args.iterations):
                    source_chunk = SOURCE_CHUNK_ORIGIN + args.warmup + iteration
                    wall_ms, total_candidates = synchronized_wall_time(
                        stream,
                        lambda source_chunk=source_chunk: process_complete_workload(
                            states, source_chunk, concatenate_raw_candidates
                        ),
                    )
                    _assert_zero_candidates(total_candidates, "timed complete workload")
                    for state in states:
                        for extractor in state.extractors:
                            assert_filled_streaming_halo(
                                extractor, expected_last_chunk=source_chunk
                            )

                    args.current_dm_reach = reach
                    args.current_batch_size = batch_size
                    timing_rows.append(_timing_row(
                        campaign_id=campaign_id, args=args,
                        iteration=iteration, partitions=partitions,
                        wall_ms=wall_ms, total_candidates=total_candidates,
                        chunk_duration_ms=bundle.chunk_duration_ms,
                    ))
                    validate_timing_prefix(
                        timing_rows, args.dm_reaches, args.beam_batch_sizes,
                        args.iterations, campaign_id
                    )
                    metadata["last_updated_utc"] = _utc_now()
                    metadata["progress"]["completed_timing_rows"] = len(timing_rows)
                    write_result_bundle(paths, timing_rows, tree_rows, metadata)
                    print(
                        f"checkpoint dm_reach={reach} batch={batch_size} "
                        f"iteration={iteration} wall={wall_ms:.3f} ms "
                        f"candidates={total_candidates}",
                        flush=True,
                    )

                _print_configuration_summary(timing_rows, reach, batch_size)
                # No flush: discarding these states ends this independent
                # configuration after all normal streaming calls are complete.
                del states
                stream.synchronize()

        metadata["campaign_complete"] = True
        metadata["last_updated_utc"] = _utc_now()
        metadata["completed_utc"] = metadata["last_updated_utc"]
        metadata["progress"]["completed_timing_rows"] = len(timing_rows)
        write_result_bundle(paths, timing_rows, tree_rows, metadata)
        _print_aggregate(timing_rows, args.dm_reaches)
        print(f"Results: {paths['directory']}", flush=True)


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    run_benchmark(args)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
