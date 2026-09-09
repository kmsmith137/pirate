#!/usr/bin/env python3
"""Benchmark CPU/GPU grouping after concentrating hot pixels in selected maps.

The benchmark constructs real ragged PIRATE S/N maps, runs the production
streaming peak finder and argmax decoder, then sends the same decoded GPU table
to the production GPU grouper and the exact benchmark-side CPU grouper.  Three
layouts separate candidate count from partition topology:

``single_map``
    Populate only one beam/tree map.  Every candidate belongs to one tree and
    therefore becomes a singleton event in one large partition.
``single_family``
    Populate the selected tree and every sibling with the same primary-tree
    index.  Candidates can associate across trees, but remain in one partition.
``all_maps``
    Populate every tree map of one beam.  This is the diluted control and
    normally creates one partition per primary-tree family.

All injected pixels have the same finite float16 value.  A maximum filter
retains equal plateaus, so every injected pixel is expected to survive as one
raw candidate even when populated coordinates touch.  This deliberately
adversarial choice gives exact control of the grouping input size rather than
measuring an occupancy-dependent peak-thinning effect.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import math
import os
from pathlib import Path
import platform
import statistics
import tempfile
import time
from typing import Any, Mapping, Sequence

import numpy as np
import yaml

from peakfinder_tests import benchmark_gaussian_corruption_timing as gaussian
from peakfinder_tests import benchmark_peakfinder_batch_timing as batch_benchmark


SCHEMA_NAME = "pirate-concentrated-map-grouper-timing"
SCHEMA_VERSION = 2
REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_CONFIG = REPOSITORY_ROOT / "configs/dedispersion/chord_sb2_et.yml"
DEFAULT_RESULTS_DIR = (
    Path(__file__).resolve().parent
    / "results_concentrated_map_grouper_timing_pirate15"
)
DEFAULT_LAYOUTS = ("single_map", "single_family", "all_maps")
DEFAULT_HOT_PIXEL_COUNTS = (64, 128, 256, 512, 1024, 2048, 4096)
DEFAULT_SELECTED_TREE = 5
DEFAULT_REPEATS = 5
DEFAULT_THRESHOLD = 10.0
DEFAULT_HOT_SNR = 32.0
DEFAULT_DM_REACH = 8
DEFAULT_WAIST_BINS = 1
DEFAULT_BASE_SEED = 20260903
DEFAULT_CPU_CANDIDATE_LIMIT = 5000
DEFAULT_MAXIMUM_UNBOUNDED_CANDIDATES = 5000


CSV_FIELDS = (
    "schema_version",
    "layout",
    "selected_tree",
    "selected_primary_tree",
    "populated_tree_indices",
    "populated_map_count",
    "selected_map_pixels",
    "hot_pixel_count",
    "selected_map_occupancy_percent",
    "raw_candidate_count",
    "decoded_candidate_count",
    "partition_count",
    "largest_partition",
    "repeat",
    "timing_order",
    "peakfinder_wall_ms",
    "decoder_wall_ms",
    "gpu_grouper_wall_ms",
    "gpu_retained_candidates",
    "gpu_grouped_events",
    "gpu_timed_out",
    "gpu_complete",
    "gpu_candidates_per_second",
    "cpu_status",
    "cpu_grouper_wall_ms",
    "cpu_grouped_events",
    "cpu_candidates_per_second",
    "cpu_candidate_normalization_ms",
    "cpu_gpu_to_cpu_transfer_ms",
    "cpu_grouping_ms",
    "cpu_cpu_to_gpu_result_ms",
    "gpu_over_cpu_speedup",
    "exact_parity_verified",
)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--results-dir", type=Path, default=DEFAULT_RESULTS_DIR)
    parser.add_argument(
        "--layouts", nargs="+", choices=DEFAULT_LAYOUTS,
        default=list(DEFAULT_LAYOUTS),
    )
    parser.add_argument(
        "--hot-pixel-counts", nargs="+", type=int,
        default=list(DEFAULT_HOT_PIXEL_COUNTS),
    )
    parser.add_argument("--selected-tree", type=int, default=DEFAULT_SELECTED_TREE)
    parser.add_argument("--repeats", type=int, default=DEFAULT_REPEATS)
    parser.add_argument("--threshold", type=float, default=DEFAULT_THRESHOLD)
    parser.add_argument("--hot-snr", type=float, default=DEFAULT_HOT_SNR)
    parser.add_argument("--dm-reach", type=int, default=DEFAULT_DM_REACH)
    parser.add_argument("--waist-bins", type=int, default=DEFAULT_WAIST_BINS)
    parser.add_argument("--base-seed", type=int, default=DEFAULT_BASE_SEED)
    parser.add_argument("--device", type=int, default=0)
    parser.add_argument(
        "--gpu-timeout-ms", type=float, default=0.0,
        help="production cooperative grouping timeout; zero disables it",
    )
    parser.add_argument(
        "--cpu-candidate-limit", type=int,
        default=DEFAULT_CPU_CANDIDATE_LIMIT,
        help="skip the CPU comparison above this decoded candidate count",
    )
    parser.add_argument(
        "--maximum-unbounded-candidates", type=int,
        default=DEFAULT_MAXIMUM_UNBOUNDED_CANDIDATES,
        help="largest allowed count when the GPU timeout is disabled",
    )
    parser.add_argument(
        "--allow-unbounded-gpu", action="store_true",
        help="allow larger quadratic GPU runs without a cooperative timeout",
    )
    parser.add_argument("--overwrite", action="store_true")
    return parser


def _positive_integer(value: Any, name: str) -> int:
    result = gaussian._exact_positive_integer(value, name)
    return int(result)


def _nonnegative_finite(value: Any, name: str) -> float:
    if isinstance(value, (bool, np.bool_)):
        raise TypeError(f"{name} must be a real scalar")
    result = float(value)
    if not math.isfinite(result) or result < 0.0:
        raise ValueError(f"{name} must be finite and non-negative")
    return result


def validate_arguments(args: argparse.Namespace) -> argparse.Namespace:
    args.layouts = tuple(args.layouts)
    if not args.layouts or len(set(args.layouts)) != len(args.layouts):
        raise ValueError("--layouts must be non-empty and unique")
    args.hot_pixel_counts = tuple(
        _positive_integer(value, "hot pixel count")
        for value in args.hot_pixel_counts
    )
    if len(set(args.hot_pixel_counts)) != len(args.hot_pixel_counts):
        raise ValueError("--hot-pixel-counts must not contain duplicates")
    args.selected_tree = gaussian._exact_nonnegative_integer(
        args.selected_tree, "selected_tree"
    )
    args.repeats = _positive_integer(args.repeats, "repeats")
    args.dm_reach = gaussian._exact_nonnegative_integer(
        args.dm_reach, "dm_reach"
    )
    args.waist_bins = gaussian._exact_nonnegative_integer(
        args.waist_bins, "waist_bins"
    )
    args.base_seed = gaussian._exact_nonnegative_integer(
        args.base_seed, "base_seed"
    )
    args.device = gaussian._exact_nonnegative_integer(args.device, "device")
    args.cpu_candidate_limit = _positive_integer(
        args.cpu_candidate_limit, "cpu_candidate_limit"
    )
    args.maximum_unbounded_candidates = _positive_integer(
        args.maximum_unbounded_candidates,
        "maximum_unbounded_candidates",
    )
    args.threshold = float(args.threshold)
    args.hot_snr = float(args.hot_snr)
    if (not math.isfinite(args.threshold) or not math.isfinite(args.hot_snr)
            or args.hot_snr <= args.threshold):
        raise ValueError("--hot-snr must be finite and exceed the finite threshold")
    encoded_hot_snr = np.float16(args.hot_snr)
    if not np.isfinite(encoded_hot_snr) or encoded_hot_snr <= np.float16(args.threshold):
        raise ValueError("--hot-snr must remain above threshold after float16 conversion")
    args.gpu_timeout_ms = _nonnegative_finite(
        args.gpu_timeout_ms, "gpu_timeout_ms"
    )
    if (max(args.hot_pixel_counts) > args.maximum_unbounded_candidates
            and args.gpu_timeout_ms == 0.0
            and not args.allow_unbounded_gpu):
        raise ValueError(
            "requested count exceeds --maximum-unbounded-candidates while the "
            "GPU timeout is disabled; set --gpu-timeout-ms or explicitly pass "
            "--allow-unbounded-gpu"
        )
    return args


def selected_tree_indices(
        specs: Sequence[Any], layout: str, selected_tree: int,
        ) -> tuple[int, ...]:
    """Return the exact set of maps populated by one layout."""

    if not 0 <= selected_tree < len(specs):
        raise ValueError("selected tree lies outside the producer plan")
    if layout == "single_map":
        return (selected_tree,)
    if layout == "single_family":
        family = int(specs[selected_tree].primary_tree_index)
        return tuple(
            int(spec.tree_index) for spec in specs
            if int(spec.primary_tree_index) == family
        )
    if layout == "all_maps":
        return tuple(int(spec.tree_index) for spec in specs)
    raise ValueError(f"unknown concentrated-map layout {layout!r}")


def populate_host_maps(
        clean_maps: Sequence[np.ndarray], specs: Sequence[Any], *, layout: str,
        selected_tree: int, hot_pixel_count: int, hot_snr: float,
        base_seed: int,
        ) -> tuple[tuple[np.ndarray, ...], tuple[int, ...]]:
    """Populate an exact number of equal-height pixels in selected beam-zero maps."""

    trees = selected_tree_indices(specs, layout, selected_tree)
    if len(clean_maps) != len(specs):
        raise ValueError("clean map/spec count differs")
    spans = []
    total = 0
    for tree in trees:
        array = np.asarray(clean_maps[tree])
        expected = (1, int(specs[tree].ndm), int(specs[tree].ntime))
        if array.shape != expected or array.dtype != gaussian.SNR_DTYPE:
            raise ValueError(f"clean tree {tree} must have shape {expected} and float16 dtype")
        count = int(np.prod(expected[1:], dtype=np.int64))
        spans.append((tree, total, total + count))
        total += count
    hot_pixel_count = _positive_integer(hot_pixel_count, "hot_pixel_count")
    if hot_pixel_count > total:
        raise ValueError(
            f"layout {layout!r} contains {total} pixels, fewer than requested "
            f"count {hot_pixel_count}"
        )

    seed = gaussian.derive_seed(base_seed, hot_pixel_count, selected_tree, layout)
    rng = np.random.default_rng(seed)
    selected = np.sort(
        rng.choice(total, size=hot_pixel_count, replace=False).astype(np.int64)
    )
    result = tuple(np.array(array, copy=True) for array in clean_maps)
    hot_value = np.float16(hot_snr)
    for tree, begin, end in spans:
        local = selected[(selected >= begin) & (selected < end)] - begin
        if local.size:
            result[tree][0].reshape(-1)[local] = hot_value

    observed = sum(
        int(np.count_nonzero(result[tree] == hot_value))
        for tree in trees
    )
    if observed != hot_pixel_count:
        raise AssertionError("concentrated map population lost or duplicated hot pixels")
    for tree, array in enumerate(result):
        if tree not in trees and not np.array_equal(array, clean_maps[tree]):
            raise AssertionError("unselected S/N map was modified")
    return result, trees


def _partition_statistics(cp: Any, decoded: Any) -> tuple[int, int]:
    if not len(decoded):
        return 0, 0
    primary_tree_count = int(decoded.primary_tree_index.max().item()) + 1
    keys = (
        decoded.beam_id.astype(cp.int64) * np.int64(primary_tree_count)
        + decoded.primary_tree_index.astype(cp.int64)
    )
    _, counts = cp.unique(keys, return_counts=True)
    return int(counts.size), int(cp.max(counts).item())


def _rate(count: int, wall_ms: float | None) -> Any:
    if wall_ms is None:
        return ""
    if wall_ms == 0.0:
        return math.inf
    return 1000.0 * count / wall_ms


def _write_csv(path: Path, rows: Sequence[Mapping[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=CSV_FIELDS)
        writer.writeheader()
        writer.writerows(rows)
    temporary.replace(path)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        while block := stream.read(1024 * 1024):
            digest.update(block)
    return digest.hexdigest()


def _write_metadata(
        path: Path, *, args: argparse.Namespace, bundle: Any,
        gpu: Mapping[str, Any], software: Mapping[str, Any], rows: Sequence[Any],
        ) -> None:
    source_paths = {
        "benchmark": Path(__file__).resolve(),
        "peakfinder": REPOSITORY_ROOT / "pirate_frb/Peakfinders.py",
        "decoder": REPOSITORY_ROOT / "pirate_frb/GpuArgmaxDecoder.py",
        "gpu_grouper": REPOSITORY_ROOT / "pirate_frb/OfflineCandidateGrouper.py",
        "cpu_grouper": Path(__file__).with_name("cpu_candidate_grouper.py"),
        "producer_metadata": Path(__file__).with_name("producer_metadata.py"),
    }
    metadata = {
        "schema_name": SCHEMA_NAME,
        "schema_version": SCHEMA_VERSION,
        "created_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "config": {
            "path": str(Path(args.config).resolve()),
            "sha256": _sha256(Path(args.config).resolve()),
            "producer_plan_sha256": hashlib.sha256(
                bundle.producer_plan_yaml.encode("utf-8")
            ).hexdigest(),
        },
        "producer": {
            "dcores": list(bundle.dcores),
            "argmax_encoding": bundle.argmax_encoding,
            "plan_yaml": bundle.producer_plan_yaml,
            "trees": [dict(spec.__dict__) for spec in bundle.specs],
        },
        "workload": {
            "layouts": list(args.layouts),
            "hot_pixel_counts": list(args.hot_pixel_counts),
            "selected_tree": args.selected_tree,
            "selected_primary_tree": int(
                bundle.specs[args.selected_tree].primary_tree_index
            ),
            "population": "equal finite float16 supra-threshold pixels",
            "threshold": args.threshold,
            "hot_snr": args.hot_snr,
            "base_seed": args.base_seed,
            "dm_reach": args.dm_reach,
            "waist_bins": args.waist_bins,
            "repeats": args.repeats,
        },
        "safety": {
            "gpu_timeout_ms": args.gpu_timeout_ms,
            "cpu_candidate_limit": args.cpu_candidate_limit,
            "maximum_unbounded_candidates": args.maximum_unbounded_candidates,
            "allow_unbounded_gpu": bool(args.allow_unbounded_gpu),
        },
        "environment": {
            "gpu": dict(gpu),
            "software": dict(software),
            "python": platform.python_version(),
        },
        "sources": {
            name: {"path": str(source), "sha256": _sha256(source)}
            for name, source in source_paths.items()
        },
        "csv": {
            "path": "trials.csv",
            "fields": list(CSV_FIELDS),
            "rows": len(rows),
        },
    }
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, temporary_name = tempfile.mkstemp(
        prefix=path.name + ".", suffix=".tmp", dir=path.parent
    )
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as stream:
            yaml.safe_dump(metadata, stream, sort_keys=False)
        Path(temporary_name).replace(path)
    except Exception:
        try:
            os.unlink(temporary_name)
        except FileNotFoundError:
            pass
        raise


def _prepare_output(args: argparse.Namespace) -> tuple[Path, Path]:
    directory = args.results_dir.resolve()
    csv_path = directory / "trials.csv"
    metadata_path = directory / "metadata.yaml"
    existing = [path for path in (csv_path, metadata_path) if path.exists()]
    if existing and not args.overwrite:
        raise FileExistsError(
            "result files already exist; use --overwrite or a new directory: "
            + ", ".join(map(str, existing))
        )
    return csv_path, metadata_path


def run_benchmark(args: argparse.Namespace) -> list[dict[str, Any]]:
    args = validate_arguments(args)
    csv_path, metadata_path = _prepare_output(args)

    import cupy as cp
    from pirate_frb.GpuArgmaxDecoder import GpuArgmaxDecoder
    from pirate_frb.OfflineCandidateGrouper import (
        GroupingConfig,
        GroupingGeometry,
        group_candidates,
    )
    from pirate_frb.Peakfinders import concatenate_raw_candidates
    from peakfinder_tests.cpu_candidate_grouper import (
        assert_gpu_grouping_results_equal,
        synchronized_cpu_grouping_wall_time,
    )

    with cp.cuda.Device(args.device):
        stream = cp.cuda.get_current_stream()
        bundle = batch_benchmark.load_authoritative_plan(args.config)
        if not 0 <= args.selected_tree < len(bundle.specs):
            raise ValueError("selected tree lies outside the authoritative plan")
        for layout in args.layouts:
            capacity = sum(
                int(bundle.specs[tree].pixels_per_beam)
                for tree in selected_tree_indices(
                    bundle.specs, layout, args.selected_tree
                )
            )
            if max(args.hot_pixel_counts) > capacity:
                raise ValueError(
                    f"layout {layout!r} capacity {capacity} is smaller than "
                    f"the largest requested hot-pixel count"
                )

        geometries_by_reach, _ = batch_benchmark.build_geometries(
            cp, bundle.plan, bundle.specs, (args.dm_reach,), args.waist_bins
        )
        peak_geometries = geometries_by_reach[args.dm_reach]
        decoder = GpuArgmaxDecoder(bundle.plan, cuda_device_id=args.device, dcores=bundle.dcores)
        grouping_geometry = GroupingGeometry.from_plan(bundle.plan)
        grouping_config = GroupingConfig()
        clean_host = gaussian.generate_clean_context_maps(
            bundle.specs, 1, args.base_seed, args.threshold
        )
        argmax_host, _ = gaussian.generate_valid_argmax_maps(
            bundle.specs, 1, args.base_seed
        )
        context_gpu = tuple(cp.asarray(array) for array in clean_host)
        argmax_gpu = tuple(cp.asarray(array) for array in argmax_host)
        stream.synchronize()

        def make_decoded(layout: str, count: int):
            target_host, trees = populate_host_maps(
                clean_host, bundle.specs, layout=layout,
                selected_tree=args.selected_tree, hot_pixel_count=count,
                hot_snr=args.hot_snr, base_seed=args.base_seed,
            )
            target_gpu = tuple(cp.asarray(array) for array in target_host)
            stream.synchronize()
            peakfinder_ms, extracted = gaussian.synchronized_wall_time(
                stream,
                lambda: gaussian.extract_complete_target_candidates(
                    cp, peak_geometries, target_gpu, context_gpu, argmax_gpu,
                    (0,), args.threshold, concatenate_raw_candidates,
                ),
            )
            raw, _ = extracted
            if len(raw) != count:
                raise AssertionError(
                    f"{layout} requested {count} equal-height hot pixels but "
                    f"peak finding emitted {len(raw)} candidates"
                )
            decoder_ms, decoded = gaussian.synchronized_wall_time(
                stream, lambda: decoder.decode(raw)
            )
            if len(decoded) != count:
                raise AssertionError("decoder changed the concentrated candidate count")
            del target_gpu
            return peakfinder_ms, decoder_ms, decoded, trees

        # Compile and allocate every measured path before the first result row.
        warm_count = min(32, min(args.hot_pixel_counts))
        for layout in args.layouts:
            _, _, warm_decoded, _ = make_decoded(layout, warm_count)
            warm_gpu = group_candidates(
                warm_decoded, grouping_geometry, config=grouping_config,
                timeout_ms=args.gpu_timeout_ms,
            )
            if len(warm_decoded) <= args.cpu_candidate_limit:
                _, warm_cpu, _ = synchronized_cpu_grouping_wall_time(
                    stream, warm_decoded, grouping_geometry,
                    config=grouping_config, cp_module=cp,
                )
                if warm_gpu.complete:
                    assert_gpu_grouping_results_equal(cp, warm_cpu, warm_gpu)
            stream.synchronize()

        gpu = batch_benchmark.gpu_information(cp, args.device)
        software = batch_benchmark.software_information(cp)
        rows: list[dict[str, Any]] = []
        _write_csv(csv_path, rows)
        _write_metadata(
            metadata_path, args=args, bundle=bundle, gpu=gpu,
            software=software, rows=rows,
        )

        point_ordinal = 0
        for layout in args.layouts:
            for hot_count in args.hot_pixel_counts:
                peakfinder_ms, decoder_ms, decoded, trees = make_decoded(
                    layout, hot_count
                )
                partition_count, largest_partition = _partition_statistics(
                    cp, decoded
                )
                selected_pixels = sum(
                    int(bundle.specs[tree].pixels_per_beam) for tree in trees
                )
                cpu_enabled = hot_count <= args.cpu_candidate_limit
                gpu_times = []
                cpu_times = []

                for repeat in range(args.repeats):
                    cpu_result = None
                    cpu_diagnostics = None
                    cpu_ms = None
                    gpu_result = None
                    timing_order = (
                        "gpu_then_cpu"
                        if (point_ordinal + repeat) % 2 == 0
                        else "cpu_then_gpu"
                    )

                    def run_gpu():
                        return gaussian.synchronized_wall_time(
                            stream,
                            lambda: group_candidates(
                                decoded, grouping_geometry,
                                config=grouping_config,
                                timeout_ms=args.gpu_timeout_ms,
                            ),
                        )

                    def run_cpu():
                        return synchronized_cpu_grouping_wall_time(
                            stream, decoded, grouping_geometry,
                            config=grouping_config, cp_module=cp,
                        )

                    if not cpu_enabled:
                        timing_order = "gpu_only"
                        gpu_ms, gpu_result = run_gpu()
                    elif timing_order == "gpu_then_cpu":
                        gpu_ms, gpu_result = run_gpu()
                        cpu_ms, cpu_result, cpu_diagnostics = run_cpu()
                    else:
                        cpu_ms, cpu_result, cpu_diagnostics = run_cpu()
                        gpu_ms, gpu_result = run_gpu()

                    parity = False
                    if cpu_result is not None and gpu_result.complete:
                        assert_gpu_grouping_results_equal(
                            cp, cpu_result, gpu_result
                        )
                        parity = True
                    if gpu_result.complete and len(gpu_result.candidates) != hot_count:
                        raise AssertionError("complete GPU grouping lost candidates")
                    if layout == "single_map":
                        if (cpu_result is not None
                                and len(cpu_result.events) != hot_count):
                            raise AssertionError(
                                "same-tree CPU candidates did not remain singleton events"
                            )
                        if (gpu_result.complete
                                and len(gpu_result.events) != hot_count):
                            raise AssertionError(
                                "same-tree GPU candidates did not remain singleton events"
                            )

                    cpu_events = (
                        "" if cpu_result is None else len(cpu_result.events)
                    )
                    speedup = (
                        "" if cpu_ms is None or not gpu_result.complete
                        else cpu_ms / gpu_ms
                    )
                    rows.append({
                        "schema_version": SCHEMA_VERSION,
                        "layout": layout,
                        "selected_tree": args.selected_tree,
                        "selected_primary_tree": int(
                            bundle.specs[args.selected_tree].primary_tree_index
                        ),
                        "populated_tree_indices": ",".join(map(str, trees)),
                        "populated_map_count": len(trees),
                        "selected_map_pixels": selected_pixels,
                        "hot_pixel_count": hot_count,
                        "selected_map_occupancy_percent": (
                            100.0 * hot_count / selected_pixels
                        ),
                        "raw_candidate_count": hot_count,
                        "decoded_candidate_count": hot_count,
                        "partition_count": partition_count,
                        "largest_partition": largest_partition,
                        "repeat": repeat,
                        "timing_order": timing_order,
                        "peakfinder_wall_ms": peakfinder_ms,
                        "decoder_wall_ms": decoder_ms,
                        "gpu_grouper_wall_ms": gpu_ms,
                        "gpu_retained_candidates": len(gpu_result.candidates),
                        "gpu_grouped_events": len(gpu_result.events),
                        "gpu_timed_out": int(gpu_result.timed_out),
                        "gpu_complete": int(gpu_result.complete),
                        "gpu_candidates_per_second": _rate(hot_count, gpu_ms),
                        "cpu_status": "completed" if cpu_enabled else "skipped_limit",
                        "cpu_grouper_wall_ms": "" if cpu_ms is None else cpu_ms,
                        "cpu_grouped_events": cpu_events,
                        "cpu_candidates_per_second": _rate(hot_count, cpu_ms),
                        "cpu_candidate_normalization_ms": (
                            "" if cpu_diagnostics is None else
                            cpu_diagnostics.candidate_normalization_ms
                        ),
                        "cpu_gpu_to_cpu_transfer_ms": (
                            "" if cpu_diagnostics is None else
                            cpu_diagnostics.gpu_to_cpu_transfer_ms
                        ),
                        "cpu_grouping_ms": (
                            "" if cpu_diagnostics is None else
                            cpu_diagnostics.cpu_grouping_ms
                        ),
                        "cpu_cpu_to_gpu_result_ms": (
                            "" if cpu_diagnostics is None else
                            cpu_diagnostics.cpu_to_gpu_result_ms
                        ),
                        "gpu_over_cpu_speedup": speedup,
                        "exact_parity_verified": int(parity),
                    })
                    gpu_times.append(gpu_ms)
                    if cpu_ms is not None:
                        cpu_times.append(cpu_ms)
                    _write_csv(csv_path, rows)
                    _write_metadata(
                        metadata_path, args=args, bundle=bundle, gpu=gpu,
                        software=software, rows=rows,
                    )

                gpu_label = f"{statistics.median(gpu_times):.3f} ms"
                cpu_label = (
                    f"{statistics.median(cpu_times):.3f} ms"
                    if cpu_times else "skipped"
                )
                print(
                    f"{layout:13s} hot={hot_count:5d} "
                    f"partitions={partition_count} largest={largest_partition} "
                    f"GPU={gpu_label} CPU={cpu_label}",
                    flush=True,
                )
                point_ordinal += 1

        return rows


def main() -> int:
    args = build_parser().parse_args()
    rows = run_benchmark(args)
    print(
        f"Wrote {len(rows)} rows to "
        f"{(args.results_dir.resolve() / 'trials.csv')}",
        flush=True,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
