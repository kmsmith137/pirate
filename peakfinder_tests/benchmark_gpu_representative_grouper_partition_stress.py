#!/usr/bin/env python3
"""Stress the representative grouper with one deliberately large partition.

This benchmark bypasses map generation, peak finding, and decoding so it can
control the grouping topology exactly. Both implementations receive the same
canonical candidate object.

The isolated scenario puts every candidate in one beam/family partition but
separates arrival times beyond the compatibility window. Every candidate then
becomes a representative, forcing the quadratic representative scans. The
dense scenario keeps the same partition and candidate counts but makes all
rows mutually compatible, providing a lower-event-count control.

Production remains the scientific oracle. Every timed result is compared
field-for-field with the benchmark-only persistent GPU implementation outside
the timing interval.
"""

from __future__ import annotations

import argparse
import csv
import math
import statistics
import time
from pathlib import Path
from typing import Any, Callable


DEFAULT_OUTPUT = (
    Path(__file__).resolve().parent
    / "results_gpu_representative_grouper_partition_stress.csv"
)
CSV_FIELDS = (
    "scenario",
    "candidate_count",
    "tree_count",
    "partition_count",
    "largest_partition",
    "grouped_events",
    "repeat",
    "timing_order",
    "production_wall_ms",
    "experimental_wall_ms",
    "production_candidates_per_second",
    "experimental_candidates_per_second",
    "production_over_experimental_speedup",
    "exact_parity_verified",
)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--sizes",
        nargs="+",
        type=int,
        default=[64, 128, 256, 512, 1024, 2048],
        help="candidate counts to test (default: 64 128 256 512 1024 2048)",
    )
    parser.add_argument(
        "--scenarios",
        nargs="+",
        choices=("isolated", "dense"),
        default=["isolated", "dense"],
    )
    parser.add_argument("--trees", type=int, default=10)
    parser.add_argument("--repeats", type=int, default=3)
    parser.add_argument("--device", type=int, default=0)
    parser.add_argument(
        "--production-safety-limit",
        type=int,
        default=5000,
        help=(
            "refuse larger synthetic inputs unless "
            "--allow-unsafe-production is supplied"
        ),
    )
    parser.add_argument("--allow-unsafe-production", action="store_true")
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--overwrite", action="store_true")
    return parser


def _validate_args(args: argparse.Namespace) -> None:
    if not args.sizes or any(size <= 0 for size in args.sizes):
        raise ValueError("--sizes must contain positive integers")
    if len(set(args.sizes)) != len(args.sizes):
        raise ValueError("--sizes must not contain duplicates")
    if args.trees <= 1:
        raise ValueError("--trees must be greater than one")
    if args.repeats <= 0:
        raise ValueError("--repeats must be positive")
    if args.production_safety_limit <= 0:
        raise ValueError("--production-safety-limit must be positive")
    largest = max(args.sizes)
    if (
        largest > args.production_safety_limit
        and not args.allow_unsafe_production
    ):
        raise ValueError(
            f"largest requested size {largest} exceeds the production safety "
            f"limit {args.production_safety_limit}; pass "
            "--allow-unsafe-production to acknowledge the potentially long "
            "quadratic production run"
        )
    output = args.output.resolve()
    if output.exists() and not args.overwrite:
        raise FileExistsError(
            f"{output} already exists; pass --overwrite or choose --output"
        )


def _geometry(cp: Any, grouping: Any, tree_count: int) -> Any:
    families = cp.zeros(tree_count, dtype=cp.int32)
    steps = cp.ones(tree_count, dtype=cp.float64)
    return grouping.GroupingGeometry(
        ntrees=tree_count,
        primary_tree_index_by_tree=families,
        dm_step_by_tree=steps,
        time_step_samples_by_tree=steps.copy(),
        reference_freq_MHz=300.0,
        full_band_freq_lo_MHz=300.0,
        full_band_freq_hi_MHz=1500.0,
        time_sample_ms=1.0,
        residual_slope_lo_samples_per_dm=0.0,
        residual_slope_hi_samples_per_dm=2.0,
    )


def _candidates(
    cp: Any,
    grouping: Any,
    geometry: Any,
    candidate_count: int,
    scenario: str,
) -> Any:
    index = cp.arange(candidate_count, dtype=cp.int64)
    tree = (index % geometry.ntrees).astype(cp.int32)
    if scenario == "isolated":
        toa = index.astype(cp.float64) * 10.0
    elif scenario == "dense":
        toa = cp.zeros(candidate_count, dtype=cp.float64)
    else:
        raise ValueError(f"unknown scenario {scenario!r}")

    ones = cp.ones(candidate_count, dtype=cp.float64)
    return grouping.GpuDecodedCandidates(
        beam_id=cp.zeros(candidate_count, dtype=cp.int32),
        source_chunk_index=cp.zeros(candidate_count, dtype=cp.int64),
        tree=tree,
        idm=cp.zeros(candidate_count, dtype=cp.int32),
        itime=index.astype(cp.int32),
        snr=(candidate_count + 10.0) - index.astype(cp.float64),
        argmax_token=index.astype(cp.uint32),
        edge_flags=cp.zeros(candidate_count, dtype=cp.uint8),
        dm=cp.zeros(candidate_count, dtype=cp.float64),
        toa_sample_abs=toa,
        width_samp=ones,
        width_ms=ones.copy(),
        freq_lo_MHz=cp.full(candidate_count, 300.0, dtype=cp.float64),
        freq_hi_MHz=cp.full(candidate_count, 1500.0, dtype=cp.float64),
        primary_tree_index=geometry.primary_tree_index_by_tree[tree],
        dm_step=geometry.dm_step_by_tree[tree],
        time_step_samples=geometry.time_step_samples_by_tree[tree],
    )


def _timed_call(
    cp: Any,
    operation: Callable[..., Any],
    candidates: Any,
    geometry: Any,
    config: Any,
) -> tuple[float, Any]:
    stream = cp.cuda.get_current_stream()
    stream.synchronize()
    started = time.perf_counter()
    result = operation(candidates, geometry, config=config)
    stream.synchronize()
    wall_ms = 1000.0 * (time.perf_counter() - started)
    if not math.isfinite(wall_ms) or wall_ms < 0.0:
        raise AssertionError("grouping wall time must be finite and non-negative")
    return wall_ms, result


def _rate(candidate_count: int, wall_ms: float) -> float:
    return math.inf if wall_ms == 0.0 else 1000.0 * candidate_count / wall_ms


def _write_rows(output: Path, rows: list[dict[str, Any]]) -> None:
    output.parent.mkdir(parents=True, exist_ok=True)
    temporary = output.with_suffix(output.suffix + ".tmp")
    with temporary.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=CSV_FIELDS)
        writer.writeheader()
        writer.writerows(rows)
    temporary.replace(output)


def run_benchmark(args: argparse.Namespace) -> list[dict[str, Any]]:
    _validate_args(args)

    import cupy as cp

    import pirate_frb.OfflineCandidateGrouper as grouping
    from peakfinder_tests import gpu_representative_grouper as experimental
    from peakfinder_tests.cpu_candidate_grouper import (
        assert_gpu_grouping_results_equal,
    )

    cp.cuda.Device(args.device).use()
    cp.zeros(1, dtype=cp.uint8).sum().item()
    geometry = _geometry(cp, grouping, args.trees)
    config = grouping.GroupingConfig(
        dm_tolerance_bins=1.5,
        time_padding_bins=1.0,
    )

    warm_count = min(32, min(args.sizes))
    for scenario in ("isolated", "dense"):
        warm_candidates = _candidates(
            cp, grouping, geometry, warm_count, scenario
        )
        production_warm = grouping.group_candidates(
            warm_candidates, geometry, config=config
        )
        experimental_warm = experimental.group_candidates_gpu_representative(
            warm_candidates, geometry, config=config
        )
        cp.cuda.get_current_stream().synchronize()
        assert_gpu_grouping_results_equal(
            cp, production_warm, experimental_warm
        )

    rows: list[dict[str, Any]] = []
    point_ordinal = 0
    for candidate_count in args.sizes:
        for scenario in args.scenarios:
            candidates = _candidates(
                cp, grouping, geometry, candidate_count, scenario
            )
            production_times: list[float] = []
            experimental_times: list[float] = []
            grouped_events = -1

            for repeat in range(args.repeats):
                production_first = (point_ordinal + repeat) % 2 == 0
                if production_first:
                    production_ms, production_result = _timed_call(
                        cp,
                        grouping.group_candidates,
                        candidates,
                        geometry,
                        config,
                    )
                    experimental_ms, experimental_result = _timed_call(
                        cp,
                        experimental.group_candidates_gpu_representative,
                        candidates,
                        geometry,
                        config,
                    )
                    timing_order = "production_then_experimental"
                else:
                    experimental_ms, experimental_result = _timed_call(
                        cp,
                        experimental.group_candidates_gpu_representative,
                        candidates,
                        geometry,
                        config,
                    )
                    production_ms, production_result = _timed_call(
                        cp,
                        grouping.group_candidates,
                        candidates,
                        geometry,
                        config,
                    )
                    timing_order = "experimental_then_production"

                assert_gpu_grouping_results_equal(
                    cp, production_result, experimental_result
                )
                grouped_events = len(experimental_result.events)
                expected_events = (
                    candidate_count
                    if scenario == "isolated"
                    else math.ceil(candidate_count / args.trees)
                )
                if grouped_events != expected_events:
                    raise AssertionError(
                        f"{scenario} produced {grouped_events} events; "
                        f"expected {expected_events}"
                    )

                production_times.append(production_ms)
                experimental_times.append(experimental_ms)
                rows.append(
                    {
                        "scenario": scenario,
                        "candidate_count": candidate_count,
                        "tree_count": args.trees,
                        "partition_count": 1,
                        "largest_partition": candidate_count,
                        "grouped_events": grouped_events,
                        "repeat": repeat,
                        "timing_order": timing_order,
                        "production_wall_ms": production_ms,
                        "experimental_wall_ms": experimental_ms,
                        "production_candidates_per_second": _rate(
                            candidate_count, production_ms
                        ),
                        "experimental_candidates_per_second": _rate(
                            candidate_count, experimental_ms
                        ),
                        "production_over_experimental_speedup": (
                            math.inf
                            if experimental_ms == 0.0
                            else production_ms / experimental_ms
                        ),
                        "exact_parity_verified": True,
                    }
                )
                _write_rows(args.output.resolve(), rows)

            production_median = statistics.median(production_times)
            experimental_median = statistics.median(experimental_times)
            print(
                f"{scenario:8s} N={candidate_count:5d} "
                f"events={grouped_events:5d} "
                f"production={production_median:10.3f} ms "
                f"experimental={experimental_median:10.3f} ms "
                f"speedup={production_median / experimental_median:8.2f}x",
                flush=True,
            )
            point_ordinal += 1

    return rows


def main() -> None:
    args = build_parser().parse_args()
    rows = run_benchmark(args)
    print(
        f"Wrote {len(rows)} exact-parity timing rows to "
        f"{args.output.resolve()}",
        flush=True,
    )


if __name__ == "__main__":
    main()
