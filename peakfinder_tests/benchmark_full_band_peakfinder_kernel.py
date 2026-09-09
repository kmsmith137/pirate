#!/usr/bin/env python3
"""Compare the archived CUDA peak filter with the active full-band search.

The candidate-selection semantics remain those of ``pirate_frb.Peakfinders``.
This experiment only replaces the call which computes the footprint maximum:

* ``cupyx`` is the active ``cupyx.scipy.ndimage.maximum_filter`` backend.
* ``offset_cuda_archived_fp64`` is the archived one-thread-per-output CUDA
  algorithm, fed offsets prepared from the *current* horizon-cropped full-band
  Bowtie.  It retains the original forced-double comparison.
* ``offset_cuda_native`` differs only by accumulating and comparing in the
  input scalar type, which is float32 in the production workload.

Geometry construction, host-to-device upload, offset preparation, and CUDA
JIT compilation are outside every timed interval.  The script first requires
bit-for-bit filter equality for every production tree and both CUDA variants,
then measures the isolated filters and complete streaming
``OfflinePeakExtractor`` workload.
"""

from __future__ import annotations

import argparse
import json
import math
import os
import tempfile
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable, Sequence

import numpy as np

from peakfinder_tests import benchmark_peakfinder_batch_timing as production


SCHEMA_NAME = "pirate-full-band-peakfinder-kernel-comparison"
SCHEMA_VERSION = 3
REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_RESULTS = (
    Path(__file__).resolve().parent
    / "results_full_band_peakfinder_kernel_pirate15"
    / "comparison.json"
)


@dataclass(frozen=True)
class PreparedFootprint:
    """Owning device offset arrays prepared once from a current Bowtie."""

    dm_offsets: Any
    time_offsets: Any
    nactive: int


def prepare_footprint(cp: Any, footprint: Any) -> PreparedFootprint:
    """Convert an odd, centred 2-D Boolean footprint to sorted GPU offsets."""

    host_footprint = np.asarray(cp.asnumpy(footprint), dtype=np.bool_)
    if host_footprint.ndim != 2 or any(size % 2 != 1 for size in host_footprint.shape):
        raise ValueError("full-band footprint must be odd and two-dimensional")
    centre = np.asarray(host_footprint.shape, dtype=np.int64) // 2
    offsets = np.argwhere(host_footprint) - centre
    if offsets.size == 0:
        raise ValueError("full-band footprint must not be empty")
    offsets = np.asarray(sorted(map(tuple, offsets.tolist())), dtype=np.int32)
    return PreparedFootprint(
        dm_offsets=cp.asarray(offsets[:, 0]),
        time_offsets=cp.asarray(offsets[:, 1]),
        nactive=int(offsets.shape[0]),
    )


def make_offset_kernel(cp: Any, *, native_accumulator: bool = False) -> Any:
    """Return the archived filter, optionally without its FP64 promotion."""

    accumulator = "T" if native_accumulator else "double"
    operation = r"""
        const long long flat = (long long)i;
        const int itime = (int)(flat % (long long)ntime);
        const long long beam_dm = flat / (long long)ntime;
        const int idm = (int)(beam_dm % (long long)ndm);
        const long long ibeam = beam_dm / (long long)ndm;
        __ACCUMULATOR__ best = (__ACCUMULATOR__)(-1.0 / 0.0);
        for (int k = 0; k < nactive; ++k) {
            const int neighbor_dm = idm + dm_offsets[k];
            const int neighbor_time = itime + time_offsets[k];
            if ((neighbor_dm < 0) || (neighbor_dm >= ndm) ||
                (neighbor_time < 0) || (neighbor_time >= ntime)) {
                continue;
            }
            const long long neighbor =
                (ibeam * (long long)ndm + (long long)neighbor_dm) *
                (long long)ntime + (long long)neighbor_time;
            const __ACCUMULATOR__ value = (__ACCUMULATOR__)x[neighbor];
            if (value > best) {
                best = value;
            }
        }
        y = (T)best;
        """.replace("__ACCUMULATOR__", accumulator)
    name = (
        "pirate_full_band_offset_maximum_native_benchmark_v1"
        if native_accumulator
        else "pirate_full_band_offset_maximum_benchmark_v1"
    )
    return cp.ElementwiseKernel(
        "raw T x, raw int32 dm_offsets, raw int32 time_offsets, "
        "int32 nactive, int32 ndm, int32 ntime",
        "T y",
        operation,
        name,
    )


def offset_maximum(
        cp: Any, kernel: Any, competitor_map: Any,
        prepared: PreparedFootprint) -> Any:
    """Launch the archived algorithm on a current ``(beam, DM, time)`` map."""

    if competitor_map.ndim != 3 or not competitor_map.flags.c_contiguous:
        raise ValueError("offset kernel requires a contiguous three-dimensional map")
    _, ndm, ntime = competitor_map.shape
    result = kernel(
        competitor_map,
        prepared.dm_offsets,
        prepared.time_offsets,
        np.int32(prepared.nactive),
        np.int32(ndm),
        np.int32(ntime),
        size=competitor_map.size,
    )
    return result.reshape(competitor_map.shape)


class OffsetMaximumAdapter:
    """Match the exact ``maximum_filter`` call made by production extraction."""

    def __init__(self, cp: Any, kernel: Any, geometries: Sequence[Any]):
        self.cp = cp
        self.kernel = kernel
        self.by_pointer = {
            int(geometry.full_band_bowtie.data.ptr): prepare_footprint(
                cp, geometry.full_band_bowtie
            )
            for geometry in geometries
        }

    def __call__(self, values: Any, *, footprint: Any, mode: str, cval: float) -> Any:
        if (mode != "constant" or not math.isinf(float(cval)) or float(cval) > 0.0
                or footprint.ndim != 3 or footprint.shape[0] != 1):
            raise ValueError("unexpected production maximum-filter invocation")
        try:
            prepared = self.by_pointer[int(footprint.data.ptr)]
        except KeyError as exc:
            raise ValueError("maximum-filter footprint was not prepared") from exc
        return offset_maximum(self.cp, self.kernel, values, prepared)


def repeated_source_work_map(cp: Any, source: Any, time_radius: int) -> Any:
    """Reproduce the full ``2*h`` steady-state halo from repeated clean chunks."""

    radius = int(time_radius)
    if radius:
        history = cp.concatenate((source, source), axis=2)
        source = cp.concatenate((history[:, :, -2 * radius:], source), axis=2)
    return cp.ascontiguousarray(source, dtype=cp.float32)


def timing_statistics(samples_ms: Sequence[float]) -> dict[str, Any]:
    values = np.asarray(samples_ms, dtype=np.float64)
    if values.size == 0 or not np.all(np.isfinite(values)) or np.any(values <= 0.0):
        raise ValueError("timing samples must be finite and positive")
    return {
        "samples_ms": [float(value) for value in values],
        "median_ms": float(np.median(values)),
        "minimum_ms": float(np.min(values)),
        "maximum_ms": float(np.max(values)),
        "mean_ms": float(np.mean(values)),
        "standard_deviation_ms": float(np.std(values)),
    }


def cuda_event_samples(
        cp: Any, function: Callable[[], Any], *, warmup: int,
        iterations: int) -> dict[str, Any]:
    """Time asynchronous GPU work without including host synchronization."""

    stream = cp.cuda.get_current_stream()
    output = None
    for _ in range(warmup):
        output = function()
    stream.synchronize()
    del output
    samples = []
    for _ in range(iterations):
        start = cp.cuda.Event()
        stop = cp.cuda.Event()
        start.record(stream)
        output = function()
        stop.record(stream)
        stop.synchronize()
        samples.append(float(cp.cuda.get_elapsed_time(start, stop)))
        del output
    return timing_statistics(samples)


def synchronized_wall_samples(
        stream: Any, function: Callable[[int], Any], *, warmup: int,
        iterations: int) -> dict[str, Any]:
    """Time the production Python/GPU boundary exactly as the batch benchmark."""

    for index in range(warmup):
        function(production.SOURCE_CHUNK_ORIGIN + index)
    stream.synchronize()
    samples = []
    for iteration in range(iterations):
        source = production.SOURCE_CHUNK_ORIGIN + warmup + iteration
        elapsed, count = production.synchronized_wall_time(
            stream, lambda source=source: function(source)
        )
        production._assert_zero_candidates(count, "kernel comparison workload")
        samples.append(elapsed)
    return timing_statistics(samples)


def compare_filters(
        cp: Any, geometries: Sequence[Any], work_maps: Sequence[Any],
        kernels: dict[str, Any], active_maximum_filter: Callable[..., Any],
        *, warmup: int, iterations: int) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    """Prove equality and measure both backends, per tree and collectively."""

    prepared = [prepare_footprint(cp, geometry.full_band_bowtie)
                for geometry in geometries]
    tree_rows = []
    for tree, (geometry, work, offsets) in enumerate(
            zip(geometries, work_maps, prepared)):
        baseline = active_maximum_filter(
            work, footprint=geometry.full_band_bowtie[None, :, :],
            mode="constant", cval=-cp.inf
        )
        for name, kernel in kernels.items():
            candidate = offset_maximum(cp, kernel, work, offsets)
            mismatch_count = int(cp.count_nonzero(baseline != candidate).item())
            if mismatch_count:
                raise AssertionError(
                    f"tree {tree} {name} disagrees in {mismatch_count} cells"
                )
            del candidate
        del baseline
        timings = {"cupyx": cuda_event_samples(
            cp,
            lambda geometry=geometry, work=work: active_maximum_filter(
                work, footprint=geometry.full_band_bowtie[None, :, :],
                mode="constant", cval=-cp.inf
            ),
            warmup=warmup,
            iterations=iterations,
        )}
        timings.update({
            name: cuda_event_samples(
                cp,
                lambda work=work, offsets=offsets, kernel=kernel: offset_maximum(
                    cp, kernel, work, offsets
                ),
                warmup=warmup,
                iterations=iterations,
            )
            for name, kernel in kernels.items()
        })
        row = {
            "tree": tree,
            "work_shape": list(work.shape),
            "footprint_shape": list(geometry.full_band_bowtie.shape),
            "active_footprint_cells": offsets.nactive,
            "outputs_equal": True,
            **timings,
        }
        row["speedup_vs_cupyx"] = {
            name: timings["cupyx"]["median_ms"] / timing["median_ms"]
            for name, timing in timings.items() if name != "cupyx"
        }
        tree_rows.append(row)

    def complete_cupyx() -> tuple[Any, ...]:
        return tuple(
            active_maximum_filter(
                work, footprint=geometry.full_band_bowtie[None, :, :],
                mode="constant", cval=-cp.inf
            )
            for geometry, work in zip(geometries, work_maps)
        )

    timings = {"cupyx": cuda_event_samples(
        cp, complete_cupyx, warmup=warmup, iterations=iterations
    )}
    for name, kernel in kernels.items():
        timings[name] = cuda_event_samples(
            cp,
            lambda kernel=kernel: tuple(
                offset_maximum(cp, kernel, work, offsets)
                for work, offsets in zip(work_maps, prepared)
            ),
            warmup=warmup,
            iterations=iterations,
        )
    result = {
        "clock": "CUDA events on the current stream",
        "scope": "ten full-band maximum filters, including output allocation",
        **timings,
    }
    result["speedup_vs_cupyx"] = {
        name: timings["cupyx"]["median_ms"] / timing["median_ms"]
        for name, timing in timings.items() if name != "cupyx"
    }
    return tree_rows, result


def benchmark_end_to_end(
        cp: Any, peakfinders: Any, geometries: Sequence[Any],
        snr_maps: Sequence[Any], argmax_maps: Sequence[Any], kernels: dict[str, Any],
        *, total_beams: int, beam_batch_size: int, threshold: float,
        warmup: int, iterations: int) -> dict[str, Any]:
    """Measure real streaming extraction with each maximum-filter backend."""

    stream = cp.cuda.get_current_stream()
    partitions = production.partition_beams(total_beams, beam_batch_size)
    active = peakfinders.maximum_filter
    adapters = {
        name: OffsetMaximumAdapter(cp, kernel, geometries)
        for name, kernel in kernels.items()
    }
    results = {}
    try:
        for name, backend in (("cupyx", active), *adapters.items()):
            peakfinders.maximum_filter = backend
            states = production.construct_and_prime_batch_states(
                geometries, partitions, snr_maps, argmax_maps, threshold, stream
            )
            results[name] = synchronized_wall_samples(
                stream,
                lambda source, states=states: production.process_complete_workload(
                    states, source, peakfinders.concatenate_raw_candidates
                ),
                warmup=warmup,
                iterations=iterations,
            )
            del states
            stream.synchronize()
    finally:
        peakfinders.maximum_filter = active
    results.update({
        "clock": "synchronized time.perf_counter wall clock",
        "scope": (
            "production process_chunk and per-batch candidate concatenation for "
            "all trees and beams; setup, halo priming, and JIT excluded"
        ),
        "beam_batch_size": beam_batch_size,
        "n_batches": len(partitions),
    })
    results["speedup_vs_cupyx"] = {
        name: results["cupyx"]["median_ms"] / timing["median_ms"]
        for name, timing in results.items()
        if name not in {"cupyx", "clock", "scope", "beam_batch_size", "n_batches"}
    }
    return results


def atomic_write_json(path: Path, document: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = None
    try:
        with tempfile.NamedTemporaryFile(
                mode="w", encoding="utf-8", prefix=f".{path.name}.",
                suffix=".tmp", dir=path.parent, delete=False) as stream:
            temporary = stream.name
            json.dump(document, stream, indent=2, sort_keys=False, allow_nan=False)
            stream.write("\n")
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary, path)
        temporary = None
    finally:
        if temporary is not None:
            try:
                os.unlink(temporary)
            except FileNotFoundError:
                pass


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=production.DEFAULT_CONFIG)
    parser.add_argument("--results", type=Path, default=DEFAULT_RESULTS)
    parser.add_argument("--total-beams", type=int, default=60)
    parser.add_argument("--beam-batch-size", type=int, default=60)
    parser.add_argument("--dm-reach", type=int, default=8)
    parser.add_argument("--waist-bins", type=int, default=1)
    parser.add_argument("--threshold", type=float, default=10.0)
    parser.add_argument("--warmup", type=int, default=2)
    parser.add_argument("--iterations", type=int, default=7)
    parser.add_argument("--device", type=int, default=0)
    parser.add_argument("--base-seed", type=int, default=20260904)
    parser.add_argument("--overwrite", action="store_true")
    return parser


def run(args: argparse.Namespace) -> dict[str, Any]:
    if args.results.exists() and not args.overwrite:
        raise FileExistsError(f"result exists; pass --overwrite: {args.results}")
    if min(args.total_beams, args.beam_batch_size, args.dm_reach,
           args.iterations) <= 0 or args.warmup < 0:
        raise ValueError("beam counts, DM reach, and iterations must be positive")

    import cupy as cp
    from cupyx.scipy.ndimage import maximum_filter as active_maximum_filter
    import pirate_frb.Peakfinders as peakfinders

    with cp.cuda.Device(args.device):
        bundle = production.load_authoritative_plan(args.config)
        inputs = production.generate_clean_inputs(
            bundle.specs, args.total_beams, args.base_seed, args.threshold
        )
        production.validate_tokens_with_plan(bundle.plan, inputs)
        snr_maps, argmax_maps = production.upload_inputs(cp, inputs)
        geometries_by_reach, diagnostics = production.build_geometries(
            cp, bundle.plan, bundle.specs, (args.dm_reach,), args.waist_bins
        )
        geometries = geometries_by_reach[args.dm_reach]
        work_maps = tuple(
            repeated_source_work_map(cp, source, geometry.time_radius)
            for source, geometry in zip(snr_maps, geometries)
        )
        kernels = {
            "offset_cuda_archived_fp64": make_offset_kernel(cp),
            "offset_cuda_native": make_offset_kernel(cp, native_accumulator=True),
        }
        tree_rows, isolated = compare_filters(
            cp, geometries, work_maps, kernels, active_maximum_filter,
            warmup=args.warmup, iterations=args.iterations
        )
        end_to_end = benchmark_end_to_end(
            cp, peakfinders, geometries, snr_maps, argmax_maps, kernels,
            total_beams=args.total_beams,
            beam_batch_size=args.beam_batch_size,
            threshold=args.threshold,
            warmup=args.warmup,
            iterations=args.iterations,
        )
        gpu = production.gpu_information(cp, args.device)
        software = production.software_information(cp)

    for backend in ("cupyx", *kernels):
        median = end_to_end[backend]["median_ms"]
        end_to_end[backend]["fraction_of_chunk"] = median / bundle.chunk_duration_ms
        end_to_end[backend]["realtime_capacity_beams"] = (
            bundle.chunk_duration_ms * args.total_beams / median
        )
    document = {
        "schema_name": SCHEMA_NAME,
        "schema_version": SCHEMA_VERSION,
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "configuration": {
            "config": str(Path(args.config).resolve()),
            "time_sample_ms": float(bundle.config.time_sample_ms),
            "chunk_duration_ms": bundle.chunk_duration_ms,
            "total_beams": args.total_beams,
            "beam_batch_size": args.beam_batch_size,
            "dm_reach": args.dm_reach,
            "waist_bins": args.waist_bins,
            "threshold": args.threshold,
            "warmup": args.warmup,
            "iterations": args.iterations,
            "base_seed": args.base_seed,
        },
        "producer": {
            "dcores": list(bundle.dcores),
            "argmax_encoding": bundle.argmax_encoding,
            "plan_yaml": bundle.producer_plan_yaml,
            "trees": [dict(spec.__dict__) for spec in bundle.specs],
        },
        "gpu": gpu,
        "software": software,
        "kernel_origin": str(
            REPOSITORY_ROOT
            / "old_stuff/pirate_v1_2_advanced/pirate_frb/Peakfinders.py"
        ),
        "adaptation": {
            "geometry": (
                "active offsets are prepared from each current full-band, "
                "one-chunk-horizon-cropped Bowtie"
            ),
            "offset_cuda_archived_fp64": (
                "archived double accumulator and double comparison"
            ),
            "offset_cuda_native": (
                "minimal diagnostic change: accumulator and comparison retain "
                "the input scalar type (float32 for the production workload)"
            ),
        },
        "all_filter_outputs_equal": all(row["outputs_equal"] for row in tree_rows),
        "trees": tree_rows,
        "isolated_filter_workload": isolated,
        "production_streaming_workload": end_to_end,
        "geometry_active_cells": [
            diagnostics[(args.dm_reach, tree)].active_footprint_cells
            for tree in range(len(geometries))
        ],
    }
    atomic_write_json(args.results, document)
    return document


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    result = run(args)
    isolated = result["isolated_filter_workload"]
    streaming = result["production_streaming_workload"]
    print(f"filter outputs equal: {result['all_filter_outputs_equal']}")
    print(
        "isolated ten-tree filters: "
        f"cupyx={isolated['cupyx']['median_ms']:.3f} ms, "
        f"archived_fp64={isolated['offset_cuda_archived_fp64']['median_ms']:.3f} ms, "
        f"native={isolated['offset_cuda_native']['median_ms']:.3f} ms"
    )
    print(
        "production streaming workload: "
        f"cupyx={streaming['cupyx']['median_ms']:.3f} ms, "
        f"archived_fp64={streaming['offset_cuda_archived_fp64']['median_ms']:.3f} ms, "
        f"native={streaming['offset_cuda_native']['median_ms']:.3f} ms"
    )
    print(f"results: {Path(args.results).resolve()}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
