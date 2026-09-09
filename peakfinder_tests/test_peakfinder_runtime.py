"""CUDA-event runtime benchmark for the production full-band peak finder."""

import argparse
from pathlib import Path

import cupy as cp
import numpy as np

from .producer_metadata import ARGMAX_ENCODING

from .experiment_common import (
    SCHEMA_VERSION,
    atomic_write_csv,
    file_identity,
    initialize_checkpoint,
    plan_summary,
    prepare_plan,
    update_metadata,
    validate_resume_metadata,
)
from .peakfinders import (
    BENCHMARK_METHODS,
    build_peakfinder_geometry,
    run_peakfinder,
    validate_candidate_set_containment,
)

DEFAULT_NDM = [128, 256, 512, 1024, 2048, 4096]
DEFAULT_NT = [16, 32, 64, 128, 256, 512, 1024, 2048, 4096]
SNR_DTYPE = "float16"
ARGMAX_DTYPE = "uint32"
SEED_DERIVATION = "numpy.SeedSequence([runtime_map_seed, ndm, nt]) -> uint32"

FIELDS = [
    "schema_version", "method", "geometry_tree", "ndm", "nt", "npixels",
    "map_seed", "snr_dtype", "argmax_dtype", "threshold", "dm_reach",
    "waist_bins", "warmup_calls", "timed_calls", "iteration", "gpu_time_ms",
    "mean_gpu_time_ms", "median_gpu_time_ms", "std_gpu_time_ms",
]


def _parser():
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument("--config", default="configs/dedispersion/chord_sb2.yml")
    parser.add_argument("--metadata", default="configs/xengine_metadata.yml")
    parser.add_argument("--results-dir", default="peakfinder_tests/results_final_pirate15")
    parser.add_argument("--threshold", type=float, default=10.0)
    parser.add_argument("--ndm", type=int, nargs="+", default=DEFAULT_NDM)
    parser.add_argument("--nt", type=int, nargs="+", default=DEFAULT_NT)
    parser.add_argument("--geometry-tree", type=int, default=0)
    parser.add_argument("--warmup", type=int, default=5)
    parser.add_argument("--iterations", type=int, default=50)
    parser.add_argument("--device", type=int, default=0)
    parser.add_argument("--runtime-map-seed", type=int, default=12345)
    parser.add_argument("--dm-reach", type=int, default=8)
    parser.add_argument("--waist-bins", type=int, default=1)
    mode = parser.add_mutually_exclusive_group()
    mode.add_argument("--resume", action="store_true")
    mode.add_argument("--overwrite", action="store_true")
    return parser


def _validate_dimension_grid(values, option):
    if not values or any(value < 1 for value in values):
        raise ValueError(f"{option} must contain positive integers")
    if len(set(values)) != len(values):
        raise ValueError(f"{option} must not contain duplicates")


def _shape_seed(base_seed, ndm, nt):
    """Derive an order-independent, reproducible uint32 seed for one map shape."""
    return int(np.random.SeedSequence(
        [int(base_seed), int(ndm), int(nt)]
    ).generate_state(1, dtype=np.uint32)[0])


def _make_runtime_maps(ndm, nt, nmultiplets, map_seed):
    """Generate one shared synthetic input pair outside the timed region."""
    if not 1 <= nmultiplets <= (1 << 16):
        raise ValueError(
            f"geometry has {nmultiplets} multiplets; uint32 tokens support 1..65536"
        )
    rng = cp.random.RandomState(map_seed)
    snr_map = rng.standard_normal((ndm, nt), dtype=cp.float32).astype(cp.float16)
    multiplet = rng.randint(0, nmultiplets, size=(ndm, nt), dtype=cp.uint32)
    argmax_map = multiplet << cp.uint32(16)
    return snr_map, argmax_map


def _timing_summary(measurements):
    values = np.asarray(measurements, dtype=np.float64)
    if values.ndim != 1 or values.size == 0 or not np.all(np.isfinite(values)):
        raise ValueError("timing measurements must be a non-empty finite vector")
    if np.any(values < 0):
        raise ValueError("timing measurements must be non-negative")
    return {
        "mean_gpu_time_ms": float(np.mean(values)),
        "median_gpu_time_ms": float(np.median(values)),
        "std_gpu_time_ms": float(np.std(values, ddof=0)),
    }


def _time_method(method, snr_map, argmax_map, geometry, threshold, warmup,
                 iterations):
    for _ in range(warmup):
        run_peakfinder(method, snr_map, argmax_map, geometry, threshold)
    cp.cuda.get_current_stream().synchronize()

    measurements = []
    candidates = None
    for _ in range(iterations):
        start, stop = cp.cuda.Event(), cp.cuda.Event()
        start.record()
        candidates = run_peakfinder(
            method, snr_map, argmax_map, geometry, threshold)
        stop.record()
        stop.synchronize()
        gpu_time_ms = float(cp.cuda.get_elapsed_time(start, stop))
        if not np.isfinite(gpu_time_ms) or gpu_time_ms < 0:
            raise RuntimeError(f"invalid CUDA-event time {gpu_time_ms!r} ms")
        measurements.append(gpu_time_ms)
    return measurements, candidates


def main(argv=None):
    args = _parser().parse_args(argv)
    _validate_dimension_grid(args.ndm, "--ndm")
    _validate_dimension_grid(args.nt, "--nt")
    if args.warmup < 0 or args.iterations < 1:
        raise ValueError("--warmup must be non-negative and --iterations positive")
    if args.runtime_map_seed < 0:
        raise ValueError("--runtime-map-seed must be non-negative")
    if args.dm_reach < 0 or args.waist_bins < 0:
        raise ValueError("--dm-reach and --waist-bins must be non-negative")
    if args.device < 0:
        raise ValueError("--device must be non-negative")
    if not np.isfinite(args.threshold):
        raise ValueError("--threshold must be finite")

    config, xmd, plan, dcores = prepare_plan(
        args.config, args.metadata, cuda_device_id=args.device)
    if not 0 <= args.geometry_tree < int(plan.ntrees):
        raise ValueError(f"--geometry-tree must be in [0, {plan.ntrees})")
    time_sample_s = float(config.time_sample_ms) / 1.0e3
    reference_frequency = float(np.asarray(xmd.get_channel_freq_edges())[0])
    shapes = [(ndm, nt) for ndm in args.ndm for nt in args.nt]
    shape_seeds = {
        shape: _shape_seed(args.runtime_map_seed, *shape) for shape in shapes
    }

    scientific_parameters = {
        "dcores": list(dcores),
        "argmax_encoding": ARGMAX_ENCODING,
        "schema_version": SCHEMA_VERSION,
        "methods": list(BENCHMARK_METHODS),
        "config": file_identity(args.config),
        "xengine_metadata": file_identity(args.metadata),
        "threshold": args.threshold,
        "ndm_values": list(args.ndm),
        "nt_values": list(args.nt),
        "geometry_tree": args.geometry_tree,
        "warmup": args.warmup,
        "iterations": args.iterations,
        "runtime_map_seed": args.runtime_map_seed,
        "map_seed_derivation": SEED_DERIVATION,
        "dm_reach": args.dm_reach,
        "waist_bins": args.waist_bins,
        "snr_dtype": SNR_DTYPE,
        "argmax_dtype": ARGMAX_DTYPE,
        "ntime": int(plan.nt_in),
    }
    results_dir = Path(args.results_dir)
    output = results_dir / "runtime.csv"
    metadata_path = results_dir / "metadata.yaml"
    if args.resume:
        validate_resume_metadata(metadata_path, "runtime", scientific_parameters)

    expected_rows = {
        (method, iteration)
        for method in BENCHMARK_METHODS
        for iteration in range(args.iterations)
    }
    rows, completed = initialize_checkpoint(
        output, FIELDS, resume=args.resume, overwrite=args.overwrite,
        unit_key=lambda row: (int(row["ndm"]), int(row["nt"])),
        row_key=lambda row: (row["method"], int(row["iteration"])),
        expected_row_keys=expected_rows,
    )
    intended = set(shapes)
    geometry_metadata = []

    def checkpoint_metadata():
        update_metadata(metadata_path, "runtime", {
            "scientific_parameters": scientific_parameters,
            "intended_logical_units": len(intended),
            "completed_logical_units": len(completed & intended),
            "completed_unit_keys": [
                {"ndm": ndm, "nt": nt, "map_seed": shape_seeds[(ndm, nt)]}
                for ndm, nt in sorted(completed & intended)
            ],
            "complete": intended <= completed,
            "expected_measurements": (
                len(intended) * len(BENCHMARK_METHODS) * args.iterations
            ),
            "recorded_measurements": len(rows),
            "runtime_map_seed": args.runtime_map_seed,
            "runtime_map_seed_scope": (
                "Fully reproducible synthetic CuPy maps; each shape has an "
                "order-independent seed derived from the base seed and dimensions."
            ),
            "shape_seeds": [
                {"ndm": ndm, "nt": nt, "map_seed": shape_seeds[(ndm, nt)]}
                for ndm, nt in shapes
            ],
            "timing_scope": (
                "One complete peak-finder invocation on one two-dimensional map, "
                "including per-map token selection, float16-to-float32 promotion, "
                "maximum filtering, thresholding, and candidate extraction. Input-map "
                "generation, reusable geometry construction, summary statistics, and "
                "single-method schema validation are excluded. Times are CUDA-event "
                "milliseconds from cupy.cuda.get_elapsed_time after stop synchronization."
            ),
            "timing_statistics": (
                "Per-shape and per-method arithmetic mean, median, and population "
                "standard deviation (numpy.std with ddof=0), repeated on every raw row."
            ),
            "input_reuse": (
                "Every timed invocation receives the same snr_map and argmax_map "
                "objects within each logical (ndm, nt) unit."
            ),
            "method_schema_check": (
                "The result mapping is validated against the sole retained "
                "full_band_bowtie method after timing each logical unit."
            ),
            "plan": plan_summary(plan, time_sample_s, dcores=dcores),
            "footprints": geometry_metadata,
        }, device=args.device)

    checkpoint_metadata()
    with cp.cuda.Device(args.device):
        geometry = build_peakfinder_geometry(
            plan, args.geometry_tree, dcores=dcores, time_sample_s=time_sample_s, nt_in=plan.nt_in,
            reference_freq_mhz=reference_frequency,
            dm_reach=args.dm_reach, waist_bins=args.waist_bins,
        )
        geometry_metadata.append(geometry.diagnostics())
        nmultiplets = int(plan.trees[args.geometry_tree].frequency_subbands.M)
        checkpoint_metadata()

        for ndm, nt in shapes:
            unit = (ndm, nt)
            if unit in completed:
                print(f"shape=({ndm}, {nt}): already complete, skipping")
                continue

            map_seed = shape_seeds[unit]
            snr_map, argmax_map = _make_runtime_maps(
                ndm, nt, nmultiplets, map_seed)
            cp.cuda.get_current_stream().synchronize()

            measurements_by_method = {}
            candidates_by_method = {}
            for method in BENCHMARK_METHODS:
                measurements, candidates = _time_method(
                    method, snr_map, argmax_map, geometry, args.threshold,
                    args.warmup, args.iterations,
                )
                measurements_by_method[method] = measurements
                candidates_by_method[method] = candidates

            validate_candidate_set_containment(
                candidates_by_method, context=f"runtime shape ({ndm}, {nt})"
            )

            unit_rows = []
            for method in BENCHMARK_METHODS:
                summary = _timing_summary(measurements_by_method[method])
                for iteration, gpu_time_ms in enumerate(measurements_by_method[method]):
                    unit_rows.append({
                        "schema_version": SCHEMA_VERSION,
                        "method": method,
                        "geometry_tree": args.geometry_tree,
                        "ndm": ndm,
                        "nt": nt,
                        "npixels": ndm * nt,
                        "map_seed": map_seed,
                        "snr_dtype": SNR_DTYPE,
                        "argmax_dtype": ARGMAX_DTYPE,
                        "threshold": args.threshold,
                        "dm_reach": args.dm_reach,
                        "waist_bins": args.waist_bins,
                        "warmup_calls": args.warmup,
                        "timed_calls": args.iterations,
                        "iteration": iteration,
                        "gpu_time_ms": gpu_time_ms,
                        **summary,
                    })

            rows.extend(unit_rows)
            completed.add(unit)
            atomic_write_csv(output, FIELDS, rows)
            checkpoint_metadata()
            print(f"shape=({ndm}, {nt}) map_seed={map_seed}: checkpointed")

    checkpoint_metadata()
    print(f"{output}: {len(completed & intended)}/{len(intended)} logical units complete")


if __name__ == "__main__":
    main()
