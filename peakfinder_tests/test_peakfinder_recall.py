"""Single-burst recall benchmark for the production full-band peakfinder."""

import argparse
import time
from pathlib import Path

import cupy as cp
import numpy as np

from .experiment_common import (
    DEFAULT_BANDWIDTH_MIN_MHZ,
    DEFAULT_DM_RANGE,
    DEFAULT_SNR_RANGE,
    DEFAULT_WIDTH_RANGE_MS,
    MATCHING_CONVENTION,
    REALIZED_CHANNEL_CONVENTION,
    RNG_NOTE,
    SCHEMA_VERSION,
    WIDTH_SEMANTICS,
    atomic_write_csv,
    candidate_values,
    decoded_timestamp_summary,
    file_identity,
    fine_toa_tolerance_s,
    inband_match_frequency_MHz,
    initialize_checkpoint,
    make_unit_rng,
    match_one_inband,
    plan_summary,
    prepare_plan,
    run_simulated_peakfinders,
    sample_recall_burst_parameters,
    sample_safe_toa_placement,
    update_metadata,
    validate_dm_sampling_range,
    validate_matched_cardinality_containment,
    validate_resume_metadata,
    validate_resumed_trial_count,
)
from .peakfinders import BENCHMARK_METHODS, build_peakfinder_geometry

DEFAULT_PARAMETER_SEED = 20260814
UNIT_SEED_DERIVATION = (
    "uint64 little-endian prefix of SHA256(JSON([parameter_seed, "
    "'recall', trial]))"
)
CANDIDATE_AUDIT_FIELDS = (
    "idm", "itime", "time_chunk_index", "tree", "snr", "dm",
    "toa_ref_s", "toa_inband_s", "toa_residual_ms", "argmax_token",
    "fmin", "fmax", "freq_lo_MHz", "freq_hi_MHz",
)

FIELDS = [
    "schema_version", "trial", "method", "map_id", "parameter_seed",
    "trial_seed", "tree", "estimated_idm", "lower_dm_margin_bins",
    "upper_dm_margin_bins", "injected_dm", "injected_snr",
    "injected_width_ms", "full_band_freq_lo_MHz",
    "full_band_freq_hi_MHz", "requested_bandwidth_MHz",
    "requested_freq_lo_MHz", "requested_freq_hi_MHz",
    "active_channel_indices", "active_channel_count",
    "effective_active_freq_lo_MHz", "effective_active_freq_hi_MHz",
    "lower_boundary_on_channel_edge", "upper_boundary_on_channel_edge",
    "injected_toa_s", "injected_toa_native_samples",
    "output_map_index", "output_bin_index", "output_bin_phase",
    "full_band_time_radius_bins", "full_band_time_radius_ms",
    "safe_first_output_bin_count", "safe_first_output_bin_min",
    "safe_first_output_bin_max", "pulse_support_start", "pulse_support_end",
    "detected", "matched_candidate_index", "ncandidates",
    "candidate_set_containment", "unit_wall_time_s",
    "reference_frequency_MHz", "matching_frequency_MHz",
    "injected_toa_inband_s", "toa_tolerance_ms", "matching_convention",
] + [f"candidate_{field}" for field in CANDIDATE_AUDIT_FIELDS]


def recall_unit_key(row):
    return int(row["trial"])


def _parser():
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument("--config", default="configs/dedispersion/chord_sb2.yml")
    parser.add_argument("--metadata", default="configs/xengine_metadata.yml")
    parser.add_argument("--results-dir", default="peakfinder_tests/results_final")
    parser.add_argument("--trials", type=int, default=750)
    parser.add_argument("--parameter-seed", type=int, default=DEFAULT_PARAMETER_SEED)
    parser.add_argument("--threshold", type=float, default=10.0)
    parser.add_argument("--toa-tolerance-ms", type=float)
    parser.add_argument("--nchunks", type=int, default=4)
    parser.add_argument("--ntime", type=int, default=2048)
    parser.add_argument("--device", type=int, default=0)
    parser.add_argument("--dm-reach", type=int, default=8)
    parser.add_argument("--waist-bins", type=int, default=1)
    mode = parser.add_mutually_exclusive_group()
    mode.add_argument("--resume", action="store_true")
    mode.add_argument("--overwrite", action="store_true")
    return parser


def _prefixed_candidate(values):
    return {f"candidate_{field}": values[field] for field in CANDIDATE_AUDIT_FIELDS}


def _row_bool(value):
    return value is True or str(value).strip().lower() in {"true", "1"}


def main(argv=None):
    args = _parser().parse_args(argv)
    if args.trials < 1:
        raise ValueError("--trials must be positive")
    if args.parameter_seed < 0:
        raise ValueError("--parameter-seed must be non-negative")
    if args.nchunks < 1 or args.ntime < 1:
        raise ValueError("--nchunks and --ntime must be positive")
    if args.device < 0:
        raise ValueError("--device must be non-negative")
    if args.dm_reach < 0 or args.waist_bins < 0:
        raise ValueError("--dm-reach and --waist-bins must be non-negative")
    if not np.isfinite(args.threshold):
        raise ValueError("--threshold must be finite")
    if args.toa_tolerance_ms is not None and (
        not np.isfinite(args.toa_tolerance_ms) or args.toa_tolerance_ms <= 0
    ):
        raise ValueError("--toa-tolerance-ms must be finite and positive")

    config, xmd, host_plan = prepare_plan(args.config, args.metadata)
    if args.ntime != int(host_plan.nt_in):
        raise ValueError(f"--ntime={args.ntime} must match plan nt_in={host_plan.nt_in}")
    time_sample_s = float(config.time_sample_ms) / 1.0e3
    reference_frequency_MHz = float(np.asarray(xmd.get_channel_freq_edges())[0])
    dm_range = validate_dm_sampling_range(
        host_plan, dm_reach=args.dm_reach,
        dm_min=DEFAULT_DM_RANGE[0], dm_max=DEFAULT_DM_RANGE[1],
    )
    intended_tree = int(dm_range["tree"])
    with cp.cuda.Device(args.device):
        geometry = build_peakfinder_geometry(
            host_plan, intended_tree, time_sample_s=time_sample_s,
            nt_in=host_plan.nt_in,
            reference_freq_mhz=reference_frequency_MHz,
            dm_reach=args.dm_reach, waist_bins=args.waist_bins,
        )
        geometry_diagnostics = geometry.diagnostics()

    scientific_parameters = {
        "schema_version": SCHEMA_VERSION,
        "methods": list(BENCHMARK_METHODS),
        "config": file_identity(args.config),
        "xengine_metadata": file_identity(args.metadata),
        "threshold": args.threshold,
        "dm_reach": args.dm_reach,
        "waist_bins": args.waist_bins,
        "nchunks": args.nchunks,
        "ntime": args.ntime,
        "parameter_seed": args.parameter_seed,
        "unit_seed_derivation": UNIT_SEED_DERIVATION,
        "intended_tree": intended_tree,
        "dm_distribution": {
            "name": "uniform", "low_pc_cm3": DEFAULT_DM_RANGE[0],
            "high_pc_cm3": DEFAULT_DM_RANGE[1], "high_is_exclusive": True,
        },
        "spectral_bandwidth_distribution": {
            "name": "uniform", "low_MHz": DEFAULT_BANDWIDTH_MIN_MHZ,
            "high": "full observing bandwidth", "high_is_exclusive": True,
        },
        "spectral_lower_edge_distribution": (
            "Uniform(full_band_lo, full_band_hi - requested_bandwidth)"
        ),
        "width_distribution_ms": {
            "name": "uniform", "low": DEFAULT_WIDTH_RANGE_MS[0],
            "high": DEFAULT_WIDTH_RANGE_MS[1], "high_is_exclusive": True,
        },
        "snr_distribution": {
            "name": "uniform", "low": DEFAULT_SNR_RANGE[0],
            "high": DEFAULT_SNR_RANGE[1], "high_is_exclusive": True,
        },
        "width_semantics": WIDTH_SEMANTICS,
        "toa_sampling": (
            "Uniform feasible output bin and uniform sub-bin phase after simulator "
            "support and complete Full-band-footprint validation"
        ),
        "toa_tolerance_override_ms": args.toa_tolerance_ms,
        "matching_convention": MATCHING_CONVENTION,
        "realized_channel_convention": REALIZED_CHANNEL_CONVENTION,
    }

    results_dir = Path(args.results_dir)
    output = results_dir / "recall.csv"
    metadata_path = results_dir / "metadata.yaml"
    previous = None
    if args.resume:
        previous = validate_resume_metadata(
            metadata_path, "recall", scientific_parameters)
        validate_resumed_trial_count(previous, args.trials)

    rows, completed = initialize_checkpoint(
        output, FIELDS, resume=args.resume, overwrite=args.overwrite,
        unit_key=recall_unit_key, row_key=lambda row: row["method"],
        expected_row_keys=BENCHMARK_METHODS,
    )
    intended = set(range(args.trials))

    def checkpoint_metadata():
        completed_rows = [row for row in rows if recall_unit_key(row) in intended]
        wall_times = {
            recall_unit_key(row): float(row["unit_wall_time_s"])
            for row in completed_rows
        }
        detected = {
            method: sum(
                _row_bool(row["detected"])
                for row in completed_rows if row["method"] == method
            )
            for method in BENCHMARK_METHODS
        }
        update_metadata(metadata_path, "recall", {
            "scientific_parameters": scientific_parameters,
            "intended_trials": args.trials,
            "intended_logical_units": len(intended),
            "completed_logical_units": len(completed & intended),
            "completed_trial_indices": sorted(completed & intended),
            "complete": intended <= completed,
            "recorded_rows": len(completed_rows),
            "detected_by_method": detected,
            "parameter_seed": args.parameter_seed,
            "intended_tree": intended_tree,
            "dm_sampling_validation": dm_range,
            "reference_frequency_MHz": reference_frequency_MHz,
            "width_semantics": WIDTH_SEMANTICS,
            "matching_convention": MATCHING_CONVENTION,
            "realized_channel_convention": REALIZED_CHANNEL_CONVENTION,
            "decoded_timestamp": decoded_timestamp_summary(
                host_plan, intended_tree, time_sample_s),
            "candidate_set_containment_required": (
                "The result mapping contains exactly the retained full-band method"
            ),
            "candidate_set_containment_passed": all(
                _row_bool(row["candidate_set_containment"])
                for row in completed_rows
            ),
            "methods_received_identical_maps": True,
            "unit_wall_time_definition": (
                "One real simulation/dedispersion and the full-band peakfinder; "
                "safe-placement preview is excluded."
            ),
            "median_completed_unit_wall_time_s": (
                float(np.median(list(wall_times.values()))) if wall_times else None
            ),
            "plan": plan_summary(host_plan, time_sample_s),
            "footprints": [geometry_diagnostics],
            "rng_limitation": RNG_NOTE,
        }, device=args.device)

    checkpoint_metadata()
    for trial in range(args.trials):
        if trial in completed:
            print(f"trial={trial}: already complete, skipping")
            continue

        rng, trial_seed = make_unit_rng(args.parameter_seed, "recall", trial)
        injected = sample_recall_burst_parameters(
            rng, host_plan, xmd, dm_reach=args.dm_reach)
        if int(injected["tree"]) != intended_tree:
            raise RuntimeError("sampled DM belongs to an unexpected plan tree")
        placement = sample_safe_toa_placement(
            rng,
            metadata_path=args.metadata,
            nchunks=args.nchunks,
            ntime=args.ntime,
            dm=injected["dm"],
            snr=injected["snr"],
            width_ms=injected["width_ms"],
            freq_lo_MHz=injected["requested_freq_lo_MHz"],
            freq_hi_MHz=injected["requested_freq_hi_MHz"],
            geometry=geometry,
        )
        injected_toa_s = float(placement["toas_s"][0])
        map_id = f"recall-{trial:04d}"

        start = time.perf_counter()
        decoded, info = run_simulated_peakfinders(
            metadata_path=args.metadata,
            config_path=args.config,
            nchunks=args.nchunks,
            ntime=args.ntime,
            dm=injected["dm"],
            snr=injected["snr"],
            width_ms=injected["width_ms"],
            toas=[injected_toa_s],
            threshold=args.threshold,
            device=args.device,
            dm_reach=args.dm_reach,
            waist_bins=args.waist_bins,
            freq_lo_MHz=injected["requested_freq_lo_MHz"],
            freq_hi_MHz=injected["requested_freq_hi_MHz"],
            expected_tree=intended_tree,
            output_map_index=placement["output_map_index"],
        )
        wall_time_s = time.perf_counter() - start
        if info["tree"] != intended_tree or not info["methods_received_identical_maps"]:
            raise RuntimeError("simulation did not use the intended tree/shared maps")

        matching_frequency_MHz = inband_match_frequency_MHz(
            injected["requested_freq_lo_MHz"], injected["requested_freq_hi_MHz"])
        toa_tolerance_s, _ = fine_toa_tolerance_s(
            time_sample_s, injected["width_ms"] / 1.0e3,
            args.toa_tolerance_ms,
        )
        assignments = {}
        candidate_toas = {}
        injected_toas_inband = {}
        for method in BENCHMARK_METHODS:
            index, injected_inband, recovered_inband = match_one_inband(
                decoded[method],
                injected_dm=injected["dm"],
                injected_toa_ref_s=injected_toa_s,
                reference_freq_MHz=reference_frequency_MHz,
                match_freq_MHz=matching_frequency_MHz,
                toa_tolerance_s=toa_tolerance_s,
            )
            assignments[method] = [index]
            candidate_toas[method] = recovered_inband
            injected_toas_inband[method] = injected_inband
        validate_matched_cardinality_containment(
            assignments, context=f"recall trial={trial}")
        if len(set(injected_toas_inband.values())) != 1:
            raise RuntimeError("methods disagree on the injected in-band TOA")

        support = placement["pulse_supports"][0]
        unit_rows = []
        for method in BENCHMARK_METHODS:
            index = assignments[method][0]
            recovered = candidate_values(
                decoded[method], index,
                candidate_toas_match=candidate_toas[method],
                injected_dm=injected["dm"],
                injected_toa_match_s=injected_toas_inband[method],
                dm_step=info["dm_step"],
            )
            unit_rows.append({
                "schema_version": SCHEMA_VERSION,
                "trial": trial,
                "method": method,
                "map_id": map_id,
                "parameter_seed": args.parameter_seed,
                "trial_seed": trial_seed,
                "tree": intended_tree,
                "estimated_idm": injected["estimated_idm"],
                "lower_dm_margin_bins": injected["lower_dm_margin_bins"],
                "upper_dm_margin_bins": injected["upper_dm_margin_bins"],
                "injected_dm": injected["dm"],
                "injected_snr": injected["snr"],
                "injected_width_ms": injected["width_ms"],
                "full_band_freq_lo_MHz": injected["full_band_freq_lo_MHz"],
                "full_band_freq_hi_MHz": injected["full_band_freq_hi_MHz"],
                "requested_bandwidth_MHz": injected["requested_bandwidth_MHz"],
                "requested_freq_lo_MHz": injected["requested_freq_lo_MHz"],
                "requested_freq_hi_MHz": injected["requested_freq_hi_MHz"],
                "active_channel_indices": injected["active_channel_indices_compact"],
                "active_channel_count": injected["active_channel_count"],
                "effective_active_freq_lo_MHz": injected["effective_freq_lo_MHz"],
                "effective_active_freq_hi_MHz": injected["effective_freq_hi_MHz"],
                "lower_boundary_on_channel_edge": injected["lower_boundary_on_channel_edge"],
                "upper_boundary_on_channel_edge": injected["upper_boundary_on_channel_edge"],
                "injected_toa_s": injected_toa_s,
                "injected_toa_native_samples": info["preflight"]["toa_native_samples"][0],
                "output_map_index": placement["output_map_index"],
                "output_bin_index": placement["output_bin_indices"][0],
                "output_bin_phase": placement["output_bin_phases"][0],
                "full_band_time_radius_bins": placement["full_band_time_radius_bins"],
                "full_band_time_radius_ms": placement["full_band_time_radius_ms"],
                "safe_first_output_bin_count": placement["safe_first_output_bin_count"],
                "safe_first_output_bin_min": placement["safe_first_output_bin_min"],
                "safe_first_output_bin_max": placement["safe_first_output_bin_max"],
                "pulse_support_start": support[0],
                "pulse_support_end": support[1],
                "detected": index is not None,
                "matched_candidate_index": "" if index is None else int(index),
                "ncandidates": len(decoded[method]["snr"]),
                "candidate_set_containment": info["candidate_set_containment"],
                "unit_wall_time_s": wall_time_s,
                "reference_frequency_MHz": reference_frequency_MHz,
                "matching_frequency_MHz": matching_frequency_MHz,
                "injected_toa_inband_s": injected_toas_inband[method],
                "toa_tolerance_ms": 1.0e3 * toa_tolerance_s,
                "matching_convention": MATCHING_CONVENTION,
                **_prefixed_candidate(recovered),
            })

        rows.extend(unit_rows)
        completed.add(trial)
        atomic_write_csv(output, FIELDS, rows)
        checkpoint_metadata()
        print(f"trial={trial} seed={trial_seed}: checkpointed")

    checkpoint_metadata()
    print(f"{output}: {len(completed & intended)}/{len(intended)} trials complete")


if __name__ == "__main__":
    main()
