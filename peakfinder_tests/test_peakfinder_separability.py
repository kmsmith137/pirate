"""Two-burst separability benchmark for the production full-band peakfinder."""

import argparse
import time
from pathlib import Path

import cupy as cp
import numpy as np

from .producer_metadata import ARGMAX_ENCODING

from .experiment_common import (
    DEFAULT_BANDWIDTH_MIN_MHZ,
    DEFAULT_DM_RANGE,
    MATCHING_CONVENTION,
    REALIZED_CHANNEL_CONVENTION,
    RNG_NOTE,
    SCHEMA_VERSION,
    WIDTH_SEMANTICS,
    atomic_write_csv,
    candidate_values,
    canonical_decimal,
    decoded_timestamp_summary,
    file_identity,
    fine_toa_tolerance_s,
    inband_match_frequency_MHz,
    initialize_checkpoint,
    make_unit_rng,
    match_two_inband,
    plan_summary,
    prepare_plan,
    run_simulated_peakfinders,
    sample_injected_dm,
    sample_safe_toa_placement,
    sample_spectral_interval,
    update_metadata,
    validate_dm_sampling_range,
    validate_matched_cardinality_containment,
    validate_resume_metadata,
    validate_resumed_trial_count,
)
from .peakfinders import BENCHMARK_METHODS, build_peakfinder_geometry

DEFAULT_SEPARATIONS_MS = tuple(range(5, 21))
DEFAULT_PARAMETER_SEED = 20260814
INJECTED_WIDTH_MS = 1.0
INJECTED_SNR = 20.0
UNIT_SEED_DERIVATION = (
    "uint64 little-endian prefix of SHA256(JSON([parameter_seed, "
    "'separability', separation_id, trial]))"
)
CANDIDATE_AUDIT_FIELDS = (
    "idm", "itime", "time_chunk_index", "tree", "snr", "dm",
    "toa_ref_s", "toa_inband_s", "toa_residual_ms", "argmax_token",
    "fmin", "fmax", "freq_lo_MHz", "freq_hi_MHz",
)

FIELDS = [
    "schema_version", "separation_id", "separation_ms", "trial", "method",
    "map_id", "parameter_seed", "trial_seed", "tree", "estimated_idm",
    "lower_dm_margin_bins", "upper_dm_margin_bins", "injected_dm",
    "injected_snr", "injected_width_ms", "full_band_freq_lo_MHz",
    "full_band_freq_hi_MHz", "requested_bandwidth_MHz",
    "requested_freq_lo_MHz", "requested_freq_hi_MHz",
    "active_channel_indices", "active_channel_count",
    "effective_active_freq_lo_MHz", "effective_active_freq_hi_MHz",
    "lower_boundary_on_channel_edge", "upper_boundary_on_channel_edge",
    "injected_toa_1_s", "injected_toa_2_s",
    "injected_toa_1_native_samples", "injected_toa_2_native_samples",
    "output_map_index", "output_bin_index_1", "output_bin_index_2",
    "first_toa_output_bin_phase", "full_band_time_radius_bins",
    "full_band_time_radius_ms", "safe_first_output_bin_count",
    "safe_first_output_bin_min", "safe_first_output_bin_max",
    "pulse1_support_start", "pulse1_support_end",
    "pulse2_support_start", "pulse2_support_end",
    "burst1_detected", "burst2_detected", "both_separately_detected",
    "nmatched", "ncandidates", "candidate_set_containment",
    "unit_wall_time_s", "reference_frequency_MHz", "matching_frequency_MHz",
    "injected_toa_1_inband_s", "injected_toa_2_inband_s",
    "toa_tolerance_ms", "matching_convention",
] + [
    f"candidate{number}_{field}"
    for number in (1, 2) for field in CANDIDATE_AUDIT_FIELDS
] + ["candidate1_index", "candidate2_index"]


def separability_unit_key(row):
    return (row["separation_id"], int(row["trial"]))


def _parser():
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument("--config", default="configs/dedispersion/chord_sb2.yml")
    parser.add_argument("--metadata", default="configs/xengine_metadata.yml")
    parser.add_argument("--results-dir", default="peakfinder_tests/results_final_pirate15")
    parser.add_argument(
        "--separations-ms", nargs="+",
        default=[str(value) for value in DEFAULT_SEPARATIONS_MS])
    parser.add_argument("--trials", type=int, default=50)
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


def _parse_separations(values):
    parsed = []
    for raw in values:
        text = canonical_decimal(raw)
        value = float(text)
        if not np.isfinite(value) or value < 5.0 or value > 20.0:
            raise ValueError("--separations-ms values must lie in [5, 20] ms")
        if not value.is_integer():
            raise ValueError("--separations-ms values must be whole milliseconds")
        parsed.append((f"ms-{text}", int(value)))
    if not parsed:
        raise ValueError("--separations-ms requires at least one value")
    numeric = [value for _, value in parsed]
    if len(set(numeric)) != len(numeric):
        raise ValueError("--separations-ms must not contain duplicates")
    if numeric != sorted(numeric):
        raise ValueError("--separations-ms must be strictly increasing")
    return tuple(parsed)


def _prefixed_candidate(values, number):
    return {
        f"candidate{number}_{field}": values[field]
        for field in CANDIDATE_AUDIT_FIELDS
    }


def _row_bool(value):
    return value is True or str(value).strip().lower() in {"true", "1"}


def main(argv=None):
    args = _parser().parse_args(argv)
    separations = _parse_separations(args.separations_ms)
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

    config, xmd, host_plan, dcores = prepare_plan(
        args.config, args.metadata, cuda_device_id=args.device)
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
            host_plan, intended_tree, dcores=dcores, time_sample_s=time_sample_s,
            nt_in=host_plan.nt_in,
            reference_freq_mhz=reference_frequency_MHz,
            dm_reach=args.dm_reach, waist_bins=args.waist_bins,
        )
        geometry_diagnostics = geometry.diagnostics()

    minimum_separation_s = min(value for _, value in separations) / 1.0e3
    toa_tolerance_s, toa_tolerance_derivation = fine_toa_tolerance_s(
        time_sample_s, INJECTED_WIDTH_MS / 1.0e3, args.toa_tolerance_ms)
    if toa_tolerance_s >= 0.5 * minimum_separation_s:
        raise ValueError(
            f"TOA tolerance {1.0e3 * toa_tolerance_s:.6g} ms must be below "
            f"half the minimum separation ({0.5e3 * minimum_separation_s:.6g} ms)"
        )

    scientific_parameters = {
        "dcores": list(dcores),
        "argmax_encoding": ARGMAX_ENCODING,
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
        "separations_ms": [value for _, value in separations],
        "injected_width_ms": INJECTED_WIDTH_MS,
        "injected_snr": INJECTED_SNR,
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
        "width_semantics": WIDTH_SEMANTICS,
        "toa_sampling": (
            "Uniform feasible first-burst output bin and uniform sub-bin phase; "
            "both bursts, simulator support, and complete Full-band footprints "
            "are validated inside one selected output map"
        ),
        "toa_tolerance_ms": 1.0e3 * toa_tolerance_s,
        "toa_tolerance_override_ms": args.toa_tolerance_ms,
        "matching_convention": MATCHING_CONVENTION,
        "realized_channel_convention": REALIZED_CHANNEL_CONVENTION,
    }

    results_dir = Path(args.results_dir)
    output = results_dir / "separability.csv"
    metadata_path = results_dir / "metadata.yaml"
    previous = None
    if args.resume:
        previous = validate_resume_metadata(
            metadata_path, "separability", scientific_parameters)
        validate_resumed_trial_count(previous, args.trials)

    rows, completed = initialize_checkpoint(
        output, FIELDS, resume=args.resume, overwrite=args.overwrite,
        unit_key=separability_unit_key, row_key=lambda row: row["method"],
        expected_row_keys=BENCHMARK_METHODS,
    )
    intended = {
        (separation_id, trial)
        for separation_id, _ in separations for trial in range(args.trials)
    }

    def checkpoint_metadata():
        completed_rows = [
            row for row in rows if separability_unit_key(row) in intended
        ]
        wall_times = {
            separability_unit_key(row): float(row["unit_wall_time_s"])
            for row in completed_rows
        }
        separated = {
            method: sum(
                _row_bool(row["both_separately_detected"])
                for row in completed_rows if row["method"] == method
            )
            for method in BENCHMARK_METHODS
        }
        update_metadata(metadata_path, "separability", {
            "scientific_parameters": scientific_parameters,
            "intended_trials": args.trials,
            "intended_logical_units": len(intended),
            "completed_logical_units": len(completed & intended),
            "completed_unit_keys": [
                {"separation_id": key[0], "trial": key[1]}
                for key in sorted(completed & intended)
            ],
            "complete": intended <= completed,
            "recorded_rows": len(completed_rows),
            "both_separately_detected_by_method": separated,
            "parameter_seed": args.parameter_seed,
            "separations_ms": [value for _, value in separations],
            "intended_tree": intended_tree,
            "dm_sampling_validation": dm_range,
            "reference_frequency_MHz": reference_frequency_MHz,
            "injected_width_ms": INJECTED_WIDTH_MS,
            "injected_snr": INJECTED_SNR,
            "width_semantics": WIDTH_SEMANTICS,
            "toa_tolerance_s": toa_tolerance_s,
            "toa_tolerance_derivation": toa_tolerance_derivation,
            "matching_convention": MATCHING_CONVENTION,
            "realized_channel_convention": REALIZED_CHANNEL_CONVENTION,
            "decoded_timestamp": decoded_timestamp_summary(
                host_plan, intended_tree, time_sample_s, dcores=dcores),
            "candidate_set_containment_required": (
                "The result mapping contains exactly the retained full-band method"
            ),
            "candidate_set_containment_passed": all(
                _row_bool(row["candidate_set_containment"])
                for row in completed_rows
            ),
            "methods_received_identical_maps": True,
            "unit_wall_time_definition": (
                "One real two-burst simulation/dedispersion and the full-band "
                "peakfinder; safe-placement preview is excluded."
            ),
            "median_completed_unit_wall_time_s": (
                float(np.median(list(wall_times.values()))) if wall_times else None
            ),
            "plan": plan_summary(host_plan, time_sample_s, dcores=dcores),
            "footprints": [geometry_diagnostics],
            "rng_limitation": RNG_NOTE,
        }, device=args.device)

    checkpoint_metadata()
    for separation_id, separation_ms in separations:
        separation_s = separation_ms / 1.0e3
        for trial in range(args.trials):
            unit = (separation_id, trial)
            if unit in completed:
                print(f"{separation_id} trial={trial}: already complete, skipping")
                continue

            rng, trial_seed = make_unit_rng(
                args.parameter_seed, "separability", separation_id, trial)
            injected = sample_injected_dm(
                rng, host_plan, dm_reach=args.dm_reach,
                intended_tree=intended_tree)
            injected.update(sample_spectral_interval(rng, xmd))
            placement = sample_safe_toa_placement(
                rng,
                metadata_path=args.metadata,
                nchunks=args.nchunks,
                ntime=args.ntime,
                dm=injected["dm"],
                snr=[INJECTED_SNR, INJECTED_SNR],
                width_ms=[INJECTED_WIDTH_MS, INJECTED_WIDTH_MS],
                freq_lo_MHz=injected["requested_freq_lo_MHz"],
                freq_hi_MHz=injected["requested_freq_hi_MHz"],
                geometry=geometry,
                burst_offsets_s=(0.0, separation_s),
            )
            toas = tuple(float(value) for value in placement["toas_s"])
            map_id = f"separability-{separation_id}-{trial:03d}"

            start = time.perf_counter()
            decoded, info = run_simulated_peakfinders(
                metadata_path=args.metadata,
                config_path=args.config,
                nchunks=args.nchunks,
                ntime=args.ntime,
                dm=injected["dm"],
                snr=[INJECTED_SNR, INJECTED_SNR],
                width_ms=[INJECTED_WIDTH_MS, INJECTED_WIDTH_MS],
                toas=toas,
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
            if not np.isclose(
                1.0e3 * (toas[1] - toas[0]), separation_ms,
                rtol=0.0, atol=1.0e-9,
            ):
                raise RuntimeError("stored continuous TOAs changed the requested separation")

            matching_frequency_MHz = inband_match_frequency_MHz(
                injected["requested_freq_lo_MHz"], injected["requested_freq_hi_MHz"])
            assignments = {}
            candidate_toas = {}
            injected_toas_inband = {}
            for method in BENCHMARK_METHODS:
                match, injected_inband, recovered_inband = match_two_inband(
                    decoded[method],
                    injected_dm=injected["dm"],
                    injected_toas_ref_s=toas,
                    reference_freq_MHz=reference_frequency_MHz,
                    match_freq_MHz=matching_frequency_MHz,
                    toa_tolerance_s=toa_tolerance_s,
                )
                assignments[method] = match
                candidate_toas[method] = recovered_inband
                injected_toas_inband[method] = injected_inband
            matched_counts = validate_matched_cardinality_containment(
                assignments,
                context=f"separability {separation_id} trial={trial}",
            )
            first_inband = injected_toas_inband[BENCHMARK_METHODS[0]]
            if any(
                not np.array_equal(first_inband, injected_toas_inband[method])
                for method in BENCHMARK_METHODS[1:]
            ):
                raise RuntimeError("methods disagree on injected in-band TOAs")

            supports = placement["pulse_supports"]
            native_toas = info["preflight"]["toa_native_samples"]
            unit_rows = []
            for method in BENCHMARK_METHODS:
                first, second = assignments[method]
                recovered1 = candidate_values(
                    decoded[method], first,
                    candidate_toas_match=candidate_toas[method],
                    injected_dm=injected["dm"],
                    injected_toa_match_s=injected_toas_inband[method][0],
                    dm_step=info["dm_step"],
                )
                recovered2 = candidate_values(
                    decoded[method], second,
                    candidate_toas_match=candidate_toas[method],
                    injected_dm=injected["dm"],
                    injected_toa_match_s=injected_toas_inband[method][1],
                    dm_step=info["dm_step"],
                )
                unit_rows.append({
                    "schema_version": SCHEMA_VERSION,
                    "separation_id": separation_id,
                    "separation_ms": separation_ms,
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
                    "injected_snr": INJECTED_SNR,
                    "injected_width_ms": INJECTED_WIDTH_MS,
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
                    "injected_toa_1_s": toas[0],
                    "injected_toa_2_s": toas[1],
                    "injected_toa_1_native_samples": native_toas[0],
                    "injected_toa_2_native_samples": native_toas[1],
                    "output_map_index": placement["output_map_index"],
                    "output_bin_index_1": placement["output_bin_indices"][0],
                    "output_bin_index_2": placement["output_bin_indices"][1],
                    "first_toa_output_bin_phase": placement["first_toa_output_bin_phase"],
                    "full_band_time_radius_bins": placement["full_band_time_radius_bins"],
                    "full_band_time_radius_ms": placement["full_band_time_radius_ms"],
                    "safe_first_output_bin_count": placement["safe_first_output_bin_count"],
                    "safe_first_output_bin_min": placement["safe_first_output_bin_min"],
                    "safe_first_output_bin_max": placement["safe_first_output_bin_max"],
                    "pulse1_support_start": supports[0][0],
                    "pulse1_support_end": supports[0][1],
                    "pulse2_support_start": supports[1][0],
                    "pulse2_support_end": supports[1][1],
                    "burst1_detected": first is not None,
                    "burst2_detected": second is not None,
                    "both_separately_detected": first is not None and second is not None,
                    "nmatched": matched_counts[method],
                    "ncandidates": len(decoded[method]["snr"]),
                    "candidate_set_containment": info["candidate_set_containment"],
                    "unit_wall_time_s": wall_time_s,
                    "reference_frequency_MHz": reference_frequency_MHz,
                    "matching_frequency_MHz": matching_frequency_MHz,
                    "injected_toa_1_inband_s": injected_toas_inband[method][0],
                    "injected_toa_2_inband_s": injected_toas_inband[method][1],
                    "toa_tolerance_ms": 1.0e3 * toa_tolerance_s,
                    "matching_convention": MATCHING_CONVENTION,
                    **_prefixed_candidate(recovered1, 1),
                    **_prefixed_candidate(recovered2, 2),
                    "candidate1_index": "" if first is None else int(first),
                    "candidate2_index": "" if second is None else int(second),
                })

            rows.extend(unit_rows)
            completed.add(unit)
            atomic_write_csv(output, FIELDS, rows)
            checkpoint_metadata()
            print(f"{separation_id} trial={trial} seed={trial_seed}: checkpointed")

    checkpoint_metadata()
    print(
        f"{output}: {len(completed & intended)}/{len(intended)} "
        "separation/trial units complete"
    )


if __name__ == "__main__":
    main()
