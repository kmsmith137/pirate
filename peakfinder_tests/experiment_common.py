"""Shared utilities for retained full-band peak-finder experiments."""

import csv
import hashlib
import json
import os
import subprocess
import tempfile
from collections import defaultdict
from dataclasses import dataclass
from decimal import Decimal, InvalidOperation
from pathlib import Path

import cupy as cp
import numpy as np
import yaml

from pirate_frb import DedispersionConfig
from pirate_frb.OfflineDedisperser import OfflineDedisperser
from pirate_frb.core import XEngineMetadata
from pirate_frb.make_simulated_acq import create_simulated_acquisition
from pirate_frb.simpulse import dispersion_delay

from .producer_metadata import ARGMAX_ENCODING, build_producer_plan

from .peakfinders import (
    BENCHMARK_METHODS,
    METHOD_LABELS,
    build_peakfinder_geometry,
    decode_candidates,
    empty_decoded_candidates,
    run_peakfinder,
    producer_dcore_array,
    validate_candidate_set_containment,
)

SCHEMA_VERSION = 5
SCHEMA_NAME = "pirate-peakfinder-full-band-benchmarks"
DEFAULT_SEPARATION_BINS = (
    "0.50", "0.75", "1.00", "1.25", "1.50", "1.75", "2.00",
    "2.25", "2.50", "2.75", "3.00", "3.25", "3.50", "3.75",
    "4.00", "5.00", "6.00", "8.00", "10.00", "12.00", "16.00",
)
MATCHING_CONVENTION = (
    "in-band-v1: transform injected and decoded reference-frequency TOAs to the "
    "inverse-square midpoint of the injected frequency interval; associate by in-band "
    "TOA tolerance only; maximize cardinality, minimize total absolute residual, "
    "then prefer higher total S/N and lower candidate indices"
)
REALIZED_CHANNEL_CONVENTION = (
    "SinglePulse overlap: channel i is active iff edge[i+1] > requested_lo and "
    "edge[i] < requested_hi; boundary-cut channels are included whole"
)
RNG_NOTE = (
    "The parameter seed controls burst parameters, not simulation noise. "
    "PIRATE 1.5 seeds AVX2 noise from the calling thread's ksgpu default RNG "
    "on first use; these campaigns do not seed or reset that noise stream per unit. "
    "All methods receive identical maps within a logical unit. Resume preserves "
    "completed units, but an interrupted unit rerun is not guaranteed to reproduce "
    "its previous noise."
)

TIMESTAMP_SEMANTICS = (
    "decode_argmax2 returns a chunk-relative pulse-centre estimate in full-resolution "
    "input samples, extrapolated to the lowest edge of the complete observing band."
)
WIDTH_SEMANTICS = (
    "width_ms is the frequency-independent standard deviation sigma of the "
    "intrinsic Gaussian temporal profile, not its FWHM; "
    "FWHM = 2*sqrt(2*ln(2))*sigma ~= 2.354820045*width_ms"
)

DEFAULT_DM_RANGE = (20.0, 150.0)
DEFAULT_BANDWIDTH_MIN_MHZ = 100.0
DEFAULT_WIDTH_RANGE_MS = (0.5, 20.0)
DEFAULT_SNR_RANGE = (10.0, 50.0)


def file_identity(path):
    path = Path(path).resolve()
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return {"path": str(path), "sha256": digest.hexdigest()}


def prepare_plan(config_path, metadata_path, *, cuda_device_id=None):
    """Prepare NEW simulations and return config, metadata, plan, producer Dcores."""
    config = DedispersionConfig.from_yaml(config_path)
    xmd = XEngineMetadata.from_yaml_file(metadata_path)
    xmd.validate()
    config.beams_per_gpu = 1
    config.beams_per_batch = 1
    config.num_active_batches = 1
    config.zone_nfreq = list(xmd.zone_nfreq)
    config.zone_freq_edges = list(xmd.zone_freq_edges)
    config.time_sample_ms = xmd.dt_ns_per_seq * xmd.seq_per_frb_time_sample / 1.0e6
    config.validate()
    producer = build_producer_plan(config, cuda_device_id=cuda_device_id)
    return config, xmd, producer.plan, producer.dcores


def plan_summary(plan, time_sample_s, *, dcores):
    dcores = producer_dcore_array(plan, dcores)
    return [{
        "dcore": int(dcores[itree]),
        "argmax_encoding": ARGMAX_ENCODING,
        "tree": itree,
        "dm_min": float(tree.dm_min),
        "dm_max": float(tree.dm_max),
        "ndm_out": int(tree.ndm_out),
        "nt_out": int(tree.nt_out),
        "dt_out_ms": 1.0e3 * time_sample_s * int(plan.nt_in) / int(tree.nt_out),
        "multiplets": int(tree.frequency_subbands.M),
        "subbands": int(tree.frequency_subbands.N),
    } for itree, tree in enumerate(plan.trees)]


def tree_for_dm(plan, dm):
    matching = [itree for itree, tree in enumerate(plan.trees)
                if float(tree.dm_min) <= dm < float(tree.dm_max)]
    if len(matching) != 1:
        raise ValueError(f"DM {dm} belongs to {len(matching)} plan trees, expected one")
    return matching[0]

def make_unit_rng(parameter_seed, *unit_key):
    """Return a stable per-unit NumPy RNG and its derived unsigned 64-bit seed."""
    if isinstance(parameter_seed, (bool, np.bool_)):
        raise ValueError("parameter_seed must be a non-negative integer")
    try:
        seed = int(parameter_seed)
    except (TypeError, ValueError, OverflowError) as exc:
        raise ValueError("parameter_seed must be a non-negative integer") from exc
    if seed != parameter_seed or seed < 0:
        raise ValueError("parameter_seed must be a non-negative integer")
    try:
        encoded = json.dumps(
            [seed, *unit_key], sort_keys=True, separators=(",", ":"),
            allow_nan=False,
            default=lambda value: value.item() if isinstance(value, np.generic) else str(value),
        ).encode("utf-8")
    except (TypeError, ValueError) as exc:
        raise ValueError(f"unit key is not deterministically serializable: {unit_key!r}") from exc
    digest = hashlib.sha256(encoded).digest()
    unit_seed = int.from_bytes(digest[:8], byteorder="little", signed=False)
    return np.random.default_rng(unit_seed), unit_seed


def _validate_dm_reach(dm_reach):
    if isinstance(dm_reach, (bool, np.bool_)) or int(dm_reach) != dm_reach or dm_reach < 0:
        raise ValueError("dm_reach must be a non-negative integer")
    return int(dm_reach)


def validate_dm_for_peakfinder(plan, dm, *, dm_reach, intended_tree=None):
    """Validate tree membership and physical clearance for a centred DM footprint."""
    if not np.isfinite(dm):
        raise ValueError("DM must be finite")
    reach = _validate_dm_reach(dm_reach)
    itree = tree_for_dm(plan, float(dm))
    if intended_tree is not None and itree != int(intended_tree):
        raise ValueError(
            f"DM {dm} belongs to tree {itree}, not intended tree {intended_tree}"
        )
    tree = plan.trees[itree]
    ndm = int(tree.ndm_out)
    if ndm <= 2 * reach:
        raise ValueError(
            f"tree {itree} has ndm_out={ndm}, insufficient for dm_reach={reach}"
        )
    tree_min = float(tree.dm_min)
    tree_max = float(tree.dm_max)
    dm_step = (tree_max - tree_min) / ndm
    safe_min = tree_min + reach * dm_step
    safe_max = tree_max - reach * dm_step
    if not safe_min <= float(dm) < safe_max:
        raise ValueError(
            f"DM {dm} lacks dm_reach={reach} boundary clearance in tree {itree}; "
            f"required [{safe_min}, {safe_max}) pc cm^-3"
        )
    return {
        "tree": itree,
        "dm_step": dm_step,
        "safe_dm_min": safe_min,
        "safe_dm_max": safe_max,
        "lower_dm_margin_bins": (float(dm) - tree_min) / dm_step,
        "upper_dm_margin_bins": (tree_max - float(dm)) / dm_step,
        "estimated_idm": int(np.floor((float(dm) - tree_min) / dm_step)),
    }


def validate_dm_sampling_range(
    plan, *, dm_reach, dm_min=DEFAULT_DM_RANGE[0],
    dm_max=DEFAULT_DM_RANGE[1], intended_tree=None,
):
    """Require the complete half-open sampling interval to fit one safe tree."""
    if not (np.isfinite(dm_min) and np.isfinite(dm_max) and dm_min < dm_max):
        raise ValueError("DM sampling range must be finite and strictly increasing")
    lower = validate_dm_for_peakfinder(
        plan, float(dm_min), dm_reach=dm_reach, intended_tree=intended_tree
    )
    upper_probe = float(np.nextafter(float(dm_max), -np.inf))
    upper = validate_dm_for_peakfinder(
        plan, upper_probe, dm_reach=dm_reach,
        intended_tree=lower["tree"] if intended_tree is None else intended_tree,
    )
    if lower["tree"] != upper["tree"]:
        raise ValueError("DM sampling range spans more than one intended plan tree")
    if float(dm_max) > lower["safe_dm_max"]:
        raise ValueError(
            f"DM sampling upper bound {dm_max} exceeds safe exclusive bound "
            f"{lower['safe_dm_max']} for tree {lower['tree']}"
        )
    return {
        "tree": lower["tree"],
        "dm_step": lower["dm_step"],
        "safe_dm_min": lower["safe_dm_min"],
        "safe_dm_max": lower["safe_dm_max"],
        "sampling_dm_min": float(dm_min),
        "sampling_dm_max": float(dm_max),
    }


def sample_injected_dm(
    rng, plan, *, dm_reach, dm_min=DEFAULT_DM_RANGE[0],
    dm_max=DEFAULT_DM_RANGE[1], intended_tree=None,
):
    """Draw DM uniformly and return its validated tree/boundary audit."""
    range_info = validate_dm_sampling_range(
        plan, dm_reach=dm_reach, dm_min=dm_min, dm_max=dm_max,
        intended_tree=intended_tree,
    )
    dm = float(rng.uniform(float(dm_min), float(dm_max)))
    result = validate_dm_for_peakfinder(
        plan, dm, dm_reach=dm_reach, intended_tree=range_info["tree"]
    )
    result.update({
        "dm": dm,
        "sampling_dm_min": float(dm_min),
        "sampling_dm_max": float(dm_max),
    })
    return result


def observing_band_bounds_MHz(xmd):
    edges = np.asarray(xmd.get_channel_freq_edges(), dtype=np.float64)
    if edges.ndim != 1 or len(edges) < 2 or not np.all(np.diff(edges) > 0):
        raise ValueError("X-engine channel edges must be one-dimensional and increasing")
    return float(edges[0]), float(edges[-1])


def validate_requested_frequency_interval(
    xmd, freq_lo_MHz, freq_hi_MHz, *, minimum_bandwidth_MHz=0.0,
):
    """Validate an arbitrary requested interval and return realized active channels."""
    full_lo, full_hi = observing_band_bounds_MHz(xmd)
    lo, hi = float(freq_lo_MHz), float(freq_hi_MHz)
    minimum = float(minimum_bandwidth_MHz)
    if not (np.isfinite(lo) and np.isfinite(hi) and lo < hi):
        raise ValueError("requested frequency interval must be finite and increasing")
    if not np.isfinite(minimum) or minimum < 0:
        raise ValueError("minimum bandwidth must be finite and non-negative")
    if lo < full_lo or hi > full_hi:
        raise ValueError(
            f"requested interval [{lo}, {hi}] MHz is outside full observing band "
            f"[{full_lo}, {full_hi}] MHz"
        )
    if hi - lo < minimum:
        raise ValueError(
            f"requested bandwidth {hi - lo} MHz is below minimum {minimum} MHz"
        )
    coverage = realized_channel_coverage(xmd, lo, hi)
    return {
        "full_band_freq_lo_MHz": full_lo,
        "full_band_freq_hi_MHz": full_hi,
        "full_bandwidth_MHz": full_hi - full_lo,
        "requested_freq_lo_MHz": lo,
        "requested_freq_hi_MHz": hi,
        "requested_bandwidth_MHz": hi - lo,
        "active_channel_indices": coverage["active_channel_indices"].tolist(),
        "active_channel_indices_compact": coverage["active_channel_indices_compact"],
        "active_channel_count": coverage["active_channel_count"],
        "effective_freq_lo_MHz": coverage["effective_freq_lo_MHz"],
        "effective_freq_hi_MHz": coverage["effective_freq_hi_MHz"],
        "lower_boundary_on_channel_edge": coverage["lower_boundary_on_channel_edge"],
        "upper_boundary_on_channel_edge": coverage["upper_boundary_on_channel_edge"],
    }


def sample_spectral_interval(
    rng, xmd, *, minimum_bandwidth_MHz=DEFAULT_BANDWIDTH_MIN_MHZ,
):
    """Draw bandwidth then lower edge uniformly over the complete observing band."""
    full_lo, full_hi = observing_band_bounds_MHz(xmd)
    full_width = full_hi - full_lo
    minimum = float(minimum_bandwidth_MHz)
    if not np.isfinite(minimum) or minimum <= 0 or minimum > full_width:
        raise ValueError(
            f"minimum bandwidth must be in (0, {full_width}] MHz, got {minimum}"
        )
    bandwidth = (
        full_width if minimum == full_width
        else float(rng.uniform(minimum, full_width))
    )
    maximum_lo = full_hi - bandwidth
    freq_lo = full_lo if maximum_lo == full_lo else float(rng.uniform(full_lo, maximum_lo))
    freq_hi = freq_lo + bandwidth
    return validate_requested_frequency_interval(
        xmd, freq_lo, freq_hi, minimum_bandwidth_MHz=minimum
    )


def sample_recall_burst_parameters(rng, plan, xmd, *, dm_reach=8):
    """Draw and validate all non-TOA parameters for one recall realization."""
    result = sample_injected_dm(rng, plan, dm_reach=dm_reach)
    result.update(sample_spectral_interval(rng, xmd))
    result["width_ms"] = float(rng.uniform(*DEFAULT_WIDTH_RANGE_MS))
    result["snr"] = float(rng.uniform(*DEFAULT_SNR_RANGE))
    result["width_semantics"] = WIDTH_SEMANTICS
    return result


def decoded_timestamp_summary(plan, itree, time_sample_s, *, dcores):
    dcore = int(producer_dcore_array(plan, dcores)[itree])
    tree = plan.trees[itree]
    profiles = []
    for profile in range(int(tree.nprofiles)):
        lpf = ((profile - 1) // 3) if profile else 0
        token_quant_tree_samples = min(dcore, 1 << lpf)
        full_samples = token_quant_tree_samples * (1 << int(tree.primary_tree_index))
        profiles.append({
            "profile": profile,
            "token_quantization_input_samples": full_samples,
            "token_quantization_ms": 1.0e3 * full_samples * time_sample_s,
        })
    return {
        "semantics": TIMESTAMP_SEMANTICS,
        "reference_frequency": "lowest edge of the complete observing band",
        "primary_tree_index": int(tree.primary_tree_index),
        "Dcore": dcore,
        "input_time_sample_ms": 1.0e3 * time_sample_s,
        "profile_quantization": profiles,
    }


def fine_toa_tolerance_s(input_time_sample_s, injected_width_s, override_ms=None):
    if input_time_sample_s <= 0 or injected_width_s <= 0:
        raise ValueError("input sample time and injected width must be positive")
    if override_ms is not None:
        if not np.isfinite(override_ms) or override_ms <= 0:
            raise ValueError("--toa-tolerance-ms must be finite and positive")
        return float(override_ms) / 1.0e3, "explicit --toa-tolerance-ms override"
    value = max(float(input_time_sample_s), float(injected_width_s))
    return value, "max(effective input-sample duration, injected Gaussian sigma)"


def validate_fully_inside(simulation):
    bad = [summary for summary in simulation.burst_summaries
           if summary["status"] != "fully inside"]
    if not bad:
        return
    details = [
        f"burst {s['index']}: TOA={s['requested_toa']:.9f} s, "
        f"support=[{s['sample_start']}, {s['sample_end']}), status={s['status']}"
        for s in bad
    ]
    raise ValueError(
        "Every benchmark burst must be fully inside the simulated observation "
        f"(duration={simulation.stream_sec:.9f} s).\n" + "\n".join(details)
    )


def make_validated_simulation(**kwargs):
    simulation = create_simulated_acquisition(**kwargs)
    validate_fully_inside(simulation)
    return simulation

def full_band_temporal_radius_bins(geometry):
    """Return the largest active time offset in the complete-band footprint."""
    source = geometry.full_band_footprint
    if isinstance(source, cp.ndarray):
        with cp.cuda.Device(int(source.device.id)):
            footprint = np.asarray(cp.asnumpy(source), dtype=np.bool_)
    else:
        footprint = np.asarray(source, dtype=np.bool_)
    if footprint.ndim != 2 or not footprint.any():
        raise ValueError("full-band footprint must be a non-empty 2-D boolean array")
    if any(size % 2 == 0 for size in footprint.shape):
        raise ValueError("full-band footprint must have odd dimensions")
    active_time = np.flatnonzero(footprint.any(axis=0))
    center_time = footprint.shape[1] // 2
    return int(np.max(np.abs(active_time - center_time)))


def _placement_dimensions(simulation, geometry, output_map_index):
    if int(output_map_index) != output_map_index:
        raise ValueError("output_map_index must be an integer")
    map_index = int(output_map_index)
    if not 0 <= map_index < int(simulation.nchunks):
        raise ValueError(
            f"output_map_index={map_index} is outside [0, {simulation.nchunks})"
        )
    input_dt_s = float(simulation.time_sample_ms) / 1.0e3
    map_duration_s = int(simulation.ntime) * input_dt_s
    represented_duration_s = int(geometry.nt) * float(geometry.time_step_s)
    if not np.isclose(
        represented_duration_s, map_duration_s, rtol=1.0e-12, atol=1.0e-12
    ):
        raise ValueError(
            f"geometry represents {represented_duration_s} s but one input map "
            f"represents {map_duration_s} s"
        )
    return map_index, input_dt_s, map_duration_s


def validate_safe_toa_placement(
    simulation, *, geometry, output_map_index, expected_offsets_s=None,
):
    """Validate support, same-map placement, and complete-band edge clearance."""
    validate_fully_inside(simulation)
    map_index, _, map_duration_s = _placement_dimensions(
        simulation, geometry, output_map_index
    )
    radius = full_band_temporal_radius_bins(geometry)
    if int(geometry.nt) <= 2 * radius:
        raise ValueError(
            f"map nt={geometry.nt} is insufficient for full-band time radius {radius}"
        )
    summaries = list(simulation.burst_summaries)
    if len(summaries) not in (1, 2):
        raise ValueError("final benchmarks require exactly one or two bursts")
    toas = np.asarray(
        [float(summary["requested_toa"]) for summary in summaries], dtype=np.float64
    )
    map_start_s = map_index * map_duration_s
    relative_bins = (toas - map_start_s) / float(geometry.time_step_s)
    output_bins = np.floor(relative_bins).astype(np.int64)
    phases = relative_bins - output_bins
    if not np.all((output_bins >= radius) & (output_bins < int(geometry.nt) - radius)):
        raise ValueError(
            f"burst output bins {output_bins.tolist()} lack full-band temporal "
            f"radius {radius} in map {map_index}"
        )
    if expected_offsets_s is not None:
        expected = np.asarray(expected_offsets_s, dtype=np.float64)
        if expected.shape != toas.shape or not np.all(np.isfinite(expected)):
            raise ValueError("expected_offsets_s must contain one finite offset per burst")
        observed = toas - toas[0]
        if not np.allclose(
            observed, expected, rtol=0.0,
            atol=8 * np.finfo(np.float64).eps * max(1.0, float(np.max(np.abs(toas)))),
        ):
            raise ValueError(
                f"stored TOA offsets {observed.tolist()} disagree with "
                f"{expected.tolist()}"
            )
    return {
        "toas_s": toas.tolist(),
        "output_map_index": map_index,
        "output_bin_indices": output_bins.tolist(),
        "output_bin_phases": phases.tolist(),
        "first_toa_output_bin_phase": float(phases[0]),
        "full_band_time_radius_bins": radius,
        "full_band_time_radius_ms": 1.0e3 * radius * float(geometry.time_step_s),
        "pulse_supports": [
            [int(summary["sample_start"]), int(summary["sample_end"])]
            for summary in summaries
        ],
        "all_bursts_fully_inside_observation": True,
        "all_bursts_in_same_output_map": True,
        "full_band_edge_clearance_passed": True,
    }


def _preview_support_offsets_s(simulation, burst_offsets_s):
    """Return conservative support offsets valid for every input-sample phase."""
    input_dt_s = float(simulation.time_sample_ms) / 1.0e3
    result = []
    for summary, burst_offset_s in zip(simulation.burst_summaries, burst_offsets_s):
        result.append((
            int(summary["sample_start"]) * input_dt_s - float(burst_offset_s) - input_dt_s,
            int(summary["sample_end"]) * input_dt_s - float(burst_offset_s) + input_dt_s,
        ))
    return result


def _safe_first_bins(
    *, geometry, map_index, nchunks, ntime, input_dt_s,
    burst_offsets_s, support_offsets_s,
):
    radius = full_band_temporal_radius_bins(geometry)
    map_duration_s = int(ntime) * float(input_dt_s)
    output_dt_s = float(geometry.time_step_s)
    if not np.isclose(
        int(geometry.nt) * output_dt_s, map_duration_s,
        rtol=1.0e-12, atol=1.0e-12,
    ):
        raise ValueError("geometry/output-map duration disagrees with simulation ntime")
    map_start_s = int(map_index) * map_duration_s
    stream_end_s = int(nchunks) * map_duration_s
    feasible = []
    for first_bin in range(radius, int(geometry.nt) - radius):
        first_bin_start_s = map_start_s + first_bin * output_dt_s
        first_bin_end_s = first_bin_start_s + output_dt_s
        good = True
        for offset_s, (support_start_s, support_end_s) in zip(
            burst_offsets_s, support_offsets_s
        ):
            earliest_toa_s = first_bin_start_s + offset_s
            latest_toa_s = first_bin_end_s + offset_s
            earliest_relative = (earliest_toa_s - map_start_s) / output_dt_s
            latest_relative = np.nextafter(
                (latest_toa_s - map_start_s) / output_dt_s, -np.inf
            )
            earliest_bin = int(np.floor(earliest_relative))
            latest_bin = int(np.floor(latest_relative))
            if not (
                radius <= earliest_bin
                and latest_bin < int(geometry.nt) - radius
                and earliest_toa_s + support_start_s >= 0.0
                and latest_toa_s + support_end_s <= stream_end_s
            ):
                good = False
                break
        if good:
            feasible.append(first_bin)
    return feasible


def sample_safe_toa_placement(
    rng, *, metadata_path, nchunks, ntime, dm, snr, width_ms,
    freq_lo_MHz, freq_hi_MHz, geometry, burst_offsets_s=(0.0,),
    output_map_index=None,
):
    """Sample a uniform output-bin phase subject to all temporal safety constraints."""
    offsets = tuple(float(value) for value in burst_offsets_s)
    if len(offsets) not in (1, 2) or not np.all(np.isfinite(offsets)):
        raise ValueError("burst_offsets_s must contain one or two finite values")
    if offsets[0] != 0.0 or any(b <= a for a, b in zip(offsets, offsets[1:])):
        raise ValueError("burst_offsets_s must start at zero and be strictly increasing")
    if int(nchunks) != nchunks or nchunks < 1:
        raise ValueError("nchunks must be a positive integer")
    if int(ntime) != ntime or ntime < 1:
        raise ValueError("ntime must be a positive integer")

    preview = create_simulated_acquisition(
        metadata_path,
        nchunks=int(nchunks),
        ntime=int(ntime),
        dm=float(dm),
        snr=snr,
        width_ms=width_ms,
        toa=list(offsets),
        subband_lo_MHz=float(freq_lo_MHz),
        subband_hi_MHz=float(freq_hi_MHz),
    )
    validate_requested_frequency_interval(
        preview.xmd, freq_lo_MHz, freq_hi_MHz,
        minimum_bandwidth_MHz=0.0,
    )
    support_offsets = _preview_support_offsets_s(preview, offsets)
    input_dt_s = float(preview.time_sample_ms) / 1.0e3

    if output_map_index is None:
        centre = int(nchunks) // 2
        map_indices = sorted(range(int(nchunks)), key=lambda value: (abs(value - centre), value))
    else:
        if int(output_map_index) != output_map_index:
            raise ValueError("output_map_index must be an integer")
        map_indices = [int(output_map_index)]

    selected_map = None
    feasible_bins = []
    for map_index in map_indices:
        if not 0 <= map_index < int(nchunks):
            continue
        feasible = _safe_first_bins(
            geometry=geometry,
            map_index=map_index,
            nchunks=int(nchunks),
            ntime=int(ntime),
            input_dt_s=input_dt_s,
            burst_offsets_s=offsets,
            support_offsets_s=support_offsets,
        )
        if feasible:
            selected_map, feasible_bins = map_index, feasible
            break
    if selected_map is None:
        raise ValueError(
            "no output map has a complete Full-band footprint and fully contained "
            "pulse support for the requested burst parameters"
        )

    first_bin = int(rng.choice(np.asarray(feasible_bins, dtype=np.int64)))
    phase = float(rng.random())
    map_duration_s = int(ntime) * input_dt_s
    first_toa_s = (
        selected_map * map_duration_s
        + (first_bin + phase) * float(geometry.time_step_s)
    )
    toas = [first_toa_s + offset for offset in offsets]
    simulation = create_simulated_acquisition(
        metadata_path,
        nchunks=int(nchunks),
        ntime=int(ntime),
        dm=float(dm),
        snr=snr,
        width_ms=width_ms,
        toa=toas,
        subband_lo_MHz=float(freq_lo_MHz),
        subband_hi_MHz=float(freq_hi_MHz),
    )
    audit = validate_safe_toa_placement(
        simulation,
        geometry=geometry,
        output_map_index=selected_map,
        expected_offsets_s=offsets,
    )
    audit.update({
        "safe_first_output_bin_count": len(feasible_bins),
        "safe_first_output_bin_min": min(feasible_bins),
        "safe_first_output_bin_max": max(feasible_bins),
        "width_semantics": WIDTH_SEMANTICS,
    })
    return audit

def _fsync_and_replace(temp_name, destination):
    os.replace(temp_name, destination)


def atomic_write_csv(path, fieldnames, rows):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temp_name = None
    try:
        with tempfile.NamedTemporaryFile(
            "w", newline="", dir=path.parent, prefix=path.name + ".",
            suffix=".tmp", delete=False,
        ) as stream:
            temp_name = stream.name
            writer = csv.DictWriter(stream, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(rows)
            stream.flush()
            os.fsync(stream.fileno())
        _fsync_and_replace(temp_name, path)
    finally:
        if temp_name is not None and os.path.exists(temp_name):
            os.unlink(temp_name)


def read_csv(path, expected_fieldnames):
    path = Path(path)
    with path.open(newline="") as stream:
        reader = csv.DictReader(stream)
        if reader.fieldnames != list(expected_fieldnames):
            observed = reader.fieldnames or []
            if not observed or observed[0] != "schema_version":
                raise ValueError(
                    f"{path}: incompatible pre-v{SCHEMA_VERSION} CSV schema; "
                    "start a new results directory or use --overwrite"
                )
            raise ValueError(
                f"{path}: CSV columns do not match schema v{SCHEMA_VERSION}; "
                "start a new results directory or use --overwrite"
            )
        rows = list(reader)
    for lineno, row in enumerate(rows, start=2):
        try:
            version = int(row["schema_version"])
        except (KeyError, TypeError, ValueError) as exc:
            raise ValueError(f"{path}:{lineno}: invalid schema_version") from exc
        if version != SCHEMA_VERSION:
            raise ValueError(
                f"{path}:{lineno}: schema_version={version}, expected {SCHEMA_VERSION}"
            )
    return rows


write_csv = atomic_write_csv


def clean_completed_units(rows, unit_key, row_key, expected_row_keys):
    """Retain only exact, duplicate-free logical units."""
    expected = set(expected_row_keys)
    groups = defaultdict(list)
    for row in rows:
        groups[unit_key(row)].append(row)
    retained, completed = [], set()
    for key, unit_rows in groups.items():
        actual = [row_key(row) for row in unit_rows]
        if len(actual) == len(expected) and set(actual) == expected:
            retained.extend(unit_rows)
            completed.add(key)
    return retained, completed


def initialize_checkpoint(path, fieldnames, *, resume, overwrite,
                          unit_key, row_key, expected_row_keys):
    if resume and overwrite:
        raise ValueError("--resume and --overwrite are mutually exclusive")
    path = Path(path)
    if path.exists() and not (resume or overwrite):
        raise FileExistsError(
            f"{path} exists; use --resume to continue or --overwrite to replace it"
        )
    if resume and not path.exists():
        raise FileNotFoundError(f"cannot --resume because {path} does not exist")
    rows = read_csv(path, fieldnames) if resume else []
    rows, completed = clean_completed_units(
        rows, unit_key, row_key, expected_row_keys
    )
    atomic_write_csv(path, fieldnames, rows)
    return rows, completed


def read_metadata(path):
    path = Path(path)
    if not path.exists():
        return {}
    with path.open() as stream:
        return yaml.safe_load(stream) or {}


def require_metadata_schema(metadata, path="metadata.yaml"):
    suite = metadata.get("suite", {})
    observed = suite.get("schema_version")
    if observed != SCHEMA_VERSION or suite.get("schema_name") != SCHEMA_NAME:
        raise ValueError(
            f"{path}: incompatible benchmark schema {observed!r}; expected "
            f"{SCHEMA_NAME} v{SCHEMA_VERSION}. Use a new results directory."
        )
    if tuple(suite.get("methods", [])) != BENCHMARK_METHODS:
        raise ValueError(
            f"{path}: method schema {suite.get('methods')!r} does not match {BENCHMARK_METHODS!r}"
        )


def validate_resume_metadata(path, section, scientific_parameters):
    metadata = read_metadata(path)
    require_metadata_schema(metadata, path)
    previous = metadata.get(section)
    if previous is None:
        raise ValueError(f"cannot --resume: {path} has no {section!r} metadata")
    old_parameters = previous.get("scientific_parameters")
    if old_parameters != scientific_parameters:
        keys = sorted(set((old_parameters or {}).keys()) | set(scientific_parameters))
        differences = [
            f"{key}: previous={(old_parameters or {}).get(key)!r}, "
            f"current={scientific_parameters.get(key)!r}"
            for key in keys
            if (old_parameters or {}).get(key) != scientific_parameters.get(key)
        ]
        raise ValueError(
            f"cannot --resume {section}: scientific parameters differ:\n"
            + "\n".join(differences)
        )
    return previous


def validate_resumed_trial_count(previous_section, intended_trials):
    previous_intended = int(previous_section.get("intended_trials", 0))
    if intended_trials < previous_intended:
        raise ValueError(
            f"cannot reduce --trials while resuming ({intended_trials} < "
            f"{previous_intended}); use --overwrite"
        )


def _git_commit():
    try:
        return subprocess.check_output(
            ["git", "rev-parse", "HEAD"], text=True,
            stderr=subprocess.DEVNULL,
        ).strip()
    except (OSError, subprocess.CalledProcessError):
        return None


def _gpu_name(device):
    try:
        name = cp.cuda.runtime.getDeviceProperties(device)["name"]
        return name.decode() if isinstance(name, bytes) else str(name)
    except Exception:
        return None


def atomic_write_yaml(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temp_name = None
    try:
        with tempfile.NamedTemporaryFile(
            "w", dir=path.parent, prefix=path.name + ".",
            suffix=".tmp", delete=False,
        ) as stream:
            temp_name = stream.name
            yaml.safe_dump(value, stream, sort_keys=False)
            stream.flush()
            os.fsync(stream.fileno())
        _fsync_and_replace(temp_name, path)
    finally:
        if temp_name is not None and os.path.exists(temp_name):
            os.unlink(temp_name)


def update_metadata(path, section, values, *, device=0):
    path = Path(path)
    metadata = read_metadata(path)
    metadata["suite"] = {
        "schema_name": SCHEMA_NAME,
        "schema_version": SCHEMA_VERSION,
        "methods": list(BENCHMARK_METHODS),
        "method_labels": {
            method: METHOD_LABELS[method] for method in BENCHMARK_METHODS},
        "git_commit": _git_commit(),
        "gpu": _gpu_name(device),
        "noise_reproducible": False,
        "rng_note": RNG_NOTE,
    }
    values = dict(values)
    footprint_entries = values.pop("footprints", [])
    if footprint_entries:
        by_tree = metadata.setdefault("footprints_by_tree", {})
        for entry in footprint_entries:
            by_tree[str(entry["tree"])] = entry
    metadata[section] = values
    atomic_write_yaml(path, metadata)

def select_subbands(subbands, requested_ids):
    """Validate a CLI sub-band subset while preserving plan hierarchy order."""
    available = {band.subband_id: band for band in subbands}
    if requested_ids is None:
        return tuple(subbands)
    if len(requested_ids) != len(set(requested_ids)):
        duplicates = sorted({value for value in requested_ids
                             if requested_ids.count(value) > 1})
        raise ValueError(f"--subband-ids contains duplicates: {duplicates}")
    unknown = [value for value in requested_ids if value not in available]
    if unknown:
        listing = ", ".join(
            f"{b.subband_id}=[{b.freq_lo_MHz:.6f},{b.freq_hi_MHz:.6f}] MHz"
            for b in subbands
        )
        raise ValueError(
            f"unknown --subband-ids {unknown}; available sub-bands: {listing}"
        )
    selected = set(requested_ids)
    return tuple(band for band in subbands if band.subband_id in selected)


def canonical_decimal(value):
    """Return a stable non-exponent decimal string without changing its value."""
    try:
        decimal_value = value if isinstance(value, Decimal) else Decimal(str(value))
    except (InvalidOperation, ValueError) as exc:
        raise ValueError(f"invalid decimal separation {value!r}") from exc
    if not decimal_value.is_finite():
        raise ValueError(f"separation must be finite, got {value!r}")
    normalized = decimal_value.normalize()
    result = format(normalized, "f")
    return "0" if result in ("-0", "") else result


@dataclass(frozen=True)
class SeparationPoint:
    separation_id: str
    bins_decimal: str
    bins: float
    milliseconds: float
    native_samples: float
    native_samples_decimal: str

    def as_dict(self):
        return {
            "separation_id": self.separation_id,
            "separation_bins": self.bins,
            "separation_bins_decimal": self.bins_decimal,
            "separation_ms": self.milliseconds,
            "separation_native_samples": self.native_samples,
            "separation_native_samples_decimal": self.native_samples_decimal,
        }


def parse_separation_grid(values, *, time_sample_s, ntime, nt_out):
    """Preserve decimal input and continuous native-sample positions."""
    dt_decimal = Decimal(str(float(time_sample_s)))
    ratio = Decimal(int(ntime)) / Decimal(int(nt_out))
    points, physical_keys = [], set()
    for raw in values:
        bins_text = canonical_decimal(raw)
        bins_decimal = Decimal(bins_text)
        if bins_decimal <= 0:
            raise ValueError("--separation-bins values must be strictly positive")
        native = bins_decimal * ratio
        physical_key = native.normalize()
        if physical_key in physical_keys:
            raise ValueError(
                f"--separation-bins contains duplicate physical separation at "
                f"{canonical_decimal(native)} native samples"
            )
        physical_keys.add(physical_key)
        points.append(SeparationPoint(
            separation_id=f"bins-{bins_text}",
            bins_decimal=bins_text,
            bins=float(bins_decimal),
            milliseconds=float(bins_decimal * ratio * dt_decimal * Decimal(1000)),
            native_samples=float(native),
            native_samples_decimal=canonical_decimal(native),
        ))
    if not points:
        raise ValueError("--separation-bins requires at least one value")
    return tuple(points)


def inband_match_frequency_MHz(freq_lo_MHz, freq_hi_MHz):
    if not 0 < freq_lo_MHz < freq_hi_MHz:
        raise ValueError("matching interval must have 0 < freq_lo < freq_hi")
    return ((freq_lo_MHz**-2 + freq_hi_MHz**-2) / 2.0) ** -0.5


def toa_at_frequency(toa_ref_s, dm, reference_freq_MHz, target_freq_MHz):
    """Transform reference-edge TOAs with PIRATE's dispersion-delay helper."""
    delta_per_dm = (
        float(dispersion_delay(1.0, target_freq_MHz))
        - float(dispersion_delay(1.0, reference_freq_MHz))
    )
    return (np.asarray(toa_ref_s, dtype=np.float64)
            + np.asarray(dm, dtype=np.float64) * delta_per_dm)


def realized_channel_coverage(xmd, freq_lo_MHz, freq_hi_MHz):
    """Apply exactly SinglePulse's open-overlap rule to X-engine channels."""
    edges = np.asarray(xmd.get_channel_freq_edges(), dtype=np.float64)
    if edges.ndim != 1 or len(edges) < 2 or not np.all(np.diff(edges) > 0):
        raise ValueError("X-engine channel edges must be one-dimensional and increasing")
    active = np.flatnonzero(
        (edges[1:] > float(freq_lo_MHz)) & (edges[:-1] < float(freq_hi_MHz))
    ).astype(np.int64)
    if not active.size:
        raise ValueError(
            f"requested sub-band [{freq_lo_MHz}, {freq_hi_MHz}] MHz activates no channels"
        )
    contiguous = bool(np.all(np.diff(active) == 1))
    if not contiguous:
        raise ValueError(
            "SinglePulse active channels are non-contiguous even though the simulated "
            "X-engine metadata defines a contiguous increasing channel grid"
        )
    first, stop = int(active[0]), int(active[-1]) + 1
    return {
        "active_channel_indices": active,
        "active_channel_indices_compact": f"{first}:{stop}",
        "active_channel_count": int(active.size),
        "effective_freq_lo_MHz": float(edges[first]),
        "effective_freq_hi_MHz": float(edges[stop]),
        "lower_boundary_on_channel_edge": bool(np.any(edges == float(freq_lo_MHz))),
        "upper_boundary_on_channel_edge": bool(np.any(edges == float(freq_hi_MHz))),
        "contiguous": contiguous,
        "convention": REALIZED_CHANNEL_CONVENTION,
    }


def validate_simulation_frequency_interval(
    simulation, freq_lo_MHz, freq_hi_MHz, coverage, *, expected_bursts=None,
):
    """Validate arbitrary requested bounds and realized channels for one/two bursts."""
    npulses = len(simulation.pulses)
    if npulses not in (1, 2) or len(simulation.burst_summaries) != npulses:
        raise ValueError("final benchmarks require exactly one or two simulated bursts")
    if expected_bursts is not None and npulses != int(expected_bursts):
        raise ValueError(f"simulation has {npulses} bursts, expected {expected_bursts}")
    requested_toas = [
        float(summary["requested_toa"]) for summary in simulation.burst_summaries
    ]
    if len(requested_toas) == 2 and requested_toas[0] == requested_toas[1]:
        raise ValueError("the two requested pulse-centre TOAs must be distinct")

    expected_active = np.asarray(coverage["active_channel_indices"], dtype=np.int64)
    actual_by_burst = []
    for iburst, pulse in enumerate(simulation.pulses):
        if (
            float(pulse.subband_freq_lo_MHz) != float(freq_lo_MHz)
            or float(pulse.subband_freq_hi_MHz) != float(freq_hi_MHz)
        ):
            raise ValueError(
                f"burst {iburst} requested frequency interval disagrees with "
                f"[{freq_lo_MHz}, {freq_hi_MHz}] MHz"
            )
        actual = np.flatnonzero(np.asarray(pulse.freq_nt) > 0).astype(np.int64)
        if not np.array_equal(actual, expected_active):
            raise ValueError(
                f"burst {iburst} active channels disagree with SinglePulse overlap preview"
            )
        actual_by_burst.append(actual)
    if any(
        not np.array_equal(actual_by_burst[0], actual)
        for actual in actual_by_burst[1:]
    ):
        raise ValueError("the bursts do not use identical active X-engine channels")

    offsets = [toa - requested_toas[0] for toa in requested_toas]
    result = {
        "requested_toas_s": requested_toas,
        "toa_native_samples": [
            toa / (simulation.time_sample_ms / 1.0e3) for toa in requested_toas
        ],
        "stored_toa_offsets_s": offsets,
        "pulse_supports": [
            [int(summary["sample_start"]), int(summary["sample_end"])]
            for summary in simulation.burst_summaries
        ],
        "active_channels_verified_for_all_bursts": True,
        "active_channel_indices_compact": coverage["active_channel_indices_compact"],
        "active_channel_count": coverage["active_channel_count"],
    }
    if len(requested_toas) == 2:
        result["stored_toa_difference_s"] = offsets[1]
    return result


def validate_simulation_subband(simulation, subband, coverage):
    """Backward-compatible validator for historical plan-subband two-burst tests."""
    return validate_simulation_frequency_interval(
        simulation,
        subband.freq_lo_MHz,
        subband.freq_hi_MHz,
        coverage,
        expected_bursts=2,
    )


def concatenate_decoded(parts):
    if not parts:
        return empty_decoded_candidates()
    return {key: np.concatenate([part[key] for part in parts])
            for key in empty_decoded_candidates()}


def run_simulated_peakfinders(
    *, metadata_path, config_path, nchunks, ntime, dm, snr, width_ms, toas,
    threshold, device, dm_reach, waist_bins, freq_lo_MHz, freq_hi_MHz,
    expected_tree=None, output_map_index=None,
):
    """Run the retained production peak finder on real simulated acquisitions."""
    requested_toas = np.atleast_1d(np.asarray(toas, dtype=np.float64))
    if requested_toas.ndim != 1 or len(requested_toas) not in (1, 2):
        raise ValueError("toas must contain exactly one or two values")
    if not np.all(np.isfinite(requested_toas)):
        raise ValueError("toas must be finite")

    simulation = make_validated_simulation(
        metadata_yaml=metadata_path,
        nchunks=nchunks,
        ntime=ntime,
        dm=dm,
        snr=snr,
        width_ms=width_ms,
        toa=requested_toas.tolist(),
        subband_lo_MHz=freq_lo_MHz,
        subband_hi_MHz=freq_hi_MHz,
    )
    requested_interval = validate_requested_frequency_interval(
        simulation.xmd, freq_lo_MHz, freq_hi_MHz
    )
    coverage = realized_channel_coverage(
        simulation.xmd, freq_lo_MHz, freq_hi_MHz
    )
    preflight = validate_simulation_frequency_interval(
        simulation,
        freq_lo_MHz,
        freq_hi_MHz,
        coverage,
        expected_bursts=len(requested_toas),
    )

    config = DedispersionConfig.from_yaml(config_path)
    od = OfflineDedisperser(config, cuda_device_id=device)
    decoded_parts = {method: [] for method in BENCHMARK_METHODS}
    containment_checks = []
    geometry = None
    dm_audit = None
    placement = None
    itree = None

    with cp.cuda.Device(device):
        for frame in simulation.iter_frames():
            with od.dedisperse(frame) as outputs:
                if geometry is None:
                    if int(ntime) != int(od.nt_in):
                        raise ValueError(
                            f"ntime={ntime} must match dedispersion plan nt_in={od.nt_in}"
                        )
                    if not np.isclose(
                        float(simulation.time_sample_ms),
                        float(od.time_sample_ms),
                        rtol=1.0e-12,
                        atol=1.0e-12,
                    ):
                        raise ValueError(
                            "simulation and dedispersion time-sample durations disagree"
                        )
                    dm_audit = validate_dm_for_peakfinder(
                        od.plan,
                        dm,
                        dm_reach=dm_reach,
                        intended_tree=expected_tree,
                    )
                    itree = dm_audit["tree"]
                    dcores = producer_dcore_array(od.plan, od.dd.Dcores)
                    geometry = build_peakfinder_geometry(
                        od.plan,
                        itree,
                        dcores=dcores,
                        time_sample_s=od.time_sample_ms / 1.0e3,
                        nt_in=od.nt_in,
                        reference_freq_mhz=simulation.reference_frequency_MHz,
                        dm_reach=dm_reach,
                        waist_bins=waist_bins,
                    )
                    if output_map_index is not None:
                        placement = validate_safe_toa_placement(
                            simulation,
                            geometry=geometry,
                            output_map_index=output_map_index,
                            expected_offsets_s=requested_toas - requested_toas[0],
                        )

                snr_map = cp.asarray(outputs.out_max[itree][0])
                argmax_map = cp.asarray(outputs.out_argmax[itree][0])
                expected_shape = (geometry.ndm, geometry.nt)
                if snr_map.shape != expected_shape or argmax_map.shape != expected_shape:
                    raise RuntimeError(
                        f"tree {itree} shapes {snr_map.shape}/{argmax_map.shape}, "
                        f"expected {expected_shape}"
                    )
                if argmax_map.dtype != cp.uint32:
                    raise RuntimeError(f"tree {itree} argmax dtype is {argmax_map.dtype}")
                batches = {
                    method: run_peakfinder(
                        method, snr_map, argmax_map, geometry, threshold
                    )
                    for method in BENCHMARK_METHODS
                }
                validate_candidate_set_containment(
                    batches, context=f"chunk {frame.time_chunk_index}, tree {itree}"
                )
                containment_checks.append(int(frame.time_chunk_index))
                for method, candidates in batches.items():
                    decoded_parts[method].append(decode_candidates(
                        od.plan,
                        candidates,
                        dcores=dcores,
                        itree=itree,
                        time_chunk_index=frame.time_chunk_index,
                        ntime=od.nt_in,
                        time_sample_s=od.time_sample_ms / 1.0e3,
                    ))

    if geometry is None:
        raise RuntimeError("simulation yielded no frames")
    with cp.cuda.Device(device):
        geometry_diagnostics = geometry.diagnostics()
    decoded = {
        method: concatenate_decoded(decoded_parts[method])
        for method in BENCHMARK_METHODS
    }
    time_sample_s = od.time_sample_ms / 1.0e3
    information = {
        "tree": itree,
        "dm_step": geometry.dm_step,
        "dm_validation": dm_audit,
        "time_bin_s": geometry.time_step_s,
        "geometry": geometry_diagnostics,
        "plan": plan_summary(od.plan, time_sample_s, dcores=dcores),
        "dcores": dcores.tolist(),
        "argmax_encoding": ARGMAX_ENCODING,
        "time_sample_ms": float(od.time_sample_ms),
        "reference_frequency_MHz": float(simulation.reference_frequency_MHz),
        "burst_summaries": [dict(summary) for summary in simulation.burst_summaries],
        "width_semantics": WIDTH_SEMANTICS,
        "decoded_timestamp": decoded_timestamp_summary(od.plan, itree, time_sample_s, dcores=dcores),
        "requested_frequency_interval": requested_interval,
        "realized_channel_coverage": {
            key: value for key, value in coverage.items()
            if key != "active_channel_indices"
        },
        "preflight": preflight,
        "temporal_placement": placement,
        "candidate_set_containment": True,
        "containment_checked_chunks": containment_checks,
        "methods_received_identical_maps": True,
    }
    return decoded, information


def match_one_inband(
    candidates, *, injected_dm, injected_toa_ref_s, reference_freq_MHz,
    match_freq_MHz, toa_tolerance_s,
):
    """Associate one injected burst by in-band TOA, with deterministic tie-breaking."""
    if not np.isfinite(toa_tolerance_s) or toa_tolerance_s <= 0:
        raise ValueError("toa_tolerance_s must be finite and positive")
    injected_toa_match = float(np.asarray(toa_at_frequency(
        injected_toa_ref_s,
        injected_dm,
        reference_freq_MHz,
        match_freq_MHz,
    )))
    candidate_toas_match = np.asarray(toa_at_frequency(
        candidates["toa_ref_s"],
        candidates["dm"],
        reference_freq_MHz,
        match_freq_MHz,
    ), dtype=np.float64)
    residuals = np.abs(candidate_toas_match - injected_toa_match)
    valid = np.flatnonzero(residuals <= toa_tolerance_s)
    if not len(valid):
        return None, injected_toa_match, candidate_toas_match
    index = min(
        (
            float(residuals[i]),
            -float(candidates["snr"][i]),
            int(i),
        )
        for i in valid
    )[2]
    return index, injected_toa_match, candidate_toas_match


def match_two_inband(
    candidates, *, injected_dm, injected_toas_ref_s, reference_freq_MHz,
    match_freq_MHz, toa_tolerance_s,
):
    """Maximum-cardinality one-to-one matching using only in-band TOA residual."""
    if not np.isfinite(toa_tolerance_s) or toa_tolerance_s <= 0:
        raise ValueError("toa_tolerance_s must be finite and positive")
    injected_toas_match = np.asarray(toa_at_frequency(
        injected_toas_ref_s, injected_dm, reference_freq_MHz, match_freq_MHz
    ), dtype=np.float64)
    if injected_toas_match.shape != (2,):
        raise ValueError("injected_toas_ref_s must contain exactly two TOAs")
    candidate_toas_match = np.asarray(toa_at_frequency(
        candidates["toa_ref_s"], candidates["dm"],
        reference_freq_MHz, match_freq_MHz
    ), dtype=np.float64)
    residuals = [np.abs(candidate_toas_match - toa) for toa in injected_toas_match]
    valid = [np.flatnonzero(residual <= toa_tolerance_s) for residual in residuals]

    assignments = [None, None]
    pairs = []
    for left in valid[0]:
        for right in valid[1]:
            if left == right:
                continue
            pairs.append((
                float(residuals[0][left] + residuals[1][right]),
                -float(candidates["snr"][left] + candidates["snr"][right]),
                int(left), int(right),
            ))
    if pairs:
        _, _, assignments[0], assignments[1] = min(pairs)
        return assignments, injected_toas_match, candidate_toas_match

    singles = []
    for iburst in range(2):
        for index in valid[iburst]:
            singles.append((
                float(residuals[iburst][index]),
                -float(candidates["snr"][index]),
                iburst,
                int(index),
            ))
    if singles:
        _, _, iburst, index = min(singles)
        assignments[iburst] = index
    return assignments, injected_toas_match, candidate_toas_match


def validate_matched_cardinality_containment(assignments_by_method, *, context):
    """Validate the full-band method schema and return its matched count.

    The historical function name remains for checkpoint-reader compatibility;
    with one active peakfinder there is no cross-method containment claim.
    """
    if set(assignments_by_method) != set(BENCHMARK_METHODS):
        raise ValueError(
            f"{context}: assignment methods {sorted(assignments_by_method)} do not "
            f"match {list(BENCHMARK_METHODS)}"
        )
    counts = {
        method: sum(index is not None for index in assignments_by_method[method])
        for method in BENCHMARK_METHODS
    }
    return counts


def candidate_values(
    candidates, index, *, candidate_toas_match, injected_dm,
    injected_toa_match_s, dm_step, injected_fmin=None, injected_fmax=None,
):
    """Return auditable recovered values; exact plan bounds are optional."""
    if (injected_fmin is None) != (injected_fmax is None):
        raise ValueError("injected_fmin and injected_fmax must both be supplied or omitted")
    fields = {
        "idm": "", "itime": "", "time_chunk_index": "", "tree": "",
        "snr": "", "dm": "", "toa_ref_s": "", "toa_inband_s": "",
        "toa_residual_ms": "", "dm_residual": "", "dm_residual_bins": "",
        "width_ms": "", "profile": "", "argmax_token": "", "fmin": "",
        "fmax": "", "freq_lo_MHz": "", "freq_hi_MHz": "",
        "subband_matches_injected": "",
    }
    if index is None:
        return fields
    i = int(index)
    subband_matches = ""
    if injected_fmin is not None:
        subband_matches = bool(
            int(candidates["fmin"][i]) == int(injected_fmin)
            and int(candidates["fmax"][i]) == int(injected_fmax)
        )
    fields.update({
        "idm": int(candidates["idm"][i]),
        "itime": int(candidates["itime"][i]),
        "time_chunk_index": int(candidates["time_chunk_index"][i]),
        "tree": int(candidates["tree"][i]),
        "snr": float(candidates["snr"][i]),
        "dm": float(candidates["dm"][i]),
        "toa_ref_s": float(candidates["toa_ref_s"][i]),
        "toa_inband_s": float(candidate_toas_match[i]),
        "toa_residual_ms": 1.0e3 * float(candidate_toas_match[i] - injected_toa_match_s),
        "dm_residual": float(candidates["dm"][i] - injected_dm),
        "dm_residual_bins": float((candidates["dm"][i] - injected_dm) / dm_step),
        "width_ms": 1.0e3 * float(candidates["width_s"][i]),
        "profile": int(candidates["profile"][i]),
        "argmax_token": int(candidates["argmax_token"][i]),
        "fmin": int(candidates["fmin"][i]),
        "fmax": int(candidates["fmax"][i]),
        "freq_lo_MHz": float(candidates["freq_lo_MHz"][i]),
        "freq_hi_MHz": float(candidates["freq_hi_MHz"][i]),
        "subband_matches_injected": subband_matches,
    })
    return fields

def validate_gpu_times(rows):
    for index, row in enumerate(rows):
        value = float(row["gpu_time_ms"])
        if not np.isfinite(value) or value < 0:
            raise ValueError(f"runtime row {index} has invalid gpu_time_ms={value!r}")


def serial_all_tree_gpu_times(rows, plan_tree_ids):
    """Build serial native all-tree sums only from complete matching iterations."""
    plan_tree_ids = tuple(sorted(int(value) for value in plan_tree_ids))
    native = [row for row in rows if int(row["scale_factor"]) == 1]
    validate_gpu_times(native)
    present_trees = {int(row["tree"]) for row in native}
    if present_trees != set(plan_tree_ids):
        return []
    grouped = defaultdict(dict)
    for row in native:
        key = (row["method"], int(row["iteration"]))
        tree = int(row["tree"])
        if tree in grouped[key]:
            raise ValueError(f"duplicate native runtime row for {key}, tree {tree}")
        grouped[key][tree] = float(row["gpu_time_ms"])
    iterations_by_method = {}
    for method in BENCHMARK_METHODS:
        iterations = {iteration for (name, iteration), values in grouped.items()
                      if name == method and set(values) == set(plan_tree_ids)}
        iterations_by_method[method] = iterations
    if not iterations_by_method or len({frozenset(v) for v in iterations_by_method.values()}) != 1:
        return []
    common = next(iter(iterations_by_method.values()))
    if not common:
        return []
    return [
        {
            "method": method,
            "iteration": iteration,
            "gpu_time_ms": sum(grouped[(method, iteration)].values()),
        }
        for method in BENCHMARK_METHODS for iteration in sorted(common)
    ]
