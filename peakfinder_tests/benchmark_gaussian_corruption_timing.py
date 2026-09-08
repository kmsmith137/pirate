#!/usr/bin/env python3
"""Monte Carlo timing of PIRATE's post-peakfinder candidate pipeline.

Each trial creates one independent Gaussian-white-noise realization and one
independent mixture of clipped two-dimensional Gaussians for every beam.  The
same noise and mixture are reused across the requested corruption percentages;
only one non-negative amplitude scale per beam changes.  Peakfinding uses the
active production streaming extractor.  Only the complete decoder and grouper
calls are timed, with a CUDA-stream synchronization on both sides.

CuPy and production PIRATE imports intentionally remain inside GPU-facing
functions so generation, calibration, schema, resume, and notebook tests remain
usable on CPU-only hosts.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import os
import platform
import sys
import tempfile
import time
from dataclasses import dataclass, fields
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

import numpy as np
import yaml

from peakfinder_tests import benchmark_peakfinder_batch_timing as batch_benchmark


SCHEMA_NAME = "pirate-gaussian-corruption-timing"
SCHEMA_VERSION = 1
CORRUPTION_MODEL = "gaussian_mixture"
METHOD = "full_band"
SNR_DTYPE = np.dtype(np.float16)
WORKING_DTYPE = np.dtype(np.float32)
ARGMAX_DTYPE = np.dtype(np.uint32)

REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_CONFIG = REPOSITORY_ROOT / "configs/dedispersion/chord_sb2_et.yml"
DEFAULT_RESULTS_DIR = (
    Path(__file__).resolve().parent / "results_gaussian_corruption_timing"
)
NOTEBOOK_PATH = (
    Path(__file__).resolve().parent / "analyze_gaussian_corruption_timing.ipynb"
)

EXPECTED_SHAPES = batch_benchmark.EXPECTED_SHAPES
PIXELS_PER_BEAM = batch_benchmark.EXPECTED_PIXELS_PER_BEAM
DEFAULT_TOTAL_BEAMS = 60
DEFAULT_BEAM_BATCH_SIZE = 60
DEFAULT_THRESHOLD = 10.0
DEFAULT_DM_REACH = 8
DEFAULT_WAIST_BINS = 1
DEFAULT_TRIALS = 50
DEFAULT_DEVICE = 0
DEFAULT_BASE_SEED = 20260825
DEFAULT_PERCENTAGES = (0.0, 0.001, 0.003, 0.01, 0.03, 0.1, 0.3, 1.0)
DEFAULT_GAUSSIANS_MIN = 24
DEFAULT_GAUSSIANS_MAX = 48
DEFAULT_DM_WIDTH_MIN = 4.0
DEFAULT_DM_WIDTH_MAX = 32.0
DEFAULT_TIME_WIDTH_MIN = 1.0
DEFAULT_TIME_WIDTH_MAX = 4.0
DEFAULT_MAX_ABS_CORRELATION = 0.75
DEFAULT_RELATIVE_AMPLITUDE_MIN = 0.5
DEFAULT_RELATIVE_AMPLITUDE_MAX = 2.0
DEFAULT_RENDER_SIGMA_CUTOFF = 8.0
DEFAULT_TEMPLATE_ATTEMPT_LIMIT = 8
DEFAULT_CALIBRATION_PIXEL_TOLERANCE = 1
DEFAULT_CALIBRATION_MAX_ITERATIONS = 32
DEFAULT_CANDIDATE_SAFETY_LIMIT = 5_000
DEFAULT_DM_TOLERANCE_BINS = 1.5
DEFAULT_TIME_PADDING_BINS = 1.0
TARGET_CHUNK_INDEX = 1_000_000


TRIAL_FIELDS = (
    "schema_version",
    "campaign_id",
    "trial",
    "seed",
    "total_beams",
    "beam_batch_size",
    "threshold",
    "method",
    "dm_reach",
    "waist_bins",
    "target_corruption_percent",
    "achieved_corruption_percent",
    "target_pixels_above_threshold",
    "achieved_pixels_above_threshold",
    "pixel_count_error",
    "max_abs_beam_pixel_count_error",
    "per_beam_calibration_json",
    "total_gaussians",
    "gaussians_per_beam_min",
    "gaussians_per_beam_median",
    "gaussians_per_beam_max",
    "gaussian_dm_width_bins_min",
    "gaussian_dm_width_bins_median",
    "gaussian_dm_width_bins_max",
    "gaussian_time_width_bins_min",
    "gaussian_time_width_bins_median",
    "gaussian_time_width_bins_max",
    "gaussian_correlation_min",
    "gaussian_correlation_median",
    "gaussian_correlation_max",
    "gaussian_relative_amplitude_min",
    "gaussian_relative_amplitude_median",
    "gaussian_relative_amplitude_max",
    "template_attempt_per_beam_min",
    "template_attempt_per_beam_median",
    "template_attempt_per_beam_max",
    "total_pixels_above_threshold",
    "pixels_above_threshold_per_beam_min",
    "pixels_above_threshold_per_beam_median",
    "pixels_above_threshold_per_beam_max",
    "total_peakfinder_candidates",
    "peakfinder_candidates_per_beam_min",
    "peakfinder_candidates_per_beam_median",
    "peakfinder_candidates_per_beam_max",
    "peakfinder_survival_fraction",
    "decoder_wall_ms",
    "decoder_candidates_per_second",
    "total_decoded_candidates",
    "grouper_wall_ms",
    "grouper_candidates_per_second",
    "total_grouped_events",
    "grouped_events_per_beam_min",
    "grouped_events_per_beam_median",
    "grouped_events_per_beam_max",
    "decoder_plus_grouper_wall_ms",
    "post_peakfinder_load_fraction",
    "candidate_safety_limit",
    "status",
    "failure_reason",
)

SUMMARY_METRICS = (
    "total_pixels_above_threshold",
    "total_peakfinder_candidates",
    "peakfinder_survival_fraction",
    "decoder_wall_ms",
    "total_decoded_candidates",
    "grouper_wall_ms",
    "total_grouped_events",
    "decoder_plus_grouper_wall_ms",
    "post_peakfinder_load_fraction",
)
SUMMARY_STATISTICS = ("median", "minimum", "maximum", "q25", "q75", "iqr")
SUMMARY_FIELDS = (
    "schema_version",
    "campaign_id",
    "target_corruption_percent",
    "recorded_configurations",
    "completed_trials",
    "skipped_trials",
    "failed_trials",
    *tuple(
        f"{metric}_{statistic}"
        for metric in SUMMARY_METRICS
        for statistic in SUMMARY_STATISTICS
    ),
)


@dataclass(frozen=True)
class GaussianConfig:
    """Configured distributions for one beam's Gaussian mixture."""

    count_min: int = DEFAULT_GAUSSIANS_MIN
    count_max: int = DEFAULT_GAUSSIANS_MAX
    dm_width_min: float = DEFAULT_DM_WIDTH_MIN
    dm_width_max: float = DEFAULT_DM_WIDTH_MAX
    time_width_min: float = DEFAULT_TIME_WIDTH_MIN
    time_width_max: float = DEFAULT_TIME_WIDTH_MAX
    max_abs_correlation: float = DEFAULT_MAX_ABS_CORRELATION
    relative_amplitude_min: float = DEFAULT_RELATIVE_AMPLITUDE_MIN
    relative_amplitude_max: float = DEFAULT_RELATIVE_AMPLITUDE_MAX
    render_sigma_cutoff: float = DEFAULT_RENDER_SIGMA_CUTOFF


@dataclass(frozen=True)
class GaussianComponent:
    """One tree-local correlated Gaussian, expressed in native map bins."""

    tree_index: int
    centre_dm: float
    centre_time: float
    sigma_dm: float
    sigma_time: float
    correlation: float
    relative_amplitude: float


@dataclass(frozen=True)
class BeamMorphology:
    """One beam's reusable float32 noise and non-negative template."""

    beam_id: int
    noise_by_tree: tuple[np.ndarray, ...]
    template_by_tree: tuple[np.ndarray, ...]
    components: tuple[GaussianComponent, ...]
    noise_seed: int
    template_seed: int
    template_attempt: int
    validated_target_counts: tuple[int, ...] = ()


@dataclass(frozen=True)
class TrialMorphology:
    trial: int
    seed: int
    beams: tuple[BeamMorphology, ...]


@dataclass(frozen=True)
class BeamCalibration:
    beam_id: int
    target_count: int
    achieved_count: int
    pixel_count_error: int
    scale: float
    iterations: int
    maps_by_tree: tuple[np.ndarray, ...]


class MorphologyGenerationError(RuntimeError):
    """Raised after deterministic template regeneration is exhausted."""

    def __init__(self, beam_id: int, attempts: int, attainable_pixels: int,
                 target_pixels: int):
        self.beam_id = int(beam_id)
        self.attempts = int(attempts)
        self.attainable_pixels = int(attainable_pixels)
        self.target_pixels = int(target_pixels)
        super().__init__(
            f"beam {beam_id} failed calibration after {attempts} deterministic "
            f"template attempts; best positive support was {attainable_pixels} "
            f"pixels and the largest requested target was {target_pixels}"
        )


def _exact_positive_integer(value: Any, name: str) -> int:
    return batch_benchmark._exact_positive_integer(value, name)


def _exact_nonnegative_integer(value: Any, name: str) -> int:
    return batch_benchmark._exact_nonnegative_integer(value, name)


def _finite_float(value: Any, name: str) -> float:
    result = float(value)
    if not math.isfinite(result):
        raise ValueError(f"{name} must be finite")
    return result


def canonical_percentage(value: Any) -> str:
    """Stable percentage spelling used by signatures and resume keys."""

    percentage = _finite_float(value, "corruption percentage")
    if percentage < 0.0 or percentage > 100.0:
        raise ValueError("corruption percentages must lie in [0, 100]")
    return format(percentage, ".12g")


def validate_percentage_grid(values: Sequence[Any]) -> tuple[float, ...]:
    percentages = tuple(float(canonical_percentage(value)) for value in values)
    if not percentages:
        raise ValueError("at least one corruption percentage is required")
    if len(set(map(canonical_percentage, percentages))) != len(percentages):
        raise ValueError("corruption percentages must be unique")
    if any(right <= left for left, right in zip(percentages, percentages[1:])):
        raise ValueError("corruption percentages must be strictly increasing")
    return percentages


def target_count_from_percentage(percentage: Any,
                                 pixels_per_beam: int = PIXELS_PER_BEAM) -> int:
    """Round one beam's requested percentage to the nearest integer pixel."""

    p = float(canonical_percentage(percentage))
    pixels = _exact_positive_integer(pixels_per_beam, "pixels_per_beam")
    return int(round((p / 100.0) * pixels))


def derive_seed(base_seed: Any, trial: Any, beam_id: Any, purpose: str,
                attempt: Any = 0) -> int:
    """Derive a stable, independent uint64 stream key."""

    base = _exact_nonnegative_integer(base_seed, "base_seed")
    trial_index = _exact_nonnegative_integer(trial, "trial")
    beam = _exact_nonnegative_integer(beam_id, "beam_id")
    attempt_index = _exact_nonnegative_integer(attempt, "attempt")
    payload = json.dumps(
        [SCHEMA_NAME, base, trial_index, beam, str(purpose), attempt_index],
        separators=(",", ":"),
    ).encode("utf-8")
    return int.from_bytes(hashlib.sha256(payload).digest()[:8], "little")


def _validate_gaussian_config(config: GaussianConfig) -> GaussianConfig:
    if not isinstance(config, GaussianConfig):
        raise TypeError("config must be GaussianConfig")
    if config.count_min <= 0 or config.count_max < config.count_min:
        raise ValueError("Gaussian count range is invalid")
    for low, high, name in (
        (config.dm_width_min, config.dm_width_max, "DM width"),
        (config.time_width_min, config.time_width_max, "time width"),
        (config.relative_amplitude_min, config.relative_amplitude_max,
         "relative amplitude"),
    ):
        if not (math.isfinite(low) and math.isfinite(high) and 0.0 < low <= high):
            raise ValueError(f"{name} range is invalid")
    if not (math.isfinite(config.max_abs_correlation)
            and 0.0 <= config.max_abs_correlation < 1.0):
        raise ValueError("maximum absolute correlation must lie in [0, 1)")
    if not (math.isfinite(config.render_sigma_cutoff)
            and config.render_sigma_cutoff > 0.0):
        raise ValueError("render sigma cutoff must be positive")
    return config


def _log_uniform(rng: np.random.Generator, low: float, high: float) -> float:
    if low == high:
        return float(low)
    return float(np.exp(rng.uniform(np.log(low), np.log(high))))


def _uniform_coordinate(rng: np.random.Generator, length: int) -> float:
    """Draw a continuous native-bin centre, including the one-bin edge case."""

    if length <= 0:
        raise ValueError("map dimensions must be positive")
    return 0.0 if length == 1 else float(rng.uniform(0.0, length - 1))


def sample_gaussian_components(
        specs: Sequence[Any], rng: np.random.Generator,
        config: GaussianConfig) -> tuple[GaussianComponent, ...]:
    """Sample one independent beam morphology from documented distributions."""

    config = _validate_gaussian_config(config)
    if not specs:
        raise ValueError("at least one tree specification is required")
    pixel_counts = np.asarray(
        [int(spec.ndm) * int(spec.ntime) for spec in specs], dtype=np.float64
    )
    if np.any(pixel_counts <= 0):
        raise ValueError("tree shapes must be positive")
    probabilities = pixel_counts / np.sum(pixel_counts)
    count = int(rng.integers(config.count_min, config.count_max + 1))
    components = []
    for _ in range(count):
        tree_index = int(rng.choice(len(specs), p=probabilities))
        spec = specs[tree_index]
        components.append(GaussianComponent(
            tree_index=tree_index,
            centre_dm=_uniform_coordinate(rng, int(spec.ndm)),
            centre_time=_uniform_coordinate(rng, int(spec.ntime)),
            sigma_dm=_log_uniform(
                rng, config.dm_width_min, config.dm_width_max
            ),
            sigma_time=_log_uniform(
                rng, config.time_width_min, config.time_width_max
            ),
            correlation=float(rng.uniform(
                -config.max_abs_correlation, config.max_abs_correlation
            )),
            relative_amplitude=_log_uniform(
                rng, config.relative_amplitude_min,
                config.relative_amplitude_max,
            ),
        ))
    return tuple(components)


def add_gaussian_component(
        target: np.ndarray, component: GaussianComponent,
        sigma_cutoff: float = DEFAULT_RENDER_SIGMA_CUTOFF) -> None:
    """Add a clipped bivariate Gaussian without periodic wraparound."""

    if target.ndim != 2 or target.dtype != WORKING_DTYPE:
        raise ValueError("Gaussian target must be a two-dimensional float32 map")
    cutoff = _finite_float(sigma_cutoff, "sigma_cutoff")
    if cutoff <= 0.0:
        raise ValueError("sigma_cutoff must be positive")
    rho = float(component.correlation)
    if not (-1.0 < rho < 1.0):
        raise ValueError("Gaussian correlation must lie strictly between -1 and 1")
    if (component.sigma_dm <= 0.0 or component.sigma_time <= 0.0
            or component.relative_amplitude <= 0.0):
        raise ValueError("Gaussian widths and amplitude must be positive")

    ndm, ntime = target.shape
    dm_radius = int(math.ceil(cutoff * component.sigma_dm))
    time_radius = int(math.ceil(cutoff * component.sigma_time))
    dm_start = max(0, int(math.floor(component.centre_dm)) - dm_radius)
    dm_stop = min(ndm, int(math.floor(component.centre_dm)) + dm_radius + 1)
    time_start = max(0, int(math.floor(component.centre_time)) - time_radius)
    time_stop = min(
        ntime, int(math.floor(component.centre_time)) + time_radius + 1
    )
    if dm_start >= dm_stop or time_start >= time_stop:
        return

    dm = (
        np.arange(dm_start, dm_stop, dtype=np.float32)
        - np.float32(component.centre_dm)
    ) / np.float32(component.sigma_dm)
    time_axis = (
        np.arange(time_start, time_stop, dtype=np.float32)
        - np.float32(component.centre_time)
    ) / np.float32(component.sigma_time)
    dd = dm[:, None]
    tt = time_axis[None, :]
    denominator = np.float32(1.0 - rho * rho)
    quadratic = (dd * dd - np.float32(2.0 * rho) * dd * tt + tt * tt) / denominator
    patch = np.exp(np.float32(-0.5) * quadratic).astype(np.float32, copy=False)
    patch *= np.float32(component.relative_amplitude)
    target[dm_start:dm_stop, time_start:time_stop] += patch


def render_gaussian_template(
        specs: Sequence[Any], components: Sequence[GaussianComponent],
        sigma_cutoff: float = DEFAULT_RENDER_SIGMA_CUTOFF,
        ) -> tuple[np.ndarray, ...]:
    """Render a non-negative, ragged, tree-local float32 mixture."""

    maps = tuple(
        np.zeros((int(spec.ndm), int(spec.ntime)), dtype=WORKING_DTYPE)
        for spec in specs
    )
    for component in components:
        if component.tree_index < 0 or component.tree_index >= len(maps):
            raise ValueError("Gaussian component references an invalid tree")
        add_gaussian_component(
            maps[component.tree_index], component, sigma_cutoff=sigma_cutoff
        )
    if any(np.any(~np.isfinite(array)) or np.any(array < 0.0) for array in maps):
        raise AssertionError("Gaussian template must remain finite and non-negative")
    return maps


def generate_white_noise(
        specs: Sequence[Any], seed: Any, threshold: float,
        ) -> tuple[np.ndarray, ...]:
    """Generate one beam's independent standard Gaussian float32 maps."""

    threshold_value = _finite_float(threshold, "threshold")
    rng = np.random.default_rng(_exact_nonnegative_integer(seed, "seed"))
    maps = tuple(
        rng.standard_normal(
            (int(spec.ndm), int(spec.ntime)), dtype=np.float32
        )
        for spec in specs
    )
    if any(array.dtype != WORKING_DTYPE or np.any(~np.isfinite(array)) for array in maps):
        raise AssertionError("white-noise generation produced invalid values")
    if any(np.any(array.astype(SNR_DTYPE) >= threshold_value) for array in maps):
        raise AssertionError("clean float16 white noise unexpectedly crosses threshold")
    return maps


def _threshold_rounding_boundary(threshold: float) -> np.float32:
    """Approximate the float32 boundary that rounds to the first valid float16."""

    threshold_value = _finite_float(threshold, "threshold")
    upper = np.float16(threshold_value)
    if float(upper) < threshold_value:
        upper = np.nextafter(upper, np.float16(np.inf), dtype=np.float16)
    lower = np.nextafter(upper, np.float16(-np.inf), dtype=np.float16)
    return np.float32(
        (np.float32(lower) + np.float32(upper)) * np.float32(0.5)
    )


def maps_at_scale(
        noise_by_tree: Sequence[np.ndarray],
        template_by_tree: Sequence[np.ndarray], scale: float,
        ) -> tuple[np.ndarray, ...]:
    """Apply one common float32 scale and perform the production float16 cast."""

    scale_value = float(scale)
    if (not math.isfinite(scale_value) or scale_value < 0.0
            or scale_value > float(np.finfo(np.float32).max)):
        raise ValueError("corruption amplitude scale must be finite and non-negative")
    scale32 = np.float32(scale_value)
    if len(noise_by_tree) != len(template_by_tree):
        raise ValueError("noise and template tree counts differ")
    result = []
    for noise, template in zip(noise_by_tree, template_by_tree):
        if (noise.shape != template.shape or noise.dtype != WORKING_DTYPE
                or template.dtype != WORKING_DTYPE):
            raise ValueError("noise/template map shape or dtype mismatch")
        with np.errstate(over="ignore", invalid="ignore"):
            working = np.add(
                noise, np.multiply(template, scale32, dtype=np.float32),
                dtype=np.float32,
            )
            production = working.astype(SNR_DTYPE)
        if np.any(~np.isfinite(working)) or np.any(~np.isfinite(production)):
            raise ValueError(
                "corruption scale would create non-finite production S/N pixels"
            )
        result.append(production)
    return tuple(result)


def count_above_threshold(maps: Sequence[np.ndarray], threshold: float) -> int:
    threshold_value = _finite_float(threshold, "threshold")
    return sum(int(np.count_nonzero(array >= threshold_value)) for array in maps)


def calibrate_beam(
        morphology: BeamMorphology, target_count: Any, threshold: float,
        *, pixel_tolerance: Any = DEFAULT_CALIBRATION_PIXEL_TOLERANCE,
        max_iterations: Any = DEFAULT_CALIBRATION_MAX_ITERATIONS,
        ) -> BeamCalibration:
    """Calibrate one beam with an order statistic and bounded bisection.

    The objective is evaluated only after the final float16 cast.  The returned
    solution is the best evaluated count; a strict per-beam tolerance is enforced.
    Continuous Gaussian parameters normally make the requested count exact, while
    the tolerance permits the nearest attainable count at a quantized tie.
    """

    target = _exact_nonnegative_integer(target_count, "target_count")
    tolerance = _exact_nonnegative_integer(pixel_tolerance, "pixel_tolerance")
    iteration_limit = _exact_positive_integer(max_iterations, "max_iterations")
    pixels = sum(array.size for array in morphology.noise_by_tree)
    if target > pixels:
        raise ValueError("target_count exceeds the beam pixel count")

    if target == 0:
        maps = maps_at_scale(
            morphology.noise_by_tree, morphology.template_by_tree, 0.0
        )
        achieved = count_above_threshold(maps, threshold)
        if achieved != 0:
            raise AssertionError("zero corruption must contain zero threshold crossings")
        return BeamCalibration(
            morphology.beam_id, target, achieved, achieved - target,
            0.0, 0, maps,
        )

    boundary = _threshold_rounding_boundary(threshold)
    ratio_parts = []
    for noise, template in zip(
            morphology.noise_by_tree, morphology.template_by_tree):
        positive = template > 0.0
        if np.any(positive):
            ratios = (
                (boundary - noise[positive]).astype(np.float64)
                / template[positive].astype(np.float64)
            )
            ratio_parts.append(ratios[ratios >= 0.0])
    if not ratio_parts:
        raise ValueError("Gaussian template has no positive support")
    ratios = np.concatenate(ratio_parts)
    if ratios.size < target:
        raise ValueError(
            f"template has only {ratios.size} potentially attainable pixels, "
            f"below target {target}"
        )
    estimate = float(np.partition(ratios, target - 1)[target - 1])
    if not math.isfinite(estimate) or estimate < 0.0:
        raise ValueError("order-statistic calibration estimate is invalid")

    evaluations: list[tuple[int, int, float, tuple[np.ndarray, ...]]] = []

    def evaluate(scale: float) -> tuple[int, tuple[np.ndarray, ...]]:
        rendered = maps_at_scale(
            morphology.noise_by_tree, morphology.template_by_tree, scale
        )
        count = count_above_threshold(rendered, threshold)
        evaluations.append((abs(count - target), count, float(np.float32(scale)), rendered))
        return count, rendered

    initial_count, _ = evaluate(estimate)
    if initial_count < target:
        low = float(np.float32(estimate))
        high = max(
            float(np.nextafter(np.float32(low), np.float32(np.inf))),
            low * 1.0001 + np.finfo(np.float32).tiny,
        )
        high_count, _ = evaluate(high)
        expansions = 0
        while high_count < target and expansions < iteration_limit:
            low = high
            high = high * 2.0 if high > 0.0 else 1.0
            high_count, _ = evaluate(high)
            expansions += 1
        if high_count < target:
            raise ValueError("bounded calibration could not bracket the target count")
    elif initial_count > target:
        high = float(np.float32(estimate))
        low = max(0.0, high * 0.9999)
        low_count, _ = evaluate(low)
        expansions = 0
        while low_count > target and expansions < iteration_limit:
            high = low
            low *= 0.5
            low_count, _ = evaluate(low)
            expansions += 1
        if low_count > target:
            raise ValueError("bounded calibration could not bracket the target count")
    else:
        low = high = float(np.float32(estimate))

    if initial_count != target:
        for iteration in range(iteration_limit):
            low32 = np.float32(low)
            high32 = np.float32(high)
            if low32 == high32:
                break
            midpoint = float(np.float32((float(low32) + float(high32)) * 0.5))
            if midpoint == float(low32) or midpoint == float(high32):
                break
            midpoint_count, _ = evaluate(midpoint)
            if midpoint_count < target:
                low = midpoint
            elif midpoint_count > target:
                high = midpoint
            else:
                break

    # Error first, then smaller count error sign magnitude through count, then
    # smaller scale gives a deterministic nearest-attainable tie break.
    best = min(evaluations, key=lambda item: (item[0], abs(item[1] - target), item[2]))
    error, achieved, scale, maps = best
    if error > tolerance:
        raise ValueError(
            f"nearest evaluated float16 count {achieved} differs from target "
            f"{target} by {error} pixels (tolerance {tolerance})"
        )
    verified = count_above_threshold(maps, threshold)
    if verified != achieved:
        raise AssertionError("calibration count changed after final validation")
    return BeamCalibration(
        morphology.beam_id, target, achieved, achieved - target,
        scale, len(evaluations), maps,
    )


def generate_beam_morphology(
        specs: Sequence[Any], *, base_seed: int, trial: int, beam_id: int,
        gaussian_config: GaussianConfig, threshold: float,
        maximum_target_count: int, attempt_limit: int,
        calibration_pixel_tolerance: int,
        calibration_max_iterations: int,
        required_target_counts: Sequence[int] | None = None,
        ) -> BeamMorphology:
    """Generate a beam, deterministically retrying inadequate templates."""

    attempts = _exact_positive_integer(attempt_limit, "attempt_limit")
    validation_targets = tuple(sorted(set(
        _exact_nonnegative_integer(value, "required_target_count")
        for value in (
            (maximum_target_count,)
            if required_target_counts is None else required_target_counts
        )
    )))
    if not validation_targets:
        validation_targets = (maximum_target_count,)
    if max(validation_targets) != maximum_target_count:
        raise ValueError(
            "maximum_target_count must equal the largest required target count"
        )
    noise_seed = derive_seed(base_seed, trial, beam_id, "white_noise")
    noise = generate_white_noise(specs, noise_seed, threshold)
    best_support = 0
    best_morphology: BeamMorphology | None = None
    best_score: tuple[int, int, int, int] | None = None
    for attempt in range(attempts):
        template_seed = derive_seed(
            base_seed, trial, beam_id, "gaussian_template", attempt
        )
        rng = np.random.default_rng(template_seed)
        components = sample_gaussian_components(specs, rng, gaussian_config)
        template = render_gaussian_template(
            specs, components, gaussian_config.render_sigma_cutoff
        )
        support = sum(int(np.count_nonzero(array > 0.0)) for array in template)
        best_support = max(best_support, support)
        provisional = BeamMorphology(
            beam_id=beam_id,
            noise_by_tree=noise,
            template_by_tree=template,
            components=components,
            noise_seed=noise_seed,
            template_seed=template_seed,
            template_attempt=attempt,
        )
        validated_targets: list[int] = []
        for target_count in validation_targets:
            if target_count > support:
                continue
            try:
                calibrate_beam(
                    provisional, target_count, threshold,
                    pixel_tolerance=calibration_pixel_tolerance,
                    max_iterations=calibration_max_iterations,
                )
            except ValueError:
                continue
            validated_targets.append(target_count)
        morphology = BeamMorphology(
            beam_id=beam_id,
            noise_by_tree=noise,
            template_by_tree=template,
            components=components,
            noise_seed=noise_seed,
            template_seed=template_seed,
            template_attempt=attempt,
            validated_target_counts=tuple(validated_targets),
        )
        score = (
            len(validated_targets),
            max(validated_targets, default=-1),
            support,
            -attempt,
        )
        if best_score is None or score > best_score:
            best_morphology = morphology
            best_score = score
        if len(validated_targets) == len(validation_targets):
            return morphology
    if best_morphology is not None:
        # Preserve paired comparisons for every attainable level.  The caller
        # records each target absent from this tuple as an explicit failed
        # configuration; no pixels or alternate morphology are substituted.
        return best_morphology
    raise MorphologyGenerationError(
        beam_id, attempts, best_support, maximum_target_count
    )


def generate_trial_morphology(
        specs: Sequence[Any], *, base_seed: int, trial: int, total_beams: int,
        gaussian_config: GaussianConfig, threshold: float,
        maximum_target_count: int, attempt_limit: int,
        calibration_pixel_tolerance: int,
        calibration_max_iterations: int,
        required_target_counts: Sequence[int] | None = None,
        ) -> TrialMorphology:
    """Generate all independent beams for one paired Monte Carlo trial."""

    beams = tuple(
        generate_beam_morphology(
            specs,
            base_seed=base_seed,
            trial=trial,
            beam_id=beam_id,
            gaussian_config=gaussian_config,
            threshold=threshold,
            maximum_target_count=maximum_target_count,
            attempt_limit=attempt_limit,
            calibration_pixel_tolerance=calibration_pixel_tolerance,
            calibration_max_iterations=calibration_max_iterations,
            required_target_counts=required_target_counts,
        )
        for beam_id in range(total_beams)
    )
    return TrialMorphology(
        trial=trial,
        seed=derive_seed(base_seed, trial, 0, "trial"),
        beams=beams,
    )


def calibrate_trial_morphology(
        morphology: TrialMorphology, specs: Sequence[Any], percentage: float,
        threshold: float, *, pixel_tolerance: int,
        max_iterations: int,
        ) -> tuple[tuple[np.ndarray, ...], tuple[BeamCalibration, ...]]:
    """Calibrate every beam independently and assemble ten beam-major maps."""

    target = target_count_from_percentage(percentage)
    tree_maps = [
        np.empty(
            (len(morphology.beams), int(spec.ndm), int(spec.ntime)),
            dtype=SNR_DTYPE,
        )
        for spec in specs
    ]
    calibrations = []
    for beam in morphology.beams:
        calibration = calibrate_beam(
            beam, target, threshold,
            pixel_tolerance=pixel_tolerance,
            max_iterations=max_iterations,
        )
        if abs(calibration.pixel_count_error) > pixel_tolerance:
            raise AssertionError("per-beam calibration exceeded its strict tolerance")
        for tree_index, source in enumerate(calibration.maps_by_tree):
            tree_maps[tree_index][beam.beam_id] = source
        calibrations.append(calibration)
    return tuple(tree_maps), tuple(calibrations)


def summarize_counts(values: Sequence[Any]) -> dict[str, float]:
    array = np.asarray(values, dtype=np.float64)
    if array.ndim != 1 or array.size == 0 or np.any(~np.isfinite(array)):
        raise ValueError("count summary requires a nonempty finite vector")
    return {
        "min": float(np.min(array)),
        "median": float(np.median(array)),
        "max": float(np.max(array)),
    }


def derive_post_peakfinder_metrics(
        decoder_wall_ms: float, grouper_wall_ms: float,
        decoded_candidates: int, chunk_duration_ms: float,
        ) -> dict[str, float]:
    decoder_ms = _finite_float(decoder_wall_ms, "decoder_wall_ms")
    grouper_ms = _finite_float(grouper_wall_ms, "grouper_wall_ms")
    duration = _finite_float(chunk_duration_ms, "chunk_duration_ms")
    candidates = _exact_nonnegative_integer(
        decoded_candidates, "decoded_candidates"
    )
    if decoder_ms < 0.0 or grouper_ms < 0.0 or duration <= 0.0:
        raise ValueError("timings must be non-negative and duration positive")
    combined = decoder_ms + grouper_ms
    return {
        "decoder_candidates_per_second": (
            0.0 if candidates == 0 or decoder_ms == 0.0
            else 1000.0 * candidates / decoder_ms
        ),
        "grouper_candidates_per_second": (
            0.0 if candidates == 0 or grouper_ms == 0.0
            else 1000.0 * candidates / grouper_ms
        ),
        "decoder_plus_grouper_wall_ms": combined,
        "post_peakfinder_load_fraction": combined / duration,
    }


def grouping_safety_status(
        candidate_count: Any, safety_limit: Any, allow_unsafe: bool,
        ) -> str:
    count = _exact_nonnegative_integer(candidate_count, "candidate_count")
    limit = _exact_positive_integer(safety_limit, "safety_limit")
    if count > limit and not bool(allow_unsafe):
        return "skipped_candidate_safety_limit"
    return "run"


def validate_target_maps(
        maps: Sequence[np.ndarray], specs: Sequence[Any], total_beams: int,
        threshold: float) -> np.ndarray:
    if len(maps) != len(specs):
        raise ValueError("one S/N map is required for every tree")
    counts = np.zeros(total_beams, dtype=np.int64)
    for tree_index, (array, spec) in enumerate(zip(maps, specs)):
        expected = (total_beams, int(spec.ndm), int(spec.ntime))
        if array.shape != expected or array.dtype != SNR_DTYPE:
            raise ValueError(
                f"tree {tree_index} S/N map must have shape {expected} and float16 dtype"
            )
        if np.any(~np.isfinite(array)):
            raise ValueError(f"tree {tree_index} S/N map contains non-finite values")
        counts += np.count_nonzero(array >= threshold, axis=(1, 2))
    return counts


def generate_valid_argmax_maps(
        specs: Sequence[Any], total_beams: int, base_seed: int,
        ) -> tuple[tuple[np.ndarray, ...], tuple[dict[str, int], ...]]:
    maps = []
    policy = []
    for spec in specs:
        multiplet = derive_seed(
            base_seed, 0, int(spec.tree_index), "argmax_multiplet"
        ) % int(spec.multiplets)
        token = int(np.uint32(multiplet) << np.uint32(16))
        shape = (total_beams, int(spec.ndm), int(spec.ntime))
        maps.append(np.full(shape, np.uint32(token), dtype=ARGMAX_DTYPE))
        policy.append({
            "tree_index": int(spec.tree_index),
            "multiplet": int(multiplet),
            "profile": 0,
            "fine_time": 0,
            "token_uint32": token,
        })
    result = tuple(maps)
    batch_benchmark.validate_argmax_tokens(specs, result)
    return result, tuple(policy)


def validate_tokens_with_plan(
        plan: Any, specs: Sequence[Any], token_policy: Sequence[Mapping[str, int]],
        ) -> tuple[tuple[int, ...], ...]:
    decoded = []
    for spec, item in zip(specs, token_policy):
        values = tuple(int(value) for value in plan.decode_argmax(
            int(item["token_uint32"]), int(spec.tree_index), 0, 0
        ))
        if len(values) != 5 or values[-1] != 0:
            raise ValueError(
                f"tree {spec.tree_index} constant token is not a valid profile-0 token"
            )
        decoded.append(values)
    return tuple(decoded)


def generate_clean_context_maps(
        specs: Sequence[Any], total_beams: int, base_seed: int,
        threshold: float) -> tuple[np.ndarray, ...]:
    maps = []
    for spec in specs:
        seed = derive_seed(
            base_seed, 0, int(spec.tree_index), "streaming_context"
        )
        rng = np.random.default_rng(seed)
        array = rng.standard_normal(
            (total_beams, int(spec.ndm), int(spec.ntime)), dtype=np.float32
        ).astype(SNR_DTYPE)
        if np.any(array >= threshold) or np.any(~np.isfinite(array)):
            raise AssertionError("streaming context must be finite and below threshold")
        maps.append(array)
    return tuple(maps)


def required_successor_calls(geometries: Sequence[Any]) -> int:
    """Chunks needed to move every tree's emission frontier past target end."""

    return max(
        math.ceil(int(geometry.time_radius) / int(geometry.ntime))
        for geometry in geometries
    )


def gpu_per_beam_counts(cp: Any, beam_id: Any, total_beams: int) -> np.ndarray:
    """Return dense host counts, avoiding CuPy's empty ``bincount`` failure."""

    beams = _exact_positive_integer(total_beams, "total_beams")
    if getattr(beam_id, "ndim", None) != 1:
        raise ValueError("beam_id must be a one-dimensional GPU array")
    if int(beam_id.size) == 0:
        return np.zeros(beams, dtype=np.int64)
    counts = cp.asnumpy(cp.bincount(
        beam_id.astype(cp.int64), minlength=beams
    )).astype(np.int64, copy=False)
    if counts.shape != (beams,):
        raise ValueError("candidate beam IDs lie outside the configured beam range")
    return counts


def extract_complete_target_candidates(
        cp: Any, geometries: Sequence[Any], target_maps: Sequence[Any],
        context_maps: Sequence[Any], argmax_maps: Sequence[Any],
        beam_ids: Sequence[int], threshold: float,
        concatenate_raw_candidates: Any,
        ) -> tuple[Any, np.ndarray]:
    """Use production streaming extraction and return all target-owned rows once."""

    from pirate_frb.Peakfinders import OfflinePeakExtractor

    extractors = tuple(
        OfflinePeakExtractor(
            geometry, threshold=threshold, beam_ids=beam_ids,
            assume_steady_state=True,
        )
        for geometry in geometries
    )
    for tree_index, extractor in enumerate(extractors):
        calls = batch_benchmark.required_priming_calls(
            extractor.geometry.time_radius, extractor.geometry.ntime
        )
        for source_chunk in range(TARGET_CHUNK_INDEX - calls, TARGET_CHUNK_INDEX):
            part = extractor.process_chunk(
                context_maps[tree_index], argmax_maps[tree_index], source_chunk
            )
            if len(part):
                raise AssertionError("clean halo priming unexpectedly produced candidates")
        batch_benchmark.assert_filled_streaming_halo(
            extractor, expected_last_chunk=TARGET_CHUNK_INDEX - 1
        )

    parts = []
    for tree_index, extractor in enumerate(extractors):
        parts.append(extractor.process_chunk(
            target_maps[tree_index], argmax_maps[tree_index], TARGET_CHUNK_INDEX
        ))
    successor_calls = required_successor_calls(geometries)
    for offset in range(1, successor_calls + 1):
        for tree_index, extractor in enumerate(extractors):
            parts.append(extractor.process_chunk(
                context_maps[tree_index], argmax_maps[tree_index],
                TARGET_CHUNK_INDEX + offset,
            ))
    raw = concatenate_raw_candidates(parts)
    if len(raw) and not bool(cp.all(
            raw.source_chunk_index == TARGET_CHUNK_INDEX).item()):
        raise AssertionError("non-target source ownership leaked into raw candidates")
    for extractor in extractors:
        batch_benchmark.assert_filled_streaming_halo(
            extractor,
            expected_last_chunk=TARGET_CHUNK_INDEX + successor_calls,
        )
    beam_counts = gpu_per_beam_counts(cp, raw.beam_id, len(beam_ids))
    if int(np.sum(beam_counts)) != len(raw):
        raise AssertionError("per-beam raw candidate counts do not sum to total")
    return raw, beam_counts


def synchronized_wall_time(stream: Any, operation: Any) -> tuple[float, Any]:
    """Time a complete operation between explicit current-stream barriers."""

    stream.synchronize()
    started = time.perf_counter()
    result = operation()
    stream.synchronize()
    elapsed_ms = (time.perf_counter() - started) * 1_000.0
    if not math.isfinite(elapsed_ms) or elapsed_ms < 0.0:
        raise AssertionError("synchronized wall-clock timing is not finite/non-negative")
    return elapsed_ms, result


def _raw_prefix(raw: Any, limit: int) -> Any:
    from pirate_frb.Peakfinders import GpuRawCandidates

    stop = min(len(raw), int(limit))
    return GpuRawCandidates(**{
        field.name: getattr(raw, field.name)[:stop]
        for field in fields(GpuRawCandidates)
    })


def _distribution_summary(values: Sequence[float], prefix: str) -> dict[str, float]:
    summary = summarize_counts(values)
    return {
        f"{prefix}_min": summary["min"],
        f"{prefix}_median": summary["median"],
        f"{prefix}_max": summary["max"],
    }


def morphology_summary(morphology: TrialMorphology) -> dict[str, Any]:
    components = tuple(
        component for beam in morphology.beams for component in beam.components
    )
    counts = [len(beam.components) for beam in morphology.beams]
    attempts = [beam.template_attempt for beam in morphology.beams]
    result: dict[str, Any] = {
        "total_gaussians": len(components),
        "gaussians_per_beam_min": min(counts),
        "gaussians_per_beam_median": float(np.median(counts)),
        "gaussians_per_beam_max": max(counts),
    }
    result.update(_distribution_summary(
        [component.sigma_dm for component in components],
        "gaussian_dm_width_bins",
    ))
    result.update(_distribution_summary(
        [component.sigma_time for component in components],
        "gaussian_time_width_bins",
    ))
    result.update(_distribution_summary(
        [component.correlation for component in components],
        "gaussian_correlation",
    ))
    result.update(_distribution_summary(
        [component.relative_amplitude for component in components],
        "gaussian_relative_amplitude",
    ))
    result.update(_distribution_summary(attempts, "template_attempt_per_beam"))
    return result


def build_trial_row(
        *, campaign_id: str, args: argparse.Namespace,
        morphology: TrialMorphology, percentage: float,
        calibrations: Sequence[BeamCalibration], pixel_counts: np.ndarray,
        raw_counts: np.ndarray, decoder_wall_ms: float,
        decoded_count: int, grouper_wall_ms: float | None,
        event_counts: np.ndarray | None, status: str,
        chunk_duration_ms: float,
        failure_reason: str = "") -> dict[str, Any]:
    target_per_beam = target_count_from_percentage(percentage)
    target_total = target_per_beam * args.total_beams
    achieved_total = int(np.sum(pixel_counts))
    pixel_summary = summarize_counts(pixel_counts)
    raw_summary = summarize_counts(raw_counts)
    total_raw = int(np.sum(raw_counts))
    survival = 0.0 if achieved_total == 0 else total_raw / achieved_total
    morphology_values = morphology_summary(morphology)
    calibration_json = json.dumps([
        {
            "beam_id": item.beam_id,
            "target_count": item.target_count,
            "achieved_count": item.achieved_count,
            "pixel_count_error": item.pixel_count_error,
            "scale": item.scale,
            "calibration_evaluations": item.iterations,
        }
        for item in calibrations
    ], separators=(",", ":"))
    decoder_rate = (
        0.0 if decoded_count == 0 or decoder_wall_ms == 0.0
        else 1000.0 * decoded_count / decoder_wall_ms
    )
    if grouper_wall_ms is None or event_counts is None:
        grouper_rate: Any = ""
        grouped_total: Any = ""
        event_summary = {"min": "", "median": "", "max": ""}
        combined: Any = ""
        load: Any = ""
    else:
        post = derive_post_peakfinder_metrics(
            decoder_wall_ms, grouper_wall_ms, decoded_count,
            chunk_duration_ms,
        )
        grouper_rate = post["grouper_candidates_per_second"]
        grouped_total = int(np.sum(event_counts))
        event_summary = summarize_counts(event_counts)
        combined = post["decoder_plus_grouper_wall_ms"]
        load = post["post_peakfinder_load_fraction"]
    max_error = max(
        (abs(item.pixel_count_error) for item in calibrations), default=0
    )
    row = {
        "schema_version": SCHEMA_VERSION,
        "campaign_id": campaign_id,
        "trial": morphology.trial,
        "seed": morphology.seed,
        "total_beams": args.total_beams,
        "beam_batch_size": args.beam_batch_size,
        "threshold": args.threshold,
        "method": args.method,
        "dm_reach": args.dm_reach,
        "waist_bins": args.waist_bins,
        "target_corruption_percent": percentage,
        "achieved_corruption_percent": (
            100.0 * achieved_total / (args.total_beams * PIXELS_PER_BEAM)
        ),
        "target_pixels_above_threshold": target_total,
        "achieved_pixels_above_threshold": achieved_total,
        "pixel_count_error": achieved_total - target_total,
        "max_abs_beam_pixel_count_error": max_error,
        "per_beam_calibration_json": calibration_json,
        **morphology_values,
        "total_pixels_above_threshold": achieved_total,
        "pixels_above_threshold_per_beam_min": pixel_summary["min"],
        "pixels_above_threshold_per_beam_median": pixel_summary["median"],
        "pixels_above_threshold_per_beam_max": pixel_summary["max"],
        "total_peakfinder_candidates": total_raw,
        "peakfinder_candidates_per_beam_min": raw_summary["min"],
        "peakfinder_candidates_per_beam_median": raw_summary["median"],
        "peakfinder_candidates_per_beam_max": raw_summary["max"],
        "peakfinder_survival_fraction": survival,
        "decoder_wall_ms": decoder_wall_ms,
        "decoder_candidates_per_second": decoder_rate,
        "total_decoded_candidates": decoded_count,
        "grouper_wall_ms": grouper_wall_ms if grouper_wall_ms is not None else "",
        "grouper_candidates_per_second": grouper_rate,
        "total_grouped_events": grouped_total,
        "grouped_events_per_beam_min": event_summary["min"],
        "grouped_events_per_beam_median": event_summary["median"],
        "grouped_events_per_beam_max": event_summary["max"],
        "decoder_plus_grouper_wall_ms": combined,
        "post_peakfinder_load_fraction": load,
        "candidate_safety_limit": args.candidate_safety_limit,
        "status": status,
        "failure_reason": failure_reason,
    }
    return _validate_rows([row], TRIAL_FIELDS)[0]


def _blank_failed_row(
        *, campaign_id: str, args: argparse.Namespace, trial: int,
        percentage: float, status: str, reason: str) -> dict[str, Any]:
    row = {field: "" for field in TRIAL_FIELDS}
    row.update({
        "schema_version": SCHEMA_VERSION,
        "campaign_id": campaign_id,
        "trial": trial,
        "seed": derive_seed(args.base_seed, trial, 0, "trial"),
        "total_beams": args.total_beams,
        "beam_batch_size": args.beam_batch_size,
        "threshold": args.threshold,
        "method": args.method,
        "dm_reach": args.dm_reach,
        "waist_bins": args.waist_bins,
        "target_corruption_percent": percentage,
        "target_pixels_above_threshold": (
            target_count_from_percentage(percentage) * args.total_beams
        ),
        "candidate_safety_limit": args.candidate_safety_limit,
        "status": status,
        "failure_reason": reason,
    })
    return _validate_rows([row], TRIAL_FIELDS)[0]


def _validate_rows(
        rows: Iterable[Mapping[str, Any]], fields_: Sequence[str]) -> list[dict[str, Any]]:
    expected = set(fields_)
    normalized = []
    for index, row in enumerate(rows):
        if set(row) != expected:
            raise ValueError(
                f"row {index} schema mismatch: missing={sorted(expected-set(row))}, "
                f"extra={sorted(set(row)-expected)}"
            )
        normalized.append(dict(row))
    return normalized


def _atomic_write_csv(
        path: Path, fields_: Sequence[str], rows: Iterable[Mapping[str, Any]]) -> None:
    rows = _validate_rows(rows, fields_)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = None
    try:
        with tempfile.NamedTemporaryFile(
                mode="w", encoding="utf-8", newline="", prefix=f".{path.name}.",
                suffix=".tmp", dir=path.parent, delete=False) as stream:
            temporary = stream.name
            writer = csv.DictWriter(stream, fieldnames=fields_, lineterminator="\n")
            writer.writeheader()
            writer.writerows(rows)
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


def _atomic_write_yaml(path: Path, document: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = None
    try:
        with tempfile.NamedTemporaryFile(
                mode="w", encoding="utf-8", prefix=f".{path.name}.",
                suffix=".tmp", dir=path.parent, delete=False) as stream:
            temporary = stream.name
            yaml.safe_dump(dict(document), stream, sort_keys=False)
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


def _read_csv_exact(path: Path, fields_: Sequence[str]) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8", newline="") as stream:
        reader = csv.DictReader(stream)
        if tuple(reader.fieldnames or ()) != tuple(fields_):
            raise ValueError(
                f"{path} has columns {reader.fieldnames!r}; expected {tuple(fields_)!r}"
            )
        rows = list(reader)
    _validate_rows(rows, fields_)
    return rows


def _quantile(values: np.ndarray, fraction: float) -> float:
    try:
        return float(np.quantile(values, fraction, method="linear"))
    except TypeError:  # NumPy < 1.22
        return float(np.quantile(values, fraction, interpolation="linear"))


def summarize_trials(
        rows: Sequence[Mapping[str, Any]], percentages: Sequence[float],
        campaign_id: str) -> list[dict[str, Any]]:
    """Summarize completed trials only; skipped/failed rows remain in trials.csv."""

    result = []
    for percentage in percentages:
        matching = [
            row for row in rows
            if canonical_percentage(row["target_corruption_percent"])
            == canonical_percentage(percentage)
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
            "failed_trials": sum(
                str(row["status"]).startswith("failed_") for row in matching
            ),
        }
        for metric in SUMMARY_METRICS:
            values = np.asarray([
                float(row[metric]) for row in completed if row[metric] != ""
            ], dtype=np.float64)
            if values.size:
                q25 = _quantile(values, 0.25)
                q75 = _quantile(values, 0.75)
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
            for name, value in statistics.items():
                summary[f"{metric}_{name}"] = value
        result.append(summary)
    return _validate_rows(result, SUMMARY_FIELDS)


def result_paths(results_dir: Path | str) -> dict[str, Path]:
    directory = Path(results_dir).resolve()
    return {
        "directory": directory,
        "trials": directory / "trials.csv",
        "summary": directory / "summary.csv",
        "metadata": directory / "metadata.yaml",
    }


def expected_trial_keys(
        trials: int, percentages: Sequence[float]) -> tuple[tuple[int, str], ...]:
    return tuple(
        (trial, canonical_percentage(percentage))
        for trial in range(trials)
        for percentage in percentages
    )


def validate_trial_prefix(
        rows: Sequence[Mapping[str, Any]], trials: int,
        percentages: Sequence[float], campaign_id: str) -> None:
    expected = expected_trial_keys(trials, percentages)
    actual = tuple(
        (int(row["trial"]), canonical_percentage(row["target_corruption_percent"]))
        for row in rows
    )
    if actual != expected[:len(actual)]:
        raise ValueError("trials.csv is not an exact ordered campaign prefix")
    if len(actual) > len(expected):
        raise ValueError("trials.csv contains more configurations than requested")
    if any(str(row["campaign_id"]) != campaign_id for row in rows):
        raise ValueError("trials.csv campaign_id is incompatible")


def validate_resumed_trial_rows(
        rows: Sequence[Mapping[str, Any]], args: argparse.Namespace,
        campaign_id: str, chunk_duration_ms: float) -> None:
    """Validate scientific invariants before appending to a resumed campaign."""

    duration = _finite_float(chunk_duration_ms, "chunk_duration_ms")
    if duration <= 0.0:
        raise ValueError("chunk_duration_ms must be strictly positive")
    allowed_statuses = {
        "completed",
        "skipped_candidate_safety_limit",
        "failed_template_unattainable",
        "failed_calibration",
    }
    for index, row in enumerate(rows):
        context = f"resumed trials.csv row {index + 2}"
        integer_expectations = {
            "schema_version": SCHEMA_VERSION,
            "total_beams": args.total_beams,
            "beam_batch_size": args.beam_batch_size,
            "dm_reach": args.dm_reach,
            "waist_bins": args.waist_bins,
            "candidate_safety_limit": args.candidate_safety_limit,
        }
        for field_name, expected in integer_expectations.items():
            if int(row[field_name]) != int(expected):
                raise ValueError(f"{context}: incompatible {field_name}")
        if str(row["campaign_id"]) != campaign_id:
            raise ValueError(f"{context}: incompatible campaign_id")
        if str(row["method"]) != args.method:
            raise ValueError(f"{context}: incompatible peakfinder method")
        if not math.isclose(
                float(row["threshold"]), args.threshold, rel_tol=0.0,
                abs_tol=1.0e-12):
            raise ValueError(f"{context}: incompatible threshold")
        trial = int(row["trial"])
        expected_seed = derive_seed(args.base_seed, trial, 0, "trial")
        if int(row["seed"]) != expected_seed:
            raise ValueError(f"{context}: incompatible trial seed")
        percentage = float(canonical_percentage(
            row["target_corruption_percent"]
        ))
        expected_target = (
            target_count_from_percentage(percentage) * args.total_beams
        )
        if int(row["target_pixels_above_threshold"]) != expected_target:
            raise ValueError(f"{context}: incompatible target pixel count")
        status = str(row["status"])
        if status not in allowed_statuses:
            raise ValueError(f"{context}: unsupported status {status!r}")
        if status.startswith("failed_"):
            if not str(row["failure_reason"]).strip():
                raise ValueError(f"{context}: failed row lacks a reason")
            continue

        total_pixels = int(row["total_pixels_above_threshold"])
        achieved_pixels = int(row["achieved_pixels_above_threshold"])
        raw_count = int(row["total_peakfinder_candidates"])
        decoded_count = int(row["total_decoded_candidates"])
        if min(total_pixels, achieved_pixels, raw_count, decoded_count) < 0:
            raise ValueError(f"{context}: negative count")
        if achieved_pixels != total_pixels or decoded_count != raw_count:
            raise ValueError(f"{context}: count conservation failed")
        if int(row["pixel_count_error"]) != total_pixels - expected_target:
            raise ValueError(f"{context}: pixel_count_error is inconsistent")
        try:
            calibration_rows = json.loads(str(row["per_beam_calibration_json"]))
        except (TypeError, ValueError, json.JSONDecodeError) as exc:
            raise ValueError(
                f"{context}: per-beam calibration JSON is invalid"
            ) from exc
        if (not isinstance(calibration_rows, list)
                or len(calibration_rows) != args.total_beams):
            raise ValueError(
                f"{context}: per-beam calibration row count is inconsistent"
            )
        target_per_beam = target_count_from_percentage(percentage)
        calibration_by_beam: dict[int, Mapping[str, Any]] = {}
        achieved_by_beam = []
        errors_by_beam = []
        for calibration in calibration_rows:
            if not isinstance(calibration, dict):
                raise ValueError(f"{context}: calibration entry is not a mapping")
            beam_id = int(calibration["beam_id"])
            if beam_id in calibration_by_beam:
                raise ValueError(f"{context}: duplicate calibration beam_id")
            calibration_by_beam[beam_id] = calibration
            target_count = int(calibration["target_count"])
            achieved_count = int(calibration["achieved_count"])
            pixel_error = int(calibration["pixel_count_error"])
            scale = float(calibration["scale"])
            evaluations = int(calibration["calibration_evaluations"])
            if (target_count != target_per_beam
                    or achieved_count - target_count != pixel_error
                    or abs(pixel_error) > args.calibration_pixel_tolerance
                    or achieved_count < 0 or evaluations < 0
                    or not math.isfinite(scale) or scale < 0.0):
                raise ValueError(
                    f"{context}: invalid calibration values for beam {beam_id}"
                )
            achieved_by_beam.append(achieved_count)
            errors_by_beam.append(pixel_error)
        if set(calibration_by_beam) != set(range(args.total_beams)):
            raise ValueError(f"{context}: calibration beam IDs are incomplete")
        if (sum(achieved_by_beam) != achieved_pixels
                or sum(errors_by_beam) != int(row["pixel_count_error"])):
            raise ValueError(f"{context}: per-beam calibration sums are inconsistent")
        maximum_error = max(map(abs, errors_by_beam), default=0)
        if (int(row["max_abs_beam_pixel_count_error"]) != maximum_error
                or maximum_error > args.calibration_pixel_tolerance):
            raise ValueError(f"{context}: per-beam calibration tolerance is inconsistent")
        expected_percent = (
            100.0 * total_pixels / (args.total_beams * PIXELS_PER_BEAM)
        )
        if not math.isclose(
                float(row["achieved_corruption_percent"]), expected_percent,
                rel_tol=1.0e-12, abs_tol=1.0e-12):
            raise ValueError(f"{context}: achieved percentage is inconsistent")
        expected_survival = 0.0 if total_pixels == 0 else raw_count / total_pixels
        if not math.isclose(
                float(row["peakfinder_survival_fraction"]), expected_survival,
                rel_tol=1.0e-12, abs_tol=1.0e-12):
            raise ValueError(f"{context}: survival fraction is inconsistent")
        decoder_ms = float(row["decoder_wall_ms"])
        if not math.isfinite(decoder_ms) or decoder_ms < 0.0:
            raise ValueError(f"{context}: invalid decoder timing")

        if status == "skipped_candidate_safety_limit":
            if decoded_count <= args.candidate_safety_limit:
                raise ValueError(f"{context}: safety skip is below the limit")
            for field_name in (
                    "grouper_wall_ms", "total_grouped_events",
                    "decoder_plus_grouper_wall_ms",
                    "post_peakfinder_load_fraction"):
                if str(row[field_name]).strip():
                    raise ValueError(
                        f"{context}: safety-skipped {field_name} must be empty"
                    )
            continue

        grouper_ms = float(row["grouper_wall_ms"])
        combined = float(row["decoder_plus_grouper_wall_ms"])
        load = float(row["post_peakfinder_load_fraction"])
        events = int(row["total_grouped_events"])
        if (not all(math.isfinite(value) and value >= 0.0 for value in
                    (grouper_ms, combined, load)) or events < 0):
            raise ValueError(f"{context}: invalid completed grouping result")
        if not math.isclose(
                combined, decoder_ms + grouper_ms,
                rel_tol=1.0e-12, abs_tol=1.0e-12):
            raise ValueError(f"{context}: combined timing is inconsistent")
        if not math.isclose(
                load, combined / duration,
                rel_tol=1.0e-12, abs_tol=1.0e-12):
            raise ValueError(f"{context}: real-time load is inconsistent")


def campaign_signature(payload: Mapping[str, Any]) -> str:
    encoded = json.dumps(
        payload, sort_keys=True, separators=(",", ":"), ensure_ascii=True
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def _sha256_path(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _source_information() -> dict[str, Any]:
    paths = {
        name: REPOSITORY_ROOT / relative
        for name, relative in {
            "Peakfinders": "pirate_frb/Peakfinders.py",
            "GpuArgmaxDecoder": "pirate_frb/GpuArgmaxDecoder.py",
            "OfflineCandidateGrouper": "pirate_frb/OfflineCandidateGrouper.py",
            "FrbOfflineGrouper": "pirate_frb/FrbOfflineGrouper.py",
            "run_offline_grouper": "pirate_frb/run_offline_grouper.py",
        }.items()
    }
    return {
        name: {"path": str(path.resolve()), "sha256": _sha256_path(path)}
        for name, path in paths.items()
    }


def _benchmark_source_information() -> dict[str, Any]:
    paths = {
        "gaussian_corruption_benchmark": Path(__file__).resolve(),
        "peakfinder_batch_helpers": Path(batch_benchmark.__file__).resolve(),
    }
    return {
        name: {"path": str(path), "sha256": _sha256_path(path)}
        for name, path in paths.items()
    }


def _signature_payload(
        args: argparse.Namespace, bundle: Any, gaussian_config: GaussianConfig,
        gpu: Mapping[str, Any], software: Mapping[str, Any]) -> dict[str, Any]:
    return {
        "schema_name": SCHEMA_NAME,
        "schema_version": SCHEMA_VERSION,
        "config_path": str(args.config),
        "config_sha256": _sha256_path(args.config),
        "producer_plan_sha256": hashlib.sha256(
            bundle.producer_plan_yaml.encode("utf-8")
        ).hexdigest(),
        "tree_shapes": [list(spec.shape) for spec in bundle.specs],
        "percentages": [canonical_percentage(p) for p in args.percentages],
        "total_beams": args.total_beams,
        "beam_batch_size": args.beam_batch_size,
        "trials": args.trials,
        "threshold": args.threshold,
        "method": args.method,
        "dm_reach": args.dm_reach,
        "chunk_duration_ms": bundle.chunk_duration_ms,
        "waist_bins": args.waist_bins,
        "device": args.device,
        "base_seed": args.base_seed,
        "gaussian_config": gaussian_config.__dict__,
        "template_attempt_limit": args.template_attempt_limit,
        "calibration_pixel_tolerance": args.calibration_pixel_tolerance,
        "calibration_max_iterations": args.calibration_max_iterations,
        "dm_tolerance_bins": args.dm_tolerance_bins,
        "time_padding_bins": args.time_padding_bins,
        "candidate_safety_limit": args.candidate_safety_limit,
        "allow_unsafe_grouping": args.allow_unsafe_grouping,
        "gpu_model": gpu.get("model"),
        "cuda_runtime_version": gpu.get("cuda_runtime_version"),
        "python": software.get("python"),
        "numpy": software.get("numpy"),
        "cupy": software.get("cupy"),
        "production_sources": _source_information(),
        "benchmark_sources": _benchmark_source_information(),
    }


def build_metadata(
        *, args: argparse.Namespace, bundle: Any,
        gaussian_config: GaussianConfig, token_policy: Sequence[Mapping[str, int]],
        token_decodes: Sequence[Sequence[int]], geometries: Sequence[Any],
        gpu: Mapping[str, Any], software: Mapping[str, Any], git: Mapping[str, Any],
        signature: str, campaign_id: str, created_utc: str,
        rows: Sequence[Mapping[str, Any]],
        signature_payload: Mapping[str, Any] | None = None,
        ) -> dict[str, Any]:
    expected = expected_trial_keys(args.trials, args.percentages)
    actual = {
        (int(row["trial"]), canonical_percentage(row["target_corruption_percent"]))
        for row in rows
    }
    statuses: dict[str, int] = {}
    for row in rows:
        statuses[str(row["status"])] = statuses.get(str(row["status"]), 0) + 1
    successor_calls = required_successor_calls(geometries)
    payload = dict(
        _signature_payload(args, bundle, gaussian_config, gpu, software)
        if signature_payload is None else signature_payload
    )
    if campaign_signature(payload) != signature:
        raise ValueError("campaign signature does not match its metadata payload")
    return {
        "schema_name": SCHEMA_NAME,
        "schema_version": SCHEMA_VERSION,
        "campaign_id": campaign_id,
        "campaign_signature": signature,
        "campaign_signature_payload": payload,
        "created_utc": created_utc,
        "last_updated_utc": _utc_now(),
        "campaign": {
            "trials": args.trials,
            "percentage_grid": list(args.percentages),
            "total_beams": args.total_beams,
            "beam_batch_size": args.beam_batch_size,
            "device": args.device,
            "base_seed": args.base_seed,
            "chunk_duration_ms": bundle.chunk_duration_ms,
            "total_pixels_per_trial": args.total_beams * PIXELS_PER_BEAM,
        },
        "plan": {
            "config_path": str(args.config),
            "config_sha256": _sha256_path(args.config),
            "config_document": dict(bundle.config_document),
            "producer_plan_yaml": bundle.producer_plan_yaml,
            "producer_plan_sha256": hashlib.sha256(
                bundle.producer_plan_yaml.encode("utf-8")
            ).hexdigest(),
            "consumer_reconstruction": (
                "DedispersionPlan(config,gpu_runnable=True), serialize, then "
                "DedispersionPlan.make_incomplete_plan_from_yaml"
            ),
            "tree_shapes_ndm_ntime": [list(spec.shape) for spec in bundle.specs],
            "tree_plan": [
                {
                    "tree_index": spec.tree_index,
                    "primary_tree_index": spec.primary_tree_index,
                    "early_trigger_level": spec.early_trigger_level,
                    "ndm": spec.ndm,
                    "ntime": spec.ntime,
                    "pixels_per_beam": spec.pixels_per_beam,
                    "multiplets": spec.multiplets,
                    "profiles": spec.profiles,
                    "token_dout": spec.token_dout,
                }
                for spec in bundle.specs
            ],
            "pixels_per_beam": PIXELS_PER_BEAM,
            "expected_total_beams": DEFAULT_TOTAL_BEAMS,
            "expected_total_pixels_for_60_beams": DEFAULT_TOTAL_BEAMS * PIXELS_PER_BEAM,
            "chunk_duration_ms": bundle.chunk_duration_ms,
        },
        "corruption_model": {
            "name": CORRUPTION_MODEL,
            "percentage_convention": (
                "100 * count(final_float16_snr >= threshold) / 983040, "
                "calibrated independently for every beam"
            ),
            "rounding": "Python round of percentage/100 * 983040 per beam",
            "zero_corruption": (
                "unscaled standard Gaussian white noise; asserted to have zero "
                "final-float16 pixels at or above threshold"
            ),
            "noise": "independent standard Gaussian per trial, beam, tree in float32",
            "template": (
                "non-negative sum of clipped correlated 2-D Gaussians; no wraparound, "
                "no random hot pixels, and no cross-tree parent events"
            ),
            "component_distributions": {
                "default_rationale": (
                    "conservative native-bin scales for plan trees spanning DM 512..4096 "
                    "and time 16..128: broad enough for 1% support after clipping while "
                    "keeping the number of morphology peaks modest"
                ),
                "custom_high_percentage_note": (
                    "custom levels such as 5% or 20% are accepted, but the conservative "
                    "default mixture may require larger count/width ranges; any level "
                    "that remains unattainable is recorded explicitly rather than "
                    "overflowing float16 or substituting pixels"
                ),
                "count_per_beam": {
                    "distribution": "discrete_uniform_inclusive",
                    "minimum": gaussian_config.count_min,
                    "maximum": gaussian_config.count_max,
                },
                "tree": "categorical proportional to native tree pixel count",
                "centre": "independent uniform continuous coordinates within selected tree",
                "sigma_dm_bins": {
                    "distribution": "log_uniform",
                    "minimum": gaussian_config.dm_width_min,
                    "maximum": gaussian_config.dm_width_max,
                },
                "sigma_time_bins": {
                    "distribution": "log_uniform",
                    "minimum": gaussian_config.time_width_min,
                    "maximum": gaussian_config.time_width_max,
                },
                "correlation": {
                    "distribution": "uniform",
                    "minimum": -gaussian_config.max_abs_correlation,
                    "maximum": gaussian_config.max_abs_correlation,
                },
                "relative_amplitude": {
                    "distribution": "log_uniform",
                    "minimum": gaussian_config.relative_amplitude_min,
                    "maximum": gaussian_config.relative_amplitude_max,
                },
                "render_sigma_cutoff": gaussian_config.render_sigma_cutoff,
            },
            "reuse_policy": (
                "within a trial, every percentage reuses identical noise and templates; "
                "only one common amplitude scale per beam changes"
            ),
            "generation_dtype": "float32",
            "production_snr_dtype": "float16",
            "calibration": {
                "algorithm": (
                    "float16 threshold-boundary order statistic followed by bounded "
                    "float32-scale bracketing/bisection and final float16 count validation"
                ),
                "pixel_tolerance_per_beam": args.calibration_pixel_tolerance,
                "maximum_iterations": args.calibration_max_iterations,
                "deterministic_template_attempt_limit": args.template_attempt_limit,
                "attempt_acceptance": (
                    "try deterministic templates until every configured rounded target "
                    "validates against the final float16 maps; if the attempt limit is "
                    "exhausted, retain the deterministically best template for paired "
                    "attainable levels and fail only its unattainable levels"
                ),
                "failure_policy": (
                    "record failed_template_unattainable/failed_calibration; never "
                    "substitute hot pixels or change the target"
                ),
            },
            "seed_policy": (
                "uint64 little-endian prefix of SHA256 over schema, base seed, trial, "
                "beam, purpose, and deterministic attempt; noise and templates use "
                "separate streams"
            ),
        },
        "argmax_tokens": {
            "policy": (
                "one deterministic valid tree-specific multiplet, profile 0, fine time 0; "
                "constant over every pixel and reused across trials/percentages"
            ),
            "trees": [
                {**dict(item), "plan_decode_at_idm0_itime0": list(decoded)}
                for item, decoded in zip(token_policy, token_decodes)
            ],
        },
        "peakfinder": {
            "method": args.method,
            "threshold": args.threshold,
            "dm_reach": args.dm_reach,
            "waist_bins": args.waist_bins,
            "timed": False,
            "input_shape_policy": "all beams together: (total_beams, ndm, ntime)",
            "raw_combination": "production concatenate_raw_candidates global stable sort",
        },
        "streaming_state": {
            "startup": "assume_steady_state=True with production all-valid masks",
            "priming": (
                "per tree, required_priming_calls=(2*time_radius)//ntime+1 clean "
                "ordinary process_chunk calls; exact filled 2*h tails validated"
            ),
            "target_chunk_index": TARGET_CHUNK_INDEX,
            "right_seam": (
                "ordinary clean successor process_chunk calls; no flush because target "
                "right seam is not a physical acquisition edge"
            ),
            "successor_calls": successor_calls,
            "successor_formula": "max_tree ceil(time_radius/ntime)",
            "ownership_validation": (
                "all concatenated candidates must retain target source_chunk_index exactly"
            ),
        },
        "decoder_timing": {
            "metric": "decoder_wall_ms",
            "boundary": (
                "synchronize current stream; perf_counter; decoder.decode(raw) once for "
                "the joint beam batch; synchronize; elapsed wall time"
            ),
            "excluded": (
                "construction, warm-up, map generation/calibration/upload, peakfinding, "
                "correctness counts, and output"
            ),
        },
        "grouper_timing": {
            "metric": "grouper_wall_ms",
            "production_strategy": (
                "one joint group_candidates call for the compatible beam batch; the "
                "compatibility predicate enforces equal beam_id internally"
            ),
            "boundary": (
                "synchronize current stream; perf_counter; group_candidates(decoded, "
                "grouping_geometry, config=grouping_config); synchronize; elapsed wall time"
            ),
            "dm_tolerance_bins": args.dm_tolerance_bins,
            "time_padding_bins": args.time_padding_bins,
            "complexity_warning": (
                "the active representative-seeded Python loop sees the global joint "
                "candidate table and can require O(N^2) predicate work and O(N) "
                "CPU-GPU scalar synchronizations in the worst case"
            ),
            "combined_metric": (
                "decoder_plus_grouper_wall_ms is the sum of two separately synchronized "
                "measurements, not a fused pipeline interval"
            ),
        },
        "safety_limit": {
            "candidate_limit": args.candidate_safety_limit,
            "unsafe_override": bool(args.allow_unsafe_grouping),
            "policy": (
                "decode is always retained; above the limit grouping is skipped with "
                "status=skipped_candidate_safety_limit unless --allow-unsafe-grouping"
            ),
        },
        "environment": {
            "gpu": dict(gpu),
            "software": dict(software),
            "git": dict(git),
            "production_sources": _source_information(),
            "benchmark_sources": _benchmark_source_information(),
        },
        "csv_schemas": {
            "trials.csv": list(TRIAL_FIELDS),
            "summary.csv": list(SUMMARY_FIELDS),
            "summary_metrics": list(SUMMARY_METRICS),
            "summary_statistics": list(SUMMARY_STATISTICS),
            "zero_denominator_peakfinder_survival_fraction": 0.0,
        },
        "campaign_completeness": {
            "expected_configurations": len(expected),
            "recorded_configurations": len(rows),
            "status_counts": statuses,
            "missing_configurations": [
                {"trial": trial, "target_corruption_percent": percentage}
                for trial, percentage in expected if (trial, percentage) not in actual
            ],
            "complete": len(actual) == len(expected),
        },
        "outputs": {
            "trials": "trials.csv",
            "summary": "summary.csv",
            "metadata": "metadata.yaml",
            "analysis_notebook": str(NOTEBOOK_PATH.resolve()),
        },
    }


def write_checkpoint(
        paths: Mapping[str, Path], rows: Sequence[Mapping[str, Any]],
        percentages: Sequence[float], campaign_id: str,
        metadata: Mapping[str, Any]) -> None:
    _atomic_write_csv(paths["trials"], TRIAL_FIELDS, rows)
    _atomic_write_csv(
        paths["summary"], SUMMARY_FIELDS,
        summarize_trials(rows, percentages, campaign_id),
    )
    _atomic_write_yaml(paths["metadata"], metadata)


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
        return [], _utc_now()
    if existing and not resume:
        raise FileExistsError(
            "result files already exist; use --resume or --overwrite: "
            + ", ".join(str(path) for path in existing)
        )
    if not existing:
        return [], _utc_now()
    if len(existing) != len(files):
        raise ValueError("resume requires all three result files")
    with paths["metadata"].open("r", encoding="utf-8") as stream:
        metadata = yaml.safe_load(stream)
    if not isinstance(metadata, dict):
        raise ValueError("metadata.yaml must contain a mapping")
    stored_payload = metadata.get("campaign_signature_payload")
    try:
        stored_signature = (
            campaign_signature(stored_payload)
            if isinstance(stored_payload, dict) else None
        )
    except (TypeError, ValueError):
        stored_signature = None
    if (metadata.get("schema_name") != SCHEMA_NAME
            or int(metadata.get("schema_version", -1)) != SCHEMA_VERSION
            or metadata.get("campaign_signature") != signature
            or stored_signature != signature
            or metadata.get("campaign_id") != campaign_id):
        raise ValueError("resume metadata is incompatible with this campaign")
    rows: list[dict[str, Any]] = [
        dict(row) for row in _read_csv_exact(paths["trials"], TRIAL_FIELDS)
    ]
    _read_csv_exact(paths["summary"], SUMMARY_FIELDS)
    validate_trial_prefix(rows, trials, percentages, campaign_id)
    return rows, str(metadata.get("created_utc") or _utc_now())


def _make_gaussian_config(args: argparse.Namespace) -> GaussianConfig:
    return _validate_gaussian_config(GaussianConfig(
        count_min=args.gaussians_min,
        count_max=args.gaussians_max,
        dm_width_min=args.dm_width_min,
        dm_width_max=args.dm_width_max,
        time_width_min=args.time_width_min,
        time_width_max=args.time_width_max,
        max_abs_correlation=args.max_abs_correlation,
        relative_amplitude_min=args.relative_amplitude_min,
        relative_amplitude_max=args.relative_amplitude_max,
        render_sigma_cutoff=args.render_sigma_cutoff,
    ))


def _make_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Monte Carlo Gaussian-corruption benchmark of the production decoder "
            "and joint multi-beam grouper"
        ),
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--config", default=str(DEFAULT_CONFIG))
    parser.add_argument("--results-dir", default=str(DEFAULT_RESULTS_DIR))
    parser.add_argument("--total-beams", type=int, default=DEFAULT_TOTAL_BEAMS)
    parser.add_argument("--beam-batch-size", type=int, default=DEFAULT_BEAM_BATCH_SIZE)
    parser.add_argument("--threshold", type=float, default=DEFAULT_THRESHOLD)
    parser.add_argument("--dm-reach", type=int, default=DEFAULT_DM_REACH)
    parser.add_argument("--waist-bins", type=int, default=DEFAULT_WAIST_BINS)
    parser.add_argument("--trials", type=int, default=DEFAULT_TRIALS)
    parser.add_argument("--device", type=int, default=DEFAULT_DEVICE)
    parser.add_argument("--base-seed", type=int, default=DEFAULT_BASE_SEED)
    parser.add_argument(
        "--corruption-percentages", "--percentages", dest="percentages",
        type=float, nargs="+", default=list(DEFAULT_PERCENTAGES),
        help="percentage units: 0.01 means 0.01 percent",
    )
    parser.add_argument("--gaussians-min", type=int, default=DEFAULT_GAUSSIANS_MIN)
    parser.add_argument("--gaussians-max", type=int, default=DEFAULT_GAUSSIANS_MAX)
    parser.add_argument("--dm-width-min", type=float, default=DEFAULT_DM_WIDTH_MIN)
    parser.add_argument("--dm-width-max", type=float, default=DEFAULT_DM_WIDTH_MAX)
    parser.add_argument("--time-width-min", type=float, default=DEFAULT_TIME_WIDTH_MIN)
    parser.add_argument("--time-width-max", type=float, default=DEFAULT_TIME_WIDTH_MAX)
    parser.add_argument(
        "--max-abs-correlation", type=float,
        default=DEFAULT_MAX_ABS_CORRELATION,
    )
    parser.add_argument(
        "--relative-amplitude-min", type=float,
        default=DEFAULT_RELATIVE_AMPLITUDE_MIN,
    )
    parser.add_argument(
        "--relative-amplitude-max", type=float,
        default=DEFAULT_RELATIVE_AMPLITUDE_MAX,
    )
    parser.add_argument(
        "--render-sigma-cutoff", type=float,
        default=DEFAULT_RENDER_SIGMA_CUTOFF,
    )
    parser.add_argument(
        "--template-attempt-limit", type=int,
        default=DEFAULT_TEMPLATE_ATTEMPT_LIMIT,
    )
    parser.add_argument(
        "--calibration-pixel-tolerance", type=int,
        default=DEFAULT_CALIBRATION_PIXEL_TOLERANCE,
    )
    parser.add_argument(
        "--calibration-max-iterations", type=int,
        default=DEFAULT_CALIBRATION_MAX_ITERATIONS,
    )
    parser.add_argument(
        "--dm-tolerance-bins", type=float,
        default=DEFAULT_DM_TOLERANCE_BINS,
    )
    parser.add_argument(
        "--time-padding-bins", type=float,
        default=DEFAULT_TIME_PADDING_BINS,
    )
    parser.add_argument(
        "--candidate-safety-limit", type=int,
        default=DEFAULT_CANDIDATE_SAFETY_LIMIT,
    )
    parser.add_argument(
        "--allow-unsafe-grouping", action="store_true",
        help="explicitly run production grouping above the candidate safety limit",
    )
    control = parser.add_mutually_exclusive_group()
    control.add_argument("--resume", action="store_true")
    control.add_argument("--overwrite", action="store_true")
    return parser


def validate_arguments(args: argparse.Namespace) -> argparse.Namespace:
    # Production exposes no peakfinder selector.  The constant remains in
    # result rows solely to make their provenance self-describing.
    args.method = METHOD
    args.total_beams = _exact_positive_integer(args.total_beams, "total_beams")
    args.beam_batch_size = _exact_positive_integer(
        args.beam_batch_size, "beam_batch_size"
    )
    # The requested study is explicitly one joint production beam batch.  This
    # also keeps decoder/grouper semantics unambiguous for custom smoke sizes.
    if args.beam_batch_size != args.total_beams:
        raise ValueError(
            "this focused benchmark requires beam_batch_size == total_beams so "
            "the production decoder and grouper each receive one joint batch"
        )
    args.trials = _exact_positive_integer(args.trials, "trials")
    args.device = _exact_nonnegative_integer(args.device, "device")
    args.base_seed = _exact_nonnegative_integer(args.base_seed, "base_seed")
    args.dm_reach = _exact_nonnegative_integer(args.dm_reach, "dm_reach")
    args.waist_bins = _exact_nonnegative_integer(args.waist_bins, "waist_bins")
    args.gaussians_min = _exact_positive_integer(args.gaussians_min, "gaussians_min")
    args.gaussians_max = _exact_positive_integer(args.gaussians_max, "gaussians_max")
    args.template_attempt_limit = _exact_positive_integer(
        args.template_attempt_limit, "template_attempt_limit"
    )
    args.calibration_pixel_tolerance = _exact_nonnegative_integer(
        args.calibration_pixel_tolerance, "calibration_pixel_tolerance"
    )
    args.calibration_max_iterations = _exact_positive_integer(
        args.calibration_max_iterations, "calibration_max_iterations"
    )
    args.candidate_safety_limit = _exact_positive_integer(
        args.candidate_safety_limit, "candidate_safety_limit"
    )
    args.threshold = _finite_float(args.threshold, "threshold")
    args.dm_tolerance_bins = _finite_float(
        args.dm_tolerance_bins, "dm_tolerance_bins"
    )
    args.time_padding_bins = _finite_float(
        args.time_padding_bins, "time_padding_bins"
    )
    if args.dm_tolerance_bins < 0.0 or args.time_padding_bins < 0.0:
        raise ValueError("grouping tolerances must be non-negative")
    args.percentages = validate_percentage_grid(args.percentages)
    args.config = Path(args.config).resolve()
    args.results_dir = Path(args.results_dir).resolve()
    _make_gaussian_config(args)
    return args


def _print_percentage_summary(rows: Sequence[Mapping[str, Any]], percentage: float) -> None:
    matching = [
        row for row in rows
        if canonical_percentage(row["target_corruption_percent"])
        == canonical_percentage(percentage)
    ]
    completed = [row for row in matching if row["status"] == "completed"]
    if not completed:
        print(
            f"corruption={percentage:g}% recorded={len(matching)} completed=0",
            flush=True,
        )
        return
    decoder = np.asarray([float(row["decoder_wall_ms"]) for row in completed])
    grouper = np.asarray([float(row["grouper_wall_ms"]) for row in completed])
    candidates = np.asarray([
        float(row["total_peakfinder_candidates"]) for row in completed
    ])
    print(
        f"corruption={percentage:g}% completed={len(completed)} "
        f"candidates_median={np.median(candidates):.0f} "
        f"decoder_median={np.median(decoder):.3f}ms "
        f"grouper_median={np.median(grouper):.3f}ms",
        flush=True,
    )


def _print_aggregate(rows: Sequence[Mapping[str, Any]], percentages: Sequence[float]) -> None:
    print("aggregate summary", flush=True)
    for percentage in percentages:
        _print_percentage_summary(rows, percentage)


def run_benchmark(args: argparse.Namespace) -> None:
    args = validate_arguments(args)
    gaussian_config = _make_gaussian_config(args)
    paths = result_paths(args.results_dir)

    import cupy as cp
    from pirate_frb.GpuArgmaxDecoder import GpuArgmaxDecoder
    from pirate_frb.OfflineCandidateGrouper import (
        GroupingConfig,
        GroupingGeometry,
        group_candidates,
    )
    from pirate_frb.Peakfinders import concatenate_raw_candidates

    with cp.cuda.Device(args.device):
        stream = cp.cuda.get_current_stream()
        active_sources = batch_benchmark.active_repository_sources()
        bundle = batch_benchmark.load_authoritative_plan(args.config)
        if int(bundle.config.beams_per_gpu) != DEFAULT_TOTAL_BEAMS:
            raise ValueError("authoritative config no longer declares 60 beams per GPU")
        if tuple(spec.shape for spec in bundle.specs) != EXPECTED_SHAPES:
            raise AssertionError("authoritative plan shapes changed")
        if args.total_beams == DEFAULT_TOTAL_BEAMS:
            total_pixels = args.total_beams * PIXELS_PER_BEAM
            if total_pixels != 58_982_400:
                raise AssertionError("60-beam pixel total changed")

        geometries_by_reach, _ = batch_benchmark.build_geometries(
            cp, bundle.plan, bundle.specs, (args.dm_reach,), args.waist_bins
        )
        geometries = geometries_by_reach[args.dm_reach]
        decoder = GpuArgmaxDecoder(bundle.plan, cuda_device_id=args.device)
        grouping_geometry = GroupingGeometry.from_plan(bundle.plan)
        grouping_config = GroupingConfig(
            dm_tolerance_bins=args.dm_tolerance_bins,
            time_padding_bins=args.time_padding_bins,
        )

        argmax_host, token_policy = generate_valid_argmax_maps(
            bundle.specs, args.total_beams, args.base_seed
        )
        token_decodes = validate_tokens_with_plan(
            bundle.plan, bundle.specs, token_policy
        )
        context_host = generate_clean_context_maps(
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
        signature_payload = _signature_payload(
            args, bundle, gaussian_config, gpu, software
        )
        signature = campaign_signature(signature_payload)
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
        validate_trial_prefix(
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
                geometries=geometries,
                gpu=gpu,
                software=software,
                git=git,
                signature=signature,
                campaign_id=campaign_id,
                created_utc=created_utc,
                rows=rows,
                signature_payload=signature_payload,
            )
            write_checkpoint(
                paths, rows, args.percentages, campaign_id, metadata
            )

        checkpoint()
        expected_keys = expected_trial_keys(args.trials, args.percentages)
        if len(rows) == len(expected_keys):
            print("Campaign is already complete; nothing to resume.", flush=True)
            _print_aggregate(rows, args.percentages)
            return

        if args.allow_unsafe_grouping:
            print(
                "WARNING: --allow-unsafe-grouping is active. The production grouper "
                "can be quadratic and may take a very long time or exhaust resources.",
                file=sys.stderr,
                flush=True,
            )

        # Empty-path warm-up constructs all result types without entering any timer.
        from pirate_frb.Peakfinders import GpuRawCandidates
        empty_decoded = decoder.decode(GpuRawCandidates.empty())
        group_candidates(
            empty_decoded, grouping_geometry, config=grouping_config
        )
        stream.synchronize()
        decoder_nonempty_warmed = False
        grouper_nonempty_warmed = False
        required_target_counts = tuple(
            target_count_from_percentage(percentage)
            for percentage in args.percentages
        )
        maximum_target = max(required_target_counts)

        for trial in range(args.trials):
            trial_keys = {
                (int(row["trial"]), canonical_percentage(
                    row["target_corruption_percent"]
                ))
                for row in rows
            }
            pending = [
                percentage for percentage in args.percentages
                if (trial, canonical_percentage(percentage)) not in trial_keys
            ]
            if not pending:
                continue
            try:
                morphology = generate_trial_morphology(
                    bundle.specs,
                    base_seed=args.base_seed,
                    trial=trial,
                    total_beams=args.total_beams,
                    gaussian_config=gaussian_config,
                    threshold=args.threshold,
                    maximum_target_count=maximum_target,
                    attempt_limit=args.template_attempt_limit,
                    calibration_pixel_tolerance=args.calibration_pixel_tolerance,
                    calibration_max_iterations=args.calibration_max_iterations,
                    required_target_counts=required_target_counts,
                )
            except MorphologyGenerationError as exc:
                for percentage in pending:
                    rows.append(_blank_failed_row(
                        campaign_id=campaign_id,
                        args=args,
                        trial=trial,
                        percentage=percentage,
                        status="failed_template_unattainable",
                        reason=str(exc),
                    ))
                    validate_trial_prefix(
                        rows, args.trials, args.percentages, campaign_id
                    )
                    checkpoint()
                continue

            for percentage in args.percentages:
                key = (trial, canonical_percentage(percentage))
                if key in trial_keys:
                    continue
                target_count = target_count_from_percentage(percentage)
                unattainable_beams = [
                    beam.beam_id for beam in morphology.beams
                    if target_count not in beam.validated_target_counts
                ]
                if unattainable_beams:
                    preview = ",".join(map(str, unattainable_beams[:12]))
                    suffix = "..." if len(unattainable_beams) > 12 else ""
                    rows.append(_blank_failed_row(
                        campaign_id=campaign_id,
                        args=args,
                        trial=trial,
                        percentage=percentage,
                        status="failed_template_unattainable",
                        reason=(
                            f"rounded per-beam target {target_count} did not calibrate "
                            f"within {args.template_attempt_limit} deterministic template "
                            f"attempts for beam(s) {preview}{suffix}"
                        ),
                    ))
                    validate_trial_prefix(
                        rows, args.trials, args.percentages, campaign_id
                    )
                    checkpoint()
                    continue
                try:
                    target_host, calibrations = calibrate_trial_morphology(
                        morphology,
                        bundle.specs,
                        percentage,
                        args.threshold,
                        pixel_tolerance=args.calibration_pixel_tolerance,
                        max_iterations=args.calibration_max_iterations,
                    )
                    pixel_counts = validate_target_maps(
                        target_host, bundle.specs, args.total_beams,
                        args.threshold,
                    )
                except (ValueError, AssertionError) as exc:
                    rows.append(_blank_failed_row(
                        campaign_id=campaign_id,
                        args=args,
                        trial=trial,
                        percentage=percentage,
                        status="failed_calibration",
                        reason=str(exc),
                    ))
                    validate_trial_prefix(
                        rows, args.trials, args.percentages, campaign_id
                    )
                    checkpoint()
                    continue

                target_gpu = tuple(cp.asarray(array) for array in target_host)
                del target_host
                stream.synchronize()
                raw, raw_counts = extract_complete_target_candidates(
                    cp,
                    geometries,
                    target_gpu,
                    context_gpu,
                    argmax_gpu,
                    tuple(range(args.total_beams)),
                    args.threshold,
                    concatenate_raw_candidates,
                )
                del target_gpu
                total_raw = len(raw)

                predecode_safety = grouping_safety_status(
                    total_raw,
                    args.candidate_safety_limit,
                    args.allow_unsafe_grouping,
                )
                # A small first non-empty production batch warms candidate-sized
                # paths outside authoritative timers.  A configuration above the
                # safety limit never reaches even this bounded grouper warm-up.
                needs_warmup = (
                    not decoder_nonempty_warmed
                    or (predecode_safety == "run" and not grouper_nonempty_warmed)
                )
                if total_raw and needs_warmup:
                    warm_raw = _raw_prefix(raw, min(32, args.candidate_safety_limit))
                    warm_decoded = decoder.decode(warm_raw)
                    decoder_nonempty_warmed = True
                    if predecode_safety == "run" and not grouper_nonempty_warmed:
                        group_candidates(
                            warm_decoded, grouping_geometry, config=grouping_config
                        )
                        grouper_nonempty_warmed = True
                    stream.synchronize()

                decoder_wall_ms, decoded = synchronized_wall_time(
                    stream, lambda: decoder.decode(raw)
                )
                decoded_count = len(decoded)
                if decoded_count != total_raw:
                    raise AssertionError("production decoder silently changed candidate count")

                safety = grouping_safety_status(
                    decoded_count, args.candidate_safety_limit,
                    args.allow_unsafe_grouping,
                )
                if safety != predecode_safety:
                    raise AssertionError(
                        "decoder count changed the grouping safety decision"
                    )
                if safety == "skipped_candidate_safety_limit":
                    grouper_wall_ms = None
                    event_counts = None
                    status = safety
                    failure_reason = (
                        f"{decoded_count} candidates exceed safety limit "
                        f"{args.candidate_safety_limit}"
                    )
                else:
                    if (decoded_count > args.candidate_safety_limit
                            and args.allow_unsafe_grouping):
                        print(
                            f"WARNING: trial={trial} corruption={percentage:g}% is "
                            f"grouping {decoded_count} candidates above safety limit "
                            f"{args.candidate_safety_limit}",
                            file=sys.stderr,
                            flush=True,
                        )
                    grouper_wall_ms, grouped = synchronized_wall_time(
                        stream,
                        lambda: group_candidates(
                            decoded, grouping_geometry, config=grouping_config
                        ),
                    )
                    if (len(grouped.candidates) != decoded_count
                            or len(grouped.members) != decoded_count):
                        raise AssertionError("production grouping lost candidate members")
                    event_counts = gpu_per_beam_counts(
                        cp, grouped.events.beam_id, args.total_beams
                    )
                    if int(np.sum(event_counts)) != len(grouped.events):
                        raise AssertionError("per-beam event counts do not sum to total")
                    del grouped
                    status = "completed"
                    failure_reason = ""

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
                    grouper_wall_ms=grouper_wall_ms,
                    event_counts=event_counts,
                    status=status,
                    chunk_duration_ms=bundle.chunk_duration_ms,
                    failure_reason=failure_reason,
                )
                rows.append(row)
                validate_trial_prefix(
                    rows, args.trials, args.percentages, campaign_id
                )
                checkpoint()
                print(
                    f"checkpoint trial={trial} corruption={percentage:g}% "
                    f"pixels={int(np.sum(pixel_counts))} candidates={total_raw} "
                    f"decoder={decoder_wall_ms:.3f}ms "
                    f"grouper={'skipped' if grouper_wall_ms is None else f'{grouper_wall_ms:.3f}ms'} "
                    f"status={status}",
                    flush=True,
                )
                del raw, decoded, calibrations
                stream.synchronize()
            del morphology

        checkpoint()
        for percentage in args.percentages:
            _print_percentage_summary(rows, percentage)
        _print_aggregate(rows, args.percentages)
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
