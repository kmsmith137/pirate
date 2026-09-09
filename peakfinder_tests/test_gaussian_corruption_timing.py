"""Focused tests for the Gaussian-corruption candidate-pipeline benchmark.

The morphology, calibration, schema, resume, and notebook tests deliberately use
small NumPy maps.  The final test exercises the production GPU path when CuPy,
PIRATE, and a CUDA device are available; it is skipped cleanly otherwise.
"""

from __future__ import annotations

import csv
import json
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest
import yaml

from . import benchmark_gaussian_corruption_timing as benchmark
from .benchmark_gaussian_corruption_timing import (
    ARGMAX_DTYPE,
    DEFAULT_BASE_SEED,
    DEFAULT_BEAM_BATCH_SIZE,
    DEFAULT_CANDIDATE_SAFETY_LIMIT,
    DEFAULT_DM_REACH,
    DEFAULT_PERCENTAGES,
    DEFAULT_THRESHOLD,
    DEFAULT_TOTAL_BEAMS,
    DEFAULT_TRIALS,
    DEFAULT_WAIST_BINS,
    EXPECTED_SHAPES,
    PIXELS_PER_BEAM,
    SCHEMA_NAME,
    SCHEMA_VERSION,
    SNR_DTYPE,
    SUMMARY_FIELDS,
    SUMMARY_METRICS,
    SUMMARY_STATISTICS,
    TRIAL_FIELDS,
    BeamMorphology,
    GaussianComponent,
    GaussianConfig,
    add_gaussian_component,
    build_metadata,
    calibrate_beam,
    calibrate_trial_morphology,
    campaign_signature,
    count_above_threshold,
    derive_post_peakfinder_metrics,
    derive_seed,
    generate_beam_morphology,
    generate_trial_morphology,
    generate_valid_argmax_maps,
    generate_white_noise,
    gpu_per_beam_counts,
    grouping_safety_status,
    maps_at_scale,
    prepare_result_state,
    render_gaussian_template,
    result_paths,
    sample_gaussian_components,
    summarize_counts,
    summarize_trials,
    target_count_from_percentage,
    validate_target_maps,
    validate_tokens_with_plan,
    write_checkpoint,
)


TEST_CHUNK_DURATION_MS = 2048.0


_PRIMARY = (0, 1, 1, 2, 2, 2, 3, 3, 3, 3)
_EARLY = (0, 1, 0, 2, 1, 0, 3, 2, 1, 0)
_MULTIPLETS = (91, 86, 91, 34, 86, 91, 20, 34, 86, 91)
_PROFILES = (16, 13, 13, 10, 10, 10, 7, 7, 7, 7)
_DOUT = (16, 16, 16, 8, 16, 16, 8, 8, 16, 16)


def test_cli_uses_the_only_production_peakfinder_implicitly():
    parser = benchmark._make_arg_parser()
    assert "--method" not in parser.format_help()
    with pytest.raises(SystemExit):
        parser.parse_args(["--method", "full_band"])
    args = benchmark.validate_arguments(parser.parse_args([]))
    assert args.method == benchmark.METHOD == "full_band"


def _fake_plan():
    trees = []
    for index, ((ndm, ntime), multiplets, profiles, dout) in enumerate(
            zip(EXPECTED_SHAPES, _MULTIPLETS, _PROFILES, _DOUT)):
        trees.append(SimpleNamespace(
            ndm_out=ndm,
            nt_out=ntime,
            nt_ds=ntime * dout,
            primary_tree_index=_PRIMARY[index],
            early_trigger_level=_EARLY[index],
            nprofiles=profiles,
            dm_downsampling=1,
            frequency_subbands=SimpleNamespace(M=multiplets, pf_rank=0),
        ))
    return SimpleNamespace(ntrees=10, trees=tuple(trees), nt_in=2048)


def _specs():
    return benchmark.batch_benchmark.assert_expected_plan(_fake_plan(), dcores=_DOUT)


def _toy_specs():
    return (
        SimpleNamespace(
            tree_index=0, ndm=24, ntime=20, multiplets=7, shape=(24, 20)
        ),
        SimpleNamespace(
            tree_index=1, ndm=16, ntime=16, multiplets=5, shape=(16, 16)
        ),
    )


def _toy_gaussian_config():
    return GaussianConfig(
        count_min=3,
        count_max=3,
        dm_width_min=2.0,
        dm_width_max=4.0,
        time_width_min=1.0,
        time_width_max=3.0,
        max_abs_correlation=0.5,
        relative_amplitude_min=0.7,
        relative_amplitude_max=1.4,
        render_sigma_cutoff=5.0,
    )


def _small_trial_morphology(*, trial=2, base_seed=55):
    return generate_trial_morphology(
        _toy_specs(),
        base_seed=base_seed,
        trial=trial,
        total_beams=2,
        gaussian_config=_toy_gaussian_config(),
        threshold=DEFAULT_THRESHOLD,
        maximum_target_count=2,
        attempt_limit=3,
        calibration_pixel_tolerance=0,
        calibration_max_iterations=64,
    )


def test_fixed_campaign_exact_plan_geometry_and_rounded_targets():
    assert DEFAULT_TOTAL_BEAMS == DEFAULT_BEAM_BATCH_SIZE == 60
    assert DEFAULT_TRIALS == 50
    assert DEFAULT_THRESHOLD == 10.0
    assert DEFAULT_DM_REACH == 8
    assert DEFAULT_WAIST_BINS == 1
    assert DEFAULT_BASE_SEED == 20260825
    assert DEFAULT_PERCENTAGES == (
        0.0, 0.001, 0.003, 0.01, 0.03, 0.1, 0.3, 1.0
    )

    specs = _specs()
    assert len(specs) == 10
    assert tuple(spec.shape for spec in specs) == EXPECTED_SHAPES
    assert sum(spec.pixels_per_beam for spec in specs) == PIXELS_PER_BEAM == 983_040
    assert DEFAULT_TOTAL_BEAMS * PIXELS_PER_BEAM == 58_982_400
    assert tuple(spec.primary_tree_index for spec in specs) == _PRIMARY
    assert tuple(spec.early_trigger_level for spec in specs) == _EARLY

    expected_counts = (0, 10, 29, 98, 295, 983, 2949, 9830)
    assert tuple(
        target_count_from_percentage(value) for value in DEFAULT_PERCENTAGES
    ) == expected_counts
    assert target_count_from_percentage(50.0, pixels_per_beam=3) == 2


def test_active_authoritative_plan_geometry_when_runtime_is_importable():
    """Exercise the real YAML/extension path when the active runtime can load."""

    try:
        bundle = benchmark.batch_benchmark.load_authoritative_plan(
            benchmark.DEFAULT_CONFIG
        )
    except Exception as exc:  # pragma: no cover - environment dependent
        pytest.skip(f"active authoritative plan runtime is unavailable: {exc}")
    assert tuple(spec.shape for spec in bundle.specs) == EXPECTED_SHAPES
    assert sum(spec.pixels_per_beam for spec in bundle.specs) == PIXELS_PER_BEAM


def test_white_noise_components_and_trial_generation_are_deterministic():
    specs = _toy_specs()
    seed = derive_seed(123, 4, 1, "white_noise")
    assert seed == derive_seed(123, 4, 1, "white_noise")
    assert seed != derive_seed(123, 4, 0, "white_noise")
    assert seed != derive_seed(123, 4, 1, "gaussian_template")

    noise = generate_white_noise(specs, seed, DEFAULT_THRESHOLD)
    repeated_noise = generate_white_noise(specs, seed, DEFAULT_THRESHOLD)
    other_noise = generate_white_noise(specs, seed + 1, DEFAULT_THRESHOLD)
    for first, repeated in zip(noise, repeated_noise):
        assert first.dtype == np.float32
        assert np.array_equal(first, repeated)
        assert np.all(first.astype(SNR_DTYPE) < DEFAULT_THRESHOLD)
    assert any(
        not np.array_equal(first, other)
        for first, other in zip(noise, other_noise)
    )

    config = _toy_gaussian_config()
    components = sample_gaussian_components(
        specs, np.random.default_rng(999), config
    )
    repeated_components = sample_gaussian_components(
        specs, np.random.default_rng(999), config
    )
    other_components = sample_gaussian_components(
        specs, np.random.default_rng(1000), config
    )
    assert components == repeated_components
    assert components != other_components
    assert len(components) == 3
    assert all(0 <= item.tree_index < len(specs) for item in components)

    trial = _small_trial_morphology()
    repeated_trial = _small_trial_morphology()
    assert trial.seed == repeated_trial.seed
    assert len(trial.beams) == 2
    for beam, repeated_beam in zip(trial.beams, repeated_trial.beams):
        assert beam.components == repeated_beam.components
        assert beam.noise_seed == repeated_beam.noise_seed
        assert beam.template_seed == repeated_beam.template_seed
        for first, repeated in zip(beam.noise_by_tree, repeated_beam.noise_by_tree):
            assert np.array_equal(first, repeated)
        for first, repeated in zip(
                beam.template_by_tree, repeated_beam.template_by_tree):
            assert np.array_equal(first, repeated)

    # Different beams have independent seed streams and complete morphologies.
    left, right = trial.beams
    assert left.noise_seed != right.noise_seed
    assert left.template_seed != right.template_seed
    assert left.components != right.components
    assert any(
        not np.array_equal(first, second)
        for first, second in zip(left.noise_by_tree, right.noise_by_tree)
    )


def test_trial_morphology_is_reused_unchanged_across_corruption_levels():
    specs = _toy_specs()
    morphology = _small_trial_morphology()
    fingerprints = tuple(
        (
            tuple(array.tobytes() for array in beam.noise_by_tree),
            tuple(array.tobytes() for array in beam.template_by_tree),
            beam.components,
        )
        for beam in morphology.beams
    )

    zero_maps, zero_calibrations = calibrate_trial_morphology(
        morphology, specs, 0.0, DEFAULT_THRESHOLD,
        pixel_tolerance=0, max_iterations=64,
    )
    positive_maps, positive_calibrations = calibrate_trial_morphology(
        morphology, specs, 0.0002, DEFAULT_THRESHOLD,
        pixel_tolerance=0, max_iterations=64,
    )
    repeated_maps, repeated_calibrations = calibrate_trial_morphology(
        morphology, specs, 0.0002, DEFAULT_THRESHOLD,
        pixel_tolerance=0, max_iterations=64,
    )

    assert all(item.achieved_count == 0 for item in zero_calibrations)
    assert all(item.achieved_count == 2 for item in positive_calibrations)
    assert [item.scale for item in positive_calibrations] == [
        item.scale for item in repeated_calibrations
    ]
    assert all(
        np.array_equal(first, repeated)
        for first, repeated in zip(positive_maps, repeated_maps)
    )
    assert any(np.any(array != zero) for array, zero in zip(positive_maps, zero_maps))

    after = tuple(
        (
            tuple(array.tobytes() for array in beam.noise_by_tree),
            tuple(array.tobytes() for array in beam.template_by_tree),
            beam.components,
        )
        for beam in morphology.beams
    )
    assert after == fingerprints


def test_gaussian_rendering_clips_at_edges_without_wraparound():
    target = np.zeros((11, 13), dtype=np.float32)
    component = GaussianComponent(
        tree_index=0,
        centre_dm=0.0,
        centre_time=0.0,
        sigma_dm=0.75,
        sigma_time=0.5,
        correlation=0.3,
        relative_amplitude=2.0,
    )
    add_gaussian_component(target, component, sigma_cutoff=2.0)
    assert target[0, 0] == pytest.approx(2.0)
    assert np.all(target >= 0.0)
    assert np.count_nonzero(target) > 1
    assert np.all(target[3:, :] == 0.0)
    assert np.all(target[:, 2:] == 0.0)
    assert target[-1, -1] == 0.0

    maps = render_gaussian_template(
        _toy_specs(), (component,), sigma_cutoff=2.0
    )
    assert maps[0].shape == (24, 20)
    assert maps[1].shape == (16, 16)
    assert np.count_nonzero(maps[0]) > 0
    assert np.count_nonzero(maps[1]) == 0


def test_calibration_is_deterministic_validates_float16_and_enforces_tolerance():
    noise = np.zeros((10, 10), dtype=np.float32)
    template = np.linspace(
        0.1, 2.0, noise.size, dtype=np.float32
    ).reshape(noise.shape)
    morphology = BeamMorphology(
        beam_id=7,
        noise_by_tree=(noise,),
        template_by_tree=(template,),
        components=(),
        noise_seed=1,
        template_seed=2,
        template_attempt=0,
    )
    first = calibrate_beam(
        morphology, 7, DEFAULT_THRESHOLD,
        pixel_tolerance=0, max_iterations=64,
    )
    repeated = calibrate_beam(
        morphology, 7, DEFAULT_THRESHOLD,
        pixel_tolerance=0, max_iterations=64,
    )
    assert first.scale == repeated.scale
    assert first.achieved_count == first.target_count == 7
    assert first.pixel_count_error == 0
    assert first.maps_by_tree[0].dtype == SNR_DTYPE
    assert np.array_equal(first.maps_by_tree[0], repeated.maps_by_tree[0])
    assert count_above_threshold(first.maps_by_tree, DEFAULT_THRESHOLD) == 7
    assert np.count_nonzero(first.maps_by_tree[0] >= DEFAULT_THRESHOLD) == 7

    zero = calibrate_beam(
        morphology, 0, DEFAULT_THRESHOLD,
        pixel_tolerance=0, max_iterations=64,
    )
    assert zero.scale == 0.0
    assert zero.achieved_count == 0
    assert count_above_threshold(zero.maps_by_tree, DEFAULT_THRESHOLD) == 0

    tied = BeamMorphology(
        beam_id=0,
        noise_by_tree=(noise,),
        template_by_tree=(np.ones_like(noise),),
        components=(),
        noise_seed=1,
        template_seed=2,
        template_attempt=0,
    )
    with pytest.raises(ValueError, match="tolerance 0"):
        calibrate_beam(
            tied, 3, DEFAULT_THRESHOLD,
            pixel_tolerance=0, max_iterations=64,
        )

    with pytest.raises(ValueError, match="non-finite production"):
        maps_at_scale(
            (noise,), (np.ones_like(noise),),
            2.0 * float(np.finfo(np.float16).max),
        )


def test_template_attempts_retain_paired_attainable_lower_targets():
    specs = _toy_specs()
    impossible = sum(spec.ndm * spec.ntime for spec in specs) + 1
    morphology = generate_beam_morphology(
        specs,
        base_seed=17,
        trial=0,
        beam_id=0,
        gaussian_config=_toy_gaussian_config(),
        threshold=DEFAULT_THRESHOLD,
        maximum_target_count=impossible,
        attempt_limit=2,
        calibration_pixel_tolerance=0,
        calibration_max_iterations=64,
        required_target_counts=(0, 2, impossible),
    )
    assert 0 in morphology.validated_target_counts
    assert 2 in morphology.validated_target_counts
    assert impossible not in morphology.validated_target_counts
    assert morphology.template_attempt in (0, 1)


def test_target_map_validation_counts_every_beam_and_tree():
    specs = _toy_specs()
    maps = [
        np.zeros((3, spec.ndm, spec.ntime), dtype=SNR_DTYPE)
        for spec in specs
    ]
    maps[0][0, 1, 1] = np.float16(10.0)
    maps[0][1, 2, 2] = np.float16(11.0)
    maps[1][1, 3, 3] = np.float16(12.0)
    maps[1][2, 4, 4] = np.float16(20.0)
    counts = validate_target_maps(maps, specs, 3, DEFAULT_THRESHOLD)
    assert counts.tolist() == [1, 2, 1]
    assert int(np.sum(counts)) == 4

    wrong = list(maps)
    wrong[0] = wrong[0].astype(np.float32)
    with pytest.raises(ValueError, match="float16"):
        validate_target_maps(wrong, specs, 3, DEFAULT_THRESHOLD)


def test_valid_constant_argmax_tokens_cover_all_ten_trees():
    specs = _specs()
    maps, policy = generate_valid_argmax_maps(specs, 2, DEFAULT_BASE_SEED)
    assert len(maps) == len(policy) == 10
    for spec, array, item in zip(specs, maps, policy):
        assert array.shape == (2, spec.ndm, spec.ntime)
        assert array.dtype == ARGMAX_DTYPE
        assert np.unique(array).tolist() == [item["token_uint32"]]
        token = np.uint32(item["token_uint32"])
        assert int(token & np.uint32(0xFF)) == 0  # fine time
        assert int((token >> np.uint32(8)) & np.uint32(0xFF)) == 0  # profile
        assert int((token >> np.uint32(16)) & np.uint32(0xFF)) == item["multiplet"]
        assert int(token >> np.uint32(24)) == item["extra_dm"] == 0
        assert 0 <= item["multiplet"] < spec.multiplets
        assert token != np.uint32(0xFFFFFFFF)

    class CheckingPlan:
        def decode_argmax(self, token, tree_index, dcore, idm, itime):
            assert dcore == specs[tree_index].dcore
            assert token & 0xFFFF == 0
            assert token >> 16 < specs[tree_index].multiplets
            assert idm == itime == 0
            return (tree_index, idm, itime, token >> 16, 0)

    decoded = validate_tokens_with_plan(CheckingPlan(), specs, policy)
    assert len(decoded) == 10
    assert all(values[-1] == 0 for values in decoded)


def test_count_summaries_timing_formulas_and_candidate_safety():
    assert summarize_counts([1, 7, 4, 9]) == {
        "min": 1.0, "median": 5.5, "max": 9.0
    }
    metrics = derive_post_peakfinder_metrics(
        decoder_wall_ms=2.0,
        grouper_wall_ms=6.0,
        decoded_candidates=16,
        chunk_duration_ms=TEST_CHUNK_DURATION_MS,
    )
    assert metrics == pytest.approx({
        "decoder_candidates_per_second": 8_000.0,
        "grouper_candidates_per_second": 16_000.0 / 6.0,
        "decoder_plus_grouper_wall_ms": 8.0,
        "post_peakfinder_load_fraction": 8.0 / TEST_CHUNK_DURATION_MS,
    })
    longer_duration = 1.31072 * 2048
    assert derive_post_peakfinder_metrics(
        2.0, 6.0, 16, longer_duration
    )["post_peakfinder_load_fraction"] == pytest.approx(
        8.0 / longer_duration
    )
    assert derive_post_peakfinder_metrics(
        0.0, 0.0, 0, TEST_CHUNK_DURATION_MS
    ) == {
        "decoder_candidates_per_second": 0.0,
        "grouper_candidates_per_second": 0.0,
        "decoder_plus_grouper_wall_ms": 0.0,
        "post_peakfinder_load_fraction": 0.0,
    }
    with pytest.raises(ValueError, match="non-negative"):
        derive_post_peakfinder_metrics(
            -1.0, 0.0, 0, TEST_CHUNK_DURATION_MS
        )

    limit = DEFAULT_CANDIDATE_SAFETY_LIMIT
    assert grouping_safety_status(limit, limit, False) == "run"
    assert grouping_safety_status(limit + 1, limit, False) == (
        "skipped_candidate_safety_limit"
    )
    assert grouping_safety_status(limit + 1, limit, True) == "run"


def test_gpu_per_beam_counts_avoids_empty_cupy_bincount_reduction():
    class FakeIds:
        def __init__(self, values):
            self.values = np.asarray(values, dtype=np.int32)
            self.ndim = self.values.ndim
            self.size = self.values.size

        def astype(self, dtype):
            return self.values.astype(dtype)

    class FakeCupy:
        int64 = np.int64
        bincount_calls = 0

        @classmethod
        def bincount(cls, values, minlength):
            cls.bincount_calls += 1
            return np.bincount(values, minlength=minlength)

        @staticmethod
        def asnumpy(values):
            return np.asarray(values)

    empty = gpu_per_beam_counts(FakeCupy, FakeIds([]), 3)
    assert empty.dtype == np.int64
    assert empty.tolist() == [0, 0, 0]
    assert FakeCupy.bincount_calls == 0

    populated = gpu_per_beam_counts(FakeCupy, FakeIds([0, 1, 1]), 3)
    assert populated.tolist() == [1, 2, 0]
    assert FakeCupy.bincount_calls == 1

    with pytest.raises(ValueError, match="outside"):
        gpu_per_beam_counts(FakeCupy, FakeIds([3]), 3)


def _complete_trial_row(*, trial=0, percentage=0.0,
                        campaign_id="test-campaign"):
    row = {field: "" for field in TRIAL_FIELDS}
    row.update({
        "schema_version": SCHEMA_VERSION,
        "campaign_id": campaign_id,
        "trial": trial,
        "seed": 123,
        "total_beams": 60,
        "beam_batch_size": 60,
        "threshold": 10.0,
        "method": "full_band",
        "dm_reach": 8,
        "waist_bins": 1,
        "target_corruption_percent": percentage,
        "achieved_corruption_percent": percentage,
        "target_pixels_above_threshold": 0,
        "achieved_pixels_above_threshold": 0,
        "pixel_count_error": 0,
        "max_abs_beam_pixel_count_error": 0,
        "per_beam_calibration_json": "[]",
        "total_gaussians": 1800,
        "gaussians_per_beam_min": 30,
        "gaussians_per_beam_median": 30.0,
        "gaussians_per_beam_max": 30,
        "gaussian_dm_width_bins_min": 2.0,
        "gaussian_dm_width_bins_median": 4.0,
        "gaussian_dm_width_bins_max": 8.0,
        "gaussian_time_width_bins_min": 1.0,
        "gaussian_time_width_bins_median": 2.0,
        "gaussian_time_width_bins_max": 4.0,
        "gaussian_correlation_min": -0.5,
        "gaussian_correlation_median": 0.0,
        "gaussian_correlation_max": 0.5,
        "gaussian_relative_amplitude_min": 0.5,
        "gaussian_relative_amplitude_median": 1.0,
        "gaussian_relative_amplitude_max": 2.0,
        "template_attempt_per_beam_min": 0,
        "template_attempt_per_beam_median": 0.0,
        "template_attempt_per_beam_max": 0,
        "total_pixels_above_threshold": 0,
        "pixels_above_threshold_per_beam_min": 0,
        "pixels_above_threshold_per_beam_median": 0.0,
        "pixels_above_threshold_per_beam_max": 0,
        "total_peakfinder_candidates": 0,
        "peakfinder_candidates_per_beam_min": 0,
        "peakfinder_candidates_per_beam_median": 0.0,
        "peakfinder_candidates_per_beam_max": 0,
        "peakfinder_survival_fraction": 0.0,
        "decoder_wall_ms": 0.25,
        "decoder_candidates_per_second": 0.0,
        "total_decoded_candidates": 0,
        "grouper_wall_ms": 0.5,
        "grouper_candidates_per_second": 0.0,
        "total_grouped_events": 0,
        "grouped_events_per_beam_min": 0,
        "grouped_events_per_beam_median": 0.0,
        "grouped_events_per_beam_max": 0,
        "decoder_plus_grouper_wall_ms": 0.75,
        "post_peakfinder_load_fraction": 0.75 / TEST_CHUNK_DURATION_MS,
        "candidate_safety_limit": DEFAULT_CANDIDATE_SAFETY_LIMIT,
        "status": "completed",
        "failure_reason": "",
    })
    assert tuple(row) == TRIAL_FIELDS
    return row


def test_summary_csv_schemas_resume_and_overwrite(tmp_path):
    rows = [
        _complete_trial_row(trial=0),
        _complete_trial_row(trial=1),
    ]
    summaries = summarize_trials(rows, (0.0,), "test-campaign")
    assert len(summaries) == 1
    summary = summaries[0]
    assert tuple(summary) == SUMMARY_FIELDS
    assert summary["recorded_configurations"] == 2
    assert summary["completed_trials"] == 2
    assert len(SUMMARY_FIELDS) == 7 + len(SUMMARY_METRICS) * len(
        SUMMARY_STATISTICS
    )
    for metric in SUMMARY_METRICS:
        values = np.asarray([float(row[metric]) for row in rows])
        assert summary[f"{metric}_median"] == pytest.approx(np.median(values))
        assert summary[f"{metric}_minimum"] == pytest.approx(np.min(values))
        assert summary[f"{metric}_maximum"] == pytest.approx(np.max(values))
        assert summary[f"{metric}_q25"] == pytest.approx(np.quantile(values, 0.25))
        assert summary[f"{metric}_q75"] == pytest.approx(np.quantile(values, 0.75))
        assert summary[f"{metric}_iqr"] == pytest.approx(
            np.quantile(values, 0.75) - np.quantile(values, 0.25)
        )

    paths = result_paths(tmp_path / "results")
    signature_payload = {"trials": 2, "percentages": [0.0]}
    signature = campaign_signature(signature_payload)
    metadata = {
        "schema_name": SCHEMA_NAME,
        "schema_version": SCHEMA_VERSION,
        "campaign_id": "test-campaign",
        "campaign_signature": signature,
        "campaign_signature_payload": signature_payload,
        "created_utc": "2026-08-25T00:00:00+00:00",
    }
    write_checkpoint(paths, rows, (0.0,), "test-campaign", metadata)
    with paths["trials"].open(newline="", encoding="utf-8") as stream:
        reader = csv.DictReader(stream)
        loaded = list(reader)
        assert tuple(reader.fieldnames or ()) == TRIAL_FIELDS
    assert len(loaded) == 2
    with paths["summary"].open(newline="", encoding="utf-8") as stream:
        reader = csv.DictReader(stream)
        assert tuple(reader.fieldnames or ()) == SUMMARY_FIELDS
        assert len(list(reader)) == 1
    with paths["metadata"].open(encoding="utf-8") as stream:
        assert yaml.safe_load(stream) == metadata

    with pytest.raises(FileExistsError, match="--resume or --overwrite"):
        prepare_result_state(
            paths, resume=False, overwrite=False, signature=signature,
            campaign_id="test-campaign", trials=2, percentages=(0.0,),
        )
    resumed, created = prepare_result_state(
        paths, resume=True, overwrite=False, signature=signature,
        campaign_id="test-campaign", trials=2, percentages=(0.0,),
    )
    assert len(resumed) == 2
    assert created == metadata["created_utc"]
    fresh, fresh_created = prepare_result_state(
        paths, resume=False, overwrite=True, signature=signature,
        campaign_id="test-campaign", trials=2, percentages=(0.0,),
    )
    assert fresh == []
    assert isinstance(fresh_created, str) and fresh_created
    with pytest.raises(ValueError, match="incompatible"):
        prepare_result_state(
            paths, resume=True, overwrite=False, signature="0" * 64,
            campaign_id="test-campaign", trials=2, percentages=(0.0,),
        )

    parser = benchmark._make_arg_parser()
    with pytest.raises(SystemExit):
        parser.parse_args(["--resume", "--overwrite"])


def _metadata_args():
    return SimpleNamespace(
        trials=2,
        percentages=(0.0, 0.01),
        total_beams=60,
        beam_batch_size=60,
        device=0,
        base_seed=DEFAULT_BASE_SEED,
        config=benchmark.DEFAULT_CONFIG,
        threshold=DEFAULT_THRESHOLD,
        method="full_band",
        dm_reach=DEFAULT_DM_REACH,
        waist_bins=DEFAULT_WAIST_BINS,
        calibration_pixel_tolerance=1,
        calibration_max_iterations=32,
        template_attempt_limit=8,
        dm_tolerance_bins=1.5,
        time_padding_bins=1.0,
        candidate_safety_limit=DEFAULT_CANDIDATE_SAFETY_LIMIT,
        allow_unsafe_grouping=False,
    )


def test_metadata_records_reproducibility_timing_and_safety_contracts():
    specs = _specs()
    args = _metadata_args()
    bundle = SimpleNamespace(
        config_document={"test": True},
        producer_plan_yaml="test-plan-yaml",
        dcores=tuple(spec.dcore for spec in specs),
        argmax_encoding=benchmark.batch_benchmark.ARGMAX_ENCODING,
        specs=specs,
        chunk_duration_ms=TEST_CHUNK_DURATION_MS,
    )
    token_policy = tuple({
        "tree_index": spec.tree_index,
        "multiplet": 0,
        "profile": 0,
        "fine_time": 0,
        "token_uint32": 0,
    } for spec in specs)
    token_decodes = tuple((0, 0, 0, 0, 0) for _ in specs)
    geometries = tuple(
        SimpleNamespace(time_radius=1, ntime=spec.ntime) for spec in specs
    )
    gaussian_config = _toy_gaussian_config()
    gpu = {"model": "test GPU", "cuda_runtime_version": 12000}
    software = {"python": "test", "cupy": "test"}
    signature_payload = benchmark._signature_payload(
        args, bundle, gaussian_config, gpu, software
    )
    signature = campaign_signature(signature_payload)
    metadata = build_metadata(
        args=args,
        bundle=bundle,
        gaussian_config=gaussian_config,
        token_policy=token_policy,
        token_decodes=token_decodes,
        geometries=geometries,
        gpu=gpu,
        software=software,
        git={"commit": "deadbeef", "dirty": True},
        signature=signature,
        campaign_id=signature[:16],
        created_utc="2026-08-25T00:00:00+00:00",
        rows=(),
        signature_payload=signature_payload,
    )
    required_sections = {
        "schema_name", "schema_version", "campaign", "plan",
        "corruption_model", "argmax_tokens", "peakfinder",
        "streaming_state", "decoder_timing", "grouper_timing",
        "safety_limit", "environment", "csv_schemas",
        "campaign_completeness",
    }
    assert required_sections <= set(metadata)
    assert metadata["schema_name"] == SCHEMA_NAME
    assert metadata["schema_version"] == SCHEMA_VERSION == 2
    assert metadata["plan"]["dcores"] == list(bundle.dcores)
    assert metadata["plan"]["argmax_encoding"] == bundle.argmax_encoding
    assert signature_payload["dcores"] == list(bundle.dcores)
    assert signature_payload["argmax_encoding"] == bundle.argmax_encoding
    changed_payload = dict(signature_payload, dcores=[1] * len(bundle.dcores))
    assert campaign_signature(changed_payload) != signature
    assert campaign_signature(metadata["campaign_signature_payload"]) == signature
    assert metadata["plan"]["tree_shapes_ndm_ntime"] == [
        list(shape) for shape in EXPECTED_SHAPES
    ]
    assert metadata["plan"]["pixels_per_beam"] == 983_040
    assert metadata["plan"]["chunk_duration_ms"] == TEST_CHUNK_DURATION_MS
    assert metadata["campaign"]["chunk_duration_ms"] == TEST_CHUNK_DURATION_MS
    assert metadata["campaign"]["total_pixels_per_trial"] == 58_982_400
    corruption = metadata["corruption_model"]
    assert corruption["name"] == "gaussian_mixture"
    assert "float16" in corruption["percentage_convention"]
    assert corruption["generation_dtype"] == "float32"
    assert corruption["production_snr_dtype"] == "float16"
    assert "order statistic" in corruption["calibration"]["algorithm"]
    assert "noise and templates" in corruption["reuse_policy"]
    assert "SHA256" in corruption["seed_policy"]
    assert "process_chunk" in metadata["streaming_state"]["priming"]
    assert "right seam" in metadata["streaming_state"]["right_seam"]
    assert "perf_counter" in metadata["decoder_timing"]["boundary"]
    assert "one joint" in metadata["grouper_timing"]["production_strategy"]
    assert "equal beam_id" in metadata["grouper_timing"]["production_strategy"]
    assert "separately synchronized" in metadata["grouper_timing"][
        "combined_metric"
    ]
    assert "skipped_candidate_safety_limit" in metadata["safety_limit"]["policy"]
    assert metadata["csv_schemas"]["trials.csv"] == list(TRIAL_FIELDS)
    assert metadata["csv_schemas"]["summary.csv"] == list(SUMMARY_FIELDS)
    assert metadata["campaign_completeness"]["complete"] is False


def test_analysis_notebook_is_cleared_compilable_and_has_exact_nine_plots():
    path = Path(__file__).with_name("analyze_gaussian_corruption_timing.ipynb")
    notebook = json.loads(path.read_text(encoding="utf-8"))
    assert notebook["nbformat"] == 4
    code_cells = [cell for cell in notebook["cells"] if cell["cell_type"] == "code"]
    assert code_cells

    sources = {}
    for index, cell in enumerate(code_cells):
        assert cell.get("execution_count") is None
        assert cell.get("outputs") == []
        cell_id = cell.get("id")
        assert cell_id and cell_id not in sources
        source = "".join(cell.get("source", ()))
        compile(source, f"{path.name}:cell-{index}:{cell_id}", "exec")
        sources[cell_id] = source

    setup = sources["load-and-validate"] + sources["validate-schemas"]
    for filename in ("trials.csv", "summary.csv", "metadata.yaml"):
        assert filename in setup
    for concept in (
        "EXPECTED_SHAPES", "983_040", "gaussian_mixture",
        "total_pixels_per_trial", "decoder lost candidates",
        "campaign is incomplete", "peakfinder_survival_fraction",
        "decoder_plus_grouper_wall_ms", "post_peakfinder_load_fraction",
        "json.loads", "calibration_tolerance", "safety-skipped",
        "campaign signature does not match its payload",
    ):
        assert concept in setup

    plot_ids = {
        "achieved-vs-target",
        "candidates-vs-pixels",
        "survival-vs-corruption",
        "candidates-vs-gaussians",
        "decoder-vs-candidates",
        "grouper-vs-candidates",
        "events-vs-candidates",
        "grouper-vs-events",
        "post-peakfinder-vs-corruption",
    }
    show_ids = {
        cell_id for cell_id, source in sources.items() if "plt.show()" in source
    }
    assert show_ids == plot_ids
    assert len(plot_ids) == 9
    for cell_id in plot_ids:
        assert sources[cell_id].count("plt.show()") == 1
        assert "scatter(" in sources[cell_id]
    helpers = sources["plot-helpers"]
    for concept in (
        "np.quantile(values, 0.25)", "np.median(values)",
        "np.quantile(values, 0.75)", "fill_between", "errorbar",
    ):
        assert concept in helpers
    assert "coloured by corruption percentage" in sources[
        "candidates-vs-gaussians"
    ]
    markdown = "".join(
        "".join(cell.get("source", ())) for cell in notebook["cells"]
        if cell["cell_type"] == "markdown"
    )
    assert "candidate counts" in markdown


def _cuda_or_skip():
    cp = pytest.importorskip("cupy")
    try:
        if cp.cuda.runtime.getDeviceCount() < 1:
            pytest.skip("no CUDA device is available")
        cp.cuda.Device(0).use()
        cp.cuda.get_current_stream().synchronize()
    except Exception as exc:  # pragma: no cover - environment dependent
        pytest.skip(f"CUDA is unavailable: {exc}")
    try:
        from pirate_frb.GpuArgmaxDecoder import GpuArgmaxDecoder
        from pirate_frb.OfflineCandidateGrouper import (
            GroupingConfig,
            GroupingGeometry,
            group_candidates,
        )
        from pirate_frb.Peakfinders import concatenate_raw_candidates
    except (ImportError, OSError) as exc:  # pragma: no cover - environment dependent
        pytest.skip(f"production PIRATE GPU modules are unavailable: {exc}")
    return (
        cp, GpuArgmaxDecoder, GroupingConfig, GroupingGeometry,
        group_candidates, concatenate_raw_candidates,
    )


def test_cuda_production_path_candidate_decoder_and_grouper_counts():
    """Tiny optional integration check; this is not a benchmark campaign."""

    (
        cp, GpuArgmaxDecoder, GroupingConfig, GroupingGeometry,
        group_candidates, concatenate_raw_candidates,
    ) = _cuda_or_skip()
    bundle = benchmark.batch_benchmark.load_authoritative_plan(
        benchmark.DEFAULT_CONFIG
    )
    geometries_by_reach, _ = benchmark.batch_benchmark.build_geometries(
        cp, bundle.plan, bundle.specs, (DEFAULT_DM_REACH,),
        DEFAULT_WAIST_BINS,
    )
    geometries = geometries_by_reach[DEFAULT_DM_REACH]
    assert tuple(spec.shape for spec in bundle.specs) == EXPECTED_SHAPES

    total_beams = 1
    argmax_host, policy = generate_valid_argmax_maps(
        bundle.specs, total_beams, DEFAULT_BASE_SEED
    )
    validate_tokens_with_plan(bundle.plan, bundle.specs, policy)
    context_host = tuple(
        np.zeros((total_beams, spec.ndm, spec.ntime), dtype=SNR_DTYPE)
        for spec in bundle.specs
    )
    argmax_gpu = tuple(cp.asarray(array) for array in argmax_host)
    context_gpu = tuple(cp.asarray(array) for array in context_host)

    def extract(host_maps):
        return benchmark.extract_complete_target_candidates(
            cp,
            geometries,
            tuple(cp.asarray(array) for array in host_maps),
            context_gpu,
            argmax_gpu,
            (0,),
            DEFAULT_THRESHOLD,
            concatenate_raw_candidates,
        )

    zero_raw, zero_counts = extract(context_host)
    assert len(zero_raw) == 0
    assert zero_counts.tolist() == [0]

    isolated = [array.copy() for array in context_host]
    for spec, array in zip(bundle.specs, isolated):
        array[0, spec.ndm // 2, spec.ntime // 2] = np.float16(20.0)
    raw, raw_counts = extract(isolated)
    repeated_raw, repeated_counts = extract(isolated)
    assert len(raw) == len(bundle.specs)
    assert len(raw) == len(repeated_raw)
    assert raw_counts.tolist() == repeated_counts.tolist() == [len(raw)]
    assert int(np.sum(raw_counts)) == len(raw)

    decoder = GpuArgmaxDecoder(bundle.plan, cuda_device_id=0, dcores=bundle.dcores)
    decoded = decoder.decode(raw)
    zero_decoded = decoder.decode(zero_raw)
    assert len(decoded) == len(raw)
    assert len(zero_decoded) == 0

    geometry = GroupingGeometry.from_plan(bundle.plan)
    config = GroupingConfig(dm_tolerance_bins=1.5, time_padding_bins=1.0)
    grouped = group_candidates(decoded, geometry, config=config)
    zero_grouped = group_candidates(zero_decoded, geometry, config=config)
    cp.cuda.get_current_stream().synchronize()
    assert len(grouped.candidates) == len(raw)
    assert len(grouped.members) == len(raw)
    assert 0 < len(grouped.events) <= len(raw)
    assert len(zero_grouped.events) == 0
    assert len(zero_grouped.members) == 0
