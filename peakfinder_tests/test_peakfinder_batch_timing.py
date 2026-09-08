"""CPU-only tests for the clean multi-beam peakfinder timing benchmark."""

from __future__ import annotations

import csv
import inspect
import json
import math
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest
import yaml

from . import benchmark_peakfinder_batch_timing as benchmark
from .benchmark_peakfinder_batch_timing import (
    ARGMAX_DTYPE,
    DEFAULT_BASE_SEED,
    DEFAULT_BEAM_BATCH_SIZES,
    DEFAULT_DEVICE,
    DEFAULT_DM_REACHES,
    DEFAULT_ITERATIONS,
    DEFAULT_THRESHOLD,
    DEFAULT_TOTAL_BEAMS,
    DEFAULT_WAIST_BINS,
    DEFAULT_WARMUP,
    EXPECTED_PIXELS_PER_BEAM,
    EXPECTED_SHAPES,
    SCHEMA_NAME,
    SCHEMA_VERSION,
    SNR_DTYPE,
    SUMMARY_FIELDS,
    SUMMARY_METRICS,
    SUMMARY_STATISTICS,
    TIMING_FIELDS,
    TREE_DIAGNOSTIC_FIELDS,
    GeometryDiagnostic,
    assert_filled_streaming_halo,
    assert_expected_plan,
    build_tree_diagnostic_rows,
    campaign_signature,
    derive_timing_metrics,
    expected_tree_specs,
    generate_clean_inputs,
    halo_fill_calls,
    partition_beams,
    prepare_result_state,
    required_priming_calls,
    result_paths,
    summarize_timings,
    synchronized_wall_time,
    tree_seed,
    validate_argmax_tokens,
    validate_resumed_timing_rows,
    validate_tokens_with_plan,
    validate_timing_prefix,
    write_result_bundle,
)


_PRIMARY = (0, 1, 1, 2, 2, 2, 3, 3, 3, 3)
_EARLY = (0, 1, 0, 2, 1, 0, 3, 2, 1, 0)
_MULTIPLETS = (91, 86, 91, 34, 86, 91, 20, 34, 86, 91)
_PROFILES = (16, 13, 13, 10, 10, 10, 7, 7, 7, 7)
_DOUT = (16, 16, 16, 8, 16, 16, 8, 8, 16, 16)


def test_cli_uses_the_only_production_peakfinder_implicitly():
    parser = benchmark.build_parser()
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
            frequency_subbands=SimpleNamespace(M=multiplets),
        ))
    return SimpleNamespace(ntrees=len(trees), trees=tuple(trees), nt_in=2048)


def _specs():
    return assert_expected_plan(_fake_plan())


def test_default_campaign_and_exact_plan_geometry():
    assert DEFAULT_TOTAL_BEAMS == 60
    assert DEFAULT_BEAM_BATCH_SIZES == (1, 2, 3, 4, 5, 6, 10, 12, 15, 20, 30, 60)
    assert DEFAULT_DM_REACHES == (1, 2, 4, 8, 16, 32)
    assert DEFAULT_THRESHOLD == 10.0
    assert DEFAULT_WAIST_BINS == 1
    assert DEFAULT_WARMUP == 1
    assert DEFAULT_ITERATIONS == 10
    assert DEFAULT_DEVICE == 0
    assert isinstance(DEFAULT_BASE_SEED, int) and DEFAULT_BASE_SEED >= 0
    shape_only = expected_tree_specs()
    assert tuple(spec.shape for spec in shape_only) == EXPECTED_SHAPES
    assert sum(spec.pixels_per_beam for spec in shape_only) == EXPECTED_PIXELS_PER_BEAM

    specs = expected_tree_specs(_fake_plan())
    assert len(specs) == 10
    assert tuple(spec.shape for spec in specs) == EXPECTED_SHAPES
    assert sum(spec.pixels_per_beam for spec in specs) == 983_040
    assert tuple(spec.primary_tree_index for spec in specs) == _PRIMARY
    assert tuple(spec.early_trigger_level for spec in specs) == _EARLY
    assert tuple(spec.multiplets for spec in specs) == _MULTIPLETS
    assert tuple(spec.profiles for spec in specs) == _PROFILES
    assert tuple(spec.token_dout for spec in specs) == _DOUT

    bad = _fake_plan()
    bad.ntrees = 9
    with pytest.raises(ValueError, match="exactly 10"):
        assert_expected_plan(bad)

    bad = _fake_plan()
    bad.trees[0].ndm_out = 4095
    with pytest.raises(ValueError, match="unexpected plan tree shapes"):
        assert_expected_plan(bad)


def test_exact_beam_partitions_and_nondivisor_final_batch():
    batches = partition_beams(60, 4)
    assert len(batches) == 15
    assert all(batch.size == 4 for batch in batches)
    assert batches[0].beam_ids == (0, 1, 2, 3)
    assert batches[-1].beam_ids == (56, 57, 58, 59)

    batches = partition_beams(60, 7)
    assert len(batches) == 9
    assert tuple(batch.size for batch in batches) == (7,) * 8 + (4,)
    assert batches[-1].start == 56
    assert batches[-1].stop == 60
    assert tuple(beam for batch in batches for beam in batch.beam_ids) == tuple(range(60))

    oversized = partition_beams(4, 20)
    assert len(oversized) == 1
    assert oversized[0].beam_ids == (0, 1, 2, 3)
    assert oversized[0].size == 4

    for args in ((0, 1), (1, 0), (-1, 1), (1, -1), (True, 1), (1, False), (3.5, 1)):
        with pytest.raises(ValueError):
            partition_beams(*args)


def test_clean_inputs_are_reproducible_independent_and_below_threshold():
    specs = _specs()
    first = generate_clean_inputs(specs, total_beams=2, base_seed=12345)
    repeated = generate_clean_inputs(specs, total_beams=2, base_seed=12345)
    changed = generate_clean_inputs(specs, total_beams=2, base_seed=12346)

    assert len(first) == len(repeated) == len(changed) == 10
    assert len({item.noise_seed for item in first}) == 10
    assert len({tree_seed(12345, index, "snr") for index in range(10)}) == 10
    assert tree_seed(12345, 0, "snr") != tree_seed(12345, 0, "token")

    for spec, item, duplicate, other in zip(specs, first, repeated, changed):
        assert item.spec == spec
        assert item.snr.shape == (2, spec.ndm, spec.ntime)
        assert item.snr.dtype == SNR_DTYPE
        assert np.all(np.isfinite(item.snr))
        assert np.all(item.snr < DEFAULT_THRESHOLD)
        assert not np.any(item.snr >= DEFAULT_THRESHOLD)
        assert np.array_equal(item.snr, duplicate.snr)
        assert np.array_equal(item.argmax, duplicate.argmax)
        assert not np.array_equal(item.snr, other.snr)

    # Equal-shaped trees use distinct derived streams, not slices of one shared
    # RNG stream. Comparing complete arrays avoids a flaky single-sample check.
    assert first[1].snr.shape == first[3].snr.shape
    assert not np.array_equal(first[1].snr, first[3].snr)
    with pytest.raises(ValueError, match="not above"):
        generate_clean_inputs(specs, total_beams=1, base_seed=12345, threshold=-100.0)


def test_constant_argmax_tokens_follow_current_tree_abi():
    specs = _specs()
    inputs = generate_clean_inputs(specs, total_beams=1, base_seed=6789)
    maps = tuple(item.argmax for item in inputs)
    validate_argmax_tokens(specs, maps)

    for spec, item in zip(specs, inputs):
        assert item.argmax.dtype == ARGMAX_DTYPE
        assert item.argmax.shape == (1, spec.ndm, spec.ntime)
        assert np.unique(item.argmax).tolist() == [item.token]
        token = np.uint32(item.token)
        assert int(token & np.uint32(0xFF)) == 0
        assert int((token >> np.uint32(8)) & np.uint32(0xFF)) == 0
        multiplet = int(token >> np.uint32(16))
        assert multiplet == item.token_multiplet
        assert 0 <= multiplet < spec.multiplets
        assert token != np.uint32(0xFFFFFFFF)

    bad = [array.copy() for array in maps]
    bad[0][0, 0, 0] = np.uint32(0xFFFFFFFF)
    with pytest.raises(ValueError, match="invalid sentinel"):
        validate_argmax_tokens(specs, bad)

    bad = [array.copy() for array in maps]
    bad[1][0, 0, 0] = np.uint32(specs[1].multiplets << 16)
    with pytest.raises(ValueError, match="invalid tokens"):
        validate_argmax_tokens(specs, bad)


def test_constant_tokens_are_decoded_by_the_authoritative_plan():
    inputs = generate_clean_inputs(_specs()[:2], total_beams=1, base_seed=6789)

    class Plan:
        def __init__(self, profile=0):
            self.profile = profile
            self.calls = []

        def decode_argmax(self, token, tree, idm, itime):
            self.calls.append((token, tree, idm, itime))
            return (0, 1, -1, 1, self.profile)

    plan = Plan()
    assert validate_tokens_with_plan(plan, inputs) == (
        (0, 1, -1, 1, 0), (0, 1, -1, 1, 0)
    )
    assert plan.calls == [
        (item.token, item.spec.tree_index, 0, 0) for item in inputs
    ]
    with pytest.raises(ValueError, match="profile 0"):
        validate_tokens_with_plan(Plan(profile=1), inputs)


def test_priming_depth_fills_halo_and_moves_past_left_edge():
    # Halo fill alone is ceil(2h/n); an interior-ready next call requires
    # strictly more than 2h prior columns. The distinction matters when exact.
    assert halo_fill_calls(32, 64) == 1
    assert required_priming_calls(32, 64) == 2
    assert required_priming_calls(0, 64) == 1
    assert required_priming_calls(9, 128) == 1
    assert required_priming_calls(33, 64) == 2
    assert required_priming_calls(65, 32) == 5
    assert required_priming_calls(257, 32) == 17

    for radius, ntime in ((0, 16), (1, 16), (32, 64), (257, 32)):
        calls = required_priming_calls(radius, ntime)
        assert calls * ntime > 2 * radius
        if calls > 1:
            assert (calls - 1) * ntime <= 2 * radius

    for args in ((-1, 16), (1, 0), (True, 16), (1, False), (1.5, 16)):
        with pytest.raises(ValueError):
            required_priming_calls(*args)


def test_actual_streaming_state_validation_is_not_a_startup_flag_proxy():
    radius = 2
    extractor = SimpleNamespace(
        geometry=SimpleNamespace(time_radius=radius, ndm=3),
        beam_ids=(10, 20),
        _tail_snr=np.empty((2, 3, 2 * radius), dtype=np.float16),
        _tail_argmax=np.empty((2, 3, 2 * radius), dtype=np.uint32),
        _tail_valid=np.empty((3, 2 * radius), dtype=np.bool_),
        _tail_chunk=np.empty(2 * radius, dtype=np.int64),
        _tail_itime=np.empty(2 * radius, dtype=np.int32),
        _total_columns=12,
        _next_emit=10,
        _last_chunk=99,
        _flushed=False,
        assume_steady_state=True,
    )
    assert_filled_streaming_halo(extractor, expected_last_chunk=99)

    # assume_steady_state is still true, but an undersized tail is correctly
    # rejected: startup provenance and streaming halo state are independent.
    extractor._tail_snr = np.empty((2, 3, 3), dtype=np.float16)
    with pytest.raises(AssertionError, match="S/N halo shape"):
        assert_filled_streaming_halo(extractor, expected_last_chunk=99)


def test_tree_diagnostics_cover_every_tree_reach_and_final_batch_shape():
    specs = _specs()
    reaches = (1, 8)
    diagnostics = {}
    for reach in reaches:
        for spec in specs:
            radius = reach + spec.tree_index
            diagnostics[(reach, spec.tree_index)] = GeometryDiagnostic(
                tree_index=spec.tree_index,
                dm_reach=reach,
                dm_radius=reach,
                active_footprint_cells=2 * radius + 1,
                left_time_radius=radius,
                right_time_radius=radius,
                time_radius=radius,
                halo_columns=2 * radius,
            )
    rows = build_tree_diagnostic_rows(
        "test-campaign", specs, diagnostics, reaches, (7,), total_beams=60
    )
    assert len(rows) == len(specs) * len(reaches)
    assert {
        (int(row["dm_reach"]), int(row["tree_index"])) for row in rows
    } == set(diagnostics)
    assert all(int(row["n_batches"]) == 9 for row in rows)
    assert all(int(row["final_batch_size"]) == 4 for row in rows)
    for row in rows:
        spec = specs[int(row["tree_index"])]
        radius = int(row["time_radius"])
        assert json.loads(row["regular_work_shape"]) == [
            7, spec.ndm, spec.ntime + 2 * radius
        ]
        assert json.loads(row["final_work_shape"]) == [
            4, spec.ndm, spec.ntime + 2 * radius
        ]
        assert int(row["required_priming_calls"]) == required_priming_calls(
            radius, spec.ntime
        )


def test_authoritative_workload_has_no_flush_and_wall_timer_synchronizes(monkeypatch):
    workload_source = inspect.getsource(benchmark.process_complete_workload)
    assert ".process_chunk(" in workload_source
    assert "concatenate_raw_candidates" in workload_source
    assert ".flush(" not in workload_source

    order = []

    class FakeStream:
        def synchronize(self):
            order.append("synchronize")

    ticks = iter((20.0, 20.0125))
    monkeypatch.setattr(benchmark.time, "perf_counter", lambda: next(ticks))

    def operation():
        order.append("operation")
        return "result"

    elapsed_ms, result = synchronized_wall_time(FakeStream(), operation)
    assert order == ["synchronize", "operation", "synchronize"]
    assert elapsed_ms == pytest.approx(12.5)
    assert result == "result"


def test_derived_timing_formulas_and_validation():
    metrics = derive_timing_metrics(
        wall_ms=120.0,
        total_beams=60,
        n_batches=6,
        chunk_duration_ms=2048.0,
    )
    assert metrics == pytest.approx({
        "peakfinding_ms_per_beam": 2.0,
        "equivalent_ms_per_batch": 20.0,
        "beam_chunks_per_second": 500.0,
        "realtime_beam_capacity": 1024.0,
        "load_fraction_for_total_beams": 120.0 / 2048.0,
    })
    for wall in (0.0, -1.0, math.nan, math.inf):
        with pytest.raises(ValueError, match="wall_ms"):
            derive_timing_metrics(wall, 60, 6, 2048.0)
    with pytest.raises(ValueError, match="total_beams"):
        derive_timing_metrics(1.0, 0, 1, 2048.0)
    with pytest.raises(ValueError, match="n_batches"):
        derive_timing_metrics(1.0, 1, 0, 2048.0)
    with pytest.raises(ValueError, match="chunk_duration_ms"):
        derive_timing_metrics(1.0, 1, 1, math.nan)


# Checkpoint, aggregation, schema, and round-trip tests follow below. Keeping
# their row construction centralized makes changes to the explicit CSV schema
# fail in one readable place.
def _timing_row(dm_reach=1, batch_size=2, iteration=0, wall_ms=12.0,
                campaign_id="test-campaign"):
    batches = partition_beams(60, batch_size)
    metrics = derive_timing_metrics(wall_ms, 60, len(batches), 2048.0)
    row = {
        "schema_version": SCHEMA_VERSION,
        "campaign_id": campaign_id,
        "iteration": iteration,
        "base_seed": 12345,
        "device": 0,
        "method": "full_band",
        "total_beams": 60,
        "beam_batch_size": batch_size,
        "n_batches": len(batches),
        "actual_batch_sizes": json.dumps([batch.size for batch in batches]),
        "final_batch_size": batches[-1].size,
        "threshold": 10.0,
        "dm_reach": dm_reach,
        "waist_bins": 1,
        "warmup": 1,
        "total_pixels_per_beam": EXPECTED_PIXELS_PER_BEAM,
        "total_input_pixels": 60 * EXPECTED_PIXELS_PER_BEAM,
        "total_candidates": 0,
        "chunk_duration_ms": 2048.0,
        "total_beams_peakfinding_wall_ms": wall_ms,
        **metrics,
    }
    assert tuple(row) == TIMING_FIELDS
    return row


def _tree_row(campaign_id="test-campaign"):
    row = {
        "schema_version": SCHEMA_VERSION,
        "campaign_id": campaign_id,
        "dm_reach": 1,
        "beam_batch_size": 2,
        "n_batches": 30,
        "final_batch_size": 2,
        "tree_index": 0,
        "primary_tree_index": 0,
        "early_trigger_level": 0,
        "ndm": 4096,
        "ntime": 128,
        "pixels_per_beam": 4096 * 128,
        "dm_radius": 1,
        "active_footprint_cells": 7,
        "left_time_radius": 2,
        "right_time_radius": 2,
        "time_radius": 2,
        "halo_columns": 4,
        "required_priming_calls": 1,
        "regular_batch_size": 2,
        "regular_work_shape": json.dumps([2, 4096, 132]),
        "final_work_shape": json.dumps([2, 4096, 132]),
    }
    assert tuple(row) == TREE_DIAGNOSTIC_FIELDS
    return row


def test_summary_is_wide_and_reports_median_min_max_and_iqr():
    rows = [
        _timing_row(iteration=0, wall_ms=10.0),
        _timing_row(iteration=1, wall_ms=20.0),
        _timing_row(iteration=2, wall_ms=40.0),
    ]
    summary = summarize_timings(rows)
    assert len(summary) == 1
    result = summary[0]
    assert tuple(result) == SUMMARY_FIELDS
    assert result["completed_iterations"] == 3
    assert result["total_beams_peakfinding_wall_ms_median"] == 20.0
    assert result["total_beams_peakfinding_wall_ms_minimum"] == 10.0
    assert result["total_beams_peakfinding_wall_ms_maximum"] == 40.0
    assert result["total_beams_peakfinding_wall_ms_q25"] == 15.0
    assert result["total_beams_peakfinding_wall_ms_q75"] == 30.0
    assert result["total_beams_peakfinding_wall_ms_iqr"] == 15.0

    for metric in SUMMARY_METRICS:
        values = np.asarray([float(row[metric]) for row in rows])
        assert result[f"{metric}_median"] == pytest.approx(np.median(values))
        assert result[f"{metric}_minimum"] == pytest.approx(np.min(values))
        assert result[f"{metric}_maximum"] == pytest.approx(np.max(values))
        assert result[f"{metric}_q25"] == pytest.approx(np.quantile(values, 0.25))
        assert result[f"{metric}_q75"] == pytest.approx(np.quantile(values, 0.75))
        assert result[f"{metric}_iqr"] == pytest.approx(
            np.quantile(values, 0.75) - np.quantile(values, 0.25)
        )

    assert len(SUMMARY_FIELDS) == 5 + len(SUMMARY_METRICS) * len(SUMMARY_STATISTICS)
    mixed = [dict(rows[0]), dict(rows[1])]
    mixed[1]["campaign_id"] = "different"
    with pytest.raises(ValueError, match="mixes campaign IDs"):
        summarize_timings(mixed)
    invalid = [dict(rows[0])]
    invalid[0]["peakfinding_ms_per_beam"] = math.nan
    with pytest.raises(ValueError, match="invalid peakfinding_ms_per_beam"):
        summarize_timings(invalid)


def test_campaign_signature_is_canonical_and_timing_resume_is_strict_prefix():
    payload = {
        "total_beams": 60,
        "batch_sizes": [1, 2],
        "dm_reaches": [1, 8],
        "threshold": 10.0,
    }
    signature = campaign_signature(payload)
    assert len(signature) == 64
    assert signature == campaign_signature(dict(reversed(tuple(payload.items()))))
    assert signature != campaign_signature({**payload, "threshold": 11.0})
    with pytest.raises(ValueError):
        campaign_signature({"not_finite": math.nan})

    rows = [
        _timing_row(dm_reach=1, batch_size=1, iteration=0),
        _timing_row(dm_reach=1, batch_size=1, iteration=1),
        _timing_row(dm_reach=1, batch_size=2, iteration=0),
    ]
    keys = validate_timing_prefix(
        rows, dm_reaches=(1, 8), batch_sizes=(1, 2), iterations=2,
        campaign_id="test-campaign",
    )
    assert keys == ((1, 1, 0), (1, 1, 1), (1, 2, 0))

    with pytest.raises(ValueError, match="exact ordered campaign prefix"):
        validate_timing_prefix(
            (rows[0], rows[2]), (1, 8), (1, 2), 2, "test-campaign"
        )
    with pytest.raises(ValueError, match="exact ordered campaign prefix"):
        validate_timing_prefix(
            (rows[0], rows[0]), (1, 8), (1, 2), 2, "test-campaign"
        )
    wrong_campaign = dict(rows[0], campaign_id="different")
    with pytest.raises(ValueError, match="different campaign ID"):
        validate_timing_prefix(
            (wrong_campaign,), (1, 8), (1, 2), 2, "test-campaign"
        )


def test_resume_validates_static_values_partitions_candidates_and_formulas():
    rows = [_timing_row(dm_reach=1, batch_size=7, iteration=0, wall_ms=12.0)]
    arguments = {
        "campaign_id": "test-campaign",
        "dm_reaches": (1,),
        "batch_sizes": (7,),
        "iterations": 1,
        "total_beams": 60,
        "base_seed": 12345,
        "device": 0,
        "method": "full_band",
        "threshold": 10.0,
        "waist_bins": 1,
        "warmup": 1,
        "chunk_duration_ms": 2048.0,
    }
    validate_resumed_timing_rows(rows, **arguments)

    for field, invalid, message in (
            ("actual_batch_sizes", "[7,7]", "beam partition"),
            ("total_candidates", 1, "total_candidates"),
            ("base_seed", 999, "base_seed"),
            ("peakfinding_ms_per_beam", 999.0, "peakfinding_ms_per_beam"),
            ("total_beams_peakfinding_wall_ms", math.nan,
             "authoritative wall time")):
        corrupted = dict(rows[0])
        corrupted[field] = invalid
        with pytest.raises(ValueError, match=message):
            validate_resumed_timing_rows([corrupted], **arguments)


def test_csv_metadata_round_trip_and_resume_overwrite_contract(tmp_path):
    paths = result_paths(tmp_path / "results")
    signature_payload = {
        "total_beams": 60,
        "batch_sizes": [2],
        "dm_reaches": [1],
        "iterations": 3,
    }
    signature = campaign_signature(signature_payload)
    metadata = {
        "schema_name": SCHEMA_NAME,
        "schema_version": SCHEMA_VERSION,
        "campaign_id": "test-campaign",
        "campaign_signature": signature,
        "complete": False,
        "config": {"path": "configs/dedispersion/chord_sb2_et.yml"},
        "plan": {
            "tree_shapes_ndm_ntime": [list(shape) for shape in EXPECTED_SHAPES],
            "pixels_per_beam": EXPECTED_PIXELS_PER_BEAM,
            "chunk_duration_ms": 2048.0,
        },
        "timing": {
            "authoritative": "synchronized wall clock",
            "excludes": ["flush", "decoder", "grouper"],
        },
    }
    timing_rows = [
        _timing_row(iteration=0, wall_ms=12.0),
        _timing_row(iteration=1, wall_ms=13.0),
    ]
    tree_rows = [_tree_row()]
    write_result_bundle(paths, timing_rows, tree_rows, metadata)

    for key in ("timings", "summary", "tree_diagnostics", "metadata"):
        assert paths[key].is_file()
    with paths["timings"].open(newline="", encoding="utf-8") as stream:
        reader = csv.DictReader(stream)
        loaded_timings = list(reader)
        assert tuple(reader.fieldnames or ()) == TIMING_FIELDS
    assert len(loaded_timings) == 2
    assert all(int(row["total_candidates"]) == 0 for row in loaded_timings)
    for row in loaded_timings:
        for field in ("total_beams_peakfinding_wall_ms", *SUMMARY_METRICS):
            value = float(row[field])
            assert math.isfinite(value) and value >= 0.0

    with paths["summary"].open(newline="", encoding="utf-8") as stream:
        reader = csv.DictReader(stream)
        loaded_summary = list(reader)
        assert tuple(reader.fieldnames or ()) == SUMMARY_FIELDS
    assert len(loaded_summary) == 1
    assert int(loaded_summary[0]["completed_iterations"]) == 2

    with paths["tree_diagnostics"].open(newline="", encoding="utf-8") as stream:
        reader = csv.DictReader(stream)
        loaded_trees = list(reader)
        assert tuple(reader.fieldnames or ()) == TREE_DIAGNOSTIC_FIELDS
    assert len(loaded_trees) == 1
    with paths["metadata"].open(encoding="utf-8") as stream:
        assert yaml.safe_load(stream) == metadata

    resumed, resumed_metadata = prepare_result_state(
        paths,
        resume=True,
        overwrite=False,
        signature=signature,
        campaign_id="test-campaign",
        tree_rows=tree_rows,
        dm_reaches=(1,),
        batch_sizes=(2,),
        iterations=3,
    )
    assert len(resumed) == 2
    assert resumed_metadata == metadata
    with pytest.raises(ValueError, match="mutually exclusive"):
        prepare_result_state(
            paths, resume=True, overwrite=True, signature=signature,
            campaign_id="test-campaign", tree_rows=tree_rows,
            dm_reaches=(1,), batch_sizes=(2,), iterations=3,
        )
    with pytest.raises(FileExistsError, match="use --resume or --overwrite"):
        prepare_result_state(
            paths, resume=False, overwrite=False, signature=signature,
            campaign_id="test-campaign", tree_rows=tree_rows,
            dm_reaches=(1,), batch_sizes=(2,), iterations=3,
        )
    fresh_rows, fresh_metadata = prepare_result_state(
        paths, resume=False, overwrite=True, signature=signature,
        campaign_id="test-campaign", tree_rows=tree_rows,
        dm_reaches=(1,), batch_sizes=(2,), iterations=3,
    )
    assert fresh_rows == [] and fresh_metadata is None

    malformed = dict(timing_rows[0])
    malformed["unexpected"] = 1
    with pytest.raises(ValueError, match="incompatible schema"):
        write_result_bundle(
            result_paths(tmp_path / "bad-schema"), [malformed], tree_rows, metadata
        )


def test_resume_rejects_a_nonprefix_checkpoint_bundle(tmp_path):
    paths = result_paths(tmp_path / "bad-prefix")
    payload = {"dm_reaches": [1], "batch_sizes": [2], "iterations": 3}
    signature = campaign_signature(payload)
    metadata = {
        "schema_name": SCHEMA_NAME,
        "schema_version": SCHEMA_VERSION,
        "campaign_id": "test-campaign",
        "campaign_signature": signature,
    }
    rows = [
        _timing_row(iteration=0),
        _timing_row(iteration=2),
    ]
    write_result_bundle(paths, rows, [_tree_row()], metadata)
    with pytest.raises(ValueError, match="exact ordered campaign prefix"):
        prepare_result_state(
            paths, resume=True, overwrite=False, signature=signature,
            campaign_id="test-campaign", tree_rows=[_tree_row()],
            dm_reaches=(1,), batch_sizes=(2,), iterations=3,
        )


def test_analysis_notebook_is_cleared_compilable_and_has_exact_outputs():
    path = Path(__file__).with_name("analyze_peakfinder_batch_timing.ipynb")
    notebook = json.loads(path.read_text(encoding="utf-8"))
    assert notebook["nbformat"] == 4
    code_cells = [cell for cell in notebook["cells"] if cell["cell_type"] == "code"]
    assert code_cells

    sources = {}
    for index, cell in enumerate(code_cells):
        assert cell.get("execution_count") is None
        assert cell.get("outputs") == []
        source = "".join(cell.get("source", ()))
        compile(source, f"{path.name}:cell-{index}:{cell.get('id')}", "exec")
        assert cell.get("id") not in sources
        sources[cell.get("id")] = source

    setup = sources["load-and-validate"]
    assert "# Editable result location" in setup
    assert "RESULTS_DIR =" in setup
    assert "PIRATE_PEAKFINDER_BATCH_TIMING_RESULTS" in setup
    for filename in (
            "timings.csv", "summary.csv", "tree_diagnostics.csv", "metadata.yaml"):
        assert filename in setup

    mandatory = {
        "wall-vs-dm": (
            "total_beams_peakfinding_wall_ms",
            "Total {total_beams}-beam wall time versus DM reach",
            "ms chunk budget",
        ),
        "wall-vs-batch": (
            "total_beams_peakfinding_wall_ms",
            "Total {total_beams}-beam wall time versus batch size",
            "ms chunk budget",
        ),
        "per-beam-vs-batch": (
            "peakfinding_ms_per_beam",
            "Milliseconds per beam versus batch size",
            "peakfinding time per beam (ms)",
        ),
        "throughput-vs-batch": (
            "beam_chunks_per_second",
            "Beam-chunks per second versus batch size",
            "beam-chunks per second",
        ),
        "load-heatmap": (
            "load_fraction_for_total_beams",
            "Real-time load for {total_beams} beams",
            "real-time load (%)",
        ),
        "capacity-heatmap": (
            "realtime_beam_capacity",
            "Equivalent real-time beam capacity",
            "equivalent real-time beams",
        ),
        "footprint-vs-dm": (
            "active_footprint_cells",
            "Per-tree footprint size versus DM reach",
            "active full-band footprint cells",
        ),
    }
    optional_id = "fastest-table-and-optional-diagnostics"
    show_ids = {
        cell_id for cell_id, source in sources.items() if "plt.show()" in source
    }
    assert show_ids == set(mandatory) | {optional_id}
    assert len(mandatory) == 7
    for cell_id, required_labels in mandatory.items():
        source = sources[cell_id]
        assert source.count("plt.show()") == 1
        for label in required_labels:
            assert label in source

    # Raw iteration scatter, median curves, and interquartile spread are one
    # common plotting contract used by the four readable curve figures.
    helpers = sources["plot-helpers"]
    for concept in (
            "raw_values", "ax.scatter", "medians", "q25", "q75",
            "ax.fill_between", "ax.plot"):
        assert concept in helpers

    optional = sources[optional_id]
    assert "if diagnostic_field is not None:" in optional
    assert "diagnostic CUDA-event time (ms)" in optional
    assert "not authoritative wall time" in optional
    assert "No optional per-tree GPU diagnostic timing column was recorded." in optional
    assert "Fastest median batch size for each DM reach" in optional
    for column in (
            "wall_ms", "ms_per_beam", "load_percent", "realtime_beam_capacity"):
        assert column in optional

    # Keep the scientific validation gate explicit and ahead of plotting.
    validation_concepts = (
        "EXPECTED_SHAPES",
        "EXPECTED_PIXELS_PER_BEAM",
        "983,040",
        "expected_timing_keys",
        "completed_iterations",
        "actual_batch_sizes",
        "beam_ids",
        "total_candidates",
        "np.isfinite",
        "expected_derived",
        "metadata/CSV geometry mismatch",
        "configuration grid is incomplete",
        "campaign_complete",
    )
    for concept in validation_concepts:
        assert concept in setup

    # The requested real-time references are encoded in the relevant figures.
    assert "chunk_duration_ms" in sources["wall-vs-dm"]
    assert "chunk_duration_ms" in sources["wall-vs-batch"]
    assert "levels=[100.0]" in sources["load-heatmap"]
    assert "levels=[total_beams]" in sources["capacity-heatmap"]
