"""Focused regression tests for the final peak-finder benchmark suite."""

import tempfile
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import cupy as cp
import numpy as np

from . import experiment_common as common
from .experiment_common import (
    DEFAULT_SEPARATION_BINS,
    RNG_NOTE,
    SCHEMA_VERSION,
    WIDTH_SEMANTICS,
    atomic_write_csv,
    atomic_write_yaml,
    candidate_values,
    clean_completed_units,
    fine_toa_tolerance_s,
    full_band_temporal_radius_bins,
    inband_match_frequency_MHz,
    initialize_checkpoint,
    make_unit_rng,
    match_one_inband,
    match_two_inband,
    parse_separation_grid,
    prepare_plan,
    read_csv,
    realized_channel_coverage,
    require_metadata_schema,
    sample_recall_burst_parameters,
    sample_safe_toa_placement,
    sample_spectral_interval,
    select_subbands,
    serial_all_tree_gpu_times,
    toa_at_frequency,
    validate_matched_cardinality_containment,
    validate_dm_for_peakfinder,
    validate_requested_frequency_interval,
    validate_resume_metadata,
    validate_safe_toa_placement,
    validate_simulation_frequency_interval,
    validate_simulation_subband,
)
from .peakfinders import (
    BENCHMARK_METHODS,
    DECODED_DTYPES,
    METHODS,
    METHOD_LABELS,
    CandidateBatch,
    build_peakfinder_geometry,
    decode_candidates,
    empty_decoded_candidates,
    enumerate_plan_subbands,
    validate_candidate_set_containment,
)

ROOT = Path(__file__).resolve().parents[1]
CONFIG = str(ROOT / "configs/dedispersion/chord_sb2.yml")
METADATA = str(ROOT / "configs/xengine_metadata.yml")


def _expect_raises(exception, function, contains=None):
    try:
        function()
    except exception as exc:
        if contains is not None:
            assert contains in str(exc), str(exc)
        return exc
    raise AssertionError(f"expected {exception.__name__}")


def _candidate_dict(toas, dms=None, snrs=None):
    n = len(toas)
    return {
        "toa_ref_s": np.asarray(toas, dtype=np.float64),
        "dm": np.asarray([100.0] * n if dms is None else dms, dtype=np.float64),
        "snr": np.asarray(list(range(n, 0, -1)) if snrs is None else snrs,
                          dtype=np.float32),
    }


def test_method_schema_and_old_schema_rejection():
    assert METHODS == ("full_band_bowtie",)
    assert BENCHMARK_METHODS == METHODS
    assert SCHEMA_VERSION == 5
    assert common.SCHEMA_NAME == "pirate-peakfinder-full-band-benchmarks"
    assert METHOD_LABELS == {"full_band_bowtie": "Full-band bowtie"}
    with tempfile.TemporaryDirectory() as tmpdir:
        path = Path(tmpdir) / "old.csv"
        path.write_text("method,value\nadaptive,1\n")
        _expect_raises(ValueError, lambda: read_csv(path, ["schema_version", "method"]),
                       "incompatible pre-v")
        metadata = Path(tmpdir) / "metadata.yaml"
        atomic_write_yaml(metadata, {"suite": {"methods": ["adaptive"]}})
        _expect_raises(ValueError,
                       lambda: require_metadata_schema(common.read_metadata(metadata), metadata),
                       "incompatible benchmark schema")


def test_plan_subbands_and_full_band_geometry_all_trees():
    config, xmd, plan, dcores = prepare_plan(CONFIG, METADATA)
    reference = float(np.asarray(xmd.get_channel_freq_edges())[0])
    expected_counts = [5, 9, 7, 3, 1]
    for itree in range(int(plan.ntrees)):
        m_to_band, bands, full_index = enumerate_plan_subbands(plan, itree, dcores=dcores)
        assert len(bands) == int(plan.trees[itree].frequency_subbands.N)
        assert len({band.subband_id for band in bands}) == len(bands)
        assert [sum(b.level == level for b in bands) for level in range(5)] == expected_counts
        assert [b.order_within_level for b in bands if b.level == 2] == list(range(7))
        full = bands[full_index]
        assert sum(b.is_full_band for b in bands) == 1
        assert full.fmin == min(b.fmin for b in bands)
        assert full.fmax == max(b.fmax for b in bands)
        assert np.array_equal(m_to_band, np.asarray(plan.trees[itree].frequency_subbands.m_to_n))

        geometry = build_peakfinder_geometry(
            plan, itree, dcores=dcores, time_sample_s=float(config.time_sample_ms) / 1.0e3,
            nt_in=int(plan.nt_in), reference_freq_mhz=reference)
        footprint = geometry.full_band_footprint
        assert footprint.dtype == cp.bool_
        assert footprint.ndim == 2
        assert all(int(size) % 2 == 1 for size in footprint.shape)
        assert bool(footprint[tuple(int(size) // 2 for size in footprint.shape)])
        diagnostics = geometry.diagnostics()
        assert diagnostics["peakfinder"] == "full_band_bowtie"
        assert diagnostics["full_band_index"] == full_index


def test_candidate_result_mapping_requires_only_full_band_method():
    batch = CandidateBatch(
        cp.asarray([3], dtype=cp.int64),
        cp.asarray([4], dtype=cp.int64),
        cp.asarray([20.0], dtype=cp.float32),
        cp.asarray([0], dtype=cp.uint32),
    )
    assert validate_candidate_set_containment({"full_band_bowtie": batch})
    _expect_raises(
        ValueError,
        lambda: validate_candidate_set_containment({}),
        "expected methods",
    )
    _expect_raises(
        ValueError,
        lambda: validate_candidate_set_containment({"removed_method": batch}),
        "expected methods",
    )


def test_candidate_decoding_and_empty_schema():
    empty = empty_decoded_candidates()
    assert tuple(empty) == tuple(DECODED_DTYPES)
    assert all(empty[key].dtype == dtype and empty[key].shape == (0,)
               for key, dtype in DECODED_DTYPES.items())

    class FakePlan:
        ntrees = 3
        trees = [SimpleNamespace(nt_ds=800, nt_out=100)] * 3

        def decode_argmax_batch(self, tokens, itrees, idm, itime, *, dcores):
            assert np.array_equal(dcores, [8, 4, 2])
            assert tokens.dtype == np.uint32
            return (np.array([100, 200]), np.array([199, 399]),
                    np.array([1, 2]), np.array([3, 4]), np.array([7, 9]))

        def decode_argmax2_batch(self, itrees, fmin, fmax, tlo, thi, profile):
            return (np.array([400.0, 500.0]), np.array([500.0, 700.0]),
                    np.array([99.5, 100.5]), np.array([5.0, 6.0]),
                    np.array([2.0, 4.0]))

    batch = CandidateBatch(
        cp.asarray([3, 4], dtype=cp.int64),
        cp.asarray([8, 9], dtype=cp.int64),
        cp.asarray([20, 15], dtype=cp.float16),
        cp.asarray([11, 12], dtype=cp.uint32),
    )
    decoded = decode_candidates(
        FakePlan(), batch, dcores=(8, 4, 2), itree=2, time_chunk_index=3,
        ntime=1000, time_sample_s=0.001)
    assert tuple(decoded) == tuple(DECODED_DTYPES)
    assert np.array_equal(decoded["fmin"], [100, 200])
    assert np.array_equal(decoded["fmax"], [199, 399])
    assert np.array_equal(decoded["profile"], [7, 9])
    assert np.array_equal(decoded["freq_lo_MHz"], [400, 500])
    assert np.array_equal(decoded["freq_hi_MHz"], [500, 700])
    assert np.array_equal(decoded["argmax_token"], [11, 12])
    assert np.allclose(decoded["toa_ref_s"], [3.005, 3.006])


def test_separation_grid_and_continuous_offsets():
    points = parse_separation_grid(
        DEFAULT_SEPARATION_BINS, time_sample_s=0.00131072,
        ntime=2048, nt_out=128)
    assert len(points) == 21
    assert [point.native_samples for point in points] == [
        8, 12, 16, 20, 24, 28, 32, 36, 40, 44, 48,
        52, 56, 60, 64, 80, 96, 128, 160, 192, 256,
    ]
    expected_ms = [
        10.48576, 15.72864, 20.97152, 26.21440, 31.45728,
        36.70016, 41.94304, 47.18592, 52.42880, 57.67168,
        62.91456, 68.15744, 73.40032, 78.64320, 83.88608,
        104.85760, 125.82912, 167.77216, 209.71520,
        251.65824, 335.54432,
    ]
    assert np.allclose([point.milliseconds for point in points], expected_ms,
                       rtol=0, atol=1.0e-10)
    fractional = parse_separation_grid(
        ["0.1"], time_sample_s=0.001, ntime=10, nt_out=3)[0]
    assert fractional.native_samples_decimal.startswith("0.333333")
    assert not fractional.native_samples.is_integer()
    _expect_raises(
        ValueError,
        lambda: parse_separation_grid(
            ["1.0", "1.00"], time_sample_s=0.001, ntime=16, nt_out=1),
        "duplicate physical separation",
    )


def test_channel_coverage_and_subband_selection():
    class FakeXmd:
        def get_channel_freq_edges(self):
            return np.array([300.0, 310.0, 320.0, 340.0])

    coverage = realized_channel_coverage(FakeXmd(), 305.0, 320.0)
    assert coverage["active_channel_indices_compact"] == "0:2"
    assert coverage["active_channel_count"] == 2
    assert coverage["effective_freq_lo_MHz"] == 300.0
    assert coverage["effective_freq_hi_MHz"] == 320.0
    assert not coverage["lower_boundary_on_channel_edge"]
    assert coverage["upper_boundary_on_channel_edge"]
    _expect_raises(ValueError,
                   lambda: realized_channel_coverage(FakeXmd(), 100.0, 200.0),
                   "activates no channels")

    _, _, plan, dcores = prepare_plan(CONFIG, METADATA)
    _, bands, _ = enumerate_plan_subbands(plan, 0, dcores=dcores)
    selected = select_subbands(bands, [bands[-1].subband_id, bands[0].subband_id])
    assert selected == (bands[0], bands[-1])
    _expect_raises(ValueError,
                   lambda: select_subbands(bands, [bands[0].subband_id] * 2),
                   "duplicates")
    _expect_raises(ValueError, lambda: select_subbands(bands, ["missing"]),
                   "available sub-bands")


def test_deterministic_parameter_sampling_and_dm_boundaries():
    rng1, unit_seed1 = make_unit_rng(12345, "recall", 7)
    rng2, unit_seed2 = make_unit_rng(12345, "recall", 7)
    _, different_seed = make_unit_rng(12345, "recall", 8)
    assert unit_seed1 == unit_seed2
    assert unit_seed1 != different_seed

    _, xmd, plan, dcores = prepare_plan(CONFIG, METADATA)
    sampled1 = sample_recall_burst_parameters(rng1, plan, xmd, dm_reach=8)
    sampled2 = sample_recall_burst_parameters(rng2, plan, xmd, dm_reach=8)
    assert sampled1 == sampled2
    assert 20.0 <= sampled1["dm"] < 150.0
    assert common.DEFAULT_WIDTH_RANGE_MS[0] <= sampled1["width_ms"] < (
        common.DEFAULT_WIDTH_RANGE_MS[1]
    )
    assert common.DEFAULT_SNR_RANGE[0] <= sampled1["snr"] < (
        common.DEFAULT_SNR_RANGE[1]
    )
    assert common.DEFAULT_BANDWIDTH_MIN_MHZ <= (
        sampled1["requested_bandwidth_MHz"]
    )
    assert (
        sampled1["full_band_freq_lo_MHz"]
        <= sampled1["requested_freq_lo_MHz"]
        < sampled1["requested_freq_hi_MHz"]
        <= sampled1["full_band_freq_hi_MHz"]
    )
    assert sampled1["active_channel_count"] == len(
        sampled1["active_channel_indices"])
    assert sampled1["width_semantics"] == WIDTH_SEMANTICS
    _expect_raises(
        ValueError,
        lambda: validate_dm_for_peakfinder(plan, 0.0, dm_reach=8),
        "boundary clearance",
    )


def test_arbitrary_interval_validation_and_sampling():
    class FakeXmd:
        def get_channel_freq_edges(self):
            return np.array([300.0, 310.0, 320.0, 340.0])

    info = validate_requested_frequency_interval(
        FakeXmd(), 305.25, 327.75, minimum_bandwidth_MHz=10.0)
    assert info["requested_bandwidth_MHz"] == 22.5
    assert info["active_channel_indices"] == [0, 1, 2]
    assert info["active_channel_indices_compact"] == "0:3"
    assert info["effective_freq_lo_MHz"] == 300.0
    assert info["effective_freq_hi_MHz"] == 340.0
    _expect_raises(
        ValueError,
        lambda: validate_requested_frequency_interval(
            FakeXmd(), 299.0, 330.0, minimum_bandwidth_MHz=10.0),
        "outside full observing band",
    )

    sampled = sample_spectral_interval(
        np.random.default_rng(17), FakeXmd(), minimum_bandwidth_MHz=10.0)
    assert 10.0 <= sampled["requested_bandwidth_MHz"] <= 40.0
    assert 300.0 <= sampled["requested_freq_lo_MHz"]
    assert sampled["requested_freq_hi_MHz"] <= 340.0


def test_safe_toa_placement_worst_case_and_pair():
    config, xmd, plan, dcores = prepare_plan(CONFIG, METADATA)
    time_sample_s = float(config.time_sample_ms) / 1.0e3
    tree = plan.trees[0]
    geometry = SimpleNamespace(
        nt=int(tree.nt_out),
        time_step_s=time_sample_s * int(plan.nt_in) / int(tree.nt_out),
        full_band_footprint=cp.ones((17, 19), dtype=cp.bool_),
    )
    assert full_band_temporal_radius_bins(geometry) == 9
    full_edges = np.asarray(xmd.get_channel_freq_edges(), dtype=np.float64)

    placement = sample_safe_toa_placement(
        np.random.default_rng(5),
        metadata_path=METADATA,
        nchunks=4,
        ntime=2048,
        dm=150.0,
        snr=20.0,
        width_ms=64.0,
        freq_lo_MHz=float(full_edges[0]),
        freq_hi_MHz=float(full_edges[-1]),
        geometry=geometry,
    )
    assert placement["output_map_index"] == 3  # 0.9984 ms cadence: wide pulse fits the fourth map
    assert 9 <= placement["output_bin_indices"][0] < int(tree.nt_out) - 9
    assert 0.0 <= placement["first_toa_output_bin_phase"] < 1.0
    assert 0 <= placement["pulse_supports"][0][0]
    assert placement["pulse_supports"][0][1] <= 4 * 2048

    pair = sample_safe_toa_placement(
        np.random.default_rng(6),
        metadata_path=METADATA,
        nchunks=4,
        ntime=2048,
        dm=100.0,
        snr=[20.0, 20.0],
        width_ms=[1.0, 1.0],
        freq_lo_MHz=500.25,
        freq_hi_MHz=1100.75,
        geometry=geometry,
        burst_offsets_s=(0.0, 0.020),
    )
    assert len(pair["toas_s"]) == 2
    assert np.isclose(pair["toas_s"][1] - pair["toas_s"][0], 0.020)
    assert pair["all_bursts_in_same_output_map"]
    assert all(9 <= value < int(tree.nt_out) - 9
               for value in pair["output_bin_indices"])

    simulation = common.create_simulated_acquisition(
        METADATA,
        nchunks=4,
        ntime=2048,
        dm=100.0,
        snr=[20.0, 20.0],
        width_ms=[1.0, 1.0],
        toa=pair["toas_s"],
        subband_lo_MHz=500.25,
        subband_hi_MHz=1100.75,
    )
    validated = validate_safe_toa_placement(
        simulation,
        geometry=geometry,
        output_map_index=pair["output_map_index"],
        expected_offsets_s=(0.0, 0.020),
    )
    assert validated["output_bin_indices"] == pair["output_bin_indices"]


def test_simulation_interval_propagation_and_support_semantics():
    band = SimpleNamespace(
        subband_id="f0_1", freq_lo_MHz=305.0, freq_hi_MHz=320.0)
    coverage = {
        "active_channel_indices": np.array([0, 1], dtype=np.int64),
        "active_channel_indices_compact": "0:2",
        "active_channel_count": 2,
    }
    pulses = [
        SimpleNamespace(subband_freq_lo_MHz=305.0, subband_freq_hi_MHz=320.0,
                        freq_nt=np.array([2, 3, 0])),
        SimpleNamespace(subband_freq_lo_MHz=305.0, subband_freq_hi_MHz=320.0,
                        freq_nt=np.array([4, 5, 0])),
    ]
    simulation = SimpleNamespace(
        pulses=pulses,
        burst_summaries=[
            {"requested_toa": 1.0, "sample_start": 90, "sample_end": 111},
            {"requested_toa": 1.1, "sample_start": 98, "sample_end": 120},
        ],
        time_sample_ms=1.0,
    )
    result = validate_simulation_subband(simulation, band, coverage)
    assert result["toa_native_samples"] == [1000.0, 1100.0]
    assert result["pulse_supports"] == [[90, 111], [98, 120]]
    assert np.allclose(result["stored_toa_offsets_s"], [0.0, 0.1])
    assert result["toa_native_samples"][0] != result["pulse_supports"][0][0]

    single = SimpleNamespace(
        pulses=pulses[:1],
        burst_summaries=simulation.burst_summaries[:1],
        time_sample_ms=1.0,
    )
    one = validate_simulation_frequency_interval(
        single, 305.0, 320.0, coverage, expected_bursts=1
    )
    assert one["requested_toas_s"] == [1.0]
    assert one["active_channels_verified_for_all_bursts"]

    inconsistent = SimpleNamespace(
        pulses=[pulses[0], SimpleNamespace(
            subband_freq_lo_MHz=305.0, subband_freq_hi_MHz=320.0,
            freq_nt=np.array([4, 0, 0]))],
        burst_summaries=simulation.burst_summaries,
        time_sample_ms=1.0,
    )
    _expect_raises(
        ValueError,
        lambda: validate_simulation_subband(inconsistent, band, coverage),
        "active channels disagree",
    )

    captured = {}

    class StopAfterSimulation(Exception):
        pass

    def fake_simulation(**kwargs):
        captured.update(kwargs)
        raise StopAfterSimulation

    with patch.object(common, "make_validated_simulation", side_effect=fake_simulation):
        _expect_raises(StopAfterSimulation, lambda: common.run_simulated_peakfinders(
            metadata_path=METADATA, config_path=CONFIG, nchunks=1, ntime=2048,
            dm=100, snr=30, width_ms=2, toas=[1],
            threshold=10, device=0, dm_reach=8, waist_bins=1,
            freq_lo_MHz=305.0, freq_hi_MHz=320.0))
    assert captured["subband_lo_MHz"] == 305.0
    assert captured["subband_hi_MHz"] == 320.0


def test_inband_transform_and_maximum_cardinality_matching():
    nu_match = inband_match_frequency_MHz(400.0, 600.0)
    assert 400.0 < nu_match < 600.0
    transformed = toa_at_frequency([5.0], [100.0], 300.0, nu_match)[0]
    expected = (5.0 + common.dispersion_delay(100.0, nu_match)
                - common.dispersion_delay(100.0, 300.0))
    assert np.isclose(transformed, expected)

    # Recovered DM is deliberately far away: no DM veto is part of association.
    recall_candidates = _candidate_dict(
        [0.99, 1.01, 1.20], dms=[100, 100, 100], snrs=[10, 20, 50])
    recall_index, recall_injected, recall_recovered = match_one_inband(
        recall_candidates,
        injected_dm=100,
        injected_toa_ref_s=1.0,
        reference_freq_MHz=300,
        match_freq_MHz=300,
        toa_tolerance_s=0.02,
    )
    assert recall_index == 1  # equal residual: higher S/N wins
    assert recall_injected == 1.0
    assert np.array_equal(recall_recovered, recall_candidates["toa_ref_s"])

    candidates = _candidate_dict([1.055, 0.95], dms=[5000.0, 0.0], snrs=[20, 19])
    assignments, injected, recovered = match_two_inband(
        candidates, injected_dm=100.0, injected_toas_ref_s=(1.0, 1.1),
        reference_freq_MHz=300.0, match_freq_MHz=300.0,
        toa_tolerance_s=0.06)
    assert assignments == [1, 0]
    assert np.array_equal(injected, [1.0, 1.1])
    assert np.array_equal(recovered, candidates["toa_ref_s"])

    shared = _candidate_dict([1.05], snrs=[20])
    one, _, _ = match_two_inband(
        shared, injected_dm=100, injected_toas_ref_s=(1.0, 1.1),
        reference_freq_MHz=300, match_freq_MHz=300, toa_tolerance_s=0.06)
    assert sum(index is not None for index in one) == 1
    expanded = _candidate_dict([1.05, 1.0], snrs=[20, 10])
    two, _, _ = match_two_inband(
        expanded, injected_dm=100, injected_toas_ref_s=(1.0, 1.1),
        reference_freq_MHz=300, match_freq_MHz=300, toa_tolerance_s=0.06)
    assert sum(index is not None for index in two) == 2
    counts = validate_matched_cardinality_containment(
        {"full_band_bowtie": two},
        context="test",
    )
    assert counts == {"full_band_bowtie": 2}
    _expect_raises(
        ValueError,
        lambda: validate_matched_cardinality_containment(
            {"removed_method": one},
            context="test",
        ),
        "do not match",
    )

    tolerance, _ = fine_toa_tolerance_s(0.00131072, 0.002)
    min_sep = parse_separation_grid(
        DEFAULT_SEPARATION_BINS, time_sample_s=0.00131072,
        ntime=2048, nt_out=128)[0].milliseconds / 1000
    assert tolerance < 0.5 * min_sep


def test_candidate_values_exact_subband_quality_control():
    decoded = empty_decoded_candidates()
    values = {
        "idm": [3], "itime": [4], "time_chunk_index": [2], "tree": [0],
        "snr": [20], "argmax_token": [123], "dm": [101], "toa_ref_s": [5],
        "width_s": [0.004], "profile": [7], "fmin": [0], "fmax": [4095],
        "freq_lo_MHz": [900], "freq_hi_MHz": [1500],
    }
    for key, value in values.items():
        decoded[key] = np.asarray(value, dtype=DECODED_DTYPES[key])
    result = candidate_values(
        decoded, 0, candidate_toas_match=np.array([4.5]),
        injected_dm=100, injected_toa_match_s=4.49, dm_step=0.5,
        injected_fmin=0, injected_fmax=4095)
    assert result["profile"] == 7 and result["argmax_token"] == 123
    assert result["subband_matches_injected"] is True
    assert result["dm_residual"] == 1 and result["dm_residual_bins"] == 2
    assert np.isclose(result["toa_residual_ms"], 10)
    arbitrary = candidate_values(
        decoded, 0, candidate_toas_match=np.array([4.5]),
        injected_dm=100, injected_toa_match_s=4.49, dm_step=0.5)
    assert arbitrary["subband_matches_injected"] == ""
    _expect_raises(
        ValueError,
        lambda: candidate_values(
            decoded, 0, candidate_toas_match=np.array([4.5]),
            injected_dm=100, injected_toa_match_s=4.49, dm_step=0.5,
            injected_fmin=0),
        "both be supplied or omitted",
    )


def test_composite_checkpoint_and_resume_schema():
    def unit_key(row):
        return (row["separation_ms"], int(row["trial"]))

    rows = [
        {
            "separation_ms": "5",
            "trial": 0,
            "method": method,
            "parameter_seed": 123,
            "unit_seed": 456,
        }
        for method in BENCHMARK_METHODS
    ]
    rows.append({
        "separation_ms": "6",
        "trial": 0,
        "method": BENCHMARK_METHODS[0],
        "parameter_seed": 123,
        "unit_seed": 789,
    })
    # With one retained method, an ordinary row is already a complete unit.
    # A duplicate is the malformed/incomplete checkpoint case under test.
    rows.append(dict(rows[-1]))
    retained, complete = clean_completed_units(
        rows, unit_key, lambda row: row["method"], BENCHMARK_METHODS)
    assert complete == {("5", 0)}
    assert len(retained) == len(BENCHMARK_METHODS)

    with tempfile.TemporaryDirectory() as tmpdir:
        directory = Path(tmpdir)
        csv_path = directory / "checkpoint.csv"
        fields = [
            "schema_version", "separation_ms", "trial", "method",
            "parameter_seed", "unit_seed",
        ]
        csv_rows = [dict(row, schema_version=SCHEMA_VERSION) for row in rows]
        atomic_write_csv(csv_path, fields, csv_rows)
        kept, done = initialize_checkpoint(
            csv_path, fields, resume=True, overwrite=False,
            unit_key=unit_key, row_key=lambda row: row["method"],
            expected_row_keys=BENCHMARK_METHODS)
        assert done == {("5", 0)}
        assert len(kept) == len(BENCHMARK_METHODS)
        _expect_raises(
            FileExistsError,
            lambda: initialize_checkpoint(
                csv_path, fields, resume=False, overwrite=False,
                unit_key=unit_key, row_key=lambda row: row["method"],
                expected_row_keys=BENCHMARK_METHODS),
            "exists",
        )

        params = {
            "schema_version": SCHEMA_VERSION,
            "separations_ms": [5],
            "parameter_seed": 123,
        }
        metadata_path = directory / "metadata.yaml"
        atomic_write_yaml(metadata_path, {
            "suite": {
                "schema_name": common.SCHEMA_NAME,
                "schema_version": SCHEMA_VERSION,
                "methods": list(BENCHMARK_METHODS),
            },
            "separability": {"scientific_parameters": params, "intended_trials": 1},
        })
        assert validate_resume_metadata(metadata_path, "separability", params)
        _expect_raises(
            ValueError,
            lambda: validate_resume_metadata(
                metadata_path, "separability", dict(params, parameter_seed=124)),
            "scientific parameters differ",
        )


def test_rng_note_and_runtime_aggregation():
    assert "not simulation noise" in RNG_NOTE
    assert "on first use" in RNG_NOTE
    assert "same maps" not in RNG_NOTE  # Wording says identical maps within a logical unit.
    assert "identical maps within a logical unit" in RNG_NOTE
    rows = []
    for method in BENCHMARK_METHODS:
        for iteration in (0, 1):
            for tree in range(4):
                rows.append({
                    "method": method, "iteration": iteration, "tree": tree,
                    "scale_factor": 1,
                    "gpu_time_ms": float(tree + iteration + 1),
                })
    serial = serial_all_tree_gpu_times(rows, [0, 1, 2, 3])
    assert len(serial) == len(BENCHMARK_METHODS) * 2
    assert serial[0]["gpu_time_ms"] == 10.0
    assert serial[1]["gpu_time_ms"] == 14.0
    partial = serial_all_tree_gpu_times(rows[:-1], [0, 1, 2, 3])
    assert partial == [serial[0]]
    assert serial_all_tree_gpu_times(
        [row for row in rows if row["tree"] != 3], [0, 1, 2, 3]
    ) == []
    bad = list(rows)
    bad[0] = dict(bad[0], gpu_time_ms=-1)
    _expect_raises(ValueError, lambda: serial_all_tree_gpu_times(bad, [0, 1, 2, 3]),
                   "invalid gpu_time_ms")


def main():
    tests = [value for name, value in sorted(globals().items())
             if name.startswith("test_") and callable(value)]
    for test in tests:
        test()
        print(f"{test.__name__}: pass")
    print(f"test_experiment_common: {len(tests)} tests passed")


if __name__ == "__main__":
    main()
