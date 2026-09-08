"""Exact parity tests for the benchmark-only persistent GPU grouper.

The production representative-seeded grouper is the oracle in every scientific
test below.  Synthetic inputs use the canonical normalized GPU schema so that
failures isolate grouping work distribution, ordering, or floating-point
decisions rather than decoder setup.  Host copies are used only after both
groupers finish, for focused assertions and diagnostics outside the hot path.
"""

from __future__ import annotations

import ast
from dataclasses import replace
import inspect
import textwrap

import numpy as np
import pytest


cp = pytest.importorskip("cupy")

from pirate_frb.OfflineCandidateGrouper import (  # noqa: E402
    GpuDecodedCandidates,
    GroupingConfig,
    GroupingGeometry,
    _group_candidates_serial_oracle,
    group_candidates,
)
from pirate_frb.Peakfinders import EdgeFlag  # noqa: E402
from peakfinder_tests import gpu_representative_grouper as prototype  # noqa: E402
from peakfinder_tests.cpu_candidate_grouper import (  # noqa: E402
    assert_gpu_grouping_results_equal,
)


@pytest.fixture(scope="module", autouse=True)
def _require_cuda_device():
    """Skip cleanly when CuPy imports but no usable CUDA device is present."""

    try:
        if cp.cuda.runtime.getDeviceCount() < 1:
            pytest.skip("no CUDA device is available")
        cp.cuda.Device(0).use()
        # Force context creation here so driver/runtime failures become a skip,
        # while failures raised by the tests after ``yield`` remain failures.
        cp.zeros(1, dtype=cp.uint8).sum().item()
    except Exception as exc:
        pytest.skip(f"CUDA is unavailable: {exc}")
    with cp.cuda.Device(0):
        yield


def _geometry(
        families,
        *,
        dm_steps=None,
        time_steps=None,
        slope_lo=0.0,
        slope_hi=2.0,
        ):
    """Build canonical grouping geometry without constructing a producer plan."""

    families = np.asarray(families, dtype=np.int32)
    ntrees = int(families.size)
    if dm_steps is None:
        dm_steps = np.ones(ntrees, dtype=np.float64)
    if time_steps is None:
        time_steps = np.ones(ntrees, dtype=np.float64)
    dm_steps = np.asarray(dm_steps, dtype=np.float64)
    time_steps = np.asarray(time_steps, dtype=np.float64)
    assert dm_steps.shape == time_steps.shape == families.shape
    return GroupingGeometry(
        ntrees=ntrees,
        primary_tree_index_by_tree=cp.asarray(families, dtype=cp.int32),
        dm_step_by_tree=cp.asarray(dm_steps, dtype=cp.float64),
        time_step_samples_by_tree=cp.asarray(
            time_steps, dtype=cp.float64
        ),
        reference_freq_MHz=300.0,
        full_band_freq_lo_MHz=300.0,
        full_band_freq_hi_MHz=1500.0,
        time_sample_ms=1.0,
        residual_slope_lo_samples_per_dm=float(slope_lo),
        residual_slope_hi_samples_per_dm=float(slope_hi),
    )


def _candidates(geometry, rows):
    """Upload compact row dictionaries as canonical ``GpuDecodedCandidates``."""

    rows = tuple(rows)
    n = len(rows)

    def values(name, default):
        return [row.get(name, default(i, row)) for i, row in enumerate(rows)]

    tree = cp.asarray(
        values("tree", lambda i, row: i % geometry.ntrees),
        dtype=cp.int32,
    )
    return GpuDecodedCandidates(
        beam_id=cp.asarray(
            values("beam_id", lambda i, row: 7), dtype=cp.int32
        ),
        source_chunk_index=cp.asarray(
            values("source_chunk_index", lambda i, row: 0), dtype=cp.int64
        ),
        tree=tree,
        idm=cp.asarray(
            values("idm", lambda i, row: i % 1009), dtype=cp.int32
        ),
        itime=cp.asarray(
            values("itime", lambda i, row: (3 * i) % 1013), dtype=cp.int32
        ),
        snr=cp.asarray(
            values("snr", lambda i, row: 100.0 - i), dtype=cp.float64
        ),
        argmax_token=cp.asarray(
            values("argmax_token", lambda i, row: i), dtype=cp.uint32
        ),
        edge_flags=cp.asarray(
            values("edge_flags", lambda i, row: 0), dtype=cp.uint8
        ),
        dm=cp.asarray(
            values("dm", lambda i, row: 100.0), dtype=cp.float64
        ),
        toa_sample_abs=cp.asarray(
            values("toa_sample_abs", lambda i, row: 1000.0),
            dtype=cp.float64,
        ),
        width_samp=cp.asarray(
            values("width_samp", lambda i, row: 2.0), dtype=cp.float64
        ),
        width_ms=cp.asarray(
            values("width_ms", lambda i, row: 2.0), dtype=cp.float64
        ),
        freq_lo_MHz=cp.asarray(
            values("freq_lo_MHz", lambda i, row: 300.0), dtype=cp.float64
        ),
        freq_hi_MHz=cp.asarray(
            values("freq_hi_MHz", lambda i, row: 1500.0),
            dtype=cp.float64,
        ),
        primary_tree_index=geometry.primary_tree_index_by_tree[tree],
        dm_step=geometry.dm_step_by_tree[tree],
        time_step_samples=geometry.time_step_samples_by_tree[tree],
    )


def _run_parity(rows, geometry, *, config=None):
    """Compare production and prototype with the independent serial oracle."""

    candidates = _candidates(geometry, rows)
    expected = _group_candidates_serial_oracle(
        candidates, geometry, config=config
    )
    production = group_candidates(candidates, geometry, config=config)
    assert_gpu_grouping_results_equal(cp, expected, production)
    assert production.complete is True
    assert production.timed_out is False
    assert cp.array_equal(
        production.input_candidate_index,
        cp.arange(len(candidates), dtype=cp.int64),
    ).item()
    actual = prototype.group_candidates_gpu_representative(
        candidates, geometry, config=config
    )
    assert_gpu_grouping_results_equal(cp, expected, actual)
    return expected, actual


def _host(array):
    return cp.asnumpy(array)


def _member_indices_by_event(result):
    event_id = _host(result.members.event_id)
    candidate_index = _host(result.members.candidate_index)
    return [
        candidate_index[event_id == ievent].tolist()
        for ievent in range(len(result.events))
    ]


@pytest.mark.parametrize(
    ("rows", "expected_events"),
    [
        pytest.param([], 0, id="empty"),
        pytest.param(
            [{"tree": 2, "snr": 37.0, "dm": -12.5}],
            1,
            id="singleton",
        ),
    ],
)
def test_empty_and_single_candidate_exact_tables(rows, expected_events):
    """Categories 1-2: empty and singleton outputs match every field/dtype."""

    geometry = _geometry([0, 0, 0])
    _, actual = _run_parity(rows, geometry)
    assert len(actual.events) == expected_events
    assert len(actual.members) == len(rows)
    assert actual.candidate_event_id.dtype == cp.int64
    assert actual.events.event_id.dtype == cp.int64
    assert actual.events.member_count.dtype == cp.int32
    assert actual.members.candidate_index.dtype == cp.int64
    assert actual.members.is_representative.dtype == cp.bool_
    if rows:
        assert _host(actual.candidate_event_id).tolist() == [0]
        assert _host(actual.events.representative_candidate_index).tolist() == [0]
        assert _host(actual.events.member_count).tolist() == [1]


def test_all_candidates_physically_isolated():
    """Category 3: incompatible rows each seed their own ordered event."""

    geometry = _geometry([0] * 6)
    rows = [
        {
            "tree": tree,
            "snr": 40.0 - tree,
            "dm": 100.0 + 20.0 * tree,
            "toa_sample_abs": 1000.0 + 100.0 * tree,
        }
        for tree in range(6)
    ]
    _, actual = _run_parity(rows, geometry)
    assert _host(actual.candidate_event_id).tolist() == list(range(6))
    assert _host(actual.events.representative_candidate_index).tolist() == list(
        range(6)
    )
    assert _host(actual.events.member_count).tolist() == [1] * 6


def test_one_dense_directly_compatible_event():
    """Category 4: one representative directly claims one row from each tree."""

    geometry = _geometry([3] * 8)
    rows = [
        {
            "tree": tree,
            "snr": 50.0 - tree,
            "dm": 200.0,
            "toa_sample_abs": 4000.0,
        }
        for tree in range(8)
    ]
    _, actual = _run_parity(rows, geometry)
    assert len(actual.events) == 1
    assert _host(actual.candidate_event_id).tolist() == [0] * 8
    assert _host(actual.events.representative_candidate_index).tolist() == [0]
    assert _host(actual.events.member_count).tolist() == [8]
    assert _member_indices_by_event(actual) == [list(range(8))]


def test_only_first_priority_candidate_from_each_other_tree_joins():
    """Category 5: minimum global rank, not input position, wins per tree."""

    geometry = _geometry([0, 0, 0])
    rows = [
        {"tree": 0, "snr": 30.0, "dm": 100.0},
        {"tree": 1, "snr": 24.0, "idm": 9, "dm": 100.1},
        {"tree": 1, "snr": 25.0, "idm": 8, "dm": 100.2},
        {"tree": 2, "snr": 23.0, "dm": 99.9},
    ]
    _, actual = _run_parity(rows, geometry)
    assert _host(actual.candidate_event_id).tolist() == [0, 1, 0, 0]
    assert _host(actual.events.representative_candidate_index).tolist() == [0, 1]
    assert _host(actual.events.member_count).tolist() == [3, 1]
    assert _member_indices_by_event(actual) == [[0, 2, 3], [1]]


def test_same_tree_candidates_never_group():
    """Category 6: even identical physical rows remain separate in one tree."""

    geometry = _geometry([0])
    rows = [
        {
            "tree": 0,
            "snr": 20.0 - i,
            "dm": 42.0,
            "toa_sample_abs": 1234.5,
        }
        for i in range(7)
    ]
    _, actual = _run_parity(rows, geometry)
    assert _host(actual.candidate_event_id).tolist() == list(range(7))
    assert _host(actual.events.member_count).tolist() == [1] * 7


def test_beams_and_primary_tree_families_are_exact_partitions():
    """Categories 7-8: beam and family isolation are both non-negotiable."""

    geometry = _geometry([0, 0, 1])
    rows = [
        {"beam_id": 1, "tree": 0, "snr": 40.0},
        {"beam_id": 2, "tree": 1, "snr": 39.0},
        {"beam_id": 1, "tree": 2, "snr": 38.0},
        {"beam_id": 1, "tree": 1, "snr": 37.0},
    ]
    _, actual = _run_parity(rows, geometry)
    assert _host(actual.candidate_event_id).tolist() == [0, 1, 2, 0]
    assert _host(actual.events.representative_candidate_index).tolist() == [0, 1, 2]
    assert _host(actual.events.member_count).tolist() == [2, 1, 1]


def test_cross_chunk_grouping_uses_absolute_toa():
    """Category 9: chunk provenance does not veto matching absolute arrivals."""

    geometry = _geometry([0, 0], dm_steps=[1.0, 2.0], time_steps=[1.0, 2.0])
    rows = [
        {
            "tree": 0,
            "source_chunk_index": -8,
            "itime": 1023,
            "snr": 20.0,
            "dm": 42.0,
            "toa_sample_abs": 1023.75,
        },
        {
            "tree": 1,
            "source_chunk_index": 2**53 + 7,
            "itime": 0,
            "snr": 18.0,
            "dm": 42.1,
            "toa_sample_abs": 1024.0,
        },
    ]
    _, actual = _run_parity(rows, geometry)
    assert _host(actual.candidate_event_id).tolist() == [0, 0]
    assert _host(actual.events.source_chunk_index).tolist() == [-8]


_DM_BOUNDARY_CASES = (
    ("positive_on", 1.0, True),
    ("positive_inside", np.nextafter(1.0, 0.0), True),
    ("positive_outside", np.nextafter(1.0, np.inf), False),
    ("negative_on", -1.0, True),
    ("negative_inside", np.nextafter(-1.0, 0.0), True),
    ("negative_outside", np.nextafter(-1.0, -np.inf), False),
)


@pytest.mark.parametrize(
    ("name", "delta_dm", "compatible"),
    _DM_BOUNDARY_CASES,
    ids=[case[0] for case in _DM_BOUNDARY_CASES],
)
def test_positive_and_negative_dm_nextafter_boundaries(
        name, delta_dm, compatible):
    """Categories 10-11: both signed DM limits are exact and inclusive."""

    del name
    geometry = _geometry([0, 0], slope_lo=0.0, slope_hi=0.0)
    config = GroupingConfig(dm_tolerance_bins=1.0, time_padding_bins=1.0)
    rows = [
        {"tree": 0, "snr": 20.0, "dm": 0.0, "toa_sample_abs": 0.0},
        {
            "tree": 1,
            "snr": 10.0,
            "dm": np.float64(delta_dm),
            "toa_sample_abs": 0.0,
        },
    ]
    _, actual = _run_parity(rows, geometry, config=config)
    assignment = _host(actual.candidate_event_id)
    assert bool(assignment[0] == assignment[1]) is compatible


def test_direct_float32_dm_and_step_boundary_regression():
    """Direct float32 columns retain production subtraction rounding.

    The next float32 above +1 minus -1 rounds back to exactly 2 in float32.
    Production therefore accepts this pair at a two-bin inclusive DM limit.
    Promoting the operands to float64 *before* subtracting incorrectly rejects
    it, which was the reproduced fixed-double RawKernel defect.
    """

    geometry = replace(
        _geometry(
            [0, 0],
            dm_steps=[2.0, 2.0],
            time_steps=[1.0, 1.0],
            slope_lo=0.0,
            slope_hi=0.0,
        ),
        dm_step_by_tree=cp.asarray([2.0, 2.0], dtype=cp.float32),
    )
    candidates = _candidates(geometry, [
        {"tree": 0, "snr": 20.0, "toa_sample_abs": 0.0},
        {"tree": 1, "snr": 10.0, "toa_sample_abs": 0.0},
    ])
    dm = np.asarray([
        np.float32(-1.0),
        np.nextafter(np.float32(1.0), np.float32(np.inf)),
    ], dtype=np.float32)
    candidates = replace(candidates, dm=cp.asarray(dm, dtype=cp.float32))
    config = GroupingConfig(dm_tolerance_bins=1.0, time_padding_bins=1.0)

    expected = group_candidates(candidates, geometry, config=config)
    assert expected.candidates.dm.dtype == cp.float32
    assert expected.candidates.dm_step.dtype == cp.float32
    assert _host(expected.candidate_event_id).tolist() == [0, 0]

    actual = prototype.group_candidates_gpu_representative(
        candidates, geometry, config=config
    )
    assert_gpu_grouping_results_equal(cp, expected, actual)


def test_direct_mixed_float32_coordinates_float64_steps():
    """Mixed predicate columns follow CuPy's operation-level promotion rules."""

    geometry = _geometry(
        [0, 0],
        dm_steps=[2.0, 2.0],
        time_steps=[2.0, 2.0],
        slope_lo=0.0,
        slope_hi=0.0,
    )
    candidates = _candidates(geometry, [
        {"tree": 0, "snr": 20.0},
        {"tree": 1, "snr": 10.0},
    ])
    boundary = np.asarray([
        np.float32(-1.0),
        np.nextafter(np.float32(1.0), np.float32(np.inf)),
    ], dtype=np.float32)
    candidates = replace(
        candidates,
        dm=cp.asarray(boundary, dtype=cp.float32),
        toa_sample_abs=cp.asarray(boundary, dtype=cp.float32),
    )
    config = GroupingConfig(dm_tolerance_bins=1.0, time_padding_bins=1.0)

    expected = group_candidates(candidates, geometry, config=config)
    assert expected.candidates.dm.dtype == cp.float32
    assert expected.candidates.toa_sample_abs.dtype == cp.float32
    assert expected.candidates.dm_step.dtype == cp.float64
    assert expected.candidates.time_step_samples.dtype == cp.float64
    assert _host(expected.candidate_event_id).tolist() == [0, 0]

    actual = prototype.group_candidates_gpu_representative(
        candidates, geometry, config=config
    )
    assert_gpu_grouping_results_equal(cp, expected, actual)


_TIME_BOUNDARY_CASES = (
    ("lower_on", -1.0, True),
    ("lower_inside", np.nextafter(-1.0, np.inf), True),
    ("lower_outside", np.nextafter(-1.0, -np.inf), False),
    ("upper_on", 3.0, True),
    ("upper_inside", np.nextafter(3.0, -np.inf), True),
    ("upper_outside", np.nextafter(3.0, np.inf), False),
)


@pytest.mark.parametrize(
    ("name", "delta_toa", "compatible"),
    _TIME_BOUNDARY_CASES,
    ids=[case[0] for case in _TIME_BOUNDARY_CASES],
)
def test_both_residual_time_interval_edges_with_nextafter(
        name, delta_toa, compatible):
    """Category 12: lower/upper residual-time comparisons use inclusive bounds."""

    del name
    # delta_dm=1 gives edges [0, 2]; one time bin of padding gives [-1, 3].
    geometry = _geometry([0, 0], slope_lo=0.0, slope_hi=2.0)
    config = GroupingConfig(dm_tolerance_bins=2.0, time_padding_bins=1.0)
    rows = [
        {"tree": 0, "snr": 20.0, "dm": 0.0, "toa_sample_abs": 0.0},
        {
            "tree": 1,
            "snr": 10.0,
            "dm": 1.0,
            "toa_sample_abs": np.float64(delta_toa),
        },
    ]
    _, actual = _run_parity(rows, geometry, config=config)
    assignment = _host(actual.candidate_event_id)
    assert bool(assignment[0] == assignment[1]) is compatible


def test_equal_snr_exercises_every_provenance_tie_breaker():
    """Category 13: native integer keys reproduce exact production priority."""

    geometry = _geometry([0, 0, 1])
    common = {
        "snr": 20.0,
        "beam_id": 1,
        "source_chunk_index": 0,
        "tree": 0,
        "idm": 0,
        "itime": 0,
        "argmax_token": 0,
    }
    rows = [
        dict(common),
        dict(common),
        {**common, "argmax_token": np.uint32(1)},
        {**common, "itime": 1},
        {**common, "idm": 1},
        {**common, "tree": 1},
        {**common, "source_chunk_index": 2**53 + 7},
        {**common, "tree": 2},
        {**common, "beam_id": 2},
        {**common, "snr": 21.0, "beam_id": 99},
        {**common, "snr": 19.0, "beam_id": 0},
    ]
    # Physical fields are deliberately not priority keys.  Separate them enough
    # that every row is isolated and event representatives reveal the full sort.
    for i, row in enumerate(rows):
        row["dm"] = 10_000.0 * i
        row["toa_sample_abs"] = 1_000_000.0 * i
    expected_order = [9, 0, 1, 2, 3, 4, 5, 6, 7, 8, 10]
    _, actual = _run_parity(rows, geometry)
    assert _host(actual.events.representative_candidate_index).tolist() == (
        expected_order
    )


def test_equal_provenance_uses_original_stable_input_order():
    """Category 14: exact grouping-key ties retain original row order."""

    geometry = _geometry([0])
    rows = [
        {
            "beam_id": -4,
            "source_chunk_index": -(2**53) - 9,
            "tree": 0,
            "idm": 8,
            "itime": 13,
            "argmax_token": np.uint32(0xF0000001),
            "snr": 12.5,
            "dm": float(i),
            "toa_sample_abs": float(100 * i),
            "edge_flags": i,
        }
        for i in range(8)
    ]
    _, actual = _run_parity(rows, geometry)
    assert _host(actual.events.representative_candidate_index).tolist() == list(
        range(8)
    )
    assert _host(actual.events.edge_flags).tolist() == list(range(8))


def test_representative_and_member_flags_remain_distinct():
    """Categories 15-16: member flags survive; event flags are seed-only."""

    geometry = _geometry([0, 0, 0])
    representative_flag = int(EdgeFlag.DM_LOW)
    member_flag = int(EdgeFlag.STARTUP_INCOMPLETE)
    isolated_flag = int(EdgeFlag.DM_HIGH | EdgeFlag.STARTUP_INCOMPLETE)
    rows = [
        {
            "beam_id": 7,
            "tree": 0,
            "snr": 30.0,
            "edge_flags": representative_flag,
        },
        {
            "beam_id": 7,
            "tree": 1,
            "snr": 25.0,
            "edge_flags": member_flag,
        },
        {
            "beam_id": 8,
            "tree": 2,
            "snr": 20.0,
            "edge_flags": isolated_flag,
        },
    ]
    _, actual = _run_parity(rows, geometry)
    assert _host(actual.events.edge_flags).tolist() == [
        representative_flag,
        isolated_flag,
    ]
    assert _host(actual.candidates.edge_flags).tolist() == [
        representative_flag,
        member_flag,
        isolated_flag,
    ]
    assert _member_indices_by_event(actual) == [[0, 1], [2]]
    assert _host(actual.events.edge_flags)[0] != (
        representative_flag | member_flag
    )


def test_mandatory_non_transitive_a_b_c_chain():
    """Category 17: A~B and B~C never permits C to grow A's event."""

    geometry = _geometry([0, 0, 0], slope_lo=0.0, slope_hi=0.0)
    config = GroupingConfig(dm_tolerance_bins=1.0, time_padding_bins=1.0)
    rows = [
        {"tree": 0, "snr": 30.0, "dm": 0.0, "toa_sample_abs": 0.0},
        {"tree": 1, "snr": 20.0, "dm": 0.75, "toa_sample_abs": 0.0},
        {"tree": 2, "snr": 10.0, "dm": 1.5, "toa_sample_abs": 0.0},
    ]
    _, actual = _run_parity(rows, geometry, config=config)
    assert _host(actual.candidate_event_id).tolist() == [0, 0, 1]
    assert _host(actual.events.representative_candidate_index).tolist() == [0, 2]
    assert _host(actual.events.member_count).tolist() == [2, 1]
    assert _member_indices_by_event(actual) == [[0, 1], [2]]


def test_partition_events_are_remapped_by_global_representative_rank():
    """Category 18: independent partitions restore interleaved global event IDs."""

    geometry = _geometry([0, 0, 1, 1])
    rows = [
        {"beam_id": 0, "tree": 0, "snr": 100.0},
        {"beam_id": 0, "tree": 0, "snr": 85.0},
        {"beam_id": 0, "tree": 1, "snr": 60.0},
        {"beam_id": 1, "tree": 0, "snr": 95.0},
        {"beam_id": 1, "tree": 1, "snr": 50.0},
        {"beam_id": 0, "tree": 2, "snr": 90.0},
        {"beam_id": 0, "tree": 3, "snr": 40.0},
    ]
    candidates = _candidates(geometry, rows)
    expected = group_candidates(candidates, geometry)
    actual = prototype.group_candidates_gpu_representative(candidates, geometry)
    assert_gpu_grouping_results_equal(cp, expected, actual)
    assert _host(actual.events.representative_candidate_index).tolist() == [
        0, 3, 5, 1
    ]
    assert _host(actual.candidate_event_id).tolist() == [0, 3, 0, 1, 1, 2, 2]
    assert _member_indices_by_event(actual) == [[0, 2], [3, 4], [5, 6], [1]]
    assert prototype.gpu_partition_statistics(candidates) == (3, 3)


def test_partition_containing_only_one_tree():
    """Category 19: a one-tree partition emits one event per candidate."""

    geometry = _geometry([5, 5, 5])
    rows = [
        {"beam_id": -3, "tree": 2, "snr": snr}
        for snr in (10.0, 40.0, 20.0, 30.0, 30.0)
    ]
    _, actual = _run_parity(rows, geometry)
    assert _host(actual.events.representative_candidate_index).tolist() == [
        1, 3, 4, 2, 0
    ]
    assert _host(actual.events.member_count).tolist() == [1] * len(rows)


def test_arbitrary_labels_and_more_than_four_trees():
    """Category 20: no four-tree assumption enters winner bookkeeping."""

    ntrees = 12
    geometry = _geometry(
        [4] * ntrees,
        dm_steps=np.linspace(0.5, 2.5, ntrees),
        time_steps=np.linspace(1.0, 3.0, ntrees),
    )
    labels = [11, 2, 9, 4, 7, 0]
    rows = [
        {"tree": tree, "snr": 70.0 - i, "dm": 88.0, "toa_sample_abs": 9.0}
        for i, tree in enumerate(labels)
    ]
    _, actual = _run_parity(rows, geometry)
    assert len(actual.events) == 1
    assert _host(actual.events.member_count).tolist() == [len(labels)]
    member_trees = _host(
        actual.candidates.tree[actual.members.candidate_index]
    ).tolist()
    assert member_trees == labels
    assert len(set(member_trees)) > 4


def test_repeated_execution_is_bitwise_deterministic():
    """Category 21: identical input produces identical complete tables repeatedly."""

    geometry = _geometry([0, 0, 0, 1, 1, 1])
    rows = [
        {
            "beam_id": i % 2,
            "tree": i % 6,
            "source_chunk_index": (i % 3) - 1,
            "snr": float(30 - (i % 5)),
            "dm": 100.0 + 0.2 * (i % 3),
            "toa_sample_abs": 1000.0 + 0.25 * (i % 4),
        }
        for i in range(36)
    ]
    candidates = _candidates(geometry, rows)
    expected = group_candidates(candidates, geometry)
    runs = [
        prototype.group_candidates_gpu_representative(candidates, geometry)
        for _ in range(4)
    ]
    for actual in runs:
        assert_gpu_grouping_results_equal(cp, expected, actual)
        assert_gpu_grouping_results_equal(cp, runs[0], actual)


def _random_rows(seed, n=120):
    """Generate tied, multi-partition dense and sparse rows deterministically."""

    rng = np.random.default_rng(seed)
    rows = []
    for i in range(n):
        if i < n // 2:
            tree = int(rng.integers(0, 4))
            beam = int(rng.integers(0, 2))
            dm = 100.0 + float(rng.choice([-0.3, 0.0, 0.3]))
            toa = 1000.0 + float(rng.choice([-0.5, 0.0, 0.5]))
        else:
            tree = int(rng.integers(0, 12))
            beam = int(rng.integers(0, 3))
            dm = 1000.0 + 10.0 * i + float(rng.uniform(-0.1, 0.1))
            toa = -5000.0 + 20.0 * i + float(rng.uniform(-0.2, 0.2))
        rows.append({
            "beam_id": beam,
            "tree": tree,
            "source_chunk_index": int(rng.integers(-2, 3)),
            "idm": int(rng.integers(0, 256)),
            "itime": int(rng.integers(0, 1024)),
            "argmax_token": np.uint32(rng.integers(0, 2**32, dtype=np.uint32)),
            "snr": float(rng.integers(10, 17)),
            "edge_flags": int(rng.integers(0, 32)),
            "dm": dm,
            "toa_sample_abs": toa,
            "width_samp": float(rng.choice([1.0, 2.0, 4.0, 8.0])),
            "width_ms": float(rng.choice([1.0, 2.0, 4.0, 8.0])),
        })
    return rows


@pytest.mark.parametrize("seed", [0, 1, 17, 2026, 8675309])
def test_seeded_randomized_dense_and_sparse_parity(seed):
    """Category 22: randomized exact parity spans beams/families/chunks/ties."""

    geometry = _geometry(
        [0] * 4 + [1] * 4 + [2] * 4,
        dm_steps=[0.5, 1.0, 2.0, 4.0] * 3,
        time_steps=[1.0, 2.0, 4.0, 8.0] * 3,
        slope_lo=0.0,
        slope_hi=2.0,
    )
    _run_parity(_random_rows(seed), geometry)


def test_medium_catalog_uses_multiple_blocks_and_scan_strides():
    """Category 23: two 300-row partitions exceed one 256-thread scan tile."""

    ntrees = 300
    geometry = _geometry([0] * ntrees)
    rows = [
        {
            "beam_id": beam,
            "tree": tree,
            "snr": 20.0,
            "dm": 123.0,
            "toa_sample_abs": 456.0,
        }
        for beam in (3, 7)
        for tree in range(ntrees)
    ]
    candidates = _candidates(geometry, rows)
    expected = group_candidates(candidates, geometry)
    actual = prototype.group_candidates_gpu_representative(candidates, geometry)
    assert_gpu_grouping_results_equal(cp, expected, actual)
    assert prototype.gpu_partition_statistics(candidates) == (2, ntrees)
    assert len(actual.events) == 2
    assert _host(actual.events.member_count).tolist() == [ntrees, ntrees]


def test_all_isolated_worst_case_representative_iterations():
    """Category 24: an all-isolated partition exercises quadratic seed scans."""

    n = 96
    geometry = _geometry([0])
    rows = [
        {
            "tree": 0,
            "snr": float(100 - (i % 11)),
            "idm": i % 7,
            "itime": i % 13,
            "argmax_token": np.uint32(i % 5),
            "dm": float(i),
            "toa_sample_abs": float(i),
        }
        for i in range(n)
    ]
    _, actual = _run_parity(rows, geometry)
    assert len(actual.events) == n
    assert _host(actual.events.member_count).tolist() == [1] * n


def test_highly_dense_catalog_exercises_winner_reduction():
    """Category 25: many directly compatible trees collapse in one event."""

    ntrees = 128
    geometry = _geometry(
        [9] * ntrees,
        dm_steps=1.0 + (np.arange(ntrees) % 5),
        time_steps=1.0 + (np.arange(ntrees) % 7),
    )
    rows = [
        {
            "tree": tree,
            "snr": float(100 - (tree % 9)),
            "idm": tree % 11,
            "itime": tree % 13,
            "argmax_token": np.uint32(tree % 17),
            "dm": 42.0,
            "toa_sample_abs": -17.0,
        }
        for tree in range(ntrees)
    ]
    _, actual = _run_parity(rows, geometry)
    assert len(actual.events) == 1
    assert _host(actual.events.member_count).tolist() == [ntrees]
    assert len(set(_host(actual.members.candidate_index).tolist())) == ntrees


def test_hot_path_structure_has_no_host_controlled_representative_loop():
    """Audit fixed launches, device-side looping, and forbidden host transfers."""

    entry_source = textwrap.dedent(inspect.getsource(
        prototype.group_candidates_gpu_representative
    ))
    entry_tree = ast.parse(entry_source)
    assert not any(
        isinstance(node, (ast.For, ast.AsyncFor, ast.While))
        for node in ast.walk(entry_tree)
    )
    assert "cp.asnumpy" not in entry_source
    assert ".get(" not in entry_source
    assert "np.asarray" not in entry_source
    # Both scalar reads are fixed defensive all-assigned invariants; neither is
    # nested in candidate/event control flow (the AST loop check above proves it).
    assert entry_source.count(".item()") <= 2
    assert entry_source.count("_raw_kernel(") == 1

    imported_from_production = {
        alias.name
        for node in ast.walk(ast.parse(inspect.getsource(prototype)))
        if isinstance(node, ast.ImportFrom)
        and node.module == "pirate_frb.OfflineCandidateGrouper"
        for alias in node.names
    }
    assert "group_candidates" not in imported_from_production

    fixed_sort_source = textwrap.dedent(inspect.getsource(
        prototype._stable_sort_from_least_to_most
    ))
    fixed_sort_tree = ast.parse(fixed_sort_source)
    loops = [node for node in ast.walk(fixed_sort_tree) if isinstance(node, ast.For)]
    assert len(loops) == 1
    assert isinstance(loops[0].iter, ast.Name)
    assert loops[0].iter.id == "keys"

    module_source = inspect.getsource(prototype)
    assert "cp.asnumpy" not in module_source
    assert "np.asarray(" not in module_source
    assert "DBSCAN" not in module_source
    assert "connected component" not in module_source.lower()
    assert "(n, n)" not in module_source.lower()
    assert prototype.SHARED_MEMORY_CANDIDATE_TILING is False
    assert prototype.RAW_MODULE_BACKEND == "nvrtc"
    assert "--fmad=false" in prototype.RAW_MODULE_COMPILE_OPTIONS
    assert not any(
        "fast_math" in option or "use_fast_math" in option
        for option in prototype.RAW_MODULE_COMPILE_OPTIONS
    )

    cuda_source = prototype._CUDA_SOURCE
    assert "for (long long representative_position" in cuda_source
    assert "atomicMin" in cuda_source
    assert "__syncthreads" in cuda_source
    assert "representative_position + 1 + tid" in cuda_source
    assert "tree[position] == representative_tree" in cuda_source
