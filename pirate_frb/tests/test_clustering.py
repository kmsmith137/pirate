"""Synthetic scientific checks for representative-based GPU grouping."""

import ast
from dataclasses import fields
import inspect
import textwrap

import numpy as np

from ..Clustering import (
    GpuDecodedCandidates,
    ClusteringTolerances,
    ClusteringGeometry,
    compatible_with_representative,
    cluster_candidates,
)
from ..BowtiePeakfinding import EdgeFlag


def _geometry(cp):
    """Return four trees spanning two primary-tree families."""

    return ClusteringGeometry(
        ntrees=4,
        primary_tree_index_by_tree=cp.asarray([0, 0, 1, 0], cp.int32),
        dm_step_by_tree=cp.asarray([1.0, 2.0, 1.0, 4.0], cp.float64),
        time_step_samples_by_tree=cp.asarray(
            [1.0, 2.0, 1.0, 4.0], cp.float64
        ),
        reference_freq_MHz=300.0,
        full_band_freq_lo_MHz=300.0,
        full_band_freq_hi_MHz=1500.0,
        time_sample_ms=1.0,
        residual_slope_lo_samples_per_dm=0.0,
        residual_slope_hi_samples_per_dm=2.0,
    )


def _candidates(cp, geometry, rows):
    """Build the production GPU table from compact test dictionaries."""

    n = len(rows)

    def values(name, default):
        return [row.get(name, default(i, row)) for i, row in enumerate(rows)]

    tree_host = np.asarray(
        values("tree", lambda i, row: i % geometry.ntrees), dtype=np.int32
    )
    families = cp.asnumpy(geometry.primary_tree_index_by_tree)[tree_host]
    dm_steps = cp.asnumpy(geometry.dm_step_by_tree)[tree_host]
    time_steps = cp.asnumpy(geometry.time_step_samples_by_tree)[tree_host]
    return GpuDecodedCandidates(
        beam_id=cp.asarray(values("beam_id", lambda i, row: 7), cp.int32),
        source_chunk_index=cp.asarray(
            values("source_chunk_index", lambda i, row: 0), cp.int64
        ),
        tree=cp.asarray(tree_host, cp.int32),
        idm=cp.asarray(values("idm", lambda i, row: i), cp.int32),
        itime=cp.asarray(values("itime", lambda i, row: i), cp.int32),
        snr=cp.asarray(values("snr", lambda i, row: 20.0 - i), cp.float64),
        argmax_token=cp.asarray(
            values("argmax_token", lambda i, row: i), cp.uint32
        ),
        edge_flags=cp.asarray(
            values("edge_flags", lambda i, row: 0), cp.uint8
        ),
        dm=cp.asarray(values("dm", lambda i, row: 100.0), cp.float64),
        toa_sample_abs=cp.asarray(
            values("toa", lambda i, row: 1000.0), cp.float64
        ),
        width_samp=cp.asarray(
            values("width_samp", lambda i, row: 2.0), cp.float64
        ),
        width_ms=cp.asarray(
            values("width_ms", lambda i, row: 2.0), cp.float64
        ),
        freq_lo_MHz=cp.asarray(
            values("freq_lo_MHz", lambda i, row: 300.0), cp.float64
        ),
        freq_hi_MHz=cp.asarray(
            values("freq_hi_MHz", lambda i, row: 1500.0), cp.float64
        ),
        primary_tree_index=cp.asarray(families, cp.int32),
        dm_step=cp.asarray(dm_steps, cp.float64),
        time_step_samples=cp.asarray(time_steps, cp.float64),
    )


def test_pair_compatibility_physical_boundaries(cuda_device_id=0):
    """Beam, family, tree, DM, and residual-time conditions are all required."""

    import cupy as cp

    with cp.cuda.Device(cuda_device_id):
        geometry = _geometry(cp)
        candidates = _candidates(cp, geometry, [
            {"tree": 0, "dm": 100.0, "toa": 1000.0},
            {"tree": 1, "dm": 101.0, "toa": 1001.0},
            {"tree": 0, "dm": 100.0, "toa": 1000.0},
            {"tree": 1, "beam_id": 8, "dm": 100.0, "toa": 1000.0},
            {"tree": 2, "dm": 100.0, "toa": 1000.0},
            {"tree": 3, "dm": 110.1, "toa": 1000.0},
            {"tree": 3, "dm": 100.0, "toa": 1010.1},
        ])
        match = compatible_with_representative(
            candidates,
            0,
            cp.arange(1, len(candidates), dtype=cp.int64),
            geometry,
            ClusteringTolerances(dm_tolerance_bins=1.5, time_padding_bins=1.0),
        )
        assert np.array_equal(
            cp.asnumpy(match),
            np.asarray([True, False, False, False, False, False]),
        )


def test_greedy_grouping_tree_choice_and_isolation(cuda_device_id=0):
    """The loudest compatible row per tree joins; all isolations remain exact."""

    import cupy as cp

    with cp.cuda.Device(cuda_device_id):
        geometry = _geometry(cp)
        candidates = _candidates(cp, geometry, [
            {"tree": 0, "snr": 30.0, "dm": 100.0, "toa": 1000.0},
            {"tree": 1, "snr": 25.0, "dm": 100.5, "toa": 1001.0},
            {"tree": 1, "snr": 24.0, "dm": 100.4, "toa": 1000.5},
            {"tree": 3, "snr": 23.0, "dm": 100.2, "toa": 999.5},
            {
                "tree": 1, "snr": 22.0, "beam_id": 8,
                "dm": 100.0, "toa": 1000.0,
            },
            {"tree": 2, "snr": 21.0, "dm": 100.0, "toa": 1000.0},
            {"tree": 3, "snr": 20.0, "dm": 100.0, "toa": 1100.0},
        ])
        grouped = cluster_candidates(candidates, geometry)
        event_id = cp.asnumpy(grouped.candidate_event_id)
        assert event_id[0] == event_id[1] == event_id[3]
        assert event_id[2] != event_id[0]
        assert event_id[4] != event_id[0]
        assert event_id[5] != event_id[0]
        assert event_id[6] != event_id[0]
        first_members = cp.asnumpy(
            grouped.members.candidate_index[
                grouped.members.event_id == event_id[0]
            ]
        )
        assert set(first_members.tolist()) == {0, 1, 3}
        member_trees = cp.asnumpy(candidates.tree[first_members])
        assert len(set(member_trees.tolist())) == len(member_trees)
        assert int(cp.asnumpy(grouped.events.representative_candidate_index)[0]) == 0


def test_grouping_across_chunks_uses_absolute_toa(cuda_device_id=0):
    """Chunk labels do not block association when absolute arrivals agree."""

    import cupy as cp

    with cp.cuda.Device(cuda_device_id):
        geometry = _geometry(cp)
        candidates = _candidates(cp, geometry, [
            {
                "tree": 0, "source_chunk_index": 0, "itime": 1023,
                "snr": 20.0, "dm": 42.0, "toa": 1023.75,
            },
            {
                "tree": 1, "source_chunk_index": 1, "itime": 0,
                "snr": 18.0, "dm": 42.1, "toa": 1024.0,
            },
        ])
        grouped = cluster_candidates(candidates, geometry)
        assert len(grouped.events) == 1
        assert len(grouped.members) == 2
        assert np.array_equal(
            cp.asnumpy(grouped.candidate_event_id), np.asarray([0, 0])
        )


def test_grouping_preserves_representative_and_member_startup_flags(
        cuda_device_id=0):
    """Event flags come from the seed; member flags remain per candidate."""

    import cupy as cp

    with cp.cuda.Device(cuda_device_id):
        geometry = _geometry(cp)
        startup = int(EdgeFlag.STARTUP_INCOMPLETE)
        candidates = _candidates(cp, geometry, [
            {
                "tree": 0, "snr": 30.0, "dm": 100.0, "toa": 1000.0,
                "edge_flags": 0,
            },
            {
                "tree": 1, "snr": 25.0, "dm": 100.2, "toa": 1000.2,
                "edge_flags": startup,
            },
            {
                "tree": 0, "beam_id": 8, "snr": 20.0,
                "dm": 100.0, "toa": 1000.0, "edge_flags": startup,
            },
        ])
        grouped = cluster_candidates(candidates, geometry)

        assert cp.asnumpy(grouped.events.edge_flags).tolist() == [0, startup]
        first_members = cp.asnumpy(
            grouped.members.candidate_index[
                grouped.members.event_id == cp.int64(0)
            ]
        )
        assert set(first_members.tolist()) == {0, 1}
        assert cp.asnumpy(
            grouped.candidates.edge_flags[first_members]
        ).tolist() == [0, startup]
        assert grouped.events.edge_flags.dtype == cp.uint8
        assert grouped.candidates.edge_flags.dtype == cp.uint8


def test_grouping_empty_and_gpu_residency(cuda_device_id=0):
    """Empty input is valid and production grouping has no array host copy."""

    import cupy as cp

    with cp.cuda.Device(cuda_device_id):
        geometry = _geometry(cp)
        empty = _candidates(cp, geometry, [])
        grouped = cluster_candidates(empty, geometry)
        assert len(grouped.events) == 0
        assert len(grouped.members) == 0
        assert int(grouped.candidate_event_id.size) == 0
        assert grouped.complete is True
        assert grouped.timed_out is False
        assert grouped.input_candidate_index.dtype == cp.int64
        assert int(grouped.input_candidate_index.size) == 0

        source = textwrap.dedent(inspect.getsource(cluster_candidates))
        assert "asnumpy" not in source and ".get(" not in source
        assert "N * N" not in source and "N, N" not in source
        tree = ast.parse(source)
        assert not any(
            isinstance(node, (ast.For, ast.AsyncFor, ast.While))
            for node in ast.walk(tree)
        )
        assert source.count("_raw_grouping_kernel(") == 1
        module_source = inspect.getsource(inspect.getmodule(cluster_candidates))
        assert "%globaltimer" in module_source
        assert "atomicCAS(\n                launch_deadline_ns" in module_source
        assert "test_timeout_after_work_tiles" in module_source
        assert "provisional_representative" in module_source


def _timeout_geometry(cp):
    """Return two compatible trees in one persistent-kernel partition."""

    return ClusteringGeometry(
        ntrees=2,
        primary_tree_index_by_tree=cp.asarray([0, 0], cp.int32),
        dm_step_by_tree=cp.asarray([1.0, 1.0], cp.float64),
        time_step_samples_by_tree=cp.asarray([1.0, 1.0], cp.float64),
        reference_freq_MHz=300.0,
        full_band_freq_lo_MHz=300.0,
        full_band_freq_hi_MHz=1500.0,
        time_sample_ms=1.0,
        residual_slope_lo_samples_per_dm=0.0,
        residual_slope_hi_samples_per_dm=0.0,
    )


def _assert_grouping_tables_equal(cp, left, right):
    """Assert bitwise equality of all GPU-resident grouping table columns."""

    assert cp.array_equal(
        left.candidate_event_id, right.candidate_event_id
    ).item()
    assert cp.array_equal(
        left.input_candidate_index, right.input_candidate_index
    ).item()
    for table_name in ("candidates", "events", "members"):
        left_table = getattr(left, table_name)
        right_table = getattr(right, table_name)
        for field in fields(left_table):
            assert cp.array_equal(
                getattr(left_table, field.name),
                getattr(right_table, field.name),
                equal_nan=True,
            ).item(), (table_name, field.name)


def test_timeout_discards_in_progress_event_and_rebases_indices(
        cuda_device_id=0):
    """Only the first complete event survives cancellation during event two."""

    import cupy as cp

    with cp.cuda.Device(cuda_device_id):
        geometry = _timeout_geometry(cp)
        # Event zero scans two 256-thread tiles and commits input rows 0/299.
        # The deterministic GPU hook expires on tile three, midway through the
        # next same-tree representative event.
        rows = [
            {
                "tree": 0,
                "snr": 1000.0 - i,
                "dm": 0.0,
                "toa": 0.0,
            }
            for i in range(299)
        ] + [{"tree": 1, "snr": 1.0, "dm": 0.0, "toa": 0.0}]
        candidates = _candidates(cp, geometry, rows)
        partial = cluster_candidates(
            candidates,
            geometry,
            _test_timeout_after_work_tiles=3,
        )

        assert partial.timed_out is True
        assert partial.complete is False
        assert cp.asnumpy(partial.input_candidate_index).tolist() == [0, 299]
        assert cp.asnumpy(partial.candidate_event_id).tolist() == [0, 0]
        assert cp.asnumpy(
            partial.events.representative_candidate_index
        ).tolist() == [0]
        assert cp.asnumpy(partial.events.member_count).tolist() == [2]
        assert cp.asnumpy(partial.members.event_id).tolist() == [0, 0]
        assert cp.asnumpy(partial.members.candidate_index).tolist() == [0, 1]
        assert cp.asnumpy(
            partial.members.is_representative
        ).tolist() == [True, False]
        assert len(partial.candidates) == len(partial.members) == 2
        assert int(cp.sum(partial.events.member_count).item()) == 2

        # The deterministic hook and integer winner selection produce the same
        # clean partial tables on every execution.
        repeated = cluster_candidates(
            candidates,
            geometry,
            _test_timeout_after_work_tiles=3,
        )
        assert repeated.timed_out is True
        _assert_grouping_tables_equal(cp, partial, repeated)


def test_timeout_before_first_commit_returns_typed_empty_partial(
        cuda_device_id=0):
    """Cancellation during event zero never leaks its provisional seed."""

    import cupy as cp

    with cp.cuda.Device(cuda_device_id):
        geometry = _timeout_geometry(cp)
        rows = [
            {"tree": i % 2, "snr": 500.0 - i, "dm": 0.0, "toa": 0.0}
            for i in range(300)
        ]
        partial = cluster_candidates(
            _candidates(cp, geometry, rows),
            geometry,
            _test_timeout_after_work_tiles=1,
        )
        assert partial.timed_out is True
        assert partial.complete is False
        assert len(partial.candidates) == 0
        assert len(partial.events) == 0
        assert len(partial.members) == 0
        assert partial.candidate_event_id.dtype == cp.int64
        assert partial.input_candidate_index.dtype == cp.int64
        assert partial.events.member_count.dtype == cp.int32
        assert partial.members.is_representative.dtype == cp.bool_


def test_timeout_after_final_commit_remains_explicit(cuda_device_id=0):
    """A deadline observed after an indivisible final commit stays a timeout."""

    import cupy as cp

    with cp.cuda.Device(cuda_device_id):
        geometry = _timeout_geometry(cp)
        candidates = _candidates(cp, geometry, [
            {"tree": 0, "snr": 20.0, "dm": 0.0, "toa": 0.0},
            {"tree": 1, "snr": 10.0, "dm": 0.0, "toa": 0.0},
        ])
        partial = cluster_candidates(
            candidates,
            geometry,
            _test_timeout_after_committed_events=1,
        )
        assert partial.timed_out is True
        assert partial.complete is False
        assert cp.asnumpy(partial.input_candidate_index).tolist() == [0, 1]
        assert cp.asnumpy(partial.candidate_event_id).tolist() == [0, 0]
        assert cp.asnumpy(partial.events.member_count).tolist() == [2]
        assert cp.asnumpy(partial.members.candidate_index).tolist() == [0, 1]


def test_timeout_defaults_validation_and_complete_input_mapping(
        cuda_device_id=0):
    """Zero disables timeout; statuses and original-row mapping are explicit."""

    import cupy as cp

    with cp.cuda.Device(cuda_device_id):
        assert ClusteringTolerances().dm_tolerance_bins == 1.5
        assert ClusteringTolerances().time_padding_bins == 1.5
        geometry = _timeout_geometry(cp)
        candidates = _candidates(cp, geometry, [
            {"tree": 0, "snr": 20.0},
            {"tree": 1, "snr": 10.0},
        ])
        complete = cluster_candidates(candidates, geometry, timeout_ms=0)
        assert complete.complete is True
        assert complete.timed_out is False
        assert cp.asnumpy(complete.input_candidate_index).tolist() == [0, 1]

        invalid = (
            (True, TypeError),
            (-1, ValueError),
            (np.nan, ValueError),
            (np.inf, ValueError),
            (1.0e308, ValueError),
            ("1", TypeError),
        )
        for value, error_type in invalid:
            try:
                cluster_candidates(candidates, geometry, timeout_ms=value)
            except error_type:
                pass
            else:
                raise AssertionError(
                    f"timeout_ms={value!r} did not raise {error_type.__name__}"
                )


def test_grouping_rejects_noncurrent_owning_device(cuda_device_id=0):
    """Kernel/workspace allocation cannot silently mix two CUDA devices."""

    import cupy as cp

    device_count = cp.cuda.runtime.getDeviceCount()
    if device_count < 2:
        return
    owner = int(cuda_device_id) % device_count
    other = (owner + 1) % device_count
    with cp.cuda.Device(owner):
        geometry = _timeout_geometry(cp)
        candidates = _candidates(cp, geometry, [
            {"tree": 0, "snr": 20.0},
            {"tree": 1, "snr": 10.0},
        ])
    with cp.cuda.Device(other):
        try:
            cluster_candidates(candidates, geometry)
        except ValueError as exc:
            assert "current CUDA device" in str(exc)
        else:
            raise AssertionError("cross-device grouping did not fail explicitly")
