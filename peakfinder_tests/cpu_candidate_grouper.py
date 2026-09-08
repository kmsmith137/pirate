"""Exact benchmark-side CPU implementation of PIRATE candidate grouping.

The production grouper intentionally keeps candidate-sized columns on the GPU,
but its representative-seeded Python loop performs many tiny CUDA operations
and scalar synchronizations.  This module is a benchmark alternative, not a
production replacement: it normalizes with the production schema, copies only
the columns needed for grouping to the host, performs the same deterministic
algorithm within independent ``(beam_id, primary_tree_index)`` partitions, and
reconstructs the normal GPU-resident result from compact integer indices.

Decoded physical/reporting columns remain resident on the GPU throughout.  A
downstream GPU classifier can therefore consume the reconstructed event table
without round-tripping S/N maps or event feature columns through host memory.
"""

from __future__ import annotations

from dataclasses import dataclass, fields
import math
import time
from typing import Any, Mapping

import numpy as np


HOST_COLUMN_NAMES = (
    "beam_id",
    "primary_tree_index",
    "tree",
    "source_chunk_index",
    "idm",
    "itime",
    "argmax_token",
    "snr",
    "dm",
    "toa_sample_abs",
    "dm_step",
    "time_step_samples",
)


@dataclass(frozen=True)
class HostGroupingDecisions:
    """Compact host result sufficient to rebuild production GPU tables."""

    processing_order: np.ndarray
    candidate_event_id: np.ndarray
    representatives: np.ndarray
    member_count: np.ndarray
    member_order: np.ndarray
    partition_count: int
    largest_partition: int


@dataclass(frozen=True)
class CpuGroupingDiagnostics:
    """Measured subdivisions of one complete CPU grouping round trip."""

    candidate_normalization_ms: float
    gpu_to_cpu_transfer_ms: float
    cpu_grouping_ms: float
    cpu_to_gpu_result_ms: float
    gpu_to_cpu_bytes: int
    cpu_to_gpu_bytes: int
    partition_count: int
    largest_partition: int

    @property
    def stage_sum_ms(self) -> float:
        """Return the diagnostic sum; the enclosing wall timer is authoritative."""

        return (
            self.candidate_normalization_ms
            + self.gpu_to_cpu_transfer_ms
            + self.cpu_grouping_ms
            + self.cpu_to_gpu_result_ms
        )


def _validate_host_columns(columns: Mapping[str, np.ndarray]) -> int:
    if set(columns) != set(HOST_COLUMN_NAMES):
        missing = sorted(set(HOST_COLUMN_NAMES) - set(columns))
        extra = sorted(set(columns) - set(HOST_COLUMN_NAMES))
        raise ValueError(
            f"host grouping columns disagree with the schema: "
            f"missing={missing}, extra={extra}"
        )
    arrays = tuple(columns[name] for name in HOST_COLUMN_NAMES)
    if any(not isinstance(array, np.ndarray) or array.ndim != 1 for array in arrays):
        raise TypeError("every host grouping column must be a one-dimensional ndarray")
    n = int(arrays[0].size)
    if any(int(array.size) != n for array in arrays):
        raise ValueError("host grouping columns differ in length")
    expected_dtypes = {
        "beam_id": np.dtype(np.int32),
        "primary_tree_index": np.dtype(np.int32),
        "tree": np.dtype(np.int32),
        "source_chunk_index": np.dtype(np.int64),
        "idm": np.dtype(np.int32),
        "itime": np.dtype(np.int32),
        "argmax_token": np.dtype(np.uint32),
        "snr": np.dtype(np.float64),
        "dm": np.dtype(np.float64),
        "toa_sample_abs": np.dtype(np.float64),
        "dm_step": np.dtype(np.float64),
        "time_step_samples": np.dtype(np.float64),
    }
    for name, dtype in expected_dtypes.items():
        if columns[name].dtype != dtype:
            raise TypeError(
                f"host grouping column {name!r} has dtype "
                f"{columns[name].dtype}, expected {dtype}"
            )
    return n


def host_processing_order(columns: Mapping[str, np.ndarray]) -> np.ndarray:
    """Reproduce production's stable, exact-dtype candidate priority order."""

    n = _validate_host_columns(columns)
    order = np.arange(n, dtype=np.int64)
    keys = (
        columns["argmax_token"],
        columns["itime"],
        columns["idm"],
        columns["tree"],
        columns["source_chunk_index"],
        columns["primary_tree_index"],
        columns["beam_id"],
        -columns["snr"],
    )
    # Match production: stable sorts from least- to most-significant key.
    for key in keys:
        order = order[np.argsort(key[order], kind="stable")]
    return order


def group_host_columns(
        columns: Mapping[str, np.ndarray], geometry: Any, config: Any,
        ) -> HostGroupingDecisions:
    """Apply production grouping semantics within exact independent partitions.

    Candidates in different beams or primary-tree families can never be
    compatible in production.  Grouping those partitions independently does
    not change membership.  Sorting the resulting representatives by their
    rank in the global production order then restores exact event numbering.
    """

    n = _validate_host_columns(columns)
    for name in (
            "residual_slope_lo_samples_per_dm",
            "residual_slope_hi_samples_per_dm"):
        value = float(getattr(geometry, name))
        if not math.isfinite(value):
            raise ValueError(f"geometry {name} must be finite")
    dm_tolerance = float(getattr(config, "dm_tolerance_bins"))
    time_padding = float(getattr(config, "time_padding_bins"))
    if (not math.isfinite(dm_tolerance) or dm_tolerance < 0.0
            or not math.isfinite(time_padding) or time_padding < 0.0):
        raise ValueError("grouping tolerances must be finite and non-negative")

    order = host_processing_order(columns)
    rank_by_candidate = np.empty(n, dtype=np.int64)
    rank_by_candidate[order] = np.arange(n, dtype=np.int64)

    partitions: dict[tuple[int, int], list[int]] = {}
    for candidate in order:
        key = (
            int(columns["beam_id"][candidate]),
            int(columns["primary_tree_index"][candidate]),
        )
        partitions.setdefault(key, []).append(int(candidate))

    assigned = np.zeros(n, dtype=np.bool_)
    partition_events: list[tuple[int, np.ndarray]] = []
    slope_lo = float(geometry.residual_slope_lo_samples_per_dm)
    slope_hi = float(geometry.residual_slope_hi_samples_per_dm)
    largest_partition = max(map(len, partitions.values()), default=0)

    for partition_values in partitions.values():
        partition_order = np.asarray(partition_values, dtype=np.int64)
        for seed_value in partition_order:
            seed = int(seed_value)
            if assigned[seed]:
                continue
            other = partition_order[~assigned[partition_order]]
            ddm = columns["dm"][other] - columns["dm"][seed]
            dtoa = (
                columns["toa_sample_abs"][other]
                - columns["toa_sample_abs"][seed]
            )
            dm_limit = dm_tolerance * np.maximum(
                columns["dm_step"][other], columns["dm_step"][seed]
            )
            edge_lo = ddm * slope_lo
            edge_hi = ddm * slope_hi
            padding = time_padding * np.maximum(
                columns["time_step_samples"][other],
                columns["time_step_samples"][seed],
            )
            compatible = other[
                (columns["tree"][other] != columns["tree"][seed])
                & (np.abs(ddm) <= dm_limit)
                & (dtoa >= np.minimum(edge_lo, edge_hi) - padding)
                & (dtoa <= np.maximum(edge_lo, edge_hi) + padding)
            ]
            if compatible.size:
                # Filtering retains global priority.  np.unique, like CuPy's
                # production call, returns the first position for each sorted
                # tree value and therefore selects one loudest row per tree.
                _, first = np.unique(
                    columns["tree"][compatible], return_index=True
                )
                selected = np.concatenate((
                    np.asarray([seed], dtype=np.int64),
                    compatible[first],
                ))
            else:
                selected = np.asarray([seed], dtype=np.int64)
            assigned[selected] = True
            partition_events.append((seed, selected))

    if n and not np.all(assigned):
        raise RuntimeError("CPU grouping left at least one candidate unassigned")

    partition_events.sort(key=lambda event: int(rank_by_candidate[event[0]]))
    representatives = np.asarray(
        [event[0] for event in partition_events], dtype=np.int64
    )
    member_count = np.asarray(
        [event[1].size for event in partition_events], dtype=np.int32
    )
    candidate_event_id = np.full(n, -1, dtype=np.int64)
    for event_id, (_, selected) in enumerate(partition_events):
        candidate_event_id[selected] = event_id
    if n and np.any(candidate_event_id < 0):
        raise RuntimeError("CPU grouping produced an incomplete event assignment")
    member_order = order[
        np.argsort(candidate_event_id[order], kind="stable")
    ]
    return HostGroupingDecisions(
        processing_order=order,
        candidate_event_id=candidate_event_id,
        representatives=representatives,
        member_count=member_count,
        member_order=member_order,
        partition_count=len(partitions),
        largest_partition=largest_partition,
    )


def _copy_normalized_columns_to_host(cp: Any, candidates: Any) -> dict[str, np.ndarray]:
    columns = {
        name: cp.asnumpy(getattr(candidates, name))
        for name in HOST_COLUMN_NAMES
    }
    _validate_host_columns(columns)
    return columns


def _rebuild_gpu_result(cp: Any, candidates: Any, decisions: HostGroupingDecisions):
    from pirate_frb.OfflineCandidateGrouper import (
        GpuEventTable,
        GpuGroupingResult,
        GpuMemberTable,
    )

    assignment = cp.asarray(decisions.candidate_event_id)
    representatives = cp.asarray(decisions.representatives)
    counts = cp.asarray(decisions.member_count)
    member_order = cp.asarray(decisions.member_order)
    nevents = int(decisions.representatives.size)
    events = GpuEventTable(
        event_id=cp.arange(nevents, dtype=cp.int64),
        representative_candidate_index=representatives,
        member_count=counts,
        beam_id=candidates.beam_id[representatives],
        primary_tree_index=candidates.primary_tree_index[representatives],
        tree=candidates.tree[representatives],
        source_chunk_index=candidates.source_chunk_index[representatives],
        idm=candidates.idm[representatives],
        itime=candidates.itime[representatives],
        snr=candidates.snr[representatives],
        argmax_token=candidates.argmax_token[representatives],
        edge_flags=candidates.edge_flags[representatives],
        dm=candidates.dm[representatives],
        toa_sample_abs=candidates.toa_sample_abs[representatives],
        width_samp=candidates.width_samp[representatives],
        width_ms=candidates.width_ms[representatives],
        freq_lo_MHz=candidates.freq_lo_MHz[representatives],
        freq_hi_MHz=candidates.freq_hi_MHz[representatives],
    )
    member_event_id = assignment[member_order]
    members = GpuMemberTable(
        event_id=member_event_id,
        candidate_index=member_order,
        is_representative=(
            member_order == representatives[member_event_id]
        ),
    )
    return GpuGroupingResult(
        candidates=candidates,
        events=events,
        members=members,
        candidate_event_id=assignment,
    )


def group_candidates_on_cpu(
        decoded: Any, geometry: Any, *, config: Any = None,
        cp_module: Any = None,
        ) -> tuple[Any, CpuGroupingDiagnostics]:
    """Run the complete GPU→CPU→GPU grouping path and return diagnostics."""

    if cp_module is None:
        import cupy as cp
    else:
        cp = cp_module
    from pirate_frb.OfflineCandidateGrouper import (
        GpuDecodedCandidates,
        GroupingConfig,
    )

    config = GroupingConfig() if config is None else config
    if not isinstance(config, GroupingConfig):
        raise TypeError("config must be GroupingConfig")
    stream = cp.cuda.get_current_stream()

    started = time.perf_counter()
    candidates = GpuDecodedCandidates.from_decoder_result(decoded, geometry)
    stream.synchronize()
    normalized = time.perf_counter()

    host_columns = _copy_normalized_columns_to_host(cp, candidates)
    transferred = time.perf_counter()

    decisions = group_host_columns(host_columns, geometry, config)
    grouped = time.perf_counter()

    result = _rebuild_gpu_result(cp, candidates, decisions)
    stream.synchronize()
    rebuilt = time.perf_counter()

    d2h_bytes = sum(int(array.nbytes) for array in host_columns.values())
    h2d_bytes = sum((
        int(decisions.candidate_event_id.nbytes),
        int(decisions.representatives.nbytes),
        int(decisions.member_count.nbytes),
        int(decisions.member_order.nbytes),
    ))
    diagnostics = CpuGroupingDiagnostics(
        candidate_normalization_ms=1_000.0 * (normalized - started),
        gpu_to_cpu_transfer_ms=1_000.0 * (transferred - normalized),
        cpu_grouping_ms=1_000.0 * (grouped - transferred),
        cpu_to_gpu_result_ms=1_000.0 * (rebuilt - grouped),
        gpu_to_cpu_bytes=d2h_bytes,
        cpu_to_gpu_bytes=h2d_bytes,
        partition_count=decisions.partition_count,
        largest_partition=decisions.largest_partition,
    )
    if not all(
            math.isfinite(value) and value >= 0.0
            for value in (
                diagnostics.candidate_normalization_ms,
                diagnostics.gpu_to_cpu_transfer_ms,
                diagnostics.cpu_grouping_ms,
                diagnostics.cpu_to_gpu_result_ms,
            )):
        raise AssertionError("CPU grouping diagnostics must be finite and non-negative")
    return result, diagnostics


def synchronized_cpu_grouping_wall_time(
        stream: Any, decoded: Any, geometry: Any, *, config: Any = None,
        cp_module: Any = None,
        ) -> tuple[float, Any, CpuGroupingDiagnostics]:
    """Measure the authoritative complete CPU grouper round-trip wall time."""

    stream.synchronize()
    started = time.perf_counter()
    result, diagnostics = group_candidates_on_cpu(
        decoded, geometry, config=config, cp_module=cp_module
    )
    stream.synchronize()
    elapsed_ms = 1_000.0 * (time.perf_counter() - started)
    if not math.isfinite(elapsed_ms) or elapsed_ms < 0.0:
        raise AssertionError("CPU grouper wall time must be finite and non-negative")
    return elapsed_ms, result, diagnostics


def assert_gpu_grouping_results_equal(cp: Any, expected: Any, actual: Any) -> None:
    """Assert exact production/CPU equality for candidates, events, and members."""

    if len(expected.candidates) != len(actual.candidates):
        raise AssertionError("candidate counts differ")
    comparisons = {
        "candidate_event_id": (
            expected.candidate_event_id, actual.candidate_event_id
        ),
    }
    for table_name in ("candidates", "events", "members"):
        expected_table = getattr(expected, table_name)
        actual_table = getattr(actual, table_name)
        for field in fields(expected_table):
            comparisons[f"{table_name}.{field.name}"] = (
                getattr(expected_table, field.name),
                getattr(actual_table, field.name),
            )
    for name, (left, right) in comparisons.items():
        if left.shape != right.shape or left.dtype != right.dtype:
            raise AssertionError(
                f"{name} shape/dtype differs: {left.shape}/{left.dtype} "
                f"!= {right.shape}/{right.dtype}"
            )
        if not bool(cp.array_equal(left, right, equal_nan=True).item()):
            raise AssertionError(f"{name} differs between production and CPU grouping")


__all__ = [
    "CpuGroupingDiagnostics",
    "HOST_COLUMN_NAMES",
    "HostGroupingDecisions",
    "assert_gpu_grouping_results_equal",
    "group_candidates_on_cpu",
    "group_host_columns",
    "host_processing_order",
    "synchronized_cpu_grouping_wall_time",
]
