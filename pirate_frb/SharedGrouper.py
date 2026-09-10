"""Shared causal peak-finding and grouping for offline and live adapters.

The processor consumes one source chunk at a time. It owns bounded copies of
map tails and compact candidate state, and never retains a producer's map views.
Both adapters use this state machine and the same extraction, decoding, and
grouping implementations. Input ordering and physical completion are explicit.
"""

from __future__ import annotations

from dataclasses import dataclass, fields
import operator

import numpy as np

from .GpuArgmaxDecoder import GpuArgmaxDecoder
from .OfflineCandidateGrouper import (
    GpuDecodedCandidates,
    GpuEventTable,
    GpuGroupingResult,
    GpuMemberTable,
    GroupingConfig,
    GroupingGeometry,
    group_candidates,
)
from .Peakfinders import (
    GpuRawCandidates,
    OfflinePeakExtractor,
    PeakFinderGeometry,
    concatenate_raw_candidates,
    make_startup_valid_mask,
)
from .TriggerCatalog import (
    catalog_batch_from_grouping_result,
    make_catalog_metadata,
    write_trigger_catalog,
)


@dataclass(frozen=True)
class GroupingWindow:
    """Final output and provenance for one owner-chunk grouping attempt."""

    owner_source_chunk_index: int
    grouped: GpuGroupingResult
    output_status: str

    def __post_init__(self):
        if self.output_status not in ("complete", "discarded", "partial"):
            raise ValueError("invalid grouping-window output status")
        expected_timeout = self.output_status != "complete"
        if self.grouped.timed_out != expected_timeout:
            raise ValueError("grouping-window status disagrees with GPU result")


def _optional_nonnegative_integer(value, name):
    """Normalize an optional exact non-negative orchestration integer."""

    if value is None:
        return None
    if isinstance(value, (bool, np.bool_)):
        raise TypeError(f"{name} must be an integer, not bool")
    try:
        value = int(operator.index(value))
    except TypeError as exc:
        raise TypeError(f"{name} must be an integer") from exc
    if value < 0:
        raise ValueError(f"{name} must be non-negative")
    return value


def _effective_grouping_halo_columns(geometries, halo_size):
    """Return each tree's resolved part of its configured next-chunk halo.

    Cross-window association is defined in native map coordinates.  The
    configured prefix is ``halo_size * time_radius``; after processing chunk
    ``i+1``, only its first ``ntime - time_radius`` centres are peak-resolved.
    Their intersection is the complete eligible domain for owner ``i``.
    """

    halo_size = _optional_nonnegative_integer(halo_size, "halo_size")
    if halo_size < 2:
        raise ValueError("halo_size must be at least 2")
    effective = []
    for tree, geometry in enumerate(geometries):
        ntime = _optional_nonnegative_integer(
            geometry.ntime, f"ntime for tree {tree}"
        )
        radius = _optional_nonnegative_integer(
            geometry.time_radius, f"time radius for tree {tree}"
        )
        if not 0 <= radius <= ntime:
            raise ValueError(f"tree {tree} has an invalid time radius")
        resolvable = ntime - radius
        effective.append(min(halo_size * radius, resolvable))
    return tuple(effective)


def _raw_take(raw, selector):
    """Gather a raw-candidate table without moving candidate data to host."""

    return GpuRawCandidates(**{
        field.name: getattr(raw, field.name)[selector]
        for field in fields(GpuRawCandidates)
    })


def _raw_concatenate(parts):
    """Concatenate compact raw pieces and restore deterministic GPU order."""

    return concatenate_raw_candidates(parts)


def _project_grouping_events(cp, grouped, keep_event):
    """Return a compact, internally valid projection of selected events.

    Candidate rows stay in their relative grouping-input order.  Event and
    member identifiers, representative indices, assignments, and the mapping
    back to the grouping call are all rebased on the GPU.
    """

    nevents = len(grouped.events)
    if (not isinstance(keep_event, cp.ndarray)
            or keep_event.dtype != cp.bool_
            or keep_event.shape != (nevents,)):
        raise TypeError("keep_event must be one Boolean GPU value per event")

    old_event = cp.flatnonzero(keep_event).astype(cp.int64, copy=False)
    event_by_old = cp.full(nevents, -1, dtype=cp.int64)
    event_by_old[old_event] = cp.arange(int(old_event.size), dtype=cp.int64)

    candidate_keep = keep_event[grouped.candidate_event_id]
    old_candidate = cp.flatnonzero(candidate_keep).astype(
        cp.int64, copy=False
    )
    candidate_by_old = cp.full(len(grouped.candidates), -1, dtype=cp.int64)
    candidate_by_old[old_candidate] = cp.arange(
        int(old_candidate.size), dtype=cp.int64
    )
    candidates = GpuDecodedCandidates(**{
        field.name: getattr(grouped.candidates, field.name)[old_candidate]
        for field in fields(GpuDecodedCandidates)
    })

    event_values = {
        field.name: getattr(grouped.events, field.name)[old_event]
        for field in fields(GpuEventTable)
    }
    event_values["event_id"] = cp.arange(
        int(old_event.size), dtype=cp.int64
    )
    event_values["representative_candidate_index"] = candidate_by_old[
        grouped.events.representative_candidate_index[old_event]
    ]
    events = GpuEventTable(**event_values)

    member_keep = keep_event[grouped.members.event_id]
    old_member_candidate = grouped.members.candidate_index[member_keep]
    members = GpuMemberTable(
        event_id=event_by_old[grouped.members.event_id[member_keep]],
        candidate_index=candidate_by_old[old_member_candidate],
        is_representative=grouped.members.is_representative[member_keep],
    )
    candidate_event_id = event_by_old[
        grouped.candidate_event_id[old_candidate]
    ]
    return GpuGroupingResult(
        candidates=candidates,
        events=events,
        members=members,
        candidate_event_id=candidate_event_id,
        input_candidate_index=grouped.input_candidate_index[old_candidate],
        timed_out=grouped.timed_out,
        complete=grouped.complete,
    )


def _owned_grouping_result(cp, grouped, owner_source_chunk_index):
    """Select groups containing at least one member from the owner chunk."""

    nevents = len(grouped.events)
    if nevents == 0:
        return _project_grouping_events(
            cp, grouped, cp.empty(0, dtype=cp.bool_)
        )
    owner_member = (
        grouped.candidates.source_chunk_index[
            grouped.members.candidate_index
        ] == owner_source_chunk_index
    )
    # CuPy's bincount rejects an empty input on the supported deployment
    # version.  Counting every member with a zero/one owner weight handles the
    # important future-only case without a host branch or candidate transfer.
    owned_count = cp.bincount(
        grouped.members.event_id,
        weights=owner_member.astype(cp.int32),
        minlength=nevents,
    )
    return _project_grouping_events(cp, grouped, owned_count != 0)


def _group_streaming_window(
        cp, owner_raw, halo_raw, owner_source_chunk_index, halo_chunk_index,
        decoder, grouping_geometry, grouping_config, *, timeout_ms,
        timeout_policy, group_function=None):
    """Group one owner plus next-chunk halo and apply its timeout policy.

    Returns ``(window, retained_halo)``.  Retained halo rows are candidates from
    fully processed future-only groups; they remain ordinary bounded state for
    their own future window.  Unprocessed halo rows are never carried after a
    timeout.
    """

    window_raw = _raw_concatenate((owner_raw, halo_raw))
    decoded = decoder.decode(window_raw)
    function = group_candidates if group_function is None else group_function
    grouped = function(
        decoded,
        grouping_geometry,
        config=grouping_config,
        timeout_ms=timeout_ms,
    )
    owned = _owned_grouping_result(
        cp, grouped, owner_source_chunk_index
    )

    if grouped.timed_out and timeout_policy == "discard":
        output = _project_grouping_events(
            cp, grouped, cp.zeros(len(grouped.events), dtype=cp.bool_)
        )
        status = "discarded"
    elif grouped.timed_out:
        output = owned
        status = "partial"
    else:
        output = owned
        status = "complete"

    if halo_chunk_index is None or not len(halo_raw):
        retained_halo = GpuRawCandidates.empty()
    elif grouped.timed_out and timeout_policy == "discard":
        retained_halo = GpuRawCandidates.empty()
    else:
        # input_candidate_index addresses window_raw for both complete and
        # partial results.  Only fully committed rows exist in a partial result.
        committed = cp.zeros(len(window_raw), dtype=cp.bool_)
        committed[grouped.input_candidate_index] = True
        consumed = cp.zeros(len(window_raw), dtype=cp.bool_)
        consumed[output.input_candidate_index] = True
        retain = (
            (window_raw.source_chunk_index == halo_chunk_index)
            & committed
            & ~consumed
        )
        retained_halo = _raw_take(window_raw, retain)

    return (
        GroupingWindow(
            owner_source_chunk_index=owner_source_chunk_index,
            grouped=output,
            output_status=status,
        ),
        retained_halo,
    )


def _split_emitted_chunks(
        cp, emitted, owner_source_chunk_index, current_source_chunk_index):
    """Split newly resolved peaks into the previous and current chunk."""

    owner = emitted.source_chunk_index == owner_source_chunk_index
    current = emitted.source_chunk_index == current_source_chunk_index
    if len(emitted) and not bool(cp.all(owner | current).item()):
        raise RuntimeError(
            "peak extractor emitted a candidate outside the adjacent chunks"
        )
    return _raw_take(emitted, owner), _raw_take(emitted, current)


def _partition_current_halo(cp, current_raw, halo_columns_by_tree):
    """Split current-chunk rows at each ragged tree's configured left halo."""

    if not len(current_raw):
        empty = GpuRawCandidates.empty()
        return empty, empty
    limit = halo_columns_by_tree[current_raw.tree]
    in_halo = current_raw.itime.astype(cp.int64, copy=False) < limit
    return _raw_take(current_raw, in_halo), _raw_take(current_raw, ~in_halo)


class StreamingGrouper:
    """Single-use causal processor for a fixed physical beam batch.

    Input reads use the current CuPy stream and must complete before the adapter
    releases live output memory. Persistent tails and candidates own their memory.
    A failure latches the processor so partially advanced state cannot be reused.
    """

    def __init__(self, geometries, decoder, grouping_geometry, *, beam_ids,
                 producer_start_chunk, threshold, halo_size, timeout_ms,
                 timeout_policy, grouping_config, consume_window,
                 assume_steady_state=False,
                 grouping_halo_columns_by_tree=None,
                 extractor_factory=None, group_function=None):
        import cupy as cp
        if not callable(consume_window):
            raise TypeError("consume_window must be callable")
        if timeout_policy not in ("discard", "emit_partial"):
            raise ValueError("invalid grouping timeout policy")
        self.beam_ids = tuple(beam_ids)
        if not self.beam_ids or len(set(self.beam_ids)) != len(self.beam_ids):
            raise ValueError("beam IDs must be nonempty and unique")
        for beam in self.beam_ids:
            _optional_nonnegative_integer(beam, "beam ID")
        self.producer_start_chunk = _optional_nonnegative_integer(
            producer_start_chunk, "producer_start_chunk")
        self.assumed = producer_start_chunk is None
        if self.assumed and not assume_steady_state:
            raise ValueError(
                f"producer-start provenance is missing for beams {self.beam_ids}; "
                "pass assume_steady_state=True explicitly")
        self.startup_status = "assumed" if self.assumed else "authoritative"
        self.geometries = tuple(geometries)
        if not self.geometries:
            raise ValueError("at least one tree is required")
        halo_columns = (
            _effective_grouping_halo_columns(self.geometries, halo_size)
            if grouping_halo_columns_by_tree is None
            else tuple(grouping_halo_columns_by_tree))
        if len(halo_columns) != len(self.geometries):
            raise ValueError("grouping halo has the wrong tree count")
        for tree, (columns, geometry) in enumerate(zip(halo_columns, self.geometries)):
            value = _optional_nonnegative_integer(columns, "grouping halo")
            if value is None or value > geometry.ntime:
                raise ValueError(f"grouping halo for tree {tree} is outside its map")
        self.halo_columns_gpu = cp.asarray(halo_columns, dtype=cp.int64)
        factory = OfflinePeakExtractor if extractor_factory is None else extractor_factory
        self.extractors = tuple(factory(
            geometry, threshold=threshold, beam_ids=self.beam_ids,
            assume_steady_state=self.assumed, halo_size=halo_size,
        ) for geometry in self.geometries)
        self.decoder = decoder
        self.grouping_geometry = grouping_geometry
        self.grouping_config = grouping_config
        self.group_function = group_function
        self.consume_window = consume_window
        self.timeout_ms = timeout_ms
        self.timeout_policy = timeout_policy
        self.pending = GpuRawCandidates.empty()
        self.previous_chunk = None
        self.source_chunks = []
        self.closed = False
        self.complete = False
        self.failed = None

    @property
    def coverage(self):
        return tuple((beam, chunk) for beam in self.beam_ids for chunk in self.source_chunks)

    def _check_open(self):
        if self.failed is not None:
            raise RuntimeError("processor failed and cannot be reused") from self.failed
        if self.closed:
            raise RuntimeError("processor has already finished")

    def _group(self, owner, halo, halo_chunk):
        import cupy as cp
        return _group_streaming_window(
            cp, owner, halo, self.previous_chunk, halo_chunk, self.decoder,
            self.grouping_geometry, self.grouping_config,
            timeout_ms=self.timeout_ms, timeout_policy=self.timeout_policy,
            group_function=self.group_function)

    def process_chunk(self, snr_by_tree, argmax_by_tree, source_chunk_index):
        """Consume one consecutive source batch, emitting at most one window."""
        import cupy as cp
        self._check_open()
        source = _optional_nonnegative_integer(source_chunk_index, "source chunk")
        if source is None:
            raise ValueError("source chunk is required")
        if self.previous_chunk is not None and source != self.previous_chunk + 1:
            raise ValueError("source chunks must be consecutive")
        if len(snr_by_tree) != len(self.extractors) or len(argmax_by_tree) != len(self.extractors):
            raise ValueError("map tree count disagrees with the producer plan")
        try:
            emitted = _raw_concatenate(
                extractor.process_chunk(
                    snr_by_tree[tree], argmax_by_tree[tree], source,
                    startup_valid_mask=(None if self.assumed else make_startup_valid_mask(
                        self.geometries[tree], source, self.producer_start_chunk)),
                ) for tree, extractor in enumerate(self.extractors))
            if self.previous_chunk is None:
                if len(emitted) and not bool(cp.all(emitted.source_chunk_index == source).item()):
                    raise RuntimeError("first peak emission has invalid ownership")
                self.pending = emitted
            else:
                owner_tail, current = _split_emitted_chunks(
                    cp, emitted, self.previous_chunk, source)
                owner = _raw_concatenate((self.pending, owner_tail))
                halo, outside = _partition_current_halo(cp, current, self.halo_columns_gpu)
                window, retained = self._group(owner, halo, source)
                self.consume_window(window)
                self.pending = _raw_concatenate((outside, retained))
            self.previous_chunk = source
            self.source_chunks.append(source)
        except BaseException as exc:
            self.failed = exc
            raise

    def finish(self, *, physical_end):
        """Close the stream; flush only at the verified physical observation end."""
        import cupy as cp
        self._check_open()
        if type(physical_end) is not bool:
            raise TypeError("physical_end must be bool")
        try:
            if physical_end and self.previous_chunk is not None:
                tail = _raw_concatenate(extractor.flush() for extractor in self.extractors)
                if len(tail) and not bool(cp.all(
                        tail.source_chunk_index == self.previous_chunk).item()):
                    raise RuntimeError("final peak emission has invalid ownership")
                window, _ = self._group(
                    _raw_concatenate((self.pending, tail)), GpuRawCandidates.empty(), None)
                self.consume_window(window)
            self.closed = True
            self.complete = physical_end
            self.pending = GpuRawCandidates.empty()
        except BaseException as exc:
            self.failed = exc
            raise


class GrouperSetup:
    """Common geometry and decoder constructed on the selected producer device."""

    def __init__(self, plan, dcores, configuration, *, cuda_device_id):
        from .ArgmaxMetadata import ARGMAX_ENCODING, read_argmax_metadata
        self.plan = plan
        self.configuration = configuration
        self.dcores = read_argmax_metadata(
            dict(argmax_encoding=ARGMAX_ENCODING, dcores=dcores), ntrees=plan.ntrees,
            douts=[int(t.nt_ds) // int(t.nt_out) for t in plan.trees])
        self.geometries = tuple(PeakFinderGeometry.from_plan(
            plan, tree, dcore=self.dcores[tree],
            dm_reach=configuration.peakfinding.dm_reach,
            waist_bins=configuration.peakfinding.waist_bins,
        ) for tree in range(plan.ntrees))
        self.halo_columns = _effective_grouping_halo_columns(
            self.geometries, configuration.grouping.halo_size)
        self.decoder = GpuArgmaxDecoder(plan, cuda_device_id=cuda_device_id, dcores=self.dcores)
        self.grouping_geometry = GroupingGeometry.from_plan(plan)
        self.grouping_config = GroupingConfig(
            dm_tolerance_bins=configuration.grouping.dm_tolerance,
            time_padding_bins=configuration.grouping.time_tolerance)

    def processor(self, beam_ids, producer_start_chunk, consume_window, *, assume_steady_state=False):
        cfg = self.configuration
        return StreamingGrouper(
            self.geometries, self.decoder, self.grouping_geometry,
            beam_ids=beam_ids, producer_start_chunk=producer_start_chunk,
            threshold=cfg.peakfinding.snr_threshold, halo_size=cfg.grouping.halo_size,
            timeout_ms=cfg.execution.timeout_ms, timeout_policy=cfg.execution.timeout_policy,
            grouping_config=self.grouping_config, consume_window=consume_window,
            assume_steady_state=assume_steady_state,
            grouping_halo_columns_by_tree=self.halo_columns)


class CatalogRecorder:
    """Common host reporting boundary for shared processor windows."""

    def __init__(self, setup, *, config_yaml, plan_yaml, argmax_encoding, pipeline="offline"):
        self.setup = setup
        self.config_yaml = config_yaml
        self.plan_yaml = plan_yaml
        self.argmax_encoding = argmax_encoding
        self.pipeline = pipeline
        self.batches = []
        self.windows = []
        self.event_offset = 0
        self.candidate_offset = 0

    def consume(self, beam_ids, window):
        batch = catalog_batch_from_grouping_result(
            window.grouped, event_id_offset=self.event_offset,
            candidate_id_offset=self.candidate_offset, grouping_window_id=len(self.windows),
            grouping_timed_out=window.grouped.timed_out)
        self.batches.append(batch)
        self.windows.append(dict(
            grouping_window_id=len(self.windows), beam_ids=list(beam_ids),
            owner_source_chunk_index=window.owner_source_chunk_index,
            timed_out=window.grouped.timed_out, output_status=window.output_status))
        self.event_offset += len(window.grouped.events)
        self.candidate_offset += len(window.grouped.candidates)
        return batch

    def write(self, path, processors, *, complete):
        cfg = self.setup.configuration
        geometries = self.setup.geometries
        startup, coverage = [], []
        for processor in processors:
            coverage.extend(processor.coverage)
            for beam in processor.beam_ids:
                record = dict(beam_id=beam, status=processor.startup_status)
                if not processor.assumed:
                    record["producer_start_chunk_index"] = processor.producer_start_chunk
                startup.append(record)
        metadata = make_catalog_metadata(
            config_yaml=self.config_yaml, plan_yaml=self.plan_yaml,
            dcores=self.setup.dcores, argmax_encoding=self.argmax_encoding,
            snr_threshold=cfg.peakfinding.snr_threshold,
            dm_reach_by_tree=tuple(g.dm_radius for g in geometries),
            waist_bins_by_tree=tuple(g.waist_bins for g in geometries),
            time_radius_by_tree=tuple(g.time_radius for g in geometries),
            requested_time_radius_by_tree=tuple(g.requested_time_radius for g in geometries),
            halo_size=cfg.grouping.halo_size,
            effective_grouping_halo_columns_by_tree=self.setup.halo_columns,
            dm_tolerance=cfg.grouping.dm_tolerance,
            time_tolerance=cfg.grouping.time_tolerance,
            beam_batch_size=cfg.execution.beam_batch_size,
            timeout_ms=cfg.execution.timeout_ms, timeout_policy=cfg.execution.timeout_policy,
            grouping_windows=self.windows, startup_by_beam=startup, complete=complete,
            pipeline=self.pipeline)
        return write_trigger_catalog(path, self.batches, coverage=coverage, metadata=metadata)
