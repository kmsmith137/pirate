"""Bounded offline peak finding and representative grouping.

The runner keeps one :class:`~pirate_frb.Peakfinders.OfflinePeakExtractor`
per ragged producer tree and finalizes candidates with one source-chunk of
latency.  A grouping window contains every candidate owned by chunk ``i`` and
only candidates in the effective left halo of chunk ``i+1``.  The configured
radius multiplier defines that map-coordinate association domain; DM/time
tolerances are applied only after this gate and never enlarge it.  A group belongs
to ``i`` when *any* of its members belongs to ``i``; ownership therefore does
not change when a louder representative comes from the next chunk.  Halo
members consumed by such a group are removed before the next window, while
future-only halo groups are reconsidered when their chunk becomes the owner.

This rule makes chunk ``i+2`` unable to change an event already finalized for
``i`` and bounds both map and candidate state.  The final owner is flushed only
at the physical acquisition end.  A ``max_chunks`` prefix is deliberately not
flushed because its right-hand context is unknown.

Grouping runs in one persistent GPU launch per streaming window and compatible
beam batch.  A cooperative timeout applies only to that launch.  ``discard``
publishes no rows and drops every candidate which entered the timed-out window;
``emit_partial`` publishes only fully committed owner groups and drops every
unprocessed candidate.  Timeout provenance is a window property, not a
peakfinder edge flag.
"""

from __future__ import annotations

from dataclasses import dataclass, fields
import operator

import numpy as np

from .FrbOfflineGrouper import FrbOfflineGrouper
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
from .OfflineGrouperConfig import load_offline_grouper_config
from .Peakfinders import (
    EdgeFlag,
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
from .utils import atomic_print


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
        timeout_policy):
    """Group one owner plus next-chunk halo and apply its timeout policy.

    Returns ``(window, retained_halo)``.  Retained halo rows are candidates from
    fully processed future-only groups; they remain ordinary bounded state for
    their own future window.  Unprocessed halo rows are never carried after a
    timeout.
    """

    window_raw = _raw_concatenate((owner_raw, halo_raw))
    decoded = decoder.decode(window_raw)
    grouped = group_candidates(
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


def _event_host_columns(cp, events):
    """Copy the final terminal event projection to contiguous host columns."""

    names = (
        "beam_id", "tree", "snr", "dm", "toa_sample_abs",
        "width_samp", "width_ms", "freq_lo_MHz", "freq_hi_MHz",
        "argmax_token", "source_chunk_index", "idm", "itime",
        "member_count", "edge_flags",
    )
    return {name: cp.asnumpy(getattr(events, name)) for name in names}


def _print_events(
        cp, grouped, event_offset, grouping_window_id, time_sample_ms, *,
        verbose=False):
    """Print representative events with explicit grouping-window provenance."""

    columns = _event_host_columns(cp, grouped.events)
    seconds_per_sample = float(time_sample_ms) * 1.0e-3
    timeout_text = str(grouped.timed_out).lower()
    for local_event in range(len(grouped.events)):
        event_id = event_offset + local_event
        toa_sec = (
            float(columns["toa_sample_abs"][local_event])
            * seconds_per_sample
        )
        startup_incomplete = bool(
            int(columns["edge_flags"][local_event])
            & int(EdgeFlag.STARTUP_INCOMPLETE)
        )
        atomic_print(
            f"event={event_id} "
            f"grouping_window={grouping_window_id} "
            f"grouping_timed_out={timeout_text} "
            f"beam={int(columns['beam_id'][local_event])} "
            f"snr={float(columns['snr'][local_event]):.6g} "
            f"dm={float(columns['dm'][local_event]):.9g} pc/cm3 "
            f"toa={toa_sec:.12g} s "
            f"width={float(columns['width_samp'][local_event]):.6g} samples/"
            f"{float(columns['width_ms'][local_event]):.6g} ms "
            f"freq=[{float(columns['freq_lo_MHz'][local_event]):.6g},"
            f"{float(columns['freq_hi_MHz'][local_event]):.6g}] MHz "
            f"tree={int(columns['tree'][local_event])} "
            f"token=0x{int(columns['argmax_token'][local_event]):08x} "
            f"chunk={int(columns['source_chunk_index'][local_event])} "
            f"idm={int(columns['idm'][local_event])} "
            f"itime={int(columns['itime'][local_event])} "
            f"startup_incomplete={str(startup_incomplete).lower()} "
            f"members={int(columns['member_count'][local_event])}"
        )

    if verbose and len(grouped.members):
        links = {
            "event_id": cp.asnumpy(grouped.members.event_id),
            "candidate_index": cp.asnumpy(grouped.members.candidate_index),
            "is_representative": cp.asnumpy(
                grouped.members.is_representative
            ),
        }
        candidates = grouped.candidates
        rows = zip(
            links["event_id"], links["candidate_index"],
            links["is_representative"],
        )
        for member_id, (local_event, candidate_index, representative) in enumerate(rows):
            i = int(candidate_index)
            atomic_print(
                f"  member={member_id} "
                f"event={event_offset + int(local_event)} "
                f"grouping_window={grouping_window_id} "
                f"grouping_timed_out={timeout_text} "
                f"candidate={i} representative={bool(representative)} "
                f"tree={int(candidates.tree[i].item())} "
                f"snr={float(candidates.snr[i].item()):.6g} "
                f"chunk={int(candidates.source_chunk_index[i].item())} "
                f"idm={int(candidates.idm[i].item())} "
                f"itime={int(candidates.itime[i].item())}"
            )


def _extract_beam_batch(
        loader, beam_batch, geometries, decoder, grouping_geometry, *,
        threshold, halo_size, timeout_ms, timeout_policy, max_chunks,
        assume_steady_state, grouping_config, consume_window,
        grouping_halo_columns_by_tree=None):
    """Extract and finalize bounded owner/next-halo windows for one beam batch.

    ``consume_window`` is called synchronously for each finalized window before
    extraction advances.  Returns ``(coverage, startup_status, complete)``.
    There is at most one unresolved chunk and one completed window of compact
    GPU candidate state at any point; map state remains bounded inside each
    tree extractor.  Tests which need a history may explicitly pass
    ``list.append`` as the consumer, but production never materializes one.
    """

    import cupy as cp

    if not callable(consume_window):
        raise TypeError("consume_window must be callable")

    producer_start = beam_batch.producer_start_chunk
    assumed = producer_start is None
    if assumed and not assume_steady_state:
        raise ValueError(
            "producer-start provenance is missing for beams "
            f"{beam_batch.beam_ids}; pass assume_steady_state=True explicitly"
        )

    if grouping_halo_columns_by_tree is None:
        halo_columns = _effective_grouping_halo_columns(
            geometries, halo_size
        )
    else:
        halo_columns = tuple(grouping_halo_columns_by_tree)
        if len(halo_columns) != len(geometries):
            raise ValueError("grouping halo has the wrong tree count")
        for tree, (columns, geometry) in enumerate(zip(
                halo_columns, geometries)):
            if (isinstance(columns, (bool, np.bool_))
                    or not isinstance(columns, (int, np.integer))
                    or not 0 <= int(columns) <= geometry.ntime):
                raise ValueError(
                    f"grouping halo for tree {tree} is outside its map"
                )
        halo_columns = tuple(int(value) for value in halo_columns)
    # The extractor independently retains ``halo_size * h`` map columns for
    # peak competition.  This smaller/equal prefix gates only cross-window
    # candidate association: it is the intersection of that configured map
    # halo with centres already resolved after one following chunk.  Some real
    # ragged trees have 2*h > ntime-h, so admitting every retained context
    # column as a candidate would require looking into i+2.
    halo_columns_gpu = cp.asarray(halo_columns, dtype=cp.int64)

    extractors = tuple(
        OfflinePeakExtractor(
            geometry,
            threshold=threshold,
            beam_ids=beam_batch.beam_ids,
            assume_steady_state=assumed,
            halo_size=halo_size,
        )
        for geometry in geometries
    )
    all_chunks = beam_batch.source_chunk_indices
    selected_chunks = (
        all_chunks if max_chunks is None else all_chunks[:max_chunks]
    )

    pending = GpuRawCandidates.empty()
    previous_chunk = None
    for source_chunk in selected_chunks:
        maps = loader.load_beam_chunk(beam_batch.beam_ids, source_chunk)
        emitted_parts = []
        for tree, extractor in enumerate(extractors):
            startup_mask = (
                None if assumed else make_startup_valid_mask(
                    geometries[tree], source_chunk, producer_start
                )
            )
            emitted_parts.append(extractor.process_chunk(
                maps.snr_by_tree[tree],
                maps.argmax_by_tree[tree],
                source_chunk,
                startup_valid_mask=startup_mask,
            ))
        del maps
        emitted = _raw_concatenate(emitted_parts)

        if previous_chunk is None:
            if len(emitted) and not bool(cp.all(
                    emitted.source_chunk_index == source_chunk).item()):
                raise RuntimeError("first peak emission has invalid ownership")
            pending = emitted
            previous_chunk = source_chunk
            continue

        owner_tail, current_resolved = _split_emitted_chunks(
            cp, emitted, previous_chunk, source_chunk
        )
        owner_raw = _raw_concatenate((pending, owner_tail))
        halo_raw, outside_halo = _partition_current_halo(
            cp, current_resolved, halo_columns_gpu
        )
        window, retained_halo = _group_streaming_window(
            cp,
            owner_raw,
            halo_raw,
            previous_chunk,
            source_chunk,
            decoder,
            grouping_geometry,
            grouping_config,
            timeout_ms=timeout_ms,
            timeout_policy=timeout_policy,
        )
        consume_window(window)
        pending = _raw_concatenate((outside_halo, retained_halo))
        previous_chunk = source_chunk

    complete = len(selected_chunks) == len(all_chunks)
    if complete and selected_chunks:
        final_tail = _raw_concatenate(
            extractor.flush() for extractor in extractors
        )
        if len(final_tail) and not bool(cp.all(
                final_tail.source_chunk_index == previous_chunk).item()):
            raise RuntimeError("final peak emission has invalid ownership")
        owner_raw = _raw_concatenate((pending, final_tail))
        window, _ = _group_streaming_window(
            cp,
            owner_raw,
            GpuRawCandidates.empty(),
            previous_chunk,
            None,
            decoder,
            grouping_geometry,
            grouping_config,
            timeout_ms=timeout_ms,
            timeout_policy=timeout_policy,
        )
        consume_window(window)

    coverage = tuple(
        (beam_id, source_chunk)
        for beam_id in beam_batch.beam_ids
        for source_chunk in selected_chunks
    )
    startup_status = "assumed" if assumed else "authoritative"
    return coverage, startup_status, complete


def run_offline_grouper(
        acqdir, config_file, max_chunks=None, cuda_device_id=0, *,
        assume_steady_state=False, output=None, verbose=False):
    """Run the finalized bounded offline peakfinder/grouping pipeline.

    Parameters
    ----------
    acqdir : path-like
        Directory containing version-2 offline-dedisperser S/N-map ASDF files.
    config_file : path-like
        Required strict YAML file with exactly ``peakfinding``, ``grouping``,
        and ``execution`` sections.  See ``configs/offline_grouper/example.yml``.
    max_chunks : int or None, optional
        Process only a leading prefix per compatible beam stream.  Its final
        unresolved owner is not flushed or falsely treated as a physical edge.
    cuda_device_id : int, optional
        CUDA device for maps, peak extraction, decoding, and grouping.
    assume_steady_state : bool, keyword-only
        Explicitly permit missing producer-start provenance without adding a
        startup-incomplete label.
    output : path-like or None, keyword-only
        Optional atomically written, reopen-validated ASDF trigger catalog.
    verbose : bool, keyword-only
        Print member diagnostics in addition to representative events.

    Returns
    -------
    str or None
        Absolute catalog path, or ``None`` when no catalog was requested.

    Notes
    -----
    The cooperative timeout covers only each persistent association kernel.
    Compilation, sorting/layout, peakfinding, decoding, I/O, result compaction,
    catalog writing, and terminal printing are outside the deadline.  Checks
    occur at bounded GPU work tiles; a timeout can overshoot by one predicate
    tile plus one completed event-commit pass, but no half event is published.
    """

    # Validate before constructing the acquisition loader or touching CUDA
    # state.  Peakfinder/grouping modules import CuPy at module import time.
    configuration = load_offline_grouper_config(config_file)
    max_chunks = _optional_nonnegative_integer(max_chunks, "max_chunks")

    import cupy as cp

    peakfinding = configuration.peakfinding
    grouping = configuration.grouping
    execution = configuration.execution
    loader = FrbOfflineGrouper(acqdir, cuda_device_id=cuda_device_id)
    grouping_config = GroupingConfig(
        dm_tolerance_bins=grouping.dm_tolerance,
        time_padding_bins=grouping.time_tolerance,
    )

    catalog_batches = []
    catalog_coverage = []
    startup_by_beam = []
    grouping_windows = []
    all_complete = True
    with cp.cuda.Device(cuda_device_id):
        geometries = tuple(
            PeakFinderGeometry.from_plan(
                loader.plan,
                tree,
                dm_reach=peakfinding.dm_reach,
                waist_bins=peakfinding.waist_bins,
            )
            for tree in range(loader.ntrees)
        )
        effective_grouping_halo_columns_by_tree = (
            _effective_grouping_halo_columns(
                geometries, grouping.halo_size
            )
        )
        decoder = GpuArgmaxDecoder(
            loader.plan, cuda_device_id=cuda_device_id
        )
        grouping_geometry = GroupingGeometry.from_plan(loader.plan)
        event_offset = 0
        candidate_offset = 0
        window_id = 0

        for beam_batch in loader.iter_beam_batches(
                execution.beam_batch_size):
            batch_window_count = 0

            def consume_window(window):
                nonlocal event_offset, candidate_offset, window_id
                nonlocal batch_window_count
                grouped = window.grouped
                _print_events(
                    cp,
                    grouped,
                    event_offset,
                    window_id,
                    loader.plan.config.time_sample_ms,
                    verbose=bool(verbose),
                )
                if output is not None:
                    catalog_batches.append(
                        catalog_batch_from_grouping_result(
                            grouped,
                            event_id_offset=event_offset,
                            candidate_id_offset=candidate_offset,
                            grouping_window_id=window_id,
                            grouping_timed_out=grouped.timed_out,
                        )
                    )
                grouping_windows.append({
                    "grouping_window_id": window_id,
                    "beam_ids": list(beam_batch.beam_ids),
                    "owner_source_chunk_index": (
                        window.owner_source_chunk_index
                    ),
                    "timed_out": grouped.timed_out,
                    "output_status": window.output_status,
                })
                atomic_print(
                    f"summary grouping_window={window_id} "
                    f"beams={beam_batch.beam_ids} "
                    f"owner_chunk={window.owner_source_chunk_index} "
                    f"events={len(grouped.events)} "
                    f"grouping_timed_out={str(grouped.timed_out).lower()} "
                    f"output_status={window.output_status}"
                )
                event_offset += len(grouped.events)
                candidate_offset += len(grouped.candidates)
                window_id += 1
                batch_window_count += 1

            coverage, startup_status, complete = _extract_beam_batch(
                loader,
                beam_batch,
                geometries,
                decoder,
                grouping_geometry,
                threshold=peakfinding.snr_threshold,
                halo_size=grouping.halo_size,
                timeout_ms=execution.timeout_ms,
                timeout_policy=execution.timeout_policy,
                max_chunks=max_chunks,
                assume_steady_state=bool(assume_steady_state),
                grouping_config=grouping_config,
                consume_window=consume_window,
                grouping_halo_columns_by_tree=(
                    effective_grouping_halo_columns_by_tree
                ),
            )

            catalog_coverage.extend(coverage)
            for beam_id in beam_batch.beam_ids:
                startup = {"beam_id": beam_id, "status": startup_status}
                if startup_status == "authoritative":
                    startup["producer_start_chunk_index"] = (
                        beam_batch.producer_start_chunk
                    )
                startup_by_beam.append(startup)
            all_complete = all_complete and complete
            atomic_print(
                f"summary beams={beam_batch.beam_ids} "
                f"chunks={len(coverage) // len(beam_batch.beam_ids)} "
                f"windows={batch_window_count} startup={startup_status} "
                f"complete={str(complete).lower()}"
            )

    if output is None:
        return None

    metadata = make_catalog_metadata(
        config_yaml=loader.config_yaml,
        plan_yaml=loader.plan_yaml,
        snr_threshold=peakfinding.snr_threshold,
        dm_reach_by_tree=tuple(
            geometry.dm_radius for geometry in geometries
        ),
        waist_bins_by_tree=tuple(
            geometry.waist_bins for geometry in geometries
        ),
        time_radius_by_tree=tuple(
            geometry.time_radius for geometry in geometries
        ),
        requested_time_radius_by_tree=tuple(
            geometry.requested_time_radius for geometry in geometries
        ),
        halo_size=grouping.halo_size,
        effective_grouping_halo_columns_by_tree=(
            effective_grouping_halo_columns_by_tree
        ),
        dm_tolerance=grouping.dm_tolerance,
        time_tolerance=grouping.time_tolerance,
        beam_batch_size=execution.beam_batch_size,
        timeout_ms=execution.timeout_ms,
        timeout_policy=execution.timeout_policy,
        grouping_windows=grouping_windows,
        startup_by_beam=startup_by_beam,
        complete=all_complete,
    )
    written = write_trigger_catalog(
        output,
        catalog_batches,
        coverage=catalog_coverage,
        metadata=metadata,
    )
    atomic_print(f"catalog={written}")
    return written


__all__ = ["GroupingWindow", "run_offline_grouper"]
