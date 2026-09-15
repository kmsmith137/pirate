"""Bounded offline peak finding and representative grouping.

The runner keeps one :class:`~pirate_frb.BowtiePeakfinding.StreamingPeakExtractor`
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
import cupy as cp

from .OfflineMapReader import OfflineMapReader
from .GpuArgmaxDecoder import GpuArgmaxDecoder
from .Clustering import (
    GpuDecodedCandidates,
    GpuEventTable,
    GpuClusteringResult,
    GpuMemberTable,
    ClusteringTolerances,
    ClusteringGeometry,
    cluster_candidates,
)
from .GrouperConfig import load_grouper_config
from .BowtiePeakfinding import (
    EdgeFlag,
    GpuRawCandidates,
    StreamingPeakExtractor,
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


# Compatibility exports for callers of the original offline helpers.
from .GrouperPipeline import (
    ClusteringWindow, StreamingGrouper, GrouperSetup, CatalogRecorder,
    _optional_nonnegative_integer, _effective_grouping_halo_columns,
    _raw_take, _raw_concatenate, _project_grouping_events,
    _owned_grouping_result, _split_emitted_chunks, _partition_current_halo,
    _group_streaming_window as _shared_group_streaming_window,
)


def _group_streaming_window(*args, **kwargs):
    # Keep the historical test seam; actual association remains shared.
    return _shared_group_streaming_window(
        *args, **kwargs, group_function=cluster_candidates
    )


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

    processor = StreamingGrouper(
        geometries, decoder, grouping_geometry,
        beam_ids=beam_batch.beam_ids,
        producer_start_chunk=beam_batch.producer_start_chunk,
        threshold=threshold, halo_size=halo_size, timeout_ms=timeout_ms,
        timeout_policy=timeout_policy, assume_steady_state=assume_steady_state,
        grouping_config=grouping_config, consume_window=consume_window,
        grouping_halo_columns_by_tree=grouping_halo_columns_by_tree,
        extractor_factory=StreamingPeakExtractor, group_function=cluster_candidates,
    )
    all_chunks = beam_batch.source_chunk_indices
    selected = all_chunks if max_chunks is None else all_chunks[:max_chunks]
    for source_chunk in selected:
        maps = loader.load_beam_chunk(beam_batch.beam_ids, source_chunk)
        processor.process_chunk(
            maps.snr_by_tree, maps.argmax_by_tree, source_chunk
        )
    complete = len(selected) == len(all_chunks)
    processor.finish(physical_end=complete)
    return processor.coverage, processor.startup_status, complete


def run_offline_grouper(
        acqdir, config_file, max_chunks=None, cuda_device_id=0, *,
        assume_steady_state=False, output=None, verbose=False):
    """Run the finalized bounded offline peakfinder/grouping pipeline.

    Parameters
    ----------
    acqdir : path-like
        Directory containing version-3 offline-dedisperser S/N-map ASDF files.
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
    configuration = load_grouper_config(config_file)
    max_chunks = _optional_nonnegative_integer(max_chunks, "max_chunks")

    peakfinding = configuration.peakfinding
    grouping = configuration.grouping
    execution = configuration.execution
    loader = OfflineMapReader(acqdir, cuda_device_id=cuda_device_id)
    grouping_config = ClusteringTolerances(
        dm_tolerance_bins=grouping.dm_tolerance,
        time_padding_bins=grouping.time_tolerance,
    )

    catalog_batches = []
    catalog_coverage = []
    startup_by_beam = []
    grouping_windows = []
    all_complete = True
    with cp.cuda.Device(cuda_device_id):
        setup = GrouperSetup(
            loader.plan, loader.dcores, configuration, cuda_device_id=cuda_device_id
        )
        geometries = setup.geometries
        effective_grouping_halo_columns_by_tree = setup.halo_columns
        decoder = setup.decoder
        grouping_geometry = setup.grouping_geometry
        grouping_config = setup.grouping_config
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
        dcores=loader.dcores,
        argmax_encoding=loader.argmax_encoding,
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


__all__ = ["ClusteringWindow", "GroupingWindow", "run_offline_grouper"]


# Import compatibility for callers of the offline runner.
GroupingWindow = ClusteringWindow
