"""Versioned ASDF output for the offline candidate pipeline.

The catalog contains two column-oriented tables.  `events` has one row per
group and reports the physical measurements of that group's representative
(the loudest seed candidate).  `members` has one row per decoded candidate
and preserves both its own measurements and the event to which it was
assigned.  Event fields are a projection of the representative; in
particular, event `edge_flags` is *not* the bitwise union of member flags.
This matters for quality bits such as `EdgeFlag.STARTUP_INCOMPLETE`: the
event bit describes its representative, while the member table retains the
quality of every contributing candidate.  That bit does not modify catalogued
S/N or width; it warns that those measured values may be biased by
pre-acquisition zero padding.

This module is the intentional final device-to-host boundary.  Event columns
and candidate rows selected by the member-link table are copied from CuPy to
contiguous NumPy arrays once per finalized streaming window and compatible
beam batch.  The much larger S/N and argmax maps are neither accepted nor
transferred by this API.  Batches are later concatenated on the host and
written as a small, self-describing ASDF tree with exact producer YAML,
processing parameters, processed coverage, and startup provenance.

Coordinates deliberately retain their producer meanings. `source_chunk_index`
is the producer's absolute chunk number; `idm` and `itime` are coordinates
within that tree's map for that chunk; and `toa_sample_abs` is a potentially
fractional full-resolution sample coordinate referenced to the lowest
frequency of the complete observing band.  Catalog IDs are separate,
zero-based global identifiers assigned while streaming grouping windows are
appended.  Both row tables carry ``grouping_window_id`` and
``grouping_timed_out``.  The former joins rows to the per-window provenance in
``metadata.grouping_windows``; the latter makes a partial-timeout batch
immediately recognizable without overloading the peakfinder's ``edge_flags``.
Processing metadata also names the bounded association domain and records each
tree's resolved intersection with its configured next-chunk map halo.
"""

from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from math import isfinite
import operator
import os
import tempfile

import numpy as np


CATALOG_FORMAT = "pirate_frb.offline_candidate_catalog"
CATALOG_VERSION = 2


def _dtypes(**groups):
    """Expand compact dtype groups into a column-name-to-NumPy-dtype map.

    Parameters
    ----------
    **groups
        Keyword names are NumPy dtype specifications and values are iterables
        of catalog column names. A column must occur in only one group; the
        module-level declarations below satisfy that invariant.

    Returns
    -------
    dict
        Insertion-ordered mapping used both to coerce outgoing columns and to
        require an exact schema when validating reopened ASDF tables.
    """

    return {name: np.dtype(dtype) for dtype, names in groups.items()
            for name in names}


# Events contain representative-derived values only. IDs and counts remain
# integers; physical quantities are promoted to float64 so catalog precision
# does not depend on the storage dtype of the producer's S/N maps.
EVENT_DTYPES = _dtypes(
    int64=("event_id", "representative_candidate_id", "source_chunk_index",
           "grouping_window_id"),
    int32=("member_count", "beam_id", "primary_tree_index", "tree", "idm",
           "itime"),
    float64=("snr", "dm", "toa_sample_abs", "width_samp", "width_ms",
             "freq_lo_MHz", "freq_hi_MHz"),
    uint32=("argmax_token",),
    uint8=("edge_flags",),
    bool=("grouping_timed_out",),
)
# Members preserve every decoded candidate plus its native DM/time resolution.
# `edge_flags` stays uint8 end to end: individual IntFlag bits, including
# STARTUP_INCOMPLETE, are copied without reinterpretation or aggregation.
MEMBER_DTYPES = _dtypes(
    int64=("candidate_id", "event_id", "source_chunk_index",
           "grouping_window_id"),
    int32=("beam_id", "primary_tree_index", "tree", "idm", "itime"),
    float64=("snr", "dm", "toa_sample_abs", "width_samp", "width_ms",
             "freq_lo_MHz", "freq_hi_MHz", "dm_step", "time_step_samples"),
    uint32=("argmax_token",),
    uint8=("edge_flags",),
    bool=("grouping_timed_out",),
)
# Coverage is independent of detections: an empty processed chunk still gets a
# row so absence of members can be distinguished from absence of processing.
COVERAGE_DTYPES = _dtypes(int32=("beam_id",),
                          int64=("source_chunk_index",))


@dataclass(frozen=True)
class CatalogBatch:
    """Host event and member columns from one GPU grouping result.

    `events` follows :data:`EVENT_DTYPES` and `members` follows
    :data:`MEMBER_DTYPES`; every value is a contiguous one-dimensional NumPy
    array. IDs already include the caller-supplied offsets and are therefore
    global across all batches destined for one file. The dataclass is frozen
    to make the hand-off from GPU processing to host catalog assembly
    explicit, although the arrays themselves remain mutable NumPy objects.
    """

    events: Mapping[str, np.ndarray]
    members: Mapping[str, np.ndarray]


def _index(value, name):
    """Return one non-negative Python index without accepting booleans.

    `operator.index` intentionally accepts integer scalar types such as
    `np.int64` but rejects lossy conversions from floats and strings. IDs,
    offsets, beam numbers, and chunk numbers use this helper before they are
    stored in fixed-width catalog columns.
    """

    if isinstance(value, (bool, np.bool_)):
        raise TypeError(f"{name} must be an integer")
    try:
        value = operator.index(value)
    except TypeError as exc:
        raise TypeError(f"{name} must be an integer") from exc
    if value < 0:
        raise ValueError(f"{name} must be non-negative")
    return value


def _host(value, dtype, name):
    """Materialize one final one-dimensional column as contiguous NumPy data.

    A CuPy-like value is transferred by its `get()` method; an existing host
    value is passed through `np.asarray`. The requested catalog dtype is
    applied at this intentional reporting boundary. Rank is checked before
    returning because ASDF tables are column-oriented and all columns must
    share one row dimension.
    """

    getter = getattr(value, "get", None)
    array = np.asarray(getter() if callable(getter) else value)
    if array.ndim != 1:
        raise ValueError(f"{name} must be one-dimensional")
    return np.ascontiguousarray(array, dtype=dtype)

def catalog_batch_from_grouping_result(
        grouped, *, event_id_offset, candidate_id_offset,
        grouping_window_id=0, grouping_timed_out=None):
    """Transfer one final grouping result to host and assign global IDs.

    Parameters
    ----------
    grouped
        A GPU grouping result. `grouped.events` contains representative
        projections, `grouped.members` contains event/candidate links, and
        `grouped.candidates` owns all raw and decoded candidate columns.
    event_id_offset, candidate_id_offset : int
        Non-negative numbers of events and candidates in preceding beam
        batches. Adding them converts batch-local indices into file-global,
        zero-based identifiers.
    grouping_window_id : int, optional
        Global streaming-window identifier attached to every emitted row.
    grouping_timed_out : bool or None, optional
        Whether this is partial output from a timed-out grouping call. ``None``
        reads ``grouped.timed_out`` when available and otherwise uses false.

    Returns
    -------
    CatalogBatch
        Contiguous host columns with exact :data:`EVENT_DTYPES` and
        :data:`MEMBER_DTYPES`. Member rows follow the grouper's link order,
        which is normally event-major; `candidate_id` retains the original
        candidate identity and therefore need not equal the member row number.

    Notes
    -----
    The gathers `candidate_field[order]` happen on the GPU. Consequently
    only final event columns and the member-ordered candidate columns cross to
    the CPU; no normal-path transfer of pre-group candidate columns is added.
    Event physical fields and event `edge_flags` have already been selected
    from the representative. Per-candidate flags remain in `members`.
    """

    event_offset = _index(event_id_offset, "event_id_offset")
    candidate_offset = _index(candidate_id_offset, "candidate_id_offset")
    window_id = _index(grouping_window_id, "grouping_window_id")
    if grouping_timed_out is None:
        grouping_timed_out = bool(getattr(grouped, "timed_out", False))
    if not isinstance(grouping_timed_out, (bool, np.bool_)):
        raise TypeError("grouping_timed_out must be bool")
    gpu_events = grouped.events
    links = grouped.members
    candidates = grouped.candidates
    nevent = int(gpu_events.event_id.size)
    order = links.candidate_index

    # The grouper numbers events and candidates locally. Do not serialize
    # those local event IDs directly: batches must form one contiguous global
    # namespace after concatenation.
    events = {
        "event_id": np.arange(event_offset, event_offset + nevent,
                              dtype=np.int64),
        "representative_candidate_id": _host(
            gpu_events.representative_candidate_index, np.int64,
            "events.representative_candidate_index",
        ) + candidate_offset,
        "grouping_window_id": np.full(
            nevent, window_id, dtype=np.int64
        ),
        "grouping_timed_out": np.full(
            nevent, grouping_timed_out, dtype=np.bool_
        ),
    }
    for name, dtype in EVENT_DTYPES.items():
        if name not in events:
            events[name] = _host(
                getattr(gpu_events, name), dtype, f"events.{name}"
            )

    # `order` is both the member-to-candidate link and the GPU gather index.
    # Gathering before calling _host avoids a candidate-sized host staging
    # table and preserves every member's own quality flags.
    members = {
        "candidate_id": _host(order, np.int64, "members.candidate_index")
        + candidate_offset,
        "event_id": _host(links.event_id, np.int64, "members.event_id")
        + event_offset,
        "grouping_window_id": np.full(
            int(order.size), window_id, dtype=np.int64
        ),
        "grouping_timed_out": np.full(
            int(order.size), grouping_timed_out, dtype=np.bool_
        ),
    }
    for name, dtype in MEMBER_DTYPES.items():
        if name not in members:
            members[name] = _host(
                getattr(candidates, name)[order], dtype, f"members.{name}"
            )
    return CatalogBatch(events=events, members=members)

def make_catalog_metadata(
        *, config_yaml, plan_yaml, snr_threshold,
        dm_reach_by_tree, waist_bins_by_tree, time_radius_by_tree,
        requested_time_radius_by_tree, halo_size,
        effective_grouping_halo_columns_by_tree,
        dm_tolerance, time_tolerance, beam_batch_size,
        timeout_ms, timeout_policy, grouping_windows,
        startup_by_beam, complete=True):
    """Build scientific, processing, and producer-provenance metadata.

    Parameters
    ----------
    config_yaml, plan_yaml : str
        Exact non-empty YAML text embedded by the S/N-map producer. Together
        these reconstruct the frequency/time geometry and dedispersion trees
        needed to interpret DM, TOA, and argmax tokens.
    snr_threshold : float
        Finite S/N threshold applied to map cells.
    dm_reach_by_tree, waist_bins_by_tree, time_radius_by_tree,
    requested_time_radius_by_tree : sequence of int
        Non-negative peakfinder geometry in each tree's native map bins. The
        time radii distinguish the effective CHIME-horizon crop from the
        uncropped physical Bowtie request. All sequences have equal length.
    halo_size : int
        Requested per-tree temporal-radius multiplier used for seam retention.
    effective_grouping_halo_columns_by_tree : sequence of int
        Actual next-chunk ``itime`` selection limit for each ragged tree.  It is
        the intersection of the configured radius-multiple prefix with the
        candidate centres peak-resolved after one following chunk.
    dm_tolerance, time_tolerance : float
        Finite, non-negative dimensionless grouping tolerances. Their physical
        scale is recovered from the producer plan.
    beam_batch_size, timeout_ms : int
        Positive compatible-beam limit and non-negative cooperative grouping
        timeout in milliseconds. Zero disables the timeout.
    timeout_policy : {"discard", "emit_partial"}
        Action applied when a grouping window reaches its deadline.
    grouping_windows : iterable of mappings
        One record per attempted streaming window. Records bind a contiguous
        global window ID to its beams, owner chunk, timeout state, and output
        status (``complete``, ``discarded``, or ``partial``).
    startup_by_beam : iterable of mappings
        Exactly one provenance record is expected for each processed beam.
        `status="authoritative"` requires the producer's absolute start
        chunk; `status="assumed"` records the user's explicit steady-state
        assumption and must not claim a producer start.
    complete : bool, optional
        Whether all discovered chunks were processed and each extractor's
        physical right edge was flushed. A run truncated by `max_chunks` is
        recorded as incomplete.

    Returns
    -------
    dict
        ASDF-compatible metadata. Physical units are recorded explicitly;
        `toa_sample_abs` is in full-resolution samples at the complete-band
        low-frequency reference, not seconds or chunk-relative samples.

    Raises
    ------
    TypeError, ValueError
        If strings, numeric tolerances, per-tree geometry, or per-beam startup
        provenance cannot be represented unambiguously.
    """

    if not all(isinstance(x, str) and x for x in (config_yaml, plan_yaml)):
        raise ValueError("config_yaml and plan_yaml must be non-empty strings")
    numeric = snr_threshold, dm_tolerance, time_tolerance
    if not all(not isinstance(x, bool) and isinstance(x, (int, float))
               and isfinite(float(x)) for x in numeric):
        raise ValueError("processing thresholds must be finite numbers")
    if dm_tolerance < 0 or time_tolerance < 0:
        raise ValueError("grouping tolerances must be non-negative")

    halo = _index(halo_size, "halo_size")
    beam_batch = _index(beam_batch_size, "beam_batch_size")
    timeout = _index(timeout_ms, "timeout_ms")
    if halo < 2:
        raise ValueError("halo_size must be at least 2")
    if beam_batch < 1:
        raise ValueError("beam_batch_size must be positive")
    if timeout_policy not in ("discard", "emit_partial"):
        raise ValueError("timeout_policy must be discard or emit_partial")

    reaches = [_index(x, "dm_reach_by_tree") for x in dm_reach_by_tree]
    waists = [_index(x, "waist_bins_by_tree") for x in waist_bins_by_tree]
    radii = [_index(x, "time_radius_by_tree") for x in time_radius_by_tree]
    requested_radii = [
        _index(x, "requested_time_radius_by_tree")
        for x in requested_time_radius_by_tree
    ]
    grouping_halos = [
        _index(x, "effective_grouping_halo_columns_by_tree")
        for x in effective_grouping_halo_columns_by_tree
    ]
    if (not reaches or any(len(values) != len(reaches) for values in (
            waists, radii, requested_radii, grouping_halos))):
        raise ValueError("peakfinder geometry must have equal nonzero lengths")
    if any(value < 0 for values in (
            reaches, waists, radii, requested_radii) for value in values):
        raise ValueError("peakfinder geometry must be non-negative")
    if any(effective > requested for effective, requested in zip(
            radii, requested_radii)):
        raise ValueError("effective time radius exceeds its requested radius")
    if any(effective > halo * radius for effective, radius in zip(
            grouping_halos, radii)):
        raise ValueError("effective grouping halo exceeds its configured halo")
    startup = []
    for row in startup_by_beam:
        if not isinstance(row, Mapping):
            raise TypeError("startup rows must be mappings")
        status = row.get("status")
        if status not in ("authoritative", "assumed"):
            raise ValueError("startup status must be authoritative or assumed")
        item = {"beam_id": _index(row.get("beam_id"), "startup beam_id"),
                "status": status}
        start = row.get("producer_start_chunk_index")
        if status == "authoritative":
            item["producer_start_chunk_index"] = _index(
                start, "producer_start_chunk_index"
            )
        elif start is not None:
            raise ValueError("assumed startup cannot claim a producer start")
        startup.append(item)
    if len({row["beam_id"] for row in startup}) != len(startup):
        raise ValueError("startup metadata contains a duplicate beam")

    windows = []
    for row in grouping_windows:
        if not isinstance(row, Mapping):
            raise TypeError("grouping window rows must be mappings")
        window_id = _index(
            row.get("grouping_window_id"), "grouping_window_id"
        )
        owner = _index(
            row.get("owner_source_chunk_index"),
            "owner_source_chunk_index",
        )
        beams = row.get("beam_ids")
        if (not isinstance(beams, Sequence)
                or isinstance(beams, (str, bytes))):
            raise TypeError("grouping window beam_ids must be a sequence")
        beams = [_index(x, "grouping window beam_id") for x in beams]
        if not beams or len(set(beams)) != len(beams):
            raise ValueError(
                "grouping window beam_ids must be nonempty and unique"
            )
        timed_out = row.get("timed_out")
        if not isinstance(timed_out, (bool, np.bool_)):
            raise TypeError("grouping window timed_out must be bool")
        output_status = row.get("output_status")
        if output_status not in ("complete", "discarded", "partial"):
            raise ValueError("invalid grouping window output_status")
        if bool(timed_out) != (output_status != "complete"):
            raise ValueError(
                "grouping window timeout and output status disagree"
            )
        windows.append({
            "grouping_window_id": window_id,
            "beam_ids": beams,
            "owner_source_chunk_index": owner,
            "timed_out": bool(timed_out),
            "output_status": output_status,
        })
    if [row["grouping_window_id"] for row in windows] != list(
            range(len(windows))):
        raise ValueError("grouping window IDs must be contiguous from zero")

    # Keep the exact producer text rather than a parsed/re-emitted equivalent:
    # the decoder and loader reconstruct the plan from these authoritative
    # strings, and textual retention also makes provenance auditing possible.
    return {
        "pipeline": "offline",
        "processing": {
            "complete": bool(complete),
            "peakfinder": "full_band",
            "snr_threshold": float(snr_threshold),
            "dm_reach_by_tree": reaches,
            "waist_bins_by_tree": waists,
            "time_radius_by_tree": radii,
            "requested_time_radius_by_tree": requested_radii,
            "grouping_method": "persistent_partition_representative",
            "grouping_association_domain": (
                "owner_plus_resolved_next_map_halo"
            ),
            "halo_size": halo,
            "effective_grouping_halo_columns_by_tree": grouping_halos,
            "dm_tolerance": float(dm_tolerance),
            "time_tolerance": float(time_tolerance),
            "beam_batch_size": beam_batch,
            "timeout_ms": timeout,
            "timeout_policy": timeout_policy,
        },
        "producer": {"config_yaml": config_yaml, "plan_yaml": plan_yaml},
        "startup_by_beam": startup,
        "grouping_windows": windows,
        "units": {
            "dm": "pc cm^-3",
            "toa_sample_abs": "full-resolution samples at complete-band low frequency",
            "width_samp": "full-resolution samples",
            "width_ms": "ms", "freq_lo_MHz": "MHz", "freq_hi_MHz": "MHz",
        },
    }

def _table(columns, dtypes, name):
    """Normalize and shape-check one host or reopened ASDF table.

    `columns` must be a mapping with exactly the schema keys in `dtypes`;
    missing and unknown fields are both rejected. Values are materialized as
    contiguous one-dimensional NumPy arrays of the declared dtype and all
    columns must have equal length. This same path validates freshly built and
    eagerly reopened ASDF trees, preventing serialization from changing the
    logical table shape or dtype.
    """

    if not isinstance(columns, Mapping) or set(columns) != set(dtypes):
        raise ValueError(f"{name} columns disagree with catalog schema")
    result = {
        key: _host(columns[key], dtype, f"{name}.{key}")
        for key, dtype in dtypes.items()
    }
    if len({len(x) for x in result.values()}) > 1:
        raise ValueError(f"{name} columns have unequal lengths")
    return result


def _validate_tables(events, members):
    """Validate identifiers, links, counts, and representative projections.

    Event IDs must equal `0..nevent-1` in row order. Candidate IDs may occur
    in event-major member order, but must be a permutation of
    `0..nmember-1`. Every member must reference an event, every event count
    must equal its number of member links, and every representative ID must
    identify a member of that event. Finally, every event column also present
    in the member schema is required to equal the representative member's
    value. This last invariant enforces representative-derived S/N, physical
    coordinates, token, and `edge_flags` rather than an event-level merge.
    """

    events = _table(events, EVENT_DTYPES, "events")
    members = _table(members, MEMBER_DTYPES, "members")
    nevent, nmember = len(events["event_id"]), len(members["event_id"])
    if not np.array_equal(events["event_id"], np.arange(nevent)):
        raise ValueError("event IDs must be contiguous from zero")
    if not np.array_equal(np.sort(members["candidate_id"]), np.arange(nmember)):
        raise ValueError("candidate IDs must be unique and contiguous from zero")
    event_id = members["event_id"]
    if nmember and ((event_id < 0).any() or (event_id >= nevent).any()):
        raise ValueError("member references a nonexistent event")
    if not np.array_equal(
            np.bincount(event_id, minlength=nevent), events["member_count"]):
        raise ValueError("event member_count disagrees with member links")
    representative = events["representative_candidate_id"]
    if nevent and ((representative < 0).any()
                   or (representative >= nmember).any()):
        raise ValueError("event representative candidate ID is invalid")
    # Candidate IDs are identities, not necessarily row positions. Invert the
    # permutation once so representatives can be projected into member rows.
    candidate_row = np.empty(nmember, dtype=np.int64)
    candidate_row[members["candidate_id"]] = np.arange(nmember)
    rep_row = candidate_row[representative]
    if not np.array_equal(
            members["event_id"][rep_row], events["event_id"]):
        raise ValueError("representative links are inconsistent")
    for name in EVENT_DTYPES:
        if name in MEMBER_DTYPES and name not in ("event_id",):
            if not np.array_equal(events[name], members[name][rep_row]):
                raise ValueError(f"event {name} is not its representative value")


def _coverage_table(coverage):
    """Return sorted processed `(beam, source chunk)` coverage columns.

    Coverage records work performed, including chunks that yielded no
    candidates. Rows must be unique two-item iterables of non-negative integer
    coordinates; mappings are rejected because their iteration order would
    make the two coordinates ambiguous. Sorting makes output deterministic
    across compatible-beam batch orderings.
    """

    pairs = []
    for row in coverage:
        if isinstance(row, Mapping):
            raise TypeError("coverage rows must be (beam_id, source_chunk_index)")
        try:
            beam, chunk = row
        except (TypeError, ValueError) as exc:
            raise ValueError("coverage rows must contain exactly two values") from exc
        pairs.append((_index(beam, "coverage beam_id"),
                      _index(chunk, "coverage source_chunk_index")))
    if len(set(pairs)) != len(pairs):
        raise ValueError("coverage contains duplicate rows")
    pairs.sort()
    return {
        "beam_id": np.asarray([x[0] for x in pairs], dtype=np.int32),
        "source_chunk_index": np.asarray([x[1] for x in pairs], dtype=np.int64),
    }


def _validate_metadata(metadata, coverage, events, members):
    """Validate essential provenance needed to interpret a reopened catalog.

    This function checks the version-2 contract required by consumers: offline
    pipeline identity, producer/startup provenance, strict timeout controls,
    and the join between emitted rows and grouping-window records.
    """

    if not isinstance(metadata, Mapping) or metadata.get("pipeline") != "offline":
        raise ValueError("catalog metadata must identify the offline pipeline")
    for key in (
            "processing", "producer", "startup_by_beam", "grouping_windows",
            "units"):
        if key not in metadata:
            raise ValueError(f"catalog metadata is missing {key}")
    producer = metadata["producer"]
    if (not isinstance(producer, Mapping)
            or not isinstance(producer.get("config_yaml"), str)
            or not producer["config_yaml"]
            or not isinstance(producer.get("plan_yaml"), str)
            or not producer["plan_yaml"]):
        raise ValueError("catalog must embed exact config_yaml and plan_yaml")
    processing = metadata["processing"]
    if not isinstance(processing, Mapping):
        raise ValueError("catalog processing metadata must be a mapping")
    processing_fields = {
        "complete", "peakfinder", "snr_threshold", "dm_reach_by_tree",
        "waist_bins_by_tree", "time_radius_by_tree",
        "requested_time_radius_by_tree", "grouping_method",
        "grouping_association_domain", "halo_size",
        "effective_grouping_halo_columns_by_tree",
        "dm_tolerance", "time_tolerance", "beam_batch_size", "timeout_ms",
        "timeout_policy",
    }
    if set(processing) != processing_fields:
        raise ValueError("catalog processing metadata has the wrong fields")
    if (processing["peakfinder"] != "full_band"
            or processing["grouping_method"]
            != "persistent_partition_representative"):
        raise ValueError("catalog records an unsupported processing method")
    if (processing["grouping_association_domain"]
            != "owner_plus_resolved_next_map_halo"):
        raise ValueError("catalog records an unsupported association domain")
    if not isinstance(processing["complete"], (bool, np.bool_)):
        raise ValueError("catalog processing complete must be bool")
    for name in ("snr_threshold", "dm_tolerance", "time_tolerance"):
        value = processing[name]
        if (isinstance(value, (bool, np.bool_))
                or not isinstance(value, (int, float, np.integer, np.floating))
                or not isfinite(float(value))):
            raise ValueError(f"catalog processing {name} must be finite")
    if processing["dm_tolerance"] < 0 or processing["time_tolerance"] < 0:
        raise ValueError("catalog grouping tolerances must be non-negative")
    reaches = processing["dm_reach_by_tree"]
    waists = processing["waist_bins_by_tree"]
    radii = processing["time_radius_by_tree"]
    requested_radii = processing["requested_time_radius_by_tree"]
    grouping_halos = processing["effective_grouping_halo_columns_by_tree"]
    sequences = reaches, waists, radii, requested_radii, grouping_halos
    if (any(not isinstance(values, Sequence)
            or isinstance(values, (str, bytes)) for values in sequences)
            or not reaches
            or any(len(values) != len(reaches) for values in sequences[1:])):
        raise ValueError("catalog peakfinder geometry is malformed")
    for name, values in (
            ("dm_reach_by_tree", reaches),
            ("waist_bins_by_tree", waists),
            ("time_radius_by_tree", radii),
            ("requested_time_radius_by_tree", requested_radii),
            ("effective_grouping_halo_columns_by_tree", grouping_halos)):
        if any(isinstance(value, (bool, np.bool_))
               or not isinstance(value, (int, np.integer)) or value < 0
               for value in values):
            raise ValueError(f"catalog processing {name} is malformed")
    if any(effective > requested for effective, requested in zip(
            radii, requested_radii)):
        raise ValueError("catalog effective time radius exceeds requested")
    timeout_policy = processing.get("timeout_policy")
    if timeout_policy not in ("discard", "emit_partial"):
        raise ValueError("catalog has an invalid timeout policy")
    for name in ("halo_size", "beam_batch_size", "timeout_ms"):
        value = processing.get(name)
        if isinstance(value, (bool, np.bool_)) or not isinstance(
                value, (int, np.integer)):
            raise ValueError(f"catalog processing {name} must be an integer")
    if (processing["halo_size"] < 2
            or processing["beam_batch_size"] < 1
            or processing["timeout_ms"] < 0):
        raise ValueError("catalog processing integer is outside its range")
    if any(effective > processing["halo_size"] * radius
           for effective, radius in zip(grouping_halos, radii)):
        raise ValueError("catalog effective grouping halo exceeds configured")

    startup = metadata["startup_by_beam"]
    if not isinstance(startup, Sequence) or isinstance(startup, (str, bytes)):
        raise ValueError("startup_by_beam must be a sequence")
    startup_beams = set()
    for row in startup:
        if not isinstance(row, Mapping):
            raise ValueError("startup metadata row must be a mapping")
        beam = row.get("beam_id")
        status = row.get("status")
        if (isinstance(beam, (bool, np.bool_))
                or not isinstance(beam, (int, np.integer)) or beam < 0
                or status not in ("authoritative", "assumed")):
            raise ValueError("invalid startup metadata row")
        if beam in startup_beams:
            raise ValueError("startup metadata contains a duplicate beam")
        startup_beams.add(int(beam))
        start = row.get("producer_start_chunk_index")
        if status == "authoritative":
            if (isinstance(start, (bool, np.bool_))
                    or not isinstance(start, (int, np.integer)) or start < 0):
                raise ValueError("authoritative startup needs a valid start")
        elif start is not None:
            raise ValueError("assumed startup cannot claim a producer start")
    covered_beams = set(coverage["beam_id"].tolist())
    if not covered_beams.issubset(startup_beams):
        raise ValueError("every processed beam needs startup validity metadata")

    windows = metadata["grouping_windows"]
    if not isinstance(windows, Sequence) or isinstance(windows, (str, bytes)):
        raise ValueError("grouping_windows must be a sequence")
    by_id = {}
    owner_window_pairs = set()
    covered_pairs = set(zip(
        coverage["beam_id"].tolist(), coverage["source_chunk_index"].tolist()
    ))
    for expected_id, row in enumerate(windows):
        if not isinstance(row, Mapping):
            raise ValueError("grouping window metadata row must be a mapping")
        if row.get("grouping_window_id") != expected_id:
            raise ValueError("grouping window IDs must be contiguous from zero")
        timed_out = row.get("timed_out")
        status = row.get("output_status")
        if not isinstance(timed_out, (bool, np.bool_)):
            raise ValueError("grouping window timed_out must be bool")
        if status not in ("complete", "discarded", "partial"):
            raise ValueError("invalid grouping window output status")
        if bool(timed_out) != (status != "complete"):
            raise ValueError("grouping window timeout/status mismatch")
        if status == "discarded" and timeout_policy != "discard":
            raise ValueError("discarded window disagrees with timeout policy")
        if status == "partial" and timeout_policy != "emit_partial":
            raise ValueError("partial window disagrees with timeout policy")
        owner = row.get("owner_source_chunk_index")
        beams = row.get("beam_ids")
        if (isinstance(owner, (bool, np.bool_))
                or not isinstance(owner, (int, np.integer)) or owner < 0
                or not isinstance(beams, Sequence)
                or isinstance(beams, (str, bytes)) or not beams):
            raise ValueError("invalid grouping window owner/beam metadata")
        invalid_beam = any(
            isinstance(beam, (bool, np.bool_))
            or not isinstance(beam, (int, np.integer)) or beam < 0
            for beam in beams
        )
        if (invalid_beam
                or len(beams) > processing["beam_batch_size"]
                or len(set(beams)) != len(beams)):
            raise ValueError("invalid grouping window beam metadata")
        if any((beam, owner) not in covered_pairs for beam in beams):
            raise ValueError("grouping window is absent from processed coverage")
        if processing["timeout_ms"] == 0 and timed_out:
            raise ValueError("disabled timeout cannot produce a timed-out window")
        pairs = {(int(beam), int(owner)) for beam in beams}
        if owner_window_pairs & pairs:
            raise ValueError("a beam/owner pair occurs in multiple windows")
        owner_window_pairs.update(pairs)
        by_id[expected_id] = {
            "timed_out": bool(timed_out),
            "status": status,
            "owner": int(owner),
            "beams": {int(beam) for beam in beams},
        }

    # Empty and discarded windows have no event/member rows, so coverage is
    # the only independent evidence that their provenance record exists.  A
    # complete run owns every covered chunk.  An incomplete max-chunks prefix
    # may omit only the final covered chunk of each beam (the deliberately
    # unresolved right edge); a fully processed shorter beam may still own all
    # of its chunks when another beam made the overall run incomplete.
    coverage_by_beam = {}
    for beam, chunk in covered_pairs:
        coverage_by_beam.setdefault(int(beam), []).append(int(chunk))
    for beam, chunks in coverage_by_beam.items():
        chunks.sort()
        if any(right != left + 1 for left, right in zip(chunks, chunks[1:])):
            raise ValueError("catalog coverage must be contiguous per beam")
        actual = {
            owner for candidate_beam, owner in owner_window_pairs
            if candidate_beam == beam
        }
        full = set(chunks)
        if processing["complete"]:
            valid = actual == full
        else:
            valid = actual == full or actual == set(chunks[:-1])
        if not valid:
            raise ValueError(
                "grouping window provenance does not match processed coverage"
            )

    for table_name, table in (("events", events), ("members", members)):
        for row_index, (window_id, timed_out) in enumerate(zip(
                table["grouping_window_id"].tolist(),
                table["grouping_timed_out"].tolist())):
            if window_id not in by_id:
                raise ValueError(f"{table_name} references an unknown window")
            window = by_id[window_id]
            if bool(timed_out) != window["timed_out"]:
                raise ValueError(
                    f"{table_name} timeout tag disagrees with its window"
                )
            if window["status"] == "discarded":
                raise ValueError("discarded grouping window contains output rows")
            if int(table["beam_id"][row_index]) not in window["beams"]:
                raise ValueError(f"{table_name} beam is absent from its window")
            source = int(table["source_chunk_index"][row_index])
            if source not in (window["owner"], window["owner"] + 1):
                raise ValueError(
                    f"{table_name} row lies outside its owner/next window"
                )

    # Every published event is owned by at least one member from the window's
    # owner chunk, even if its representative (and event projection) is from
    # the louder next-chunk halo.  Members also remain inside one independent
    # beam/primary-tree domain and one provenance window.
    for event_row, event_id in enumerate(events["event_id"]):
        member_rows = members["event_id"] == event_id
        window_id = int(events["grouping_window_id"][event_row])
        window = by_id[window_id]
        if not np.any(
                members["source_chunk_index"][member_rows] == window["owner"]):
            raise ValueError("event has no member from its owner chunk")
        if (not np.all(members["grouping_window_id"][member_rows] == window_id)
                or not np.all(
                    members["beam_id"][member_rows]
                    == events["beam_id"][event_row]
                )
                or not np.all(
                    members["primary_tree_index"][member_rows]
                    == events["primary_tree_index"][event_row]
                )):
            raise ValueError("event members cross a grouping domain or window")


def build_trigger_catalog_tree(batches, *, coverage, metadata):
    """Concatenate beam batches into one validated in-memory ASDF tree.

    Parameters
    ----------
    batches : iterable of CatalogBatch
        Host batches in global-ID order. The caller must already have applied
        cumulative offsets with :func:`catalog_batch_from_grouping_result`.
    coverage : iterable of pair-like rows
        Every processed `(beam_id, source_chunk_index)` pair, whether or not
        that chunk produced a member.
    metadata : mapping
        Provenance normally returned by :func:`make_catalog_metadata`.

    Returns
    -------
    dict
        Materialized ASDF tree containing `format`, `format_version`,
        `events`, `members`, `coverage`, and `metadata`.

    Notes
    -----
    Concatenation occurs only on final host columns. Empty runs receive
    correctly typed zero-length arrays. Full validation after concatenation
    catches incorrect offsets, lost member rows, representative mismatches,
    and incomplete coverage metadata before any file is written.
    """

    batches = tuple(batches)
    for batch in batches:
        if not isinstance(batch, CatalogBatch):
            raise TypeError("batches must contain CatalogBatch objects")
    # Tree shapes are no longer ragged here: each logical table is a set of
    # one-dimensional columns, so batches can be concatenated independently by
    # column while retaining the schema's exact dtype for an empty run.
    concatenate = lambda name, schema: {
        key: (np.concatenate([getattr(b, name)[key] for b in batches])
              if batches else np.empty(0, dtype=dtype))
        for key, dtype in schema.items()
    }
    events = concatenate("events", EVENT_DTYPES)
    members = concatenate("members", MEMBER_DTYPES)
    coverage_table = _coverage_table(coverage)
    tree = {
        "format": CATALOG_FORMAT,
        "format_version": CATALOG_VERSION,
        "events": events,
        "members": members,
        "coverage": coverage_table,
        "metadata": dict(metadata),
    }
    validate_trigger_catalog_tree(tree)
    return tree


def validate_trigger_catalog_tree(tree):
    """Validate a materialized or reopened version-2 offline catalog.

    The validator checks format/version, exact table schemas and dtypes,
    sorted-unique coverage, global ID/link consistency, representative-derived
    event fields, and required producer/startup metadata. Member provenance
    must be a subset of processed coverage: a catalog may legitimately contain
    covered chunks with no detections, but never a member from an unreported
    chunk. The validated input tree is returned unchanged for convenient use
    after an eager ASDF reopen.
    """

    if (tree.get("format") != CATALOG_FORMAT
            or tree.get("format_version") != CATALOG_VERSION):
        raise ValueError("unsupported offline catalog format or version")
    events = _table(tree.get("events"), EVENT_DTYPES, "events")
    members = _table(tree.get("members"), MEMBER_DTYPES, "members")
    coverage = _table(tree.get("coverage"), COVERAGE_DTYPES, "coverage")
    pairs = list(zip(coverage["beam_id"], coverage["source_chunk_index"]))
    if pairs != sorted(set(pairs)):
        raise ValueError("coverage rows must be sorted and unique")
    _validate_tables(events, members)
    member_sources = set(zip(
        members["beam_id"].tolist(),
        members["source_chunk_index"].tolist(),
    ))
    if not member_sources.issubset(set(pairs)):
        raise ValueError("a member source is absent from processed coverage")
    _validate_metadata(tree.get("metadata"), coverage, events, members)
    return tree


def write_trigger_catalog(
        path, batches, *, coverage, metadata, compression="zlib"):
    """Atomically write, reopen, and validate one offline ASDF catalog.

    Parameters
    ----------
    path : path-like
        Destination file. Its parent directory must already exist. The result
        is returned as an absolute path.
    batches, coverage, metadata
        Inputs accepted by :func:`build_trigger_catalog_tree`.
    compression : str, optional
        ASDF array-compression name; `"zlib"` is the default.

    Returns
    -------
    str
        Absolute destination path after successful replacement.

    Notes
    -----
    A temporary file is created in the destination directory so
    :func:`os.replace` is a same-filesystem atomic publication step. The ASDF
    payload uses checksums and is eagerly reopened and fully validated before
    publication. If writing or validation raises (including interruption), the
    temporary is removed and an existing destination is left untouched. The
    original exception is then re-raised; no partial catalog is returned.
    """

    import asdf
    output = os.path.abspath(os.fspath(path))
    parent = os.path.dirname(output) or os.curdir
    if not os.path.isdir(parent):
        raise ValueError(f"catalog directory does not exist: {parent}")
    tree = build_trigger_catalog_tree(
        batches, coverage=coverage, metadata=metadata)
    # Same-directory staging is required for atomic replacement and avoids
    # exposing a partially compressed ASDF file to readers.
    fd, temporary = tempfile.mkstemp(
        prefix=f".{os.path.basename(output)}.tmp-", suffix=".asdf", dir=parent
    )
    os.close(fd)
    try:
        asdf.AsdfFile(tree).write_to(
            temporary, all_array_compression=compression, write_checksums=True)
        # Validate what ASDF actually serialized, not merely the source dict.
        # Eager loading also forces checksum/decompression errors before the
        # destination name becomes visible.
        with asdf.open(temporary, lazy_load=False) as reopened:
            validate_trigger_catalog_tree(reopened.tree)
        os.replace(temporary, output)
    except BaseException:
        try:
            os.unlink(temporary)
        except FileNotFoundError:
            pass
        raise
    return output


__all__ = [
    "CATALOG_FORMAT", "CATALOG_VERSION", "CatalogBatch",
    "build_trigger_catalog_tree", "catalog_batch_from_grouping_result",
    "make_catalog_metadata", "validate_trigger_catalog_tree",
    "write_trigger_catalog",
]
