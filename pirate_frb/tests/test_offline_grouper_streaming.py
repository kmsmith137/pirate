"""Focused GPU regressions for bounded offline grouping windows."""

from dataclasses import fields, replace
from types import SimpleNamespace

import numpy as np

from ..OfflineMapReader import BeamBatch
from ..Clustering import (
    GpuDecodedCandidates,
    ClusteringTolerances,
)
from ..BowtiePeakfinding import GpuRawCandidates
from .. import run_offline_grouper as runner
from .test_clustering import _geometry


def _raw(cp, rows):
    """Build compact raw rows whose token is a synthetic absolute TOA."""

    def values(name, default):
        return [row.get(name, default(i, row)) for i, row in enumerate(rows)]

    return GpuRawCandidates(
        beam_id=cp.asarray(values("beam_id", lambda i, row: 7), cp.int32),
        source_chunk_index=cp.asarray(
            values("source_chunk_index", lambda i, row: 0), cp.int64
        ),
        tree=cp.asarray(values("tree", lambda i, row: 0), cp.int32),
        idm=cp.asarray(values("idm", lambda i, row: i), cp.int32),
        itime=cp.asarray(values("itime", lambda i, row: i), cp.int32),
        snr=cp.asarray(values("snr", lambda i, row: 20.0 - i), cp.float32),
        argmax_token=cp.asarray(
            values("toa", lambda i, row: 100 + i), cp.uint32
        ),
        edge_flags=cp.asarray(
            values("edge_flags", lambda i, row: 0), cp.uint8
        ),
    )


def _decode_raw(cp, raw, geometry):
    """Decode synthetic raw rows without involving a producer plan."""

    tree = raw.tree.astype(cp.int32, copy=False)
    n = len(raw)
    return GpuDecodedCandidates(
        beam_id=raw.beam_id.astype(cp.int32, copy=False),
        source_chunk_index=raw.source_chunk_index.astype(
            cp.int64, copy=False
        ),
        tree=tree,
        idm=raw.idm.astype(cp.int32, copy=False),
        itime=raw.itime.astype(cp.int32, copy=False),
        snr=raw.snr.astype(cp.float64),
        argmax_token=raw.argmax_token.astype(cp.uint32, copy=False),
        edge_flags=raw.edge_flags.astype(cp.uint8, copy=False),
        dm=raw.idm.astype(cp.float64),
        toa_sample_abs=raw.argmax_token.astype(cp.float64),
        width_samp=cp.ones(n, dtype=cp.float64),
        width_ms=cp.ones(n, dtype=cp.float64),
        freq_lo_MHz=cp.full(n, 300.0, dtype=cp.float64),
        freq_hi_MHz=cp.full(n, 1500.0, dtype=cp.float64),
        primary_tree_index=geometry.primary_tree_index_by_tree[tree],
        dm_step=geometry.dm_step_by_tree[tree],
        time_step_samples=geometry.time_step_samples_by_tree[tree],
    )


class _RawDecoder:
    """Minimal decoder adapter used at the runner's raw/physical boundary."""

    def __init__(self, cp, geometry):
        self.cp = cp
        self.geometry = geometry

    def decode(self, raw):
        return _decode_raw(self.cp, raw, self.geometry)


def _take_decoded(cp, candidates, selector):
    """Gather a decoded GPU table for deterministic timeout injection."""

    return GpuDecodedCandidates(**{
        field.name: getattr(candidates, field.name)[selector]
        for field in fields(GpuDecodedCandidates)
    })


def _candidate_signatures(cp, grouped_or_candidates):
    """Return small host signatures at the explicit test boundary."""

    candidates = getattr(
        grouped_or_candidates, "candidates", grouped_or_candidates
    )
    return list(zip(
        cp.asnumpy(candidates.source_chunk_index).tolist(),
        cp.asnumpy(candidates.tree).tolist(),
        cp.asnumpy(candidates.idm).tolist(),
        cp.asnumpy(candidates.itime).tolist(),
    ))


def test_streaming_ownership_louder_next_chunk_and_no_i_plus_2_influence(
        cuda_device_id=0):
    """Owned seam groups finalize once even when the halo row is louder."""

    import cupy as cp

    with cp.cuda.Device(cuda_device_id):
        geometry = _geometry(cp)
        decoder = _RawDecoder(cp, geometry)
        config = ClusteringTolerances(
            dm_tolerance_bins=1.5, time_padding_bins=1.5
        )

        # Chunk zero owns one same-chunk group and one seam group.  The seam
        # representative is the louder chunk-one row.  A distant chunk-one
        # row is a future-only halo group and must be retained, not emitted.
        owner_zero = _raw(cp, [
            {
                "source_chunk_index": 0, "tree": 0, "idm": 10,
                "itime": 2, "toa": 10, "snr": 15.0,
            },
            {
                "source_chunk_index": 0, "tree": 1, "idm": 10,
                "itime": 2, "toa": 10, "snr": 14.0,
            },
            {
                "source_chunk_index": 0, "tree": 0, "idm": 20,
                "itime": 7, "toa": 20, "snr": 20.0,
            },
        ])
        halo_one = _raw(cp, [
            {
                "source_chunk_index": 1, "tree": 1, "idm": 20,
                "itime": 0, "toa": 20, "snr": 30.0,
            },
            {
                "source_chunk_index": 1, "tree": 3, "idm": 50,
                "itime": 1, "toa": 50, "snr": 12.0,
            },
        ])
        window_zero, retained_one = runner._group_streaming_window(
            cp,
            owner_zero,
            halo_one,
            0,
            1,
            decoder,
            geometry,
            config,
            timeout_ms=0,
            timeout_policy="discard",
        )

        assert window_zero.output_status == "complete"
        assert window_zero.grouped.complete is True
        assert len(window_zero.grouped.events) == 2
        event_dm = cp.asnumpy(window_zero.grouped.events.dm)
        same_chunk_event = int(np.flatnonzero(event_dm == 10.0)[0])
        same_chunk_members = window_zero.grouped.members.candidate_index[
            window_zero.grouped.members.event_id == same_chunk_event
        ]
        assert set(cp.asnumpy(
            window_zero.grouped.candidates.tree[same_chunk_members]
        ).tolist()) == {0, 1}
        assert set(cp.asnumpy(
            window_zero.grouped.candidates.source_chunk_index[
                same_chunk_members
            ]
        ).tolist()) == {0}
        seam_event = int(np.flatnonzero(event_dm == 20.0)[0])
        representative = int(
            window_zero.grouped.events.representative_candidate_index[
                seam_event
            ].item()
        )
        assert int(
            window_zero.grouped.candidates.source_chunk_index[
                representative
            ].item()
        ) == 1
        seam_members = window_zero.grouped.members.candidate_index[
            window_zero.grouped.members.event_id == seam_event
        ]
        assert set(cp.asnumpy(
            window_zero.grouped.candidates.source_chunk_index[seam_members]
        ).tolist()) == {0, 1}
        assert cp.asnumpy(
            window_zero.grouped.events.member_count
        ).tolist() == [2, 2]
        assert _candidate_signatures(cp, retained_one) == [(1, 3, 50, 1)]

        # A deliberately compatible and much louder i+2 row cannot revisit
        # the already-published i/i+1 event.  With the consumed seam row gone,
        # it remains future-only while chunk one owns its independent row.
        halo_two = _raw(cp, [{
            "source_chunk_index": 2, "tree": 0, "idm": 20,
            "itime": 0, "toa": 20, "snr": 100.0,
        }])
        window_one, retained_two = runner._group_streaming_window(
            cp,
            retained_one,
            halo_two,
            1,
            2,
            decoder,
            geometry,
            config,
            timeout_ms=0,
            timeout_policy="discard",
        )
        assert len(window_one.grouped.events) == 1
        assert _candidate_signatures(cp, window_one.grouped) == [
            (1, 3, 50, 1)
        ]
        assert _candidate_signatures(cp, retained_two) == [(2, 0, 20, 0)]

        window_two, final_retained = runner._group_streaming_window(
            cp,
            retained_two,
            GpuRawCandidates.empty(),
            2,
            None,
            decoder,
            geometry,
            config,
            timeout_ms=0,
            timeout_policy="discard",
        )
        assert len(window_two.grouped.events) == 1
        assert _candidate_signatures(cp, window_two.grouped) == [
            (2, 0, 20, 0)
        ]
        assert len(final_retained) == 0

        emitted = sum((
            _candidate_signatures(cp, window.grouped)
            for window in (window_zero, window_one, window_two)
        ), [])
        assert len(emitted) == len(set(emitted)) == 6
        assert emitted.count((1, 1, 20, 0)) == 1
        # The saved first-window table is immutable and contains no i+2 row.
        assert set(
            cp.asnumpy(
                window_zero.grouped.candidates.source_chunk_index
            ).tolist()
        ) == {0, 1}


def test_streaming_timeout_policies_drop_backlog_and_keep_complete_events(
        cuda_device_id=0):
    """Discard emits nothing; partial emits one clean owner group only."""

    import cupy as cp

    with cp.cuda.Device(cuda_device_id):
        geometry = _geometry(cp)
        decoder = _RawDecoder(cp, geometry)
        config = ClusteringTolerances()
        owner = _raw(cp, [{
            "source_chunk_index": 0, "tree": 0, "idm": 20,
            "itime": 7, "toa": 20, "snr": 20.0,
        }])
        halo = _raw(cp, [
            {
                "source_chunk_index": 1, "tree": 1, "idm": 20,
                "itime": 0, "toa": 20, "snr": 30.0,
            },
            {
                "source_chunk_index": 1, "tree": 3, "idm": 50,
                "itime": 1, "toa": 50, "snr": 10.0,
            },
        ])

        real_group = runner.cluster_candidates
        seen_timeouts = []

        def timed_group(
                candidates, grouping_geometry, *, config=None, timeout_ms=0):
            seen_timeouts.append(timeout_ms)
            # Commit only the DM-20 seam group.  The DM-50 halo candidate is
            # unprocessed and must not enter the next owner window.
            keep = cp.flatnonzero(candidates.dm == 20.0).astype(
                cp.int64, copy=False
            )
            partial = real_group(
                _take_decoded(cp, candidates, keep),
                grouping_geometry,
                config=config,
                timeout_ms=0,
            )
            return replace(
                partial,
                input_candidate_index=keep,
                timed_out=True,
                complete=False,
            )

        runner.cluster_candidates = timed_group
        try:
            discarded, retained_discard = runner._group_streaming_window(
                cp,
                owner,
                halo,
                0,
                1,
                decoder,
                geometry,
                config,
                timeout_ms=17,
                timeout_policy="discard",
            )
            partial, retained_partial = runner._group_streaming_window(
                cp,
                owner,
                halo,
                0,
                1,
                decoder,
                geometry,
                config,
                timeout_ms=17,
                timeout_policy="emit_partial",
            )
        finally:
            runner.cluster_candidates = real_group

        assert seen_timeouts == [17, 17]
        assert discarded.output_status == "discarded"
        assert discarded.grouped.timed_out is True
        assert len(discarded.grouped.events) == 0
        assert len(discarded.grouped.candidates) == 0
        assert len(retained_discard) == 0

        assert partial.output_status == "partial"
        assert partial.grouped.timed_out is True
        assert partial.grouped.complete is False
        assert len(partial.grouped.events) == 1
        assert len(partial.grouped.candidates) == 2
        assert set(cp.asnumpy(
            partial.grouped.candidates.source_chunk_index
        ).tolist()) == {0, 1}
        assert cp.asnumpy(partial.grouped.events.member_count).tolist() == [2]
        assert np.all(cp.asnumpy(partial.grouped.candidates.edge_flags) == 0)
        # The consumed committed halo row and unprocessed halo row are both
        # absent, so neither policy can create a growing retry backlog.
        assert len(retained_partial) == 0


def _scripted_extract(
        cp, schedule, source_chunks, *, max_chunks=None, halo_size=2):
    """Run `_extract_beam_batch` with deterministic compact peak emissions."""

    geometries = (
        SimpleNamespace(tree=0, time_radius=1, ntime=8),
        SimpleNamespace(tree=1, time_radius=2, ntime=8),
    )
    grouping_geometry = _geometry(cp)

    class ScriptedExtractor:
        instances = []

        def __init__(
                self, geometry, *, threshold, beam_ids,
                assume_steady_state, halo_size):
            self.geometry = geometry
            self.threshold = threshold
            self.beam_ids = tuple(beam_ids)
            self.assume_steady_state = assume_steady_state
            self.halo_size = halo_size
            self.processed = []
            self.flush_count = 0
            ScriptedExtractor.instances.append(self)

        def process_chunk(
                self, snr_map, argmax_map, source_chunk_index, *,
                startup_valid_mask=None):
            self.processed.append(source_chunk_index)
            rows = [
                dict(row, tree=self.geometry.tree)
                for row in schedule.get(
                    (self.geometry.tree, source_chunk_index), ()
                )
            ]
            return _raw(cp, rows)

        def flush(self):
            self.flush_count += 1
            rows = [
                dict(row, tree=self.geometry.tree)
                for row in schedule.get((self.geometry.tree, "flush"), ())
            ]
            return _raw(cp, rows)

    class ScriptedLoader:
        def __init__(self):
            self.loads = []

        def load_beam_chunk(self, beam_ids, source_chunk_index):
            self.loads.append((tuple(beam_ids), source_chunk_index))
            return SimpleNamespace(
                snr_by_tree=(None, None),
                argmax_by_tree=(None, None),
            )

    loader = ScriptedLoader()
    decoder = _RawDecoder(cp, grouping_geometry)
    captures = []
    real_extractor = runner.StreamingPeakExtractor
    real_group = runner.cluster_candidates

    def recording_group(
            candidates, geometry, *, config=None, timeout_ms=0):
        captures.append({
            "source": cp.asnumpy(candidates.source_chunk_index).tolist(),
            "tree": cp.asnumpy(candidates.tree).tolist(),
            "itime": cp.asnumpy(candidates.itime).tolist(),
        })
        return real_group(
            candidates,
            geometry,
            config=config,
            timeout_ms=timeout_ms,
        )

    runner.StreamingPeakExtractor = ScriptedExtractor
    runner.cluster_candidates = recording_group
    windows = []
    try:
        coverage, startup, complete = runner._extract_beam_batch(
            loader,
            BeamBatch(
                beam_ids=(7,),
                source_chunk_indices=tuple(source_chunks),
                producer_start_chunk=None,
            ),
            geometries,
            decoder,
            grouping_geometry,
            threshold=10.0,
            halo_size=halo_size,
            timeout_ms=0,
            timeout_policy="discard",
            max_chunks=max_chunks,
            assume_steady_state=True,
            grouping_config=ClusteringTolerances(),
            consume_window=windows.append,
        )
    finally:
        runner.StreamingPeakExtractor = real_extractor
        runner.cluster_candidates = real_group

    return (
        (tuple(windows), coverage, startup, complete),
        captures,
        tuple(ScriptedExtractor.instances),
        loader.loads,
    )


def test_extract_batch_uses_ragged_halos_and_bounded_candidate_windows(
        cuda_device_id=0):
    """Each tree uses its own halo, and no call sees acquisition-wide rows."""

    import cupy as cp

    with cp.cuda.Device(cuda_device_id):
        schedule = {}
        for source in (1, 2):
            schedule[(0, source)] = (
                {
                    "source_chunk_index": source, "idm": 10 + 100 * source,
                    "itime": 1, "toa": 10 + 100 * source, "snr": 20.0,
                },
                {
                    "source_chunk_index": source, "idm": 20 + 100 * source,
                    "itime": 2, "toa": 20 + 100 * source, "snr": 19.0,
                },
            )
            schedule[(1, source)] = (
                {
                    "source_chunk_index": source, "idm": 30 + 100 * source,
                    "itime": 3, "toa": 30 + 100 * source, "snr": 18.0,
                },
                {
                    "source_chunk_index": source, "idm": 40 + 100 * source,
                    "itime": 4, "toa": 40 + 100 * source, "snr": 17.0,
                },
            )

        (windows, coverage, startup, complete), captures, extractors, loads = (
            _scripted_extract(cp, schedule, (0, 1, 2))
        )

        assert complete is True and startup == "assumed"
        assert coverage == ((7, 0), (7, 1), (7, 2))
        assert [window.owner_source_chunk_index for window in windows] == [
            0, 1, 2
        ]
        assert [len(window.grouped.events) for window in windows] == [0, 4, 4]
        # Radius one retains itime < 2; radius two retains itime < 4.
        assert set(zip(
            captures[0]["tree"], captures[0]["itime"]
        )) == {(0, 1), (1, 3)}
        assert len(captures[1]["source"]) == 6
        assert set(zip(
            captures[1]["source"],
            captures[1]["tree"],
            captures[1]["itime"],
        )) == {
            (1, 0, 1), (1, 0, 2), (1, 1, 3), (1, 1, 4),
            (2, 0, 1), (2, 1, 3),
        }
        assert set(zip(
            captures[2]["tree"], captures[2]["itime"]
        )) == {(0, 1), (0, 2), (1, 3), (1, 4)}
        # Eight acquisition candidates were processed, but bounded owner+halo
        # windows contain at most six and the empty chunk-zero window is kept.
        assert max(len(capture["source"]) for capture in captures) == 6
        emitted = sum((
            _candidate_signatures(cp, window.grouped)
            for window in windows
        ), [])
        assert len(emitted) == len(set(emitted)) == 8
        assert all(extractor.halo_size == 2 for extractor in extractors)
        assert all(extractor.processed == [0, 1, 2] for extractor in extractors)
        assert all(extractor.flush_count == 1 for extractor in extractors)
        assert loads == [((7,), 0), ((7,), 1), ((7,), 2)]


def test_extract_batch_eof_empty_chunks_and_max_chunks_prefix(
        cuda_device_id=0):
    """Only the physical EOF flushes; empty owners still produce provenance."""

    import cupy as cp

    with cp.cuda.Device(cuda_device_id):
        complete_schedule = {
            (0, 5): ({
                "source_chunk_index": 5, "idm": 10, "itime": 1,
                "toa": 10, "snr": 20.0,
            },),
            (1, "flush"): ({
                "source_chunk_index": 5, "idm": 20, "itime": 7,
                "toa": 20, "snr": 19.0,
            },),
        }
        (windows, coverage, startup, complete), captures, extractors, loads = (
            _scripted_extract(cp, complete_schedule, (5,))
        )
        assert complete is True and startup == "assumed"
        assert coverage == ((7, 5),)
        assert len(windows) == 1
        assert windows[0].owner_source_chunk_index == 5
        assert len(windows[0].grouped.events) == 2
        assert len(captures) == 1 and len(captures[0]["source"]) == 2
        assert all(extractor.flush_count == 1 for extractor in extractors)
        assert loads == [((7,), 5)]

        # The same first chunk is unresolved when it is merely a prefix of a
        # longer acquisition: there is no right-edge flush and no window yet.
        (windows, coverage, _, complete), captures, extractors, loads = (
            _scripted_extract(
                cp, complete_schedule, (5, 6), max_chunks=1
            )
        )
        assert complete is False
        assert coverage == ((7, 5),)
        assert windows == () and captures == []
        assert all(extractor.flush_count == 0 for extractor in extractors)
        assert loads == [((7,), 5)]

        # Two entirely empty physical chunks still generate two traceable
        # complete grouping windows, including the final EOF owner.
        (windows, coverage, _, complete), captures, extractors, _ = (
            _scripted_extract(cp, {}, (0, 1))
        )
        assert complete is True
        assert coverage == ((7, 0), (7, 1))
        assert [window.owner_source_chunk_index for window in windows] == [0, 1]
        assert all(len(window.grouped.events) == 0 for window in windows)
        assert len(captures) == 2
        assert all(capture["source"] == [] for capture in captures)
        assert all(extractor.flush_count == 1 for extractor in extractors)


def test_effective_grouping_halo_is_configured_resolved_intersection():
    """Grouping tolerances never enlarge the configured one-chunk domain."""

    geometries = (
        SimpleNamespace(ntime=8, time_radius=1),
        SimpleNamespace(ntime=8, time_radius=0),
    )
    assert runner._effective_grouping_halo_columns(
        geometries, 2
    ) == (2, 0)
    # A larger multiplier cannot admit the unresolved final radius of chunk
    # i+1.  Those rows belong to its later owner window and cannot revise i.
    assert runner._effective_grouping_halo_columns(
        geometries, 10
    ) == (7, 0)
