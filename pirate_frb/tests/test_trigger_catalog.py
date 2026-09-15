"""Round-trip tests for the minimal offline trigger catalog."""

import os
import tempfile

import numpy as np

from ..Clustering import cluster_candidates
from ..BowtiePeakfinding import EdgeFlag
from ..ArgmaxMetadata import ARGMAX_ENCODING
from ..TriggerCatalog import (
    CATALOG_FORMAT,
    CATALOG_VERSION,
    build_trigger_catalog_tree,
    catalog_batch_from_grouping_result,
    make_catalog_metadata,
    validate_trigger_catalog_tree,
    write_trigger_catalog,
)
from .test_clustering import _candidates, _geometry


def _grouped_batches(cp):
    """Return two final GPU results with distinct beams and chunks."""

    geometry = _geometry(cp)
    first = cluster_candidates(_candidates(cp, geometry, [
        {
            "beam_id": 100, "tree": 0, "source_chunk_index": 2,
            "snr": 20.0, "dm": 50.0, "toa": 4095.75,
            "argmax_token": 0x00010000, "edge_flags": 0,
        },
        {
            "beam_id": 100, "tree": 1, "source_chunk_index": 3,
            "snr": 18.0, "dm": 50.2, "toa": 4096.0,
            "argmax_token": 0x00020000,
            "edge_flags": int(EdgeFlag.STARTUP_INCOMPLETE),
        },
    ]), geometry)
    second = cluster_candidates(_candidates(cp, geometry, [
        {
            "beam_id": 101, "tree": 0, "source_chunk_index": 7,
            "snr": 16.0, "dm": 80.0, "toa": 9000.0,
            "argmax_token": 0x00030000,
            "edge_flags": int(EdgeFlag.STARTUP_INCOMPLETE),
        },
    ]), geometry)
    return first, second


def _metadata():
    """Build complete provenance for both synthetic beams."""

    return make_catalog_metadata(
        config_yaml="config: exact\n",
        plan_yaml="plan: exact\n",
        snr_threshold=10.0,
        dm_reach_by_tree=(4, 4, 4, 4),
        dcores=(1, 2, 4, 8),
        argmax_encoding=ARGMAX_ENCODING,
        waist_bins_by_tree=(1, 1, 1, 1),
        time_radius_by_tree=(3, 3, 3, 3),
        requested_time_radius_by_tree=(5, 5, 5, 5),
        halo_size=2,
        effective_grouping_halo_columns_by_tree=(6, 6, 6, 6),
        dm_tolerance=1.5,
        time_tolerance=1.5,
        beam_batch_size=1,
        timeout_ms=400,
        timeout_policy="emit_partial",
        grouping_windows=(
            {
                "grouping_window_id": 0,
                "beam_ids": [100],
                "owner_source_chunk_index": 2,
                "timed_out": False,
                "output_status": "complete",
            },
            {
                "grouping_window_id": 1,
                "beam_ids": [100],
                "owner_source_chunk_index": 3,
                "timed_out": False,
                "output_status": "complete",
            },
            {
                "grouping_window_id": 2,
                "beam_ids": [101],
                "owner_source_chunk_index": 7,
                "timed_out": True,
                "output_status": "partial",
            },
        ),
        startup_by_beam=(
            {
                "beam_id": 100,
                "status": "authoritative",
                "producer_start_chunk_index": 0,
            },
            {"beam_id": 101, "status": "assumed"},
        ),
        complete=True,
    )


def test_discarded_timeout_window_is_traceable_without_rows():
    """A discard-policy timeout survives as metadata with empty row tables."""

    metadata = make_catalog_metadata(
        config_yaml="config: exact\n",
        plan_yaml="plan: exact\n",
        snr_threshold=10.0,
        dm_reach_by_tree=(8,),
        dcores=(1,),
        argmax_encoding=ARGMAX_ENCODING,
        waist_bins_by_tree=(1,),
        time_radius_by_tree=(7,),
        requested_time_radius_by_tree=(15,),
        halo_size=2,
        effective_grouping_halo_columns_by_tree=(14,),
        dm_tolerance=1.5,
        time_tolerance=1.5,
        beam_batch_size=1,
        timeout_ms=1,
        timeout_policy="discard",
        grouping_windows=({
            "grouping_window_id": 0,
            "beam_ids": [100],
            "owner_source_chunk_index": 2,
            "timed_out": True,
            "output_status": "discarded",
        },),
        startup_by_beam=({
            "beam_id": 100,
            "status": "authoritative",
            "producer_start_chunk_index": 0,
        },),
        complete=True,
    )
    tree = build_trigger_catalog_tree(
        (), coverage=((100, 2),), metadata=metadata
    )
    assert len(tree["events"]["event_id"]) == 0
    assert len(tree["members"]["candidate_id"]) == 0
    assert tree["metadata"]["grouping_windows"] == [{
        "grouping_window_id": 0,
        "beam_ids": [100],
        "owner_source_chunk_index": 2,
        "timed_out": True,
        "output_status": "discarded",
    }]
    tree["metadata"]["grouping_windows"] = []
    try:
        validate_trigger_catalog_tree(tree)
    except ValueError as exc:
        assert "processed coverage" in str(exc)
    else:
        raise AssertionError("missing discarded-window provenance was accepted")


def test_catalog_gpu_boundary_offsets_and_members(cuda_device_id=0):
    """Final rows cross once with global event and candidate identifiers."""

    import cupy as cp

    with cp.cuda.Device(cuda_device_id):
        first, second = _grouped_batches(cp)
        batch0 = catalog_batch_from_grouping_result(
            first, event_id_offset=0, candidate_id_offset=0,
            grouping_window_id=0,
        )
        batch1 = catalog_batch_from_grouping_result(
            second,
            event_id_offset=len(first.events),
            candidate_id_offset=len(first.candidates),
            grouping_window_id=1,
            grouping_timed_out=True,
        )
        assert np.array_equal(
            batch0.events["event_id"],
            np.arange(len(first.events), dtype=np.int64),
        )
        assert np.array_equal(
            batch1.events["event_id"],
            np.arange(
                len(first.events),
                len(first.events) + len(second.events),
                dtype=np.int64,
            ),
        )
        assert set(batch0.members["candidate_id"].tolist()) == {0, 1}
        assert set(batch1.members["candidate_id"].tolist()) == {2}
        assert set(batch0.members) >= {
            "event_id", "candidate_id", "beam_id", "tree", "snr", "dm",
            "toa_sample_abs", "width_samp", "width_ms", "freq_lo_MHz",
            "freq_hi_MHz", "argmax_token", "source_chunk_index", "idm",
            "itime", "edge_flags",
        }
        startup = int(EdgeFlag.STARTUP_INCOMPLETE)
        assert batch0.events["edge_flags"].dtype == np.dtype(np.uint8)
        assert batch0.members["edge_flags"].dtype == np.dtype(np.uint8)
        assert batch0.events["edge_flags"].tolist() == [0]
        assert set(batch0.members["edge_flags"].tolist()) == {0, startup}
        assert batch1.events["edge_flags"].tolist() == [startup]
        assert batch1.members["edge_flags"].tolist() == [startup]
        assert batch0.events["grouping_window_id"].tolist() == [0]
        assert batch0.members["grouping_timed_out"].tolist() == [False, False]
        assert batch1.events["grouping_window_id"].tolist() == [1]
        assert batch1.events["grouping_timed_out"].tolist() == [True]
        assert batch1.members["grouping_timed_out"].tolist() == [True]


def test_trigger_catalog_write_reopen_and_validate(cuda_device_id=0):
    """Atomic output reopens with exact provenance, coverage, and links."""

    import asdf
    import cupy as cp

    with cp.cuda.Device(cuda_device_id):
        first, second = _grouped_batches(cp)
        batch0 = catalog_batch_from_grouping_result(
            first, event_id_offset=0, candidate_id_offset=0,
            grouping_window_id=0,
        )
        batch1 = catalog_batch_from_grouping_result(
            second,
            event_id_offset=len(first.events),
            candidate_id_offset=len(first.candidates),
            grouping_window_id=2,
            grouping_timed_out=True,
        )
        coverage = ((100, 2), (100, 3), (101, 7))
        with tempfile.TemporaryDirectory(prefix="pirate-catalog-test-") as tmp:
            requested = os.path.join(tmp, "events.asdf")
            output = write_trigger_catalog(
                requested,
                (batch0, batch1),
                coverage=coverage,
                metadata=_metadata(),
            )
            assert output == os.path.abspath(requested)
            assert os.path.isfile(output)
            assert not [
                name for name in os.listdir(tmp) if ".tmp-" in name
            ]
            with asdf.open(output, lazy_load=False) as af:
                validate_trigger_catalog_tree(af.tree)
                assert af.tree["format"] == CATALOG_FORMAT
                assert af.tree["format_version"] == CATALOG_VERSION
                assert af.tree["metadata"]["pipeline"] == "offline"
                assert (
                    af.tree["metadata"]["processing"]["peakfinder"]
                    == "full_band"
                )
                assert list(af.tree["metadata"]["processing"][
                    "time_radius_by_tree"
                ]) == [3, 3, 3, 3]
                assert list(af.tree["metadata"]["processing"][
                    "requested_time_radius_by_tree"
                ]) == [5, 5, 5, 5]
                assert list(af.tree["metadata"]["processing"][
                    "effective_grouping_halo_columns_by_tree"
                ]) == [6, 6, 6, 6]
                assert af.tree["metadata"]["processing"][
                    "grouping_association_domain"
                ] == "owner_plus_resolved_next_map_halo"
                assert af.tree["format_version"] == 3
                assert af.tree["metadata"]["producer"]["dcores"] == [1, 2, 4, 8]
                assert af.tree["metadata"]["producer"]["argmax_encoding"] == ARGMAX_ENCODING
                assert (
                    af.tree["metadata"]["grouping_windows"][2]["output_status"]
                    == "partial"
                )
                assert (
                    af.tree["metadata"]["producer"]["config_yaml"]
                    == "config: exact\n"
                )
                assert np.array_equal(
                    np.asarray(af.tree["coverage"]["beam_id"]),
                    np.asarray([100, 100, 101], dtype=np.int32),
                )
                events = af.tree["events"]
                members = af.tree["members"]
                assert len(events["event_id"]) == (
                    len(first.events) + len(second.events)
                )
                assert len(members["candidate_id"]) == 3
                assert np.array_equal(
                    np.sort(np.asarray(members["candidate_id"])),
                    np.arange(3, dtype=np.int64),
                )
                startup = int(EdgeFlag.STARTUP_INCOMPLETE)
                assert np.asarray(events["edge_flags"]).dtype == np.uint8
                assert np.asarray(members["edge_flags"]).dtype == np.uint8
                assert np.asarray(events["edge_flags"]).tolist() == [
                    0, startup
                ]
                assert set(np.asarray(members["edge_flags"]).tolist()) == {
                    0, startup
                }
                assert np.asarray(
                    events["grouping_window_id"]
                ).tolist() == [0, 2]
                assert np.asarray(
                    events["grouping_timed_out"]
                ).tolist() == [False, True]


def test_catalog_rejects_missing_decoder_provenance():
    """Version 3 requires encoding and Dcores even for an empty catalog."""
    import copy
    tree = build_trigger_catalog_tree(
        (), coverage=((100, 2), (100, 3), (101, 7)), metadata=_metadata(),
    )
    for mutation in (
            lambda t: t.update(format_version=2),
            lambda t: t["metadata"]["producer"].pop("dcores"),
            lambda t: t["metadata"]["producer"].pop("argmax_encoding"),
            lambda t: t["metadata"]["producer"].update(argmax_encoding="legacy"),
            lambda t: t["metadata"]["producer"].update(dcores=[1]),
            lambda t: t["metadata"]["producer"].update(dcores=[1, 2, 3, 4])):
        invalid = copy.deepcopy(tree)
        mutation(invalid)
        try:
            validate_trigger_catalog_tree(invalid)
        except ValueError:
            pass
        else:
            raise AssertionError("catalog accepted incompatible decoder provenance")
