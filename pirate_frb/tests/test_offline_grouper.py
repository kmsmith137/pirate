"""Loader and end-to-end checks on the supplied offline acquisitions."""

import os
import tempfile

import numpy as np

from ..FrbOfflineGrouper import FrbOfflineGrouper
from ..GpuArgmaxDecoder import GpuArgmaxDecoder
from ..OfflineCandidateGrouper import (
    GroupingConfig,
    GroupingGeometry,
    group_candidates,
)
from ..Peakfinders import EdgeFlag, PeakFinderGeometry
from ..TriggerCatalog import validate_trigger_catalog_tree
from .. import run_offline_grouper as runner
from .test_offline_candidate_grouper import _candidates, _geometry


_SIX_TREE_EXAMPLE = "/home/mtrudu/chordsim"
_GROUPTEST = "/home/mtrudu/chordsim/grouptest"


def _runner_config(directory, *, snr_threshold=10.0):
    """Write the strict runner fixture without depending on source-tree data."""

    path = os.path.join(directory, "offline-grouper.yml")
    with open(path, "w", encoding="utf-8") as stream:
        stream.write(
            "peakfinding:\n"
            f"  snr_threshold: {float(snr_threshold):.17e}\n"
            "  dm_reach: 8\n"
            "  waist_bins: 1\n"
            "grouping:\n"
            "  halo_size: 2\n"
            "  dm_tolerance: 1.5\n"
            "  time_tolerance: 1.5\n"
            "execution:\n"
            "  beam_batch_size: 1\n"
            "  timeout_ms: 0\n"
            "  timeout_policy: discard\n"
        )
    return path


def _examples_available():
    """Return whether the optional repository-external integration data exists."""

    return all(os.path.isdir(path) for path in (
        _SIX_TREE_EXAMPLE, _GROUPTEST
    ))


def test_six_tree_ragged_loading_and_startup(cuda_device_id=0):
    """The six native map shapes upload independently and require an assumption."""

    import cupy as cp

    if not _examples_available():
        return
    loader = FrbOfflineGrouper(
        _SIX_TREE_EXAMPLE, cuda_device_id=cuda_device_id
    )
    expected_shapes = (
        (512, 256), (128, 128), (256, 128),
        (128, 128), (128, 64), (256, 64),
    )
    assert loader.beam_ids == (100,)
    assert loader.tree_shapes == expected_shapes
    assert loader.chunks_by_beam[100] == tuple(range(10))
    assert loader.producer_start_by_beam[100] is None
    assert len(set(expected_shapes)) > 1

    with cp.cuda.Device(cuda_device_id):
        maps = loader.load_beam_chunk((100,), 0)
        assert maps.source_chunk_index == 0
        assert maps.beam_ids == (100,)
        assert len(maps.snr_by_tree) == len(maps.argmax_by_tree) == 6
        for snr, argmax, shape in zip(
                maps.snr_by_tree, maps.argmax_by_tree, expected_shapes):
            assert isinstance(snr, cp.ndarray)
            assert isinstance(argmax, cp.ndarray)
            assert snr.shape == argmax.shape == (1,) + shape
            assert argmax.dtype == cp.uint32

    with tempfile.TemporaryDirectory(
            prefix="pirate-offline-config-") as tmp:
        config_file = _runner_config(tmp, snr_threshold=1.0e30)
        try:
            runner.run_offline_grouper(
                _SIX_TREE_EXAMPLE,
                config_file,
                max_chunks=1,
                cuda_device_id=cuda_device_id,
            )
        except ValueError as exc:
            assert "producer-start provenance is missing" in str(exc)
        else:
            raise AssertionError("missing startup provenance was accepted")

        lines = []
        original_print = runner.atomic_print
        runner.atomic_print = lines.append
        try:
            runner.run_offline_grouper(
                _SIX_TREE_EXAMPLE,
                config_file,
                max_chunks=1,
                cuda_device_id=cuda_device_id,
                assume_steady_state=True,
            )
        finally:
            runner.atomic_print = original_print
    assert any(
        "startup=assumed" in line and "complete=false" in line
        for line in lines
    )


def test_grouptest_full_band_golden_and_catalog(cuda_device_id=0):
    """The local DM-200 burst is emitted, decoded, labelled, and catalogued."""

    import asdf
    import cupy as cp

    if not _examples_available():
        return
    loader = FrbOfflineGrouper(_GROUPTEST, cuda_device_id=cuda_device_id)
    assert loader.beam_ids == (100,)
    assert loader.producer_start_by_beam == {100: 0}
    assert loader.tree_shapes == (
        (4096, 128), (1024, 64), (2048, 64), (1024, 64),
        (1024, 32), (2048, 32), (512, 32), (1024, 32),
        (1024, 16), (2048, 16),
    )

    with cp.cuda.Device(cuda_device_id):
        geometries = tuple(
            PeakFinderGeometry.from_plan(
                loader.plan, tree, dm_reach=8, waist_bins=1,
            )
            for tree in range(loader.ntrees)
        )
        windows = []
        coverage, startup, complete = runner._extract_beam_batch(
            loader,
            next(loader.iter_beam_batches()),
            geometries,
            GpuArgmaxDecoder(loader.plan, cuda_device_id=cuda_device_id),
            GroupingGeometry.from_plan(loader.plan),
            threshold=10.0,
            halo_size=2,
            timeout_ms=0,
            timeout_policy="discard",
            max_chunks=None,
            assume_steady_state=False,
            grouping_config=GroupingConfig(),
            consume_window=windows.append,
        )
        assert len(windows) == 10
        nonempty = [window for window in windows if len(window.grouped.events)]
        assert len(nonempty) == 1
        assert nonempty[0].output_status == "complete"
        grouped = nonempty[0].grouped
        candidates = grouped.candidates
        assert len(candidates) == 1
        assert cp.asnumpy(candidates.source_chunk_index).tolist() == [0]
        assert cp.asnumpy(candidates.tree).tolist() == [0]
        assert cp.asnumpy(candidates.idm).tolist() == [421]
        assert cp.asnumpy(candidates.itime).tolist() == [127]
        assert float(candidates.snr[0].item()) > 40.0
        assert candidates.edge_flags.dtype == cp.uint8
        assert cp.asnumpy(candidates.edge_flags).tolist() == [
            int(EdgeFlag.STARTUP_INCOMPLETE)
        ]
        assert 199.8 < float(candidates.dm[0].item()) < 200.2
        toa_s = (
            float(candidates.toa_sample_abs[0].item())
            * float(loader.plan.config.time_sample_ms) * 1.0e-3
        )
        assert np.isclose(toa_s, 2.6791791641, rtol=0.0, atol=1.0e-7)
        assert np.isclose(
            float(candidates.width_ms[0].item()),
            3.93216,
            rtol=0.0,
            atol=1.0e-9,
        )
        absolute_coarse_output = (
            int(candidates.source_chunk_index[0].item())
            * geometries[0].ntime
            + int(candidates.itime[0].item())
        )
        assert absolute_coarse_output == 127
        assert int(geometries[0].steady_state_it0[421].item()) == 430
        assert coverage == tuple((100, chunk) for chunk in range(10))
        assert startup == "authoritative" and complete
        assert len(grouped.events) == 1
        assert np.array_equal(
            cp.asnumpy(grouped.events.member_count), np.ones(1, np.int32)
        )
        assert cp.asnumpy(grouped.events.edge_flags).tolist() == [
            int(EdgeFlag.STARTUP_INCOMPLETE)
        ]

    lines = []
    original_print = runner.atomic_print
    runner.atomic_print = lines.append
    try:
        with tempfile.TemporaryDirectory(
                prefix="pirate-grouptest-catalog-") as tmp:
            catalog = os.path.join(tmp, "events.asdf")
            config_file = _runner_config(tmp)
            returned = runner.run_offline_grouper(
                _GROUPTEST,
                config_file,
                cuda_device_id=cuda_device_id,
                output=catalog,
            )
            assert returned == os.path.abspath(catalog)
            with asdf.open(catalog, lazy_load=False) as af:
                validate_trigger_catalog_tree(af.tree)
                assert len(af.tree["events"]["event_id"]) == 1
                assert len(af.tree["members"]["candidate_id"]) == 1
                assert np.asarray(
                    af.tree["events"]["edge_flags"]
                ).tolist() == [int(EdgeFlag.STARTUP_INCOMPLETE)]
                assert np.asarray(
                    af.tree["members"]["edge_flags"]
                ).tolist() == [int(EdgeFlag.STARTUP_INCOMPLETE)]
                assert (
                    af.tree["metadata"]["startup_by_beam"][0]["status"]
                    == "authoritative"
                )
    finally:
        runner.atomic_print = original_print
    event_lines = [line for line in lines if line.startswith("event=")]
    assert len(event_lines) == 1
    assert "dm=" in event_lines[0] and "toa=2.679179" in event_lines[0]
    assert "startup_incomplete=true" in event_lines[0]
    assert any("complete=true" in line for line in lines)

    # The measured S/N and decoded physical width are intentionally not
    # corrected for the conservative startup warning.


def test_terminal_output_names_startup_incomplete(cuda_device_id=0):
    """Terminal event rows expose the representative startup bit by name."""

    import cupy as cp

    with cp.cuda.Device(cuda_device_id):
        geometry = _geometry(cp)
        grouped = group_candidates(_candidates(cp, geometry, [{
            "tree": 0,
            "snr": 30.0,
            "dm": 100.0,
            "toa": 1000.0,
            "edge_flags": int(EdgeFlag.STARTUP_INCOMPLETE),
        }]), geometry)

        lines = []
        original_print = runner.atomic_print
        runner.atomic_print = lines.append
        try:
            runner._print_events(
                cp,
                grouped,
                event_offset=12,
                grouping_window_id=3,
                time_sample_ms=1.0,
            )
        finally:
            runner.atomic_print = original_print
        assert len(lines) == 1
        assert lines[0].startswith("event=12 ")
        assert "grouping_window=3" in lines[0]
        assert "grouping_timed_out=false" in lines[0]
        assert "startup_incomplete=true" in lines[0]
