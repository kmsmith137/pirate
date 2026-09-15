"""Portable v3 writer, loader, grouping, and catalog regressions.

Maps here are synthetic and use explicit producer Dcores and valid 1.5 tokens.
Raw-frame dedispersion is verified separately; historical 1.4 golden data stays
with the preserved 1.4 checkout rather than being relabelled for these tests.
"""
import os
import tempfile
from types import SimpleNamespace

import numpy as np

from ..ArgmaxMetadata import ARGMAX_ENCODING
from ..OfflineMapReader import OfflineMapReader
from ..Clustering import cluster_candidates
from ..BowtiePeakfinding import EdgeFlag
from ..TriggerCatalog import validate_trigger_catalog_tree
from ..run_offline_dedisperser import _write_snr_asdf, _validate_snr_asdf_tree
from .. import run_offline_grouper as runner
from ..pirate_pybind11 import DedispersionPlan
from .test_gpu_argmax_decoder import _make_plan
from .test_clustering import _candidates, _geometry

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



def _rewrite_map(path, change):
    """Modify only a freshly generated test file, never historical acquisitions."""
    import asdf
    with asdf.open(path, mode="rw", lazy_load=False, memmap=False) as af:
        change(af.tree)
        af.update()


def _write_toy_acquisition(directory, *, beams=(7, 11), chunks=(0, 1, 2),
                           include_start=True):
    """Exercise the real map writer with a small synthetic producer."""
    plan, dcores = _make_plan()
    # Deliberately choose a different first-tree granularity from _make_plan.
    # Its only authority is the producer's dd.Dcores member below.
    dcores = (2,) + dcores[1:]
    producer_plan = DedispersionPlan(plan.config)
    od = SimpleNamespace(
        plan=producer_plan, config=producer_plan.config,
        dd=SimpleNamespace(Dcores=dcores),
        ntrees=int(plan.ntrees), trees=plan.trees,
        nt_in=int(plan.nt_in), nfreq=int(plan.nfreq),
        time_sample_ms=float(plan.config.time_sample_ms),
    )
    itree = 0
    idm = int(plan.trees[itree].ndm_out) // 3
    itime = int(plan.trees[itree].nt_out) // 2
    token = (1 << 24) | (4 << 8)  # nonzero extra DM, temporal profile 4.
    assert int(plan.trees[0].dm_downsampling) >> int(
        plan.trees[0].frequency_subbands.pf_rank) > 1
    files = {}
    for beam in beams:
        for chunk in chunks:
            snr_maps = [
                np.zeros((int(t.ndm_out), int(t.nt_out)), dtype=np.float32)
                for t in plan.trees
            ]
            argmax_maps = [np.zeros(a.shape, dtype=np.uint32) for a in snr_maps]
            if beam == beams[0] and chunk == 1:
                snr_maps[itree][idm, itime] = 25.0
                argmax_maps[itree][idm, itime] = np.uint32(token)
            frame = SimpleNamespace(beam_id=beam, time_chunk_index=chunk)
            path = os.path.join(directory, f"frame_b{beam}_t{chunk}_snrmap.asdf")
            _write_snr_asdf(
                path, f"synthetic_frame_b{beam}_t{chunk}.asdf",
                frame, od, snr_maps, argmax_maps,
                producer_start_time_chunk_index=0,
            )
            if not include_start:
                _rewrite_map(path, lambda tree: tree.pop("producer_start_time_chunk_index"))
            files[(beam, chunk)] = path
    return SimpleNamespace(plan=plan, dcores=dcores, producer=od, files=files,
                           token=token, tree=itree, idm=idm, itime=itime)


def test_v3_writer_and_ragged_loader(cuda_device_id=0):
    """Writer Dcores and all ragged map shapes survive ASDF and GPU upload."""
    import asdf
    import cupy as cp
    with cp.cuda.Device(cuda_device_id), tempfile.TemporaryDirectory(
            prefix="pirate-v3-loader-") as tmp:
        fixture = _write_toy_acquisition(tmp)
        loader = OfflineMapReader(tmp, cuda_device_id=cuda_device_id)
        assert loader.ntrees == 6
        assert loader.beam_ids == (7, 11)
        assert loader.dcores == fixture.dcores
        assert loader.argmax_encoding == ARGMAX_ENCODING
        assert loader.plan_yaml == fixture.producer.plan.to_yaml_string()
        assert loader.producer_start_by_beam == {7: 0, 11: 0}
        assert len(set(loader.tree_shapes)) > 1
        for path in fixture.files.values():
            with asdf.open(path) as af:
                assert af.tree["format_version"] == 3
                assert tuple(af.tree["dcores"]) == fixture.dcores
                assert af.tree["argmax_encoding"] == ARGMAX_ENCODING
        maps = loader.load_beam_chunk((7, 11), 1)
        assert maps.beam_ids == (7, 11) and maps.source_chunk_index == 1
        for snr, tokens, shape in zip(
                maps.snr_by_tree, maps.argmax_by_tree, loader.tree_shapes):
            assert snr.shape == tokens.shape == (2,) + shape
            assert snr.dtype == cp.float32 and tokens.dtype == cp.uint32
        assert int(maps.argmax_by_tree[0][0, fixture.idm, fixture.itime].item()) == fixture.token


def test_v3_saved_map_to_catalog(cuda_device_id=0):
    """A nonzero-mu candidate traverses the saved-map pipeline and keeps provenance."""
    import asdf
    import cupy as cp
    with cp.cuda.Device(cuda_device_id), tempfile.TemporaryDirectory(
            prefix="pirate-v3-catalog-") as tmp:
        fixture = _write_toy_acquisition(tmp)
        catalog = os.path.join(tmp, "events.asdf")
        config_file = _runner_config(tmp)
        lines = []
        original_print = runner.atomic_print
        runner.atomic_print = lines.append
        try:
            returned = runner.run_offline_grouper(
                tmp, config_file, cuda_device_id=cuda_device_id, output=catalog,
            )
        finally:
            runner.atomic_print = original_print
        assert returned == os.path.abspath(catalog)
        integer = fixture.plan.decode_argmax(
            fixture.token, fixture.tree, fixture.dcores[fixture.tree],
            fixture.idm, fixture.itime,
        )
        freq_lo, freq_hi, dm, toa, width = fixture.plan.decode_argmax2(
            fixture.tree, *integer,
        )
        with asdf.open(catalog, lazy_load=False) as af:
            validate_trigger_catalog_tree(af.tree)
            assert af.tree["format_version"] == 3
            producer = af.tree["metadata"]["producer"]
            assert tuple(producer["dcores"]) == fixture.dcores
            assert producer["argmax_encoding"] == ARGMAX_ENCODING
            assert producer["plan_yaml"] == fixture.producer.plan.to_yaml_string()
            assert af.tree["metadata"]["processing"]["complete"]
            events, members = af.tree["events"], af.tree["members"]
            assert len(events["event_id"]) == len(members["candidate_id"]) == 1
            assert list(members["argmax_token"]) == [fixture.token]
            assert list(members["source_chunk_index"]) == [1]
            assert list(members["tree"]) == [0]
            assert list(members["beam_id"]) == [7]
            assert list(events["snr"]) == [25.0]
            assert np.allclose(events["dm"], [dm], rtol=0, atol=1e-9)
            assert np.allclose(events["toa_sample_abs"], [fixture.plan.nt_in + toa],
                               rtol=0, atol=1e-9)
            assert np.allclose(events["width_samp"], [width], rtol=0, atol=1e-9)
            assert len(af.tree["coverage"]["beam_id"]) == 6
        assert len([line for line in lines if line.startswith("event=")]) == 1


def test_v3_unknown_startup_requires_explicit_policy(cuda_device_id=0):
    """Missing start stays unknown even with otherwise complete v3 metadata."""
    import cupy as cp
    with cp.cuda.Device(cuda_device_id), tempfile.TemporaryDirectory(
            prefix="pirate-v3-startup-") as tmp:
        _write_toy_acquisition(tmp, include_start=False)
        loader = OfflineMapReader(tmp, cuda_device_id=cuda_device_id)
        assert loader.producer_start_by_beam == {7: None, 11: None}
        config_file = _runner_config(tmp, snr_threshold=1e30)
        try:
            runner.run_offline_grouper(
                tmp, config_file, max_chunks=1, cuda_device_id=cuda_device_id,
            )
        except ValueError as exc:
            assert "producer-start provenance is missing" in str(exc)
        else:
            raise AssertionError("unknown startup was silently assumed")
        lines = []
        original_print = runner.atomic_print
        runner.atomic_print = lines.append
        try:
            runner.run_offline_grouper(
                tmp, config_file, max_chunks=1, cuda_device_id=cuda_device_id,
                assume_steady_state=True,
            )
        finally:
            runner.atomic_print = original_print
        assert any("startup=assumed" in line and "complete=false" in line for line in lines)


def test_v3_rejects_legacy_and_corrupt_decoder_metadata():
    """Both writer validation and the loader reject incomplete decoder provenance."""
    import asdf
    mutations = (
        (lambda t: t.update(format_version=2, config_yaml="legacy YAML"), "format_version=3"),
        (lambda t: t.update(format_version=3.0), "format_version=3"),
        (lambda t: t.pop("argmax_encoding"), "argmax_encoding"),
        (lambda t: t.update(argmax_encoding="t8-p8-m16"), "argmax_encoding"),
        (lambda t: t.pop("dcores"), "dcores"),
        (lambda t: t.update(dcores=[]), "dcores"),
        (lambda t: t["dcores"].__setitem__(0, True), "dcores"),
        (lambda t: t["dcores"].__setitem__(0, 3), "dcores"),
        (lambda t: t["dcores"].__setitem__(0, 256), "Dout"),
        (lambda t: t.update(time_sample_ms=2.0), "time_sample_ms"),
    )
    for mutation, expected in mutations:
        with tempfile.TemporaryDirectory(prefix="pirate-v3-invalid-") as tmp:
            fixture = _write_toy_acquisition(tmp, beams=(7,), chunks=(0,))
            path = fixture.files[(7, 0)]
            _rewrite_map(path, mutation)
            with asdf.open(path) as af:
                try:
                    _validate_snr_asdf_tree(af.tree)
                except ValueError as exc:
                    assert expected in str(exc), str(exc)
                else:
                    raise AssertionError("writer validator accepted malformed metadata")
            try:
                OfflineMapReader(tmp)
            except ValueError as exc:
                assert path in str(exc) and expected in str(exc), str(exc)
            else:
                raise AssertionError("loader accepted malformed metadata")


def test_v3_rejects_changed_producer_dcores():
    """Equal YAML and array shapes cannot hide a different producer time grid."""
    with tempfile.TemporaryDirectory(prefix="pirate-v3-mixed-") as tmp:
        fixture = _write_toy_acquisition(tmp, beams=(7,), chunks=(0, 1))
        _rewrite_map(fixture.files[(7, 1)],
                     lambda tree: tree["dcores"].__setitem__(0, 1))
        try:
            OfflineMapReader(tmp)
        except ValueError as exc:
            assert "Dcores" in str(exc) and "differ from the first file" in str(exc)
        else:
            raise AssertionError("mixed producer Dcores accepted")


def test_v3_writer_requires_actual_producer_dcores():
    """No file is written when the producer has not supplied kernel metadata."""
    with tempfile.TemporaryDirectory(prefix="pirate-v3-uninitialized-") as tmp:
        fixture = _write_toy_acquisition(tmp, beams=(7,), chunks=(0,))
        fixture.producer.dd = None
        snrs = [np.zeros((int(t.ndm_out), int(t.nt_out)), np.float32)
                for t in fixture.plan.trees]
        tokens = [np.zeros(a.shape, np.uint32) for a in snrs]
        output = os.path.join(tmp, "must-not-exist.asdf")
        try:
            _write_snr_asdf(
                output, "synthetic.asdf",
                SimpleNamespace(beam_id=7, time_chunk_index=0),
                fixture.producer, snrs, tokens, producer_start_time_chunk_index=0,
            )
        except ValueError as exc:
            assert "GpuDedisperser.Dcores" in str(exc)
        else:
            raise AssertionError("writer invented producer metadata")
        assert not os.path.exists(output)


def test_terminal_output_names_startup_incomplete(cuda_device_id=0):
    """Terminal event rows expose the representative startup bit by name."""

    import cupy as cp

    with cp.cuda.Device(cuda_device_id):
        geometry = _geometry(cp)
        grouped = cluster_candidates(_candidates(cp, geometry, [{
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
