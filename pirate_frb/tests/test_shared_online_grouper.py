"""Adapter parity, IPC ownership, and finite-observation regressions."""
from contextlib import contextmanager
import json
import os
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest
import yaml

from ..OnlineGrouper import representative_events, validate_live_handshake


def _columns():
    return dict(beam_id=np.array([100, 901], np.int32),
                toa_sample_abs=np.array([1024.25, 1300.5]),
                dm=np.array([100., 2000.]), snr=np.array([30., 40.]),
                width_ms=np.array([2., 4.]), freq_lo_MHz=np.array([300., 416.]),
                freq_hi_MHz=np.array([1500., 1500.]), tree=np.array([0, 3], np.int32))


def test_representatives_use_absolute_arrival_not_owner_chunk():
    events = representative_events(_columns(), owner_source_chunk=3, nt_in=256, seq_per_sample=4)
    assert (events.chunk_fpga_start, events.chunk_fpga_end) == (3072, 4096)
    # Both arrivals can lie beyond the owner window; do not add its start again.
    assert events.fpga_timestamps.tolist() == [4097, 5202]
    assert events.tree_index.dtype == np.int32
    assert events.tree_index.tolist() == [0, 3]
    assert events.beam_ids.tolist() == [100, 901]
    assert np.all(events.rfi_probs == 0)


def test_representatives_reject_invalid_timestamp_and_keep_empty_messages():
    columns = _columns()
    columns["toa_sample_abs"][0] = float("nan")
    with pytest.raises(ValueError, match="finite"):
        representative_events(columns, owner_source_chunk=0, nt_in=256, seq_per_sample=4)
    columns = {name: value[:0] for name, value in _columns().items()}
    events = representative_events(columns, owner_source_chunk=17, nt_in=256, seq_per_sample=4)
    assert len(events) == 0 and events.chunk_fpga_start == 17408
    with pytest.raises(OverflowError):
        representative_events(columns, owner_source_chunk=2**63, nt_in=256, seq_per_sample=4)


def test_handshake_checks_nondense_beams_and_nonzero_start():
    g = SimpleNamespace(initial_chunk=17, nt_in=256,
                        xengine_metadata=SimpleNamespace(beam_ids=[100, 901]),
                        dedispersion_config=SimpleNamespace(time_sample_ms=0.9984))
    bundle = dict(samples_per_chunk=256, beam_ids=[100, 901], time_sample_ms=0.9984, metadata={})
    validate_live_handshake(g, bundle, expected_start_chunk=17, expected_nchunks=3)
    with pytest.raises(ValueError, match="start"):
        validate_live_handshake(g, bundle, expected_start_chunk=0, expected_nchunks=3)
    g.xengine_metadata.beam_ids = [901, 100]
    with pytest.raises(ValueError, match="beam"):
        validate_live_handshake(g, bundle, expected_start_chunk=17, expected_nchunks=3)


def _canonical_catalog(path):
    import asdf
    from ..TriggerCatalog import validate_trigger_catalog_tree
    with asdf.open(path, lazy_load=False) as af:
        tree = validate_trigger_catalog_tree(af.tree)
        events, members = tree["events"], tree["members"]
        def source(columns, i):
            return tuple(int(columns[name][i]) for name in
                         ("beam_id", "source_chunk_index", "tree", "idm", "itime"))
        result = {}
        for i, event_id in enumerate(events["event_id"]):
            indices = np.flatnonzero(members["event_id"] == event_id)
            values = tuple(float(events[name][i]) for name in
                           ("dm", "toa_sample_abs", "snr", "width_ms", "freq_lo_MHz", "freq_hi_MHz"))
            member_rows = tuple(sorted((source(members, int(j)),
                                        int(members["argmax_token"][j]),
                                        int(members["edge_flags"][j])) for j in indices))
            result[source(events, i)] = (values, member_rows, int(events["edge_flags"][i]))
        coverage = set(zip(tree["coverage"]["beam_id"].tolist(),
                           tree["coverage"]["source_chunk_index"].tolist()))
        return result, coverage, bool(tree["metadata"]["processing"]["complete"])


@pytest.mark.parametrize("interrupt", [False, True])
def test_live_recycled_maps_match_offline_and_flush_only_real_observation(tmp_path, monkeypatch, interrupt):
    """Actual shared GPU processing survives immediate producer-buffer reuse."""
    import cupy as cp
    from .. import rpc
    from .. import ControlledObservation
    from ..OnlineGrouper import run_online_grouper
    from ..SharedGrouper import GrouperSetup, CatalogRecorder
    from ..FrbOfflineGrouper import BeamBatch
    from ..OfflineGrouperConfig import load_offline_grouper_config
    from ..run_offline_grouper import _extract_beam_batch
    from ..ArgmaxMetadata import ARGMAX_ENCODING
    from .test_gpu_argmax_decoder import _make_plan

    device = int(os.environ.get("PIRATE_TEST_GPU", "0"))
    with cp.cuda.Device(device):
        plan, dcores = _make_plan()
        config_yaml = plan.config.to_yaml_string()
        plan_yaml = plan.to_yaml_string()
        first, count, beam_ids = 17, 3, (100, 901)
        cfg = dict(peakfinding=dict(snr_threshold=10., dm_reach=1, waist_bins=0),
                   grouping=dict(halo_size=2, dm_tolerance=1.5, time_tolerance=1.5),
                   execution=dict(beam_batch_size=1, timeout_ms=0, timeout_policy="discard"))
        cfg_path = tmp_path / "grouper.yml"
        cfg_path.write_text(yaml.safe_dump(cfg))
        configuration = load_offline_grouper_config(cfg_path)
        setup = GrouperSetup(plan, dcores, configuration, cuda_device_id=device)
        raw = {}
        for chunk in range(first, first + count):
            for beam in beam_ids:
                maps = [np.zeros((1, int(t.ndm_out), int(t.nt_out)), np.float32) for t in plan.trees]
                tokens = [np.zeros(m.shape, np.uint32) for m in maps]
                t = plan.trees[0]
                dm = int(t.ndm_out) // 2
                # Ordinary event, a seam competitor, and an unresolved final edge.
                if chunk == first:
                    maps[0][0, dm, int(t.nt_out) // 2] = 25. + (beam == 901)
                    maps[0][0, dm + 20, -1] = 26.
                if chunk == first + 1:
                    maps[0][0, dm + 20, 0] = 31.
                if chunk == first + count - 1:
                    maps[0][0, dm, -1] = 35.
                raw[beam, chunk] = maps, tokens
        class Loader:
            def load_beam_chunk(self, ids, chunk):
                snr, token = raw[ids[0], chunk]
                return SimpleNamespace(snr_by_tree=tuple(cp.asarray(a) for a in snr),
                                       argmax_by_tree=tuple(cp.asarray(a) for a in token))
        recorder = CatalogRecorder(setup, config_yaml=config_yaml, plan_yaml=plan_yaml,
                                   argmax_encoding=ARGMAX_ENCODING)
        processors = []
        for beam in beam_ids:
            coverage, startup, complete = _extract_beam_batch(
                Loader(), BeamBatch((beam,), tuple(range(first, first + count)), first),
                setup.geometries, setup.decoder, setup.grouping_geometry,
                threshold=10., halo_size=2, timeout_ms=0, timeout_policy="discard",
                max_chunks=None, assume_steady_state=False, grouping_config=setup.grouping_config,
                consume_window=lambda window, beam=beam: recorder.consume((beam,), window))
            processors.append(SimpleNamespace(coverage=coverage, beam_ids=(beam,), assumed=False,
                                              producer_start_chunk=first, startup_status=startup))
            assert complete
        offline_path = tmp_path / "offline.asdf"
        recorder.write(offline_path, processors, complete=True)
        bundle = dict(initial_chunk=first, nchunks=count, samples_per_chunk=int(plan.nt_in),
                      beam_ids=list(beam_ids), time_sample_ms=float(plan.config.time_sample_ms),
                      metadata={}, grouper_config_path=str(cfg_path))
        monkeypatch.setattr(ControlledObservation, "load_experiment_bundle", lambda *a, **k: bundle)
        seen = []
        class FakeLiveGrouper:
            cuda_device_id = device
            initial_chunk = first
            nt_in = int(plan.nt_in)
            nbatches = 2
            beams_per_batch = 1
            dedispersion_plan = plan
            dedispersion_config = plan.config
            dedispersion_config_yaml_string = config_yaml
            dedispersion_plan_yaml_string = plan_yaml
            xengine_metadata = SimpleNamespace(beam_ids=beam_ids, beamset=0, seq_per_frb_time_sample=4)
            is_stopped = True
            def __init__(self, *a, **k):
                self.dcores = dcores
                self.storage = [cp.zeros(a.shape, cp.float32) for a in raw[beam_ids[0], first][0]]
                self.tokens = [cp.zeros(a.shape, cp.uint32) for a in raw[beam_ids[0], first][1]]
            def __enter__(self):
                return self
            def __exit__(self, *args):
                return False
            @contextmanager
            def get_output(self, relative, ibatch):
                assert relative < count, "transport tails must not enter scientific processing"
                if interrupt and relative == 1:
                    raise RuntimeError("simulated producer disconnect")
                chunk, beam = first + relative, beam_ids[ibatch]
                seen.append((chunk, beam))
                host_snr, host_tokens = raw[beam, chunk]
                for target, source in zip(self.storage, host_snr):
                    target.set(source)
                for target, source in zip(self.tokens, host_tokens):
                    target.set(source)
                try:
                    yield SimpleNamespace(ichunk_fpga_based=chunk, ibeam=ibatch,
                                          out_max=self.storage, out_argmax=self.tokens)
                finally:
                    cp.cuda.get_current_stream().synchronize()
                    for array in self.storage:
                        array.fill(-999.)
                    for array in self.tokens:
                        array.fill(np.uint32(0xffffffff))
        monkeypatch.setattr(rpc, "FrbGrouper", FakeLiveGrouper)
        online_path = tmp_path / "online.asdf"
        if interrupt:
            with pytest.raises(RuntimeError, match="simulated producer disconnect"):
                run_online_grouper(tmp_path, online_path, "unused",
                                   expected_start_chunk=first, expected_nchunks=count)
            report = json.loads(Path(str(online_path) + ".online.json").read_text())
            assert not report["complete"] and not online_path.exists()
            assert len(report["chunks"]) == len(beam_ids)
            return
        progress = []
        run_online_grouper(tmp_path, online_path, "unused", expected_start_chunk=first, expected_nchunks=count, progress=progress.append)
        assert [p["completed_chunks"] for p in progress if p["phase"] == "chunk"] == list(range(1, count + 1))
        assert sum(p["events"] for p in progress if p["phase"] == "window") > 0
        import asdf
        with asdf.open(online_path) as af:
            assert af.tree["metadata"]["pipeline"] == "online"
        offline = _canonical_catalog(offline_path)
        online = _canonical_catalog(online_path)
        assert offline == online
        assert offline[0], "test requires actual detections"
        assert len(offline[1]) == len(beam_ids) * count and offline[2]
        assert any(key[1] == first + count - 1 for key in offline[0]), "physical EOF must flush the final edge"
        assert seen == [(chunk, beam) for chunk in range(first, first + count) for beam in beam_ids]
        report = json.loads(Path(str(online_path) + ".online.json").read_text())
        assert report["complete"] and len(report["chunks"]) == count * len(beam_ids)


def test_processor_rejects_gaps_and_interruption_does_not_flush():
    import cupy as cp
    from ..SharedGrouper import StreamingGrouper
    from ..OfflineCandidateGrouper import GroupingConfig
    from ..Peakfinders import GpuRawCandidates
    from .test_offline_candidate_grouper import _geometry
    from .test_offline_grouper_streaming import _RawDecoder
    device = int(os.environ.get("PIRATE_TEST_GPU", "0"))
    with cp.cuda.Device(device):
        geometry = _geometry(cp)
        instances = []
        class Extractor:
            def __init__(self, *a, **k):
                self.flushed = False
                instances.append(self)
            def process_chunk(self, *a, **k):
                return GpuRawCandidates.empty()
            def flush(self):
                self.flushed = True
                return GpuRawCandidates.empty()
        processor = StreamingGrouper(
            (SimpleNamespace(ntime=8, time_radius=1),), _RawDecoder(cp, geometry), geometry,
            beam_ids=(901,), producer_start_chunk=None, threshold=10., halo_size=2,
            timeout_ms=0, timeout_policy="discard", grouping_config=GroupingConfig(),
            consume_window=lambda _: pytest.fail("prefix cannot emit a final window"),
            assume_steady_state=True, extractor_factory=Extractor)
        processor.process_chunk((None,), (None,), 17)
        with pytest.raises(ValueError, match="consecutive"):
            processor.process_chunk((None,), (None,), 19)
        processor.finish(physical_end=False)
        assert not processor.complete and not instances[0].flushed
        with pytest.raises(RuntimeError, match="finished"):
            processor.process_chunk((None,), (None,), 18)


def test_handshake_rejects_changed_scientific_configuration(tmp_path):
    config = dict(time_sample_ms=1.0, beams_per_gpu=2, dtype="float16",
                  primary_trees=[dict(num_early_triggers=1, max_width=16)])
    path = tmp_path / "dedispersion.yml"
    path.write_text(yaml.safe_dump(config))
    live_config = dict(config, time_sample_ms=0.9984)
    g = SimpleNamespace(initial_chunk=0, nt_in=256,
                        xengine_metadata=SimpleNamespace(beam_ids=[100, 901]),
                        dedispersion_config=SimpleNamespace(time_sample_ms=0.9984),
                        dedispersion_config_yaml_string=yaml.safe_dump(live_config))
    bundle = dict(samples_per_chunk=256, beam_ids=[100, 901], time_sample_ms=0.9984,
                  metadata={}, dedispersion_config_path=str(path))
    validate_live_handshake(g, bundle, expected_start_chunk=0, expected_nchunks=3)
    live_config["primary_trees"] = [dict(num_early_triggers=0, max_width=16)]
    g.dedispersion_config_yaml_string = yaml.safe_dump(live_config)
    with pytest.raises(ValueError, match="num_early_triggers"):
        validate_live_handshake(g, bundle, expected_start_chunk=0, expected_nchunks=3)


@pytest.mark.parametrize("noise_yaml", [1.0, [1.0, 1.0]])
def test_handshake_normalizes_scalar_and_vector_noise_with_native_metadata(tmp_path, noise_yaml):
    from ..pirate_pybind11 import XEngineMetadata
    native = XEngineMetadata.make_fiducial([4, 8], [300., 600., 1500.], [100, 901], 0.9984)
    canonical = yaml.safe_load(native.to_yaml_string())
    canonical["noise_variance"] = noise_yaml
    metadata_path = tmp_path / "metadata.yml"
    metadata_path.write_text(yaml.safe_dump(canonical))
    # The real sender/server use this same parser; even scalar input becomes a
    # per-zone vector in the handshake and in producer execution evidence.
    actual = XEngineMetadata.from_yaml_file(str(metadata_path))
    assert list(actual.noise_variance) == [1.0, 1.0]
    cadence = float(actual.dt_ns_per_seq * actual.seq_per_frb_time_sample / 1.e6)
    bundle = dict(initial_chunk=7, samples_per_chunk=256, beam_ids=[100, 901],
                  time_sample_ms=cadence, metadata=canonical, metadata_path=str(metadata_path))
    g = SimpleNamespace(initial_chunk=7, nt_in=256, xengine_metadata=actual,
                        dedispersion_config=SimpleNamespace(time_sample_ms=cadence))
    validate_live_handshake(g, bundle, expected_start_chunk=7, expected_nchunks=3)
    actual.noise_variance = [1.0, 2.0]
    with pytest.raises(ValueError, match="noise_variance"):
        validate_live_handshake(g, bundle, expected_start_chunk=7, expected_nchunks=3)
