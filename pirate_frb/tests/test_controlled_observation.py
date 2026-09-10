"""CPU checks for the finite CHORD observation contract and startup plan."""
import json
from pathlib import Path

import pytest
import yaml

from pirate_frb import ControlledObservation as co

ROOT = Path(__file__).resolve().parents[2]
EXAMPLE = ROOT / "configs/experiments/chord_replay.yml"


def _config(tmp_path, mutate=None):
    config = co._read_yaml(EXAMPLE)
    config["metadata"] = str(ROOT / "configs/xengine_metadata.yml")
    config["dedispersion"] = str(ROOT / "configs/dedispersion/chord_sb2_et.yml")
    if mutate:
        mutate(config)
    path = tmp_path / "experiment.yml"
    path.write_text(yaml.safe_dump(config))
    return path


@pytest.fixture
def prepared(tmp_path):
    root = tmp_path / "bundle"
    data = co.prepare_controlled_observation(_config(tmp_path), root)
    return root, data


def test_chord_prepare_resolves_actual_cadence_startup_and_separated_bursts(prepared):
    root, data = prepared
    assert data["state"] == "prepared"
    assert data["beam_ids"] == [100, 101]
    assert data["metadata"]["zone_nfreq"] == [8192, 8192, 6144, 2048, 3584]
    assert data["time_sample_ms"] == .9984
    assert data["nchunks"] == 108
    assert data["duration_seconds"] == pytest.approx(220.8301056)
    assert data["frame_entries"] == []
    truth = json.loads((root / "injections.json").read_text())["bursts"]
    assert [burst["dm"] for burst in truth] == [100., 2000.]
    assert truth[0]["sample_end"] < truth[1]["sample_start"]
    assert truth[1]["dispersion_sweep_seconds"] == pytest.approx(88.507904)
    assert [item["tree_index"] for item in truth[1]["expected_tree_arrivals"]] == [1, 2]
    plan = json.loads((root / "plan_check.json").read_text())
    assert plan["ringbuf_nchunks"] == 30
    assert 0 < truth[1]["planned_capture_margin_seconds"] < 15
    for burst in truth:
        for target in burst["expected_tree_arrivals"]:
            tree = plan["trees"][target["tree_index"]]
            assert target["arrival_seconds"] > tree["conservative_ready_seconds"] + plan["chunk_seconds"]
    assert not list((root / "acq").iterdir())
    with pytest.raises(FileExistsError):
        co.prepare_controlled_observation(EXAMPLE, root)


def test_runtime_loader_never_reads_injection_truth(prepared, monkeypatch):
    root, _ = prepared
    original = co._read_json

    def guarded(path):
        assert Path(path).name != "injections.json"
        return original(path)

    monkeypatch.setattr(co, "_read_json", guarded)
    (root / "injections.json").write_text("inaccessible as scientific input")
    data = co.load_experiment_bundle(root, require_complete=False)
    assert "bursts" not in data
    with pytest.raises(ValueError, match="not complete"):
        co.load_experiment_bundle(root)


@pytest.mark.parametrize("mutation,match", [
    (lambda c: c["bursts"][1].update(dm=800.), "does not exercise"),
    (lambda c: c["bursts"][0].update(toa_seconds=10.), "too early"),
    (lambda c: c["bursts"][1].update(toa_seconds=165.), "non-overlapping"),
    (lambda c: c["capture"].update(buffer_seconds=100.), "must exceed retention"),
    (lambda c: c["observation"].update(duration_seconds=190.), "clipped"),
    (lambda c: c["observation"].update(beam_ids=[100]), "twice the active"),
    (lambda c: c["grouper"]["execution"].update(timeout_ms=400), "disable grouping timeouts"),
    (lambda c: c["bursts"][0].update(reference_frequency_MHz=400), "referenced to 300"),
    (lambda c: c["observation"].update(noise_seed=True), "must be numeric"),
])
def test_unsafe_or_ambiguous_experiment_rejected_before_creation(tmp_path, mutation, match):
    with pytest.raises(ValueError, match=match):
        co.prepare_controlled_observation(_config(tmp_path, mutation), tmp_path / "bundle")
    assert not (tmp_path / "bundle").exists()


def _complete_one_chunk(root):
    data = co._read_json(root / co.MANIFEST_NAME)
    data.update(state="complete", nchunks=1,
                duration_seconds=data["samples_per_chunk"] * data["time_sample_ms"] * .001)
    for beam in data["beam_ids"]:
        relative = f"acq/frame_b{beam}_t0.asdf"
        path = root / relative
        path.write_bytes(f"mock payload {beam}".encode())
        data["frame_entries"].append(dict(beam_id=beam, time_chunk_index=0,
            path=relative, sha256=co._sha256(path), size_bytes=path.stat().st_size))
    co._json_write(root / co.MANIFEST_NAME, data)
    return data


def test_complete_bundle_integrity_and_coverage(prepared):
    root, _ = prepared
    manifest = _complete_one_chunk(root)
    assert len(co.load_experiment_bundle(root, verify_hashes=True)["frame_entries"]) == 2
    path = root / manifest["frame_entries"][0]["path"]
    path.write_bytes(b"x" * path.stat().st_size)
    with pytest.raises(ValueError, match="hash mismatch"):
        co.load_experiment_bundle(root, verify_hashes=True)
    manifest["frame_entries"].reverse()
    co._json_write(root / co.MANIFEST_NAME, manifest)
    with pytest.raises(ValueError, match="coverage/order"):
        co.load_experiment_bundle(root)


def test_bundle_rejects_config_changes_and_path_escape(prepared):
    root, _ = prepared
    manifest = co._read_json(root / co.MANIFEST_NAME)
    manifest["metadata_file"] = "../metadata.yml"
    co._json_write(root / co.MANIFEST_NAME, manifest)
    with pytest.raises(ValueError, match="escapes"):
        co.load_experiment_bundle(root, require_complete=False)
    manifest["metadata_file"] = "metadata.yml"
    co._json_write(root / co.MANIFEST_NAME, manifest)
    with (root / "metadata.yml").open("a") as stream:
        stream.write("\n# changed\n")
    with pytest.raises(ValueError, match="configuration hash mismatch"):
        co.load_experiment_bundle(root, require_complete=False)


def test_generation_failure_latches_bundle_and_never_resumes(prepared, monkeypatch):
    root, _ = prepared

    def failure(*args, **kwargs):
        raise RuntimeError("injected writer failure")

    monkeypatch.setattr(co.subprocess, "run", failure)
    with pytest.raises(RuntimeError, match="writer failure"):
        co.generate_controlled_observation(root)
    manifest = co._read_json(root / co.MANIFEST_NAME)
    assert manifest["state"] == "failed"
    assert (root / "generation.lock").exists()
    with pytest.raises(ValueError, match="cannot be used"):
        co.generate_controlled_observation(root)
