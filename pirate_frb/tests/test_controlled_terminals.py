"""Session coordination and owned-process cleanup, without native GPU work."""
import json
import multiprocessing
import os
from pathlib import Path
import signal
import socket
import sys
import time

import pytest

from pirate_frb import ControlledTerminals as terminal


@pytest.fixture
def session(tmp_path, monkeypatch):
    from pirate_frb import ControlledExperiment
    monkeypatch.setattr(ControlledExperiment, "_source_identity", lambda: {"module_sha256": {"a": "b"}})
    (tmp_path / "_session").mkdir()
    manifest = dict(version=1, session_id="test-session", hostname=socket.gethostname(),
        boot_id=terminal._boot_id(), source={"module_sha256": {"a": "b"}},
        python_executable=str(Path(sys.executable).resolve()),
        cuda_visible_devices=os.environ.get("CUDA_VISIBLE_DEVICES"),
        startup_timeout_seconds=3, drain_timeout_seconds=3, write_timeout_seconds=3,
        duration_seconds=1, addresses={}, test_behavior="success")
    manifest["configuration_sha256"] = terminal._configuration_digest(manifest)
    terminal._write(tmp_path / "session.json", manifest)
    return tmp_path, manifest


def _fake_entry(output, manifest, role, updates, stop, parent_pid):
    terminal._parent_death_guard(parent_pid)
    behavior = manifest["test_behavior"]
    if behavior == "failure":
        updates.put(dict(state="failed", error="deliberate worker failure"))
        raise RuntimeError("deliberate worker failure")
    if behavior == "silent_exit":
        return
    if behavior == "stall":
        stop.wait(30)
    updates.put(dict(state="finished", result=None))


def test_process_identity_detects_missing_and_reused_pids(session):
    output, manifest = session
    assert terminal._process_token(os.getpid()) is not None
    assert terminal._process_token(999999999) is None
    status = terminal._claim(output, manifest, "grouper")
    status["process_token"] = "not-the-current-process"
    terminal._write(output / "_session/grouper.json", status)
    assert terminal.session_status(output)["state"] == "failed"


def test_duplicate_claim_does_not_change_running_owner(session):
    output, manifest = session
    status = terminal._claim(output, manifest, "replay")
    terminal._write(output / "_session/replay.json", status)
    before = (output / "_session/replay.json").read_bytes()
    with pytest.raises(RuntimeError, match="already launched"):
        terminal._claim(output, manifest, "replay")
    assert (output / "_session/replay.json").read_bytes() == before


@pytest.mark.parametrize("key,value,match", [
    ("hostname", "another-host", "another host"),
    ("boot_id", "previous-boot", "earlier boot"),
    ("python_executable", "/another/python", "same Python"),
    ("cuda_visible_devices", "a-different-mask", "CUDA_VISIBLE_DEVICES"),
    ("source", {"module_sha256": {"a": "changed"}}, "source changed"),
])
def test_session_rejects_host_environment_and_source_drift(session, key, value, match):
    output, manifest = session
    manifest[key] = value
    manifest["configuration_sha256"] = terminal._configuration_digest(manifest)
    terminal._write(output / "session.json", manifest)
    with pytest.raises(ValueError, match=match):
        terminal._manifest(output, check_source=True)


def test_wrong_session_status_is_never_accepted(session):
    output, manifest = session
    status = terminal._claim(output, manifest, "server")
    status["session_id"] = "old-session"
    terminal._write(output / "_session/server.json", status)
    with pytest.raises(ValueError, match="another session"):
        terminal._check_session(output, manifest)


def test_stop_before_launch_prevents_worker_start(session):
    output, manifest = session
    assert terminal.stop_session(output)["state"] == "stop_requested"
    assert terminal.session_status(output)["state"] == "failed"
    with pytest.raises(InterruptedError, match="stop requested"):
        terminal.run_component(output, "replay")
    assert not (output / "_session/replay.claim").exists()


def test_stop_after_success_keeps_completed_session(session):
    output, manifest = session
    for role in terminal.ROLES:
        terminal._write(output / "_session" / f"{role}.json",
            dict(role=role, session_id=manifest["session_id"], configuration_sha256=manifest["configuration_sha256"],
                 state="complete", exitcode=0))
    assert terminal.stop_session(output)["state"] == "complete"
    assert not (output / "_session/stop.json").exists()


def test_missing_peer_is_waiting_but_failed_peer_propagates(session):
    output, manifest = session
    terminal._check_session(output, manifest)
    assert terminal.session_status(output)["state"] == "waiting_or_running"
    terminal._write(output / "_session/capture.json", dict(role="capture", session_id=manifest["session_id"],
        configuration_sha256=manifest["configuration_sha256"],
        state="failed", error="disk write failed"))
    with pytest.raises(RuntimeError, match="capture failed: disk write failed"):
        terminal._check_session(output, manifest)


def test_deadline_switches_from_manual_startup_to_finite_replay(session):
    _, manifest = session
    assert terminal._deadline(manifest, {"started_monotonic_s": 20}, None) == 23
    assert terminal._deadline(manifest, {"started_monotonic_s": 20},
                              {"replay_started_monotonic_s": 22}) == 59


@pytest.mark.parametrize("behavior,expected", [("success", None), ("failure", "deliberate worker failure"),
                                               ("silent_exit", "without a completion acknowledgment"),
                                               ("stall", "deadline exceeded")])
def test_supervisor_requires_acknowledgment_and_reaps_its_worker(session, monkeypatch, behavior, expected):
    output, manifest = session
    manifest["test_behavior"] = behavior
    if behavior == "stall":
        manifest["startup_timeout_seconds"] = 0.5
    manifest["configuration_sha256"] = terminal._configuration_digest(manifest)
    terminal._write(output / "session.json", manifest)
    monkeypatch.setattr(terminal, "_worker_entry", _fake_entry)
    began = time.monotonic()
    if expected is None:
        terminal.run_component(output, "replay")
    else:
        with pytest.raises((RuntimeError, TimeoutError), match=expected):
            terminal.run_component(output, "replay")
    status = terminal._json(output / "_session/replay.json")
    assert status["state"] == ("complete" if expected is None else "failed")
    assert terminal._process_token(status["worker_pid"]) is None
    assert not multiprocessing.active_children()
    assert time.monotonic() - began < 12


def test_server_cannot_publish_success_without_peer_completion(session, monkeypatch):
    output, manifest = session
    monkeypatch.setattr(terminal, "_worker_entry", _fake_entry)
    with pytest.raises(RuntimeError, match="clean completion of all other components"):
        terminal.run_component(output, "server")
    assert terminal._json(output / "run.json")["complete"] is False


@pytest.mark.parametrize("kwargs", [{"cuda_device_id": True}, {"base_port": 65534},
                                    {"startup_timeout_seconds": 0}, {"max_lag_seconds": float("nan")}])
def test_prepare_rejects_invalid_settings_before_creating_directory(tmp_path, kwargs):
    output = tmp_path / "output"
    with pytest.raises(ValueError):
        terminal.prepare_session("missing-bundle", output, **kwargs)
    assert not output.exists()


def test_cli_dispatch_is_distinct_from_combined_experiment():
    from pirate_frb.__main__ import get_parser
    parser = get_parser()
    args = parser.parse_args(["experiment", "session", "grouper", "/tmp/session"])
    assert args.func is terminal.session_command
    assert args.session_command == "grouper"
    old = parser.parse_args(["experiment", "online", "bundle", "output"])
    assert old.func is not terminal.session_command

def test_modified_session_settings_are_rejected(session):
    output, manifest = session
    manifest["duration_seconds"] = 500
    terminal._write(output / "session.json", manifest)
    with pytest.raises(ValueError, match="configuration changed"):
        terminal._manifest(output)

def test_failed_status_write_cannot_skip_worker_cleanup(session, monkeypatch):
    output, manifest = session
    manifest["test_behavior"] = "failure"
    manifest["configuration_sha256"] = terminal._configuration_digest(manifest)
    terminal._write(output / "session.json", manifest)
    original = terminal._write
    def write(path, value):
        if value.get("state") == "failed":
            raise OSError("simulated full disk")
        return original(path, value)
    monkeypatch.setattr(terminal, "_write", write)
    monkeypatch.setattr(terminal, "_worker_entry", _fake_entry)
    with pytest.raises(RuntimeError, match="deliberate worker failure") as caught:
        terminal.run_component(output, "replay")
    assert any("persist component failure" in note for note in caught.value.__notes__)
    assert not multiprocessing.active_children()

@pytest.mark.parametrize("corrupt", [None, "missing_ack", "wrong_session", "nonzero_exit", "stop", "changed_settings"])
def test_archived_terminal_completion_is_required(session, corrupt):
    from pirate_frb.ControlledComparison import _validate_terminal_session
    output, manifest = session
    manifest["bundle_manifest_sha256"] = "bundle-hash"
    manifest["configuration_sha256"] = terminal._configuration_digest(manifest)
    terminal._write(output / "session.json", manifest)
    peers = {}
    for role in terminal.ROLES:
        status = dict(role=role, session_id=manifest["session_id"], state="complete", exitcode=0,
                      configuration_sha256=manifest["configuration_sha256"])
        terminal._write(output / "_session" / f"{role}.json", status)
        if role != "server":
            peers[role] = status
    run = dict(session_id=manifest["session_id"], bundle_manifest_sha256="bundle-hash",
               source=manifest["source"], component_status=peers)
    if corrupt in ("missing_ack", "wrong_session", "nonzero_exit"):
        path = output / "_session/grouper.json"
        status = terminal._json(path)
        if corrupt == "missing_ack": status["state"] = "drained"
        if corrupt == "wrong_session": status["session_id"] = "previous-session"
        if corrupt == "nonzero_exit": status["exitcode"] = 1
        terminal._write(path, status)
    elif corrupt == "stop":
        terminal._write(output / "_session/stop.json", {})
    elif corrupt == "changed_settings":
        manifest["duration_seconds"] = 999
        terminal._write(output / "session.json", manifest)
    if corrupt is None:
        assert _validate_terminal_session(output, run)["complete"]
    else:
        with pytest.raises(ValueError):
            _validate_terminal_session(output, run)
