"""Failure-path checks that do not initialize CUDA or start a native server."""
import hashlib
import json
import multiprocessing
import queue
import threading
import time
from types import SimpleNamespace

import numpy as np
import pytest

from .. import ControlledExperiment as experiment


def test_partial_server_construction_stops_every_created_resource(monkeypatch):
    calls = []
    def resource(name, fail=False):
        def stop():
            calls.append(name)
            if fail:
                raise RuntimeError(name + " cleanup failed")
        return SimpleNamespace(stop=stop)
    def fail_build(self, *args):
        self.server = resource("server", fail=True)
        self.receivers = [resource("receiver", fail=True)]
        self.allocator = resource("allocator")
        self.file_writer = resource("writer")
        raise ValueError("construction failed")
    monkeypatch.setattr(experiment._ExperimentServer, "_build", fail_build)
    with pytest.raises(ValueError, match="construction failed") as raised:
        experiment._ExperimentServer(None, None, None, None, None, None, None)
    assert calls == ["server", "receiver", "allocator", "writer"]
    assert len(raised.value.controlled_cleanup_errors) == 2
    assert any("Partial server construction" in note for note in raised.value.__notes__)


def test_cleanup_failures_do_not_skip_child_kill_or_queue_join():
    calls = []
    def fail(name):
        calls.append(name)
        raise RuntimeError(name)
    capture = SimpleNamespace(close=lambda: fail("capture.close"))
    server = SimpleNamespace(stop=lambda: fail("server.stop"),
                             server=SimpleNamespace(poll_from_python=lambda **k: fail("server.poll")))
    event = SimpleNamespace(set=lambda: calls.append("stop_event.set"))
    class Child:
        pid = 123
        exitcode = None
        alive = True
        joins = 0
        def join(self, timeout):
            self.joins += 1
            calls.append(f"join{self.joins}")
            if self.joins == 1:
                raise RuntimeError("initial join failed")
        def is_alive(self):
            return self.alive
        def terminate(self):
            fail("terminate")
        def kill(self):
            calls.append("kill")
            self.alive, self.exitcode = False, -9
    child = Child()
    status = SimpleNamespace(close=lambda: fail("queue.close"),
                             join_thread=lambda: calls.append("queue.join"))
    report = dict(state="complete", complete=True)
    errors = experiment._cleanup_online_resources(capture, server, event, child, status, report)
    assert not child.alive and child.joins == 3
    assert calls == ["capture.close", "server.stop", "stop_event.set", "join1", "terminate",
                     "join2", "kill", "join3", "server.poll", "queue.close", "queue.join"]
    assert errors and report["state"] == "failed" and not report["complete"]


class _Event:
    def __init__(self):
        self.value = False
    def set(self):
        self.value = True
    def is_set(self):
        return self.value
    def wait(self, timeout=None):
        return self.value


class _Queue:
    def __init__(self):
        self.closed = self.joined = False
    def get_nowait(self):
        raise queue.Empty
    def close(self):
        self.closed = True
    def join_thread(self):
        self.joined = True


class _Context:
    def __init__(self):
        self.events = []
        self.status = _Queue()
        self.child = None
    def Event(self):
        event = _Event()
        self.events.append(event)
        return event
    def Queue(self):
        return self.status
    def Process(self, **kwargs):
        context = self
        class Child:
            pid = None
            exitcode = None
            alive = False
            def start(self):
                self.pid, self.alive = 123, True
            def join(self, timeout):
                if context.events[0].is_set():
                    self.alive, self.exitcode = False, 0
            def is_alive(self):
                return self.alive
            def terminate(self):
                self.alive, self.exitcode = False, -15
            def kill(self):
                self.alive, self.exitcode = False, -9
        self.child = Child()
        return self.child


def _configure_run(tmp_path, monkeypatch, *, fail_replay=False, fail_capture_close=False,
                   fail_construction=False, context=None):
    from .. import ControlledObservation, ReplayObservation
    calls = []
    config = tmp_path / "capture.yml"
    config.write_text("{}\n")
    bundle = dict(bundle_dir=str(tmp_path), capture_config_path=str(config), manifest_sha256="abc",
                  metadata=dict(noise_variance=[1.0]), samples_per_chunk=256, duration_seconds=6.0)
    policy = SimpleNamespace(buffer_seconds=60., write_timeout_seconds=20.)
    monkeypatch.setattr(ControlledObservation, "load_experiment_bundle", lambda *a, **k: bundle)
    monkeypatch.setattr(experiment, "CapturePolicy", SimpleNamespace(from_dict=lambda value: policy))
    monkeypatch.setattr(experiment, "_source_identity", lambda: {"unit_test": True})
    monkeypatch.setattr(experiment, "_local_addresses", lambda count: ["unused"] * count)
    context = _Context() if context is None else context
    monkeypatch.setattr(experiment.multiprocessing, "get_context", lambda mode: context)
    class Server:
        memory = {}
        def __init__(self, *a, **k):
            if fail_construction:
                raise ValueError("server construction failed")
            self.server = SimpleNamespace(poll_from_python=lambda **k: False)
        def start(self, *a):
            calls.append("server.start")
        def stop(self):
            calls.append("server.stop")
        def producer_metadata(self):
            return dict(dcores=[8], plan_yaml="actual producer plan", config_yaml="actual config",
                        noise_variance=[1.0])
    class Capture:
        cv = threading.Condition()
        ledger = {"errors": []}
        def __init__(self, *a, **k):
            assert k["subscription_timeout_seconds"] == 6.0 + 2 * 120. + 20. + 60.
        def start(self, *a):
            calls.append("capture.start")
        def wait_for_writes(self):
            calls.append("capture.wait")
        def close(self):
            calls.append("capture.close")
            if fail_capture_close:
                raise RuntimeError("capture close failed")
    def replay(*a, **kwargs):
        assert kwargs["verify_hashes"] is True
        if fail_replay:
            raise ValueError("original replay failure")
        context.events[1].set()
        context.events[2].set()
        kwargs["progress"]({})
        return {"state": "complete"}
    monkeypatch.setattr(experiment, "_ExperimentServer", Server)
    monkeypatch.setattr(experiment, "ControlledCaptureReceiver", Capture)
    monkeypatch.setattr(ReplayObservation, "replay_observation", replay)
    return context, calls


def test_original_failure_survives_cleanup_failure_and_report_is_durable(tmp_path, monkeypatch):
    context, calls = _configure_run(tmp_path, monkeypatch, fail_replay=True, fail_capture_close=True)
    output = tmp_path / "failed-run"
    with pytest.raises(ValueError, match="original replay failure") as raised:
        experiment.run_controlled_online(tmp_path, output)
    assert not context.child.is_alive() and context.status.closed and context.status.joined
    assert "server.stop" in calls and "capture.close" in calls
    report = json.loads((output / "run.json").read_text())
    assert report["state"] == "failed" and not report["complete"]
    assert report["error"] == "ValueError: original replay failure"
    assert report["cleanup_errors"][0]["action"] == "capture.close"
    assert "Additional controlled-run cleanup failures" in raised.value.__notes__[0]


def test_cleanup_failure_cannot_publish_success(tmp_path, monkeypatch):
    context, _ = _configure_run(tmp_path, monkeypatch, fail_capture_close=True)
    output = tmp_path / "cleanup-failure"
    with pytest.raises(RuntimeError, match="did not complete"):
        experiment.run_controlled_online(tmp_path, output)
    report = json.loads((output / "run.json").read_text())
    assert report["scientific_outputs_drained"] and report["promised_files_drained"]
    assert report["producer"]["plan_yaml"] == "actual producer plan"
    assert report["state"] == "failed" and not report["complete"]
    assert not context.child.is_alive()


def test_success_is_published_after_child_shutdown(tmp_path, monkeypatch):
    context, _ = _configure_run(tmp_path, monkeypatch)
    output = tmp_path / "successful-run"
    report = experiment.run_controlled_online(tmp_path, output)
    assert report["state"] == "complete" and report["complete"]
    assert report["grouper_exitcode"] == 0 and not report["cleanup_errors"]
    assert not context.child.is_alive() and context.status.closed and context.status.joined
    assert json.loads((output / "run.json").read_text()) == report


def test_server_startup_failure_reaps_a_real_spawned_child(tmp_path, monkeypatch):
    real_context = multiprocessing.get_context("spawn")
    children = []
    class Context:
        Event = staticmethod(real_context.Event)
        Queue = staticmethod(real_context.Queue)
        @staticmethod
        def Process(**kwargs):
            child = real_context.Process(target=time.sleep, args=(60,), daemon=True)
            children.append(child)
            return child
    _configure_run(tmp_path, monkeypatch, fail_construction=True, context=Context())
    monkeypatch.setattr(experiment, "_SHUTDOWN_TIMEOUT_SECONDS", 0.1)
    output = tmp_path / "startup-failure"
    try:
        with pytest.raises(ValueError, match="server construction failed"):
            experiment.run_controlled_online(tmp_path, output)
        assert len(children) == 1 and children[0].pid is not None
        assert not children[0].is_alive()
        assert children[0] not in multiprocessing.active_children()
        report = json.loads((output / "run.json").read_text())
        assert report["state"] == "failed" and not report["complete"]
        assert report["error"] == "ValueError: server construction failed"
    finally:
        for child in children:
            if child.is_alive():
                child.kill()
                child.join(timeout=5)


def test_producer_evidence_uses_actual_objects_without_a_probe():
    server = object.__new__(experiment._ExperimentServer)
    variances = np.array([1.0, 4.0, 9.0], dtype="<f8")
    metadata = SimpleNamespace(noise_variance=[1.0, 4.0, 9.0],
                               get_channel_variances=lambda: variances,
                               to_yaml_string=lambda: "received metadata")
    plan = SimpleNamespace(config=SimpleNamespace(to_yaml_string=lambda: "actual config"),
                           to_yaml_string=lambda: "actual plan")
    server.server = SimpleNamespace(plan=plan, dedisperser=SimpleNamespace(Dcores=[8, 4]))
    server.allocator = SimpleNamespace(metadata=metadata)
    result = server.producer_metadata()
    assert result["config_yaml"] == "actual config" and result["plan_yaml"] == "actual plan"
    assert result["dcores"] == [8, 4] and result["noise_variance"] == [1., 4., 9.]
    assert result["channel_variance_sha256"] == hashlib.sha256(variances.tobytes()).hexdigest()
    assert result["channel_variance_count"] == 3 and result["weight_values_read_back"] is False
