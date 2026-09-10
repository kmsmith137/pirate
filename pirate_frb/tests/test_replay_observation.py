"""Finite replay causality, ownership and exact raw-data preservation."""

from types import SimpleNamespace
import threading

import numpy as np
import pytest

from pirate_frb.ReplayObservation import (
    ReplayPlan, drive_replay, metadata_identity, copy_packed_frame,
    PackedFrameSource, _BlockingDeadline, _METADATA_FIELDS,
)


class Clock:
    def __init__(self):
        self.now = 20.0

    def __call__(self):
        return self.now

    def sleep(self, seconds):
        assert seconds >= 0
        self.now += seconds


class Sender:
    nworkers = 2
    minichunks_per_chunk = 2

    def __init__(self, clock, *, block_delay=0.0):
        self.clock = clock
        self.sent = []
        self.waited = []
        self.drained = []
        self.block_delay = block_delay
        self.stop_at = None

    def wait_until_processed(self, worker, index):
        self.waited.append((worker, index))
        assert any(w == worker and i == index for w, i, *_ in self.sent)
        return True

    def enqueue_send_minichunk(self, worker, index, fset):
        if index == self.stop_at:
            return False
        self.sent.append((worker, index, "science", fset.time_chunk_index, self.clock()))
        return True

    def enqueue_send_junk(self, worker, index):
        self.sent.append((worker, index, "tail", None, self.clock()))
        return True

    def synchronize(self, worker):
        self.clock.sleep(self.block_delay)
        self.drained.append(worker)
        return True


def plan(**overrides):
    values = dict(initial_chunk=7, nchunks=3, samples_per_chunk=512,
                  time_sample_s=0.001, beam_ids=(100, 101))
    values.update(overrides)
    return ReplayPlan(**values)


def test_absolute_indices_deadlines_and_only_two_transport_tail_chunks():
    clock = Clock()
    sender = Sender(clock)
    loaded, callbacks = [], []
    def load(chunk):
        loaded.append(chunk)
        return SimpleNamespace(time_chunk_index=chunk)
    result = drive_replay(plan(), sender, load, clock=clock, sleep=clock.sleep,
                          progress=lambda report: callbacks.append(report["state"]))
    assert loaded == [7, 8, 9]
    assert result["state"] == "sender_complete"
    assert result["server_gpu_grouper_and_files_drained"] is False
    assert len(result["minichunks"]) == 10
    assert len(result["chunks"]) == 5
    assert sender.drained == [0, 1] * 5
    for worker in range(2):
        transmissions = [s for s in sender.sent if s[0] == worker]
        assert [s[1] for s in transmissions] == list(range(14, 24))
        assert [s[2] for s in transmissions] == ["science"] * 6 + ["tail"] * 4
        for offset, sent in enumerate(transmissions):
            assert sent[-1] == pytest.approx(20.0 + (offset+1) * 0.256)
    assert result["max_enqueue_lag_s"] < 1e-12
    assert result["elapsed_s"] == pytest.approx(2.56)
    assert callbacks[-1] == "sender_complete"


def test_backpressure_lag_measured_after_wire_dispatch_not_only_enqueue():
    clock = Clock()
    sender = Sender(clock, block_delay=0.2)
    result = drive_replay(plan(nchunks=1), sender,
                          lambda chunk: SimpleNamespace(time_chunk_index=chunk),
                          clock=clock, sleep=clock.sleep)
    assert result["max_chunk_dispatch_lag_s"] >= 0.4 - 1e-12
    assert result["chunks"][0]["dispatch_lag_s"] == pytest.approx(0.4)
    assert result["max_enqueue_lag_s"] > 0


def test_failure_retains_progress_without_claiming_sender_completion():
    clock = Clock()
    sender = Sender(clock)
    sender.stop_at = 16
    report, callbacks, persisted = {}, [], []
    with pytest.raises(RuntimeError, match="stopped before completing"):
        drive_replay(plan(), sender, lambda chunk: SimpleNamespace(time_chunk_index=chunk),
                     clock=clock, sleep=clock.sleep, report=report,
                     progress=lambda result: callbacks.append(result["state"]),
                     persist=lambda result: persisted.append(result["state"]))
    assert report["state"] == "failed"
    assert len(report["chunks"]) == 1
    assert callbacks[-1] == "running"
    assert persisted[-1] == "failed"
    assert "enqueue worker" in report["error"]
    assert not any(s[2] == "tail" for s in sender.sent)


def test_configured_dispatch_deadline_failure_and_status_retention():
    clock = Clock()
    sender = Sender(clock, block_delay=0.2)
    report = {}
    status = dict(num_connections=2, num_bytes=10, rb_start=0, rb_reaped=4,
                  rb_processed=12, rb_streamed=12, rb_assembled=14, rb_end=16)
    with pytest.raises(RuntimeError, match="dispatch lag"):
        drive_replay(plan(nchunks=1), sender,
                     lambda chunk: SimpleNamespace(time_chunk_index=chunk),
                     clock=clock, sleep=clock.sleep, max_lag_s=0.3,
                     status_reader=lambda: status, report=report)
    assert report["server_status"][0]["assembled_backlog_chunks"] == 1
    assert report["server_status"][0]["logical_retention_s"] == pytest.approx(4.096)
    assert report["server_status"][0]["unreaped_retention_s"] == pytest.approx(3.072)
    assert report["state"] == "failed"


@pytest.mark.parametrize("overrides", [
    {"rate": 0}, {"rate": float("nan")}, {"initial_chunk": -1}, {"nchunks": 0},
    {"samples_per_chunk": 257}, {"beam_ids": (1, 1)}, {"beam_ids": ()},
    {"time_sample_s": float("inf")}, {"nchunks": True},
])
def test_replay_plan_rejects_invalid_geometry(overrides):
    with pytest.raises(ValueError):
        plan(**overrides)


def test_nonunit_rate_is_recorded_and_scales_only_delivery_clock():
    clock = Clock()
    sender = Sender(clock)
    result = drive_replay(plan(rate=2, nchunks=1), sender,
                          lambda chunk: SimpleNamespace(time_chunk_index=chunk),
                          clock=clock, sleep=clock.sleep)
    assert result["science_duration_s"] == pytest.approx(0.512)
    assert result["elapsed_s"] == pytest.approx(3 * 0.512 / 2)
    assert result["realtime_rate"] is False


def metadata():
    values = {key: 1 for key in _METADATA_FIELDS}
    values.update(beam_ids=[100, 101], beam_positions_x=[0.1, 0.2],
                  beam_positions_y=[0.3, 0.4], zone_nfreq=[4],
                  zone_freq_edges=[300.0, 1500.0], noise_variance=[1.0])
    return SimpleNamespace(**values)


def frame(xmd, beam=100, chunk=7, *, projected=True):
    m = metadata_identity(xmd, beam_id=beam) if projected else metadata_identity(xmd)
    return SimpleNamespace(beam_id=beam, time_chunk_index=chunk, ntime=512, nfreq=4,
        metadata=SimpleNamespace(**m), data=np.zeros((4, 256), dtype=np.uint8),
        scales_offsets=np.zeros((4, 2, 2), dtype=np.float16))


def test_packed_copy_preserves_every_data_and_scale_bit_without_mutating_source():
    xmd = metadata()
    source = frame(xmd, beam=101)
    destination = frame(xmd, beam=101, projected=False)
    source.data[:] = np.arange(source.data.size, dtype=np.uint8).reshape(source.data.shape)
    # Negative zero, a NaN payload, and subnormals expose unintended conversion.
    bits = source.scales_offsets.view(np.uint16)
    bits.flat[:] = [0x8000, 0x7E11, 0x0001, 0xFFFF] * 4
    original_data, original_scales = source.data.tobytes(), source.scales_offsets.tobytes()
    copy_packed_frame(source, destination, xmd, 101, 7)
    assert destination.data.tobytes() == original_data
    assert destination.scales_offsets.tobytes() == original_scales
    assert source.data.tobytes() == original_data
    assert source.scales_offsets.tobytes() == original_scales
    assert xmd.beam_ids == [100, 101]


def test_packed_copy_rejects_clock_beam_or_shape_mismatch_before_copying():
    xmd = metadata()
    source, destination = frame(xmd), frame(xmd, projected=False)
    source.metadata.unix_ns_at_seq_0 += 1
    with pytest.raises(ValueError, match="metadata mismatch"):
        copy_packed_frame(source, destination, xmd, 100, 7)
    source = frame(xmd, beam=101)
    with pytest.raises(ValueError, match="beam/chunk identity"):
        copy_packed_frame(source, destination, xmd, 100, 7)
    source = frame(xmd)
    source.data = np.zeros((4, 128), dtype=np.uint8)
    with pytest.raises(ValueError, match="shape/dtype"):
        copy_packed_frame(source, destination, xmd, 100, 7)


def test_frame_source_uses_canonical_beam_order_and_exact_frame_paths(tmp_path):
    xmd = metadata()
    targets = [frame(xmd, beam=beam, projected=False) for beam in xmd.beam_ids]
    fset = SimpleNamespace(get_frame=lambda index: targets[index], validate=lambda: None)
    requested = []
    def read(path):
        requested.append(path)
        return frame(xmd, beam=101 if path.endswith("101.asdf") else 100)
    bundle = {"frame_entries": [
        {"time_chunk_index": 7, "beam_id": 101, "path": "acq/101.asdf"},
        {"time_chunk_index": 7, "beam_id": 100, "path": "acq/100.asdf"},
    ]}
    source = PackedFrameSource(bundle, tmp_path, xmd,
                               SimpleNamespace(get_frame_set=lambda chunk: fset), read)
    assert source(7) is fset
    assert requested == [str(tmp_path / "acq/100.asdf"), str(tmp_path / "acq/101.asdf")]


def test_blocking_native_operation_has_a_bounded_watchdog():
    stopped = threading.Event()
    deadline = _BlockingDeadline(0.02, stopped.set)
    try:
        with pytest.raises(TimeoutError, match="blocked sender"):
            deadline.call("blocked sender", lambda: stopped.wait(timeout=1))
        assert stopped.is_set()
    finally:
        deadline.close()


def test_progress_failure_is_not_reentered_and_original_error_is_persisted():
    clock = Clock()
    sender = Sender(clock)
    report, persisted, calls = {}, [], []
    def progress(result):
        calls.append(result["state"])
        if len(calls) == 1:
            raise ValueError("original grouper handshake failure")
        raise RuntimeError("secondary server session closed")
    with pytest.raises(ValueError, match="original grouper handshake failure"):
        drive_replay(plan(), sender, lambda c: SimpleNamespace(time_chunk_index=c),
                     clock=clock, sleep=clock.sleep, progress=progress,
                     persist=lambda r: persisted.append(dict(r)), report=report)
    assert calls == ["running"]
    assert report["state"] == "failed"
    assert report["error"] == "ValueError: original grouper handshake failure"
    assert persisted[-1]["error"] == report["error"]


def test_failure_persistence_error_does_not_replace_original_exception():
    clock = Clock()
    sender = Sender(clock)
    def load(chunk):
        raise ValueError("bad input frame")
    def persist(report):
        raise OSError("report disk full")
    with pytest.raises(ValueError, match="bad input frame") as error:
        drive_replay(plan(), sender, load, clock=clock, sleep=clock.sleep, persist=persist)
    assert any("report disk full" in note for note in error.value.__notes__)
