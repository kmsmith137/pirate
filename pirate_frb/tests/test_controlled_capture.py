"""CPU acceptance checks for causal early-trigger capture, using a finite server.

The fake server implements an independently chosen raw-time retention boundary;
its WriteFiles result is a promise, and notification delivery is separate.
"""
import math
import threading
from pathlib import Path
from types import SimpleNamespace

import grpc
import pytest
import yaml

from pirate_frb.ControlledCapture import (
    CapturePolicy, ControlledCaptureReceiver, capture_interval,
    expected_capture_files, _DeadlineCaptureClient,
)
from pirate_frb.rpc.grpc import frb_sifter_pb2 as pb


METADATA = dict(zone_freq_edges=[300., 350., 450., 600., 800., 1500.],
                zone_nfreq=[8192, 8192, 6144, 2048, 3584],
                beam_ids=[100, 101], beamset=42, dt_ns_per_seq=5120,
                seq_per_frb_time_sample=195, unix_ns_at_seq_0=1772483060000000000)
SAMPLES = 2048
SEQ_PER_CHUNK = SAMPLES * METADATA["seq_per_frb_time_sample"]
TICK = METADATA["dt_ns_per_seq"] * 1.e-9
CHUNK_SECONDS = SEQ_PER_CHUNK * TICK
TREES = [dict(tree_index=0, early_trigger_level=0),
         dict(tree_index=1, early_trigger_level=1),
         dict(tree_index=2, early_trigger_level=0)]


class Aborted(Exception):
    def __init__(self, code, message):
        self.code = code
        super().__init__(message)


class Context:
    def abort(self, code, message):
        raise Aborted(code, message)


class FiniteServer:
    """Raw frames older than retain_from_chunk cannot be newly promised."""
    def __init__(self):
        self.current_chunk = 0
        self.retain_from_chunk = 0
        self.calls = []
        self.closed = False

    def advance_to(self, seconds):
        self.current_chunk = math.floor(seconds / CHUNK_SECONDS)
        self.retain_from_chunk = max(0, self.current_chunk - 29)

    def get_status(self):
        return SimpleNamespace(rb_start=2*self.retain_from_chunk,
            rb_reaped=2*self.retain_from_chunk, rb_processed=2*self.current_chunk,
            rb_streamed=2*self.current_chunk, rb_assembled=2*self.current_chunk,
            rb_end=2*(self.current_chunk+1))

    def write_files(self, beams, start, end, acqdir):
        self.calls.append(dict(beams=beams, start=start, end=end, acqdir=acqdir,
                               current_chunk=self.current_chunk))
        # Half-open requested intervals intersect a chunk iff left < end and
        # right > start. Future chunks are promises, not completed files.
        return [f"{acqdir}/frame_b{beam}_t{chunk}.asdf"
                for chunk in range(self.retain_from_chunk, math.ceil(end / SEQ_PER_CHUNK))
                if chunk * SEQ_PER_CHUNK < end and (chunk+1) * SEQ_PER_CHUNK > start
                for beam in beams]

    def close(self):
        self.closed = True


def configure(receiver, **changes):
    values = dict(protocol_version=pb.PROTOCOL_VERSION_CURRENT,
        pirate_yaml="dtype: float16\n", xengine_yaml=yaml.safe_dump(METADATA),
        dedispersion_plan_yaml=yaml.safe_dump(dict(trees=TREES)),
        grouper_yaml="peakfinding: shared\n", search_ip_addr="127.0.0.1:6000")
    values.update(changes)
    return receiver.CheckConfiguration(pb.ConfigMessage(**values), Context())


def event(*, dm=2000., toa=200., tree=1, snr=50., beam=100, width=12.):
    return pb.FrbEvent(beam_id=beam, fpga_timestamp=round(toa/TICK),
        dm=dm, snr=snr, width_ms=width, subband_freq_lo_MHz=416.025147 if tree == 1 else 300.,
        subband_freq_hi_MHz=1500., tree_index=tree)


def submit(receiver, ev, *, from_simulator=False):
    # Intentionally allow the canonical low-band timestamp to lie well outside
    # this source chunk. That is the central early-trigger protocol contract.
    request = pb.FrbEventsMessage(from_simulator=from_simulator, beam_set_id=42,
        chunk_fpga_start=round(155./TICK), chunk_fpga_end=round(158./TICK), events=[ev])
    return receiver.FrbEvents(request, Context())


def receiver(tmp_path, *, accept_early=True, policy=None):
    client = FiniteServer()
    obj = ControlledCaptureReceiver(METADATA, SAMPLES, policy or CapturePolicy(),
        "127.0.0.1:6000", tmp_path / "capture.json", client=client,
        accept_early=accept_early)
    assert configure(obj).ok
    return obj, client


def notify_all_promises(obj, *, error=""):
    promises = {name for cap in obj.ledger["captures"] for req in cap["requests"]
                for name in req["promised_files"]}
    def feed():
        for name in sorted(promises):
            yield name, error, ""
        # This finite test feed models deliberate subscription cancellation,
        # never an unexpected server EOF.
        obj.closing = True
    obj.subscriber = feed()
    obj._notifications()
    obj.closing = False
    return promises


def test_early_request_preserves_high_dm_past_and_future_before_full_detection(tmp_path):
    obj, server = receiver(tmp_path)
    server.advance_to(160.)
    assert submit(obj, event()).ok
    assert len(server.calls) == 1  # request already issued, with no full-band confirmation
    cap = obj.ledger["captures"][0]
    req = cap["requests"][0]
    assert not req["missing_files"]
    assert server.calls[0]["start"] * TICK < 112.
    assert server.calls[0]["end"] * TICK > 201.
    assert any(int(Path(f).stem.rsplit("t", 1)[1]) > server.current_chunk
               for f in req["promised_files"])
    assert not obj.ledger["notifications"]
    assert not obj.ledger["complete"]
    # The later full-band detection identifies the same physical event even
    # though its source chunk is separated by approximately 44 seconds.
    server.advance_to(205.)
    submit(obj, event(tree=2))
    assert len(obj.ledger["captures"]) == 1
    assert len(cap["detections"]) == 2
    assert obj.ledger["decisions"][-1]["action"] == "associated"
    assert len(server.calls) == 1
    notify_all_promises(obj)
    assert obj.wait_for_writes(timeout=0)["complete"]


def test_suppression_control_loses_high_dm_prefix_but_keeps_low_dm_capture(tmp_path):
    obj, server = receiver(tmp_path, accept_early=False)
    server.advance_to(160.)
    submit(obj, event(tree=1))
    assert not server.calls
    assert obj.ledger["decisions"][-1]["action"] == "early_capture_suppressed"
    server.advance_to(205.)
    submit(obj, event(tree=2))
    high = obj.ledger["captures"][0]["requests"][0]
    assert high["missing_files"]
    earliest_expected = min(int(Path(f).stem.rsplit("t", 1)[1]) for f in high["expected_files"])
    earliest_promised = min(int(Path(f).stem.rsplit("t", 1)[1]) for f in high["promised_files"])
    assert earliest_expected < earliest_promised
    # A separate run at the low-DM arrival can capture the whole low sweep.
    low_obj, low_server = receiver(tmp_path / "low", accept_early=False)
    low_server.advance_to(95.)
    submit(low_obj, event(dm=100., toa=90., tree=0))
    low = low_obj.ledger["captures"][0]["requests"][0]
    assert not low["missing_files"]
    # Draining promises is deliberately different from complete physical
    # coverage: the suppressed control still records its missing prefix.
    notify_all_promises(obj)
    assert obj.wait_for_writes(timeout=0)["complete"]
    assert high["missing_files"]


def test_physical_interval_covers_full_chord_band_and_rounds_outward():
    ev = dict(dm=2000., fpga_timestamp=round(200./TICK), width_ms=12.)
    policy = CapturePolicy(pre_padding_seconds=1., post_padding_seconds=2.)
    start, end = capture_interval(ev, METADATA, policy)
    # Independent numerical sweep known for 300–1500 MHz and DM 2000.
    wanted_start_seconds = ev["fpga_timestamp"] * TICK - 88.507904 - 1. - .012
    wanted_end_seconds = ev["fpga_timestamp"] * TICK + 2. + .012
    assert start*TICK <= wanted_start_seconds < (start+1)*TICK
    assert (end-1)*TICK < wanted_end_seconds <= end*TICK
    ev.update(dm=0., fpga_timestamp=0)
    assert capture_interval(ev, METADATA, policy)[0] == 0
    assert expected_capture_files("event", 100, 2*SEQ_PER_CHUNK, 4*SEQ_PER_CHUNK,
        SEQ_PER_CHUNK) == ["event/frame_b100_t2.asdf", "event/frame_b100_t3.asdf"]


def test_truth_messages_never_reach_threshold_policy_or_server(tmp_path):
    obj, server = receiver(tmp_path)
    with pytest.raises(Aborted, match="injection truth") as exc:
        submit(obj, event(snr=1.e9), from_simulator=True)
    assert exc.value.code == grpc.StatusCode.FAILED_PRECONDITION
    assert not obj.ledger["decisions"]
    assert not server.calls


def test_threshold_and_beam_identity_keep_unrelated_events_separate(tmp_path):
    obj, server = receiver(tmp_path)
    server.advance_to(160.)
    submit(obj, event(snr=9.99))
    assert not server.calls
    submit(obj, event(snr=10.))
    submit(obj, event(beam=101))
    submit(obj, event(toa=202.))
    assert len(obj.ledger["captures"]) == 3
    assert [cap["first_event"]["beam_id"] for cap in obj.ledger["captures"]] == [100, 101, 100]


def test_late_refinement_can_extend_capture_without_delaying_first_request(tmp_path):
    obj, server = receiver(tmp_path)
    server.advance_to(160.)
    submit(obj, event())
    first = dict(server.calls[0])
    server.advance_to(205.)
    submit(obj, event(tree=2, width=30.))
    cap = obj.ledger["captures"][0]
    assert len(cap["requests"]) == 2
    assert len(cap["detections"]) == 2
    assert server.calls[0] == first
    assert server.calls[1]["start"] < first["start"]
    assert server.calls[1]["end"] > first["end"]
    # Old files omitted from a later promise do not erase earlier promises.
    union = {f for req in cap["requests"] for f in req["promised_files"]}
    assert set(cap["requests"][0]["promised_files"]) <= union


def test_promised_files_are_not_completion_notifications(tmp_path):
    obj, server = receiver(tmp_path)
    server.advance_to(160.)
    submit(obj, event())
    assert obj.ledger["captures"][0]["requests"][0]["promised_files"]
    with pytest.raises(TimeoutError, match="promised files"):
        obj.wait_for_writes(timeout=0)
    assert not obj.ledger["complete"]
    assert obj.ledger["errors"]


def test_write_failure_notification_is_never_treated_as_success(tmp_path):
    obj, server = receiver(tmp_path)
    server.advance_to(160.)
    submit(obj, event())
    notify_all_promises(obj, error="disk full")
    with pytest.raises(RuntimeError, match="disk full"):
        obj.wait_for_writes(timeout=0)
    assert not obj.ledger["complete"]


def test_configuration_and_handshake_reject_wrong_producer(tmp_path):
    obj, server = receiver(tmp_path)
    bad = dict(METADATA, beam_ids=[100])
    with pytest.raises(Aborted, match="metadata mismatch"):
        configure(obj, xengine_yaml=yaml.safe_dump(bad))
    with pytest.raises(Aborted, match="protocol mismatch"):
        configure(obj, protocol_version=0)
    with pytest.raises(Aborted, match="configuration changed"):
        configure(obj, pirate_yaml="dtype: float32\n")
    with pytest.raises(Aborted, match="invalid producer tree enumeration"):
        configure(obj, dedispersion_plan_yaml=yaml.safe_dump(dict(trees=[dict(tree_index=3)])))
    assert not server.calls


@pytest.mark.parametrize("values", [
    dict(snr_threshold=False), dict(buffer_seconds=0), dict(write_timeout_seconds=0),
    dict(pre_padding_seconds=-1), dict(association_dm_tolerance=float("nan")),
    dict(classifier_mode="classifier"), dict(unknown_setting=1),
    dict(version=True), dict(version=2), dict(snr_threshold="10"),
    dict(version=1, schema_version=2),
])
def test_invalid_capture_policy_rejected(values):
    with pytest.raises((TypeError, ValueError)):
        CapturePolicy.from_dict(values)


def test_shipped_capture_configuration_is_accepted():
    path = Path(__file__).resolve().parents[2] / "configs/experiments/chord_replay.yml"
    config = yaml.safe_load(path.read_text())["capture"]
    policy = CapturePolicy.from_dict(config)
    assert policy.classifier_mode == "bypass"
    assert policy.buffer_seconds == 60.
    assert policy.pre_padding_seconds == policy.post_padding_seconds == 1.


def test_wrong_beamset_is_rejected_without_capture(tmp_path):
    obj, server = receiver(tmp_path)
    request = pb.FrbEventsMessage(beam_set_id=43, events=[event()])
    with pytest.raises(Aborted, match="beamset"):
        obj.FrbEvents(request, Context())
    assert not obj.ledger["decisions"]
    assert not server.calls


def test_unexpected_notification_eof_invalidates_completion(tmp_path):
    obj, _ = receiver(tmp_path)
    obj.ledger["complete"] = True
    obj.subscriber = iter(())
    obj._notifications()
    assert not obj.ledger["complete"]
    with pytest.raises(RuntimeError, match="subscription ended"):
        obj.wait_for_writes(timeout=0)


def test_cleanup_attempts_every_resource_after_failure(tmp_path):
    obj, client = receiver(tmp_path)
    actions = []

    class BadServer:
        def stop(self, grace):
            actions.append("grpc")
            raise RuntimeError("grpc stop failed")

    class BadSubscriber:
        def close(self):
            actions.append("subscriber")
            raise RuntimeError("subscription close failed")

    class StuckThread:
        def join(self, timeout):
            assert timeout == 5
            actions.append("join")
        def is_alive(self):
            return True

    obj.grpc_server = BadServer()
    obj.subscriber = BadSubscriber()
    obj.notification_thread = StuckThread()
    obj.ledger["complete"] = True
    with pytest.raises(RuntimeError, match="grpc stop failed"):
        obj.close()
    assert actions == ["grpc", "subscriber", "join"]
    assert client.closed
    assert not obj.ledger["complete"]
    assert any("notification thread did not stop" in error for error in obj.ledger["errors"])
    obj.close()  # idempotent even after an unsuccessful close


def test_default_capture_client_bounds_unary_rpcs(monkeypatch):
    from pirate_frb import ControlledCapture as cc
    calls = []

    class Stub:
        def GetStatus(self, request, timeout):
            calls.append(("status", timeout))
            return SimpleNamespace()
        def WriteFiles(self, request, timeout):
            calls.append(("write", timeout))
            return SimpleNamespace(filename_list=[])

    def base_init(client, address):
        client.stub = Stub()

    monkeypatch.setattr(cc.FrbSearchClient, "__init__", base_init)
    client = _DeadlineCaptureClient("server", 2., .1, 300.)
    client.get_status()
    client.write_files([100], 0, 1, "test")
    assert calls == [("status", 2.), ("write", 2.)]


def test_subscription_ready_timeout_cancels_blocked_stream(monkeypatch):
    from pirate_frb import ControlledCapture as cc
    cancelled = threading.Event()
    seen_deadlines = []

    class BlockingCall:
        def __iter__(self):
            return self
        def __next__(self):
            if not cancelled.wait(1.):
                raise AssertionError("readiness watchdog did not cancel the stream")
            raise StopIteration
        def cancel(self):
            cancelled.set()

    class Stub:
        def SubscribeFiles(self, request, timeout):
            seen_deadlines.append(timeout)
            return BlockingCall()

    def base_init(client, address):
        client.stub = Stub()

    monkeypatch.setattr(cc.FrbSearchClient, "__init__", base_init)
    client = _DeadlineCaptureClient("server", 2., .01, 300.)
    with pytest.raises(TimeoutError, match="did not become ready"):
        client.subscribe_files()
    assert cancelled.is_set()
    assert seen_deadlines == [300.]


def test_ready_subscription_cancels_only_its_watchdog(monkeypatch):
    from pirate_frb import ControlledCapture as cc
    cancelled = threading.Event()

    class ReadyCall:
        def __iter__(self):
            return self
        def __next__(self):
            return SimpleNamespace(WhichOneof=lambda field: "ready")
        def cancel(self):
            cancelled.set()

    class Stub:
        def SubscribeFiles(self, request, timeout):
            assert timeout == 300.
            return ReadyCall()

    def base_init(client, address):
        client.stub = Stub()

    monkeypatch.setattr(cc.FrbSearchClient, "__init__", base_init)
    client = _DeadlineCaptureClient("server", 2., .01, 300.)
    subscriber = client.subscribe_files()
    assert not cancelled.wait(.03)
    subscriber.close()
    assert cancelled.is_set()
