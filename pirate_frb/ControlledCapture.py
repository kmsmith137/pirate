"""Threshold-based capture receiver for the controlled replay experiment.

This module reads structural metadata and detected sifter events only.  The
injection manifest is deliberately absent from the decision path.  Scientific
grouping is performed by the shared grouper; the association below only joins
later tree detections to an already-issued capture request.
"""

from __future__ import annotations

from concurrent import futures
from dataclasses import dataclass
import json
import math
from pathlib import Path
import threading
import time

import grpc
import yaml

from .rpc import FrbSearchClient
from .rpc.FileSubscriber import FileSubscriber
from .rpc.grpc import frb_sifter_pb2 as pb
from .rpc.grpc import frb_sifter_pb2_grpc


K_DM = 4148.808


def write_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".tmp")
    temporary.write_text(json.dumps(value, indent=2, allow_nan=False) + "\n")
    temporary.replace(path)


@dataclass(frozen=True)
class CapturePolicy:
    snr_threshold: float = 10.0
    buffer_seconds: float = 60.0
    pre_padding_seconds: float = 1.0
    post_padding_seconds: float = 1.0
    association_dm_tolerance: float = 5.0
    association_toa_tolerance_seconds: float = 0.1
    write_timeout_seconds: float = 60.0
    classifier_mode: str = "bypass"

    @classmethod
    def from_dict(cls, values):
        values = dict(values)
        if "version" in values and "schema_version" in values:
            raise ValueError("capture configuration cannot contain two version keys")
        version = values.pop("version", values.pop("schema_version", 1))
        if type(version) is not int or version != 1:
            raise ValueError("unsupported capture configuration schema")
        obj = cls(**values)
        if obj.classifier_mode != "bypass":
            raise ValueError("controlled capture requires classifier_mode: bypass")
        for key, value in vars(obj).items():
            if key == "classifier_mode":
                continue
            if type(value) not in (int, float) or not math.isfinite(value) or value < 0:
                raise ValueError(f"capture {key} must be finite and nonnegative")
        if min(obj.snr_threshold, obj.buffer_seconds, obj.write_timeout_seconds) <= 0:
            raise ValueError("threshold, buffer duration and write timeout must be positive")
        return obj


def capture_interval(event, metadata, policy):
    """Return half-open absolute FPGA counts, including the full-band sweep."""
    lo, hi = metadata["zone_freq_edges"][0], metadata["zone_freq_edges"][-1]
    tick = float(metadata["dt_ns_per_seq"]) * 1e-9
    sweep = K_DM * event["dm"] * (lo ** -2 - hi ** -2)
    width = event["width_ms"] * 1e-3
    start = math.floor(event["fpga_timestamp"] -
                       (sweep + policy.pre_padding_seconds + width) / tick)
    end = math.ceil(event["fpga_timestamp"] +
                    (policy.post_padding_seconds + width) / tick)
    return max(0, start), end


def expected_capture_files(acqdir, beam_id, start, end, seq_per_chunk):
    return [f"{acqdir}/frame_b{beam_id}_t{chunk}.asdf"
            for chunk in range(start // seq_per_chunk,
                               (end + seq_per_chunk - 1) // seq_per_chunk)]


def associated_capture(captures, event, metadata, policy):
    tick = float(metadata["dt_ns_per_seq"]) * 1e-9
    matches = []
    for cap in captures:
        first = cap["first_event"]
        if (first["beam_id"] == event["beam_id"] and
                abs(first["dm"] - event["dm"]) <= policy.association_dm_tolerance and
                abs(first["fpga_timestamp"] - event["fpga_timestamp"]) * tick <=
                policy.association_toa_tolerance_seconds):
            matches.append(cap)
    # Stable nearest-time association when several prior captures qualify.
    return min(matches, key=lambda cap: (
        abs(cap["first_event"]["fpga_timestamp"] - event["fpga_timestamp"]),
        cap["capture_id"])) if matches else None


def _status_dict(status):
    keys = ("rb_start", "rb_reaped", "rb_processed", "rb_streamed",
            "rb_assembled", "rb_end")
    return {key: int(getattr(status, key)) for key in keys}


class _DeadlineCaptureClient(FrbSearchClient):
    """Keep capture RPC failures bounded without changing other PIRATE clients."""

    def __init__(self, address, rpc_timeout, ready_timeout, stream_timeout):
        super().__init__(address)
        self.rpc_timeout = rpc_timeout
        self.ready_timeout = ready_timeout
        self.stream_timeout = stream_timeout
        raw_stub = self.stub

        class UnaryDeadlines:
            def __getattr__(proxy, name):
                method = getattr(raw_stub, name)
                if name in ("GetStatus", "WriteFiles"):
                    return lambda request: method(request, timeout=rpc_timeout)
                return method

        self.stub = UnaryDeadlines()

    def subscribe_files(self):
        client = self
        pending = {"call": None, "timer": None}
        timed_out = threading.Event()

        class ReadyDeadline:
            def SubscribeFiles(proxy, request):
                call = client.stub.SubscribeFiles(request, timeout=client.stream_timeout)
                pending["call"] = call

                def expired():
                    timed_out.set()
                    call.cancel()

                timer = threading.Timer(client.ready_timeout, expired)
                timer.daemon = True
                pending["timer"] = timer
                timer.start()
                return call

        try:
            subscriber = FileSubscriber(ReadyDeadline())
            if timed_out.is_set():
                subscriber.close()
                raise TimeoutError("file subscription did not become ready before its deadline")
            return subscriber
        except BaseException as exc:
            if pending["call"] is not None:
                pending["call"].cancel()
            if timed_out.is_set():
                raise TimeoutError("file subscription did not become ready before its deadline") from exc
            raise
        finally:
            if pending["timer"] is not None:
                pending["timer"].cancel()


class ControlledCaptureReceiver(frb_sifter_pb2_grpc.FrbSifterServicer):
    """Sifter-protocol receiver with an auditable past/future write ledger."""

    def __init__(self, metadata, samples_per_chunk, policy, search_address,
                 report_path, *, accept_early=True, client=None,
                 rpc_timeout_seconds=10.0, subscription_ready_timeout_seconds=10.0,
                 subscription_timeout_seconds=600.0):
        self.metadata = metadata
        self.seq_per_chunk = int(samples_per_chunk) * int(metadata["seq_per_frb_time_sample"])
        self.policy = policy
        self.search_address = search_address
        self.report_path = Path(report_path)
        self.accept_early = bool(accept_early)
        for name, value in (("rpc_timeout_seconds", rpc_timeout_seconds),
                            ("subscription_ready_timeout_seconds", subscription_ready_timeout_seconds),
                            ("subscription_timeout_seconds", subscription_timeout_seconds)):
            if type(value) not in (int, float) or not math.isfinite(value) or value <= 0:
                raise ValueError(f"{name} must be finite and positive")
        self.client = client if client is not None else _DeadlineCaptureClient(
            search_address, rpc_timeout_seconds, subscription_ready_timeout_seconds,
            subscription_timeout_seconds)
        self.started = time.monotonic()
        self.cv = threading.Condition(threading.RLock())
        self.config_signature = None
        self.trees = None
        self.grpc_server = None
        self.subscriber = None
        self.notification_thread = None
        self.closing = False
        self._closed = False
        self.ledger = dict(schema_version=1, classifier_mode="bypass",
                           accept_early=self.accept_early, complete=False,
                           decisions=[], captures=[], notifications={}, errors=[])

    def _save(self):
        write_json(self.report_path, self.ledger)

    def start(self, address):
        # The ready sentinel registers this subscription before any trigger can
        # arrive. Reading in a dedicated thread also covers slow future writes.
        self.subscriber = self.client.subscribe_files()
        self.notification_thread = threading.Thread(target=self._notifications,
                                                    name="capture-file-notifications", daemon=True)
        self.notification_thread.start()
        self.grpc_server = grpc.server(futures.ThreadPoolExecutor(max_workers=2),
                                      options=[("grpc.so_reuseport", 0)])
        frb_sifter_pb2_grpc.add_FrbSifterServicer_to_server(self, self.grpc_server)
        if not self.grpc_server.add_insecure_port(address):
            self.close()
            raise RuntimeError(f"cannot bind capture receiver at {address}")
        self.grpc_server.start()
        with self.cv:
            self._save()

    def _notifications(self):
        try:
            for filename, error, stream in self.subscriber:
                with self.cv:
                    self.ledger["notifications"][filename] = dict(
                        error=error, stream_name=stream,
                        elapsed_seconds=time.monotonic() - self.started)
                    if error:
                        self.ledger["errors"].append(f"{filename}: {error}")
                        self.ledger["complete"] = False
                    self._save()
                    self.cv.notify_all()
            with self.cv:
                if not self.closing:
                    self.ledger["errors"].append("file subscription ended before capture receiver closed")
                    self.ledger["complete"] = False
                    self._save()
                self.cv.notify_all()
        except Exception as exc:
            with self.cv:
                if not self.closing:
                    self.ledger["errors"].append(f"file subscription: {exc}")
                    self.ledger["complete"] = False
                    self._save()
                self.cv.notify_all()

    def CheckConfiguration(self, request, context):
        if request.protocol_version != pb.PROTOCOL_VERSION_CURRENT:
            context.abort(grpc.StatusCode.FAILED_PRECONDITION, "sifter protocol mismatch")
        try:
            metadata = yaml.safe_load(request.xengine_yaml)
            for key in ("zone_freq_edges", "zone_nfreq", "dt_ns_per_seq",
                        "seq_per_frb_time_sample", "beam_ids", "unix_ns_at_seq_0"):
                if metadata.get(key) != self.metadata.get(key):
                    raise ValueError(f"producer metadata mismatch: {key}")
            if request.search_ip_addr and request.search_ip_addr != self.search_address:
                raise ValueError("producer search address differs from configured capture target")
            plan = yaml.safe_load(request.dedispersion_plan_yaml)
            trees = plan["trees"]
            if not trees or any(int(t["tree_index"]) != i for i, t in enumerate(trees)):
                raise ValueError("invalid producer tree enumeration")
            signature = (request.pirate_yaml, request.xengine_yaml,
                         request.dedispersion_plan_yaml, request.grouper_yaml)
            with self.cv:
                if self.config_signature is not None and signature != self.config_signature:
                    raise ValueError("producer configuration changed during capture")
                self.config_signature = signature
                self.trees = trees
                self.ledger["producer_configuration"] = dict(
                    pirate_yaml=request.pirate_yaml, xengine_yaml=request.xengine_yaml,
                    dedispersion_plan_yaml=request.dedispersion_plan_yaml,
                    grouper_yaml=request.grouper_yaml)
                self._save()
        except (KeyError, TypeError, ValueError, yaml.YAMLError) as exc:
            context.abort(grpc.StatusCode.INVALID_ARGUMENT, str(exc))
        return pb.ConfigReply(ok=True)

    def FrbEvents(self, request, context):
        if request.from_simulator:
            context.abort(grpc.StatusCode.FAILED_PRECONDITION,
                          "injection truth is not accepted by the capture decision path")
        if request.beam_set_id != self.metadata["beamset"]:
            context.abort(grpc.StatusCode.FAILED_PRECONDITION, "detected event beamset differs from acquisition")
        try:
            with self.cv:
                if self.trees is None:
                    raise ValueError("CheckConfiguration must precede detected events")
                for ev in request.events:
                    event = {name: getattr(ev, name) for name in (
                        "beam_id", "fpga_timestamp", "dm", "snr", "width_ms",
                        "subband_freq_lo_MHz", "subband_freq_hi_MHz", "tree_index")}
                    self._process_event(event, request.chunk_fpga_start, request.chunk_fpga_end)
                self._save()
                self.cv.notify_all()
        except Exception as exc:
            with self.cv:
                self.ledger["errors"].append(f"trigger processing: {exc}")
                self._save()
                self.cv.notify_all()
            context.abort(grpc.StatusCode.INTERNAL, str(exc))
        return pb.FrbEventsReply(ok=True, message="classifier bypass; threshold capture policy")

    def _process_event(self, event, chunk_start, chunk_end):
        if (event["beam_id"] not in self.metadata["beam_ids"] or
                not 0 <= event["tree_index"] < len(self.trees) or
                event["fpga_timestamp"] < 0 or event["dm"] < 0 or event["width_ms"] < 0 or
                not all(math.isfinite(value) for value in event.values())):
            raise ValueError("invalid detected event")
        level = int(self.trees[event["tree_index"]]["early_trigger_level"])
        decision = dict(event=event, early_trigger_level=level,
                        chunk_fpga_start=int(chunk_start), chunk_fpga_end=int(chunk_end),
                        received_elapsed_seconds=time.monotonic() - self.started)
        self.ledger["decisions"].append(decision)
        if event["snr"] < self.policy.snr_threshold:
            decision["action"] = "below_threshold"
            return
        if level and not self.accept_early:
            decision["action"] = "early_capture_suppressed"
            return
        cap = associated_capture(self.ledger["captures"], event, self.metadata, self.policy)
        start, end = capture_interval(event, self.metadata, self.policy)
        if cap is None:
            cap_id = len(self.ledger["captures"])
            cap = dict(capture_id=cap_id, acqdir=f"event_{cap_id:06d}",
                       first_event=event, first_early_trigger_level=level,
                       start_seq=start, end_seq=end, requests=[], detections=[])
            self.ledger["captures"].append(cap)
            decision["action"] = "capture"
        else:
            decision["action"] = "associated"
        decision["capture_id"] = cap["capture_id"]
        cap["detections"].append(event)
        # Later detections can refine/extend coverage but never delay the first
        # request. Repeating an already-covered range would add no information.
        if cap["requests"] and start >= cap["start_seq"] and end <= cap["end_seq"]:
            return
        cap["start_seq"] = min(start, cap["start_seq"])
        cap["end_seq"] = max(end, cap["end_seq"])
        status = _status_dict(self.client.get_status())
        began = time.monotonic()
        promised = self.client.write_files([event["beam_id"]], cap["start_seq"],
                                          cap["end_seq"], cap["acqdir"])
        expected = expected_capture_files(cap["acqdir"], event["beam_id"],
                                          cap["start_seq"], cap["end_seq"], self.seq_per_chunk)
        cap["requests"].append(dict(start_seq=cap["start_seq"], end_seq=cap["end_seq"],
                                   status_before=status, expected_files=expected,
                                   promised_files=promised,
                                   missing_files=sorted(set(expected) - set(promised)),
                                   requested_elapsed_seconds=began - self.started,
                                   rpc_seconds=time.monotonic() - began))

    def wait_for_writes(self, timeout=None):
        deadline = time.monotonic() + (self.policy.write_timeout_seconds if timeout is None else timeout)
        with self.cv:
            while True:
                if self.ledger["errors"]:
                    raise RuntimeError("; ".join(self.ledger["errors"]))
                expected = {f for c in self.ledger["captures"] for r in c["requests"]
                            for f in r["promised_files"]}
                missing = expected - self.ledger["notifications"].keys()
                if not missing:
                    self.ledger["complete"] = True
                    self._save()
                    return self.ledger
                left = deadline - time.monotonic()
                if left <= 0:
                    self.ledger["errors"].append(f"timed out waiting for {len(missing)} promised files")
                    self._save()
                    raise TimeoutError(self.ledger["errors"][-1])
                self.cv.wait(min(left, 0.5))

    def close(self):
        """Attempt every cleanup even when one fails; do not abandon live threads."""
        if self._closed:
            return
        self.closing = True
        failures = []

        def attempt(label, action):
            try:
                action()
            except BaseException as exc:
                failures.append(f"{label}: {type(exc).__name__}: {exc}")

        if self.grpc_server is not None:
            def stop_rpc():
                if self.grpc_server.stop(grace=1).wait(timeout=5) is False:
                    raise TimeoutError("capture gRPC handlers did not stop")
            attempt("capture gRPC shutdown", stop_rpc)
        if self.subscriber is not None:
            attempt("file subscription shutdown", self.subscriber.close)
        if self.notification_thread is not None:
            def join_notifications():
                self.notification_thread.join(timeout=5)
                if self.notification_thread.is_alive():
                    raise TimeoutError("file notification thread did not stop")
            attempt("notification thread shutdown", join_notifications)
        attempt("search client shutdown", self.client.close)
        self._closed = True
        with self.cv:
            self.ledger["errors"].extend(failures)
            if self.ledger["errors"]:
                self.ledger["complete"] = False
            attempt("capture ledger save", self._save)
            all_errors = list(dict.fromkeys(self.ledger["errors"] + failures))
        if all_errors:
            raise RuntimeError("; ".join(all_errors))
