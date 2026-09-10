"""Launch the finite controlled experiment from four independent terminals.

The shared directory is local session coordination, not a scientific input.
Each terminal supervises only its own native worker. The server releases CUDA
IPC only after scientific output and capture writes drain. A failed, interrupted
or lost component fails the session; it is never treated as physical EOF.
"""
from __future__ import annotations

from contextlib import contextmanager
import ctypes
import fcntl
import hashlib
import json
import math
import multiprocessing
import os
from pathlib import Path
import queue
import shlex
import signal
import socket
import sys
import time
import traceback
import uuid

ROLES = ("grouper", "server", "capture", "replay")
POLL_SECONDS = 0.1


def _json(path):
    return json.loads(Path(path).read_text())


def _write(path, value):
    from .ControlledCapture import write_json
    write_json(path, value)


def _boot_id():
    return Path("/proc/sys/kernel/random/boot_id").read_text().strip()


def _process_token(pid):
    try:
        fields = Path(f"/proc/{pid}/stat").read_text().rsplit(")", 1)[1].split()
        return None if fields[0] == "Z" else fields[19]
    except (FileNotFoundError, ProcessLookupError):
        return None


def _log(role, message):
    print(f"[{role}] {message}", flush=True)


def _positive(value, name):
    if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value) or value <= 0:
        raise ValueError(f"{name} must be positive and finite")
    return float(value)


def _configuration_digest(manifest):
    contents = {k: v for k, v in manifest.items() if k != "configuration_sha256"}
    return hashlib.sha256(json.dumps(contents, sort_keys=True, allow_nan=False).encode()).hexdigest()


def prepare_session(bundle_dir, output_dir, *, cuda_device_id=0, base_port=None,
                    accept_early=True, startup_timeout_seconds=600.0,
                    drain_timeout_seconds=120.0, max_lag_seconds=1.0):
    from .ControlledObservation import load_experiment_bundle
    from .ControlledExperiment import _local_addresses, _source_identity
    import yaml

    if type(cuda_device_id) is not int or cuda_device_id < 0:
        raise ValueError("GPU must be a nonnegative integer")
    if type(accept_early) is not bool:
        raise ValueError("accept_early must be Boolean")
    startup = _positive(startup_timeout_seconds, "startup timeout")
    drain = _positive(drain_timeout_seconds, "drain timeout")
    lag = _positive(max_lag_seconds, "sender lag limit")
    if base_port is None:
        addresses = _local_addresses(4)
    else:
        if type(base_port) is not int or not 1024 <= base_port <= 65532:
            raise ValueError("base port must be an integer from 1024 to 65532")
        sockets = []
        try:
            for port in range(base_port, base_port + 4):
                sock = socket.socket()
                sockets.append(sock)
                sock.bind(("127.0.0.1", port))
            addresses = [f"127.0.0.1:{port}" for port in range(base_port, base_port + 4)]
        finally:
            for sock in sockets:
                sock.close()
    bundle = load_experiment_bundle(bundle_dir, verify_hashes=True)
    policy = yaml.safe_load(Path(bundle["capture_config_path"]).read_text())
    output = Path(output_dir).resolve()
    output.mkdir(parents=True, exist_ok=False)
    (output / "_session").mkdir()
    manifest = dict(version=1, session_id=uuid.uuid4().hex, hostname=socket.gethostname(),
        boot_id=_boot_id(), bundle_dir=bundle["bundle_dir"],
        bundle_manifest_sha256=bundle["manifest_sha256"],
        duration_seconds=bundle["duration_seconds"], source=_source_identity(),
        python_executable=str(Path(sys.executable).resolve()),
        cuda_visible_devices=os.environ.get("CUDA_VISIBLE_DEVICES"),
        cuda_device_id=cuda_device_id, accept_early=accept_early,
        startup_timeout_seconds=startup, drain_timeout_seconds=drain,
        max_lag_seconds=lag, write_timeout_seconds=policy["write_timeout_seconds"],
        addresses=dict(zip(("data", "rpc", "grouper", "sifter"), addresses)))
    manifest["configuration_sha256"] = _configuration_digest(manifest)
    _write(output / "session.json", manifest)
    _log("session", f"Prepared {output}; {bundle['duration_seconds']:.3f} seconds, 1× replay.")
    for role in ROLES:
        print(f"python -P -m pirate_frb experiment session {role} {shlex.quote(str(output))}", flush=True)
    return manifest


def _manifest(output, *, check_source=False):
    manifest = _json(output / "session.json")
    if type(manifest.get("version")) is not int or manifest["version"] != 1 or not manifest.get("session_id"):
        raise ValueError("invalid controlled terminal session")
    if manifest.get("configuration_sha256") != _configuration_digest(manifest):
        raise ValueError("session configuration changed; prepare a fresh session")
    if manifest["hostname"] != socket.gethostname() or manifest["boot_id"] != _boot_id():
        raise ValueError("session belongs to another host or an earlier boot; prepare a fresh session")
    if check_source:
        from .ControlledExperiment import _source_identity
        if manifest["source"]["module_sha256"] != _source_identity()["module_sha256"]:
            raise ValueError("processing source changed since session preparation; prepare a fresh session")
        if manifest["python_executable"] != str(Path(sys.executable).resolve()):
            raise ValueError("use the same Python environment in all terminals")
        if manifest["cuda_visible_devices"] != os.environ.get("CUDA_VISIBLE_DEVICES"):
            raise ValueError("CUDA_VISIBLE_DEVICES must match in all terminals")
    return manifest


def _role_status(output, manifest, role):
    path = output / "_session" / f"{role}.json"
    if not path.exists():
        return None
    status = _json(path)
    if (status.get("session_id") != manifest["session_id"] or status.get("role") != role
            or status.get("configuration_sha256") != manifest["configuration_sha256"]):
        raise ValueError(f"{role} status belongs to another session")
    if status["state"] not in ("complete", "failed"):
        if _process_token(status["pid"]) != status["process_token"]:
            status = dict(status, state="failed", error="terminal supervisor disappeared")
    return status


def _check_session(output, manifest, *, ignore_role=None):
    if (output / "_session" / "stop.json").exists():
        raise InterruptedError("session stop requested")
    for role in ROLES:
        if role == ignore_role:
            continue
        status = _role_status(output, manifest, role)
        if status is not None and status["state"] == "failed":
            raise RuntimeError(f"{role} failed: {status.get('error', 'unknown error')}")


@contextmanager
def _decision_lock(output):
    # Serializes explicit stop with the final successful publication. Roles
    # write their own status files; only the server publishes run.json success.
    with (output / "_session" / "decision.lock").open("a") as stream:
        fcntl.flock(stream, fcntl.LOCK_EX)
        try:
            yield
        finally:
            fcntl.flock(stream, fcntl.LOCK_UN)


def session_status(output_dir):
    output = Path(output_dir).resolve()
    manifest = _manifest(output)
    roles = {role: _role_status(output, manifest, role) or dict(state="not_started") for role in ROLES}
    stopped = (output / "_session" / "stop.json").exists()
    if stopped or any(r["state"] == "failed" for r in roles.values()):
        state = "failed"
    elif all(r["state"] == "complete" for r in roles.values()):
        state = "complete"
    else:
        state = "waiting_or_running"
    return dict(state=state, output=str(output), addresses=manifest["addresses"], roles=roles)


def stop_session(output_dir):
    output = Path(output_dir).resolve()
    _manifest(output)
    with _decision_lock(output):
        if session_status(output)["state"] == "complete":
            return dict(state="complete", message="session already completed")
        _write(output / "_session" / "stop.json", dict(reason="explicit stop", monotonic_s=time.monotonic()))
    return dict(state="stop_requested", message="Each terminal will stop its own worker; use a fresh session for another run.")


class _WorkerContext:
    def __init__(self, output, manifest, role, updates, stop):
        self.output, self.manifest, self.role = output, manifest, role
        self.updates, self.stop = updates, stop

    def emit(self, state, **fields):
        self.updates.put(dict(state=state, **fields))

    def status(self, role):
        return _role_status(self.output, self.manifest, role) or {}

    def check(self):
        if self.stop.is_set():
            raise InterruptedError("terminal supervisor requested shutdown")
        _check_session(self.output, self.manifest)

    def wait(self, predicate):
        while True:
            self.check()
            if predicate():
                return
            self.stop.wait(POLL_SECONDS)


def _bundle(context):
    from .ControlledObservation import load_experiment_bundle
    bundle = load_experiment_bundle(context.manifest["bundle_dir"], verify_hashes=False)
    if bundle["manifest_sha256"] != context.manifest["bundle_manifest_sha256"]:
        raise ValueError("observation manifest changed after session preparation")
    return bundle


def _server(context):
    import yaml
    from .ControlledCapture import CapturePolicy
    from .ControlledExperiment import _ExperimentServer, _source_identity

    cfg, output = context.manifest, context.output
    bundle = _bundle(context)
    policy = CapturePolicy.from_dict(yaml.safe_load(Path(bundle["capture_config_path"]).read_text()))
    addresses = cfg["addresses"]
    report = dict(version=1, mode="online", launch_mode="separate_terminals", state="starting", complete=False,
        session_id=cfg["session_id"], accept_early=cfg["accept_early"], classifier_mode="bypass",
        bundle_manifest_sha256=bundle["manifest_sha256"], source=_source_identity(),
        cuda_device_id=cfg["cuda_device_id"], addresses=addresses,
        weights_mode="analytic", detrending=False, shared_processing="SharedGrouper",
        noise_variance=bundle["metadata"].get("noise_variance"))
    _write(output / "run.json", report)
    server = None
    stopped = False
    primary = None
    try:
        server = _ExperimentServer(bundle, output, policy, addresses["data"], addresses["rpc"],
                                   addresses["grouper"], cfg["cuda_device_id"])
        report["memory"] = server.memory
        _log("server", f"Waiting for grouper at {addresses['grouper']}.")
        context.wait(lambda: context.status("grouper").get("listening", False))
        server.start(context.check, cfg["drain_timeout_seconds"])
        context.emit("ready", ready=True)
        _log("server", f"Ready: data {addresses['data']}, RPC {addresses['rpc']}; waiting for replay.")
        report["state"] = "ready"
        _write(output / "run.json", report)
        while True:
            context.check()
            server.server.poll_from_python(timeout_ms=0)
            if context.status("grouper").get("ready") and "producer" not in report:
                producer = server.producer_metadata()
                if producer is not None:
                    report.update(producer=producer, noise_variance=producer["noise_variance"], state="running")
                    _write(output / "run.json", report)
                    _log("server", "Dedispersion running; producer metadata recorded.")
            if (context.status("replay").get("state") == "complete"
                    and context.status("grouper").get("scientific_drained")
                    and context.status("capture").get("state") == "complete"):
                break
            context.stop.wait(POLL_SECONDS)
        if "producer" not in report:
            raise RuntimeError("completed live run has no producer metadata")
        report.update(state="stopping", scientific_outputs_drained=True, promised_files_drained=True,
                      sender_state="sender_complete")
        _write(output / "run.json", report)
        _log("server", "Scientific outputs and promised writes drained; stopping producer before grouper exits.")
        server.stop()
        stopped = True
        context.emit("stopped", producer_stopped=True)
        context.wait(lambda: context.status("grouper").get("state") == "complete")
        server.server.poll_from_python(timeout_ms=0)
        report["grouper_exitcode"] = 0
        return report
    except BaseException as exc:
        primary = exc
        raise
    finally:
        if server is not None and not stopped:
            try:
                server.stop()
            except BaseException as exc:
                if primary is None:
                    raise
                primary.add_note(f"Additional server cleanup failure: {exc!r}")


class _Flag:
    def __init__(self, callback):
        self.callback = callback

    def set(self):
        self.callback()


def _grouper(context):
    from .OnlineGrouper import run_online_grouper
    cfg = context.manifest
    _bundle(context)
    context.emit("listening", listening=True)
    _log("grouper", f"Waiting for producer at {cfg['addresses']['grouper']}.")

    def progress(update):
        context.check()
        if update["phase"] == "window" and update["events"]:
            _log("grouper", f"Owner chunk {update['owner_source_chunk_index']}: {update['events']} grouped event(s).")
        elif update["phase"] == "chunk" and (update["completed_chunks"] % 8 == 0):
            _log("grouper", f"Processed {update['completed_chunks']}/{update['expected_chunks']} science chunks per beam.")

    def ready():
        context.emit("ready", ready=True)
        _log("grouper", "Handshake and capture configuration accepted; shared peak finding/grouping running.")

    def drained():
        context.emit("drained", scientific_drained=True)
        _log("grouper", "Catalog complete; holding CUDA IPC until the server stops.")

    run_online_grouper(cfg["bundle_dir"], str(context.output / "events.asdf"),
        cfg["addresses"]["grouper"], sifter_addr=cfg["addresses"]["sifter"],
        stop_event=context.stop, ready_event=_Flag(ready), completion_event=_Flag(drained),
        progress=progress)


def _capture(context):
    import yaml
    from .ControlledCapture import CapturePolicy, ControlledCaptureReceiver
    cfg = context.manifest
    bundle = _bundle(context)
    policy = CapturePolicy.from_dict(yaml.safe_load(Path(bundle["capture_config_path"]).read_text()))
    _log("capture", "Waiting for server RPC; classifier bypass is enabled.")
    context.wait(lambda: context.status("server").get("ready", False))
    receiver = ControlledCaptureReceiver(bundle["metadata"], bundle["samples_per_chunk"], policy,
        cfg["addresses"]["rpc"], context.output / "capture.json", accept_early=cfg["accept_early"],
        subscription_timeout_seconds=(cfg["startup_timeout_seconds"] + cfg["duration_seconds"]
                                      + 2 * cfg["drain_timeout_seconds"] + policy.write_timeout_seconds + 60))
    primary = None
    try:
        receiver.start(cfg["addresses"]["sifter"])
        context.emit("ready", ready=True)
        _log("capture", f"Listening at {cfg['addresses']['sifter']}; early capture={cfg['accept_early']}.")
        seen = 0
        while True:
            context.check()
            with receiver.cv:
                if receiver.ledger["errors"]:
                    raise RuntimeError("; ".join(receiver.ledger["errors"]))
                decisions = list(receiver.ledger["decisions"][seen:])
                seen += len(decisions)
            for decision in decisions:
                event = decision["event"]
                _log("capture", f"beam={event['beam_id']} tree={event['tree_index']} DM={event['dm']:.4f} "
                     f"S/N={event['snr']:.3f}: {decision['action']}")
            if (context.status("grouper").get("scientific_drained")
                    and context.status("replay").get("state") == "complete"):
                break
            context.stop.wait(POLL_SECONDS)
        receiver.wait_for_writes()
        _log("capture", "All promised file writes completed. Physical coverage is checked by experiment compare.")
    except BaseException as exc:
        primary = exc
        raise
    finally:
        try:
            receiver.close()
        except BaseException as exc:
            if primary is None:
                raise
            primary.add_note(f"Additional capture cleanup failure: {exc!r}")


def _replay(context):
    from .ReplayObservation import replay_observation
    cfg = context.manifest
    _bundle(context)
    _log("replay", "Waiting for server, listening grouper and capture receiver.")
    context.wait(lambda: context.status("server").get("ready")
                 and context.status("grouper").get("listening")
                 and context.status("capture").get("ready"))
    context.emit("replaying", replay_started_monotonic_s=time.monotonic())
    _log("replay", "Verifying input hashes, then sending the saved observation at 1×.")

    def progress(report):
        context.check()
        count = len(report["chunks"])
        context.emit("replaying", chunks_dispatched=count)
        if count % 8 == 0:
            _log("replay", f"Dispatched {count} chunks; maximum completed dispatch lag "
                 f"{report['max_chunk_dispatch_lag_s'] * 1000:.3f} ms.")

    replay_observation(cfg["bundle_dir"], cfg["addresses"]["rpc"], str(context.output / "replay.json"),
        rate=1.0, nworkers=1, startup_timeout_s=cfg["drain_timeout_seconds"],
        drain_timeout_s=cfg["drain_timeout_seconds"], max_lag_s=cfg["max_lag_seconds"],
        progress=progress, verify_hashes=True)
    _log("replay", "All science frames and two transport-tail chunks dispatched; the server coordinates downstream completion.")


def _parent_death_guard(expected_parent):
    # Linux-only, like the CUDA IPC session. Do not leave a native worker behind
    # if its terminal supervisor is killed without executing Python cleanup.
    libc = ctypes.CDLL(None, use_errno=True)
    if libc.prctl(1, signal.SIGKILL, 0, 0, 0) != 0:
        raise OSError(ctypes.get_errno(), "could not install worker parent-death guard")
    if os.getppid() != expected_parent:
        raise RuntimeError("terminal supervisor exited before worker startup")


def _worker_entry(output, manifest, role, updates, stop, parent_pid):
    try:
        _parent_death_guard(parent_pid)
        context = _WorkerContext(Path(output), manifest, role, updates, stop)
        result = {"server": _server, "grouper": _grouper, "capture": _capture, "replay": _replay}[role](context)
        updates.put(dict(state="finished", result=result))
    except BaseException as exc:
        updates.put(dict(state="failed", error=f"{type(exc).__name__}: {exc}", traceback=traceback.format_exc()))
        traceback.print_exc()
        raise


def _shutdown_worker(worker, stop):
    stop.set()
    if worker.pid is None:
        return
    worker.join(timeout=1)
    if worker.is_alive():
        try:
            os.kill(worker.pid, signal.SIGINT)
        except ProcessLookupError:
            pass
        worker.join(timeout=3)
    if worker.is_alive():
        worker.terminate()
        worker.join(timeout=3)
    if worker.is_alive():
        worker.kill()
        worker.join(timeout=3)
    if worker.is_alive():
        raise RuntimeError(f"worker {worker.pid} did not exit after kill")


def _claim(output, manifest, role):
    if role not in ROLES:
        raise ValueError(f"unknown session component: {role}")
    claim = output / "_session" / f"{role}.claim"
    try:
        with claim.open("x") as stream:
            stream.write(str(os.getpid()) + "\n")
    except FileExistsError as exc:
        raise RuntimeError(f"{role} was already launched in this session; prepare a fresh session for another run") from exc
    return dict(session_id=manifest["session_id"], configuration_sha256=manifest["configuration_sha256"],
                role=role, state="starting", pid=os.getpid(),
                process_token=_process_token(os.getpid()), started_monotonic_s=time.monotonic())


def _deadline(manifest, own_status, replay_status):
    started = replay_status.get("replay_started_monotonic_s") if replay_status else None
    if started is None:
        return own_status["started_monotonic_s"] + manifest["startup_timeout_seconds"]
    return (started + manifest["duration_seconds"] + manifest["drain_timeout_seconds"]
            + manifest["write_timeout_seconds"] + 30.0)


def _failed_run(output, manifest, error):
    path = output / "run.json"
    report = _json(path) if path.exists() else dict(version=1, mode="online", launch_mode="separate_terminals")
    report.update(state="failed", complete=False, session_id=manifest["session_id"], error=error)
    _write(path, report)


def run_component(output_dir, role):
    output = Path(output_dir).resolve()
    manifest = _manifest(output, check_source=True)
    _check_session(output, manifest)
    status = _claim(output, manifest, role)
    status_path = output / "_session" / f"{role}.json"
    _write(status_path, status)
    ctx = multiprocessing.get_context("spawn")
    updates, stop = ctx.Queue(), ctx.Event()
    worker = ctx.Process(target=_worker_entry,
        args=(str(output), manifest, role, updates, stop, os.getpid()))
    finished = False
    result = None
    previous_term = signal.getsignal(signal.SIGTERM)

    def interrupted(signum, frame):
        raise KeyboardInterrupt(f"received signal {signum}")

    signal.signal(signal.SIGTERM, interrupted)
    try:
        worker.start()
        status["worker_pid"] = worker.pid
        _write(status_path, status)
        while True:
            try:
                update = updates.get(timeout=POLL_SECONDS)
            except queue.Empty:
                update = None
            if update is not None:
                if update["state"] == "failed":
                    raise RuntimeError(update["error"])
                if update["state"] == "finished":
                    finished, result = True, update.get("result")
                else:
                    status.update(update)
                    _write(status_path, status)
            _check_session(output, manifest, ignore_role=role)
            if time.monotonic() > _deadline(manifest, status, _role_status(output, manifest, "replay")):
                raise TimeoutError("session startup or completion deadline exceeded")
            if worker.exitcode is not None:
                if worker.exitcode != 0:
                    raise RuntimeError(f"{role} worker exited with status {worker.exitcode}")
                if finished:
                    break
                # Queue data must be consumed before recognizing clean exit.
                try:
                    update = updates.get(timeout=1)
                except queue.Empty as exc:
                    raise RuntimeError("worker exited without a completion acknowledgment") from exc
                if update["state"] == "finished":
                    finished, result = True, update.get("result")
                elif update["state"] == "failed":
                    raise RuntimeError(update["error"])
                else:
                    status.update(update)
                    _write(status_path, status)
        worker.join()
        with _decision_lock(output):
            _check_session(output, manifest, ignore_role=role)
            if role == "server":
                peers = {r: _role_status(output, manifest, r) for r in ROLES if r != "server"}
                if not all(s and s["state"] == "complete" and s.get("exitcode") == 0 for s in peers.values()):
                    raise RuntimeError("server completion requires clean completion of all other components")
                result.update(state="complete", complete=True, component_status=peers)
                _write(output / "run.json", result)
            status.update(state="complete", exitcode=0, completed_monotonic_s=time.monotonic())
            _write(status_path, status)
        _log(role, "Complete.")
        return status
    except BaseException as exc:
        status.update(state="failed", error=f"{type(exc).__name__}: {exc}")
        try:
            _write(status_path, status)
        except BaseException as persistence:
            exc.add_note(f"Could not persist component failure: {persistence!r}")
        _log(role, f"Stopping: {status['error']}")
        previous_int = signal.signal(signal.SIGINT, signal.SIG_IGN)
        try:
            _shutdown_worker(worker, stop)
        except BaseException as cleanup:
            exc.add_note(f"Worker shutdown failure: {cleanup!r}")
        finally:
            signal.signal(signal.SIGINT, previous_int)
        if role == "server":
            try:
                _failed_run(output, manifest, status["error"])
            except BaseException as persistence:
                exc.add_note(f"Could not persist failed run: {persistence!r}")
        raise
    finally:
        signal.signal(signal.SIGTERM, previous_term)
        updates.close()
        updates.join_thread()


def add_session_parser(subparsers):
    parser = subparsers.add_parser("session", help="Run a controlled observation in four separate terminals")
    commands = parser.add_subparsers(dest="session_command", required=True)
    prepare = commands.add_parser("prepare", help="Create a fresh shared session and print the four commands")
    prepare.add_argument("bundle")
    prepare.add_argument("output")
    prepare.add_argument("--gpu", type=int, default=0)
    prepare.add_argument("--base-port", type=int)
    prepare.add_argument("--suppress-early-capture", action="store_true")
    prepare.add_argument("--startup-timeout-seconds", type=float, default=600.0)
    prepare.add_argument("--drain-timeout-seconds", type=float, default=120.0)
    prepare.add_argument("--max-lag-seconds", type=float, default=1.0)
    descriptions = dict(grouper="Run the shared scientific grouper", server="Run FrbServer and the GPU dedisperser",
        capture="Run the classifier-free capture receiver", replay="Replay the saved observation at 1×",
        status="Show component state and addresses", stop="Request coordinated session shutdown")
    for role, description in descriptions.items():
        command = commands.add_parser(role, help=description)
        command.add_argument("output", help="Shared directory created by session prepare")
    parser.set_defaults(func=session_command)


def session_command(args):
    command = args.session_command
    if command == "prepare":
        prepare_session(args.bundle, args.output, cuda_device_id=args.gpu, base_port=args.base_port,
            accept_early=not args.suppress_early_capture, startup_timeout_seconds=args.startup_timeout_seconds,
            drain_timeout_seconds=args.drain_timeout_seconds, max_lag_seconds=args.max_lag_seconds)
    elif command == "status":
        print(json.dumps(session_status(args.output), indent=2))
    elif command == "stop":
        print(json.dumps(stop_session(args.output), indent=2))
    else:
        run_component(args.output, command)
