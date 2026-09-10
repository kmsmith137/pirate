"""Finite CHORD replay orchestration and command-line entry points."""

from __future__ import annotations

import hashlib
import json
import math
import multiprocessing
from pathlib import Path
import queue
import socket
import time
import traceback

import yaml

from .ControlledCapture import CapturePolicy, ControlledCaptureReceiver, write_json


def _fresh_directory(path):
    path = Path(path).resolve()
    path.mkdir(parents=True, exist_ok=False)
    return path


def _local_addresses(count):
    sockets = []
    try:
        for _ in range(count):
            sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
            sock.bind(("127.0.0.1", 0))
            sockets.append(sock)
        return [f"127.0.0.1:{s.getsockname()[1]}" for s in sockets]
    finally:
        for sock in sockets:
            sock.close()


def _source_identity():
    # Hash executed modules as well as recording a Git revision; experiments
    # conducted with local edits must not be attributed to an unchanged HEAD.
    import subprocess
    root = Path(__file__).resolve().parent.parent
    modules = ("ControlledObservation.py", "ReplayObservation.py", "SharedGrouper.py",
               "OnlineGrouper.py", "ControlledCapture.py", "ControlledExperiment.py",
               "Peakfinders.py", "GpuArgmaxDecoder.py", "OfflineCandidateGrouper.py")
    result = {name: hashlib.sha256((root / "pirate_frb" / name).read_bytes()).hexdigest()
              for name in modules}
    try:
        revision = subprocess.check_output(["git", "-C", str(root), "rev-parse", "HEAD"], text=True).strip()
    except (OSError, subprocess.CalledProcessError):
        revision = None
    return dict(git_revision=revision, module_sha256=result)


def run_controlled_offline(bundle_dir, output_dir, *, cuda_device_id=0):
    """Save offline maps and run the shared scientific grouper on exact raw input."""
    from .ControlledObservation import load_experiment_bundle
    from .run_offline_dedisperser import run_offline_dedisperser
    from .run_offline_grouper import run_offline_grouper
    import cupy as cp

    bundle = load_experiment_bundle(bundle_dir, verify_hashes=True)
    output = _fresh_directory(output_dir)
    acq = output / "maps"
    acq.mkdir()
    report = dict(version=1, mode="offline", state="running", complete=False,
                  bundle_manifest_sha256=bundle["manifest_sha256"],
                  source=_source_identity(), cuda_device_id=cuda_device_id,
                  weights_mode="analytic", detrending=False, shared_processing="SharedGrouper",
                  noise_variance=bundle["metadata"].get("noise_variance"))
    began = time.monotonic()
    write_json(output / "run.json", report)
    try:
        for entry in bundle["frame_entries"]:
            src = Path(bundle["bundle_dir"]) / entry["path"]
            (acq / src.name).symlink_to(src)
        # The legacy offline dedisperser currently selects CUDA device 0
        # explicitly. Keep its public behavior and fail clearly on unsupported
        # selection until that API gains a device argument.
        if cuda_device_id != 0:
            raise ValueError("controlled offline dedispersion currently requires CUDA device 0")
        with cp.cuda.Device(cuda_device_id):
            run_offline_dedisperser(str(acq), bundle["dedispersion_config_path"], save=True)
            run_offline_grouper(str(acq), bundle["grouper_config_path"],
                                cuda_device_id=cuda_device_id, output=str(output / "events.asdf"))
        report.update(state="complete", complete=True, elapsed_seconds=time.monotonic() - began)
        write_json(output / "run.json", report)
        return report
    except BaseException as exc:
        report.update(state="failed", error=f"{type(exc).__name__}: {exc}",
                      elapsed_seconds=time.monotonic() - began)
        write_json(output / "run.json", report)
        raise


def _grouper_process(bundle_dir, output, address, sifter_address,
                     stop_event, ready_event, completion_event, status_queue):
    import signal
    signal.signal(signal.SIGINT, signal.SIG_IGN)
    from .OnlineGrouper import run_online_grouper
    try:
        run_online_grouper(bundle_dir, output, address, sifter_addr=sifter_address,
                           stop_event=stop_event, ready_event=ready_event,
                           completion_event=completion_event, status_queue=status_queue)
    except BaseException:
        traceback.print_exc()
        raise


class _ExperimentServer:
    """One real FrbServer with finite, separately sized raw and dedispersion pools."""

    def __init__(self, bundle, output, policy, data_address, rpc_address,
                 grouper_address, cuda_device_id):
        self.server = None
        self.file_writer = None
        self.allocator = None
        self.grouper_client = None
        self.receivers = []
        self.objects = []
        try:
            self._build(bundle, output, policy, data_address, rpc_address,
                        grouper_address, cuda_device_id)
        except BaseException as exc:
            # Construction may have started writer threads before a subsequent
            # allocation or Receiver/FrbServer constructor failed.
            errors = self._stop_resources()
            if errors:
                exc.add_note("Partial server construction cleanup: " + "; ".join(errors))
                exc.controlled_cleanup_errors = errors
            raise

    def _build(self, bundle, output, policy, data_address, rpc_address,
               grouper_address, cuda_device_id):
        import cupy as cp
        import ksgpu
        from .core import (AssembledFrameAllocator, BumpAllocator, CudaStreamPool,
                           FileWriter, Receiver, SlabAllocator)
        from .pirate_pybind11 import (DedispersionConfig, DedispersionPlan,
                                     FrbGrouperClient, FrbServer, GpuDedisperser)
        config = DedispersionConfig.from_yaml(bundle["dedispersion_config_path"])
        md = bundle["metadata"]
        config.zone_nfreq = list(md["zone_nfreq"])
        config.zone_freq_edges = list(md["zone_freq_edges"])
        config.time_sample_ms = bundle["time_sample_ms"]
        config.beams_per_gpu = len(bundle["beam_ids"])
        config.validate()
        nbeam, nfreq, ntime = len(bundle["beam_ids"]), sum(md["zone_nfreq"]), bundle["samples_per_chunk"]
        nstream, batch = config.num_active_batches, config.beams_per_batch
        if 2 * nstream * batch > nbeam:
            raise ValueError("live grouper needs at least 2*num_active_batches*beams_per_batch beams")
        chunk_seconds = ntime * bundle["time_sample_ms"] * 1e-3
        ring_chunks = max(7, math.ceil(policy.buffer_seconds / chunk_seconds))
        (output / "ssd").mkdir()
        (output / "captures").mkdir()
        with cp.cuda.Device(cuda_device_id):
            plan = DedispersionPlan(config)
            streams = CudaStreamPool(nstream)
            probe = GpuDedisperser(plan, streams, cuda_device_id=cuda_device_id,
                                   num_consumers=1, nbatches_out=2*nstream, nbatches_wt=nstream)
            gpu_dedispersion_bytes = probe.resource_tracker.get_gmem_footprint()
            host_dedispersion_bytes = probe.resource_tracker.get_hmem_footprint()
            scratch = nstream * batch * nfreq * (ntime // 2 + (ntime // 256) * 4)
            gpu_bytes = int(gpu_dedispersion_bytes + scratch + (1 << 16))
            host_bytes = int(host_dedispersion_bytes + (1 << 16))
            slab_bytes = int(AssembledFrameAllocator.slab_nbytes(nfreq, ntime))
            # Separate pools make dedispersion allocation independent of raw
            # retention. Extra slabs cover assembly/allocator/writer queues.
            raw_bytes = slab_bytes * nbeam * (ring_chunks + 8) + (1 << 16)
            raw_bump = BumpAllocator(ksgpu.af_rhost | ksgpu.af_zero, raw_bytes,
                                      cuda_device=cuda_device_id)
            self.objects.append(raw_bump)
            raw_slabs = SlabAllocator(raw_bump)
            self.objects.append(raw_slabs)
            self.allocator = AssembledFrameAllocator(raw_slabs, num_consumers=1,
                                                       time_samples_per_chunk=ntime)
            host_bump = BumpAllocator(ksgpu.af_rhost | ksgpu.af_zero, host_bytes,
                                       cuda_device=cuda_device_id)
            self.objects.append(host_bump)
            gpu_bump = BumpAllocator(ksgpu.af_gpu | ksgpu.af_zero, gpu_bytes,
                                      cuda_device=cuda_device_id)
            self.objects.append(gpu_bump)
            self.file_writer = FileWriter(str(output / "ssd"), str(output / "captures"),
                                          num_ssd_threads=1, num_nfs_threads=1)
            self.receivers = [Receiver(address=data_address, allocator=self.allocator)]
            self.grouper_client = FrbGrouperClient(grouper_address)
            self.server = FrbServer(config, self.receivers, self.file_writer,
                                    rpc_address, ring_chunks, min_data_mtu=1500,
                                    host_allocator=host_bump, gpu_allocator=gpu_bump,
                                    cuda_device_id=cuda_device_id,
                                    grouper_client=self.grouper_client,
                                    nbatches_wt=nstream, quiet=True,
                                    max_unprocessed_chunks=5)
        self.memory = dict(raw_pool_bytes=raw_bytes, raw_slab_bytes=slab_bytes,
                           host_dedispersion_pool_bytes=host_bytes, gpu_pool_bytes=gpu_bytes,
                           ringbuf_nchunks=ring_chunks,
                           nominal_retention_seconds=ring_chunks * chunk_seconds,
                           raw_headroom_chunks=8, shared_host_allocator=False)

    def start(self, check_child, timeout=120):
        self.grouper_client.ping(timeout_ms=int(timeout * 1000))
        self.server.start()
        deadline = time.monotonic() + timeout
        for receiver in self.receivers:
            while not receiver.wait_until_listening(timeout_sec=0.25):
                check_child()
                self.server.poll_from_python(timeout_ms=0)
                if time.monotonic() > deadline:
                    raise TimeoutError("receiver did not begin listening")

    def _stop_resources(self):
        errors = []
        actions = [("server.stop", self.server)]
        actions.extend((f"receiver[{i}].stop", receiver)
                       for i, receiver in enumerate(self.receivers))
        actions.extend((("allocator.stop", self.allocator),
                        ("file_writer.stop", self.file_writer)))
        for name, resource in actions:
            if resource is None:
                continue
            try:
                resource.stop()
            except BaseException as exc:
                errors.append(f"{name}: {type(exc).__name__}: {exc}")
        return errors

    def stop(self):
        errors = self._stop_resources()
        if errors:
            exc = RuntimeError("server resource cleanup failed: " + "; ".join(errors))
            exc.controlled_cleanup_errors = errors
            raise exc

    def producer_metadata(self):
        """Snapshot the running producer, never the allocation-footprint probe."""
        import numpy as np
        from .ArgmaxMetadata import ARGMAX_ENCODING
        plan = self.server.plan if self.server is not None else None
        dd = self.server.dedisperser if self.server is not None else None
        metadata = self.allocator.metadata if self.allocator is not None else None
        if plan is None or dd is None or metadata is None:
            return None
        variances = np.ascontiguousarray(metadata.get_channel_variances(), dtype="<f8")
        return dict(
            config_yaml=plan.config.to_yaml_string(), plan_yaml=plan.to_yaml_string(),
            dcores=[int(value) for value in dd.Dcores], argmax_encoding=ARGMAX_ENCODING,
            noise_variance=[float(value) for value in metadata.noise_variance],
            channel_variance_sha256=hashlib.sha256(variances.tobytes()).hexdigest(),
            channel_variance_dtype="<f8", channel_variance_count=int(variances.size),
            metadata_yaml=metadata.to_yaml_string(),
            weights_evidence="FrbServer analytic initialization from received channel variances",
            weight_values_read_back=False)




_SHUTDOWN_TIMEOUT_SECONDS = 10.0


def _cleanup_online_resources(capture, server, stop_event, child, status_queue, report):
    """Attempt every cleanup independently; return all failures as report data."""
    errors = report.setdefault("cleanup_errors", [])

    def failed(name, exc):
        errors.append(dict(action=name, error=f"{type(exc).__name__}: {exc}"))
        report.update(state="failed", complete=False)

    def attempt(name, action):
        try:
            return action()
        except BaseException as exc:
            failed(name, exc)
            return None

    if capture is not None:
        attempt("capture.close", capture.close)
    if server is not None:
        attempt("server.stop", server.stop)
    attempt("grouper.stop_event.set", stop_event.set)
    pid = attempt("grouper.pid", lambda: child.pid)
    if pid is not None:
        def alive():
            try:
                return child.is_alive()
            except BaseException as exc:
                failed("grouper.is_alive", exc)
                return True
        attempt("grouper.join", lambda: child.join(timeout=_SHUTDOWN_TIMEOUT_SECONDS))
        if alive():
            failed("grouper.shutdown", RuntimeError("grouper required termination"))
            attempt("grouper.terminate", child.terminate)
            attempt("grouper.join_after_terminate", lambda: child.join(timeout=_SHUTDOWN_TIMEOUT_SECONDS))
        if alive():
            attempt("grouper.kill", child.kill)
            attempt("grouper.join_after_kill", lambda: child.join(timeout=_SHUTDOWN_TIMEOUT_SECONDS))
        if alive():
            failed("grouper.shutdown", RuntimeError(f"grouper PID {pid} remains alive after kill"))
        else:
            code = attempt("grouper.exitcode", lambda: child.exitcode)
            report["grouper_exitcode"] = code
            if code not in (None, 0):
                failed("grouper.exitcode", RuntimeError(f"grouper exited with status {code}"))
    if server is not None and server.server is not None:
        attempt("server.poll_from_python", lambda: server.server.poll_from_python(timeout_ms=0))
    # This process only reads the status queue, so it has no pending producer
    # feeder data. Close and join are nevertheless separate cleanup attempts.
    attempt("status_queue.close", status_queue.close)
    attempt("status_queue.join_thread", status_queue.join_thread)
    return errors


def run_controlled_online(bundle_dir, output_dir, *, accept_early=True,
                          cuda_device_id=0, max_lag_seconds=1.0, timeout_seconds=120.0):
    """Replay at 1x, drain the scientific consumer and verify promised writes."""
    from .ControlledObservation import load_experiment_bundle
    from .ReplayObservation import replay_observation
    bundle = load_experiment_bundle(bundle_dir, verify_hashes=True)
    policy = CapturePolicy.from_dict(yaml.safe_load(Path(bundle["capture_config_path"]).read_text()))
    output = _fresh_directory(output_dir)
    data_addr, rpc_addr, grouper_addr, sifter_addr = _local_addresses(4)
    ctx = multiprocessing.get_context("spawn")
    stop_event, ready_event, completion_event = ctx.Event(), ctx.Event(), ctx.Event()
    status_queue = ctx.Queue()
    child = ctx.Process(target=_grouper_process, args=(bundle["bundle_dir"],
        str(output / "events.asdf"), grouper_addr, sifter_addr, stop_event,
        ready_event, completion_event, status_queue), daemon=True)
    server = capture = None
    report = dict(version=1, mode="online", state="starting", complete=False, accept_early=bool(accept_early),
                  classifier_mode="bypass", bundle_manifest_sha256=bundle["manifest_sha256"],
                  source=_source_identity(), cuda_device_id=cuda_device_id,
                  weights_mode="analytic", detrending=False, shared_processing="SharedGrouper",
                  noise_variance=bundle["metadata"].get("noise_variance"),
                  addresses=dict(data=data_addr, rpc=rpc_addr, grouper=grouper_addr, sifter=sifter_addr),
                  grouper_messages=[])
    primary_failure = None
    work_complete = False

    def record_producer(required=False):
        if "producer" in report:
            return
        actual = server.producer_metadata() if server is not None else None
        if actual is None:
            if required:
                raise RuntimeError("completed live run has no published producer metadata")
            return
        report["producer"] = actual
        report["noise_variance"] = actual["noise_variance"]
        write_json(output / "run.json", report)

    def check_child():
        while True:
            try:
                message = status_queue.get_nowait()
            except queue.Empty:
                break
            report["grouper_messages"].append(message)
            if message.get("phase") == "error":
                raise RuntimeError(f"online grouper failed: {message}")
        if child.exitcode is not None:
            raise RuntimeError(f"online grouper exited before server shutdown: {child.exitcode}")

    try:
        write_json(output / "run.json", report)
        child.start()
        server = _ExperimentServer(bundle, output, policy, data_addr, rpc_addr,
                                    grouper_addr, cuda_device_id)
        report["memory"] = server.memory
        server.start(check_child, timeout_seconds)
        capture = ControlledCaptureReceiver(bundle["metadata"], bundle["samples_per_chunk"],
                                             policy, rpc_addr, output / "capture.json",
                                             accept_early=accept_early,
                                             rpc_timeout_seconds=min(10.0, timeout_seconds),
                                             subscription_ready_timeout_seconds=timeout_seconds,
                                             subscription_timeout_seconds=(bundle["duration_seconds"]
                                                 + 2 * timeout_seconds
                                                 + policy.write_timeout_seconds + 60.0))
        capture.start(sifter_addr)
        report["state"] = "replaying"
        write_json(output / "run.json", report)

        def progress(sender_report):
            check_child()
            server.server.poll_from_python(timeout_ms=0)
            if ready_event.is_set():
                record_producer()
            with capture.cv:
                if capture.ledger["errors"]:
                    raise RuntimeError("; ".join(capture.ledger["errors"]))

        replay = replay_observation(bundle_dir, rpc_addr, str(output / "replay.json"),
                                    rate=1.0, nworkers=1, max_lag_s=max_lag_seconds,
                                    startup_timeout_s=timeout_seconds,
                                    drain_timeout_s=timeout_seconds, progress=progress,
                                    verify_hashes=True)
        report["state"] = "draining"
        write_json(output / "run.json", report)
        deadline = time.monotonic() + timeout_seconds
        while not completion_event.wait(0.25):
            check_child()
            server.server.poll_from_python(timeout_ms=0)
            if time.monotonic() > deadline:
                raise TimeoutError("online grouper did not drain all observation outputs")
        check_child()
        capture.wait_for_writes()
        record_producer(required=True)
        work_complete = True
        report.update(state="stopping", complete=False, scientific_outputs_drained=True, promised_files_drained=True,
                      sender_state=replay.get("state"))
        write_json(output / "run.json", report)
    except BaseException as exc:
        primary_failure = exc
        report.update(state="failed", complete=False,
                      error=f"{type(exc).__name__}: {exc}", traceback=traceback.format_exc())
        construction_errors = getattr(exc, "controlled_cleanup_errors", None)
        if construction_errors:
            report["construction_cleanup_errors"] = list(construction_errors)
        raise
    finally:
        # Each action runs even when an earlier cleanup raises. The producer
        # stop is attempted before asking the consumer to close its IPC view.
        errors = _cleanup_online_resources(
            capture, server, stop_event, child, status_queue, report)
        if primary_failure is None and work_complete and not errors:
            report.update(state="complete", complete=True)
        else:
            report.update(state="failed", complete=False)
        try:
            write_json(output / "run.json", report)
        except BaseException as exc:
            report.setdefault("cleanup_errors", []).append(dict(
                action="run_report.write", error=f"{type(exc).__name__}: {exc}"))
            report.update(state="failed", complete=False)
            # A transient publication failure may permit the failed state to
            # be saved on a second attempt. Never replace the primary error.
            try:
                write_json(output / "run.json", report)
            except BaseException as retry:
                report["cleanup_errors"].append(dict(
                    action="run_report.retry", error=f"{type(retry).__name__}: {retry}"))
        if primary_failure is not None and report.get("cleanup_errors"):
            primary_failure.add_note("Additional controlled-run cleanup failures: "
                                     + repr(report["cleanup_errors"]))
    if report["state"] != "complete":
        raise RuntimeError(f"controlled online run did not complete: {report}")
    return report


def add_experiment_parser(subparsers):
    parser = subparsers.add_parser("experiment", help="Controlled CHORD online/offline observation")
    sub = parser.add_subparsers(dest="experiment_command", required=True)
    prepare = sub.add_parser("prepare", help="Validate configuration and prepare an observation bundle")
    prepare.add_argument("config")
    prepare.add_argument("bundle")
    generate = sub.add_parser("generate", help="Generate and hash raw observation frames once")
    generate.add_argument("bundle")
    for name in ("offline", "online"):
        run = sub.add_parser(name, help=f"Run the controlled observation {name}")
        run.add_argument("bundle")
        run.add_argument("output")
        run.add_argument("--gpu", type=int, default=0)
        if name == "online":
            run.add_argument("--suppress-early-capture", action="store_true")
            run.add_argument("--max-lag-seconds", type=float, default=1.0)
    compare = sub.add_parser("compare", help="Verify recovery agreement and online capture outcomes")
    for name in ("bundle", "offline", "online_early", "online_full", "report"):
        compare.add_argument(name)
    compare.add_argument("--markdown")
    parser.set_defaults(func=experiment_command)


def experiment_command(args):
    from .ControlledObservation import prepare_controlled_observation, generate_controlled_observation
    command = args.experiment_command
    if command == "prepare":
        result = prepare_controlled_observation(args.config, args.bundle)
    elif command == "generate":
        result = generate_controlled_observation(args.bundle)
    elif command == "offline":
        result = run_controlled_offline(args.bundle, args.output, cuda_device_id=args.gpu)
    elif command == "online":
        result = run_controlled_online(args.bundle, args.output,
                                       accept_early=not args.suppress_early_capture,
                                       cuda_device_id=args.gpu, max_lag_seconds=args.max_lag_seconds)
    elif command == "compare":
        from .ControlledComparison import compare_controlled_experiment
        result = compare_controlled_experiment(args.bundle, args.offline, args.online_early,
                                               args.online_full, args.report,
                                               output_markdown=args.markdown)
    else:
        raise ValueError(f"unknown experiment command {command}")
    print(json.dumps({"command": command, "state": result.get("state", result.get("status", "complete"))}, indent=2))
