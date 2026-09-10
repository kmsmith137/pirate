"""Finite, paced replay of a saved controlled observation through FakeXEngine.

The wire transport is unchanged.  Two final junk chunks flush Receiver's
two-chunk assembly window; they are transport padding, never science input.
Sender completion does not imply GPU, grouper, or FileWriter completion.
"""

from __future__ import annotations

from dataclasses import dataclass
import hashlib
import json
import math
from pathlib import Path
import threading
import time


TRANSPORT_TAIL_CHUNKS = 2
_STATUS_FIELDS = ("num_connections", "num_bytes", "rb_start", "rb_reaped",
                  "rb_processed", "rb_streamed", "rb_assembled", "rb_end")
_METADATA_FIELDS = (
    "version", "zone_nfreq", "zone_freq_edges", "beamset", "beam_ids",
    "beam_positions_x", "beam_positions_y", "unix_ns_at_seq_0", "dt_ns_per_seq",
    "seq_per_frb_time_sample", "tel_origin_itrs_lat_deg", "tel_origin_itrs_lon_deg",
    "tel_grid_x_axis", "tel_grid_y_axis", "tel_dish_elev_axis", "tel_dish_vert_axis",
    "tel_dish_coelev_deg", "tel_dish_separation_x_m", "tel_dish_separation_y_m",
    "noise_variance",
)


@dataclass(frozen=True)
class ReplayPlan:
    initial_chunk: int
    nchunks: int
    samples_per_chunk: int
    time_sample_s: float
    beam_ids: tuple[int, ...]
    rate: float = 1.0

    def __post_init__(self):
        for name in ("initial_chunk", "nchunks", "samples_per_chunk"):
            value = getattr(self, name)
            if isinstance(value, bool) or not isinstance(value, int):
                raise ValueError(f"{name} must be an integer")
        if self.initial_chunk < 0 or self.nchunks < 1:
            raise ValueError("initial_chunk must be nonnegative and nchunks positive")
        if self.samples_per_chunk < 256 or self.samples_per_chunk % 256:
            raise ValueError("samples_per_chunk must be a positive multiple of 256")
        if not math.isfinite(self.time_sample_s) or self.time_sample_s <= 0:
            raise ValueError("time_sample_s must be positive and finite")
        if not math.isfinite(self.rate) or self.rate <= 0:
            raise ValueError("rate must be positive and finite")
        if (not self.beam_ids or len(set(self.beam_ids)) != len(self.beam_ids)
                or any(isinstance(b, bool) or not isinstance(b, int) or b < 0
                       for b in self.beam_ids)):
            raise ValueError("beam_ids must be distinct nonnegative integers")

    @property
    def minichunks_per_chunk(self):
        return self.samples_per_chunk // 256

    @property
    def chunk_duration_s(self):
        return self.samples_per_chunk * self.time_sample_s


def _checked(result, operation):
    if result is False:
        raise RuntimeError(f"Replay stopped before completing {operation}")


def _status_snapshot(status, plan, elapsed_s):
    get = status.get if isinstance(status, dict) else lambda k: getattr(status, k)
    values = {key: int(get(key)) for key in _STATUS_FIELDS}
    nbeams = len(plan.beam_ids)
    values.update(
        elapsed_s=float(elapsed_s),
        assembled_backlog_chunks=(values["rb_assembled"] - values["rb_processed"]) / nbeams,
        logical_retention_s=(values["rb_end"] - values["rb_start"]) / nbeams
            * plan.chunk_duration_s,
        unreaped_retention_s=(values["rb_end"] - values["rb_reaped"]) / nbeams
            * plan.chunk_duration_s,
    )
    return values


def drive_replay(plan, sender, frame_loader, *, status_reader=None,
                 clock=time.monotonic, sleep=time.sleep, max_lag_s=None,
                 progress=None, persist=None, report=None):
    """Drive an injectable sender. No CUDA or package-native imports occur here.

    A deadline denotes the END of a minichunk's observing interval. A sender
    may be late but must never send before that deadline. All workers are
    synchronized at each chunk boundary: this bounds ownership and measures
    completed wire dispatch, including native pacing waits, not just enqueue
    latency. ``progress(report)`` runs at successful chunk boundaries and completion;
    ``persist(report)`` also runs on failure without invoking progress callbacks.

    ``report`` is populated even when an exception interrupts the run. The
    caller owns sender lifetime; keeping it alive after this returns permits
    the harness to drain the downstream pipeline before orderly shutdown.
    """
    if max_lag_s is not None and (not math.isfinite(max_lag_s) or max_lag_s < 0):
        raise ValueError("max_lag_s must be nonnegative and finite")
    if int(sender.minichunks_per_chunk) != plan.minichunks_per_chunk:
        raise ValueError("sender minichunk geometry differs from observation")
    if int(sender.nworkers) < 1:
        raise ValueError("sender must have at least one worker")
    result = {} if report is None else report
    result.update(
        format_version=1, state="running", rate=plan.rate,
        realtime_rate=plan.rate == 1.0, initial_chunk=plan.initial_chunk,
        science_nchunks=plan.nchunks, transport_tail_chunks=TRANSPORT_TAIL_CHUNKS,
        samples_per_chunk=plan.samples_per_chunk, beam_ids=list(plan.beam_ids),
        time_sample_s=plan.time_sample_s,
        science_duration_s=plan.nchunks * plan.chunk_duration_s,
        minichunks=[], chunks=[], server_status=[],
        server_gpu_grouper_and_files_drained=False,
    )
    mpc = plan.minichunks_per_chunk
    first_mc = plan.initial_chunk * mpc
    interval_s = 256 * plan.time_sample_s / plan.rate
    origin = clock()
    result["clock_origin_monotonic_s"] = origin
    result["max_enqueue_lag_s"] = 0.0
    result["max_chunk_dispatch_lag_s"] = 0.0
    try:
        for offset in range(plan.nchunks + TRANSPORT_TAIL_CHUNKS):
            chunk = plan.initial_chunk + offset
            science = offset < plan.nchunks
            # Load once per real chunk. C++ sender command queues retain their
            # own shared_ptr until dispatch; no frame is reused or modified.
            fset = frame_loader(chunk) if science else None
            if science and fset is None:
                raise RuntimeError(f"No frame set for science chunk {chunk}")
            if science and int(fset.time_chunk_index) != chunk:
                raise ValueError(f"Frame source returned wrong chunk for {chunk}")
            for within in range(mpc):
                imc = chunk * mpc + within
                relative_mc = imc - first_mc
                deadline = origin + (relative_mc + 1) * interval_s
                # Native queue ownership stays at most two minichunks ahead
                # of every worker, even when loading/sending falls behind.
                if relative_mc >= 2:
                    for worker in range(sender.nworkers):
                        _checked(sender.wait_until_processed(worker, imc - 2),
                                 f"worker {worker}, minichunk {imc - 2}")
                while True:
                    remaining = deadline - clock()
                    if remaining <= 0:
                        break
                    sleep(min(remaining, 0.25))
                started = clock()
                for worker in range(sender.nworkers):
                    sent = (sender.enqueue_send_minichunk(worker, imc, fset)
                            if science else sender.enqueue_send_junk(worker, imc))
                    _checked(sent, f"enqueue worker {worker}, minichunk {imc}")
                finished = clock()
                lag = max(0.0, finished - deadline)
                result["max_enqueue_lag_s"] = max(result["max_enqueue_lag_s"], lag)
                result["minichunks"].append(dict(
                    minichunk_index=imc, time_chunk_index=chunk,
                    kind="science" if science else "transport_tail",
                    deadline_elapsed_s=deadline-origin,
                    enqueue_started_elapsed_s=started-origin,
                    enqueue_finished_elapsed_s=finished-origin, enqueue_lag_s=lag))
                if max_lag_s is not None and lag > max_lag_s:
                    raise RuntimeError(f"Replay enqueue lag {lag:.6f}s exceeds {max_lag_s}s")
            for worker in range(sender.nworkers):
                _checked(sender.synchronize(worker), f"drain worker {worker}, chunk {chunk}")
            completed = clock()
            dispatch_lag = max(0.0, completed - deadline)
            result["max_chunk_dispatch_lag_s"] = max(
                result["max_chunk_dispatch_lag_s"], dispatch_lag)
            result["chunks"].append(dict(
                time_chunk_index=chunk, kind="science" if science else "transport_tail",
                dispatch_complete_elapsed_s=completed-origin, dispatch_lag_s=dispatch_lag))
            if status_reader is not None:
                result["server_status"].append(
                    _status_snapshot(status_reader(), plan, clock()-origin))
            if persist is not None:
                persist(result)
            if progress is not None:
                progress(result)
            if max_lag_s is not None and dispatch_lag > max_lag_s:
                raise RuntimeError(
                    f"Replay dispatch lag {dispatch_lag:.6f}s exceeds {max_lag_s}s")
        result["state"] = "sender_complete"
        result["elapsed_s"] = clock() - origin
        if persist is not None:
            persist(result)
        if progress is not None:
            progress(result)
        return result
    except BaseException as exc:
        result.update(state="failed", error=f"{type(exc).__name__}: {exc}",
                      elapsed_s=clock()-origin)
        _persist_failure(persist, result, exc)
        raise


def _persist_failure(persist, report, original):
    """Preserve the first error even if recording its evidence also fails."""
    if persist is not None:
        try:
            persist(report)
        except BaseException as secondary:
            original.add_note(f"Failure report could not be persisted: {type(secondary).__name__}: {secondary}")


def metadata_identity(metadata, *, beam_id=None):
    """Plain metadata identity, optionally projected to one saved frame beam."""
    identity = {}
    for name in _METADATA_FIELDS:
        value = getattr(metadata, name)
        identity[name] = list(value) if isinstance(value, (list, tuple)) else value
    if beam_id is not None:
        index = identity["beam_ids"].index(beam_id)
        for name in ("beam_ids", "beam_positions_x", "beam_positions_y"):
            identity[name] = [identity[name][index]]
    return identity


def copy_packed_frame(source, destination, canonical_metadata, beam_id, chunk):
    """Validate a saved frame and copy packed bytes without requantization."""
    import numpy as np

    if (int(source.beam_id) != beam_id or int(source.time_chunk_index) != chunk
            or int(destination.beam_id) != beam_id
            or int(destination.time_chunk_index) != chunk):
        raise ValueError("Saved frame beam/chunk identity does not match replay destination")
    if source.ntime != destination.ntime or source.nfreq != destination.nfreq:
        raise ValueError("Saved frame shape does not match replay destination")
    if metadata_identity(source.metadata) != metadata_identity(canonical_metadata, beam_id=beam_id):
        raise ValueError(f"Saved frame metadata mismatch for beam {beam_id}, chunk {chunk}")
    for name, dtype in (("data", np.uint8), ("scales_offsets", np.float16)):
        src, dst = getattr(source, name), getattr(destination, name)
        if src.shape != dst.shape or src.dtype != dtype or dst.dtype != dtype:
            raise ValueError(f"Saved frame {name} shape/dtype mismatch")
        np.copyto(dst, src, casting="no")


class PackedFrameSource:
    def __init__(self, bundle, bundle_dir, metadata, allocator, read_frame):
        self.metadata = metadata
        self.allocator = allocator
        self.read_frame = read_frame
        self.paths = {}
        for entry in bundle["frame_entries"]:
            key = int(entry["time_chunk_index"]), int(entry["beam_id"])
            if key in self.paths:
                raise ValueError(f"Duplicate acquisition frame {key}")
            self.paths[key] = Path(bundle_dir) / entry["path"]

    def __call__(self, chunk):
        fset = self.allocator.get_frame_set(chunk)
        for index, beam in enumerate(self.metadata.beam_ids):
            key = chunk, int(beam)
            if key not in self.paths:
                raise ValueError(f"Missing acquisition frame {key}")
            source = self.read_frame(str(self.paths[key]))
            copy_packed_frame(source, fset.get_frame(index), self.metadata, int(beam), chunk)
        fset.validate()
        return fset


class _BlockingDeadline:
    """One watchdog for native calls that expose no Python timeout argument."""
    def __init__(self, timeout_s, stop):
        self.timeout_s = timeout_s
        self.stop = stop
        self.condition = threading.Condition()
        self.deadline = None
        self.operation = None
        self.closed = False
        self.timed_out = None
        self.thread = threading.Thread(target=self._watch, daemon=True)
        self.thread.start()

    def call(self, operation, function, *args):
        with self.condition:
            if self.timed_out is not None:
                raise TimeoutError(self.timed_out)
            self.operation = operation
            self.deadline = time.monotonic() + self.timeout_s
            self.condition.notify_all()
        try:
            value = function(*args)
        finally:
            with self.condition:
                self.deadline = None
                self.condition.notify_all()
        if self.timed_out is not None:
            raise TimeoutError(self.timed_out)
        return value

    def _watch(self):
        with self.condition:
            while not self.closed:
                if self.deadline is None:
                    self.condition.wait()
                    continue
                delay = self.deadline - time.monotonic()
                if delay > 0:
                    self.condition.wait(delay)
                    continue
                self.timed_out = f"Replay timed out after {self.timeout_s}s in {self.operation}"
                break
            else:
                return
        self.stop()

    def close(self):
        with self.condition:
            self.closed = True
            self.condition.notify_all()
        self.thread.join(timeout=2.0)


class _TimedSender:
    def __init__(self, sender, deadline):
        self.sender, self.deadline = sender, deadline
        self.nworkers = sender.nworkers
        self.minichunks_per_chunk = sender.minichunks_per_chunk

    def __getattr__(self, name):
        method = getattr(self.sender, name)
        return lambda *args: self.deadline.call(name, method, *args)


def _persist_report(path, report, *, create=False):
    if create:
        with path.open("x") as stream:
            json.dump(report, stream, indent=2, allow_nan=False)
            stream.write("\n")
        return
    temporary = path.with_name(path.name + ".tmp")
    with temporary.open("w") as stream:
        json.dump(report, stream, indent=2, allow_nan=False)
        stream.write("\n")
    temporary.replace(path)


def replay_observation(bundle_dir, rpc_address, result_path, *, rate=1.0,
                       nworkers=1, startup_timeout_s=120.0, drain_timeout_s=120.0,
                       max_lag_s=None, progress=None, verify_hashes=True):
    """Replay a complete bundle and write a durable report, including on failure.

    This function owns the sender, never the server or grouper. It sends exactly
    the saved science frames and two explicit transport-tail chunks, waits for
    sender dispatch, and closes sender connections. The harness must separately
    await expected outputs and file completions before stopping the server.
    Runtime does not read ``injections.json`` or use injection truth.
    """
    for name, value in (("startup_timeout_s", startup_timeout_s),
                        ("drain_timeout_s", drain_timeout_s)):
        if not math.isfinite(value) or value <= 0:
            raise ValueError(f"{name} must be positive and finite")
    if isinstance(nworkers, bool) or not isinstance(nworkers, int) or nworkers < 1:
        raise ValueError("nworkers must be a positive integer")
    output = Path(result_path)
    output.parent.mkdir(parents=True, exist_ok=True)
    report = dict(format_version=1, state="preparing", bundle_dir=str(Path(bundle_dir).resolve()),
                  rpc_address=rpc_address)
    _persist_report(output, report, create=True)
    sender = allocator = client = watchdog = None

    def save(current):
        _persist_report(output, current)

    try:
        from .ControlledObservation import load_experiment_bundle
        from .core import (AssembledFrame, AssembledFrameAllocator, BumpAllocator,
                           FakeXEngine, SlabAllocator, XEngineMetadata)
        from .Hardware import Hardware
        from .rpc import FrbSearchClient
        from .rpc.grpc import frb_search_pb2
        from .utils import ThreadAffinity, check_mtu, extract_ip

        bundle = load_experiment_bundle(bundle_dir, verify_hashes=verify_hashes,
                                        require_complete=True)
        root = Path(bundle_dir)
        xmd = XEngineMetadata.from_yaml_file(str(root / bundle["metadata_file"]))
        plan = ReplayPlan(int(bundle["initial_chunk"]), int(bundle["nchunks"]),
                          int(bundle["samples_per_chunk"]),
                          float(xmd.dt_ns_per_seq) * int(xmd.seq_per_frb_time_sample) * 1e-9,
                          tuple(int(b) for b in xmd.beam_ids), rate=float(rate))
        if list(plan.beam_ids) != list(bundle["beam_ids"]):
            raise ValueError("Bundle beam ordering differs from canonical metadata")
        if not math.isclose(plan.time_sample_s * 1e3, float(bundle["time_sample_ms"]),
                            rel_tol=1e-12, abs_tol=1e-12):
            raise ValueError("Bundle cadence differs from canonical metadata")
        report["bundle_sha256"] = hashlib.sha256((root / "bundle.json").read_bytes()).hexdigest()
        report["metadata_sha256"] = hashlib.sha256(
            (root / bundle["metadata_file"]).read_bytes()).hexdigest()
        report["input_hashes_verified"] = verify_hashes
        client = FrbSearchClient(rpc_address)
        version = frb_search_pb2.PROTOCOL_VERSION_CURRENT
        cfg = client.stub.GetConfig(frb_search_pb2.GetConfigRequest(protocol_version=version),
                                    timeout=startup_timeout_s, wait_for_ready=True)
        existing = client.stub.GetXEngineMetadata(
            frb_search_pb2.GetXEngineMetadataRequest(protocol_version=version),
            timeout=startup_timeout_s)
        if existing.yaml_string:
            raise ValueError("Replay requires a fresh server with no previous X-engine acquisition")
        addrs = list(cfg.data_ip_addrs)
        if not addrs or nworkers % len(addrs):
            raise ValueError("nworkers must be a multiple of the nonempty receiver address count")
        if nworkers > xmd.get_total_nfreq():
            raise ValueError("nworkers exceeds observation frequency channel count")
        if cfg.time_samples_per_chunk != plan.samples_per_chunk:
            raise ValueError("Server chunk length differs from observation")
        if cfg.beams_per_batch < 1 or len(plan.beam_ids) % cfg.beams_per_batch:
            raise ValueError("Observation beam count must be divisible by server beams_per_batch")
        report["server_config"] = dict(data_ip_addrs=addrs,
            ringbuf_nchunks=int(cfg.ringbuf_nchunks), beams_per_batch=int(cfg.beams_per_batch),
            time_samples_per_chunk=int(cfg.time_samples_per_chunk))

        hardware = Hardware()
        affinities = []
        for address in addrs:
            ip = extract_ip(address)
            affinity = hardware.vcpu_list_from_ip_addr(ip, is_dst_addr=True)
            check_mtu(hardware, "controlled observation replay", ip, cfg.min_data_mtu,
                      "min_data_mtu", is_dst_addr=True)
            affinities.append(affinity)
        if any(hardware.cpu_from_vcpu_list(a) != hardware.cpu_from_vcpu_list(affinities[0])
               for a in affinities):
            raise ValueError("Replay receiver routes must use NICs on one CPU")
        with ThreadAffinity(affinities[0]):
            # Five queued allocator sets plus the set being built and up to
            # two sender-held sets. Never retain the full observation in RAM.
            capacity = 8 * len(plan.beam_ids) * AssembledFrameAllocator.slab_nbytes(
                xmd.get_total_nfreq(), plan.samples_per_chunk)
            bump = BumpAllocator("af_rhost", capacity)
            slabs = SlabAllocator(bump)
            allocator = AssembledFrameAllocator(slabs, num_consumers=1,
                time_samples_per_chunk=plan.samples_per_chunk, throw_exception_if_empty=False)
            allocator.initialize_metadata(xmd)
            allocator.initialize_initial_chunk(plan.initial_chunk)
            sender = FakeXEngine(xmd, addrs, nworkers,
                time_samples_per_chunk=plan.samples_per_chunk, debug=False,
                paced=True, rpc_address=rpc_address)
            source = PackedFrameSource(bundle, root, xmd, allocator, AssembledFrame.from_asdf)

            def stop_native():
                sender.stop()
                allocator.stop()

            watchdog = _BlockingDeadline(drain_timeout_s, stop_native)
            timed_sender = _TimedSender(sender, watchdog)

            def read_status():
                return client.stub.GetStatus(
                    frb_search_pb2.GetStatusRequest(protocol_version=version),
                    timeout=min(drain_timeout_s, 10.0))

            drive_replay(plan, timed_sender,
                lambda chunk: watchdog.call(f"load chunk {chunk}", source, chunk),
                status_reader=read_status, max_lag_s=max_lag_s,
                progress=progress, persist=save, report=report)
        return report
    except BaseException as exc:
        report.update(state="failed", error=f"{type(exc).__name__}: {exc}")
        _persist_failure(save, report, exc)
        raise
    finally:
        if watchdog is not None:
            watchdog.close()
        if sender is not None:
            sender.stop()
        if allocator is not None:
            allocator.stop()
        if client is not None:
            client.close()
