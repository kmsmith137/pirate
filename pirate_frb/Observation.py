"""Recipe validation and bounded in-memory intensity streaming."""
from __future__ import annotations

from dataclasses import dataclass
import math
import threading
import time

import numpy as np
import yaml

from .GrouperConfig import GrouperConfig, _StrictSafeLoader

SCHEMA_VERSION = 1
CHORD_ZONE_EDGES = [300, 350, 450, 600, 800, 1500]
TRANSPORT_TAIL_CHUNKS = 2


def _keys(mapping, required, name):
    if not isinstance(mapping, dict) or set(mapping) != set(required):
        raise ValueError(f"{name} requires exactly these keys: {', '.join(required)}")


def _number(value, name, *, minimum=0, integer=False):
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ValueError(f"{name} must be numeric")
    if not math.isfinite(value) or value < minimum:
        raise ValueError(f"{name} must be finite and >= {minimum}")
    if integer and (not isinstance(value, int)):
        raise ValueError(f"{name} must be an integer")
    return int(value) if integer else float(value)


def _read_yaml(path):
    with open(path) as stream:
        return yaml.load(stream, Loader=_StrictSafeLoader)


def validate_recipe(config):
    _keys(config, ("version", "metadata", "dedispersion", "observation",
                   "dedispersion_overrides", "bursts", "grouper"), "observation recipe")
    if type(config["version"]) is not int or config["version"] != SCHEMA_VERSION:
        raise ValueError("unsupported observation recipe version")
    obs = config["observation"]
    _keys(obs, ("duration_seconds", "initial_chunk", "beam_ids", "noise_seed"), "observation")
    _number(obs["duration_seconds"], "duration_seconds", minimum=1)
    if _number(obs["initial_chunk"], "initial_chunk", integer=True) != 0:
        raise ValueError("observations must begin at chunk zero")
    _number(obs["noise_seed"], "noise_seed", integer=True)
    if obs["noise_seed"] >= 2**32:
        raise ValueError("noise_seed must fit uint32")
    beams = obs["beam_ids"]
    if not isinstance(beams, list) or not beams:
        raise ValueError("beam_ids must be a nonempty list")
    for beam in beams:
        _number(beam, "beam_id", integer=True)
    if len(set(beams)) != len(beams):
        raise ValueError("beam_ids must be unique")
    overrides = config["dedispersion_overrides"]
    _keys(overrides, ("beams_per_batch", "num_active_batches"), "dedispersion_overrides")
    for key, value in overrides.items():
        _number(value, key, minimum=1, integer=True)
    if len(beams) % overrides["beams_per_batch"]:
        raise ValueError("beam count must be divisible by beams_per_batch")
    if len(beams) < overrides["beams_per_batch"] * overrides["num_active_batches"]:
        raise ValueError("beam count must cover the active beam batch capacity")
    GrouperConfig.from_mapping(config["grouper"])
    if config["grouper"]["execution"]["beam_batch_size"] != overrides["beams_per_batch"]:
        raise ValueError("grouper beam_batch_size must match the producer beams_per_batch")
    bursts = config["bursts"]
    if not isinstance(bursts, list):
        raise ValueError("bursts must be a list (use [] for noise only)")
    for burst in bursts:
        required = {"id", "beam_id", "dm", "toa_seconds", "reference_frequency_MHz",
                    "width_ms", "snr", "spectral_index", "subband_lo_MHz", "subband_hi_MHz"}
        if (not isinstance(burst, dict) or not required <= burst.keys()
                or burst.keys() - required - {"role"}):
            raise ValueError(f"burst requires {sorted(required)}; optional field: role")
        if not isinstance(burst["id"], str) or not burst["id"]:
            raise ValueError("burst id must be a nonempty string")
        if "role" in burst and (not isinstance(burst["role"], str) or not burst["role"]):
            raise ValueError("burst role must be a nonempty descriptive string")
        _number(burst["beam_id"], "burst.beam_id", integer=True)
        if burst["beam_id"] not in beams:
            raise ValueError("burst beam is absent from canonical metadata")
        _number(burst["dm"], "burst.dm")
        for key in ("width_ms", "snr", "reference_frequency_MHz"):
            _number(burst[key], "burst." + key, minimum=1.e-12)
        for key in ("toa_seconds", "spectral_index"):
            _number(burst[key], "burst." + key, minimum=-math.inf)
        lo = _number(burst["subband_lo_MHz"], "burst.subband_lo_MHz", minimum=1.e-12)
        hi = _number(burst["subband_hi_MHz"], "burst.subband_hi_MHz", minimum=1.e-12)
        if not CHORD_ZONE_EDGES[0] <= lo < hi <= CHORD_ZONE_EDGES[-1]:
            raise ValueError("burst frequency interval must satisfy 300 <= low < high <= 1500 MHz")
    if len({burst["id"] for burst in bursts}) != len(bursts):
        raise ValueError("burst IDs must be unique")


def make_pulse(burst, xmd):
    from .simpulse import SinglePulse, dispersion_delay
    return SinglePulse(
        dm=burst["dm"], sm=0.0, intrinsic_width=burst["width_ms"] * .001,
        spectral_index=burst["spectral_index"],
        undispersed_arrival_time_sec=burst["toa_seconds"] - dispersion_delay(
            burst["dm"], burst["reference_frequency_MHz"]),
        time_sample_ms=xmd.dt_ns_per_seq * xmd.seq_per_frb_time_sample / 1.e6,
        snr=burst["snr"], freq_edges_MHz=np.asarray(xmd.get_channel_freq_edges()),
        freq_variances=np.asarray(xmd.get_channel_variances()),
        subband_freq_lo_MHz=burst["subband_lo_MHz"],
        subband_freq_hi_MHz=burst["subband_hi_MHz"],
    )


@dataclass(frozen=True)
class ObservationPlan:
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
        raise RuntimeError(f"Observation stopped before completing {operation}")


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
                self.timed_out = f"Observation timed out after {self.timeout_s}s in {self.operation}"
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


def _drive_observation(plan, sender, frame_source, progress):
    """Pace minichunks, keeping native queues at most two minichunks ahead."""
    if int(sender.minichunks_per_chunk) != plan.minichunks_per_chunk:
        raise ValueError("sender minichunk geometry differs from observation")
    if int(sender.nworkers) < 1:
        raise ValueError("sender must have at least one worker")
    origin = time.monotonic()
    first_mc = plan.initial_chunk * plan.minichunks_per_chunk
    for offset in range(plan.nchunks + TRANSPORT_TAIL_CHUNKS):
        chunk = plan.initial_chunk + offset
        science = offset < plan.nchunks
        frames = frame_source(chunk) if science else None
        if science and (frames is None or int(frames.time_chunk_index) != chunk):
            raise ValueError(f"Frame source returned the wrong chunk for {chunk}")
        for within in range(plan.minichunks_per_chunk):
            imc = chunk * plan.minichunks_per_chunk + within
            relative = imc - first_mc
            if relative >= 2:
                for worker in range(sender.nworkers):
                    _checked(sender.wait_until_processed(worker, imc - 2),
                             f"worker {worker}, minichunk {imc - 2}")
            deadline = origin + (relative + 1) * 256 * plan.time_sample_s / plan.rate
            while True:
                remaining = deadline - time.monotonic()
                if remaining <= 0:
                    break
                time.sleep(min(remaining, 0.25))
            for worker in range(sender.nworkers):
                sent = (sender.enqueue_send_minichunk(worker, imc, frames) if science
                        else sender.enqueue_send_junk(worker, imc))
                _checked(sent, f"enqueue worker {worker}, minichunk {imc}")
        for worker in range(sender.nworkers):
            _checked(sender.synchronize(worker), f"drain worker {worker}, chunk {chunk}")
        if progress is not None:
            progress(dict(completed_chunks=min(offset + 1, plan.nchunks),
                          state="sender_complete" if offset == plan.nchunks + TRANSPORT_TAIL_CHUNKS - 1 else "running"))


def stream_observation(bundle, rpc_address, *, frame_source_factory, progress=None,
                       nworkers=1, startup_timeout_s=600.0, drain_timeout_s=120.0):
    """Stream generated frames and two transport-tail chunks without saving data.

    The source factory receives metadata and a bounded allocator. Sender
    completion does not imply that the dedisperser or grouper has finished.
    """
    for name, value in (("startup_timeout_s", startup_timeout_s),
                        ("drain_timeout_s", drain_timeout_s)):
        if not math.isfinite(value) or value <= 0:
            raise ValueError(f"{name} must be positive and finite")
    if isinstance(nworkers, bool) or not isinstance(nworkers, int) or nworkers < 1:
        raise ValueError("nworkers must be a positive integer")
    sender = allocator = client = watchdog = None

    try:
        from .core import (AssembledFrameAllocator, BumpAllocator,
                           FakeXEngine, SlabAllocator, XEngineMetadata)
        from .Hardware import Hardware
        from .rpc import FrbSearchClient
        from .rpc.grpc import frb_search_pb2
        from .utils import ThreadAffinity, check_mtu, extract_ip

        xmd = XEngineMetadata.from_yaml_string(bundle["metadata_yaml"])
        plan = ObservationPlan(int(bundle["initial_chunk"]), int(bundle["nchunks"]),
                          int(bundle["samples_per_chunk"]),
                          float(xmd.dt_ns_per_seq) * int(xmd.seq_per_frb_time_sample) * 1e-9,
                          tuple(int(b) for b in xmd.beam_ids), rate=1.0)
        if list(plan.beam_ids) != list(bundle["beam_ids"]):
            raise ValueError("Bundle beam ordering differs from canonical metadata")
        if not math.isclose(plan.time_sample_s * 1e3, float(bundle["time_sample_ms"]),
                            rel_tol=1e-12, abs_tol=1e-12):
            raise ValueError("Bundle cadence differs from canonical metadata")
        client = FrbSearchClient(rpc_address)
        version = frb_search_pb2.PROTOCOL_VERSION_CURRENT
        cfg = client.stub.GetConfig(frb_search_pb2.GetConfigRequest(protocol_version=version),
                                    timeout=startup_timeout_s, wait_for_ready=True)
        existing = client.stub.GetXEngineMetadata(
            frb_search_pb2.GetXEngineMetadataRequest(protocol_version=version),
            timeout=startup_timeout_s)
        if existing.yaml_string:
            raise ValueError("Observation requires a fresh server with no previous X-engine acquisition")
        addrs = list(cfg.data_ip_addrs)
        if not addrs or nworkers % len(addrs):
            raise ValueError("nworkers must be a multiple of the nonempty receiver address count")
        if nworkers > xmd.get_total_nfreq():
            raise ValueError("nworkers exceeds observation frequency channel count")
        if cfg.time_samples_per_chunk != plan.samples_per_chunk:
            raise ValueError("Server chunk length differs from observation")
        if cfg.beams_per_batch < 1 or len(plan.beam_ids) % cfg.beams_per_batch:
            raise ValueError("Observation beam count must be divisible by server beams_per_batch")
        hardware = Hardware()
        affinities = []
        for address in addrs:
            ip = extract_ip(address)
            affinity = hardware.vcpu_list_from_ip_addr(ip, is_dst_addr=True)
            check_mtu(hardware, "live observation", ip, cfg.min_data_mtu,
                      "min_data_mtu", is_dst_addr=True)
            affinities.append(affinity)
        if any(hardware.cpu_from_vcpu_list(a) != hardware.cpu_from_vcpu_list(affinities[0])
               for a in affinities):
            raise ValueError("Observation receiver routes must use NICs on one CPU")
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
            source = frame_source_factory(xmd, allocator)

            def stop_native():
                sender.stop()
                allocator.stop()

            watchdog = _BlockingDeadline(drain_timeout_s, stop_native)
            timed_sender = _TimedSender(sender, watchdog)

            _drive_observation(plan, timed_sender,
                lambda chunk: watchdog.call(f"generate chunk {chunk}", source, chunk), progress)
    finally:
        if watchdog is not None:
            watchdog.close()
        if sender is not None:
            sender.stop()
        if allocator is not None:
            allocator.stop()
        if client is not None:
            client.close()
