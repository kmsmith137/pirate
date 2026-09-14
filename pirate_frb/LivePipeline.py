"""Four-terminal, in-memory detection diagnostics driven by an existing recipe."""
from __future__ import annotations

import math
import os
from pathlib import Path
import signal
import threading

import yaml

from .utils import atomic_print


def load_live_recipe(filename):
    from .Observation import _read_yaml, validate_recipe
    from .core import XEngineMetadata
    from . import DedispersionConfig, DedispersionPlan

    source = Path(filename).resolve()
    recipe = _read_yaml(source)
    validate_recipe(recipe)
    obs = recipe["observation"]
    metadata = _read_yaml(source.parent / recipe["metadata"])
    metadata.update(freq_channels=[], beam_ids=obs["beam_ids"],
                    beam_positions_x=[0.] * len(obs["beam_ids"]),
                    beam_positions_y=[0.] * len(obs["beam_ids"]))
    xmd = XEngineMetadata.from_yaml_string(yaml.safe_dump(metadata))
    xmd.validate()
    cadence = xmd.dt_ns_per_seq * xmd.seq_per_frb_time_sample * 1e-6
    dd = _read_yaml(source.parent / recipe["dedispersion"])
    dd.update(recipe["dedispersion_overrides"])
    dd.update(beams_per_gpu=len(obs["beam_ids"]), zone_freq_edges=metadata["zone_freq_edges"],
              zone_nfreq=metadata["zone_nfreq"], time_sample_ms=cadence)
    config = DedispersionConfig.from_yaml_string(yaml.safe_dump(dd))
    config.validate()
    plan = DedispersionPlan(config, mega_ringbuf=False, gpu_kernels=False)
    ntime = int(plan.nt_in)
    count = math.ceil(obs["duration_seconds"] / (ntime * cadence * .001))
    bundle = dict(metadata=yaml.safe_load(xmd.to_yaml_string()), metadata_yaml=xmd.to_yaml_string(),
                  dedispersion_config_yaml=config.to_yaml_string(),
                  grouper_config=recipe["grouper"], grouper_config_yaml=yaml.safe_dump(recipe["grouper"]),
                  beam_ids=list(obs["beam_ids"]), initial_chunk=0, nchunks=count,
                  samples_per_chunk=ntime, time_sample_ms=cadence)
    return recipe, bundle


def addresses(base_port):
    if type(base_port) is not int or not 1024 <= base_port <= 65532:
        raise ValueError("base port must be between 1024 and 65532")
    return dict(zip(("data", "rpc", "grouper", "event_monitor"),
                    (f"127.0.0.1:{base_port + i}" for i in range(4))))


def format_detection(event, tick_seconds, reference_mhz):
    return (f"beam={event.beam_id} tree={event.tree_index} DM={event.dm:.4f} "
            f"S/N={event.snr:.3f} "
            f"sub-band={event.subband_freq_lo_MHz:.3f}-{event.subband_freq_hi_MHz:.3f} MHz "
            f"TOA={event.fpga_timestamp * tick_seconds:.6f} s "
            f"(at {reference_mhz:g} MHz, from observation start)")


def run_event_monitor(bundle, address):
    from concurrent import futures
    import grpc
    from .rpc.grpc import frb_sifter_pb2 as pb, frb_sifter_pb2_grpc as services

    class Printer(services.FrbSifterServicer):
        def __init__(self):
            self.lock = threading.Lock()
            self.configured = False
            self.count = 0

        def CheckConfiguration(self, request, context):
            if request.protocol_version != pb.PROTOCOL_VERSION_CURRENT:
                context.abort(grpc.StatusCode.FAILED_PRECONDITION, "protocol mismatch")
            md = yaml.safe_load(request.xengine_yaml)
            if md != bundle["metadata"]:
                context.abort(grpc.StatusCode.FAILED_PRECONDITION, "event_monitor YAML differs from dedisperser metadata")
            if yaml.safe_load(request.grouper_yaml) != bundle["grouper_config"]:
                context.abort(grpc.StatusCode.FAILED_PRECONDITION, "event_monitor YAML differs from grouper")
            with self.lock:
                self.configured = True
            return pb.ConfigReply(ok=True)

        def FrbEvents(self, request, context):
            with self.lock:
                if not self.configured or request.from_simulator:
                    context.abort(grpc.StatusCode.FAILED_PRECONDITION, "expected configured grouper detections")
                if request.beam_set_id != bundle["metadata"]["beamset"]:
                    context.abort(grpc.StatusCode.FAILED_PRECONDITION, "beam set mismatch")
                for event in request.events:
                    atomic_print("[event_monitor] " + format_detection(
                        event, bundle["metadata"]["dt_ns_per_seq"] * 1e-9,
                        bundle["metadata"]["zone_freq_edges"][0]))
                    self.count += 1
            return pb.FrbEventsReply(ok=True, message="printed; no data saved")

    printer = Printer()
    with futures.ThreadPoolExecutor(max_workers=2) as pool:
        server = grpc.server(pool, options=[("grpc.so_reuseport", 0)])
        services.add_FrbSifterServicer_to_server(printer, server)
        if not server.add_insecure_port(address):
            raise RuntimeError(f"cannot bind event_monitor at {address}")
        server.start()
        atomic_print(f"[event_monitor] Listening at {address}; prints only. Ctrl-C to stop.")
        try:
            server.wait_for_termination()
        finally:
            server.stop(0).wait()
            atomic_print(f"[event_monitor] Printed {printer.count} detection(s).")


def run_dedisperser(bundle, endpoints, gpu):
    from .DedispersionServer import DedispersionServer
    # Only assembly/processing retention is needed; no burst capture buffer.
    server = DedispersionServer(bundle,
                               endpoints["data"], endpoints["rpc"], endpoints["grouper"], gpu)
    try:
        atomic_print("[dedisperser] Waiting for grouper.")
        server.start(timeout=600)
        atomic_print("[dedisperser] Listening. After grouper reports complete, stop this terminal first (Ctrl-C).")
        while not server.server.poll_from_python(timeout_ms=250):
            pass
    finally:
        server.stop()


def run_observation(recipe, bundle, rpc):
    from .Observation import make_pulse, stream_observation
    import ksgpu

    ksgpu.seed_default_rng(recipe["observation"]["noise_seed"])

    def factory(xmd, allocator):
        pulses = {int(beam): [make_pulse(b, xmd) for b in recipe["bursts"] if b["beam_id"] == beam]
                  for beam in xmd.beam_ids}
        for beam_pulses in pulses.values():
            for pulse in beam_pulses:
                if pulse.it_start < 0 or pulse.it_end > bundle["nchunks"] * bundle["samples_per_chunk"]:
                    raise ValueError("a burst is clipped: extend duration or move its arrival time")

        def generate(chunk):
            frames = allocator.get_frame_set(chunk)
            for index, beam in enumerate(xmd.beam_ids):
                frames.get_frame(index).randomize_many(
                    normalize=True, gaussian=True, pulses=pulses[int(beam)],
                    dt_sp=chunk * bundle["samples_per_chunk"])
            frames.validate()
            return frames
        return generate

    for burst in recipe["bursts"]:
        atomic_print(f"[observation] Planned {burst['id']}: beam={burst['beam_id']} DM={burst['dm']} "
              f"S/N={burst['snr']} TOA={burst['toa_seconds']} s at {burst['reference_frequency_MHz']} MHz "
              f"band={burst['subband_lo_MHz']}-{burst['subband_hi_MHz']} MHz")

    def progress(report):
        done = report["completed_chunks"]
        if done % 8 == 0 or report["state"] == "sender_complete":
            atomic_print(f"[observation] Sent {min(done, bundle['nchunks'])}/{bundle['nchunks']} science chunks.")

    stream_observation(bundle, rpc, frame_source_factory=factory,
                       startup_timeout_s=600, progress=progress)
    atomic_print("[observation] Sending complete. Wait for the grouper's processing-complete message.")


def live_command(args):
    # Set before any live allocation/compilation. The documented -B invocation
    # also suppresses bytecode produced while importing the package itself.
    os.environ["CUPY_CACHE_IN_MEMORY"] = "1"
    os.environ["CUDA_CACHE_DISABLE"] = "1"
    endpoints = addresses(args.base_port)
    recipe, bundle = load_live_recipe(args.recipe)
    duration = bundle["nchunks"] * bundle["samples_per_chunk"] * bundle["time_sample_ms"] * .001
    atomic_print(f"[{args.role}] {len(bundle['beam_ids'])} beams, {duration:.3f} s; no output files.")

    def interrupt(signum, frame):
        raise KeyboardInterrupt
    previous = signal.signal(signal.SIGTERM, interrupt)
    try:
        if args.role == "dedisperser":
            run_dedisperser(bundle, endpoints, args.gpu)
        elif args.role == "event_monitor":
            run_event_monitor(bundle, endpoints["event_monitor"])
        elif args.role == "observation":
            run_observation(recipe, bundle, endpoints["rpc"])
        else:
            from .OnlineGrouper import run_online_grouper
            def progress(update):
                if update["phase"] == "complete":
                    atomic_print(f"[grouper] Processing complete: {update['events']} detection(s). "
                          "Stop dedisperser first, then event_monitor.")
                elif update["phase"] == "chunk" and update["completed_chunks"] % 8 == 0:
                    atomic_print(f"[grouper] Processed {update['completed_chunks']}/{update['expected_chunks']} chunks.")
            atomic_print("[grouper] Waiting for dedisperser and observation.")
            run_online_grouper(bundle, endpoints["grouper"],
                               sifter_addr=endpoints["event_monitor"], progress=progress)
    except KeyboardInterrupt:
        atomic_print(f"[{args.role}] Stopped.")
    finally:
        signal.signal(signal.SIGTERM, previous)


def add_live_parser(subparsers):
    parser = subparsers.add_parser("live", help="Four-terminal diagnostics without output files")
    roles = parser.add_subparsers(dest="role", required=True)
    descriptions = (
        ("grouper", "Find and group candidates from dedispersion outputs"),
        ("dedisperser", "Receive intensity data and run GPU dedispersion"),
        ("event_monitor", "Print detected events without saving data"),
        ("observation", "Generate and stream the YAML-defined observation"),
    )
    for role, description in descriptions:
        command = roles.add_parser(role, help=description, description=description)
        command.add_argument("recipe", help="YAML observation and burst recipe")
        command.add_argument("--base-port", type=int, default=19700,
                             help="First of four local ports (same in every terminal; default: 19700)")
        if role == "dedisperser":
            command.add_argument("--gpu", type=int, default=0, help="Logical CUDA device (default: 0)")
        command.set_defaults(func=live_command)
