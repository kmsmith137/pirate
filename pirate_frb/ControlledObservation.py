"""A finite CHORD observation, saved once for offline and network replay.

``bundle.json`` and ``load_experiment_bundle`` contain structural acquisition
information only. Injection truth is deliberately stored separately and is never
loaded by a runtime input adapter. Preparing a bundle does not allocate a GPU.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
from pathlib import Path
import subprocess
import sys

import numpy as np
import yaml

from .OfflineGrouperConfig import OfflineGrouperConfig, _StrictSafeLoader

SCHEMA_VERSION = 1
MANIFEST_NAME = "bundle.json"
CHORD_ZONE_EDGES = [300, 350, 450, 600, 800, 1500]
CHORD_ZONE_NFREQ = [8192, 8192, 6144, 2048, 3584]


def _sha256(path):
    digest = hashlib.sha256()
    with open(path, "rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _json_write(path, value):
    temporary = Path(str(path) + ".tmp")
    with temporary.open("x") as stream:
        json.dump(value, stream, indent=2, sort_keys=True, allow_nan=False)
        stream.write("\n")
    os.replace(temporary, path)


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


def _within(root, relative):
    if not isinstance(relative, str) or Path(relative).is_absolute():
        raise ValueError("bundle paths must be relative strings")
    path = (root / relative).resolve()
    if not path.is_relative_to(root):
        raise ValueError(f"bundle path escapes its directory: {relative!r}")
    return path


def _read_json(path):
    with open(path) as stream:
        return json.load(stream)


def _read_yaml(path):
    with open(path) as stream:
        return yaml.load(stream, Loader=_StrictSafeLoader)


def _yaml_write(path, value):
    with open(path, "x") as stream:
        yaml.safe_dump(value, stream, sort_keys=False)


def _check_experiment(config):
    _keys(config, ("version", "metadata", "dedispersion", "observation",
                   "dedispersion_overrides", "bursts", "grouper", "capture"), "experiment")
    if type(config["version"]) is not int or config["version"] != SCHEMA_VERSION:
        raise ValueError("unsupported experiment version")
    obs = config["observation"]
    _keys(obs, ("duration_seconds", "initial_chunk", "beam_ids", "noise_seed",
                "latency_budget_seconds"), "observation")
    _number(obs["duration_seconds"], "duration_seconds", minimum=1)
    if _number(obs["initial_chunk"], "initial_chunk", integer=True) != 0:
        raise ValueError("controlled observations currently begin at chunk zero")
    _number(obs["noise_seed"], "noise_seed", integer=True)
    if obs["noise_seed"] >= 2**32:
        raise ValueError("noise_seed must fit uint32")
    _number(obs["latency_budget_seconds"], "latency_budget_seconds")
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
    if len(beams) < 2 * overrides["beams_per_batch"] * overrides["num_active_batches"]:
        raise ValueError("online server requires at least twice the active beam batch capacity")
    OfflineGrouperConfig.from_mapping(config["grouper"])
    if config["grouper"]["execution"]["beam_batch_size"] != overrides["beams_per_batch"]:
        raise ValueError("grouper beam_batch_size must match the producer beams_per_batch")
    # A deadline-based partial grouping result is not an equality reference.
    if config["grouper"]["execution"]["timeout_ms"] != 0:
        raise ValueError("disable grouping timeouts for the controlled equality experiment")
    capture = config["capture"]
    _keys(capture, ("version", "snr_threshold", "buffer_seconds", "pre_padding_seconds",
                    "post_padding_seconds", "association_dm_tolerance",
                    "association_toa_tolerance_seconds", "write_timeout_seconds",
                    "classifier_mode"), "capture")
    if type(capture["version"]) is not int or capture["version"] != 1 or capture["classifier_mode"] != "bypass":
        raise ValueError("capture must use version 1 and explicit classifier_mode: bypass")
    for key in set(capture) - {"version", "classifier_mode"}:
        _number(capture[key], "capture." + key, minimum=0)
    if capture["buffer_seconds"] <= 0 or capture["write_timeout_seconds"] <= 0:
        raise ValueError("capture buffer and write timeout must be positive")
    if capture["snr_threshold"] < config["grouper"]["peakfinding"]["snr_threshold"]:
        raise ValueError("capture threshold cannot be below the extraction threshold")
    bursts = config["bursts"]
    if not isinstance(bursts, list) or len(bursts) != 2:
        raise ValueError("this acceptance experiment requires exactly two bursts")
    roles = []
    for burst in bursts:
        _keys(burst, ("id", "role", "beam_id", "dm", "toa_seconds", "reference_frequency_MHz",
                      "width_ms", "snr", "spectral_index", "subband_lo_MHz", "subband_hi_MHz"), "burst")
        if not isinstance(burst["id"], str) or not burst["id"]:
            raise ValueError("burst id must be a nonempty string")
        roles.append(burst["role"])
        if burst["beam_id"] not in beams:
            raise ValueError("burst beam is absent from canonical metadata")
        for key in ("dm", "toa_seconds", "width_ms", "snr"):
            _number(burst[key], "burst." + key, minimum=1.e-12)
        _number(burst["spectral_index"], "spectral_index", minimum=-100)
        if (burst["reference_frequency_MHz"] != 300 or burst["subband_lo_MHz"] != 300
                or burst["subband_hi_MHz"] != 1500):
            raise ValueError("first CHORD acceptance experiment uses broadband 300–1500 MHz bursts referenced to 300 MHz")
    if sorted(roles) != ["early_trigger", "full_band"]:
        raise ValueError("require one full_band and one early_trigger burst")
    if len({burst["id"] for burst in bursts}) != len(bursts):
        raise ValueError("burst IDs must be unique")


def _pulse(burst, xmd):
    from .simpulse import SinglePulse, dispersion_delay
    return SinglePulse(
        dm=burst["dm"], sm=0.0, intrinsic_width=burst["width_ms"] * .001,
        spectral_index=burst["spectral_index"],
        undispersed_arrival_time_sec=burst["toa_seconds"] - dispersion_delay(burst["dm"], 300.0),
        time_sample_ms=xmd.dt_ns_per_seq * xmd.seq_per_frb_time_sample / 1.e6,
        snr=burst["snr"], freq_edges_MHz=np.asarray(xmd.get_channel_freq_edges()),
        freq_variances=np.asarray(xmd.get_channel_variances()),
        subband_freq_lo_MHz=burst["subband_lo_MHz"],
        subband_freq_hi_MHz=burst["subband_hi_MHz"],
    )


def prepare_controlled_observation(config_file, bundle_dir):
    """Validate a CHORD experiment and exclusively create its prepared bundle.

    The CPU plan computes conservative steady-state bounds for all output trees.
    Target-tree arrival times must follow those bounds plus a chunk guard. No
    Dcore values are inferred or recorded: those belong to the actual producer.
    """
    from . import DedispersionConfig, DedispersionPlan
    from .core import XEngineMetadata
    from .simpulse import dispersion_delay

    source = Path(config_file).resolve()
    config = _read_yaml(source)
    _check_experiment(config)
    obs = config["observation"]
    metadata = _read_yaml(source.parent / config["metadata"])
    if metadata["zone_freq_edges"] != CHORD_ZONE_EDGES or metadata["zone_nfreq"] != CHORD_ZONE_NFREQ:
        raise ValueError("metadata must preserve the CHORD band and all 28160 channels")
    metadata["freq_channels"] = []
    metadata["beam_ids"] = obs["beam_ids"]
    # These are artificial co-pointed toy beams; preserve the telescope geometry.
    metadata["beam_positions_x"] = [0.] * len(obs["beam_ids"])
    metadata["beam_positions_y"] = [0.] * len(obs["beam_ids"])
    xmd = XEngineMetadata.from_yaml_string(yaml.safe_dump(metadata))
    xmd.validate()
    cadence = xmd.dt_ns_per_seq * xmd.seq_per_frb_time_sample / 1.e6
    dd = _read_yaml(source.parent / config["dedispersion"])
    dd.update(config["dedispersion_overrides"])
    dd.update(beams_per_gpu=len(obs["beam_ids"]), zone_freq_edges=CHORD_ZONE_EDGES,
              zone_nfreq=CHORD_ZONE_NFREQ, time_sample_ms=cadence)
    producer_config = DedispersionConfig.from_yaml_string(yaml.safe_dump(dd))
    producer_config.validate()
    plan = DedispersionPlan(producer_config, mega_ringbuf=False, gpu_kernels=False)
    ntime = int(plan.nt_in)
    chunk_seconds = ntime * cadence * .001
    nchunks = math.ceil(obs["duration_seconds"] / chunk_seconds)
    duration = nchunks * chunk_seconds
    capture = config["capture"]
    ring_chunks = max(7, math.ceil(capture["buffer_seconds"] / chunk_seconds))
    nominal_retention = ring_chunks * chunk_seconds
    # Completion of the current chunk, two receiver assembly lookahead chunks,
    # and the shared grouper's one-chunk lookahead precede additional latency.
    planned_latency = 4 * chunk_seconds + obs["latency_budget_seconds"]
    tree_reports = []
    for itree, tree in enumerate(plan.trees):
        steady = np.asarray(plan.compute_steady_state_it0(itree))
        tree_reports.append(dict(tree_index=itree, primary_tree_index=int(tree.primary_tree_index),
            early_trigger_level=int(tree.early_trigger_level), dm_min=float(tree.dm_min),
            dm_max=float(tree.dm_max), trigger_frequency_MHz=float(tree.trigger_frequency),
            conservative_ready_seconds=float(steady.max()) * chunk_seconds / tree.nt_out))
    truth = []
    for burst in config["bursts"]:
        pulse = _pulse(burst, xmd)
        if pulse.it_start < 0 or pulse.it_end > nchunks * ntime:
            raise ValueError(f"burst {burst['id']} is clipped; extend observation or move its arrival")
        trees = [tree for tree in tree_reports if tree["dm_min"] <= burst["dm"] < tree["dm_max"]]
        if not trees or not any(tree["early_trigger_level"] == 0 for tree in trees):
            raise ValueError(f"burst {burst['id']} is outside the full-band search")
        arrivals = []
        for tree in trees:
            arrival = burst["toa_seconds"] - (dispersion_delay(burst["dm"], 300.)
                       - dispersion_delay(burst["dm"], tree["trigger_frequency_MHz"]))
            if arrival < tree["conservative_ready_seconds"] + chunk_seconds:
                raise ValueError(f"burst {burst['id']} arrives too early for tree {tree['tree_index']}; "
                                 "move the burst later to provide authoritative startup")
            arrivals.append(dict(tree_index=tree["tree_index"],
                                 early_trigger_level=tree["early_trigger_level"],
                                 trigger_frequency_MHz=tree["trigger_frequency_MHz"],
                                 arrival_seconds=float(arrival)))
        sweep = dispersion_delay(burst["dm"], 300.) - dispersion_delay(burst["dm"], 1500.)
        earliest_raw = int(pulse.it_start) * cadence * .001
        earliest_trigger = min(item["arrival_seconds"] for item in arrivals)
        budgeted_age = earliest_trigger + planned_latency - earliest_raw + capture["pre_padding_seconds"]
        if burst["role"] == "full_band":
            if sweep + planned_latency + capture["pre_padding_seconds"] >= nominal_retention:
                raise ValueError("low-DM burst does not fit the buffer with the planned latency margin")
        else:
            if not any(item["early_trigger_level"] > 0 for item in arrivals):
                raise ValueError("high-DM burst does not exercise an early-trigger tree")
            if sweep <= nominal_retention + chunk_seconds:
                raise ValueError("high-DM sweep must exceed retention by at least one chunk")
            if budgeted_age >= nominal_retention:
                raise ValueError("early trigger has insufficient planned capture margin")
            future_samples = math.ceil((burst["toa_seconds"] + capture["post_padding_seconds"]
                                       - earliest_trigger) / (cadence * .001))
            if future_samples > producer_config.future_write_max_samples:
                raise ValueError("high-DM capture exceeds future_write_max_samples")
        truth.append(dict(**burst, sample_start=int(pulse.it_start), sample_end=int(pulse.it_end),
                          dispersion_sweep_seconds=float(sweep), expected_tree_arrivals=arrivals,
                          planned_capture_margin_seconds=nominal_retention - budgeted_age))
    ordered = sorted(truth, key=lambda burst: burst["sample_start"])
    if any(a["sample_end"] > b["sample_start"] for a, b in zip(ordered, ordered[1:])):
        raise ValueError("first acceptance experiment requires non-overlapping dispersed burst intervals")
    root = Path(bundle_dir).resolve()
    root.mkdir(parents=True, exist_ok=False)
    manifest = dict(version=SCHEMA_VERSION, state="preparing", metadata_file="metadata.yml",
                    dedispersion_config_file="dedispersion.yml", grouper_config_file="grouper.yml",
                    capture_config_file="capture.yml", initial_chunk=obs["initial_chunk"], nchunks=nchunks,
                    samples_per_chunk=ntime, time_sample_ms=cadence, beam_ids=obs["beam_ids"],
                    duration_seconds=duration, frame_entries=[])
    try:
        _json_write(root / MANIFEST_NAME, manifest)
        _yaml_write(root / "metadata.yml", metadata)
        _yaml_write(root / "dedispersion.yml", dd)
        _yaml_write(root / "grouper.yml", config["grouper"])
        _yaml_write(root / "capture.yml", capture)
        _yaml_write(root / "experiment.yml", config)
        _json_write(root / "injections.json", dict(version=1, bursts=truth,
                    width_convention="intrinsic Gaussian sigma in milliseconds",
                    reference_frequency_MHz=300.))
        frame_bytes = sum(CHORD_ZONE_NFREQ) * ntime // 2 + sum(CHORD_ZONE_NFREQ) * (ntime // 256) * 4
        _json_write(root / "plan_check.json", dict(version=1, trees=tree_reports,
                    requested_duration_seconds=obs["duration_seconds"], duration_seconds=duration,
                    chunk_seconds=chunk_seconds, ringbuf_nchunks=ring_chunks,
                    nominal_retention_seconds=nominal_retention,
                    receiver_assembly_lookahead_chunks=2, chunk_completion_delay_chunks=1,
                    grouping_lookahead_chunks=1,
                    planned_non_chunk_latency_seconds=obs["latency_budget_seconds"],
                    planned_total_trigger_latency_seconds=planned_latency,
                    estimated_raw_frame_bytes=frame_bytes,
                    estimated_raw_archive_bytes=frame_bytes * nchunks * len(obs["beam_ids"]),
                    estimated_raw_ring_bytes=frame_bytes * ring_chunks * len(obs["beam_ids"]),
                    note="Byte estimates exclude ASDF headers and allocator overhead; retention and latency must be measured online. Startup bounds do not mask other trees."))
        (root / "acq").mkdir()
        manifest["artifact_hashes"] = {name: _sha256(root / name) for name in
            ("metadata.yml", "dedispersion.yml", "grouper.yml", "capture.yml", "experiment.yml", "injections.json", "plan_check.json")}
        manifest["state"] = "prepared"
        _json_write(root / MANIFEST_NAME, manifest)
    except BaseException as exc:
        manifest.update(state="failed", error=f"{type(exc).__name__}: {exc}")
        _json_write(root / MANIFEST_NAME, manifest)
        raise
    return load_experiment_bundle(root, require_complete=False)


def load_experiment_bundle(bundle_dir, *, verify_hashes=False, require_complete=True):
    """Load validated structural metadata; never read the injection truth.

    All filenames are relative to the bundle. Resolved ``*_path`` fields are
    returned for convenience. Complete bundles must cover every (chunk, beam)
    exactly once in producer order. ``verify_hashes`` also reads frame payloads.
    """
    root = Path(bundle_dir).resolve()
    manifest_path = root / MANIFEST_NAME
    data = _read_json(manifest_path)
    if type(data.get("version")) is not int or data["version"] != SCHEMA_VERSION:
        raise ValueError("unsupported controlled-observation bundle version")
    if require_complete and data.get("state") != "complete":
        raise ValueError(f"observation bundle is not complete: {data.get('state')}")
    if data.get("state") not in ("prepared", "generating", "complete"):
        raise ValueError(f"observation bundle cannot be used in state {data.get('state')}")
    for key in ("initial_chunk", "nchunks", "samples_per_chunk"):
        _number(data[key], key, minimum=0 if key == "initial_chunk" else 1, integer=True)
    _number(data["time_sample_ms"], "time_sample_ms", minimum=1.e-12)
    beams = data["beam_ids"]
    if not isinstance(beams, list) or not beams or len(set(beams)) != len(beams):
        raise ValueError("invalid canonical beam IDs")
    for beam in beams:
        _number(beam, "beam_id", integer=True)
    artifacts = data["artifact_hashes"]
    # Runtime paths are checked independently; truth is neither opened nor used.
    for stem in ("metadata", "dedispersion_config", "grouper_config", "capture_config"):
        relative = data[stem + "_file"]
        path = _within(root, relative)
        if _sha256(path) != artifacts.get(relative):
            raise ValueError(f"bundle configuration hash mismatch: {relative}")
        data[stem + "_path"] = str(path)
    duration = data["nchunks"] * data["samples_per_chunk"] * data["time_sample_ms"] * .001
    if not math.isclose(data["duration_seconds"], duration, rel_tol=1.e-14):
        raise ValueError("manifest duration disagrees with its chunk coverage")
    dd = _read_yaml(data["dedispersion_config_path"])
    if dd["time_samples_per_chunk"] != data["samples_per_chunk"] or dd["beams_per_gpu"] != len(beams):
        raise ValueError("manifest dimensions disagree with dedispersion configuration")
    metadata = _read_yaml(data["metadata_path"])
    if metadata["beam_ids"] != beams or metadata.get("freq_channels") != []:
        raise ValueError("canonical metadata must contain every beam and frequency channel")
    if metadata["zone_nfreq"] != CHORD_ZONE_NFREQ or metadata["zone_freq_edges"] != CHORD_ZONE_EDGES:
        raise ValueError("canonical metadata no longer describes full CHORD channelization")
    implied_cadence = metadata["dt_ns_per_seq"] * metadata["seq_per_frb_time_sample"] / 1.e6
    if implied_cadence != data["time_sample_ms"]:
        raise ValueError("manifest cadence disagrees with metadata")
    entries = data["frame_entries"]
    if data["state"] == "complete":
        expected = [(chunk, beam) for chunk in range(data["initial_chunk"], data["initial_chunk"] + data["nchunks"]) for beam in beams]
        actual = [(entry["time_chunk_index"], entry["beam_id"]) for entry in entries]
        if actual != expected:
            raise ValueError("raw frame coverage/order differs from the declared observation")
        paths = set()
        for entry in entries:
            _number(entry["time_chunk_index"], "frame chunk", integer=True)
            _number(entry["beam_id"], "frame beam", integer=True)
            _number(entry["size_bytes"], "frame size", minimum=1, integer=True)
            digest = entry["sha256"]
            if not isinstance(digest, str) or len(digest) != 64 or any(c not in "0123456789abcdef" for c in digest):
                raise ValueError("invalid raw frame SHA256")
            path = _within(root, entry["path"])
            if path in paths:
                raise ValueError("a raw frame path occurs more than once")
            paths.add(path)
            if path.stat().st_size != entry["size_bytes"]:
                raise ValueError(f"raw frame size mismatch: {entry['path']}")
            if verify_hashes and _sha256(path) != entry["sha256"]:
                raise ValueError(f"raw frame hash mismatch: {entry['path']}")
    data.update(bundle_dir=str(root), manifest_path=str(manifest_path),
                manifest_sha256=_sha256(manifest_path), metadata=metadata)
    return data


def generate_controlled_observation(bundle_dir):
    """Generate the raw archive in a fresh process, exclusively and without resume.

    A fresh process ensures that seed_default_rng runs before this thread's first
    SIMD-noise RNG initialization. Hashes of saved frames, rather than a promise
    of cross-version RNG stability, define the identical replay input.
    """
    root = Path(bundle_dir).resolve()
    manifest = _read_json(root / MANIFEST_NAME)
    load_experiment_bundle(root, require_complete=False)
    if manifest["state"] != "prepared":
        raise ValueError("generation requires a prepared, never-generated bundle")
    with (root / "generation.lock").open("x"):
        pass
    manifest["state"] = "generating"
    _json_write(root / MANIFEST_NAME, manifest)
    try:
        subprocess.run([sys.executable, "-P", "-m", "pirate_frb.ControlledObservation",
                        "generate-worker", str(root)], check=True)
    except BaseException as exc:
        manifest = _read_json(root / MANIFEST_NAME)
        manifest.update(state="failed", error=f"{type(exc).__name__}: {exc}")
        _json_write(root / MANIFEST_NAME, manifest)
        raise
    return load_experiment_bundle(root, verify_hashes=True)


def _generate_worker(bundle_dir):
    import importlib.metadata
    import platform
    import ksgpu
    from .core import AssembledFrame, XEngineMetadata
    source_sha256 = _sha256(Path(__file__))
    root = Path(bundle_dir).resolve()
    manifest = _read_json(root / MANIFEST_NAME)
    data = load_experiment_bundle(root, require_complete=False)
    if manifest["state"] != "generating" or not (root / "generation.lock").exists():
        raise ValueError("worker requires the parent's exclusive generation claim")
    for name, digest in manifest["artifact_hashes"].items():
        if _sha256(_within(root, name)) != digest:
            raise ValueError(f"generation input changed: {name}")
    config = _read_yaml(root / "experiment.yml")
    ksgpu.seed_default_rng(config["observation"]["noise_seed"])
    xmd = XEngineMetadata.from_yaml_file(data["metadata_path"])
    pulses = {beam: [_pulse(burst, xmd) for burst in config["bursts"] if burst["beam_id"] == beam]
              for beam in data["beam_ids"]}
    entries = []
    for chunk in range(data["initial_chunk"], data["initial_chunk"] + data["nchunks"]):
        for beam in data["beam_ids"]:
            frame = AssembledFrame.make_uninitialized(xmd, ntime=data["samples_per_chunk"],
                                                    beam_id=beam, time_chunk_index=chunk)
            frame.randomize_many(normalize=True, gaussian=True, pulses=pulses[beam],
                                 dt_sp=chunk * data["samples_per_chunk"])
            relative = f"acq/frame_b{beam}_t{chunk}.asdf"
            final = root / relative
            temporary = Path(str(final) + ".tmp")
            if final.exists() or temporary.exists():
                raise FileExistsError(final)
            frame.write_asdf(str(temporary))
            os.link(temporary, final)  # publication fails rather than replacing a file
            temporary.unlink()
            entries.append(dict(beam_id=beam, time_chunk_index=chunk, path=relative,
                                size_bytes=final.stat().st_size, sha256=_sha256(final)))
            del frame
        print(f"Generated raw chunk {chunk - data['initial_chunk'] + 1}/{data['nchunks']}", flush=True)
    manifest.update(state="complete", frame_entries=entries, generation=dict(
        noise_seed=config["observation"]["noise_seed"],
        software_versions={"python": platform.python_version(),
                           "pirate_frb": importlib.metadata.version("pirate_frb"),
                           "ksgpu": importlib.metadata.version("ksgpu"),
                           "numpy": np.__version__},
        generator_source_sha256=source_sha256,
        noise_method="fresh-process ksgpu thread RNG plus AVX2 quantized Gaussian noise",
        reproducibility="Exact saved frame hashes define replay input; RNG stability across software versions is not assumed."))
    _json_write(root / MANIFEST_NAME, manifest)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    prepare = commands.add_parser("prepare")
    prepare.add_argument("config_file")
    prepare.add_argument("bundle_dir")
    generate = commands.add_parser("generate")
    generate.add_argument("bundle_dir")
    worker = commands.add_parser("generate-worker", help=argparse.SUPPRESS)
    worker.add_argument("bundle_dir")
    verify = commands.add_parser("verify")
    verify.add_argument("bundle_dir")
    args = parser.parse_args(argv)
    if args.command == "prepare":
        result = prepare_controlled_observation(args.config_file, args.bundle_dir)
        print(f"Prepared {result['nchunks']} chunks x {len(result['beam_ids'])} beams; "
              f"{result['duration_seconds']:.6f} seconds; no raw frames generated")
    elif args.command == "generate":
        result = generate_controlled_observation(args.bundle_dir)
        print(f"Generated and verified {len(result['frame_entries'])} raw frames")
    elif args.command == "generate-worker":
        _generate_worker(args.bundle_dir)
    else:
        result = load_experiment_bundle(args.bundle_dir, verify_hashes=True)
        print(f"Verified {len(result['frame_entries'])} raw frames")


if __name__ == "__main__":
    main()
