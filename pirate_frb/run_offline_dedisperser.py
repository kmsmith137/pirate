"""
Toy offline dedispersion driver + rudimentary peak-finding over an acquisition directory.

Reads a directory of acquired AssembledFrame ".asdf" files (one file per
(beam, time chunk) -- as written by 'pirate_frb rpc_start_stream'). Beams and their
time chunks are enumerated with pirate_frb.Acquisition. Each beam is
processed independently: a fresh single-beam OfflineDedisperser (nbeams == 1) is
built from scratch, and for each of that beam's time chunks the driver:

  1. uploads the quantized (int4) data to the GPU,
  2. dequantizes it to float16,
  3. tree-dedisperses it on the GPU,
  4. does rudimentary peak finding (per-chunk max SNR over the DM-vs-time plane of
     every dedispersion tree),

and prints the peak SNR of each chunk (one beam+chunk pair per line). With
``--save``, it also writes each S/N map beside its input frame using the name
``frame_b(BEAM)_t(CHUNK)_snrmap.asdf``.

Run via: python -m pirate_frb run_offline_dedisperser ACQDIR CONFIG
         [--max-chunks N] [--save]
"""
import os
from numbers import Real

import numpy as np

from .utils import atomic_print


DEFAULT_ACQDIR = "/mnt/cs00/data/kmsmith/2026-07-05/streams/kmsmith_26_07_05_202822"
DEFAULT_CONFIG = "configs/dedispersion/chord_sb2.yml"


def _validate_snr_asdf_tree(asdf_tree):
    """Validate a version-2 offline-dedisperser ASDF tree.

    Returns the incomplete DedispersionPlan reconstructed from the saved YAML.
    This validates the decoder metadata and supplies authoritative tree shapes.
    """
    import numpy as np
    from .pirate_pybind11 import DedispersionPlan

    if asdf_tree.get("format") != "pirate_frb.offline_dedisperser_snr_maps":
        raise ValueError("offline dedisperser ASDF: unexpected format")
    if asdf_tree.get("format_version") != 2:
        raise ValueError("offline dedisperser ASDF: expected format_version=2")

    config_yaml = asdf_tree.get("config_yaml")
    plan_yaml = asdf_tree.get("plan_yaml")
    if not isinstance(config_yaml, str) or not config_yaml:
        raise ValueError("offline dedisperser ASDF: config_yaml must be a nonempty string")
    if not isinstance(plan_yaml, str) or not plan_yaml:
        raise ValueError("offline dedisperser ASDF: plan_yaml must be a nonempty string")

    plan = DedispersionPlan.make_incomplete_plan_from_yaml(config_yaml, plan_yaml)

    # The serialized config/plan pair is authoritative for all geometry. The
    # optional top-level scalar is redundant convenience metadata: validate it
    # on writer and reader paths, but never let it override the plan.
    authoritative_time_sample_ms = float(plan.config.time_sample_ms)
    if (not np.isfinite(authoritative_time_sample_ms)
            or authoritative_time_sample_ms <= 0.0):
        raise ValueError(
            "offline dedisperser ASDF: authoritative plan time_sample_ms "
            "must be finite and positive"
        )
    saved_time_sample_ms = asdf_tree.get("time_sample_ms")
    if saved_time_sample_ms is not None:
        if (isinstance(saved_time_sample_ms, (bool, np.bool_))
                or not isinstance(saved_time_sample_ms, Real)):
            raise ValueError(
                "offline dedisperser ASDF: time_sample_ms must be numeric"
            )
        saved_time_sample_ms = float(saved_time_sample_ms)
        if not np.isfinite(saved_time_sample_ms) or saved_time_sample_ms <= 0.0:
            raise ValueError(
                "offline dedisperser ASDF: time_sample_ms must be finite and positive"
            )
        if saved_time_sample_ms != authoritative_time_sample_ms:
            raise ValueError(
                "offline dedisperser ASDF: saved time_sample_ms disagrees with "
                "the authoritative plan"
            )
    trees = asdf_tree.get("trees")
    if not isinstance(trees, (list, tuple)) or len(trees) != plan.ntrees:
        raise ValueError(
            f"offline dedisperser ASDF: expected {plan.ntrees} trees, "
            f"got {None if trees is None else len(trees)}"
        )

    for itree, (saved_tree, plan_tree) in enumerate(zip(trees, plan.trees)):
        if "snr" not in saved_tree or "argmax" not in saved_tree:
            raise ValueError(
                f"offline dedisperser ASDF: tree {itree} must contain both "
                "snr and argmax"
            )

        expected_shape = (int(plan_tree.ndm_out), int(plan_tree.nt_out))
        snr = np.asarray(saved_tree["snr"])
        argmax = np.asarray(saved_tree["argmax"])
        if snr.shape != expected_shape:
            raise ValueError(
                f"offline dedisperser ASDF: tree {itree} snr shape {snr.shape} "
                f"!= expected {expected_shape}"
            )
        if argmax.shape != expected_shape:
            raise ValueError(
                f"offline dedisperser ASDF: tree {itree} argmax shape "
                f"{argmax.shape} != expected {expected_shape}"
            )
        if not np.issubdtype(snr.dtype, np.floating):
            raise ValueError(
                f"offline dedisperser ASDF: tree {itree} snr dtype "
                f"{snr.dtype} is not floating"
            )
        if argmax.dtype != np.dtype(np.uint32):
            raise ValueError(
                f"offline dedisperser ASDF: tree {itree} argmax dtype "
                f"{argmax.dtype} != uint32"
            )

    return plan


def _write_snr_asdf(filename, input_filename, frame, od, snr_maps, argmax_maps,
                    *, producer_start_time_chunk_index):
    """Write and validate host-resident per-tree S/N and argmax maps.

    The three integer ``xengine_timing`` fields are sufficient to convert the
    decoder's canonical full-resolution sample coordinate to FPGA sequence and
    UNIX time without reopening ``input_filename``. Keeping this provenance in
    the derived product matters because acquisitions and S/N maps are often
    archived independently. The complete producer YAML is retained as well so
    later catalog writers can audit the timing projection against its source.
    """
    import asdf

    if len(snr_maps) != od.ntrees or len(argmax_maps) != od.ntrees:
        raise ValueError(
            f"offline dedisperser ASDF: expected {od.ntrees} S/N and argmax "
            f"maps, got {len(snr_maps)} and {len(argmax_maps)}"
        )

    trees = []
    for itree, (tree, snr_map, argmax_map) in enumerate(
            zip(od.trees, snr_maps, argmax_maps)):
        trees.append({
            "tree_index": itree,
            "dm_min": float(tree.dm_min),
            "dm_max": float(tree.dm_max),
            "trigger_frequency": float(tree.trigger_frequency),
            "ndm": int(tree.ndm_out),
            "ntime": int(tree.nt_out),
            "snr": snr_map,
            # Same indices: snr[idm,itime] <-> argmax[idm,itime].
            "argmax": argmax_map,
        })

    asdf_tree = {
        "format": "pirate_frb.offline_dedisperser_snr_maps",
        "format_version": 2,
        "config_yaml": od.config.to_yaml_string(),
        "plan_yaml": od.plan.to_yaml_string(),
        "source": {
            "filename": os.path.abspath(input_filename),
            "beam_id": int(frame.beam_id),
            "time_chunk_index": int(frame.time_chunk_index),
        },
        # Authoritative for this saved-map producer: OfflineDedisperser is created
        # immediately before the first frame in this beam is processed. Older v2
        # files lack this optional field and remain "startup unknown" unless the
        # offline grouper is given an explicit policy/value.
        "producer_start_time_chunk_index": int(
            producer_start_time_chunk_index),

        "time_sample_ms": float(od.time_sample_ms),
        "nt_in": int(od.nt_in),
        "nfreq": int(od.nfreq),
        "trees": trees,
    }
    # Some synthetic/backward-compatibility producers expose only frame
    # beam/chunk provenance.  Omission is explicit and later produces invalid
    # FPGA timestamps; we never invent clock constants for those products.
    metadata = getattr(frame, "metadata", None)
    if metadata is not None:
        asdf_tree["xengine_timing"] = {
            "unix_ns_at_seq_0": int(metadata.unix_ns_at_seq_0),
            "dt_ns_per_seq": int(metadata.dt_ns_per_seq),
            "seq_per_frb_time_sample": int(
                metadata.seq_per_frb_time_sample),
        }
        to_yaml_string = getattr(metadata, "to_yaml_string", None)
        if callable(to_yaml_string):
            asdf_tree["xengine_metadata_yaml"] = to_yaml_string()
    _validate_snr_asdf_tree(asdf_tree)
    asdf.AsdfFile(asdf_tree).write_to(filename)


def _process_beam(beam_id, files, config, save=False):
    """Dedisperse all of one beam's chunks and print each chunk's peak SNR.

    Builds a fresh OfflineDedisperser for this beam (which lazily initializes its
    GpuDedisperser + analytic weights from the first frame's metadata), then
    processes the frames in time order (one line per beam+chunk). Rudimentary peak
    finding: the printed SNR is the max out_max over the DM-vs-time plane of every
    tree, done on the GPU.
    """
    import cupy as cp
    from .core import AssembledFrame
    from .OfflineDedisperser import OfflineDedisperser

    # FIXME: there is currently no way to reset an OfflineDedisperser, so we
    # construct a new one from scratch for every beam.
    od = OfflineDedisperser(config)
    atomic_print(f"\nBeam {beam_id}: {len(files)} chunks")

    producer_start_time_chunk_index = None
    for ichunk, path in enumerate(files):
        frame = AssembledFrame.from_asdf(path)
        assert frame.beam_id == beam_id, (frame.beam_id, beam_id)

        if producer_start_time_chunk_index is None:
            producer_start_time_chunk_index = int(frame.time_chunk_index)
        # The first dedisperse() lazily builds the pipeline from the frame's
        # metadata; the outputs are valid only inside the 'with' block, so the peak
        # is host-copied there. out_max[t] has shape (beams_per_batch=1, ndm, nt).
        with od.dedisperse(frame) as outputs:
            if save:
                # Copy inside the context: GPU ring-buffer views are invalid afterward.
                snr_maps = [cp.asnumpy(outputs.out_max[t][0]) for t in range(od.ntrees)]
                argmax_maps = [
                    cp.asnumpy(outputs.out_argmax[t][0]) for t in range(od.ntrees)
                ]
                snr = max(float(a.max()) for a in snr_maps)
            else:
                snr = max(float(cp.asarray(outputs.out_max[t]).max()) for t in range(od.ntrees))

        saved_text = ""
        if save:
            assert path.endswith(".asdf"), path
            output_path = path[:-5] + "_snrmap.asdf"
            _write_snr_asdf(
                output_path, path, frame, od, snr_maps, argmax_maps,
                producer_start_time_chunk_index=producer_start_time_chunk_index,
            )
            saved_text = f"  saved={output_path}"

        if ichunk == 0:
            atomic_print(f"  nfreq={od.nfreq}, nt_in={od.nt_in}, ntrees={od.ntrees}, dtype={od.dtype}, "
                         f"time_sample_ms={od.time_sample_ms:.4f}")
        atomic_print(f"  beam {beam_id}  chunk {ichunk:3d} (tci={frame.time_chunk_index:3d}):  "
                     f"snr_max={snr:7.3f}{saved_text}")


def run_offline_dedisperser(acqdir=DEFAULT_ACQDIR, config_filename=DEFAULT_CONFIG,
                            max_chunks=None, save=False):
    """Offline single-beam dedispersion + peak-SNR print over an acqdir (see module docstring).

    Parameters
    ----------
    acqdir : str
        Directory of 'frame_b(BEAM)_t(CHUNK).asdf' files (a 'start_stream' acqdir).
    config_filename : str
        Dedispersion config YAML (loaded with DedispersionConfig.from_yaml).
    max_chunks : int or None
        If given, only process the first 'max_chunks' time chunks of each beam.
    save : bool
        If true, save each chunk's S/N maps beside its input ASDF file.
    """
    from . import Acquisition, DedispersionConfig

    acq = Acquisition(acqdir)
    atomic_print(f"Found {len(acq.beam_ids)} beam(s) in {acqdir}: {acq.beam_ids}")

    config = DedispersionConfig.from_yaml(config_filename)
    for beam_id in acq.beam_ids:
        files = acq.per_beam_filenames[beam_id]
        if max_chunks is not None:
            files = files[:max_chunks]
        _process_beam(beam_id, files, config, save=save)
