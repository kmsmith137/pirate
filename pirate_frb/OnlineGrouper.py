"""Controlled live adapter for the shared causal candidate processor.

This module never reads injection truth. It consumes actual producer outputs,
records the same scientific catalog as offline processing, and optionally sends
representatives to a threshold-only sifter. RFI probability zero is an explicit
classifier-bypass placeholder, not an estimated probability.
"""
from __future__ import annotations

from contextlib import ExitStack
import json
import math
import operator
from pathlib import Path
import time
import traceback

import numpy as np


def _exact_integer(value, name, *, minimum=0):
    if isinstance(value, (bool, np.bool_)):
        raise TypeError(f"{name} must be an integer")
    result = int(operator.index(value))
    if result < minimum:
        raise ValueError(f"{name} must be at least {minimum}")
    return result


def representative_events(columns, *, owner_source_chunk, nt_in, seq_per_sample):
    """Convert common decoded host columns, including next-chunk representatives.

    Arrival samples already carry the representative's absolute source chunk;
    the grouping owner is used only for the message coverage window. Arrival at
    the full band's low edge may precede or follow that window for early trees.
    """
    from .rpc.FrbSifterClient import FrbSifterEvents
    owner = _exact_integer(owner_source_chunk, "owner_source_chunk")
    nt = _exact_integer(nt_in, "nt_in", minimum=1)
    seq = _exact_integer(seq_per_sample, "seq_per_sample", minimum=1)
    start, end = owner * nt * seq, (owner + 1) * nt * seq
    limit = np.iinfo(np.int64).max
    if end > limit:
        raise OverflowError("FPGA message interval exceeds int64")
    samples = np.asarray(columns["toa_sample_abs"], dtype=np.float64)
    if samples.ndim != 1 or not np.all(np.isfinite(samples)):
        raise ValueError("arrival samples must be a finite one-dimensional array")
    # Carry out the unit conversion with extended precision where available;
    # no new scientific decoding is performed by this transport adapter.
    timestamps = np.maximum(np.rint(samples.astype(np.longdouble) * seq), 0)
    if np.any(timestamps > limit):
        raise OverflowError("event FPGA timestamp exceeds int64")
    return FrbSifterEvents(
        beam_ids=columns["beam_id"], fpga_timestamps=timestamps.astype(np.int64),
        dms=columns["dm"], snrs=columns["snr"], rfi_probs=np.zeros(samples.size),
        widths_ms=columns["width_ms"], subband_freqs_lo_MHz=columns["freq_lo_MHz"],
        subband_freqs_hi_MHz=columns["freq_hi_MHz"], tree_index=columns["tree"],
        chunk_fpga_start=start, chunk_fpga_end=end)


def validate_live_handshake(grouper, bundle, *, expected_start_chunk, expected_nchunks):
    """Check temporal, beam, and frequency identities before accepting live maps."""
    if _exact_integer(grouper.initial_chunk, "initial_chunk") != expected_start_chunk:
        raise ValueError("live producer start differs from the controlled observation")
    if int(grouper.nt_in) != int(bundle["samples_per_chunk"]):
        raise ValueError("live samples per chunk differ from the saved observation")
    if tuple(int(b) for b in grouper.xengine_metadata.beam_ids) != tuple(bundle["beam_ids"]):
        raise ValueError("live beam IDs/order differ from the saved observation")
    cadence = float(grouper.dedispersion_config.time_sample_ms)
    if not math.isclose(cadence, float(bundle["time_sample_ms"]), rel_tol=0, abs_tol=1e-12):
        raise ValueError("live cadence differs from the saved observation")
    metadata = bundle["metadata"]
    if "metadata_path" in bundle:
        import yaml
        from .pirate_pybind11 import XEngineMetadata
        # Compare the producer's canonical representation with the same native
        # parser used to generate/send frames. In particular a YAML scalar
        # noise_variance broadcasts to a vector with one value per zone.
        expected_metadata = XEngineMetadata.from_yaml_file(bundle["metadata_path"])
        metadata = yaml.safe_load(expected_metadata.to_yaml_string())
    for field in ("zone_nfreq", "zone_freq_edges", "noise_variance"):
        if field in metadata and list(getattr(grouper.xengine_metadata, field)) != list(metadata[field]):
            raise ValueError(f"live {field} differs from the saved observation")
    for field in ("dt_ns_per_seq", "seq_per_frb_time_sample", "unix_ns_at_seq_0", "beamset"):
        if field in metadata and getattr(grouper.xengine_metadata, field) != metadata[field]:
            raise ValueError(f"live {field} differs from the saved observation")
    if expected_nchunks < 1:
        raise ValueError("controlled observation must contain at least one chunk")

    if "dedispersion_config_path" in bundle:
        import yaml
        expected = yaml.safe_load(Path(bundle["dedispersion_config_path"]).read_text())
        actual = yaml.safe_load(grouper.dedispersion_config_yaml_string)
        # Metadata is authoritative for the fields FrbServer postfills.
        expected["time_sample_ms"] = float(bundle["time_sample_ms"])
        expected["beams_per_gpu"] = len(bundle["beam_ids"])
        for key in ("zone_nfreq", "zone_freq_edges"):
            if key in metadata:
                expected[key] = metadata[key]

        def compare(want, got, path):
            if isinstance(want, dict):
                if not isinstance(got, dict):
                    raise ValueError(f"live dedispersion {path} is not a mapping")
                for key, value in want.items():
                    if key not in got:
                        raise ValueError(f"live dedispersion {path}.{key} is missing")
                    compare(value, got[key], f"{path}.{key}")
            elif isinstance(want, list):
                if not isinstance(got, list) or len(want) != len(got):
                    raise ValueError(f"live dedispersion {path} has a different length")
                for i, (left, right) in enumerate(zip(want, got)):
                    compare(left, right, f"{path}[{i}]")
            elif want != got:
                raise ValueError(f"live dedispersion {path} differs from experiment configuration")
        compare(expected, actual, "config")


def run_online_grouper(bundle_dir, output, grouper_addr, *, sifter_addr=None,
                       stop_event=None, ready_event=None, completion_event=None,
                       status_queue=None, expected_start_chunk=None,
                       expected_nchunks=None, report_path=None):
    """Process a finite controlled observation, then hold IPC until server stops.

    The sender must supply its two transport-drain chunks. They cause final real
    observation outputs to assemble but do not become scientific map context.
    ``completion_event`` means every real output was processed, the final owner
    flushed, synchronous sifter deliveries acknowledged, and the catalog saved.
    Only then should the coordinator finish writes, stop the server, and set
    ``stop_event``. A premature stop or disconnect is an error, never a physical
    end. ``ready_event`` follows the producer handshake, so the sender must run
    before the coordinator waits for it. CUDA must use spawn, not fork.
    """
    from .ControlledObservation import load_experiment_bundle
    from .OfflineGrouperConfig import load_offline_grouper_config
    from .ArgmaxMetadata import ARGMAX_ENCODING
    from .SharedGrouper import GrouperSetup, CatalogRecorder
    from .rpc import FrbGrouper
    from .rpc.FrbSifterClient import FrbSifterClient

    bundle = load_experiment_bundle(bundle_dir, verify_hashes=False, require_complete=True)
    start = _exact_integer(bundle["initial_chunk"], "initial_chunk")
    count = _exact_integer(bundle["nchunks"], "nchunks", minimum=1)
    if expected_start_chunk is not None and _exact_integer(expected_start_chunk, "expected_start_chunk") != start:
        raise ValueError("expected start chunk disagrees with bundle")
    if expected_nchunks is not None and _exact_integer(expected_nchunks, "expected_nchunks", minimum=1) != count:
        raise ValueError("expected chunk count disagrees with bundle")
    configuration = load_offline_grouper_config(bundle["grouper_config_path"])
    output = Path(output).resolve()
    report_path = Path(report_path).resolve() if report_path is not None else Path(str(output) + ".online.json")
    for path in (output, report_path):
        if path.exists():
            raise FileExistsError(f"refusing to replace existing experiment result: {path}")
        if not path.parent.is_dir():
            raise ValueError(f"result directory does not exist: {path.parent}")
    report = dict(format="pirate_frb.controlled_online_grouper", format_version=1,
                  complete=False, classifier="bypassed", initial_chunk=start,
                  nchunks=count, chunks=[], windows=[], started_monotonic_ns=time.monotonic_ns())

    def notify(phase, **details):
        if status_queue is not None:
            status_queue.put(dict(phase=phase, **details))

    def write_report():
        # New experiment result paths are exclusively created. All report fields
        # are CPU scalars; the scientific catalog has its own atomic validator.
        with report_path.open("x") as stream:
            json.dump(report, stream, indent=2, allow_nan=False)
            stream.write("\n")

    try:
        with ExitStack() as stack:
            g = stack.enter_context(FrbGrouper(grouper_addr, restore_cuda_device=False))
            validate_live_handshake(g, bundle, expected_start_chunk=start, expected_nchunks=count)
            setup = GrouperSetup(g.dedispersion_plan, g.dcores, configuration,
                                cuda_device_id=g.cuda_device_id)
            recorder = CatalogRecorder(
                setup, config_yaml=g.dedispersion_config_yaml_string,
                plan_yaml=g.dedispersion_plan_yaml_string, argmax_encoding=ARGMAX_ENCODING,
                pipeline="online")
            report.update(cuda_device_id=int(g.cuda_device_id), dcores=list(setup.dcores),
                          argmax_encoding=ARGMAX_ENCODING, nt_in=int(g.nt_in),
                          time_sample_ms=float(g.dedispersion_config.time_sample_ms),
                          receiver_assembly_lookahead_chunks=2, grouping_lookahead_chunks=1)
            sifter = None
            if sifter_addr is not None:
                # Synchronous delivery makes a rejected final message visible to
                # the coordinator before scientific completion is announced.
                sifter = stack.enter_context(FrbSifterClient(sifter_addr, timeout=0))
                sifter.send_configuration(
                    g.dedispersion_config_yaml_string, g.xengine_metadata_yaml_string,
                    g.dedispersion_plan_yaml_string,
                    Path(bundle["grouper_config_path"]).read_text(), g.search_ip_addr)
            import cupy as cp
            beam_ids = tuple(int(b) for b in g.xengine_metadata.beam_ids)
            processors, pending_messages, coarsegrain = [], {}, {}
            active_source = start
            active_available = None

            def consume(batch_beams, window):
                batch = recorder.consume(batch_beams, window)
                owner = window.owner_source_chunk_index
                report["windows"].append(dict(
                    grouping_window_id=len(recorder.windows) - 1,
                    owner_source_chunk_index=owner, beam_ids=list(batch_beams),
                    available_source_chunk_index=active_source,
                    output_available_monotonic_ns=active_available,
                    grouped_monotonic_ns=time.monotonic_ns(), events=len(batch.events["event_id"]),
                    timed_out=window.grouped.timed_out, output_status=window.output_status))
                pending_messages.setdefault(owner, []).append(batch.events)
                if len(pending_messages[owner]) == int(g.nbatches):
                    parts = pending_messages.pop(owner)
                    columns = {name: np.concatenate([p[name] for p in parts]) for name in parts[0]}
                    if sifter is not None:
                        events = representative_events(
                            columns, owner_source_chunk=owner, nt_in=g.nt_in,
                            seq_per_sample=g.xengine_metadata.seq_per_frb_time_sample)
                        sifter.send_events(int(g.xengine_metadata.beamset), events, coarsegrain[owner])
                    coarsegrain.pop(owner, None)

            for ibatch in range(int(g.nbatches)):
                first = ibatch * int(g.beams_per_batch)
                ids = beam_ids[first:first + int(g.beams_per_batch)]
                processors.append(setup.processor(
                    ids, start, lambda window, ids=ids: consume(ids, window)))
            notify("ready", cuda_device_id=int(g.cuda_device_id))
            if ready_event is not None:
                ready_event.set()
            for relative in range(count):
                active_source = start + relative
                coarsegrain[active_source] = np.zeros(len(beam_ids), dtype=np.float64)
                for ibatch, processor in enumerate(processors):
                    if stop_event is not None and stop_event.is_set():
                        raise InterruptedError("stopped before every observation output arrived")
                    acquired_begin = time.monotonic_ns()
                    with g.get_output(relative, ibatch) as maps:
                        active_available = time.monotonic_ns()
                        if int(maps.ichunk_fpga_based) != active_source:
                            raise ValueError("live output has an unexpected absolute source chunk")
                        first = int(maps.ibeam)
                        if beam_ids[first:first + len(processor.beam_ids)] != processor.beam_ids:
                            raise ValueError("live output beam axis changed")
                        maxima = cp.zeros(len(processor.beam_ids), dtype=cp.float64)
                        for array in maps.out_max:
                            maxima = cp.maximum(maxima, cp.max(array, axis=(1, 2)))
                        coarsegrain[active_source][first:first + len(processor.beam_ids)] = cp.asnumpy(maxima)
                        processor.process_chunk(maps.out_max, maps.out_argmax, active_source)
                    report["chunks"].append(dict(
                        source_chunk_index=active_source, batch_index=ibatch,
                        beam_ids=list(processor.beam_ids), wait_started_monotonic_ns=acquired_begin,
                        available_monotonic_ns=active_available,
                        released_monotonic_ns=time.monotonic_ns()))
            for processor in processors:
                processor.finish(physical_end=True)
            if pending_messages:
                raise RuntimeError("incomplete beam coverage in finalized grouping windows")
            written = recorder.write(output, processors, complete=True)
            report.update(complete=True, catalog=written, events=recorder.event_offset,
                          completed_monotonic_ns=time.monotonic_ns())
            write_report()
            notify("complete", catalog=written, report=str(report_path))
            if completion_event is not None:
                completion_event.set()
            # The producer retains ownership of exported CUDA memory. Do not
            # close IPC when only the consumer is finished: pending disk writes
            # and server shutdown are coordinated by the parent process.
            while not g.is_stopped:
                if stop_event is not None:
                    if stop_event.wait(0.1):
                        break
                else:
                    time.sleep(0.1)
            return written
    except BaseException as exc:
        report.update(complete=False, error=repr(exc), traceback=traceback.format_exc(),
                      failed_monotonic_ns=time.monotonic_ns())
        if not report_path.exists():
            write_report()
        notify("error", error=repr(exc), traceback=report["traceback"], report=str(report_path))
        raise
