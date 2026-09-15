"""Stream detected events from native dedispersion maps to the event monitor."""
from __future__ import annotations

from contextlib import ExitStack
import math
import operator
import time

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
        raise ValueError("live producer start differs from the observation")
    if int(grouper.nt_in) != int(bundle["samples_per_chunk"]):
        raise ValueError("live samples per chunk differ from the observation")
    if tuple(int(b) for b in grouper.xengine_metadata.beam_ids) != tuple(bundle["beam_ids"]):
        raise ValueError("live beam IDs/order differ from the observation")
    cadence = float(grouper.dedispersion_config.time_sample_ms)
    if not math.isclose(cadence, float(bundle["time_sample_ms"]), rel_tol=0, abs_tol=1e-12):
        raise ValueError("live cadence differs from the observation")
    metadata = bundle["metadata"]
    for field in ("zone_nfreq", "zone_freq_edges", "noise_variance"):
        if field in metadata and list(getattr(grouper.xengine_metadata, field)) != list(metadata[field]):
            raise ValueError(f"live {field} differs from the observation")
    for field in ("dt_ns_per_seq", "seq_per_frb_time_sample", "unix_ns_at_seq_0", "beamset"):
        if field in metadata and getattr(grouper.xengine_metadata, field) != metadata[field]:
            raise ValueError(f"live {field} differs from the observation")
    if expected_nchunks < 1:
        raise ValueError("observation must contain at least one chunk")

    if "dedispersion_config_yaml" in bundle:
        import yaml
        expected = yaml.safe_load(bundle["dedispersion_config_yaml"])
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
                raise ValueError(f"live dedispersion {path} differs from observation configuration")
        compare(expected, actual, "config")


def run_online_grouper(bundle, grouper_addr, *, sifter_addr=None, progress=None):
    """Process a finite observation without saving catalogs or reports.

    After final event delivery, keep producer-owned CUDA mappings open until
    the dedisperser stops. Premature disconnects remain processing errors.
    """
    from .GrouperConfig import GrouperConfig
    from .ArgmaxMetadata import ARGMAX_ENCODING
    from .GrouperPipeline import GrouperSetup, CatalogRecorder
    from .rpc import FrbGrouper
    from .rpc.FrbSifterClient import FrbSifterClient

    start = _exact_integer(bundle["initial_chunk"], "initial_chunk")
    count = _exact_integer(bundle["nchunks"], "nchunks", minimum=1)
    configuration = GrouperConfig.from_mapping(bundle["grouper_config"])
    with ExitStack() as stack:
        g = stack.enter_context(FrbGrouper(grouper_addr, restore_cuda_device=False))
        validate_live_handshake(g, bundle, expected_start_chunk=start, expected_nchunks=count)
        setup = GrouperSetup(g.dedispersion_plan, g.dcores, configuration,
                            cuda_device_id=g.cuda_device_id)
        recorder = CatalogRecorder(
            setup, config_yaml=g.dedispersion_config_yaml_string,
            plan_yaml=g.dedispersion_plan_yaml_string, argmax_encoding=ARGMAX_ENCODING,
            pipeline="online")
        sifter = None
        if sifter_addr is not None:
            # Synchronous delivery exposes rejected final messages before completion.
            sifter = stack.enter_context(FrbSifterClient(sifter_addr, timeout=0))
            sifter.send_configuration(
                g.dedispersion_config_yaml_string, g.xengine_metadata_yaml_string,
                g.dedispersion_plan_yaml_string, bundle["grouper_config_yaml"], g.search_ip_addr)
        import cupy as cp
        beam_ids = tuple(int(b) for b in g.xengine_metadata.beam_ids)
        processors, pending_messages, coarsegrain = [], {}, {}

        def consume(batch_beams, window):
            batch = recorder.consume(batch_beams, window)
            recorder.batches.clear()
            owner = window.owner_source_chunk_index
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
            if progress is not None:
                progress(dict(phase="window", owner_source_chunk_index=owner,
                              events=len(batch.events["event_id"]),
                              timed_out=window.grouped.timed_out))

        for ibatch in range(int(g.nbatches)):
            first = ibatch * int(g.beams_per_batch)
            ids = beam_ids[first:first + int(g.beams_per_batch)]
            processors.append(setup.processor(
                ids, start, lambda window, ids=ids: consume(ids, window)))
        for relative in range(count):
            source = start + relative
            coarsegrain[source] = np.zeros(len(beam_ids), dtype=np.float64)
            for ibatch, processor in enumerate(processors):
                with g.get_output(relative, ibatch) as maps:
                    if int(maps.ichunk_fpga_based) != source:
                        raise ValueError("live output has an unexpected absolute source chunk")
                    first = int(maps.ibeam)
                    if beam_ids[first:first + len(processor.beam_ids)] != processor.beam_ids:
                        raise ValueError("live output beam axis changed")
                    maxima = cp.zeros(len(processor.beam_ids), dtype=cp.float64)
                    for array in maps.out_max:
                        maxima = cp.maximum(maxima, cp.max(array, axis=(1, 2)))
                    coarsegrain[source][first:first + len(processor.beam_ids)] = cp.asnumpy(maxima)
                    processor.process_chunk(maps.out_max, maps.out_argmax, source)
            if progress is not None:
                progress(dict(phase="chunk", completed_chunks=relative + 1, expected_chunks=count))
        for processor in processors:
            processor.finish(physical_end=True)
        if pending_messages:
            raise RuntimeError("incomplete beam coverage in finalized grouping windows")
        if progress is not None:
            progress(dict(phase="complete", events=recorder.event_offset))
        while not g.is_stopped:
            time.sleep(0.1)
