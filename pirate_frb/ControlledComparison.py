"""Scientific equality and online-capture acceptance for controlled observations.

Only this validation layer reads injection truth. Catalog comparisons join on
source cells and grouping partitions, never traversal-dependent catalog IDs.
"""
from __future__ import annotations

from collections import Counter
import hashlib
import json
import math
from pathlib import Path

import numpy as np
import yaml


SOURCE_FIELDS = ("beam_id", "source_chunk_index", "tree", "idm", "itime", "argmax_token")
ID_FIELDS = {"event_id", "candidate_id", "representative_candidate_id", "grouping_window_id"}
BATCH_FIELDS = {"beams_per_gpu", "beams_per_batch", "num_active_batches"}
K_DM = 4148.808
FLOAT_TOLERANCES = {
    "snr": (1.e-3, 2.e-3),  # One float16 relative ULP plus a small absolute floor.
    "toa_sample_abs": (0.0, 1.e-6),  # 1e-9 s at a 1 ms cadence.
    "dm": (1.e-12, 1.e-9),
    "width_samp": (0.0, 1.e-12), "width_ms": (1.e-12, 1.e-12),
    "freq_lo_MHz": (0.0, 1.e-9), "freq_hi_MHz": (0.0, 1.e-9),
    "dm_step": (1.e-12, 1.e-12), "time_step_samples": (0.0, 1.e-12),
}


def _require(condition, message):
    if not condition:
        raise ValueError(message)


def _json(path):
    return json.loads(Path(path).read_text())


def _plain(value):
    if isinstance(value, dict):
        return {str(k): _plain(v) for k, v in value.items()}
    if isinstance(value, (tuple, list)):
        return [_plain(v) for v in value]
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.generic):
        return value.item()
    return value


def _zone_variances(value, zone_nfreq):
    """XEngineMetadata permits a scalar variance broadcast across zones."""
    variances = np.asarray(value, dtype="<f8")
    if variances.ndim == 0:
        variances = np.full(len(zone_nfreq), float(variances), dtype="<f8")
    _require(variances.shape == (len(zone_nfreq),) and np.isfinite(variances).all()
             and (variances > 0).all(), "Invalid per-zone noise variances")
    return variances.tolist()


def _normalized_metadata(metadata):
    """Normalize only the two equivalent forms accepted by native metadata."""
    metadata = _plain(metadata)
    metadata.setdefault("freq_channels", [])
    metadata["noise_variance"] = _zone_variances(metadata["noise_variance"], metadata["zone_nfreq"])
    return metadata


def _hash(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1024*1024), b""):
            digest.update(block)
    return digest.hexdigest()


def read_catalog(path):
    import asdf
    from .TriggerCatalog import validate_trigger_catalog_tree

    with asdf.open(path, lazy_load=False, memmap=False) as handle:
        validate_trigger_catalog_tree(handle.tree)
        return dict(
            events={k: np.array(v, copy=True) for k, v in handle.tree["events"].items()},
            members={k: np.array(v, copy=True) for k, v in handle.tree["members"].items()},
            coverage={k: np.array(v, copy=True) for k, v in handle.tree["coverage"].items()},
            metadata=_plain(handle.tree["metadata"]),
        )


def _source(table, row):
    return tuple(int(table[key][row]) for key in SOURCE_FIELDS)


def _groups(catalog):
    events, members = catalog["events"], catalog["members"]
    owners = {int(w["grouping_window_id"]): int(w["owner_source_chunk_index"])
              for w in catalog["metadata"]["grouping_windows"]}
    result = {}
    for event_row, event_id in enumerate(events["event_id"]):
        rows = np.flatnonzero(members["event_id"] == event_id)
        keys = [_source(members, row) for row in rows]
        _require(len(keys) == len(set(keys)), "Duplicate source cell within one grouping partition")
        owner = owners[int(events["grouping_window_id"][event_row])]
        signature = (owner, tuple(sorted(keys)))
        _require(signature not in result, "Duplicate grouping partition")
        result[signature] = (event_row, dict(zip(keys, rows)))
    return result


def _compare_row(left, il, right, ir, label, maxima):
    _require(set(left) == set(right), f"{label}: catalog columns differ")
    for key in left:
        if key in ID_FIELDS:
            continue
        a, b = left[key][il], right[key][ir]
        if key in FLOAT_TOLERANCES:
            rtol, atol = FLOAT_TOLERANCES[key]
            _require(np.isfinite(a) and np.isfinite(b), f"{label}.{key}: nonfinite measurement")
            delta = abs(float(a)-float(b))
            maxima[key] = max(maxima.get(key, 0.0), delta)
            _require(np.isclose(a, b, rtol=rtol, atol=atol),
                     f"{label}.{key}: {a} != {b} (rtol={rtol}, atol={atol})")
        else:
            _require(a == b, f"{label}.{key}: {a} != {b}")


def _metadata_science(metadata):
    producer = metadata["producer"]
    config = yaml.safe_load(producer["config_yaml"])
    plan = yaml.safe_load(producer["plan_yaml"])
    # GPU/host ring shapes and beam launch scheduling differ between the
    # offline one-beam producer and the live server. Tree geometry may not.
    geometry = {key: plan[key] for key in (
        "dtype", "nfreq", "nt_in", "toplevel_tree_rank", "num_primary_trees",
        "stage1_dd_rank", "stage1_amb_rank", "ntrees", "trees") if key in plan}
    return dict(
        config={k: v for k, v in config.items() if k not in BATCH_FIELDS},
        geometry=geometry, dcores=list(producer["dcores"]),
        argmax_encoding=producer["argmax_encoding"],
        processing={k: v for k, v in metadata["processing"].items() if k != "beam_batch_size"},
        startup=sorted(metadata["startup_by_beam"], key=lambda r: r["beam_id"]),
        units=metadata["units"],
    )


def compare_catalogs(reference, actual, *, label="online"):
    """Compare scientific cells, representatives and grouping partitions."""
    _require(_metadata_science(reference["metadata"]) == _metadata_science(actual["metadata"]),
             f"{label}: scientific producer/processing/startup metadata differs")
    coverage = lambda cat: set(zip(cat["coverage"]["beam_id"].tolist(),
                                   cat["coverage"]["source_chunk_index"].tolist()))
    _require(coverage(reference) == coverage(actual), f"{label}: processed coverage differs")
    ref_members, out_members = reference["members"], actual["members"]
    ref_keys = Counter(_source(ref_members, row) for row in range(len(ref_members["candidate_id"])))
    out_keys = Counter(_source(out_members, row) for row in range(len(out_members["candidate_id"])))
    _require(ref_keys == out_keys,
             f"{label}: source candidates differ: {sum((ref_keys-out_keys).values())} missing, "
             f"{sum((out_keys-ref_keys).values())} extra")
    expected, observed = _groups(reference), _groups(actual)
    _require(expected.keys() == observed.keys(), f"{label}: grouping partitions differ")
    maxima = {}
    for signature, (left_event, left_rows) in expected.items():
        right_event, right_rows = observed[signature]
        _require(_source(reference["events"], left_event) == _source(actual["events"], right_event),
                 f"{label}: representative source cell differs")
        _compare_row(reference["events"], left_event, actual["events"], right_event,
                     f"{label}.event", maxima)
        for key, left_row in left_rows.items():
            _compare_row(ref_members, left_row, out_members, right_rows[key],
                         f"{label}.member{key}", maxima)
    return dict(pass_=True, events=len(expected), members=sum(ref_keys.values()),
                source_cells=len(ref_keys), max_absolute_differences=maxima,
                ignored_identity_fields=sorted(ID_FIELDS),
                ignored_producer_batch_fields=sorted(BATCH_FIELDS))


def _truth_tolerances(member, tree, dcore, time_sample_s):
    # One delay sample at this tree's downsampling, measured across the actual
    # selected frequency subband, gives the DM trial resolution. Use two such
    # trials to allow a noisy maximum to move to a neighboring trial. The
    # decoded TOA is extrapolated to 300 MHz; propagate that DM uncertainty.
    ipri = int(member["primary_tree_index"])
    lo, hi = float(member["freq_lo_MHz"]), float(member["freq_hi_MHz"])
    lever = K_DM * (lo**-2-hi**-2)
    _require(lever > 0, "Invalid detected frequency span")
    fine_dm = (2**ipri) * time_sample_s / lever
    dm_tolerance = 2 * fine_dm
    profile = (int(member["argmax_token"]) >> 8) & 255
    level = (profile-1)//3 if profile else 0
    time_trial_s = min(int(dcore), 2**level) * 2**ipri * time_sample_s
    extrapolation = K_DM * (300.0**-2-lo**-2)
    # A full filter's nominal extent is deliberately NOT compared to injected
    # Gaussian sigma. Only its sampling quantum enters the timing tolerance.
    toa_tolerance_s = 2*time_trial_s + dm_tolerance*max(0.0, extrapolation)
    return dm_tolerance, toa_tolerance_s


def validate_recovery(catalog, bursts, time_sample_s):
    members = catalog["members"]
    plan = yaml.safe_load(catalog["metadata"]["producer"]["plan_yaml"])
    dcores = catalog["metadata"]["producer"]["dcores"]
    results = []
    for burst in bursts:
        detections = []
        for expected in burst["expected_tree_arrivals"]:
            itree = int(expected["tree_index"])
            indices = np.flatnonzero((members["beam_id"] == burst["beam_id"])
                                     & (members["tree"] == itree))
            matches = []
            for row in indices:
                item = {key: values[row] for key, values in members.items()}
                dm_tol, toa_tol = _truth_tolerances(item, plan["trees"][itree],
                                                    dcores[itree], time_sample_s)
                dm_error = float(item["dm"]) - burst["dm"]
                toa = float(item["toa_sample_abs"]) * time_sample_s
                toa_error = toa - burst["toa_seconds"]
                if abs(dm_error) <= dm_tol and abs(toa_error) <= toa_tol:
                    _require(not (int(item["edge_flags"]) & (1 << 4)),
                             f"Burst {burst['id']} tree {itree} is startup-incomplete")
                    matches.append((float(item["snr"]), row, dm_error, toa_error, dm_tol, toa_tol))
            _require(matches, f"Burst {burst['id']} was not recovered in expected tree {itree}")
            _, row, dm_error, toa_error, dm_tol, toa_tol = max(matches)
            detections.append(dict(tree=itree, early_trigger_level=expected["early_trigger_level"],
                event_id=int(members["event_id"][row]), source_cell=list(_source(members, row)),
                dm=float(members["dm"][row]), dm_error=dm_error, dm_tolerance=dm_tol,
                toa_seconds=float(members["toa_sample_abs"][row])*time_sample_s,
                toa_error_seconds=toa_error, toa_tolerance_seconds=toa_tol,
                snr=float(members["snr"][row]), nominal_filter_width_ms=float(members["width_ms"][row])))
        results.append(dict(burst_id=burst["id"], role=burst["role"], detections=detections,
                            injected_gaussian_sigma_ms=burst["width_ms"]))
    return results


def _safe_file(root, relative):
    root = Path(root).resolve()
    _require(isinstance(relative, str) and not Path(relative).is_absolute(), "Capture path must be relative")
    path = (root / relative).resolve()
    _require(path.is_relative_to(root), "Capture filename escapes output directory")
    return path


def packed_frame_signature(path):
    from .core import AssembledFrame
    from .ReplayObservation import metadata_identity

    frame = AssembledFrame.from_asdf(str(path))
    return dict(beam_id=int(frame.beam_id), time_chunk_index=int(frame.time_chunk_index),
                ntime=int(frame.ntime), nfreq=int(frame.nfreq),
                metadata=metadata_identity(frame.metadata),
                data_sha256=hashlib.sha256(frame.data).hexdigest(),
                scales_sha256=hashlib.sha256(frame.scales_offsets).hexdigest())


def _capture_matches_burst(cap, burst, policy, tick_s):
    first = cap["first_event"]
    return (int(first["beam_id"]) == burst["beam_id"]
            and abs(first["dm"]-burst["dm"]) <= policy["association_dm_tolerance"]
            and abs(first["fpga_timestamp"]*tick_s-burst["toa_seconds"])
                <= policy["association_toa_tolerance_seconds"])


def validate_captures(ledger, bundle, run_dir, bursts, policy, *, accept_early,
                      signature_reader=packed_frame_signature, source_cache=None):
    _require(ledger.get("complete") is True and not ledger.get("errors"),
             "Capture ledger is incomplete or contains errors")
    _require(ledger.get("classifier_mode") == "bypass", "Capture classifier bypass not recorded")
    _require(ledger.get("accept_early") is accept_early, "Wrong early-trigger capture policy")
    notifications = ledger["notifications"]
    entries = {(int(e["beam_id"]), int(e["time_chunk_index"])): e for e in bundle["frame_entries"]}
    ntime = int(bundle["samples_per_chunk"])
    tick_s = float(bundle["metadata"]["dt_ns_per_seq"]) * 1e-9
    seq_per_chunk = ntime * int(bundle["metadata"]["seq_per_frb_time_sample"])
    nbeams = len(bundle["beam_ids"])
    source_cache = {} if source_cache is None else source_cache
    captured = {}
    checked_files = set()
    request_count = 0
    for cap in ledger["captures"]:
        cap_id = cap["capture_id"]
        captured[cap_id] = set()
        for request in cap["requests"]:
            request_count += 1
            expected, promised = set(request["expected_files"]), set(request["promised_files"])
            beam = int(cap["first_event"]["beam_id"])
            derived = {f"{cap['acqdir']}/frame_b{beam}_t{chunk}.asdf" for chunk in range(
                int(request["start_seq"]) // seq_per_chunk,
                (int(request["end_seq"])+seq_per_chunk-1)//seq_per_chunk)}
            _require(expected == derived, "Requested frame coverage differs from the FPGA interval")
            _require(promised <= expected, "Server promised a file outside the requested capture interval")
            _require(set(request["missing_files"]) == expected-promised, "Capture truncation ledger is inconsistent")
            for name in promised:
                _require(name in notifications and not notifications[name]["error"],
                         f"Missing successful completion notification for {name}")
                path = _safe_file(Path(run_dir) / "captures", name)
                _require(path.is_file(), f"Promised capture file does not exist: {path}")
                signature = signature_reader(path)
                key = int(signature["beam_id"]), int(signature["time_chunk_index"])
                _require(key in entries, f"Captured file has no matching science input: {name}")
                if key not in source_cache:
                    source_path = _safe_file(bundle["bundle_dir"], entries[key]["path"])
                    source_cache[key] = signature_reader(source_path)
                _require(signature == source_cache[key], f"Captured packed data or metadata differs from input: {name}")
                captured[cap_id].add(key)
                checked_files.add(name)
    burst_results = []
    for burst in bursts:
        matches = [cap for cap in ledger["captures"] if _capture_matches_burst(cap, burst, policy, tick_s)]
        _require(matches, f"No capture decision matches burst {burst['id']}")
        matches.sort(key=lambda c: c["requests"][0]["requested_elapsed_seconds"])
        first = matches[0]
        needed = {(burst["beam_id"], chunk) for chunk in range(
            int(burst["sample_start"]) // ntime, (int(burst["sample_end"])+ntime-1)//ntime)}
        actual = set().union(*(captured[cap["capture_id"]] for cap in matches))
        missing = sorted(needed-actual)
        expect_complete = accept_early or burst["role"] == "full_band"
        _require((not missing) == expect_complete,
                 f"Burst {burst['id']}: expected capture complete={expect_complete}, missing {len(missing)} chunks")
        if expect_complete:
            _require(needed <= captured[first["capture_id"]],
                     f"Burst {burst['id']} complete only by combining separate captures")
        if burst["role"] == "early_trigger":
            _require((first["first_early_trigger_level"] > 0) == accept_early,
                     "High-DM first capture used the wrong trigger tree policy")
        first_request = first["requests"][0]
        status = first_request["status_before"]
        beam_index = bundle["beam_ids"].index(burst["beam_id"])
        earliest_frame = min(chunk for _, chunk in needed) * nbeams + beam_index
        reap_bound = max(int(status["rb_start"]), int(status["rb_reaped"]))
        # Reaper can retain already-written backing data for another capture;
        # output coverage is authoritative. For this prescribed separated
        # experiment, the full-band control must show the first raw frame
        # outside the logical/physical live interval at request time.
        if expect_complete:
            _require(earliest_frame >= reap_bound,
                     f"Burst {burst['id']} request was after earliest raw-frame expiry")
        else:
            _require(earliest_frame < reap_bound,
                     "Incomplete high-DM control capture lacks evidence of past-data expiry")
            _require((burst["beam_id"], min(chunk for _, chunk in needed)) in missing,
                     "High-DM control did not lose the leading raw data")
        burst_results.append(dict(burst_id=burst["id"], role=burst["role"], complete=not missing,
            first_capture_id=first["capture_id"], first_early_trigger_level=first["first_early_trigger_level"],
            needed_chunks=len(needed), saved_chunks=len(needed & actual),
            missing_chunks=[chunk for _, chunk in missing],
            earliest_required_frame_id=earliest_frame, live_frame_lower_bound_at_request=reap_bound,
            first_request_receiver_elapsed_seconds=float(first_request["requested_elapsed_seconds"]),
            request_time_reference="seconds since this capture receiver started; not replay elapsed time",
            first_request_rpc_seconds=first_request.get("rpc_seconds"),
            first_request_missing_files=list(first_request["missing_files"]),
            leading_frame_retention_margin_chunks=(earliest_frame-reap_bound)/nbeams,
            first_request_status=status))
    if not accept_early:
        _require(any(d["action"] == "early_capture_suppressed" for d in ledger["decisions"]),
                 "Control run did not record any suppressed early-trigger capture decisions")
    return dict(complete=True, checked_files=len(checked_files), requests=request_count,
                bursts=burst_results)


def _validate_catalog_completion(catalog, bundle, pipeline):
    metadata = catalog["metadata"]
    _require(metadata["pipeline"] == pipeline, f"Expected {pipeline} catalog provenance")
    _require(metadata["processing"]["complete"] is True, "Scientific processing did not complete")
    _require(metadata["processing"]["timeout_ms"] == 0, "Disable grouping deadlines for equality validation")
    _require(all(row["status"] == "authoritative" and row["producer_start_chunk_index"] == bundle["initial_chunk"]
                 for row in metadata["startup_by_beam"]), "Authoritative startup provenance differs")
    _require(all(not w["timed_out"] and w["output_status"] == "complete"
                 for w in metadata["grouping_windows"]), "Incomplete grouping window")
    expected = {(beam, chunk) for beam in bundle["beam_ids"] for chunk in range(
        bundle["initial_chunk"], bundle["initial_chunk"]+bundle["nchunks"])}
    observed = set(zip(catalog["coverage"]["beam_id"].tolist(),
                       catalog["coverage"]["source_chunk_index"].tolist()))
    _require(observed == expected, "Catalog coverage does not match the complete science observation")


def _validate_replay(replay, bundle, max_lag_s):
    _require(replay["state"] == "sender_complete", "Replay sender did not complete")
    _require(replay["rate"] == 1.0, "Online acceptance requires 1x observing rate")
    _require(replay["input_hashes_verified"] is True, "Replay did not verify raw input hashes")
    _require(replay["bundle_sha256"] == bundle["manifest_sha256"], "Replay used a different observation bundle")
    _require(replay["science_nchunks"] == bundle["nchunks"] and replay["transport_tail_chunks"] == 2,
             "Replay science length or transport tail differs")
    _require(replay["max_chunk_dispatch_lag_s"] <= max_lag_s,
             f"Online sender dispatch exceeded latency budget {max_lag_s}s")
    _require(replay["max_enqueue_lag_s"] <= max_lag_s,
             f"Online sender enqueue exceeded latency budget {max_lag_s}s")
    _require(replay["server_status"], "Replay has no observed retention/backlog evidence")
    return dict(rate=1.0, max_enqueue_lag_s=replay["max_enqueue_lag_s"],
        max_chunk_dispatch_lag_s=replay["max_chunk_dispatch_lag_s"],
        allowed_sender_lag_s=max_lag_s,
        max_observed_backlog_chunks=max(s["assembled_backlog_chunks"] for s in replay["server_status"]),
        min_observed_unreaped_retention_s=min(s["unreaped_retention_s"] for s in replay["server_status"]),
        max_observed_unreaped_retention_s=max(s["unreaped_retention_s"] for s in replay["server_status"]))


def _save_result(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".tmp")
    temporary.write_text(json.dumps(_plain(value), indent=2, allow_nan=False) + "\n")
    temporary.replace(path)


def detection_timing(catalog, recovery, truth, replay, online_report):
    """Join timing by catalog window identity using the common monotonic clock.

    This is additional diagnostic evidence, not another acceptance condition.
    Capture receiver-relative timestamps have a different origin and are never
    subtracted from these replay-relative timestamps.
    """
    if not online_report.get("complete"):
        return dict(available=False, reason="online grouper timing report is incomplete")
    by_window = {int(w["grouping_window_id"]): w for w in online_report["windows"]}
    by_burst = {b["id"]: b for b in truth}
    events = catalog["events"]
    by_event = {int(value): row for row, value in enumerate(events["event_id"])}
    origin = float(replay["clock_origin_monotonic_s"])
    observation_start = replay["initial_chunk"]*replay["samples_per_chunk"]*replay["time_sample_s"]
    rows = []
    for burst_result in recovery:
        burst = by_burst[burst_result["burst_id"]]
        expected = {int(t["tree_index"]): t for t in burst["expected_tree_arrivals"]}
        for detection in burst_result["detections"]:
            event_row = by_event[detection["event_id"]]
            window_id = int(events["grouping_window_id"][event_row])
            window = by_window.get(window_id)
            if window is None:
                rows.append(dict(burst_id=burst["id"], tree=detection["tree"],
                                 available=False, reason="grouping window has no timing row"))
                continue
            grouped_elapsed = int(window["grouped_monotonic_ns"])*1e-9-origin
            available_ns = window.get("output_available_monotonic_ns")
            available_elapsed = None if available_ns is None else int(available_ns)*1e-9-origin
            arrival = float(expected[detection["tree"]]["arrival_seconds"])-observation_start
            rows.append(dict(burst_id=burst["id"], tree=detection["tree"], available=True,
                early_trigger_level=detection["early_trigger_level"], grouping_window_id=window_id,
                owner_source_chunk_index=window["owner_source_chunk_index"],
                available_source_chunk_index=window["available_source_chunk_index"],
                expected_tree_arrival_elapsed_seconds=arrival,
                output_available_replay_elapsed_seconds=available_elapsed,
                grouped_replay_elapsed_seconds=grouped_elapsed,
                grouping_after_tree_arrival_seconds=grouped_elapsed-arrival,
                lead_to_300MHz_arrival_seconds=burst["toa_seconds"]-observation_start-grouped_elapsed))
    return dict(available=True, time_reference="seconds since replay monotonic observation origin",
                receiver_assembly_lookahead_chunks=online_report.get("receiver_assembly_lookahead_chunks"),
                grouping_lookahead_chunks=online_report.get("grouping_lookahead_chunks"), detections=rows)


def _validate_run(run, catalog, bundle, *, online):
    _require(run.get("state") == "complete" and run.get("complete") is True,
             "Run lifecycle did not complete cleanly")
    _require(run.get("mode") == ("online" if online else "offline"), "Wrong run mode")
    _require(run.get("bundle_manifest_sha256") == bundle["manifest_sha256"],
             "Run provenance identifies a different observation")
    _require(run.get("weights_mode") == "analytic" and run.get("detrending") is False,
             "Analytic-weight and unmodified dequantization execution provenance is absent")
    _require(run.get("shared_processing") == "SharedGrouper", "Run does not identify shared scientific processing")
    canonical = _normalized_metadata(bundle["metadata"])
    _require(_zone_variances(run.get("noise_variance"), canonical["zone_nfreq"])
             == canonical["noise_variance"],
             "Run noise variances differ from the acquisition metadata")
    _require(run.get("source", {}).get("module_sha256"), "Run has no source-module fingerprints")
    result = dict(weights_mode="analytic", detrending=False,
                  weight_values_read_back=False, noise_variance=canonical["noise_variance"],
                  source=run["source"])
    if online:
        _require(run.get("scientific_outputs_drained") is True
                 and run.get("promised_files_drained") is True,
                 "Online run lacks scientific/file drain completion evidence")
        producer = run.get("producer")
        _require(isinstance(producer, dict), "Online run lacks actual producer metadata snapshot")
        original = catalog["metadata"]["producer"]
        for name in ("dcores", "argmax_encoding"):
            _require(producer[name] == original[name], f"Running producer {name} differs from catalog")
        for name in ("config_yaml", "plan_yaml"):
            _require(yaml.safe_load(producer[name]) == yaml.safe_load(original[name]),
                     f"Running producer {name} differs from catalog")
        _require(_normalized_metadata(yaml.safe_load(producer["metadata_yaml"])) == canonical,
                 "Running producer X-engine metadata differs from canonical input")
        variances = np.repeat(np.asarray(canonical["noise_variance"], dtype="<f8"),
                              canonical["zone_nfreq"])
        variance_hash = hashlib.sha256(variances.tobytes()).hexdigest()
        _require(producer.get("channel_variance_sha256") == variance_hash
                 and producer.get("channel_variance_count") == len(variances)
                 and producer.get("channel_variance_dtype") == "<f8",
                 "Running producer analytic-weight input variances differ from canonical input")
        _require(producer.get("weight_values_read_back") is False,
                 "Weight evidence must distinguish initialization from GPU readback")
        result.update(channel_variance_sha256=variance_hash,
                      weights_evidence=producer.get("weights_evidence"))
    return result


def _markdown(report):
    lines = ["# Controlled CHORD online/offline validation", "",
             f"Status: **{report['status'].upper()}**", "",
             "This experiment checks scientific agreement and timely raw capture; it is not a throughput benchmark.", ""]
    if report.get("error"):
        lines += ["Failure: " + report["error"], ""]
    for mode, value in report.get("catalog_comparisons", {}).items():
        lines += [f"- {mode}: {value['events']} grouping partitions and {value['members']} candidate memberships match offline."]
    lines += ["", "Recovered measurements (arrival times referenced to 300 MHz):", "",
              "| Run | Burst | Tree / early level | DM (pc cm^-3) | TOA (s) | TOA error (ms) | S/N | Nominal filter width (ms) |",
              "|---|---|---:|---:|---:|---:|---:|---:|"]
    for mode, recovery in report.get("recovery", {}).items():
        for burst in recovery:
            for detection in burst["detections"]:
                lines.append(f"| {mode} | {burst['burst_id']} | {detection['tree']} / {detection['early_trigger_level']} | "
                             f"{detection['dm']:.6f} | {detection['toa_seconds']:.9f} | "
                             f"{detection['toa_error_seconds']*1000:+.3f} | {detection['snr']:.4f} | "
                             f"{detection['nominal_filter_width_ms']:.4f} |")
    lines += ["", "Online timing, measured against the replay's observing clock:", "",
              "| Run | Max enqueue lag (ms) | Max completed dispatch lag (ms) | Max assembled backlog (chunks) | Maximum observed unreaped retention (s) |",
              "|---|---:|---:|---:|---:|"]
    for mode, timing in report.get("timing", {}).items():
        lines.append(f"| {mode} | {timing['max_enqueue_lag_s']*1000:.3f} | "
                     f"{timing['max_chunk_dispatch_lag_s']*1000:.3f} | "
                     f"{timing['max_observed_backlog_chunks']:.3f} | "
                     f"{timing['max_observed_unreaped_retention_s']:.3f} |")
    lines += ["", "| Run | Burst / tree | Grouped at replay time (s) | Delay after tree arrival (s) | Lead before 300 MHz arrival (s) |",
              "|---|---|---:|---:|---:|"]
    for mode, timing in report.get("detection_timing", {}).items():
        for row in timing.get("detections", []):
            if row.get("available"):
                lines.append(f"| {mode} | {row['burst_id']} / {row['tree']} | "
                             f"{row['grouped_replay_elapsed_seconds']:.6f} | "
                             f"{row['grouping_after_tree_arrival_seconds']:.6f} | "
                             f"{row['lead_to_300MHz_arrival_seconds']:.6f} |")
    lines += ["", "A positive lead means the grouped detection was available before arrival at 300 MHz. "
              "These delays include chunk assembly and grouping lookahead. Retention grows during startup; "
              "the maximum above is observational evidence, not a guaranteed memory capacity.", ""]
    lines += ["", "| Run | Burst | Capture complete | Required chunks | Saved chunks | First trigger level |",
              "|---|---|---:|---:|---:|---:|"]
    for mode, capture in report.get("capture_checks", {}).items():
        for burst in capture["bursts"]:
            lines.append(f"| {mode} | {burst['burst_id']} | {burst['complete']} | {burst['needed_chunks']} | {burst['saved_chunks']} | {burst['first_early_trigger_level']} |")
    lines += ["", "Raw-data expiry evidence at the first capture request:", "",
              "| Run | Burst | Earliest required frame ID | Live frame lower bound | Leading-frame retention margin (chunks) | First request (receiver-relative s) |",
              "|---|---|---:|---:|---:|---:|"]
    for mode, capture in report.get("capture_checks", {}).items():
        for burst in capture["bursts"]:
            lines.append(f"| {mode} | {burst['burst_id']} | {burst['earliest_required_frame_id']} | "
                         f"{burst['live_frame_lower_bound_at_request']} | "
                         f"{burst['leading_frame_retention_margin_chunks']:+.3f} | "
                         f"{burst['first_request_receiver_elapsed_seconds']:.6f} |")
    lines += ["", "A negative retention margin puts the leading required raw frame before the live buffer boundary. "
              "Capture success above requires successful write notifications and byte-identical packed samples/scales "
              "in the files actually saved. Receiver-relative request times use the capture receiver's own startup origin; "
              "they must not be directly subtracted from replay-relative times.", ""]
    lines += ["", "Source cells, tokens, grouping membership and representatives must match exactly. S/N allows one float16 relative ULP (rtol 0.001, atol 0.002); decoded timing allows 1e-6 input samples. Full tolerances and provenance are in the JSON report.", "",
              "Injected width is Gaussian sigma; recovered width is a nominal filter width. They are reported separately.", ""]
    return "\n".join(lines)



def _validate_terminal_session(directory, run):
    """Validate archived component acknowledgments without probing old PIDs."""
    from .ControlledTerminals import ROLES, _configuration_digest

    manifest = _json(directory / "session.json")
    _require(type(manifest.get("version")) is int and manifest["version"] == 1,
             "Invalid terminal-session version")
    digest = _configuration_digest(manifest)
    _require(manifest.get("configuration_sha256") == digest,
             "Terminal-session configuration changed")
    _require(manifest["session_id"] == run.get("session_id"),
             "Run belongs to a different terminal session")
    _require(manifest["bundle_manifest_sha256"] == run["bundle_manifest_sha256"],
             "Terminal session used a different observation")
    _require(manifest["source"]["module_sha256"] == run["source"]["module_sha256"],
             "Terminal-session source differs from executed source")
    _require(not (directory / "_session" / "stop.json").exists(),
             "Terminal session was explicitly stopped")
    hashes = {}
    for role in ROLES:
        path = directory / "_session" / f"{role}.json"
        status = _json(path)
        _require(status.get("session_id") == manifest["session_id"]
                 and status.get("configuration_sha256") == digest and status.get("role") == role,
                 f"{role}: component acknowledgment belongs to another session")
        _require(status.get("state") == "complete" and status.get("exitcode") == 0,
                 f"{role}: component did not acknowledge clean completion")
        if role != "server":
            _require(run.get("component_status", {}).get(role) == status,
                     f"{role}: final run acknowledgment differs from component record")
        hashes[role] = _hash(path)
    return dict(complete=True, configuration_sha256=digest,
                session_manifest_sha256=_hash(directory / "session.json"), component_status_sha256=hashes)

def compare_controlled_experiment(bundle_dir, offline_dir, online_early_dir,
                                  online_full_dir, output_json, output_markdown=None):
    """Validate all modes, persist evidence, and raise on any mismatch."""
    report = dict(schema_version=1, status="running", catalog_comparisons={},
                  recovery={}, capture_checks={}, timing={}, detection_timing={}, execution_provenance={},
                  numerical_tolerances={k: dict(rtol=v[0], atol=v[1]) for k, v in FLOAT_TOLERANCES.items()},
                  performance_benchmark=False)
    output_markdown = Path(output_markdown) if output_markdown else Path(output_json).with_suffix(".md")
    try:
        from .ControlledObservation import load_experiment_bundle
        bundle = load_experiment_bundle(bundle_dir, verify_hashes=True, require_complete=True)
        root = Path(bundle_dir)
        truth_path = root / "injections.json"
        truth = _json(truth_path)
        expected_hash = _json(root / "bundle.json")["artifact_hashes"]["injections.json"]
        _require(_hash(truth_path) == expected_hash, "Injection truth hash differs from saved bundle")
        bursts = truth["bursts"]
        _require(len(bursts) == 2, "Expected exactly two injected bursts")
        policy = yaml.safe_load(Path(bundle["capture_config_path"]).read_text())
        experiment = yaml.safe_load((root / "experiment.yml").read_text())
        max_lag = float(experiment["observation"]["latency_budget_seconds"])
        report["bundle_sha256"] = bundle["manifest_sha256"]
        report["truth_sha256"] = expected_hash
        paths = dict(offline=Path(offline_dir), online_early=Path(online_early_dir), online_full=Path(online_full_dir))
        catalogs = {}
        for mode, directory in paths.items():
            catalog = read_catalog(directory / "events.asdf")
            _validate_catalog_completion(catalog, bundle, "offline" if mode == "offline" else "online")
            catalogs[mode] = catalog
            run = _json(directory / "run.json")
            report["execution_provenance"][mode] = _validate_run(run,
                catalog, bundle, online=mode != "offline")
            if run.get("launch_mode") == "separate_terminals":
                report["execution_provenance"][mode]["terminal_session"] = _validate_terminal_session(directory, run)
            report["recovery"][mode] = validate_recovery(catalog, bursts, bundle["time_sample_ms"]*.001)
        signatures = [value["source"]["module_sha256"] for value in report["execution_provenance"].values()]
        _require(all(value == signatures[0] for value in signatures[1:]),
                 "Source modules changed between compared runs")
        for mode in ("online_early", "online_full"):
            report["catalog_comparisons"][mode] = compare_catalogs(catalogs["offline"], catalogs[mode], label=mode)
            replay = _json(paths[mode] / "replay.json")
            report["timing"][mode] = _validate_replay(replay, bundle, max_lag)
            timing_path = paths[mode] / "events.asdf.online.json"
            if timing_path.is_file():
                report["detection_timing"][mode] = detection_timing(
                    catalogs[mode], report["recovery"][mode], bursts, replay, _json(timing_path))
            else:
                report["detection_timing"][mode] = dict(available=False,
                    reason="online grouper timing sidecar was not provided")
        source_cache = {}
        for mode, accept in (("online_early", True), ("online_full", False)):
            report["capture_checks"][mode] = validate_captures(_json(paths[mode] / "capture.json"),
                bundle, paths[mode], bursts, policy, accept_early=accept, source_cache=source_cache)
        report["producer_science"] = _metadata_science(catalogs["offline"]["metadata"])
        report["artifact_sha256"] = {mode: {name: _hash(directory / name)
            for name in ("events.asdf", "events.asdf.online.json", "run.json", "capture.json", "replay.json")
            if (directory / name).is_file()} for mode, directory in paths.items()}
        report["status"] = "pass"
        return report
    except BaseException as exc:
        report.update(status="fail", error=f"{type(exc).__name__}: {exc}")
        raise
    finally:
        _save_result(output_json, report)
        output_markdown.parent.mkdir(parents=True, exist_ok=True)
        output_markdown.write_text(_markdown(report))
