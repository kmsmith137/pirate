"""Independent source identities and capture evidence, without a live server."""
from copy import deepcopy
from pathlib import Path
import re

import numpy as np
import pytest
import yaml

from pirate_frb.ControlledComparison import (
    compare_catalogs, validate_captures, _truth_tolerances, _validate_replay,
    _validate_run, compare_controlled_experiment, _normalized_metadata, detection_timing,
)
from pirate_frb.TriggerCatalog import EVENT_DTYPES, MEMBER_DTYPES


def catalog():
    members = {key: np.zeros(3, dtype=dtype) for key, dtype in MEMBER_DTYPES.items()}
    for key, values in dict(candidate_id=[0, 1, 2], event_id=[0, 0, 1],
        grouping_window_id=[0, 0, 1], source_chunk_index=[7, 7, 8],
        beam_id=[100, 100, 101], tree=[0, 0, 1], primary_tree_index=[0, 0, 1],
        idm=[10, 11, 20], itime=[4, 4, 5], argmax_token=[0x300, 0x300, 0x400],
        snr=[30, 20, 40], dm=[100, 100.1, 2000], toa_sample_abs=[45, 45.001, 200],
        width_samp=[3, 3, 6], width_ms=[3, 3, 6], freq_lo_MHz=[300, 300, 416],
        freq_hi_MHz=[1500]*3, dm_step=[1]*3, time_step_samples=[8]*3).items():
        members[key][:] = values
    events = {key: np.zeros(2, dtype=dtype) for key, dtype in EVENT_DTYPES.items()}
    for key in events:
        if key in members:
            events[key][:] = members[key][[0, 2]]
    events["event_id"][:] = [0, 1]
    events["representative_candidate_id"][:] = [0, 2]
    events["member_count"][:] = [2, 1]
    metadata = dict(pipeline="offline", units={"dm": "pc cm^-3"},
        processing=dict(complete=True, beam_batch_size=1, snr_threshold=10),
        startup_by_beam=[dict(beam_id=100, status="authoritative", producer_start_chunk_index=0)],
        grouping_windows=[dict(grouping_window_id=0, owner_source_chunk_index=7),
                          dict(grouping_window_id=1, owner_source_chunk_index=8)],
        producer=dict(config_yaml=yaml.safe_dump(dict(dtype="float16", beams_per_gpu=1,
            beams_per_batch=1, num_active_batches=1, zone_freq_edges=[300, 1500])),
            plan_yaml=yaml.safe_dump(dict(nt_in=2048, nfreq=28160, ntrees=2,
                trees=[dict(tree_index=0, ndm_out=100), dict(tree_index=1, ndm_out=50)])),
            dcores=[8, 8], argmax_encoding="pirate-1.5:t8-p8-m8-mu8"))
    return dict(events=events, members=members, metadata=metadata,
                coverage=dict(beam_id=np.array([100, 101]), source_chunk_index=np.array([7, 8])))


def test_catalog_equality_ignores_ids_and_traversal_but_preserves_partitions():
    reference = catalog()
    actual = deepcopy(reference)
    actual["metadata"]["pipeline"] = "online"
    config = yaml.safe_load(actual["metadata"]["producer"]["config_yaml"])
    config["beams_per_gpu"] = 2
    actual["metadata"]["producer"]["config_yaml"] = yaml.safe_dump(config)
    actual["metadata"]["processing"]["beam_batch_size"] = 2
    # Renumber candidate IDs, reverse event traversal, permute member storage,
    # and renumber grouping windows independently of source-cell identities.
    actual["members"]["candidate_id"] = 2-actual["members"]["candidate_id"]
    actual["events"]["representative_candidate_id"] = 2-actual["events"]["representative_candidate_id"]
    actual["events"] = {key: value[::-1].copy() for key, value in actual["events"].items()}
    actual["events"]["event_id"][:] = [0, 1]
    actual["members"]["event_id"] = 1-actual["members"]["event_id"]
    actual["members"] = {key: value[[2, 0, 1]].copy() for key, value in actual["members"].items()}
    for table in (actual["events"], actual["members"]):
        table["grouping_window_id"] = 1-table["grouping_window_id"]
    for window in actual["metadata"]["grouping_windows"]:
        window["grouping_window_id"] = 1-window["grouping_window_id"]
    result = compare_catalogs(reference, actual)
    assert result["events"] == 2
    assert result["members"] == 3
    assert max(result["max_absolute_differences"].values()) == 0


@pytest.mark.parametrize("mutation, message", [
    (lambda c: c["members"]["argmax_token"].__setitem__(0, 5), "source candidates differ"),
    (lambda c: c["members"]["event_id"].__setitem__(1, 1), "grouping partitions differ"),
    (lambda c: c["metadata"]["producer"]["dcores"].__setitem__(0, 4), "metadata differs"),
    (lambda c: c["events"]["dm"].__setitem__(0, 101), "event.dm"),
])
def test_catalog_comparison_rejects_scientific_mismatches(mutation, message):
    expected = catalog()
    actual = deepcopy(expected)
    mutation(actual)
    with pytest.raises(ValueError, match=message):
        compare_catalogs(expected, actual)


def test_float16_snr_tolerance_does_not_relax_source_or_decoded_geometry():
    expected = catalog()
    actual = deepcopy(expected)
    actual["events"]["snr"] += .005
    actual["members"]["snr"] += .005
    assert compare_catalogs(expected, actual)["max_absolute_differences"]["snr"] == pytest.approx(.005)
    actual["members"]["toa_sample_abs"][0] += .001
    with pytest.raises(ValueError, match="toa_sample_abs"):
        compare_catalogs(expected, actual)


def test_truth_tolerance_uses_trial_quantization_and_dm_extrapolation():
    member = dict(primary_tree_index=1, freq_lo_MHz=416.0,
                  freq_hi_MHz=1500.0, argmax_token=0x400)
    dm, toa = _truth_tolerances(member, {}, 8, .001)
    assert 0 < dm < .3
    assert 0 < toa < .02
    full = dict(member, freq_lo_MHz=300.0)
    full_dm, full_toa = _truth_tolerances(full, {}, 8, .001)
    assert full_dm < dm
    assert full_toa < toa


def capture_fixture(tmp_path, accept):
    bundle_root = tmp_path / "bundle"
    run_root = tmp_path / "run"
    entries = []
    for beam in (100, 101):
        for chunk in range(8):
            path = bundle_root / f"acq/frame_b{beam}_t{chunk}.asdf"
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_bytes(b"fixture")
            entries.append(dict(beam_id=beam, time_chunk_index=chunk,
                                path=str(path.relative_to(bundle_root))))
    bundle = dict(bundle_dir=str(bundle_root), beam_ids=[100, 101], frame_entries=entries,
                  samples_per_chunk=16, metadata=dict(dt_ns_per_seq=1000000, seq_per_frb_time_sample=1))
    policy = dict(association_dm_tolerance=5, association_toa_tolerance_seconds=.1)
    bursts = [dict(id="low", role="full_band", beam_id=100, dm=100, toa_seconds=45,
                   sample_start=0, sample_end=16),
              dict(id="high", role="early_trigger", beam_id=101, dm=2000, toa_seconds=200,
                   sample_start=32, sample_end=96)]
    ledger = dict(complete=True, errors=[], classifier_mode="bypass", accept_early=accept,
                  captures=[], notifications={}, decisions=[])
    for index, burst in enumerate(bursts):
        first, last = (0, 0) if index == 0 else (2, 5)
        expected = [f"event_{index:06d}/frame_b{burst['beam_id']}_t{c}.asdf" for c in range(first, last+1)]
        promised = expected if accept or index == 0 else expected[2:]
        for name in promised:
            path = run_root / "captures" / name
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_bytes(b"fixture")
            ledger["notifications"][name] = dict(error="", stream_name="")
        request = dict(start_seq=first*16, end_seq=(last+1)*16,
            expected_files=expected, promised_files=promised,
            missing_files=sorted(set(expected)-set(promised)), requested_elapsed_seconds=index*100,
            status_before=dict(rb_start=0 if accept or index == 0 else 8,
                               rb_reaped=0 if accept or index == 0 else 8))
        ledger["captures"].append(dict(capture_id=index, acqdir=f"event_{index:06d}",
            first_event=dict(beam_id=burst["beam_id"], dm=burst["dm"],
                             fpga_timestamp=int(burst["toa_seconds"]/.001)),
            first_early_trigger_level=1 if accept and index else 0, requests=[request]))
    if not accept:
        ledger["decisions"].append(dict(action="early_capture_suppressed"))
    def signature(path):
        beam, chunk = re.search(r"frame_b(\d+)_t(\d+)", str(path)).groups()
        return dict(beam_id=int(beam), time_chunk_index=int(chunk), data="same", scales="same")
    return ledger, bundle, run_root, bursts, policy, signature


@pytest.mark.parametrize("accept", [True, False])
def test_capture_checks_actual_files_notifications_and_expected_early_benefit(tmp_path, accept):
    ledger, bundle, run_root, bursts, policy, signature = capture_fixture(tmp_path, accept)
    result = validate_captures(ledger, bundle, run_root, bursts, policy,
                               accept_early=accept, signature_reader=signature)
    assert result["bursts"][0]["complete"] is True
    assert result["bursts"][1]["complete"] is accept
    assert result["bursts"][1]["saved_chunks"] == (4 if accept else 2)


def test_capture_fails_if_only_promised_or_packed_payload_differs(tmp_path):
    ledger, bundle, run_root, bursts, policy, signature = capture_fixture(tmp_path, True)
    name = next(iter(ledger["notifications"]))
    del ledger["notifications"][name]
    with pytest.raises(ValueError, match="completion notification"):
        validate_captures(ledger, bundle, run_root, bursts, policy,
                          accept_early=True, signature_reader=signature)
    ledger["notifications"][name] = dict(error="")
    def corrupt(path):
        result = signature(path)
        if "captures" in Path(path).parts:
            result["data"] = "corrupted"
        return result
    with pytest.raises(ValueError, match="packed data"):
        validate_captures(ledger, bundle, run_root, bursts, policy,
                          accept_early=True, signature_reader=corrupt)


def test_fast_replay_cannot_pass_real_time_acceptance():
    with pytest.raises(ValueError, match="1x observing rate"):
        _validate_replay(dict(state="sender_complete", rate=2), {}, 2)


def test_run_requires_explicit_analytic_weights_and_same_input_variances():
    bundle = dict(manifest_sha256="bundle", metadata=dict(noise_variance=[1., 2.], zone_nfreq=[1, 1]))
    run = dict(state="complete", complete=True, mode="offline",
        bundle_manifest_sha256="bundle", weights_mode="analytic", detrending=False,
        shared_processing="SharedGrouper", noise_variance=[1., 2.],
        source=dict(module_sha256={"SharedGrouper.py": "digest"}))
    assert _validate_run(run, {}, bundle, online=False)["weight_values_read_back"] is False
    run["noise_variance"] = [1., 1.]
    with pytest.raises(ValueError, match="noise variances differ"):
        _validate_run(run, {}, bundle, online=False)


def test_failed_comparison_still_writes_machine_and_human_readable_evidence(tmp_path):
    import json
    report = tmp_path / "comparison.json"
    with pytest.raises(FileNotFoundError):
        compare_controlled_experiment(tmp_path / "missing", "offline", "early", "full", report)
    evidence = json.loads(report.read_text())
    assert evidence["status"] == "fail"
    assert "FileNotFoundError" in evidence["error"]
    assert evidence["performance_benchmark"] is False
    assert "FAIL" in report.with_suffix(".md").read_text()


def test_native_metadata_scalar_noise_and_omitted_empty_channels_are_equivalent():
    canonical = dict(zone_nfreq=[3, 2], noise_variance=1.0, freq_channels=[],
                     zone_freq_edges=[300, 600, 1500])
    native = dict(zone_nfreq=[3, 2], noise_variance=[1.0, 1.0],
                  zone_freq_edges=[300, 600, 1500])
    assert _normalized_metadata(native) == _normalized_metadata(canonical)
    assert canonical["noise_variance"] == 1.0
    native["noise_variance"] = [1.0, 2.0]
    assert _normalized_metadata(native) != _normalized_metadata(canonical)


def test_detection_timing_joins_window_ids_on_shared_monotonic_clock():
    cat = catalog()
    recovery = [
        dict(burst_id="low", detections=[dict(tree=0, early_trigger_level=0, event_id=0)]),
        dict(burst_id="high", detections=[dict(tree=1, early_trigger_level=1, event_id=1)]),
    ]
    truth = [dict(id="low", toa_seconds=45, expected_tree_arrivals=[dict(tree_index=0, arrival_seconds=45)]),
             dict(id="high", toa_seconds=200, expected_tree_arrivals=[dict(tree_index=1, arrival_seconds=155)])]
    replay = dict(clock_origin_monotonic_s=100, initial_chunk=0, samples_per_chunk=2048,
                  time_sample_s=.001)
    online = dict(complete=True, receiver_assembly_lookahead_chunks=2, grouping_lookahead_chunks=1,
        windows=[
            dict(grouping_window_id=1, owner_source_chunk_index=8, available_source_chunk_index=9,
                 grouped_monotonic_ns=256_000_000_000, output_available_monotonic_ns=255_500_000_000),
            dict(grouping_window_id=0, owner_source_chunk_index=7, available_source_chunk_index=8,
                 grouped_monotonic_ns=146_000_000_000, output_available_monotonic_ns=145_500_000_000),
        ])
    result = detection_timing(cat, recovery, truth, replay, online)
    high = result["detections"][1]
    assert high["grouped_replay_elapsed_seconds"] == pytest.approx(156)
    assert high["grouping_after_tree_arrival_seconds"] == pytest.approx(1)
    assert high["lead_to_300MHz_arrival_seconds"] == pytest.approx(44)
    assert result["detections"][0]["lead_to_300MHz_arrival_seconds"] == pytest.approx(-1)
    assert result["time_reference"] == "seconds since replay monotonic observation origin"
