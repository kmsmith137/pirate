"""Host-only schema and CLI tests for offline-grouper configuration."""

import copy
import tempfile
from pathlib import Path

from ..OfflineGrouperConfig import (
    OfflineGrouperConfig,
    OfflineGrouperConfigError,
)


def _valid_mapping():
    return {
        "peakfinding": {
            "snr_threshold": 10.0,
            "dm_reach": 8,
            "waist_bins": 1,
        },
        "grouping": {
            "halo_size": 2,
            "dm_tolerance": 1.5,
            "time_tolerance": 1.5,
        },
        "execution": {
            "beam_batch_size": 4,
            "timeout_ms": 0,
            "timeout_policy": "discard",
        },
    }


def _expect_invalid(value, expected_text):
    try:
        OfflineGrouperConfig.from_mapping(value)
    except OfflineGrouperConfigError as exc:
        assert expected_text in str(exc)
    else:
        raise AssertionError(f"invalid configuration was accepted: {value!r}")


def test_offline_grouper_config_accepts_exact_schema():
    """Every public value is typed and exposed through immutable sections."""

    config = OfflineGrouperConfig.from_mapping(_valid_mapping())
    assert config.peakfinding.snr_threshold == 10.0
    assert config.peakfinding.dm_reach == 8
    assert config.peakfinding.waist_bins == 1
    assert config.grouping.halo_size == 2
    assert config.grouping.dm_tolerance == 1.5
    assert config.grouping.time_tolerance == 1.5
    assert config.execution.beam_batch_size == 4
    assert config.execution.timeout_ms == 0
    assert config.execution.timeout_policy == "discard"

    alternate = copy.deepcopy(_valid_mapping())
    alternate["peakfinding"]["snr_threshold"] = -3
    alternate["peakfinding"]["dm_reach"] = 0
    alternate["peakfinding"]["waist_bins"] = 0
    alternate["grouping"]["dm_tolerance"] = 0
    alternate["grouping"]["time_tolerance"] = 0
    alternate["execution"]["timeout_policy"] = "emit_partial"
    parsed = OfflineGrouperConfig.from_mapping(alternate)
    assert parsed.peakfinding.snr_threshold == -3.0
    assert parsed.execution.timeout_policy == "emit_partial"


def test_offline_grouper_config_rejects_schema_drift_and_bad_values():
    """Missing/unknown fields, booleans, nonfinite values, and ranges fail."""

    missing_section = _valid_mapping()
    del missing_section["execution"]
    _expect_invalid(missing_section, "missing required key(s): execution")

    unknown_section = _valid_mapping()
    unknown_section["experimental"] = {}
    _expect_invalid(unknown_section, "unknown key(s): 'experimental'")

    missing_field = _valid_mapping()
    del missing_field["grouping"]["time_tolerance"]
    _expect_invalid(missing_field, "missing required key(s): time_tolerance")

    unknown_field = _valid_mapping()
    unknown_field["peakfinding"]["method"] = "bank"
    _expect_invalid(unknown_field, "unknown key(s): 'method'")

    cases = (
        (("peakfinding", "snr_threshold"), True, "finite number"),
        (("peakfinding", "snr_threshold"), float("inf"), "finite number"),
        (("peakfinding", "dm_reach"), True, "must be an integer"),
        (("peakfinding", "dm_reach"), 1.0, "must be an integer"),
        (("peakfinding", "waist_bins"), -1, "must be at least 0"),
        (("grouping", "halo_size"), 1, "must be at least 2"),
        (("grouping", "dm_tolerance"), -0.1, "nonnegative"),
        (("grouping", "time_tolerance"), False, "finite number"),
        (("execution", "beam_batch_size"), 0, "must be positive"),
        (("execution", "timeout_ms"), -1, "must be at least 0"),
        (("execution", "timeout_ms"), False, "must be an integer"),
        (("execution", "timeout_policy"), "partial", "exactly"),
    )
    for (section, field), bad_value, message in cases:
        value = _valid_mapping()
        value[section][field] = bad_value
        _expect_invalid(value, message)


def test_offline_grouper_config_uses_safe_strict_yaml():
    """Files load normally while unsafe tags and duplicate keys are rejected."""

    valid_yaml = """\
peakfinding:
  snr_threshold: 12.5
  dm_reach: 4
  waist_bins: 0
grouping:
  halo_size: 3
  dm_tolerance: 2
  time_tolerance: 0.5
execution:
  beam_batch_size: 2
  timeout_ms: 250
  timeout_policy: emit_partial
"""
    with tempfile.TemporaryDirectory(prefix="pirate-grouper-config-") as tmp:
        filename = Path(tmp) / "config.yml"
        filename.write_text(valid_yaml, encoding="utf-8")
        config = OfflineGrouperConfig.from_yaml(filename)
        assert config.grouping.halo_size == 3
        assert config.execution.timeout_ms == 250

        filename.write_text(
            valid_yaml.replace(
                "  dm_reach: 4", "  dm_reach: 4\n  dm_reach: 5"
            ),
            encoding="utf-8",
        )
        try:
            OfflineGrouperConfig.from_yaml(filename)
        except OfflineGrouperConfigError as exc:
            assert "duplicate configuration key 'dm_reach'" in str(exc)
        else:
            raise AssertionError("duplicate YAML key was accepted")

        filename.write_text(
            "!!python/object/apply:os.system ['echo unsafe']\n",
            encoding="utf-8",
        )
        try:
            OfflineGrouperConfig.from_yaml(filename)
        except OfflineGrouperConfigError as exc:
            assert "invalid offline-grouper YAML" in str(exc)
        else:
            raise AssertionError("unsafe YAML constructor was accepted")


def test_offline_grouper_cli_has_only_operational_overrides():
    """Scientific and execution controls come only from the YAML file."""

    from ..__main__ import get_parser, run_offline_grouper_command

    parser = get_parser()
    args = parser.parse_args((
        "run", "offline_grouper",
        "/data/acquisition",
        "offline.yml",
        "--device", "3",
        "--output", "events.asdf",
        "--max-chunks", "7",
        "--verbose",
        "--assume-steady-state",
    ))
    assert args.command == "run"
    assert args.run_command == "offline_grouper"
    assert args.func is run_offline_grouper_command
    assert args.acqdir == "/data/acquisition"
    assert args.config_file == "offline.yml"
    assert args.device == 3
    assert args.output == "events.asdf"
    assert args.max_chunks == 7
    assert args.verbose
    assert args.assume_steady_state

    subparser_action = next(
        action for action in parser._actions
        if action.__class__.__name__ == "_SubParsersAction"
    )
    run_subparsers = next(
        action for action in subparser_action.choices["run"]._actions
        if action.__class__.__name__ == "_SubParsersAction"
    )
    help_text = run_subparsers.choices["offline_grouper"].format_help()
    for removed in (
        "--snr-threshold",
        "--beam-batch-size",
        "--peakfinder",
        "--dm-tolerance-bins",
        "--time-padding-bins",
        "--dm-reach",
        "--waist-bins",
        "--timeout-ms",
        "--timeout-policy",
    ):
        assert removed not in help_text
    for retained in (
        "--device",
        "--output",
        "--max-chunks",
        "--verbose",
        "--assume-steady-state",
    ):
        assert retained in help_text
    assert "ACQDIR" in help_text and "CONFIG.yml" in help_text
