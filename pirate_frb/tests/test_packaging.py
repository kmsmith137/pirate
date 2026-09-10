"""Distribution-contract checks for the offline peak-extraction modules.

These tests use only the Python standard library so they can run through
PIRATE's installed-package test runner without pulling pytest into production.
The Makefile declarations are the authoritative inputs to both generated file
manifests; built archives are inspected too whenever they exist locally.
"""

import importlib
import os
from pathlib import Path
import tarfile
import zipfile


_REQUIRED_PYFILES = (
    "pirate_frb/__main__.py",
    "pirate_frb/Peakfinders.py",
    "pirate_frb/OfflineGrouperConfig.py",
    "pirate_frb/ArgmaxMetadata.py",
    "pirate_frb/GpuArgmaxDecoder.py",
    "pirate_frb/OfflineCandidateGrouper.py",
    "pirate_frb/FrbOfflineGrouper.py",
    "pirate_frb/TriggerCatalog.py",
    "pirate_frb/run_offline_grouper.py",
    "pirate_frb/SharedGrouper.py",
    "pirate_frb/OnlineGrouper.py",
    "pirate_frb/ControlledObservation.py",
    "pirate_frb/ReplayObservation.py",
    "pirate_frb/ControlledCapture.py",
    "pirate_frb/ControlledExperiment.py",
    "pirate_frb/ControlledComparison.py",
    "pirate_frb/tests/__init__.py",
    "pirate_frb/tests/test_peakfinders_stream.py",
    "pirate_frb/tests/test_gpu_argmax_decoder.py",
    "pirate_frb/tests/test_offline_candidate_grouper.py",
    "pirate_frb/tests/test_offline_grouper.py",
    "pirate_frb/tests/test_offline_grouper_streaming.py",
    "pirate_frb/tests/test_offline_grouper_config.py",
    "pirate_frb/tests/test_packaging.py",
    "pirate_frb/tests/test_trigger_catalog.py",
)
_REQUIRED_DATA_FILES = (
    "pirate_frb/tests/data/toy.yml",
    "pirate_frb/tests/data/chord_sb2_et.yml",
)
_REQUIRED_DISTRIBUTION_FILES = _REQUIRED_PYFILES + _REQUIRED_DATA_FILES
_REQUIRED_SDIST_ONLY_FILES = (
    "configs/offline_grouper/example.yml",
    "configs/experiments/chord_replay.yml",
    "configs/dedispersion/chord_sb2_et.yml",
    "configs/xengine_metadata.yml",
    "notes/controlled_chord_experiment.md",
)
_FORBIDDEN_DISTRIBUTION_FILES = (
    "pirate_frb/BowtieBank.py",
)


def _archive_members(filename):
    """Return normalized member names from a wheel or source tar archive."""
    if filename.suffix == ".whl":
        with zipfile.ZipFile(filename) as archive:
            return tuple(archive.namelist())
    with tarfile.open(filename, mode="r:*") as archive:
        return tuple(member.name for member in archive.getmembers())


def test_offline_peak_modules_packaged():
    """Fail when a required module falls out of PYFILES or a built archive."""
    repository = Path(__file__).resolve().parents[2]
    package = Path(__file__).resolve().parents[1]
    makefile = repository / "Makefile"

    # This check also runs from an isolated wheel installation, where neither
    # the Makefile nor a local dist/ directory exists.  Resolve every required
    # path relative to the installed ``pirate_frb`` directory so omitting either
    # production module from a future wheel fails the installed custom runner.
    for required in _REQUIRED_PYFILES:
        relative = Path(required).relative_to("pirate_frb")
        assert (package / relative).is_file(), (
            f"installed pirate_frb package omits {required}"
        )
    for forbidden in _FORBIDDEN_DISTRIBUTION_FILES:
        relative = Path(forbidden).relative_to("pirate_frb")
        assert not (package / relative).exists(), (
            f"installed pirate_frb package retains removed {forbidden}"
        )
    # The pytest-free decoder/integration tests reconstruct producer plans from
    # two package-local wheel data files.  ``repository`` is the source root
    # in-tree and the isolated installation target when installed.
    for required in _REQUIRED_DATA_FILES:
        assert (repository / required).is_file(), (
            f"installed offline milestone tests omit fixture {required}"
        )

    # In a checkout, pin the installed copies byte-for-byte to the authoritative
    # repository fixtures so package data cannot silently drift.  The source
    # config tree is intentionally absent from an isolated wheel target.
    source_config = repository / "configs" / "dedispersion"
    if source_config.is_dir():
        for required in _REQUIRED_DATA_FILES:
            packaged = repository / required
            authoritative = source_config / packaged.name
            assert packaged.read_bytes() == authoritative.read_bytes(), (
                f"packaged fixture {required} differs from {authoritative}"
            )

    # Installed wheels do not contain their source Makefile.  In a source tree,
    # however, checking PYFILES catches omissions before stale generated lists or
    # an old wheel can hide the packaging error.
    if makefile.exists():
        declarations = makefile.read_text(encoding="utf-8")
        for required in _REQUIRED_SDIST_ONLY_FILES:
            assert (repository / required).is_file(), (
                f"source tree omits {required}"
            )
            assert required in declarations, (
                f"{required} is absent from Makefile source-distribution inputs"
            )
        pyfiles = declarations.split("PYFILES =", 1)[1].split(
            "CUDAGEN_PYFILES =", 1
        )[0]
        for required in _REQUIRED_PYFILES:
            assert required in pyfiles, f"{required} is absent from Makefile PYFILES"
        for forbidden in _FORBIDDEN_DISTRIBUTION_FILES:
            assert forbidden not in pyfiles, (
                f"removed {forbidden} remains in Makefile PYFILES"
            )

        # The manifests are generated by Makefile targets.  If present, they must
        # agree with PYFILES; a future stale or hand-edited list fails explicitly.
        for manifest_name in ("wheel_files.txt", "sdist_files.txt"):
            manifest = repository / manifest_name
            if not manifest.exists():
                continue
            entries = set(manifest.read_text(encoding="utf-8").splitlines())
            required = set(_REQUIRED_DISTRIBUTION_FILES)
            if manifest_name == "sdist_files.txt":
                required.update(_REQUIRED_SDIST_ONLY_FILES)
            missing = required - entries
            assert not missing, f"{manifest_name} omits {sorted(missing)}"
            forbidden = set(_FORBIDDEN_DISTRIBUTION_FILES) & entries
            assert not forbidden, (
                f"{manifest_name} retains removed {sorted(forbidden)}"
            )

        # Archive inspection is conditional so ordinary source tests need not
        # build distributions.  Packaging verification builds both first, making
        # this branch mandatory in the release-style correction-pass check.
        dist = repository / "dist"
        wheels = sorted(dist.glob("pirate_frb-*.whl"))
        sdists = sorted(dist.glob("pirate_frb-*.tar.gz"))
        require_built = os.environ.get("PIRATE_REQUIRE_BUILT_DISTS") == "1"
        if require_built:
            assert wheels, "distribution verification requires a built wheel"
            assert sdists, "distribution verification requires a built sdist"

        # Inspect only freshly built artifacts during an ordinary source-tree
        # run.  A dirty development checkout may contain an older release
        # artifact whose contents necessarily predate the current manifest;
        # release verification sets PIRATE_REQUIRE_BUILT_DISTS after rebuilding
        # and therefore always exercises both newest archives.
        archives = wheels[-1:] + sdists[-1:]
        if not require_built:
            manifest_mtime = max(
                (repository / name).stat().st_mtime
                for name in ("wheel_files.txt", "sdist_files.txt")
            )
            archives = [
                archive for archive in archives
                if archive.stat().st_mtime >= manifest_mtime
            ]
        for archive in archives:
            members = _archive_members(archive)
            required_files = _REQUIRED_DISTRIBUTION_FILES
            if archive.suffixes[-2:] == [".tar", ".gz"]:
                required_files += _REQUIRED_SDIST_ONLY_FILES
            for required in required_files:
                assert any(
                    member == required or member.endswith("/" + required)
                    for member in members
                ), f"{archive.name} omits {required}"
            for forbidden in _FORBIDDEN_DISTRIBUTION_FILES:
                assert not any(
                    member == forbidden or member.endswith("/" + forbidden)
                    for member in members
                ), f"{archive.name} retains removed {forbidden}"

    # Exercise the same public import and parser surface from both a source
    # checkout and an isolated wheel target.  The release verification also
    # invokes ``python -m pirate_frb --help`` in a subprocess from /tmp.
    for module_name in (
        "pirate_frb.Peakfinders",
        "pirate_frb.OfflineGrouperConfig",
        "pirate_frb.ArgmaxMetadata",
        "pirate_frb.GpuArgmaxDecoder",
        "pirate_frb.OfflineCandidateGrouper",
        "pirate_frb.FrbOfflineGrouper",
        "pirate_frb.TriggerCatalog",
        "pirate_frb.run_offline_grouper",
    ):
        importlib.import_module(module_name)
    main_module = importlib.import_module("pirate_frb.__main__")
    parser = main_module.get_parser()
    subparsers = next(
        action for action in parser._actions
        if action.__class__.__name__ == "_SubParsersAction"
    )
    run_subparsers = next(
        action for action in subparsers.choices["run"]._actions
        if action.__class__.__name__ == "_SubParsersAction"
    )
    help_text = run_subparsers.choices["offline_grouper"].format_help()
    assert "--output" in help_text
    assert "--assume-steady-state" in help_text
    assert "--device" in help_text
    assert "--max-chunks" in help_text
    assert "--verbose" in help_text
    assert "CONFIG.yml" in help_text
    for removed in (
            "--snr-threshold", "--beam-batch-size", "--peakfinder",
            "--dm-tolerance-bins", "--time-padding-bins", "--dm-reach",
            "--waist-bins", "--timeout-ms", "--timeout-policy",
            "--strict-audit", "--alpha-dm", "--alpha-time",
            "--exact-time-tile-columns", "--diagnostics", "--timings",
            "--peakfinder-method", "--edge-policy", "--startup-policy"):
        assert removed not in help_text
