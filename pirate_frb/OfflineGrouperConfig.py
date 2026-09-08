"""Strict YAML configuration for the offline candidate grouper.

The offline grouper intentionally has a small, closed configuration schema.
Every section and field is required, unknown fields are rejected, and YAML
booleans are never accepted as numbers.  Keeping parsing in this lightweight
module lets callers validate a run before importing CuPy or allocating GPU
state.
"""

from __future__ import annotations

import math
import os
from dataclasses import dataclass
from typing import Any, Mapping

import yaml


class OfflineGrouperConfigError(ValueError):
    """Raised when an offline-grouper YAML file violates its schema."""


class _StrictSafeLoader(yaml.SafeLoader):
    """SafeLoader variant which also rejects duplicate mapping keys."""


def _construct_unique_mapping(loader, node, deep=False):
    loader.flatten_mapping(node)
    mapping = {}
    for key_node, value_node in node.value:
        key = loader.construct_object(key_node, deep=deep)
        try:
            duplicate = key in mapping
        except TypeError as exc:
            raise OfflineGrouperConfigError(
                "configuration mapping keys must be scalar values"
            ) from exc
        if duplicate:
            raise OfflineGrouperConfigError(
                f"duplicate configuration key {key!r}"
            )
        mapping[key] = loader.construct_object(value_node, deep=deep)
    return mapping


_StrictSafeLoader.add_constructor(
    yaml.resolver.BaseResolver.DEFAULT_MAPPING_TAG,
    _construct_unique_mapping,
)


def _mapping(value: Any, path: str) -> Mapping[str, Any]:
    if not isinstance(value, Mapping):
        raise OfflineGrouperConfigError(f"{path} must be a mapping")
    return value


def _exact_keys(
        value: Mapping[str, Any], expected: tuple[str, ...], path: str) -> None:
    missing = [key for key in expected if key not in value]
    unknown = [key for key in value if key not in expected]
    if missing:
        raise OfflineGrouperConfigError(
            f"{path} is missing required key(s): {', '.join(missing)}"
        )
    if unknown:
        rendered = ", ".join(repr(key) for key in unknown)
        raise OfflineGrouperConfigError(
            f"{path} contains unknown key(s): {rendered}"
        )


def _finite_number(value: Any, path: str, *, nonnegative: bool = False) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise OfflineGrouperConfigError(f"{path} must be a finite number")
    try:
        answer = float(value)
    except OverflowError as exc:
        raise OfflineGrouperConfigError(
            f"{path} must be a finite number"
        ) from exc
    if not math.isfinite(answer):
        raise OfflineGrouperConfigError(f"{path} must be a finite number")
    if nonnegative and answer < 0.0:
        raise OfflineGrouperConfigError(
            f"{path} must be a finite nonnegative number"
        )
    return answer


def _integer(value: Any, path: str, *, minimum: int) -> int:
    if isinstance(value, bool) or not isinstance(value, int):
        raise OfflineGrouperConfigError(f"{path} must be an integer")
    if value < minimum:
        comparison = "positive" if minimum == 1 else f"at least {minimum}"
        raise OfflineGrouperConfigError(f"{path} must be {comparison}")
    return value


@dataclass(frozen=True)
class PeakfindingConfig:
    """Scientific controls for full-band Bowtie peak extraction."""

    snr_threshold: float
    dm_reach: int
    waist_bins: int


@dataclass(frozen=True)
class GroupingConfig:
    """Streaming-halo and cross-tree candidate association controls."""

    halo_size: int
    dm_tolerance: float
    time_tolerance: float


@dataclass(frozen=True)
class ExecutionConfig:
    """Beam batching and bounded-wait behavior."""

    beam_batch_size: int
    timeout_ms: int
    timeout_policy: str


@dataclass(frozen=True)
class OfflineGrouperConfig:
    """Validated, immutable configuration for one offline-grouper run."""

    peakfinding: PeakfindingConfig
    grouping: GroupingConfig
    execution: ExecutionConfig

    @classmethod
    def from_mapping(cls, value: Mapping[str, Any]) -> "OfflineGrouperConfig":
        """Validate an already decoded YAML mapping."""

        root = _mapping(value, "configuration")
        _exact_keys(root, ("peakfinding", "grouping", "execution"), "configuration")

        peakfinding = _mapping(root["peakfinding"], "peakfinding")
        _exact_keys(
            peakfinding,
            ("snr_threshold", "dm_reach", "waist_bins"),
            "peakfinding",
        )
        peakfinding_config = PeakfindingConfig(
            snr_threshold=_finite_number(
                peakfinding["snr_threshold"], "peakfinding.snr_threshold"
            ),
            dm_reach=_integer(
                peakfinding["dm_reach"], "peakfinding.dm_reach", minimum=0
            ),
            waist_bins=_integer(
                peakfinding["waist_bins"], "peakfinding.waist_bins", minimum=0
            ),
        )

        grouping = _mapping(root["grouping"], "grouping")
        _exact_keys(
            grouping,
            ("halo_size", "dm_tolerance", "time_tolerance"),
            "grouping",
        )
        grouping_config = GroupingConfig(
            # Two temporal radii are the mathematical minimum needed to
            # reconsider a centre delayed at a chunk seam with full context.
            halo_size=_integer(
                grouping["halo_size"], "grouping.halo_size", minimum=2
            ),
            dm_tolerance=_finite_number(
                grouping["dm_tolerance"],
                "grouping.dm_tolerance",
                nonnegative=True,
            ),
            time_tolerance=_finite_number(
                grouping["time_tolerance"],
                "grouping.time_tolerance",
                nonnegative=True,
            ),
        )

        execution = _mapping(root["execution"], "execution")
        _exact_keys(
            execution,
            ("beam_batch_size", "timeout_ms", "timeout_policy"),
            "execution",
        )
        timeout_policy = execution["timeout_policy"]
        if timeout_policy not in ("discard", "emit_partial"):
            raise OfflineGrouperConfigError(
                "execution.timeout_policy must be exactly 'discard' or "
                "'emit_partial'"
            )
        execution_config = ExecutionConfig(
            beam_batch_size=_integer(
                execution["beam_batch_size"],
                "execution.beam_batch_size",
                minimum=1,
            ),
            timeout_ms=_integer(
                execution["timeout_ms"], "execution.timeout_ms", minimum=0
            ),
            timeout_policy=timeout_policy,
        )

        return cls(
            peakfinding=peakfinding_config,
            grouping=grouping_config,
            execution=execution_config,
        )

    @classmethod
    def from_yaml(
            cls, config_file: os.PathLike[str] | str
    ) -> "OfflineGrouperConfig":
        """Load and strictly validate a safe YAML file."""

        return load_offline_grouper_config(config_file)


def load_offline_grouper_config(
        config_file: os.PathLike[str] | str) -> OfflineGrouperConfig:
    """Load ``config_file`` with a safe YAML loader and validate its schema."""

    filename = os.fspath(config_file)
    try:
        with open(filename, "r", encoding="utf-8") as stream:
            decoded = yaml.load(stream, Loader=_StrictSafeLoader)
    except OfflineGrouperConfigError:
        raise
    except yaml.YAMLError as exc:
        raise OfflineGrouperConfigError(
            f"invalid offline-grouper YAML in {filename!r}: {exc}"
        ) from exc
    return OfflineGrouperConfig.from_mapping(decoded)


__all__ = [
    "ExecutionConfig",
    "GroupingConfig",
    "OfflineGrouperConfig",
    "OfflineGrouperConfigError",
    "PeakfindingConfig",
    "load_offline_grouper_config",
]
