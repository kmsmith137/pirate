"""Serialized decoder metadata shared by offline maps and trigger catalogs.

PIRATE 1.5 plans specify tree geometry. The actual producer separately supplies
Dcores: a consumer must carry these through storage and must never derive them
from its own kernel registry. This module performs CPU metadata checks only.
"""
from collections.abc import Mapping
from math import isfinite
from numbers import Integral, Real

import numpy as np

SNR_MAP_FORMAT = "pirate_frb.offline_dedisperser_snr_maps"
SNR_MAP_VERSION = 3
ARGMAX_ENCODING = "pirate-1.5:t8-p8-m8-mu8"


def validate_snr_map_format(metadata):
    """Reject legacy maps before attempting to reconstruct their old YAML."""
    if not isinstance(metadata, Mapping) or metadata.get("format") != SNR_MAP_FORMAT:
        raise ValueError("unexpected offline S/N-map format")
    version = metadata.get("format_version")
    if (isinstance(version, (bool, np.bool_))
            or not isinstance(version, Integral) or version != SNR_MAP_VERSION):
        raise ValueError(
            "expected format_version=3 (PIRATE 1.5); version-2 maps require "
            "the preserved PIRATE 1.4 environment or regeneration from raw frames; "
            "changing only the version label does not convert tokens"
        )


def read_argmax_metadata(metadata, *, ntrees, douts=None):
    """Return exact producer Dcores after validating token layout and bounds.

    With a reconstructed plan, pass its per-tree nt_ds // nt_out as douts.
    Catalog validation can check the stored encoding and vector shape without
    constructing a GPU-capable plan.
    """
    if not isinstance(metadata, Mapping) or metadata.get("argmax_encoding") != ARGMAX_ENCODING:
        raise ValueError(f"argmax_encoding must be {ARGMAX_ENCODING!r}")
    values = metadata.get("dcores")
    if not isinstance(values, (list, tuple, np.ndarray)):
        raise ValueError("dcores must be a sequence with one producer value per tree")
    if len(values) != ntrees:
        raise ValueError("dcores must contain one producer value per tree")
    if douts is not None and len(douts) != ntrees:
        raise ValueError("Dout count disagrees with the producer plan")
    result = []
    for i, value in enumerate(values):
        if isinstance(value, (bool, np.bool_)) or not isinstance(value, Integral):
            raise ValueError(f"dcores[{i}] must be an integer")
        value = int(value)
        if not 1 <= value <= 256 or value & (value - 1):
            raise ValueError(f"dcores[{i}] must be a power of two in [1, 256]")
        if douts is not None and (value > douts[i] or douts[i] % value):
            raise ValueError(f"dcores[{i}] is incompatible with this tree's Dout")
        result.append(value)
    return tuple(result)


def validate_saved_time_sample_ms(metadata, plan):
    """A redundant saved time sample may not override the producer plan."""
    expected = float(plan.config.time_sample_ms)
    if not isfinite(expected) or expected <= 0:
        raise ValueError("authoritative plan time_sample_ms must be finite and positive")
    value = metadata.get("time_sample_ms")
    if value is None:
        return
    if isinstance(value, (bool, np.bool_)) or not isinstance(value, Real):
        raise ValueError("time_sample_ms must be numeric")
    if not isfinite(float(value)) or value <= 0:
        raise ValueError("time_sample_ms must be finite and positive")
    if float(value) != expected:
        raise ValueError("saved time_sample_ms disagrees with the authoritative plan")
