"""Compatibility imports; use :mod:`pirate_frb.BowtiePeakfinding` for new code."""

from . import BowtiePeakfinding as _implementation

_ALIASES = {'OfflinePeakExtractor': 'StreamingPeakExtractor'}
__all__ = ['EdgeFlag',
 'GpuRawCandidates',
 'K_DM',
 'OfflinePeakExtractor',
 'PeakFinderGeometry',
 'build_full_band_bowtie',
 'concatenate_raw_candidates',
 'extract_candidates',
 'make_startup_valid_mask']


def __getattr__(name):
    return getattr(_implementation, _ALIASES.get(name, name))


def __dir__():
    return sorted(set(globals()) | set(dir(_implementation)) | set(_ALIASES))
