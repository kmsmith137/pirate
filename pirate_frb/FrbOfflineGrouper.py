"""Compatibility imports; use :mod:`pirate_frb.OfflineMapReader` for new code."""

from . import OfflineMapReader as _implementation

_ALIASES = {'FrbOfflineGrouper': 'OfflineMapReader'}
__all__ = ['BeamBatch', 'FrbOfflineGrouper', 'GpuMapChunk']


def __getattr__(name):
    return getattr(_implementation, _ALIASES.get(name, name))


def __dir__():
    return sorted(set(globals()) | set(dir(_implementation)) | set(_ALIASES))
