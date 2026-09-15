"""Compatibility imports; use :mod:`pirate_frb.GrouperConfig` for new code."""

from . import GrouperConfig as _implementation

_ALIASES = {'OfflineGrouperConfig': 'GrouperConfig',
 'OfflineGrouperConfigError': 'GrouperConfigError',
 'GroupingConfig': 'ClusteringConfig',
 'load_offline_grouper_config': 'load_grouper_config'}
__all__ = ['ExecutionConfig',
 'GroupingConfig',
 'OfflineGrouperConfig',
 'OfflineGrouperConfigError',
 'PeakfindingConfig',
 'load_offline_grouper_config']


def __getattr__(name):
    return getattr(_implementation, _ALIASES.get(name, name))


def __dir__():
    return sorted(set(globals()) | set(dir(_implementation)) | set(_ALIASES))
