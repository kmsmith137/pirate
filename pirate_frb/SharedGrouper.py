"""Compatibility imports; use :mod:`pirate_frb.GrouperPipeline` for new code."""

from . import GrouperPipeline as _implementation

_ALIASES = {'GroupingWindow': 'ClusteringWindow',
 'GroupingConfig': 'ClusteringTolerances',
 'GroupingGeometry': 'ClusteringGeometry',
 'GpuGroupingResult': 'GpuClusteringResult',
 'group_candidates': 'cluster_candidates',
 'OfflinePeakExtractor': 'StreamingPeakExtractor'}
__all__ = [name for name in dir(_implementation) if not name.startswith('_')] + list(_ALIASES)


def __getattr__(name):
    return getattr(_implementation, _ALIASES.get(name, name))


def __dir__():
    return sorted(set(globals()) | set(dir(_implementation)) | set(_ALIASES))
