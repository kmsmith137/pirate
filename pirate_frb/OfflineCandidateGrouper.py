"""Compatibility imports; use :mod:`pirate_frb.Clustering` for new code."""

from . import Clustering as _implementation

_ALIASES = {'GroupingConfig': 'ClusteringTolerances',
 'GroupingGeometry': 'ClusteringGeometry',
 'GpuGroupingResult': 'GpuClusteringResult',
 'group_candidates': 'cluster_candidates',
 '_group_candidates_serial_oracle': '_cluster_candidates_serial_oracle'}
__all__ = ['GpuDecodedCandidates',
 'GpuEventTable',
 'GpuGroupingResult',
 'GpuMemberTable',
 'GroupingConfig',
 'GroupingGeometry',
 'compatible_with_representative',
 'group_candidates']


def __getattr__(name):
    return getattr(_implementation, _ALIASES.get(name, name))


def __dir__():
    return sorted(set(globals()) | set(dir(_implementation)) | set(_ALIASES))
