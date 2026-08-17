# Low-rank representations of the variance map A of a DedispersionTree (notes/variance_map.tex).
#
# This package supersedes the variance-map half of pirate_frb.slow_avar (VarMapDistance,
# varmap_eval, variance_map_io, brute_force*). Those stay in place, unmodified, for as long as
# they are the reference the new code is checked against, so varmap is purely additive until
# that equivalence is established.
#
# Not superseded, and callable from here: slow_avar's SparseTile, PfVariance, TmpVmap and
# check_* modules.
from .distance import YTRUE_FLOOR, f, fprime, AdmissibilityResult, DistanceEstimate
from .VarianceMap import VarianceMap
from .VarianceMultiMap import VarianceMultiMap
