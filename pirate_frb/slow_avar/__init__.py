# Pure-Python "slow" reference implementations of the analytic-variance (avar) machinery
# (see notes/tree_dedispersion.tex).
from .SparseTile import SparseTile, SparseTileTriple, SparseTilePerM
from .PfVariance import PfVarianceConvolver, PfVariance, PfAvarExact, PfAvarApproximation
from .check_approximation import check_approximation
from .check_mc import check_avar_mc
