# Pure-Python "slow" reference implementations of the analytic-variance (avar) machinery
# (see notes/tree_dedispersion.tex), plus file I/O for the variance maps they produce.
from .SparseTile import SparseTile, SparseTileTriple, SparseTilePerM
from .PfVariance import PfVarianceConvolver, PfVariance, PfAvarExact, PfAvarApproximation
from .VarianceMap import VarianceMapBlock, VarianceMapBase
from .VarianceMapExact import VarianceMapExact
from .VarianceMapApproximation import VarianceMapApproximation
from .brute_force import BruteForceVarianceMap
from .brute_force_gpu import GpuBruteForceVarianceMap
from .variance_map_io import (VarianceMapFile, VarianceMapTree,
                              read_variance_map, write_variance_map)
from .check_approximation import check_approximation
from .check_mc import check_avar_mc
