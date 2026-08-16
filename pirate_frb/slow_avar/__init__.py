# Pure-Python "slow" reference implementations of the analytic-variance (avar) machinery
# (see notes/variance_map.tex), plus file I/O for the variance maps they produce.
from .SparseTile import SparseTile, SparseTileTriple, SparseTilePerM

# The PfAvar* and TmpVmap* classes on the next two lines were a first pass at representing a
# variance map, predating the VarianceMap representation. They are unchanged for now, and are
# deliberately outside it -- the 'TmpVmap' prefix is the reminder. We may revisit them later.
from .PfVariance import PfVarianceConvolver, PfVariance, PfAvarExact, PfAvarApproximation
from .TmpVmap import TmpVmapBlock, TmpVmapBase, TmpVmapExact, TmpVmapApproximation

from .VarMapDistance import VarMapDistance
from .varmap_eval import (LowRankApprox, FactoredApprox, ClusterEnvelope, DenseApprox,
                          evaluate, evaluate_reduced, reduce_map, frontier, row_distances)
from .brute_force import BruteForceVarianceMap
from .brute_force_gpu import GpuBruteForceVarianceMap
from .variance_map_io import (VarianceMapFile, VarianceMapTree,
                              read_variance_map, write_variance_map)
from .check_approximation import check_approximation
from .check_mc import check_avar_mc
