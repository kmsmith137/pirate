# Fast C++ ports of the slow_avar analytic-variance machinery (see pirate_frb/slow_avar).
# The C++ classes are defined in src_pybind11/pirate_pybind11_avar.cpp.

# Import C++ classes from pirate_pybind11
from ..pirate_pybind11 import (
    SparseTile,
    SparseTileTriple,
    PfVarianceConvolver,
    PfVariance,
    PfAvarApproximation,
)
