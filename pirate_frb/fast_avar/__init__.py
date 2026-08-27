# Fast C++ ports of two analytic-variance code paths, both defined in
# src_lib/varmap.cpp and bound in src_pybind11/pirate_pybind11_avar.cpp:
#
#   - the slow_avar machinery (see pirate_frb/slow_avar): SparseTile and the PfAvar* classes;
#   - the detrender-free variance vectors of pirate_frb/varmap/detrender_free.py.
#
# In both cases the PYTHON is the reference implementation and stays where it is; these are
# for speed, and pirate_frb/fast_avar/test_fast_avar.py asserts the two agree.

# Import C++ classes and functions from pirate_pybind11
from ..pirate_pybind11 import (
    SparseTile,
    SparseTileTriple,
    PfVarianceConvolver,
    PfVariance,
    PfAvarApproximation,
    compute_detrender_free_varfine,
    compute_detrender_free_varcoarse,
)
