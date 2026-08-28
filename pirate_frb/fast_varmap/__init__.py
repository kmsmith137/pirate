# Fast C++ ports of the analytic-variance machinery, defined in src_lib/varmap.cpp and bound
# in src_pybind11/pirate_pybind11_varmap.cpp:
#
#   - the low-level primitives of pirate_frb/varmap/{SparseTile,PfVarianceConvolver}.py;
#   - the detrender-free variance vectors of pirate_frb/varmap/detrender_free.py.
#
# In both cases the PYTHON is the reference implementation and stays where it is; these are
# for speed, and pirate_frb/fast_varmap/test_fast_varmap.py asserts the two agree. Both are
# dispatched from 'pirate_frb test --varmap', next to the python they guard.

# Import C++ classes and functions from pirate_pybind11
from ..pirate_pybind11 import (
    SparseTile,
    SparseTileTriple,
    PfVarianceConvolver,
    compute_detrender_free_varfine,
    compute_detrender_free_varcoarse,
)
