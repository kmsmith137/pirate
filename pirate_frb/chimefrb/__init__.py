"""Reading data files written by the old CHIME FRB search.

See ``notes/chimefrb.md`` for the porting rules this subpackage follows.
"""

# Import C++ classes from pirate_pybind11
from ..pirate_pybind11 import AssembledChunk

# GpuWiDownsampler has method injections, which live in ReferenceWiDownsampler.py
# alongside its numpy reference. That module both applies the injections (as an
# import side effect) and re-exports the class.
from .ReferenceWiDownsampler import GpuWiDownsampler, ReferenceWiDownsampler
from .ReferenceWrms import GpuWrms, ReferenceWrms, wrms_iterate, iclip
