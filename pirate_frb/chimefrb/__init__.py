"""Pieces of the old CHIME FRB search, ported to pirate: a reader for its msgpack data
files, GPU ports of the transforms in its production RFI chain (each with a numpy
reference), and the two containers that chain them -- WiPipeline and RfiMaskPipeline --
which read the old json configs and read/write a yaml format of their own.

Every transform follows one interface, stated in ``transform_io.py``; ``CupyTransformBase``
and the worked example ``ExampleCupyTransform`` are how to add a transform in cupy.

See ``notes/chimefrb.md`` for the porting rules this subpackage follows.
"""

# Import C++ classes from pirate_pybind11
from ..pirate_pybind11 import AssembledChunk, GpuClipperBase

# GpuWiDownsampler has method injections, which live in ReferenceWiDownsampler.py
# alongside its numpy reference. That module both applies the injections (as an
# import side effect) and re-exports the class.
from .ReferenceWiDownsampler import GpuWiDownsampler, ReferenceWiDownsampler
from .ReferenceWeightUpsampler import GpuWeightUpsampler, ReferenceWeightUpsampler
from .ReferenceWrms import GpuWrms, ReferenceWrms, wrms_iterate, iclip
from .ReferenceIntensityClipper import (AXIS_FREQ, AXIS_TIME, AXIS_NONE, ClipperAxis,
                                        GpuIntensityClipper, ReferenceIntensityClipper,
                                        intensity_clip, wrms_view)
from .ReferenceStdDevClipper import (GpuStdDevClipper, ReferenceStdDevClipper, clip_1d,
                                     std_dev_apply)
from .ReferenceBadChannelMask import GpuBadChannelMask, ReferenceBadChannelMask, badchannel_keep
from .ReferenceSplineDetrender import GpuSplineDetrender, ReferenceSplineDetrender
from .ReferencePolynomialDetrender import GpuPolynomialDetrender, ReferencePolynomialDetrender

# The transform interface (the "protocol"), and the two pipeline classes that run transforms.
from .transform_io import (CHIME_FREQ_RANGE, IGNORED_JSON_CLASSES, LEGACY_JSON_CLASS_NAMES,
                           axis_from_str, axis_to_str, check_yaml_keys, resolve_class,
                           transform_from_json_dict, transform_from_yaml_dict)
from .CupyTransformBase import CupyTransformBase
from .ExampleCupyTransform import ExampleCupyTransform
from .WiPipeline import WiPipeline
from .RfiMaskPipeline import RfiMaskPipeline
