"""Pieces of the old CHIME FRB search, ported to pirate: a reader for its msgpack data
files (one at a time with ``AssembledChunk``, or a whole list, several at once, with
``AssembledChunkReader``), GPU ports of the transforms in its production RFI chain (each
with a numpy reference), and the two containers that chain them -- Pipeline and
RfiMaskPipeline -- which read the old json configs and read/write a yaml format of their own.

Every transform is a subclass of ``GpuTransform`` (the interface is stated in
``utils.py``); one written in python subclasses ``GpuPythonTransform``, and the
worked example ``ExamplePythonTransform`` shows how. One that RUNS other transforms
subclasses ``GpuContainerBase``, as the two containers do.

Where things live: the ported transforms and kernels are C++, bound with pybind11, and
their python side (the method injections) is in ``cpp_transforms.py``; each has a numpy
reference in its own ``Reference<ClassName>.py``; and ``utils.py`` holds the class-name
lookup, the legacy-json tables, and the yaml/json file functions.

See ``notes/chimefrb.md`` for the porting rules this subpackage follows.
"""

# The two pybind11 classes with no method injections. Every other one comes from
# cpp_transforms below.
from ..pirate_pybind11 import AssembledChunk, GpuClipperBase

# cpp_transforms.py holds the python side of every OTHER pybind11 class in this subpackage.
# It applies the injections as an import side effect and re-exports the classes, so it is
# imported first: from here on, these names are the injected classes.
from .cpp_transforms import (GpuTransform, GpuBadChannelMask, GpuIntensityClipper,
                             GpuPolynomialDetrender, GpuSplineDetrender, GpuStdDevClipper,
                             GpuWiDownsamplingKernel, GpuWrmsKernel, GpuWtUpsamplingKernel)

# AssembledChunkReader has injections of its own (iteration, the context manager), in a file
# of its own: it is the step BEFORE the transforms, not part of the transform interface.
from .AssembledChunkReader import AssembledChunkReader

# GpuPythonTransform, the base class of a transform written in python, is plain python on
# top of GpuTransform.
from .GpuPythonTransform import GpuPythonTransform

# The numpy reference for each class above, and the standalone functions alongside them.
from .ReferenceWiDownsamplingKernel import ReferenceWiDownsamplingKernel
from .ReferenceWtUpsamplingKernel import ReferenceWtUpsamplingKernel
from .ReferenceWrmsKernel import ReferenceWrmsKernel, wrms_iterate, iclip
from .ReferenceIntensityClipper import ReferenceIntensityClipper, intensity_clip, wrms_view
from .ReferenceStdDevClipper import ReferenceStdDevClipper, clip_1d, std_dev_apply
from .ReferenceBadChannelMask import ReferenceBadChannelMask, badchannel_keep
from .ReferenceSplineDetrender import ReferenceSplineDetrender
from .ReferencePolynomialDetrender import ReferencePolynomialDetrender

# The transform interface (the "protocol"), and the two pipeline classes that run transforms.
# GpuContainerBase imports utils, so it is imported here rather than with the base classes.
from .utils import (CHIME_FREQ_RANGE, IGNORED_JSON_CLASSES, LEGACY_JSON_CLASS_NAMES,
                    PIPELINE_YAML_HEADER, YAML_WIDTH,
                    read_json, read_yaml, resolve_class, transform_from_json_dict,
                    transform_from_yaml_dict, write_yaml, yaml_string)
from .ExamplePythonTransform import ExamplePythonTransform
from .GpuContainerBase import GpuContainerBase
from .Pipeline import Pipeline
from .RfiMaskPipeline import RfiMaskPipeline
