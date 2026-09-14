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

# The numpy reference for each class above. The standalone functions in those files
# (badchannel_keep, clip_1d, wrms_iterate, ...) are NOT re-exported: import them from the
# module that defines them, e.g. 'from pirate_frb.chimefrb.ReferenceStdDevClipper import
# clip_1d'. They are pieces of the ports, each validated on its own by a spot check, rather
# than part of the transform interface.
from .ReferenceWiDownsamplingKernel import ReferenceWiDownsamplingKernel
from .ReferenceWtUpsamplingKernel import ReferenceWtUpsamplingKernel
from .ReferenceWrmsKernel import ReferenceWrmsKernel
from .ReferenceIntensityClipper import ReferenceIntensityClipper
from .ReferenceStdDevClipper import ReferenceStdDevClipper
from .ReferenceBadChannelMask import ReferenceBadChannelMask
from .ReferenceSplineDetrender import ReferenceSplineDetrender
from .ReferencePolynomialDetrender import ReferencePolynomialDetrender

# The two pipeline classes that run transforms. Nothing from utils.py is re-exported either
# (read_yaml, transform_from_yaml_dict, YAML_WIDTH, ...): import from pirate_frb.chimefrb.utils.
from .ExamplePythonTransform import ExamplePythonTransform
from .GpuContainerBase import GpuContainerBase
from .Pipeline import Pipeline
from .RfiMaskPipeline import RfiMaskPipeline

# WARNING: the CLASSES above are load-bearing, and not merely a convenience. utils.
# resolve_class() turns a yaml 'class_name' into a class by getattr() on this package, so
# dropping one from this file makes every yaml file that names it unreadable -- reported as
# "unknown transform class_name ...", which reads like a bad file rather than a missing
# export. Functions are not resolved this way, which is why they can go.
