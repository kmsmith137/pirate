"""Pieces of the old CHIME FRB search, ported to pirate: a reader for its msgpack data
files, GPU ports of the transforms in its production RFI chain (each with a numpy
reference), and the two containers that chain them -- Pipeline and RfiMaskPipeline --
which read the old json configs and read/write a yaml format of their own.

Every transform is a subclass of ``GpuTransform`` (the interface is stated in
``transform_io.py``); one written in python subclasses ``GpuPythonTransform``, and the
worked example ``ExamplePythonTransform`` shows how. One that RUNS other transforms
subclasses ``GpuContainerBase``, as the two containers do.

See ``notes/chimefrb.md`` for the porting rules this subpackage follows.
"""

# Import C++ classes from pirate_pybind11
from ..pirate_pybind11 import AssembledChunk, GpuClipperBase

# GpuTransform, the base class of every transform, has method injections (the python
# side shared by C++ and python transforms) in GpuTransform.py, which applies them and
# re-exports the class. GpuPythonTransform, the base of a transform written in python, is
# plain python on top of it.
from .GpuTransform import GpuTransform
from .GpuPythonTransform import GpuPythonTransform

# GpuContainerBase, the base of a transform that runs other transforms, imports transform_io,
# so it is imported below with the two containers rather than here.

# GpuWiDownsamplingKernel has method injections, which live in ReferenceWiDownsamplingKernel.py
# alongside its numpy reference. That module both applies the injections (as an
# import side effect) and re-exports the class.
from .ReferenceWiDownsamplingKernel import GpuWiDownsamplingKernel, ReferenceWiDownsamplingKernel
from .ReferenceWtUpsamplingKernel import GpuWtUpsamplingKernel, ReferenceWtUpsamplingKernel
from .ReferenceWrmsKernel import GpuWrmsKernel, ReferenceWrmsKernel, wrms_iterate, iclip
from .ReferenceIntensityClipper import (GpuIntensityClipper, ReferenceIntensityClipper,
                                        intensity_clip, wrms_view)
from .ReferenceStdDevClipper import (GpuStdDevClipper, ReferenceStdDevClipper, clip_1d,
                                     std_dev_apply)
from .ReferenceBadChannelMask import GpuBadChannelMask, ReferenceBadChannelMask, badchannel_keep
from .ReferenceSplineDetrender import GpuSplineDetrender, ReferenceSplineDetrender
from .ReferencePolynomialDetrender import GpuPolynomialDetrender, ReferencePolynomialDetrender

# The transform interface (the "protocol"), and the two pipeline classes that run transforms.
from .transform_io import (CHIME_FREQ_RANGE, IGNORED_JSON_CLASSES, LEGACY_JSON_CLASS_NAMES,
                           PIPELINE_YAML_HEADER, YAML_WIDTH,
                           read_json, read_yaml, resolve_class, transform_from_json_dict,
                           transform_from_yaml_dict, write_yaml, yaml_string)
from .ExamplePythonTransform import ExamplePythonTransform
from .GpuContainerBase import GpuContainerBase
from .Pipeline import Pipeline
from .RfiMaskPipeline import RfiMaskPipeline
