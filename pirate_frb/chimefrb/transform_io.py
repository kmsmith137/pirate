"""The interface every chimefrb transform follows ("the transform protocol"), and the
helpers behind it: turning a yaml or legacy-json ``class_name`` into a class, the axis
strings, and the argument checks a python-side ``launch()`` makes.

WHAT A TRANSFORM IS. Any python object that a :class:`WiPipeline` or
:class:`RfiMaskPipeline` can run. Five come from C++ (GpuBadChannelMask,
GpuIntensityClipper, GpuStdDevClipper, GpuPolynomialDetrender, GpuSplineDetrender), the two
pipeline classes are transforms themselves so that they nest, and any number can be written
in cupy on top of :class:`CupyTransformBase`. A transform has::

    nbeams, nfreq, ntime    ints: the (beams, channels, time samples) block it processes,
                            fixed at construction
    scratch_nelts           int: float32 scratch elements launch() needs; may be 0

    launch(intensity, weights, scratch, stream=None)
    to_yaml_dict()                              -> dict
    from_yaml_dict(d, nbeams, nfreq, ntime)     classmethod -> instance

and, for the five C++ transforms and the two pipelines only (the "legacy" transforms,
which the old rf_pipelines json format describes)::

    from_json_dict(d, nbeams, nfreq, ntime)     classmethod -> instance

THE launch() CONTRACT.

- ``intensity`` and ``weights`` are cupy float32 arrays of shape (nbeams, nfreq, ntime),
  C-contiguous, and distinct. Either or both may be modified in place; which one is the
  transform's business, stated in its docstring. Weights are nonnegative, and a zero weight
  means "ignore this sample".
- ``scratch`` is a REQUIRED positional argument: a 1-d C-contiguous cupy float32 array with
  at least ``scratch_nelts`` elements (any array, even an empty one, when scratch_nelts is
  0), or the literal ``None``, which allocates one. There is deliberately no default, so
  that a caller who wants the allocation says ``scratch=None`` and can see it in the call.
  Contents on entry are ignored and on exit are garbage; a transform uses a prefix of it,
  and it must not alias the data arrays. One call processes exactly one block, and every
  launch validates all of this and raises on a violation.
- ``stream`` is a cupy stream, or None for the current cupy stream. The launch is
  asynchronous on that stream; nothing synchronizes.

SERIALIZATION.

- ``to_yaml_dict()`` returns a plain dict of python scalars and lists that
  ``yaml.safe_dump`` accepts: a ``class_name`` key holding the transform's PYTHON class
  name (``GpuBadChannelMask``, ``WiPipeline``, ...), then its semantic parameters. It never
  writes nbeams/nfreq/ntime (they are properties of the data, supplied when reading) and
  never writes performance knobs, so a file is not tied to a GPU.
- ``from_yaml_dict(d, nbeams, nfreq, ntime)`` checks that ``d['class_name']`` is the class's
  own name and that the other keys are EXACTLY the expected ones (a misspelled key must not
  silently become a default), then constructs. :func:`check_yaml_keys` does both checks.
- ``from_json_dict`` reads one element of the old rf_pipelines json, whose ``class_name`` is
  the legacy name (``badchannel_mask``, ``intensity_clipper``, ...).

HOW A class_name BECOMES A CLASS. :func:`resolve_class` looks the name up first among the
``classes`` the caller passed in (matched by ``__name__``; this is how a transform you wrote
yourself, in your own module or in a notebook, gets found), then as an attribute of the
``pirate_frb.chimefrb`` package. For the legacy json, a fixed table maps the old names to
class names first.
"""

import json

import numpy as np
import yaml

from ..pirate_pybind11 import ClipperAxis
from ..utils import atomic_print


# The CHIME band, which the legacy json never recorded: rf_pipelines' badchannel_mask read
# it from the stream at bind time. GpuBadChannelMask.from_json_dict() assumes it.
CHIME_FREQ_RANGE = (400.0, 800.0)


# -------------------------------------------------------------------------------------------------
#
# Class resolution


def resolve_class(class_name, classes=None):
    """The transform class named ``class_name`` (section "how a class_name becomes a class"
    in the module docstring).

    Parameters
    ----------
    class_name : str
        A python class name, as written by ``to_yaml_dict()``.
    classes : sequence of type or None, optional
        The caller's own transform classes, searched first, by ``__name__``.

    Raises
    ------
    ValueError
        If nothing of that name is found, or what is found has no ``from_yaml_dict``.
    """

    if not isinstance(class_name, str):
        raise ValueError(f"expected 'class_name' to be a string, got {class_name!r}")

    for cls in (classes or ()):
        if getattr(cls, '__name__', None) == class_name:
            return _checked_transform_class(cls, class_name)

    import pirate_frb.chimefrb as pkg
    cls = getattr(pkg, class_name, None)
    if cls is None:
        extra = f", nor among the {len(classes)} class(es) passed in classes=[...]" if classes else \
            ". A transform of your own must be passed to the reader in classes=[...]"
        raise ValueError(f"unknown transform class_name {class_name!r}: not found in"
                         f" pirate_frb.chimefrb{extra}")
    return _checked_transform_class(cls, class_name)


def _checked_transform_class(cls, class_name):
    if not (isinstance(cls, type) and hasattr(cls, 'from_yaml_dict')):
        raise ValueError(f"{class_name!r} is not a transform class (it has no from_yaml_dict())")
    return cls


def _pipeline_classes():
    """The two container classes, whose factories take an extra argument (see
    transform_from_yaml_dict() and transform_from_json_dict())."""
    import pirate_frb.chimefrb as pkg
    return (pkg.WiPipeline, pkg.RfiMaskPipeline)


def transform_from_yaml_dict(d, nbeams, nfreq, ntime, classes=None):
    """Build the transform that the yaml dict ``d`` describes, at the given geometry.

    Dispatches on ``d['class_name']`` (see :func:`resolve_class`). The two pipeline classes
    are the only ones whose ``from_yaml_dict`` takes ``classes``, and it is passed only to
    them, so that a leaf transform's factory keeps the plain four-argument signature.
    """

    if not isinstance(d, dict) or ('class_name' not in d):
        raise ValueError(f"expected a dict with a 'class_name' key describing a transform, got {d!r}")

    cls = resolve_class(d['class_name'], classes)
    if issubclass(cls, _pipeline_classes()):
        return cls.from_yaml_dict(d, nbeams, nfreq, ntime, classes=classes)
    return cls.from_yaml_dict(d, nbeams, nfreq, ntime)


# -------------------------------------------------------------------------------------------------
#
# Legacy json


# Legacy rf_pipelines class_name -> python class name, for everything that has a port. Closed
# set: a new transform has no legacy form.
LEGACY_JSON_CLASS_NAMES = {
    'pipeline': 'WiPipeline',
    'wi_sub_pipeline': 'RfiMaskPipeline',
    'badchannel_mask': 'GpuBadChannelMask',
    'std_dev_clipper': 'GpuStdDevClipper',
    'intensity_clipper': 'GpuIntensityClipper',
    'polynomial_detrender': 'GpuPolynomialDetrender',
    'spline_detrender': 'GpuSplineDetrender',
}

# Legacy classes with no port that DO NOT CHANGE THE DATA -- counters, writers, and consumers
# that read the stream without touching it. transform_from_json_dict() skips these, with a
# printed note. Anything else unported raises, so that a transform that does change the
# data (mask_expander, chime_16k_derippler, noise_filler, ...) is refused rather than
# silently dropped.
IGNORED_JSON_CLASSES = frozenset([
    'mask_counter', 'chime_slow_pulsar_writer', 'chime_file_writer',
    'chime_assembled_chunk_file_writer', 'chime_packetizer', 'bonsai_dedisperser',
    'plotter_transform',
])


def transform_from_json_dict(d, nbeams, nfreq, ntime, nds=1):
    """Build the transform that one element of a legacy rf_pipelines json describes, at the
    given geometry -- or return None if the element is one of :data:`IGNORED_JSON_CLASSES`.

    ``nds`` is the time downsampling of the data relative to the native stream (1 at top
    level; ``nds*Dt`` inside a wi_sub_pipeline), which only the pipelines need, to resolve a
    ``wi_sub_pipeline`` given as ``nds_out``.
    """

    if not isinstance(d, dict) or ('class_name' not in d):
        raise ValueError(f"expected a legacy json element with a 'class_name' key, got {d!r}")

    name = d['class_name']
    if name in IGNORED_JSON_CLASSES:
        # To stderr, so that a converter writing yaml to stdout stays clean.
        atomic_print(f"transform_from_json_dict: skipping a '{name}' element (no pirate counterpart,"
                     f" and it does not modify the data)", fd=2)
        return None
    if name not in LEGACY_JSON_CLASS_NAMES:
        raise ValueError(f"legacy json class_name {name!r} has no pirate counterpart and is not"
                         f" known to leave the data unchanged; ported names: "
                         f"{sorted(LEGACY_JSON_CLASS_NAMES)}, skipped names: {sorted(IGNORED_JSON_CLASSES)}")

    cls = resolve_class(LEGACY_JSON_CLASS_NAMES[name])
    if issubclass(cls, _pipeline_classes()):
        return cls.from_json_dict(d, nbeams, nfreq, ntime, nds=nds)
    return cls.from_json_dict(d, nbeams, nfreq, ntime)


def check_json_keys(d, class_name, required):
    """Check that a legacy json element has the expected ``class_name`` and carries every
    key in ``required`` (extra keys are allowed: old files sometimes carry more)."""

    if d.get('class_name') != class_name:
        raise ValueError(f"expected legacy class_name {class_name!r}, got {d.get('class_name')!r}")
    missing = [k for k in required if k not in d]
    if missing:
        raise ValueError(f"legacy json for {class_name!r} is missing key(s) {missing}")


# -------------------------------------------------------------------------------------------------
#
# Yaml helpers


def check_yaml_keys(d, class_name, keys):
    """The check every ``from_yaml_dict`` starts with: ``d['class_name'] == class_name``, and
    the other keys of ``d`` are exactly ``keys`` -- a missing key and an unexpected key are
    both errors, with a message naming them."""

    if not isinstance(d, dict):
        raise ValueError(f"{class_name}.from_yaml_dict: expected a dict, got {type(d).__name__}")
    if d.get('class_name') != class_name:
        raise ValueError(f"{class_name}.from_yaml_dict: expected class_name {class_name!r},"
                         f" got {d.get('class_name')!r}")

    got = set(d) - {'class_name'}
    want = set(keys)
    if got != want:
        parts = []
        if want - got:
            parts.append(f"missing key(s) {sorted(want - got)}")
        if got - want:
            parts.append(f"unexpected key(s) {sorted(got - want)}")
        raise ValueError(f"{class_name}.from_yaml_dict: " + ", ".join(parts))


_AXIS_STR = {int(ClipperAxis.FREQ): 'freq', int(ClipperAxis.TIME): 'time', int(ClipperAxis.NONE): 'none'}
_AXIS_FROM_STR = {'freq': ClipperAxis.FREQ, 'time': ClipperAxis.TIME, 'none': ClipperAxis.NONE}
_AXIS_FROM_JSON = {'AXIS_FREQ': ClipperAxis.FREQ, 'AXIS_TIME': ClipperAxis.TIME, 'AXIS_NONE': ClipperAxis.NONE}


def axis_to_str(axis):
    """'freq', 'time' or 'none', from a ClipperAxis or the AXIS_* integer."""
    return _AXIS_STR[int(axis)]


def axis_from_str(s):
    """A ClipperAxis, from the yaml strings 'freq', 'time', 'none'."""
    if s not in _AXIS_FROM_STR:
        raise ValueError(f"expected axis to be one of {sorted(_AXIS_FROM_STR)}, got {s!r}")
    return _AXIS_FROM_STR[s]


def axis_from_json(s):
    """A ClipperAxis, from the legacy json strings 'AXIS_FREQ', 'AXIS_TIME', 'AXIS_NONE'."""
    if s not in _AXIS_FROM_JSON:
        raise ValueError(f"expected a legacy axis string, one of {sorted(_AXIS_FROM_JSON)}, got {s!r}")
    return _AXIS_FROM_JSON[s]


# -------------------------------------------------------------------------------------------------
#
# launch() helpers


def default_scratch_and_stream(scratch, stream, scratch_nelts):
    """Resolve the two launch() arguments that may be None: ``scratch=None`` allocates a
    cupy float32 array of ``scratch_nelts`` elements, and ``stream=None`` is the current cupy
    stream. Returns ``(scratch, stream)``."""

    import cupy as cp

    if stream is None:
        stream = cp.cuda.get_current_stream()
    if scratch is None:
        scratch = cp.empty(int(scratch_nelts), dtype=cp.float32)
    return (scratch, stream)


def check_launch_args(intensity, weights, scratch, shape, scratch_nelts, who):
    """The python-side version of the launch() argument checks (module docstring): both data
    arrays cupy float32, C-contiguous, of ``shape``, and distinct; ``scratch`` a 1-d
    C-contiguous cupy float32 array with at least ``scratch_nelts`` elements, aliasing
    neither. ``who`` names the caller in the message. Call after
    :func:`default_scratch_and_stream`, so that ``scratch`` is an array."""

    import cupy as cp

    for (name, arr) in (('intensity', intensity), ('weights', weights)):
        if not isinstance(arr, cp.ndarray):
            raise TypeError(f"{who}.launch(): expected '{name}' to be a cupy array, got {type(arr).__name__}")
        if arr.dtype != np.float32:
            raise TypeError(f"{who}.launch(): expected '{name}' to be float32, got {arr.dtype}")
        if arr.shape != tuple(shape):
            raise ValueError(f"{who}.launch(): expected '{name}' of shape {tuple(shape)}, got {arr.shape}")
        if not arr.flags.c_contiguous:
            raise ValueError(f"{who}.launch(): expected '{name}' to be C-contiguous")

    if intensity.data.ptr == weights.data.ptr:
        raise ValueError(f"{who}.launch(): 'intensity' and 'weights' must be distinct arrays")

    if not isinstance(scratch, cp.ndarray):
        raise TypeError(f"{who}.launch(): expected 'scratch' to be a cupy array or None, got {type(scratch).__name__}")
    if scratch.dtype != np.float32:
        raise TypeError(f"{who}.launch(): expected 'scratch' to be float32, got {scratch.dtype}")
    if (scratch.ndim != 1) or (not scratch.flags.c_contiguous):
        raise ValueError(f"{who}.launch(): expected 'scratch' to be a 1-d contiguous array, got shape {scratch.shape}")
    if scratch.size < scratch_nelts:
        raise ValueError(f"{who}.launch(): 'scratch' has {scratch.size} elements, need at least {scratch_nelts}")
    if scratch.size > 0:
        for (name, arr) in (('intensity', intensity), ('weights', weights)):
            lo = min(scratch.data.ptr, arr.data.ptr)
            hi_s = scratch.data.ptr + 4 * scratch.size
            hi_a = arr.data.ptr + 4 * arr.size
            if (scratch.data.ptr < hi_a) and (arr.data.ptr < hi_s):
                raise ValueError(f"{who}.launch(): 'scratch' overlaps '{name}'")


# -------------------------------------------------------------------------------------------------
#
# yaml / json files. Plain functions on plain dicts; WiPipeline.write_yaml_file() and friends
# are thin wrappers around these.


# The comment WiPipeline/RfiMaskPipeline.write_yaml_file() put at the top of a file.
PIPELINE_YAML_HEADER = ('# A pirate_frb.chimefrb transform chain. Read with\n'
                        '#   WiPipeline.read_yaml_file(filename, nbeams=..., nfreq=..., ntime=...)\n'
                        '# (or RfiMaskPipeline.read_yaml_file, if that is the top-level class_name).\n')


class _YamlDumper(yaml.SafeDumper):
    """yaml.SafeDumper, except that a list of scalars is written on one line
    (``freq_range: [400.0, 800.0]``, one ``[lo, hi]`` per mask range) while lists of mappings
    -- the transforms -- stay one element per line."""


def _represent_list(dumper, data):
    flow = (len(data) > 0) and all(isinstance(x, (int, float, str, bool)) for x in data)
    return dumper.represent_sequence('tag:yaml.org,2002:seq', data, flow_style=flow)


_YamlDumper.add_representer(list, _represent_list)


def yaml_string(data, header=None):
    """``data`` (a plain dict, e.g. a ``to_yaml_dict()``) as yaml text, keys in their natural
    order and lists of numbers on one line, preceded by ``header`` (a string of ``#`` comment
    lines, ending in a newline) if one is given."""
    s = yaml.dump(data, Dumper=_YamlDumper, sort_keys=False)
    return (header + s) if (header is not None) else s


def write_yaml(filename, data, header=None):
    """Write :func:`yaml_string` of ``data`` to a file."""
    with open(filename, 'w') as f:
        f.write(yaml_string(data, header))


def read_yaml(filename):
    """The dict in a yaml file (``yaml.safe_load``)."""
    with open(filename) as f:
        return yaml.safe_load(f)


def read_json(filename):
    """The dict in a json file, e.g. a legacy rf_pipelines config."""
    with open(filename) as f:
        return json.load(f)
