"""The interface every chimefrb transform follows, and the helpers behind it: turning a yaml
or legacy-json ``class_name`` into a class, the axis strings, and the yaml/json file functions.

WHAT A TRANSFORM IS. A subclass of :class:`GpuTransformBase`, which is anything a
:class:`WiPipeline` or :class:`RfiMaskPipeline` can run. Five are C++ (GpuBadChannelMask,
GpuIntensityClipper, GpuStdDevClipper, GpuPolynomialDetrender, GpuSplineDetrender), on that
class directly; a transform written in python subclasses :class:`GpuPythonTransform`, which
is plain python on top of it -- the two pipeline classes (transforms themselves, so that
they nest), ``ExampleCupyTransform`` (the worked example), and anything a user writes.
``GpuPythonTransform``'s docstring says how. Every transform has::

    nbeams, nfreq, ntime    ints: the (beams, channels, time samples) block it processes,
                            fixed at construction
    scratch_nelts           int: float32 scratch elements launch() needs; may be 0

    launch(intensity, weights, scratch, stream=None)     inherited from GpuTransformBase
    to_yaml_dict()                              -> dict
    from_yaml_dict(d, nbeams, nfreq, ntime)     classmethod -> instance

and, for the five C++ transforms and the two pipelines only (the "legacy" transforms,
which the old rf_pipelines json format describes)::

    from_json_dict(d, nbeams, nfreq, ntime)     classmethod -> instance

THE launch() CONTRACT. ``launch()`` is GpuTransformBase's; it checks everything below and
raises RuntimeError, naming the transform, on a violation, before running the transform's
``launch_checked()``.

- ``intensity`` and ``weights`` are cupy float32 arrays of shape (nbeams, nfreq, ntime),
  C-contiguous, and distinct. Either or both may be modified in place; which one is the
  transform's business, stated in its docstring. Weights are nonnegative, and a zero weight
  means "ignore this sample".
- ``scratch`` is a REQUIRED positional argument: a 1-d C-contiguous cupy float32 array with
  at least ``scratch_nelts`` elements (any array, even an empty one, when scratch_nelts is
  0), or the literal ``None``, which allocates one. There is deliberately no default, so
  that a caller who wants the allocation says ``scratch=None`` and can see it in the call.
  Contents on entry are ignored and on exit are garbage; a transform uses a prefix of it,
  and it must not alias the data arrays. One call processes exactly one block.
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
  silently become a default), then constructs. The ``check_yaml_keys()`` classmethod of
  GpuTransformBase does both checks.
- ``from_json_dict`` reads one element of the old rf_pipelines json, whose ``class_name`` is
  the legacy name (``badchannel_mask``, ``intensity_clipper``, ...).

HOW A class_name BECOMES A CLASS. :func:`resolve_class` looks the name up first among the
``classes`` the caller passed in (matched by ``__name__``; this is how a transform you wrote
yourself, in your own module or in a notebook, gets found), then as an attribute of the
``pirate_frb.chimefrb`` package. For the legacy json, a fixed table maps the old names to
class names first.
"""

import json

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
        If nothing of that name is found, or what is found is not a GpuTransformBase subclass.
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
    from .GpuTransformBase import GpuTransformBase     # here, not at module level: import cycle
    if not (isinstance(cls, type) and issubclass(cls, GpuTransformBase)):
        raise ValueError(f"{class_name!r} is not a transform class (it does not subclass GpuTransformBase)")
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
# Axis strings (yaml and legacy json)


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
# yaml / json files. Plain functions on plain dicts; WiPipeline.write_yaml_file() and friends
# are thin wrappers around these.


# The comment WiPipeline/RfiMaskPipeline.write_yaml_file() put at the top of a file.
PIPELINE_YAML_HEADER = ('# A pirate_frb.chimefrb transform chain. Read with\n'
                        '#   WiPipeline.read_yaml_file(filename, nbeams=..., nfreq=..., ntime=...)\n'
                        '# (or RfiMaskPipeline.read_yaml_file, if that is the top-level class_name).\n')


# Line width that _YamlDumper wraps a transform's parameters at. A soft target: pyyaml
# breaks at the last comma that fits, so a line can overshoot it a little (to 105 columns
# on the old search's production chain, whose longest transform would be 136 on one line).
YAML_WIDTH = 100

# Columns per level of nesting. Note pyyaml pads the '-' of a block sequence entry out to
# the full indent, so a nested transform in block style starts '-   class_name:'.
YAML_INDENT = 4


class _YamlDumper(yaml.SafeDumper):
    """yaml.SafeDumper, except that anything whose entries are all scalars is written inline:
    a list (``freq_range: [400.0, 800.0]``, one ``[lo, hi]`` per mask range), and a mapping,
    which is how a transform's parameters end up on one line

        - {class_name: GpuStdDevClipper, nt_chunk: 4096, axis: freq, sigma: 3.0, Df: 1, ...}

    rather than nine. Anything holding a list or another mapping -- the two pipeline classes,
    and GpuBadChannelMask with its list of mask ranges -- stays in block style, so the
    structure of a chain is still one element per line."""


def _is_scalar(x):
    return isinstance(x, (int, float, str, bool)) or (x is None)


def _represent_list(dumper, data):
    flow = (len(data) > 0) and all(_is_scalar(x) for x in data)
    return dumper.represent_sequence('tag:yaml.org,2002:seq', data, flow_style=flow)


def _represent_dict(dumper, data):
    flow = (len(data) > 0) and all(_is_scalar(v) for v in data.values())
    return dumper.represent_mapping('tag:yaml.org,2002:map', data, flow_style=flow)


_YamlDumper.add_representer(list, _represent_list)
_YamlDumper.add_representer(dict, _represent_dict)


def yaml_string(data, header=None, width=None):
    """``data`` (a plain dict, e.g. a ``to_yaml_dict()``) as yaml text, keys in their natural
    order and each transform's parameters inline (:class:`_YamlDumper`), preceded by
    ``header`` (a string of ``#`` comment lines, ending in a newline) if one is given.

    ``width`` is the wrap column, defaulting to :data:`YAML_WIDTH`. Note that pyyaml ignores
    a width of 4 or less (it falls back to 80), so callers that expose this should refuse a
    small one rather than pass it through."""
    s = yaml.dump(data, Dumper=_YamlDumper, sort_keys=False, indent=YAML_INDENT,
                  width=(YAML_WIDTH if (width is None) else width))
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
