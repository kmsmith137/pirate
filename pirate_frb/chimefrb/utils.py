"""The interface every chimefrb transform follows, and the helpers behind it: turning a yaml
or legacy-json ``class_name`` into a class, the axis strings, and the yaml/json file functions.

WHAT A TRANSFORM IS. A subclass of :class:`GpuTransform`, which is anything a
:class:`Pipeline` or :class:`RfiMaskPipeline` can run. Five are C++ (GpuBadChannelMask,
GpuIntensityClipper, GpuStdDevClipper, GpuPolynomialDetrender, GpuSplineDetrender), on that
class directly; a transform written in python subclasses :class:`GpuPythonTransform`, which
is plain python on top of it -- ``ExamplePythonTransform`` (the worked example) and anything a
user writes. ``GpuPythonTransform``'s docstring says how. A transform that RUNS other
transforms subclasses :class:`GpuContainerBase` one level further down: the two pipeline
classes are those, and its ``from_yaml_dict`` / ``from_json_dict`` take the extra
``classes`` / ``nds`` arguments that the two dispatchers below forward to a container and
not to a leaf. Every transform has::

    nbeams, nfreq, ntime    ints: the (beams, channels, time samples) block it processes,
                            fixed at construction
    scratch_nelts           int: float32 scratch elements launch() needs; may be 0

    launch(intensity, weights, scratch, stream=None)     inherited from GpuTransform
    to_yaml_dict()                              -> dict
    from_yaml_dict(d, nbeams, nfreq, ntime)     classmethod -> instance

and, for the transforms with a legacy form only -- the five C++ transforms, the two
pipelines, and :class:`RfiMaskExtractor`, which is the old ``mask_counter`` in its
mask-saving role::

    from_json_dict(d, nbeams, nfreq, ntime)     classmethod -> instance

THE launch() CONTRACT. ``launch()`` is GpuTransform's; it checks everything below and
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
  and it must not alias the data arrays. One call processes exactly one block. The caller is
  responsible for passing a 128-byte-aligned array, which is not checked (a cupy allocation
  always is): the sub-arrays a transform carves out are aligned relative to the base, so a
  misaligned base misaligns all of them.
- ``stream`` is a cupy stream, or None for the current cupy stream. The launch is
  asynchronous on that stream; nothing synchronizes.

SERIALIZATION.

- ``to_yaml_dict()`` returns a plain dict of python scalars and lists that
  ``yaml.safe_dump`` accepts: a ``class_name`` key holding the transform's PYTHON class
  name (``GpuBadChannelMask``, ``Pipeline``, ...), then its semantic parameters. It never
  writes nbeams/nfreq/ntime (they are properties of the data, supplied when reading) and
  never writes performance knobs, so a file is not tied to a GPU.
- ``from_yaml_dict(d, nbeams, nfreq, ntime)`` checks that ``d['class_name']`` is the class's
  own name and that the other keys are EXACTLY the expected ones (a misspelled key must not
  silently become a default), then constructs. The ``check_yaml_keys()`` classmethod of
  GpuTransform does both checks.
- ``from_json_dict`` reads one element of the old rf_pipelines json, whose ``class_name`` is
  the legacy name (``badchannel_mask``, ``intensity_clipper``, ...). A WHOLE legacy file goes
  through :func:`legacy_chain_from_json`, not ``from_json_dict`` directly, because of the
  ``mask_counter`` rule described there.

HOW A class_name BECOMES A CLASS. :func:`resolve_class` looks the name up first among the
``classes`` the caller passed in (matched by ``__name__``; this is how a transform you wrote
yourself, in your own module or in a notebook, gets found), then as an attribute of the
``pirate_frb.chimefrb`` package. For the legacy json, a fixed table maps the old names to
class names first.
"""

import copy
import json

import yaml

from ..utils import atomic_print


# The CHIME band, which the legacy json never recorded: rf_pipelines' badchannel_mask read
# it from the stream at bind time. GpuBadChannelMask.from_json_dict() assumes it.
CHIME_FREQ_RANGE = (400.0, 800.0)


# float32 elements per 128-byte GPU cache line (constants::bytes_per_gpu_cache_line in C++).
SCRATCH_ALIGN = 32


def padded_scratch_nelts(nelts):
    """How many float32 elements a scratch sub-array of ``nelts`` OCCUPIES, once padded to a
    128-byte boundary.

    A transform that carves several sub-arrays out of one scratch array concatenates them
    back-to-back, and each one should start on a cache line, which is what a coalesced GPU
    load wants. Pad every piece with this, both when laying the array out and when adding up
    ``scratch_nelts``, so the two cannot drift apart. The C++ twin is
    ``padded_scratch_nelts()`` in include/pirate/chimefrb/Transform.hpp."""
    return ((nelts + SCRATCH_ALIGN - 1) // SCRATCH_ALIGN) * SCRATCH_ALIGN


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
        If nothing of that name is found, or what is found is not a GpuTransform subclass.
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
    from .cpp_transforms import GpuTransform   # here, not at module level: import cycle
    if not (isinstance(cls, type) and issubclass(cls, GpuTransform)):
        raise ValueError(f"{class_name!r} is not a transform class (it does not subclass GpuTransform)")
    return cls


def transform_from_yaml_dict(d, nbeams, nfreq, ntime, classes=None):
    """Build the transform that the yaml dict ``d`` describes, at the given geometry.

    Dispatches on ``d['class_name']`` (see :func:`resolve_class`). A CONTAINER -- a
    transform that runs other transforms, i.e. a :class:`GpuContainerBase` -- is the only
    kind whose ``from_yaml_dict`` takes ``classes``, and it is passed only to those, so that
    a leaf transform's factory keeps the plain four-argument signature. A container needs it
    in order to resolve its own elements, at any depth.
    """

    from .GpuContainerBase import GpuContainerBase     # here, not at module level: import cycle

    if not isinstance(d, dict) or ('class_name' not in d):
        raise ValueError(f"expected a dict with a 'class_name' key describing a transform, got {d!r}")

    cls = resolve_class(d['class_name'], classes)
    if issubclass(cls, GpuContainerBase):
        return cls.from_yaml_dict(d, nbeams, nfreq, ntime, classes=classes)
    return cls.from_yaml_dict(d, nbeams, nfreq, ntime)


# -------------------------------------------------------------------------------------------------
#
# Legacy json


# Legacy rf_pipelines class_name -> python class name, for everything that has a port. Closed
# set: a new transform has no legacy form. 'mask_counter' is special: only the one that
# legacy_chain_from_json() has marked as the extraction point becomes an RfiMaskExtractor,
# and the others are skipped (see transform_from_json_dict()).
LEGACY_JSON_CLASS_NAMES = {
    'pipeline': 'Pipeline',
    'wi_sub_pipeline': 'RfiMaskPipeline',
    'badchannel_mask': 'GpuBadChannelMask',
    'std_dev_clipper': 'GpuStdDevClipper',
    'intensity_clipper': 'GpuIntensityClipper',
    'polynomial_detrender': 'GpuPolynomialDetrender',
    'spline_detrender': 'GpuSplineDetrender',
    'mask_counter': 'RfiMaskExtractor',
}

# The key legacy_chain_from_json() adds to the one 'mask_counter' element that is the chain's
# extraction point. No old file carries it.
LEGACY_EXTRACT_KEY = 'pirate_extract'

# Legacy classes with no port that DO NOT CHANGE THE DATA -- writers and consumers that read
# the stream without touching it. transform_from_json_dict() skips these, with a printed
# note. Anything else unported raises, so that a transform that does change the data
# (mask_expander, chime_16k_derippler, noise_filler, ...) is refused rather than silently
# dropped.
IGNORED_JSON_CLASSES = frozenset([
    'chime_slow_pulsar_writer', 'chime_file_writer',
    'chime_assembled_chunk_file_writer', 'chime_packetizer', 'bonsai_dedisperser',
    'plotter_transform',
])


def transform_from_json_dict(d, nbeams, nfreq, ntime, nds=1):
    """Build the transform that one element of a legacy rf_pipelines json describes, at the
    given geometry -- or return None if the element is one of :data:`IGNORED_JSON_CLASSES`,
    or a ``mask_counter`` that is not the marked extraction point.

    A whole file should go through :func:`legacy_chain_from_json`, which does the marking;
    called on an unmarked tree, this skips every ``mask_counter``. ``nds`` is the time
    downsampling of the data relative to the native stream (1 at top level; ``nds*Dt`` inside
    a wi_sub_pipeline), which only a container needs, to resolve a ``wi_sub_pipeline`` given
    as ``nds_out``.
    """

    from .GpuContainerBase import GpuContainerBase     # here, not at module level: import cycle

    if not isinstance(d, dict) or ('class_name' not in d):
        raise ValueError(f"expected a legacy json element with a 'class_name' key, got {d!r}")

    name = d['class_name']
    if (name == 'mask_counter') and not d.get(LEGACY_EXTRACT_KEY, False):
        atomic_print(f"transform_from_json_dict: skipping a 'mask_counter' element (where="
                     f"{d.get('where')!r}): in the old code it only fed monitoring statistics."
                     f" The LAST mask_counter of a chain is its extraction point and becomes an"
                     f" RfiMaskExtractor (see legacy_chain_from_json)", fd=2)
        return None
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
    if issubclass(cls, GpuContainerBase):
        return cls.from_json_dict(d, nbeams, nfreq, ntime, nds=nds)
    return cls.from_json_dict(d, nbeams, nfreq, ntime)


def _legacy_leaves(d):
    """The leaf elements of a legacy json tree, in document order: a 'pipeline' yields its
    elements' leaves, a 'wi_sub_pipeline' its sub-pipeline's, anything else itself."""

    name = d.get('class_name') if isinstance(d, dict) else None
    if name == 'pipeline':
        for e in d.get('elements', []):
            yield from _legacy_leaves(e)
    elif name == 'wi_sub_pipeline':
        yield from _legacy_leaves(d.get('sub_pipeline', {}))
    else:
        yield d


def legacy_chain_from_json(d, nbeams, nfreq, ntime):
    """Build the transform that a WHOLE legacy rf_pipelines json describes, at the given
    geometry: the entry point for a legacy file (``Pipeline.read_json_file`` and
    ``pirate_frb cfrb json2yaml`` both come here).

    THE LAST ``mask_counter`` IS THE EXTRACTION POINT. The old CHIME L1 server saved the RFI
    mask at the last ``mask_counter`` of its chain, in document order, and used the others
    for monitoring statistics only. So this marks the last one (a private key on a COPY of
    the tree; the caller's dict is untouched), which :func:`transform_from_json_dict` then
    turns into an :class:`RfiMaskExtractor`, and skips the rest with a note.

    The L1 server also refused a chain in which a clipper followed that mask_counter. Pirate
    runs such a chain -- the mask is whatever the extractor sees -- but prints a note on
    stderr, since the mask then differs from what the old server would have saved.

    Returns the top-level transform, or None if the top-level element has no pirate
    counterpart (a lone writer, say).
    """

    d = copy.deepcopy(d)
    leaves = list(_legacy_leaves(d))
    counters = [i for (i, e) in enumerate(leaves) if e.get('class_name') == 'mask_counter']

    if counters:
        last = counters[-1]
        leaves[last][LEGACY_EXTRACT_KEY] = True

        clippers_after = [e['class_name'] for e in leaves[last+1:]
                          if 'clipper' in str(e.get('class_name', ''))]
        if clippers_after:
            atomic_print(f"legacy_chain_from_json: {len(clippers_after)} clipper(s) follow the last"
                         f" mask_counter ({clippers_after}). The old L1 server refused such a"
                         f" chain; pirate takes the mask at the RfiMaskExtractor, before them",
                         fd=2)

    chain = transform_from_json_dict(d, nbeams, nfreq, ntime)

    if counters and (chain is not None) and (chain.get_mask_extractor() is None):
        raise RuntimeError('legacy_chain_from_json: internal error: the marked mask_counter did'
                           ' not become an RfiMaskExtractor')
    return chain


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


# The one place the OLD spelling of an axis survives. Everywhere else -- a constructor
# argument, a yaml file, a C++ printout -- an axis is 'freq', 'time' or 'none'.
_AXIS_FROM_JSON = {'AXIS_FREQ': 'freq', 'AXIS_TIME': 'time', 'AXIS_NONE': 'none'}


def axis_from_json(s):
    """The axis name, from the legacy json spellings 'AXIS_FREQ', 'AXIS_TIME', 'AXIS_NONE'."""
    if s not in _AXIS_FROM_JSON:
        raise ValueError(f"expected a legacy axis string, one of {sorted(_AXIS_FROM_JSON)}, got {s!r}")
    return _AXIS_FROM_JSON[s]


# -------------------------------------------------------------------------------------------------
#
# yaml / json files. Plain functions on plain dicts; Pipeline.write_yaml_file() and friends
# are thin wrappers around these.


def pipeline_yaml_header(class_name, from_json=False):
    """The comment that Pipeline/RfiMaskPipeline.write_yaml_file() and 'pirate_frb cfrb
    json2yaml' put at the top of a file: what the format is, and the line of python that
    reads the file back.

    ``class_name`` must be the file's top-level ``class_name`` ('Pipeline' or
    'RfiMaskPipeline'), since that is the class whose ``read_yaml_file()`` accepts the
    file -- ``GpuTransform.check_yaml_keys()`` rejects a file whose top-level class is a
    different one, so a header naming the wrong class would send the reader into that
    error. ``from_json`` adds a line saying the chain was converted from the legacy
    rf_pipelines json format.

    The result is ``#`` comment lines ending in a blank line, ready to be the ``header`` of
    :func:`yaml_string` or :func:`write_yaml`."""

    lines = ['This is the "pirate" yaml format for a CHIMEFRB transform chain.']

    if from_json:
        lines += ['It was converted from the "rf_pipelines" json format,'
                  " using 'pirate_frb cfrb json2yaml'."]

    lines += ['To read it:',
              f'  pipeline = pirate_frb.chimefrb.{class_name}.read_yaml_file'
              '(filename, nbeams=..., nfreq=..., ntime=...)']

    return ''.join(f'# {line}\n' for line in lines) + '\n'


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
    ``header`` (a string of ``#`` comment lines, ending in a newline -- see
    :func:`pipeline_yaml_header`) if one is given.

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
