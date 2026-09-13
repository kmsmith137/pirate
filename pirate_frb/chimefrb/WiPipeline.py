"""WiPipeline: run a list of chimefrb transforms in order on one block of data.

A python port of rf_pipelines::pipeline, the container the old CHIME FRB search's RFI chain
was built from. ("wi" is the old code's abbreviation for a (weights, intensity) pair.) Its
companion RfiMaskPipeline (rf_pipelines::wi_sub_pipeline) runs a list of transforms on a
downsampled copy of the data instead; both are transforms themselves (subclasses of
GpuTransformBase, like every transform), so they nest.
"""

from .GpuTransformBase import GpuTransformBase
from .transform_io import (PIPELINE_YAML_HEADER, check_json_keys, check_yaml_keys, read_json,
                           read_yaml, transform_from_json_dict, transform_from_yaml_dict,
                           write_yaml)


def check_transforms(transforms, who):
    """Validate a sequence of transforms that are to share one block: each is a
    :class:`GpuTransformBase` (which guarantees the geometry attributes and the checked
    ``launch()``), and all have the same geometry. Returns
    ``(tuple_of_transforms, (nbeams, nfreq, ntime))``."""

    transforms = tuple(transforms)
    if len(transforms) == 0:
        raise ValueError(f'{who}: expected at least one transform')

    geometry = None
    for (i, t) in enumerate(transforms):
        what = f'{who}: transforms[{i}] ({type(t).__name__})'
        if not isinstance(t, GpuTransformBase):
            raise TypeError(f"{what} is not a transform: it does not subclass GpuTransformBase; see"
                            f" pirate_frb.chimefrb.transform_io for what a transform is")

        g = (t.nbeams, t.nfreq, t.ntime)
        if geometry is None:
            geometry = g
        elif g != geometry:
            raise ValueError(f'{what} has geometry (nbeams, nfreq, ntime) = {g}, but transforms[0]'
                             f' has {geometry}; every transform in a pipeline processes the same block')

    return (transforms, geometry)


def describe_lines(transform, depth=0):
    """The lines of :meth:`WiPipeline.describe`: one per transform, with its yaml parameters,
    recursing into anything that has a ``transforms`` attribute."""

    pad = '  ' * depth
    d = transform.to_yaml_dict()
    if hasattr(transform, 'transforms'):
        params = {k: v for (k, v) in d.items() if k not in ('class_name', 'transforms')}
        head = f"{pad}{d['class_name']}" + (f" {params}" if params else '') \
            + f"   [{len(transform.transforms)} transform(s), ({transform.nbeams}, {transform.nfreq}, {transform.ntime})]"
        lines = [head]
        for t in transform.transforms:
            lines += describe_lines(t, depth + 1)
        return lines

    params = {k: v for (k, v) in d.items() if k != 'class_name'}
    return [f"{pad}{d['class_name']} {params}"]


class WiPipeline(GpuTransformBase):
    """An ordered list of transforms, run one after another on the same block of data.

    A port of rf_pipelines::pipeline, the container the old CHIME FRB search's RFI chain was
    built from ("wi" is the old code's abbreviation for a (weights, intensity) pair). Each
    transform sees the output of the one before it. A WiPipeline is itself a transform (a
    :class:`GpuTransformBase`, like every transform), so pipelines nest, and an
    :class:`RfiMaskPipeline` -- the old code's downsampled sub-pipeline -- can be one of its
    elements. ``launch()`` runs every transform in order; whichever arrays they modify, it
    modifies.

    One ``launch()`` processes exactly one (nbeams, nfreq, ntime) block. Assembling that block
    from the data source (four 1024-sample AssembledChunks for the production chain, whose
    clippers need 4096 samples) is the caller's job; the pipeline knows nothing about streams
    of chunks. Because every transform is stateless across blocks, running the pipeline block
    by block reproduces the old streaming pipeline exactly, provided ``ntime`` is a multiple of
    every transform's ``nt_chunk``.

    A WiPipeline holds no per-launch state, so one instance may be launched on several
    streams at once, provided each stream has its own scratch array.

    Attributes (read-only):

    - ``transforms`` (tuple) -- the transforms, in launch order.
    - ``nbeams``, ``nfreq``, ``ntime`` -- the block shape, shared by every transform.
    - ``scratch_nelts`` (int) -- float32 scratch elements ``launch()`` needs: the largest of
      the transforms', since they run in sequence and share one array.
    """

    def __init__(self, transforms):
        """Create a WiPipeline.

        Parameters
        ----------
        transforms : sequence
            One or more transforms (:class:`GpuTransformBase` subclasses), all with the same
            (nbeams, nfreq, ntime).
        """
        (transforms, (nbeams, nfreq, ntime)) = check_transforms(transforms, 'WiPipeline')
        super().__init__(nbeams, nfreq, ntime, max(t.scratch_nelts for t in transforms))
        self.transforms = transforms

    def launch_checked(self, intensity, weights, scratch):
        # The pipeline's stream is current (GpuTransformBase.launch() made it so), and each
        # transform's launch() defaults to it. 'scratch' is the largest any of them needs.
        for t in self.transforms:
            t.launch(intensity, weights, scratch)

    def to_yaml_dict(self):
        """``{'class_name': 'WiPipeline', 'transforms': [...]}``, each element its own
        ``to_yaml_dict()``."""
        return {'class_name': 'WiPipeline', 'transforms': [t.to_yaml_dict() for t in self.transforms]}

    @classmethod
    def from_yaml_dict(cls, d, nbeams, nfreq, ntime, classes=None):
        """The inverse of :meth:`to_yaml_dict`, at the given geometry. ``classes`` is passed to
        the reader of each element (see :meth:`read_yaml_file`)."""
        check_yaml_keys(d, 'WiPipeline', ['transforms'])
        if not (isinstance(d['transforms'], list) and (len(d['transforms']) > 0)):
            raise ValueError("WiPipeline.from_yaml_dict: 'transforms' must be a non-empty list")
        return cls([transform_from_yaml_dict(e, nbeams, nfreq, ntime, classes) for e in d['transforms']])

    @classmethod
    def from_json_dict(cls, d, nbeams, nfreq, ntime, nds=1):
        """From a legacy rf_pipelines ``pipeline`` element. Its ``name`` is ignored; elements
        with no pirate counterpart that do not modify the data are skipped with a printed note
        (see ``transform_io``); ``nds`` is the data's time downsampling relative to the native
        stream, needed only by nested ``wi_sub_pipeline`` elements."""
        check_json_keys(d, 'pipeline', ['elements'])
        transforms = [transform_from_json_dict(e, nbeams, nfreq, ntime, nds) for e in d['elements']]
        transforms = [t for t in transforms if t is not None]
        if len(transforms) == 0:
            raise ValueError("WiPipeline.from_json_dict: the legacy 'pipeline' has no element with a pirate counterpart")
        return cls(transforms)

    @classmethod
    def read_yaml_file(cls, filename, *, nbeams, nfreq, ntime, classes=None):
        """Read a yaml file written by :meth:`write_yaml_file`, building the pipeline for the
        given data geometry.

        Parameters
        ----------
        filename : str
        nbeams, nfreq, ntime : int
            The block shape the pipeline will be launched on. A yaml file records no
            geometry; the same file serves any geometry its transforms accept.
        classes : sequence of type or None, optional
            Transform classes of your own that the file may name (matched by class name);
            anything in ``pirate_frb.chimefrb`` is found without this. See
            ``pirate_frb.chimefrb.transform_io``.
        """
        return cls.from_yaml_dict(read_yaml(filename), nbeams, nfreq, ntime, classes=classes)

    @classmethod
    def read_json_file(cls, filename, *, nbeams, nfreq, ntime):
        """Read a legacy rf_pipelines json file (one written by the old ``jsonize()``), building
        the pipeline for the given data geometry. Elements with no pirate counterpart that do
        not modify the data are skipped, with a note on stderr; see ``transform_io``."""
        return cls.from_json_dict(read_json(filename), nbeams, nfreq, ntime)

    def write_yaml_file(self, filename):
        """Write :meth:`to_yaml_dict` to a yaml file, after a comment saying how to read it
        (``transform_io.write_yaml``)."""
        write_yaml(filename, self.to_yaml_dict(), header=PIPELINE_YAML_HEADER)

    def describe(self):
        """A multi-line listing of the pipeline: one indented line per transform, with its yaml
        parameters, nested pipelines indented under their parent."""
        return '\n'.join(describe_lines(self))

    def __repr__(self):
        return (f'WiPipeline(nbeams={self.nbeams}, nfreq={self.nfreq}, ntime={self.ntime},'
                f' {len(self.transforms)} transform(s))')
