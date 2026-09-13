"""GpuContainerBase: the base class of every chimefrb transform that runs OTHER transforms.

WiPipeline and RfiMaskPipeline are the two in this package; a user may write another. See
the class docstring for what a subclass supplies and what it inherits.
"""

from .GpuPythonTransform import GpuPythonTransform
from .GpuTransformBase import GpuTransformBase
from .transform_io import (PIPELINE_YAML_HEADER, read_json, read_yaml,
                           transform_from_json_dict, transform_from_yaml_dict, write_yaml)


class GpuContainerBase(GpuPythonTransform):
    """Base class of a transform that runs other transforms.

    A container is a transform whose work is done by a list of other transforms: it holds
    them in ``self.transforms``, and its ``launch_checked()`` runs them, possibly with
    something of its own around them. :class:`WiPipeline` runs them in order on the caller's
    block; :class:`RfiMaskPipeline` runs them on a downsampled copy and feeds the resulting
    mask back. A container is itself a transform, so containers nest.

    A subclass supplies its ``__init__`` (which calls :meth:`check_transforms` first, then
    ``super().__init__()``, then sets ``self.transforms``), its ``launch_checked()``, and
    its yaml and legacy-json methods. This class supplies the pieces those share: the
    validation, the sequential launch, the yaml list of elements, the two element-building
    helpers, and the three file methods.

    WHY THIS CLASS IS A SEPARATE ONE, and not just shared code. A container's
    ``from_yaml_dict()`` and ``from_json_dict()`` take an argument that a leaf transform's
    do not -- ``classes`` and ``nds`` respectively -- because a container has to pass them
    down when it builds its elements. ``transform_io.transform_from_yaml_dict()`` and
    ``transform_from_json_dict()`` decide whether to supply that argument by testing
    ``issubclass(cls, GpuContainerBase)``, so a container written outside this package is
    recognized exactly as the two here are. Deriving from :class:`GpuPythonTransform` and
    merely LOOKING like a container is not enough: the argument would be dropped, and the
    container's own elements would then fail to resolve.

    So a container's factory signatures are::

        from_yaml_dict(cls, d, nbeams, nfreq, ntime, classes=None)
        from_json_dict(cls, d, nbeams, nfreq, ntime, nds=1)

    which is the one way they differ from the signatures in
    :class:`GpuPythonTransform`'s docstring. They are not declared here (there is nothing
    useful a base could do in them), just required.

    Attributes (read-only):

    - ``transforms`` (tuple) -- the elements, in launch order, at whatever geometry the
      container runs them. Set by the subclass's ``__init__``.
    - ``nbeams``, ``nfreq``, ``ntime``, ``scratch_nelts`` -- as for any transform.
    """

    # ---------------------------------------------------------------------------------
    #
    # Construction. A subclass calls these BEFORE super().__init__(), since what they
    # return is what the constructor needs -- and an attribute cannot be set before the
    # base class is initialized anyway.

    @classmethod
    def check_transforms(cls, transforms):
        """Validate a sequence of transforms that are to share one block: each is a
        :class:`GpuTransformBase` (which guarantees the geometry attributes and the checked
        ``launch()``), and all have the same geometry. Returns
        ``(tuple_of_transforms, (nbeams, nfreq, ntime))``.

        Raises
        ------
        ValueError
            If the sequence is empty, or the geometries disagree.
        TypeError
            If an element is not a transform.
        """

        who = cls.__name__
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

    @staticmethod
    def max_scratch_nelts(transforms):
        """The largest ``scratch_nelts`` among ``transforms``: what a container needs for
        them, since they run one after another and share one scratch array."""
        return max(t.scratch_nelts for t in transforms)

    # ---------------------------------------------------------------------------------
    #
    # Launching

    def launch_transforms(self, intensity, weights, scratch):
        """Launch every element of ``self.transforms``, in order, on the current stream.

        Called from a subclass's ``launch_checked()``, so the stream is already current.
        Each element checks its own arguments, which is why a container does not check them
        again: a too-small scratch or a wrong-shaped array is reported by the first element,
        naming that element."""
        for t in self.transforms:
            t.launch(intensity, weights, scratch)

    # ---------------------------------------------------------------------------------
    #
    # The yaml and legacy-json forms of the element list

    def transforms_yaml_list(self):
        """The ``transforms`` value of a container's yaml form: each element's own
        ``to_yaml_dict()``."""
        return [t.to_yaml_dict() for t in self.transforms]

    @classmethod
    def transforms_from_yaml_list(cls, d, nbeams, nfreq, ntime, classes):
        """The inverse: the transforms that ``d['transforms']`` describes, built at the
        given geometry -- which is the geometry the container RUNS them at, so a container
        that downsamples passes the inner one. ``classes`` is passed down, so that a
        transform of the caller's own is found at any depth."""

        if not (isinstance(d['transforms'], list) and (len(d['transforms']) > 0)):
            raise ValueError(f"{cls.__name__}.from_yaml_dict: 'transforms' must be a non-empty list")
        return [transform_from_yaml_dict(e, nbeams, nfreq, ntime, classes) for e in d['transforms']]

    @classmethod
    def transforms_from_json_elements(cls, elements, nbeams, nfreq, ntime, nds, what):
        """The transforms that a list of legacy rf_pipelines elements describes, built at the
        given geometry. Elements with no pirate counterpart that do not modify the data are
        skipped (``transform_io.transform_from_json_dict`` returns None and prints a note);
        ``what`` is the legacy key they came from, named in the message if that leaves
        nothing at all."""

        transforms = [transform_from_json_dict(e, nbeams, nfreq, ntime, nds) for e in elements]
        transforms = [t for t in transforms if t is not None]
        if len(transforms) == 0:
            raise ValueError(f"{cls.__name__}.from_json_dict: the legacy {what!r} has no element"
                             f" with a pirate counterpart")
        return transforms

    # ---------------------------------------------------------------------------------
    #
    # Files

    @classmethod
    def read_yaml_file(cls, filename, *, nbeams, nfreq, ntime, classes=None):
        """Read a yaml file written by :meth:`write_yaml_file`, building the container for
        the given data geometry.

        Parameters
        ----------
        filename : str
        nbeams, nfreq, ntime : int
            The block shape the container will be launched on, which for a container that
            downsamples is the FULL-RESOLUTION shape. A yaml file records no geometry; the
            same file serves any geometry its transforms accept.
        classes : sequence of type or None, optional
            Transform classes of your own that the file may name (matched by class name);
            anything in ``pirate_frb.chimefrb`` is found without this. See
            ``pirate_frb.chimefrb.transform_io``.
        """
        return cls.from_yaml_dict(read_yaml(filename), nbeams, nfreq, ntime, classes=classes)

    @classmethod
    def read_json_file(cls, filename, *, nbeams, nfreq, ntime):
        """Read a legacy rf_pipelines json file (one written by the old ``jsonize()``) whose
        top-level element is this class's, building it for the given data geometry. Elements
        with no pirate counterpart that do not modify the data are skipped, with a note on
        stderr; see ``transform_io``."""
        return cls.from_json_dict(read_json(filename), nbeams, nfreq, ntime)

    def write_yaml_file(self, filename):
        """Write :meth:`to_yaml_dict` to a yaml file, after a comment saying how to read it
        (``transform_io.write_yaml``)."""
        write_yaml(filename, self.to_yaml_dict(), header=PIPELINE_YAML_HEADER)
