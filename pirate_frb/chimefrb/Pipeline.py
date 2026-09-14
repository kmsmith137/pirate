"""Pipeline: run a list of chimefrb transforms in order on one block of data.

A python port of rf_pipelines::pipeline, the container the old CHIME FRB search's RFI chain
was built from. Its companion RfiMaskPipeline (rf_pipelines::wi_sub_pipeline; "wi" is the old
code's abbreviation for a (weights, intensity) pair, and survives in the name
GpuWiDownsamplingKernel) runs a list of transforms on a downsampled copy of the data
instead. Both are containers (GpuContainerBase, which supplies what they share) and
transforms themselves, so they nest.
"""

from .GpuContainerBase import GpuContainerBase
from .utils import check_json_keys


class Pipeline(GpuContainerBase):
    """An ordered list of transforms, run one after another on the same block of data.

    A port of rf_pipelines::pipeline, the container the old CHIME FRB search's RFI chain was
    built from. Each transform sees the output of the one before it. A Pipeline is itself a
    transform (a :class:`GpuContainerBase`, so it may hold other containers), so pipelines
    nest, and an :class:`RfiMaskPipeline` -- the old code's downsampled sub-pipeline -- can be
    one of its elements. ``launch()`` runs every transform in order; whichever arrays they
    modify, it modifies.

    One ``launch()`` processes exactly one (nbeams, nfreq, ntime) block. Assembling that block
    from the data source (four 1024-sample AssembledChunks for the production chain, whose
    clippers need 4096 samples) is the caller's job; the pipeline knows nothing about streams
    of chunks. Because every transform is stateless across blocks, running the pipeline block
    by block reproduces the old streaming pipeline exactly, provided ``ntime`` is a multiple of
    every transform's ``nt_chunk``.

    A Pipeline holds no per-launch state, so one instance may be launched on several
    streams at once, provided each stream has its own scratch array.

    Attributes (read-only):

    - ``transforms`` (tuple) -- the transforms, in launch order.
    - ``nbeams``, ``nfreq``, ``ntime`` -- the block shape, shared by every transform.
    - ``scratch_nelts`` (int) -- float32 scratch elements ``launch()`` needs: the largest of
      the transforms', since they run in sequence and share one array.
    """

    def __init__(self, transforms):
        """Create a Pipeline.

        Parameters
        ----------
        transforms : sequence
            One or more transforms (:class:`GpuTransform` subclasses), all with the same
            (nbeams, nfreq, ntime).
        """
        (transforms, (nbeams, nfreq, ntime)) = self.check_transforms(transforms)
        super().__init__(nbeams, nfreq, ntime, self.max_scratch_nelts(transforms))
        self.transforms = transforms

    def launch_checked(self, intensity, weights, scratch):
        # 'scratch' is the largest any of them needs, and the pipeline's stream is current
        # (GpuTransform.launch() made it so), which each transform's launch() defaults to.
        self.launch_transforms(intensity, weights, scratch)

    def to_yaml_dict(self):
        """``{'class_name': 'Pipeline', 'transforms': [...]}``, each element its own
        ``to_yaml_dict()``."""
        return {'class_name': 'Pipeline', 'transforms': self.transforms_yaml_list()}

    @classmethod
    def from_yaml_dict(cls, d, nbeams, nfreq, ntime, classes=None):
        """The inverse of :meth:`to_yaml_dict`, at the given geometry. ``classes`` is passed to
        the reader of each element (see :meth:`read_yaml_file`)."""
        cls.check_yaml_keys(d, ['transforms'])
        return cls(cls.transforms_from_yaml_list(d, nbeams, nfreq, ntime, classes))

    @classmethod
    def from_json_dict(cls, d, nbeams, nfreq, ntime, nds=1):
        """From a legacy rf_pipelines ``pipeline`` element. Its ``name`` is ignored; elements
        with no pirate counterpart that do not modify the data are skipped with a printed note
        (see ``chimefrb.utils``); ``nds`` is the data's time downsampling relative to the native
        stream, needed only by nested ``wi_sub_pipeline`` elements."""
        check_json_keys(d, 'pipeline', ['elements'])
        return cls(cls.transforms_from_json_elements(d['elements'], nbeams, nfreq, ntime, nds, 'pipeline'))

    def __repr__(self):
        return (f'Pipeline(nbeams={self.nbeams}, nfreq={self.nfreq}, ntime={self.ntime},'
                f' {len(self.transforms)} transform(s))')
