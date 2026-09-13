"""RfiMaskPipeline: run a list of chimefrb transforms on a downsampled copy of the data, and
feed only the resulting mask back to full resolution.

A python port of rf_pipelines::wi_sub_pipeline. See the class docstring, and WiPipeline.py
for the plain (undownsampled) container.
"""

import math

from .GpuTransformBase import GpuTransformBase
from .GpuPythonTransform import GpuPythonTransform
from .ReferenceWeightUpsampler import GpuWeightUpsampler
from .ReferenceWiDownsampler import GpuWiDownsampler
from .WiPipeline import WiPipeline, describe_lines
from .transform_io import (PIPELINE_YAML_HEADER, check_json_keys, read_json,
                           read_yaml, transform_from_json_dict, transform_from_yaml_dict,
                           write_yaml)


def _round_up(n, m):
    return ((n + m - 1) // m) * m


class RfiMaskPipeline(GpuPythonTransform):
    """Transforms run on a (Df, Dt)-downsampled copy of the data, whose mask is then applied
    to the full-resolution weights.

    A port of rf_pipelines::wi_sub_pipeline. The old CHIME FRB search ran most of its RFI
    chain this way: at 1024 channels instead of 16384, so that 108 clippers cost a sixteenth
    of what they would at full resolution, with the resulting mask upsampled back. One launch
    does three things:

    1. Downsample (intensity, weights) by (Df, Dt) with :class:`GpuWiDownsampler` into
       scratch. Downsampled weights are the SUM of a cell's weights, not the mean, so {0,1}
       weights become counts up to Df*Dt.
    2. Run the transforms, in order, on the downsampled pair.
    3. With :class:`GpuWeightUpsampler`, zero every full-resolution weight whose cell's
       downsampled weight is ``<= w_cutoff``; leave every other weight bit-identical.

    Two consequences worth knowing: ``launch()`` never modifies the full-resolution INTENSITY
    (only the weights, zeroed under every downsampled cell whose weight ends up
    ``<= w_cutoff`` and bit-identical elsewhere), and whatever the transforms do to the
    downsampled intensity is discarded with the scratch. The transforms' geometry is the
    INNER one, (nbeams, nfreq/Df, ntime/Dt); the pipeline's own is the full-resolution shape.

    An RfiMaskPipeline is itself a transform (a :class:`GpuTransformBase`), so it is normally
    one element of a :class:`WiPipeline`. Like a WiPipeline it holds no
    per-launch state, so one instance may be launched on several streams at once with one
    scratch per stream; and one ``launch()`` processes exactly one block, which the caller
    assembles.

    Attributes (read-only):

    - ``transforms`` (tuple) -- the transforms, in launch order, at the inner geometry.
    - ``Df``, ``Dt``, ``w_cutoff`` -- the constructor arguments.
    - ``nbeams``, ``nfreq``, ``ntime`` -- the FULL-RESOLUTION block shape.
    - ``scratch_nelts`` (int) -- float32 scratch elements ``launch()`` needs: the two
      downsampled arrays plus what the transforms need.
    """

    def __init__(self, transforms, Df, Dt, w_cutoff=0.0):
        """Create an RfiMaskPipeline.

        Parameters
        ----------
        transforms : sequence
            One or more transforms at the INNER geometry (nbeams, nfreq/Df, ntime/Dt), all
            alike. The inner nfreq and ntime must be multiples of 32 (GpuWiDownsampler's
            output tile).
        Df, Dt : int
            Downsampling factors in frequency and time, each >= 1 and not both 1: at (1, 1)
            the bracket would only copy the data, and it is not a no-op even then (the
            transforms' intensity changes would be discarded), so a WiPipeline should be
            used instead.
        w_cutoff : float, optional
            A full-resolution weight is zeroed when its cell's downsampled weight is
            ``<= w_cutoff`` (strictly: a downsampled weight equal to the cutoff masks). The
            production chain uses 0.
        """
        inner = WiPipeline(transforms)      # validates them, and owns the sequential launch
        (nbeams, nfreq_ds, ntime_ds) = (inner.nbeams, inner.nfreq, inner.ntime)

        for (name, x) in (('Df', Df), ('Dt', Dt)):
            if not (isinstance(x, int) and (x >= 1)):
                raise ValueError(f'RfiMaskPipeline: expected {name} to be an integer >= 1, got {x!r}')
        if (Df, Dt) == (1, 1):
            raise ValueError('RfiMaskPipeline: (Df, Dt) = (1, 1) is not supported. The bracket would'
                             ' only copy the data (and would still discard the transforms\' intensity'
                             ' changes); use a WiPipeline if you do not need the downsampling')
        w_cutoff = float(w_cutoff)
        if not (w_cutoff >= 0.0) or math.isnan(w_cutoff):
            raise ValueError(f'RfiMaskPipeline: expected w_cutoff >= 0, got {w_cutoff!r}')
        if (nfreq_ds % 32 != 0) or (ntime_ds % 32 != 0):
            raise ValueError(f'RfiMaskPipeline: the transforms\' (nfreq, ntime) = ({nfreq_ds}, {ntime_ds})'
                             f' must both be multiples of 32, the output tile of GpuWiDownsampler'
                             f' (the full-resolution shape is then a multiple of (32*Df, 32*Dt))')

        # Scratch layout: the downsampled intensity, the downsampled weights, then (at an
        # offset rounded to 64 elements, for the transforms' wide loads) the transforms' own
        # scratch.
        ds_nelts = nbeams * nfreq_ds * ntime_ds
        sub_offset = _round_up(2 * ds_nelts, 64)
        super().__init__(nbeams, nfreq_ds * Df, ntime_ds * Dt, sub_offset + inner.scratch_nelts)

        self._inner = inner
        self.transforms = inner.transforms
        self.Df = Df
        self.Dt = Dt
        self.w_cutoff = w_cutoff

        self._downsampler = GpuWiDownsampler(Df, Dt, False)
        self._upsampler = GpuWeightUpsampler(Df, Dt, w_cutoff)
        self._ds_shape = (nbeams, nfreq_ds, ntime_ds)
        self._ds_nelts = ds_nelts
        self._sub_offset = sub_offset

    def launch_checked(self, intensity, weights, scratch):
        # Steps 1-3 of the class docstring. The pipeline's stream is current
        # (GpuTransformBase.launch() made it so), and every launch below defaults to it.
        n = self._ds_nelts
        i_ds = scratch[:n].reshape(self._ds_shape)
        w_ds = scratch[n:2*n].reshape(self._ds_shape)
        sub = scratch[self._sub_offset:]

        self._downsampler.launch(i_ds, w_ds, intensity, weights)
        self._inner.launch(i_ds, w_ds, sub)
        self._upsampler.launch(weights, w_ds)

    def to_yaml_dict(self):
        """``{'class_name': 'RfiMaskPipeline', 'Df', 'Dt', 'w_cutoff', 'transforms': [...]}``."""
        return {'class_name': 'RfiMaskPipeline', 'Df': int(self.Df), 'Dt': int(self.Dt),
                'w_cutoff': float(self.w_cutoff),
                'transforms': [t.to_yaml_dict() for t in self.transforms]}

    @classmethod
    def from_yaml_dict(cls, d, nbeams, nfreq, ntime, classes=None):
        """The inverse of :meth:`to_yaml_dict`, at the given FULL-RESOLUTION geometry; the
        transforms are built at (nbeams, nfreq/Df, ntime/Dt). ``classes`` is passed to the
        reader of each element (see :meth:`read_yaml_file`)."""
        cls.check_yaml_keys(d, ['Df', 'Dt', 'w_cutoff', 'transforms'])
        (Df, Dt) = (d['Df'], d['Dt'])
        for (name, x) in (('Df', Df), ('Dt', Dt)):
            if not (isinstance(x, int) and (x >= 1)):
                raise ValueError(f'RfiMaskPipeline.from_yaml_dict: expected {name} to be an integer >= 1, got {x!r}')
        if (nfreq % Df != 0) or (ntime % Dt != 0):
            raise ValueError(f'RfiMaskPipeline.from_yaml_dict: (nfreq, ntime) = ({nfreq}, {ntime}) is not'
                             f' divisible by (Df, Dt) = ({Df}, {Dt})')
        if not (isinstance(d['transforms'], list) and (len(d['transforms']) > 0)):
            raise ValueError("RfiMaskPipeline.from_yaml_dict: 'transforms' must be a non-empty list")
        transforms = [transform_from_yaml_dict(e, nbeams, nfreq // Df, ntime // Dt, classes)
                      for e in d['transforms']]
        return cls(transforms, Df, Dt, d['w_cutoff'])

    @classmethod
    def from_json_dict(cls, d, nbeams, nfreq, ntime, nds=1):
        """From a legacy rf_pipelines ``wi_sub_pipeline`` element, at the given FULL-RESOLUTION
        geometry.

        The old object could be given (Df, Dt) directly, or as the downsampled channel count
        ``nfreq_out`` and the downsampled time resolution ``nds_out`` (relative to the native
        stream), with 0 meaning "not given"; both spellings are resolved as the old bind step
        did. ``nds`` is the time downsampling of the data arriving here (1 at top level). The
        old ``sub_pipeline`` is always a ``pipeline`` in practice; its elements become this
        object's transforms (skipping the inert unported ones, see ``transform_io``), and a
        bare transform is accepted as a list of one.
        """
        check_json_keys(d, 'wi_sub_pipeline', ['sub_pipeline', 'w_cutoff', 'nfreq_out', 'nds_out', 'Df', 'Dt'])
        (Df_j, Dt_j, nfreq_out, nds_out) = (int(d['Df']), int(d['Dt']), int(d['nfreq_out']), int(d['nds_out']))
        who = 'RfiMaskPipeline.from_json_dict'

        if min(Df_j, Dt_j, nfreq_out, nds_out) < 0:
            raise ValueError(f'{who}: Df, Dt, nfreq_out and nds_out must all be >= 0 (0 means unspecified)')
        if (Df_j == 0) and (nfreq_out == 0):
            raise ValueError(f'{who}: either nfreq_out or Df must be specified')
        if (Dt_j == 0) and (nds_out == 0):
            raise ValueError(f'{who}: either nds_out or Dt must be specified')

        if Df_j:
            Df = Df_j
            if nfreq_out and (nfreq != nfreq_out * Df):
                raise ValueError(f'{who}: nfreq={nfreq} does not match nfreq_out*Df = {nfreq_out}*{Df}')
        else:
            if nfreq % nfreq_out:
                raise ValueError(f'{who}: nfreq={nfreq} is not a multiple of nfreq_out={nfreq_out}')
            Df = nfreq // nfreq_out

        if Dt_j:
            Dt = Dt_j
            if nds_out and (nds_out != nds * Dt):
                raise ValueError(f'{who}: nds_out={nds_out} does not match nds*Dt = {nds}*{Dt}')
        else:
            if nds_out % nds:
                raise ValueError(f'{who}: nds_out={nds_out} is not a multiple of the incoming nds={nds}')
            Dt = nds_out // nds

        if (nfreq % Df) or (ntime % Dt):
            raise ValueError(f'{who}: (nfreq, ntime) = ({nfreq}, {ntime}) is not divisible by (Df, Dt) = ({Df}, {Dt})')

        sub = d['sub_pipeline']
        inner = (nbeams, nfreq // Df, ntime // Dt)
        if isinstance(sub, dict) and (sub.get('class_name') == 'pipeline'):
            check_json_keys(sub, 'pipeline', ['elements'])
            elements = sub['elements']
        else:
            elements = [sub]
        transforms = [transform_from_json_dict(e, *inner, nds=nds*Dt) for e in elements]
        transforms = [t for t in transforms if t is not None]
        if len(transforms) == 0:
            raise ValueError(f"{who}: the legacy 'sub_pipeline' has no element with a pirate counterpart")

        return cls(transforms, Df, Dt, float(d['w_cutoff']))

    @classmethod
    def read_yaml_file(cls, filename, *, nbeams, nfreq, ntime, classes=None):
        """Read a yaml file written by :meth:`write_yaml_file` whose top-level class_name is
        RfiMaskPipeline, building it for the given FULL-RESOLUTION data geometry.

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
        """Read a legacy rf_pipelines json file whose top-level element is a ``wi_sub_pipeline``,
        building it for the given FULL-RESOLUTION data geometry. Elements with no pirate counterpart that do
        not modify the data are skipped, with a note on stderr; see ``transform_io``."""
        return cls.from_json_dict(read_json(filename), nbeams, nfreq, ntime)

    def write_yaml_file(self, filename):
        """Write :meth:`to_yaml_dict` to a yaml file, after a comment saying how to read it
        (``transform_io.write_yaml``)."""
        write_yaml(filename, self.to_yaml_dict(), header=PIPELINE_YAML_HEADER)

    def describe(self):
        """A multi-line listing: this pipeline's parameters, then one indented line per
        transform (see :meth:`WiPipeline.describe`)."""
        return '\n'.join(describe_lines(self))

    def __repr__(self):
        return (f'RfiMaskPipeline(nbeams={self.nbeams}, nfreq={self.nfreq}, ntime={self.ntime},'
                f' Df={self.Df}, Dt={self.Dt}, w_cutoff={self.w_cutoff}, {len(self.transforms)} transform(s))')
