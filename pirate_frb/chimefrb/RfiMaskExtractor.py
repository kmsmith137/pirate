"""RfiMaskExtractor: the transform that marks where in a chain the RFI mask is taken, and
packs it.

The pirate counterpart of the old rf_pipelines mask_counter in its one interesting role --
the point at which the CHIME L1 server captured the mask it saved -- with the packing done
by RfiMaskPackingKernel. See the class docstring for how the mask leaves a launch.
"""

from .GpuPythonTransform import GpuPythonTransform
from .RfiMaskPackingKernel import RfiMaskPackingKernel
from .utils import check_json_keys


class RfiMaskExtractor(GpuPythonTransform):
    """Packs the weights at its position in a chain into a bit-packed RFI mask.

    ITS POSITION IN THE CHAIN DEFINES THE MASK: the mask is ``weights > 0`` as they are when
    the chain reaches this transform, at this transform's resolution -- inside an
    :class:`RfiMaskPipeline`, the downsampled one. Nothing is modified; the intensity and
    weights pass through untouched. The packing is :class:`RfiMaskPackingKernel`'s: one row
    per (beam, channel), LSB-first within each byte, a SET bit meaning GOOD data, which is the
    layout of a chimefrb data file's ``rfi_mask``.

    This is the pirate form of the old ``mask_counter`` in its mask-saving role. The CHIME L1
    server saved the mask at the LAST mask_counter of its chain, so ``pirate_frb cfrb
    json2yaml`` turns that element into one of these and skips the others, which only fed
    monitoring statistics (see ``pirate_frb.chimefrb.utils``).

    WHERE THE MASK GOES. A transform's ``launch()`` carries only (intensity, weights, scratch,
    stream), so the destination is PLANTED beforehand with :meth:`set_rfi_mask`: a cupy uint8
    array of shape ``(m, nbeams, nfreq, ntime // (8*m))``, C-contiguous. A launch packs window
    ``i`` of the block's time axis, ``ntime/m`` samples wide, into slice ``i``, so that a driver
    feeding the chain from 1024-sample chunks plants ``m = ntime/1024`` and gets one contiguous
    mask per chunk without this transform knowing what a chunk is; ``m = 1`` is the whole
    block's mask. Each window must be a multiple of 1024 samples (the packing kernel's
    tiling). Launching with nothing planted raises: the mask is this transform's only output,
    so a chain containing one is always run by something that wants the mask, or that plants
    a throwaway array to say it does not (``pirate_frb cfrb time_pipeline`` does the latter).

    A planted array makes the chain not stream-reentrant: plant and launch from one thread at
    a time. :class:`ChimePreDedisperser` is the driver that does all of this for a stream of
    :class:`AssembledChunk` files.

    Attributes (read-only):

    - ``nbeams``, ``nfreq``, ``ntime`` -- the block shape, from the constructor; ``ntime`` is a
      multiple of 1024.
    - ``scratch_nelts`` -- always 0.
    - ``rfi_mask`` -- the planted destination, or None.
    """

    def __init__(self, nbeams, nfreq, ntime):
        """Create an RfiMaskExtractor.

        Parameters
        ----------
        nbeams, nfreq, ntime : int
            The block shape at this transform's position in the chain; ``ntime`` must be a
            multiple of 1024.

        Raises
        ------
        ValueError
            If ``ntime`` is not a multiple of 1024.
        """
        super().__init__(nbeams, nfreq, ntime)

        if ntime % 1024 != 0:
            raise ValueError(f'RfiMaskExtractor: expected ntime to be a multiple of 1024 (the tiling'
                             f' of RfiMaskPackingKernel), got ntime={ntime}')

        self._rfi_mask = None      # the planted destination (see the class docstring)
        self._packer = None        # RfiMaskPackingKernel for the current window width

    @property
    def rfi_mask(self):
        """The planted destination (see :meth:`set_rfi_mask`), or None."""
        return self._rfi_mask

    def set_rfi_mask(self, rfi_mask):
        """Plant the array that later launches pack the mask into, or unplant with None.

        Parameters
        ----------
        rfi_mask : cupy.ndarray or None
            uint8, C-contiguous, on the GPU, of shape ``(m, nbeams, nfreq, ntime // (8*m))``
            for some ``m >= 1`` dividing ``ntime`` into windows that are multiples of 1024
            samples. Window ``i`` of the time axis is packed into ``rfi_mask[i]``. None
            unplants, after which a launch raises.

        Raises
        ------
        TypeError
            If ``rfi_mask`` is not a cupy uint8 array.
        ValueError
            If its shape, contiguity or window width is wrong.
        """
        if rfi_mask is None:
            self._rfi_mask = None
            self._packer = None
            return

        import cupy as cp

        who = 'RfiMaskExtractor.set_rfi_mask()'
        if not isinstance(rfi_mask, cp.ndarray):
            raise TypeError(f'{who}: expected a cupy array (on the GPU), got {type(rfi_mask).__name__}')
        if rfi_mask.dtype != cp.uint8:
            raise TypeError(f'{who}: expected dtype uint8, got {rfi_mask.dtype}')
        if rfi_mask.ndim != 4:
            raise ValueError(f'{who}: expected a 4-d array (m, nbeams, nfreq, ntime//(8*m)),'
                             f' got shape {rfi_mask.shape}')

        m = int(rfi_mask.shape[0])
        if (m < 1) or (self.ntime % m != 0) or ((self.ntime // m) % 1024 != 0):
            raise ValueError(f'{who}: the leading axis ({m}) must divide ntime={self.ntime} into'
                             f' windows that are multiples of 1024 samples')

        w = self.ntime // m
        want = (m, self.nbeams, self.nfreq, w // 8)
        if tuple(rfi_mask.shape) != want:
            raise ValueError(f'{who}: expected shape {want} for {m} window(s) of {w} samples,'
                             f' got {tuple(rfi_mask.shape)}')
        if not rfi_mask.flags.c_contiguous:
            raise ValueError(f'{who}: expected a C-contiguous array')

        if (self._packer is None) or (self._packer.nt != w):
            self._packer = RfiMaskPackingKernel(self.nfreq, w)
        self._rfi_mask = rfi_mask

    def launch_checked(self, intensity, weights, scratch):
        dst = self._rfi_mask
        if dst is None:
            raise RuntimeError('RfiMaskExtractor: no destination is planted; call set_rfi_mask()'
                               ' before launching a chain that contains an RfiMaskExtractor')

        # One packing launch per (window, beam): dst[i, b] is a contiguous (nfreq, w/8) mask,
        # and the weights window is time-contiguous with a free row stride, which is exactly
        # what the kernel takes. The pipeline's stream is current, and launch() defaults to it.
        m = dst.shape[0]
        w = self.ntime // m
        for i in range(m):
            for b in range(self.nbeams):
                self._packer.launch(dst[i, b], weights[b, :, i*w:(i+1)*w])

    def get_mask_extractor(self):
        """This transform (see :meth:`GpuTransform.get_mask_extractor`)."""
        return self

    def to_yaml_dict(self):
        """``{'class_name': 'RfiMaskExtractor'}``: the transform has no parameters, since its
        position is its whole meaning."""
        return {'class_name': 'RfiMaskExtractor'}

    @classmethod
    def from_yaml_dict(cls, d, nbeams, nfreq, ntime):
        """The inverse of :meth:`to_yaml_dict`, at the given geometry."""
        cls.check_yaml_keys(d, [])
        return cls(nbeams, nfreq, ntime)

    @classmethod
    def from_json_dict(cls, d, nbeams, nfreq, ntime):
        """From a legacy rf_pipelines ``mask_counter`` element -- the one the conversion
        marked as the extraction point (see ``chimefrb.utils.legacy_chain_from_json``). Its
        ``nt_chunk`` (the old per-file packing granularity) and ``where`` (a label for the
        statistics we do not port) are ignored."""
        check_json_keys(d, 'mask_counter', ['nt_chunk', 'where'])
        return cls(nbeams, nfreq, ntime)

    def __repr__(self):
        planted = '' if (self._rfi_mask is None) else f', {self._rfi_mask.shape[0]} window(s) planted'
        return f'RfiMaskExtractor(nbeams={self.nbeams}, nfreq={self.nfreq}, ntime={self.ntime}{planted})'
