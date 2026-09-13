"""Numpy reference for GpuBadChannelMask, that class's method injections, and
badchannel_keep(): the conversion from MHz ranges to channels.

rf_pipelines::badchannel_mask zeroes whole frequency channels, chosen by a list of (freq_lo,
freq_hi) ranges in MHz. The zeroing is trivial. The conversion from MHz to channel indices is
not: GpuBadChannelMask's constructor does it in C++ (src_lib/chimefrb/BadChannelMask.cu), and
badchannel_keep() below is the python transcription of the same arithmetic, which the unit
test holds the C++ to exactly.

badchannel_keep() is transcribed from _bind_transform() in
../../extern/rf_pipelines/badchannel_mask.cpp. The old code has no reference implementation
of that arithmetic (its python transform in rf_pipelines/retirement_home follows different
rules), so misc/chimefrb/rfi_badchannel_mask/ runs the real transform to compare.
"""

import math
import operator

import numpy as np

import ksgpu
from ..utils import atomic_print
from ..pirate_pybind11 import GpuBadChannelMask
from .transform_io import (CHIME_FREQ_RANGE, check_json_keys, check_yaml_keys)


# The old code's allowance for a frequency that is meant to be a channel edge but is off by
# roundoff, in CHANNEL units (not MHz). See badchannel_keep().
FUDGE = 1.0e-3


def badchannel_keep(mask_ranges, nfreq, freq_lo_MHz, freq_hi_MHz):
    """The per-channel ``keep`` array that rf_pipelines::badchannel_mask derives from a list
    of MHz ranges.

    Channel 0 is the HIGHEST frequency: channel ``i`` holds the frequencies ``x`` whose
    channel coordinate ``u(x) = (freq_hi_MHz - x) * nfreq / (freq_hi_MHz - freq_lo_MHz)``
    lies in ``[i, i+1)``. Each range is first clipped to the band; then channel ``i`` is
    masked iff the range reaches more than 1e-3 of a channel into it from both sides::

        u(lo) > i + 1e-3   and   (u(hi) + 1e-3 < i + 1   or   i == nfreq - 1)

    The 1e-3 stops a range whose ends are channel edges, up to roundoff, from masking a
    neighbouring channel. Three quirks of the old code are reproduced deliberately:

    - The ``i == nfreq - 1`` clause: a range that ends at the bottom of the band (or within
      1e-3 of a channel of it) masks the bottom channel, even at zero width. So at 400-800
      MHz, ``(390, 400)`` masks channel ``nfreq - 1``, while ``(800, 810)`` masks nothing.
    - A range entirely outside the band raises, and so does a range covering the WHOLE band.
      To mask every channel, pass ``(freq_lo_MHz, freq_hi_MHz)``.
    - A range narrower than 1e-3 of a channel can mask nothing.

    Parameters
    ----------
    mask_ranges : sequence of (lo, hi) pairs
        Frequency ranges to mask, in MHz, each with lo < hi. Order and overlaps do not
        matter: the result is the union.
    nfreq : int
        Number of frequency channels.
    freq_lo_MHz, freq_hi_MHz : float
        The band. The old code reads these from the stream (400 and 800 for CHIME); here they
        have no defaults.

    Returns
    -------
    numpy.ndarray
        Shape ``(nfreq,)``, dtype uint8: 1 = keep, 0 = mask.

    Raises
    ------
    ValueError
        Where the old code throws (lo >= hi, or a range outside or covering the band), and
        on nfreq < 1 or freq_lo_MHz >= freq_hi_MHz.
    """

    nfreq = operator.index(nfreq)
    (flo, fhi) = (float(freq_lo_MHz), float(freq_hi_MHz))

    if nfreq < 1:
        raise ValueError(f'badchannel_keep: expected nfreq >= 1, got {nfreq}')
    if not (flo < fhi):
        raise ValueError(f'badchannel_keep: expected freq_lo_MHz < freq_hi_MHz, got ({flo}, {fhi})')

    # Step 0, from the old constructor: every range must be nonempty.
    ranges = []
    for r in mask_ranges:
        if len(r) != 2:
            raise ValueError(f'badchannel_keep: expected each mask range to be a (lo, hi) pair, got {r!r}')
        (lo, hi) = (float(r[0]), float(r[1]))
        if not (lo < hi):
            raise ValueError(f'badchannel_keep: expected lo < hi, got the range ({lo}, {hi})')
        ranges.append((lo, hi))

    # Step 1, from _bind_transform(): clip each range to the band. The three branches, and the
    # throw when none of them matches, are the old code's. Note that a range covering the whole
    # band matches none of them.
    clipped = []
    for (lo, hi) in ranges:
        if (lo >= flo) and (hi <= fhi):
            clipped.append((lo, hi))
        elif (lo < flo) and (hi >= flo) and (hi <= fhi):
            clipped.append((flo, hi))
        elif (hi > fhi) and (lo <= fhi) and (lo >= flo):
            clipped.append((lo, fhi))
        else:
            raise ValueError(f'badchannel_keep: the range ({lo}, {hi}) MHz lies entirely outside'
                             f' the band [{flo}, {fhi}], or covers all of it; rf_pipelines'
                             f' rejects both')

    # Step 2: to channel indices. The expressions and their order of evaluation are the old
    # code's, in double precision, so that floor() and ceil() see the same values. (The old
    # build may contract 'factor - x*scale' into a fused multiply-add, which rounds once where
    # this rounds twice. That can change the answer only when a value lands within an ulp of
    # an integer +/- FUDGE; see near_fudge_boundary() in test_badchannel_mask.py.)
    scale = nfreq / (fhi - flo)
    factor = scale * fhi
    keep = np.ones(nfreq, dtype=np.uint8)

    for (lo, hi) in clipped:
        start = int(math.floor((factor - hi*scale) + FUDGE))
        end = int(math.ceil((factor - lo*scale) - FUDGE))

        # The two ends are clamped differently, as in the old code: start into [0, nfreq-1]
        # and end into [0, nfreq]. start's upper clamp is the only one that can bind on a
        # clipped range, and it is the first quirk in the docstring.
        start = min(max(start, 0), nfreq - 1)
        end = min(max(end, 0), nfreq)
        keep[start:end] = 0   # empty when end <= start

    return keep


# The yaml keys of GpuBadChannelMask, which are also its constructor's argument names after
# the geometry.
BADCHANNEL_MASK_YAML_KEYS = ('mask_ranges', 'freq_range')


def _as_range_list(mask_ranges):
    """A list of (float, float) pairs, from any sequence of pairs (a numpy (n, 2) array
    included), with a clear error for anything else."""
    ranges = []
    for r in mask_ranges:
        r = tuple(r)
        if len(r) != 2:
            raise ValueError(f'GpuBadChannelMask: expected each mask range to be a (lo, hi) pair, got {r!r}')
        ranges.append((float(r[0]), float(r[1])))
    return ranges


@ksgpu.inject_methods(GpuBadChannelMask)
class GpuBadChannelMaskInjections:
    # No class docstring here: GpuBadChannelMask's docstring lives in the pybind11 binding
    # (option 1 in notes/docstrings.md). launch() is inherited from GpuTransformBase; this
    # injector normalizes the constructor's range arguments, and adds the yaml and
    # legacy-json methods (transform_io.py).

    # Save references to C++ methods
    _cpp_init = GpuBadChannelMask.__init__

    def __init__(self, nbeams, nfreq, ntime, mask_ranges, freq_range, warps_per_block=4):
        """Create a GpuBadChannelMask.

        Parameters
        ----------
        nbeams, nfreq, ntime : int
            The array shape launch() will be given.
        mask_ranges : sequence of (lo, hi) pairs
            Frequency ranges to mask, in MHz, each with lo < hi, in any order. A numpy array
            of shape (n, 2) is fine.
        freq_range : (lo, hi)
            The band in MHz, channel 0 at the top; (400, 800) for CHIME.
        warps_per_block : int, optional
            Performance knob, 4, 8, 16 or 32, which must not change the result. See
            :meth:`time_selected`.
        """
        band = _as_range_list([freq_range])[0]
        self._cpp_init(int(nbeams), int(nfreq), int(ntime), _as_range_list(mask_ranges), band,
                       int(warps_per_block))

    def to_yaml_dict(self):
        """The yaml form (see ``transform_io``): the class name, ``freq_range`` and
        ``mask_ranges``, in MHz as given to the constructor."""
        return {'class_name': 'GpuBadChannelMask',
                'freq_range': [float(self.freq_range[0]), float(self.freq_range[1])],
                'mask_ranges': [[float(lo), float(hi)] for (lo, hi) in self.mask_ranges]}

    @classmethod
    def from_yaml_dict(cls, d, nbeams, nfreq, ntime):
        """The inverse of :meth:`to_yaml_dict`, at the given geometry."""
        check_yaml_keys(d, 'GpuBadChannelMask', BADCHANNEL_MASK_YAML_KEYS)
        return cls(nbeams, nfreq, ntime, d['mask_ranges'], d['freq_range'])

    @classmethod
    def from_json_dict(cls, d, nbeams, nfreq, ntime):
        """From the legacy rf_pipelines json element (``class_name: badchannel_mask``).

        The legacy json carries the MHz ranges but NOT the band, which rf_pipelines read
        from the stream at bind time; the CHIME band (400, 800) is assumed, and a line saying
        so is printed to stderr. A nonempty ``mask_path`` (a file of extra ranges) is not
        supported.
        """
        check_json_keys(d, 'badchannel_mask', ['mask_ranges', 'mask_path'])
        if d['mask_path']:
            raise NotImplementedError(f"GpuBadChannelMask.from_json_dict: mask_path={d['mask_path']!r}"
                                      f" (a file of extra ranges) is not supported; only 'mask_ranges' is")
        atomic_print(f'GpuBadChannelMask.from_json_dict: assuming freq_range = {CHIME_FREQ_RANGE} MHz'
                     f' (the legacy json does not record the band; rf_pipelines took it from the stream)',
                     fd=2)   # stderr, so that a converter writing yaml to stdout stays clean
        return cls(nbeams, nfreq, ntime, d['mask_ranges'], CHIME_FREQ_RANGE)


class ReferenceBadChannelMask:
    """Numpy reference for GpuBadChannelMask (src_lib/chimefrb/BadChannelMask.cu): zeroes
    the weights of every channel whose ``keep`` entry is 0.

    Trivial on purpose. The part of badchannel_mask that is hard to get right is the
    conversion from MHz ranges, and its references are the old code
    (misc/chimefrb/rfi_badchannel_mask/) and a second statement of the rule
    (test_badchannel_mask.keep_by_rule()), not a second implementation.
    """

    def __init__(self, keep):
        keep = np.asarray(keep)
        assert (keep.ndim == 1) and (keep.size >= 1)
        self.keep = (keep != 0)

    @classmethod
    def from_mask_ranges(cls, mask_ranges, nfreq, freq_lo_MHz, freq_hi_MHz):
        return cls(badchannel_keep(mask_ranges, nfreq, freq_lo_MHz, freq_hi_MHz))

    def apply(self, weights):
        """Returns a copy of 'weights', shape (B, F, T), with every masked channel set to 0.
        Does not modify its argument."""

        w = np.array(weights, copy=True)
        assert (w.ndim == 3) and (w.shape[1] == self.keep.size)
        w[:, ~self.keep, :] = 0
        return w
