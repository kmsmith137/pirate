"""Numpy reference for GpuBadChannelMask, that class's method injections, and
badchannel_keep(): the conversion from MHz ranges to channels.

rf_pipelines::badchannel_mask zeroes whole frequency channels, chosen by a list of (freq_lo,
freq_hi) ranges in MHz. The zeroing is trivial. The conversion from MHz to channel indices is
not, and it is done here, in python only: GpuBadChannelMask takes a per-channel 'keep' array
and never sees a frequency.

badchannel_keep() is transcribed from _bind_transform() in
../../extern/rf_pipelines/badchannel_mask.cpp. The old code has no reference implementation
of that arithmetic (its python transform in rf_pipelines/retirement_home follows different
rules), so misc/chimefrb/rfi_badchannel_mask/ runs the real transform to compare.
"""

import math
import operator

import numpy as np

import ksgpu
from ..pirate_pybind11 import GpuBadChannelMask


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


def _as_keep(keep):
    """Converts GpuBadChannelMask's constructor argument to the uint8 array the C++ takes."""

    keep = np.asarray(keep)   # a cupy array fails here, as it should: 'keep' is host-side

    if keep.ndim != 1:
        raise ValueError(f'GpuBadChannelMask: expected a 1-d keep array, got shape {keep.shape}')
    if keep.dtype == np.uint8:
        return keep   # the C++ constructor treats any nonzero value as "keep"
    if (keep.dtype == np.bool_) or np.issubdtype(keep.dtype, np.integer):
        return (keep != 0).astype(np.uint8)

    # A float array is more likely a row of weights than a mask.
    raise TypeError(f'GpuBadChannelMask: expected a bool or integer keep array, got dtype {keep.dtype}')


@ksgpu.inject_methods(GpuBadChannelMask)
class GpuBadChannelMaskInjections:
    # No class docstring here: GpuBadChannelMask's docstring lives in the pybind11 binding
    # (option 1 in notes/docstrings.md). This injector widens the constructor to bool and
    # integer arrays, adds the from_mask_ranges() factory, and adds a stream argument for
    # launch().

    # Save references to C++ methods
    _cpp_init = GpuBadChannelMask.__init__
    _cpp_launch = GpuBadChannelMask.launch

    def __init__(self, keep, warps_per_block=4):
        """Create a GpuBadChannelMask.

        Parameters
        ----------
        keep : array-like
            Shape ``(F,)``, one entry per frequency channel, of bool or integer dtype, in host
            memory. False (or 0) masks the channel, and anything else keeps it. Copied to the
            GPU, so the caller may reuse its array.
        warps_per_block : int, optional
            Performance knob, 4, 8, 16 or 32, which must not change the result. See
            :meth:`time_selected`.
        """
        self._cpp_init(_as_keep(keep), warps_per_block)

    @staticmethod
    def from_mask_ranges(mask_ranges, nfreq, freq_lo_MHz, freq_hi_MHz, warps_per_block=4):
        """The old transform's constructor syntax: masks the channels that a list of MHz
        ranges touches, by the rule (and the quirks) in ``badchannel_keep()``.

        Parameters
        ----------
        mask_ranges : sequence of (lo, hi) pairs
            Frequency ranges to mask, in MHz.
        nfreq : int
            Number of frequency channels, F.
        freq_lo_MHz, freq_hi_MHz : float
            The band, with channel 0 at the top.
        warps_per_block : int, optional
            As for the constructor.

        Returns
        -------
        GpuBadChannelMask
        """
        keep = badchannel_keep(mask_ranges, nfreq, freq_lo_MHz, freq_hi_MHz)
        return GpuBadChannelMask(keep, warps_per_block)

    def launch(self, weights, stream=None):
        """GPU kernel launch (async, does not sync stream).

        Parameters
        ----------
        weights : cupy.ndarray
            Shape ``(B, F, T)``, float32, fully contiguous, on GPU, with any B and T. MODIFIED
            IN PLACE: every weight in a masked channel becomes +0.0, and every other weight is
            left bit-identical.
        stream : cupy.cuda.Stream or None, optional
            CUDA stream to use. If None, uses current cupy stream.
        """
        import cupy as cp

        if stream is None:
            stream = cp.cuda.get_current_stream()

        self._cpp_launch(weights, stream.ptr)


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
