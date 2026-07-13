"""GrouperHistogram: GPU-side SNR histogram accumulator for grouper main loops."""

import numpy as np


class GrouperHistogram:
    """Accumulates a histogram of dedispersion output (out_max) SNR values on the GPU.

    Intended for grouper main loops (see run_toy_grouper.py for example usage):
    call add_tree() on each per-tree out_max array as it is processed (optionally
    restricted by a boolean mask, e.g. FrbGrouper.steady_state_mask()), then write()
    on termination. The accumulation state lives on the GPU (one cp.histogram call
    per add_tree); nothing is copied to the host until write() / to_dict().

    Constructor args:

      - stem: output filename stem; write() pickles to '<stem>.pkl'. Must not
        contain a '.' (guards against accidentally passing a full filename;
        callers with multiple groupers append a per-grouper index to the stem,
        see 'pirate_frb run_toy_grouper --histogram').
      - lo, hi, bin_width: nominal histogram range and bin width. The outermost
        bin edges are widened to +-1e6, so the first/last bins act as catch-alls
        for out-of-range values (every SNR value lands in some bin).

    The pickled payload is dict(histogram=..., histogram_bins=...), both numpy
    arrays (lengths nbins and nbins+1), where 'histogram_bins' contains the bin
    edges (including the widened outermost edges).
    """

    def __init__(self, stem, lo=-10.0, hi=100.0, bin_width=0.1):
        if '.' in stem:
            raise ValueError(f"GrouperHistogram: stem {stem!r} contains a '.' -- expected a "
                             f"filename stem, not a full filename (the '.pkl' suffix is "
                             f"appended automatically)")
        if not (lo < hi) or not (bin_width > 0):
            raise ValueError(f"GrouperHistogram: expected lo < hi and bin_width > 0, "
                             f"got (lo, hi, bin_width) = ({lo}, {hi}, {bin_width})")

        self.stem = stem
        self.filename = stem + '.pkl'
        self.lo = float(lo)
        self.hi = float(hi)
        self.nbins = int((self.hi - self.lo) / bin_width)

        # GPU arrays, allocated lazily on the first add_tree() call -- this keeps the
        # constructor cheap and device-agnostic (the arrays land on whatever CUDA
        # device is current when the grouper loop starts accumulating).
        self._histogram = None    # cupy int64 array, shape (nbins,)
        self._bins = None         # cupy float64 array, shape (nbins+1,)

    def add_tree(self, tree_out, mask=None):
        """Accumulate the values of 'tree_out' (a cupy array, e.g. one tree's out_max
        array of shape (beams, ndm_out, nt_out)) into the histogram.

        If 'mask' is not None, it must be a boolean cupy array matching the TRAILING
        axes of 'tree_out' (e.g. shape (ndm_out, nt_out), applied to every beam), and
        only values where the mask is True are accumulated -- e.g. pass
        FrbGrouper.steady_state_mask(itree, ichunk) to exclude warmup values."""
        import cupy as cp

        if self._histogram is None:
            self._histogram = cp.zeros(self.nbins, int)
            self._bins = cp.linspace(self.lo, self.hi, self.nbins + 1)
            self._bins[0] = -1e6    # catch-all outermost edges (see class docstring)
            self._bins[-1] = +1e6

        vals = tree_out[..., mask] if (mask is not None) else tree_out
        h, _ = cp.histogram(vals.ravel(), bins=self._bins)
        self._histogram += h

    def to_dict(self):
        """Return dict(histogram=..., histogram_bins=...) as host (numpy) arrays --
        the payload written by write(). If add_tree() was never called, the counts
        are all zero."""
        if self._histogram is not None:
            return dict(histogram = self._histogram.get(),
                        histogram_bins = self._bins.get())

        # add_tree() never ran (e.g. the producer disconnected during the handshake):
        # return an all-zero histogram, so write() still produces a valid file.
        bins = np.linspace(self.lo, self.hi, self.nbins + 1)
        bins[0], bins[-1] = -1e6, +1e6
        return dict(histogram = np.zeros(self.nbins, dtype=int),
                    histogram_bins = bins)

    def write(self):
        """Pickle to_dict() to '<stem>.pkl' (see class docstring for the payload)."""
        import pickle
        print(f'GrouperHistogram: writing {self.filename}', flush=True)
        with open(self.filename, 'wb') as f:
            pickle.dump(self.to_dict(), f)

    def __repr__(self):
        state = 'empty' if (self._histogram is None) else 'accumulating'
        return f'GrouperHistogram(stem={self.stem!r}, nbins={self.nbins}, {state})'
