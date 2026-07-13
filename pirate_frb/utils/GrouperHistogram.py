"""GrouperHistogram: GPU-side SNR histogram accumulator for grouper main loops."""

import numpy as np


class GrouperHistogram:
    """Accumulates histograms of dedispersion output (out_max) SNR values on the GPU.

    Intended for grouper main loops (see run_toy_grouper.py for example usage):
    call add_tree() on each per-tree out_max array as it is processed (optionally
    restricted by a boolean mask, e.g. FrbGrouper.steady_state_mask()), then write()
    on termination. The accumulation state lives on the GPU (a few small cupy calls
    per add_tree); nothing is copied to the host until write() / to_dict().

    Two histograms are accumulated, sharing the same bins:

      - 'histogram': every (masked) out_max value.
      - 'max_histogram': one sample per (beam, time-chunk) -- the max, over all
        trees/DMs/times, of the (masked) out_max values for that beam and chunk.
        Built from the 'itree' argument of add_tree(), where itree=0 marks the
        start of a new (beam set, time chunk) group; a (beam, chunk) whose values
        are entirely masked out contributes no sample.

    Constructor args:

      - stem: output filename stem; write() pickles to '<stem>.pkl'. Must not
        contain a '.' (guards against accidentally passing a full filename;
        callers with multiple groupers append a per-grouper index to the stem,
        see 'pirate_frb run_toy_grouper --histogram').
      - lo, hi, bin_width: nominal histogram range and bin width. The outermost
        bin edges are widened to +-1e6, so the first/last bins act as catch-alls
        for out-of-range values (every accumulated value lands in some bin).

    The pickled payload is dict(histogram=..., max_histogram=..., histogram_bins=...),
    all numpy arrays (lengths nbins, nbins and nbins+1), where 'histogram_bins'
    contains the shared bin edges (including the widened outermost edges).
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
        self._histogram = None      # cupy int64 array, shape (nbins,): all values
        self._max_histogram = None  # cupy int64 array, shape (nbins,): per-(beam, chunk) maxes
        self._bins = None           # cupy float64 array, shape (nbins+1,)

        # Pending state for the current add_tree() group (see add_tree docstring):
        # whether a group is open (itree=0 seen, not yet flushed), the last itree
        # within it (enforces strictly-increasing calls), and the running per-beam
        # max (a cupy array of shape (beams,), or None if all values so far were
        # masked out).
        self._group_open = False
        self._last_itree = -1
        self._group_max = None

    def add_tree(self, tree_out, itree, mask=None):
        """Accumulate the values of 'tree_out' (a cupy array, e.g. one tree's out_max
        array of shape (beams, ndm_out, nt_out)) into the histograms.

        'itree' is the dedispersion-tree index, used to build the per-(beam, chunk)
        max histogram: itree=0 commits the previous group's per-beam maxes (one
        sample per beam) and starts a new group, and subsequent calls (strictly
        increasing itree, same leading beam axis) fold into the group's running
        per-beam max. This matches the natural grouper loop, where the trees of one
        (beam set, time chunk) are visited as itree = 0, 1, ..., ntrees-1; the final
        group is committed by write()/to_dict().

        If 'mask' is not None, it must be a boolean cupy array matching the TRAILING
        axes of 'tree_out' (e.g. shape (ndm_out, nt_out), applied to every beam), and
        only values where the mask is True are accumulated -- e.g. pass
        FrbGrouper.steady_state_mask(itree, ichunk) to exclude warmup values.
        """
        import cupy as cp

        if self._histogram is None:
            self._histogram = cp.zeros(self.nbins, int)
            self._max_histogram = cp.zeros(self.nbins, int)
            self._bins = cp.linspace(self.lo, self.hi, self.nbins + 1)
            self._bins[0] = -1e6    # catch-all outermost edges (see class docstring)
            self._bins[-1] = +1e6

        vals = tree_out[..., mask] if (mask is not None) else tree_out
        h, _ = cp.histogram(vals.ravel(), bins=self._bins)
        self._histogram += h

        if itree == 0:
            self._flush_group()
            self._group_open = True
        elif not self._group_open:
            raise RuntimeError(f"GrouperHistogram.add_tree: itree={itree} without a "
                               f"preceding itree=0 call (itree=0 starts a new "
                               f"(beam set, time chunk) group)")
        elif itree <= self._last_itree:
            raise RuntimeError(f"GrouperHistogram.add_tree: itree={itree} is not strictly "
                               f"increasing within the group (previous itree = "
                               f"{self._last_itree})")
        self._last_itree = itree

        v = vals.reshape(vals.shape[0], -1)   # (beams, everything-else)
        if v.shape[1] > 0:                    # fully-masked trees contribute nothing
            bmax = v.max(axis=1)
            self._group_max = bmax if (self._group_max is None) \
                              else cp.maximum(self._group_max, bmax)

    def _flush_group(self):
        """Commit the pending per-beam maxes (one sample per beam) to max_histogram."""
        import cupy as cp
        if self._group_max is not None:
            h, _ = cp.histogram(self._group_max, bins=self._bins)
            self._max_histogram += h
            self._group_max = None
        self._group_open = False
        self._last_itree = -1

    def to_dict(self):
        """Return dict(histogram=..., max_histogram=..., histogram_bins=...) as host
        (numpy) arrays -- the payload written by write(). If add_tree() was never
        called, the counts are all zero.

        Note: finalizes the pending add_tree() group, so it should only be called
        once accumulation is complete (as write() does, on termination)."""
        self._flush_group()

        if self._histogram is not None:
            return dict(histogram = self._histogram.get(),
                        max_histogram = self._max_histogram.get(),
                        histogram_bins = self._bins.get())

        # add_tree() never ran (e.g. the producer disconnected during the handshake):
        # return all-zero histograms, so write() still produces a valid file.
        bins = np.linspace(self.lo, self.hi, self.nbins + 1)
        bins[0], bins[-1] = -1e6, +1e6
        return dict(histogram = np.zeros(self.nbins, dtype=int),
                    max_histogram = np.zeros(self.nbins, dtype=int),
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
