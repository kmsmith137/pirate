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

      - lo, hi, bin_width: nominal histogram range and bin width. The outermost
        bin edges are widened to +-1e6, so the first/last bins act as catch-alls
        for out-of-range values (every accumulated value lands in some bin).

    write(filename) pickles dict(histogram=..., max_histogram=..., histogram_bins=...),
    all numpy arrays (lengths nbins, nbins and nbins+1), where 'histogram_bins'
    contains the shared bin edges (including the widened outermost edges).
    """

    def __init__(self, lo=-10.0, hi=100.0, bin_width=0.1):
        if not (lo < hi) or not (bin_width > 0):
            raise ValueError(f"GrouperHistogram: expected lo < hi and bin_width > 0, "
                             f"got (lo, hi, bin_width) = ({lo}, {hi}, {bin_width})")

        self.lo = float(lo)
        self.hi = float(hi)
        self.nbins = int((self.hi - self.lo) / bin_width)

        # True for objects built by from_file(): arrays are host (numpy), and
        # add_tree() is disallowed (there is no GPU accumulation state).
        self._cpu_only = False

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
        if self._cpu_only:
            raise RuntimeError("GrouperHistogram.add_tree: object was loaded from a file "
                               "(CPU-only, no GPU accumulation state); add_tree() is not "
                               "supported -- construct a fresh GrouperHistogram to accumulate")
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
        if self._group_max is not None:
            import cupy as cp   # only reachable with GPU accumulation state (never CPU-only)
            h, _ = cp.histogram(self._group_max, bins=self._bins)
            self._max_histogram += h
            self._group_max = None
        self._group_open = False
        self._last_itree = -1

    @staticmethod
    def _to_host(a):
        """Return 'a' as a numpy array, whether it is a cupy array (GPU) or already numpy."""
        return a.get() if hasattr(a, 'get') else np.asarray(a)

    def to_dict(self):
        """Return dict(histogram=..., max_histogram=..., histogram_bins=...) as host
        (numpy) arrays -- the payload written by write(). If add_tree() was never
        called, the counts are all zero.

        Note: finalizes the pending add_tree() group, so it should only be called
        once accumulation is complete (as write() does, on termination)."""
        self._flush_group()

        if self._histogram is not None:
            return dict(histogram = self._to_host(self._histogram),
                        max_histogram = self._to_host(self._max_histogram),
                        histogram_bins = self._to_host(self._bins))

        # add_tree() never ran (e.g. the producer disconnected during the handshake):
        # return all-zero histograms, so write() still produces a valid file.
        bins = np.linspace(self.lo, self.hi, self.nbins + 1)
        bins[0], bins[-1] = -1e6, +1e6
        return dict(histogram = np.zeros(self.nbins, dtype=int),
                    max_histogram = np.zeros(self.nbins, dtype=int),
                    histogram_bins = bins)

    def write(self, filename):
        """Pickle to_dict() to 'filename' (see class docstring for the payload)."""
        import pickle
        print(f'GrouperHistogram: writing {filename}', flush=True)
        with open(filename, 'wb') as f:
            pickle.dump(self.to_dict(), f)

    @staticmethod
    def from_file(filename):
        """Load a pickle written by write() and return a CPU-only GrouperHistogram: the
        histograms/bins are held as host (numpy) arrays (no GPU, no cupy needed), and
        add_tree() raises. Use to_dict()/plot() to inspect the loaded histograms."""
        import pickle
        with open(filename, 'rb') as f:
            d = pickle.load(f)

        self = GrouperHistogram()
        self._cpu_only = True
        self._histogram = np.asarray(d['histogram'])
        self._max_histogram = np.asarray(d['max_histogram'])
        self._bins = np.asarray(d['histogram_bins'])
        self.nbins = len(self._bins) - 1
        return self

    def plot(self, filename):
        """Write a two-panel PDF to 'filename': the 'histogram' (all steady-state
        out_max values) in the top panel and 'max_histogram' (per-(beam, chunk)
        maxes) in the bottom. Counts are on a log scale; x-limits are taken from the
        populated (nonzero-count) bins."""
        import matplotlib
        matplotlib.use('Agg')   # headless (writing a file, no display)
        import matplotlib.pyplot as plt

        d = self.to_dict()

        # Bin edges: the outermost edges are the +-1e6 catch-alls (see class docstring).
        # Replace them with one more interior-width step so those bins render at the
        # edge instead of blowing the x-axis out to +-1e6.
        edges = d['histogram_bins'].astype(float).copy()
        w = edges[2] - edges[1]        # uniform interior bin width
        edges[0], edges[-1] = edges[1] - w, edges[-2] + w
        centers = 0.5 * (edges[:-1] + edges[1:])

        panels = [('All out_max values (steady-state)', d['histogram']),
                  ('Per-(beam, chunk) max', d['max_histogram'])]

        fig, axes = plt.subplots(2, 1, figsize=(8, 8))
        for ax, (title, counts) in zip(axes, panels):
            nz = counts > 0                       # log scale: skip empty bins
            ax.bar(centers[nz], counts[nz], width=w)
            ax.set_yscale('log')
            if nz.any():
                lo = edges[np.argmax(nz)]
                hi = edges[len(nz) - np.argmax(nz[::-1])]
                ax.set_xlim(lo, hi)
            ax.set_title(title)
            ax.set_xlabel('SNR')
            ax.set_ylabel('counts')

        fig.tight_layout()
        print(f'GrouperHistogram: writing {filename}', flush=True)
        fig.savefig(filename)
        plt.close(fig)

    def __repr__(self):
        state = 'cpu-only' if self._cpu_only else \
                ('empty' if (self._histogram is None) else 'accumulating')
        return f'GrouperHistogram(nbins={self.nbins}, {state})'
