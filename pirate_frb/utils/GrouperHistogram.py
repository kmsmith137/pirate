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
        trees/DMs/times, of the out_max values for that beam and chunk. Built from
        the 'itree' argument of add_tree() (itree=0 marks the start of a new (beam
        set, time chunk) group), and only for chunks passed with full_steady=True
        (an entirely steady-state chunk), so every sample is a max over the same
        full set of cells. Chunks with full_steady=False contribute no sample, so
        this histogram can legitimately be empty (e.g. a short run in which no chunk
        is fully steady-state).

    Constructor args:

      - lo, hi, bin_width: nominal histogram range and bin width. The outermost
        bin edges are widened to +-1e6, so the first/last bins act as catch-alls
        for out-of-range values (every accumulated value lands in some bin).

    write(filename) pickles dict(histogram=..., max_histogram=..., histogram_bins=...),
    all numpy arrays (lengths nbins, nbins and nbins+1), where 'histogram_bins'
    contains the shared bin edges (including the widened outermost edges).

    analyze() fits each histogram's tail to a 'max of N i.i.d. standard Gaussians'
    model and stores the best-fit N in self.fit_N; plot() then overlays the model.
    """

    # analyze() fit region (SNR): everything below _FIT_LO is one aggregated bin and
    # everything above _FIT_HI is another, with the fine bins between them resolved.
    _FIT_LO = 3.0
    _FIT_HI = 10.0

    def __init__(self, lo=-10.0, hi=100.0, bin_width=0.1):
        if not (lo < hi) or not (bin_width > 0):
            raise ValueError(f"GrouperHistogram: expected lo < hi and bin_width > 0, "
                             f"got (lo, hi, bin_width) = ({lo}, {hi}, {bin_width})")

        self.lo = float(lo)
        self.hi = float(hi)
        self.nbins = int((self.hi - self.lo) / bin_width)

        # Best-fit max-of-N-Gaussian N for each histogram, set by analyze() (a dict
        # keyed by 'histogram' / 'max_histogram'). None until analyze() runs.
        self.fit_N = None

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

    def add_tree(self, tree_out, itree, full_steady, mask=None):
        """Accumulate the values of 'tree_out' (a cupy array, e.g. one tree's out_max
        array of shape (beams, ndm_out, nt_out)) into the histograms.

        'itree' is the dedispersion-tree index, used to build the per-(beam, chunk)
        max histogram: itree=0 commits the previous group's per-beam maxes (one
        sample per beam) and starts a new group, and subsequent calls (strictly
        increasing itree, same leading beam axis) fold into the group's running
        per-beam max. This matches the natural grouper loop, where the trees of one
        (beam set, time chunk) are visited as itree = 0, 1, ..., ntrees-1; the final
        group is committed by write()/to_dict().

        'full_steady' (bool) gates ONLY the per-(beam, chunk) max histogram: the
        group's per-beam max is accumulated only when full_steady is True for every
        call of the group. Pass True only when the WHOLE chunk (all trees, all
        dm/time) is steady-state (e.g. ichunk >= FrbGrouper.full_steady_ichunk), so
        that every max sample is a max over the same full set of cells (a clean
        max-of-N). A chunk with full_steady=False contributes no max sample. (The
        plain 'histogram' is unaffected -- it accumulates every masked value.)

        If 'mask' is not None, it must be a boolean cupy array matching the TRAILING
        axes of 'tree_out' (e.g. shape (ndm_out, nt_out), applied to every beam), and
        only values where the mask is True are accumulated -- e.g. pass
        FrbGrouper.steady_state_mask(itree, ichunk) to exclude warmup values. When
        full_steady is True the mask is all-True, so pass mask=None to skip it.
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

        # Per-(beam, chunk) max: only for fully-steady chunks (see full_steady above).
        v = vals.reshape(vals.shape[0], -1)   # (beams, everything-else)
        if full_steady and v.shape[1] > 0:    # v.shape[1]==0 only if fully masked out
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

    @staticmethod
    def _max_of_n_bin_probs(logN, edges):
        """Per-bin probabilities of the 'max of N i.i.d. standard Gaussians' model
        (CDF F(x) = Phi(x)**N) over the consecutive 'edges' (which may include
        +-inf), computed from the exact edges. Uses log(N) as the argument and is
        numerically stable for large N: with L = log Phi,

            Phi(b)**N - Phi(a)**N = exp(N*L_b) * (1 - exp(N*(L_a - L_b)))

        and the 1 - exp(...) factor is evaluated with expm1 (no cancellation, and
        exact at the +-inf edges, where L = -inf / 0)."""
        from scipy.special import log_ndtr
        N = np.exp(logN)
        L = log_ndtr(np.asarray(edges, dtype=float))   # log Phi at each edge
        La, Lb = L[:-1], L[1:]
        return np.exp(N * Lb) * (-np.expm1(N * (La - Lb)))

    @classmethod
    def _make_fit_bins(cls, counts, edges):
        """Aggregate the fine histogram (counts over edges) into the analyze() fit
        bins: [-inf, ~_FIT_LO), the fine bins between ~_FIT_LO and ~_FIT_HI, and
        [~_FIT_HI, +inf). The _FIT_LO/_FIT_HI boundaries snap to the nearest actual
        bin edge (so counts and model probabilities use identical boundaries), and
        the outer aggregated bins use +-inf -- so their model probabilities are
        exactly Phi(_FIT_LO)**N and 1 - Phi(_FIT_HI)**N."""
        i_lo = int(np.argmin(np.abs(edges - cls._FIT_LO)))
        i_hi = int(np.argmin(np.abs(edges - cls._FIT_HI)))
        fit_edges = np.concatenate(([-np.inf], edges[i_lo:i_hi + 1], [np.inf]))
        fit_counts = np.concatenate(([counts[:i_lo].sum()],
                                     counts[i_lo:i_hi],
                                     [counts[i_hi:].sum()]))
        return fit_counts, fit_edges

    def analyze(self):
        """Fit the tail of each histogram to a 'max of N i.i.d. standard Gaussians'
        model (CDF Phi(x)**N, with N real and >= 1) and store the best-fit N in the
        self.fit_N dict; returns that dict. A histogram with zero total counts (e.g.
        an empty max_histogram) is skipped, so its key is absent from self.fit_N and
        plot() omits its model/panel. plot() then overlays the fitted models.

        The fit is a binned maximum-likelihood (multinomial) fit over the exact
        histogram bin edges: everything below SNR=_FIT_LO is lumped into one bin and
        everything above SNR=_FIT_HI into another, with the fine bins between them
        resolved -- so the fit is driven by the tail shape while conserving total
        probability. The optimization variable is log(N)."""
        from scipy.optimize import minimize_scalar
        d = self.to_dict()
        edges = d['histogram_bins'].astype(float)

        self.fit_N = {}
        for name in ('histogram', 'max_histogram'):
            counts = d[name].astype(float)
            if counts.sum() == 0:
                continue   # no events (e.g. no fully-steady chunk yet): skip the fit
            fit_counts, fit_edges = self._make_fit_bins(counts, edges)

            def nll(logN):
                p = self._max_of_n_bin_probs(logN, fit_edges)
                return -np.sum(fit_counts * np.log(np.clip(p, 1e-300, None)))

            res = minimize_scalar(nll, bounds=(0.0, np.log(1e13)), method='bounded')
            self.fit_N[name] = float(np.exp(res.x))

        return self.fit_N

    def plot(self, filename):
        """Write a PDF to 'filename' with one panel per non-empty histogram: the
        'histogram' (all steady-state out_max values) and 'max_histogram'
        (per-(beam, chunk) maxes over fully-steady chunks). An empty max_histogram
        (no fully-steady chunk) is omitted, so the plot may have one or two panels.
        Counts are on a log scale; x-limits are taken from the populated
        (nonzero-count) bins. If analyze() has been called, each panel also shows the
        best-fit max-of-N-Gaussian model as a dashed line (N in the legend)."""
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

        # One panel per histogram that has any counts (an empty max_histogram -- no
        # fully-steady chunk -- is omitted; see add_tree's full_steady argument).
        panels = [(name, title) for name, title in
                  (('histogram',     'All out_max values (steady-state)'),
                   ('max_histogram', 'Per-(beam, chunk) max (fully-steady chunks)'))
                  if d[name].sum() > 0]
        if not panels:   # nothing accumulated at all: show the (empty) top panel
            panels = [('histogram', 'All out_max values (steady-state)')]

        fig, axes = plt.subplots(len(panels), 1, figsize=(8, 4 * len(panels)), squeeze=False)
        for ax, (name, title) in zip(axes[:, 0], panels):
            counts = d[name]
            nz = counts > 0                       # log scale: skip empty bins
            ax.bar(centers[nz], counts[nz], width=w, label='data')

            # Overlay the fitted max-of-N-Gaussian model (dashed), if analyze() fit
            # this histogram. model[j] = (total counts) * P(value in fine bin j).
            if self.fit_N is not None and name in self.fit_N:
                N = self.fit_N[name]
                model = counts.sum() * self._max_of_n_bin_probs(np.log(N), edges)
                mm = model > 0
                ax.plot(centers[mm], model[mm], 'k--', lw=1.5,
                        label=f'max-of-N Gaussian fit (N = {N:.4g})')
                ax.legend()

            ax.set_yscale('log')
            if nz.any():
                lo = edges[np.argmax(nz)]
                hi = edges[len(nz) - np.argmax(nz[::-1])]
                ax.set_xlim(lo, hi)
                # y-range from the data (counts >= 1); the model line falls off the
                # bottom where it is << 1 (its values reach ~1e-300 at low SNR, which
                # would otherwise flatten the whole plot onto the log axis).
                ax.set_ylim(0.5, counts.max() * 3)
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
