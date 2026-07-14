"""Utility functions and context managers for pirate_frb.

This module was flattened from the former ``pirate_frb.utils`` package. The
sections below (each introduced by a ``####`` comment banner) correspond to
the former source files pirate_frb/utils/<name>.py.
"""


####################################################################################################
#
# former utils/__init__.py -- package glue: pybind11 re-exports and __all__.
#
# Names re-exported directly from the pybind11 module. (get_thread_affinity /
# set_thread_affinity are also imported in the ThreadAffinity section below,
# which needs them at module scope.)


from .pirate_pybind11 import (get_thread_affinity, set_thread_affinity,
                              test_avx2_simulate_4bit_noise, time_avx2_simulate_4bit_noise)

__all__ = ['integer_log2', 'run_processes',
           'ThreadAffinity', 'get_thread_affinity', 'set_thread_affinity',
           'GrouperHistogram', 'GpuGrouperHistogram',
           'time_cupy_dedisperser', 'show_asdf',
           'extract_ip', 'check_mtu', 'resolve_ip_spec', 'resolve_addr',
           'safe_h2g_copy', 'safe_g2h_copy',
           'test_avx2_simulate_4bit_noise', 'time_avx2_simulate_4bit_noise']


####################################################################################################
#
# former utils/core.py -- integer/bit helpers + subprocess-group orchestration.


import subprocess
import time

from .pirate_pybind11 import constants


def integer_log2(n):
    """Return log2(n) as an int, where n must be a positive power of two.

    Raises ValueError otherwise. Python analog of C++ pirate::integer_log2().
    """
    n = int(n)
    if n <= 0 or (n & (n - 1)) != 0:
        raise ValueError(f"integer_log2: argument {n} is not a positive power of two")
    return n.bit_length() - 1


def run_processes(multi_args):
    """Run several commands as child processes in parallel, fail-fast.

    'multi_args' is a list of argv lists (each a list of strings, as passed to
    subprocess.Popen). Each command is launched as its own child process; this
    function then blocks until ANY child exits -- for any reason -- at which point
    it terminates the remaining children and returns. This makes the group
    fail-fast: one process going down brings the whole set down.

    Returns 0 if every child that had exited did so cleanly (exit code 0), else 1.
    In status messages, each child is labelled by its command string.
    """
    procs = []   # list of (label, Popen)
    rc = 0
    try:
        for argv in multi_args:
            # A fresh process (not fork) avoids CUDA-after-fork hazards. stdout is
            # inherited, so the children's messages appear interleaved on our stdout.
            procs.append((" ".join(argv), subprocess.Popen(argv)))
        rc = _monitor_children(procs)
    except KeyboardInterrupt:
        print("run_processes: interrupted; stopping all processes", flush=True)
    finally:
        _terminate_children(procs)
    return rc


def _monitor_children(procs):
    """Block until any child exits; return 0 if all dead children exited cleanly,
    else 1. (The caller then terminates the survivors.) 'procs' is a list of
    (label, Popen) pairs."""
    while True:
        dead = [(label, p) for label, p in procs if p.poll() is not None]
        if dead:
            for label, p in dead:
                print(f"run_processes: child [{label}] exited (code {p.returncode}); "
                      f"stopping the other processes", flush=True)
            return 0 if all(p.returncode == 0 for _, p in dead) else 1
        time.sleep(constants.default_poll_cadence_ms / 1000)


def _terminate_children(procs, grace_sec=constants.default_shutdown_timeout_sec):
    """SIGTERM all still-running children, then SIGKILL any that don't exit within
    grace_sec. 'procs' is a list of (label, Popen) pairs."""
    for _, p in procs:
        if p.poll() is None:
            p.terminate()
    deadline = time.monotonic() + grace_sec
    for _, p in procs:
        try:
            p.wait(timeout=max(0.0, deadline - time.monotonic()))
        except subprocess.TimeoutExpired:
            p.kill()
            p.wait()


####################################################################################################
#
# former utils/ThreadAffinity.py -- context manager for temporarily setting thread CPU affinity.


from .pirate_pybind11 import get_thread_affinity, set_thread_affinity


class ThreadAffinity:
    """Context manager for temporarily setting thread CPU affinity.

    On entry, sets the calling thread's affinity to the specified vCPUs.
    On exit, restores the original affinity.

    Example:
        with ThreadAffinity([2, 3]):
            # Thread is pinned to vCPUs 2 and 3
            do_work()
        # Original affinity is restored

    Args:
        vcpu_list: List of vCPU indices to pin the thread to.
    """

    def __init__(self, vcpu_list):
        self.vcpu_list = vcpu_list
        self.saved_affinity = None

    def __enter__(self):
        self.saved_affinity = get_thread_affinity()
        set_thread_affinity(self.vcpu_list)
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        set_thread_affinity(self.saved_affinity)
        return False  # Don't suppress exceptions


####################################################################################################
#
# former utils/GrouperHistogram.py -- SNR histograms for grouper main loops:
# GPU accumulation + host-side analysis.
#
# Two classes, split along the finalization boundary:
#
#   - GpuGrouperHistogram: GPU-side accumulator (all state is cupy). Used in a
#     grouper main loop: add_tree() per tree, then finalize() on termination.
#   - GrouperHistogram: host-side result (all arrays are numpy; no GPU or cupy
#     anywhere). Produced by GpuGrouperHistogram.finalize() (online) or by
#     GrouperHistogram.from_file() (offline); supports write()/analyze()/plot().


import numpy as np


class GrouperHistogram:
    """Host-side (numpy) SNR histograms, with fitting analysis and plotting.

    Produced by GpuGrouperHistogram.finalize() (online, in a grouper main loop) or
    by GrouperHistogram.from_file() (offline re-analysis); the two paths yield
    identical objects. All arrays are numpy -- no GPU or cupy in this class.

    Two histograms, sharing the same bins:

      - 'histogram': every steady-state out_max value.
      - 'max_histogram': one sample per (beam, time-chunk) -- the max, over all
        trees/DMs/times, of the out_max values for that beam and chunk, accumulated
        only over entirely steady-state chunks (see GpuGrouperHistogram.add_tree()).
        Can legitimately be empty (e.g. a short run with no fully-steady chunk).

    Public members:

      - histogram, max_histogram: integer count arrays, shape (nbins,).
      - histogram_bins: bin edges, shape (nbins+1,). The outermost edges are +-1e6
        catch-alls (see GpuGrouperHistogram), so every value lands in some bin.
      - fit_N: None until analyze() runs; then a dict mapping the name of each
        non-empty histogram ('histogram', 'max_histogram') to its best-fit
        max-of-N-Gaussian N.

    write(filename) pickles dict(histogram=..., max_histogram=..., histogram_bins=...);
    from_file() reads the same format.
    """

    # analyze() fit region (SNR): everything below _FIT_LO is one aggregated bin and
    # everything above _FIT_HI is another, with the fine bins between them resolved.
    _FIT_LO = 3.0
    _FIT_HI = 10.0

    def __init__(self, histogram, max_histogram, histogram_bins):
        self.histogram = np.asarray(histogram)
        self.max_histogram = np.asarray(max_histogram)
        self.histogram_bins = np.asarray(histogram_bins)
        self.nbins = len(self.histogram)
        self.fit_N = None

        if ((self.nbins < 2) or (self.histogram.shape != (self.nbins,))
                or (self.max_histogram.shape != (self.nbins,))
                or (self.histogram_bins.shape != (self.nbins + 1,))):
            raise ValueError(f"GrouperHistogram: inconsistent array shapes: histogram "
                             f"{self.histogram.shape}, max_histogram "
                             f"{self.max_histogram.shape}, histogram_bins "
                             f"{self.histogram_bins.shape}")

    @staticmethod
    def from_file(filename):
        """Load a pickle written by write(), returning a GrouperHistogram."""
        import pickle
        with open(filename, 'rb') as f:
            d = pickle.load(f)
        return GrouperHistogram(d['histogram'], d['max_histogram'], d['histogram_bins'])

    def write(self, filename):
        """Pickle dict(histogram=..., max_histogram=..., histogram_bins=...) to 'filename'."""
        import pickle
        print(f'GrouperHistogram: writing {filename}', flush=True)
        with open(filename, 'wb') as f:
            pickle.dump(dict(histogram = self.histogram,
                             max_histogram = self.max_histogram,
                             histogram_bins = self.histogram_bins), f)

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
        edges = self.histogram_bins.astype(float)

        self.fit_N = {}
        for name in ('histogram', 'max_histogram'):
            counts = getattr(self, name).astype(float)
            if counts.sum() == 0:
                continue   # no events (e.g. no fully-steady chunk): skip the fit
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

        # Bin edges: the outermost edges are the +-1e6 catch-alls (see class docstring).
        # Replace them with one more interior-width step so those bins render at the
        # edge instead of blowing the x-axis out to +-1e6.
        edges = self.histogram_bins.astype(float).copy()
        w = edges[2] - edges[1]        # uniform interior bin width
        edges[0], edges[-1] = edges[1] - w, edges[-2] + w
        centers = 0.5 * (edges[:-1] + edges[1:])

        # One panel per histogram that has any counts (an empty max_histogram -- no
        # fully-steady chunk -- is omitted; see GpuGrouperHistogram.add_tree()).
        panels = [(name, title) for name, title in
                  (('histogram',     'All out_max values (steady-state)'),
                   ('max_histogram', 'Per-(beam, chunk) max (fully-steady chunks)'))
                  if getattr(self, name).sum() > 0]
        if not panels:   # nothing accumulated at all: show the (empty) top panel
            panels = [('histogram', 'All out_max values (steady-state)')]

        fig, axes = plt.subplots(len(panels), 1, figsize=(8, 4 * len(panels)), squeeze=False)
        for ax, (name, title) in zip(axes[:, 0], panels):
            counts = getattr(self, name)
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
        return (f'GrouperHistogram(nbins={self.nbins}, '
                f'counts={int(self.histogram.sum())}, '
                f'max_counts={int(self.max_histogram.sum())})')


class GpuGrouperHistogram:
    """GPU-side SNR histogram accumulator for grouper main loops.

    Intended usage (see run_toy_grouper.py): call add_tree() on each per-tree
    out_max array as it is processed, then finalize() on termination to obtain a
    host-side GrouperHistogram, which handles saving/fitting/plotting. All
    accumulation state is cupy (a few small GPU calls per add_tree); nothing is
    copied to the host until finalize().

    See the GrouperHistogram docstring for the meaning of the two accumulated
    histograms, and the add_tree() docstring for how they are built.

    Constructor args:

      - lo, hi, bin_width: nominal histogram range and bin width. The outermost
        bin edges are widened to +-1e6, so the first/last bins act as catch-alls
        for out-of-range values (every accumulated value lands in some bin).
    """

    def __init__(self, lo=-10.0, hi=100.0, bin_width=0.1):
        if not (lo < hi) or not (bin_width > 0):
            raise ValueError(f"GpuGrouperHistogram: expected lo < hi and bin_width > 0, "
                             f"got (lo, hi, bin_width) = ({lo}, {hi}, {bin_width})")

        self.lo = float(lo)
        self.hi = float(hi)
        self.nbins = int((self.hi - self.lo) / bin_width)

        # Bin edges, built once (host-side); add_tree() makes the GPU copy.
        self._bins_np = np.linspace(self.lo, self.hi, self.nbins + 1)
        self._bins_np[0] = -1e6     # catch-all outermost edges (see class docstring)
        self._bins_np[-1] = +1e6

        # GPU arrays, allocated lazily on the first add_tree() call -- this keeps the
        # constructor cheap and device-agnostic (the arrays land on whatever CUDA
        # device is current when the grouper loop starts accumulating).
        self._histogram = None      # cupy int64, shape (nbins,): all values
        self._max_histogram = None  # cupy int64, shape (nbins,): per-(beam, chunk) maxes
        self._bins = None           # cupy float64, shape (nbins+1,): = _bins_np

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
        group is committed by finalize().

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
        import cupy as cp

        if self._histogram is None:
            self._histogram = cp.zeros(self.nbins, int)
            self._max_histogram = cp.zeros(self.nbins, int)
            self._bins = cp.asarray(self._bins_np)

        vals = tree_out[..., mask] if (mask is not None) else tree_out
        h, _ = cp.histogram(vals.ravel(), bins=self._bins)
        self._histogram += h

        if itree == 0:
            self._flush_group()
            self._group_open = True
        elif not self._group_open:
            raise RuntimeError(f"GpuGrouperHistogram.add_tree: itree={itree} without a "
                               f"preceding itree=0 call (itree=0 starts a new "
                               f"(beam set, time chunk) group)")
        elif itree <= self._last_itree:
            raise RuntimeError(f"GpuGrouperHistogram.add_tree: itree={itree} is not strictly "
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
            import cupy as cp
            h, _ = cp.histogram(self._group_max, bins=self._bins)
            self._max_histogram += h
            self._group_max = None
        self._group_open = False
        self._last_itree = -1

    def finalize(self):
        """Commit the pending add_tree() group and return the accumulated histograms
        as a host-side (numpy) GrouperHistogram, which handles saving/fitting/
        plotting. If add_tree() never ran (e.g. the producer disconnected during the
        handshake), the returned histograms are all-zero.

        Intended to be called once, on termination: it commits the current
        per-(beam, chunk) max group, so finalizing mid-group and then continuing to
        accumulate would split that group's max samples."""
        self._flush_group()

        if self._histogram is not None:
            return GrouperHistogram(self._histogram.get(), self._max_histogram.get(),
                                    self._bins_np)
        return GrouperHistogram(np.zeros(self.nbins, dtype=int),
                                np.zeros(self.nbins, dtype=int), self._bins_np)

    def __repr__(self):
        state = 'empty' if (self._histogram is None) else 'accumulating'
        return f'GpuGrouperHistogram(nbins={self.nbins}, {state})'


####################################################################################################
#
# former utils/show_asdf.py -- displaying ASDF file YAML headers.


def show_asdf(f, out=None):
    """Print the YAML header of an ASDF file (everything before the '...' line).

    ASDF files have a YAML header followed by binary data blocks. The YAML
    header ends with a line containing exactly '...'. This function reads
    and prints everything up to but not including that line.

    Args:
        f: Either a filename (str) or a file-like object opened in binary mode.
        out: Output file-like object for printing (default: sys.stdout).
    """
    import sys

    if out is None:
        out = sys.stdout

    # Handle both filename and file-like object
    if isinstance(f, str):
        with open(f, 'rb') as fp:
            _show_asdf_impl(fp, out)
    else:
        _show_asdf_impl(f, out)


def _show_asdf_impl(fp, out):
    """Implementation of show_asdf that reads from an open file object."""
    for line in fp:
        # Decode bytes to string, handling potential encoding issues
        try:
            line_str = line.decode('utf-8')
        except UnicodeDecodeError:
            # If we hit binary data before finding '...', stop
            break

        # Stop at the end-of-document marker without printing it.
        if line_str.rstrip('\r\n') == '...':
            break

        out.write(line_str)


####################################################################################################
#
# former utils/network.py -- small network/NIC helpers shared by the server and
# fake X-engine entry points.


import re
import fnmatch


def extract_ip(addr):
    """Extract the IP part from an 'ip:port' string (splits on last ':').

    Raises RuntimeError if 'addr' does not contain a colon.
    """
    i = addr.rfind(':')
    if i < 0:
        raise RuntimeError(f"Expected 'ip:port' string, got {addr!r}")
    return addr[:i]


# IPv4 octet (0-255), and a full dotted-quad matcher.
_OCTET = r'(?:25[0-5]|2[0-4][0-9]|1[0-9][0-9]|[1-9]?[0-9])'
_IPV4_RE = re.compile(rf'{_OCTET}(?:\.{_OCTET}){{3}}$')
_GLOB_CHARS = '*?['


def resolve_ip_spec(hw, ipspec, context=''):
    """Resolve the 'ip' part of a config 'ip:port' entry to a concrete local IPv4
    address on THIS machine. 'ipspec' may be written in any of three forms:

      - a literal IPv4 address (e.g. '10.0.0.2')    -> returned unchanged;
      - a glob (e.g. '10.0.0.*') matched against this machine's IPv4 addresses
        -> the unique matching address (it is an error to match zero or >1);
      - a network device / NIC name (e.g. 'enp13s0f0np0') -> that device's IPv4
        address.

    The glob and device-name forms let ONE config file be shared across a cluster
    of machines that have different IP addresses: each machine resolves the spec
    to its own local address. 'hw' is a pirate_frb.Hardware instance.

    'context' is prepended to every error message (e.g. the YAML field that the
    spec came from). Raises RuntimeError with a verbose explanation on failure.
    """
    if not isinstance(ipspec, str) or ipspec == '':
        raise RuntimeError(f"{context}expected a non-empty IP spec, got {ipspec!r}.")

    # (1) Glob: match against this machine's IPv4 addresses.
    if any(c in ipspec for c in _GLOB_CHARS):
        local_ips = sorted(set(hw.ip_addrs))
        matches = sorted(ip for ip in local_ips if fnmatch.fnmatch(ip, ipspec))
        if len(matches) == 1:
            return matches[0]
        if not matches:
            raise RuntimeError(
                f"{context}IP glob {ipspec!r} matched none of this machine's IPv4 "
                f"addresses {local_ips}. Check that this machine has an interface on "
                f"the expected subnet, or fix the glob.")
        raise RuntimeError(
            f"{context}IP glob {ipspec!r} is ambiguous: it matches multiple local "
            f"IPv4 addresses {matches}. Tighten the glob so it matches exactly one.")

    # (2) Literal IPv4 address: pass through (locality is checked downstream).
    if _IPV4_RE.fullmatch(ipspec):
        return ipspec

    # (2b) Looks numeric (digits/dots) but is not a valid dotted-quad.
    if re.fullmatch(r'[0-9.]+', ipspec):
        raise RuntimeError(
            f"{context}{ipspec!r} looks like an IPv4 address but is malformed "
            f"(expected four octets in 0-255, e.g. '10.0.0.2').")

    # (3) Otherwise: a network device (NIC) name.
    nics = sorted(set(hw.nics))
    if ipspec not in nics:
        listing = ', '.join(f'{n} -> {hw.ip_addr_from_nic(n)}' for n in nics)
        raise RuntimeError(
            f"{context}{ipspec!r} is not a literal IPv4 address, contains no glob "
            f"metacharacter ('*', '?', '['), and is not a network device with an "
            f"assigned IPv4 address on this machine. Devices with IPv4 addresses "
            f"are: [{listing}].")
    return hw.ip_addr_from_nic(ipspec)


def resolve_addr(hw, addr, context=''):
    """Resolve a config 'ipspec:port' entry to a concrete 'ip:port' string, where
    'ipspec' is resolved by resolve_ip_spec() (literal IPv4 / glob / device name).

    'context' is prepended to every error message. Raises RuntimeError on failure.
    """
    if not isinstance(addr, str):
        raise RuntimeError(f"{context}expected an 'ip:port' string, got {type(addr).__name__}.")
    ncolon = addr.count(':')
    if ncolon != 1:
        raise RuntimeError(
            f"{context}expected exactly one ':' separating ip and port in 'ip:port', "
            f"got {addr!r} ({ncolon} colon(s)).")
    ipspec, port = addr.split(':', 1)
    if not re.fullmatch(r'[0-9]+', port):
        raise RuntimeError(
            f"{context}port must be a positive integer, got {port!r} in {addr!r}.")
    pnum = int(port)
    if not (1 <= pnum <= 65535):
        raise RuntimeError(f"{context}port {pnum} is out of range [1, 65535] in {addr!r}.")
    ip = resolve_ip_spec(hw, ipspec, context=context)
    return f'{ip}:{pnum}'


def check_mtu(hw, label, ip_addr, min_mtu, min_mtu_param, is_dst_addr=False):
    """Raise RuntimeError if the NIC routing for 'ip_addr' has MTU below min_mtu.

    'label' is a free-form descriptor (e.g. 'FrbServer 0 data[1]') shown in
    the exception text. 'min_mtu_param' is the YAML key name (e.g.
    'min_data_mtu') so the error message points the user at the right knob.
    Set is_dst_addr=True for FakeXEngine destinations.

    Called by pirate_frb.run_server and pirate_frb.run_fake_xengine.
    """
    nic = hw.nic_from_ip_addr(ip_addr, is_dst_addr=is_dst_addr)
    mtu = hw.mtu_from_nic(nic)
    if mtu < min_mtu:
        raise RuntimeError(
            f"{label}: NIC {nic!r} ({ip_addr}) has MTU {mtu}, below the required "
            f"minimum {min_mtu} (config param {min_mtu_param!r}).\n"
            f"  - If the small MTU is intentional, lower {min_mtu_param!r} in the "
            f"server YAML config to <= {mtu}.\n"
            f"  - If the small MTU is unintentional, reconfigure the NIC to MTU "
            f">= {min_mtu} (e.g. 'sudo ip link set {nic} mtu {min_mtu}')."
        )


####################################################################################################
#
# former utils/safe_memcpy.py -- host<->device cudaMemcpy* wrappers that handle
# BumpAllocator chunked hugepage registration.
#
# cupy's `ndarray.set()` / `.get()` call cudaMemcpyAsync directly with no
# splitting at cudaHostRegister chunk boundaries. When the host buffer
# lives in a pirate hugepage-backed BumpAllocator
# (`af_rhost | af_zero | af_mmap_huge`), such a copy can fail with
# cudaErrorInvalidValue. The wrappers below delegate to pirate's
# `safe_memcpy_*` C++ helpers, which split the copy at
# cuda_host_register_chunk_size-aligned host addresses. See
# `plans/python_h2g_chunking.md` and the doc-comment block at the top of
# `include/pirate/utils.hpp`.
#
# Note: currently, safe_h2g_copy() is only called from python in
# time_cupy_dedisperser, and safe_g2h_copy() is not called from python.


from .pirate_pybind11 import (
    safe_memcpy_h2g_async as _safe_memcpy_h2g_async_cpp,
    safe_memcpy_g2h_async as _safe_memcpy_g2h_async_cpp,
)


def _host_ptr(arr):
    """Extract a host data pointer from a numpy ndarray."""
    return int(arr.__array_interface__['data'][0])


def _check_h2g(gpu_arr, cpu_arr):
    if cpu_arr.nbytes != gpu_arr.nbytes:
        raise ValueError(
            f"nbytes mismatch: cpu={cpu_arr.nbytes}, gpu={gpu_arr.nbytes}")
    if not cpu_arr.flags.c_contiguous:
        raise ValueError("cpu_arr must be C-contiguous")
    if not gpu_arr.flags.c_contiguous:
        raise ValueError("gpu_arr must be C-contiguous")


def safe_h2g_copy(gpu_arr, cpu_arr, stream):
    """Async host->device copy with BumpAllocator chunk-aware splitting.

    Drop-in replacement for ``gpu_arr.set(cpu_arr, stream=stream)`` when
    ``cpu_arr`` may live in hugepage-backed pinned memory from a pirate
    BumpAllocator.

    Parameters
    ----------
    gpu_arr : cupy.ndarray
        Contiguous destination.
    cpu_arr : numpy.ndarray
        Contiguous source with matching nbytes.
    stream : ksgpu.CudaStreamWrapper or cupy.cuda.Stream
        Stream to issue the async copy on; must expose ``.ptr``.
    """
    _check_h2g(gpu_arr, cpu_arr)
    _safe_memcpy_h2g_async_cpp(
        int(gpu_arr.data.ptr),
        _host_ptr(cpu_arr),
        int(cpu_arr.nbytes),
        int(stream.ptr),
    )


def safe_g2h_copy(cpu_arr, gpu_arr, stream):
    """Async device->host copy with BumpAllocator chunk-aware splitting.

    Drop-in replacement for ``gpu_arr.get(out=cpu_arr, stream=stream)``
    when ``cpu_arr`` may live in hugepage-backed pinned memory from a
    pirate BumpAllocator.

    Parameters
    ----------
    cpu_arr : numpy.ndarray
        Contiguous destination.
    gpu_arr : cupy.ndarray
        Contiguous source with matching nbytes.
    stream : ksgpu.CudaStreamWrapper or cupy.cuda.Stream
        Stream to issue the async copy on; must expose ``.ptr``.
    """
    _check_h2g(gpu_arr, cpu_arr)  # same checks, names swapped in callers
    _safe_memcpy_g2h_async_cpp(
        _host_ptr(cpu_arr),
        int(gpu_arr.data.ptr),
        int(cpu_arr.nbytes),
        int(stream.ptr),
    )


####################################################################################################
#
# former utils/time_cupy_dedisperser.py -- timing benchmark for GpuDedisperser
# using Python/cupy.


import cupy as cp

from .pirate_pybind11 import GpuDequantizationKernel


def time_cupy_dedisperser(dedisperser, gpu_allocator, cpu_allocator, niterations):
    """
    Time the GpuDedisperser using Python/cupy, similar to C++ GpuDedisperser::time().

    To run from command line:  'python -m pirate_frb time_dedisperser config.yml --python'.
    (Note that omitting the --python flag will run the C++ version of the timing benchmark,
    which is in GpuDedisperser::time().)

    This function reimplements the timing logic from C++ using cupy for array/stream
    management. It:

    1. Creates the GpuDequantizationKernel
    2. Allocates raw data + scales_offsets arrays using the provided allocators
    3. Runs a timing loop that:
       - Copies raw data (int4) and scales_offsets (float16) from host to device
         on h2g_stream (back-to-back; the stream sequences them)
       - Uses CUDA events to synchronize between h2g_stream and compute_stream
       - Runs dequantization kernel on compute_stream
       - Runs dedispersion kernels via get_input() context manager
       - Measures timing per iteration

    Args:
        dedisperser: An allocated GpuDedisperser instance
        gpu_allocator: BumpAllocator for GPU memory
        cpu_allocator: BumpAllocator for CPU (pinned) memory
        niterations: Number of timing iterations to run
    """

    # Extract key parameters from dedisperser
    config = dedisperser.config
    plan = dedisperser.plan
    stream_pool = dedisperser.stream_pool

    dtype = plan.dtype
    B = plan.beams_per_batch          # beams per batch
    F = plan.nfreq                    # total frequency channels
    T = plan.nt_in                    # time samples per chunk
    S = plan.num_active_batches       # number of streams/active batches
    Tc = 1.0e-3 * T * config.time_sample_ms  # chunk duration in seconds

    assert niterations > 2 * S, f"niterations ({niterations}) must be > 2*num_active_batches ({2*S})"
    assert T % 256 == 0, f"T ({T}) must be divisible by 256"

    print(f"time_cupy_dedisperser: B={B}, F={F}, T={T}, S={S}, Tc={Tc:.3f}s")
    print()

    # Create dequantization kernel (int4 -> float16/float32, with affine transform).
    dequantization_kernel = GpuDequantizationKernel(dtype, B, F, T)

    # Resource tracking for bandwidth calculations
    rt = dedisperser.resource_tracker.clone()
    rt += dequantization_kernel.resource_tracker
    raw_nbytes   = (B * F * T) // 2                  # int4: T elements = T/2 bytes
    scoff_nbytes = B * F * (T // 256) * 2 * 2        # fp16: (scale, offset) per 256 samples
    rt.add_memcpy_h2g("raw_data",       raw_nbytes)
    rt.add_memcpy_h2g("scales_offsets", scoff_nbytes)

    h2g_bw = rt.get_h2g_bw()
    g2h_bw = rt.get_g2h_bw()
    gmem_bw = rt.get_gmem_bw()

    print(f"Expected bandwidth per iteration: h2g={h2g_bw/1e9:.2f} GB, gmem={gmem_bw/1e9:.2f} GB")
    print()

    # Create raw data + scales_offsets arrays.
    # int4 is represented as uint8 with half the elements (two int4 values per uint8 byte),
    # so the raw data shape is (S, B, F, T//2). scales_offsets is fp16 with shape
    # (S, B, F, T//256, 2); last axis is (scale, offset).
    # Note that cpu_allocator returns pinned memory.
    print("time_cupy_dedisperser: allocating raw data + scales_offsets arrays")

    multi_raw_shape   = (S, B, F, T // 2)
    multi_scoff_shape = (S, B, F, T // 256, 2)
    multi_raw_cpu   = cpu_allocator.allocate_array(np.uint8,   multi_raw_shape)
    multi_raw_gpu   = gpu_allocator.allocate_array(cp.uint8,   multi_raw_shape)
    multi_scoff_cpu = cpu_allocator.allocate_array(np.float16, multi_scoff_shape)
    multi_scoff_gpu = gpu_allocator.allocate_array(cp.float16, multi_scoff_shape)

    # Timing loop
    print(f"time_cupy_dedisperser: running {niterations} iterations...")
    print()

    # Warmup and drain any pending work
    cp.cuda.Device().synchronize()

    timestamps = [time.perf_counter()]
    event = cp.cuda.Event(disable_timing=True)

    for iteration in range(niterations):
        # iteration is the seq_id (global batch index).
        istream = iteration % S

        # Setup for current iteration.
        # Use h2g_stream for host->GPU copies, compute_stream for kernels.
        h2g_stream = stream_pool.high_priority_h2g_stream
        compute_stream = stream_pool.compute_streams[istream]
        raw_cpu   = multi_raw_cpu[istream]
        raw_gpu   = multi_raw_gpu[istream]
        scoff_cpu = multi_scoff_cpu[istream]
        scoff_gpu = multi_scoff_gpu[istream]

        # Copy raw data + scales_offsets from CPU to GPU on h2g_stream
        # (back-to-back; the stream sequences them). Use safe_h2g_copy
        # instead of cupy's .set() because raw_cpu / scoff_cpu may live in
        # hugepage-backed BumpAllocator memory whose chunked
        # cudaHostRegister layout breaks an unsplit cudaMemcpyAsync.
        # See plans/python_h2g_chunking.md.
        safe_h2g_copy(raw_gpu,   raw_cpu,   h2g_stream)
        safe_h2g_copy(scoff_gpu, scoff_cpu, h2g_stream)

        # Synchronize compute_stream before recording timestamp.
        # This ensures we measure wall-clock time for the previous iteration's work.
        compute_stream.synchronize()
        timestamps.append(time.perf_counter())

        # Use CUDA event to synchronize: compute_stream waits for h2g_stream.
        # This ensures dequantization kernel doesn't start until H2G copies complete.
        event.record(h2g_stream)
        compute_stream.wait_event(event)

        # Run dequantization and dedispersion kernels.
        # The get_input() context manager handles synchronization with dedisperser.
        with dedisperser.get_input(iteration, stream=compute_stream) as dd_in:
            # The kernel expects uint8 data which it interprets as int4.
            dequantization_kernel.launch(dd_in, scoff_gpu, raw_gpu, compute_stream)
            # Note: exiting the context manager triggers all the dedispersion kernels.

        # We're throwing away the output for timing purposes, but we still call get_output()
        # since it performs important synchronization. (get_output() yields an Outputs
        # object with .out_max / .out_argmax attributes; we don't use them here.)
        with dedisperser.get_output(iteration, stream=compute_stream) as outputs:
            pass

        # Calculate and print timing after warmup
        # Use the same averaging logic as C++ KernelTimer
        it_min = S + 1
        if iteration > it_min:
            i0 = it_min + (iteration - it_min) // 2
            dt = (timestamps[iteration + 1] - timestamps[i0 + 1]) / (iteration - i0)

            real_time_beams = B * Tc / dt
            gmem_bw_achieved = 1.0e-9 * gmem_bw / dt
            g2h_bw_achieved = 1.0e-9 * g2h_bw / dt
            h2g_bw_achieved = 1.0e-9 * h2g_bw / dt

            print(f"  iteration {iteration}: real-time beams = {real_time_beams:.2f}, "
                  f"gmem_bw = {gmem_bw_achieved:.2f}, "
                  f"g2h_bw = {g2h_bw_achieved:.2f}, "
                  f"h2g_bw = {h2g_bw_achieved:.2f} GB/s")

    print()
    print("time_cupy_dedisperser: timing complete!")
