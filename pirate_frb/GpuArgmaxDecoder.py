"""Decode PIRATE peak-finder winners into physical candidate parameters.

Each cell of a producer tree's S/N map has a matching unsigned 32-bit argmax
token.  The token identifies the winning point in the finer trial space that
was reduced into that map cell::

    bits  0..7   fine time ``t`` within the coarse output-time cell
    bits  8..15  peak-finding profile ``p`` (the temporal matched filter)
    bits 16..23  multiplet ``m`` (frequency subband plus fine-DM trial)
    bits 24..31  extra-DM index ``mu`` within the coarse output-DM cell

The token alone is not a physical event.  Decoding also needs the coarse
``(tree, idm, itime)`` coordinates and the *exact* producer
``DedispersionPlan`` and explicit per-tree ``dcores``: dimensions, subband
mappings, and producer kernel time granularity determine which bit patterns are
legal.

There are deliberately two conceptual decoding stages.  Integer decoding first
recovers the winning top-level frequency-channel bounds, the exclusive trailing
time edge at each bound, and the profile index.  Physical decoding then derives
subband frequencies, DM, low-frequency arrival time, and matched-filter width.
Keeping these stages visible mirrors the C++ reference decoder and makes the
coordinate conversions reviewable.

:class:`GpuArgmaxDecoder` performs plan inspection once on the CPU, uploads
small immutable lookup tables, and applies ordinary CuPy array operations to an
exact-sized candidate batch.  Candidate-shaped columns remain on one GPU during
normal operation.  The only normal host synchronization is one aggregate
validity scalar; an exceptional invalid batch additionally copies its first bad
row so that the raised error is useful.
"""

from dataclasses import dataclass
from enum import IntEnum
from numbers import Integral
from typing import Any

import numpy as np
import cupy as cp
class ArgmaxDecodeStatus(IntEnum):
    """Validation result for one raw candidate, stored as ``uint8``.

    A row can violate several constraints (the all-ones sentinel is also an
    invalid profile and multiplet, for example).  Validation is therefore
    ordered, and the first failing check performed by :meth:`decode` wins.  The
    check order is tree, sentinel, coarse DM, coarse time, multiplet, extra DM, profile,
    fine time, fine-time granularity, and absolute-chunk multiplication.  The
    numeric enum value is a stable diagnostic code, not a severity ordering.
    """

    OK = 0                       # Every coordinate and token field is legal.
    INVALID_TREE = 1             # ``tree`` is outside ``[0, plan.ntrees)``.
    INVALID_COARSE_DM = 2         # ``idm`` is outside this tree's map rows.
    INVALID_COARSE_TIME = 3       # ``itime`` is outside this tree's map columns.
    INVALID_MULTIPLET = 4         # Token bits 16..23 do not name a tree multiplet.
    INVALID_PROFILE = 5           # Token bits 8..15 do not name a tree profile.
    INVALID_FINE_TIME = 6         # Token bits 0..7 exceed the coarse time cell.
    INVALID_TIME_GRANULARITY = 7  # Fine time violates this profile's ``Dcore`` grid.
    INVALID_SENTINEL = 9          # Producer's ``0xffffffff`` no-winner sentinel.
    ABSOLUTE_TIME_OVERFLOW = 10   # ``source_chunk * plan.nt_in`` overflows int64.
    INVALID_EXTRA_DM = 11        # Token bits 24..31 exceed this tree's DM sub-bins.


class GpuArgmaxDecodeError(ValueError):
    """Describe the first invalid row in a rejected GPU candidate batch.

    The decoder rejects an invalid batch as a unit.  Only the first offending
    row is copied to the host on this exceptional path; its input index, ordered
    :class:`ArgmaxDecodeStatus`, raw map coordinates, token, and source chunk are
    retained as attributes/message context.
    """

    def __init__(self, index, status, *, tree, idm, itime, token,
                 source_chunk_index):
        """Initialize an error from the small host snapshot of one bad row."""

        self.index = int(index)
        self.status = ArgmaxDecodeStatus(int(status))
        super().__init__(
            f"candidate {self.index}: {self.status.name.lower().replace('_', ' ')} "
            f"(tree={int(tree)}, idm={int(idm)}, itime={int(itime)}, "
            f"token=0x{int(token):08x}, "
            f"source_chunk_index={int(source_chunk_index)})"
        )


@dataclass(frozen=True)
class GpuDecodedCandidateBatch:
    """Raw provenance and decoded, candidate-major CuPy columns.

    Every array is one-dimensional with length ``Ncandidate`` and resides on the
    decoder's CUDA device.  ``raw`` is the original ``GpuRawCandidates`` object,
    retained without copying, so S/N, token, edge flags, beam, chunk, and coarse
    coordinates preserve their producer dtypes.  In particular ``edge_flags``
    remains ``uint8`` and is passed through unchanged.  A
    ``STARTUP_INCOMPLETE`` bit changes none of the formulas below: decoded DM,
    TOA, width, and measured S/N are reported as obtained.  It warns downstream
    users that pre-acquisition zero padding may have biased S/N and the winning
    profile width.

    ``fmin`` and ``fmax`` are inclusive ``int64`` indices on the rank-
    ``toplevel_tree_rank`` frequency grid.  ``tlo`` and ``thi`` are ``int64``
    exclusive trailing time edges at those frequency indices, measured in
    full-resolution input samples relative to the start of the source chunk;
    negative values legitimately refer to preceding chunks.  ``profile`` is the
    decoded ``int64`` temporal-filter index.

    The physical columns are ``float64``.  DM is in pc cm^-3;
    ``width_samp`` is in full-resolution input samples and ``width_ms`` in
    milliseconds; frequencies are in MHz.  ``toa_sample_abs`` is a signed,
    unclipped, and potentially fractional full-resolution-sample coordinate from
    FPGA sequence zero.  It estimates the pulse centre at
    ``plan.config.zone_freq_edges[0]``, the lowest edge of the complete observing
    band, rather than at infinite frequency or at the winning subband edge.
    ``status`` is ``uint8`` with :class:`ArgmaxDecodeStatus` values.
    """

    raw: Any
    primary_tree_index: Any  # int32
    fmin: Any                # int64, inclusive tree-frequency bound
    fmax: Any                # int64, inclusive tree-frequency bound
    tlo: Any                 # int64, exclusive trailing edge
    thi: Any                 # int64, exclusive trailing edge
    profile: Any             # int64
    dm: Any                  # float64, pc cm^-3
    toa_sample_abs: Any      # float64, full-resolution samples from FPGA seq zero
    width_samp: Any          # float64, full-resolution samples
    width_ms: Any            # float64
    freq_lo_MHz: Any         # float64
    freq_hi_MHz: Any         # float64
    status: Any              # uint8, values from ArgmaxDecodeStatus

    def __len__(self):
        """Return the number of candidate rows (array metadata; no device copy)."""

        return int(self.status.size)

    @property
    def valid_mask(self):
        """Device mask derived from the single stored status column."""
        return self.status == int(ArgmaxDecodeStatus.OK)

    @property
    def beam_id(self):
        """Return the raw ``int32`` beam identifier column on the GPU."""

        return self.raw.beam_id

    @property
    def source_chunk_index(self):
        """Return raw ``int64`` source-chunk indices used for absolute time."""

        return self.raw.source_chunk_index

    @property
    def tree(self):
        """Return raw ``int32`` producer-tree indices."""

        return self.raw.tree

    @property
    def idm(self):
        """Return raw ``int32`` coarse DM-row coordinates."""

        return self.raw.idm

    @property
    def itime(self):
        """Return raw ``int32`` chunk-relative coarse time coordinates."""

        return self.raw.itime

    @property
    def snr(self):
        """Return the raw floating-point peak S/N column without recasting it."""

        return self.raw.snr

    @property
    def argmax_token(self):
        """Return raw ``uint32`` packed argmax tokens."""

        return self.raw.argmax_token

    @property
    def edge_flags(self):
        """Return raw ``uint8`` candidate quality/edge bitmasks unchanged."""

        return self.raw.edge_flags


def _plan_integer(value, name):
    """Return an exact plan integer, rejecting booleans and numeric coercion.

    Producer-plan dimensions are array bounds and bit-field limits.  Accepting a
    float such as ``3.0`` would conceal malformed serialized metadata, while a
    boolean is technically an ``Integral`` but is never a meaningful dimension.
    """

    if isinstance(value, (bool, np.bool_)) or not isinstance(value, Integral):
        raise ValueError(f"{name} must be an integer")
    return int(value)


def _power_of_two(value):
    """Return whether a positive Python integer has exactly one set bit."""

    return value > 0 and (value & (value - 1)) == 0


class GpuArgmaxDecoder:
    """Decode exact-sized raw-candidate batches on one CUDA device.

    Parameters
    ----------
    plan
        The exact producer ``DedispersionPlan``, normally reconstructed from the
        config and plan YAML saved beside the S/N maps.  Reconstructing a fresh
        plan from config alone is insufficient: compiled kernels may have used a
        different per-tree ``Dcore``, which controls legal fine-time tokens.
    dcores : sequence of int
        Required per-tree time granularities from the actual producer
        (``GpuDedisperser.Dcores`` or the grouper handshake). PIRATE 1.5
        no longer stores these in the plan. Never infer them from a
        consumer's compiled kernels.
    cuda_device_id : int or None
        Device that owns input/output candidate arrays.  ``None`` captures the
        currently active CUDA device at construction.

    Notes
    -----
    Construction is the CPU setup boundary.  It validates plan invariants,
    flattens ragged per-tree multiplet/profile metadata into compact NumPy
    tables, and uploads those tables once.  :meth:`decode` thereafter allocates
    only candidate-sized CuPy columns and performs indexed table lookup on this
    device.  The object is immutable by convention and reusable across chunks
    and beam batches produced by the same plan.
    """

    _RAW_DTYPES = {
        "beam_id": np.dtype(np.int32),
        "source_chunk_index": np.dtype(np.int64),
        "tree": np.dtype(np.int32),
        "idm": np.dtype(np.int32),
        "itime": np.dtype(np.int32),
        "argmax_token": np.dtype(np.uint32),
        "edge_flags": np.dtype(np.uint8),
    }

    def __init__(self, plan, cuda_device_id=None, *, dcores):
        """Validate one producer plan and upload reusable decoder tables.

        Parameters
        ----------
        plan
            Exact producer ``DedispersionPlan`` reconstructed from the saved
            configuration and plan YAML.  Per-tree token limits, subbands,
            profiles, frequency conversion, and chunk length are inspected on
            CPU.
        dcores : sequence of int
            Producer time granularity for each tree, carried separately
            from the plan YAML. Required even for an empty candidate batch.
        cuda_device_id : int or None, optional
            CUDA device that will own the lookup tables and every input/output
            candidate batch.  ``None`` records the currently active device.

        Notes
        -----
        :meth:`_make_host_tables` performs plan validation and flattens ragged
        multiplet/profile tables into exact-dtype NumPy arrays.  This
        constructor uploads only those small immutable arrays once.  It stores
        no source maps or candidates and can decode many batches from the same
        plan.  Invalid metadata raises before a usable decoder is returned.
        """

        if cuda_device_id is None:
            cuda_device_id = cp.cuda.runtime.getDevice()
        self.cuda_device_id = int(cuda_device_id)
        host = self._make_host_tables(plan, dcores)
        self.ntrees = host.pop("ntrees")
        self.ntree = host.pop("ntree")
        self.nt_in = host.pop("nt_in")
        self.dm_per_unit_delay = host.pop("dm_per_unit_delay")
        self.time_sample_ms = host.pop("time_sample_ms")
        with cp.cuda.Device(self.cuda_device_id):
            self._tables = {name: cp.asarray(value) for name, value in host.items()}

    @staticmethod
    def _make_host_tables(plan, dcores):
        """Validate producer metadata and build CPU-side decoder lookup tables.

        The returned mapping contains Python scalar acquisition constants plus
        exact-dtype NumPy arrays.  Per-tree arrays are ``int64``.  Multiplet and
        profile arrays are flattened because their lengths differ by tree; each
        tree stores an offset and count into the corresponding flat table.
        Frequencies and physical profile values are ``float64``.  The caller
        removes the scalar entries and uploads all arrays in one setup phase.

        Validation deliberately precedes GPU work.  It checks that dimensions
        fit token fields/candidate coordinate dtypes, time/frequency ranks agree,
        ``Dcore`` gives the producer's legal token granularity, subbands are
        dyadic and in range, and all physical conversions are finite.  A
        ``ValueError`` identifies the first inconsistent plan field.
        """

        # These acquisition-wide values define the coordinate system shared by
        # all trees.  ``ntree`` is the number of top-level tree-frequency cells,
        # not the instrument's raw channel count.
        trees = tuple(plan.trees)
        ntrees = _plan_integer(plan.ntrees, "plan.ntrees")
        nt_in = _plan_integer(plan.nt_in, "plan.nt_in")
        config = plan.config
        top_rank = _plan_integer(
            config.toplevel_tree_rank, "config.toplevel_tree_rank")
        if ntrees <= 0 or ntrees != len(trees):
            raise ValueError("plan.ntrees disagrees with plan.trees")
        dcores = tuple(
            _plan_integer(value, f"dcores[{i}]")
            for i, value in enumerate(dcores)
        )
        if len(dcores) != ntrees:
            raise ValueError("dcores must contain one producer value per tree")
        if not 0 < top_rank <= 16:
            raise ValueError("config.toplevel_tree_rank must be in [1, 16]")
        if nt_in <= 0 or nt_in > np.iinfo(np.int64).max:
            raise ValueError("plan.nt_in must be a positive int64")
        if _plan_integer(
                config.time_samples_per_chunk,
                "config.time_samples_per_chunk") != nt_in:
            raise ValueError("config time_samples_per_chunk disagrees with plan.nt_in")
        ntree = 1 << top_rank
        dm_per_unit_delay = float(config.dm_per_unit_delay())
        time_sample_ms = float(config.time_sample_ms)
        frequency_edges = tuple(float(x) for x in config.zone_freq_edges)
        if not np.isfinite(dm_per_unit_delay) or dm_per_unit_delay <= 0.0:
            raise ValueError("config.dm_per_unit_delay() must be finite and positive")
        if not np.isfinite(time_sample_ms) or time_sample_ms <= 0.0:
            raise ValueError("config.time_sample_ms must be finite and positive")
        if len(frequency_edges) < 2 or not 0.0 < frequency_edges[0] < frequency_edges[-1]:
            raise ValueError("config.zone_freq_edges do not define a positive band")
        # Tree maps are ragged: ndm_out, nt_out, multiplet count, and profile
        # count can all differ.  Small tree-indexed tables and flattened
        # variable-length tables preserve that layout without padding.
        names = ("Dout", "ndm_out", "nt_out", "nprofiles", "primary",
                 "time_scale", "m_offset", "m_count", "extra_dm_count",
                 "profile_offset")
        per_tree = {name: [] for name in names}
        m_fmin, m_fmax, m_high_lag, m_bandwidth, m_dfine = [], [], [], [], []
        profile_dt, profile_width, profile_shift = [], [], []
        for itree, tree in enumerate(trees):
            # Validate the same structural relationships assumed by the C++
            # producer decoder before deriving any indices from this tree.
            label = f"tree {itree}"
            ipri = _plan_integer(tree.primary_tree_index, f"{label}.primary_tree_index")
            early = _plan_integer(tree.early_trigger_level, f"{label}.early_trigger_level")
            nt_ds = _plan_integer(tree.nt_ds, f"{label}.nt_ds")
            ndm_out = _plan_integer(tree.ndm_out, f"{label}.ndm_out")
            nt_out = _plan_integer(tree.nt_out, f"{label}.nt_out")
            nprofiles = _plan_integer(tree.nprofiles, f"{label}.nprofiles")
            Dcore = dcores[itree]
            max_width = _plan_integer(
                tree.primary_tree.max_width, f"{label}.primary_tree.max_width")
            fs = tree.frequency_subbands
            pf_rank = _plan_integer(fs.pf_rank, f"{label}.frequency_subbands.pf_rank")
            rr = _plan_integer(tree.tree_rank, f"{label}.tree_rank") + (ipri > 0)
            if not 0 <= ipri <= top_rank or early < 0:
                raise ValueError(f"{label} has invalid primary/early-trigger indices")
            if rr != top_rank - early or not 0 <= pf_rank <= rr:
                raise ValueError(f"{label} has inconsistent tree/subband ranks")
            if nt_ds <= 0 or nt_out <= 0 or nt_ds % nt_out:
                raise ValueError(f"{label} has inconsistent time dimensions")
            Dout = nt_ds // nt_out
            if not 0 < Dout <= 256:
                raise ValueError(f"{label} Dout does not fit the token fine-time field")
            if not _power_of_two(Dcore) or Dcore > Dout or Dout % Dcore:
                raise ValueError(f"{label} has invalid producer Dcore")
            if not 0 < ndm_out <= np.iinfo(np.int32).max:
                raise ValueError(f"{label}.ndm_out does not fit candidate coordinates")
            if not 0 < nt_out <= np.iinfo(np.int32).max:
                raise ValueError(f"{label}.nt_out does not fit candidate coordinates")
            if not 0 < nprofiles <= 256:
                raise ValueError(f"{label}.nprofiles does not fit the token profile field")
            if (not _power_of_two(max_width)
                    or nprofiles != 1 + 3 * (max_width.bit_length() - 1)):
                raise ValueError(f"{label} has inconsistent peak-finder profiles")
            time_scale = 1 << ipri
            if nt_ds * time_scale != nt_in:
                raise ValueError(f"{label} time span disagrees with plan.nt_in")
            M = _plan_integer(fs.M, f"{label}.frequency_subbands.M")
            N = _plan_integer(fs.N, f"{label}.frequency_subbands.N")
            m_to_n = tuple(int(x) for x in fs.m_to_n)
            m_to_d = tuple(int(x) for x in fs.m_to_d)
            n_to_flo = tuple(int(x) for x in fs.n_to_flo)
            n_to_fhi = tuple(int(x) for x in fs.n_to_fhi)
            if not 0 < M <= 256 or len(m_to_n) != M or len(m_to_d) != M:
                raise ValueError(f"{label} has inconsistent multiplet tables")
            if N <= 0 or len(n_to_flo) != N or len(n_to_fhi) != N:
                raise ValueError(f"{label} has inconsistent subband tables")
            coarse_nfreq = 1 << pf_rank
            dm_downsampling = _plan_integer(
                tree.dm_downsampling, f"{label}.dm_downsampling")
            extra_dm_count, remainder = divmod(dm_downsampling, coarse_nfreq)
            if (remainder or not _power_of_two(extra_dm_count)
                    or extra_dm_count > 256):
                raise ValueError(f"{label} has invalid extra-DM token dimensions")
            per_tree["extra_dm_count"].append(extra_dm_count)
            per_tree["m_offset"].append(len(m_fmin))
            per_tree["m_count"].append(M)
            for m, (n, dfine) in enumerate(zip(m_to_n, m_to_d)):
                # A multiplet is the producer's packed combination of one
                # searched frequency subband ``n`` and its fine-DM offset.
                # Convert its coarse subband bounds to inclusive top-level
                # frequency indices now so candidate decoding is table lookup.
                if not 0 <= n < N:
                    raise ValueError(f"{label} multiplet {m} has invalid subband index")
                flo, fhi = n_to_flo[n], n_to_fhi[n]
                width = fhi - flo
                if not (0 <= flo < fhi <= coarse_nfreq and _power_of_two(width)):
                    raise ValueError(f"{label} subband {n} has invalid bounds")
                if not 0 <= dfine < width:
                    raise ValueError(f"{label} multiplet {m} has invalid fine DM")
                fmin = int(tree.n_to_toplevel_flo(n))
                fmax = int(tree.n_to_toplevel_fhi(n)) - 1
                if not 0 <= fmin < fmax < ntree:
                    raise ValueError(f"{label} multiplet {m} has invalid output band")
                m_fmin.append(fmin)
                m_fmax.append(fmax)
                # ``high_lag`` counts coarse frequency cells above the winning
                # subband; ``bandwidth`` supplies its across-subband delay.
                m_high_lag.append(coarse_nfreq - fhi)
                m_bandwidth.append(width)
                m_dfine.append(dfine)
            per_tree["profile_offset"].append(len(profile_dt))
            for p in range(nprofiles):
                # Profile zero is the native one-bin boxcar.  Each later
                # peak-finding level contributes a boxcar, triangular, and
                # trapezoidal profile.  ``profile_dt`` is the legal token-time
                # quantum; width and centre shift use full-resolution samples.
                pdiv, pmod = divmod(p, 3)
                lpf = (p - 1) // 3 if p else 0
                profile_dt.append(min(Dcore, 1 << lpf))
                if p == 0:
                    scale = 1 << ipri
                    width, shift = float(scale), 0.5 * scale
                elif pmod == 1:
                    scale = 1 << (ipri + pdiv + 1)
                    width, shift = float(scale), 0.5 * scale
                elif pmod == 2:
                    scale = 1 << (ipri + pdiv)
                    width, shift = 2.0 * scale, 1.5 * scale
                else:
                    scale = 1 << (ipri + pdiv - 1)
                    width, shift = 3.0 * scale, 2.0 * scale
                profile_width.append(width)
                profile_shift.append(shift)
            for name, value in (
                ("Dout", Dout), ("ndm_out", ndm_out),
                ("nt_out", nt_out), ("nprofiles", nprofiles),
                ("primary", ipri), ("time_scale", time_scale),
            ):
                per_tree[name].append(value)
        # Tree-frequency index increases toward lower radio frequency.  Include
        # both ends so [fmin, fmax] maps to physical edges
        # [delay_to_frequency(fmax + 1), delay_to_frequency(fmin)].
        freq_by_delay = np.fromiter(
            (float(config.delay_to_frequency(d)) for d in range(ntree + 1)),
            dtype=np.float64,
            count=ntree + 1,
        )
        if (not np.all(np.isfinite(freq_by_delay))
                or not np.all(freq_by_delay > 0.0)
                or not np.all(np.diff(freq_by_delay) < 0.0)):
            raise ValueError("config.delay_to_frequency() produced an invalid lookup")
        ret = {
            "ntrees": ntrees,
            "ntree": ntree,
            "nt_in": nt_in,
            "dm_per_unit_delay": dm_per_unit_delay,
            "time_sample_ms": time_sample_ms,
            "m_fmin": np.asarray(m_fmin, dtype=np.int64),
            "m_fmax": np.asarray(m_fmax, dtype=np.int64),
            "m_high_lag": np.asarray(m_high_lag, dtype=np.int64),
            "m_bandwidth": np.asarray(m_bandwidth, dtype=np.int64),
            "m_dfine": np.asarray(m_dfine, dtype=np.int64),
            "profile_dt": np.asarray(profile_dt, dtype=np.int64),
            "profile_width": np.asarray(profile_width, dtype=np.float64),
            "profile_shift": np.asarray(profile_shift, dtype=np.float64),
            "freq_by_delay": freq_by_delay,
        }
        ret.update({
            f"tree_{name}": np.asarray(values, dtype=np.int64)
            for name, values in per_tree.items()
        })
        return ret

    @staticmethod
    def _raw_field(raw, name):
        """Read a required raw column and turn a missing attribute into TypeError."""

        try:
            return getattr(raw, name)
        except AttributeError as exc:
            raise TypeError(f"raw candidates are missing field {name!r}") from exc

    def _validate_raw(self, raw, cp):
        """Validate raw GPU column shape, dtype, device, and equal length.

        Parameters are checked from CuPy array metadata, so this routine does not
        copy candidate values to the CPU.  S/N may be ``float32`` or ``float64``;
        provenance, token, and flag columns must retain their exact schema dtypes.
        The returned dictionary aliases the original arrays.
        """

        fields = {}
        expected = dict(self._RAW_DTYPES)
        expected["snr"] = None
        sizes = set()
        for name, dtype in expected.items():
            value = self._raw_field(raw, name)
            if not isinstance(value, cp.ndarray):
                raise TypeError(f"raw.{name} must be a cupy.ndarray")
            if value.ndim != 1:
                raise ValueError(f"raw.{name} must be one-dimensional")
            if int(value.device.id) != self.cuda_device_id:
                raise ValueError(f"raw.{name} is on the wrong CUDA device")
            if name == "snr":
                if value.dtype not in (cp.dtype(cp.float32), cp.dtype(cp.float64)):
                    raise TypeError("raw.snr must have dtype float32 or float64")
            elif value.dtype != dtype:
                raise TypeError(f"raw.{name} must have dtype {dtype}, got {value.dtype}")
            sizes.add(int(value.size))
            fields[name] = value
        if len(sizes) > 1:
            raise ValueError("raw candidate fields must have equal lengths")
        return fields

    @staticmethod
    def _set_first_status(cp, status, condition, code):
        """Set ``code`` only on failing rows that have no earlier error.

        This candidate-sized operation stays on the GPU and makes the sequence
        of calls in :meth:`decode` the authoritative validation precedence.
        """

        return cp.where(
            (status == int(ArgmaxDecodeStatus.OK)) & condition,
            cp.uint8(code),
            status,
        )

    @staticmethod
    def _raise_first_invalid(cp, status, raw):
        """Copy and raise the first invalid row after aggregate validation fails.

        This is the exceptional host-transfer boundary.  ``flatnonzero`` and
        slicing first compact to at most one GPU row; a single small stacked
        array is then copied, rather than materializing the full candidate batch.
        """

        bad = cp.flatnonzero(status != int(ArgmaxDecodeStatus.OK))[:1]
        snapshot = cp.column_stack((
            bad,
            status[bad].astype(cp.int64),
            raw["tree"][bad].astype(cp.int64),
            raw["idm"][bad].astype(cp.int64),
            raw["itime"][bad].astype(cp.int64),
            raw["argmax_token"][bad].astype(cp.int64),
            raw["source_chunk_index"][bad],
        )).get()
        if snapshot.shape[0]:
            row = snapshot[0]
            raise GpuArgmaxDecodeError(
                row[0], row[1], tree=row[2], idm=row[3], itime=row[4],
                token=row[5], source_chunk_index=row[6],
            )
        raise RuntimeError("invalid decoder status could not be located")

    def decode(self, raw):
        """Validate and decode one raw candidate batch on the configured GPU.

        ``raw`` must expose equal-length, one-dimensional CuPy columns named by
        ``_RAW_DTYPES`` plus floating-point ``snr``.  All columns must occupy the
        configured device.  Map coordinates are chunk-relative; the signed
        ``source_chunk_index`` converts decoded time to the absolute FPGA-sample
        coordinate ``source_chunk_index * plan.nt_in + relative_toa``.

        Returns
        -------
        GpuDecodedCandidateBatch
            Candidate-major GPU columns.  Integer geometry matches
            ``DedispersionPlan.decode_argmax`` exactly; physical ``float64``
            columns match ``decode_argmax2`` within the documented FMA rounding
            tolerance.  Raw S/N and edge flags are not numerically modified.

        Raises
        ------
        TypeError, ValueError
            If the raw column schema, dimensionality, dtype, or device is wrong.
        GpuArgmaxDecodeError
            If any coordinate/token row is invalid.  No partially decoded batch
            is returned.

        Notes
        -----
        Candidate arrays stay on the GPU.  Normal processing transfers only the
        scalar result of ``all(status == OK)``; an invalid batch additionally
        transfers its first diagnostic row.
        """

        with cp.cuda.Device(self.cuda_device_id):
            raw_fields = self._validate_raw(raw, cp)
            n = int(raw_fields["argmax_token"].size)
            t = self._tables
            tree = raw_fields["tree"].astype(cp.int64)
            idm = raw_fields["idm"].astype(cp.int64)
            itime = raw_fields["itime"].astype(cp.int64)
            source_chunk = raw_fields["source_chunk_index"]
            token = raw_fields["argmax_token"]
            # PIRATE 1.5 packs four independent bytes: t, p, m, mu.
            # Coarse DM/time and tree indices are separate raw columns.
            fine_time = (token & cp.uint32(0xff)).astype(cp.int64)
            profile = ((token >> cp.uint32(8)) & cp.uint32(0xff)).astype(cp.int64)
            multiplet = ((token >> cp.uint32(16)) & cp.uint32(0xff)).astype(cp.int64)
            extra_dm = (token >> cp.uint32(24)).astype(cp.int64)
            tree_ok = (tree >= 0) & (tree < self.ntrees)
            # Clipped "safe" indices prevent invalid rows from causing an
            # out-of-bounds table access while their ordered status is built.
            # They never legitimize a row: any failed predicate is raised below.
            safe_tree = cp.clip(tree, 0, self.ntrees - 1)
            Dout = t["tree_Dout"][safe_tree]
            ndm_out = t["tree_ndm_out"][safe_tree]
            nt_out = t["tree_nt_out"][safe_tree]
            nprofiles = t["tree_nprofiles"][safe_tree]
            primary = t["tree_primary"][safe_tree]
            time_scale = t["tree_time_scale"][safe_tree]
            m_count = t["tree_m_count"][safe_tree]
            extra_dm_count = t["tree_extra_dm_count"][safe_tree]

            # Unsigned token fields cannot be negative.  Upper-bound predicates
            # are therefore sufficient before indexing flattened ragged tables.
            m_ok = multiplet < m_count
            safe_m = cp.minimum(multiplet, m_count - 1)
            flat_m = t["tree_m_offset"][safe_tree] + safe_m
            p_ok = profile < nprofiles
            safe_profile = cp.minimum(profile, nprofiles - 1)
            flat_profile = t["tree_profile_offset"][safe_tree] + safe_profile
            dt = t["profile_dt"][flat_profile]
            # Check the integer chunk-start multiplication before performing it.
            # Relative decoded times are small float64 offsets and may validly
            # fall outside their source chunk.
            int64_min = int(np.iinfo(np.int64).min)
            int64_max = int(np.iinfo(np.int64).max)
            min_chunk = -((-int64_min) // self.nt_in)
            max_chunk = int64_max // self.nt_in
            absolute_ok = (
                (source_chunk >= min_chunk) & (source_chunk <= max_chunk)
            )

            # ``_set_first_status`` only updates OK rows.  Consequently this call
            # order, rather than enum numeric order, defines which error is
            # reported when a row is malformed in more than one way.
            status = cp.zeros(n, dtype=cp.uint8)
            status = self._set_first_status(
                cp, status, ~tree_ok, ArgmaxDecodeStatus.INVALID_TREE)
            status = self._set_first_status(
                cp, status, token == cp.uint32(0xffffffff),
                ArgmaxDecodeStatus.INVALID_SENTINEL)
            status = self._set_first_status(
                cp, status, (idm < 0) | (idm >= ndm_out),
                ArgmaxDecodeStatus.INVALID_COARSE_DM)
            status = self._set_first_status(
                cp, status, (itime < 0) | (itime >= nt_out),
                ArgmaxDecodeStatus.INVALID_COARSE_TIME)
            status = self._set_first_status(
                cp, status, ~m_ok, ArgmaxDecodeStatus.INVALID_MULTIPLET)
            status = self._set_first_status(
                cp, status, extra_dm >= extra_dm_count,
                ArgmaxDecodeStatus.INVALID_EXTRA_DM)
            status = self._set_first_status(
                cp, status, ~p_ok, ArgmaxDecodeStatus.INVALID_PROFILE)
            status = self._set_first_status(
                cp, status, fine_time >= Dout,
                ArgmaxDecodeStatus.INVALID_FINE_TIME)
            status = self._set_first_status(
                cp, status, (fine_time % dt) != 0,
                ArgmaxDecodeStatus.INVALID_TIME_GRANULARITY)
            status = self._set_first_status(
                cp, status, ~absolute_ok,
                ArgmaxDecodeStatus.ABSOLUTE_TIME_OVERFLOW)

            # This aggregate boolean is the sole normal value synchronized to
            # the host.  Candidate-sized status and decoded columns stay device
            # resident; only the exceptional branch snapshots one offending row.
            if n and not bool(cp.all(status == 0).item()):
                self._raise_first_invalid(cp, status, raw_fields)

            # Integer decoding.  The multiplet lookup supplies the winning
            # inclusive subband and its fine-DM term.  A downsampled primary
            # family searches the upper half of its coarse-delay interval, hence
            # the primary-family offset. Each coarse DM bin now contains
            # extra_dm_count sub-bins, selected by the independent mu byte.
            fmin = t["m_fmin"][flat_m]
            fmax = t["m_fmax"][flat_m]
            high_lag = t["m_high_lag"][flat_m]
            bandwidth = t["m_bandwidth"][flat_m]
            dfine = t["m_dfine"][flat_m]
            dhi = (idm + cp.where(primary > 0, ndm_out, 0)) * extra_dm_count + extra_dm

            # ``end`` is Tpf+1: coarse-bin start + token fine time + that
            # profile's token-time quantum.  Subtracting high-band lag and
            # across-subband delay yields the CPU decoder's exclusive trailing
            # edges, first in tree time and then scaled to full-resolution input
            # samples.  These signed edges are chunk-relative and may be negative.
            end = itime * Dout + fine_time + dt
            thi = (end - high_lag * dhi) * time_scale
            tlo = (end - (high_lag + bandwidth) * dhi - dfine) * time_scale

            # Physical decoding.  The slope is delay in full-resolution samples
            # per top-level tree-frequency cell.  Extending it over all ntree
            # cells and multiplying by the plan's DM-per-unit-delay gives DM in
            # pc cm^-3.  Integer edges divided by an integer frequency span can
            # make both the DM and final arrival coordinate fractional.
            dslope = (
                (thi - tlo).astype(cp.float64)
                / (fmax - fmin).astype(cp.float64)
            )
            dm = dslope * float(self.ntree) * self.dm_per_unit_delay
            # Profile tables convert the winning temporal filter to its nominal
            # full-resolution width and the centre-of-mass offset from its
            # exclusive trailing edge.  Width is a measured winning-trial value,
            # not an additional fit performed here.
            pf_width = t["profile_width"][flat_profile]
            pf_shift = t["profile_shift"][flat_profile]

            # Extrapolate from the trailing edge at the winning subband's low
            # frequency (tree coordinate fmax + 0.5) to tree coordinate ntree,
            # the complete band's lowest-frequency edge, then subtract the
            # profile centre shift.  The result is chunk-relative and may lie
            # before or after the chunk; adding its signed absolute chunk start
            # gives the unclipped FPGA-relative arrival coordinate.
            #
            # The CPU oracle rounds ``dslope * extrapolation + thi`` as an FMA.
            # CuPy has no public fma ufunc, so this readable GPU version uses
            # separate multiply and add operations.  CPU/GPU parity tests allow
            # an absolute tolerance of 1e-9 full-resolution samples for the
            # resulting harmless last-bit difference.
            extrapolation = float(self.ntree) - 0.5 - fmax.astype(cp.float64)
            relative_toa = (
                dslope * extrapolation + thi.astype(cp.float64) - pf_shift
            )
            absolute_chunk_start = source_chunk * self.nt_in
            toa_sample_abs = absolute_chunk_start.astype(cp.float64) + relative_toa
            width_ms = pf_width * self.time_sample_ms

            # delay_to_frequency is decreasing: the inclusive tree interval
            # [fmin, fmax] corresponds to physical edges [fmax+1, fmin].
            # Construction only gathers GPU lookup columns; it does not copy a
            # candidate-sized result back to the host.
            return GpuDecodedCandidateBatch(
                raw=raw,
                primary_tree_index=primary.astype(cp.int32),
                fmin=fmin,
                fmax=fmax,
                tlo=tlo,
                thi=thi,
                profile=profile,
                dm=dm,
                toa_sample_abs=toa_sample_abs,
                width_samp=pf_width,
                width_ms=width_ms,
                freq_lo_MHz=t["freq_by_delay"][fmax + 1],
                freq_hi_MHz=t["freq_by_delay"][fmin],
                status=status,
            )


__all__ = ["ArgmaxDecodeStatus", "GpuArgmaxDecodeError",
           "GpuArgmaxDecoder", "GpuDecodedCandidateBatch"]
