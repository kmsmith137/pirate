"""Associate duplicate PIRATE candidates into representative events.

A dispersed pulse can be reported by several overlapping dedispersion trees.
This module groups those candidate rows when their beam, primary-tree family,
DM, and absolute low-frequency arrival coordinates are compatible.  It retains
the original candidate table and a member-link table, while the compact event
table takes its measured physical fields from one loudest representative.
Candidate edge/quality flags are never combined: an event receives its
representative's flags and every member's own flags remain accessible through
the candidate table.

All candidate-sized columns remain on one CUDA device.  Independent
``(beam_id, primary_tree_index)`` partitions are grouped by persistent CUDA
blocks, so the representative loop and all candidate comparisons execute on
the GPU.  Compact device-side sorting and remapping restore the exact global
event order after the partitions finish.  Final event/member array transfer
belongs to the catalog/output layer, not this module.

The association rule is intentionally representative-seeded.  Every event is
defined by candidates compatible with its loudest seed, with at most one
candidate selected from each tree.  It is neither transitive connected-component
clustering nor complete-link clustering: compatibility between nonrepresentative
members is not evaluated and cannot grow an event beyond the seed's geometry.
"""

from dataclasses import dataclass, fields
from math import ceil, isfinite
from numbers import Real
import operator
from typing import Any

import cupy as cp
import numpy as np


# Cold-plasma delay constant for MHz, pc cm^-3, and seconds.
_DISPERSION_CONSTANT_S_MHZ2 = 4148.808


_RAW_MODULE_COMPILE_OPTIONS = ("--std=c++11", "--fmad=false")
_RAW_MODULE_BACKEND = "nvrtc"
_THREADS_PER_BLOCK = 256
_SUPPORTED_PREDICATE_DTYPES = frozenset((
    np.dtype(np.float32), np.dtype(np.float64),
))
_CANONICAL_PREDICATE_DTYPES = (
    np.dtype(np.float64),
    np.dtype(np.float64),
    np.dtype(np.float64),
    np.dtype(np.float64),
)


_CUDA_SOURCE_TEMPLATE = r"""
// Header-free for NVRTC installations without host standard include paths.
#define PIRATE_NO_WINNER 0xffffffffffffffffULL

__device__ __forceinline__ unsigned long long pirate_globaltimer_ns()
{
    unsigned long long value;
    // %globaltimer is one device-wide nanosecond counter, unlike clock64(),
    // whose epoch is multiprocessor-local and cannot define one grid deadline.
    asm volatile("mov.u64 %0, %%globaltimer;" : "=l"(value));
    return value;
}

__device__ __forceinline__ int pirate_stop_requested(
        const unsigned long long deadline_ns,
        const unsigned long long timeout_ns,
        const int checks_enabled,
        int *timeout_flag)
{
    if (!checks_enabled) {
        return 0;
    }
    if (atomicAdd(timeout_flag, 0) != 0) {
        return 1;
    }
    if (timeout_ns != 0ULL && pirate_globaltimer_ns() >= deadline_ns) {
        atomicExch(timeout_flag, 1);
        return 1;
    }
    return 0;
}

extern "C" __global__
void group_representative_partitions(
        const long long *partition_offsets,
        const long long *partition_tree_offsets,
        const long long *candidate_index,
        const long long *candidate_tree_slot,
        const int *tree,
        const PIRATE_DM_TYPE *dm,
        const PIRATE_TOA_TYPE *toa_sample_abs,
        const PIRATE_DM_STEP_TYPE *dm_step,
        const PIRATE_TIME_STEP_TYPE *time_step_samples,
        const double dm_tolerance_bins,
        const double time_padding_bins,
        const double residual_slope_lo_samples_per_dm,
        const double residual_slope_hi_samples_per_dm,
        const int use_shared_winners,
        const unsigned long long timeout_ns,
        const int timeout_checks_enabled,
        const long long test_timeout_after_work_tiles,
        const long long test_timeout_after_committed_events,
        unsigned long long *launch_deadline_ns,
        unsigned long long *completed_work_tiles,
        unsigned long long *completed_events,
        int *timeout_flag,
        unsigned long long *global_winners,
        long long *provisional_event_by_position,
        long long *provisional_representative,
        int *provisional_member_count)
{
    const int partition = (int)blockIdx.x;
    const int tid = (int)threadIdx.x;
    const int nthreads = (int)blockDim.x;
    const long long begin = partition_offsets[partition];
    const long long end = partition_offsets[partition + 1];
    const long long tree_begin = partition_tree_offsets[partition];
    const long long tree_end = partition_tree_offsets[partition + 1];
    const long long tree_count = tree_end - tree_begin;

    extern __shared__ unsigned long long shared_winners[];
    unsigned long long *winners = use_shared_winners
        ? shared_winners
        : global_winners + tree_begin;
    __shared__ unsigned long long deadline_ns;
    __shared__ int stop_requested;
    __shared__ int committed_member_count;

    if (tid == 0) {
        if (timeout_ns != 0ULL) {
            const unsigned long long now = pirate_globaltimer_ns();
            const unsigned long long proposed =
                0xffffffffffffffffULL - now < timeout_ns
                ? 0xffffffffffffffffULL : now + timeout_ns;
            const unsigned long long prior = atomicCAS(
                launch_deadline_ns, 0ULL, proposed
            );
            deadline_ns = prior == 0ULL ? proposed : prior;
        } else {
            deadline_ns = 0ULL;
        }
        // Private deterministic test hook.  Normal calls pass -1.  A zero
        // budget requests cancellation before the first event can commit.
        if (test_timeout_after_work_tiles == 0LL
                || test_timeout_after_committed_events == 0LL) {
            atomicExch(timeout_flag, 1);
        }
    }
    __syncthreads();

    int local_event_id = 0;
    for (long long representative_position = begin;
         representative_position < end;
         ++representative_position) {
        if (tid == 0) {
            stop_requested = pirate_stop_requested(
                deadline_ns, timeout_ns, timeout_checks_enabled, timeout_flag
            );
        }
        __syncthreads();
        if (stop_requested) {
            break;
        }
        if (provisional_event_by_position[representative_position] >= 0) {
            continue;
        }

        // Clear the compact per-tree winner workspace cooperatively.  Checking
        // after each block-sized tile bounds cancellation latency even for an
        // unusually large set of trees.
        for (long long tree_tile = 0;
             tree_tile < tree_count;
             tree_tile += nthreads) {
            const long long local_tree = tree_tile + tid;
            if (local_tree < tree_count) {
                winners[local_tree] = PIRATE_NO_WINNER;
            }
            __syncthreads();
            if (tid == 0) {
                stop_requested = pirate_stop_requested(
                    deadline_ns, timeout_ns, timeout_checks_enabled,
                    timeout_flag
                );
            }
            __syncthreads();
            if (stop_requested) {
                break;
            }
        }
        if (stop_requested) {
            break;
        }

        const int representative_tree = tree[representative_position];
        const PIRATE_DM_TYPE representative_dm = dm[representative_position];
        const PIRATE_TOA_TYPE representative_toa =
            toa_sample_abs[representative_position];
        const PIRATE_DM_STEP_TYPE representative_dm_step =
            dm_step[representative_position];
        const PIRATE_TIME_STEP_TYPE representative_time_step =
            time_step_samples[representative_position];

        // Scan in fixed block-sized tiles.  Winners stay provisional until the
        // entire representative event has been scanned, so timeout can discard
        // the in-progress event without exposing a partial assignment.
        for (long long tile_begin = representative_position + 1;
             tile_begin < end;
             tile_begin += nthreads) {
            const long long position = tile_begin + tid;
            if (position < end
                    && provisional_event_by_position[position] < 0
                    && tree[position] != representative_tree) {
                // Preserve CuPy's operation-stage rounding for direct float32
                // and mixed-precision GpuDecodedCandidates inputs.
                const PIRATE_DM_TYPE delta_dm = (PIRATE_DM_TYPE)(
                    dm[position] - representative_dm
                );
                const PIRATE_TOA_TYPE delta_toa = (PIRATE_TOA_TYPE)(
                    toa_sample_abs[position] - representative_toa
                );
                const PIRATE_DM_STEP_TYPE other_dm_step = dm_step[position];
                const PIRATE_DM_STEP_TYPE maximum_dm_step =
                    other_dm_step > representative_dm_step
                    ? other_dm_step : representative_dm_step;
                const PIRATE_DM_STEP_TYPE dm_limit = (PIRATE_DM_STEP_TYPE)(
                    (PIRATE_DM_STEP_TYPE)dm_tolerance_bins * maximum_dm_step
                );
                const PIRATE_DM_TYPE absolute_delta_dm =
                    delta_dm < (PIRATE_DM_TYPE)0 ? -delta_dm : delta_dm;

                if ((PIRATE_DM_COMPARE_TYPE)absolute_delta_dm
                        <= (PIRATE_DM_COMPARE_TYPE)dm_limit) {
                    const PIRATE_DM_TYPE edge_lo = (PIRATE_DM_TYPE)(
                        delta_dm
                        * (PIRATE_DM_TYPE)residual_slope_lo_samples_per_dm
                    );
                    const PIRATE_DM_TYPE edge_hi = (PIRATE_DM_TYPE)(
                        delta_dm
                        * (PIRATE_DM_TYPE)residual_slope_hi_samples_per_dm
                    );
                    const PIRATE_TIME_STEP_TYPE other_time_step =
                        time_step_samples[position];
                    const PIRATE_TIME_STEP_TYPE maximum_time_step =
                        other_time_step > representative_time_step
                        ? other_time_step : representative_time_step;
                    const PIRATE_TIME_STEP_TYPE padding =
                        (PIRATE_TIME_STEP_TYPE)(
                            (PIRATE_TIME_STEP_TYPE)time_padding_bins
                            * maximum_time_step
                        );
                    const PIRATE_DM_TYPE minimum_edge =
                        edge_lo < edge_hi ? edge_lo : edge_hi;
                    const PIRATE_DM_TYPE maximum_edge =
                        edge_lo > edge_hi ? edge_lo : edge_hi;
                    const PIRATE_INTERVAL_TYPE interval_lo =
                        (PIRATE_INTERVAL_TYPE)(
                            (PIRATE_INTERVAL_TYPE)minimum_edge
                            - (PIRATE_INTERVAL_TYPE)padding
                        );
                    const PIRATE_INTERVAL_TYPE interval_hi =
                        (PIRATE_INTERVAL_TYPE)(
                            (PIRATE_INTERVAL_TYPE)maximum_edge
                            + (PIRATE_INTERVAL_TYPE)padding
                        );

                    // CuPy minimum/maximum propagate NaNs.  Inputs are finite,
                    // but an overflowing intermediate must still reject exactly
                    // as the production array comparisons did.
                    if (edge_lo == edge_lo && edge_hi == edge_hi
                            && interval_lo == interval_lo
                            && interval_hi == interval_hi
                            && (PIRATE_TIME_COMPARE_TYPE)delta_toa
                                >= (PIRATE_TIME_COMPARE_TYPE)interval_lo
                            && (PIRATE_TIME_COMPARE_TYPE)delta_toa
                                <= (PIRATE_TIME_COMPARE_TYPE)interval_hi) {
                        const long long local_tree_slot =
                            candidate_tree_slot[position] - tree_begin;
                        atomicMin(
                            &winners[local_tree_slot],
                            (unsigned long long)position
                        );
                    }
                }
            }
            __syncthreads();
            if (tid == 0) {
                if (test_timeout_after_work_tiles > 0LL) {
                    const unsigned long long completed = atomicAdd(
                        completed_work_tiles, 1ULL
                    ) + 1ULL;
                    if (completed >= (unsigned long long)
                            test_timeout_after_work_tiles) {
                        atomicExch(timeout_flag, 1);
                    }
                }
                stop_requested = pirate_stop_requested(
                    deadline_ns, timeout_ns, timeout_checks_enabled,
                    timeout_flag
                );
            }
            __syncthreads();
            if (stop_requested) {
                break;
            }
        }
        if (stop_requested) {
            break;
        }

        // A representative with no later scan tile still needs a final
        // deadline check before entering the indivisible commit section.
        if (tid == 0) {
            stop_requested = pirate_stop_requested(
                deadline_ns, timeout_ns, timeout_checks_enabled, timeout_flag
            );
        }
        __syncthreads();
        if (stop_requested) {
            break;
        }

        // Commit is intentionally indivisible with respect to cancellation.
        // Once it starts, all winners and the seed are assigned before the
        // representative marker publishes this event as valid.  The next
        // deadline check may therefore overshoot by this one bounded pass.
        const long long provisional_slot = begin + (long long)local_event_id;
        if (tid == 0) {
            provisional_event_by_position[representative_position] =
                provisional_slot;
            committed_member_count = 1;
        }
        __syncthreads();
        for (long long local_tree = tid;
             local_tree < tree_count;
             local_tree += nthreads) {
            const unsigned long long winner = winners[local_tree];
            if (winner != PIRATE_NO_WINNER) {
                provisional_event_by_position[(long long)winner] =
                    provisional_slot;
                atomicAdd(&committed_member_count, 1);
            }
        }
        __syncthreads();
        if (tid == 0) {
            provisional_member_count[provisional_slot] =
                committed_member_count;
            // This is the validity/commit marker and is written last.
            provisional_representative[provisional_slot] =
                candidate_index[representative_position];
            if (test_timeout_after_committed_events > 0LL) {
                const unsigned long long completed = atomicAdd(
                    completed_events, 1ULL
                ) + 1ULL;
                if (completed >= (unsigned long long)
                        test_timeout_after_committed_events) {
                    atomicExch(timeout_flag, 1);
                }
            }
            // Observe a deadline crossed by the indivisible commit even when
            // this was the partition's final event and there is no next loop
            // iteration in which to perform the usual pre-event check.
            pirate_stop_requested(
                deadline_ns, timeout_ns, timeout_checks_enabled, timeout_flag
            );
        }
        __syncthreads();
        ++local_event_id;
    }
}
"""


@dataclass(frozen=True)
class ClusteringTolerances:
    """Dimensionless tolerances applied in the native search grids.

    Parameters
    ----------
    dm_tolerance_bins : float
        Maximum DM separation in units of the coarser of the two candidates'
        native tree DM steps.  The default 1.5 allows modest cross-tree
        discretization disagreement without imposing an absolute DM scale.
    time_padding_bins : float
        Symmetric padding added to the residual-dispersion time interval, in
        units of the coarser of the two native tree time steps.  The default
        1.5 matches the DM tolerance's modest cross-tree grid allowance.

    Both values must be finite, real, and non-negative.  Pulse width is
    deliberately absent: grouping tolerances describe search-grid resolution,
    whereas decoded width is a measured property retained for reporting.
    """

    dm_tolerance_bins: float = 1.5
    time_padding_bins: float = 1.5

    def __post_init__(self):
        """Normalize accepted real scalars to float and reject invalid tolerances."""

        for name in ("dm_tolerance_bins", "time_padding_bins"):
            value = getattr(self, name)
            if isinstance(value, bool) or not isinstance(value, Real):
                raise TypeError(f"{name} must be a real scalar")
            value = float(value)
            if not isfinite(value) or value < 0.0:
                raise ValueError(f"{name} must be finite and non-negative")
            object.__setattr__(self, name, value)


@dataclass(frozen=True)
class ClusteringGeometry:
    """Producer search resolution and full-band residual-dispersion geometry.

    The three per-tree fields are one-dimensional CuPy lookup tables of length
    'ntrees' on the grouping device:

    * 'primary_tree_index_by_tree' is int32 and identifies which trees belong
      to the same producer search family;
    * 'dm_step_by_tree' is float64 in pc cm^-3 per native map row;
    * 'time_step_samples_by_tree' is float64 in full-resolution input samples
      per native map column.

    The remaining fields are host scalars.  Frequencies are MHz and
    'time_sample_ms' converts one full-resolution input sample to milliseconds.
    Decoded absolute TOA is referenced to 'reference_freq_MHz', which is
    required to equal the lowest edge of the complete observing band.

    For frequency f, residual slope is

        K_DM * (f_ref**-2 - f**-2) / sample_time_s

    in input samples per pc cm^-3.  The stored low/high slopes bound how the
    low-frequency-referenced TOA can shift when two nearby DM trials describe
    the same signal over any part of the complete band.  At the low reference
    edge the slope is exactly zero; at the high edge it is positive.
    """

    ntrees: int
    primary_tree_index_by_tree: Any
    dm_step_by_tree: Any
    time_step_samples_by_tree: Any
    reference_freq_MHz: float
    full_band_freq_lo_MHz: float
    full_band_freq_hi_MHz: float
    time_sample_ms: float
    residual_slope_lo_samples_per_dm: float
    residual_slope_hi_samples_per_dm: float

    @classmethod
    def from_plan(cls, plan):
        """Construct grouping geometry from a producer plan.

        Parameters
        ----------
        plan
            The reconstructed producer DedispersionPlan whose tree maps were
            peak-found and decoded.

        Returns
        -------
        ClusteringGeometry
            Validated geometry.  Three small per-tree arrays are uploaded to the
            currently active CUDA device; scalar band/time metadata stays on the
            CPU.  No candidate-sized data is transferred.

        Raises
        ------
        ValueError
            If the plan has no trees, invalid time or frequency sampling, or a
            tree with non-positive map dimensions/native resolution.
        """

        # Read the compact producer metadata on CPU.  Tree-specific output maps
        # need not share shape, so grouping keeps only their physical row/column
        # resolutions rather than attempting to stack or regrid them.
        trees = tuple(plan.trees)
        nt_in = int(plan.nt_in)
        time_sample_ms = float(plan.config.time_sample_ms)
        edges = tuple(float(x) for x in plan.config.zone_freq_edges)
        if not trees:
            raise ValueError("the producer plan contains no trees")
        if nt_in <= 0 or not isfinite(time_sample_ms) or time_sample_ms <= 0.0:
            raise ValueError("the producer plan has invalid time sampling")
        if (len(edges) < 2 or not 0.0 < edges[0] < edges[-1]
                or not all(isfinite(x) for x in edges)):
            raise ValueError("the producer plan has invalid frequency edges")

        families = []
        dm_steps = []
        time_steps = []
        for itree, tree in enumerate(trees):
            ndm = int(tree.ndm_out)
            ntime = int(tree.nt_out)
            family = int(tree.primary_tree_index)
            if ndm <= 0 or ntime <= 0 or family < 0:
                raise ValueError(f"tree {itree} has invalid map metadata")
            dm_step = (
                float(tree.dm_max) - float(tree.dm_min)
            ) / float(ndm)
            # Every tree spans the same input chunk duration but may expose a
            # different number of coarse time cells.
            time_step = float(nt_in) / float(ntime)
            if not isfinite(dm_step) or dm_step <= 0.0 or time_step <= 0.0:
                raise ValueError(f"tree {itree} has invalid native resolution")
            families.append(family)
            dm_steps.append(dm_step)
            time_steps.append(time_step)

        freq_lo, freq_hi = edges[0], edges[-1]
        sample_time_s = time_sample_ms / 1.0e3
        # Cold-plasma residual delay relative to the catalog's low-frequency TOA
        # convention.  Evaluating both complete-band edges gives a conservative
        # interval independent of which subband won either candidate.
        slope = lambda f: _DISPERSION_CONSTANT_S_MHZ2 * (
            freq_lo**-2 - f**-2
        ) / sample_time_s
        # This is the only setup CPU-to-GPU boundary: the tables have one entry
        # per tree and are reused for every candidate comparison.
        geometry = cls(
            ntrees=len(trees),
            primary_tree_index_by_tree=cp.asarray(families, dtype=cp.int32),
            dm_step_by_tree=cp.asarray(dm_steps, dtype=cp.float64),
            time_step_samples_by_tree=cp.asarray(
                time_steps, dtype=cp.float64
            ),
            reference_freq_MHz=freq_lo,
            full_band_freq_lo_MHz=freq_lo,
            full_band_freq_hi_MHz=freq_hi,
            time_sample_ms=time_sample_ms,
            residual_slope_lo_samples_per_dm=slope(freq_lo),
            residual_slope_hi_samples_per_dm=slope(freq_hi),
        )
        _validate_geometry(geometry)
        return geometry


@dataclass(frozen=True)
class GpuDecodedCandidates:
    """Candidate table used by grouping and downstream reporting.

    Every field is a one-dimensional CuPy array of equal candidate length on the
    same device as the ClusteringGeometry tables.  The normalized schema produced
    by 'from_decoder_result' is:

    * beam_id, tree, idm, itime, and primary_tree_index: int32 producer
      provenance;
    * source_chunk_index: int64 source-map chunk index;
    * argmax_token: uint32 and edge_flags: uint8, both preserved bit-for-bit
      from peakfinding;
    * snr, dm, toa_sample_abs, width_samp, width_ms, freq_lo_MHz, and
      freq_hi_MHz: float64 physical/reporting values;
    * dm_step and time_step_samples: float64 native resolution looked up from
      the candidate's tree.

    DM and DM step use pc cm^-3.  Time and width in samples use the producer's
    full-resolution input sampling; width_ms is milliseconds and frequency
    bounds are MHz.  itime is a coordinate within one tree's source chunk,
    whereas toa_sample_abs is a potentially fractional arrival coordinate from
    FPGA sequence zero, referenced to the complete band's lowest frequency.
    Grouping uses absolute TOA, so otherwise compatible candidates can associate
    across a chunk boundary.  A STARTUP_INCOMPLETE edge bit is carried without
    changing S/N or any decoded physical field; it warns that zero padding may
    have biased the measured S/N and winning-profile width.
    """

    beam_id: Any
    source_chunk_index: Any
    tree: Any
    idm: Any
    itime: Any
    snr: Any
    argmax_token: Any
    edge_flags: Any
    dm: Any
    toa_sample_abs: Any
    width_samp: Any
    width_ms: Any
    freq_lo_MHz: Any
    freq_hi_MHz: Any
    primary_tree_index: Any
    dm_step: Any
    time_step_samples: Any

    def __len__(self):
        """Return candidate count from CuPy array metadata without a host copy."""

        return int(self.beam_id.size)

    @classmethod
    def from_decoder_result(cls, decoded, geometry):
        """Normalize decoder output into the grouping table on the GPU.

        'decoded' may already be a GpuDecodedCandidates, or it may be a decoder
        batch whose provenance columns live in decoded.raw and physical columns
        live on decoded.  All required fields must be one-dimensional CuPy
        arrays of equal length on one device.

        GPU astype(copy=False) operations enforce the schema dtypes, and two
        per-candidate resolution columns are gathered from geometry by tree
        index.  Candidate edge flags, including quality bits such as startup
        incompleteness, remain uint8 and are not interpreted here.

        Returns
        -------
        GpuDecodedCandidates
            A validated GPU table.  An input already of this type is returned
            unchanged after validation.

        Notes
        -----
        Candidate columns are never copied to CPU.  Aggregate all checks do
        synchronize scalar booleans so invalid decoder status, device mismatch,
        or nonphysical rows can fail explicitly.
        """

        _validate_geometry(geometry)
        if isinstance(decoded, cls):
            _validate_candidates(decoded, geometry)
            return decoded
        raw = getattr(decoded, "raw", None)
        valid_mask = getattr(decoded, "valid_mask", None)
        # Decoding is an all-or-nothing boundary.  Refuse a batch that exposes a
        # non-OK status rather than allowing invalid lookup-derived numbers into
        # scientific grouping.
        if valid_mask is not None and not bool(cp.all(valid_mask).item()):
            raise ValueError("decoder result contains an invalid candidate")

        def read(name):
            """Return one required decoded/raw GPU column without copying it."""

            owner = decoded if hasattr(decoded, name) else raw
            if owner is None or not hasattr(owner, name):
                raise TypeError(f"decoder result is missing {name!r}")
            value = getattr(owner, name)
            if not isinstance(value, cp.ndarray) or value.ndim != 1:
                raise TypeError(f"decoder field {name!r} must be a 1-D CuPy array")
            return value

        names = (
            "beam_id", "source_chunk_index", "tree", "idm", "itime", "snr",
            "argmax_token", "edge_flags", "dm", "toa_sample_abs", "width_samp",
            "width_ms", "freq_lo_MHz", "freq_hi_MHz", "primary_tree_index",
        )
        source = {name: read(name) for name in names}
        n = int(source["beam_id"].size)
        device = int(source["beam_id"].device.id)
        if any(int(x.size) != n or int(x.device.id) != device
               for x in source.values()):
            raise ValueError("decoder fields differ in length or CUDA device")

        # Normalize on-device.  copy=False preserves an existing correctly typed
        # column, while a required cast allocates only another GPU column.
        typed = {
            "beam_id": (cp.int32, source["beam_id"]),
            "source_chunk_index": (cp.int64, source["source_chunk_index"]),
            "tree": (cp.int32, source["tree"]),
            "idm": (cp.int32, source["idm"]),
            "itime": (cp.int32, source["itime"]),
            "snr": (cp.float64, source["snr"]),
            "argmax_token": (cp.uint32, source["argmax_token"]),
            "edge_flags": (cp.uint8, source["edge_flags"]),
            "dm": (cp.float64, source["dm"]),
            "toa_sample_abs": (cp.float64, source["toa_sample_abs"]),
            "width_samp": (cp.float64, source["width_samp"]),
            "width_ms": (cp.float64, source["width_ms"]),
            "freq_lo_MHz": (cp.float64, source["freq_lo_MHz"]),
            "freq_hi_MHz": (cp.float64, source["freq_hi_MHz"]),
            "primary_tree_index": (cp.int32, source["primary_tree_index"]),
        }
        columns = {
            name: value.astype(dtype, copy=False)
            for name, (dtype, value) in typed.items()
        }
        tree = columns["tree"]
        if n and not bool(cp.all(
                (tree >= 0) & (tree < geometry.ntrees)).item()):
            raise ValueError("decoded candidate has an invalid tree index")
        # Attach each candidate's native grid resolution by a device-side gather
        # rather than duplicating producer-plan metadata in the decoder.
        columns["dm_step"] = geometry.dm_step_by_tree[tree]
        columns["time_step_samples"] = (
            geometry.time_step_samples_by_tree[tree]
        )
        candidates = cls(**columns)
        _validate_candidates(candidates, geometry)
        return candidates


@dataclass(frozen=True)
class GpuEventTable:
    """One GPU row per grouped event, derived only from its representative.

    event_id and representative_candidate_index are int64; member_count is
    int32.  All remaining columns have the same dtypes, units, and coordinate
    conventions as GpuDecodedCandidates.

    There is intentionally no averaging, refitting, or bitwise flag reduction.
    S/N, DM, TOA, width, winning subband, provenance, and event-level edge_flags
    are gathered from the event's loudest representative.  A nonrepresentative
    member may therefore carry a different edge flag; that per-member value
    remains available through GpuMemberTable.candidate_index and the result's
    full candidate table.
    """

    event_id: Any
    representative_candidate_index: Any
    member_count: Any
    beam_id: Any
    primary_tree_index: Any
    tree: Any
    source_chunk_index: Any
    idm: Any
    itime: Any
    snr: Any
    argmax_token: Any
    edge_flags: Any
    dm: Any
    toa_sample_abs: Any
    width_samp: Any
    width_ms: Any
    freq_lo_MHz: Any
    freq_hi_MHz: Any

    def __len__(self):
        """Return event count from array metadata without transferring rows."""

        return int(self.event_id.size)


@dataclass(frozen=True)
class GpuMemberTable:
    """GPU links from grouped events to their original candidate rows.

    There is one row per retained candidate.  A complete result retains every
    input row; a timed-out result retains only rows in fully committed events.
    event_id and candidate_index are int64; is_representative is boolean.
    Candidate physical fields and per-member edge_flags are deliberately not
    duplicated: index GpuClusteringResult.candidates with candidate_index.  Rows
    are grouped by event and retain deterministic candidate processing order
    within each event.
    """

    event_id: Any
    candidate_index: Any
    is_representative: Any

    def __len__(self):
        """Return member-link count, equal to retained candidate count."""

        return int(self.event_id.size)


@dataclass(frozen=True)
class GpuClusteringResult:
    """GPU-resident complete result or clean event-boundary timeout subset.

    A complete result retains every normalized input row.  A timed-out result
    contains only candidates belonging to fully committed events, in their
    original relative input order.  ``input_candidate_index`` is int64 and maps
    each retained row back to the grouping call's input index.  Event
    representative indices, member candidate indices, and ``candidate_event_id``
    address this returned (possibly filtered) candidate table and are always
    compact and non-negative; no provisional or dangling assignment is exposed.

    ``timed_out`` and ``complete`` are explicit host control scalars and are
    logical opposites.  All scientific/result arrays remain on the GPU, and no
    candidate or event array has crossed to CPU at this boundary.
    """

    candidates: GpuDecodedCandidates
    events: GpuEventTable
    members: GpuMemberTable
    candidate_event_id: Any
    input_candidate_index: Any = None
    timed_out: bool = False
    complete: bool = True

    def __post_init__(self):
        """Fill backward-compatible identity indices and enforce status shape."""

        if self.input_candidate_index is None:
            with cp.cuda.Device(int(self.candidates.beam_id.device.id)):
                object.__setattr__(
                    self,
                    "input_candidate_index",
                    cp.arange(len(self.candidates), dtype=cp.int64),
                )
        index = self.input_candidate_index
        if (not isinstance(index, cp.ndarray) or index.ndim != 1
                or index.dtype != cp.int64
                or int(index.size) != len(self.candidates)
                or int(index.device.id)
                    != int(self.candidates.beam_id.device.id)):
            raise TypeError(
                "input_candidate_index must be one int64 CuPy index per candidate"
            )
        if type(self.timed_out) is not bool or type(self.complete) is not bool:
            raise TypeError("timed_out and complete must be bool")
        if self.complete == self.timed_out:
            raise ValueError("complete must be the logical inverse of timed_out")


def _require_current_grouping_device(geometry):
    """Reject cross-device execution before CuPy allocates mixed buffers.

    Geometry construction and decoder normalization target the active CUDA
    device.  Array operands remember their owner, but allocation helpers and
    RawModule dispatch use the current device; mixing those contexts could
    otherwise trigger peer access and produce an invalid result.

    Invalid/non-array geometry is left to _validate_geometry so callers retain
    its established TypeError diagnostics.
    """

    if (isinstance(geometry, ClusteringGeometry)
            and isinstance(geometry.dm_step_by_tree, cp.ndarray)
            and int(geometry.dm_step_by_tree.device.id)
                != int(cp.cuda.Device().id)):
        raise ValueError(
            "grouping geometry must reside on the current CUDA device"
        )


def _validate_geometry(geometry):
    """Check geometry shape, device ownership, and physical invariants.

    The three tree tables must be one-dimensional CuPy arrays of length
    geometry.ntrees on one device.  The complete-band low edge must be the TOA
    reference, and all per-tree resolutions must be finite and positive.
    Structural checks use array metadata on the host; the elementwise table
    check remains on GPU until one aggregate boolean is synchronized.

    This helper returns None on success and raises TypeError or ValueError
    before candidate association if the setup cannot define meaningful
    physical comparisons.
    """

    if not isinstance(geometry, ClusteringGeometry) or geometry.ntrees <= 0:
        raise TypeError("geometry must be a non-empty ClusteringGeometry")
    tables = (
        geometry.primary_tree_index_by_tree,
        geometry.dm_step_by_tree,
        geometry.time_step_samples_by_tree,
    )
    if any(not isinstance(x, cp.ndarray) or x.ndim != 1
           or int(x.size) != geometry.ntrees for x in tables):
        raise TypeError("geometry tree tables must be one-dimensional CuPy arrays")
    device = int(tables[0].device.id)
    if any(int(x.device.id) != device for x in tables):
        raise ValueError("geometry tables span CUDA devices")
    if not (
        0.0 < geometry.full_band_freq_lo_MHz
        == geometry.reference_freq_MHz
        < geometry.full_band_freq_hi_MHz
    ):
        raise ValueError("TOA reference must be the complete-band low frequency")
    physical_scalars = (
        geometry.time_sample_ms,
        geometry.residual_slope_lo_samples_per_dm,
        geometry.residual_slope_hi_samples_per_dm,
    )
    if not all(isfinite(float(x)) for x in physical_scalars):
        raise ValueError("geometry contains a non-finite physical scalar")
    # Validate all tree rows in one GPU expression.  Only the reduction result,
    # not the small lookup tables themselves, crosses back to the host.
    valid = (
        (tables[0] >= 0)
        & cp.isfinite(tables[1]) & (tables[1] > 0.0)
        & cp.isfinite(tables[2]) & (tables[2] > 0.0)
    )
    if not bool(cp.all(valid).item()):
        raise ValueError("geometry contains invalid tree metadata")


def _validate_candidates(candidates, geometry):
    """Validate candidate table residence and scientific row invariants.

    Every dataclass field must be a one-dimensional CuPy array with the same
    length and CUDA device as geometry.  Tokens and edge flags are required to
    retain their uint32/uint8 schema dtypes.  For nonempty input, one GPU
    predicate checks finite physical values, positive widths, ordered subband
    frequencies, legal tree indices, and exact agreement with the per-tree
    family/resolution lookup tables.  Only the aggregate predicate is copied as
    a scalar.

    The function returns None, accepts an empty table, and raises before grouping
    if any row could make association physically or numerically ambiguous.
    """

    n = len(candidates)
    device = int(geometry.dm_step_by_tree.device.id)
    for field in fields(GpuDecodedCandidates):
        value = getattr(candidates, field.name)
        if not isinstance(value, cp.ndarray) or value.ndim != 1:
            raise TypeError(f"candidate field {field.name!r} must be 1-D CuPy")
        if int(value.size) != n or int(value.device.id) != device:
            raise ValueError("candidate fields differ in length or CUDA device")
    if (candidates.argmax_token.dtype != cp.uint32
            or candidates.edge_flags.dtype != cp.uint8):
        raise TypeError("token and edge flags must retain uint32/uint8 dtypes")
    if not n:
        return
    tree = candidates.tree
    safe_tree = cp.clip(tree, 0, geometry.ntrees - 1)
    # Use a clipped index only to keep the validation gather in bounds.  The
    # explicit tree-range predicate still makes every out-of-range row invalid.
    valid = (
        (tree >= 0) & (tree < geometry.ntrees)
        & cp.isfinite(candidates.snr)
        & cp.isfinite(candidates.dm)
        & cp.isfinite(candidates.toa_sample_abs)
        & cp.isfinite(candidates.width_samp) & (candidates.width_samp > 0.0)
        & cp.isfinite(candidates.width_ms) & (candidates.width_ms > 0.0)
        & cp.isfinite(candidates.freq_lo_MHz)
        & cp.isfinite(candidates.freq_hi_MHz)
        & (candidates.freq_lo_MHz < candidates.freq_hi_MHz)
        & (candidates.primary_tree_index
           == geometry.primary_tree_index_by_tree[safe_tree])
        & (candidates.dm_step == geometry.dm_step_by_tree[safe_tree])
        & (candidates.time_step_samples
           == geometry.time_step_samples_by_tree[safe_tree])
    )
    if not bool(cp.all(valid).item()):
        raise ValueError("decoded candidates contain an invalid scientific row")


def compatible_with_representative(
        candidates, representative_index, other_indices, geometry,
        config=None):
    """Test candidate association with one representative on the GPU.

    Parameters
    ----------
    candidates : GpuDecodedCandidates
        Validated candidate table.  DM uses pc cm^-3 and absolute TOA uses
        full-resolution samples at the complete-band low-frequency reference.
    representative_index : int
        Scalar index of the event seed.
    other_indices : array-like
        One-dimensional candidate indices, converted to int64 on the current
        GPU.  The returned mask has this same logical length.
    geometry : ClusteringGeometry
        Tree resolution and complete-band residual-delay slopes.
    config : ClusteringTolerances or None
        Dimensionless native-bin tolerances; None selects defaults.

    Returns
    -------
    cupy.ndarray
        Boolean compatibility mask resident on the GPU.

    Notes
    -----
    Let ΔDM = DM_other - DM_rep and Δt = TOA_other - TOA_rep.  DM compatibility
    requires

        abs(ΔDM) <= dm_tolerance_bins * max(dm_step_other, dm_step_rep).

    A DM mismatch predicts a residual arrival-time displacement
    slope(f) * ΔDM.  Over the complete band the allowed interval is bounded by
    the stored low/high edge slopes, with symmetric padding

        P = time_padding_bins * max(time_step_other, time_step_rep)

    and the time requirement is

        min(slope_lo*ΔDM, slope_hi*ΔDM) - P <= Δt
            <= max(slope_lo*ΔDM, slope_hi*ΔDM) + P.

    Taking min/max is important for negative ΔDM, which reverses the edge order.
    Candidates must also have the same beam and primary-tree family, and must
    originate from different trees.  The latter enforces the event invariant of
    at most one candidate per tree; same-tree peaks remain separate.

    Decoded width is intentionally not part of compatibility.  It is a measured
    winning-profile property that can vary across trees, while these tolerances
    are anchored to the coarser searched DM/time grid.
    """

    config = ClusteringTolerances() if config is None else config
    if not isinstance(config, ClusteringTolerances):
        raise TypeError("config must be ClusteringTolerances")
    rep = int(representative_index)
    other = cp.asarray(other_indices, dtype=cp.int64)
    if other.ndim != 1:
        raise ValueError("other_indices must be one-dimensional")

    # Differences are always oriented other-minus-representative.  The sign is
    # retained because it predicts the direction of residual dispersion in TOA.
    ddm = candidates.dm[other] - candidates.dm[rep]
    dtoa = candidates.toa_sample_abs[other] - candidates.toa_sample_abs[rep]
    dm_limit = config.dm_tolerance_bins * cp.maximum(
        candidates.dm_step[other], candidates.dm_step[rep]
    )
    edge_lo = ddm * geometry.residual_slope_lo_samples_per_dm
    edge_hi = ddm * geometry.residual_slope_hi_samples_per_dm
    padding = config.time_padding_bins * cp.maximum(
        candidates.time_step_samples[other],
        candidates.time_step_samples[rep],
    )
    # Combine categorical provenance requirements with the two physical
    # inequalities in one candidate-sized GPU mask.
    return (
        (candidates.beam_id[other] == candidates.beam_id[rep])
        & (candidates.primary_tree_index[other]
           == candidates.primary_tree_index[rep])
        & (candidates.tree[other] != candidates.tree[rep])
        & (cp.abs(ddm) <= dm_limit)
        & (dtoa >= cp.minimum(edge_lo, edge_hi) - padding)
        & (dtoa <= cp.maximum(edge_lo, edge_hi) + padding)
    )


def _processing_order(candidates):
    """Return deterministic candidate indices in representative priority order.

    The primary key is descending S/N.  Equal-S/N rows are ordered ascending by
    beam, primary-tree family, source chunk, tree, coarse DM, coarse time, and
    argmax token, with original input order as the final stable tie-breaker.
    Exact integer provenance is sorted in its native dtype rather than packed
    into a floating-point compound key.  The returned int64 CuPy array remains
    on the GPU.
    """

    order = cp.arange(len(candidates), dtype=cp.int64)
    keys = (
        candidates.argmax_token, candidates.itime, candidates.idm,
        candidates.tree, candidates.source_chunk_index,
        candidates.primary_tree_index, candidates.beam_id, -candidates.snr,
    )
    # Stable sorts are applied from least- to most-significant key.  This avoids
    # coercing exact uint32/int64 provenance into floats and makes repeat runs
    # independent of an implementation-specific unstable sort.
    for key in keys:
        order = order[cp.argsort(key[order], kind="stable")]
    return order


def _predicate_dtypes(candidates):
    """Return exact arithmetic dtypes in DM/TOA/DM-step/time-step order."""

    result = tuple(np.dtype(column.dtype) for column in (
        candidates.dm,
        candidates.toa_sample_abs,
        candidates.dm_step,
        candidates.time_step_samples,
    ))
    unsupported = sorted({
        dtype.name for dtype in result
        if dtype not in _SUPPORTED_PREDICATE_DTYPES
    })
    if unsupported:
        raise TypeError(
            "GPU predicate columns must use float32 or float64; "
            "decoder-normalized inputs use float64 (unsupported: "
            + ", ".join(unsupported) + ")"
        )
    return result


def _render_cuda_source(predicate_dtypes):
    """Specialize kernel pointer/intermediate types to CuPy promotion rules."""

    dm_dtype, toa_dtype, dm_step_dtype, time_step_dtype = predicate_dtypes
    interval_dtype = np.promote_types(dm_dtype, time_step_dtype)
    dm_compare_dtype = np.promote_types(dm_dtype, dm_step_dtype)
    time_compare_dtype = np.promote_types(toa_dtype, interval_dtype)
    cuda_names = {
        np.dtype(np.float32): "float",
        np.dtype(np.float64): "double",
    }
    replacements = {
        "PIRATE_DM_TYPE": cuda_names[dm_dtype],
        "PIRATE_TOA_TYPE": cuda_names[toa_dtype],
        "PIRATE_DM_STEP_TYPE": cuda_names[dm_step_dtype],
        "PIRATE_TIME_STEP_TYPE": cuda_names[time_step_dtype],
        "PIRATE_INTERVAL_TYPE": cuda_names[interval_dtype],
        "PIRATE_DM_COMPARE_TYPE": cuda_names[dm_compare_dtype],
        "PIRATE_TIME_COMPARE_TYPE": cuda_names[time_compare_dtype],
    }
    source = _CUDA_SOURCE_TEMPLATE
    for placeholder, cuda_name in replacements.items():
        source = source.replace(placeholder, cuda_name)
    return source


_CUDA_SOURCE = _render_cuda_source(_CANONICAL_PREDICATE_DTYPES)
_MODULE_BY_DEVICE = {}


def _raw_grouping_kernel(
        predicate_dtypes=_CANONICAL_PREDICATE_DTYPES):
    """Return a device/dtype-local NVRTC kernel, compiling on first use."""

    device_id = int(cp.cuda.Device().id)
    predicate_dtypes = tuple(np.dtype(dtype) for dtype in predicate_dtypes)
    cache_key = (device_id, tuple(dtype.str for dtype in predicate_dtypes))
    module = _MODULE_BY_DEVICE.get(cache_key)
    if module is None:
        module = cp.RawModule(
            code=_render_cuda_source(predicate_dtypes),
            options=_RAW_MODULE_COMPILE_OPTIONS,
            backend=_RAW_MODULE_BACKEND,
        )
        _MODULE_BY_DEVICE[cache_key] = module
    return module.get_function("group_representative_partitions")


@dataclass(frozen=True)
class _PartitionLayout:
    """GPU-only candidate and distinct-tree layout for one kernel launch."""

    candidate_order: Any
    partition_offsets: Any
    partition_tree_offsets: Any
    tree_slot_by_position: Any
    partition_count: int
    distinct_partition_tree_count: int
    maximum_trees_per_partition: int


def _stable_sort_from_least_to_most(indices, keys):
    """Apply a fixed schema-key sequence without packing or coercion."""

    result = indices
    for key in keys:  # Fixed schema loop, never candidate/event controlled.
        result = result[cp.argsort(key[result], kind="stable")]
    return result


def _partition_layout(candidates, order):
    """Build exact beam/family partitions and compact partition/tree slots."""

    n = len(candidates)
    if not n:
        empty = cp.empty(0, dtype=cp.int64)
        return _PartitionLayout(
            candidate_order=empty,
            partition_offsets=cp.zeros(1, dtype=cp.int64),
            partition_tree_offsets=cp.zeros(1, dtype=cp.int64),
            tree_slot_by_position=empty,
            partition_count=0,
            distinct_partition_tree_count=0,
            maximum_trees_per_partition=0,
        )

    # Stable sorting starts from global representative priority, so rank inside
    # every independent partition remains exact.
    candidate_order = _stable_sort_from_least_to_most(
        order,
        (candidates.primary_tree_index, candidates.beam_id),
    )
    beam = candidates.beam_id[candidate_order]
    family = candidates.primary_tree_index[candidate_order]
    partition_boundary = cp.concatenate((
        cp.ones(1, dtype=cp.bool_),
        (beam[1:] != beam[:-1]) | (family[1:] != family[:-1]),
    ))
    partition_starts = cp.flatnonzero(partition_boundary).astype(
        cp.int64, copy=False
    )
    partition_offsets = cp.concatenate((
        partition_starts,
        cp.asarray([n], dtype=cp.int64),
    ))
    partition_count = int(partition_starts.size)
    partition_by_position = cp.cumsum(
        partition_boundary, dtype=cp.int64
    ) - 1
    partition_by_candidate = cp.empty(n, dtype=cp.int64)
    partition_by_candidate[candidate_order] = partition_by_position

    # Assign one compact winner slot to each distinct (partition, tree) pair.
    pair_order = _stable_sort_from_least_to_most(
        cp.arange(n, dtype=cp.int64),
        (candidates.tree, partition_by_candidate),
    )
    pair_partition = partition_by_candidate[pair_order]
    pair_tree = candidates.tree[pair_order]
    pair_boundary = cp.concatenate((
        cp.ones(1, dtype=cp.bool_),
        (pair_partition[1:] != pair_partition[:-1])
        | (pair_tree[1:] != pair_tree[:-1]),
    ))
    tree_slot_by_candidate = cp.empty(n, dtype=cp.int64)
    tree_slot_by_candidate[pair_order] = (
        cp.cumsum(pair_boundary, dtype=cp.int64) - 1
    )
    tree_slot_by_position = tree_slot_by_candidate[candidate_order]
    distinct_tree_count = int(cp.count_nonzero(pair_boundary).item())
    unique_pair_partitions = pair_partition[pair_boundary]
    tree_counts = cp.bincount(
        unique_pair_partitions, minlength=partition_count
    ).astype(cp.int64, copy=False)
    partition_tree_offsets = cp.concatenate((
        cp.zeros(1, dtype=cp.int64),
        cp.cumsum(tree_counts, dtype=cp.int64),
    ))
    maximum_trees = int(cp.max(tree_counts).item())
    return _PartitionLayout(
        candidate_order=candidate_order,
        partition_offsets=partition_offsets,
        partition_tree_offsets=partition_tree_offsets,
        tree_slot_by_position=tree_slot_by_position,
        partition_count=partition_count,
        distinct_partition_tree_count=distinct_tree_count,
        maximum_trees_per_partition=maximum_trees,
    )


def _shared_winner_policy(maximum_trees):
    """Use at most half the device's per-block shared memory for winners."""

    if maximum_trees <= 0:
        return False, 0
    properties = cp.cuda.runtime.getDeviceProperties(int(cp.cuda.Device().id))
    shared_capacity = int(properties["sharedMemPerBlock"])
    required = maximum_trees * np.dtype(np.uint64).itemsize
    use_shared = required <= shared_capacity // 2
    return use_shared, required if use_shared else 0


def _timeout_nanoseconds(timeout_ms):
    """Validate a public timeout and round positive values up to one ns."""

    if isinstance(timeout_ms, bool) or not isinstance(timeout_ms, Real):
        raise TypeError("timeout_ms must be a real scalar")
    value = float(timeout_ms)
    if not isfinite(value) or value < 0.0:
        raise ValueError("timeout_ms must be finite and non-negative")
    if value == 0.0:
        return 0
    maximum_ms = float(np.iinfo(np.uint64).max) / 1.0e6
    if value > maximum_ms:
        raise ValueError("timeout_ms is too large for the GPU deadline")
    nanoseconds = ceil(value * 1.0e6)
    if nanoseconds > np.iinfo(np.uint64).max:
        raise ValueError("timeout_ms is too large for the GPU deadline")
    return nanoseconds


def _test_counter_limit(value, name):
    """Validate a private single-partition device-timeout test hook."""

    if value is None:
        return -1
    if isinstance(value, (bool, np.bool_)):
        raise TypeError(f"{name} must be an integer")
    try:
        value = operator.index(value)
    except TypeError as exc:
        raise TypeError(
            f"{name} must be an integer"
        ) from exc
    if value < 0 or value > np.iinfo(np.int64).max:
        raise ValueError(f"{name} must be a non-negative int64")
    return value


def _take_candidates(candidates, indices):
    """Gather every candidate column with one local int64 GPU index array."""

    return GpuDecodedCandidates(**{
        field.name: getattr(candidates, field.name)[indices]
        for field in fields(GpuDecodedCandidates)
    })


def _construct_grouping_result(
        candidates, order, assignment, representatives, counts,
        input_candidate_index, *, timed_out):
    """Construct exact tables from compact, fully committed local indices."""

    nevents = int(representatives.size)
    events = GpuEventTable(
        event_id=cp.arange(nevents, dtype=cp.int64),
        representative_candidate_index=representatives,
        member_count=counts,
        beam_id=candidates.beam_id[representatives],
        primary_tree_index=candidates.primary_tree_index[representatives],
        tree=candidates.tree[representatives],
        source_chunk_index=candidates.source_chunk_index[representatives],
        idm=candidates.idm[representatives],
        itime=candidates.itime[representatives],
        snr=candidates.snr[representatives],
        argmax_token=candidates.argmax_token[representatives],
        edge_flags=candidates.edge_flags[representatives],
        dm=candidates.dm[representatives],
        toa_sample_abs=candidates.toa_sample_abs[representatives],
        width_samp=candidates.width_samp[representatives],
        width_ms=candidates.width_ms[representatives],
        freq_lo_MHz=candidates.freq_lo_MHz[representatives],
        freq_hi_MHz=candidates.freq_hi_MHz[representatives],
    )
    member_order = order[cp.argsort(assignment[order], kind="stable")]
    member_event_id = assignment[member_order]
    members = GpuMemberTable(
        event_id=member_event_id,
        candidate_index=member_order,
        is_representative=(
            member_order == representatives[member_event_id]
        ),
    )
    return GpuClusteringResult(
        candidates=candidates,
        events=events,
        members=members,
        candidate_event_id=assignment,
        input_candidate_index=input_candidate_index,
        timed_out=timed_out,
        complete=not timed_out,
    )


def _cluster_candidates_serial_oracle(decoded, geometry, *, config=None):
    """Reference the original representative algorithm for parity tests.

    This private implementation deliberately retains the host-controlled
    representative loop.  Production calls use :func:`cluster_candidates`, whose
    persistent kernel has the same scientific semantics without per-event
    Python control.  Keeping this small oracle independent makes exact boundary,
    ordering, and dtype tests sensitive to regressions in the CUDA work split.

    Parameters
    ----------
    decoded
        A valid decoder batch or GpuDecodedCandidates table.  Candidate columns
        must occupy one CUDA device.
    geometry : ClusteringGeometry
        Producer tree resolutions and complete-band time/DM geometry on that
        same device.
    config : ClusteringTolerances or None, keyword-only
        Association tolerances; None uses ClusteringTolerances defaults.

    Returns
    -------
    GpuClusteringResult
        The normalized candidates, representative-derived event rows, member
        links, and one event assignment per original candidate, all GPU-resident.

    Algorithm
    ---------
    Candidates are visited in deterministic decreasing-S/N order.  The loudest
    still-unassigned candidate seeds the next event.  Every still-unassigned row
    is compared only with that representative using
    :func:`compatible_with_representative`.  For each compatible *other* tree,
    the first row in global processing order -- therefore the loudest with
    deterministic tie-breaking -- joins the event.  The seed and selected rows
    are assigned permanently before the next seed is chosen.

    This construction guarantees one candidate per tree in an event.  It is not
    transitive: a candidate compatible only with a nonrepresentative member does
    not join.  It is not complete-link: selected nonrepresentative members are
    not required to be mutually compatible.  These are intentional semantics,
    not approximations hidden by the implementation.

    Event physical/provenance fields, including edge flags, are gathered from
    the representative without refitting or flag union.  Per-member flags remain
    in the retained candidate table and are reached through member candidate
    indices.

    Performance
    -----------
    The serial control loop may compare a shrinking unassigned set for every
    event, giving O(N^2) representative-to-candidate predicate evaluations in
    the worst case.  Stable ordering costs O(N log N); repeated CuPy unique
    operations can add sorting cost beyond the quadratic predicate arithmetic
    in adversarial dense inputs.  Persistent and per-iteration GPU storage are
    O(N).  Seed/assignment decisions require O(N) scalar device synchronizations
    in the worst case, but no candidate-sized column is transferred to CPU.
    """

    config = ClusteringTolerances() if config is None else config
    if not isinstance(config, ClusteringTolerances):
        raise TypeError("config must be ClusteringTolerances")
    # Normalize and validate before allocating assignments.  This is GPU-only
    # for candidate columns, aside from aggregate validity synchronizations.
    candidates = GpuDecodedCandidates.from_decoder_result(decoded, geometry)
    n = len(candidates)
    order = _processing_order(candidates)
    # Allocate maximum-size bookkeeping once.  Event arrays are trimmed after
    # the number of representatives is known; -1 marks an unassigned candidate.
    assignment = cp.full(n, -1, dtype=cp.int64)
    representatives = cp.full(n, -1, dtype=cp.int64)
    counts = cp.zeros(n, dtype=cp.int32)
    nevents = 0

    for rank in range(n):
        # Python needs the next seed index and its current assignment as scalars.
        # These are intentional control-flow synchronizations; candidate
        # comparisons and compaction below remain vectorized on the GPU.
        seed = int(order[rank].item())
        if int(assignment[seed].item()) >= 0:
            continue
        # Rebuild the still-unassigned view in global priority order.  Because
        # compatibility requires a different tree, this view may include the
        # seed without causing it to appear in the compatible subset.
        unassigned = order[assignment[order] < 0]
        compatible = unassigned[compatible_with_representative(
            candidates, seed, unassigned, geometry, config
        )]
        if int(compatible.size):
            # Filtering preserves global S/N order, so each first occurrence
            # is the loudest compatible candidate from that tree.  cp.unique
            # identifies those first positions; the seed is added explicitly.
            _, first = cp.unique(candidates.tree[compatible], return_index=True)
            selected = cp.concatenate((
                order[rank:rank + 1], compatible[first]
            ))
        else:
            selected = order[rank:rank + 1]
        assignment[selected] = nevents
        representatives[nevents] = seed
        counts[nevents] = int(selected.size)
        nevents += 1

    # Every candidate must be consumed exactly once.  The aggregate check is a
    # defensive scalar synchronization, not a candidate-table host transfer.
    if n and not bool(cp.all(assignment >= 0).item()):
        raise RuntimeError("internal grouping error left a candidate unassigned")
    representatives = representatives[:nevents].copy()
    counts = counts[:nevents].copy()
    # Event semantics are representative-derived by construction.  In
    # particular, edge_flags is gathered from the seed; it is not ORed across
    # members.  Thus a representative's STARTUP_INCOMPLETE label propagates to
    # the event, while differing member labels remain in candidates.edge_flags.
    events = GpuEventTable(
        event_id=cp.arange(nevents, dtype=cp.int64),
        representative_candidate_index=representatives,
        member_count=counts,
        beam_id=candidates.beam_id[representatives],
        primary_tree_index=candidates.primary_tree_index[representatives],
        tree=candidates.tree[representatives],
        source_chunk_index=candidates.source_chunk_index[representatives],
        idm=candidates.idm[representatives],
        itime=candidates.itime[representatives],
        snr=candidates.snr[representatives],
        argmax_token=candidates.argmax_token[representatives],
        edge_flags=candidates.edge_flags[representatives],
        dm=candidates.dm[representatives],
        toa_sample_abs=candidates.toa_sample_abs[representatives],
        width_samp=candidates.width_samp[representatives],
        width_ms=candidates.width_ms[representatives],
        freq_lo_MHz=candidates.freq_lo_MHz[representatives],
        freq_hi_MHz=candidates.freq_hi_MHz[representatives],
    )

    # Group member links by event id.  Stable sorting preserves global
    # candidate priority within an event, and candidate_index keeps every
    # member's complete original row available without column duplication.
    member_order = order[cp.argsort(assignment[order], kind="stable")]
    member_event_id = assignment[member_order]
    members = GpuMemberTable(
        event_id=member_event_id,
        candidate_index=member_order,
        is_representative=(
            member_order == representatives[member_event_id]
        ),
    )
    return GpuClusteringResult(
        candidates=candidates,
        events=events,
        members=members,
        candidate_event_id=assignment,
        input_candidate_index=cp.arange(n, dtype=cp.int64),
        timed_out=False,
        complete=True,
    )


def cluster_candidates(
        decoded, geometry, *, config=None, timeout_ms=0,
        _test_timeout_after_work_tiles=None,
        _test_timeout_after_committed_events=None):
    """Build deterministic representative-seeded events in one GPU launch.

    Parameters
    ----------
    decoded
        A valid decoder batch or :class:`GpuDecodedCandidates` table.  All
        candidate columns must occupy one CUDA device.
    geometry : ClusteringGeometry
        Producer tree resolutions and complete-band time/DM geometry on that
        same device.
    config : ClusteringTolerances or None, keyword-only
        Association tolerances; ``None`` selects the defaults.
    timeout_ms : real, keyword-only
        Cooperative persistent-kernel deadline in milliseconds.  Zero (the
        default) disables the deadline.  Positive values are rounded up to one
        device-global-timer nanosecond.

    Returns
    -------
    GpuClusteringResult
        A complete result, or a clean partial result containing only fully
        committed events when the persistent association launch times out.
        ``timed_out``/``complete`` report which case occurred, and
        ``input_candidate_index`` maps retained candidates to this call's input.

    Notes
    -----
    One persistent block processes each independent
    ``(beam_id, primary_tree_index)`` partition.  Within a partition the first
    unassigned row in exact global priority order is the representative.  Every
    later unassigned row is compared directly with that representative, and an
    integer ``atomicMin`` keeps the earliest compatible row from each other
    tree.  This is deliberately representative-seeded and non-transitive.
    Compact GPU remapping then orders committed events by representative global
    rank, reproducing the serial scientific construction exactly when complete.

    The deadline is initialized from CUDA's device-wide ``%globaltimer`` by the
    first scheduled block and shared by the whole launch.  Blocks check before
    representatives and between winner/scan tiles of at most 256 entries.  A
    commit that has begun is allowed to finish so an event is never torn; the
    cooperative deadline can consequently overshoot by one predicate tile plus
    one parallel per-tree winner commit pass.  Kernel compilation, fixed GPU
    sorting/layout work before the launch, and compact result construction after
    it are outside this deadline.  The private work-tile keyword is solely a
    deterministic GPU cancellation hook for focused single-partition tests;
    its global counter is intentionally not a multi-partition scheduling API.
    """

    config = ClusteringTolerances() if config is None else config
    if not isinstance(config, ClusteringTolerances):
        raise TypeError("config must be ClusteringTolerances")
    timeout_ns = _timeout_nanoseconds(timeout_ms)
    test_tile_limit = _test_counter_limit(
        _test_timeout_after_work_tiles,
        "_test_timeout_after_work_tiles",
    )
    test_event_limit = _test_counter_limit(
        _test_timeout_after_committed_events,
        "_test_timeout_after_committed_events",
    )
    timeout_checks_enabled = (
        timeout_ns != 0 or test_tile_limit >= 0 or test_event_limit >= 0
    )
    _require_current_grouping_device(geometry)
    candidates = GpuDecodedCandidates.from_decoder_result(decoded, geometry)
    n = len(candidates)
    order = _processing_order(candidates)
    if not n:
        empty_i64 = cp.empty(0, dtype=cp.int64)
        return _construct_grouping_result(
            candidates,
            order,
            empty_i64,
            empty_i64,
            cp.empty(0, dtype=cp.int32),
            empty_i64,
            timed_out=False,
        )

    layout = _partition_layout(candidates, order)
    candidate_order = layout.candidate_order
    predicate_dtypes = _predicate_dtypes(candidates)

    # Only hot predicate columns are gathered into partition priority order.
    # Reporting columns remain in the normalized candidate table.
    ordered_tree = candidates.tree[candidate_order].astype(cp.int32, copy=False)
    ordered_dm = candidates.dm[candidate_order]
    ordered_toa = candidates.toa_sample_abs[candidate_order]
    ordered_dm_step = candidates.dm_step[candidate_order]
    ordered_time_step = candidates.time_step_samples[candidate_order]

    provisional_by_position = cp.full(n, -1, dtype=cp.int64)
    provisional_representative = cp.full(n, -1, dtype=cp.int64)
    provisional_count = cp.zeros(n, dtype=cp.int32)
    use_shared, shared_bytes = _shared_winner_policy(
        layout.maximum_trees_per_partition
    )
    global_winners = cp.empty(
        1 if use_shared else layout.distinct_partition_tree_count,
        dtype=cp.uint64,
    )
    launch_deadline_ns = cp.zeros(1, dtype=cp.uint64)
    completed_work_tiles = cp.zeros(1, dtype=cp.uint64)
    completed_events = cp.zeros(1, dtype=cp.uint64)
    timeout_flag = cp.zeros(1, dtype=cp.int32)

    _raw_grouping_kernel(predicate_dtypes)(
        (layout.partition_count,),
        (_THREADS_PER_BLOCK,),
        (
            layout.partition_offsets,
            layout.partition_tree_offsets,
            candidate_order,
            layout.tree_slot_by_position,
            ordered_tree,
            ordered_dm,
            ordered_toa,
            ordered_dm_step,
            ordered_time_step,
            np.float64(config.dm_tolerance_bins),
            np.float64(config.time_padding_bins),
            np.float64(geometry.residual_slope_lo_samples_per_dm),
            np.float64(geometry.residual_slope_hi_samples_per_dm),
            np.int32(use_shared),
            np.uint64(timeout_ns),
            np.int32(timeout_checks_enabled),
            np.int64(test_tile_limit),
            np.int64(test_event_limit),
            launch_deadline_ns,
            completed_work_tiles,
            completed_events,
            timeout_flag,
            global_winners,
            provisional_by_position,
            provisional_representative,
            provisional_count,
        ),
        shared_mem=shared_bytes,
    )

    # Synchronize one scalar launch status, never a candidate-sized column.
    raw_timeout = bool(timeout_flag.item())

    # A representative marker is the event commit record.  Compact only those
    # slots, then restore serial global event numbering by representative rank.
    valid_slots = cp.flatnonzero(provisional_representative >= 0).astype(
        cp.int64, copy=False
    )
    representative_unsorted = provisional_representative[valid_slots]
    rank_by_candidate = cp.empty(n, dtype=cp.int64)
    rank_by_candidate[order] = cp.arange(n, dtype=cp.int64)
    final_slot_order = cp.argsort(
        rank_by_candidate[representative_unsorted], kind="stable"
    )
    ordered_slots = valid_slots[final_slot_order]
    representatives = provisional_representative[ordered_slots]
    counts = provisional_count[ordered_slots]
    nevents = int(ordered_slots.size)

    # Map provisional slots to compact event IDs.  The +1 offset gives the -1
    # unassigned sentinel an explicit invalid lookup entry.
    provisional_to_final = cp.full(n + 1, -1, dtype=cp.int64)
    provisional_to_final[ordered_slots + 1] = cp.arange(
        nevents, dtype=cp.int64
    )
    original_assignment = cp.empty(n, dtype=cp.int64)
    original_assignment[candidate_order] = provisional_to_final[
        provisional_by_position + 1
    ]
    all_assigned = bool(cp.all(original_assignment >= 0).item())
    if not raw_timeout and not all_assigned:
        raise RuntimeError(
            "internal GPU representative grouping left a candidate unassigned"
        )
    # The device flag is authoritative.  An indivisible final commit may cross
    # the deadline and assign every remaining row; that is still a timed-out
    # launch whose policy/provenance must not be silently reported complete.
    timed_out = raw_timeout

    if not timed_out:
        return _construct_grouping_result(
            candidates,
            order,
            original_assignment,
            representatives,
            counts,
            cp.arange(n, dtype=cp.int64),
            timed_out=False,
        )

    # Discard every row not owned by a committed event.  Each block contributes
    # an event-boundary prefix of its own partition; scheduling means their
    # union need not be a prefix of global representative order.  Retained
    # candidates stay in original relative input order, while all externally
    # visible candidate indices are rebased into this compact table.
    input_candidate_index = cp.flatnonzero(
        original_assignment >= 0
    ).astype(cp.int64, copy=False)
    local_by_input = cp.full(n, -1, dtype=cp.int64)
    local_by_input[input_candidate_index] = cp.arange(
        int(input_candidate_index.size), dtype=cp.int64
    )
    retained_candidates = _take_candidates(
        candidates, input_candidate_index
    )
    retained_assignment = original_assignment[input_candidate_index]
    retained_representatives = local_by_input[representatives]
    committed_priority = order[original_assignment[order] >= 0]
    retained_order = local_by_input[committed_priority]
    return _construct_grouping_result(
        retained_candidates,
        retained_order,
        retained_assignment,
        retained_representatives,
        counts,
        input_candidate_index,
        timed_out=True,
    )


__all__ = [
    "GpuDecodedCandidates",
    "GpuEventTable",
    "GpuClusteringResult",
    "GpuMemberTable",
    "ClusteringTolerances",
    "ClusteringGeometry",
    "compatible_with_representative",
    "cluster_candidates",
]
