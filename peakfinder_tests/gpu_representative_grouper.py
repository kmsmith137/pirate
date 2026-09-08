"""Legacy benchmark snapshot of persistent-GPU PIRATE candidate grouping.

Production now contains the promoted persistent implementation plus cooperative
timeout/partial-result support.  This benchmark-side copy remains intentionally
independent only as a no-timeout parity and historical performance baseline; it
is never imported by production and application code must call
``pirate_frb.OfflineCandidateGrouper.group_candidates``.  Both implementations
run the representative loop inside one persistent CUDA block per independent
``(beam_id, primary_tree_index)`` partition.

Scientific semantics
--------------------
Within a partition, candidates are already arranged in the production global
priority order.  The first unassigned row is the next representative.  Threads
cooperatively compare all later, still-unassigned rows *directly with that
representative*.  An exact integer ``atomicMin`` retains the earliest priority
position for every compatible other tree.  Only those winners and the seed are
assigned.  Compatibility with a non-representative member is never evaluated,
so the algorithm is representative-seeded and intentionally non-transitive.

Partitions cannot interact under the production predicate.  Each block emits
provisional event slots from its own candidate-sized range.  After the kernel,
valid representatives are compacted and sorted on the GPU by their exact rank
in the production processing order.  This merge restores precisely the serial
global event numbering even when representatives from different partitions
interleave in S/N.

Memory and execution model
--------------------------
The hot predicate columns are gathered into structure-of-arrays in partition
priority order, giving coalesced reads during cooperative scans.  Canonical
decoder output uses float64 throughout.  If production is given an already
instantiated float32/mixed-precision candidate table, a dtype-specialized
kernel preserves the same CuPy operation-stage rounding instead of promoting
before subtraction.  Per-tree winner positions live in dynamic shared memory
when the largest partition's tree set fits a conservative device-derived
budget.  Otherwise a device-only global winner workspace with one entry per
distinct partition/tree pair is used; the total number of such pairs is at
most the candidate count.  Candidate tiles are not staged in shared memory
because one authoritative representative is processed at a time and therefore
a loaded tile has no cross-target reuse.

The kernel launch count is independent of candidates, representatives, and
events.  Pre/post-processing consists of a fixed collection of GPU sorts,
gathers, scans, compactions, and remaps.  A few aggregate host synchronizations
remain in production validation and in obtaining dynamic partition/event sizes
needed for launches and result shapes.  No candidate-sized column crosses to
the host.

Worst-case predicate work remains O(sum(partition_size**2)), with O(N)
persistent and temporary storage.  One block per partition can underutilize the
GPU when there are few partitions, creates load imbalance for unequal
partitions, and can leave a block running for a long time on a large all-isolated
partition.  These are explicit benchmark-snapshot limitations; production's
cooperative timeout bounds the long-running kernel case without changing the
scientific predicate.  No spatial bucketing or approximation is introduced.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import cupy as cp
import numpy as np

from pirate_frb.OfflineCandidateGrouper import (
    GpuDecodedCandidates,
    GpuEventTable,
    GpuGroupingResult,
    GpuMemberTable,
    GroupingConfig,
    _processing_order,
)


RAW_MODULE_COMPILE_OPTIONS = ("--std=c++11", "--fmad=false")
"""NVRTC options used by the persistent grouping kernel.

Fast math is intentionally absent.  ``--fmad=false`` prevents contraction of
the compatibility multiply/subtract expressions: production CuPy evaluates
those array operations in separate kernels, so each multiplication is rounded
to float64 before the following addition/subtraction.  Disabling contraction
preserves those exact boundary decisions on both Pascal and Ampere.
"""

RAW_MODULE_BACKEND = "nvrtc"
THREADS_PER_BLOCK = 256
SHARED_MEMORY_CANDIDATE_TILING = False
PROTOTYPE_IMPLEMENTATION_NAME = "persistent-partition representative grouper"


_CUDA_SOURCE_TEMPLATE = r"""
// Header-free for NVRTC installations without host standard include paths.
#define PIRATE_NO_WINNER 0xffffffffffffffffULL

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
        unsigned long long *global_winners,
        long long *provisional_event_by_position,
        long long *provisional_representative,
        int *provisional_member_count,
        int *partition_event_count)
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

    int local_event_id = 0;
    for (long long representative_position = begin;
         representative_position < end;
         ++representative_position) {
        // The previous event's assignments must be visible before every thread
        // makes the identical skip/seed decision for this priority position.
        __syncthreads();
        if (provisional_event_by_position[representative_position] >= 0) {
            continue;
        }

        const long long provisional_slot = begin + (long long)local_event_id;
        const int representative_tree = tree[representative_position];
        const PIRATE_DM_TYPE representative_dm = dm[representative_position];
        const PIRATE_TOA_TYPE representative_toa =
            toa_sample_abs[representative_position];
        const PIRATE_DM_STEP_TYPE representative_dm_step =
            dm_step[representative_position];
        const PIRATE_TIME_STEP_TYPE representative_time_step =
            time_step_samples[representative_position];

        if (tid == 0) {
            provisional_event_by_position[representative_position] =
                provisional_slot;
            provisional_representative[provisional_slot] =
                candidate_index[representative_position];
            provisional_member_count[provisional_slot] = 1;
        }
        for (long long local_tree = tid;
             local_tree < tree_count;
             local_tree += nthreads) {
            winners[local_tree] = PIRATE_NO_WINNER;
        }
        __syncthreads();

        // Candidate columns are laid out in partition priority order, so this
        // strided block scan performs coalesced structure-of-arrays reads.
        for (long long position = representative_position + 1 + tid;
             position < end;
             position += nthreads) {
            if (provisional_event_by_position[position] >= 0
                    || tree[position] == representative_tree) {
                continue;
            }

            // Each intermediate is explicitly rounded to the dtype selected
            // by the corresponding production CuPy ufunc.  In particular,
            // accepted float32 GpuDecodedCandidates must subtract in float32;
            // promoting their stored values before subtraction can flip an
            // inclusive boundary decision.
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
                    > (PIRATE_DM_COMPARE_TYPE)dm_limit) {
                continue;
            }

            const PIRATE_DM_TYPE edge_lo = (PIRATE_DM_TYPE)(
                delta_dm
                * (PIRATE_DM_TYPE)residual_slope_lo_samples_per_dm
            );
            const PIRATE_DM_TYPE edge_hi = (PIRATE_DM_TYPE)(
                delta_dm
                * (PIRATE_DM_TYPE)residual_slope_hi_samples_per_dm
            );
            // CuPy minimum/maximum propagate NaNs.  Finite validated inputs can
            // still overflow an intermediate (for example, inf * 0), so reject
            // that row exactly as the production comparisons do.
            if (edge_lo != edge_lo || edge_hi != edge_hi) {
                continue;
            }
            const PIRATE_TIME_STEP_TYPE other_time_step =
                time_step_samples[position];
            const PIRATE_TIME_STEP_TYPE maximum_time_step =
                other_time_step > representative_time_step
                ? other_time_step : representative_time_step;
            const PIRATE_TIME_STEP_TYPE padding = (PIRATE_TIME_STEP_TYPE)(
                (PIRATE_TIME_STEP_TYPE)time_padding_bins * maximum_time_step
            );
            const PIRATE_DM_TYPE minimum_edge =
                edge_lo < edge_hi ? edge_lo : edge_hi;
            const PIRATE_DM_TYPE maximum_edge =
                edge_lo > edge_hi ? edge_lo : edge_hi;
            const PIRATE_INTERVAL_TYPE interval_lo = (PIRATE_INTERVAL_TYPE)(
                (PIRATE_INTERVAL_TYPE)minimum_edge
                - (PIRATE_INTERVAL_TYPE)padding
            );
            const PIRATE_INTERVAL_TYPE interval_hi = (PIRATE_INTERVAL_TYPE)(
                (PIRATE_INTERVAL_TYPE)maximum_edge
                + (PIRATE_INTERVAL_TYPE)padding
            );
            if (interval_lo != interval_lo || interval_hi != interval_hi) {
                continue;
            }
            if ((PIRATE_TIME_COMPARE_TYPE)delta_toa
                        < (PIRATE_TIME_COMPARE_TYPE)interval_lo
                    || (PIRATE_TIME_COMPARE_TYPE)delta_toa
                        > (PIRATE_TIME_COMPARE_TYPE)interval_hi) {
                continue;
            }

            const long long local_tree_slot =
                candidate_tree_slot[position] - tree_begin;
            // Position is exact production priority within this partition.
            // Integer atomicMin is deterministic and never compares S/N floats.
            atomicMin(
                &winners[local_tree_slot],
                (unsigned long long)position
            );
        }
        __syncthreads();

        // One thread owns each distinct tree slot.  Each retained position is
        // unique, so assignments cannot race; the integer count has one exact
        // increment per winning other tree.
        for (long long local_tree = tid;
             local_tree < tree_count;
             local_tree += nthreads) {
            const unsigned long long winner = winners[local_tree];
            if (winner != PIRATE_NO_WINNER) {
                provisional_event_by_position[(long long)winner] =
                    provisional_slot;
                atomicAdd(&provisional_member_count[provisional_slot], 1);
            }
        }
        __syncthreads();
        ++local_event_id;
    }

    if (tid == 0) {
        partition_event_count[partition] = local_event_id;
    }
}
"""


_SUPPORTED_PREDICATE_DTYPES = frozenset((
    np.dtype(np.float32), np.dtype(np.float64),
))
_CANONICAL_PREDICATE_DTYPES = (
    np.dtype(np.float64),
    np.dtype(np.float64),
    np.dtype(np.float64),
    np.dtype(np.float64),
)


def _predicate_dtypes(candidates: GpuDecodedCandidates) -> tuple[np.dtype, ...]:
    """Return accepted arithmetic dtypes in DM/TOA/DM-step/time-step order.

    Decoder results normalize these columns to float64.  Production also
    accepts an already-instantiated candidate table unchanged, however, so the
    prototype additionally dispatches exact float32 and mixed float32/float64
    arithmetic rather than silently promoting before a pairwise subtraction.
    """

    result = tuple(np.dtype(column.dtype) for column in (
        candidates.dm,
        candidates.toa_sample_abs,
        candidates.dm_step,
        candidates.time_step_samples,
    ))
    unsupported = sorted({dtype.name for dtype in result
                          if dtype not in _SUPPORTED_PREDICATE_DTYPES})
    if unsupported:
        raise TypeError(
            "experimental GPU predicate columns must use float32 or float64; "
            "decoder-normalized inputs use float64 (unsupported: "
            + ", ".join(unsupported) + ")"
        )
    return result


def _render_cuda_source(predicate_dtypes: tuple[np.dtype, ...]) -> str:
    """Specialize pointer and intermediate types to CuPy ufunc promotion."""

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


# Canonical source remains available for structural inspection and eager
# benchmark compilation; noncanonical accepted precision variants are rendered
# lazily and cached by device plus dtype signature.
_CUDA_SOURCE = _render_cuda_source(_CANONICAL_PREDICATE_DTYPES)
_MODULE_BY_DEVICE: dict[tuple[int, tuple[str, ...]], cp.RawModule] = {}


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


def _raw_kernel(
        predicate_dtypes: tuple[np.dtype, ...] = _CANONICAL_PREDICATE_DTYPES,
        ) -> Any:
    """Return a device/dtype-local NVRTC kernel, compiling it on first use."""

    device_id = int(cp.cuda.Device().id)
    predicate_dtypes = tuple(np.dtype(dtype) for dtype in predicate_dtypes)
    cache_key = (device_id, tuple(dtype.str for dtype in predicate_dtypes))
    module = _MODULE_BY_DEVICE.get(cache_key)
    if module is None:
        module = cp.RawModule(
            code=_render_cuda_source(predicate_dtypes),
            options=RAW_MODULE_COMPILE_OPTIONS,
            backend=RAW_MODULE_BACKEND,
        )
        # Keep one module per CUDA context/device.  CuPy's disk cache can still
        # reuse NVRTC output between Python processes and compatible devices.
        _MODULE_BY_DEVICE[cache_key] = module
    return module.get_function("group_representative_partitions")


def compile_gpu_representative_kernels() -> None:
    """Compile/load the prototype kernel for the currently active CUDA device."""

    _raw_kernel()


def _stable_sort_from_least_to_most(indices: Any, keys: tuple[Any, ...]) -> Any:
    """Apply a fixed schema-key sequence without packing or dtype coercion."""

    result = indices
    for key in keys:  # Fixed schema loop; never candidate/event controlled.
        result = result[cp.argsort(key[result], kind="stable")]
    return result


def _partition_layout(candidates: GpuDecodedCandidates, order: Any) -> _PartitionLayout:
    """Build exact beam/family partitions and compact per-partition tree slots.

    All construction stays on the active GPU.  Stable sorting starts from the
    production priority order, so relative rank inside each partition remains
    exact.  A second fixed-key sort groups ``(partition, tree)`` pairs; its
    compact slot count is at most N and supplies either shared or global winner
    storage to the persistent kernel.
    """

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

    # Family is the less-significant partition key and beam the more-significant
    # one.  Stable sorts preserve global candidate priority inside equal keys.
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
    partition_by_position = (
        cp.cumsum(partition_boundary, dtype=cp.int64) - 1
    )
    partition_by_candidate = cp.empty(n, dtype=cp.int64)
    partition_by_candidate[candidate_order] = partition_by_position

    # Assign one compact slot to every distinct (partition, tree) pair.  Slots
    # for a partition are contiguous, allowing a block-local winner array.
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
    distinct_partition_tree_count = int(cp.count_nonzero(pair_boundary))
    unique_pair_partitions = pair_partition[pair_boundary]
    tree_counts = cp.bincount(
        unique_pair_partitions, minlength=partition_count
    ).astype(cp.int64, copy=False)
    partition_tree_offsets = cp.concatenate((
        cp.zeros(1, dtype=cp.int64),
        cp.cumsum(tree_counts, dtype=cp.int64),
    ))
    # One aggregate scalar controls shared-memory launch sizing.  It does not
    # expose candidate columns or introduce representative-dependent control.
    maximum_trees = int(cp.max(tree_counts).item())
    return _PartitionLayout(
        candidate_order=candidate_order,
        partition_offsets=partition_offsets,
        partition_tree_offsets=partition_tree_offsets,
        tree_slot_by_position=tree_slot_by_position,
        partition_count=partition_count,
        distinct_partition_tree_count=distinct_partition_tree_count,
        maximum_trees_per_partition=maximum_trees,
    )


def _shared_winner_policy(maximum_trees: int) -> tuple[bool, int]:
    """Select dynamic shared winners without assuming a GPU architecture.

    Half of the device-reported per-block shared-memory capacity is used as the
    budget, which avoids consuming the entire resource for one block.  Larger
    arbitrary tree sets use the compact O(N) global workspace instead.
    """

    if maximum_trees <= 0:
        return False, 0
    properties = cp.cuda.runtime.getDeviceProperties(int(cp.cuda.Device().id))
    shared_capacity = int(properties["sharedMemPerBlock"])
    required = maximum_trees * np.dtype(np.uint64).itemsize
    use_shared = required <= shared_capacity // 2
    return use_shared, required if use_shared else 0


def _construct_result(
        candidates: GpuDecodedCandidates,
        order: Any,
        assignment: Any,
        representatives: Any,
        counts: Any,
        ) -> GpuGroupingResult:
    """Construct production dataclasses and exact stable member ordering."""

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
    return GpuGroupingResult(
        candidates=candidates,
        events=events,
        members=members,
        candidate_event_id=assignment,
    )


def group_candidates_gpu_representative(
        decoded: Any,
        geometry: Any,
        *,
        config: GroupingConfig | None = None,
        ) -> GpuGroupingResult:
    """Group candidates with exact production semantics in a persistent kernel.

    The public scientific inputs and returned production dataclasses match
    :func:`pirate_frb.OfflineCandidateGrouper.group_candidates`.  Candidate
    normalization/validation and exact processing order are reused directly
    from production.  This legacy benchmark entry point has no timeout; normal
    callers must use production ``group_candidates``.

    The host performs no candidate-, representative-, or event-controlled loop.
    Dynamic partition and event counts incur a fixed number of aggregate CuPy
    synchronization/allocation boundaries.  Candidate-sized data never leaves
    the active CUDA device.
    """

    config = GroupingConfig() if config is None else config
    if not isinstance(config, GroupingConfig):
        raise TypeError("config must be GroupingConfig")
    candidates = GpuDecodedCandidates.from_decoder_result(decoded, geometry)
    n = len(candidates)
    order = _processing_order(candidates)
    if not n:
        return _construct_result(
            candidates,
            order,
            cp.empty(0, dtype=cp.int64),
            cp.empty(0, dtype=cp.int64),
            cp.empty(0, dtype=cp.int32),
        )

    layout = _partition_layout(candidates, order)
    candidate_order = layout.candidate_order
    predicate_dtypes = _predicate_dtypes(candidates)
    # Gather only the predicate columns.  This aligned SoA is the kernel's
    # coalesced scan layout; the full normalized candidate table is retained.
    # Preserve accepted float32 columns so subtraction and scalar arithmetic
    # round at the same stages as production CuPy ufuncs.
    ordered_tree = candidates.tree[candidate_order].astype(cp.int32, copy=False)
    ordered_dm = candidates.dm[candidate_order]
    ordered_toa = candidates.toa_sample_abs[candidate_order]
    ordered_dm_step = candidates.dm_step[candidate_order]
    ordered_time_step = candidates.time_step_samples[candidate_order]

    provisional_by_position = cp.full(n, -1, dtype=cp.int64)
    provisional_representative = cp.full(n, -1, dtype=cp.int64)
    provisional_count = cp.zeros(n, dtype=cp.int32)
    partition_event_count = cp.zeros(
        layout.partition_count, dtype=cp.int32
    )
    use_shared, shared_bytes = _shared_winner_policy(
        layout.maximum_trees_per_partition
    )
    global_winners = cp.empty(
        1 if use_shared else layout.distinct_partition_tree_count,
        dtype=cp.uint64,
    )

    _raw_kernel(predicate_dtypes)(
        (layout.partition_count,),
        (THREADS_PER_BLOCK,),
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
            global_winners,
            provisional_by_position,
            provisional_representative,
            provisional_count,
            partition_event_count,
        ),
        shared_mem=shared_bytes,
    )

    # Compact provisional slots and restore serial global event numbering by
    # the exact production rank of each representative.
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

    # Offset the lookup table by one so the kernel's -1 sentinel maps to an
    # explicit invalid entry instead of relying on negative-index semantics.
    # This keeps validation to one aggregate scalar synchronization after the
    # complete remap.
    provisional_to_final = cp.full(n + 1, -1, dtype=cp.int64)
    provisional_to_final[ordered_slots + 1] = cp.arange(
        nevents, dtype=cp.int64
    )
    assignment = cp.empty(n, dtype=cp.int64)
    assignment[candidate_order] = provisional_to_final[
        provisional_by_position + 1
    ]
    # One defensive aggregate synchronization mirrors production's invariant
    # check and catches any kernel/remapping failure before table construction.
    if not bool(cp.all(assignment >= 0).item()):
        raise RuntimeError(
            "internal GPU representative grouping left a candidate unassigned"
        )
    return _construct_result(
        candidates, order, assignment, representatives, counts
    )


def gpu_partition_statistics(candidates: GpuDecodedCandidates) -> tuple[int, int]:
    """Return aggregate partition count/largest size without host row copies.

    This diagnostic helper is intended for benchmark reporting outside timed
    grouping calls.  It performs only two scalar transfers regardless of N.
    """

    order = _processing_order(candidates)
    layout = _partition_layout(candidates, order)
    if not layout.partition_count:
        return 0, 0
    lengths = layout.partition_offsets[1:] - layout.partition_offsets[:-1]
    return layout.partition_count, int(cp.max(lengths).item())


__all__ = [
    "PROTOTYPE_IMPLEMENTATION_NAME",
    "RAW_MODULE_BACKEND",
    "RAW_MODULE_COMPILE_OPTIONS",
    "SHARED_MEMORY_CANDIDATE_TILING",
    "THREADS_PER_BLOCK",
    "compile_gpu_representative_kernels",
    "gpu_partition_statistics",
    "group_candidates_gpu_representative",
]
