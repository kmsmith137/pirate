"""Find scientifically interpretable peaks in PIRATE S/N maps.

The dedispersion plan produces one map per tree with axis order ``(beam, DM,
time)``.  Trees generally have different native DM and time resolutions, so
this module never pads or resamples them onto a common rectangular grid.  Each
tree uses the full-band residual-dispersion Bowtie, symmetrically time-cropped
when required by the documented CHIME one-chunk processing horizon.

Source maps enter this module as CuPy arrays.  Filtering, Boolean selection,
exact compaction, flag construction, and deterministic ordering all remain on
the selected GPU.  The resulting one-dimensional candidate columns retain
chunk-local integer coordinates; the downstream decoder is responsible for
turning those coordinates and the argmax token into physical DM, absolute
low-frequency TOA, and width.

Startup provenance is deliberately a quality annotation, not a search mask.
When the producer start is known, :func:`make_startup_valid_mask` evaluates
the plan's conservative steady-state boundary.  Every finite, token-valid
cell still competes in one ordinary maximum-filter pass, and a selected cell
before that boundary receives :attr:`EdgeFlag.STARTUP_INCOMPLETE`.
"""

from __future__ import annotations

import operator
from dataclasses import dataclass, fields
from enum import IntFlag
from typing import Any

import cupy as cp
import numpy as np
from cupyx.scipy.ndimage import maximum_filter


K_DM = 4148.808  # seconds MHz^2 per pc cm^-3


class EdgeFlag(IntFlag):
    """Bitwise quality/provenance annotations stored in a ``uint8`` column.

    ``DM_LOW`` and ``DM_HIGH`` mean that part of the peakfinder footprint lay
    beyond the searched DM axis.  ``ACQUISITION_LEFT`` and
    ``ACQUISITION_RIGHT`` mean that temporal context was absent at a *real*
    acquisition boundary.  Ordinary chunk seams are not physical edges: the
    streaming extractor delays their centres until neighboring context is
    available.

    ``STARTUP_INCOMPLETE`` has a different meaning.  It says exactly that the
    candidate centre lies in a producer cell that is not guaranteed to be
    unaffected by pre-acquisition zero padding according to
    ``compute_steady_state_it0()``.  It does not necessarily mean that the
    burst centre at the top of the observing band predates the acquisition.
    The plan calculation is conservative because it uses the maximum internal
    delay of the DM bin and a worst-case peakfinder-width guard.  Measured S/N
    and width are reported unchanged, but either may be biased for a candidate
    carrying this bit.
    To put it in a simpler way: for an offline observation a burst might not fully reside
    whithin the acquisition as the burst arrived at the top of the band before
    the acquisition started. This means that the burst is not "fully complete" as there are
    missing data. The STARTUP_INCOMPLETE flag is just a warning to the user of this fact.

    """

    #: The competition footprint crosses the map's minimum-DM boundary.
    DM_LOW = 1 << 0
    #: The competition footprint crosses the map's maximum-DM boundary.
    DM_HIGH = 1 << 1
    #: Required left context lies before the physical acquisition start.
    ACQUISITION_LEFT = 1 << 2
    #: Required right context lies after the physical acquisition end.
    ACQUISITION_RIGHT = 1 << 3
    #: The producer does not guarantee this centre is startup-complete.
    STARTUP_INCOMPLETE = 1 << 4


_RAW_DTYPES = {
    "beam_id": np.dtype(np.int32),
    "source_chunk_index": np.dtype(np.int64),
    "tree": np.dtype(np.int32),
    "idm": np.dtype(np.int32),
    "itime": np.dtype(np.int32),
    "argmax_token": np.dtype(np.uint32),
    "edge_flags": np.dtype(np.uint8),
}


def _integer(value, name):
    """Validate and return an integral public coordinate.

    Parameters
    ----------
    value
        A Python or NumPy object implementing ``operator.index``.  Floating
        values and Booleans are rejected instead of being silently truncated.
    name : str
        Parameter name used in diagnostic messages.

    Returns
    -------
    int
        The exact host integer used for chunk, tree, or geometry indexing.
    """

    if isinstance(value, (bool, np.bool_)):
        raise TypeError(f"{name} must be an integer, not bool")
    try:
        return int(operator.index(value))
    except TypeError as exc:
        raise TypeError(f"{name} must be an integer") from exc


@dataclass(frozen=True)
class GpuRawCandidates:
    """Exact-sized raw candidate columns resident on one CUDA device.

    Every field has shape ``(ncandidate,)``.  Production dtypes are ``int32``
    for ``beam_id``, ``tree``, ``idm``, and ``itime``; ``int64`` for
    ``source_chunk_index``; ``uint32`` for ``argmax_token``; and ``uint8`` for
    ``edge_flags``.  Production float16 S/N maps are promoted to float32 before
    selection, so ``snr`` is normally float32 (other supported floating input
    dtypes are retained).  ``idm`` and ``itime`` remain coordinates in the
    named tree and source chunk, not global or physical coordinates.

    The structure intentionally contains no Python row objects and performs no
    candidate-sized device-to-host transfer.  It is passed directly to the GPU
    argmax decoder.
    """

    beam_id: Any
    source_chunk_index: Any
    tree: Any
    idm: Any
    itime: Any
    snr: Any
    argmax_token: Any
    edge_flags: Any

    def __len__(self):
        """Return the candidate count from CuPy array metadata only."""

        return int(self.snr.size)

    @classmethod
    def empty(cls):
        """Allocate an empty production-schema batch on the current GPU.

        Returns
        -------
        GpuRawCandidates
            Zero-length columns with the exact integer dtypes in
            :data:`_RAW_DTYPES` and a float32 S/N column.
        """

        values = {
            name: cp.empty(0, dtype=dtype)
            for name, dtype in _RAW_DTYPES.items()
        }
        values["snr"] = cp.empty(0, dtype=cp.float32)
        return cls(**values)


@dataclass(frozen=True)
class PeakFinderGeometry:
    """Immutable peak-search geometry for one native producer tree.

    ``ndm`` and ``ntime`` are the tree map dimensions, while ``dm_step`` is in
    pc cm^-3 per row and ``time_step_s`` is seconds per coarse output
    column.  Frequencies are MHz.  TOA-related geometry is referenced to
    ``reference_freq_mhz``, which the active plan path sets to the lowest edge
    of the complete observing band.

    ``full_band_bowtie`` is a 2-D Boolean CuPy footprint in ``(DM offset, time
    offset)`` order, symmetrically horizon-cropped when the physical footprint
    is wider than one native chunk. ``time_radius`` is its effective radius and
    ``requested_time_radius`` records the uncropped physical value.
    ``profile_dt`` and ``steady_state_it0`` are small int64 CuPy lookup tables.
    The latter holds, for every coarse DM row, the first producer output-time
    index guaranteed steady by the plan. All scalar members are host metadata;
    all array members live on the selected GPU.
    """

    tree: int
    ndm: int
    ntime: int
    dm_step: float
    time_step_s: float
    freq_low_mhz: float
    freq_high_mhz: float
    reference_freq_mhz: float
    waist_bins: int
    full_band_bowtie: Any
    dm_radius: int
    time_radius: int
    requested_time_radius: int
    token_multiplets: int
    token_extra_dm: int
    token_profiles: int
    token_dout: int
    profile_dt: Any
    steady_state_it0: Any

    @classmethod
    def from_plan(cls, plan, tree, *, dcore, dm_reach=8, waist_bins=1):
        """Derive and upload the static search setup for one producer tree.

        Parameters
        ----------
        plan
            Reconstructed producer ``DedispersionPlan``.  Its per-tree native
            dimensions, DM interval, time downsampling, token layout, observing
            band, and steady-state table define the consumer geometry.
        tree : int
            Zero-based tree index.
        dcore : int
            Time granularity supplied by the producer for this tree.
            It is separate from the plan in PIRATE 1.5.
        dm_reach : int, optional
            Number of native DM rows searched on either side of a centre.
        waist_bins : int, optional
            Extra coarse-time tolerance at the Bowtie waist.

        Returns
        -------
        PeakFinderGeometry
            Immutable scalar metadata plus small CuPy arrays on the current
            device.

        Raises
        ------
        TypeError, ValueError
            If integral arguments, plan dimensions, physical sampling, token
            bounds, or the producer steady-state table are inconsistent.

        Notes
        -----
        Setup is linear in the number of profiles and DM rows and occurs once
        per tree.  No S/N or argmax map is transferred here.
        """

        tree = _integer(tree, "tree")
        dm_reach = _integer(dm_reach, "dm_reach")
        waist_bins = _integer(waist_bins, "waist_bins")
        if not 0 <= tree < int(plan.ntrees):
            raise ValueError("tree index is outside the producer plan")
        if dm_reach < 0 or waist_bins < 0:
            raise ValueError("Bowtie DM reach and waist must be non-negative")

        # Preserve the producer's ragged grid exactly.  A common rectangular
        # shape would silently change native DM/time competition scales.
        plan_tree = plan.trees[tree]
        ndm = int(plan_tree.ndm_out)
        ntime = int(plan_tree.nt_out)
        nt_in = int(plan.nt_in)
        if ndm <= 0 or ntime <= 0 or nt_in <= 0:
            raise ValueError("producer plan contains an empty map dimension")

        # Producer DM intervals are half-open grids: dividing the interval by
        # ndm gives the native row spacing used by peakfinding and grouping.
        dm_step = (
            float(plan_tree.dm_max) - float(plan_tree.dm_min)
        ) / float(ndm)
        time_step_s = (
            float(plan.config.time_sample_ms) * 1.0e-3
            * float(nt_in) / float(ntime)
        )
        frequency_edges = tuple(
            float(value) for value in plan.config.zone_freq_edges
        )
        if len(frequency_edges) < 2:
            raise ValueError("producer plan has no complete observing band")
        freq_low, freq_high = frequency_edges[0], frequency_edges[-1]
        reference = freq_low
        if (not np.isfinite(dm_step) or dm_step <= 0.0
                or not np.isfinite(time_step_s) or time_step_s <= 0.0
                or not 0.0 < reference <= freq_low < freq_high):
            raise ValueError("producer plan has invalid physical peak geometry")

        # The footprint is inexpensive to construct on CPU once; only its
        # compact Boolean mask crosses to the GPU below.
        footprint = build_full_band_bowtie(
            dm_reach=dm_reach,
            dm_step=dm_step,
            time_step_s=time_step_s,
            freq_low_mhz=freq_low,
            freq_high_mhz=freq_high,
            reference_freq_mhz=reference,
            waist_bins=waist_bins,
        )
        # A real-time owner must be final after its immediately following
        # source chunk.  Some low-resolution trees have a physical full-band
        # Bowtie wider than one native map chunk, which would otherwise delay
        # centres until i+2 (or later).  CHIME L1b applies this same temporal
        # horizon by symmetrically removing outer Bowtie columns until the
        # footprint fits one chunk.  The DM axis retains its requested extent,
        # but time competition outside this horizon is deliberately omitted;
        # an outer DM row can consequently become empty.  Retain the uncropped
        # radius explicitly so output provenance exposes this approximation.
        requested_time_radius = footprint.shape[1] // 2
        while footprint.shape[1] > ntime:
            footprint = footprint[:, 1:-1]

        # The low token byte stores a fine-time position inside one coarse map
        # column.  dout and each profile's legal alignment are needed for cheap
        # pre-validation before a token may suppress a neighboring peak.
        nt_ds = int(plan_tree.nt_ds)
        if nt_ds <= 0 or nt_ds % ntime:
            raise ValueError("producer tree has inconsistent time dimensions")
        dout = nt_ds // ntime
        dcore = _integer(dcore, "dcore")
        nprofiles = int(plan_tree.nprofiles)
        nmultiplets = int(plan_tree.frequency_subbands.M)
        coarse_nfreq = 1 << int(plan_tree.frequency_subbands.pf_rank)
        nextra_dm, remainder = divmod(int(plan_tree.dm_downsampling), coarse_nfreq)
        if (not 0 < dout <= 256 or not 0 < dcore <= dout
                or dcore & (dcore - 1) or dout % dcore
                or not 0 < nprofiles <= 256
                or not 0 < nmultiplets <= 256
                or remainder or not 0 < nextra_dm <= 256
                or nextra_dm & (nextra_dm - 1)):
            raise ValueError("producer tree has invalid argmax token dimensions")
        profile_dt = []
        for profile in range(nprofiles):
            lpf = (profile - 1) // 3 if profile else 0
            profile_dt.append(min(dcore, 1 << lpf))

        # This plan method is the authoritative producer calculation.  Retain
        # its per-DM values even though they will annotate rather than veto
        # candidate centres.
        steady = np.asarray(plan.compute_steady_state_it0(tree))
        if steady.shape != (ndm,) or not np.issubdtype(steady.dtype, np.integer):
            raise ValueError("compute_steady_state_it0 returned an invalid table")
        return cls(
            tree=tree,
            ndm=ndm,
            ntime=ntime,
            dm_step=dm_step,
            time_step_s=time_step_s,
            freq_low_mhz=freq_low,
            freq_high_mhz=freq_high,
            reference_freq_mhz=reference,
            waist_bins=waist_bins,
            full_band_bowtie=cp.asarray(footprint),
            dm_radius=footprint.shape[0] // 2,
            time_radius=footprint.shape[1] // 2,
            requested_time_radius=requested_time_radius,
            token_multiplets=nmultiplets,
            token_extra_dm=nextra_dm,
            token_profiles=nprofiles,
            token_dout=dout,
            profile_dt=cp.asarray(profile_dt, dtype=cp.int64),
            steady_state_it0=cp.asarray(steady, dtype=cp.int64),
        )


def build_full_band_bowtie(
        *, dm_reach, dm_step, time_step_s, freq_low_mhz, freq_high_mhz,
        reference_freq_mhz, waist_bins):
    """Construct a full-band residual-dispersion footprint on the CPU.

    Parameters
    ----------
    dm_reach : int
        Native DM-bin radius on both sides of the candidate centre.
    dm_step : float
        Trial-DM spacing in pc cm^-3.
    time_step_s : float
        Duration of one coarse map-time pixel in seconds.
    freq_low_mhz, freq_high_mhz : float
        Complete observing-band edges in MHz.
    reference_freq_mhz : float
        Frequency to which map time is referred, in MHz.  The active pipeline
        uses ``freq_low_mhz``.
    waist_bins : int
        Non-negative discrete-time allowance at the central ridge.

    Returns
    -------
    numpy.ndarray
        A Boolean array with axis order ``(DM offset, time offset)`` and odd
        dimensions.  A true element means that offset participates when
        testing the centre with ``maximum_filter``.

    Notes
    -----
    For a competitor separated by ``delta_dm``, an imperfect trial DM leaves
    a cold-plasma residual

    ``delta_t(f) = K_DM * delta_dm * (f_ref^-2 - f^-2)``.

    Evaluating that relation at both band edges gives the two Bowtie edges;
    division by ``time_step_s`` expresses them in coarse pixels.  With the
    low band edge as reference, the low-frequency edge is zero and the
    high-frequency slope is positive.  Positive DM offsets therefore occupy
    non-negative time offsets and negative DM offsets the mirrored side under
    the coordinate convention used by the filter.  The asymmetric waist
    extension preserves the archived discrete ridge convention.

    Construction costs ``O(dm_reach * time_reach)`` CPU work and memory, but
    happens only once per tree before the mask is uploaded.
    """

    if dm_reach < 0 or waist_bins < 0:
        raise ValueError("Bowtie reach and waist must be non-negative")
    if not reference_freq_mhz <= freq_low_mhz < freq_high_mhz:
        raise ValueError("Bowtie frequencies are not ordered")
    # Work in float64 so footprint rounding is determined by the physical
    # formula, rather than the storage precision of the source S/N map.
    dm_offsets = (
        np.arange(-dm_reach, dm_reach + 1, dtype=np.float64) * dm_step
    )
    slope_low = K_DM * (
        reference_freq_mhz**-2 - freq_low_mhz**-2
    )
    slope_high = K_DM * (
        reference_freq_mhz**-2 - freq_high_mhz**-2
    )
    edge_a = dm_offsets * slope_low / time_step_s
    edge_b = dm_offsets * slope_high / time_step_s
    # min/max makes the construction valid for either sign of delta-DM.  The
    # footprint is expressed in competitor offsets about the centre pixel.
    lower = np.minimum(edge_a, edge_b)
    upper = np.maximum(edge_a, edge_b)
    upper[dm_offsets > 0.0] += waist_bins
    lower[dm_offsets < 0.0] -= waist_bins
    lower[dm_offsets == 0.0] = -waist_bins
    upper[dm_offsets == 0.0] = waist_bins
    # Round outward: omitting a fractional edge pixel could allow two samples
    # from the same physical residual ridge to survive as independent peaks.
    time_reach = int(np.ceil(max(-lower.min(), upper.max())))
    time_offsets = np.arange(-time_reach, time_reach + 1)
    footprint = (
        (time_offsets[None, :] >= lower[:, None])
        & (time_offsets[None, :] <= upper[:, None])
    )
    footprint[dm_reach, time_reach] = True
    return footprint


def make_startup_valid_mask(
        geometry, source_chunk_index, producer_start_chunk):
    """Evaluate the authoritative producer-start label for one source chunk.

    Parameters
    ----------
    geometry : PeakFinderGeometry
        Tree geometry whose int64 ``steady_state_it0`` table came directly
        from ``plan.compute_steady_state_it0(tree)``.
    source_chunk_index : int
        Absolute producer chunk label of the map being classified.
    producer_start_chunk : int
        Absolute chunk label at which this producer acquisition began.

    Returns
    -------
    cupy.ndarray
        Boolean GPU array of shape ``(ndm, ntime)`` in coarse tree-map
        coordinates.  True means that the cell is guaranteed unaffected by
        pre-acquisition zero padding; false means only that it is *not
        guaranteed* unaffected.

    Raises
    ------
    TypeError, ValueError
        If chunk labels are non-integral or the requested source precedes the
        known producer start.

    Notes
    -----
    ``compute_steady_state_it0`` counts coarse output samples from the real
    producer start, not from the filename's arbitrary chunk number.  The
    elapsed-chunk conversion below puts both coordinates on that same axis.
    The table is conservative: it uses the maximum internal delay in each DM
    bin and a worst-case peakfinder-width guard.  Consequently a false value
    must not be interpreted as proof that the top-of-band burst centre is
    before acquisition.  It causes a ``STARTUP_INCOMPLETE`` quality bit after
    peak selection; it never removes the cell from peak competition.  S/N and
    width can be biased where that bit is present.
    """

    source = _integer(source_chunk_index, "source_chunk_index")
    producer_start = _integer(producer_start_chunk, "producer_start_chunk")
    elapsed_chunks = source - producer_start
    if elapsed_chunks < 0:
        raise ValueError("source chunk precedes the producer start")
    # Each source file contributes exactly geometry.ntime coarse columns for
    # this tree; adding the local range gives absolute producer-output time.
    output_time = (
        cp.arange(geometry.ntime, dtype=cp.int64)[None, :]
        + elapsed_chunks * geometry.ntime
    )
    return output_time >= geometry.steady_state_it0[:, None]


def _token_validity(argmax_map, geometry):
    """Return a GPU mask for tokens safe to decode and use as competitors.

    ``argmax_map`` has shape ``(beam, ndm, ntime)`` and dtype ``uint32``.
    From low to high, the four bytes store fine time, profile, frequency
    multiplet, and extra DM. ``0xffffffff`` is the producer's invalid
    sentinel. Besides range-checking the four fields independently, a
    fine-time value must be aligned to the profile's integration stride from
    ``geometry.profile_dt``.

    The returned Boolean CuPy array has the input shape and remains on GPU.
    Pre-validating tokens is scientifically important: a corrupt high-S/N cell
    must neither become a candidate nor suppress a decodable neighbor in the
    maximum filter.  Full physical decoding remains the decoder's job.
    """

    token = argmax_map
    fine_time = token & cp.uint32(0xff)
    profile = (token >> cp.uint32(8)) & cp.uint32(0xff)
    multiplet = (token >> cp.uint32(16)) & cp.uint32(0xff)
    extra_dm = token >> cp.uint32(24)
    # Clamp only for safe lookup.  The explicit profile-range predicate below
    # still marks an out-of-range profile invalid.
    safe_profile = cp.minimum(
        profile.astype(cp.int64), geometry.token_profiles - 1
    )
    dt = geometry.profile_dt[safe_profile]
    return (
        (token != cp.uint32(0xffffffff))
        & (multiplet < geometry.token_multiplets)
        & (extra_dm < geometry.token_extra_dm)
        & (profile < geometry.token_profiles)
        & (fine_time < geometry.token_dout)
        & ((fine_time.astype(cp.int64) % dt) == 0)
    )


def _sort_candidates(candidates):
    """Return candidates in deterministic S/N-first order on the GPU.

    The primary key is descending measured S/N.  Equal-S/N rows are ordered by
    beam, source chunk, tree, DM coordinate, time coordinate, argmax token,
    then edge flags, all ascending.  Each operation permutes one-dimensional
    CuPy columns; candidate data never cross to the host.

    Repeated stable sorts are ``O(n log n)`` per key.  This is intentional for
    the modest compacted candidate count because the installed CuPy cannot
    lexsort mixed exact integer and floating dtypes without coercion.
    """

    if len(candidates) < 2:
        return candidates
    order = cp.arange(len(candidates), dtype=cp.int64)
    # Apply least-significant through most-significant keys.  Stability keeps
    # earlier tie decisions while preserving every key's native exact dtype.
    for key in (
        candidates.edge_flags,
        candidates.argmax_token,
        candidates.itime,
        candidates.idm,
        candidates.tree,
        candidates.source_chunk_index,
        candidates.beam_id,
        -candidates.snr,
    ):
        order = order[cp.argsort(key[order], kind="stable")]
    return GpuRawCandidates(**{
        field.name: getattr(candidates, field.name)[order]
        for field in fields(GpuRawCandidates)
    })


def concatenate_raw_candidates(parts):
    """Combine raw GPU batches and restore their global deterministic order.

    Parameters
    ----------
    parts : iterable of GpuRawCandidates
        Exact one-dimensional batches, normally one emitted piece per
        tree/chunk plus each tree's final flush.  Empty pieces are ignored.

    Returns
    -------
    GpuRawCandidates
        A single GPU-resident batch.  With no non-empty input, typed empty
        columns are allocated on the current device; a sole part is returned
        unchanged; otherwise columns are concatenated and stably sorted.

    Notes
    -----
    Work and temporary storage scale with the compacted candidate count, not
    the much larger S/N-map size.  No candidate value is inspected on CPU.
    """

    parts = tuple(part for part in parts if len(part))
    if not parts:
        return GpuRawCandidates.empty()
    if len(parts) == 1:
        return parts[0]
    combined = GpuRawCandidates(**{
        field.name: cp.concatenate([
            getattr(part, field.name) for part in parts
        ])
        for field in fields(GpuRawCandidates)
    })
    return _sort_candidates(combined)


def extract_candidates(
        snr_map, argmax_map, geometry, *, threshold, beam_ids,
        source_chunk_index, startup_valid_mask=None,
        assume_steady_state=False, physical_left_edge=True,
        physical_right_edge=True):
    """Select local maxima in one tree and return exact GPU columns.

    Parameters
    ----------
    snr_map : cupy.ndarray
        Floating S/N values with axis order ``(beam, coarse DM, coarse time)``.
        The time extent may be one native chunk or a halo-augmented streaming
        work array.  Float16 is promoted to float32 before comparison so the
        selected production S/N column is not left in half precision.
    argmax_map : cupy.ndarray
        ``uint32`` tokens with exactly the same shape and device residence as
        ``snr_map``.  Each selected token identifies the producer profile,
        frequency multiplet, and fine time needed by physical decoding.
    geometry : PeakFinderGeometry
        Geometry for this tree.  Its DM dimension must equal the map's; the
        streaming time dimension is allowed to exceed ``geometry.ntime``.
    threshold : float
        Inclusive finite S/N threshold.
    beam_ids : sequence of int
        Unique physical beam identifiers, one per leading map axis.  This tiny
        host vector is uploaded once for candidate provenance.
    source_chunk_index : int
        Chunk label written to selected rows.  The streaming wrapper passes a
        placeholder and later restores exact per-column ownership.
    startup_valid_mask : cupy.ndarray, optional
        Authoritative Boolean array of shape ``(ndm, ntime)`` from producer
        provenance.  It broadcasts across beams and labels selected false
        cells; it does not affect local-maximum competition.
    assume_steady_state : bool, optional
        Explicitly permit missing producer-start provenance.  In that case an
        all-true GPU mask is used and no startup-incomplete bit is invented.
    physical_left_edge, physical_right_edge : bool, optional
        Whether each work-array side is a true acquisition boundary.  False
        marks unresolved streaming context: centres within the temporal radius
        are withheld until more data arrive, rather than emitted as edge peaks.

    Returns
    -------
    GpuRawCandidates
        Exact-sized, deterministically sorted one-dimensional GPU columns.
        Coordinates are local to the supplied work map until a streaming
        owner projection restores chunk-local coordinates.

    Raises
    ------
    TypeError, ValueError, OverflowError
        For non-CuPy inputs, incompatible shapes/dtypes, invalid beam or
        startup metadata, a non-finite threshold, or a chunk label outside
        int64.

    Notes
    -----
    There is exactly one peak search.  Finite, pre-validated-token cells enter
    the same ``maximum_filter`` whether startup-complete or incomplete.  A
    strong incomplete cell can therefore suppress a weaker complete neighbor,
    preserving ordinary non-maximum-suppression semantics.  Only after
    coordinates are compacted is ``STARTUP_INCOMPLETE`` assigned on GPU.
    S/N, token, coordinates, and later decoded DM/TOA/width are not modified;
    the label warns that zero-padding can bias measured S/N and width.
    """

    if not isinstance(snr_map, cp.ndarray) or not isinstance(argmax_map, cp.ndarray):
        raise TypeError("snr_map and argmax_map must already be CuPy arrays")
    if snr_map.shape != argmax_map.shape or snr_map.ndim != 3:
        raise ValueError("S/N and argmax maps must share a 3-D shape")
    if not np.issubdtype(snr_map.dtype, np.floating):
        raise TypeError("S/N map must have a floating dtype")
    if argmax_map.dtype != cp.uint32:
        raise TypeError("argmax map must have dtype uint32")
    nbeam, ndm, ntime = snr_map.shape
    if ndm != geometry.ndm:
        raise ValueError("map DM dimension disagrees with peakfinder geometry")
    if not np.isfinite(threshold):
        raise ValueError("threshold must be finite")

    # Beam labels are orchestration metadata, not candidate data.  Validate
    # this tiny host sequence before its single upload and reject aliases that
    # would make downstream same-beam grouping ambiguous.
    host_beams = np.asarray(beam_ids)
    if (host_beams.shape != (nbeam,)
            or not np.issubdtype(host_beams.dtype, np.integer)
            or host_beams.dtype == np.bool_):
        raise ValueError("beam_ids must be one integer ID per map beam")
    if np.unique(host_beams).size != nbeam:
        raise ValueError("beam_ids must be unique")
    beam_ids_gpu = cp.asarray(host_beams, dtype=cp.int32)

    # Missing provenance is never silently classified as incomplete.  The
    # caller must either provide the authoritative plan mask or explicitly
    # assert steady state, in which case all cells remain unlabelled.
    if startup_valid_mask is None:
        if not assume_steady_state:
            raise ValueError(
                "startup provenance is missing; set assume_steady_state=True "
                "only when scientifically justified"
            )
        startup_valid = cp.ones((ndm, ntime), dtype=cp.bool_)
    else:
        if (not isinstance(startup_valid_mask, cp.ndarray)
                or startup_valid_mask.dtype != cp.bool_
                or startup_valid_mask.shape != (ndm, ntime)):
            raise ValueError("startup validity shape disagrees with the S/N map")
        startup_valid = startup_valid_mask

    # Half precision is adequate for stored producer maps but unnecessarily
    # fragile for threshold/filter comparison and final reported S/N.
    working_snr = (
        snr_map.astype(cp.float32) if snr_map.dtype == cp.float16 else snr_map
    )
    finite = cp.isfinite(working_snr)
    token_valid = _token_validity(argmax_map, geometry)
    # Startup status is intentionally absent here: complete and incomplete
    # cells must compete in this one shared non-maximum-suppression pass.
    valid_for_search = finite & token_valid
    competitor_map = cp.where(valid_for_search, working_snr, -cp.inf)
    # The singleton beam dimension prevents cross-beam competition.  Constant
    # -inf outside the map clips the footprint at physical DM/time bounds
    # without allowing an absent value to suppress an edge peak.
    local_max = maximum_filter(
        competitor_map,
        footprint=geometry.full_band_bowtie[None, :, :],
        mode="constant",
        cval=-cp.inf,
    )
    # Equality is inclusive by design.  Every cell on an exactly equal-valued
    # plateau survives; deterministic sorting orders them but does not invent a
    # plateau tie-breaker that could depend on traversal details.
    candidate_mask = (
        valid_for_search
        & (working_snr >= threshold)
        & (working_snr == local_max)
    )

    # A nonphysical side means context is pending, not absent.  Withhold its
    # unresolved centres so they can be reconsidered against neighboring chunk
    # data; only the final acquisition boundary deserves an edge annotation.
    if not physical_left_edge and geometry.time_radius:
        candidate_mask[:, :, :geometry.time_radius] = False
    if not physical_right_edge and geometry.time_radius:
        candidate_mask[:, :, -geometry.time_radius:] = False

    # cp.nonzero performs exact GPU compaction: storage scales with the number
    # of selected cells instead of reserving a fixed worst-case candidate
    # buffer.  No candidate coordinate is copied to CPU here.
    ibeam, idm, itime = cp.nonzero(candidate_mask)
    count = int(ibeam.size)
    flags = cp.zeros(count, dtype=cp.uint8)
    # Startup provenance is gathered only for selected coordinates.  This
    # candidate-sized operation and every subsequent bitwise assignment remain
    # on GPU and preserve the public uint8 flag dtype.
    flags |= (~startup_valid[idm, itime]).astype(cp.uint8) * np.uint8(
        EdgeFlag.STARTUP_INCOMPLETE
    )
    flags |= (idm < geometry.dm_radius).astype(cp.uint8) * np.uint8(
        EdgeFlag.DM_LOW
    )
    flags |= (
        idm >= ndm - geometry.dm_radius
    ).astype(cp.uint8) * np.uint8(EdgeFlag.DM_HIGH)
    candidate_time_radius = geometry.time_radius
    if physical_left_edge:
        flags |= (
            itime < candidate_time_radius
        ).astype(cp.uint8) * np.uint8(EdgeFlag.ACQUISITION_LEFT)
    if physical_right_edge:
        flags |= (
            itime >= ntime - candidate_time_radius
        ).astype(cp.uint8) * np.uint8(EdgeFlag.ACQUISITION_RIGHT)

    # Chunk provenance is exact int64.  Building columns directly on-device
    # avoids a host row table even when many peaks survive a plateau.
    source = _integer(source_chunk_index, "source_chunk_index")
    bounds = np.iinfo(np.int64)
    if not bounds.min <= source <= bounds.max:
        raise OverflowError("source_chunk_index does not fit int64")
    candidates = GpuRawCandidates(
        beam_id=beam_ids_gpu[ibeam],
        source_chunk_index=cp.full(count, source, dtype=cp.int64),
        tree=cp.full(count, geometry.tree, dtype=cp.int32),
        idm=idm.astype(cp.int32, copy=False),
        itime=itime.astype(cp.int32, copy=False),
        snr=working_snr[ibeam, idm, itime],
        argmax_token=argmax_map[ibeam, idm, itime],
        edge_flags=flags,
    )
    return _sort_candidates(candidates)


def _owned_centres(candidates, owner_chunk, owner_itime, start, stop):
    """Project selected work-array coordinates back to their source chunks.

    Parameters
    ----------
    candidates : GpuRawCandidates
        GPU rows whose ``itime`` currently indexes a concatenated streaming
        work array.
    owner_chunk : cupy.ndarray
        Int64 vector with one source chunk label per work-array time column.
    owner_itime : cupy.ndarray
        Int32 vector with the corresponding time coordinate local to that
        source chunk.
    start, stop : int
        Half-open work-array centre interval owned by this emission step.

    Returns
    -------
    GpuRawCandidates
        GPU rows in the interval, with ``source_chunk_index`` and ``itime``
        replaced by the owner vectors.  S/N, token, flags, and other
        coordinates are gathered unchanged.

    Notes
    -----
    This ownership projection prevents a halo copy from creating a duplicate
    candidate or leaking a concatenated coordinate into physical decoding.
    It performs only Boolean/index gathers on GPU.
    """

    working_itime = candidates.itime.astype(cp.int64, copy=False)
    keep = (working_itime >= start) & (working_itime < stop)
    position = working_itime[keep]
    return GpuRawCandidates(
        beam_id=candidates.beam_id[keep],
        source_chunk_index=owner_chunk[position],
        tree=candidates.tree[keep],
        idm=candidates.idm[keep],
        itime=owner_itime[position],
        snr=candidates.snr[keep],
        argmax_token=candidates.argmax_token[keep],
        edge_flags=candidates.edge_flags[keep],
    )


class StreamingPeakExtractor:
    """Stateful one-tree peak extraction across consecutive source chunks.

    A maximum-filter footprint of temporal radius ``h`` cannot decide the last
    ``h`` centres until the next chunk supplies right context.  To reconsider
    those delayed centres, the next work array must also retain their required
    left context: the earliest delayed centre is ``h`` columns before the seam
    and itself needs ``h`` older columns.  The persistent tail is therefore
    ``halo_size*h`` columns (or all data seen so far when shorter), where
    ``halo_size`` is at least two.

    S/N, argmax, startup-validity, and two ownership vectors persist on GPU
    between :meth:`process_chunk` calls.  Ownership maps concatenated time back
    to absolute source chunk plus chunk-local coarse ``itime``.  Internal seams
    are nonphysical and never receive acquisition-edge flags; emission is
    delayed there.  :meth:`flush` turns the last side into a physical edge only
    when the caller knows the acquisition is complete.
    """

    def __init__(self, geometry, *, threshold, beam_ids,
                 assume_steady_state=False, halo_size=2):
        """Initialize persistent state without allocating map-sized buffers.

        Parameters
        ----------
        geometry : PeakFinderGeometry
            Static setup for the single tree consumed by this instance.
        threshold : float
            Inclusive S/N threshold forwarded to every search.
        beam_ids : sequence of int
            Fixed beam-axis identity and order for all chunks in the stream.
        assume_steady_state : bool, optional
            Allow missing startup masks and leave all candidates unlabelled.
            False requires an authoritative mask on every processed chunk.
        halo_size : int, optional
            Number of tree-specific temporal radii retained between chunks.
            It must be at least two: delayed centres require one radius of
            future context and one radius of older left context.  Default 2.

        Notes
        -----
        Tail arrays are allocated lazily by the first call to
        :meth:`process_chunk`.  The object is single-use after :meth:`flush`
        and requires strictly consecutive source chunk labels.
        """
        self.geometry = geometry
        self.threshold = float(threshold)
        self.beam_ids = tuple(int(value) for value in beam_ids)
        self.assume_steady_state = bool(assume_steady_state)
        self.halo_size = _integer(halo_size, "halo_size")
        if self.halo_size < 2:
            raise ValueError(
                "halo_size must be at least 2 for complete seam context"
            )
        if self.geometry.time_radius > self.geometry.ntime:
            raise ValueError(
                "peakfinder time_radius must not exceed one source chunk; "
                "the one-chunk streaming horizon cannot be satisfied"
            )
        self._tail_snr = None
        self._tail_argmax = None
        self._tail_valid = None
        self._tail_chunk = None
        self._tail_itime = None
        self._last_chunk = None
        self._total_columns = 0
        self._next_emit = 0
        self._flushed = False

    def process_chunk(self, snr_map, argmax_map, source_chunk_index, *,
                      startup_valid_mask=None):
        """Consume one contiguous tree map and emit newly resolved centres.

        Parameters
        ----------
        snr_map, argmax_map : cupy.ndarray
            GPU arrays with shape ``(nbeam, geometry.ndm, geometry.ntime)``.
            S/N is floating; argmax tokens are uint32.  Beam order must match
            the IDs fixed at construction.
        source_chunk_index : int
            Absolute producer chunk label.  Labels must increase by exactly
            one because halo concatenation assumes no missing time interval.
        startup_valid_mask : cupy.ndarray, optional
            Authoritative Boolean GPU map of shape ``(ndm, ntime)`` for this
            source chunk.  It is required unless ``assume_steady_state`` was
            explicitly selected at construction.

        Returns
        -------
        GpuRawCandidates
            Candidate centres whose complete left/right context is now known,
            with the original source chunk and chunk-local time restored.  The
            final ``geometry.time_radius`` centres remain delayed.

        Raises
        ------
        RuntimeError
            If called after :meth:`flush`.
        ValueError
            For a chunk gap, map-shape mismatch, or missing/invalid startup
            provenance.

        Notes
        -----
        The method updates persistent GPU tails and ownership vectors as a side
        effect.  Map-sized data never move to CPU.  A false startup cell remains
        in the shared peak search and is annotated only if selected.
        """

        source = _integer(source_chunk_index, "source_chunk_index")
        if self._flushed:
            raise RuntimeError("cannot process a chunk after final flush")
        if self._last_chunk is not None and source != self._last_chunk + 1:
            raise ValueError("source chunks must be consecutive")
        expected = (len(self.beam_ids), self.geometry.ndm, self.geometry.ntime)
        if snr_map.shape != expected or argmax_map.shape != expected:
            raise ValueError(f"source map shape must be {expected}")

        if startup_valid_mask is None:
            if not self.assume_steady_state:
                raise ValueError(
                    "startup provenance is missing; explicit assumption required"
                )
            valid = cp.ones(
                (self.geometry.ndm, self.geometry.ntime), dtype=cp.bool_
            )
        else:
            if (not isinstance(startup_valid_mask, cp.ndarray)
                    or startup_valid_mask.dtype != cp.bool_
                    or startup_valid_mask.shape != (
                        self.geometry.ndm, self.geometry.ntime)):
                raise ValueError(
                    "startup_valid_mask must be a Boolean GPU tree map"
                )
            valid = startup_valid_mask

        # These small GPU owner vectors are the only reliable way to undo a
        # work array that may contain columns from several native chunks.
        local_itime = cp.arange(self.geometry.ntime, dtype=cp.int32)
        source_owner = cp.full(
            self.geometry.ntime, source, dtype=cp.int64
        )
        old_total = self._total_columns
        if self._tail_snr is None:
            working_snr = snr_map
            working_argmax = argmax_map
            working_valid = valid
            owner_chunk = source_owner
            owner_itime = local_itime
            working_start = old_total
        else:
            # Concatenation is the single map construction needed for seamless
            # filtering.  Startup labels and provenance follow exactly the
            # same time-axis layout as S/N and token data.
            working_snr = cp.concatenate((self._tail_snr, snr_map), axis=2)
            working_argmax = cp.concatenate(
                (self._tail_argmax, argmax_map), axis=2
            )
            working_valid = cp.concatenate((self._tail_valid, valid), axis=1)
            owner_chunk = cp.concatenate((self._tail_chunk, source_owner))
            owner_itime = cp.concatenate((self._tail_itime, local_itime))
            working_start = old_total - self._tail_snr.shape[2]

        # Work in a conceptual absolute coarse-column axis to decide ownership;
        # candidate rows themselves continue to store chunk-local coordinates.
        new_total = old_total + self.geometry.ntime
        emit_stop = max(
            self._next_emit, new_total - self.geometry.time_radius
        )
        centre_start = self._next_emit - working_start
        centre_stop = emit_stop - working_start
        # source_chunk_index=0 is deliberately temporary.  _owned_centres uses
        # the GPU owner vectors to assign the exact source provenance after
        # selecting only the interval newly owned by this call.
        found = extract_candidates(
            working_snr,
            working_argmax,
            self.geometry,
            threshold=self.threshold,
            beam_ids=self.beam_ids,
            source_chunk_index=0,
            startup_valid_mask=working_valid,
            physical_left_edge=(working_start == 0),
            physical_right_edge=False,
        )
        emitted = _owned_centres(
            found, owner_chunk, owner_itime, centre_start, centre_stop
        )

        # Delayed centres require a minimum 2*h tail.  Larger configured halos
        # retain additional context while preserving the same ownership and
        # emission rules.  Copies sever references to the potentially larger
        # concatenated work arrays and bound persistent GPU memory.
        keep = min(self.halo_size * self.geometry.time_radius, new_total)
        tail = slice(-keep, None) if keep else slice(0, 0)
        self._tail_snr = working_snr[:, :, tail].copy()
        self._tail_argmax = working_argmax[:, :, tail].copy()
        self._tail_valid = working_valid[:, tail].copy()
        self._tail_chunk = owner_chunk[tail].copy()
        self._tail_itime = owner_itime[tail].copy()
        self._last_chunk = source
        self._total_columns = new_total
        self._next_emit = emit_stop
        return emitted

    def flush(self):
        """Resolve delayed centres against the physical right acquisition edge.

        Returns
        -------
        GpuRawCandidates
            Every not-yet-emitted tail centre, with source ownership restored.
            Candidates whose full-band footprint is clipped by the true end
            may carry ``ACQUISITION_RIGHT``.  An empty typed batch is
            returned when no centres remain.

        Raises
        ------
        RuntimeError
            If the extractor has already been flushed.

        Notes
        -----
        Flushing is a terminal state transition.  Callers processing only a
        prefix must *not* flush: the prefix boundary is missing future context,
        not a physical end of acquisition.  Startup annotations stored in the
        tail remain authoritative and are propagated unchanged.
        """

        if self._flushed:
            raise RuntimeError("extractor has already been flushed")
        self._flushed = True
        if (self._tail_snr is None
                or self._next_emit >= self._total_columns):
            return GpuRawCandidates.empty()
        # The retained tail already contains the complete left context required
        # by delayed centres.  Mark only its right side physical for this final
        # search, then project its coordinates through the saved owner vectors.
        working_start = self._total_columns - self._tail_snr.shape[2]
        found = extract_candidates(
            self._tail_snr,
            self._tail_argmax,
            self.geometry,
            threshold=self.threshold,
            beam_ids=self.beam_ids,
            source_chunk_index=0,
            startup_valid_mask=self._tail_valid,
            physical_left_edge=(working_start == 0),
            physical_right_edge=True,
        )
        start = self._next_emit - working_start
        return _owned_centres(
            found,
            self._tail_chunk,
            self._tail_itime,
            start,
            self._total_columns - working_start,
        )


__all__ = [
    "EdgeFlag",
    "GpuRawCandidates",
    "K_DM",
    "StreamingPeakExtractor",
    "PeakFinderGeometry",
    "build_full_band_bowtie",
    "concatenate_raw_candidates",
    "extract_candidates",
    "make_startup_valid_mask",
]
