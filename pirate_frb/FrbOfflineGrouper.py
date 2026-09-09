"""Discover, validate, and stage offline dedispersion maps for GPU grouping.

The offline dedisperser writes one ASDF file per `(beam, source chunk)` and
one S/N/argmax array per dedispersion tree inside that file.  The arrays of a
tree have axis order `(coarse DM, coarse time)`; different trees generally
have different DM and time resolutions.  This module preserves that ragged
layout instead of padding or resampling scientifically distinct grids.

ASDF is the CPU-side input boundary of the candidate pipeline.  Construction
of :class:`FrbOfflineGrouper` validates all files and reconstructs the exact
producer plan from the serialized configuration and plan YAML.  A later call
to :meth:`FrbOfflineGrouper.load_beam_chunk` copies one bounded collection of
host arrays and performs the single large CPU-to-GPU transfer.  Peak finding,
argmax decoding, and grouping can then consume the returned CuPy arrays
without transferring candidate-sized columns back to the host.

The loader also retains the producer's real chunk labels and, when present,
the chunk at which production began.  Those two coordinates are needed to
decide whether a dedispersed cell has enough pre-history to be in steady
state; a filename's first available chunk is not necessarily producer time
zero.
"""

from __future__ import annotations

import operator
import os
import re
from dataclasses import dataclass

import numpy as np

from .ArgmaxMetadata import (
    read_argmax_metadata, validate_snr_map_format, validate_saved_time_sample_ms,
)


_SNRMAP_RE = re.compile(r"^frame_b(\d+)_t(\d+)_snrmap\.asdf$")


def _integer(value, name, *, dtype=np.int64):
    """Validate scalar metadata as an exactly representable integer.

    Parameters
    ----------
    value
        Object implementing Python's integer-index protocol.  Floating-point
        values and booleans are deliberately rejected rather than coerced;
        accepting either could silently change a beam or time coordinate.
    name : str
        Field name used in validation errors.
    dtype : numpy dtype, optional
        Signed or unsigned integer storage type whose numerical bounds the
        result must fit.  The default is `numpy.int64`.

    Returns
    -------
    int
        A host Python integer.  No GPU allocation or transfer occurs.

    Raises
    ------
    ValueError
        If `value` is not an integer or lies outside `dtype`'s range.
    """

    if isinstance(value, (bool, np.bool_)):
        raise ValueError(f"{name} must be an integer, not bool")
    try:
        result = operator.index(value)
    except TypeError as exc:
        raise ValueError(f"{name} must be an integer") from exc
    bounds = np.iinfo(dtype)
    if not bounds.min <= result <= bounds.max:
        raise ValueError(f"{name} does not fit {np.dtype(dtype).name}")
    return int(result)


def _require_text(value, name):
    """Return a nonempty producer-serialized text field unchanged.

    `config_yaml` and `plan_yaml` are authoritative scientific metadata:
    the consumer reconstructs the producer's tree geometry from their exact
    contents.  This helper therefore accepts only an existing nonempty
    :class:`str`; it does not stringify arbitrary objects or normalize YAML.

    Parameters
    ----------
    value
        Candidate text value read from an ASDF tree on the CPU.
    name : str
        Metadata key used in a possible :class:`ValueError`.

    Returns
    -------
    str
        The original string, with whitespace and serialization preserved.

    Raises
    ------
    ValueError
        If `value` is not a string or is the empty string.
    """

    if not isinstance(value, str) or not value:
        raise ValueError(f"{name} must be a nonempty string")
    return value


@dataclass(frozen=True)
class GpuMapChunk:
    """GPU-resident maps for one source chunk and a compatible beam batch.

    Attributes
    ----------
    snr_by_tree : tuple of cupy.ndarray
        One floating array per producer tree.  Tree `t` has shape
        `(nbeam, ndm_out[t], nt_out[t])` and axis order `(beam, coarse DM,
        coarse time)`.  Each array preserves the producer's floating dtype.
    argmax_by_tree : tuple of cupy.ndarray
        Arrays matching `snr_by_tree` exactly in shape and GPU residence,
        with dtype `uint32`.  Tokens encode the fine time, profile, and
        frequency-subband multiplet and extra DM selected by dedispersion.
    beam_ids : tuple of int
        Physical beam identifiers in the order used by axis zero of every
        array.  They are not assumed to be dense or zero based.
    source_chunk_index : int
        The producer's absolute chunk coordinate.  It is not the position of
        this chunk in a loader loop and must be retained when time coordinates
        are decoded.

    Notes
    -----
    The dataclass is frozen so its provenance cannot be rebound accidentally;
    the contained CuPy arrays themselves retain their normal mutable semantics.
    """

    snr_by_tree: tuple
    argmax_by_tree: tuple
    beam_ids: tuple[int, ...]
    source_chunk_index: int


@dataclass(frozen=True)
class BeamBatch:
    """A bounded set of beam streams with identical temporal provenance.

    Beams may be processed together only when they contain the same ordered
    `source_chunk_indices` and report the same `producer_start_chunk`.
    Those invariants let the streaming peak extractor share loop structure and
    seam-state lifetimes without confusing absolute time between beams.

    `producer_start_chunk` is `None` only when that provenance was absent
    from every file for the beam.  Missing provenance is intentionally
    different from a known start at chunk zero; the caller must either raise
    or explicitly assume steady state.  All fields are small CPU tuples or
    scalars and contain no map data.
    """

    beam_ids: tuple[int, ...]
    source_chunk_indices: tuple[int, ...]
    producer_start_chunk: int | None


class FrbOfflineGrouper:
    """Validated, read-only view of a version-3 offline S/N-map acquisition.

    Construction discovers files named
    `frame_b<BEAM>_t<CHUNK>_snrmap.asdf`, checks their ASDF schema and source
    coordinates, and reconstructs a :class:`DedispersionPlan` from the first
    file's embedded `config_yaml` and `plan_yaml`.  Every later file must
    contain exactly equal serialized plan text and maps whose shape and dtype agree
    with that reconstructed plan.  Source chunks must be contiguous within a
    beam, although different compatible beam groups may cover different chunk
    intervals.

    The object stores filenames and small plan/provenance metadata, not a
    persistent in-memory copy of the maps.  :meth:`load_beam_chunk` stages one
    requested beam/chunk slab on the CPU and uploads each tree independently;
    keeping a tuple of ragged arrays avoids an artificial common grid and its
    potentially large padding cost.

    Parameters
    ----------
    acqdir : path-like
        Directory containing producer ASDF maps.  The path is made absolute;
        files outside the strict naming convention are ignored.
    cuda_device_id : integer, optional
        CUDA device on which later map batches are allocated.  It is validated
        as `int32` metadata but no CUDA work is performed by construction.

    Attributes
    ----------
    plan
        Exact incomplete producer plan reconstructed from serialized YAML.
        It supplies the scientific DM/time geometry used downstream.
    dcores : tuple[int, ...]
        Actual producer kernel time granularities, separate from the plan.
    argmax_encoding : str
        Validated PIRATE 1.5 four-byte token layout identifier.
    beam_ids : tuple[int, ...]
        Sorted beam identifiers found in the directory.
    chunks_by_beam : dict[int, tuple[int, ...]]
        Sorted, contiguous absolute source-chunk labels for each beam.
    producer_start_by_beam : dict[int, int | None]
        Authoritative producer start for each beam, or `None` when missing.
    tree_shapes : tuple[tuple[int, int], ...]
        Per-tree `(coarse DM, coarse time)` shapes; deliberately ragged.
    snr_dtypes, argmax_dtypes : tuple[numpy.dtype, ...]
        Per-tree on-disk dtypes.  Argmax dtypes are validated as `uint32`.

    Raises
    ------
    ValueError
        If discovery, schema, provenance, plan, shape, dtype, or time-sample
        validation fails.  Caught schema and metadata errors include the ASDF
        path so corrupt inputs can be identified.

    Notes
    -----
    Validation necessarily visits every tree array, so its time cost scales
    with total stored map size under ASDF's lazy array model.  Loading remains
    explicit and bounded, and this class never rewrites its input files.
    """

    def __init__(self, acqdir, cuda_device_id=0):
        """Discover the acquisition and establish all loader invariants.

        Parameters
        ----------
        acqdir : path-like
            Directory searched non-recursively for version-3 S/N-map ASDF
            filenames.  Unrelated entries are ignored.
        cuda_device_id : integer, optional
            Device identifier retained for later calls to `load_beam_chunk()`.

        Raises
        ------
        ValueError
            If no maps are found or any filename, embedded source coordinate,
            producer-start value, YAML plan, tree shape, dtype, or acquisition
            time sample is inconsistent.

        Notes
        -----
        This is the metadata/validation phase described in the class docstring.
        It opens every matching input file read-only and assigns validated
        filenames, coverage, plan geometry, dtypes, and provenance to `self`.
        ASDF arrays are inspected on the CPU while their files are open, but no
        map is retained and no GPU allocation occurs.  A failure leaves no
        rewritten files or external state.
        """

        import asdf
        from .pirate_pybind11 import DedispersionConfig, DedispersionPlan

        self.acqdir = os.path.abspath(os.fspath(acqdir))
        self.cuda_device_id = _integer(
            cuda_device_id, "cuda_device_id", dtype=np.int32
        )
        if not os.path.isdir(self.acqdir):
            raise ValueError(f"acquisition directory does not exist: {self.acqdir}")

        # Discovery is intentionally filename driven.  The embedded `source`
        # record is checked below, so a renamed or misplaced map cannot silently
        # acquire a different beam/time identity.
        files = {}
        for entry in os.scandir(self.acqdir):
            match = _SNRMAP_RE.fullmatch(entry.name)
            if match is None or not entry.is_file():
                continue
            key = (int(match.group(1)), int(match.group(2)))
            if key in files:
                raise ValueError(
                    f"duplicate logical beam/chunk pair {key}: "
                    f"{os.path.basename(files[key])!r} and {entry.name!r}"
                )
            files[key] = entry.path
        if not files:
            raise ValueError(
                "no frame_b<BEAM>_t<CHUNK>_snrmap.asdf files in "
                f"{self.acqdir}"
            )

        # Establish deterministic beam ordering and require contiguous source
        # chunks independently for each beam.  A gap cannot be passed through
        # the peakfinder's temporal halo as if the missing samples existed.
        self.beam_ids = tuple(sorted({beam for beam, _ in files}))
        for beam_id in self.beam_ids:
            _integer(beam_id, "beam ID", dtype=np.int32)

        chunks_by_beam = {}
        for beam_id in self.beam_ids:
            chunks = tuple(sorted(chunk for beam, chunk in files if beam == beam_id))
            for chunk in chunks:
                _integer(chunk, "source chunk index")
            gaps = [
                (left, right)
                for left, right in zip(chunks, chunks[1:])
                if right != left + 1
            ]
            if gaps:
                raise ValueError(
                    f"beam {beam_id} has non-contiguous source chunks: {gaps}"
                )
            chunks_by_beam[beam_id] = chunks

        self._files = files
        self.chunks_by_beam = chunks_by_beam
        self.coverage = tuple(sorted(files))

        # The first file supplies the authoritative serialized plan.  All maps
        # are checked against the same reconstructed object, while exact YAML,
        # per-tree shape, and dtype equality prevents accidental mixing of two
        # producer runs that happen to share filenames.
        reference = None
        starts_by_beam = {beam: [] for beam in self.beam_ids}
        for beam_id, source_chunk_index in self.coverage:
            path = files[(beam_id, source_chunk_index)]
            try:
                with asdf.open(path) as af:
                    saved = af.tree
                    validate_snr_map_format(saved)

                    config_yaml = _require_text(
                        saved.get("config_yaml"), "config_yaml"
                    )
                    plan_yaml = _require_text(saved.get("plan_yaml"), "plan_yaml")
                    if reference is None:
                        config = DedispersionConfig.from_yaml_string(config_yaml)
                        plan = DedispersionPlan.from_yaml_string(config, plan_yaml)
                    else:
                        plan = self.plan
                    dcores = read_argmax_metadata(
                        saved, ntrees=int(plan.ntrees),
                        douts=tuple(int(t.nt_ds) // int(t.nt_out) for t in plan.trees),
                    )
                    validate_saved_time_sample_ms(saved, plan)

                    # Source coordinates are duplicated in filename and ASDF
                    # metadata by design; requiring agreement catches stale
                    # copies before any decoded absolute TOA can be trusted.
                    source = saved.get("source")
                    if not isinstance(source, dict):
                        raise ValueError("source metadata must be a mapping")
                    saved_beam = _integer(
                        source.get("beam_id"), "source beam ID", dtype=np.int32
                    )
                    saved_chunk = _integer(
                        source.get("time_chunk_index"), "source chunk index"
                    )
                    if (saved_beam, saved_chunk) != (
                        beam_id, source_chunk_index
                    ):
                        raise ValueError(
                            "filename beam/chunk disagrees with source metadata: "
                            f"{(beam_id, source_chunk_index)} != "
                            f"{(saved_beam, saved_chunk)}"
                        )

                    # Producer start may be absent even in a v3 map,
                    # but a known start cannot lie after the chunk being read.
                    # Its absence is preserved rather than interpreted as zero.
                    producer_start = saved.get(
                        "producer_start_time_chunk_index"
                    )
                    if producer_start is not None:
                        producer_start = _integer(
                            producer_start, "producer start chunk"
                        )
                        if producer_start > source_chunk_index:
                            raise ValueError(
                                "producer start chunk is after the source chunk"
                            )
                    starts_by_beam[beam_id].append(producer_start)

                    # Tree grids are intrinsically ragged: each plan tree has
                    # its own downsampling and output DM count.  Validate each
                    # native two-dimensional grid rather than stacking trees.
                    trees = saved.get("trees")
                    if not isinstance(trees, (list, tuple)):
                        raise ValueError("trees must be a sequence")
                    if len(trees) != int(plan.ntrees):
                        raise ValueError(
                            f"expected {plan.ntrees} trees, got {len(trees)}"
                        )

                    shapes = []
                    snr_dtypes = []
                    argmax_dtypes = []
                    for itree, (saved_tree, plan_tree) in enumerate(
                            zip(trees, plan.trees)):
                        if not isinstance(saved_tree, dict):
                            raise ValueError(f"tree {itree} must be a mapping")
                        if "snr" not in saved_tree or "argmax" not in saved_tree:
                            raise ValueError(
                                f"tree {itree} must contain snr and argmax"
                            )
                        # np.asarray materializes/inspects ASDF-backed host data
                        # while the file is open.  Scientific S/N may use any
                        # floating dtype, whereas the token ABI is exactly u32.
                        snr = np.asarray(saved_tree["snr"])
                        argmax = np.asarray(saved_tree["argmax"])
                        expected_shape = (
                            int(plan_tree.ndm_out), int(plan_tree.nt_out)
                        )
                        if snr.shape != expected_shape:
                            raise ValueError(
                                f"tree {itree} snr shape {snr.shape} "
                                f"!= {expected_shape}"
                            )
                        if argmax.shape != expected_shape:
                            raise ValueError(
                                f"tree {itree} argmax shape {argmax.shape} "
                                f"!= {expected_shape}"
                            )
                        if not np.issubdtype(snr.dtype, np.floating):
                            raise ValueError(
                                f"tree {itree} snr dtype must be floating"
                            )
                        if argmax.dtype != np.dtype(np.uint32):
                            raise ValueError(
                                f"tree {itree} argmax dtype must be uint32"
                            )
                        shapes.append(expected_shape)
                        snr_dtypes.append(snr.dtype)
                        argmax_dtypes.append(argmax.dtype)

                    # Dtype equality is part of acquisition compatibility: a
                    # later concatenation must not silently promote precision
                    # or reinterpret the argmax token representation.
                    metadata = (
                        config_yaml,
                        plan_yaml,
                        tuple(shapes),
                        tuple(snr_dtypes),
                        tuple(argmax_dtypes),
                        dcores,
                        saved["argmax_encoding"],
                    )
                    if reference is None:
                        reference = metadata
                        self.plan = plan
                    elif metadata != reference:
                        raise ValueError(
                            "producer config, plan, Dcores, encoding, tree shapes, or dtypes "
                            "differ from the first file"
                        )
            # Add the concrete pathname without swallowing validation detail.
            except (KeyError, TypeError, ValueError, RuntimeError) as exc:
                raise ValueError(f"{path}: {exc}") from exc

        # A beam's start provenance describes the whole producer stream and
        # therefore must not vary from chunk to chunk.  Cross-beam equality is
        # not required here; iter_beam_batches() handles compatibility later.
        starts = {}
        for beam_id, values in starts_by_beam.items():
            if any(value != values[0] for value in values[1:]):
                raise ValueError(
                    f"beam {beam_id} has inconsistent producer start chunks"
                )
            starts[beam_id] = values[0]

        # Publish the immutable scientific/layout facts used by downstream
        # geometry builders and bounded loaders.  No map arrays remain open.
        self.producer_start_by_beam = starts
        self.config_yaml = reference[0]
        self.plan_yaml = reference[1]
        self.tree_shapes = reference[2]
        self.snr_dtypes = reference[3]
        self.argmax_dtypes = reference[4]
        self.dcores = reference[5]
        self.argmax_encoding = reference[6]
        self.ntrees = int(self.plan.ntrees)
        self.nt_in = int(self.plan.nt_in)
        self.nfreq = int(self.plan.nfreq)
        self.time_sample_ms = float(self.plan.config.time_sample_ms)
        if not np.isfinite(self.time_sample_ms) or self.time_sample_ms <= 0.0:
            raise ValueError("producer plan has an invalid time_sample_ms")

    def iter_beam_batches(self, beam_batch_size=1):
        """Yield deterministic, temporally compatible beam batches.

        Parameters
        ----------
        beam_batch_size : integer, optional
            Maximum number of beams in a yielded batch.  It must be positive;
            smaller values reduce peak GPU memory approximately linearly.

        Yields
        ------
        BeamBatch
            A CPU metadata record whose beams have exactly the same ordered
            chunk coverage and producer-start value.  A compatibility group
            larger than `beam_batch_size` is split without changing either
            invariant.

        Notes
        -----
        This method is a lazy metadata iterator: it performs no ASDF reads and
        allocates no GPU arrays.  Grouping is linear in the number of beams,
        apart from the small dictionary bookkeeping.
        """

        size = _integer(beam_batch_size, "beam_batch_size")
        if size <= 0:
            raise ValueError("beam_batch_size must be positive")

        # Chunk coverage and startup origin together define the time coordinate
        # system seen by a stateful peak extractor.  Only beams with an exactly
        # equal key can safely advance through one shared processing loop.
        compatible = {}
        for beam_id in self.beam_ids:
            key = (
                self.chunks_by_beam[beam_id],
                self.producer_start_by_beam[beam_id],
            )
            compatible.setdefault(key, []).append(beam_id)

        # Beam IDs entered the dictionary in sorted order, so each bounded
        # slice has stable ordering and aligns reproducibly with GPU beam axis 0.
        for (chunks, producer_start), beams in compatible.items():
            for first in range(0, len(beams), size):
                yield BeamBatch(
                    beam_ids=tuple(beams[first:first + size]),
                    source_chunk_indices=chunks,
                    producer_start_chunk=producer_start,
                )

    def load_beam_chunk(self, beam_ids, source_chunk_index):
        """Copy one compatible beam/chunk slab from ASDF to one CUDA device.

        Parameters
        ----------
        beam_ids : iterable of integers
            Nonempty, unique beam IDs.  All requested beams must belong to one
            compatibility group (identical chunk coverage and producer start).
            Their order becomes axis zero of every returned array.
        source_chunk_index : integer
            Absolute producer chunk coordinate that must exist for every beam.

        Returns
        -------
        GpuMapChunk
            Tuple-backed ragged CuPy arrays.  For tree `t`, both maps have
            shape `(len(beam_ids), *tree_shapes[t])`; S/N retains its saved
            floating dtype and argmax retains exact `uint32` tokens.

        Raises
        ------
        ValueError
            If beam IDs are empty, repeated, unknown, incompatible, or lack the
            requested chunk.

        Notes
        -----
        This method is the intentional large CPU-to-GPU boundary.  Each
        ASDF-backed `(DM, time)` array is copied into preallocated NumPy
        storage before its file closes, then the complete per-tree beam slab is
        uploaded under `cuda_device_id`.  Peakfinding can consequently use
        map data after all ASDF handles are closed.  Runtime and memory scale as
        `O(nbeam * sum_t(ndm[t] * ntime[t]))`; during upload, one host and one
        device copy coexist.  Input files are opened read-only.
        """

        import asdf
        import cupy as cp

        # Convert external identifiers once on the CPU; duplicate beams would
        # make the meaning of the leading map axis and candidate provenance
        # ambiguous.
        beam_ids = tuple(
            _integer(beam, "beam ID", dtype=np.int32) for beam in beam_ids
        )
        if not beam_ids or len(set(beam_ids)) != len(beam_ids):
            raise ValueError("beam_ids must be a nonempty unique sequence")
        source_chunk_index = _integer(
            source_chunk_index, "source chunk index"
        )

        # Compatibility is checked again at this public boundary rather than
        # trusting callers to have used iter_beam_batches().
        stream_keys = {
            (
                self.chunks_by_beam.get(beam),
                self.producer_start_by_beam.get(beam),
            )
            for beam in beam_ids
        }
        if None in {key[0] for key in stream_keys} or len(stream_keys) != 1:
            raise ValueError("beam_ids are not one compatible stream batch")
        missing = [
            (beam, source_chunk_index)
            for beam in beam_ids
            if (beam, source_chunk_index) not in self._files
        ]
        if missing:
            raise ValueError(f"missing source maps: {missing}")

        # Allocate one host slab per native tree shape.  Trees are not padded to
        # a rectangle because their pixels represent different physical grids.
        host_snr = [
            np.empty((len(beam_ids),) + shape, dtype=dtype)
            for shape, dtype in zip(self.tree_shapes, self.snr_dtypes)
        ]
        host_argmax = [
            np.empty((len(beam_ids),) + shape, dtype=dtype)
            for shape, dtype in zip(self.tree_shapes, self.argmax_dtypes)
        ]
        # ASDF arrays can be lazy and tied to their file handle.  np.copyto()
        # gives the staging arrays independent ownership before each close.
        for ibeam, beam_id in enumerate(beam_ids):
            path = self._files[(beam_id, source_chunk_index)]
            with asdf.open(path) as af:
                for itree, saved_tree in enumerate(af.tree["trees"]):
                    np.copyto(host_snr[itree][ibeam], saved_tree["snr"])
                    np.copyto(host_argmax[itree][ibeam], saved_tree["argmax"])

        # This is the one bulk host/device crossing in the normal loader path.
        # Separate transfers retain each tree's native shape and saved dtype.
        with cp.cuda.Device(self.cuda_device_id):
            snr_by_tree = tuple(cp.asarray(array) for array in host_snr)
            argmax_by_tree = tuple(cp.asarray(array) for array in host_argmax)

        return GpuMapChunk(
            snr_by_tree=snr_by_tree,
            argmax_by_tree=argmax_by_tree,
            beam_ids=beam_ids,
            source_chunk_index=source_chunk_index,
        )


__all__ = ["BeamBatch", "FrbOfflineGrouper", "GpuMapChunk"]
