"""Test helpers for the one retained production full-band peak finder.

The executed notebooks and result directories beside this module are historical
benchmark artifacts.  They are deliberately not imported here.  This live test
surface exposes only the production full-band search.
"""

from dataclasses import asdict, dataclass, replace

import cupy as cp
import numpy as np

from pirate_frb.Peakfinders import (
    EdgeFlag,
    PeakFinderGeometry as _ProductionPeakFinderGeometry,
    extract_candidates,
)


METHODS = ("full_band_bowtie",)
BENCHMARK_METHODS = METHODS
METHOD_LABELS = {"full_band_bowtie": "Full-band bowtie"}


@dataclass(frozen=True)
class CandidateBatch:
    """Historical four-column result consumed by benchmark decoders."""

    idm: cp.ndarray
    itime: cp.ndarray
    snr: cp.ndarray
    argmax_token: cp.ndarray

    def __len__(self):
        return int(self.snr.size)

    def __getitem__(self, name):
        return getattr(self, name)


@dataclass(frozen=True)
class SubbandTarget:
    """Decoded producer sub-band metadata used only to inject simulations."""

    subband_id: str
    band_index: int
    level: int
    order_within_level: int
    fmin: int
    fmax: int
    freq_lo_MHz: float
    freq_hi_MHz: float
    is_full_band: bool

    def as_dict(self):
        return asdict(self)


@dataclass(frozen=True)
class PeakFinderGeometry:
    """Compatibility view over production full-band geometry."""

    production: _ProductionPeakFinderGeometry
    subbands: tuple
    full_band_index: int

    @property
    def itree(self):
        return self.production.tree

    @property
    def ndm(self):
        return self.production.ndm

    @property
    def nt(self):
        return self.production.ntime

    @property
    def dm_step(self):
        return self.production.dm_step

    @property
    def time_step_s(self):
        return self.production.time_step_s

    @property
    def reference_freq_mhz(self):
        return self.production.reference_freq_mhz

    @property
    def waist_bins(self):
        return self.production.waist_bins

    @property
    def full_band_footprint(self):
        return self.production.full_band_bowtie

    def diagnostics(self):
        footprint = self.production.full_band_bowtie
        true_cells = int(cp.count_nonzero(footprint).item())
        return {
            "tree": self.production.tree,
            "subbands": [band.as_dict() for band in self.subbands],
            "full_band_index": self.full_band_index,
            "full_band_bowtie": {
                "shape": [int(value) for value in footprint.shape],
                "true_cells": true_cells,
                "occupancy": true_cells / int(footprint.size),
            },
            "peakfinder": "full_band_bowtie",
        }


def _stable_subband_id(fmin, fmax):
    return f"f{int(fmin)}_{int(fmax)}"


def enumerate_plan_subbands(plan, itree):
    """Decode producer intervals for simulation setup, never peak selection."""
    tree = plan.trees[int(itree)]
    fs = tree.frequency_subbands
    nmultiplets, nbands = int(fs.M), int(fs.N)
    m_to_band = np.asarray(fs.m_to_n, dtype=np.int32)
    if m_to_band.shape != (nmultiplets,):
        raise RuntimeError(
            f"tree {itree}: m_to_n shape {m_to_band.shape}, expected ({nmultiplets},)"
        )

    multiplets = np.arange(nmultiplets, dtype=np.uint32)
    tokens = np.ascontiguousarray(multiplets << np.uint32(16))
    itrees = np.full(nmultiplets, int(itree), dtype=np.int64)
    zeros = np.zeros(nmultiplets, dtype=np.int64)
    fmins, fmaxs, tlos, this_, profiles = plan.decode_argmax_batch(
        tokens, itrees, zeros, zeros
    )
    freq_los, freq_his, _, _, _ = plan.decode_argmax2_batch(
        itrees, fmins, fmaxs, tlos, this_, profiles
    )
    fmins = np.asarray(fmins, dtype=np.int64)
    fmaxs = np.asarray(fmaxs, dtype=np.int64)
    freq_los = np.asarray(freq_los, dtype=np.float64)
    freq_his = np.asarray(freq_his, dtype=np.float64)

    decoder_span = int(fmaxs.max()) - int(fmins.min()) + 1
    hierarchy_span = 1 << int(fs.pf_rank)
    if decoder_span % hierarchy_span:
        raise RuntimeError(
            f"tree {itree}: decoder span {decoder_span} is not divisible by "
            f"FrequencySubbands span {hierarchy_span}"
        )
    frequency_scale = decoder_span // hierarchy_span
    raw, level_orders, seen_pairs = [], {}, set()
    for band_index in range(nbands):
        members = np.flatnonzero(m_to_band == band_index)
        if not members.size:
            raise RuntimeError(
                f"tree {itree}: decoded band {band_index} has no multiplets"
            )
        i0 = int(members[0])
        pair = (int(fmins[i0]), int(fmaxs[i0]))
        if pair in seen_pairs:
            raise RuntimeError(f"tree {itree}: duplicate decoded sub-band bounds {pair}")
        seen_pairs.add(pair)
        if not (np.all(fmins[members] == pair[0])
                and np.all(fmaxs[members] == pair[1])):
            raise RuntimeError(
                f"tree {itree}: multiplets for band {band_index} have inconsistent bounds"
            )
        coarse_lo = int(fs.n_to_flo[band_index])
        coarse_hi = int(fs.n_to_fhi[band_index])
        expected = (coarse_lo * frequency_scale, coarse_hi * frequency_scale - 1)
        if pair != expected:
            raise RuntimeError(
                f"tree {itree}: decoder bounds {pair} disagree with scaled "
                f"FrequencySubbands interval [{coarse_lo}, {coarse_hi})"
            )
        coarse_width = coarse_hi - coarse_lo
        if coarse_width <= 0 or coarse_width & (coarse_width - 1):
            raise RuntimeError(f"tree {itree}: sub-band width is not a power of two")
        level = coarse_width.bit_length() - 1
        order = level_orders.get(level, 0)
        level_orders[level] = order + 1
        if not (np.all(freq_los[members] == freq_los[i0])
                and np.all(freq_his[members] == freq_his[i0])):
            raise RuntimeError(f"tree {itree}: interval {pair} has inconsistent frequencies")
        raw.append({
            "subband_id": _stable_subband_id(*pair),
            "band_index": band_index,
            "level": level,
            "order_within_level": order,
            "fmin": pair[0],
            "fmax": pair[1],
            "freq_lo_MHz": float(freq_los[i0]),
            "freq_hi_MHz": float(freq_his[i0]),
        })

    global_pair = (
        min(entry["fmin"] for entry in raw),
        max(entry["fmax"] for entry in raw),
    )
    full_indices = [
        index for index, entry in enumerate(raw)
        if (entry["fmin"], entry["fmax"]) == global_pair
    ]
    if len(full_indices) != 1:
        raise RuntimeError(f"tree {itree}: expected exactly one full-band interval")
    full_index = full_indices[0]
    bands = tuple(
        SubbandTarget(**entry, is_full_band=(index == full_index))
        for index, entry in enumerate(raw)
    )
    return m_to_band, bands, full_index


def build_peakfinder_geometry(plan, itree, *, time_sample_s, nt_in,
                              reference_freq_mhz, dm_reach=8, waist_bins=1):
    """Build the exact production full-band geometry for benchmark use."""
    production = _ProductionPeakFinderGeometry.from_plan(
        plan, itree, dm_reach=dm_reach, waist_bins=waist_bins
    )
    expected_time_step = float(time_sample_s) * int(nt_in) / production.ntime
    if not np.isclose(expected_time_step, production.time_step_s):
        raise ValueError("benchmark and production time sampling disagree")
    if not np.isclose(float(reference_freq_mhz), production.reference_freq_mhz):
        raise ValueError("benchmark and production reference frequencies disagree")
    _, subbands, full_index = enumerate_plan_subbands(plan, itree)
    return PeakFinderGeometry(production, subbands, full_index)


def _legacy_candidate_batch(batch, snr_map):
    """Retain the benchmark package's historical coordinate/S/N dtypes."""
    if snr_map.ndim != 2:
        raise ValueError(
            f"legacy peakfinder compatibility requires a 2-D map, got {snr_map.shape}"
        )
    return CandidateBatch(
        batch.idm.astype(cp.int64, copy=False),
        batch.itime.astype(cp.int64, copy=False),
        snr_map[batch.idm, batch.itime],
        batch.argmax_token,
    )


def run_peakfinder(method, snr_map, argmax_map, geometry, threshold,
                   *, edge_policy="exclude"):
    """Run the sole production peakfinder through the historical 2-D API."""
    if method != "full_band_bowtie":
        raise ValueError(f"unknown method {method!r}; expected one of {METHODS}")
    if edge_policy not in ("exclude", "include"):
        raise ValueError("edge_policy must be 'exclude' or 'include'")
    if not isinstance(geometry, PeakFinderGeometry):
        raise TypeError("geometry must come from build_peakfinder_geometry")
    if snr_map.ndim != 2 or argmax_map.shape != snr_map.shape:
        raise ValueError("benchmark S/N and argmax maps must share a 2-D shape")
    production = replace(
        geometry.production,
        ndm=int(snr_map.shape[0]),
        ntime=int(snr_map.shape[1]),
        steady_state_it0=cp.zeros(int(snr_map.shape[0]), dtype=cp.int64),
    )
    found = extract_candidates(
        snr_map[None, :, :],
        argmax_map[None, :, :],
        production,
        threshold=threshold,
        beam_ids=(0,),
        source_chunk_index=0,
        assume_steady_state=True,
    )
    if edge_policy == "exclude":
        edge_bits = np.uint8(
            EdgeFlag.DM_LOW | EdgeFlag.DM_HIGH
            | EdgeFlag.ACQUISITION_LEFT | EdgeFlag.ACQUISITION_RIGHT
        )
        keep = (found.edge_flags & edge_bits) == 0
        found = type(found)(**{
            field: getattr(found, field)[keep]
            for field in (
                "beam_id", "source_chunk_index", "tree", "idm", "itime",
                "snr", "argmax_token", "edge_flags",
            )
        })
    return _legacy_candidate_batch(found, snr_map)


def candidate_coordinate_set(candidates):
    if not len(candidates):
        return set()
    coordinates = cp.stack((candidates.idm, candidates.itime), axis=1).get()
    return {tuple(map(int, pair)) for pair in coordinates}


def validate_candidate_set_containment(candidates_by_method, *, context="map"):
    """Validate the single-method benchmark schema.

    Kept under its historical name so checkpoint and campaign code does not
    need a broad rewrite.  There is no cross-method containment relation now.
    """
    methods = set(candidates_by_method)
    if methods != set(BENCHMARK_METHODS):
        raise ValueError(
            f"{context}: expected methods {list(BENCHMARK_METHODS)}, "
            f"got {sorted(methods)}"
        )
    return True

DECODED_DTYPES = {
    "idm": np.int64, "itime": np.int64, "time_chunk_index": np.int64,
    "tree": np.int64, "snr": np.float32, "argmax_token": np.uint32,
    "dm": np.float64, "toa_ref_s": np.float64, "width_s": np.float64,
    "profile": np.int64, "fmin": np.int64, "fmax": np.int64,
    "freq_lo_MHz": np.float64, "freq_hi_MHz": np.float64,
}


def empty_decoded_candidates():
    return {name: np.empty(0, dtype=dtype) for name, dtype in DECODED_DTYPES.items()}


def decode_candidates(plan, candidates, *, itree, time_chunk_index, ntime,
                      time_sample_s):
    """Decode survivors, retaining all sub-band/profile fields used by this study."""
    n = len(candidates)
    if not n:
        return empty_decoded_candidates()
    idm = np.ascontiguousarray(cp.asnumpy(candidates.idm), dtype=np.int64)
    itime = np.ascontiguousarray(cp.asnumpy(candidates.itime), dtype=np.int64)
    snr = np.asarray(cp.asnumpy(candidates.snr), dtype=np.float32)
    tokens = np.ascontiguousarray(cp.asnumpy(candidates.argmax_token), dtype=np.uint32)
    itrees = np.full(n, int(itree), dtype=np.int64)
    fmin, fmax, tlo, thi, profile = plan.decode_argmax_batch(
        tokens, itrees, idm, itime
    )
    freq_lo, freq_hi, dm, timestamp_samp, width_samp = plan.decode_argmax2_batch(
        itrees, fmin, fmax, tlo, thi, profile
    )
    chunk_start_s = int(time_chunk_index) * int(ntime) * float(time_sample_s)
    return {
        "idm": idm,
        "itime": itime,
        "time_chunk_index": np.full(n, int(time_chunk_index), dtype=np.int64),
        "tree": itrees,
        "snr": snr,
        "argmax_token": tokens,
        "dm": np.asarray(dm, dtype=np.float64),
        "toa_ref_s": chunk_start_s + np.asarray(timestamp_samp, dtype=np.float64) * time_sample_s,
        "width_s": np.asarray(width_samp, dtype=np.float64) * time_sample_s,
        "profile": np.asarray(profile, dtype=np.int64),
        "fmin": np.asarray(fmin, dtype=np.int64),
        "fmax": np.asarray(fmax, dtype=np.int64),
        "freq_lo_MHz": np.asarray(freq_lo, dtype=np.float64),
        "freq_hi_MHz": np.asarray(freq_hi, dtype=np.float64),
    }
