"""
Unit test for AssembledFrame ASDF file I/O.

Constructs a frame via an AssembledFrameAllocator (so its metadata is naturally
non-null), writes it to an ASDF file in /dev/shm, reads it back, and verifies
both the per-frame fields and the projected XEngineMetadata.

Note: ASDF is a *projected* serialization for XEngineMetadata. A saved ASDF
file describes a single (beam, time-chunk), so on read:

  - metadata.beam_ids                 -> length-1, equal to [frame.beam_id]
  - metadata.beam_positions_{x,y}     -> length-1, equal to the saved beam's positions
  - metadata.freq_channels            -> empty

All other XEngineMetadata fields round-trip bit-exactly.
"""

import os
import secrets

import numpy as np

from ..core import (
    AssembledFrame,
    AssembledFrameAllocator,
    SlabAllocator,
    XEngineMetadata,
)


def _assert_metadata_unchanged(md1, md2):
    """Assert the ~17 round-trip-stable fields match exactly."""
    assert md2.version == md1.version
    assert list(md2.zone_nfreq) == list(md1.zone_nfreq)
    assert list(md2.zone_freq_edges) == list(md1.zone_freq_edges)
    assert md2.beamset == md1.beamset
    assert md2.unix_ns_at_seq_0 == md1.unix_ns_at_seq_0
    assert md2.dt_ns_per_seq == md1.dt_ns_per_seq
    assert md2.seq_per_frb_time_sample == md1.seq_per_frb_time_sample
    assert md2.tel_origin_itrs_lat_deg == md1.tel_origin_itrs_lat_deg
    assert md2.tel_origin_itrs_lon_deg == md1.tel_origin_itrs_lon_deg
    assert list(md2.tel_grid_x_axis) == list(md1.tel_grid_x_axis)
    assert list(md2.tel_grid_y_axis) == list(md1.tel_grid_y_axis)
    assert list(md2.tel_dish_elev_axis) == list(md1.tel_dish_elev_axis)
    assert list(md2.tel_dish_vert_axis) == list(md1.tel_dish_vert_axis)
    assert md2.tel_dish_coelev_deg == md1.tel_dish_coelev_deg
    assert md2.tel_dish_separation_x_m == md1.tel_dish_separation_x_m
    assert md2.tel_dish_separation_y_m == md1.tel_dish_separation_y_m
    assert list(md2.noise_variance) == list(md1.noise_variance)


def test_assembled_frame_asdf():
    """Round-trip an AssembledFrame through ASDF and check per-frame projection."""
    print("  test_assembled_frame_asdf()...")

    # Build a multi-beam XEngineMetadata via make_random (fuzz-style), but
    # overwrite beam_ids and beam_positions_{x,y} to a small known length so
    # the test can index deterministically. We could read them back instead,
    # but rewriting also exercises the def_readwrite path under the new
    # shared_ptr holder. Then re-validate.
    md = XEngineMetadata.make_random()
    md.beam_ids = [11, 22, 33, 44]
    md.beam_positions_x = [0.10, -0.05, 0.07, -0.12]
    md.beam_positions_y = [-0.03, 0.08, 0.00, 0.15]
    # freq_channels left untouched (may be empty from make_random).
    md.validate()

    nbeams = len(md.beam_ids)
    nfreq = md.get_total_nfreq()
    ntime = 64  # small for a fast test

    # Use slab capacity large enough for at least one frame.
    slab_capacity = nbeams * nfreq * (ntime // 2) * 2
    slab = SlabAllocator("af_rhost", slab_capacity)
    alloc = AssembledFrameAllocator(slab, num_consumers=1, time_samples_per_chunk=ntime)
    alloc.initialize_metadata(md)
    alloc.initialize_initial_chunk(0)

    frame = alloc.get_frame(0)

    # Sanity checks on the frame we built.
    assert frame.nfreq == nfreq
    assert frame.ntime == ntime
    assert frame.beam_id in md.beam_ids
    idx = list(md.beam_ids).index(frame.beam_id)

    # Mutate the data so the round-trip check isn't trivial (frames come from
    # the allocator pre-filled with 0x88; rewrite with random bytes).
    data = np.asarray(frame.data)
    data[:] = np.random.randint(0, 256, size=data.shape, dtype=np.uint8)
    data_copy = data.copy()

    # Generate temp filename in /dev/shm so the test doesn't touch persistent storage.
    filename = f"/dev/shm/test_assembled_frame_asdf_{secrets.token_hex(8)}.asdf"
    print(f"    filename={filename}")

    try:
        frame.write_asdf(filename)
        frame2 = AssembledFrame.from_asdf(filename)
    finally:
        try:
            os.unlink(filename)
        except FileNotFoundError:
            pass

    # ---- Per-frame fields ----
    assert frame2.nfreq == frame.nfreq
    assert frame2.ntime == frame.ntime
    assert frame2.beam_id == frame.beam_id
    assert frame2.time_chunk_index == frame.time_chunk_index

    # Data bytes match.
    data2 = np.asarray(frame2.data)
    assert data2.shape == data_copy.shape
    assert data2.dtype == data_copy.dtype
    assert np.array_equal(data2, data_copy), "ASDF round-trip changed data bytes"

    # ---- XEngineMetadata projection ----
    md2 = frame2.metadata
    assert md2 is not None, "frame2.metadata must be non-null"

    # The four "special" members are projected to per-frame scalars/empties.
    assert list(md2.beam_ids) == [frame.beam_id]
    assert list(md2.beam_positions_x) == [md.beam_positions_x[idx]]
    assert list(md2.beam_positions_y) == [md.beam_positions_y[idx]]
    assert list(md2.freq_channels) == []

    # Everything else survives intact.
    _assert_metadata_unchanged(md, md2)
