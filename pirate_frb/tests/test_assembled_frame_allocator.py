"""
Unit tests for AssembledFrameAllocator.

Tests covered:
  - Test 3: Frame set allocation and properties (metadata, data shape, initialization)
  - Test 4: Sequence ordering (set / beam cycling, time chunks)
  - Test 5: Multi-consumer scenarios (set/frame identity, independent progress)
  - Test 6 (partial): Set recycling

The allocator hands out one AssembledFrameSet (= nbeams frames for one time
chunk) per get_frame_set() call. Frame identity within a set is determined
by the allocator -- frames[i].beam_id == metadata.beam_ids[i].

Run via: python -m pirate_frb test --net
"""

import numpy as np
from ..core import AssembledFrameAllocator, SlabAllocator, XEngineMetadata


def make_slab_allocator(capacity=4*1024*1024, aflags='af_rhost'):
    """Helper to create a host-memory SlabAllocator."""
    return SlabAllocator(aflags, capacity)


def _test_metadata(nfreq, beam_ids):
    """Helper: construct a fully-valid XEngineMetadata for one-zone tests."""
    return XEngineMetadata.make_fiducial([nfreq], [400.0, 800.0], beam_ids, 1.0)


def test_frame_properties():
    """
    Test 3: Frame set + frame allocation and properties.

    Verifies:
      - Set metadata (nfreq, ntime, nbeams, time_chunk_index, len(frames))
      - Each frame's metadata (nfreq, ntime, beam_id, time_chunk_index)
      - Data array has correct shape (nfreq, ntime/2 as uint8)
      - Data is initialized to 0x88 (representing -8 in int4)
    """
    print("  test_frame_properties()...")

    nfreq = 128
    time_samples_per_chunk = 256
    beam_ids = [10, 20, 30]

    slab = make_slab_allocator()
    alloc = AssembledFrameAllocator(slab, num_consumers=1, time_samples_per_chunk=time_samples_per_chunk)
    alloc.initialize_metadata(_test_metadata(nfreq, beam_ids))
    alloc.initialize_initial_chunk(0)

    # Get first set and check set-level properties.
    fset = alloc.get_frame_set(0)
    assert fset.nfreq == nfreq, f"Expected nfreq={nfreq}, got {fset.nfreq}"
    assert fset.ntime == time_samples_per_chunk
    assert fset.nbeams == len(beam_ids)
    assert fset.time_chunk_index == 0
    assert len(fset.frames) == len(beam_ids)

    # Check each frame in the set.
    for ibeam, frame in enumerate(fset.frames):
        assert frame.nfreq == nfreq
        assert frame.ntime == time_samples_per_chunk
        assert frame.beam_id == beam_ids[ibeam], \
            f"frames[{ibeam}].beam_id: expected {beam_ids[ibeam]}, got {frame.beam_id}"
        assert frame.time_chunk_index == 0

        # Check data shape: exposed as uint8 with shape (nfreq, ntime/2).
        data = np.asarray(frame.data)
        expected_shape = (nfreq, time_samples_per_chunk // 2)
        assert data.shape == expected_shape
        assert data.dtype == np.uint8

        # Check data initialization: all bytes should be 0x88 (int4 value -8 packed twice).
        assert np.all(data == 0x88), \
            f"frames[{ibeam}]: expected all data bytes 0x88, got non-0x88"

    # Verify we can modify a frame's data (tests that it's writable).
    np.asarray(fset.frames[0].data)[0, 0] = 0x12
    assert np.asarray(fset.frames[0].data)[0, 0] == 0x12

    # get_frame(ibeam) accessor should match frames[ibeam].
    assert fset.get_frame(0) is fset.frames[0]
    assert fset.get_frame(2) is fset.frames[2]

    # validate() should not throw on a freshly-allocated set.
    fset.validate()

    print("    PASSED")


def test_sequence_ordering():
    """
    Test 4: Sequence ordering.

    Verifies that get_frame_set() returns one set per call with monotonically
    increasing time_chunk_index, and that each set contains frames in
    metadata.beam_ids order.
    """
    print("  test_sequence_ordering()...")

    nfreq = 64
    time_samples_per_chunk = 256
    beam_ids = [5, 15, 25]
    num_chunks = 4

    slab = make_slab_allocator()
    alloc = AssembledFrameAllocator(slab, num_consumers=1, time_samples_per_chunk=time_samples_per_chunk)
    alloc.initialize_metadata(_test_metadata(nfreq, beam_ids))
    alloc.initialize_initial_chunk(0)

    # Verify allocator state after initialization.
    assert alloc.nfreq == nfreq
    assert alloc.time_samples_per_chunk == time_samples_per_chunk
    assert list(alloc.beam_ids) == beam_ids

    # Walk through num_chunks sets and verify each.
    for chunk_idx in range(num_chunks):
        fset = alloc.get_frame_set(0)
        assert fset.time_chunk_index == chunk_idx, \
            f"Set {chunk_idx}: expected time_chunk_index={chunk_idx}, got {fset.time_chunk_index}"
        assert len(fset.frames) == len(beam_ids)

        for beam_idx, frame in enumerate(fset.frames):
            assert frame.beam_id == beam_ids[beam_idx], \
                f"Chunk {chunk_idx}, frame {beam_idx}: expected beam_id={beam_ids[beam_idx]}, got {frame.beam_id}"
            assert frame.time_chunk_index == chunk_idx

    print("    PASSED")


def test_single_beam_sequence():
    """
    Test 4 (edge case): Single beam sequence.

    Each set has exactly one frame; time_chunk_index increments by one per call.
    """
    print("  test_single_beam_sequence()...")

    nfreq = 32
    time_samples_per_chunk = 256
    beam_ids = [42]

    slab = make_slab_allocator()
    alloc = AssembledFrameAllocator(slab, num_consumers=1, time_samples_per_chunk=time_samples_per_chunk)
    alloc.initialize_metadata(_test_metadata(nfreq, beam_ids))
    alloc.initialize_initial_chunk(0)

    for chunk_idx in range(5):
        fset = alloc.get_frame_set(0)
        assert fset.time_chunk_index == chunk_idx
        assert len(fset.frames) == 1
        frame = fset.frames[0]
        assert frame.beam_id == 42
        assert frame.time_chunk_index == chunk_idx

    print("    PASSED")


def test_multi_consumer_frame_identity():
    """
    Test 5: Multi-consumer set+frame identity.

    Verifies that multiple consumers calling get_frame_set() at the same chunk
    index receive the exact same set object (and therefore the exact same
    frame objects inside the set).
    """
    print("  test_multi_consumer_frame_identity()...")

    nfreq = 64
    time_samples_per_chunk = 256
    beam_ids = [1, 2]
    num_consumers = 3

    slab = make_slab_allocator()
    alloc = AssembledFrameAllocator(slab, num_consumers=num_consumers, time_samples_per_chunk=time_samples_per_chunk)

    # Initialize the allocator (one initialize_metadata call is enough, but
    # the multi-call path is tested incidentally).
    for consumer_id in range(num_consumers):
        alloc.initialize_metadata(_test_metadata(nfreq, beam_ids))
    alloc.initialize_initial_chunk(0)

    # Get the first set from each consumer; all should be the same object.
    sets0 = [alloc.get_frame_set(consumer_id) for consumer_id in range(num_consumers)]
    for i in range(1, num_consumers):
        assert sets0[i] is sets0[0], \
            f"Consumer {i} got a different set object than consumer 0"
        # And, therefore, every inner frame is identical too.
        for ibeam in range(len(beam_ids)):
            assert sets0[i].frames[ibeam] is sets0[0].frames[ibeam], \
                f"Consumer {i} frame {ibeam} differs from consumer 0"

    # Verify modification is visible to all references.
    np.asarray(sets0[0].frames[0].data)[0, 0] = 0xAB
    for i in range(1, num_consumers):
        assert np.asarray(sets0[i].frames[0].data)[0, 0] == 0xAB

    # Get the second set and verify identity again.
    sets1 = [alloc.get_frame_set(consumer_id) for consumer_id in range(num_consumers)]
    for i in range(1, num_consumers):
        assert sets1[i] is sets1[0], \
            f"Consumer {i} got a different second-set object than consumer 0"

    # Second set should be a different object from the first.
    assert sets1[0] is not sets0[0], "Second set should be a different object from first"

    print("    PASSED")


def test_multi_consumer_independent_progress():
    """
    Test 5: Multi-consumer independent progress.

    Verifies that consumers can progress at different rates (in chunk units),
    each receiving the correct sequence of AssembledFrameSets.
    """
    print("  test_multi_consumer_independent_progress()...")

    nfreq = 32
    time_samples_per_chunk = 256
    beam_ids = [100, 200]
    num_consumers = 2

    slab = make_slab_allocator()
    alloc = AssembledFrameAllocator(slab, num_consumers=num_consumers, time_samples_per_chunk=time_samples_per_chunk)

    for consumer_id in range(num_consumers):
        alloc.initialize_metadata(_test_metadata(nfreq, beam_ids))
    alloc.initialize_initial_chunk(0)

    # Consumer 0 reads 2 chunks (= 2 sets).
    sets_c0 = [alloc.get_frame_set(0) for _ in range(2)]

    # Consumer 1 reads only 1 chunk so far.
    sets_c1 = [alloc.get_frame_set(1) for _ in range(1)]

    # Verify consumer 0's chunk indices and per-frame beam_ids.
    for chunk_idx, fset in enumerate(sets_c0):
        assert fset.time_chunk_index == chunk_idx
        for beam_idx in range(len(beam_ids)):
            assert fset.frames[beam_idx].beam_id == beam_ids[beam_idx]
            assert fset.frames[beam_idx].time_chunk_index == chunk_idx

    # Verify consumer 1's first set matches.
    assert sets_c1[0].time_chunk_index == 0
    assert sets_c1[0] is sets_c0[0], \
        "First set should be shared between consumers"

    # Consumer 1 catches up.
    sets_c1.append(alloc.get_frame_set(1))
    assert sets_c1[1] is sets_c0[1], \
        "Second set should be shared between consumers"

    print("    PASSED")


def test_frame_recycling():
    """
    Test 6 (partial): Set recycling.

    Verifies that frame sets are returned to the pool when all consumers have
    received and released them. Uses a small slab allocator to force recycling.

    With nbeams=1, one set = one slab; allocator capacity for N slabs means N
    sets resident at a time. We verify we can allocate many more than N sets
    in succession, proving that recycling is happening.

    With the worker thread, exact slab counts are non-deterministic (the worker
    can grab returned slabs to pre-create sets). So we test recycling by
    verifying the no-block / no-deadlock property over many iterations.
    """
    print("  test_frame_recycling()...")

    nfreq = 64
    time_samples_per_chunk = 256
    beam_ids = [1]  # Single beam: one slab per set.
    num_consumers = 2

    # Per-frame slab size; with nbeams=1, also the per-set slab footprint.
    # Each slab holds scales_offsets (nfreq, mpc, 2) float16 = nfreq*mpc*4 bytes
    # plus int4 data (nfreq, tspc) = nfreq*tspc/2 bytes.
    mpc = time_samples_per_chunk // 256
    slab_size = nfreq * mpc * 4 + (nfreq * time_samples_per_chunk) // 2

    # Create allocator with capacity for exactly 3 slabs (= 3 sets).
    capacity = slab_size * 3 + 1024  # +margin for alignment
    slab = make_slab_allocator(capacity=capacity)
    alloc = AssembledFrameAllocator(slab, num_consumers=num_consumers, time_samples_per_chunk=time_samples_per_chunk)

    for consumer_id in range(num_consumers):
        alloc.initialize_metadata(_test_metadata(nfreq, beam_ids))
    alloc.initialize_initial_chunk(0)

    # First set establishes slab size in the underlying allocator.
    set0_c0 = alloc.get_frame_set(0)
    initial_total = alloc.num_total_frames()
    assert initial_total == 3, f"Expected 3 total frames, got {initial_total}"

    # Consumer 1 gets set 0 - same set object, no new allocation.
    set0_c1 = alloc.get_frame_set(1)
    assert set0_c1 is set0_c0

    # Release references - set should be recycled.
    del set0_c0
    del set0_c1

    # Allocate again after recycling.
    set1_c0 = alloc.get_frame_set(0)
    assert set1_c0.time_chunk_index == 1
    set1_c1 = alloc.get_frame_set(1)
    assert set1_c1 is set1_c0
    del set1_c0
    del set1_c1

    # Prove recycling works by allocating many more sets than we have slabs.
    # With only 3 slabs but 20 sets, recycling must be happening.
    num_sets_to_allocate = 20
    for i in range(num_sets_to_allocate):
        s0 = alloc.get_frame_set(0)
        assert s0.time_chunk_index == 2 + i, \
            f"Expected chunk index {2+i}, got {s0.time_chunk_index}"
        s1 = alloc.get_frame_set(1)
        assert s1 is s0
        del s0
        del s1

    # If we got here without blocking/deadlock, recycling is working.
    print("    PASSED")


def test_frame_recycling_with_held_reference():
    """
    Test 6 (partial): Set recycling with held reference.

    Verifies that:
    - Sets are recycled when all consumers have received them AND no Python
      references remain.
    - If a consumer holds a reference, the underlying slab won't be freed
      even after the allocator drops its reference.
    """
    print("  test_frame_recycling_with_held_reference()...")

    nfreq = 64
    time_samples_per_chunk = 256
    beam_ids = [1]  # Single beam: one slab per set.
    num_consumers = 2

    # See test_frame_recycling for slab_size derivation.
    mpc = time_samples_per_chunk // 256
    slab_size = nfreq * mpc * 4 + (nfreq * time_samples_per_chunk) // 2
    capacity = slab_size * 4 + 1024  # 4 slabs
    slab = make_slab_allocator(capacity=capacity)
    alloc = AssembledFrameAllocator(slab, num_consumers=num_consumers, time_samples_per_chunk=time_samples_per_chunk)

    for consumer_id in range(num_consumers):
        alloc.initialize_metadata(_test_metadata(nfreq, beam_ids))
    alloc.initialize_initial_chunk(0)

    # Consumer 0 gets sets 0, 1, 2.
    set0 = alloc.get_frame_set(0)
    assert alloc.num_total_frames() == 4

    # Note: with the worker thread, exact free counts are non-deterministic
    # (worker may pre-allocate). We only check bounds.
    initial_free = alloc.num_free_frames()
    assert 0 <= initial_free <= 3, f"Free frames out of range: {initial_free}"

    set1 = alloc.get_frame_set(0)
    set2 = alloc.get_frame_set(0)

    # Consumer 1 gets set 0 - same object as set0.
    set0_c1 = alloc.get_frame_set(1)
    assert set0_c1 is set0

    # Both consumers have received set 0; allocator drops its reference.
    # But consumer 0 and consumer 1 both hold Python references, so the
    # underlying slabs aren't freed.

    # Release consumer 1's reference to set 0.
    del set0_c1
    # Consumer 0 still holds set0, so slab not recycled yet.

    # Release consumer 0's references to sets 1 and 2 (these are still
    # in the allocator's queue since consumer 1 hasn't received them).
    del set1
    del set2

    # Consumer 1 catches up. Consumer 1 is the last receiver, so the
    # allocator drops its reference. Consumer 0 already released its
    # reference, so sets 1 and 2 are recycled immediately.
    set1_c1 = alloc.get_frame_set(1)
    free_after_set1 = alloc.num_free_frames()

    set2_c1 = alloc.get_frame_set(1)
    free_after_set2 = alloc.num_free_frames()

    assert free_after_set2 >= free_after_set1, \
        f"Free frames should not decrease: {free_after_set1} -> {free_after_set2}"

    del set1_c1
    del set2_c1

    # Finally release set 0 -- it should now be recycled.
    del set0

    final_free = alloc.num_free_frames()
    assert final_free >= 1, \
        f"Expected at least 1 free after releasing all, got {final_free}"

    print("    PASSED")


def test_assembled_frame_allocator():
    """
    Run all AssembledFrameAllocator unit tests.

    Raises an exception if any test fails.
    """
    print("Testing AssembledFrameAllocator...")

    # Test 3: Frame allocation and properties
    test_frame_properties()

    # Test 4: Sequence ordering
    test_sequence_ordering()
    test_single_beam_sequence()

    # Test 5: Multi-consumer scenarios
    test_multi_consumer_frame_identity()
    test_multi_consumer_independent_progress()

    # Test 6 (partial): Set recycling
    test_frame_recycling()
    test_frame_recycling_with_held_reference()

    print("All AssembledFrameAllocator tests PASSED!")
