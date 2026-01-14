"""
Unit tests for AssembledFrameAllocator.

Tests covered:
  - Test 3: Frame allocation and properties (metadata, data shape, initialization)
  - Test 4: Sequence ordering (beam cycling, time chunks)
  - Test 5: Multi-consumer scenarios (frame identity, independent progress)
  - Test 6 (partial): Frame recycling

Run via: python -m pirate_frb test --net
"""

import numpy as np
from ..core import SlabAllocator, AssembledFrameAllocator


def make_slab_allocator(capacity=4*1024*1024, aflags='af_rhost'):
    """Helper to create a host-memory SlabAllocator."""
    return SlabAllocator(aflags, capacity)


def test_frame_properties():
    """
    Test 3: Frame allocation and properties.
    
    Verifies:
      - Frame metadata (nfreq, ntime, beam_id, time_chunk_index) match expected values
      - Data array has correct shape (nfreq, ntime/2 as uint8)
      - Data is initialized to 0x88 (representing -8 in int4)
    """
    print("  test_frame_properties()...")
    
    nfreq = 128
    time_samples_per_chunk = 256
    beam_ids = [10, 20, 30]
    
    slab = make_slab_allocator()
    alloc = AssembledFrameAllocator(slab, num_consumers=1)
    alloc.initialize(0, nfreq, time_samples_per_chunk, beam_ids)
    
    # Get first frame and check properties
    frame = alloc.get_frame(0)
    
    # Check metadata
    assert frame.nfreq == nfreq, f"Expected nfreq={nfreq}, got {frame.nfreq}"
    assert frame.ntime == time_samples_per_chunk, f"Expected ntime={time_samples_per_chunk}, got {frame.ntime}"
    assert frame.beam_id == beam_ids[0], f"Expected beam_id={beam_ids[0]}, got {frame.beam_id}"
    assert frame.time_chunk_index == 0, f"Expected time_chunk_index=0, got {frame.time_chunk_index}"
    
    # Check data shape: exposed as uint8 with shape (nfreq, ntime/2)
    data = np.asarray(frame.data)
    expected_shape = (nfreq, time_samples_per_chunk // 2)
    assert data.shape == expected_shape, f"Expected shape {expected_shape}, got {data.shape}"
    assert data.dtype == np.uint8, f"Expected dtype uint8, got {data.dtype}"
    
    # Check data initialization: all bytes should be 0x88 (int4 value -8 packed twice)
    assert np.all(data == 0x88), "Expected all data bytes to be 0x88 (int4 value -8)"
    
    # Verify we can modify the data (tests that it's writable)
    data[0, 0] = 0x12
    assert np.asarray(frame.data)[0, 0] == 0x12, "Data modification did not persist"
    
    print("    PASSED")


def test_sequence_ordering():
    """
    Test 4: Sequence ordering.
    
    Verifies that frames cycle through beam_ids before incrementing time_chunk_index:
      - For beam_ids=[A, B, C] and multiple time chunks:
        Frame 0: beam_id=A, time_chunk_index=0
        Frame 1: beam_id=B, time_chunk_index=0
        Frame 2: beam_id=C, time_chunk_index=0
        Frame 3: beam_id=A, time_chunk_index=1
        ...
    """
    print("  test_sequence_ordering()...")
    
    nfreq = 64
    time_samples_per_chunk = 128
    beam_ids = [5, 15, 25]
    num_beams = len(beam_ids)
    num_chunks = 4
    
    slab = make_slab_allocator()
    alloc = AssembledFrameAllocator(slab, num_consumers=1)
    alloc.initialize(0, nfreq, time_samples_per_chunk, beam_ids)
    
    # Verify allocator state after initialization
    assert alloc.nfreq == nfreq, f"Allocator nfreq mismatch: expected {nfreq}, got {alloc.nfreq}"
    assert alloc.time_samples_per_chunk == time_samples_per_chunk
    assert list(alloc.beam_ids) == beam_ids
    
    # Get frames and verify sequence
    for chunk_idx in range(num_chunks):
        for beam_idx in range(num_beams):
            frame = alloc.get_frame(0)
            expected_beam = beam_ids[beam_idx]
            expected_chunk = chunk_idx
            
            assert frame.beam_id == expected_beam, \
                f"Frame {chunk_idx * num_beams + beam_idx}: expected beam_id={expected_beam}, got {frame.beam_id}"
            assert frame.time_chunk_index == expected_chunk, \
                f"Frame {chunk_idx * num_beams + beam_idx}: expected time_chunk_index={expected_chunk}, got {frame.time_chunk_index}"
    
    print("    PASSED")


def test_single_beam_sequence():
    """
    Test 4 (edge case): Single beam sequence.
    
    Verifies correct behavior when there's only one beam_id.
    """
    print("  test_single_beam_sequence()...")
    
    nfreq = 32
    time_samples_per_chunk = 64
    beam_ids = [42]
    
    slab = make_slab_allocator()
    alloc = AssembledFrameAllocator(slab, num_consumers=1)
    alloc.initialize(0, nfreq, time_samples_per_chunk, beam_ids)
    
    # With single beam, each frame should have beam_id=42 and incrementing time_chunk_index
    for chunk_idx in range(5):
        frame = alloc.get_frame(0)
        assert frame.beam_id == 42, f"Expected beam_id=42, got {frame.beam_id}"
        assert frame.time_chunk_index == chunk_idx, \
            f"Expected time_chunk_index={chunk_idx}, got {frame.time_chunk_index}"
    
    print("    PASSED")


def test_multi_consumer_frame_identity():
    """
    Test 5: Multi-consumer frame identity.
    
    Verifies that multiple consumers calling get_frame() at the same sequence index
    receive the exact same object (identity check, not just equality).
    """
    print("  test_multi_consumer_frame_identity()...")
    
    nfreq = 64
    time_samples_per_chunk = 128
    beam_ids = [1, 2]
    num_consumers = 3
    
    slab = make_slab_allocator()
    alloc = AssembledFrameAllocator(slab, num_consumers=num_consumers)
    
    # Initialize all consumers with same parameters
    for consumer_id in range(num_consumers):
        alloc.initialize(consumer_id, nfreq, time_samples_per_chunk, beam_ids)
    
    # Get first frame from all consumers and verify identity
    frames = [alloc.get_frame(consumer_id) for consumer_id in range(num_consumers)]
    
    # All frames should be the exact same object
    for i in range(1, num_consumers):
        assert frames[i] is frames[0], \
            f"Consumer {i} got different frame object than consumer 0"
    
    # Verify modification is visible to all references
    np.asarray(frames[0].data)[0, 0] = 0xAB
    for i in range(1, num_consumers):
        assert np.asarray(frames[i].data)[0, 0] == 0xAB, \
            f"Modification not visible to consumer {i}"
    
    # Get second frame and verify all consumers get the same new frame
    frames2 = [alloc.get_frame(consumer_id) for consumer_id in range(num_consumers)]
    
    for i in range(1, num_consumers):
        assert frames2[i] is frames2[0], \
            f"Consumer {i} got different second frame object than consumer 0"
    
    # Second frame should be different from first frame
    assert frames2[0] is not frames[0], "Second frame should be different object from first"
    
    print("    PASSED")


def test_multi_consumer_independent_progress():
    """
    Test 5: Multi-consumer independent progress.
    
    Verifies that consumers can progress at different rates, each receiving
    the correct sequence of frames.
    """
    print("  test_multi_consumer_independent_progress()...")
    
    nfreq = 32
    time_samples_per_chunk = 64
    beam_ids = [100, 200]
    num_consumers = 2
    
    slab = make_slab_allocator()
    alloc = AssembledFrameAllocator(slab, num_consumers=num_consumers)
    
    for consumer_id in range(num_consumers):
        alloc.initialize(consumer_id, nfreq, time_samples_per_chunk, beam_ids)
    
    # Consumer 0 gets 4 frames (2 chunks * 2 beams)
    frames_c0 = [alloc.get_frame(0) for _ in range(4)]
    
    # Consumer 1 gets only 2 frames (1 chunk * 2 beams)
    frames_c1 = [alloc.get_frame(1) for _ in range(2)]
    
    # Verify consumer 0's sequence
    expected_c0 = [
        (100, 0), (200, 0),  # chunk 0
        (100, 1), (200, 1),  # chunk 1
    ]
    for i, (beam, chunk) in enumerate(expected_c0):
        assert frames_c0[i].beam_id == beam, \
            f"Consumer 0, frame {i}: expected beam_id={beam}, got {frames_c0[i].beam_id}"
        assert frames_c0[i].time_chunk_index == chunk, \
            f"Consumer 0, frame {i}: expected time_chunk_index={chunk}, got {frames_c0[i].time_chunk_index}"
    
    # Verify consumer 1's sequence
    expected_c1 = [(100, 0), (200, 0)]
    for i, (beam, chunk) in enumerate(expected_c1):
        assert frames_c1[i].beam_id == beam, \
            f"Consumer 1, frame {i}: expected beam_id={beam}, got {frames_c1[i].beam_id}"
        assert frames_c1[i].time_chunk_index == chunk, \
            f"Consumer 1, frame {i}: expected time_chunk_index={chunk}, got {frames_c1[i].time_chunk_index}"
    
    # Verify frame identity: first 2 frames should be shared
    assert frames_c1[0] is frames_c0[0], "First frame should be shared between consumers"
    assert frames_c1[1] is frames_c0[1], "Second frame should be shared between consumers"
    
    # Now consumer 1 catches up
    frames_c1.append(alloc.get_frame(1))  # beam 100, chunk 1
    frames_c1.append(alloc.get_frame(1))  # beam 200, chunk 1
    
    assert frames_c1[2] is frames_c0[2], "Third frame should be shared"
    assert frames_c1[3] is frames_c0[3], "Fourth frame should be shared"
    
    print("    PASSED")


def test_frame_recycling():
    """
    Test 6 (partial): Frame recycling.
    
    Verifies that frames are returned to the pool when all consumers have
    received and released them. Uses a small slab allocator to force recycling.
    """
    print("  test_frame_recycling()...")
    
    nfreq = 64
    time_samples_per_chunk = 128
    beam_ids = [1]  # Single beam for simplicity
    num_consumers = 2
    
    # Calculate slab size: (nfreq * time_samples_per_chunk) / 2 bytes
    slab_size = (nfreq * time_samples_per_chunk) // 2  # 4096 bytes
    
    # Create allocator with capacity for exactly 3 slabs
    # Add some margin for alignment (SlabAllocator aligns to 128 bytes)
    capacity = slab_size * 3 + 1024
    slab = make_slab_allocator(capacity=capacity)
    alloc = AssembledFrameAllocator(slab, num_consumers=num_consumers)
    
    for consumer_id in range(num_consumers):
        alloc.initialize(consumer_id, nfreq, time_samples_per_chunk, beam_ids)
    
    # Note: num_free_frames() can only be called after first get_frame() establishes slab size
    
    # Consumer 0 gets frame 0 - this establishes the slab size
    frame0_c0 = alloc.get_frame(0)
    
    # Now we can check slab counts
    initial_total = alloc.num_total_frames()
    assert initial_total == 3, f"Expected 3 total frames, got {initial_total}"
    assert alloc.num_free_frames() == 2, f"Expected 2 free after first get_frame, got {alloc.num_free_frames()}"
    
    # Consumer 1 gets frame 0 - same slab, no new allocation
    frame0_c1 = alloc.get_frame(1)
    assert frame0_c1 is frame0_c0, "Should be same frame object"
    # Both consumers have received frame 0, so allocator drops its reference
    assert alloc.num_free_frames() == 2, f"Expected 2 free frames, got {alloc.num_free_frames()}"
    
    # Now release references from both consumers - frame should be recycled
    del frame0_c0
    del frame0_c1
    
    # Frame should now be back in the pool
    assert alloc.num_free_frames() == 3, f"Expected 3 free after releasing, got {alloc.num_free_frames()}"
    
    # Test that we can allocate again after recycling
    frame1_c0 = alloc.get_frame(0)
    assert frame1_c0.time_chunk_index == 1, "Should be chunk index 1"
    assert alloc.num_free_frames() == 2
    
    frame1_c1 = alloc.get_frame(1)
    assert frame1_c1 is frame1_c0
    
    del frame1_c0
    del frame1_c1
    assert alloc.num_free_frames() == 3, "Should have 3 free after cleanup"
    
    print("    PASSED")


def test_frame_recycling_with_held_reference():
    """
    Test 6 (partial): Frame recycling with held reference.
    
    Verifies frame recycling behavior:
    - Frames are recycled when all consumers have received them AND no Python references remain
    - If a consumer holds a reference, the slab won't be freed even after the allocator drops it
    """
    print("  test_frame_recycling_with_held_reference()...")
    
    nfreq = 64
    time_samples_per_chunk = 128
    beam_ids = [1]
    num_consumers = 2
    
    slab_size = (nfreq * time_samples_per_chunk) // 2
    capacity = slab_size * 4 + 1024  # 4 slabs
    slab = make_slab_allocator(capacity=capacity)
    alloc = AssembledFrameAllocator(slab, num_consumers=num_consumers)
    
    for consumer_id in range(num_consumers):
        alloc.initialize(consumer_id, nfreq, time_samples_per_chunk, beam_ids)
    
    # Consumer 0 gets frames 0, 1, 2 (first get_frame establishes slab size)
    frame0 = alloc.get_frame(0)
    
    # Now we can verify total slabs
    assert alloc.num_total_frames() == 4, f"Expected 4 total frames, got {alloc.num_total_frames()}"
    assert alloc.num_free_frames() == 3, f"Expected 3 free after first frame, got {alloc.num_free_frames()}"
    
    frame1 = alloc.get_frame(0)
    frame2 = alloc.get_frame(0)
    
    # 3 slabs allocated, 1 free
    assert alloc.num_free_frames() == 1, f"Expected 1 free, got {alloc.num_free_frames()}"
    
    # Consumer 1 gets frame 0
    frame0_c1 = alloc.get_frame(1)
    assert frame0_c1 is frame0
    
    # Both consumers received frame 0 - allocator drops its reference.
    # But consumer 0 and consumer 1 both hold Python references, so slab not freed.
    assert alloc.num_free_frames() == 1
    
    # Release consumer 1's reference to frame 0
    del frame0_c1
    # Consumer 0 still holds frame0, so slab not recycled
    assert alloc.num_free_frames() == 1
    
    # Release consumer 0's references to frames 1 and 2
    del frame1
    del frame2
    # These frames are still in allocator's queue (consumer 1 hasn't received them)
    # So still 1 free
    assert alloc.num_free_frames() == 1
    
    # Consumer 1 catches up on frame 1
    # Consumer 1 is the last receiver, allocator drops its reference.
    # Consumer 0 already released its reference, so frame 1 is recycled immediately.
    frame1_c1 = alloc.get_frame(1)
    assert alloc.num_free_frames() == 2, f"Expected 2 free after frame1 recycled, got {alloc.num_free_frames()}"
    
    # Consumer 1 gets frame 2 - same situation, frame 2 is recycled
    frame2_c1 = alloc.get_frame(1)
    assert alloc.num_free_frames() == 3, f"Expected 3 free after frame2 recycled, got {alloc.num_free_frames()}"
    
    # frame1_c1 and frame2_c1 are already recycled (their slabs returned to pool)
    # but we still hold the Python shared_ptr references
    del frame1_c1
    del frame2_c1
    assert alloc.num_free_frames() == 3, f"Expected 3 free, got {alloc.num_free_frames()}"
    
    # Finally release frame 0
    del frame0
    assert alloc.num_free_frames() == 4, f"Expected 4 free, got {alloc.num_free_frames()}"
    
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
    
    # Test 6 (partial): Frame recycling
    test_frame_recycling()
    test_frame_recycling_with_held_reference()
    
    print("All AssembledFrameAllocator tests PASSED!")
