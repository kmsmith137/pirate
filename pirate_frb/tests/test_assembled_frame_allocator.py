"""
Unit tests for AssembledFrameAllocator.

Tests covered:
  - Frame set allocation and properties (metadata, data shape, initialization)
  - Sequence ordering (set / beam cycling, time chunks)
  - Multi-consumer scenarios (set/frame identity, independent progress)
  - Set recycling
  - throw_exception_if_empty (startup burst, fail-fast get_frame_set)

get_frame_set(time_chunk_index) returns the AssembledFrameSet (= nbeams
frames) for one time chunk; each chunk must be requested exactly
num_consumers times in total (the set is evicted on its last receipt).
Frame identity within a set is determined by the allocator --
frames[i].beam_id == metadata.beam_ids[i].

Run via: python -m pirate_frb test --net
"""

import numpy as np
from ..core import AssembledFrameAllocator, BumpAllocator, SlabAllocator, XEngineMetadata
from ..pirate_pybind11 import constants
from ..utils import atomic_print


def make_slab_allocator(capacity=4*1024*1024, aflags='af_rhost'):
    """Create a host-memory SlabAllocator.

    Backed by a dedicated BumpAllocator of the given capacity; slabs are carved
    on demand."""
    bump = BumpAllocator(aflags, capacity)
    return SlabAllocator(bump)


def _align_up_128(n):
    """Round n up to a 128-byte boundary.

    The SlabAllocator rounds each requested slab size up to a 128-byte (GPU cache
    line) boundary."""
    return -(-n // 128) * 128


def _make_counted_slab_allocator(slab_size, num_slabs_requested):
    """Helper for tests that need to know the EXACT slab count.

    Returns (slab_allocator, num_slabs).

    The BumpAllocator rounds its capacity up to a page multiple, so simply
    requesting num_slabs_requested * slab_size bytes can yield extra slabs.
    Derive the actual count from bump.capacity (the rounded value) instead
    of assuming."""
    aligned_slab = _align_up_128(slab_size)
    bump = BumpAllocator('af_rhost', num_slabs_requested * aligned_slab)
    num_slabs = bump.capacity // aligned_slab
    assert num_slabs >= num_slabs_requested
    return SlabAllocator(bump), num_slabs


def _test_metadata(nfreq, beam_ids):
    """Helper: construct a fully-valid XEngineMetadata for one-zone tests."""
    return XEngineMetadata.make_fiducial([nfreq], [400.0, 800.0], beam_ids, 1.0)


def test_frame_properties():
    """
    Frame set + frame allocation and properties.

    Verifies:
      - Set metadata (nfreq, ntime, nbeams, time_chunk_index, len(frames))
      - Each frame's metadata (nfreq, ntime, beam_id, time_chunk_index)
      - Data array has correct shape (nfreq, ntime/2 as uint8)
      - Data is initialized to 0x88 (representing -8 in int4)
    """
    atomic_print("  test_frame_properties()...")

    nfreq = 128
    time_samples_per_chunk = 256
    beam_ids = [10, 20, 30]

    slab = make_slab_allocator()
    alloc = AssembledFrameAllocator(slab, num_consumers=1, time_samples_per_chunk=time_samples_per_chunk, throw_exception_if_empty=False)
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

    atomic_print("    PASSED")


def test_sequence_ordering():
    """
    Sequence ordering.

    Verifies that get_frame_set(chunk_idx) returns the set for exactly that
    chunk index, and that each set contains frames in metadata.beam_ids order.

    Run at nbeams > 1 and at nbeams == 1: a one-frame set is the edge case
    where a per-beam indexing error has nothing to disagree with.
    """
    atomic_print("  test_sequence_ordering()...")

    time_samples_per_chunk = 256

    for (nfreq, beam_ids, num_chunks) in [(64, [5, 15, 25], 4), (32, [42], 5)]:
        slab = make_slab_allocator()
        alloc = AssembledFrameAllocator(slab, num_consumers=1, time_samples_per_chunk=time_samples_per_chunk, throw_exception_if_empty=False)
        alloc.initialize_metadata(_test_metadata(nfreq, beam_ids))
        alloc.initialize_initial_chunk(0)

        # Verify allocator state after initialization.
        assert alloc.nfreq == nfreq
        assert alloc.time_samples_per_chunk == time_samples_per_chunk
        assert list(alloc.beam_ids) == beam_ids

        # Walk through num_chunks sets and verify each.
        for chunk_idx in range(num_chunks):
            fset = alloc.get_frame_set(chunk_idx)
            assert fset.time_chunk_index == chunk_idx, \
                f"Set {chunk_idx}: expected time_chunk_index={chunk_idx}, got {fset.time_chunk_index}"
            assert len(fset.frames) == len(beam_ids)

            for beam_idx, frame in enumerate(fset.frames):
                assert frame.beam_id == beam_ids[beam_idx], \
                    f"Chunk {chunk_idx}, frame {beam_idx}: expected beam_id={beam_ids[beam_idx]}, got {frame.beam_id}"
                assert frame.time_chunk_index == chunk_idx

    atomic_print("    PASSED")


def test_multi_consumer_frame_identity():
    """
    Multi-consumer set+frame identity.

    Verifies that multiple consumers requesting the same chunk index receive
    the exact same set object (and therefore the exact same frame objects
    inside the set).
    """
    atomic_print("  test_multi_consumer_frame_identity()...")

    nfreq = 64
    time_samples_per_chunk = 256
    beam_ids = [1, 2]
    num_consumers = 3

    slab = make_slab_allocator()
    alloc = AssembledFrameAllocator(slab, num_consumers=num_consumers, time_samples_per_chunk=time_samples_per_chunk, throw_exception_if_empty=False)

    # Initialize the allocator (one initialize_metadata call is enough, but
    # the multi-call path is tested incidentally).
    for _ in range(num_consumers):
        alloc.initialize_metadata(_test_metadata(nfreq, beam_ids))
    alloc.initialize_initial_chunk(0)

    # Request chunk 0 once per consumer; all should get the same object.
    sets0 = [alloc.get_frame_set(0) for _ in range(num_consumers)]
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

    # Request chunk 1 once per consumer and verify identity again.
    sets1 = [alloc.get_frame_set(1) for _ in range(num_consumers)]
    for i in range(1, num_consumers):
        assert sets1[i] is sets1[0], \
            f"Consumer {i} got a different second-set object than consumer 0"

    # Second set should be a different object from the first.
    assert sets1[0] is not sets0[0], "Second set should be a different object from first"

    atomic_print("    PASSED")


def test_multi_consumer_independent_progress():
    """
    Multi-consumer independent progress.

    Verifies that consumers can progress at different rates (in chunk units),
    each receiving the correct sequence of AssembledFrameSets.
    """
    atomic_print("  test_multi_consumer_independent_progress()...")

    nfreq = 32
    time_samples_per_chunk = 256
    beam_ids = [100, 200]
    num_consumers = 2

    slab = make_slab_allocator()
    alloc = AssembledFrameAllocator(slab, num_consumers=num_consumers, time_samples_per_chunk=time_samples_per_chunk, throw_exception_if_empty=False)

    for _ in range(num_consumers):
        alloc.initialize_metadata(_test_metadata(nfreq, beam_ids))
    alloc.initialize_initial_chunk(0)

    # Consumer 0 reads 2 chunks (= 2 sets).
    sets_c0 = [alloc.get_frame_set(chunk) for chunk in range(2)]

    # Consumer 1 reads only 1 chunk so far.
    sets_c1 = [alloc.get_frame_set(0)]

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

    # Consumer 1 catches up (requests chunk 1).
    sets_c1.append(alloc.get_frame_set(1))
    assert sets_c1[1] is sets_c0[1], \
        "Second set should be shared between consumers"

    atomic_print("    PASSED")


def test_frame_recycling():
    """
    Set recycling.

    Verifies that frame sets are returned to the pool when all consumers have
    received and released them. Uses a small slab allocator to force recycling.

    With nbeams=1, one set = one slab; allocator capacity for N slabs means N
    sets resident at a time. We verify we can allocate many more than N sets
    in succession, proving that recycling is happening.

    With the worker thread, exact slab counts are non-deterministic (the worker
    can grab returned slabs to pre-create sets). So we test recycling by
    verifying the no-block / no-deadlock property over many iterations.
    """
    atomic_print("  test_frame_recycling()...")

    nfreq = 64
    time_samples_per_chunk = 256
    beam_ids = [1]  # Single beam: one slab per set.
    num_consumers = 2

    # Per-frame slab size; with nbeams=1, also the per-set slab footprint.
    # Each slab holds scales_offsets (nfreq, mpc, 2) float16 = nfreq*mpc*4 bytes
    # plus int4 data (nfreq, tspc) = nfreq*tspc/2 bytes.
    mpc = time_samples_per_chunk // 256
    slab_size = nfreq * mpc * 4 + (nfreq * time_samples_per_chunk) // 2

    # Small pool (~3 slabs = 3 sets); the exact count is derived from the
    # page-rounded BumpAllocator capacity, not assumed.
    slab, num_slabs = _make_counted_slab_allocator(slab_size, 3)
    alloc = AssembledFrameAllocator(slab, num_consumers=num_consumers, time_samples_per_chunk=time_samples_per_chunk, throw_exception_if_empty=False)

    for _ in range(num_consumers):
        alloc.initialize_metadata(_test_metadata(nfreq, beam_ids))
    alloc.initialize_initial_chunk(0)

    # First set establishes slab size in the underlying allocator.
    set0_c0 = alloc.get_frame_set(0)

    # Consumer 1 requests chunk 0 - same set object, no new allocation.
    set0_c1 = alloc.get_frame_set(0)
    assert set0_c1 is set0_c0

    # Release references - set should be recycled.
    del set0_c0
    del set0_c1

    # Allocate again after recycling.
    set1_c0 = alloc.get_frame_set(1)
    assert set1_c0.time_chunk_index == 1
    set1_c1 = alloc.get_frame_set(1)
    assert set1_c1 is set1_c0
    del set1_c0
    del set1_c1

    # Prove recycling works by allocating many more sets than we have slabs.
    # With only num_slabs slabs but far more sets, recycling must be happening.
    num_sets_to_allocate = 5 * num_slabs + 5
    for i in range(num_sets_to_allocate):
        s0 = alloc.get_frame_set(2 + i)
        assert s0.time_chunk_index == 2 + i, \
            f"Expected chunk index {2+i}, got {s0.time_chunk_index}"
        s1 = alloc.get_frame_set(2 + i)
        assert s1 is s0
        del s0
        del s1

    # If we got here without blocking/deadlock, recycling is working.
    atomic_print("    PASSED")


def test_frame_recycling_with_held_reference():
    """
    Set recycling with held reference.

    Verifies that:
    - Sets are recycled when all consumers have received them AND no Python
      references remain.
    - If a consumer holds a reference, the underlying slab won't be freed
      even after the allocator drops its reference.
    """
    atomic_print("  test_frame_recycling_with_held_reference()...")

    nfreq = 64
    time_samples_per_chunk = 256
    beam_ids = [1]  # Single beam: one slab per set.
    num_consumers = 2

    # See test_frame_recycling for slab_size derivation. Small pool (~4
    # slabs); the exact count is derived from the page-rounded BumpAllocator
    # capacity, not assumed.
    mpc = time_samples_per_chunk // 256
    slab_size = nfreq * mpc * 4 + (nfreq * time_samples_per_chunk) // 2
    slab, num_slabs = _make_counted_slab_allocator(slab_size, 4)
    alloc = AssembledFrameAllocator(slab, num_consumers=num_consumers, time_samples_per_chunk=time_samples_per_chunk, throw_exception_if_empty=False)

    for _ in range(num_consumers):
        alloc.initialize_metadata(_test_metadata(nfreq, beam_ids))
    alloc.initialize_initial_chunk(0)

    # Consumer 0 gets sets 0, 1, 2.
    set0 = alloc.get_frame_set(0)

    # Write a marker into set0's frame data. If set0's slab were ever wrongly
    # recycled while we hold this reference, the worker's pre-init memset
    # (0x88 fill) would clobber the marker -- checked at the end of the test.
    set0.frames[0].data[:] = 0x77

    set1 = alloc.get_frame_set(1)
    set2 = alloc.get_frame_set(2)

    # Consumer 1 requests chunk 0 - same object as set0.
    set0_c1 = alloc.get_frame_set(0)
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

    # Consumer 1 catches up (chunks 1 and 2). Consumer 1 is the last
    # receiver, so the allocator drops its reference. Consumer 0 already
    # released its reference, so sets 1 and 2 are recycled immediately.
    set1_c1 = alloc.get_frame_set(1)
    set2_c1 = alloc.get_frame_set(2)
    del set1_c1
    del set2_c1

    # With set0 still held (pinning one of the num_slabs slabs), cycle
    # through more chunks than the remaining slabs. This only completes if
    # released sets are recycled (on last receipt + refcount zero) -- i.e.
    # recycling works even while another set's slab is pinned by a held
    # reference.
    for i in range(num_slabs + 6):
        sa = alloc.get_frame_set(3 + i)
        sb = alloc.get_frame_set(3 + i)
        assert sb is sa
        del sa, sb

    # set0's data must have survived all that recycling untouched.
    assert np.all(set0.frames[0].data == 0x77), \
        "Held set's slab was clobbered -- wrongly recycled while referenced"

    # Finally release set 0.
    del set0

    atomic_print("    PASSED")


def test_throw_exception_if_empty():
    """
    throw_exception_if_empty=True (startup burst + fail-fast get_frame_set).

    Verifies:
      (a) the worker's startup burst pre-allocates
          constants.assembled_frame_allocator_initial_size sets, and chunks
          within the burst window are served;
      (b) requesting a chunk past the queue frontier raises the
          "not immediately ready" fail-fast error (instead of blocking);
      (c) a pool too small for the burst raises the verbose
          pool-exhausted-during-startup error from get_frame_set();
      (d) throw_exception_if_empty=True with a dummy-mode slab allocator raises at
          construction.
    """
    atomic_print("  test_throw_exception_if_empty()...")

    nfreq = 64
    time_samples_per_chunk = 256
    beam_ids = [1]  # Single beam: one slab per set.
    initial_size = constants.assembled_frame_allocator_initial_size

    mpc = time_samples_per_chunk // 256
    slab_size = nfreq * mpc * 4 + (nfreq * time_samples_per_chunk) // 2

    # (a) + (b): pool sized for exactly the burst. The determinism of (b)
    # below relies on the pool having NO slabs beyond the burst (the worker
    # would otherwise carve extras in its main loop, racing our frontier
    # request); page rounding happens not to add any at this slab_size, and
    # this assert makes it loud if that ever changes.
    slab, num_slabs = _make_counted_slab_allocator(slab_size, initial_size)
    assert num_slabs == initial_size, \
        f"page rounding added slabs ({num_slabs} != {initial_size}); test needs adjusting"
    alloc = AssembledFrameAllocator(slab, num_consumers=1, time_samples_per_chunk=time_samples_per_chunk,
                                    throw_exception_if_empty=True)
    alloc.initialize_metadata(_test_metadata(nfreq, beam_ids))
    alloc.initialize_initial_chunk(0)

    # Consume a few burst chunks, HOLDING the sets (so their slabs are never
    # recycled). The receipts shrink the queue below the steady-state bound,
    # so the worker wakes -- but the pool is exhausted and nothing is ever
    # freed, so it parks in blocking get_slab() and the queue frontier is
    # deterministically frozen at chunk initial_size.
    held = [alloc.get_frame_set(c) for c in range(3)]
    for c, fset in enumerate(held):
        assert fset.time_chunk_index == c

    # (b) Request the frontier chunk itself: deterministically not in the
    # queue (the worker cannot have built it -- we hold all freed slabs),
    # so throw_exception_if_empty must raise rather than block.
    try:
        alloc.get_frame_set(initial_size)
        raise AssertionError("expected 'not immediately ready' error")
    except RuntimeError as e:
        assert "not immediately ready" in str(e), f"unexpected error: {e}"

    # The fail-fast throw stopped the allocator (entry-point policy).
    del held, alloc, slab

    # (c) Pool too small for the burst: the worker's fail-fast error
    # surfaces from get_frame_set() (which waits on the queue_initialized
    # latch, wakes on the stop, and rethrows the saved burst error).
    slab, num_slabs = _make_counted_slab_allocator(slab_size, 2)
    assert num_slabs < initial_size   # else the burst would succeed
    alloc = AssembledFrameAllocator(slab, num_consumers=1, time_samples_per_chunk=time_samples_per_chunk,
                                    throw_exception_if_empty=True)
    alloc.initialize_metadata(_test_metadata(nfreq, beam_ids))
    alloc.initialize_initial_chunk(0)
    try:
        alloc.get_frame_set(0)
        raise AssertionError("expected startup-burst pool-exhausted error")
    except RuntimeError as e:
        assert "startup burst" in str(e), f"unexpected error: {e}"
    del alloc, slab

    # (d) Production mode requires a non-dummy slab allocator.
    try:
        AssembledFrameAllocator(SlabAllocator('af_rhost'), num_consumers=1,
                                time_samples_per_chunk=time_samples_per_chunk, throw_exception_if_empty=True)
        raise AssertionError("expected throw_exception_if_empty+dummy constructor error")
    except RuntimeError as e:
        assert "throw_exception_if_empty" in str(e), f"unexpected error: {e}"

    atomic_print("    PASSED")


def test_assembled_frame_allocator():
    """
    Run all AssembledFrameAllocator unit tests.

    Raises an exception if any test fails.
    """
    atomic_print("Testing AssembledFrameAllocator...")

    test_frame_properties()
    test_sequence_ordering()
    test_multi_consumer_frame_identity()
    test_multi_consumer_independent_progress()
    test_frame_recycling()
    test_frame_recycling_with_held_reference()
    test_throw_exception_if_empty()

    atomic_print("All AssembledFrameAllocator tests PASSED!")
