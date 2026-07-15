#ifndef _PIRATE_ASSEMBLED_FRAME_HPP
#define _PIRATE_ASSEMBLED_FRAME_HPP

#include <ksgpu/Array.hpp>

#include <mutex>
#include <deque>
#include <vector>
#include <thread>
#include <memory>
#include <exception>
#include <filesystem>
#include <condition_variable>


namespace pirate {
#if 0
}  // editor auto-indent
#endif

class SlabAllocator;     // defined in SlabAllocator.hpp
struct XEngineMetadata;  // defined in XEngineMetadata.hpp
struct FileStream;       // defined in FileWriter.hpp
namespace simpulse { struct SinglePulse; }  // defined in simpulse.hpp


// AssembledFrame: This central data structure represents one data frame on the server,
// after they have been received from the X-engine and "assembled" (parsed/reordered).
//
// Reminders:
//
//  - A "frame" is a (time_chunk, beam_id) pair.
//
//  - A "chunk", aka "time chunk", is a fixed number of samples, defined in
//      a high-level parameter file (yaml key 'time_samples_per_chunk')
//
//  - A "minichunk" is 256 time samples. This is the time cadence of the network
//      protocol, and is used throughout the network receive code, before chunks
//      are "assembled".

struct AssembledFrame
{
    // Initialized at creation, not protected by lock.
    long nfreq = 0;
    long ntime = 0;
    long beam_id = 0;
    long time_chunk_index = 0;

    // Shared X-engine metadata for this frame. Set at frame creation by
    // AssembledFrameAllocator::_create_frame (or by AssembledFrame::make_uninitialized
    // / AssembledFrame::from_asdf for non-allocator code paths). Treated as
    // immutable after creation -- all frames from one allocator share a single
    // XEngineMetadata instance via this shared_ptr. INVARIANT: non-null on
    // every constructed AssembledFrame.
    //
    // freq_channels: always FREQUENCY-SCRUBBED (empty). The allocator path
    // scrubs in AssembledFrameAllocator::_initialize_metadata; the make_uninitialized
    // and from_asdf paths produce frames whose metadata starts scrubbed.
    //
    // Note: XEngineMetadata round-trips bit-exactly through YAML, but is
    // projected through ASDF (see XEngineMetadata.hpp for details). After a
    // from_asdf() the metadata's beam_ids / beam_positions_x / beam_positions_y
    // are length-1 (just this frame's beam).
    std::shared_ptr<const XEngineMetadata> metadata;

    // 'scales_offsets' and 'data' share a single slab, and BOTH arrays are
    // individually aligned to constants::bytes_per_gpu_cache_line:
    // 'scales_offsets' is at slab offset 0 (the slab base is cache-line aligned
    // by the SlabAllocator), and 'data' is at slab offset
    // align_up(scales_offsets_nbytes, bytes_per_gpu_cache_line) so its base is
    // cache-line aligned too. This ordering matches the per-minichunk wire
    // layout (scales_offsets precedes the int4 payload). See
    // AssembledFrameAllocator::get_layout() for the exact offsets/sizes -- the
    // single source of truth shared by the allocator's worker thread (which
    // builds the frames) and external slab-pool sizing. _reap_locked()
    // releases both arrays together by dropping the slab shared_ptr.
    //
    // Warning: if the AssembledFrame has been "reaped" (under memory
    // pressure, or because a disk write failed -- see _reap_locked()), then
    // both 'scales_offsets' and 'data' are empty arrays. The array state
    // (empty vs nonempty) is protected by the lock, but the array contents
    // are not lock-protected.

    // dtype float16, shape (nfreq, mpc, 2) where mpc = ntime / 256. The last
    // axis is {scale, offset}. Newly allocated frames have all elements zero.
    ksgpu::Array<void> scales_offsets;

    // dtype int4, shape (nfreq, ntime). Newly allocated frames have all
    // 'data' elements set to (-8) (the "missing sample" mask).
    ksgpu::Array<void> data;

    // ASDF file I/O.
    // Note: int4 dtype is stored as uint8 with shape (nfreq, ntime/2).
    //
    // Note: if sync=true in write_asdf(), then we fsync() the file at the end.
    // This is the default, to avoid runaway page cache usage. I tested that it
    // still gives full ssd bandwith, if multiple writer threads are used.
    //
    // A future project: faster asdf?
    //
    //  - Currently, memory bandwidth of AssembledFrame::write_asdf() is 4-5x the memory
    //    footprint. (Empirical, with 'pirate_frb hwtest configs/hwtest/cf00_asdf.yml'
    //    and 'pcm-memory'.) This is potentially an issue (see 'misc/plan_hmem_bw.py').
    //
    //  - For binary "blobs", writing with open(..., O_DIRECT) -> write() is 1x the
    //    footprint, and does not pollute the page cache, which seems important in
    //    the larger footprint.
    //
    //  - Achieving this performance in a general purpose asdf implementation seems
    //    challenging due to dual alignment requirements (O_DIRECT requires all writes
    //    to be logically aligned on 4k boundaries within the file, and pointers to be
    //    4k-aligned).
    //
    //  - Idea: write a short standalone asdf writer which is integrated with the
    //    AssembledFrameAllocator (see below). The allocator uses a single memory
    //    slab, which is both "backing" memory for the C++ arrays in the real-time
    //    server, and "backing" memory for the asdf file, so that the file can be
    //    written with a single call to write().
    //
    //  - Layout is nontrivial -- need to leave space for the yaml header, and gaps
    //    between the arrays. There are some mild alignment requirements (C++ code
    //    generally assumes that Arrays are 128-byte aligned, and O_DIRECT assumes
    //    that the base pointer is page-aligned. I think that the asdf format is
    //    flexible enough that these alignment requirements are not a problem.
    //
    //  - Note: according to google gemini, to determine the O_DIRECT page alignment
    //    requirement, you should call ioctl(fd, BLKPBSZGET), not BLKSSZGET.
    
    // If verbose=true, write_asdf() emits comments throughout the YAML header.
    // The comments should be detailed enough that the header can serve as
    // self-contained documentation of the file format. Intended for use
    // with the 'pirate_frb show_file_format' CLI hook.
    void write_asdf(const std::string &filename, bool sync=true, bool verbose=false) const;
    static std::shared_ptr<AssembledFrame> from_asdf(const std::string &filename);    // Call without lock held.

    // FPGA sequence number at the start / end of this frame (i.e. of time chunk
    // 'time_chunk_index'). fpga_seq_start() = time_chunk_index * ntime *
    // metadata->seq_per_frb_time_sample; fpga_seq_end() is the start of the NEXT
    // chunk (one-past-the-end). Requires non-null 'metadata' (invariant).
    long fpga_seq_start() const;
    long fpga_seq_end() const;

    // Dequantize this frame's int4 'data' to float32, applying the per-(freq,
    // minichunk) affine transform from 'scales_offsets' (via a
    // ReferenceDequantizationKernel). Returns a freshly-allocated host
    // Array<float> of shape (nfreq, ntime); the -8 "missing sample" sentinel maps
    // to 0. Single-beam, so the result has no leading beam axis.
    ksgpu::Array<float> dequantize() const;

    // --------------------  private/internal  --------------------

    // Call with lock held!
    // Called by reaper thread, or ssd writer thread.
    void _reap_locked();

    // Create an AssembledFrame with freshly-allocated (registered-host) data and
    // scales_offsets arrays whose CONTENTS ARE UNSPECIFIED -- the caller is
    // responsible for filling them (e.g. via randomize(), an ASDF read, or a
    // memcpy). Used for testing and by from_asdf(). Note: ntime must be a
    // multiple of 256. Throws if 'xmd' is null, or if 'beam_id' is not in
    // 'xmd->beam_ids'. The returned frame's metadata is set to 'xmd' (shared,
    // no copy); nfreq is taken from xmd->get_total_nfreq().
    //
    // xmd.freq_channels: IGNORED (only beam_ids / beam_positions_* / zone_nfreq /
    // zone_freq_edges are read out of xmd here). Callers should typically pass a
    // frequency-scrubbed xmd, so the returned frame's metadata matches the
    // "always frequency-scrubbed" invariant on AssembledFrame::metadata.
    static std::shared_ptr<AssembledFrame>
    make_uninitialized(const std::shared_ptr<const XEngineMetadata> &xmd,
                       long ntime, long beam_id, long time_chunk_index);

    // randomize(): fill the frame with random test data (optionally with an injected FRB pulse).
    //   - gaussian=false: data buffer = uniform bits (int4s uniform over [-8,+7]).
    //   - gaussian=true:  data buffer = simulated Gaussian noise quantized to int4, clamped to
    //         [-7,+7] via avx2_simulate_4bit_noise() (the -8 sentinel never occurs).
    //
    // scales/offsets are filled the same way regardless of 'gaussian':
    //   - normalize=false: scales uniform in [0,1], offsets uniform in [-1,1].
    //   - normalize=true:  CALIBRATED -- offset = 0 and scale = S per frequency zone, chosen so the
    //         dequantized data has the per-zone noise variance in this frame's own 'metadata'
    //         (metadata->noise_variance). (S = sqrt(V/Vq), where Vq is the int4 data variance:
    //         17.5 for uniform [folding the -8 sentinel to 0], avx2_4bit_postquant_noise_rms()^2 for
    //         gaussian. See src_lib/AssembledFrame.cpp for the derivation.) 'metadata' is non-null
    //         by invariant, and its zone_nfreq sums to this frame's nfreq by construction.
    //
    // Pulse injection ('sp' nonempty): add a simulated FRB on top of the noise. REQUIRES
    // gaussian=true and normalize=true (throws otherwise), and that 'sp' is consistent with this
    // frame's 'metadata' -- same nfreq, per-channel freq edges, per-channel noise variances, and
    // time-sample duration (throws otherwise). The pulse channel i maps DIRECTLY to frame row i
    // (both ordered low-to-high in frequency). 'dt_sp' is the frame's time offset relative to the
    // pulse: frame time sample it_frame corresponds to pulse sample (it_frame + dt_sp). The pulse
    // is sparse, so most of the frame is still the fast avx2 noise fill; only each channel's
    // (contiguous) pulse samples are recomputed as quantize(signal/S[f] + prequant_rms*gaussian).
    // Partial overlap (part of the pulse falls outside [0,ntime)) is fine -- it is simply clipped.
    // sp->freq_it0 may be negative (a pulse whose arrival extends to t < 0); the per-channel
    // window clipping handles any (freq_it0, dt_sp) combination.
    //
    // Thread-safety: snapshots the lock-protected 'scales_offsets'/'data' Arrays
    // under the lock, then fills them without the lock held (the snapshot pins
    // the slab so a concurrent _reap_locked() cannot free it underneath). 'metadata'
    // and 'sp' are immutable, so reading them needs no lock. Safe to call
    // concurrently with reaping; on a reaped (empty) frame it is a no-op. Callers
    // still must not run two randomize() (or other writers) on the SAME frame
    // concurrently -- the buffer contents are not lock-protected.

    void randomize(bool normalize, bool gaussian,
                   const std::shared_ptr<const simpulse::SinglePulse> &sp, long dt_sp);
    
    // Members after this point are internal state.
    // These members are protected by the mutex, and are not saved to the ASDF file.

    mutable std::mutex mutex;
    long finalize_count = 0;    // incremented by FrbServer receiver thread(s)

    // One queued write request: a relative output path, plus the FileStream
    // that requested it (null pointer = WriteFiles-triggered). The pointer
    // gives the FileWriter worker threads access to the stream throughout
    // the file-writing code: at completion they bump
    // FileStream::num_files_written / num_files_errored, and copy
    // stream->stream_name into the per-file WriteStatus so SubscribeFiles
    // notifications can report which stream (if any) triggered each file.
    struct SaveRequest {
        std::filesystem::path path;
        std::shared_ptr<FileStream> stream;
    };

    // NOTE: save_paths are never taken verbatim from RPC clients: the client
    // supplies only an acquisition directory, which the server validates with
    // pirate::validate_acqdir() (nonempty, safe, canonical relative path)
    // BEFORE building the path via pirate::make_acq_relpath() -- the basename
    // is server-generated and safe by construction.
    //
    // Duplicate paths are allowed (e.g. two streams registered with the same
    // acqdir); FileWriter's NFS thread skips the filesystem operation for a
    // duplicate of an earlier entry, but still emits a WriteStatus for it
    // (one notification per entry, always).

    std::vector<SaveRequest> save_paths;

    // Save state. These members are protected by the lock, and can only be modified
    // by member functions of 'class FileWriter'. In particular, when an RPC thread
    // wants to add a save_path, it should acquire the lock, call save_paths.push_back(),
    // drop the lock, and call FileWriter::process_frame().

    std::exception_ptr save_error = nullptr;
    bool in_ssd_queue = false;
    bool in_nfs_queue = false;
    bool on_ssd = false;
    long nfs_count = 0;
};


// AssembledFrameSet: container class containing (nbeams) AssembledFrames.
// Represents one time chunk, and all beams.
//
// This is a thin aggregate -- all members are immutable after construction
// and there is no internal locking. Default copy/move semantics. Callers
// typically manipulate AssembledFrameSets via shared_ptr (held in the
// allocator's frame_set_queue, in Receiver::curr_frame_sets, etc.).

struct AssembledFrameSet
{
    // Initialized at creation, not protected by lock.
    long nfreq = 0;
    long ntime = 0;
    long nbeams = 0;  // = metadata->beam_ids.size()
    long time_chunk_index = 0;

    // Nonempty pointer, initialized at creation, not protected by lock.
    // Same shared pointer as AssembledFrameAllocator::metadata, so always
    // FREQUENCY-SCRUBBED.
    std::shared_ptr<const XEngineMetadata> metadata;

    // Length nbeams, all pointers nonempty, initialized at creation, not protected by lock.
    std::vector<std::shared_ptr<AssembledFrame>> frames;

    // Defensive consistency check. Throws if any member is inconsistent:
    //   - metadata is null
    //   - nbeams != metadata->beam_ids.size()
    //   - frames.size() != nbeams
    //   - any frames[i] is null
    //   - any frames[i] does not agree with the set on
    //     (metadata, time_chunk_index, nfreq, ntime), or with beam_ids[i]
    // Cheap: O(nbeams). Called by AssembledFrameAllocator after constructing
    // a fresh set.
    void validate() const;

    // Convenience accessor (bounds-checked). Equivalent to frames.at(ibeam).
    const std::shared_ptr<AssembledFrame> &get_frame(long ibeam) const;

    // randomize(): fill every frame in the set with random test data, by calling
    // AssembledFrame::randomize(normalize, gaussian) on each frame in turn.
    // 'normalize' and 'gaussian' are forwarded unchanged; see AssembledFrame::
    // randomize() for their meaning (normalize=false -> arbitrary scales/offsets;
    // normalize=true -> scales/offsets calibrated to each frame's metadata
    // noise_variance).
    //
    // This is the SERIAL (single-threaded) path -- simple, and used by tests.
    // To randomize a stream of sets in PARALLEL (per-beam work distributed over
    // a randomizer-thread pool), use a SimulatedFrameFactory instead.
    //
    // Note that we don't currently include SinglePulse arguments (to simulate
    // FRBs), as we do in AssembledFrame::randomize() above. This is because
    // AssembledFrameSet::randomize() is only called by 'pirate_frb test --net',
    // which doesn't simulate FRBs.
    void randomize(bool normalize, bool gaussian);
};


// AssembledFrameAllocator: Thread-backed class that allocates AssembledFrames,
// handed out one AssembledFrameSet (= nbeams frames for one time chunk) at a
// time via get_frame_set().
//
// A worker thread -- the sole producer of frame sets, in both dummy and
// non-dummy mode -- pre-initializes frames (calls memset to fill with 0x88)
// so that get_frame_set() callers never pay allocation/memset latency. The
// worker pre-initializes at most constants::assembled_frame_allocator_queue_size
// sets ahead of full consumption; in dummy mode (slab_allocator->is_dummy())
// this queue bound is the only limit on memory use, since dummy-mode
// get_slab() never blocks. The worker thread inherits its vcpu affinity from
// the caller of the constructor. Python callers should call the
// AssembledFrameAllocator constructor within a ThreadAffinity context manager.
//
// Entry points (following thread-backed class pattern): initialize_metadata(),
// get_metadata(), initialize_initial_chunk(), wait_for_initial_chunk(),
// get_frame_set(), block_until_low_memory(). These will throw if the
// allocator is stopped, and will call stop() on exception.

struct AssembledFrameAllocator
{
    AssembledFrameAllocator(const std::shared_ptr<SlabAllocator> &slab_allocator,
                            int num_consumers,
                            long time_samples_per_chunk);
    ~AssembledFrameAllocator();

    // Non-copyable, non-movable (required for thread-backed class pattern).
    AssembledFrameAllocator(const AssembledFrameAllocator &) = delete;
    AssembledFrameAllocator &operator=(const AssembledFrameAllocator &) = delete;
    AssembledFrameAllocator(AssembledFrameAllocator &&) = delete;
    AssembledFrameAllocator &operator=(AssembledFrameAllocator &&) = delete;

    long time_samples_per_chunk = 0;   // ctor-constant, safe to read without the lock

    // Lock-synchronized getters for the metadata-derived parameters (nfreq,
    // beam_ids -- PRIVATE members, written by initialize_metadata() under
    // 'lock', so an unsynchronized public read would be a data race / torn
    // vector copy). Used by the pybind11 property bindings and by any C++
    // caller. For the canonical metadata pointer itself, use get_metadata().
    long get_nfreq() const;
    std::vector<long> get_beam_ids() const;

    // Sets the canonical (consensus-across-senders) XEngineMetadata on the
    // allocator. Typically called by each Receiver's reader thread once it
    // has parsed a peer's YAML metadata -- so many calls per allocator are
    // expected, not just one per consumer.
    //
    // metadata.freq_channels: expected MEANINGFUL on input (one specific
    // sender's frequency-channel subset). EXPLICITLY SCRUBBED internally on
    // first init before being stored as the canonical copy -- senders disagree
    // on freq_channels by design.
    //
    // The first call (from any caller) stores the metadata. Subsequent calls
    // run XEngineMetadata::check_sender_consistency() against the canonical
    // copy and throw if the new metadata is inconsistent.
    //
    // Thread-safe.
    //
    // Entry point: throws if stopped, calls stop() on exception.

    void initialize_metadata(const XEngineMetadata &metadata);

    // Returns the shared XEngineMetadata pointer once it has been set
    // (i.e. once any caller has called initialize_metadata()).
    //
    // If blocking=true: blocks until metadata is available.
    // If blocking=false: returns nullptr if metadata is not yet available.
    // A stopped allocator throws in BOTH blocking modes (the stopped check
    // precedes the metadata check).
    //
    // The returned shared_ptr points at the canonical (shared, immutable)
    // metadata. Always FREQUENCY-SCRUBBED (freq_channels is empty) -- see
    // _initialize_metadata() for rationale -- so callers should not look
    // there for the per-sender frequency subset.
    //
    // Entry point: throws if stopped (in both blocking modes).
    std::shared_ptr<const XEngineMetadata> get_metadata(bool blocking);

    // Establishes (on the first call from any caller) the canonical
    // 'initial_time_chunk' for the whole pipeline. Returns the established
    // value: on the first call the return value is target_time_chunk; on
    // subsequent calls target_time_chunk is ignored and the previously-
    // established value is returned.
    //
    // get_frame_set() accepts chunk indices starting at initial_time_chunk.
    // Typically called by each Receiver's reader thread after it has parsed
    // the first per-minichunk header.
    //
    // Thread-safe.
    //
    // Entry point: throws if stopped, calls stop() on exception.
    long initialize_initial_chunk(long target_time_chunk);

    // Blocks until some caller has invoked initialize_initial_chunk(), then
    // returns the established initial_time_chunk.
    //
    // Entry point: throws if stopped.
    long wait_for_initial_chunk();

    // Returns the AssembledFrameSet for the given time_chunk_index, blocking
    // until the allocator's worker thread has created it. The returned set
    // contains one AssembledFrame per beam (in beam_ids order), all sharing
    // this allocator's metadata pointer. All callers requesting the same
    // time_chunk_index receive the same shared_ptr<AssembledFrameSet> (not a
    // copy) -- and therefore the same shared_ptr<AssembledFrame> for each
    // beam inside the set.
    //
    // RECEIPT CONTRACT. The allocator holds a reference to each set until
    // the set has been requested num_consumers times, then drops it
    // ("eviction" -- if all callers have also dropped their references, the
    // underlying frames are deallocated). So every chunk index >=
    // initial_time_chunk must be requested exactly num_consumers times in
    // total: in the intended usage, once by each of num_consumers logical
    // consumers, each requesting consecutive indices starting at
    // initial_time_chunk. A request may run at most one chunk past the
    // newest created set (it then blocks until the worker catches up);
    // requests must not skip a chunk. Receipts are anonymous (the allocator
    // cannot attribute them to callers), so misuse is only partially
    // detectable:
    //   - requesting an already-evicted chunk throws an informative error.
    //     Note that a double-request inflates a receipt count and evicts the
    //     set prematurely, so this throw can surface in a DIFFERENT,
    //     innocent caller;
    //   - a double-request that pushes a set's receipt count PAST
    //     num_consumers (possible when the set is not yet at the queue
    //     front) is caught by an xassert at the offending call;
    //   - a skipped chunk leaves its receipt count forever incomplete, so
    //     the queue jams at capacity and all callers deadlock. Skips past
    //     the creation frontier are caught by an xassert; skips within the
    //     resident queue window are not detectable.
    //
    // Frame memory is allocated from 'slab_allocator', with nbytes =
    // (nfreq * time_samples_per_chunk) / 2 per frame. One frame set
    // corresponds to nbeams slab allocations (done by the worker thread).
    //
    // Entry point: throws if stopped, calls stop() on exception.

    std::shared_ptr<AssembledFrameSet> get_frame_set(long time_chunk_index);

    // Returns the num_consumers constructor argument: the receipt count at
    // which a set is evicted from the internal queue (see get_frame_set()).
    // Constructor-constant, so no lock is needed.
    int get_num_consumers() const { return num_consumers; }

    // Returns the number of "available" frames: pre-initialized frames waiting for their first
    // consumer, plus free slabs in the underlying slab_allocator.
    // If permissive=false (default): throws in dummy mode, if initialize_metadata()
    // has not been called, or (from the slab_allocator) before the worker's first
    // allocation creates the slab pool.
    // If permissive=true: never throws -- returns a best-effort count, with 0 for
    // whatever is not ready yet ('permissive' is forwarded into
    // slab_allocator->num_free_slabs()). Used by the GetStatus RPC, which must
    // work at any point in the server's lifetime.
    long num_free_frames(bool permissive = false) const;

    // Entry point: Block until slab allocator is empty (all slabs in use), AND the number of
    // pre-initialized frames waiting for first consumer is <= nframe_threshold.
    // Throws exception in dummy mode, or if stop() is called from another thread.
    void block_until_low_memory(long nframe_threshold);

    // Returns the total number of frames (same as num_total_slabs() from the underlying slab_allocator).
    // Throws in dummy mode (an AFA-local throw: stops nothing).
    // If blocking=true: blocks until the worker's first allocation creates the
    // slab pool (interruptible by stop()).
    // If blocking=false (default), a call before the pool exists throws -- and
    // NOTE that this throw comes from a SlabAllocator ENTRY POINT, so it STOPS
    // the slab allocator (and thence this allocator, via the worker), per the
    // strict stoppable-class policy. Callers probing before the pipeline is
    // running should pass blocking=true (like FrbServer::_check_frame_pool_size)
    // or use num_free_frames(permissive=true).
    long num_total_frames(bool blocking = false) const;

    // Returns true if in dummy mode.
    bool is_dummy() const { return is_dummy_mode; }

    // Returns true if the underlying memory is ready to serve allocations.
    // Delegates to slab_allocator->is_initialized(), which in turn delegates
    // to bump_allocator->is_initialized(). Use for sanity-checking that
    // wait_until_initialized() has been called on all the async
    // BumpAllocators upstream before kicking off pipeline activity.
    bool is_initialized() const;

    // Stop the allocator. After calling stop(), entry points will throw.
    // If 'e' is non-null, it represents an error; if null, it's normal termination.
    // Thread-safe; first call sets the error.
    void stop(std::exception_ptr e = nullptr) const;

private:
    std::shared_ptr<SlabAllocator> slab_allocator;
    int num_consumers;
    bool is_dummy_mode;  // cached from slab_allocator->is_dummy()
    
    // Stop-pattern state ('mutable' since stop() is const -- see
    // notes/stoppable_class.md). is_stopped/error are protected by 'lock'.
    mutable std::mutex lock;

    // One condition variable per wait-predicate (see the "Concurrency"
    // section of notes/cpp.md):
    //
    // metadata_cv -- waiters: get_metadata(blocking=true) callers, and phase
    //   1 of the worker thread's init gate (predicate: metadata set, or
    //   stopped). Signaled on: the metadata latch in _initialize_metadata()
    //   (notify_all -- one-shot latch), and stop().
    //
    // chunk_cv -- waiters: wait_for_initial_chunk() callers, and phase 2 of
    //   the worker thread's init gate (predicate: initial_chunk_set, or
    //   stopped). Signaled on: the latch in _initialize_initial_chunk()
    //   (notify_all -- one-shot latch), and stop().
    //
    // queue_cv -- waiters: _get_frame_set() callers (predicate: the
    //   requested chunk's set is in frame_set_queue, or stopped). Signaled
    //   on: set push in the worker thread (notify_all -- REQUIRED: every
    //   consumer receives every set once, so one new set can make several
    //   waiters ready; this is not a work-queue handoff), and stop().
    //
    // slot_cv -- waiter: the worker thread (predicate: frame_set_queue size
    //   below constants::assembled_frame_allocator_queue_size, or stopped).
    //   Signaled on: pop_front in _get_frame_set() (notify_one -- sound
    //   because the worker is structurally the only waiter), and stop().
    //
    // lowmem_cv -- waiters: block_until_low_memory() callers (predicate:
    //   num_preinitialized decreased below the caller's threshold, or
    //   stopped). Signaled on: first_unreceived_chunk_index advance in
    //   _get_frame_set() (notify_all -- several waiters with different
    //   thresholds can become ready), and stop().
    mutable std::condition_variable metadata_cv;
    mutable std::condition_variable chunk_cv;
    mutable std::condition_variable queue_cv;
    mutable std::condition_variable slot_cv;
    mutable std::condition_variable lowmem_cv;

    mutable bool is_stopped = false;
    mutable std::exception_ptr error;

    std::thread worker_thread;

    // Metadata-derived parameters and the canonical metadata pointer. All
    // three are written by _initialize_metadata() under 'lock'; private, so
    // no unsynchronized external read is possible (external readers use
    // get_nfreq / get_beam_ids / get_metadata).
    //
    // metadata: set on the first initialize_metadata() call (from any
    // caller); subsequent calls must pass
    // XEngineMetadata::check_sender_consistency against it. Propagated by
    // the worker thread into every AssembledFrame (via
    // AssembledFrame::metadata) and AssembledFrameSet.
    //
    // freq_channels: always FREQUENCY-SCRUBBED (empty). The canonical copy
    // represents consensus across senders; senders disagree on freq_channels
    // by design, so it is explicitly cleared in _initialize_metadata.
    long nfreq = 0;
    std::vector<long> beam_ids;
    std::shared_ptr<const XEngineMetadata> metadata;

    // Metadata-initialization state. Set to true the first time any caller
    // invokes initialize_metadata(); subsequent calls go through the
    // consistency-check path. Protected by 'lock'.
    bool metadata_is_initialized = false;

    // Initial-chunk state. Set to true the first time any caller invokes
    // initialize_initial_chunk(); immutable after that. get_frame_set()
    // accepts chunk indices starting at initial_time_chunk. Both members
    // protected by 'lock'.
    //
    // When initial_chunk_set transitions to true (inside
    // _initialize_initial_chunk), we seed queue_start_chunk_index and
    // first_unreceived_chunk_index to initial_time_chunk -- so all
    // chunk-indexed counters below are only meaningful once
    // initial_chunk_set is true.
    bool initial_chunk_set = false;
    long initial_time_chunk = 0;

    // Queue of sets: (set, num_consumers_received).
    // The worker thread -- the sole producer, in both dummy and non-dummy
    // mode -- pushes pre-initialized sets to the back; it never lets the
    // queue grow beyond constants::assembled_frame_allocator_queue_size
    // sets (this bound is the worker's only throttle in dummy mode, where
    // get_slab() never blocks). Sets are popped from the front when all
    // consumers have received them.
    //
    // The early part of the queue contains sets with (num_consumers_received > 0) that
    // are waiting for subsequent consumers. The later part contains sets with
    // (num_consumers_received == 0) that have been pre-initialized, waiting for their
    // first consumer. The boundary between these regions is at first_unreceived_chunk_index.
    std::deque<std::pair<std::shared_ptr<AssembledFrameSet>, int>> frame_set_queue;
    long queue_start_chunk_index = 0;          // time_chunk_index of first set in queue
    long first_unreceived_chunk_index = 0;     // time_chunk_index of first set not yet received by any consumer

    // Helper for entry points. Caller must hold lock.
    void _throw_if_stopped(const char *method_name);

    // Worker thread functions (sole producer of frame sets, both modes).
    // _worker_main() builds each set with 'lock' dropped (slab acquisition
    // + memset of the nbeams slabs), then stamps its time_chunk_index and
    // pushes it under the lock. Slab-pool sizing assumption: the build
    // holds up to nbeams slabs simultaneously, so the pool must have at
    // least nbeams slabs total; this is already a hard prerequisite for
    // the Receiver's 2-chunk window (which pins 2*nbeams slabs).
    void _worker_main();
    void worker_main();

    // Internal implementations of entry points.
    void _initialize_metadata(const XEngineMetadata &metadata);
    long _initialize_initial_chunk(long target_time_chunk);
    std::shared_ptr<AssembledFrameSet> _get_frame_set(long time_chunk_index);
    void _block_until_low_memory(long nframe_threshold);

public:
    // ------------------------  Slab layout  ------------------------
    //
    // Single source of truth for one AssembledFrame's (single-beam) slab byte
    // layout. The slab holds both arrays:
    //   - scales_offsets: dtype float16, shape (nfreq, mpc, 2), mpc = ntime/256
    //   - data:           dtype int4,    shape (nfreq, ntime)
    // Both array bases are INDIVIDUALLY aligned to
    // constants::bytes_per_gpu_cache_line: scales_offsets at slab offset 0 (the
    // slab base is cache-line aligned by the SlabAllocator), and data at the
    // cache-line-aligned offset align_up(scales_offsets_nbytes, ...).
    //
    // Used by the worker thread (the authoritative allocator) AND by external
    // code that sizes a slab pool for AssembledFrameSets (e.g.
    // pirate_frb.run_fake_xengine). Static -- callable without an instance,
    // since pool sizing happens before the allocator is constructed.
    struct SlabLayout
    {
        long scales_offsets_nbytes;  // bytes of the (nfreq, mpc, 2) float16 array
        long data_offset;            // slab offset of 'data' (cache-line aligned)
        long data_nbytes;            // bytes of the (nfreq, ntime) int4 array
        long slab_nbytes;            // total slab size = data_offset + data_nbytes
    };

    // Compute the slab layout for one AssembledFrame.
    // Throws unless nfreq > 0 and time_samples_per_chunk is a positive multiple of 256.
    static SlabLayout get_layout(long nfreq, long time_samples_per_chunk);

    // Convenience: total backing bytes for one AssembledFrame's slab (one beam),
    // i.e. get_layout(...).slab_nbytes. Python-callable (static) -- lets callers
    // size slab pools without duplicating the byte arithmetic.
    static long slab_nbytes(long nfreq, long time_samples_per_chunk);
};


}  // namespace pirate

#endif // _PIRATE_ASSEMBLED_FRAME_HPP
