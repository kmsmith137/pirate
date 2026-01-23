#include "../include/pirate/AssembledFrame.hpp"
#include "../include/pirate/file_utils.hpp"   // FileDeleteGuard
#include "../include/pirate/inlines.hpp"      // xdiv()

#include <ksgpu/xassert.hpp>
#include <ksgpu/mem_utils.hpp>
#include <ksgpu/rand_utils.hpp>    // rand_int()

#include <asdf/asdf.hxx>

#include <map>
#include <sstream>
#include <stdexcept>

using namespace std;
using namespace ksgpu;

namespace pirate {
#if 0
}  // editor auto-indent
#endif


// -------------------------------------------------------------------------------------------------
//
// AssembledFrame::write_asdf()


void AssembledFrame::write_asdf(const std::string &filename) const
{
    xassert(nfreq > 0);
    xassert(ntime > 0);
    xassert((ntime % 2) == 0);
    
    // Verify data array is valid and contiguous.
    xassert(data.data != nullptr);
    xassert(data.ndim == 2);
    xassert(data.shape[0] == nfreq);
    xassert(data.shape[1] == ntime);
    xassert(data.is_fully_contiguous());

    // Create ASDF group with metadata and array.
    auto grp = make_shared<ASDF::group>();
    
    // Add scalar metadata.
    grp->emplace("nfreq", make_shared<ASDF::int_entry>(int64_t(nfreq)));
    grp->emplace("ntime", make_shared<ASDF::int_entry>(int64_t(ntime)));
    grp->emplace("beam_id", make_shared<ASDF::int_entry>(int64_t(beam_id)));
    grp->emplace("time_chunk_index", make_shared<ASDF::int_entry>(int64_t(time_chunk_index)));
    
    // Create ndarray for data.
    // int4 dtype (4 bits per element) is stored as uint8 with shape (nfreq, ntime/2).
    // Use ptr_block_t to avoid copying data.
    long nbytes = nfreq * (ntime / 2);
    auto block = make_shared<ASDF::ptr_block_t>(data.data, nbytes);
    auto mblock = ASDF::make_constant_memoized(shared_ptr<ASDF::block_t>(block));
    
    auto arr = make_shared<ASDF::ndarray>(
        mblock,
        std::optional<ASDF::block_info_t>(),
        ASDF::block_format_t::block,
        ASDF::compression_t::none,
        0,  // compression_level
        vector<bool>(),  // mask
        make_shared<ASDF::datatype_t>(ASDF::id_uint8),
        ASDF::host_byteorder(),
        vector<int64_t>{nfreq, ntime/2}
    );
    grp->emplace("data", arr);
    
    // Write to file.
    auto project = make_shared<ASDF::asdf>(map<string, string>(), grp);
    project->write(filename);
}


// -------------------------------------------------------------------------------------------------
//
// AssembledFrame::from_asdf()


shared_ptr<AssembledFrame> AssembledFrame::from_asdf(const std::string &filename)
{
    // Read ASDF file.
    ASDF::asdf project(filename);
    auto grp = project.get_group();
    xassert(grp != nullptr);
    
    // Read scalar metadata.
    auto nfreq_entry = grp->at("nfreq");
    auto ntime_entry = grp->at("ntime");
    auto beam_id_entry = grp->at("beam_id");
    auto time_chunk_index_entry = grp->at("time_chunk_index");
    
    auto nfreq_opt = nfreq_entry->get_maybe_int();
    auto ntime_opt = ntime_entry->get_maybe_int();
    auto beam_id_opt = beam_id_entry->get_maybe_int();
    auto time_chunk_index_opt = time_chunk_index_entry->get_maybe_int();
    
    xassert(nfreq_opt.has_value());
    xassert(ntime_opt.has_value());
    xassert(beam_id_opt.has_value());
    xassert(time_chunk_index_opt.has_value());
    
    long nfreq = nfreq_opt.value();
    long ntime = ntime_opt.value();
    long beam_id = beam_id_opt.value();
    long time_chunk_index = time_chunk_index_opt.value();
    
    xassert(nfreq > 0);
    xassert(ntime > 0);
    xassert((ntime % 2) == 0);
    
    // Read data array.
    auto data_entry = grp->at("data");
    auto arr = data_entry->get_maybe_ndarray();
    xassert(arr != nullptr);
    
    // Verify shape: uint8 with shape (nfreq, ntime/2).
    auto shape = arr->get_shape();
    xassert(shape.size() == 2);
    xassert(shape[0] == nfreq);
    xassert(shape[1] == ntime/2);
    
    // Get data pointer.
    auto mdata = arr->get_data();
    const void *src_ptr = mdata->ptr();
    size_t nbytes = mdata->nbytes();
    xassert(nbytes == (size_t)(nfreq * (ntime / 2)));
    
    // Allocate AssembledFrame with host memory.
    auto frame = make_shared<AssembledFrame>();
    frame->nfreq = nfreq;
    frame->ntime = ntime;
    frame->beam_id = beam_id;
    frame->time_chunk_index = time_chunk_index;
    frame->data = Array<void>(Dtype(df_int, 4), {nfreq, ntime}, af_rhost);
    
    // Copy data from ASDF file.
    memcpy(frame->data.data, src_ptr, nbytes);
    
    return frame;
}


// -------------------------------------------------------------------------------------------------
//
// AssembledFrame::make_random()


// Static member function.
shared_ptr<AssembledFrame> AssembledFrame::make_random(long nfreq, long ntime, long beam_id, long time_chunk_index)
{
    xassert(nfreq > 0);
    xassert(ntime > 0);
    xassert((ntime % 2) == 0);
    
    auto frame = make_shared<AssembledFrame>();
    frame->nfreq = nfreq;
    frame->ntime = ntime;
    frame->beam_id = beam_id;
    frame->time_chunk_index = time_chunk_index;
    frame->data = Array<void>(Dtype(df_int, 4), {nfreq, ntime}, af_rhost);
    
    // Fill data with random bytes.
    // (int4 dtype packs 2 elements per byte, so nbytes = nfreq * ntime / 2)
    long nbytes = nfreq * (ntime / 2);
    char *p = static_cast<char *>(frame->data.data);
    for (long i = 0; i < nbytes; i++)
        p[i] = (char) rand_int(0, 256);
    
    return frame;
}


// Static member function.
shared_ptr<AssembledFrame> AssembledFrame::make_random()
{
    // Random parameters (ntime must be even).
    long nfreq = rand_int(10, 200);
    long ntime = 2 * rand_int(10, 200);
    long beam_id = rand_int(0, 1000);
    long time_chunk_index = rand_int(0, 1000);
    
    return make_random(nfreq, ntime, beam_id, time_chunk_index);
}


// -------------------------------------------------------------------------------------------------
//
// AssembledFrame::test_asdf()


// Static member function.
void AssembledFrame::test_asdf()
{
    cout << "AssembledFrame::test_asdf()..." << endl;
    
    auto frame1 = AssembledFrame::make_random();
    long nfreq = frame1->nfreq;
    long ntime = frame1->ntime;
    long nbytes = nfreq * (ntime / 2);
    
    cout << "  nfreq=" << nfreq << ", ntime=" << ntime
         << ", beam_id=" << frame1->beam_id << ", time_chunk_index=" << frame1->time_chunk_index << endl;
    
    // Generate random filename in /dev/shm.
    string filename = "/dev/shm/test_assembled_frame_" + make_random_hex_string(8) + ".asdf";
    cout << "  filename=" << filename << endl;
    
    // UnlinkGuard ensures temp file gets cleaned up (no commit()).
    UnlinkGuard guard(filename, /*exist_ok=*/ true);
    
    // Write to ASDF file.
    frame1->write_asdf(filename);
    
    // Read back from ASDF file.
    auto frame2 = AssembledFrame::from_asdf(filename);
    
    // Verify metadata matches.
    xassert_eq(frame2->nfreq, nfreq);
    xassert_eq(frame2->ntime, ntime);
    xassert_eq(frame2->beam_id, frame1->beam_id);
    xassert_eq(frame2->time_chunk_index, frame1->time_chunk_index);
    
    // Verify data matches.
    xassert(frame2->data.data != nullptr);
    xassert_eq(frame2->data.ndim, 2);
    xassert_eq(frame2->data.shape[0], nfreq);
    xassert_eq(frame2->data.shape[1], ntime);
    xassert(frame2->data.is_fully_contiguous());
    
    const char *p1 = static_cast<const char *>(frame1->data.data);
    const char *p2 = static_cast<const char *>(frame2->data.data);
    
    for (long i = 0; i < nbytes; i++) {
        if (p1[i] != p2[i]) {
            stringstream ss;
            ss << "AssembledFrame::test_asdf(): data mismatch at byte " << i
               << ": expected " << (int)(unsigned char)p1[i]
               << ", got " << (int)(unsigned char)p2[i];
            throw runtime_error(ss.str());
        }
    }
    
    cout << "AssembledFrame::test_asdf() passed!" << endl;
}


// -------------------------------------------------------------------------------------------------
//
// Constructor


AssembledFrameAllocator::AssembledFrameAllocator(const shared_ptr<SlabAllocator> &slab_allocator_, int num_consumers_)
    : slab_allocator(slab_allocator_), num_consumers(num_consumers_), is_dummy_mode(slab_allocator_->is_dummy())
{
    xassert(slab_allocator);
    xassert_gt(num_consumers, 0);

    if (!ksgpu::af_on_host(slab_allocator->aflags))
        throw runtime_error("AssembledFrameAllocator: slab_allocator must be on host");
    
    is_initialized.resize(num_consumers, false);
    consumer_next_index.resize(num_consumers, 0);
    
    // Spawn worker thread if not in dummy mode.
    if (!is_dummy_mode)
        worker_thread = thread(&AssembledFrameAllocator::worker_main, this);
}


// -------------------------------------------------------------------------------------------------
//
// Destructor


AssembledFrameAllocator::~AssembledFrameAllocator()
{
    this->stop();
    if (worker_thread.joinable())
        worker_thread.join();
}


// -------------------------------------------------------------------------------------------------
//
// stop()


void AssembledFrameAllocator::stop(exception_ptr e)
{
    unique_lock<mutex> guard(lock);

    if (is_stopped)
        return;

    error = e;
    is_stopped = true;
    
    guard.unlock();
    cv.notify_all();

    // Propagate stop to SlabAllocator (wakes up worker thread if blocked in get_slab).
    slab_allocator->stop(e);
}


// -------------------------------------------------------------------------------------------------
//
// _throw_if_stopped(): helper for entry points. Caller must hold lock.


void AssembledFrameAllocator::_throw_if_stopped(const char *method_name)
{
    if (error)
        rethrow_exception(error);
    
    if (is_stopped)
        throw runtime_error(string(method_name) + " called on stopped instance");
}


// -------------------------------------------------------------------------------------------------
//
// Worker thread (non-dummy mode only)
//
// The worker thread pre-initializes frames and pushes them directly to frame_queue.
// This reduces latency for callers of get_frame(), since the memset() is already done.


void AssembledFrameAllocator::_worker_main()
{
    unique_lock<mutex> guard(lock);
        
    // Wait until: stopped, or initialization is complete.
    // We need nfreq, time_samples_per_chunk, beam_ids to create frames.
    while (!is_stopped && (num_initialized == 0))
        cv.wait(guard);
        
    // Main loop: create frames and add directly to frame_queue.
    while (!is_stopped)
        _create_frame(guard);
}


void AssembledFrameAllocator::worker_main()
{
    try {
        _worker_main();  // only returns if is_stopped
    } catch (...) {
        stop(current_exception());
    }
}


// -------------------------------------------------------------------------------------------------
//
// Create a new frame and add it to the queue.
// Caller must currently hold lock via 'guard'.
// The lock will be dropped and re-acquired.


void AssembledFrameAllocator::_create_frame(unique_lock<mutex> &guard)
{
    xassert(num_initialized > 0);

    // If two threads are creating a frame simultaneously, then something is wrong.
    xassert(!frame_creation_underway);
    frame_creation_underway = true;

    // Drop lock while allocating.
    guard.unlock();

    // Allocate data using slab allocator.
    // int4 dtype means 4 bits per element, so nbytes = (nfreq * ntime) / 2.
    // Array is initialized to all (-8) values.
    long nbytes = nfreq * xdiv(time_samples_per_chunk, 2);
    shared_ptr<void> slab = slab_allocator->get_slab(nbytes, /*blocking=*/true);
    memset(slab.get(), 0x88, nbytes);

    // Note that frame->{beam_id, time_chunk_index} are initialized later.
    auto frame = make_shared<AssembledFrame>();
    frame->nfreq = nfreq;
    frame->ntime = time_samples_per_chunk;
    
    // Initialize ksgpu::Array<void> manually.
    frame->data.data = slab.get();
    frame->data.ndim = 2;
    frame->data.shape[0] = nfreq;
    frame->data.shape[1] = time_samples_per_chunk;
    frame->data.size = nfreq * time_samples_per_chunk;
    frame->data.strides[0] = time_samples_per_chunk;
    frame->data.strides[1] = 1;
    frame->data.dtype = ksgpu::Dtype(ksgpu::df_int, 4);
    frame->data.aflags = slab_allocator->aflags;
    frame->data.base = slab;
    frame->data.check_invariants("AssembledFrameAllocator::create_frame_unlocked()");
    
    guard.lock();

    // Get seq_index at end of queue, after acquiring the lock 
    // (in case seq_index changed while the lock was dropped).

    long nbeams = beam_ids.size();
    long seq_index = queue_start_index + frame_queue.size();

    frame->time_chunk_index = seq_index / nbeams;
    frame->beam_id = beam_ids[seq_index % nbeams];

    this->frame_queue.push_back({frame, 0});
    this->frame_creation_underway = false;

    cv.notify_all();  // Wake up any waiting get_frame() callers
}


// -------------------------------------------------------------------------------------------------
//
// initialize() - Entry point


void AssembledFrameAllocator::initialize(int consumer_id, long nfreq_, long time_samples_per_chunk_, const vector<long> &beam_ids_)
{
    {
        unique_lock<mutex> guard(lock);
        _throw_if_stopped("AssembledFrameAllocator::initialize");
    }
    
    try {
        _initialize(consumer_id, nfreq_, time_samples_per_chunk_, beam_ids_);
    } catch (...) {
        this->stop(current_exception());
        throw;
    }
}


void AssembledFrameAllocator::_initialize(int consumer_id, long nfreq_, long time_samples_per_chunk_, const vector<long> &beam_ids_)
{
    xassert_ge(consumer_id, 0);
    xassert_lt(consumer_id, num_consumers);
    xassert_gt(nfreq_, 0L);
    xassert_gt(time_samples_per_chunk_, 0L);
    xassert(!beam_ids_.empty());
    
    unique_lock<mutex> guard(lock);
    
    // Check for double initialization
    if (is_initialized[consumer_id]) {
        stringstream ss;
        ss << "AssembledFrameAllocator::initialize(): consumer_id=" << consumer_id << " already initialized";
        throw runtime_error(ss.str());
    }
    
    if (num_initialized == 0) {
        // First consumer: establish parameters
        nfreq = nfreq_;
        time_samples_per_chunk = time_samples_per_chunk_;
        beam_ids = beam_ids_;
    }
    else {
        // Subsequent consumers: validate parameters match
        if (nfreq_ != nfreq) {
            stringstream ss;
            ss << "AssembledFrameAllocator::initialize(): consumer_id=" << consumer_id
               << " has nfreq=" << nfreq_ << ", expected " << nfreq;
            throw runtime_error(ss.str());
        }
        if (time_samples_per_chunk_ != time_samples_per_chunk) {
            stringstream ss;
            ss << "AssembledFrameAllocator::initialize(): consumer_id=" << consumer_id
               << " has time_samples_per_chunk=" << time_samples_per_chunk_
               << ", expected " << time_samples_per_chunk;
            throw runtime_error(ss.str());
        }
        if (beam_ids_ != beam_ids) {
            stringstream ss;
            ss << "AssembledFrameAllocator::initialize(): consumer_id=" << consumer_id
               << " has mismatched beam_ids";
            throw runtime_error(ss.str());
        }
    }
    
    is_initialized[consumer_id] = true;
    num_initialized++;
    
    // Notify worker thread that initialization is complete.
    cv.notify_all();
}


// -------------------------------------------------------------------------------------------------
//
// get_frame() - Entry point


shared_ptr<AssembledFrame> AssembledFrameAllocator::get_frame(int consumer_id)
{
    {
        unique_lock<mutex> guard(lock);
        _throw_if_stopped("AssembledFrameAllocator::get_frame");
    }
    
    try {
        return _get_frame(consumer_id);
    } catch (...) {
        this->stop(current_exception());
        throw;
    }
}


shared_ptr<AssembledFrame> AssembledFrameAllocator::_get_frame(int consumer_id)
{
    xassert_ge(consumer_id, 0);
    xassert_lt(consumer_id, num_consumers);
    
    unique_lock<mutex> guard(lock);
    
    // Check that this consumer has been initialized
    if (!is_initialized[consumer_id]) {
        stringstream ss;
        ss << "AssembledFrameAllocator::get_frame(): consumer_id=" << consumer_id
           << " has not called initialize()";
        throw runtime_error(ss.str());
    }

    for (;;) {
        if (is_stopped)
            _throw_if_stopped("AssembledFrameAllocator::get_frame");

        // Get this consumer's next sequence index.
        // Note that we re-read this data in each iteration of the for-loop,
        // in case values changed while we dropped the lock.

        long seq_index = consumer_next_index[consumer_id];
        long queue_pos = seq_index - queue_start_index; 
        long queue_size = frame_queue.size();

        xassert(queue_pos >= 0);
        xassert(queue_pos <= queue_size);

        if (queue_pos >= queue_size) {
            // Frame is not in queue. If another thread is currently creating a new frame, then
            // wait for that thread to finish. Otherwise, current thread creates a new frame.

            if (!is_dummy_mode || frame_creation_underway)
                cv.wait(guard);
            else
                _create_frame(guard);

            // Back to top of loop, since lock was dropped in either cv.wait() or _create_frame().
            continue;
        }

        // Get frame and increment receive count.
        auto &[frame_ref, num_received] = frame_queue[queue_pos];
        num_received++;
        
        // IMPORTANT: Make a copy of the shared_ptr BEFORE popping from queue!
        // frame_ref becomes a dangling reference after pop_front().
        auto result = frame_ref;
    
        // Increment this consumer's sequence index, and update first_unreceived_index.
        // (first_unreceived_index is the max of all consumer_next_index values.)
        consumer_next_index[consumer_id]++;
        if (consumer_next_index[consumer_id] > first_unreceived_index) {
            first_unreceived_index = consumer_next_index[consumer_id];
            cv.notify_all();  // Wake up block_until_low_memory() if waiting
        }
    
        // Pop frames from front that all consumers have received.
        while (!frame_queue.empty() && (frame_queue.front().second == num_consumers)) {
            frame_queue.pop_front();
            queue_start_index++;
        }
    
        return result;
    }
}


// -------------------------------------------------------------------------------------------------
//
// num_free_frames() and num_total_frames()


long AssembledFrameAllocator::num_free_frames(bool permissive) const
{
    // In dummy mode: throw or return 0. (is_dummy_mode is set once in constructor, safe without lock.)
    if (is_dummy_mode) {
        if (permissive)
            return 0;
        throw runtime_error("AssembledFrameAllocator::num_free_frames(): not available in dummy mode");
    }

    // Check num_initialized and compute num_preinitialized under lock.
    unique_lock<mutex> guard(lock);

    if (num_initialized == 0) {
        if (permissive)
            return 0;
        throw runtime_error("AssembledFrameAllocator::num_free_frames(): allocator not initialized");
    }

    long queue_end_index = queue_start_index + frame_queue.size();
    long num_preinitialized = queue_end_index - first_unreceived_index;
    guard.unlock();    

    return num_preinitialized + slab_allocator->num_free_slabs();
}


long AssembledFrameAllocator::num_total_frames(bool blocking) const
{
    return slab_allocator->num_total_slabs(blocking);
}


// -------------------------------------------------------------------------------------------------
//
// block_until_low_memory() - Entry point


void AssembledFrameAllocator::block_until_low_memory(long nframe_threshold)
{
    try {
        _block_until_low_memory(nframe_threshold);
    } catch (...) {
        this->stop(current_exception());
        throw;
    }
}


void AssembledFrameAllocator::_block_until_low_memory(long nframe_threshold)
{
    unique_lock<mutex> guard(lock);

    for (;;) {
        _throw_if_stopped("AssembledFrameAllocator::block_until_low_memory");
        guard.unlock();

        // Block until slab allocator is empty (all slabs allocated).
        slab_allocator->block_until_empty();
        
        // Check num_preinitialized under lock.
        guard.lock();
        _throw_if_stopped("AssembledFrameAllocator::block_until_low_memory");
        
        long queue_end_index = queue_start_index + frame_queue.size();
        long num_preinitialized = queue_end_index - first_unreceived_index;
        
        if (num_preinitialized <= nframe_threshold)
            return;
        
        // Wait for something to change (either num_preinitialized decreases,
        // or a slab is returned and a new frame is created).
        cv.wait(guard);
    }
}


}  // namespace pirate
