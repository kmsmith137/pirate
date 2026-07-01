#include "../include/pirate/SimulatedFrameFactory.hpp"
#include "../include/pirate/system_utils.hpp"   // get_thread_affinity()

#include <algorithm>   // std::min
#include <memory>
#include <stdexcept>
#include <string>

#include <ksgpu/xassert.hpp>

using namespace std;


namespace pirate {
#if 0
}  // editor auto-indent
#endif


// File-scope helper: validate Params and fetch the allocator's (already-set)
// metadata. Called from the constructor's initializer list, so the const members
// 'xmd' / 'nbeams' can be initialized directly.
static shared_ptr<const XEngineMetadata>
validate_and_get_metadata(const SimulatedFrameFactory::Params &params)
{
    if (!params.allocator)
        throw runtime_error("SimulatedFrameFactory: params.allocator is null");

    xassert_ge(params.frame_set_queue_size, 1);

    shared_ptr<const XEngineMetadata> md = params.allocator->get_metadata(/*blocking=*/false);
    if (!md)
        throw runtime_error("SimulatedFrameFactory: allocator has no metadata yet"
                            " (construct the factory after allocator.initialize_metadata())");
    return md;
}


SimulatedFrameFactory::SimulatedFrameFactory(const Params &params) :
    allocator(params.allocator),
    normalized(params.normalized),
    gaussian(params.gaussian),
    frame_set_queue_size(params.frame_set_queue_size),
    xmd(validate_and_get_metadata(params)),
    nbeams(xmd->get_nbeams())
{
    // Size the randomizer pool: min(nbeams, num_vcpus/2), where num_vcpus is the
    // size of THIS (constructor-calling) thread's vcpu affinity. The spawned
    // randomizer threads (and the producer) inherit that same affinity. The /2
    // leaves headroom for the FakeXEngine worker threads that share the vcpus.
    // num_randomizers can legitimately be 0 (e.g. single-vcpu affinity);
    // _randomize_set() handles that by doing the work serially.
    {
        vector<int> vcpu_list = get_thread_affinity();
        num_randomizers = std::min(nbeams, long(vcpu_list.size()) / 2);
        if (num_randomizers < 0)
            num_randomizers = 0;
    }
    randomizer_threads.reserve(num_randomizers);

    // Spawn the randomizer pool, then the producer. If any spawn throws, wake and
    // join whatever started, then rethrow. We deliberately do NOT call the full
    // stop() here (which would stop the caller-owned allocator on a construction
    // failure): the producer -- the only thread that touches the allocator --
    // either never started or is joined below, and the randomizers just need
    // is_stopped + a notify to exit.
    try {
        for (long i = 0; i < num_randomizers; i++)
            randomizer_threads.push_back(std::thread(&SimulatedFrameFactory::randomizer_main, this));

        producer_thread = std::thread(&SimulatedFrameFactory::producer_main, this);
    } catch (...) {
        {
            lock_guard<mutex> lk(lock);
            is_stopped = true;
        }
        rand_cv.notify_all();
        rand_done_cv.notify_all();
        queue_cv.notify_all();
        space_cv.notify_all();

        if (producer_thread.joinable())
            producer_thread.join();
        for (auto &rt : randomizer_threads)
            if (rt.joinable())
                rt.join();
        throw;
    }
}


SimulatedFrameFactory::~SimulatedFrameFactory()
{
    this->stop();

    // Join the producer BEFORE the randomizers: a producer blocked mid-job (in
    // _randomize_set, on rand_done_cv) can only exit once the randomizers finish
    // draining the in-flight job, so they must still be running here.
    if (producer_thread.joinable())
        producer_thread.join();

    for (auto &rt : randomizer_threads)
        if (rt.joinable())
            rt.join();
}


void SimulatedFrameFactory::stop(std::exception_ptr e)
{
    {
        lock_guard<mutex> lk(lock);
        if (is_stopped)
            return;               // first caller wins; idempotent
        is_stopped = true;
        error = e;
    }

    // Wake every waiter. Setting is_stopped under the lock (above) before these
    // notifies means no wakeup can be lost (waiters re-check the predicate).
    queue_cv.notify_all();        // consumer in get_frame_set()
    space_cv.notify_all();        // producer blocked pushing to a full queue
    rand_cv.notify_all();         // idle randomizers
    rand_done_cv.notify_all();    // producer mid-randomize

    // Unblock a producer parked in allocator->get_frame_set() on an exhausted
    // pool -- the factory's own cvs are not enough. MUST be called with 'lock'
    // released (never hold the factory lock while calling into the allocator).
    // No argument -> normal termination for the allocator.
    allocator->stop();
}


shared_ptr<AssembledFrameSet> SimulatedFrameFactory::get_frame_set()
{
    unique_lock<mutex> lk(lock);

    for (;;) {
        if (is_stopped)
            _throw_if_stopped("SimulatedFrameFactory::get_frame_set");

        if (!ready_queue.empty()) {
            shared_ptr<AssembledFrameSet> s = std::move(ready_queue.front());
            ready_queue.pop_front();
            space_cv.notify_one();    // wake a producer waiting for queue space
            return s;
        }

        queue_cv.wait(lk);
    }
}


void SimulatedFrameFactory::_throw_if_stopped(const char *method_name)
{
    // Caller must hold 'lock'.
    if (error)
        std::rethrow_exception(error);

    if (is_stopped)
        throw runtime_error(string(method_name) + " called on stopped instance");
}


void SimulatedFrameFactory::_producer_main()
{
    for (;;) {
        {
            unique_lock<mutex> lk(lock);
            if (is_stopped)
                return;
        }

        // Blocks on slab-pool backpressure; throws if the allocator is stopped.
        shared_ptr<AssembledFrameSet> fset = allocator->get_frame_set(consumer_id);

        // Dispatch per-beam randomization across the pool and block until done.
        // Throws if the factory was stopped (the in-flight job still fully drains
        // first, so 'fset' stays safe to drop).
        _randomize_set(*fset);

        // Push to the bounded output queue (block while full).
        unique_lock<mutex> lk(lock);
        while (!is_stopped && (long(ready_queue.size()) >= frame_set_queue_size))
            space_cv.wait(lk);
        if (is_stopped)
            return;    // drop 'fset' -> its slabs return to the pool
        ready_queue.push_back(std::move(fset));
        queue_cv.notify_one();
    }
}


void SimulatedFrameFactory::producer_main()
{
    try {
        _producer_main();   // only returns if is_stopped
    } catch (...) {
        stop(std::current_exception());
    }
}


void SimulatedFrameFactory::_randomize_set(AssembledFrameSet &fset)
{
    // Sets from our allocator always have exactly 'nbeams' frames.
    xassert_eq(long(fset.frames.size()), nbeams);

    // Degenerate pool (e.g. single-vcpu affinity): honor the stop contract, then
    // randomize serially in this (producer) thread.
    if (num_randomizers == 0) {
        {
            lock_guard<mutex> lk(lock);
            _throw_if_stopped("SimulatedFrameFactory::_randomize_set");
        }
        fset.randomize(normalized ? xmd : nullptr, gaussian);
        return;
    }

    unique_lock<mutex> lk(lock);
    _throw_if_stopped("SimulatedFrameFactory::_randomize_set");

    // Single in-flight job (one producer thread).
    xassert(rand_total == 0);
    rand_fset  = &fset;
    rand_next  = 0;
    rand_ndone = 0;
    rand_total = nbeams;
    rand_cv.notify_all();

    // Block until every beam has been randomized. NOT stop-sensitive on purpose:
    // a published job always drains to completion (randomize() never blocks, and
    // randomizers keep draining even after stop()), so rand_ndone is guaranteed
    // to reach rand_total and no randomizer is still touching 'fset' when we
    // return -- exactly the condition that makes 'fset' safe to release.
    while (rand_ndone < rand_total)
        rand_done_cv.wait(lk);

    bool stopped = is_stopped;
    std::exception_ptr err = error;
    rand_fset  = nullptr;
    rand_total = 0;
    rand_next  = 0;
    rand_ndone = 0;
    lk.unlock();

    // If stop() raced in during the job, surface it -- but only AFTER the job
    // fully drained above, so 'fset' is safe to release regardless.
    if (stopped) {
        if (err)
            std::rethrow_exception(err);
        throw runtime_error("SimulatedFrameFactory::_randomize_set called on stopped instance");
    }
}


void SimulatedFrameFactory::_randomizer_main()
{
    unique_lock<mutex> lk(lock);

    for (;;) {
        // Wait for a job (rand_next < rand_total) or for stop().
        while (!is_stopped && (rand_next >= rand_total))
            rand_cv.wait(lk);

        // No claimable beam and we woke -> stopped. (We never abandon a published
        // job: while rand_next < rand_total there is work to drain, even after
        // stop().)
        if (rand_next >= rand_total)
            return;

        // Claim a beam. Copy the frame's shared_ptr under the lock so it stays
        // alive while we randomize with the lock released ('fset' itself is kept
        // alive by the blocked _randomize_set() caller until rand_ndone reaches
        // rand_total).
        long b = rand_next++;
        shared_ptr<AssembledFrame> frame = rand_fset->frames[b];
        lk.unlock();

        std::exception_ptr ex;
        try {
            frame->randomize(normalized ? xmd : nullptr, gaussian);
        } catch (...) {
            ex = std::current_exception();
        }

        lk.lock();
        if (++rand_ndone == rand_total)
            rand_done_cv.notify_all();

        if (ex) {
            lk.unlock();
            stop(ex);     // idempotent; wakes the producer
            lk.lock();
        }
    }
}


void SimulatedFrameFactory::randomizer_main()
{
    try {
        _randomizer_main();
    } catch (...) {
        // _randomizer_main() catches per-beam randomize() exceptions itself;
        // reaching here means something unexpected (e.g. a bad_alloc from the
        // deque/cv machinery). Propagate via stop().
        stop(std::current_exception());
    }
}


}  // namespace pirate
