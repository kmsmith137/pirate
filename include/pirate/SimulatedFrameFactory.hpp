#ifndef _PIRATE_SIMULATED_FRAME_FACTORY_HPP
#define _PIRATE_SIMULATED_FRAME_FACTORY_HPP

#include <deque>
#include <memory>
#include <mutex>
#include <thread>
#include <vector>
#include <exception>
#include <condition_variable>

#include "AssembledFrame.hpp"   // AssembledFrameSet, AssembledFrameAllocator


namespace pirate {
#if 0
}   // pacify editor auto-indent
#endif


// SimulatedFrameFactory: a thread-backed class that hands an external consumer a
// stream of PRE-RANDOMIZED AssembledFrameSets, staying a few frames ahead so the
// consumer never blocks on randomization.
//
// This consolidates functionality that used to be split between the Python
// "frame-provider" thread in run_fake_xengine.py and the randomizer-thread pool
// inside 'struct FakeXEngine'. The factory owns:
//
//   - one "producer" thread, which loops: pull one set from the allocator
//     (blocks on slab-pool backpressure), randomize it (dispatching per-beam
//     work to the randomizer pool), then push it to the bounded output queue
//     (blocks when the queue is full);
//
//   - a small pool of "randomizer" threads that parallelize the per-beam
//     AssembledFrame::randomize() calls for the set currently in flight; and
//
//   - a bounded output queue (depth = Params::frame_set_queue_size) drained by
//     the consumer via get_frame_set().
//
// Exactly one set is in flight at a time, so sets are handed out in allocator
// order (FIFO). Randomization is parallel WITHIN a set, but sets are processed
// one at a time -- there is no reordering.
//
// Backpressure: there are two points, both stop()-interruptible and neither able
// to deadlock (progress relies on the consumer draining the queue and on the
// FakeXEngine workers releasing sets as they send). (1) The output queue is
// bounded, so the producer blocks once it is 'frame_set_queue_size' ahead.
// (2) The slab pool: the producer blocks in allocator->get_frame_set() once the
// pool is exhausted. The pool (sized by the caller) should be large enough to
// hold frame_set_queue_size + the set being randomized + the sets the consumer
// pins, or the pool -- not the queue -- becomes the effective lookahead limit.
//
// Ownership / preconditions:
//   - The factory is the SOLE consumer of Params::allocator (constructed with
//     num_consumers == 1) and propagates stop() to it. Because nothing else
//     consumes that allocator, stopping it on shutdown is safe.
//   - The allocator MUST already be initialized (initialize_metadata() and
//     initialize_initial_chunk()) before the factory is constructed. The
//     constructor reads the metadata (for nbeams) via get_metadata(blocking=false)
//     and throws if it is not yet available. (Per-frame calibration reads each
//     frame's own metadata, so the factory need not hold the metadata itself.)
//   - vcpu affinity: the constructor spawns the producer + randomizer threads,
//     which inherit the caller's affinity. Python callers MUST construct the
//     factory inside a ThreadAffinity context manager.
//
// Follows the "thread-backed class" pattern (see notes/thread_backed_class.md):
// stop() puts the object in a stopped state and wakes all threads; entry points
// throw in the stopped state; the destructor calls stop() and joins.

struct SimulatedFrameFactory
{
    struct Params
    {
        // Source of AssembledFrameSets. Must be non-null, and must already have
        // been initialized (see class doc). num_consumers is expected to be 1
        // (the factory is the sole consumer, at consumer_id 0).
        std::shared_ptr<AssembledFrameAllocator> allocator;

        // If true, each frame's scales/offsets are calibrated to the per-zone
        // noise variance in that frame's metadata (AssembledFrame::randomize is
        // called with normalize=true); if false they get arbitrary (uniform-junk)
        // values (normalize=false). See AssembledFrame::randomize.
        bool normalized = true;

        // If true, the int4 data is simulated Gaussian noise clamped to [-7,+7]
        // (avx2_simulate_4bit_noise); if false it is uniform over [-8,+7]. See
        // AssembledFrame::randomize.
        bool gaussian = true;

        // Depth of the bounded output queue -- i.e. how many randomized sets the
        // producer may stay ahead of the consumer. Must be >= 1.
        long frame_set_queue_size = 4;
    };

    // ----- Params-derived, constant after construction -----

    const std::shared_ptr<AssembledFrameAllocator> allocator;
    const bool normalized;
    const bool gaussian;
    const long frame_set_queue_size;

    // The factory is the sole consumer of 'allocator'.
    static constexpr int consumer_id = 0;

    // Beam count, read from the allocator's metadata at construction (the
    // allocator must already have been initialized).
    const long nbeams;

    // ----- Synchronization (all members below protected by 'lock') -----

    mutable std::mutex lock;
    std::condition_variable queue_cv;      // consumer waits: ready_queue non-empty or stopped
    std::condition_variable space_cv;      // producer waits: ready_queue not full or stopped
    std::condition_variable rand_cv;       // randomizers wait: a job is available or stopped
    std::condition_variable rand_done_cv;  // producer waits: current randomize job complete

    bool is_stopped = false;
    std::exception_ptr error;

    // Bounded output queue (producer -> consumer). Bounded at frame_set_queue_size
    // by the producer; the slab pool is the secondary limiter (see class doc).
    std::deque<std::shared_ptr<AssembledFrameSet>> ready_queue;

    // ----- Randomizer-pool job state (single in-flight job) -----
    //
    // Job model (identical to the pool that used to live in FakeXEngine, but
    // guarded by 'lock' rather than a separate rand_lock):
    //   - _randomize_set() publishes a job (rand_fset + rand_total = nbeams,
    //     rand_next/rand_ndone = 0, notify rand_cv), then blocks on rand_done_cv
    //     until rand_ndone == rand_total.
    //   - each randomizer claims a beam by post-incrementing rand_next
    //     (work-stealing), randomizes it with 'lock' RELEASED, then re-acquires
    //     and increments rand_ndone.
    //
    // Lifetime invariant: rand_ndone == rand_total implies NO randomizer is still
    // dereferencing rand_fset (each bumps rand_ndone only after randomize()
    // returns) -- so once _randomize_set() returns, the set is safe to hand off /
    // release. A published job ALWAYS runs to completion, even if stop() races in
    // (randomize() never blocks), so the producer's rand_done_cv wait is
    // deliberately NOT stop-sensitive and can never hang.
    AssembledFrameSet *rand_fset = nullptr;
    long rand_total = 0;
    long rand_next  = 0;
    long rand_ndone = 0;

    // ----- Threads -----

    std::thread producer_thread;
    long num_randomizers = 0;                   // = min(nbeams, num_vcpus/2); ctor requires > 0
    std::vector<std::thread> randomizer_threads;

    // ----- Public interface -----

    // Constructor: validates params, reads the (already-initialized) allocator
    // metadata, sizes + spawns the randomizer pool and the producer thread. The
    // spawned threads inherit the caller's vcpu affinity, so Python callers must
    // construct inside a ThreadAffinity context manager. Throws if the allocator
    // is null, frame_set_queue_size < 1, the allocator has no metadata yet, or
    // num_randomizers would be 0 (needs nbeams >= 1 and a >= 2-vcpu affinity).
    explicit SimulatedFrameFactory(const Params &params);

    // Destructor: stop() (which also stops the allocator) then joins the
    // producer and randomizer threads.
    ~SimulatedFrameFactory();

    // Entry point: block until a randomized AssembledFrameSet is available, then
    // return it (removing it from the output queue). Sets are returned in
    // allocator order. Throws if the factory is stopped (rethrows the stored
    // error, or "called on stopped instance"). The pybind11 wrapper releases the
    // GIL.
    std::shared_ptr<AssembledFrameSet> get_frame_set();

    // Put the factory into the stopped state and wake every thread. First caller
    // wins (stores 'e'); later callers return immediately. Also propagates stop()
    // to the allocator (normal termination), which is what unblocks a producer
    // parked in allocator->get_frame_set() on an exhausted pool. Thread-safe;
    // callable from any thread (including the factory's own worker threads on
    // error).
    void stop(std::exception_ptr e = nullptr);

    // ----- Noncopyable, nonmovable -----

    SimulatedFrameFactory(const SimulatedFrameFactory &) = delete;
    SimulatedFrameFactory &operator=(const SimulatedFrameFactory &) = delete;
    SimulatedFrameFactory(SimulatedFrameFactory &&) = delete;
    SimulatedFrameFactory &operator=(SimulatedFrameFactory &&) = delete;

private:
    // Producer thread main + try/catch wrapper (wrapper calls stop() on throw).
    void _producer_main();
    void producer_main();

    // Randomizer-pool thread main + try/catch wrapper.
    void _randomizer_main();
    void randomizer_main();

    // Randomize one set: dispatch its per-beam work across the randomizer pool
    // and block until complete. Throws if the factory is stopped (the in-flight
    // job is still fully drained first, so 'fset' remains safe to release).
    void _randomize_set(AssembledFrameSet &fset);

    // Helper for entry points. Caller must hold 'lock'. Rethrows 'error' if
    // non-null, else throws runtime_error("... called on stopped instance").
    void _throw_if_stopped(const char *method_name);
};


}  // namespace pirate

#endif // _PIRATE_SIMULATED_FRAME_FACTORY_HPP
