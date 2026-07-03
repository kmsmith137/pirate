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
// consumer never blocks on randomization. Optionally (Params::simulate_frbs),
// simulated FRBs are injected into the stream.
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
//     AssembledFrame::randomize() calls for the set currently in flight;
//
//   - a bounded output queue (depth = Params::frame_set_queue_size) drained by
//     the consumer via get_frame_set(); and
//
//   - if Params::simulate_frbs: a pool of "frb_simulator" threads that create
//     random simpulse::SinglePulse objects into a bounded pulse queue (depth =
//     Params::single_pulse_queue_size), consumed by the producer (see below).
//
// Exactly one set is in flight at a time, so sets are handed out in allocator
// order (FIFO). Randomization is parallel WITHIN a set, but sets are processed
// one at a time -- there is no reordering.
//
// FRB simulation model: the producer maintains one "active" pulse per beam
// (producer-thread-local state). No pulses are injected into the first set.
// At the start of each later set (time_chunk_index tci), a beam is "FRB-ready" if
// it has no active pulse, or its active pulse is entirely in the past
// (sp->it_end <= tci * time_samples_per_chunk). Each FRB-ready beam pops a fresh
// pulse from the pulse queue (blocking if empty) and shift_samples() it so its
// earliest sample lands at a uniformly random phase within the current chunk
// (delta_it = tci*tspc + randint(0,tspc) - sp->it_start). After the shift the
// pulse's sample indices ARE absolute frame-sample indices, so it carries its own
// placement. Injection happens inside the per-beam randomize() call:
// frame->randomize(normalized, gaussian, sp, dt_sp) with dt_sp = tci*tspc (mapping
// the set's frame-local itime to the absolute sample index), so successive chunks
// receive successive time-slices of the same pulse until it retires. After warmup,
// every beam always has an active pulse; the per-beam FRB rate is roughly one per
// max(1, pulse extent / tspc) chunks.
//
// Backpressure: there are three points, all stop()-interruptible and none able
// to deadlock (progress relies on the consumer draining the queue and on the
// FakeXEngine workers releasing sets as they send). (1) The output queue is
// bounded, so the producer blocks once it is 'frame_set_queue_size' ahead.
// (2) The slab pool: the producer blocks in allocator->get_frame_set() once the
// pool is exhausted. The pool (sized by the caller) should be large enough to
// hold frame_set_queue_size + the set being randomized + the sets the consumer
// pins, or the pool -- not the queue -- becomes the effective lookahead limit.
// (3) The pulse queue (simulate_frbs only): frb_simulator threads block once it
// holds 'single_pulse_queue_size' pulses; the producer blocks popping it when
// empty. The frb threads depend only on queue space (drained by the producer),
// so no cycle arises.
//
// Ownership / preconditions:
//   - The factory is the SOLE consumer of Params::allocator (constructed with
//     num_consumers == 1) and propagates stop() to it. Because nothing else
//     consumes that allocator, stopping it on shutdown is safe.
//   - The allocator MUST already be initialized (initialize_metadata() and
//     initialize_initial_chunk()) before the factory is constructed. The
//     constructor reads and retains the metadata (for nbeams, the time-sample
//     duration, the channel freq edges + noise variances, and -- for FRB event
//     recording -- beam_ids, the full-band low edge, and the timekeeping
//     fields). (Per-frame calibration still reads each frame's own metadata.)
//   - vcpu affinity: the constructor spawns the producer + randomizer (+ frb
//     simulator) threads, which inherit the caller's affinity. Python callers
//     MUST construct the factory inside a ThreadAffinity context manager.
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

        // Number of randomizer threads in the pool (parallelizes the per-beam
        // AssembledFrame::randomize() calls within a set). Must be >= 1; the
        // default 0 means "unset". run_fake_xengine sizes this as
        // min(nbeams, num_vcpus/2).
        long num_randomizer_threads = 0;

        // If true, simulated FRBs (simpulse::SinglePulse) are injected into the
        // stream -- see the class doc for the per-beam active-pulse model.
        // Requires normalized == true and gaussian == true (a precondition of
        // AssembledFrame::randomize() pulse injection).
        bool simulate_frbs = false;

        // The remaining params are only used (and only validated) if
        // simulate_frbs == true.
        //
        // Random pulse parameters. Each pulse draws:
        //   - DM in [0, frb_max_dm], via u = log(DM + frb_dm0) uniform (so
        //     frb_dm0 sets the scale below which DMs are ~uniform);
        //   - intrinsic width log-uniform over [wmin, frb_max_width_ms] ms,
        //     wmin = min(time_sample_ms/3, frb_max_width_ms);
        //   - a subband uniformly among {[frb_subband_fmin_MHz[n],
        //     frb_subband_fmax_MHz[n]]} (each must overlap the metadata's
        //     frequency band);
        //   - snr = frb_snr (matched-filter SNR over the subband's channels);
        //   - sm = 0 and spectral_index = 0 (for now).
        double frb_dm0 = -1.0;
        double frb_max_dm = -1.0;
        double frb_max_width_ms = -1.0;
        double frb_snr = -1.0;
        std::vector<double> frb_subband_fmin_MHz;  // length num_subbands
        std::vector<double> frb_subband_fmax_MHz;  // length num_subbands

        // Number of frb_simulator threads, and depth of the bounded pulse
        // queue they fill. Both must be >= 1 (defaults 0 mean "unset").
        // Guidance: at a chunk boundary up to nbeams beams can become
        // FRB-ready at once, so single_pulse_queue_size ~ nbeams avoids
        // producer stalls; pulse construction costs ~ (channels in subband)
        // FFTs of size 2*internal_nt, so a few threads suffice unless pulses
        // retire faster than one per beam per few chunks.
        long num_frb_simulator_threads = 0;
        long single_pulse_queue_size = 0;
    };

    // ----- Public interface -----

    // Constructor: validates params, reads the (already-initialized) allocator
    // metadata, and spawns the randomizer pool, the frb_simulator pool (if
    // simulate_frbs), and the producer thread. The spawned threads inherit the
    // caller's vcpu affinity, so Python callers must construct inside a
    // ThreadAffinity context manager. Throws if the allocator is null or has no
    // metadata yet, frame_set_queue_size < 1, num_randomizer_threads < 1, or
    // (when simulate_frbs) any frb_* param is invalid -- see the Params docs.
    explicit SimulatedFrameFactory(const Params &params);

    // Noncopyable, nonmovable
    SimulatedFrameFactory(const SimulatedFrameFactory &) = delete;
    SimulatedFrameFactory &operator=(const SimulatedFrameFactory &) = delete;
    SimulatedFrameFactory(SimulatedFrameFactory &&) = delete;
    SimulatedFrameFactory &operator=(SimulatedFrameFactory &&) = delete;

    // Destructor: stop() (which also stops the allocator) then joins the
    // producer, randomizer, and frb_simulator threads.
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

    struct Event
    {
        long beam_id;
        long fpga_timestamp;  // arrival time at lowest frequency in band, not lowest frequency of frb subband
        double dm;
        double snr;
        double width_ms;
        double subband_freq_lo_MHz;
        double subband_freq_hi_MHz;
    };

    // Returns the recorded FRB-injection events (one per injected FRB) and clears the internal
    // list, so each Event is returned exactly once. Thread-safe (locks); safe to call after stop()
    // to drain any final events. Empty unless Params::simulate_frbs.
    std::vector<Event> pop_events();

    // ----- Synchronization (all members below protected by 'lock') -----

    // ----- Synchronization (all members below protected by 'lock') -----
    
    // Construction parameters (validated in the constructor), immutable after
    // construction.
    const Params params;

    // The factory is the sole consumer of 'params.allocator'.
    static constexpr int consumer_id = 0;

    // ----- Derived from the allocator + its metadata at construction -----

    // The allocator's (already-set) metadata, retained for nbeams and -- when
    // simulate_frbs -- for beam_ids, the full-band low edge, and the timekeeping
    // fields used to compute an injected FRB's fpga_timestamp (see _make_event).
    // Immutable and never reaped (per the allocator invariant).
    const std::shared_ptr<const XEngineMetadata> metadata;

    // Beam count (the allocator must already have been initialized).
    const long nbeams;

    // Copied from allocator->time_samples_per_chunk ("tspc" in comments).
    const long time_samples_per_chunk;

    // Frame time-sample duration, = dt_ns_per_seq * seq_per_frb_time_sample
    // / 1.0e6 (the same expression AssembledFrame::randomize() checks pulses
    // against). Only used if simulate_frbs.
    const double time_sample_ms;

    // Per-channel frequency edges (nfreq+1) and noise variances (nfreq),
    // copied once from the metadata accessors (XEngineMetadata::
    // get_channel_freq_edges / get_channel_variances). Every simulated pulse's
    // SinglePulse::Params references these buffers (ksgpu::Array copies share
    // the refcounted base, which also keeps them alive). Empty unless
    // simulate_frbs.
    ksgpu::Array<double> channel_freq_edges;
    ksgpu::Array<double> channel_variances;

    // ----- Synchronization (all members below protected by 'lock') -----

    mutable std::mutex lock;
    std::condition_variable queue_cv;      // consumer waits: ready_queue non-empty or stopped
    std::condition_variable space_cv;      // producer waits: ready_queue not full or stopped
    std::condition_variable rand_cv;       // randomizers wait: a job is available or stopped
    std::condition_variable rand_done_cv;  // producer waits: current randomize job complete
    std::condition_variable sp_queue_cv;   // producer waits: pulse_queue non-empty or stopped
    std::condition_variable sp_space_cv;   // frb simulators wait: pulse_queue not full or stopped

    bool is_stopped = false;
    std::exception_ptr error;

    // Bounded output queue (producer -> consumer). Bounded at frame_set_queue_size
    // by the producer; the slab pool is the secondary limiter (see class doc).
    std::deque<std::shared_ptr<AssembledFrameSet>> ready_queue;

    // Bounded pulse queue (frb simulators -> producer), depth
    // single_pulse_queue_size. Unused unless simulate_frbs. Non-const: the producer shifts a
    // popped pulse into absolute frame coordinates (shift_samples) before injecting it.
    std::deque<std::shared_ptr<simpulse::SinglePulse>> pulse_queue;

    // FRB-injection events: the producer push_back()s one per injected FRB.
    // Drained by pop_events(). Unused unless simulate_frbs.
    std::vector<Event> events;

    // ----- Randomizer-pool job state (single in-flight job) -----
    //
    // Job model (identical to the pool that used to live in FakeXEngine, but
    // guarded by 'lock' rather than a separate rand_lock):
    //   - _randomize_set() publishes a job (rand_fset + rand_total = nbeams,
    //     rand_next/rand_ndone = 0, per-beam rand_sp/rand_dt_sp, notify
    //     rand_cv), then blocks on rand_done_cv until rand_ndone == rand_total.
    //   - each randomizer claims a beam by post-incrementing rand_next
    //     (work-stealing), copies its (sp, dt_sp) under the lock, randomizes
    //     with 'lock' RELEASED, then re-acquires and increments rand_ndone.
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

    // Per-beam pulse injection args for the in-flight job (length nbeams; all
    // null / zero unless simulate_frbs). Written by _randomize_set() at job
    // publication, read by randomizers at beam-claim time -- both under 'lock'.
    std::vector<std::shared_ptr<const simpulse::SinglePulse>> rand_sp;
    std::vector<long> rand_dt_sp;

    // ----- Threads -----

    std::thread producer_thread;
    std::vector<std::thread> randomizer_threads;      // size = Params::num_randomizer_threads
    std::vector<std::thread> frb_simulator_threads;   // size = Params::num_frb_simulator_threads (0 unless simulate_frbs)

private:
    // Producer thread main + try/catch wrapper (wrapper calls stop() on throw).
    void _producer_main();
    void producer_main();

    // Randomizer-pool thread main + try/catch wrapper.
    void _randomizer_main();
    void randomizer_main();

    // frb_simulator thread main + try/catch wrapper: loop { construct a random
    // pulse (lock released -- construction is expensive), push to the bounded
    // pulse_queue (blocking while full) }.
    void _frb_simulator_main();
    void frb_simulator_main();

    // Construct one random SinglePulse (see the Params doc for the parameter
    // distributions). undispersed_arrival_time_sec is just uniform(0, dt) --
    // the sub-sample phase -- so small-DM pulses have negative freq_it0
    // (SinglePulse always allows this); absolute placement happens when the
    // producer shift_samples() the popped pulse. Called by frb_simulator threads
    // with the lock RELEASED; uses the per-thread ksgpu::default_rng().
    std::shared_ptr<simpulse::SinglePulse> _make_random_pulse();

    // Pop one pulse from pulse_queue, blocking while it is empty. Returns an
    // empty pointer if the factory is stopped (the producer then just returns).
    // Non-const pulse: the producer shift_samples() it before injecting.
    std::shared_ptr<simpulse::SinglePulse> _pop_pulse();

    // Build the Event describing an injected FRB. Called AFTER the pulse has been shifted into
    // absolute frame-sample coordinates (see _producer_main), so its arrival maps directly to an
    // FPGA sequence number. Reads the (immutable) metadata; no lock needed.
    Event _make_event(long beam_index, const simpulse::SinglePulse &sp) const;

    // Randomize one set: dispatch its per-beam work across the randomizer pool and block until
    // complete. sp_vec[b] is the beam's active pulse (already shifted into absolute frame
    // coordinates), or null; the per-beam dt_sp is tci * time_samples_per_chunk (maps the set's
    // frame-local itime to the absolute sample index). Throws if the factory is stopped (the
    // in-flight job is still fully drained first, so 'fset' remains safe to release).
    void _randomize_set(AssembledFrameSet &fset,
                        const std::vector<std::shared_ptr<simpulse::SinglePulse>> &sp_vec,
                        long tci);

    // Helper for entry points. Caller must hold 'lock'. Rethrows 'error' if
    // non-null, else throws runtime_error("... called on stopped instance").
    void _throw_if_stopped(const char *method_name);
};


}  // namespace pirate

#endif // _PIRATE_SIMULATED_FRAME_FACTORY_HPP
