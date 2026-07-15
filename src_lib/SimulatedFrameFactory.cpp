#include "../include/pirate/SimulatedFrameFactory.hpp"
#include "../include/pirate/XEngineMetadata.hpp"
#include "../include/pirate/simpulse.hpp"

#include <cmath>
#include <cstring>
#include <memory>
#include <random>
#include <sstream>
#include <algorithm>
#include <stdexcept>
#include <string>

#include <ksgpu/xassert.hpp>
#include <ksgpu/mem_utils.hpp>    // ksgpu::af_uhost
#include <ksgpu/rand_utils.hpp>   // ksgpu::default_rng()

using namespace std;


namespace pirate {
#if 0
}  // editor auto-indent
#endif


// File-scope helper: validate Params and fetch the allocator's (already-set)
// metadata. Called from the constructor's initializer list, so the const member
// 'nbeams' can be initialized directly.
static shared_ptr<const XEngineMetadata>
validate_and_get_metadata(const SimulatedFrameFactory::Params &params)
{
    if (!params.allocator)
        throw runtime_error("SimulatedFrameFactory: params.allocator is null");

    xassert_ge(params.frame_set_queue_size, 1);
    xassert_ge(params.num_randomizer_threads, 1);   // caller-supplied (see run_fake_xengine)

    shared_ptr<const XEngineMetadata> md = params.allocator->get_metadata(/*blocking=*/false);
    if (!md)
        throw runtime_error("SimulatedFrameFactory: allocator has no metadata yet"
                            " (construct the factory after allocator.initialize_metadata())");

    if (params.simulate_frbs) {
        // AssembledFrame::randomize() requires (normalize && gaussian) for pulse injection,
        // so catch a bad combination here rather than per-frame on a worker thread.
        if (!params.normalized || !params.gaussian)
            throw runtime_error("SimulatedFrameFactory: simulate_frbs=true requires"
                                " normalized=true and gaussian=true");

        xassert_gt(params.frb_dm0, 0.0);   // log(DM + frb_dm0) must be defined at DM=0
        xassert_gt(params.frb_max_dm, 0.0);
        xassert_gt(params.frb_max_width_ms, 0.0);
        xassert_gt(params.frb_snr, 0.0);
        xassert_ge(params.frb_gap_sec, 0.0);   // extra inter-FRB padding (0 = none)
        xassert_ge(params.num_frb_simulator_threads, 1);
        xassert_ge(params.single_pulse_queue_size, 1);

        long nsub = long(params.frb_subband_fmin_MHz.size());
        xassert_eq(long(params.frb_subband_fmax_MHz.size()), nsub);
        xassert_ge(nsub, 1L);

        // Each subband must overlap the frequency band (strict inequalities, mirroring the
        // SinglePulse constructor's channel-skip rule) -- otherwise pulse construction would
        // throw later, on an frb_simulator thread.
        xassert(md->zone_freq_edges.size() >= 2);
        double band_lo = md->zone_freq_edges.front();
        double band_hi = md->zone_freq_edges.back();

        for (long n = 0; n < nsub; n++) {
            double lo = params.frb_subband_fmin_MHz[n];
            double hi = params.frb_subband_fmax_MHz[n];
            xassert_lt(lo, hi);
            if ((hi <= band_lo) || (lo >= band_hi)) {
                stringstream ss;
                ss << "SimulatedFrameFactory: frb subband " << n << " = [" << lo << ", " << hi
                   << "] MHz does not overlap the metadata frequency band [" << band_lo
                   << ", " << band_hi << "] MHz";
                throw runtime_error(ss.str());
            }
        }
    }

    return md;
}


SimulatedFrameFactory::SimulatedFrameFactory(const Params &params_) :
    params(params_),
    metadata(validate_and_get_metadata(params_)),   // validates params + returns the (retained) metadata
    nbeams(metadata->get_nbeams()),
    time_samples_per_chunk(params_.allocator->time_samples_per_chunk),
    // Frame time-sample duration in ms (same expression AssembledFrame::randomize() checks pulses against).
    time_sample_ms((double) metadata->dt_ns_per_seq * (double) metadata->seq_per_frb_time_sample / 1.0e6)
{
    // Per-channel calibration arrays for simulated pulses (see header). Copied from the
    // metadata accessors once; every pulse's SinglePulse::Params then shares these buffers.
    if (params.simulate_frbs) {
        vector<double> ce = metadata->get_channel_freq_edges();
        vector<double> cv = metadata->get_channel_variances();

        channel_freq_edges = ksgpu::Array<double> ({long(ce.size())}, ksgpu::af_uhost);
        channel_variances  = ksgpu::Array<double> ({long(cv.size())}, ksgpu::af_uhost);
        memcpy(channel_freq_edges.data, ce.data(), ce.size() * sizeof(double));
        memcpy(channel_variances.data, cv.data(), cv.size() * sizeof(double));
    }

    // Per-beam job state (stays all-null / zero when no pulses are being injected).
    rand_sp.assign(nbeams, nullptr);
    rand_dt_sp.assign(nbeams, 0);

    // The randomizer pool parallelizes the per-beam AssembledFrame::randomize()
    // calls within a set. num_randomizer_threads is caller-supplied (validated
    // >= 1 above); run_fake_xengine sizes it as min(nbeams, num_vcpus/2). The
    // spawned threads (and the producer) inherit the caller's vcpu affinity, so
    // a Python caller must construct inside a ThreadAffinity context.
    long n_frb = params.simulate_frbs ? params.num_frb_simulator_threads : 0;
    randomizer_threads.reserve(params.num_randomizer_threads);
    frb_simulator_threads.reserve(n_frb);

    // Spawn the randomizer pool + frb simulators, then the producer. If any spawn
    // throws, wake and join whatever started, then rethrow. We deliberately do NOT
    // call the full stop() here (which would stop the caller-owned allocator on a
    // construction failure): the producer -- the only thread that touches the
    // allocator -- either never started or is joined below, and the other threads
    // just need is_stopped + a notify to exit.
    try {
        for (long i = 0; i < params.num_randomizer_threads; i++)
            randomizer_threads.push_back(std::thread(&SimulatedFrameFactory::randomizer_main, this));

        for (long i = 0; i < n_frb; i++)
            frb_simulator_threads.push_back(std::thread(&SimulatedFrameFactory::frb_simulator_main, this));

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
        sp_queue_cv.notify_all();
        sp_space_cv.notify_all();

        if (producer_thread.joinable())
            producer_thread.join();
        for (auto &rt : randomizer_threads)
            if (rt.joinable())
                rt.join();
        for (auto &ft : frb_simulator_threads)
            if (ft.joinable())
                ft.join();
        throw;
    }
}


SimulatedFrameFactory::~SimulatedFrameFactory()
{
    this->stop();

    // Join the producer first, then the randomizers, then the frb simulators.
    // The order is a readability convention, not load-bearing: join() only
    // waits (it does not halt a thread), so joining the randomizers first
    // would also complete -- they drain the in-flight job and exit once
    // is_stopped, after which a producer blocked on rand_done_cv wakes and
    // exits. Both orders are deadlock-free.
    if (producer_thread.joinable())
        producer_thread.join();

    for (auto &rt : randomizer_threads)
        if (rt.joinable())
            rt.join();

    for (auto &ft : frb_simulator_threads)
        if (ft.joinable())
            ft.join();
}


void SimulatedFrameFactory::stop(std::exception_ptr e) const
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
    sp_queue_cv.notify_all();     // producer blocked in _pop_pulse()
    sp_space_cv.notify_all();     // frb simulators blocked pushing to a full pulse queue

    // Unblock a producer parked in allocator->get_frame_set() on an exhausted
    // pool -- the factory's own cvs are not enough. MUST be called with 'lock'
    // released (never hold the factory lock while calling into the allocator).
    // 'e' is forwarded (see "Error reporting" in notes/stoppable_class.md), so
    // other allocator users see the root cause rather than a generic message.
    params.allocator->stop(e);
}


shared_ptr<AssembledFrameSet> SimulatedFrameFactory::get_frame_set()
{
    unique_lock<mutex> lk(lock);

    for (;;) {
        // Stop takes precedence over a nonempty ready_queue. On an
        // error-stop, rethrow the root cause; on a clean stop (normal
        // termination), return nullptr -- Python None -- which is the
        // consumer loop's end-of-production signal.
        if (error)
            std::rethrow_exception(error);
        if (is_stopped)
            return nullptr;

        if (!ready_queue.empty()) {
            shared_ptr<AssembledFrameSet> s = std::move(ready_queue.front());
            ready_queue.pop_front();
            space_cv.notify_one();    // wake a producer waiting for queue space
            return s;
        }

        queue_cv.wait(lk);
    }
}


std::vector<SimulatedFrameFactory::Event> SimulatedFrameFactory::pop_events()
{
    lock_guard<mutex> lk(lock);
    std::vector<Event> ret;
    ret.swap(events);   // hand out the recorded events and leave 'events' empty (each popped once)
    return ret;
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
    // Per-beam active-FRB state (see class doc). Producer-thread-local: no other thread touches
    // this vector, so no locking is needed. Each pulse is shift_samples()'d into ABSOLUTE
    // frame-sample coordinates when assigned, so pulse b occupies absolute samples
    // [active_frb[b]->it_start, active_frb[b]->it_end) directly -- no separate offset is kept.
    vector<shared_ptr<simpulse::SinglePulse>> active_frb(nbeams);
    bool first_set = true;

    std::mt19937 &rng = ksgpu::default_rng();
    std::uniform_int_distribution<long> phase_dist(0, time_samples_per_chunk - 1);

    // Extra inter-FRB padding (Params::frb_gap_sec) in whole time samples, added to
    // every pulse's placement target below. A beam only becomes FRB-ready once its
    // previous pulse has entirely passed, so this offset lands as fresh spacing
    // between consecutive FRBs rather than a constant shift of the whole stream.
    const long frb_gap_samples = std::llround(params.frb_gap_sec / (1.0e-3 * time_sample_ms));

    // The producer thread is the allocator's sole consumer (num_consumers == 1)
    // and requests consecutive chunks starting at the initial chunk. The class
    // doc requires the allocator to be initialized before the factory is
    // constructed, so this returns immediately in practice; if it does block
    // and the factory is stopped, factory.stop() stops the allocator too, and
    // the throw lands in producer_main()'s catch handler.
    long tci = params.allocator->wait_for_initial_chunk();

    for (;;) {
        {
            unique_lock<mutex> lk(lock);
            if (is_stopped)
                return;
        }

        // Blocks on slab-pool backpressure; throws if the allocator is stopped.
        shared_ptr<AssembledFrameSet> fset = params.allocator->get_frame_set(tci);

        // Assign a fresh pulse to every FRB-ready beam.
        if (params.simulate_frbs) {
            for (long b = 0; b < nbeams; b++) {
                // Ready iff no active pulse, or the active pulse is entirely in the past (its
                // samples -- now in absolute coords -- all precede the current chunk).
                bool ready = !active_frb[b] || (active_frb[b]->it_end <= tci * time_samples_per_chunk);
                if (!ready)
                    continue;

                active_frb[b] = _pop_pulse();   // blocks while the pulse queue is empty
                if (!active_frb[b])
                    return;                     // stopped

                // Shift the pulse so its EARLIEST sample (it_start) lands at a uniformly random
                // phase within the current chunk. After the shift, the pulse's sample indices ARE
                // absolute frame-sample indices. (delta_it can be negative -- shift_samples
                // handles either sign.)
                long target = tci * time_samples_per_chunk + phase_dist(rng) + frb_gap_samples;
                if (first_set)
                    target += time_samples_per_chunk;  // don't put FRBs in first chunk
                active_frb[b]->shift_samples(target - active_frb[b]->it_start);

                // Record the injection event (drained by pop_events()). Brief lock; the producer
                // holds no other lock here.
                Event ev = _make_event(b, *active_frb[b]);
                {
                    lock_guard<mutex> lk(lock);
                    events.push_back(ev);
                }
            }
        }
        first_set = false;

        // Dispatch per-beam randomization (+ pulse injection) across the pool and
        // block until done. Throws if the factory was stopped (the in-flight job
        // still fully drains first, so 'fset' stays safe to drop).
        _randomize_set(*fset, active_frb, tci);

        // Push to the bounded output queue (block while full).
        unique_lock<mutex> lk(lock);
        while (!is_stopped && (long(ready_queue.size()) >= params.frame_set_queue_size))
            space_cv.wait(lk);
        if (is_stopped)
            return;    // drop 'fset' -> its slabs return to the pool
        ready_queue.push_back(std::move(fset));
        queue_cv.notify_one();
        tci++;
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


void SimulatedFrameFactory::_randomize_set(AssembledFrameSet &fset,
                                           const vector<shared_ptr<simpulse::SinglePulse>> &sp_vec,
                                           long tci)
{
    // Sets from our allocator always have exactly 'nbeams' frames.
    xassert_eq(long(fset.frames.size()), nbeams);
    xassert_eq(long(sp_vec.size()), nbeams);

    unique_lock<mutex> lk(lock);
    _throw_if_stopped("SimulatedFrameFactory::_randomize_set");

    // Single in-flight job (one producer thread).
    xassert(rand_total == 0);
    rand_fset  = &fset;
    rand_next  = 0;
    rand_ndone = 0;
    rand_total = nbeams;

    // Per-beam pulse injection args. The pulse (if any) is in ABSOLUTE frame-sample coordinates
    // (the producer shift_samples()'d it), so dt_sp maps the set's frame-local itime to the
    // absolute sample index: it_sp = it_frame + tci*tspc (see AssembledFrame::randomize). Same
    // for every beam; unused (forced to 0) when sp is null.
    for (long b = 0; b < nbeams; b++) {
        rand_sp[b] = sp_vec[b];
        rand_dt_sp[b] = sp_vec[b] ? (tci * time_samples_per_chunk) : 0;
    }

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
    std::fill(rand_sp.begin(), rand_sp.end(), nullptr);   // drop pulse refs (hygiene; the
                                                          // producer still holds the active pulses)
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

        // Claim a beam. Copy the frame's shared_ptr (and the beam's pulse-injection
        // args) under the lock, so they stay valid while we randomize with the lock
        // released ('fset' itself is kept alive by the blocked _randomize_set()
        // caller until rand_ndone reaches rand_total).
        long b = rand_next++;
        shared_ptr<AssembledFrame> frame = rand_fset->frames[b];
        shared_ptr<const simpulse::SinglePulse> sp = rand_sp[b];
        long dt_sp = rand_dt_sp[b];
        lk.unlock();

        std::exception_ptr ex;
        try {
            frame->randomize(params.normalized, params.gaussian, sp, dt_sp);
        } catch (...) {
            ex = std::current_exception();
        }

        lk.lock();
        // notify_one is sound here: the only rand_done_cv waiter is the single
        // producer blocked in _randomize_set() (single in-flight job, asserted
        // there; single producer thread).
        if (++rand_ndone == rand_total)
            rand_done_cv.notify_one();

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


void SimulatedFrameFactory::_frb_simulator_main()
{
    for (;;) {
        {
            unique_lock<mutex> lk(lock);
            if (is_stopped)
                return;
        }

        // Construct the pulse with the lock RELEASED -- this is the expensive part
        // (one FFT per channel in the subband).
        shared_ptr<simpulse::SinglePulse> sp = _make_random_pulse();

        // Push to the bounded pulse queue (block while full).
        unique_lock<mutex> lk(lock);
        while (!is_stopped && (long(pulse_queue.size()) >= params.single_pulse_queue_size))
            sp_space_cv.wait(lk);
        if (is_stopped)
            return;    // drop the pulse
        pulse_queue.push_back(std::move(sp));
        sp_queue_cv.notify_one();
    }
}


void SimulatedFrameFactory::frb_simulator_main()
{
    try {
        _frb_simulator_main();   // only returns if is_stopped
    } catch (...) {
        stop(std::current_exception());
    }
}


shared_ptr<simpulse::SinglePulse> SimulatedFrameFactory::_make_random_pulse()
{
    std::mt19937 &rng = ksgpu::default_rng();   // per-thread; no locking needed

    // DM in [0, frb_max_dm]: u = log(DM + frb_dm0) is uniform. The max() clamps a
    // possible -epsilon from exp(log(x)) rounding below x (simpulse requires dm >= 0).
    std::uniform_real_distribution<double> dm_dist(std::log(params.frb_dm0),
                                                   std::log(params.frb_max_dm + params.frb_dm0));
    double dm = max(0.0, std::exp(dm_dist(rng)) - params.frb_dm0);

    // Intrinsic width: log-uniform over [wmin, wmax] ms. (If wmax <= dt/3, the range
    // degenerates and the width is just wmax.) NOTE: SinglePulse wants SECONDS.
    double wmax_ms = params.frb_max_width_ms;
    double wmin_ms = min(time_sample_ms / 3.0, wmax_ms);
    double w_ms = wmax_ms;
    if (wmin_ms < wmax_ms) {
        std::uniform_real_distribution<double> w_dist(std::log(wmin_ms), std::log(wmax_ms));
        w_ms = std::exp(w_dist(rng));
    }

    // Subband: uniformly random choice among the configured subbands.
    std::uniform_int_distribution<long> sub_dist(0, long(params.frb_subband_fmin_MHz.size()) - 1);
    long n = sub_dist(rng);

    // Arrival time: just the sub-sample phase, uniform over one time sample. There is NO
    // constraint on where the pulse sits on the simpulse grid -- the producer shift_samples()'s
    // it into absolute frame coordinates (see class doc), so small-DM pulses simply get negative
    // freq_it0 (SinglePulse always allows this; the producer uses it_start/it_end to place it).
    double dt_sec = 1.0e-3 * time_sample_ms;
    std::uniform_real_distribution<double> phase_dist(0.0, dt_sec);

    simpulse::SinglePulse::Params sp;
    sp.dm = dm;
    sp.sm = 0.0;
    sp.intrinsic_width = 1.0e-3 * w_ms;   // ms -> seconds
    sp.spectral_index = 0.0;
    sp.undispersed_arrival_time_sec = phase_dist(rng);
    sp.time_sample_ms = time_sample_ms;
    sp.snr = params.frb_snr;
    sp.freq_edges_MHz = channel_freq_edges;   // stored by reference; shares the factory's buffer
    sp.freq_variances = channel_variances;
    sp.subband_freq_lo_MHz = params.frb_subband_fmin_MHz[n];
    sp.subband_freq_hi_MHz = params.frb_subband_fmax_MHz[n];

    return make_shared<simpulse::SinglePulse>(sp);
}


SimulatedFrameFactory::Event
SimulatedFrameFactory::_make_event(long beam_index, const simpulse::SinglePulse &sp) const
{
    // fpga_timestamp: the pulse's arrival time at the LOWEST frequency of the FULL band
    // (metadata->zone_freq_edges.front(), NOT the pulse's subband), as an FPGA sequence number.
    // The pulse has already been shifted into absolute frame-sample coordinates, so
    // undispersed_arrival_time_sec is the (shifted) absolute arrival: the arrival at freq_lo is at
    //   t = undispersed_arrival_time_sec + dispersion_delay(dm, freq_lo)   [seconds],
    // i.e. absolute frame sample t/dt (dt = 1e-3*time_sample_ms sec). One frame sample spans
    // seq_per_frb_time_sample FPGA seq ticks (frame sample 0 <-> seq 0), so multiplying an absolute
    // frame-sample index by seq_per_frb_time_sample converts it to an fpga timestamp.
    const long seq = metadata->seq_per_frb_time_sample;
    double dm = sp.params.dm;
    double freq_lo = metadata->zone_freq_edges.front();
    double dt_sec = 1.0e-3 * time_sample_ms;
    double t_arrival = sp.params.undispersed_arrival_time_sec + simpulse::dispersion_delay(dm, freq_lo);

    Event ev;
    ev.beam_id             = metadata->beam_ids.at(beam_index);
    ev.fpga_timestamp      = std::llround((t_arrival / dt_sec) * (double) seq);
    ev.dm                  = dm;
    ev.snr                 = sp.params.snr;
    ev.width_ms            = sp.params.intrinsic_width * 1.0e3;
    ev.subband_freq_lo_MHz = sp.params.subband_freq_lo_MHz;
    ev.subband_freq_hi_MHz = sp.params.subband_freq_hi_MHz;
    return ev;
}


shared_ptr<simpulse::SinglePulse> SimulatedFrameFactory::_pop_pulse()
{
    unique_lock<mutex> lk(lock);

    while (!is_stopped && pulse_queue.empty())
        sp_queue_cv.wait(lk);

    if (is_stopped)
        return nullptr;   // producer returns cleanly (stop() has already run)

    shared_ptr<simpulse::SinglePulse> sp = std::move(pulse_queue.front());
    pulse_queue.pop_front();
    sp_space_cv.notify_one();   // wake an frb simulator waiting for queue space
    return sp;
}


}  // namespace pirate
