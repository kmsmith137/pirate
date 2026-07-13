#include "../include/pirate/FakeXEngine.hpp"
#include "../include/pirate/network_utils.hpp"
#include "../include/pirate/system_utils.hpp"   // get_thread_affinity()

#include <algorithm>     // std::max (paced-mode bootstrap), std::min (ack-read cap)
#include <cstring>
#include <iostream>
#include <memory>
#include <sstream>
#include <stdexcept>

#include <ksgpu/xassert.hpp>

// gRPC client + generated stubs for the paced-mode pacing thread.
// Kept out of FakeXEngine.hpp -- see the forward-decl comment there.
//
// grpc/protobuf headers pull in conda-forge's libabseil, which was built
// with -DNDEBUG. absl::Mutex::Dtor() is only inlined when NDEBUG is
// defined at the include site; otherwise it becomes an undefined
// external symbol that the abseil DSO does not export, and libpirate.so
// fails to load. So we push_macro NDEBUG on, include grpc, then pop it
// back. See notes/build.md.
#pragma push_macro("NDEBUG")
#ifndef NDEBUG
#  define NDEBUG
#endif
// Silence -Wdeprecated-declarations warnings emitted from grpc's own public
// headers (e.g. IdentityKeyCertPair, set_certificate_provider). They come
// from grpc-internal code that pirate does not call directly.
#pragma GCC diagnostic push
#pragma GCC diagnostic ignored "-Wdeprecated-declarations"
#include "../grpc/frb_search.grpc.pb.h"
#include <grpcpp/grpcpp.h>
#pragma GCC diagnostic pop
#pragma pop_macro("NDEBUG")

using namespace std;


namespace pirate {
#if 0
}  // editor auto-indent
#endif


// File-scope helper: atomically sets dst = max(dst, src). Portable
// C++17 CAS-loop equivalent of C++26 std::atomic::fetch_max. Used by
// the FLAG_ACK ack-prediction tracking (max_sent_minichunk and
// max_acked_minichunk).
//
// Memory ordering: relaxed throughout. These atomics are running
// counters with no synchronization relationship to other state. The
// happens-before relation that makes the ack-prediction bounds hold
// goes THROUGH the network (worker A sends bytes -> server processes
// -> worker B receives ack), which the C++ memory model doesn't
// capture anyway -- any reasonable ordering works in practice.
static inline void fetch_max(std::atomic<long> &dst, long src)
{
    long expected = dst.load(std::memory_order_relaxed);
    while (expected < src) {
        if (dst.compare_exchange_weak(expected, src,
                                      std::memory_order_relaxed,
                                      std::memory_order_relaxed))
            return;
        // CAS failed; `expected` was updated to the current value of dst.
    }
}


// File-scope helper: sets w.is_stopped, w.error, notifies all of w's cvs.
// Idempotent (no-op if w.is_stopped is already true). Used by
// FakeXEngine::stop()'s per-worker sweep; never called from anywhere else.
static void _stop_one_worker(FakeXEngine::Worker &w, std::exception_ptr e)
{
    {
        std::lock_guard<std::mutex> lock(w.mutex);
        if (w.is_stopped) return;
        w.is_stopped = true;
        w.error = e;
    }
    w.cmd_cv.notify_all();
    w.gate_cv.notify_all();
    w.processed_cv.notify_all();
    w.drain_cv.notify_all();
}


FakeXEngine::FakeXEngine(const std::shared_ptr<const XEngineMetadata> &xmd_,
                         const std::vector<std::string> &ip_addrs_,
                         int nworkers_, long time_samples_per_chunk_,
                         bool debug_, bool paced_,
                         const std::string &rpc_address_) :
    xmd(xmd_),
    ip_addrs(ip_addrs_),
    nworkers(nworkers_),
    time_samples_per_chunk(time_samples_per_chunk_),
    debug(debug_),
    paced(paced_),
    rpc_address(rpc_address_),
    nbeams(xmd_ ? xmd_->get_nbeams() : 0),  // null xmd is rejected below; guard avoids null-deref
    minichunks_per_chunk(time_samples_per_chunk_ / 256),
    num_receivers(long(ip_addrs_.size()))
{
    if (!xmd)
        throw runtime_error("FakeXEngine: xmd is null");
    if (ip_addrs.empty())
        throw runtime_error("FakeXEngine: ip_addrs is empty");
    xassert(nworkers > 0);

    // Paced mode requires an rpc_address. Non-paced silently accepts a
    // non-empty rpc_address (and ignores it).
    if (paced && rpc_address.empty()) {
        throw runtime_error(
            "FakeXEngine: paced=true requires a non-empty rpc_address");
    }

    if (time_samples_per_chunk <= 0) {
        stringstream ss;
        ss << "FakeXEngine: time_samples_per_chunk=" << time_samples_per_chunk
           << " must be positive";
        throw runtime_error(ss.str());
    }
    if ((time_samples_per_chunk % 256) != 0) {
        stringstream ss;
        ss << "FakeXEngine: time_samples_per_chunk=" << time_samples_per_chunk
           << " must be a multiple of 256";
        throw runtime_error(ss.str());
    }

    long naddrs = ip_addrs.size();
    if (nworkers % naddrs != 0) {
        stringstream ss;
        ss << "FakeXEngine: nworkers=" << nworkers
           << " is not a multiple of ip_addrs.size()=" << naddrs;
        throw runtime_error(ss.str());
    }

    // Validate all addresses (parse early to catch errors before threads spawn).
    for (const auto &addr : ip_addrs) {
        string ip;
        uint16_t port;
        parse_ip_address(addr, ip, port);
    }

    // Validate that XEngineMetadata has enough frequency channels for all workers.
    long total_nfreq = xmd->get_total_nfreq();
    if (total_nfreq < nworkers) {
        stringstream ss;
        ss << "FakeXEngine: nworkers=" << nworkers
           << " but total frequency channels=" << total_nfreq;
        throw runtime_error(ss.str());
    }

    xmd->validate();

    // Size the per-Receiver ack high-water-mark vector and seed each
    // entry to -1. Can't use std::vector(n, atomic<long>{-1}) because
    // std::atomic is not copyable; have to default-construct then
    // explicitly store.
    max_acked_minichunk = std::vector<std::atomic<long>>(num_receivers);
    for (long i = 0; i < num_receivers; i++)
        max_acked_minichunk[i].store(-1, std::memory_order_relaxed);

    // Zero the debug counters (default-constructed std::atomic has
    // indeterminate initial value pre-C++20).
    for (auto &c : debug_counters)
        c.store(0, std::memory_order_relaxed);

    // Allocate Worker objects first (heap-allocated, stable address --
    // Worker embeds std::mutex / std::condition_variable, which are
    // neither copyable nor movable). Then spawn the worker threads.
    // Workers inherit the vcpu affinity of the caller (the documented
    // constructor contract).
    workers.reserve(nworkers);
    for (int i = 0; i < nworkers; i++)
        workers.push_back(std::make_unique<Worker>());

    // In paced mode, allocate the MonitorRingbuf ClientContext before
    // spawning any threads, so stop() (which TryCancels via this ctx)
    // is safe to call from any of them. The channel + Stub are NOT
    // members -- they live as locals in _pacing_thread_main; see
    // FakeXEngine.hpp for the forward-decl rationale.
    if (paced)
        pacing_ctx = std::make_unique<grpc::ClientContext>();

    try {
        for (int i = 0; i < nworkers; i++)
            workers[i]->worker_thread = std::thread(&FakeXEngine::worker_main, this, i);

        // Spawn the pacing thread after the workers so that any incoming
        // MonitorRingbuf message can immediately broadcast to a fully
        // populated 'workers' vector.
        if (paced)
            pacing_thread = std::thread(&FakeXEngine::pacing_thread_main, this);
    } catch (...) {
        // Partial-spawn cleanup: signal stop so any workers that did start
        // exit promptly, then join whichever ones are joinable. stop()'s
        // atomic CAS ensures only the first call sweeps; the destructor
        // (which also calls stop()) will then be a no-op for the atomic and
        // the per-worker sweep is idempotent.
        stop(std::current_exception());
        for (auto &wp : workers) {
            if (wp->worker_thread.joinable())
                wp->worker_thread.join();
        }
        if (pacing_thread.joinable())
            pacing_thread.join();
        throw;
    }
}


FakeXEngine::~FakeXEngine()
{
    this->stop();

    for (auto &wp : workers) {
        if (wp->worker_thread.joinable())
            wp->worker_thread.join();
    }

    // Join the pacing thread AFTER the workers -- the pacing thread's
    // critical section iterates over 'workers' under each Worker::mutex,
    // and the workers are easier to reason about as "fully stopped"
    // before we let the pacing thread's destructor implicitly run.
    // (Both orders are correct given stop()'s independent signaling of
    // each thread; this join order happens to match the ctor's spawn
    // order -- workers spawned first, joined first.)
    if (pacing_thread.joinable())
        pacing_thread.join();
}


bool FakeXEngine::_stopped_or_rethrow(Worker &w)
{
    // Caller must hold w.mutex. Non-null error = error shutdown (rethrow
    // the root cause); null-stopped = normal termination (the caller
    // returns its "stopped" done-value).
    if (w.error)
        std::rethrow_exception(w.error);

    return w.is_stopped;
}


void FakeXEngine::stop(std::exception_ptr e) const
{
    // Re-entry guard: only the first caller's compare-exchange wins.
    // Subsequent concurrent or later callers see the cache already true
    // and return immediately, so 'e' from the first caller is the one
    // propagated to every Worker::error.
    bool expected = false;
    if (!is_stopped_cache.compare_exchange_strong(expected, true))
        return;

    // Per-worker sweep. Each iteration takes one Worker's mutex briefly
    // to set is_stopped+error and notify its cv. No nested locks; no
    // possibility of deadlock with anything else in this class.
    for (auto &wp : workers)
        _stop_one_worker(*wp, e);

    // Paced mode: cancel the MonitorRingbuf streaming RPC so the
    // pacing thread's blocked Read() returns false. TryCancel is
    // safe to call concurrently with any gRPC operation; it's a
    // no-op if the call has already completed.
    if (paced && pacing_ctx)
        pacing_ctx->TryCancel();
}


vector<long> FakeXEngine::get_worker_freq_channels(long worker_id) const
{
    xassert(worker_id >= 0);
    xassert(worker_id < nworkers);

    long total_nfreq = xmd->get_total_nfreq();

    // Assign frequency channels round-robin to this worker.
    // Worker 'worker_id' gets channels: worker_id, worker_id + nworkers, worker_id + 2*nworkers, ...
    vector<long> freq_channels;
    for (long ch = worker_id; ch < total_nfreq; ch += nworkers)
        freq_channels.push_back(ch);

    return freq_channels;
}


XEngineMetadata FakeXEngine::make_worker_metadata(int worker_id) const
{
    // Copy the master metadata, patch freq_channels for this worker, return.
    XEngineMetadata ret = *xmd;
    ret.freq_channels = get_worker_freq_channels(worker_id);
    return ret;
}


void FakeXEngine::worker_main(int worker_id)
{
    try {
        _worker_main(worker_id);
    } catch (...) {
        stop(std::current_exception());
    }
}


bool FakeXEngine::_send_all(Worker &w, Socket &sock, const void *buf, long nbytes)
{
    using clock = std::chrono::steady_clock;
    static constexpr auto stall_period = std::chrono::seconds(1);

    const char *ptr = static_cast<const char *>(buf);
    long pos = 0;

    // Wall-clock budget for finishing this _send_all (without making
    // any further progress). Reset after each stall-detection print
    // in the non-debug branch below.
    auto stall_deadline = clock::now() + stall_period;

    while (pos < nbytes) {
        // Check this worker's stopped flag periodically (bounded by
        // send_timeout_ms = 10ms granularity). Locks only w.mutex.
        {
            std::lock_guard<std::mutex> lock(w.mutex);
            if (w.is_stopped) return false;
        }

        long n = sock.send_with_timeout(ptr + pos, nbytes - pos, send_timeout_ms);

        if (sock.connreset) {
            // Receiver hung up. The FakeXEngine cannot know WHY the peer
            // closed (operator ended the run vs the server crashed), so it
            // reports the hangup as an error and lets the caller judge --
            // there is deliberately no "benign hangup" category. Error-stop
            // the whole FakeXEngine so sibling workers exit too, and so
            // every entry point reports this root cause. stop()'s atomic
            // CAS makes repeated connreset->stop() calls cheap (only the
            // first one actually sweeps).
            stringstream ss;
            ss << "FakeXEngine: receiver " << w.ip_addr << ":" << w.port
               << " closed connection";
            stop(std::make_exception_ptr(runtime_error(ss.str())));
            return false;
        }

        pos += n;

        // Stall detection. If _send_all has been running for >1s
        // without returning, the receiver is either down or very
        // slow. In debug mode, this is a hard failure (we only run
        // debug=true in controlled subscale unit tests, where a 1-
        // second stall almost certainly means something has
        // deadlocked). In production (debug=false), print a one-line
        // warning to cout and reset the timer so we'll print again
        // after each additional 1s of stuckness.
        if (clock::now() >= stall_deadline) {
            if (debug) {
                std::stringstream ss;
                ss << "FakeXEngine::_send_all: stalled >1s sending to "
                   << w.ip_addr << ":" << w.port
                   << " (sent " << pos << " of " << nbytes
                   << " bytes); treating as deadlock in debug mode";
                throw runtime_error(ss.str());
            }
            // std::endl flushes; we want the warning visible
            // immediately even if cout is fully buffered (e.g. when
            // redirected to a file).
            std::cout << "FakeXEngine: receiver " << w.ip_addr << ":" << w.port
                      << " is down or very slow (sent " << pos << " of "
                      << nbytes << " bytes in 1s)"
                      << std::endl;
            stall_deadline = clock::now() + stall_period;
        }
    }

    return true;
}


// -------------------------------------------------------------------------------------------------
//
// External-thread entry points: enqueue_send_junk(),
// enqueue_skip_minichunk(), enqueue_send_minichunk(),
// enqueue_disconnect(), wait_until_processed(). All but the last
// share an _enqueue() helper that does the worker_id range check +
// lock + stopped-check (false return / error rethrow) + (for
// state-advancing commands) queue-time sequentiality check + push +
// notify pattern. The is_state_advancing argument is passed explicitly
// per call rather than inspected from cmd.kind, so each call site
// documents its intent.


bool FakeXEngine::_enqueue(long worker_id, Command &&cmd, bool is_state_advancing,
                           const char *method_name)
{
    if (worker_id < 0 || worker_id >= long(nworkers)) {
        stringstream ss;
        ss << method_name << ": worker_id=" << worker_id
           << " out of range [0, " << nworkers << ")";
        throw runtime_error(ss.str());
    }

    Worker &w = *workers[worker_id];

    std::unique_lock<std::mutex> lock(w.mutex);
    if (_stopped_or_rethrow(w))
        return false;   // cleanly stopped: command not enqueued

    // Queue-time sequentiality check for state-advancing commands
    // (SEND_JUNK / SKIP_MINICHUNK / SEND_MINICHUNK). DISCONNECT does
    // not participate -- its minichunk_index is unused.
    if (is_state_advancing) {
        if (w.last_queued_minichunk < 0) {
            // First state-advancing command on this worker. Any
            // minichunk_index >= 0 is allowed (supports NOTE-2-style
            // nonzero-initial-chunk tests). The entry-point
            // xassert_ge already enforced cmd.minichunk_index >= 0.
            w.first_minichunk = cmd.minichunk_index;
        } else if (cmd.minichunk_index != w.last_queued_minichunk + 1L) {
            stringstream ss;
            ss << method_name << ": worker_id=" << worker_id
               << " expected minichunk_index="
               << (w.last_queued_minichunk + 1L)
               << " (strict +1 monotonicity), got "
               << cmd.minichunk_index;
            throw runtime_error(ss.str());
        }
        w.last_queued_minichunk = cmd.minichunk_index;

        // Debug-mode back-channel: append STATUS_QUEUED. This
        // push_back may reallocate w.minichunk_status WITH THE LOCK
        // HELD -- normally a performance smell, but acceptable here
        // because debug mode is testing-only and the worker thread
        // isn't on the latency-critical path. After the push, the
        // invariant
        //   minichunk_status.size() == last_queued_minichunk - first_minichunk + 1
        // holds.
        if (debug)
            w.minichunk_status.push_back(STATUS_QUEUED);
    }

    w.command_queue.push_back(std::move(cmd));

    // The lock has been held continuously from is_stopped check
    // through the queue-time check, the counter updates, and the
    // push -- so the recorded last_queued_minichunk is consistent
    // with the FIFO position of this command in command_queue. Now
    // drop it and wake the worker (notify_one: the worker thread is
    // the only cmd_cv waiter).
    lock.unlock();
    w.cmd_cv.notify_one();
    return true;
}


// Note: per the strict stoppable-class policy (notes/stoppable_class.md),
// ANY exception thrown from an entry point stops the FakeXEngine --
// including argument errors (bad worker_id, non-monotonic minichunk_index).

bool FakeXEngine::enqueue_send_junk(long worker_id, long minichunk_index)
{
    try {
        xassert_ge(minichunk_index, 0L);

        Command cmd;
        cmd.kind = Command::Kind::SEND_JUNK;
        cmd.minichunk_index = minichunk_index;
        return _enqueue(worker_id, std::move(cmd), /*is_state_advancing=*/true,
                        "FakeXEngine::enqueue_send_junk");
    } catch (...) {
        stop(std::current_exception());
        throw;
    }
}


bool FakeXEngine::enqueue_skip_minichunk(long worker_id, long minichunk_index)
{
    try {
        xassert_ge(minichunk_index, 0L);

        Command cmd;
        cmd.kind = Command::Kind::SKIP_MINICHUNK;
        cmd.minichunk_index = minichunk_index;
        return _enqueue(worker_id, std::move(cmd), /*is_state_advancing=*/true,
                        "FakeXEngine::enqueue_skip_minichunk");
    } catch (...) {
        stop(std::current_exception());
        throw;
    }
}


bool FakeXEngine::enqueue_send_minichunk(long worker_id, long minichunk_index,
                                         std::shared_ptr<AssembledFrameSet> frame_set)
{
    try {
        xassert_ge(minichunk_index, 0L);
        if (!frame_set)
            throw runtime_error("FakeXEngine::enqueue_send_minichunk: frame_set is null");

        Command cmd;
        cmd.kind = Command::Kind::SEND_MINICHUNK;
        cmd.minichunk_index = minichunk_index;
        cmd.frame_set = std::move(frame_set);
        return _enqueue(worker_id, std::move(cmd), /*is_state_advancing=*/true,
                        "FakeXEngine::enqueue_send_minichunk");
    } catch (...) {
        stop(std::current_exception());
        throw;
    }
}


bool FakeXEngine::enqueue_disconnect(long worker_id)
{
    try {
        Command cmd;
        cmd.kind = Command::Kind::DISCONNECT;
        // minichunk_index stays -1; frame_set stays null. Neither is read
        // by _disconnect.
        return _enqueue(worker_id, std::move(cmd), /*is_state_advancing=*/false,
                        "FakeXEngine::enqueue_disconnect");
    } catch (...) {
        stop(std::current_exception());
        throw;
    }
}


bool FakeXEngine::is_connected(long worker_id) const
{
    if (worker_id < 0 || worker_id >= long(nworkers)) {
        stringstream ss;
        ss << "FakeXEngine::is_connected: worker_id=" << worker_id
           << " out of range [0, " << nworkers << ")";
        throw runtime_error(ss.str());
    }
    // No stopped-check: this is an informational query, and the
    // last-known per-worker connected state is meaningful even after
    // FakeXEngine::stop(). Same semantics as the is_stopped property.
    return workers[worker_id]->connected.load(std::memory_order_relaxed);
}


bool FakeXEngine::wait_until_processed(long worker_id, long minichunk_index)
{
    try {
        if (worker_id < 0 || worker_id >= long(nworkers)) {
            stringstream ss;
            ss << "FakeXEngine::wait_until_processed: worker_id=" << worker_id
               << " out of range [0, " << nworkers << ")";
            throw runtime_error(ss.str());
        }

        Worker &w = *workers[worker_id];

        std::unique_lock<std::mutex> lock(w.mutex);
        for (;;) {
            if (_stopped_or_rethrow(w))
                return false;   // cleanly stopped before the condition was reached
            if (w.last_processed_minichunk >= minichunk_index)
                return true;
            w.processed_cv.wait(lock);
        }
    } catch (...) {
        stop(std::current_exception());
        throw;
    }
}


bool FakeXEngine::enqueue_wait_for_acks(long worker_id)
{
    try {
        if (!debug)
            throw runtime_error(
                "FakeXEngine::enqueue_wait_for_acks: FakeXEngine was not"
                " constructed with debug=true (so there are no acks to wait for)");

        Command cmd;
        cmd.kind = Command::Kind::WAIT_FOR_ACKS;
        return _enqueue(worker_id, std::move(cmd), /*is_state_advancing=*/false,
                        "FakeXEngine::enqueue_wait_for_acks");
    } catch (...) {
        stop(std::current_exception());
        throw;
    }
}


bool FakeXEngine::synchronize(long worker_id)
{
    try {
        if (worker_id < 0 || worker_id >= long(nworkers)) {
            stringstream ss;
            ss << "FakeXEngine::synchronize: worker_id=" << worker_id
               << " out of range [0, " << nworkers << ")";
            throw runtime_error(ss.str());
        }

        // If debug mode is enabled, enqueue a WAIT_FOR_ACKS first. The
        // worker eventually pops it and calls _read_acks(blocking=true);
        // when that finishes the centralized drain_cv notify in
        // _worker_main fires, which wakes us from the wait below. A false
        // return means "cleanly stopped" -- report the same from here.
        if (debug) {
            if (!enqueue_wait_for_acks(worker_id))
                return false;
        }

        // Wait for the queue to drain. Predicate is (command_queue.empty()
        // && ack_queue.empty() && !cmd_in_flight):
        //
        //   - command_queue.empty() is the basic "all commands processed"
        //     barrier. Sensitive to ANY commands in the queue, including
        //     ones enqueued by other threads AFTER we called
        //     enqueue_wait_for_acks above (synchronize is a "drain
        //     everything in the queue" barrier, not "drain everything as
        //     of when I was called").
        //
        //   - !cmd_in_flight closes the window where the worker has popped
        //     the last command (making command_queue.empty() == true) but
        //     is still dispatching it -- its side effects (wire bytes,
        //     last_processed_minichunk, ack_queue push) are not yet
        //     published. Without this clause, a debug=false synchronize
        //     could return one command early (in debug mode the
        //     WAIT_FOR_ACKS we appended is itself covered by the
        //     ack_queue clause below, but SEND/DISCONNECT dispatches
        //     were not).
        //
        //   - ack_queue.empty() covers acks that are still outstanding
        //     BETWEEN commands (pushed by an earlier SEND's dispatch,
        //     drained later by _read_acks) -- cmd_in_flight cannot see
        //     those.
        //
        //   - When debug=false, ack_queue is always empty, so the
        //     predicate reduces to (command_queue.empty() &&
        //     !cmd_in_flight).
        Worker &w = *workers[worker_id];
        std::unique_lock<std::mutex> lock(w.mutex);
        for (;;) {
            if (_stopped_or_rethrow(w))
                return false;   // cleanly stopped before the queue drained
            if (w.command_queue.empty() && w.ack_queue.empty() && !w.cmd_in_flight)
                return true;
            w.drain_cv.wait(lock);
        }
    } catch (...) {
        stop(std::current_exception());
        throw;
    }
}


unsigned char FakeXEngine::get_minichunk_status(long worker_id, long minichunk_index) const
{
    if (worker_id < 0 || worker_id >= long(nworkers)) {
        stringstream ss;
        ss << "FakeXEngine::get_minichunk_status: worker_id=" << worker_id
           << " out of range [0, " << nworkers << ")";
        throw runtime_error(ss.str());
    }
    if (!debug)
        throw runtime_error(
            "FakeXEngine::get_minichunk_status: FakeXEngine was not"
            " constructed with debug=true");

    const Worker &w = *workers[worker_id];

    std::lock_guard<std::mutex> lock(w.mutex);

    if (w.first_minichunk < 0)
        throw runtime_error(
            "FakeXEngine::get_minichunk_status: no state-advancing"
            " commands have been enqueued yet on this worker");

    long idx = minichunk_index - w.first_minichunk;
    if (idx < 0 || idx >= long(w.minichunk_status.size())) {
        stringstream ss;
        ss << "FakeXEngine::get_minichunk_status: minichunk_index="
           << minichunk_index << " out of range ["
           << w.first_minichunk << ", "
           << (w.first_minichunk + long(w.minichunk_status.size()))
           << ") for worker_id=" << worker_id;
        throw runtime_error(ss.str());
    }
    return w.minichunk_status[idx];
}


std::array<long, 4> FakeXEngine::get_debug_counters() const
{
    std::array<long, 4> out;
    for (size_t i = 0; i < out.size(); i++)
        out[i] = debug_counters[i].load(std::memory_order_relaxed);
    return out;
}


// -------------------------------------------------------------------------------------------------
//
// _initialize: one-time setup, called from the top of _worker_main.
//
// Populates the worker's per-thread state (xmd, send_buf, header_nbytes,
// mc_nbytes, ip_addr, port). The TCP socket is NOT opened here -- that
// happens lazily on the first SEND_JUNK inside _send_junk().
//
// Combined "send buffer" laid out as:
//
//   [ 12-byte conn header + padded YAML ][ 12-byte mc header + (nbeams,nfreq,256) int4 data ]
//   ^                                    ^
//   send_buf.data()                      send_buf.data() + header_nbytes
//
// The first SEND_JUNK sends the whole buffer in one _send_all() call
// (connection header + first minichunk); subsequent SEND_JUNKs send only
// the minichunk portion. Combining the two saves a _send_all() call on
// the first send and avoids a TCP-level Nagle round-trip between the
// header and the first minichunk.
//
// The data is "junk" -- all zeros -- and stays that way; only the
// minichunk's seq field (at offset header_nbytes + 4) is rewritten per
// SEND_JUNK. See notes/network_protocol.md for the v2 wire format.


void FakeXEngine::_initialize(int worker_id)
{
    Worker &w = *workers[worker_id];

    // Round-robin assignment of worker -> Receiver, matching the
    // round-robin ip_addr assignment below. Indexes
    // max_acked_minichunk[] from _skip_or_send / _read_acks.
    w.receiver_id = long(worker_id) % num_receivers;

    // Per-worker metadata (round-robin freq-channel subset for this worker).
    w.xmd = make_worker_metadata(worker_id);
    long nfreq = w.xmd.freq_channels.size();
    long nbeams = w.xmd.get_nbeams();

    static constexpr long mc_header_nbytes = 12;
    long scales_offsets_nbytes = nbeams * nfreq * 4;     // (nbeams, nfreq, 2) float16
    long data_nbytes           = nbeams * nfreq * 128;   // (nbeams, nfreq, 256) int4
    w.mc_nbytes = mc_header_nbytes + scales_offsets_nbytes + data_nbytes;

    string yaml_str = w.xmd.to_yaml_string();
    long str_len = long(yaml_str.size()) + 1;     // +1 for the null terminator
    long padded_len = ((str_len + 3) / 4) * 4;    // 4-byte align
    w.header_nbytes = 12 + padded_len;

    w.send_buf.assign(w.header_nbytes + w.mc_nbytes, 0);

    // Stamp the connection header (one-time fields: protocol magic +
    // flags + yaml_len + padded YAML). 'flags' carries FLAG_ACK if
    // the FakeXEngine was constructed with debug=true.
    {
        uint32_t magic = protocol_magic;
        uint32_t flags = debug ? FLAG_ACK : 0u;
        uint32_t len32 = static_cast<uint32_t>(padded_len);
        std::memcpy(w.send_buf.data() + 0, &magic, 4);
        std::memcpy(w.send_buf.data() + 4, &flags, 4);
        std::memcpy(w.send_buf.data() + 8, &len32, 4);
        std::memcpy(w.send_buf.data() + 12, yaml_str.data(), yaml_str.size());
    }

    // Stamp the per-minichunk magic. Constant for this worker's lifetime;
    // only the seq field at offset (header_nbytes + 4) is rewritten per
    // SEND_JUNK.
    {
        uint32_t mc_magic = protocol_magic;
        std::memcpy(w.send_buf.data() + w.header_nbytes, &mc_magic, 4);
    }

    // Parsed destination (ip:port). Round-robin: workers share IPs cyclically.
    parse_ip_address(ip_addrs[worker_id % ip_addrs.size()], w.ip_addr, w.port);

    // w.sock stays default-constructed (fd == -1); w.connected stays false.
    // _skip_or_send's first need_send branch performs the connect().
}


// -------------------------------------------------------------------------------------------------
//
// _populate_minichunk_buf: gather one minichunk's worth of (beam, freq,
// 256-time-sample) int4 data from an AssembledFrameSet into the
// worker's send_buf. Hot path.
//
// Loop structure: outer over beams, inner over freqs, with
// memcpy(dst, src, 128) per (beam, freq). The 128 is a literal so the
// compiler lowers it to a small fixed sequence of SIMD load/store
// pairs (4x 32-byte on AVX2; 2x 64-byte on AVX-512). The destination
// pointer is contiguous within each beam's slab, which is friendly to
// the hardware prefetcher.
//
// REAPER RACE WARNING (see header): this function reads each frame's
// data buffer WITHOUT taking frame.mutex. The AssembledFrame reaper
// can concurrently zero data.size and release the underlying slab,
// which would make the const-char-* read here a use-after-free. We
// rely on the assumption that the reaper (which lives in FrbServer)
// is NEVER colocated with FakeXEngine in the same process. The
// defensive xassert_gt(frame.data.size, 0) below catches "frame was
// reaped before this call started", but is NOT a guard against an
// actively-running reaper that races us mid-loop. If we ever want to
// colocate FakeXEngine with a reaper, replace the unlocked read with
// a per-beam acquire of frame.mutex.


void FakeXEngine::_populate_minichunk_buf(Worker &w,
                                          const AssembledFrameSet &fset,
                                          long minichunk_index)
{
    // Number of bytes on the wire per (beam, freq) for one minichunk:
    // 256 int4 samples = 128 bytes. Compile-time constant for the
    // memcpy lowering.
    static constexpr long mc_time_bytes = 128;
    static constexpr long mc_so_bytes   = 4;     // 2 float16 = 4 bytes

    long worker_nfreq = long(w.xmd.freq_channels.size());
    long nbeams = fset.nbeams;
    long ntime = fset.ntime;

    // Cheap defensive consistency checks.
    xassert_eq(nbeams, w.xmd.get_nbeams());
    xassert((ntime % 256) == 0);

    long minichunks_per_chunk = ntime / 256;
    long imc_within_chunk = minichunk_index
                          - fset.time_chunk_index * minichunks_per_chunk;
    xassert_ge(imc_within_chunk, 0L);
    xassert_lt(imc_within_chunk, minichunks_per_chunk);

    // Source-side layouts:
    //   data:           (nfreq_total, ntime) int4, packed as (nfreq_total, ntime/2) bytes.
    //                   per-freq row stride = ntime/2; per-minichunk offset = imc * 128.
    //   scales_offsets: (nfreq_total, mpc, 2) float16, packed as (nfreq_total, mpc, 4) bytes.
    //                   per-freq row stride = mpc * 4; per-minichunk offset = imc * 4.
    long src_freq_stride    = ntime / 2;
    long src_time_offset    = imc_within_chunk * mc_time_bytes;
    long src_so_freq_stride = minichunks_per_chunk * mc_so_bytes;
    long src_so_offset      = imc_within_chunk * mc_so_bytes;

    // Destination-side layout in send_buf, starting at (header_nbytes + 12):
    //   [nbeams * worker_nfreq * 4 bytes scales_offsets][nbeams * worker_nfreq * 128 bytes data].
    char *dst_so   = w.send_buf.data() + w.header_nbytes + 12;
    char *dst_data = dst_so + nbeams * worker_nfreq * mc_so_bytes;

    const long *freq_channels = w.xmd.freq_channels.data();

    // First pass: scales_offsets (4 bytes per (beam, freq)).
    for (long ibeam = 0; ibeam < nbeams; ibeam++) {
        const AssembledFrame &frame = *fset.frames[ibeam];

        // Defense-in-depth: a reaped frame has scales_offsets.size == 0.
        // Same caveat as for data below (see the race warning above).
        xassert_gt(frame.scales_offsets.size, 0L);

        const char *frame_so = static_cast<const char *>(frame.scales_offsets.data);
        char *dst_beam_so = dst_so + ibeam * worker_nfreq * mc_so_bytes;

        for (long ifreq_local = 0; ifreq_local < worker_nfreq; ifreq_local++) {
            long freq_global = freq_channels[ifreq_local];
            const char *src = frame_so
                            + freq_global * src_so_freq_stride
                            + src_so_offset;
            char *dst = dst_beam_so + ifreq_local * mc_so_bytes;
            std::memcpy(dst, src, 4);
        }
    }

    // Second pass: int4 data (128 bytes per (beam, freq)).
    for (long ibeam = 0; ibeam < nbeams; ibeam++) {
        const AssembledFrame &frame = *fset.frames[ibeam];

        // Defense-in-depth: a reaped frame has data.size == 0. NOT a
        // safe guard against a concurrently-running reaper -- see the
        // race warning at the top of this function -- but cheap and
        // catches "frame was already reaped before this call started".
        xassert_gt(frame.data.size, 0L);

        const char *frame_data = static_cast<const char *>(frame.data.data);
        char *dst_beam = dst_data + ibeam * worker_nfreq * mc_time_bytes;

        for (long ifreq_local = 0; ifreq_local < worker_nfreq; ifreq_local++) {
            long freq_global = freq_channels[ifreq_local];
            const char *src = frame_data
                            + freq_global * src_freq_stride
                            + src_time_offset;
            char *dst = dst_beam + ifreq_local * mc_time_bytes;
            std::memcpy(dst, src, 128);
        }
    }
}


// -------------------------------------------------------------------------------------------------
//
// _disconnect: handler for DISCONNECT.
//
// Closes the worker's TCP socket (Socket::close() is public, idempotent,
// and resets fd back to -1) and flips the atomic flag. Idempotent --
// repeated DISCONNECTs on an already-disconnected worker are no-ops.
//
// Crucially, does NOT touch w.last_processed_minichunk or
// w.last_queued_minichunk; the next SEND_* on this worker will hit
// the !w.connected branch in _skip_or_send and transparently reopen
// the connection + re-send the protocol header (bundled with the
// next minichunk in one _send_all). Also does no notifying of its
// own: no waiter's predicate depends on the connected flag.


// -------------------------------------------------------------------------------------------------
//
// _read_acks: drain ack bytes from the worker's socket and update
// Worker::minichunk_status / Worker::ack_queue. Runs on the worker
// thread.
//
// blocking=false: a single non-blocking recv attempt (one syscall).
// Returns whatever's available -- zero bytes is fine.
//
// blocking=true: loops with 10ms inner timeouts (so is_stopped is
// observed within ~10ms of being set) and a 1-second outer deadline.
// On any of:
//   - peer EOF before ack_queue is drained
//   - invalid ack byte (not 0 or 1)
//   - the 1-second deadline expires
// _read_acks THROWS rather than recovering gracefully -- worker_main's
// catch handler calls stop(e) and the existing cascade tears down all
// controllers + workers with a descriptive error. This matches the
// "throw on anything wrong" policy for the debug-mode test-only path.


void FakeXEngine::_read_acks(Worker &w, bool blocking)
{
    // ensures that we never call Socket.read() in production (when
    // no acks will be sent)
    xassert(debug);

    if (!w.connected.load(std::memory_order_relaxed))
        return;

    // Snapshot the number of acks we need under the lock.
    long need;
    {
        std::lock_guard<std::mutex> lock(w.mutex);
        need = long(w.ack_queue.size());
    }
    if (need == 0) return;

    using clock = std::chrono::steady_clock;
    static constexpr auto blocking_deadline = std::chrono::seconds(1);
    static constexpr int blocking_inner_ms = 10;   // is_stopped check granularity
    auto deadline = clock::now() + blocking_deadline;

    char buf[256];

    while (need > 0) {
        // Stop-responsiveness: short-circuit if a sibling worker
        // or external thread has called stop().
        {
            std::lock_guard<std::mutex> lock(w.mutex);
            if (w.is_stopped) return;
        }

        long max_read = std::min<long>(long(sizeof(buf)), need);
        int inner_timeout = blocking ? blocking_inner_ms : 0;
        long n = w.sock.read_with_timeout(buf, max_read, inner_timeout);

        if (w.sock.eof) {
            // Peer closed before we received all expected acks.
            // Per the FLAG_ACK policy, throw rather than try to
            // recover -- worker_main's catch will call stop(e) and
            // propagate via the existing cascade.
            std::stringstream ss;
            ss << "FakeXEngine::_read_acks: receiver closed connection"
               << " with " << need << " ack(s) still expected";
            throw runtime_error(ss.str());
        }

        if (n == 0) {
            if (!blocking) return;   // single-shot non-blocking: empty is fine
            if (clock::now() >= deadline)
                throw runtime_error(
                    "FakeXEngine::_read_acks: timeout (1s) waiting for FLAG_ACK acks");
            continue;
        }

        // Process the n bytes under the lock.
        bool drained;
        {
            std::lock_guard<std::mutex> lock(w.mutex);
            // Defense-in-depth invariants. Should never trip under
            // correct locking discipline (only the worker thread
            // pops ack_queue and changes minichunk_status), but
            // they're cheap and would catch a queue/status drift.
            xassert(long(w.ack_queue.size()) == need);
            xassert(long(w.minichunk_status.size()) ==
                    w.last_queued_minichunk - w.first_minichunk + 1);

            for (long i = 0; i < n; i++) {
                if (w.ack_queue.empty())
                    throw runtime_error(
                        "FakeXEngine::_read_acks: received more ack bytes than expected");
                PendingAck done = w.ack_queue.front();
                unsigned char ack = static_cast<unsigned char>(buf[i]);
                if (ack != STATUS_DROPPED && ack != STATUS_ASSEMBLED) {
                    std::stringstream ss;
                    ss << "FakeXEngine::_read_acks: invalid ack byte "
                       << int(ack) << " from receiver (expected 0 or 1)";
                    throw runtime_error(ss.str());
                }
                long idx = done.minichunk_index - w.first_minichunk;
                xassert_ge(idx, 0L);
                xassert_lt(idx, long(w.minichunk_status.size()));
                xassert_eq(w.minichunk_status[idx], STATUS_SENT);

                // The STATUS_DROPPED == 0 and STATUS_ASSEMBLED == 1
                // numeric coincidence with the wire's ack-byte
                // values lets us assign minichunk_status[idx] = ack
                // directly.
                w.minichunk_status[idx] = ack;
                w.ack_queue.pop_front();

                // Update per-Receiver ack high-water mark: an ack for
                // done.minichunk_index from THIS worker's Receiver B
                // implies B has processed that minichunk.
                fetch_max(max_acked_minichunk[w.receiver_id],
                          done.minichunk_index);

                // Three-way ack-prediction assertion. Bounds the
                // receiver-side curr_base_chunk just BEFORE B
                // processed done.minichunk_index, then checks ack
                // against the prediction:
                //
                //   cbcp_lower = ichunk(done.max_ack_at_submission) - 1
                //     LOWER BOUND on B's cbc at processing time. Since
                //     B had processed done.max_ack_at_submission BEFORE
                //     this worker sent done.minichunk_index (the
                //     snapshot was taken before the send), B's cbc was
                //     >= ichunk(snapshot) - 1 by the time it processed
                //     done.minichunk_index.
                //   cbcp_upper = ichunk(max_sent_minichunk_now) - 1
                //     UPPER BOUND. Every advance to B's cbc (via B's
                //     own _process_data OR via cross-Receiver evict
                //     cascade from FrbServer) traces back to some
                //     sent minichunk m_trigger, and
                //     fetch_max(max_sent_minichunk, m_trigger + shift)
                //     ran BEFORE m_trigger left the kernel. The +mpc
                //     shift on first sends folds the receiver's
                //     initial_time_chunk into max_sent so this bound
                //     covers the init state too (see _skip_or_send).
                //
                // Three cases:
                //   i = ichunk(done.minichunk_index)
                //   i >= cbcp_upper -> B definitely processed with
                //                       cbc <= i -> must be ASSEMBLED.
                //   i < cbcp_lower  -> B definitely processed with
                //                       cbc > i  -> must be DROPPED.
                //   otherwise        -> ambiguous, no assertion.
                long imc = done.minichunk_index;
                long max_sent_now = max_sent_minichunk.load(std::memory_order_relaxed);

                auto ichunk_of = [this](long x) -> long {
                    return (x >= 0) ? (x / minichunks_per_chunk) : -1L;
                };

                long ii         = imc / minichunks_per_chunk;
                long cbcp_lower = ichunk_of(done.max_ack_at_submission) - 1L;
                long cbcp_upper = ichunk_of(max_sent_now) - 1L;

                if (ii >= cbcp_upper) {
                    if (ack != STATUS_ASSEMBLED) {
                        std::stringstream ss;
                        ss << "FakeXEngine::_read_acks: ack-prediction"
                           << " mismatch: minichunk_index=" << imc
                           << " predicted ASSEMBLED (ichunk=" << ii
                           << " >= cbcp_upper=" << cbcp_upper
                           << ", max_ack_at_submission=" << done.max_ack_at_submission
                           << ", max_sent=" << max_sent_now
                           << "), got ack=" << int(ack);
                        throw runtime_error(ss.str());
                    }
                    debug_counters[1].fetch_add(1, std::memory_order_relaxed);
                } else if (ii < cbcp_lower) {
                    if (ack != STATUS_DROPPED) {
                        std::stringstream ss;
                        ss << "FakeXEngine::_read_acks: ack-prediction"
                           << " mismatch: minichunk_index=" << imc
                           << " predicted DROPPED (ichunk=" << ii
                           << " < cbcp_lower=" << cbcp_lower
                           << ", max_ack_at_submission=" << done.max_ack_at_submission
                           << ", max_sent=" << max_sent_now
                           << "), got ack=" << int(ack);
                        throw runtime_error(ss.str());
                    }
                    debug_counters[0].fetch_add(1, std::memory_order_relaxed);
                } else {
                    // Ambiguous band: no assertion fires; bucket by the
                    // actual ack value (2 = DROPPED, 3 = ASSEMBLED).
                    debug_counters[(ack == STATUS_ASSEMBLED) ? 3 : 2]
                        .fetch_add(1, std::memory_order_relaxed);
                }
            }
            drained = w.ack_queue.empty();
        }

        // Wake synchronize() waiters, whose predicate (command_queue and
        // ack_queue both empty) may have just become true. Load-bearing for
        // the idle-loop drain in _worker_main, which is not followed by any
        // command-completion notify -- without this, a synchronize() caller
        // could sleep forever after the idle drain pops the last ack.
        // notify_all: when the queues drain, every waiter becomes ready.
        if (drained)
            w.drain_cv.notify_all();

        need -= n;
    }
}


void FakeXEngine::_disconnect(Worker &w)
{
    // Drain remaining acks BEFORE closing. If anything goes wrong
    // during the drain (peer EOF / invalid ack byte / 1-second
    // deadline), _read_acks throws -- worker_main's catch handler
    // calls stop(e) and the cascade tears down. No defensive
    // ack_queue.clear() is needed: either we drained successfully
    // (ack_queue is empty) or we threw (the worker is exiting).
    if (debug)
        _read_acks(w, /*blocking=*/true);

    if (w.connected.load(std::memory_order_relaxed)) {
        w.sock.close();
        w.connected.store(false, std::memory_order_relaxed);
    }
}


// -------------------------------------------------------------------------------------------------
//
// _skip_or_send: handler for SEND_JUNK / SKIP_MINICHUNK / SEND_MINICHUNK.
// See header for contract. Three discriminators:
//
//   - first_command (last_processed_minichunk == -1): the very first
//     state-advancing command on this worker may pick any
//     minichunk_index >= 0 (for NOTE-2 nonzero-initial-chunk tests).
//     Subsequent state-advancing commands must advance by exactly +1.
//     (DISCONNECT is not state-advancing -- it lives in _disconnect.)
//
//   - need_send (kind in {SEND_JUNK, SEND_MINICHUNK}): does this
//     command put bytes on the wire?
//
//   - first_send_after_connect: did this _skip_or_send call open the
//     TCP socket? If so, the conn header rides along in the same
//     _send_all call as the first minichunk. Note this branch is
//     taken both on the very first SEND_* AND after a DISCONNECT (in
//     which case it opens a fresh TCP connection + re-sends the
//     handshake transparently to the controller).


bool FakeXEngine::_skip_or_send(Worker &w, const Command &cmd)
{
    // Defense-in-depth monotonicity check. In principle this is
    // redundant with the queue-time sequentiality check that
    // _enqueue has already performed on every state-advancing
    // command (queue-time + FIFO ordering of command_queue together
    // guarantee that the order observed here matches the order
    // submitted), but we keep the processing-time check out of
    // caution -- the cost is negligible, and a violation would
    // immediately surface a queue-time logic bug as a worker-thread
    // exception via the existing stop() cascade.
    //
    // Safe to read w.last_processed_minichunk without w.mutex: the
    // worker thread is the sole writer.
    bool first_command = (w.last_processed_minichunk == -1);
    if (!first_command)
        xassert_eq(cmd.minichunk_index, w.last_processed_minichunk + 1L);
    xassert_ge(cmd.minichunk_index, 0L);

    bool need_send = (cmd.kind == Command::Kind::SEND_JUNK ||
                      cmd.kind == Command::Kind::SEND_MINICHUNK);

    // Snapshot of max_acked_minichunk[receiver_id] taken just before
    // _send_all. Attached to the PendingAck pushed at the bottom of
    // this function. Only set in the need_send branch below; the
    // SKIP_MINICHUNK path leaves it at -1 (which is never read,
    // since SKIP doesn't push to ack_queue).
    long max_ack_at_submission = -1;

    if (need_send) {
        // Paced-mode bootstrap + gate. See plans/fake_xengine_pacing.md.
        // The bootstrap runs only on this worker's very first SEND_*
        // (!first_send_done). The gate runs on every SEND_*.
        if (paced) {
            long ichunk = cmd.minichunk_index / minichunks_per_chunk;

            if (!w.first_send_done) {
                // CAS the engine-wide floor from -1 to (ichunk * nbeams).
                // First-worker-to-send wins; later workers' CAS fails and
                // their `expected` is loaded with the winner's value. We
                // then re-load explicitly for clarity and use std::max
                // when seeding our own Worker::rb_processed.
                //
                // Safe to read w.first_send_done without w.mutex: only
                // the worker thread writes it (existing invariant; see
                // Worker doc-comment in FakeXEngine.hpp).
                long my_floor = ichunk * nbeams;
                long expected = -1;
                rb_processed_floor.compare_exchange_strong(
                    expected, my_floor,
                    std::memory_order_acq_rel,
                    std::memory_order_acquire);
                long current_floor =
                    rb_processed_floor.load(std::memory_order_acquire);

                std::lock_guard<std::mutex> lk(w.mutex);
                w.rb_processed = std::max(w.rb_processed, current_floor);
                // No gate_cv notify needed -- this worker is the only
                // thread that would be waiting on its own gate_cv for an
                // rb_processed change, and it's not waiting; it's
                // running this code.
            }

            // Gate: wait until rb_processed catches up to within 5
            // chunks of this worker's most recent successful SEND, or
            // stop. The required > 0 fast path skips the lock for the
            // (common) case where last_ichunk_sent <= 5 (worker hasn't
            // sent enough yet for the gate to bite). See the paced-mode
            // section of FakeXEngine.hpp's class doc-comment for why
            // the horizon is last_ichunk_sent rather than the pending
            // command's ichunk: SKIPs let a worker's pending ichunk
            // run arbitrarily far ahead of its actual SENDs without
            // advancing the server, so gating against the pending
            // ichunk would deadlock after a SKIP-heavy stretch.
            long required = (w.last_ichunk_sent - 5) * nbeams;
            if (required > 0) {
                std::unique_lock<std::mutex> lk(w.mutex);
                w.gate_cv.wait(lk, [&] {
                    return w.is_stopped || w.rb_processed >= required;
                });
                if (w.is_stopped)
                    return false;
            }
        }

        // Lazy connect: the first SEND_* opens the socket. SKIP_MINICHUNK
        // never opens the socket -- so a worker that only ever sees
        // SKIPs leaves its TCP connection un-established.
        bool first_send_after_connect = false;
        if (!w.connected.load(std::memory_order_relaxed)) {
            // Connect failures throw (e.g. ECONNREFUSED); worker_main's
            // catch handler then calls FakeXEngine::stop().
            //
            // This branch is also entered after a DISCONNECT command
            // has flipped w.connected back to false -- in that case
            // the same lazy-connect path opens a fresh TCP connection
            // and re-sends the protocol handshake (since the
            // first_send_after_connect=true branch packs both the
            // connection header and this minichunk into one _send_all).
            w.sock = Socket(PF_INET, SOCK_STREAM);

            // Non-blocking connect + poll, rechecking w.is_stopped every
            // 100 ms. (A plain blocking connect() could stall for the
            // kernel's SYN-retry timeout, ~2 minutes, if the receiver is
            // unreachable -- blocking stop() and the destructor for that
            // long.)
            w.sock.start_connect(w.ip_addr, w.port);

            while (!w.sock.wait_for_connect(100)) {
                std::lock_guard<std::mutex> lk(w.mutex);
                if (w.is_stopped)
                    return false;
            }

            w.connected.store(true, std::memory_order_relaxed);
            first_send_after_connect = true;
        }

        // SEND_MINICHUNK: gather real data into the minichunk_buf.
        // SEND_JUNK: leave the data area as the all-zero initialization
        // from _initialize().
        if (cmd.kind == Command::Kind::SEND_MINICHUNK) {
            // _enqueue() already rejected null frame_set, but
            // defense-in-depth in case a future caller bypasses it.
            xassert(cmd.frame_set);
            _populate_minichunk_buf(w, *cmd.frame_set, cmd.minichunk_index);
        }

        // Stamp the wire-seq for this minichunk.
        char *mc_ptr = w.send_buf.data() + w.header_nbytes;
        uint64_t mc_seq = uint64_t(cmd.minichunk_index) * 256ULL
                        * uint64_t(w.xmd.seq_per_frb_time_sample);
        std::memcpy(mc_ptr + 4, &mc_seq, 8);

        // First send after connect: conn header + minichunk in one shot.
        // Subsequent sends: just the minichunk portion of send_buf.
        const char *send_ptr = first_send_after_connect ? w.send_buf.data() : mc_ptr;
        long send_nbytes = first_send_after_connect ? long(w.send_buf.size()) : w.mc_nbytes;

        // FLAG_ACK ack-prediction bookkeeping. Snapshot the per-Receiver
        // ack high-water mark BEFORE the wire send, then update the
        // global "max sent" mark (also BEFORE the send). Both timings
        // are load-bearing for the bounds used by _read_acks's
        // ack-prediction assertion (see _read_acks for the full
        // assertion logic).
        max_ack_at_submission =
            max_acked_minichunk[w.receiver_id].load(std::memory_order_relaxed);

        // SHIFT TRICK for first send: a worker's very first SEND
        // contributes (cmd.minichunk_index + minichunks_per_chunk) to
        // max_sent_minichunk; subsequent sends contribute just
        // cmd.minichunk_index. Non-obvious; here's why:
        //
        // The receiver-side Receiver::curr_base_chunk (cbc) starts at
        // `initial_time_chunk`, which is set on the first per-minichunk
        // header to arrive at ANY peer of ANY Receiver -- i.e. by the
        // chunk-index of WHICHEVER worker's first SEND lands first. We
        // (the FakeXEngine) don't know which worker that is.
        //
        // The ack-prediction upper bound on cbc_B uses
        //    cbcp_upper = ichunk(max_sent_minichunk) - 1
        // (where ichunk(x) = x / minichunks_per_chunk). For this to be
        // a valid upper bound at the moment the receiver's cbc is just
        // sitting at init (no minichunks processed yet), we need
        //    ichunk(max_sent_minichunk) - 1 >= init.
        // Without the shift, max_sent_minichunk >= m_init_setter and
        // ichunk(max_sent) - 1 = init - 1 < init -- BOUND BREAKS, can
        // produce false-positive ASSEMBLED assertions.
        //
        // With the shift, the init-setter worker's contribution is
        //    m_init_setter + minichunks_per_chunk,
        // so ichunk(max_sent) - 1 >= ichunk(m_init_setter) = init.
        // Bound holds.
        //
        // We apply the shift to EVERY worker's first send because we
        // don't know at runtime which worker won the init race -- the
        // shift is the same whether it's the init-setter or not, and
        // overestimating max_sent_minichunk only makes the upper bound
        // looser (safe direction).
        //
        // first_send_done persists across DISCONNECT/reconnect cycles
        // -- the receiver's initial_time_chunk is set ONCE per server
        // process and never reset, so re-shifting on reconnect would
        // be incorrect bookkeeping (still safe but pointless).
        long shift = w.first_send_done ? 0L : minichunks_per_chunk;
        fetch_max(max_sent_minichunk, cmd.minichunk_index + shift);
        w.first_send_done = true;

        if (!_send_all(w, w.sock, send_ptr, send_nbytes))
            return false;

        // Record this send's chunk index for the paced-mode gate. Only
        // meaningful when paced=true, but cheap and harmless otherwise.
        // Set AFTER _send_all() so a stop / connreset-bailed send isn't
        // counted -- the worker is exiting anyway and the value would
        // not be referenced again. See the paced-mode section of
        // FakeXEngine.hpp's class doc-comment for the role of this
        // field in the gate predicate.
        w.last_ichunk_sent = cmd.minichunk_index / minichunks_per_chunk;
    }
    // SKIP_MINICHUNK falls through here -- no wire activity, just
    // advance state below. SKIPs do NOT touch max_sent_minichunk,
    // max_ack_at_submission, first_send_done, or last_ichunk_sent:
    // nothing went on the wire, so the server cannot have seen this
    // minichunk from us and cannot ever ack it (and the paced gate
    // explicitly ignores SKIP-driven ichunk advances).

    // Determine the new minichunk_status entry for this cmd.
    // Successful SEND_* -> STATUS_SENT (the wire bytes went out;
    // the ack hasn't arrived yet). SKIP_MINICHUNK -> STATUS_SKIPPED.
    unsigned char new_status;
    bool push_ack;
    if (cmd.kind == Command::Kind::SKIP_MINICHUNK) {
        new_status = STATUS_SKIPPED;
        push_ack = false;
    } else {
        new_status = STATUS_SENT;
        push_ack = true;
    }

    // Publish last_processed_minichunk + (if debug) update status /
    // push ack_queue under lock. (The processed_cv/drain_cv notifies are
    // centralized in _worker_main, fired once per command dispatched -- so
    // wait_until_processed AND synchronize waiters wake after each command
    // is fully processed.)
    {
        std::lock_guard<std::mutex> lock(w.mutex);
        if (debug) {
            long idx = cmd.minichunk_index - w.first_minichunk;
            xassert_ge(idx, 0L);
            xassert_lt(idx, long(w.minichunk_status.size()));
            xassert_eq(w.minichunk_status[idx], STATUS_QUEUED);
            w.minichunk_status[idx] = new_status;
            if (push_ack)
                w.ack_queue.push_back({cmd.minichunk_index, max_ack_at_submission});
        }
        w.last_processed_minichunk = cmd.minichunk_index;
    }

    // Drain any acks that arrived while we were sending. Cheap
    // single-syscall non-blocking attempt -- if no bytes are
    // queued, the recv returns 0 and we proceed.
    if (debug)
        _read_acks(w, /*blocking=*/false);

    return true;
}


// -------------------------------------------------------------------------------------------------
//
// Worker main loop. Drained by external controller thread(s) via
// enqueue_send_junk().


void FakeXEngine::_worker_main(int worker_id)
{
    Worker &w = *workers[worker_id];

    _initialize(worker_id);

    for (;;) {
        Command cmd;
        {
            std::unique_lock<std::mutex> lock(w.mutex);
            for (;;) {
                if (w.is_stopped) return;
                if (!w.command_queue.empty()) break;

                // When ack_queue is empty (always so when debug=
                // false), wait unbounded -- no spurious wakeups.
                // When ack_queue is non-empty, periodically drain
                // acks so we don't let the receiver's send buffer
                // for ack bytes fill up.
                if (w.ack_queue.empty()) {
                    w.cmd_cv.wait(lock);
                } else {
                    auto status = w.cmd_cv.wait_for(lock, std::chrono::milliseconds(1));
                    if (status == std::cv_status::timeout) {
                        lock.unlock();
                        _read_acks(w, /*blocking=*/false);
                        lock.lock();
                    }
                }
            }

            cmd = w.command_queue.front();
            w.command_queue.pop_front();

            // The popped command is now invisible to synchronize()'s
            // queue-empty clause; cmd_in_flight covers it until the
            // dispatch below has fully published its side effects.
            w.cmd_in_flight = true;
        }

        // Dispatch. cmd_in_flight is reset on every exit path (normal,
        // stop-return, throw); on the two worker-exit paths the engine is
        // (or is about to be) stopped, so synchronize() waiters leave via
        // their stopped-check either way -- the reset just keeps the
        // "in-progress flags" invariant clean (notes/cpp.md).
        bool worker_exiting = false;
        try {
            switch (cmd.kind) {
            case Command::Kind::SEND_JUNK:
            case Command::Kind::SKIP_MINICHUNK:
            case Command::Kind::SEND_MINICHUNK:
                worker_exiting = !_skip_or_send(w, cmd);
                break;
            case Command::Kind::DISCONNECT:
                _disconnect(w);
                break;
            case Command::Kind::WAIT_FOR_ACKS:
                // WAIT_FOR_ACKS is enqueued only by
                // enqueue_wait_for_acks() / synchronize(), both of which
                // check debug first. xassert here as a defense-in-depth
                // invariant.
                xassert(debug);
                _read_acks(w, /*blocking=*/true);
                break;
            default: {
                // Defensive -- UNINITIALIZED Commands should never be enqueued.
                stringstream ss;
                ss << "FakeXEngine worker " << worker_id
                   << ": got Command with kind=" << uint32_t(cmd.kind);
                throw runtime_error(ss.str());
            }
            }
        } catch (...) {
            std::lock_guard<std::mutex> lock(w.mutex);
            w.cmd_in_flight = false;
            throw;   // worker_main's catch-all calls stop(e)
        }

        {
            std::lock_guard<std::mutex> lock(w.mutex);
            w.cmd_in_flight = false;
        }

        if (worker_exiting)
            return;

        // Centralized notify, one per command dispatched. Wakes both
        // wait_until_processed waiters (whenever last_processed_minichunk
        // advanced) AND synchronize() waiters (whenever command_queue
        // empties or shrinks). Notifying *after* the dispatch returns
        // is what makes synchronize() wait for WAIT_FOR_ACKS's
        // blocking drain to finish before returning. Both cvs are
        // notified unconditionally (cheap when nobody waits); notify_all
        // because several callers can wait on each.
        w.processed_cv.notify_all();
        w.drain_cv.notify_all();
    }
}


// -------------------------------------------------------------------------------------------------
//
// Pacing thread (paced=true only). Holds a MonitorRingbuf streaming RPC
// to the FrbServer at rpc_address. For each pushed rb_processed value,
// sweeps `workers` and monotonically advances each Worker::rb_processed
// (under each Worker::mutex), notifying its cv so any worker blocked in
// the paced-mode gate wakes up to re-check the predicate.
//
// The channel + Stub are LOCALS in this function, not FakeXEngine
// members. Only the ClientContext is a member (so FakeXEngine::stop()
// can TryCancel it). See FakeXEngine.hpp for why the Stub isn't a
// member (nested-class forward-decl issue).
//
// Stream termination:
//   - We cancelled (stop() called TryCancel): silent exit.
//   - Server-initiated close or network error: throw runtime_error;
//     the wrapper catches and calls stop(current_exception()), which
//     cascades to all workers.


void FakeXEngine::_pacing_thread_main()
{
    using namespace frb::search::v1;

    // Local channel + stub: the pacing thread is the sole user.
    auto channel = grpc::CreateChannel(rpc_address,
                                       grpc::InsecureChannelCredentials());
    auto stub = FrbSearch::NewStub(channel);

    // Stamp our wire-protocol version on the stream-opening request (see
    // notes/grpc.md); the server rejects a mismatch. FakeXEngine and FrbServer
    // are the same pirate build here, so this always matches -- but the field
    // must be set (a default 0 would be rejected as PROTOCOL_VERSION_UNSPECIFIED).
    MonitorRingbufRequest mr_req;
    mr_req.set_protocol_version(PROTOCOL_VERSION_CURRENT);
    auto reader = stub->MonitorRingbuf(pacing_ctx.get(), mr_req);

    MonitorRingbufResponse resp;
    while (reader->Read(&resp)) {
        long v = resp.rb_processed();

        // Broadcast to every Worker. One Worker mutex at a time; never
        // hold more than one. The pacing thread and FakeXEngine::stop()
        // both take Worker mutexes one at a time -- their per-worker
        // critical sections interleave freely.
        for (auto &wp : workers) {
            Worker &w = *wp;
            bool advanced = false;
            {
                std::lock_guard<std::mutex> lk(w.mutex);
                long new_val = std::max(w.rb_processed, v);
                if (new_val != w.rb_processed) {
                    w.rb_processed = new_val;
                    advanced = true;
                }
            }
            // Notify after releasing the mutex, so a gate-blocked worker
            // isn't woken straight into a lock it can't take. notify_one:
            // the worker thread's paced gate is the only gate_cv waiter.
            if (advanced)
                w.gate_cv.notify_one();
        }
    }

    // Stream ended. Finish() returns the final Status.
    grpc::Status status = reader->Finish();

    // Acquire so we synchronize with stop()'s write of is_stopped_cache.
    // (stop()'s compare_exchange_strong uses the default memory order,
    // seq_cst, which includes the release semantics this pairing relies
    // on.) A relaxed load could miss a just-completed stop() call and
    // produce a spurious "stream closed unexpectedly" error.
    if (is_stopped_cache.load(std::memory_order_acquire))
        return;   // we cancelled via stop(); silent exit

    // Server-initiated close, network error, or shutdown race. Convert
    // to an exception so the wrapper propagates via
    // stop(current_exception()).
    stringstream ss;
    ss << "FakeXEngine pacing: MonitorRingbuf stream closed unexpectedly";
    if (!status.ok()) {
        ss << " (code=" << status.error_code()
           << ", message='" << status.error_message() << "')";
    }
    throw runtime_error(ss.str());
}


void FakeXEngine::pacing_thread_main()
{
    // Per the error-reporting convention (notes/stoppable_class.md),
    // worker threads never print exceptions: stop(e) saves the root
    // cause, and entry points rethrow it to the caller.
    try {
        _pacing_thread_main();
    } catch (...) {
        stop(std::current_exception());
    }
}


}  // namespace pirate
