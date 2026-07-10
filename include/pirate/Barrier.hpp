#ifndef _PIRATE_BARRIER_HPP
#define _PIRATE_BARRIER_HPP

#include <exception>
#include <mutex>
#include <condition_variable>

namespace pirate {
#if 0
}   // pacify editor auto-indent
#endif


// Barrier: synchronization point between N threads.
// Stoppable class (see notes/stoppable_class.md).
// (Note that std::barrier was introduced in C++20, but I'm still on C++17.)

struct Barrier
{
    // Stop-pattern state ('mutable' since stop() is const -- see
    // notes/stoppable_class.md). is_stopped/error are protected by 'lock'.
    mutable std::mutex lock;
    mutable std::condition_variable cv;
    mutable bool is_stopped = false;
    mutable std::exception_ptr error;

    // Protected by lock. wait_count is a generation counter (one increment
    // per barrier release); 'long' so it cannot overflow (UB, and a hung
    // 'wait_count > wc' predicate) in a long-running server.
    int nthreads = 0;
    int nthreads_waiting = 0;
    long wait_count = 0;

    // If constructor is called with nthreads=0, then 'nthreads' must be
    // set later, with a call to initialize().
    Barrier(int nthreads);

    // Entry point: blocks until all N threads arrive. If the Barrier is
    // stopped (including while blocked), throws the saved exception, or a
    // generic runtime_error if stop() was called with a null exception_ptr.
    void wait();

    void stop(std::exception_ptr e = nullptr) const;
    void initialize(int nthreads);
    bool is_initialized();

    // Helper for entry points. Caller must hold lock.
    void _throw_if_stopped(const char *method_name);

    // Noncopyable, nonmovable (std::mutex/std::condition_variable members).
    Barrier(const Barrier &) = delete;
    Barrier &operator=(const Barrier &) = delete;
    Barrier(Barrier &&) = delete;
    Barrier &operator=(Barrier &&) = delete;
};


}  // namespace pirate

#endif // _PIRATE_BARRIER_HPP
