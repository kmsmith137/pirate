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
    std::mutex lock;
    std::condition_variable cv;

    // Protected by lock.
    int nthreads = 0;
    int nthreads_waiting = 0;
    int wait_count = 0;
    bool is_stopped = false;
    std::exception_ptr error;

    // If constructor is called with nthreads=0, then 'nthreads' must be
    // set later, with a call to initialize().
    Barrier(int nthreads);

    // Entry point: blocks until all N threads arrive. If the Barrier is
    // stopped (including while blocked), throws the saved exception, or a
    // generic runtime_error if stop() was called with a null exception_ptr.
    void wait();

    void stop(std::exception_ptr e = nullptr);
    void initialize(int nthreads);
    bool is_initialized();

    // Helper for entry points. Caller must hold lock.
    void _throw_if_stopped(const char *method_name);

    // Noncopyable
    Barrier(const Barrier &) = delete;
    Barrier &operator=(const Barrier &) = delete;
};


}  // namespace pirate

#endif // _PIRATE_BARRIER_HPP
