#include "../include/pirate/Barrier.hpp"

#include <ksgpu/xassert.hpp>

#include <string>
#include <stdexcept>

using namespace std;


namespace pirate {
#if 0
}   // pacify editor auto-indent
#endif


Barrier::Barrier(int nthreads_)
{
    xassert_msg(nthreads_ >= 0, "Barrier constructor: expected nthreads >= 0");
    this->nthreads = nthreads_;
}


void Barrier::initialize(int nthreads_)
{
    // Per the strict stoppable-class policy (notes/stoppable_class.md), ANY
    // exception thrown from an entry point (here and in wait() below) stops
    // the Barrier -- including argument/precondition errors.
    try {
        xassert(nthreads_ > 0);
        std::unique_lock ul(lock);

        xassert_msg(this->nthreads <= 0, "Barrier::initialize() called on already-initialized Barrier");
        _throw_if_stopped("Barrier::initialize()");
        this->nthreads = nthreads_;
    } catch (...) {
        stop(std::current_exception());
        throw;
    }
}


bool Barrier::is_initialized()
{
    std::unique_lock ul(lock);
    return nthreads > 0;
}


// Helper for entry points. Caller must hold lock.
void Barrier::_throw_if_stopped(const char *method_name)
{
    if (error)
        std::rethrow_exception(error);
    if (is_stopped)
        throw std::runtime_error(std::string(method_name) + " called on stopped instance");
}


void Barrier::wait()
{
    try {
        std::unique_lock ul(lock);

        xassert_msg(nthreads > 0, "Barrier::wait() called on uninitialized Barrier");

        _throw_if_stopped("Barrier::wait()");

        if (nthreads_waiting == nthreads-1) {
            this->nthreads_waiting = 0;
            this->wait_count++;
            ul.unlock();
            cv.notify_all();
            return;
        }

        this->nthreads_waiting++;

        int wc = this->wait_count;
        cv.wait(ul, [this,wc] { return (this->is_stopped || (this->wait_count > wc)); });

        if (_unlikely(is_stopped))
            _throw_if_stopped("Barrier::wait()");
    } catch (...) {
        stop(std::current_exception());
        throw;
    }
}


void Barrier::stop(exception_ptr e) const
{
    std::unique_lock ul(lock);

    if (is_stopped)
        return;

    this->is_stopped = true;
    this->error = e;
    ul.unlock();
    cv.notify_all();
}


} // namespace pirate
