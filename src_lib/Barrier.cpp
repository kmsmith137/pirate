#include "../include/pirate/Barrier.hpp"
#include <ksgpu/xassert.hpp>

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
    xassert(nthreads_ > 0);
    std::unique_lock ul(lock);

    xassert_msg(this->nthreads <= 0, "Barrier::initialize() called on already-initialized Barrier");
    xassert_msg(!this->is_stopped, "Barrier::initialize() called on stopped Barrier");
    this->nthreads = nthreads_;
}


bool Barrier::is_initialized()
{
    std::unique_lock ul(lock);
    return nthreads > 0;
}


void Barrier::wait()
{
    std::unique_lock ul(lock);

    xassert_msg(nthreads > 0, "Barrier::wait() called on uninitialized Barrier");

    if (is_stopped) {
        if (error)
            std::rethrow_exception(error);
        return;
    }

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

    if (_unlikely(is_stopped)) {
        if (error)
            std::rethrow_exception(error);
        return;
    }
}


void Barrier::stop(exception_ptr e)
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
