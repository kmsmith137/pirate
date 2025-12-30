# Thread-Backed Class Pattern

A pattern for a class `X`:

- `X` is backed by one or more worker threads whose lifetimes are "tied" to X: threads are created when new objects are created, and joined in `~X()`. Note that it is safe for the worker thread(s) to hold a bare pointer (`X *this`), since the object always outlives the worker thread(s).

- `X` has a `stop(std::exception_ptr e)` method which can be called externally, to put the object into a "stopped" state (`X::is_stopped==true`). The value of `e` is saved in `X::error`, and is null or non-null depending on whether the call to `stop()` represents normal termination. The first caller to `stop()` sets `X::error`. 

- When the object enters its stopped state, the worker thread returns (as promptly as is practical).

- If the worker thread throws an exception `e`, then the worker thread catches the exception, calls `X::stop(e)`, and exits.

- Some public methods of `X` are labelled as "entry points". If an entry point is called in the stopped state, the exception `e` is rethrown. (If `e` is `NULL`, then a generic exception `runtime_error("X::f() called on stopped instance")` is thrown.) 

- If an entry point throws an exception, then `X::stop(e)` is called, and the exception is rethrown.

- In a context where the value of `is_stopped` is checked (e.g. entry point or `worker_thread_main`), if a blocking call is made (e.g. `cv.wait()`) then `is_stopped` is rechecked on wakeup. `X::stop()` should call `cv.notify()`. In situations where this notification mechanism doesn't work, please find an alternative if possible. (For example, a network worker thread which needs to block waiting on a socket could specify a 1-ms timeout, in order to recheck `is_stopped` every millisecond.)

- `~X()` calls `stop()` before joining the worker thread, to force the worker thread to exit.

## Example Code

```cpp
#include <condition_variable>
#include <exception>
#include <mutex>
#include <queue>
#include <thread>

class X {
    std::mutex mutex;
    std::condition_variable cv;
    std::queue<int> queue;
    bool is_stopped = false;
    std::exception_ptr error;
    std::thread worker;

    void _worker_main() {
        while (true) {
            std::unique_lock lock(mutex);

            for (;;) {
                if (is_stopped)
                    return;
                if (!queue.empty())
                    break;
                cv.wait(lock);
            }

            int req = queue.front();
            queue.pop();
            lock.unlock();

            process_request(req);
        }
    }

    void worker_main() {
        try {
            _worker_main();  // only returns if X::is_stopped
        } catch (...) {
            stop(std::current_exception());
        }
    }

    void process_request(int id) {
        // Process request here
    }

public:
    X() {
        // All members are initialized before this point (default member initializers).
        // Now safe to start the worker thread.
        worker = std::thread(&X::worker_main, this);
    }

    ~X() {
        this->stop();
        if (worker.joinable())
            worker.join();
    }

    void stop(std::exception_ptr e = nullptr) 
    {
        std::lock_guard lock(mutex);
        if (is_stopped) return;
        is_stopped = true;
        error = e;
        cv.notify_all();
    }

    void queue_request(int id) 
    {
        std::lock_guard lock(mutex);
        if (error) 
            std::rethrow_exception(error);
        if (is_stopped) 
            throw std::runtime_error("X::queue_request() called on stopped instance");
        queue.push(id);
        cv.notify_one();
    }

    void example_entry_point() {
        std::unique_lock<std::mutex> lk(mutex);
        if (error) 
            std::rethrow_exception(error);
        if (is_stopped) 
            throw std::runtime_error("X::example_entry_point() called on stopped instance");
        lk.unlock();

        try {
            _example_entry_point();
        } catch (...) {
            this->stop(std::current_exception());
            throw;
        }
    }

    void _example_entry_point() {
        // Entry point body here.
    }
};
```

