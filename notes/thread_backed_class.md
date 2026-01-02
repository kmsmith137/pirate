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

- In the example below, the worker thread is created in `X::X()`, but in other cases, the worker may be created in a different method, for example `X::start()` or `X:allocate()`.

- If `X` contains pointers to other thread-backed classes (or more generally, to any class `Y` defining `Y::stop()`), then `X::stop()` should call `ptr->stop()` for each such pointer.

## Example Code

In this toy example, `X` is backed by one worker thread, and contains a thread-safe work queue.
The worker takes integer `id` values from the queue, and calls `X::process_request(id)`.
The entry point `X::queue_request()` adds work to the queue.
(The work queue is specific to this toy example -- other thread-backed classes may not contain a `std::queue`.)

```cpp
#include <condition_variable>
#include <exception>
#include <mutex>
#include <queue>
#include <string>
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

    // A helper function for entry points. Caller must hold mutex.
    void _throw_if_stopped(const char *method_name)
    {
        if (error) 
            std::rethrow_exception(error);

        if (is_stopped) {
            throw std::runtime_error(std::string(method_name) + " called on stopped instance");
        }
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
        _throw_if_stopped("X::queue_request");
        queue.push(id);
        cv.notify_one();
    }

    void example_entry_point() 
    {
        std::unique_lock lock(mutex);
        _throw_if_stopped("X::example_entry_point");
        lock.unlock();

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

