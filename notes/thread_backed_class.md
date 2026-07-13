# Thread-Backed Class Pattern

A pattern for a class `X`:

- `X` is thread-safe.

- `X` is backed by one or more worker threads whose lifetimes are "tied" to X: threads are created when new objects are created, and joined in `~X()`. Workers hold a bare pointer (`X *this`) -- safe, since the object always outlives the worker thread(s). In fact workers MUST hold a bare pointer, never a `shared_ptr<X>` (or `shared_from_this()`): a worker's shared_ptr would keep the object alive until the worker exits, but the worker exits only when the destructor joins it, so the destructor could never run.

- `X` has a `stop(std::exception_ptr e)` method which can be called externally, to put the object into a "stopped" state (`X::is_stopped==true`). The value of `e` is saved in `X::error`, and is null or non-null depending on whether the call to `stop()` represents normal termination. The first caller to `stop()` sets `X::error`. `stop()` is declared `const`, and the stop-pattern state (mutex, condition variables, `is_stopped`, `error`) is declared `mutable` -- see notes/stoppable_class.md.

- When the object enters its stopped state, the worker thread returns (as promptly as is practical).

- If the worker thread throws an exception `e`, then the worker thread catches the exception, calls `X::stop(e)`, and exits.

- Some public methods of `X` are labelled as "entry points". If an entry point is called in the stopped state, the exception `e` is rethrown. (If `e` is `nullptr`, then a generic exception `runtime_error("X::f() called on stopped instance")` is thrown.) 

- If an entry point throws an exception, then `X::stop(e)` is called, and the exception is rethrown. This is the strict policy described in notes/stoppable_class.md, which also lists the exempt method categories (stop(), accessors, join/wait-style methods) and the "done-value" variant for blocking methods where a clean stop is an expected outcome.

- In a context where the value of `is_stopped` is checked (e.g. entry point or `worker_thread_main`), if a blocking call is made (e.g. `cv.wait()`) then `is_stopped` is rechecked on wakeup. `X::stop()` should call `notify_all()` on each of `X`'s condition variables. In situations where this notification mechanism doesn't work, please find an alternative if possible. (For example, a network worker thread which needs to block waiting on a socket could specify a 1-ms timeout, in order to recheck `is_stopped` every millisecond.)

- `~X()` calls `stop()` before joining the worker thread, to force the worker thread to exit.

- `X` is noncopyable, nonmoveable, and normally accessed through a shared_ptr (see notes/stoppable_class.md for the pybind11 holder rule and the rare sanctioned exceptions).

- In the example below, the worker thread is created in `X::X()`, but in other cases, the worker may be created in a different method, for example `X::start()` or `X::allocate()` (see "Spawning, joining, and teardown" below).

- If `X` contains pointers to other thread-backed classes (or more generally, to any class `Y` defining `Y::stop()`), then `X::stop(e)` should call `ptr->stop(e)` for each such pointer, forwarding the exception (see "Error reporting" in notes/stoppable_class.md).

Every thread-backed class is a stoppable class (see notes/stoppable_class.md), but not vice versa.
In particular, the "Locking and condition variables", "pybind11 bindings", and "Reviewer checklist" sections there apply here too.

## Spawning, joining, and teardown

Rules for classes whose threads are spawned outside the constructor, or that
have more than one thread. (The single-thread ctor-spawned example below has
none of these hazards.) Each rule's violation turned up as a real data race
or teardown bug in review:

- Publish thread handles (and similar teardown state, e.g. a grpc server
  handle) UNDER the mutex. If threads are spawned in start()/open()/
  allocate(), the handle assignments happen with the mutex held, and
  stop()/join()/close()/~X() synchronize on the mutex before reading them
  -- stop() can run concurrently with start(). Holding the lock across the
  spawns is safe as long as the spawner never waits on the spawned threads:
  a freshly-spawned worker that touches X's state just blocks briefly until
  the spawner releases.

- Join paths take the mutex briefly (to synchronize with that publication),
  then join with the mutex RELEASED -- workers take the mutex on their exit
  paths, so joining under it deadlocks.

- A constructor that spawns SEVERAL threads must handle a mid-spawn
  failure: catch, stop(), join the already-spawned threads, rethrow.
  Otherwise the std::thread members are destroyed while joinable, which is
  std::terminate. (See FakeXEngine's constructor for the model.)

- The destructor calls stop(), then joins the threads inline. It must never
  call a public join()-style method that rethrows the saved error (a
  throwing destructor). If both the destructor and a public join() need the
  joining loop, factor a private _join_threads() helper (see Hwtest).

- A close()-style teardown method needs more than an idempotency flag: a
  second concurrent caller must BLOCK until the first finishes (e.g. a
  leaf-level close_mutex held for the whole body -- see FrbGrouper). With a
  flag alone, the second caller -- often the destructor -- returns
  immediately and destroys members out from under the first caller's
  still-running teardown.

- Callback/handler threads (e.g. gRPC handlers) hold a BARE pointer to X,
  exactly like worker threads -- never a shared_ptr, and never
  weak_ptr::lock() (which manufactures one). Any owning reference on a
  handler thread can become the LAST reference, running ~X on that thread,
  where teardown joins/waits on the very thread executing the destructor
  (self-deadlock). The bare pointer is safe under the same invariant that
  protects workers: the destructor shuts down and drains the handler
  dispatcher (e.g. grpc Server::Shutdown + Wait) before members are
  destroyed, and no handler is dispatched once Shutdown has begun. See
  FrbGrouperService / FrbRpcService for the model. Corollary: a teardown
  path that blocks on handler drain (Shutdown/Wait) must never itself be
  reachable from a handler thread -- document that invariant where it
  applies (e.g. "no RPC handler may call stop()" when stop() Shutdowns).

- In a worker's catch-all wrapper, everything that can throw (argument
  lookups like vector::at, logging via stringstream) belongs INSIDE the
  try. Must-always-run bookkeeping (e.g. an exit counter that a wait()
  method depends on) stays outside, and must be unable to throw. An
  exception escaping a thread is std::terminate.

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
    // Stop-pattern state is 'mutable' and stop() is 'const' (see
    // notes/stoppable_class.md).
    mutable std::mutex mutex;
    mutable std::condition_variable cv;
    mutable bool is_stopped = false;
    mutable std::exception_ptr error;

    std::queue<int> queue;
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
    
    // Noncopyable, nonmovable.
    X(const X &) = delete;
    X &operator=(const X &) = delete;
    X(X &&) = delete;
    X &operator=(X &&) = delete;

    void stop(std::exception_ptr e = nullptr) const
    {
        std::unique_lock lock(mutex);
        if (is_stopped) return;
        is_stopped = true;
        error = e;
        lock.unlock();
        cv.notify_all();
    }

    void queue_request(int id) 
    {
        std::unique_lock lock(mutex);
        _throw_if_stopped("X::queue_request");
        queue.push(id);
        lock.unlock();
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

