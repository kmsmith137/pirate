# Stoppable Class Pattern

In many real-time systems, it's a goal to recover gracefully if something goes wrong.
Generally speaking, we have the opposite philosophy: if something goes wrong, we want to crash with a helpful log message, so that an operator notices the failure and investigates.
(Some cases are exceptions -- for example if a few "upstream" X-engine nodes go down, then we want to gracefully continue processing. But anything that indicates a bug, including an "upstream" X-engine bug, should produce a crash.)

In the code, an uncaught exception in one thread should gracefully shut down all threads, with a helpful exception-text message.
The "stoppable class" X is a code pattern for cascading exceptions through all threads:

- `X` is thread-safe.

- `X` has a `stop(std::exception_ptr e)` method which can be called externally from any thread, to put the object into a "stopped" state (`X::is_stopped==true`). The value of `e` is saved in `X::error`, and is null or non-null depending on whether the call to `stop()` represents normal termination. The first caller to `stop()` sets `X::error`. `stop()` is declared `const`, and the stop-pattern state (mutex, condition variables, `is_stopped`, `error`) is declared `mutable`: stoppability is control-plane state (analogous to a mutable mutex), and `const` entry points / accessors must be able to stop the object.

- Some public methods of `X` are labelled as "entry points". If an entry point is called in the stopped state, the exception `e` is rethrown. (If `e` is `nullptr`, then a generic exception `runtime_error("X::f() called on stopped instance")` is thrown -- always with the class-qualified method name, passed to the `_throw_if_stopped` helper as in the example below.)

- If an entry point throws an exception, then `X::stop(e)` is called, and the exception is rethrown. This is a strict policy: it applies to ALL throws, including routine argument-checking. (Rationale: a stoppable object is shared real-time infrastructure; an entry-point argument error means some pipeline thread has a bug, and the system is safest fully stopped, with the offending error preserved as the exception-text.) Implement this by wrapping the entry-point body in a try/catch that calls `stop(std::current_exception())` and rethrows, as illustrated by `example_entry_point()` in the example code below.

- Methods that must remain usable on a stopped object are NOT entry points, and may throw without stopping: `stop()` itself, `is_stopped`-style accessors, `join()`/`wait()`-style methods (including `FrbServer::poll_from_python()`), and methods documented as informational accessors (e.g. `FakeXEngine::get_minichunk_status()`).

- Classify every public method in the header: entry point, or stopped-tolerant accessor. Rule of thumb: methods that can block are entry points; instantaneous snapshots may be stopped-tolerant (their last-known values stay meaningful for diagnostics). An unlabeled method reads as an oversight to a reviewer, and a missing `_throw_if_stopped` in an accessor body deserves a one-line "deliberate" comment (see SlabAllocator for the model).

- In a context where the value of `is_stopped` is checked (e.g. entry point), if a blocking call is made (e.g. `cv.wait()`) then `is_stopped` is rechecked on wakeup. `X::stop()` should call `notify_all()` on each of `X`'s condition variables. In situations where this notification mechanism doesn't work, please find an alternative if possible. (For example, a network worker thread which needs to block waiting on a socket could specify a 1-ms timeout, in order to recheck `is_stopped` every millisecond.)

- `X` is noncopyable, nonmoveable (delete all four special members explicitly), and normally accessed through a shared_ptr (pybind11 bindings use a shared_ptr holder -- see "pybind11 bindings" below). Rare exceptions -- a by-value member, or a stack-local instance in a single-threaded timing loop -- are acceptable when the owner provably outlives all users, and should say so in a comment.

- If `X` contains pointers to other stoppable classes (any class `Y` defining `Y::stop()` is probably stoppable, please ask if it's unclear), then `X::stop(e)` should call `ptr->stop(e)` for each such pointer, forwarding the exception (see "Error reporting" below).

- If a thread throws and nothing above it catches, its catch-all wrapper calls `stop(std::current_exception())` on the stoppable object(s) it works for, before exiting (see the worker wrapper in [notes/thread_backed_class.md](thread_backed_class.md)). (Not shown in the example code below.)

The idea is that there are enough stoppable classes shared between threads that an exception in any thread will cascade its exception-text into all threads, and the server will shut down cleanly (all threads will exit).

## Error reporting

Conventions for how the exception text reaches a human:

- Cascades forward the exception: `X::stop(e)` calls `ptr->stop(e)`, not
  `ptr->stop()`. A thread that is blocked inside a downstream object then
  rethrows the root-cause exception, rather than a generic
  "called on stopped instance" message.

- Worker threads never print exceptions. A worker's catch-all wrapper just
  calls `stop(std::current_exception())` and exits. The first `stop(e)` caller
  wins: its exception is saved in `X::error` and rethrown by every subsequent
  entry-point call. (Corollary: an error that loses the first-caller race is
  dropped. This is deliberate -- the first error is the root cause, and later
  errors are usually downstream consequences of it.)

- The saved error ultimately surfaces as a python exception. For code that
  naturally blocks in entry points (e.g. a consumer loop calling
  `get_frame_set()`), nothing extra is needed. A python-driven server whose
  driver does NOT naturally block in an entry point should provide a method
  like `FrbServer::poll_from_python(timeout_ms)`: block until stopped or
  timeout; on stop, rethrow the saved error (a clean stop returns normally).
  The pybind11 binding releases the GIL, and python calls it in a loop with a
  short timeout (~500 ms) -- a fully-blocking call would never return to the
  interpreter, so Ctrl-C (KeyboardInterrupt) could never be delivered.

- A null stop() means normal termination; a non-null stop(e) means error
  shutdown. Blocking methods for which "stopped while waiting" is an EXPECTED
  outcome may adopt the poll_from_python() convention -- return a "done"
  value on a clean stop, rethrow the saved error on an error stop -- instead
  of throwing the generic "called on stopped instance" message. (Examples:
  FakeXEngine's enqueue_* / wait_until_processed / synchronize return false;
  SimulatedFrameFactory::get_frame_set() returns nullptr / Python None. A
  controller loop then terminates cleanly on the done-value, and exceptions
  are reserved for real errors.) Methods for which a stopped instance implies
  caller misuse should keep the throwing convention.

- Never make control-flow decisions by string-matching exception text (e.g.
  filtering on "called on stopped instance"). If code needs to distinguish a
  benign teardown unwind from a real error, make the distinction structural:
  a done-value return, poll_from_python(), or the null-vs-non-null test on
  the stored exception_ptr.

- Stop-ordering invariant for external teardown code (e.g. a python-level
  "stop everything" sweep): stop a holder object before its dependency,
  never the reverse. A holder thread blocked inside a stopped dependency
  unwinds via catch -> stop(current exception); that call is a harmless
  no-op only if the holder's own stop already won the first-caller race.
  Null-stopping a dependency under a still-running holder records the unwind
  as a bogus error on the holder. (Internal cascades already satisfy this:
  X::stop(e) forwards e down into its dependencies.)

## Locking and condition variables

The general rules -- one condition variable per wait-predicate, when
`notify_one()` is sound, complete "signaled on:" lists, snapshot-under-lock,
in-progress flags reset on every exit path, `long` counters in predicates,
documenting uninterruptible waits -- live in the "Concurrency" section of
[notes/cpp.md](cpp.md) and apply to every class in this pattern. In stoppable-class
terms, the "shutdown event notifies every cv" rule there is `stop()`, and
each cv's "signaled on:" list therefore includes `stop()`.

## pybind11 bindings

- Stoppable classes use a shared_ptr holder:
  `py::class_<X, std::shared_ptr<X>>`.

- Any binding that can block (entry points that wait, `poll_from_python`)
  releases the GIL via `py::call_guard<py::gil_scoped_release>()` -- the
  waker may be another python thread, which can never run while the blocked
  binding holds the GIL. See [notes/pybind11.md](pybind11.md) for the full GIL rules.

- Never `def_readonly` a lock-protected member: the read is unsynchronized
  (a torn vector copy in the worst case). Route through a lock-taking
  getter. Members published by a flag convention instead of a lock
  (handler writes, then sets a mutexed "done" flag) may be `def_readonly`,
  but document the convention and when reads are safe, at the binding.

Here is a toy example of a stoppable class X with a fixed-size ring buffer, and two entry points `push()` and `pop()`.
If `push()` is called and the ring buffer is full, the caller blocks until space is available.
If `pop()` is called and the ring buffer is empty, the caller blocks until it becomes nonempty.

## Example Code

```cpp
#include <condition_variable>
#include <exception>
#include <mutex>
#include <array>
#include <string>

class X {
    static constexpr int rb_size = 5;
    
    // The stop-pattern state is 'mutable', and stop() is declared 'const':
    // stoppability is control-plane state (like a mutex), and const entry
    // points / accessors must be able to stop the object.
    mutable std::mutex mutex;

    // Two condition variables, one per wait predicate.
    // cv_nonempty -- waiters: pop() (predicate: ring buffer nonempty, or
    //   stopped). Signaled on: push() (notify_one), stop().
    // cv_nonfull -- waiters: push() (predicate: ring buffer nonfull, or
    //   stopped). Signaled on: pop() (notify_one), stop().
    mutable std::condition_variable cv_nonempty;
    mutable std::condition_variable cv_nonfull;

    mutable bool is_stopped = false;
    mutable std::exception_ptr error;

    std::array<int,rb_size> ringbuf;
    long rb_start = 0;
    long rb_end = 0;

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
        cv_nonempty.notify_all();
        cv_nonfull.notify_all();
    }

    // Typical entry point, illustrating the "call stop() if exception is thrown" pattern.
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

    // Note: even though push() is an entry point, we don't bother wrapping in try..catch
    // and calling stop(), since the only exception-throwing path is via _throw_if_stopped(),
    // which only throws an exception if stop() has already been called.
    
    void push(int x)
    {
        std::unique_lock lock(mutex);

        for (;;) {
            _throw_if_stopped("X::push");
            
            if (rb_end < rb_start + rb_size) {
                ringbuf[(rb_end++) % rb_size] = x;
                lock.unlock();
                // notify_one is sound: all cv_nonempty waiters share the
                // predicate, and one pushed item satisfies exactly one pop()
                // (work-queue handoff).
                cv_nonempty.notify_one();
                return;
            }
            
            cv_nonfull.wait(lock);
        }
    }

    // Note: even though pop() is an entry point, we don't bother wrapping in try..catch,
    // for the same reason as in push() above.
    
    int pop()
    {
        std::unique_lock lock(mutex);

        for (;;) {
            _throw_if_stopped("X::pop");
            
            if (rb_start < rb_end) {
                int ret = ringbuf[(rb_start++) % rb_size];
                lock.unlock();
                // notify_one is sound: one freed slot admits exactly one
                // push() (work-queue handoff).
                cv_nonfull.notify_one();
                return ret;
            }
            
            cv_nonempty.wait(lock);
        }
    }
};
```
## Reviewer checklist

When reviewing a class against this pattern, these are the failure modes to
look for -- each occurred at least once in a past full review of the
stoppable/thread-backed classes:

1. Entry point missing the try/catch stop-and-rethrow wrapper, or a throw
   path (argument check, precondition xassert) sitting BEFORE the wrapper /
   outside the try.
2. Entry point that silently succeeds, or returns stale data, on a stopped
   instance -- with no comment claiming that's deliberate; or stopped
   behavior that differs across modes (e.g. dummy vs normal).
3. A stop() cascade that drops the exception: `ptr->stop()` instead of
   `ptr->stop(e)`.
4. Shared cv + notify_one() with waiters on different predicates (lost
   wakeup); or a state change with no notify at all (missed wakeup).
5. A "drained"/"idle" barrier whose predicate misses in-flight work
   (popped from the queue but not yet fully dispatched).
6. State read after unlock, or published outside the mutex -- thread
   handles and grpc server handles are the recurring case (see the
   "Concurrency" section of notes/cpp.md).
7. An in-progress flag not reset on a throw path, or reset without
   notifying the cv whose waiters test it.
8. A blocking wait that stop() cannot interrupt, undocumented.
9. Control flow that string-matches exception text.
10. Comment/code drift: stale method names, incomplete "signaled on" lists,
    absolute claims ("can never hang", "only returns if stopped") that the
    code does not implement. Verify every synchronization comment against
    the code -- in past review, stale comments were about as common as real
    bugs, and each one misleads the next reviewer.
11. pybind11: missing shared_ptr holder; missing GIL release on a blocking
    binding; `def_readonly` of a lock-protected member.

Closely related: the "thread-backed class" is a special case of a stoppable class, in which objects have one or more worker threads
whose lifetime is tied to the object lifetime. Every thread-backed class is stoppable, but not vice versa. See notes/thread_backed_class.md
for more info.

