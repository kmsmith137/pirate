# C++ guidelines

- Prefer private (or static) functions over anonymous scopes `{ ... }`.

- Use `//` comments, not `/* */`. Exception: `/* */` is a good choice for inline
  argument-name annotations at a call site, e.g. `f(/*count=*/3, /*noisy=*/true)`.

- Use `long` for sizes and indices, not `int` or `size_t`

- Use spaces, not tabs.

- Functions which take an `ostream &` argument should not modify stream state in the caller.

- If a class `X` derives from `std::enable_shared_from_this`, then its constructor(s) must be protected, and the class must define `create()` static method(s) that return `shared_ptr<X>`.

- Avoid lambdas inside templates defined in `.hpp` files that are instantiated
  from multiple `.cpp` files (nvcc may miscompile these).

- Minimize header `#include` dependencies. In a `.hpp`, if a type is only used by
  pointer, reference, or `std::shared_ptr<T>` (not by value, as a base class, or
  via member access / `sizeof`), forward-declare it instead of `#include`-ing its
  header, and move the `#include` into the `.cpp` file(s) that use the full type.
  This cuts recompilation when the included header changes. Match the declared
  keyword (`class` vs `struct`) and note where it lives, e.g.:

  ```
  class SlabAllocator;     // defined in SlabAllocator.hpp
  struct XEngineMetadata;  // defined in XEngineMetadata.hpp
  ```

  Caveat: keep the full `#include` in the header when the complete type is needed
  there -- a base class, a by-value member, an inline function that touches the
  type, or a `std::unique_ptr<T>` member of a class whose destructor is
  compiler-generated in the header (`unique_ptr` needs a complete type at the
  destructor; `shared_ptr` does not). When unsure, include it.

## Concurrency

Rules for any threaded C++ code; violations of each one turned up as real
bugs in review. (In this codebase, threaded classes should additionally
follow the stoppable / thread-backed patterns -- see notes/stoppable_class.md
and notes/thread_backed_class.md, which build on these rules and add
shutdown/error-propagation semantics.)

### Locking and condition variables

- One condition variable per wait-predicate. `notify_one()` is allowed ONLY
  when (a) every waiter on that cv has the same predicate and one event
  satisfies exactly one waiter (work-queue handoff), or (b) at most one
  thread can ever be blocked on that cv (structurally single waiter) --
  either way, write the justification as a comment at the notify site. If
  waiters with different predicates share a cv, a targeted notify can wake
  the wrong waiter while the intended one sleeps forever (lost wakeup).
  When in doubt, split the cv or use `notify_all()`. One-shot latch events
  (init flags) keep `notify_all()` -- the cost is paid once, and it is
  robust against waiters added later. A shutdown/stop event must
  `notify_all()` every cv of the class. A wait on a CONJUNCTION of set-once
  latch flags does not force a shared cv: wait for each flag sequentially,
  each on its own cv (equivalent because the flags are never cleared -- see
  AssembledFrameAllocator's worker init gate). FrbServer (7 cvs) is the big
  worked example; CudaEventRingbuf's `_release()` shows how to justify NOT
  notifying a cv whose waiters mention the changed state.

- Keep a COMPLETE "signaled on:" list next to each cv declaration. A wait
  predicate is correct only if every state change it tests is followed by a
  notify; an incomplete list hides missed-notify bugs (e.g. draining a
  queue on a side path without notifying the waiter whose predicate just
  became true). Reviewers should check the list against the code.

- Notify with the mutex RELEASED where cleanly possible -- otherwise a
  woken thread immediately blocks re-acquiring the lock the notifier still
  holds. This is a throughput nicety, never a correctness requirement, so
  it never justifies weakening a critical section -- and legitimate
  under-lock notifies exist and deserve no "fix": a lock deliberately held
  into the next loop iteration (FileWriter's ssd->nfs handoff), a
  caller-owned guard (AssembledFrameAllocator::_create_frame_set), a
  failure path where the lock must stay held (the throw in
  FrbGrouper::start_listening).

- Never read a lock-protected member after dropping the lock -- snapshot it
  into a local under the lock and use the local.

- An "in progress" flag that other threads wait on (e.g. an init-underway
  or creation-underway flag) must be reset on EVERY exit path, including
  throws -- use a catch block or a scope guard. A flag left set wedges all
  future waiters, even if some other coupling (e.g. an error path that
  shuts down the whole object) happens to mask it today. And the reset is
  itself a state change that waiters' predicates test: the reset path --
  including a scope guard's unwind path -- must notify the corresponding
  cv, and that cv's "signaled on:" list must include it. (See
  AssembledFrameAllocator's CreationFlagGuard.)

- Counters used in wait predicates are `long`, never `int`: signed overflow
  is UB and a hung predicate in a long-running server.

- If a blocking call genuinely cannot be interrupted by your shutdown
  mechanism (e.g. `cudaEventSynchronize` vs the stoppable pattern's
  `stop()`), document the trade-off at the declaration: what invariant
  bounds the wait in practice, and what happens in the pathological case.
  (See the `blocking_sync` note in CudaEventRingbuf.hpp for the model.)

### Spawning, joining, and teardown

- An exception escaping a thread is `std::terminate`. Every thread's main
  function is a catch-all wrapper, and everything that can throw (argument
  lookups like `vector::at`, logging via stringstream) belongs INSIDE the
  try. Must-always-run bookkeeping (e.g. an exit counter that a `wait()`
  method depends on) stays outside, and must be unable to throw. (See
  HwtestSender::worker_main for the model.)

- Publish thread handles (and similar teardown state, e.g. a grpc server
  handle) UNDER the mutex. If threads are spawned outside the constructor
  (in a `start()` / `open()` / `allocate()`), the handle assignments happen
  with the mutex held, and every teardown path (stop/join/close/destructor)
  synchronizes on the mutex before reading them -- teardown can run
  concurrently with startup. Holding the lock across the spawns is safe as
  long as the spawner never waits on the spawned threads: a freshly-spawned
  worker that touches shared state just blocks briefly until the spawner
  releases.

- Join paths take the mutex briefly (to synchronize with that publication),
  then join with the mutex RELEASED -- workers take the mutex on their exit
  paths, so joining under it deadlocks.

- A constructor that spawns SEVERAL threads must handle a mid-spawn
  failure: catch, shut the workers down, join the already-spawned threads,
  rethrow. Otherwise the `std::thread` members are destroyed while
  joinable, which is `std::terminate`. (See FakeXEngine's constructor for
  the model.)

- The destructor joins threads inline, and must never call a public
  `join()`-style method that rethrows a saved error (a throwing
  destructor). If both the destructor and a public `join()` need the
  joining loop, factor a private `_join_threads()` helper (see Hwtest).

- A `close()`-style teardown method needs more than an idempotency flag: a
  second concurrent caller must BLOCK until the first finishes (e.g. a
  leaf-level `close_mutex` held for the whole body -- see FrbGrouper). With
  a flag alone, the second caller -- often the destructor -- returns
  immediately and destroys members out from under the first caller's
  still-running teardown.

## ksgpu

Uses the `ksgpu` helper library, especially the Array class (ksgpu/Array.hpp), memory managment (ksgpu/mem_utils.hpp),
and xassert macros (ksgpu/xassert.hpp, see below).

CRITICAL: please complain if you don't see the ksgpu library in the cursor/claude workspace.

## xassert macros

The following macros are similar to `assert()`, but throw an exception instead of terminating:
```
xassert(cond);           // throw exception unless bool(cond)==true
xassert_eq(x,y);         // throw exception unless x==y
xassert_divisible(x,y);  // throw exception unless (x % y) == 0
// also: xassert_ne(), xassert_lt(), xassert_le(), xassert_gt(), xassert_ge()

// Throw exception unless 'arr' (type ksgpu::Array) has shape {3,4,5}.
// IMPORTANT: note parentheses around the shape -- these are needed to compile!
xassert_shape_eq(arr, ({3,4,5}));
```
The exception text shows the file/line (like regular `assert()`), plus values of the arguments `x` and `y`.
In the case of `xassert_shape_eq()`, both array shapes are shown.

Please use `xassert()` for argument-checking and error-checking, unless there is a reason to create a more verbose error message.
For example:
```
void f(int x, int y, int z)
{
    // Example 1: you should replace this by xassert_eq(x,y), since the xassert_eq() error message 
    // contains the same information as the message below (namely, numerical values of x and y).

    if (x != y) {
        stringstream ss;
        ss << "f(): expected x==y, got x=" << x << ", y=" << y;
        throw runtime_error(ss.str());
    }

    // Example 2: you should not replace this by xassert_lt(x+y,z), since the xassert_lt() error message
    // contains less information than the message below (which shows x,y,z individually)

    if ((x+y) >= z) {
        stringstream ss;
        ss << "f(): expected (x+y) < z, got x=" << x << ", y=" << y << ", z=" << z;
        throw runtime_error(ss.str());
    }

    // Example 3: you should replace this by xassert_shape_eq(arr, ({M,N})), since the xassert_shape_eq()
    // message contains more information (namely, the actual and expected array shapes).

    if ((arr.ndim != 2) || (arr.shape[0] != x) || (arr.shape[1] != y)) {
        stringstream ss;
        ss << "f(): expected shape (x,y) = (" << x << "," << y << "), got " << arr.shape_str();
        throw runtime_error(ss.str());
    }
}
```
