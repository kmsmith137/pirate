# CudaEventRingbuf

This class is used ubiquitously, to chain kernel launchers together.
A single producer produces `CudaEvents`, and multiple consumers consume them.
Events can be consumed in two ways: either by synchronizing a stream on the event (`wait()`),
or synchronizing a host thread on the event (`synchronize()`).
Each event has a `seq_id = 0,1,...` which must be consecutive.

## Class definition (simplified)

```
struct CudaEventRingbuf
{
    CudaEventRingbuf(const std::string &name, int nconsumers, long max_size=1000);

    // Producer: records a cuda event from 'stream', and saves it in the ring buffer.
    // If there is no room in the ring buffer, then the 'blocking' argument determines
    // whether the calling thread blocks (blocking=true) or throws an exception (false);
    
    void record(cudaStream_t stream, long seq_id, bool blocking = false);

    // Consumer: retrieve event from ringbuf, and call cudaStreamWaitEvent(stream, event).
    // If seq_id < 0, this is a no-op.
    //
    // If the seq_id has not yet been produced (via record()):
    //   - If blocking=false (default), throws an exception.
    //   - If blocking=true, the calling thread blocks until another thread produces the seq_id.
    
    void wait(cudaStream_t stream, long seq_id, bool blocking = false);

    // Consumer: retrieve event from ringbuf, and call cudaEventSynchronize(event).
    // If seq_id < 0, this is a no-op.
    //
    // If the seq_id has not yet been produced (via record()):
    //   - If blocking=false (default), throws an exception.
    //   - If blocking=true, the calling thread blocks until another thread produces the seq_id.

    void synchronize(long seq_id, bool blocking = false);
};
```

## Example

Usage of `CudaEventRingbuf` is best explained by example.

Consider the following example, with two host threads (named ``h2g'' and ``g2h'')
and three GPU kernels (f,g,h), and four GPU ring buffers (in,x,y,out):

  - The h2g thread runs in a loop: data "batches" arrive unpredictably
    by calling a function `get_data()`. When a batch of data arrives, it is copied
    from host memory to a GPU ring buffer ``in``, which can hold Nin batches.
  
  - When the data arrives in the ``in`` buffer, two GPU kernels can be launched:

      - ``f()`` whose input is an `in` batch, and whose output is an  `x` batch.
        Here, `x` is a GPU ring buffer which can hold `Nx` batches.

      - ``g()`` whose input is an `in` batch, and whose output is an  `y` batch.
        Here, `y` is a GPU ring buffer which can hold `Ny` batches.

  - When the ``f()`` and ``g()`` kernels have completed, a third GPU kernel can
    be launched:

      - ``h()`` whose inputs are `x` and `y` batches (two inputs) and whose
        output is an `out` batch. Here, `out` is a GPU ring buffer which can
        hold `Nout` batches.

  - The g2h thread runs in a loop: when output "batches" are ready in the
    `out` ring buffer, it copies them to the CPU and calls a host function
    `process_data()`.
  
  - In this toy example, we assumed that all four ring buffers have independent
    sizes `Nin`, `Nx`, `Ny`, `Nout`. We'll also assume that all three kernels
    (plus the h2g and g2h copies) execute on five different CUDA streams
    `sf`, `sg`, `sh`, `sin`, and `sout`. These assumptions are artificial,
    but help illustrate generality.

This example could be implemented as follows.

```cpp
    // One CudaEventRingbuf for each kernel which is launched (including copy kernels).
    // Reminder: constructor syntax is CudaEventRingbuf(name, nconsumers, max_size).
    //
    // Determining the parameters (nconsumers, max_size) is nontrivial!
    // When I wrote the code below, I left placeholders for these parameters.
    // When the rest of the code was finished, I went back and filled in the
    // parameter values, by revisiting each CudaEventRingbuf individually:
    //
    //   evrb_h2g: Immediately consumed (twice) in same thread
    //               =>  nconsumers=2, max_size=1
    //
    //   evrb_f: Consumed twice in the same thread. First consumer is earlier in the
    //           seq_id loop, but has a lag (seq_id-Nin). Second consumer is immediate.
    //               => nconsumers=2, max_size=Nin
    //
    //   evrb_g: Same story as evrb_f
    //               => nconsumers=2, max_size=Nin
    //
    //   evrb_h: Three consumers. Two are in the same thread, earlier in the seq_id
    //           loop, but with lags (seq_id-Nx) and (seq_id-Ny). This gives a lower
    //           bound max(Nx,Ny) on max_size, needed to avoid a deadlock(!). The
    //           third consumer is in another thread, so the choice of max_size is
    //           flexible (just provides scheduling "slack" between the two threads).
    //           I decided to use Nout as a guess for the amount of "slack" that might
    //           be appropriate.
    //               => nconsumers=3, max_size=max(max(Nx,Ny), Nout)
    //           
    //   evrb_g2h: Two consumers. One is immediate in the same thread, and the other
    //             is on another thread. Similarly, I decided to use Nout here.
    //               => nconsumers=2, max_size=1
    
    auto evrb_h2g  = make_shared<CudaEventRingbuf> ("h2g", 2, 1);
    auto evrb_f    = make_shared<CudaEventRingbuf> ("f", 2, Nin);
    auto evrb_g    = make_shared<CudaEventRingbuf> ("g", 2, Nin);
    auto evrb_h    = make_shared<CudaEventRingbuf> ("h", 3, max(max(Nx,Ny), Nout));
    auto evrb_g2h  = make_shared<CudaEventRingbuf> ("g2h", 2, Nout);

    void h2g_thread()
    {
        for (long seq_id = 0; true; seq_id++) {
            get_data();  // blocks until 'host_batch' is available in host memory

            // Launch the h2g copy kernel.
            //
            // Generally speaking, before launching kernels, we need synchronization
            // logic to ensure that their source and destination buffers are available.
            // Here, the source buffer 'host_batch' is guaranteed available, but
            // the destination slot in the 'in' ring buffer is not available until
            // the kernels f(seq_id-Nin) and g(seq_id-Nin) have completed.
            //
            // Note: blocking=false is okay here, since f,g events were already
            // produced by the same thread in a previous loop iteration. See below
            // for systematic discussion.

            evrb_f->wait(s_h2g, seq_id-Nin, /*blocking=*/ false);
            evrb_g->wait(s_h2g, seq_id-Nin, /*blocking=*/ false);
            cudaMemcpyAsync(in[seq_id % Nin], host_batch, s_h2g);  // schematic syntax

            // Record event for the h2g memcpy, so that other kernels can wait on it.
            // Note: blocking=false is okay here, since h2g events are consumed in
            // in the same thread. See below for systematic discussion.
            
            evrb_h2g->record(s_h2g, seq_id, /*blocking=*/ false);

            // Launch the f() kernel. Here, we need to wait on:
            //   source buffer: h2g(seq_id)
            //   destination buffer: h(seq_id - Nx)
            
            evrb_h2g->wait(s_f, seq_id, /*blocking=*/ false);
            evrb_h->wait(s_f, seq_id-Nx, /*blocking=*/ false);
            launch_f(x[seq_id % Nx], in[seq_id % Nin], s_f);     // schematic syntax
            evrb_f->record(s_f, seq_id, /*blocking=*/ false);

            // Same story for g() kernel.
            
            evrb_h2g->wait(s_g, seq_id, /*blocking=*/ false);
            evrb_h->wait(s_g, seq_id-Ny, /*blocking=*/ false);
            launch_g(x[seq_id % Ny], in[seq_id % Nin], s_g);     // schematic syntax
            evrb_g->record(s_g, seq_id, /*blocking=*/ false);

            // Launch the h() kernel. Here, we need to wait on:
            //    source buffers: f(seq_id), g(seq_id)
            //    destination buffer: g2h(seq_id - Nout)
            //
            // Note that we need blocking=true in couple of places here:
            // when consuming an event that was produced on another thread,
            // and when producing an event that will be consumed in another
            // thread. See below for systematic discussion.

            evrb_f->wait(s_h, seq_id, /*blocking=*/ false);
            evrb_g->wait(s_h, seq_id, /*blocking=*/ false);
            evrb_g2h->wait(s_h, seq_id - Nout, /*blocking=*/ true);
            launch_h(out[seq_id % Nout], x[seq_id % Nx], y[seq_id % Nx], s_h);
            evrb_h->record(s_h, seq_id, /*blocking=*/ true);

            // The h2g thread stops processing here.
            // Processing will resume on the g2h thread, which will wait on evrb_h.
        }
    }

    void g2h_thread()
    {
        // In this example, we assume that the g2h thread can hold a single
        // 'out' batch ("local_batch") in host memory, after copying from GPU.
        
        for (long seq_id = 0; true; seq_id++) {
            // Launch the g2h copy kernel, but kernel waits on the h() event.
            // Note that we need blocking=true here: the h() events are produced
            // by a different thread, so there's no guarantee that the events are
            // produced before we call evrb_h->wait().

            evrb_h->wait(s_g2h, seq_id, /*blocking=*/ true);
            cudaMemcpyAync(local_batch, out[seq_id % Nout], s_g2h);  // schematic
            evrb_g2h->record(s_g2h, seq_id, /*blocking=*/ true);

            // Wait for data to arrive, and process it on the CPU.
            evrb_g2h->synchronize(seq_id, /*blocking=*/ false);
            process_data(local_batch);

            // 'local_batch' can be reused now. Back to top of loop.
        }
    }
```
Comments on this example:

  - A reasonable baseline design is to have one CudaEventRingbuf for
    every "kernel" (either a compute kernel or a copy kernel).

    Before launching a kernel, make sure that all source buffers are ready
    (usually this means calling CudaEventRingbuf::wait(seq_id)).

    Before launching a kernel, make sure that all destination buffers are ready
    (usually this means calling CudaEventRingbuf::wait(seq_id - ringbuf_length)).

    After launching a kernel, call CudaEventRingbuf::record(seq_id).
    
  - General comments on the `blocking` argument to `CudaEventRingbuf::wait()`:

    In some situations, we know that the event has been produced (e.g.
    producer is a previous loop iteration on the same thread). In such
    situations, it's usually better to set `blocking=false`, so that we
    get an exception (facilitating debugging) if the event is not present.

    In other situations (e.g. event produced by another thread), setting
    `blocking=true` is logically necessary.

  - General comments on the `blocking` arugment to `CudaEventRingbuf::record()`:

    In some situations, we know that all consumers wait on the event with an
    upper bound on the lag (e.g. all consumers are in previous loop iterations
    on the same thread). In such situations, it's usually better to set
    `blocking=false`.

    In other situations (e.g. one or more consumers are on different threads),
    setting `blocking=true` is logically necessary.
    
  - In the h2g path, we implicitly assumed that the lifetime of the `host_batch`
    was infinite. In a real system, we would probably want to keep a reference
    to the `host_batch` during the g2h copy, and drop the reference when the
    copy is complete. One way to implement this would be to have another thread
    which consumes events from `evrb_g2h` and drops a reference after each event
    is consumed. (Note that `evrb_g2h::num_consumers` would increase by 1, and
    the call to `evrb_g2h->record()` would need `blocking=true`.)
    
  - Similarly, in the g2h path, we assumed for simplicity that the `g2h_thread`
    could only store a single batch in host memory. If we wanted to put a ring
    buffer in host memory instead, this could be implemented by having two
    `g2h_threads`: one to wait for space in the host ring buffer and launch
    the g2h copy kernel (calling `evrb_g2h->record()`), and one to
    call `evrb_g2h->sychronize()` and process the data.

    The point of the previous two bullet points is that `CudaEventRingbuf`
    is general enough to implement a wide range of syncrhonization schemes
    between host threads and gpu kernels.
  
If you are writing, reviewing, debugging code involving `CudaEventRingbufs`, here
is a good mental checklist:

  - Before each launch, double-check that you've waited on all source buffers,
    and all destination buffers. Double-check that you've included the proper
    lags (sometimes you'll call `evrb->wait()` with `(seq_id - lag)`, where
    `lag` could be the size of a ring buffer).

  - After each launch, make sure that you've called CudaEventRingbuf::record().

  - Check that the `blocking` arguments to `CudaEventRingbuf::wait()` and
    `CudaEventRingbuf::record()` are consistent with the above discussion.
    
  - Check that when the `CudaEventRingbuf` is constructed, the value of
    `num_consumers` matches the number of callers of `wait()` and/or
    `synchronize()`.

  - Setting the proper value of `CudaEventRingbuf::max_size` is tricky!!

    In situations where the producer and consumer(s) are all on the same
    thread, then there may be a max lag between the producer and the slowest
    consumer. (For example, `evrb_f` and `evrb_g` in the example.) Then
    there will be a clear choice of `max_lag`.

    In more complex situations, you'll want to analyze individual consumers.
    A good example is the `evrb_h` thread above, where `max_size >= max(Nx,Ny)`
    is needed to avoid crashes in two of the consumers (the ones that call
    `evrb_h->wait()` with `blocking=false`, on the same thread as the producer).
    The third consumer runs on a different thread and calls `evrb_h->wait()`
    with `blocking=true`. In this case, any value of `max_lag` is correct,
    but a larger value leaves more scheduling "slack" between threads.
    I ended up with `max_size=max(max(Nx,Ny), Nout)` as a heurstic.
    
    

