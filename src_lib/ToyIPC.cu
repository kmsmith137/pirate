#include "../include/pirate/ToyIPC.hpp"

#include "../grpc/frb_grouper.grpc.pb.h"
#include "../grpc/frb_grouper.pb.h"
#include <grpcpp/grpcpp.h>

#include <ksgpu/cuda_utils.hpp>   // CUDA_CALL
#include <ksgpu/xassert.hpp>      // xassert, xassert_eq, ...

#include <cuda_runtime.h>         // cudaIpcGetMemHandle, cudaIpcMemHandle_t

#include <cstring>                // strstr
#include <iostream>
#include <random>
#include <sstream>
#include <stdexcept>

using namespace std;
using namespace ksgpu;

namespace fg = frb::grouper::v1;


namespace pirate {
#if 0
}   // editor auto-indent
#endif


// gRPC client state. Hidden behind pImpl so the .hpp does not pull in
// grpc++ headers; sized/laid-out only here.
struct ToyIPC::GrpcState {
    std::shared_ptr<grpc::Channel>            channel;
    std::unique_ptr<fg::ToyIpcStream::Stub>   stub;
    std::unique_ptr<grpc::ClientContext>      context;
    std::unique_ptr<grpc::ClientReaderWriter<fg::ProducerMessage, fg::ConsumerMessage>> stream;
};


// Helper: "called on stopped instance" is the normal cascade-stop
// exception thrown by entry points after a stop(). We don't want to
// flood stderr with these from the worker-thread wrappers.
static bool _is_cascade_stop_exception(const std::exception &e)
{
    return std::strstr(e.what(), "called on stopped instance") != nullptr;
}


// -------------------------------------------------------------------------------------------------
//
// Construction


std::shared_ptr<ToyIPC> ToyIPC::create(const std::string &server_address, int cuda_device_id)
{
    // Can't use make_shared (private constructor).
    auto p = std::shared_ptr<ToyIPC>(new ToyIPC(server_address, cuda_device_id));
    p->_start(server_address);
    return p;
}


ToyIPC::ToyIPC(const std::string & /*server_address*/, int cuda_device_id_)
    : cuda_device_id(cuda_device_id_),
      rng(std::random_device{}())
{
    xassert(cuda_device_id >= 0);
}


// Construction body. Runs after the shared_ptr exists. Any exception
// thrown here propagates out of create(); since the worker thread has
// not yet been started, no joining is needed -- the default destructors
// of grpc_state / ringbuf clean up.
void ToyIPC::_start(const std::string &server_address)
{
    xassert(server_address.size() > 0);

    // 1. Pin to the GPU. All later cudaMalloc / cudaIpcGetMemHandle /
    //    cudaMemcpy calls bind to the current device of this thread.
    CUDA_CALL(cudaSetDevice(cuda_device_id));

    // 2. Allocate the ring buffer. af_alloc(af_gpu) routes through
    //    cudaMalloc (verified in ksgpu/src_lib/mem_utils.cpp). Plain
    //    cudaMalloc is IPC-eligible; cudaMallocAsync would NOT be.
    this->ringbuf = Array<float>({5,2}, af_gpu | af_zero);
    xassert(ringbuf.data != nullptr);

    // 3. Export the cudaIpcMemHandle_t. The handle is a 64-byte POD.
    cudaIpcMemHandle_t ipc_handle;
    CUDA_CALL(cudaIpcGetMemHandle(&ipc_handle, static_cast<void *>(ringbuf.data)));
    static_assert(sizeof(cudaIpcMemHandle_t) == 64,
                  "cudaIpcMemHandle_t is expected to be exactly 64 bytes");

    // 4. Bring up gRPC: channel + stub + context + bidi stream.
    auto gs = std::make_unique<GrpcState>();
    gs->channel = grpc::CreateChannel(server_address, grpc::InsecureChannelCredentials());
    gs->stub    = fg::ToyIpcStream::NewStub(gs->channel);
    gs->context = std::make_unique<grpc::ClientContext>();
    gs->stream  = gs->stub->Stream(gs->context.get());
    if (!gs->stream)
        throw runtime_error("ToyIPC: failed to open gRPC bidi stream to " + server_address);

    // 5. Send the handshake (IpcHandle) as the FIRST message on the
    //    stream. The server's reader thread expects this before any
    //    Produced{} message.
    fg::ProducerMessage handshake;
    fg::IpcHandle *h = handshake.mutable_ipc_handle();
    h->set_handle(reinterpret_cast<const char *>(&ipc_handle), sizeof(ipc_handle));
    h->set_device_id(cuda_device_id);
    if (!gs->stream->Write(handshake)) {
        // Either the server refused us (e.g. duplicate client) or the
        // channel is broken. Surface a useful message; the Finish()
        // status will have details but is unavailable until WritesDone.
        throw runtime_error("ToyIPC: failed to write IpcHandle handshake to " + server_address);
    }

    // 6. Publish gRPC state and spawn the worker thread.
    this->grpc_state = std::move(gs);
    this->worker = std::thread(&ToyIPC::worker_main, this);
}


ToyIPC::~ToyIPC()
{
    stop();
    if (worker.joinable())
        worker.join();
    // grpc_state, ringbuf destructors run after this point.
}


// -------------------------------------------------------------------------------------------------
//
// stop() + helpers


void ToyIPC::stop(std::exception_ptr e)
{
    std::unique_lock<std::mutex> lock(mutex);
    if (is_stopped) return;
    is_stopped = true;
    if (!error)
        error = e;
    cv.notify_all();
    lock.unlock();

    // Unblock the worker thread (it's parked in stream->Read).
    if (grpc_state && grpc_state->context)
        grpc_state->context->TryCancel();
}


void ToyIPC::_throw_if_stopped(const char *method_name)
{
    if (error)
        std::rethrow_exception(error);
    if (is_stopped)
        throw runtime_error(string(method_name) + " called on stopped instance");
}


long ToyIPC::get_rb_start()
{
    std::lock_guard<std::mutex> lock(mutex);
    return rb_start;
}


long ToyIPC::get_rb_end()
{
    std::lock_guard<std::mutex> lock(mutex);
    return rb_end;
}


// -------------------------------------------------------------------------------------------------
//
// Worker thread: reads Consumed{slot=N} off the bidi stream and bumps rb_start.


void ToyIPC::_worker_main()
{
    while (true) {
        fg::ConsumerMessage msg;
        if (!grpc_state->stream->Read(&msg)) {
            // Stream closed or cancelled. If we cancelled it ourselves
            // (via stop -> TryCancel), the outer wrapper will see
            // is_stopped and skip the "terminated with exception" log.
            // Otherwise it's a real server-side close (e.g. the server
            // rejected us as a duplicate client).
            std::lock_guard<std::mutex> lock(mutex);
            if (!is_stopped) {
                // Surface a generic error so the next send() entry
                // point re-throws something informative. We can't
                // easily reach Finish() here without WritesDone, but
                // the stream itself was closed by the peer.
                throw runtime_error("ToyIPC: server closed the Stream RPC");
            }
            return;
        }

        long n = msg.consumed().slot();
        std::lock_guard<std::mutex> lock(mutex);
        if (is_stopped)
            return;

        // Invariant: the consumer should be acknowledging exactly the
        // slot we'd expect next, i.e. rb_start. Any other value means
        // out-of-order / replayed / fabricated CONSUMED messages.
        xassert_eq(n, rb_start);

        rb_start = n + 1;
        xassert(rb_start <= rb_end);            // can't consume unproduced
        xassert(rb_end <= rb_start + 5);        // ring buffer capacity

        cv.notify_all();   // wake send() blockers
    }
}


void ToyIPC::worker_main()
{
    try {
        _worker_main();
    } catch (const std::exception &e) {
        if (!_is_cascade_stop_exception(e)) {
            std::cerr << "ToyIPC: worker thread terminated with exception: "
                      << e.what() << std::endl;
        }
        stop(std::current_exception());
    } catch (...) {
        std::cerr << "ToyIPC: worker thread terminated with unknown exception" << std::endl;
        stop(std::current_exception());
    }
}


// -------------------------------------------------------------------------------------------------
//
// send() entry point.


void ToyIPC::send()
{
    std::unique_lock<std::mutex> lock(mutex);
    _throw_if_stopped("ToyIPC::send");

    try {
        // 1. Wait for a free slot. rb_end - rb_start is the number of
        // occupied slots; we need at least one free, i.e. (rb_end -
        // rb_start) <= 4.
        while (rb_end - rb_start > 4) {
            cv.wait(lock);
            _throw_if_stopped("ToyIPC::send");
        }

        // 2. Capture the slot index and a couple of random values.
        // (send() is single-threaded by contract, so rb_end won't move
        // under us between dropping the lock and re-acquiring it for
        // the bump below.)
        long n = rb_end;
        xassert(rb_start <= rb_end);
        xassert(rb_end - rb_start <= 4);
        long slot = n % 5;

        std::uniform_real_distribution<float> dist(0.0f, 1.0f);
        float v[2] = { dist(rng), dist(rng) };

        // Drop the lock for the memcpy + stdout print + gRPC write.
        // The worker thread may run in parallel and bump rb_start,
        // which is fine -- rb_end is not touched by anyone else.
        lock.unlock();

        // 3. Fill the slot.
        //
        // CUDA IPC memory-ordering note: we use the SYNCHRONOUS
        // cudaMemcpy (NOT cudaMemcpyAsync). The host thread blocks
        // until the device-side write commits, so the producer-side
        // synchronization is "free". Combined with gRPC TCP-order
        // delivery of the Produced{} message below (and single-GPU
        // global-memory coherence on the consumer side), this is
        // sufficient cross-process synchronization -- no IPC event
        // handle is needed.
        //
        // DO NOT "optimize" this to cudaMemcpyAsync, or replace it
        // with a kernel launch, without ALSO threading a
        // cudaIpcEventHandle through the gRPC handshake and having
        // the consumer wait on the event before reading. Without
        // that, the consumer can read garbage.
        CUDA_CALL(cudaMemcpy(ringbuf.data + 2*slot, v, sizeof(v),
                             cudaMemcpyHostToDevice));

        // Mirror the values to stdout so the human running the toy
        // can compare them against the consumer's print.
        std::cout << "ToyIPC::send: produced slot=" << slot
                  << " values=(" << v[0] << ", " << v[1] << ")"
                  << std::endl;

        // 4. Send PRODUCED(n).
        fg::ProducerMessage out;
        out.mutable_produced()->set_slot(n);
        if (!grpc_state->stream->Write(out))
            throw runtime_error("ToyIPC::send: stream->Write(Produced) failed "
                                "(server cancelled the stream?)");

        // 5. Bump rb_end. C++ sync gRPC permits one concurrent Read
        // (worker) + one concurrent Write (here), so the Write above
        // doesn't conflict with the worker's Read.
        lock.lock();
        xassert_eq(rb_end, n);   // single-threaded contract
        rb_end = n + 1;
        xassert(rb_end <= rb_start + 5);
        // No notify here: send() is the only thing that bumps rb_end,
        // and the only waiter on cv is send() itself (in the loop
        // above) which we don't need to wake from here.
    } catch (...) {
        if (lock.owns_lock()) lock.unlock();
        this->stop(std::current_exception());
        throw;
    }
}


}   // namespace pirate
