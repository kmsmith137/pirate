#ifndef _PIRATE_TOY_IPC_HPP
#define _PIRATE_TOY_IPC_HPP

// Toy CUDA-IPC ring buffer producer. Pairs with the Python class
// pirate_frb.utils.ToyGrouper. See plans/grouper.md for the design.
//
// ToyIPC is a thread-backed class. Its constructor:
//
//   1. cudaMalloc's a shape (5,2) float32 ring buffer on the GPU (via
//      ksgpu::Array<float>({5,2}, af_gpu | af_zero)).
//   2. Opens a gRPC bidi Stream to server_address and sends the
//      cudaIpcMemHandle_t of the ring buffer as the FIRST message on
//      the stream.
//   3. Spawns a worker thread that reads Consumed{slot=N} messages off
//      the stream and updates rb_start.
//
// The python-callable entry point send() blocks until a slot is free,
// writes two random floats into the slot (synchronous cudaMemcpy, see
// comment in ToyIPC.cu near the memcpy for why no IPC event is needed),
// then sends a Produced{slot=N} message and bumps rb_end.
//
// Invariants (checked under the mutex):
//
//   0 <= rb_start <= rb_end <= rb_start + 5

#include <ksgpu/Array.hpp>

#include <condition_variable>
#include <exception>
#include <memory>
#include <mutex>
#include <random>
#include <string>
#include <thread>

namespace pirate {
#if 0
}   // editor auto-indent
#endif


struct ToyIPC : public std::enable_shared_from_this<ToyIPC>
{
    // Factory (constructor is private; mirrors FrbServer / GpuDedisperser
    // conventions in this repo).
    static std::shared_ptr<ToyIPC> create(
        const std::string &server_address,
        int cuda_device_id);

    ~ToyIPC();   // calls stop() then joins the worker thread

    // Entry point. Produces one slot; blocks until a slot is free.
    // Throws if the instance is stopped.
    void send();

    // Idempotent. Calls TryCancel() on the client context, which
    // unblocks any in-flight stream->Read() in the worker thread.
    void stop(std::exception_ptr e = nullptr);

    // Lock-protected snapshots of the ring-buffer counters.
    long get_rb_start();
    long get_rb_end();

    // Noncopyable, nonmoveable.
    ToyIPC(const ToyIPC &) = delete;
    ToyIPC &operator=(const ToyIPC &) = delete;
    ToyIPC(ToyIPC &&) = delete;
    ToyIPC &operator=(ToyIPC &&) = delete;

private:
    ToyIPC(const std::string &server_address, int cuda_device_id);

    // Mutex-protected state.
    std::mutex mutex;
    std::condition_variable cv;
    long rb_start = 0;
    long rb_end   = 0;
    bool is_stopped = false;
    std::exception_ptr error;

    // GPU memory + identity.
    int cuda_device_id;
    ksgpu::Array<float> ringbuf;   // shape (5,2)

    // send() is single-threaded by contract -- the python driver calls
    // it from one thread -- so the RNG needs no extra synchronization.
    std::mt19937 rng;

    // gRPC client state (channel + stub + context + stream). Hidden
    // behind pImpl so this header does not pull in grpc++ headers.
    struct GrpcState;
    std::unique_ptr<GrpcState> grpc_state;

    std::thread worker;

    void _start(const std::string &server_address);  // ctor body after pImpl exists
    void _throw_if_stopped(const char *method_name);  // caller holds mutex
    void _worker_main();
    void worker_main();
};


}   // namespace pirate

#endif // _PIRATE_TOY_IPC_HPP
