#include "../include/pirate/FrbGrouper.hpp"
#include "../include/pirate/YamlFile.hpp"         // YamlFile (DedispersionConfig::from_yaml)
#include "../include/pirate/network_utils.hpp"    // parse_ip_address, is_loopback_address

#include "../grpc/frb_grouper.grpc.pb.h"
#include "../grpc/frb_grouper.pb.h"
#include <grpcpp/grpcpp.h>

#include <cuda_runtime.h>

#include <cstring>    // memcpy
#include <iostream>
#include <sstream>
#include <stdexcept>

using namespace std;
using namespace ksgpu;


namespace pirate {
#if 0
}  // editor auto-indent
#endif

namespace fg = frb::grouper::v1;


// -------------------------------------------------------------------------------------------------
//
// Helper: build a strided Array<void> view from an ArrayDescriptor. ksgpu's
// default ctor + manual field init + check_invariants() is the documented way
// to wrap externally-owned (here, IPC-mapped) memory.


static Array<void> _array_from_descriptor(const fg::ArrayDescriptor &ad,
                                          void *base,
                                          const shared_ptr<void> &base_sp)
{
    Array<void> a;                            // default ctor
    a.dtype = Dtype::from_str(ad.dtype());    // canonical ksgpu dtype string

    // Defensive: shape[]/strides[] are fixed-size long[ksgpu::ArrayMaxDim], so
    // validate the wire dims before indexing into them.
    xassert_eq(ad.shape_size(), ad.strides_size());
    xassert((ad.shape_size() >= 1) && (ad.shape_size() <= ArrayMaxDim));

    a.ndim   = ad.shape_size();
    a.aflags = af_gpu;
    long size = 1;
    for (int i = 0; i < a.ndim; i++) { a.shape[i] = ad.shape(i);  size *= a.shape[i]; }
    for (int i = 0; i < a.ndim; i++) { a.strides[i] = ad.strides(i); }
    a.size = size;
    a.data = static_cast<void*>(static_cast<char*>(base) + ad.byte_offset());
    a.base = base_sp;                         // keep the IPC mapping alive
    a.check_invariants();
    // (The shape cross-check against (ring_nbeams, ndm_out[t], nt_out[t]) lives
    // in the caller _process_handshake, which has those geometry members.)
    return a;
}


// -------------------------------------------------------------------------------------------------
//
// gRPC service + pImpl


// gRPC service: forwards Session() into the FrbGrouper (held by weak_ptr,
// mirroring FrbRpcService in FrbServer.cpp).
class FrbGrouperService final : public fg::FrbGrouper::Service {
public:
    std::weak_ptr<FrbGrouper> state;
    explicit FrbGrouperService(std::weak_ptr<FrbGrouper> s) : state(std::move(s)) {}

    grpc::Status Session(grpc::ServerContext* context,
                         grpc::ServerReaderWriter<fg::ConsumerMessage,
                                                  fg::ProducerMessage>* stream) override
    {
        auto s = state.lock();
        if (!s)
            return grpc::Status(grpc::StatusCode::UNAVAILABLE, "FrbGrouper is shutting down");

        // _run_session takes void* (grpc types kept out of FrbGrouper.hpp); it
        // casts them back. Map its result to a grpc::Status.
        switch (s->_run_session(context, stream)) {
            case FrbGrouper::SessionResult::stopped:
                return grpc::Status(grpc::StatusCode::UNAVAILABLE, "FrbGrouper stopped");
            case FrbGrouper::SessionResult::busy:
                return grpc::Status(grpc::StatusCode::FAILED_PRECONDITION,
                                    "FrbGrouper: a client is already connected");
            case FrbGrouper::SessionResult::ok:
            default:
                return grpc::Status::OK;
        }
    }
};


struct FrbGrouper::GrpcState {
    std::unique_ptr<FrbGrouperService> service;
    std::unique_ptr<grpc::Server> server;
    // Published by the active Session handler (under FrbGrouper::mutex):
    grpc::ServerContext* context = nullptr;
    grpc::ServerReaderWriter<fg::ConsumerMessage, fg::ProducerMessage>* stream = nullptr;
};


// -------------------------------------------------------------------------------------------------
//
// create() / constructor


std::shared_ptr<FrbGrouper> FrbGrouper::create(const std::string &addr)
{
    auto p = std::shared_ptr<FrbGrouper>(new FrbGrouper(addr));
    p->grpc_state = std::make_unique<GrpcState>();
    // Construct the RPC service, but do not listen. weak_from_this() is valid
    // now (the shared_ptr exists), so we can build the service here.
    p->grpc_state->service = std::make_unique<FrbGrouperService>(p->weak_from_this());
    return p;
}


FrbGrouper::FrbGrouper(const std::string &addr) : listen_address(addr)
{
    xassert(addr.size() > 0);

    // CUDA IPC requires producer (FrbServer) + consumer (FrbGrouper) to be on the
    // same physical GPU, so the grouper must listen on a loopback address. Enforce
    // this at construction. (parse_ip_address() also throws on a malformed
    // "ip:port" string, which is likewise a configuration error.)
    string ip; uint16_t port;
    parse_ip_address(listen_address, ip, port);   // network_utils.hpp
    if (!is_loopback_address(ip)) {
        stringstream ss;
        ss << "FrbGrouper: listen_address=" << listen_address
           << " is not a loopback address (CUDA IPC requires the grouper to be on "
           << "the same node / GPU as the FrbServer producer)";
        throw runtime_error(ss.str());
    }
}


// -------------------------------------------------------------------------------------------------
//
// open() / start_listening() / wait_for_handshake()
//
// open() is split into two primitives so the pybind binding can drive the wait
// in 0.5s steps and poll for Ctrl-C between them. The plain C++ open() is a
// convenience that loops over them with no signal check.


void FrbGrouper::start_listening()
{
    {
        unique_lock<std::mutex> lock(mutex);
        _throw_if_stopped("FrbGrouper::open");
        if (opened)   // single session only
            throw runtime_error("FrbGrouper::open() called twice (single session only)");
        opened = true;
    }

    grpc::ServerBuilder builder;
    builder.AddListeningPort(listen_address, grpc::InsecureServerCredentials());
    builder.RegisterService(grpc_state->service.get());
    grpc_state->server = builder.BuildAndStart();
    if (!grpc_state->server)
        throw runtime_error("FrbGrouper: failed to bind " + listen_address);

    send_thread = std::thread(&FrbGrouper::send_thread_main, this);
}


bool FrbGrouper::wait_for_handshake(int timeout_ms)
{
    unique_lock<std::mutex> lock(mutex);
    _throw_if_stopped("FrbGrouper::open");
    if (handshake_done)
        return true;

    // Print a "waiting for client" message at ~1/sec (this is polled every
    // ~0.5s by open() / the pybind binding). Throttled via last_waiting_print.
    // Stop once a TCP connection is established (session_active is set by the
    // Session handler's guard at connect time): past that point we are waiting
    // on the handshake, not the connection, so the message would be misleading.
    auto now = std::chrono::steady_clock::now();
    if (!session_active && (now - last_waiting_print >= std::chrono::seconds(1))) {
        last_waiting_print = now;
        std::cout << "FrbGrouper: waiting for client to connect at "
                  << listen_address << " ..." << std::endl;
    }

    cv.wait_for(lock, std::chrono::milliseconds(timeout_ms));
    _throw_if_stopped("FrbGrouper::open");
    return handshake_done;
}


void FrbGrouper::open()
{
    start_listening();
    while (!wait_for_handshake(500))
        ;   // spin in 0.5s steps until the handshake is processed (or throw on stop)
}


// -------------------------------------------------------------------------------------------------
//
// _run_session() (gRPC handler / "receive" thread)


FrbGrouper::SessionResult FrbGrouper::_run_session(void *ctx_, void *stream_)
{
    auto *ctx = static_cast<grpc::ServerContext*>(ctx_);
    auto *stream = static_cast<grpc::ServerReaderWriter<fg::ConsumerMessage,
                                                        fg::ProducerMessage>*>(stream_);

    // Single-client guard + publish the stream/context.
    {
        unique_lock<std::mutex> lock(mutex);
        if (is_stopped)
            return SessionResult::stopped;
        if (session_active)
            return SessionResult::busy;
        session_active = true;
        grpc_state->context = ctx;
        grpc_state->stream  = stream;
    }

    // The TCP connection is now open (gRPC invoked our Session handler).
    // Announce it immediately -- do NOT wait for the handshake (the producer may
    // take a while to build its dedisperser before sending the Handshake).
    std::cout << "FrbGrouper: client connected at " << listen_address << std::endl;

    try {
        // 1. Read the Handshake (must be the first message).
        fg::ProducerMessage first;
        if (!stream->Read(&first) || first.body_case() != fg::ProducerMessage::kHandshake)
            throw runtime_error("FrbGrouper: first message was not a Handshake");

        // 2. Open IPC + build output_ringbuf + parse metadata (sets members).
        _process_handshake(first.handshake());

        // 3. Publish "ready": wakes open() and the send thread (which writes the
        //    HandshakeReply and then drains consumed_seq_ids).
        {
            lock_guard<std::mutex> lock(mutex);
            is_connected = true;
            handshake_done = true;
            cv.notify_all();
        }

        // Announce handshake completion (output_ringbuf is now valid). The
        // connection itself was already announced at TCP-open (above).
        std::cout << "FrbGrouper: handshake processed (cuda_device_id=" << cuda_device_id
                  << ", ntrees=" << ntrees
                  << ", total_beams=" << total_beams
                  << ", nbatches=" << nbatches << ")" << std::endl;

        // 4. Receive loop (this thread). Returns when the stream closes / is cancelled.
        _receive_loop();
    } catch (...) {
        stop(std::current_exception());
    }

    // 5. Session over. Ensure the send thread has stopped touching the stream
    //    before we return (returning destroys the stream).
    stop();   // idempotent; guarantees is_stopped so the send thread exits
    {
        unique_lock<std::mutex> lock(mutex);
        cv.wait(lock, [&]{ return send_io_done; });
        grpc_state->stream = nullptr;
        grpc_state->context = nullptr;
        session_active = false;
    }
    return SessionResult::ok;
}


void FrbGrouper::_process_handshake(const fg::Handshake &hs)
{
    // Reject a mismatched wire-protocol version. fg::PROTOCOL_VERSION_CURRENT is
    // the shared constant from the proto (ProtocolVersion enum) -- the producer
    // sends the same value. (Compared as integers; protocol_version is a uint32
    // field so a newer producer's out-of-range value still reports cleanly.)
    if (long(hs.protocol_version()) != long(fg::PROTOCOL_VERSION_CURRENT)) {
        stringstream ss;
        ss << "FrbGrouper: protocol version mismatch: producer sent "
           << hs.protocol_version() << ", this consumer expects "
           << int(fg::PROTOCOL_VERSION_CURRENT);
        throw runtime_error(ss.str());
    }

    // Capture device.
    cuda_device_id = hs.cuda_device_id();
    CUDA_CALL(cudaSetDevice(cuda_device_id));

    // Stash the wire strings, then deserialize.
    xengine_metadata_yaml_string    = hs.xengine_metadata_yaml();
    dedispersion_config_yaml_string = hs.dedispersion_config_yaml();
    dedispersion_plan_yaml_string   = hs.dedispersion_plan_yaml();

    xengine_metadata = std::make_shared<XEngineMetadata>(
        XEngineMetadata::from_yaml_string(xengine_metadata_yaml_string));

    // DedispersionConfig has no from_yaml_string(); go via YamlFile(name, node).
    YAML::Node cfg_node = YAML::Load(dedispersion_config_yaml_string);
    dedispersion_config = DedispersionConfig::from_yaml(
        YamlFile("dedispersion_config", cfg_node));

    dedispersion_plan_yaml = YAML::Load(dedispersion_plan_yaml_string);

    // Convenience accessors from the config.
    dtype           = dedispersion_config.dtype;
    nt_in           = dedispersion_config.time_samples_per_chunk;
    total_beams     = dedispersion_config.beams_per_gpu;
    beams_per_batch = dedispersion_config.beams_per_batch;
    xassert_divisible(total_beams, beams_per_batch);
    nbatches        = total_beams / beams_per_batch;   // beam-batches per chunk (NOT num_batch_slots)

    // ... and from the plan YAML.
    ntrees = dedispersion_plan_yaml["ntrees"].as<long>();
    ndm_out.clear(); nt_out.clear();
    for (long t = 0; t < ntrees; t++) {
        const YAML::Node tn = dedispersion_plan_yaml["trees"][t];
        ndm_out.push_back(tn["ndm_out"].as<long>());
        nt_out.push_back(tn["nt_out"].as<long>());
    }

    // Geometry cross-checks (defensive).
    num_batch_slots = hs.num_batch_slots();
    xassert_eq(hs.num_trees(), ntrees);
    xassert_eq(hs.beams_per_batch(), beams_per_batch);
    // The output ring buffer's leading (beam) axis is num_batch_slots *
    // beams_per_batch (== producer nbatches_out * beams_per_batch), NOT
    // total_beams. It is <= total_beams (beams_per_gpu); equality holds only in
    // the degenerate case num_active_batches == beams_per_gpu/beams_per_batch.
    long ring_nbeams = num_batch_slots * beams_per_batch;
    xassert_le(ring_nbeams, total_beams);
    xassert_eq(hs.arrays_size(), int(2 * ntrees));   // out_max + out_argmax per tree

    // Open the IPC handle ONCE; wrap as a shared_ptr<void> whose deleter closes
    // it (with the device set), so it outlives all Array views via refcount.
    xassert_eq(hs.ipc_mem_handle().size(), size_t(sizeof(cudaIpcMemHandle_t)));
    cudaIpcMemHandle_t h;
    memcpy(&h, hs.ipc_mem_handle().data(), sizeof(h));
    void *base = nullptr;
    CUDA_CALL(cudaIpcOpenMemHandle(&base, h, cudaIpcMemLazyEnablePeerAccess));
    int dev = cuda_device_id;
    ipc_base = std::shared_ptr<void>(base, [dev](void *p) {
        cudaSetDevice(dev);
        cudaIpcCloseMemHandle(p);   // best-effort; do not throw from a deleter
    });

    // Reconstruct output_ringbuf from the ArrayDescriptors. Each array is a
    // strided view: data = base + byte_offset, shape/strides as sent, base
    // shared_ptr = ipc_base. We do NOT call Outputs::allocate() (memory is
    // IPC-mapped, not allocated here).
    output_ringbuf = GpuDedisperser::Outputs();
    output_ringbuf.dtype   = dtype;
    output_ringbuf.nbeams  = ring_nbeams;   // num_batch_slots * beams_per_batch (NOT total_beams)
    output_ringbuf.ndm_out = ndm_out;
    output_ringbuf.nt_out  = nt_out;
    output_ringbuf.out_max.assign(ntrees, Array<void>());
    output_ringbuf.out_argmax.assign(ntrees, Array<uint>());

    for (const fg::ArrayDescriptor &ad : hs.arrays()) {
        Array<void> arr = _array_from_descriptor(ad, base, ipc_base);
        long t = ad.tree_index();
        xassert((t >= 0) && (t < ntrees));

        // Cross-check the reconstructed view's shape against the geometry we
        // derived from the run-context YAML. The leading axis is ring_nbeams
        // (num_batch_slots * beams_per_batch), NOT total_beams.
        xassert_eq(arr.ndim, 3);
        xassert_eq(arr.shape[0], ring_nbeams);
        xassert_eq(arr.shape[1], ndm_out.at(t));
        xassert_eq(arr.shape[2], nt_out.at(t));

        if (ad.name() == "out_max")         output_ringbuf.out_max[t] = arr;
        else if (ad.name() == "out_argmax") output_ringbuf.out_argmax[t] = arr.cast<uint>("FrbGrouper");
        else throw runtime_error("FrbGrouper: unknown array name '" + ad.name() + "'");
    }
}


void FrbGrouper::_receive_loop()
{
    auto *stream = grpc_state->stream;   // valid: handler owns it for this call
    for (;;) {
        fg::ProducerMessage pm;
        if (!stream->Read(&pm)) {
            // Read returns false on EITHER our own stop()->TryCancel OR an
            // unexpected client/stream close. Distinguish: if we did not
            // initiate the stop, treat it as an error so it propagates through
            // _run_session's catch -> stop(error) -> the consumer.
            { lock_guard<std::mutex> lock(mutex); if (is_stopped) return; }
            throw runtime_error("FrbGrouper: producer closed the Session stream");
        }
        if (pm.body_case() != fg::ProducerMessage::kProducedSeqId)
            throw runtime_error("FrbGrouper: expected produced_seq_id");
        long n = pm.produced_seq_id();
        {
            lock_guard<std::mutex> lock(mutex);
            xassert_eq(n, rb_produced);   // producer sends in order 0,1,2,...
            rb_produced = n + 1;
            cv.notify_all();              // unblocks acquire_output()
        }
    }
}


// -------------------------------------------------------------------------------------------------
//
// Send thread: writes HandshakeReply + consumed_seq_id


void FrbGrouper::_send_loop()
{
    // Wait for the handshake to be processed (stream published, ready to write).
    {
        unique_lock<std::mutex> lock(mutex);
        cv.wait(lock, [&]{ return is_stopped || handshake_done; });
        if (is_stopped) return;
    }
    auto *stream = grpc_state->stream;

    // Write the HandshakeReply (ready). This is the first server->client msg.
    {
        fg::ConsumerMessage reply;
        reply.mutable_handshake_reply()->set_ready(true);
        if (!stream->Write(reply))
            throw runtime_error("FrbGrouper: Write(HandshakeReply) failed");
    }

    // Drain consumed_seq_ids requested by release_output().
    for (;;) {
        long seq;
        {
            unique_lock<std::mutex> lock(mutex);
            cv.wait(lock, [&]{ return is_stopped || (rb_consumed_sent < rb_consumed); });
            if (is_stopped) return;
            seq = rb_consumed_sent;
        }
        fg::ConsumerMessage cm;
        cm.set_consumed_seq_id(seq);
        if (!stream->Write(cm))
            throw runtime_error("FrbGrouper: Write(consumed_seq_id) failed");
        { lock_guard<std::mutex> lock(mutex); rb_consumed_sent = seq + 1; }
    }
}


void FrbGrouper::send_thread_main()
{
    // No cudaSetDevice() here: the send thread does no CUDA work (it only writes
    // to the stream). All IPC/CUDA work runs on the handler thread, which sets
    // the device from the handshake before cudaIpcOpenMemHandle. (The send
    // thread is also spawned in start_listening(), before any handshake, so
    // cuda_device_id is not even known yet at this point.)
    try {
        _send_loop();
    } catch (const std::exception &e) {
        std::cerr << "FrbGrouper: send thread terminated: " << e.what() << std::endl;
        stop(std::current_exception());
    } catch (...) {
        stop(std::current_exception());
    }
    // Tell _run_session we have stopped touching the stream.
    { lock_guard<std::mutex> lock(mutex); send_io_done = true; cv.notify_all(); }
}


// -------------------------------------------------------------------------------------------------
//
// acquire_output() / release_output()


GpuDedisperser::Outputs FrbGrouper::acquire_output(long seq_id)
{
    xassert(seq_id >= 0);
    unique_lock<std::mutex> lock(mutex);

    // Defensive: acquire_output() must be called consecutively (seq_id = 0,1,2,...).
    // (The producer-side GpuDedisperser enforces the analogous invariant on its
    // own acquire cursor; we duplicate it here so a misbehaving consumer fails
    // loudly rather than silently reading the wrong ring slot.)
    xassert_eq(seq_id, rb_acquired);

    // Block until the handshake is done AND produced_seq_id has reached seq_id.
    cv.wait(lock, [&]{ return is_stopped || (handshake_done && rb_produced > seq_id); });
    _throw_if_stopped("FrbGrouper::acquire_output");

    rb_acquired = seq_id + 1;
    long iout = seq_id % num_batch_slots;
    return output_ringbuf.slice(iout * beams_per_batch, (iout + 1) * beams_per_batch);
}


void FrbGrouper::release_output(long seq_id)
{
    unique_lock<std::mutex> lock(mutex);
    _throw_if_stopped("FrbGrouper::release_output");

    // Defensive cursor checks (mirroring the producer-side GpuDedisperser checks):
    //   - release_output() called consecutively (seq_id = 0,1,2,...);
    //   - release stays strictly BEHIND acquire (a batch can't be released
    //     before it has been acquired), i.e. rb_consumed < rb_acquired.
    xassert_eq(seq_id, rb_consumed);
    xassert_lt(rb_consumed, rb_acquired);

    rb_consumed = seq_id + 1;
    cv.notify_all();                   // wake the send thread
}


// -------------------------------------------------------------------------------------------------
//
// stop() / close() / destructor / misc


void FrbGrouper::stop(std::exception_ptr e)
{
    std::unique_lock<std::mutex> lock(mutex);
    if (is_stopped) return;
    is_stopped = true;
    if (!error) error = e;

    // Unblock the gRPC handler (pool thread) parked in a blocking stream->Read,
    // and the send thread parked in stream->Write. TryCancel() is non-blocking
    // and thread-safe, so we call it WHILE holding the mutex. This is the
    // use-after-free-safe ordering: the handler clears grpc_state->context
    // (under this same mutex) only just before it returns, so observing a
    // non-null context here guarantees the ServerContext is still alive.
    if (grpc_state && grpc_state->context)
        grpc_state->context->TryCancel();

    cv.notify_all();

    // NOTE: we deliberately do NOT call the blocking Server::Shutdown() here.
    // stop() can be invoked FROM the handler (pool) thread (on exception), and
    // Server::Shutdown() blocks until in-flight RPCs finish -- calling it from
    // the handler would self-deadlock. Server teardown lives in close() /
    // ~FrbGrouper() (always on the consumer thread). The TryCancel() above is
    // what unblocks the handler so it can return and free its pool thread.
}


void FrbGrouper::close()
{
    { std::lock_guard<std::mutex> lock(mutex); if (closed) return; closed = true; }

    stop();   // TryCancel -> the handler's Read returns false -> handler returns

    if (send_thread.joinable())
        send_thread.join();

    if (grpc_state && grpc_state->server) {
        // The in-flight RPC was already cancelled by stop(), so this no-arg
        // Shutdown() returns as soon as the handler has returned (prompt).
        grpc_state->server->Shutdown();
        grpc_state->server->Wait();
    }
}


FrbGrouper::~FrbGrouper()
{
    close();   // close() is idempotent
}


bool FrbGrouper::is_stopped_pub()
{
    std::lock_guard<std::mutex> lock(mutex);
    return is_stopped;
}


void FrbGrouper::_throw_if_stopped(const char *method_name)
{
    // Caller must hold mutex.
    if (error)
        std::rethrow_exception(error);
    if (is_stopped)
        throw std::runtime_error(std::string(method_name) + " called on stopped instance");
}


}  // namespace pirate
