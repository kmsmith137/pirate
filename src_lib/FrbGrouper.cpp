#include "../include/pirate/FrbGrouper.hpp"
#include "../include/pirate/YamlFile.hpp"         // YamlFile (DedispersionConfig::from_yaml)
#include "../include/pirate/network_utils.hpp"    // parse_ip_address, is_loopback_address

// See "NDEBUG and libabseil" in notes/build.md for the push_macro trick,
// and its companion comment on -Wdeprecated-declarations. Uniform grpc
// wrap idiom shared with FrbServer.cpp and FakeXEngine.cpp.
#pragma push_macro("NDEBUG")
#ifndef NDEBUG
#  define NDEBUG
#endif
#pragma GCC diagnostic push
#pragma GCC diagnostic ignored "-Wdeprecated-declarations"
#include "../grpc/frb_grouper.grpc.pb.h"
#include "../grpc/frb_grouper.pb.h"
#include <grpcpp/grpcpp.h>
#pragma GCC diagnostic pop
#pragma pop_macro("NDEBUG")

#include <cuda_runtime.h>

#include <chrono>     // Server::Shutdown() deadline
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
// FrbGrouperClient (gRPC *client* side; declared in FrbGrouper.hpp). Owns the
// producer-side channel / stub / Session stream + the connect/ping/cancel logic.
// FrbServer drives the protocol (Handshake + produced/consumed loops) over it.


struct FrbGrouperClient::GrpcState
{
    std::shared_ptr<grpc::Channel> channel;
    std::unique_ptr<fg::FrbGrouper::Stub> stub;
    std::unique_ptr<grpc::ClientContext> context;
    std::unique_ptr<grpc::ClientReaderWriter<fg::ProducerMessage,
                                             fg::ConsumerMessage>> stream;
};


// Channel args shared by ping() and connect(): cap the connection-reconnect
// backoff at 1s (gRPC defaults to exponential backoff up to 120s, which would
// make a bounded READY wait racy). initial == max == 1s => retry ~once/second.
static grpc::ChannelArguments _grouper_chan_args()
{
    grpc::ChannelArguments a;
    a.SetInt(GRPC_ARG_INITIAL_RECONNECT_BACKOFF_MS, 1000);
    a.SetInt(GRPC_ARG_MAX_RECONNECT_BACKOFF_MS, 1000);
    return a;
}


// Bring 'channel' to GRPC_CHANNEL_READY within timeout_ms, else throw.
static void _wait_ready_or_throw(const shared_ptr<grpc::Channel> &channel,
                                 const string &grouper_ip_addr, int timeout_ms)
{
    auto deadline = std::chrono::system_clock::now() + std::chrono::milliseconds(timeout_ms);
    while (true) {
        // Observe the state once (try_to_connect=true kicks the channel out of
        // IDLE and starts connecting), and pass that same observation to
        // WaitForStateChange -- so we wait for a change away from exactly what we
        // saw, with no GetState()-called-twice race.
        auto state = channel->GetState(/*try_to_connect=*/true);
        if (state == GRPC_CHANNEL_READY)
            return;
        // Returns false iff the deadline expired with no change. Re-check once on
        // expiry, in case we became READY right at the deadline.
        if (!channel->WaitForStateChange(state, deadline)) {
            if (channel->GetState(/*try_to_connect=*/false) == GRPC_CHANNEL_READY)
                return;
            stringstream ss;
            ss << "FrbGrouperClient: grouper at " << grouper_ip_addr
               << " not reachable within " << timeout_ms
               << " ms (is run_toy_grouper running?)";
            throw runtime_error(ss.str());
        }
    }
}


FrbGrouperClient::FrbGrouperClient(const string &grouper_ip_addr_)
    : grouper_ip_addr(grouper_ip_addr_), grpc_state(make_unique<GrpcState>())
{ }

FrbGrouperClient::~FrbGrouperClient() = default;


void FrbGrouperClient::ping(int timeout_ms)
{
    // Channel-level connectivity check: a throwaway channel, brought to READY,
    // then dropped. No Session RPC / Handshake, so the grouper's single-session
    // state is untouched.
    auto channel = grpc::CreateCustomChannel(grouper_ip_addr,
                                             grpc::InsecureChannelCredentials(),
                                             _grouper_chan_args());
    _wait_ready_or_throw(channel, grouper_ip_addr, timeout_ms);
}


void FrbGrouperClient::connect(int timeout_ms)
{
    // Fresh channel + stub (ping()'s throwaway channel was dropped). Wait for
    // READY, then open the Session stream.
    grpc_state->channel = grpc::CreateCustomChannel(grouper_ip_addr,
                                                    grpc::InsecureChannelCredentials(),
                                                    _grouper_chan_args());
    grpc_state->stub = fg::FrbGrouper::NewStub(grpc_state->channel);
    _wait_ready_or_throw(grpc_state->channel, grouper_ip_addr, timeout_ms);
    grpc_state->context = make_unique<grpc::ClientContext>();
    grpc_state->stream  = grpc_state->stub->Session(grpc_state->context.get());
}


bool FrbGrouperClient::write(const fg::ProducerMessage &msg)
{
    xassert(grpc_state->stream);   // connect() must have run first
    return grpc_state->stream->Write(msg);
}


bool FrbGrouperClient::read(fg::ConsumerMessage *msg)
{
    xassert(grpc_state->stream);   // connect() must have run first
    return grpc_state->stream->Read(msg);
}


void FrbGrouperClient::cancel()
{
    // Idempotent; safe from any thread. TryCancel() unblocks any in-flight
    // Write/Read (they return false).
    if (grpc_state->context)
        grpc_state->context->TryCancel();
}


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


FrbGrouper::FrbGrouper(const std::string &ip_addr) : grouper_ip_addr(ip_addr)
{
    xassert(ip_addr.size() > 0);

    // CUDA IPC requires producer (FrbServer) + consumer (FrbGrouper) to be on the
    // same physical GPU, so the grouper must listen on a loopback address. Enforce
    // this at construction. (parse_ip_address() also throws on a malformed
    // "ip:port" string, which is likewise a configuration error.)
    string ip; uint16_t port;
    parse_ip_address(grouper_ip_addr, ip, port);   // network_utils.hpp
    if (!is_loopback_address(ip)) {
        stringstream ss;
        ss << "FrbGrouper: grouper_ip_addr=" << grouper_ip_addr
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
    // Per the strict stoppable-class policy (notes/stoppable_class.md), ANY
    // exception thrown from an entry point stops the object -- including the
    // "called twice" precondition, a bind failure, and thread-creation
    // failure. Without the stop, a caller that catches the exception would be
    // left with a zombie grouper: 'opened' is already set, no server is
    // listening, and wait_for_handshake() / acquire_output() would block
    // forever instead of throwing.
    try {
        // Hold the mutex across BuildAndStart + the send-thread spawn, so
        // 'grpc_state->server' and 'send_thread' are PUBLISHED under the
        // mutex (close() synchronizes on it before reading them). Safe:
        // nothing here waits on the send thread or the Session handler,
        // which both block briefly on the mutex until we release.
        unique_lock<std::mutex> lock(mutex);
        _throw_if_stopped("FrbGrouper::open");
        if (opened)   // single session only
            throw runtime_error("FrbGrouper::open() called twice (single session only)");
        opened = true;

        try {
            // Bind + start. The 2-arg AddListeningPort does NOT report a bind failure:
            // if the port is already in use, BuildAndStart() still returns a non-null
            // server, so we would "wait for FrbServer to connect" while not actually
            // listening -- and the FrbServer that connects would hang forever. Use the
            // 3-arg overload and check selected_port, which is 0 iff the bind failed.
            grpc::ServerBuilder builder;
            int selected_port = 0;
            builder.AddListeningPort(grouper_ip_addr, grpc::InsecureServerCredentials(), &selected_port);
            builder.RegisterService(grpc_state->service.get());
            grpc_state->server = builder.BuildAndStart();
            if (!grpc_state->server || (selected_port == 0))
                throw runtime_error("FrbGrouper: failed to bind " + grouper_ip_addr
                                    + " (already in use, or malformed 'ip:port'?)");

            send_thread = std::thread(&FrbGrouper::send_thread_main, this);
        } catch (...) {
            // If the failure occurred after BuildAndStart() (e.g. send-thread
            // creation failed), the server is briefly listening, and a client's
            // Session handler may already be parked in its initial stream->Read.
            // The outer catch's stop() TryCancels that Read, but the
            // handler then waits (in its step-5 teardown) for send_io_done --
            // which is normally set by the send thread on exit, and the send
            // thread may never have been created. Set it here (vacuously true:
            // no send thread will ever touch the stream; the enclosing 'lock'
            // is still held), so the handler can return instead of hanging
            // close()'s server->Wait() forever.
            send_io_done = true;
            throw;
        }
    } catch (...) {
        stop(std::current_exception());
        throw;
    }
}


bool FrbGrouper::wait_for_handshake(int timeout_ms)
{
    unique_lock<std::mutex> lock(mutex);
    _throw_if_stopped("FrbGrouper::open");
    if (handshake_done)
        return true;

    // Print a one-time "waiting for FrbServer to connect" message (this is polled
    // every ~0.5s by open() / the pybind binding). Guarded by waiting_print_done so
    // it prints once, not repeatedly. Suppressed once a TCP connection is
    // established (session_active is set by the Session handler's guard at connect
    // time): past that point we are waiting on the handshake, not the connection,
    // so the message would be misleading.
    if (!session_active && !waiting_print_done) {
        waiting_print_done = true;
        std::cout << "FrbGrouper: waiting for FrbServer to connect at "
                  << grouper_ip_addr << " ..." << std::endl;
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
    std::cout << "FrbGrouper: client connected at " << grouper_ip_addr << std::endl;

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

    // The producer's own RPC endpoint (not deserialized -- just stored).
    search_ip_addr = hs.rpc_ip_addr();

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
    initial_chunk = hs.initial_chunk();   // producer's GpuDedisperser::Params::initial_chunk
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
    } catch (...) {
        // No print here (per the error-reporting convention in
        // notes/stoppable_class.md): the saved error surfaces via the
        // consumer's entry points. This also avoids a misleading "send thread
        // terminated" message when a normal close() lands between the
        // drain-wait check and a stream Write (the Write then fails benignly).
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
    // Per the strict stoppable-class policy, ANY throw (including the
    // defensive cursor xasserts) stops the grouper.
    try {
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
        GpuDedisperser::Outputs out =
            output_ringbuf.slice(iout * beams_per_batch, (iout + 1) * beams_per_batch);

        // Set the chunk/beam identity fields on the returned slice. These override
        // the values slice() copied from output_ringbuf: the ring slot 'iout'
        // (= seq_id % num_batch_slots) is NOT the true chunk/beam index. Reconstruct
        // them from seq_id = ichunk*nbatches + ibatch (the producer-side mapping in
        // GpuDedisperser::acquire_output()), using initial_chunk from the handshake.
        out.ichunk_zero_based = seq_id / nbatches;
        out.ichunk_fpga_based = out.ichunk_zero_based + initial_chunk;
        out.ibeam = (seq_id % nbatches) * beams_per_batch;
        return out;
    } catch (...) {
        stop(std::current_exception());
        throw;
    }
}


void FrbGrouper::release_output(long seq_id)
{
    try {
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
    } catch (...) {
        stop(std::current_exception());
        throw;
    }
}


// -------------------------------------------------------------------------------------------------
//
// stop() / close() / destructor / misc


void FrbGrouper::stop(std::exception_ptr e) const
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
    // Serialize concurrent close() calls: a second caller must BLOCK here
    // until the first finishes tearing down. (With only the 'closed' flag,
    // the second caller would return immediately -- letting e.g. the
    // destructor run and free members while the first close() is still
    // joining the send thread / shutting the server down.) close_mutex is
    // leaf-level: it is never acquired while 'mutex' is held.
    std::lock_guard<std::mutex> close_lock(close_mutex);

    { std::lock_guard<std::mutex> lock(mutex); if (closed) return; closed = true; }

    stop();   // TryCancel -> the handler's Read returns false -> handler returns

    // Reading send_thread / grpc_state->server without 'mutex' is safe here:
    // open() publishes both under 'mutex', the closed-check above acquired
    // 'mutex' (happens-before), and nothing writes them after open().
    if (send_thread.joinable())
        send_thread.join();

    if (grpc_state && grpc_state->server) {
        // Shutdown with an (essentially immediate) deadline, NOT the no-arg
        // overload. The in-flight RPC was already cancelled by stop() above, so
        // there is nothing to drain -- but the deadline-less Server::Shutdown()
        // still blocks ~3.5s internally before returning. Passing a deadline
        // makes it return as soon as the (already-returned) handler is reaped;
        // any RPC somehow still in flight at the deadline is force-cancelled,
        // which is fine since we are tearing the server down.
        grpc_state->server->Shutdown(std::chrono::system_clock::now()
                                     + std::chrono::milliseconds(100));
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
