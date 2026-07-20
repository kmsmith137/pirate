#ifndef _PIRATE_CONSTANTS_HPP
#define _PIRATE_CONSTANTS_HPP

namespace pirate {
#if 0
}  // editor auto-indent
#endif


// NOTE 1: most of these constants don't really need to be known at compile time, and
// could easily be promoted to "runtime" parameters, or moved to a config file.
//
// NOTE 2: selected constants here are exposed to python as pirate_frb.constants.<name>,
// via the py::class_<constants> block in src_pybind11/pirate_pybind11.cpp (read-only). If you
// add a constant that should be visible from python, add a def_readonly_static() line there too.


struct constants
{
    // GPU hardware params.
    // https://docs.nvidia.com/cuda/cuda-c-programming-guide/index.html#features-and-technical-specifications
    static constexpr int bytes_per_gpu_cache_line = 128;
    static constexpr int cuda_max_static_shmem_bytes = 48 * 1024;
    static constexpr int cuda_max_dynamic_shmem_bytes = 99 * 1024;
    static constexpr int cuda_max_y_blocks = 65535;
    static constexpr int cuda_max_z_blocks = 65535;

    // Memory management. The 'cuda_host_register_chunk_size' param is part of a complicated
    // workaround for an undocumented(!!) cudaHostRegister() limit -- see BumpAllocator.{hpp,cpp}.
    // The chunk size also bounds stop()/control-C latency during async init (the registrar
    // checks for stop between chunks, and an in-flight cudaHostRegister can't be interrupted),
    // which is why it's much smaller than the ~511 GiB limit requires. Trade-off: smaller
    // chunks mean more registration seams, and every host<->GPU copy that might be backed
    // by a BumpAllocator must split at seams (see safe_memcpy_* in utils.hpp).
    static constexpr long host_page_size = 4096;          // 4 KiB
    static constexpr long host_hugepage_size = 2L << 20;  // 2 MiB
    static constexpr long cuda_host_register_chunk_size = 1L << 30;  // 1 GiB

    // Dedispersion params.
    static constexpr int max_tree_rank = 16;          // hard to change
    static constexpr int max_primary_trees = 7;       // straightforward to change
    static constexpr int max_pf_width = 32;           // hard to change
    static constexpr int max_peak_finding_rank = 4;   // hard to change

    // FRB params.
    // Dispersion delay (ms) = k_dm * DM * (f_lo^{-2} - f_hi^{-2}), with freqs in MHz.
    // Scattering time tau ~ (radio frequency)^{-frb_scattering_index}.
    // We simulate DMs by taking u = log(DM + frb_dm0) to be uniform-random.
    static constexpr double k_dm = 4.148808e6;
    static constexpr double frb_scattering_index = 4.0;
    static constexpr double frb_dm0 = 50.0;   // gives median DM ~ 600

    // Frame-pool / backpressure sizing, in time-chunks.
    //
    //  - server_max_unprocessed_chunks: fail if server can't keep up with x-engine
    //  - assembled_frame_allocator_queue_size: fail if frame queue becomes empty
    //  - assembled_frame_allocator_initial_size: allocate initial "burst" and fail-fast.
    
    static constexpr int server_max_unprocessed_chunks = 5;
    static constexpr int assembled_frame_allocator_queue_size = 5;
    static constexpr int assembled_frame_allocator_initial_size = 7;

    // Timeouts.
    //
    //   - poll_cadence: low-level polling (e.g. for catching control-C)
    //   - print_cadence: monitoring print-statements (e.g. pirate_frb run_rpc_status)
    //   - shutdown_timeout: joining threads/processes, waiting for SIGTERM/SIGKILL
    //   - grpc_reconnect_backoff: client-side reconnection cadence
    //   - grpc_forced_shutdown_deadline: server-side shutdown time (recommend short)
    //   - grouper_ping_timeout: initial connection during server startup, "fail fast"
    //   - grouper_connect_timeout: the real reconnect done later

    static constexpr int default_poll_cadence_ms = 250;
    static constexpr double default_print_cadence_sec = 1.0;
    static constexpr double default_shutdown_timeout_sec = 5.0;
    static constexpr int grpc_reconnect_backoff_ms = 1000;
    static constexpr int grpc_forced_shutdown_deadline_ms = 100;
    static constexpr int grouper_ping_timeout_ms = 5000;
    static constexpr int grouper_connect_timeout_ms = 2000;

    // Number of inactive (expired/cancelled) streams retained for ShowStreams history.
    static constexpr int inactive_file_stream_capacity = 5;

    // Max queued-but-unsent file notifications per SubscribeFiles subscriber.
    // A subscriber that falls this far behind (e.g. a client that keeps the
    // stream open but never reads) is stopped and its queue freed, so server
    // memory stays bounded. ~200-300 bytes/entry -> ~10 MB per subscriber
    // at the cap.
    static constexpr long max_file_subscriber_backlog = 50000;

    // ---------------------------------------------------------------------------------------------
    //
    // Static asserts and derived params.

    // These assumptions are made all over the place.
    static_assert(sizeof(int) == 4);
    static_assert(sizeof(long) == 8);

    static_assert((max_pf_width & (max_pf_width-1)) == 0,
                  "max_pf_kernels must be a power of two");
    
    // The constant is power-of-two so the splitter can use bit-arithmetic.
    static_assert((cuda_host_register_chunk_size
                   & (cuda_host_register_chunk_size - 1)) == 0,
                  "cuda_host_register_chunk_size must be a power of two");
    
    static_assert(cuda_host_register_chunk_size <= (511L << 30),
                  "cuda_host_register_chunk_size must be <= 511 GiB");

    // The FakeXEngine pacing lookahead is derived from this constant
    // (pacing_chunks = server_max_unprocessed_chunks - 1, in FakeXEngine.cpp,
    // leaving one chunk of margin below the bound). The lookahead must be
    // >= 3 chunks (the pre-existing lower bound on the pacing constant this
    // replaced): assembly needs data from chunk c+2 to complete chunk c, so
    // too small a lookahead stalls or deadlocks the paced pipeline.
    static_assert(server_max_unprocessed_chunks - 1 >= 3);

    // The startup burst (throw_exception_if_empty) should at least fill the pre-init
    // queue, so the worker parks after it (drain-then-replenish semantics)
    // and the startup runway is never less than the steady-state headroom.
    static_assert(assembled_frame_allocator_initial_size >=
                  assembled_frame_allocator_queue_size);
};


}  // namespace pirate

#endif // _PIRATE_CONSTANTS_HPP
