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
    static constexpr long host_page_size = 4096;          // 4 KiB
    static constexpr long host_hugepage_size = 2L << 20;  // 2 MiB
    static constexpr long cuda_host_register_chunk_size = 64L << 30;  // 64 GiB

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

    // Frame-pool / pacing sizing, in "chunks" (one chunk = nbeams frames).
    static constexpr int server_min_total_chunks = 10;
    static constexpr int reaper_lowmem_chunks = 2;
    static constexpr int fake_xengine_pacing_chunks = 5;

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

    static_assert(reaper_lowmem_chunks >= 2);
    static_assert(fake_xengine_pacing_chunks >= 3);
    
    // The frame pool must hold the reaper's pre-init reserve + the FakeXEngine
    // pacing lookahead, plus headroom for the in-flight chunks.
    static_assert(server_min_total_chunks >= fake_xengine_pacing_chunks + reaper_lowmem_chunks + 2,
                  "server_min_total_chunks too small for the pacing + reaper reserve");
};


}  // namespace pirate

#endif // _PIRATE_CONSTANTS_HPP
