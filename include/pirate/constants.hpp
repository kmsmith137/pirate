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
    // https://docs.nvidia.com/cuda/cuda-c-programming-guide/index.html#features-and-technical-specifications
    static constexpr int bytes_per_gpu_cache_line = 128;
    static constexpr int cuda_max_static_shmem_bytes = 48 * 1024;
    static constexpr int cuda_max_dynamic_shmem_bytes = 99 * 1024;
    static constexpr int cuda_max_y_blocks = 65535;
    static constexpr int cuda_max_z_blocks = 65535;
    
    // Currently all Dedispersers are two-stage, and each stage has rank <= 8,
    // so max total rank is 16.
    
    static constexpr int max_tree_rank = 16;

    // If you need to change 'max_primary_trees', there should be no issues
    // (besides needing to recompile). However, if max_primary_trees is
    // gratuitously large, then compilation time may be an issue (the
    // LaggedDownsampling GPU kernels are instantiated once per allowed
    // number of primary trees).

    static constexpr int max_primary_trees = 7;

    // Max width of a peak-finding kernel (PrimaryTree::max_width, in "tree" time
    // samples). Must be a power of two. Bounds both DedispersionConfig::validate() and the
    // make_random() config generator. (Production configs currently use 16, and the compiled
    // GPU kernel registry currently provides Wmax in {8, 16}; this looser bound matches the
    // largest width the make_random() reference path exercises.)

    static constexpr int max_pf_width = 32;

    // Max peak-finding rank supported by the peak-finding kernel
    // (FrequencySubbands enforces pf_rank <= this).
    static constexpr int max_peak_finding_rank = 4;

    // Dispersion constant K_DM, in (ms . MHz^2) per (pc cm^{-3}):
    //   dispersion delay (ms) = k_dm * DM * (f_lo^{-2} - f_hi^{-2}),
    // with DM in pc cm^{-3} and f_lo, f_hi in MHz. (Equivalently, 4.148808e3 s MHz^2.)
    static constexpr double k_dm = 4.148808e6;

    // FRB pulse-scattering spectral index: scattering time tau ~ nu^-frb_scattering_index
    // (used by simpulse::scattering_time).
    static constexpr double frb_scattering_index = 4.0;

    // DM-scale offset (pc cm^-3) for the log-uniform DM distribution of simulated
    // FRBs: DM is drawn with u = log(DM + frb_dm0) uniform on [0, frb_max_dm], so
    // frb_dm0 sets the DM scale below which the distribution is ~uniform. Used by
    // run_fake_xengine (passed to SimulatedFrameFactory). (Exposed to python.)
    static constexpr double frb_dm0 = 50.0;

    // Number of inactive (expired/cancelled) FileStreams retained by an
    // FrbServer for ShowStreams history; the oldest are dropped beyond this.
    // (Exposed to python -- see the reminder at the top of this struct. The
    // network test reads it at runtime and assumes >= 5.)
    static constexpr int inactive_file_stream_capacity = 5;

    // Frame-pool / pacing sizing, in "chunks" (one chunk = nbeams frames):
    //   - server_min_total_chunks: the FrbServer frame pool must hold at least this
    //     many chunks; checked at startup (FrbServer::_check_frame_pool_size).
    //   - reaper_lowmem_chunks: chunks the reaper keeps pre-initialized (its
    //     low-memory back-pressure threshold).
    //   - fake_xengine_pacing_chunks: paced mode -- each FakeXEngine worker stays at
    //     most this many chunks ahead of the server's rb_processed.
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

    // Host memory page sizes (system assumptions; BumpAllocator uses these for
    // mmap alignment, and utils.cpp's prefault loop for the page stride).
    static constexpr long host_page_size = 4096;          // 4 KiB
    static constexpr long host_hugepage_size = 2L << 20;  // 2 MiB

    // The CUDA driver caps a single cudaHostRegister() call at ~511 GiB (undocumented!!)
    //
    // Calling cudaHostRegister() in chunks works, but creates a new problem:
    // calls to cudaMemcpy*() fail if they cross chunk boundaries.
    //
    // In pirate::BumpAllocator, we implement a complicated workaround:
    //
    //   - register BumpAllocator backing memory in chunks aligned
    //     to absolute host addresses.
    //
    //   - in situations where a cudaMemcpy* may be backed by a BumpAllocator,
    //     we call safe_memcpy_{h2g,g2h}_{sync,async}() (see utils.hpp) which
    //     splits host<->device copies at chunk boundaries.
    //
    // Re-test whether the 511 GiB cap is still present on a newer CUDA /
    // driver version with: `python -m pirate_frb revisit_512gb [-H]`.

    // Chunk size for cudaHostRegister().
    static constexpr long cuda_host_register_chunk_size = 64L << 30;  // 64 GiB

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
