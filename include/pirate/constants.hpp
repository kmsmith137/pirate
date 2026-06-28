#ifndef _PIRATE_CONSTANTS_HPP
#define _PIRATE_CONSTANTS_HPP

namespace pirate {
#if 0
}  // editor auto-indent
#endif


struct constants
{
    // REMINDER: selected constants here are exposed to python as pirate_frb.constants.<name>,
    // via the py::class_<constants> block in src_pybind11/pirate_pybind11.cpp (read-only). If you
    // add a constant that should be visible from python, add a def_readonly_static() line there too.

    static constexpr int bytes_per_gpu_cache_line = 128;
    
    // Currently all Dedispersers are two-stage, and each stage has rank <= 8,
    // so max total rank is 16.
    
    static constexpr int max_tree_rank = 16;

    // If you need to change 'max_downsampling_level', there should be no issues
    // (besides needing to recompile). However, if max_downsampling_level is
    // gratuitously large, then compilation time may be an issue.
    
    static constexpr int max_downsampling_level = 6;

    // Max width of a peak-finding kernel (PeakFindingConfig::max_width, in "tree" time
    // samples). Must be a power of two. Bounds both DedispersionConfig::validate() and the
    // make_random() config generator. (Production configs currently use 16, and the compiled
    // GPU kernel registry currently provides Wmax in {8, 16}; this looser bound matches the
    // largest width the make_random() reference path exercises.)

    static constexpr int max_pf_width = 32;

    // Constants in peak-finding kernels
    static constexpr float pf_a = 0.5;
    static constexpr float pf_b = 0.5;

    // Dispersion constant K_DM, in (ms . MHz^2) per (pc cm^{-3}):
    //   dispersion delay (ms) = k_dm * DM * (f_lo^{-2} - f_hi^{-2}),
    // with DM in pc cm^{-3} and f_lo, f_hi in MHz. (Equivalently, 4.148808e3 s MHz^2.)
    static constexpr double k_dm = 4.148808e6;

    // https://docs.nvidia.com/cuda/cuda-c-programming-guide/index.html#features-and-technical-specifications
    static constexpr int cuda_max_y_blocks = 65535;
    static constexpr int cuda_max_z_blocks = 65535;

    // These assumptions are made all over the place.
    // (Placement of static_asserts in this source file is arbitrary.)
    static_assert(sizeof(int) == 4);
    static_assert(sizeof(long) == 8);

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
    
    // The constant is power-of-two so the splitter can use bit-arithmetic.
    static_assert((cuda_host_register_chunk_size
                   & (cuda_host_register_chunk_size - 1)) == 0,
                  "cuda_host_register_chunk_size must be a power of two");
    
    static_assert(cuda_host_register_chunk_size <= (511L << 30),
                  "cuda_host_register_chunk_size must be <= 511 GiB");
};


}  // namespace pirate

#endif // _PIRATE_CONSTANTS_HPP
