#ifndef _PIRATE_TESTS_HPP
#define _PIRATE_TESTS_HPP

namespace pirate {
#if 0
}  // editor auto-indent
#endif


// Defined in DedispersionConfig.hpp
struct DedispersionConfig;

// test_dedispersion_basics.cu
extern void test_non_incremental_dedispersion(int rank, int ntime, int dm_brev, int t0);
extern void test_non_incremental_dedispersion();
extern void test_reference_lagbuf();
extern void test_reference_tree();
extern void test_tree_recursion(int rank0, int rank1, int nt_chunk, int nchunks);
extern void test_tree_recursion();
extern void test_dedispersion_basics();
    
// test_gpu_lagged_downsampler.cu
extern void test_gpu_lagged_downsampling_kernel();

// test_gpu_dedispersion_kernels.cu
extern void test_gpu_dedispersion_kernels();

// test_gpu_ringbuf_copy_kernel.cu
extern void test_gpu_ringbuf_copy_kernel();

// test_gpu_peak_finding_kernel.cu
extern void test_gpu_peak_finding_kernel(bool reduce_only);

// test_gpu_tree_gridding_kernel.cu
extern void test_gpu_tree_gridding_kernel();

// test_dedisperser.cu
extern void test_dedisperser(const DedispersionConfig &config, int nchunks);
extern void test_dedisperser();

// "Zombie" tests (code that I wrote during protoyping that may never get used)
extern void test_avx2_m64_outbuf();
extern void test_cpu_downsampler();
extern void test_gpu_downsample();
extern void test_gpu_transpose();
extern void test_gpu_reduce2();


}  // namespace pirate

#endif // _PIRATE_TESTS_HPP
