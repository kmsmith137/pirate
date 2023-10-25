#ifndef _PIRATE_INTERNALS_GPU_LAGGED_DOWNSAMPLING_KERNEL_HPP
#define _PIRATE_INTERNALS_GPU_LAGGED_DOWNSAMPLING_KERNEL_HPP

#include <memory>       // shared_ptr
#include <cuda_fp16.h>  // __half, __half2
#include <gputils/Array.hpp>

#include "inlines.hpp"  // simd32_type

namespace pirate {
#if 0
}  // editor auto-indent
#endif


template<typename T>
class GpuLaggedDownsamplingKernel
{
public:
    struct Params {
	// "Primary" parameters specified when make() is called.
	int small_input_rank = 0;
	int large_input_rank = 0;
	int num_downsampling_levels = 0;

	// "Derived" parameters.
	int ntime_divisibility_requirement = 0;
	int shmem_nbytes_per_threadblock = 0;
	int state_nelts_per_beam = 0;

	// These parameters are supplied automatically by make().
	// They determine how the kernel is divided into threadblocks.
	// See GpuLaggedDownsamplingKernel.cu for more info.
	int M_W = 0;
	int M_B = 0;
	int A_W = 0;
	int A_B = 0;

	int warps_per_threadblock() const { return M_W * A_W; }
	int threadblocks_per_beam() const { return M_B * A_B; }
    };

    const Params params;

    // To construct instances of 'class GpuLaggedDownsamplingKernel',
    // use this factory function (not constructor, which  is private).
    
    static std::shared_ptr<GpuLaggedDownsamplingKernel<T>> make(
        int small_input_rank,
	int large_input_rank,
	int num_downsampling_levels
    );

    // launch() arguments:
    //
    //  - 'in': array of shape (nbeams, 2^(large_input_rank), ntime).
    //
    //  - 'out': vector of length (num_downsampling_levels).
    //      The i-th element should have shape (nbeams, 2^(large_input_rank-1), ntime/2^(i+1)).
    //
    //  - 'persistent_state': contiguous array of shape (nbeams, state_nelts_per_beam)
    //      Must be zeroed on first call to launch().
    // 
    // XXX explain stride requirements for 'in' and 'out'.
    
    void launch(const gputils::Array<T> &in,
		std::vector<gputils::Array<T>> &out,
		gputils::Array<T> &persistent_state,
		long ntime_cumulative,
		cudaStream_t stream = nullptr) const;

    void print(std::ostream &os=std::cout, int indent=0) const;

    using T32 = typename simd32_type<T>::type;

    using kernel_t = void (*)(const T32 *,   // in
			      T32 *,         // out
			      int,           // ntime_in
			      long,          // ntime_cumulative
			      long,          // bstride_in
			      long,          // bstride_out
			      T32 *);        // persistent_state
    
protected:
    kernel_t kernel;

    // Constructor is protected (to construct GpuLaggedDownsamplingKernel
    // instances, call the public factory function make() above).

    GpuLaggedDownsamplingKernel(const Params &params, kernel_t kernel);
};


}  // namespace pirate

#endif // _PIRATE_INTERNALS_GPU_LAGGED_DOWNSAMPLING_KERNEL_HPP
