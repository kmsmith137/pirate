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
	int state_nelts_per_beam = 0;
	int shmem_nbytes = 0;

	// Each threadblock is a shape-(Wx,Wy,Wz) thread array, where Wx=32.
	int Wy = 0;
	int Wz = 0;
	int warps_per_threadblock = 0;  // = (Wy * Wz)

	 // Each kernel is a shape-(Bx,By,B) array, where B = number of beams 
	int Bx = 0;
	int By = 0;
	int threadblocks_per_beam = 0;  // = (Bx * By)
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
    
protected:
    using T32 = typename simd32_type<T>::type;

    using kernel_t = void (*)(const T32 *,   // in
			      T32 *,         // out
			      int,           // ntime_in
			      long,          // ntime_cumulative
			      long,          // bstride_in
			      long,          // bstride_out
			      T32 *);        // persistent_state

    kernel_t kernel;

    // Constructor is protected (to construct GpuLaggedDownsamplingKernel
    // instances, call the public factory function make() above).

    GpuLaggedDownsamplingKernel(const Params &params, kernel_t kernel);
};


}  // namespace pirate

#endif // _PIRATE_INTERNALS_GPU_LAGGED_DOWNSAMPLING_KERNEL_HPP
