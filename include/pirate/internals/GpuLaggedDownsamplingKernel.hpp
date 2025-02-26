#ifndef _PIRATE_INTERNALS_GPU_LAGGED_DOWNSAMPLING_KERNEL_HPP
#define _PIRATE_INTERNALS_GPU_LAGGED_DOWNSAMPLING_KERNEL_HPP

#include <memory>
#include <iostream>
#include <cuda_fp16.h>  // __half, __half2

#include <ksgpu/Array.hpp>
#include "UntypedArray.hpp"


namespace pirate {
#if 0
}  // editor auto-indent
#endif


class GpuLaggedDownsamplingKernel
{
public:
    struct Params {
	
	// A potential source of confusion: denote
	//
	//    ld_nds = GpuLaggedDownsampler::Params::num_downsmapling_levels
	//    dc_nds = DedispersionConfig::num_downsampling_levels
	//
	// Then ld_nds = (dc_nds - 1)!

	std::string dtype;      // either "float32" or "float16"
	
	long small_input_rank = 0;
	long large_input_rank = 0;
	long num_downsampling_levels = 0;

	// Returns true if (dtype == "float32"), false if (dtype == "float16").
	// Otherwise, throws an exception.
	bool is_float32() const;

	// Throws an exception if anything is wrong.
	void validate() const;
    };

    // Factory function used to construct new GpuLaggedDownsamplingKernel objects.
    static std::shared_ptr<GpuLaggedDownsamplingKernel> make(const Params &params);
    
    // launch() arguments:
    //
    //  - 'in': array of shape (nbeams, 2^(large_input_rank), ntime).
    //
    //      Must have contiguous freq/time axes, but beam axis can have arbitrary stride.
    //
    //  - 'out': vector of length (num_downsampling_levels).
    //
    //       The i-th element should have shape (nbeams, 2^(large_input_rank-1), ntime/2^(i+1)).
    //       Must have contiguous freq/time axes, but beam axis can have arbitrary stride.
    //
    //       There is also an "adjacency" requirement: out[i][0,:,:] and out[i+1][0,:,:] must
    //       be adjacent in memory. Relatedly, all 'out' arrays must have the same beam stride.
    //
    //  - 'persistent_state': contiguous array of shape (nbeams, state_nelts_per_beam)
    //      Must be zeroed on first call to launch().
    
    virtual void launch(const UntypedArray &in,
			std::vector<UntypedArray> &out,
			UntypedArray &persistent_state,
			long ntime_cumulative,
			cudaStream_t stream=nullptr) = 0;

    const Params params;
    
    // These parameters determine how the kernel is divided into threadblocks.
    // See GpuLaggedDownsamplingKernel.cu for more info.
    int M_W = 0;
    int M_B = 0;
    int A_W = 0;
    int A_B = 0;
    
    // More parameters computed in constructor.
    int ntime_divisibility_requirement = 0;
    int shmem_nbytes_per_threadblock = 0;
    int state_nelts_per_beam = 0;

    void print(std::ostream &os=std::cout, int indent=0) const;

protected:
    // Constructor is protected -- use GpuLaggedDownsamplingKernel::make() instead.
    GpuLaggedDownsamplingKernel(const Params &params);
};


}  // namespace pirate

#endif // _PIRATE_INTERNALS_GPU_LAGGED_DOWNSAMPLING_KERNEL_HPP
