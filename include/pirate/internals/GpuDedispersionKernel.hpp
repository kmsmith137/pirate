#ifndef _PIRATE_INTERNALS_GPU_DEDISPERSION_KERNEL_HPP
#define _PIRATE_INTERNALS_GPU_DEDISPERSION_KERNEL_HPP

#include "inlines.hpp"  // simd32_type
#include <gputils/Array.hpp>

namespace pirate {
#if 0
}  // editor auto-indent
#endif


template<typename T>
class GpuDedispersionKernel
{
public:
    // Each kernel has an rlag_type which determines whether residual lags are applied.
    // See long comment below for details.
    
    enum RLagType {
	RLagInvalid = 0,
	RLagNone = 1,
	RLagInput = 2
	// I may add more later! (FIXME explain this)
    };

    struct Params {
	int rank = 0;
	RLagType rlag_type = RLagInvalid;

	// Kernel persistent state is an array of shape
	// { nbeams, nambient, state_nelts_per_small_tree }.
	long state_nelts_per_small_tree = 0;

	// Used internally by GpuDedispersionKernel::launch().
	int warps_per_threadblock = 0;
	int shmem_nbytes = 0;
    };

    const Params params;

    // Array interface to launch().
    //
    // The 'iobuf' amd 'rstate' arrays must have shapes
    //   int N = constants::bytes_per_gpu_cache_line / sizeof(T);
    //   iobuf.shape = { nbeams, nambient, 2^rank, ntime }
    //   rstate.shape = { nbeams, nambient, state_nelts_per_small_tree }
    //
    // This may be more array dimensions than you need -- if so, just call Array::reshape().
    //
    // The last axis of 'iobuf' (representing time) must be contiguous, and the 'rstate'
    // array must be fully contiguous. Remaining iobuf axis strides (beam, ambient, row)
    // are kernel arguments.
    //
    // The 'beam' index is always a "pure spectator" index.
    // The meaning of the 'ambient' index depends on the rlag_type:
    //
    //   - RLagNone:
    //
    //       Ambient index is a pure spectator.
    //
    //   - RLagInput:
    //
    //       Ambient index represents a bit-reversed DM 0 <= d < 2^(ambient_rank).
    //       The "row" index represents a coarse frequency 0 <= f < 2^(rank).
    //       Before dedispersing the data, the following residual lag is applied:
    //
    //        constexpr int N = constants::bytes_per_gpu_cache_line / sizeof(T)
    //        int ff = 2^rank - 1 - f;
    //        int lag = (ff * d) % N;
    
    void launch(gputils::Array<T> &iobuf,
		gputils::Array<T> &rstate,
		cudaStream_t stream = nullptr) const;
    
    // Bare pointer interface to launch().
    // All stride arguments are "T" strides, i.e. byte offset is obtained by multiplying by sizeof(T).
    
    void launch(T *iobuf, T *rstate,
		long nbeams, long beam_stride,
		long nambient, long ambient_stride,
		long row_stride, long ntime,   // number of rows is always 2^rank
		cudaStream_t stream = nullptr) const;
    
    // Use this factory function to create GpuDedispersionKernel instances (constructor is protected).
    static std::shared_ptr<GpuDedispersionKernel> make(int rank, RLagType rlag_type);

    void print(std::ostream &os=std::cout, int indent=0) const;
    
    // For printing RLagType (defining operator<<() didn't work due to C++ weirdness)
    static std::string rlag_str(RLagType rlag_type);

protected:
    // If T==float, then T32 is also 'float'.
    // If T==__half, then T32 is '__half2'.
    using T32 = typename simd32_type<T>::type;
    
    // This is the __global__ CUDA kernel (defined in GpuDedispersionKernel.cu)
    // FIXME replace 'integer_constants' argument with constant memory.
    //
    // A few notes, relative to launch() above:
    //
    //   - The 'nbeams' and 'nambient' arguments to launch() are implicitly
    //     supplied to the kernel via (GridDims.y, GridDims.x).
    //
    //   - Instead of using 'ntime' as a kernel argument, we use
    //
    //       nt_cl = number of cache lines spanned by 'ntime' time samples
    //             = (ntime * sizeof(T)) / 128
    //
    //   - Some arguments (row_stride, nt_cl) are 32-bit in the cuda kernel,
    //     but 64-bit in launch() above. The necessary overflow checking
    //     is done in launch().
    
    using kernel_t = void (*)(T32 *,            // iobuf
			      T32 *,            // rstate
			      long,             // beam_stride
			      long,             // ambient_stride
			      int,              // row_stride
			      int,              // nt_cl
			      unsigned int *);  // integer_constants

    kernel_t kernel;

    // FIXME only on current cuda device (at time of construction).
    // Should either add run-time check, or switch to using constant memory.
    gputils::Array<unsigned int> integer_constants;
    
    // Protected constructor (called by GpuDedispersionKernel::make())
    GpuDedispersionKernel(const Params &params, kernel_t kernel,
			  const gputils::Array<unsigned int> &integer_constants);
};


}  // namespace pirate

#endif // _PIRATE_INTERNALS_GPU_DEDISPERSION_KERNEL_HPP
