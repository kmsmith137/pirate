#ifndef _PIRATE_INTERNALS_GPU_DEDISPERSION_KERNEL_HPP
#define _PIRATE_INTERNALS_GPU_DEDISPERSION_KERNEL_HPP

#include <gputils/Array.hpp>

namespace pirate {
#if 0
}  // editor auto-indent
#endif


// FIXME temporary hack that will go away, after I implement gputils::Array<void>.
struct UntypedArray
{
    gputils::Array<float> data_float32;
    gputils::Array<__half> data_float16;

    UntypedArray slice(int axis, int start, int stop) const;
    void fill(const UntypedArray &x);

    // Helper for slice(), fill()
    bool _is_float32(const char *name) const;
};


// The GpuDedispersionKernel uses externally-allocated buffers for its inputs/outputs,
// but internally allocates and manages its persistent state ("rstate").

class GpuDedispersionKernel
{
public:
   
    // The meaning of Params::apply_residual_lags needs some explanation!
    //
    // This is used in the second dedisperser stage, where each tree channel is labelled
    // by two indices:
    //
    //   - a bit-reversed DM 0 <= d < 2^(total_rank-active_rank)
    //   - a coarse frequency 0 <= f < 2^(active_rank).
    //
    // Before dedispersing the data, the following residual lag is applied:
    //
    //   int lag = rb_lag(f, d, total_rank-active_rank, active_rank, params.input_is_downsampled_tree);
    //   int residual_lag = lag % nelts_per_segment;

    struct Params {
	std::string dtype;      // either "float32" or "float16"
	int rank = -1;          // satisfies 1 <= rank <= 8
	int nambient = 0;
	int total_beams = 0;
	int beams_per_kernel_launch = 0;
	int ntime = 0;

	// Input/output buffer types.
	bool input_is_ringbuf = false;
	bool output_is_ringbuf = false;

	// Residual lags (see comment above).
	bool apply_input_residual_lags = false;
	bool input_is_downsampled_tree = false;   // only matters if apply_input_residual_lags=true
	int nelts_per_segment = 0;                // only matters if apply_input_residual_lags=true

	// Placeholder for future expansion.
	// Only used if (input_is_ringbuf || output_is_ringbuf)
	gputils::Array<uint> ringbuf_locations;

	// Returns true if (dtype == "float32"), false if (dtype == "float16").
	// Otherwise, throws an exception.
	bool is_float32() const;

	// Throws an exception if anything is wrong.
	void validate() const;
    };

    // To construct GpuDedispersionKernel instances, call this function.
    static std::shared_ptr<GpuDedispersionKernel> make(const Params &params);
    
    Params params;
    
    // Used internally by GpuDedispersionKernel::launch().
    int state_nelts_per_beam = 0;
    int warps_per_threadblock = 0;
    int shmem_nbytes = 0;
    
    // launch(): asynchronously launch dedispersion kernel, and return without synchronizing stream.
    //
    // The 'in' array has different meanings, depending on Params::input_is_ringbuf:
    //   - If (!input_is_ringbuf): shape is (nbeams, nambient, pow2(rank), ntime).
    //   - If (input_is_ringbuf): shape is (ntime/nelts_per_segment, nambient, pow2(rank), 4).
    //
    // Similarly, the 'out' array has different meanings, depending on Params::output_is_ringbuf:
    //   - If (!output_is_ringbuf): shape is (nbeams, nambient, pow2(rank), ntime).
    //   - If (output_is_ringbuf): shape is (ntime/nelts_per_segment, nambient, pow2(rank), 4).
    //
    // The 'itime' and 'ibeam' arguments are not logically necessary, but enable a debug check.

    virtual void launch(const UntypedArray &in, UntypedArray &out, long itime, long ibeam, cudaStream_t stream=nullptr) = 0;

protected:
    // Don't call constructor directly -- call GpuDedispersionKernel::make() instead!
    GpuDedispersionKernel(const Params &params);

    // Shape (total_beams, state_nelts_per_beam).
    UntypedArray persistent_state;

    // Enables a debug check.
    long expected_itime = 0;
    long expected_ibeam = 0;

    // FIXME only on current cuda device (at time of construction).
    // FIXME should either add run-time check, or switch to using constant memory.
    gputils::Array<uint> integer_constants;
};


}  // namespace pirate

#endif // _PIRATE_INTERNALS_GPU_DEDISPERSION_KERNEL_HPP
