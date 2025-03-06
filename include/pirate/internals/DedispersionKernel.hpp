#ifndef _PIRATE_INTERNALS_DEDISPERSION_KERNEL_HPP
#define _PIRATE_INTERNALS_DEDISPERSION_KERNEL_HPP

#include <ksgpu/Array.hpp>
#include "ReferenceTree.hpp"
#include "ReferenceLagbuf.hpp"

namespace pirate {
#if 0
}  // editor auto-indent
#endif


// Dedispersion kernels have two ranks:
//
//   dd_rank = "dedispersion" rank
//           = log2(number of "active" tree channels)
//
//   amb_rank = "ambient" rank
//            = log2(number of "spectator" tree channels)
//
// The dedispersion kernel operates on a buffer of logical shape
//   (beams_per_batch, pow2(amb_rank), pow2(dd_rank), ntime).
//
// The buffer is not assumed contiguous -- this detail is important, since the
// amb/dd axes may be swapped (relative to contiguous ordering). To explain this in
// more detail, consider two-stage tree dedispersion with total rank (rank1 + rank2):
//
//  - Let's "reshape" the dedispersion buffer to a 4-d array
//
//      (nbeams, pow2(rank2), pow2(rank1), ntime)
//
//    This 4-d shape will be preserved throughout the two-stage dedispersion, but
//    the meaning of the axes will be different at different stages.
//
//  - In the input array, the meaning of the axes is:
//
//        (beam, coarse freq, fine freq, time).
//
//  - When we apply the first dedispersion kernel, we map these axes to:
//
//        (beam, amb, dd, time)       (*)
//
//  - After the first dedispersion kernel, the meaning of the axes is:
//
//        (beam, coarse freq, bit-reversed coarse dm, time)
//
//  - When we apply the second dedispersion kernel, we map these axes to:
//
//        (beam, dd, amb, time)      (**)
//
//    Note the transpose in (**) relative to (*), which implies that
//    dedispersion kernels will sometimes need non-contiguous strides.
//
//  - In the output array (after the second dedispersion kernel), the
//    meaning of the axes is:
//
//        (beam, bit-reversed fine dm, bit-reversed coarse dm, time)
//
//   Note that this cleanly "reshapes" to a 3-d array:
//
//        (beam, bit-reversed total dm, time)
    

struct DedispersionKernelParams
{
    ksgpu::Dtype dtype;   // either float32 or float16
    long dd_rank = -1;    // satisfies 1 <= dd_rank <= 8
    long amb_rank = -1;   // satisfies 0 <= amb_rank <= 8
    long total_beams = 0;
    long beams_per_batch = 0;
    long ntime = 0;       // includes downsampling factor, if any
    
    // Input/output buffer types.
    bool input_is_ringbuf = false;
    bool output_is_ringbuf = false;
    
    // Residual lags (see comment above).
    // The meaning of Params::apply_residual_lags needs some explanation!
    //
    // This is used in the second dedisperser stage, where each tree channel is labelled
    // by two indices:
    //
    //   - a bit-reversed DM 0 <= d < 2^amb_rank
    //   - a coarse frequency 0 <= f < 2^dd_rank
    //
    // Before dedispersing the data, the following residual lag is applied:
    //
    //   int lag = rb_lag(f, d, amb_rank, dd_rank, params.input_is_downsampled_tree);
    //   int residual_lag = lag % nelts_per_segment;
    
    bool apply_input_residual_lags = false;
    bool input_is_downsampled_tree = false;   // only matters if apply_input_residual_lags=true
    int nelts_per_segment = 0;                // only matters if apply_input_residual_lags=true

    // Only used if (input_is_ringbuf || output_is_ringbuf)
    // Reminder: ringbuf_locations has shape (nsegments_per_tree, 4), where:
    //   nsegments_per_tree = pow2(dd_rank + amb_rank) * xdiv(ntime,nelts_per_segment)
    ksgpu::Array<uint> ringbuf_locations;
    long ringbuf_nseg = 0;
    
    // Throws an exception if anything is wrong.
    void validate(bool on_gpu) const;
};


// The reference kernel allocates persistent state internally.
// (Note that the apply() method takes an 'ibatch' argument, so that the correct persistent state can be used.)

struct ReferenceDedispersionKernel
{
    using Params = DedispersionKernelParams;

    ReferenceDedispersionKernel(const Params &params);

    const Params params;  // reminder: contains 'ringbuf_locations' array.
    
    // The 'in' and 'out' arrays are either dedispersion buffers or ringbufs, depending on
    // values of Params::input_is_ringbuf and Params::output_is_ringbuf. Shapes are:
    //
    //   - dedispersion buffer has shape (params.beams_per_batch, pow2(amb_rank), pow2(dd_rank), ntime).
    //   - ringbuf has 1-d shape (params.ringbuf_nseg * params.nelts_per_segment,)

    void apply(ksgpu::Array<void> &in, ksgpu::Array<void> &out, long ibatch, long it_chunk);

    long nbatches = 0;     // same as (params.total_beams / params.beams_per_batch)
    std::vector<std::shared_ptr<ReferenceTree>> trees;        // length (nbatches)
    std::vector<std::shared_ptr<ReferenceLagbuf>> rlag_bufs;  // length (params.apply_input_residual_lags ? nbatches : 0)

    // Used internally
    void _copy_to_ringbuf(const ksgpu::Array<float> &in, ksgpu::Array<float> &out, long rb_pos);
    void _copy_from_ringbuf(const ksgpu::Array<float> &in, ksgpu::Array<float> &out, long rb_pos);
};


// The GpuDedispersionKernel uses externally-allocated buffers for its inputs/outputs,
// but internally allocates and manages its persistent state ("rstate").

class GpuDedispersionKernel
{
    using Params = DedispersionKernelParams;
    
public:   
    // To construct GpuDedispersionKernel instances, call this function.
    static std::shared_ptr<GpuDedispersionKernel> make(const Params &params);
    
    Params params;   // reminder: contains 'ringbuf_locations' array

    void allocate();
    bool is_allocated() const;
    
    // launch(): asynchronously launch dedispersion kernel, and return without synchronizing stream.
    //
    // The 'in' and 'out' arrays are either dedispersion buffers or ringbufs, depending on
    // values of Params::input_is_ringbuf and Params::output_is_ringbuf. Shapes are:
    //
    //   - dedispersion buffer has shape (params.beams_per_batch, pow2(amb_rank), pow2(dd_rank), ntime).
    //   - ringbuf has 1-d shape (params.ringbuf_nseg * params.nelts_per_segment,)
    
    virtual void launch(
        ksgpu::Array<void> &in,
	ksgpu::Array<void> &out,
	long ibatch,
	long it_chunk,
	cudaStream_t stream  // NULL stream is allowed, but is not the default
    ) = 0;
    
    // Used internally by GpuDedispersionKernel::launch().
    long nbatches = 0;
    long state_nelts_per_beam = 0;
    long warps_per_threadblock = 0;
    long shmem_nbytes = 0;

protected:
    // Don't call constructor directly -- call GpuDedispersionKernel::make() instead!
    GpuDedispersionKernel(const Params &params);

    // Shape (total_beams, state_nelts_per_beam).
    ksgpu::Array<void> persistent_state;

    // FIXME only on current cuda device (at time of construction).
    // FIXME should either add run-time check, or switch to using constant memory.
    ksgpu::Array<uint> integer_constants;
};


}  // namespace pirate

#endif // _PIRATE_INTERNALS_DEDISPERSION_KERNEL_HPP
