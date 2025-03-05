#ifndef _PIRATE_INTERNALS_LAGGED_DOWNSAMPLING_KERNEL_HPP
#define _PIRATE_INTERNALS_LAGGED_DOWNSAMPLING_KERNEL_HPP

#include <vector>
#include <memory>
#include <iostream>
#include "ReferenceLagbuf.hpp"

#include <ksgpu/Dtype.hpp>
#include <ksgpu/Array.hpp>


namespace pirate {
#if 0
}  // editor auto-indent
#endif

class DedispersionPlan;


struct LaggedDownsamplingKernelParams
{
    // FIXME explain these parameters better.
    //
    // FIXME a potential source of confusion: denote
    //
    //    ld_nds = LaggedDownsamplingKernelParams::num_downsmapling_levels
    //    dc_nds = DedispersionConfig::num_downsampling_levels
    //
    // Then ld_nds = (dc_nds - 1)!

    ksgpu::Dtype dtype;                 // same as DedispersionConfig::dtype
    long small_input_rank = -1;         // same as DedispersionPlan::stage0_trees[1].rank0 + 1
    long large_input_rank = -1;         // same as DedispersionConfig::tree_rank;
    long num_downsampling_levels = -1;  // same as (DedispersionConfig::num_downsampling_levels - 1)
    long total_beams = 0;               // same as DedispersionConfig::beams_per_gpu
    long beams_per_batch = 0;           // same as DedispersionConfig::beams_per_batch
    long ntime = 0;                     // same as DedispersionConfig::time_samples_per_chunk

    LaggedDownsamplingKernelParams() { }
    LaggedDownsamplingKernelParams(const std::shared_ptr<DedispersionPlan> &plan);
    LaggedDownsamplingKernelParams(const LaggedDownsamplingKernelParams &) = default;
    
    bool operator==(const LaggedDownsamplingKernelParams &) const;

    void print(std::ostream &os=std::cout, int indent=0) const;
    void validate() const;  // throws an exception if anything is wrong
};


// Output buffers for one "batch" of beams.
struct LaggedDownsamplingKernelOutbuf
{
    const LaggedDownsamplingKernelParams params;
    const long min_beam_stride = 0;

    // big_arr has shape (beams_per_batch, min_beam_stride_out).
    // The beam axis may have a non-contiguous stride.
    ksgpu::Array<void> big_arr;

    // small_args has length (params.num_downsampling_levels).
    // small_arrs[ids] has shape (beams_per_batch, 2^(large_input_rank-1), nt_chunk/2^(ids+1)).
    // The beam axis will usually have a non-contiguous stride.
    std::vector<ksgpu::Array<void>> small_arrs;

    LaggedDownsamplingKernelOutbuf(const LaggedDownsamplingKernelParams &params);
    LaggedDownsamplingKernelOutbuf(const std::shared_ptr<DedispersionPlan> &plan);
    
    void allocate(long beam_stride, int aflags);
    void allocate(int aflags);  // use min_beam_stride_out

    bool is_allocated() const;
    bool on_host() const;  // returns true if unallocated, or if (num_downsamping_levels == 0)
    bool on_gpu() const;   // returns true if unallocated, or if (num_downsamping_levels == 0)
};


// The reference kernel allocates persistent state internally.
// (Note that the apply() method takes an 'ibatch' argument, so that the correct persistent state can be used.)
struct ReferenceLaggedDownsamplingKernel2
{
    const LaggedDownsamplingKernelParams params;
    const int nbatches;  // same as (params.total_beams / params.beams_per_batch)

    ReferenceLaggedDownsamplingKernel2(const LaggedDownsamplingKernelParams &params);
    
    void apply(const ksgpu::Array<void> &in, LaggedDownsamplingKernelOutbuf &out, long ibatch);
    
    std::vector<ReferenceLagbuf> lagbufs_small;  // length nbatches
    std::vector<ReferenceLagbuf> lagbufs_large;  // length nbatches
    std::shared_ptr<ReferenceLaggedDownsamplingKernel2> next;

    // Helper for apply().
    void _apply(const ksgpu::Array<float> &in, ksgpu::Array<void> *outp, long ibatch);
};
    

// The GPU kernel does not allocate persistent state -- the caller is responsible.
// (Note that the launch() method takes a 'persistent_state' argument, but not an 'ibatch' argument.)

class GpuLaggedDownsamplingKernel2
{
    using Params = LaggedDownsamplingKernelParams;
    
public:
    const Params params;
    
    // Factory function used to construct new GpuLaggedDownsamplingKernel objects.
    static std::shared_ptr<GpuLaggedDownsamplingKernel2> make(const Params &params);
    
    // These parameters determine how the kernel is divided into threadblocks.
    // See LaggedDownsamplingKernel.cu for more info.
    int M_W = 0;
    int M_B = 0;
    int A_W = 0;
    int A_B = 0;
    
    // More parameters computed in constructor.
    int shmem_nbytes_per_threadblock = 0;
    int state_nelts_per_beam = 0;
    
    // 'in': array of shape (nbeams_per_batch, 2^(large_input_rank), ntime).
    // Must have contiguous freq/time axes, but beam axis can have arbitrary stride.
    //
    // 'persistent_state': contiguous array of shape (nbeams_per_batch, state_nelts_per_beam)
    // Must be zeroed on first call to launch().
    //
    // 'ntime_cumulative': total number of time samples processed by previous
    // calls to launch(), **for this beam batch**.
    //
    // (FIXME could get rid of ntime_cumulative -- it's a crutch for the GPU kernel
    // that isn't really necessary.)
    
    virtual void launch(
        const ksgpu::Array<void> &in,
	LaggedDownsamplingKernelOutbuf &out,
	ksgpu::Array<void> &persistent_state,
	long ntime_cumulative,
	cudaStream_t stream   // NULL stream is allowed, but is not the default
    ) = 0;

    void print(std::ostream &os=std::cout, int indent=0) const;

protected:
    // Constructor is protected -- use GpuLaggedDownsamplingKernel2::make() instead.
    GpuLaggedDownsamplingKernel2(const Params &params);
};


}  // namespace pirate

#endif // _PIRATE_INTERNALS_AGGED_DOWNSAMPLING_KERNEL_HPP
