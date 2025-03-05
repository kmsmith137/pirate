#ifndef _PIRATE_INTERNALS_DEDISPERSION_BUFFERS_HPP
#define _PIRATE_INTERNALS_DEDISPERSION_BUFFERS_HPP

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


struct DedispersionInbufParams
{
    // FIXME rename and explain the *_input_rank params.
    ksgpu::Dtype dtype;                 // same as DedispersionConfig::dtype
    long small_input_rank = -1;         // same as DedispersionPlan::stage0_trees[1].rank0 + 1
    long large_input_rank = -1;         // same as DedispersionConfig::tree_rank;
    long num_downsampling_levels = -1;  // same as DedispersionConfig::num_downsampling_levels
    long total_beams = 0;               // same as DedispersionConfig::beams_per_gpu
    long beams_per_batch = 0;           // same as DedispersionConfig::beams_per_batch
    long ntime = 0;                     // same as DedispersionConfig::time_samples_per_chunk

    DedispersionInbufParams() { }
    DedispersionInbufParams(const std::shared_ptr<DedispersionPlan> &plan);
    DedispersionInbufParams(const DedispersionInbufParams &) = default;
    
    bool operator==(const DedispersionInbufParams &) const;

    void print(std::ostream &os=std::cout, int indent=0) const;
    void validate() const;  // throws an exception if anything is wrong
};


// Input buffers for one "batch" of beams.
struct DedispersionInbuf
{
    const DedispersionInbufParams params;

    // If allocate() has not been called, then 'bufs' is an empty vector.
    //
    // If allocate() has been called, then:
    //   - The 'bufs' vector has length (num_downsampling_levels).
    //   - bufs[0] has shape (beams_per_batch, 2^(large_input_rank), ntime).
    //   - bufs[i] has shape (beams_per_batch, 2^(large_input_rank-1), ntime/2^i), for i >= 1.
    //
    // The beam axes of these arrays will have a non-contiguous stride (unless
    // num_downsampling_levels == 0).
    
    std::vector<ksgpu::Array<void>> bufs;

    DedispersionInbuf(const DedispersionInbufParams &params);
    DedispersionInbuf(const std::shared_ptr<DedispersionPlan> &plan);
    
    void allocate(int aflags);

    bool is_allocated() const;
    bool on_host() const;  // throws exception if unallocated
    bool on_gpu() const;   // throws exception if unallocated

    // Reference to "ambient" array (each element of 'bufs' is a non-contiguous subarray).
    ksgpu::Array<void> ref;
};


}  // namespace pirate

#endif // _PIRATE_INTERNALS_DEDISPERSION_BUFFERS_HPP
