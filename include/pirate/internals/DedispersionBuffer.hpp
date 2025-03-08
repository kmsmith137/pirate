#ifndef _PIRATE_INTERNALS_DEDISPERSION_BUFFER_HPP
#define _PIRATE_INTERNALS_DEDISPERSION_BUFFER_HPP

#include <vector>
#include <memory>
#include <iostream>

#include <ksgpu/Dtype.hpp>
#include <ksgpu/Array.hpp>


namespace pirate {
#if 0
}  // editor auto-indent
#endif


// Represents a sequence of buffers, with shape (beams_per_batch, 2^rank, ntime).
struct DedispersionBufferParams
{
    ksgpu::Dtype dtype;

    long beams_per_batch = 0;
    long nbuf = -1;

    std::vector<long> buf_rank;   // length nbuf
    std::vector<long> buf_ntime;  // length nbuf

    DedispersionBufferParams() { }
    DedispersionBufferParams(const DedispersionBufferParams &) = default;

    void print(std::ostream &os=std::cout, int indent=0) const;
    void validate() const;    // throws an exception if anything is wrong

    // Total number of array elements (not bytes), for one batch of beams.
    long get_nelts() const;
};


// Input buffers for one "batch" of beams.
struct DedispersionBuffer
{
    DedispersionBuffer() { }
    DedispersionBuffer(const DedispersionBufferParams &params);
    
    DedispersionBufferParams params;

    // If allocate() has not been called, then 'bufs' is an empty vector.
    //
    // If allocate() has been called, then:
    //   - The 'bufs' vector has length (params.nbuf).
    //   - bufs[i] has shape (params.beams_per_batch, pow2(params.buf_rank[i]), params.buf_ntime[i]).
    //
    // The beam axes of these arrays will have non-contiguous strides (unless params.nbuf==1).
    
    std::vector<ksgpu::Array<void>> bufs;
    bool is_allocated = false;
    
    void allocate(int aflags);
    bool on_host() const;  // throws exception if unallocated
    bool on_gpu() const;   // throws exception if unallocated

    // Total number of array elements (not bytes), for one batch of beams.
    // Does not require array to be allocated.
    long get_nelts() const { return params.get_nelts(); }

    // Reference to "ambient" array (each element of 'bufs' is a non-contiguous subarray).
    ksgpu::Array<void> ref;
};


}  // namespace pirate

#endif // _PIRATE_INTERNALS_DEDISPERSION_BUFFER_HPP
