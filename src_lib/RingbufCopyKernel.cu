#include "../include/pirate/RingbufCopyKernel.hpp"
#include "../include/pirate/BumpAllocator.hpp"
#include "../include/pirate/constants.hpp"  // bytes_per_gpu_cache_line
#include "../include/pirate/inlines.hpp"    // xdiv()

#include <ksgpu/xassert.hpp>
#include <ksgpu/cuda_utils.hpp>
#include <ksgpu/rand_utils.hpp>
#include <ksgpu/test_utils.hpp>

using namespace std;
using namespace ksgpu;

namespace pirate {
#if 0
}  // editor auto-indent
#endif


// -------------------------------------------------------------------------------------------------


const RingbufCopyKernelParams &RingbufCopyKernelParams::validate() const
{
    xassert_gt(total_beams, 0);
    xassert_gt(beams_per_batch, 0);
    xassert(octuples.on_host());

    // Currently this is all we need.
    xassert((nelts_per_segment == 32) || (nelts_per_segment == 64));
    
    // Currently assumed throughout the pirate code.
    xassert_divisible(total_beams, beams_per_batch);

    // Locations array can either be size-zero, or shape-(2N,4) contiguous.
    if (octuples.size != 0) {
        xassert_eq(octuples.ndim, 2);
        xassert_eq(octuples.shape[1], 4);
        xassert_divisible(octuples.shape[0], 2);
        xassert(octuples.is_fully_contiguous());
    }

    return *this;
}


// -------------------------------------------------------------------------------------------------


CpuRingbufCopyKernel::CpuRingbufCopyKernel(const RingbufCopyKernelParams &params_) :
    params(params_.validate()),
    noctuples(xdiv(params_.octuples.size, 8))
{
    // Note: only correct if (nbytes_per_segment == constants::bytes_per_gpu_cache_line).
    long nbytes_per_octuple = (2 * params.beams_per_batch * constants::bytes_per_gpu_cache_line) + 8;
    bw_per_launch.nbytes_hmem = noctuples * nbytes_per_octuple;
}


// Helper for CpuRingbufCopyKernel::apply()
// B = bytes per segment

template<int B>
static void _cpu_copy(void *ringbuf, const uint *octuples, long noctuples, int nbeams, ulong iframe)
{
    char *rp = reinterpret_cast<char *> (ringbuf);
    
    for (long i = 0; i < noctuples; i++) {
        uint src_offset = octuples[8*i];     // in segments, not bytes
        uint src_phase = octuples[8*i+1];    // index of (time chunk, beam) pair, relative to current pair
        uint src_len = octuples[8*i+2];      // number of (time chunk, beam) pairs in ringbuf (same as Ringbuf::frames_in_zone)
        uint src_nseg = octuples[8*i+3];     // number of segments per (time chunk, beam) (same as Ringbuf::nseg_per_beam)
        
        uint dst_offset = octuples[8*i+4];
        uint dst_phase = octuples[8*i+5];
        uint dst_len = octuples[8*i+6];
        uint dst_nseg = octuples[8*i+7];

        // Absorb iframe into phases. (Note that 'iframe' has ulong type.)
        src_phase = (ulong(src_phase) + iframe) % ulong(src_len);
        dst_phase = (ulong(dst_phase) + iframe) % ulong(dst_len);

        for (int j = 0; j < nbeams; j++) {
            ulong s = ulong(src_offset + src_phase * src_nseg) * B;
            ulong d = ulong(dst_offset + dst_phase * dst_nseg) * B;

            // FIXME is memmove() fastest here?
            memmove(rp + d, rp + s, B);
            
            // Equivalent to (phase = (phase+1) % len), but avoids cost of %-operator.
            src_phase = (src_phase == src_len-1) ? 0 : (src_phase+1);
            dst_phase = (dst_phase == dst_len-1) ? 0 : (dst_phase+1);
        }
    }
}


void CpuRingbufCopyKernel::apply(ksgpu::Array<void> &ringbuf, long ichunk, long ibatch)
{
    xassert(ringbuf.on_host());
    xassert(ringbuf.ndim == 1);
    xassert(ringbuf.is_fully_contiguous());
    
    xassert(ibatch >= 0);
    xassert(ibatch * params.beams_per_batch < params.total_beams);
    xassert(ichunk >= 0);
    
    ulong iframe = (ichunk * params.total_beams) + (ibatch * params.beams_per_batch);
    long nbits_per_segment = params.nelts_per_segment * ringbuf.dtype.nbits;

    // These two cases are all we currently need.
    if (nbits_per_segment == 1024)
        _cpu_copy<128> (ringbuf.data, params.octuples.data, noctuples, params.beams_per_batch, iframe);
    else if (nbits_per_segment == 2048)
        _cpu_copy<256> (ringbuf.data, params.octuples.data, noctuples, params.beams_per_batch, iframe);
    else {
        stringstream ss;
        ss << "CpuRingbufCopyKernel: expected nbits_per_segment in {1024,2048}, got "
           << params.nelts_per_segment << " (nelts_per_segment=" << params.nelts_per_segment
           << ", dtype=" << ringbuf.dtype << ")";
        throw runtime_error(ss.str());
    }
}


// -------------------------------------------------------------------------------------------------


GpuRingbufCopyKernel::GpuRingbufCopyKernel(const RingbufCopyKernelParams &params_) :
    params(params_.validate()),
    noctuples(xdiv(params_.octuples.size, 8))
{
    long nbytes_per_octuple = (2 * params.beams_per_batch * constants::bytes_per_gpu_cache_line) + 8;
    bw_per_launch.nbytes_gmem = noctuples * nbytes_per_octuple;

    // Compute GPU memory footprint, reflecting logic in allocate().
    long octuples_nbytes = params.octuples.shape[0] * params.octuples.shape[1] * 4;
    this->gmem_footprint_nbytes = align_up(octuples_nbytes, BumpAllocator::nalign);
}


void GpuRingbufCopyKernel::allocate(BumpAllocator &allocator)
{
    if (is_allocated)
        throw runtime_error("double call to GpuRingbufCopyKernel::allocate()");

    if (!(allocator.aflags & af_gpu))
        throw runtime_error("GpuRingbufCopyKernel::allocate(): allocator.aflags must contain af_gpu");
    if (!(allocator.aflags & af_zero))
        throw runtime_error("GpuRingbufCopyKernel::allocate(): allocator.aflags must contain af_zero");

    long nbytes_before = allocator.nbytes_allocated.load();

    // Copy host -> GPU.
    this->gpu_octuples = allocator.allocate_array<uint>({params.octuples.shape[0], params.octuples.shape[1]});
    this->gpu_octuples.fill(params.octuples);

    long nbytes_allocated = allocator.nbytes_allocated.load() - nbytes_before;
    // cout << "GpuRingbufCopyKernel: " << nbytes_allocated << " bytes allocated" << endl;
    xassert_eq(nbytes_allocated, this->gmem_footprint_nbytes);

    this->is_allocated = true;
}


// Thread grid: { 32*W, 1, 1 }.
// Block grid: { B, 1, 1 }.

__global__ void gpu_copy_kernel(uint4 *ringbuf, const uint *octuples, long noctuples, int nbeams, ulong iframe)
{
    // Global thread ID
    long tid = long(blockIdx.x) * long(blockDim.x) + threadIdx.x;

    // "Regulated" thread ID (avoids out-of-range)
    long treg = ((noctuples-1) << 3) + (threadIdx.x & 0x7);
    treg = min(tid, treg);

    // Each warp reads 4 octuples (i.e. 4 src+dst pairs).
    uint loc_data = octuples[treg];

    // Absorb iframe into phases (only valid on laneId=1 mod 4).
    // Note: currently using 9 __shfl_syncs in this function, optimal number is 7.
    uint loc_len = __shfl_sync(~0u, loc_data, (threadIdx.x & 0x1c) + 2);
    uint loc_phase = (ulong(loc_data) + iframe) % ulong(loc_len);

    // "Allgather" src/dst offsets in groups of 8.
    uint src_offset = __shfl_sync(~0u, loc_data, (threadIdx.x & 0x18));
    uint src_phase = __shfl_sync(~0u, loc_phase, (threadIdx.x & 0x18) + 1);   // Note loc_phase here
    uint src_len = __shfl_sync(~0u, loc_data, (threadIdx.x & 0x18) + 2);
    uint src_nseg = __shfl_sync(~0u, loc_data, (threadIdx.x & 0x18) + 3);

    uint dst_offset = __shfl_sync(~0u, loc_data, (threadIdx.x & 0x18) + 4);
    uint dst_phase = __shfl_sync(~0u, loc_phase, (threadIdx.x & 0x18) + 5);   // Note loc_phase here
    uint dst_len = __shfl_sync(~0u, loc_data, (threadIdx.x & 0x18) + 6);
    uint dst_nseg = __shfl_sync(~0u, loc_data, (threadIdx.x & 0x18) + 7);
    
    for (int b = 0; b < nbeams; b++) {
        ulong s = ulong(src_offset + src_phase * src_nseg) << 3;   // int4 offset
        ulong d = ulong(dst_offset + dst_phase * dst_nseg) << 3;   // int4 offset
        uint4 rb_data = ringbuf[s + (threadIdx.x & 0x7)];
        
        if (tid == treg)
            ringbuf[d + (threadIdx.x & 0x7)] = rb_data;

        // Equivalent to (phase = (phase+1) % len), but avoids cost of %-operator.
        src_phase = (src_phase == src_len-1) ? 0 : (src_phase+1);
        dst_phase = (dst_phase == dst_len-1) ? 0 : (dst_phase+1);
    }
}


void GpuRingbufCopyKernel::launch(ksgpu::Array<void> &ringbuf, long ichunk, long ibatch, cudaStream_t stream)
{
    xassert(ringbuf.on_gpu());
    xassert(ringbuf.ndim == 1);
    xassert(ringbuf.is_fully_contiguous());

    xassert(ibatch >= 0);
    xassert(ibatch * params.beams_per_batch < params.total_beams);
    xassert(ichunk >= 0);
    
    long nbits_per_segment = params.nelts_per_segment * ringbuf.dtype.nbits;
    xassert_eq(nbits_per_segment, 1024);  // currently assuemd in gpu kernel
    xassert(this->is_allocated);

    if (noctuples == 0)
        return;
    
    int W = 4;
    int B = (noctuples + 4*W - 1) / (4*W);  // each block does 4*W octuples
    ulong iframe = (ichunk * params.total_beams) + (ibatch * params.beams_per_batch);
    
    gpu_copy_kernel<<< B, 32*W, 0, stream >>> (
        reinterpret_cast<uint4 *> (ringbuf.data),   // uint4 *ringbuf
        gpu_octuples.data,                         // const uint *octuples
        noctuples,                                 // long noctuples
        params.beams_per_batch,                     // int nbeams
        iframe                                      // ulong iframe
    );

    CUDA_PEEK("gpu_copy_kernel");
}


// -------------------------------------------------------------------------------------------------
//
// GpuRingbufCopyKernel::test()
// FIXME could use a little cleanup
//
// Test helpers are in anonymous namespace to avoid cluttering header.


namespace {

struct TestRingbuf
{
    long num_frames = 0;           // number of frames in periodic buffer ("frame" = time chunk + beam pair)
    long segments_per_frame = 0;   // number of 128-byte "segments" per frame
    long base_segment = -1;        // offset (in segments) relative to base memory address on either GPU or CPU
};


struct TestLocationPair
{
    uint src_seg_offset;
    uint src_frame_phase;
    uint src_num_frames;           // same as TestRingbuf::num_frames
    uint src_segments_per_frame;   // same as TestRingbuf::segments_per_frame
    
    uint dst_seg_offset;
    uint dst_frame_phase;
    uint dst_num_frames;           // same as TestRingbuf::num_frames
    uint dst_segments_per_frame;   // same as TestRingbuf::segments_per_frame
};
    

static vector<TestRingbuf> make_ringbufs(int nbuf, long beams_per_batch)
{
    xassert_ge(nbuf, 0);
    xassert_ge(beams_per_batch, 0);
    vector<TestRingbuf> ret(nbuf);

    for (int i = 0; i < nbuf; i++) {
        do {
            auto v = ksgpu::random_integers_with_bounded_product(2,10000);
            ret[i].num_frames = v[0];
            ret[i].segments_per_frame = v[1];
        } while (ret[i].num_frames < beams_per_batch);
    }

    return ret;
}


static Array<uint> make_octuple_array(const vector<TestLocationPair> &v, bool permute)
{
    if (permute) {
        vector<TestLocationPair> w = v;        // copy
        randomly_permute(w);
        return make_octuple_array(w, false);  // permute=false
    } 
        
    long n = v.size();
    Array<uint> ret({2*n,4}, af_rhost);

    // I got paranoid and decided not to memcpy() here.
    for (long i = 0; i < n; i++) {
        ret.data[8*i] = v[i].src_seg_offset;
        ret.data[8*i+1] = v[i].src_frame_phase;
        ret.data[8*i+2] = v[i].src_num_frames;
        ret.data[8*i+3] = v[i].src_segments_per_frame;
        ret.data[8*i+4] = v[i].dst_seg_offset;
        ret.data[8*i+5] = v[i].dst_frame_phase;
        ret.data[8*i+6] = v[i].dst_num_frames;
        ret.data[8*i+7] = v[i].dst_segments_per_frame;
    }

    return ret;
}

}  // anonymous namespace


void GpuRingbufCopyKernel::test()
{
    cout << "GpuRingbufCopyKernel::test()" << endl;

    long nbatches = rand_int(1,5);
    long beams_per_batch = rand_int(1,5);
    long ibatch = rand_int(0, nbatches);
    long ichunk = rand_int(0, 1000);
    long nbits = (rand_uniform() < 0.5) ? 16 : 32;       // this test will use either uint16 or uint32
    int nbuf_src = rand_int(1, 5);
    int nbuf_dst = rand_int(1, 5);
    
    long total_beams = nbatches * beams_per_batch;  // only used to convert (ichunk,ibatch) -> iframe
    int nelts_per_segment = 1024 / nbits;

    vector<TestRingbuf> src_ringbufs = make_ringbufs(nbuf_src, beams_per_batch);
    vector<TestRingbuf> dst_ringbufs = make_ringbufs(nbuf_dst, beams_per_batch);

    long nseg_tot = 0;
    vector<TestRingbuf *> all_ringbufs(nbuf_src + nbuf_dst);
    for (int i = 0; i < nbuf_src; i++)
        all_ringbufs[i] = &src_ringbufs[i];
    for (int i = 0; i < nbuf_dst; i++)
        all_ringbufs[nbuf_src + i] = &dst_ringbufs[i];

    randomly_permute(all_ringbufs);

    for (TestRingbuf *rb: all_ringbufs) {
        rb->base_segment = nseg_tot;
        nseg_tot += rb->num_frames * rb->segments_per_frame;
    }

    vector<TestLocationPair> lpairs;
    for (TestRingbuf &rb_dst: dst_ringbufs) {
        for (long sdst = 0; sdst < rb_dst.segments_per_frame; sdst++) {
            long nf_dst = rb_dst.num_frames;
            long fdst = rand_int(0, nf_dst);
            long fend = fdst + nf_dst - beams_per_batch;
            
            while (fdst <= fend) {
                if (rand_uniform() < 0.5) {
                    fdst++;
                    continue;
                }
                
                int isrc = rand_int(0, nbuf_src);
                TestRingbuf &rb_src = src_ringbufs[isrc];       
                TestLocationPair lpair;
                
                lpair.src_seg_offset = rb_src.base_segment + rand_int(0, rb_src.segments_per_frame);
                lpair.src_frame_phase = rand_int(0, 1000*1000);
                lpair.src_num_frames = rb_src.num_frames;
                lpair.src_segments_per_frame = rb_src.segments_per_frame;
                
                lpair.dst_seg_offset = rb_dst.base_segment + sdst;
                lpair.dst_frame_phase = fdst + rand_int(0,1000) * rb_dst.num_frames;
                lpair.dst_num_frames = rb_dst.num_frames;
                lpair.dst_segments_per_frame = rb_dst.segments_per_frame;
                
                lpairs.push_back(lpair);
                fdst += beams_per_batch;
            }
        }
    }

    RingbufCopyKernelParams hparams;
    hparams.total_beams = total_beams;
    hparams.beams_per_batch = beams_per_batch;
    hparams.nelts_per_segment = nelts_per_segment;
    hparams.octuples = make_octuple_array(lpairs, true);   // permute=true

    RingbufCopyKernelParams gparams;
    gparams.total_beams = total_beams;
    gparams.beams_per_batch = beams_per_batch;
    gparams.nelts_per_segment = nelts_per_segment;
    gparams.octuples = make_octuple_array(lpairs, true);   // permute=true

    Dtype dtype(df_uint, nbits);
    long nelts_tot = nseg_tot * nelts_per_segment;
    
    Array<void> hbuf(dtype, {nelts_tot}, af_rhost);
    
    // Randomize hbuf
    uint *p = reinterpret_cast<uint *> (hbuf.data);
    long nuint_tot = nseg_tot * (128 / sizeof(uint));
    for (long i = 0; i < nuint_tot; i++)
        p[i] = default_rng();
    
    // Copy to GPU before running copy kernel
    Array<void> gbuf = hbuf.to_gpu();

    CpuRingbufCopyKernel hkernel(hparams);
    hkernel.apply(hbuf, ichunk, ibatch);
    
    GpuRingbufCopyKernel gkernel(gparams);
    BumpAllocator allocator(af_gpu | af_zero, -1);  // dummy allocator
    gkernel.allocate(allocator);
    gkernel.launch(gbuf, ichunk, ibatch, nullptr);   // default stream
    
    assert_arrays_equal(hbuf, gbuf, "hbuf", "gbuf", {"i"});
}


}  // namespace pirate
