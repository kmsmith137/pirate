#include "../include/pirate/RingbufCopyKernel.hpp"

#include <ksgpu/Array.hpp>
#include <ksgpu/xassert.hpp>
#include <ksgpu/rand_utils.hpp>

using namespace std;
using namespace ksgpu;

namespace pirate {
#if 0
}  // editor auto-indent
#endif


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


static Array<uint> make_location_array(const vector<TestLocationPair> &v, bool permute)
{
    if (permute) {
        vector<TestLocationPair> w = v;        // copy
        randomly_permute(w);
        return make_location_array(w, false);  // permute=false
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

        
void test_gpu_ringbuf_copy_kernel()
{
    cout << "test_gpu_ringbuf_copy_kernel()" << endl;

    long nbatches = rand_int(1,5);
    long beams_per_batch = rand_int(1,5);
    long ibatch = rand_int(0, nbatches);
    long it_chunk = rand_int(0, 1000);
    long nbits = (rand_uniform() < 0.5) ? 16 : 32;       // this test will use either uint16 or uint32
    int nbuf_src = rand_int(1, 5);
    int nbuf_dst = rand_int(1, 5);
    
    long total_beams = nbatches * beams_per_batch;  // only used to convert (it_chunk, ibatch) -> iframe
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
    hparams.locations = make_location_array(lpairs, true);   // permute=true

    RingbufCopyKernelParams gparams;
    gparams.total_beams = total_beams;
    gparams.beams_per_batch = beams_per_batch;
    gparams.nelts_per_segment = nelts_per_segment;
    gparams.locations = make_location_array(lpairs, true);   // permute=true

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
    hkernel.apply(hbuf, ibatch, it_chunk);
    
    GpuRingbufCopyKernel gkernel(gparams);
    gkernel.allocate();
    gkernel.launch(gbuf, ibatch, it_chunk, nullptr);   // default stream
    
    assert_arrays_equal(hbuf, gbuf, "hbuf", "gbuf", {"i"});
}


}  // namespace pirate
