#include <iostream>
#include <immintrin.h>

#include <ksgpu/Array.hpp>
#include <ksgpu/CpuThreadPool.hpp>
#include <ksgpu/string_utils.hpp>
#include <ksgpu/time_utils.hpp>
#include <ksgpu/xassert.hpp>

#include "../../include/pirate/loose_ends/timing.hpp"
#include "../../include/pirate/loose_ends/cpu_downsample.hpp"

using namespace std;
using namespace ksgpu;

namespace pirate {
#if 0
}  // editor auto-indent
#endif



// -------------------------------------------------------------------------------------------------
//
// fake_5bit_downsampler(): some hacked-up code to answer the question, how fast would the 5-bit
// downsampler be, if memory access patterns were optimal?


inline void fake_5bit_store(__m256i *p, __m256i x)
{
    constexpr bool use_streaming_writes = true;

    if constexpr (use_streaming_writes)
        _mm256_stream_si256(p, x);
    else
        _mm256_store_si256(p, x);
}
    
    

static void fake_5bit_downsampler(const uint8_t *src, uint8_t *dst, long src_nbytes, long dst_nbytes)
{
    constexpr int src_bytes_per_chunk = 5*64;
    constexpr int dst_bytes_per_chunk = 3*64;

    // FIXME pointer alignment asserts
    xassert(src_nbytes >= 0);
    xassert(dst_nbytes >= 0);

    long nchunks = src_nbytes / src_bytes_per_chunk;
    xassert(src_nbytes == nchunks * src_bytes_per_chunk);
    xassert(dst_nbytes == nchunks * dst_bytes_per_chunk);

    const __m256i *sp = (const __m256i *) src;
    __m256i *dp = (__m256i *) dst;
    __m256i x = _mm256_setzero_si256();
    
    for (long i = 0; i < nchunks; i++) {
        x = _mm256_xor_si256(x, _mm256_load_si256(sp));
        x = _mm256_xor_si256(x, _mm256_load_si256(sp+1));
        x = _mm256_xor_si256(x, _mm256_load_si256(sp+2));
        x = _mm256_xor_si256(x, _mm256_load_si256(sp+3));

        fake_5bit_store(dp, x);
        fake_5bit_store(dp+1, x);

        x = _mm256_xor_si256(x, _mm256_load_si256(sp+4));
        x = _mm256_xor_si256(x, _mm256_load_si256(sp+5));
        x = _mm256_xor_si256(x, _mm256_load_si256(sp+6));
        
        fake_5bit_store(dp+2, x);
        fake_5bit_store(dp+3, x);

        x = _mm256_xor_si256(x, _mm256_load_si256(sp+7));
        x = _mm256_xor_si256(x, _mm256_load_si256(sp+8));
        x = _mm256_xor_si256(x, _mm256_load_si256(sp+9));
        
        fake_5bit_store(dp+4, x);
        fake_5bit_store(dp+5, x);

        sp += 10;
        dp += 6;
    }
}


// -------------------------------------------------------------------------------------------------


struct TimingSetup
{
    int src_bit_depth = 0;
    int nthreads = 0;
    
    long src_nbytes = 0;
    long dst_nbytes = 0;
    double gb_per_thread = 0.0;
    
    Array<uint8_t> src;  // shape (nthreads, src_nbytes)
    Array<uint8_t> dst;  // shape (nthreads, dst_nbytes)
    
    TimingSetup(int src_bit_depth_, int nthreads_)
    {
        this->src_bit_depth = src_bit_depth_;
        this->nthreads = nthreads_;
        
        cout << "time_cpu_downsample(src_bit_depth=" << src_bit_depth
             << ", nthreads=" << nthreads << "): start" << endl;

        long approximate_bytes_per_thread = 1L << 30;
        int src_bytes_per_chunk = cpu_downsample_src_bytes_per_chunk(src_bit_depth);
        int dst_bytes_per_chunk = (src_bytes_per_chunk * (src_bit_depth+1)) / (2*src_bit_depth);
        
        long nchunks = approximate_bytes_per_thread / (src_bytes_per_chunk + dst_bytes_per_chunk);
        nchunks = 2 * (nchunks / 2);  // force even
        
        this->src_nbytes = nchunks * src_bytes_per_chunk;
        this->dst_nbytes = nchunks * dst_bytes_per_chunk;
        this->gb_per_thread = (src_nbytes + dst_nbytes) / pow(2,30.);
        
        this->src = Array<uint8_t> ({nthreads,src_nbytes}, af_uhost | af_zero);
        this->dst = Array<uint8_t> ({nthreads,dst_nbytes}, af_uhost | af_zero);
    };

    void time_downsample()
    {
        uint8_t *sp = src.data;
        uint8_t *dp = dst.data;
        long ss = src_nbytes;
        long ds = dst_nbytes;
        
        auto callback = [&](const CpuThreadPool &pool, int ithread)
            {
                cpu_downsample(src_bit_depth, sp + ithread*ss, dp + ithread*ds, src_nbytes, dst_nbytes);
            };

        stringstream name;
        name << "cpu_downsample(src_bit_depth=" << src_bit_depth << ",nthreads=" << nthreads << ")";
        
        CpuThreadPool tp(callback, nthreads, 10, name.str());
        tp.monitor_throughput("GB/sec", gb_per_thread);
        tp.run();
    }
    
    void time_fake_downsample()
    {
        // Only this case is implemented.
        xassert(src_bit_depth == 5);
        
        uint8_t *sp = src.data;
        uint8_t *dp = dst.data;
        long ss = src_nbytes;
        long ds = dst_nbytes;
        
        auto callback = [&](const CpuThreadPool &pool, int ithread)
            {
                fake_5bit_downsampler(sp + ithread*ss, dp + ithread*ds, src_nbytes, dst_nbytes);
            };

        stringstream name;
        name << "fake_downsample(src_bit_depth=" << src_bit_depth << ",nthreads=" << nthreads << ")";
        
        CpuThreadPool tp(callback, nthreads, 10, name.str());
        tp.monitor_throughput("GB/sec", gb_per_thread);
        tp.run();
    }
};


void time_cpu_downsample(int nthreads)
{
    for (int src_bit_depth = 4; src_bit_depth <= 7; src_bit_depth++) {
        TimingSetup ts(src_bit_depth, nthreads);
        ts.time_downsample();

        if (src_bit_depth == 5)
            ts.time_fake_downsample();
    }
}


}  // namespace pirate
