#ifndef _PIRATE_MEGA_RINGBUF_HPP
#define _PIRATE_MEGA_RINGBUF_HPP

#include "constants.hpp"

#include <vector>
#include <ksgpu/Array.hpp>

namespace pirate {
#if 0
}  // editor auto-indent
#endif


// MegaRingbuf: This data structure is the "nerve center" of the real-time FRB
// search. It stores a collection of "segments", where each segment is a 128 bytes
// (the size of a GPU cache line), that are produced by a GPU "producer" kernel,
// and consumed later (after some time lag) by one or more "consumer" kernels.
//
// Each segment has multiple lags (since there can be multiple consumers per
// producer), the lags are per-segment (but must be the same for all beams),
// and the buffered data is opaque to the MegaRingbuf.
//
// The total size of the MegaRingbuf can be much larger than GPU memory.
// In this case, the MegaRingbuf will strategize "shuffling" data between
// GPU and host memory. The main consideration here is minimizing PCIe
// bandwidth, by keeping low-lag segments on the GPU, and copying high-lag
// segments from CPU->GPU->CPU on the fly. It's also crucial to ensure
// that CPU<->GPU copies consist of large contiguous blocks (by using
// staging buffers and inserting CPU<->CPU and GPU<->GPU shuffling
// operations as needed). Finally, it's important to run all three types
// of copying operations (CPU<->GPU, CPU<->GPU, GPU<->GPU) asynchronously,
// in order to overlap transfer and compute.
// 
// The purpose of the MegaRingbuf is to hide all of this complexity from the
// GPU compute kernels, so that the compute kernels "see" GPU memory as a large
// ring buffer of independent segments (cache lines), with the large capacity
// of host memory, and (hopefully) the high bandwidth of GPU memory.
//  
// Notes:
//
//   - One "time chunk", or sometimes "chunk" for short, is the cadence for
//     processing. In a real-time FRB search, this might be around 1 or 2 seconds.
//
//   - One ring buffer "frame" is a (time chunk, beam) pair, where the time index
//     is slowly varying (i.e. iframe = ichunk * total_beams + ibatch).
//
//     Note that there is no notion of beam "batches" in the ring buffer code --
//     each frame is one beam. (Other parts of the dedispersion code define beam
//     "batches".)
//
//     Ring buffer lags (or indices) can appear in the code either as chunk counts
//     or frame counts. To distinguish, we use variable names like "chunk_lag" or 
//     "frame_lag" (the two would be related by flag = clag * num_batches).
//
//   - The ring buffer is divided into "zones". Each zone has a different length, and
//     can either be on the CPU or GPU. We sometimes have short zones (a few frames)
//     which are "staging areas" for copies (CPU<->GPU, CPU<->CPU, or GPU<->GPU)
//
//   - The ring buffer has multiple producers, indexed by 0 <= producer_id < num_producers.
//     (And similarly for consumers.) 
// 
//     Currently, we use the following scheme:
//
//       - stage 1 dedispersion kernel (0 <= i < num_stage1_trees)
//         has producer_id == i
//
//       - stage 2 dedispersion (or cdd2) kernel (0 <= j < num_stage2_trees)
//         has consumer_id == j
// 
//     In the future, when kernels are more coalesced, we'll use a scheme like this:
//
//       - cdd1 kernel has producer_id == 0
//
//       - stage1 downsampled dedispersion kernel (1 <= i < num_stage1_trees)
//         has producer_id == i, and consumer_id == (i-1)
//
//       - stage2 dedispersion (or ccd2) kernel (0 <= j < num_stage2 trees)
//         has consumer_id == (j + stage1_trees - 1).
// 
//   - "Segments" and "views". Each consumer reads "segments" from the ring buffer, 
//     where a segment is always 128 bytes (constants::bytes_per_gpu_cache_line). 
//
//     The number of ** segments per beam ** that each consumer reads is denoted
//
//         long consumer_nviews[num_consumers];
//
//     where a "view" is a logical triple (zone, frame_lag, segment_within_frame).
//     (Note however that views are usually represented differently as quadruples,
//     see below).
//
//     Each consumer (specified by 0 <= consumer_id < num_consumers) indexes its
//     views by an index:
//
//         0 <= iview < consumer_nviews[consumer_id]
//
//     The consumer can choose whatever ordering is convenient (i.e. the index 'iview'
//     is opaque to the MegaRingbuf). For example, the dedispersion kernels use a
//     different iview-ordering scheme for their input/output buffers.
//
//     Producers are similar to consumers, except that the logical frame_lag is zero.
//
//   - "Quadruples". In GPU kernels, ring buffer views (either producer or consumer)
//      are represented as quadruples:
//
//        uint giant_segment_offset;       // segment count
//        uint frame_offset_within_zone;   // frame count
//        uint frames_in_zone;             // frame count
//        uint segments_per_frame;         // segment count
//
//     To define these precisely, here is some code showing how a compute kernel
//     computes the memory address of a segment in the buffer.
//
//        // Setup
//        uint quadruple[4];        // see above
//        char *giant_base = ...;   // "giant" buffer (i.e. all zones) on either CPU or GPU
//        long frame0 = ...;        // context-dependent frame index
//        long b = ...;             // context-dependent beam index (just gets added to frame0)
//
//        uint giant_segment_offset = quadruple[0];
//        uint frame_offset_within_zone = quadruple[1];
//        uint frames_in_zone = quadruple[2];
//        uint segments_per_frame = quadruple[3];
//
//         uint frame = (frame0 + frame_offset_within_zone + b) % frames_in_zone;
//         long seg = giant_segment_offset + (frame * segments_per_frame);
//         char *p = giant_base + (128 * seg);   // 128 bytes per segmnet
//
//      The point of the quadruple is that it allows segment addresses to be computed
//      on a single GPU thread, which stores only the uint quadruple[4].
//
//   - "Octuples". CPU->CPU and GPU->GPU copies are represented by a list of "octuples".
//      Each octuple is a (src,dst) pair of quadruples, with the format above.
//
// Future features:
//
//   - Constructor arguments to select between different strategies. (For example,
//     is it better to do GPU->CPU copies as one big coalesced transfer, or a
//     small transfer for each host_lag? (We currently do the latter). The answer
//     depends on whether PCIe bandwidth or host memory bandwidth is a larger
//     bottleneck.
//
//   - Some form of beam "batching" (this batch size can be independent of the
//     batch size used for compute!).
//
//   - Update-in-place (memory optimization).


struct MegaRingbuf {
    static constexpr int max_consumers_per_producer = constants::max_downsampling_level + 1;

    // Parameters specified at construction.
    struct Params {
        long total_beams = 0;
        long active_beams = 0;

        // Number of segments (or "views", see above) for each producer/consumer.
        std::vector<long> producer_nviews;  // Length (num_producers)
        std::vector<long> consumer_nviews;  // Length (num_consumers)

        // For testing: limit on-gpu ring buffers to (clag) <= (gpu_clag_maxfrac) * (max_clag)
        double gpu_clag_maxfrac = 1.0;   // set to 1 to disable (default)
    };

    
    MegaRingbuf(const Params &params);

    // After constructing the MegaRingbuf, the caller calls add_segment() many
    // times, then finalize(). Other member functions generally need to enforce
    // that finalize() has been called (e.g. get_quadruples(), launch_gpu_copy_kernel()).

    void add_segment(long producer_id, long producer_iview, long consumer_id, long consumer_iview, long chunk_lag);
    void finalize(bool delete_internals=true);
    void allocate(bool hugepages);  // allocate memory on both CPU and GPU

    long num_producers = 0;
    long num_consumers = 0;

    // These arrays are created in finalize().
    // (producer_id) -> (array of shape (producer_num_views[producer_id],4))
    // (consumer_id) -> (array of shape (consumer_num_views[consumer_id],4))
    std::vector<ksgpu::Array<uint>> producer quadruples;
    std::vector<ksgpu::Array<uint>> consumer_quadruples;

    // "Giant" buffers are created in allocate(), and contain all zones.
    ksgpu::Array<void> cpu_giant_buffer;
    ksgpu::Array<void> gpu_giant_buffer;

    // ------------------------------------------------------------------------
    //
    // Internals.

    // Segments just "record" calls to add_segment(), until finalize() is called.
    struct Segment
    {
        long num_consumers = 0;
        long consumer_iviews[max_consumers_per_producer];
        long chunk_lags[max_consumers_per_producer];
    };

    // (producer_id) -> (producer_iview) -> (consumer ids, iviews)
    std::vector<std::vector<Segment>> segments;
    int max_clag = 0;


    // Zones are created in finalize().
    struct Zone
    {
        long num_frames = 0;
        long segments_per_frame = 0;
        long giant_segment_offset = 0;
    };

    // All vector<Zone> objects have length (max_clag + 1).
    // BT = total beams, BA = active beams, BB = beams per batch.
    
    std::vector<Zone> gpu_zones;    // rb_len = (clag*BT + BA), on GPU
    std::vector<Zone> host_zones;   // rb_len = (clag*BT + BA), on host
    std::vector<Zone> xfer_zones;   // rb_len = (2*BA), on GPU

    Zone et_host_zone;  // rb_len = BA, on host (send buffer)
    Zone et_gpu_zone;   // rb_len = BA, on GPU (recv buffer)
};


#endif // _PIRATE_MEGA_RINGBUF_HPP