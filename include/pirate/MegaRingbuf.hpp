#ifndef _PIRATE_MEGA_RINGBUF_HPP
#define _PIRATE_MEGA_RINGBUF_HPP

#include "constants.hpp"

#include <vector>
#include <ksgpu/Array.hpp>

// Forward declaration for YAML
namespace YAML { class Emitter; }

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
//       - stage 1 dedispersion kernel (0 <= ids < num_downsampling_levels)
//         has producer_id == ids
//
//       - stage 2 dedispersion (or cdd2) kernel (0 <= itree < num_trees)
//         has consumer_id == itree
// 
//     In the future, when kernels are more coalesced, we might use a scheme like this:
//
//       - cdd1 kernel has producer_id == 0
//
//       - stage1 downsampled dedispersion kernel (1 <= ids < num_downsampling_levels
//         has producer_id == ids, and consumer_id == (ids-1)
//
//       - stage2 dedispersion (or ccd2) kernel (0 <= itree < num_trees)
//         has consumer_id == (itree + num_downsampling_levels - 1).
// 
//   - "Segments" and "quadruples". Each consumer reads "segments" from the ring buffer, 
//     where a segment is always 128 bytes (constants::bytes_per_gpu_cache_line). 
//
//     The number of ** segments per beam ** that each consumer reads is denoted
//
//         long consumer_nquads[num_consumers];
//
//     where a "quadruple" (or "quad") is a logical triple (zone, frame_lag, segment_within_frame),
//     represented as four uint32 values (see below).
//
//     Each consumer (specified by 0 <= consumer_id < num_consumers) indexes its
//     quadruples by an index:
//
//         0 <= iquad < consumer_nquads[consumer_id]
//
//     The consumer can choose whatever ordering is convenient (i.e. the index 'iquad'
//     is opaque to the MegaRingbuf). For example, the dedispersion kernels use a
//     different iquad-ordering scheme for their input/output buffers.
//
//     Producers are similar to consumers, except that the logical frame_lag is zero.
//
//   - "Quadruples". In GPU kernels, ring buffer quadruples (either producer or consumer)
//      are represented as:
//
//        uint global_segment_offset;      // segment count
//        uint frame_offset_within_zone;   // frame count
//        uint frames_in_zone;             // frame count
//        uint segments_per_frame;         // segment count
//
//     To define these precisely, here is some code showing how a compute kernel
//     computes the memory address of a segment in the buffer.
//
//        // Setup
//        uint quadruple[4];         // see above
//        char *global_base = ...;   // "global" buffer (i.e. all zones) on either CPU or GPU
//        long frame0 = ...;         // context-dependent frame index
//        long b = ...;              // context-dependent beam index (just gets added to frame0)
//
//        uint global_segment_offset = quadruple[0];
//        uint frame_offset_within_zone = quadruple[1];
//        uint frames_in_zone = quadruple[2];
//        uint segments_per_frame = quadruple[3];
//
//        uint frame = (frame0 + frame_offset_within_zone + b) % frames_in_zone;
//        long seg = global_segment_offset + (frame * segments_per_frame);
//        char *p = global_base + (128 * seg);   // 128 bytes per segmnet
//
//      The point of the quadruple is that it allows segment addresses to be computed
//      on a single GPU thread, which stores only the uint quadruple[4].
//
//   - "Octuples". CPU->CPU and GPU->GPU copies are represented by a list of "octuples".
//      Each octuple is a (src,dst) pair of quadruples, with the format above.
//
// Future features:
//
//   - Constructor arguments to select between different strategies, which trade off
//     resources (GPU memory bandwidth, host memory bandwidth, PICe bandwidth).
//
//   - Constructor args to limit GPU memory usage.
//
//   - Some form of beam "batching" (this batch size can be independent of the
//     batch size used for compute!).
//
//   - Memory optimization: if a kernel is both a consumer and a producer, then there
//     should be a way for it to update segments of the ring buffer "in-place".
//
//   - Standalone (i.e. not dedispersion-specific) end-to-end unit test of the 
//     MegaRingbuf.
//
//   - Better ability to inspect the MegaRingbuf -- for example, how well is
//     it coalescing GPU<->CPU copies?


struct MegaRingbuf {
    static constexpr int max_consumers_per_producer = constants::max_downsampling_level + 1;

    // Parameters specified at construction.
    struct Params {
        long total_beams = 0;
        long active_beams = 0;

        // Number of quadruples (see above) for each producer/consumer.
        std::vector<long> producer_nquads;  // Length (num_producers)
        std::vector<long> consumer_nquads;  // Length (num_consumers)

        // For testing: limit on-gpu ring buffers to (clag) <= max_gpu_clag
        long max_gpu_clag = 10000;   // set to 10000 to disable (default)
    };

    MegaRingbuf(const Params &params);

    // After constructing the MegaRingbuf, the caller calls add_segment() many
    // times, then finalize(). Other member functions generally need to enforce
    // that finalize() has been called (e.g. get_quadruples(), launch_gpu_copy_kernel()).

    void add_segment(long producer_id, long producer_iquad, long consumer_id, long consumer_iquad, long chunk_lag);
    void finalize(bool delete_internals=true);

    // Initialized in constructor.
    Params params;
    long num_producers = 0;
    long num_consumers = 0;
    bool is_finalized = false;

    // Quadruples/octuples (see above).
    //
    // These arrays are created in finalize(), and are always on the host.
    // The MegaRingbuf does not contain code to copy them to the GPU.
    // Instead, they get copied to the GPU in the individual compute kernels
    // (for example, GpuDedispersionKernel::allocate()).
    //
    // (producer_id) -> (array of shape (producer_nquads[producer_id], 4))
    // (consumer_id) -> (array of shape (consumer_nquads[consumer_id], 4))
    
    std::vector<ksgpu::Array<uint>> producer_quadruples;
    std::vector<ksgpu::Array<uint>> consumer_quadruples;
    ksgpu::Array<uint> g2g_octuples;   // shape (segments_to_copy, 8)
    ksgpu::Array<uint> h2h_octuples;   // shape (segments_to_copy, 8)

    // Size of "global" ring buffers on CPU and GPU, in "segments" not bytes.
    // (Usually, a segment is 128 bytes, but there are some testing contexts
    // where this isn't true, and it's more accurate to use segments here.)
    
    long host_global_nseg = 0;
    long gpu_global_nseg = 0;

    // Make a random "simplified" (pure-gpu) MegaRingbuf, intended for standalone
    // testing of dedispersion kernels. See MegaRingbuf.cu for more info.
    static std::shared_ptr<MegaRingbuf> make_random_simplified(long total_beams, long active_beams, long nchunks, long nquads);

    // Create the simplest possible MegaRingbuf, for standalone timing of dedispersion kernels.
    // FIXME I might implement something more realistic later.
    static std::shared_ptr<MegaRingbuf> make_trivial(long total_beams, long nquads);

    // Serialize this MegaRingbuf to YAML format
    void to_yaml(YAML::Emitter &emitter, double frames_per_second, long nfreq, long time_samples_per_chunk, bool verbose=false) const;

    // ------------------------------------------------------------------------
    //
    // Internals.

    // Segments just "record" calls to add_segment(), until finalize() is called.
    struct Segment
    {
        long num_consumers = 0;
        long consumer_ids[max_consumers_per_producer];
        long consumer_iquads[max_consumers_per_producer];
        long chunk_lags[max_consumers_per_producer];
    };

    // Segments are updated in add_segment(), and used in finalize().
    // segments[producer_id][producer_iquad] = (Segment containing consumer ids/iquads/lags)
    std::vector<std::vector<Segment>> segments;

    // max_clag: largest lag (in chunks) of any zone
    // max_gpu_clag: largest lag of any _nonempty_ gpu_zone
    // min_host_clag: smallest lag of any _nonempty_ host zone
    // min_et_clag: smallest lag between (segment arrives on host) and (et_h2h ingests segment)
    // min_et_headroom: smallest lag between (et_h2h ingests segment) and (segment leaves host)

    long max_clag = 0;         // updated in add_segment()
    long max_gpu_clag = 0;     // initialized in finalize()
    long min_host_clag = 0;    // initialized in finalize()
    long min_et_clag = 0;      // initialized in finalize()
    long min_et_headroom = 0;  // initialized in finalize()  

    // Zones are created in finalize().
    struct Zone
    {
        long num_frames = 0;
        long segments_per_frame = 0;
        long global_segment_offset = -1;

        long segment_offset_of_frame(long iframe) const;
    };

    // All vector<Zone> objects have length (max_clag + 1).
    // BT = total beams, BA = active beams, BB = beams per batch.
    
    std::vector<Zone> gpu_zones;    // frames_in_zone = (clag*BT + BA), on GPU
    std::vector<Zone> host_zones;   // frames_in_zone = (clag*BT + BA), on host
    std::vector<Zone> g2h_zones;    // frames_in_zone = (BA), on GPU
    std::vector<Zone> h2g_zones;    // frames_in_zone = (BA), on GPU

    Zone et_host_zone;  // frames_in_zone = BA, on host (send buffer)
    Zone et_gpu_zone;   // frames_in_zone = BA, on GPU (recv buffer)

    // Triples are used temporarily in finalize().
    struct Triple
    {
        Zone *zone = nullptr;
        long frame_lag = 0;
        long segment_within_frame = 0;
    };


    std::vector< std::vector<Triple> > producer_triples;
    std::vector< std::vector<Triple> > consumer_triples;
    std::vector<Triple> g2g_triples;   // (src,dst) pairs
    std::vector<Triple> h2h_triples;   // (src,dst) pairs

    // Helpers for _finalize().
    void _set_triple(Triple &t, Zone *zp, long frame_lag);
    void _push_triple(std::vector<Triple> &triples, Zone *zp, long frame_lag);
    void _lay_out_zones(std::vector<Zone> &zones, bool on_gpu);
    void _lay_out_zone(Zone &z, bool on_gpu);
    void _delete_internals();

    ksgpu::Array<uint> _triples_to_quadruples(const std::vector<Triple> &triples);
    std::vector<ksgpu::Array<uint>> _triples_to_quadruples(const std::vector<std::vector<Triple>> &triples);
};


} // namespace pirate

#endif // _PIRATE_MEGA_RINGBUF_HPP