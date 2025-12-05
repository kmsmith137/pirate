#include "../include/pirate/MegaRingbuf.hpp"
#include <ksgpu/xassert.hpp>

using namespace std;
using namespace ksgpu;

namespace pirate {
#if 0
}  // editor auto-indent
#endif


// Formerly:
// struct RingbufEntry
//{
//    Ringbuf *rb = nullptr;
//    long xlag = 0;   // lag of (time chunk, beam) pair (usually clag * total beams)
//    long iseg = 0;   // ring buffer segment index, within (time chunk, beam) pair.
//}

struct Triple 
{
    MegaRingbuf::Zone *zone = nullptr;
    long xlag = 0;   // lag of (time chunk, beam) pair (usually clag * total beams)
    long iseg = 0;   // ring buffer segment index, within (time chunk, beam) pair.
};

MegaRingbuf::MegaRingbuf(const Params &params_) :
    params(params_)
{
    xassert(params.active_beams > 0);
    xassert(params.active_beams <= params.total_beams);
    xassert(params.max_chunk_lag >= 0);
    xassert(params.max_gpu_chunk_lag >= 0);

    this->num_producers = params.producer_nviews.size();
    this->num_consumers = params.consumer_nviews.size();
    this->segments.resize(num_producers);

    for (long producer_id = 0; producer_id < num_producers; producer_id++) {
        long nviews = params.producer_nviews[producer_id];
        this->segments[producer_id].resize(nviews);
    }
}


void MegaRingbuf::add_segment(long producer_id, long producer_iview, long consumer_id, long consumer_iview, long chunk_lag)
{
    xassert(producer_id >= 0);
    xassert(producer_id < num_producers);
    xassert(producer_iview >= 0);
    xassert(producer_iview < params.producer_nviews[producer_id]);

    xassert(consumer_id >= 0);
    xassert(consumer_id < num_consumers);
    xassert(consumer_iview >= 0);
    xassert(consumer_iview < params.consumer_nviews[consumer_id]);
    
    xassert(chunk_lag >= 0);
    xassert(chunk_lag <= params.max_chunk_lag);

    Segment &s = this->segments[producer_id][producer_iview];
    xassert(s.num_consumers < max_consumers_per_producer);

    s.consumer_ids[s.num_consumers] = consumer_id;
    s.consumer_iviews[s.num_consumers] = consumer_iview;
    s.chunk_lags[s.num_consumers] = chunk_lag;
    s.num_consumers++;
}


void MegaRingbuf::finalize(bool delete_internals)
{
    if (is_finalized)
        throw runtime_error("double call to MegaRingbuf::finalize()");

    // Part 3:
    //  - allocate ringbufs
    //  - initialize Ringbuf::rb_len.
    
    const int BT = this->config.beams_per_gpu;            // total beams
    const int BB = this->config.beams_per_batch;          // beams per batch
    const int BA = this->config.num_active_batches * BB;  // active beams

    this->gpu_ringbufs.resize(max_clag + 1);
    this->host_ringbufs.resize(max_clag + 1);
    this->xfer_ringbufs.resize(max_clag + 1);
    
    for (int clag = 0; clag <= max_clag; clag++) {
        this->gpu_ringbufs.at(clag).num_frames = clag*BT + BA;
        this->host_ringbufs.at(clag).num_frames = clag*BT + BA;
        this->xfer_ringbufs.at(clag).num_frames = 2*BA;
    }

    this->et_host_zone.num_frames = BA;
    this->et_gpu_zone.num_frames = BA;

    // Part 4:
    //
    //   - Initialize "local" RingbufEntry vectors.
    //     These will be converted to integer-valued "_locs" arrays in Part 5.
    //
    //   - Initialize Ringbuf::nseg_per_beam.

    vector<vector<Triple>> 

    this->is_finalized = true;
}


}  // namespace pirate
