#include "../include/pirate/MegaRingbuf.hpp"
#include "../include/pirate/inlines.hpp"  // xmod()

#include <ksgpu/xassert.hpp>

using namespace std;
using namespace ksgpu;

namespace pirate {
#if 0
}  // editor auto-indent
#endif



MegaRingbuf::MegaRingbuf(const Params &params_) :
    params(params_)
{
    xassert(params.active_beams > 0);
    xassert(params.active_beams <= params.total_beams);

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
    this->max_clag = max(max_clag, chunk_lag);

    Segment &s = this->segments[producer_id][producer_iview];
    xassert(s.num_consumers < max_consumers_per_producer);

    long n = s.num_consumers++;
    s.consumer_ids[n] = consumer_id;
    s.consumer_iviews[n] = consumer_iview;
    s.chunk_lags[n] = chunk_lag;
}


void MegaRingbuf::finalize(bool delete_internals)
{
    if (is_finalized)
        throw runtime_error("double call to MegaRingbuf::finalize()");

    this->max_gpu_clag = long(max_clag * params.gpu_clag_maxfrac + 0.5);
    this->max_gpu_clag = min(max_gpu_clag, max_clag);
    this->max_gpu_clag = max(max_gpu_clag, 0L);
    
    // Part 1: partially initialize zones
    //
    //  - Number of zones is determined
    //  - Zone::num_frames is initialized
    //  - Zone::segments_per_frame is not initialized
    //  - Zone::giant_segment_offset is not initialized
    
    const int BT = params.total_beams;
    const int BA = params.active_beams;

    this->gpu_zones.resize(max_clag + 1);
    this->host_zones.resize(max_clag + 1);
    this->xfer_zones.resize(max_clag + 1);
    
    for (int clag = 0; clag <= max_clag; clag++) {
        this->gpu_zones.at(clag).num_frames = clag*BT + BA;
        this->host_zones.at(clag).num_frames = clag*BT + BA;
        this->xfer_zones.at(clag).num_frames = 2*BA;
    }

    this->et_host_zone.num_frames = BA;
    this->et_gpu_zone.num_frames = BA;

    // Part 2: 
    //
    //   - Initialize all "triples" (producer_triples, consumer_triples,
    //     g2g_triples, h2h_triples).
    //
    //   - Initialize Zone::segments_per_beam (but not Zone::giant_segment_offset).

    this->producer_triples.resize(num_producers);
    this->consumer_triples.resize(num_consumers);

    for (long producer_id = 0; producer_id < num_producers; producer_id++) {
        long producer_nviews = params.producer_nviews.at(producer_id);
        this->producer_triples.at(producer_id).resize(producer_nviews);
    }

    for (long consumer_id = 0; consumer_id < num_consumers; consumer_id++) {
        long consumer_nviews = params.consumer_nviews.at(consumer_id);
        this->consumer_triples.at(consumer_id).resize(consumer_nviews);
    }

    // Outer loop over segments (with associated producer info).
    for (long producer_id = 0; producer_id < num_producers; producer_id++) {
        long producer_nviews = params.producer_nviews[producer_id];
        for (long producer_iview = 0; producer_iview < producer_nviews; producer_iview++) {
            Segment &s = this->segments[producer_id][producer_iview];

            // FIXME hmm when writing code in this loop, I had to be careful to
            // distinguish between this->num_consumers and s.num_consumers. I should
            // probably rename things, to avoid a potential source of bugs.

            // Check that every (producer_id, producer_iview) pair has received at
            // least one call to add_segment().
            xassert(s.num_consumers > 0);

            // Check that s.chunk_lags is sorted. (This is true in our current use cases,
            // but a fully general MagaRingbuf implementation wouldn't need to assume it.
            // In the future I might relax this assumption, e.g. by putting a sort here.)

            for (long i = 1; i < s.num_consumers; i++)
                xassert(s.chunk_lags[i-1] <= s.chunk_lags[i]);

            // Split num_consumers = ngpu + ncpu.
            // In the general case (ngpu >= 1) and (ncpu >= 2), we have the following steps:
            //
            //  1. goes to a gpu_zone for the first (ngpu) consumers
            //  2. gets copied (GPU->GPU) from a gpu_zone to an xfer_zone (using g2g_triple_pairs)
            //  3. gets copied (GPU->CPU) from an xfer_zone to a host_zone
            //  4. for (ncpu-1) consumers
            //      - gets copied (CPU->CPU) from a host_zone to the et_host_zone
            //      - gets copied (CPU->GPU) from the et_host_zone to the et_gpu_zone
            //  5. for the last consumer
            //      - gets copied (GPU->CPU) from a host_zone to an xfer_zone
            //
            // If (ngpu == 0), then copy directly to xfer_zone (i.e. skip step 1).
            // If (ncpu == 0), then skip steps 2-5.
            // If (ncpu <= 1), then skip step 4.

            long ngpu = 0;
            while (ngpu < s.num_consumers) {
                long prev_clag = (ngpu > 0) ? s.chunk_lags[ngpu-1] : 0;
                if (s.chunk_lags[ngpu] > prev_clag + max_gpu_clag)
                    break;
                ngpu++;
            }

            long ncpu = s.num_consumers - ngpu;
            long gpu_clag = (ngpu > 0) ? s.chunk_lags[ngpu-1] : 0;
            long host_clag = (ncpu > 0) ? (s.chunk_lags[s.num_consumers-1] - gpu_clag) : 0;
            xassert((ncpu == 0) || (host_clag > max_gpu_clag));

            Zone *gpu_zone = (ngpu > 0) ? &gpu_zones.at(gpu_clag) : nullptr;
            Zone *host_zone = (ncpu > 0) ? &host_zones.at(host_clag) : nullptr;
            Zone *xfer_zone = (ncpu > 0) ? &xfer_zones.at(host_clag) : nullptr;

            long gpu_iseg = (ngpu > 0) ? (gpu_zone->segments_per_frame++) : (-1);
            long host_iseg = (ncpu > 0) ? (host_zone->segments_per_frame++) : (-1);
            long et_iseg = (ncpu > 1) ? (et_host_zone.segments_per_frame++) : (-1);

            // Keep xfer_zone->segments_per_frame in sync with host_zone->segments_per_frame.
            if (ncpu > 0)
                xfer_zone->segments_per_frame++;

            // Keep et_gpu_zone.segments_per_frame in sync with et_host_zone.segments_per_frame.
            if (ncpu > 1)
                et_gpu_zone.segments_per_frame++;
            
            // Now we're ready to implement steps 1 and 2 above.

            Triple &producer = producer_triples.at(producer_id).at(producer_iview);
            xassert(producer.zone == nullptr);   // paranoid

            if (ncpu == 0)
                // producer -> gpu_zone
                _set_triple(producer, gpu_zone, 0, gpu_iseg);
            else if (ngpu == 0)
                // producer -> xfer_zone
                _set_triple(producer, xfer_zone, 0, host_iseg);
            else {
                // producer -> gpu_zone -> xfer zone 
                _set_triple(producer, gpu_zone, 0, gpu_iseg);
                _push_triple(g2g_triples, gpu_zone, gpu_clag * BT, gpu_iseg);  // g2g src
                _push_triple(g2g_triples, xfer_zone, 0, host_iseg);            // g2g dst
            }

            // Inner loop over consumers. This implements steps 4 and 5 above.

            for (long c = 0; c < s.num_consumers; c++) {
                long consumer_id = s.consumer_ids[c];
                long consumer_iview = s.consumer_iviews[c];
                long chunk_lag = s.chunk_lags[c];

                // Check that each (consumer_id, consumer_iview) pair has received at
                // most one call to add_segment().
                Triple &consumer = consumer_triples.at(consumer_id).at(consumer_iview);
                xassert(consumer.zone == nullptr);

                if (c < ngpu)
                    // gpu_zone -> consumer
                    _set_triple(consumer, gpu_zone, chunk_lag * BT, gpu_iseg);
                else if (c == s.num_consumers-1)
                    // xfer_zone -> last consumer
                    _set_triple(consumer, xfer_zone, BA, host_iseg);
                else {
                    // host_zone -> et_host_zone
                    // et_gpu_zone -> consumer
                    _push_triple(h2h_triples, host_zone, (chunk_lag-gpu_clag) * BT, host_iseg);  // h2h src
                    _push_triple(h2h_triples, &et_host_zone, 0, et_iseg);                        // h2h dst
                    _set_triple(consumer, &et_gpu_zone, 0, et_iseg);
                }
            }
        }
    }

    // Check that each (consumer_id, consumer_iview) pair has received at
    // least one call to add_segment().

    for (long consumer_id = 0; consumer_id < num_consumers; consumer_id++) {
        long consumer_nviews = params.consumer_nviews.at(consumer_id);
        for (long consumer_iview = 0; consumer_iview < consumer_nviews; consumer_iview++) {
            Triple &consumer = consumer_triples.at(consumer_id).at(consumer_iview);
            xassert(consumer.zone != nullptr);
        }
    }

    // Part 3: Lay out zones consecutively in memory (boolean argument is "on_gpu").

    _lay_out_zones(gpu_zones, true);
    _lay_out_zones(host_zones, false);
    _lay_out_zones(xfer_zones, true);
    _lay_out_zone(et_host_zone, false);
    _lay_out_zone(et_gpu_zone, true);

    // Part 4: Initialize quadruples/octuples from triples.

    this->producer_quadruples = _triples_to_quadruples(producer_triples);
    this->consumer_quadruples = _triples_to_quadruples(consumer_triples);
    this->g2g_octuples = _triples_to_quadruples(g2g_triples);
    this->h2h_octuples = _triples_to_quadruples(h2h_triples);

    // Part 5: clean up.

    if (delete_internals) {
        this->segments.clear();
        this->producer_triples.clear();
        this->consumer_triples.clear();
        this->g2g_triples.clear();
        this->h2h_triples.clear();
        // Zones can stay, since they don't take up much memory.
    }

    this->is_finalized = true;
}


void MegaRingbuf::allocate(Dtype dtype, int nelts_per_segment, bool hugepages)
{
    if (!is_finalized)
        throw runtime_error("MegaRingbuf::allocate() called before finalize()");
    if (is_allocated)
        throw runtime_error("double call to MegaRingbuf::allocate()");

    xassert(nelts_per_segment > 0);
    uint host_aflags = af_rhost | af_zero | (hugepages ? af_mmap_huge : 0);

    this->host_giant_buffer = Array<void> (dtype, {host_giant_nseg, nelts_per_segment}, host_aflags);
    this->gpu_giant_buffer = Array<void> (dtype, {gpu_giant_nseg, nelts_per_segment}, af_gpu | af_zero);
    this->is_allocated = true;
}


void MegaRingbuf::_set_triple(Triple &t, Zone *zp, long frame_lag, long segment_within_frame)
{
    xassert(zp != nullptr);
    xassert((frame_lag >= 0) && (frame_lag < zp->num_frames));
    xassert(segment_within_frame == zp->segments_per_frame-1);  // hmmm
    xassert(t.zone == nullptr);

    t.zone = zp;
    t.frame_lag = frame_lag;
    t.segment_within_frame = segment_within_frame;
}

void MegaRingbuf::_push_triple(std::vector<Triple> &triples, Zone *zp, long frame_lag, long segment_within_frame)
{
    Triple t;
    _set_triple(t, zp, frame_lag, segment_within_frame);
    triples.push_back(t);
}

void MegaRingbuf::_lay_out_zones(std::vector<Zone> &zones, bool on_gpu)
{
    long n = zones.size();
    for (long i = 0; i < n; i++)
        _lay_out_zone(zones.at(i), on_gpu);
}

void MegaRingbuf::_lay_out_zone(Zone &z, bool on_gpu)
{
    xassert(z.giant_segment_offset < 0);

    if (on_gpu) {
        z.giant_segment_offset = gpu_giant_nseg;
        gpu_giant_nseg += z.num_frames * z.segments_per_frame;
    } else {
        z.giant_segment_offset = host_giant_nseg;
        host_giant_nseg += z.num_frames * z.segments_per_frame;
    }
}

// Check for 32-bit overflows when converting long -> uint.
inline uint to_uint(long n)
{
    xassert(n >= 0);
    xassert(n < (1L << 32));
    return uint(n);
}

ksgpu::Array<uint> MegaRingbuf::_triples_to_quadruples(const std::vector<Triple> &triples)
{
    long n = triples.size();
    ksgpu::Array<uint> ret({n,4}, af_rhost);

    for (long i = 0; i < n; i++) {
        const Triple &t = triples.at(i);

        xassert(t.zone != nullptr);
        xassert(t.zone->giant_segment_offset >= 0);
        xassert(t.frame_lag < t.zone->num_frames);
        xassert(t.segment_within_frame < t.zone->segments_per_frame);

        uint giant_segment_offset = to_uint(t.zone->giant_segment_offset + t.segment_within_frame);
        uint frame_offset_within_zone = to_uint(xmod(t.zone->num_frames - t.frame_lag, t.zone->num_frames));
        uint frames_in_zone = to_uint(t.zone->num_frames);
        uint segments_per_frame = to_uint(t.zone->segments_per_frame);

        ret.data[4*i] = giant_segment_offset;
        ret.data[4*i+1] = frame_offset_within_zone;
        ret.data[4*i+2] = frames_in_zone;
        ret.data[4*i+3] = segments_per_frame;
    }

    return ret;
}

std::vector<ksgpu::Array<uint>> MegaRingbuf::_triples_to_quadruples(const std::vector<std::vector<Triple>> &triples)
{
    long n = triples.size();
    std::vector<ksgpu::Array<uint>> ret(n);

    for (long i = 0; i < n; i++)
        ret[i] = _triples_to_quadruples(triples.at(i));

    return ret;
}

}  // namespace pirate
