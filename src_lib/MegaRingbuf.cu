#include "../include/pirate/MegaRingbuf.hpp"
#include "../include/pirate/inlines.hpp"  // xmod()

#include <sstream>
#include <iomanip>
#include <ksgpu/xassert.hpp>
#include <yaml-cpp/emitter.h>

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

    this->num_producers = params.producer_nquads.size();
    this->num_consumers = params.consumer_nquads.size();
    this->segments.resize(num_producers);

    for (long producer_id = 0; producer_id < num_producers; producer_id++) {
        long nquads = params.producer_nquads[producer_id];
        this->segments[producer_id].resize(nquads);
    }
}


void MegaRingbuf::add_segment(long producer_id, long producer_iquad, long consumer_id, long consumer_iquad, long chunk_lag)
{
    xassert(producer_id >= 0);
    xassert(producer_id < num_producers);
    xassert(producer_iquad >= 0);
    xassert(producer_iquad < params.producer_nquads[producer_id]);

    xassert(consumer_id >= 0);
    xassert(consumer_id < num_consumers);
    xassert(consumer_iquad >= 0);
    xassert(consumer_iquad < params.consumer_nquads[consumer_id]);
    
    xassert(chunk_lag >= 0);
    this->max_clag = max(max_clag, chunk_lag);

    Segment &s = this->segments[producer_id][producer_iquad];
    xassert(s.num_consumers < max_consumers_per_producer);

    long n = s.num_consumers++;
    s.consumer_ids[n] = consumer_id;
    s.consumer_iquads[n] = consumer_iquad;
    s.chunk_lags[n] = chunk_lag;
}


void MegaRingbuf::finalize(bool delete_internals)
{
    if (is_finalized)
        throw runtime_error("double call to MegaRingbuf::finalize()");

    this->max_gpu_clag = min(params.max_gpu_clag, max_clag);
    this->max_gpu_clag = max(max_gpu_clag, 0L);
    
    // Part 1: partially initialize zones
    //
    //  - Number of zones is determined
    //  - Zone::num_frames is initialized
    //  - Zone::segments_per_frame is not initialized
    //  - Zone::global_segment_offset is not initialized
    
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
    //   - Initialize Zone::segments_per_beam (but not Zone::global_segment_offset).

    this->producer_triples.resize(num_producers);
    this->consumer_triples.resize(num_consumers);

    for (long producer_id = 0; producer_id < num_producers; producer_id++) {
        long producer_nquads = params.producer_nquads.at(producer_id);
        this->producer_triples.at(producer_id).resize(producer_nquads);
    }

    for (long consumer_id = 0; consumer_id < num_consumers; consumer_id++) {
        long consumer_nquads = params.consumer_nquads.at(consumer_id);
        this->consumer_triples.at(consumer_id).resize(consumer_nquads);
    }

    // Outer loop over segments (with associated producer info).
    for (long producer_id = 0; producer_id < num_producers; producer_id++) {
        long producer_nquads = params.producer_nquads[producer_id];
        for (long producer_iquad = 0; producer_iquad < producer_nquads; producer_iquad++) {
            Segment &s = this->segments[producer_id][producer_iquad];

            // FIXME hmm when writing code in this loop, I had to be careful to
            // distinguish between this->num_consumers and s.num_consumers. I should
            // probably rename things, to avoid a potential source of bugs.

            // Check that every (producer_id, producer_iquad) pair has received at
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

            // Now we're ready to implement steps 1 and 2 above.

            Triple &producer = producer_triples.at(producer_id).at(producer_iquad);
            xassert(producer.zone == nullptr);   // paranoid

            if (ncpu == 0) {
                // producer -> gpu_zone
                gpu_zone->segments_per_frame++;  // see comment just above _set_triple()
                _set_triple(producer, gpu_zone, 0);
            }
            else if (ngpu == 0) {
                // producer -> xfer_zone
                host_zone->segments_per_frame++;
                xfer_zone->segments_per_frame++;
                _set_triple(producer, xfer_zone, 0);
            }
            else {
                // producer -> gpu_zone -> xfer zone 
                gpu_zone->segments_per_frame++; 
                host_zone->segments_per_frame++;
                xfer_zone->segments_per_frame++;
                _set_triple(producer, gpu_zone, 0);
                _push_triple(g2g_triples, gpu_zone, gpu_clag * BT);  // g2g src
                _push_triple(g2g_triples, xfer_zone, 0);            // g2g dst
            }

            // Inner loop over consumers. This implements steps 4 and 5 above.

            for (long c = 0; c < s.num_consumers; c++) {
                long consumer_id = s.consumer_ids[c];
                long consumer_iquad = s.consumer_iquads[c];
                long chunk_lag = s.chunk_lags[c];

                // Check that each (consumer_id, consumer_iquad) pair has received at
                // most one call to add_segment().
                Triple &consumer = consumer_triples.at(consumer_id).at(consumer_iquad);
                xassert(consumer.zone == nullptr);

                if (c < ngpu)
                    // gpu_zone -> consumer
                    _set_triple(consumer, gpu_zone, chunk_lag * BT);
                else if (chunk_lag == gpu_clag + host_clag)
                    // xfer_zone -> consumer
                    _set_triple(consumer, xfer_zone, BA);
                else {
                    // host_zone -> et_host_zone
                    // et_gpu_zone -> consumer
                    et_host_zone.segments_per_frame++;
                    et_gpu_zone.segments_per_frame++;
                    _push_triple(h2h_triples, host_zone, (chunk_lag-gpu_clag) * BT);  // h2h src
                    _push_triple(h2h_triples, &et_host_zone, 0);                      // h2h dst
                    _set_triple(consumer, &et_gpu_zone, 0);

                    long headroom = gpu_clag + host_clag - chunk_lag;
                    xassert(headroom > 0);

                    if ((et_h2h_headroom < 0) || (et_h2h_headroom > headroom))
                        this->et_h2h_headroom = headroom;
                }
            }
        }
    }

    // Check that each (consumer_id, consumer_iquad) pair has received at
    // least one call to add_segment().

    for (long consumer_id = 0; consumer_id < num_consumers; consumer_id++) {
        long consumer_nquads = params.consumer_nquads.at(consumer_id);
        for (long consumer_iquad = 0; consumer_iquad < consumer_nquads; consumer_iquad++) {
            Triple &consumer = consumer_triples.at(consumer_id).at(consumer_iquad);
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

    if (delete_internals)
        this->_delete_internals();

    this->is_finalized = true;
}


// Note: no 'segment_within_frame' arg!
// Instead, we set t.segment_within_frame to point to the last segment in the zone 'zp'.
// Caller is responsible for incrementing zp->segments_per_frame if needed.
void MegaRingbuf::_set_triple(Triple &t, Zone *zp, long frame_lag)
{
    xassert(zp != nullptr);
    xassert(zp->segments_per_frame > 0);
    xassert((frame_lag >= 0) && (frame_lag < zp->num_frames));
    xassert(t.zone == nullptr);

    t.zone = zp;
    t.frame_lag = frame_lag;
    t.segment_within_frame = zp->segments_per_frame - 1;  // last segment in zone
}

void MegaRingbuf::_push_triple(std::vector<Triple> &triples, Zone *zp, long frame_lag)
{
    Triple t;
    _set_triple(t, zp, frame_lag);
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
    xassert(z.global_segment_offset < 0);

    if (on_gpu) {
        z.global_segment_offset = gpu_global_nseg;
        gpu_global_nseg += z.num_frames * z.segments_per_frame;
    } else {
        z.global_segment_offset = host_global_nseg;
        host_global_nseg += z.num_frames * z.segments_per_frame;
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
        xassert(t.zone->global_segment_offset >= 0);
        xassert(t.frame_lag < t.zone->num_frames);
        xassert(t.segment_within_frame < t.zone->segments_per_frame);

        uint global_segment_offset = to_uint(t.zone->global_segment_offset + t.segment_within_frame);
        uint frame_offset_within_zone = to_uint(xmod(t.zone->num_frames - t.frame_lag, t.zone->num_frames));
        uint frames_in_zone = to_uint(t.zone->num_frames);
        uint segments_per_frame = to_uint(t.zone->segments_per_frame);

        ret.data[4*i] = global_segment_offset;
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

void MegaRingbuf::_delete_internals()
{
    this->segments.clear();
    this->producer_triples.clear();
    this->consumer_triples.clear();
    this->g2g_triples.clear();
    this->h2h_triples.clear();
    // Zones can stay, since they don't take up much memory.
}

// This static member function makes a random simplified MegaRingbuf as follows:
//
//   - num_consumers == num_producers == 1.
//   - "pure gpu", i.e. no host_zones, xfer_zones, et_*_zones.
//   - for each zone, frames_in_zone is random (i.e. unrelated to zone index)
//   - segment assignment is random (i.e. unrelated to dedispersion)
//
// This is intended for standalone testing of (uncoalseced or coalesced) dedispersion
// kernels, but is probably not useful for much else.

shared_ptr<MegaRingbuf> MegaRingbuf::make_random_simplified(long total_beams, long active_beams, long nchunks, long nquads)
{
    Params params;
    params.total_beams = total_beams;
    params.active_beams = active_beams;
    params.producer_nquads = { nquads };
    params.consumer_nquads = { nquads };

    shared_ptr<MegaRingbuf> ret = make_shared<MegaRingbuf> (params);

    long num_zones = rand_int(1, 11);
    ret->gpu_zones.resize(num_zones);

    for (long i = 0; i < num_zones; i++)
        ret->gpu_zones.at(i).num_frames = rand_int(2*active_beams, nchunks*total_beams + active_beams + 1);

    ret->producer_triples.resize(1);
    ret->consumer_triples.resize(1);

    for (long i = 0; i < nquads; i++) {
        long izone = rand_int(0, num_zones);
        Zone *zone = &ret->gpu_zones.at(izone);

        long num_frames = zone->num_frames;
        long producer_frame_lag = rand_int(0, num_frames - 2*active_beams + 1);
        long consumer_frame_lag = rand_int(producer_frame_lag + active_beams, num_frames - active_beams + 1);

        zone->segments_per_frame++;
        ret->_push_triple(ret->producer_triples.at(0), zone, producer_frame_lag);
        ret->_push_triple(ret->consumer_triples.at(0), zone, consumer_frame_lag);
    }

    ksgpu::randomly_permute(ret->producer_triples[0]);
    ksgpu::randomly_permute(ret->consumer_triples[0]);

    ret->_lay_out_zones(ret->gpu_zones, true);
    ret->producer_quadruples = ret->_triples_to_quadruples(ret->producer_triples);
    ret->consumer_quadruples = ret->_triples_to_quadruples(ret->consumer_triples);

    ret->_delete_internals();
    ret->is_finalized = true;

    return ret;
}


shared_ptr<MegaRingbuf> MegaRingbuf::make_trivial(long total_beams, long nquads)
{
    Params params;
    params.total_beams = total_beams;
    params.active_beams = total_beams;
    params.producer_nquads = { nquads };
    params.consumer_nquads = { nquads };

    shared_ptr<MegaRingbuf> ret = make_shared<MegaRingbuf> (params);

    ret->gpu_zones.resize(1);
    ret->producer_triples.resize(1);
    ret->consumer_triples.resize(1);

    Zone *z = &ret->gpu_zones.at(0);
    z->num_frames = 2 * total_beams;

    for (long i = 0; i < nquads; i++) {
        z->segments_per_frame++;
        ret->_push_triple(ret->producer_triples.at(0), z, 0);
        ret->_push_triple(ret->consumer_triples.at(0), z, total_beams);
    }

    ret->_lay_out_zones(ret->gpu_zones, true);
    ret->producer_quadruples = ret->_triples_to_quadruples(ret->producer_triples);
    ret->consumer_quadruples = ret->_triples_to_quadruples(ret->consumer_triples);

    ret->_delete_internals();
    ret->is_finalized = true;

    return ret;
}


// Helper function for printing memory zone info in verbose mode
static void _emit_memory_zones(YAML::Emitter &emitter, const string &key, double total_gib, const vector<MegaRingbuf::Zone> &zones, bool verbose)
{
    emitter << YAML::Key << key;
    
    if (!verbose) {
        stringstream ss;
        ss << fixed << setprecision(2) << total_gib << " GiB";
        emitter << YAML::Value << ss.str();
    }
    else {
        emitter << YAML::Value << YAML::BeginMap;
        
        // Print total
        {
            stringstream ss;
            ss << fixed << setprecision(2) << total_gib << " GiB";
            emitter << YAML::Key << "total" << YAML::Value << ss.str();
        }
        
        // Print zones (at most 5 per line)
        emitter << YAML::Key << "zones" << YAML::Value << YAML::Flow << YAML::BeginSeq;
        
        int zones_on_line = 0;
        for (long clag = 0; clag < (long)zones.size(); clag++) {
            const MegaRingbuf::Zone &z = zones.at(clag);
            double capacity_gib = z.num_frames * z.segments_per_frame * 128.0 / (1L << 30);
            
            if (capacity_gib > 0) {
                if (zones_on_line == 5) {
                    emitter << YAML::Newline;
                    zones_on_line = 0;
                }
                
                emitter << YAML::BeginSeq;
                emitter << clag;
                
                stringstream ss;
                ss << fixed << setprecision(2) << capacity_gib << " GiB";
                emitter << ss.str();
                
                emitter << YAML::EndSeq;
                zones_on_line++;
            }
        }
        emitter << YAML::EndSeq;
        
        emitter << YAML::EndMap;
    }
}


void MegaRingbuf::to_yaml(YAML::Emitter &emitter, double frames_per_second, long nfreq, long time_samples_per_chunk, bool verbose) const
{
    emitter << YAML::BeginMap;

    // Compute intermediate values
    long beams_per_gpu = params.total_beams;
    double T = beams_per_gpu / frames_per_second;
    
    long xseg = 0;
    for (const Zone &z: xfer_zones)
        xseg += z.segments_per_frame;
    xseg /= 2;
    
    long etseg = et_gpu_zone.segments_per_frame;
    double nsamp = double(beams_per_gpu) * nfreq * time_samples_per_chunk;

    double host_gib = host_global_nseg * 128.0 / (1L << 30);
    double gpu_gib = gpu_global_nseg * 128.0 / (1L << 30);
    double host_to_gpu_gbps = (xseg + etseg) * 128.0 * beams_per_gpu / T / 1.0e9;
    double gpu_to_host_gbps = xseg * 128.0 * beams_per_gpu / T / 1.0e9;
    double et_h2h_gbps = (h2h_octuples.size / 8) * 2.0 * 128.0 * beams_per_gpu / T / 1.0e9;

    _emit_memory_zones(emitter, "host_zones", host_gib, host_zones, verbose);
    _emit_memory_zones(emitter, "gpu_zones", gpu_gib, gpu_zones, verbose);
    {
        stringstream ss;
        ss << fixed << setprecision(3) << host_to_gpu_gbps << " GB/s";
        emitter << YAML::Key << "h2g bandwidth" << YAML::Value << ss.str();
        if (verbose) {
            stringstream cs;
            cs << "plus raw data: " << fixed << setprecision(3) << (nsamp / T * 0.5 / 1.0e9)
               << " GB/s if 4-bit, " << (nsamp / T / 1.0e9) << " GB/s if 8-bit";
            emitter << YAML::Comment(cs.str());
        }
    }
    {
        stringstream ss;
        ss << fixed << setprecision(3) << gpu_to_host_gbps << " GB/s";
        emitter << YAML::Key << "g2h bandwidth" << YAML::Value << ss.str();
    }
    {
        stringstream ss;
        ss << fixed << setprecision(3) << et_h2h_gbps << " GB/s";
        emitter << YAML::Key << "et_h2h bandwidth" << YAML::Value << ss.str();
    }

    if (et_h2h_headroom > 0) {
        stringstream ss;
        ss << fixed << setprecision(2) << (et_h2h_headroom * T) << " seconds";
        emitter << YAML::Key << "et_h2h_headroom" << YAML::Value << ss.str();
    }

    emitter << YAML::Key << "max_clag" << YAML::Value << max_clag;
    emitter << YAML::Key << "max_gpu_clag" << YAML::Value << max_gpu_clag;

    emitter << YAML::EndMap;
}


}  // namespace pirate
