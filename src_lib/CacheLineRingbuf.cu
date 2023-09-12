#include "../include/pirate/internals/CacheLineRingbuf.hpp"
#include "../include/pirate/internals/LaggedCacheLine.hpp"
#include "../include/pirate/internals/inlines.hpp"   // xdiv(), align_up(), print_kv(), Indent
#include "../include/pirate/constants.hpp"

#include <cassert>

using namespace std;

namespace pirate {
#if 0
}  // editor auto-indent
#endif


CacheLineRingbuf::CacheLineRingbuf(const DedispersionConfig &config_, const ConstructorArgs &cargs_)
    : config(config_), cargs(cargs_)
{
    config.validate();
    
    assert(cargs.num_src_segments > 0);
    assert(cargs.num_dst_segments > 0);
    assert(cargs.gmem_nbytes_used_so_far > 0);
    assert((cargs.gmem_nbytes_used_so_far % constants::bytes_per_gpu_cache_line) == 0);

    // Sentinel buffer with rb_lag=0.
    buffers.push_back(Buffer());
}


void CacheLineRingbuf::add(const LaggedCacheLine &cl)
{
    assert(cl.ndst > 0);
    assert(!this->finalized);

    // For each destination (clag, segment), we will either:
    //   - add an Entry to a Buffer
    //   - define a cache line copy (either stage0->stage1 or stage1->stage1)

    // The prev_* variables are initialized when a new Entry is added to a Buffer.
    
    int prev_clag = 0;           // dst_clag which triggered the new Entry
    int prev_dseg = -1;          // dst_segment which triggered the new Entry
    int prev_rb_lag = -1;        // rb_lag of new Entry
    int prev_rb_index = -1;      // index of new Entry (in Buffer::primary_entries or Buffer::secondary_entries)
    int prev_rb_primary = true;  // is new Entry in Buffer::primary_entries, or Buffer::secondary_entries?
	
    for (int idst = 0; idst < cl.ndst; idst++) {
	int dst_clag = cl.dst_clag[idst];
	int dst_segment = cl.dst_segment[idst];

	assert(dst_segment >= 0);
	assert(dst_clag >= prev_clag);
	
	// Case 1: define a stage0->stage1 cache line copy (no ring buffering needed).
	    
	if (dst_clag == 0) {
	    PrimaryEntry e;
	    e.src_segment = cl.src_segment;
	    e.dst_segment = dst_segment;
	    
	    this->stage0_stage1_copies.push_back(e);
	    continue;
	}
	
	// Case 2: define a stage1->stage1 cache line copy (no ring buffering needed).
	
	if (dst_clag == prev_clag) {
	    PrimaryEntry e;
	    e.src_segment = prev_dseg;
	    e.dst_segment = dst_segment;
	    
	    assert(prev_dseg >= 0);	    
	    this->stage1_stage1_copies.push_back(e);
	    continue;
	}

	// If we get here, then we will add a new Entry to a Buffer.
	// First, make sure that the appropriate Buffer has been defined.
	
	int rb_lag = dst_clag - prev_clag;
	assert(rb_lag > 0);

	while ((int)buffers.size() <= rb_lag)
	    buffers.push_back(Buffer());

	Buffer &buf = buffers.at(rb_lag);

	// Now there are two cases, depending on whether this is the first Entry
	// that has been added (PrimaryEntry), or not (SecondaryEntry).
	
	if (prev_rb_lag < 0) {
	    PrimaryEntry e;
	    e.src_segment = cl.src_segment;
	    e.dst_segment = dst_segment;

	    prev_rb_primary = true;
	    prev_rb_index = buf.primary_entries.size();
	    buf.primary_entries.push_back(e);
	}
	else {
	    SecondaryEntry e;
	    e.src_rb_lag = prev_rb_lag;
	    e.src_rb_index = prev_rb_index;
	    e.src_is_primary = prev_rb_primary;
	    e.dst_segment = dst_segment;
	    
	    prev_rb_primary = false;
	    prev_rb_index = buf.secondary_entries.size();
	    buf.secondary_entries.push_back(e);
	}
	
	prev_clag = dst_clag;
	prev_dseg = dst_segment;
	prev_rb_lag = rb_lag;
	
	// At bottom of loop, all five prev_* variables have been appropriately initialized
	// for the new Entry.
    }
}


void CacheLineRingbuf::finalize()
{
    assert(!this->finalized);
    assert(this->buffers.size() > 0);
    assert(this->buffers[0].primary_entries.size() == 0);
    assert(this->buffers[0].secondary_entries.size() == 0);

    // FIXME: coverage check would be awesome
    // this->_check_coverage();

    ssize_t active_beams = config.beams_per_batch * config.num_active_batches;
    ssize_t beams_per_gpu = config.beams_per_gpu;
    ssize_t gmem_offset = cargs.gmem_nbytes_used_so_far;
    int sb = config.get_bytes_per_compressed_segment();
    
    // First loop over Buffers assigns staging buffer sizes.
    
    for (size_t rb_lag = 0; rb_lag < this->buffers.size(); rb_lag++) {
	Buffer &buf = this->buffers[rb_lag];

	ssize_t npri = align_up(sb * buf.primary_entries.size(), constants::bytes_per_segment);
	ssize_t nsec = align_up(sb * buf.secondary_entries.size(), constants::bytes_per_segment);
	
	buf.primary_nbytes_per_beam_per_chunk = npri;
	buf.secondary_nbytes_per_beam_per_chunk = nsec;
	buf.total_nbytes_per_beam_per_chunk = (npri + nsec);
	buf.total_nbytes = rb_lag * beams_per_gpu * (npri + nsec);
	buf.staging_buffer_byte_offset = gmem_offset;

	ssize_t nstage = 2 * active_beams * buf.total_nbytes_per_beam_per_chunk;
	this->gmem_nbytes_staging_buf += nstage;
	gmem_offset += nstage;
    }
    
    if (gmem_offset > config.gmem_nbytes_per_gpu) {
	stringstream ss;
	ss << "DedispersionConfig::gmem_nbytes_per_gpu (" << gputils::nbytes_to_str(config.gmem_nbytes_per_gpu)
	   << ") is too small (must be at least " << gputils::nbytes_to_str(gmem_offset) << ")";
	throw runtime_error(ss.str());
    }

    // Second loop over Buffers assigns each Buffer to either GPU memory or host memory.

    for (size_t rb_lag = 0; rb_lag < this->buffers.size(); rb_lag++) {
	Buffer &buf = this->buffers[rb_lag];
	ssize_t n0 = buf.total_nbytes_per_beam_per_chunk;
	ssize_t ntot = buf.total_nbytes;
	
	if (!config.force_ring_buffers_to_host && (gmem_offset + ntot <= config.gmem_nbytes_per_gpu)) {
	    buf.on_gpu = true;
	    buf.gmem_byte_offset = gmem_offset;
	    
	    this->gmem_nbytes_ringbuf += ntot;
	    gmem_offset += ntot;
	}
	else {
	    buf.on_gpu = false;
	    buf.hmem_byte_offset = this->hmem_nbytes_ringbuf;
	    buf.pcie_xfer_size = config.beams_per_batch * n0;
	    
	    this->hmem_nbytes_ringbuf += ntot;
	    this->pcie_nbytes_per_chunk += beams_per_gpu * n0;
	    this->pcie_memcopies_per_chunk += xdiv(beams_per_gpu, config.beams_per_batch);
	}
    }

    this->finalized = true;
}


void CacheLineRingbuf::print(std::ostream &os, int indent) const
{
    assert(this->finalized);

    // Usage reminder: print_kv(key, val, os, indent, units=nullptr)

    print_kv("stage0->stage1",
	     config.beams_per_batch * stage0_stage1_copies.size(),
	     os, indent, " cachelines/kernel");
	     
    print_kv("stage1->stage1",
	     config.beams_per_batch * stage1_stage1_copies.size(),
	     os, indent, " cachelines/kernel");

    os << Indent(indent) << "Buffers" << endl;
    
    for (unsigned int rb_lag = 0; rb_lag < buffers.size(); rb_lag++) {
	const CacheLineRingbuf::Buffer &buf = buffers.at(rb_lag);

	os << Indent(indent+4)
	   << "rb_lag=" << rb_lag
	   << ": on_gpu=" << buf.on_gpu
	   << ", nbytes=" << gputils::nbytes_to_str(buf.total_nbytes);

	if (buf.on_gpu)
	    os << ", gmem_offset=" << gputils::nbytes_to_str(buf.gmem_byte_offset);
	else
	    os << ", hmem_offset=" << gputils::nbytes_to_str(buf.hmem_byte_offset)
	       << ", pcie_xfer_size=" << gputils::nbytes_to_str(buf.pcie_xfer_size);
	
	os << endl;
    }
}


}  // namespace pirate
