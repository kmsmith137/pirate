#include "../include/pirate/DedispersionPlan.hpp"
#include "../include/pirate/constants.hpp"
#include "../include/pirate/inlines.hpp"  // align_up(), pow2(), print_kv(), Indent
#include "../include/pirate/utils.hpp"    // bit_reverse_slow(), rb_lag(), rstate_len(), mean_bytes_per_unaligned_chunk()

#include <ksgpu/xassert.hpp>

using namespace std;
using namespace ksgpu;

namespace pirate {
#if 0
}  // editor auto-indent
#endif


// -------------------------------------------------------------------------------------------------
//
// RingbufEntry: helper for DedispersionPlan constructor.


struct RingbufEntry
{
    using Ringbuf = DedispersionPlan::Ringbuf;
    
    Ringbuf *rb = nullptr;
    long xlag = 0;   // lag of (time chunk, beam) pair (usually clag * total beams)
    long iseg = 0;   // ring buffer segment index, within (time chunk, beam) pair.

    RingbufEntry(Ringbuf *rb_, int xlag_, uint iseg_) :
	rb(rb_), xlag(xlag_), iseg(iseg_)
    {
	xassert(rb != nullptr);
	xassert(xlag >= 0);
	xassert(iseg >= 0);
    }

    RingbufEntry() { }
};


inline uint to_uint(long n)
{
    xassert(n >= 0);
    xassert(n < (1L << 32));
    return uint(n);
}


static Array<uint> rb_locs_from_entries(const vector<RingbufEntry> &entries)
{
    using Ringbuf = DedispersionPlan::Ringbuf;
    
    long n = entries.size();
    Array<uint> ret({n,4}, af_rhost);

    for (long i = 0; i < n; i++) {
	const RingbufEntry &e = entries[i];
	const Ringbuf *rb = e.rb;
	xassert(rb != nullptr);
	
	long rb_len = rb->rb_len;
	long rb_nseg = rb->nseg_per_beam;
	
	xassert(rb->base_segment >= 0);	
	xassert(e.xlag < rb_len);
	xassert(e.iseg < rb_nseg);

	//  uint rb_offset;  // in segments, not bytes
	//  uint rb_phase;   // index of (time chunk, beam) pair, relative to current pair
	//  uint rb_len;     // number of (time chunk, beam) pairs in ringbuf (same as Ringbuf::rb_len)
	//  uint rb_nseg;    // number of segments per (time chunk, beam) (same as Ringbuf::nseg_per_beam)
	
	ret.data[4*i] = to_uint(rb->base_segment + e.iseg);
	ret.data[4*i+1] = to_uint(xmod(rb_len - e.xlag, rb_len));
	ret.data[4*i+2] = to_uint(rb_len);
	ret.data[4*i+3] = to_uint(rb_nseg);
    }
    
    return ret;
}


static void lay_out_ringbuf(DedispersionPlan::Ringbuf &rb, long &nseg_tracker)
{
    xassert(nseg_tracker >= 0);
    xassert(rb.base_segment < 0);
    
    rb.base_segment = nseg_tracker;
    nseg_tracker += rb.rb_len * rb.nseg_per_beam;
}


// -------------------------------------------------------------------------------------------------


DedispersionPlan::DedispersionPlan(const DedispersionConfig &config_) :
    config(config_)
{
    config.validate();

    // 'nelts_per_segment' is always (constants::bytes_per_gpu_cache_line / sizeof(dtype)).
    this->nelts_per_segment = config.get_nelts_per_segment();
    this->nbytes_per_segment = constants::bytes_per_gpu_cache_line;

    // Part 1:
    //   - Initialize stage1_trees, stage2_trees.
    //   - Initialize max_n1 (max number of Stage2Trees, per Stage1Tree).
    //   - Initialize stage1_ntrees, stage2_ntrees.

    int max_n1 = 0;
    
    for (int ids = 0; ids < config.num_downsampling_levels; ids++) {
	
	// Note that Stage1Tree::rank0 can be different for downsampled trees vs the
	// non-downsampled tree, but is the same for different downsampled trees.
	// This property is necessary in order for the LaggedDownsampler to work later.
	
	int st1_rank = ids ? (config.tree_rank - 1) : config.tree_rank;
	int st1_rank0 = st1_rank/2;
	vector<int> trigger_ranks;
	
	for (const DedispersionConfig::EarlyTrigger &et: config.early_triggers) {
	    if (et.ds_level == ids)
		trigger_ranks.push_back(et.tree_rank);
	}

	trigger_ranks.push_back(st1_rank);
	xassert(is_sorted(trigger_ranks));

	Stage1Tree st1;
	st1.ds_level = ids;
	st1.rank0 = st1_rank0;
	st1.rank1 = st1_rank - st1.rank0;
	st1.nt_ds = xdiv(config.time_samples_per_chunk, pow2(ids));
	st1.segments_per_beam = pow2(st1_rank) * xdiv(st1.nt_ds, nelts_per_segment);
	st1.base_segment = this->stage1_total_segments_per_beam;

	// FIXME should replace hardcoded 7,8 by something more descriptive
	// (GpuDedispersionKernel::max_rank?)
	int max_rank = ids ? 7 : 8;
	xassert((st1.rank0 >= 0) && (st1.rank0 <= max_rank));
	xassert(st1.nt_ds > 0);
	
	this->stage1_trees.push_back(st1);
	this->stage1_total_segments_per_beam += st1.segments_per_beam;

	for (int trigger_rank: trigger_ranks) {
	    Stage2Tree st2;
	    st2.ds_level = ids;
	    st2.rank0 = st1.rank0;
	    st2.rank1_ambient = st1.rank1;
	    st2.rank1_trigger = trigger_rank - st2.rank0;
	    st2.nt_ds = st1.nt_ds;
	    st2.segments_per_beam = pow2(trigger_rank) * xdiv(st2.nt_ds, nelts_per_segment);
	    st2.base_segment = this->stage2_total_segments_per_beam;

	    xassert((st2.rank1_trigger >= 0) && (st2.rank1_trigger <= 8));
	    xassert(st2.rank1_trigger <= st2.rank1_ambient);
		     
	    this->stage2_trees.push_back(st2);
	    this->stage2_total_segments_per_beam += st2.segments_per_beam;
	}

	max_n1 = max(max_n1, int(trigger_ranks.size()));
    }

    this->stage1_ntrees = stage1_trees.size();
    this->stage2_ntrees = stage2_trees.size();
    xassert(stage1_ntrees == config.num_downsampling_levels);
    
    // Part 2:
    //  - Initialize this->max_clag, this->max_gpu_clag.
    //  - Allocate 'segmap', which maps iseg0 -> (list of (clag,iseg1) pairs).
    //    (This is a temporary object that will be used "locally" in this function.)
    
    int nseg0 = this->stage1_total_segments_per_beam;
    int nseg1 = this->stage2_total_segments_per_beam;

    Array<uint> segmap_n1({nseg0}, af_uhost | af_zero);
    Array<uint> segmap_clag({nseg0,max_n1}, af_uhost | af_zero);
    Array<uint> segmap_iseg1({nseg0,max_n1}, af_uhost | af_zero);
    
    this->max_clag = 0;
    
    for (const Stage2Tree &st2: this->stage2_trees) {
	const Stage1Tree &st1 = this->stage1_trees.at(st2.ds_level);

	// Some truly paranoid asserts.
	xassert(st1.nt_ds == st2.nt_ds);
	xassert(st1.rank0 == st2.rank0);
	xassert(st1.ds_level == st2.ds_level);
	xassert(st1.rank1 == st2.rank1_ambient);

	int nchan0 = pow2(st2.rank0);
	int nchan1 = pow2(st2.rank1_trigger);  // not rank1_ambient
	int ns = xdiv(st2.nt_ds, this->nelts_per_segment);
	bool is_downsampled = (st2.ds_level > 0);
	
	for (int i0 = 0; i0 < nchan0; i0++) {
	    for (int i1 = 0; i1 < nchan1; i1++) {
		int lag = rb_lag(i1, i0, st2.rank0, st2.rank1_trigger, is_downsampled);
		int slag = lag / nelts_per_segment;  // round down
		
		for (int s0 = 0; s0 < ns; s0++) {
		    int clag = (s0 + slag) / ns;
		    int s1 = (s0 + slag) - (clag * ns);
		    xassert((s1 >= 0) && (s1 < ns));

		    // iseg0 -> (s,i1,i0)
		    int iseg0 = (s0 * pow2(st1.rank1)) + i1;
		    iseg0 = (iseg0 * pow2(st1.rank0)) + i0;
		    iseg0 += st1.base_segment;
		    xassert((iseg0 >= 0) && (iseg0 < nseg0));

		    // iseg1 -> (s,i0,i1)
		    int iseg1 = (s1 * nchan0) + i0;
		    iseg1 = (iseg1 * nchan1) + i1;
		    iseg1 += st2.base_segment;
		    xassert((iseg1 >= 0) && (iseg1 < nseg1));

		    // Add (clag, iseg1) to segmap[iseg0].
		    int n1 = segmap_n1.data[iseg0]++;
		    xassert((n1 >= 0) && (n1 < max_n1));
		    segmap_clag.data[iseg0*max_n1 + n1] = clag;
		    segmap_iseg1.data[iseg0*max_n1 + n1] = iseg1;

		    // Check that clags are sorted.
		    if (n1 > 0)
			xassert(segmap_clag.data[iseg0*max_n1 + n1-1] <= uint(clag));

		    // Update max_clag.
		    this->max_clag = max(max_clag, clag);
		}
	    }
	}
    }
    
    this->max_gpu_clag = int(max_clag * config.gpu_clag_maxfrac + 0.5);
    this->max_gpu_clag = min(max_gpu_clag, max_clag);
    this->max_gpu_clag = max(max_gpu_clag, 0);

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
	this->gpu_ringbufs.at(clag).rb_len = clag*BT + BA;
	this->host_ringbufs.at(clag).rb_len = clag*BT + BA;
	this->xfer_ringbufs.at(clag).rb_len = 2*BA;
    }

    this->et_host_ringbuf.rb_len = BA;
    this->et_gpu_ringbuf.rb_len = BA;

    // Part 4:
    //
    //   - Initialize "local" RingbufEntry vectors.
    //     These will be converted to integer-valued "_locs" arrays in Part 5.
    //
    //   - Initialize Ringbuf::nseg_per_beam.

    vector<RingbufEntry> stage1_entries(nseg0);  // output locations for stage1 dd kernels
    vector<RingbufEntry> stage2_entries(nseg1);  // input locations for stage2 dd kernels
    vector<RingbufEntry> g2g_entries;            // (src,dst) pairs for g2g kernels (copy gpu -> xfer)
    vector<RingbufEntry> h2h_entries;            // (src,dst) pairs for h2h kernels (copy host -> et_host)

    // Outer loop over stage1 segments.
    for (int iseg0 = 0; iseg0 < nseg0; iseg0++) {
	int n1 = segmap_n1.data[iseg0];
	xassert((n1 > 0) && (n1 <= max_n1));

	// Reminder: clag_vec is sorted (this is xassert()-ed above).
	const uint *clag_vec = segmap_clag.data + (iseg0 * max_n1);    // length n1
	const uint *iseg1_vec = segmap_iseg1.data + (iseg0 * max_n1);  // length n1

	// Split total clag between GPU and host.
	int gpu_clag = 0;
	for (int i1 = 0; i1 < n1; i1++) {
	    uint clag_prev = i1 ? clag_vec[i1-1] : 0;
	    if (clag_vec[i1] > clag_prev + max_gpu_clag)
		break;
	    gpu_clag = clag_vec[i1];
	}

	int host_clag = clag_vec[n1-1] - gpu_clag;
	bool goes_to_host_ringbuf = (host_clag > 0);
	bool goes_to_gpu_ringbuf = (clag_vec[0] <= uint(gpu_clag));  // includes case clag_vec[0]==0
	
	// Note: it's possible to have (goes_to_host_ringbuf == goes_to_gpu_ringbuf == true),
	// but only if there are early triggers.
	xassert(goes_to_host_ringbuf || goes_to_gpu_ringbuf);
	
	Ringbuf *gpu_rb = goes_to_gpu_ringbuf ? &this->gpu_ringbufs.at(gpu_clag) : nullptr;
	Ringbuf *host_rb = goes_to_host_ringbuf ? &this->host_ringbufs.at(host_clag) : nullptr;
	Ringbuf *xfer_rb = goes_to_host_ringbuf ? &this->xfer_ringbufs.at(host_clag) : nullptr;
	
	long gpu_iseg = goes_to_gpu_ringbuf ? (gpu_rb->nseg_per_beam++) : (-1);
	long host_iseg = goes_to_host_ringbuf ? (host_rb->nseg_per_beam++) : (-1);

	// (gpu dedispersion buffer) -> (either gpu or xfer ringbuf).
	if (goes_to_gpu_ringbuf)
	    stage1_entries[iseg0] = RingbufEntry(gpu_rb, 0, gpu_iseg);
	else
	    stage1_entries[iseg0] = RingbufEntry(xfer_rb, 0, host_iseg);

	// If needed, copy (gpu ringbuf) -> (xfer ringbuf).
	if (goes_to_gpu_ringbuf && goes_to_host_ringbuf) {
	    RingbufEntry g2g_src(gpu_rb, gpu_clag * BT, gpu_iseg);
	    RingbufEntry g2g_dst(xfer_rb, 0, host_iseg);
	    g2g_entries.push_back(g2g_src);
	    g2g_entries.push_back(g2g_dst);
	}

	// Inner loop over stage2 segments
	for (int i1 = 0; i1 < n1; i1++) {
	    int clag = clag_vec[i1];
	    int iseg1 = iseg1_vec[i1];

	    xassert((iseg1 >= 0) && (iseg1 < nseg1));
	    xassert(stage2_entries.at(iseg1).rb == nullptr);

	    if (clag <= gpu_clag) {
		// get segment from gpu ringbuf
		xassert(goes_to_gpu_ringbuf);
		stage2_entries[iseg1] = RingbufEntry(gpu_rb, clag * BT,  gpu_iseg);
	    }
	    else if (clag == gpu_clag + host_clag) {
		// get segment from xfer ringbuf (after transfer host -> gpu)
		stage2_entries[iseg1] = RingbufEntry(xfer_rb, BA, host_iseg);
	    }
	    else {
		// early trigger: get segment from et ringbuf
		long et_iseg = this->et_host_ringbuf.nseg_per_beam++;
		stage2_entries[iseg1] = RingbufEntry(&et_gpu_ringbuf, 0, et_iseg);
		RingbufEntry h2h_src(host_rb, (clag-gpu_clag) * BT, host_iseg);
		RingbufEntry h2h_dst(&et_host_ringbuf, 0, et_iseg);
		h2h_entries.push_back(h2h_src);
		h2h_entries.push_back(h2h_dst);
	    }
	}
    }

    // At the end of part 4, we initialized Ringbuf::nseg_per_beam for some ringbufs
    // (host, gpu, et_host), but not others (xfer, et_gpu). Tie up this loose end.

    for (int clag = 0; clag <= max_clag; clag++)
	this->xfer_ringbufs.at(clag).nseg_per_beam = this->host_ringbufs.at(clag).nseg_per_beam;

    this->et_gpu_ringbuf.nseg_per_beam = this->et_host_ringbuf.nseg_per_beam;

    
    // Part 5:
    //
    //  - Lay ring buffers out in memory:
    //      - Initialize Ringbuf::base_segment
    //      - Initialize DedispersionPlan::{gmem,hmem}_ringbuf_nseg
    //
    //  - Initialize all uint[4*N] arrays used in GPU kernels:
    //      - DedispersionPlan::{stage1,stage2}_rb_locs
    //      - DedispersionPlan::{g2g,h2h}_rb_locs

    
    this->gmem_ringbuf_nseg = 0;
    this->hmem_ringbuf_nseg = 0;

    for (int clag = 0; clag <= max_clag; clag++)
	lay_out_ringbuf(gpu_ringbufs.at(clag), this->gmem_ringbuf_nseg);
    
    for (int clag = 0; clag <= max_clag; clag++)
	lay_out_ringbuf(host_ringbufs.at(clag), this->hmem_ringbuf_nseg);
    
    for (int clag = 0; clag <= max_clag; clag++)
	lay_out_ringbuf(xfer_ringbufs.at(clag), this->gmem_ringbuf_nseg);

    lay_out_ringbuf(et_host_ringbuf, this->hmem_ringbuf_nseg);
    lay_out_ringbuf(et_gpu_ringbuf, this->gmem_ringbuf_nseg);

    this->stage1_rb_locs = rb_locs_from_entries(stage1_entries);
    this->stage2_rb_locs = rb_locs_from_entries(stage2_entries);
    this->g2g_rb_locs = rb_locs_from_entries(g2g_entries);
    this->h2h_rb_locs = rb_locs_from_entries(h2h_entries);

    
    // Part 6: initialize all "params" members:
    //
    //   DedispersionBufferParams stage1_dd_buf_params;
    //   DedispersionBufferParams stage2_dd_buf_params;
    //
    //   std::vector<DedispersionKernelParams> stage1_dd_kernel_params;  // length stage1_ntrees
    //   std::vector<DedispersionKernelParams> stage2_dd_kernel_params;  // length stage2_ntrees
    //   std::vector<long> stage2_ds_level;                              // length stage2_ntrees
    //
    //   LaggedDownsamplingKernelParams lds_params;
    //   RingbufCopyKernelParams g2g_copy_kernel_params;
    //   RingbufCopyKernelParams h2h_copy_kernel_params;
    
    stage1_dd_buf_params.dtype = config.dtype;
    stage1_dd_buf_params.beams_per_batch = config.beams_per_batch;
    stage1_dd_buf_params.nbuf = stage1_ntrees;

    for (Stage1Tree &st1: stage1_trees) {
	long pos = st1.base_segment;
	long nseg = st1.segments_per_beam;

	DedispersionKernelParams kparams;
	kparams.dtype = config.dtype;
	kparams.dd_rank = st1.rank0;
	kparams.amb_rank = st1.rank1;
	kparams.total_beams = config.beams_per_gpu;
	kparams.beams_per_batch = config.beams_per_batch;
	kparams.ntime = st1.nt_ds;
	kparams.nspec = 1;
	kparams.input_is_ringbuf = false;
	kparams.output_is_ringbuf = true;   // note output_is_ringbuf = true
	kparams.apply_input_residual_lags = false;
	kparams.input_is_downsampled_tree = (st1.ds_level > 0);
	kparams.nt_per_segment = this->nelts_per_segment;
	kparams.ringbuf_locations = this->stage1_rb_locs.slice(0, pos, pos + nseg);
	kparams.ringbuf_nseg = this->gmem_ringbuf_nseg;
	kparams.validate();
	
	stage1_dd_buf_params.buf_rank.push_back(st1.rank0 + st1.rank1);
	stage1_dd_buf_params.buf_ntime.push_back(st1.nt_ds);
	stage1_dd_kernel_params.push_back(kparams);
    }

    stage2_dd_buf_params.dtype = config.dtype;
    stage2_dd_buf_params.beams_per_batch = config.beams_per_batch;
    stage2_dd_buf_params.nbuf = stage2_ntrees;

    for (Stage2Tree &st2: stage2_trees) {
	long pos = st2.base_segment;
	long nseg = st2.segments_per_beam;
	long ds_level = st2.ds_level;

	xassert(st2.nt_ds == xdiv(config.time_samples_per_chunk, pow2(ds_level)));

	DedispersionKernelParams kparams;
	kparams.dtype = config.dtype;
	kparams.dd_rank = st2.rank1_trigger;
	kparams.amb_rank = st2.rank0;
	kparams.total_beams = config.beams_per_gpu;
	kparams.beams_per_batch = config.beams_per_batch;
	kparams.ntime = st2.nt_ds;
	kparams.nspec = 1;
	kparams.input_is_ringbuf = true;   // note input_is_ringbuf = true
	kparams.output_is_ringbuf = false;
	kparams.apply_input_residual_lags = true;
	kparams.input_is_downsampled_tree = (ds_level > 0);
	kparams.nt_per_segment = this->nelts_per_segment;
	kparams.ringbuf_locations = this->stage2_rb_locs.slice(0, pos, pos + nseg);
	kparams.ringbuf_nseg = this->gmem_ringbuf_nseg;
	kparams.validate();

	stage2_dd_buf_params.buf_rank.push_back(st2.rank0 + st2.rank1_trigger);
	stage2_dd_buf_params.buf_ntime.push_back(st2.nt_ds);
	stage2_dd_kernel_params.push_back(kparams);
	stage2_ds_level.push_back(ds_level);
    }

    // Note that 'output_dd_rank' is guaranteed to be the same for all downsampled trees.
    lds_params.dtype = config.dtype;
    lds_params.input_total_rank = config.tree_rank;
    lds_params.output_dd_rank = (stage1_ntrees > 1) ? stage1_trees.at(1).rank0 : 0;
    lds_params.num_downsampling_levels = config.num_downsampling_levels;
    lds_params.total_beams = config.beams_per_gpu;
    lds_params.beams_per_batch = config.beams_per_batch;
    lds_params.ntime = config.time_samples_per_chunk;

    g2g_copy_kernel_params.total_beams = config.beams_per_gpu;
    g2g_copy_kernel_params.beams_per_batch = config.beams_per_batch;
    g2g_copy_kernel_params.nelts_per_segment = this->nelts_per_segment;
    g2g_copy_kernel_params.locations = g2g_rb_locs;
    
    h2h_copy_kernel_params.total_beams = config.beams_per_gpu;
    h2h_copy_kernel_params.beams_per_batch = config.beams_per_batch;
    h2h_copy_kernel_params.nelts_per_segment = this->nelts_per_segment;
    h2h_copy_kernel_params.locations = h2h_rb_locs;
    
    lds_params.validate();
    stage1_dd_buf_params.validate();
    stage2_dd_buf_params.validate();
    g2g_copy_kernel_params.validate();
    h2h_copy_kernel_params.validate();
}


// ------------------------------------------------------------------------------------------------


void DedispersionPlan::print(ostream &os, int indent) const
{
    xassert(long(stage1_trees.size()) == stage1_ntrees);
    xassert(long(stage2_trees.size()) == stage2_ntrees);
    
    os << Indent(indent) << "DedispersionPlan" << endl;
    this->config.print(os, indent+4);
    
    print_kv("nelts_per_segment", nelts_per_segment, os, indent);
    print_kv("nbytes_per_segment", nbytes_per_segment, os, indent);
    print_kv("max_clag", max_clag, os, indent);

    os << Indent(indent) << "Stage1Trees" << endl;

    for (long i = 0; i < stage1_ntrees; i++) {
	const Stage1Tree &st1 = stage1_trees.at(i);
	
	os << Indent(indent+4) << i
	   << ": ds_level=" << st1.ds_level
	   << ", rank0=" << st1.rank0
	   << ", rank1=" << st1.rank1
	   << ", nt_ds=" << st1.nt_ds
	   << endl;
    }
    
    os << Indent(indent) << "Stage2Trees" << endl;

    for (long i = 0; i < stage2_ntrees; i++) {
	const Stage2Tree &st2 = stage2_trees.at(i);
	
	os << Indent(indent+4) << i
	   << ": ds_level=" << st2.ds_level
	   << ", rank0=" << st2.rank0
	   << ", rank1_amb=" << st2.rank1_ambient
	   << ", rank1_tri=" << st2.rank1_trigger
	   << ", nt_ds=" << st2.nt_ds
	   << endl;
    }
}


}  // namespace pirate
