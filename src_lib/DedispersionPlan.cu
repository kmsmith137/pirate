#include "../include/pirate/DedispersionPlan.hpp"
#include "../include/pirate/constants.hpp"
#include "../include/pirate/internals/utils.hpp"    // bit_reverse_slow(), rb_lag(), rstate_len(), mean_bytes_per_unaligned_chunk()
#include "../include/pirate/internals/inlines.hpp"  // align_up(), pow2(), print_kv(), Indent

#include <ksgpu/xassert.hpp>

using namespace std;
using namespace ksgpu;

namespace pirate {
#if 0
}  // editor auto-indent
#endif


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

    int max_n1 = 0;
    
    for (int ids = 0; ids < config.num_downsampling_levels; ids++) {
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
    
    // Part 2:
    //  - Initialize this->max_clag
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

		    // Update max_clag.
		    this->max_clag = max(max_clag, clag);
		}
	    }
	}
    }

    // Part 3:
    //  - allocate ringbufs
    //  - initialize Ringbuf::rb_len.

    this->gmem_ringbufs.resize(max_clag + 1);
    
    int BT = this->config.beams_per_gpu;            // total beams
    int BB = this->config.beams_per_batch;          // beams per batch
    int BA = this->config.num_active_batches * BB;  // active beams
    
    for (int clag = 0; clag <= max_clag; clag++)
	gmem_ringbufs.at(clag).rb_len = clag*BT + BA;

    // Part 4:
    //  - initialize Ringbuf::nseg_per_beam
    //  - pseudo-initialize stage1_rb_locs (*)
    //  - pseudo-initialize stage2_rb_locs (*)
    //
    // (*) "Pseudo-initialize" means that we use the following temporary rb_loc layout:
    //
    //   uint rb_seg;     // Segment within (time chunk, beam) pair
    //   uint rb_phase;   // Index of (time chunk, beam) pair, relative to current pair
    //   uint rb_clag;
    //     (fourth 'uint' is unused)

    this->stage1_rb_locs = Array<uint> ({nseg0,4}, af_rhost);
    this->stage2_rb_locs = Array<uint> ({nseg1,4}, af_rhost);

    // Not logically necessary, but enables a real-time consistency check.
    vector<bool> coverage(nseg1, false);
    
    for (int iseg0 = 0; iseg0 < nseg0; iseg0++) {
	int n1 = segmap_n1.data[iseg0];
	xassert((n1 >= 0) && (n1 <= max_n1));
	
	int rb_clag = segmap_clag.data[iseg0*max_n1 + n1-1];
	xassert((rb_clag >= 0) && (rb_clag <= max_clag));
	
	Ringbuf &rb = gmem_ringbufs.at(rb_clag);
	int rb_seg = rb.nseg_per_beam++;

	uint *rb_loc0 = stage1_rb_locs.data + (4*iseg0);
	rb_loc0[0] = rb_seg;
	rb_loc0[1] = 0;  // rb_phase
	rb_loc0[2] = rb_clag;
	
	for (int i1 = 0; i1 < n1; i1++) {
	    int clag1 = segmap_clag.data[iseg0*max_n1 + i1];
	    int iseg1 = segmap_iseg1.data[iseg0*max_n1 + i1];
	    
	    xassert((clag1 >= 0) && (clag1 <= rb_clag));  // segmap_clags must be sorted
	    xassert((iseg1 >= 0) && (iseg1 < nseg1));
	    xassert(!coverage[iseg1]);
	    coverage[iseg1] = true;

	    int rb_phase = rb.rb_len - (clag1 * BT);
	    xassert(rb_phase > 0);
	    
	    uint *rb_loc1 = stage2_rb_locs.data + (4*iseg1);
	    rb_loc1[0] = rb_seg;
	    rb_loc1[1] = rb_phase;
	    rb_loc1[2] = rb_clag;   // not clag1
	}
    }

    for (int iseg1 = 0; iseg1 < nseg1; iseg1++)
	xassert(coverage[iseg1]);

    // Part 5:
    //  - initialize this->gmem_ringbuf_nbytes
    //  - initialize Ringbuf::base_segment
    //  - fully initialize stage1_rb_locs (**)
    //  - fully initialize stage2_rb_locs (**)
    //
    // (**) "Fully initialize" means that we convert from the temporary rb_loc layout (*) to:
    //
    //   uint rb_offset;  // in segments, not bytes
    //   uint rb_phase;   // index of (time chunk, beam) pair, relative to current pair
    //   uint rb_len;     // number of (time chunk, beam) pairs in ringbuf (same as Ringbuf::rb_len)
    //   uint rb_nseg;    // number of segments per (time chunk, beam) (same as Ringbuf::nseg_per_beam)

    this->gmem_ringbuf_nseg = 0;

    for (int clag = 0; clag <= max_clag; clag++) {
	Ringbuf &rb = gmem_ringbufs.at(clag);
	rb.base_segment = this->gmem_ringbuf_nseg;
	gmem_ringbuf_nseg += rb.rb_len * rb.nseg_per_beam;
    }

    for (int iseg0 = 0; iseg0 < nseg0; iseg0++) {
	uint *p = stage1_rb_locs.data + (4*iseg0);
	Ringbuf &rb = gmem_ringbufs.at(p[2]);  // p[2] = clag

	p[0] += rb.base_segment;
	p[2] = rb.rb_len;
	p[3] = rb.nseg_per_beam;
    }
    
    for (int iseg1 = 0; iseg1 < nseg1; iseg1++) {
	uint *p = stage2_rb_locs.data + (4*iseg1);
	Ringbuf &rb = gmem_ringbufs.at(p[2]);  // p[2] = clag

	p[0] += rb.base_segment;
	p[2] = rb.rb_len;
	p[3] = rb.nseg_per_beam;
    }

    // Part 6: initialize all "params" members:
    //
    //   DedispersionBufferParams stage1_dd_buf_params;
    //   DedispersionBufferParams stage2_dd_buf_params;
    //   LaggedDownsamplingKernelParams lds_params;
    //
    //   // Member of Stage1Tree and Stage2Tree
    //   DedispersionKernelParams kernel_params;

    stage1_dd_buf_params.dtype = config.dtype;
    stage1_dd_buf_params.beams_per_batch = config.beams_per_batch;
    stage1_dd_buf_params.nbuf = stage1_trees.size();

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
	kparams.input_is_ringbuf = false;
	kparams.output_is_ringbuf = true;   // note output_is_ringbuf = true
	kparams.apply_input_residual_lags = false;
	kparams.input_is_downsampled_tree = (st1.ds_level > 0);
	kparams.nelts_per_segment = this->nelts_per_segment;
	kparams.ringbuf_locations = this->stage1_rb_locs.slice(0, pos, pos + nseg);
	kparams.ringbuf_nseg = this->gmem_ringbuf_nseg;
	kparams.validate(false);  // on_gpu=false
	
	stage1_dd_buf_params.buf_rank.push_back(st1.rank0 + st1.rank1);
	stage1_dd_buf_params.buf_ntime.push_back(st1.nt_ds);
	stage1_dd_kernel_params.push_back(kparams);
    }

    stage2_dd_buf_params.dtype = config.dtype;
    stage2_dd_buf_params.beams_per_batch = config.beams_per_batch;
    stage2_dd_buf_params.nbuf = stage2_trees.size();

    for (Stage2Tree &st2: stage2_trees) {
	long pos = st2.base_segment;
	long nseg = st2.segments_per_beam;

	DedispersionKernelParams kparams;
	kparams.dtype = config.dtype;
	kparams.dd_rank = st2.rank1_trigger;
	kparams.amb_rank = st2.rank0;
	kparams.total_beams = config.beams_per_gpu;
	kparams.beams_per_batch = config.beams_per_batch;
	kparams.ntime = st2.nt_ds;
	kparams.input_is_ringbuf = true;   // note input_is_ringbuf = true
	kparams.output_is_ringbuf = false;
	kparams.apply_input_residual_lags = true;
	kparams.input_is_downsampled_tree = (st2.ds_level > 0);
	kparams.nelts_per_segment = this->nelts_per_segment;
	kparams.ringbuf_locations = this->stage2_rb_locs.slice(0, pos, pos + nseg);
	kparams.ringbuf_nseg = this->gmem_ringbuf_nseg;
	kparams.validate(false);  // on_gpu=false

	stage2_dd_buf_params.buf_rank.push_back(st2.rank0 + st2.rank1_trigger);
	stage2_dd_buf_params.buf_ntime.push_back(st2.nt_ds);
	stage2_dd_kernel_params.push_back(kparams);
    }
    
    lds_params.dtype = config.dtype;
    lds_params.small_input_rank = (stage1_trees.size() > 1) ? (stage1_trees.at(1).rank0 + 1) : 0;
    lds_params.large_input_rank = config.tree_rank;
    lds_params.num_downsampling_levels = config.num_downsampling_levels;
    lds_params.total_beams = config.beams_per_gpu;
    lds_params.beams_per_batch = config.beams_per_batch;
    lds_params.ntime = config.time_samples_per_chunk;

    stage1_dd_buf_params.validate();
    stage2_dd_buf_params.validate();
    lds_params.validate();
}


// ------------------------------------------------------------------------------------------------


void DedispersionPlan::print(ostream &os, int indent) const
{
    os << Indent(indent) << "DedispersionConfig" << endl;
    this->config.print(os, indent+4);
    
    print_kv("nelts_per_segment", nelts_per_segment, os, indent);
    print_kv("nbytes_per_segment", nbytes_per_segment, os, indent);

    os << Indent(indent) << "Stage1Trees" << endl;

    for (unsigned int i = 0; i < stage1_trees.size(); i++) {
	const Stage1Tree &st1 = stage1_trees.at(i);
	
	os << Indent(indent+4) << i
	   << ": ds_level=" << st1.ds_level
	   << ", rank0=" << st1.rank0
	   << ", rank1=" << st1.rank1
	   << ", nt_ds=" << st1.nt_ds
	   << endl;
    }
    
    os << Indent(indent) << "Stage2Trees" << endl;

    for (unsigned int i = 0; i < stage2_trees.size(); i++) {
	const Stage2Tree &st2 = stage2_trees.at(i);;
	
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
