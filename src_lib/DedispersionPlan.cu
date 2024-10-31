#include "../include/pirate/DedispersionPlan.hpp"

#include "../include/pirate/constants.hpp"
#include "../include/pirate/internals/utils.hpp"    // bit_reverse_slow(), rb_lag(), rstate_len(), mean_bytes_per_unaligned_chunk()
#include "../include/pirate/internals/inlines.hpp"  // align_up(), pow2(), print_kv(), Indent

using namespace std;
using namespace gputils;

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
    //   - Initialize stage0_trees, stage1_trees.
    //   - Initialize max_n1 (max number of Stage1Trees, per Stage0Tree).

    int max_n1 = 0;
    
    for (int ids = 0; ids < config.num_downsampling_levels; ids++) {
	int st0_rank = ids ? (config.tree_rank - 1) : config.tree_rank;
	int st0_rank0 = st0_rank/2;
	vector<int> trigger_ranks;
	
	for (const DedispersionConfig::EarlyTrigger &et: config.early_triggers) {
	    if (et.ds_level == ids)
		trigger_ranks.push_back(et.tree_rank);
	}

	trigger_ranks.push_back(st0_rank);
	assert(is_sorted(trigger_ranks));

	Stage0Tree st0;
	st0.ds_level = ids;
	st0.rank0 = st0_rank0;
	st0.rank1 = st0_rank - st0.rank0;
	st0.nt_ds = xdiv(config.time_samples_per_chunk, pow2(ids));
	st0.segments_per_beam = pow2(st0_rank) * xdiv(st0.nt_ds, nelts_per_segment);
	st0.base_segment = this->stage0_total_segments_per_beam;

	// FIXME should replace hardcoded 7,8 by something more descriptive
	// (GpuDedispersionKernel::max_rank?)
	int max_rank = ids ? 7 : 8;
	assert((st0.rank0 >= 0) && (st0.rank0 <= max_rank));
	assert(st0.nt_ds > 0);
	
	this->stage0_trees.push_back(st0);
	this->stage0_total_segments_per_beam += st0.segments_per_beam;

	for (int trigger_rank: trigger_ranks) {
	    Stage1Tree st1;
	    st1.ds_level = ids;
	    st1.rank0 = st0.rank0;
	    st1.rank1_ambient = st0.rank1;
	    st1.rank1_trigger = trigger_rank - st1.rank0;
	    st1.nt_ds = st0.nt_ds;
	    st1.segments_per_beam = pow2(trigger_rank) * xdiv(st1.nt_ds, nelts_per_segment);
	    st1.base_segment = this->stage1_total_segments_per_beam;

	    assert((st1.rank1_trigger >= 0) && (st1.rank1_trigger <= 8));
	    assert(st1.rank1_trigger <= st1.rank1_ambient);
		     
	    this->stage1_trees.push_back(st1);
	    this->stage1_total_segments_per_beam += st1.segments_per_beam;
	}

	max_n1 = max(max_n1, int(trigger_ranks.size()));
    }
    
    // Part 2:
    //  - Initialize this->max_clag
    //  - Allocate 'segmap', which maps iseg0 -> (list of (clag,iseg1) pairs).
    //    (This is a temporary object that will be used "locally" in this function.)
    
    int nseg0 = this->stage0_total_segments_per_beam;
    int nseg1 = this->stage1_total_segments_per_beam;

    Array<uint> segmap_n1({nseg0}, af_uhost | af_zero);
    Array<uint> segmap_clag({nseg0,max_n1}, af_uhost | af_zero);
    Array<uint> segmap_iseg1({nseg0,max_n1}, af_uhost | af_zero);
    
    this->max_clag = 0;
    
    for (const Stage1Tree &st1: this->stage1_trees) {
	const Stage0Tree &st0 = this->stage0_trees.at(st1.ds_level);

	// Some truly paranoid asserts.
	assert(st0.nt_ds == st1.nt_ds);
	assert(st0.rank0 == st1.rank0);
	assert(st0.ds_level == st1.ds_level);
	assert(st0.rank1 == st1.rank1_ambient);

	int nchan0 = pow2(st1.rank0);
	int nchan1 = pow2(st1.rank1_trigger);  // not rank1_ambient
	int ns = xdiv(st1.nt_ds, this->nelts_per_segment);
	bool is_downsampled = (st1.ds_level > 0);
	
	for (int i0 = 0; i0 < nchan0; i0++) {
	    for (int i1 = 0; i1 < nchan1; i1++) {
		int lag = rb_lag(i1, i0, st1.rank0, st1.rank1_trigger, is_downsampled);
		int slag = lag / nelts_per_segment;  // round down
		
		for (int s0 = 0; s0 < ns; s0++) {
		    int clag = (s0 + slag) / ns;
		    int s1 = (s0 + slag) - (clag * ns);
		    assert((s1 >= 0) && (s1 < ns));

		    // iseg0 -> (s,i1,i0)
		    int iseg0 = (s0 * pow2(st0.rank1)) + i1;
		    iseg0 = (iseg0 * pow2(st0.rank0)) + i0;
		    iseg0 += st0.base_segment;
		    assert((iseg0 >= 0) && (iseg0 < nseg0));

		    // iseg1 -> (s,i0,i1)
		    int iseg1 = (s1 * nchan0) + i0;
		    iseg1 = (iseg1 * nchan1) + i1;
		    iseg1 += st1.base_segment;
		    assert((iseg1 >= 0) && (iseg1 < nseg1));

		    // Add (clag, iseg1) to segmap[iseg0].
		    int n1 = segmap_n1.data[iseg0]++;
		    assert((n1 >= 0) && (n1 < max_n1));
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
    //  - pseudo-initialize stage0_rb_locs (*)
    //  - pseudo-initialize stage1_rb_locs (*)
    //
    // (*) "Pseudo-initialize" means that we use the following temporary rb_loc layout:
    //
    //   uint rb_seg;     // Segment within (time chunk, beam) pair
    //   uint rb_phase;   // Index of (time chunk, beam) pair, relative to current pair
    //   uint rb_clag;
    //     (fourth 'uint' is unused)

    this->stage0_rb_locs = Array<uint> ({nseg0,4}, af_rhost);
    this->stage1_rb_locs = Array<uint> ({nseg1,4}, af_rhost);

    // Not logically necessary, but enables a real-time consistency check.
    vector<bool> coverage(nseg1, false);
    
    for (int iseg0 = 0; iseg0 < nseg0; iseg0++) {
	int n1 = segmap_n1.data[iseg0];
	assert((n1 >= 0) && (n1 <= max_n1));
	
	int rb_clag = segmap_clag.data[iseg0*max_n1 + n1-1];
	assert((rb_clag >= 0) && (rb_clag <= max_clag));
	
	Ringbuf &rb = gmem_ringbufs.at(rb_clag);
	int rb_seg = rb.nseg_per_beam++;

	uint *rb_loc0 = stage0_rb_locs.data + (4*iseg0);
	rb_loc0[0] = rb_seg;
	rb_loc0[1] = 0;  // rb_phase
	rb_loc0[2] = rb_clag;
	
	for (int i1 = 0; i1 < n1; i1++) {
	    int clag1 = segmap_clag.data[iseg0*max_n1 + i1];
	    int iseg1 = segmap_iseg1.data[iseg0*max_n1 + i1];
	    
	    assert((clag1 >= 0) && (clag1 <= rb_clag));  // segmap_clags must be sorted
	    assert((iseg1 >= 0) && (iseg1 < nseg1));
	    assert(!coverage[iseg1]);
	    coverage[iseg1] = true;

	    int rb_phase = rb.rb_len - (clag1 * BT);
	    assert(rb_phase > 0);
	    
	    uint *rb_loc1 = stage1_rb_locs.data + (4*iseg1);
	    rb_loc1[0] = rb_seg;
	    rb_loc1[1] = rb_phase;
	    rb_loc1[2] = rb_clag;   // not clag1
	}
    }

    for (int iseg1 = 0; iseg1 < nseg1; iseg1++)
	assert(coverage[iseg1]);

    // Part 5:
    //  - initialize this->gmem_ringbuf_nbytes
    //  - initialize Ringbuf::base_segment
    //  - fully initialize stage0_rb_locs (**)
    //  - fully initialize stage1_rb_locs (**)
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
	uint *p = stage0_rb_locs.data + (4*iseg0);
	Ringbuf &rb = gmem_ringbufs.at(p[2]);  // p[2] = clag

	p[0] += rb.base_segment;
	p[2] = rb.rb_len;
	p[3] = rb.nseg_per_beam;
    }
    
    for (int iseg1 = 0; iseg1 < nseg1; iseg1++) {
	uint *p = stage1_rb_locs.data + (4*iseg1);
	Ringbuf &rb = gmem_ringbufs.at(p[2]);  // p[2] = clag

	p[0] += rb.base_segment;
	p[2] = rb.rb_len;
	p[3] = rb.nseg_per_beam;
    }
}


// ------------------------------------------------------------------------------------------------


void DedispersionPlan::print(ostream &os, int indent) const
{
    os << Indent(indent) << "DedispersionConfig" << endl;
    this->config.print(os, indent+4);
    
    print_kv("nelts_per_segment", nelts_per_segment, os, indent);
    print_kv("nbytes_per_segment", nbytes_per_segment, os, indent);

    os << Indent(indent) << "Stage0Trees" << endl;

    for (unsigned int i = 0; i < stage0_trees.size(); i++) {
	const Stage0Tree &st0 = stage0_trees.at(i);
	
	os << Indent(indent+4) << i
	   << ": ds_level=" << st0.ds_level
	   << ", rank0=" << st0.rank0
	   << ", rank1=" << st0.rank1
	   << ", nt_ds=" << st0.nt_ds
	   << endl;
    }
    
    os << Indent(indent) << "Stage1Trees" << endl;

    for (unsigned int i = 0; i < stage1_trees.size(); i++) {
	const Stage1Tree &st1 = stage1_trees.at(i);;
	
	os << Indent(indent+4) << i
	   << ": ds_level=" << st1.ds_level
	   << ", rank0=" << st1.rank0
	   << ", rank1_amb=" << st1.rank1_ambient
	   << ", rank1_tri=" << st1.rank1_trigger
	   << ", nt_ds=" << st1.nt_ds
	   << endl;
    }
}


}  // namespace pirate
