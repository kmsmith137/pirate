#include "../include/pirate/DedispersionPlan.hpp"

#include "../include/pirate/constants.hpp"
#include "../include/pirate/internals/utils.hpp"    // bit_reverse_slow(), rb_lag(), rstate_len()
#include "../include/pirate/internals/inlines.hpp"  // align_up(), pow2(), print_kv(), Indent
#include "../include/pirate/internals/CacheLineRingbuf.hpp"

using namespace std;

namespace pirate {
#if 0
}  // editor auto-indent
#endif


DedispersionPlan::DedispersionPlan(const DedispersionConfig &config_) :
    config(config_)
{
    if (config.planner_verbosity >= 1)
	cout << "DedispersionPlan constructor: start" << endl;
    
    config.validate();

    this->nelts_per_segment = config.get_nelts_per_segment();
    this->uncompressed_dtype_size = config.get_uncompressed_dtype_size();
    this->bytes_per_compressed_segment = config.get_bytes_per_compressed_segment();

    if (config.planner_verbosity >= 1)
	this->print_segment_info(cout, 4);  // indent=4
    
    this->_init_trees();
    this->_init_lags();
    this->_init_rstate_footprints();
    this->_init_ring_buffers();
    
    if (config.planner_verbosity >= 1)
	cout << "DedispersionPlan constructor: done" << endl;
}


void DedispersionPlan::_init_trees()
{
    if (config.planner_verbosity >= 1)
	cout << "DedispersionPlan constructor: creating trees" << endl;
	
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
	st0.num_stage1_trees = trigger_ranks.size();
	st0.stage1_base_tree_index = this->stage1_trees.size();
	st0.segments_per_row = xdiv(st0.nt_ds, nelts_per_segment);
	st0.iobuf_base_segment = this->stage0_iobuf_segments_per_beam;

	int max_rank = ids ? 7 : 8;
	assert((st0.rank0 >= 0) && (st0.rank0 <= max_rank));
	assert(st0.nt_ds > 0);
	
	this->stage0_trees.push_back(st0);
	this->stage0_iobuf_segments_per_beam += pow2(st0_rank) * st0.segments_per_row;

	for (int trigger_rank: trigger_ranks) {
	    Stage1Tree st1;
	    st1.ds_level = ids;
	    st1.rank0 = st0.rank0;
	    st1.rank1_ambient = st0.rank1;
	    st1.rank1_trigger = trigger_rank - st1.rank0;
	    st1.nt_ds = st0.nt_ds;
	    st1.segments_per_row = xdiv(st1.nt_ds, nelts_per_segment);
	    st1.iobuf_base_segment = this->stage1_iobuf_segments_per_beam;
	    st1.segment_lags = gputils::Array<int> ({pow2(trigger_rank)}, gputils::af_rhost);
	    st1.residual_lags = gputils::Array<int> ({pow2(trigger_rank)}, gputils::af_rhost);

	    assert((st1.rank1_trigger >= 0) && (st1.rank1_trigger <= 8));
	    assert(st1.rank1_trigger <= st1.rank1_ambient);

	    // The 'segment_lags' and 'residual_lags' will be initialized later, in DedispersionPlan::_init_lags().
	    for (int i = 0; i < pow2(trigger_rank); i++) {
		st1.segment_lags.data[i] = -1;
		st1.residual_lags.data[i] = -1;
	    }
		     
	    this->stage1_trees.push_back(st1);
	    this->stage1_iobuf_segments_per_beam += pow2(trigger_rank) * st1.segments_per_row;
	}
    }
    
    int active_beams = config.num_active_batches * config.beams_per_batch;
    this->gmem_nbytes_stage0_iobufs = active_beams * stage0_iobuf_segments_per_beam * constants::bytes_per_segment;
    this->gmem_nbytes_stage1_iobufs = active_beams * stage1_iobuf_segments_per_beam * constants::bytes_per_segment;

    this->gmem_nbytes_tot += this->gmem_nbytes_stage0_iobufs;
    this->gmem_nbytes_tot += this->gmem_nbytes_stage1_iobufs;
    
    if (config.planner_verbosity >= 1)
	this->print_trees(cout, 4);  // indent=4
}


void DedispersionPlan::_init_lags()
{
    if (config.planner_verbosity >= 1)
	cout << "DedispersionPlan constructor: initializing lags" << endl;

    for (Stage1Tree &st1: this->stage1_trees) {
	int rank0 = st1.rank0;
	int rank1 = st1.rank1_trigger;
	bool is_downsampled = (st1.ds_level > 0);

	for (ssize_t i = 0; i < pow2(rank1); i++) {       // pow2(rank1) in outer loop
	    for (ssize_t j = 0; j < pow2(rank0); j++) {   // pow2(rank0) in inner loop
		int row = i * pow2(rank0) + j;
		int lag = rb_lag(i, j, rank0, rank1, is_downsampled);
		
		// Split the lag into 'segment_lag' and 'residual_lag'.
		st1.segment_lags.at({row}) = lag / nelts_per_segment;   // round down
		st1.residual_lags.at({row}) = lag % nelts_per_segment;
	    }
	}
    }
}


void DedispersionPlan::_init_rstate_footprints()
{
    if (config.planner_verbosity >= 1)
	cout << "DedispersionPlan constructor: initializing rstate footprints" << endl;
    
    for (Stage0Tree &st0: this->stage0_trees) {
	// FIXME in the future, I'll probably bookkeep rstate_ds_len separately (rather than including
	// it in the Stage0Tree) since it will correspond to a separate GPU kernel.
	ssize_t nelts_per_small_tree = rstate_len(st0.rank0) + (st0.ds_level ? rstate_ds_len(st0.rank0) : 0);
	
	st0.rstate_nbytes_per_beam = nelts_per_small_tree * pow2(st0.rank1) * uncompressed_dtype_size;
	st0.rstate_nbytes_per_beam = align_up(st0.rstate_nbytes_per_beam, constants::bytes_per_gpu_cache_line);
	this->gmem_nbytes_stage0_rstate += config.beams_per_gpu * st0.rstate_nbytes_per_beam;
    }

    for (Stage1Tree &st1: this->stage1_trees) {
	ssize_t nelts_per_beam = pow2(st1.rank0) * rstate_len(st1.rank1_trigger);
	
	for (int i = 0; i < st1.residual_lags.size; i++) {
	    int rlag = st1.residual_lags.at({i});
	    assert(rlag >= 0);
	    nelts_per_beam += rlag;
	}
	
	st1.rstate_nbytes_per_beam = nelts_per_beam * uncompressed_dtype_size;
	st1.rstate_nbytes_per_beam = align_up(st1.rstate_nbytes_per_beam, constants::bytes_per_gpu_cache_line);
	this->gmem_nbytes_stage1_rstate += config.beams_per_gpu * st1.rstate_nbytes_per_beam;
    }
    
    this->gmem_nbytes_tot += this->gmem_nbytes_stage0_rstate;
    this->gmem_nbytes_tot += this->gmem_nbytes_stage1_rstate;
}


void DedispersionPlan::_init_ring_buffers()
{
    if (config.planner_verbosity >= 1)
	cout << "DedispersionPlan constructor: creating CacheLineRingbuf" << endl;
    
    CacheLineRingbuf::ConstructorArgs cargs;
    cargs.num_src_segments = this->stage0_iobuf_segments_per_beam;
    cargs.num_dst_segments = this->stage1_iobuf_segments_per_beam;
    cargs.gmem_nbytes_used_so_far = align_up(this->gmem_nbytes_tot, constants::bytes_per_gpu_cache_line);
    
    this->cache_line_ringbuf = make_shared<CacheLineRingbuf> (this->config, cargs);
       
    // The purpose of the nested loop below is to loop over "LaggedCacheLines",
    // representing one source segment, and one or more destination (lagged)
    // segments.
    
    for (const Stage0Tree &st0: stage0_trees) {
	// For iterating later over Stage1Trees which are associated with 'st0'.
	auto st1_begin = stage1_trees.begin() + st0.stage1_base_tree_index;
	auto st1_end = st1_begin + st0.num_stage1_trees;
	int nrows0 = pow2(st0.rank0 + st0.rank1);

	for (int row = 0; row < nrows0; row++) {
	    for (int s0 = 0; s0 < st0.segments_per_row; s0++) {
		int src_segment = st0.iobuf_base_segment + (row * st0.segments_per_row) + s0;

		// The nested loops so far have determined a source segment,
		// parameterized by a triple (st0,row,s0) and identified by
		// a segment ID 'src_segment'. It remains to loop over
		// destination (lagged) segments.
		
		// Create a LaggedCacheLine, which represents one source segment,
		// and one or more destination segments.
		    
		LaggedCacheLine cl(src_segment);

		for (auto p = st1_begin; p != st1_end; p++) {
		    Stage1Tree &st1 = *p;
		    assert(st1.rank0 == st0.rank0);
		    assert(st1.rank1_trigger <= st0.rank1);
		    assert(st1.nt_ds == st0.nt_ds);

		    int nrows1 = pow2(st1.rank0 + st1.rank1_trigger);

		    if (row >= nrows1)
			continue;

		    int slag = st1.segment_lags.at({row});

		    // Convert the segment lag 'slag' into a pair (clag, s1), where 'clag' is a
		    // chunk count, and (0 <= s1 < segments_per_row) is a segment.

		    int clag = (s0 + slag) / st1.segments_per_row;   // round down
		    int s1 = (s0 + slag) - (clag * st1.segments_per_row);
		    assert((s1 >= 0) && (s1 < st1.segments_per_row));

		    // Add destination segment to the LaggedCacheLine
			
		    int dst_segment = st1.iobuf_base_segment + (row * st1.segments_per_row) + s1;
		    cl.add_dst(clag, dst_segment);
		}

		// The LaggedCacheLine is complete.
		
		if (config.bloat_dedispersion_plan)
		    this->lagged_cache_lines.push_back(cl);

		this->cache_line_ringbuf->add(cl);
	    }
	}
    }
    
    if (config.planner_verbosity >= 1)
	cout << "DedispersionPlan constructor: finalizing CacheLineRingbuf" << endl;
    
    cache_line_ringbuf->finalize();

    if (config.planner_verbosity >= 2)
	this->cache_line_ringbuf->print(cout, 4);  // indent=4

    // This boilerplate is a little awkward, but it didn't seem worth the trouble of defining a new helper class.

    this->gmem_nbytes_staging_buf = cache_line_ringbuf->gmem_nbytes_staging_buf;
    this->gmem_nbytes_ringbuf = cache_line_ringbuf->gmem_nbytes_ringbuf;
    this->hmem_nbytes_ringbuf = cache_line_ringbuf->hmem_nbytes_ringbuf;
    this->pcie_nbytes_per_chunk = cache_line_ringbuf->pcie_nbytes_per_chunk;
    this->pcie_memcopies_per_chunk = cache_line_ringbuf->pcie_memcopies_per_chunk;
    
    this->gmem_nbytes_tot += this->gmem_nbytes_staging_buf;
    this->gmem_nbytes_tot += this->gmem_nbytes_ringbuf;

    if (config.planner_verbosity >= 1)
	this->print_footprints(cout, 4);  // indent=4
}


// ------------------------------------------------------------------------------------------------


void DedispersionPlan::print(ostream &os, int indent) const
{
    os << Indent(indent) << "DedispersionConfig" << endl;
    this->config.print(os, indent+4);

    this->print_segment_info(os, indent);
    this->print_trees(os, indent);
    this->cache_line_ringbuf->print(os, indent);
    this->print_footprints(os, indent);
}


void DedispersionPlan::print_segment_info(ostream &os, int indent) const
{
    print_kv("nelts_per_segment", nelts_per_segment, os, indent);
    print_kv("uncompressed_dtype_size", uncompressed_dtype_size, os, indent);
    print_kv("bytes_per_compressed_segment", bytes_per_compressed_segment, os, indent);
    print_kv("bytes_per_segment", constants::bytes_per_segment, os, indent);
}


void DedispersionPlan::print_trees(ostream &os, int indent) const
{
    os << Indent(indent) << "Stage0Trees" << endl;

    for (unsigned int i = 0; i < stage0_trees.size(); i++) {
	const Stage0Tree &st0 = stage0_trees.at(i);
	
	os << Indent(indent+4) << i
	   << ": ds_level=" << st0.ds_level
	   << ", rank0=" << st0.rank0
	   << ", rank1=" << st0.rank1
	   << ", nt_ds=" << st0.nt_ds
	   << ", nst1=" << st0.num_stage1_trees
	   << ", st1_base=" << st0.stage1_base_tree_index
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


void DedispersionPlan::print_footprints(ostream &os, int indent) const
{
    // Usage reminder: print_kv(key, val, os, indent, units=nullptr)
    print_kv("gmem_nbytes_tot", gputils::nbytes_to_str(this->gmem_nbytes_tot), os, indent);
    print_kv("gmem_nbytes_stage0_rstate", gputils::nbytes_to_str(this->gmem_nbytes_stage0_rstate), os, indent);
    print_kv("gmem_nbytes_stage1_rstate", gputils::nbytes_to_str(this->gmem_nbytes_stage1_rstate), os, indent);
    print_kv("gmem_nbytes_stage0_iobufs", gputils::nbytes_to_str(this->gmem_nbytes_stage0_iobufs), os, indent);
    print_kv("gmem_nbytes_stage1_iobufs", gputils::nbytes_to_str(this->gmem_nbytes_stage1_iobufs), os, indent);
    print_kv("gmem_nbytes_staging_buf", gputils::nbytes_to_str(this->gmem_nbytes_staging_buf), os, indent);
    print_kv("gmem_nbytes_ringbuf", gputils::nbytes_to_str(this->gmem_nbytes_ringbuf), os, indent);
    print_kv("hmem_nbytes_ringbuf", gputils::nbytes_to_str(this->hmem_nbytes_ringbuf), os, indent, " per GPU");

    // FIXME PCI-E bandwidth reported in GB/chunk. Should be GB/s, but DedispersionPlan is
    // currently unaware of the time sample length in seconds.

    print_kv("pcie_nbytes_per_chunk", 1.0e-9 * this->pcie_nbytes_per_chunk,
	     os, indent, "GB/chunk (not GB/s!), per-GPU each way");

    print_kv("pcie_memcopies_per_chunk", this->pcie_memcopies_per_chunk,
	     os, indent, "copies/chunk (not copies/s!), per-GPU each way");
}


}  // namespace pirate
