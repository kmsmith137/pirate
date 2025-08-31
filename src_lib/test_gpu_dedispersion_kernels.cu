#include "../include/pirate/DedispersionKernel.hpp"
#include "../include/pirate/constants.hpp"  // constants::bytes_per_gpu_cache_line
#include "../include/pirate/inlines.hpp"    // pow2()
#include "../include/pirate/utils.hpp"      // integer_log2()

#include <ksgpu/Array.hpp>
#include <ksgpu/cuda_utils.hpp>
#include <ksgpu/rand_utils.hpp>    // rand_int()
#include <ksgpu/test_utils.hpp>    // make_random_strides()

using namespace std;
using namespace ksgpu;

namespace pirate {
#if 0
}  // editor auto-indent
#endif


struct DedispTestInstance
{
    // Reminder: includes 'dtype', 'input_is_ringbuf', 'output_is_ringbuf'.
    DedispersionKernelParams params;

    long nchunks = 0;

    // Can only be 'true' if params.input_is_ringbuf == params.output_is_ringbuf == false.
    bool in_place = false;
    
    // Ring buffer (only used if params.input_is_ringbuf || params.output_is_ringbuf)
    long rb_nzones = 0;
    vector<long> rb_zone_len;   // length rb_nzones
    vector<long> rb_zone_nseg;  // length rb_nzones
    vector<long> rb_zone_base_seg;  // length rb_nzones
    
    // Strides for input/output arrays.
    // Reminder: arrays have shape (params.beams_per_batch, pow2(params.amb_rank), pow2(params.dd_rank), params.ntime).
    vector<long> gpu_istrides;
    vector<long> gpu_ostrides;
    vector<long> cpu_istrides;
    vector<long> cpu_ostrides;

    // If one_hot==true, then input array will be "one-hot" rather than random.
    bool one_hot = false;
    long one_hot_ichunk = 0;    // 0 <= ichunk < nchunks
    long one_hot_ibeam = 0;     // 0 <= ibeam < params.total_beams
    long one_hot_iamb = 0;      // 0 <= iamb < pow2(params.amb_rank)
    long one_hot_iact = 0;      // 0 <= iact < pow2(params.dd_rank)
    long one_hot_itime = 0;     // 0 <= itime < params.ntime

    
    void randomize()
    {
	const long max_nelts = 100 * 1000 * 1000;
	auto k = GpuDedispersionKernel::get_random_registry_key();
	auto v = GpuDedispersionKernel::query_registry(k);
	
	params.dtype = k.dtype;
	params.nspec = k.nspec;
	params.dd_rank = k.rank;
	params.input_is_ringbuf = k.input_is_ringbuf;
	params.output_is_ringbuf = k.output_is_ringbuf;
	params.apply_input_residual_lags = k.apply_input_residual_lags;
	params.input_is_downsampled_tree = (rand_uniform() < 0.5);
	params.nt_per_segment = v.nt_per_segment;
	
	this->in_place = !params.input_is_ringbuf && !params.output_is_ringbuf && (rand_uniform() < 0.5);

	long nchan = pow2(params.dd_rank);
	params.ntime = rand_int(1, 2*nchan + 2*params.nt_per_segment);
	params.ntime = align_up(params.ntime, params.nt_per_segment);

	long cmax = (10*nchan + 10*params.ntime) / params.ntime;
	this->nchunks = rand_int(1, cmax+1);

	// pow2(amb_rank), (total_beams/beams_per_batch), beams_per_batch
	long pmax = max_nelts / (pow2(params.dd_rank) * params.ntime * nchunks);
	pmax = max(pmax, 4L);
	pmax = min(pmax, 42L);
	
	auto s = ksgpu::random_integers_with_bounded_product(4, pmax);
	params.amb_rank = int(log2(s[0]) + 0.99999);  // round up
	params.total_beams = s[1] * s[2];
	params.beams_per_batch = s[2];

	if (params.input_is_ringbuf || params.output_is_ringbuf)
	    randomize_ringbuf();

	randomize_strides();
    }


    void randomize_ringbuf()
    {
	// Now a sequence of steps to initialize:
	//
	//   params.ringbuf_locations;
        //   params.ringbuf_nseg;
	//   this->rb_nzones;
	//   this->rb_zone_len;
	//   this->rb_zone_nseg;
	//   this->rb_zone_base_seg;

	this->rb_nzones = rand_int(1, 11);
	this->rb_zone_len = vector<long> (rb_nzones, 0);
	this->rb_zone_nseg = vector<long> (rb_nzones, 0);
	this->rb_zone_base_seg = vector<long> (rb_nzones, 0);
	
	for (long z = 0; z < rb_nzones; z++) {
	    long lmin = params.beams_per_batch;   // assumes that GPU does not run multiple batches in parallel
	    long lmax = params.total_beams * nchunks;
	    rb_zone_len[z] = rand_int(lmin, lmax+1);
	}

	long nchan = pow2(params.dd_rank);
	long nseg = nchan * pow2(params.amb_rank) * xdiv(params.ntime, params.nt_per_segment);
	vector<pair<long,long>> pairs(nseg);   // map segment -> (rb_zone, rb_seg)
	
	for (long iseg = 0; iseg < nseg; iseg++) {
	    long z = rand_int(0, rb_nzones);
	    long n = rb_zone_nseg[z]++;
	    pairs[iseg] = {z,n};
	}
	
	ksgpu::randomly_permute(pairs);

	params.ringbuf_nseg = 0;
	for (long z = 0; z < rb_nzones; z++) {
	    this->rb_zone_base_seg[z] = params.ringbuf_nseg;
	    params.ringbuf_nseg += rb_zone_len[z] * rb_zone_nseg[z];
	}
	
	params.ringbuf_locations = Array<uint> ({nseg,4}, af_rhost | af_zero);
	uint *rb_loc = params.ringbuf_locations.data;
	
	for (long iseg = 0; iseg < nseg; iseg++) {
	    long z = pairs[iseg].first;
	    long n = pairs[iseg].second;
	    long s = rb_zone_nseg[z];
	    long l = rb_zone_len[z];
	    
	    rb_loc[4*iseg] = rb_zone_base_seg[z] + n;    // rb_offset (in segments, not bytes)
	    rb_loc[4*iseg+1] = rand_int(0, 2*l+1);        // rb_phase
	    rb_loc[4*iseg+2] = l;                         // rb_len
	    rb_loc[4*iseg+3] = s;                         // rb_nseg
	}
    }


    void randomize_strides()
    {
	xassert(params.nspec == 1);  // FIXME for now
	
	// Dedispersion buffer strides (note that ringbufs are always contiguous).
	vector<long> small_shape = { params.beams_per_batch, pow2(params.amb_rank), pow2(params.dd_rank), params.ntime };
	this->cpu_istrides = ksgpu::make_random_strides(small_shape, 1, params.nt_per_segment);
	this->gpu_istrides = ksgpu::make_random_strides(small_shape, 1, params.nt_per_segment);
	this->cpu_ostrides = in_place ? cpu_istrides : ksgpu::make_random_strides(small_shape, 1, params.nt_per_segment);
	this->gpu_ostrides = in_place ? gpu_istrides : ksgpu::make_random_strides(small_shape, 1, params.nt_per_segment);
    }


    void set_contiguous_strides()
    {
	xassert(params.nspec == 1);  // FIXME for now
	
	// Dedispersion buffer strides (note that ringbufs are always contiguous).
	vector<long> small_shape = { params.beams_per_batch, pow2(params.amb_rank), pow2(params.dd_rank), params.ntime };
	vector<long> strides = ksgpu::make_contiguous_strides(small_shape);
	
	this->cpu_istrides = strides;
	this->gpu_istrides = strides;
	this->cpu_ostrides = strides;
	this->gpu_ostrides = strides;
    }
};


// Another helper class.
struct TestArrays
{
    DedispTestInstance tp;
    long nbatches;
    
    Array<void> big_inbuf;      // either a "big" ddbuf (nchunks), or a ringbuf
    Array<void> big_outbuf;     // either a "big" ddbuf (nchunks), or a ringbuf
    Array<void> active_inbuf;   // either a "small" ddbuf (1 chunk), or a reference to 'big_inbuf'
    Array<void> active_outbuf;  // either a "small" ddbuf (1 chunk), or a reference to 'big_outbuf'
    
    TestArrays(const DedispTestInstance &tp_, const Dtype &dtype, bool on_gpu) :
	tp(tp_),
	nbatches(xdiv(tp.params.total_beams, tp.params.beams_per_batch))
    {
	const DedispersionKernelParams &p = tp.params;
	int aflags = (on_gpu ? af_gpu : af_rhost) | af_zero;
	xassert(p.nspec == 1);   // FIXME for now
	
	vector<long> rb_shape = { p.ringbuf_nseg * p.nt_per_segment * p.nspec };
	vector<long> big_dshape = { p.total_beams, pow2(p.amb_rank), pow2(p.dd_rank), tp.nchunks * p.ntime };
	vector<long> big_ishape = p.input_is_ringbuf ? rb_shape : big_dshape;
	vector<long> big_oshape = p.output_is_ringbuf ? rb_shape : big_dshape;
	vector<long> chunk_dshape = { p.beams_per_batch, pow2(p.amb_rank), pow2(p.dd_rank), p.ntime };
	vector<long> chunk_istrides = on_gpu ? tp.gpu_istrides : tp.cpu_istrides;
	vector<long> chunk_ostrides = on_gpu ? tp.gpu_ostrides : tp.cpu_ostrides;

	this->big_inbuf = Array<void> (dtype, big_ishape, aflags);
	this->big_outbuf = Array<void> (dtype, big_oshape, aflags);

	if (p.input_is_ringbuf)
	    this->active_inbuf = big_inbuf;
	else
	    this->active_inbuf = Array<void> (dtype, chunk_dshape, chunk_istrides, aflags);
	
	if (tp.in_place)
	    this->active_outbuf = active_inbuf;
	else if (p.output_is_ringbuf)
	    this->active_outbuf = big_outbuf;
	else
	    this->active_outbuf = Array<void> (dtype, chunk_dshape, chunk_ostrides, aflags);
    }

    void copy_input(long ichunk, long ibatch)
    {
	const DedispersionKernelParams &p = tp.params;
	xassert((ichunk >= 0) && (ichunk < tp.nchunks));
	xassert((ibatch >= 0) && (ibatch < nbatches));
	
	if (!p.input_is_ringbuf) {
	    Array<void> s = big_inbuf.slice(0, ibatch * p.beams_per_batch, (ibatch+1) * p.beams_per_batch);
	    s = s.slice(3, ichunk * p.ntime, (ichunk+1) * p.ntime);
	    active_inbuf.fill(s);   // (slice of big_inbuf) -> (active_inbuf)
	}
    }

    void copy_output(long ichunk, long ibatch)
    {
	const DedispersionKernelParams &p = tp.params;
	xassert((ichunk >= 0) && (ichunk < tp.nchunks));
	xassert((ibatch >= 0) && (ibatch < nbatches));

	if (!p.output_is_ringbuf) {
	    Array<void> s = big_outbuf.slice(0, ibatch * p.beams_per_batch, (ibatch+1) * p.beams_per_batch);
	    s = s.slice(3, ichunk * p.ntime, (ichunk+1) * p.ntime);
	    s.fill(active_outbuf);  // (active_outbuf) -> (slice of big_outbuf)
	}
    }
};



static void run_test(const DedispTestInstance &tp)
{
    const DedispersionKernelParams &p = tp.params;
    
    long nbatches = xdiv(p.total_beams, p.beams_per_batch);
    xassert(p.nspec == 1);  // FIXME for now

    cout << "\nTest GpuDedispersionKernel\n"
	 << "    ti.params.dtype = " << p.dtype << ";\n"
	 << "    ti.params.dd_rank = " << p.dd_rank << ";\n"
	 << "    ti.params.amb_rank = " << p.amb_rank << ";\n"
	 << "    ti.params.total_beams = " << p.total_beams << ";\n"
	 << "    ti.params.beams_per_batch = " << p.beams_per_batch << ";\n"
	 << "    ti.params.ntime = " << p.ntime << ";\n"
	 << "    ti.params.nspec = " << p.nspec << ";\n"
	 << "    ti.params.input_is_ringbuf = " << (p.input_is_ringbuf ? "true" : "false")  << ";\n"
	 << "    ti.params.output_is_ringbuf = " << (p.output_is_ringbuf ? "true" : "false")  << ";\n"
	 << "    ti.params.apply_input_residual_lags = " << (p.apply_input_residual_lags ? "true" : "false")  << ";\n"
	 << "    ti.params.input_is_downsampled_tree = " << (p.input_is_downsampled_tree ? "true" : "false")  << ";\n"
	 << "    ti.params.nt_per_segment = " << p.nt_per_segment << ";\n"
	 << "    ti.nchunks = " << tp.nchunks << ";\n"
	 << "    ti.in_place = " << (tp.in_place ? "true" : "false") << ";\n"
	 << "    ti.gpu_istrides = " << ksgpu::tuple_str(tp.gpu_istrides) << ";\n"
	 << "    ti.gpu_ostrides = " << ksgpu::tuple_str(tp.gpu_ostrides) << ";\n"
	 << "    ti.cpu_istrides = " << ksgpu::tuple_str(tp.cpu_istrides) << ";\n"
	 << "    ti.cpu_ostrides = " << ksgpu::tuple_str(tp.cpu_ostrides) << ";\n"
	 << endl;

    if (tp.one_hot) {
	cout << "    ti.one_hot = true\n"
	     << "    ti.one_hot_ichunk = " << tp.one_hot_ichunk << "\n"
	     << "    ti.one_hot_ibeam = " << tp.one_hot_ibeam << "\n"
	     << "    ti.one_hot_iamb = " << tp.one_hot_iamb << "\n"
	     << "    ti.one_hot_iact = " << tp.one_hot_iact << "\n"
	     << "    ti.one_hot_itime = " << tp.one_hot_itime
	     << endl;
    }

    
    shared_ptr<ReferenceDedispersionKernel> ref_kernel = make_shared<ReferenceDedispersionKernel> (p);
    shared_ptr<GpuDedispersionKernel> gpu_kernel;

    gpu_kernel = make_shared<GpuDedispersionKernel> (p);
    gpu_kernel->allocate();

    TestArrays cpu_arrs(tp, Dtype::native<float>(), false);  // on_gpu=false
    TestArrays gpu_arrs(tp, tp.params.dtype, true);          // on_gpu=true

    if (!tp.one_hot) {
	// Randomize (cpu_arrs.big_inbuf).
	// FIXME ksgpu should contain a function to randomize an array.
	Array<void> &a = cpu_arrs.big_inbuf;
	xassert(a.is_fully_contiguous());
	ksgpu::_randomize(a.dtype, a.data, a.size);
    }
    else {
	xassert(!p.input_is_ringbuf);
	xassert((tp.one_hot_ichunk >= 0) && (tp.one_hot_ichunk < tp.nchunks));
	xassert((tp.one_hot_ibeam >= 0) && (tp.one_hot_ibeam < p.total_beams));
	xassert((tp.one_hot_iamb >= 0) && (tp.one_hot_iamb < pow2(p.amb_rank)));
	xassert((tp.one_hot_iact >= 0) && (tp.one_hot_iact < pow2(p.dd_rank)));
	xassert((tp.one_hot_itime >= 0) && (tp.one_hot_itime < p.ntime));

	long b = tp.one_hot_ibeam;
	long a = tp.one_hot_iamb;
	long d = tp.one_hot_iact;
	long t = tp.one_hot_ichunk * p.ntime + tp.one_hot_itime;

	Array<float> dst = cpu_arrs.big_inbuf.cast<float>();
	dst.at({b,a,d,t}) = 1.0f;
    }

    // Copy (cpu_arrs.big_inbuf) -> (gpu_arrs.big_inbuf), converting dtype if necessary.
    // FIXME ksgpu should contain a function for this.
    Array<void> src = cpu_arrs.big_inbuf;
    if (src.dtype != tp.params.dtype)
	src = src.convert(tp.params.dtype);
    gpu_arrs.big_inbuf.fill(src);
    src = Array<void>();  // free memory
    
    for (long ichunk = 0; ichunk < tp.nchunks; ichunk++) {
	for (long ibatch = 0; ibatch < nbatches; ibatch++) {
	    // Reference dedispersion.
	    cpu_arrs.copy_input(ichunk, ibatch);
	    ref_kernel->apply(cpu_arrs.active_inbuf, cpu_arrs.active_outbuf, ibatch, ichunk);
	    cpu_arrs.copy_output(ichunk, ibatch);
	    
	    // GPU dedipersion.
	    gpu_arrs.copy_input(ichunk, ibatch);
	    gpu_kernel->launch(gpu_arrs.active_inbuf, gpu_arrs.active_outbuf, ibatch, ichunk, nullptr);  // stream=nullptr
	    gpu_arrs.copy_output(ichunk, ibatch);
	}
    }
    
    // FIXME revisit epsilon if we change the normalization of the dedispersion transform.
    double epsrel = 3 * tp.params.dtype.precision();
    double epsabs = 3 * tp.params.dtype.precision() * pow(1.414, p.dd_rank);

    if (p.output_is_ringbuf)
	ksgpu::assert_arrays_equal(cpu_arrs.big_outbuf, gpu_arrs.big_outbuf, "cpu", "gpu", {"i"}, epsabs, epsrel);
    else
	ksgpu::assert_arrays_equal(cpu_arrs.big_outbuf, gpu_arrs.big_outbuf, "cpu", "gpu", {"beam","amb","dmbr","time"}, epsabs, epsrel);
}


void test_gpu_dedispersion_kernels()
{
#if 0
    // Debug
    DedispTestInstance ti;
    ti.params.dtype = Dtype::native<float>();
    ti.params.dd_rank = 3;
    ti.params.amb_rank = 1;
    ti.params.total_beams = 2;
    ti.params.beams_per_batch = 2; 
    ti.params.ntime = 128;
    ti.params.nspec = 1;
    ti.params.input_is_ringbuf = false;
    ti.params.output_is_ringbuf = true;
    ti.params.apply_input_residual_lags = false;
    ti.params.input_is_downsampled_tree = true;
    ti.params.nt_per_segment = 32;
    ti.nchunks = 1;
    ti.in_place = false;
    ti.randomize_ringbuf();
    ti.set_contiguous_strides();
    // ti.one_hot = true;
    // ti.one_hot_iact = 0;   // rand_int(0, pow2(ti.params.dd_rank));
    // ti.one_hot_itime = 0;  // rand_int(0, ti.params.ntime);
    run_test(ti);
#endif    

#if 1
    DedispTestInstance ti;
    ti.randomize();
    run_test(ti);
#endif
}


}  // namespace pirate
