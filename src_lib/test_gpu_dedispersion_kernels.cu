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


// FIXME buffer managment (in particular the _setup_io_arrays() function) in this
// code looks dubious -- revisit at some point.

struct TestInstanceDK
{
    // Reminder: includes 'dtype', 'input_is_ringbuf', 'output_is_ringbuf'.
    DedispersionKernelParams params;
    bool new_code = false;

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

    
    void randomize_old()
    {
	const long max_nelts = 100 * 1000 * 1000;
	this->new_code = false;

	// Initialize:
	//  params.input_is_ringbuf
	//  params.output_is_ringbuf
	//  params.apply_input_residual_lags
	//  this->in_place

	if (rand_uniform() < 0.33) {
	     params.input_is_ringbuf = true;
	     params.output_is_ringbuf = false;
	     params.apply_input_residual_lags = true;
	     this->in_place = false;
	}
	else if (rand_uniform() < 0.67) {
	    params.input_is_ringbuf = false;
	    params.output_is_ringbuf = true;
	    params.apply_input_residual_lags = false;
	    this->in_place = false;
	}
	else {
	    params.input_is_ringbuf = false;
	    params.output_is_ringbuf = false;
	    params.apply_input_residual_lags = (rand_uniform() < 0.5);
	    this->in_place = (rand_uniform() < 0.5);
	}
	
	// Initialize:
	//   params.dtype
	//   params.input_is_downsampled_tree
	//   params.nelts_per_segment
	//   params.dd_rank
	//   params.amb_rank
	//   params.ntime
	//   params.total_beams
	//   params.beams_per_batch
	//   this->nchunks
	
	bool is_float32 = (rand_uniform() < 0.5);
	params.dtype = is_float32 ? Dtype::native<float>() : Dtype::native<__half>();
	params.input_is_downsampled_tree = (rand_uniform() < 0.5);

	params.nelts_per_segment = is_float32 ? 32 : 64;
	params.dd_rank = rand_int(1, 9);

	long nchan = pow2(params.dd_rank);
	params.ntime = rand_int(1, 2*nchan + 2*params.nelts_per_segment);
	params.ntime = align_up(params.ntime, params.nelts_per_segment);

	long cmax = (10*nchan + 10*params.ntime) / params.ntime;
	nchunks = rand_int(1, cmax+1);
	
	// pow2(amb_rank), (total_beams/beams_per_batch), beams_per_batch
	long pmax = max_nelts / (pow2(params.dd_rank) * params.ntime * nchunks);
	pmax = max(pmax, 4L);
	pmax = min(pmax, 42L);

	auto v = ksgpu::random_integers_with_bounded_product(4, pmax);
	params.amb_rank = int(log2(v[0]) + 0.99999);  // round up
	params.total_beams = v[1] * v[2];
	params.beams_per_batch = v[2];

	if (params.input_is_ringbuf || params.output_is_ringbuf)
	    randomize_ringbuf();

	randomize_strides();
    }

    
    void randomize_new()
    {
	const long max_nelts = 100 * 1000 * 1000;
	auto k = NewGpuDedispersionKernel::get_random_registry_key();
	
	params.dtype = k.dtype;
	params.dd_rank = k.rank;
	params.input_is_ringbuf = k.input_is_ringbuf;
	params.output_is_ringbuf = k.output_is_ringbuf;
	params.apply_input_residual_lags = k.apply_input_residual_lags;
	params.input_is_downsampled_tree = (rand_uniform() < 0.5);
	params.nelts_per_segment = xdiv(8 * constants::bytes_per_gpu_cache_line, params.dtype.nbits);
	
	this->new_code = true;
	this->in_place = !params.input_is_ringbuf && !params.output_is_ringbuf && (rand_uniform() < 0.5);

	long nchan = pow2(params.dd_rank);
	params.ntime = rand_int(1, 2*nchan + 2*params.nelts_per_segment);
	params.ntime = align_up(params.ntime, params.nelts_per_segment);

	long cmax = (10*nchan + 10*params.ntime) / params.ntime;
	this->nchunks = rand_int(1, cmax+1);

	// pow2(amb_rank), (total_beams/beams_per_batch), beams_per_batch
	long pmax = max_nelts / (pow2(params.dd_rank) * params.ntime * nchunks);
	pmax = max(pmax, 4L);
	pmax = min(pmax, 42L);
	
	auto v = ksgpu::random_integers_with_bounded_product(4, pmax);
	params.amb_rank = int(log2(v[0]) + 0.99999);  // round up
	params.total_beams = v[1] * v[2];
	params.beams_per_batch = v[2];

	if (params.input_is_ringbuf || params.output_is_ringbuf)
	    randomize_ringbuf();

	// Dedispersion buffer strides (note that ringbufs are always contiguous).
	vector<long> small_shape = { params.beams_per_batch, pow2(params.amb_rank), pow2(params.dd_rank), params.ntime };
	this->cpu_istrides = ksgpu::make_random_strides(small_shape, 1, params.nelts_per_segment);
	this->gpu_istrides = ksgpu::make_random_strides(small_shape, 1, params.nelts_per_segment);
	this->cpu_ostrides = in_place ? cpu_istrides : ksgpu::make_random_strides(small_shape, 1, params.nelts_per_segment);
	this->gpu_ostrides = in_place ? gpu_istrides : ksgpu::make_random_strides(small_shape, 1, params.nelts_per_segment);
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
	long nseg = nchan * pow2(params.amb_rank) * xdiv(params.ntime, params.nelts_per_segment);
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
	// Dedispersion buffer strides (note that ringbufs are always contiguous).
	vector<long> small_shape = { params.beams_per_batch, pow2(params.amb_rank), pow2(params.dd_rank), params.ntime };
	this->cpu_istrides = ksgpu::make_random_strides(small_shape, 1, params.nelts_per_segment);
	this->gpu_istrides = ksgpu::make_random_strides(small_shape, 1, params.nelts_per_segment);
	this->cpu_ostrides = in_place ? cpu_istrides : ksgpu::make_random_strides(small_shape, 1, params.nelts_per_segment);
	this->gpu_ostrides = in_place ? gpu_istrides : ksgpu::make_random_strides(small_shape, 1, params.nelts_per_segment);
    }


    void set_contiguous_strides()
    {
	// Dedispersion buffer strides (note that ringbufs are always contiguous).
	vector<long> small_shape = { params.beams_per_batch, pow2(params.amb_rank), pow2(params.dd_rank), params.ntime };
	vector<long> strides = ksgpu::make_contiguous_strides(small_shape);
	
	this->cpu_istrides = strides;
	this->gpu_istrides = strides;
	this->cpu_ostrides = strides;
	this->gpu_ostrides = strides;
    }
};


// Helper for run_test()
static void _setup_io_arrays(Array<void> &in, Array<void> &out, const Array<void> &in_big, const Array<void> &out_big, const TestInstanceDK &tp, bool on_gpu)
{
    int aflags = (on_gpu ? af_gpu : af_uhost) | af_zero;
    vector<long> istrides = on_gpu ? tp.gpu_istrides : tp.cpu_istrides;
    vector<long> ostrides = on_gpu ? tp.gpu_ostrides : tp.cpu_ostrides;
    vector<long> small_shape = { tp.params.beams_per_batch, pow2(tp.params.amb_rank), pow2(tp.params.dd_rank), tp.params.ntime };
    
    if (tp.params.input_is_ringbuf)
	in = in_big;
    else
	in = Array<void> (in_big.dtype, small_shape, istrides, aflags);

    if (tp.in_place)
	out = in;
    else if (tp.params.output_is_ringbuf)
	out = out_big;
    else
	out = Array<void> (out_big.dtype, small_shape, ostrides, aflags);
}


static void run_test(const TestInstanceDK &tp)
{
    const DedispersionKernelParams &p = tp.params;
    
    cout << "\nTest GpuDedispersionKernel\n"
	 << "    ti.params.dtype = " << p.dtype << ";\n"
	 << "    ti.params.dd_rank = " << p.dd_rank << ";\n"
	 << "    ti.params.amb_rank = " << p.amb_rank << ";\n"
	 << "    ti.params.total_beams = " << p.total_beams << ";\n"
	 << "    ti.params.beams_per_batch = " << p.beams_per_batch << ";\n"
	 << "    ti.params.ntime = " << p.ntime << ";\n"
	 << "    ti.params.input_is_ringbuf = " << (p.input_is_ringbuf ? "true" : "false")  << ";\n"
	 << "    ti.params.output_is_ringbuf = " << (p.output_is_ringbuf ? "true" : "false")  << ";\n"
	 << "    ti.params.apply_input_residual_lags = " << (p.apply_input_residual_lags ? "true" : "false")  << ";\n"
	 << "    ti.params.input_is_downsampled_tree = " << (p.input_is_downsampled_tree ? "true" : "false")  << ";\n"
	 << "    ti.params.nelts_per_segment = " << p.nelts_per_segment << ";\n"
	 << "    ti.nchunks = " << tp.nchunks << ";\n"
	 << "    ti.in_place = " << (tp.in_place ? "true" : "false") << ";\n"
	 << "    ti.gpu_istrides = " << ksgpu::tuple_str(tp.gpu_istrides) << ";\n"
	 << "    ti.gpu_ostrides = " << ksgpu::tuple_str(tp.gpu_ostrides) << ";\n"
	 << "    ti.cpu_istrides = " << ksgpu::tuple_str(tp.cpu_istrides) << ";\n"
	 << "    ti.cpu_ostrides = " << ksgpu::tuple_str(tp.cpu_ostrides) << ";\n"
	 << "    ti.new_code = " << tp.new_code << ";"
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
    shared_ptr<GpuDedispersionKernel> old_gpu_kernel;
    shared_ptr<NewGpuDedispersionKernel> new_gpu_kernel;

    if (tp.new_code) {
	new_gpu_kernel = make_shared<NewGpuDedispersionKernel> (p);
	new_gpu_kernel->allocate();
    }
    else {
	old_gpu_kernel = GpuDedispersionKernel::make(p);
	old_gpu_kernel->allocate();
    }

    // Array allocation starts here.
    // "Big" arrays can be either ringbufs, or "big" dedispersion bufs.

    long nbatches = xdiv(p.total_beams, p.beams_per_batch);
    vector<long> rb_shape = { p.ringbuf_nseg * p.nelts_per_segment };
    vector<long> dd_shape = { p.total_beams, pow2(p.amb_rank), pow2(p.dd_rank), tp.nchunks * p.ntime };
    vector<long> big_ishape = p.input_is_ringbuf ? rb_shape : dd_shape;
    vector<long> big_oshape = p.output_is_ringbuf ? rb_shape : dd_shape;

    int iflag = tp.one_hot ? af_zero : af_random;
    Array<float> cpu_in_big(big_ishape, af_rhost | iflag);
    Array<float> cpu_out_big(big_oshape, af_uhost | af_zero);   // contiguous

    if (tp.one_hot) {
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
	cpu_in_big.at({b,a,d,t}) = 1.0f;
    }

    // These are the input/output arrays for the ReferenceDedispersionKernel.
    // They can be either "small" dedispersion bufs, or references to a "big" ringbuf.
    Array<float> cpu_in, cpu_out;
    _setup_io_arrays(cpu_in, cpu_out, cpu_in_big, cpu_out_big, tp, false);  // on_gpu = false

    Array<void> gpu_in_big = cpu_in_big.to_gpu(tp.params.dtype);
    Array<void> gpu_out_big(tp.params.dtype, big_oshape, af_gpu);  // contiguous

    Array<void> gpu_in, gpu_out;
    _setup_io_arrays(gpu_in, gpu_out, gpu_in_big, gpu_out_big, tp, true);  // on_gpu = true
    
    for (long ichunk = 0; ichunk < tp.nchunks; ichunk++) {
	for (long ibatch = 0; ibatch < nbatches; ibatch++) {
	    Array<float> s;
	    Array<void> t;

	    // Reference dedispersion.

	    if (!p.input_is_ringbuf) {
		s = cpu_in_big.slice(0, ibatch * p.beams_per_batch, (ibatch+1) * p.beams_per_batch);
		s = s.slice(3, ichunk * p.ntime, (ichunk+1) * p.ntime);
		cpu_in.fill(s);
	    }

	    ref_kernel->apply(cpu_in, cpu_out, ibatch, ichunk);

	    if (!p.output_is_ringbuf) {
		s = cpu_out_big.slice(0, ibatch * p.beams_per_batch, (ibatch+1) * p.beams_per_batch);
		s = s.slice(3, ichunk * p.ntime, (ichunk+1) * p.ntime);
		s.fill(cpu_out);
	    }
	    
	    // GPU dedipersion.

	    if (!p.input_is_ringbuf) {
		t = gpu_in_big.slice(0, ibatch * p.beams_per_batch, (ibatch+1) * p.beams_per_batch);
		t = t.slice(3, ichunk * p.ntime, (ichunk+1) * p.ntime);
		gpu_in.fill(t);
	    }

	    if (tp.new_code)
		new_gpu_kernel->launch(gpu_in, gpu_out, ibatch, ichunk, nullptr);  // stream=nullptr
	    else
		old_gpu_kernel->launch(gpu_in, gpu_out, ibatch, ichunk, nullptr);  // stream=nullptr

	    if (!p.output_is_ringbuf) {
		t = gpu_out_big.slice(0, ibatch * p.beams_per_batch, (ibatch+1) * p.beams_per_batch);
		t = t.slice(3, ichunk * p.ntime, (ichunk+1) * p.ntime);
		t.fill(gpu_out);
	    }
	}
    }
    
    // FIXME revisit epsilon if we change the normalization of the dedispersion transform.
    double epsrel = 3 * tp.params.dtype.precision();
    double epsabs = 3 * tp.params.dtype.precision() * pow(1.414, p.dd_rank);

    if (p.output_is_ringbuf)
	ksgpu::assert_arrays_equal(cpu_out_big, gpu_out_big, "cpu", "gpu", {"i"}, epsabs, epsrel);
    else
	ksgpu::assert_arrays_equal(cpu_out_big, gpu_out_big, "cpu", "gpu", {"beam","amb","dmbr","time"}, epsabs, epsrel);
}


void test_gpu_dedispersion_kernels()
{
#if 0
    // Debug
    TestInstanceDK ti;
    ti.params.dtype = Dtype::native<float>();
    ti.params.dd_rank = 3;
    ti.params.amb_rank = 1;
    ti.params.total_beams = 2;
    ti.params.beams_per_batch = 2; 
    ti.params.ntime = 128;
    ti.params.input_is_ringbuf = false;
    ti.params.output_is_ringbuf = true;
    ti.params.apply_input_residual_lags = false;
    ti.params.input_is_downsampled_tree = true;
    ti.params.nelts_per_segment = 32;
    ti.nchunks = 1;
    ti.in_place = false;
    ti.randomize_ringbuf();
    ti.set_contiguous_strides();
    ti.new_code = 1;
    // ti.one_hot = true;
    // ti.one_hot_iact = 0;   // rand_int(0, pow2(ti.params.dd_rank));
    // ti.one_hot_itime = 0;  // rand_int(0, ti.params.ntime);
    run_test(ti);
#endif    

#if 1
    TestInstanceDK ti_old;
    ti_old.randomize_old();
    run_test(ti_old);
#endif

#if 1
    TestInstanceDK ti_new;
    ti_new.randomize_new();
    run_test(ti_new);
#endif
}


}  // namespace pirate
