#include "../include/pirate/internals/GpuDedispersionKernel.hpp"
#include "../include/pirate/internals/ReferenceDedispersionKernel.hpp"
#include "../include/pirate/internals/inlines.hpp"    // pow2()
#include "../include/pirate/internals/utils.hpp"      // integer_log2()
#include "../include/pirate/constants.hpp"            // constants::bytes_per_gpu_cache_line

#include <ksgpu/Array.hpp>
#include <ksgpu/cuda_utils.hpp>
#include <ksgpu/rand_utils.hpp>    // rand_int()
#include <ksgpu/test_utils.hpp>    // assert_arrays_equal()

using namespace std;
using namespace pirate;
using namespace ksgpu;


struct TestInstance
{
    // Reminder: includes 'dtype', 'input_is_ringbuf', 'output_is_ringbuf'.
    GpuDedispersionKernel::Params params;

    long nchunks = 0;

    // Can only be 'true' if params.input_is_ringbuf == params.output_is_ringbuf == false.
    bool in_place = false;
    
    // Ring buffer (only used if params.input_is_ringbuf || params.output_is_ringbuf)
    long rb_nzones = 0;
    vector<long> rb_zone_len;   // length rb_nzones
    vector<long> rb_zone_nseg;  // length rb_nzones
    vector<long> rb_zone_base_seg;  // length rb_nzones

    // "Big" shapes are either ringbufs, or dedisp bufs with B = (params.beams_per_gpu) and T = (params.ntime * this->nchunks).
    // "Small" shapes are dedisp bufs with B = (params.beams_per_batch) and T = (params.ntime).
    vector<long> big_ishape;
    vector<long> big_oshape;
    vector<long> small_shape;
    
    // Strides for input/output arrays.
    // Reminder: arrays have shape (params.beams_per_batch, params.nambient, pow2(params.rank), params.ntime).
    vector<long> gpu_istrides;
    vector<long> gpu_ostrides;
    vector<long> cpu_istrides;
    vector<long> cpu_ostrides;
    
    void randomize()
    {
	const long max_nelts = 100 * 1000 * 1000;

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
	else if (rand_uniform() < 0.5) {
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
	//   params.rank
	//   params.ntime
	//   params.nambient
	//   params.total_beams
	//   params.beams_per_batch
	//   this->nchunks
	
	bool is_float32 = (rand_uniform() < 0.5);
	params.dtype = is_float32 ? Dtype::native<float>() : Dtype::native<__half>();
	params.input_is_downsampled_tree = (rand_uniform() < 0.5);

	params.nelts_per_segment = is_float32 ? 32 : 64;
	params.rank = rand_int(1, 9);

	long nchan = pow2(params.rank);
	params.ntime = rand_int(1, 2*nchan + 2*params.nelts_per_segment);
	params.ntime = align_up(params.ntime, params.nelts_per_segment);

	long cmax = (10*nchan + 10*params.ntime) / params.ntime;
	nchunks = rand_int(1, cmax+1);
	
	// nambient, (total_beams/beams_per_batch), beams_per_batch
	long pmax = max_nelts / (pow2(params.rank) * params.ntime * nchunks);
	pmax = max(pmax, 4L);
	pmax = min(pmax, 42L);

	auto v = ksgpu::random_integers_with_bounded_product(4, pmax);
	params.nambient = round_up_to_power_of_two(v[0]);
	params.total_beams = v[1] * v[2];
	params.beams_per_batch = v[2];
	
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

	long nseg = nchan * params.nambient * xdiv(params.ntime, params.nelts_per_segment);
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

	// Shape, strides.

	vector<long> rb_shape = { params.ringbuf_nseg * params.nelts_per_segment };
	vector<long> dd_shape = { params.total_beams, params.nambient, pow2(params.rank), nchunks * params.ntime };
	this->big_ishape = params.input_is_ringbuf ? rb_shape : dd_shape;
	this->big_oshape = params.output_is_ringbuf ? rb_shape : dd_shape;
	this->small_shape = { params.beams_per_batch, params.nambient, pow2(params.rank), params.ntime };
	
	// Dedispersion buffer strides (note that ringbufs are always contiguous).
	this->cpu_istrides = ksgpu::make_random_strides(small_shape, 1, params.nelts_per_segment);
	this->gpu_istrides = ksgpu::make_random_strides(small_shape, 1, params.nelts_per_segment);
	this->cpu_ostrides = in_place ? cpu_istrides : ksgpu::make_random_strides(small_shape, 1, params.nelts_per_segment);
	this->gpu_ostrides = in_place ? gpu_istrides : ksgpu::make_random_strides(small_shape, 1, params.nelts_per_segment);
    }
};


// Helper for run_test()
void _setup_io_arrays(Array<void> &in, Array<void> &out, const Array<void> &in_big, const Array<void> &out_big, const TestInstance &tp, bool on_gpu)
{
    int aflags = (on_gpu ? af_gpu : af_uhost) | af_zero;
    vector<long> istrides = on_gpu ? tp.gpu_istrides : tp.cpu_istrides;
    vector<long> ostrides = on_gpu ? tp.gpu_ostrides : tp.cpu_ostrides;
    
    if (tp.params.input_is_ringbuf)
	in = in_big;
    else
	in = Array<void> (in_big.dtype, tp.small_shape, istrides, aflags);

    if (tp.in_place)
	out = in;
    else if (tp.params.output_is_ringbuf)
	out = out_big;
    else
	out = Array<void> (out_big.dtype, tp.small_shape, ostrides, aflags);
}


static void run_test(const TestInstance &tp)
{
    const GpuDedispersionKernel::Params &p = tp.params;
    
    cout << "Test GpuDedispersionKernel\n"
	 << "    dtype = " << p.dtype << "\n"
	 << "    rank = " << p.rank << "\n"
	 << "    nambient = " << p.nambient << "\n"
	 << "    total_beams = " << p.total_beams << "\n"
	 << "    beams_per_batch = " << p.beams_per_batch << "\n"
	 << "    ntime = " << p.ntime << "\n"
	 << "    nchunks = " << tp.nchunks << "\n"
	 << "    input_is_ringbuf = " << (p.input_is_ringbuf ? "true" : "false")  << "\n"
	 << "    output_is_ringbuf = " << (p.output_is_ringbuf ? "true" : "false")  << "\n"
	 << "    apply_input_residual_lags = " << (p.apply_input_residual_lags ? "true" : "false")  << "\n"
	 << "    input_is_downsampled_tree = " << (p.input_is_downsampled_tree ? "true" : "false")  << "\n"
	 << "    nelts_per_segment = " << p.nelts_per_segment << "\n"
	 << "    in_place = " << (tp.in_place ? "true" : "false") << "\n"
	 << "    gpu_istrides = " << ksgpu::tuple_str(tp.gpu_istrides) << "\n"
	 << "    gpu_ostrides = " << ksgpu::tuple_str(tp.gpu_ostrides) << "\n"
	 << "    cpu_istrides = " << ksgpu::tuple_str(tp.cpu_istrides) << "\n"
	 << "    cpu_ostrides = " << ksgpu::tuple_str(tp.cpu_ostrides)
	 << endl;

    GpuDedispersionKernel::Params gp = p;
    if (p.input_is_ringbuf || p.output_is_ringbuf)
	gp.ringbuf_locations = gp.ringbuf_locations.to_gpu();
    
    shared_ptr<ReferenceDedispersionKernel> ref_kernel = make_shared<ReferenceDedispersionKernel> (p);
    shared_ptr<GpuDedispersionKernel> gpu_kernel = GpuDedispersionKernel::make(gp);

    // Array allocation starts here.
    
    // "Big" arrays can be either ringbufs, or "big" dedispersion bufs.
    Array<float> cpu_in_big(tp.big_ishape, af_rhost | af_random);  // contiguous
    Array<float> cpu_out_big(tp.big_oshape, af_uhost | af_zero);   // contiguous

    // These are the input/output arrays for the ReferenceDedispersionKernel.
    // They can be either "small" dedispersion bufs, or references to a "big" ringbuf.
    Array<float> cpu_in, cpu_out;
    _setup_io_arrays(cpu_in, cpu_out, cpu_in_big, cpu_out_big, tp, false);  // on_gpu = false

    Array<void> gpu_in_big = cpu_in_big.to_gpu(tp.params.dtype);
    Array<void> gpu_out_big(tp.params.dtype, tp.big_oshape, af_gpu);  // contiguous

    Array<void> gpu_in, gpu_out;
    _setup_io_arrays(gpu_in, gpu_out, gpu_in_big, gpu_out_big, tp, true);  // on_gpu = true
    
    for (long ichunk = 0; ichunk < tp.nchunks; ichunk++) {
	for (int b = 0; b < p.total_beams; b += p.beams_per_batch) {
	    Array<float> s;
	    Array<void> t;

	    // Reference dedispersion.

	    if (!p.input_is_ringbuf) {
		s = cpu_in_big.slice(0, b, b + p.beams_per_batch);
		s = s.slice(3, ichunk * p.ntime, (ichunk+1) * p.ntime);
		cpu_in.fill(s);
	    }

	    ref_kernel->apply(cpu_in, cpu_out, ichunk, b);

	    if (!p.output_is_ringbuf) {
		s = cpu_out_big.slice(0, b, b + p.beams_per_batch);
		s = s.slice(3, ichunk * p.ntime, (ichunk+1) * p.ntime);
		s.fill(cpu_out);
	    }
	    
	    // GPU dedipersion.

	    if (!p.input_is_ringbuf) {
		t = gpu_in_big.slice(0, b, b + p.beams_per_batch);
		t = t.slice(3, ichunk * p.ntime, (ichunk+1) * p.ntime);
		gpu_in.fill(t);
	    }

	    gpu_kernel->launch(gpu_in, gpu_out, ichunk, b);

	    if (!p.output_is_ringbuf) {
		t = gpu_out_big.slice(0, b, b + p.beams_per_batch);
		t = t.slice(3, ichunk * p.ntime, (ichunk+1) * p.ntime);
		t.fill(gpu_out);
	    }
	}
    }
    
    // FIXME revisit epsilon if we change the normalization of the dedispersion transform.
    double epsrel = 3 * tp.params.dtype.precision();
    double epsabs = 3 * tp.params.dtype.precision() * pow(1.414, p.rank);

    if (p.output_is_ringbuf)
	ksgpu::assert_arrays_equal(cpu_out_big, gpu_out_big, "cpu", "gpu", {"i"}, epsabs, epsrel);
    else
	ksgpu::assert_arrays_equal(cpu_out_big, gpu_out_big, "cpu", "gpu", {"beam","amb","dmbr","time"}, epsabs, epsrel);
    
    cout << endl;
}


// -------------------------------------------------------------------------------------------------


int main(int argc, char **argv)
{
    const int niter = 200;
    
    for (int i = 0; i < niter; i++) {
	cout << "Iteration " << i << "/" << niter << "\n\n";
	TestInstance ti;
	ti.randomize();
	run_test(ti);
    }

    cout << "test-gpu-dedispersion-kernels: pass" << endl;
    return 0;
}

