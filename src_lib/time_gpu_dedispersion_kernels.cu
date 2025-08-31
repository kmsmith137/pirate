#include <iostream>
#include <ksgpu/Array.hpp>
#include <ksgpu/cuda_utils.hpp>   // CUDA_CALL(), CudaStreamWrapper
#include <ksgpu/time_utils.hpp>   // get_time(), time_diff()

#include "../include/pirate/timing.hpp"
#include "../include/pirate/inlines.hpp"  // pow2()
#include "../include/pirate/DedispersionKernel.hpp"

using namespace std;
using namespace ksgpu;

namespace pirate {
#if 0
}  // editor auto-indent
#endif


// Uses one stream per "beam batch".
static void time_gpu_dedispersion_kernel(const DedispersionKernelParams &params, long nchunks=24)
{
    cout << "\nTime GPU dedispersion kernel\n";
    params.print();
    
    long nbatches = xdiv(params.total_beams, params.beams_per_batch);
    vector<ksgpu::CudaStreamWrapper> streams(nbatches);   // creates one stream per batch
    vector<struct timeval> tv(nchunks * nbatches);

    shared_ptr<GpuDedispersionKernel> kernel = make_shared<GpuDedispersionKernel> (params);
    kernel->allocate();
    
    vector<long> dd_shape = { params.total_beams, pow2(params.amb_rank), pow2(params.dd_rank), params.ntime, params.nspec };
    vector<long> rb_shape = { params.ringbuf_nseg * params.nt_per_segment * params.nspec };
    vector<long> in_shape = params.input_is_ringbuf ? rb_shape : dd_shape;
    vector<long> out_shape = params.output_is_ringbuf ? rb_shape : dd_shape;

    Array<void> in_big(params.dtype, in_shape, af_gpu | af_zero);
    Array<void> out_big(params.dtype, out_shape, af_gpu | af_zero);
    double gb = 1.0e-9 * kernel->bw_per_launch.nbytes_gmem;

    for (long ichunk = 0; ichunk < nchunks; ichunk++) {
	for (long ibatch = 0; ibatch < nbatches; ibatch++) {
	    long k = ichunk*nbatches + ibatch;
	    CUDA_CALL(cudaStreamSynchronize(streams[ibatch]));
	    tv[k] = ksgpu::get_time();

	    Array<void> in_slice = in_big;
	    Array<void> out_slice = out_big;

	    if (!params.input_is_ringbuf)
		in_slice = in_big.slice(0, ibatch * params.beams_per_batch, (ibatch+1) * params.beams_per_batch);
	    if (!params.output_is_ringbuf)
		out_slice = out_big.slice(0, ibatch * params.beams_per_batch, (ibatch+1) * params.beams_per_batch);

	    kernel->launch(in_slice, out_slice, ibatch, ichunk, streams[ibatch]);

	    if ((ichunk % 4) != 3)
		continue;
	    
	    int j = k - (k/(2*nbatches))*nbatches;
	    if (j < k)
		cout << "    " << ((k-j) * gb / ksgpu::time_diff(tv[j],tv[k])) << " GB/s\n";
	}
    }
}


void time_gpu_dedispersion_kernels()
{
    long nstreams = 2;
    
    for (int dd_rank: {4,8}) {
	for (int stage: {0,1,2}) {
	    for (Dtype dtype: { Dtype::native<float>(), Dtype::native<__half>() }) {
		long nspec = 1;  // FIXME
		long nbeams = pow2(19 - 2*dd_rank);
    
		DedispersionKernelParams params;
		params.dtype = dtype;
		params.dd_rank = dd_rank;
		params.amb_rank = dd_rank;
		params.beams_per_batch = nbeams;
		params.total_beams = nbeams * nstreams;
		params.ntime = xdiv(2048, nspec);
		params.nspec = nspec;
		params.input_is_ringbuf = (stage == 2);
		params.output_is_ringbuf = (stage == 1);	
		params.apply_input_residual_lags = (stage == 2);
		params.input_is_downsampled_tree = false;  // shouldn't affect timing
		params.nt_per_segment = xdiv(1024, dtype.nbits * nspec);

		if (params.input_is_ringbuf || params.output_is_ringbuf) {
		    // Make some nominal ringbuf locations.
		    // The details shouldn't affect the timing much.

		    long rb_len = 2 * params.total_beams;
		    long nrows_per_tree = pow2(params.dd_rank + params.amb_rank);
		    long nseg_per_row = xdiv(params.ntime, params.nt_per_segment);
		    long nseg_per_tree = nrows_per_tree * nseg_per_row;

		    params.ringbuf_nseg = rb_len * nseg_per_tree;
		    params.ringbuf_locations = Array<uint> ({nseg_per_tree,4}, af_rhost | af_zero);
		    uint *rp = params.ringbuf_locations.data;

		    for (long iseg = 0; iseg < nseg_per_tree; iseg++) {
			rp[4*iseg] = iseg;             // rb_offset
			rp[4*iseg+1] = 0;              // rb_phase
			rp[4*iseg+2] = rb_len;         // rb_len
			rp[4*iseg+3] = nseg_per_tree;  // rb_nseg
		    }
		}

		time_gpu_dedispersion_kernel(params);
	    }
	}
    }
}


}  // namespace pirate
