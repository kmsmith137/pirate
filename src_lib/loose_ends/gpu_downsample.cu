#include "../../include/pirate/loose_ends/gpu_downsample.hpp"
#include "../../include/pirate/loose_ends/DownsampleKernel.hpp"

#include <stdexcept>
#include <ksgpu/Array.hpp>
#include <ksgpu/xassert.hpp>

using namespace std;
using namespace ksgpu;

namespace pirate {
#if 0
}  // editor auto-indent
#endif


void launch_downsample(Array<float> &dst_i, Array<float> &dst_w,
		       const Array<float> &src_i, const Array<float> &src_w,
		       int Df, int Dt, bool transpose_output, cudaStream_t stream)
{
    xassert(Df >= 1);
    xassert(Dt >= 1);
    xassert((Dt == 1) || (Dt == 2) || ((Dt % 4) == 0));

    if (!transpose_output)
	throw runtime_error("launch_downsample(transpose_output=false) not implemented yet!");

    xassert(dst_i.ndim == 3);
    xassert(dst_w.ndim == 3);
    xassert(src_i.ndim == 3);
    xassert(src_w.ndim == 3);
    
    for (int d = 0; d < 3; d++) {
	xassert(dst_i.shape[d] == dst_w.shape[d]);
	xassert(dst_i.strides[d] == dst_w.strides[d]);
	xassert(src_i.shape[d] == src_w.shape[d]);
	xassert(src_i.strides[d] == src_w.strides[d]);
    }
    
    // Source array: (beam, freq, time)
    unsigned int nbeams_src = src_i.shape[0];
    unsigned int nfreq_src = src_i.shape[1];
    unsigned int ntime_src = src_i.shape[2];
    
    // Destination array: (beam, time, freq)
    unsigned int nbeams_dst = dst_i.shape[0];
    unsigned int ntime_dst = dst_i.shape[1];
    unsigned int nfreq_dst = dst_i.shape[2];

    xassert(nbeams_src == nbeams_dst);
    xassert(nfreq_src == Df * nfreq_dst);
    xassert(ntime_src == Dt * ntime_dst);

    xassert((src_i.strides[0] % 4) == 0);
    xassert((src_i.strides[1] % 4) == 0);
    xassert(src_i.strides[2] == 1);
    
    xassert((dst_i.strides[0] % 4) == 0);
    xassert((dst_i.strides[1] % 4) == 0);
    xassert(dst_i.strides[2] == 1);
    
    int src_bstride = src_i.strides[0];
    int src_fstride = src_i.strides[1];
    int dst_bstride = dst_i.strides[0];
    int dst_tstride = dst_i.strides[1];

    dim3 nblocks;
    nblocks.x = ntime_dst / 32;
    nblocks.y = nfreq_dst / 32;
    nblocks.z = nbeams_dst;

    if (Dt == 1)
	downsample_kernel<1>
	    <<< nblocks, 32*DownsampleKernel::warps_per_block, 0, stream >>>
	    (dst_i.data, dst_w.data, src_i.data, src_w.data, Df, Dt, src_fstride, src_bstride, dst_tstride, dst_bstride);
    else if (Dt == 2)
	downsample_kernel<2>
	    <<< nblocks, 32*DownsampleKernel::warps_per_block, 0, stream >>>
	    (dst_i.data, dst_w.data, src_i.data, src_w.data, Df, Dt, src_fstride, src_bstride, dst_tstride, dst_bstride);
    else if (Dt >= 4)
	downsample_kernel<4>
	    <<< nblocks, 32*DownsampleKernel::warps_per_block, 0, stream >>>
	    (dst_i.data, dst_w.data, src_i.data, src_w.data, Df, Dt, src_fstride, src_bstride, dst_tstride, dst_bstride);
}


}  // namespace pirate
