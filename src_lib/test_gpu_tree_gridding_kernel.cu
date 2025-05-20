#include "../include/pirate/TreeGriddingKernel.hpp"
#include "../include/pirate/inlines.hpp"  // xdiv

#include <ksgpu/Array.hpp>
#include <ksgpu/rand_utils.hpp>
#include <ksgpu/test_utils.hpp>
#include <algorithm>  // std::sort()

using namespace std;
using namespace ksgpu;

namespace pirate {
#if 0
}  // editor auto-indent
#endif


void test_gpu_tree_gridding_kernel()
{
    TreeGriddingKernelParams params;
    params.dtype = (rand_uniform() < 0.5) ? Dtype(df_float,16) : Dtype(df_float,32);

    auto v = ksgpu::random_integers_with_bounded_product(3, 1000);
    int B = v[0];
    int F = v[1];
    int T = v[2] * xdiv(1024, params.dtype.nbits);
    int N = rand_int(1, 2000/(v[0]*v[2]));
    
    long hs_in = rand_int(F*T, 2*F*T);   // host beam stride, input array
    long gs_in = rand_int(F*T, 2*F*T);   // gpu beam stride, input array
    long hs_out = rand_int(N*T, 2*N*T);  // host beam stride, output array
    long gs_out = rand_int(N*T, 2*N*T);  // gpu beam stride, output array
    
    params.beams_per_batch = B;
    params.nfreq = F;
    params.nchan = N;
    params.ntime = T;

    cout << "test_gpu_tree_gridding_kernel()\n"
	 << "    dtype = " << params.dtype << "\n"
	 << "    beams_per_batch = " << B << "\n"
	 << "    nfreq = " << F << "\n"
	 << "    nchan = " << N << "\n"
	 << "    ntime = " << T << "\n"
	 << "    host_src_beam_stride = " << hs_in << "\n"
	 << "    host_dst_beam_stride = " << hs_out << "\n"
	 << "    gpu_src_beam_stride = " << gs_in << "\n"
	 << "    gpu_dst_beam_stride = " << gs_out << endl;
    
    vector<float> cvec(N+1);
    cvec[0] = 0;
    cvec[N] = F;
    for (int i = 1; i < N; i++)
	cvec[i] = rand_uniform(0, F);

    std::sort(cvec.begin(), cvec.end());

    params.channel_map = Array<float> ({N+1}, af_rhost | af_zero);
    memcpy(params.channel_map.data, &cvec[0], (N+1) * sizeof(float));
    
    Array<float> hsrc({B,F,T}, {hs_in,T,1}, af_uhost | af_random | af_guard);
    Array<float> hdst({B,N,T}, {hs_out,T,1}, af_uhost | af_zero | af_guard);

    Array<void> gsrc(params.dtype, {B,F,T}, {gs_in,T,1}, af_gpu | af_zero | af_guard);
    Array<void> gdst(params.dtype, {B,N,T}, {gs_out,T,1}, af_gpu | af_zero | af_guard);
    gsrc.fill(hsrc.convert(params.dtype));

    ReferenceTreeGriddingKernel hkernel(params);
    hkernel.apply(hdst, hsrc);

    GpuTreeGriddingKernel gkernel(params);
    gkernel.allocate();
    gkernel.launch(gdst, gsrc, nullptr);   // null stream
    
    assert_arrays_equal(hdst, gdst, "hdst", "gdst", {"b","n","t"});
}


}  // namespace pirate
