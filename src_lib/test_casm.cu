#include "../include/pirate/casm.hpp"

#include <cassert>
#include <ksgpu/Array.hpp>
#include <ksgpu/cuda_utils.hpp>
#include <ksgpu/rand_utils.hpp>

using namespace std;
using namespace ksgpu;

namespace pirate {
#if 0
}  // editor auto-indent
#endif


// Caller must check that 1 <= x <= (n-2) within roundoff error.
// (FIXME comment on how this happens in full kernel.)

__device__ void grid_interpolation_site(float x, int n, int &ix, float &dx)
{
    ix = int(x);
    ix = (ix >= 1) ? ix : 0;
    ix = (ix <= (n-3)) ? ix : (n-3);
    dx = x - float(ix);
}


__device__ void compute_interpolation_weights(float dx, float &w0, float &w1, float &w2, float &w3)
{
    static constexpr float one_sixth = 1.0f / 6.0f;
    static constexpr float one_half = 1.0f / 2.0f;
    
    w0 = -one_sixth * (dx) * (dx-1.0f) * (dx-2.0f);
    w1 = one_half * (dx+1.0f) * (dx-1.0f) * (dx-2.0f);
    w2 = -one_half * (dx+1.0f) * (dx) * (dx-2.0f);
    w3 = one_sixth * (dx+1.0f) * (dx) * (dx-1.0f);
}


// Helper for interpolate_slow()
__device__ float _interpolate_slow_1d(const float *sp, float wy0, float wy1, float wy2, float wy3)
{
    return wy0*sp[0] + wy1*sp[1] + wy2*sp[2] + wy3*sp[3];
}


// Interpolate on (24,128) grid in shared memory, stride=133.
// Caller must check that 1 <= x <= 22, and 1 <= y <= 126, within roundoff error.
__device__ float interpolate_slow(const float *sp, float x, float y)
{
    int ix, iy;
    float dx, dy;

    grid_interpolation_site(x, 24, ix, dx);
    grid_interpolation_site(y, 128, iy, dy);
    sp += 133*(ix-1) + (iy-1);

    float wx0, wx1, wx2, wx3, wy0, wy1, wy2, wy3;
    compute_interpolation_weights(dx, wx0, wx1, wx2, wx3);
    compute_interpolation_weights(dy, wy0, wy1, wy2, wy3);

    float ret = wx0 * _interpolate_slow_1d(sp, wy0, wy1, wy2, wy3);
    ret += wx1 * _interpolate_slow_1d(sp+133, wy0, wy1, wy2, wy3);
    ret += wx2 * _interpolate_slow_1d(sp+2*133, wy0, wy1, wy2, wy3);
    ret += wx3 * _interpolate_slow_1d(sp+3*133, wy0, wy1, wy2, wy3);

    return ret;
}


// Factor interpolation weight as w_j = pf * (x+a) * (x+b) * (x+c), where 0 <= j < 4
__device__ void compute_abc(int j, float &pf, float &a, float &b, float &c)
{
    static constexpr float one_sixth = 1.0f / 6.0f;
    static constexpr float one_half = 1.0f / 2.0f;

    pf = ((j==0) || (j==3)) ? one_sixth : one_half;
    pf = (j & 1) ? pf : (-pf);
	
    a = (j > 0) ? 1.0f : 0.0f;
    b = (j > 1) ? 0.0f : -1.0f;
    c = (j > 2) ? -1.0f : -2.0f;
}


template<bool Debug>
__device__ float interpolate_fast(const float *sp, float x, float y)
{
    int ix_g, iy_g;
    float dx_g, dy_g;
    float ret = 0.0;
    
    grid_interpolation_site(x, 24, ix_g, dx_g);
    grid_interpolation_site(y, 128, iy_g, dy_g);
    
    int jx = (threadIdx.x >> 2) & 3;
    int jy = (threadIdx.x & 3);
    int ds = 133*(jx-1) + (jy-1);
    int sg = 133*ix_g + iy_g;

    float pfx, pfy, ax, bx, cx, ay, by, cy;
    compute_abc(jx, pfx, ax, bx, cx);
    compute_abc(jy, pfy, ay, by, cy);
    pfx *= pfy;  // save one register
    
    for (int iouter = 0; iouter < 16; iouter++) {
	int src_lane = (threadIdx.x & 0x10) | iouter;
	
	int s = __shfl_sync(0xffffffff, sg, src_lane) + ds;
	float dx = __shfl_sync(0xffffffff, dx_g, src_lane);
	float dy = __shfl_sync(0xffffffff, dy_g, src_lane);

	if constexpr (Debug) {
	    assert(s >= 0);
	    assert(s < 24*133);
	    uint m = __match_any_sync(0xffffffff, s & 31);
	    assert(__popc(m) <= 2);  // at most 2:1 bank conflict
	}
	
	float w = pfx * (dx+ax) * (dx+bx) * (dx+cx) * (dy+ay) * (dy+by) * (dy+cy);
	float t = w * sp[s];

	// FIXME placeholder for fast reduce
	t += __shfl_sync(0xffffffff, t, threadIdx.x ^ 1);
	t += __shfl_sync(0xffffffff, t, threadIdx.x ^ 2);
	t += __shfl_sync(0xffffffff, t, threadIdx.x ^ 4);
	t += __shfl_sync(0xffffffff, t, threadIdx.x ^ 8);
	ret = ((threadIdx.x & 15) == iouter) ? t : ret;
    }

    return ret;
}


// Launch with 32 threads, 1 block.
//   - out_slow: shape (32,)
//   - out_fast: shape (32,)
//   - xy: shape (2,32)
//   - grid: shape (24,133)

__global__ void casm_interpolation_test_kernel(float *out_slow, float *out_fast, const float *xy, const float *grid)
{
    __shared__ float sgrid[24*133];

    for (int i = threadIdx.x; i < 24*133; i += 32)
	sgrid[i] = grid[i];
    
    float x = xy[threadIdx.x];
    float y = xy[threadIdx.x + 32];
    
    out_slow[threadIdx.x] = interpolate_slow(sgrid, x, y);
    out_fast[threadIdx.x] = interpolate_fast<true> (sgrid, x, y);
}


static void test_casm_interpolation()
{
    Array<float> xy({64}, af_rhost);
    Array<float> grid({24,133}, af_random | af_gpu);
    Array<float> out_slow({32}, af_random | af_gpu);
    Array<float> out_fast({32}, af_random | af_gpu);

    for (int i = 0; i < 32; i++) {
	xy.data[i] = rand_uniform(1.0f, 22.0f);
	xy.data[i+32] = rand_uniform(1.0f, 126.0f);
    }

    xy = xy.to_gpu();
    
    casm_interpolation_test_kernel<<<1,32>>> (out_slow.data, out_fast.data, xy.data, grid.data);
    CUDA_PEEK("casm_interpolation_test_kernel");
    
    assert_arrays_equal(out_slow, out_fast, "slow", "fast", {"i"});
    cout << "test_casm_interpolation: pass" << endl;
}


void test_casm()
{
    test_casm_interpolation();
}


}  // namespace pirate
