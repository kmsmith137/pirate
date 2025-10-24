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


// -------------------------------------------------------------------------------------------------


// Complex out += x*y
__device__ void zma(float &out_re, float &out_im, float xre, float xim, float yre, float yim)
{
    out_re += (xre*yre - xim*yim);
    out_im += (xre*yim + xim*yre);
}

// Complex out += e^(i*theta)*y
__device__ void zma_expi(float &out_re, float &out_im, float theta, float yre, float yim)
{
    float xre, xim;
    sincosf(theta, &xim, &xre);   // note ordering (im,re)
    zma(out_re, out_im, xre, xim, yre, yim);
}

// FIXME by calling warp_transpose() in pairs, am I being suboptimal?
template<typename T>
__device__ void warp_transpose(T &x, T &y, uint bit)
{
    static_assert(sizeof(T) == 4);

    bool flag = (threadIdx.x & bit) != 0;
    T z = __shfl_sync(0xffffffff, (flag ? x : y), threadIdx.x ^ bit);
    x = flag ? z : x;  // compiles to conditional (predicated) move, not branch
    y = flag ? y : z;  // compiles to conditional (predicated) move, not branch
}


template<bool Debug>
__device__ void check_bank_conflict_free(int offset, int max_conflicts=1)
{
    if constexpr (Debug) {
	uint m = __match_any_sync(0xffffffff, offset & 31);
	assert(__popc(m) <= max_conflicts);
	assert(offset >= 0);
    }
}


// -------------------------------------------------------------------------------------------------
//
// Memory layouts


// Global memory layout for 'gpu_persistent_data'.
//
// This is a ~200KB region of global GPU memory, which is initialized by the
// CasmBeamformer constructor, and passed to the beamformer on every kernel launch.
//
//   uint gridding[256];                  contains (43*ew+ns) for each dish
//   float ns_phases[32];                 contains cos(2pi * t / 128) for 0 <= t < 32
//   float per_frequency_data[F][3][32];  middle index is 0 <= ew_feed < 3, see (*) below
//   float beam_locs[B][2];               contains sin(za), ordering is {ns,ew}.
//
// (*) The 32-element per_frequency_data "inner" index is laid out as follows:
//
//   float ew_phases[12][2];              (ew_beam, cos/sin) for fixed 0 <= ew_feed < 3
//   float freq;                          array element 25

struct gmem_layout
{
    // All _*base quantities are 32-bit offsets, not byte offsets.
    static constexpr int gridding_base = 0;
    static constexpr int ns_phases_base = 256;
    static __host__ __device__ constexpr int per_frequency_data_base(int f=0)  { return ns_phases_base + 32 + 96*f; }
    static __host__ __device__ constexpr int beam_locs_base(int F, int b=0)    { return ns_phases_base + 32 + 96*F + 2*b; }
};


// Shared memory layout (columns are [32-bit offset], physical layout, logical layout).
// Currently using MAX_BEAMS=4704, which gives exactly 99 KB.
//
// [0]      uint gridding[256];                                       // contains (43*ew+ns), derived from 'feed_indices'
// [256]    float ns_phases[32];                                      // contains cos(2pi * t / 128) for 0 <= t < 32
// [288]    float per_frequency_data[3][32];                          // outer index is 0 <= ew_feed < 3, see (*) below
// [384]    uint E[24][259]      union { E[24][256], E[24][6][43] }   // (j,dish) or (j,ew,ns), outer stride 259
// [6600]   float I[24][132]     float I[24][128];                    // (ew,ns), ew-stride 132
// [9768]   float G[24][257]     float G[2][2][6][2][128];            // (time,pol,ew,reim,ns), ew-stride 257
// [15936]  float beam_locs[2][MAX_BEAMS];                            // contains sin(za), ordering is {ns,ew}

struct shmem_layout
{
    // All _*stride and _*base quantities are 32-bit offsets, not byte offsets.
    static constexpr int E_jstride = 259;
    static constexpr int I_ew_stride = 132;
    static constexpr int G_ew_stride = 257;
    
    static constexpr int gridding_base = 0;
    static constexpr int ns_phases_base = 256;
    static constexpr int per_frequency_data_base = ns_phases_base + 32;
    static constexpr int E_base = per_frequency_data_base + 96;
    static constexpr int I_base = E_base + 24 * E_jstride;
    static constexpr int G_base = I_base + 24 * I_ew_stride;
    static constexpr int beam_locs_base = G_base + 24 * G_ew_stride;

    static constexpr int max_beams = ((99*1024 - 4*beam_locs_base) / 8) & ~31;
    static constexpr int beam_stride = max_beams;
    static constexpr int nbytes = (beam_locs_base + 2*max_beams) * 4;
    
    // During initialization, the E[], I[], and G[] arrays aren't needed yet,
    // so we temporarily use their shared memory to store feed_weights:
    //
    // [384]    float ungridded_wts[8][259];   uwt[2][2][2][256]    // (duplicator, reim, pol, dish)
    // [2440]   float gridded_wts[4][259];     gwt[2][2][6][43]     // (reim, pol, ew, ns)

    static constexpr int wt_pol_stride = 259;
    static constexpr int ungridded_wts_base = E_base;
    static constexpr int gridded_wts_base = E_base + 8 * wt_pol_stride;
};

    
// -------------------------------------------------------------------------------------------------
//
// Host-side CasmBeamformer object
//
// FIXME make this a class, with some members protected.
// FIXME switch to an API which doesn't use ksgpu::Array or xassert()
// FIXME constructor should save current cuda device, and check equality in beamform()


struct CasmBeamformer
{
    // speed of light in weird units meters-MHz
    static constexpr float speed_of_light = 299.79;

    inline static const float default_ew_feed_spacings[5]
	= { 0.38f, 0.445f, 0.38f, 0.445f, 0.38f };  // meters
    
    CasmBeamformer(
        Array<float> &frequencies,     // shape (F,)
	Array<int> &feed_indices,      // shape (256,2)
	Array<float> &beam_locations,  // shape (B,2)
	int downsampling_factor,
	float ns_feed_spacing = 0.50,  // meters
	const float *ew_feed_spacing = default_ew_feed_spacings
    );
    
    int F = 0;  // number of frequency channels (on one GPU)
    int B = 0;  // number of output beams
    int downsampling_factor = 0;
    
    ksgpu::Array<float> &frequencies;     // shape (F,)
    ksgpu::Array<int> feed_indices;       // shape (256,2)
    ksgpu::Array<float> &beam_locations;  // shape (B,2)
    float ns_feed_spacing = 0.0;

    float ew_feed_spacing[5];     // meters
    float ew_feed_positions[6];   // meters
    float ew_beam_locations[24];  // sin(ZA)

    static int get_max_beams();
    
    // This is a ~200KB region of global GPU memory, which is initialized by the
    // CasmBeamformer constructor, and passed to the beamformer on every kernel launch.
    // See "global memory layout" in CasmBeamformer.cu for more info.
    Array<float> gpu_persistent_data;
    
    // For unit tests
    int nominal_Tin_for_unit_tests = 0;
    static CasmBeamformer make_random();
    static void show_shared_memory_layout();
};


CasmBeamformer::CasmBeamformer(
    Array<float> &frequencies_,     // shape (F,)
    Array<int> &feed_indices_,      // shape (256,2)
    Array<float> &beam_locations_,  // shape (B,2)
    int downsampling_factor_,
    float ns_feed_spacing_,
    const float *ew_feed_spacing_)
    : frequencies(frequencies_),
      feed_indices(feed_indices_),
      beam_locations(beam_locations_),
      downsampling_factor(downsampling_factor_),
      ns_feed_spacing(ns_feed_spacing_)
{
    for (int i = 0; i < 5; i++)
	ew_feed_spacing[i] = ew_feed_spacing_[i];
    
    // Argument checking
    xassert_eq(frequencies.ndim, 1);
    xassert_eq(beam_locations.ndim, 2);
    xassert_eq(beam_locations.shape[1], 2);
    xassert_shape_eq(feed_indices, ({256,2}));
    
    this->F = frequencies.shape[0];
    this->B = beam_locations.shape[0];

    xassert_gt(F, 0);
    xassert_gt(B, 0);
    xassert_divisible(B, 32);  // currently required by GPU kernel, but not python reference code
    xassert_gt(downsampling_factor, 0);	       
    xassert_ge(ns_feed_spacing, 0.3);
    xassert_lt(ns_feed_spacing, 0.6);
    
    for (int i = 0; i < 5; i++) {
	xassert_ge(ew_feed_spacing[i], 0.3);
	xassert_le(ew_feed_spacing[i], 0.6);
	xassert(fabs(ew_feed_spacing[i] - ew_feed_spacing[4-i]) < 1.0e-5);  // flip-symmetric
    }

    ew_feed_positions[3] = ew_feed_spacing[2] / 2.;
    ew_feed_positions[4] = ew_feed_positions[3] + (ew_feed_spacing[1] + ew_feed_spacing[3]) / 2.;
    ew_feed_positions[5] = ew_feed_positions[4] + (ew_feed_spacing[0] + ew_feed_spacing[4]) / 2.;

    for (int i = 0; i < 24; i++)
	ew_beam_locations[i] = (23-2*i) / 21.;
    
    // The rest of this function fills 'gpu_persistent_data' and copies to the GPU.
    gpu_persistent_data = Array<float> ({256 + 32 + 96*F + 2*B}, af_rhost | af_zero);
    
    // Compute 'gridding' part of 'gpu_persistent_data' (with error-checking).
    uint *gp = (uint *) (gpu_persistent_data.data);
    vector<int> duplicate_checker({6*43}, -1);
    
    for (int d = 0; d < 256; d++) {
	int ns = feed_indices.at({d,0});  // 0 <= ns < 43
	int ew = feed_indices.at({d,1});  // 0 <= ew < 6

	if ((ns < 0) || (ns >= 43) || (ew < 0) || (ew >= 43)) {
	    stringstream ss;
	    ss << "CasmBeamformer constructor: got feed_indices[" << d << "]=("
	       << ns << "," << ew << "). Expected pair (j,k) where 0 <= j < 43"
	       << " and 0 <= k < 6";
	    throw runtime_error(ss.str());
	}

	int g = 43*ew + ns;
	
	if (duplicate_checker[g] >= 0) {
	    stringstream ss;
	    ss << "CasmBeamformer constructor: duplicate feed_indices["
	       << duplicate_checker[g] << "] = feed_indices[" << d << "] = ("
	       << ns << "," << ew << ")";
	    throw runtime_error(ss.str());
	}

	duplicate_checker[g] = d;
	gp[d] = g;
    }

    // Compute 'ns_phases' part of 'gpu_persistent_data'.
    float *nsp = gpu_persistent_data.data + shmem_layout::ns_phases_base;
    for (int i = 0; i < 32; i++)
	nsp[i] = cosf(2 * M_PI * i / 128.0);

    // Compute 'per_frequency_data' part of 'gpu_persistent_data'.    
    for (int f = 0; f < F; f++) {
	float freq = frequencies.at({f});
	
	// Beamformer has only been validated for frequencies in [400,500] MHz.
        // This assert also guards against using the wrong units (should be MHz).
	xassert_ge(freq, 399.0);
	xassert_lt(freq, 501.0);

	for (int ew_feed = 0; ew_feed < 3; ew_feed++) {
	    // Points to 32-element "inner" region (see (*) above).
	    float *pf32 = gpu_persistent_data.data + gmem_layout::per_frequency_data_base(f) + 32*ew_feed;
	    pf32[25] = freq;
	    
	    for (int ew_beam = 0; ew_beam < 12; ew_beam++) {
		float feed_pos = ew_feed_positions[ew_feed + 3];
		float beam_loc = ew_beam_locations[ew_beam + 12];
		float theta = (2*M_PI / speed_of_light) * freq * feed_pos * beam_loc;
		sincosf(theta, &pf32[2*ew_beam+1], &pf32[2*ew_beam]);  // note (sin, cos) ordering
	    }
	}
    }

    // Compute 'beam_locations' part of gpu_persistent_data
    float *blp = gpu_persistent_data.data + gmem_layout::beam_locs_base(F);
    
    for (int b = 0; b < B; b++) {
	for (int j = 0; j < 2; j++) {
	    float beam_location = beam_locations.at({b,j});
	    xassert_ge(beam_location, -1.0);
	    xassert_le(beam_location, 1.0);
	    blp[2*b+j] = beam_location;
	}
    }

    // All done! Copy to GPU.
    gpu_persistent_data = gpu_persistent_data.to_gpu();
}


// Static member function.
void CasmBeamformer::show_shared_memory_layout()
{
    using SL = shmem_layout;

    cout << "[" << SL::gridding_base << "]      uint gridding[256];\n"
	 << "[" << SL::ns_phases_base << "]    float ns_phases[32];\n"
	 << "[" << SL::per_frequency_data_base << "]    float per_frequency_data[3][32];\n"
	 << "[" << SL::E_base << "]    uint E[24][" << SL::E_jstride << "];\n"
	 << "[" << SL::I_base << "]   float I[24][" << SL::I_ew_stride << "];\n"
	 << "[" << SL::G_base << "]   float G[24][" << SL::G_ew_stride << "];\n"
	 << "[" << SL::beam_locs_base << "]  float beam_locs[2][" << SL::max_beams << "];\n"
	 << "  Total size = " << SL::nbytes << " bytes = " << (SL::nbytes/1024.0) << " KB"
	 << endl;
}


// Static member function.
int CasmBeamformer::get_max_beams()
{
    // Maximum number of beams is currently determined by shared memory constraints.
    return shmem_layout::max_beams;
}


// Static member function.
CasmBeamformer CasmBeamformer::make_random()
{
    auto w = ksgpu::random_integers_with_bounded_product(3, 100);
    int B = 32 * ksgpu::rand_int(1,6);  // FIXME increase
    int T = w[0] * w[1];
    int ds = w[1];
    int F = w[2];

    // FIXME currently we have a requirement that (T % 48) == 0,
    // in addition to the requirement that (T % ds) == 0.
    int d = std::lcm(ds, 48);
    T = ((T+d-1) / d) * d;  // round up to nearest multiple of d

    Array<float> frequencies({F}, af_uhost);
    Array<int> feed_indices({256,2}, af_uhost);
    Array<float> beam_locations({B,2}, af_uhost);
    float ns_feed_spacing = ksgpu::rand_uniform(0.3, 0.6);
    float ew_feed_spacing[5];

    // Make random 'frequencies' array.
    for (int f = 0; f < F; f++)
	frequencies.at({f}) = rand_uniform(400.0, 500.0);
    
    // Make random 'beam_locations' array.
    for (int b = 0; b < B; b++)
	for (int j = 0; j < 2; j++)
	    beam_locations.at({b,j}) = rand_uniform(-1.0, 1.0);

    // Make random 'ew_feed_spacing' array.
    for (int i = 0; i < 3; i++)
	ew_feed_spacing[i] = ew_feed_spacing[4-i] = rand_uniform(0.3, 0.6);
    
    // Make random 'feed_indices' array.
    vector<uint> v(43*6);
    for (uint ns = 0; ns < 43; ns++)
	for (uint ew = 0; ew < 6; ew++)
	    v.at(6*ns+ew) = (ew << 16) | ns;

    ksgpu::randomly_permute(v);

    for (uint d = 0; d < 256; d++) {
	feed_indices.at({d,0}) = v[d] & 0xffff;  // ns
	feed_indices.at({d,1}) = v[d] >> 16;     // ew
    }

    CasmBeamformer ret(frequencies, feed_indices, beam_locations, ds, ns_feed_spacing, ew_feed_spacing);
    ret.nominal_Tin_for_unit_tests = T;
    
    return ret;
}


// -------------------------------------------------------------------------------------------------
//
// casm_shuffle_state
//
// Reindex (time,pol) as (i,j) where
//   i1 i0 <-> t2 t1
//   j2* j1 j0 <-> t3* t0 pol
//
// load_ungridded_e(): Delta(t)=24
// write_ungridded_e(): Delta(t)=24
// grid_shared_e(): Delta(t)=48
// load_gridded_e(): Delta(t)=8
// unpack_e(): Delta(t)=2
//
// Shared memory layout:
//
//   uint E[24][259];   // first index is j, corresponds to 48 time samples
//
// The length-259 axis represents either a shape-(256,) or shape-(6,43) array,
// padded to stride 259.
//
// Assumes {32,24,1} thread grid, not {32*24,1,1}.

__device__ void double_byte_perm(uint &x, uint &y, uint s1, uint s2)
{
    uint x0 = x;
    x = __byte_perm(x0, y, s1);
    y = __byte_perm(x0, y, s2);
}

__device__ float unpack_int4(uint x, uint s)
{
    x = ((x >> s) ^ 0x88888888) & 0xf;
    return float(x) - 8.0f;
}


template<bool Debug>
struct casm_shuffle_state
{
    // Managed by setup_e_pointer(), load_ungridded_e(), write_ungridded_e()
    const uint4 *ep4;
    uint4 e4;

    // Managed by load_gridded_e(), unpack_e().
    uint e0, e1;

    // Gains in persistent registers.
    // float g0_re, g0_im, g1_re, g1_im;
    
    // 'global_e' argument: points to (T,F,2,256)
    // 'feed_weights' argument: points to (F,2,256,2)
    // 'gpu_persistent_data' argument: see "global memory layout" earlier in source file.

    __device__ casm_shuffle_state(const uint8_t *global_e, const float *feed_weights, const float *gpu_persistent_data, int nbeams)
    {
	if constexpr (Debug) {
	    assert(blockDim.x == 32);
	    assert(blockDim.y == 24);
	    assert(blockDim.z == 1);
	}

	copy_global_to_shared_memory(gpu_persistent_data, feed_weights, nbeams);
	__syncthreads();

	setup_e_pointer(global_e);
    }

    // copy_global_to_shared_memory() performs the following copies
    //
    //    gridding (256 elts): gpu_persistent_data -> shmem
    //    ns_phases (32 elts): gpu_persistent_data -> shmem (32 elts)
    //    per_frequency_data (96 elts, not 96*F elts): gpu_persistent_data -> shmem
    //    feed_weights (1024 elts, not 1024*F elts): feed_weights_global -> shmem
    //    beam_locations (2*B elts): gpu_persistent_data -> shmem
    //
    // 'feed_weights' argument: points to (F,2,256,2)
    // 'gpu_persistent_data' argument: see "global memory layout" earlier in source file.
    //
    // Note: caller is responsible for calling __syncthreads() afterwards!

    __device__ void copy_global_to_shared_memory(const float *gpu_persistent_data, const float *feed_weights, int nbeams)
    {
	extern __shared__ float shmem[];

	uint w = threadIdx.y;          // warp id
	uint l = threadIdx.x;          // lane id
	uint f = blockIdx.x;           // frequency channel
	uint F = gridDim.x;            // number of frequency channels
	uint B32 = nbeams >> 5;

	if (w < 9) {
	    // gridding + ns_phases (256+32 elts)
	    uint s = 32*w + l;
	    shmem[s] = gpu_persistent_data[s];
	    w += 32;
	}

	if (w < 12) {
	    // per_frequency_data (96 elts)
	    uint s = 32*(w-9) + l;
	    uint dst = shmem_layout::per_frequency_data_base + s;
	    uint src = gmem_layout::per_frequency_data_base(f) + s;
	    shmem[dst] = gpu_persistent_data[src];
	    w += 32;
	}
	    
	if (w < 28) {
	    // feed_weights (512 complex elements, where each "element" is a pol+dish).
	    // Note that the 'feed_weights' array ordering is feed_weights[F][512][2];
	    // It's easiest to read these as float2.
	    
	    const float2 *fw2 = (const float2 *) (feed_weights + 1024*f);  // length-512 float2
	    constexpr uint S = shmem_layout::wt_pol_stride;
	    
	    uint e = 32*(w-12) + l;  // "element" (pol+dish)
	    uint d = e & 0xff;       // dish
	    uint pol = e >> 8;       // pol
	    uint dst = shmem_layout::ungridded_wts_base + pol*S + d;

	    float2 fw = fw2[e];
	    shmem[dst] = fw.x;         // real part
	    shmem[dst + 2*S] = fw.y;   // imag part
	    shmem[dst + 4*S] = fw.x;   // duplicated real part
	    shmem[dst + 6*S] = fw.y;   // duplicated imag part
	    w += 32;
	}

	while (w < B32+28) {
	    // beam_locations (B*2 floats).
	    // It's easiest to read these as float2.
	    
	    const float2 *bl2 = (const float2 *) (gpu_persistent_data + gmem_layout::beam_locs_base(F));
	    constexpr uint S = shmem_layout::beam_stride;
			  
	    uint b = 32*(w-28) + l;   // beam id
	    uint dst = shmem_layout::beam_locs_base + b;

	    float2 bl = bl2[b];
	    shmem[dst] = bl.x;      // north-south beam location
	    shmem[dst + S] = bl.y;  // east-west beam location
	    w += 32;
	}
	
	// Important: zero 'gridded_wts' in shared memory, in order to capture the two
	// "missing" feeds. (It's easy to miss this, since it only matters if the number
	// of threadblocks is larger than the number of SMs.)

	constexpr int zstart = shmem_layout::gridded_wts_base;
	constexpr int zsize = 4 * shmem_layout::wt_pol_stride;
	
	for (uint i = 32*threadIdx.y + threadIdx.x; i < zsize; i += 24*32)
	    shmem[zstart + i] = 0.0f;

	// Note: no __syncthreads() here, caller is responsible for calling __syncthreads().
    }
    
    // These member functions manage copying the ungridded E-array from global
    // memory to shared memory, with Delta(t)=24.
    //
    //   setup_e_pointer()
    //   load_ungridded_e()
    //   write_ungridded_e()
    //
    // Data is read from global memory with (T=24, I=4, J=48, D=128):
    //
    //   b1 b0 <-> d1 d0
    //   r1 r0 <-> d3 d2
    //   l4 l3 l2 l1 l0 <-> i1 i0 d6 d5 d4   (= t2 t1 d6 d5 d4)
    //   w1* w0 <-> j0* d7                   (= ... t3* t0 pol d7)
    //
    // Then some shuffling operations are performed, to obtain:
    //
    //   b1 b0 <-> i1 i0
    //   r1 r0 <-> d6 d5
    //   l4 l3 l2 l1 l0 <-> d3 d2 d1 d0 d4
    //   w1* w0 <-> j0* d7
    //
    // before writing to shared memory.
    
    __device__ void setup_e_pointer(const uint8_t *global_e)
    {
	uint l = threadIdx.x;  // lane id
	uint w = threadIdx.y;  // warp id
	uint f = blockIdx.x;   // frequency channel for this threadblock
	uint F = gridDim.x;    // total frequency channels

	// Initialize the 'ep4' global memory pointer.
	//
	//   b1 b0 <-> d1 d0
	//   r1 r0 <-> d3 d2
	//   l4 l3 l2 l1 l0 <-> i1 i0 d6 d5 d4   (= t2 t1 d6 d5 d4)
	//   w1* w0 <-> j0* d7                   (= ... t3* t0 pol d7)

	// (warpId, laneId) -> (i, j, d16)
	uint i = (l >> 3);
	uint j = (w >> 1);
	uint d16 = (l & 0x7) | ((w & 1) << 3);

	// (i, j) -> (pol, t)
	//   i1 i0 <-> t2 t1
	//   j2* j1 j0 <-> t3* t0 pol
	
	uint pol = (j & 1);
	uint t = (i << 1) | ((j & 2) >> 1) | ((j >> 2) << 3);
	
	// uint8 E[T][F][2][256];
	// uint128 E[T][F][2][16];
	ep4 = (const uint4 *) global_e;
	ep4 += 32*(t*F+f) + 16*pol + d16;
    }
    
    __device__ void load_ungridded_e()
    {
	e4 = *ep4;
    }

    // "phase" is either 0 or 1
    __device__ void write_ungridded_e(int phase)
    {
	extern __shared__ uint shmem_u[];
	
	// At top, 'e4' has register assignment:
	//
	//   b1 b0 <-> d1 d0
	//   r1 r0 <-> d3 d2
	//   l4 l3 l2 l1 l0 <-> i1 i0 d6 d5 d4
	//   w1* w0 <-> j0* d7

	// swap (r1,r0) <-> (l4,l3)
	warp_transpose(e4.x, e4.y, 8);
	warp_transpose(e4.z, e4.w, 8);
	warp_transpose(e4.x, e4.z, 16);
	warp_transpose(e4.y, e4.w, 16);

	// b1 b0 <-> d1 d0
	// r1 r0 <-> i1 10
	// l4 l3 l2 l1 l0 <-> d3 d2 d6 d5 d4
	// w1* w0 <-> j0* d7

	// swap (b1,b0) <-> (r1,r0)
	double_byte_perm(e4.x, e4.y, 0x6240, 0x7351);
	double_byte_perm(e4.z, e4.w, 0x6240, 0x7351);
	double_byte_perm(e4.x, e4.z, 0x5410, 0x7632);
	double_byte_perm(e4.y, e4.w, 0x5410, 0x7632);
	
	// b1 b0 <-> i1 i0
	// r1 r0 <-> d1 d0
	// l4 l3 l2 l1 l0 <-> d3 d2 d6 d5 d4
	// w1* w0 <-> j0* d7

	// swap (r1,r0) <-> (l2,l1)
	warp_transpose(e4.x, e4.y, 2);
	warp_transpose(e4.z, e4.w, 2);
	warp_transpose(e4.x, e4.z, 4);
	warp_transpose(e4.y, e4.w, 4);
	
	// At bottom, we have the register assignment:
	//
	//   b1 b0 <-> i1 i0
	//   r1 r0 <-> d6 d5
	//   l4 l3 l2 l1 l0 <-> d3 d2 d1 d0 d4   (note permutation)
	//   w1* w0 <-> j0* d7
	//
	// Compute the shared memory offset 's' (see shared memory layout above).
	
	constexpr int SE = shmem_layout::E_base;
	constexpr int SJ = shmem_layout::E_jstride;
	
	uint l = threadIdx.x;  // lane id
	uint w = threadIdx.y;  // warp id
	uint j = (w >> 1);
	uint d = (l >> 1) | ((l & 0x1) << 4) | ((w & 0x1) << 7);
	uint s = SE + ((12*phase+j) * SJ) + d;
	check_bank_conflict_free<Debug>(s);
	
	shmem_u[s] = e4.x;
	shmem_u[s + 32] = e4.y;
	shmem_u[s + 64] = e4.z;
	shmem_u[s + 96] = e4.w;

	// Advance global E-array pointer 'ep4'.
	
	uint F = gridDim.x;    // total frequency channels
	ep4 += 24*32*F;        // advance by 24 time samples
    }

    // grid_shared_e(): "gridding" the E-array in shared memory.
    //
    // A little awkward, since we want to loop over 256 dishes with 24 warps.
    // Note that 256 = 10*240 + 16.
    
    __device__ void grid_shared_e()
    {
	extern __shared__ uint shmem_u[];
	
	uint e[11];
	uint j = threadIdx.x;  // lane
	uint w = threadIdx.y;  // warp
	uint d0 = (w < 16) ? (11*w) : (10*w+16);
	uint s = shmem_layout::E_base + 259*j;

	#pragma unroll
	for (int i = 0; i < 10; i++)
	    e[i] = (j < 24) ? shmem_u[s+d0+i] : 0;
	
	if (w < 16)
	    e[10] = (j < 24) ? shmem_u[s+d0+10] : 0;
	
	__syncthreads();

	// FIXME temporary convenience in testing!
	for (int i = 32*w + j; i < 24*259; i += 24*32)
	    shmem_u[i + shmem_layout::E_base] = 0;
	__syncthreads();
	
	#pragma unroll
	for (int i = 0; i < 10; i++) {
	    uint dst = s + shmem_u[d0+i];  // 'gridding' (see shared memory layout above)
	    if (j < 24)
		shmem_u[dst] = e[i];
	}

	if (w < 16) {
	    uint dst = s + shmem_u[d0+10];  // 'gridding' (see shared memory layout above)
	    if (j < 24)
		shmem_u[dst] = e[10];
	}
    }


#if 0
    __device__ void setup_gains()
    {
	// Shared memory workspace (~22 KB!)
	//
	//   uint gridding[256];
	//   float ungridded_gains[2][2][2][256];  // (dummy,reim,pol,dish) with strides (
	//   float gridded_gains[2][2][6*43];      // (reim,pol,dish) with reim-stride = (2*6*43+1)

	// Step 1: grid the gains.
	// Each dish is processed by a group of 4 warps.
	uint d0 = (32*threadIdx.y + threadIdx.x) >> 2;

	for (uint d = d0; d < 256; d += 24*8) {
	    
	}
    }
#endif

    // The member functions
    //
    //   load_gridded_e()
    //   unpack_e()
    //
    // manage reading the int4+4 gridded E-array from shared memory with
    // Delta(t)=8, and "unpacking" to float registers with Delta(t)=2.
    //
    // The int4+4 E-array uses register assignment (T=8, P=2, EW=6, NS=64):
    //
    //   b2 b1 b0 <-> t2 t1 ReIm     (= i1 i0 ReIm)
    //   r0 <-> ns5
    //   l4 l3 l2 l1 l0 <-> ns4 ns3 ns2 ns1 ns0
    //   w2* w1 w0 <-> ew t0 pol  (= j1 j0)
    //
    // and is "unpacked" to floats with assignment (T=2, P=2, EW=6, NS=64):
    //
    //   r1 r0 <-> ns5 ReIm
    //   l4 l3 l2 l1 l0 <-> ns4 ns3 ns2 ns1 ns0
    //   w2* w1 w0 <-> ew t0 pol  (= j1 j0)

    // "phase" should satisfy 0 <= phase < 6.
    __device__ void load_gridded_e(int phase)
    {
	extern __shared__ uint shmem_u[];
	
	uint j = (threadIdx.y & 3);
	uint ew = (threadIdx.y >> 2);
	uint ns = threadIdx.x;
	uint s = shmem_layout::E_base + (4*phase+j)*shmem_layout::E_jstride + 43*ew + ns;
	
	e0 = shmem_u[s];
	e1 = (threadIdx.x < 11) ? shmem_u[s+32] : 0;
    }
    
    __device__ void unpack_e(int i, float &e0_re, float &e0_im, float &e1_re, float &e1_im)
    {
	// i is the length-4 index (t2 t1).
	// FIXME save a few cycles by offset encoding earlier?

	e0_re = unpack_int4(e0, 8*i);
	e0_im = unpack_int4(e0, 8*i+4);
	e1_re = unpack_int4(e1, 8*i);
	e1_im = unpack_int4(e1, 8*i+4);
    }
};


// Launch with {32,24,1} threads and {F,1,1} blocks.
// T must be a multiple of 48.

__global__ void casm_shuffle_test_kernel(
    const uint8_t *e_in,                // (T,F,2,D) = (time,freq,pol,dish)
    const float *feed_weights,          // (F,2,256,2) = (freq,pol,dish,reim)
    const float *gpu_persistent_data,   // see "global memory layout" earlier in source file
    float *out,                         // (T,F,2,6,64,2) = (time,freq,pol,ew,ns,reim)
    int T)
{
    assert(blockDim.x == 32);
    assert(blockDim.y == 24);
    assert(blockDim.z == 1);
    assert(gridDim.y == 1);
    assert(gridDim.z == 1);
    assert((T % 48) == 0);

    // Debug=true, nbeams=0
    casm_shuffle_state<true> shuffle(e_in, feed_weights, gpu_persistent_data, 0);

    // Set up writing to the 'out' array, with Delta(t)=2.
    // Output from casm_shuffle_state::unpack_e() has register assignment:
    //
    //   r1 r0 <-> ns5 ReIm
    //   l4 l3 l2 l1 l0 <-> ns4 ns3 ns2 ns1 ns0
    //   w2* w1 w0 <-> ew t0 pol

    uint f = blockIdx.x;   // frequency channel of threadblock
    uint F = gridDim.x;    // total frequency channels
    uint pol = (threadIdx.y & 1);
    uint t0 = (threadIdx.y & 2) >> 1;
    uint ew = (threadIdx.y >> 2);
    uint ns = threadIdx.x;

    // float out[T][F][2][6][64][2];   // time,freq,pol,ew,ns,reim
    out += 2 * ns;
    out += 64*2 * ew;
    out += 6*64*2 * pol;
    out += 2*6*64*2 * f;
    out += F*2*6*64*2 * t0;
    
    for (int touter = 0; touter < T; touter += 48) {
	for (int s = 0; s < 2; s++) {
	    // Delta(t)=24, Delta(j)=12
	    shuffle.load_ungridded_e();
	    shuffle.write_ungridded_e(s);
	}

	__syncthreads();
	
	shuffle.grid_shared_e();
	
	__syncthreads();

	for (int s = 0; s < 6; s++) {
	    // Delta(t)=8, Delta(j)=4
	    shuffle.load_gridded_e(s);
	    
	    for (int i = 0; i < 4; i++) {
		// Delta(t)=2
		float e0_re, e0_im, e1_re, e1_im;
		shuffle.unpack_e(i, e0_re, e0_im, e1_re, e1_im);
		
		// r1 r0 <-> ns5 ReIm
		// l4 l3 l2 l1 l0 <-> ns4 ns3 ns2 ns1 ns0
		// w2* w1 w0 <-> ew t0 pol

		// float out[T][F][2][6][64][2]
		out[0] = e0_re;
		out[1] = e0_im;
		out[64] = e1_re;
		out[65] = e1_im;

		// Delta(t)=2
		out += (2*F*2*6*64*2);
	    }
	}

	__syncthreads();
    }
}


void casm_shuffle_reference_kernel(
    const CasmBeamformer &bf,
    const uint8_t *e_in,                // (T,F,2,D) = (time,freq,pol,dish)
    const float *feed_weights,          // (F,2,256,2) = (freq,pol,dish,reim)
    float *out,                         // (T,F,2,6,64,2) = (time,freq,pol,ew,ns,reim)
    int T)
{
    int TFP = 2 * T * bf.F;
    memset(out, 0, TFP * 6*64*2 * sizeof(float));

    xassert(bf.feed_indices.is_fully_contiguous());
    const int *feed_indices = bf.feed_indices.data;
    
    for (int tfp = 0; tfp < TFP; tfp++) {
	const uint8_t *e2 = e_in + 256*tfp;  // points to shape (256,)
	float *out2 = out + 6*64*2*tfp;    // points to shape (6,64,2)

	for (int d = 0; d < 256; d++) {
	    uint8_t e = e2[d] ^ 0x88888888;
	    float e_re = float(e & 0xf) - 8.0f;
	    float e_im = float(e >> 4) - 8.0f;

	    int ns = feed_indices[2*d];
	    int ew = feed_indices[2*d+1];
	    int g = 64*ew + ns;

	    out2[2*g] = e_re;
	    out2[2*g+1] = e_im;
	}
    }
}


// FIXME should this be a CasmBeamformer member function?
static Array<uint8_t> make_random_e_array(int T, int F)
{
    Array<uint8_t> ret({T,F,2,256}, af_rhost | af_zero);

    uint *p = (uint *) (ret.data);
    int n = (ret.size >> 2);

    for (int i = 0; i < n; i++) {
	uint t1 = ksgpu::default_rng();
	uint t2 = ksgpu::default_rng();
	p[i] = t1 ^ (t2 << 16);
    }

    return ret;
}


// FIXME should this be a CasmBeamformer member function?
static Array<float> make_random_feed_weights(int F)
{
    return Array<float> ({F,2,256,2}, af_rhost | af_random);
}


void test_casm_shuffle(const CasmBeamformer &bf)
{    
    static bool flag = false;

    if (!flag) {
        CUDA_CALL(cudaFuncSetAttribute(
	    casm_shuffle_test_kernel,
            cudaFuncAttributeMaxDynamicSharedMemorySize,
            99 * 1024
        ));
	flag = true;
    }
    
    int F = bf.F;
    int T = bf.nominal_Tin_for_unit_tests;
    xassert(T > 0);
    
    Array<uint8_t> e = make_random_e_array(T,F);
    Array<float> feed_weights = make_random_feed_weights(F);
    Array<float> out_cpu({T,F,2,6,64,2}, af_random | af_rhost);
    Array<float> out_gpu({T,F,2,6,64,2}, af_random | af_gpu);
    
    casm_shuffle_reference_kernel(bf, e.data, feed_weights.data, out_cpu.data, T);

    e = e.to_gpu();
    feed_weights = feed_weights.to_gpu();
    casm_shuffle_test_kernel<<< F, {32,24,1}, 99*1024, 0 >>> (e.data, feed_weights.data, bf.gpu_persistent_data.data, out_gpu.data, T);
    CUDA_PEEK("casm_shuffle_test_kernel");

    assert_arrays_equal(out_cpu, out_gpu, "cpu", "gpu", {"t","f","p","ew","ns","reim"});
    cout << "test_casm_shuffle(T=" << T << ",F=" << F << "): pass" << endl;
}


// -------------------------------------------------------------------------------------------------
//
// fft_c2c (helper for "FFT1" kernel)
//
// FIXME implementing fft_c2c_state<2> could save two persistent registers and a few FMAs


__device__ void fft0(float &xre, float &xim)
{
    float t = xre - xim;
    xre += xim;
    xim = t;
}


template<int R>
struct fft_c2c_state
{
    // Implements a c2c FFT with 2^R elements.
    //
    // Input register assignment:
    //   r1 r0 <-> x_{r-1} ReIm
    //   t4 t3 t2 t1 t0 <-> s_{5-r} ... s0 x_{r-2} ... x_0
    //
    // Output register assignment:
    //   r1 r0 <-> y_{r-1} ReIm
    //   t4 t3 t2 t1 t0 <-> s_{5-r} ... s0 y_0 ... y_{r-2}

    fft_c2c_state<R-1> next_fft;
    float cre, cim;
    
    __device__ fft_c2c_state()
    {
	constexpr float a = 6.283185307f / (1<<R);   // 2*pi/2^r
	uint t = threadIdx.x & ((1 << (R-1)) - 1);   // 0 <= t < 2^{r-1}
	sincosf(a*t, &cim, &cre);                    // phase is exp(2*pi*t / 2^r)
    }

    __device__ void apply(float  &x0_re, float &x0_im, float &x1_re, float &x1_im)
    {
	// (x0,x1) = (x0+x1,x0-x1)
	fft0(x0_re, x1_re);
	fft0(x0_im, x1_im);

	// x1 *= phase
	float yre = cre * x1_re - cim * x1_im;
	float yim = cim * x1_re + cre * x1_im;
	x1_re = yre;
	x1_im = yim;

	// swap "01" register bit with thread bit (R-2)
	warp_transpose(x0_re, x1_re, (1 << (R-2)));
	warp_transpose(x0_im, x1_im, (1 << (R-2)));
	
	next_fft.apply(x0_re, x0_im, x1_re, x1_im);
    }
};


template<>
struct fft_c2c_state<1>
{
    __device__ void apply(float  &x0_re, float &x0_im, float &x1_re, float &x1_im)
    {
	fft0(x0_re, x1_re);
	fft0(x0_im, x1_im);
    }
};


// Call with {1,32} threads.
// Input and output arrays have shape (2^(6-R), 2^R, 2)
template<int R>
__global__ void fft_c2c_test_kernel(const float *in, float *out)
{
    // Input register assignment:
    //   r1 r0 <-> x_{r-1} ReIm
    //   t4 t3 t2 t1 t0 <-> s_{5-r} ... s0 x_{r-2} ... x_0
    
    int ss = threadIdx.x >> (R-1);            // spectator index
    int sx = threadIdx.x & ((1<<(R-1)) - 1);  // x-index
    int sin = (ss << (R+1)) | (sx << 1);
    
    float x0_re = in[sin];
    float x0_im = in[sin + 1];
    float x1_re = in[sin + (1<<R)];
    float x1_im = in[sin + (1<<R) + 1];

    fft_c2c_state<R> fft;
    fft.apply(x0_re, x0_im, x1_re, x1_im);

    // Output register assignment:
    //   r1 r0 <-> y_{r-1} ReIm
    //   t4 t3 t2 t1 t0 <-> s_{5-r} ... s0 y_0 ... y_{r-2}

    int sy = __brev(threadIdx.x << (33-R));
    int sout = (ss << (R+1)) | (sy << 1);

    out[sout] = x0_re;
    out[sout + 1] = x0_im;
    out[sout + (1<<R)] = x1_re;
    out[sout + (1<<R) + 1] = x1_im;
}


void test_casm_fft_c2c()
{
    constexpr int R = 6;
    constexpr int N = (1 << R);
    constexpr int S = (1 << (6-R));
    
    Array<float> in({S,N,2}, af_random | af_rhost);
    Array<float> out_cpu({S,N,2}, af_zero | af_rhost);
    Array<float> out_gpu({S,N,2}, af_random | af_gpu);
    
    for (int j = 0; j < N; j++) {
	for (int k = 0; k < N; k++) {
	    float theta = (2*M_PI/N) * ((j*k) % N);
	    float cth = cosf(theta);
	    float sth = sinf(theta);
	    
	    for (int s = 0; s < S; s++) {
		float xre = in.at({s,k,0});
		float xim = in.at({s,k,1});
		
		out_cpu.at({s,j,0}) += (cth*xre - sth*xim);
		out_cpu.at({s,j,1}) += (sth*xre + cth*xim);
	    }
	}
    }

    in = in.to_gpu();
    fft_c2c_test_kernel<R> <<<1,32>>> (in.data, out_gpu.data);
    CUDA_PEEK("fft_c2c_test_kernel");

    assert_arrays_equal(out_cpu, out_gpu, "cpu", "gpu", {"s","i","reim"}, 1.0e-3);
    cout << "test_casm_fft_c2c: pass" << endl;
}


// -------------------------------------------------------------------------------------------------
//
// FFT1
//
// Implements a zero-padded c2c FFT with 64 inputs and 128 outputs
//
// Input register assignment:
//   r1 r0 <-> x5 ReIm
//   t4 t3 t2 t1 t0 <-> x4 x3 x2 x1 x0
//
// Outputs are written to shared memory:
//   float G[W][2][128]   strides (257,128,1)   W=warps per threadblock
//
// NOTE: assumes that threads are a {32,W,1} grid (not a {32*W,1,1} grid).


template<bool Debug>
struct fft1_state
{
    fft_c2c_state<6> next_fft;
    float cre, cim;   // "twiddle" factor exp(2*pi*i t / 128)
    int sbase;        // shared memory offset

    __device__ fft1_state()
    {
	if constexpr (Debug) {
	    assert(blockDim.x == 32);
	    assert(blockDim.z == 1);
	}
	
	float x = 0.04908738521234052f * threadIdx.x;   // constant is (2pi)/128
	sincosf(x, &cim, &cre);                         // note ordering (im,re)

	// Shared memory writes will use register assignment (see below):
	//   t4 t3 t2 t1 t0 <-> y1 y2 y3 y4 y0
	//
	// The 'sbase' offset assumes y5=y6=ReIm=0
	
	sbase = (threadIdx.y) * 257;              // warp id
	sbase += (threadIdx.x & 1);               // t0 <-> y0
	sbase += __brev(threadIdx.x >> 1) >> 27;  // t4 t3 t2 t1 <-> y1 y2 y3 y4
	
	check_bank_conflict_free<Debug> (sbase);
    }
    
    __device__ void apply(float x0_re, float x0_im, float x1_re, float x1_im, float *sp)
    {
	// y0 = x0 * exp(2*pi*i t / 128) = c*x0
	// y1 = x1 * exp(2*pi*i (t+32) / 128) = i*c*x0
	//   where t = 0, ..., 31
	
	float y0_re = cre*x0_re - cim*x0_im;
	float y0_im = cim*x0_re + cre*x0_im;

	float y1_re = -cim*x1_re - cre*x1_im;
	float y1_im = cre*x1_re - cim*x1_im;

	// xy r1 r0 <-> y0 x5 ReIm
	// t4 t3 t2 t1 t0 <-> x4 x3 x2 x1 x0

	next_fft.apply(x0_re, x0_im, x1_re, x1_im);
	next_fft.apply(y0_re, y0_im, y1_re, y1_im);
	
	// xy r1 r0 <-> y0 y6 ReIm
	// t4 t3 t2 t1 t0 <-> y1 y2 y3 y4 y5

	// Exchange "xy" and "thread 0" bits
	warp_transpose(x0_re, y0_re, 1);
	warp_transpose(x0_im, y0_im, 1);
	warp_transpose(x1_re, y1_re, 1);
	warp_transpose(x1_im, y1_im, 1);
	
	// xy r1 r0 <-> y5 y6 ReIm
	// t4 t3 t2 t1 t0 <-> y1 y2 y3 y4 y0

	// Strides: xy=32, 01=64, ReIm=128
	sp[sbase] = x0_re;
	sp[sbase+32] = y0_re;
	sp[sbase+64] = x1_re;
	sp[sbase+96] = y1_re;
	sp[sbase+128] = x0_im;
	sp[sbase+160] = y0_im;
	sp[sbase+192] = x1_im;	
	sp[sbase+224] = y1_im;	
    }
};


// Call with {W,32} threads.
// Input array has shape (W,64,2).
// Output array has shape (W,128,2).

template<int W>
__global__ void fft1_test_kernel(const float *in, float *out)
{
    __shared__ float shmem[W*257];

    // Input register assignment:
    //   r1 r0 <-> x5 ReIm
    //   t4 t3 t2 t1 t0 <-> x4 x3 x2 x1 x0

    int w = threadIdx.y;  // warp id
    float x0_re = in[128*w + 2*threadIdx.x];
    float x0_im = in[128*w + 2*threadIdx.x + 1];
    float x1_re = in[128*w + 2*threadIdx.x + 64];
    float x1_im = in[128*w + 2*threadIdx.x + 65];

    fft1_state<true> fft1;  // Debug=true
    fft1.apply(x0_re, x0_im, x1_re, x1_im, shmem);

    for (int reim = 0; reim < 2; reim++)
	for (int y = threadIdx.x; y < 128; y += 32)
	    out[256*w + 2*y + reim] = shmem[257*w + 128*reim + y];
}


void test_casm_fft1()
{
    static constexpr int W = 24;

    Array<float> in({W,64,2}, af_random | af_rhost);
    Array<float> out_cpu({W,128,2}, af_zero | af_rhost);
    Array<float> out_gpu({W,128,2}, af_random | af_gpu);

    for (int j = 0; j < 128; j++) {
	for (int k = 0; k < 64; k++) {
	    float theta = (2*M_PI/128) * ((j*k) % 128);
	    float cth = cosf(theta);
	    float sth = sinf(theta);
	    
	    for (int w = 0; w < W; w++) {
		out_cpu.at({w,j,0}) += (cth * in.at({w,k,0})) - (sth * in.at({w,k,1}));
		out_cpu.at({w,j,1}) += (sth * in.at({w,k,0})) + (cth * in.at({w,k,1}));
	    }
	}
    }

    in = in.to_gpu();
    fft1_test_kernel<W> <<<1,{32,W,1}>>> (in.data, out_gpu.data);
    CUDA_PEEK("fft1_test_kernel");
    
    assert_arrays_equal(out_cpu, out_gpu, "cpu", "gpu", {"w","i","reim"}, 1.0e-3);
    cout << "test_casm_fft1: pass" << endl;
}


// -------------------------------------------------------------------------------------------------
//
// FFT2
//
// There are 24 east-west beams 0 <= b < 24 and 6 east-west feeds 0 <= f < 6.
//
// Alternate parameterization: (bouter,binner) and (fouter,finner) where:
//
//   b = bouter ? (12+binner) : (11-binner)   0 <= bouter < 2    0 <= binner < 12
//   f = fouter ? (3+finner)  : (2-finner)    0 <= fouter < 2    0 <= finner < 3
//
// This parameterization is defined so that if we flip 'bouter' at fixed 'binner',
// the beam location goes to its negative, and likewise for feeds.
//
// We sometimes write 'binner' as:
//
//   binner = 4*b2 + 2*b1 + b0     0 <= b2 < 3    0 <= b1 < 2    0 <= b0 < 2
//
// NS beam locations are represented by an index 0 <= ns < 128,
// which we sometimes decompose into its base-2 digits [ns6,...,ns0].
// These are spectator indices, as far as the FFT2 kernel is concerned.
//
// We store EW beamforming phases for fouter=bouter=0 only, since flipping
// 'bouter' or 'fouter' sends the phase to its complex conjugate. The phase
// can be written in the form (where alpha[3] is a kernel argument):
//
//   phase[binner,finner] = exp( i * (1+2*binner) * alpha[finner] )
//
// The I-array is distributed as follows:
//
//   r1 r0 <-> (bouter) (b1)
//   t4 t3 t2 t1 t0 <-> (ns4) (ns3) (ns2) (ns1) (b0)
//   24 warps <-> (b2) (ns6) (ns5) (ns0)
//
// The G-array shared memory layout is (where "f" is an EW feed):
//
//   float G[6][2][128];  // (f,reim,ns), strides (257,128,1)
//
// We read from the G-array in register assignment:
//
//   t4 t3 t2 t1 t0 <-> (ns4) (ns3) (ns2) (ns1) (fouter)
//
// This is bank conflict free, since flipping 'fouter' always produces
// an odd change in f, and the f-stride is odd (257).
//
// The I-array shared memory layout is (where "b" indexes an EW beam):
//
//   float I[24][128];   // (b,ns), strides (133,1)
//
// We write to the I-array in the bank conflict free register assignment:
//
//   t4 t3 t2 t1 t0 <-> (ns4) (ns3) (ns2) (ns1) (b0)


template<bool Debug>
struct fft2_state
{
    // Note: we use 12 persistent registers/thread to store beamforming
    // phases, but the number of distinct phases is 24/warp or 72/block.
    // Maybe better to distribute registers as needed with __shfl_sync()?
     
    float I[2][2];      // beams are indexed by (bouter, b1)
    float pcos[2][3];   // beamforming phases are indexed by (b1, finner)
    float psin[2][3];   // beamforming phases are indexed by (b1, finner)
    
    int soff_g;     // base shared memory offset in G-array.
    int soff_i0;    // base shared memory offset in I-array, bouter=0
    int soff_i1;    // base shared memory offset in I-array, bouter=1


    // FIXME is alpha[3] the best interface here?
    __device__ fft2_state(float alpha[3])
    {
	if constexpr (Debug) {
	    assert(blockDim.x == 32);
	    assert(blockDim.y == 24);
	    assert(blockDim.z == 1);
	}
	
	I[0][0] = I[0][1] = I[1][0] = I[1][1] = 0.0f;

	// Each warp maps to an (b2, ns56, ns0) triple.
	uint ns0 = (threadIdx.y & 1);        // 0 <= ns0 < 2
	uint ns56 = (threadIdx.y >> 1) & 3;  // 0 <= ns56 < 4
	uint b2 = (threadIdx.y >> 3);        // 0 <= b2 < 3

	// Each thread maps to a (ns14, b0) pair.
	uint b0 = (threadIdx.x & 1);
	uint ns14 = (threadIdx.x >> 1);

	// Beamforming phases are indexed by (b1, finner)
	#pragma unroll
	for (uint b1 = 0; b1 < 2; b1++) {
	    // Beamforming phase is exp(i*t*alpha[finner]) where t = 1+2*binner
	    float t = (b2 << 3) | (b1 << 2) | (b0 << 1) | 1;
	    
	    #pragma unroll
	    for (uint finner = 0; finner < 3; finner++)
		sincosf(t * alpha[finner], &psin[b1][finner], &pcos[b1][finner]);
	}

	// Shared memory offset for reading G-array:
	//
	//   float G[6][2][128];  // (f,reim,ns), strides (257,128,1)
	//
	// When we read from the G-array, we read it as:
	//
	//   t4 t3 t2 t1 t0 <-> (ns4) (ns3) (ns2) (ns1) (fouter)
	//
	// 'soff_g' is the offset assuming finner=reim=0.
	
	uint fouter = threadIdx.x & 1;
	uint ns = (ns56 << 5) | (ns14 << 1) | ns0;
	soff_g = (fouter ? (2*257) : (3*257)) + ns;

	// Shared memory offset for writing I-array:
	//
	//   float I[24][128];   // (b,ns), strides (133,1)
	//
	// When we write to the I-array, we write as
	//
	//   t4 t3 t2 t1 t0 <-> (ns4) (ns3) (ns2) (ns1) (b0)
	//
	// 'soff_i{0,1}' is the offset with bouter={0,1} and b1=0.

	uint binner = (b2 << 2) | b0;
	soff_i0 = 133*(12+binner) + ns;  // offset for bouter=0
	soff_i1 = 133*(11-binner) + ns;  // offset for binner=1

	check_bank_conflict_free<Debug> (soff_i0);
	check_bank_conflict_free<Debug> (soff_i1);
    }

    
    // Accumulates one (time,pol) into I-registers.
    __device__ void apply(const float *sp)
    {
	// Beamformed electric fields are accumulated here.
	float Fre[2][2];   // (bouter, b1)
	float Fim[2][2];   // (bouter, b1)

	Fre[0][0] = Fre[0][1] = Fre[1][0] = Fre[1][1] = 0.0f;
	Fim[0][0] = Fim[0][1] = Fim[1][0] = Fim[1][1] = 0.0f;

	// finner-stride in the G-array
	int fouter = threadIdx.x & 1;
	int ds = fouter ? (-257) : 257;
	
        #pragma unroll
	for (int finner = 0; finner < 3; finner++) {
	    int s = soff_g + finner*ds;
	    check_bank_conflict_free<Debug> (s);
	    
	    float tre = sp[s];
	    float tim = sp[s + 128];

	    // FIXME can be improved.
	    // u{0,1} index is fouter.
	    float u0_re = __shfl_sync(0xffffffff, tre, threadIdx.x & ~1);
	    float u1_re = __shfl_sync(0xffffffff, tre, threadIdx.x | 1);
	    float u0_im = __shfl_sync(0xffffffff, tim, threadIdx.x & ~1);
	    float u1_im = __shfl_sync(0xffffffff, tim, threadIdx.x | 1);

	    #pragma unroll
	    for (int b1 = 0; b1 < 2; b1++) {
		// FIXME can be sped up with FFT-style trick.

		// F[0][b1] += (phase) (u0)
		// F[0][b1] += (phase^*) (u1)
		// F[1][b1] += (phase^*) (u0)
		// F[1][b1] += (phase) (u1)
		
		// FIXME I don't think zma() will be called here, in the final kernel.
		zma(Fre[0][b1], Fim[0][b1], pcos[b1][finner],  psin[b1][finner], u0_re, u0_im);
		zma(Fre[0][b1], Fim[0][b1], pcos[b1][finner], -psin[b1][finner], u1_re, u1_im);  // note (-psin)
		zma(Fre[1][b1], Fim[1][b1], pcos[b1][finner], -psin[b1][finner], u0_re, u0_im);  // note (-psin)
		zma(Fre[1][b1], Fim[1][b1], pcos[b1][finner],  psin[b1][finner], u1_re, u1_im);
	    }
	}

	#pragma unroll
	for (int bouter = 0; bouter < 2; bouter++) {
	    #pragma unroll
	    for (int b1 = 0; b1 < 2; b1++)
		I[bouter][b1] += (Fre[bouter][b1] * Fre[bouter][b1]) + (Fim[bouter][b1] * Fim[bouter][b1]);
	}
    }

    // Writes I[] register to shared memory and zeroes the registers.
    __device__ void write_and_reset(float *sp)
    {
	// Beams are indexed by (bouter, b1).
	//   float I[24][128];   // (b,ns), strides (133,1)

	sp[soff_i0] = I[0][0];
	sp[soff_i1] = I[1][0];
	
	sp[soff_i0 + 2*133] = I[0][1];
	sp[soff_i1 - 2*133] = I[1][1];

	I[0][0] = I[0][1] = I[1][0] = I[1][1] = 0.0f;
    }
};


// float G[TP][6][128][2];
// float I[24][128];
// Launch with {32,24,1} threads.

__global__ void fft2_test_kernel(const float *gp, float *ip, int TP, const float *alpha)
{
    __shared__ float shmem_g[6*257];   // size (6,2,128), strides (257,128,1)
    __shared__ float shmem_i[24*133];  // size (24,128), strides (133,1)

    assert(blockDim.x == 32);
    assert(blockDim.y == 24);
    assert(blockDim.z == 1);

    float a[3];
    a[0] = alpha[0];
    a[1] = alpha[1];
    a[2] = alpha[2];
    
    fft2_state<true> fft2(a);  // Debug=true
    
    // Set up G-array copy (global) -> (shared)
    int gns = ((threadIdx.y & 3) << 5) + threadIdx.x;
    int gew = (threadIdx.y >> 2);
    int gsh = (257*gew) + gns;   // array offset in shmem_g[] array
    float2 *gp2 = (float2 *)(gp) + 128*gew + gns;  // per-(warp+thread) offsets applied
    
    for (int tp = 0; tp < TP; tp++) {
	// G-array copy (global) -> (shared)
	float2 g = *gp2;
	shmem_g[gsh] = g.x;      // real part
	shmem_g[gsh+128] = g.y;  // imag part
	gp2 += 6*128;

	__syncthreads();

	fft2.apply(shmem_g);

	__syncthreads();
    }

    fft2.write_and_reset(shmem_i);
    __syncthreads();
	
    // Set up I-array copy (shared) -> (global)
    int goff = 128*threadIdx.y + threadIdx.x;  // array offset in ip[] global array
    int soff = 133*threadIdx.y + threadIdx.x;  // array offset in shmem_i[] shared array
    
    for (int j = 0; j < 4; j++)
	ip[goff + 32*j] = shmem_i[soff + 32*j];
}


// float G[TP][6][128][2];
// float I[24][128];

void fft2_reference_kernel(const float *gp, float *ip, int TP, const float *alpha)
{
    // beamforming phase is exp(i * bloc[b] * floc[f])
    float bloc[24];
    float floc[6];

    for (int binner = 0; binner < 12; binner++) {
	bloc[12+binner] = 1 + 2*binner;
	bloc[11-binner] = -(1 + 2*binner);
    }

    for (int finner = 0; finner < 3; finner++) {
	floc[3+finner] = alpha[finner];
	floc[2-finner] = -alpha[finner];
    }
    
    for (int ns = 0; ns < 128; ns++) {
	for (int b = 0; b < 24; b++) {
	    float I = 0.0f;
	    
	    for (int tp = 0; tp < TP; tp++) {
		float zre = 0.0f;
		float zim = 0.0f;

		for (int f = 0; f < 6; f++) {
		    float xre = cosf(bloc[b] * floc[f]);
		    float xim = sinf(bloc[b] * floc[f]);
		    float yre = gp[6*256*tp + 256*f + 2*ns];
		    float yim = gp[6*256*tp + 256*f + 2*ns + 1];
		    
		    zre += xre*yre - xim*yim;
		    zim += xre*yim + xim*yre;
		}

		I += (zre*zre + zim*zim);
	    }
	    
	    ip[128*b + ns] = I;
	}
    }
}


void test_casm_fft2()
{
    int TP = 4;
    Array<float> g({TP,6,128,2}, af_rhost | af_random);
    Array<float> i_cpu({24,128}, af_rhost | af_random);
    Array<float> i_gpu({24,128}, af_gpu | af_random);
    Array<float> alpha({3}, af_rhost | af_random);
    
    fft2_reference_kernel(g.data, i_cpu.data, TP, alpha.data);
    
    g = g.to_gpu();
    alpha = alpha.to_gpu();
    
    fft2_test_kernel<<< 1, {32,24,1} >>> (g.data, i_gpu.data, TP, alpha.data);
    CUDA_PEEK("fft2_test_kernel");

    assert_arrays_equal(i_cpu, i_gpu, "cpu", "gpu", {"b","ns"}, 1.0e-4);
    cout << "test_casm_fft2: pass" << endl;
}


// -------------------------------------------------------------------------------------------------
//
// Interpolation


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
	check_bank_conflict_free<Debug> (s, 2);   // at most 2:1 bank conflict
	
	float dx = __shfl_sync(0xffffffff, dx_g, src_lane);
	float dy = __shfl_sync(0xffffffff, dy_g, src_lane);
	
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


// -------------------------------------------------------------------------------------------------


void test_casm()
{
    CasmBeamformer bf = CasmBeamformer::make_random();
    
    test_casm_shuffle(bf);
    test_casm_fft_c2c();
    test_casm_fft1();
    test_casm_fft2();
    test_casm_interpolation();
}


}  // namespace pirate
