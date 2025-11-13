#include "../include/pirate/CasmBeamformer.hpp"

#include <cassert>
#include <iomanip>
#include <sys/time.h>
#include <ksgpu/cuda_utils.hpp>
#include <ksgpu/rand_utils.hpp>

using namespace std;
using namespace ksgpu;

namespace pirate {
#if 0
}  // editor auto-indent
#endif


// -------------------------------------------------------------------------------------------------
//
// CUDA utils


// Setup: suppose we have a pair of registers [x,y] on each thread,
// which represent a 64-element array with register assignment:
//
//   l0 l1 l2 l3 l4 <-> i0 i1 i2 i3 i4   r <-> j
//
// Calling warp_transpose(x, y, 1 << N) will swap bits i_N and j,
// so that the register assignment is:
//
//   l0 l1 l2 l3 l4 <-> i0 ... i_{N-1} j i_{N+1} ... i4    r <-> i_N

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
__device__ void check_bank_conflict_free(int offset_32bit, int max_conflicts=1)
{
    if constexpr (Debug) {
        uint m = __match_any_sync(0xffffffff, offset_32bit & 31);
        assert(__popc(m) <= max_conflicts);
        assert(offset_32bit >= 0);
    }
}


// Sets max shared memory size of a GPU kernel to 99KB (max possible on Ada architecture).
template<typename F>
struct shmem_99kb
{
    bool flag = false;
    F *func = nullptr;
    
    inline shmem_99kb(const F &func_) : func(func_) { }
    
    inline void set()
    {
        if (!flag) {
            CUDA_CALL(cudaFuncSetAttribute(
                func,
                cudaFuncAttributeMaxDynamicSharedMemorySize,
                99 * 1024
            ));
            flag = true;
        }
    }
};


// -------------------------------------------------------------------------------------------------
//
// RNG utils


static std::mt19937 casm_rng;

static inline long rand_int(long lo, long hi)
{
    assert(lo < hi);
    return std::uniform_int_distribution<long>(lo,hi-1)(casm_rng);
}

static inline float rand_uniform(float lo=0.0, float hi=1.0)
{
    return std::uniform_real_distribution<float>(lo,hi) (casm_rng);
}

static vector<double> random_doubles_with_fixed_sum(int n, double sum)
{
    if (n < 1)
        throw runtime_error("random_doubles_with_fixed_sum(): 'n' argument must be >= 1");
    if (sum <= 0.0)
        throw runtime_error("random_doubles_with_fixed_sum(): 'sum' argument must be > 0");

    vector<double> ret(n);

    for (int i = 0; i < n-1; i++) {
        double t = rand_uniform();
        double p = 1.0 / double(n-1-i);
        ret[i] = sum * (1 - pow(t,p));
        sum -= ret[i];
    }

    ret[n-1] = sum;
    return ret;
}

static vector<long> random_integers_with_bounded_product(int n, long bound)
{
    if (n < 1)
        throw runtime_error("random_integers_with_bounded_product(): 'n' argument must be >= 1");
    if (bound < 1)
        throw runtime_error("random_integers_with_bounded_product(): 'bound' argument must be >= 1");

    double target_log = log(rand_uniform(1.01, bound+0.99));
    vector<double> x = random_doubles_with_fixed_sum(n, target_log);

    vector<long> ret(n);
    for (int i = 0; i < n; i++)
        ret[i] = long(exp(x[i]) + 0.5);

    int i0 = rand_int(0, n);

    for (int i1 = 0; i1 < n; i1++) {
        int i = (i0 + i1) % n;

        long p = 1;
        for (int j = 0; j < n; j++)
            if (j != i)
                p *= ret[j];

        ret[i] = long(exp(target_log-log(p)) + 0.5);
        ret[i] = min(ret[i], bound/p);
        ret[i] = max(ret[i], 1L);
    }

    long p = 1;
    for (int j = 0; j < n; j++)
        p *= ret[j];

    assert(p > 0);
    assert(p <= bound);
    return ret;
}

// Called by cpu_alloc(..., randomize=true)
template<typename T>
static void randomize_array(T *buf, long nelts)
{
    if constexpr (std::is_same_v<T, float>) {
        for (long i = 0; i < nelts; i++)
            buf[i] = rand_uniform();
    }
    else if constexpr (std::is_integral_v<T>) {
        long nbytes = nelts * sizeof(T);
        uint *buf32 = (uint *) buf;

        while (nbytes >= 4) {
            *buf32++ = casm_rng();
            nbytes -= 4;
        }
        
        if (nbytes == 0)
            return;
        
        uint8_t *buf8 = (uint8_t *) buf32;
        uint x = casm_rng();

        while (nbytes > 0) {
            *buf8++ = (x >> (8*nbytes-8));
            nbytes--;
        }
    }
    else
        throw runtime_error("CasmBeamformer: unsupported dtype in randomize_array()");
}


// -------------------------------------------------------------------------------------------------
//
// Helpers for allocating memory, copying to/from GPU, and random number generation.


template<typename T>
static shared_ptr<T> cpu_alloc(long nelts, bool randomize=false)
{
    T *p = nullptr;
    CUDA_CALL(cudaHostAlloc((void **) &p, nelts * sizeof(T), 0));
    
    if (randomize)
        randomize_array(p, nelts);
    else
        memset(p, 0, nelts * sizeof(T));
    
    return shared_ptr<T> (p, cudaFreeHost);  // cudaFreeHost() will be called by shared_ptr destructor
}


template<typename T>
static shared_ptr<T> gpu_alloc(long nelts)
{
    T *p = nullptr;
    CUDA_CALL(cudaMalloc((void **) &p, nelts * sizeof(T)));
    CUDA_CALL(cudaMemset(p, 0, nelts * sizeof(T)));
    return shared_ptr<T> (p, cudaFree);   // cudaFree() will be called by shared_ptr destructor
}


template<typename T>
static shared_ptr<T> to_gpu(const shared_ptr<T> &src, long nelts)
{
    shared_ptr<T> dst = gpu_alloc<T> (nelts);
    CUDA_CALL(cudaMemcpy(dst.get(), src.get(), nelts * sizeof(T), cudaMemcpyHostToDevice));
    return dst;
}


template<typename T>
static shared_ptr<T> to_cpu(const shared_ptr<T> &src, long nelts)
{
    shared_ptr<T> dst = cpu_alloc<T> (nelts);
    CUDA_CALL(cudaMemcpy(dst.get(), src.get(), nelts * sizeof(T), cudaMemcpyDeviceToHost));
    return dst;
}


// -------------------------------------------------------------------------------------------------
//
// Memory layouts
//
// NOTE: we currently keep the beam locations in shared memory. In hindsight, maybe registers
// would have been better? (I don't think it's worth revisiting for CASM-256, but I'm leaving
// this note, in case the beamformer is ever extended to a larger array.)


// Global memory layout for 'gpu_persistent_data'.
//
// This is a ~200KB region of global GPU memory, which is initialized by the
// CasmBeamformer constructor, and passed to the beamformer on every kernel launch.
//
//   uint gridding[256];                  contains (43*ew+ns) for each dish
//   float ns_phases[32];                 contains cos(2pi * t / 128) for 0 <= t < 32
//   float per_frequency_data[F][3][32];  middle index is 0 <= ew_feed < 3, see (*) below
//   float beam_locs[B][2];               contains { feed_spacing_ns * sin(za_ns), sin(za_ew) } (**)
//
// (*) The length-32 "inner" index in 'per_frequency_data' is laid out as follows:
//
//   float ew_phases[12][2];              (ew_beam, cos/sin) for fixed 0 <= ew_feed < 3
//   float freq;                          array element 25
//
// (**) Note that beam locations are specified in **host-side** code as (sin(za_ns), sin(za_ew)),
//  where (za_ns, za_ew) are the zenith angles. However, in the GPU kernel, it's more convenient
//  to represent beam locations as (feed_spacing_ns * sin(za_ns), sin(za_ew)). The conversion is
//  done in the CasmBeamformer constructor.

struct gmem_layout
{
    // All quantities below are 32-bit offsets, not byte offsets.
    static constexpr int gridding_base = 0;
    static constexpr int ns_phases_base = 256;
    
    static __host__ __device__ constexpr int per_frequency_data_base(int f=0)  { return ns_phases_base + 32 + 96*f; }
    static __host__ __device__ constexpr int beam_locs_base(int F, int b=0)    { return ns_phases_base + 32 + 96*F + 2*b; }
    static __host__ __device__ constexpr int nelts(int F, int B)               { return ns_phases_base + 32 + 96*F + 2*B; }
};


// Shared memory layout.
// Note that max number of beams is 4672, which gives 99KB shared memory.
// Some of these arrays have non-contiguous strides, in order to avoid a bank conflict.
//
// The E[] array stores int4+4 voltages, which may be either gridded or ungridded.
// The I[] array stores a (24,128) regular grid of beamformed intensities, before interpolation.
// The G[] array stores the "partially beamformed" intermediate array (beamformed in NS direction but not EW).
//
// For reference, here is the output of CasmBeamformer::show_shared_memory_layout().
// Columns are ([32-bit offset], physical layout, logical layout).
//
// [0]      uint gridding[256];                                         // contains (43*ew+ns), derived from 'feed_indices'
// [256]    float ns_phases[32];                                        // contains cos(2pi * t / 128) for 0 <= t < 32
// [288]    float per_frequency_data[3][32];                            // outer index is 0 <= ew_feed < 3, see (*) below
// [384]    uint E[24][259];       union { E[24][256], E[24][6][43] }   // (j,dish) or (j,ew,ns), outer stride 259
// [6600]   float I[24][132];      float I[24][128];                    // (ew,ns), ew-stride 132
// [9768]   float G[8][772];       float G[2][2][2][6][128];            // (time,pol,reim,ew,ns), reim-stride 6*128 + 4 (=772)
// [15944]  float beam_locs[2][4672];                                   // contains { feed_spacing_ns * sin(za_ns), sin(za_ew) }

struct shmem_layout
{
    // All _*stride and _*base quantities are 32-bit offsets, not byte offsets.
    static constexpr int E_jstride = 259;
    static constexpr int I_ew_stride = 132;
    static constexpr int G_reim_stride = 6*128 + 4;
    
    static constexpr int gridding_base = 0;
    static constexpr int ns_phases_base = 256;
    static constexpr int per_frequency_data_base = ns_phases_base + 32;
    static constexpr int E_base = per_frequency_data_base + 96;
    static constexpr int I_base = E_base + 24 * E_jstride;
    static constexpr int G_base = I_base + 24 * I_ew_stride;
    static constexpr int beam_locs_base = G_base + 8 * G_reim_stride;

    static constexpr int max_beams = ((99*1024 - 4*beam_locs_base) / 8) & ~31;
    static constexpr int beam_stride = max_beams;
    static constexpr int nbytes = (beam_locs_base + 2*max_beams) * 4;
    
    // During initialization, the E[], I[], and G[] arrays aren't needed yet,
    // so we temporarily use their shared memory to store feed_weights:
    //
    // [384]    float ungridded_wts[4][259];   uwt[2][2][256]       // (reim, pol, dish)
    // [1420]   float gridded_wts[4][259];     gwt[2][2][6][43]     // (reim, pol, ew, ns)

    static constexpr int wt_pol_stride = 259;
    static constexpr int ungridded_wts_base = E_base;
    static constexpr int gridded_wts_base = E_base + 4 * wt_pol_stride;
};


// -------------------------------------------------------------------------------------------------
//
// casm_controller: this is one of four "microkernels" that get coalesced into the casm beamformer.
// It is responsible for one-time initializations:
//
//    - copy_global_to_shared_memory(): gridding, fft phases, feed weights, beam locations, etc.
//    - normalize_beam_locations(): (ns_feed_spacing * sin(za_ns), sin(za_ew)) -> (xns, xew)
//    - setup_feed_weights(): convert 'feed_weights' array from shape (256,) to shape (6,43).
//
// and the following steps in the main kernel loop:
// 
//    - load_ungridded_e(): called every 24 time samples, to start reading E-array into registers
//    - write_ungridded_e(): called every 24 time samples, to write E-array to shared memory
//    - grid_shared_e(): called every 48 time samples, to "grid" E-array in shared memory
//    - load_gridded_e(): called every 8 time samples, to read E-array from shared -> registers
//    - unpack_e(): called every 2 time samples, to convert int4+4 -> float32+32 and apply weights
//
// The idea is that the casm_controller handles global memory management, and management of
// the E-array in shared memory. It presents the E-array to "downstream" microkernels as a
// sequence of floats (with feed_weights applied), so that these microkernels don't need to
// worry about memory management details.
//
// **Note 1** the E-array shared memory layout is nontrivial to describe!
// It's convenient to reindex (time,pol) as (i,j) where
//
//   i1 i0 <-> t2 t1
//   j2* j1 j0 <-> t3* t0 pol
//
// The int4+4 E-array is "promoted" to uint32, using bits [i1,i0].
// Then the E-array can be viewed as follows:
//
//   uint E[J][256];     // if ungridded
//   uint E[J][6][43];   // if gridded
//
// where J=24 corresponds to 48 time samples. The actual shared memory layout is:
//
//   uint E[24][259];  union { E[24][256], E[24][6][43] }   // (j,dish) or (j,ew,ns), outer stride 259
//
// where we use j-stride 259 in order to avoid a shared memory bank conflict.
//
//
// **Note 2** all kernels use {32,24,1} threads/block, and {F,1,1} blocks.


__device__ inline void double_byte_perm(uint &x, uint &y, uint s1, uint s2)
{
    uint x0 = x;
    x = __byte_perm(x0, y, s1);
    y = __byte_perm(x0, y, s2);
}

__device__ inline float unpack_int4(uint x, uint s)
{
    x = ((x >> s) ^ 0x88888888) & 0xf;
    return float(x) - 8.0f;
}


template<bool Debug>
struct casm_controller
{
    // Managed by setup_e_pointer(), load_ungridded_e(), write_ungridded_e()
    const uint4 *ep4;  // global memory pointer
    uint4 e4;          // stores E-array, en route from global->shared

    // Managed by setup_feed_weights(), load_gridded_e(), unpack_e().
    float fw0_re, fw0_im, fw1_re, fw1_im;  // persistent registers to store feed weights
    uint e0, e1;       // stores E-array, en route from shared->unpacked
    
    // Constructor does one-time initializations:
    //
    //    - copy_global_to_shared_memory(): gridding, fft phases, feed weights, beam locations, etc.
    //    - normalize_beam_locations(): (ns_feed_spacing * sin(za_ns), sin(za_ew)) -> (xns, xew)
    //    - setup_feed_weights(): convert 'feed_weights' array from shape (256,) to shape (6,43).
    //
    // 'global_e' argument: points to (T,F,2,256)
    // 'feed_weights' argument: points to (F,2,256,2)
    // 'gpu_persistent_data' argument: see "global memory layout" earlier in source file.
    //
    // Warning: caller must call __syncthreads() after calling constructor, and before calling load_ungridded_e().
    
    __device__ inline casm_controller(const uint8_t *global_e, const float *feed_weights, const float *gpu_persistent_data, int nbeams)
    {
        if constexpr (Debug) {
            assert(blockDim.x == 32);
            assert(blockDim.y == 24);
            assert(blockDim.z == 1);
        }

        copy_global_to_shared_memory(gpu_persistent_data, feed_weights, nbeams);
        __syncthreads();

        normalize_beam_locations(nbeams);
        setup_e_pointer(global_e);

        // Calls __syncthreads() in the middle of gridding process.
        setup_feed_weights();
    }

    // copy_global_to_shared_memory(): called by constructor, performs the following copies
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

    __device__ inline void copy_global_to_shared_memory(const float *gpu_persistent_data, const float *feed_weights, int nbeams)
    {
        extern __shared__ float shmem_f[];

        uint w = threadIdx.y;          // warp id
        uint l = threadIdx.x;          // lane id
        uint f = blockIdx.x;           // frequency channel
        uint F = gridDim.x;            // number of frequency channels
        uint B32 = nbeams >> 5;

        if (w < 9) {
            // gridding + ns_phases (256+32 elts)
            uint s = 32*w + l;
            shmem_f[s] = gpu_persistent_data[s];
            w += 24;
        }

        if (w < 12) {
            // per_frequency_data (96 elts)
            uint s = 32*(w-9) + l;
            uint dst = shmem_layout::per_frequency_data_base + s;
            uint src = gmem_layout::per_frequency_data_base(f) + s;
            shmem_f[dst] = gpu_persistent_data[src];
            w += 24;
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
            shmem_f[dst] = fw.x;         // real part
            shmem_f[dst + 2*S] = fw.y;   // imag part
                
            w += 24;
        }

        while (w < B32+28) {
            // beam_locations (B*2 floats).
            // It's easiest to read these as float2.
            
            const float2 *bl2 = (const float2 *) (gpu_persistent_data + gmem_layout::beam_locs_base(F));
            constexpr uint S = shmem_layout::beam_stride;
                          
            uint b = 32*(w-28) + l;   // beam id
            uint dst = shmem_layout::beam_locs_base + b;

            float2 bl = bl2[b];
            shmem_f[dst] = bl.x;      // north-south beam location
            shmem_f[dst + S] = bl.y;  // east-west beam location
            w += 24;
        }
        
        // Important: zero 'gridded_wts' in shared memory, in order to capture the two
        // "missing" feeds. (It's easy to miss this, since it only matters if the number
        // of threadblocks is larger than the number of SMs.)

        constexpr int zstart = shmem_layout::gridded_wts_base;
        constexpr int zsize = 4 * shmem_layout::wt_pol_stride;
        
        for (uint i = 32*threadIdx.y + threadIdx.x; i < zsize; i += 24*32)
            shmem_f[zstart + i] = 0.0f;

        // Note: no __syncthreads() here, caller is responsible for calling __syncthreads().
    }

    // normalize_beam_locations(): called by constructor
    //
    // Input: (ns_feed_spacing * sin(za_ns), sin(za_ew)) in shared memory
    //
    // Output: (xns, xew) in shared memory, where 0 <= xns <= 128 is a periodic
    // grid coordinate, and 1 <= xew <= 22 is a non-periodic grid coordinate.

    __device__ inline void normalize_beam_locations(int nbeams)
    {
        constexpr int F0 = shmem_layout::per_frequency_data_base + 25;
        constexpr int NSB = shmem_layout::beam_locs_base;
        constexpr int EWB = NSB + shmem_layout::max_beams;

        extern __shared__ float shmem_f[];      
        float freq = shmem_f[F0];
        
        int i = (32 * threadIdx.y) + threadIdx.x;

        while (i < nbeams) {
            // xns = 128 * (freq/c) * ns_feed_spacing * sin(za_ns)
            constexpr float a = 1.0f / CasmBeamformer::speed_of_light;
            float t = a * freq * shmem_f[NSB + i];   // shmem_f[...] = ns_feed_spacing * sin(za_ns)
            t -= int(t);
            t = (t >= 0) ? t : (t+1.0f);
            
            float xns = 128.0f * t;
            shmem_f[NSB + i] = xns;
            i += 24*32;
        }

        i -= nbeams;

        while (i < nbeams) {
            // xew = 10.5*sin(za_ew) + 11.5
            float sza = shmem_f[EWB + i];
            float xew = (10.5f * sza) + 11.5f;
            shmem_f[EWB + i] = xew;
            i += 24*32;
        }
    }
    
    // These member functions manage copying the ungridded E-array from global
    // memory to shared memory, with Delta(t)=24.
    //
    //   setup_e_pointer(): called by constructor
    //   load_ungridded_e(): called in kernel loop, every 24 time samples
    //   write_ungridded_e(): called in kernel loop, every 24 time samples
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
    
    __device__ inline void setup_e_pointer(const uint8_t *global_e)
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

    // load_ungridded_e(): called every 24 time samples, to start reading
    // E-array into registers.
    //
    // The value of 't0' is the same on all threads, and advances by 24
    // after each call to load_ungridded_e(). A thread-dependent offset
    // 'dt' will be added, and load_ungridded_e() no-ops if (t0+dt) >= T,
    // where T is the total number of time samples.
    
    __device__ inline void load_ungridded_e(int t0, int T)
    {
        uint i = (threadIdx.x >> 3);
        uint j = (threadIdx.y >> 1);
        uint dt = (i << 1) | ((j & 2) >> 1) | ((j >> 2) << 3);
        
        if ((t0 + dt) < T)
            e4 = *ep4;
    }

    // write_ungridded_e(): called every 24 time samples, to write E-array
    // to shared memory. The "phase" argument is either 0 or 1, and indicates
    // the phase within the 48-sample shared memory window.

    __device__ inline void write_ungridded_e(int phase)
    {
        extern __shared__ uint shmem_u[];

        if constexpr (Debug)
            assert((phase >= 0) && (phase <= 1));
        
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

    // grid_shared_e(): called every 48 time samples, to "grid" E-array in shared memory.
    //
    // A little awkward, since we want to loop over 256 dishes with 24 warps.
    // Note that 256 = 10*240 + 16.
    //
    // Note that the two "missing" grid locations will end up with uninitialized values
    // in shared memory! This sounds like a bug, but is actually okay since unpack_e()
    // multiplies by the gridded feed_weights, which are always zero at missing grid
    // locations.
    
    __device__ inline void grid_shared_e()
    {
        extern __shared__ uint shmem_u[];
        
        uint e[11];
        uint j = threadIdx.x;  // lane
        uint w = threadIdx.y;  // warp
        uint d0 = (w < 16) ? (11*w) : (10*w+16);
        uint s = shmem_layout::E_base + (j * shmem_layout::E_jstride);

        #pragma unroll
        for (int i = 0; i < 10; i++)
            e[i] = (j < 24) ? shmem_u[s+d0+i] : 0;
        
        if (w < 16)
            e[10] = (j < 24) ? shmem_u[s+d0+10] : 0;
        
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

    // The member functions
    //
    //   setup_feed_weights(): called by constructor
    //   load_gridded_e(): called every 8 time samples
    //   unpack_e(): called every 2 time samples
    //
    // manage reading the int4+4 gridded E-array from shared memory, "unpacking"
    // to float32+32, and applying feed weights.
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


    // Called by constructor, converts 'feed_weights' array from shape (256,) to shape (6,43).
    __device__ inline void setup_feed_weights()
    {
        extern __shared__ float shmem_f[];
        extern __shared__ uint shmem_u[];
        
        constexpr int WS = shmem_layout::wt_pol_stride;
        
        // Recall the shared memory layout:
        //
        // [384]    float ungridded_wts[4][259];   uwt[2][2][256]    // (reim, pol, dish)
        // [1420]   float gridded_wts[4][259];     gwt[2][2][6][43]  // (reim, pol, ew, ns)
        //
        // When this function is called, the weights are ungridded.
        // The first step is gridding the weights.

        int w = threadIdx.y;  // warp id
        int l = threadIdx.x;  // lane id
        int d = 32*w + l;

        if (d < 256) {
            // Only 8 active warps! But that should enough, since shared memory IO is low latency.
            uint g = shmem_u[d];  // destination (gridded) address.
            uint src = shmem_layout::ungridded_wts_base + d;
            uint dst = shmem_layout::gridded_wts_base + g;

            // These shared memory loads are bank conflict free.
            float fw0 = shmem_f[src];
            float fw1 = shmem_f[src + WS];
            float fw2 = shmem_f[src + 2*WS];
            float fw3 = shmem_f[src + 3*WS];

            // The following 4 lines of code would be logically correct, but can produce arbitrarily
            // bad bank conflicts, so we shuffle things around first.
            //
            //   shmem_f[dst] = fw0;
            //   shmem_f[dst + WS] = fw1;
            //   shmem_f[dst + 2*WS] = fw2;
            //   shmem_f[dst + 3*WS] = fw3;

            // If 'flag' is set, swap (fw0 <-> fw1) and (fw2 <-> fw3).
            bool flag = ((dst ^ threadIdx.x) & 1);
            float x0 = flag ? fw1 : fw0;
            float x1 = flag ? fw0 : fw1;
            float x2 = flag ? fw3 : fw2;
            float x3 = flag ? fw2 : fw3;
            int dd0 = flag ? (-WS) : WS;
            dst = flag ? (dst+WS) : dst;

            // If 'flag' is set, swap (fw0 <-> fw2) and (fw1 <-> fw3).
            flag = ((dst ^ threadIdx.x) & 2);
            float y0 = flag ? x2 : x0;
            float y1 = flag ? x3 : x1;
            float y2 = flag ? x0 : x2;
            float y3 = flag ? x1 : x3;
            int dd1 = flag ? (-2*WS) : (2*WS);
            dst = flag ? (dst+2*WS) : dst;

            // This assert implies that bank conflict is at most 8-to-1 (ugh).
            // Unfortunately, it's hard to do better! This bank conflict only happens
            // during initialization, so it shouldn't be a serious issue.
            
            if constexpr (Debug)
                assert(((dst ^ threadIdx.x) & 3) == 0);

            shmem_f[dst] = y0;
            shmem_f[dst + dd0] = y1;
            shmem_f[dst + dd1] = y2;
            shmem_f[dst + dd0 + dd1] = y3;
        }

        __syncthreads();
            
        // Second step is reading gridded weights into registers.
        
        // l4 l3 l2 l1 l0 <-> ns4 ns3 ns2 ns1 ns0
        // w2* w1 w0 <-> ew t0 pol  (= j1 j0)
        int pol = w & 1;
        int ew = w >> 2;
        int ns = l;

        // float gridded_wts[4][259];  gwt[2][2][6][43]  (reim, pol, ew, ns)
        int s = shmem_layout::gridded_wts_base + (pol*WS) + 43*ew + ns;

        // r1 r0 <-> ns5 ReIm
        fw0_re = shmem_f[s];
        fw0_im = shmem_f[s + 2*WS];
        fw1_re = (l < 11) ? shmem_f[s + 32] : 0.0f;          // If (ns > 43), assign zero weight
        fw1_im = (l < 11) ? shmem_f[s + 2*WS + 32] : 0.0f;   // If (ns > 43), assign zero weight
    }

    // load_gridded_e(): called every 8 time samples, to read E-array from shared -> registers.
    // The "phase" argument satisfies 0 <= phase < 6, and indicates the phase within the 48-sample
    // shared memory window.
    
    __device__ inline void load_gridded_e(int phase)
    {
        extern __shared__ uint shmem_u[];

        if constexpr (Debug)
            assert((phase >= 0) && (phase < 6));
        
        uint j = (threadIdx.y & 3);
        uint ew = (threadIdx.y >> 2);
        uint ns = threadIdx.x;
        uint s = shmem_layout::E_base + (4*phase+j)*shmem_layout::E_jstride + 43*ew + ns;
        
        e0 = shmem_u[s];
        e1 = (threadIdx.x < 11) ? shmem_u[s+32] : 0;
    }

    // unpack_e(): called every 2 time samples, to convert int4+4 -> float32+32 and apply weights.
    
    __device__ inline void unpack_e(int i, float &e0_re, float &e0_im, float &e1_re, float &e1_im)
    {
        // i is the length-4 index (t2 t1).
        
        if constexpr (Debug)
            assert((i >= 0) && (i < 4));

        // Unweighted E-array values, before multiplying by feed weights.
        float u0_re = unpack_int4(e0, 8*i);
        float u0_im = unpack_int4(e0, 8*i+4);
        float u1_re = unpack_int4(e1, 8*i);
        float u1_im = unpack_int4(e1, 8*i+4);

        // Multiply by feed wweights.
        e0_re = (fw0_re * u0_re) - (fw0_im * u0_im);
        e0_im = (fw0_re * u0_im) + (fw0_im * u0_re);
        e1_re = (fw1_re * u1_re) - (fw1_im * u1_im);
        e1_im = (fw1_re * u1_im) + (fw1_im * u1_re);
    }
};


// Unit test for 'struct casm_controller'.
__global__ void casm_controller_test_kernel(
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

    // Debug=true, nbeams=0
    casm_controller<true> controller(e_in, feed_weights, gpu_persistent_data, 0);
    __syncthreads();  // must call __syncthreads() after calling constructor, and before calling load_ungridded_e()

    // Set up writing to the 'out' array, with Delta(t)=2.
    // Output from casm_controller::unpack_e() has register assignment:
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
            controller.load_ungridded_e(touter + 24*s, T);
            controller.write_ungridded_e(s);
        }

        __syncthreads();
        
        controller.grid_shared_e();
        
        __syncthreads();

        for (int s = 0; s < 6; s++) {
            // Delta(t)=8, Delta(j)=4
            controller.load_gridded_e(s);
            
            for (int i = 0; i < 4; i++) {
                // Delta(t)=2
                float e0_re, e0_im, e1_re, e1_im;
                controller.unpack_e(i, e0_re, e0_im, e1_re, e1_im);
                
                // r1 r0 <-> ns5 ReIm
                // l4 l3 l2 l1 l0 <-> ns4 ns3 ns2 ns1 ns0
                // w2* w1 w0 <-> ew t0 pol

                int t0 = (threadIdx.y & 2) >> 1;
                int tinner = touter + 8*s + 2*i + t0;

                if (tinner < T) {
                    // float out[T][F][2][6][64][2]
                    out[0] = e0_re;
                    out[1] = e0_im;
                    out[64] = e1_re;
                    out[65] = e1_im;
                }

                // Delta(t)=2
                out += (2*F*2*6*64*2);
            }
        }

        __syncthreads();
    }
}


// Unit test for 'struct casm_controller'.
static void casm_controller_reference_kernel(
    const CasmBeamformer &bf,
    const uint8_t *e_in,                // (T,F,2,D) = (time,freq,pol,dish)
    const float *feed_weights,          // (F,2,256,2) = (freq,pol,dish,reim)
    float *out,                         // (T,F,2,6,64,2) = (time,freq,pol,ew,ns,reim)
    int T)
{
    int FP = 2 * bf.F;
    memset(out, 0, T * FP * 6*64*2 * sizeof(float));

    for (int t = 0; t < T; t++) {
        for (int fp = 0; fp < FP; fp++) {
            int tfp = t*FP + fp;
            
            const uint8_t *e2 = e_in + 256*tfp;        // points to shape (256,)
            const float *fw2 = feed_weights + 512*fp;  // points to shape (256,2)
            float *out2 = out + 6*64*2*tfp;            // points to shape (6,64,2)

            for (int d = 0; d < 256; d++) {
                uint8_t e = e2[d] ^ 0x88888888;
                float e_re = float(e & 0xf) - 8.0f;
                float e_im = float(e >> 4) - 8.0f;
                float fw_re = fw2[2*d];
                float fw_im = fw2[2*d+1];
                
                int ns = bf.feed_indices[2*d];
                int ew = bf.feed_indices[2*d+1];
                int g = 64*ew + ns;
                
                out2[2*g] = (fw_re * e_re) - (fw_im * e_im);
                out2[2*g+1] = (fw_re * e_im) + (fw_im * e_re);
            }
        }
    }
}


// Unit test for 'struct casm_controller'.
static void test_casm_controller(const CasmBeamformer &bf)
{
    // Allow kernel to use 99KB shared memory.
    static shmem_99kb s(casm_controller_test_kernel);
    s.set();
    
    int F = bf.F;
    int T = bf.nominal_Tin_for_unit_tests;
    cout << "test_casm_controller(T=" << T << ", F=" << F << ", D=" << bf.downsampling_factor << ")" << endl;

    shared_ptr<uint8_t> e = cpu_alloc<uint8_t> (T*F*2*256, true);         // shape (T,F,2,256), randomize=true
    shared_ptr<float> feed_weights = cpu_alloc<float> (F*2*256*2, true);  // shape (F,2,256,2), randomize=true
    shared_ptr<float> out_cpu = cpu_alloc<float> (T*F*2*6*64*2);          // shape (T,F,2,6,64,2)
    shared_ptr<float> out_gpu = gpu_alloc<float> (T*F*2*6*64*2);          // shape (T,F,2,6,64,2)

    // Run CPU reference kernel.
    casm_controller_reference_kernel(bf, e.get(), feed_weights.get(), out_cpu.get(), T);

    // Run GPU kernel.
    e = to_gpu(e, T*F*2*256);
    feed_weights = to_gpu(feed_weights, F*2*256*2);
    casm_controller_test_kernel<<< F, {32,24,1}, 99*1024 >>> (e.get(), feed_weights.get(), bf.gpu_persistent_data.get(), out_gpu.get(), T);
    CUDA_PEEK("casm_controller_test_kernel");

    // Check that arrays are equal.
    // out_cpu.shape == out_gpu.shape == (T,F,2,6,64,2).
    
    out_gpu = to_cpu(out_gpu, T*F*2*6*64*2);
    float *cp = out_cpu.get();
    float *gp = out_gpu.get();
    int pos = 0;
    
    for (int t = 0; t < T; t++) {
        for (int f = 0; f < F; f++) {
            for (int pol = 0; pol < 2; pol++) {
                for (int ew = 0; ew < 6; ew++) {
                    for (int ns = 0; ns < 64; ns++) {
                        for (int reim = 0; reim < 2; reim++) {
                            if (fabsf(cp[pos] - gp[pos]) > 1.0e-5) {
                                stringstream ss;
                                ss << "failed: t=" << t << ", f=" << f << ", pol=" << pol
                                   << ", ew=" << ew << ", ns=" << ns << ", reim=" << reim
                                   << ": cpu=" << cp[pos] << ", gpu=" << gp[pos];
                                throw runtime_error(ss.str());
                            }
                            pos++;
                        }
                    }
                }
            }
        }
    }
}


// -------------------------------------------------------------------------------------------------
//
// fft_c2c_microkernel: implements a c2c FFT with 2^R "active" elements and 2^{6-R} "spectators".
// (This is a helper for fft1_microkernel, see below.)


__device__ inline void fft0(float &xre, float &xim)
{
    float t = xre - xim;
    xre += xim;
    xim = t;
}


template<int R>
struct fft_c2c_microkernel
{
    // Implements a c2c FFT with 2^R "active" elements and 2^{6-R} "spectators".
    //
    // Input register assignment:
    //   r1 r0 <-> x_{r-1} ReIm
    //   l4 l3 l2 l1 l0 <-> s_{5-r} ... s0 x_{r-2} ... x_0
    //
    // Output register assignment:
    //   r1 r0 <-> y_{r-1} ReIm
    //   l4 l3 l2 l1 l0 <-> s_{5-r} ... s0 y_0 ... y_{r-2}

    fft_c2c_microkernel<R-1> next_fft;
    float cre, cim;

    // Caller passes phase128 = cos(2*pi*l/128), where 0 <= l < 32 is laneId.
    __device__ inline void init(float phase128)
    {
        // We want to compute the phase:
        //    exp(2*pi*i * t / 2^r)  where t = threadIdx.x mod 2^{r-1}
        //  = exp(2*pi*i * u / 128)  where u = t * (128/2^r)
        
        uint u = (threadIdx.x << (7-R)) & 63;
        cre = __shfl_sync(0xffffffff, phase128, (u < 32) ? u : (64-u));
        cim = __shfl_sync(0xffffffff, phase128, (u < 32) ? (32-u) : (u-32));
        cre = (u < 32) ? cre : (-cre);
        cre = (u != 32) ? cre : 0.0f;
        cim = (u != 0) ? cim : 0.0f;
        
        next_fft.init(phase128);
    }
    
    __device__ inline void apply(float  &x0_re, float &x0_im, float &x1_re, float &x1_im)
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


// Specializing fft_c2c_microkernel<R> for R=2 saves two persistent registers (per thread).
template<>
struct fft_c2c_microkernel<2>
{
    __device__ inline void init(float phase128) { }
    
    __device__ inline void apply(float  &x0_re, float &x0_im, float &x1_re, float &x1_im)
    {
        // (x0,x1) = (x0+x1,x0-x1)
        fft0(x0_re, x1_re);
        fft0(x0_im, x1_im);

        // x1 *= (either i or 1)
        uint t = threadIdx.x & 1;
        float yre = t ? (-x1_im) : x1_re;
        float yim = t ? (x1_re) : x1_im;
        x1_re = yre;
        x1_im = yim;

        // swap "01" register bit with thread bit 0
        warp_transpose(x0_re, x1_re, 1);
        warp_transpose(x0_im, x1_im, 1);

        // (x0,x1) = (x0+x1,x0-x1)
        fft0(x0_re, x1_re);
        fft0(x0_im, x1_im);
    }
};


// Unit test for 'struct fft_c2c_microkernel'.
// Input and output arrays have shape (2^(6-R), 2^R, 2)
// Call with 32 threads.

template<int R>
__global__ void fft_c2c_test_kernel(const float *in, float *out)
{
    assert(blockDim.x == 32);
    assert(blockDim.y == 1);
    assert(blockDim.z == 1);
    assert(gridDim.x == 1);
    assert(gridDim.y == 1);
    assert(gridDim.z == 1);

    constexpr float a = 6.283185307f / 128.0;   // 2*pi / 128
    float phase128 = cosf(a * threadIdx.x);
    
    // Input register assignment:
    //   r1 r0 <-> x_{r-1} ReIm
    //   l4 l3 l2 l1 l0 <-> s_{5-r} ... s0 x_{r-2} ... x_0
    
    int ss = threadIdx.x >> (R-1);            // spectator index
    int sx = threadIdx.x & ((1<<(R-1)) - 1);  // x-index
    int sin = (ss << (R+1)) | (sx << 1);
    
    float x0_re = in[sin];
    float x0_im = in[sin + 1];
    float x1_re = in[sin + (1<<R)];
    float x1_im = in[sin + (1<<R) + 1];
    
    fft_c2c_microkernel<R> fft;
    fft.init(phase128);
    fft.apply(x0_re, x0_im, x1_re, x1_im);

    // Output register assignment:
    //   r1 r0 <-> y_{r-1} ReIm
    //   l4 l3 l2 l1 l0 <-> s_{5-r} ... s0 y_0 ... y_{r-2}

    int sy = __brev(threadIdx.x << (33-R));
    int sout = (ss << (R+1)) | (sy << 1);

    out[sout] = x0_re;
    out[sout + 1] = x0_im;
    out[sout + (1<<R)] = x1_re;
    out[sout + (1<<R) + 1] = x1_im;
}


// Unit test for 'struct fft_c2c_microkernel'.
// Length-N FFT, with S spectator indices
// out.shape == in.shape == (S,N,2)
static void fft_c2c_reference_kernel(float *out, const float *in, int N, int S)
{
    memset(out, 0, S*N*2 * sizeof(float));
    
    for (int j = 0; j < N; j++) {
        for (int k = 0; k < N; k++) {
            float theta = (2*M_PI/N) * ((j*k) % N);
            float cth = cosf(theta);
            float sth = sinf(theta);
            
            for (int s = 0; s < S; s++) {
                float xre = in[s*2*N + 2*k];
                float xim = in[s*2*N + 2*k+1];

                out[s*2*N + 2*j] += (cth*xre - sth*xim);
                out[s*2*N + 2*j+1] += (sth*xre + cth*xim);
            }
        }
    }
}


// Unit test for 'struct fft_c2c_microkernel'.
static void test_casm_fft_c2c_microkernel()
{
    constexpr int R = 6;
    constexpr int N = (1 << R);
    constexpr int S = (1 << (6-R));

    cout << "test_casm_fft_c2c_microkernel()" << endl;
    shared_ptr<float> in = cpu_alloc<float> (S*N*2, true);   // shape (S,N,2), randomize=true
    shared_ptr<float> out_cpu = cpu_alloc<float> (S*N*2);    // shape (S,N,2)
    shared_ptr<float> out_gpu = gpu_alloc<float> (S*N*2);

    // Run CPU reference kernel.
    fft_c2c_reference_kernel(out_cpu.get(), in.get(), N, S);

    // Run GPU kernel.
    in = to_gpu(in, S*N*2);
    fft_c2c_test_kernel<R> <<<1,32>>> (in.get(), out_gpu.get());
    CUDA_PEEK("fft_c2c_test_kernel");

    // Check that arrays are equal
    // out_cpu.shape == out_gpu.shape == (S,N,2)

    out_gpu = to_cpu(out_gpu, S*N*2);
    float *gp = out_gpu.get();
    float *cp = out_cpu.get();
    int pos = 0;

    for (int s = 0; s < S; s++) {
        for (int i = 0; i < N; i++) {
            for (int reim = 0; reim < 2; reim++) {
                if (fabsf(gp[pos] - cp[pos]) > 1.0e-3) {
                    stringstream ss;
                    ss << "failed: s=" << s << ", i=" << i << ", reim=" << reim
                       << ": cpu=" << cp[pos] << ", gpu=" << gp[pos];
                    throw runtime_error(ss.str());
                }
                pos++;
            }
        }
    }
}


// -------------------------------------------------------------------------------------------------
//
// FFT1 microkernel: implements a zero-padded c2c FFT with 64 inputs and 128 outputs.
//
// This is called in the main kernel every 2 time samples, to do the 43->128 beamforming
// along the NS axis, and write the resulting "partially beamformed" (6,128) array to G[]
// shared memory, for 2 time samples and both polarizations.
//
// Input array should be in registers, with the same register assignment as
// casm_controller::load_gridded_e():
//
//   r1 r0 <-> ns5 ReIm
//   l4 l3 l2 l1 l0 <-> ns4 ns3 ns2 ns1 ns0
//   w2* w1 w0 <-> ew t0 pol
//
// Output array is written to shared memory:
// float G[8][772];  float G[2][2][2][6][128];  (time,pol,reim,ew,ns), reim-stride 6*128 + 4 (=772)


template<bool Debug>
struct fft1_microkernel
{
    fft_c2c_microkernel<6> next_fft;
    float cre, cim;   // "twiddle" factor exp(2*pi*i t / 128)
    int sbase;        // shared memory offset

    __device__ inline fft1_microkernel()
    {
        extern __shared__ float shmem_f[];
        
        if constexpr (Debug) {
            assert(blockDim.x == 32);
            assert(blockDim.y == 24);
            assert(blockDim.z == 1);
        }
        
        int w = threadIdx.y;  // warp id
        int l = threadIdx.x;  // lane id

        // phase cos(2*pi*l/128), where 0 <= l < 32 is laneId.
        float phase128 = shmem_f[shmem_layout::ns_phases_base + l];
        next_fft.init(phase128);
        
        cre = phase128;
        cim = __shfl_sync(0xffffffff, phase128, 32-l);
        cim = l ? cim : 0.0f;
        
        // Just before writing to shared memory (see below), the FFT-ed array will have
        // register assignment:
        //
        //   rxy r1 r0 <-> ns5 ns6 ReIm
        //   l4 l3 l2 l1 l0 <-> ns1 ns2 ns3 ns4 ns0
        //   w2* w1 w0 <-> ew t0 pol

        int ns = (l & 1) | (__brev(l >> 1) >> 27);
        int pol = w & 1;
        int t0 = (w >> 1) & 1;
        int ew = w >> 2;

        // float G[8][772];  float G[2][2][2][6][128];  (time,pol,reim,ew,ns), reim-stride 6*128 + 4 (=772)
        constexpr uint GB = shmem_layout::G_base;
        constexpr uint GS = shmem_layout::G_reim_stride;
        sbase = GB + (4*t0 + 2*pol) * GS + (128*ew + ns);

        // Check that G-array write is bank conflict free.
        check_bank_conflict_free<Debug> (sbase);
    }
    
    __device__ inline void apply(float x0_re, float x0_im, float x1_re, float x1_im)
    {
        extern __shared__ float shmem_f[];
            
        // y0 = x0 * exp(2*pi*i l / 128) = c*x0
        // y1 = x1 * exp(2*pi*i (l+32) / 128) = i*c*x0
        //   where l = 0, ..., 31
        
        float y0_re = cre*x0_re - cim*x0_im;
        float y0_im = cim*x0_re + cre*x0_im;

        float y1_re = -cim*x1_re - cre*x1_im;
        float y1_im = cre*x1_re - cim*x1_im;

        // xy r1 r0 <-> y0 x5 ReIm
        // l4 l3 l2 l1 l0 <-> x4 x3 x2 x1 x0

        next_fft.apply(x0_re, x0_im, x1_re, x1_im);
        next_fft.apply(y0_re, y0_im, y1_re, y1_im);
        
        // xy r1 r0 <-> y0 y6 ReIm
        // l4 l3 l2 l1 l0 <-> y1 y2 y3 y4 y5

        // Exchange "xy" and "thread 0" bits
        warp_transpose(x0_re, y0_re, 1);
        warp_transpose(x0_im, y0_im, 1);
        warp_transpose(x1_re, y1_re, 1);
        warp_transpose(x1_im, y1_im, 1);
        
        // xy r1 r0 <-> y5 y6 ReIm
        // l4 l3 l2 l1 l0 <-> y1 y2 y3 y4 y0

        // Strides: xy=32, 01=64, ReIm=GS
        constexpr uint GS = shmem_layout::G_reim_stride;

        shmem_f[sbase] = x0_re;
        shmem_f[sbase+32] = y0_re;
        shmem_f[sbase+64] = x1_re;
        shmem_f[sbase+96] = y1_re;
        shmem_f[sbase+GS] = x0_im;
        shmem_f[sbase+GS+32] = y0_im;
        shmem_f[sbase+GS+64] = x1_im;   
        shmem_f[sbase+GS+96] = y1_im;   
    }
};


// Unit test for 'struct fft1_microkernel'.
// Input array has shape (2,2,6,64,2) where axes are (time,pol,ew,ns,reim).
// Output array has shape (2,2,6,128,2) where axes have same meaning.
// Launch with {32,24,1} threads.

__global__ void fft1_test_kernel(const float *in, float *out)
{
    extern __shared__ float shmem_f[];
    
    assert(blockDim.x == 32);
    assert(blockDim.y == 24);
    assert(blockDim.z == 1);
    assert(gridDim.x == 1);
    assert(gridDim.y == 1);
    assert(gridDim.z == 1);
 
    int w = threadIdx.y;  // warp id
    int l = threadIdx.x;  // lane id

    // We initialize the 'ns_phases' part of shared memory (see "shared memory
    // layout" above). Note that there are other parts of this shared memory
    // layout that are not initialized, for example 'gridding', since they
    // aren't needed by the fft1_test_kernel.

    if (w == 0) {
        constexpr float a = 6.283185307f / 128.0;   // 2*pi / 128
        shmem_f[shmem_layout::ns_phases_base + l] = cosf(a*l);
    }

    __syncthreads();
    
    // Input register assignment for fft1_microkernel::apply() is:
    //
    //   r1 r0 <-> ns5 ReIm
    //   l4 l3 l2 l1 l0 <-> ns4 ns3 ns2 ns1 ns0
    //   w2* w1 w0 <-> ew t0 pol

    int pol = (w & 1);
    int t0 = (w >> 1) & 1;
    int ew = (w >> 2);
    int ns = l;

    // These "flattened" index combinations ire convenient in two places below.
    int tp = 2*t0 + pol;           // (time, pol)
    int tpe = 12*t0 + 6*pol + ew;  // (time, pol, ew)
    
    // Input array has shape (2,2,6,64,2) where axes are (time,pol,ew,ns,reim).
    float x0_re = in[128*tpe + 2*ns];
    float x0_im = in[128*tpe + 2*ns + 1];
    float x1_re = in[128*tpe + 2*(ns+32)];
    float x1_im = in[128*tpe + 2*(ns+32) + 1];

    fft1_microkernel<true> fft1;  // Debug=true
    fft1.apply(x0_re, x0_im, x1_re, x1_im);
    __syncthreads();

    // Shared memory output array has layout:
    //   float G[8][772];  float G[2][2][2][6][128];  (time,pol,reim,ew,ns), reim-stride 6*128 + 4 (=772)
    //
    // Output array has shape (2,2,6,128,2) where axes are (time,pol,ew,ns,reim)

    constexpr uint GB = shmem_layout::G_base;
    constexpr uint GS = shmem_layout::G_reim_stride;
    
    for (int reim = 0; reim < 2; reim++)
        for (int ns2 = ns; ns2 < 128; ns2 += 32)
            out[256*tpe + 2*ns2 + reim] = shmem_f[GB + (2*tp+reim)*GS + (128*ew+ns2)];
}


// Unit test for 'struct fft1_microkernel'.
// out.shape == (2,2,6,128,2), axes (time,pol,ew,ns,reim)
// in.shape == (2,2,6,64,2), axes (time,pol,ew,ns,reim)

static void fft1_reference_kernel(float *out, const float *in)
{
    memset(out, 0, 24*128*2 * sizeof(float));
    
    for (int j = 0; j < 128; j++) {
        for (int k = 0; k < 64; k++) {
            float theta = (2*M_PI/128) * ((j*k) % 128);
            float cth = cosf(theta);
            float sth = sinf(theta);

            // The (time,pol,ew) indices can be "flattened" into a single index 0 <= tpe < 24.
            for (int tpe = 0; tpe < 24; tpe++) {
                float xre = in[128*tpe + 2*k];
                float xim = in[128*tpe + 2*k+1];

                out[256*tpe + 2*j] += (cth * xre) - (sth * xim);
                out[256*tpe + 2*j+1] += (sth * xre) + (cth * xim);
            }
        }
    }
}


// Unit test for 'struct fft1_microkernel'.
static void test_casm_fft1_microkernel()
{
    // Allow kernel to use 99KB shared memory.
    static shmem_99kb s(fft1_test_kernel);
    s.set();
    
    // Axis ordering (time,pol,ew,ns,reim).
    cout << "test_casm_fft1_microkernel()" << endl;
    shared_ptr<float> in = cpu_alloc<float> (2*2*6*64*2, true);   // shape (2,2,6,64,2), randomize=true
    shared_ptr<float> out_cpu = cpu_alloc<float> (2*2*6*128*2);   // shape (2,2,6,128,2)
    shared_ptr<float> out_gpu = gpu_alloc<float> (2*2*6*128*2);   // shape (2,2,6,128,2)

    // Run CPU reference kernel.
    fft1_reference_kernel(out_cpu.get(), in.get());

    in = to_gpu(in, 2*2*6*64*2);
    fft1_test_kernel <<< 1, {32,24,1}, 99*1024 >>> (in.get(), out_gpu.get());
    CUDA_PEEK("fft1_test_kernel");
    
    // Check that arrays are equal
    // out_cpu.shape == out_gpu.shape == (2,2,6,128,2), axes (time,pol,ew,ns,reim)

    out_gpu = to_cpu(out_gpu, 2*2*6*128*2);
    float *gp = out_gpu.get();
    float *cp = out_cpu.get();
    int pos = 0;

    for (int t = 0; t < 2; t++) {
        for (int pol = 0; pol < 2; pol++) {
            for (int ew = 0; ew < 6; ew++) {
                for (int ns = 0; ns < 128; ns++) {
                    for (int reim = 0; reim < 2; reim++) {
                        if (fabsf(gp[pos] - cp[pos]) > 1.0e-3) {
                            stringstream ss;
                            ss << "failed: t=" << t << ", pol=" << pol << ", ew=" << ew << ", ns=" << ns
                               << ", reim=" << reim << ": cpu=" << cp[pos] << ", gpu=" << gp[pos];
                            throw runtime_error(ss.str());
                        }
                        pos++;
                    }
                }
            }
        }
    }
}


// -------------------------------------------------------------------------------------------------
//
// FFT2 microkernel.
//
// This is the 6->24 beamforming FFT along the EW axis. We read the input from G[] shared memory,
// but the output isn't written anywhere. Instead, we square and accumulate to persistent registers
// which store beamformed intensities on the (24,128) grid. The FFT2 microkernel is called once per
// (time,polarization) in the main kernel.
//
// When the number of accumulated time samples is a multiple of 'downsampling_factor', we call
// fft2_microkernel::write_and_reset(), which writes the (24,128) beamformed intensity grid to I[]
// shared memory, in preparation for interpolation.
//
// The indexing in this kernel is complicated  -- here is a high-level summary!
//
// There are 24 east-west beams 0 <= b < 24 and 6 east-west feeds 0 <= f < 6.
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
// The persistent I-arary is distributed in registers as follows:
//
//   r1 r0 <-> (bouter) (b1)
//   l4 l3 l2 l1 l0 <-> (ns4) (ns3) (ns1) (ns0) (b0)
//   24 warps <-> (b2) (ns6) (ns5) (ns2)
//
// The G-array shared memory layout is (where "f" is an EW feed):
//
//   float G[8][772];  float G[2][2][2][6][128];  (time,pol,reim,f,ns), reim-stride 6*128 + 4 (=772)
//
// We read from the G-array in register assignment (bank conflict free,
// since ReIm stride = 4 mod 8):
//
//   l4 l3 l2 l1 l0 <-> (ns4) (ns3) (ns1) (ns0) (ReIm)
//
// The I-array shared memory layout is (where "b" indexes an EW beam):
//
//   float I[24][128];   // (b,ns), strides (132,1)
//
// We write to the I-array in the bank conflict free register assignment:
//
//   l4 l3 l2 l1 l0 <-> (ns4) (ns3) (ns1) (ns0) (b0)


template<bool Debug>
struct fft2_microkernel
{
    // We store EW beamforming phases for fouter=bouter=0 only, since
    // flipping 'bouter' or 'fouter' sends the phase to its complex
    // conjugate. This argument depends on the feed locations having
    // the EW "flip symmetry"!
    //
    // Note: we use 12 persistent registers/thread to store beamforming
    // phases, but the number of distinct phases is 24/warp or 72/block.
    // Instead, one could consider distributing phases as needed with
    // __shfl_sync(). This would cost flops but save registers.
     
    float I[2][2];      // beams are indexed by (bouter, b1)
    float pcos[2][3];   // beamforming phases are indexed by (b1, finner)
    float psin[2][3];   // beamforming phases are indexed by (b1, finner)
    
    int soff_g;     // base shared memory offset in G-array.
    int soff_i0;    // base shared memory offset in I-array, bouter=0
    int soff_i1;    // base shared memory offset in I-array, bouter=1


    __device__ inline fft2_microkernel()
    {
        extern __shared__ float shmem_f[];
        
        if constexpr (Debug) {
            assert(blockDim.x == 32);
            assert(blockDim.y == 24);
            assert(blockDim.z == 1);
        }
        
        I[0][0] = I[0][1] = I[1][0] = I[1][1] = 0.0f;

        // Unpack warp and lane indices:
        //
        //   l4 l3 l2 l1 l0 <-> (ns4) (ns3) (ns1) (ns0) (b0)
        //   w3* w2 w1 w0 <-> (b2) (ns6) (ns5) (ns2)
        //
        // into 7-bit (ns), and partial binner (b02) = (b2) (0) (b0)

        uint w = threadIdx.y;  // warp id
        uint l = threadIdx.x;  // lane id
        uint ns = ((w & 0x6) << 4) | (l & 0x18) | ((w & 0x1) << 2) | ((l & 0x6) >> 1);
        uint b02 = ((w & 0x18) >> 1) | (l & 0x1);

        // Load beamforming phases from shared memory.
        // (These were precomputed in the host-side CasmBeamformer constructor,
        // copied to GPU global memory, and then copied from global->shared in the
        // casm_controller constructor.)

        #pragma unroll
        for (uint finner = 0; finner < 3; finner++) {
            // ew_phases[12][2] indexed by (binner, cos/sin)
            float ew_phases = shmem_f[shmem_layout::per_frequency_data_base + 32*finner + l];
            pcos[0][finner] = __shfl_sync(0xffffffff, ew_phases, 2*b02);    // b1=0, cos
            psin[0][finner] = __shfl_sync(0xffffffff, ew_phases, 2*b02+1);  // b1=0, sin
            pcos[1][finner] = __shfl_sync(0xffffffff, ew_phases, 2*b02+4);  // b1=1, cos
            psin[1][finner] = __shfl_sync(0xffffffff, ew_phases, 2*b02+5);  // b1=1, sin
        }

        // Shared memory offset for reading G-array:
        //   float G[8][772];  float G[2][2][2][6][128];  (time,pol,reim,f,ns), reim-stride 6*128 + 4 (=772)
        //
        // When we read from the G-array, we read it as:
        //   l4 l3 l2 l1 l0 <-> (ns4) (ns3) (ns1) (ns0) (ReIm)
        //
        // 'soff_g' is the offset assuming time = pol = f = 0.

        uint reim = threadIdx.x & 1;
        soff_g = shmem_layout::G_base + (reim * shmem_layout::G_reim_stride) + ns;

        // Shared memory offset for writing I-array:
        //   float I[24][132];  float I[24][128];   (ew,ns), ew-stride 132
        //
        // When we write to the I-array, we write as
        //   l4 l3 l2 l1 l0 <-> (ns4) (ns3) (ns1) (ns0) (b0)
        //
        // 'soff_i{0,1}' is the offset with bouter={0,1} and b1=0.

        soff_i0 = shmem_layout::I_base + (12+b02) * shmem_layout::I_ew_stride + ns;  // offset for bouter=0
        soff_i1 = shmem_layout::I_base + (11-b02) * shmem_layout::I_ew_stride + ns;  // offset for bouter=1

        check_bank_conflict_free<Debug> (soff_g);
        check_bank_conflict_free<Debug> (soff_i0);
        check_bank_conflict_free<Debug> (soff_i1);
    }


    // apply(): called once per (time,polarization) in the main kernel.
    //
    // Reads the partially beamformed array from G[] shared memory (hence no 'float'
    // function args), does the 6->24 beamforming FFT, squares, and accumulates the
    // result to persistent registers which store fully beamformed intensities on the
    // (24,128) grid.
    //
    // The 'tpol' arg satisfies 0 <= tpol < 4, and specifies the "phase" within the G[]
    // shared memory window (two time samples and both polarizations).

    __device__ inline void apply(uint tpol)
    {
        extern __shared__ float shmem_f[];
        
        if constexpr (Debug)
            assert(tpol < 4);  // note tpol is unsigned
        
        // We accumulate the beamformed electric field (denoted F), for two choices of 'b1'
        // and two choices of 'bouter'. Each F-element gets G-contributions from three choices
        // of 'finner' and two choices of 'fouter'.
        //
        // For fixed (b1,finner), denote the F-array by (F0,F1), the G-array by (G0,G1),
        // and the beamforming phase by z. The "inner" update step is:
        //
        //   F0 += (z G0) + (z^* G1)
        //   F1 += (z^* G0) + (z G1)
        //
        // This could be done more efficiently by changing basis to F_{pm} = (F0 pm F1) / 2:
        //
        //   (F_+) += Re(z) * (G_+)      where G_+ = (G0+G1)
        //   (F_-) += i Im(z) * (G_-)    where G_- = (G0-G1)
   
        float Fp_re[2];  // length-2 axis is b1
        float Fp_im[2];  // length-2 axis is b1
        float Fm_re[2];  // length-2 axis is b1
        float Fm_im[2];  // length-2 axis is b1
        Fp_re[0] = Fp_im[0] = Fm_re[0] = Fm_im[0] = 0.0f;
        Fp_re[1] = Fp_im[1] = Fm_re[1] = Fm_im[1] = 0.0f;

        constexpr int GS = 2 * shmem_layout::G_reim_stride;
        int s = soff_g + (tpol * GS);   // shared memory offset assuming f=0
        
        #pragma unroll
        for (int finner = 0; finner < 3; finner++) {
            
            // Read G-array elements from shared memory
            //   r <-> fouter
            //   l4 l3 l2 l1 l0 <-> (ns4) (ns3) (ns1) (ns0) (ReIm)
            //   float G[8][772];  float G[2][2][2][6][128];  (time,pol,reim,f,ns), reim-stride 6*128 + 4 (=772)
            
            float t0 = shmem_f[s + ((3+finner) * 128)];  // fouter=0
            float t1 = shmem_f[s + ((2-finner) * 128)];  // fouter=1

            // Change basis (G0,G1) -> G_{pm} = (G0 \pm G1).
            float tp = (t0 + t1);
            float tm = (t0 - t1);
            float up = __shfl_sync(0xffffffff, tp, threadIdx.x ^ 1);
            float um = __shfl_sync(0xffffffff, tm, threadIdx.x ^ 1);
            
            bool flag = (threadIdx.x & 1);
            float Gp_re = flag ? up : tp;
            float Gp_im = flag ? tp : up;
            float Gm_re = flag ? um : tm;
            float Gm_im = flag ? tm : um;

            #pragma unroll
            for (int b1 = 0; b1 < 2; b1++) {
                float zre = pcos[b1][finner];
                float zim = psin[b1][finner];

                // (F_+) += Re(z) * (G_+)      where G_+ = (G0+G1)
                // (F_-) += i Im(z) * (G_-)    where G_- = (G0-G1)

                Fp_re[b1] += (zre * Gp_re);
                Fp_im[b1] += (zre * Gp_im);
                Fm_re[b1] -= (zim * Gm_im);
                Fm_im[b1] += (zim * Gm_re);
            }
        }

        #pragma unroll
        for (int b1 = 0; b1 < 2; b1++) {
            // Change basis from F_{pm} = (F0 \pm F1)/2 to {F0,F1}.
            float F0_re = Fp_re[b1] + Fm_re[b1];
            float F0_im = Fp_im[b1] + Fm_im[b1];
            float F1_re = Fp_re[b1] - Fm_re[b1];
            float F1_im = Fp_im[b1] - Fm_im[b1];

            // Accumulate I[2][2], where axes are (bouter,b1).
            I[0][b1] += (F0_re * F0_re) + (F0_im * F0_im);
            I[1][b1] += (F1_re * F1_re) + (F1_im * F1_im);
        }
    }


    // write_and_reset(): called in the main kernel when the number of accumulated
    // time samples is a multiple of 'downsampling_factor'.
    //
    // Writes the (24,128) beamformed intensity grid from persistent registers to I[]
    // shared memory (in preparation for interpolation), and zeroes the registers (in
    // preparation for the next call to fft2_microkernel.apply()).

    __device__ inline void write_and_reset()
    {
        extern __shared__ float shmem_f[];

        // Beam-stride in shared memory I[] array.
        //   float I[24][132];  float I[24][128];   (b,ns), b-stride 132
        constexpr int IS = shmem_layout::I_ew_stride;
        
        // Beams are indexed by (bouter, b1).   
        shmem_f[soff_i0] = I[0][0];
        shmem_f[soff_i1] = I[1][0];
        
        shmem_f[soff_i0 + 2*IS] = I[0][1];
        shmem_f[soff_i1 - 2*IS] = I[1][1];

        I[0][0] = I[0][1] = I[1][0] = I[1][1] = 0.0f;
    }
};


// Unit testing 'struct fft2_microkernel'.
// Launch with {32,24,1} threads and {F,1,1} blocks.
__global__ void fft2_test_kernel(
    const float *g_in,                  // (TP,F,6,128,2) = (tpol,freq,ewfeed,ns,reim)
    const float *feed_weights,          // (F,2,256,2), not used
    const float *gpu_persistent_data,   // see "global memory layout" earlier in source file
    float *i_out,                       // (F,24,128) = (freq,ewbeam,ns)
    int TP)
{
    extern __shared__ float shmem_f[];

    assert(blockDim.x == 32);
    assert(blockDim.y == 24);
    assert(blockDim.z == 1);
        
    casm_controller<true> controller(nullptr, feed_weights, gpu_persistent_data, 0);
    __syncthreads(); // I dont think I actually need this
    
    fft2_microkernel<true> fft2;  // Debug=true

    // Divide (F,6,128) array between threads.
    int f = blockIdx.x;
    int F = gridDim.x;
    int gns = ((threadIdx.y & 3) << 5) + threadIdx.x;
    int gew = (threadIdx.y >> 2);

    // G-array source global memory pointer. For each tpol, each thread copies one complex32+32.
    float2 *gp2 = (float2 *)(g_in) + f*6*128 + 128*gew + gns;

    // G-array destination shared memory index (offset 0 <= tp_sh < 4 remains to be applied)
    //   float G[8][772];  float G[2][2][2][6][128];  (time,pol,reim,ew,ns), reim-stride 6*128 + 4 (=772)
    constexpr int GB = shmem_layout::G_base;
    constexpr int GS = shmem_layout::G_reim_stride;
    uint gsh = GB + 128*gew + gns;

    for (int tp_glo = 0; tp_glo < TP; tp_glo++) {
        int tp_sh = tp_glo & 3;
        
        // Copy G[6][128][2] from (global) -> (shared).
        float2 g = *gp2;
        shmem_f[gsh + (2*tp_sh)*GS] = g.x;    // real part
        shmem_f[gsh + (2*tp_sh+1)*GS] = g.y;  // imag part
        gp2 += F*6*128;

        __syncthreads();

        fft2.apply(tp_sh);

        __syncthreads();
    }

    fft2.write_and_reset();
    __syncthreads();
        
    // Set up I-array copy (shared) -> (global)
    constexpr int IB = shmem_layout::I_base;
    constexpr int IS = shmem_layout::I_ew_stride;
    int goff = 24*128*f + 128*threadIdx.y + threadIdx.x;  // array offset in ip[] global array
    int soff = IB + IS*threadIdx.y + threadIdx.x;         // array offset in shmem_i[] shared array
    
    for (int j = 0; j < 4; j++)
        i_out[goff + 32*j] = shmem_f[soff + 32*j];
}


// Unit testing 'struct fft2_microkernel'.
// G.shape = {TP,F,6,128,2}
// I.shape = {F,24,128}

static void fft2_reference_kernel(const CasmBeamformer &bf, float *I, const float *G)
{
    int F = bf.F;
    int TP = 2 * bf.nominal_Tin_for_unit_tests;

    memset(I, 0, F*24*128 * sizeof(float));
    
    for (int ifreq = 0; ifreq < F; ifreq++) {
        for (int ew_beam = 0; ew_beam < 24; ew_beam++) {
            // Beamforming phase is exp(2*pi*i * freq * beam * feed / c)
            float bre[6];
            float bim[6];

            for (int ew_feed = 0; ew_feed < 6; ew_feed++) {
                float c = bf.speed_of_light;
                float freq = bf.frequencies[ifreq];
                float ew_bpos = bf.ew_beam_locations[ew_beam];
                float ew_fpos = bf.ew_feed_positions[ew_feed];
                float theta = 2*M_PI * freq * ew_bpos * ew_fpos / c;

                bre[ew_feed] = cosf(theta);
                bim[ew_feed] = sinf(theta);
            }
            
            for (int tpol = 0; tpol < TP; tpol++) {
                float *I2 = I + (ifreq*24 + ew_beam) * 128;         // shape (128,)
                const float *G2 = G + (tpol*F + ifreq) * 6*128*2;   // shape (6,128,2)

                for (int ns = 0; ns < 128; ns++) {
                    // Beamformed electric field.
                    float Fre = 0.0f;
                    float Fim = 0.0f;
                
                    for (int ew_feed = 0; ew_feed < 6; ew_feed++) {
                        float Bre = bre[ew_feed];
                        float Bim = bim[ew_feed];
                        float Gre = G2[256*ew_feed + 2*ns];
                        float Gim = G2[256*ew_feed + 2*ns + 1];
                        
                        Fre += (Bre*Gre) - (Bim*Gim);
                        Fim += (Bre*Gim) + (Bim*Gre);
                    }

                    I2[ns] += (Fre*Fre) + (Fim*Fim);
                }
            }
        }               
    }
}


// Unit testing 'struct fft2_microkernel'.
static void test_casm_fft2_microkernel(const CasmBeamformer &bf)
{
    // Allow kernel to use 99KB shared memory.
    static shmem_99kb s(fft2_test_kernel);
    s.set();

    int F = bf.F;
    int TP = 2 * bf.nominal_Tin_for_unit_tests;
    cout << "test_casm_fft2_microkernel(TP=" << TP << ", F=" << F << ")" << endl;

    // Note: feed_weights must be allocated, but are not actually used.
    shared_ptr<float> feed_weights = cpu_alloc<float> (F*2*256*2);  // shape (F,2,256,2)
    shared_ptr<float> g = cpu_alloc<float> (TP*F*6*128*2, true);    // shape (TP,F,6,128,2), randomize=true
    shared_ptr<float> i_cpu = cpu_alloc<float> (F*24*128);          // shape (F,24,128)
    shared_ptr<float> i_gpu = gpu_alloc<float> (F*24*128);          // shape (F,24,128)

    // Run CPU reference kernel.
    fft2_reference_kernel(bf, i_cpu.get(), g.get());

    // Run GPU kernel.
    g = to_gpu(g, TP*F*6*128*2);
    fft2_test_kernel<<< F, {32,24,1}, 99*1024 >>> (g.get(), feed_weights.get(), bf.gpu_persistent_data.get(), i_gpu.get(), TP);
    CUDA_PEEK("fft2_test_kernel");
    
    // Check that arrays are equal.
    // i_cpu.shape == i_gpu.shape == (F,24,128)

    i_gpu = to_cpu(i_gpu, F*24*128);
    float *gp = i_gpu.get();
    float *cp = i_cpu.get();
    int pos = 0;

    for (int f = 0; f < F; f++) {
        for (int ew = 0; ew < 24; ew++) {
            for (int ns = 0; ns < 128; ns++) {
                float x = cp[pos];
                float y = gp[pos];
                if (fabsf(x-y) > 1.0e-5 * (fabsf(x)+fabsf(y))) {
                    stringstream ss;
                    ss << "failed: f=" << f << ", ew=" << ew << ", ns=" << ns << ": cpu=" << x << ", gpu=" << y;
                    throw runtime_error(ss.str());
                }
                pos++;
            }
        }
    }
}


// -------------------------------------------------------------------------------------------------
//
// Interpolation microkernel.
//
// - Input: gridded I-values and beam locations from shared memory.
//
//     float I[24][132];  float I[24][128];  (ew,ns), ew-stride 132
//     float beam_locs[2][4672];             contains { feed_spacing_ns * sin(za_ns), sin(za_ew) }
//
// - Output: writes interpolated I-values to global memory, and advances output pointer.
// 
//     float I[Tout][F][B];
//
// NOTE: interpolation currently has a 2-to-1 shared memory bank conflict. In principle,
// this could be fixed by doing two interpolations in parallel (either two times, or two
// beams if there are enough beams), and doubling the size of I[] in shared memory. In
// practice, I didn't think this was worth implementing, since interpolation is fast
// compared to other parts of the beamformer.


template<bool Debug>
__host__ __device__ inline int split_integer_and_fractional(float &x, int imin, int imax)
{
    if constexpr (Debug) {
        constexpr float eps = 1.0e-5;
        assert(x >= imin - eps);
        assert(x <= imax + 1 + eps);
    }
               
    int ix = int(x);
    ix = (ix >= imin) ? ix : imin;
    ix = (ix <= imax) ? ix : imax;
    
    x = x - ix;  // update x -> (fractional part) in place
    return ix;   // return (integer part)
}


template<bool Debug>
struct interpolation_microkernel
{
    float *out;
    int nbeams;
    float normalization;
    
    __device__ inline interpolation_microkernel(float *out_, int nbeams_, float normalization_)
    {
        if constexpr (Debug) {
            assert(blockDim.x == 32);
            assert(blockDim.y == 24);
            assert(blockDim.z == 1);
        }
        
        int f = blockIdx.x;  // frequency channel
        out = out_ + f * nbeams_;
        nbeams = nbeams_;
        normalization = normalization_;
    }

    // Helper for apply().
    __device__ inline float compute_wk(int k, float x)
    {
        static constexpr float one_sixth = 1.0f / 6.0f;
        static constexpr float one_half = 1.0f / 2.0f;

        if constexpr (Debug)
            assert((k >= 0) && (k < 4));
        
        // Factor cubic interpolation weight as w_k = w0 * (x+a) * (x+b) * (x+c).
        // Could save a few clock cycles here by writing cryptic code, but I didn't
        // bother since interpolation is a small fraction of the running time.

        float w = ((k==0) || (k==3)) ? one_sixth : one_half;
        w = (k & 1) ? w : (-w);
        
        float a = (k > 0) ? 1.0f : 0.0f;
        float b = (k > 1) ? 0.0f : -1.0f;
        float c = (k > 2) ? -1.0f : -2.0f;
        return w * (x+a) * (x+b) * (x+c);
    }

    // apply(): called in the main kernel when the number of accumulated time samples is a
    // multiple of 'downsampling_factor', after the call to fft2_microkernel.write_and_reset().
    //
    // Interpolates the (24,128) beamformed intensity grid in I[] shared memory to the
    // specified beam locations, and writes the result to global memory.
    
    __device__ inline void apply()
    {
        extern __shared__ float shmem_f[];
        
        constexpr int IB = shmem_layout::I_base;
        constexpr int IS = shmem_layout::I_ew_stride;
        constexpr int NSB = shmem_layout::beam_locs_base;
        constexpr int EWB = NSB + shmem_layout::max_beams;

        int l = threadIdx.x;  // lane id
        
        for (int b = 32*threadIdx.y + threadIdx.x; b < nbeams; b += 24*32) {
            // Read grid locations from shared memory.
            float xns = shmem_f[NSB + b];  // 0 <= xns <= 128
            float xew = shmem_f[EWB + b];  // 1 <= xew <= 22

            // Split grid locations into integer/fractional parts.
            // (The values of 'xns', 'xew' are updated in-place to their fractional parts.)
            int ins = split_integer_and_fractional<Debug> (xns, 0, 127);
            int iew = split_integer_and_fractional<Debug> (xew, 1, 21);

            // The logic below does some nontrivial reshuffling, in order to reduce
            // shared memory bank conflicts. If we omitted reshuffling logic entirely,
            // we could get an arbitrarily bad bank conflict (32-to-1) which would
            // really be a disaster. The purpose of the reshuffling is to reduce the
            // bank conflict to 2-to-1 by permuting the 4-by-4 bicubic interpolation tile.
            
            // "Reference" index for bank conflicts, see below.
            // Constructed so that sref = (iew-1)*IS + (ins-1) mod 32.
            int sref = (iew << 2) + ins + 27;

            float ret = 0.0f;

            // Cubic interpolation.
            for (int jns = 0; jns < 4; jns++) {
                // Find kns such that (sref+kns) == (jns+l) mod 4
                int kns = (jns + l - sref) & 3;
                float wns = compute_wk(kns, xns);
                int sns = (ins + kns - 1) & 127;
                int sref2 = sref + kns;  // = (iew-1)*IS + (ins+kns-1) mod 32

                for (int jew = 0; jew < 4; jew++) {
                    // Find kew such that (sref2 + 4*kew) == (4*jew + jns + l) mod 16
                    int kew = ((4*jew + jns + l - sref2) >> 2) & 3;
                    float wew = compute_wk(kew, xew);
                    int sew = (iew + kew - 1);
                    int s = sew*IS + sns;

                    if constexpr (Debug) {
                        assert((s & 0xf) == ((4*jew+jns+l) & 0xf));
                        check_bank_conflict_free<Debug> (s, 2);  // max_conflicts=2
                    }

                    // Read gridded I-value from shared memory.
                    ret += wns * wew * shmem_f[IB + s];
                }
            }

            out[b] = normalization * ret;
            // __stcs(out + b, normalization * ret);  // streaming write made no difference here
        }

        // Advance output pointer.
        int F = gridDim.x;
        out += F * nbeams;
    }
};


// Unit test of 'struct interpolation_microkernel'.
__global__ void casm_interpolation_test_kernel(
    const float *i_in,                  // (Tout,F,24,128)
    const float *feed_weights,          // (F,2,256,2) = (freq,pol,dish,reim), dereferenced but not used
    const float *gpu_persistent_data,   // see "global memory layout" earlier in source file
    float *i_out,                       // (Tout,F,B)
    int Tout, int B, float normalization)
{
    extern __shared__ float shmem_f[];
    
    assert(blockDim.x == 32);
    assert(blockDim.y == 24);
    assert(blockDim.z == 1);
    assert(gridDim.y == 1);
    assert(gridDim.z == 1);
    assert((B % 32) == 0);

    // Debug=true, e_in=nullptr
    casm_controller<true> controller(nullptr, feed_weights, gpu_persistent_data, B);
    __syncthreads();  // must call __syncthreads() after calling constructor, and before calling interpolator.apply()

    interpolation_microkernel<true> interpolator(i_out, B, normalization);
    
    for (int t = 0; t < Tout; t++) {
        // Read gridded I-array (global memory) -> (shared).
        //   Global: i_in[Tout][F][24][128]
        //   Shared: float I[24][132];
        
        constexpr int IB = shmem_layout::I_base;
        constexpr int IS = shmem_layout::I_ew_stride;
        
        int w = threadIdx.y;  // warp id
        int l = threadIdx.x;  // lane id
        int f = blockIdx.x;   // current frequency
        int F = gridDim.x;    // total frequencies

        for (int ns = l; ns < 128; ns += 32)
            shmem_f[IB + w*IS + ns] = i_in[t*F*24*128 + f*24*128 + w*128 + ns];

        __syncthreads();

        // Reads from shared memory, writes to global memory, advances output pointer.
        interpolator.apply();
        
        __syncthreads();
    }
}


// Helper for interpolation_reference_kernel().
__host__ inline void compute_interpolation_weights(float dx, float w[4])
{
    static constexpr float one_sixth = 1.0f / 6.0f;
    static constexpr float one_half = 1.0f / 2.0f;
    
    w[0] = -one_sixth * (dx) * (dx-1.0f) * (dx-2.0f);
    w[1] = one_half * (dx+1.0f) * (dx-1.0f) * (dx-2.0f);
    w[2] = -one_half * (dx+1.0f) * (dx) * (dx-2.0f);
    w[3] = one_sixth * (dx+1.0f) * (dx) * (dx-1.0f);
}


// Unit test of 'struct interpolation_microkernel'.
// out.shape = (Tout,F,B)
// in.shape = (Tout,F,24,128)

static void interpolation_reference_kernel(const CasmBeamformer &bf, float *out, const float *in, int Tout, float normalization)
{
    int F = bf.F;
    int B = bf.B;

    for (int f = 0; f < F; f++) {
        for (int b = 0; b < B; b++) {
            constexpr float c = CasmBeamformer::speed_of_light;
            
            float freq = bf.frequencies[f];
            float sza_ns = bf.beam_locations[2*b];
            float sza_ew = bf.beam_locations[2*b+1];
            float dns = bf.ns_feed_spacing;
            
            // Grid coordinates (xns, xew).
            float xns = 128 * (freq/c) * dns * sza_ns;  // 128-periodic
            float xew = 11.5f + (10.5f * sza_ew);       // 1 <= xew <= 22

            // Normalize the periodic coordinate 'xns' to 0 <= xns <= 128.
            xns = fmodf(xns, 128.0f);
            xns = (xns > 0.0f) ? xns : (xns+128.0f);

            int ins = split_integer_and_fractional<true> (xns, 0, 127);
            int iew = split_integer_and_fractional<true> (xew, 1, 21);
            
            float wns[4], wew[4];
            compute_interpolation_weights(xns, wns);
            compute_interpolation_weights(xew, wew);

            for (int t = 0; t < Tout; t++) {
                float x = 0.0f;
                
                for (int kns = 0; kns < 4; kns++) {
                    int sns = (ins + kns - 1) & 127;
                    for (int kew = 0; kew < 4; kew++) {
                        int sew = (iew + kew - 1);
                        int sin = (t*F+f)*24*128 + 128*sew + sns;  // index in 'in' array
                        x += wns[kns] * wew[kew] * in[sin];
                    }
                }

                int sout = t*F*B + f*B + b;  // index in 'out' array
                out[sout] = normalization * x;
            }
        }
    }
}


// Unit test of 'struct interpolation_microkernel'.
static void test_casm_interpolation_microkernel(const CasmBeamformer &bf)
{
    // Allow kernel to use 99KB shared memory.
    static shmem_99kb s(casm_interpolation_test_kernel);
    s.set();

    int F = bf.F;
    int B = bf.B;
    int Tout = rand_int(1,5);
    float normalization = rand_uniform();
    
    cout << "test_casm_interpolation_microkernel(Tout=" << Tout << ", F=" << F
         << ", B=" << B << ", normalization=" << normalization << ")" << endl;

    // Note: feed_weights must be allocated, but are not actually used.
    shared_ptr<float> feed_weights = cpu_alloc<float> (F*2*256*2);  // shape (F,2,256,2)
    shared_ptr<float> in = cpu_alloc<float> (Tout*F*24*128, true);  // shape (Tout,F,24,128)
    shared_ptr<float> out_cpu = cpu_alloc<float> (Tout*F*B);        // shape (Tout,F,B)
    shared_ptr<float> out_gpu = gpu_alloc<float> (Tout*F*B);        // shape (Tout,F,B)

    // Run CPU reference kernel.
    interpolation_reference_kernel(bf, out_cpu.get(), in.get(), Tout, normalization);

    // Run GPU kernel.
    
    in = to_gpu(in, Tout*F*24*128);
    
    casm_interpolation_test_kernel<<< F, {32,24,1}, 99*1024 >>>
        (in.get(), feed_weights.get(), bf.gpu_persistent_data.get(), out_gpu.get(), Tout, B, normalization);

    CUDA_PEEK("casm_interpolation_test_kernel launch");

    // Check that arrays are equal.
    // out_cpu.shape == out_gpu.shape == (Tout,F,B)

    out_gpu = to_cpu(out_gpu, Tout*F*B);
    float *gp = out_gpu.get();
    float *cp = out_cpu.get();
    int pos = 0;

    for (int t = 0; t < Tout; t++) {
        for (int f = 0; f < F; f++) {
            for (int b = 0; b < B; b++) {
                if (fabsf(gp[pos] - cp[pos]) > 1.0e-4) {
                    stringstream ss;
                    ss << "failed: t=" << t << ", f=" << f << ", b=" << b
                       << ": cpu=" << cp[pos] << ", gpu=" << gp[pos];
                    throw runtime_error(ss.str());
                }
                pos++;
            }
        }
    }
}


// -------------------------------------------------------------------------------------------------
//
// Putting it all together: casm_beamforming_kernel.
//
// The number of input times (Tin) must be a multiple of 'downsampling_factor',
// i.e. Tin = Tout * downsampling_factor.
//
// The casm_beamforming_kernel is unit-tested from python, so there's no unit
// test in this source file.
//
// Launch with {32,24,1} threads/block, and F blocks/kernel.


__global__ void __launch_bounds__(24*32, 1)
casm_beamforming_kernel(
    const uint8_t *e_in,                // (Tin,F,2,D) = (time,freq,pol,dish)
    const float *feed_weights,          // (F,2,256,2) = (freq,pol,dish,reim)
    const float *gpu_persistent_data,   // see "global memory layout" earlier in source file
    float *i_out,                       // (Tout,F,B)
    int Tout,                           // Number of output times
    int Tds,                            // Downsampling factor Tin/Tout
    int B,                              // Number of beams
    float normalization)                // Currently equal to 1/(2*Tds)
{
    constexpr bool Debug = false;
    
    casm_controller<Debug> controller(e_in, feed_weights, gpu_persistent_data, B);

    // Must call __syncthreads() after constructor, and before calling load_ungridded_e().
    __syncthreads();

    fft1_microkernel<Debug> fft1;
    fft2_microkernel<Debug> fft2;
    interpolation_microkernel<Debug> interpolator(i_out, B, normalization);

    int Tin = Tout * Tds;
    int ds_counter = Tds;
    int touter = -48;

    // In each iteration of the outer loop:
    //
    //   - read input times (touter+48):(touter+72)
    //   - process input times (touter):(touter+24)
    //   - the value of 'touter' advances by 24
    //
    // The loop runs over -48 <= t < Tin, but the first two iterations just load
    // data from global memory (no processing), and the last two iterations just
    // process pre-loaded data (no loading).

    for (;;) {
        // Every 24 time samples, start loading E-array from global memory.
        // No-ops on threads which would read past the end of input data.
        // No need for __syncthreads() here, since we're just reading into registers.
        controller.load_ungridded_e(touter+48, Tin);

        // outer_phase = {0,1}, depending on whether touter = {0,24} mod 48.
        int outer_phase = (touter & 0x8) >> 3;

        if (touter < 0)
            goto write_e;  // hmmm, 'goto' seems to be least ugly option here

        if (outer_phase == 0) {
            // Every 48 time samples, we "grid" the E-array in shared memory.
            __syncthreads();   // wait for write_ungridded_e() in previous loop iteration.
            controller.grid_shared_e();
            __syncthreads();   // barrier before calling load_gridded_e() below.
        }

        // In each iteration of the middle loop, we process times (touter+8*m):(touter+8*m+8).
        for (int m = 0; m < 3; m++) {
            // Every 8 time samples, we read the E-array from shared memory -> registers.
            // No syncthreads() needed before or after load_gridded_e(), since:
            //   - grid_shared_e() has syncthreads() after it, see above
            //   - write_ungridded_e() has syncthreads() before it, see above
            
            controller.load_gridded_e(3*outer_phase + m);  // phase = 3*flag + m

            for (int tpol = 0; tpol < 16; tpol++) {
                if ((tpol & 3) == 0) {
                    // Every 2 time samples, we "unpack" the E-array from int4+4 -> float32+32,
                    // apply feed weights, run the FFT1 kernel (beamforming 43->128 along NS axis),
                    // and write the result to G[] shared memory.
                    
                    float e0_re, e0_im, e1_re, e1_im;
                    controller.unpack_e(tpol >> 2, e0_re, e0_im, e1_re, e1_im);
                    
                    __syncthreads();  // wait for calls to fft2.apply() in previous loop iteration
                    fft1.apply(e0_re, e0_im, e1_re, e1_im);
                    __syncthreads();  // barrier before more calls to fft2.apply()
                }

                // For every (time,pol), we run the FFT2 kernel (beamform 6->24 along EW axis),
                // square, and accumulate the result into persistent registers which store
                // beamformed intensities on a (24,128) grid.
                fft2.apply(tpol & 3);
                
                if ((tpol & 1) && !(--ds_counter)) {
                    // When the number of accumulated time samples is a multiple of 'downsampling_factor',
                    // we write the (24,128) grid of beamformed intensities to I[] shared memory, interpolate
                    // to the target beam locations, and write the result to global memory.
                    
                    __syncthreads();         // wait for previous interpolator.apply()
                    fft2.write_and_reset();  // writes I-array to shared memory
                    __syncthreads();
                    interpolator.apply();    // reads I-array from shared, writes to global memory
                    ds_counter = Tds;

                    int tinner = touter + 8*m + ((tpol+1) >> 1);

                    // This is the exit point from the kernel.
                    if (tinner >= Tin)
                        return;
                }
            }
        }

    write_e:
        if (touter + 48 < Tin) {
            // Every 24 samples, we write the ungridded E-array from registers -> shared memory.
            // Note that the load (global -> registers) is at the top of the loop, and the store
            // (registers -> shared) is at the bottom. This way of organizing things is intended
            // to hide the latency of global memory, but may not actually matter, since the kernel
            // is not memory bandwidth limited anyway.
            
            __syncthreads();  // wait for load_gridded_e() above
            controller.write_ungridded_e(outer_phase);
        }

        // In each iteration of the outer loop, the value of 'touter' advances by 24.
        touter += 24;
    }
}


// -------------------------------------------------------------------------------------------------
//
// class CasmBeamformer


CasmBeamformer::CasmBeamformer(
    const Array<float> &frequencies_,     // shape (F,)
    const Array<int> &feed_indices_,      // shape (256,2)
    const Array<float> &beam_locations_,  // shape (B,2)
    int downsampling_factor_,
    float ns_feed_spacing_,
    const Array<float> &ew_feed_spacings_)
{
    // Argument checking
    
    xassert_eq(frequencies_.ndim, 1);
    xassert_eq(beam_locations_.ndim, 2);
    xassert_eq(beam_locations_.shape[1], 2);
    xassert_shape_eq(feed_indices_, ({256,2}));

    xassert(frequencies_.is_fully_contiguous());
    xassert(feed_indices_.is_fully_contiguous());
    xassert(beam_locations_.is_fully_contiguous());

    xassert(frequencies_.on_host());
    xassert(feed_indices_.on_host());
    xassert(beam_locations_.on_host());

    if (ew_feed_spacings_.size > 0) {
        xassert_shape_eq(ew_feed_spacings_, ({5}));
        xassert(ew_feed_spacings_.is_fully_contiguous());
        xassert(ew_feed_spacings_.on_host());
    }

    // Delegate to bare-pointer constructor.
    
    this->_construct(
        frequencies_.data,
        feed_indices_.data,
        beam_locations_.data,
        downsampling_factor_,
        frequencies_.shape[0],     // F = number of frequency channels
        beam_locations_.shape[0],  // B = number of beams
        ns_feed_spacing_,
        (ew_feed_spacings_.size > 0) ? ew_feed_spacings_.data : nullptr
    );
}


CasmBeamformer::CasmBeamformer(
    const float *frequencies_,        // shape (nfreq,)
    const int *feed_indices_,         // shape (256,2)
    const float *beam_locations_,     // shape (nbeams,2)
    int downsampling_factor_,
    int nfreq_,
    int nbeams_,
    float ns_feed_spacing_,
    const float *ew_feed_spacings_)
{
    this->_construct(
        frequencies_, feed_indices_, beam_locations_,
        downsampling_factor_, nfreq_, nbeams_,
        ns_feed_spacing_, ew_feed_spacings_
    );
}


template<typename T>
static vector<T> copy_to_vector(const T *src, long nelts)
{
    vector<T> ret(nelts);
    memcpy(&ret[0], src, nelts * sizeof(T));
    return ret;
}

void CasmBeamformer::_construct(
    const float *frequencies_,       // shape (F,)
    const int *feed_indices_,        // shape (256,2)
    const float *beam_locations_,    // shape (B,2)
    int downsampling_factor_,
    int nfreq_,
    int nbeams_,
    float ns_feed_spacing_,
    const float *ew_feed_spacings_)  // either shape (5,) or NULL
{
    this->F = nfreq_;
    this->B = nbeams_;
    this->downsampling_factor = downsampling_factor_;
    this->ns_feed_spacing = ns_feed_spacing_;

    if (!frequencies_)
        throw runtime_error("CasmBeamformer constructor: 'frequencies' pointer is NULL");
    if (!feed_indices_)
        throw runtime_error("CasmBeamformer constructor: 'feed_indices' pointer is NULL");
    if (!beam_locations_)
        throw runtime_error("CasmBeamformer constructor: 'beam_locations' pointer is NULL");
    
    if (!ew_feed_spacings_)
        ew_feed_spacings_ = CasmBeamformer::default_ew_feed_spacings;

    if (F <= 0)
        throw runtime_error("CasmBeamformer constructor: num_freqs must be > 0");
    if (B <= 0)
        throw runtime_error("CasmBeamformer constructor: num_beams must be > 0");
    if (B % 32)
        throw runtime_error("CasmBeamformer constructor: num_beams must be a multiple of 32");
    if (downsampling_factor <= 0)
        throw runtime_error("CasmBeamformer constructor: downsampling_factor must be > 0");
    
    if (B > shmem_layout::max_beams) {
        stringstream ss;
        ss << "CasmBeamformer constructor: num_beams(=" << B << ") must be <="
           << " " << shmem_layout::max_beams << ". (This limit is set by GPU"
           << " shared memory considerations.)";
        throw runtime_error(ss.str());
    }

    if ((ns_feed_spacing < 0.3) || (ns_feed_spacing > 0.6)) {
        stringstream ss;
        ss << "CasmBeamformer constructor: ns_feed_spacing(=" << ns_feed_spacing
           << ") must be in the range [0.3, 0.6]. Note that the units are meters, not cm!";
        throw runtime_error(ss.str());
    }

    CUDA_CALL(cudaGetDevice(&this->constructor_device));
    
    this->frequencies = copy_to_vector(frequencies_, F);
    this->feed_indices = copy_to_vector(feed_indices_, 256*2);
    this->beam_locations = copy_to_vector(beam_locations_, B*2);

    for (int i = 0; i < 5; i++) {
        this->ew_feed_spacings[i] = ew_feed_spacings_[i];
        
        if ((ew_feed_spacings[i] < 0.3) || (ew_feed_spacings[i] > 0.6)) {
            stringstream ss;
            ss << "CasmBeamformer constructor: ew_feed_spacings (" << ew_feed_spacings[0];
            for (int j = 1; j < 5; j++)
                ss << ", " << ew_feed_spacings[j];
            ss << ") must all be in the range [0.3, 0.6]. Note that the units are meters, not cm!";
            throw runtime_error(ss.str());
        }

        if (fabsf(ew_feed_spacings_[i] - ew_feed_spacings_[4-i]) > 1.0e-5) {
            stringstream ss;
            ss << "CasmBeamformer constructor: ew_feed_spacings (" << ew_feed_spacings[0];
            for (int j = 1; j < 5; j++)
                ss << ", " << ew_feed_spacings[j];
            ss << ") must be \"flip-symmetric\"";
            throw runtime_error(ss.str());
        }
    }
    
    float s0 = (ew_feed_spacings[0] + ew_feed_spacings[4]) / 2.;
    float s1 = (ew_feed_spacings[1] + ew_feed_spacings[3]) / 2.;
    float s2 = (ew_feed_spacings[2]);

    // Use EW coordinates such that the array is centered at zero.
    ew_feed_positions[0] = -s0 - s1 - s2/2.;
    ew_feed_positions[1] =     - s1 - s2/2.;
    ew_feed_positions[2] =          - s2/2.;
    ew_feed_positions[3] =            s2/2.;
    ew_feed_positions[4] =       s1 + s2/2.;
    ew_feed_positions[5] =  s0 + s1 + s2/2.;

    // EW beam locations (these are sin(ZA) values, not ZA).
    // Note that ew_beam_locations[1] = -1, and ew_beam_locations[0] < -1.
    // Similarly, ew_beam_locations[22] = 1, and ew_beam_locations[23] > 1.
    // This padding (by one element) is needed for cubic interpolation.
    for (int i = 0; i < 24; i++)
        ew_beam_locations[i] = (2*i-23) / 21.;
    
    // The rest of this function fills 'gpu_persistent_data' and copies to the GPU.
    // See comment near the beginning of this file for the memory layout.
    gpu_persistent_data = cpu_alloc<float> (gmem_layout::nelts(F,B));
    
    // Compute 'gridding' part of 'gpu_persistent_data' (with error-checking).
    // This contains the same info as the 'feed_indices' arg, just reparameterized
    // as (ns,ew) -> (43*ew+ns).
    
    uint *gp = (uint *) (gpu_persistent_data.get());
    vector<int> duplicate_checker({6*43}, -1);
    
    for (int d = 0; d < 256; d++) {
        int ns = feed_indices[2*d];    // 0 <= ns < 43
        int ew = feed_indices[2*d+1];  // 0 <= ew < 6

        if ((ns < 0) || (ns >= 43) || (ew < 0) || (ew >= 6)) {
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
    // This is just the phase cos(2pi*i/128) for 0 <= i < 32, which gets used
    // in the length-128 FFT. Precomputing these phases on the host is faster,
    // since it avoids the overhead of trig functions in the GPU kernel.
    
    float *nsp = gpu_persistent_data.get() + shmem_layout::ns_phases_base;
    for (int i = 0; i < 32; i++)
        nsp[i] = cosf(2 * M_PI * i / 128.0);

    // Compute 'per_frequency_data' part of 'gpu_persistent_data'.
    // See (*) near the beginning of this file for details.
    
    for (int f = 0; f < F; f++) {
        float freq = frequencies[f];

        // Beamformer has only been validated for frequencies in [400,500] MHz.
        // This check also guards against using the wrong units (should be MHz).
        
        if ((freq < 399.0) || (freq > 501.0)) {
            stringstream ss;
            ss << "CasmBeamformer constructor: got freq=" << freq << ", expected all frequencies"
               << " to be in the range [400,500]. The beamforming algorithm has only been validated"
               << " in this range. Note that units are MHz!";
            throw runtime_error(ss.str());
        }

        for (int ew_feed = 0; ew_feed < 3; ew_feed++) {
            // Points to 32-element "inner" region (see (*) above).
            float *pf32 = gpu_persistent_data.get() + gmem_layout::per_frequency_data_base(f) + 32*ew_feed;
            pf32[25] = freq;

            // Instead of passing the ew_beam_locations and ew_feed_positions to the GPU
            // kernel, we precompute the phases sincos(2pi/c * freq * feed_pos * beam_loc).
            // This uses a little bit of GPU memory (~200 KB), but speeds things up by
            // avoiding the overhead of trig functions in the GPU kernel.
            
            for (int ew_beam = 0; ew_beam < 12; ew_beam++) {
                float feed_pos = ew_feed_positions[ew_feed + 3];
                float beam_loc = ew_beam_locations[ew_beam + 12];
                float theta = (2*M_PI / speed_of_light) * freq * feed_pos * beam_loc;
                sincosf(theta, &pf32[2*ew_beam+1], &pf32[2*ew_beam]);  // note (sin, cos) ordering
            }
        }
    }

    // Compute 'beam_locations' part of gpu_persistent_data.
    float *blp = gpu_persistent_data.get() + gmem_layout::beam_locs_base(F);
    
    for (int b = 0; b < B; b++) {
        for (int j = 0; j < 2; j++) {
            // We store {ns_feed_spacing * sin(za_ns), sin(za_ew)} in GPU memory.
            // See (**) near the beginning of this file.
            
            float prefactor = j ? 1.0 : ns_feed_spacing;
            float beam_location = beam_locations[2*b+j];

            if ((beam_location < -1.0) || (beam_location > 1.0)) {
                stringstream ss;
                ss << "CasmBeamformer constructor: got beam_location = ("
                   << beam_locations[2*b] << ", " << beam_locations[2*b+1]
                   << "), expected both coordinates to be in range [-1,1]."
                   << " Note that beam coordinates are sines of zenith angles.";
                throw runtime_error(ss.str());
            }
            
            blp[2*b+j] = prefactor * beam_location;
        }
    }

    // All done! Copy to GPU.
    gpu_persistent_data = to_gpu(gpu_persistent_data, gmem_layout::nelts(F,B));
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
         << "[" << SL::G_base << "]   float G[8][" << SL::G_reim_stride << "];\n"
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
shared_ptr<int> CasmBeamformer::make_random_feed_indices()
{
    // It's convenient to overallocate (256 -> 258).
    shared_ptr<int> feed_indices = cpu_alloc<int> (43*6*2);
    int *fp = feed_indices.get();

    for (uint ns = 0; ns < 43; ns++) {
        for (uint ew = 0; ew < 6; ew++ ) {
            int i = 6*ns + ew;
            fp[2*i] = ns;
            fp[2*i+1] = ew;
        }
    }

    // Randomly permute.
    for (int i = 1; i < 43*6; i++) {
        int j = rand_int(0, i+1);
        std::swap(fp[2*i], fp[2*j]);
        std::swap(fp[2*i+1], fp[2*j+1]);
    }

    return feed_indices;
}


// Static member function.
shared_ptr<int> CasmBeamformer::make_regular_feed_indices()
{
    shared_ptr<int> feed_indices = cpu_alloc<int> (256*2);
    int *fp = feed_indices.get();
    
    for (uint d = 0; d < 256; d++) {
        fp[2*d] = d % 43;    // ns
        fp[2*d+1] = d / 43;  // ew
    }

    return feed_indices;
}


// Static member function.
CasmBeamformer CasmBeamformer::make_random(bool randomize_feed_indices)
{
    auto w = random_integers_with_bounded_product(3, 1000);
    int B = 32 * rand_int(1,100);
    int T = w[0] * w[1];   // number of input time samples
    int ds = w[1];         // downsampling factor
    int F = w[2];          // number of frequency channels

    vector<float> frequencies(F);
    vector<float> beam_locations(B*2);
    vector<float> ew_feed_spacings(5);
    float ns_feed_spacing = rand_uniform(0.3, 0.6);

    // Make random 'frequencies' array.
    for (int f = 0; f < F; f++)
        frequencies.at(f) = rand_uniform(400.0, 500.0);
    
    // Make random 'beam_locations' array.
    for (int b = 0; b < B; b++)
        for (int j = 0; j < 2; j++)
            beam_locations.at(2*b+j) = rand_uniform(-1.0, 1.0);

    // Make random 'ew_feed_spacings' array.
    for (int i = 0; i < 3; i++)
        ew_feed_spacings.at(i) = ew_feed_spacings.at(4-i) = rand_uniform(0.3, 0.6);
    
    shared_ptr<int> feed_indices = randomize_feed_indices ? make_random_feed_indices() : make_regular_feed_indices();

    CasmBeamformer ret(&frequencies[0], feed_indices.get(), &beam_locations[0], ds, F, B, ns_feed_spacing, &ew_feed_spacings[0]);
    ret.nominal_Tin_for_unit_tests = T;
    
    return ret;
}


void CasmBeamformer::launch_beamformer(
    const Array<uint8_t> &e_in,
    const Array<float> &feed_weights,
    Array<float> &i_out,
    cudaStream_t stream) const
{
    xassert(e_in.ndim >= 1);    
    xassert_shape_eq(e_in, ({ e_in.shape[0], F, 2, 256 }));
    xassert(e_in.is_fully_contiguous());
    xassert(e_in.on_gpu());

    xassert_shape_eq(feed_weights, ({ F,2,256,2 }));
    xassert(feed_weights.is_fully_contiguous());
    xassert(feed_weights.on_gpu());

    xassert(i_out.ndim >= 1);
    xassert_shape_eq(i_out, ({ i_out.shape[0], F, B }));
    xassert(i_out.is_fully_contiguous());
    xassert(i_out.on_gpu());

    int Tin = e_in.shape[0];
    int Tout = i_out.shape[0];
    xassert(Tin == Tout * downsampling_factor);

    this->launch_beamformer(e_in.data, feed_weights.data, i_out.data, Tin, stream);
}


void CasmBeamformer::launch_beamformer(
    const uint8_t *e_arr,        // shape (Tin,F,2,256), axes (time,freq,pol,dish)
    const float *feed_weights,   // shape (F,2,256,2), axes (freq,pol,dish,reim)
    float *i_out,                // shape (Tout,F,B)
    int Tin,                     // number of input times Tin = Tout * downsampling_factor
    cudaStream_t stream) const
{
    // Allow kernel to use 99KB shared memory.
    static shmem_99kb s(casm_beamforming_kernel);
    s.set();
    
    if (!e_arr)
        throw runtime_error("CasmBeamformer::launch_beamformer(): 'e_arr' pointer is NULL");
    if (!feed_weights)
        throw runtime_error("CasmBeamformer::launch_beamformer(): 'feed_weights' pointer is NULL");
    if (!i_out)
        throw runtime_error("CasmBeamformer::launch_beamformer(): 'i_out' pointer is NULL");

    if (Tin <= 0)
        throw runtime_error("CasmBeamformer::launch_beamformer(): 'Tin' must be > 0");
    
    int Tout = Tin / downsampling_factor;
    if (Tin != Tout * downsampling_factor) {
        stringstream ss;
        ss << "CasmBeamformer::launch_beamformer: Tin=" << Tin << " must be a multiple"
           << " of downsampling_factor=" << downsampling_factor;
        throw runtime_error(ss.str());
    }

    int d = -2;
    CUDA_CALL(cudaGetDevice(&d));
    
    if (this->constructor_device != d) {
        stringstream ss;
        ss << "CasmBeamformer: CUDA device at construction (dev=" << this->constructor_device << ")"
           << " differs from CUDA device in launch_beamformer() (dev=" << d << "). This is an"
           << " error, since the constructor stores precomputations on the GPU.";
        throw runtime_error(ss.str());
    }
    
    float normalization = 1.0f / (2.0f * downsampling_factor);

    casm_beamforming_kernel<<< F, {32,24,1}, 99*1024, stream >>>
        (e_arr, feed_weights, gpu_persistent_data.get(),
         i_out, Tout, downsampling_factor, B, normalization);

    CUDA_PEEK("casm_beamforming_kernel launch");    
}


// Static member function.
void CasmBeamformer::test_microkernels()
{
    // CasmBeamformer::show_shared_memory_layout();
    CasmBeamformer bf = CasmBeamformer::make_random();
    
    test_casm_controller(bf);
    test_casm_fft_c2c_microkernel();
    test_casm_fft1_microkernel();
    test_casm_fft2_microkernel(bf);
    test_casm_interpolation_microkernel(bf);
}


// Static member function.
void CasmBeamformer::time()
{
    long F = 512;          // frequency channels per gpu
    long D = 32;           // time downsampling factor
    long Tout = 512;       // output time samples per kernel launch
    long Tin = D * Tout;   // number of input time samples
    long niter = 20;

    vector<long> Bvec = {32,1024,2048,3072,4096};   // number of beams
    vector<double> lfvec(Bvec.size());
    
    double ts = 4096 / (125 * 1.0e6);    // time sampling rate, in seconds
    double dt_rt = Tin * ts;             // real time elapsed per kernel launch, in seconds

    vector<float> frequencies(F);
    shared_ptr<int> feed_indices = CasmBeamformer::make_regular_feed_indices();
    shared_ptr<uint8_t> e_in = gpu_alloc<uint8_t> (Tin*F*2*256);
    shared_ptr<float> feed_weights = gpu_alloc<float> (F*2*256*2);
    
    for (long f = 0; f < F; f++)
        frequencies.at(f) = rand_uniform(400.0, 500.0);

    for (ulong ib = 0; ib < Bvec.size(); ib++) {
        int B = Bvec[ib];
        cout << "Starting timing, B=" << B << ", Tin=" << Tin << endl;
        cout << "Input array: " << (1.0e-9 * Tin * F * 2 * 256) << " GB" << endl;
        cout << "Output array: " << (4.0e-9 * Tout * F * B) << " GB" << endl;

        vector<float> beam_locations(B*2);
        shared_ptr<float> i_out = gpu_alloc<float> (Tout*F*B);

        for (long b = 0; b < B; b++)
            for (long j = 0; j < 2; j++)
                beam_locations.at(2*b+j) = rand_uniform(-1.0, 1.0);
           
        CasmBeamformer bf(&frequencies[0], feed_indices.get(), &beam_locations[0], D, F, B);
        vector<struct timeval> tv(niter);
        
        for (long i = 0; i < niter; i++) {
            CUDA_CALL(cudaDeviceSynchronize());
            
            int err = gettimeofday(&tv[i], NULL);
            if (err != 0)
                throw runtime_error("gettimeofday() failed?!");
    
            bf.launch_beamformer(e_in.get(), feed_weights.get(), i_out.get(), Tin);
            
            long k = i - (i/2);
            if (i > k) {
                double dt = (tv[i].tv_sec - tv[k].tv_sec) + 1.0e-6 * (tv[i].tv_usec - tv[k].tv_usec);
                double loadfrac = dt / ((i-k) * dt_rt);
                cout << "    " << i << " iterations, loadfrac = " << loadfrac << " (lower is better)" << endl;
                lfvec[ib] = loadfrac;
            }
        }
    }

    cout << "# B loadfrac\n";
    for (ulong ib = 0; ib < Bvec.size(); ib++)
        cout << Bvec[ib] << " " << lfvec[ib] << endl;
}


}  // namespace pirate
