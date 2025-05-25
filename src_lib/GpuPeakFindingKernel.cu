#include <ksgpu/constexpr_functions.hpp>
#include <ksgpu/device_transposes.hpp>   // warp_transpose(), FULL_MASK

// Inputs to peak-finding kernel (and their constraints) are as follows.
//
//   - Dt = time downsampling factor. Must be a power of two, and <= 16.
//
//   - Dd = DM downsampling factor. Must equal either 1 or Dt.
//
//   - E = a "width" parameter. Must be a power of two, and <= Dt.
//     The E-parameter determines the number of peak-finding kernels:
//
//      - If E=1: P=1 (single sample)
//      - If E=2: P=4 (+ length-2 boxcar + length-3 Gaussian + length-4 Gaussian)
//      - If E=4: P=7 (+ length-4 boxcar + length-6 Gaussian + length-12 Gaussian)
//      - If E=8: P=10 (+ length-8 boxcar + length-12 Gaussian + length-16 Gaussian)
//      - If E=16: P=13 (+ length-16 boxcar + length-24 Gaussian + length-32 Gaussian)
//
//   - data = array of shape (M,T), output from dedispersion transform.
//
//     Here, M = (number of DMs) must be a multiple of Dd, and
//     T = (number of time samples) must be a multiple of Dt.
//
//     (Throughout the peak-finding code, we denote DM-indices by m (not d)
//      to avoid confusion with downsampling indices.)
//
//   - weights = array of shape (P,M), applied after peak finding.
//
// Outputs of peak-finding kernel:
//
//   - pf_out: shape (P, M/Dd, T/Dt) array containing peak-finding output,
//     after taking "max" over coarse-grained axes.
//
//   - pf_var: shape (Pout, M/Dd, T/Dt) array containing peak-finding variance,
//     after taking mean-square over all coarse-grained axes.
//
//     Note: the output arrays are "lagged" by one coarse time sample, relative
//     to the input arrays.
//
// This design suffices to implement near-future cases:
//
//   - "Version 0" of the pipeline will just process data from disk, one
//     beam at a time. In this case a simple variance estimation strategy
//     is to not coarse-grain over dm (i.e. set Dd=1), median-filter the
//     'pf_var' array, and do a second pass to normalize and coarse-grain
//     the 'pf_out'array. (Needs some new kernels, but I wonder if I could
//     just write these in cupy.)
//
//     This approach won't work in the full-blown real-time pipeline, for
//     both speed and GPU memory reasons, but should suffice to start
//     developing RFI removal.
//
//   - Studies of variance estimation algorithms, for the full-blown
//     real-time pipeline.
//
//   - "Profiling mode", where the 
//
// Limitations of the current design:
//
//   - Weights are assumed to be constant in time.
//
//   - No "feedback loop" between the 'pf_var' and 'weights' arrays
//
//   - Output arrays 'pf_out' and 'pf_var' are required (maybe one/both
//     should be allowed to be null?)
//
//   - Contraints (Dt <= 16) and Dd=(1 or Dt) are just for convenience,
//     and could be relaxed if needed.
//
//   - No ability to coarse-grain over the length-P axis (representing
//     choice of peak-finding kernel).
//
// Since I know I'll be revisiting this code soon, but I don't know exactly what
// changes will be needed, I tried to make the code modular and well-documented!



// -------------------------------------------------------------------------------------------------
//
// class pf_tile: slightly higher level than pf_core
//
//  - Input is an array x[M], where indices 0 <= m < M represent trial
//    DMs, and threads/simd-lanes represent time. The pf_tile supplies
//    the transposes and outer loops needed by pf_core.
//
//  - Fully coarse-grains, applies weights, and writes results to GPU
//    global memory. Therefore, pf_tile::advance() returns void.
//
//  - Works with trial DMs, whereas 'pf_core' works with abstract spectator
//    indices. When we implement subbands, pf_core should stay the same,
//    whereas pf_tile may change.
//
//  - The value of M (number of trial DMs per warp) is supplied by the
//    caller, but there is a technical constraint that (M % Core::S) == 0.
//    (If this constraint is violated, then a static_assert will fail.)


template<typename T32, int M, int Dd, int Dt, int E, bool cg_pf>
struct pf_tile
{
    static_assert(ksgpu::constexpr_is_pow2(M));
    xxx;  // Lots more asserts!

    using Core = pf_core<T32,Dt,E>;    
    static_assert(ksgpu::constexpr_is_divisible(M, Core::S));
    
    static constexpr int C = M / Core::S;  // number of cores
    using Ringbuf = pf_ringbuf<T32, Core::RBI, C * Core::RBO>;

    
    __device__ inline void advance(T32 x[M], Ringbuf &ringbuf)
    {
	// Main steps:
	//  transpose
	//  Core::advance()
	//  apply weights
	//  coarse-grain and write

	// *** Let's consider the first two steps only ***

	// Case 1: float32
	// Transpose t0 <-> r0, t1 <-> r1, ..., t(L-1) <-> r(L-1), where L = log2(Dt)
	// Call Core::advance() in contiguous blocks of (M/Dt) registers

	// Case 2: float16, Dt > 1
	// Transpose b0 <-> r0, t0 <-> r1, ..., t(L-2) <-> r(L-1)
	// Call Core::advance() in contiguous blocks of (M/Dt) registers

	// Case 3: float16, Dt==1 (which implies Dd==1)
	// Group x's into consecutive pairs (representing consecutive trial DMs).
	// Do a special-case transpose
	// Call Core::advance() on even m's, then odd m's.

	// *** Next step: applying weights ***
	// Looks straightforward, should it be in pf_core or here?

	// *** Next step: coarse-grain and write ***
	// Let's do something poorly optimized for now.
	//
	// For each pf_register
	//  - write code path depends on whether Dd==1
	//
	// Hmmm not totally happy with this.

	// Now we just cycle through, in batches of Dt.
	for constexpr (I) {
	    Core::advance<I> (&x[I*XX], tile.ringbuf, weights + I*XX, wstride);
	}
    }
};


// -------------------------------------------------------------------------------------------------


template<typename T32, int D, int E, int M>
__global__ void pf_kernel(...)
{
    using Tile = pf_tile<T32, D, E, M>;
    Tile tile(...);

    // Load shmem
    // Hmm should we transpose the weights array?
    
    for (...) {
	T32 x[M] = load...;
	
	if constexpr (...) {
	    tile.advance();
	}
	else {
	    tile.advance();
	    tile.advance();
	}
    }

    // Save shmem
};
