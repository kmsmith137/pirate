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
