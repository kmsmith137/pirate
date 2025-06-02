#include <ksgpu/constexpr_functions.hpp>
#include <ksgpu/device_transposes.hpp>   // warp_transpose(), FULL_MASK


// Defined in include/pirate/cuda_kernels/peak_finding.hpp
// Instantiated in src_lib/template_instantiations/*.cu

template<typename T, int Dd, int Dt, >
    
// Defined in peak_finding_kernel.hpp
template<typename T>
extern 


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
