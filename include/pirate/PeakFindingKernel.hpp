#ifndef _PIRATE_PEAK_FINDING_KERNEL_HPP
#define _PIRATE_PEAK_FINDING_KERNEL_HPP

#include <vector>

// #include <ksgpu/Dtype.hpp>
// #include <ksgpu/Array.hpp>
// #include "trackers.hpp"  // BandwidthTracker


namespace pirate {
#if 0
}  // editor auto-indent
#endif


struct PeakFindingKernelParams
{
    // dtype=float for now
    long dm_downsampling_factor = 0;
    long time_downsampling_factor = 0;
    long max_kernel_width = 0;
    long beams_per_batch = 0;
    long nt_in = 0;

    void validate() const;  // throws an exception if anything is wrong
};


struct GpuPeakFindingKernel
{
    // Placeholder.
};


// pf_kernel: represents a precompiled low-level cuda kernel.
//
// Each .cu source file populates a 'struct pf_kernel' with function pointers,
// then calls pf_kernel::register().

struct pf_kernel {
    
    // We define two low-level cuda kernels: a "full peak-finding kernel" which is externally
    // useful, and a "reduce-only" kernel which serves an internal debugging purpose.
    //
    // The full peak-finding kernel is called as:
    //
    //   void pf_full(out_max, out_ssq, pstate, in, wt, Mout, Tout);
    //
    // and the reduce-only kernel is called as:
    //
    //   void pf_reduce(out_max, out_ssq, in_max, in_ssq, wt, Mout, Tout);
    //
    // where:
    //
    //  - 'out_max' and 'out_ssq' have shape (B, P, Mout, Tout).
    //  - 'in_max' and 'in_ssq' have shape (B, P, Mout*M, Tout*(Dout/Dcore))
    //  - 'pstate' has shape (B, Mout, RW).
    //  - 'in' has shape (B, Mout*M, Tout*Dout).
    //  - 'wt' has shape (B, P, Mout*M).
    //  - 'B' is the number of beams, and other params (P, M, ...) are defined below
    //
    // Kernels are launched with {BM,B,1} blocks and {32*W,1} threads, where BM = ceil(Mout/W).
    // Shared memory is statically allocated.

    int M = 0;       // same as PeakFindingKernelParams::dm_downsampling_factor
    int E = 0;       // same as PeakFindingKernelParams::max_kernel_width
    int Dout = 0;    // same as PeakFindingKernelParams::time_downsampling_factor
    int Dcore = 0;   // internal downsampling factor
    int W = 0;       // warps per threadblock
    int P = 0;       // number of peak-finding kernels (= 3*log2(E) + 1)
    int RW = 0;      // ring buffer elements per output (beam, mout)

    // The "full peak-finding" and "reduce-only" kernels have the same call signatures
    // (five pointers and two ints), but the meaning of the args is different, see above.
    using cuda_kernel_t = void (*) (float *, float *, float *, float *, float *, int, int);
    
    cuda_kernel_t full_kernel = nullptr;
    cuda_kernel_t reduce_only_kernel = nullptr;
    
    pf_kernel *next = nullptr;   // used internally
    bool debug = false;          // a "debug" kernel takes precedence over non-debug
    
    void register_kernel();

    // Throw exception if no registered kernel can be found.
    static pf_kernel get(int M, int E, int Dout);

    // Returns all registered kernels.
    static std::vector<pf_kernel> enumerate();
};


}  // namespace pirate

#endif // _PIRATE_AGGED_DOWNSAMPLING_KERNEL_HPP
