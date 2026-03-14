#include "../include/pirate/ChimeBeamformer.hpp"

#include <algorithm>   // std::max
#include <cmath>       // cos, sin, M_PI
#include <cuda_fp16.h>
#include <cufftdx.hpp>

#include <iostream>

#include <ksgpu/xassert.hpp>
#include <ksgpu/cuda_utils.hpp>   // CUDA_PEEK
#include <ksgpu/KernelTimer.hpp>

using namespace std;
using namespace ksgpu;

namespace pirate {
#if 0
}  // editor auto-indent
#endif


// Currently, the kernel doesn't compile, due to the xxx's in the code.
// When you're ready to try compiling it, remove this #if 0 ... #endif.
#if 0

// This is the "beamforming" part of the CHIME FRB beamforming kernel.
// It runs before the "upchannelization" part.
//
// The input data is the uint4+4 electric field array, sampled at 4x256 feed
// locations. The length-4 axis is "east-west" and the length-256 axis
// is "north-south". The output is a float16+16 array of beamformed electric
// fields, at 4x256 beam locations.
//
//  'inputData':  shape (T,F,2,4,256), dtype uint4+4, axes (time,freq,pol,ew,ns)
//  'map':        shape (F,256), dtype uint, axes (freq,ns)
//  'co':         shape (F,4,4,2), dtype float, axes (freq,ewout,ewin,ReIm)
//  'outputData': shape (T,F,2,4,256), dtype float16+16, axes (time,freq,pol,ew,ns)
//  'gains':      shape (F,2,4,256), dtype float32+32, axes (freq,pol,ew,ns)
//
// Computational steps are as follows. Each (time,freq,pol) is processed independently,
// so we streamline notation by omitting those axes.
//
//  1. Unpack each uint4+4 inputData element to a float32+32.
//       real part = float(x >> 4) - 8.0
//       imag part = float(x & 0xf) - 8.0
//
//  2. Multiply by the complex conjugate of the associated gain.
//     (Denote the product by F = G^* E).
//
//  3. East-west beamforming: for each (time,freq,pol,ns), we have a length-4
//     complex array F[4] indexed by EW feed location (ew_in). We matrix-multiply
//     by the 4-by-4 complex matrix co[4,4], to get a complex array H[4].
//
//       H[ew_out] = (1/4) sum_j co[ew_out,ew_in] F[ew_in]^*
//
//     The H-array can be interpreted as "partially beamformed electric field":
//     beamforming has been performed along the ew-axis, but not the ns-axis.
//
//  4. North-south beamforming: restoring implicit axes, we now have an array
//     H[t,f,pol,ew,ns]. For each (t,f,pol,ew), we have an array H[256].
//     We zero-pad to length 512 and take an FFT:
//
//       J[j] = sum_k exp(2pi * i * j * k / 512) H[k]
//
//  5. Clamping: here is where the 'map' argument comes in. We reduce J
//     from length-512 to length-256 by selecting indices as follows:
//
//       K[i] = J[map[i]]   (where 0 <= i < 256, and 0 <= map[i] < 512)
//
//  6. Restoring implicit axes, we now have a float32+32 array K[T,F,2,4,256].
//     All math so far has been 32-bit. We convert to float16+16 and write to
//     global GPU memory.
//
// Some of the conventions above are a little unusual (uint4+4 encoding in step 1,
// complex conjugates in step 2,3) but are preserved for consistency with chime.
//
// Each threadblock processes (128 times, 1 freq, 1 pol).
// Thus, gridDim = { T/128, F, 2 }. We assume that T is a multiple of 128.
// The values of (T,F) are supplied to the kernel via gridDims (not kernel args).
//
// The kernel is launched with 32 warps, and blockDim = { 64, 16 }.


__global__ void __launch_bounds___(1024,1)
chime_frb_beamform(
    const uint8_t *__restrict__ inputData,  // shape (T,F,2,4,256)
    const uint *__restrict__ map,           // shape (F,256)
    const float *__restrict__ co,           // shape (F,4,4,2), indexed by (ewout,ewin,ReIm)
    __half2 *__restrict__ outputData,       // shape (T,F,2,4,256)
    const float2 *__restrict__ gains)       // shape (F,2,4,256)
{
    // Shared memory layout (96 KB total):
    //
    //   uint smem_E[32][256];            // 32 KB, axes (time,ns)
    //   union {
    //       float2 smem_H[4][4][256];   // 32 KB, axes (time,ew,ns)
    //       float2 smem_J[4][4][512];   // 64 KB, axes (time,ew,ns)
    //       char smem_fft[64*1024];     // 64 KB, cufftdx scratch space
    //   };
    //
    // Kernel outline:
    //
    //   for each "outer" block of 32 times
    //      load uint4+4 E-array from global memory (T=32)
    //      store E-array in smem_E
    //      for each "inner" block of 4 times
    //          load E-array from smem_E
    //          compute H-array: convert to float32+32, apply gains, ew beamforming
    //          store in smem_H
    //          load from smem_H in a different ordering
    //          compute J-array: do NS FFT
    //          write to smem_J
    //          compute K-array: read from smem_J with 'map' reindexing
    //          write K-array to global memory
    //
    // Notation: in register mappings throughout the code, we use index bits (e1 e0)_2
    // for the length-4 EW axis, and index bits (n7 n6 n5 n4 n3 n2 n1 n0)_2 for the
    // length-256 NS axis.

    // Must allocate dynamically, since > 48KB.
    extern __shared__ char smem_all[];

    // First, apply per-block pointer offsets to all pointer arguments.
    xxx;     // apply per-block pointer offsets to all pointer arguments.

    // Initialize "mapval" (1 register/thread).
    // This is the value of map[ns], for a specific value of ns that will be
    // useful later in the kernel.
    uint map_ns = (threadIdx.x | (threadIdx.y << 6)) & 255;
    uint mapval = map[map_ns];

    // Initialize the "A-coefficients" (8 registers/thread).
    //
    // This step needs some explanation! Later in the kernel (see below), we'll
    // combine the gains and the EW beamforming into a single operation:
    //
    //   H[eo,ns] = sum_ei A[eo,ns,ei] * E[ei,ns]^*
    //
    // where:
    //
    //    0 <= eo < 4 is an output ew beam location
    //    0 <= ei < 4 is an input ew feed location
    //    0 <= ns < 256 is a ns feed location
    //
    // and we define A[eo,ns,ei] = (1/4) co[eo,ei] gain[ei,ns]
    //
    // In this step, the 1024 threads in the kernel will be mapped to (eo,ns)
    // beam locations as follows:
    //
    //   thread:  n4 n3 n2 n1 n0
    //   warp:    eo1 eo0 n7 n6 n5
    //
    // In preparation, we precompute the A-coefficients A[eo,ns,ei] for 0 <= ei < 4
    // and the appropriate per-thread value of (eo,ns).

    float A0re, A0im, A1re, A1im, A2re, A2im, A3re, A3im;

    // Write code here to compute the A-coefficients. For now, don't bother
    // with code optimization -- just write something that's easy to understand.
    // This part can be optimized later, after passing unit tests.
    xxx;
        
    for (int touter = 0; touter < 128; touter += 32) {
        
        // Load E-array from inputData (32 times, 4 ew feeds, 32 ns feeds).
        // Register mapping (dtype uint4+4):
        //  simd:      n1 n0
        //  register:  e1 e0 n2
        //  thread:    t0 n6 n5 n4 n3
        //  warp:      t4 t3 t2 t1 n7

        uint E0, E1, E2, E3, E4, E5, E6, E7;

        xxx;  // load E0..E7 from 'inputData'. Use 64-bit load instructions.

        // Now we do a lot of shuffling operations, to change the register assignment.

        xxx;  // local tranpose (simd s0) <-> (register r2)
        xxx;  // local tranpose (simd s1) <-> (register r1)

        // At this point, the register assignment is (dtype uint4+4):
        //  simd:      e1 e0
        //  register:  n0 n1 n2
        //  thread:    t0 n6 n5 n4 n3
        //  warp:      t4 t3 t2 t1 n7

        xxx;  // warp transpose (register r0) <-> (thread t4)
        xxx;  // warp transpose (register r1) <-> (thread t3)

        // At this point, the register assignment is (dtype uint4+4):
        //  simd:      e1 e0
        //  register:  n0 n6 t0
        //  thread:    n2 n1 n5 n4 n3
        //  warp:      t4 t3 t2 t1 n7

        // We now call __shfl_sync() on each register, to permute threads
        // (n2 n1 n5 n4 n3) -> (n5 n4 n3 n2 n1).
        
        xxx;  // compute lane permutation, call __shfl_sync()

        // At this point, the register assignment is (dtype uint4+4)
        //  simd:      e1 e0
        //  register:  n0 n6 t0
        //  thread:    n5 n4 n3 n2 n1
        //  warp:      t4 t3 t2 t1 n7

        // Now we write to shared memory:
        //   uint smem_E[32][256];   // (time,ns)
        //
        // where there is no length-4 ew axis since we have packed four uint4+4s
        // into a uint, with simd (s1 s0) <-> (e1 e0). We can use a 64-bit,
        // bank-conflict-free store instruction hree.

        xxx;   // store to smem_E

        for (int tmid = 0; tmid < 32; tmid += 4) {

            for (int tinner = 0; tinner < 4; tinner++) {
                // In this step, we load uint4+4 data from shared memory, apply gains,
                // and do EW beamforming. As explained above, these steps are coalesced
                // into the "A-matrix" multiplication.
                //
                //   E1[eo,ns] = sum_ei A[eo,ns,ei] * E0[ei,ns]^*
                //
                // We load data from smem_E:
                //   uint smem_E[32][256];   // (time,ns)
                //
                // into register assignment
                //   simd:    ei1 ei0
                //   thread:  n4 n3 n2 n1 n0
                //   warp:    eo1 eo0 n7 n6 n5

                xxx;  // load from smem_E (32-bit load, bank-conflict-free)

                // Convert uint4+4 -> float32+32, multiply by A-matrix.

                xxx;  // convert uint4+4 -> float32+32, multiply by A-matrix

                // Now we have a float2 on each thread, corresponding to the H-array in
                // register assigment:
                //   register:  ReIm
                //   thread:    n4 n3 n2 n1 n0
                //   warp:      eo1 eo0 n7 n6 n5
                //
                // Write to smem_H, using 64-bit, bank-conflict-free store.
                //   float2 smem_H[4][4][256];   // (tinner,ew,ns)

                xxx;   // store to smem_H
            }

            // Load partially beamformed data from shared memory, using 64-bit,
            // bank-conflict-free load:
            //   float2 smem_H[4][4][256];   // (tinner,ew,ns)
            //
            // in the following register assignment:
            //   register:  n7 n6 ReIm
            //   thread:    n4 n3 n2 n1 n0
            //   warp:      t1 t0 e1 e0 n5

            xxx;  // load from smem_H

            // Now do the NS FFT using cufftdx.
            // The FFT is zero-padded, and takes (length 256) -> (length 512).
            // This adds an 'n8' bit to the register assignment.

            xxx;  // cufftdx

            // After cufftdx, we get the J-array in register assignment:
            //   register:  n8 n7 n6 ReIm
            //   thread:    n4 n3 n2 n1 n0
            //   warp:      t1 t0 e1 e0 n5

            // Write to smem_J, using 64-bit, bank-conflict-free stores:
            //   float2 smem_J[4][4][512];  // (time,ew,ns)

            xxx;   // write smem_J

            for (int tinner = 0; tinner < 4; tinner++) {
                
                // Load K-array (one float2 on each thread), in register assignment:
                //   register:  ReIm
                //   thread:    n4 n3 n2 n1 n0
                //   warp:      e1 e0 n7 n6 n5
                //
                // This can be done by loading smem_J[] at a memory location that
                // depends on 'mapval', using a 64-bit load instruction. This load
                // is generally not bank conflict free (!!). This can be fixed, but
                // it involves some nontrivial extra steps that I'll introduce later.
                // (Let's pass unit tests first!)

                xxx;  // load K-array

                // Write K-array element to global memory
                
                xxx;  // store K-array
            }
        }
    }
}

#endif


} // namespace pirate
