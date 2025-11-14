//x This script is processed with 'vendorize.py' (in the toplevel pirate dir)
//x to produce a "vendorized" source file for the casm pipeline. The markup
//x in comments (e.g. "//x") is parsed by vendorize.py.
//x
#ifndef _CASM_BEAMFORMER_HPP
#define _CASM_BEAMFORMER_HPP

#include <vector>
#include <memory>
#include <cstdint>
#include <ksgpu/Array.hpp>  //x

namespace pirate {  //x
#if 0               //x
}                   //x editor auto-indent
#endif              //x

// CasmBeamformer: this is constructed once on the host, with some constructor
// arguments that are assumed to be independent of time (e.g. beam locations on
// the sky). The constructor does some one-time initializations, and allocates
// some constant data on the GPU (a few hundred KB).
//
// After construction, the launch_beamformer() member function can be called
// on regular chunks of data (say 0.1-1 sec) to generate beamformed intensities
// from the electric field samples, with per-feed complex weights which can be
// used to correct for delays (gains), and apply additional weighting (e.g.
// downweighting noisy feeds, masking bad feeds).
//
// NOTE: the C++ CasmBeamformer interface is identical to the python reference
// implementation (CasmReferenceBeamformer.py). There is a unit test that tests
// machine-precision equality, for random input data.) Therefore, for understanding
// details such as sign conventions, you may prefer to refer to the python
// reference implementation.
//
// Critical assumptions that would be very painful to change
// ---------------------------------------------------------
//
//  - Long axis is uniformly spaced, and has length approximately 43.
//
//  - Short axis has length 6, with approximate spacings [40cm, 50cm, 40cm, 50cm, 40cm].
//    The details of the spacings are not so important, but the spacings being invariant
//    under reversing the order is important.
//
// Assumptions that would be moderately painful to change
// ------------------------------------------------------
//
//  - Beams are non-tracking.
//
//  - Both polarizations use the same index ordering.
//   
//  - 256 dual-pol antennas (or fewer) in total.
//
//  - Electric field array is int4+4, and laid out in global GPU memory with
//    axes ordered (time,freq,pol,dish) from slowest to fastest, and the inner
//    two axes (pol,dish) having shape (2,256).
// 
// Assumptions that would be non-painful to change
// -----------------------------------------------
//
//  - Phase conventions are such that the beamforming phase (before squaring
//    the electric field to get intensity) is:
//
//       exp(+ 2*pi*i*freq*c * (dish location) . (beam direction))
//
//    where "." denotes the 3-d vector dot product, and the sign inside the
//    exp(...) is "+" not "-".
//
// Notation
// --------
// 
//   T = number of time samples, must be a multiple of 'downsampling_factor'.
//   F = number of frequency channels
//   B = number of output beams
//
// Beam locations are represented by two "zenith angles" (theta_N, theta_E),
// defined formally as follows.  In a coordinate system where
//
//   (0,0,1) = unit vector pointing toward zenith
//   (1,0,0) = unit vector pointing north
//   (0,1,0) = unit vector pointing west
//
// each beam location can be represented by a unit vector (nx,ny,nz). Then
// the zenith angles (theta_N, theta_E) are defined by
//
//   (nx, ny) = (sin(theta_N), sin(theta_E))


struct CasmBeamformer
{
    // speed of light in weird units meters-MHz
    static constexpr float speed_of_light = 299.79;
    static constexpr float default_ns_feed_spacing = 0.50;  // meters
    
    inline static const float default_ew_feed_spacings[5]
        = { 0.38f, 0.445f, 0.38f, 0.445f, 0.38f };  // meters
                 
    // Constructor arguments
    // ---------------------
    //
    // - frequencies: shape (F,) array containing frequencies in MHz.
    //
    // - feed_indices: integer-valued shape (256,2) array which encodes
    //   the mapping between a 1-d antenna index 0 <= i < 256, and a 2-d
    //   array location (j,k), where 0 <= j < 43 and 0 <= k < 6. This
    //   mapping i -> (j,k) is assumed to be the same for both polarizations,
    //   and is given by:
    //
    //     j = feed_indices[i,0]
    //     k = feed_indices[i,1]
    //
    // - beam_locations: shape (B,2) array containing sky locations of beams,
    //   represented as (sin(theta_N), sin(theta_E)) where the zenith angles
    //   (theta_N, theta_E) are defined above. Note the sines!
    //
    // - downsampling_factor: integer level of downsampling between electric
    //   field array (after complex-valued PFB channelization) and FRB
    //   beamformed timestreams.
    //
    //   Note that for CASM, the time sampling rate of the channelized electric
    //   field array is:
    //
    //     dt_in = 4096 / (125 MHz) = 32.768 microseconds
    //
    //   and so the time sampling rate of the FRB beamformed timestreams will be
    //
    //     dt_out = (dt_in / downsampling_factor)
    //            = (1.049 ms) * (32 / downsampling_factor).
    //
    // - ns_feed_spacing: spacing (in meters) of feeds along the north-south axis
    //
    // - ew_feed_spacings: length-5 array containing spacings (in meters) along
    //   the east-west axis. Must be flip-symmetric.

    //xbegin "ksgpu::Array" constructor.
    CasmBeamformer(
        const ksgpu::Array<float> &frequencies,     // shape (nfreq,)
        const ksgpu::Array<int> &feed_indices,      // shape (256,2)
        const ksgpu::Array<float> &beam_locations,  // shape (nbeams,2)
        int downsampling_factor,
        float ns_feed_spacing = default_ns_feed_spacing,
        const ksgpu::Array<float> &ew_feed_spacings = ksgpu::Array<float>()
    );
    //xend
    
    // "Bare-pointer" constructor.
    CasmBeamformer(
        const float *frequencies,        // shape (nfreq,)
        const int *feed_indices,         // shape (256,2)
        const float *beam_locations,     // shape (nbeams,2)
        int downsampling_factor,
        int nfreq,
        int nbeams,
        float ns_feed_spacing = default_ns_feed_spacing,
        const float *ew_feed_spacings = default_ew_feed_spacings
    );
    
    // The launch_beamformer() member function launches the beamforming kernel
    // asynchronously on a caller-specified stream. Caller is responsible for
    // synchronization.
    //
    // Arguments
    // ---------
    // 
    //  - e_arr: shape (T,F,2,256) complex-valued array, where:
    //
    //      T = number of time samples, must be a multiple of 'downsampling_factor'.
    //      F = number of frequency channels
    //      2 = number of polarizations
    //      256 = number of antennas ("dishes" in the code)
    //
    //  - feed_weights: shape (F,2,256) complex-valued array containing
    //    per-feed beamforming weights. The first step in the beamformer
    //    is multiplying 'e_arr' by the weights. The optimal beamforming
    //    weight is roughly (g^*/sigma^2), where g is the complex gain
    //    and sigma is the variance of the timestream (without undoing
    //    the gain).
    //
    // Outputs an array of shape (Tout,F,B) containing beamformed intensities.

    //xbegin "ksgpu::Array" version.
    void launch_beamformer(
        const ksgpu::Array<uint8_t> &e_arr,        // shape (Tin,F,2,256), axes (time,freq,pol,dish)
        const ksgpu::Array<float> &feed_weights,   // shape (F,2,256,2), axes (freq,pol,dish,reim)
        ksgpu::Array<float> &i_out,                // shape (Tout,F,B)
        cudaStream_t stream = nullptr              // nullptr = "default cuda stream"
    ) const;
    //xend

    // "Bare-pointer" version.
    void launch_beamformer(
        const uint8_t *e_arr,                      // shape (Tin,F,2,256), axes (time,freq,pol,dish)
        const float *feed_weights,                 // shape (F,2,256,2), axes (freq,pol,dish,reim)
        float *i_out,                              // shape (Tout,F,B)
        int Tin,                                   // number of input times Tin = Tout * downsampling_factor
        cudaStream_t stream = nullptr              // nullptr = "default cuda stream"
    ) const;    

    // There is a maximum beam count that the beamformer can support
    // (currently 4672), due to GPU shared memory limitations.
    static int get_max_beams();
    
    static void show_shared_memory_layout();
    static void test_microkernels();
    static void run_timings();

    // ---------------------------------------------------------------------------------------------
    
    int F = 0;  // number of frequency channels (on one GPU)
    int B = 0;  // number of output beams
    int constructor_device = -1;  // cuda device when constructor was called
    int downsampling_factor = 0;

    std::vector<float> frequencies;     // shape (F,)
    std::vector<int> feed_indices;      // shape (256,2)
    std::vector<float> beam_locations;  // shape (B,2)
    float ns_feed_spacing = 0.0;

    float ew_feed_spacings[5];    // meters
    float ew_feed_positions[6];   // meters
    float ew_beam_locations[24];  // sin(ZA)
    
    // This is a ~200KB region of global GPU memory, which is initialized by the
    // CasmBeamformer constructor, and passed to the beamformer on every kernel launch.
    // See "global memory layout" in CasmBeamformer.cu for more info.
    // (Represented as a std::shared_ptr whose deleter calls cudaFree().)
    std::shared_ptr<float> gpu_persistent_data;
    
    // For unit tests.
    int nominal_Tin_for_unit_tests = 0;
    static CasmBeamformer make_random(bool randomize_feed_indices=true);
    static std::shared_ptr<int> make_random_feed_indices();   // helper for make_random()
    static std::shared_ptr<int> make_regular_feed_indices();  // helper for make_random()

    // Helper function called by constructor.
    void _construct(
        const float *frequencies,        // shape (F,)
        const int *feed_indices,         // shape (256,2)
        const float *beam_locations,     // shape (B,2)
        int downsampling_factor,
        int nfreq,
        int nbeams,
        float ns_feed_spacing,
        const float *ew_feed_spacings    // either shape (5,) or NULL
    );
};

// For python ctypes
//i extern "C" {
//i
//i void casm_bf_test_microkernels();
//i
//i void casm_bf_run_timings();
//i
//i int casm_bf_get_max_beams();
//i
//i void casm_bf_one_shot_for_testing(
//i     const float *frequencies,        // shape (nfreq,)
//i     const int *feed_indices,         // shape (256,2)
//i     const float *beam_locations,     // shape (nbeams,2)
//i     long downsampling_factor,
//i     long nfreq,
//i     long nbeams,
//i     float ns_feed_spacing,
//i     const float *ew_feed_spacings,
//i     const uint8_t *e_arr,            // shape (Tin,F,2,256), on gpu
//i     const float *feed_weights,       // shape (F,2,256,2), on gpu
//i     float *i_out,                    // shape (Tout,F,B), on gpu
//i     long Tin
//i );
//i
//i }  // extern "C"

}  //x namespace pirate

#endif  // _CASM_BEAMFORMER_HPP
