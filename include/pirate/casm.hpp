#ifndef _PIRATE_CASM_HPP
#define _PIRATE_CASM_HPP

#include <ksgpu/Array.hpp>

namespace pirate {
#if 0
}  // editor auto-indent
#endif


// Called by 'python -m pirate_frb test [--casm]'
extern void test_casm_microkernels();

    
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
        ksgpu::Array<float> &frequencies,     // shape (F,)
	ksgpu::Array<int> &feed_indices,      // shape (256,2)
	ksgpu::Array<float> &beam_locations,  // shape (B,2)
	int downsampling_factor,
	float ns_feed_spacing = 0.50,  // meters
	const float *ew_feed_spacing = default_ew_feed_spacings
    );

    // Beamforming kernel is launched asychronously, caller is responsible for synchronization.
    void launch_beamformer(
        const ksgpu::Array<uint8_t> &e_in,         // shape (T,F,2,256), axes (time,freq,pol,dish)
	const ksgpu::Array<float> &feed_weights,   // shape (F,2,256,2), axes (freq,pol,dish,reim)
	ksgpu::Array<float> &i_out,                // shape (Tout,F,B)
	cudaStream_t stream = nullptr
    ) const;

    // ---------------------------------------------------------------------------------------------
    
    int F = 0;  // number of frequency channels (on one GPU)
    int B = 0;  // number of output beams
    int downsampling_factor = 0;
    
    ksgpu::Array<float> frequencies;     // shape (F,)
    ksgpu::Array<int> feed_indices;      // shape (256,2)
    ksgpu::Array<float> beam_locations;  // shape (B,2)
    float ns_feed_spacing = 0.0;

    float ew_feed_spacing[5];     // meters
    float ew_feed_positions[6];   // meters
    float ew_beam_locations[24];  // sin(ZA)

    static int get_max_beams();
    
    // This is a ~200KB region of global GPU memory, which is initialized by the
    // CasmBeamformer constructor, and passed to the beamformer on every kernel launch.
    // See "global memory layout" in CasmBeamformer.cu for more info.
    ksgpu::Array<float> gpu_persistent_data;
    
    // For unit tests
    int nominal_Tin_for_unit_tests = 0;
    static CasmBeamformer make_random(bool randomize_feed_indices=true);
    static ksgpu::Array<int> make_random_feed_indices();   // helper for make_random()
    static ksgpu::Array<int> make_regular_feed_indices();  // helper for make_random()
    static void show_shared_memory_layout();
};


}  // namespace pirate

#endif  // _PIRATE_CASM_HPP
