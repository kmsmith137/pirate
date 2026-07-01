#ifndef _PIRATE_SIMPULSE_HPP
#define _PIRATE_SIMPULSE_HPP

// pirate::simpulse: simulating FRB pulses in the (intensity, time) plane.
//
// Ported (vendorized) from the standalone 'simpulse' library, keeping only the FRB
// 'single_pulse' functionality (no pulsar phase_model / von_mises_profile). Modernized
// to the pirate conventions: C++17, pybind11, MKL (instead of FFTW), 'long' indices,
// ksgpu::Array interfaces, and xassert error-checking.

#include <string>
#include <iostream>

#include <ksgpu/Array.hpp>

namespace pirate {
namespace simpulse {
#if 0
}}  // editor auto-indent
#endif


// Returns dispersion delay in seconds. 'dm' is the dispersion measure in standard
// units (pc cm^{-3}). Uses pirate::constants::k_dm (see constants.hpp); this differs
// from the original simpulse constant (4.148806e3 s) at the ~1e-6 level.
extern double dispersion_delay(double dm, double freq_MHz);

// Returns scattering time in seconds. We define the "scattering measure" SM to be the
// scattering time in MILLISECONDS (not seconds) at 1 GHz.
extern double scattering_time(double sm, double freq_MHz);


// -------------------------------------------------------------------------------------------------
//
// struct SinglePulse: one dispersed, scattered pulse, on a fixed frequency channelization and a
// fixed (zero-based) time sampling. The constructor precomputes the pulse as a sparse array of
// per-channel time samples; add_to_timestream() then scatters it into a dense (nfreq, out_nt) array.


struct SinglePulse {
    // Constructor arguments, bundled into a Params struct (see the constructor below).
    struct Params
    {
        long   pulse_nt = 1024;                 // "under the hood" samples (power of two; 1024 is a good default)
        double time_sample_ms = 0.0;            // time-sample duration in ms; REQUIRED (dt = 1e-3*time_sample_ms sec, must be > 0)

        // Sorted (strictly increasing) array of length (nfreq+1). The i-th frequency channel spans
        // frequency range [freq_edges_MHz[i], freq_edges_MHz[i+1]]. Channel widths need NOT be equal.
        // So nfreq = freq_edges_MHz.size - 1, freq_lo_MHz = freq_edges_MHz[0], and
        // freq_hi_MHz = freq_edges_MHz[nfreq]. The constructor deep-copies this array.
        ksgpu::Array<double> freq_edges_MHz;

        double dm = 0.0;                        // dispersion measure in standard units (pc cm^{-3})
        double sm = 0.0;                        // scattering measure: scattering time in milliseconds (not seconds) at 1 GHz
        double intrinsic_width = 0.0;           // frequency-independent Gaussian width in seconds
        double fluence = 1.0;                   // integrated flux (units: whatever add_to_timestream() outputs, times seconds)
        double spectral_index = 0.0;            // exponent alpha in F(nu) = F(nu_0) (nu/nu_0)^alpha
        double undispersed_arrival_time = 0.0;  // arrival time at nu=infty, in seconds relative to an arbitrary origin
    };

    // Construction parameters, immutable after construction. Public so callers can read them; also
    // exposed to python as read-only attributes (SinglePulse.pulse_nt, .time_sample_ms, .dm, ...,
    // .freq_edges_MHz, plus the derived .nfreq / .freq_lo_MHz / .freq_hi_MHz).
    const Params params;

    // Precomputed SPARSE representation of the pulse, on the zero-based time grid where sample 'it'
    // (0 <= it) spans [it*dt, (it+1)*dt] seconds, dt = 1e-3*time_sample_ms. Computed in the ctor.
    // Frequencies are ordered LOW to HIGH. (Also exposed to python, read-only.)
    ksgpu::Array<long>  sparse_i0;      // (nfreq,) first dense-grid sample index of channel i's pulse
    ksgpu::Array<long>  sparse_n;       // (nfreq,) number of samples of channel i's pulse
    ksgpu::Array<long>  sparse_offset;  // (nfreq,) offset into sparse_data (= sum_{j<i} sparse_n[j])
    ksgpu::Array<float> sparse_data;    // (sum(sparse_n),) packed samples; fluence + spectral weight baked in
    long nt_min;                        // = max_i (sparse_i0[i] + sparse_n[i]); smallest out_nt with no clipping

    SinglePulse(const Params &params);

    // Adds the pulse to a "block" of (frequency, time) samples 'out', a 2-d float32 array of shape
    // (nfreq, out_nt), scaled by 'weight'. The grid is zero-based (sample 'it' spans
    // [it*dt, (it+1)*dt] seconds); samples at index >= out_nt are clipped (use out_nt >= nt_min for
    // no clipping). Time samples must be contiguous (out.strides[1] == 1).
    //
    // Frequencies are ordered LOW to HIGH (pulse channel i -> row i). NOTE: this is the opposite of
    // bonsai/rf_pipelines (and pirate intensity arrays), which is high to low; a caller targeting
    // those must reverse the frequency axis itself.
    void add_to_timestream(ksgpu::Array<float> &out, double weight = 1.0) const;

    // Returns total signal-to-noise over all frequency channels and time samples combined, computed
    // from the precomputed samples on the construction-time grid. 'sample_rms' is the per-sample RMS
    // noise (scalar overload) or a length-nfreq array (vector overload). For the vector overload, if
    // 'channel_weights' is empty (the default), 1/sample_rms^2 weighting is assumed.
    double get_signal_to_noise(double sample_rms = 1.0) const;
    double get_signal_to_noise(const ksgpu::Array<double> &sample_rms,
                               const ksgpu::Array<double> &channel_weights = ksgpu::Array<double>()) const;

    // String representation.
    void print(std::ostream &os) const;
    std::string str() const;

    // We make SinglePulse noncopyable, even though the default copy constructor is a sensible
    // "deep" copy, to catch performance bugs (a deep copy is probably unintentional).
    SinglePulse(const SinglePulse &) = delete;
    SinglePulse &operator=(const SinglePulse &) = delete;

private:
    // Validates freq_edges_MHz (1-d, contiguous, length >= 2, positive, strictly increasing) and
    // returns a copy of 'params' whose freq_edges_MHz is an owned deep copy -- so the stored member
    // is self-contained (independent of the caller's array). Called from the ctor's init list.
    static Params _validate(const Params &params);
};


inline std::ostream &operator<<(std::ostream &os, const SinglePulse &sp) { sp.print(os); return os; }


}}  // namespace pirate::simpulse

#endif // _PIRATE_SIMPULSE_HPP
