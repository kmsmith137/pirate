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
        double dm = -1.0;                              // dispersion measure in standard units (pc cm^{-3})
        double sm = -1.0;                              // scattering measure: scattering time in ms (not se) at 1 GHz
        double intrinsic_width = -1.0;                 // frequency-independent Gaussian width in seconds
        double spectral_index = -1000.0;               // exponent alpha in F(nu) = F(nu_0) (nu/nu_0)^alpha
        double undispersed_arrival_time_sec = -1.0e6;  // arrival time at nu=infty, in seconds
        double time_sample_ms = -10.0;                 // time-sample duration in ms
        double snr = -1.0;                             // signal-to-noise, assuming perfect matched filter.
        
        // Sorted (strictly increasing) array of length (nfreq+1). The i-th frequency channel spans
        // frequency range [freq_edges_MHz[i], freq_edges_MHz[i+1]]. Channel widths need NOT be equal.
        ksgpu::Array<double> freq_edges_MHz;

        // Length (nfreq), must be positive.
        ksgpu::Array<double> freq_variances;

        // Subband. If specified, then the simulated pulse is restricted to frequency channels
        // that overlap the subbands.
        double subband_freq_lo_MHz = 0.0;
        double subband_freq_hi_MHz = 1.0e9;

        // If false, then an exception is raised if any freq_it0 is < 0.
        // If true, then freq_it0 may be negative (the pulse may have samples at t < 0, which
        // are all kept). Consumers are responsible for clipping to their own time range
        // (e.g. add_to_timestream() clips to [0, out_nt)).
        bool allow_negative_arrival_times = false;

        // "under the hood" samples (power of two; 1024 is a good default)
        long internal_nt = 1024;
    };

    // Construction parameters, immutable after construction. Public so callers can read them; also
    // exposed to python as read-only attributes (SinglePulse.internal_nt, .time_sample_ms, .dm, ...,
    // .freq_edges_MHz, plus the derived .nfreq / .freq_lo_MHz / .freq_hi_MHz).
    const Params params;

    // Precomputed SPARSE representation of the pulse, on the integer time grid where sample 'it'
    // spans [it*dt, (it+1)*dt] seconds, dt = 1e-3*time_sample_ms. Computed in the ctor. Sample
    // indices are >= 0, unless allow_negative_arrival_times=true (then freq_it0 may be negative).
    // Frequencies are ordered LOW to HIGH. (Also exposed to python, read-only.)
    ksgpu::Array<long>  freq_it0;     // (nfreq,) first dense-grid sample index of channel i's pulse
    ksgpu::Array<long>  freq_nt;      // (nfreq,) number of samples of channel i's pulse
    ksgpu::Array<long>  freq_sd_off;  // (nfreq,) offset into sparse_data (= sum_{j<i} freq_nt[j])
    ksgpu::Array<float> sparse_data;  // (sum(freq_nt),) packed samples; spectral weight + snr norm baked in
    long                nt_min;       // = max_i (freq_it0[i] + freq_nt[i]); smallest out_nt with no
                                      //   high-end clipping (samples at negative indices are clipped
                                      //   by add_to_timestream() regardless of out_nt)

    SinglePulse(const Params &params);

    // Adds the pulse to a "block" of (frequency, time) samples 'out', a 2-d, host (CPU) float32 array
    // of shape (nfreq, out_nt), scaled by 'weight'. Column 'it' of 'out' spans [it*dt, (it+1)*dt]
    // seconds; samples outside [0, out_nt) are clipped (use out_nt >= nt_min so that nothing is
    // clipped at the high end; samples at negative indices, possible only with
    // allow_negative_arrival_times=true, are always clipped). Time samples must be contiguous
    // (out.strides[1] == 1).
    void add_to_timestream(ksgpu::Array<float> &out, double weight = 1.0) const;

    // String representation.
    void print(std::ostream &os) const;
    std::string str() const;

    // We make SinglePulse noncopyable, even though the default copy constructor is a sensible
    // "deep" copy, to catch performance bugs (a deep copy is probably unintentional).
    SinglePulse(const SinglePulse &) = delete;
    SinglePulse &operator=(const SinglePulse &) = delete;

private:
    // Validates freq_edges_MHz (1-d, contiguous, length >= 2, positive, strictly increasing) and
    // freq_variances (length nfreq, positive), and returns 'params' unchanged. The array members are
    // stored BY REFERENCE (not deep-copied). Called from the ctor's init list.
    static Params _validate(const Params &params);
};


inline std::ostream &operator<<(std::ostream &os, const SinglePulse &sp) { sp.print(os); return os; }


}}  // namespace pirate::simpulse

#endif // _PIRATE_SIMPULSE_HPP
