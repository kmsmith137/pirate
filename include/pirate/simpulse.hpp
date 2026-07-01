#ifndef _PIRATE_SIMPULSE_HPP
#define _PIRATE_SIMPULSE_HPP

// pirate::simpulse: simulating FRB pulses in the (intensity, time) plane.
//
// Ported (vendorized) from the standalone 'simpulse' library, keeping only the FRB
// 'single_pulse' functionality (no pulsar phase_model / von_mises_profile). Modernized
// to the pirate conventions: C++17, pybind11, MKL (instead of FFTW), 'long' indices,
// ksgpu::Array interfaces, and xassert error-checking.

#include <vector>
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
// struct SinglePulse: one dispersed, scattered pulse, in a fixed frequency channelization.


struct SinglePulse {
    // Constructor arguments, bundled into a Params struct (see the constructor below).
    struct Params
    {
        long   pulse_nt = 1024;                 // "under the hood" samples (power of two; 1024 is a good default)
        long   nfreq = 0;                       // number of frequency channels (assumed equally spaced)
        double freq_lo_MHz = 0.0;               // lower limit of frequency band
        double freq_hi_MHz = 0.0;               // upper limit of frequency band
        double dm = 0.0;                        // dispersion measure in standard units (pc cm^{-3})
        double sm = 0.0;                        // scattering measure: scattering time in milliseconds (not seconds) at 1 GHz
        double intrinsic_width = 0.0;           // frequency-independent Gaussian width in seconds
        double fluence = 1.0;                   // integrated flux (units: whatever add_to_timestream() outputs, times seconds)
        double spectral_index = 0.0;            // exponent alpha in F(nu) = F(nu_0) (nu/nu_0)^alpha
        double undispersed_arrival_time = 0.0;  // arrival time at nu=infty, in seconds relative to an arbitrary origin
    };

    // Construction parameters, immutable after construction. Public so callers can read them;
    // also exposed to python as read-only attributes (SinglePulse.pulse_nt, .nfreq, .dm, ...).
    const Params params;

    // Under-the-hood representation of the pulse (not normally needed from outside).
    // All "internal" times are relative to undispersed_arrival_time.
    std::vector<double> pulse_t0;       // (nfreq,) start time of pulse in i-th channel
    std::vector<double> pulse_t1;       // (nfreq,) end time of pulse in i-th channel
    std::vector<double> pulse_freq_wt;  // (nfreq,) per-channel weight from spectral index
    std::vector<double> pulse_cumsum;   // (nfreq, pulse_nt+1) cumulative sum of pulse, normalized to sum=1
    double min_t0;                      // minimum of all pulse_t0 values
    double max_t1;                      // maximum of all pulse_t1 values
    double max_dt;                      // maximum of all (pulse_t1 - pulse_t0) values

    SinglePulse(const Params &params);

    // Returns the earliest and latest arrival times in the band [freq_lo_MHz, freq_hi_MHz].
    // (Both are larger than undispersed_arrival_time, unless the intrinsic width is very large.)
    void get_endpoints(double &t0, double &t1) const;

    // Adds the pulse to a "block" of (frequency, time) samples (sometimes called incrementally,
    // as a stream of blocks is generated). 'out' is a 2-d array with shape (nfreq, out_nt), and
    // 'out_t0'/'out_t1' are the endpoints of the sampled region (seconds, same origin as
    // undispersed_arrival_time). Time samples must be contiguous (out.strides[1] == 1).
    //
    // Frequencies are ordered LOW to HIGH (pulse channel i -> row i). NOTE: this is the opposite
    // of the ordering used in bonsai/rf_pipelines (and pirate intensity arrays), which is high to
    // low. The original simpulse offered a 'freq_hi_to_lo' option, implemented with a NEGATIVE row
    // stride; ksgpu::Array forbids negative strides (Array::check_invariants() asserts strides>=0),
    // so we drop that option. A caller wanting high-to-low ordering must reverse the frequency axis
    // itself (e.g. add into a temporary, then copy reversed).
    void add_to_timestream(ksgpu::Array<float> &out, double out_t0, double out_t1,
                           double weight = 1.0) const;

    // Returns the total number of samples needed to represent this pulse in sparse form (summed
    // over all frequency channels). Use this to size the 'out' array for add_to_timestream_sparse().
    long get_n_sparse(double out_t0, double out_t1, long out_nt) const;

    // Sparse version of add_to_timestream():
    //   - 'out': 1-d, length >= get_n_sparse(). Per-channel samples are packed in order (low->high freq).
    //   - 'out_i0': length nfreq. Channel i's samples start at dense time index out_i0[i].
    //   - 'out_n': length nfreq. Channel i has out_n[i] samples; its packed data starts at sum(out_n[:i]).
    void add_to_timestream_sparse(ksgpu::Array<float> &out, ksgpu::Array<long> &out_i0,
                                  ksgpu::Array<long> &out_n, double out_t0, double out_t1,
                                  long out_nt, double weight = 1.0) const;

    // Returns total signal-to-noise over all frequency channels and time samples combined.
    // Depends on 'sample_dt' (sample length in seconds) and weakly on 'sample_t0' (start time of
    // an arbitrarily chosen sample).
    double get_signal_to_noise(double sample_dt, double sample_rms = 1.0,
                               double sample_t0 = 0.0) const;

    // Vector version: 'sample_rms' and 'channel_weights' are length-nfreq arrays. If
    // 'channel_weights' is empty (the default), then 1/sample_rms^2 weighting is assumed.
    double get_signal_to_noise(double sample_dt, const ksgpu::Array<double> &sample_rms,
                               const ksgpu::Array<double> &channel_weights = ksgpu::Array<double>(),
                               double sample_t0 = 0.0) const;

    // String representation.
    void print(std::ostream &os) const;
    std::string str() const;

    // Internal helper (recomputes pulse_freq_wt from spectral_index).
    void _compute_freq_wt();

    // We make SinglePulse noncopyable, even though the default copy constructor is a sensible
    // "deep" copy, to catch performance bugs (a deep copy is probably unintentional).
    SinglePulse(const SinglePulse &) = delete;
    SinglePulse &operator=(const SinglePulse &) = delete;
};


inline std::ostream &operator<<(std::ostream &os, const SinglePulse &sp) { sp.print(os); return os; }


}}  // namespace pirate::simpulse

#endif // _PIRATE_SIMPULSE_HPP
