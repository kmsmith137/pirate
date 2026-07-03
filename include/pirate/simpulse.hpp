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

        // "under the hood" samples (power of two; 1024 is a good default)
        long internal_nt = 1024;
    };

    // Construction parameters. Public so callers can read them; also exposed to python as read-only
    // attributes (SinglePulse.internal_nt, .time_sample_ms, .dm, ..., .freq_edges_MHz, plus the
    // derived .nfreq / .freq_lo_MHz / .freq_hi_MHz). Treat as immutable after construction, EXCEPT
    // that shift_samples() updates undispersed_arrival_time_sec.
    Params params;

    // Precomputed SPARSE representation of the pulse, on the integer time grid where sample 'it'
    // spans [it*dt, (it+1)*dt] seconds, dt = 1e-3*time_sample_ms. Computed in the ctor. Sample
    // indices may be negative (a pulse whose arrival extends to t < 0). Frequencies are ordered
    // LOW to HIGH. (Also exposed to python, read-only.)
    //
    // Every channel -- including subband-skipped "inactive" channels, which have freq_nt == 0 --
    // satisfies it_start <= freq_it0 <= (freq_it0 + freq_nt) <= it_end. (Inactive channels are
    // assigned freq_it0 = it_start so this invariant holds without special-casing them.)
    ksgpu::Array<long>  freq_it0;     // (nfreq,) first grid sample index of channel i's pulse
    ksgpu::Array<long>  freq_nt;      // (nfreq,) number of samples of channel i's pulse (0 if inactive)
    ksgpu::Array<long>  freq_sd_off;  // (nfreq,) offset into sparse_data (= sum_{j<i} freq_nt[j])
    ksgpu::Array<float> sparse_data;  // (sum(freq_nt),) packed samples; spectral weight + snr norm baked in
    long                it_start;     // = min_i freq_it0[i]                (min over active channels)
    long                it_end;       // = max_i (freq_it0[i] + freq_nt[i]) (max over active channels)

    SinglePulse(const Params &params);

    // Adds the pulse to a "block" of (frequency, time) samples 'out', a 2-d, host (CPU) float32 array
    // of shape (nfreq, out_nt), scaled by 'weight'. Column 'it' of 'out' represents grid sample index
    // (out_it0 + it), i.e. 'out' spans sample indices [out_it0, out_it0 + out_nt) (out_it0 may be
    // negative). 'out' MUST span the pulse's full time range: an exception is raised unless
    // out_it0 <= it_start and out_it0 + out_nt >= it_end. Time samples must be contiguous
    // (out.strides[1] == 1), ordered low to high in frequency.
    void add_to_timestream(ksgpu::Array<float> out, long out_it0, float weight = 1.0f) const;

    // Shift the pulse forward in time by delta_it samples (delta_it may be negative): adds delta_it
    // to every freq_it0 and to it_start / it_end, and adds (1e-3 * delta_it * time_sample_ms) to
    // params.undispersed_arrival_time_sec. The sparse sample values (sparse_data) and per-channel
    // counts/offsets (freq_nt / freq_sd_off) are unchanged.
    void shift_samples(long delta_it);

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
