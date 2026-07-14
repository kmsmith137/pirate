#include "../include/pirate/simpulse.hpp"
#include "../include/pirate/constants.hpp"   // pirate::constants::k_dm
#include "../include/pirate/inlines.hpp"     // pirate::square, pirate::is_sorted

#include <cmath>
#include <vector>
#include <limits>
#include <complex>
#include <sstream>
#include <algorithm>
#include <stdexcept>

#include <ksgpu/xassert.hpp>
#include <ksgpu/mem_utils.hpp>   // af_uhost

#include <mkl_dfti.h>

using namespace std;

namespace pirate {
namespace simpulse {
#if 0
}}  // editor auto-indent
#endif


// -------------------------------------------------------------------------------------------------
//
// Free functions + file-static helpers.


double dispersion_delay(double dm, double freq_MHz)
{
    xassert(dm >= 0.0);
    xassert(freq_MHz > 0.0);   // used as a divisor
    // constants::k_dm is in (ms . MHz^2) per (pc cm^{-3}); convert the delay to seconds.
    return 1.0e-3 * constants::k_dm * dm / (freq_MHz * freq_MHz);
}


double scattering_time(double sm, double freq_MHz)
{
    xassert(sm >= 0.0);
    xassert(freq_MHz > 0.0);   // used as a divisor (and pow(negative, index) is NaN)
    // 'sm' is the scattering time in milliseconds at 1 GHz.
    return 1.0e-3 * sm / pow(freq_MHz/1000.0, constants::frb_scattering_index);
}


// Returns sin(x)/x. (Named bessj0() in the original simpulse; this is actually the zeroth
// spherical Bessel function j_0, i.e. sinc, not the cylindrical J_0 -- but we keep the name
// for continuity with the original code.)
static double bessj0(double x)
{
    return (x*x > 1.0e-100) ? (sin(x)/x) : 1.0;
}


// Throws if a MKL DFTI call returned an error status (used only in the constructor, not a hot loop).
static void check_dfti(MKL_LONG status, const char *what)
{
    if ((status != 0) && !DftiErrorClass(status, DFTI_NO_ERROR)) {
        stringstream ss;
        ss << "pirate::simpulse: MKL DFTI error in " << what << ": " << DftiErrorMessage(status);
        throw runtime_error(ss.str());
    }
}


// RAII wrapper for a MKL DFTI descriptor, so it is freed on every exit path from the constructor
// (including exceptions thrown in the per-channel loop, e.g. the "samples at t<0" error).
struct DftiHandle
{
    DFTI_DESCRIPTOR_HANDLE h = nullptr;
    ~DftiHandle() { if (h) DftiFreeDescriptor(&h); }
};


// Helper called by the constructor. 'arr' has length (internal_nt+1); 's' is a time in "sample coords",
// i.e. elements of 'arr' correspond to times s = 0, 1, ..., internal_nt.
static double interpolate_cumsum(long internal_nt, const double *arr, double s)
{
    if (s < 1.0e-10)
        return 0.0;
    if (s > internal_nt - 1.0e-10)
        return arr[internal_nt];

    long is = (long)s;
    double ds = s - is;
    xassert((is >= 0) && (is < internal_nt));

    return (1.0-ds)*arr[is] + ds*arr[is+1];
}


// -------------------------------------------------------------------------------------------------
//
// SinglePulse: constructor + methods.


// Validates freq_edges_MHz and freq_variances, and returns 'params' unchanged. The array members
// are stored BY REFERENCE (the ksgpu::Array copy shares the caller's data; not deep-copied).
// Declared in simpulse.hpp; called from the constructor's member-init list.
SinglePulse::Params SinglePulse::_validate(const Params &params)
{
    xassert(params.freq_edges_MHz.ndim == 1);
    xassert(params.freq_edges_MHz.is_fully_contiguous());
    xassert(params.freq_edges_MHz.on_host());   // host code dereferences .data below

    long nfreq = params.freq_edges_MHz.size - 1;
    xassert(nfreq >= 1);   // need at least one frequency channel (i.e. >= 2 edges)

    const double *edges = params.freq_edges_MHz.data;
    xassert(edges[0] > 0.0);   // lowest frequency edge must be positive (freqs are used as divisors)
    xassert(is_sorted(std::vector<double>(edges, edges + nfreq + 1)));   // strictly increasing

    // freq_variances: length nfreq, contiguous, all strictly positive.
    xassert(params.freq_variances.ndim == 1);
    xassert(params.freq_variances.is_fully_contiguous());
    xassert(params.freq_variances.on_host());   // host code dereferences .data below
    xassert(params.freq_variances.size == nfreq);
    const double *var = params.freq_variances.data;
    for (long ifreq = 0; ifreq < nfreq; ifreq++)
        xassert(var[ifreq] > 0.0);

    return params;   // arrays stored by reference (not deep-copied)
}


SinglePulse::SinglePulse(const Params &p)
    : params(_validate(p))
{
    // Local aliases (read-only) for the construction params. 'params' is the (validated) member;
    // its freq_edges_MHz and freq_variances are stored by reference (not deep-copied).
    const long internal_nt = params.internal_nt;
    const double time_sample_ms = params.time_sample_ms;
    const double dm = params.dm;
    const double sm = params.sm;
    const double intrinsic_width = params.intrinsic_width;
    const double undispersed_arrival_time_sec = params.undispersed_arrival_time_sec;
    const double subband_freq_lo_MHz = params.subband_freq_lo_MHz;
    const double subband_freq_hi_MHz = params.subband_freq_hi_MHz;

    // The i-th channel spans [edges[i], edges[i+1]]; channel widths need not be equal. 'variances'
    // is the per-channel noise variance, used to normalize the pulse to params.snr (after the loop).
    const double *edges = params.freq_edges_MHz.data;
    const double *variances = params.freq_variances.data;
    const long nfreq = params.freq_edges_MHz.size - 1;

    xassert(internal_nt >= 64);       // using fewer time samples than this is probably a mistake
    xassert(is_power_of_two(internal_nt));   // the FFT size is 2*internal_nt
    // These params have out-of-range sentinel defaults (see simpulse.hpp), so a value the caller
    // forgot to set is caught here rather than silently used.
    xassert_msg(dm >= 0.0, "pirate::simpulse::SinglePulse: dm is negative or uninitialized");
    xassert_msg(sm >= 0.0, "pirate::simpulse::SinglePulse: sm is negative or uninitialized");
    xassert_msg(intrinsic_width >= 0.0, "pirate::simpulse::SinglePulse: intrinsic_width is negative or uninitialized");
    xassert_msg(params.spectral_index > -1000.0, "pirate::simpulse::SinglePulse: spectral_index is uninitialized");
    xassert_msg(undispersed_arrival_time_sec > -1.0e6, "pirate::simpulse::SinglePulse: undispersed_arrival_time_sec is uninitialized");
    xassert_msg(time_sample_ms > 0.0, "pirate::simpulse::SinglePulse: time_sample_ms is negative or uninitialized");
    xassert_msg(params.snr >= 0.0, "pirate::simpulse::SinglePulse: snr is negative or uninitialized");

    // Subband: the pulse is simulated only in channels overlapping [subband_freq_lo_MHz, subband_freq_hi_MHz]
    // (the default [0, 1e9] overlaps every channel). The subband must overlap the frequency band.
    xassert_lt(subband_freq_lo_MHz, subband_freq_hi_MHz);
    if ((subband_freq_hi_MHz <= edges[0]) || (subband_freq_lo_MHz >= edges[nfreq]))
        throw runtime_error("pirate::simpulse::SinglePulse: subband [subband_freq_lo_MHz, subband_freq_hi_MHz]"
                            " does not overlap the frequency band");

    // Implementing delta-function pulses wouldn't be a big deal, but creates corner cases, and so
    // far there hasn't been a strong reason to implement it.
    if ((dm == 0.0) && (sm == 0.0) && (intrinsic_width == 0.0))
        throw runtime_error("pirate::simpulse::SinglePulse: delta-function pulse (dm=sm=width=0) is currently not allowed");

    const double dt = 1.0e-3 * time_sample_ms;          // sample duration in seconds
    const double nu0 = 0.5 * (edges[0] + edges[nfreq]);  // spectral reference frequency nu_0

    // Per-channel index arrays (members).
    freq_it0     = ksgpu::Array<long>({nfreq}, ksgpu::af_uhost);
    freq_nt      = ksgpu::Array<long>({nfreq}, ksgpu::af_uhost);
    freq_sd_off = ksgpu::Array<long>({nfreq}, ksgpu::af_uhost);

    // FFT scratch + a per-channel cumsum temp (reused each iteration -- these are the old
    // "time-sampling-agnostic" arrays, now one dimension lower since they aren't retained).
    long nfft = 2 * internal_nt;
    long nfft2 = nfft/2 + 1;
    vector<double> bufr(nfft, 0.0);
    vector<complex<double>> bufc(nfft2, complex<double>(0.0, 0.0));
    vector<double> cumsum(internal_nt + 1);

    // Accumulates the packed per-channel samples; copied into the member sparse_data after the loop.
    vector<float> data;

    // MKL c2r (complex-to-real) inverse FFT, replacing the original FFTW fftw_plan_dft_c2r_1d().
    // DftiComputeBackward on a DFTI_REAL descriptor is the c2r transform with the same '+i' sign
    // convention as FFTW, and (like FFTW) MKL is UNNORMALIZED by default, so no scale factor is
    // needed. DFTI_COMPLEX_COMPLEX selects the (n/2+1) packed conjugate-even layout that matches
    // FFTW's c2r input. We build the descriptor once and reuse it across the nfreq loop.
    DftiHandle plan_guard;                        // frees the descriptor on any exit path (RAII)
    DFTI_DESCRIPTOR_HANDLE &plan = plan_guard.h;
    check_dfti(DftiCreateDescriptor(&plan, DFTI_DOUBLE, DFTI_REAL, 1, nfft), "DftiCreateDescriptor");
    check_dfti(DftiSetValue(plan, DFTI_CONJUGATE_EVEN_STORAGE, DFTI_COMPLEX_COMPLEX), "DftiSetValue(CONJUGATE_EVEN_STORAGE)");
    check_dfti(DftiSetValue(plan, DFTI_PLACEMENT, DFTI_NOT_INPLACE), "DftiSetValue(PLACEMENT)");
    check_dfti(DftiCommitDescriptor(plan), "DftiCommitDescriptor");

    long total = 0;       // running sum of freq_nt (== length of 'data')
    double snr_sq = 0.0;  // sum over samples of (sample^2 / variance), at the arbitrary initial norm
    // it_start/it_end: min freq_it0 and max (freq_it0 + freq_nt) over ACTIVE channels. Seeded to
    // sentinels; there is always >= 1 active channel (guaranteed by the subband-overlap check + the
    // total > 0 check below), so both are overwritten before use.
    it_start = std::numeric_limits<long>::max();
    it_end   = std::numeric_limits<long>::min();

    for (long ifreq = 0; ifreq < nfreq; ifreq++) {
        // --- per-channel pulse shape (delays, widths, spectral weight) ---
        double nu_lo = edges[ifreq];
        double nu_hi = edges[ifreq+1];

        // Subband restriction: skip channels that don't overlap [subband_freq_lo_MHz, subband_freq_hi_MHz].
        // A skipped channel contributes no samples (freq_nt = 0), and does not enter the SNR normalization.
        // Its freq_it0 is left for the post-loop fixup (set to it_start).
        if ((nu_hi <= subband_freq_lo_MHz) || (nu_lo >= subband_freq_hi_MHz)) {
            freq_nt.data[ifreq] = 0;
            freq_sd_off.data[ifreq] = total;
            continue;
        }

        double nu_c = (nu_lo + nu_hi) / 2.0;

        double dm_delay0 = dispersion_delay(dm, nu_hi);
        double dm_delay1 = dispersion_delay(dm, nu_lo);
        double dm_width = dm_delay1 - dm_delay0;
        double tscatt = scattering_time(sm, nu_c);

        double t0 = dm_delay0 - 0.1*dm_width - 4.0*intrinsic_width - tscatt;
        double t1 = dm_delay1 + 0.1*dm_width + 4.0*intrinsic_width + 10.0*tscatt;
        double tc = (dm_delay0 + dm_delay1) / 2.0;              // pulse center in channel
        double dt_center = tc - (t0 + (t1-t0)/(2.0*internal_nt));  // pulse center relative to first sample
        double freq_wt = pow(nu_c/nu0, params.spectral_index);  // spectral weight (folds in _compute_freq_wt)

        xassert(t0 < t1);

        // --- FFT synthesis of the pulse -> cumsum[] (normalized to sum=1) ---
        // We sample the pulse at t_i = t0 + (i+0.5)*(t1-t0)/internal_nt.
        double omega0 = 2.0*M_PI * (double)internal_nt / (double)nfft / (t1-t0);
        for (long j = 0; j < nfft2; j++) {
            double omega = omega0 * j;

            // Fourier transform of the pulse.
            // Note: FFTW/MKL sign convention is T(x) = sum_k T(k) e^{ik.x}.
            complex<double> z = exp(-square(intrinsic_width*omega)/2.0);        // Gaussian pulse
            z *= bessj0(dm_width * omega / 2.0);                                // dispersion broadening
            z /= complex<double>(1.0, tscatt * omega);                         // scatter broadening
            z *= complex<double>(cos(dt_center*omega), -sin(dt_center*omega));  // phase shift to center
            bufc[j] = z;
        }

        check_dfti(DftiComputeBackward(plan, (void *)bufc.data(), (void *)bufr.data()), "DftiComputeBackward");

        // Cumulative sum (cleaning up samples which are negative due to discretization), normalized.
        cumsum[0] = 0.0;
        for (long it = 0; it < internal_nt; it++)
            cumsum[it+1] = cumsum[it] + max(bufr[it], 0.0);

        // Guard the normalization against a degenerate (all-nonpositive) synthesized profile, which
        // would divide by zero below and poison the pulse with NaNs (otherwise only caught much later).
        if (cumsum[internal_nt] <= 0.0) {
            stringstream ss;
            ss << "pirate::simpulse::SinglePulse: channel " << ifreq << " produced a degenerate pulse"
               << " profile (FFT synthesis yielded no positive samples); check the pulse parameters";
            throw runtime_error(ss.str());
        }

        for (long it = 0; it < internal_nt+1; it++)
            cumsum[it] = cumsum[it] / cumsum[internal_nt];

        // --- sparse indices on the integer grid (sample 'it' spans [it*dt, (it+1)*dt]) ---
        // Sample indices may be negative (a pulse whose arrival extends to t < 0); this is always
        // allowed. Consumers handle the range via it_start/it_end (see add_to_timestream()).
        double pt0 = undispersed_arrival_time_sec + t0;   // absolute pulse start (seconds)
        double pt1 = undispersed_arrival_time_sec + t1;   // absolute pulse end
        long i0 = (long)floor(pt0 / dt);   // first sample index (may be < 0)
        long i1 = (long)ceil(pt1 / dt);
        long n = i1 - i0;   // >= 1, since pt1 > pt0

        freq_it0.data[ifreq] = i0;
        freq_nt.data[ifreq] = n;
        freq_sd_off.data[ifreq] = total;
        total += n;
        it_start = std::min(it_start, i0);
        it_end   = std::max(it_end, i0 + n);

        // --- fill this channel's samples for dense indices [i0, i1) into 'data' ---
        // Arbitrary initial normalization (spectral weight, no fluence); the whole pulse is rescaled
        // to params.snr after the loop. Accumulate snr_sq = sum(sample^2 / variance) meanwhile.
        double w = freq_wt / dt;
        double var_i = variances[ifreq];
        for (long it = i0; it < i1; it++) {
            double a = interpolate_cumsum(internal_nt, cumsum.data(), internal_nt * ((double)it     * dt - pt0) / (t1 - t0));
            double b = interpolate_cumsum(internal_nt, cumsum.data(), internal_nt * ((double)(it+1) * dt - pt0) / (t1 - t0));
            double sk = w * (b - a);
            data.push_back((float)sk);
            snr_sq += sk * sk / var_i;
        }
    }

    // Defensive: every channel overlapping the subband contributes n >= 1 samples, and the subband
    // overlap was validated above, so this should be unreachable. (Also guarantees it_start/it_end
    // were set by at least one active channel.)
    if (total <= 0)
        throw runtime_error("pirate::simpulse::SinglePulse: pulse has no samples (internal error --"
                            " this should be unreachable)");

    // Assign inactive (subband-skipped, freq_nt == 0) channels freq_it0 = it_start, so that every
    // channel satisfies it_start <= freq_it0 <= (freq_it0 + freq_nt) <= it_end.
    for (long ifreq = 0; ifreq < nfreq; ifreq++)
        if (freq_nt.data[ifreq] == 0)
            freq_it0.data[ifreq] = it_start;

    // Normalize the pulse to the requested matched-filter SNR. The initial (arbitrary) normalization
    // has SNR = sqrt(snr_sq) = sqrt(sum sample^2/variance); scaling every sample by
    // (params.snr / initial_snr) makes sum(sample^2 / variance) == params.snr^2.
    xassert(snr_sq > 0.0);
    double scale = params.snr / sqrt(snr_sq);

    sparse_data = ksgpu::Array<float>({total}, ksgpu::af_uhost);
    for (long k = 0; k < total; k++)
        sparse_data.data[k] = (float)(data[k] * scale);
}


void SinglePulse::add_to_timestream(ksgpu::Array<float> out, long out_it0, float weight) const
{
    const long nfreq = params.freq_edges_MHz.size - 1;

    xassert(out.on_host());            // host code dereferences out.data below
    xassert_eq(out.ndim, 2);
    xassert_eq(out.shape[0], nfreq);
    xassert_gt(out.shape[1], 0L);
    xassert_eq(out.strides[1], 1L);    // time samples must be contiguous in memory

    const long out_nt = out.shape[1];

    // 'out' spans grid sample indices [out_it0, out_it0 + out_nt) and MUST cover the pulse's full
    // range [it_start, it_end) -- no clipping. (This guarantees every write lands in bounds.)
    if ((out_it0 > it_start) || (out_it0 + out_nt < it_end)) {
        stringstream ss;
        ss << "pirate::simpulse::SinglePulse::add_to_timestream: output array spans time samples ["
           << out_it0 << ", " << (out_it0 + out_nt) << "), which does not cover the pulse's range ["
           << it_start << ", " << it_end << ")";
        throw runtime_error(ss.str());
    }

    const long row_stride = out.strides[0];   // row stride, in elements (ksgpu strides are in dtype units)
    const float wf = weight;

    for (long ifreq = 0; ifreq < nfreq; ifreq++) {
        long i0 = freq_it0.data[ifreq];
        long n = freq_nt.data[ifreq];
        long off = freq_sd_off.data[ifreq];

        // base >= 0 and base + n <= out_nt, guaranteed by the span check + the class invariant
        // it_start <= freq_it0 <= freq_it0 + freq_nt <= it_end.
        long base = i0 - out_it0;
        float *row = out.data + ifreq * row_stride;
        const float *src = sparse_data.data + off;
        for (long j = 0; j < n; j++)
            row[base + j] += wf * src[j];
    }
}


void SinglePulse::shift_samples(long delta_it)
{
    const long nfreq = params.freq_edges_MHz.size - 1;
    long *it0 = freq_it0.data;
    for (long ifreq = 0; ifreq < nfreq; ifreq++)
        it0[ifreq] += delta_it;

    it_start += delta_it;
    it_end   += delta_it;
    params.undispersed_arrival_time_sec += 1.0e-3 * (double) delta_it * params.time_sample_ms;
}


void SinglePulse::print(ostream &os) const
{
    const double *edges = params.freq_edges_MHz.data;
    const long nfreq = params.freq_edges_MHz.size - 1;

    os << "SinglePulse(internal_nt=" << params.internal_nt << ",time_sample_ms=" << params.time_sample_ms
       << ",nfreq=" << nfreq << ",freq_lo_MHz=" << edges[0] << ",freq_hi_MHz=" << edges[nfreq]
       << ",dm=" << params.dm << ",sm=" << params.sm << ",intrinsic_width=" << params.intrinsic_width
       << ",snr=" << params.snr << ",spectral_index=" << params.spectral_index
       << ",undispersed_arrival_time_sec=" << params.undispersed_arrival_time_sec
       << ",subband_freq_lo_MHz=" << params.subband_freq_lo_MHz
       << ",subband_freq_hi_MHz=" << params.subband_freq_hi_MHz << ")";
}


string SinglePulse::str() const
{
    stringstream ss;
    this->print(ss);
    return ss.str();
}


}}  // namespace pirate::simpulse
