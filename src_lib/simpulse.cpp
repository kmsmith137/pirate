#include "../include/pirate/simpulse.hpp"
#include "../include/pirate/constants.hpp"   // pirate::constants::k_dm
#include "../include/pirate/inlines.hpp"     // pirate::square, pirate::is_sorted

#include <cmath>
#include <vector>
#include <complex>
#include <sstream>
#include <cstring>
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
    // constants::k_dm is in (ms . MHz^2) per (pc cm^{-3}); convert the delay to seconds.
    return 1.0e-3 * constants::k_dm * dm / (freq_MHz * freq_MHz);
}


double scattering_time(double sm, double freq_MHz)
{
    // 'sm' is the scattering time in milliseconds at 1 GHz.
    return 1.0e-3 * sm / pow(freq_MHz/1000.0, 4.4);
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


// Helper called by the constructor. 'arr' has length (pulse_nt+1); 's' is a time in "sample coords",
// i.e. elements of 'arr' correspond to times s = 0, 1, ..., pulse_nt.
static double interpolate_cumsum(long pulse_nt, const double *arr, double s)
{
    if (s < 1.0e-10)
        return 0.0;
    if (s > pulse_nt - 1.0e-10)
        return arr[pulse_nt];

    long is = (long)s;
    double ds = s - is;
    xassert((is >= 0) && (is < pulse_nt));

    return (1.0-ds)*arr[is] + ds*arr[is+1];
}


// -------------------------------------------------------------------------------------------------
//
// SinglePulse: constructor + methods.


// Validates freq_edges_MHz and returns a copy of 'params' whose freq_edges_MHz is an owned deep
// copy, so the constructed pulse is self-contained. (Declared in simpulse.hpp; called from the
// constructor's member-init list.)
SinglePulse::Params SinglePulse::_validate(const Params &params)
{
    xassert(params.freq_edges_MHz.ndim == 1);
    xassert(params.freq_edges_MHz.is_fully_contiguous());

    long nfreq = params.freq_edges_MHz.size - 1;
    xassert(nfreq >= 1);   // need at least one frequency channel (i.e. >= 2 edges)

    const double *edges = params.freq_edges_MHz.data;
    xassert(edges[0] > 0.0);   // lowest frequency edge must be positive (freqs are used as divisors)
    xassert(is_sorted(std::vector<double>(edges, edges + nfreq + 1)));   // strictly increasing

    Params ret = params;
    ret.freq_edges_MHz = params.freq_edges_MHz.clone();   // deep copy -> the pulse owns its edges
    return ret;
}


SinglePulse::SinglePulse(const Params &p)
    : params(_validate(p))
{
    // Local aliases (read-only) for the construction params. 'params' is the (validated) member,
    // whose freq_edges_MHz is an owned deep copy.
    const long pulse_nt = params.pulse_nt;
    const double time_sample_ms = params.time_sample_ms;
    const double dm = params.dm;
    const double sm = params.sm;
    const double intrinsic_width = params.intrinsic_width;
    const double fluence = params.fluence;
    const double undispersed_arrival_time = params.undispersed_arrival_time;

    // The i-th channel spans [edges[i], edges[i+1]]; channel widths need not be equal.
    const double *edges = params.freq_edges_MHz.data;
    const long nfreq = params.freq_edges_MHz.size - 1;

    xassert(pulse_nt >= 64);          // using fewer time samples than this is probably a mistake
    xassert(time_sample_ms > 0.0);    // required; determines the (zero-based) time grid
    xassert(dm >= 0.0);
    xassert(sm >= 0.0);
    xassert(intrinsic_width >= 0.0);
    xassert(fluence >= 0.0);

    // Implementing delta-function pulses wouldn't be a big deal, but creates corner cases, and so
    // far there hasn't been a strong reason to implement it.
    if ((dm == 0.0) && (sm == 0.0) && (intrinsic_width == 0.0))
        throw runtime_error("pirate::simpulse::SinglePulse: delta-function pulse (dm=sm=width=0) is currently not allowed");

    const double dt = 1.0e-3 * time_sample_ms;          // sample duration in seconds
    const double nu0 = 0.5 * (edges[0] + edges[nfreq]);  // spectral reference frequency nu_0

    // Per-channel index arrays (members).
    sparse_i0     = ksgpu::Array<long>({nfreq}, ksgpu::af_uhost);
    sparse_n      = ksgpu::Array<long>({nfreq}, ksgpu::af_uhost);
    sparse_offset = ksgpu::Array<long>({nfreq}, ksgpu::af_uhost);

    // FFT scratch + a per-channel cumsum temp (reused each iteration -- these are the old
    // "time-sampling-agnostic" arrays, now one dimension lower since they aren't retained).
    long nfft = 2 * pulse_nt;
    long nfft2 = nfft/2 + 1;
    vector<double> bufr(nfft, 0.0);
    vector<complex<double>> bufc(nfft2, complex<double>(0.0, 0.0));
    vector<double> cumsum(pulse_nt + 1);

    // Accumulates the packed per-channel samples; copied into the member sparse_data after the loop.
    vector<float> data;

    // MKL c2r (complex-to-real) inverse FFT, replacing the original FFTW fftw_plan_dft_c2r_1d().
    // DftiComputeBackward on a DFTI_REAL descriptor is the c2r transform with the same '+i' sign
    // convention as FFTW, and (like FFTW) MKL is UNNORMALIZED by default, so no scale factor is
    // needed. DFTI_COMPLEX_COMPLEX selects the (n/2+1) packed conjugate-even layout that matches
    // FFTW's c2r input. We build the descriptor once and reuse it across the nfreq loop.
    DFTI_DESCRIPTOR_HANDLE plan = nullptr;
    check_dfti(DftiCreateDescriptor(&plan, DFTI_DOUBLE, DFTI_REAL, 1, nfft), "DftiCreateDescriptor");
    check_dfti(DftiSetValue(plan, DFTI_CONJUGATE_EVEN_STORAGE, DFTI_COMPLEX_COMPLEX), "DftiSetValue(CONJUGATE_EVEN_STORAGE)");
    check_dfti(DftiSetValue(plan, DFTI_PLACEMENT, DFTI_NOT_INPLACE), "DftiSetValue(PLACEMENT)");
    check_dfti(DftiCommitDescriptor(plan), "DftiCommitDescriptor");

    long total = 0;   // running sum of sparse_n (== length of 'data')
    nt_min = 0;

    for (long ifreq = 0; ifreq < nfreq; ifreq++) {
        // --- per-channel pulse shape (delays, widths, spectral weight) ---
        double nu_lo = edges[ifreq];
        double nu_hi = edges[ifreq+1];
        double nu_c = (nu_lo + nu_hi) / 2.0;

        double dm_delay0 = dispersion_delay(dm, nu_hi);
        double dm_delay1 = dispersion_delay(dm, nu_lo);
        double dm_width = dm_delay1 - dm_delay0;
        double tscatt = scattering_time(sm, nu_c);

        double t0 = dm_delay0 - 0.1*dm_width - 4.0*intrinsic_width - tscatt;
        double t1 = dm_delay1 + 0.1*dm_width + 4.0*intrinsic_width + 10.0*tscatt;
        double tc = (dm_delay0 + dm_delay1) / 2.0;              // pulse center in channel
        double dt_center = tc - (t0 + (t1-t0)/(2.0*pulse_nt));  // pulse center relative to first sample
        double freq_wt = pow(nu_c/nu0, params.spectral_index);  // spectral weight (folds in _compute_freq_wt)

        xassert(t0 < t1);

        // --- FFT synthesis of the pulse -> cumsum[] (normalized to sum=1) ---
        // We sample the pulse at t_i = t0 + (i+0.5)*(t1-t0)/pulse_nt.
        double omega0 = 2.0*M_PI * (double)pulse_nt / (double)nfft / (t1-t0);
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
        for (long it = 0; it < pulse_nt; it++)
            cumsum[it+1] = cumsum[it] + max(bufr[it], 0.0);
        for (long it = 0; it < pulse_nt+1; it++)
            cumsum[it] = cumsum[it] / cumsum[pulse_nt];

        // --- sparse indices on the zero-based grid (sample 'it' spans [it*dt, (it+1)*dt]) ---
        double pt0 = undispersed_arrival_time + t0;   // absolute pulse start (seconds)
        double pt1 = undispersed_arrival_time + t1;   // absolute pulse end
        long i0 = std::max(0L, (long)floor(pt0 / dt));
        long i1 = std::max(0L, (long)ceil (pt1 / dt));
        long n = i1 - i0;

        sparse_i0.data[ifreq] = i0;
        sparse_n.data[ifreq] = n;
        sparse_offset.data[ifreq] = total;
        total += n;
        nt_min = std::max(nt_min, i0 + n);

        // --- fill this channel's samples for dense indices [i0, i1) into 'data' ---
        // (This is the old add_pulse_to_frequency_channel, inlined; 'weight' baked in as 1.)
        double w = fluence * freq_wt / dt;   // per-call weight is applied later, in add_to_timestream()
        for (long it = i0; it < i1; it++) {
            double a = interpolate_cumsum(pulse_nt, cumsum.data(), pulse_nt * ((double)it     * dt - pt0) / (t1 - t0));
            double b = interpolate_cumsum(pulse_nt, cumsum.data(), pulse_nt * ((double)(it+1) * dt - pt0) / (t1 - t0));
            data.push_back((float)(w * (b - a)));
        }
    }

    DftiFreeDescriptor(&plan);

    if (total <= 0)
        throw runtime_error("pirate::simpulse::SinglePulse: pulse has no samples on the zero-based time grid"
                            " (is undispersed_arrival_time very negative?)");

    // Copy the packed samples into the member array.
    sparse_data = ksgpu::Array<float>({total}, ksgpu::af_uhost);
    memcpy(sparse_data.data, data.data(), total * sizeof(float));
}


void SinglePulse::add_to_timestream(ksgpu::Array<float> &out, double weight) const
{
    const long nfreq = params.freq_edges_MHz.size - 1;

    xassert(out.ndim == 2);
    xassert(out.shape[0] == nfreq);
    xassert(out.shape[1] > 0);
    xassert(out.strides[1] == 1);   // time samples must be contiguous in memory

    const long out_nt = out.shape[1];
    const long row_stride = out.strides[0];   // row stride, in elements (ksgpu strides are in dtype units)
    const float wf = (float)weight;

    for (long ifreq = 0; ifreq < nfreq; ifreq++) {
        long i0 = sparse_i0.data[ifreq];
        long n = sparse_n.data[ifreq];
        long off = sparse_offset.data[ifreq];

        // Clip to [0, out_nt). i0 >= 0 by construction; clip the high end.
        long n_eff = std::min(n, out_nt - i0);
        if ((i0 >= out_nt) || (n_eff <= 0))
            continue;

        float *row = out.data + ifreq * row_stride;
        const float *src = sparse_data.data + off;
        for (long j = 0; j < n_eff; j++)
            row[i0 + j] += wf * src[j];
    }
}


double SinglePulse::get_signal_to_noise(double sample_rms) const
{
    xassert(sample_rms > 0.0);

    double acc = 0.0;
    for (long k = 0; k < sparse_data.size; k++)
        acc += square((double)sparse_data.data[k]);

    return sqrt(acc) / sample_rms;
}


double SinglePulse::get_signal_to_noise(const ksgpu::Array<double> &sample_rms,
                                        const ksgpu::Array<double> &channel_weights) const
{
    const long nfreq = params.freq_edges_MHz.size - 1;
    xassert((sample_rms.ndim == 1) && (sample_rms.shape[0] == nfreq) && sample_rms.is_fully_contiguous());

    const double *rms = sample_rms.data;

    // An empty 'channel_weights' (the default) means "use 1/sample_rms^2 weighting".
    const double *cw = nullptr;
    bool have_cw = (channel_weights.data != nullptr) && (channel_weights.size > 0);
    if (have_cw) {
        xassert((channel_weights.ndim == 1) && (channel_weights.shape[0] == nfreq) && channel_weights.is_fully_contiguous());
        cw = channel_weights.data;
    }

    for (long ifreq = 0; ifreq < nfreq; ifreq++) {
        xassert(rms[ifreq] >= 0.0);
        if (cw != nullptr)
            xassert(cw[ifreq] >= 0.0);
    }

    vector<double> wtmp;
    if (cw == nullptr) {
        wtmp.resize(nfreq);
        for (long ifreq = 0; ifreq < nfreq; ifreq++) {
            xassert(rms[ifreq] > 0.0);   // need positive rms to form 1/rms^2 default weighting
            wtmp[ifreq] = 1.0 / (rms[ifreq] * rms[ifreq]);
        }
        cw = &wtmp[0];
    }

    double sig_ampl = 0.0;
    double noise_var = 0.0;

    for (long ifreq = 0; ifreq < nfreq; ifreq++) {
        long off = sparse_offset.data[ifreq];
        long n = sparse_n.data[ifreq];

        double t = 0.0;   // sum of squared samples in this channel
        for (long j = 0; j < n; j++)
            t += square((double)sparse_data.data[off + j]);

        sig_ampl += cw[ifreq] * t;
        noise_var += square(cw[ifreq] * rms[ifreq]) * t;
    }

    xassert(noise_var > 0.0);   // too many sample_rms (or channel_weights) values were zero
    return sig_ampl / sqrt(noise_var);
}


void SinglePulse::print(ostream &os) const
{
    const double *edges = params.freq_edges_MHz.data;
    const long nfreq = params.freq_edges_MHz.size - 1;

    os << "SinglePulse(pulse_nt=" << params.pulse_nt << ",time_sample_ms=" << params.time_sample_ms
       << ",nfreq=" << nfreq << ",freq_lo_MHz=" << edges[0] << ",freq_hi_MHz=" << edges[nfreq]
       << ",dm=" << params.dm << ",sm=" << params.sm << ",intrinsic_width=" << params.intrinsic_width
       << ",fluence=" << params.fluence << ",spectral_index=" << params.spectral_index
       << ",undispersed_arrival_time=" << params.undispersed_arrival_time << ")";
}


string SinglePulse::str() const
{
    stringstream ss;
    this->print(ss);
    return ss.str();
}


}}  // namespace pirate::simpulse
