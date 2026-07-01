#include "../include/pirate/simpulse.hpp"
#include "../include/pirate/constants.hpp"   // pirate::constants::k_dm
#include "../include/pirate/inlines.hpp"     // pirate::square

#include <cmath>
#include <vector>
#include <complex>
#include <sstream>
#include <algorithm>
#include <stdexcept>

#include <ksgpu/xassert.hpp>

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


// Helper called by add_pulse_to_frequency_channel(). 'arr' has length (pulse_nt+1); 's' is a time
// in "sample coords", i.e. elements of 'arr' correspond to times s = 0, 1, ..., pulse_nt.
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


// Adds the pulse (in frequency channel 'ifreq') to a dense, single-channel output buffer 'out' of
// length 'out_nt', spanning [out_t0, out_t1]. Templated on the output type T: T=float for the public
// add_to_timestream() output, T=double for the get_signal_to_noise() scratch buffer.
template<typename T>
static void add_pulse_to_frequency_channel(const SinglePulse &sp, T *out, double out_t0, double out_t1,
                                           long out_nt, long ifreq, double weight)
{
    const long pulse_nt = sp.params.pulse_nt;
    const double undispersed_arrival_time = sp.params.undispersed_arrival_time;

    xassert(out != nullptr);
    xassert(out_nt > 0);
    xassert(out_t0 < out_t1);
    xassert((ifreq >= 0) && (ifreq < sp.params.freq_edges_MHz.size - 1));

    // Convert input times to "sample coords".
    double s0 = pulse_nt * (out_t0 - undispersed_arrival_time - sp.pulse_t0[ifreq]) / (sp.pulse_t1[ifreq] - sp.pulse_t0[ifreq]);
    double s1 = pulse_nt * (out_t1 - undispersed_arrival_time - sp.pulse_t0[ifreq]) / (sp.pulse_t1[ifreq] - sp.pulse_t0[ifreq]);

    if ((s0 >= pulse_nt) || (s1 <= 0))
        return;

    double out_dt = (out_t1 - out_t0) / out_nt;
    double w = weight * sp.params.fluence * sp.pulse_freq_wt[ifreq] / out_dt;
    const double *cs = &sp.pulse_cumsum[ifreq * (pulse_nt+1)];

    for (long it = 0; it < out_nt; it++) {
        double a = interpolate_cumsum(pulse_nt, cs, s0 + (it)   * (s1-s0) / (double)out_nt);
        double b = interpolate_cumsum(pulse_nt, cs, s0 + (it+1) * (s1-s0) / (double)out_nt);
        out[it] += w * (b - a);
    }
}


// Computes the dense output-sample range [*sparse_i0, *sparse_i0 + *sparse_n) spanned by the pulse
// in frequency channel 'ifreq'. Sets *sparse_n=0 if the pulse does not overlap the output.
static void get_pulse_n_samples(const SinglePulse &sp, long *sparse_i0, long *sparse_n,
                                double out_t0, double out_t1, long out_nt, long ifreq)
{
    const double undispersed_arrival_time = sp.params.undispersed_arrival_time;

    xassert(out_nt > 0);
    xassert(out_t0 < out_t1);
    xassert((ifreq >= 0) && (ifreq < sp.params.freq_edges_MHz.size - 1));

    double out_dt = (out_t1 - out_t0) / out_nt;
    double pulse_t0 = undispersed_arrival_time + sp.pulse_t0[ifreq];
    double pulse_t1 = undispersed_arrival_time + sp.pulse_t1[ifreq];

    if ((pulse_t0 > out_t1) || (pulse_t1 < out_t0)) {
        *sparse_i0 = 0;
        *sparse_n = 0;
        return;
    }

    double out_i0_f = (pulse_t0 - out_t0) / out_dt;
    double out_i1_f = (pulse_t1 - out_t0) / out_dt;

    // Clip both endpoints to [0, out_nt].
    long out_i0 = (long)floor(out_i0_f);
    long out_i1 = (long)ceil(out_i1_f);
    out_i0 = min(out_nt, max(0L, out_i0));
    out_i1 = min(out_nt, max(0L, out_i1));

    *sparse_i0 = out_i0;
    *sparse_n = out_i1 - out_i0;
}


// Sparse counterpart of add_pulse_to_frequency_channel(): writes the channel's samples densely into
// 'out' (length *sparse_n), and reports the dense-array offset/count in *sparse_i0 / *sparse_n.
template<typename T>
static void add_pulse_to_frequency_channel_sparse(const SinglePulse &sp, T *out, long *sparse_i0, long *sparse_n,
                                                  double out_t0, double out_t1, long out_nt, long ifreq, double weight)
{
    xassert(out != nullptr);
    xassert(out_nt > 0);
    xassert(out_t0 < out_t1);
    xassert((ifreq >= 0) && (ifreq < sp.params.freq_edges_MHz.size - 1));

    double out_dt = (out_t1 - out_t0) / out_nt;
    get_pulse_n_samples(sp, sparse_i0, sparse_n, out_t0, out_t1, out_nt, ifreq);
    if (*sparse_n == 0)
        return;

    double ot0 = out_t0 + (*sparse_i0)              * out_dt;
    double ot1 = out_t0 + (*sparse_i0 + *sparse_n)  * out_dt;
    add_pulse_to_frequency_channel(sp, out, ot0, ot1, *sparse_n, ifreq, weight);
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
    // Local aliases (read-only) for the construction params used below, so the synthesis math reads
    // cleanly. 'params' is the (validated) member, whose freq_edges_MHz is an owned deep copy.
    const long pulse_nt = params.pulse_nt;
    const double dm = params.dm;
    const double sm = params.sm;
    const double intrinsic_width = params.intrinsic_width;
    const double fluence = params.fluence;

    // The i-th channel spans [edges[i], edges[i+1]]; channel widths need not be equal.
    const double *edges = params.freq_edges_MHz.data;
    const long nfreq = params.freq_edges_MHz.size - 1;

    xassert(pulse_nt >= 64);   // using fewer time samples than this is probably a mistake
    xassert(dm >= 0.0);
    xassert(sm >= 0.0);
    xassert(intrinsic_width >= 0.0);
    xassert(fluence >= 0.0);

    // Implementing delta-function pulses wouldn't be a big deal, but creates corner cases, and so
    // far there hasn't been a strong reason to implement it.
    if ((dm == 0.0) && (sm == 0.0) && (intrinsic_width == 0.0))
        throw runtime_error("pirate::simpulse::SinglePulse: delta-function pulse (dm=sm=width=0) is currently not allowed");

    pulse_t0.resize(nfreq, 0.0);
    pulse_t1.resize(nfreq, 0.0);
    pulse_freq_wt.resize(nfreq, 0.0);
    pulse_cumsum.resize(nfreq * (pulse_nt+1), 0.0);

    this->_compute_freq_wt();

    long nfft = 2 * pulse_nt;
    long nfft2 = nfft/2 + 1;

    vector<double> bufr(nfft, 0.0);
    vector<complex<double>> bufc(nfft2, complex<double>(0.0, 0.0));

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

    // The following loop synthesizes the pulse. We sample at t_i = t0 + (i+0.5)*(t1-t0)/pulse_nt.
    for (long ifreq = 0; ifreq < nfreq; ifreq++) {
        double nu_lo = edges[ifreq];
        double nu_hi = edges[ifreq+1];
        double nu_c = (nu_lo + nu_hi) / 2.0;

        double dm_delay0 = dispersion_delay(dm, nu_hi);
        double dm_delay1 = dispersion_delay(dm, nu_lo);
        double dm_width = dm_delay1 - dm_delay0;
        double tscatt = scattering_time(sm, nu_c);

        double t0 = dm_delay0 - 0.1*dm_width - 4.0*intrinsic_width - tscatt;
        double t1 = dm_delay1 + 0.1*dm_width + 4.0*intrinsic_width + 10.0*tscatt;
        double tc = (dm_delay0 + dm_delay1) / 2.0;        // pulse center in channel
        double dt = tc - (t0 + (t1-t0)/(2.0*pulse_nt));   // pulse center relative to first sample

        xassert(t0 < t1);
        pulse_t0[ifreq] = t0;
        pulse_t1[ifreq] = t1;

        double *p = &pulse_cumsum[ifreq * (pulse_nt+1)];
        double omega0 = 2.0*M_PI * (double)pulse_nt / (double)nfft / (t1-t0);

        for (long j = 0; j < nfft2; j++) {
            double omega = omega0 * j;

            // Fourier transform of the pulse.
            // Note: FFTW/MKL sign convention is T(x) = sum_k T(k) e^{ik.x}.
            complex<double> z = exp(-square(intrinsic_width*omega)/2.0);     // Gaussian pulse
            z *= bessj0(dm_width * omega / 2.0);                             // dispersion broadening
            z /= complex<double>(1.0, tscatt * omega);                      // scatter broadening
            z *= complex<double>(cos(dt*omega), -sin(dt*omega));            // phase shift to center
            bufc[j] = z;
        }

        // FFT bufc -> bufr (complex-to-real).
        check_dfti(DftiComputeBackward(plan, (void *)bufc.data(), (void *)bufr.data()), "DftiComputeBackward");

        // Evaluate cumsum, cleaning up samples which are negative due to discretization effects.
        for (long it = 0; it < pulse_nt; it++)
            p[it+1] = p[it] + max(bufr[it], 0.0);

        // Normalize to sum=1.
        for (long it = 0; it < pulse_nt+1; it++)
            p[it] = p[it] / p[pulse_nt];
    }

    DftiFreeDescriptor(&plan);

    // Initialize min_t0, max_t1, max_dt.
    min_t0 = pulse_t0[0];
    max_t1 = pulse_t1[0];
    max_dt = pulse_t1[0] - pulse_t0[0];

    for (long ifreq = 1; ifreq < nfreq; ifreq++) {
        min_t0 = min(min_t0, pulse_t0[ifreq]);
        max_t1 = max(max_t1, pulse_t1[ifreq]);
        max_dt = max(max_dt, pulse_t1[ifreq] - pulse_t0[ifreq]);
    }
}


void SinglePulse::_compute_freq_wt()
{
    const double spectral_index = params.spectral_index;
    const double *edges = params.freq_edges_MHz.data;
    const long nfreq = params.freq_edges_MHz.size - 1;

    // Disallow 'extreme' spectral_index values (pow() can blow up).
    xassert((spectral_index >= -20.1) && (spectral_index <= 20.1));

    double nu0 = 0.5 * (edges[0] + edges[nfreq]);   // band-center reference frequency nu_0

    for (long ifreq = 0; ifreq < nfreq; ifreq++) {
        double nu = 0.5 * (edges[ifreq] + edges[ifreq+1]);   // channel center frequency
        pulse_freq_wt[ifreq] = pow(nu/nu0, spectral_index);
    }
}


void SinglePulse::get_endpoints(double &t0, double &t1) const
{
    t0 = params.undispersed_arrival_time + min_t0;
    t1 = params.undispersed_arrival_time + max_t1;
}


void SinglePulse::add_to_timestream(ksgpu::Array<float> &out, double out_t0, double out_t1, double weight) const
{
    const long nfreq = params.freq_edges_MHz.size - 1;
    const double undispersed_arrival_time = params.undispersed_arrival_time;

    xassert(out.ndim == 2);
    xassert(out.shape[0] == nfreq);
    xassert(out.shape[1] > 0);
    xassert(out.strides[1] == 1);   // time samples must be contiguous in memory
    xassert(out_t0 < out_t1);

    long out_nt = out.shape[1];
    long stride = out.strides[0];   // row stride, in elements (ksgpu strides are in dtype units)

    // Return early if data does not overlap the pulse.
    if (out_t0 > undispersed_arrival_time + max_t1)
        return;
    if (out_t1 < undispersed_arrival_time + min_t0)
        return;

    for (long ifreq = 0; ifreq < nfreq; ifreq++)
        add_pulse_to_frequency_channel(*this, out.data + ifreq*stride, out_t0, out_t1, out_nt, ifreq, weight);
}


long SinglePulse::get_n_sparse(double out_t0, double out_t1, long out_nt) const
{
    const long nfreq = params.freq_edges_MHz.size - 1;
    const double undispersed_arrival_time = params.undispersed_arrival_time;

    xassert(out_nt > 0);
    xassert(out_t0 < out_t1);

    if (out_t0 > undispersed_arrival_time + max_t1)
        return 0;
    if (out_t1 < undispersed_arrival_time + min_t0)
        return 0;

    long ntotal = 0;
    for (long ifreq = 0; ifreq < nfreq; ifreq++) {
        long i0 = 0, sn = 0;
        get_pulse_n_samples(*this, &i0, &sn, out_t0, out_t1, out_nt, ifreq);
        ntotal += sn;
    }
    return ntotal;
}


void SinglePulse::add_to_timestream_sparse(ksgpu::Array<float> &out, ksgpu::Array<long> &out_i0,
                                           ksgpu::Array<long> &out_n, double out_t0, double out_t1,
                                           long out_nt, double weight) const
{
    const long nfreq = params.freq_edges_MHz.size - 1;
    const double undispersed_arrival_time = params.undispersed_arrival_time;

    xassert(out.ndim == 1);
    xassert(out.is_fully_contiguous());
    xassert((out_i0.ndim == 1) && (out_i0.shape[0] == nfreq) && out_i0.is_fully_contiguous());
    xassert((out_n.ndim == 1) && (out_n.shape[0] == nfreq) && out_n.is_fully_contiguous());
    xassert(out_nt > 0);
    xassert(out_t0 < out_t1);

    // 'out' must be large enough to hold the packed samples for all channels.
    xassert(out.shape[0] >= get_n_sparse(out_t0, out_t1, out_nt));

    long *i0 = out_i0.data;
    long *n = out_n.data;

    // Initialize to an empty pulse.
    for (long ifreq = 0; ifreq < nfreq; ifreq++) {
        i0[ifreq] = 0;
        n[ifreq] = 0;
    }

    // Return early if data does not overlap the pulse.
    if (out_t0 > undispersed_arrival_time + max_t1)
        return;
    if (out_t1 < undispersed_arrival_time + min_t0)
        return;

    long ntotal = 0;
    for (long ifreq = 0; ifreq < nfreq; ifreq++) {
        long nt = 0;
        add_pulse_to_frequency_channel_sparse(*this, out.data + ntotal, i0 + ifreq, &nt, out_t0, out_t1, out_nt, ifreq, weight);
        n[ifreq] = nt;
        ntotal += nt;
    }
}


double SinglePulse::get_signal_to_noise(double sample_dt, double sample_rms, double sample_t0) const
{
    const long nfreq = params.freq_edges_MHz.size - 1;
    const double undispersed_arrival_time = params.undispersed_arrival_time;

    xassert(sample_dt > 0.0);
    xassert(sample_rms > 0.0);

    long nsamp_max = (long)(max_dt/sample_dt) + 3;
    vector<double> buf(nsamp_max, 0.0);

    double acc = 0.0;

    for (long ifreq = 0; ifreq < nfreq; ifreq++) {
        // Range of samples spanned by the pulse.
        double s0 = (undispersed_arrival_time + pulse_t0[ifreq] - sample_t0) / sample_dt;
        double s1 = (undispersed_arrival_time + pulse_t1[ifreq] - sample_t0) / sample_dt;

        long j = (long)floor(s0);
        long k = (long)ceil(s1);
        xassert(k - j <= nsamp_max);

        fill(buf.begin(), buf.end(), 0.0);
        add_pulse_to_frequency_channel(*this, &buf[0], sample_t0 + j*sample_dt, sample_t0 + k*sample_dt, k-j, ifreq, 1.0);

        for (long i = 0; i < k-j; i++)
            acc += buf[i]*buf[i];
    }

    return sqrt(acc) / sample_rms;
}


double SinglePulse::get_signal_to_noise(double sample_dt, const ksgpu::Array<double> &sample_rms,
                                        const ksgpu::Array<double> &channel_weights, double sample_t0) const
{
    const long nfreq = params.freq_edges_MHz.size - 1;
    const double undispersed_arrival_time = params.undispersed_arrival_time;

    xassert(sample_dt > 0.0);
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

    long nsamp_max = (long)(max_dt/sample_dt) + 3;
    vector<double> buf(nsamp_max, 0.0);

    double sig_ampl = 0.0;
    double noise_var = 0.0;

    for (long ifreq = 0; ifreq < nfreq; ifreq++) {
        // Range of samples spanned by the pulse.
        double s0 = (undispersed_arrival_time + pulse_t0[ifreq] - sample_t0) / sample_dt;
        double s1 = (undispersed_arrival_time + pulse_t1[ifreq] - sample_t0) / sample_dt;

        long j = (long)floor(s0);
        long k = (long)ceil(s1);
        xassert(k - j <= nsamp_max);

        fill(buf.begin(), buf.end(), 0.0);
        add_pulse_to_frequency_channel(*this, &buf[0], sample_t0 + j*sample_dt, sample_t0 + k*sample_dt, k-j, ifreq, 1.0);

        double t = 0.0;
        for (long i = 0; i < k-j; i++)
            t += square(buf[i]);

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

    os << "SinglePulse(pulse_nt=" << params.pulse_nt << ",nfreq=" << nfreq
       << ",freq_lo_MHz=" << edges[0] << ",freq_hi_MHz=" << edges[nfreq]
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
