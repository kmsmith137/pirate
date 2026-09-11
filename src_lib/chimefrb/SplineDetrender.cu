#include "../../include/pirate/chimefrb/SplineDetrender.hpp"
#include "../../include/pirate/detrender_kernels.hpp"

#include <cmath>
#include <sstream>
#include <iostream>
#include <ksgpu/xassert.hpp>
#include <ksgpu/cuda_utils.hpp>
#include <ksgpu/KernelTimer.hpp>

using namespace std;
using namespace ksgpu;

namespace pirate {
namespace chimefrb {
#if 0
}}  // editor auto-indent
#endif


// The kernels are GpuDetrenderLps2d's, from detrender_kernels.hpp, and that header
// carries the description of what they do. This file is what makes them compute the
// old code's spline detrender instead: it builds the tables and descriptors they read.
//
//   - BASIS TABLES. The kernels want, per channel, the n_phi+1 = 4 basis functions that
//     are nonzero there and their 10 pairwise products. For a channel in bin b these are
//     the four cubic Hermite functions of x, the position within the bin,
//
//         h00 = (1-x)^2 (1+2x)    h10 = (1-x)^2 x    h01 = x^2 (3-2x)    h11 = x^2 (x-1),
//
//     multiplying the coefficients (value at edge b, slope at edge b, value at edge b+1,
//     slope at edge b+1). Coefficient-major indexing puts those at 2b .. 2b+3, so the
//     freq-range descriptor's "span index" is j0 = 2b + 3 (the kernels take the four
//     coefficients to be j0-3 .. j0). Consecutive bins share their edge coefficients,
//     which is what makes the spline C^1 and the assembled matrix banded with
//     half-bandwidth 3. The formulas, the bin edges and the fractional bin coordinate
//     are copied from rf_kernels' _spline_detrender_init(), so the bin geometry is the
//     old code's exactly.
//
//   - REGULATOR TABLE. The slope penalty sum_bins int_0^1 (db/dx)^2 dx is, per bin, the
//     fixed 4x4 matrix Q_ac = int_0^1 h_a' h_c' dx on those same four coefficients. The
//     table holds it banded, rows accumulating the two bins that share an edge, and the
//     kernel adds strength * Q with strength = (epsilon / nbins) * W_tot, W_tot being the
//     sample's total weight. Recovering W_tot from the Gram matrix needs the coefficient
//     vector of the constant function 1, which for the Hermite basis is
//     (1, 0, 1, 0, ...) -- values one, slopes zero -- since h00 + h01 = 1.
//
//   - ONE ZONE, since the C^1 joins couple every bin to the next, and (n, W) = (0, 0):
//     no time window, one fit per time sample.


// Q_ac = int_0^1 h_a'(x) h_c'(x) dx for the Hermite basis (h00, h10, h01, h11), by
// Gauss-Legendre quadrature: the integrand is a quartic, and 4 points are exact through
// degree 7. Checked against the fractions the old code hard-codes (rf_kernels'
// spline_detrender_internals.hpp: 6/5, 1/10, 2/15, -1/30, and their signed copies),
// so a slip in either the basis or the derivatives below is caught at construction.
static void hermite_slope_penalty(double Q[4][4])
{
    const double xg[4] = { 0.5 - 0.5*sqrt((3.0 + 2.0*sqrt(1.2))/7.0), 0.5 - 0.5*sqrt((3.0 - 2.0*sqrt(1.2))/7.0),
                           0.5 + 0.5*sqrt((3.0 - 2.0*sqrt(1.2))/7.0), 0.5 + 0.5*sqrt((3.0 + 2.0*sqrt(1.2))/7.0) };
    const double wg[4] = { (18.0 - sqrt(30.0))/72.0, (18.0 + sqrt(30.0))/72.0,
                           (18.0 + sqrt(30.0))/72.0, (18.0 - sqrt(30.0))/72.0 };

    for (int a = 0; a < 4; a++)
        for (int c = 0; c < 4; c++)
            Q[a][c] = 0.0;

    for (int g = 0; g < 4; g++) {
        const double x = xg[g];
        const double d[4] = { 6*x*x - 6*x,        // h00'
                              3*x*x - 4*x + 1,    // h10'
                              -6*x*x + 6*x,       // h01'
                              3*x*x - 2*x };      // h11'
        for (int a = 0; a < 4; a++)
            for (int c = 0; c < 4; c++)
                Q[a][c] += wg[g] * d[a] * d[c];
    }

    const double expected[4][4] = { {  6./5.,  1./10., -6./5.,  1./10. },
                                    {  1./10., 2./15., -1./10., -1./30. },
                                    { -6./5., -1./10.,  6./5., -1./10. },
                                    {  1./10., -1./30., -1./10.,  2./15. } };
    for (int a = 0; a < 4; a++)
        for (int c = 0; c < 4; c++)
            xassert_lt(fabs(Q[a][c] - expected[a][c]), 1.0e-12);
}


GpuSplineDetrender::GpuSplineDetrender(long nfreq_, long nbins_, double epsilon_, long M_, long T_) :
    nfreq(nfreq_), nbins(nbins_), epsilon(epsilon_), M(M_), T(T_)
{
    if (nbins < 1)
        throw runtime_error("GpuSplineDetrender: expected nbins >= 1");
    if (nfreq < nbins) {
        stringstream ss;
        ss << "GpuSplineDetrender: nfreq=" << nfreq << " is smaller than nbins=" << nbins
           << "; every bin must hold at least one channel";
        throw runtime_error(ss.str());
    }
    if (!(epsilon > 0.0)) {
        stringstream ss;
        ss << "GpuSplineDetrender: epsilon=" << epsilon << " must be > 0";
        throw runtime_error(ss.str());
    }
    if (M < 1)
        throw runtime_error("GpuSplineDetrender: expected M >= 1");
    if ((T <= 0) || (T % 32 != 0)) {
        stringstream ss;
        ss << "GpuSplineDetrender: T=" << T << " must be a positive multiple of 32";
        throw runtime_error(ss.str());
    }

    N_phi = 2*(nbins+1);

    // ---- Bin edges, as in rf_kernels' _spline_detrender_init(): b*nfreq/nbins rounded
    // to the nearest channel. The expression is written in the same order as the old
    // code's, so the rounding agrees even at exact half-integers.
    _bin_edges.resize(nbins+1);
    for (long b = 0; b <= nbins; b++)
        _bin_edges[b] = long(double(b) / double(nbins) * double(nfreq) + 0.5);

    xassert_eq(_bin_edges[0], 0);
    xassert_eq(_bin_edges[nbins], nfreq);
    for (long b = 0; b < nbins; b++)
        xassert_lt(_bin_edges[b], _bin_edges[b+1]);   // guaranteed by nfreq >= nbins

    // ---- Basis tables, built in float64 and cast. Channel f in bin b has fractional
    // bin coordinate x = nbins*(f+1/2)/nfreq - b, in [0,1]; again the old code's
    // expression, in its order.
    const int n_phi = 3;
    const long npair = 10;
    const long phi_stride = 4;      // 4*((n_phi+1 + 3)/4), as the pirate constructor pads
    const long prod_stride = 12;    // 4*((npair + 3)/4)

    Array<float> phi_host({nfreq, phi_stride}, af_uhost | af_zero);
    Array<float> prod_host({nfreq, prod_stride}, af_uhost | af_zero);

    for (long b = 0; b < nbins; b++) {
        for (long f = _bin_edges[b]; f < _bin_edges[b+1]; f++) {
            const double x = double(nbins) * (double(f) + 0.5) / double(nfreq) - double(b);
            xassert(x > -1.0e-10);
            xassert(x < 1.0 + 1.0e-10);

            const double h[4] = { (1-x)*(1-x)*(1+2*x), (1-x)*(1-x)*x, x*x*(3-2*x), x*x*(x-1) };
            for (long a = 0; a < 4; a++)
                phi_host.data[f*phi_stride + a] = float(h[a]);

            long p = 0;
            for (long a = 0; a < 4; a++)
                for (long c = a; c < 4; c++, p++)
                    prod_host.data[f*prod_stride + p] = float(h[a] * h[c]);
            xassert_eq(p, npair);
        }
    }

    phi_tab = phi_host.to_gpu();
    prod_tab = prod_host.to_gpu();

    // ---- Freq-ranges: each bin cut into pieces of about channels_per_range channels.
    // The floor is 32 rather than the pirate detrender's 128: at the production shape
    // nfreq = 1024, nbins = 6, the bins are 171 channels, and one range per bin would give
    // the accumulate and subtract kernels only 6 * ceil(T/256) blocks per beam. Smaller
    // ranges cost a few more shared-memory adds in the solve kernel's staging, which is
    // nothing. The beam axis is the other occupancy knob: batch beams.
    channels_per_range = derive_channels_per_range(nfreq, T, /*cpr_min=*/32);

    vector<long> fr_lo, fr_hi, fr_j0;
    for (long b = 0; b < nbins; b++) {
        const long lo0 = _bin_edges[b];
        const long len = _bin_edges[b+1] - lo0;
        long k = (len + channels_per_range/2) / channels_per_range;
        if (k < 1)
            k = 1;
        for (long i = 0; i < k; i++) {
            const long lo = lo0 + (len*i)/k;
            const long hi = lo0 + (len*(i+1))/k;
            if (hi <= lo)
                continue;
            fr_lo.push_back(lo);
            fr_hi.push_back(hi);
            fr_j0.push_back(2*b + 3);
        }
    }
    nfrange = long(fr_lo.size());
    xassert_gt(nfrange, 0);

    Array<int> fr_host({nfrange, 4}, af_uhost | af_zero);
    for (long i = 0; i < nfrange; i++) {
        fr_host.data[4*i + 0] = int(fr_lo[i]);
        fr_host.data[4*i + 1] = int(fr_hi[i]);
        fr_host.data[4*i + 2] = int(fr_j0[i]);
        fr_host.data[4*i + 3] = 0;                  // zone
    }
    fr_desc = fr_host.to_gpu();

    // ---- One zone: all freq-ranges, all coefficients.
    Array<int> zone_host({1, 4}, af_uhost | af_zero);
    zone_host.data[0] = 0;
    zone_host.data[1] = int(nfrange);
    zone_host.data[2] = 0;
    zone_host.data[3] = int(N_phi);
    zone_desc = zone_host.to_gpu();

    // ---- Regulator table and the constant function's coefficients (see the top of the
    // file). Q's row a, column c >= a lands in table row 2b+a, band c-a; the rows of an
    // interior edge accumulate from both adjacent bins.
    {
        double Q[4][4];
        hermite_slope_penalty(Q);

        const long nreg = reg_table_width(n_phi);   // 4
        Array<float> reg_host({N_phi, nreg}, af_uhost | af_zero);
        for (long b = 0; b < nbins; b++)
            for (long a = 0; a < 4; a++)
                for (long c = a; c < 4; c++)
                    reg_host.data[(2*b + a)*nreg + (c - a)] += float(Q[a][c]);
        reg_tab = reg_host.to_gpu();

        Array<float> unit_host({N_phi}, af_uhost | af_zero);
        for (long j = 0; j < N_phi; j += 2)
            unit_host.data[j] = 1.0f;
        unit_coef = unit_host.to_gpu();
    }

    // ---- The time basis: (n, W) = (0, 0), a single unit stencil.
    tb_blob = new TimeStencils(make_time_stencils(0, 0));

    // ---- Solve-kernel block size, from the shared-memory budget. At nbins = 6 a
    // 128-thread block fits and a 256-thread one does not; the chooser sorts it out.
    const long NB = bandwidth(n_phi, 0);            // 3
    const long nblk_max = N_phi;                    // (n+1) = 1 coefficient per basis function
    const long ncompz_max = N_phi * (n_phi + 2);    // banded G plus U, per coefficient

    solve_threads = choose_solve_threads<3, WeightScaledStrength>(T, nblk_max, NB, ncompz_max, /*W=*/0);
    if (solve_threads == 0) {
        stringstream ss;
        ss << "GpuSplineDetrender: nbins=" << nbins << " gives " << N_phi << " coefficients,"
           << " which need more shared memory than this GPU offers even at 8 threads per block";
        throw runtime_error(ss.str());
    }

    // ---- Per-launch scratch.
    const long ncomp = npair + n_phi + 1;           // 14
    gu = Array<float>({M, nfrange, ncomp, T}, af_gpu | af_zero);
    acoef = Array<float>({M, N_phi, T}, af_gpu | af_zero);
    rmin = Array<float>({M, 1, T}, af_gpu | af_zero);
}


GpuSplineDetrender::~GpuSplineDetrender()
{
    delete reinterpret_cast<TimeStencils *>(tb_blob);
    tb_blob = nullptr;
}


vector<long> GpuSplineDetrender::bin_edges() const
{
    return _bin_edges;
}


void GpuSplineDetrender::launch(Array<float> &intensity, const Array<float> &weights, cudaStream_t stream) const
{
    xassert_shape_eq(intensity, ({M, nfreq, T}));
    xassert_shape_eq(weights, ({M, nfreq, T}));
    xassert(intensity.is_fully_contiguous());
    xassert(weights.is_fully_contiguous());
    xassert(intensity.on_gpu());
    xassert(weights.on_gpu());
    xassert(intensity.data != weights.data);

    const TimeStencils &tb = *reinterpret_cast<const TimeStencils *>(tb_blob);

    constexpr int NPHI = 3;
    const int NB = bandwidth(NPHI, 0);
    const int nbuf = int(T);                        // no time window, so no padding
    const int S = int(solve_threads);
    const int phi_stride = 4, prod_stride = 12;

    // Kernel 1: (intensity, weights) -> per-freq-range Gram matrices and data moments.
    {
        dim3 nblocks((nbuf + PASS_THREADS - 1)/PASS_THREADS, int(nfrange), int(M));
        detrend_2d_accum_kernel<NPHI, WeightedInput> <<< nblocks, PASS_THREADS, 0, stream >>>
            (intensity.data, weights.data, gu.data, phi_tab.data, prod_tab.data, fr_desc.data,
             int(nfreq), int(nfrange), nbuf, phi_stride, prod_stride);
        CUDA_PEEK("detrend_2d_accum_kernel<WeightedInput>");
    }

    // Kernel 2: assemble, regularize, solve. The kernel multiplies reg_strength by the
    // sample's total weight, so reg_strength carries the 1/nbins of the old code's
    // epsilon * wsum / nbins. eps = 0 disables the mask expansion the kernel can do:
    // no zone is ever flagged, and 'rmin' is written but not read.
    {
        const long nblk_max = N_phi;
        const long ncompz_max = N_phi * (NPHI + 2);
        const long shmem = solve_shmem_bytes(nblk_max, NB, ncompz_max, S, /*W=*/0, /*scaled=*/true);

        dim3 nblocks(int(T/S), 1, int(M));
        detrend_2d_solve_kernel<NPHI, WeightScaledStrength> <<< nblocks, S, size_t(shmem), stream >>>
            (gu.data, acoef.data, rmin.data, zone_desc.data, fr_desc.data,
             reg_tab.data, unit_coef.data, /*nreg_bands=*/4,
             int(nfrange), /*nzone=*/1, int(N_phi), nbuf, int(nblk_max),
             /*n_deg=*/0, /*W=*/0, int(T), /*reg_strength=*/float(epsilon / double(nbins)), /*eps=*/0.0f, tb);
        CUDA_PEEK("detrend_2d_solve_kernel<WeightScaledStrength>");
    }

    // Kernel 3: evaluate the spline and subtract it at every channel.
    {
        dim3 nblocks((nbuf + PASS_THREADS - 1)/PASS_THREADS, int(nfrange), int(M));
        detrend_2d_subtract_kernel<NPHI, WeightedInput> <<< nblocks, PASS_THREADS, 0, stream >>>
            (intensity.data, weights.data, acoef.data, rmin.data, phi_tab.data, fr_desc.data,
             int(nfreq), int(N_phi), /*nzone=*/1, nbuf, /*W=*/0, int(T), phi_stride, /*eps=*/0.0f);
        CUDA_PEEK("detrend_2d_subtract_kernel<WeightedInput>");
    }
}


// -------------------------------------------------------------------------------------------------


void GpuSplineDetrender::time_selected()
{
    // The two shapes the production RFI chain runs this detrender at (nbins = 6,
    // epsilon = 3e-4 in both): the 16x-downsampled sub-pipelines, and the full band.
    // 8 beams and 1024-sample chunks, as the other chimefrb timing functions use.
    struct TimingConfig { long nfreq; long nbins; const char *what; };
    const vector<TimingConfig> configs = {
        { 1024,  6, "sub-pipeline instances (16x downsampled, count-valued weights)" },
        { 16384, 6, "top-level instance (full resolution, {0,1} weights)" },
    };
    const long M = 8, T = 1024;
    const double epsilon = 3.0e-4;
    const int niter = 20;

    for (const TimingConfig &c: configs) {
        GpuSplineDetrender det(c.nfreq, c.nbins, epsilon, M, T);

        // Random intensity and unit weights. The kernels are branch-free and their work
        // is weight-independent, so the timing does not depend on the data; unit weights
        // time the ordinary path.
        Array<float> intensity({M, c.nfreq, T}, af_gpu);
        Array<float> weights({M, c.nfreq, T}, af_gpu);
        {
            Array<float> hi({M, c.nfreq, T}, af_uhost);
            Array<float> hw({M, c.nfreq, T}, af_uhost);
            hi.randomize();
            for (long j = 0; j < M*c.nfreq*T; j++)
                hw.data[j] = 1.0f;
            intensity.fill(hi);
            weights.fill(hw);
        }

        // Global memory traffic of an ideal implementation: kernel 1 reads intensity and
        // weights (8 bytes per sample), kernel 3 reads both and writes the intensity (12).
        // Everything else is per-freq-range or per-coefficient and negligible.
        const double nbytes = double(M) * double(c.nfreq) * double(T) * 20.0;

        cout << "\nGpuSplineDetrender::time_selected()\n"
             << "    (nfreq, nbins, epsilon) = (" << c.nfreq << ", " << c.nbins << ", " << epsilon
             << "):  " << c.what << "\n"
             << "    M = " << M << ", T = " << T << ", N_phi = " << det.N_phi
             << ", nfrange = " << det.nfrange << ", channels_per_range = " << det.channels_per_range
             << ", solve_threads = " << det.solve_threads << "\n"
             << "    global memory traffic per launch = " << (nbytes / 1.0e9) << " GB"
             << endl;

        KernelTimer kt(niter, 1);
        double dt = 0.0;
        while (kt.next()) {
            det.launch(intensity, weights, kt.stream);
            if (kt.warmed_up)
                dt = kt.dt;
        }

        cout << "    dt = " << (dt * 1.0e3) << " ms"
             << ",  bandwidth = " << (nbytes / dt / 1.0e9) << " GB/s" << endl;
    }
}


}}  // namespace pirate::chimefrb
