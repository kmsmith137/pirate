#include "../include/pirate/Detrender.hpp"
#include "../include/pirate/detrender_kernels.hpp"

#include <cmath>
#include <sstream>
#include <iostream>
#include <algorithm>
#include <yaml-cpp/yaml.h>
#include <ksgpu/xassert.hpp>
#include <ksgpu/cuda_utils.hpp>
#include <ksgpu/KernelTimer.hpp>

#include "../include/pirate/YamlFile.hpp"

using namespace std;
using namespace ksgpu;

namespace pirate {
#if 0
}  // editor auto-indent
#endif


// The three kernels, the launch-geometry helpers and the time-stencil struct live in
// include/pirate/detrender_kernels.hpp, together with the overview of the algorithm's
// structure (three kernels, freq-ranges, zones, chunk invariance, NaN safety). This file
// is the host side of GpuDetrenderLps2d: parameter validation and yaml, the knot-vector
// machinery and basis tables, the D_1 regulator table, and the launch dispatch.


// Largest TIME polynomial degree we are willing to RUN, as opposed to the largest the
// kernel's arrays are sized for (MAX_NDEG in detrender_kernels.hpp). The two differ on
// purpose. The numpy reference (pirate_frb/detrending/lps2d) rejects n > 2, so n = 3 would
// be a configuration no test can validate -- and the parts of the kernel that n touches
// are exactly the parts a test would need to check: the (q,r) assembly loops, the moment
// stencils, and the Theta = I structure of the regulator. Refusing it is better than
// shipping an unvalidated path. Raising it is a two-step change, reference first, then
// here.
//
// This is specific to n. The SPLINE degree n_phi = 3 is fully supported and tested on
// both sides.
static constexpr int MAX_ORACLE_NDEG = 2;


// The compiled configurations. To add one: add a row here, and a line to the dispatch in
// launch(). T is deliberately NOT part of a configuration -- it is a runtime kernel
// argument (see detrender_kernels.hpp), so a caller may pick any chunk length without
// a recompile, and test_gpu_kernel() uses a small one to keep its numpy oracle cheap.
struct GpuDetrenderLps2dConfig { long n_phi; };

static constexpr GpuDetrenderLps2dConfig detrender_2d_configs[] = {
    { 0 },
    { 1 },
    { 2 },
    { 3 },
};


static string config_list_str()
{
    stringstream ss;
    for (const GpuDetrenderLps2dConfig &c: detrender_2d_configs)
        ss << ((&c == &detrender_2d_configs[0]) ? "" : ", ")
           << "(n_phi=" << c.n_phi << ")";
    return ss.str();
}


vector<long> GpuDetrenderLps2d::configs()
{
    vector<long> ret;
    for (const GpuDetrenderLps2dConfig &c: detrender_2d_configs)
        ret.push_back(c.n_phi);
    return ret;
}


// Evaluate the n_phi+1 nonzero B-splines at x, whose knot span is j0. This is the
// standard triangular form of the Cox-de Boor recursion (the NURBS book's Algorithm
// A2.2), which has no zero denominators at all provided the span is non-empty -- which
// is why there is no "drop the term with a vanishing denominator" special case here even
// though repeated knots are fully supported.
static void eval_basis(const vector<long> &knots, long n_phi, double x, long j0, double *out)
{
    vector<double> left(n_phi+1, 0.0), right(n_phi+1, 0.0);
    out[0] = 1.0;
    for (long p = 1; p <= n_phi; p++)
        out[p] = 0.0;

    for (long p = 1; p <= n_phi; p++) {
        left[p]  = x - double(knots[j0 + 1 - p]);
        right[p] = double(knots[j0 + p]) - x;
        double saved = 0.0;
        for (long r = 0; r < p; r++) {
            // The denominator straddles the span [knots[j0], knots[j0+1]) and is
            // therefore strictly positive.
            const double temp = out[r] / (right[r+1] + left[p-r]);
            out[r] = saved + right[r+1]*temp;
            saved = left[p-r]*temp;
        }
        out[p] = saved;
    }
}


// The orthonormal polynomial basis on the window, by modified Gram-Schmidt on the
// monomials s^q, s = -W..W.
//
// ORTHONORMAL, NOT MONOMIAL, and this is the one choice here that moves a threshold
// rather than a constant. For a window-constant mask the assembled matrix is exactly
// (G + eta D_1) kron Theta with Theta_qr = sum_s p_q p_r, and equilibration of a
// Kronecker product is Kronecker, so the pivots multiply: r_min(2d) = r_min(1d) *
// r_min(Theta). With raw monomials r_min(Theta) is 0.18 to 0.25 at n = 2, which would
// multiply the 1-d conditioning margin by that factor -- enough to push the worst
// adversarial mask below eps and expand zones whose fits are perfectly accurate. With
// p_q orthonormal, Theta = I and r_min(2d) = r_min(1d) exactly, and it costs nothing:
// the basis enters only through stencil coefficients that are precomputed either way.
//
// Gram-Schmidt on a symmetric grid preserves parity, because <s^i, s^j> = 0 whenever i+j
// is odd. moments.py's window folding depends on that being exact.
static void build_time_basis(long n, long W, vector<double> &P)
{
    const long nk = 2*W + 1;
    P.assign(nk*(n+1), 0.0);

    for (long k = 0; k < nk; k++) {
        const double s = double(k - W);
        double v = 1.0;
        for (long q = 0; q <= n; q++) {
            P[k*(n+1) + q] = v;
            v *= s;
        }
    }

    for (long q = 0; q <= n; q++) {
        for (long r = 0; r < q; r++) {
            double dot = 0.0;
            for (long k = 0; k < nk; k++)
                dot += P[k*(n+1) + r] * P[k*(n+1) + q];
            for (long k = 0; k < nk; k++)
                P[k*(n+1) + q] -= dot * P[k*(n+1) + r];
        }
        double nrm = 0.0;
        for (long k = 0; k < nk; k++)
            nrm += P[k*(n+1) + q] * P[k*(n+1) + q];
        nrm = sqrt(nrm);
        // Positive normalization fixes the sign convention (numpy's QR needs an explicit
        // sign(diag(R)) correction to get the same basis).
        for (long k = 0; k < nk; k++)
            P[k*(n+1) + q] /= nrm;
    }
}


static void fill_stencils(TimeStencils &tb, const vector<double> &P, int n_deg, int W)
{
    const int NPAIR_T = (n_deg+1)*(n_deg+2)/2;
    const int nq = n_deg + 1;

    double parity[MAX_NDEG+1];
    for (int q = 0; q <= n_deg; q++)
        parity[q] = (q % 2 == 0) ? 1.0 : -1.0;

    int p = 0;
    for (int q = 0; q <= n_deg; q++) {
        for (int r = q; r <= n_deg; r++, p++) {
            for (int k = 0; k <= W; k++)
                tb.gs[p][k] = float(P[(W+k)*nq + q] * P[(W+k)*nq + r]);
            tb.gpar[p] = float(parity[q] * parity[r]);
        }
    }
    xassert_eq(p, NPAIR_T);

    for (int q = 0; q <= n_deg; q++) {
        for (int k = 0; k <= W; k++)
            tb.us[q][k] = float(P[(W+k)*nq + q]);
        tb.upar[q] = float(parity[q]);
        tb.eval0[q] = float(P[W*nq + q]);
    }
}


// Declared in detrender_kernels.hpp.
TimeStencils make_time_stencils(long n, long W)
{
    vector<double> P;
    build_time_basis(n, W, P);
    TimeStencils tb{};
    fill_stencils(tb, P, int(n), int(W));
    return tb;
}


void DetrenderLps2dParams::validate() const
{
    bool found = false;
    for (const GpuDetrenderLps2dConfig &c: detrender_2d_configs)
        if (c.n_phi == n_phi)
            found = true;

    if (!found) {
        stringstream ss;
        ss << "DetrenderLps2dParams: no kernel is compiled for n_phi=" << n_phi
           << "; available configurations are " << config_list_str();
        throw runtime_error(ss.str());
    }

    // n is runtime. The bound is what the reference can validate, not what the kernel
    // could execute; see MAX_ORACLE_NDEG.
    if ((n < 0) || (n > MAX_ORACLE_NDEG)) {
        stringstream ss;
        ss << "DetrenderLps2dParams: n=" << n << " must be in [0, " << MAX_ORACLE_NDEG << "]. "
           << "(The kernel's arrays are sized up to n = " << MAX_NDEG << ", but the numpy "
           << "reference pirate_frb.detrending.ReferenceDetrenderLps2d rejects n > "
           << MAX_ORACLE_NDEG << ", so a larger n would be a configuration that no test "
           << "validates. To raise this, extend the reference first, then this check. "
           << "Note this is the TIME polynomial degree; the spline degree n_phi = 3 is "
           << "supported.)";
        throw runtime_error(ss.str());
    }

    // W is runtime.  2W+1 >= n+1 is the algebraic minimum -- below it the time fit is
    // underdetermined before any masking -- and MAX_W exists only to give the by-value
    // stencil struct a compile-time size.
    if ((W < 0) || (W > MAX_W)) {
        stringstream ss;
        ss << "DetrenderLps2dParams: W=" << W << " must be in [0, " << MAX_W << "]";
        throw runtime_error(ss.str());
    }
    if (2*W + 1 < n + 1) {
        stringstream ss;
        ss << "DetrenderLps2dParams: a degree-" << n << " fit in time needs 2W+1 >= n+1, but W="
           << W << " gives a " << (2*W+1) << "-sample window";
        throw runtime_error(ss.str());
    }

    // T is runtime, but kernel 2's grid is T/solve_threads blocks and solve_threads is
    // chosen from {256,128,64,32}, so a multiple of 32 guarantees one of them divides T.
    // Requiring it keeps that kernel free of a predicated tail; every realistic chunk
    // length satisfies it.
    if ((T <= 0) || (T % 32 != 0)) {
        stringstream ss;
        ss << "DetrenderLps2dParams: T=" << T << " must be a positive multiple of 32";
        throw runtime_error(ss.str());
    }

    if (nfreq < 1)
        throw runtime_error("DetrenderLps2dParams: nfreq must be >= 1");
    if (M < 1)
        throw runtime_error("DetrenderLps2dParams: M must be >= 1");
    if (eta <= 0.0)
        throw runtime_error("DetrenderLps2dParams: eta must be > 0");
    if (eps <= 0.0)
        throw runtime_error("DetrenderLps2dParams: eps must be > 0");

    // ---- Validate the knot vector.
    //
    // Strict, because the array comes from the caller. The end-multiplicity rule is the
    // one that is not merely stylistic: clamped ends are what put the constant function
    // in the span and make the basis a partition of unity on each zone, which is in turn
    // what makes the regulator's null space exactly the constants -- hence what makes a
    // constant baseline removable EXACTLY rather than shrunk. Reducing it does not
    // degrade gracefully, it destroys the property (at end multiplicity n_phi the best
    // fit to the constant 1 is off by 0.99).
    const vector<long> &kn = knots;
    if (long(kn.size()) < n_phi + 2)
        throw runtime_error("DetrenderLps2dParams: knot vector is too short");
    for (size_t i = 1; i < kn.size(); i++)
        if (kn[i] < kn[i-1])
            throw runtime_error("DetrenderLps2dParams: knots must be non-decreasing");
    if ((kn.front() != 0) || (kn.back() != nfreq)) {
        stringstream ss;
        ss << "DetrenderLps2dParams: knots must run from 0 to nfreq=" << nfreq
           << ", got [" << kn.front() << ", " << kn.back() << "]";
        throw runtime_error(ss.str());
    }
    for (int which = 0; which < 2; which++) {
        const long val = which ? nfreq : 0;
        long mult = 0;
        for (long v: kn)
            if (v == val)
                mult++;
        if (mult != n_phi + 1) {
            stringstream ss;
            ss << "DetrenderLps2dParams: the " << (which ? "last" : "first") << " knot (" << val
               << ") has multiplicity " << mult << ", expected exactly n_phi+1 = "
               << (n_phi+1) << ". Clamped ends are what put the constant function in the"
               << " span and make the basis complete on the whole band.";
            throw runtime_error(ss.str());
        }
    }

    if (long(kn.size()) - n_phi - 1 < 1)
        throw runtime_error("DetrenderLps2dParams: N_phi = len(knots)-n_phi-1 must be >= 1");

    // Interior multiplicities. A multiplicity of exactly n_phi+1 is a zone boundary and is
    // allowed (see the constructor); above that, the basis would be discontinuous.
    for (long i = 0; i < long(kn.size()); ) {
        long j = i;
        while ((j < long(kn.size())) && (kn[j] == kn[i]))
            j++;
        if ((kn[i] > 0) && (kn[i] < nfreq) && (j - i > n_phi + 1)) {
            stringstream ss;
            ss << "DetrenderLps2dParams: interior knot " << kn[i] << " has multiplicity "
               << (j - i) << ", above n_phi+1 = " << (n_phi+1);
            throw runtime_error(ss.str());
        }
        i = j;
    }
}


// -------------------------------------------------------------------------------------------------
//
// Yaml I/O.
//
// The yaml keys are spelled out, rather than matching the (terse) member names. The mapping
// is defined here and nowhere else -- if you add a member, it needs a row in both functions
// below, and the key should read as English.


void DetrenderLps2dParams::to_yaml(YAML::Emitter &emitter, bool verbose) const
{
    this->validate();

    emitter << YAML::BeginMap;

    if (verbose) {
        stringstream ss;
        ss << "DetrenderLps2dParams: the parameters of a 2-d spline detrender, which fits a\n";
        ss << "B-spline in frequency times a local polynomial in time and subtracts it. See\n";
        ss << "the class comments in Detrender.hpp, and notes/detrending.tex section\n";
        ss << "\"2-d detrending\".";
        emitter << YAML::Comment(ss.str()) << YAML::Newline << YAML::Newline;
    }

    emitter << YAML::Key << "nfreq" << YAML::Value << nfreq;
    emitter << YAML::Key << "num_beams" << YAML::Value << M;

    if (verbose) {
        stringstream ss;
        ss << "The fit. The window is (2*time_halfwidth + 1) samples, and only its middle\n";
        ss << "time_samples_per_chunk samples are written; the caller owns the 2*time_halfwidth\n";
        ss << "padding. Only spline_degree_freq is a compile-time property of the cuda kernel:\n";
        ss << "the compiled values are " << config_list_str() << ".";
        emitter << YAML::Newline << YAML::Comment(ss.str()) << YAML::Newline;
    }

    emitter << YAML::Key << "spline_degree_freq" << YAML::Value << n_phi;
    emitter << YAML::Key << "poly_degree_time" << YAML::Value << n;
    emitter << YAML::Key << "time_halfwidth" << YAML::Value << W;
    emitter << YAML::Key << "time_samples_per_chunk" << YAML::Value << T;

    if (verbose) {
        stringstream ss;
        ss << "Tuning parameters; both are optional and default to the values below.\n";
        ss << "  regularization_strength: first-difference regulator on the frequency\n";
        ss << "    coefficients. A baseline that is constant in frequency within a zone is\n";
        ss << "    removed EXACTLY at any value.\n";
        ss << "  conditioning_threshold: a zone whose conditioning statistic r_min falls\n";
        ss << "    below this has all of its channels dropped for that time sample.";
        emitter << YAML::Newline << YAML::Comment(ss.str()) << YAML::Newline;
    }

    emitter << YAML::Key << "regularization_strength" << YAML::Value << eta;
    emitter << YAML::Key << "conditioning_threshold" << YAML::Value << eps;

    if (verbose) {
        // Derived quantities, so a reader can sanity-check a knot vector without doing the
        // arithmetic. Recomputed here rather than taken from a GpuDetrenderLps2d, since a DetrenderLps2dParams
        // can be written without ever constructing one.
        long nzone = 1;
        for (long i = 0; i < long(knots.size()); ) {
            long j = i;
            while ((j < long(knots.size())) && (knots[j] == knots[i]))
                j++;
            if ((knots[i] > 0) && (knots[i] < nfreq) && (j - i == n_phi + 1))
                nzone++;
            i = j;
        }

        stringstream ss;
        ss << "Knot vector: a non-decreasing list of channel indices, running from 0 to\n";
        ss << "nfreq, with the first and last values repeated exactly spline_degree_freq+1\n";
        ss << "times. An interior value repeated that many times is a zone boundary, and\n";
        ss << "zones decouple the fit exactly.\n";
        ss << "In this file: " << (long(knots.size()) - n_phi - 1) << " basis functions, "
           << nzone << " zone" << ((nzone != 1) ? "s" : "") << ".";
        emitter << YAML::Newline << YAML::Comment(ss.str()) << YAML::Newline;
    }

    emitter << YAML::Key << "knots" << YAML::Value << YAML::Flow << YAML::BeginSeq;
    for (long k: knots)
        emitter << k;
    emitter << YAML::EndSeq;

    emitter << YAML::EndMap;
}


string DetrenderLps2dParams::to_yaml_string(bool verbose) const
{
    YAML::Emitter emitter;
    this->to_yaml(emitter, verbose);
    return emitter.c_str();
}


// static member function
DetrenderLps2dParams DetrenderLps2dParams::from_yaml(const string &filename)
{
    YamlFile f = YamlFile::from_file(filename);
    return DetrenderLps2dParams::from_yaml(f);
}


// static member function
DetrenderLps2dParams DetrenderLps2dParams::from_yaml_string(const string &yaml_string)
{
    YamlFile f = YamlFile::from_string(yaml_string, "<detrender params string>");
    return DetrenderLps2dParams::from_yaml(f);
}


// static member function
DetrenderLps2dParams DetrenderLps2dParams::from_yaml(const YamlFile &f)
{
    // 'channels_per_range' was a constructor argument before it became a derived member.
    // A file carrying it was written against an interface where it could be requested, so
    // silently ignoring it would silently change the caller's meaning.
    if (f.has_key("channels_per_range")) {
        stringstream ss;
        ss << f.name << ": key 'channels_per_range' is no longer part of GpuDetrenderLps2d's"
           << " interface -- it is always derived from (nfreq, knots,"
           << " time_samples_per_chunk). Remove the key.";
        throw runtime_error(ss.str());
    }

    DetrenderLps2dParams p;
    p.nfreq = f.get_scalar<long> ("nfreq");
    p.M = f.get_scalar<long> ("num_beams");
    p.n_phi = f.get_scalar<long> ("spline_degree_freq");
    p.n = f.get_scalar<long> ("poly_degree_time");
    p.W = f.get_scalar<long> ("time_halfwidth");
    p.T = f.get_scalar<long> ("time_samples_per_chunk");

    // Tuning parameters: optional, defaulting to the member initializers.
    p.eta = f.get_scalar<double> ("regularization_strength", DetrenderLps2dParams().eta);
    p.eps = f.get_scalar<double> ("conditioning_threshold", DetrenderLps2dParams().eps);

    p.knots = f.get_vector<long> ("knots");

    f.check_for_invalid_keys();
    p.validate();
    return p;
}


// -------------------------------------------------------------------------------------------------


// One line per compiled n_phi; see detrender_2d_configs[] above.
static long _choose_solve_threads(long n_phi, long T, long nblk_max, long NB, long ncompz_max, long W)
{
    switch (n_phi) {
        case 0: return choose_solve_threads<0, FixedStrength>(T, nblk_max, NB, ncompz_max, W);
        case 1: return choose_solve_threads<1, FixedStrength>(T, nblk_max, NB, ncompz_max, W);
        case 2: return choose_solve_threads<2, FixedStrength>(T, nblk_max, NB, ncompz_max, W);
        case 3: return choose_solve_threads<3, FixedStrength>(T, nblk_max, NB, ncompz_max, W);
    }
    throw runtime_error("GpuDetrenderLps2d: internal error in _choose_solve_threads()");
}


GpuDetrenderLps2d::GpuDetrenderLps2d(const DetrenderLps2dParams &params_) :
    params(params_), nbuf(params_.T + 2*params_.W)
{
    params.validate();

    // Local aliases for the params the constructor uses repeatedly.
    const long nfreq = params.nfreq;
    const long M = params.M;
    const long n_phi = params.n_phi;
    const long n = params.n;
    const long W = params.W;
    const long T = params.T;
    const vector<long> &kn = params.knots;

    const long nk = long(kn.size());
    N_phi = nk - n_phi - 1;

    // Zone boundaries: an interior knot of multiplicity exactly n_phi+1. No basis function
    // straddles one -- phi_j has support [k_j, k_{j+n_phi+1}), and if the boundary occupies
    // knot indices i..i+n_phi then j <= i-1 gives supp_hi <= v and j >= i gives supp_lo >= v
    // -- so G and D_1 are exactly block diagonal there and the fits on the two sides
    // decouple. (Larger multiplicities are rejected by validate().)
    vector<long> bounds;
    for (long i = 0; i < nk; ) {
        long j = i;
        while ((j < nk) && (kn[j] == kn[i]))
            j++;
        if ((kn[i] > 0) && (kn[i] < nfreq) && (j - i == n_phi + 1))
            bounds.push_back(kn[i]);
        i = j;
    }
    nzone = long(bounds.size()) + 1;

    // Span index of each channel: the largest j with knots[j] <= f, which lands on the
    // last knot of a repeated group and hence always on a NON-EMPTY span. That is what
    // the Cox-de Boor recursion needs. Channel f occupies [f, f+1) and its data sits at
    // f + 1/2, so no data point ever coincides with a knot.
    vector<long> j0(nfreq);
    {
        long j = 0;
        for (long f = 0; f < nfreq; f++) {
            while ((j+1 < nk) && (kn[j+1] <= f))
                j++;
            j0[f] = j;
        }
    }

    // The zone of phi_j is decided by its supp_lo = knots[j] alone (see above).
    vector<long> zone_of_coef(N_phi);
    for (long j = 0; j < N_phi; j++) {
        long z = 0;
        while ((z < long(bounds.size())) && (bounds[z] <= kn[j]))
            z++;
        zone_of_coef[j] = z;
    }

    // ---- Freq-range width, derived from the instance size (see derive_channels_per_range()).
    channels_per_range = derive_channels_per_range(nfreq, nbuf);

    // ---- Basis tables, built in float64 and cast, so that the working dtype affects the
    // arithmetic that uses the basis but not the basis itself.

    const long npair_f = (n_phi+1)*(n_phi+2)/2;
    phi_stride  = 4*((n_phi + 1 + 3) / 4);
    prod_stride = 4*((npair_f + 3) / 4);

    Array<float> phi_host({nfreq, phi_stride}, af_uhost | af_zero);
    Array<float> prod_host({nfreq, prod_stride}, af_uhost | af_zero);
    vector<double> nb(n_phi+1);

    for (long f = 0; f < nfreq; f++) {
        eval_basis(kn, n_phi, double(f) + 0.5, j0[f], &nb[0]);
        for (long a = 0; a <= n_phi; a++)
            phi_host.data[f*phi_stride + a] = float(nb[a]);
        long p = 0;
        for (long a = 0; a <= n_phi; a++)
            for (long b = a; b <= n_phi; b++, p++)
                prod_host.data[f*prod_stride + p] = float(nb[a] * nb[b]);
    }

    phi_tab = phi_host.to_gpu();
    prod_tab = prod_host.to_gpu();

    // ---- Freq-ranges: split each non-empty knot interval into pieces of about
    // channels_per_range channels. A freq-range never crosses a knot, so j0 is fixed on
    // it, and because a zone boundary IS a knot it never crosses a zone either.
    vector<long> fr_lo, fr_hi, fr_j0, fr_zone;
    for (long f = 0; f < nfreq; ) {
        long g = f;
        while ((g < nfreq) && (j0[g] == j0[f]))
            g++;
        const long len = g - f;
        long k = (len + channels_per_range/2) / channels_per_range;
        if (k < 1)
            k = 1;
        for (long i = 0; i < k; i++) {
            const long lo = f + (len*i)/k;
            const long hi = f + (len*(i+1))/k;
            if (hi <= lo)
                continue;
            fr_lo.push_back(lo);
            fr_hi.push_back(hi);
            fr_j0.push_back(j0[f]);
            fr_zone.push_back(zone_of_coef[j0[f]]);
        }
        f = g;
    }
    nfrange = long(fr_lo.size());
    xassert_gt(nfrange, 0);

    Array<int> fr_host({nfrange, 4}, af_uhost | af_zero);
    for (long i = 0; i < nfrange; i++) {
        fr_host.data[4*i + 0] = int(fr_lo[i]);
        fr_host.data[4*i + 1] = int(fr_hi[i]);
        fr_host.data[4*i + 2] = int(fr_j0[i]);
        fr_host.data[4*i + 3] = int(fr_zone[i]);
    }
    fr_desc = fr_host.to_gpu();

    // ---- Zone descriptors. Zones are contiguous in both the coefficient index and the
    // channel index, so a zone's freq-ranges are a contiguous run.
    Array<int> zone_host({nzone, 4}, af_uhost | af_zero);
    nphi_zone_max = 0;
    for (long z = 0; z < nzone; z++) {
        long clo = -1, chi = -1;
        for (long j = 0; j < N_phi; j++) {
            if (zone_of_coef[j] != z)
                continue;
            if (clo < 0)
                clo = j;
            chi = j + 1;
        }
        // Every zone spans a non-empty channel range (its boundaries are distinct channel
        // indices), so it always holds at least one coefficient and one freq-range.
        xassert_ge(clo, 0);

        long flo = -1, fhi = -1;
        for (long i = 0; i < nfrange; i++) {
            if (fr_zone[i] != z)
                continue;
            if (flo < 0)
                flo = i;
            fhi = i + 1;
        }
        xassert_ge(flo, 0);

        zone_host.data[4*z + 0] = int(flo);
        zone_host.data[4*z + 1] = int(fhi);
        zone_host.data[4*z + 2] = int(clo);
        zone_host.data[4*z + 3] = int(chi - clo);
        nphi_zone_max = max(nphi_zone_max, chi - clo);
    }
    zone_desc = zone_host.to_gpu();

    // ---- The regulator table: D_1, the per-zone first-difference penalty, in the banded
    // layout kernel 2 reads (see detrender_kernels.hpp); the kernel multiplies it by eta.
    //
    // D_1's null space is the zone's all-ones vector, which (because the basis is a
    // partition of unity on each zone) IS the constant function, so a baseline constant
    // in frequency is removed exactly at any eta. It is assembled per zone, never across
    // a boundary: a difference penalty spanning two zones would couple them with weight 1
    // and drop the null space to a single global constant. Only bands 0 and 1 are
    // nonzero, hence nreg_bands = 2 in _launch().
    {
        const long nreg = reg_table_width(int(n_phi));
        Array<float> reg_host({N_phi, nreg}, af_uhost | af_zero);
        for (long z = 0; z < nzone; z++) {
            const long clo = zone_host.data[4*z + 2];
            const long nphi_z = zone_host.data[4*z + 3];
            for (long j = 0; j < nphi_z; j++) {
                const float d0 = (nphi_z == 1) ? 0.0f : (((j == 0) || (j == nphi_z-1)) ? 1.0f : 2.0f);
                reg_host.data[(clo+j)*nreg + 0] = d0;
                if (j+1 < nphi_z)
                    reg_host.data[(clo+j)*nreg + 1] = -1.0f;
            }
        }
        reg_tab = reg_host.to_gpu();
    }

    // ---- The time basis (a function of (n, W) only, so this runs once and never changes).
    tb_blob = new TimeStencils(make_time_stencils(n, W));

    // ---- Kernel-2 block size, from the shared-memory budget (see choose_solve_threads()
    // in detrender_kernels.hpp, which also raises this instantiation's dynamic
    // shared-memory limit).
    const long NB = bandwidth(int(n_phi), int(n));
    const long nblk_max = nphi_zone_max * (n+1);
    const long ncompz_max = nphi_zone_max * (n_phi+2);

    solve_threads = _choose_solve_threads(n_phi, T, nblk_max, NB, ncompz_max, W);
    if (solve_threads == 0) {
        stringstream ss;
        ss << "GpuDetrenderLps2d: the largest zone has " << nphi_zone_max << " basis functions,"
           << " which needs more shared memory than this GPU offers even at 8 threads per"
           << " block. Use more zone boundaries, i.e. interior knots of multiplicity"
           << " n_phi+1, to split the frequency band.";
        throw runtime_error(ss.str());
    }

    // ---- Per-launch scratch.
    const long ncomp = npair_f + n_phi + 1;
    gu = Array<float>({M, nfrange, ncomp, nbuf}, af_gpu | af_zero);
    acoef = Array<float>({M, N_phi, T}, af_gpu | af_zero);
    rmin = Array<float>({M, nzone, T}, af_gpu | af_zero);
}


GpuDetrenderLps2d::~GpuDetrenderLps2d()
{
    delete reinterpret_cast<TimeStencils *>(tb_blob);
    tb_blob = nullptr;
}


template<int NPHI>
static void _launch(const GpuDetrenderLps2d &d, float *data, unsigned char *mask,
                    float *gu, float *acoef, float *rmin,
                    const float *phi_tab, const float *prod_tab, const float *reg_tab,
                    const int *fr_desc, const int *zone_desc,
                    long nphi_zone_max, long solve_threads,
                    long phi_stride, long prod_stride, const void *tb_blob,
                    cudaStream_t stream)
{
    const int NB = bandwidth(NPHI, int(d.params.n));
    const TimeStencils &tb = *reinterpret_cast<const TimeStencils *>(tb_blob);

    const int nbuf = int(d.nbuf);
    const int n_deg = int(d.params.n);
    const int W = int(d.params.W);
    const int T = int(d.params.T);
    const int M = int(d.params.M);
    const int S = int(solve_threads);

    // Kernel 1.
    {
        dim3 nblocks((nbuf + PASS_THREADS - 1)/PASS_THREADS, int(d.nfrange), M);
        detrend_2d_accum_kernel<NPHI, MaskedInput> <<< nblocks, PASS_THREADS, 0, stream >>>
            (data, mask, gu, phi_tab, prod_tab, fr_desc,
             int(d.params.nfreq), int(d.nfrange), nbuf, int(phi_stride), int(prod_stride));
        CUDA_PEEK("detrend_2d_accum_kernel");
    }

    // Kernel 2. The shared-memory request usually exceeds the 48 KB default, so opt in.
    // Done once per (configuration, process) rather than per launch.
    {
        const long nblk_max = long(nphi_zone_max)*(long(n_deg)+1);
        const long ncompz_max = long(nphi_zone_max)*(NPHI+2);
        const long shmem = solve_shmem_bytes(nblk_max, NB, ncompz_max, S, W, /*scaled=*/false);

        // No cudaFuncSetAttribute here: the constructor already raised the limit to the
        // device maximum, which is idempotent across instances and immune to the ordering
        // hazard a per-instance value has (an instance with a small zone must never lower
        // a limit an earlier instance with a big zone depends on).

        dim3 nblocks(int(T/S), int(d.nzone), M);
        detrend_2d_solve_kernel<NPHI, FixedStrength> <<< nblocks, S, size_t(shmem), stream >>>
            (gu, acoef, rmin, zone_desc, fr_desc, reg_tab, /*unit_coef=*/nullptr, /*nreg_bands=*/2,
             int(d.nfrange), int(d.nzone), int(d.N_phi), nbuf, int(nphi_zone_max),
             n_deg, W, T, float(d.params.eta), float(d.params.eps), tb);
        CUDA_PEEK("detrend_2d_solve_kernel");
    }

    // Kernel 3.
    {
        dim3 nblocks((nbuf + PASS_THREADS - 1)/PASS_THREADS, int(d.nfrange), M);
        detrend_2d_subtract_kernel<NPHI, MaskedInput> <<< nblocks, PASS_THREADS, 0, stream >>>
            (data, mask, acoef, rmin, phi_tab, fr_desc,
             int(d.params.nfreq), int(d.N_phi), int(d.nzone), nbuf, W, T, int(phi_stride),
             float(d.params.eps));
        CUDA_PEEK("detrend_2d_subtract_kernel");
    }
}


void GpuDetrenderLps2d::launch(Array<float> &data, Array<unsigned char> &mask, cudaStream_t stream) const
{
    xassert_eq(data.ndim, 3);
    xassert_shape_eq(data, ({params.M, params.nfreq, nbuf}));
    xassert_shape_eq(mask, ({params.M, params.nfreq, nbuf}));
    xassert(data.is_fully_contiguous());
    xassert(mask.is_fully_contiguous());
    xassert(data.on_gpu());
    xassert(mask.on_gpu());

    // One line per compiled n_phi; see detrender_2d_configs[] above.
    #define _DT2D_DISPATCH(P)                                                          \
        _launch<P> (*this, data.data, mask.data, gu.data, acoef.data, rmin.data,        \
                    phi_tab.data, prod_tab.data, reg_tab.data, fr_desc.data, zone_desc.data, \
                    nphi_zone_max, solve_threads, phi_stride, prod_stride, tb_blob, stream)

    if (params.n_phi == 0)
        _DT2D_DISPATCH(0);
    else if (params.n_phi == 1)
        _DT2D_DISPATCH(1);
    else if (params.n_phi == 2)
        _DT2D_DISPATCH(2);
    else if (params.n_phi == 3)
        _DT2D_DISPATCH(3);
    else
        throw runtime_error("GpuDetrenderLps2d::launch: internal error, unhandled configuration");

    #undef _DT2D_DISPATCH
}


void GpuDetrenderLps2d::time_selected()
{
    // The timing configuration: 2 beams, 30000 channels, 4 equal zones with 3 equally
    // spaced simple interior knots each.
    const long nfreq = 30000;
    const long M = 2;
    const long nzone = 4;
    const long kint = 3;

    for (const GpuDetrenderLps2dConfig &c: detrender_2d_configs) {
        const long n_phi = c.n_phi;

        vector<long> knots;
        for (long i = 0; i <= n_phi; i++)
            knots.push_back(0);
        const long zw = nfreq / nzone;
        for (long z = 0; z < nzone; z++) {
            const long base = z*zw;
            for (long i = 1; i <= kint; i++)
                knots.push_back(base + (i*zw)/(kint+1));
            if (z < nzone-1)
                for (long i = 0; i <= n_phi; i++)
                    knots.push_back(base + zw);
        }
        for (long i = 0; i <= n_phi; i++)
            knots.push_back(nfreq);

        DetrenderLps2dParams p;
        p.nfreq = nfreq;
        p.knots = knots;
        p.M = M;
        p.n_phi = n_phi;
        p.n = 2;
        p.W = 4;
        p.T = 2048;
        GpuDetrenderLps2d det(p);

        // Global memory traffic: kernel 1 reads the whole buffer, kernel 3 reads and
        // writes the output region. Every byte of an ideal implementation is touched
        // exactly once per pass, so (time -> bandwidth) is the figure of merit: the
        // kernels are expected to be memory bound.
        const double nbytes = double(M) * double(nfreq)
            * (double(det.nbuf) + 2.0*double(p.T)) * 5.0;   // 4 bytes data + 1 byte mask

        // The kernels are branch-free and the work is mask-independent, so the timing
        // does not depend on the data. An all-valid mask is used so that the "normal"
        // path is what gets timed.
        Array<float> data({M, nfreq, det.nbuf}, af_gpu | af_zero);
        Array<unsigned char> mask({M, nfreq, det.nbuf}, af_gpu);
        CUDA_CALL(cudaMemset(mask.data, 1, M*nfreq*det.nbuf));

        cout << "\nGpuDetrenderLps2d::time_selected()\n"
             << "    (n_phi, n, W, T) = (" << p.n_phi << ", " << p.n << ", " << p.W
             << ", " << p.T << "), M = " << M << ", nfreq = " << nfreq << "\n"
             << "    N_phi = " << det.N_phi << ", nzone = " << det.nzone
             << ", nfrange = " << det.nfrange << ", solve_threads = " << det.solve_threads
             << ", channels_per_range = " << det.channels_per_range << "\n"
             << "    data = " << (double(M)*nfreq*det.nbuf*4 / 1.0e9) << " GB, "
             << "mask = " << (double(M)*nfreq*det.nbuf / 1.0e9) << " GB\n"
             << "    global memory traffic per launch = " << (nbytes / 1.0e9) << " GB\n"
             << endl;

        const int niter = 20;
        const int print_interval = 5;
        KernelTimer kt(niter, 1);

        while (kt.next()) {
            det.launch(data, mask, kt.stream);

            if (kt.warmed_up && ((kt.curr_iteration+1) % print_interval == 0)) {
                cout << "    iter " << (kt.curr_iteration+1) << "/" << niter
                     << ": dt = " << (kt.dt * 1.0e3) << " ms"
                     << ", bandwidth = " << (nbytes / kt.dt / 1.0e9) << " GB/s" << endl;
            }
        }
    }
}


}  // namespace pirate
