#include "../include/pirate/varmap.hpp"
#include "../include/pirate/DedispersionTree.hpp"
#include "../include/pirate/utils.hpp"    // integer_log2()

#include <cmath>      // M_SQRT1_2, ldexp
#include <cstring>    // memcpy
#include <sstream>    // stringstream
#include <limits>     // numeric_limits (coarse_grain_vector's -inf identity)
#include <stdexcept>  // runtime_error
#include <algorithm>  // std::min, std::max, std::max_element

using namespace std;
using namespace ksgpu;

namespace pirate {
#if 0
}  // editor auto-indent
#endif


// -------------------------------------------------------------------------------------------------
//
// SparseTile


SparseTile SparseTile::alloc(long r, long k, long f0, long nf, long nt, long dbits,
                             const long *tshifts, long t0, double scale)
{
    xassert(k >= 0 && k <= constants::max_tree_rank);
    xassert(nf >= 1 && nt >= 1);

    SparseTile out;
    out.r = r;
    out.k = k;
    out.f0 = f0;
    out.nf = nf;
    out.nt = nt;
    out.dbits = dbits;
    out.t0 = t0;
    out.scale = scale;
    for (long j = 0; j < k; j++)
        out.tshifts[j] = tshifts ? tshifts[j] : 0;

    long n = nf * out.S() * nt;
    out.base = std::shared_ptr<double[]>(new double[n]());   // () -> zero-filled (matches np.zeros)
    out.data = out.base.get();
    return out;
}


SparseTile::SparseTile(long r, long k, long f0, long nf, long nt, long dbits,
                       const double *src_data, const long *src_tshifts, long t0, double scale)
{
    *this = alloc(r, k, f0, nf, nt, dbits, src_tshifts, t0, scale);
    if (src_data)
        memcpy(data, src_data, (size_t)(nf * S() * nt) * sizeof(double));
    check_invariants();
}


void SparseTile::check_invariants() const
{
    xassert(0 <= k && k <= r);
    xassert(r <= constants::max_tree_rank);
    xassert(0 <= f0 && nf >= 1 && (f0 + nf) <= (1L << (r - k)));
    xassert(nt >= 1);
    xassert(0 <= dbits && dbits < (1L << k));
    xassert(t0 >= 0);
    for (long j = 0; j < k; j++)
        xassert(tshifts[j] >= 0);
}


SparseTile SparseTile::slice(long c0, long c1) const
{
    xassert(f0 <= c0 && c0 < c1 && c1 <= f0 + nf);
    SparseTile out = *this;                  // copies scalars + tshifts, shares 'base' (refcount++)
    out.f0 = c0;
    out.nf = c1 - c0;
    out.data = data + (c0 - f0) * S() * nt;  // zero-copy view (axis-0 slices are contiguous)
    return out;
}


long SparseTile::remap_d(long d, long dbits_in, long dbits_out)
{
    // dbits_out must be a subset of dbits_in. Map packed index d (over dbits_in) to packed index
    // over dbits_out, both highest-bit-first.
    long dout = 0;
    long tmp = dbits_out;
    while (tmp) {
        long bout = bit_floor(tmp);                  // highest set bit of tmp
        tmp &= ~bout;
        int shift_in  = popcount(dbits_in  & (bout - 1));
        int shift_out = popcount(dbits_out & (bout - 1));
        dout |= ((d >> shift_in) & 1L) << shift_out;
    }
    return dout;
}


long SparseTile::eval_tshifts(long d, long dbits, const long *tshifts)
{
    long T = 0;
    long tmp = dbits;
    while (tmp) {
        int b = bit_length(tmp) - 1;                 // highest set bit position
        tmp &= ~(1L << b);
        int shift = popcount(dbits & ((1L << b) - 1));
        T += ((d >> shift) & 1L) * tshifts[b];
    }
    return T;
}


void SparseTile::dd_tshifts(long k, long *out)
{
    out[0] = 1;
    for (long j = 1; j <= k; j++)
        out[j] = 1L << (j - 1);
}


// Closed-form 'dbits' for an f-range, after 'kmax' iteration steps -- no iteration, no data.
// C++ port of pirate_frb/varmap/SparseTile.py::_predict_dbits(); keep the two in sync (they
// are compared by test_fast_varmap.py::test_cpp_predict_dbits()).
//
// The function is TOTAL: there is no precondition relating 'kmax' to (f0, nf). When the range
// collapses to a single level-kmax tile -- i.e. (f0 >> kmax) == ((f0+nf-1) >> kmax) -- the
// return value is exactly that tile's dbits. Otherwise the range covers several level-kmax
// tiles, whose dbits differ, and the return value is their UNION: the smallest pattern that
// suffices for every f-index in the range. A caller who wants one specific tile F, rather than
// the union, clips to that tile's block first:
//
//    long lo = std::max(f0, F << kmax);
//    long hi = std::min(f0 + nf - 1, ((F+1) << kmax) - 1);
//    long dbits_of_tile_F = predict_dbits(kmax, lo, hi - lo + 1);

long SparseTile::predict_dbits(long kmax, long f0, long nf)
{
    // The upper bound on 'kmax' is not decoration: the return value shifts left by up to kmax,
    // so an unbounded kmax would be a shift overflow (UB). The python twin asserts the same
    // bound, even though python would not overflow, so that the two behave identically.
    xassert(kmax >= 0 && kmax <= constants::max_tree_rank);
    xassert(f0 >= 0);
    xassert(nf >= 1);

    // Iterating sets bits of 'dbits' one level at a time (iterate_aligned() saturates to all
    // bits, iterate_singletons() sets bit 0 when both halves are present, and iterate_lower() /
    // iterate_upper() just shift). Writing a = f0, b = f0+nf-1 and span(j) = (b>>j) - (a>>j),
    // the recurrence is
    //
    //    span(j+1) = (span(j) + a_j) / 2,     a_j = bit j of f0
    //
    // and step j sets a bit iff the span strictly drops, i.e. iff span(j) > a_j. That gives
    // three phases: span >= 2 drops at every step (since a_j <= 1); span == 1 drops iff
    // a_j == 0, and that drop ends it (span goes to 0 and stays); span == 0 does nothing
    // further. So 'dbits' is always a run of high bits plus one isolated lower bit -- never an
    // arbitrary pattern.

    long d = nf - 1;
    if (d == 0)
        return 0;                              // a single channel resolves no delays

    // j1 = length of the leading run, i.e. the number of steps with span >= 2. With
    // e = bit_length(d): for j <= e-2 we have span(j) >= d >= 2^(j+1) >= 2, and for j >= e we
    // have span(j) <= 1 (since d <= 2^j). Only j == e-1 is undecided, and one comparison
    // settles it. Beware: j1 genuinely depends on 'nf', so it CANNOT be read off f0's bit
    // pattern alone (f0=1 gives j1=0 at b=2, but j1=1 at b=3).

    long e = bit_length(d);                    // 2^(e-1) <= d < 2^e
    long j1 = ((f0 & ((1L << (e-1)) - 1)) + d < (1L << e)) ? (e-1) : e;

    // h = position of the isolated bit: the highest bit where the two ends of the range differ,
    // hence the level at which they lie in adjacent blocks but share the block above
    // (span(h) == 1 and span(h+1) == 0). Always h >= j1, since span == 1 throughout [j1, h] --
    // so the run and the isolated bit never collide below.

    long h = bit_length(f0 ^ (f0 + d)) - 1;

    // A bit set at step j is left-shifted once per subsequent step, so after kmax steps it sits
    // at position (kmax-1-j). A step j >= kmax HAS NOT HAPPENED YET, so its bit is simply
    // absent: truncate the run at kmax, and include the isolated bit only when h < kmax. This
    // truncation is what makes the function total, and it is also what makes the union come out
    // right for a range straddling a level-kmax boundary.

    j1 = std::min(j1, kmax);
    long out = ((1L << j1) - 1) << (kmax - j1);
    if (h < kmax)
        out |= 1L << (kmax - 1 - h);
    return out;
}


SparseTile SparseTile::iterate_aligned(const SparseTile &t)
{
    long k = t.k;
    xassert((t.f0 % 2 == 0) && (t.nf % 2 == 0) && (t.nf >= 2));   // "even-aligned"
    xassert(k < t.r);

    long nf_out = t.nf / 2;
    long dbits_out = (1L << (k + 1)) - 1;            // all k+1 bits
    long m_out = k + 1;                              // popcount(dbits_out)
    long nt_in = t.nt;
    long nt_alloc = nt_in + (1L << k);
    long S_in = t.S();
    long S_out = 1L << m_out;                        // 2^(k+1)

    long tshifts_out[constants::max_tree_rank];      // length k+1 = concat([0], t.tshifts)
    tshifts_out[0] = 0;
    for (long j = 0; j < k; j++)
        tshifts_out[j + 1] = t.tshifts[j];

    // scale is folded into the data here, so the output tile has scale = 1.0.
    SparseTile out = alloc(t.r, k + 1, t.f0 / 2, nf_out, nt_alloc, dbits_out, tshifts_out, t.t0, 1.0);
    const double *din = t.data;                      // (nf, S_in, nt_in)
    double *dout = out.data;                         // (nf_out, S_out, nt_alloc)
    double s = t.scale * M_SQRT1_2;

    for (long d = 0; d < (1L << k); d++) {
        long di = remap_d(d, (1L << k) - 1, t.dbits);    // packed index of full delay d in din
        for (long F = 0; F < nf_out; F++) {
            const double *gl = din + (2 * F)     * S_in * nt_in + di * nt_in;   // lower half (even f)
            const double *gu = din + (2 * F + 1) * S_in * nt_in + di * nt_in;   // upper half (odd f)
            double *o2d  = dout + F * S_out * nt_alloc + (2 * d)     * nt_alloc;
            double *o2d1 = dout + F * S_out * nt_alloc + (2 * d + 1) * nt_alloc;
            for (long it = 0; it < nt_in; it++) {
                double u = s * gu[it];               // upper -> both children, unshifted
                double l = s * gl[it];               // lower -> child 2d shifted by d, child 2d+1 by d+1
                o2d [it]         += u;
                o2d1[it]         += u;
                o2d [d + it]     += l;
                o2d1[d + 1 + it] += l;
            }
        }
    }
    return out;
}


SparseTile SparseTile::iterate_lower(const SparseTile &lower)
{
    long k = lower.k;
    xassert(lower.nf == 1 && lower.k < lower.r);

    SparseTile out = lower;                           // shares 'base' (data unchanged); defer 1/sqrt2 into scale
    long ddts[constants::max_tree_rank];
    dd_tshifts(k, ddts);                              // length k+1
    out.tshifts[0] = ddts[0];                         // + 0
    for (long j = 0; j < k; j++)
        out.tshifts[j + 1] = ddts[j + 1] + lower.tshifts[j];
    out.k = k + 1;
    out.f0 = lower.f0 / 2;
    out.nf = 1;
    out.dbits = lower.dbits << 1;
    out.scale = lower.scale * M_SQRT1_2;
    return out;
}


SparseTile SparseTile::iterate_upper(const SparseTile &upper)
{
    long k = upper.k;
    xassert(upper.nf == 1 && upper.k < upper.r);

    SparseTile out = upper;                           // shares 'base' (data unchanged)
    out.tshifts[0] = 0;
    for (long j = 0; j < k; j++)
        out.tshifts[j + 1] = upper.tshifts[j];
    out.k = k + 1;
    out.f0 = upper.f0 / 2;
    out.nf = 1;
    out.dbits = upper.dbits << 1;
    out.scale = upper.scale * M_SQRT1_2;
    return out;
}


SparseTile SparseTile::iterate_singletons(const SparseTile *lower, const SparseTile *upper)
{
    xassert(lower || upper);
    if (!upper)
        return iterate_lower(*lower);
    if (!lower)
        return iterate_upper(*upper);

    // Both halves present: the standard aligned DD(k) merge.
    xassert((lower->r == upper->r) && (lower->k == upper->k));
    xassert(lower->nf == 1 && upper->nf == 1);
    long r = lower->r, k = lower->k;
    xassert(k < r);
    xassert(lower->f0 + 1 == upper->f0);

    long ddts[constants::max_tree_rank];
    dd_tshifts(k, ddts);                              // length k+1
    long s_L[constants::max_tree_rank], s_U[constants::max_tree_rank];   // length k+1
    long res_L[constants::max_tree_rank], res_U[constants::max_tree_rank];
    long tmin[constants::max_tree_rank];

    s_L[0] = ddts[0];                                 // tlo + concat([0], lower.tshifts)
    s_U[0] = 0;                                       // concat([0], upper.tshifts)
    for (long j = 0; j < k; j++) {
        s_L[j + 1] = ddts[j + 1] + lower->tshifts[j];
        s_U[j + 1] = upper->tshifts[j];
    }
    long sumL = 0, sumU = 0;
    for (long j = 0; j <= k; j++) {
        tmin[j] = std::min(s_L[j], s_U[j]);
        res_L[j] = s_L[j] - tmin[j];
        res_U[j] = s_U[j] - tmin[j];
        sumL += res_L[j];
        sumU += res_U[j];
    }

    long t0_out = std::min(lower->t0, upper->t0);
    long c_L = lower->t0 - t0_out, c_U = upper->t0 - t0_out;

    long dbits_out = (lower->dbits | upper->dbits) << 1;   // lifting every bit one level == left shift
    for (long i = 0; i <= k; i++)
        if (res_L[i] + res_U[i] != 0)
            dbits_out |= (1L << i);

    long nt_alloc = std::max(lower->nt + c_L + sumL, upper->nt + c_U + sumU);
    long m_out = popcount(dbits_out);
    double ls = lower->scale * M_SQRT1_2;
    double us = upper->scale * M_SQRT1_2;
    long ldb = lower->dbits << 1, udb = upper->dbits << 1;   // each half's bits, lifted (subset of dbits_out)

    SparseTile out = alloc(r, k + 1, lower->f0 / 2, 1, nt_alloc, dbits_out, tmin, t0_out, 1.0);
    double *dout = out.data;                          // (1, 2^m_out, nt_alloc)

    for (long s_out = 0; s_out < (1L << m_out); s_out++) {
        double *o = dout + s_out * nt_alloc;

        long rL = c_L + eval_tshifts(s_out, dbits_out, res_L);
        const double *colL = lower->data + remap_d(s_out, dbits_out, ldb) * lower->nt;
        for (long it = 0; it < lower->nt; it++)
            o[rL + it] += ls * colL[it];

        long rU = c_U + eval_tshifts(s_out, dbits_out, res_U);
        const double *colU = upper->data + remap_d(s_out, dbits_out, udb) * upper->nt;
        for (long it = 0; it < upper->nt; it++)
            o[rU + it] += us * colU[it];
    }
    return out;
}


ksgpu::Array<double> SparseTile::unpack(long ntime) const
{
    long nd_full = 1L << k;
    long all_k = (1L << k) - 1;
    long S_ = S();

    long maxsh = 0;
    for (long d = 0; d < nd_full; d++)
        maxsh = std::max(maxsh, eval_tshifts(d, all_k, tshifts));
    xassert(ntime >= nt + t0 + maxsh);

    Array<double> out({nf, nd_full, ntime}, af_uhost | af_zero);
    double *o = out.data;
    for (long f = 0; f < nf; f++) {
        for (long d = 0; d < nd_full; d++) {
            long sh = t0 + eval_tshifts(d, all_k, tshifts);
            long j = remap_d(d, all_k, dbits);
            const double *src = data + f * S_ * nt + j * nt;
            double *dst = o + f * nd_full * ntime + d * ntime + sh;
            for (long it = 0; it < nt; it++)
                dst[it] = scale * src[it];
        }
    }
    return out;
}


// -------------------------------------------------------------------------------------------------
//
// SparseTileTriple


void SparseTileTriple::tile_bounds(long f0, long nf, long out_c0[3], long out_c1[3], int &n)
{
    if (nf == 1) {
        out_c0[0] = f0; out_c1[0] = f0 + 1;
        n = 1;
    } else if (nf == 2) {
        out_c0[0] = f0;     out_c1[0] = f0 + 1;
        out_c0[1] = f0 + 1; out_c1[1] = f0 + 2;
        n = 2;
    } else {
        out_c0[0] = f0;          out_c1[0] = f0 + 1;
        out_c0[1] = f0 + 1;      out_c1[1] = f0 + nf - 1;
        out_c0[2] = f0 + nf - 1; out_c1[2] = f0 + nf;
        n = 3;
    }
}


SparseTileTriple SparseTileTriple::from_tile(const SparseTile &t)
{
    SparseTileTriple out;
    out.r = t.r;
    out.k = t.k;
    out.f0 = t.f0;
    out.nf = t.nf;

    long c0[3], c1[3];
    int n;
    tile_bounds(t.f0, t.nf, c0, c1, n);
    out.ntiles = n;
    for (int i = 0; i < n; i++)
        out.tiles[i] = t.slice(c0[i], c1[i]);
    return out;
}


// Emulate np.searchsorted(neg, value, side) where neg[i] = -cm[i] is strictly increasing.
// right==false -> side='left'  (first i with neg[i] >= value);
// right==true  -> side='right' (first i with neg[i] >  value).
static long searchsorted_neg(const double *cm, long len, double value, bool right)
{
    long lo = 0, hi = len;
    while (lo < hi) {
        long mid = (lo + hi) >> 1;
        double neg_mid = -cm[mid];
        bool go_right = right ? (neg_mid <= value) : (neg_mid < value);
        if (go_right)
            lo = mid + 1;
        else
            hi = mid;
    }
    return lo;
}


SparseTileTriple SparseTileTriple::make_tree_gridding_output(const double *cm, long cm_len, long ifreq,
                                                             long flo, long fhi)
{
    long nchan = cm_len - 1;
    long r = integer_log2(nchan);                     // cm_len must be 2^rank + 1
    xassert(ifreq >= 0);

    long f1 = searchsorted_neg(cm, cm_len, -(double)ifreq, false);
    long f0 = searchsorted_neg(cm, cm_len, -(double)(ifreq + 1), true) - 1;
    f0 = std::max(f0, 0L);
    f1 = std::min(f1, nchan);
    xassert(f0 < f1);                                 // ifreq must overlap some tree channel

    // The clip, applied to the source rather than to the result; see the .hpp.
    f0 = std::max(f0, flo);
    if (fhi >= 0)
        f1 = std::min(f1, fhi);
    xassert(f0 < f1);                                 // clip range must meet ifreq's footprint

    long nf = f1 - f0;

    SparseTile tile = SparseTile::alloc(r, 0, f0, nf, 1, 0, nullptr, 0, 1.0);   // k=0 -> S=1, nt=1
    double *d = tile.data;                            // (nf, 1, 1)
    for (long n = 0; n < nf; n++) {
        long idx = f0 + n;
        double w = std::min(cm[idx], (double)(ifreq + 1)) - std::max(cm[idx + 1], (double)ifreq);
        d[n] = std::max(w, 0.0);
    }
    return from_tile(tile);
}


bool SparseTileTriple::get_singleton(long f, SparseTile &out) const
{
    for (int i = 0; i < ntiles; i++) {
        const SparseTile &t = tiles[i];
        if (t.f0 <= f && f < t.f0 + t.nf) {
            out = t.slice(f, f + 1);
            return true;
        }
    }
    return false;
}


SparseTileTriple SparseTileTriple::iterate() const
{
    xassert(k < r);
    long F0 = f0 / 2;
    long last = f0 + nf - 1;
    long Fmax = last / 2;
    long nf_out = Fmax - F0 + 1;

    SparseTileTriple out;
    out.r = r;
    out.k = k + 1;
    out.f0 = F0;
    out.nf = nf_out;

    SparseTile lo, up;
    int idx = 0;

    bool has_lo = get_singleton(2 * F0, lo);
    bool has_up = get_singleton(2 * F0 + 1, up);
    out.tiles[idx++] = SparseTile::iterate_singletons(has_lo ? &lo : nullptr, has_up ? &up : nullptr);

    if (nf_out >= 3) {
        SparseTile mid_in = tiles[1].slice(2 * F0 + 2, 2 * Fmax);
        out.tiles[idx++] = SparseTile::iterate_aligned(mid_in);
    }
    if (nf_out >= 2) {
        has_lo = get_singleton(2 * Fmax, lo);
        has_up = get_singleton(2 * Fmax + 1, up);
        out.tiles[idx++] = SparseTile::iterate_singletons(has_lo ? &lo : nullptr, has_up ? &up : nullptr);
    }
    out.ntiles = idx;
    return out;
}


ksgpu::Array<double> SparseTileTriple::unpack(long ntime) const
{
    long nfreq_full = 1L << (r - k);
    long nd = 1L << k;
    Array<double> out({nfreq_full, nd, ntime}, af_uhost | af_zero);
    for (int i = 0; i < ntiles; i++) {
        Array<double> u = tiles[i].unpack(ntime);     // (nf, nd, ntime), contiguous
        memcpy(out.data + tiles[i].f0 * nd * ntime, u.data,
               (size_t)(tiles[i].nf * nd * ntime) * sizeof(double));
    }
    return out;
}


// -------------------------------------------------------------------------------------------------
//
// PfVarianceConvolver


PfVarianceConvolver::PfVarianceConvolver()
{
    long Wmax = constants::max_pf_width;
    long Lq = integer_log2(Wmax);            // number of levels carrying q=1,2,3 profiles
    Pmax = 3 * Lq + 1;

    // Build the peak-finding kernels (python peak_finding_kernels()). We only need each kernel's
    // one-sided autocorrelation (its A row), but it's clearest to materialize the kernel first.
    std::vector<std::vector<double>> kernels;
    kernels.push_back(std::vector<double>(1, 1.0));            // p=0: finest single sample
    for (long l = 0; l < Lq; l++) {
        long w = 1L << l;
        kernels.push_back(std::vector<double>(2 * w, 1.0));    // q=1: ones(2w)
        {
            std::vector<double> h;                            // q=2: [0.5]*w + [1]*w + [0.5]*w
            h.reserve(2 * w);
            for (long i = 0; i < w; i++) h.push_back(0.5);
            for (long i = 0; i < w; i++) h.push_back(1.0);
            for (long i = 0; i < w; i++) h.push_back(0.5);
            kernels.push_back(std::move(h));
        }
        {
            std::vector<double> h;                            // q=3: [0.5]*w + [1]*2w + [0.5]*w
            h.reserve(4 * w);
            for (long i = 0; i < w; i++)     h.push_back(0.5);
            for (long i = 0; i < 2 * w; i++) h.push_back(1.0);
            for (long i = 0; i < w; i++)     h.push_back(0.5);
            kernels.push_back(std::move(h));
        }
    }
    xassert((long)kernels.size() == Pmax);

    Tmax.resize(Pmax);
    Tmax_last = (long)kernels.back().size();              // longest kernel is the last
    A.assign(Pmax * Tmax_last, 0.0);

    for (long p = 0; p < Pmax; p++) {
        const std::vector<double> &h = kernels[p];
        long T = (long)h.size();
        Tmax[p] = T;
        double *Ap = &A[p * Tmax_last];
        for (long lag = 0; lag < T; lag++) {             // one-sided autocorrelation, zero-padded
            double acc = 0.0;
            for (long t = 0; t + lag < T; t++)
                acc += h[t] * h[t + lag];
            Ap[lag] = acc;
        }
    }
}


void PfVarianceConvolver::variance(const double *x, long S, long nt, long P, double *out) const
{
    xassert(P >= 1 && P <= Pmax);
    xassert(nt >= 1);
    long d = std::min(nt, Tmax[P - 1]);                  // longest kernel among the first P profiles

    double rho[2 * constants::max_pf_width];             // d <= Tmax_last = 2*max_pf_width
    for (long s = 0; s < S; s++) {
        const double *xs = x + s * nt;
        for (long lag = 0; lag < d; lag++) {
            double acc = 0.0;
            for (long t = 0; t + lag < nt; t++)
                acc += xs[t] * xs[t + lag];
            rho[lag] = (lag == 0) ? acc : (2.0 * acc);   // +/- delta symmetry of R_x
        }
        double *os = out + s * P;
        for (long p = 0; p < P; p++) {
            const double *Ap = &A[p * Tmax_last];
            double v = 0.0;
            for (long lag = 0; lag < d; lag++)
                v += rho[lag] * Ap[lag];
            os[p] = v;
        }
    }
}


// -------------------------------------------------------------------------------------------------
//
// SdPlan


// static member
void SdPlan::iter_bits(uint64_t bits, std::vector<long> &out)
{
    out.clear();
    while (bits) {
        uint64_t b = bits & (~bits + 1);              // lowest set bit
        out.push_back(bit_length(long(b)) - 1);
        bits ^= b;
    }
}


SdPlan::SdPlan(const DedispersionConfig &config, const Array<double> &freq_variances)
{
    // EVERY ENTRY POINT ON THIS PATH GOES THROUGH HERE, so this is the one place the config is
    // checked. const and microseconds. It also subsumes what would otherwise need its own tripwire
    // below: the alpha convention assumes 2^R coarse DM channels per multiplet, which is what an
    // unset (auto) dm_downsampling gives, and validate() requires the config's value to be 0 for
    // every primary tree.
    config.validate();

    this->itree0 = config.dedispersion_tree_index(0, 0);

    // Dcore_from_cdd2_registry=false: varmap never reads Dcore, and requiring the registry would
    // make this fail on any build whose compiled cdd2 kernel set does not cover the config. Same
    // choice as python make_tree().
    this->tree0 = DedispersionTree(config, itree0, /*Dcore_from_cdd2_registry=*/ false);

    const FrequencySubbands &fs = tree0.frequency_subbands;
    this->r = tree0.total_rank();
    this->R = fs.pf_rank;
    this->N = fs.N;
    this->M = fs.M;
    this->P = tree0.nprofiles;
    this->nfreq = config.get_total_nfreq();
    this->ndm = 1L << (r - R);
    this->nalpha = ndm * M * P;

    // The sdbits packing has exactly these two headrooms; see sbits_width in the .hpp.
    xassert_le(N, long(sbits_width));
    xassert_le(r, long(constants::max_tree_rank));

    xassert(freq_variances.on_host());
    xassert(freq_variances.is_fully_contiguous());
    if (freq_variances.size != nfreq) {
        stringstream ss;
        ss << "SdPlan: expected freq_variances of length nfreq=" << nfreq
           << ", got length " << freq_variances.size;
        throw runtime_error(ss.str());
    }
    this->freq_variances_vec.assign(freq_variances.data, freq_variances.data + nfreq);

    this->subband_geometry();

    this->cmap = config.make_channel_map();
    xassert(cmap.on_host() && cmap.is_fully_contiguous());

    this->footprint.resize(nfreq);
    this->var_scratch.reserve(64 * P);

    this->plan_pass();
    this->size_pass();
    this->tile_pass();
}


// Set the length-N per-subband tables of the algorithm.
//
// All in TOPLEVEL TREE-FREQ units: a coarse channel is 2^(r-R) tree-freqs wide, so subband n
// occupies [I_lo[n], I_hi[n]), of width 2^c[n] with c[n] = r-R+l[n] its own tree depth.
void SdPlan::subband_geometry()
{
    const FrequencySubbands &fs = tree0.frequency_subbands;

    this->lev = fs.n_to_level;
    this->mbase = fs.n_to_mbase;
    this->c.resize(N);
    this->I_lo.resize(N);
    this->I_hi.resize(N);
    this->I_mid.resize(N);
    this->case1.resize(N);

    for (long n = 0; n < N; n++) {
        long l = lev[n], flo = fs.n_to_flo[n], fhi = fs.n_to_fhi[n];
        c[n] = (r - R) + l;
        I_lo[n] = flo << (r - R);
        I_hi[n] = fhi << (r - R);

        // Case 1 (aligned): I_n is a node of the toplevel tree at level c, so ordinary aligned
        // iteration reproduces the subband's merges. Case 2 (half-aligned, l > 0 and odd index):
        // I_n starts at an odd multiple of 2^(c-1) and is NOT a node of the tree. See
        // notes/dedispersion.tex, section "Subbanded dedispersion".
        case1[n] = ((flo & ((1L << l) - 1)) == 0) ? 1 : 0;

        // Exact: I_hi - I_lo = 2^c, and the only branch that reads I_mid has l >= 1 hence c >= 1,
        // so I_mid = I_lo + 2^(c-1). Note the midpoint is generic -- a case-1 subband's top merge
        // joins its two halves at the same point -- but for case 1 the halves are the ALIGNED
        // pair, which SparseTileTriple::iterate() already merges correctly, so there is nothing to
        // detect and I_mid is never consulted there.
        I_mid[n] = (I_lo[n] + I_hi[n]) / 2;
    }
}


void SdPlan::intersect(long j0, long j1, long n, long &lo, long &hi) const
{
    lo = std::max(j0, I_lo[n]);
    hi = std::min(j1, I_hi[n]);
}


long SdPlan::predict_dbits_r(long lo, long hi, long n) const
{
    // predict_dbits() is called in SUBBAND-LOCAL f-coordinates (f - I_lo[n]), run for the
    // subband's own depth c[n], and then shifted into the level-r labelling.
    //
    // Local coordinates are the right ones on BOTH branches, because I_lo[n] is a multiple of
    // 2^(c-1) in every case (a multiple of 2^c in case 1). Subtracting it preserves the block
    // structure at every level up to c-1, and the subband's own top merge -- the aligned merge of
    // local blocks 0 and 1 -- is the aligned merge predict_dbits() assumes. In other words THE
    // SUBBAND'S DEDISPERSION TREE IS THE ORDINARY ALIGNED TREE IN LOCAL COORDINATES, for case 2 as
    // much as for case 1. That is also why the straddle branch can use
    // SparseTile::iterate_singletons(), which reads only the relative order of its two arguments
    // and not their absolute f-index.
    long cc = c[n];
    return SparseTile::predict_dbits(cc, lo - I_lo[n], hi - lo) << (r - cc);
}


// Per input channel, which subbands see it, over what range, and with what dbits.
//
// TWO CHECKS THE PYTHON HAS AND THIS DOES NOT. Both are guarded by the python's 'debug' flag,
// both are O(subbands) per entry, and both are statements about the PLAN rather than about the
// arithmetic -- so running them once in python covers this port too. varmap/tests.py calls
// compute_detrender_free_varfine(config, v, debug=True), which is where they still run:
//
//   (a) every subband of an entry predicts the same dbits, which is what makes ONE shared row
//       legitimate;
//   (b) within one ifreq, distinct entries carry DISJOINT sbits, so no group ever gets two
//       contributions from one input channel -- which would silently DOUBLE-COUNT here.
//
// The unconditional asserts below are a different matter and are all kept.
void SdPlan::plan_pass()
{
    // Measured, there is about one unstraddled entry per channel (16407 for 16384 channels at
    // chime_sb2_et.yml) and a handful of straddled ones in total.
    unstraddled_plan.reserve(nfreq);
    straddled_plan.reserve(64);

    uint64_t seen_subbands = 0;

    // A LIST OF (lo, hi, sbits), NOT A MAP KEYED BY (lo, hi). It holds at most one entry per
    // unstraddled subband seeing a channel, and measured it is almost always length 1 and never
    // longer than 3 (at chord_sb2_et.yml, N = 25), so a linear scan beats hashing a composite key.
    // Appending also makes the entry order explicit rather than a property of the container:
    // entries come out in ascending-n order, which is the order they reach unstraddled_plan.
    //
    // Declared out here and cleared per channel, so it does not reallocate nfreq times.
    struct LocalEntry { long lo, hi; uint64_t sbits; };
    std::vector<LocalEntry> local_plan;
    local_plan.reserve(8);

    for (long ifreq = 0; ifreq < nfreq; ifreq++) {
        SparseTileTriple tri = SparseTileTriple::make_tree_gridding_output(cmap.data, cmap.size, ifreq);
        long j0 = tri.f0, j1 = tri.f0 + tri.nf;
        footprint[ifreq] = Footprint { j0, j1 };

        local_plan.clear();

        for (long n = 0; n < N; n++) {
            long lo, hi;
            intersect(j0, j1, n, lo, hi);
            if (lo >= hi)
                continue;                       // this subband does not see this channel

            seen_subbands |= (uint64_t(1) << n);

            if ((!case1[n]) && (lo < I_mid[n]) && (I_mid[n] < hi)) {
                long dbits = predict_dbits_r(lo, hi, n);
                straddled_plan.push_back(StraddledEntry {
                    ifreq, n, (uint64_t(dbits) << sbits_width) | (uint64_t(1) << n) });
                continue;
            }

            bool merged = false;
            for (LocalEntry &e : local_plan) {
                if ((e.lo == lo) && (e.hi == hi)) {
                    e.sbits |= (uint64_t(1) << n);
                    merged = true;
                    break;
                }
            }
            if (!merged)
                local_plan.push_back(LocalEntry { lo, hi, uint64_t(1) << n });
        }

        for (const LocalEntry &e : local_plan) {
            long n0 = bit_length(long(e.sbits & (~e.sbits + 1))) - 1;   // any subband will do
            long dbits = predict_dbits_r(e.lo, e.hi, n0);

            // The rule agrees with the simpler GLOBAL form on this branch: [lo,hi) lies inside a
            // single 2^(c-1)-aligned block, so neither f0's low bits nor the XOR of the two
            // endpoints is changed by the shift of origin. This is the cheapest available check
            // that the entry was classified as unstraddled correctly.
            xassert_eq(dbits, SparseTile::predict_dbits(r, e.lo, e.hi - e.lo));

            unstraddled_plan.push_back(UnstraddledEntry {
                ifreq, e.lo, e.hi, (uint64_t(dbits) << sbits_width) | e.sbits });
        }
    }

    this->n_straddled = long(straddled_plan.size());

    // A subband seeing no channel at all would give identically-zero rows of A, which breaks
    // y_true and hence get_distance(). (The python asserts the same thing, per multiplet
    // rather than per subband; see detrender_free.py.)
    uint64_t all_subbands = (N == 64) ? ~uint64_t(0) : ((uint64_t(1) << N) - 1);
    if (seen_subbands != all_subbands) {
        iter_bits(all_subbands & ~seen_subbands, bit_scratch);
        stringstream ss;
        ss << "SdPlan: subband(s)";
        for (long n : bit_scratch)
            ss << " " << n;
        ss << " see no input channel";
        throw runtime_error(ss.str());
    }

    // Every subband of an entry must have R - l[n] zero low bits in the entry's dbits, since the
    // lift's virtual level-r delay index (d << R) | (e << (R-l)) has none there. The argument that
    // this holds is EXACTLY the straddle discriminant, which is why the assertion cannot fire for
    // any other reason:
    //
    //   Write [lo,hi) for the intersection and use the closed form of predict_dbits(). The leading
    //   run occupies bit positions [r-j1, r-1], so it dips below R-l iff j1 > c; the isolated bit
    //   sits at r-1-h, so it dips iff h >= c. Now j1 <= bit_length(hi-lo-1) <= c always, since
    //   [lo,hi) is contained in I_n and I_n has width 2^c -- the run never dips. And h is the
    //   highest bit at which lo and hi-1 differ. In case 1, I_n is 2^c-aligned, so lo and hi-1
    //   agree above bit c-1 and h <= c-1. In case 2, I_n spans the two 2^(c-1)-aligned blocks
    //   either side of its midpoint: if [lo,hi) lies in one of them, again h <= c-2; if it
    //   STRADDLES, then lo >> (c-1) and (hi-1) >> (c-1) differ, so h >= c and the assertion fails.
    //
    // So assertion failures and midpoint straddles are the same set. With the straddle branch
    // taken explicitly the assertion now holds everywhere -- trivially so on that branch, where
    // the << (r - c) supplies R - l[n] zero low bits. Keep it: it is the statement that makes the
    // lift's index formula well defined.
    for (const UnstraddledEntry &e : unstraddled_plan) {
        long dbits = long(e.sdbits >> sbits_width);
        iter_bits(e.sdbits & sbits_mask, bit_scratch);
        for (long n : bit_scratch) {
            long low_bits = dbits & ((1L << (R - lev[n])) - 1);
            xassert_eq(low_bits, 0L);
        }
    }
}

// One plan entry's contribution to the accumulator's size. Idempotent: the FIRST entry carrying a
// given sdbits creates its group, and later ones find it.
void SdPlan::count(uint64_t sdbits)
{
    if (sd_vector_index.count(sdbits))
        return;

    long dbits = long(sdbits >> sbits_width);
    long D_full = 1L << popcount(dbits);

    sd_vector_index[sdbits] = long(sd_vectors.size());
    sd_vectors.push_back(SdVector { sdbits, D_full, std::vector<double>(D_full * P, 0.0) });
}


// Allocate the accumulator. ONE loop over each plan: nothing stops a straddled entry's sdbits from
// coinciding with an unstraddled one's (measured, it never does, but two separate containers merged
// afterwards would silently drop a group).
void SdPlan::size_pass()
{
    for (const UnstraddledEntry &e : unstraddled_plan)
        count(e.sdbits);

    for (const StraddledEntry &e : straddled_plan) {
        // A straddling subband always gets a row to itself, so the two fields of its sdbits are
        // redundant with each other -- and checkable.
        uint64_t sbits = e.sdbits & sbits_mask;
        xassert_eq(sbits, uint64_t(1) << e.n);
        count(e.sdbits);
    }
}


// The level-r normalization, then one term of the accumulator.
//
// A subband's tile lives at its own level c = r-R+l, but rows from subbands of DIFFERENT levels
// share a group, so they must be stored in a COMMON labelling -- and level r is it. Lifting a tile
// from level c to level r is r-c = R-l single-leg dedispersion steps (the other half of each merge
// is absent, since no subband sees a footprint wider than itself), and a single-leg step leaves
// 'data' untouched, shifts dbits left by one, and multiplies 'scale' by 1/sqrt(2). Time shifts
// change, but the variance does not see them. So
//
//     dbits(level r) = dbits(level c) << (R-l)
//     var  (level r) = var  (level c) *  2^-(R-l)
//
// The lift undoes the second line per subband, with a factor 2^(R-l[n]). It CANNOT be folded into
// the stored row: two subbands sharing a row have different l.
void SdPlan::emit(long ifreq, const SparseTile &tile, long klev, uint64_t sdbits)
{
    // This is the assert that closes the loop between the planning pass's closed form and the
    // actual iteration, on both plans. Everything downstream -- the sizing, the shared rows, the
    // lift -- is built on that prediction being right.
    uint64_t dbits_at_r = uint64_t(tile.dbits) << (r - klev);
    uint64_t dbits_key = sdbits >> sbits_width;
    xassert_eq(dbits_at_r, dbits_key);
    xassert_eq(tile.nf, 1L);

    long D_full = tile.S();
    long nvar = D_full * P;
    var_scratch.resize(nvar);

    // tile.data is (nf=1, S, nt), so the leading axis is the python's [0].
    convolver.variance(tile.data, D_full, tile.nt, P, var_scratch.data());

    // TWO SEPARATE SCALINGS, matching the python line for line. scale**2 because variance is
    // quadratic; omitting it is silently wrong wherever an edge tile deferred its 1/sqrt(2)
    // (measured: it moves both varfine and the map by a factor of ~250). The second is the level-r normalization above. They
    // could be folded into one multiply -- 2^-(r-klev) is a power of two, so the fold happens to be
    // exact -- but keeping them apart keeps the correspondence with the reference readable.
    double s2 = tile.scale * tile.scale;
    double p2 = std::ldexp(1.0, -(r - klev));
    for (long i = 0; i < nvar; i++)
        var_scratch[i] = (var_scratch[i] * s2) * p2;

    // THE WEIGHT GOES HERE AND NOWHERE ELSE. With freq_variances all ones this multiply is bitwise
    // the identity, which is what makes compute_detrender_free_varfine(config, ones) equal to the
    // base map's pre-truncation row sum on the python side.
    double w = freq_variances_vec[ifreq];
    std::vector<double> &yg = sd_vectors[sd_vector_index.at(sdbits)].y;
    xassert_eq(long(yg.size()), nvar);
    for (long i = 0; i < nvar; i++)
        yg[i] += w * var_scratch[i];
}


// Build the tiles and fill the accumulator. Two straight-line loops, one per plan.
//
// Two calls to make_tree_gridding_output() are in play and they are NOT the same call: the planning
// pass needed the UNCLIPPED footprint once per channel (cached in 'footprint'), while these need a
// triple CLIPPED to the entry's own [lo, hi), which cannot be hoisted. Its repeated searchsorted
// over cmap is negligible at these sizes.
void SdPlan::tile_pass()
{
    for (const UnstraddledEntry &e : unstraddled_plan) {
        SparseTileTriple tri = SparseTileTriple::make_tree_gridding_output(
            cmap.data, cmap.size, e.ifreq, e.lo, e.hi);
        for (long i = 0; i < r; i++)
            tri = tri.iterate();
        xassert_eq(tri.nf, 1L);
        xassert_eq(tri.ntiles, 1);
        emit(e.ifreq, tri.tiles[0], r, e.sdbits);
    }

    for (const StraddledEntry &e : straddled_plan) {
        long lo, hi;
        intersect(footprint[e.ifreq].j0, footprint[e.ifreq].j1, e.n, lo, hi);
        long cc = c[e.n];

        SparseTileTriple tri = SparseTileTriple::make_tree_gridding_output(
            cmap.data, cmap.size, e.ifreq, lo, hi);
        for (long i = 0; i < cc - 1; i++)
            tri = tri.iterate();

        // The subband's top merge combines the level-(cc-1) blocks either side of its midpoint,
        // which is NOT an aligned pair -- aligned pairs are (2F, 2F+1). Ordinary iteration would
        // merge the lower block with its (absent) left neighbour and the upper with its (absent)
        // right one, producing two tiles that never combine the way the dedisperser combines them.
        // Indexing the pair off the midpoint is the same pair as the "2f+1, 2f+2" of
        // notes/dedispersion.tex Case 2, without needing that section's 'f' convention. Both halves
        // are present by the definition of "straddle", which is what the two asserts say -- the
        // python gets them from get_singleton()'s default allow_none=False.
        long ublk = I_mid[e.n] >> (cc - 1);
        SparseTile lower, upper;
        xassert(tri.get_singleton(ublk - 1, lower));
        xassert(tri.get_singleton(ublk, upper));

        emit(e.ifreq, SparseTile::iterate_singletons(&lower, &upper), cc, e.sdbits);
    }
}


Array<double> SdPlan::lift_sd_vectors() const
{
    Array<double> out({ndm, M, P}, af_uhost | af_zero);

    // ACCUMULATE, not assign: one alpha receives contributions from many groups. The body is the
    // fine Q lift's index arithmetic, over one vector per group instead of one factor block.
    //
    // THE INDEX IS FACTORED, which the python does not do. This is the one place this function
    // departs from its reference for speed, so here is the argument.
    //
    // The python builds the whole (ndm, 2^ll) index array with a VECTORIZED _remap_d() and gathers
    // through it. Calling the scalar remap_d() once per element instead costs O(popcount(dbits))
    // -- up to r = 16 bit operations -- against an inner loop of only P multiply-adds, and
    // measured that way it was 0.78 of the 0.89 seconds this whole path takes at chord_sb2_et.yml.
    //
    // remap_d() is a per-bit GATHER, hence additive over disjoint bit sets: with dbits_in all ones
    // each output bit depends only on which input bit it came from, and an output position
    // popcount(dbits & (bit-1)) is monotone in the input bit. The two halves of
    // dfull = (d << R) | (e << (R-ll)) are disjoint by construction -- e < 2^ll puts e << (R-ll)
    // strictly below bit R, and d << R at or above it -- so
    //
    //     remap_d(dfull) == remap_d(d << R) | remap_d(e << (R-ll))
    //
    // and two small tables replace every call in the inner loop. idx_d depends only on the group,
    // idx_e on the group and the subband's level. Measured: 0.78 s -> 0.47 s, which puts the lift
    // back on the memory traffic it cannot avoid (it touches ndm*2^ll*P doubles of a 48 MB 'out'
    // per (group, subband)) rather than on bit arithmetic.
    std::vector<long> idx_d, idx_e;

    for (const SdVector &g : sd_vectors) {
        long dbits = long(g.sdbits >> sbits_width);
        iter_bits(g.sdbits & sbits_mask, bit_scratch);

        idx_d.resize(ndm);
        for (long d = 0; d < ndm; d++)
            idx_d[d] = SparseTile::remap_d(d << R, (1L << r) - 1, dbits);

        for (long n : bit_scratch) {
            long ll = lev[n], mb = mbase[n], ne = 1L << ll;
            double w = std::ldexp(1.0, R - ll);

            idx_e.resize(ne);
            for (long e = 0; e < ne; e++)
                idx_e[e] = SparseTile::remap_d(e << (R - ll), (1L << r) - 1, dbits);

            for (long d = 0; d < ndm; d++) {
                double *orow = out.data + (d * M + mb) * P;
                for (long e = 0; e < ne; e++) {
                    const double *yrow = &g.y[(idx_d[d] | idx_e[e]) * P];
                    double *o = orow + e * P;
                    for (long p = 0; p < P; p++)
                        o[p] += w * yrow[p];
                }
            }
        }
    }

    return out;
}

// -------------------------------------------------------------------------------------------------
//
// coarse_grain_vector(), expand_fine_vectors(), and the two entry points


Array<double> coarse_grain_vector(const DedispersionTree &tree, const double *y, long ylen, long L)
{
    const FrequencySubbands &fs = tree.frequency_subbands;
    long r = tree.total_rank(), R = fs.pf_rank;
    long N = fs.N, M = fs.M, P = tree.nprofiles;
    long D = 1L << (r - R);

    if (!((R <= L) && (L <= r))) {
        stringstream ss;
        ss << "coarse_grain_vector: L=" << L << " is out of range [R, r] = [" << R << ", " << r << "]";
        throw runtime_error(ss.str());
    }
    xassert_eq(ylen, D * M * P);

    // LABEL-FREE, which is why this is not a gather through an alpha -> beta table. The two fixed
    // shapes of the index convention make it a reshape and two reductions: y is (D, M, P) in C
    // order, subband n owns a CONTIGUOUS multiplet range, and the coarse DM index is a dyadic block
    // of d. The label form would cost a random gather of nalpha elements, which is the dominant
    // term at CHORD's row count.
    //
    // The price is an assumption STRONGER than "multiplets are grouped by subband": they must be
    // ordered BY SUBBAND. FrequencySubbands builds them that way; this is the tripwire, and it is
    // the same one python _subband_tables() applies (it rebuilds n_to_level / n_to_mbase from
    // m_to_n rather than reading them, for exactly this reason).
    for (long m = 1; m < M; m++)
        xassert_le(fs.m_to_n[m-1], fs.m_to_n[m]);
    for (long n = 0; n < N; n++) {
        long mend = fs.n_to_mbase[n] + (1L << fs.n_to_level[n]);
        long mnext = (n+1 < N) ? fs.n_to_mbase[n+1] : M;
        xassert_eq(mend, mnext);
    }

    long f = 1L << (L - R);
    long Dc = D / f;
    Array<double> out({Dc, N, P}, af_uhost);

    // THE REDUCTION IS A MAX, not a mean: a stored variance has to dominate every output it covers.
    // Being a max it is exact and order-independent, so the two stages can be fused without any
    // rounding question -- unlike everything else on this path.
    for (long dc = 0; dc < Dc; dc++) {
        for (long n = 0; n < N; n++) {
            long m0 = fs.n_to_mbase[n], nm = 1L << fs.n_to_level[n];
            double *orow = out.data + (dc * N + n) * P;
            for (long p = 0; p < P; p++)
                orow[p] = -std::numeric_limits<double>::infinity();

            for (long j = 0; j < f; j++) {
                long d = dc * f + j;
                for (long m = m0; m < m0 + nm; m++) {
                    const double *yrow = y + (d * M + m) * P;
                    for (long p = 0; p < P; p++)
                        orow[p] = std::max(orow[p], yrow[p]);
                }
            }
        }
    }

    return out;
}


// Restrict a parent tree's FINE (length-nalpha) vector to a child tree's rows; python
// restrict_fine_vector(). File-static: expand_fine_vectors() is its only caller, and exporting a
// second name would widen this file's surface for nothing.
//
// This is the row map of Proposition 1: the child's variance map is a subset of the parent's ROWS,
// so the child's apply() result is the corresponding subset of the parent's. Row selection commutes
// with a matrix-vector product -- if A_child = A_parent[rows] then A_child @ v == (A_parent @ v)
// [rows] -- which is what lets a caller restrict the small vector rather than the large matrix.
static Array<double> restrict_fine_vector(const double *y, long ylen,
                                          const DedispersionTree &parent,
                                          const DedispersionTree &child)
{
    long D_p = 1L << (parent.total_rank() - parent.frequency_subbands.pf_rank);
    long D_c = 1L << (child.total_rank() - child.frequency_subbands.pf_rank);
    long P_p = parent.nprofiles, P_c = child.nprofiles;

    // Both are equal for every tree of a primary-tree family -- see the appendix's Observation (b)
    // for D, and DedispersionTree's copy of config.primary_trees[ipri] for nprofiles -- so these
    // are tripwires against a future change to the tree construction, not validation of a caller.
    // m_index_mapping() deliberately checks neither.
    xassert_eq(D_c, D_p);
    xassert_eq(P_c, P_p);

    long M_p = parent.frequency_subbands.M;
    long M_c = child.frequency_subbands.M;
    xassert_eq(ylen, D_p * M_p * P_p);

    std::vector<long> m_map = DedispersionTree::m_index_mapping(parent, child);
    xassert_eq(long(m_map.size()), M_c);

    Array<double> out({D_p, M_c, P_p}, af_uhost);
    for (long d = 0; d < D_p; d++) {
        for (long mc = 0; mc < M_c; mc++) {
            const double *src = y + (d * M_p + m_map[mc]) * P_p;
            double *dst = out.data + (d * M_c + mc) * P_p;
            memcpy(dst, src, P_p * sizeof(double));
        }
    }
    return out;
}


std::vector<Array<double>>
expand_fine_vectors(const DedispersionConfig &config, const std::vector<Array<double>> &per_primary)
{
    long npri = config.num_primary_trees();
    if (long(per_primary.size()) != npri) {
        stringstream ss;
        ss << "expand_fine_vectors: got " << per_primary.size() << " vectors for " << npri
           << " primary trees. One per PRIMARY tree is required, in gamma order -- a short list"
           << " would leave holes in the result.";
        throw runtime_error(ss.str());
    }

    long ntrees = config.num_dedispersion_trees();
    std::vector<DedispersionTree> trees;
    trees.reserve(ntrees);
    for (long i = 0; i < ntrees; i++)
        trees.push_back(DedispersionTree(config, i, /*Dcore_from_cdd2_registry=*/ false));

    std::vector<Array<double>> out(ntrees);

    for (long gamma = 0; gamma < npri; gamma++) {
        // See the appendix's fact (a): every primary tree HAS an e == 0 tree, so this cannot fail.
        long iparent = config.dedispersion_tree_index(gamma, 0);
        const DedispersionTree &parent = trees[iparent];

        long D = 1L << (parent.total_rank() - parent.frequency_subbands.pf_rank);
        long M = parent.frequency_subbands.M, P = parent.nprofiles;

        const Array<double> &y = per_primary[gamma];
        xassert(y.on_host() && y.is_fully_contiguous());
        if (y.size != D * M * P) {
            stringstream ss;
            ss << "expand_fine_vectors: primary tree " << gamma << " needs a flat length-"
               << (D*M*P) << " FINE vector, got length " << y.size;
            throw runtime_error(ss.str());
        }

        out[iparent] = y.reshape({D, M, P});

        long net = config.primary_trees.at(gamma).num_early_triggers;
        for (long e = 1; e <= net; e++) {
            long ichild = config.dedispersion_tree_index(gamma, e);
            out[ichild] = restrict_fine_vector(y.data, y.size, parent, trees[ichild]);
        }
    }

    for (long i = 0; i < ntrees; i++)
        xassert(out[i].data != nullptr);

    return out;
}


std::vector<Array<double>>
compute_detrender_free_varfine(const DedispersionConfig &config, const Array<double> &freq_variances)
{
    // Validate BEFORE building the plan: the tile pass is seconds at CHORD, and both a bad config
    // and a length mismatch are knowable without it. SdPlan validates too, but not until after the
    // geometry below.
    config.validate();

    long nfreq = config.get_total_nfreq();
    xassert(freq_variances.on_host());
    xassert(freq_variances.is_fully_contiguous());
    if (freq_variances.size != nfreq) {
        stringstream ss;
        ss << "compute_detrender_free_varfine: expected freq_variances of length nfreq="
           << nfreq << ", got length " << freq_variances.size;
        throw runtime_error(ss.str());
    }

    // Same argument for the restriction geometry: O(N) per tree, and it decides whether the slice
    // below is legitimate at all.
    long npri = config.num_primary_trees();
    DedispersionTree tree0(config, config.dedispersion_tree_index(0, 0), false);
    long M0 = tree0.frequency_subbands.M;
    long D0 = 1L << (tree0.total_rank() - tree0.frequency_subbands.pf_rank);
    long P0 = tree0.nprofiles;

    std::vector<DedispersionTree> trees;
    std::vector<long> Ps;
    for (long g = 1; g < npri; g++) {
        trees.push_back(DedispersionTree(config, config.dedispersion_tree_index(g, 0), false));
        Ps.push_back(trees.back().nprofiles);
    }

    // The slice's one silently-failing precondition, and the only one of Proposition 2's three
    // facts left as a check: the other primary trees must see the same multiplets in the same
    // order, or the slice below is taking the wrong rows.
    for (const DedispersionTree &t : trees) {
        std::vector<long> m_map = DedispersionTree::m_index_mapping(tree0, t);
        xassert_eq(long(m_map.size()), M0);
        for (long m = 0; m < M0; m++)
            xassert_eq(m_map[m], m);
    }

    SdPlan plan(config, freq_variances);
    Array<double> y0 = plan.lift_sd_vectors();          // (ndm, M, P0) == A_base @ v
    xassert_eq(y0.size, D0 * M0 * P0);

    // Proposition 2 as an array operation: the map of primary tree gamma > 0 is the UPPER DM half
    // of the base tree's, truncated to gamma's own profile count.
    //
    // Every entry is a fresh allocation, including gamma == 0. The python can hand back a reshaped
    // view of y0 there, and says so; here y0 is a local, so a view would dangle. The cost is one
    // memcpy of nalpha doubles.
    std::vector<Array<double>> per_primary;
    per_primary.reserve(npri);
    {
        Array<double> y({D0 * M0 * P0}, af_uhost);
        memcpy(y.data, y0.data, y0.size * sizeof(double));
        per_primary.push_back(y);
    }
    for (long g = 1; g < npri; g++) {
        long Pg = Ps[g-1];
        Array<double> y({(D0/2) * M0 * Pg}, af_uhost);
        for (long d = 0; d < D0/2; d++) {
            for (long m = 0; m < M0; m++) {
                const double *src = y0.data + ((d + D0/2) * M0 + m) * P0;
                double *dst = y.data + (d * M0 + m) * Pg;
                memcpy(dst, src, Pg * sizeof(double));
            }
        }
        per_primary.push_back(y);
    }

    // Proposition 1 (early triggers).
    return expand_fine_vectors(config, per_primary);
}


std::vector<Array<double>>
compute_detrender_free_varcoarse(const DedispersionConfig &config, const Array<double> &freq_variances)
{
    std::vector<Array<double>> varfine = compute_detrender_free_varfine(config, freq_variances);

    long ntrees = config.num_dedispersion_trees();
    xassert_eq(long(varfine.size()), ntrees);

    std::vector<Array<double>> out;
    out.reserve(ntrees);

    for (long itree = 0; itree < ntrees; itree++) {
        DedispersionTree tree(config, itree, /*Dcore_from_cdd2_registry=*/ false);
        const FrequencySubbands &fs = tree.frequency_subbands;
        long L = integer_log2(tree.pf.wt_dm_downsampling);

        // ndm_wt is computed by the DedispersionTree constructor as 2^r / wt_dm_downsampling, and
        // L here comes from wt_dm_downsampling directly, so this ties the rank the reduction uses
        // to the shape the weights array actually has. coarse_grain_vector() checks R <= L <= r
        // itself; the tree constructor is what guarantees it.
        long ndm_wt = tree.ndm_wt;
        long ndm_from_L = 1L << (tree.total_rank() - L);
        xassert_eq(ndm_from_L, ndm_wt);

        const Array<double> &y = varfine[itree];
        Array<double> yc = coarse_grain_vector(tree, y.data, y.size, L);
        xassert_eq(yc.size, ndm_wt * fs.N * tree.nprofiles);

        out.push_back(yc.reshape({ndm_wt, fs.N, tree.nprofiles}));
    }

    return out;
}

}  // namespace pirate
