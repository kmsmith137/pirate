#include "../include/pirate/varmap.hpp"
#include "../include/pirate/DedispersionTree.hpp"
#include "../include/pirate/utils.hpp"    // integer_log2()

#include <cmath>      // M_SQRT1_2, ldexp
#include <cstring>    // memcpy
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
// C++ port of pirate_frb/slow_avar/SparseTile.py::_predict_dbits(); keep the two in sync (they
// are compared by test_fast_avar.py::test_cpp_predict_dbits()).
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


SparseTileTriple SparseTileTriple::make_tree_gridding_output(const double *cm, long cm_len, long ifreq)
{
    long nchan = cm_len - 1;
    long r = integer_log2(nchan);                     // cm_len must be 2^rank + 1
    xassert(ifreq >= 0);

    long f1 = searchsorted_neg(cm, cm_len, -(double)ifreq, false);
    long f0 = searchsorted_neg(cm, cm_len, -(double)(ifreq + 1), true) - 1;
    f0 = std::max(f0, 0L);
    f1 = std::min(f1, nchan);
    xassert(f0 < f1);                                 // ifreq must overlap some tree channel
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
// PfVariance


PfVariance::PfVariance(long rank_, long P_)
    : rank(rank_), P(P_)
{
    xassert(rank >= 0);
    xassert(P >= 1);
}


long PfVariance::get_all_dbits() const
{
    long all_dbits = 0;
    for (const Term &t : terms)
        all_dbits |= t.dbits;
    return all_dbits;
}


void PfVariance::accumulate(long dbits, const double *src, long row_off, long nrows, long src_P, double scale)
{
    xassert(0 <= dbits && dbits < (1L << rank));
    xassert(nrows == (1L << popcount(dbits)));
    xassert(src_P >= P);

    Term *term = nullptr;
    for (Term &x : terms)
        if (x.dbits == dbits) { term = &x; break; }
    if (!term) {
        terms.push_back(Term { dbits, std::vector<double>((size_t)(nrows * P), 0.0) });
        term = &terms.back();
    }

    double *dst = term->arr.data();
    for (long i = 0; i < nrows; i++) {
        const double *s = src + (row_off + i) * src_P;
        double *o = dst + i * P;
        for (long p = 0; p < P; p++)
            o[p] += scale * s[p];
    }
}


void PfVariance::add_tile(const SparseTile &t, const PfVarianceConvolver &conv)
{
    xassert(t.nf == 1);
    xassert(t.k == rank);

    long S = t.S();
    double sc2 = t.scale * t.scale;     // variance is quadratic in the data

    // Hot path: from_tile() builds a fresh PfVariance, so the term for t.dbits is new -- compute the
    // variance straight into the term buffer (no extra temporary). The (rare) existing-term case
    // falls back to a temporary + accumulate().
    Term *term = nullptr;
    for (Term &x : terms)
        if (x.dbits == t.dbits) { term = &x; break; }

    if (!term) {
        terms.push_back(Term { t.dbits, std::vector<double>((size_t)(S * P)) });
        double *arr = terms.back().arr.data();
        conv.variance(t.data, S, t.nt, P, arr);
        if (sc2 != 1.0)
            for (long i = 0; i < S * P; i++) arr[i] *= sc2;
        return;
    }

    std::vector<double> tmp((size_t)(S * P));
    conv.variance(t.data, S, t.nt, P, tmp.data());
    accumulate(t.dbits, tmp.data(), 0, S, P, sc2);
}


void PfVariance::add(const PfVariance &src, bool upper_half, double scale)
{
    xassert(this != &src);
    xassert(src.rank == rank + (upper_half ? 1 : 0));
    xassert(src.P >= P);

    long topbit = 1L << rank;
    long sP = src.P;

    for (const Term &t : src.terms) {
        long src_nrows = 1L << popcount(t.dbits);
        if (!upper_half || (t.dbits & topbit) == 0) {
            long dbits = t.dbits;                          // topbit absent -> dbits unchanged
            accumulate(dbits, t.arr.data(), 0, src_nrows, sP, scale);
        } else {
            long dbits = t.dbits & ~topbit;                // drop the (set) top bit
            long nrows = 1L << popcount(dbits);            // == src_nrows / 2
            accumulate(dbits, t.arr.data(), src_nrows / 2, nrows, sP, scale);   // upper half rows
        }
    }
}


PfVariance PfVariance::from_tile(const SparseTile &t, long P, const PfVarianceConvolver &conv)
{
    PfVariance pv(t.k, P);
    pv.add_tile(t, conv);
    return pv;
}


ksgpu::Array<double> PfVariance::unpack(long dbits) const
{
    xassert((get_all_dbits() & ~dbits) == 0);          // dbits must be a superset of every term

    long m = popcount(dbits);
    long nrows = 1L << m;
    Array<double> out({nrows, P}, af_uhost | af_zero);
    double *o = out.data;

    for (const Term &t : terms) {
        const double *ta = t.arr.data();
        for (long row = 0; row < nrows; row++) {
            long j = SparseTile::remap_d(row, dbits, t.dbits);
            const double *s = ta + j * P;
            double *od = o + row * P;
            for (long p = 0; p < P; p++)
                od[p] += s[p];
        }
    }
    return out;
}


// -------------------------------------------------------------------------------------------------
//
// PfAvarApproximation


PfAvarApproximation::PfAvarApproximation(const shared_ptr<DedispersionPlan> &plan, const Array<double> &freq_variances)
{
    xassert(plan);
    nfreq = plan->nfreq;
    ntrees = plan->ntrees;

    xassert(freq_variances.ndim == 1 && freq_variances.shape[0] == nfreq);
    freq_variances_vec.resize(nfreq);
    for (long i = 0; i < nfreq; i++) {
        freq_variances_vec[i] = freq_variances.data[freq_variances.strides[0] * i];
        xassert_gt(freq_variances_vec[i], 0.0);
    }

    tree_r.resize(ntrees);
    tree_R.resize(ntrees);
    tree_L.resize(ntrees);
    tree_P.resize(ntrees);
    tree_ipri.resize(ntrees);
    tree_N.resize(ntrees);
    tree_klevel.resize(ntrees);
    tree_n_to_flo.resize(ntrees);
    tree_n_to_fhi.resize(ntrees);

    for (long t = 0; t < ntrees; t++) {
        const DedispersionTree &tr = plan->trees[t];
        const FrequencySubbands &fs = tr.frequency_subbands;
        tree_r[t] = tr.total_rank();
        tree_R[t] = fs.pf_rank;
        tree_L[t] = integer_log2(tr.pf.wt_dm_downsampling);
        tree_P[t] = tr.nprofiles;
        tree_ipri[t] = tr.primary_tree_index;
        tree_N[t] = fs.N;
        tree_n_to_flo[t] = fs.n_to_flo;
        tree_n_to_fhi[t] = fs.n_to_fhi;
        xassert((tree_R[t] >= 0) && (tree_R[t] <= tree_L[t]) && (tree_L[t] <= tree_r[t]));
        tree_klevel[t] = tree_r[t] - tree_L[t] + (tree_ipri[t] > 0 ? 1 : 0);
    }

    max_klevel = ntrees ? *std::max_element(tree_klevel.begin(), tree_klevel.end()) : 0;
    klevel_Pmax.assign(max_klevel + 1, -1);
    klevel_Lmax.assign(max_klevel + 1, -1);
    for (long t = 0; t < ntrees; t++) {
        long k = tree_klevel[t];
        klevel_Pmax[k] = std::max(klevel_Pmax[k], tree_P[t]);
        klevel_Lmax[k] = std::max(klevel_Lmax[k], tree_L[t]);
    }

    Array<double> cm = plan->config.make_channel_map();
    xassert(cm.ndim == 1);
    channel_map.resize(cm.size);
    for (long i = 0; i < cm.size; i++)
        channel_map[i] = cm.data[cm.strides[0] * i];

    tree_variance.resize(ntrees);
    per_tf.resize(ntrees);
    for (long t = 0; t < ntrees; t++) {
        long r = tree_r[t], L = tree_L[t], P = tree_P[t], R = tree_R[t], N = tree_N[t];
        tree_variance[t] = Array<double>({N, 1L << (r - L), P}, af_uhost | af_zero);
        per_tf[t].clear();
        per_tf[t].reserve(1L << R);
        for (long f = 0; f < (1L << R); f++)
            per_tf[t].emplace_back(r - L, P);
    }

    // Main sweep: for each input frequency channel, grid it and iterate k = 0,1,2,...,max_klevel,
    // accumulating each klevel's singletons into per_tf.
    for (long ifreq = 0; ifreq < nfreq; ifreq++) {
        SparseTileTriple sarr = SparseTileTriple::make_tree_gridding_output(
            channel_map.data(), (long)channel_map.size(), ifreq);
        for (long k = 0; k <= max_klevel; k++) {
            process_klevel(sarr, k, ifreq);
            if (k < max_klevel)
                sarr = sarr.iterate();
        }
    }

    // Final reduction: per tree, per subband n, average per_tf over n's coarse-freq range and
    // densify into tree_variance[t][n].
    for (long t = 0; t < ntrees; t++) {
        long r = tree_r[t], L = tree_L[t], P = tree_P[t], N = tree_N[t];
        long all_dbits = (1L << (r - L)) - 1;
        long blk = (1L << (r - L)) * P;
        double *tv = tree_variance[t].data;            // (N, 2^(r-L), P), contiguous
        for (long n = 0; n < N; n++) {
            long flo = tree_n_to_flo[t][n], fhi = tree_n_to_fhi[t][n];
            PfVariance pv(r - L, P);
            double inv = 1.0 / (double)(fhi - flo);
            for (long f = flo; f < fhi; f++)
                pv.add(per_tf[t][f], false, inv);
            Array<double> u = pv.unpack(all_dbits);    // (2^(r-L), P), contiguous
            memcpy(tv + n * blk, u.data, (size_t)blk * sizeof(double));
        }
        for (long i = 0; i < tree_variance[t].size; i++)
            xassert_gt(tv[i], 0.0);
    }
}


void PfAvarApproximation::process_klevel(const SparseTileTriple &sarr, long k, long ifreq)
{
    if (klevel_Lmax[k] < 0)
        return;                                        // no trees at this klevel

    long f0 = sarr.f0;
    long f1 = std::min(sarr.f0 + sarr.nf, 1L << klevel_Lmax[k]);

    SparseTile tile;
    for (long fp = f0; fp < f1; fp++) {
        if (!sarr.get_singleton(fp, tile))
            continue;

        // Build the per-singleton variance once at the klevel's max P; each tree truncates P and
        // optionally takes the upper DM half in add().
        PfVariance pv = PfVariance::from_tile(tile, klevel_Pmax[k], convolver);

        for (long t = 0; t < ntrees; t++) {
            if (tree_klevel[t] != k)
                continue;
            long R = tree_R[t], L = tree_L[t];
            if (fp >= (1L << L))
                continue;                              // sub-block fp is outside this tree
            bool upper_half = (tree_ipri[t] > 0);
            double norm = std::ldexp(1.0, -(int)(L - R));   // 2^-(L-R)
            long f = fp >> (L - R);                          // coarsify f-index by 2^(L-R)
            per_tf[t][f].add(pv, upper_half, norm * freq_variances_vec[ifreq]);
        }
    }
}


}  // namespace pirate
