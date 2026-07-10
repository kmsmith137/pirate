#include "../include/pirate/SparseTile.hpp"
#include "../include/pirate/utils.hpp"    // integer_log2()

#include <cmath>      // M_SQRT1_2
#include <cstring>    // memcpy
#include <algorithm>  // std::min, std::max

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


}  // namespace pirate
