#include "../include/pirate/PfVariance.hpp"
#include "../include/pirate/DedispersionTree.hpp"
#include "../include/pirate/utils.hpp"    // integer_log2()

#include <cmath>      // ldexp
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
    Array<double> out({nrows, P}, af_rhost | af_zero);
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
        tree_variance[t] = Array<double>({N, 1L << (r - L), P}, af_rhost | af_zero);
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
