#include "../include/pirate/ReferenceTree.hpp"
#include "../include/pirate/constants.hpp"
#include "../include/pirate/inlines.hpp"  // pow2(), bit_reverse_slow()
#include "../include/pirate/utils.hpp"    // check_rank()

#include <ksgpu/xassert.hpp>
#include <ksgpu/test_utils.hpp>           // make_random_strides()

using namespace std;
using namespace ksgpu;

namespace pirate {
#if 0
}  // editor auto-indent
#endif


// Assumes array has been transposed so that shape is (spectator_indices, nfreq, ntime*nspec)
static float *_dedisperse(float *arr, int ndim, const long *shape, const long *strides, int rank, long ntime, long nspec, float *rp, float *sp)
{
    // Recursively reduce to the case ndim == 2.
    if (ndim > 2) {
        for (long i = 0; i < shape[0]; i++)
            rp = _dedisperse(arr + i*strides[0], ndim-1, shape+1, strides+1, rank, ntime, nspec, rp, sp);
        return rp;
    }

    xassert(ndim == 2);    
    xassert(shape[0] == pow2(rank));
    xassert(shape[1] == ntime * nspec);
    xassert((shape[1] == 1) || (strides[1] == 1));

    long ninner = ntime * nspec;
    long rstride = strides[0];
    
    for (int r = 0; r < rank; r++) {
        long ni = pow2(rank-r-1);
        long nj = pow2(r);

        // The index 'i' represents a coarse frequency.
        // The index 'j' represents a bit-reversed delay.
        
        for (long i = 0; i < ni; i++) {
            for (long j = 0; j < nj; j++) {
                long row0 = i*(2*nj) + j;
                long row1 = row0 + nj;
                
                float *a0 = arr + (row0 * rstride);
                float *a1 = arr + (row1 * rstride);

                // FIXME precompute these!!!
                long lag1 = bit_reverse_slow(j,r) * nspec;
                long lag0 = lag1 + nspec;

                // Fill 'scratch' with n=(ntime+1)*nspec samples, obtained by applying lag0.
                long n = (ntime+1) * nspec;   // total samples in scratch buffer
                long n0 = min(lag0, n);       // number of samples which come from ring buffer
                long n1 = n - n0;             // number of samples which come from 'a0'
                memcpy(sp, rp, n0 * sizeof(float));
                memcpy(sp+n0, a0, n1 * sizeof(float));

                // Advance ring buffer (length lag0) by ninner samples.
                long m0 = max(lag0-ninner, 0L);  // number of samples which stay in ring buffer
                long m1 = lag0 - m0;             // number of samples which come from 'a0'
                memmove(rp, rp+ninner, m0 * sizeof(float));
                memcpy(rp+m0, a0+ninner-m1, m1 * sizeof(float));
                rp += lag0;

                // Apply dedispersion (sp,a1) -> (a0,a1).
                for (long k = 0; k < ninner; k++) {
                    float x0 = sp[k];
                    float x1 = sp[k+nspec];
                    float y = a1[k];
                    a0[k] = x1 + y;
                    a1[k] = x0 + y;
                }
            }           
        }
    }

    return rp;
}


ReferenceTree::ReferenceTree(const std::vector<long> &shape_, long nspec_) :
    ReferenceTree(shape_.size(), &shape_[0], nspec_)
{ }


ReferenceTree::ReferenceTree(int ndim_, const long *shape_, long nspec_) :
    ndim(ndim_), nspec(nspec_)
{
    xassert(ndim >= 2);
    xassert(nspec > 0);
    xassert_divisible(shape_[ndim-1], nspec);
    
    this->shape.resize(ndim);

    for (int d = 0; d < ndim; d++) {
        shape[d] = shape_[d];
        xassert(shape[d] > 0);
    }
    
    this->nfreq = shape[ndim-2];
    this->ntime = xdiv(shape[ndim-1], nspec);
    
    xassert(is_power_of_two(nfreq));
    this->rank = integer_log2(nfreq);

    this->npstate = rstate_len(rank) * nspec;   // rstate_len() is declared in utils.hpp
    for (int d = 0; d < ndim-2; d++)
        npstate *= shape[d];
    
    this->pstate = Array<float> ({npstate}, af_uhost | af_zero);
    this->scratch = Array<float> ({(ntime+1)*nspec}, af_uhost | af_zero);
}


void ReferenceTree::dedisperse(ksgpu::Array<float> &arr)
{
    xassert(arr.on_host());
    xassert(arr.shape_equals(this->shape));
    xassert((ntime == 1) || (arr.strides[ndim-1] == 1));
        
    if (rank > 0) {
        float *rp_end = _dedisperse(arr.data, ndim, arr.shape, arr.strides, rank, ntime, nspec, pstate.data, scratch.data);
        xassert(rp_end == pstate.data + npstate);
    }
}    


// static member function
shared_ptr<ReferenceTree> ReferenceTree::make(std::initializer_list<long> shape, long nspec_)
{
    return make_shared<ReferenceTree> (shape.size(), shape.begin(), nspec_);
}


// -------------------------------------------------------------------------------------------------
//
// ReferenceTreeWithSubbands


ReferenceTreeWithSubbands::ReferenceTreeWithSubbands(const Params &params_) :
    params(params_), fs(params_.subband_counts)
{
    xassert(params.num_beams > 0);
    xassert(params.amb_rank >= 0);
    xassert(params.dd_rank >= 0);
    xassert(params.ntime > 0);
    xassert(params.nspec > 0);
        
    xassert(fs.pf_rank <= params.dd_rank);

    long T = params.ntime;
    long S = params.nspec;
    long R = params.dd_rank;
    long B = params.num_beams;
    long A = pow2(params.amb_rank);
    long Dpf = pow2(params.amb_rank + params.dd_rank - fs.pf_rank);

    long scratch_nelts = S * (T + 1 + (R ? pow2(R-1) : 0));
    this->scratch = Array<float> ({scratch_nelts}, af_uhost | af_zero);

    // The rest of the constructor is dedicated to computing 'pstate_nelts'.
    // This computation depends on the details of how dedisperse() is implemented.
    // If these details change, then the constructor will probably need to be revisited.

    long pstate_nelts = 0;

    // Dedispersion contibution to pstate_nelts.
    // At each rank 0 <= r < R, the contribution to pstate_nelts is:
    //   N = 2^(R-r-1) * sum_{0 <= l < 2^r} (l+1)
    //     = (2^{R-1} (2^r+1)) / 2

    for (long r = 0; r < R; r++) {
        // This way of writing N makes sense for R=1 (which implies r=0).
        long N = (pow2(R-1) * (pow2(r)+1)) >> 1;
        pstate_nelts += B * A * N * S;
    }

    // Peak-finding contribution to pstate_nelts.
    //
    // At each 1 <= level <= pf_rank, define r = (dd_rank - pf_rank + level - 1).
    // Each output array element is the sum of 2^{r+1} frequency channels, and the last
    // step is a (2^r x 2^r) -> 2^{r+1} call to dedisperse_1d(). The contribution to
    // pstate_nelts is:
    //
    //   B * A * subband_counts[level] * N
    //
    // where N = sum_{0 <= l < 2^r} (l+1)
    //         = (2^r (2^r+1)) / 2

    for (long level = 1; level <= fs.pf_rank; level++) {
        // This way of writing N makes sense for r=0 (i.e. level=1).
        long r = params.dd_rank - fs.pf_rank + level - 1;
        long N = (pow2(r) * (pow2(r)+1)) >> 1;
        long pf_ns = fs.subband_counts.at(level);
        pstate_nelts += B * A * pf_ns * N * S;
    }

    this->pstate = Array<float> ({pstate_nelts}, af_uhost | af_zero);
}


void ReferenceTreeWithSubbands::dedisperse(Array<float> &buf, Array<float> &out)
{
    long M = fs.M;
    long T = params.ntime;
    long S = params.nspec;
    long B = params.num_beams;
    long A = pow2(params.amb_rank);
    long D = pow2(params.dd_rank);
    long Dpf = pow2(params.amb_rank + params.dd_rank - fs.pf_rank);

    xassert_shape_eq(buf, ({B,A,D,T*S}));
    xassert(buf.get_ncontig() >= 1);
    xassert(buf.on_host());

    if (out.size != 0) {
        xassert_shape_eq(out, ({B,Dpf,M,T*S}));
        xassert(out.get_ncontig() >= 1);
        xassert(out.on_host());
    }
    else if (fs.M != 1)
        throw runtime_error("ReferenceTreeWithSubbands::dedisperse(): if subbands are "
                            "defined (M > 1), then 'out' must be a nonempty array");
    
    float *ps = pstate.data;

    for (long b = 0; b < B; b++) {
        for (long a = 0; a < A; a++) {
            float *bufp = &buf.at({b,a,0,0});
            long buf_dstride = buf.strides[2];

            float *outp = nullptr;
            long out_dstride = 0;
            long out_mstride = 0;

            if (out.size != 0) {
                // The ambient index 'a' is a bit-reversed coarse DM index.
                long pfdm = bit_reverse_slow(a, params.amb_rank) << (params.dd_rank - fs.pf_rank);
                outp = &out.at({b,pfdm,0,0});
                out_dstride = out.strides[1];
                out_mstride = out.strides[2];
            }

            ps = dedisperse_2d(bufp, buf_dstride, outp, out_dstride, out_mstride, ps);
        }
    }

    // cout << " (ps-pstate.data) = " << (ps-pstate.data) << endl;
    // cout << "  pstate.size = " << pstate.size << endl;
    xassert(ps == pstate.data + pstate.size);
}


// 'bufp': shape (2^dd_rank, ntime * nspec), will be dedispersed in place
// 'outp': shape (pf_ndm, M, ntime * nspec), can be NULL.
// 'ps': pointer to current location in persistent_state.
// Returns pointer to updated location in persistent_state.

float *ReferenceTreeWithSubbands::dedisperse_2d(
    float *bufp, long buf_dstride, 
    float *outp, long out_dstride, long out_mstride, 
    float *ps)
{
    // Input array shape: (pow2(dd_rank), T*S)
    // Output array shape: (pf_ndm, M, T*S). Can be NULL.

    long T = params.ntime;
    long S = params.nspec;
    long dd_rank = params.dd_rank;
    long pf_rank = fs.pf_rank;
    long pf_ndm = pow2(dd_rank - pf_rank);
    long mcurr = 0;   // current position 0 <= m < M

    // Note that the loop includes the "senintel" case r=dd_rank at the end.
    for (long r = 0; r <= dd_rank; r++) {

        // In each iteration of the outer loop, we have nf -> (nf/2),
        // ndm -> (2*ndm), and pf_level -> (pf_level+1).

        long nf_in = pow2(dd_rank - r);
        long ndm_in = pow2(r);

        if (outp && (r == (dd_rank - pf_rank))) {

            // Create outputs at pf_level 0. I decided to treat this as a special case,
            // because the bands are "unstaggered" (see comments in FrequencySubbands.cu)
            //
            // Input array has shape (nf_in, ndm_in, T), which we can rewrite as:
            //   (2^pf_rank, pf_ndm, T)     where pf_ndm = 2^(dd_rank-pf_rank)
            //
            // Output array has shape
            //   (pf_ndm, pf_ns, T)         where pf_ns = subband_counts[0] <= 2^pf_rank
            //
            // We just need to transpose (and slice) the input array, but note that:
            //   (1) the input dm-index 0 <= dm_in < pf_ndm is bit-reversed. 
            //   (2) the output array is a "slice" of a shape (pf_ndm, M, T) array.

            long pf_ns = fs.subband_counts.at(0);

            xassert((pf_ns >= 0) && (pf_ns <= pow2(pf_rank)));
            xassert_eq(nf_in, pow2(pf_rank));
            xassert_eq(ndm_in, pf_ndm);

            // Consistency check on m-indices.
            for (long pfs = 0; pfs < pf_ns; pfs++) {
                xassert_eq(fs.m_to_ilo(pfs), pfs);
                xassert_eq(fs.m_to_ihi(pfs), pfs+1);
                xassert_eq(fs.m_to_d.at(pfs), 0);
            }

            // Loop over 0 <= dm_in < ndm_in.
            // Reminder: ndm_in == pf_ndm == 2^r.
            for (long dm_in = 0; dm_in < ndm_in; dm_in++) {
                long dm_in_brev = bit_reverse_slow(dm_in, r);

                for (long pfs = 0; pfs < pf_ns; pfs++) {
                    // Input array at indices (pfs, dm_in_brev, 0).
                    // Output array at indices (dm_in, pfs, 0).
                    long isrc = (pfs * ndm_in) + dm_in_brev;
                    float *src = bufp + (isrc * buf_dstride);
                    float *dst = outp + (dm_in * out_dstride) + (pfs * out_mstride);

                    for (long ts = 0; ts < T*S; ts++)
                        dst[ts] = src[ts];
                }
            }

            mcurr += pf_ns;
        }

        // "Sentinel" case r=dd_rank ends here.
        if (r >= dd_rank)
            break;

        // Next, consider creating "staggered" outputs at pf_level > 0.
        // This is not the optimal way to do it (it would be better to do even/odd differently),
        // but I'm prioritizing simplicity in this reference implementation.

        long pf_level = r + pf_rank - dd_rank + 1;

        if (outp && (pf_level > 0)) {

            // If we get here, we'll create staggered pf_outputs at level (pf_level).
            // The hard part is keeping all the indexing straight!
            //
            // We parameterize the outputs by:
            //
            //  0 <= pf_dm < pf_ndm   where pf_ndm = 2^(dd_rank-pf_rank)
            //  0 <= pfs < pf_ns      where pf_ns = subband_counts[pf_level]
            //  0 <= d2 < pf_nd2      where pf_nd2 = 2^(pf_level-1)
            //  0 <= d1 < 2
            //
            // where we have written the index 0 <= pfd < 2^(pf_level) as pf2 = 2*d2+d1.
            //
            // In this notation, the input array shape (pf_in, ndm_in, T) can be rewritten
            //   (nf_in, pf_ndm, pf_nd2, T)
            //
            // The output array shape can be written as:
            //   (pf_ndm, pf_ns, pf_nd2, 2, T)    where 0 <= pf_ns < (nf_in-1)
            //
            // The inputs and outputs are related as follows: the shape-(2,T) arrays
            //   out[pf_dm, pfs, d2, :, :]
            //   in[pfs:pfs+2, pf_dm, d2, :]
            //
            // are related by dedisperse_1d() with:
            // 
            //   lag = (pf_dm * pf_nd2) + d2
            //
            // Note that the input array is indexed with a bit_reverse()!

            long pf_ns = fs.subband_counts.at(pf_level);
            long pf_nd2 = 1 << (pf_level - 1);

            // Some checks on the above picture.
            xassert((pf_ns >= 0) && (pf_ns <= pow2(pf_rank-pf_level+1)-1));
            xassert_eq(nf_in, pow2(pf_rank - pf_level + 1));
            xassert_eq(ndm_in, pf_ndm * pf_nd2);

            // Consistency check on m-indices.
            for (long pfs = 0; pfs < pf_ns; pfs++) {
                for (long pfd = 0; pfd < 2*pf_nd2; pfd++) {
                    long m = mcurr + (pfs * 2*pf_nd2) + pfd;
                    xassert_eq(fs.m_to_ilo(m), (pfs) * pf_nd2);
                    xassert_eq(fs.m_to_ihi(m), (pfs+2) * pf_nd2);
                    xassert_eq(fs.m_to_d.at(m), pfd);
                }
            }

            // Triple loop over (pf_dm, d2, pfs)
            for (long pf_dm = 0; pf_dm < pf_ndm; pf_dm++) {
                for (long d2 = 0; d2 < pf_nd2; d2++) {
                    long dm_in = (pf_dm * pf_nd2) + d2;
                    long dm_in_brev = bit_reverse_slow(dm_in, r);

                    for (long pfs = 0; pfs < pf_ns; pfs++) {
                        // Input array at indices (pfs:pfs+2, dm_in_brev, 0)
                        long isrc = pfs * ndm_in + dm_in_brev;
                        float *src = bufp + isrc * buf_dstride;
                        long src_stride = ndm_in * buf_dstride;

                        // Output array at "logical" indices (pf_dm, pfs, d2, 0, 0).
                        // The output_array has "physical" shape (pf_ndm, M, T*S)
                        long m = mcurr + (pfs * (2*pf_nd2)) + (2*d2);
                        float *dst = outp + (pf_dm * out_dstride) + (m * out_mstride);
                        long dst_stride = out_mstride;

                        ps = dedisperse_1d(dst, dst_stride, src, src_stride, ps, dm_in);  // lag = dm_in
                    }
                }
            }

            mcurr += pf_ns * (2 * pf_nd2);
        }  // if (pf_level > 0)

        // Dedisperse in-place

        long nf_out = pow2(dd_rank - r - 1);  // = (nf_in / 2)
        long ndm_out = pow2(r+1);             // = (2 * nf_out)

        for (long fout = 0; fout < nf_out; fout++) {
            for (long dm_in_brev = 0; dm_in_brev < ndm_in; dm_in_brev++) {
                long dm_in = bit_reverse_slow(dm_in_brev, r);
                float *p = bufp + (fout * ndm_out + dm_in_brev) * buf_dstride;
                long pstride = ndm_in * buf_dstride;
                ps = dedisperse_1d(p, pstride, p, pstride, ps, dm_in);  // lag = dm_in
            }
        }
    }

    long expected_mcurr = outp ? fs.M : 0;
    xassert(mcurr == expected_mcurr);

    return ps;
}


// 'dst': shape (2,T*S), where T=params.ntime.
// 'src': shape (2,T*S), okay if dst==src.
// 'ps' points to a buffer of length (lag+1)*S.
// FIXME slow implementation -- may improve after tests pass.
inline float *ReferenceTreeWithSubbands::dedisperse_1d(
    float *dst, long dstride, 
    float *src, long sstride, 
    float *ps, long lag)
{
    long T = params.ntime;
    long S = params.nspec;
    float *tmp = scratch.data;

    // (pstate[:], src[0,:]) -> scratch (length (T+lag+1)*S)
    xassert(scratch.size >= (T+lag+1) * S);
    memcpy(tmp, ps, (lag+1) * S * sizeof(float));
    memcpy(tmp + (lag+1)*S, src, T * S * sizeof(float));

    // scratch[(-lag+1):] -> pstate
    memcpy(ps, tmp + T*S, (lag+1) * S * sizeof(float));

    for (long ts = T*S-1; ts >= 0; ts--) {
        // At top of loop, x1 = src[0,t-lag]
        float x0 = tmp[ts];           // x0 = src[0, ts - (lag+1)*S]
        float x1 = tmp[ts+S];         // x1 = src[0, ts - lag*S]
        float y = src[sstride + ts];  // y = src[1, ts]
        
        dst[ts] = x1 + y;             // dst[0, ts]
        dst[dstride + ts] = x0 + y;   // dst[1, ts]
    }

    return ps + (lag+1)*S;
}


// Static member function
void ReferenceTreeWithSubbands::test()
{
    FrequencySubbands fs = FrequencySubbands::make_random();
    auto v = ksgpu::random_integers_with_bounded_product(6, 100000/fs.M);

    long nchunks = v[0];
    long B = v[1];  // number of beams
    long T = v[2];  // time samples per chunk
    long S = v[3];  // spectator indices
    long amb_rank = long(log2(v[4]) + 0.5);
    long dd_rank = fs.pf_rank + long(log2(v[5]) + 0.5);

    long F = fs.F;
    long M = fs.M;
    long A = pow2(amb_rank);
    long Din = pow2(dd_rank);
    long Dpf = pow2(amb_rank + dd_rank - fs.pf_rank);
    long pf_rank = fs.pf_rank;

    // Strides for shape (B,A,Din,T*S) dd array, and shape (B,Dpf,M,T*S) output array.
    // Reminder: make_random_strides() is defined in ksgpu/test_utils.hpp.
    vector<long> dd_strides = ksgpu::make_random_strides({B,A,Din,T*S}, 1);   // ncontig=1
    vector<long> out_strides = ksgpu::make_random_strides({B,Dpf,M,T*S}, 1);  // ncontig=1

    cout << "ReferenceTreeWithSubbands::test():"
         << " nchunks=" << nchunks
         << ", num_beams=" << B
         << ", amb_rank=" << amb_rank
         << ", dd_rank=" << dd_rank
         << ", ntime=" << T
         << ", nspec=" << S
         << ", subband_counts=" << ksgpu::tuple_str(fs.subband_counts)
         << ", dd_strides=" << ksgpu::tuple_str(dd_strides)
         << ", out_strides=" << ksgpu::tuple_str(out_strides)
         << endl;
    
    Params params;
    params.num_beams = B;
    params.amb_rank = amb_rank;
    params.dd_rank = dd_rank;
    params.ntime = T;
    params.nspec = S;
    params.subband_counts = fs.subband_counts;

    ReferenceTreeWithSubbands tree_with_subbands(params);

    std::vector<std::shared_ptr<ReferenceTree>> subtrees(fs.F);

    // Initialize subtrees.
    for (long f = 0; f < F; f++) {
        long ilo = fs.f_to_ilo.at(f) << (dd_rank - pf_rank);
        long ihi = fs.f_to_ihi.at(f) << (dd_rank - pf_rank);
        long subtree_size = ihi - ilo;
       
        std::initializer_list<long> shape = { B, A, subtree_size, T*S };
        subtrees[f] = ReferenceTree::make(shape, S);
    }

    // FIXME should test strides! ('buf' and 'out' only)
    Array<float> in({B,A,Din,T*S}, af_uhost | af_zero);
    Array<float> buf({B,A,Din,T*S}, dd_strides, af_uhost | af_zero);
    Array<float> out({B,Dpf,M,T*S}, out_strides, af_uhost | af_zero);

    // For subtrees.
    Array<float> buf2({B,A,Din,T*S}, af_uhost | af_zero);
    Array<float> out2({B,Dpf,M,T*S}, af_uhost | af_zero);

    for (long ichunk = 0; ichunk < nchunks; ichunk++) {
        in.randomize();
        // in.set_zero();
        // in.at({0,0,0,0}) = 1.0f;

        // Call ReferenceTreeWithSubbands.dedisperse().
        buf.fill(in);
        tree_with_subbands.dedisperse(buf, out);

        // Compare the result with running multiple ReferenceTrees.
        for (long f = 0; f < F; f++) {
            long ilo = fs.f_to_ilo.at(f) << (dd_rank - pf_rank);
            long ihi = fs.f_to_ihi.at(f) << (dd_rank - pf_rank);
            long pf_level = integer_log2(fs.f_to_ihi.at(f) - fs.f_to_ilo.at(f));

            long subtree_size = ihi - ilo;
            long subtree_rank = integer_log2(subtree_size);

            // Copy in -> buf2
            Array<float> dd = buf2.slice(2, 0, subtree_size);
            Array<float> src = in.slice(2, ilo, ihi);
            dd.fill(src);

            subtrees[f]->dedisperse(dd);

            // Copy dd -> out2.
            // The ambient index 'a' is a bit-reversed coarse DM.
            // The subtree index 'd' is a bit-reversed fine DM.
            for (long a = 0; a < A; a++) {
                long dm_c = bit_reverse_slow(a, amb_rank);

                for (long d = 0; d < subtree_size; d++) {
                    long dm_f = bit_reverse_slow(d, subtree_rank);

                    // (dm_c, dm_f) -> (dpf, m)
                    long dpf = (dm_c << (dd_rank - pf_rank)) + (dm_f >> pf_level);
                    long m = fs.f_to_mbase.at(f) + (dm_f & (pow2(pf_level)-1));

                    // dd_slice: shape (B, A, subtree_size, T*S) -> (B, T)
                    Array<float> dd_slice = dd.slice(1, a);
                    dd_slice = dd_slice.slice(1, d);

                    // out_slice: shape (B, Dpf, M, T) -> (B, T*S)
                    Array<float> out_slice = out2.slice(1, dpf);  // note 'out2' on rhs (not 'out')
                    out_slice = out_slice.slice(1, m);

                    out_slice.fill(dd_slice);
                }
            }
        }

        // Check that full band is last, so that we can directly compare 'buf' and 'buf2'.
        xassert(fs.f_to_ilo.at(F-1) == 0);
        xassert(fs.f_to_ihi.at(F-1) == pow2(pf_rank));

        stringstream ss;
        ss << "(chunk=" << ichunk << ")";

        assert_arrays_equal(buf, buf2, "buf"+ss.str(), "buf2", {"b","a","dmbr","ts"});
        assert_arrays_equal(out, out2, "out"+ss.str(), "out2", {"b","dpf","m","ts"});
    }
}


}  // namespace pirate
