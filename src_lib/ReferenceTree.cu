#include "../include/pirate/ReferenceTree.hpp"
#include "../include/pirate/constants.hpp"
#include "../include/pirate/inlines.hpp"  // pow2(), bit_reverse_slow()
#include "../include/pirate/utils.hpp"    // check_rank()

#include <ksgpu/xassert.hpp>

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
        
    xassert(fs.pf_rank <= params.dd_rank);

    long B = params.num_beams;
    long A = pow2(params.amb_rank);
    long R = params.dd_rank;
    long Dpf = pow2(params.amb_rank + params.dd_rank - fs.pf_rank);

    long scratch_nelts = params.ntime + 1 + (R ? pow2(R-1) : 0);
    long pstate_nelts = 0;

    // Dedispersion contibution to pstate_nelts.
    // At each rank 0 <= r < R, the contribution to pstate_nelts is:
    //   N = 2^(R-r-1) * sum_{0 <= l < 2^r} (l+1)
    //     = (2^{R-1} (2^r+1)) / 2

    for (long r = 0; r < R; r++) {
        // This way of writing N makes sense for R=1 (which implies r=0).
        long N = (pow2(R-1) * (pow2(r)+1)) >> 1;
        pstate_nelts += B * A * N;
    }

    // Peak-finding contribution to pstate_nelts.
    // At each 1 <= level <= pf_rank, the "mini" dedispersion rank is r = (level-1)
    // and the "mini" contribution to pstate_nelts is 
    //   N = sum_{0 <= l < 2^r} (l+1)
    //     = (2^r (2^r+1)) / 2

    for (long level = 1; level <= fs.pf_rank; level++) {
        // This way of writing N makes sense for r=0 (i.e. level=1).
        long pf_ns = fs.subband_counts.at(level);
        long N = (pow2(level-1) * (pow2(level-1)+1)) >> 1;
        pstate_nelts += B * Dpf * pf_ns * N;
    }

    this->pstate = Array<float> ({pstate_nelts}, af_uhost | af_zero);
    this->scratch = Array<float> ({scratch_nelts}, af_uhost | af_zero);
}


void ReferenceTreeWithSubbands::dedisperse(Array<float> &buf, Array<float> &out)
{
    long M = fs.M;
    long T = params.ntime;
    long B = params.num_beams;
    long A = pow2(params.amb_rank);
    long D = pow2(params.dd_rank);
    long Dpf = pow2(params.amb_rank + params.dd_rank - fs.pf_rank);

    xassert(buf.shape_equals({B,A,D,T}));
    xassert(buf.get_ncontig() >= 1);
    xassert(buf.on_host());

    xassert(out.shape_equals({B,Dpf,M,T}));
    xassert(out.get_ncontig() >= 1);
    xassert(out.on_host());
    
    float *ps = pstate.data;

    for (long b = 0; b < B; b++) {
        for (long a = 0; a < A; a++) {
            float *bufp = &buf.at({b,a,0,0});
            long buf_dstride = buf.strides[2];

            // The ambient index 'a' is a bit-reversed coarse DM index.
            long pfdm = bit_reverse_slow(a, params.amb_rank) << (params.dd_rank - fs.pf_rank);
            float *outp = &out.at({b,pfdm,0,0});
            long out_dstride = out.strides[1];
            long out_mstride = out.strides[2];

            ps = dedisperse_2d(bufp, buf_dstride, outp, out_dstride, out_mstride, ps);
        }
    }

    xassert(ps == pstate.data + pstate.size);
}


// 'bufp': shape (2^dd_rank, ntime), will be dedispersed in place
// 'outp': shape (pf_ndm, M, ntime)
// 'ps': pointer to current location in persistent_state.
// Returns pointer to updated location in persistent_state.

float *ReferenceTreeWithSubbands::dedisperse_2d(
    float *bufp, long buf_dstride, 
    float *outp, long out_dstride, long out_mstride, 
    float *ps)
{
    // Input array shape: (pow2(dd_rank), T)
    // Output array shape: (pf_ndm, M, T).

    long dd_rank = params.dd_rank;
    long pf_rank = fs.pf_rank;
    long pf_ndm = pow2(dd_rank - pf_rank);
    long T = params.ntime;
    long mcurr = 0;   // current position 0 <= m < M

    // Note that the loop includes the "senintel" case r=dd_rank at the end.
    for (int r = 0; r <= dd_rank; r++) {

        // In each iteration of the outer loop, we have nf -> (nf/2),
        // ndm -> (2*ndm), and pf_level -> (pf_level+1).

        int nf_in = pow2(dd_rank - r);
        int ndm_in = pow2(r);

        if (r == (dd_rank - pf_rank)) {

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

            xassert((pf_ns >= 0) && (pf_ns < pow2(pf_rank)));
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
            for (int dm_in = 0; dm_in < ndm_in; dm_in++) {
                long dm_in_brev = bit_reverse_slow(dm_in, r);

                for (int pfs = 0; pfs < pf_ns; pfs++) {
                    // Input array at indices (pfs, dm_in_brev, 0).
                    // Output array at indices (dm_in, pfs, 0).
                    long isrc = (pfs * ndm_in) + dm_in_brev;
                    float *src = bufp + (isrc * buf_dstride);
                    float *dst = outp + (dm_in * out_dstride) + (pfs * out_mstride);

                    for (int t = 0; t < T; t++)
                        dst[t] = src[t];
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

        int pf_level = r + pf_rank - dd_rank + 1;

        if (pf_level > 0) {

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
            // are related by dedisperse_1d() with lag=d2.
            // Note that the input array is indexed with a bit_reverse()!

            long pf_ns = fs.subband_counts.at(pf_level);
            long pf_nd2 = 1 << (pf_level - 1);

            // Some checks on the above picture.
            xassert((pf_ns >= 0) && (pf_ns < pow2(pf_rank-pf_level+1)-1));
            xassert_eq(nf_in, pow2(pf_rank - pf_level + 1));
            xassert_eq(ndm_in, pf_ndm * pf_nd2);

            // Consistency check on m-indices.
            for (int pfs = 0; pfs < pf_ns; pfs++) {
                for (int pfd = 0; pfd < 2*pf_nd2; pfd++) {
                    int m = mcurr + (pfs * 2*pf_nd2) + pfd;
                    xassert_eq(fs.m_to_ilo(m), (pfs) * pf_nd2);
                    xassert_eq(fs.m_to_ihi(m), (pfs+2) * pf_nd2);
                    xassert_eq(fs.m_to_d.at(m), pfd);
                }
            }

            // Triple loop over (pf_dm, d2, pfs)
            for (int pf_dm = 0; pf_dm < pf_ndm; pf_dm++) {
                for (int d2 = 0; d2 < pf_nd2; d2++) {
                    int dm_in = (pf_dm * pf_nd2) + d2;
                    int dm_in_brev = bit_reverse_slow(dm_in, r);

                    for (int pfs = 0; pfs < pf_ns; pfs++) {
                        // Input array at indices (pfs:pfs+2, dm_in_brev, 0)
                        long isrc = pfs * ndm_in + dm_in_brev;
                        float *src = bufp + isrc * buf_dstride;
                        long src_stride = ndm_in * buf_dstride;

                        // Output array at "logical" indices (pf_dm, pfs, d2, 0, 0).
                        // The output_array has "physical" shape (pf_ndm, M, T)
                        long m = mcurr + (pfs * (2*pf_nd2)) + (2*d2);
                        float *dst = outp + (pf_dm * out_dstride) + (m * out_mstride);
                        long dst_stride = T;

                        ps = dedisperse_1d(dst, dst_stride, src, src_stride, ps, d2);  // lag=d2
                    }
                }
            }

            mcurr += pf_ns * (2 * pf_nd2);
        }  // if (pf_level > 0)

        // Dedisperse in-place

        int nf_out = pow2(dd_rank - r - 1);  // = (nf_in / 2)
        int ndm_out = pow2(r+1);             // = (2 * nf_out)

        for (int fout = 0; fout < nf_out; fout++) {
            for (int dm_in_brev = 0; dm_in_brev < ndm_in; dm_in_brev++) {
                long dm_in = bit_reverse_slow(dm_in_brev, r);
                float *p = bufp + (fout * ndm_out + dm_in_brev) * buf_dstride;
                long pstride = ndm_in * buf_dstride;
                ps = dedisperse_1d(p, pstride, p, pstride, ps, dm_in);  // lag = dm_in
            }
        }
    }

    xassert(mcurr == fs.M);

    return ps;
}

// 'dst': shape (2,T), where T=params.ntime.
// 'src': shape (2,T), okay if dst==src.
// 'ps' points to a buffer of length (lag+1).
// FIXME slow implementation -- may improve after tests pass.
inline float *ReferenceTreeWithSubbands::dedisperse_1d(
    float *dst, long dstride, 
    float *src, long sstride, 
    float *ps, long lag)
{
    long T = params.ntime;
    float *tmp = scratch.data;

    // (pstate[:], src[0,:]) -> scratch (length T+lag+1)
    xassert(scratch.size >= T+lag+1);
    memcpy(tmp, ps, (lag+1) * sizeof(float));
    memcpy(tmp + (lag+1), src, T * sizeof(float));

    // scratch[(-lag+1):] -> pstate
    memcpy(ps, tmp + T, (lag+1) * sizeof(float));

    float x1 = tmp[T];  // src[0,T-1-lag]

    for (long t = T-1; t >= 0; t--) {
        // At top of loop, x1 = src[0,t-lag]
        float x0 = tmp[t];  // x0 = src[0,t-lag-1]
        float y = src[sstride + t];
        
        dst[t] = x1 + y;
        dst[dstride + t] = x0 + y;
        x1 = x0;
    }

    return ps;
}


}  // namespace pirate
