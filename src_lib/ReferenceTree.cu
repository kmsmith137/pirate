#include "../include/pirate/ReferenceTree.hpp"
#include "../include/pirate/ReferenceLagbuf.hpp"
#include "../include/pirate/constants.hpp"
#include "../include/pirate/inlines.hpp"  // pow2(), bit_reverse_slow()
#include "../include/pirate/utils.hpp"    // dedispersion_delay(), dedisperse_non_incremental()

#include <ksgpu/xassert.hpp>
#include <ksgpu/rand_utils.hpp>           // rand_int(), random_integers_with_bounded_product()
#include <ksgpu/string_utils.hpp>         // tuple_str()
#include <ksgpu/test_utils.hpp>           // make_random_strides(), assert_arrays_equal()

using namespace std;
using namespace ksgpu;

namespace pirate {
#if 0
}  // editor auto-indent
#endif


ReferenceTree::ReferenceTree(const Params &params_) :
    params(params_), fs(params_.subband_counts)
{
    xassert(params.num_beams > 0);
    xassert(params.amb_rank >= 0);
    xassert(params.dd_rank >= 0);
    xassert(params.ntime > 0);
    xassert(params.nspec > 0);
        
    xassert(fs.pf_rank <= params.dd_rank);

    long M = fs.M;
    long T = params.ntime;
    long S = params.nspec;
    long R = params.dd_rank;
    long B = params.num_beams;
    long A = pow2(params.amb_rank);
    long Dpf = pow2(params.amb_rank + params.dd_rank - fs.pf_rank);

    long scratch_nelts = S * (T + pow2(R) + 1);
    this->scratch = Array<float> ({scratch_nelts}, af_uhost | af_zero);

    Array<int> final_lags({B,Dpf,M}, af_uhost | af_zero);
    for (long b = 0; b < B; b++) {
        for (long dpf = 0; dpf < Dpf; dpf++) {
            for (long m = 0; m < M; m++) {
                long ff = pow2(fs.pf_rank) - fs.m_to_ihi(m);
                long dd = dpf & (pow2(params.dd_rank-fs.pf_rank) - 1);
                final_lags.at({b,dpf,m}) = ff * dd * S;
            }
        }
    }

    this->final_lagbuf = make_shared<ReferenceLagbuf> (final_lags, T*S);    

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

    // Contribution to pstate_nelts from peak-finding at pf_level >= 1.
    // Written in a brain-dead way which "mirrors" ReferenceTree::dedisperse_2d().

    for (long pf_level = 1; pf_level <= fs.pf_rank; pf_level++) {
        long nf_in = pow2(fs.pf_rank - pf_level + 1);
        long pf_ndm = pow2(params.dd_rank - fs.pf_rank);
        long pf_nd2 = pow2(pf_level - 1);

        for (long pfs = 0; pfs < fs.subband_counts.at(pf_level); pfs++) {
            for (long pf_dm = 0; pf_dm < pf_ndm; pf_dm++) {
                for (long d2 = 0; d2 < pf_nd2; d2++) {
                    long dm_in = (pf_dm * pf_nd2) + d2;
                    long dd_lag = dm_in;
                    pstate_nelts += B * A * (dd_lag+1) * S;
                }
            }
        }
    }

    this->pstate = Array<float> ({pstate_nelts}, af_uhost | af_zero);
}


void ReferenceTree::dedisperse(Array<float> &buf, Array<float> &out)
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
        throw runtime_error("ReferenceTree::dedisperse(): if subbands are "
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

    if (out.size != 0)
        final_lagbuf->apply_lags(out);

    // cout << " (ps-pstate.data) = " << (ps-pstate.data) << endl;
    // cout << "  pstate.size = " << pstate.size << endl;
    xassert(ps == pstate.data + pstate.size);
}


// 'bufp': shape (2^dd_rank, ntime * nspec), will be dedispersed in place
// 'outp': shape (pf_ndm, M, ntime * nspec), can be NULL.
// 'ps': pointer to current location in persistent_state.
// Returns pointer to updated location in persistent_state.

float *ReferenceTree::dedisperse_2d(
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
            //   (3) in order to match the GPU kernel, we apply lag (nf_in-pfs-1) * (dm_in).

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
            //   dedispersion lag = (pf_dm * pf_nd2) + d2
            //
            // Note that the input array is indexed with a bit_reverse()!

            long pf_ns = fs.subband_counts.at(pf_level);
            long pf_nd2 = 1 << (pf_level - 1);

            // Some checks on the above picture.
            xassert((pf_ns >= 0) && (pf_ns <= pow2(pf_rank-pf_level+1)-1));
            xassert_eq(nf_in, pow2(pf_rank - pf_level + 1));
            xassert_eq(ndm_in, pow2(dd_rank - pf_rank + pf_level - 1));
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

                        long dd_lag = dm_in;
                        ps = dedisperse_1d(dst, dst_stride, src, src_stride, ps, dd_lag);
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
inline float *ReferenceTree::dedisperse_1d(
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


// -------------------------------------------------------------------------------------------------
//
// ReferenceTree::test_basics()


// Static member function
void ReferenceTree::test_basics()
{
    vector<long> v = ksgpu::random_integers_with_bounded_product(6, 300000);

    long N = v[0];   // number of chunks
    long B = v[1];   // number of beams
    long T = v[2];   // number of time samples per chunk
    long S = v[3];   // number of "inner" spectator indices
    long amb_rank = long(log2(v[4]) + 0.5);
    long dd_rank = long(log2(v[5]) + 0.5);

    amb_rank = min(amb_rank, 8L);
    dd_rank = min(dd_rank, 8L);

    long A = pow2(amb_rank);
    long D = pow2(dd_rank);
    vector<long> dd_strides = ksgpu::make_random_strides({B,A,D,T*S}, 1);  // ncontig=2

    cout << "test_reference_tree:"
         << " nchunks=" << N
         << ", nbeams=" << B
         << ", ntime=" << T
         << ", nspec=" << S
         << ", amb_rank=" << amb_rank
         << ", dd_rank=" << dd_rank 
         << ", dd_strides=" << ksgpu::tuple_str(dd_strides)
         << endl;

    Array<float> in({B,A,D,N*T*S}, af_uhost | af_random);
    Array<float> out_dni({B,A,D,N*T*S}, af_uhost | af_zero);     // using dedisperse_non_incremental()
    Array<float> out_tree({B,A,D,N*T*S}, af_uhost | af_zero);    // using ReferenceTree

    // Temp buffers for dedispersion.
    Array<float> dd_dni({D,N*T*S}, af_uhost | af_zero);
    Array<float> dd_tree({B,A,D,T*S}, dd_strides, af_uhost | af_zero);
    Array<float> dd_bfs({S}, af_uhost | af_zero);                // "bfs" = "brute-force sum"

    // Dedisperse using dedisperse_non_incremental().

    for (long b = 0; b < B; b++) {
        for (long a = 0; a < A; a++) {
            Array<float> in_slice = in.slice(0,b);   // shape (A,D,N*T*S)
            in_slice = in_slice.slice(0,a);          // shape (D,N*T*S)

            dd_dni.fill(in_slice);
            dedisperse_non_incremental(dd_dni, S);

            Array<float> out_slice = out_dni.slice(0,b);   // shape (A,D,N*T*S)
            out_slice = out_slice.slice(0,a);              // shape (D,N*T*S)

            out_slice.fill(dd_dni);
        }
    }

    // Check a few random entries of 'out_dni', by comparing them to brute-force summation.

    for (int e = 0; e < 10; e++) {
        long b = rand_int(0,B);
        long a = rand_int(0,A);
        long t = rand_int(0,N*T);   // note (N*T), not T
        long dm_brev = rand_int(0,D);

        Array<float> out_slice = out_dni.slice(0,b);    // (A,D,N*T*S)
        out_slice = out_slice.slice(0,a);               // (D,N*T*S)
        out_slice = out_slice.slice(0,dm_brev);         // (N*T*S,)
        out_slice = out_slice.slice(0,t*S,(t+1)*S);     // (S,)

        Array<float> in_slice = in.slice(0,b);          // (A,D,N*T*S)
        in_slice = in_slice.slice(0,a);                 // (D,N*T*S)
        
        dd_bfs.set_zero();

        for (long f = 0; f < D; f++) {
            long t0 = t - dedispersion_delay(dd_rank, f, dm_brev);
            if (t0 < 0)
                continue;
            for (long s = 0; s < S; s++)
                dd_bfs.data[s] += in_slice.data[f*N*T*S + t0*S + s];
        }

        assert_arrays_equal(out_slice, dd_bfs, "dedisperse_non_incremental", "brute_force_sum", {"s"});
    }

    // Dedisperse using ReferenceTree.

    ReferenceTree::Params params;
    params.num_beams = B;
    params.amb_rank = amb_rank;
    params.dd_rank = dd_rank;
    params.ntime = T;
    params.nspec = S;
    params.subband_counts = {1};

    ReferenceTree tree(params);
    Array<float> sb_empty;

    for (long ichunk = 0; ichunk < N; ichunk++) {
        Array<float> in_chunk = in.slice(3, ichunk*T*S, (ichunk+1)*T*S);   // (B,A,D,N*T*S) -> (B,A,D,T*S)
        dd_tree.fill(in_chunk);

        tree.dedisperse(dd_tree, sb_empty);

        Array<float> out_chunk = out_tree.slice(3, ichunk*T*S, (ichunk+1)*T*S);   // (B,A,D,N*T*S) -> (B,A,D,T*S)
        out_chunk.fill(dd_tree);
    }

    assert_arrays_equal(out_tree, out_dni, "ReferenceTree", "dedisperse_non_incremental", {"b","a","d","ts"});
}


// -------------------------------------------------------------------------------------------------
//
// ReferenceTree::test_subbands()


// Static member function
void ReferenceTree::test_subbands()
{
    FrequencySubbands fs = FrequencySubbands::make_random();
    auto v = ksgpu::random_integers_with_bounded_product(6, 100000/fs.M);

    long nchunks = v[0];
    long B = v[1];  // number of beams
    long T = v[2];  // time samples per chunk
    long S = v[3];  // spectator indices
    long amb_rank = long(log2(v[4]) + 0.5);
    long dd_rank = fs.pf_rank + long(log2(v[5]) + 0.5);

    // Strides for shape (B,A,Din,T*S) dd array, and shape (B,Dpf,M,T*S) output array.
    // Reminder: make_random_strides() is defined in ksgpu/test_utils.hpp.
    vector<long> dd_strides = ksgpu::make_random_strides({B, pow2(amb_rank), pow2(dd_rank), T*S}, 1);             // ncontig=1
    vector<long> out_strides = ksgpu::make_random_strides({B, pow2(amb_rank+dd_rank-fs.pf_rank), fs.M, T*S}, 1);  // ncontig=1

    // To simplify the test, uncomment one or more of the following lines.
    //
    // nchunks = 1;
    // B = 1;
    // amb_rank = 0;
    // dd_rank = 3;
    // T = 4;
    // S = 1;
    // fs = FrequencySubbands({1,1});
    // dd_strides = ksgpu::make_contiguous_strides({B, pow2(amb_rank), pow2(dd_rank), T*S});
    // out_strides = ksgpu::make_contiguous_strides({B, pow2(amb_rank+dd_rank-fs.pf_rank), fs.M, T*S});

    cout << "ReferenceTree::test_subbands():"
         << " nchunks=" << nchunks
         << ", num_beams=" << B
         << ", amb_rank=" << amb_rank
         << ", dd_rank=" << dd_rank
         << ", ntime=" << T
         << ", nspec=" << S
         << ", subband_counts=" << ksgpu::tuple_str(fs.subband_counts)
         << ", M=" << fs.M
         << ", dd_strides=" << ksgpu::tuple_str(dd_strides)
         << ", out_strides=" << ksgpu::tuple_str(out_strides)
         << endl;

    long F = fs.F;
    long M = fs.M;
    long A = pow2(amb_rank);
    long Din = pow2(dd_rank);
    long Dpf = pow2(amb_rank + dd_rank - fs.pf_rank);
    long pf_rank = fs.pf_rank;    

    Params params;
    params.num_beams = B;
    params.amb_rank = amb_rank;
    params.dd_rank = dd_rank;
    params.ntime = T;
    params.nspec = S;
    params.subband_counts = fs.subband_counts;

    ReferenceTree tree_with_subbands(params);
    vector<shared_ptr<ReferenceTree>> subtrees(F);

    // Initialize subtrees.
    for (long f = 0; f < F; f++) {
        long ilo = fs.f_to_ilo.at(f);
        long ihi = fs.f_to_ihi.at(f);

        long subtree_size = (ihi - ilo) << (dd_rank - pf_rank);
        long subtree_rank = integer_log2(subtree_size);
       
        Params subtree_params;
        subtree_params.num_beams = B;
        subtree_params.amb_rank = amb_rank;
        subtree_params.dd_rank = subtree_rank;
        subtree_params.ntime = T;
        subtree_params.nspec = S;

        // For some reason, GCC complains about 'subtree_params.subband_counts = {1};'
        // so we use push_back() instead. This only happens if 'subtree_params' is
        // constructed inside the for-loop (?!)
        subtree_params.subband_counts.push_back(1);

        subtrees[f] = make_shared<ReferenceTree> (subtree_params);
    }
    
    // Make a "clone" of tree_with_subbands.final_lagbuf (independent persistent_state).
    shared_ptr<ReferenceLagbuf> lb = tree_with_subbands.final_lagbuf;
    shared_ptr<ReferenceLagbuf> local_lagbuf = make_shared<ReferenceLagbuf> (lb->lags, lb->ntime);

    // FIXME should test strides! ('buf' and 'out' only)
    Array<float> in({B,A,Din,T*S}, af_uhost | af_zero);
    Array<float> buf({B,A,Din,T*S}, dd_strides, af_uhost | af_zero);
    Array<float> out({B,Dpf,M,T*S}, out_strides, af_uhost | af_zero);

    // For subtrees.
    Array<float> buf2({B,A,Din,T*S}, af_uhost | af_zero);
    Array<float> out2({B,Dpf,M,T*S}, af_uhost | af_zero);
    Array<float> sb_empty;
    
    for (long ichunk = 0; ichunk < nchunks; ichunk++) {
        in.randomize();
        // in.set_zero();
        // in.at({0,0,0,0}) = 1.0f;

        // Call ReferenceTree.dedisperse().
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

            subtrees[f]->dedisperse(dd, sb_empty);

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
                    Array<float> out2_slice = out2.slice(1, dpf); 
                    out2_slice = out2_slice.slice(1, m);

                    out2_slice.fill(dd_slice);
                }
            }
        }

        local_lagbuf->apply_lags(out2);

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
