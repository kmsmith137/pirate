#include "../include/pirate/PeakFindingKernel.hpp"
#include "../include/pirate/BumpAllocator.hpp"
#include "../include/pirate/inlines.hpp"
#include "../include/pirate/utils.hpp"
#include "../include/pirate/varmap.hpp"   // PfVarianceConvolver

#include <mutex>
#include <sstream>
#include <iomanip>
#include <unordered_map>
#include <ksgpu/Array.hpp>
#include <ksgpu/cuda_utils.hpp>

using namespace std;
using namespace ksgpu;

namespace pirate {
#if 0
}  // editor auto-indent
#endif



// -------------------------------------------------------------------------------------------------
//
// PeakFindingKernelParams


void PeakFindingKernelParams::validate() const
{
    FrequencySubbands::validate_subband_counts(subband_counts);
    
    // The argmax token gives 'm' and 'mu' a byte each (see the top of PeakFindingKernel.hpp),
    // so each needs its own bound. This is the only place the token's bit budget is stated in
    // code. Both have room to spare: validate_subband_counts() already bounds fs.M by 114
    // (max_peak_finding_rank = 4), and xdm_rank cannot exceed 4 either, since the peak-finder
    // a cdd2 kernel builds has pf_rank = dd_rank1 = pf_rank_tree + xdm_rank.
    xassert_ge(xdm_rank, 0);
    xassert_lt(FrequencySubbands(subband_counts).M, 256L);
    xassert_lt(xdm_rank, 8L);

    // Check that everything is initialized.
    xassert(max_kernel_width > 0);
    xassert(beams_per_batch > 0);
    xassert(total_beams > 0);
    xassert(ndm_out > 0);
    xassert(ndm_wt > 0);
    xassert(nt_out > 0);
    xassert(nt_in > 0);
    xassert(nt_wt > 0);

    xassert(is_power_of_two(max_kernel_width));
    xassert(is_power_of_two(ndm_out));
    xassert(is_power_of_two(ndm_wt));

    xassert_divisible(total_beams, beams_per_batch);
    xassert_divisible(ndm_out, ndm_wt);
    xassert_divisible(nt_in, nt_out);
    xassert_divisible(nt_out, nt_wt);

    // The nt_* members don't need to be powers of two, but the downsampling
    // factors which relate them do need to be power of two.

    xassert(is_power_of_two(xdiv(ndm_out, ndm_wt)));
    xassert(is_power_of_two(xdiv(nt_in, nt_out)));
    xassert(is_power_of_two(xdiv(nt_out, nt_wt)));

    // Kernels currently assume that the input spans an integer number
    // of GPU cache lines.

    long simd_width = xdiv(32, dtype.nbits);
    xassert_divisible(nt_in, 32 * simd_width);
}


// See the declaration in PeakFindingKernel.hpp for what Dcore means and who owns it.
void validate_dcore(long Dcore, long Dout)
{
    xassert(Dcore > 0);
    xassert(is_power_of_two(Dcore));
    xassert_divisible(Dout, Dcore);
}


// -------------------------------------------------------------------------------------------------
//
// GpuPfWeightLayout


void GpuPfWeightLayout::validate() const
{
    Dtype fp32 = Dtype::native<float> ();
    Dtype fp16 = Dtype::native<__half> ();

    xassert((dtype == fp32) || (dtype == fp16));
    xassert(N > 0);
    xassert(P > 0);

    xassert(is_power_of_two(Pinner));
    xassert(is_power_of_two(Tinner));
    xassert(Pouter == (P+Pinner-1)/Pinner);   // round up
    xassert(touter_byte_stride >= Pouter * N * Tinner * xdiv(dtype.nbits,8));
    xassert_divisible(touter_byte_stride, constants::bytes_per_gpu_cache_line);
}


vector<long> GpuPfWeightLayout::get_shape(long nbeams, long ndm_wt, long nt_wt) const
{
    long Touter = xdiv(nt_wt, Tinner);   // must divide evenly
    return { nbeams, ndm_wt, Touter, Pouter, N, Tinner, Pinner };
}

vector<long> GpuPfWeightLayout::get_strides(long nbeams, long ndm_wt, long nt_wt) const
{
    long Touter = xdiv(nt_wt, Tinner);   // must divide evenly
    long S = xdiv(touter_byte_stride * 8, dtype.nbits);
    return { ndm_wt*Touter*S, Touter*S, S, N*Tinner*Pinner, Tinner*Pinner, Pinner, 1 };
}

void GpuPfWeightLayout::to_gpu(Array<void> &dst, const Array<float> &src) const
{
    this->validate();
    
    if (src.ndim != 5) {
        stringstream ss;
        ss << "GpuPfWeightLayout::to_gpu(): expected shape (nbeams, ndm_wt, nt_wt, P, N), got " << src.shape_str();
        throw runtime_error(ss.str());
    }

    xassert_eq(src.shape[3], P);
    xassert_eq(src.shape[4], N);

    long nbeams = src.shape[0];
    long ndm_wt = src.shape[1];
    long nt_wt = src.shape[2];
    long Touter = xdiv(nt_wt, Tinner);   // must divide evenly
    
    vector<long> shape = this->get_shape(nbeams, ndm_wt, nt_wt);
    vector<long> strides = this->get_strides(nbeams, ndm_wt, nt_wt);

    // Assert that 'dst' is on the GPU with the expected shape and strides.
    xassert(dst.on_gpu());
    xassert(dst.dtype == dtype);
    xassert(dst.shape_equals(shape));
    xassert(dst.strides_equal(strides));

    // Note: code below is poorly optimized! (Intended for unit tests.)

    // On host, dtype=float32, GPU shape, contiguous strides.
    Array<float> tmp(shape, af_rhost | af_zero);

    for (long b = 0; b < nbeams; b++) {
        for (long dm_wt = 0; dm_wt < ndm_wt; dm_wt++) {
            for (long touter = 0; touter < Touter; touter++) {
                for (long pouter = 0; pouter < Pouter; pouter++) {
                    for (long n = 0; n < N; n++) {
                        for (long tinner = 0; tinner < Tinner; tinner++) {
                            for (long pinner = 0; pinner < Pinner; pinner++) {
                                long tw = touter*Tinner + tinner;
                                long p = min(pouter*Pinner + pinner, P-1);

                                float w = src.at({b,dm_wt,tw,p,n});
                                tmp.at({b,dm_wt,touter,pouter,n,tinner,pinner}) = w;
                            }
                        }
                    }
                }
            }
        }
    }

    Array<void> tmp2 = tmp.convert(dtype);
    dst.fill(tmp2);  // copy CPU->GPU
}


Array<void> GpuPfWeightLayout::to_gpu(const Array<float> &src) const
{
    this->validate();
    
    if (src.ndim != 5) {
        stringstream ss;
        ss << "GpuPfWeightLayout::to_gpu(): expected shape (nbeams, ndm_wt, nt_wt, P, N), got " << src.shape_str();
        throw runtime_error(ss.str());
    }

    xassert_eq(src.shape[3], P);
    xassert_eq(src.shape[4], N);

    long nbeams = src.shape[0];
    long ndm_wt = src.shape[1];
    long nt_wt = src.shape[2];
    
    vector<long> shape = this->get_shape(nbeams, ndm_wt, nt_wt);
    vector<long> strides = this->get_strides(nbeams, ndm_wt, nt_wt);

    // Allocate GPU array with non-contiguous touter-stride.
    Array<void> dst(dtype, shape, strides, af_gpu | af_zero);

    this->to_gpu(dst, src);
    return dst;
}


// -------------------------------------------------------------------------------------------------
//
// ReferencePeakFindingKernel


// The argmax token carries the two halves of this class's multiplet index in separate bytes:
// m = (m_ext >> K) at bit 16, and mu = (m_ext & (2^K - 1)) at bit 24. This helper is the
// whole of that convention on the reference side; the generated GPU kernels emit the same
// layout (see PeakFinder._m_field_expr() in the cuda generator), and
// DedispersionPlan::decode_argmax() reads it. (eval_tokens() below reads the two fields
// directly, since it range-checks them separately.)

static inline uint _m_fields(long m_ext, long K)
{
    return (uint((m_ext >> K) & 0xff) << 16) | (uint(m_ext & ((1L << K) - 1)) << 24);
}


ReferencePeakFindingKernel::ReferencePeakFindingKernel(const PeakFindingKernelParams &params_,
                                                       long Dcore_) :
    params(params_), fs(params_.subband_counts), Dcore(Dcore_)
{
    params.validate();
    validate_dcore(Dcore, xdiv(params.nt_in, params.nt_out));

    const PeakFindingKernelParams &p = params;
    long B = p.beams_per_batch;
    long D = p.ndm_out;
    long Wmax = p.max_kernel_width;
    long M = fs.M;

    this->K = p.xdm_rank;
    this->E = pow2(K);
    this->M_ext = fs.M << K;

    this->nbatches = xdiv(p.total_beams, p.beams_per_batch);
    this->nprofiles = 3 * integer_log2(p.max_kernel_width) + 1;
    this->Dout = xdiv(p.nt_in, p.nt_out);
    this->tpad = max(2*Wmax, 4L);
    this->pstate = Array<float> ({p.total_beams, p.ndm_out, E*M, tpad}, af_uhost | af_zero);
    this->num_levels = max(integer_log2(Wmax), 1);

    this->tmp_dt.resize(num_levels);
    this->tmp_nt.resize(num_levels);
    this->tmp_iout.resize(num_levels);
    this->tmp_nout.resize(num_levels);
    this->tmp_sout.resize(num_levels);
    this->tmp_arr.resize(num_levels);
    
    for (long l = 0; l < num_levels; l++) {
        long dt = min(Dcore, pow2(l));
        long nt = xdiv(p.nt_in + tpad - pow2(l), dt) + 1;

        tmp_dt[l] = dt;
        tmp_nt[l] = nt;
        tmp_nout[l] = xdiv(Dout, dt);
        tmp_sout[l] = xdiv(pow2(l), dt);
        tmp_arr[l] = Array<float> ({B,D,E*M,nt}, af_uhost | af_zero);

        // To see that this is correct, note that the "base" time sample ends at
        // time dt, and has length 2^l.
        tmp_iout[l] = xdiv(tpad + dt - pow2(l), dt);
    }
}


// helper for ReferencePeakFindingKernel::apply(): the (maxval, argmax) max-reduce.
static inline void _update_pf(float &maxval, uint &argmax, float val, uint token)
{
    argmax = (val > maxval) ? token : argmax;
    maxval = std::max(maxval, val);
}


void ReferencePeakFindingKernel::apply(
    ksgpu::Array<float> &out_max,      // shape (beams_per_batch, ndm_out, nt_out)
    ksgpu::Array<uint> &out_argmax,    // shape (beams_per_batch, ndm_out, nt_out)
    const ksgpu::Array<float> &in,     // shape (beams_per_batch, ndm_out << K, M, nt_in)
    const ksgpu::Array<float> &wt,     // shape (beams_per_batch, ndm_wt, nt_wt, nprofiles, N)
    long ibatch, bool debug)
{
    const PeakFindingKernelParams &p = params;
    xassert_shape_eq(out_max, ({p.beams_per_batch, p.ndm_out, p.nt_out}));
    xassert_shape_eq(out_argmax, ({p.beams_per_batch, p.ndm_out, p.nt_out}));
    xassert_shape_eq(in, ({p.beams_per_batch, p.ndm_out << K, fs.M, p.nt_in}));
    xassert_shape_eq(wt, ({p.beams_per_batch, p.ndm_wt, p.nt_wt, nprofiles, fs.N}));
 
    xassert(out_max.on_host());
    xassert(out_argmax.on_host());
    xassert(in.on_host());
    xassert(wt.on_host());

    xassert_eq(ibatch, expected_ibatch);
    expected_ibatch = (ibatch + 1) % nbatches;

    // ---- _init_tmp_arrays() logic starts here ----

    long nt_in = params.nt_in;
    long B = params.beams_per_batch;
    long D = params.ndm_out;
    long b0 = ibatch * B;
    long M = fs.M;

    long t1 = min(tpad, nt_in);  // this part of 'pstate' is filled from 'in'
    long t0 = tpad - t1;         // this part of 'pstate' is filled from pstate

    xassert(in.get_ncontig() >= 1);

    // Fill l=0 (with 'in' + 'pstate' wraparound).
    //
    // This is also where the input's extra DM bits are folded into the multiplet axis: input
    // DM row ((d << K) | mu) and multiplet m land at tmp_arr index em = mu*M + m. See the
    // tmp_arr comment in PeakFindingKernel.hpp for why the ordering is (mu, m) and not
    // (m, mu).

    for (long b = 0; b < B; b++) {
        for (long d = 0; d < D; d++) {
            for (long mu = 0; mu < E; mu++) {
                for (long m = 0; m < M; m++) {
                    long em = mu*M + m;
                    float *dst = &tmp_arr[0].at({b,d,em,0});          // length (nt_in+tpad)
                    float *ps = &pstate.at({b0+b,d,em,0});            // length (tpad)
                    const float *src = &in.at({b,(d << K)|mu,m,0});   // length (nt_in)

                    for (long t = 0; t < tpad; t++)
                        dst[t] = ps[t];
                    for (long t = 0; t < nt_in; t++)
                        dst[t + tpad] = src[t];

                    for (long t = 0; t < t0; t++)
                        ps[t] = ps[t + nt_in];
                    for (long t = 0; t < t1; t++)
                        ps[t + t0] = src[t + nt_in - t1];
                }
            }
        }
    }

    // Downsample l -> (l+1)

    for (long l = 0; l < num_levels-1; l++) {
        long nsrc = tmp_nt.at(l);
        long ndst = tmp_nt.at(l+1);
        long r = xdiv(tmp_dt[l+1], tmp_dt[l]);  // ratio of step sizes
        long s = xdiv(pow2(l), tmp_dt[l]);      // spacing between logically contiguous samples in source

        xassert_eq(r*(ndst-1) + s, nsrc-1);
        
        // The multiplet axis is a pure spectator here, so it can be looped over flat.
        for (long b = 0; b < B; b++) {
            for (long d = 0; d < D; d++) {
                for (long em = 0; em < E*M; em++) {
                    float *dst = &tmp_arr.at(l+1).at({b,d,em,0});
                    float *src = &tmp_arr.at(l).at({b,d,em,0});
                    
                    for (long t = 0; t < ndst; t++)
                        dst[t] = src[r*t] + src[r*t + s];
                }
            }
        }
    }

    // ---- _peak_find() logic starts here ----

    long P = nprofiles;
    long N = fs.N;

    long Wds = xdiv(params.ndm_out, params.ndm_wt);  // downsampling factor ndm_out -> ndm_wt
    long Tds = xdiv(params.nt_out, params.nt_wt);    // downsampling factor nt_out -> nt_wt
    long nt_out = params.nt_out;

    xassert_shape_eq(out_max, ({B,D,nt_out}));
    xassert_shape_eq(out_argmax, ({B,D,nt_out}));
    xassert_shape_eq(wt, ({B, params.ndm_wt, params.nt_wt, nprofiles, fs.N}));
    xassert(wt.get_ncontig() >= 2);  // (p,n) must be contiguous

    for (long b = 0; b < B; b++) {
        for (long d = 0; d < D; d++) {
            for (long tout = 0; tout < nt_out; tout++) {
                const float *wp = &wt.at({b,d/Wds,tout/Tds,0,0});  // shape (P,N) contiguous

                // Inner loops compute one output array element, by looping over
                // peak-finding kernels, with loop ordering (p,m,isamp).

                float maxval = -1.0e30f;
                uint argmax = ~0u;  // token

                for (long l = 0; l < num_levels; l++) {
                    float *tmp_in = &tmp_arr.at(l).at({b,d,0,0});
                    int mstr = tmp_nt[l];   // m-stride of input array
                    int dt = tmp_dt[l];     // used below when computing tokens
                    int nsamp = tmp_nout[l];    // count
                    int S = tmp_sout[l];    // spacing
                    int I = tmp_iout[l];    // base

                    // Four m-like quantities are live below, and they are not interchangeable:
                    //   m     -- compact multiplet index; the weights are per subband, so this
                    //            is what fs.m_to_n takes
                    //   mu    -- extra-DM index, 0 <= mu < E
                    //   em    -- index into tmp_arr's multiplet axis, = mu*M + m
                    //   m_ext -- the argmax token's m-field, = (m << K) | mu

                    for (int mu = 0; mu < E; mu++) {
                      for (int m = 0; m < M; m++) {
                        int em = mu*M + m;
                        int n = fs.m_to_n[m];
                        uint m_ext = (uint(m) << K) | uint(mu);
                        float w0 = l ? 0.0f : wp[n];      // p = 0 (only for l=0)
                        float w1 = wp[(3*l+1)*N + n];     // p = (3*l+1)
                        float w2 = wp[(3*l+2)*N + n];     // p = (3*l+2)
                        float w3 = wp[(3*l+3)*N + n];     // p = (3*l+3)

                        // Each iteration of the isamp-loop corresponds to one time sample in the
                        // tmp[l] array, or (dt) time samples in the original input array.

                        for (int isamp = 0; isamp < nsamp; isamp++) {
                            float x0 = tmp_in[em*mstr + I + tout*nsamp + isamp - 3*S];
                            float x1 = tmp_in[em*mstr + I + tout*nsamp + isamp - 2*S];
                            float x2 = tmp_in[em*mstr + I + tout*nsamp + isamp - S];
                            float x3 = tmp_in[em*mstr + I + tout*nsamp + isamp];

                            uint token0 = _m_fields(m_ext, K) | (isamp*dt);  // (m,mu,isamp), not p
                            uint token1 = token0 | ((3*l+1) << 8);    // include p=3*l+1
                            uint token2 = token0 | ((3*l+2) << 8);    // include p=3*l+2
                            uint token3 = token0 | ((3*l+3) << 8);    // include p=3*l+3

                            float y0 = x3;
                            float y1 = (x2 + x3);
                            float y2 = (0.5f*x1 + x2 + 0.5f*x3);
                            float y3 = (0.5f*x0 + x1 + x2 + 0.5f*x3);

                            if (l == 0)
                                _update_pf(maxval, argmax, w0*y0, token0);

                            if (P > 1) {
                                _update_pf(maxval, argmax, w1*y1, token1);
                                _update_pf(maxval, argmax, w2*y2, token2);
                                _update_pf(maxval, argmax, w3*y3, token3);
                            }

                            if (debug && (b == 0) && (d==0) && (tout==2)) {
                                cout << "cpu peak-finder: b=" << b << ", d=" << d << ", tout=" << tout 
                                     << ", level=" << l << ", m=" << m << ", mu=" << mu << ", isamp=" << isamp << "\n";

                                if (l == 0)
                                    cout << "   p=0" << " -> (w=" << w0 << ", y=" << y0 << ", w*y=" << (w0*y0) << endl;
                                
                                if (P > 1) {
                                    cout << "   p=" << (3*l+1) << " -> (w=" << w1 << ", y=" << y1 << ", w*y=" << (w1*y1) << endl;
                                    cout << "   p=" << (3*l+2) << " -> (w=" << w2 << ", y=" << y2 << ", w*y=" << (w2*y2) << endl;
                                    cout << "   p=" << (3*l+3) << " -> (w=" << w3 << ", y=" << y3 << ", w*y=" << (w3*y3) << endl;
                                }
                            }
                        }
                      }
                    }
                }

                out_max.at({b,d,tout}) = maxval;
                out_argmax.at({b,d,tout}) = argmax;
            }
        }
    }
}


// Note that an 'in' array is not an argument -- this function uses the contents of 'tmp_arr'.
void ReferencePeakFindingKernel::eval_tokens(Array<float> &out_max, const Array<uint> &in_tokens, const Array<float> &wt)
{
    long B = params.beams_per_batch;
    long D = params.ndm_out;
    long M = fs.M;
    long N = fs.N;
    long P = nprofiles;
    long Wds = xdiv(params.ndm_out, params.ndm_wt);  // downsampling factor ndm_out -> ndm_wt
    long Tds = xdiv(params.nt_out, params.nt_wt);    // downsampling factor nt_out -> nt_wt
    long nt_out = params.nt_out;

    xassert_shape_eq(out_max, ({B,D,nt_out}));
    xassert_shape_eq(in_tokens, ({B,D,nt_out}));
    xassert_shape_eq(wt, ({B, params.ndm_wt, params.nt_wt, P, N}));
    xassert(wt.get_ncontig() >= 2);  // (p,n) must be contiguous

    xassert(out_max.on_host());
    xassert(in_tokens.on_host());
    xassert(wt.on_host());

    // Loop are over elements of (b,d,tout) of the 'out_max' and 'in_tokens' arrays.
    for (long b = 0; b < B; b++) {
        for (long d = 0; d < D; d++) {
            for (long tout = 0; tout < nt_out; tout++) {
                uint token = in_tokens.at({b,d,tout});

                // Token parsing starts here.
                // Reminder: token = (t) | (p << 8) | (m << 16) | (mu << 24), and this class's
                // own multiplet index is m_ext = (m << K) | mu.

                long m  = (token >> 16) & 0xffu;
                long mu = (token >> 24) & 0xffu;
                long p = (token >> 8) & 0xffu;
                long t = (token & 0xffu);

                // m and mu are independently bounded now that they occupy separate bytes:
                // neither range check implies the other.
                if (m >= M)
                    throw _bad_token(token, "m out of range");
                if (mu >= E)
                    throw _bad_token(token, "mu out of range");
                if ((p < 0) || (p >= P))
                    throw _bad_token(token, "p out of range");
                if ((t < 0) || (t >= Dout))
                    throw _bad_token(token, "t out of range");

                long em = mu*M + m;   // index into tmp_arr (see PeakFindingKernel.hpp)

                // p = 3*l+q, where l is the "level".
                long l = p ? ((p-1)/3) : 0;
                long q = p - 3*l;

                // t = isamp*dt
                long dt = tmp_dt.at(l);
                long isamp = t / dt;

                if (t != isamp*dt)
                    throw _bad_token(token, "t is not divisible by dt");

                // Token parsing (token -> (m,mu,isamp,p)) ends here!

                long n = fs.m_to_n.at(m);
                float w = wt.at({b, d/Wds, tout/Tds, p, n});

                int nsamp = tmp_nout[l];       // count
                int S = tmp_sout[l];       // spacing
                int I = tmp_iout[l];       // base

                float x0 = tmp_arr.at(l).at({b, d, em, I + tout*nsamp + isamp - 3*S});
                float x1 = tmp_arr.at(l).at({b, d, em, I + tout*nsamp + isamp - 2*S});
                float x2 = tmp_arr.at(l).at({b, d, em, I + tout*nsamp + isamp - S});
                float x3 = tmp_arr.at(l).at({b, d, em, I + tout*nsamp + isamp});

                if (q == 0)
                    out_max.at({b,d,tout}) = w * x3;
                else if (q == 1)
                    out_max.at({b,d,tout}) = w * (x2 + x3);
                else if (q == 2)
                    out_max.at({b,d,tout}) = w * (0.5f*x1 + x2 + 0.5f*x3);
                else if (q == 3)
                    out_max.at({b,d,tout}) = w * (0.5f*x0 + x1 + x2 + 0.5f*x3);
                else
                    throw _bad_token(token, "bad value of q, this should never happen");

#if 0
                if ((b==0) && (d==0) && (tout==1)) {
                    cout << "\neval_tokens(): (b=" << b << ", d=" << d << ", tout=" << tout << ")"
                         << " -> " << hex_str(token)
                         << " -> (m=" << m << ", mu=" << mu << ", p=" << p << ", t=" << t << ", l=" << l << ", q=" << q << ")"
                         << " -> (w=" << w << ", x0=" << x0 << ", x1=" << x1 << ", x2=" << x2 << ", x3=" << x3 << ")"
                         << " -> " << out_max.at({b,d,tout}) << endl;

                    cout << "  wt.at(" << b << "," << (d/Wds) << "," << (tout/Tds) << "," << p << "," << n << ")"
                         << " = " << wt.at({b,d/Wds,tout/Tds,p,n}) << endl;

                    for (int i = 0; i < 4; i++)
                        cout << "  tmp_arr.at(" << l << ").at(" << b << "," << d << "," << em << "," << (I + tout*nsamp + isamp + (i-3)*S) << ")"
                             << " = " << tmp_arr.at(l).at({b, d, em, I + tout*nsamp + isamp + (i-3)*S}) << endl;

                    cout << "    at level l: tpad=" << tpad << ", dt=" << tmp_dt.at(l) << ", nsamp=" << nsamp << ", S=" << S << ", I=" << I << endl;
                }
#endif
            }
        }
    }
}


std::runtime_error ReferencePeakFindingKernel::_bad_token(uint token, const char *why)
{
    stringstream ss;
    ss << "ReferencePeakFindingKernel::eval_tokens(): bad token " << hex_str(token) << " (" << why << ")";
    return runtime_error(ss.str());
}


// Make a mean-zero input array for testing.
// Returns shape (nbeams_per_batch, ndm_out << K, fs.M, nt_in)
Array<float> ReferencePeakFindingKernel::make_random_input_array()
{
    long B = params.beams_per_batch;
    long D = params.ndm_out << K;
    long T = params.nt_in;
    long M = fs.M;

    Array<float> ret({B,D,M,T}, af_rhost);
    {
        std::mt19937 &rng = ksgpu::default_rng();
        for (long i = 0; i < ret.size; i++)
            ret.data[i] = rand_uniform(-1.0f, 1.0f, rng);
    }

    return ret;
}


// fill_host_weights(): build peak-finding weights. The per-(subband, dm, profile) base_weights
// are set one of two ways:
//   - 'variances' non-empty, shape (ndm_wt, N, nprofiles) (double):
//         base_weights[d,n,p] = 1/sqrt(variances[d,n,p])
//   - 'variances' empty: "bare-kernel" weights for unit-variance input. Feed a single unit
//         sample through the peak-finding convolver to get the per-profile output variance
//         pf_var[p] (the zero-lag autocorrelation of kernel p), and broadcast over (n,d):
//         base_weights[d,n,p] = 1/sqrt(pf_var[p]). Appropriate for testing a bare peak-finding
//         or cdd2 kernel (whose input is unit-variance).
//
//   out shape = (beams_per_batch, ndm_wt, nt_wt, nprofiles, N)   (float)
//
// Then fill 'out':
//   randomize=true:  out[b,d,t,p,n] = x * base_weights[d,n,p], where 'x' is a sparse random
//                    value -- per (b,d,t) we draw an "occupancy" p0, then for each (n,p) the
//                    weight is zero with probability ~(1-p0), else uniform in [0,1).
//   randomize=false: out[b,d,t,p,n] = base_weights[d,n,p] (no random multiplier).

void PeakFindingKernelParams::fill_host_weights(Array<float> &out, const Array<double> &variances, bool randomize) const
{
    const long B = beams_per_batch;
    const long D = ndm_wt;
    const long T = nt_wt;
    const long P = 3 * integer_log2(max_kernel_width) + 1;   // nprofiles
    const FrequencySubbands fs(subband_counts);
    const long N = fs.N;

    const bool bare = (variances.size == 0);

    xassert_shape_eq(out, ({B,D,T,P,N}));
    xassert(out.on_host());
    xassert(out.is_fully_contiguous());
    
    if (!bare) {
        xassert_shape_eq(variances, ({D,N,P}));
        xassert(variances.on_host());
        xassert(variances.is_fully_contiguous());
    }

    // Phase 1: base_weights[d,n,p] (float, (D,N,P)).
    Array<float> base_weights({D,N,P}, af_uhost);
    if (bare) {
        // "Bare-kernel" weights for unit-variance input: per-profile output variance for a
        // single unit sample (pf_var[p] = zero-lag autocorrelation of kernel p), broadcast
        // over (subband n, dm d).
        PfVarianceConvolver conv;
        double x = 1.0;
        std::vector<double> pf_var(P);
        conv.variance(&x, /*S=*/1, /*nt=*/1, P, pf_var.data());
        float *bp = base_weights.data;        // (D,N,P)
        for (long d = 0; d < D; d++)
            for (long n = 0; n < N; n++)
                for (long p = 0; p < P; p++)
                    bp[(d*N + n)*P + p] = rsqrtf(pf_var[p]);
    }
    else {
        // base_weights[d,n,p] = rsqrtf(variances[d,n,p]).
        const double *vp = variances.data;    // (D,N,P) contiguous, double
        float *bp = base_weights.data;        // (D,N,P) contiguous, float
        for (long i = 0; i < D*N*P; i++) {
            xassert(vp[i] > 0.0);
            bp[i] = rsqrtf(vp[i]);
        }
    }

    // Phase 2: fill 'out'. The (p,n) block of 'out' is contiguous (p outer, n inner), so
    // we write it sequentially through 'op'. For a fixed d, base_weights is a tiny (N*P)
    // array that stays in cache, so the strided read base_weights[d,n,p] = bw_d[n*P+p] is cheap.
    std::mt19937 &rng = ksgpu::default_rng();
    const float *bw = base_weights.data;      // (D,N,P)
    float *op = out.data;                      // (B,D,T,P,N), fully contiguous

    for (long b = 0; b < B; b++) {
        for (long d = 0; d < D; d++) {
            const float *bw_d = bw + d*N*P;            // base_weights[d], layout [n][p]
            for (long t = 0; t < T; t++) {
                // 'randomize' is loop-invariant; branch here rather than per element.
                if (randomize) {
                    float p0 = rand_uniform(0.01f, 1.1f, rng);
                    for (long p = 0; p < P; p++) {
                        for (long n = 0; n < N; n++) {
                            float r = rand_uniform(0.0f, 1.0f, rng);
                            float x = (r < p0) ? rand_uniform(0.0f, 1.0f, rng) : 0.0f;
                            *op++ = x * bw_d[n*P + p];
                        }
                    }
                }
                else {
                    for (long p = 0; p < P; p++)
                        for (long n = 0; n < N; n++)
                            *op++ = bw_d[n*P + p];
                }
            }
        }
    }
}


// -------------------------------------------------------------------------------------------------
//
// GpuPeakFindingKernel


// File-local.
static GpuPeakFindingKernel::RegistryKey _make_registry_key(const PeakFindingKernelParams &pf_params)
{
    GpuPeakFindingKernel::RegistryKey key;
    key.dtype = pf_params.dtype;
    key.subband_counts = pf_params.subband_counts;
    key.Dout = xdiv(pf_params.nt_in, pf_params.nt_out);
    key.Wmax = pf_params.max_kernel_width;

    // Recall the definition of Tinner (used for weight layout, see comments in
    // cuda_generator.PeakFinder.py):
    //
    //   Tinner = max(32*SW/nt_in_per_wt, 1)

    long SW = xdiv(32, pf_params.dtype.nbits);      // simd width
    long nt_in_per_wt = xdiv(pf_params.nt_in, pf_params.nt_wt);
    key.Tinner = (nt_in_per_wt < 32*SW) ? xdiv(32*SW, nt_in_per_wt) : 1;

    return key;
}


GpuPeakFindingKernel::GpuPeakFindingKernel(const PeakFindingKernelParams &params_) :
    params(params_), fs(params_.subband_counts)
{
    params.validate();

    if (params.xdm_rank != 0)
        throw runtime_error("GpuPeakFindingKernel: xdm_rank > 0 is not implemented -- a standalone"
                            " GPU peak-finder does not do the extra-DM reduction. On the GPU, K > 0"
                            " is handled by CoalescedDdKernel2, which folds the extra DMs in as it"
                            " dedisperses.");

    registry_key = _make_registry_key(params);
    registry_value = registry().get(registry_key);

    pf_weight_layout = registry_value.pf_weight_layout;
    expected_wt_shape = pf_weight_layout.get_shape(params.beams_per_batch, params.ndm_wt, params.nt_wt);
    expected_wt_strides = pf_weight_layout.get_strides(params.beams_per_batch, params.ndm_wt, params.nt_wt);
    Dcore = registry_value.Dcore;
    dtype = params.dtype;
    Dout = xdiv(params.nt_in, params.nt_out);
    nbatches = xdiv(params.total_beams, params.beams_per_batch);
    nprofiles = pf_weight_layout.P;

    // FIXME add bandwidth tracking later.
    // this->bw_per_launch.nbytes_gmem = params.beams_per_batch * isamp * xdiv(params.dtype.nbits,8);
    // this->bw_per_launch.kernel_launches = 1;

    // Compute GPU memory footprint, reflecting logic in allocate().
    long pstate_nbytes = params.total_beams * params.ndm_out * registry_value.PW32 * 4;
    resource_tracker.add_gmem_footprint("persistent_state", pstate_nbytes, true);
}


void GpuPeakFindingKernel::allocate(BumpAllocator &allocator)
{
    if (is_allocated)
        throw runtime_error("GpuPeakFindingKernel: double call to allocate()");

    if (!(allocator.aflags & af_gpu))
        throw runtime_error("GpuPeakFindingKernel::allocate(): allocator.aflags must contain af_gpu");
    if (!(allocator.aflags & af_zero))
        throw runtime_error("GpuPeakFindingKernel::allocate(): allocator.aflags must contain af_zero");

    long nbytes_before = allocator.get_nbytes_allocated();

    // Allocate persistent_state.
    std::initializer_list<long> shape = { params.total_beams, params.ndm_out, registry_value.PW32 };
    this->persistent_state = allocator.allocate_array<uint>(shape);

    long nbytes_allocated = allocator.get_nbytes_allocated() - nbytes_before;
    // cout << "GpuPeakFindingKernel: " << nbytes_allocated << " bytes allocated" << endl;
    xassert_eq(nbytes_allocated, resource_tracker.get_gmem_footprint());

    this->is_allocated = true;
}


void GpuPeakFindingKernel::launch(
    ksgpu::Array<void> &out_max,      // shape (beams_per_batch, ndm_out, nt_out)
    ksgpu::Array<uint> &out_argmax,   // shape (beams_per_batch, ndm_out, nt_out)
    const ksgpu::Array<void> &in,     // shape (beams_per_batch, ndm_out, M, nt_in)
    const ksgpu::Array<void> &wt,     // from GpuPfWeightLayout::to_gpu()
    long ibatch,                      // 0 <= ibatch < nbatches
    cudaStream_t stream)              // NULL stream is allowed, but is not the default);
{
    const PeakFindingKernelParams &p = params;

    xassert(this->is_allocated);
    xassert(out_max.dtype == dtype);
    xassert(in.dtype == dtype);
    xassert(wt.dtype == dtype);

    xassert_shape_eq(out_max, ({p.beams_per_batch, p.ndm_out, p.nt_out}));
    xassert_shape_eq(out_argmax, ({p.beams_per_batch, p.ndm_out, p.nt_out}));
    xassert_shape_eq(in, ({p.beams_per_batch, p.ndm_out, fs.M, p.nt_in}));

    // Validate 'wt' array. These checks will pass if 'wt' is the output of GpuPfWeightLayout::to_gpu().

    if (!wt.shape_equals(expected_wt_shape)) {
        stringstream ss;
        ss << "GpuPeakFindingKernel::launch(): wt.shape=" << wt.shape_str()
           << ", expected_wt_shape=" << ksgpu::tuple_str(expected_wt_shape);
        throw runtime_error(ss.str());
    }

    if (!wt.strides_equal(expected_wt_strides)) {
        stringstream ss;
        ss << "GpuPeakFindingKernel::launch(): wt.strides=" << wt.stride_str()
           << ", expected_wt_strides=" << ksgpu::tuple_str(expected_wt_strides);
        throw runtime_error(ss.str());
    }

    xassert(out_max.is_fully_contiguous());
    xassert(out_argmax.is_fully_contiguous());
    xassert(in.is_fully_contiguous());
    // Weights array is not fully contiguous -- see above.

    xassert(out_max.on_gpu());
    xassert(out_argmax.on_gpu());
    xassert(in.on_gpu());
    xassert(wt.on_gpu());

    xassert(ibatch == expected_ibatch);
    expected_ibatch = (ibatch + 1) % nbatches;

    long s = (nprofiles > 0) ? (ibatch * p.beams_per_batch * persistent_state.strides[0]) : 0;
    uint *pstate = persistent_state.data + s;

    // FIXME using 1 warp/threadblock for now! Not totally trivial to fix.
    uint nwarps = p.beams_per_batch * p.ndm_out;
    dim3 nblocks = { nwarps, 1, 1 };
    dim3 nthreads = { 32, 1, 1 };

    long ndm_out_per_wt = xdiv(p.ndm_out, p.ndm_wt);
    long nt_in_per_wt = xdiv(p.nt_in, p.nt_wt);

    // cuda_kernel(const void *in, void *out_max, uint *out_argmax, const void *wt, void *pstate, uint nt_in, uint ndm_out_per_wt, uint nt_in_per_wt)
    registry_value.cuda_kernel <<< nblocks, nthreads, 0, stream >>> 
       (in.data, out_max.data, out_argmax.data, wt.data, pstate, p.nt_in, ndm_out_per_wt, nt_in_per_wt);

    CUDA_PEEK("pf kernel launch");
}


// Static member function.
// If short_circuit=true, then we run some ReferencePeakFindingKernel tests, 
// but don't test the GPU peak-finder.
void GpuPeakFindingKernel::test_random(bool short_circuit)
{
    RegistryKey key = registry().get_random_key();
    long simd_width = xdiv(32, key.dtype.nbits);
    long Tinner = key.Tinner;

    long nt_in_per_wt = (Tinner > 1) ? xdiv(32*simd_width,Tinner) : ((32 * simd_width) << rand_int(0,3));
    long nt_in_divisor = max(32*simd_width, nt_in_per_wt);

    auto v = ksgpu::random_integers_with_bounded_product(6, 200000 / (nt_in_divisor));
    long nchunks = v[0];
    long nt_in_per_chunk = nt_in_divisor * v[1];
    long beams_per_batch = v[2];
    long total_beams = v[2] * v[3];
    long ndm_wt = round_down_to_power_of_two(v[4]);
    long ndm_out = ndm_wt * round_down_to_power_of_two(v[5]);

    long nt_out_per_chunk = xdiv(nt_in_per_chunk, key.Dout);
    long nt_wt_per_chunk = xdiv(nt_in_per_chunk, nt_in_per_wt);

    PeakFindingKernelParams params_small;
    params_small.subband_counts = key.subband_counts;
    params_small.dtype = key.dtype;
    params_small.max_kernel_width = key.Wmax;
    params_small.beams_per_batch = beams_per_batch;
    params_small.total_beams = total_beams;
    params_small.ndm_out = ndm_out;
    params_small.ndm_wt = ndm_wt;
    params_small.nt_in = nt_in_per_chunk;
    params_small.nt_out = nt_out_per_chunk;
    params_small.nt_wt = nt_wt_per_chunk;

    params_small.validate();

    PeakFindingKernelParams params_large;
    params_large.subband_counts = key.subband_counts;
    params_large.dtype = key.dtype;
    params_large.max_kernel_width = key.Wmax;
    params_large.beams_per_batch = total_beams;
    params_large.total_beams = total_beams;
    params_large.ndm_out = ndm_out;
    params_large.ndm_wt = ndm_wt;
    params_large.nt_in = nchunks * nt_in_per_chunk;
    params_large.nt_out = nchunks * nt_out_per_chunk;
    params_large.nt_wt = nchunks * nt_wt_per_chunk;
    params_large.validate();

    GpuPeakFindingKernel gpu_kernel(params_small);   // just test constructor for now

    // Nothing else ever hands this class xdm_rank > 0, so its rejection of that case is
    // checked here rather than left untested.
    {
        PeakFindingKernelParams p = params_small;
        p.xdm_rank = 1;
        bool threw = false;
        try { GpuPeakFindingKernel rejected(p); }
        catch (const std::exception &) { threw = true; }
        xassert(threw);
    }

    // The reference kernels must emit the same tokens as the GPU kernel, so they take its
    // compiled-in Dcore. (params_large has the same registry key: nchunks scales nt_* together.)
    ReferencePeakFindingKernel ref_kernel_small(params_small, gpu_kernel.Dcore);
    ReferencePeakFindingKernel ref_kernel_large(params_large, gpu_kernel.Dcore);

    cout << "GpuPeakFindingKernel::test():"
         << " dtype=" << key.dtype.str() 
         << ", subbands=" << ksgpu::tuple_str(key.subband_counts)
         << ", Wmax=" << key.Wmax
         << ", Dcore=" << gpu_kernel.Dcore
         << ", Dout=" << key.Dout
         << ", Tinner=" << key.Tinner
         << ", M=" << gpu_kernel.fs.M
         << ", beams_per_batch=" << beams_per_batch
         << ", total_beams=" << total_beams
         << ", ndm_out=" << ndm_out
         << ", ndm_wt=" << ndm_wt
         << ", nt_in_per_chunk=" << nt_in_per_chunk
         << ", nt_out_per_chunk=" << nt_out_per_chunk
         << ", nt_wt_per_chunk=" << nt_wt_per_chunk
         << ", nchunks=" << nchunks
         << endl;
    
    long P = gpu_kernel.nprofiles;
    long N = gpu_kernel.fs.N;
    long M = gpu_kernel.fs.M;

    Array<float> cpu_in_large = ref_kernel_large.make_random_input_array();
    xassert_shape_eq(cpu_in_large, ({total_beams, ndm_out, M, nchunks * nt_in_per_chunk}));

    Array<float> cpu_wt_large({total_beams, ndm_wt, nchunks * nt_wt_per_chunk, P, N}, af_rhost | af_zero);
    params_large.fill_host_weights(cpu_wt_large, Array<double>(), /*randomize=*/true);
 
    Array<float> cpu_out_large({total_beams, ndm_out, nchunks * nt_out_per_chunk}, af_rhost | af_zero);
    Array<uint> cpu_argmax_large({total_beams, ndm_out, nchunks * nt_out_per_chunk}, af_rhost | af_zero);
    ref_kernel_large.apply(cpu_out_large, cpu_argmax_large, cpu_in_large, cpu_wt_large, 0);

    // Use eval_tokens() to get a nontrivial test of the reference peak-finder.
    // (We haven't compared the reference and GPU peak-finders yet.)
    Array<float> cpu_out2_large({total_beams, ndm_out, nchunks * nt_out_per_chunk}, af_rhost | af_zero);
    ref_kernel_large.eval_tokens(cpu_out2_large, cpu_argmax_large, cpu_wt_large);
    assert_arrays_equal(cpu_out_large, cpu_out2_large, "cpu_out_large", "cpu_out2_large", {"b","d","tout"});

    BumpAllocator allocator(af_gpu | af_zero, -1);  // dummy allocator
    gpu_kernel.allocate(allocator);

    for (long ichunk = 0; ichunk < nchunks; ichunk++) {
        long tin0 = (ichunk) * nt_in_per_chunk;
        long tin1 = (ichunk+1) * nt_in_per_chunk;
        long tout0 = (ichunk) * nt_out_per_chunk;
        long tout1 = (ichunk+1) * nt_out_per_chunk;
        long tw0 = (ichunk) * nt_wt_per_chunk;
        long tw1 = (ichunk+1) * nt_wt_per_chunk;

        for (long ibatch = 0; ibatch < xdiv(total_beams,beams_per_batch); ibatch++) {
            long b0 = ibatch * beams_per_batch;
            long b1 = (ibatch+1) * beams_per_batch;

            Array<float> cpu_in_small = cpu_in_large.slice(0, b0, b1);
            cpu_in_small = cpu_in_small.slice(3, tin0, tin1);
            cpu_in_small = cpu_in_small.clone();  // contiguous deep copy

            Array<float> cpu_wt_small = cpu_wt_large.slice(0, b0, b1);
            cpu_wt_small = cpu_wt_small.slice(2, tw0, tw1);
            cpu_wt_small = cpu_wt_small.clone();  // contiguous deep copy

            Array<float> cpu_out_small({beams_per_batch, ndm_out, nt_out_per_chunk}, af_rhost | af_zero);
            Array<uint> cpu_argmax_small({beams_per_batch, ndm_out, nt_out_per_chunk}, af_rhost | af_zero);
            ref_kernel_small.apply(cpu_out_small, cpu_argmax_small, cpu_in_small, cpu_wt_small, ibatch);

            // Use eval_tokens() to get a nontrivial test of the reference peak-finder.
            // (We haven't compared the reference and GPU peak-finders yet.)
            Array<float> cpu_out2_small({beams_per_batch, ndm_out, nt_out_per_chunk}, af_rhost | af_zero);
            ref_kernel_small.eval_tokens(cpu_out2_small, cpu_argmax_small, cpu_wt_small);
            assert_arrays_equal(cpu_out_small, cpu_out2_small, "cpu_out_small", "cpu_out2_small", {"b","d","tout"});

            Array<float> cpu_out3_small = cpu_out_large.slice(0, b0, b1);
            cpu_out3_small = cpu_out3_small.slice(2, tout0, tout1);
            assert_arrays_equal(cpu_out_small, cpu_out3_small, "cpu_out_small", "cpu_out3_small", {"b","d","tout"});

            if (short_circuit) {
                cout << "!!! short-circuiting !!!" << endl;
                continue;
            }

            Array<void> gpu_in = cpu_in_small.convert(key.dtype);
            gpu_in = gpu_in.to_gpu();

            Array<void> gpu_wt = gpu_kernel.pf_weight_layout.to_gpu(cpu_wt_small);

            Array<void> gpu_out(key.dtype, {beams_per_batch, ndm_out, nt_out_per_chunk}, af_gpu | af_zero);
            Array<uint> gpu_argmax({beams_per_batch, ndm_out, nt_out_per_chunk}, af_gpu | af_zero);
            gpu_kernel.launch(gpu_out, gpu_argmax, gpu_in, gpu_wt, ibatch, NULL);

            // Now we can test the GPU peak-finder, by comparing to the reference peak-finder.
            // The 'out_max' arrays can be compared directly.
            assert_arrays_equal(cpu_out_small, gpu_out, "cpu_out_small", "gpu_out", {"b","d","tout"});

            // We can't compare argmax arrays directly -- they can disagree due to near-ties and
            // roundoff error. Instead, we use the following two-step procedure (see more discussion
            // in PeakFindingKernel.hpp):
            //
            //    eval_tokens(gpu_argmax) -> gpu_out2  (temp array)
            //    assert_arrays_equal(cpu_out, gpu_out2)

            gpu_argmax = gpu_argmax.to_host();
            Array<float> gpu_out2({beams_per_batch, ndm_out, nt_out_per_chunk}, af_rhost | af_zero);
            ref_kernel_small.eval_tokens(gpu_out2, gpu_argmax, cpu_wt_small);

            double eps = 5.0 * key.dtype.precision();
            assert_arrays_equal(cpu_out_small, gpu_out2, "cpu_out_small", "gpu_out2", {"b","d","tout"}, eps, eps);
        }
    }
}


// -------------------------------------------------------------------------------------------------
//
// Kernel registry.


struct GpuPfRegistry : public GpuPeakFindingKernel::Registry
{
    using Key = GpuPeakFindingKernel::RegistryKey;
    using Val = GpuPeakFindingKernel::RegistryValue;

    virtual void add(const Key &key, const Val &val, bool debug) override
    {
        // Just check that all members have been initialized.
        // (In the future, I may add more argument checking here.)
        
        xassert((key.dtype == Dtype::native<float>()) || (key.dtype == Dtype::native<__half>()));
        xassert_ge(key.subband_counts.size(), 1);
        xassert(key.Tinner > 0);
        xassert(key.Dout > 0);
        xassert(key.Wmax > 0);
        
        xassert(val.cuda_kernel != nullptr);
        xassert(val.Dcore > 0);
        xassert(val.PW32 >= 0);
        
        val.pf_weight_layout.validate();
        
        // Call add() in base class.
        GpuPeakFindingKernel::Registry::add(key, val, debug);
    }
};


// Static member function
GpuPeakFindingKernel::Registry &GpuPeakFindingKernel::registry()
{
    // Instead of declaring the registry as a static global variable, we declare it as a
    // static local variable in the static member function GpuPeakFindingKernel::registry().
    // The registry will be initialized the first time that GpuPeakFindingKernel::registry()
    // is called.
    //
    // This kludge is necessary because the registry is accessed at library initialization
    // time, by callers in other source files, and source files are executed in an
    // arbitrary order.
    
    static GpuPfRegistry reg;
    return reg;  // note: thread-safe (as of c++11)
}

bool operator==(const GpuPeakFindingKernel::RegistryKey &k1, const GpuPeakFindingKernel::RegistryKey &k2)
{
    return (k1.dtype == k2.dtype)
        && (k1.subband_counts == k2.subband_counts)
        && (k1.Tinner == k2.Tinner)
        && (k1.Dout == k2.Dout)
        && (k1.Wmax == k2.Wmax);
}

ostream &operator<<(ostream &os, const GpuPeakFindingKernel::RegistryKey &k)
{
    FrequencySubbands fs(k.subband_counts);
    
    os << "GpuPeakFindingKernel(dtype=" << k.dtype
       << ", rank=" << fs.pf_rank
       << ", subband_counts=" << ksgpu::tuple_str(k.subband_counts)
       << ", Tinner=" << k.Tinner
       << ", Dout=" << k.Dout
       << ", Wmax=" << k.Wmax
       << ", N=" << fs.N
       << ", M=" << fs.M
       << ")";
    
    return os;
}

ostream &operator<<(ostream &os, const GpuPeakFindingKernel::RegistryValue &v)
{
    os << "Dcore=" << v.Dcore << ", pstate_32_bit_registers_per_warp=" << v.PW32;
    return os;
}


// -------------------------------------------------------------------------------------------------
//
// PfWeightReaderMicrokernel


struct PfWeightReaderMicrokernelRegistry : public PfWeightReaderMicrokernel::Registry
{
    using Key = PfWeightReaderMicrokernel::RegistryKey;
    using Val = PfWeightReaderMicrokernel::RegistryValue;

    virtual void add(const Key &key, const Val &val, bool debug) override
    {
        // Just check that all members have been initialized.
        // (In the future, I may add more argument checking here.)
        
        xassert((key.dtype == Dtype::native<float>()) || (key.dtype == Dtype::native<__half>()));
        xassert_ge(key.subband_counts.size(), 1);
        xassert_ge(key.Dcore, 0);
        xassert_ge(key.Tinner, 0);
        xassert_ge(key.P, 0);
        
        xassert(val.cuda_kernel != nullptr);
        xassert(val.Mouter > 0);
        xassert(val.Minner > 0);
        
        val.pf_weight_layout.validate();
        
        // Call add() in base class.
        PfWeightReaderMicrokernel::Registry::add(key, val, debug);
    }
};


// Static member function
PfWeightReaderMicrokernel::Registry &PfWeightReaderMicrokernel::registry()
{
    // Instead of declaring the registry as a static global variable, we declare it as a
    // static local variable in the static member function PfWeightReaderMicrokernel::registry().
    // The registry will be initialized the first time that PfWeightReaderMicrokernel::registry()
    // is called.
    //
    // This kludge is necessary because the registry is accessed at library initialization
    // time, by callers in other source files, and source files are executed in an
    // arbitrary order.
    
    static PfWeightReaderMicrokernelRegistry reg;
    return reg;  // note: thread-safe (as of c++11)
}

bool operator==(const PfWeightReaderMicrokernel::RegistryKey &k1, const PfWeightReaderMicrokernel::RegistryKey &k2)
{
    return (k1.dtype == k2.dtype)
        && (k1.subband_counts == k2.subband_counts)
        && (k1.Dcore == k2.Dcore)
        && (k1.Tinner == k2.Tinner)
        && (k1.P == k2.P);
}

ostream &operator<<(ostream &os, const PfWeightReaderMicrokernel::RegistryKey &k)
{
    FrequencySubbands fs(k.subband_counts);
    
    os << "PfWeightReaderMicrokernel(dtype=" << k.dtype
       << ", rank=" << fs.pf_rank
       << ", subband_counts=" << ksgpu::tuple_str(k.subband_counts)
       << ", Dcore=" << k.Dcore
       << ", Tinner=" << k.Tinner
       << ", P=" << k.P
       << ", N=" << fs.N
       << ", M=" << fs.M
       << ")";
    
    return os;
}

ostream &operator<<(ostream &os, const PfWeightReaderMicrokernel::RegistryValue &v)
{
    return os;
}


void PfWeightReaderMicrokernel::test_random()
{
    PfWeightReaderMicrokernel::RegistryKey key = PfWeightReaderMicrokernel::registry().get_random_key();
    PfWeightReaderMicrokernel::RegistryValue val = PfWeightReaderMicrokernel::registry().get(key);

    FrequencySubbands fs(key.subband_counts);
    GpuPfWeightLayout &wl = val.pf_weight_layout;
    
    Dtype dtype = key.dtype;
    int SW = xdiv(32, dtype.nbits);   // simd width
    
    int N = fs.N;
    int M = fs.M;
    int P = wl.P;
    int Dcore = key.Dcore;
    int Tinner = key.Tinner;
    
    // Choose nt_in_per_wt, nt_in.
    // If Tinner > 1, then nt_in_per_wt must equal (32*SW)/Tinner, and Tin must be a multiple of (32*SW).
    // If Tinner == 1, then nt_in_per_wt must be a multiple of (32*SW), and Tin must be a multiple of nt_in_per_wt.
    
    auto v = ksgpu::random_integers_with_bounded_product(2, 20);
    int nt_in_per_wt = (Tinner > 1) ? xdiv(32*SW,Tinner) : (32*SW*v[0]);
    int nt_in = (Tinner > 1) ? (32*SW*v[0]*v[1]) : (nt_in_per_wt*v[1]);  // number of tree samples (not used for anything)

    cout << "test_pf_weight_reader_microkernel: dtype=" << dtype
         << ", subband_counts=" << ksgpu::tuple_str(key.subband_counts)
         << ", Dcore=" << key.Dcore
         << ", P=" << key.P
         << ", Tinner=" << Tinner
         << ", nt_in_per_wt=" << nt_in_per_wt
         << ", nt_in=" << nt_in << endl;
    
    int nt_wt = xdiv(nt_in, nt_in_per_wt);     // number of time samples in weights array (input array to test kernel)
    int nt_out = xdiv(nt_in, Dcore);   // number of time samples in output array of test kernel
    int Tspec = xdiv(nt_out, nt_wt);  // number of "spectator" time samples in test kernel
    int Mpad = val.Mouter * val.Minner;
    int Ppad = wl.Pouter * wl.Pinner;    
    
    // Input array: (1,1,nt_wt,P,N), where the length-1 axes are beams and DMs.
    Array<float> in_cpu({1,1,nt_wt,P,N}, af_rhost | af_random);

    // Output array: (nt_out, Mouter*Minner, Pouter*Pinner)
    Array<float> out_cpu({nt_out,Mpad,Ppad}, af_rhost | af_zero);

    // Emulate PfWeightReader kernel on the CPU.
    for (int tw = 0; tw < nt_wt; tw++) {
        for (int tout = tw*Tspec; tout < (tw+1)*Tspec; tout++) {
            for (int mpad = 0; mpad < Mpad; mpad++) {
                int m = min(mpad, M-1);
                int n = fs.m_to_n.at(m);
                
                for (int ppad = 0; ppad < Ppad; ppad++) {
                    int p = min(ppad, P-1);
                    out_cpu.at({tout,mpad,ppad}) = in_cpu.at({0,0,tw,p,n});
                }
            }
        }
    }

    // Send input array to GPU, using GpuPfWeightLayout::to_gpu().
    Array<void> in_gpu = val.pf_weight_layout.to_gpu(in_cpu);

    // Run kernel on GPU.
    // cuda_kernel(void *out, const void *in, uint nt_in, uint nt_in_per_wt)
    Array<void> out_gpu(dtype, {nt_out,Mpad,Ppad}, af_gpu | af_zero | af_guard);
    val.cuda_kernel <<<1,32>>> (out_gpu.data, in_gpu.data, nt_in, nt_in_per_wt);
    CUDA_PEEK("pf_weight_reader");

    // Compare.
    assert_arrays_equal(out_cpu, out_gpu, "out_cpu", "out_gpu", {"tout","mpad","ppad"});
}


// -------------------------------------------------------------------------------------------------
//
// PfOutputMicrokernel


struct PfOutputMicrokernelRegistry : public PfOutputMicrokernel::Registry
{
    using Key = PfOutputMicrokernel::RegistryKey;
    using Val = PfOutputMicrokernel::RegistryValue;

    virtual void add(const Key &key, const Val &val, bool debug) override
    {
        // Just check that all members have been initialized.
        // (In the future, I may add more argument checking here.)
        
        xassert((key.dtype == Dtype::native<float>()) || (key.dtype == Dtype::native<__half>()));
        xassert(key.Dout > 0);
        xassert(val.cuda_kernel != nullptr);

        // Call add() in base class.
        PfOutputMicrokernel::Registry::add(key, val, debug);
    }
};

// Static member function
PfOutputMicrokernel::Registry &PfOutputMicrokernel::registry()
{
    // Instead of declaring the registry as a static global variable, we declare it as a
    // static local variable in the static member function PfOutputMicrokernel::registry().
    // The registry will be initialized the first time that PfOutputMicrokernel::registry()
    // is called.
    //
    // This kludge is necessary because the registry is accessed at library initialization
    // time, by callers in other source files, and source files are executed in an
    // arbitrary order.
    
    static PfOutputMicrokernelRegistry reg;
    return reg;  // note: thread-safe (as of c++11)
}

bool operator==(const PfOutputMicrokernel::RegistryKey &k1, const PfOutputMicrokernel::RegistryKey &k2)
{
    return (k1.dtype == k2.dtype) && (k1.Dout == k2.Dout);
}

ostream &operator<<(ostream &os, const PfOutputMicrokernel::RegistryKey &k)
{
    os << "PfOutputMicrokernel(dtype=" << k.dtype << ", Dout=" << k.Dout << ")";
    return os;
}

ostream &operator<<(ostream &os, const PfOutputMicrokernel::RegistryValue &v)
{
    return os;
}


void PfOutputMicrokernel::test_random()
{
    PfOutputMicrokernel::RegistryKey key = PfOutputMicrokernel::registry().get_random_key();
    
    Dtype dtype = key.dtype;
    uint Dout = key.Dout;
    uint nt_in = xdiv(1024, dtype.nbits) * rand_int(1, 100);
    uint nt_out = xdiv(nt_in, Dout);
    
    cout << "test_pf_output_microkernel: dtype=" << dtype << ", Dout=" << Dout << ", nt_in=" << nt_in << endl;

    Array<float> zin_cpu({4,nt_in}, af_uhost | af_random);
    Array<float> zout_cpu({nt_out}, af_uhost);
    Array<uint> ain_cpu({4,nt_in}, af_uhost);

    // Each (s,tin) pair gets a random uint token.
    //   - token_mapping: (token) -> (s,tin)
    //   - ain_cpu: inverse (s,tin) -> (token)

    std::unordered_map<uint, std::pair<uint,uint>> token_mapping;
    std::mt19937 &rng = ksgpu::default_rng();

    for (uint s = 0; s < 4; s++) {
        for (uint tin = 0; tin < nt_in; tin++) {
            for (;;) {
                uint token = rng();
                if (token_mapping.find(token) == token_mapping.end()) {
                    token_mapping[token] = std::pair<int,int> (s,tin);
                    ain_cpu.at({s,tin}) = token;
                    break;
                }
            }
        }
    }

    // Compute 'zout_cpu' (reference CPU implementation).

    for (uint tout = 0; tout < nt_out; tout++) {
        float zmax = -1.0e10f;
        for (uint s = 0; s < 4; s++)
            for (uint tin = tout*Dout; tin < (tout+1)*Dout; tin++)
                zmax = fmaxf(zmax, zin_cpu.at({s,tin}));
        zout_cpu.at({tout}) = zmax;
    }

    // Run GPU kernel.
    Array<void> zin_gpu = zin_cpu.convert(dtype).to_gpu();
    Array<uint> ain_gpu = ain_cpu.to_gpu();
    Array<void> zout_gpu(dtype, {nt_out}, af_gpu | af_guard);
    Array<uint> aout_gpu({nt_out}, af_gpu | af_guard);

    // cuda_kernel(void *zout, uint *aout, void *zin, uint *ain, uint nt_in)
    auto kernel = PfOutputMicrokernel::registry().get(key).cuda_kernel;

    kernel<<<1,32>>> (zout_gpu.data, aout_gpu.data, zin_gpu.data, ain_gpu.data, nt_in);
    CUDA_PEEK("pf_output_test_kernel");

    zout_gpu = zout_gpu.to_host();
    aout_gpu = aout_gpu.to_host();
    
    // The 'zout_gpu' array can be directly compared to the 'zout_cpu' array.
    // However, 'aout_gpu' cannot be directly compared to a CPU reference implementation,
    // because of (near-)ties. Therefore, we compute 'za_gpu', by evaluating the
    // 'zin_cpu' array at the array locations given by the 'aout_gpu' tokens. If 'za_gpu'
    // agrees with 'zout_cpu' (within roundoff error), then the 'aout_gpu' array is
    // correct.

    Array<float> za_gpu({nt_out}, af_uhost);

    for (uint tout = 0; tout < nt_out; tout++) {
        uint token = aout_gpu.at({tout});
        
        auto it = token_mapping.find(token);
        if (token_mapping.find(token) == token_mapping.end())
            throw runtime_error("aout_gpu contains invalid token?!");

        auto [s,tin] = it->second;
        if ((tin < tout*Dout) || (tin >= (tout+1)*Dout))
            throw runtime_error("tin is out-of-range?!");

        za_gpu.at({tout}) = zin_cpu.at({s,tin});
    }

    // Now we can compare everything.

    double eps = 10 * dtype.precision();
    assert_arrays_equal(zout_cpu, zout_gpu, "zout_cpu", "zout_gpu", {"tout"}, eps);
    assert_arrays_equal(zout_cpu, za_gpu, "zout_cpu", "za_gpu", {"tout"}, eps);
}


// -------------------------------------------------------------------------------------------------
//
// ReferencePfSquare


ReferencePfSquare::ReferencePfSquare(long max_kernel_width_, long total_beams_, long beams_per_batch_,
                                     long ndm_, long nt_in_) :
    max_kernel_width(max_kernel_width_),
    total_beams(total_beams_),
    beams_per_batch(beams_per_batch_),
    ndm(ndm_),
    nt_in(nt_in_)
{
    xassert(max_kernel_width > 0);
    xassert(is_power_of_two(max_kernel_width));
    xassert_le(max_kernel_width, long(constants::max_pf_width));
    xassert(total_beams > 0);
    xassert(beams_per_batch > 0);
    xassert_divisible(total_beams, beams_per_batch);
    xassert(ndm > 0);
    xassert(nt_in > 0);

    // Note: unlike GpuPfSquare, there is no constraint relating nt_in to 32 or to tpad.
    // See the class comment in PeakFindingKernel.hpp.

    this->nprofiles = 3 * integer_log2(max_kernel_width) + 1;
    this->nbatches = xdiv(total_beams, beams_per_batch);
    this->nrows = beams_per_batch * ndm;
    this->num_levels = max(integer_log2(max_kernel_width), 1);
    this->tpad = max(2 * max_kernel_width, 32L);

    // Zero-initialized, which is the correct state for ichunk=0: it says that all samples
    // preceding the stream are zero, matching the convention in the dedispersion tex notes.
    this->persistent_state = Array<float> ({total_beams, ndm, tpad}, af_uhost | af_zero);
    this->boxcars = Array<float> ({num_levels+1, tpad + nt_in}, af_uhost | af_zero);
}


void ReferencePfSquare::apply(Array<double> &acc, const Array<float> &in, long ibatch)
{
    xassert((ibatch >= 0) && (ibatch < nbatches));
    xassert_eq(ibatch, expected_ibatch);
    expected_ibatch = (ibatch + 1) % nbatches;

    xassert_shape_eq(acc, ({ beams_per_batch, ndm, nprofiles }));
    xassert_shape_eq(in, ({ beams_per_batch, ndm, nt_in }));
    xassert(acc.on_host());
    xassert(in.on_host());
    xassert(acc.get_ncontig() >= 1);   // profile axis must be contiguous
    xassert(in.get_ncontig() >= 1);    // time axis must be contiguous

    const long L = num_levels;
    const long nt = tpad + nt_in;
    const long b0 = ibatch * beams_per_batch;
    float *bcar0 = &boxcars.at({0,0});   // level 0 of the cascade, length nt

    for (long b = 0; b < beams_per_batch; b++) {
        for (long d = 0; d < ndm; d++) {
            const float *src = &in.at({b,d,0});             // length nt_in
            float *ps = &persistent_state.at({b0+b,d,0});   // length tpad

            // Level 0: the samples preceding the chunk, followed by the chunk itself.

            for (long t = 0; t < tpad; t++)
                bcar0[t] = ps[t];
            for (long t = 0; t < nt_in; t++)
                bcar0[tpad+t] = src[t];

            // Boxcar cascade b_{j+1}[u] = b_j[u] + b_j[u - 2^j], so that b_j[u] is the sum of
            // the 2^j input samples ending at u. (A running sliding-window update would be
            // cheaper, but its float32 error accumulates along the stream, whereas this
            // recursion is a fresh balanced sum at every step -- and it is what the GPU
            // kernel does, which is the comparison that matters.)
            //
            // Samples before the start of the buffer are taken to be zero. That cannot affect
            // the output: the longest profile reaches (2*max_kernel_width - 1) <= (tpad-1)
            // samples back, and the accumulation below starts at u = tpad.

            for (long j = 0; j < L; j++) {
                const float *bj = &boxcars.at({j,0});
                float *bj1 = &boxcars.at({j+1,0});
                long lag = pow2(j);

                for (long t = 0; t < nt; t++)
                    bj1[t] = bj[t] + ((t >= lag) ? bj[t-lag] : 0.0f);
            }

            // Profiles at level 'lam' (S = 2^lam), from the "Peak-finding kernels" section of
            // notes/dedispersion.tex, written in terms of b_{lam+1} exactly as the GPU kernel
            // writes them (see gpu_pf_square_kernel()):
            //
            //   h_{lam,0} = [1]^S                   -> y = b_lam[u]      (lam == 0 only)
            //   h_{lam,1} = [1]^2S                  -> y = b_{lam+1}[u]
            //   h_{lam,2} = [1/2]^S [1]^S [1/2]^S   -> y = (b_{lam+1}[u] + b_{lam+1}[u-S])/2
            //   h_{lam,3} = [1/2]^S [1]^2S [1/2]^S  -> y = (b_{lam+1}[u] + b_{lam+1}[u-S]
            //                                                + b_{lam+1}[u-2S])/2
            //
            // Profile index is p = 3*lam + q. Level 0 contributes q=0..3 and higher levels
            // q=1..3, which the two conditionals below encode: q=0 exists only at lam==0, and
            // (3*lam+3 < nprofiles) is false only in the degenerate case max_kernel_width==1,
            // where nprofiles==1 and p=0 is the only profile.

            double *a = &acc.at({b,d,0});   // length nprofiles

            for (long lam = 0; lam < L; lam++) {
                const float *bl = &boxcars.at({lam+1,0});
                long S = pow2(lam);

                for (long t = tpad; t < nt; t++) {
                    if (lam == 0) {
                        float y0 = bcar0[t];
                        a[0] += double(y0) * y0;
                    }

                    if (3*lam + 3 < nprofiles) {
                        float y1 = bl[t];
                        float y2 = 0.5f * (bl[t] + bl[t-S]);
                        float y3 = 0.5f * (bl[t] + bl[t-S] + bl[t-2*S]);

                        a[3*lam+1] += double(y1) * y1;
                        a[3*lam+2] += double(y2) * y2;
                        a[3*lam+3] += double(y3) * y3;
                    }
                }
            }

            // Save the last 'tpad' samples of the stream, for the next chunk's history. These
            // are read out of the level-0 buffer rather than out of 'src', which is what makes
            // nt_in < tpad work: the tail of the new history is then part of the old one.

            for (long t = 0; t < tpad; t++)
                ps[t] = bcar0[nt_in + t];
        }
    }
}


// Static member function: the cross-family test.
//
// With weights = 1, Dcore = 1 and Dout = 1, the peak-finder's eval_tokens() at the token
// (m, p, t=0) returns the LINEAR response y_{m,p}(tout) = (h_p * x_m)(tout), while a PfSquare
// returns sum_t y_{m,p}(t)^2 over the same time grid. So the two agree iff
//
//   sum_tout eval_tokens(m,p)[b,d,tout]^2  ==  acc[b, d*M+m, p]
//
// for every (b,d,m,p). The identity is exact; the tolerance below is for float summation
// order only.
//
// This is the ONLY test linking the peak-finding kernels to the squaring kernels. Together
// with --gpfk and --pfsq it means every path from one family to the other crosses a tested
// edge: if h_{lambda,q} diverged between the families this test fails, and if it diverged
// between a reference and its GPU twin one of those two fails.
//
// Pushing random data through both full pipelines, rather than comparing extracted h_p
// coefficients, is what makes this catch profile mis-ordering, an off-by-one in the boxcar
// cascade, a wrong history length, and a multiplet-axis divergence.

void ReferencePfSquare::test_vs_peak_finder()
{
    // pf_rank >= 1 gives M >= 2. M == 1 would make the multiplet axis a spectator on both
    // sides, so a divergence along it would be invisible.
    long pf_rank = rand_int(1, 4);
    vector<long> subband_counts = FrequencySubbands::make_random_subband_counts(pf_rank);
    FrequencySubbands fs(subband_counts);
    long M = fs.M;

    long Wmax = pow2(rand_int(0, integer_log2(constants::max_pf_width) + 1));
    long D = pow2(rand_int(0, 3));

    // K > 0 is where the peak-finder folds extra DM bits into its multiplet index. This test
    // is the cheapest check of that bookkeeping: the PfSquare knows nothing about 'mu', so a
    // (mu, m) mix-up shows up here with no dedispersion machinery in the way.
    long K = rand_int(0, 3);
    long E = pow2(K);
    long Dpf = D << K;

    long nchunks = rand_int(1, 5);
    long beams_per_batch = rand_int(1, 3);
    long nbatches = rand_int(1, 3);
    long total_beams = beams_per_batch * nbatches;

    // PeakFindingKernelParams::validate() requires nt_in to be a multiple of 32 (for float32).
    // Note that nt_in < tpad is reachable here, and is intentionally not excluded.
    long nt_in = 32 * rand_int(1, 5);

    // We take nt_wt = nt_in rather than 1, which lets nt_in be any multiple of 32 (the
    // reference requires the ratio nt_out/nt_wt to be a power of two).

    PeakFindingKernelParams pf_params;
    pf_params.subband_counts = subband_counts;
    pf_params.dtype = Dtype::native<float> ();
    pf_params.max_kernel_width = Wmax;
    pf_params.beams_per_batch = beams_per_batch;
    pf_params.total_beams = total_beams;
    pf_params.ndm_out = D;
    pf_params.ndm_wt = 1;
    pf_params.nt_out = nt_in;   // Dout = 1, i.e. no time coarse-graining
    pf_params.nt_in = nt_in;
    pf_params.nt_wt = nt_in;
    pf_params.xdm_rank = K;
    pf_params.validate();

    ReferencePeakFindingKernel pf(pf_params, /*Dcore=*/ 1);   // evaluate h_p at every time sample

    // One PfSquare row per (coarse dm, multiplet) pair of the peak-finder's input array, in
    // the input's own (dpf, m) order. The peak-finder's m_ext is a DIFFERENT ordering of the
    // same rows, and translating between the two is what the test below checks.
    ReferencePfSquare sq(Wmax, total_beams, beams_per_batch, Dpf*M, nt_in);

    long P = pf.nprofiles;
    long N = fs.N;
    xassert_eq(sq.nprofiles, P);
    xassert_eq(pf.M_ext, M*E);
    xassert_eq(D << pf.K, Dpf);

    // The identity is exact only if both objects carry at least the longest kernel's history,
    // i.e. (2*Wmax) input samples. Both do by construction, but the two 'tpad' definitions are
    // independent of each other, so check rather than assume.
    xassert_ge(pf.tpad, 2*Wmax);
    xassert_ge(sq.tpad, 2*Wmax);

    cout << "\nReferencePfSquare::test_vs_peak_finder()\n"
         << "    subband_counts = " << ksgpu::tuple_str(subband_counts) << "\n"
         << "    max_kernel_width = " << Wmax << "\n"
         << "    nprofiles = " << P << "\n"
         << "    M = " << M << "\n"
         << "    xdm_rank = " << K << "\n"
         << "    ndm_out = " << D << "\n"
         << "    beams_per_batch = " << beams_per_batch << "\n"
         << "    total_beams = " << total_beams << "\n"
         << "    nt_in = " << nt_in << "\n"
         << "    nchunks = " << nchunks << endl;

    // Weights = 1, so the peak-finder's (w*y) is the raw linear response y.
    Array<float> wt({beams_per_batch, 1L, nt_in, P, N}, af_uhost | af_zero);
    for (long i = 0; i < wt.size; i++)
        wt.data[i] = 1.0f;

    // One accumulator per beam, so that batches don't overwrite each other.
    Array<double> acc_sq({total_beams, Dpf*M, P}, af_uhost | af_zero);
    Array<double> acc_pf({total_beams, Dpf*M, P}, af_uhost | af_zero);

    Array<float> out_max({beams_per_batch, D, nt_in}, af_uhost | af_zero);
    Array<uint> out_argmax({beams_per_batch, D, nt_in}, af_uhost | af_zero);
    Array<uint> tokens({beams_per_batch, D, nt_in}, af_uhost | af_zero);
    Array<float> out_tok({beams_per_batch, D, nt_in}, af_uhost | af_zero);

    for (long ichunk = 0; ichunk < nchunks; ichunk++) {
        for (long ibatch = 0; ibatch < nbatches; ibatch++) {
            long b0 = ibatch * beams_per_batch;
            Array<float> in({beams_per_batch, Dpf, M, nt_in}, af_uhost | af_random);

            // apply() both advances the peak-finder's persistent state and populates the
            // 'tmp_arr' that eval_tokens() reads.
            pf.apply(out_max, out_argmax, in, wt, ibatch);

            for (long m_ext = 0; m_ext < M*E; m_ext++) {
                long m = m_ext >> K;
                long mu = m_ext & (E-1);

                for (long p = 0; p < P; p++) {
                    // Token format is (t) | (p << 8) | (m << 16) | (mu << 24), and with
                    // Dcore == 1 the only legal fine time is t = 0. This test draws K > 0
                    // most of the time, so it is the one place a (m, mu) field mix-up shows
                    // up with no dedispersion machinery in the way.
                    uint token = _m_fields(m_ext, K) | (uint(p) << 8);
                    for (long i = 0; i < tokens.size; i++)
                        tokens.data[i] = token;

                    pf.eval_tokens(out_tok, tokens, wt);

                    for (long b = 0; b < beams_per_batch; b++) {
                        for (long d = 0; d < D; d++) {
                            double s = 0.0;
                            for (long tout = 0; tout < nt_in; tout++) {
                                float y = out_tok.at({b,d,tout});
                                s += double(y) * y;
                            }
                            // Peak-finder output DM 'd' and token fields (m, mu) read input
                            // DM row ((d << K) | mu), multiplet m.
                            acc_pf.at({b0+b, ((d << K) | mu)*M + m, p}) += s;
                        }
                    }
                }
            }

            Array<float> in_sq = in.reshape({beams_per_batch, Dpf*M, nt_in});
            Array<double> acc_slice = acc_sq.slice(0, b0, b0 + beams_per_batch);
            sq.apply(acc_slice, in_sq, ibatch);
        }
    }

    // Tolerance: the two sides sum the same non-negative terms, but in different orders and
    // through different (algebraically equivalent) float32 expressions for y.
    double eps = 1.0e-5;
    assert_arrays_equal(acc_pf, acc_sq, "acc_pf", "acc_sq", {"b","dm","p"}, eps, eps);
}


// -------------------------------------------------------------------------------------------------
//
// GpuPfSquare


// Returns b at time (u0 + lane - k), given that 'cur' holds b at (u0 + lane), and 'prev'
// holds b at (u0 - 32 + lane), on every lane of the warp. Requires 0 < k <= 32.
//
// Note that __shfl_sync() reduces its srcLane argument mod 32, so the two shuffles read the
// same lane; all we do is select which of the two 32-sample blocks the value came from.

__device__ __forceinline__ float _pfsq_shift(float cur, float prev, int k, int lane)
{
    float a = __shfl_sync(0xffffffffU, cur, lane - k);
    float b = __shfl_sync(0xffffffffU, prev, lane - k);
    return (lane >= k) ? a : b;
}


// One warp per row; the 32 lanes of a warp hold 32 consecutive time samples, so every
// boxcar value maintained in the inner loop has register assignment
//
//   thread:  t4 t3 t2 t1 t0  <->  u4 u3 u2 u1 u0    (u = time index within a 32-sample block)
//   warp:    (blockIdx.x, threadIdx.y)  <->  row
//
// Why a warp per row, rather than a thread per row: the profiles need boxcar values at time
// offsets up to 2^Lambda = max_kernel_width, and with one thread per row those would have to
// live in a shift register indexed by the loop counter -- which spills. Spreading time
// across the warp turns every such offset into a shuffle of the current or previous block,
// so the only carried state is one register per boxcar level. The maximum offset is exactly
// max_kernel_width <= 32, i.e. at most one 32-sample block back, which is why the single
// 'prev' register per level suffices.
//
// Template parameter W = log2(max_kernel_width), so 0 <= W <= log2(constants::max_pf_width).

// Warps per threadblock. Small enough that the register budget (~55/thread at
// max_kernel_width=32, dominated by the P accumulators) is never the occupancy limit.
static constexpr int pfsq_warps_per_block = 4;

template<int W>
__global__ void __launch_bounds__(32 * pfsq_warps_per_block, 1)
gpu_pf_square_kernel(
    double *acc,        // shape (nrows, P), accumulated into
    const float *in,    // shape (nrows, nt_in)
    float *pstate,      // shape (nrows, tpad), already sliced to this batch
    long nrows,
    int nt_in,
    int tpad)
{
    constexpr int P = 3*W + 1;             // number of peak-finding profiles
    constexpr int L = (W > 0) ? W : 1;     // number of peak-finding levels
    constexpr int NB = L + 1;              // number of boxcars maintained (b_0 .. b_L)

    const int lane = threadIdx.x;
    long row = (long(blockIdx.x) * long(blockDim.y)) + threadIdx.y;

    // Note that 'row' is warp-uniform, so the whole warp returns together, and the
    // __shfl_sync() calls below always have their full mask converged.

    if (row >= nrows)
        return;

    // Apply per-warp offsets.
    //   acc:     before shape (nrows,P) contiguous;      after: length P, contiguous
    //   in:      before shape (nrows,nt_in) contiguous;  after: length nt_in, contiguous
    //   pstate:  before shape (nrows,tpad) contiguous;   after: length tpad, contiguous

    acc += row * P;
    in += row * long(nt_in);
    pstate += row * long(tpad);

    float accum[P];

    #pragma unroll
    for (int p = 0; p < P; p++)
        accum[p] = 0.0f;

    // prev[j] = b_j at time (u0 - 32 + lane), i.e. the previous loop iteration's value.
    // Zeroing it is harmless: the first 'tpad' samples are warm-up (see below).

    float prev[NB];

    #pragma unroll
    for (int j = 0; j < NB; j++)
        prev[j] = 0.0f;

    // Pass 0 replays the 'tpad' samples preceding the chunk. The boxcar cascade is wrong
    // until it has seen 2*max_kernel_width samples, so pass 0 accumulates nothing; it exists
    // only to bring the cascade into a correct state at the start of the chunk. Pass 1 is
    // the chunk itself.

    for (int pass = 0; pass < 2; pass++) {
        const float *src = pass ? in : pstate;
        int nsamp = pass ? nt_in : tpad;

        for (int u0 = 0; u0 < nsamp; u0 += 32) {
            float cur[NB];
            float sh[NB];

            cur[0] = src[u0 + lane];

            // Boxcar cascade b_{j+1}[u] = b_j[u] + b_j[u - 2^j], where b_j is the sum of the
            // 2^j input samples ending at u. (A running sliding-window update would be
            // cheaper, but its float32 error accumulates along the stream, whereas this
            // recursion is a fresh balanced sum at every step.)

            #pragma unroll
            for (int j = 0; j < L; j++) {
                sh[j] = _pfsq_shift(cur[j], prev[j], 1 << j, lane);
                cur[j+1] = cur[j] + sh[j];
            }

            sh[L] = _pfsq_shift(cur[L], prev[L], 1 << L, lane);

            if (pass) {
                // Profiles at level 'lam' (S = 2^lam), from the "Peak-finding kernels"
                // section of notes/dedispersion.tex:
                //
                //   h_{lam,0} = [1]^S                   -> y = b_lam[u]
                //   h_{lam,1} = [1]^2S                  -> y = b_{lam+1}[u]
                //   h_{lam,2} = [1/2]^S [1]^S [1/2]^S   -> y = (b_{lam+1}[u] + b_{lam+1}[u-S])/2
                //   h_{lam,3} = [1/2]^S [1]^2S [1/2]^S  -> y = (b_{lam+1}[u] + b_{lam+1}[u-S]
                //                                                + b_{lam+1}[u-2S])/2
                //
                // The last two identities are what make this cheap. Written in terms of
                // b_lam they need four taps at spacing S; written in terms of b_{lam+1} they
                // need two, and one of those (the u-2S tap) is the cascade shift sh[lam+1],
                // already computed above.
                //
                // Profile index is p = 3*lam + q. Level 0 contributes q=0..3 and higher
                // levels contribute q=1..3, which the two conditionals below encode: q=0
                // exists only at lam==0, and (3*lam+3 < P) is false only in the degenerate
                // case max_kernel_width==1, where P==1 and p=0 is the only profile.

                #pragma unroll
                for (int lam = 0; lam < L; lam++) {
                    if (lam == 0) {
                        float y = cur[0];
                        accum[0] = fmaf(y, y, accum[0]);
                    }

                    if (3*lam + 3 < P) {
                        float s1 = _pfsq_shift(cur[lam+1], prev[lam+1], 1 << lam, lane);
                        float y1 = cur[lam+1];
                        float y2 = 0.5f * (y1 + s1);
                        float y3 = 0.5f * (y1 + s1 + sh[lam+1]);

                        accum[3*lam+1] = fmaf(y1, y1, accum[3*lam+1]);
                        accum[3*lam+2] = fmaf(y2, y2, accum[3*lam+2]);
                        accum[3*lam+3] = fmaf(y3, y3, accum[3*lam+3]);
                    }
                }
            }

            #pragma unroll
            for (int j = 0; j < NB; j++)
                prev[j] = cur[j];
        }
    }

    // Warp-reduce each accumulator, and fold the result into the float64 output.
    //
    // This is the kernel's only float64 arithmetic, and it runs once per row per chunk
    // rather than once per sample. That is deliberate: float64 runs at 1/64 of float32 on
    // the consumer GPUs we target, so P float64 adds in the inner loop would cost ~1000
    // float32-equivalent flops per sample and make the kernel compute bound by a factor of
    // a few, where it is otherwise comfortably memory-bandwidth bound.
    //
    // No float32 compensation is needed below the float64 level: each lane's partial holds
    // only (nt_in/32) terms, all non-negative (they are squares, so there is no
    // cancellation), and the tree reduction adds only log2(32) further levels.
    //
    // The store below is single-lane and therefore not cache-friendly, unlike every other
    // global access in this kernel. That is deliberate: it moves P*8 bytes once per row per
    // chunk, against nt_in*4 bytes of input read, so at nt_in=2048, P=16 it is under 2% of
    // the traffic -- not worth a transpose to coalesce.

    #pragma unroll
    for (int p = 0; p < P; p++) {
        float s = accum[p];

        #pragma unroll
        for (int d = 16; d > 0; d >>= 1)
            s += __shfl_down_sync(0xffffffffU, s, d);

        if (lane == 0)
            acc[p] += double(s);
    }

    // Save the last 'tpad' input samples, for the next chunk's warm-up pass.

    for (int i = lane; i < tpad; i += 32)
        pstate[i] = in[nt_in - tpad + i];
}


GpuPfSquare::GpuPfSquare(long max_kernel_width_, long total_beams_, long beams_per_batch_,
                         long ndm_, long nt_in_) :
    max_kernel_width(max_kernel_width_),
    total_beams(total_beams_),
    beams_per_batch(beams_per_batch_),
    ndm(ndm_),
    nt_in(nt_in_)
{
    xassert(max_kernel_width > 0);
    xassert(is_power_of_two(max_kernel_width));
    xassert_le(max_kernel_width, long(constants::max_pf_width));
    xassert(total_beams > 0);
    xassert(beams_per_batch > 0);
    xassert_divisible(total_beams, beams_per_batch);
    xassert(ndm > 0);
    xassert(nt_in > 0);

    // The kernel processes 32 time samples per warp-wide iteration, and copies the chunk's
    // last 'tpad' samples into the persistent state.
    xassert_divisible(nt_in, 32);

    this->nprofiles = 3 * integer_log2(max_kernel_width) + 1;
    this->nbatches = xdiv(total_beams, beams_per_batch);
    this->nrows = beams_per_batch * ndm;
    this->tpad = std::max(2 * max_kernel_width, 32L);

    xassert_ge(nt_in, tpad);

    // Global memory traffic per launch: the input, plus the warm-up re-read of 'tpad'
    // samples and the read+write of the persistent state, plus the float64 accumulator
    // read-modify-write.

    long nbytes = nrows * ((nt_in + 3*tpad) * 4L + nprofiles * 16L);
    resource_tracker.add_kernel("pf_square", nbytes);
    resource_tracker.add_gmem_footprint("pf_square_pstate", total_beams * ndm * tpad * 4L, true);
}


void GpuPfSquare::allocate(BumpAllocator &allocator)
{
    if (is_allocated)
        throw runtime_error("double call to GpuPfSquare::allocate()");
    if (!(allocator.aflags & af_gpu))
        throw runtime_error("GpuPfSquare::allocate(): allocator.aflags must contain af_gpu");
    if (!(allocator.aflags & af_zero))
        throw runtime_error("GpuPfSquare::allocate(): allocator.aflags must contain af_zero");

    long nbytes_before = allocator.get_nbytes_allocated();

    // Zero-initialized, which is the correct state for ichunk=0: it says that all samples
    // preceding the stream are zero, matching the convention in the dedispersion tex notes.
    this->persistent_state = allocator.allocate_array<float> ({total_beams, ndm, tpad});

    long nbytes_allocated = allocator.get_nbytes_allocated() - nbytes_before;
    xassert_eq(nbytes_allocated, resource_tracker.get_gmem_footprint());

    this->is_allocated = true;
}


void GpuPfSquare::launch(Array<double> &acc, const Array<float> &in, long ibatch, cudaStream_t stream)
{
    xassert(this->is_allocated);
    xassert((ibatch >= 0) && (ibatch < nbatches));
    xassert_eq(ibatch, expected_ibatch);
    expected_ibatch = (ibatch + 1) % nbatches;

    xassert_shape_eq(acc, ({ beams_per_batch, ndm, nprofiles }));
    xassert_shape_eq(in, ({ beams_per_batch, ndm, nt_in }));

    // The kernel derives all of its strides from (nt_in, tpad, nprofiles).
    xassert(acc.is_fully_contiguous());
    xassert(in.is_fully_contiguous());
    xassert(acc.on_gpu());
    xassert(in.on_gpu());

    // Slice the persistent state along its beam axis. Note that the beam axis is outermost,
    // so the slice is still fully contiguous, and reshapes to (nrows, tpad).
    long b0 = ibatch * beams_per_batch;
    long b1 = b0 + beams_per_batch;
    Array<float> pstate = this->persistent_state.slice(0, b0, b1);

    dim3 grid_dims = { uint((nrows + pfsq_warps_per_block - 1) / pfsq_warps_per_block), 1, 1 };
    dim3 block_dims = { 32, pfsq_warps_per_block, 1 };

    // 'W' is a compile-time parameter, so we dispatch on log2(max_kernel_width). There are
    // only (1 + log2(constants::max_pf_width)) possible values, so a switch is enough -- no
    // KernelRegistry is needed, unlike the peak-finding kernels above.

    void (*kernel)(double *, const float *, float *, long, int, int) = nullptr;

    switch (integer_log2(max_kernel_width)) {
        case 0: kernel = gpu_pf_square_kernel<0>; break;
        case 1: kernel = gpu_pf_square_kernel<1>; break;
        case 2: kernel = gpu_pf_square_kernel<2>; break;
        case 3: kernel = gpu_pf_square_kernel<3>; break;
        case 4: kernel = gpu_pf_square_kernel<4>; break;
        case 5: kernel = gpu_pf_square_kernel<5>; break;
        default: throw runtime_error("GpuPfSquare::launch(): unsupported max_kernel_width");
    }

    static_assert(constants::max_pf_width == 32, "GpuPfSquare: switch above needs updating");

    kernel <<< grid_dims, block_dims, 0, stream >>>
        (acc.data, in.data, pstate.data, nrows, int(nt_in), int(tpad));

    CUDA_PEEK("GpuPfSquare::launch");
}


// Static member function: runs one randomized test iteration.
//
// The reference is ReferencePfSquare, which has the same interface and computes the same
// quantity, so nothing needs to be neutralized to make the comparison meaningful.
//
// The two computations share no code -- the GPU side is a warp-shuffle cascade, the CPU side
// a plain loop over explicitly materialized boxcar arrays -- so this is a real test of the
// shuffle cascade rather than a restatement of it. What it does NOT test is whether the
// PfSquare kernel bank h_{lambda,q} agrees with the one the peak-finders use; that is
// ReferencePfSquare::test_vs_peak_finder().

void GpuPfSquare::test_random()
{
    long Wmax = pow2(rand_int(0, integer_log2(constants::max_pf_width) + 1));
    long ndm = rand_int(1, 65);   // a row count, so deliberately not always a power of two
    long nchunks = rand_int(1, 5);

    auto v = ksgpu::random_integers_with_bounded_product(3, std::max(2000/ndm, 8L));
    long beams_per_batch = v[0];
    long nbatches = v[1];
    long total_beams = beams_per_batch * nbatches;

    // The GPU kernel requires nt_in to be a multiple of 32, and at least tpad = max(2*Wmax,32).
    // Both bounds are multiples of 32, so the max below is too.
    long nt_in = std::max(32 * v[2], std::max(2*Wmax, 32L));

    GpuPfSquare gpu_kernel(Wmax, total_beams, beams_per_batch, ndm, nt_in);
    ReferencePfSquare ref_kernel(Wmax, total_beams, beams_per_batch, ndm, nt_in);

    long P = gpu_kernel.nprofiles;
    xassert_eq(ref_kernel.nprofiles, P);
    xassert_eq(ref_kernel.tpad, gpu_kernel.tpad);

    cout << "\nGpuPfSquare::test_random()\n"
         << "    max_kernel_width = " << Wmax << "\n"
         << "    nprofiles = " << P << "\n"
         << "    tpad = " << gpu_kernel.tpad << "\n"
         << "    ndm = " << ndm << "\n"
         << "    beams_per_batch = " << beams_per_batch << "\n"
         << "    total_beams = " << total_beams << "\n"
         << "    nt_in = " << nt_in << "\n"
         << "    nchunks = " << nchunks << endl;

    BumpAllocator allocator(af_gpu | af_zero, -1);  // dummy allocator
    gpu_kernel.allocate(allocator);

    // One accumulator per beam, so that batches don't overwrite each other. Both kernels
    // accumulate over all chunks.
    Array<double> acc_gpu({total_beams, ndm, P}, af_gpu | af_zero);
    Array<double> acc_ref({total_beams, ndm, P}, af_uhost | af_zero);

    for (long ichunk = 0; ichunk < nchunks; ichunk++) {
        for (long ibatch = 0; ibatch < nbatches; ibatch++) {
            long b0 = ibatch * beams_per_batch;

            Array<float> in_cpu({beams_per_batch, ndm, nt_in}, af_uhost | af_random);
            Array<float> in_gpu = in_cpu.to_gpu();

            Array<double> gpu_slice = acc_gpu.slice(0, b0, b0 + beams_per_batch);
            gpu_kernel.launch(gpu_slice, in_gpu, ibatch, nullptr);   // null stream

            Array<double> ref_slice = acc_ref.slice(0, b0, b0 + beams_per_batch);
            ref_kernel.apply(ref_slice, in_cpu, ibatch);
        }
    }

    // Tolerance: both kernels sum the same non-negative terms, but in different orders and
    // in float32 arithmetic, so the error is set by float32 roundoff over the number of
    // accumulated terms -- not by the float64 accumulators.
    double eps = 1.0e-4;
    assert_arrays_equal(acc_ref, acc_gpu, "acc_ref", "acc_gpu", {"b","d","p"}, eps, eps);
}


}  // namespace pirate
