#include "../include/pirate/PeakFindingKernel.hpp"
#include "../include/pirate/BumpAllocator.hpp"
#include "../include/pirate/inlines.hpp"
#include "../include/pirate/utils.hpp"
#include "../include/pirate/PfVariance.hpp"   // PfVarianceConvolver

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
    xassert_divisible(touter_byte_stride, 128);
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


ReferencePeakFindingKernel::ReferencePeakFindingKernel(const PeakFindingKernelParams &params_, long Dcore_) :
    params(params_), fs(params_.subband_counts), Dcore(Dcore_)
{
    params.validate();

    const PeakFindingKernelParams &p = params;
    long B = p.beams_per_batch;
    long D = p.ndm_out;
    long Wmax = p.max_kernel_width;
    long M = fs.M;

    this->nbatches = xdiv(p.total_beams, p.beams_per_batch);
    this->nprofiles = 3 * integer_log2(p.max_kernel_width) + 1;
    this->Dout = xdiv(p.nt_in, p.nt_out);
    this->tpad = max(2*Wmax, 4L);
    this->pstate = Array<float> ({p.total_beams, p.ndm_out, fs.M, tpad}, af_uhost | af_zero); 
    this->num_levels = max(integer_log2(Wmax), 1);

    xassert(Dcore > 0);
    xassert(is_power_of_two(Dcore));
    xassert_divisible(Dout, Dcore);
    
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
        tmp_arr[l] = Array<float> ({B,D,M,nt}, af_uhost | af_zero);

        // To see that this is correct, note that the "base" time sample ends at 
        // time dt, and has length 2^l.
        tmp_iout[l] = xdiv(tpad + dt - pow2(l), dt);
    }
}


// helper for ReferencePeakFindingKernel::apply()
// In addition to the (maxval, argmax) max-reduce, accumulate val^2 into 'sumsq'
// (a running sum-of-squares, later normalized into out_var; see apply()).
static inline void _update_pf(float &maxval, uint &argmax, double &sumsq, float val, uint token)
{
    argmax = (val > maxval) ? token : argmax;
    maxval = std::max(maxval, val);
    sumsq += double(val) * val;
}


void ReferencePeakFindingKernel::apply(
    ksgpu::Array<float> &out_max,      // shape (beams_per_batch, ndm_out, nt_out)
    ksgpu::Array<uint> &out_argmax,    // shape (beams_per_batch, ndm_out, nt_out)
    ksgpu::Array<double> &out_var,     // shape (beams_per_batch, ndm_out, M, nprofiles), or empty
    const ksgpu::Array<float> &in,     // shape (beams_per_batch, ndm_out, M, nt_in)
    const ksgpu::Array<float> &wt,     // shape (beams_per_batch, ndm_wt, nt_wt, nprofiles, N)
    long ibatch, bool debug)
{
    const PeakFindingKernelParams &p = params;
    xassert_shape_eq(out_max, ({p.beams_per_batch, p.ndm_out, p.nt_out}));
    xassert_shape_eq(out_argmax, ({p.beams_per_batch, p.ndm_out, p.nt_out}));
    xassert_shape_eq(in, ({p.beams_per_batch, p.ndm_out, fs.M, p.nt_in}));
    xassert_shape_eq(wt, ({p.beams_per_batch, p.ndm_wt, p.nt_wt, nprofiles, fs.N}));
 
    xassert(out_max.on_host());
    xassert(out_argmax.on_host());
    xassert(in.on_host());
    xassert(wt.on_host());

    // Optional out_var: an empty array disables the feature; otherwise it is overwritten
    // with per-chunk variances (see comments in PeakFindingKernel.hpp).
    bool do_var = (out_var.size > 0);
    if (do_var) {
        xassert_shape_eq(out_var, ({p.beams_per_batch, p.ndm_out, fs.M, nprofiles}));
        xassert(out_var.on_host());
        xassert(out_var.is_fully_contiguous());
    }

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

    xassert_shape_eq(in, ({B,D,M,nt_in}));
    xassert(in.get_ncontig() >= 1);

    // Fill l=0 (with 'in' + 'pstate' wraparound)
 
    for (long b = 0; b < B; b++) {
        for (long d = 0; d < D; d++) {
            for (long m = 0; m < M; m++) {
                float *dst = &tmp_arr[0].at({b,d,m,0});  // length (nt_in+tpad)
                float *ps = &pstate.at({b0+b,d,m,0});    // length (tpad)
                const float *src = &in.at({b,d,m,0});    // length (nt_in)

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

    // Downsample l -> (l+1)

    for (long l = 0; l < num_levels-1; l++) {
        long nsrc = tmp_nt.at(l);
        long ndst = tmp_nt.at(l+1);
        long r = xdiv(tmp_dt[l+1], tmp_dt[l]);  // ratio of step sizes
        long s = xdiv(pow2(l), tmp_dt[l]);      // spacing between logically contiguous samples in source

        xassert_eq(r*(ndst-1) + s, nsrc-1);
        
        for (long b = 0; b < B; b++) {
            for (long d = 0; d < D; d++) {
                for (long m = 0; m < M; m++) {
                    float *dst = &tmp_arr.at(l+1).at({b,d,m,0});
                    float *src = &tmp_arr.at(l).at({b,d,m,0});
                    
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
            // out_var[b,d] is a contiguous (M,P) block, overwritten (zeroed, then accumulated
            // across the tout loop below) so the caller always gets a single-chunk variance.
            double *var_bd = do_var ? &out_var.at({b,d,0,0}) : nullptr;
            if (do_var)
                for (long i = 0; i < M*P; i++)
                    var_bd[i] = 0.0;

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
                    double wvar = 1.0 / double(nt_out * nsamp);  // 1/count for level l (sum-of-squares -> variance)

                    for (int m = 0; m < M; m++) {
                        int n = fs.m_to_n[m];
                        float w0 = l ? 0.0f : wp[n];      // p = 0 (only for l=0)
                        float w1 = wp[(3*l+1)*N + n];     // p = (3*l+1)
                        float w2 = wp[(3*l+2)*N + n];     // p = (3*l+2)
                        float w3 = wp[(3*l+3)*N + n];     // p = (3*l+3)

                        double *var_m = do_var ? (var_bd + (long)m * P) : nullptr;  // out_var row, p=0..P-1
                        double s0 = 0.0, s1 = 0.0, s2 = 0.0, s3 = 0.0;  // per-(tout,l,m) sum of squares

                        // Each iteration of the isamp-loop corresponds to one time sample in the
                        // tmp[l] array, or (dt) time samples in the original input array.

                        for (int isamp = 0; isamp < nsamp; isamp++) {
                            float x0 = tmp_in[m*mstr + I + tout*nsamp + isamp - 3*S];
                            float x1 = tmp_in[m*mstr + I + tout*nsamp + isamp - 2*S];
                            float x2 = tmp_in[m*mstr + I + tout*nsamp + isamp - S];
                            float x3 = tmp_in[m*mstr + I + tout*nsamp + isamp];

                            uint token0 = (m << 16)| (isamp*dt);  // includes (m,isamp) but not p
                            uint token1 = token0 | ((3*l+1) << 8);    // include p=3*l+1
                            uint token2 = token0 | ((3*l+2) << 8);    // include p=3*l+2
                            uint token3 = token0 | ((3*l+3) << 8);    // include p=3*l+3

                            float y0 = x3;
                            float y1 = (x2 + x3);
                            float y2 = (0.5f*x1 + x2 + 0.5f*x3);
                            float y3 = (0.5f*x0 + x1 + x2 + 0.5f*x3);

                            if (l == 0)
                                _update_pf(maxval, argmax, s0, w0*y0, token0);

                            if (P > 1) {
                                _update_pf(maxval, argmax, s1, w1*y1, token1);
                                _update_pf(maxval, argmax, s2, w2*y2, token2);
                                _update_pf(maxval, argmax, s3, w3*y3, token3);
                            }

                            if (debug && (b == 0) && (d==0) && (tout==2)) {
                                cout << "cpu peak-finder: b=" << b << ", d=" << d << ", tout=" << tout 
                                     << ", level=" << l << ", m=" << m << ", isamp=" << isamp << "\n";

                                if (l == 0)
                                    cout << "   p=0" << " -> (w=" << w0 << ", y=" << y0 << ", w*y=" << (w0*y0) << endl;
                                
                                if (P > 1) {
                                    cout << "   p=" << (3*l+1) << " -> (w=" << w1 << ", y=" << y1 << ", w*y=" << (w1*y1) << endl;
                                    cout << "   p=" << (3*l+2) << " -> (w=" << w2 << ", y=" << y2 << ", w*y=" << (w2*y2) << endl;
                                    cout << "   p=" << (3*l+3) << " -> (w=" << w3 << ", y=" << y3 << ", w*y=" << (w3*y3) << endl;
                                }
                            }
                        }

                        // Fold this (tout,l,m) block's sum-of-squares into out_var, normalized by
                        // wvar = 1/count(level l). The += accumulates across the tout loop; var_bd
                        // was zeroed per (b,d), so out_var ends as the per-chunk variance estimate.
                        if (do_var) {
                            if (l == 0)
                                var_m[0] += s0 * wvar;
                            if (P > 1) {
                                var_m[3*l+1] += s1 * wvar;
                                var_m[3*l+2] += s2 * wvar;
                                var_m[3*l+3] += s3 * wvar;
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
                // Reminder: token = (t) | (p << 8) | (m << 16).

                long m = (token >> 16) & 0xffu;
                long p = (token >> 8) & 0xffu;
                long t = (token & 0xffu);

                if ((m < 0) || (m >= M))
                    throw _bad_token(token, "m out of range");
                if ((p < 0) || (p >= P))
                    throw _bad_token(token, "p out of range");
                if ((t < 0) || (t >= Dout))
                    throw _bad_token(token, "t out of range");

                // p = 3*l+q, where l is the "level".
                long l = p ? ((p-1)/3) : 0;
                long q = p - 3*l;

                // t = isamp*dt
                long dt = tmp_dt.at(l);
                long isamp = t / dt;

                if (t != isamp*dt)
                    throw _bad_token(token, "t is not divisible by dt");

                // Token parsing (token -> (m,isamp,p)) ends here!

                long n = fs.m_to_n.at(m);
                float w = wt.at({b, d/Wds, tout/Tds, p, n});

                int nsamp = tmp_nout[l];       // count
                int S = tmp_sout[l];       // spacing
                int I = tmp_iout[l];       // base

                float x0 = tmp_arr.at(l).at({b, d, m, I + tout*nsamp + isamp - 3*S});
                float x1 = tmp_arr.at(l).at({b, d, m, I + tout*nsamp + isamp - 2*S});
                float x2 = tmp_arr.at(l).at({b, d, m, I + tout*nsamp + isamp - S});
                float x3 = tmp_arr.at(l).at({b, d, m, I + tout*nsamp + isamp});

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
                         << " -> (m=" << m << ", p=" << p << ", t=" << t << ", l=" << l << ", q=" << q << ")"
                         << " -> (w=" << w << ", x0=" << x0 << ", x1=" << x1 << ", x2=" << x2 << ", x3=" << x3 << ")"
                         << " -> " << out_max.at({b,d,tout}) << endl;

                    cout << "  wt.at(" << b << "," << (d/Wds) << "," << (tout/Tds) << "," << p << "," << n << ")"
                         << " = " << wt.at({b,d/Wds,tout/Tds,p,n}) << endl;

                    for (int i = 0; i < 4; i++)
                        cout << "  tmp_arr.at(" << l << ").at(" << b << "," << d << "," << m << "," << (I + tout*nsamp + isamp + (i-3)*S) << ")"
                             << " = " << tmp_arr.at(l).at({b, d, m, I + tout*nsamp + isamp + (i-3)*S}) << endl;

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
// Returns shape (nbeams_per_batch, ndm_out, fs.M, nt_in)
Array<float> ReferencePeakFindingKernel::make_random_input_array()
{
    long B = params.beams_per_batch;
    long D = params.ndm_out;
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


// fill_host_weights(): build peak-finding weights from per-(subband, dm, profile) input
// variances.
//
//   out shape       = (beams_per_batch, ndm_wt, nt_wt, nprofiles, N)   (float)
//   variances shape = (N, ndm_wt, nprofiles)                            (double)
//
// In both modes base_weights[d,n,p] = 1/sqrt(variances[n,d,p]) (note the N <-> ndm_wt
// transpose). Then:
//   randomize=true:  out[b,d,t,p,n] = x * base_weights[d,n,p], where 'x' is a sparse random
//                    value -- per (b,d,t) we draw an "occupancy" p0, then for each (n,p) the
//                    weight is zero with probability ~(1-p0), else uniform in [0,1).
//   randomize=false: out[b,d,t,p,n] = base_weights[d,n,p] (no random multiplier).

void ReferencePeakFindingKernel::fill_host_weights(Array<float> &out, const Array<double> &variances, bool randomize)
{
    const long B = params.beams_per_batch;
    const long D = params.ndm_wt;
    const long T = params.nt_wt;
    const long P = nprofiles;
    const long N = fs.N;

    xassert_shape_eq(out, ({B,D,T,P,N}));
    xassert_shape_eq(variances, ({N,D,P}));
    xassert(out.on_host());
    xassert(variances.on_host());
    xassert(out.is_fully_contiguous());
    xassert(variances.is_fully_contiguous());

    // Phase 1: base_weights[d,n,p] = rsqrtf(variances[n,d,p]).
    // (variances is double, (N,D,P); base_weights is float, (D,N,P) -- transpose the first two
    // axes. rsqrtf() keeps the base-weight computation in float.)
    Array<float> base_weights({D,N,P}, af_uhost);
    {
        const double *vp = variances.data;    // (N,D,P) contiguous, double
        float *bp = base_weights.data;        // (D,N,P) contiguous, float
        for (long n = 0; n < N; n++) {
            for (long d = 0; d < D; d++) {
                const double *vrow = vp + (n*D + d)*P;  // variances[n,d,:]
                float *brow = bp + (d*N + n)*P;         // base_weights[d,n,:]
                for (long p = 0; p < P; p++) {
                    double var = vrow[p];
                    xassert(var > 0.0);
                    brow[p] = rsqrtf(var);
                }
            }
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


// make_bare_random_weights_for_testing(): random weights for testing a "bare" peak-finding
// or cdd2 kernel, whose input is unit-variance (see the header comment). We feed a single
// unit-variance input sample -- a (1,1) array with x[0,0]=1 -- through PfVarianceConvolver,
// giving the per-profile output variance (1,P). (out[0,p] is the zero-lag autocorrelation of
// peak-finding kernel p, i.e. the sum of its squared taps.) We then trivially broadcast these
// P variances across all (subband, dm) and hand off to fill_host_weights() (randomize=true).

void ReferencePeakFindingKernel::make_bare_random_weights_for_testing(Array<float> &out)
{
    const long B = params.beams_per_batch;
    const long D = params.ndm_wt;
    const long T = params.nt_wt;
    const long P = nprofiles;
    const long N = fs.N;

    xassert_shape_eq(out, ({B,D,T,P,N}));
    xassert(out.on_host());
    xassert(out.is_fully_contiguous());

    // Per-profile output variance for one unit-variance input sample: (1,1) -> (1,P).
    PfVarianceConvolver conv;
    double x = 1.0;
    std::vector<double> pf_var(P);
    conv.variance(&x, /*S=*/1, /*nt=*/1, P, pf_var.data());

    // Trivially expand (1,P) -> variances (N, ndm_wt, P): same per-profile variance for
    // every (subband n, dm d).
    Array<double> variances({N,D,P}, af_uhost);
    double *vp = variances.data;
    for (long n = 0; n < N; n++)
        for (long d = 0; d < D; d++)
            for (long p = 0; p < P; p++)
                vp[(n*D + d)*P + p] = pf_var[p];

    this->fill_host_weights(out, variances, /*randomize=*/true);
}


// -------------------------------------------------------------------------------------------------
//
// GpuPeakFindingKernel


GpuPeakFindingKernel::GpuPeakFindingKernel(const PeakFindingKernelParams &params_) :
    params(params_), fs(params_.subband_counts)
{
    params.validate();

    registry_key.dtype = params.dtype;
    registry_key.subband_counts = fs.subband_counts;
    registry_key.Dout = xdiv(params.nt_in, params.nt_out);
    registry_key.Wmax = params.max_kernel_width;

    // Recall the definition of Tinner (used for weight layout, see comments in
    // cuda_generator.PeakFinder.py):
    //
    //   Tinner = max(32*SW/nt_in_per_wt, 1)  

    long SW = xdiv(32, params.dtype.nbits);      // simd width
    long nt_in_per_wt = xdiv(params.nt_in, params.nt_wt);
    registry_key.Tinner = (nt_in_per_wt < 32*SW) ? xdiv(32*SW, nt_in_per_wt) : 1;

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

    long nbytes_before = allocator.nbytes_allocated.load();

    // Allocate persistent_state.
    std::initializer_list<long> shape = { params.total_beams, params.ndm_out, registry_value.PW32 };
    this->persistent_state = allocator.allocate_array<uint>(shape);

    long nbytes_allocated = allocator.nbytes_allocated.load() - nbytes_before;
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
    ref_kernel_large.make_bare_random_weights_for_testing(cpu_wt_large);
 
    Array<float> cpu_out_large({total_beams, ndm_out, nchunks * nt_out_per_chunk}, af_rhost | af_zero);
    Array<uint> cpu_argmax_large({total_beams, ndm_out, nchunks * nt_out_per_chunk}, af_rhost | af_zero);
    Array<double> cpu_var_large;  // empty -> out_var feature disabled
    ref_kernel_large.apply(cpu_out_large, cpu_argmax_large, cpu_var_large, cpu_in_large, cpu_wt_large, 0);

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
            Array<double> cpu_var_small;  // empty -> out_var feature disabled
            ref_kernel_small.apply(cpu_out_small, cpu_argmax_small, cpu_var_small, cpu_in_small, cpu_wt_small, ibatch);

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


}  // namespace pirate
