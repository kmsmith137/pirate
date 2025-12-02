#include "../include/pirate/PeakFindingKernel.hpp"
#include "../include/pirate/inlines.hpp"
#include "../include/pirate/utils.hpp"

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


void PeakFindingKernelParams::validate() const
{
    if ((dtype != Dtype::native<float>()) && (dtype != Dtype::native<__half>()))
        throw runtime_error("LaggedDownsamplingKernelParams: unsupported dtype: " + dtype.str());

    // The checks below (or a subset) arise in three other places:
    //
    //   - pirate_frb.cuda_generator.PeakFindingParams constructor
    //   - test_gpu_peak_finding_kernel(), when random params are generated
    //   - DedispersionConfig::make_random()
    //
    // so changes here should be reflected there as well.
    
    // Check that everything is initialized.
    xassert(dm_downsampling_factor > 0);
    xassert(time_downsampling_factor > 0);
    xassert(max_kernel_width > 0);
    xassert(beams_per_batch > 0);
    xassert(total_beams > 0);
    xassert(ndm_in > 0);
    xassert(nt_in > 0);

    xassert(is_power_of_two(max_kernel_width));
    xassert(is_power_of_two(time_downsampling_factor));
    xassert_divisible(total_beams, beams_per_batch);
    xassert_divisible(ndm_in, dm_downsampling_factor);
    xassert_divisible(nt_in, time_downsampling_factor);

    long ndm_out = xdiv(ndm_in, dm_downsampling_factor);
    long nt_out = xdiv(nt_in, time_downsampling_factor);
    long nbits = dtype.nbits;
    
    // Currently assumed in GPU kernel.
    xassert(max_kernel_width <= 32);
    xassert(time_downsampling_factor <= 32);
    xassert_divisible(nt_in, xdiv(1024,nbits));
    xassert_divisible(nt_out, xdiv(32,nbits));   // tout is fastest axis in out_{max,ssq}
    xassert_divisible(ndm_in, xdiv(32,nbits));   // dm_in is fastest axis in 'wt'
}

PeakFindingKernel::PeakFindingKernel(const PeakFindingKernelParams &params_, long Dcore_) :
    params(params_), Dcore(Dcore_)
{
    params.validate();

    if (Dcore == 0)
        Dcore = params.time_downsampling_factor;

    xassert_divisible(params.time_downsampling_factor, Dcore);

    this->nbatches = xdiv(params.total_beams, params.beams_per_batch);
    this->ndm_out = xdiv(params.ndm_in, params.dm_downsampling_factor);
    this->nt_out = xdiv(params.nt_in, params.time_downsampling_factor);
    this->nprofiles = 3 * integer_log2(params.max_kernel_width) + 1;
}


void PeakFindingKernel::_check_args(const Array<void> &out_max, const Array<void> &out_ssq, const Array<void> &in, const Array<void> &wt, Dtype expected_dtype, long ibatch)
{
    int B = params.beams_per_batch;
    int Min = params.ndm_in;
    int Tin = params.nt_in;    
    int Mout = this->ndm_out;
    int Tout = this->nt_out;
    int P = this->nprofiles;
    
    xassert((ibatch >= 0) && (ibatch < this->nbatches));
    
    xassert_shape_eq(out_max, ({B,P,Mout,Tout}));
    xassert_shape_eq(out_ssq, ({B,P,Mout,Tout}));
    xassert_shape_eq(in, ({B,Min,Tin}));
    xassert_shape_eq(wt, ({B,P,Min}));

    xassert(out_max.is_fully_contiguous());
    xassert(out_ssq.is_fully_contiguous());
    xassert(in.is_fully_contiguous());
    xassert(wt.is_fully_contiguous());

    xassert(out_max.dtype == expected_dtype);
    xassert(out_ssq.dtype == expected_dtype);
    xassert(in.dtype == expected_dtype);
    xassert(wt.dtype == expected_dtype);
}


// -------------------------------------------------------------------------------------------------
//
// ReferencePeakFindingKernel


ReferencePeakFindingKernel::ReferencePeakFindingKernel(const PeakFindingKernelParams &params_, long Dcore_) :
    PeakFindingKernel(params_, Dcore_)  // base class constructor calls params.validate()
{
    long E = params.max_kernel_width;
    long W = (E > 1) ? (2*E) : 1;           // width of widest kernel
    long D = (E > 1) ? min(Dcore,E/2) : 1;  // downsampling factor of widest kernel
    
    this->pstate_nt = (W - D);
    this->pstate = Array<float> ({params.total_beams, params.ndm_in, pstate_nt}, af_uhost | af_zero);
}


// Tiny helper for ReferencePeakFindingKernel::apply().
struct pf_accumulator
{
    float rmax = -1.0e20;
    float rssq = 0.0;

    inline void update(float w, float x)
    {
        rmax = max(rmax, w*x);
        rssq += square(w*x);
    }
};


void ReferencePeakFindingKernel::apply(Array<void> &out_max_, Array<void> &out_ssq_, const Array<void> &in_, const Array<void> &wt_, long ibatch)
{
    constexpr float one_over_sqrt2 = 0.7071067811865476f;
    
    int B = params.beams_per_batch;
    int M = params.dm_downsampling_factor;
    int E = params.max_kernel_width;
    int Dout = params.time_downsampling_factor;
    int Min = params.ndm_in;
    int Tin = params.nt_in;    
    int Mout = this->ndm_out;
    int Tout = this->nt_out;

    // The reference kernel uses float32, regardless of what dtype is specified.
    Array<float> out_max = out_max_.template cast<float> ("ReferencePeakFindingKernel::apply(): 'out_max' array");
    Array<float> out_ssq = out_ssq_.template cast<float> ("ReferencePeakFindingKernel::apply(): 'out_ssq' array");
    Array<float> in = in_.template cast<float> ("ReferencePeakFindingKernel::apply(): 'in' array");
    Array<float> wt = wt_.template cast<float> ("ReferencePeakFindingKernel::apply(): 'wt' array");

    _check_args(out_max, out_ssq, in, wt, Dtype::native<float>(), ibatch);
    
    xassert(out_max.on_host());
    xassert(out_ssq.on_host());
    xassert(in.on_host());
    xassert(wt.on_host());

    // Note: _check_args() asserts that 'wt' is fully contiguous.
    for (long i = 0; i < wt.size; i++)
        if (wt.data[i] < 0.0f)
            throw runtime_error("ReferencePeakFindingKernel::apply(): all weights must be positive");
    
    // This loop just does p=0.
    for (int b = 0; b < B; b++) {
        for (int mout = 0; mout < Mout; mout++) {
            for (int tout = 0; tout < Tout; tout++) {
                pf_accumulator acc0;

                for (int m = mout*M; m < (mout+1)*M; m++) {
                    float w = wt.at({b,0,m});   // p=0
                    
                    for (int t = tout*Dout; t < (tout+1)*Dout; t++) {
                        float x = in.at({b,m,t});  // decided to use 'in' here, not 'in_x'.
                        acc0.update(w, x);
                    }

                    out_max.at({b,0,mout,tout}) = acc0.rmax;  // p=0
                    out_ssq.at({b,0,mout,tout}) = acc0.rssq;  // p=0
                }
            }
        }
    }

    if (E == 1)
        return;
    
    // Throughout this kernel, 'in_x' denotes a version of 'in' which has been:
    //   - downsampled by 2**ids elements
    //   - sampled at sampling rate 2**isamp
    //   - prepadded by 'Tpp' elements
    //
    // To write this out explicitly, 'in_x' has shape (B, Min, Tin/2**isamp + Tpp).
    // Given innermost index 0 <= ix < (Tin/2**isamp + Tpp), we write tx = (ix-Tpp).
    // Array elements at this index are obtained by summing 'in' over:
    //
    //   ((tx+1) * 2**isamp) - 2**ids <= tin < ((tx+1) * 2**isamp)
    //
    // where negative values of 'tin' are saved in ReferencePeakFindingKernel::pstate.
    
    int ids = 0;
    int isamp = 0;
    int Tpp = pstate_nt;

    Array<float> ps = pstate.slice(0, ibatch*B, (ibatch+1)*B);
    Array<float> in_x({B,Min,Tin+Tpp}, af_uhost);

    in_x.slice(2,Tpp,Tin+Tpp).fill(in);
    in_x.slice(2,0,Tpp).fill(ps);
    ps.fill(in_x.slice(2,Tin,Tin+Tpp));   // update pstate, for next call to apply().

    for (;;) {
        
        // The value of 'ids' increases by 1 in every iteration of the loop,
        // but the value of isamp "saturates" at log2(Dcore).
        //
        // The output arrays have been filled for 0 <= p < (3*ids+1).
        // In this iteration of the loop, we'll fill (3*ids+1) <= p < (3*ids+4).

        long Tds = xdiv(Tin, pow2(isamp));
        xassert_shape_eq(in_x, ({B,Min,Tds+Tpp}));
        xassert_eq(isamp, min(ids, integer_log2(Dcore)));
        xassert(in_x.is_fully_contiguous());  // assumed in downsampling logic

        // For computing profiles.
        long DD = xdiv(Tds, Tout);  // Downsampling factor between 'in_ds' and 'out_{max,ssq}' arrays.
        int p0 = 3*ids+1;           // Base profile index

        // Time offset in 'in_ds' array corresponding to 2**ids samples (used for "prepadding" below)
        long dt = pow2(ids-isamp);
        xassert(Tpp >= 3*dt);
 
        // Compute 3 profiles.
        for (int b = 0; b < B; b++) {
            for (int mout = 0; mout < Mout; mout++) {
                for (int tout = 0; tout < Tout; tout++) {
                    pf_accumulator acc0;
                    pf_accumulator acc1;
                    pf_accumulator acc2;
                    
                    for (int m = mout*M; m < (mout+1)*M; m++) {
                        float w0 = wt.at({b,p0,m});
                        float w1 = wt.at({b,p0+1,m});
                        float w2 = wt.at({b,p0+2,m});
                        float *p = &in_x.at({b,m,0});
                        
                        for (int t = tout*DD; t < (tout+1)*DD; t++) {
                            float x0 = p[t+Tpp-3*dt];
                            float x1 = p[t+Tpp-2*dt];
                            float x2 = p[t+Tpp-dt];
                            float x3 = p[t+Tpp];

                            acc0.update(w0, x2 + x3);
                            acc1.update(w1, 0.5f*x1 + x2 + 0.5f*x3);
                            acc2.update(w2, 0.5f*x0 + x1 + x2 + 0.5f*x3);
                        }
                    }

                    out_max.at({b,p0,mout,tout}) = acc0.rmax;
                    out_max.at({b,p0+1,mout,tout}) = acc1.rmax;
                    out_max.at({b,p0+2,mout,tout}) = acc2.rmax;
                    
                    out_ssq.at({b,p0,mout,tout}) = acc0.rssq;
                    out_ssq.at({b,p0+1,mout,tout}) = acc1.rssq;
                    out_ssq.at({b,p0+2,mout,tout}) = acc2.rssq;
                }
            }
        }

        if (E <= 2*pow2(ids))
            return;

        // Downsample, for next iteration of the loop.

        // Define constants (s,t0,dt) so that downsampled array at time t
        // is obtained from input array at times (s*t + t0), (s*t + t0 + dt).
        // Note: t0 is a "logical" time index, that does not incorporate prepadding.
        
        int s = (isamp < integer_log2(Dcore)) ? 2 : 1;
        // int dt = pow2(ids - isamp);   // already defined above
        int t0 = (s-1) - dt;
        xassert(t0 <= 0);
        xassert(t0 >= -Tpp);

        long Tds2 = xdiv(Tds,s);
        long Tpp2 = (Tpp+t0)/s;  // round down
        xassert(Tpp2 >= 0);

        Array<float> in_x2({B,Min,Tds2+Tpp2}, af_uhost);
        
        // t1 = version of t0 which accounts for prepadding
        int t1 = (t0 - Tpp2*s) + Tpp;
        int tlast = s*(Tds2+Tpp2-1) + t1 + dt;  // only used for range check
        
        // Range check for loop below.
        xassert(t1 >= 0);
        xassert(tlast < Tds+Tpp);
        
        for (long b = 0; b < B; b++) {
            for (long m = 0; m < Min; m++) {
                float *src = &in_x.at({b,m,0});
                float *dst = &in_x2.at({b,m,0});

                for (long i = 0; i < (Tds2+Tpp2); i++)
                    dst[i] = one_over_sqrt2 * (src[s*i+t1] + src[s*i+t1+dt]);
            }
        }

        in_x = in_x2;
        Tpp = Tpp2;
        isamp += integer_log2(s);
        ids++;
    }
}


// -------------------------------------------------------------------------------------------------
//
// GpuPeakFindingKernel


GpuPeakFindingKernel::GpuPeakFindingKernel(const PeakFindingKernelParams &params_) :
    PeakFindingKernel(params_, 0)
{
    RegistryKey key;
    key.dtype = params.dtype;
    key.M = params.dm_downsampling_factor;
    key.E = params.max_kernel_width;
    key.Dout = params.time_downsampling_factor;

    // Call static member function GpuPeakFindingKernel::registry().
    this->registry_value = registry().get(key);
    this->Dcore = registry_value.Dcore;
    
    xassert(registry_value.P == nprofiles);

    // Number of elements (not bytes) per beam
    long n = 2 * nprofiles * ndm_out * nt_out;   // (out_max, out_ssq)
    n += params.ndm_in * params.nt_in;           // in
    n += nprofiles * params.ndm_in;              // wt
    
    this->bw_per_launch.nbytes_gmem = params.beams_per_batch * n * xdiv(params.dtype.nbits,8);
    this->bw_per_launch.kernel_launches = 1;
}


void GpuPeakFindingKernel::allocate()
{
    if (is_allocated)
        throw runtime_error("GpuPeakFindingKernel: double call to allocate()");

    // Note 'af_zero' flag here.
    long ninner = registry_value.P32 * xdiv(32, params.dtype.nbits);
    std::initializer_list<long> shape = { params.total_beams, ndm_out, ninner };
    this->persistent_state = Array<void> (params.dtype, shape, af_zero | af_gpu);
    this->is_allocated = true;
}


void GpuPeakFindingKernel::launch(Array<void> &out_max, Array<void> &out_ssq, const Array<void> &in, const Array<void> &wt, long ibatch, cudaStream_t stream)
{
    _check_args(out_max, out_ssq, in, wt, params.dtype, ibatch);

    xassert(this->is_allocated);    
    xassert(out_max.on_gpu());
    xassert(out_ssq.on_gpu());
    xassert(in.on_gpu());
    xassert(wt.on_gpu());

    int W = registry_value.W;
    uint Bx = (ndm_out + W - 1) / W;
    dim3 nblocks = {Bx, uint(params.beams_per_batch), 1};
        
    char *pstate = (char *) persistent_state.data;
    pstate += ibatch * params.beams_per_batch * ndm_out * registry_value.P32 * 4;

    long Tout32 = xdiv(nt_out * params.dtype.nbits, 32);

    registry_value.full_kernel <<< nblocks, 32*W, 0, stream >>>
        (out_max.data, out_ssq.data, pstate, in.data, wt.data, ndm_out, Tout32);

    CUDA_PEEK("pf kernel launch");    
}


// -------------------------------------------------------------------------------------------------
//
// Kernel registry.


struct PfRegistry : public GpuPeakFindingKernel::Registry
{
    using Key = GpuPeakFindingKernel::RegistryKey;
    using Val = GpuPeakFindingKernel::RegistryValue;

    virtual void add(const Key &key, const Val &val, bool debug) override
    {
        // Just check that all members have been initialized.
        // (In the future, I may add more argument checking here.)
        
        xassert((key.dtype == Dtype::native<float>()) || (key.dtype == Dtype::native<__half>()));
        xassert(key.M > 0);
        xassert(key.E > 0);
        xassert(key.Dout > 0);
        xassert(val.Dcore > 0);
        xassert(val.W > 0);
        xassert(val.P > 0);
        xassert((key.E == 1) || (val.P32 > 0));
        xassert(val.full_kernel != nullptr);
        xassert(val.reduce_only_kernel != nullptr);

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
    
    static PfRegistry reg;
    return reg;  // note: thread-safe (as of c++11)
}

bool operator==(const GpuPeakFindingKernel::RegistryKey &k1, const GpuPeakFindingKernel::RegistryKey &k2)
{
    return (k1.dtype == k2.dtype) && (k1.M == k2.M) && (k1.E == k2.E) && (k1.Dout == k2.Dout);
}

ostream &operator<<(ostream &os, const GpuPeakFindingKernel::RegistryKey &k)
{
    os << "GpuPeakFindingKernel(dtype=" << k.dtype << ", M=" << k.M << ", E=" << k.E << ", Dout=" << k.Dout << ")";
    return os;
}

ostream &operator<<(ostream &os, const GpuPeakFindingKernel::RegistryValue &v)
{
    os << "warps_per_threadblock=" << v.W << ", Dcore=" << v.Dcore;
    return os;
}


// -------------------------------------------------------------------------------------------------
//
// FrequencySubbands
//
// Note: there is a similar python class (pirate_frb.cuda_generator.FrequencySubbands), so changes
// made here should also be reflected there.


FrequencySubbands::FrequencySubbands(const vector<long> &subband_counts_) :
    subband_counts(subband_counts_)
{
    validate_subband_counts(subband_counts);

    this->pf_rank = subband_counts.size() - 1;
    this->F = 0;
    this->M = 0;
    
    for (long level = 0; level <= pf_rank; level++) {
        for (long b = 0; b < subband_counts.at(level); b++) {
            long s = pow2(max(level-1,0L));   // spacing between bands
            long f = this->F;                 // current value

            this->f_to_ilo.push_back(b*s);
            this->f_to_ihi.push_back(b*s + pow2(level));
                
            for (long d = 0; d < pow2(level); d++) {
                this->m_to_f.push_back(f);
                this->m_to_d.push_back(d);
            }

            this->M += pow2(level);
            this->F += 1;
        }
    }

    xassert_eq(m_to_f.size(), uint(M));
    xassert_eq(f_to_ilo.size(), uint(F));
}


// Static member function
void FrequencySubbands::validate_subband_counts(const std::vector<long> &subband_counts)
{
    long pf_rank = subband_counts.size() - 1;

    xassert(subband_counts.size() > 0);
    xassert_eq(subband_counts.at(pf_rank), 1);  // must search full band
    
    // Currently, pf_rank=4 is max value supported by the peak-finding kernel,
    // so a larger value would indicate a bug (such as using the total tree rank
    // instead of the peak-finding rank).
    if (pf_rank > 4)
        throw std::runtime_error("FrequencySubbands: max allowed pf_rank is 4. This may change in the future.");
    
    for (long level = 0; level <= pf_rank; level++) {
        // Level 0 is special (non-overlapping bands).
        long max_bands = (level > 0) ? (pow2(pf_rank+1-level)-1) : pow2(pf_rank);
        xassert_ge(subband_counts.at(level), 0);
        xassert_le(subband_counts.at(level), max_bands);
    }        
}


void FrequencySubbands::show_token(uint token, ostream &os) const
{
    // (t) | (p << 8) | (m << 16)
    uint t = (token) & 0xffu;
    uint p = (token >> 8) & 0xffu;
    uint m = (token >> 16);

    os << " -> (t=" << t << ", p=" << p << ", m=" << m << ")";

    if (m < M) {
        os << " -> BAD M-VALUE";
        return;
    }

    long f = m_to_f.at(m);
    long d = m_to_d.at(m);
    long f0 = f_to_ilo.at(f);
    long f1 = f_to_ihi.at(f);
    os << " -> (f0=" << f0 << ", f1=" << f1 << ", d=" << d << ")";
}


void FrequencySubbands::show(ostream &os) const
{
    os << "FrequencySubbands(subband_counts=" << ksgpu::tuple_str(subband_counts) << ")\n"
       << "    " << "pf_rank=" << pf_rank << ", F=" << F << ", M=" << M << "\n";

    for (long m = 0; m < M; m++) {
        long f = m_to_f.at(m);
        long d = m_to_d.at(m);
        long f0 = f_to_ilo.at(f);
        long f1 = f_to_ihi.at(f);
        os << "    m=" << m << ": d=" << d << ", f0=" << f0 << ", f1=" << f1 << "\n";
    }
}


// -------------------------------------------------------------------------------------------------
//
// GpuPfWeightLayout


void GpuPfWeightLayout::validate() const
{
    Dtype fp32 = Dtype::native<float> ();
    Dtype fp16 = Dtype::native<__half> ();

    xassert((dtype == fp32) || (dtype == fp16));
    xassert(F > 0);
    xassert(P > 0);

    xassert(is_power_of_two(Pinner));
    xassert(is_power_of_two(Tinner));
    xassert(Pouter == (P+Pinner-1)/Pinner);   // round up
    xassert(touter_byte_stride >= Pouter * F * Tinner * xdiv(dtype.nbits,8));
    xassert_divisible(touter_byte_stride, 128);
}


vector<long> GpuPfWeightLayout::get_shape(long nbeams, long ndm_wt, long nt_wt) const
{
    long Touter = xdiv(nt_wt, Tinner);   // must divide evenly
    return { nbeams, ndm_wt, Touter, Pouter, F, Tinner, Pinner };
}

vector<long> GpuPfWeightLayout::get_strides(long nbeams, long ndm_wt, long nt_wt) const
{
    long Touter = xdiv(nt_wt, Tinner);   // must divide evenly
    long S = xdiv(touter_byte_stride * 8, dtype.nbits);
    return { ndm_wt*Touter*S, Touter*S, S, F*Tinner*Pinner, Tinner*Pinner, Pinner, 1 };
}

Array<void> GpuPfWeightLayout::to_gpu(const Array<float> &src)
{
    this->validate();
    
    if (src.ndim != 5) {
        stringstream ss;
        ss << "GpuPfWeightLayout::to_gpu(): expected shape (nbeams, ndm_wt, nt_wt, P, F), got " << src.shape_str();
        throw runtime_error(ss.str());
    }

    xassert_eq(src.shape[3], P);
    xassert_eq(src.shape[4], F);

    long nbeams = src.shape[0];
    long ndm_wt = src.shape[1];
    long nt_wt = src.shape[2];
    long Touter = xdiv(nt_wt, Tinner);   // must divide evenly
    
    vector<long> shape = get_shape(nbeams, ndm_wt, nt_wt);
    vector<long> strides = get_strides(nbeams, ndm_wt, nt_wt);

    // Note: code below is poorly optimized! (Intended for unit tests.)

    // On host, dtype=float32, GPU shape, contiguous strides.
    Array<float> tmp(shape, af_rhost | af_zero);

    for (long b = 0; b < nbeams; b++) {
        for (long dm_wt = 0; dm_wt < ndm_wt; dm_wt++) {
            for (long touter = 0; touter < Touter; touter++) {
                for (long pouter = 0; pouter < Pouter; pouter++) {
                    for (long f = 0; f < F; f++) {
                        for (long tinner = 0; tinner < Tinner; tinner++) {
                            for (long pinner = 0; pinner < Pinner; pinner++) {
                                long tw = touter*Tinner + tinner;
                                long p = min(pouter*Pinner + pinner, P-1);

                                float w = src.at({b,dm_wt,tw,p,f});
                                tmp.at({b,dm_wt,touter,pouter,f,tinner,pinner}) = w;
                            }
                        }
                    }
                }
            }
        }
    }

    Array<void> tmp2 = tmp.convert(dtype);
    
    // Allocate GPU array with non-contiguous touter-stride.
    Array<void> dst(dtype, shape, strides, af_gpu | af_zero);    

    dst.fill(tmp2);  // copy CPU->GPU
    return dst;
}


// -------------------------------------------------------------------------------------------------
//
// PeakFindingKernelParams2


void PeakFindingKernelParams2::validate() const
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
// ReferencePeakFindingKernel2


ReferencePeakFindingKernel2::ReferencePeakFindingKernel2(const PeakFindingKernelParams2 &params_, long Dcore_) :
    params(params_), fs(params_.subband_counts), Dcore(Dcore_)
{
    params.validate();

    const PeakFindingKernelParams2 &p = params;
    long B = p.beams_per_batch;
    long D = p.ndm_out;
    long W = p.max_kernel_width;
    long M = fs.M;

    this->nbatches = xdiv(p.total_beams, p.beams_per_batch);
    this->nprofiles = 3 * integer_log2(p.max_kernel_width) + 1;
    this->Dout = xdiv(p.nt_in, p.nt_out);
    this->tpad = max(2*W, 4L);
    this->pstate = Array<float> ({p.total_beams, p.ndm_out, fs.M, tpad}, af_uhost | af_zero); 
    this->num_levels = max(integer_log2(W), 1);

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


void ReferencePeakFindingKernel2::apply(
    ksgpu::Array<float> &out_max,      // shape (beams_per_batch, ndm_out, nt_out)
    ksgpu::Array<uint> &out_argmax,    // shape (beams_per_batch, ndm_out, nt_out)
    const ksgpu::Array<float> &in,     // shape (beams_per_batch, ndm_out, M, nt_in)
    const ksgpu::Array<float> &wt,     // shape (beams_per_batch, ndm_wt, nt_wt, nprofiles, F)
    long ibatch, bool debug)
{
    const PeakFindingKernelParams2 &p = params;
    xassert_shape_eq(out_max, ({p.beams_per_batch, p.ndm_out, p.nt_out}));
    xassert_shape_eq(out_argmax, ({p.beams_per_batch, p.ndm_out, p.nt_out}));
    xassert_shape_eq(in, ({p.beams_per_batch, p.ndm_out, fs.M, p.nt_in}));
    xassert_shape_eq(wt, ({p.beams_per_batch, p.ndm_wt, p.nt_wt, nprofiles, fs.F}));
    // contiguity requirements are checked in _init_tmp_arrays() and _peak_find().
 
    xassert(out_max.on_host());
    xassert(out_argmax.on_host());
    xassert(in.on_host());
    xassert(wt.on_host());

    xassert_eq(ibatch, expected_ibatch);
    expected_ibatch = (ibatch + 1) % nbatches;

    _init_tmp_arrays(in, ibatch);
    _peak_find(out_max, out_argmax, wt, debug);
}


void ReferencePeakFindingKernel2::_init_tmp_arrays(const Array<float> &in, long ibatch)
{
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
}


// helper for ReferencePeakFindingKernel2::_peak_find()
static inline void _update_pf2(float &maxval, uint &argmax, float val, uint token)
{
    argmax = (val > maxval) ? token : argmax;
    maxval = std::max(maxval, val);
}


// out_*: shape (beams_per_batch, ndm_out, nt_out)
// wt: shape (beams_per_batch, ndm_wt, nt_wt, P, F)
void ReferencePeakFindingKernel2::_peak_find(Array<float> &out_max, Array<uint> &out_argmax, const Array<float> &wt, bool debug)
{
    long B = params.beams_per_batch;
    long D = params.ndm_out;
    long P = nprofiles;
    long M = fs.M;
    long F = fs.F;

    long Wds = xdiv(params.ndm_out, params.ndm_wt);  // downsampling factor ndm_out -> ndm_wt
    long Tds = xdiv(params.nt_out, params.nt_wt);    // downsampling factor nt_out -> nt_wt
    long nt_out = params.nt_out;

    xassert_shape_eq(out_max, ({B,D,nt_out}));
    xassert_shape_eq(out_argmax, ({B,D,nt_out}));
    xassert_shape_eq(wt, ({B, params.ndm_wt, params.nt_wt, nprofiles, fs.F}));
    xassert(wt.get_ncontig() >= 2);  // (p,f) must be contiguous

    for (long b = 0; b < B; b++) {
        for (long d = 0; d < D; d++) {
            for (long tout = 0; tout < nt_out; tout++) {
                const float *wp = &wt.at({b,d/Wds,tout/Tds,0,0});  // shape (P,F) contiguous

                // Inner loops compute one output array element, by looping over
                // peak-finding kernels, with loop ordering (p,m,n).

                float maxval = -1.0e30f;
                uint argmax = ~0u;  // token

                for (long l = 0; l < num_levels; l++) {
                    float *in = &tmp_arr.at(l).at({b,d,0,0});
                    int mstr = tmp_nt[l];   // m-stride of input array
                    int dt = tmp_dt[l];     // used below when computing tokens
                    int N = tmp_nout[l];    // count
                    int S = tmp_sout[l];    // spacing
                    int I = tmp_iout[l];    // base

                    for (int m = 0; m < M; m++) {
                        int f = fs.m_to_f[m];
                        float w0 = l ? 0.0f : wp[f];      // p = 0 (only for l=0)
                        float w1 = wp[(3*l+1)*F + f];     // p = (3*l+1)
                        float w2 = wp[(3*l+2)*F + f];     // p = (3*l+2)
                        float w3 = wp[(3*l+3)*F + f];     // p = (3*l+3)

                        // Each iteration of the n-loop corresponds to one time sample in the 
                        // tmp[l] array, or (dt) time samples in the original input array.

                        for (int n = 0; n < N; n++) {
                            float x0 = in[m*mstr + I + tout*N + n - 3*S];
                            float x1 = in[m*mstr + I + tout*N + n - 2*S];
                            float x2 = in[m*mstr + I + tout*N + n - S];
                            float x3 = in[m*mstr + I + tout*N + n];

                            uint token0 = (m << 16)| (n*dt);  // includes (m,n) but not p
                            uint token1 = token0 | ((3*l+1) << 8);    // include p=3*l+1
                            uint token2 = token0 | ((3*l+2) << 8);    // include p=3*l+2
                            uint token3 = token0 | ((3*l+3) << 8);    // include p=3*l+3

                            float y0 = x3;
                            float y1 = (x2 + x3);
                            float y2 = (0.5f*x1 + x2 + 0.5f*x3);
                            float y3 = (0.5f*x0 + x1 + x2 + 0.5f*x3);

                            if (l == 0)
                                _update_pf2(maxval, argmax, w0*y0, token0);

                            if (P > 1) {
                                _update_pf2(maxval, argmax, w1*y1, token1);
                                _update_pf2(maxval, argmax, w2*y2, token2);
                                _update_pf2(maxval, argmax, w3*y3, token3);
                            }

                            if (debug && (b == 0) && (d==0) && (tout==2)) {
                                cout << "cpu peak-finder: b=" << b << ", d=" << d << ", tout=" << tout 
                                     << ", level=" << l << ", m=" << m << ", n=" << n << "\n";

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

                out_max.at({b,d,tout}) = maxval;
                out_argmax.at({b,d,tout}) = argmax;
            }
        }
    }
}


// Note that an 'in' array is not an argument -- this function uses the contents of 'tmp_arr'.
void ReferencePeakFindingKernel2::eval_tokens(Array<float> &out_max, const Array<uint> &in_tokens, const Array<float> &wt)
{
    long B = params.beams_per_batch;
    long D = params.ndm_out;
    long M = fs.M;
    long F = fs.F;
    long P = nprofiles;
    long Wds = xdiv(params.ndm_out, params.ndm_wt);  // downsampling factor ndm_out -> ndm_wt
    long Tds = xdiv(params.nt_out, params.nt_wt);    // downsampling factor nt_out -> nt_wt
    long nt_out = params.nt_out;

    xassert_shape_eq(out_max, ({B,D,nt_out}));
    xassert_shape_eq(in_tokens, ({B,D,nt_out}));
    xassert_shape_eq(wt, ({B, params.ndm_wt, params.nt_wt, P, F}));
    xassert(wt.get_ncontig() >= 2);  // (p,f) must be contiguous

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

                // t = n*dt
                long dt = tmp_dt.at(l);
                long n = t / dt;

                if (t != n*dt)
                    throw _bad_token(token, "t is not divisible by dt");

                // Token parsing (token -> (m,n,p)) ends here!

                long f = fs.m_to_f.at(m);
                float w = wt.at({b, d/Wds, tout/Tds, p, f});

                int N = tmp_nout[l];       // count
                int S = tmp_sout[l];       // spacing
                int I = tmp_iout[l];       // base

                float x0 = tmp_arr.at(l).at({b, d, m, I + tout*N + n - 3*S});
                float x1 = tmp_arr.at(l).at({b, d, m, I + tout*N + n - 2*S});
                float x2 = tmp_arr.at(l).at({b, d, m, I + tout*N + n - S});
                float x3 = tmp_arr.at(l).at({b, d, m, I + tout*N + n});

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

                    cout << "  wt.at(" << b << "," << (d/Wds) << "," << (tout/Tds) << "," << p << "," << f << ")"
                         << " = " << wt.at({b,d/Wds,tout/Tds,p,f}) << endl;

                    for (int i = 0; i < 4; i++)
                        cout << "  tmp_arr.at(" << l << ").at(" << b << "," << d << "," << m << "," << (I + tout*N + n + (i-3)*S) << ")"
                             << " = " << tmp_arr.at(l).at({b, d, m, I + tout*N + n + (i-3)*S}) << endl;

                    cout << "    at level l: tpad=" << tpad << ", dt=" << tmp_dt.at(l) << ", N=" << N << ", S=" << S << ", I=" << I << endl;
                }
#endif
            }
        }
    }
}


std::runtime_error ReferencePeakFindingKernel2::_bad_token(uint token, const char *why)
{
    stringstream ss;
    ss << "ReferencePeakFindingKernel2:: eval_tokens(): bad token " << hex_str(token) << " (" << why << ")";
    return runtime_error(ss.str());
}


// Make a mean-zero input array for testing.
// Returns shape (nbeams_per_batch, ndm_out, fs.M, nt_in)
Array<float> ReferencePeakFindingKernel2::make_random_input_array()
{
    long B = params.beams_per_batch;
    long D = params.ndm_out;
    long T = params.nt_in;
    long M = fs.M;

    Array<float> ret({B,D,M,T}, af_rhost);
    for (long i = 0; i < ret.size; i++)
        ret.data[i] = rand_uniform(-1.0f, 1.0f);

    return ret;
}

// This weird procedure for making a random weights array is intended to
// expose corner cases in the peak-finding kernel.
// Returned array has shape (beams_per_batch, ndm_wt, nt_wt, nprofiles, F).
Array<float> ReferencePeakFindingKernel2::make_random_weights()
{
    long B = params.beams_per_batch;
    long D = params.ndm_wt;
    long T = params.nt_wt;
    long P = nprofiles;
    long F = fs.F;

    xassert_eq(P, 3*(P/3)+1);

    Array<float> ret({B,D,T,P,F}, af_rhost);
    vector<float> wp(P);

    long nouter = B*D*T;

    for (long i = 0; i < nouter; i++) {
        float p0 = rand_uniform(1.0f, 2.0f);

        wp[0] = (rand_uniform() < p0) ? rand_uniform() : 0.0f;
        for (long l = 0; l < (P/3); l++) {
            wp[3*l+1] = (rand_uniform() < p0) ? (rand_uniform() * rsqrtf(2.0f * pow2(l))) : 0.0f;
            wp[3*l+2] = (rand_uniform() < p0) ? (rand_uniform() * rsqrtf(1.5f * pow2(l))) : 0.0f;
            wp[3*l+3] = (rand_uniform() < p0) ? (rand_uniform() * rsqrtf(2.5f * pow2(l))) : 0.0f;
        }

        for (long p = 0; p < P; p++)
            for (long f = 0; f < F; f++)
                ret.data[i*P*F + p*F + f] = rand_uniform(1.0f, 2.0f) * wp[p];
    }

    return ret;
}


// -------------------------------------------------------------------------------------------------
//
// GpuPeakFindingKernel2


GpuPeakFindingKernel2::GpuPeakFindingKernel2(const PeakFindingKernelParams2 &params_) :
    params(params_), fs(params_.subband_counts)
{
    params.validate();

    registry_key.dtype = params.dtype;
    registry_key.subband_counts = fs.subband_counts;
    registry_key.Dout = xdiv(params.nt_in, params.nt_out);
    registry_key.E = params.max_kernel_width;

    // Recall the definition of Tinner (used for weight layout, see comments in
    // cuda_generator.PeakFinder2.y):
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
    // this->bw_per_launch.nbytes_gmem = params.beams_per_batch * n * xdiv(params.dtype.nbits,8);
    // this->bw_per_launch.kernel_launches = 1;
}


void GpuPeakFindingKernel2::allocate()
{
    if (is_allocated)
        throw runtime_error("GpuPeakFindingKernel2: double call to allocate()");

    // Allocate persistent_state. Note 'af_zero' flag here.
    std::initializer_list<long> shape = { params.total_beams, params.ndm_out, registry_value.PW32 };
    this->persistent_state = Array<uint> (shape, af_zero | af_gpu);
    this->is_allocated = true;
}


void GpuPeakFindingKernel2::launch(
    ksgpu::Array<void> &out_max,      // shape (beams_per_batch, ndm_out, nt_out)
    ksgpu::Array<uint> &out_argmax,   // shape (beams_per_batch, ndm_out, nt_out)
    const ksgpu::Array<void> &in,     // shape (beams_per_batch, ndm_out, M, nt_in)
    const ksgpu::Array<void> &wt,     // from GpuPfWeightLayout::to_gpu()
    long ibatch,                      // 0 <= ibatch < nbatches
    cudaStream_t stream)              // NULL stream is allowed, but is not the default);
{
    const PeakFindingKernelParams2 &p = params;

    xassert(this->is_allocated);
    xassert(out_max.dtype == dtype);
    xassert(in.dtype == dtype);
    xassert(wt.dtype == dtype);

    xassert_shape_eq(out_max, ({p.beams_per_batch, p.ndm_out, p.nt_out}));
    xassert_shape_eq(out_argmax, ({p.beams_per_batch, p.ndm_out, p.nt_out}));
    xassert_shape_eq(in, ({p.beams_per_batch, p.ndm_out, fs.M, p.nt_in}));

    if (!wt.shape_equals(expected_wt_shape)) {
        stringstream ss;
        ss << "GpuPeakFindingKernel2::launch(): wt.shape=" << wt.shape_str()
           << ", expected_wt_shape=" << ksgpu::tuple_str(expected_wt_shape);
        throw runtime_error(ss.str());
    }

    if (!wt.strides_equal(expected_wt_strides)) {
        stringstream ss;
        ss << "GpuPeakFindingKernel2::launch(): wt.strides=" << wt.stride_str()
           << ", expected_wt_strides=" << ksgpu::tuple_str(expected_wt_strides);
        throw runtime_error(ss.str());
    }

    xassert(out_max.is_fully_contiguous());
    xassert(out_argmax.is_fully_contiguous());
    xassert(in.is_fully_contiguous());

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


// Static member function
void GpuPeakFindingKernel2::test(bool short_circuit)
{
    RegistryKey key = registry().get_random_key();
    long simd_width = xdiv(32, key.dtype.nbits);
    long Tinner = key.Tinner;

    long nt_in_per_wt = (Tinner > 1) ? xdiv(32*simd_width,Tinner) : ((32 * simd_width) << rand_int(0,3));
    long nt_in_divisor = max(32*simd_width, nt_in_per_wt);
    long ndm_wt = 1 << rand_int(0,3);
    long ndm_out = ndm_wt << rand_int(0,3);

    auto v = ksgpu::random_integers_with_bounded_product(4, 100000 / (nt_in_divisor * ndm_out));
    long nchunks = v[0];
    long nt_in_per_chunk = nt_in_divisor * v[1];
    long beams_per_batch = v[2];
    long total_beams = v[2] * v[3];

    long nt_out_per_chunk = xdiv(nt_in_per_chunk, key.Dout);
    long nt_wt_per_chunk = xdiv(nt_in_per_chunk, nt_in_per_wt);

    PeakFindingKernelParams2 params_small;
    params_small.subband_counts = key.subband_counts;
    params_small.dtype = key.dtype;
    params_small.max_kernel_width = key.E;
    params_small.beams_per_batch = beams_per_batch;
    params_small.total_beams = total_beams;
    params_small.ndm_out = ndm_out;
    params_small.ndm_wt = ndm_wt;
    params_small.nt_in = nt_in_per_chunk;
    params_small.nt_out = nt_out_per_chunk;
    params_small.nt_wt = nt_wt_per_chunk;
    params_small.validate();

    PeakFindingKernelParams2 params_large;
    params_large.subband_counts = key.subband_counts;
    params_large.dtype = key.dtype;
    params_large.max_kernel_width = key.E;
    params_large.beams_per_batch = total_beams;
    params_large.total_beams = total_beams;
    params_large.ndm_out = ndm_out;
    params_large.ndm_wt = ndm_wt;
    params_large.nt_in = nchunks * nt_in_per_chunk;
    params_large.nt_out = nchunks * nt_out_per_chunk;
    params_large.nt_wt = nchunks * nt_wt_per_chunk;
    params_large.validate();

    GpuPeakFindingKernel2 gpu_kernel(params_small);   // just test constructor for now
    ReferencePeakFindingKernel2 ref_kernel_small(params_small, gpu_kernel.Dcore);
    ReferencePeakFindingKernel2 ref_kernel_large(params_large, gpu_kernel.Dcore);

    cout << "GpuPeakFindingKernel2::test():"
         << " dtype=" << key.dtype.str() 
         << ", subbands=" << ksgpu::tuple_str(key.subband_counts)
         << ", W=" << key.E
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
    long F = gpu_kernel.fs.F;
    long M = gpu_kernel.fs.M;

    Array<float> cpu_in_large = ref_kernel_large.make_random_input_array();
    xassert_shape_eq(cpu_in_large, ({total_beams, ndm_out, M, nchunks * nt_in_per_chunk}));

    Array<float> cpu_wt_large = ref_kernel_large.make_random_weights();
    xassert_shape_eq(cpu_wt_large, ({total_beams, ndm_wt, nchunks * nt_wt_per_chunk, P, F}));

    Array<float> cpu_out_large({total_beams, ndm_out, nchunks * nt_out_per_chunk}, af_rhost | af_zero);
    Array<uint> cpu_argmax_large({total_beams, ndm_out, nchunks * nt_out_per_chunk}, af_rhost | af_zero);
    ref_kernel_large.apply(cpu_out_large, cpu_argmax_large, cpu_in_large, cpu_wt_large, 0);

    // This is a nontrivial test of the reference peak-finder.
    Array<float> cpu_out2_large({total_beams, ndm_out, nchunks * nt_out_per_chunk}, af_rhost | af_zero);
    ref_kernel_large.eval_tokens(cpu_out2_large, cpu_argmax_large, cpu_wt_large);
    assert_arrays_equal(cpu_out_large, cpu_out2_large, "cpu_out_large", "cpu_out2_large", {"b","d","tout"});

    gpu_kernel.allocate();

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

            assert_arrays_equal(cpu_out_small, gpu_out, "cpu_out_small", "gpu_out", {"b","d","tout"});

            gpu_argmax = gpu_argmax.to_host();
            Array<float> gpu_out2({beams_per_batch, ndm_out, nt_out_per_chunk}, af_rhost | af_zero);
            ref_kernel_small.eval_tokens(gpu_out2, gpu_argmax, cpu_wt_small);

            double eps = 10.0 * key.dtype.precision();
            assert_arrays_equal(cpu_out_small, gpu_out2, "cpu_out_small", "gpu_out2", {"b","d","tout"}, eps, eps);
        }
    }
}


// -------------------------------------------------------------------------------------------------
//
// Kernel registry.


struct GpuPf2Registry : public GpuPeakFindingKernel2::Registry
{
    using Key = GpuPeakFindingKernel2::RegistryKey;
    using Val = GpuPeakFindingKernel2::RegistryValue;

    virtual void add(const Key &key, const Val &val, bool debug) override
    {
        // Just check that all members have been initialized.
        // (In the future, I may add more argument checking here.)
        
        xassert((key.dtype == Dtype::native<float>()) || (key.dtype == Dtype::native<__half>()));
        xassert_ge(key.subband_counts.size(), 1);
        xassert(key.Tinner > 0);
        xassert(key.Dout > 0);
        xassert(key.E > 0);
        
        xassert(val.cuda_kernel != nullptr);
        xassert(val.Dcore > 0);
        xassert(val.PW32 >= 0);
        
        val.pf_weight_layout.validate();
        
        // Call add() in base class.
        GpuPeakFindingKernel2::Registry::add(key, val, debug);
    }
};


// Static member function
GpuPeakFindingKernel2::Registry &GpuPeakFindingKernel2::registry()
{
    // Instead of declaring the registry as a static global variable, we declare it as a
    // static local variable in the static member function GpuPeakFindingKernel2::registry().
    // The registry will be initialized the first time that GpuPeakFindingKernel2::registry()
    // is called.
    //
    // This kludge is necessary because the registry is accessed at library initialization
    // time, by callers in other source files, and source files are executed in an
    // arbitrary order.
    
    static GpuPf2Registry reg;
    return reg;  // note: thread-safe (as of c++11)
}

bool operator==(const GpuPeakFindingKernel2::RegistryKey &k1, const GpuPeakFindingKernel2::RegistryKey &k2)
{
    return (k1.dtype == k2.dtype)
        && (k1.subband_counts == k2.subband_counts)
        && (k1.Tinner == k2.Tinner)
        && (k1.Dout == k2.Dout)
        && (k1.E == k2.E);
}

ostream &operator<<(ostream &os, const GpuPeakFindingKernel2::RegistryKey &k)
{
    FrequencySubbands fs(k.subband_counts);
    
    os << "GpuPeakFindingKernel2(dtype=" << k.dtype
       << ", rank=" << fs.pf_rank
       << ", subband_counts=" << ksgpu::tuple_str(k.subband_counts)
       << ", Tinner=" << k.Tinner
       << ", Dout=" << k.Dout
       << ", E=" << k.E
       << ", F=" << fs.F
       << ", M=" << fs.M
       << ")";
    
    return os;
}

ostream &operator<<(ostream &os, const GpuPeakFindingKernel2::RegistryValue &v)
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
       << ", F=" << fs.F
       << ", M=" << fs.M
       << ")";
    
    return os;
}

ostream &operator<<(ostream &os, const PfWeightReaderMicrokernel::RegistryValue &v)
{
    return os;
}


void PfWeightReaderMicrokernel::test()
{
    PfWeightReaderMicrokernel::RegistryKey key = PfWeightReaderMicrokernel::registry().get_random_key();
    PfWeightReaderMicrokernel::RegistryValue val = PfWeightReaderMicrokernel::registry().get(key);

    FrequencySubbands fs(key.subband_counts);
    GpuPfWeightLayout &wl = val.pf_weight_layout;
    
    Dtype dtype = key.dtype;
    int SW = xdiv(32, dtype.nbits);   // simd width
    
    int F = fs.F;
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
    
    // Input array: (1,1,nt_wt,P,F), where the length-1 axes are beams and DMs.
    Array<float> in_cpu({1,1,nt_wt,P,F}, af_rhost | af_random);

    // Output array: (nt_out, Mouter*Minner, Pouter*Pinner)
    Array<float> out_cpu({nt_out,Mpad,Ppad}, af_rhost | af_zero);

    // Emulate PfWeightReader kernel on the CPU.
    for (int tw = 0; tw < nt_wt; tw++) {
        for (int tout = tw*Tspec; tout < (tw+1)*Tspec; tout++) {
            for (int mpad = 0; mpad < Mpad; mpad++) {
                int m = min(mpad, M-1);
                int f = fs.m_to_f.at(m);
                
                for (int ppad = 0; ppad < Ppad; ppad++) {
                    int p = min(ppad, P-1);
                    out_cpu.at({tout,mpad,ppad}) = in_cpu.at({0,0,tw,p,f});
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


void PfOutputMicrokernel::test()
{
    PfOutputMicrokernel::RegistryKey key = PfOutputMicrokernel::registry().get_random_key();
    
    Dtype dtype = key.dtype;
    uint Dout = key.Dout;
    uint nt_in = xdiv(1024, dtype.nbits) * rand_int(1, 100);
    uint nt_out = xdiv(nt_in, Dout);
    
    cout << "test_pf_output2_microkernel: dtype=" << dtype << ", Dout=" << Dout << ", nt_in=" << nt_in << endl;

    Array<float> zin_cpu({4,nt_in}, af_uhost | af_random);
    Array<float> zout_cpu({nt_out}, af_uhost);
    Array<uint> ain_cpu({4,nt_in}, af_uhost);

    // Each (s,tin) pair gets a random uint token.
    //   - token_mapping: (token) -> (s,tin)
    //   - ain_cpu: inverse (s,tin) -> (token)

    std::unordered_map<uint, std::pair<uint,uint>> token_mapping;

    for (uint s = 0; s < 4; s++) {
        for (uint tin = 0; tin < nt_in; tin++) {
            for (;;) {
                uint token = ksgpu::default_rng();
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
    CUDA_PEEK("pf_output2_test_kernel");

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
