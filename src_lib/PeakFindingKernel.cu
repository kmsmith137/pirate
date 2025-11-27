#include "../include/pirate/PeakFindingKernel.hpp"
#include "../include/pirate/inlines.hpp"
#include "../include/pirate/utils.hpp"

#include <mutex>
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
    this->pf_rank = subband_counts.size() - 1;
    this->F = 0;
    this->M = 0;
    
    xassert_ge(subband_counts.size(), 1);
    xassert_eq(subband_counts.at(pf_rank), 1);  // must search full band

    // Currently, pf_rank=4 is max value supported by the peak-finding kernel,
    // so a larger value would indicate a bug (such as using the total tree rank
    // instead of the peak-finding rank).
    xassert_le(pf_rank, 4);

    for (long level = 0; level <= pf_rank; level++) {
        // Level 0 is special (non-overlapping bands).
        long max_bands = (level > 0) ? (pow2(pf_rank+1-level)-1) : pow2(pf_rank);
        xassert_ge(subband_counts.at(level), 0);
        xassert_le(subband_counts.at(level), max_bands);

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



Array<void> GpuPfWeightLayout::to_gpu(const Array<float> &src)
{
    this->validate();
    
    if (src.ndim != 5) {
        stringstream ss;
        ss << "GpuPfWeightLayout::to_gpu(): expected shape (nbeams, Dbar, Tbar, P, F), got " << src.shape_str();
        throw runtime_error(ss.str());
    }

    xassert_eq(src.shape[3], P);
    xassert_eq(src.shape[4], F);

    long nbeams = src.shape[0];
    long Dbar = src.shape[1];
    long Tbar = src.shape[2];
    long Touter = xdiv(Tbar, Tinner);   // must divide evenly
    
    long S = xdiv(touter_byte_stride * 8, dtype.nbits);
    vector<long> shape = { nbeams, Dbar, Touter, Pouter, F, Tinner, Pinner };
    vector<long> strides = { Dbar*Touter*S, Touter*S, S, F*Tinner*Pinner, Tinner*Pinner, Pinner, 1 };

    // Note: code below is poorly optimized! (Intended for unit tests.)

    // On host, dtype=float32, GPU shape, contiguous strides.
    Array<float> tmp(shape, af_rhost | af_zero);

    for (long b = 0; b < nbeams; b++) {
        for (long dbar = 0; dbar < Dbar; dbar++) {
            for (long touter = 0; touter < Touter; touter++) {
                for (long pouter = 0; pouter < Pouter; pouter++) {
                    for (long f = 0; f < F; f++) {
                        for (long tinner = 0; tinner < Tinner; tinner++) {
                            for (long pinner = 0; pinner < Pinner; pinner++) {
                                long tbar = touter*Tinner + tinner;
                                long p = min(pouter*Pinner + pinner, P-1);

                                float w = src.at({b,dbar,tbar,p,f});
                                tmp.at({b,dbar,touter,pouter,f,tinner,pinner}) = w;
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
// TestPfWeightReader


struct TestPfWeightReaderRegistry : public TestPfWeightReader::Registry
{
    using Key = TestPfWeightReader::RegistryKey;
    using Val = TestPfWeightReader::RegistryValue;

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
        TestPfWeightReader::Registry::add(key, val, debug);
    }
};


// Static member function
TestPfWeightReader::Registry &TestPfWeightReader::registry()
{
    // Instead of declaring the registry as a static global variable, we declare it as a
    // static local variable in the static member function TestPfWeightReader::registry().
    // The registry will be initialized the first time that TestPfWeightReader::registry()
    // is called.
    //
    // This kludge is necessary because the registry is accessed at library initialization
    // time, by callers in other source files, and source files are executed in an
    // arbitrary order.
    
    static TestPfWeightReaderRegistry reg;
    return reg;  // note: thread-safe (as of c++11)
}

bool operator==(const TestPfWeightReader::RegistryKey &k1, const TestPfWeightReader::RegistryKey &k2)
{
    return (k1.dtype == k2.dtype)
        && (k1.subband_counts == k2.subband_counts)
        && (k1.Dcore == k2.Dcore)
        && (k1.Tinner == k2.Tinner)
        && (k1.P == k2.P);
}

ostream &operator<<(ostream &os, const TestPfWeightReader::RegistryKey &k)
{
    FrequencySubbands fs(k.subband_counts);
    
    os << "TestPfWeightReader(dtype=" << k.dtype
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

ostream &operator<<(ostream &os, const TestPfWeightReader::RegistryValue &v)
{
    return os;
}


void test_pf_weight_reader_microkernel()
{
    TestPfWeightReader::RegistryKey key = TestPfWeightReader::registry().get_random_key();
    TestPfWeightReader::RegistryValue val = TestPfWeightReader::registry().get(key);

    FrequencySubbands fs(key.subband_counts);
    GpuPfWeightLayout &wl = val.pf_weight_layout;
    
    Dtype dtype = key.dtype;
    int SW = xdiv(32, dtype.nbits);   // simd width
    
    int F = fs.F;
    int M = fs.M;
    int P = wl.P;
    int Dcore = key.Dcore;
    int Tinner = key.Tinner;
    
    // Choose WDt, Tin.
    // If Tinner > 1, then WDt must equal (32*SW)/Tinner, and Tin must be a multiple of (32*SW).
    // If Tinner == 1, then WDt must be a multiple of (32*SW), and Tin must be a multiple of WDt.
    
    auto v = ksgpu::random_integers_with_bounded_product(2, 20);
    int WDt = (Tinner > 1) ? xdiv(32*SW,Tinner) : (32*SW*v[0]);
    int Tin = (Tinner > 1) ? (32*SW*v[0]*v[1]) : (WDt*v[1]);  // number of tree samples (not used for anything)

    cout << "test_pf_weight_reader_microkernel: dtype=" << dtype
         << ", subband_counts=" << ksgpu::tuple_str(key.subband_counts)
         << ", Dcore=" << key.Dcore
         << ", P=" << key.P
         << ", Tinner=" << Tinner
         << ", WDt=" << WDt
         << ", Tin=" << Tin << endl;
    
    int Tbar = xdiv(Tin, WDt);     // number of time samples in weights array (input array to test kernel)
    int Tout = xdiv(Tin, Dcore);   // number of time samples in output array of test kernel
    int Tspec = xdiv(Tout, Tbar);  // number of "spectator" time samples in test kernel
    int Mpad = val.Mouter * val.Minner;
    int Ppad = wl.Pouter * wl.Pinner;    
    
    // Input array: (1,1,Tbar,P,F), where the length-1 axes are beams and DMs.
    Array<float> in_cpu({1,1,Tbar,P,F}, af_rhost | af_random);

    // Output array: (Tout, Mouter*Minner, Pouter*Pinner)
    Array<float> out_cpu({Tout,Mpad,Ppad}, af_rhost | af_zero);

    // Emulate PfWeightReader kernel on the CPU.
    for (int tbar = 0; tbar < Tbar; tbar++) {
        for (int tout = tbar*Tspec; tout < (tbar+1)*Tspec; tout++) {
            for (int mpad = 0; mpad < Mpad; mpad++) {
                int m = min(mpad, M-1);
                int f = fs.m_to_f.at(m);
                
                for (int ppad = 0; ppad < Ppad; ppad++) {
                    int p = min(ppad, P-1);
                    out_cpu.at({tout,mpad,ppad}) = in_cpu.at({0,0,tbar,p,f});
                }
            }
        }
    }

    // Send input array to GPU, using GpuPfWeightLayout::to_gpu().
    Array<void> in_gpu = val.pf_weight_layout.to_gpu(in_cpu);

    // Run kernel on GPU.
    Array<void> out_gpu(dtype, {Tout,Mpad,Ppad}, af_gpu | af_zero | af_guard);
    val.cuda_kernel <<<1,32>>> (out_gpu.data, in_gpu.data, Tin, WDt);
    CUDA_PEEK("pf_weight_reader");

    // Compare.
    assert_arrays_equal(out_cpu, out_gpu, "out_cpu", "out_gpu", {"tout","mpad","ppad"});
}


// -------------------------------------------------------------------------------------------------
//
// TestPfOutput2


struct TestPfOutput2Registry : public TestPfOutput2::Registry
{
    using Key = TestPfOutput2::RegistryKey;
    using Val = TestPfOutput2::RegistryValue;

    virtual void add(const Key &key, const Val &val, bool debug) override
    {
        // Just check that all members have been initialized.
        // (In the future, I may add more argument checking here.)
        
        xassert((key.dtype == Dtype::native<float>()) || (key.dtype == Dtype::native<__half>()));
        xassert(key.Dout > 0);
        xassert(val.cuda_kernel != nullptr);

        // Call add() in base class.
        TestPfOutput2::Registry::add(key, val, debug);
    }
};

// Static member function
TestPfOutput2::Registry &TestPfOutput2::registry()
{
    // Instead of declaring the registry as a static global variable, we declare it as a
    // static local variable in the static member function TestPfOutput2::registry().
    // The registry will be initialized the first time that TestPfOutput2::registry()
    // is called.
    //
    // This kludge is necessary because the registry is accessed at library initialization
    // time, by callers in other source files, and source files are executed in an
    // arbitrary order.
    
    static TestPfOutput2Registry reg;
    return reg;  // note: thread-safe (as of c++11)
}

bool operator==(const TestPfOutput2::RegistryKey &k1, const TestPfOutput2::RegistryKey &k2)
{
    return (k1.dtype == k2.dtype) && (k1.Dout == k2.Dout);
}

ostream &operator<<(ostream &os, const TestPfOutput2::RegistryKey &k)
{
    os << "TestPfOutput2(dtype=" << k.dtype << ", Dout=" << k.Dout << ")";
    return os;
}

ostream &operator<<(ostream &os, const TestPfOutput2::RegistryValue &v)
{
    return os;
}


void test_pf_output_microkernel()
{
    TestPfOutput2::RegistryKey key = TestPfOutput2::registry().get_random_key();
    
    Dtype dtype = key.dtype;
    uint Dout = key.Dout;
    uint Tin = xdiv(1024, dtype.nbits) * rand_int(1, 100);
    uint Tout = xdiv(Tin, Dout);
    
    cout << "test_pf_output2_microkernel: dtype=" << dtype << ", Dout=" << Dout << ", Tin=" << Tin << endl;

    Array<float> zin_cpu({4,Tin}, af_uhost | af_random);
    Array<float> zout_cpu({Tout}, af_uhost);
    Array<uint> ain_cpu({4,Tin}, af_uhost);

    // Each (s,tin) pair gets a random uint token.
    //   - token_mapping: (token) -> (s,tin)
    //   - ain_cpu: inverse (s,tin) -> (token)

    std::unordered_map<uint, std::pair<uint,uint>> token_mapping;

    for (uint s = 0; s < 4; s++) {
        for (uint tin = 0; tin < Tin; tin++) {
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

    for (uint tout = 0; tout < Tout; tout++) {
        float zmax = -1.0e10f;
        for (uint s = 0; s < 4; s++)
            for (uint tin = tout*Dout; tin < (tout+1)*Dout; tin++)
                zmax = fmaxf(zmax, zin_cpu.at({s,tin}));
        zout_cpu.at({tout}) = zmax;
    }

    // Run GPU kernel.

    Array<void> zin_gpu = zin_cpu.convert(dtype).to_gpu();
    Array<uint> ain_gpu = ain_cpu.to_gpu();
    Array<void> zout_gpu(dtype, {Tout}, af_gpu | af_guard);
    Array<uint> aout_gpu({Tout}, af_gpu | af_guard);

    // void kernel(void *zout, uint *aout, void *zin, uint *ain, uint Tin);
    auto kernel = TestPfOutput2::registry().get(key).cuda_kernel;

    kernel<<<1,32>>> (zout_gpu.data, aout_gpu.data, zin_gpu.data, ain_gpu.data, Tin);
    CUDA_PEEK("pf_output2_test_kernel");

    zout_gpu = zout_gpu.to_host();
    aout_gpu = aout_gpu.to_host();
    
    // The 'zout_gpu' array can be directly compared to the 'zout_cpu' array.
    // However, 'aout_gpu' cannot be directly compared to a CPU reference implementation,
    // because of (near-)ties. Therefore, we compute 'za_gpu', by evaluating the
    // 'zin_cpu' array at the array locations given by the 'aout_gpu' tokens. If 'za_gpu'
    // agrees with 'zout_cpu' (within roundoff error), then the 'aout_gpu' array is
    // correct.

    Array<float> za_gpu({Tout}, af_uhost);

    for (uint tout = 0; tout < Tout; tout++) {
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
