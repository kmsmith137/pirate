#include "../include/pirate/CoalescedDdKernel2.hpp"
#include "../include/pirate/MegaRingbuf.hpp"
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


CoalescedDdKernel2::CoalescedDdKernel2(const DedispersionKernelParams &dd_params_, const PeakFindingKernelParams &pf_params_) :
    dd_params(dd_params_), pf_params(pf_params_), fs(pf_params_.subband_counts)
{
    dd_params.validate();
    pf_params.validate();
    xassert(dd_params.dd_rank > 0);  // FIXME define _r0 for testing

    xassert(dd_params.nspec == 1);
    xassert(dd_params.apply_input_residual_lags);
    xassert(dd_params.input_is_ringbuf);
    xassert(!dd_params.output_is_ringbuf);
    xassert(dd_params.mega_ringbuf);
    xassert(dd_params.consumer_id >= 0);
    xassert(dd_params.consumer_id < dd_params.mega_ringbuf->num_consumers);

    long nsegments_per_beam = pow2(dd_params.dd_rank+dd_params.amb_rank) * xdiv(dd_params.ntime,dd_params.nt_per_segment);
    xassert_shape_eq(dd_params.mega_ringbuf->consumer_quadruples.at(dd_params.consumer_id), ({nsegments_per_beam,4}));

    xassert_eq(pf_params.dtype, dd_params.dtype);
    xassert_eq(pf_params.beams_per_batch, dd_params.beams_per_batch);
    xassert_eq(pf_params.total_beams, dd_params.total_beams);
    xassert_eq(pf_params.nt_in, dd_params.ntime);
    xassert_eq(pf_params.ndm_out, pow2(dd_params.dd_rank + dd_params.amb_rank - fs.pf_rank));

    // The initialization logic below is mostly cut-and-paste from either the
    // PeakFindingKernel or GpuDedispersionKernel constructor.

    this->dtype = dd_params.dtype;
    this->nbatches = xdiv(dd_params.total_beams, dd_params.beams_per_batch);
    this->Dout = xdiv(pf_params.nt_in, pf_params.nt_out);
    this->nprofiles = 3 * log2(pf_params.max_kernel_width) + 1;

    this->registry_key.dtype = pf_params.dtype;
    this->registry_key.dd_rank = dd_params.dd_rank;
    this->registry_key.Dout = xdiv(pf_params.nt_in, pf_params.nt_out);
    this->registry_key.W = pf_params.max_kernel_width;
    this->registry_key.subband_counts = fs.subband_counts;

    long SW = xdiv(32, pf_params.dtype.nbits);      // simd width
    long nt_in_per_wt = xdiv(pf_params.nt_in, pf_params.nt_wt);
    this->registry_key.Tinner = (nt_in_per_wt < 32*SW) ? xdiv(32*SW, nt_in_per_wt) : 1;

    // Call static member function CoalescedDdKernel2::registry().
    this->registry_value = registry().get(registry_key);

    // Derived parameters chosen by the kernel.
    this->pf_weight_layout = registry_value.pf_weight_layout;
    this->expected_wt_shape = pf_weight_layout.get_shape(pf_params.beams_per_batch, pf_params.ndm_wt, pf_params.nt_wt);
    this->expected_wt_strides = pf_weight_layout.get_strides(pf_params.beams_per_batch, pf_params.ndm_wt, pf_params.nt_wt);
    this->Dcore = registry_value.Dcore;
    
    // Important: ensure that caller-specified 'nt_per_segment' matches GPU kernel.
    xassert_eq(dd_params.nt_per_segment, registry_value.nt_per_segment);

    // FIXME add bandwidth tracking later.
}

void CoalescedDdKernel2::allocate()
{
    if (is_allocated)
        throw runtime_error("double call to CoalescedDdKernel2::allocate()");

    // Note 'af_zero' flag here.
    long ninner = registry_value.pstate32_per_small_tree * xdiv(32, dd_params.dtype.nbits);
    std::initializer_list<long> shape = { dd_params.total_beams, pow2(dd_params.amb_rank), ninner };
    this->persistent_state = Array<void> (dd_params.dtype, shape, af_zero | af_gpu);

    // Copy host -> GPU.
    this->gpu_ringbuf_quadruples = dd_params.mega_ringbuf->consumer_quadruples.at(dd_params.consumer_id).to_gpu();

    // Shape/stride check (paranoid).
    long nsegments_per_beam = pow2(dd_params.dd_rank+dd_params.amb_rank) * xdiv(dd_params.ntime,dd_params.nt_per_segment);
    xassert_shape_eq(gpu_ringbuf_quadruples, ({nsegments_per_beam,4}));
    xassert(gpu_ringbuf_quadruples.is_fully_contiguous());
    xassert(gpu_ringbuf_quadruples.on_gpu());

    this->is_allocated = true;
}


void CoalescedDdKernel2::launch(
    ksgpu::Array<void> &out_max,      // shape (beams_per_batch, ndm_out, nt_out)
    ksgpu::Array<uint> &out_argmax,   // shape (beams_per_batch, ndm_out, nt_out)
    const ksgpu::Array<void> &in,     // shape (mega_ringbuf->gpu_giant_nseg * nt_per_segment * nspec,)
    const ksgpu::Array<void> &wt,     // from GpuPfWeightLayout::to_gpu()
    long ibatch,                      // 0 <= ibatch < nbatches
    long it_chunk,                    // time-chunk index 0, 1, ...
    cudaStream_t stream)              // NULL stream is allowed, but is not the default);
{
    xassert(this->is_allocated);
    xassert((ibatch >= 0) && (ibatch < nbatches));
    xassert(it_chunk >= 0);

    xassert(out_max.dtype == dtype);
    xassert(in.dtype == dtype);
    xassert(wt.dtype == dtype);

    // Validate 'in' array: shape (mega_ringbuf->gpu_giant_nseg * nt_per_segment * nspec,)
    long giant_nseg = dd_params.mega_ringbuf->gpu_giant_nseg;
    xassert_shape_eq(in, ({ giant_nseg * dd_params.nt_per_segment * dd_params.nspec }));

    // Validate 'out' and 'out_argmax' arrays: shape (beams_per_batch, ndm_out, nt_out)
    xassert_shape_eq(out_max, ({ pf_params.beams_per_batch, pf_params.ndm_out, pf_params.nt_out }));
    xassert_shape_eq(out_argmax, ({ pf_params.beams_per_batch, pf_params.ndm_out, pf_params.nt_out }));

    // Validate 'wt' array. These checks will pass if 'wt' is the output of GpuPfWeightLayout::to_gpu().

    if (!wt.shape_equals(expected_wt_shape)) {
        stringstream ss;
        ss << "CoalescedDdKernel2::launch(): wt.shape=" << wt.shape_str()
           << ", expected_wt_shape=" << ksgpu::tuple_str(expected_wt_shape);
        throw runtime_error(ss.str());
    }

    if (!wt.strides_equal(expected_wt_strides)) {
        stringstream ss;
        ss << "CoalescedDdKernel2::launch(): wt.strides=" << wt.stride_str()
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

    // The global persistent_state array has shape { total_beams, pow2(params.amb_rank), ninner }.
    // We want to select a "slice"" of beams corresponding to the current batch.
    long b0 = (ibatch) * dd_params.beams_per_batch;
    long b1 = (ibatch+1) * dd_params.beams_per_batch;
    Array<void> pstate = this->persistent_state.slice(0, b0, b1);

    ulong nt_cumul = it_chunk * dd_params.ntime;
    long rb_pos = (it_chunk * dd_params.total_beams) + (ibatch * dd_params.beams_per_batch);
    long ndm_out_per_wt = xdiv(pf_params.ndm_out, pf_params.ndm_wt);
    long nt_in_per_wt = xdiv(pf_params.nt_in, pf_params.nt_wt);

    dim3 grid_dims = { uint(pow2(dd_params.amb_rank)), uint(dd_params.beams_per_batch), 1 };
    dim3 block_dims = { 32, uint(registry_value.warps_per_threadblock), 1 };

    registry_value.cuda_kernel<<< grid_dims, block_dims, registry_value.shmem_nbytes, stream >>>
        (in.data, gpu_ringbuf_quadruples.data, rb_pos,   // void *grb_base_, uint *grb_loc_, long grb_pos,
         out_max.data, out_argmax.data, wt.data,         // void *out_max_, uint *out_argmax, const void *wt_,
         pstate.data, dd_params.ntime,                   // void *pstate_, int ntime,
         nt_cumul, dd_params.input_is_downsampled_tree,  // ulong nt_cumul, bool input_is_downsampled_tree,
         ndm_out_per_wt, nt_in_per_wt);                  // uint ndm_out_per_wt, uint nt_in_per_wt

    CUDA_PEEK("coalesced_dd_kernel2 launch");
}


// Static member function: runs one randomized test iteration.
void CoalescedDdKernel2::test()
{
    RegistryKey key = registry().get_random_key();
    Dtype dtype = key.dtype;

    long simd_width = xdiv(32, key.dtype.nbits);
    long pf_rank = key.subband_counts.size() - 1;
    long dd_rank = key.dd_rank;
    long Tinner = key.Tinner;

    long nt_in_per_wt = (Tinner > 1) ? xdiv(32*simd_width,Tinner) : ((32 * simd_width) << rand_int(0,3));
    long nt_in_divisor = max(32*simd_width, nt_in_per_wt);

    auto v = ksgpu::random_integers_with_bounded_product(5, 200000 / pow2(dd_rank));
    long nchunks = v[0];
    long nt_in_per_chunk = nt_in_divisor * v[1];
    long beams_per_batch = v[2];
    long num_batches = v[3];
    long total_beams = beams_per_batch * num_batches;
    long amb_rank = max(8L, long(log2(v[4] + 0.5)));
    long lg_ndm_out = amb_rank + dd_rank - pf_rank;

    // Uncomment one or more lines below, to make the test instance smaller.
    nchunks = 1;
    nt_in_per_wt = (Tinner > 1) ? xdiv(32*simd_width,Tinner) : (32 * simd_width);
    nt_in_per_chunk = max(32*simd_width, nt_in_per_wt);
    beams_per_batch = 1;
    num_batches = 1;
    total_beams = beams_per_batch * num_batches;
    amb_rank = 0;
    lg_ndm_out = amb_rank + dd_rank - pf_rank;

    DedispersionKernelParams dd_params;
    dd_params.dtype = dtype;
    dd_params.dd_rank = key.dd_rank;
    dd_params.amb_rank = amb_rank;
    dd_params.beams_per_batch = beams_per_batch;
    dd_params.total_beams = total_beams;
    dd_params.ntime = nt_in_per_chunk;
    dd_params.nspec = 1;
    dd_params.input_is_ringbuf = true;
    dd_params.output_is_ringbuf = false;
    dd_params.apply_input_residual_lags = true;
    dd_params.input_is_downsampled_tree = rand_bool();
    dd_params.nt_per_segment = xdiv(1024, dtype.nbits);

    long nviews = pow2(key.dd_rank + amb_rank) * xdiv(nt_in_per_chunk, dd_params.nt_per_segment);
    dd_params.mega_ringbuf = MegaRingbuf::make_random_simplified(total_beams, beams_per_batch, nchunks, nviews);
    dd_params.consumer_id = 0;
    
    PeakFindingKernelParams pf_params;
    pf_params.subband_counts = key.subband_counts;
    pf_params.dtype = dtype;
    pf_params.max_kernel_width = key.W;
    pf_params.beams_per_batch = beams_per_batch;
    pf_params.total_beams = total_beams;
    pf_params.ndm_out = pow2(lg_ndm_out);
    pf_params.ndm_wt = pow2(rand_int(0, lg_ndm_out+1));
    pf_params.nt_out = xdiv(nt_in_per_chunk, key.Dout);
    pf_params.nt_in = nt_in_per_chunk;
    pf_params.nt_wt = xdiv(nt_in_per_chunk, nt_in_per_wt);

    CoalescedDdKernel2 cdd2_kernel(dd_params, pf_params);
    cdd2_kernel.allocate();

    ReferenceDedispersionKernel ref_dd_kernel(dd_params);
    ReferencePeakFindingKernel ref_pf_kernel(pf_params, cdd2_kernel.Dcore);

    FrequencySubbands &fs = cdd2_kernel.fs;
    GpuPfWeightLayout &wl = cdd2_kernel.pf_weight_layout;

    // Print this monstrosity.
    cout << "CoalescedDdKernel2::test()\n"
         << "    dtype = " << dtype.str() << "\n"
         << "    dd_rank = " << dd_params.dd_rank << "\n"
         << "    amb_rank = " << dd_params.amb_rank << "\n"
         << "    pf_rank = " << pf_rank << "\n"
         << "    subbands = " << ksgpu::tuple_str(key.subband_counts) << "\n"
         << "    W = " << key.W << "\n"
         << "    Dcore = " << cdd2_kernel.Dcore << "\n"
         << "    Dout = " << key.Dout << "\n"
         << "    Tinner = " << key.Tinner << "\n"
         << "    M = " << fs.M << "\n"
         << "    beams_per_batch = " << beams_per_batch << "\n"
         << "    total_beams = " << total_beams << "\n"
         << "    ndm_out = " << pf_params.ndm_out << "\n"
         << "    ndm_wt = "  << pf_params.ndm_wt << "\n"
         << "    nt_in_per_chunk = " << nt_in_per_chunk << "\n"
         << "    nt_out_per_chunk = " << pf_params.nt_out << "\n"
         << "    nt_wt_per_chunk = " << pf_params.nt_wt << "\n"
         << "    nchunks = " << nchunks << "\n" 
         << endl;

    // No subbands for now!
    xassert(fs.pf_rank == pf_rank);
    xassert(fs.F == 1);
    xassert(fs.M == pow2(pf_rank));
    xassert(fs.f_to_ilo[0] == 0);
    xassert(fs.f_to_ihi[0] == pow2(pf_rank));
    for (int m = 0; m < fs.M; m++) {
        xassert(fs.m_to_f[m] == 0);
        xassert(fs.m_to_d[m] == m);
    }

    long rb_nseg = dd_params.mega_ringbuf->gpu_giant_nseg;
    long rb_nelts = rb_nseg * dd_params.nt_per_segment;
    Array<float> in_cpu({rb_nelts}, af_rhost);

    // Fill input ring buffer with fixed random data.
    // Some data may be "replayed" across multiple time chunks, but that's okay.
    for (long i = 0; i < rb_nelts; i++)
        in_cpu.data[i] = rand_uniform(-1.0f, 1.0f);

    // Copy to GPU (converting dtype if necessary)
    Array<void> in_gpu = in_cpu.to_gpu(dtype);

    // Set up tmp/output buffers
    long B = dd_params.beams_per_batch;
    long A = pow2(dd_params.amb_rank);
    long T = nt_in_per_chunk;
    long D = pow2(dd_params.dd_rank);
    long M = fs.M;
    long Dout = pow2(lg_ndm_out);
    long Tout = pf_params.nt_out;

    // Output buffer for ref dedispersion kernel, shape (beams_per_batch, pow2(pf.amb_rank), pow2(pf.dd_rank), nt_in)
    Array<float> tmp_cpu({B,A,D,T}, af_uhost);

    // Input buffer for ref peak-finding kernel, shape beams_per_batch, pf.ndm_out, M, nt_in)
    Array<float> tmp2_cpu({B,Dout,M,T}, af_uhost);
    xassert(Dout*M == A*D);

    Array<float> max_cpu({B,Dout,Tout}, af_uhost | af_zero);
    Array<uint> argmax_cpu({B,Dout,Tout}, af_uhost | af_zero);

    Array<void> max_gpu(dtype, {B,Dout,Tout}, af_gpu | af_zero);
    Array<uint> argmax_gpu({B,Dout,Tout}, af_gpu | af_zero);

    // Tmp buffer for comparing "argmax" arrays, see below.
    Array<float> max_x({B,Dout,Tout}, af_uhost | af_zero);

    for (long ichunk = 0; ichunk < nchunks; ichunk++) {
        for (long ibatch = 0; ibatch < num_batches; ibatch++) {
            ref_dd_kernel.apply(in_cpu, tmp_cpu, ibatch, ichunk);

            //  -------- FIXME ad hoc shuffling operation tmp_cpu -> tmp2_cpu starts here ------

            for (long dm = 0; dm < A*D; dm++) {
                // Convert "flattened" DM index -> (indices (a,d) in tmp array)
                // The index 0 <= a < A represents a bit-reversed coarse DM.
                // The index 0 <= d < D represents a bit-reversed fine DM.

                long dm_brev = bit_reverse_slow(dm, dd_params.amb_rank + dd_params.dd_rank);
                long a = dm_brev & (A-1);
                long d = dm_brev >> dd_params.amb_rank;

                float *src = &tmp_cpu.at({0,a,d,0});
                long sstride = tmp_cpu.strides[0];

                // Convert "flattened" DM index -> (indices dout,m) in tmp2 array)

                long dout = dm >> pf_rank;
                long m = dm & (M-1);

                float *dst = &tmp2_cpu.at({0,dout,m,0});
                long dstride = tmp2_cpu.strides[0];

                // Copy tmp -> tmp2
                for (long b = 0; b < B; b++)
                    for (long t = 0; t < T; t++)
                        dst[b*dstride + t] = src[b*sstride + t];
            }

            //  -------- FIXME ad hoc shuffling operation tmp_cpu -> tmp2_cpu ends here ------

            // FIXME revisit ReferencePeakFindingKernel::make_random_weights() with subbands.
            Array<float> wt_cpu = ref_pf_kernel.make_random_weights();
            ref_pf_kernel.apply(max_cpu, argmax_cpu, tmp2_cpu, wt_cpu, ibatch);

            // CPU kernel done! Now run the GPU kernel.
            Array<void> wt_gpu = wl.to_gpu(wt_cpu);
            cdd2_kernel.launch(max_gpu, argmax_gpu, in_gpu, wt_gpu, ibatch, ichunk, NULL);

            // The "max" arrays can be compared straightforwardly.
            assert_arrays_equal(max_cpu, max_gpu, "max_cpu", "max_gpu", {"b","d","tout"});

            // For the "argmax" arrays, we have to do something weird.
            // On the CPU, evaluate triggers at the "argmax_gpu" values.
            Array<uint> argmax_x = argmax_gpu.to_host();
            ref_pf_kernel.eval_tokens(max_x, argmax_x, wt_cpu);

            // Then compare to "max_cpu", possibly at reduced precision.
            double eps = 10 * dtype.precision();
            assert_arrays_equal(max_cpu, max_x, "max_cpu", "max_x", {"b","d","tout"}, eps);
        }
    }
}


void CoalescedDdKernel2::time()
{
    cout << "CoalescedDdKernel2::time() placeholder" << endl;
}


// -------------------------------------------------------------------------------------------------
//
// Registry and related functions


CoalescedDdKernel2::Registry &CoalescedDdKernel2::registry()
{
    // This kludge implements "construct on first use". It's necessary because the
    // registry is accessed at library initialization time (when kernel .cu files
    // call CoalescedDdKernel2::registry().add() to register themselves).
    //
    // Using a static variable in this way (instead of a global variable) ensures
    // that the registry is constructed before CoalescedDdKernel2::registry().add()
    // is called.
    //
    // This kludge is necessary because the registry is accessed at library initialization
    // time, by callers in other source files, and source files are executed in an
    // arbitrary order.
    
    static CoalescedDdKernel2::Registry reg;
    return reg;  // note: thread-safe (as of c++11)
}

bool operator==(const CoalescedDdKernel2::RegistryKey &k1, const CoalescedDdKernel2::RegistryKey &k2)
{
    return (k1.dtype == k2.dtype)
        && (k1.dd_rank == k2.dd_rank)
        && (k1.subband_counts == k2.subband_counts)
        && (k1.Tinner == k2.Tinner)
        && (k1.Dout == k2.Dout)
        && (k1.W == k2.W);
}

ostream &operator<<(ostream &os, const CoalescedDdKernel2::RegistryKey &k)
{
    FrequencySubbands fs(k.subband_counts);
    os << "CoalescedDdKernel2(dtype=" << k.dtype.str()
       << ", dd_rank=" << k.dd_rank
       << ", subbands=" << tuple_str(k.subband_counts)
       << ", Tinner=" << k.Tinner
       << ", Dout=" << k.Dout
       << ", W=" << k.W
       << ", F=" << fs.F
       << ", M=" << fs.M
       << ")";
    return os;
}

ostream &operator<<(ostream &os, const CoalescedDdKernel2::RegistryValue &v)
{
    os << "(Dcore=" << v.Dcore
       << ", shmem=" << v.shmem_nbytes
       << ", warps=" << v.warps_per_threadblock
       << ", pstate32=" << v.pstate32_per_small_tree
       << ", nt_seg=" << v.nt_per_segment
       << ")";
    return os;
}


}  // namespace pirate
