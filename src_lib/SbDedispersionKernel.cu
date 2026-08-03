#include "../include/pirate/SbDedispersionKernel.hpp"
#include "../include/pirate/MegaRingbuf.hpp"
#include "../include/pirate/ReferenceTree.hpp"
#include "../include/pirate/inlines.hpp"
#include "../include/pirate/utils.hpp"
#include "../include/pirate/constants.hpp"     // cuda_max_static_shmem_bytes

#include <mutex>
#include <sstream>

#include <ksgpu/Array.hpp>
#include <ksgpu/cuda_utils.hpp>
#include <ksgpu/rand_utils.hpp>     // rand_int(), random_integers_with_bounded_product()
#include <ksgpu/string_utils.hpp>
#include <ksgpu/test_utils.hpp>     // assert_arrays_equal()

using namespace std;
using namespace ksgpu;

namespace pirate {
#if 0
}  // editor auto-indent
#endif


GpuSbDedispersionKernel::GpuSbDedispersionKernel(
    const DedispersionKernelParams &dd_params_, const FrequencySubbands &frequency_subbands) :
    dd_params(dd_params_), fs(frequency_subbands)
{
    dd_params.validate();

    // Same constraints as CoalescedDdKernel2, plus float32-only. See SbDedispersionKernel.hpp.
    xassert_eq(dd_params.dtype, Dtype::native<float> ());
    xassert_ge(dd_params.dd_rank, 3);   // kernel needs the two-stage dedisperser
    xassert_eq(dd_params.nspec, 1);
    xassert(dd_params.apply_input_residual_lags);
    xassert(dd_params.input_is_ringbuf);
    xassert(!dd_params.output_is_ringbuf);
    xassert(dd_params.mega_ringbuf);
    xassert(dd_params.consumer_id >= 0);
    xassert(dd_params.consumer_id < dd_params.mega_ringbuf->num_consumers);

    // The kernel gets the per-subband time lag "for free" from the shared memory ring buffer
    // of the two-stage dedisperser, which is only correct if the peak-finding rank equals the
    // dedisperser's second-stage rank. (Without this check, the registry lookup below would
    // just fail to find a kernel, with a less informative error.) See SbDedispersionKernel.hpp.
    xassert_eq(fs.pf_rank, dd_params.dd_rank - (dd_params.dd_rank / 2));

    this->nsegments_per_beam = pow2(dd_params.dd_rank + dd_params.amb_rank) * xdiv(dd_params.ntime, dd_params.nt_per_segment);
    xassert_shape_eq(dd_params.mega_ringbuf->consumer_quadruples.at(dd_params.consumer_id), ({nsegments_per_beam,4}));

    this->Dpf = pow2(dd_params.amb_rank + dd_params.dd_rank - fs.pf_rank);
    this->nbatches = xdiv(dd_params.total_beams, dd_params.beams_per_batch);

    this->registry_key.dd_rank = dd_params.dd_rank;
    this->registry_key.subband_counts = fs.subband_counts;
    this->registry_value = registry().get(registry_key);

    // Important: ensure that caller-specified 'nt_per_segment' matches GPU kernel.
    xassert_eq(dd_params.nt_per_segment, registry_value.nt_per_segment);

    // The kernel indexes the m-axis of 'sb_out' with 32-bit arithmetic (see 'm_stride' in
    // the generated code), which is only safe if the m-stride times M fits in an int.
    xassert_lt(fs.M * dd_params.ntime, 1L << 31);

    long B = dd_params.beams_per_batch;
    long A = pow2(dd_params.amb_rank);
    long D = pow2(dd_params.dd_rank);
    long S = 4;   // sizeof(float)
    long bw_in = B * A * D * dd_params.ntime * S;
    long bw_out = B * Dpf * fs.M * dd_params.ntime * S;
    long quads_nbytes = nsegments_per_beam * 4 * 4;
    long pstate_nbytes_per_beam = A * registry_value.pstate32_per_small_tree * 4;

    // Note bw_out/bw_in = M / 2^pf_rank, which is ~5.7 for the CHORD subband counts.
    resource_tracker.add_kernel("sbdd", bw_in + bw_out);
    resource_tracker.add_gmem_bw("sbdd_quads", B * quads_nbytes);
    resource_tracker.add_gmem_bw("sbdd_pstate", 2 * B * pstate_nbytes_per_beam);

    resource_tracker.add_gmem_footprint("persistent_state", dd_params.total_beams * pstate_nbytes_per_beam, true);
    resource_tracker.add_gmem_footprint("quadruples", quads_nbytes, true);
}


void GpuSbDedispersionKernel::allocate(BumpAllocator &allocator)
{
    if (is_allocated)
        throw runtime_error("double call to GpuSbDedispersionKernel::allocate()");

    if (!(allocator.aflags & af_gpu))
        throw runtime_error("GpuSbDedispersionKernel::allocate(): allocator.aflags must contain af_gpu");
    if (!(allocator.aflags & af_zero))
        throw runtime_error("GpuSbDedispersionKernel::allocate(): allocator.aflags must contain af_zero");

    long nbytes_before = allocator.get_nbytes_allocated();

    // Allocate persistent_state.
    std::initializer_list<long> shape = {
        dd_params.total_beams,
        pow2(dd_params.amb_rank),
        registry_value.pstate32_per_small_tree
    };
    this->persistent_state = allocator.allocate_array<float> (shape);

    // Copy host -> GPU.
    const Array<uint> &src = dd_params.mega_ringbuf->consumer_quadruples.at(dd_params.consumer_id);
    this->gpu_ringbuf_quadruples = allocator.allocate_array<uint> ({nsegments_per_beam, 4});
    this->gpu_ringbuf_quadruples.fill(src);

    // Shape/stride check (paranoid).
    xassert_shape_eq(gpu_ringbuf_quadruples, ({nsegments_per_beam,4}));
    xassert(gpu_ringbuf_quadruples.is_fully_contiguous());
    xassert(gpu_ringbuf_quadruples.on_gpu());

    long nbytes_allocated = allocator.get_nbytes_allocated() - nbytes_before;
    xassert_eq(nbytes_allocated, resource_tracker.get_gmem_footprint());

    this->is_allocated = true;
}


void GpuSbDedispersionKernel::launch(
    ksgpu::Array<float> &sb_out,      // shape (beams_per_batch, Dpf, fs.M, ntime)
    const ksgpu::Array<float> &in,    // shape (mega_ringbuf->gpu_global_nseg * nt_per_segment,)
    long ichunk,                      // time-chunk index 0, 1, ...
    long ibatch,                      // 0 <= ibatch < nbatches
    cudaStream_t stream)              // NULL stream is allowed, but is not the default
{
    xassert(this->is_allocated);
    xassert((ibatch >= 0) && (ibatch < nbatches));
    xassert(ichunk >= 0);

    // Validate 'in' array (reminder: nspec==1, so nt_per_segment == nelts_per_segment).
    long global_nseg = dd_params.mega_ringbuf->gpu_global_nseg;
    xassert_shape_eq(in, ({ global_nseg * dd_params.nt_per_segment }));

    // Validate 'sb_out' array. The kernel derives all of its strides from (fs.M, ntime),
    // so full contiguity is required (not just contiguity of the last axis).
    xassert_shape_eq(sb_out, ({ dd_params.beams_per_batch, Dpf, fs.M, dd_params.ntime }));

    xassert(in.is_fully_contiguous());
    xassert(sb_out.is_fully_contiguous());
    xassert(in.on_gpu());
    xassert(sb_out.on_gpu());

    // The global persistent_state array has shape { total_beams, pow2(amb_rank), pstate32 }.
    // We want to select a "slice" of beams corresponding to the current batch.
    long b0 = (ibatch) * dd_params.beams_per_batch;
    long b1 = (ibatch+1) * dd_params.beams_per_batch;
    Array<float> pstate = this->persistent_state.slice(0, b0, b1);

    ulong nt_cumul = ichunk * dd_params.ntime;
    long rb_frame0 = (ichunk * dd_params.total_beams) + (ibatch * dd_params.beams_per_batch);

    dim3 grid_dims = { uint(pow2(dd_params.amb_rank)), uint(dd_params.beams_per_batch), 1 };
    dim3 block_dims = { 32, uint(registry_value.warps_per_threadblock), 1 };

    registry_value.cuda_kernel<<< grid_dims, block_dims, registry_value.shmem_nbytes, stream >>>
        (in.data, gpu_ringbuf_quadruples.data, rb_frame0,  // void *grb_base_, uint *grb_quads_, long grb_frame0,
         sb_out.data,                                      // float *sb_out_,
         pstate.data, dd_params.ntime,                     // void *pstate_, int ntime,
         nt_cumul, dd_params.input_is_downsampled_tree);   // ulong nt_cumul, bool is_downsampled_tree

    CUDA_PEEK("sb_dedispersion kernel");
}


// -------------------------------------------------------------------------------------------------
//
// GpuSbDedispersionKernel::test_random()


// Static member function: runs one randomized test iteration.
void GpuSbDedispersionKernel::test_random()
{
    RegistryKey key = registry().get_random_key();
    FrequencySubbands fs(key.subband_counts);

    long dd_rank = key.dd_rank;
    long pf_rank = fs.pf_rank;
    Dtype dtype = Dtype::native<float> ();

    // Bound the size of the test instance. The output array is (B, Dpf, M, nt), which is
    // (M / 2^pf_rank) times larger than the dedispersion buffer, so we shrink the budget
    // by that factor. Note that the 'm_factor' and 2^(dd_rank-pf_rank) factors then cancel:
    // the output array is bounded by ~(32 * 30000) elements, independently of which kernel
    // was drawn from the registry.

    long m_factor = std::max(1L, fs.M / pow2(pf_rank));
    auto v = ksgpu::random_integers_with_bounded_product(5, 30000 / (pow2(dd_rank) * m_factor));

    long nchunks = v[0];
    long nt_in_per_chunk = 32 * v[1];   // multiple of nt_per_segment
    long beams_per_batch = v[2];
    long num_batches = v[3];
    long total_beams = beams_per_batch * num_batches;
    long amb_rank = min(8L, long(log2(v[4] + 0.5)));
    bool is_downsampled_tree = rand_bool();

    // Uncomment one or more lines below, to make the test instance smaller.
    // nchunks = 1;
    // nt_in_per_chunk = 32;
    // is_downsampled_tree = false;
    // beams_per_batch = 1;
    // num_batches = 1;
    // amb_rank = 0;
    //
    // *** YOU MUST ALSO UNCOMMENT THE NEXT LINE ***
    // total_beams = beams_per_batch * num_batches;

    DedispersionKernelParams dd_params;
    dd_params.dtype = dtype;
    dd_params.dd_rank = dd_rank;
    dd_params.amb_rank = amb_rank;
    dd_params.beams_per_batch = beams_per_batch;
    dd_params.total_beams = total_beams;
    dd_params.ntime = nt_in_per_chunk;
    dd_params.nspec = 1;
    dd_params.input_is_ringbuf = true;
    dd_params.output_is_ringbuf = false;
    dd_params.apply_input_residual_lags = true;
    dd_params.input_is_downsampled_tree = is_downsampled_tree;
    dd_params.nt_per_segment = xdiv(1024, dtype.nbits);

    long nquads = pow2(dd_rank + amb_rank) * xdiv(nt_in_per_chunk, dd_params.nt_per_segment);
    dd_params.mega_ringbuf = MegaRingbuf::make_random_simplified(total_beams, beams_per_batch, nchunks, nquads);
    dd_params.consumer_id = 0;

    GpuSbDedispersionKernel kernel(dd_params, fs);
    BumpAllocator allocator(af_gpu | af_zero, -1);  // dummy allocator
    kernel.allocate(allocator);

    ReferenceDedispersionKernel ref_kernel(dd_params, key.subband_counts);
    xassert_eq(kernel.Dpf, ref_kernel.Dpf);

    cout << "GpuSbDedispersionKernel::test()\n"
         << "    dd_rank = " << dd_rank << "\n"
         << "    amb_rank = " << amb_rank << "\n"
         << "    pf_rank = " << pf_rank << "\n"
         << "    is_downsampled_tree = " << is_downsampled_tree << "\n"
         << "    subbands = " << ksgpu::tuple_str(key.subband_counts) << "\n"
         << "    M = " << fs.M << "\n"
         << "    N = " << fs.N << "\n"
         << "    beams_per_batch = " << beams_per_batch << "\n"
         << "    total_beams = " << total_beams << "\n"
         << "    ndm_out = " << kernel.Dpf << "\n"
         << "    nt_in_per_chunk = " << nt_in_per_chunk << "\n"
         << "    nchunks = " << nchunks << "\n"
         << endl;

    long rb_nseg = dd_params.mega_ringbuf->gpu_global_nseg;
    long rb_nelts = rb_nseg * dd_params.nt_per_segment;

    // Fill input ring buffer with fixed random data. Some data may be "replayed" across
    // multiple time chunks, but that's okay.
    Array<float> in_cpu({rb_nelts}, af_rhost | af_random);
    Array<float> in_gpu = in_cpu.to_gpu();

    long B = beams_per_batch;
    long A = pow2(amb_rank);
    long D = pow2(dd_rank);
    long T = nt_in_per_chunk;
    long M = fs.M;
    long Dpf = kernel.Dpf;

    Array<float> dd_cpu({B,A,D,T}, af_uhost);            // 'dd_out' scratch for ref_kernel
    Array<float> sb_cpu({B,Dpf,M,T}, af_uhost);          // 'sb_out' from ref_kernel
    Array<float> sb_gpu({B,Dpf,M,T}, af_gpu | af_zero);  // 'sb_out' from GPU kernel

    // Both kernels are stateful across chunks (the GPU kernel via 'persistent_state', the
    // reference via ReferenceTree::pstate + final_lagbuf + rlag_bufs), so the chunks must
    // be processed in order, with neither side reset. This is what exercises that state.

    for (long ichunk = 0; ichunk < nchunks; ichunk++) {
        for (long ibatch = 0; ibatch < num_batches; ibatch++) {
            ref_kernel.apply(in_cpu, dd_cpu, sb_cpu, ichunk, ibatch);
            kernel.launch(sb_gpu, in_gpu, ichunk, ibatch, NULL);

            stringstream ss;
            ss << "sb_cpu(chunk=" << ichunk << ",batch=" << ibatch << ")";
            assert_arrays_equal(sb_cpu, sb_gpu, ss.str(), "sb_gpu", {"b","dpf","m","t"});
        }
    }
}


// -------------------------------------------------------------------------------------------------
//
// Registry and related functions


struct SbddRegistry : public GpuSbDedispersionKernel::Registry
{
    using Key = GpuSbDedispersionKernel::RegistryKey;
    using Val = GpuSbDedispersionKernel::RegistryValue;

    virtual void add(const Key &key, const Val &val, bool debug) override
    {
        // Just check that all members have been initialized.
        // (In the future, I may add more argument checking here.)

        xassert_ge(key.dd_rank, 3);
        xassert_ge(key.subband_counts.size(), 3);   // pf_rank >= 2, since dd_rank >= 3

        xassert(val.cuda_kernel != nullptr);
        xassert(val.shmem_nbytes >= 0);
        xassert(val.warps_per_threadblock > 0);
        xassert(val.pstate32_per_small_tree >= 0);
        xassert(val.nt_per_segment > 0);

        // Call add() in base class.
        GpuSbDedispersionKernel::Registry::add(key, val, debug);
    }


    // Setting shared memory size is "deferred" from when the kernel is registered, to when
    // the kernel is first used. Deferring is important, since cudaFuncSetAttribute() creates
    // hard-to-debug problems if called at library initialization time, but behaves normally
    // if deferred. (Here, "hard-to-debug" means that the call appears to succeed, but an
    // unrelated kernel launch will fail later with error 400 ("invalid resource handle").)

    virtual void deferred_initialization(Val &val) override
    {
        if (val.shmem_nbytes > constants::cuda_max_static_shmem_bytes) {
            CUDA_CALL(cudaFuncSetAttribute(
                val.cuda_kernel,
                cudaFuncAttributeMaxDynamicSharedMemorySize,
                val.shmem_nbytes
            ));
        }
    }
};


GpuSbDedispersionKernel::Registry &GpuSbDedispersionKernel::registry()
{
    // This kludge implements "construct on first use". It's necessary because the
    // registry is accessed at library initialization time (when kernel .cu files
    // call GpuSbDedispersionKernel::registry().add() to register themselves), by
    // callers in other source files, and source files are executed in an arbitrary
    // order. Using a static variable in this way (instead of a global variable)
    // ensures that the registry is constructed before add() is called.

    static SbddRegistry reg;
    return reg;  // note: thread-safe (as of c++11)
}


bool operator==(const GpuSbDedispersionKernel::RegistryKey &k1, const GpuSbDedispersionKernel::RegistryKey &k2)
{
    return (k1.dd_rank == k2.dd_rank)
        && (k1.subband_counts == k2.subband_counts);
}

ostream &operator<<(ostream &os, const GpuSbDedispersionKernel::RegistryKey &k)
{
    FrequencySubbands fs(k.subband_counts);
    os << "GpuSbDedispersionKernel(dd_rank=" << k.dd_rank
       << ", subbands=" << tuple_str(k.subband_counts)
       << ", N=" << fs.N
       << ", M=" << fs.M
       << ")";
    return os;
}

ostream &operator<<(ostream &os, const GpuSbDedispersionKernel::RegistryValue &v)
{
    os << "(shmem=" << v.shmem_nbytes
       << ", warps=" << v.warps_per_threadblock
       << ", pstate32=" << v.pstate32_per_small_tree
       << ", nt_seg=" << v.nt_per_segment
       << ")";
    return os;
}


}  // namespace pirate
