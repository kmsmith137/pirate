#include "../include/pirate/DedispersionKernel.hpp"
#include "../include/pirate/ReferenceLagbuf.hpp"
#include "../include/pirate/ReferenceTree.hpp"
#include "../include/pirate/constants.hpp"
#include "../include/pirate/inlines.hpp"   // pow2(), is_aligned(), simd_type
#include "../include/pirate/utils.hpp"     // bit_reverse_slow(

#include <mutex>
#include <sstream>
#include <iostream>

#include <ksgpu/cuda_utils.hpp>  // CUDA_CALL()
#include <ksgpu/rand_utils.hpp>  // rand_int()
#include <ksgpu/test_utils.hpp>  // make_random_strides(), assert_arrays_equal()
#include <ksgpu/string_utils.hpp>
#include <ksgpu/KernelTimer.hpp>

using namespace std;
using namespace ksgpu;

namespace pirate {
#if 0
}  // editor auto-indent
#endif


// -------------------------------------------------------------------------------------------------
//
// DedispersionKernelParams


void DedispersionKernelParams::validate() const
{
    xassert_ge(dd_rank, 0);
    xassert_le(dd_rank, 8);
    xassert_ge(amb_rank, 0);
    xassert_le(amb_rank, 8);
    xassert_gt(total_beams, 0);
    xassert_gt(beams_per_batch, 0);
    xassert_le(beams_per_batch, constants::cuda_max_y_blocks);
    xassert_ge(nt_per_segment, 0);
    xassert_ge(nspec, 0);
    xassert_ge(ntime, 0);

    xassert((dtype == Dtype::native<float>()) || (dtype == Dtype::native<__half>()));
    xassert_divisible(ntime, nt_per_segment);

    // (ringbuf -> ringbuf) doesn't make sense.
    xassert(!input_is_ringbuf || !output_is_ringbuf);
    
    // Currently assumed throughout the pirate code.
    xassert_divisible(total_beams, beams_per_batch);
    
    if (input_is_ringbuf || output_is_ringbuf) {
        long nsegments_per_tree = pow2(dd_rank+amb_rank) * xdiv(ntime,nt_per_segment);
        xassert_shape_eq(ringbuf_locations, ({ nsegments_per_tree, 4 }));
        xassert(ringbuf_locations.is_fully_contiguous());
        xassert(ringbuf_nseg > 0);
        xassert(ringbuf_nseg <= UINT_MAX);
        xassert(ringbuf_locations.on_host());
        
        for (long iseg = 0; iseg < nsegments_per_tree; iseg++) {
            const uint *rb_locs = ringbuf_locations.data + (4*iseg);
            long rb_offset = rb_locs[0];  // in segments, not bytes
            // long rb_phase = rb_locs[1];   // index of (time chunk, beam) pair, relative to current pair
            long rb_len = rb_locs[2];     // number of (time chunk, beam) pairs in ringbuf (same as Ringbuf::rb_len)
            long rb_nseg = rb_locs[3];    // number of segments per (time chunk, beam) (same as Ringbuf::nseg_per_beam)
            xassert(rb_offset + (rb_len-1)*rb_nseg < ringbuf_nseg);
        }
    }
}


void DedispersionKernelParams::print(const char *prefix) const
{
    if (!prefix)
        prefix = "";
    
    cout << prefix << "dtype = " << this->dtype << ";\n"
         << prefix << "dd_rank = " << this->dd_rank << ";\n"
         << prefix << "amb_rank = " << this->amb_rank << ";\n"
         << prefix << "total_beams = " << this->total_beams << ";\n"
         << prefix << "beams_per_batch = " << this->beams_per_batch << ";\n"
         << prefix << "ntime = " << this->ntime << ";\n"
         << prefix << "nspec = " << this->nspec << ";\n"
         << prefix << "input_is_ringbuf = " << (this->input_is_ringbuf ? "true" : "false")  << ";\n"
         << prefix << "output_is_ringbuf = " << (this->output_is_ringbuf ? "true" : "false")  << ";\n"
         << prefix << "apply_input_residual_lags = " << (this->apply_input_residual_lags ? "true" : "false")  << ";\n"
         << prefix << "input_is_downsampled_tree = " << (this->input_is_downsampled_tree ? "true" : "false")  << ";\n"
         << prefix << "nt_per_segment = " << this->nt_per_segment << ";\n";
}


// -------------------------------------------------------------------------------------------------
//
// DedispersionKernelIobuf


DedispersionKernelIobuf::DedispersionKernelIobuf(const DedispersionKernelParams &params, const Array<void> &arr, bool is_ringbuf_, bool on_gpu_)
{
    this->buf = arr.data;
    this->is_ringbuf = is_ringbuf_;
    this->on_gpu = on_gpu_;

    // Note: we don't call params.validate(), since it can be a "heavyweight" operation
    // (since ringbuf_locations are checked), and the DedispersionKernelIobuf constructor
    // needs to be "lightweight" (since called on every kernel launch).
    
    xassert_eq(arr.dtype, params.dtype);
    
    // Check alignment. Not strictly necessary, but failure would be unintentional and indicate a bug somewhere.
    xassert(is_aligned(buf, constants::bytes_per_gpu_cache_line));   // also checks non_NULL
    
    if (on_gpu)
        xassert(arr.on_gpu());
    else
        xassert(arr.on_host());

    // FIXME constructor should include overflow checks on strides.
    // (Check on act_stride is nontrivial, since it gets multiplied by a small integer in the kernel.)
    
    if (is_ringbuf) {
        // Case 1: ringbuf, 1-d array of length (ringbuf_nseg * nt_per_segment * nspec).
        xassert_shape_eq(arr, ({ params.ringbuf_nseg * params.nt_per_segment * params.nspec }));
        xassert(arr.get_ncontig() == 1);  // fully contiguous
        return;
    }
    
    // Case 2: simple buf. Shape is either:
    //     (beams_per_batch, pow2(amb_rank), pow2(dd_rank), ntime, nspec)
    //  or (beams_per_batch, pow2(amb_rank), pow2(dd_rank), ntime)  if nspec==1

    long B = params.beams_per_batch;
    long A = pow2(params.amb_rank);
    long N = pow2(params.dd_rank);
    long T = params.ntime;
    long S = params.nspec;

    std::initializer_list<long> shape1 = {B,A,N,T,S};
    std::initializer_list<long> shape2 = {B,A,N,T};  // also allowed if S==1
    
    bool valid1 = arr.shape_equals(shape1);
    bool valid2 = (S==1) && arr.shape_equals(shape2);

    if (!valid1 && !valid2) {
        stringstream ss;
        ss << "DedispersionKernelIobuf: got shape " << arr.shape_str() << ", expected shape " << ksgpu::tuple_str(shape1);
        if (S == 1)
            ss << " or " << ksgpu::tuple_str(shape2);
        throw runtime_error(ss.str());
    }
 
    // Valid for both shapes.
    xassert(arr.get_ncontig() >= arr.ndim-3);

    long denom = xdiv(32, arr.dtype.nbits);
    this->beam_stride32 = xdiv(arr.strides[0], denom);   // 32-bit stride
    this->amb_stride32 = xdiv(arr.strides[1], denom);    // 32-bit stride
    this->act_stride32 = xdiv(arr.strides[2], denom);    // 32-bit stride
    
    // FIXME could improve these checks, by verifying that strides are non-overlapping.
    xassert((params.beams_per_batch == 1) || (beam_stride32 != 0));
    xassert((params.amb_rank == 0) || (amb_stride32 != 0));
    xassert((params.dd_rank == 0) || (act_stride32 != 0));

    if (on_gpu) {
        // Check alignment. Not strictly necessary, but failure would be unintentional and indicate a bug somewhere.
        xassert_divisible(beam_stride32 * 4, constants::bytes_per_gpu_cache_line);
        xassert_divisible(amb_stride32 * 4, constants::bytes_per_gpu_cache_line);
        xassert_divisible(act_stride32 * 4, constants::bytes_per_gpu_cache_line);
    }
}


// -------------------------------------------------------------------------------------------------
//
// ReferenceDedispersionKernel
//
// GPU dedispersion kernels assumes (nelts_per_segment == nelts_per_cache_line), but
// reference kernels allow (nelts_per_segment) to be a multiple of (nelts_per_cache_line),
// where:
//
//   nelts_per_cache_line = (8 * constants::bytes_per_gpu_cache_line) / dtype.nbits.
//
// This is in order to enable a unit test where we check agreement between a float16
// GPU kernel, and a float32 reference kernel derived from the same DedispersionPlan.
// In this case, we want the reference kernel to have dtype float32, but use a value
// of 'nelts_per_segment' which matched to the float16 GPU kernel.
//
// Enabling this feature is straightforward: we just compute residual lags and ring
// buffer offsets using the value of (params.nelts_per_segment), rather than assuming
// nelts_per_segment == 32.


ReferenceDedispersionKernel::ReferenceDedispersionKernel(const Params &params_) :
    params(params_)
{
    params.dtype = Dtype::native<float>();
    params.validate();

    this->nbatches = xdiv(params.total_beams, params.beams_per_batch);
    
    long B = params.beams_per_batch;
    long A = pow2(params.amb_rank);
    long F = pow2(params.dd_rank);
    long T = params.ntime;
    long S = params.nspec;
    
    this->trees.resize(nbatches);                         
    for (long n = 0; n < nbatches; n++)
        trees[n] = ReferenceTree::make({B,A,F,T*S}, S);   // (shape, nspec)

    if (params.apply_input_residual_lags)
        this->_init_rlags();
}


void ReferenceDedispersionKernel::apply(Array<void> &in_, Array<void> &out_, long ibatch, long it_chunk)
{
    long B = params.beams_per_batch;
    long A = pow2(params.amb_rank);
    long N = pow2(params.dd_rank);
    long T = params.ntime;
    long S = params.nspec;
    
    // Error checking, shape checking.
    DedispersionKernelIobuf inbuf(params, in_, params.input_is_ringbuf, false);     // on_gpu=false
    DedispersionKernelIobuf outbuf(params, out_, params.output_is_ringbuf, false);  // on_gpu=false
    
    xassert((ibatch >= 0) && (ibatch < nbatches));
    xassert(it_chunk >= 0);

    // The reference kernel uses float32, regardless of what dtype is specified.
    Array<float> in = in_.template cast<float> ("ReferenceDedispersionKernel::apply(): 'in' array");
    Array<float> out = out_.template cast<float> ("ReferenceDedispersionKernel::apply(): 'out' array");

    // Reshape "simple" bufs to 4-d.
    in = params.input_is_ringbuf ? in : in.reshape({B,A,N,T*S});
    out = params.output_is_ringbuf ? out : out.reshape({B,A,N,T*S});

    long rb_pos = it_chunk * params.total_beams + (ibatch * params.beams_per_batch);
    Array<float> dd = params.output_is_ringbuf ? in : out;

    if (params.input_is_ringbuf)
        _copy_from_ringbuf(in, dd, rb_pos);
    else if (dd.data != in.data)
        dd.fill(in);
        
    if (params.apply_input_residual_lags)
        rlag_bufs.at(ibatch)->apply_lags(dd);

    trees.at(ibatch)->dedisperse(dd);

    if (params.output_is_ringbuf)
        _copy_to_ringbuf(dd, out, rb_pos);
    else
        xassert(out.data == dd.data);   // FIXME in-place assumed
}


// Helper function called by constructor.
void ReferenceDedispersionKernel::_init_rlags()
{
    long B = params.beams_per_batch;
    long A = pow2(params.amb_rank);
    long F = pow2(params.dd_rank);
    long N = params.nt_per_segment;
    long T = params.ntime;
    long S = params.nspec;
    
    Array<int> rlags({B,A,F}, af_uhost);
    
    for (long b = 0; b < B; b++) {
        for (long a = 0; a < A; a++) {
            // Ambient index 'a' represents a bit-reversed coarse DM.
            // Index 'f' represents a fine frequency.
            for (long f = 0; f < F; f++) {
                long lag = rb_lag(f, a, params.amb_rank, params.dd_rank, params.input_is_downsampled_tree);
                rlags.data[b*A*F + a*F + f] = (lag % N) * S;  // residual lag (including factor nspec)
            }
        }
    }
    
    this->rlag_bufs.resize(nbatches);
    for (long n = 0; n < nbatches; n++)
        rlag_bufs[n] = make_shared<ReferenceLagbuf> (rlags, T*S);
}

// Helper function called by apply().
void ReferenceDedispersionKernel::_copy_to_ringbuf(const Array<float> &in, Array<float> &out, long rb_pos)
{
    long B = params.beams_per_batch;
    long A = pow2(params.amb_rank);
    long N = pow2(params.dd_rank);
    long T = params.ntime;
    long S = params.nspec;
    long R = params.ringbuf_nseg;
    long RS = params.nt_per_segment * params.nspec;
    long ns = xdiv(T, params.nt_per_segment);

    xassert_shape_eq(in, ({B,A,N,T*S}));  // dedispersion buffer (4-d assumed)
    xassert_shape_eq(out, ({R*RS}));      // ringbuf
    xassert_shape_eq(params.ringbuf_locations, ({ns*A*N,4}));

    xassert(in.get_ncontig() >= 1);
    xassert(out.is_fully_contiguous());

    long dd_bstride = in.strides[0];
    long dd_astride = in.strides[1];
    long dd_nstride = in.strides[2];

    const uint *rb_loc = params.ringbuf_locations.data;
    const float *dd = in.data;
    float *ringbuf = out.data;
    
    // Loop over segments in tree.
    for (long s = 0; s < ns; s++) {
        for (long a = 0; a < A; a++) {
            for (long n = 0; n < N; n++) {
                long iseg = s*A*N + a*N + n;               // index in rb_loc array (same for all beams)
                const float *dd0 = dd + a*dd_astride + n*dd_nstride + s*RS;  // address in dedispersion buf (at beam 0)

                uint rb_offset = rb_loc[4*iseg];    // in segments, not bytes
                uint rb_phase = rb_loc[4*iseg+1];   // index of (time chunk, beam) pair, relative to current pair
                uint rb_len = rb_loc[4*iseg+2];     // number of (time chunk, beam) pairs in ringbuf (same as Ringbuf::rb_len)
                uint rb_nseg = rb_loc[4*iseg+3];    // number of segments per (time chunk, beam) (same as Ringbuf::nseg_per_beam)
                
                for (long b = 0; b < B; b++) {
                    uint i = (rb_pos + rb_phase + b) % rb_len;  // note "+b" here
                    long s = rb_offset + (i * rb_nseg);         // segment offset, relative to (float *ringbuf)
                    memcpy(ringbuf + s*RS, dd0 + b*dd_bstride, RS * sizeof(float));
                }
            }
        }
    }
}


// Helper function called by apply().
void ReferenceDedispersionKernel::_copy_from_ringbuf(const Array<float> &in, Array<float> &out, long rb_pos)
{
    long B = params.beams_per_batch;
    long A = pow2(params.amb_rank);
    long N = pow2(params.dd_rank);
    long T = params.ntime;
    long S = params.nspec;
    long R = params.ringbuf_nseg;
    long RS = params.nt_per_segment * params.nspec;
    long ns = xdiv(T, params.nt_per_segment);

    xassert_shape_eq(in, ({R*RS}));        // ringbuf
    xassert_shape_eq(out, ({B,A,N,T*S}));  // dedispersion buffer (4-d assumed)
    xassert_shape_eq(params.ringbuf_locations, ({ns*A*N,4}));
    
    xassert(in.is_fully_contiguous());
    xassert(out.get_ncontig() >= 1);

    const uint *rb_loc = params.ringbuf_locations.data;
    const float *ringbuf = in.data;
    float *dd = out.data;
    
    long dd_bstride = out.strides[0];
    long dd_astride = out.strides[1];
    long dd_nstride = out.strides[2];
    xassert(out.strides[3] == 1);

    // Loop over segments in tree.
    for (long s = 0; s < ns; s++) {
        for (long a = 0; a < A; a++) {
            for (long n = 0; n < N; n++) {
                long iseg = s*A*N + a*N + n;         // index in rb_loc array (same for all beams)
                float *dd0 = dd + n*dd_nstride + a*dd_astride + s*RS; // address in dedispersion buf (at beam 0)
                
                uint rb_offset = rb_loc[4*iseg];    // in segments, not bytes
                uint rb_phase = rb_loc[4*iseg+1];   // index of (time chunk, beam) pair, relative to current pair
                uint rb_len = rb_loc[4*iseg+2];     // number of (time chunk, beam) pairs in ringbuf (same as Ringbuf::rb_len)
                uint rb_nseg = rb_loc[4*iseg+3];    // number of segments per (time chunk, beam) (same as Ringbuf::nseg_per_beam)
                
                for (long b = 0; b < B; b++) {
                    uint i = (rb_pos + rb_phase + b) % rb_len;  // note "+b" here
                    long s = rb_offset + (i * rb_nseg);         // segment offset, relative to (float *ringbuf)
                    memcpy(dd0 + b*dd_bstride, ringbuf + s*RS, RS * sizeof(float));
                }
            }
        }
    }
}




GpuDedispersionKernel::GpuDedispersionKernel(const Params &params_) :
    params(params_)
{
    params.validate();
    xassert(params.dd_rank > 0);  // FIXME define _r0 for testing

    RegistryKey key;
    key.dtype = params.dtype;
    key.rank = params.dd_rank;
    key.input_is_ringbuf = params.input_is_ringbuf;
    key.output_is_ringbuf = params.output_is_ringbuf;
    key.apply_input_residual_lags = params.apply_input_residual_lags;
    key.nspec = params.nspec;

    // Call static member function GpuDedispersionKernel::registry().
    this->registry_value = registry().get(key);
    
    this->nbatches = xdiv(params.total_beams, params.beams_per_batch);

    int ST = xdiv(params.dtype.nbits, 8);    
    this->bw_per_launch.kernel_launches = 1;
    this->bw_per_launch.nbytes_gmem += 2 * params.beams_per_batch * pow2(params.dd_rank+params.amb_rank) * params.ntime * params.nspec * ST;
    this->bw_per_launch.nbytes_gmem += 8 * params.beams_per_batch * pow2(params.amb_rank) * registry_value.pstate32_per_small_tree;
    // FIXME(?) not currently including ringbuf_locations.

    // Important: ensure that caller-specified 'nt_per_segment' matches GPU kernel.
    xassert_eq(params.nt_per_segment, registry_value.nt_per_segment);
}


void GpuDedispersionKernel::allocate()
{
    if (is_allocated)
        throw runtime_error("double call to GpuDedispersionKernel::allocate()");
    
    // Note 'af_zero' flag here.
    long ninner = registry_value.pstate32_per_small_tree * xdiv(32, params.dtype.nbits);
    std::initializer_list<long> shape = { params.total_beams, pow2(params.amb_rank), ninner };
    this->persistent_state = Array<void> (params.dtype, shape, af_zero | af_gpu);

    // Copy host -> GPU.
    if (params.input_is_ringbuf || params.output_is_ringbuf) {
        this->gpu_ringbuf_locations = params.ringbuf_locations.to_gpu();

        long nrb = pow2(params.amb_rank + params.dd_rank) * xdiv(params.ntime, params.nt_per_segment);
        xassert_shape_eq(gpu_ringbuf_locations, ({nrb,4}));
        xassert(gpu_ringbuf_locations.is_fully_contiguous());
        xassert(gpu_ringbuf_locations.on_gpu());
    }

    this->is_allocated = true;
}


void GpuDedispersionKernel::launch(Array<void> &in_arr, Array<void> &out_arr, long ibatch, long it_chunk, cudaStream_t stream)
{
    xassert(this->is_allocated);
    xassert((ibatch >= 0) && (ibatch < nbatches));
    xassert(it_chunk >= 0);

    DedispersionKernelIobuf in(params, in_arr, params.input_is_ringbuf, true);     // on_gpu=true
    DedispersionKernelIobuf out(params, out_arr, params.output_is_ringbuf, true);  // on_gpu=true

    // The global persistent_state array has shape { total_beams, pow2(params.amb_rank), ninner }.
    // We want to select a subset of beams corresponding to the current batch.
    long b0 = (ibatch) * params.beams_per_batch;
    long b1 = (ibatch+1) * params.beams_per_batch;
    Array<void> pstate = this->persistent_state.slice(0, b0, b1);
    
    // Only used if (params.input_is_ringbuf || params.output_is_ringbuf)
    long rb_pos = (it_chunk * params.total_beams) + (ibatch * params.beams_per_batch);

    dim3 grid_dims = { uint(pow2(params.amb_rank)), uint(params.beams_per_batch), 1 };
    dim3 block_dims = { 32, uint(registry_value.warps_per_threadblock), 1 };
    ulong nt_cumul = it_chunk * params.ntime;

    if (!params.input_is_ringbuf && !params.output_is_ringbuf) {
        // Case 1: neither input nor output are ringbufs.
        auto cuda_kernel = this->registry_value.cuda_kernel_no_rb;
        xassert(cuda_kernel != nullptr);
            
        cuda_kernel<<< grid_dims, block_dims, registry_value.shmem_nbytes, stream >>>
            (in.buf, in.beam_stride32, in.amb_stride32, in.act_stride32,
             out.buf, out.beam_stride32, out.amb_stride32, out.act_stride32,
             pstate.data, params.ntime, nt_cumul, params.input_is_downsampled_tree);
    }
    else if (params.input_is_ringbuf && !params.output_is_ringbuf) {
        // Case 2: input is ringbuf.
        auto cuda_kernel = this->registry_value.cuda_kernel_in_rb;
        xassert(cuda_kernel != nullptr);
        
        cuda_kernel<<< grid_dims, block_dims, registry_value.shmem_nbytes, stream >>>
            (in.buf, gpu_ringbuf_locations.data, rb_pos,
             out.buf, out.beam_stride32, out.amb_stride32, out.act_stride32,
             pstate.data, params.ntime, nt_cumul, params.input_is_downsampled_tree);
    }   
    else if (!params.input_is_ringbuf && params.output_is_ringbuf) {
        // Case 3: output is ringbuf.
        auto cuda_kernel = this->registry_value.cuda_kernel_out_rb;
        xassert(cuda_kernel != nullptr);
            
        cuda_kernel<<< grid_dims, block_dims, registry_value.shmem_nbytes, stream >>>
            (in.buf, in.beam_stride32, in.amb_stride32, in.act_stride32,
             out.buf, gpu_ringbuf_locations.data, rb_pos,
             pstate.data, params.ntime, nt_cumul, params.input_is_downsampled_tree);
    }
    else
        throw runtime_error("DedispersionKernelParams::{input,output}_is_ringbuf flags are both set");
    
    CUDA_PEEK("dedispersion kernel");
}


// -------------------------------------------------------------------------------------------------
//
// Kernel registry.


// Helper for DedispRegistry::deferred_initialization().
template<typename F>
inline void _set_shmem(F kernel, uint nbytes)
{
    if ((kernel != nullptr) && (nbytes > 48*1024)) {
        CUDA_CALL(cudaFuncSetAttribute(
            kernel,
            cudaFuncAttributeMaxDynamicSharedMemorySize,
            nbytes
        ));
    }
}


struct DedispRegistry : public GpuDedispersionKernel::Registry
{
    using Key = GpuDedispersionKernel::RegistryKey;
    using Val = GpuDedispersionKernel::RegistryValue;
    
    virtual void add(const Key &key, const Val &val, bool debug) override
    {
        // Just check that all members have been initialized.
        // (In the future, I may add more argument checking here.)
    
        xassert((key.dtype == Dtype::native<float>()) || (key.dtype == Dtype::native<__half>()));
        xassert(key.nspec > 0);
        
        xassert(val.warps_per_threadblock > 0);
        xassert(val.nt_per_segment > 0);
        
        auto k1 = val.cuda_kernel_no_rb;
        auto k2 = val.cuda_kernel_in_rb;
        auto k3 = val.cuda_kernel_out_rb;
        
        if (!key.input_is_ringbuf && !key.output_is_ringbuf)
            xassert(k1 && !k2 && !k3);
        else if (key.input_is_ringbuf && !key.output_is_ringbuf)
            xassert(!k1 && k2 && !k3);
        else if (!key.input_is_ringbuf && key.output_is_ringbuf)
            xassert(!k1 && !k2 && k3);
        else
            throw runtime_error("DedispersionKernelParams::{input,output}_is_ringbuf flags are both set");

        // Call add() in base class.
        GpuDedispersionKernel::Registry::add(key, val, debug);
    }
    
    // Setting shared memory size is "deferred" from when the kernel is registered, to when
    // the kernel is first used. Deferring is important, since cudaFuncSetAttribute() creates
    // hard-to-debug problems if called at library initialization time, but behaves normally
    // if deferred. (Here, "hard-to-debug" means that the call appears to succeed, but an
    // unrelated kernel launch will fail later with error 400 ("invalid resource handle").)

    virtual void deferred_initialization(Val &val) override
    {
        _set_shmem(val.cuda_kernel_no_rb, val.shmem_nbytes);
        _set_shmem(val.cuda_kernel_in_rb, val.shmem_nbytes);
        _set_shmem(val.cuda_kernel_out_rb, val.shmem_nbytes);
    }
};


// Static member function
GpuDedispersionKernel::Registry &GpuDedispersionKernel::registry()
{
    // Instead of declaring the registry as a static global variable, we declare it as a
    // static local variable in the static member function GpuDedispersionKernel::registry().
    // The registry will be initialized the first time that GpuDedispersionKernel::registry()
    // is called.
    //
    // This kludge is necessary because the registry is accessed at library initialization
    // time, by callers in other source files, and source files are executed in an
    // arbitrary order.
    
    static DedispRegistry reg;
    return reg;  // note: thread-safe (as of c++11)
}


bool operator==(const GpuDedispersionKernel::RegistryKey &k1, const GpuDedispersionKernel::RegistryKey &k2)
{
    return (k1.dtype == k2.dtype) &&
        (k1.rank == k2.rank) &&
        (k1.nspec == k2.nspec) &&
        (k1.input_is_ringbuf == k2.input_is_ringbuf) &&
        (k1.output_is_ringbuf == k2.output_is_ringbuf) &&
        (k1.apply_input_residual_lags == k2.apply_input_residual_lags);
}


ostream &operator<<(ostream &os, const GpuDedispersionKernel::RegistryKey &k)
{
    os << "GpuDedispersionKernel(dtype=" << k.dtype
       << ", rank=" << k.rank
       << ", nspec=" << k.nspec
       << ", input_is_ringbuf=" << k.input_is_ringbuf
       << ", output_is_ringbuf=" << k.output_is_ringbuf
       << ", apply_input_residual_lags=" << k.apply_input_residual_lags
       << ")";

    return os;
}


ostream &operator<<(ostream &os, const GpuDedispersionKernel::RegistryValue &v)
{
    os << "warps_per_threadblock=" << v.warps_per_threadblock << ", shmem_nbytes=" << v.shmem_nbytes;
    return os;
}


// -------------------------------------------------------------------------------------------------
//
// GpuDedispersionKernel::test()
//
// Helpers are in anonymous namespace to avoid cluttering the header.
// FIXME code below really needs cleanup.


namespace {

struct DedispTestInstance
{
    // Reminder: includes 'dtype', 'input_is_ringbuf', 'output_is_ringbuf'.
    DedispersionKernelParams params;

    long nchunks = 0;

    // Can only be 'true' if params.input_is_ringbuf == params.output_is_ringbuf == false.
    bool in_place = false;
    
    // Ring buffer (only used if params.input_is_ringbuf || params.output_is_ringbuf)
    long rb_nzones = 0;
    vector<long> rb_zone_len;   // length rb_nzones
    vector<long> rb_zone_nseg;  // length rb_nzones
    vector<long> rb_zone_base_seg;  // length rb_nzones
    
    // Strides for input/output arrays.
    // Reminder: arrays have shape (params.beams_per_batch, pow2(params.amb_rank), pow2(params.dd_rank), params.ntime).
    vector<long> gpu_istrides;
    vector<long> gpu_ostrides;
    vector<long> cpu_istrides;
    vector<long> cpu_ostrides;

    // If one_hot==true, then input array will be "one-hot" rather than random.
    bool one_hot = false;
    long one_hot_ichunk = 0;    // 0 <= ichunk < nchunks
    long one_hot_ibeam = 0;     // 0 <= ibeam < params.total_beams
    long one_hot_iamb = 0;      // 0 <= iamb < pow2(params.amb_rank)
    long one_hot_iact = 0;      // 0 <= iact < pow2(params.dd_rank)
    long one_hot_itime = 0;     // 0 <= itime < params.ntime
    long one_hot_ispec = 0;     // 0 <= ispec < params.nspec

    
    void randomize()
    {
        const long max_nelts = 100 * 1000 * 1000;
        auto k = GpuDedispersionKernel::registry().get_random_key();
        auto v = GpuDedispersionKernel::registry().get(k);
        
        params.dtype = k.dtype;
        params.nspec = k.nspec;
        params.dd_rank = k.rank;
        params.input_is_ringbuf = k.input_is_ringbuf;
        params.output_is_ringbuf = k.output_is_ringbuf;
        params.apply_input_residual_lags = k.apply_input_residual_lags;
        params.input_is_downsampled_tree = (rand_uniform() < 0.5);
        params.nt_per_segment = v.nt_per_segment;
        
        this->in_place = !params.input_is_ringbuf && !params.output_is_ringbuf && (rand_uniform() < 0.5);

        long nchan = pow2(params.dd_rank);
        params.ntime = rand_int(1, 2*nchan + 2*params.nt_per_segment);
        params.ntime = align_up(params.ntime, params.nt_per_segment);

        long cmax = (10*nchan + 10*params.ntime) / params.ntime;
        this->nchunks = rand_int(1, cmax+1);

        // pow2(amb_rank), (total_beams/beams_per_batch), beams_per_batch
        long pmax = max_nelts / (nchunks * pow2(params.dd_rank) * params.ntime * params.nspec);
        pmax = max(pmax, 4L);
        pmax = min(pmax, 42L);
        
        auto s = ksgpu::random_integers_with_bounded_product(4, pmax);
        params.amb_rank = int(log2(s[0]) + 0.99999);  // round up
        params.total_beams = s[1] * s[2];
        params.beams_per_batch = s[2];

        if (params.input_is_ringbuf || params.output_is_ringbuf)
            randomize_ringbuf();

        randomize_strides();
    }


    void randomize_ringbuf()
    {
        // Now a sequence of steps to initialize:
        //
        //   params.ringbuf_locations;
        //   params.ringbuf_nseg;
        //   this->rb_nzones;
        //   this->rb_zone_len;
        //   this->rb_zone_nseg;
        //   this->rb_zone_base_seg;

        this->rb_nzones = rand_int(1, 11);
        this->rb_zone_len = vector<long> (rb_nzones, 0);
        this->rb_zone_nseg = vector<long> (rb_nzones, 0);
        this->rb_zone_base_seg = vector<long> (rb_nzones, 0);
        
        for (long z = 0; z < rb_nzones; z++) {
            long lmin = params.beams_per_batch;   // assumes that GPU does not run multiple batches in parallel
            long lmax = params.total_beams * nchunks;
            rb_zone_len[z] = rand_int(lmin, lmax+1);
        }

        long nchan = pow2(params.dd_rank);
        long nseg = nchan * pow2(params.amb_rank) * xdiv(params.ntime, params.nt_per_segment);
        vector<pair<long,long>> pairs(nseg);   // map segment -> (rb_zone, rb_seg)
        
        for (long iseg = 0; iseg < nseg; iseg++) {
            long z = rand_int(0, rb_nzones);
            long n = rb_zone_nseg[z]++;
            pairs[iseg] = {z,n};
        }
        
        ksgpu::randomly_permute(pairs);

        params.ringbuf_nseg = 0;
        for (long z = 0; z < rb_nzones; z++) {
            this->rb_zone_base_seg[z] = params.ringbuf_nseg;
            params.ringbuf_nseg += rb_zone_len[z] * rb_zone_nseg[z];
        }
        
        params.ringbuf_locations = Array<uint> ({nseg,4}, af_rhost | af_zero);
        uint *rb_loc = params.ringbuf_locations.data;
        
        for (long iseg = 0; iseg < nseg; iseg++) {
            long z = pairs[iseg].first;
            long n = pairs[iseg].second;
            long s = rb_zone_nseg[z];
            long l = rb_zone_len[z];
            
            rb_loc[4*iseg] = rb_zone_base_seg[z] + n;    // rb_offset (in segments, not bytes)
            rb_loc[4*iseg+1] = rand_int(0, 2*l+1);        // rb_phase
            rb_loc[4*iseg+2] = l;                         // rb_len
            rb_loc[4*iseg+3] = s;                         // rb_nseg
        }
    }


    void randomize_strides()
    {
        // Dedispersion buffer strides (note that ringbufs are always contiguous).
        vector<long> small_shape = { params.beams_per_batch, pow2(params.amb_rank), pow2(params.dd_rank), params.ntime, params.nspec };
        long nalign = params.nt_per_segment * params.nspec;
        
        this->cpu_istrides = ksgpu::make_random_strides(small_shape, 2, nalign);
        this->gpu_istrides = ksgpu::make_random_strides(small_shape, 2, nalign);
        this->cpu_ostrides = in_place ? cpu_istrides : ksgpu::make_random_strides(small_shape, 2, nalign);
        this->gpu_ostrides = in_place ? gpu_istrides : ksgpu::make_random_strides(small_shape, 2, nalign);
    }

#if 0
    void set_contiguous_strides()
    {
        // Dedispersion buffer strides (note that ringbufs are always contiguous).
        vector<long> small_shape = { params.beams_per_batch, pow2(params.amb_rank), pow2(params.dd_rank), params.ntime, params.nspec };
        vector<long> strides = ksgpu::make_contiguous_strides(small_shape);
        
        this->cpu_istrides = strides;
        this->gpu_istrides = strides;
        this->cpu_ostrides = strides;
        this->gpu_ostrides = strides;
    }
#endif
};


// Another helper class.
struct TestArrays
{
    DedispTestInstance tp;
    long nbatches;
    
    Array<void> big_inbuf;      // either a "big" ddbuf (nchunks), or a ringbuf
    Array<void> big_outbuf;     // either a "big" ddbuf (nchunks), or a ringbuf
    Array<void> active_inbuf;   // either a "small" ddbuf (1 chunk), or a reference to 'big_inbuf'
    Array<void> active_outbuf;  // either a "small" ddbuf (1 chunk), or a reference to 'big_outbuf'
    
    TestArrays(const DedispTestInstance &tp_, const Dtype &dtype, bool on_gpu) :
        tp(tp_),
        nbatches(xdiv(tp.params.total_beams, tp.params.beams_per_batch))
    {
        const DedispersionKernelParams &p = tp.params;
        int aflags = (on_gpu ? af_gpu : af_rhost) | af_zero;
        
        vector<long> rb_shape = { p.ringbuf_nseg * p.nt_per_segment * p.nspec };
        vector<long> big_dshape = { p.total_beams, pow2(p.amb_rank), pow2(p.dd_rank), tp.nchunks * p.ntime, p.nspec };
        vector<long> big_ishape = p.input_is_ringbuf ? rb_shape : big_dshape;
        vector<long> big_oshape = p.output_is_ringbuf ? rb_shape : big_dshape;
        vector<long> chunk_dshape = { p.beams_per_batch, pow2(p.amb_rank), pow2(p.dd_rank), p.ntime, p.nspec };
        vector<long> chunk_istrides = on_gpu ? tp.gpu_istrides : tp.cpu_istrides;
        vector<long> chunk_ostrides = on_gpu ? tp.gpu_ostrides : tp.cpu_ostrides;

        this->big_inbuf = Array<void> (dtype, big_ishape, aflags);
        this->big_outbuf = Array<void> (dtype, big_oshape, aflags);

        if (p.input_is_ringbuf)
            this->active_inbuf = big_inbuf;
        else
            this->active_inbuf = Array<void> (dtype, chunk_dshape, chunk_istrides, aflags);
        
        if (tp.in_place)
            this->active_outbuf = active_inbuf;
        else if (p.output_is_ringbuf)
            this->active_outbuf = big_outbuf;
        else
            this->active_outbuf = Array<void> (dtype, chunk_dshape, chunk_ostrides, aflags);
    }

    void copy_input(long ichunk, long ibatch)
    {
        const DedispersionKernelParams &p = tp.params;
        xassert((ichunk >= 0) && (ichunk < tp.nchunks));
        xassert((ibatch >= 0) && (ibatch < nbatches));
        
        if (!p.input_is_ringbuf) {
            Array<void> s = big_inbuf.slice(0, ibatch * p.beams_per_batch, (ibatch+1) * p.beams_per_batch);
            s = s.slice(3, ichunk * p.ntime, (ichunk+1) * p.ntime);
            active_inbuf.fill(s);   // (slice of big_inbuf) -> (active_inbuf)
        }
    }

    void copy_output(long ichunk, long ibatch)
    {
        const DedispersionKernelParams &p = tp.params;
        xassert((ichunk >= 0) && (ichunk < tp.nchunks));
        xassert((ibatch >= 0) && (ibatch < nbatches));

        if (!p.output_is_ringbuf) {
            Array<void> s = big_outbuf.slice(0, ibatch * p.beams_per_batch, (ibatch+1) * p.beams_per_batch);
            s = s.slice(3, ichunk * p.ntime, (ichunk+1) * p.ntime);
            s.fill(active_outbuf);  // (active_outbuf) -> (slice of big_outbuf)
        }
    }
};


static void run_test(const DedispTestInstance &tp)
{
    const DedispersionKernelParams &p = tp.params;
    long nbatches = xdiv(p.total_beams, p.beams_per_batch);

    cout << "\nGpuDedispersionKernel::test()\n";
    tp.params.print("    ti.params.");
    
    cout << "    ti.nchunks = " << tp.nchunks << ";\n"
         << "    ti.in_place = " << (tp.in_place ? "true" : "false") << ";\n"
         << "    ti.gpu_istrides = " << ksgpu::tuple_str(tp.gpu_istrides) << ";\n"
         << "    ti.gpu_ostrides = " << ksgpu::tuple_str(tp.gpu_ostrides) << ";\n"
         << "    ti.cpu_istrides = " << ksgpu::tuple_str(tp.cpu_istrides) << ";\n"
         << "    ti.cpu_ostrides = " << ksgpu::tuple_str(tp.cpu_ostrides) << ";\n";

    if (tp.one_hot) {
        cout << "    ti.one_hot = true\n"
             << "    ti.one_hot_ichunk = " << tp.one_hot_ichunk << "\n"
             << "    ti.one_hot_ibeam = " << tp.one_hot_ibeam << "\n"
             << "    ti.one_hot_iamb = " << tp.one_hot_iamb << "\n"
             << "    ti.one_hot_iact = " << tp.one_hot_iact << "\n"
             << "    ti.one_hot_itime = " << tp.one_hot_itime << "\n"
             << "    ti.one_hot_ispec = " << tp.one_hot_ispec << "\n";
    }
    
    shared_ptr<ReferenceDedispersionKernel> ref_kernel = make_shared<ReferenceDedispersionKernel> (p);
    shared_ptr<GpuDedispersionKernel> gpu_kernel;

    gpu_kernel = make_shared<GpuDedispersionKernel> (p);
    gpu_kernel->allocate();

    TestArrays cpu_arrs(tp, Dtype::native<float>(), false);  // on_gpu=false
    TestArrays gpu_arrs(tp, tp.params.dtype, true);          // on_gpu=true

    if (!tp.one_hot) {
        // Randomize (cpu_arrs.big_inbuf).
        // FIXME ksgpu should contain a function to randomize an array.
        Array<void> &a = cpu_arrs.big_inbuf;
        xassert(a.is_fully_contiguous());
        ksgpu::_randomize(a.dtype, a.data, a.size);
    }
    else {
        xassert(!p.input_is_ringbuf);
        xassert((tp.one_hot_ichunk >= 0) && (tp.one_hot_ichunk < tp.nchunks));
        xassert((tp.one_hot_ibeam >= 0) && (tp.one_hot_ibeam < p.total_beams));
        xassert((tp.one_hot_iamb >= 0) && (tp.one_hot_iamb < pow2(p.amb_rank)));
        xassert((tp.one_hot_iact >= 0) && (tp.one_hot_iact < pow2(p.dd_rank)));
        xassert((tp.one_hot_itime >= 0) && (tp.one_hot_itime < p.ntime));
        xassert((tp.one_hot_ispec >= 0) && (tp.one_hot_ispec < p.nspec));

        long b = tp.one_hot_ibeam;
        long a = tp.one_hot_iamb;
        long d = tp.one_hot_iact;
        long t = tp.one_hot_ichunk * p.ntime + tp.one_hot_itime;
        long s = tp.one_hot_ispec;

        Array<float> dst = cpu_arrs.big_inbuf.cast<float>();
        dst.at({b,a,d,t,s}) = 1.0f;
    }

    // Copy (cpu_arrs.big_inbuf) -> (gpu_arrs.big_inbuf), converting dtype if necessary.
    // FIXME ksgpu should contain a function for this.
    Array<void> src = cpu_arrs.big_inbuf;
    if (src.dtype != tp.params.dtype)
        src = src.convert(tp.params.dtype);
    gpu_arrs.big_inbuf.fill(src);
    src = Array<void>();  // free memory
    
    for (long ichunk = 0; ichunk < tp.nchunks; ichunk++) {
        for (long ibatch = 0; ibatch < nbatches; ibatch++) {
            // Reference dedispersion.
            cpu_arrs.copy_input(ichunk, ibatch);
            ref_kernel->apply(cpu_arrs.active_inbuf, cpu_arrs.active_outbuf, ibatch, ichunk);
            cpu_arrs.copy_output(ichunk, ibatch);
            
            // GPU dedipersion.
            gpu_arrs.copy_input(ichunk, ibatch);
            gpu_kernel->launch(gpu_arrs.active_inbuf, gpu_arrs.active_outbuf, ibatch, ichunk, nullptr);  // stream=nullptr
            gpu_arrs.copy_output(ichunk, ibatch);
        }
    }
    
    // FIXME revisit epsilon if we change the normalization of the dedispersion transform.
    double epsrel = 3 * tp.params.dtype.precision();
    double epsabs = 3 * tp.params.dtype.precision() * pow(1.414, p.dd_rank);

    if (p.output_is_ringbuf)
        ksgpu::assert_arrays_equal(cpu_arrs.big_outbuf, gpu_arrs.big_outbuf, "cpu", "gpu", {"i"}, epsabs, epsrel);
    else
        ksgpu::assert_arrays_equal(cpu_arrs.big_outbuf, gpu_arrs.big_outbuf, "cpu", "gpu", {"beam","amb","dmbr","time","spec"}, epsabs, epsrel);
}

}  // anonymous namespace


// Static member function
void GpuDedispersionKernel::test()
{
#if 0
    // Debug
    DedispTestInstance ti;
    ti.params.dtype = Dtype::native<float>();
    ti.params.dd_rank = 3;
    ti.params.amb_rank = 0;
    ti.params.total_beams = 1;
    ti.params.beams_per_batch = 1;
    ti.params.ntime = 4;
    ti.params.nspec = 8;
    ti.params.input_is_ringbuf = false;
    ti.params.output_is_ringbuf = false;
    ti.params.apply_input_residual_lags = false;
    ti.params.input_is_downsampled_tree = false;
    ti.params.nt_per_segment = 4;
    ti.nchunks = 2;
    ti.randomize_ringbuf();
    ti.set_contiguous_strides();
    // ti.one_hot = true;
    // ti.one_hot_iact = 0;   // rand_int(0, pow2(ti.params.dd_rank));
    // ti.one_hot_itime = 0;  // rand_int(0, ti.params.ntime);
    run_test(ti);
#endif    

#if 1
    DedispTestInstance ti;
    ti.randomize();
    run_test(ti);
#endif
}


// -------------------------------------------------------------------------------------------------
//
// GpuDedispersionKernel::time() implementation


// Static member function.
// Uses one stream per "beam batch".
void GpuDedispersionKernel::_time(const DedispersionKernelParams &params, long nchunks)
{
    cout << "\nTime GPU dedispersion kernel\n";
    params.print();
    
    long nbatches = xdiv(params.total_beams, params.beams_per_batch);

    shared_ptr<GpuDedispersionKernel> kernel = make_shared<GpuDedispersionKernel> (params);
    kernel->allocate();
    
    vector<long> dd_shape = { params.total_beams, pow2(params.amb_rank), pow2(params.dd_rank), params.ntime, params.nspec };
    vector<long> rb_shape = { params.ringbuf_nseg * params.nt_per_segment * params.nspec };
    vector<long> in_shape = params.input_is_ringbuf ? rb_shape : dd_shape;
    vector<long> out_shape = params.output_is_ringbuf ? rb_shape : dd_shape;

    Array<void> in_big(params.dtype, in_shape, af_gpu | af_zero);
    Array<void> out_big(params.dtype, out_shape, af_gpu | af_zero);
    double gb_per_launch = 1.0e-9 * kernel->bw_per_launch.nbytes_gmem;

    KernelTimer kt(nbatches);   // one stream per batch

    for (long ichunk = 0; ichunk < nchunks; ichunk++) {
        for (long ibatch = 0; ibatch < nbatches; ibatch++) {
            Array<void> in_slice = in_big;
            Array<void> out_slice = out_big;

            if (!params.input_is_ringbuf)
                in_slice = in_big.slice(0, ibatch * params.beams_per_batch, (ibatch+1) * params.beams_per_batch);
            if (!params.output_is_ringbuf)
                out_slice = out_big.slice(0, ibatch * params.beams_per_batch, (ibatch+1) * params.beams_per_batch);

            kernel->launch(in_slice, out_slice, ibatch, ichunk, kt.stream);

            if (kt.advance() && (ichunk % 2))
                cout << "   [ " << (gb_per_launch/kt.dt) << " GB/s ]\n";
        }
    }

    cout << endl;
}


// static
void GpuDedispersionKernel::time()
{
#if 0
    // Time specific kernel.
    DedispersionKernelParams p;
    p.dtype = Dtype::native<float> ();
    p.dd_rank = 8;
    p.amb_rank = 1;
    p.total_beams = 1;
    p.beams_per_batch = 1;
    p.ntime = 32;
    p.nspec = 1;
    p.input_is_ringbuf = false;
    p.output_is_ringbuf = false;
    p.apply_input_residual_lags = false;
    p.input_is_downsampled_tree = false;
    p.nt_per_segment = 32;
    time_gpu_dedispersion_kernel(p, 1);  // nchunks=1
#endif

#if 1
    // Time a few representative kernels.
    long nstreams = 2;

    for (int dd_rank: {4,8}) {
        for (int stage: {0,1,2}) {
            for (Dtype dtype: { Dtype::native<float>(), Dtype::native<__half>() }) {
                long nspec = 1;  // FIXME
                long nbeams = pow2(19 - 2*dd_rank);
    
                DedispersionKernelParams params;
                params.dtype = dtype;
                params.dd_rank = dd_rank;
                params.amb_rank = dd_rank;
                params.beams_per_batch = nbeams;
                params.total_beams = nbeams * nstreams;
                params.ntime = xdiv(2048, nspec);
                params.nspec = nspec;
                params.input_is_ringbuf = (stage == 2);
                params.output_is_ringbuf = (stage == 1);        
                params.apply_input_residual_lags = (stage == 2);
                params.input_is_downsampled_tree = false;  // shouldn't affect timing
                params.nt_per_segment = xdiv(1024, dtype.nbits * nspec);

                if (params.input_is_ringbuf || params.output_is_ringbuf) {
                    // Make some nominal ringbuf locations.
                    // The details shouldn't affect the timing much.

                    long rb_len = 2 * params.total_beams;
                    long nrows_per_tree = pow2(params.dd_rank + params.amb_rank);
                    long nseg_per_row = xdiv(params.ntime, params.nt_per_segment);
                    long nseg_per_tree = nrows_per_tree * nseg_per_row;

                    params.ringbuf_nseg = rb_len * nseg_per_tree;
                    params.ringbuf_locations = Array<uint> ({nseg_per_tree,4}, af_rhost | af_zero);
                    uint *rp = params.ringbuf_locations.data;

                    for (long iseg = 0; iseg < nseg_per_tree; iseg++) {
                        rp[4*iseg] = iseg;             // rb_offset
                        rp[4*iseg+1] = 0;              // rb_phase
                        rp[4*iseg+2] = rb_len;         // rb_len
                        rp[4*iseg+3] = nseg_per_tree;  // rb_nseg
                    }
                }

               GpuDedispersionKernel::_time(params);
            }
        }
    }
#endif
}


}  // namespace pirate
