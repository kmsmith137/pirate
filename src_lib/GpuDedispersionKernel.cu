#include "../include/pirate/DedispersionKernel.hpp"
#include "../include/pirate/KernelRegistry.hpp"
#include "../include/pirate/constants.hpp"
#include "../include/pirate/inlines.hpp"   // pow2(), is_aligned(), simd_type
#include "../include/pirate/utils.hpp"     // bit_reverse_slow()

#include "../include/pirate/cuda_kernels/dedispersion_iobufs.hpp"  // struct dedispersion_{simple,ring}_{inbuf,output}

#include <mutex>
#include <sstream>
#include <ksgpu/xassert.hpp>
#include <ksgpu/cuda_utils.hpp>  // CUDA_CALL()
#include <ksgpu/rand_utils.hpp>  // rand_int()
#include <ksgpu/string_utils.hpp>

using namespace std;
using namespace ksgpu;

namespace pirate {
#if 0
}  // editor auto-indent
#endif


// Defined in include/pirate/cuda_kernels/dedispersion_kernel.hpp
// Instantiated in src_lib/template_instantiations/*.cu
template<typename T, class Inbuf, class Outbuf> extern void dedisperse_r1(typename Inbuf::device_args, typename Outbuf::device_args, T *rstate, long ntime, uint *integer_constants, long rb_pos);
template<typename T, class Inbuf, class Outbuf> extern void dedisperse_r2(typename Inbuf::device_args, typename Outbuf::device_args, T *rstate, long ntime, uint *integer_constants, long rb_pos);
template<typename T, class Inbuf, class Outbuf> extern void dedisperse_r3(typename Inbuf::device_args, typename Outbuf::device_args, T *rstate, long ntime, uint *integer_constants, long rb_pos);
template<typename T, class Inbuf, class Outbuf> extern void dedisperse_r4(typename Inbuf::device_args, typename Outbuf::device_args, T *rstate, long ntime, uint *integer_constants, long rb_pos);
template<typename T, class Inbuf, class Outbuf> extern void dedisperse_r5(typename Inbuf::device_args, typename Outbuf::device_args, T *rstate, long ntime, uint *integer_constants, long rb_pos);
template<typename T, class Inbuf, class Outbuf> extern void dedisperse_r6(typename Inbuf::device_args, typename Outbuf::device_args, T *rstate, long ntime, uint *integer_constants, long rb_pos);
template<typename T, class Inbuf, class Outbuf> extern void dedisperse_r7(typename Inbuf::device_args, typename Outbuf::device_args, T *rstate, long ntime, uint *integer_constants, long rb_pos);
template<typename T, class Inbuf, class Outbuf> extern void dedisperse_r8(typename Inbuf::device_args, typename Outbuf::device_args, T *rstate, long ntime, uint *integer_constants, long rb_pos);


template<typename T> struct _is_float32 { };
template<> struct _is_float32<float>   { static constexpr bool value = true; };
template<> struct _is_float32<__half>  { static constexpr bool value = false; };
template<> struct _is_float32<__half2> { static constexpr bool value = false; };


// When shared memory ring buffer is saved/restored in global memory, how many cache lines do we need?
template<typename T, int Rank> struct _gs_ncl { };

// Precomputed in git/chord/frb_search/r8_hacking.py
template<> struct _gs_ncl<float,5>    { static constexpr int value = 6; };
template<> struct _gs_ncl<__half2,5>  { static constexpr int value = 3; };
template<> struct _gs_ncl<float,6>    { static constexpr int value = 25; };
template<> struct _gs_ncl<__half2,6>  { static constexpr int value = 12; };
template<> struct _gs_ncl<float,7>    { static constexpr int value = 105; };
template<> struct _gs_ncl<__half2,7>  { static constexpr int value = 52; };
template<> struct _gs_ncl<float,8>    { static constexpr int value = 450; };
template<> struct _gs_ncl<__half2,8>  { static constexpr int value = 224; };

// Number of global (not shared) memory cache lines needed to store ring buffers.
static __host__ int get_gs_ncl(int rank, bool is_float32)
{
    if (rank <= 4)
	return 0;
    else if (rank == 5)
	return is_float32 ? _gs_ncl<float,5>::value : _gs_ncl<__half2,5>::value;
    else if (rank == 6)
	return is_float32 ? _gs_ncl<float,6>::value : _gs_ncl<__half2,6>::value;
    else if (rank == 7)
	return is_float32 ? _gs_ncl<float,7>::value : _gs_ncl<__half2,7>::value;
    else if (rank == 8)
	return is_float32 ? _gs_ncl<float,8>::value : _gs_ncl<__half2,8>::value;
    else
	throw runtime_error("bad arguments to get_gs_ncl()");
}

// Shared memory ring buffer footprint is larger than global memory, by 2^rank cache lines.
static __host__ int get_shmem_nbytes(int rank, bool is_float32)
{
    int ncl = pow2(rank) + get_gs_ncl(rank, is_float32);
    return ncl * 128;  // 1 cache line = 128 bytes
}


// The "integer constants" array looks like this:
//
//   uint32 control_words[2^rank1][2^rank0];  // indexed by (i,j)
//   uint32 gmem_specs[gs_ncl][2];
//
// A ring buffer "control word" consists of:
//
//   uint15 rb_base;   // base shared memory location of ring buffer (in 32-bit registers)
//   uint8  rb_pos;    // current position, satisfying 0 <= rb_pos < (rb_lag + 32)
//   uint8  rb_lag;    // ring buffer lag (in 32-bit registers), note that capacity = lag + 32.
//
// Depending on context, 'shmem_curr_pos' may point to either the end of the buffer
// (writer thread context), or be appropriately lagged (reader thread context).
//
// A "gmem spec" is a pair describing how a global memory cache line gets scattered into shared memory.
//
//   uint32 shmem_base;  // in 32-bit registers, will always be a multiple of 32
//   uint32 gap_bits;    // FIXME write comment explaining this
//
// FIXME it would probably be better to keep the integer constants array in GPU constant memory.
// (Currently we keep it in global memory.) Before doing this, I wanted to answer some initial
// questions about constant memory (search "CHORD TODO" google doc for "constant memory").


// If on_gpu=false, array is returned on host
// If on_gpu=true, array is returned on GPU.

static __host__ Array<uint> make_integer_constants(int rank, bool is_float32, bool on_gpu)
{
    if (rank <= 4)
	return Array<uint> ();

    xassert_le(rank, 8);
    int rank0 = rank >> 1;  // round down
    int rank1 = rank - rank0;
    
    int gs_ncl = get_gs_ncl(rank, is_float32);
    int shmem_nbytes = get_shmem_nbytes(rank, is_float32);
    
    // Total size of integer_constants array (control_words + gmem_specs)
    int ret_nelts = align_up(pow2(rank) + 2*gs_ncl, 32);
    Array<uint> ret({ret_nelts}, af_rhost | af_zero);

    // Tracks current shared memory footprint.
    uint shmem_nreg = 0;

    // Tracks current gmem spec.
    uint gs_icl = 0;    // global memory cache line index
    uint gs_ireg = 0;   // global memory register within cache line (in 0,1,...,31).
    uint gs_spos = 0;   // shared memory position (in 32-bit registers)
    uint gs_sbase = 0;  // shared memory "base" position (i.e. value at ireg=0).
    uint gs_gbits = 0;  // gap bits (initialized at ...)

    // We order the ring buffers so that all the zero-lag buffers are first, followed
    // by the nonzero-lag buffers. We implement this by running the loop twice.
    //
    // (This is necessary because the 'gap_bits' logic doesn't allow two 32-register
    //  "gaps" in a row, so we can't put zero-lag buffers between nonzero-lag buffers.)

    for (int pass = 0; pass < 2; pass++) {
	for (int i = 0; i < pow2(rank1); i++) {
	    for (int j = 0; j < pow2(rank0); j++) {
		// Ring buffer lag, in 32-bit registers.
		int ff = pow2(rank1) - i - 1;
		int dm = bit_reverse_slow(j, rank0);
		int lag = (ff*dm) >> (is_float32 ? 0 : 1);

		// Process zero-lag buffers in first pass, nonzero-lag in second pass.
		if (pass != (lag ? 1 : 0))
		    continue;

		// Ensure no overflow in control word.
		xassert_lt(shmem_nreg, 32768);  // uint15 rb_base
		xassert_lt(lag, 256);           // uint8 rb_lag
		
		// Control words are stored in global memory at "writer offset".
		// To get "reader offset", set pos=0 by applying mask 0xff007fff.
		// (See read_control_words() below.)
		
		int s = i * pow2(rank0) + j;
		ret.at({s}) = shmem_nreg | (lag << 15) | (lag << 24);  // control word
	    
		for (int l = 0; l < lag; l++) {
		    if (gs_ireg == 32) {
			// Write completed gmem_spec to 'integer_constants' array.
			xassert(int(gs_icl) < gs_ncl);
			ret.at({pow2(rank) + 2*gs_icl}) = gs_sbase;
			ret.at({pow2(rank) + 2*gs_icl+1}) = gs_gbits;
			gs_icl++;
			gs_ireg = 0;
			gs_sbase = gs_spos;
			gs_gbits = 0;
		    }

		    uint spos = shmem_nreg + l;
		    
		    if (gs_ireg == 0)
			gs_sbase = gs_spos = spos;
		    
		    if (spos == gs_spos + 32) {
			gs_gbits |= (1 << gs_ireg);
			gs_spos += 32;
		    }

		    xassert(gs_spos == spos);
		    gs_ireg++;
		    gs_spos++;
		}

		// Ring buffer size = (lag + 32).
		shmem_nreg += (lag + 32);
	    }
	}
    }
    
    // After loop completes, the last gmem spec should be partially or fully complete.
    xassert(gs_ireg > 0);
    xassert(gs_ireg <= 32);
    xassert(int(gs_icl) == (gs_ncl-1));

    // This assert ensures that we have enough shared memory "headroom".
    int gs_smax = gs_spos + (32 - gs_ireg);
    xassert_le(gs_smax, int(shmem_nreg));
    xassert_le(4*int(shmem_nreg), shmem_nbytes);

    // Write last gmem spec.
    ret.at({pow2(rank) + 2*gs_icl}) = gs_sbase;
    ret.at({pow2(rank) + 2*gs_icl+1}) = gs_gbits;

    return on_gpu ? ret.to_gpu() : ret;
}


// -------------------------------------------------------------------------------------------------


template<typename T, bool Lagged>
dedispersion_simple_inbuf<T,Lagged>::device_args::device_args(const Array<void> &in_arr_, const GpuDedispersionKernel &kernel)
{
    // If T==float, then T32 is also 'float'.
    // If T==__half, then T32 is '__half2'.
    using T32 = typename simd32_type<T>::type;

    constexpr int elts_per_cache_line = constants::bytes_per_gpu_cache_line / 4;
    constexpr int denom = 4 / sizeof(T);
    static_assert(denom * sizeof(T) == 4);

    Array<T> in_arr = in_arr_.template cast<T> ("dedispersion_simple_inbuf input array");
    const DedispersionKernelParams &params = kernel.params;
    
    // Expected shape is (nbeams, pow2(amb_rank), pow2(dd_rank), ntime).
    xassert_shape_eq(in_arr, ({params.beams_per_batch, pow2(params.amb_rank), pow2(params.dd_rank), params.ntime}));
    xassert(in_arr.get_ncontig() >= 1);
    xassert(in_arr.on_gpu());

    this->in = (T32 *) in_arr.data;
    this->beam_stride32 = xdiv(in_arr.strides[0], denom);     // 32-bit stride
    this->ambient_stride32 = xdiv(in_arr.strides[1], denom);  // 32-bit stride
    this->freq_stride32 = xdiv(in_arr.strides[2], denom);     // 32-bit stride
    this->is_downsampled = params.input_is_downsampled_tree;

    // Check alignment. Not strictly necessary, but failure would be unintentional and indicate a bug somewhere.
    xassert(is_aligned(in, constants::bytes_per_gpu_cache_line));   // also checks non_NULL
    xassert_divisible(beam_stride32, elts_per_cache_line);
    xassert_divisible(ambient_stride32, elts_per_cache_line);
    xassert_divisible(freq_stride32, elts_per_cache_line);
    
    // FIXME could improve these checks, by verifying that strides are non-overlapping.
    xassert((params.beams_per_batch == 1) || (beam_stride32 != 0));
    xassert((params.amb_rank == 0) || (ambient_stride32 != 0));
    xassert((params.dd_rank == 0) || (freq_stride32 != 0));
}


// FIXME reduce cut-and-paste between Inbuf::host_args and Outbuf::host_args constructors.
template<typename T>
dedispersion_simple_outbuf<T>::device_args::device_args(const Array<void> &out_arr_, const GpuDedispersionKernel &kernel)
{
    // If T==float, then T32 is also 'float'.
    // If T==__half, then T32 is '__half2'.
    using T32 = typename simd32_type<T>::type;

    constexpr int elts_per_cache_line = constants::bytes_per_gpu_cache_line / 4;
    constexpr int denom = 4 / sizeof(T);
    static_assert(denom * sizeof(T) == 4);

    Array<T> out_arr = out_arr_.template cast<T> ("dedispersion_simple_outbuf output array");
    const DedispersionKernelParams &params = kernel.params;
    
    // Expected shape is (nbeams, pow2(amb_rank), pow2(dd_rank), ntime)
    xassert_shape_eq(out_arr, ({ params.beams_per_batch, pow2(params.amb_rank), pow2(params.dd_rank), params.ntime }));
    xassert(out_arr.get_ncontig() >= 1);
    xassert(out_arr.on_gpu());

    this->out = (T32 *) out_arr.data;
    this->beam_stride32 = xdiv(out_arr.strides[0], denom);     // 32-bit stride
    this->ambient_stride32 = xdiv(out_arr.strides[1], denom);  // 32-bit stride
    this->dm_stride32 = xdiv(out_arr.strides[2], denom);     // 32-bit stride
    
    // Check alignment. Not strictly necessary, but failure would be unintentional and indicate a bug somewhere.
    xassert(is_aligned(out, constants::bytes_per_gpu_cache_line));   // also checks non-NULL
    xassert_divisible(beam_stride32, elts_per_cache_line);
    xassert_divisible(ambient_stride32, elts_per_cache_line);
    xassert_divisible(dm_stride32, elts_per_cache_line);
    
    // FIXME could improve these checks, by verifying that strides are non-overlapping.
    xassert((params.beams_per_batch == 1) || (beam_stride32 != 0));
    xassert((params.amb_rank == 0) || (ambient_stride32 != 0));
    xassert((params.dd_rank == 0) || (dm_stride32 != 0));
}


// -------------------------------------------------------------------------------------------------


template<typename T>
dedispersion_ring_inbuf<T>::device_args::device_args(const Array<void> &in_arr_, const GpuDedispersionKernel &kernel)
{
    // If T==float, then T32 is also 'float'.
    // If T==__half, then T32 is '__half2'.
    using T32 = typename simd32_type<T>::type;

    constexpr int denom = 4 / sizeof(T);
    static_assert(denom * sizeof(T) == 4);

    Array<T> in_arr = in_arr_.template cast<T> ("dedispersion_ring_inbuf input array");
    const DedispersionKernelParams &params = kernel.params;
	
    xassert_shape_eq(in_arr, ({ params.ringbuf_nseg * params.nelts_per_segment }));
    xassert(in_arr.get_ncontig() == 1);
    xassert(in_arr.on_gpu());

    const Array<uint> &rb_loc = kernel.gpu_ringbuf_locations;  // not params.ringbuf_locations, which is on the host!
    xassert_shape_eq(rb_loc, ({ pow2(params.amb_rank + params.dd_rank) * xdiv(params.ntime, params.nelts_per_segment), 4 }));
    xassert(rb_loc.is_fully_contiguous());
    xassert(rb_loc.on_gpu());

    this->rb_base = (const T32 *) in_arr.data;
    this->rb_loc = (const uint4 *) rb_loc.data;
    this->is_downsampled = params.input_is_downsampled_tree;
}


template<typename T>
dedispersion_ring_outbuf<T>::device_args::device_args(const Array<void> &out_arr_, const GpuDedispersionKernel &kernel)
{
    // If T==float, then T32 is also 'float'.
    // If T==__half, then T32 is '__half2'.
    using T32 = typename simd32_type<T>::type;

    constexpr int denom = 4 / sizeof(T);
    static_assert(denom * sizeof(T) == 4);

    Array<T> out_arr = out_arr_.template cast<T> ("dedispersion_ring_outbuf output array");
    const DedispersionKernelParams &params = kernel.params;
	
    xassert_shape_eq(out_arr, ({ params.ringbuf_nseg * params.nelts_per_segment }));
    xassert(out_arr.get_ncontig() == 1);
    xassert(out_arr.on_gpu());

    const Array<uint> &rb_loc = kernel.gpu_ringbuf_locations;  // not params.ringbuf_locations, which is on the host!
    xassert_shape_eq(rb_loc, ({ pow2(params.amb_rank + params.dd_rank) * xdiv(params.ntime, params.nelts_per_segment), 4 }));
    xassert(rb_loc.is_fully_contiguous());
    xassert(rb_loc.on_gpu());

    this->rb_base = (T32 *) out_arr.data;
    this->rb_loc = (const uint4 *) rb_loc.data;
}


// -------------------------------------------------------------------------------------------------


template<typename T, class Inbuf, class Outbuf>
struct GpuDedispersionKernelImpl : public GpuDedispersionKernel
{    
    // If T==float, then T32 is also 'float'.
    // If T==__half, then T32 is '__half2'.
    using T32 = typename simd32_type<T>::type;

    GpuDedispersionKernelImpl(const DedispersionKernelParams &params);

    virtual void launch(Array<void> &in, Array<void> &out, long ibatch, long it_chunk, cudaStream_t stream) override;

    // (inbuf, outbuf, rstate, ntime, integer_constants, rb_pos)
    void (*cuda_kernel)(typename Inbuf::device_args, typename Outbuf::device_args, T32 *, long, uint *, long) = nullptr;
};


template<typename T, class Inbuf, class Outbuf>
GpuDedispersionKernelImpl<T,Inbuf,Outbuf>::GpuDedispersionKernelImpl(const Params &params_)
    : GpuDedispersionKernel(params_)   // calls params_.validate()
{
    if (params.dd_rank == 1)
	this->cuda_kernel = dedisperse_r1<T32, Inbuf, Outbuf>;
    else if (params.dd_rank == 2)
	this->cuda_kernel = dedisperse_r2<T32, Inbuf, Outbuf>;
    else if (params.dd_rank == 3)
	this->cuda_kernel = dedisperse_r3<T32, Inbuf, Outbuf>;
    else if (params.dd_rank == 4)
	this->cuda_kernel = dedisperse_r4<T32, Inbuf, Outbuf>;
    else if (params.dd_rank == 5)
	this->cuda_kernel = dedisperse_r5<T32, Inbuf, Outbuf>;
    else if (params.dd_rank == 6)
	this->cuda_kernel = dedisperse_r6<T32, Inbuf, Outbuf>;
    else if (params.dd_rank == 7)
	this->cuda_kernel = dedisperse_r7<T32, Inbuf, Outbuf>;
    else if (params.dd_rank == 8)
	this->cuda_kernel = dedisperse_r8<T32, Inbuf, Outbuf>;
    else
	throw runtime_error("expected 1 <= DedispersionKernelParams::dd_rank <= 8");

    // Note: this->shmem_bytes is initialized by the base class constructor.
    
    if (shmem_nbytes > 48*1024) {
        CUDA_CALL(cudaFuncSetAttribute(
	    cuda_kernel,
            cudaFuncAttributeMaxDynamicSharedMemorySize,
            shmem_nbytes
        ));
    }
}


// virtual override
template<typename T, class Inbuf, class Outbuf>
void GpuDedispersionKernelImpl<T,Inbuf,Outbuf>::launch(Array<void> &in_arr, Array<void> &out_arr, long ibatch, long it_chunk, cudaStream_t stream)
{
    xassert(this->is_allocated);
    xassert((ibatch >= 0) && (ibatch < nbatches));
    xassert(it_chunk >= 0);
    
    // These constructors error-check their arguments (including array shapes).
    typename Inbuf::device_args in(in_arr, *this);
    typename Outbuf::device_args out(out_arr, *this);

    Array<T> pstate = this->persistent_state.template cast<T> ("pstate");
    T *pp = pstate.data + (ibatch * params.beams_per_batch * this->state_nelts_per_beam);
    long rb_pos = (it_chunk * params.total_beams) + (ibatch * params.beams_per_batch);

    // Note: the number of beams and 'ambient' tree channels are implicitly supplied
    // to the kernel via gridDim.y, gridDim.x.
    dim3 grid_dims;
    grid_dims.x = pow2(params.amb_rank);
    grid_dims.y = params.beams_per_batch;
    grid_dims.z = 1;

    this->cuda_kernel
	<<< grid_dims, 32 * warps_per_threadblock, shmem_nbytes, stream >>>
	(in, out, (T32 *) pp, params.ntime, this->integer_constants.data, rb_pos);
    
    CUDA_PEEK("dedispersion kernel");
}


// -------------------------------------------------------------------------------------------------


GpuDedispersionKernel::GpuDedispersionKernel(const Params &params_) :
    params(params_)
{
    params.validate(true);    // on_gpu=true
    xassert(params.dd_rank > 0);  // FIXME define _r0 for testing
	
    // FIXME remaining code is cut-and-paste from previous API -- could use a rethink.

    bool is_float32 = (params.dtype.nbits == 32);
    int nrs_per_thread;
    
    if (params.dd_rank == 1) {
	this->warps_per_threadblock = 1;
	nrs_per_thread = 1;
    }
    else if (params.dd_rank == 2) {
	this->warps_per_threadblock = 1;
	nrs_per_thread = 1;
    }
    else if (params.dd_rank == 3) {
	this->warps_per_threadblock = 1;
	nrs_per_thread = 1;
    }
    else if (params.dd_rank == 4) {
	this->warps_per_threadblock = 1;
	nrs_per_thread = is_float32 ? 3 : 2;
    }
    else if (params.dd_rank == 5) {
	this->warps_per_threadblock = 4;
	nrs_per_thread = 1;
    }
    else if (params.dd_rank == 6) {
	this->warps_per_threadblock = 8;
	nrs_per_thread = is_float32 ? 2 : 1;
    }
    else if (params.dd_rank == 7) {
	this->warps_per_threadblock = 8;
	nrs_per_thread = is_float32 ? 4 : 3;
    }
    else if (params.dd_rank == 8) {
	this->warps_per_threadblock = 16;
	nrs_per_thread = is_float32 ? 5 : 4;
    }
    else
	throw runtime_error("GpuDedispersionKernel constructor: should never get here");
    
    long swflag = (warps_per_threadblock == 1);
    long rp_ncl = params.apply_input_residual_lags ? (pow2(params.dd_rank) - swflag) : 0;
    long rs_ncl = warps_per_threadblock * nrs_per_thread;
    long gs_ncl = get_gs_ncl(params.dd_rank, is_float32);
    long nelts_per_small_tree = (rs_ncl + rp_ncl + gs_ncl) * (is_float32 ? 32 : 64);

    this->nbatches = xdiv(params.total_beams, params.beams_per_batch);
    this->state_nelts_per_beam = pow2(params.amb_rank) * nelts_per_small_tree;
    
    if (gs_ncl > 0)
	this->shmem_nbytes = 128 * (gs_ncl + pow2(params.dd_rank));

    int ST = xdiv(params.dtype.nbits, 8);
    
    this->bw_per_launch.kernel_launches = 1;
    this->bw_per_launch.nbytes_gmem += params.total_beams * pow2(params.dd_rank+params.amb_rank) * params.ntime * ST;
    this->bw_per_launch.nbytes_gmem += 2 * params.total_beams * state_nelts_per_beam * ST;
    // FIXME(?) not currently including ringbuf_locations.
}


void GpuDedispersionKernel::allocate()
{
    if (is_allocated)
	throw runtime_error("double call to GpuDedispersionKernel::allocate()");
    
    // Note 'af_zero' flag here.
    std::initializer_list<long> shape = { params.total_beams, state_nelts_per_beam };
    this->persistent_state = Array<void> (params.dtype, shape, af_zero | af_gpu);
    
    bool is_float32 = (params.dtype.nbits == 32);
    this->integer_constants = make_integer_constants(params.dd_rank, is_float32, true);   // on_gpu=true

    // Copy host -> GPU.
    if (params.input_is_ringbuf || params.output_is_ringbuf)
	this->gpu_ringbuf_locations = params.ringbuf_locations.to_gpu();

    this->is_allocated = true;
}


// Static member function
shared_ptr<GpuDedispersionKernel> GpuDedispersionKernel::make(const Params &params)
{
    params.validate(true);  // on_gpu=true
    
    bool rb_in = params.input_is_ringbuf;
    bool rb_out = params.output_is_ringbuf;
    bool rlag = params.apply_input_residual_lags;
    bool is_float32 = (params.dtype.nbits == 32);

    // Select subclass template instantiation.
    // Note: templates are instantiated and compiled in src_lib/template_instatiations/.cu
    // FIXME could reduce cut-and-paste here.

    if (!rb_in && !rb_out && !rlag && is_float32)
	return make_shared<GpuDedispersionKernelImpl<float, dedispersion_simple_inbuf<float,false>, dedispersion_simple_outbuf<float>>> (params);
    else if (!rb_in && !rb_out && !rlag && !is_float32)
	return make_shared<GpuDedispersionKernelImpl<__half, dedispersion_simple_inbuf<__half,false>, dedispersion_simple_outbuf<__half>>> (params);
    else if (!rb_in && !rb_out && rlag && is_float32)
	return make_shared<GpuDedispersionKernelImpl<float, dedispersion_simple_inbuf<float,true>, dedispersion_simple_outbuf<float>>> (params);
    else if (!rb_in && !rb_out && rlag && !is_float32)
	return make_shared<GpuDedispersionKernelImpl<__half, dedispersion_simple_inbuf<__half,true>, dedispersion_simple_outbuf<__half>>> (params);
    else if (!rb_in && rb_out && !rlag && is_float32)
	return make_shared<GpuDedispersionKernelImpl<float, dedispersion_simple_inbuf<float,false>, dedispersion_ring_outbuf<float>>> (params);
    else if (!rb_in && rb_out && !rlag && !is_float32)
	return make_shared<GpuDedispersionKernelImpl<__half, dedispersion_simple_inbuf<__half,false>, dedispersion_ring_outbuf<__half>>> (params);
    else if (rb_in && !rb_out && rlag && is_float32)
	return make_shared<GpuDedispersionKernelImpl<float, dedispersion_ring_inbuf<float>, dedispersion_simple_outbuf<float>>> (params);
    else if (rb_in && !rb_out && rlag && !is_float32)
	return make_shared<GpuDedispersionKernelImpl<__half, dedispersion_ring_inbuf<__half>, dedispersion_simple_outbuf<__half>>> (params);
    
    throw runtime_error("GpuDedispersionKernel::make(): no suitable precompiled kernel could be found");
}


// -------------------------------------------------------------------------------------------------
//
// NewGpuDedispersionKernel


    
// dd_iobuf: this helper class is used in NewGpuDedispersionKenrel::launch(), to process
// and error-check the input/output arrays.
//
// Recall that the 'in' and 'out' arrays are either "simple" buffers or ringbufs, depending on
// values of Params::input_is_ringbuf and Params::output_is_ringbuf. Shapes are:
//
//   - simple buf has shape (params.beams_per_batch, pow2(amb_rank), pow2(dd_rank), ntime).
//   - ringbuf has 1-d shape (params.ringbuf_nseg * params.nelts_per_segment,)

struct dd_iobuf
{
    bool is_ringbuf;
    void *buf = nullptr;

    // If (is_ringbuf == false), then these members are valid.
    long beam_stride32 = 0;
    int amb_stride32 = 0;
    int act_stride32 = 0;
    
    dd_iobuf(const DedispersionKernelParams &params, const Array<void> &arr, bool is_ringbuf_)
    {
	this->is_ringbuf = is_ringbuf_;
	this->buf = arr.data;

	xassert_eq(arr.dtype, params.dtype);

	// Check alignment. Not strictly necessary, but failure would be unintentional and indicate a bug somewhere.
	xassert(is_aligned(buf, constants::bytes_per_gpu_cache_line));   // also checks non_NULL

	// FIXME constructor should include overflow checks on strides.
	// (Check on act_stride is nontrivial, since it gets multiplied by a small integer in the kernel.)
	
	if (is_ringbuf) {
	    // Case 1: ringbuf, with 1-d shape (params.ringbuf_nseg * params.nelts_per_segment,)
	    xassert_shape_eq(arr, ({ params.ringbuf_nseg * params.nelts_per_segment }));
	    xassert(arr.get_ncontig() == 1);
	    xassert(arr.on_gpu());
	}
	else {
	    // Case 2: simple buf, with shape  (params.beams_per_batch, pow2(amb_rank), pow2(dd_rank), ntime).
	    xassert_shape_eq(arr, ({params.beams_per_batch, pow2(params.amb_rank), pow2(params.dd_rank), params.ntime}));
	    xassert(arr.get_ncontig() >= 1);
	    xassert(arr.on_gpu());

	    long denom = xdiv(32, arr.dtype.nbits);
	    this->beam_stride32 = xdiv(arr.strides[0], denom);   // 32-bit stride
	    this->amb_stride32 = xdiv(arr.strides[1], denom);    // 32-bit stride
	    this->act_stride32 = xdiv(arr.strides[2], denom);    // 32-bit stride

	    // Check alignment. Not strictly necessary, but failure would be unintentional and indicate a bug somewhere.
	    xassert_divisible(beam_stride32 * 4, constants::bytes_per_gpu_cache_line);
	    xassert_divisible(amb_stride32 * 4, constants::bytes_per_gpu_cache_line);
	    xassert_divisible(act_stride32 * 4, constants::bytes_per_gpu_cache_line);
	    
	    // FIXME could improve these checks, by verifying that strides are non-overlapping.
	    xassert((params.beams_per_batch == 1) || (beam_stride32 != 0));
	    xassert((params.amb_rank == 0) || (amb_stride32 != 0));
	    xassert((params.dd_rank == 0) || (act_stride32 != 0));
	}
    }
};


NewGpuDedispersionKernel::NewGpuDedispersionKernel(const Params &params_) :
    params(params_)
{
    params.validate(true);        // on_gpu=true
    xassert(params.dd_rank > 0);  // FIXME define _r0 for testing

    RegistryKey key;
    key.dtype = params.dtype;
    key.rank = params.dd_rank;
    key.input_is_ringbuf = params.input_is_ringbuf;
    key.output_is_ringbuf = params.output_is_ringbuf;
    key.apply_input_residual_lags = params.apply_input_residual_lags;

    this->registry_value = query_registry(key);
    this->nbatches = xdiv(params.total_beams, params.beams_per_batch);

    int ST = xdiv(params.dtype.nbits, 8);    
    this->bw_per_launch.kernel_launches = 1;
    this->bw_per_launch.nbytes_gmem += 2 * params.beams_per_batch * pow2(params.dd_rank+params.amb_rank) * params.ntime * ST;
    this->bw_per_launch.nbytes_gmem += 8 * params.beams_per_batch * pow2(params.amb_rank) * registry_value.pstate32_per_small_tree;
    // FIXME(?) not currently including ringbuf_locations.
}


void NewGpuDedispersionKernel::allocate()
{
    if (is_allocated)
	throw runtime_error("double call to NewGpuDedispersionKernel::allocate()");
    
    // Note 'af_zero' flag here.
    long ninner = registry_value.pstate32_per_small_tree * xdiv(32, params.dtype.nbits);
    std::initializer_list<long> shape = { params.total_beams, pow2(params.amb_rank), ninner };
    this->persistent_state = Array<void> (params.dtype, shape, af_zero | af_gpu);

    // Copy host -> GPU.
    if (params.input_is_ringbuf || params.output_is_ringbuf) {
	this->gpu_ringbuf_locations = params.ringbuf_locations.to_gpu();

	long nrb = pow2(params.amb_rank + params.dd_rank) * xdiv(params.ntime, params.nelts_per_segment);
	xassert_shape_eq(gpu_ringbuf_locations, ({nrb,4}));
	xassert(gpu_ringbuf_locations.is_fully_contiguous());
	xassert(gpu_ringbuf_locations.on_gpu());
    }

    this->is_allocated = true;
}


void NewGpuDedispersionKernel::launch(Array<void> &in_arr, Array<void> &out_arr, long ibatch, long it_chunk, cudaStream_t stream)
{
    xassert(this->is_allocated);
    xassert((ibatch >= 0) && (ibatch < nbatches));
    xassert(it_chunk >= 0);

    dd_iobuf in(params, in_arr, params.input_is_ringbuf);
    dd_iobuf out(params, out_arr, params.output_is_ringbuf);

    // The global persistent_state array has shape { total_beams, pow2(params.amb_rank), ninner }.
    // We want to select a subset of beams corresponding to the current batch.
    long b0 = (ibatch) * params.beams_per_batch;
    long b1 = (ibatch+1) * params.beams_per_batch;
    Array<void> pstate = this->persistent_state.slice(0, b0, b1);
    
    // XXX confused by rb_pos
    // long rb_pos = (it_chunk * params.total_beams) + (ibatch * params.beams_per_batch);

    dim3 grid_dims = { uint(pow2(params.amb_rank)), uint(params.beams_per_batch), 1 };
    dim3 block_dims = { 32, uint(registry_value.warps_per_threadblock), 1 };
    ulong nt_cumul = it_chunk * params.ntime;

    // NewGpuDedispersionKernel does not yet support ringbufs
    xassert(!params.input_is_ringbuf);
    xassert(!params.output_is_ringbuf);

    this->registry_value.cuda_kernel <<< grid_dims, block_dims, registry_value.shmem_nbytes, stream >>>
	(in.buf, in.beam_stride32, in.amb_stride32, in.act_stride32,
	 out.buf, out.beam_stride32, out.amb_stride32, out.act_stride32,
	 pstate.data, params.ntime, nt_cumul);
    
    CUDA_PEEK("dedispersion kernel");
}


// -------------------------------------------------------------------------------------------------
//
// Kernel registry.


using DedispRegistry = typename pirate::KernelRegistry<NewGpuDedispersionKernel::RegistryKey, NewGpuDedispersionKernel::RegistryValue>;

// Instead of declaring the registry as a static global variable, we declare it
// as a static local variable in the function dd_registry(). The registry will
// be initialized the first time that dd_registry() is called.
//
// This kludge is necessary because the registry is accessed at library initialization
// time, by callers in other source files, and source files are executed in an
// arbitrary order.

static DedispRegistry &dd_registry()
{
    static DedispRegistry reg;
    return reg;  // note: thread-safe (as of c++11)
}


// Static member function.
NewGpuDedispersionKernel::RegistryValue NewGpuDedispersionKernel::query_registry(const RegistryKey &k)
{
    return dd_registry().query(k);
}

// Static member function.
NewGpuDedispersionKernel::RegistryKey NewGpuDedispersionKernel::get_random_registry_key()
{
    return dd_registry().get_random_key();
}


// Static member function for adding to the registry.
// Called during library initialization, from source files with gpu kernels.
void NewGpuDedispersionKernel::register_kernel(const RegistryKey &key, const RegistryValue &val, bool debug)
{
    // Just check that all members have been initialized.
    // (In the future, I may add more argument checking here.)
    
    xassert((key.dtype == Dtype::native<float>()) || (key.dtype == Dtype::native<__half>()));
    xassert(val.warps_per_threadblock > 0);
    xassert(val.cuda_kernel != nullptr);
    
    if (val.shmem_nbytes > 48*1024) {
        CUDA_CALL(cudaFuncSetAttribute(
	    val.cuda_kernel,
            cudaFuncAttributeMaxDynamicSharedMemorySize,
            val.shmem_nbytes
        ));
    }
    
    return dd_registry().add(key, val, debug);
}


bool operator==(const NewGpuDedispersionKernel::RegistryKey &k1, const NewGpuDedispersionKernel::RegistryKey &k2)
{
    return (k1.dtype == k2.dtype) &&
	(k1.rank == k2.rank) &&
	(k1.input_is_ringbuf == k2.input_is_ringbuf) &&
	(k1.output_is_ringbuf == k2.output_is_ringbuf) &&
	(k1.apply_input_residual_lags == k2.apply_input_residual_lags);
}


ostream &operator<<(ostream &os, const NewGpuDedispersionKernel::RegistryKey &k)
{
    os << "GpuDedispersionKernel(dtype=" << k.dtype
       << ", rank=" << k.rank
       << ", input_is_ringbuf=" << k.input_is_ringbuf
       << ", output_is_ringbuf=" << k.output_is_ringbuf
       << ", apply_input_residual_lags=" << k.apply_input_residual_lags
       << ")";

    return os;
}


}  // namespace pirate
