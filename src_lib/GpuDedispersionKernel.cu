#include "../include/pirate/internals/GpuDedispersionKernel.hpp"
#include "../include/pirate/internals/dedispersion_inbufs.hpp"
#include "../include/pirate/internals/dedispersion_outbufs.hpp"
#include "../include/pirate/internals/inlines.hpp"   // pow2(), is_aligned(), simd_type
#include "../include/pirate/internals/utils.hpp"     // bit_reverse_slow()
#include "../include/pirate/constants.hpp"

#include <sstream>
#include <gputils/cuda_utils.hpp>  // CUDA_CALL()

using namespace std;
using namespace gputils;

namespace pirate {
#if 0
}  // editor auto-indent
#endif


// Defined in dedispersion_kernel_implementation.hpp
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

    assert(rank <= 8);
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
		assert(shmem_nreg < 32768);  // uint15 rb_base
		assert(lag < 256);           // uint8 rb_lag
		
		// Control words are stored in global memory at "writer offset".
		// To get "reader offset", set pos=0 by applying mask 0xff007fff.
		// (See read_control_words() below.)
		
		int s = i * pow2(rank0) + j;
		ret.at({s}) = shmem_nreg | (lag << 15) | (lag << 24);  // control word
	    
		for (int l = 0; l < lag; l++) {
		    if (gs_ireg == 32) {
			// Write completed gmem_spec to 'integer_constants' array.
			assert(gs_icl < gs_ncl);
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

		    assert(gs_spos == spos);
		    gs_ireg++;
		    gs_spos++;
		}

		// Ring buffer size = (lag + 32).
		shmem_nreg += (lag + 32);
	    }
	}
    }
    
    // After loop completes, the last gmem spec should be partially or fully complete.
    assert(gs_ireg > 0);
    assert(gs_ireg <= 32);
    assert(gs_icl == (gs_ncl-1));

    // This assert ensures that we have enough shared memory "headroom".
    int gs_smax = gs_spos + (32 - gs_ireg);
    assert(gs_smax <= shmem_nreg);
    assert(shmem_nreg*4 <= shmem_nbytes);

    // Write last gmem spec.
    ret.at({pow2(rank) + 2*gs_icl}) = gs_sbase;
    ret.at({pow2(rank) + 2*gs_icl+1}) = gs_gbits;

    return on_gpu ? ret.to_gpu() : ret;
}


// -------------------------------------------------------------------------------------------------
//
// Some helper functions which should go away, after I implement gputils::Array<void>.


// Usage: Array<float> arr = uarr_get(x, "x");   // where x is an UntypedArray
template<typename T> Array<T> uarr_get(const UntypedArray &arr, const char *arr_name);

template<> Array<float> uarr_get(const UntypedArray &uarr, const char *arr_name)
{
    bool have_float32 = (uarr.data_float32.data != nullptr);
    bool have_float16 = (uarr.data_float16.data != nullptr);

    if (!have_float32 || have_float16) {
	stringstream ss;
	ss << "GpuDedispersionKernel::launch: expected "
	   << arr_name << ".data_float32 to be nonempty, and "
	   << arr_name << ".data_float16 to be empty";
	throw runtime_error(ss.str());
    }

    return uarr.data_float32;
}

template<> Array<__half> uarr_get(const UntypedArray &uarr, const char *arr_name)
{
    bool have_float32 = (uarr.data_float32.data != nullptr);
    bool have_float16 = (uarr.data_float16.data != nullptr);

    if (have_float32 || !have_float16) {
	stringstream ss;
	ss << "GpuDedispersionKernel::launch: expected "
	   << arr_name << ".data_float32 to be empty, and "
	   << arr_name << ".data_float16 to be nonempty";
	throw runtime_error(ss.str());
    }

    return uarr.data_float16;
}


bool UntypedArray::_is_float32(const char *name) const
{
    bool have_float32 = (this->data_float32.data != nullptr);
    bool have_float16 = (this->data_float16.data != nullptr);

    if (have_float32 && !have_float16)
	return true;
    else if (!have_float32 && have_float16)
	return false;
    else if (have_float32 && have_float16)
	throw runtime_error(string(name) + " has multiple dtypes?!");
    else
	throw runtime_error(string(name) + " is empty or uninitialized");
}


UntypedArray UntypedArray::slice(int axis, int start, int stop) const
{
    UntypedArray ret;

    if (this->_is_float32("UntypedArray::slice() argument"))
	ret.data_float32 = this->data_float32.slice(axis, start, stop);
    else
	ret.data_float16 = this->data_float16.slice(axis, start, stop);

    return ret;
}


void UntypedArray::fill(const UntypedArray &x)
{
    bool dst32 = this->_is_float32("UntypedArray::fill() destination argument");
    bool src32 = x._is_float32("UntypedArray::fill() source argument");

    if (dst32 != src32)
	throw runtime_error("UntypedArray::fill(): source and destination types do not match");
    
    if (dst32)
	this->data_float32.fill(x.data_float32);
    else
	this->data_float16.fill(x.data_float16);
}


bool GpuDedispersionKernel::Params::is_float32() const
{
    // Currently only "float32" and "float16" are allowed.
    if (dtype == "float32")
	return true;
    else if (dtype == "float16")
	return false;
    else if (dtype.empty())
	throw runtime_error("GpuDedispersionKernel::Params::dtype is uninitialized (or empty string)");
    else
	throw runtime_error("GpuDedispersionKernel::Params: unrecognizd dtype '" + dtype + "' (expected 'float32' or 'float16')");
}


void GpuDedispersionKernel::Params::validate(bool on_gpu) const
{
    assert(rank >= 0);
    assert(rank <= 8);
    assert(nambient > 0);
    assert(total_beams > 0);
    assert(beams_per_kernel_launch > 0);
    assert(beams_per_kernel_launch <= constants::cuda_max_y_blocks);
    assert(ntime > 0);

    // Not really necessary, but failure probably indicates an unintentional bug.
    assert(is_power_of_two(nambient));
    
    // Currently assumed throughout the pirate code.
    assert((total_beams % beams_per_kernel_launch) == 0);

    // Currently assumed by the GPU kernels.
    int nelts_per_cache_line = is_float32() ? 32 : 64;
    assert(nelts_per_segment == nelts_per_cache_line);

    assert((ntime % nelts_per_segment) == 0);
    assert(!input_is_ringbuf || !output_is_ringbuf);
    
    if (input_is_ringbuf || output_is_ringbuf) {
	long nseg = xdiv(ntime,nelts_per_segment) * nambient * pow2(rank);
	assert(ringbuf_locations.shape_equals({ nseg, 4 }));
	assert(ringbuf_locations.is_fully_contiguous());
	assert(ringbuf_nseg > 0);
	assert(ringbuf_nseg <= UINT_MAX);

	if (on_gpu) {
	    assert(ringbuf_locations.on_gpu());
	    return;
	}

	assert(ringbuf_locations.on_host());
	
	for (long iseg = 0; iseg < nseg; iseg++) {
	    const uint *rb_locs = ringbuf_locations.data + (4*iseg);
	    long rb_offset = rb_locs[0];  // in segments, not bytes
	    // long rb_phase = rb_locs[1];   // index of (time chunk, beam) pair, relative to current pair
	    long rb_len = rb_locs[2];     // number of (time chunk, beam) pairs in ringbuf (same as Ringbuf::rb_len)
	    long rb_nseg = rb_locs[3];    // number of segments per (time chunk, beam) (same as Ringbuf::nseg_per_beam)
	    assert(rb_offset + (rb_len-1)*rb_nseg < ringbuf_nseg);
	}
    }
}


// -------------------------------------------------------------------------------------------------


template<typename T, bool Lagged>
dedispersion_simple_inbuf<T,Lagged>::device_args::device_args(const UntypedArray &in_uarr, const GpuDedispersionKernel::Params &params)
{
    // If T==float, then T32 is also 'float'.
    // If T==__half, then T32 is '__half2'.
    using T32 = typename simd32_type<T>::type;

    constexpr int elts_per_cache_line = constants::bytes_per_gpu_cache_line / 4;
    constexpr int denom = 4 / sizeof(T);
    static_assert(denom * sizeof(T) == 4);

    Array<T> in_arr = uarr_get<T> (in_uarr, "in");
    
    // Expected shape is (nbeams, nambient, pow2(rank), ntime).
    assert(in_arr.ndim == 4);
    assert(in_arr.shape[0] == params.beams_per_kernel_launch);
    assert(in_arr.shape[1] == params.nambient);
    assert(in_arr.shape[2] == pow2(params.rank));
    assert(in_arr.shape[3] == params.ntime);
    assert(in_arr.get_ncontig() >= 1);
    assert(in_arr.on_gpu());

    this->in = (T32 *) in_arr.data;
    this->beam_stride32 = xdiv(in_arr.strides[0], denom);     // 32-bit stride
    this->ambient_stride32 = xdiv(in_arr.strides[1], denom);  // 32-bit stride
    this->freq_stride32 = xdiv(in_arr.strides[2], denom);     // 32-bit stride
    this->is_downsampled = params.input_is_downsampled_tree;

    // Check alignment. Not strictly necessary, but failure would be unintentional and indicate a bug somewhere.
    assert(is_aligned(in, constants::bytes_per_gpu_cache_line));   // also checks non_NULL
    assert((beam_stride32 % elts_per_cache_line) == 0);
    assert((ambient_stride32 % elts_per_cache_line) == 0);
    assert((freq_stride32 % elts_per_cache_line) == 0);
    
    // FIXME could improve these checks, by verifying that strides are non-overlapping.
    assert(beam_stride32 != 0);
    assert(ambient_stride32 != 0);
    assert(freq_stride32 != 0);
}


template<typename T>
dedispersion_ring_outbuf<T>::device_args::device_args(const UntypedArray &in_uarr, const GpuDedispersionKernel::Params &params)
{
    // If T==float, then T32 is also 'float'.
    // If T==__half, then T32 is '__half2'.
    using T32 = typename simd32_type<T>::type;

    constexpr int denom = 4 / sizeof(T);
    static_assert(denom * sizeof(T) == 4);

    Array<T> in_arr = uarr_get<T> (in_uarr, "in");
    assert(in_arr.ndim == 1);
    assert(in_arr.shape[0] == params.ringbuf_nseg * params.nelts_per_segment);
    assert(in_arr.get_ncontig() == 1);
    assert(in_arr.on_gpu());

    Array<uint> rb_loc = params.ringbuf_locations;
    assert(rb_loc.ndim == 2);
    assert(rb_loc.shape[0] == params.nambient * pow2(params.rank) * xdiv(params.ntime, params.nelts_per_segment));
    assert(rb_loc.shape[1] == 4);
    assert(rb_loc.is_fully_contiguous());
    assert(rb_loc.on_gpu());

    this->rb_base = (T32 *) in_arr.data;
    this->rb_loc = (const uint4 *) rb_loc.data;
}


// -------------------------------------------------------------------------------------------------


// FIXME reduce cut-and-paste between Inbuf::host_args and Outbuf::host_args constructors.
template<typename T>
dedispersion_simple_outbuf<T>::device_args::device_args(const UntypedArray &out_uarr, const GpuDedispersionKernel::Params &params)
{
    // If T==float, then T32 is also 'float'.
    // If T==__half, then T32 is '__half2'.
    using T32 = typename simd32_type<T>::type;

    constexpr int elts_per_cache_line = constants::bytes_per_gpu_cache_line / 4;
    constexpr int denom = 4 / sizeof(T);
    static_assert(denom * sizeof(T) == 4);

    Array<T> out_arr = uarr_get<T> (out_uarr, "in");
    
    // Expected shape is (nbeams, nambient, pow2(rank), ntime)
    assert(out_arr.ndim == 4);
    assert(out_arr.shape[0] == params.beams_per_kernel_launch);
    assert(out_arr.shape[1] == params.nambient);
    assert(out_arr.shape[2] == pow2(params.rank));
    assert(out_arr.shape[3] == params.ntime);
    assert(out_arr.get_ncontig() >= 1);
    assert(out_arr.on_gpu());

    this->out = (T32 *) out_arr.data;
    this->beam_stride32 = xdiv(out_arr.strides[0], denom);     // 32-bit stride
    this->ambient_stride32 = xdiv(out_arr.strides[1], denom);  // 32-bit stride
    this->dm_stride32 = xdiv(out_arr.strides[2], denom);     // 32-bit stride
    
    // Check alignment. Not strictly necessary, but failure would be unintentional and indicate a bug somewhere.
    assert(is_aligned(out, constants::bytes_per_gpu_cache_line));   // also checks non-NULL
    assert((beam_stride32 % elts_per_cache_line) == 0);
    assert((ambient_stride32 % elts_per_cache_line) == 0);
    assert((dm_stride32 % elts_per_cache_line) == 0);
    
    // FIXME could improve these checks, by verifying that strides are non-overlapping.
    assert(beam_stride32 != 0);
    assert(ambient_stride32 != 0);
    assert(dm_stride32 != 0);
}


// -------------------------------------------------------------------------------------------------


template<typename T, class Inbuf, class Outbuf>
struct GpuDedispersionKernelImpl : public GpuDedispersionKernel
{    
    // If T==float, then T32 is also 'float'.
    // If T==__half, then T32 is '__half2'.
    using T32 = typename simd32_type<T>::type;

    GpuDedispersionKernelImpl(const GpuDedispersionKernel::Params &params);

    virtual void launch(const UntypedArray &in, UntypedArray &out, long itime, long ibeam, cudaStream_t stream) override;

    // (inbuf, outbuf, rstate, ntime, integer_constants, rb_pos)
    void (*cuda_kernel)(typename Inbuf::device_args, typename Outbuf::device_args, T32 *, long, uint *, long) = nullptr;
};


template<typename T, class Inbuf, class Outbuf>
GpuDedispersionKernelImpl<T,Inbuf,Outbuf>::GpuDedispersionKernelImpl(const Params &params_) :
    GpuDedispersionKernel(params_)
{
    if (params.rank == 1)
	this->cuda_kernel = dedisperse_r1<T32, Inbuf, Outbuf>;
    else if (params.rank == 2)
	this->cuda_kernel = dedisperse_r2<T32, Inbuf, Outbuf>;
    else if (params.rank == 3)
	this->cuda_kernel = dedisperse_r3<T32, Inbuf, Outbuf>;
    else if (params.rank == 4)
	this->cuda_kernel = dedisperse_r4<T32, Inbuf, Outbuf>;
    else if (params.rank == 5)
	this->cuda_kernel = dedisperse_r5<T32, Inbuf, Outbuf>;
    else if (params.rank == 6)
	this->cuda_kernel = dedisperse_r6<T32, Inbuf, Outbuf>;
    else if (params.rank == 7)
	this->cuda_kernel = dedisperse_r7<T32, Inbuf, Outbuf>;
    else if (params.rank == 8)
	this->cuda_kernel = dedisperse_r8<T32, Inbuf, Outbuf>;
    else
	throw runtime_error("expected 1 <= GpuDedispersionKernel::Params::active_rank <= 8");

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
void GpuDedispersionKernelImpl<T,Inbuf,Outbuf>::launch(const UntypedArray &in_arr, UntypedArray &out_arr, long itime, long ibeam, cudaStream_t stream)
{
    typename Inbuf::device_args in(in_arr, params);
    typename Outbuf::device_args out(out_arr, params);

    // Compare (itime, ibeam) with expected values.
    assert(itime == expected_itime);
    assert(ibeam == expected_ibeam);

    // Update expected (itime, ibeam).
    expected_ibeam += params.beams_per_kernel_launch;
    assert(expected_ibeam <= params.total_beams);
    
    if (expected_ibeam == params.total_beams) {
	expected_ibeam = 0;
	expected_itime++;
    }

    Array<T> rstate = uarr_get<T> (this->persistent_state, "rstate");
    T *rp = rstate.data + (ibeam * this->state_nelts_per_beam);
    long rb_pos = itime * params.total_beams + ibeam;

    // Note: the number of beams and 'ambient' tree channels are implicitly supplied
    // to the kernel via gridDim.y, gridDim.x.
    dim3 grid_dims;
    grid_dims.x = params.nambient;
    grid_dims.y = params.beams_per_kernel_launch;
    grid_dims.z = 1;

    this->cuda_kernel
	<<< grid_dims, 32 * warps_per_threadblock, shmem_nbytes, stream >>>
	(in, out, (T32 *) rp, params.ntime, this->integer_constants.data, rb_pos);
    
    CUDA_PEEK("dedispersion kernel");
}


// -------------------------------------------------------------------------------------------------


GpuDedispersionKernel::GpuDedispersionKernel(const Params &params_) :
    params(params_)
{
    params.validate(true);    // on_gpu=true
    assert(params.rank > 0);  // FIXME define _r0 for testing
    assert(!params.input_is_ringbuf);   // FIXME
	
    // FIXME remaining code is cut-and-paste from previous API -- could use a rethink.

    bool is_float32 = params.is_float32();  // note: error-checks dtype
    int nrs_per_thread;
    
    if (params.rank == 1) {
	this->warps_per_threadblock = 1;
	nrs_per_thread = 1;
    }
    else if (params.rank == 2) {
	this->warps_per_threadblock = 1;
	nrs_per_thread = 1;
    }
    else if (params.rank == 3) {
	this->warps_per_threadblock = 1;
	nrs_per_thread = 1;
    }
    else if (params.rank == 4) {
	this->warps_per_threadblock = 1;
	nrs_per_thread = is_float32 ? 3 : 2;
    }
    else if (params.rank == 5) {
	this->warps_per_threadblock = 4;
	nrs_per_thread = 1;
    }
    else if (params.rank == 6) {
	this->warps_per_threadblock = 8;
	nrs_per_thread = is_float32 ? 2 : 1;
    }
    else if (params.rank == 7) {
	this->warps_per_threadblock = 8;
	nrs_per_thread = is_float32 ? 4 : 3;
    }
    else if (params.rank == 8) {
	this->warps_per_threadblock = 16;
	nrs_per_thread = is_float32 ? 5 : 4;
    }
    else
	throw runtime_error("GpuDedispersionKernel constructor: should never get here");
    
    long swflag = (warps_per_threadblock == 1);
    long rp_ncl = params.apply_input_residual_lags ? (pow2(params.rank) - swflag) : 0;
    long rs_ncl = warps_per_threadblock * nrs_per_thread;
    long gs_ncl = get_gs_ncl(params.rank, is_float32);
    long nelts_per_small_tree = (rs_ncl + rp_ncl + gs_ncl) * (is_float32 ? 32 : 64);
    this->state_nelts_per_beam = params.nambient * nelts_per_small_tree;
    
    if (gs_ncl > 0)
	this->shmem_nbytes = 128 * (gs_ncl + pow2(params.rank));

    if (is_float32)
	this->persistent_state.data_float32 = Array<float> ({params.total_beams, state_nelts_per_beam}, af_zero | af_gpu);    
    else
	this->persistent_state.data_float16 = Array<__half> ({params.total_beams, state_nelts_per_beam}, af_zero | af_gpu);
    
    this->integer_constants = make_integer_constants(params.rank, is_float32, true);   // on_gpu=true
}


// Static member function
shared_ptr<GpuDedispersionKernel> GpuDedispersionKernel::make(const Params &params)
{
    bool rb_in = params.input_is_ringbuf;
    bool rb_out = params.output_is_ringbuf;
    bool rlag = params.apply_input_residual_lags;
    bool is_float32 = params.is_float32();

    // Select subclass template instantiation.
    // Currently 6 cases here -- more to come.

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
    
    throw runtime_error("GpuDedispersionKernel::make(): no suitable precompiled kernel could be found");
}


}  // namespace pirate
