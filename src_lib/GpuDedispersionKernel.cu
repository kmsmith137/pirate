#include "../include/pirate/internals/GpuDedispersionKernel.hpp"
#include "../include/pirate/internals/inlines.hpp"   // pow2()
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
//   uint9  rb_pos;    // current position, satisfying 0 <= rb_pos < (rb_lag + 32)
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


template<typename T>
void GpuDedispersionKernel<T>::launch(T *iobuf, T *rstate,
				      long nbeams, long beam_stride,
				      long nambient, long ambient_stride,
				      long row_stride, long ntime,   // number of rows is always 2^rank
				      cudaStream_t stream) const
{
    constexpr int elts_per_cache_line = constants::bytes_per_gpu_cache_line / sizeof(T);

    // Check alignment (also checks that 'iobuf' and 'rstate' are non-NULL)
    // Most of these are not strictly necessary, but failure would be unintentional and indicate a bug somewhere.
    
    assert(is_aligned(iobuf, constants::bytes_per_gpu_cache_line));
    assert(is_aligned(rstate, constants::bytes_per_gpu_cache_line));
    assert((beam_stride % elts_per_cache_line) == 0);
    assert((ambient_stride % elts_per_cache_line) == 0);
    assert((row_stride % elts_per_cache_line) == 0);
    assert((ntime % elts_per_cache_line) == 0);

    assert(ntime > 0);
    assert(nbeams > 0);
    assert(nambient > 0);
    assert(is_power_of_two(nambient));

    // Currently we only support two-stage dedispersion, where each stage has rank <= 8.
    // Therefore, we expect nambient <= 2^8. (The kernel can handle larger values, but larger values
    // would be unintentional and indicate a bug somewhere.)
    assert(nambient <= 256);

    // Required by CUDA (max allowed value of gridDims.y)
    assert(nbeams < 65536);
    
    // FIXME could improve these checks, by verifying that strides are non-overlapping.
    assert(beam_stride != 0);
    assert(ambient_stride != 0);
    assert(row_stride != 0);

    // Overflow checking.

    long max_offset = pow2(params.rank) * abs(row_stride) / 2;

    if (max_offset >= (1L << 31))
	throw runtime_error("row_stride 32-bit overflow");
    if (ntime >= (1L << 31))
	throw runtime_error("ntime 32-bit overflow");
    
    T32 *iobuf2 = reinterpret_cast<T32 *> (iobuf);
    T32 *rstate2 = reinterpret_cast<T32 *> (rstate);
    long nt_cl = ntime / elts_per_cache_line;
    bool flags = params.is_downsampled_tree ? 1 : 0;

    // Convert (T strides) to (T32 strides).
    int s = integer_log2(4 / sizeof(T));
    long beam_stride2 = beam_stride >> s;
    long ambient_stride2 = ambient_stride >> s;
    long row_stride2 = row_stride >> s;
    
    dim3 grid_dims;
    grid_dims.x = nambient;
    grid_dims.y = nbeams;
    grid_dims.z = 1;

    this->kernel
	<<< grid_dims, 32 * params.warps_per_threadblock, params.shmem_nbytes, stream >>>
        (iobuf2, rstate2, beam_stride2, ambient_stride2, row_stride2, nt_cl, this->integer_constants.data, flags);
    
    CUDA_PEEK("dedispersion kernel");
}


template<typename T>
void GpuDedispersionKernel<T>::launch(Array<T> &iobuf, Array<T> &rstate, cudaStream_t stream) const
{
    if (!iobuf.on_gpu())
	throw runtime_error("GpuDedispersionKernel::launch_kernek(): iobuf array must be on GPU");
    if (!rstate.on_gpu())
	throw runtime_error("GpuDedispersionKernel::launch(): rstate array must be on GPU");
    
    // The 'iobuf' amd 'rstate' arrays must have shapes
    //   iobuf.shape = { nbeams, nambient, 2^rank, ntime }   // (beam, ambient, row, time)
    //   rstate.shape = { nbeams, nambient, state_nelts_per_small_tree }

    assert(iobuf.ndim == 4);
    assert(rstate.ndim == 3);
    assert(iobuf.shape[2] == pow2(params.rank));
    assert(rstate.shape[2] == params.state_nelts_per_small_tree);
    assert(iobuf.shape[0] == rstate.shape[0]);
    assert(iobuf.shape[1] == rstate.shape[1]);

    assert(iobuf.get_ncontig() >= 1);
    assert(rstate.is_fully_contiguous());

    // Hand off to "bare pointer" version of launch(), which will do more shape/stride
    // checking, and then launch the cuda kernel.
    
    this->launch(iobuf.data,
		 rstate.data,
		 iobuf.shape[0],    // nbeams
		 iobuf.strides[0],  // beam_stride
		 iobuf.shape[1],    // nambient
		 iobuf.strides[1],  // ambient_stride
		 iobuf.strides[2],  // row_stride,
		 iobuf.shape[3],    // ntime
		 stream);
}
    

// -------------------------------------------------------------------------------------------------
//
// Protected constructor and public factory function.


template<typename T>
GpuDedispersionKernel<T>::GpuDedispersionKernel(const Params &params_, kernel_t kernel_,
						const Array<uint> &integer_constants_) :
    params(params_), kernel(kernel_), integer_constants(integer_constants_)
{
    assert(params.rank > 0);
    assert(params.warps_per_threadblock > 0);
    assert(params.state_nelts_per_small_tree > 0);
    assert(kernel != nullptr);

    if (params.shmem_nbytes > 48*1024) {
	// FIXME: I'm asusming here that cudaFuncSetAttribute() is thread-safe.
	// Should try to confirm this somehow!	
	CUDA_CALL(cudaFuncSetAttribute(
	    kernel,
	    cudaFuncAttributeMaxDynamicSharedMemorySize,
	    params.shmem_nbytes
	));
    }	
}


// Static member function.
template<typename T>
shared_ptr<GpuDedispersionKernel<T>> GpuDedispersionKernel<T>::make(int rank, bool apply_input_residual_lags, bool is_downsampled_tree)
{
    constexpr int is_float32 = _is_float32<T>::value;
    
    Params params;
    params.rank = rank;
    params.apply_input_residual_lags = apply_input_residual_lags;
    params.is_downsampled_tree = is_downsampled_tree;

    int nrs_per_thread = 0;
    kernel_t kernel = nullptr;

    // Remaining code should initialize:
    //   kernel
    //   params.warps_per_threadblock
    //   nrs_per_thread
    
    if (rank == 1) {
	kernel = apply_input_residual_lags ? dedisperse_r1<T32,true> : dedisperse_r1<T32,false>;
	params.warps_per_threadblock = 1;
	nrs_per_thread = 1;
    }
    else if (rank == 2) {
	kernel = apply_input_residual_lags ? dedisperse_r2<T32,true> : dedisperse_r2<T32,false>;
	params.warps_per_threadblock = 1;
	nrs_per_thread = 1;
    }
    else if (rank == 3) {
	kernel = apply_input_residual_lags ? dedisperse_r3<T32,true> : dedisperse_r3<T32,false>;
	params.warps_per_threadblock = 1;
	nrs_per_thread = 1;
    }
    else if (rank == 4) {
	kernel = apply_input_residual_lags ? dedisperse_r4<T32,true> : dedisperse_r4<T32,false>;
	params.warps_per_threadblock = 1;
	nrs_per_thread = is_float32 ? 3 : 2;
    }
    else if (rank == 5) {
	kernel = apply_input_residual_lags ? dedisperse_r5<T32,true> : dedisperse_r5<T32,false>;
	params.warps_per_threadblock = 4;
	nrs_per_thread = 1;
    }
    else if (rank == 6) {
	kernel = apply_input_residual_lags ? dedisperse_r6<T32,true> : dedisperse_r6<T32,false>;
	params.warps_per_threadblock = 8;
	nrs_per_thread = is_float32 ? 2 : 1;
    }
    else if (rank == 7) {
	kernel = apply_input_residual_lags ? dedisperse_r7<T32,true> : dedisperse_r7<T32,false>;
	params.warps_per_threadblock = 8;
	nrs_per_thread = is_float32 ? 4 : 3;
    }
    else if (rank == 8) {
	kernel = apply_input_residual_lags ? dedisperse_r8<T32,true> : dedisperse_r8<T32,false>;
	params.warps_per_threadblock = 16;
	nrs_per_thread = is_float32 ? 5 : 4;
    }
    else {
	stringstream ss;
	ss << "GpuDedispersionKernel::make(): rank=" << rank << " is not implemented";
	throw runtime_error(ss.str());
    }

    assert(kernel != nullptr);
    assert(nrs_per_thread > 0);
    assert(params.warps_per_threadblock > 0);

    int swflag = (params.warps_per_threadblock == 1);
    int rp_ncl = apply_input_residual_lags ? (pow2(rank) - swflag) : 0;
    int rs_ncl = params.warps_per_threadblock * nrs_per_thread;
    int gs_ncl = get_gs_ncl(rank, is_float32);

    params.state_nelts_per_small_tree = (rs_ncl + rp_ncl + gs_ncl) * (128/sizeof(T));
    
    if (gs_ncl > 0)
	params.shmem_nbytes = 128 * (gs_ncl + pow2(rank));

    Array<uint> integer_constants = make_integer_constants(rank, is_float32, true);   // on_gpu=true
	
    auto kp = new GpuDedispersionKernel(params, kernel, integer_constants);
    return shared_ptr<GpuDedispersionKernel> (kp);
}


template<typename T>
void GpuDedispersionKernel<T>::print(ostream &os, int indent) const
{
    os << Indent(indent) << "GpuDedispersionKernel<" << gputils::type_name<T>() << ">\n"
       << Indent(indent+4) << "rank = " << params.rank << "\n"
       << Indent(indent+4) << "apply_input_residual_lags = " << (params.apply_input_residual_lags ? "true" : "false") << "\n"
       << Indent(indent+4) << "is_downsampled_tree = " << (params.is_downsampled_tree ? "true" : "false") << "\n"
       << Indent(indent+4) << "state_nelts_per_small_tree = " << params.state_nelts_per_small_tree << "\n"
       << Indent(indent+4) << "warps_per_threadblock = " << params.warps_per_threadblock << "\n"
       << Indent(indent+4) << "shmem_nbytes = " << params.shmem_nbytes
       << endl;
}


#define INSTANTIATE(T) \
    template void GpuDedispersionKernel<T>::launch(T*, T*, long, long, long, long, long, long, cudaStream_t) const; \
    template void GpuDedispersionKernel<T>::launch(Array<T> &, Array<T> &, cudaStream_t) const; \
    template GpuDedispersionKernel<T>::GpuDedispersionKernel(const Params &, kernel_t, const Array<uint> &); \
    template shared_ptr<GpuDedispersionKernel<T>> GpuDedispersionKernel<T>::make(int, bool, bool); \
    template void GpuDedispersionKernel<T>::print(ostream &os, int indent) const

INSTANTIATE(__half);
INSTANTIATE(float);


}  // namespace pirate
