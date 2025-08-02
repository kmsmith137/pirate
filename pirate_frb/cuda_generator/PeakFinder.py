import itertools

from . import utils
from .Ringbuf import Ringbuf, Ringbuf16

# The peak-finding kernel is not in its "final form" but here's what I currently have!
#
# Input compile-time params:
#
#    M = DM coarse-graining factor (see "note on M" below)
#    E = max kernel width (power of two)
#    W = warps per threadblock (currently 4, may be different in future coalesced kernel)
#    Dout = time coarse-graining factor (power of two)
#    Dcore = internal time downsampling factor (power of two and <= Dout, see below)
#
# Note on M: currently we don't implement sub-band searches, so M is just the DM
# coarse-graining factor. In the future, M will be the number of (sub-band, trial DM)
# pairs per coarse-grained DM (which could be of order 50-100). In this case, M may
# not be a power of two, so the low-level cuda kernel does not assume that.
#
# The Dcore param has the following meaning:
#
#   - Dcore can be any power of two which is <= Dout.
#   - Slightly changes behavior of kernel: profiles at downsampling level 'ids' are
#      computed at sampling rate min(2**ids, Dcore).
#   - Tradeoff: larger Dcore means less compute but more register usage.
#   - We usually use Dcore = min(Dout,8).
#
# Derived compile-time params:
#
#    P = number of peak-finding kernels = 3*log2(E) + 1
#    P32 = number of 32-bit "persistent state" registers per (beam, coarse_grained_dm).
#
# CUDA kernel:
#
#    __global__ void pf_kernel(out_max, out_ssq, pstate, in, wt, Mout, Tout32);
#
#    - 'out_max' and 'out_ssq' have shape (B, P, Mout, Tout32).
#    - 'pstate' has shape (B, Mout, P32), where P32 is defined below
#    - 'in' has shape (B, Mout*M, Tout32*Dout).
#    - 'wt' has shape (B, P, Mout*M).
#    - Mout is number of output (coarse-grained) DMs.
#    - Tout32 is number of output (coarse-grained) times (divided by two if float16).
#
# This API assumes:
#
#    - Weights are independent of time (at least on the timescale of one kernel launch)
#    - Output arrays is coarse-grained over m, but not coarse-grained over profile 0 <= p < P.
#    - Caller needs both the 'out_max' and 'out_ssq' arrays.
#    - No "argmax" info.
#
# I may rethink these decisions! In particular, the decision to not coarse-grain over
# 0 <= p < P means that the peak-finding kernel uses a ton of registers (probably
# impractical for coalesced kernel).
#
# Warp/block assignment:
#
#  - Each warp processes one "mout" (coarse-grained DM).
#  - 4 warps per threadblock (corresponding to different "mout"s in same beam).
#  - Threadblock grid has shape ((Mout+3)//4, nbeams, 1).
#  - Note that number of beams (per kernel launch) is implicitly supplied as gridDim.x.
#
# Kernel logic:
#
#   - Copy weights global memory -> shared
#   - Apply per-thread pointer offsets to (out_max, out_ssq, in)
#   - Load persistent state (ring buffers)
#
#   - Outer loop over 32 input times (or 64 input times, if float16)
#
#        - Input register assignment:
#            if float32:  th0 th1 th2 th3 th4 <-> ti0 ti1 ti2 ti3 ti4   (registers <-> m0 m1 ...)
#            if float16:  th0 th1 th2 th3 th4 <-> ti1 ti2 ti3 ti4 ti5   b <-> ti0   (registers <-> m0 m1 ...)
#
#        - First step is to transpose inputs to the following register assignment (where L = log2(Dcore)):
#
#            if float32:  th0 th1 th2 th3 th4 <-> m0 ... m_{L-1} ti_L .. ti4
#                         r0 ... r_{L-1} <-> ti_0 ... ti_{L-1}
#
#            if float16:  th0 th1 th2 th3 th4 <-> m0 ... m_{L-1} ti_L .. ti4
#                         r0 ... r_{L-1} <-> ti_0 ... ti_{L-1}
#                         b <-> ti_5
#
#          For float16, this register assignment may not be the most efficient, but it allows
#          the core peak-finding logic to be similar for float16 / float32. I may experiment with
#          a different register assignment later.
#
#          This transposing is implemented "one bit at a time" in 'class PfTransposeLayer'.
#          For float16 there is an additional step in 'class PfInitialTranspose16Layer'.
#
#        - Then we compute peak-finding kernels for 0 <= p < P locally on each thread (in 'class PfCore').
#          Kernel outputs are reduced "locally" over fine-grained time. Weights are not applied in this stage.
#
#          We use an in-register ring buffer ('class Ringbuf' or 'class Ringbuf16') to handle overlaps.
#          This is the only part of the peak-finding kernel that uses persistent state in GPU memory.
#
#          The output of 'class PfCore' is two registers 'max' and 'ssq' with the following register assignments:
#
#            if float32:  th0 th1 th2 th3 th4 <-> m0 ... m_{L-1} ti_L .. ti4
#            if float16:  th0 th1 th2 th3 th4 <-> m0 ... m_{L-1} ti_L .. ti4    b <-> ti5
#
#       - These registers are handed off to 'class PfReducer', which incrementally transposes and reduces,
#         to end up with this "outer" register assignment for 'max' and 'ssq':
#
#            if float32:  th0 th1 th2 th3 th4 <-> ti_K ... ti_{K+4}    where K=log2(Dout)
#            if float16:  th0 th1 th2 th3 th4 <-> ti_{K+1} ... ti_5    b <-> ti_K
#
#         Weights are applied in this stage.
# 
#         In the current implementation, there is one pair of "outer" registers for each 0 <= p < P.
#         When these registers have been fully populated (with either 32 or 64 coarse-grained times
#         for each p), then they are written to global memory.
#
# This chain of operations is implemented by the following code generation classes:
#
#    - PeakFinder: top level class
#        - PfInitialTranspose16Layer         #  only if float16
#        - PfTransposeLayer(Din=1)
#        - PfTransposeLayer(Din=2)
#        - PfTransposeLayer(Din=4)
#              ...
#        - PfTransposeLayer(Din=Dcore//2)    # total number of PfTransposeLayers is log2(Dcore)
#        - PfCore
#            - Ringbuf or Ringbuf16          # used to manage overlaps
#        - PfReducer                         # writes to global memory
#
# For a precise specification of what the peak-finding kernel computes, see the reference
# implementation (class ReferencePeakFindingKernel).


class PeakFindingParams:
    def __init__(self, dtype, M, E, Dout, Dcore, W, BlocksPerSM):
        """
        Input compile-time params:
        
            M = DM coarse-graining factor (see "note on M" below)
            E = max kernel width (power of two)
            W = warps per threadblock (currently 4, may be different in future coalesced kernel)
            Dout = time coarse-graining factor (power of two)
            Dcore = internal time downsampling factor (power of two and <= Dout, see below)
            BlocksPerSM = only used for __launch_bounds__
        """
        
        # FIXME incomplete
        assert M >= 1
        assert E <= 32
        assert Dout <= 32
        assert 1 <= (W * BlocksPerSM) <= 32
        assert Dcore <= Dout
        
        assert utils.is_power_of_two(E)
        assert utils.is_power_of_two(Dout)
        assert utils.is_power_of_two(Dcore)
                
        self.M = M
        self.E = E
        self.W = W
        self.Dout = Dout
        self.Dcore = Dcore
        self.BlocksPerSM = BlocksPerSM
        self.dtype = dtype

        if dtype == 'float':
            self.dt32 = 'float'
            self.dt32_scalar = ''
            self.dt32_max = 'max'
            self.dtstr = 'fp32'
            self.nbits = 32
        elif dtype == '__half':
            self.dt32 = '__half2'
            self.dt32_scalar = '__float2half2_rn'
            self.dt32_max = '__hmax2'
            self.dtstr = 'fp16'
            self.nbits = 16
        else:
            raise RuntimeError(f"Unrecognized dtype '{dtype}")

        # Number of peak-finding profiles.
        self.P = 3 * utils.integer_log2(E) + 1

        # Variables used internally by kernel.
        # To override these defaults, just assign new values after construction.
        self.wt_glo_rname = 'wt'        # global memory pointer, shape (B, P, Mout*M)
        self.out_max_rname = 'out_max'  # global memory pointer, shape (B, P, Mout, Tout32)
        self.out_ssq_rname = 'out_ssq'  # global memory pointer, shape (B, P, Mout, Tout32)
        self.tout32_rname = 'tout32'        # advances by (32/Dout) in each loop iteration
        self.Tout32_rname = 'Tout32'
        self.Mout_rname = 'Mout'


class PeakFinder:
    def __init__(self, params, reduce_only=False):
        """
        The 'Dcore' argument is an internal downsampling factor.
        Large Dcore uses fewer instructions, but more registers.
        """

        assert isinstance(params, PeakFindingParams)
        rostr = f'reduce_only_' if reduce_only else ''
        
        self.params = params
        self.reduce_only = reduce_only
        self.kernel_name = f'pf_{params.dtstr}_{rostr}kernel_M{params.M}_E{params.E}_Dcore{params.Dcore}_Dout{params.Dout}_W{params.W}_B{params.BlocksPerSM}'
        self.reducer = PfReducer(params)

        if reduce_only:
            return

        self.ringbuf = Ringbuf16() if (params.dtype == '__half') else Ringbuf()        
        self.pf = PfCore(self.reducer, self.ringbuf)
        
        while self.pf.Din > 1:
            self.pf = PfTransposeLayer(self.pf)

        if params.dtype == '__half':
            self.pf = PfInitialTranspose16Layer(self.pf)
        
        
    def emit_global(self, k):
        """Emits __global__ kernel."""
        
        pars = self.params
        dt32, nbits, M, E, Dout, Dcore, W, P = pars.dt32, pars.nbits, pars.M, pars.E, pars.Dout, pars.Dcore, pars.W, pars.P

        if not self.reduce_only:
            # Main case: full peak-finding kernel.
            in_args = ('pstate', 'in')
            in_line1 = "'pstate' has shape (B, Mout, P32), where P32 is defined below"
            in_line2 = "'in' has shape (B, Mout*M, Tout32*Dout)"
        else:
            # Debug case: reduce-only kernel (with different kernel args).
            in_args = ('in_max', 'in_ssq')
            in_line1 = "'in_max' has shape (B, P, Mout*M, Tout*(Dout/Dcore))"
            in_line2 = "'in_ssq' has shape (B, P, Mout*M, Tout*(Dout/Dcore))"

        k.reset_tmp_vars()
        
        k.emit(f'// Kernel args are (out_max, out_ssq, {in_args[0]}, {in_args[1]}, wt, Mout, Tout32).')
        k.emit(f'//')
        k.emit(f"//    - 'out_max' and 'out_ssq' have shape (B, P, Mout, Tout32).")
        k.emit(f"//    - {in_line1}.")
        k.emit(f"//    - {in_line2}.")
        k.emit(f"//    - 'wt' has shape (B, P, Mout*M).")
        k.emit("//")
        k.emit('// Each warp processes one output 0 <= mout < Mout, and all times 0 <= tout32 < Tout32.')
        k.emit('//')
        k.emit('// Launch with {BM,B,1} blocks, where BM = ceil(Mout/W), and B is number of beams.')
        k.emit('// Launch with {32*W,1} threads.')
        k.emit()
        k.emit(f'__global__ void __launch_bounds__({32*W}, {pars.BlocksPerSM})')
        k.emit(f'{self.kernel_name}(void *out_max_, void *out_ssq_, void *{in_args[0]}_, void *{in_args[1]}_, void *wt_, int Mout, int Tout32)')
        k.emit('{')

        s = '' if self.reduce_only else '// '
        k.emit(f'constexpr int {M = };      // number of "rows" (e.g. trial DMs) per warp')
        k.emit(f'constexpr int {W = };      // warps per threadblock')
        k.emit(f'constexpr int {P = };      // number of peak-finding kernels')
        k.emit(f'constexpr int {Dout = };   // output time downsampling factor')
        k.emit(f'{s}constexpr int {Dcore = };  // internal resolution parameter')
        k.emit(f'// constexpr int {E = };      // max kernel width')
        k.emit()

        # Note that in the non-reduce-only kernel, we postpone declaring "float *pstate"
        # until the persistent state is used. This avoids a compiler warning about an
        # unused variable.

        k.emit(f"{dt32} *out_max = ({dt32} *) out_max_;")
        k.emit(f"{dt32} *out_ssq = ({dt32} *) out_ssq_;")
        k.emit(f"{s}{dt32} *{in_args[0]} = ({dt32} *) {in_args[0]}_;")  # see above
        k.emit(f"{dt32} *{in_args[1]} = ({dt32} *) {in_args[1]}_;")
        k.emit(f"{dt32} *wt = ({dt32} *) wt_;")            
        k.emit()
        
        divstr = f' / {32//nbits}' if (nbits < 32) else ''
        k.emit(f"int b = blockIdx.y;  // beam index")
        k.emit(f"int mout_block = blockIdx.x * W;                    // base input m-index of block")
        k.emit(f"int mout_warp = mout_block + (threadIdx.x >> 5);    // output m-index of warp")
        k.emit()
        k.emit(f"bool warp_active = (mout_warp < Mout);")
        k.emit(f"mout_warp = warp_active ? mout_warp : (Mout-1);")
        k.emit()

        shmem_n32 = (P*W*M*nbits) // 32
        k.emit('// Shared memory layout: wt[P][W*M]')
        k.emit(f'// nelts={(P*W*M)=}, nelts32={shmem_n32} nbytes={4*shmem_n32}')
        k.emit(f'__shared__ {dt32} shmem[{shmem_n32}];')
        k.emit()

        k.emit("// Apply per-beam offset to 'wt', in preparation for loading weights.")
        k.emit("// Shape is (B, P, Mout*M).")
        k.emit()
        k.emit(f"wt += b * (P * Mout * M){divstr};")
        k.emit()
        
        k.emit("// Copy weights from global memory to shared.")
        k.emit("// The destination array has shape (P, W*M), contiguously packed.")
        k.emit("// The source array is a discontiguous subset of a shape (P, Mout*M) source array.")
        k.emit("// FIXME this loop could be optimized a little.")
        k.emit()
        k.emit(f'constexpr int ndst = (W * M){divstr};     // dst array shape is (P, ndst)')
        k.emit(f'const int nsrc = (Mout * M){divstr};      // src array shape is (P, nsrc)')
        k.emit(f'const int j0 = (mout_block * M){divstr};  // base offset in source array')
        k.emit('')
        k.emit("for (int i = threadIdx.x; i < P * ndst; i += 32*W) {")
        k.emit("    int p = i / ndst;")
        k.emit("    int j = i - p*ndst + j0;")
        k.emit("    j = min(j, nsrc-1);")
        k.emit("    shmem[i] = wt[p*nsrc + j];")
        k.emit("}")
        k.emit()
        k.emit('__syncthreads();')
        k.emit()
        
        k.emit("// Apply per-thread offset, to 'out_max' and 'out_ssq'.")
        k.emit("// Shapes are (B, P, Mout, Tout32), and pstride is Mout*Tout32.")
        k.emit("// This leaves an offset 0 <= p < P to be applied later (with stride Mout*Tout32).")
        k.emit()
        k.emit("int out_offset = (b * P * Mout + mout_warp) * Tout32 + (threadIdx.x & 0x1f);")
        k.emit("out_max += out_offset;")
        k.emit("out_ssq += out_offset;")
        k.emit()

        if not self.reduce_only:
            # Main case: full peak-finding kernel.
            k.emit("// Apply per-thread offsets to 'in'.")
            k.emit("// Shape is (B, Mout*M, Tout32*Dout)")
            k.emit()
            k.emit("int in_mstride = Tout32 * Dout;")
            k.emit(f"in += (b*Mout + mout_warp) * M * in_mstride + (threadIdx.x & 0x1f);")
            k.emit()

            # Save splice, for code to load the ring buffer.
            # (This code must be emitted after the main lloop.)
            k_rb = k.splice()

        else:
            # Debug case: reduce-only kernel (with different kernel args).
            k.emit("// The arrays 'in_max' and 'in_ssq' have shape (B, P, Mout*M, Tout*(Dout/Dcore)).")
            k.emit("// These arrays will be read into the following register assignment")
            k.emit("//   th0 th1 ... th4 <-> m_0 .. m_{K-1} ti_0 .. ti_{4-K}    where K = log2(Dcore)")
            k.emit("//")
            k.emit("// In float16, the simd bit is assigned as: b <-> ti_{5-K}")
            k.emit()
            k.emit("// Apply per-warp offsets, and per-thread time offset, to 'in_max' and 'in_ssq'.")
            k.emit("// Shape in GPU global memory is (B, P, Mout*M, Tout32*(Dout/Dcore)).")
            k.emit("// This leaves a per-thread offset 0 <= m < M to be applied later (with stride 'in_mstride').")
            k.emit("// This leaves an offset 0 <= p < P to be applied later (with stride 'in_pstride').")
            k.emit("// In each iteration of the t-loop, these pointers advance by (32/Dcore), not (32/Dout).")
            k.emit()
            k.emit(f"int in_mstride = Tout32 * (Dout/Dcore);")
            k.emit(f"int in_pstride = Mout * M * in_mstride;")
            k.emit(f"int in_bm_offset = (b * P * in_pstride) + (mout_warp * M * in_mstride);    // per-warp (b,m) offset")
            k.emit(f"int in_t_offset = ((threadIdx.x & 0x1f) / Dcore);")
            
            if pars.dtype == '__half':
                k.emit('int in_cw = (in_t_offset & 1) ? 0x7632 : 0x5410;   // control word for selecting __half from __half2')
                k.emit('in_t_offset >>= 1;')
                
            k.emit(f"in_max += (in_bm_offset + in_t_offset);")
            k.emit(f"in_ssq += (in_bm_offset + in_t_offset);")
            k.emit()
            k.emit(f'const {pars.dt32} large = {pars.dt32_scalar}(1000.0f);')
            k.emit()
                
        self.reducer.initialize(k)

        k.emit(f'// PeakFinder: mouter t-loop')
        k.emit(f"for (int tout32 = 0; tout32 < Tout32; tout32 += (32/Dout)) {{")
        k.emit("    if (!warp_active)")
        k.emit("        continue;")
        k.emit()

        if not self.reduce_only:
            # Main case: full peak-finding kernel.
            # To emit all code inside the outer t-loop, we call:
            #
            #   self.pf.advance(m=0)
            #   self.pf.advance(m=1)
            #     ...
            #   self.pf.advance(m = M-1)
            
            for m in range(M):
                rname = f'x{m}'
                s = f'{m} * in_mstride' if (m > 0) else '0'
                k.emit(f'// PeakFinder: {m=}\n')
                k.emit(f"{dt32} {rname} = in[{s}];")
                self.pf.advance(k, [rname], m)
                k.emit()

            k.emit("// Advance 'in' by 32 time samples")
            k.emit("in += 32;")

        else:
            # Debug case: reduce-only kernel (with different kernel args).
            # To emit all code inside the outer t-loop, we do (schematically):
            #
            #   for m0 in range(0,M,Dcore):
            #     for p in range(P):
            #         # read in_max, in_ssq
            #         self.reducer.advance(m0, p)

            for m0 in range(0,M,Dcore):
                k.emit(f'// PeakFinder: {m0=}')
                prefix = 'int ' if (m0 == 0) else ''
                suffix = f' + {m0}' if (m0 > 0) else ''
                k.emit(f'{prefix}pf_m = (threadIdx.x & (Dcore-1)){suffix};   // {m0=}')
                k.emit()

                for p in range(P):
                    k.emit(f'// PeakFinder: {m0=}, {p=}')
                    addr = f'({p} * in_pstride) + (pf_m * in_mstride)'
                    suff = f'_m{m0}_p{p}'
                    
                    for stem in [ 'max', 'ssq' ]:
                        k.emit(f'{dt32} in_{stem}{suff} = (pf_m < M) ? in_{stem}[{addr}] : large;')
                        if pars.dtype == '__half':
                            k.emit(f'{dt32} in2_{stem}{suff} = (pf_m < M) ? in_{stem}[{addr} + (16/Dcore)] : large;')
                            k.emit(f'in_{stem}{suff} = ksgpu::f16_perm(in_{stem}{suff}, in2_{stem}{suff}, in_cw);')
                    
                    self.reducer.advance(k, f'in_max{suff}', f'in_ssq{suff}', m0, p)  # string args are (max_rname, ssq_rname)
                
            k.emit("// Advance 'in_max' and 'in_ssq' by (32/Dcore) 32-bit registers")
            k.emit("in_max += (32/Dcore);")
            k.emit("in_ssq += (32/Dcore);")
            

        if not self.reduce_only:
            k.emit()
            self.ringbuf.advance_outer(k)
            self.P32 = self.ringbuf.get_n32_per_warp()
        
        k.emit("}  // end of t-loop")
        k.emit()

        if not self.reduce_only:
            self._load_ringbuf(k_rb)
            self._save_ringbuf(k)
        
        k.emit("}  // end of cuda kernel")


    def _load_ringbuf(self, k):
        P32 = self.ringbuf.get_n32_per_warp()

        if P32 == 0:
            k.emit("// Note: 'pstate' is not needed in this kernel")
            k.emit()
            return
        
        k.emit("// Apply per-warp offset to pstate, in preparation for loading")
        k.emit("// Shape is (B, Mout, P32).")
        k.emit()
        k.emit(f'constexpr int {P32 = };    // ring buffer 32-bit elements per warp')
        k.emit(f'{self.params.dt32} *pstate = ({self.params.dt32} *)pstate_;')
        k.emit("pstate += (b*Mout + mout_warp) * P32;")
        k.emit()

        k.emit('// Read ring buffer directly from global memory.')
        k.emit('// FIXME: would it be better to go through shared memory?')
        
        self.ringbuf.initialize(k, 'pstate')
        k.emit()

        
    def _save_ringbuf(self, k):
        P32 = self.ringbuf.get_n32_per_warp()
        
        if P32 == 0:
            k.emit("// Reminder: 'pstate' is not needed in this kernel")
            return
        
        k.emit('// Write ring buffer to global memory.')
        k.emit('// FIXME: would it be better to go through shared memory?')
        k.emit()

        k.emit('if (warp_active) {')
        self.ringbuf.finalize(k, 'pstate')
        k.emit('}')


####################################################################################################


class PfTransposeLayer:
    def __init__(self, next_layer):
        """
        A PfTransposeLayer converts this register assignment (where L = log2(Din)):

             th0 th1 th2 th3 th4 <-> m0 ... m_{L-1} ti_L .. ti4      r <-> ti_0 ... ti_{L-1} m_L

        To this register assignment:

             th0 th1 th2 th3 th4 <-> m0 ... m_L ti_{L+1} .. ti4      r <-> ti_0 ... ti_{L-1} ti_L

        In the larger kernel, PfTransposeLayers are chained together as follows:
        
          - PfTransposeLayer(Din=1)
          - PfTransposeLayer(Din=2)
          - PfTransposeLayer(Din=4)
              ...
          - PfTransposeLayer(Din=Dcore//2)    # total number of PfTransposeLayers is log2(Dcore)
          - PfCore

        Each call to PfTransposeLayer.advance() processes (Din) registers, and two calls to advance()
        perform the transpose above (with 2*Din registers). On every second call to advance(), the
        PfTransposeLayer calls next_layer.advance().

        If (params.M) is a multiple of (2*Din), then this can be done "cleanly" with (params.M // Din)
        calls to advance(). If (params.M) is not a multiple of (2*Din), then the caller should call
        advance() the following number of times:
        
            (params.M + Din - 1) // Din

        with appropriate "zero-padding" on the last call. Then, next_layer.advance() will be called
        on every second call to advance(), and the last call to advance(). Thus, the number of calls
        to next_layer.advance() is:

            (params.M + 2*Din - 1) // (2*Din)

        providing the correct recursive number of calls for the PfTransposeLayer chain above.
        """

        assert (next_layer.Din % 2) == 0
        
        self.params = next_layer.params
        self.Din = next_layer.Din // 2
        self.next_layer = next_layer
        self.saved_rnames = [ ]

        
    def advance(self, k, rnames, m):
        dt32, M, Din = self.params.dt32, self.params.M, self.Din
        assert isinstance(rnames, list)
        assert len(rnames) == Din
        assert (m % Din) == 0
        assert 0 <= m < M

        if ((m % (2*Din)) == 0) and (m+Din < M):
            # Case 1: just save rnames and return
            self.saved_rnames = rnames
            return

        if m % (2*Din):
            # Case 2: combine (save_rnames + rnames), call next emitter.
            assert len(self.saved_rnames) == Din
            for sr,r in zip(self.saved_rnames, rnames):
                k.warp_transpose(sr, r, Din, dt32)
            
            self.next_layer.advance(k, self.saved_rnames + rnames, m-Din)
            self.saved_rnames = [ ]
        
        else:
            # Case 3: combine (rnames + zeros), call next emitter.
            # This is a "sentinel" case which arises if (M % (2*Din)) < Din.
            assert len(self.saved_rnames) == 0
            znames = [ f'pf_mpad_{Din}_{i}' for i in range(Din) ]

            for r,zr in zip(rnames, znames):
                k.emit(f'{dt32} {zr} = {self.params.dt32_scalar}(0.0f);')
                k.warp_transpose(r, zr, Din, dt32)

            self.next_layer.advance(k, rnames + znames, m)


####################################################################################################


class PfInitialTranspose16Layer:
    def __init__(self, next_layer):
        """
        The PfInitialTranspose16Layer converts this float16 register assignment:

           - Input:   th0 <-> t1 t2 t3 t4 t5     b <-> t0
           - Output:  th0 <-> t0 t1 t2 t3 t4     b <-> t5

        In a float16 kernel, the PfInitialTranspose16Layer is the topmost layer.
        The 'next_layer' is either a PfTransposeLayer (if Dcore > 1) or a PfCore (if Dcore == 1).
        """
        
        # FIXME optimize by adding Din==2 layer
        assert next_layer.Din == 1
        assert next_layer.params.dt32 == '__half2'
        self.params = next_layer.params
        self.next_layer = next_layer

        
    def advance(self, k, rnames, m):
        assert isinstance(rnames, list)
        assert len(rnames) == 1
        assert 0 <= m < self.params.M

        src = rnames[0]
        tmp0, tmp1 = k.get_tmp_rname(2)

        k.emit(f'\n// PfInitialTransposeLayer.advance({m=})')
        
        if m == 0:
            k.emit("const uint itrans16_lane0 = (threadIdx.x >> 1) & 0xf;")
            k.emit("const uint itrans16_lane1 = itrans16_lane0 | 0x10;")
            
        k.emit(f"__half2 {tmp0} = __shfl_sync(0xffffffff, {src}, itrans16_lane0);")
        k.emit(f"__half2 {tmp1} = __shfl_sync(0xffffffff, {src}, itrans16_lane1);")
        k.emit(f"{src} = (threadIdx.x & 1) ? __highs2half2({tmp0},{tmp1}) : __lows2half2({tmp0},{tmp1});")
        
        self.next_layer.advance(k, rnames, m)
        

####################################################################################################


class PfCore:
    def __init__(self, reducer, ringbuf):
        """
        Convolves the input array with peak-finding kernels.
        
        Input: a list of (Dcore) registers (where L = log2(Dcore))

            if float32:  th0 th1 th2 th3 th4 <-> m0 ... m_{L-1} ti_L .. ti4
                         r0 ... r_{L-1} <-> ti_0 ... ti_{L-1}

            if float16:  th0 th1 th2 th3 th4 <-> m0 ... m_{L-1} ti_L .. ti4
                         r0 ... r_{L-1} <-> ti_0 ... ti_{L-1}
                         b <-> ti_5
        
        For each 0 <= p < P, the PfCore convolves the input array with a peak-finding kernel,
        and does a thread-local reduction over index bits (ti_0 ... ti_{L-1}), to obtain
        'max' and 'ssq' registers in this ordering:
        
            if float32:  th0 th1 th2 th3 th4 <-> m0 ... m_{L-1} ti_L .. ti4
            if float16:  th0 th1 th2 th3 th4 <-> m0 ... m_{L-1} ti_L .. ti4    b <-> ti5

        These are passed to PfReducer.advance(). Thus, the total number of advance() calls is:

            (M + Dcore - 1) // Dcore          # number of calls to PfCore.advance()
            ((M + Dcore - 1) // Dcore) * P    # number of calls to PfReducer.advance()
        
        Weights are applied in PfReducer.advance(), not here in PfCore.advance().
        
        The PfCore uses a Ringbuf (or Ringbuf16) to manage overlaps when convolving.
        This is the only part of the peak-finding kernel which uses persistent state.
        """
        
        self.params = reducer.params
        self.reducer = reducer
        self.ringbuf = ringbuf
        self.Din = reducer.params.Dcore   # expected by earlier layers (e.g. PfTransposeLayer)

        if self.params.dt32 == 'float':
            assert isinstance(ringbuf, Ringbuf)
        elif self.params.dt32 == '__half2':
            assert isinstance(ringbuf, Ringbuf16)
        else:
            raise RuntimeError(f'Unrecognized dtype32: {self.params.dt32}')
        
        cls = Ringbuf16 if (self.params.dt32 == '__half2') else Ringbuf
        assert isinstance(ringbuf, cls)

    
    def advance(self, k, rnames, m):
        k.emit(f'\n// PfCore.advance(): {m=} {rnames = }\n')
        
        dt32, M, E, Dcore = self.params.dt32, self.params.M, self.params.E, self.params.Dcore
        assert isinstance(rnames, list)
        assert len(rnames) == Dcore
        assert (m % Dcore) == 0
        assert 0 <= m < M

        if (m == 0) and (E > 1):
            k.emit(f'const {dt32} pf_a = {self.params.dt32_scalar}(0.5f);')
        if (m == 0) and (E > 2):
            k.emit(f'const {dt32} one_over_sqrt2 = {self.params.dt32_scalar}(0.7071067811865476f);')

        k.emit('// PfCore: p=0 starts here')
        
        for t in range(Dcore):
            self._update(k, rnames[t], m, 0, t, Dcore)

        if E == 1:
            return

        # Note: prepadding logic isn't perfectly optimized, but it's very
        # convenient, and I doubt it's a bottleneck.

        k.emit('// PfCore: prepadding by 3, in prep for p=1,2,3')
        self._prepad(k, rnames, m, 1, 'pf_prepad_0')   # prepends new register name to 'rnames'
        self._prepad(k, rnames, m, 1, 'pf_prepad_1')   # prepends new register name to 'rnames'
        self._prepad(k, rnames, m, 1, 'pf_prepad_2')   # prepends new register name to 'rnames'
        k.emit()

        for ids in itertools.count(0):
            # Downsampling level 2**ids
            T = max(Dcore//2**ids, 1)
            assert len(rnames) == T+3

            k.emit(f'// PfCore: p={3*ids+1} starts here')
            for t in range(T):
                tmp_rname = k.get_tmp_rname()
                k.emit(f'{dt32} {tmp_rname} = {rnames[t+2]} + {rnames[t+3]};')
                self._update(k, tmp_rname, m, 3*ids+1, t, T)

            k.emit(f'// PfCore: p={3*ids+2} starts here')
            for t in range(T):
                tmp_rname = k.get_tmp_rname()
                k.emit(f'{dt32} {tmp_rname} = {rnames[t+2]} + pf_a * ({rnames[t+1]} + {rnames[t+3]});')
                self._update(k, tmp_rname, m, 3*ids+2, t, T)

            k.emit(f'// PfCore: p={3*ids+3} starts here')
            for t in range(T):
                tmp_rname = k.get_tmp_rname()
                k.emit(f'{dt32} {tmp_rname} = {rnames[t+1]} + {rnames[t+2]} + pf_a * ({rnames[t]} + {rnames[t+3]});')
                self._update(k, tmp_rname, m, 3*ids+3, t, T)

            if 2**(ids+1) == E:
                return

            k.emit(f'// PfCore: downsampling {2**ids} -> {2**(ids+1)} and prepadding, in prep for p={3*ids+4},{3*ids+5},{3*ids+6}')
            
            if T > 1:
                self._prepad(k, rnames, m, 2**ids, f'pf_prepad_{2*ids+3}')   # (T+3) -> (T+4)
                
            # Downsample in pairs
            for j in range(len(rnames)//2):
                k.emit(f'{rnames[2*j]} = one_over_sqrt2 * ({rnames[2*j]} + {rnames[2*j+1]});')
            rnames = rnames[::2]
            
            if T > 1:
                self._prepad(k, rnames, m, 2**(ids+1), f'pf_prepad_{2*ids+4}')
            else:
                self._prepad(k, rnames, m, 2**(ids+1), f'pf_prepad_{2*ids+3}')
                self._prepad(k, rnames, m, 2**(ids+1), f'pf_prepad_{2*ids+4}')

            k.emit()

    def _update(self, k, rname, m, p, t, T):
        max_rname = f'pf_max_p{p}'
        ssq_rname = f'pf_ssq_p{p}'

        if t == 0:
            decl = f'{self.params.dt32} ' if (m == 0) else ''
            k.emit(f'{decl}{max_rname} = {rname};')
            k.emit(f'{decl}{ssq_rname} = {rname} * {rname};')
        else:
            k.emit(f'{max_rname} = {self.params.dt32_max}({max_rname}, {rname});')
            k.emit(f'{ssq_rname} += {rname} * {rname};')

        if t == T-1:
            self.reducer.advance(k, max_rname, ssq_rname, m, p)


    def _prepad(self, k, rnames, m, tds, pp_rname):
        """'tds' is the current downsampling level of elements of 'rnames'."""

        Dcore = self.params.Dcore
        
        if m == 0:
            k.emit(f'{self.params.dt32} {pp_rname};')

        if tds < Dcore:
            self.ringbuf.advance(k, rnames[Dcore//tds - 1], Dcore, dst=pp_rname)
        else:
            self.ringbuf.advance(k, rnames[0], tds, dst=pp_rname)
        
        rnames.insert(0, pp_rname)


####################################################################################################


class PfReducer:
    def __init__(self, params):
        """
        Input: a pair of registers (max, ssq) in this register assignment (where L=log2(Dcore)):
        
            if float32:  th0 th1 th2 th3 th4 <-> m0 ... m_{L-1} ti_L .. ti4
            if float16:  th0 th1 th2 th3 th4 <-> m0 ... m_{L-1} ti_L .. ti4    b <-> ti5

        PfReducer.advance() is called in the following loops:

           for (int tout32 = 0; tout32 < Tout32; tout32 += (32 / Dout)) {    // runtime cuda loop
               for m in range(0,M,Dcore):                   # "unrolled" compile-time python loop
                   for p in range(P):                       # "unrolled" compile-time python loop
                       advance(m,p)

        Applies weights, and absorbs (max, ssq) into an "inner" (imax[P], issq[P]) register array.
        the same register assignment. When this step is repeated over 'm' (in the middle loop
        above), an "inner" reduce over index bits m_L, m_{L+1}, ... is performed.

        In the last call to advance(), i.e. when (m+Dcore >= M) and (p == P-1), we do
        an "outer" reduction, into an "outer" (omax[P], ossq[P]) register array, with the following
        register assignment (where K = log2(Dout)):

            if float32:  th0 th1 th2 th3 th4 <-> ti_5 ... ti_{K+4} ti_K .. ti4
            if float16:  th0 th1 th2 th3 th4 <-> ti_6 ... ti_{K+5} ti_K .. ti4     b <-> ti_5

        After (Dout) iterations of the outermost 'tout32' loop, or in the last iteration of the
        loop, we're ready to write to global memory. First, we do one more transpose, to get the
        register assignment:

            if float32:  th0 th1 th2 th3 th4 <-> ti_K ... ti_{K+4}
            if float16:  th0 th1 th2 th3 th4 <-> ti_{K+1} .. ti_{K+5}     b <-> ti_K
        """

        self.params = params
        
        Dout, P = params.Dout, params.P

        # "Inner" reduce over m-values
        # "Outer" reduce over (Dout) iterations of t-loop.
        self.imax_rnames = [ f'pf_imax_{p}' for p in range(P) ]
        self.issq_rnames = [ f'pf_issq_{p}' for p in range(P) ]
        self.omax_rnames = [ f'pf_omax_{p}' for p in range(P) ] if (Dout > 1) else self.imax_rnames
        self.ossq_rnames = [ f'pf_ossq_{p}' for p in range(P) ] if (Dout > 1) else self.issq_rnames

        
    def initialize(self, k):
        if self.params.Dout > 1:
            k.emit('// Persistent registers for PeakFinder')
            for r in (self.omax_rnames + self.ossq_rnames):
                k.emit(f'{self.params.dt32} {r} = {self.params.dt32_scalar}(0.0f);')
            k.emit()

    
    def advance(self, k, max_rname, ssq_rname, m, p):
        pars = self.params
        M, P, W, Dout, Dcore = pars.M, pars.P, pars.W, pars.Dout, pars.Dcore

        k.emit(f"// PfReducer: processing {(m,p)=}")
        k.emit(f"// PfReducer: recall that shmem[] has shape (P, W*M)")

        self._load_wt(k, m, p)   # emits line of the form 'wt = shmem[...];'
        
        imax, issq = self.imax_rnames[p], self.issq_rnames[p]        
        mpop = min(M-m, Dcore)   # number of populated m-indices
        btmp = None            # boolean mask (may not be needed)
        
        if mpop < Dcore:
            k.emit(f'// PfReducer: m-values are partially populated ({mpop=}, {Dcore=}).')
            k.emit(f'// This mask keeps track of which threads store valid values')
            btmp = k.get_tmp_rname()
            k.emit(f'bool {btmp} = (threadIdx.x & {Dcore-1}) < {mpop};  // {(Dcore-1)=}, {mpop=}')

        # Apply weights, and absorb into an "inner" register pair (imax, issq) with the same
        # register assignment. There are 4 cases here, but they're not very different.
        # This "inner" reduction is the simple one (the "outer" reduction below is more complicated).

        if (m > 0) and (mpop >= Dcore):
            k.emit(f"// PfReducer: inner reduce over m-values, in single iteration of outer t-loop.")
            k.emit(f'{imax} = {pars.dt32_max}({imax}, wt*{max_rname});')
            k.emit(f'{issq} += wt*wt*{ssq_rname};')
        elif (m == 0) and (mpop >= Dcore):
            k.emit("// PfReducer: apply weights")
            k.emit(f'{pars.dt32} {imax} = wt*{max_rname};')
            k.emit(f'{pars.dt32} {issq} = wt*wt*{ssq_rname};')
        elif (m > 0) and (mpop < Dcore):
            k.emit(f"// PfReducer: inner reduce over m-values, in single iteration of outer t-loop.")
            k.emit(f'{imax} = {btmp} ? {pars.dt32_max}({imax}, wt*{max_rname}) : {imax};')
            k.emit(f'{issq} = {btmp} ? ({issq} + wt*wt*{ssq_rname}) : {issq};')
        else:
            k.emit("// PfReducer: apply weights")
            k.emit(f'{pars.dt32} {imax} = {btmp} ? wt*{max_rname}: {pars.dt32_scalar}(0.0f);')
            k.emit(f'{pars.dt32} {issq} = {btmp} ? wt*wt*{ssq_rname} : {pars.dt32_scalar}(0.0f);')
        
        k.emit()

        if (m + Dcore < M) or (p < P-1):
            return

        k.emit('// PfReducer: all (m,p) values have been processed, this should be near the bottom of the outer t-loop')
        k.emit()

        self._outer_reduce(k)
        self._write_to_global_memory(k)


    def _load_wt(self, k, m, p):
        pars = self.params
        M, P, W, Dout, Dcore = pars.M, pars.P, pars.W, pars.Dout, pars.Dcore
        
        if p == 0:
            decl = f'uint ' if (m==0) else ''
            k.emit(f'// pfr_m = value of m on this thread (used for reading wt[] array')
            k.emit(f'{decl}pfr_m = {m} + (threadIdx.x & {Dcore-1});  // (Dcore-1)={Dcore-1}')
            if m + Dcore > M:
                k.emit(f'pfr_m = min(pfr_m, {M-1});   // (M-1)={M-1}')
                
            k.emit(f'// pfr_j = index into inner axis (length W*M) of shape-(P,W*M) shmem array')
            k.emit(f'{decl}pfr_j = (threadIdx.x >> 5) * M + pfr_m;')
            
            if pars.dtype == '__half':
                k.emit(f'// pfr_cw = control word for selecting __half from __half2')
                k.emit(f'{decl}pfr_cw = (pfr_j & 1) ? 0x3232 : 0x1010;')
                k.emit(f'pfr_j >>= 1;')

        poff = (p * W * M * pars.nbits) // 32
        poff = f'+ {poff}' if (poff > 0) else ''
        decl = f'{pars.dt32} ' if (m==p==0) else ''        
        k.emit(f'{decl}wt = shmem[pfr_j{poff}];')

        if pars.dtype == '__half':
            k.emit(f'wt = ksgpu::f16_perm(wt, wt, pfr_cw);')
            

    
    def _outer_reduce(self, k):
        pars = self.params
        M, P, Dout, Dcore, tout32 = pars.M, pars.P, pars.Dout, pars.Dcore, pars.tout32_rname

        if Dout == 1:
            return
        
        mpop = min(M, Dcore)   # number of populated m-indices
        d = 1                  # current reduction level (power of two)
        
        k.emit(f'// PfReducer: "inner" reduction into imax/issq registers')
        k.emit(f'// Number of m-indices is {Dcore=}, and {mpop=} of these are populated.')
        k.emit(f'// Number of time indices is {(Dout//Dcore)=}.')
        k.emit()
        
        while d < Dout:
            assert min(d,Dcore) <= mpop <= Dcore
            
            if (d >= Dcore) or ((mpop & d) == 0):
                k.emit(f'// "Cleanly" increase reduction level {d=} -> {(2*d)=}.')
                for p in range(P):
                    imax, issq = self.imax_rnames[p], self.issq_rnames[p]
                    k.emit(f'{imax} = {pars.dt32_max}({imax}, __shfl_sync(0xffffffff, {imax}, threadIdx.x ^ {d}));')
                    k.emit(f'{issq} += __shfl_sync(0xffffffff, {issq}, threadIdx.x ^ {d});')
                
                k.emit()
                d *= 2

            elif (d == mpop):
                k.emit(f'// Increase reduction level {d=} -> {Dcore=}')
                k.emit(f'// This can be done with just __shfl_sync() (no max/sum ops), since d==mpop')
                
                for p in range(P):
                    mask = (32 - Dcore) + (d - 1)  # clear bits b such that d <= b < Dcore
                    imax, issq = self.imax_rnames[p], self.issq_rnames[p]
                    k.emit(f'{imax} = __shfl_sync(0xffffffff, {imax}, threadIdx.x & {mask});')
                    k.emit(f'{issq} = __shfl_sync(0xffffffff, {issq}, threadIdx.x & {mask});')
                
                k.emit()
                mpop = d = Dcore    # not (d *= 2)

            else:
                assert (mpop & (d-1)) == 0
                k.emit(f'// Increase reduction level {d=} -> {(2*d)=}.')
                k.emit(f'// Since {mpop=}, this reduction "mixes" populated/unpopulated indices')

                btmp1, btmp2 = k.get_tmp_rname(2)
                k.emit(f'bool {btmp1} = (threadIdx.x & {Dcore-1}) < {mpop};           // is current thread populated?')
                k.emit(f'bool {btmp2} = ((threadIdx.x & {Dcore-1}) ^ {d}) < {mpop};   // is "counterpart" thread populated?')
                
                for p in range(P):
                    imax, issq = self.imax_rnames[p], self.issq_rnames[p]
                    tmp1, tmp2 = k.get_tmp_rname(2)

                    k.emit(f'{pars.dt32} {tmp1} =  __shfl_sync(0xffffffff, {imax}, threadIdx.x ^ {d});')
                    k.emit(f'{tmp1} = {btmp1} ? {pars.dt32_max}({tmp1}, {imax}) : {tmp1};')
                    k.emit(f'{imax} = {btmp2} ? {tmp1} : {imax};')

                    k.emit(f'{pars.dt32} {tmp2} =  __shfl_sync(0xffffffff, {issq}, threadIdx.x ^ {d});')
                    k.emit(f'{tmp2} = {btmp1} ? ({tmp2} + {issq}) : {tmp2};')
                    k.emit(f'{issq} = {btmp2} ? {tmp2} : {issq};')
                
                k.emit()
                mpop += d
                d *= 2

        assert mpop == Dcore
        k.emit('// PfReducer: "Absorb" result of inner reduction into outer persistent registers (on a subset of threads).')
        k.emit()

        btmp_rname = k.get_tmp_rname()
        
        k.emit(f'// Set {btmp_rname} on threads where (threadIdx.x * (32/Dout)) == tout32 (mod 32)')
        k.emit(f'// These are the threads where persistent max/ssq will be updated in this iteration of the t-loop')
        k.emit(f'bool {btmp_rname} = (((threadIdx.x * (32/Dout)) ^ {tout32}) & 0x1f) == 0;')
        k.emit()

        for p in range(P):
            k.emit(f'{self.omax_rnames[p]} = {btmp_rname} ? {self.imax_rnames[p]} : {self.omax_rnames[p]};')
            k.emit(f'{self.ossq_rnames[p]} = {btmp_rname} ? {self.issq_rnames[p]} : {self.ossq_rnames[p]};')
        
        k.emit()

    
    def _write_to_global_memory(self, k):
        pars = self.params
        P, Dout, tout32, Tout32, Mout = pars.P, pars.Dout, pars.tout32_rname, pars.Tout32_rname, pars.Mout_rname

        k.emit('// PfReducer: write out/ssq to global memory')
        k.emit()

        if Dout > 1:
            k.emit(f'int {tout32}_next = tout32 + (32/Dout);   // value of tout32 in next iteration of loop')
            k.emit(f'if (({tout32}_next == {Tout32}) || (({tout32}_next & 0x1f) == 0)) {{   // current **warp** writes max/ssq to global memory')

        self._transpose_outputs(k, self.omax_rnames + self.ossq_rnames)

        if Dout > 1:
            k.emit(f'{tout32}_next = (tout32 & 0x1f) + (32/Dout);  // advance by (32/Dout)')
            k.emit(f'if ((threadIdx.x & 0x1f) < {tout32}_next) {{   // current **thread** writes max/ssq to global memory')

        k.emit('// PfReducer: Write to global memory.')
        k.emit('// Reminder: output max/ssq arrays have shape (P,Mout,Tout32).')

        for p in range(P):
            s = f'{p}*{Tout32}*{Mout}' if (p > 0) else '0'
            k.emit(f'{pars.out_max_rname}[{s}] = {self.omax_rnames[p]};')
            k.emit(f'{pars.out_ssq_rname}[{s}] = {self.ossq_rnames[p]};')
        
        if Dout > 1:
            k.emit('}  // "if current **thread** writes max/ssq to global memory..."')

        k.emit()
        k.emit('// PfReducer: advance max/ssq global memory pointers by 32 output time samples')
        k.emit(f'{pars.out_max_rname} += 32;')
        k.emit(f'{pars.out_ssq_rname} += 32;')

        if Dout > 1:
            k.emit('}  // "if current **warp* writes max/ssq to global memory..."')
        
        k.emit()


    def _transpose_outputs(self, k, rnames):
        Dout = self.params.Dout
            
        if self.params.dtype == 'float':
            if Dout == 1:
                return
            
            k.emit(f'// PfReducer: transpose output time indices')
            k.emit(f'// Before the transpose, {(32//Dout)=} "inner" time indices are assigned to "outer" threads')
            k.emit(f'// and {Dout=} "outer" time indices are assigned to "inner" threads.')

            itmp_rname = k.get_tmp_rname()
            k.emit(f'int {itmp_rname} = (threadIdx.x & 0x1f) * Dout;')
            k.emit(f'{itmp_rname} = {itmp_rname} | ({itmp_rname} >> 5);')
            
            for r in rnames:
                k.emit(f'{r} = __shfl_sync(0xffffffff, {r}, {itmp_rname});')

        elif self.params.dtype == '__half':
            if Dout == 32:
                return
            
            L = utils.integer_log2(32//Dout)
            assert 1 <= L <= 5
            
            k.emit(f'// PfReducer: transpose output time indices, before writing to global memory.')
            k.emit(f'//')
            k.emit(f'// Input (where L = log2(32/Dout) = {L})')
            k.emit(f'// b <-> ti_L   th0 th1 th2 th3 th4 <->  ti_{{L+1}} ..ti_5 ti_0 .. ti_{{L-1}}')
            k.emit(f'//')
            k.emit(f'// Output:')
            k.emit(f'// b <-> ti_0   th0 th1 th2 th3 th4 <-> ti_1 ti_2 ti_3 ti_4 ti_5')
            k.emit()
            k.emit(f'constexpr int L = {L};')
            k.emit('uint pfr_lane0a = (threadIdx.x << (6-L));')
            k.emit('uint pfr_lane0b = (threadIdx.x >> L) & ((1 << (5-L)) - 1);')
            k.emit('uint pfr_lane0 = pfr_lane0a | pfr_lane0b;   // source for ti0=0')
            k.emit('uint pfr_lane1 = pfr_lane0 ^ (1 << (5-L));  // source for ti0=1')

            for r in rnames:
                y0, y1 = k.get_tmp_rname(2)
                k.emit(f'__half2 {y0} = __shfl_sync(0xffffffff, {r}, pfr_lane0);  // ti0=0')
                k.emit(f'__half2 {y1} = __shfl_sync(0xffffffff, {r}, pfr_lane1);  // ti0=1')
                k.emit(f'{r} = (threadIdx.x & (1 << (L-1))) ? __highs2half2({y0},{y1}) : __lows2half2({y0},{y1});')

        else:
            raise RuntimeError(f'dtype={self.params.dtype} not recognized')
