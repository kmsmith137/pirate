import itertools

from . import utils
from .Ringbuf import Ringbuf


class PeakFindingParams:
    def __init__(self, M, E, Dout, W, BlocksPerSM):
        """
        All kernel params except Din/Core.

           M = number of "rows" (e.g. trial DMs) per warp
           E = max kernel width
           Dout = output time downsampling factor
           W = warps per threadblock
        """
        
        assert M >= 1
        assert E <= 32
        assert Dout <= 32
        assert 1 <= (W * BlocksPerSM) <= 32
        
        assert utils.is_power_of_two(E)
        assert utils.is_power_of_two(Dout)
                
        self.M = M
        self.E = E
        self.W = W
        self.Dout = Dout
        self.BlocksPerSM = BlocksPerSM

        # Number of peak-finding profiles.
        self.P = 3 * utils.integer_log2(E) + 1

        # Variables used internally by kernel.
        # To override these defaults, just assign new values after construction.
        self.wt_sh_rname = 'wt_sh'      # shared memory pointer, shape (P, W*M)
        self.wt_glo_rname = 'wt'        # global memory pointer, shape (B, P, Mout*M)
        self.out_max_rname = 'out_max'  # global memory pointer, shape (B, P, Mout, Tout)
        self.out_ssq_rname = 'out_ssq'  # global memory pointer, shape (B, P, Mout, Tout)
        self.tout_rname = 'tout'        # advances by (32/Dout) in each loop iteration
        self.Tout_rname = 'Tout'
        self.Mout_rname = 'Mout'


class PeakFinder:
    def __init__(self, params, Dcore, reduce_only=False):
        """
        The 'Dcore' argument is an internal downsampling factor.
        Large Dcore uses fewer instructions, but more registers.
        """

        assert utils.is_power_of_two(Dcore)
        assert Dcore <= params.Dout
        
        self.params = params
        self.Dcore = Dcore
        self.reduce_only = reduce_only

        s = f'reduce_only_' if reduce_only else ''
        self.kernel_name = f'pf_{s}kernel_M{params.M}_E{params.E}_Dcore{Dcore}_Dout{params.Dout}_W{params.W}_B{params.BlocksPerSM}'
        self.reducer = PfReducer(params, Din=Dcore)

        if not reduce_only:
            self.ringbuf = Ringbuf()
            self.pf = PfCore(self.reducer, self.ringbuf)
            while self.pf.Din > 1:
                self.pf = PfTransposeLayer(self.pf)
        
        
    def emit_global(self, k):
        pars = self.params
        M, E, Dout, W, P = pars.M, pars.E, pars.Dout, pars.W, pars.P
        Dcore = self.Dcore

        if not self.reduce_only:
            # Main case: full peak-finding kernel.
            in_args = ('pstate', 'in')
            in_line1 = "'pstate' has shape (B, Mout, RW), where RW is defined below"
            in_line2 = "'in' has shape (B, Mout*M, Tout*Dout)"
        else:
            # Debug case: reduce-only kernel (with different kernel args).
            in_args = ('in_max', 'in_ssq')
            in_line1 = "'in_max' has shape (B, P, Mout*M, Tout*(Dout/Dcore))"
            in_line2 = "'in_ssq' has shape (B, P, Mout*M, Tout*(Dout/Dcore))"
        
        k.emit(f'// Kernel args are (out_max, out_ssq, {in_args[0]}, {in_args[1]}, wt, Mout, Tout).')
        k.emit(f'//')
        k.emit(f"//    - 'out_max' and 'out_ssq' have shape (B, P, Mout, Tout).")
        k.emit(f"//    - {in_line1}.")
        k.emit(f"//    - {in_line2}.")
        k.emit(f"//    - 'wt' has shape (B, P, Mout*M).")
        k.emit("//")
        k.emit('// Each warp processes one output 0 <= mout < Mout, and all times 0 <= tout < Tout.')
        k.emit('//')
        k.emit('// Launch with {BM,B,1} blocks, where BM = ceil(Mout/W), and B is number of beams.')
        k.emit('// Launch with {32*W,1} threads.')
        k.emit()
        k.emit(f'__global__ void __launch_bounds__({32*W}, {pars.BlocksPerSM})')
        k.emit(f'{self.kernel_name}(float *out_max, float *out_ssq, float *{in_args[0]}, float *{in_args[1]}, float *wt, int Mout, int Tout)')
        k.emit('{')
        
        k.emit(f'constexpr int {M = };      // number of "rows" (e.g. trial DMs) per warp')
        k.emit(f'constexpr int {Dout = };   // output time downsampling factor')
        k.emit(f'constexpr int {Dcore = };  // internal resolution parameter')
        k.emit(f'constexpr int {W = };      // warps per threadblock')
        k.emit(f'constexpr int {P = };      // number of peak-finding kernels')
        
        if not self.reduce_only:
            k.emit(f'constexpr int {E = };      // max kernel width')

        k.emit("int b = blockIdx.y;  // beam index")
        k.emit("int mblock = blockIdx.x * W;              // base m-index of block")
        k.emit("int mout = mblock + (threadIdx.x >> 5);   // output m-index of warp")
        k.emit()
        k.emit("bool warp_active = (mout < Mout);")
        k.emit("mout = warp_active ? mout : (M-1);")
        k.emit()
        
        k.emit('// Shared memory layout: wt[P][W][M]')
        k.emit(f'// nelts={(P*W*M)=}, nbytes={(4*P*W*M)=}')
        k.emit(f'__shared__ float shmem[{P*W*M}];')
        k.emit()

        k.emit("// Apply per-beam offset to 'wt', in preparation for loading weights.")
        k.emit("// Shape is (B, P, Mout*M).")
        k.emit()
        k.emit("wt += b * (P * Mout * M);")
        k.emit()
        
        k.emit("// Copy weights from global memory to shared.")
        k.emit("// The destination array has shape (P, W*M), contiguously packed.")
        k.emit("// The source array is a discontiguous subset of a shape (B, P, Mout*M) source array.")
        k.emit("// FIXME this loop could be optimized a little.")
        k.emit()
        k.emit("for (int i = threadIdx.x; i < P*W*M; i += 32*W) {")
        k.emit("    int p = i / (W*M);")
        k.emit("    int m = i - (W*M)*p + mblock;")
        k.emit("    m = min(m, Mout*M-1);")
        k.emit("    shmem[i] = wt[p*Mout*M + m];")
        k.emit("}")
        k.emit()
        k.emit('__syncthreads();')
        k.emit()
        k.emit('// Per-warp weights array in shared memory, with shape (P,M) and stride (W*M).')
        k.emit('float *wt_sh = shmem + mout * M;')
        k.emit()
        
        k.emit("// Apply per-thread offset, to 'out_max' and 'out_ssq'.")
        k.emit("// Shapes are (B, P, Mout, Tout), and pstride is Mout*Tout.")
        k.emit("// This leaves an offset 0 <= p < P to be applied later (with stride Mout*Tout).")
        k.emit()
        k.emit("int out_offset = (b * P * Mout + mout) * Tout + (threadIdx.x & 0x1f);")
        k.emit("out_max += out_offset;")
        k.emit("out_ssq += out_offset;")
        k.emit()

        if not self.reduce_only:
            # Main case: full peak-finding kernel.
            k.emit("// Apply per-thread offsets to 'in'.")
            k.emit("// Shape is (B, Mout*M, Tout*Dout)")
            k.emit()
            k.emit("int in_mstride = Tout * Dout;")
            k.emit(f"in += (b*Mout + mout) * M * in_mstride + (threadIdx.x & 0x1f);")
            k.emit()

            # Save splice, for code to load the ring buffer.
            # (This code must be emitted after the main lloop.)
            k_rb = k.splice()

        else:
            # Debug case: reduce-only kernel (with different kernel args).
            k.emit("// Apply per-warp offsets, and per-thread time offset, to 'in_max' and 'in_ssq'.")
            k.emit("// Shape in GPU global memory is (B, P, Mout*M, Tout*(Dout/Dcore)).")
            k.emit("// This leaves a per-thread offset 0 <= m < M to be applied later (with stride 'in_mstride').")
            k.emit("// This leaves an offset 0 <= p < P to be applied later (with stride 'in_pstride').")
            k.emit("// In each iteration of the t-loop, these pointers advance by (32/Dcore), not (32/Dout).")
            k.emit()
            k.emit("int in_mstride = Tout * (Dout/Dcore);")
            k.emit("int in_pstride = Mout * M * in_mstride;")
            k.emit("int in_offset = (b*P*in_pstride) + (mout * M * in_mstride);    // per-warp offset")
            k.emit(f"in_offset += ((threadIdx.x & 0x1f) / Dcore);          // per-thread time offset")
            k.emit(f"in_max += in_offset;")
            k.emit(f"in_ssq += in_offset;")
            k.emit()

        self.reducer.initialize(k)

        k.emit(f'// PeakFinder: main outer loop')
        k.emit(f"for (int tout = 0; tout < Tout; tout += (32/Dout)) {{")
        k.emit("    if (!warp_active)")
        k.emit("        continue;")
        k.emit()

        if not self.reduce_only:
            # Main case: full peak-finding kernel.
            for m in range(M):
                rname = f'x{m}'
                s = f'{m} * in_mstride' if (m > 0) else '0'
                k.emit(f'// PeakFinder: {m=}\n')
                k.emit(f"float {rname} = in[{s}];")
                self.pf.advance(k, [rname], m)
                k.emit()

        else:
            # Debug case: reduce-only kernel (with different kernel args).
            for m0 in range(0,M,Dcore):
                k.emit(f'// PeakFinder: {m0=}')
                prefix = 'int ' if (m0 == 0) else ''
                suffix = f' + {m0}' if (m0 > 0) else ''
                k.emit(f'{prefix}pf_m = (threadIdx.x & (Dcore-1)){suffix};   // {m0=}')
                k.emit()
                                
                for p in range(P):
                    k.emit(f'// PeakFinder: {m0=}, {p=}')
                    max_rname = f'in_max_m{m0}_p{p}'
                    ssq_rname = f'in_ssq_m{m0}_p{p}'
                    k.emit(f'float {max_rname} = (pf_m < M) ? in_max[({p} * in_pstride) + (pf_m * in_mstride)] : 1000.0f;')
                    k.emit(f'float {ssq_rname} = (pf_m < M) ? in_ssq[({p} * in_pstride) + (pf_m * in_mstride)] : 1000.0f;')
                    self.reducer.advance(k, max_rname, ssq_rname, m0, p)
                
            k.emit("// Advance 'in_max' and 'in_ssq' by (32/Dcore) time samples")
            k.emit("in_max += (32/Dcore);")
            k.emit("in_ssq += (32/Dcore);")
            

        if not self.reduce_only:
            k.emit()
            self.ringbuf.advance_outer(k)
        
        k.emit("}  // end of t-loop")
        k.emit()

        if not self.reduce_only:
            self._load_ringbuf(k_rb)
            self._save_ringbuf(k)
        
        k.emit("}  // end of cuda kernel")


    def _load_ringbuf(self, k):
        RW = self.ringbuf.get_nelts_per_warp()
        
        k.emit("// Apply per-warp offset to pstate, in preparation for loading")
        k.emit("// Shape is (B, Mout, RW).")
        k.emit()
        k.emit(f'constexpr int {RW = };    // ring buffer elements per warp')
        k.emit("pstate += (b*Mout + mglo) * RW;")
        k.emit()

        k.emit('// Read ring buffer directly from global memory.')
        k.emit('// FIXME: would it be better to go through shared memory?')
        
        self.ringbuf.initialize(k, 'pstate')
        k.emit()

        
    def _save_ringbuf(self, k):
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
        PfTransposeLayer.advance() is called for m = 0, Din, (2*Din), ...
        Each call consists of Din registers which represent consecutive times.
        The innermost Din thread indices correspond to m, (m+1), ... (m+Din-1).
        The outer thread indices correspond to (time/Din).
        """

        assert (next_layer.Din % 2) == 0
        
        self.M = next_layer.M
        self.Din = next_layer.Din // 2
        self.next_layer = next_layer
        self.saved_rnames = [ ]

        
    def advance(self, k, rnames, m):
        M, Din = self.M, self.Din
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
                k.warp_transpose(sr, r, Din, 'float')
            
            self.next_layer.advance(k, self.saved_rnames + rnames, m-Din)
            self.saved_rnames = [ ]
        else:
            # Case 3: combine (rnames + zeros), call next emitter.
            # This is a "sentinel" case which arises if (M % (2*Din)) < Din.
            assert len(self.saved_rnames) == 0
            znames = [ f'pf_mpad_{Din}_{i}' for i in range(Din) ]

            for r,zr in zip(rnames, znames):
                k.warp_transpose(r, zr, Din, 'float')

            self.next_layer.advance(k, rnames + znames, m)

            
####################################################################################################


class PfCore:
    def __init__(self, reducer, ringbuf):
        self.M = reducer.params.M
        self.E = reducer.params.E
        self.Din = reducer.Din
        self.reducer = reducer
        self.ringbuf = ringbuf

    
    def advance(self, k, rnames, m):
        k.emit(f'\n// PfCore.advance(): {m=} {rnames = }\n')
        
        M, E, Din = self.M, self.E, self.Din
        assert isinstance(rnames, list)
        assert len(rnames) == Din
        assert (m % Din) == 0
        assert 0 <= m < M

        tmp_rname = k.get_tmp_rname('float')
        
        for t in range(Din):
            self._update(k, rnames[t], m, 0, t, Din)

        if E == 1:
            return

        # Note: prepadding logic isn't perfectly optimized, but it's very
        # convenient, and I doubt it's a bottleneck.

        self._prepad(k, rnames, m, 1, 'pf_prepad_0')
        self._prepad(k, rnames, m, 1, 'pf_prepad_1')
        self._prepad(k, rnames, m, 1, 'pf_prepad_2')

        for ids in itertools.count(0):
            # Downsampling level 2**ids
            T = max(Din//2**ids, 1)
            assert len(rnames) == T+3

            for t in range(T):
                k.emit(f'{tmp_rname} = {rnames[t+1]} + {rnames[t+2]};')
                self._update(k, tmp_rname, m, 3*ids+1, t, T)

            for t in range(T):
                k.emit(f'{tmp_rname} = {rnames[t+1]} + 0.5f * ({rnames[t]} + {rnames[t+2]});')
                self._update(k, tmp_rname, m, 3*ids+2, t, T)

            for t in range(T):
                k.emit(f'{tmp_rname} = {rnames[t+1]} + {rnames[t+2]} + 0.5f * ({rnames[t]} + {rnames[t+3]});')
                self._update(k, tmp_rname, m, 3*ids+3, t, T)

            if 2**(ids+1) == E:
                return

            if T > 1:
                self._prepad(k, rnames, m, 2**ids, f'pf_prepad_{2*ids+1}')   # (T+3) -> (T+4)
                
            # Downsample in pairs
            for j in range(len(rnames)//2):
                k.emit(f'{rnames[2*j]} += {rnames[2*j+1]};')
            rnames = rnames[::2]
            
            if T > 1:
                self._prepad(k, rnames, m, 2**(ids+1), f'pf_prepad_{2*ids+2}')
            else:
                self._prepad(k, rnames, m, 2**(ids+1), f'pf_prepad_{2*ids+1}')
                self._prepad(k, rnames, m, 2**(ids+1), f'pf_prepad_{2*ids+2}')
            

    def _update(self, k, rname, m, p, t, T):
        max_rname = f'pf_max_p{p}'
        ssq_rname = f'pf_ssq_p{p}'

        if t == 0:
            decl = 'float ' if (m == 0) else ''
            k.emit(f'{decl}{max_rname} = {rname};')
            k.emit(f'{decl}{ssq_rname} = {rname} * {rname};')
        else:
            k.emit(f'{max_rname} = max({max_rname}, {rname});')
            k.emit(f'{ssq_rname} += {rname} * {rname};')

        if t == T-1:
            self.reducer.advance(k, max_rname, ssq_rname, m, p)


    def _prepad(self, k, rnames, m, tds, pp_rname):
        """'tds' is the current downsampling level of elements of 'rnames'."""

        if m == 0:
            k.emit(f'float {pp_rname};')

        if tds < self.Din:
            self.ringbuf.advance(k, rnames[self.Din//tds - 1], self.Din, dst=pp_rname)
        else:
            self.ringbuf.advance(k, rnames[0], tds, dst=pp_rname)
        
        rnames.insert(0, pp_rname)


####################################################################################################


class PfReducer:
    def __init__(self, params, Din):
        assert utils.is_power_of_two(Din)
        assert Din <= params.Dout

        self.params = params
        self.Din = Din
        
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
                k.emit(f'float {r} = 0.0f;')
            k.emit()

    
    def advance(self, k, max_rname, ssq_rname, m, p):
        """
        The max/ssq register assignment is as follows:
           - (32/Dout) outer time indices
           - (Dout/Din) inner time indices (will be reduced)
           - (Din) m-indices (will be reduced)

        PfReducer.advance() should be called as follows:
           - m=0: p = 0, 1, ..., P-1
           - m=Din: p = 0, 1, ..., P-1
           - m=2*Din: p = 0, 1, ..., P-1

        Calls to PfReducer.advance() should be embedded in a loop where tout increases by (32/Dout) in each iteration.
        """
        
        Din = self.Din
        pars = self.params
        M, P, W, Dout, wt_sh = pars.M, pars.P, pars.W, pars.Dout, pars.wt_sh_rname

        k.emit(f"// PfReducer: load weights at {(p,m)=}")
        k.emit(f"// PfReducer: recall that '{wt_sh}' is a per-warp shared memory array with shape {(P,M)=} and stride {(W*M)=}")
        woff = p*W*M + m
        s = f'threadIdx.x & {(Din-1)}'
        s = f'({s}) + {woff}' if woff else s
        decl = 'float ' if (m==p==0) else ''        
        k.emit(f'{decl}wt = {wt_sh}[{s}];   // {(p*W*M + m) = }')
        
        imax, issq = self.imax_rnames[p], self.issq_rnames[p]
        
        mpop = min(M-m, Din)   # number of populated m-indices
        if mpop < Din:
            k.emit(f'// PfReducer: m-values are partially populated ({mpop=}, {Din=}).')
            k.emit(f'// This mask keeps track of which threads store valid values')
            btmp = k.get_tmp_rname('bool')
            k.emit(f'{btmp} = (threadIdx.x & {Din-1}) < {mpop};  // {(Din-1)=}, {mpop=}')

        # Ugh, 4 cases here.
        if (m > 0) and (mpop >= Din):
            k.emit(f"// PfReducer: inner reduce over m-values, in single iteration of outer t-loop.")
            k.emit(f'{imax} = max({imax}, wt*{max_rname});')
            k.emit(f'{issq} += wt*wt*{ssq_rname};')
        elif (m == 0) and (mpop >= Din):
            k.emit("// PfReducer: apply weights")
            k.emit(f'float {imax} = wt*{max_rname};')
            k.emit(f'float {issq} = wt*wt*{ssq_rname};')
        elif (m > 0) and (mpop < Din):
            k.emit(f"// PfReducer: inner reduce over m-values, in single iteration of outer t-loop.")
            k.emit(f'{imax} = {btmp} ? max({imax}, wt*{max_rname}) : {imax};')
            k.emit(f'{issq} = {btmp} ? ({issq} + wt*wt*{ssq_rname}) : {issq};')
        else:
            k.emit("// PfReducer: apply weights")
            k.emit(f'float {imax} = {btmp} ? wt*{max_rname}: 0.0f;')
            k.emit(f'float {issq} = {btmp} ? wt*wt*{ssq_rname} : 0.0f;')
        
        k.emit()

        if (m + Din < M) or (p < P-1):
            return

        k.emit('// PfReducer: all (m,p) values have been processed, this should be near the bottom of the outer t-loop')
        k.emit()

        if Dout > 1:
            self._outer_reduce(k)

        self._write_to_global_memory(k)


    def _outer_reduce(self, k):
        Din = self.Din
        pars = self.params
        M, P, Dout, tout = pars.M, pars.P, pars.Dout, pars.tout_rname
        assert Dout > 1
        
        mpop = min(M, Din)   # number of populated m-indices
        d = 1                # current reduction level
        
        k.emit(f'// PfReducer: here is a big block of instructions to further reduce by a factor {Dout=}.')
        k.emit(f'// Number of m-indices is {Din=}, and {mpop=} of these are populated.')
        k.emit(f'// Number of time indices is {(Dout//Din)=}.')
        k.emit()
        
        btmp_rname = k.get_tmp_rname('bool')
        
        while d < Dout:
            if (d >= Din) or ((mpop & d) == 0):
                k.emit(f'// "Cleanly" increase reduction level {d=} -> {(2*d)=}.')
                for p in range(P):
                    imax, issq = self.imax_rnames[p], self.issq_rnames[p]
                    k.emit(f'{imax} = max({imax}, __shfl_sync(0xffffffff, {imax}, threadIdx.x ^ {d}));')
                    k.emit(f'{issq} += __shfl_sync(0xffffffff, {issq}, threadIdx.x ^ {d});')
                
                k.emit()
                d *= 2

            elif d >= mpop:
                k.emit(f'// Increase reduction level {d=} -> {Din=}')
                k.emit(f'// This can be done with just __shfl_sync() (no max/sum ops), since {d=} exceeds {mpop=}')
                
                for p in range(P):
                    imax, issq = self.imax_rnames[p], self.issq_rnames[p]
                    k.emit(f'{max_rname} = __shfl_sync(0xffffffff, {max_rname}, threadIdx.x & {32-Din});  // {(32-Din)=}')
                    k.emit(f'{ssq_rname} = __shfl_sync(0xffffffff, {ssq_rname}, threadIdx.x & {32-Din});  // {(32-Din)=}')
                
                k.emit()
                d = Din    # not (d *= 2)

            else:
                k.emit(f'// Increase reduction level {d=} -> {(2*d)=}.')
                k.emit(f'// Since {mpop=}, this reduction "mixes" populated/unpopulated indices')
                k.emit(f'{btmp_rname} = (threadIdx.x & {Din-1}) < {mpop-d};   // {(mpop-d)=}')
                
                for p in range(P):
                    tmp_rname = k.get_tmp_rname('float')
                    imax, issq = self.imax_rnames[p], self.issq_rnames[p]
                    k.emit(f'{tmp_rname} = max({imax}, __shfl_sync(0xffffffff, {imax}, threadIdx.x ^ {d}));')
                    k.emit(f'{imax} = {btmp_rname} ? {tmp_rname} : {imax};')
                    k.emit(f'{tmp_rname} = {issq} + __shfl_sync(0xffffffff, {issq}, threadIdx.x ^ {d}));')
                    k.emit(f'{issq} = {btmp_rname} ? {tmp_rname} : {issq};')
                
                k.emit()
                d *= 2

        k.emit('// PfReducer: "Absorb" result of Dout-way reduction into persistent registers (on a subset of threads).')
        k.emit()
        
        k.emit(f'// Set {btmp_rname} on threads where (threadIdx.x * (32/Dout)) == tout (mod 32)')
        k.emit(f'// These are the threads where persistent max/ssq will be updated in this iteration of the t-loop')
        k.emit(f'{btmp_rname} = (((threadIdx.x * (32/Dout)) ^ {tout}) & 0x1f) == 0;')
        k.emit()

        for p in range(P):
            k.emit(f'{self.omax_rnames[p]} = {btmp_rname} ? {self.omax_rnames[p]} : {self.imax_rnames[p]};')
            k.emit(f'{self.ossq_rnames[p]} = {btmp_rname} ? {self.ossq_rnames[p]} : {self.issq_rnames[p]};')
        
        k.emit()

    
    def _write_to_global_memory(self, k):
        pars = self.params
        P, Dout, tout, Tout, Mout = pars.P, pars.Dout, pars.tout_rname, pars.Tout_rname, pars.Mout_rname

        k.emit('// PfReducer: write out/ssq to global memory')
        k.emit()

        if Dout > 1:
            k.emit(f'int {tout}_next = tout + (32/Dout);   // value of tout in next iteration of loop')
            k.emit(f'if (({tout}_next == {Tout}) || (({tout}_next & 0x1f) == 0)) {{   // current **warp** writes max/ssq to global memory')
            
            k.emit(f'// PfReducer: transpose output time indices')
            k.emit(f'// Before the transpose, {(32//Dout)=} "inner" time indices are assigned to "outer" threads')
            k.emit(f'// and {Dout=} "outer" time indices are assigned to "inner" threads.')

            itmp_rname = k.get_tmp_rname('int')
            k.emit(f'{itmp_rname} = (threadIdx.x & 0x1f) * Dout;')
            k.emit(f'{itmp_rname} = {itmp_rname} | ({itmp_rname} >> 5);')
            
            for r in (self.omax_rnames + self.ossq_rnames):
                k.emit(f'{r} = __shfl_sync(0xffffffff, {r}, {itmp_rname});')

            k.emit()
            k.emit(f'{tout}_next = (tout & 0x1f) + (32/Dout);  // advance by (32/Dout)')
            k.emit(f'if ((threadIdx.x & 0x1f) < {tout}_next) {{   // current **thread** writes max/ssq to global memory')

        k.emit('// PfReducer: Write to global memory.')
        k.emit('// Reminder: output max/ssq arrays have shape (P,Mout,Tout).')

        for p in range(P):
            s = f'{p}*{Tout}*{Mout}' if (p > 0) else '0'
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
