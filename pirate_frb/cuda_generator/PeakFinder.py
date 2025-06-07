import itertools

from . import utils
from .Ringbuf import Ringbuf


class PeakFinder:
    def __init__(self, M, E, Dcore, Dout, W, BlocksPerSM):
        """
        Compile-time parameters are (M, E, Dout, Dcore, W):

           M = number of "rows" (e.g. trial DMs) per warp
           E = max kernel width
           Dout = output time downsampling factor
           Dcore = internal downsampling (high Dcore is more efficient, but uses more registers)
           W = warps per threadblock
        """

        assert M >= 1
        assert E <= 32
        assert (W % 4) == 0
        assert 1 <= Dcore <= Dout <= 32
        assert 4 <= (W * BlocksPerSM) <= 32
        
        assert utils.is_power_of_two(E)
        assert utils.is_power_of_two(Dout)
        assert utils.is_power_of_two(Dcore)
        
        self.M = M
        self.E = E
        self.W = W
        self.Dout = Dout
        self.Dcore = Dcore
        self.BlocksPerSM = BlocksPerSM

        self.kernel_name = f'pf_kernel_M{M}_E{E}_Dcore{Dcore}_Dout{Dout}_W{W}_B{BlocksPerSM}'
        self.reducer = PfReducer(M, E, Dcore, Dout, W, 'tout', 'Tout', 'out_max', 'out_ssq', 'out_pstride', 'wt_sh')
        self.ringbuf = Ringbuf()
        self.P = self.reducer.P

        self.pf = PfCore(self.reducer, self.ringbuf)
        while self.pf.Din > 1:
            self.pf = PfTransposeLayer(M, self.pf.Din//2, self.pf)
        
        
    def emit_global(self, k):
        M, E, Dout, Dcore, W, P = self.M, self.E, self.Dout, self.Dcore, self.W, self.P

        k.emit('// Kernel args are (out_max, out_ssq, pstate, in, wt, Mout, Tout).')
        k.emit('//')
        k.emit("//    - 'out_max' and 'out_ssq' have shape (B, P, Mout, Tout).")
        k.emit("//    - 'pstate' has shape (B, Mout, RW), where RW is defined below.")
        k.emit("//    - 'in' has shape (B, Mout*M, Tout*Dout).")
        k.emit("//    - 'wt' has shape (B, P, Mout*M).")
        k.emit("//")
        k.emit('// Each warp processes one output 0 <= mout < Mout, and all times 0 <= tout < Tout.')
        k.emit('//')
        k.emit('// Launch with {BM,B,1} blocks, where BM = ceil(Mout/W), and B is number of beams.')
        k.emit('// Launch with {32*W,1} threads.')
        k.emit()
        k.emit(f'__global__ void __launch_bounds__({32*W}, {self.BlocksPerSM})')
        k.emit(f'{self.kernel_name}(float *out_max, float *out_ssq, float *pstate, const float *in, const float *wt, int Mout, int Tout)')
        k.emit('{')
        
        k.emit(f'constexpr int {M = };      // number of "rows" (e.g. trial DMs) per warp')
        k.emit(f'constexpr int {E = };      // max kernel width')
        k.emit(f'constexpr int {Dout = };   // output time downsampling factor')
        k.emit(f'constexpr int {Dcore = };  // output time downsampling factor')
        k.emit(f'constexpr int {W = };      // warps per threadblock')
        k.emit(f'constexpr int {P = };      // number of peak-finding kernels')
        k.emit()

        k.emit("int b = blockIdx.y;  // beam index")
        k.emit("int mblock = blockIdx.x * W;              // base m-index of block")
        k.emit("int mout = mblock + (threadIdx.x >> 5);   // output m-index of warp")
        k.emit()
        k.emit("bool warp_active = (mout < Mout);")
        k.emit("mout = warp_active ? mout : (M-1);")
        k.emit()
        
        k.emit('// Shared memory layout:')
        k.emit(f'//  - {P*W*M} weights (=P*W*M)')
        k.emit(f'//  - {4*P*W*M} bytes')
        k.emit()

        k.emit(f'__shared__ float shmem[{P*W*M}];')
        k.emit()

        k.emit("// Apply per-beam offset to 'wt', in preparation for loading weights.")
        k.emit("// Shape is (B, P, Mout*M).")
        k.emit()
        k.emit("wt += b * (P * Mout * M);")
        k.emit()
        
        k.emit("// Copy weights from global memory to shared.")
        k.emit("// The destination array has shape (P, W*M), contiguously packed.")
        k.emit("// Thie source array is a discontiguous subset of a shape (B, P, Mout*M) source array.")
        k.emit("// FIXME this loop could be optimized a little.")
        k.emit()
        k.emit("for (int i = threadIdx.x; i < P*W*M; i += 32*W) {")
        k.emit("    int p = i / (W*M);")
        k.emit("    int m = i - (W*M)*p + mblock;")
        k.emit("    shmem[i] = wt[p*Mout + min(m,M-1)];")
        k.emit("}")
        k.emit()
        k.emit('__syncthreads();')
        k.emit()
        k.emit('// Per-warp weights array in shared memory')
        k.emit('float *wt_sh = shmem + mout * M;')
        k.emit()
        
        k.emit("// Apply per-thread offsets to 'out_max' and 'out_ssq'.")
        k.emit("// Shapes are (B, P, Mout, Tout).")
        k.emit()
        k.emit("int out_woff = (b * P * Mout + mout) * Tout + (threadIdx.x & 0x1f);")
        k.emit("out_max += woff;")
        k.emit("out_ssq += woff;")
        k.emit()

        k.emit("// Apply per-thread offsets to 'in'.")
        k.emit("// Shape is (B, Mout*M, Tout*Dout)")
        k.emit()
        k.emit("int in_mstride = Tout * Dout;")
        k.emit(f"in += (b*Mout + mout) * M * in_mstride + (threadIdx.x & 0x1f);")
        k.emit()

        # Save splice, for code to load the ring buffer.
        # (This code must be emitted after the main lloop.)
        k_rb = k.splice()

        self.reducer.initialize(k)

        k.emit(f'// PeakFinder: t-loop advances by {(32//Dout)=}')
        k.emit(f"for (int tout = 0; tout < Tout; tout += {32//Dout}) {{")
        k.emit("    if (!warp_active)")
        k.emit("        continue;")
        k.emit()

        for m in range(M):
            rname = f'x{m}'
            s = f'{m} * in_mstride' if (m > 0) else '0'
            k.emit(f'// PeakFinder: {m=}\n')
            k.emit(f"float {rname} = in[{s}];")
            self.pf.advance(k, [rname], m)
            k.emit()

        k.emit()
        self.ringbuf.advance_outer(k)
        
        k.emit("}  // end of t-loop")
        k.emit()

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
    def __init__(self, M, Din, next_layer):
        """
        PfTransposeLayer.advance() is called for m = 0, Din, (2*Din), ...
        Each call consists of Din registers which represent consecutive times.
        The innermost Din thread indices correspond to m, (m+1), ... (m+Din-1).
        The outer thread indices correspond to (time/Din).

        Doubles Din, and calls next_layer.advance(), which expects Din -> (2*Din).
        """
        
        self.M = M
        self.Din = Din
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
        self.M = reducer.M
        self.E = reducer.E
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
    def __init__(self, M, E, Din, Dout, W, tout_rname, Tout_rname, out_max_rname, out_ssq_rname, out_pstride_rname, wt_rname):
        """
        Calls to PfReducer.advance() should be embedded in a loop where tout increases by (32/Dout) in each iteration.
        The 'wt' pointer should be a shared memory array of shape (P,W*M), with no offsets applied (i.e. per-block);
        """

        assert M >= 1
        assert E <= 32
        assert (W % 4) == 0
        assert 1 <= Din <= Dout <= 32
        
        assert utils.is_power_of_two(E)
        assert utils.is_power_of_two(Din)
        assert utils.is_power_of_two(Dout)

        self.M = M
        self.E = E
        self.W = W
        self.P = 3 * utils.integer_log2(E) + 1
        self.Din = Din
        self.Dout = Dout
        
        
        self.tout_rname = tout_rname
        self.Tout_rname = Tout_rname
        self.out_max_rname = out_max_rname
        self.out_ssq_rname = out_ssq_rname
        self.out_pstride_rname = out_pstride_rname
        self.wt_rname = wt_rname

        # Persistent registers.
        self.pmax_rnames = [ f'pf_pmax_{p}' for p in range(self.P) ]
        self.pssq_rnames = [ f'pf_pssq_{p}' for p in range(self.P) ]

    
    def initialize(self, k):
        k.emit('// Persistent registers for PeakFinder')
        for r in (self.pmax_rnames + self.pssq_rnames):
            k.emit(f'float {r} = 0.0f;')
        k.emit()

    
    def advance(self, k, max_rname, ssq_rname, m, p):
        M, P, W, Din, Dout = self.M, self.P, self.W, self.Din, self.Dout
        tout_rname, Tout_rname, wt_rname = self.tout_rname, self.Tout_rname, self.wt_rname

        k.emit(f'\n// PfReducer.advance({m=},{p=}) starts here: {max_rname=} {ssq_rname=}')

        decl = 'float ' if (m==p==0) else ''
        woff = p*W*M + m
        s = f'threadIdx.x & {(Din-1)}'
        s = f'({s}) + {woff}' if woff else s
        
        k.emit('// Apply weights')
        k.emit(f'{decl}wt = {wt_rname}[{s}];')
        k.emit(f'{max_rname} *= wt;')
        k.emit(f'{ssq_rname} *= (wt*wt);')

        # k.emit(f'// PfReducer: Reduce over m-indices (by a factor {Din=}), and time indices (by a factor {(Dout/Din)=}).')
        # FIXME reduction logic throughout this function is suboptimal! Can use fewer
        # __shfl_sync() instructions, by "coalescing" logic across multiple calls to advance().

        mpop = min(M-m, Din)   # number of populated m-indices
        d = 1                  # current downsampling level
        
        while d < Dout:
            if (d >= Din) or ((mpop & d) == 0):
                # Case 1: this is the typical case, where we can "cleanly" reduce d -> (2*d).
                k.emit(f'{max_rname} = max({max_rname}, __shfl_sync(0xffffffff, {max_rname}, threadIdx.x ^ {d}));')
                k.emit(f'{ssq_rname} += __shfl_sync(0xffffffff, {ssq_rname}, threadIdx.x ^ {d});')
                d *= 2

            elif d >= mpop:
                # Case 2: can finish m-reduction with just __shfl_sync() (no max/sum ops).
                k.emit(f'{max_rname} = __shfl_sync(0xffffffff, {max_rname}, threadIdx.x & {32-Din});')
                k.emit(f'{ssq_rname} = __shfl_sync(0xffffffff, {ssq_rname}, threadIdx.x & {32-Din});')
                d = Din    # not (d *= 2)

            else:
                # Case 3: m-reduction d -> (2*d) mixes populated/unpopulated indices.
                tmp_rname = k.get_tmp_rname('float')
                btmp_rname = k.get_tmp_rname('bool')
                k.emit(f'{btmp_rname} = (threadIdx.x & {Din-1}) < {mpop-d}')
                k.emit(f'{tmp_rname} = max({max_rname}, __shfl_sync(0xffffffff, {max_rname}, threadIdx.x ^ {d}));')
                k.emit(f'{max_rname} = {btmp_rname} ? {tmp_rname} : {max_rname};')
                k.emit(f'{tmp_rname} = {ssq_rname} + __shfl_sync(0xffffffff, {ssq_rname}, threadIdx.x ^ {d}));')
                k.emit(f'{ssq_rname} = {btmp_rname} ? {tmp_rname} : {ssq_rname};')
                d *= 2

        k.emit('// PfReducer: "Absorb" reduced registers into persistent registers.')
        flag_rname = 'pf_tflag'

        if m == p == 0:
            k.emit('// PfReducer: set {flag_rname} on threads where (threadIdx.x * (32/Dout)) == tout (mod 32)')
            k.emit('// PfReducer: these are the threads where persistent max/ssq will be updated in this iteration of the t-loop')
            k.emit(f'bool {flag_rname} = (((threadIdx.x * {32//Dout}) ^ {tout_rname}) & 0x1f) == 0;')
        
        k.emit(f'{self.pmax_rnames[p]} = {flag_rname} ? {self.pmax_rnames[p]} : {max_rname};')
        k.emit(f'{self.pssq_rnames[p]} = {flag_rname} ? {self.pssq_rnames[p]} : {ssq_rname};')
        k.emit(f'// PfReducer.advance({m=},{p=}) ends here')
        k.emit()
        
        if (m+Din < M) or (p < P-1):
            return

        k.emit(f'\n// PfReducer: last call to advance() in the outer t-loop.')
        k.emit(f"// PfReducer: write to global memory (maybe).\n")

        if Dout > 1:
            itmp_rname = k.get_tmp_rname('int')
            k.emit(f'{itmp_rname} = tout + {32//Dout};  // advance by (32/Dout)')
            k.emit(f'if (({itmp_rname} == {self.Tout_rname}) || !({itmp_rname} & 0x1f)) {{   // current **warp** writes max/ssq to global memory')
            
            k.emit(f'// PfReducer: transpose ({(32//Dout)=}) inner time indices, with {Dout=} outer time indices')            
            k.emit(f'{itmp_rname} = (threadIdx.x << ) | ((threadIdx.x >> ) & 0x1f);')
            
            for r in (self.pmax_rnames + self.pssq_rnames):
                k.emit(f'{r} = __shfl_sync(0xffffffff, {r}, {itmp_rname});')

            k.emit(f'{itmp_rname} = (tout & 0x1f) + {32//Dout};  // advance by (32/Dout)')
            k.emit(f'if ((threadIdx.x & 0x1f) < {itmp_rname}) {{   // current **thread** writes max/ssq to global memory')

        k.emit('// PfReducer: Write to global memory')
        k.emit('// Reminder: output max/ssq arrays have shape (P,Mout,Tout).')

        for p in range(P):
            s = f'{p}*{self.out_pstride_rname}' if (p > 0) else '0'
            k.emit(f'{self.out_max_rname}[{s}] = {self.pmax_rnames[p]};')
            k.emit(f'{self.out_ssq_rname}[{s}] = {self.pssq_rnames[p]};')
        
        if Dout > 1:
            k.emit('}  // "if current **thread** writes max/ssq to global memory..."')

        k.emit('// PfReducer: advance max/ssq global memory pointers by 32 output time samples')
        k.emit(f'{self.out_max_rname} += 32;')
        k.emit(f'{self.out_ssq_rname} += 32;')

        if Dout > 1:
            k.emit('}  // "if current **warp* writes max/ssq to global memory..."')
