# FIXME it would be nice to have a systematic unit test for the Ringbuf classes.
# FIXME this source file could use more comments.

class Ringbuf:
    def __init__(self, dtype):
        """
        Consider a situation where we are processing a timestream in 32-element chunks,
        so that thread 0 <= th < 32 holds x[t] at time index t = (32*i + th).

        Suppose we want to apply lags, so that thread 'th' also holds x[32*i + th - lag].
        This can partly be accomplished with warp shuffles, but we need a per-warp data
        structure ("class Ringbuf") to store the overlaps ('lag' registers per warp),
        and store the overlaps in global GPU memory between kernel launches.

        Example usage:

           rb = Ringbuf(dt32)
        
           # Near beginning of kernel: create splice.
           # Ring buffer initialization code will be emitted into the splice.
           krb = k.splice()

           for (time loop) {
               rb.advance(k, rname1, nelts1, dst=None, comment=True)
               rb.advance(k, rname2, nelts2, dst=None, comment=True)
                 ...
               rb.advance_outer()
           }

           # Slightly confusing: 'pstate_warp' is a pointer that is defined later in
           # the code generator (see below), but appears earlier in the generated code
           # (via the splice).
        
           rb.finalize(k, pwarp_rname = 'pstate_warp')
        
           # After calling finalize(): go back and emit initialization code into splice.
           # This example code will probably need to be customized for a particular kernel.
           # Note that the definition of 'pstate_warp' is emitted here.
           # Note 'krb' throughout, not 'k'!

           krb.emit(f'constexpr int RB32 = {rb.get_n32_per_warp()};      // 32-bit registers per warp')
           krb.emit(f'int bw = blockIdx.x * blockDim.y + threadIdx.y;    // warp index within kernel')
           krb.emit(f'{dt32} *pstate_warp = (dt32 *)pstate_ + bw*RB32;   // pstate with per-warp offset applied')
           rb.initialize(krb, 'pstate_warp')
        """

        # Check against accidentially specifying a non-32-bit dtype (such as __half).
        # (Note: the current implementation should be correct for a non-32-bit dtype,
        # but suboptimal. Specifying a non-32-bit dtype is probably unintentional, so
        # we treat it as an error for now.)
        
        assert dtype in [ 'float', '__half2' ]
        
        self.dtype = dtype
        self.nreg = 0
        self.rb_pos = [ ]        # List of length self.nreg
        self.rb_nelts = [ ]      # List of length self.nreg 
        self.rb_rnames = [ ]     # List of length self.nreg ('rb0', 'rb1', etc.)
        self.lflag_nelts = None  # Value of 'nelts' in last call to _swap_upper()
        self.advance_outer_called = False
        
        zdict = { 'float': '0.0f', '__half2': '__float2half2_rn(0.0f)' }
        self.zero = zdict[dtype]

        
    def advance(self, k, rname, nelts, dst=None, comment=True):
        """The 'nelts' arg is the lag."""
        
        assert 0 <= nelts <= 32
        assert not self.advance_outer_called
        
        if nelts == 0:
            if dst is not None:
                k.emit(f'{dst} = {rname};')
            return

        if comment:
            k.emit(f'// Ringbuf.advance(): {rname}, {nelts=}, {dst=}')
        
        if dst is None:
            tmp = k.get_tmp_rname()
            k.emit(f"{self.dtype} {tmp};")
            self.advance(k, rname, nelts, dst=tmp, comment=False)
            k.emit(f"{rname} = {tmp};")
            return
            
        # Note that _get_r() does not increment rb_nelts[r].
        r = self._get_r(nelts)
        rb_rname = self.rb_rnames[r]

        # Cycle ring buffer (if needed)
        self._cycle_register(k, rb_rname, 32 - self.rb_nelts[r] + self.rb_pos[r])
        self.rb_pos[r] = self.rb_nelts[r]
        self.rb_nelts[r] += nelts
            
        self._blend(k, dst, rname, rb_rname, nelts)
        self._blend(k, rb_rname, rb_rname, rname, nelts)
        self._cycle_register(k, dst, 32-nelts)


    def advance2(self, k, rname, nelts1, nelts2):
        """For use in dedispersion kernels."""

        assert nelts1 >= 0
        assert nelts2 >= 0
        assert (nelts1 + nelts2) <= 32
        assert not self.advance_outer_called

        if True:
            # FIXME temporary kludge
            dst0, dst1 = k.get_tmp_rname(2)
            k.emit(f'{self.dtype} {dst0}, {dst1};')
            self.advance(k, rname, nelts1, dst=dst1, comment=False)
            self.advance(k, dst1, nelts2, dst=dst0, comment=False)
            return dst0, dst1

        # FIXME code below is currently incorrect.
        # Note that _get_r() does not increment rb_nelts[r].
        r = self._get_r(nelts+1)   # note +1 here
        rb_rname = self.rb_rnames[r]

        # Cycle ring buffer (if needed)
        self._cycle_register(k, rb_rname, 32 - self.rb_nelts[r] + self.rb_pos[r])
        self.rb_pos[r] = self.rb_nelts[r]
        self.rb_nelts[r] += (nelts + 1)    # note +1 here
        
        self._blend(k, dst1, rname, rb_rname, nelts)
        self._cycle_register(k, dst1, 32-nelts)
        
        self._blend(k, dst0, rname, rb_rname, nelts+1)
        self._cycle_register(k, dst0, 32-nelts-1)
        
        self._blend(k, rb_rname, rb_rname, rname, nelts+1)
        return dst0, dst1


    def advance_outer(self, k):
        assert not self.advance_outer_called
        self.advance_outer_called = True
        
        k.emit("// Ringbuf.advance_outer()")        
        for r in range(self.nreg):
            self._cycle_register(k, self.rb_rnames[r], self.rb_pos[r])

        k.emit()


    def get_n32_per_warp(self):
        assert self.advance_outer_called
        return sum(self.rb_nelts)
    

    def initialize(self, k, pwarp_rname):
        """The 'pwarp_rname' arg is a pointer (in global memory or shared) with per-warp offsets applied."""
        
        assert self.advance_outer_called
        
        if self.nreg == 0:
            return
        
        laneId = k.get_tmp_rname()
        k.emit(f'const int {laneId} = (threadIdx.x & 0x1f);  // laneId')

        pos0 = 0    # current position relative to 'pwarp_rname'
        for r in range(self.nreg):
            lane0 = 32 - self.rb_nelts[r] 
            pos1 = pos0 - lane0   # total offset, to be added to laneId

            s = f'{pos1:+}' if (pos1 != 0) else ''
            s = f'{pwarp_rname}[{laneId}{s}]'
            s = f'({laneId} >= {lane0}) ? {s} : {self.zero}' if (lane0 != 0) else s

            k.emit(f'{self.dtype} {self.rb_rnames[r]} = {s};')
            pos0 += self.rb_nelts[r]

        assert pos0 == self.get_n32_per_warp()

    
    def finalize(self, k, pwarp_rname):
        assert self.advance_outer_called
        
        if self.nreg == 0:
            return
            
        laneId = k.get_tmp_rname()
        k.emit(f'const int {laneId} = (threadIdx.x & 0x1f);  // laneId')

        pos0 = 0    # current position relative to 'pwarp_rname'
        for r in range(self.nreg):
            lane0 = 32 - self.rb_nelts[r] 
            pos1 = pos0 - lane0   # total offset, to be added to laneId

            s = f'{pos1:+}' if (pos1 != 0) else ''
            s = f'{pwarp_rname}[{laneId}{s}]'

            if lane0 != 0:
                k.emit(f'if ({laneId} >= {lane0})')

            k.emit(f'{s} = {self.rb_rnames[r]};')
            pos0 += self.rb_nelts[r]
        
        assert pos0 == self.get_n32_per_warp()

        
    def _get_r(self, nelts):
        assert 0 < nelts <= 32
        best_score = -1
        ret = None

        # Simple heuristic: assign to largest rb which is not overfull.
        for r in range(self.nreg):
            if (self.rb_nelts[r] + nelts > 32):
                continue

            score = self.rb_nelts[r]

            if best_score < score:
                best_score = score
                ret = r

        if ret is None:
            # Add register to ring buffer.
            ret = self.nreg
            rb_name = f'rb_{self.nreg}'
            self.rb_rnames.append(rb_name)
            self.rb_nelts.append(0)
            self.rb_pos.append(0)
            self.nreg += 1
        
        return ret

    
    def _cycle_register(self, k, rname, nelts):
        assert 0 <= nelts <= 32
        
        if 0 < nelts < 32:
            k.emit(f'{rname} = __shfl_sync(0xffffffff, {rname}, threadIdx.x + {nelts});')
                                 

    def _blend(self, k, dst_rname, src_rname1, src_rname2, nelts):
        assert 0 < nelts <= 32
        assert src_rname1 != src_rname2

        if nelts == 32:
            k.emit(f'{dst_rname} = {src_rname2};')
            return
        
        if self.lflag_nelts != nelts:
            decl = 'bool ' if (self.lflag_nelts is None) else ''
            k.emit(f'{decl}rb_lflag = ((threadIdx.x & 0x1f) < {32-nelts});')

        self.lflag_nelts = nelts
        k.emit(f'{dst_rname} = rb_lflag ? {src_rname1} : {src_rname2};')
