# FIXME it would be nice to have a systematic unit test for the Ringbuf classes.
# FIXME this source file could use more comments.x

class Ringbuf:
    def __init__(self):
        """
        Consider a situation where we are processing a timestream in 32-element chunks,
        so that thread 0 <= th < 32 holds x[t] at time index t = (32*i + th).

        Suppose we want to apply lags, so that thread 'th' also holds x[32*i + th - lag].
        This can partly be accomplished with warp shuffles, but we need a per-warp data
        structure ("class Ringbuf") to store the overlaps ('lag' registers per warp),
        and store the overlaps in global GPU memory between kernel launches.
        """
        
        self.nreg = 0
        self.rb_pos = [ ]        # List of length self.nreg
        self.rb_nelts = [ ]      # List of length self.nreg 
        self.rb_rnames = [ ]     # List of length self.nreg ('rb0', 'rb1', etc.)
        self.lflag_nelts = None  # Value of 'nelts' in last call to _swap_upper()
        self.advance_outer_called = False

        
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
            k.emit(f"float {tmp};")
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


    def advance2(self, k, rname, nelts):
        """For use in dedispersion kernels."""

        assert 0 <= nelts < 31
        assert not self.advance_outer_called
        
        dst0, dst1 = k.get_tmp_rname(2)
        k.emit(f'float {dst0}, {dst1};')

        # FIXME
        self.advance(k, rname, nelts, dst=dst1, comment=False)
        self.advance(k, dst1, 1, dst=dst0, comment=False)
        return dst0, dst1
        
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
        
        laneId = k.get_tmp_rname()
        k.emit(f'const int {laneId} = (threadIdx.x & 0x1f);  // laneId')

        pos0 = 0    # current position relative to 'pwarp_rname'
        for r in range(self.nreg):
            lane0 = 32 - self.rb_nelts[r] 
            pos1 = pos0 - lane0   # total offset, to be added to laneId

            s = f'{pos1:+}' if (pos1 != 0) else ''
            s = f'{pwarp_rname}[{laneId}{s}]'
            s = f'({laneId} >= {lane0}) ? {s} : 0.0f' if (lane0 != 0) else s

            k.emit(f'float {self.rb_rnames[r]} = {s};')
            pos0 += self.rb_nelts[r]

        assert pos0 == self.get_n32_per_warp()

    
    def finalize(self, k, pwarp_rname):
        assert self.advance_outer_called
        
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
            self.kernel.emit(f'{dst_rname} = {src_rname2};')
            return
        
        if self.lflag_nelts != nelts:
            decl = 'bool ' if (self.lflag_nelts is None) else ''
            k.emit(f'{decl}rb_lflag = ((threadIdx.x & 0x1f) < {32-nelts});')

        self.lflag_nelts = nelts
        k.emit(f'{dst_rname} = rb_lflag ? {src_rname1} : {src_rname2};')


#################################################################################################


class Ringbuf16:
    def __init__(self):
        """
        The Ringbuf16 is similar to the Ringbuf, but applies in a situation where:

          - the timestream is float16
          - the simd bit 'b' of each register is mapped to b <-> ti5

        That is, we are processing a float16 timestream in 64-element chunks, where
        thread 0 <= th < 32 holds a __half2 containing { x[64*i+th], x[64*i+32+th] }.

        The Ringbuf16 applies lags, so that thread 'th' also holds a __half2 with
        { x[64*i + th - lag], x[64*i + 32 + th - lag] }.
        """
        
        self.nreg = 0
        self.rb_pos = [ ]        # List of length self.nreg, consisting of pairs (pos0,pos1)
        self.rb_nelts = [ ]      # List of length self.nreg, consisting of pairs (pos0,pos1)
        self.rb_rnames = [ ]     # List of length self.nreg ('rb0', 'rb1', etc.)
        self.lflag_nelts = None  # Value of 'nelts' in last call to _swap_upper()
        self.advance_outer_called = False

        
    def advance(self, k, rname, nelts, dst=None, comment=True):
        assert 0 <= nelts <= 32
        assert not self.advance_outer_called
        
        if nelts == 0:
           return

        if comment:
            k.emit(f'// Ringbuf.advance(): {rname}, {nelts=}, {dst=}')
        
        if dst is None:
            tmp = k.get_tmp_rname()
            k.emit(f"__half2 {tmp};")
            self.advance(k, rname, nelts, dst=tmp, comment=False)
            k.emit(f"{rname} = {tmp};")
            return
            
        # Note that _get_rb() does not increment rb_nelts[r][b]
        r, b = self._get_rb(nelts)
        rb_rname = self.rb_rnames[r]

        # Cycle ring buffer (if needed)
        dp = self.rb_nelts[r][b] - self.rb_pos[r][b]
        self._cycle_register(k, rb_rname, 32 - dp)   # no-op if dp=0 or dp=32

        self.rb_pos[r] = (self.rb_pos[r][0] + dp, self.rb_pos[r][1] + dp)
        self.rb_nelts[r] = (self.rb_nelts[r][0] + (1-b)*nelts, self.rb_nelts[r][1] + b*nelts)
        assert self.rb_pos[r][1-b] <= self.rb_nelts[r][1-b]

        # Blend (data, ring buffer) -> (wrapped data)
        control_word = '0x1076' if b else '0x1054'
        self._blend(k, dst, rname, rb_rname, nelts, control_word)

        # Blend (ring buffer, data) -> (updated ring buffer)
        control_word = '0x7610' if b else '0x3276'
        self._blend(k, rb_rname, rb_rname, rname, nelts, control_word)
        
        self._cycle_register(k, dst, 32-nelts)   # no-op if nelts=32


    def advance_outer(self, k):
        assert not self.advance_outer_called
        self.advance_outer_called = True
        
        k.emit("// Ringbuf.advance_outer()")
        
        for r in range(self.nreg):
            p0, p1 = self.rb_pos[r]
            rname = self.rb_rnames[r]
            
            # Equalize p0, p1
            if p0 != p1:
                dp = abs(p0-p1)
                tmp = k.get_tmp_rname()
                blend_args = (rname,tmp) if (p0 < p1) else (tmp,rname)
                k.emit(f'__half2 {tmp} =  __shfl_sync(0xffffffff, {rname}, threadIdx.x + {dp});')
                k.emit(f'{rname} = ksgpu::f16_blend({blend_args[0]}, {blend_args[1]});')
            
            self._cycle_register(k, rname, min(p0,p1))

        k.emit()


    def get_n32_per_warp(self):
        assert self.advance_outer_called
        return sum(max(n0,n1) for n0,n1 in self.rb_nelts)
    

    def initialize(self, k, pwarp_rname):
        """The 'pwarp_rname' arg is a pointer (in global memory or shared) with per-warp offsets applied."""
        
        assert self.advance_outer_called
        
        laneId = k.get_tmp_rname()
        k.emit(f'const int {laneId} = (threadIdx.x & 0x1f);  // laneId')

        pos0 = 0    # current position relative to 'pwarp_rname'
        for r in range(self.nreg):
            n = max(self.rb_nelts[r][0], self.rb_nelts[r][1])
            lane0 = 32 -n
            pos1 = pos0 - lane0   # total offset, to be added to laneId

            s = f'{pos1:+}' if (pos1 != 0) else ''
            s = f'{pwarp_rname}[{laneId}{s}]'
            s = f'({laneId} >= {lane0}) ? {s} : __float2half2_rn(0.0f)' if (lane0 != 0) else s

            k.emit(f'__half2 {self.rb_rnames[r]} = {s};')
            pos0 += n

        assert pos0 == self.get_n32_per_warp()

    
    def finalize(self, k, pwarp_rname):
        assert self.advance_outer_called
        
        laneId = k.get_tmp_rname()
        k.emit(f'const int {laneId} = (threadIdx.x & 0x1f);  // laneId')

        pos0 = 0    # current position relative to 'pwarp_rname'
        for r in range(self.nreg):
            n = max(self.rb_nelts[r][0], self.rb_nelts[r][1])
            lane0 = 32 - n
            pos1 = pos0 - lane0   # total offset, to be added to laneId

            s = f'{pos1:+}' if (pos1 != 0) else ''
            s = f'{pwarp_rname}[{laneId}{s}]'

            if lane0 != 0:
                k.emit(f'if ({laneId} >= {lane0})')

            k.emit(f'{s} = {self.rb_rnames[r]};')
            pos0 += n

        assert pos0 == self.get_n32_per_warp()

            
    def _get_rb(self, nelts):
        """Returns (r,b) pair."""
        
        assert 0 < nelts <= 32
        best_score = -1
        rb = None

        # Heuristic logic for choosing (r,b) pair.
        for r in range(self.nreg):
            n = self.rb_nelts[r]
            p = self.rb_pos[r]

            # If I choose register 'r', which simd lane 'b' will I use?
            if (n[0]-p[0]) > (n[1]-p[1]):
                b = 1
            elif (n[0]-p[0]) < (n[1]-p[1]):
                b = 0
            elif n[0] > n[1]:
                b = 1
            else:
                b = 0

            if (n[b] + nelts) > 32:
                continue  # register 'r' is overfull

            score = n[1-b]
            if (n[b] + nelts) < n[1-b]:
                score += 100
            if (n[b] + nelts) == n[1-b]:
                score += 200
            
            if best_score < score:
                best_score = score
                rb = (r,b)

        if rb is None:
            # Add register to ring buffer.
            rb = (self.nreg, 0)
            rb_name = f'rb_{self.nreg}'
            self.rb_rnames.append(rb_name)
            self.rb_nelts.append((0,0))
            self.rb_pos.append((0,0))
            self.nreg += 1
        
        return rb

    
    def _cycle_register(self, k, rname, nelts):
        assert 0 <= nelts <= 32
        
        if 0 < nelts < 32:
            k.emit(f'{rname} = __shfl_sync(0xffffffff, {rname}, threadIdx.x + {nelts});')

    
    def _blend(self, k, dst_rname, src_rname1, src_rname2, nelts, control_word):
        """The 'control_word' should be '0x{HI}{LO}', where LO, HI are:

           - 10 for low 16 bits of src1
           - 32 for high 16 bits of src1
           - 54 for low 16 bits of src2
           - 76 for high 16 bits of src2
        """
        
        assert 0 < nelts <= 32
        assert src_rname1 != src_rname2

        if nelts == 32:
            k.emit(f'{dst_rname} = ksgpu::f16_perm({src_rname1}, {src_rname2}, {control_word});')
            return

        if self.lflag_nelts != nelts:
            decl = 'bool ' if (self.lflag_nelts is None) else ''
            k.emit(f'{decl}rb_lflag = ((threadIdx.x & 0x1f) < {32-nelts});')

        self.lflag_nelts = nelts
        k.emit(f'{dst_rname} = ksgpu::f16_perm({src_rname1}, {src_rname2}, rb_lflag ? 0x3210 : {control_word});')
