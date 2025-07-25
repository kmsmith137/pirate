class Ringbuf:
    def __init__(self):
        self.nreg = 0
        self.rb_pos = [ ]        # List of length self.nreg
        self.rb_nelts = [ ]      # List of length self.nreg 
        self.rb_rnames = [ ]     # List of length self.nreg ('rb0', 'rb1', etc.)
        self.lflag_nelts = None  # Value of 'nelts' in last call to _swap_upper()
        self.advance_outer_called = False

        
    def advance(self, k, rname, nelts, dst=None):
        assert 0 <= nelts <= 32
        assert not self.advance_outer_called
        
        if nelts == 0:
            return

        # Note that _get_irb() does not increment rb_nelts[irb].
        irb = self._get_irb(nelts)
        rb_rname = self.rb_rnames[irb]

        k.emit(f'// Ringbuf.advance(): {rname}, {nelts=}, {dst=}')
        self._cycle_register(k, rb_rname, 32 - self.rb_nelts[irb] + self.rb_pos[irb])
        self.rb_pos[irb] = self.rb_nelts[irb]
        self.rb_nelts[irb] += nelts

        dst_rname = dst if (dst is not None) else k.get_tmp_rname('float')
        self._blend(k, dst_rname, rname, rb_rname, nelts)
        self._blend(k, rb_rname, rb_rname, rname, nelts)
        self._cycle_register(k, dst_rname, 32-nelts)

        if dst is None:
            k.emit(f'{rname} = {dst_rname};')


    def advance_outer(self, k):
        assert not self.advance_outer_called
        self.advance_outer_called = True
        
        k.emit("// Ringbuf.advance_outer()")        
        for i in range(self.nreg):
            self._cycle_register(k, self.rb_rnames[i], self.rb_pos[i])

        k.emit()


    def get_nelts_per_warp(self):
        assert self.advance_outer_called
        return sum(self.rb_nelts)
    

    def initialize(self, k, pwarp_rname):
        """The 'pwarp_rname' arg is a pointer (in global memory or shared) with per-warp offsets applied."""
        
        assert self.advance_outer_called
        
        laneId = k.get_tmp_rname('int')
        k.emit(f'{laneId} = (threadIdx.x & 0x1f);  // laneId')

        pos0 = 0    # current position relative to 'pwarp_rname'
        for i in range(self.nreg):
            lane0 = 32 - self.rb_nelts[i] 
            pos1 = pos0 - lane0   # total offset, to be added to laneId

            s = f'{pos1:+}' if (pos1 != 0) else ''
            s = f'{pwarp_rname}[{laneId}{s}]'
            s = f'({laneId} >= {lane0}) ? {s} : 0.0f' if (lane0 != 0) else s

            k.emit(f'float {self.rb_rnames[i]} = {s};')
            pos0 += self.rb_nelts[i]

    
    def finalize(self, k, pwarp_rname):
        assert self.advance_outer_called
        
        laneId = k.get_tmp_rname('int')
        k.emit(f'{laneId} = (threadIdx.x & 0x1f);  // laneId')

        pos0 = 0    # current position relative to 'pwarp_rname'
        for i in range(self.nreg):
            lane0 = 32 - self.rb_nelts[i] 
            pos1 = pos0 - lane0   # total offset, to be added to laneId

            s = f'{pos1:+}' if (pos1 != 0) else ''
            s = f'{pwarp_rname}[{laneId}{s}]'

            if lane0 != 0:
                k.emit(f'if ({laneId} >= {lane0})')

            k.emit(f'{s} = {self.rb_rnames[i]};')
            pos0 += self.rb_nelts[i]
        
        
    def _get_irb(self, nelts):
        assert 0 < nelts <= 32
        irb = None

        # Simple heuristic: assign to largest rb which is not overfull.
        for i in range(self.nreg):
            if (self.rb_nelts[i] + nelts > 32):
                continue
            if (irb is None) or (self.rb_nelts[i] > self.rb_nelts[irb]):
                irb = i

        if irb is None:
            # Add register to ring buffer.
            irb = self.nreg
            rb_name = f'rb_{irb}'
            self.rb_rnames.append(rb_name)
            self.rb_nelts.append(0)
            self.rb_pos.append(0)
            self.nreg += 1
        
        return irb

                                 
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

        lflag_rname = 'rb_lflag'
        if self.lflag_nelts is None:
            k.emit(f'bool {lflag_rname};')
        if self.lflag_nelts != nelts:
            k.emit(f'{lflag_rname} = ((threadIdx.x & 0x1f) < {32-nelts});')

        self.lflag_nelts = nelts
        k.emit(f'{dst_rname} = {lflag_rname} ? {src_rname1} : {src_rname2};')
