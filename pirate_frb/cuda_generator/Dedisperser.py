from . import utils

from .Kernel import Kernel
from .Ringbuf import Ringbuf


class DedisperserParams:
    def __init__(self, dtype, rank):
        assert dtype == 'float'   # for now
        assert 1 <= rank <= 4     # for now
        
        self.dtype = dtype
        self.rank = rank


class Dedisperser:
    def __init__(self, params):
        assert isinstance(params, DedisperserParams)

        if params.dtype == 'float':
            self.dt32 = 'float'
            self.nbits = 32
        elif params.dtype == '__half':
            self.dt32 = '__half2'
            self.nbits = 16
        
        self.params = params
        self.rank = params.rank
        self.ringbuf = Ringbuf()
        self.kernel_name = f'dd_fp{self.nbits}_r{params.rank}'
        self.warps_per_threadblock = 1   # XXX for now
        self.shmem_nbytes = 0            # XXX for now

    
    def emit_global(self, k):
        assert isinstance(k, Kernel)
        
        rank, dt32 = self.rank, self.dt32
        W = self.warps_per_threadblock
        B = utils.xdiv(16, W)   # ?

        k.emit(f'// Launch with {{ 32, {W} }} threads/warp')
        k.emit(f'// Launch with {{ Namb, Nbeams}} threadblocks')
        k.emit()
        k.emit(f'__global__ void __launch_bounds__({32*W},{B})')
        k.emit(f'{self.kernel_name}(void *inbuf_, int act_istride32, int amb_istride32, long beam_istride32,')
        k.emit(f'       void *outbuf_, int act_ostride32, int amb_ostride32, long beam_ostride32,')
        k.emit(f'       void *pstate_, int ntime, long rb_pos)')
        k.emit('{')

        # Save splice, for code to load the ring buffer.
        # (This code must be emitted after the main loop.)
        k_rb = k.splice()

        k.emit(f'// Apply per-thread offsets to inbuf (including laneId offset).')
        k.emit(f'{dt32} *inbuf = ({dt32} *) inbuf_;')
        k.emit(f'inbuf += long(amb_istride32) * long(blockIdx.x);    // ambient = blockIdx.x')
        k.emit(f'inbuf += long(beam_istride32) * long(blockIdx.y);   // beam = blockIdx.y')
        k.emit(f'inbuf += threadIdx.x;   // laneId')
        k.emit()
        
        k.emit(f'// Apply per-thread offsets to outbuf (including laneId offset).')
        k.emit(f'{dt32} *outbuf = ({dt32} *) outbuf_;')
        k.emit(f'outbuf += long(amb_ostride32) * long(blockIdx.x);    // ambient = blockIdx.x')
        k.emit(f'outbuf += long(beam_ostride32) * long(blockIdx.y);   // beam = blockIdx.y')
        k.emit(f'outbuf += threadIdx.x;   // laneId')
        k.emit()

        dt = (1024 // self.nbits)
        k.emit(f'for (int itime = 0; itime < ntime; itime += {dt}) {{')

        k.emit('// Load data from global memory into registers')
        rnames = [ f'x{i}' for i in range(2**rank) ]
        for i,rname in enumerate(rnames):
            s = f'{i} * act_istride32' if (i > 0) else '0'
            k.emit(f'{dt32} {rname} = inbuf[{s}];')

        # Dedisperse registers.
        for i in range(rank):
            self._dedisperse_registers(k, rnames, i)

        k.emit('\n//Store data from registers to global memory')
        for i,rname in enumerate(rnames):
            s = f'{i} * act_ostride32' if (i > 0) else '0'
            k.emit(f'outbuf[{s}] = {rname};')

        k.emit('\n// Advance ring buffer for next iteration of loop')
        self.ringbuf.advance_outer(k)
        
        k.emit('\n// Advance input/output pointers')
        k.emit('inbuf += 32;')
        k.emit('outbuf += 32;')
        
        k.emit('}   // outer time loop')

        self.pstate32_per_warp = self.ringbuf.get_n32_per_warp()
        self.pstate32_per_small_tree = self.pstate32_per_warp  # XXX for now
        
        self._load_ringbuf(k_rb)
        self._save_ringbuf(k)
        
        k.emit('}   // kernel')
        
            
    def _dedisperse_registers(self, k, rnames, i):
        rank = self.rank
        assert len(rnames) == 2**rank
        assert 0 <= i < rank

        k.emit(f'\n// dedisperse_registers: pass {i} starts here')
        
        # Outer loop is a spectator index.
        for s in range(0, 2**rank, 2**(i+1)):
            for j in range(2**i):
                r0 = rnames[s+j]
                r1 = rnames[s+j+2**i]
                lag = utils.bit_reverse(j,i)
                k.emit(f'// dedisperse {r0}, {r1} with lag {lag}')

                tmp0, tmp1 = self.ringbuf.advance2(k, r0, lag)
                k.emit(f'{r0} = {tmp1} + {r1};')
                k.emit(f'{r1} += {tmp0};')

    
    def _load_ringbuf(self, k):
        k.emit("// Apply per-warp offset to pstate, in preparation for loading")
        k.emit(f'constexpr int P32 = {self.pstate32_per_warp};    // ring buffer 32-bit elements per warp')
        k.emit(f'int blockId = (gridDim.x * blockIdx.y) + blockIdx.x;')
        k.emit(f'{self.dt32} *pstate = ({self.dt32} *)pstate_;')
        k.emit(f"pstate += blockId * P32;")
        k.emit()

        k.emit('// Read ring buffer directly from global memory.')
        k.emit('// FIXME: would it be better to go through shared memory?')
        
        self.ringbuf.initialize(k, 'pstate')
        k.emit()

        
    def _save_ringbuf(self, k):
        P32 = self.ringbuf.get_n32_per_warp()
        assert P32 > 0
        
        k.emit('// Write ring buffer to global memory.')
        k.emit('// FIXME: would it be better to go through shared memory?')
        k.emit()

        self.ringbuf.finalize(k, 'pstate')
