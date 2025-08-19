from . import utils

from .Kernel import Kernel
from .Ringbuf import Ringbuf


class DedisperserParams:
    def __init__(self, dtype, rank):
        assert dtype == 'float'   # for now
        assert 1 <= rank <= 8
        
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
        self.dtype = params.dtype
        self.rank = params.rank
        self.ringbuf = Ringbuf()
        self.kernel_name = f'dd_fp{self.nbits}_r{params.rank}'
        self.two_stage = (params.rank >= 5)

        if self.two_stage:
            self.rank0 = (params.rank // 2)
            self.rank1 = params.rank - self.rank0
            self.warps_per_threadblock = 2**self.rank0
            self.shmem_nbytes = 4 * self.rb_base(2**self.rank0, 0)
            self.static_asserts()
        else:
            # FIXME single-stage kernel should use 4 warps/threadblock
            self.warps_per_threadblock = 1
            self.shmem_nbytes = 0

    
    def emit_global(self, k):
        assert isinstance(k, Kernel)
        
        W = self.warps_per_threadblock
        B = utils.xdiv(16, W)   # threadblocks per SM (FIXME rethink?)

        # Number of data registers (x0, x1, ...)
        nx = (2**self.rank1) if self.two_stage else (2**self.rank) 
        xnames = [ f'x{i}' for i in range(nx) ]

        k.emit(f'// Launch with {{ 32, {W} }} threads/warp')
        k.emit(f'// Launch with {{ Namb, Nbeams}} threadblocks')
        k.emit()

        if False:   # debug
            self._show_shmem_layout_in_comment(k)
            
        k.emit(f'__global__ void __launch_bounds__({32*W},{B})')
        k.emit(f'{self.kernel_name}(void *inbuf_, long beam_istride32, int amb_istride32, int act_istride32,')
        k.emit(f'       void *outbuf_, long beam_ostride32, int amb_ostride32, int act_ostride32,')
        k.emit(f'       void *pstate_, int ntime, ulong nt_cumul)')
        k.emit('{')

        self._apply_inbuf_offsets(k)    # apply per-thread offsets to 'inbuf' (including laneId offset)
        self._apply_outbuf_offsets(k)   # apply per-thread offsets to 'outbuf' (including laneId offset)
        self._init_control_words(k)     # no-ops if (self.two_stage) is False

        # Save splice, for code to load pstate.
        # This code must be emitted near the end, after the ring buffer layout is finalized.
        k_pstate = k.splice()
        
        dt = (1024 // self.nbits)
        k.emit(f'for (int itime = 0; itime < ntime; itime += {dt}) {{')

        k.emit('// Load data from global memory into registers')
        for i,x in enumerate(xnames):
            s = f'{i} * act_istride32' if (i > 0) else '0'
            k.emit(f'{self.dt32} {x} = inbuf[{s}];')

        if self.two_stage:
            self._two_stage_dedispersion_core(k, xnames)
        else:
            self._single_stage_dedispersion_core(k, xnames)

        k.emit('\n//Store data from registers to global memory')
        for i,x in enumerate(xnames):
            # Correct for both single-stage and two-stage
            r = (2**self.rank0) if self.two_stage else 1
            s = f'{i*r} * act_ostride32' if (i > 0) else '0'
            k.emit(f'outbuf[{s}] = {x};')

        k.emit('\n// Advance ring buffer for next iteration of loop')
        self.ringbuf.advance_outer(k)

        # No-ops if (self.two-stage) is False
        self._advance_control_words(k)
        
        k.emit('\n// Advance input/output pointers')
        k.emit('inbuf += 32;')
        k.emit('outbuf += 32;')
        
        k.emit('}   // outer time loop')

        # Now that the ring buffer layout is finalized, we can emit the pstate code.
        self.rb32_per_warp = self.ringbuf.get_n32_per_warp()
        self.pstate32_per_small_tree = utils.xdiv(self.shmem_nbytes,4) + (self.rb32_per_warp * self.warps_per_threadblock)
        
        self._load_pstate(k_pstate)
        self._save_pstate(k)
        
        k.emit('}   // kernel')
        
    
    def _single_stage_dedispersion_core(self, k, xnames):
        for i in range(self.rank):
            self._dedispersion_pass(k, xnames, i)

        
    def _two_stage_dedispersion_core(self, k, xnames):
        for i in range(self.rank0):
            self._dedispersion_pass(k, xnames, i)

        k.emit()
        k.emit('__syncthreads();')
        k.emit()
        k.emit('// Write dd registers to shared memory')

        for i,x in enumerate(xnames):
            s = self._rbloc(k, i)
            k.emit(f'shmem[{s}] = {x};')

        k.emit()
        k.emit('__syncthreads();')
        k.emit()
        k.emit('// Read dd registers from shared memory')
        
        for i,x in enumerate(xnames):
            s = self._rbloc(k, 16+i)
            k.emit(f'{x} = shmem[{s}];')
        
        for i in range(self.rank1):
            self._dedispersion_pass(k, xnames, i)

    
    def _dedispersion_pass(self, k, xnames, i):
        assert (len(xnames) % 2**(i+1)) == 0

        k.emit()
        k.emit(f'// dedispersion_pass: pass {i} starts here')
        
        # Outer loop is a spectator index.
        for s in range(0, len(xnames), 2**(i+1)):
            for j in range(2**i):
                x0 = xnames[s+j]
                x1 = xnames[s+j+2**i]
                lag = utils.bit_reverse(j,i)
                k.emit(f'// dedisperse {x0}, {x1} with lag {lag}')

                tmp0, tmp1 = self.ringbuf.advance2(k, x0, lag)
                k.emit(f'{x0} = {tmp1} + {x1};')
                k.emit(f'{x1} += {tmp0};')

        k.emit(f'// dedispersion_pass: pass {i} ends here')


    def _apply_inbuf_offsets(self, k):
        k.emit(f'// Apply per-thread offsets to inbuf (including laneId offset).')
        k.emit(f'{self.dt32} *inbuf = ({self.dt32} *) inbuf_;')
        k.emit(f'inbuf += long(beam_istride32) * long(blockIdx.y);   // beam = blockIdx.y')
        k.emit(f'inbuf += long(amb_istride32) * long(blockIdx.x);    // ambient = blockIdx.x')
        
        if self.two_stage:
            k.emit(f'inbuf += {2**self.rank1} * long(act_istride32) * long(threadIdx.y);   // warpId = threadIdx.y')
            
        k.emit(f'inbuf += threadIdx.x;   // laneId')
        k.emit()


    def _apply_outbuf_offsets(self, k):
        k.emit(f'// Apply per-thread offsets to outbuf (including laneId offset).')
        k.emit(f'{self.dt32} *outbuf = ({self.dt32} *) outbuf_;')
        k.emit(f'outbuf += long(beam_ostride32) * long(blockIdx.y);   // beam = blockIdx.y')
        k.emit(f'outbuf += long(amb_ostride32) * long(blockIdx.x);    // ambient = blockIdx.x')
        
        if self.two_stage:
            k.emit(f'outbuf += long(act_ostride32) * long(threadIdx.y);   // warpId = threadIdx.y')
            
        k.emit(f'outbuf += threadIdx.x;   // laneId')
        k.emit()

    
    def _init_control_words(self, k):
        if not self.two_stage:
            k.emit('// Note: no control words, this is a single-stage kernel')
            return
        
        rank0, rank1 = self.rank0, self.rank1
        
        k.emit('// Init control words for two-stage kernel.')
        k.emit('// A ring buffer "control word" consists of:')
        k.emit('//')
        k.emit('//   uint15 rb_base;   // base shared memory location of ring buffer (in 32-bit registers)')
        k.emit('//   uint9  rb_pos;    // current position, satisfying 0 <= rb_pos < (rb_lag + 32)')
        k.emit('//   uint8  rb_lag;    // ring buffer lag (in 32-bit registers), note that capacity = lag + 32.')
        k.emit('//')
        k.emit('// Depending on context, rb_pos may point to either the end of the buffer')
        k.emit('// (writer thread context), or be appropriately lagged (reader thread context).')
        k.emit('//')
        k.emit(f'// Lanes [0:{2**rank1}] hold control words for registers in the first pass:')
        k.emit(f'//')
        k.emit(f'//   - registers form a ({2**(rank1-rank0)}, {2**rank0}) array, where first index is a "fast"')
        k.emit(f'//     frequency 0 <= ff < {2**(rank1-rank0)} and second index is 0 <= dm_brev < {2**rank0}')
        k.emit(f'//   - warpId is a "slow" frequency 0 <= fs < {2**rank0}')
        k.emit(f'//   - writer thread context')
        k.emit(f'//')
        k.emit(f'// Lanes [16:16+{2**rank1}] hold control words for registers in the second pass:')
        k.emit(f'//')
        k.emit(f'//   - register index is a frequency 0 <= f < {2**rank1})')
        k.emit(f'//   - warpId is 0 <= dm_brev < {2**rank0}')
        k.emit(f'//   - reader thread context')
        k.emit()

        first_pass, dm_brev, dm, f, frev = k.get_tmp_rname(5)

        dr = rank1 - rank0
        fpass1 = 'threadIdx.y' if (dr==0) else f'(threadIdx.y << {dr}) | ((threadIdx.x >> {rank0}) & {2**dr-1})'

        k.emit(f'// Compute frev, dm.')
        k.emit(f'bool {first_pass} = (threadIdx.x < 16);   // first_pass')
        k.emit(f'uint {dm_brev} = {first_pass} ? (threadIdx.x & {2**rank0-1}) : threadIdx.y;   // dm_brev')
        k.emit(f'uint {dm} = __brev({dm_brev}) >> {32-rank0};   // dm')
        k.emit(f'uint {f} = {first_pass} ? ({fpass1}) : (threadIdx.x & {2**rank1-1});  // f')
        k.emit(f'uint {frev} = {2**rank1-1} - {f};   // frev')
        k.emit()

        k.emit(f'// Compute (rb_lag, rb_base) from (frev, dm).')
        k.emit(f'// This part uses some cryptic, highly optimized code from Dedisperser.rb_base()')

        if self.dtype == 'float':
            a = 2**(rank1-1) * (2**rank1-1)
            b = 2**(rank1+6) - a
            
            t, rb_base, rb_lag = k.get_tmp_rname(3)
            k.emit(f'uint {t} = {a}*{dm} + {frev}*({frev}-1) + {b};')
            k.emit(f'uint {rb_base} = (({dm}*{t}) >> 1) + ({frev} << 5);  // rb_base')
            k.emit(f'uint {rb_lag} = {dm}*{frev};  // rb_lag')

        elif self.dtype == '__half':
            a = (2**rank1 - 1) * 2**(rank1-1)
            b = (256 - 2**rank1) * 2**(rank1-1)
            c = 2**(rank1-1) + 1

            u, t, rb_base, rb_lag = k.get_tmp_rname(4)
            k.emit(f'uint {u} = {a}*{dm} + {frev}*({frev}-1) + {b};')
            k.emit(f'uint {t} = ({dm}*{u}) + ({dm} & 1) * ({c}-{frev});')
            k.emit(f'uint {rb_base} = ({t} >> 2) + ({frev} << 5);  // rb_base')
            k.emit(f'uint {rb_lag} = ({dm}*{frev}) >> 1;  // rb_lag')

        else:
            raise RuntimeError('bad dtype')

        k.emit()
        k.emit(f'// Compute rb_pos, and assemble control word from (rb_lag, rb_base)')

        t, rb_pos = k.get_tmp_rname(2)
        k.emit(f'uint {t} = {first_pass} ? 0 : 32;')
        k.emit(f'uint {rb_pos} = (nt_cumul + {t}) % ({rb_lag} + 32U);   // rb_pos')
        k.emit(f'uint control_word = {rb_base} | ({rb_pos} << 15) | ({rb_lag} << 24);')
        k.emit()


    def _rbloc(self, k, control_lane):
        w, rb_base, rb_pos, rb_size, wraparound, ret = k.get_tmp_rname(6)
        k.emit(f'uint {w} = __shfl_sync(0xffffffff, control_word, {control_lane});  // rbloc({control_lane=}) starts here')
        k.emit(f'uint {rb_base} = ({w} & 0x7fff);        // rb_base, 15 bits')
        k.emit(f'uint {rb_pos} = (({w} >> 15) & 0x1ff) + threadIdx.x;   // rb_pos, 9 bits (note laneId at end)')
        k.emit(f'uint {rb_size} = ({w} >> 24) + 32;      // rb_size, 8 bits (note +32 here to convert rb_lag -> rb_size')
        k.emit(f'uint {wraparound} = ({rb_pos} >= {rb_size}) ? {rb_size} : 0;  // wraparound')
        k.emit(f'uint {ret} = {rb_base} + {rb_pos} - {wraparound};  // rbloc({control_lane=}) return value')
        return ret


    def _advance_control_words(self, k):
        if not self.two_stage:
            k.emit('// Note: no control words, this is a single-stage kernel')
            return

        pos15, lag15, dpos15 = k.get_tmp_rname(3)
        
        k.emit(f'// Advance control words')
        k.emit(f'int {pos15} = (control_word & 0xff8000);  // pos15')
        k.emit(f'int {lag15} = ((control_word >> 9) & 0xff8000);  // lag15')
        k.emit(f'int {dpos15} = ({pos15} >= {lag15}) ? {lag15} : (-(32 << 15));  // dpos15')
        k.emit(f'control_word = (control_word & 0xff007fff) | ({pos15} - {dpos15});')
        k.emit()

    
    def _load_pstate(self, k):
        k.emit("// Apply per-warp offset to pstate, in preparation for loading")
        k.emit(f'constexpr int RB32 = {self.rb32_per_warp};    // 32-bit elements per **warp**, in **register** ring buffer')
        k.emit(f'constexpr int SM32 = {utils.xdiv(self.shmem_nbytes,4)};  // 32-bit elements per **threadblock**, in **shmem** ring buffer')
        k.emit(f'constexpr int PS32 = SM32 + {self.warps_per_threadblock} * RB32;   // total persistent state per **threadblock**')
        k.emit()
        k.emit(f'int blockId = (gridDim.x * blockIdx.y) + blockIdx.x;')
        k.emit(f'{self.dt32} *pstate = ({self.dt32} *)pstate_;')
        k.emit(f"pstate += blockId * PS32;")
        k.emit()

        if self.two_stage:
            k.emit(f'extern __shared__ {self.dt32} shmem[];')
            k.emit()
            k.emit(f'// Load shmem ring buffer from global memory')
            k.emit(f'// NOTE no __syncthreads() here')
            k.emit(f'for (int s = 32*threadIdx.y + threadIdx.x; s < SM32; s += {32*self.warps_per_threadblock})')
            k.emit(f'    shmem[s] = pstate[s];')
            k.emit('__syncwarp();')
            k.emit()
            
        k.emit('// Read register ring buffer directly from global memory.')
        k.emit('// FIXME: would it be better to go through shared memory?')
        
        self.ringbuf.initialize(k, self._warp_pstate(k))
        k.emit()
        

    def _save_pstate(self, k):
        if self.two_stage:
            k.emit(f'// Write shmem ring buffer to global memory')
            k.emit(f'for (int s = 32*threadIdx.y + threadIdx.x; s < SM32; s += {32*self.warps_per_threadblock})')
            k.emit(f'    pstate[s] = shmem[s];')
            k.emit(f'__syncwarp();')
            k.emit()
            
        k.emit('// Write register ring buffer to global memory.')
        k.emit('// FIXME: would it be better to go through shared memory?')
        k.emit()
        
        self.ringbuf.finalize(k, self._warp_pstate(k))
        k.emit()


    def _warp_pstate(self, k):
        if (self.shmem_nbytes == 0) and (self.warps_per_threadblock == 1):
            return 'pstate'
        
        ps = k.get_tmp_rname()
        k.emit(f'{self.dt32} *{ps} = pstate + SM32 + (threadIdx.y * RB32);  // per-warp ring buffer in global memory')
        return ps


    def _show_shmem_layout_in_comment(self, k):
        if not self.two_stage:
            return
        
        for dm in range(2**self.rank0):
            for frev in range(2**self.rank1):
                rb_lag = (frev * dm * self.nbits) // 32
                rb_size = rb_lag + 32
                rb_base = self.rb_base(dm, frev)
                k.emit(f'//   {dm=} {frev=}: {rb_lag=} {rb_size=} {rb_base=}')
        k.emit()
    
    
    def static_asserts(self):
        assert self.two_stage
        expected_pos = 0
        
        for dm in range(2**self.rank0):
            for frev in range(2**self.rank1):
                assert self.rb_base(dm,frev) == expected_pos
                lag = (frev * dm * self.nbits) // 32
                expected_pos += (lag + 32)

        assert self.shmem_nbytes == (expected_pos * 4)
        # print(f'static_asserts(rank0={self.rank0}, rank1={self.rank1}): pass')
        

    def rb_base(self, dm, frev):
        """Called by static_asserts(). Also called directly in constructor for shmem_nbytes."""

        assert self.two_stage
        rank0, rank1 = self.rank0, self.rank1
        assert 0 <= dm <= 2**rank0   # note <=
        assert 0 <= frev < 2**rank1  # note <
        assert rank1 >= 2            # assumed below as noted

        if self.dtype == 'float':
            # "Intuitive form"
            # n1 = 2**rank1
            # term1 = ((dm*(dm-1))//2) * ((n1*(n1-1))//2)   # d < dm, all f
            # term2 = dm * ((frev*(frev-1)) // 2)           # d == dm, f < frev
            # term3 = 32 * (dm*n1 + frev)
            # return term1 + term2 + term3
            
            # "Fast form" with slightly fewer ops (assumes rank1 >= 1)
            a = 2**(rank1-1) * (2**rank1-1)  # known at compile time
            b = 2**(rank1+6) - a             # known at compile time
            t = a*dm + frev*(frev-1) + b
            return ((dm*t) >> 1) + (frev << 5)

        elif self.dtype == '__half':
            # "Intuitive form" (assumes rank1 >= 2)
            # n1 = 2**rank1
            # term1 = ((dm*(dm-1))//2) * ((n1*(n1-1)) // 4)    # d < dm, all f
            # term1 -= (dm//2) * (n1//4)                       # corrections from odd d
            # term2 = (dm * frev * (frev-1)) // 4              # d == dm, f < frev
            # term2 -= (dm & 1) * (frev // 4)                  # correction if dm is odd
            # term3 = 32 * (dm*n1 + frev)
            # return term1 + term2 + term3

            # "Fast form" with slightly fewer ops (assumes rank1 >= 2)
            a = (2**rank1 - 1) * 2**(rank1-1)     # known at compile time
            b = (256 - 2**rank1) * 2**(rank1-1)   # known at compile time
            c = 2**(rank1-1) + 1                  # known at compile time
            u = a*dm + frev*(frev-1) + b
            t = (dm*u) + (dm & 1) * (c-frev)
            return (t >> 2) + (frev << 5)
            
        else:
            raise RuntimeError('bad dtype')
