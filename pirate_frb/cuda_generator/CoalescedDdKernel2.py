import os
import re
import numpy as np

from . import utils

from .Dtype import Dtype
from .Kernel import Kernel
from .Ringbuf import Ringbuf
from .FrequencySubbands import FrequencySubbands
from .Dedisperser import Dedisperser, MultiDedisperser
from .PeakFinder import PeakFinder


class CoalescedDdKernel2:
    def __init__(self, dtype, dd_rank, frequency_subbands, Wmax, Dcore, Dout, Tinner):
        """Coalesced Dedisperser + PeakFinder.

        We currently assume apply_input_residual_lags == input_is_ringbuf == True.
        """

        self.dtype = dtype = Dtype(dtype)
        self.frequency_subbands = frequency_subbands                                            
        self.dd_rank = dd_rank
        self.Tinner = Tinner
        self.Dcore = Dcore
        self.Dout = Dout
        self.Wmax = Wmax

        self.dt32 = dtype.simd32
        self.SW = dtype.simd_width
        self.rb = Ringbuf(self.dt32)  # simd dtype, not scalar dtype

        self.dd = Dedisperser(
            dtype = dtype,
            rank = dd_rank,
            apply_input_residual_lags = True,
            input_is_ringbuf = True,
            output_is_ringbuf = False,
            nspec = 1,
            ringbuf = self.rb   # note that ringbuf is shared between Dedisperser and PeakFinder
        )

        self.xdm_rank = K = self.dd.xdm_rank(frequency_subbands)   # (dd.rank1 - fs.pf_rank)

        # The peak-finder sees one input per (multiplet, "extra DM") pair, indexed by
        # m_ext = (m << xdm_rank) | mu -- see Dedisperser.emit_subband_extraction(). Prepending
        # 'xdm_rank' zeros to the subband counts gives a FrequencySubbands whose multiplet index
        # IS m_ext: same N, same band ordering, and each band owns a run of 2^xdm_rank times as
        # many consecutive multiplets. That is all the peak-finder layer uses, so it needs no
        # notion of 'mu' at all.
        #
        # WARNING: the two FrequencySubbands objects describe different band sets. Only the
        # compact one is geometrically meaningful (prepending zeros turns the "unstaggered"
        # level-0 bands into half-overlapping level-K bands, so n_to_flo/n_to_fhi and m_to_d
        # both change). The dedisperser, the registry key and the C++ side all use the compact
        # one; only 'self.pf' uses the extended one.
        self.pf_frequency_subbands = FrequencySubbands([0]*K + list(frequency_subbands.subband_counts))

        self.pf = PeakFinder(
            dtype = dtype,
            frequency_subbands = self.pf_frequency_subbands,
            Wmax = Wmax,
            Dcore = Dcore,
            Dout = Dout,
            Tinner = Tinner,
            ringbuf = self.rb   # note that ringbuf is shared between Dedisperser and PeakFinder
        )

        # These restrictions may be relaxed in the future.
        assert self.dd.two_stage
        assert self.frequency_subbands.pf_rank <= self.dd.rank1
        assert self.pf_frequency_subbands.pf_rank == self.dd.rank1

        # From Dedisperser
        self.warps_per_threadblock = self.dd.warps_per_threadblock
        self.shmem_nbytes = self.dd.shmem_nbytes
        self.nt_per_segment = self.dd.nt_per_segment

        # From PeakFinder
        self.P = self.pf.P
        self.M_ext = self.pf.M   # = 2^xdm_rank * fs.M (emitted as 'constexpr int M', see emit_kernel)
        self.Minner = self.pf.Minner
        self.weight_layout = self.pf.weight_layout

        # Typical kernel name: cdd2_fp32_r7_n11_n6_n3_n1_W16_Dcore8_Dout16_Tinner1
        self.kernel_name = f'cdd2_{dtype.fname}_r{dd_rank}_{frequency_subbands.fstr}_W{Wmax}_Dcore{Dcore}_Dout{Dout}_Tinner{Tinner}'
        self.kernel_basename = self.kernel_name + '.cu'

        # For testing: if a peak-finding kernel is precompiled, we also precompile unit tests
        # for the associated sub-kernels. (In the case of the dedispersion sub-kernel, we precompile
        # the MultiDedisperser that contains the sub-kernel.)

        mdd = MultiDedisperser(
            dtype = self.dd.dtype,
            apply_input_residual_lags = self.dd.apply_input_residual_lags, 
            input_is_ringbuf = self.dd.input_is_ringbuf, 
            output_is_ringbuf = self.dd.output_is_ringbuf, 
            nspec = self.dd.nspec
        )

        self.all_kernel_basenames = [ self.kernel_basename, mdd.kernel_basename ] + self.pf.all_kernel_basenames


    def emit_kernel(self, k):
        """Emits the complete kernel, including prologue, body, and registry registration."""
        
        assert isinstance(k, Kernel)

        # ---------------  Prologue  ---------------
        
        k.emit('#include "../../include/pirate/CoalescedDdKernel2.hpp"')
        k.emit('#include "../../include/pirate/FrequencySubbands.hpp"')
        k.emit('#include "../../include/pirate/inlines.hpp"')
        k.emit('#include <ksgpu/device_fp16.hpp>   // f16_perm(), f16_align()')
        k.emit()
        k.emit('using namespace std;')
        k.emit('using namespace ksgpu;')
        k.emit()
        k.emit('namespace pirate {')
        k.emit('#if 0')
        k.emit('}  // editor auto-indent')
        k.emit('#endif')
        k.emit()

        # ---------------  CUDA kernel  ---------------
        
        # launch_bounds
        lb_warps = self.warps_per_threadblock
        lb_blocks = utils.xdiv(16, lb_warps)

        k.emit(f'// Autogenerated by pirate_frb.cuda_generator')
        k.emit()
        k.emit(f'// Launch with {{ 32, {self.warps_per_threadblock} }} threads/warp')
        k.emit(f'// Launch with {{ Namb, Nbeams }} threadblocks')
        k.emit()

        k.emit(f'__global__ void __launch_bounds__({32 * lb_warps}, {lb_blocks})')
        k.emit(f'{self.kernel_name}(')
        k.emit(f'    void *grb_base_, uint *grb_quads_, long grb_frame0,    // dedisperser input (ring buffer)')
        k.emit(f'    void *out_max_, uint *out_argmax, const void *wt_,     // peak-finder output')
        k.emit(f'    uint out_max_bstride32, uint out_argmax_bstride32,     // output beam-strides (multiples of 32 bits)')
        k.emit(f'    void *pstate_, int ntime,                              // shared between dedisperser and peak-finder')
        k.emit(f'    ulong nt_cumul, bool is_downsampled_tree,              // dedisperser')
        k.emit(f'    uint ndm_out_per_wt, uint nt_in_per_wt)                // peak-finder')
        k.emit('{')
        k.emit(f'constexpr int M = {self.M_ext};')
        k.emit(f'constexpr int Dout = {self.Dout};')
        k.emit(f'constexpr int Dcore = {self.Dcore};')
        k.emit(f'constexpr int Minner = {self.Minner};')
        k.emit(f'constexpr int Tinner = {self.Tinner};')
        k.emit(f'constexpr int wt_touter_stride32 = {utils.xdiv(self.pf.wt_touter_byte_stride,4)};')
        k.emit()

        self.dd._emit_rsqrt2(k)

        if self.Wmax > 1:
            k.emit(f'const {self.dt32} pf_a = {self.dtype.from_float("0.5f")};')
            k.emit()

        self.dd._apply_inbuf_offsets(k)    # operates on grb* pointers, since self.dd.input_is_ringbuf == True

        # No call to self.dd._apply_outbuf_offsets() in coalesced kernel,
        # Instead, call self._apply_pfout_offsets(), which operates on out_max, out_argmax, wt pointers.
        self._apply_pfout_offsets(k)

        # No-ops if self.two_stage == False.
        self.dd._init_srb(k)

        # PfWeightReader.top() is currently a placeholder that does not emit any code.
        self.pf.pf_weight_reader.top(k, 'wt')

        # Save splice, for code to load pstate.
        # This code must be emitted near the end, after the ring buffer layout is finalized.
        # Note that pstate is managed by the dedispersion kernel.
        k_pstate = k.splice()

        k.emit(f'for (int itime = 0; itime < ntime; itime += {self.nt_per_segment}) {{')

        self.dd._load_input_data(k)              # behaves differently, depending on self.input_is_ringbuf
        self.dd._apply_input_residual_lags(k)    # no-ops if (self.apply_input_residual_lags) is False

        k.emit()
        k.emit("// The rest of this loop body is the second half of the dedispersion transform,")
        k.emit("// with the peak-finding logic interleaved into it.")
        k.emit("// FIXME I'm a bit worried that this way of doing it will lead to register pressure")
        k.emit("// I made a note to revisit this later.")
        k.emit()

        # Dedisperser.emit_subband_extraction() emits the entire second stage, and yields one
        # (register name, multiplet index, "extra DM" index) triple per peak-finder input.
        # Note that it must be fully consumed -- see its docstring, and the 'sbx_complete'
        # assert below.

        for rname, m, mu in self.dd.emit_subband_extraction(k, self.frequency_subbands):
            self.pf.process_pf_input(k, rname, (m << self.xdm_rank) | mu)

        assert self.dd.sbx_complete

        # No call to self.dd._save_output_data() in coalesced kernel.
        # Instead, call self.pf.process_pf_outputs()
        self.pf.pf_output.apply_outer(k, 'out_max', 'out_argmax', 'itime', 'ntime')
        self.pf.pf_weight_reader.bottom(k, 'itime', 'nt_in_per_wt')

        self.dd._advance_rrb(k)
        self.dd._advance_srb(k)      # no-ops if (self.two_stage) is False
        self.dd._advance_inbuf(k)    # behaves differently, depending on self.input_is_ringbuf
        # No call to self.dd._advance_outbuf() in coalesced kernel.
        
        k.emit('}   // outer time loop')

        # Now that 'self.rrb' has been finalized, we can sort out the pstate.
        self.dd._lay_out_pstate(k_pstate)    # initializes some members, including self.dd.pstate32_per_small_tree
        self.dd._load_pstate(k_pstate)
        self.dd._save_pstate(k)

        k.emit('    // placeholder')
        k.emit('}  // end of cuda kernel')
        

    def emit_registration(self, k):
        fs = self.frequency_subbands
        wl = self.weight_layout
        sb_counts = ', '.join(str(int(x)) for x in fs.subband_counts)
        m_to_n = ', '.join(str(int(n)) for n,d in fs.m_to_nd)
        m_to_d = ', '.join(str(int(d)) for n,d in fs.m_to_nd)
        n_to_flo = ', '.join(str(int(flo)) for flo,fhi in fs.n_to_frange)
        n_to_fhi = ', '.join(str(int(fhi)) for flo,fhi in fs.n_to_frange)

        k.emit('\n// Boilerplate to register the kernel when the library is loaded.')
        k.emit('namespace {')
        k.emit('struct register_hack {')
        k.emit('register_hack() {')
        k.emit('CoalescedDdKernel2::RegistryKey k;')
        k.emit(f'k.dtype = ksgpu::Dtype::native<{self.dtype.scalar}>();')
        k.emit(f'k.dd_rank = {self.dd_rank};')
        k.emit(f'k.subband_counts = {{ {sb_counts} }};')
        k.emit(f'k.Tinner = {self.Tinner};')
        k.emit(f'k.Dout = {self.Dout};')
        k.emit(f'k.Wmax = {self.Wmax};')
        k.emit()
        k.emit('CoalescedDdKernel2::RegistryValue v;')
        k.emit(f'v.cuda_kernel = {self.kernel_name};')
        k.emit(f'v.Dcore = {self.Dcore};')
        k.emit(f'v.shmem_nbytes = {self.shmem_nbytes};')
        k.emit(f'v.warps_per_threadblock = {self.warps_per_threadblock};')
        k.emit(f'v.pstate32_per_small_tree = {self.dd.pstate32_per_small_tree};')
        k.emit(f'v.nt_per_segment = {self.nt_per_segment};')
        k.emit()
        k.emit(f'v.pf_weight_layout.dtype = ksgpu::Dtype::native<{self.dtype.scalar}>();')
        k.emit(f'v.pf_weight_layout.N = {fs.N};')
        k.emit(f'v.pf_weight_layout.P = {self.P};')
        k.emit(f'v.pf_weight_layout.Pouter = {wl.Pouter};')
        k.emit(f'v.pf_weight_layout.Pinner = {wl.Pinner};')
        k.emit(f'v.pf_weight_layout.Tinner = {self.Tinner};')
        k.emit(f'v.pf_weight_layout.touter_byte_stride = {wl.touter_byte_stride};')
        k.emit(f'v.pf_weight_layout.validate();  // throws an exception if anything is wrong')
        k.emit()
        k.emit('// Checks consistency of python/C++ FrequencySubbands')
        k.emit(f'FrequencySubbands fs({{ {sb_counts} }});')
        k.emit(f'xassert_eq(fs.N, {fs.N});')
        k.emit(f'xassert_eq(fs.M, {fs.M});')
        k.emit(f'xassert(vec_equal(fs.m_to_n, {{ {m_to_n} }}));')
        k.emit(f'xassert(vec_equal(fs.m_to_d, {{ {m_to_d} }}));')
        k.emit(f'xassert(vec_equal(fs.n_to_flo, {{ {n_to_flo} }}));')
        k.emit(f'xassert(vec_equal(fs.n_to_fhi, {{ {n_to_fhi} }}));')
        k.emit()
        k.emit('bool debug = false;')
        k.emit('CoalescedDdKernel2::registry().add(k, v, debug);')
        k.emit('} // register_hack constructor')
        k.emit('}; // struct register_hack')
        k.emit('register_hack hack;')
        k.emit('} // anonymous namespace')
        k.emit()
        k.emit('}   // namespace pirate')


    @classmethod
    def _idiv(cls, var, n):
        """Helper for _apply_pfout_offsets()."""
        return f'({var} >> {utils.integer_log2(n)})' if (n != 1) else var


    def _apply_pfout_offsets(self, k):
        dt32, SW, Dout = self.dt32, self.dtype.simd_width, self.Dout

        k.emit(f'// CoalescedDdKernel2._apply_outbuf_offsets() starts here.')
        k.emit(f"// Add per-warp pointer offsets (but not per-lane offsets) to 'out_max, 'out_argmax', and 'wt'.")
        k.emit(f'// This is tricky because the block/warp indices correspond to bit-reversed DMs, but the output')
        k.emit(f'// arrays are not bit-reversed.')
        k.emit()

        k.emit(f'const {dt32} *wt = (const {dt32} *) wt_;')
        k.emit(f'{dt32} *out_max = ({dt32} *) out_max_;')

        nt_out = self._idiv('ntime', Dout)
        nt_out32 = self._idiv('ntime', Dout*SW)

        k.emit(f'// FIXME could optimize out integer divisions')
        k.emit(f'uint ndm_out = blockDim.y * gridDim.x;')
        k.emit(f'uint lg2_ndm_out = __ffs(ndm_out) - 1;')
        k.emit(f'uint pf_beam = blockIdx.y;  // beam index is not bit-reversed')
        k.emit(f'uint dm_out_brev = threadIdx.y * gridDim.x + blockIdx.x;  // dm is bit-reversed')
        k.emit(f'uint dm_out = __brev(dm_out_brev) >> (32 - lg2_ndm_out);')
        k.emit(f'uint bd_out = pf_beam * ndm_out + dm_out;       // combined (beam,dm) index in out_max + out_argmax arrays')
        k.emit(f'uint bd_wt = bd_out / ndm_out_per_wt;           // combined (beam,dm) index in weight array')
        k.emit(f'uint Touter = ntime / (Tinner * nt_in_per_wt);  // see PfWeightLayout')
        k.emit()
        
        k.emit(f"// Add per-warp pointer offsets (but not per-lane offsets) to 'out_max, 'out_argmax', and 'wt'.")
        k.emit(f'out_max += bd_out * {nt_out32};                  // shape (beams, dm_out, ntime/Dout)')
        k.emit(f'out_argmax += bd_out * {nt_out};                 // shape (beams, dm_out, ntime/Dout)')
        k.emit(f'wt += bd_wt * Touter * wt_touter_stride32;     // shape (beams, ndm_wt, Touter,...)') 
        k.emit(f'// CoalescedDdKernel2._apply_outbuf_offsets() ends here.')
        k.emit()


    @classmethod
    def write_kernel(cls, filename):
        """Called from 'autogenerate_kernel.py' in the toplevel pirate directory."""
        
        basename = os.path.basename(filename)

        # Typical basename: cdd2_fp32_r7_n11_n6_n3_n1_W16_Dcore8_Dout16_Tinner1
        m = re.fullmatch(r'cdd2_(fp\d+)_r(\d+)_((?:n\d+_)*n\d+)_W(\d+)_Dcore(\d+)_Dout(\d+)_Tinner(\d+)\.cu', basename)
        if not m:
            raise RuntimeError(f"Couldn't match filename '{filename}'")
        
        dtype = Dtype(m.group(1))
        dd_rank = int(m.group(2))
        frequency_subbands = FrequencySubbands.from_fstr(m.group(3))
        Wmax, Dcore, Dout, Tinner = int(m.group(4)), int(m.group(5)), int(m.group(6)), int(m.group(7))

        cdd2_kernel = cls(dtype, dd_rank, frequency_subbands, Wmax, Dcore, Dout, Tinner)

        if cdd2_kernel.kernel_basename != basename:
            raise RuntimeError("CoalescedDdKernel2.write_kernel(): internal error: expected "
                               + f" {cdd2_kernel.kernel_basename=} and {basename=} to be equal")
                
        k = Kernel()
        cdd2_kernel.emit_kernel(k)
        cdd2_kernel.emit_registration(k)

        k.write_file(filename)
        
