import os
import re
import numpy as np

from . import utils
from .utils import srange

from .Dtype import Dtype
from .Kernel import Kernel
from .FrequencySubbands import FrequencySubbands


####################################################################################################


class PfWeightLayout:
    def __init__(self, frequency_subbands, dtype, P, Tinner):
        """
        PfWeightLayout: helper class for PfWeightReader.
        
        The W-array in global memory
        ----------------------------
        
        The W-array is a logical 4-d array with shape (Dbar,Tbar,P,F) parameterized by:
        
          - Dbar = number of coarse DMs
          - Tbar = number of coarse time samples
          - P = number of peak-finding profiles
          - F = number of frequency subbands F

        The coarse quantites (Dbar,Tbar) are related to the "fine" quantities (Dtree,Ttree)
        in the dedispersion tree by downsampling factors (Dt,Dd):

          Dd = Dtree / Dbar
          Dt = Ttree / Tbar

        Before describing the global memory layout, a few more definitions:

          SW = 32 / sizeof(T)         "simd width"
          Tinner = max(32*SW/Dt, 1)   see (*) below
          Pinner = SW                 see (**) below

        Then, we split P,Tbar into "outer" and "inner" parts:
        
          Pouter = ceil(P / Pinner)   where Pinner = SW (**)
          Touter = Tbar / Tinner

        The W-array global memory layout can be described as either a 6-d or a 5-d array:
        
           dtype          W[Dbar,Touter,Pouter,F,Tinner,Pinner]
           dtype*Pinnner  W[Dbar,Touter,Pouter,F,Tinner]

        Important note: we always pad so that the 'Touter' stride is 128-byte aligned!
        (See "self.touter_stride" below.)

        (*) Tinner is the number of W-array t-indices per iteration of the "outer time loop"
            in the larger kernel. Each iteration of this loop processes 128 / sizeof(dtype) / Dt
            "tree" time samples.
        
        (**) Pinner is an "innermost width" used for weights in the peak-finding kernel. Currently,
             it is always equal to the simd_width SW. However, in a future update, I may implement
             a shared-memory code path (see FIXME below) in which Pinner = 2*SW.

        
        Constructor args
        ----------------

          - frequency_subbands: instance of class FrequencySubbands
        
          - dtype: either Dtype instance or string
        
          - P: number of peak-finding kernels (see toplevel kernel).
        
          - Tinner: defined as max(32*SW/Dt, 1), see above for discussion.
        """

        assert isinstance(frequency_subbands, FrequencySubbands)
        assert utils.is_power_of_two(Tinner)
        assert 1 <= Tinner <= 32
        assert P > 0

        self.frequency_subbands = frequency_subbands
        self.dtype = dtype = Dtype(dtype)
        self.F = frequency_subbands.F
        self.Tinner = Tinner
        self.P = P
        
        # See docstring for definitions of these quantities. Note that for now, Pinner
        # is equal to the simd width, but this may change in the future.
        self.Pinner = dtype.simd_width
        self.Pouter = (self.P + self.Pinner - 1) // self.Pinner

        # The weights array is stored with a non-contiguous (128-byte) aligned touter-stride (see docstring).
        self.unpadded_byte_stride = self.Pouter * self.F * Tinner * self.Pinner * utils.xdiv(dtype.nbits,8)
        self.touter_byte_stride = (self.unpadded_byte_stride + 127) & ~127   # round up to multiple of 128
        

####################################################################################################


class PfWeightReader:
    def __init__(self, frequency_subbands, dtype, Dcore, P, Tinner):
        """
        The read_weights() method
        -------------------------

        Each call to read_weights() reads 
        
        
        Generated code looks like this
        ------------------------------

          constexpr int SW = 128 / sizeof(dtype);  // simd width
        
          // Initialization of input pointer is not supplied by PfWeightReader.
          // 'wp' is a per-warp pointer to shape (Touter,Pouter,F,Tinner,Pinner/SW),
          // where the 'touter' stride is 128-byte aligned. The pointer is "owned"
          // by the PfWeightReader class, and will be incremented as data is read
          // from global memory.
        
          T32 *wp = ...;
          pfw_reader.top('wp');
        
          // Loop over t-values is not supplied by PfOutput2.
          for (uint tin = 0; tin < Tin; t += 32*SW) {

              // Outer "unrolled" loop over 0 <= mouter < Mouter.
              // Inner "unrolled" loop over 0 <= pouter < Pouter.
              w = pfw_reader.read_weights(mouter=0, pouter=0)
              w = pfw_reader.read_weights(mouter=0, pouter=1)
                // ...
              w = pfw_reader.read_weights(mouter=Mouter-1, pouter=Pouter-1)
        
              pfw_reader.bottom('tin', 'Tin')
          }
        """

        self.dtype = dtype = Dtype(dtype)
        self.weight_layout = PfWeightLayout(frequency_subbands, dtype, P, Tinner)
        self.frequency_subbands = frequency_subbands
        self.Tinner = Tinner
        self.Dcore = Dcore
        self.P = P

        self.dt32 = dtype.simd32
        self.SW = dtype.simd_width
        self.Pouter = self.weight_layout.Pouter
        self.Pinner = self.weight_layout.Pinner
        self.M = frequency_subbands.M
        self.F = frequency_subbands.F
        
        # FIXME explain
        assert (Dcore) >= (self.SW)
        assert (Dcore * Tinner) <= (32 * self.SW)
        
        # See docstring for definitions of these quantities. Note that for now, Pinner
        # is equal to the simd width, but this may change in the future (see FIXME below).
        self.Minner = Minner = utils.xdiv(Dcore, self.SW)
        self.Mouter = Mouter = (self.M + Minner - 1) // Minner

        fstr = '_'.join(f'f{n}' for n in frequency_subbands.subband_counts)
        self.test_kernel_name = f'pf_weight_reader_test_{dtype.fname}_{fstr}_Dcore{Dcore}_P{P}_Tinner{Tinner}'
        self.test_kernel_basename = self.test_kernel_name + '.cu'

        # Throughout 'class PfWeightReader', a capitalized index 0 <= I < Pouter*F*Tinner
        # denotes a "flattened" index triple (pouter, f, tinner). Such an index I can be
        # viewed as an offset relative to the 'wp' pointer (see docstring), and (I >> 5)
        # corresponds to a cache line.

        # 'pf_I0' is the mapping (mouter,minner) -> I, for pouter=tinner=0.
        self.pf_I0 = np.zeros((Mouter,Minner), dtype=int)
        for mouter in range(Mouter):
            for minner in range(Minner):
                m = min(mouter*Minner + minner, self.M-1)
                f = frequency_subbands.m_to_fd[m][0]
                self.pf_I0[mouter, minner] = f * Tinner

        # pf_Imin = min(pf_I0) over all lanes, i.e. all (minner,tinner) pairs.
        self.pf_Imin = np.min(self.pf_I0, axis=1)
        self.pf_Imax = np.max(self.pf_I0, axis=1) + (Tinner - 1)

        # Dict (cache line index I>>5) -> (wcl string varname).
        # This gets populated as read_weights() is called.
        self.wcl_cache = { }

        self.top_called = False
        self.expected_mouter = 0
        self.expected_pouter = 0
        self.bottom_called = False
        
        # FIXME 1: shared memory
        #  - less global memory bandwidth
        #  - faster than warp shuffle, if 64-bit loads are used, on some architectures
        #  - Pinner may change to (2 * simd_width)
        #  - API change: read_weights() can now return two T32s
        #
        # FIXME 2: register limit
        #  - Use Belady's algorithm
        #
        # FIXME 3: if Dt=0 then test whether re-reading is really necessary
        #
        # The optimal interface would specify a register limit, a shared memory limit,
        # and let the constructor choose the strategy!
        #
        # FIXME 4: dynamic programming
        
        
    def top(self, k, wp):
        """Placeholder for future expansion."""
        
        assert not self.top_called
        self.top_called = True
        self.wp = wp


    def _read_wcl(self, k, Icl):
        if Icl in self.wcl_cache:
            w = self.wcl_cache[Icl]
            k.emit(f"// At this point in the code, '{w}' contains {self.wp}[{32*Icl}:{32*Icl+32}]")
            return w

        wcl = k.get_name('pf_wcl')
        k.emit(f"{self.dt32} {wcl} = {self.wp}[{32*Icl} + (threadIdx.x & 0x1f)]; // {self.wp}[{32*Icl}:{32*Icl+32}]")
        self.wcl_cache[Icl] = wcl
        return wcl

    
    def read_weights(self, k, mouter, pouter):
        assert self.top_called
        assert not self.bottom_called
        assert (mouter, pouter) == (self.expected_mouter, self.expected_pouter)
        self.expected_mouter = (mouter) if (pouter < self.Pouter-1) else (mouter+1)
        self.expected_pouter = (pouter+1) if (pouter < self.Pouter-1) else 0
        
        wp, F, Tinner = self.wp, self.F, self.Tinner
        k.emit(f'// PfWeightReader.read_weights({mouter=}, {pouter=}): start.')

        if pouter == 0:
            self._init_pf_I(k, mouter)

        dI = pouter * F * Tinner
        Imin = int(self.pf_Imin[mouter]) + dI
        Imax = int(self.pf_Imax[mouter]) + dI
        Istr = f'pf_I + {dI}' if (dI > 0) else 'pf_I'

        # Very important assert -- our algorithm depends on this!
        assert np.all(Imax < Imin + 32)

        k.emit(f'// We want to load {wp}[{Istr}] on each thread, where {Imin} <= ({Istr}) <= {Imax}.')
        wcl = self._read_wcl(k, Imin >> 5)

        if (Imin >> 5) != (Imax >> 5):
            wcl2 = self._read_wcl(k, Imax >> 5)
            wrap = k.get_name('wrap')
            k.emit(f'// Wrapped {wp}[{Imin}:{Imin+32}]')
            k.emit(f'{self.dt32} {wrap} = ((threadIdx.x & 0x1f) >= {Imin & 0x1f}) ? {wcl} : {wcl2};')
            wcl = wrap

        w = k.get_name('pf_w')
        k.emit(f'{self.dt32} {w} = __shfl_sync(~0u, {wcl}, {Istr});')
        k.emit(f"// PfWeightReader.read_weights({mouter=}, {pouter=}): end (weights are in '{w}')")
        return w


    def _init_pf_I(self, k, mouter):
        """Helper called by read_weights()."""
        
        Minner, Tinner = self.Minner, self.Tinner
        
        Mbits = utils.integer_log2(Minner)
        Tbits = utils.integer_log2(Tinner)
        minner = f'(threadIdx.x & {Minner-1})' if (Minner > 1) else '0'
        tinner = f'((threadIdx.x & 0x1f) >> {5-Tbits})' if (Tinner > 1) else '0'
        
        k.emit()
        k.emit(f'// In this part, we have {Minner=}, {Tinner=}, and the following mapping between')
        k.emit(f'// lanes and (minner,tinner) pairs:')
        k.emit(f'//    {Mbits} lane bits <-> minner')
        k.emit(f'//    {5-Mbits-Tbits} lane bits <-> spectator t-indices')
        k.emit(f'//    {Tbits} lane bits <-> tinner')
        k.emit(f'//')
        k.emit(f'// Therefore, {minner = } and {tinner = }')
        k.emit(f'//')
        k.emit(f'// Compute pf_I: the I-index for (mouter,pouter)=({mouter},0), and (minner,tinner)')
        k.emit(f'// defined by the laneId. For tinner=0, this is described by the following lookup')
        k.emit(f'// table (minner) -> (pf_I): {list(map(int,self.pf_I0[mouter,:]))}')
        k.emit('//')
        k.emit('// FIXME: placeholder algorithm for computing pf_I (dynamic programming is best!).')

        decl = 'int ' if (mouter == 0) else ''
        k.emit(f'{decl}pf_I = {int(self.pf_I0[mouter,0])};')

        for m in range(1, Minner):
            if self.pf_I0[mouter,m] != self.pf_I0[mouter,m-1]:
                k.emit(f'pf_I = ({minner} < {m}) ? pf_I : {int(self.pf_I0[mouter,m])};')

        if Tinner != 1:
            k.emit(f'pf_I += {tinner};')
        else:
            k.emit(f'// since Tinner=1, no need for "pf_I += tinner;"')

    
    def bottom(self, k, tin, Dt):
        """The 'tin', 'Dt' args are string varnames."""
        
        assert self.top_called
        assert not self.bottom_called
        assert (self.expected_mouter, self.expected_pouter) == (self.Mouter, 0)
        self.bottom_called = True
 
        nelts = self.Pouter * self.F * self.Tinner * self.Pinner
        us = self.weight_layout.unpadded_byte_stride
        bs = self.weight_layout.touter_byte_stride
        ps = utils.xdiv(bs, 4)   # since 'wp' is a 32-bit type

        k.emit()
        k.emit(f'// PfWeightReader.bottom()')
        k.emit(f'// One touter-step corresponds to (Pouter * F * Tinner * Pinner) = {nelts} W-array elements')
        k.emit(f'// unpadded_byte_stride = {us}, byte_stride = {bs}, pointer_stride = {ps}')

        if self.Tinner > 1:
            k.emit(f"// Since Tinner > 1, we ignore '{Dt}', and always increment the weight pointer '{self.wp}'.")
        else:
            k.emit("// FIXME optimize out mod-operator")
            k.emit(f"if (!(({tin} + {32*SW}) % {Dt}))")
            
        k.emit(f"{self.wp} += {ps};")
        
        
    @classmethod
    def write_test_kernel(cls, filename):
        """Called from 'autogenerate_kernel.py' in the toplevel pirate directory."""
        
        basename = os.path.basename(filename)

        # Typical basename: pf_weight_reader_test_fp32_f11_f6_f3_f1_Dcore8_P13_Tinner2.cu
        m = re.fullmatch(r'pf_weight_reader_test_(fp\d+)_((?:f\d+_)+)Dcore(\d+)_P(\d+)_Tinner(\d+)\.cu', basename)
        if not m:
            raise RuntimeError(f"Couldn't match filename '{filename}'")

        dtype = Dtype(m.group(1))
        subband_counts = list(map(int, re.findall(r'f(\d+)_', m.group(2))))
        Dcore, P, Tinner = int(m.group(3)), int(m.group(4)), int(m.group(5))
        
        frequency_subbands = FrequencySubbands(subband_counts = subband_counts)
        pf_weight_reader = PfWeightReader(frequency_subbands, dtype, Dcore, P, Tinner)

        if pf_weight_reader.test_kernel_basename != basename:
            raise RuntimeError("PfWeightReader.write_test_kernel(): internal error: expected "
                               + f" {pf_weight_reader.test_kernel_basename=} and {basename=} to be equal")

        dt32 = dtype.simd32
        SW = dtype.simd_width
        
        k = Kernel()
        
        k.emit('// Autogenerated by pirate_frb.cuda_generator')
        k.emit()
        k.emit('// For a high-level overview, see the long comment at the top of')
        k.emit('// pirate_frb/cuda_generator/PeakFinder.py')
        k.emit()
        k.emit('#include <cstdio>')
        k.emit('#include <iostream>')
        k.emit('#include "../../include/pirate/PeakFindingKernel.hpp"')
        k.emit()
        k.emit('namespace pirate {')
        k.emit()

        k.emit(f'// out.shape == (Tin/(32*SW), Mouter, Pouter, 32, Pinner)')
        k.emit(f'// in.shape == (Tin/(Dt*Tinner), Pouter, F, Tinner, Pinner)')
        k.emit(f'//')
        k.emit(f'// The test kernel does the following (schematically):')
        k.emit(f'//')
        k.emit(f'//   for (int tin = 0; tin < tin; tin += 32*SW)')
        k.emit(f'//       for (int mouter = 0; mouter < Mouter; mouter++)')
        k.emit(f'//           for (int Pouter = 0; pouter < Pouter; pouter++)')
        k.emit(f'//               call read_weights(), and write to out[tin/(32*SW), mouter, pouter, ...]')
        k.emit(f'//')
        k.emit(f"// The length-32 axis of 'out' can be viewed as (Tinner, 32/(Minner*Tinner), Minner).")
        k.emit(f"// The 'in' array can have a non-contiguous touter-index, see 'touter_byte_stride' below.")
        k.emit(f"//")
        k.emit(f'// If Tinner > 1, then Dt must equal (32*SW)/Tinner, and Tin must be a multiple of (32*SW).')
        k.emit(f'// If Tinner == 1, then Dt must be a multiple of (32*SW), and Tin must be a multiple of Dt.')
        k.emit(f"//")
        k.emit(f'// Launch with 32 threads, 1 block.')
        k.emit()
        
        k.emit(f'__global__ void {pf_weight_reader.test_kernel_name}(void *out_, const void *in_, uint Tin, uint Dt)')
        k.emit(f'{{')
        k.emit(f'{dt32} *out = ({dt32} *) out_;')
        k.emit(f'const {dt32} *in = (const {dt32} *) in_;')

        pf_weight_reader.top(k, 'in')

        k.emit()
        k.emit(f'for (uint tin = 0; tin < Tin; tin += {32*SW}) {{')

        for mouter in range(pf_weight_reader.Mouter):
            for pouter in range(pf_weight_reader.Pouter):
                w = pf_weight_reader.read_weights(k, mouter, pouter)
                k.emit()
                k.emit(f'out[threadIdx.x] = {w};')
                k.emit(f'out += 32;')
                k.emit()

        pf_weight_reader.bottom(k, 'tin', 'Dt')
        
        k.emit('}   // end of tin loop')
        k.emit('}   // end of cuda kernel')

        m_to_f = [ ]
        
        for mm in range(pf_weight_reader.Mouter * pf_weight_reader.Minner):
            m = min(mm, pf_weight_reader.M-1)
            f = pf_weight_reader.frequency_subbands.m_to_fd[m][0]
            m_to_f.append(int(f))

        m_to_f = ', '.join(str(int(x)) for x in m_to_f)
        sb_counts = ', '.join(str(int(x)) for x in pf_weight_reader.frequency_subbands.subband_counts)
        
        k.emit('\n// Boilerplate to register the kernel when the library is loaded.')
        k.emit('namespace {')
        k.emit('struct register_hack {')
        k.emit('register_hack() {')
        k.emit('TestPfWeightReader::RegistryKey k;')
        k.emit(f'k.dtype = ksgpu::Dtype::native<{dtype.scalar}>();')
        k.emit(f'k.subband_counts = {{ {sb_counts} }};')
        k.emit(f'k.rank = {pf_weight_reader.frequency_subbands.pf_rank};')
        k.emit(f'k.Dcore = {pf_weight_reader.Dcore};')
        k.emit(f'k.Tinner = {pf_weight_reader.Tinner};')
        k.emit(f'k.P = {pf_weight_reader.P};')
        k.emit()
        k.emit('TestPfWeightReader::RegistryValue v;')
        k.emit(f'v.cuda_kernel = {pf_weight_reader.test_kernel_name};')
        k.emit(f'v.Mouter = {pf_weight_reader.Mouter};')
        k.emit(f'v.Minner = {pf_weight_reader.Minner};')
        k.emit(f'v.Pouter = {pf_weight_reader.Pouter};')
        k.emit(f'v.Pinner = {pf_weight_reader.Pinner};')
        k.emit(f'v.F = {pf_weight_reader.F};')
        k.emit(f'v.touter_byte_stride = {pf_weight_reader.weight_layout.touter_byte_stride};')
        k.emit(f'v.m_to_f = {{ {m_to_f} }};')
        k.emit()
        k.emit('bool debug = false;')
        k.emit('TestPfWeightReader::registry().add(k, v, debug);')
        k.emit('} // register_hack constructor')
        k.emit('}; // struct register hack')
        k.emit('register_hack hack;')
        k.emit('} // anonymous namespace')
        k.emit()
        
        k.emit('}   // namespace pirate')

        with open(filename,'w') as f:
            with utils.clang_formatter(f) as ff:
                k.write(ff)


####################################################################################################


class PfOutput2:
    def __init__(self, dtype, Dout):
        """
        Input: partially reduced Z_{st} array, with associated 32-bit argmax values.
        Here, "s" is a spectator index (from the perspective of the PfOutput2 microkernel).
        In the larger kernel, "s" is a combination of (m,p,tlo). The register assignment is:
         
          [float32]  lane <-> s(0,L) tout(0,5-L)
          [float16]  simd <-> s(0)    lane <-> s(1,L) tout(0,6-L)

        Output: as an "outer" t-loop is iterated, the Z_{st} array gets reduced over
        spectator indices, and two length Tout=(Tin/Dout) array gets incrementally
        written to global memory (see below).

        Generated code looks like this:

          constexpr int SW = 128 / sizeof(dtype);  // simd width
        
          // Initialization of output pointers is not supplied by PfOutput2.
          T32 *zp = ...;   // per-warp output pointer, points to length (Tin/(Dout*SW))
          uint *ap = ...;  // per-warp "argmax" pointer, points to length (Tin/(Dout*SW))
          
          // Loop over t-values is not supplied by PfOutput2.
          for (uint tin = 0; tin < Tin; t += 32*SW) {
        
              // Multiple calls to PfOutput2.apply_inner().
        
              pf_output2.apply_inner(k, zname1, amax_names1);
                // ...
              pf_output2.apply_inner(k, zname2, amax_names2);
                // ...

              // One call to PfOutput2.apply_outer(), at bottom of t-loop, to write
              // output incrementally. The'zout' and 'aout' pointers are "owned" by
              // the PfOutput2 class, and these pointers will be incremented, as data
              // gets written to global memory.

              pf_output2.apply_outer(k, 'zp', 'ap', 'tin', 'Tin');
          }
        """
        
        self.Dout = Dout
        self.dtype = dtype = Dtype(dtype)
        self.L = utils.integer_log2(Dout)
        self.SW = dtype.simd_width
        self.dt32 = dtype.simd32

        self.test_kernel_name = f'pf_output2_test_fp{32//self.SW}_Dout{Dout}'
        self.test_kernel_basename = self.test_kernel_name + '.cu'
        self.apply_inner_called = False
        self.apply_outer_called = False
        
        
    def apply_inner(self, k, z, alist):
        """
        The 'z' arg is the name of a variable containing Z-values to be reduced (dtype=dt32).
        The 'alist' arg is a list of varnames for corresponding argmax values (length 1,2 for fp32,fp16).
        Contents of the 'alist' registers are opaque "tokens" in class PfOutput2.
        """

        dtype, dt32, L, SW = self.dtype, self.dt32, self.L, self.SW
        
        assert not self.apply_outer_called
        assert len(alist) == SW

        k.emit()
        k.emit(f'// PfOutput2.apply_inner() called: {z=}, {alist=}')
        k.emit('// These represent partially reduced Z-values, with associated 32-bit argmax values')
        k.emit('// Register assignment is:')

        self._emit_za_register_assignment(k)

        if not self.apply_inner_called:
            k.emit(f'// First call to apply_inner() just initializes zinner, ainner*')
            k.emit(f'{dt32} zinner = {z};')
            for s in range(SW):
                k.emit(f'uint ainner{s} = {alist[s]};')
            self.apply_inner_called = True
            return

        k.emit(f'// Absorbing {z=}, {alist=} into zinner, ainner*')

        if dtype.nbits == 32:
            k.emit(f'ainner0 = ({z} <= zinner) ? ainner0 : {alist[0]};')
            k.emit(f'zinner = fmaxf(zinner, {z});')
        elif dtype.scalar == '__half':
            cmp1, cmp2 = k.get_tmp_rname(2)
            k.emit(f'__half2 {cmp1} = __hle2({z}, zinner);')
            k.emit(f'uint {cmp2} = *reinterpret_cast<uint*>(&{cmp1});  // __half2 -> uint')
            k.emit(f'ainner0 = ({cmp2} & 0xffffu) ? ainner0 : {alist[0]};')
            k.emit(f'ainner1 = ({cmp2} & 0xffff0000u) ? ainner1 : {alist[1]};')
            k.emit(f'zinner = __hmax2(zinner, {z});')
        else:
            raise RuntimeError('should never get here')

    
    def apply_outer(self, k, zout, aout, tin, Tin):
        """
        The 'zout' arg is a per-warp (dt32 *) varname.
        The 'aout' arg is a per-warp (uint *) varname.
        The 'tin' and 'Tin' args are uint varnames.

        NOTE: The'zout' and 'aout' pointers are "owned" by the PfOutput2 class, and these
        pointers will be incremented, as data gets written to global memory.
        """
        
        dtype, L, SW = self.dtype, self.L, self.SW

        assert self.apply_inner_called
        assert not self.apply_outer_called
        self.apply_outer_called = True
        
        k.emit()
        k.emit(f'// PfOutput2.apply_outer() called: {zout=}, {aout}, {tin=}, {Tin=}')
        k.emit(f'// In this placeholder implementation, we ignore values of tin/Tin,')
        k.emit(f'// and do partial writes directly to global memory. (FIXME suboptimal)')
        k.emit(f'// Starting point is zinner, {srange("ainner",SW,sep=", ")}, with register assignemnt')

        self._emit_za_register_assignment(k)

        z = 'zinner'
        
        if dtype.scalar != 'float':
            z = 'zinner0'
            lo, hi = k.get_tmp_rname(2)
            k.emit(f'\n// Thread-local reduction from (__half2 zinner) -> (__half zinner0)')
            k.emit(f'__half zinner0 = __low2half(zinner);')
            k.emit(f'__half zinner1 = __high2half(zinner);')
            k.emit(f'ainner0 = (zinner0 < zinner1) ? ainner1 : ainner0;')
            k.emit(f'zinner0 = __hmax(zinner0, zinner1);')

        for b in range(L+1-SW):
            zz, aa = k.get_tmp_rname(2);
            k.emit(f'\n// Reduce {z}, ainner0 over lanes, stride={2**b}')
            k.emit(f'{dtype.scalar} {zz} = __shfl_sync(~0u, {z}, threadIdx.x ^ {2**b});')
            k.emit(f'uint {aa} = __shfl_sync(~0u, ainner0, threadIdx.x ^ {2**b});')
            k.emit(f'ainner0 = ({z} < {zz}) ? {aa} : ainner0;')
            k.emit(f'{z} = {self.dtype.max_scalar(z,zz)};')

        if dtype.scalar == 'float':
            k.emit(f'\n// Now {z}, ainner0 have been fully reduced, with register assignment:')
            k.emit(f'//   lane <-> {srange("s",L)} {srange("tout",5-L)}')
            
            if L >= 1:
                k.emit(f'// Gather onto initial lanes of warp, obtaining register assignment:')
                k.emit(f'//   {srange("l",5-L)} <-> {srange("tout",5-L)}')
                k.emit(f'{z} = __shfl_sync(~0u, {z}, threadIdx.x << {L});')
                k.emit(f'ainner0 = __shfl_sync(~0u, ainner0, threadIdx.x << {L});')
        
        elif dtype.scalar == '__half':
            k.emit(f'\n// Now {z}, ainner0 have been fully reduced, with register assignment:')
            k.emit(f'//   lane <-> {srange("s",1,L)} {srange("tout",6-L)}')
            
            k.emit(f'// Gather {z} into initial lanes of warp, and pack to (__half2 zinner):')
            k.emit(f'//   [zinner] simd <-> tout0,  {srange("l",5-L)} <-> {srange("tout",1,6-L)}')
            
            lo, hi = k.get_tmp_rname(2)
            k.emit(f'__half {lo} = __shfl_sync(~0u, {z}, (threadIdx.x << {L}));')
            k.emit(f'__half {hi} = __shfl_sync(~0u, {z}, (threadIdx.x << {L}) + {1<<(L-1)});')
            k.emit(f'zinner = __halves2half2({lo}, {hi});')

            if L >= 2:
                k.emit(f'// Gather ainner0 onto initial lanes of warp:')
                k.emit(f'//   [ainner0] {srange("l",6-L)} <-> {srange("tout",6-L)}')
                k.emit(f'ainner0 = __shfl_sync(~0u, ainner0, threadIdx.x << {L-1});')

        else:
            raise RuntimeError('should never get here')

        k.emit(f'\n// Now write zinner, ainner0 to global memory (may be partial writes)')
        k.emit(f'// This code could be improved, but apply_outer() is currently a placeholder anyway.')
        
        nz = 2**(5-L)
        na = 2**(4+SW-L)
        laneId = k.get_tmp_rname()
        
        k.emit(f'uint {laneId} = (threadIdx.x & 0x1f); // laneId')
        k.emit(f'if ((threadIdx.x & 0x1f) < {nz})')
        k.emit(f'    {zout}[{laneId}] = zinner;')
        k.emit(f'if ((threadIdx.x & 0x1f) < {na})')
        k.emit(f'    {aout}[{laneId}] = ainner0;')
        k.emit(f'{zout} += {nz};')
        k.emit(f'{aout} += {na};')
        
        k.emit(f'\n// PfOutput2.apply_outer() ends here')
        
    
    def _emit_za_register_assignment(self, k):
        """Helper function, called by apply_inner() and apply_outer()."""
        
        dtype, L = self.dtype, self.L
        
        if dtype.scalar == 'float':
            k.emit(f'//   [z,a0]: lane <-> {srange("s",L)} {srange("tout",5-L)}')
        elif dtype.scalar == '__half':            
            k.emit(f'//   [z]: simd <-> s0  lane <-> {srange("s",1,L)} {srange("tout",6-L)}')
            k.emit(f'//   [a0+a1]:  reg <-> s0   lane <-> {srange("s",1,L)} {srange("tout",6-L)}')
        else:
            raise RuntimeError('should never get here')

        
    @classmethod
    def write_test_kernel(cls, filename):
        """Called from 'autogenerate_kernel.py' in the toplevel pirate directory."""
        
        basename = os.path.basename(filename)
        
        m = re.fullmatch(r'pf_output2_test_(fp\d+)_Dout(\d+)\.cu', basename)
        if not m:
            raise RuntimeError(f"Couldn't match filename '{filename}'")

        dtype = Dtype(m.group(1))
        Dout = int(m.group(2))

        pf_output = PfOutput2(dtype, Dout)
        assert pf_output.test_kernel_basename == basename

        k = Kernel()
        SW = pf_output.SW
        
        k.emit('// Autogenerated by pirate_frb.cuda_generator')
        k.emit()
        k.emit('// For a high-level overview, see the long comment at the top of')
        k.emit('// pirate_frb/cuda_generator/PeakFinder.py')
        k.emit()
        k.emit('#include <cstdio>')
        k.emit('#include <iostream>')
        k.emit('#include "../../include/pirate/PeakFindingKernel.hpp"')
        k.emit()
        k.emit('namespace pirate {')
        k.emit()

        k.emit(f'// Call with 32 threads, and 1 threadblock.')
        k.emit(f'// Use 4 calls to apply_inner() inside t-loop.')
        k.emit(f'// Thus, the number of reduced spectator indices is 4*Dout = 4*{Dout} = {4*Dout}')
        k.emit(f'// zout.shape == aout.shape == (Tin//Dout) == (Tin//{Dout})')
        k.emit(f'// zin.shape == ain.shape == (4, Tin)')
        k.emit()

        k.emit(f'__global__ void {pf_output.test_kernel_name}(void *zout_, uint *aout32, void *zin_, uint *ain32, uint Tin)')
        k.emit(f'{{')

        if dtype.scalar == 'float':
            k.emit(f'float *zout32 = (float *) zout_;')
            k.emit(f'float *zin32 = (float *) zin_;')
        elif dtype.scalar == '__half':
            k.emit(f'__half2 *zout32 = (__half2 *) zout_;')
            k.emit(f'__half2 *zin32 = (__half2 *) zin_;')
            k.emit(f'uint2 *ain64 = (uint2 *) ain32;')
        else:
            raise RuntimeError('should never get here')

        k.emit(f'\nfor (uint tin = 0; tin < Tin; tin += {32*SW}) {{')
        
        for s in range(4):
            if dtype.scalar == 'float':
                p = f'{s}*Tin + ' if (s > 0) else ''
                k.emit(f'float z{s} = zin32[{p}threadIdx.x];')
                k.emit(f'uint a{s} = ain32[{p}threadIdx.x];')
                pf_output.apply_inner(k, f'z{s}', [f'a{s}'])
            elif dtype.scalar == '__half':
                p = f'{s}*(Tin>>1) + ' if (s > 0) else ''
                k.emit(f'__half2 z{s} = zin32[{p}threadIdx.x];')
                k.emit(f'uint2 a{s} = ain64[{p}threadIdx.x];')
                pf_output.apply_inner(k, f'z{s}', [f'a{s}.x', f'a{s}.y'])
            else:
                raise RuntimeError('should never get here')
                
            k.emit()

        k.emit(f'// Advance input pointers')
        k.emit(f'zin32 += 32;')
        k.emit(f'ain{32*dtype.simd_width} += 32;')  # either 'ain32' or 'ain64'
        k.emit()

        pf_output.apply_outer(k, 'zout32', 'aout32', 'tin', 'Tin')
        
        k.emit('}   // end of tin loop')
        k.emit('}   // end of cuda kernel')

        k.emit('\n// Boilerplate to register the kernel when the library is loaded.')
        k.emit('namespace {')
        k.emit('struct register_hack {')
        k.emit('register_hack() {')
        k.emit('TestPfOutput2::RegistryKey k;')
        k.emit(f'k.dtype = ksgpu::Dtype::native<{dtype.scalar}>();')
        k.emit(f'k.Dout = {pf_output.Dout};')
        k.emit()
        k.emit('TestPfOutput2::RegistryValue v;')
        k.emit(f'v.cuda_kernel = {pf_output.test_kernel_name};')
        k.emit()
        k.emit('bool debug = false;')
        k.emit('TestPfOutput2::registry().add(k, v, debug);')
        k.emit('} // register_hack constructor')
        k.emit('}; // struct register hack')
        k.emit('register_hack hack;')
        k.emit('} // anonymous namespace')
        k.emit()
        
        k.emit('}   // namespace pirate')

        with open(filename,'w') as f:
            with utils.clang_formatter(f) as ff:
                k.write(ff)
