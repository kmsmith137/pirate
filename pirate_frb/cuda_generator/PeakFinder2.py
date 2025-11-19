import os
import re

from . import utils
from .utils import srange
from .Kernel import Kernel


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

          // Initialization of output pointers is not supplied by PfOutput2.
          T32 *zp = ...;   // per-warp output pointer, points to length (Tin/(Dout*S))
          uint *ap = ...;  // per-warp "argmax" pointer, points to length (Tin/(Dout*S))
          
          // Loop over t-values is not supplied by PfOutput2.
          for (uint tin = 0; tin < Tin; t += 32*S) {
        
              // Multiple calls to PfOutput2.apply_inner().
              // Caller should not reuse 'zname' or 'amax_names' after calling apply_inner().
              apply_inner(k, zname, amax_names);
              apply_inner(k, zname, amax_names);
                // ...

              // One call to PfOutput2.apply_outer(), at bottom of t-loop.
              // Incrementally writes output.
              apply_outer(k, 'zp', 'ap', 'tin', 'Tin');
          }
        """
        
        if dtype == 'float':
            self.dt32 = 'float'
            self.dtmax = 'fmaxf'
            self.S = 1
        elif dtype == '__half':
            self.dt32 = '__half2'
            self.dtmax = '__hmax'
            self.S = 2
        else:
            raise RuntimeError(f'Unrecognized {dtype=}')
            
        assert utils.is_power_of_two(Dout)
        assert self.S <= Dout <= 32
        
        self.dtype = dtype
        self.Dout = Dout
        self.L = utils.integer_log2(Dout)
        self.apply_inner_called = False
        self.apply_outer_called = False

        self.test_kernel_name = f'pf_output2_test_fp{32//self.S}_Dout{Dout}'
        self.test_kernel_basename = self.test_kernel_name + '.cu'
        
        
    def apply_inner(self, k, z, alist):
        """
        The 'z' arg is the name of a variable containing Z-values to be reduced (dtype=dt32).
        The 'alist' arg is a list of varnames for corresponding argmax values (length 1,2 for fp32,fp16).
        """

        dtype, dt32, L, S = self.dtype, self.dt32, self.L, self.S
        
        assert not self.apply_outer_called
        assert len(alist) == S

        k.emit()
        k.emit(f'// PfOutput2.apply_inner() called: {z=}, {alist=}')
        k.emit('// These represent partially reduced Z-values, with associated 32-bit argmax values')
        k.emit('// Register assignment is:')

        self._emit_za_register_assignment(k)

        if not self.apply_inner_called:
            k.emit(f'// First call to apply_inner() just initializes zinner, ainner*')
            k.emit(f'{dt32} zinner = {z};')
            for s in range(S):
                k.emit(f'uint ainner{s} = {alist[s]};')
            self.apply_inner_called = True
            return

        k.emit(f'// Absorbing {z=}, {alist=} into zinner, ainner*')

        if dtype == 'float':
            k.emit(f'ainner0 = ({z} <= zinner) ? ainner0 : {alist[0]};')
            k.emit(f'zinner = fmaxf(zinner, {z});')
        elif dtype == '__half':
            cmp1, cmp2 = k.get_tmp_rname(2)
            k.emit(f'__half2 {cmp1} = __hle2({z}, zinner);')
            k.emit(f'uint {cmp2} = *reinterpret_cast<uint*>(&{cmp1});  // __half2 -> uint')
            k.emit(f'ainner0 = ({cmp2} & 0xffffu) ? ainner0 : {alist[0]};')
            k.emit(f'ainner1 = ({cmp2} & 0xffff0000u) ? ainner1 : {alist[1]};')
            k.emit(f'zinner = __hmax2(zinner, {z});')
        else:
            raise RuntimeError(f'Unrecognized {dtype=}')

    
    def apply_outer(self, k, zout, aout, tin, Tin):
        """
        The 'zout' arg is a per-warp (dt32 *) varname.
        The 'aout' arg is a per-warp (uint *) varname.
        The 'tin' and 'Tin' args are uint varnames;
        Both pointers will be advanced in apply_outer(), as data gets written to global memory.
        """
        
        dtype, L, S = self.dtype, self.L, self.S

        assert self.apply_inner_called
        assert not self.apply_outer_called
        self.apply_outer_called = True
        
        k.emit()
        k.emit(f'// PfOutput2.apply_outer() called: {zout=}, {aout}, {tin=}, {Tin=}')
        k.emit(f'// In this placeholder implementation, we ignore values of tin/Tin,')
        k.emit(f'// and do partial writes directly to global memory. (FIXME suboptimal)')
        k.emit(f'// Starting point is zinner, {srange("ainner",S,sep=", ")}, with register assignemnt')

        self._emit_za_register_assignment(k)

        z = 'zinner'
        
        if dtype == '__half':
            z = 'zinner0'
            lo, hi = k.get_tmp_rname(2)
            k.emit(f'\n// Thread-local reduction from (__half2 zinner) -> (__half zinner0)')
            k.emit(f'__half zinner0 = __low2half(zinner);')
            k.emit(f'__half zinner1 = __high2half(zinner);')
            k.emit(f'ainner0 = (zinner0 < zinner1) ? ainner1 : ainner0;')
            k.emit(f'zinner0 = __hmax(zinner0, zinner1);')

        for b in range(L+1-S):
            zz, aa = k.get_tmp_rname(2);
            k.emit(f'\n// Reduce {z}, ainner0 over lanes, stride={2**b}')
            k.emit(f'{dtype} {zz} = __shfl_sync(~0u, {z}, threadIdx.x ^ {2**b});')
            k.emit(f'uint {aa} = __shfl_sync(~0u, ainner0, threadIdx.x ^ {2**b});')
            k.emit(f'ainner0 = ({z} < {zz}) ? {zz} : {z};')
            k.emit(f'{z} = {self.dtmax}({z},{zz});')

        if dtype == 'float':
            k.emit(f'\n// Now {z}, ainner0 have been fully reduced, with register assignment:')
            k.emit(f'//   lane <-> {srange("s",L)} {srange("tout",5-L)}')
            
            if L >= 1:
                k.emit(f'// Gather onto initial lanes of warp, obtaining register assignment:')
                k.emit(f'//   {srange("l",5-L)} <-> {srange("tout",5-L)}')
                k.emit(f'{z} = __shfl_sync(~0u, {z}, threadIdx.x << {L});')
                k.emit(f'ainner0 = __shfl_sync(~0u, ainner0, threadIdx.x << {L});')
        
        elif dtype == '__half':
            k.emit(f'\n// Now {z}, ainner0 have been fully reduced, with register assignment:')
            k.emit(f'//   lane <-> {srange("s",1,L)} {srange("tout",6-L)}')
            
            k.emit(f'// Gather {z} into initial lanes of warp, and pack to (__half2 zinner):')
            k.emit(f'//   [zinner] simd <-> tout0,  {srange("l",5-L)} <-> {srange("tout",1,6-L)}')
            
            lo, hi = k.get_tmp_rname(2)
            k.emit(f'__half {lo} = __shfl_sync(~0u, {z}, (threadIdx.x << {L}));')
            k.emit(f'__half {hi} = __shfl_sync(~0u, {z}, (threadIdx.x << {L}) + 1);')
            k.emit(f'zinner = __halves2half2({lo}, {hi});')

            if L >= 2:
                k.emit(f'// Gather ainner0 onto initial lanes of warp:')
                k.emit(f'//   [ainner0] {srange("l",6-L)} <-> {srange("tout",6-L)}')
                k.emit(f'ainner0 = __shfl_sync(~0u, ainner0, threadIdx.x << {L-1});')

        else:
            raise RuntimeError(f'Unrecognized {dtype=}')    

        k.emit(f'\n// Now write zinner, ainner0 to global memory (may be partial writes)')
        k.emit(f'// This code could be improved, but apply_outer() is currently a placeholder anyway.')
        
        nz = 2**(5-L)
        na = 2**(4+S-L)
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
        
        if dtype == 'float':
            k.emit(f'//   [z,a0]: lane <-> {srange("s",L)} {srange("tout",5-L)}')
        elif dtype == '__half':            
            k.emit(f'//   [z]: simd <-> s0  lane <-> {srange("s",1,L)} {srange("tout",6-L)}')
            k.emit(f'//   [a0+a1]:  reg <-> s0   lane <-> {srange("s",1,L)} {srange("tout",6-L)}')
        else:
            raise RuntimeError(f'Unrecognized {dtype=}')

        
    @classmethod
    def write_test_kernel(cls, filename):
        """Called from 'autogenerate_kernel.py' in the toplevel pirate directory."""
        
        basename = os.path.basename(filename)
        
        m = re.fullmatch(r'pf_output2_test_fp(\d+)_Dout(\d+)\.cu', basename)
        if not m:
            raise RuntimeError(f"Couldn't match filename '{filename}'")

        nbits, Dout = map(int, m.groups())
        
        if nbits == 32:
            dtype = 'float'
        elif nbits == 16:
            dtype = '__half'
        else:
            raise RuntimeError(f'Invalid {nbits=}')

        pf_output = PfOutput2(dtype, Dout)
        assert pf_output.test_kernel_basename == basename

        k = Kernel()
        S = pf_output.S
        
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

        if dtype == 'float':
            k.emit(f'float *zout32 = (float *) zout_;')
            k.emit(f'float *zin32 = (float *) zin_;')
        elif dtype == '__half':
            k.emit(f'__half2 *zout32 = (__half2 *) zout_;')
            k.emit(f'__half2 *zin32 = (__half2 *) zin_;')
            k.emit(f'uint2 *ain64 = (uint2 *) ain32;')

        k.emit(f'\nfor (uint tin = 0; tin < Tin; tin += {32*S}) {{')
        
        for s in range(4):
            if dtype == 'float':
                p = f'{s}*Tin + ' if (s > 0) else ''
                k.emit(f'float z{s} = zin32[{p}threadIdx.x];')
                k.emit(f'uint a{s} = ain32[{p}threadIdx.x];')
                pf_output.apply_inner(k, f'z{s}', [f'a{s}'])
            elif dtype == '__half':
                p = f'{s}*(Tin>>1) + ' if (s > 0) else ''
                k.emit(f'__half2 z{s} = zin32[{p}threadIdx.x];')
                k.emit(f'uint2 a{s} = ain64[{p}threadIdx.x];')
                pf_output.apply_inner(k, f'z{s}', [f'a{s}.x', f'a{s}.y'])
            else:
                raise RuntimeError(f'Unrecognized {dtype=}')
                
            k.emit()

        p = '32' if (dtype=='float') else '64'
        k.emit(f'// Advance input pointers')
        k.emit(f'zin32 += 32;')
        k.emit(f'ain{p} += 32;')
        k.emit()

        pf_output.apply_outer(k, 'zout32', 'aout32', 'tin', 'Tin')
        
        k.emit('}   // end of tin loop')
        k.emit('}   // end of cuda kernel')

        k.emit('\n// Boilerplate to register the kernel when the library is loaded.')
        k.emit('namespace {')
        k.emit('struct register_hack {')
        k.emit('register_hack() {')
        k.emit('TestPfOutput2::RegistryKey k;')
        k.emit(f'k.dtype = ksgpu::Dtype::native<{pf_output.dtype}>();')
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
