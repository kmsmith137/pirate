class Dtype:
    def __init__(self, x):
        if isinstance(x, Dtype):
            x = x.fname  # fall through...
                
        if x in [ 'fp32', 'float' ]:
            self.scalar = 'float'    # scalar cuda typename
            self.simd32 = 'float'    # scalar 32-bit simd typename
            self.fname = 'fp32'      # used in kernel names and filenames
            self.simd_width = 1
            self.nbits = 32
            self._from_float_str = None
            self._max_str = 'fmaxf'
            self._max_scalar_str = 'fmaxf'
        elif x in [ 'fp16', '__half' ]:
            self.scalar = '__half'   # scalar cuda typename
            self.simd32 = '__half2'  # scalar 32-bit simd typename
            self.fname = 'fp16'      # used in kernel names and filenames
            self.simd_width = 2
            self.nbits = 16
            self._from_float_str = '__float2half2_rn'
            self._max_str = '__hmax2'
            self._max_scalar_str = '__hmax'
        else:
            raise RuntimeError(f'unrecognized dtype: {x}')

            
    def from_float(self, expr, paren=False):
        if self._from_float_str is not None:
            return f'{self._from_float_str}({expr})'
        if paren:
            return f'({expr})'
        return expr

    
    def max(self, x, y):
        return f'{self._max_str}({x}, {y})'
        
    def max_scalar(self, x, y):
        return f'{self._max_scalar_str}({x}, {y})'
