import io
import sys


class Kernel:
    def __init__(self):
        """Currently the Kernel class is minimal -- I may add more features later."""

        self.active_buffer = io.StringIO()
        self.splices = [ ]   # list of (StringIO, kernel) pairs.
        self.tmp_rnames = dict()
        self.tmp_ix = 0


    def emit(self, s=''):
        print(s, file=self.active_buffer)

    
    def splice(self):
        k = Kernel()
        k.tmp_rnames = self.tmp_rnames  # all splices share the same tmp_rnames
        
        self.splices.append((self.active_buffer, k))
        self.active_buffer = io.StringIO()
        return k

    
    def write(self, f):
        """Suggest taking 'f' to be the return value from utils.clang_formatter()."""
        
        for s,k in self.splices:
            print(s.getvalue(), file=f)
            k.write(f)
        
        print(self.active_buffer.getvalue(), file=f)


    def get_tmp_rname(self, dtype):
        if dtype not in self.tmp_rnames:
            d = { 'float': 'ftmp', '__half2': 'h2tmp', 'bool': 'btmp', 'int': 'itmp' }
            if dtype not in d:
                raise RuntimeError(f'Kernel.get_tmp_rname(): unrecognized {dtype=}')

            # Not really correct -- tmp_rname is declared in the splice where get_tmp_rname() is first called.
            self.emit(f'{dtype} {d[dtype]};')
            self.tmp_rnames[dtype] = d[dtype]
            
        return self.tmp_rnames[dtype]

    def get_tmp_rname2(self, n=None):
        if n is not None:
            return [ self.get_tmp_rname2() for _ in range(n) ]
        
        ret = f'tmp{self.tmp_ix}'
        self.tmp_ix += 1
        return ret

    def reset_tmp_vars(self):
        self.tmp_rnames = dict()
        

    def warp_transpose(self, rname1, rname2, thread_stride, dtype):
        if thread_stride not in [1,2,4,8,16]:
            raise RuntimeError(f'Invalid {thread_stride=}')

        tmp = self.get_tmp_rname(dtype)
        btmp = self.get_tmp_rname('bool')

        self.emit(f'// warp_transpose({rname1}, {rname2}, {thread_stride=})')
        self.emit(f'{btmp} = (threadIdx.x & {thread_stride});')
        self.emit(f'{tmp} = {btmp} ? {rname1} : {rname2};')
        self.emit(f'{tmp} = __shfl_sync(0xffffffff, {tmp}, threadIdx.x ^ {thread_stride});')
        self.emit(f'{rname1} = {btmp} ? {tmp} : {rname1};')
        self.emit(f'{rname2} = {btmp} ? {rname2} : {tmp};')
