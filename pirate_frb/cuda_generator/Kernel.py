import io
import sys


class Kernel:
    def __init__(self):
        """Currently the Kernel class is minimal -- I may add more features later."""

        self.active_buffer = io.StringIO()
        self.splices = [ ]   # list of (StringIO, kernel) pairs.
        self.tmp_ix = [ 0 ]  # use a length-zero list so that splices can share the same tmp_ix


    def emit(self, s=''):
        print(s, file=self.active_buffer)

    
    def splice(self):
        k = Kernel()
        k.tmp_ix = self.tmp_ix   # all splices share the same tmp_ix
        
        self.splices.append((self.active_buffer, k))
        self.active_buffer = io.StringIO()
        return k

    
    def write(self, f):
        """Suggest taking 'f' to be the return value from utils.clang_formatter()."""
        
        for s,k in self.splices:
            print(s.getvalue(), file=f)
            k.write(f)
        
        print(self.active_buffer.getvalue(), file=f)


    def get_tmp_rname(self, n=None):
        if n is not None:
            return [ self.get_tmp_rname() for _ in range(n) ]
        
        ret = f'tmp{self.tmp_ix[0]}'
        self.tmp_ix[0] += 1
        return ret

    
    def reset_tmp_vars(self):
        self.tmp_ix[0] = 0
        

    def warp_transpose(self, rname1, rname2, thread_stride, dtype):
        if thread_stride not in [1,2,4,8,16]:
            raise RuntimeError(f'Invalid {thread_stride=}')

        tmp, btmp = self.get_tmp_rname(2)

        self.emit(f'// warp_transpose({rname1}, {rname2}, {thread_stride=})')
        self.emit(f'bool {btmp} = (threadIdx.x & {thread_stride});')
        self.emit(f'{dtype} {tmp} = {btmp} ? {rname1} : {rname2};')
        self.emit(f'{tmp} = __shfl_sync(0xffffffff, {tmp}, threadIdx.x ^ {thread_stride});')
        self.emit(f'{rname1} = {btmp} ? {tmp} : {rname1};')
        self.emit(f'{rname2} = {btmp} ? {rname2} : {tmp};')
