
class RbHack:
    def __init__(self, dtype, rank0, rank1):
        assert dtype in [ 'float', '__half' ]
        self.dtype = dtype
        self.rank0 = rank0
        self.rank1 = rank1
        self.shmem_nbytes = self.rb_pos(2**rank0,0) * 4
        self.static_asserts()


    def static_asserts(self):
        expected_pos = 0
        
        for dm in range(2**self.rank0):
            for frev in range(2**self.rank1):
                assert self.rb_pos(dm,frev) == expected_pos
                lag = (frev*dm) if (self.dtype == 'float') else (frev*dm)//2
                expected_pos += (lag + 32)

        assert self.shmem_nbytes == (expected_pos * 4)
        print(f'static_asserts(rank0={self.rank0}, rank1={self.rank1}): pass')
        

    def rb_pos(self, dm, frev):
        """Called by static_asserts(). Also called directly in constructor for shmem_nbytes."""
        
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


if __name__ == '__main__':
    for dtype in [ 'float', '__half' ]:
        for rank in range(3,9):
            rank0 = rank//2
            rank1 = rank-rank0
            RbHack(dtype, rank0, rank1)
