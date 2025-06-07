import math

def is_power_of_two(n):
    return (n > 0) and ((n & (n-1)) == 0)

def integer_log2(n):
    r = int(math.log2(n) + 0.5)
    if n != 2**r:
        raise RuntimeError(f'integer_log2: argument {n} is not a power of two')
    return r
