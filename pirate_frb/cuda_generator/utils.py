import io
import sys
import math
import subprocess


def xdiv(num, den):
    assert den > 0
    assert num >= 0
    assert (num % den) == 0
    return (num // den)
    
    
def is_power_of_two(n):
    return (n > 0) and ((n & (n-1)) == 0)


def integer_log2(n):
    assert n > 0
    r = int(math.log2(n) + 0.5)
    if n != 2**r:
        raise RuntimeError(f'integer_log2: argument {n} is not a power of two')
    return r


def bit_reverse(i, nbits):
    assert 0 <= i < 2**nbits
    
    ret = 0
    for b in range(nbits):
        if (i & (1 << b)):
            ret = ret | (1 << (nbits-1-b))
            
    return ret


def tf(x):
    """Needed since C++ uses lower-case true/false, and python uses upper-case True/False."""
    return "true" if x else "false"


def clang_formatter(f = sys.stdout):
    cmd = ['clang-format', '-style={ColumnLimit: 0, IndentWidth: 4}']
    proc = subprocess.Popen(cmd, stdin = subprocess.PIPE, stdout = f)
    return io.TextIOWrapper(proc.stdin, encoding="utf-8", line_buffering=True)
