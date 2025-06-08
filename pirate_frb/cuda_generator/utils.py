import io
import sys
import math
import subprocess


def is_power_of_two(n):
    return (n > 0) and ((n & (n-1)) == 0)

def integer_log2(n):
    r = int(math.log2(n) + 0.5)
    if n != 2**r:
        raise RuntimeError(f'integer_log2: argument {n} is not a power of two')
    return r

def clang_formatter(f = sys.stdout):
    cmd = ['clang-format', '-style={ColumnLimit: 0, IndentWidth: 4}']
    proc = subprocess.Popen(cmd, stdin = subprocess.PIPE, stdout = f)
    return io.TextIOWrapper(proc.stdin, encoding="utf-8", line_buffering=True)
