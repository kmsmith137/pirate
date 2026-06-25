"""Core integer/bit utilities for pirate_frb."""


def integer_log2(n):
    """Return log2(n) as an int, where n must be a positive power of two.

    Raises ValueError otherwise. Python analog of C++ pirate::integer_log2().
    """
    n = int(n)
    if n <= 0 or (n & (n - 1)) != 0:
        raise ValueError(f"integer_log2: argument {n} is not a positive power of two")
    return n.bit_length() - 1
