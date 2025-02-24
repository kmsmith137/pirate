# Note: 'import ksgpu' must precede 'import pirate_pybind11'.
# (This is because 'import ksgpu' pulls in the libraries ksgpu.so and ksgpu_pybind11...so,
# using the "ctypes trick" to make their symbols globally visible.)
import ksgpu

from .pirate_pybind11 import *
