"""
Detrending: the pure-numpy reference detrenders, alongside the GPU kernels that are
validated against them.

    ReferenceDetrenderLps1d   <->   GpuDetrenderLps1d   (src_lib/DetrenderLps1d.cu)
    ReferenceDetrenderLps2d   <->   GpuDetrenderLps2d   (src_lib/DetrenderLps2d.cu)
    ReferenceDetrenderKf1d    <->   no GPU kernel yet

"Lps" is local polynomial subtraction and "Kf" is a fixed-lag Kalman filter. Both 1-d
detrenders fit the time axis and are meant to be run side by side and compared; the 2-d
detrender fits frequency and time jointly. All three algorithms are specified in
notes/detrending.tex, sections "Time detrending algorithm 1: local polynomial
subtraction", "Time detrending algorithm 2: Kalman filter", and "2-d detrending".

One subpackage per algorithm -- lps1d, kf1d, lps2d -- each holding its reference
detrender, the brute-force oracle that validates it, the algorithm's own machinery, and
its test suite (run by 'python -m pirate_frb test --dtl1', '--dtk1', '--dtl2'). Two
helpers are shared by all three: testutils.py, and time_masks.py, which draws the random
time-axis masks that the 1-d suites detrend.

Only the classes above are re-exported here. Everything else stays behind its subpackage,
e.g. detrending.lps2d.KnotVector, since the three algorithms have several same-named
pieces that mean different things (masks, in particular: time_masks.random_mask draws
along time, lps2d.masks.random_mask along frequency).
"""

from ..kernels import DetrenderLps2dParams, GpuDetrenderLps1d, GpuDetrenderLps2d

from .lps1d import ReferenceDetrenderLps1d
from .kf1d import ReferenceDetrenderKf1d
from .lps2d import ReferenceDetrenderLps2d
