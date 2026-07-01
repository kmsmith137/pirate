# FRB pulse simulation, vendorized from the standalone 'simpulse' library (FRB single_pulse only;
# no pulsar phase_model / von_mises_profile). The C++ class + functions are defined in
# src_pybind11/pirate_pybind11_simpulse.cpp (see include/pirate/simpulse.hpp, src_lib/simpulse.cpp).
#
# Note: the test/plot helpers (test_pulse_upsampling, plot_pulses) are NOT imported here, so that
# 'import pirate_frb.simpulse' does not pull in matplotlib. Import them explicitly if needed
# (e.g. the 'pirate_frb test_simpulse' command does).

# Import C++ class + functions from pirate_pybind11
from ..pirate_pybind11 import (
    SinglePulse,
    dispersion_delay,
    scattering_time,
)
