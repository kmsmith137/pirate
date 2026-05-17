#!/bin/bash
set +e
set +x

touch src_pybind11/pirate_pybind11.cpp
touch src_pybind11/pirate_pybind11_core.cpp
touch src_pybind11/pirate_pybind11_kernels.cpp
touch src_pybind11/pirate_pybind11_casm.cpp
touch src_pybind11/pirate_pybind11_loose_ends.cpp

time make src_pybind11/pirate_pybind11.o
time make src_pybind11/pirate_pybind11_core.o
time make src_pybind11/pirate_pybind11_kernels.o
time make src_pybind11/pirate_pybind11_casm.o
time make src_pybind11/pirate_pybind11_loose_ends.o
