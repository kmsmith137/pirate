## pirate - Perimeter Institute RAdio Transient Engine

An experimental GPU-based fast transient search, intended for use in CHORD.

### Installation

1. Make sure you have cuda, cupy, curand, pybind11, yaml-cpp
installed. This conda environment ("heavycuda") works for me:
```
conda create -c conda-forge -n heavycuda \
         cupy scipy matplotlib pybind11 
         cuda-nvcc libcurand-dev yaml-cpp
```

If you have the cuda toolkit installed outside conda, then you can
omit some of these conda packages.

2. Install the `ksgpu` library (See instructions at https://github.com/kmsmith137/ksgpu).
(This library was previously named `gputils`, but I renamed it since that
name was taken on pypi etc.)

3. Install `pirate` with either:
```
      # Clone github repo and make 'editable' install
      git clone https://github.com/kmsmith137/pirate
      cd pirate
      pip install -v -e .
```
or:
```
     # Install from pypi
     pip install -v pirate_frb
```

Contact: Kendrick Smith <kmsmith@perimeterinstitute.ca>
