## Method injections

Sometimes, we want to write methods in python, and add them to a pybind11-wrapped
C++ class. This can be done with the class decorator `ksgpu.inject_methods`:

```
    # Assume MyClass is a C++ class bound via Pybind11
    from ../ import MyClass 

    @ksgpu.inject_methods(MyClass)
    class MyClassInjections:
        # 1. Injecting a standard method
        def nice_print(self):
            print(f"Value: {self.get_value()}")

        # 2. Injecting/Overriding the Constructor (__init__)
        # We save the original C++ init so we can call it.
        _cpp_init = MyClass.__init__

        def __init__(self, value, label="Default"):
            # Call the C++ constructor
            self._cpp_init(value)
            # Add pure-Python attributes 
            # (Requires py::dynamic_attr() in C++)
            self.label = label
```

## Source file organization

Pybind11 code is in the following source files:
```
   src_pybind11/pirate_pybind11_casm.cpp
   src_pybind11/pirate_pybind11_core.cpp
   src_pybind11/pirate_pybind11_kernels.cpp
   src_pybind11/pirate_pybind11_loose_ends.cpp
   src_pybind11/pirate_pybind11.cpp   # toplevel
```
These get compiled into a single extension module `pirate_pybind11.so`.

On the python side, the `pirate_frb` toplevel package is divided into subpackages `pirate_frb.casm`, `pirate_frb.core`, etc.
Each subpackage imports the contents of the corresponding pybind11 file.
For example, `pirate_frb/casm/__init__.py` contains the line:
```py
# Note: CasmBeamformer is the only class in src_pybind11/pirate_pybind11_casm.cu.
from ..pirate_pybind11 import CasmBeamformer
```
Note that each pybind11 class will appear with two names -- for example `pirate_frb.pirate_pybind11.CasmBeamformer`
is the same as `pirate_frb.casm.CasmBeamformer`. In python code, always use the latter "non-pybind11" name if possible.

If you are asked to python-bind a new class, please make sure that it is also imported into a python subpackage,
and documented (with `autoclass`) in the sphinx docs.

## General notes

- Please write docstrings in the pybind11 code, but keep them concise and avoid superficial comments. If the meaning of a member/method is self-evident, then don't write a docstring.

- When writing docstrings, comments in the larger C++ codebase may be useful. For example, when writing python bindings for `class pirate::BumpAllocator`, comments in the source files `include/pirate/BumpAllocator.hpp` and `src_lib/BumpAllocator.cpp` may be useful for writing docstrings.

- If it's technically challenging (or awkward) to python-bind a C++ class member/method, or if the member/method seems unlikely to be useful from python, then skip it. Please list in the chat all "skipped" members/methods.

- Put all method injections in `pirate_frb/pybind11_injections.py`.

- If a class has method injections, then add a C++ comment to the pybind11 code with a concise description of the injections.

- Don't use lambda-functions in cases where a named function (or constructor) would be equivalent.

- Some C++ classes have protected constructors and a public static method `shared_ptr<X> X::create(...)`. In such cases, the python syntax should be `x = X(...)`, not `x = X.create(...)`.

## Specific argument types

- The rules below usually require method injections to implement. They apply to both constructors and non-constructor methods.

- If a C++ function takes an `aflags` argument (from `ksgpu/mem_utils.hpp`), then call `ksgpu.parse_aflags(aflags)` before passing it to the C++ function.

- If a C++ function takes a `ksgpu::Dtype` argument, then call the `ksgpu.Dtype` constructor on `x` before passing it to the C++ function.

- If a C++ function returns a bare pointer or `shared_ptr<void>`, then don't python-wrap it unless specifically requested.

- If a C++ member has type `dim3`, then don't python-wrap it unless specifically requested.

- If a C++ function has an argument of type `ostream &`, `YAML::Emitter &`, or `YamlFile &`, then don't python-wrap it unless specifically requested.

- C++ atomics must be converted to non-atomic types before converting to python.

- If a C++ function has a `cudaStream_t` argument, it should appear in python as `stream=None`, where `stream` is a `cupy.cuda.stream`, and the default is the current cupy stream.

- If a C++ function returns a `ksgpu::CudaStreamWrapper`, then the pybind11 binding will return type `ksgpu_pybind11._CudaStreamWrapperBase`. It should appear to a python caller as returning type `ksgpu.CudaStreamWrapper` instead. Define a python wrapper which does the conversion. (This is a one-liner: the `ksgpu.CudaStreamWrapper` constructor takes a `_CudaStreamWrapperBase` argument. See `ksgpu/ksgpu/CudaStreamWrapper.py` for more context.)
