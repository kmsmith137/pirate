# Pybind11

## Method injections

Sometimes, we want to write methods in python, and add them to a pybind11-wrapped
C++ class. This can be done with the class decorator `ksgpu.inject_methods`:

```
    # Assume MyClass is a C++ class bound via Pybind11
    from my_module import MyClass 

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
   src_pybind11/pirate_pybind11_avar.cpp
   src_pybind11/pirate_pybind11_casm.cpp
   src_pybind11/pirate_pybind11_chime.cpp
   src_pybind11/pirate_pybind11_core.cpp
   src_pybind11/pirate_pybind11_kernels.cpp
   src_pybind11/pirate_pybind11_loose_ends.cpp
   src_pybind11/pirate_pybind11_utils.cpp
   src_pybind11/pirate_pybind11.cpp   # toplevel
```
These get compiled into a single extension module `pirate_pybind11.so`.

On the python side, the `pirate_frb` toplevel package is divided into subpackages `pirate_frb.casm`, `pirate_frb.core`, etc.
Each subpackage imports the contents of the corresponding pybind11 file.
For example, `pirate_frb/casm/__init__.py` contains the lines:
```py
# Import C++ class from pirate_pybind11
from ..pirate_pybind11 import CasmBeamformer
```
Note that each pybind11 class will appear with two names -- for example `pirate_frb.pirate_pybind11.CasmBeamformer`
is the same as `pirate_frb.casm.CasmBeamformer`. In python code, always use the latter "non-pybind11" name if possible.

If you are asked to python-bind a new class, please make sure that it is also imported into a python subpackage,
and documented (with `autoclass`) in the sphinx docs.

## General notes

- Please write docstrings in the pybind11 code, but keep them concise and avoid superficial comments. If the meaning of a member/method is self-evident, then don't write a docstring.

- When writing docstrings, comments in the larger C++ codebase may be useful. For example, when writing python bindings for `class pirate::BumpAllocator`, comments in the source files `include/pirate/BumpAllocator.hpp` and `src_lib/BumpAllocator.cpp` may be useful for writing docstrings.

- For how to write docstrings that render well in the Sphinx docs (documenting class members, the napoleon gotchas, etc.), see `notes/docstrings.md`.

- If it's technically challenging (or awkward) to python-bind a C++ class member/method, or if the member/method seems unlikely to be useful from python, then skip it. Please list in the chat all "skipped" members/methods.

- Put method injections in `pirate_frb/pybind11_injections.py`. Exception: `FrbGrouper`'s injections live in `pirate_frb/rpc/_FrbGrouper.py` (next to the RPC clients); that file's top comment notes the exception.

- If a class has method injections, then add a C++ comment to the pybind11 code with a concise description of the injections.

- A class's docstring may live either in the pybind11 binding or in the `inject_methods` injector class (an injector docstring overrides the pybind11 one). Keep it on exactly one side and put a pointer comment on the other. See "Class docstrings for classes with method injections" in `notes/docstrings.md` for the policy and how to choose.

- Don't use lambda-functions in cases where a named function (or constructor) would be equivalent.

- Some C++ classes have protected constructors and a public static method `shared_ptr<X> X::create(...)`. In such cases, the python syntax should be `x = X(...)`, not `x = X.create(...)`.

## GIL rules

We call C++ from multithreaded python, so bindings should release the GIL
whenever it is safe: a long-running or blocking binding that holds the GIL
stalls every other python thread (and in the worst case deadlocks, see below).

Background facts, both load-bearing for the rules:

- `py::call_guard<py::gil_scoped_release>` releases the GIL only around the
  function BODY: pybind11 converts arguments before releasing, converts the
  return value after reacquiring, and the argument casters outlive the call.
  So argument/return conversion is always safe; only the body runs GIL-free.

- A `ksgpu.Array` converted from python keeps the source numpy/cupy object
  (and its DLPack managed tensor) alive via a `shared_ptr` deleter that
  acquires the GIL itself (see `release_imported_array_base()` in ksgpu's
  `src_pybind11/pybind11_utils.cpp`), so the last copy of such an Array may
  be destroyed on any thread, with or without the GIL. Two caveats: worker
  threads holding python-backed Arrays must be drained/joined before
  interpreter shutdown, and the last reference must not be dropped while
  holding a C++ mutex that a GIL-holding thread might block on (lock
  inversion).

The rules:

- Default policy: a binding whose body is pure C++ and can block or run long
  (cuda kernel launch/sync, cudaMemcpy, condition-variable or queue waits,
  file/network I/O, large CPU loops, `test_*`/`time_*` entry points) gets
  `py::call_guard<py::gil_scoped_release>()`. Releasing pays even when not
  logically required. (`py::init` accepts a call_guard too.)

- Mandatory, not optional: a binding that blocks on progress driven by
  ANOTHER python thread (condition-variable waits, event-ringbuf waits,
  blocking queue pops) MUST release the GIL. Otherwise the design deadlocks:
  the blocked thread holds the GIL, so the thread that would wake it can
  never run.

- Never release the GIL around python-touching code (`py::object` casts,
  `py::list`/`py::cast`/`py::none` construction, python C API). If a lambda
  must cast a `py::object` argument, cast FIRST (GIL held), then wrap only
  the C++ compute in a scoped block: `{ py::gil_scoped_release nogil; ... }`.
  Never put `py::call_guard` on such a lambda.

- Leave unguarded: trivial getters/setters, and properties that return
  Arrays or build python containers (the conversion needs the GIL anyway).

- Avoid dropping the last reference to a python-sourced Array inside a
  released region (e.g. by reassigning an Array argument in the body). It is
  safe (the deleter takes the GIL), but needlessly re-enters python; prefer
  working on a local copy.

- Before adding a call_guard to an existing binding, read the wrapped C++
  implementation: check for python API usage, callbacks into python, and
  what can block.

## Array conversion: streams, validation, overloads

- Array conversion is zero-copy and does NO cuda stream synchronization:
  ksgpu has no notion of a "current stream", and does not associate streams
  (or events) with arrays. Ordering between asynchronous GPU work and the
  other side of a conversion is always the caller's/binding's
  responsibility: keep producer and consumer on the same stream, or
  synchronize explicitly before the array crosses the language boundary.
  (See the SYNCHRONIZATION WARNING comment in ksgpu's
  `src_pybind11/pybind11_utils.cpp`.)

- The caster accepts both numpy and cupy arrays for any `Array<T>` argument.
  A binding that dereferences `Array::data` on the host (memcpy, loops) must
  `xassert(arr.on_host())` -- a cupy argument would otherwise be a host
  dereference of a device pointer -- and should validate the array size
  before indexing or copying.

- Binding lambdas that allocate small pure-CPU return arrays should use
  `af_uhost` (plain malloc), not `af_rhost`: the latter costs a
  `cudaHostAlloc` per call and requires a cuda context.

- Don't define pybind11 overloads that differ only in Array dtype: the ksgpu
  caster throws on dtype mismatch instead of soft-failing, so overload
  fallthrough does not work. Overload on arity only; dtype-generic functions
  should take `Array<void>` plus a runtime `ksgpu::Dtype`. (If C++-side
  dtype overloads are ever truly needed, the fix is to make the caster honor
  pybind11's two-pass 'convert' flag -- soft-fail in the first pass, throw
  the detailed error in the second -- not to blanket soft-fail.)

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
