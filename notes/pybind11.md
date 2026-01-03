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

## General notes

- Pybind11 code is in `src_pybind11/*.cu`.

- Please write docstrings in the pybind11 code, but keep them concise and avoid superficial comments. If the meaning of a member/method is self-evident, then don't write a docstring.

- When writing docstrings, comments in the larger C++ codebase may be useful. For example, when writing python bindings for `class pirate::BumpAllocator`, comments in the source files `include/pirate/BumpAllocator.hpp` and `src_lib/BumpAllocator.cu` may be useful for writing docstrings.

- If it's technically challenging (or awkward) to python-bind a C++ class member/method, or if the member/method seems unlikely to be useful from python, then skip it. Please list in the chat all "skipped" members/methods.

- Put all method injections in `pirate_frb/pybind11_injections.py`.

- If a class has method injections, then add a C++ comment to the pybind11 code with a concise description of the injections.

- Don't use lambda-functions in cases where a named function (or constructor) would be equivalent.

## Specific argument types

- The rules below usually require method injections to implement. They apply to both constructors and non-constructor methods.

- If a C++ function takes an `aflags` argument (from `ksgpu/mem_utils.hpp`), then call `ksgpu.parse_aflags(aflags)` before passing it to the C++ function.

- If a C++ function takes a `ksgpu::Dtype` argument, then call the `ksgpu.Dtype` constructor on `x` before passing it to the C++ function.

- If a C++ function returns a bare pointer, or `shared_ptr<void>`, then don't python-wrap it unless specifically requested.
