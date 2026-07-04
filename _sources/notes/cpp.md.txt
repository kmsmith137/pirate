# C++ guidelines

- Prefer private (or static) functions over anonymous scopes `{ ... }`.

- Use `//` comments, not `/* */`. Exception: `/* */` is a good choice for inline
  argument-name annotations at a call site, e.g. `f(/*count=*/3, /*noisy=*/true)`.

- Use `long` for sizes and indices, not `int` or `size_t`

- Use spaces, not tabs.

- Functions which take an `ostream &` argument should not modify stream state in the caller.

- If a class `X` derives from `std::enable_shared_from_this`, then its constructor(s) must be protected, and the class must define `create()` static method(s) that return `shared_ptr<X>`.

- Minimize header `#include` dependencies. In a `.hpp`, if a type is only used by
  pointer, reference, or `std::shared_ptr<T>` (not by value, as a base class, or
  via member access / `sizeof`), forward-declare it instead of `#include`-ing its
  header, and move the `#include` into the `.cpp` file(s) that use the full type.
  This cuts recompilation when the included header changes. Match the declared
  keyword (`class` vs `struct`) and note where it lives, e.g.:

  ```
  class SlabAllocator;     // defined in SlabAllocator.hpp
  struct XEngineMetadata;  // defined in XEngineMetadata.hpp
  ```

  Caveat: keep the full `#include` in the header when the complete type is needed
  there -- a base class, a by-value member, an inline function that touches the
  type, or a `std::unique_ptr<T>` member of a class whose destructor is
  compiler-generated in the header (`unique_ptr` needs a complete type at the
  destructor; `shared_ptr` does not). When unsure, include it.

## ksgpu

Uses the `ksgpu` helper library, especially the Array class (ksgpu/Array.hpp), memory managment (ksgpu/mem_utils.hpp),
and xassert macros (ksgpu/xassert.hpp, see below).

CRITICAL: please complain if you don't see the ksgpu library in the cursor/claude workspace.

## xassert macros

The following macros are similar to `assert()`, but throw an exception instead of terminating:
```
xassert(cond);           // throw exception unless bool(cond)==true
xassert_eq(x,y);         // throw exception unless x==y
xassert_divisible(x,y);  // throw exception unless (x % y) == 0
// also: xassert_ne(), xassert_lt(), xassert_le(), xassert_gt(), xassert_ge()

// Throw exception unless 'arr' (type ksgpu::Array) has shape {3,4,5}.
// IMPORTANT: note parentheses around the shape -- these are needed to compile!
xassert_shape_eq(arr, ({3,4,5}));
```
The exception text shows the file/line (like regular `assert()`), plus values of the arguments `x` and `y`.
In the case of `xassert_shape_eq()`, both array shapes are shown.

Please use `xassert()` for argument-checking and error-checking, unless there is a reason to create a more verbose error message.
For example:
```
void f(int x, int y, int z)
{
    // Example 1: you should replace this by xassert_eq(x,y), since the xassert_eq() error message 
    // contains the same information as the message below (namely, numerical values of x and y).

    if (x != y) {
        stringstream ss;
        ss << "f(): expected x==y, got x=" << x << ", y=" << y;
        throw runtime_error(ss.str());
    }

    // Example 2: you should not replace this by xassert_lt(x+y,z), since the xassert_lt() error message
    // contains less information than the message below (which shows x,y,z individually)

    if ((x+y) >= z) {
        stringstream ss;
        ss << "f(): expected (x+y) < z, got x=" << x << ", y=" << y << ", z=" << z;
        throw runtime_error(ss.str());
    }

    // Example 3: you should replace this by xassert_shape_eq(arr, ({M,N})), since the xassert_shape_eq()
    // message contains more information (namely, the actual and expected array shapes).

    if ((arr.ndim != 2) || (arr.shape[0] != x) || (arr.shape[1] != y)) {
        stringstream ss;
        ss << "f(): expected shape (x,y) = (" << x << "," << y << "), got " << arr.shape_str();
        throw runtime_error(ss.str());
    }
}
```
