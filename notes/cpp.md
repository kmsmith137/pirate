## C++/cuda Guidelines

- Prefer private (or static) functions over anonymous scopes `{ ... }`.

- Use `//` comments, not `/* */`

- Use `long` for sizes and indices, not `int` or `size_t`

- Use spaces, not tabs.

- Functions which take an `ostream &` argument should not modify stream state in the caller.

## xassert macros

The following macros are similar to `assert()`, but throw an exception instead of terminating:
```
xassert(cond);           // throw exception unless bool(cond)==true
xassert_eq(x,y);         // throw exception unless x==y
xassert_divisible(x,y);  // throw exception unless (x % y) == 0
// also: xassert_ne(), xassert_lt(), xassert_le(), xassert_gt(), xassert_ge()

// Throw exception unless 'arr' (type ksgpu::Array) has shape {3,4,5}.
// Note parentheses around the shape.
xassert_shape_eq(arr, ({3,4,5}));
```
The exception text shows the file/line (like regular `assert()`), plus values of the arguments `x` and `y`.

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

    // Example 2: you should not replace this by xassert_lt(x+y,z), since the xassert_eq() error message
    // contains less information than the message below (which shows x,y,z individually)

    if ((x+y) >= z) {
        stringstream ss;
        ss << "f(): expected (x+y) < z, got x=" << x << ", y=" << ", z=" << z;
        throw runtime_error(ss.str());
    }
}
```
