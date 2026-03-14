# GPU kernels

## Register assignment notation

Register assignment notation is a way of indicating in comments how a **logical**
multidimensional array is **physically** distributed across the warps, threads (lanes),
registers, and simd lanes of a GPU kernel.

It's easiest to explain the details by example:
```cpp
// This code assumes 16 warps, blockDim = {32,16}.
uint laneId = threadIdx.x;
uint warpId = threadIdx.y;

// Pointer to a 3-d array in global GPU memory p[i,j,k], shape (16,16,16).
__half *p = ...;

// Read data from global GPU memory
uint s = 2*laneId + 256*warpId;
__half2 x0 = *((__half2 *) (p+s));
__half2 x1 = *((__half2 *) (p+s+64));
__half2 x2 = *((__half2 *) (p+s+128));
__half2 x3 = *((__half2 *) (p+s+192));

// x has the following register assigment.
// See below for an explanation of this "register assignment" notation.
//   simd:      s0              <->  k0
//   register:  r1 r0           <->  j3 j2
//   thread:    t4 t3 t2 t1 t0  <->  j1 j0 k3 k2 k1
//   warp:      w3 w2 w1 w0     <->  i3 i2 i1 i0
```
The meaning of the register assignment notation is as follows.
Each 16-bit element of the array p[i,j,k] has a "logical" location which is parameterized
by four integers (i,j,k), whose base-2 representations consist of four bits each:
```
i = (i3 i2 i1 i0)_2    j = (j3 j2 j1 j0)_2    k = (k3 k2 k1 k0)_2
```
Thus, each logical array location is parameterized by **12 index bits**, each of which can
be 0 or 1.

After being loaded from global memory, each array element is physically located on a particular
(warp, thread, register, simd lane) quadruple. Thus, each array element has a "physical" location
which is parameterized by four integers (w,t,r,s), with base-two representations:
```
s = (s0)_2    r = (r1 r0)_2    t = (t4 t3 t2 t1 t0)_2    w = (w3 w2 w1 w0)_2
```
In this example, there are 16 warps, 32 threads, 4 registers per thread, and 2 simd lanes
per register (`__half2` dtype). Thus, each physical array location is also parameterized by
**12 index bits**.

The "register assignment" notation in the comment above specifies a mapping between physical
hardware locations and logical array locations, via a mapping between the 12 physical and
logical index bits (on the LHS and RHS of the `<->` symbol). This is a flexible notation which
fully specifies how data is distributed on the GPU hardware, and can describe a wide range
of possible mappings.

## Local transpose

The "local transpose" is a thread-local operation which exchanges a "simd" bit for a "register" bit.
It's easiest to explain by example:
```cpp
// Continuing the example above, suppose we have a 3-d array x[i,j,k] with the following
// register assignment. (Note that we sometimes omit the bits s_i, r_i, t_i, w_i on the
// LHS for brevity.)
//   simd:      k0
//   register:  j3 j2
//   thread:    j1 j0 k3 k2 k1
//   warp:      i3 i2 i1 i0

__half2 x0, x1, x2, x3;

// The following "local transpose" operation exchanges physical bits s0 (simd)
// with r1 (register). Recall that the cuda intrinsic `__lows2half2()` combines
// the lower halves of two `__half2` values into a new `__half2`, and analogously
// for `__highs2half2()`.

__half2 y0 = __lows2half2(x0, x2);
__half2 y2 = __highs2half2(x0, x2);
__half2 y1 = __lows2half2(x1, x3);
__half2 y3 = __highs2half2(x1, x3);

// Now the y array is the same logical array as x, but with a new register assignment:
//   simd:      j3
//   register:  k0 j2
//   thread:    j1 j0 k3 k2 k1   (unchanged)
//   warp:      i3 i2 i1 i0      (unchanged)
```

Here is another local transpose example, using 8-bit data packed into `uint` registers:
```cpp
// In this example, suppose we have an 8-bit array which has been packed into uint
// registers (i.e. 4 simd lanes per register), with register assignment:
//   simd:      i1 i0
//   register:  i2

uint x0, x1;

// The following "local transpose" operation exchanges physical bits s0 (simd)
// with r0 (register). Recall that the cuda intrinsic __byte_perm(a, b, sel)
// treats the concatenation of a (bytes 0-3) and b (bytes 4-7) as an 8-byte
// array, and selects 4 bytes using the selector sel. Each nibble of sel (from
// least to most significant) specifies which source byte (0-7) goes to the
// corresponding output byte position.

uint y0 = __byte_perm(x0, x1, 0x6240);
uint y1 = __byte_perm(x0, x1, 0x7351);

// Now the y-array contains the same data as x, but with a new register assignment:
//   simd:      i1 i2
//   register:  i0

// Note: to exchange s1 (rather than s0) with r0, we would replace the
// constants (0x6240, 0x7351) with (0x5410, 0x7632).
```

## Warp transpose

The "warp transpose" is a warp-local operation which exchanges a "register" bit
with a "thread" bit. It's easiest to explain by example:
```cpp
// Continuing the example above, 'y' has register assignment:
//   simd:      j3
//   register:  k0 j2
//   thread:    j1 j0 k3 k2 k1
//   warp:      i3 i2 i1 i0

__half2 y0, y1, y2, y3;

// The following "warp transpose" operation exchanges physical bits r0 (register)
// with t2 (thread).

// Thread with t2=0 keeps its y0, receives partner's y0 into y1
// Thread with t2=1 receives partner's y1 into y0, keeps its y1

__half2 tmp = (threadIdx.x & 0x4) ? y0 : y1;  // bit mask for t2
tmp = __shfl_sync(~0u, tmp, threadIdx.x ^ 0x4);
y0 = (threadIdx.x & 0x4) ? tmp : y0;
y1 = (threadIdx.x & 0x4) ? y1 : tmp;

tmp = (threadIdx.x & 0x4) ? y2 : y3;
tmp = __shfl_sync(~0u, tmp, threadIdx.x ^ 0x4);
y2 = (threadIdx.x & 0x4) ? tmp : y2;
y3 = (threadIdx.x & 0x4) ? y3 : tmp;

// Now the y-array has been shuffled into a new register assignment:
//   simd:      j3               (unchanged)
//   register:  k0 k3
//   thread:    j1 j0 j2 k2 k1
//   warp:      i3 i2 i1 i0      (unchanged)
```
Local and warp transposes can be used as "building blocks", to build more complicated shuffling
operations via composition.
