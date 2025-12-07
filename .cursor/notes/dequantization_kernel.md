# Context Restoration: GpuDequantizationKernel

## Introduction

**Date: December 7, 2025**

We implemented `GpuDequantizationKernel`, a CUDA kernel that converts contiguous int4 arrays of shape (B, F, T) — where B=beams, F=frequencies, T=time samples — to either float32 or float16 output arrays. The kernel is optimized for maximum GPU memory bandwidth through fully coalesced reads and 128-bit writes. We started from a spec (`spec.md`), elaborated it into a detailed implementation plan (`todo.md`), then iteratively refined the plan with optimizations documented in `wide_io.md`. The unit test passed on the first try. During timing tests, we discovered and fixed a grid dimension issue for large F values. We then added further optimizations: 128-bit stores using `float4` and `uint4`, and (32,8) thread blocks for better occupancy. Finally, we factored out the fp16 reinterpret_cast hackery into a reusable helper in `ksgpu/include/ksgpu/device_fp16.hpp`.

---

## Architecture & Patterns

### Why this kernel structure?

- **Follows existing patterns**: Modeled after `TreeGriddingKernel` and other pirate kernels with a struct containing `launch()`, `apply_reference()`, `test()`, and `time()` methods.
- **No separate Params struct**: Unlike `TreeGriddingKernel`, this kernel has no auxiliary data (like channel maps), so constructor args `(dtype, nbeams, nfreq, ntime)` suffice.
- **BandwidthTracker integration**: Tracks theoretical bandwidth usage for performance analysis.

### Why 128-bit stores?

- **Maximize memory bandwidth**: Modern GPUs (A100, H100) achieve peak bandwidth with 128-bit transactions.
- **fp32 kernel**: Uses `float4` (2 shuffles, 2 transactions) instead of the original plan (8 shuffles, 8 transactions).
- **fp16 kernel**: Uses `uint4` with **zero** warp shuffles — thread t already has elements [8t, 8t+7] after the coalesced read, matching the 128-bit write pattern perfectly.

### Why (32,8) thread blocks?

- **Better occupancy**: 256 threads per block (vs 32) allows more warps in flight.
- **threadIdx.y over frequency**: Each block now processes 8 adjacent frequencies, with bounds checking for when nfreq isn't a multiple of 8.

### Why `ksgpu/device_fp16.hpp`?

The `uint4`-to-`__half2` reinterpret_cast pattern is useful beyond this kernel. Factoring it into `half4_load/store` and `half8_load/store` functions makes the code cleaner and enables reuse. Later additions to this file include `f16_align()`, `f16_blend()`, and `f16_perm()` for __half2 permutation operations using PTX `prmt` instructions.

---

## Changes Implemented

### New Files Created
- `include/pirate/GpuDequantizationKernel.hpp` — Header declaring the kernel struct
- `src_lib/GpuDequantizationKernel.cu` — Implementation with CPU reference and two GPU kernels (fp32, fp16)
- `ksgpu/include/ksgpu/device_fp16.hpp` — Reusable fp16 load/store helpers and permutation functions

### Files Modified
- `Makefile` — Added `GpuDequantizationKernel.cu` to LIB_SRCFILES, added header to HFILES
- `src_pybind11/pirate_pybind11.cu` — Added pybind11 bindings for `test()` and `time()`
- `pirate_frb/__main__.py` — Added `--gdqk` flag to `parse_test()` and `parse_time()`

### Documentation Created
- `spec.md` — Original feature specification
- `todo.md` — Detailed implementation plan (512 lines)
- `wide_io.md` — 128-bit store optimization plan

---

## Challenges

### 1. Grid Dimension Limits (SOLVED)

**Problem**: The timing test failed with F=262144 because the original grid layout `{T/256, F, B}` placed F in `gridDim.y`, which has a maximum of 65535.

**Solution**: Changed grid to `{F, T/256, B}`. The x dimension has a limit of 2³¹-1, easily accommodating large F. Updated both kernels to use `blockIdx.x` for frequency and `blockIdx.y` for time chunks.

### 2. Coalesced Writes with Different Output Dtypes (SOLVED)

**Problem**: After a coalesced int4 read, thread t has elements [8t, 8t+7]. For coalesced float32 writes, thread t needs elements at stride 32 (t, t+32, t+64, ...). This mismatch requires data redistribution.

**Solution**: 
- **Original plan**: Use `__shfl_sync()` to transpose data before int4→float conversion.
- **Better solution (wide_io.md)**: Use 128-bit stores where each thread writes 4 consecutive floats. For fp32, this requires only 2 shuffles. For fp16, **zero shuffles** because thread t already has elements [8t, 8t+7] — exactly matching the 128-bit write pattern!

### 3. Signed int4 Interpretation (Clarified)

**Question**: Is int4 signed or unsigned?

**Resolution**: Assumed signed two's complement (range -8 to +7), which is typical for signal processing. The reference implementation converts nibbles ≥8 to negative values via `(nibble - 16)`.

### 4. Nibble Ordering (Verified)

**Question**: Which nibble corresponds to which array index?

**Resolution**: Verified from `utils.cu` example: low nibble = even index, high nibble = odd index. Index i is at byte `i/2`, nibble position `i%2`.

---

## Leftovers

### Partially Finished / Could Be Extended

1. **apply_reference() only outputs float32**: By design, the CPU reference always produces float32 even when the GPU kernel outputs float16. The test handles this by comparing float32 reference against the GPU output (which `assert_arrays_equal` handles via dtype conversion).

2. **No bfloat16 support**: Spec mentioned only float32/float16. Adding bfloat16 would follow the same fp16 pattern but use `__nv_bfloat16` types.

3. **No non-contiguous array support**: Both input and output must be fully contiguous. This is enforced by asserts in `apply_reference()` and `launch()`.

### Potential Risks

1. **Timing accuracy for small instances**: The test uses random small instances. For profiling purposes, use `time()` which creates 4GB output arrays.

2. **Grid dimension limits on T and B**: We moved F to `gridDim.x` to support large F, but `gridDim.y` (T/256) and `gridDim.z` (B) still have a 65535 limit. With T divisible by 256, this means T ≤ 16M and B ≤ 65535 — should be fine for practical use.

3. **16-byte alignment for 128-bit stores**: GPU allocations are typically 256-byte aligned, so this should be automatic. If issues arise, add explicit alignment checks.

---

## Quick Reference

```bash
# Build
make -j 32

# Run unit tests (random small instances)
python -m pirate_frb test --gdqk -n 10

# Run timing benchmark (4GB output arrays)
python -m pirate_frb time --gdqk
```

**Expected timing results**: Should approach theoretical GPU memory bandwidth (~900 GB/s A100, ~2000 GB/s H100). The fp16 kernel may be slightly faster due to requiring zero warp shuffles.

