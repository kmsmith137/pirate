
## CUDA kernels

- Global memory bandwith is usually the most important bottleneck. If possible, ensure that each warp reads/writes entire coalesced cache lines, whenever it accesses global memory.

- Use coalesced, aligned, 64-bit (e.g. float2) or 128-bit (e.g. float4) loads/stores when possible. These instructions can significantly increase global memory bandwidth (compared to 32-bit).

- In float16 kernels, 64-bit and 128-bit loads/stores are awkward, since nvidia doesn't define the appropriate built-in simd type (__half4 or __half8). There are some useful helper functions (device forceinline) in ksgpu/include/ksgpu/device_fp16.hpp.

- **Grid dimension limits**: `gridDim.x` can be up to 2³¹-1, but `gridDim.y` and `gridDim.z` are limited to 65535. Put large dimensions (like nfreq which can be ~10⁵) in `gridDim.x`, not y or z.
