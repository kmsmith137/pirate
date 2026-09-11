# chimefrb

Pieces of the old CHIME FRB search, ported to pirate: a reader for its data files, so that
old data can be processed with pirate, and GPU ports of the transforms in its production
RFI chain. See [notes/chimefrb](../../notes/chimefrb.md) for the porting rules this
subpackage follows, and for how it is spot-checked against the original code.

| Class | Description |
|---|---|
| [`AssembledChunk`](AssembledChunk.md) | One "assembled_chunk in msgpack format" data file, and its decode methods |
| [`GpuBadChannelMask`](GpuBadChannelMask.md) | Zeroes the weights of whole frequency channels (a port of `rf_pipelines::badchannel_mask`) |
| [`GpuClipperBase`](GpuClipperBase.md) | What the chimefrb RFI clippers share: geometry, the per-row statistic, argument checking |
| [`GpuStdDevClipper`](GpuStdDevClipper.md) | Zeroes channels or time samples whose variance is an outlier (a port of `rf_kernels::std_dev_clipper`) |
| [`GpuSplineDetrender`](GpuSplineDetrender.md) | Fits and subtracts a regularized cubic spline in frequency, per time sample (a port of `rf_kernels::spline_detrender`) |

```{toctree}
:hidden:
:maxdepth: 1

AssembledChunk
GpuBadChannelMask
GpuClipperBase
GpuSplineDetrender
GpuStdDevClipper
```
