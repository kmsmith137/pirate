# chimefrb

Pieces of the old CHIME FRB search, ported to pirate: a reader for its data files, so that
old data can be processed with pirate, GPU ports of the transforms in its production
RFI chain, and the two containers that chain them. See
[notes/chimefrb](../../notes/chimefrb.md) for the porting rules this subpackage follows,
and for how it is spot-checked against the original code.

Every transform follows one interface (the module docstring of
`pirate_frb.chimefrb.transform_io` states it): it processes one `(nbeams, nfreq, ntime)`
block through `launch(intensity, weights, scratch, stream=None)`, and reads and writes a
yaml form through `to_yaml_dict()` / `from_yaml_dict()`. `WiPipeline` runs a list of
transforms in order; `RfiMaskPipeline` runs a list on a downsampled copy of the data and
feeds the mask back. Both read the old rf_pipelines json configs
(`WiPipeline.read_json_file`), and `misc/chimefrb/configs/legacy_json_to_yaml.py` converts
one to yaml.

**Writing your own transform.** Subclass `CupyTransformBase`: a constructor that takes
`(nbeams, nfreq, ntime, ...)`, a `launch_checked()` that does the work in cupy, in place, and
a `to_yaml_dict()` / `from_yaml_dict()` pair. `ExampleCupyTransform` is a complete example
in forty lines. A class defined outside `pirate_frb.chimefrb` is handed to the yaml reader
as `classes=[MyTransform]`.

| Class | Description |
|---|---|
| [`AssembledChunk`](AssembledChunk.md) | One "assembled_chunk in msgpack format" data file, and its decode methods |
| [`GpuBadChannelMask`](GpuBadChannelMask.md) | Zeroes the weights of whole frequency channels (a port of `rf_pipelines::badchannel_mask`) |
| [`GpuClipperBase`](GpuClipperBase.md) | What the chimefrb RFI clippers share: geometry, the per-row statistic, argument checking |
| [`GpuIntensityClipper`](GpuIntensityClipper.md) | Zeroes the weights of samples more than `sigma` standard deviations from a weighted mean; the chain's principal flagger (a port of `rf_kernels::intensity_clipper`) |
| [`GpuPolynomialDetrender`](GpuPolynomialDetrender.md) | Fits and subtracts a polynomial in time per channel and chunk, zeroing the weights of poorly conditioned rows (a port of `rf_pipelines::polynomial_detrender`) |
| [`GpuStdDevClipper`](GpuStdDevClipper.md) | Zeroes channels or time samples whose variance is an outlier (a port of `rf_kernels::std_dev_clipper`) |
| [`GpuSplineDetrender`](GpuSplineDetrender.md) | Fits and subtracts a regularized cubic spline in frequency, per time sample (a port of `rf_kernels::spline_detrender`) |
| [`GpuWeightUpsampler`](GpuWeightUpsampler.md) | Zeroes the full-resolution weights under masked low-resolution cells (a port of `rf_kernels::weight_upsampler`) |
| [`GpuWiDownsampler`](GpuWiDownsampler.md) | Reduces an (intensity, weights) pair by `(Df, Dt)`, summing the weights rather than averaging them (a port of `rf_kernels::wi_downsampler`) |
| [`GpuWrms`](GpuWrms.md) | The weighted mean and variance of each row, refined by iterated sigma clipping; the statistic both clippers are built on (a port of `rf_kernels::weighted_mean_rms`) |
| [`WiPipeline`](WiPipeline.md) | Runs a list of transforms in order on one block (a port of `rf_pipelines::pipeline`) |
| [`RfiMaskPipeline`](RfiMaskPipeline.md) | Runs a list of transforms on a downsampled copy and feeds the mask back (a port of `rf_pipelines::wi_sub_pipeline`) |
| [`CupyTransformBase`](CupyTransformBase.md) | Base class for a transform written in cupy |
| [`ExampleCupyTransform`](ExampleCupyTransform.md) | A worked example of a cupy transform: a 3-sigma clip per channel |

```{toctree}
:hidden:
:maxdepth: 1

AssembledChunk
GpuBadChannelMask
GpuClipperBase
GpuIntensityClipper
GpuPolynomialDetrender
GpuSplineDetrender
GpuStdDevClipper
GpuWeightUpsampler
GpuWiDownsampler
GpuWrms
WiPipeline
RfiMaskPipeline
CupyTransformBase
ExampleCupyTransform
```
