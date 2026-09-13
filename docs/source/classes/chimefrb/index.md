# chimefrb

Pieces of the old CHIME FRB search, ported to pirate: a reader for its data files, so that
old data can be processed with pirate, GPU ports of the transforms in its production
RFI chain, and the two containers that chain them. See
[notes/chimefrb](../../notes/chimefrb.md) for the porting rules this subpackage follows,
and for how it is spot-checked against the original code.

Every transform is a subclass of `GpuTransformBase` (the module docstring of
`pirate_frb.chimefrb.transform_io` states the whole interface): it processes one
`(nbeams, nfreq, ntime)` block through `launch(intensity, weights, scratch, stream=None)`,
which the base class supplies and which checks its arguments before running the
transform's `launch_checked()`, and it reads and writes a yaml form through
`to_yaml_dict()` / `from_yaml_dict()`. The five ported transforms are C++; a transform
written in python subclasses `GpuPythonTransform`, a plain python class on top of the base. `WiPipeline` runs a list of transforms in order;
`RfiMaskPipeline` runs a list on a downsampled copy of the data and feeds the mask back.
Both read the old rf_pipelines json configs (`WiPipeline.read_json_file`), and
`pirate_frb cfrb json2yaml` converts one to yaml.

**Writing your own transform.** Subclass `GpuPythonTransform`: a constructor that calls
`super().__init__(nbeams, nfreq, ntime)`, a `launch_checked(intensity, weights, scratch)`
that does the work in cupy, in place, and a `to_yaml_dict()` / `from_yaml_dict()` pair:

```python
class MyTransform(GpuPythonTransform):
    def __init__(self, nbeams, nfreq, ntime, sigma=3.0):
        super().__init__(nbeams, nfreq, ntime)
        self.sigma = float(sigma)

    def launch_checked(self, intensity, weights, scratch):
        ...   # cupy code, in place; the pipeline's stream is current

    def to_yaml_dict(self):
        return {'class_name': 'MyTransform', 'sigma': self.sigma}

    @classmethod
    def from_yaml_dict(cls, d, nbeams, nfreq, ntime):
        cls.check_yaml_keys(d, ['sigma'])
        return cls(nbeams, nfreq, ntime, sigma=d['sigma'])
```

`ExampleCupyTransform` is a complete example in forty lines, and the `GpuPythonTransform`
docstring states the contract (`launch_checked()` gets checked cupy arrays, views of the
caller's, with the pipeline's stream current). `GpuPythonTransform` is plain python, in
`pirate_frb/chimefrb/GpuPythonTransform.py`; what it inherits from the C++ base is python
too (`launch()`, in `GpuTransformBase.py`) except the argument checking, and a failed check
raises `RuntimeError` with a message starting with your class's name
(`MyTransform.launch(): expected 'weights' of shape (1, 64, 64), got (1, 64, 32)`).
Forgetting `super().__init__()` is a `TypeError` at construction; forgetting
`launch_checked()` is a `NotImplementedError` at the first launch. A class defined outside
`pirate_frb.chimefrb` is handed to the yaml reader as `classes=[MyTransform]`.

| Class | Description |
|---|---|
| [`AssembledChunk`](AssembledChunk.md) | One "assembled_chunk in msgpack format" data file, and its decode methods |
| [`GpuBadChannelMask`](GpuBadChannelMask.md) | Zeroes the weights of whole frequency channels (a port of `rf_pipelines::badchannel_mask`) |
| [`GpuTransformBase`](GpuTransformBase.md) | Base class of every transform, C++ or python: the geometry and the checked `launch()` |
| [`GpuPythonTransform`](GpuPythonTransform.md) | Base class of a transform written in python: what to define, and the contract `launch_checked()` gets |
| [`GpuClipperBase`](GpuClipperBase.md) | What the chimefrb RFI clippers share on top of `GpuTransformBase`: axis, downsampling, the per-row statistic |
| [`GpuIntensityClipper`](GpuIntensityClipper.md) | Zeroes the weights of samples more than `sigma` standard deviations from a weighted mean; the chain's principal flagger (a port of `rf_kernels::intensity_clipper`) |
| [`GpuPolynomialDetrender`](GpuPolynomialDetrender.md) | Fits and subtracts a polynomial in time per channel and chunk, zeroing the weights of poorly conditioned rows (a port of `rf_pipelines::polynomial_detrender`) |
| [`GpuStdDevClipper`](GpuStdDevClipper.md) | Zeroes channels or time samples whose variance is an outlier (a port of `rf_kernels::std_dev_clipper`) |
| [`GpuSplineDetrender`](GpuSplineDetrender.md) | Fits and subtracts a regularized cubic spline in frequency, per time sample (a port of `rf_kernels::spline_detrender`) |
| [`GpuWeightUpsampler`](GpuWeightUpsampler.md) | Zeroes the full-resolution weights under masked low-resolution cells (a port of `rf_kernels::weight_upsampler`) |
| [`GpuWiDownsampler`](GpuWiDownsampler.md) | Reduces an (intensity, weights) pair by `(Df, Dt)`, summing the weights rather than averaging them (a port of `rf_kernels::wi_downsampler`) |
| [`GpuWrms`](GpuWrms.md) | The weighted mean and variance of each row, refined by iterated sigma clipping; the statistic both clippers are built on (a port of `rf_kernels::weighted_mean_rms`) |
| [`WiPipeline`](WiPipeline.md) | Runs a list of transforms in order on one block (a port of `rf_pipelines::pipeline`) |
| [`RfiMaskPipeline`](RfiMaskPipeline.md) | Runs a list of transforms on a downsampled copy and feeds the mask back (a port of `rf_pipelines::wi_sub_pipeline`) |
| [`ExampleCupyTransform`](ExampleCupyTransform.md) | A worked example of a cupy transform: a 3-sigma clip per channel |

```{toctree}
:hidden:
:maxdepth: 1

AssembledChunk
GpuTransformBase
GpuPythonTransform
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
ExampleCupyTransform
```
