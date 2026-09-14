# chimefrb

Pieces of the old CHIME FRB search, ported to pirate: a reader for its data files, so that
old data can be processed with pirate, GPU ports of the transforms in its production
RFI chain, and the two containers that chain them. See
[notes/chimefrb](../../notes/chimefrb.md) for the porting rules this subpackage follows,
and for how it is spot-checked against the original code.

Every transform is a subclass of `GpuTransform` (the module docstring of
`pirate_frb.chimefrb.utils` states the whole interface): it processes one
`(nbeams, nfreq, ntime)` block through `launch(intensity, weights, scratch, stream=None)`,
which the base class supplies and which checks its arguments before running the
transform's `launch_checked()`, and it reads and writes a yaml form through
`to_yaml_dict()` / `from_yaml_dict()`. The five ported transforms are C++; a transform
written in python subclasses `GpuPythonTransform`, a plain python class on top of the
base. `Pipeline` runs a list of transforms in order; `RfiMaskPipeline` runs a list on a
downsampled copy of the data and feeds the mask back. Both read the old rf_pipelines json
configs (`Pipeline.read_json_file`), and `pirate_frb cfrb json2yaml` converts one to yaml.
`RfiMaskExtractor` marks the point in a chain where the RFI mask is taken, and packs it;
`ChimePreDedisperser` drives a whole stream of chunks through a chain and hands back one
mask per chunk, which is how `pirate_frb cfrb reproduce_rfimask` compares pirate's masks
with the ones the telescope saved.

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

`ExamplePythonTransform` is a complete example in forty lines, and the `GpuPythonTransform`
docstring states the contract (`launch_checked()` gets checked cupy arrays, views of the
caller's, with the pipeline's stream current). `GpuPythonTransform` is plain python, in
`pirate_frb/chimefrb/GpuPythonTransform.py`; what it inherits from the C++ base is python
too (`launch()`, in `cpp_transforms.py`) except the argument checking, and a failed check
raises `RuntimeError` with a message starting with your class's name
(`MyTransform.launch(): expected 'weights' of shape (1, 64, 64), got (1, 64, 32)`).
Forgetting `super().__init__()` is a `TypeError` at construction; forgetting
`launch_checked()` is a `NotImplementedError` at the first launch. A class defined outside
`pirate_frb.chimefrb` is handed to the yaml reader as `classes=[MyTransform]`.

A transform that RUNS other transforms subclasses `GpuContainerBase` instead, one level
further down. That is how the yaml reader knows to pass `classes` on to it, so that the
elements it holds are resolved too.

The table and the sidebar below are ALPHABETICAL -- keep a new class in order rather than
next to its relatives. The subpackage is large enough that finding a name beats reading a
grouping, and the paragraphs above are where the relationships are explained.

| Class | Description |
|---|---|
| [`AssembledChunk`](AssembledChunk.md) | One "assembled_chunk in msgpack format" data file, and its decode methods |
| [`AssembledChunkReader`](AssembledChunkReader.md) | Reads a list of those files with a thread pool, and hands them back in filename order |
| [`ChimeDequantizationKernel`](ChimeDequantizationKernel.md) | Turns one chunk's raw arrays into the (intensity, weights) pair a chain runs on, on the GPU |
| [`ChimePreDedisperser`](ChimePreDedisperser.md) | Runs a chain on a stream of chunks and hands back one RFI mask per chunk: the driver around everything else here |
| [`ExamplePythonTransform`](ExamplePythonTransform.md) | A worked example of a cupy transform: a 3-sigma clip per channel |
| [`GpuBadChannelMask`](GpuBadChannelMask.md) | Zeroes the weights of whole frequency channels (a port of `rf_pipelines::badchannel_mask`) |
| [`GpuClipperBase`](GpuClipperBase.md) | What the chimefrb RFI clippers share on top of `GpuTransform`: axis, downsampling, the per-row statistic |
| [`GpuContainerBase`](GpuContainerBase.md) | Base class of a transform that RUNS other transforms: what `Pipeline` and `RfiMaskPipeline` share |
| [`GpuIntensityClipper`](GpuIntensityClipper.md) | Zeroes the weights of samples more than `sigma` standard deviations from a weighted mean; the chain's principal flagger (a port of `rf_kernels::intensity_clipper`) |
| [`GpuPolynomialDetrender`](GpuPolynomialDetrender.md) | Fits and subtracts a polynomial in time per channel and chunk, zeroing the weights of poorly conditioned rows (a port of `rf_pipelines::polynomial_detrender`) |
| [`GpuPythonTransform`](GpuPythonTransform.md) | Base class of a transform written in python: what to define, and the contract `launch_checked()` gets |
| [`GpuSplineDetrender`](GpuSplineDetrender.md) | Fits and subtracts a regularized cubic spline in frequency, per time sample (a port of `rf_kernels::spline_detrender`) |
| [`GpuStdDevClipper`](GpuStdDevClipper.md) | Zeroes channels or time samples whose variance is an outlier (a port of `rf_kernels::std_dev_clipper`) |
| [`GpuTransform`](GpuTransform.md) | Base class of every transform, C++ or python: the geometry and the checked `launch()` |
| [`GpuWiDownsamplingKernel`](GpuWiDownsamplingKernel.md) | Reduces an (intensity, weights) pair by `(Df, Dt)`, summing the weights rather than averaging them (a port of `rf_kernels::wi_downsampler`) |
| [`GpuWrmsKernel`](GpuWrmsKernel.md) | The weighted mean and variance of each row, refined by iterated sigma clipping; the statistic both clippers are built on (a port of `rf_kernels::weighted_mean_rms`) |
| [`GpuWtUpsamplingKernel`](GpuWtUpsamplingKernel.md) | Zeroes the full-resolution weights under masked low-resolution cells (a port of `rf_kernels::weight_upsampler`) |
| [`Pipeline`](Pipeline.md) | Runs a list of transforms in order on one block (a port of `rf_pipelines::pipeline`) |
| [`RfiMaskExtractor`](RfiMaskExtractor.md) | The transform whose position in a chain defines the RFI mask, and which packs it (the old `mask_counter` in its mask-saving role) |
| [`RfiMaskPackingKernel`](RfiMaskPackingKernel.md) | Packs the weights a chain leaves behind into a data file's bit-packed RFI mask, on the GPU |
| [`RfiMaskPipeline`](RfiMaskPipeline.md) | Runs a list of transforms on a downsampled copy and feeds the mask back (a port of `rf_pipelines::wi_sub_pipeline`) |

```{toctree}
:hidden:
:maxdepth: 1

AssembledChunk
AssembledChunkReader
ChimeDequantizationKernel
ChimePreDedisperser
ExamplePythonTransform
GpuBadChannelMask
GpuClipperBase
GpuContainerBase
GpuIntensityClipper
GpuPolynomialDetrender
GpuPythonTransform
GpuSplineDetrender
GpuStdDevClipper
GpuTransform
GpuWiDownsamplingKernel
GpuWrmsKernel
GpuWtUpsamplingKernel
Pipeline
RfiMaskExtractor
RfiMaskPackingKernel
RfiMaskPipeline
```
