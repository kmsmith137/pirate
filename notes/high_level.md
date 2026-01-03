## Project Overview

Pirate is a CUDA/C++ pipeline for Fast Radio Burst (FRB) detection, including:
- GPU-accelerated dedispersion kernels
- Python code generators that produce optimized CUDA kernels
- High-level python interface (`pirate_frb` module) via pybind11

Uses the `chord` branch of the ksgpu library for the Array class, and other helpers.
Please complain if you don't see the ksgpu library in the vscode workspace.


## Chunks, batches, frames, and segments

Throughout the code:
- A "chunk" (or "time chunk") is a range of time indices. The chunk size (e.g. 1024 or 2048) is defined in `DedispersionConfig::time_samples_per_chunk`.
- A "batch" (or "beam batch") is a range of beam indices. The batch size (e.g. 1,2,4) is defined in `DedispersionConfig::beams_per_batch`.
- A "frame" is a (chunk,beam) pair (not a (chunk,batch) pair!). Frames are used in `class MegaRingbuf`, and will also be used in the front-end server code and its intensity ring buffer.
- A "segment" refers to a 128-byte, memory-contiguous subset of any array in GPU memory. Segments are used in low-level GPU kernels, and data structures which are GPU kernel adjacent (e.g. `DedispersionPlan`, `MegaRingbuf`).
