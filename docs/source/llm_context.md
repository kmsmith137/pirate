# LLM context files

The markdown files below document low-level pieces of pirate. They double as
context files for LLM coding assistants -- the per-repo CLAUDE.md instructs the
assistant to read the relevant one before working on a given area (e.g. C++ code
or GPU kernels).

```{toctree}
---
maxdepth: 1
---
notes/cpp
notes/gpu_kernels
notes/pybind11
notes/grpc
notes/docstrings
notes/stoppable_class
notes/thread_backed_class
notes/cuda_event_ringbuf
```
