# Python class reference

This section is deliberately **incomplete**. It documents the classes a TYPICAL python
caller works with -- not the whole binding surface.

A class is left out when a caller only ever receives an instance from somewhere else and
never names the type (most of the `*Params` classes, `MegaRingbuf`), or when it exists for
tests and variance studies rather than for use (`GpuSbDedispersionKernel`,
`ReferencePfSquare`). Such a class carries a short comment at its pybind11 binding, or at
its python method-injector, saying it is omitted on purpose. Being referenced from a
docstring on some other page is NOT by itself a reason to add a page here: judge by who
uses the class, not by where its name appears.

This is a judgement call about typical usage, and typical usage changes -- so revisit it
when it stops matching how callers actually write code.

```{toctree}
---
maxdepth: 1
---
classes/AssembledChunk
classes/Acquisition
classes/AssembledFrame
classes/AssembledFrameAllocator
classes/AssembledFrameSet
classes/BumpAllocator
classes/DedispersionConfig
classes/DedispersionPlan
classes/DedispersionTree
classes/FakeXEngine
classes/FileSubscriber
classes/FileWriter
classes/FrbGrouper
classes/FrbGrouperClient
classes/FrbSearchClient
classes/FrbServer
classes/FrbSifterClient
classes/FrbSifterEvents
classes/FrequencySubbands
classes/GpuDedisperser
classes/GpuDedisperserOutputs
classes/GpuGrouperHistogram
classes/GrouperHistogram
classes/Hardware
classes/OfflineDedisperser
classes/PrimaryTree
classes/Receiver
classes/ReferenceDedisperser
classes/SimulatedFrameFactory
classes/SinglePulse
classes/SlabAllocator
classes/ThreadAffinity
classes/XEngineMetadata
```
