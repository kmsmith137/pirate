# Examples

Run these examples from the repository root, in the environment used to build
PIRATE. The offline and online searches require an NVIDIA GPU and the compiled
CHORD kernels. The classifier notebook uses saved data and the same PIRATE
GPU environment.

- [Simple FRB search](simple_frb/README.md): one beam, one broadband burst,
  DM 500 pc cm^-3, intrinsic Gaussian sigma 1 ms, injected S/N 50. Start here
  for a step-by-step offline search and the equivalent four-terminal online run.
- [Eight-beam live search](../README.md#try-the-live-pipeline): a longer
  observation containing broadband and narrowband bursts at different DMs.
- [Classifier window notebook](../AIclassifier/README.md): load one saved FRB
  chunk, locate the Grouper event, and extract a DM-time window with a bowtie mask.
