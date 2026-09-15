# Examples

Run examples from the repository root in the environment used to build PIRATE,
with its native dependencies, NVIDIA GPU and compiled CHORD kernels available.

| Observation example | Recipe | Description |
| --- | --- | --- |
| [One beam](simple_frb/README.md) | [observation.yml](simple_frb/observation.yml) | One broadband FRB, DM 500, Gaussian sigma 1 ms, injected S/N 50. Start here for offline and online instructions. |
| [Eight beams](chord_8beams/README.md) | [observation.yml](chord_8beams/observation.yml) | Live search with three broadband/narrowband bursts and subband/early-trigger trees. |

The [classifier window notebook](../AIclassifier/README.md) uses a small saved
result from the one-beam example to locate an event and extract its DM-time
window and bowtie mask.

## Shared Grouper settings

Offline and online use the same `GrouperConfig` schema and processing rules.
In the one-beam example, edit the `grouper:` section in `observation.yml`.
The offline generator exports that mapping to `grouper.yml` beside the generated
data; this is a snapshot, not a second configuration to maintain. Live commands
read the observation recipe directly. If you change the recipe after generation,
regenerate the offline data or use the saved `observation.yml` online.

`execution.beam_batch_size` controls offline beam batches. Online batches follow
the producer's `dedispersion_overrides.beams_per_batch`. The peakfinding, halo,
DM/time tolerances and timeout settings have the same meaning in both paths.

For an existing acquisition without an example recipe, start from the commented
[shared configuration template](../configs/grouper/example.yml).
