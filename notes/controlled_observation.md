# Controlled CHORD observation bundle

`pirate_frb.ControlledObservation` prepares a finite acquisition containing two
independently specified broadband bursts. The bundled example preserves CHORD's
300–1500 MHz frequency zones and all 28,160 channels. Both pulses are in beam 100;
beam 101 contains noise. Two beams with one active single-beam batch satisfy the
live server's beam-capacity constraint.

Prepare the example in a new, empty location, then generate the saved frames once:

```bash
python -P -m pirate_frb.ControlledObservation prepare configs/experiments/chord_replay.yml /path/to/new-bundle
python -P -m pirate_frb.ControlledObservation generate /path/to/new-bundle
python -P -m pirate_frb.ControlledObservation verify /path/to/new-bundle
```

Use the validation environment and the intended source checkout. Preparation
requires no GPU and writes no acquisition frames. It computes a CPU dedispersion
plan, validates startup and capture-budget constraints, and records storage
estimates. Generation is CPU work and writes approximately 6.42 GB plus ASDF
headers for this example. Offline maps and triggered copies require more space.
The archive is the common input to offline processing and network replay.

## Timing and pulse definitions

The example requests 220 seconds, rounded upward to 108 chunks of 2,048 samples:
220.8301056 seconds at the metadata-derived cadence of 0.9984 ms. Burst TOAs are
referenced to 300 MHz. The low-DM burst has DM 100 and TOA 90 seconds; the high-DM
burst has DM 2,000 and TOA 200 seconds. Gaussian intrinsic widths are 2 and 4 ms
(sigma), with requested full-band S/N 50 for both. Recovered filter widths are a
different quantity. These dispersed signals occupy disjoint raw-time intervals.

The actual plan places DM 2,000 in output trees 1 (early trigger at approximately
416.025 MHz) and 2 (full band). Their expected reference arrivals are 155.746048
and 200 seconds. The high-DM sweep takes 88.507904 seconds across the complete
band. Thirty raw-buffer chunks provide 61.341696 seconds of *nominal* retention.
The planner allows one chunk for completion, two for receiver assembly lookahead,
one for shared-grouper lookahead, and four seconds of additional latency. Together with one second of requested
pre-padding, the high-DM test has approximately four seconds of planned margin.
These are design budgets, not measurements of usable retention or online speed.

Preparation obtains conservative steady-state bounds from
`plan.compute_steady_state_it0()` for every tree. Each intended detection must
arrive after its target tree's all-DM-bin readiness bound plus a chunk guard. The
bounds for trees 0 and 1 are approximately 65.56 seconds; tree 2 needs
approximately 130.99 seconds. Other, higher-DM trees can remain startup-incomplete
at these times. The runtime preserves their ordinary startup flags; preparation
does not remove them or alter peak competition. Arrival times or duration that
violate these constraints are rejected before creating the bundle.

## Bundle contract

`bundle.json` is the structural acquisition manifest. It records canonical beam
IDs, initial chunk, chunk count, samples per chunk, cadence, duration, configuration
filenames and hashes. Once complete, `frame_entries` covers every chunk and beam
exactly once, ordered by chunk and then canonical beam order. Each entry includes
its relative path, byte count and SHA256.

The public input-adapter API is:

```python
from pirate_frb.ControlledObservation import load_experiment_bundle
bundle = load_experiment_bundle(bundle_dir, verify_hashes=True)
```

It returns the manifest plus `bundle_dir`, `manifest_path`, `manifest_sha256`,
`metadata_path`, `dedispersion_config_path`, `grouper_config_path`,
`capture_config_path`, and the canonical `metadata` mapping. Runtime input
adapters must use this structural information only. `injections.json` is a
separate validation truth file; this loader never opens it. `experiment.yml`
preserves the generation recipe, and `plan_check.json` records readiness and
storage/capture estimates. Dcores are not guessed by the preparation step: actual
online and offline producers supply them through their existing contracts.

`metadata.yml` contains every canonical beam and frequency channel. Individual
ASDF frames carry one beam's data; PIRATE's ASDF reader projects the per-frame
metadata to that beam. Network replay reconstructs the canonical multi-beam frame
set without generating new noise or changing time labels.

## Reproducibility and failure handling

Generation runs in a fresh child process and seeds ksgpu before the generating
thread first initializes its AVX2 noise state. It records the seed, software
versions and generator source hash. There is no promise that a seed gives identical
bytes across different native builds or software versions: the immutable saved
frames and their hashes establish the replay input.

A new bundle is created exclusively. Generation claims it once, writes each raw
frame to a temporary file, then publishes that file without replacement. The
manifest is marked complete only after all frames have been written and hashed.
Interrupted or failed generation leaves its files and a failed/incomplete state;
runtime loading rejects it. There is deliberately no overwrite or resume mode.
Preserve the failed directory for diagnosis and prepare a new bundle to retry.
