# Simple FRB search: offline and online

This example searches one beam for a single simulated FRB. Both paths use
[observation.yml](observation.yml), the same noise seed, and the same native
pulse generator. Start with the offline steps to see the saved data products,
then run the observation through the online pipeline.

The `grouper:` section of `observation.yml` supplies both paths. The generator
exports it as an offline `grouper.yml` snapshot automatically. See
[shared Grouper settings](../README.md#shared-grouper-settings).

| Parameter | Value |
| --- | --- |
| Beam ID | 1 |
| DM | 500 pc cm^-3 |
| Intrinsic Gaussian sigma | 1 ms |
| Injected S/N | 50 |
| Emission band | 300-1500 MHz (fully broadband) |
| Spectral index | 0 |
| Scattering | None |
| Arrival time | 40 s at 300 MHz, from observation start |
| Observation | 64 s requested, rounded to 32 chunks / 65.4311424 s |
| Sampling | 0.9984 ms, 2048 samples per chunk |
| Noise seed | 137 |

The search uses one primary tree (tree index 0), a full-band search, and no
early triggers. Its DM range is approximately 0-1479 pc cm^-3. The full CHORD
frequency resolution is retained so the narrow burst is not artificially
smeared by reducing the number of channels.

## Before starting

Follow the [installation instructions](../../notes/install.md), including the
native dependencies and ksgpu. Run every command below from the repository root
in that environment. Rebuild this checkout with `make -j 32` so the native
grouper supports one-beam output rings.

Allow roughly 1 GiB of disk for the offline observation and maps, plus working
host/GPU memory. The online example keeps data in memory. To select a GPU,
set the same `CUDA_VISIBLE_DEVICES` in every terminal before starting Python.

## Offline: generate, dedisperse, group

### 1. Generate the observation

Choose a new output directory; the generator refuses to reuse an existing one.

```bash
python -m examples.simple_frb.generate /tmp/pirate-simple-frb
```

This writes the simulated intensity frames to `frames/frame_b1_t*.asdf` and
saves resolved copies of the observation, metadata, dedispersion, and grouper
configs beside that directory. These copies preserve the settings used for this
run. The online observation uses the same pulse construction, seed, and chunk
order; reproducibility assumes the same software and environment.

### 2. Generate S/N and argmax maps

```bash
python -m pirate_frb run offline_dedisperser \
    /tmp/pirate-simple-frb/frames \
    /tmp/pirate-simple-frb/dedispersion.yml --save
```

Each intensity frame gets a neighboring `frame_b1_tN_snrmap.asdf`. It contains
one tree's S/N map, its matching argmax-token map, and the producer's coordinate
metadata. The S/N map has shape `(4096, 128)`, with DM rows first and time columns
second. Argmax tokens retain the finer position and boxcar-width information
selected by the producer within each output pixel.

### 3. Find peaks and group them into events

```bash
python -m pirate_frb run offline_grouper \
    /tmp/pirate-simple-frb/frames \
    /tmp/pirate-simple-frb/grouper.yml \
    --output /tmp/pirate-simple-frb/events.asdf
```

The command prints representative detections and writes their catalog. Look for
beam 1, tree 0, DM near 500 and arrival time near 40 seconds. The recovered S/N
need not be exactly 50: the injection is Gaussian, the search uses discrete
boxcar widths and DM/time grids, and the data include noise and quantization.
The recovered boxcar width is not the injected Gaussian sigma.

For seed 137 in the validation environment, the offline search recovered one
event with DM 499.997795, arrival time 40.0000664 s, S/N 46.0625, and a
2.9952 ms boxcar width. Its source pixel was beam 1, tree 0, chunk 19,
`idm=1385`, `itime=72`. Treat these as an example result, not a requirement for
bitwise agreement across software versions and hardware.

Add `--verbose` to inspect grouped members and their map coordinates. The catalog
retains the representative's `source_chunk_index`, `idm`, and `itime` alongside
its beam and tree. These identify the source map pixel for a later classifier
window; use the representative's source chunk, which can differ from the
chunk owning the grouping window. Neither `idm` nor `itime` is a physical DM or
an arrival time in seconds.

Continue with the [classifier notebook](../../AIclassifier/README.md) to locate
the representative pixel and extract a window using PIRATE's GPU functions.
Its small bundled sample uses the same environment and requires no regeneration
of the observation.

## Online: stream the same observation

Use four terminals with the same environment and repository directory. Start
the following commands in order, one per terminal:

```bash
# Terminal 1: grouper
python -B -m pirate_frb live grouper examples/simple_frb/observation.yml

# Terminal 2: dedisperser
python -B -m pirate_frb live dedisperser examples/simple_frb/observation.yml

# Terminal 3: event monitor
python -B -m pirate_frb live event_monitor examples/simple_frb/observation.yml

# Terminal 4: observation; wait for terminals 2 and 3 to print Listening.
python -B -m pirate_frb live observation examples/simple_frb/observation.yml
```

These are real network reception, GPU dedispersion, and grouping processes,
using loopback on a single host. The observation streams at the observing rate
and sends two extra transport chunks to finish assembling the science data.
The event monitor should report the same burst near DM 500 and 40 seconds.
Only the offline path saves maps and a catalog; the live monitor prints the
physical event fields, not the complete source-pixel metadata.

Wait for the grouper's `Processing complete` message. Stop the dedisperser first
with Ctrl-C, then the remaining grouper and event-monitor processes. To run
another observation, restart all four processes. Ports 19700-19703 must be free;
use the same `--base-port` in all four commands to choose another range.

If you changed the source recipe after generating the offline data, use
`/tmp/pirate-simple-frb/observation.yml` in all four online commands to replay
those saved settings. This regenerates the observation from its seed; it does
not read the offline frame files.

## Change one parameter

Edit the burst in `observation.yml`, then generate a fresh offline directory or
restart the online processes. Keep the dispersed pulse inside the observation
and its DM inside the search range. The arrival time is referenced to 300 MHz;
higher-frequency emission arrives earlier.
