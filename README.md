# pirate - Perimeter Institute RAdio Transient Engine

An experimental GPU-based fast transient search for CHORD, based on
[kmsmith137/pirate](https://github.com/kmsmith137/pirate).
This integration includes offline peak finding and candidate grouping, trigger
catalogs, and a four-terminal live pipeline for simulated observations.

The search includes network reception, ring buffers, file-writing RPCs, and GPU
dedispersion. **RFI flagging is not implemented.** The live example uses simulated
Gaussian noise and injected bursts; it is a diagnostic, not an on-sky validation.

## Build and check

Building requires Ubuntu Linux, a physical NVIDIA GPU, the system CUDA toolkit
and host compiler, and [`ksgpu >= 1.5.0`](https://github.com/kmsmith137/ksgpu).
Use the supplied conda environment and the full [installation instructions](notes/install.md)
to install the native dependencies, CuPy, and ksgpu first. From this checkout:

```bash
git submodule update --init --recursive
make -j 32
python -m pirate_frb test --live --grouper -n 1
```

The focused checks cover live recipe/command wiring and offline extraction,
grouping, and packaging. They include GPU tests. To run a broader smoke test:

```bash
python -m pirate_frb test -n 1
```

## Grouper implementation

See the [module guide](notes/grouper_modules.md) for the shared processing stages,
configuration types and import names.

## Examples

Start with the [one-beam FRB search](examples/simple_frb/README.md): generate a
DM 500, sigma 1 ms, S/N 50 broadband burst, save its offline maps and grouped
events, then run the same observation online. Both observation examples are under [examples/](examples/README.md).

For classifier development, open the [classifier notebook](AIclassifier/README.md).
It includes just the detected FRB's map chunk and catalog, and walks from the
event's beam/tree/pixel indices to a small window and bowtie mask.

## Try the live pipeline

Use the same environment and repository directory in four terminals. Start
these commands in order, one per terminal:

```bash
# Terminal 1
python -B -m pirate_frb live grouper examples/chord_8beams/observation.yml
# Terminal 2
python -B -m pirate_frb live dedisperser examples/chord_8beams/observation.yml
# Terminal 3
python -B -m pirate_frb live event_monitor examples/chord_8beams/observation.yml
# Terminal 4: wait for the dedisperser and event monitor to print "Listening".
python -B -m pirate_frb live observation examples/chord_8beams/observation.yml
```

The recipe generates eight beams with three injected bursts. The event monitor
prints detected beam IDs, DM, S/N, frequency bands, and arrival times. Processing
uses memory and does not save observation data or detection catalogs. Wait for
the grouper's `Processing complete` message, then stop the dedisperser first with
Ctrl-C, followed by any remaining grouper and event-monitor processes.

All four processes use loopback ports 19700-19703. To run another instance, pass
the same unused `--base-port` to all four commands. See the
[recipe](examples/chord_8beams/observation.yml) for duration, beam, burst, and
grouping settings; the CHORD configuration requires substantial host and GPU memory.

## Offline processing

The Grouper settings and parser are shared with online processing. The one-beam
generator exports its recipe settings as an offline `grouper.yml` snapshot.
See [shared Grouper settings](examples/README.md#shared-grouper-settings).

For an existing ASDF acquisition, run:

```bash
python -m pirate_frb run offline_grouper ACQDIR configs/grouper/example.yml
```

See the [configuration guide](configs/README.md) and
[CLI reference](docs/source/cli.md) for output options, grouping boundaries,
and timeout behavior.

## Documentation

The [upstream HTML docs](https://kmsmith137.github.io/pirate/) may describe a
different revision. Generate docs for this checkout with `make -j 32 docs`
and view them with `make docs-serve`, or browse the sources:

- [Installation](notes/install.md)
- [Introduction](notes/intro.md)
- [Quick start](notes/quick_start.md)
- [Developer notes](notes/developer.md)
- [Build system](notes/build.md)
- [Hardware](notes/hardware.md)
- [X->FRB network protocol (v2)](notes/network_protocol.md)
- [X->FRB metadata (v2)](configs/xengine_metadata.yml)
- [gRPC protocol definitions](grpc/)

Contact: Kendrick Smith <kmsmith@perimeterinstitute.ca>
