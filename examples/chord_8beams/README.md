# Eight-beam live FRB search

For the smallest offline/online walkthrough, start with the
[one-beam example](../simple_frb/README.md). This observation contains eight beams
and three injected bursts: a low-DM broadband burst, a high-DM broadband burst,
and a burst confined to 500–1000 MHz.

The [recipe](observation.yml) defines the observation and Grouper settings.
It uses the full CHORD 300–1500 MHz subband/early-trigger search, two beams per
native batch, and 220 seconds of input rounded up to complete chunks.

Follow the [installation instructions](../../notes/install.md). Run from the
repository root, with the same environment and GPU selection in four terminals.
Start these commands in order, one per terminal:

```bash
# Terminal 1: Grouper
python -B -m pirate_frb live grouper examples/chord_8beams/observation.yml
# Terminal 2: dedisperser
python -B -m pirate_frb live dedisperser examples/chord_8beams/observation.yml
# Terminal 3: event monitor
python -B -m pirate_frb live event_monitor examples/chord_8beams/observation.yml
# Terminal 4: wait for terminals 2 and 3 to print Listening.
python -B -m pirate_frb live observation examples/chord_8beams/observation.yml
```

The event monitor is a test receiver and prints detections. Input data, maps and
catalogs are not saved.

With the supplied recipe and seed, the validated A40 run produced four detections
for the three injected bursts:

| Beam | Tree | DM (pc cm^-3) | Arrival time (s, at 300 MHz) | Search band (MHz) |
| --- | --- | --- | --- | --- |
| 5 | 0 | 550.154 | 50.0050 | 474.342–750 |
| 1 | 0 | 99.991 | 89.9998 | 300–1500 |
| 8 | 1 | 2000.022 | 200.0015 | 416.025–1500 |
| 8 | 2 | 1999.991 | 200.0002 | 300–1500 |

The two beam-8 detections correspond to the same high-DM burst, reported by an
early-trigger tree and a full-band tree. The count is therefore a count of
reported detections, not distinct injected bursts. Search bands describe the
selected detection; they need not equal the injected emission band. Treat these
values as example results rather than requiring exact numerical agreement across
hardware and software environments.

The full search requires substantial host and GPU memory. Wait for the Grouper's
`Processing complete` message, then stop the dedisperser first with Ctrl-C,
followed by the remaining Grouper and event monitor. Restart all four processes
for another observation. Ports 19700–19703 must be free; use the same `--base-port`
in all commands to select another range.

See [shared Grouper settings](../README.md#shared-grouper-settings) for how the
same configuration is used offline and online.
