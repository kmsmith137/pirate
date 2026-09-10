# CLI reference

Many `pirate` features are accessed via the command-line interface:
```
pirate_frb SUBCOMMAND [ARGS...]
pirate_frb GROUP SUBCOMMAND [ARGS...]
```
where the list of subcommands, and documentation for each subcommand, are given below.
Most subcommands are nested in a group (`run`, `rpc`, `show`, `varmap`, `dev`, `experiment`), whose row
in the table below links to a page listing that group's subcommands; `test`, `time` and
`time_dedisperser` are typed directly. Each subcommand's page embeds its `--help` output,
captured from the argparse parser when the docs are built. Note that
`python -m pirate_frb ...` is equivalent to `pirate_frb ...`.

```{include} _cli_generated.md
```

## Offline grouper configuration

`run offline_grouper` takes an acquisition directory and a strict YAML file:

```
pirate_frb run offline_grouper ACQDIR CONFIG.yml
```

Scientific and execution settings come only from the YAML file. The CLI keeps
operational overrides for `--device`, `--output`, `--max-chunks`, `--verbose`,
and `--assume-steady-state`. See the commented
[`configs/offline_grouper/example.yml`](configs/offline_grouper/example.yml)
for a complete configuration.

The top-level mapping must contain exactly these required sections and fields:

```yaml
peakfinding:
  snr_threshold: 10.0
  dm_reach: 8
  waist_bins: 1
grouping:
  halo_size: 2
  dm_tolerance: 1.5
  time_tolerance: 1.5
execution:
  beam_batch_size: 1
  timeout_ms: 400
  timeout_policy: discard
```

`snr_threshold` must be finite. DM reach and waist are nonnegative integers.
Grouping tolerances are finite nonnegative factors: `dm_tolerance` scales the
coarser candidate/tree DM step and `time_tolerance` scales the coarser
candidate/tree time step in each compatibility comparison. `halo_size` is an
integer of at least two and defines that many temporal radii of seam context per
tree. Following CHIME's real-time rule, a physical full-band Bowtie wider
than a tree's native map chunk is cropped symmetrically in time to the largest
odd width that fits that chunk. This deliberately omits competition outside
the real-time horizon (and can empty an outer DM row); it is not represented as
an unmodified physical footprint. Catalogs record both requested and effective
per-tree time radii. Catalogs also record the effective grouping-halo columns.
Chunk `i` therefore never waits for `i+2`.
Beam batch size is positive; timeout is a nonnegative integer, with zero
disabling it; and timeout policy is exactly `discard` or `emit_partial`.
Boolean values are not accepted as numbers. Missing keys, unknown keys,
duplicate keys, unsafe YAML tags, and values outside these domains fail before
GPU work starts. The shown `400` ms timeout is an example, not a built-in
default; the value in the supplied file is authoritative.

### Streaming ownership and timeout behavior

Grouping is bounded to adjacent chunks. A window contains candidates owned by
chunk `i` plus the per-tree left halo from chunk `i+1`. The configured map halo
is `halo_size * bowtie_time_radius` native time columns. Its effective candidate
prefix is the intersection with centres already peak-resolved after `i+1`, or
`min(halo_size * radius, ntime - radius)`. A group is owned by
chunk `i` if any member came from `i`, even when its louder representative came
from `i+1`. Its consumed halo members are removed before the next window. This
gives one-chunk latency, prevents duplicate seam events, and ensures chunk
`i+2` cannot alter an event already finalized for `i`. The map tail always has
the configured size (bounded by data seen so far). Cross-window candidate
context is the intersection of that tail with candidate centres already
resolved after one following chunk. DM/time compatibility is applied only
inside this map-coordinate association domain and never expands the lookahead.
For a wide effective footprint, even the
default `2*h` retained halo can extend beyond the `ntime-h` current-chunk
centres already resolved at that point (and larger multipliers extend it
further). Centres that would require `i+2` are owned by the next window and
never feed back into finalized output. Consequently, detections outside this
declared domain can remain separate even when their decoded coordinates would
satisfy the compatibility predicate; that is the explicit bounded-latency
tradeoff.

`timeout_ms` applies separately to the persistent GPU clustering launch for
each owner window and compatible beam batch. It does not cover file I/O,
peakfinding, decoding, kernel compilation/layout, result compaction, terminal
printing, or catalog writing. GPU blocks share one deadline and check it at
bounded work tiles. A cooperative timeout can overshoot by one predicate tile
and one indivisible event-commit pass, but a half-constructed event is never
published.

- `discard` emits no event from a timed-out window and drops every candidate
  which entered it.
- `emit_partial` emits only fully committed owner groups and drops unprocessed
  candidates.

Processing then continues with the next window without accumulating a backlog.
Terminal summaries identify every timeout. Version-3 trigger catalogs attach
`grouping_window_id` and `grouping_timed_out` to event/member rows and store a
record for every attempted window, including empty and discarded windows.
Timeout provenance is deliberately separate from peakfinder `edge_flags`.


## Controlled CHORD online/offline experiment

The `experiment` group prepares one prescribed CHORD observation and runs its
identical saved frames through offline and live inputs to the same scientific
processor:

```text
pirate_frb experiment prepare CONFIG.yml BUNDLE
pirate_frb experiment generate BUNDLE
pirate_frb experiment offline BUNDLE OUTPUT --gpu 0
pirate_frb experiment online BUNDLE OUTPUT --gpu 0
pirate_frb experiment online BUNDLE CONTROL_OUTPUT --gpu 0 --suppress-early-capture
pirate_frb experiment compare BUNDLE OFFLINE ONLINE_EARLY ONLINE_FULL REPORT.json --markdown REPORT.md
```

Observation and run directories are created exclusively. Online replay runs at
the observing rate; `--max-lag-seconds` sets its sender abort limit (default one
second). The control suppresses early **capture decisions**, preserving the
scientific search and catalog. Classifier bypass is explicit in the configuration
and reports. Scientific agreement and timely raw capture are validated separately.

See the [controlled CHORD experiment guide](../../notes/controlled_chord_experiment.md)
for the configuration, exact environment and run commands, timing budget, output
contracts and limitations. This two-beam correctness experiment is not a
production-throughput benchmark.


## Controlled experiment in separate terminals

The working scientific grouper, native server, capture receiver and controlled
replay can also be launched independently. Prepare one fresh session directory:

```bash
pirate_frb experiment session prepare BUNDLE OUTPUT --gpu 0
```

Run each of the following in its own terminal on the same host and environment:

```bash
pirate_frb experiment session grouper OUTPUT
pirate_frb experiment session server OUTPUT
pirate_frb experiment session capture OUTPUT
pirate_frb experiment session replay OUTPUT
```

The session coordinates readiness and finite completion. The grouper uses the
same scientific core as offline, and the capture receiver bypasses the classifier.
Use `experiment session status OUTPUT` to inspect state or
`experiment session stop OUTPUT` to request shutdown. Do not reuse a completed
or failed session for another acquisition. See
[the complete terminal tutorial](../../notes/controlled_terminals.md).
