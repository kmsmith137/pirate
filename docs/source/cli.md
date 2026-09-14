# CLI reference

Many `pirate` features are accessed via the command-line interface:
```
pirate_frb SUBCOMMAND [ARGS...]
pirate_frb GROUP SUBCOMMAND [ARGS...]
```
where the list of subcommands, and documentation for each subcommand, are given below.
Most subcommands are nested in a group (`run`, `rpc`, `show`, `varmap`, `dev`, `live`), whose row
in the table below links to a page listing that group's subcommands; `test`, `time` and
`time_dedisperser` are typed directly. Each subcommand's page embeds its `--help` output,
captured from the argparse parser when the docs are built. Note that
`python -m pirate_frb ...` is equivalent to `pirate_frb ...`.

```{include} _cli_generated.md
```

## Live detection

The four commands are `live grouper`, `live dedisperser`,
`live event_monitor`, and `live observation`. Each takes the same observation
YAML. They process generated intensity data in memory and print detections,
without session preparation or saved output files.

Start the grouper, dedisperser and event monitor in that order, each in its own
terminal. Wait for the dedisperser and event monitor to print `Listening`, then
start the observation in a fourth terminal. After processing completes, stop the
dedisperser first with Ctrl-C, then the grouper and event monitor.

From the repository root, use the following commands, one per terminal:

```bash
python -B -m pirate_frb live grouper configs/experiments/chord_8beams.yml
python -B -m pirate_frb live dedisperser configs/experiments/chord_8beams.yml
python -B -m pirate_frb live event_monitor configs/experiments/chord_8beams.yml
python -B -m pirate_frb live observation configs/experiments/chord_8beams.yml
```

The default loopback ports are 19700-19703. Supply the same `--base-port`
to every command to use a different range. The dedisperser accepts `--gpu`
(default 0); use the same `CUDA_VISIBLE_DEVICES` setting in all terminals to
select the GPU used by both dedispersion and grouping.
The observation sends simulated Gaussian noise and the bursts specified in the
recipe. The event monitor prints detections, including beam, DM, S/N, sub-band,
and arrival time relative to the observation start. The observation's
`Sending complete` message precedes the grouper's `Processing complete`
message; wait for the latter before stopping the pipeline.

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
