# Run the controlled online pipeline in four terminals

The `experiment session` commands launch the working grouper, FrbServer,
classifier-free capture receiver and saved-frame replay separately. They use the
same scientific functions, finite observation and capture policy as
[`experiment online`](controlled_chord_experiment.md). This launcher currently
supports one finite controlled observation on one Linux host; it is not an
indefinite live-observation service or a multi-host CUDA IPC transport.

Each terminal supervises one native worker and prints its progress. The shared
session directory contains addresses, configuration identity and process state.
It does not contain injection truth used for detection. The detector and capture
receiver continue to use only the input metadata and detected quantities.

## Environment in every terminal

Open four terminals on the same host. Use the same Python environment, checkout
and `CUDA_VISIBLE_DEVICES` setting in each. The example assumes no device mask.

```bash
source /home/mtrudu/software/miniforge3/etc/profile.d/conda.sh
conda activate pirate-15-validation
export PYTHONPATH=/home/mtrudu/src/pirate-1.5-integration
export LD_LIBRARY_PATH="$CONDA_PREFIX/lib${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"
cd /home/mtrudu/src/pirate-1.5-integration
```

## Prepare once

Choose a fresh output directory. This example reuses the previously generated
and hashed CHORD observation; it does not generate new noise or bursts.

```bash
python -P -m pirate_frb experiment session prepare \
  /home/mtrudu/pirate-validation/chord-phase2-20260910/observation \
  /home/mtrudu/pirate-validation/chord-four-terminals-example \
  --gpu 0
```

Preparation verifies the raw inputs and prints the four launch commands. By
default it chooses four currently available loopback ports and saves them in
`session.json`. They are not reserved until the components bind. Use
`--base-port 5100` to choose data/RPC/grouper/capture ports 5100/5101/5102/5103
explicitly. Use distinct ports for concurrent sessions; preparation checks availability,
but another process can claim a port before a component starts.

All four commands must receive exactly the same output path. Shell variables
set in one terminal do not automatically exist in another; the examples use
absolute paths to avoid that mistake.

## Terminal 1: working grouper

```bash
python -P -m pirate_frb experiment session grouper /home/mtrudu/pirate-validation/chord-four-terminals-example
```

The grouper listens and waits for the producer handshake. Once data arrive, it
uses `SharedGrouper`, reports processed chunks and nonempty grouping windows,
and writes `events.asdf`. This is the working scientific grouper, not
`run toy_grouper`.

## Terminal 2: server and GPU dedisperser

```bash
python -P -m pirate_frb experiment session server /home/mtrudu/pirate-validation/chord-four-terminals-example
```

The server connects to the grouper, starts its raw-data receiver and exposes
the search/capture RPC endpoint. GPU initialization completes after the replay
provides acquisition metadata. The server preserves the same separately sized
raw and dedispersion pools as the combined launcher.

## Terminal 3: capture receiver

```bash
python -P -m pirate_frb experiment session capture /home/mtrudu/pirate-validation/chord-four-terminals-example
```

It waits for the server, registers file-completion notifications and listens for
detected events. It prints beam, tree, DM, S/N and capture/association decisions.
The classifier is explicitly bypassed. Capture requests use detected quantities;
the injection manifest is read only during validation.

## Terminal 4: controlled FakeXEngine replay

```bash
python -P -m pirate_frb experiment session replay /home/mtrudu/pirate-validation/chord-four-terminals-example
```

Replay waits for the server, a listening grouper and the capture receiver, then
verifies input hashes and sends the exact saved packed frames at 1×. It must not
wait for the grouper's completed metadata handshake: that handshake needs the
sender to start. The default observation takes approximately 3 minutes 41 seconds
plus setup, transport tails and drain time.

Components may be launched in another order; their waiting messages identify
the missing dependencies. The default startup deadline is ten minutes per
terminal. Increase `--startup-timeout-seconds` when preparing the session if
manual setup will take longer. Once replay starts, finite completion has its own
deadline. `--drain-timeout-seconds` and `--max-lag-seconds` are also preparation
options, shared by every component.

## Completion, status and stopping

All four commands exit automatically after a successful finite observation:

1. Replay dispatches every science frame and the two receiver-flush chunks.
2. The grouper finishes all science windows and holds its CUDA IPC mapping.
3. The capture receiver finishes every promised write and closes its subscription.
4. The server stops the producer, lets the grouper exit, and publishes complete
   `run.json` only after the component supervisors acknowledge clean completion.

To inspect status from an available terminal:

```bash
python -P -m pirate_frb experiment session status /home/mtrudu/pirate-validation/chord-four-terminals-example
```

To interrupt the entire session:

```bash
python -P -m pirate_frb experiment session stop /home/mtrudu/pirate-validation/chord-four-terminals-example
```

Ctrl-C in any component also fails the session and causes the other components
to shut down. A terminal supervises and signals only its own worker. A parent
death guard prevents an orphaned native worker if its supervisor is killed.
Interrupted runs remain incomplete; a disconnect is never treated as normal EOF.
Completed sessions are not altered by `session stop`.

Do not relaunch a component in an already used session directory. An exclusive
claim prevents duplicate consumers or replaying twice into the same producer.
After failure or completion, prepare a fresh session. Preserve failed outputs
for diagnosis. Do not edit processing source or bundle metadata between session
preparation and running the commands; startup rejects mismatched source,
environment or session identity.

## Compare early capture with the full-band control

Prepare another fresh session with `--suppress-early-capture`, then run its same
four commands. Early detections still remain in the scientific catalog; only
their capture decisions are suppressed.

Session outputs have the same catalog, capture ledger, timing and `run.json`
layout as the combined launcher. Use the existing comparison command:

```bash
python -P -m pirate_frb experiment compare BUNDLE OFFLINE_OUTPUT EARLY_SESSION FULL_SESSION validation.json
```

The offline reference must be produced with the same processing source version.
Old reports and raw inputs remain valid historical artifacts, but their source
fingerprints intentionally prevent attributing them to a later implementation.
Reuse the saved raw observation and rerun offline when the implementation has
changed. A completed file-write ledger alone does not establish complete burst
coverage; the comparison checks actual saved bytes, notifications and raw expiry.


## Validation recorded on 2026-09-10

The full 220.830-second CHORD observation was run offline and through four
independent terminal commands, with early capture enabled and suppressed.
All four supervisors and native workers exited successfully in both online runs.
The comparator verified their session identities and completion acknowledgments.

Both online catalogs exactly matched the offline measurements, source cells,
grouping partitions and candidate memberships: three detections from two bursts.
Early capture saved all 44 required high-DM signal chunks; the full-band-only
control saved 29/44, with 15 leading chunks already expired. Both saved all four
low-DM signal chunks. The capture validator checked actual saved packed data and
metadata against the original observation.

The 158 focused regression checks passed. A native short replay launched in
reverse component order was explicitly stopped after its first dispatched chunk:
all four supervisors failed the session and all their native workers exited.
The original combined `experiment online` command also completed a six-chunk
native replay after these changes. No native GPU processes remained after checks.

The two full online runs overlapped on separate GPUs; their timings are not a
throughput benchmark. Machine-readable evidence is in
[`controlled_terminals_20260910.json`](validation/controlled_terminals_20260910.json).
Full artifacts and logs are retained at
`/home/mtrudu/pirate-validation/chord-terminals-20260910`.
