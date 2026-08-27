# ch_test helper scripts

Helpers for the `/ch-test` full test sweep (rebuild, ksgpu + pirate unit
tests, toy + production quickstart searches, offline dedispersion). The
command itself lives in the ch_top repo, at `.claude/commands/ch-test.md`;
these scripts are the parts of it that are pure mechanics.

The intended user is an AGENT running `/ch-test`, not a human -- though
they are all runnable by hand, and `check-logs.py` / `check-truth.py` are
useful on their own whenever you have pipeline logs to look at.

## Why these exist

A sweep takes ~90 minutes, of which most is waiting. The agent's time went
instead into two things that recur identically on every run:

- **Harness plumbing.** Background processes do not survive across an
  agent's tool calls, so the pipeline needs a supervisor; and bash starts
  async (`&`) jobs with SIGINT set to `SIG_IGN`, which CPython inherits, so
  `kill -INT` on a backgrounded pirate process is silently a no-op and the
  shutdown cascade never fires. Rediscovering that costs a wasted pipeline
  run. `launch-pipeline.sh` encodes both.
- **Deterministic analysis.** Correlating injected FRBs against
  dedisperser output, checking counter monotonicity, scanning for errors --
  all described in prose in the command file, all reimplemented from
  scratch each run.

What is deliberately NOT here: judgment. Whether a failure is a bug in the
code or in the test procedure, whether an SNR is plausible, when to stop and
ask -- that stays in `ch-test.md` and with the agent. These scripts print
numbers; the agent decides what they mean. Do not wrap them in a single
`run-everything.sh`: a sweep that nobody looks at is a rubber stamp.

## The scripts

| Script | What it does |
|--------|--------------|
| `launch-pipeline.sh` | Launch a pipeline downstream-first, gating each start on the previous process's readiness marker; handles the SIGINT and background-survival problems above. |
| `check-cascade.sh` | SIGINT one process, verify the shutdown cascades to all of them, report exit AND cascade-message timings (the latter is what the cascade should be judged by) and confirm hugepages/GPU came back. |
| `check-logs.py` | The "looks reasonable" checks: per-chunk progress, grouper baseline/spikes, both sifter event streams, ring-buffer counter monotonicity, error localization. |
| `check-truth.py` | Every injected FRB on the streamed beam should produce a spike in the offline dedisperser output. Handles dispersion sweep windows and acquisition-warmup allowances. |
| `preflight-prod.py` | Before a production run: mountpoints, ssd/nfs writability, hugepage budget, GPU idleness, rpc address resolution, proxy exemption, MTU. |
| `make-loopback-config.py` | Copy the production server config and rewrite `data_ip_addrs` to loopback with unique ports (the node's 10.x NICs are invisible inside the sandbox). |
| `inventory.sh` | The acquisition inventory table for the final report, with rows classified as this-sweep vs pre-existing. |

Every script prints `--help`, and returns nonzero on failure.

## Usage

Run everything from the `pirate/` directory (config paths are relative), and
keep all output in an untracked location -- the session scratch dir, `/tmp`,
or `plans/`.

Toy search:

```sh
S=/tmp/sweep                       # scratch; anything untracked
mkdir -p $S/logs && touch $S/sweep-start

cat > $S/toy.plan <<'EOF'
sifter     | waiting for grouper(s) to connect | 1 | 30 | pirate_frb run_toy_sifter 127.0.0.1:7500
grouper    | waiting for FrbServer to connect  | 1 | 30 | pirate_frb run_toy_grouper -s 127.0.0.1:7500 127.0.0.1:7000
server     | server(s) started                 | 1 | 90 | pirate_frb run_server configs/frb_server/toy.yml configs/dedispersion/toy.yml
xengine    | FakeXEngine(s) running            | 1 | 30 | pirate_frb run_fake_xengine -f -g 30 -s 127.0.0.1:7500 127.0.0.1:6000
rpc_status | Running get_status                | 1 | 30 | pirate_frb rpc status 127.0.0.1:6000
EOF

misc/ch_test/launch-pipeline.sh $S/toy.plan $S/logs &      # blocks; run in background
# poll $S/logs/supervisor.log for "ALL READY" (or "STARTUP FAILED")

# ... stream, rand_write, cancel (see ch-test.md) ...

misc/ch_test/check-cascade.sh $S/logs sifter
misc/ch_test/check-logs.py --logdir $S/logs --cascade --acqdir toy_stream_...
pirate_frb run_offline_dedisperser ~/pirate_toy/toy_stream_... configs/dedispersion/toy.yml > $S/dedisp.log
misc/ch_test/check-truth.py --xengine-log $S/logs/xengine.log --dedisp-log $S/dedisp.log \
                            --beam 10 --freq-lo 400 --freq-hi 800
```

Production, additionally:

```sh
misc/ch_test/preflight-prod.py configs/frb_server/cf05_production.yml   # STOP if this fails
misc/ch_test/make-loopback-config.py configs/frb_server/cf05_production.yml $S/cf05_loopback.yml
# ... plan file uses $S/cf05_loopback.yml, two grouper addresses (count=2 for
#     the grouper marker), a ~300 s server timeout, and 10.222.3.5 rpc addrs ...
misc/ch_test/check-truth.py ... --beam 100 --freq-lo 300 --freq-hi 1500 --warmup 8
```

At the end:

```sh
misc/ch_test/inventory.sh --since $S/sweep-start ~/pirate_toy /mnt/cs00/data/$USER
```

The `--freq-lo` / `--freq-hi` arguments are the first and last entries of
`zone_freq_edges` in the dedispersion config used for the OFFLINE pass (note
that for a production acquisition that config differs from the one the server
ran with -- see quick_start.md).

## Staleness: the contract these scripts must keep

`ch-test.md` deliberately tells the agent NOT to trust memorized command
lines, and to re-derive commands, ports and readiness markers from
`notes/quick_start.md` and the source, because they change. Scripts cut
against that: they hardcode log formats, and a format change makes them go
stale silently.

Two rules keep that honest, and both are load-bearing:

1. **Command lines stay outside the scripts.** `launch-pipeline.sh` takes a
   plan file supplying the commands and readiness markers. Nothing in this
   directory hardcodes a pirate command line, port, or config path.

2. **No vacuous passes.** Every parser fails loudly when it parses nothing.
   `check-truth.py` with zero injections parsed, or zero dedisperser chunks,
   is an ERROR with a message naming the likely cause -- never a cheerful
   `0/0 matched, PASS`. Same for `check-logs.py`: a log that exists but
   yields no parseable lines is a FAIL, not a skip. If you add a check, add
   its guard too.

**If you change a log or print format in pirate, update the parser here in
the same commit.** That is why these live in the pirate repo rather than
next to the command file: the formats and the code that parse them version
together. The formats currently parsed:

| Producer | Format |
|----------|--------|
| `FrbServer.cpp` | `FrbServer: beamset=B, ichunk=N, fpga=[a:b], ...` |
| `run_toy_grouper.py` | `toy_grouper: ... coarse_snr_max=X, nevents=N` |
| `run_toy_sifter.py` | `toy_sifter [FROM_SIMULATOR ]beamset=B ...` + indented event lines |
| `run_fake_xengine.py` | `injected FRB: beam_id=B, dm=D, fpga_timestamp=T, ..., snr=S`, and the startup `time_samples_per_chunk` / `seq_per_frb_time_sample` lines |
| `run_rpc_status.py` | `[addr] connections=N, rb=[...]` and `[addr] FILE received` |
| `run_offline_dedisperser.py` | `beam B chunk N (tci=T): snr_max=S` and the `nt_in=..., time_sample_ms=...` header |
