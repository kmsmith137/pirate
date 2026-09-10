# Phase 2: controlled CHORD results

Validated locally on 2026-09-10. **Both scientific agreement and the online
capture experiment passed.** The implementation is on
`feature/controlled-chord-experiment`, based on the preserved phase-1 commit
`65896a7b`. No push or deployment was performed.

The [reusable recipe and commands](controlled_chord_experiment.md) generate one
220.8301056-second, two-beam observation using all 28,160 CHORD channels from
300 to 1500 MHz. Its 216 raw frames were generated once and hashed, then processed
offline and replayed twice at 1× through the native FakeXEngine/FrbServer path.
The scientific algorithms and source module hashes were identical in all runs.

## Scientific recovery

All three expected detections, their source cells, tokens, grouping membership,
representatives, DM, TOA, S/N and nominal filter widths matched **exactly**
between offline and both online runs. Suppressing early capture decisions did
not remove early detections from the scientific catalog.

| Detection | Recovered DM | TOA error at 300 MHz | S/N | Filter width |
| --- | ---: | ---: | ---: | ---: |
| Low DM, tree 0 | 99.990535 | −0.197 ms | 48.7500 | 5.9904 ms |
| High DM, early tree 1 | 2000.021699 | +1.514 ms | 35.6875 | 11.9808 ms |
| High DM, full tree 2 | 1999.991181 | +0.163 ms | 49.6875 | 11.9808 ms |

Both bursts had requested full-band S/N 50. Injected widths were Gaussian sigma
2 and 4 ms; the recovered widths describe the selected filter. Recovery satisfied
the predefined sampling-based DM and timing tolerances. Producer geometry,
Dcores, token encoding, analytic-weight inputs, startup treatment and complete
beam/chunk coverage were also checked.

## Online capture

The measured raw-buffer retention reached 61.341696 seconds. The full-band
dispersion sweeps were 4.4253952 seconds for DM 100 and 88.507904 seconds for DM 2000.

| Capture policy | Low-DM support saved | High-DM support saved |
| --- | ---: | ---: |
| Accept early detections | 4/4 chunks | 44/44 chunks |
| Suppress early capture | 4/4 chunks | 29/44 chunks |

The early high-DM detection became available at replay time 161.870117 seconds,
**38.129883 seconds before** its 300 MHz arrival. At its first request, the
leading signal frame was still six chunks inside the retained range. All 49
saved files, including padding, matched the original packed samples, scales and
metadata and had successful write-completion notifications.

In the control run, the high-DM full-band detection became available at replay
time 204.809842 seconds. Its earliest required raw frame was already 15 chunks
before the live buffer boundary. Those leading chunks were absent from the
actual capture; the remaining 34 saved files across both bursts passed the same
content and completion checks. This establishes the benefit of early capture
without changing the search or adding a classifier.

Maximum completed sender delays were 7.570 ms and 11.473 ms. No assembled
backlog was observed at the per-chunk snapshots. All 108 science chunks per beam
were processed, both transport-tail chunks were excluded from scientific
coverage, and both runs completed processing, promised writes and shutdown.

## Verification and evidence

**128 focused checks passed** in the integration checkout, including shared
GPU processing, recycled live-buffer lifetime, startup/seams, decoder provenance,
offline maps/catalogs, controlled generation/replay, capture deadlines and
completion, failure cleanup, comparison rejection cases and package file lists.
The exact test file list and machine-readable experiment results are in
[the evidence snapshot](validation/controlled_chord_20260910.json).

Full raw inputs, maps, catalogs, capture ledgers, timing records and JSON/Markdown
reports are preserved at:

```text
/home/mtrudu/pirate-validation/chord-phase2-20260910
```

Runs were executed from an isolated staging checkout. Every recorded processing
module hash was then checked against the integration checkout before saving this
checkpoint. Large observation artifacts remain outside Git.

The offline run used GPU 0, the early-enabled run GPU 1, and the control GPU 0
on this host's A40s. Some runs and checks overlapped on separate GPUs. These
timing samples establish feasibility for this controlled two-beam case; they
are not isolated performance measurements or a production capacity estimate.
The classifier remains explicitly bypassed. Broader performance, RFI and
sensitivity campaigns belong to phase 3.
