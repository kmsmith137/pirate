# PIRATE 1.5 migration: implementation and validation

Recorded 2026-09-09. The custom offline pipeline, decoder, benchmark entry points
and active toy notebooks have been adapted to the pinned PIRATE 1.5 release and
validated in an isolated worktree/environment. This record was assembled before
the local merge commit; the commit containing it identifies the integrated source.
The custom online grouping implementation and final performance study are later
phases. No push or deployment was performed during this migration.

## Source and environment

| Component | Revision or location |
| --- | --- |
| Original PIRATE base | `946d18434c075a0dfa0fc3191d11f12f51ec729a` |
| Grouper checkpoint, first merge parent | `22df18df146893f6aaf0643e5949ad5d59415e0d` |
| Upstream PIRATE 1.5, second merge parent | `255edd26c1c77e3f154fc5accd8014f073582b66` |
| ksgpu 1.5 | `5c81be2dea54226587f80eb206a35ca92c7160c5` |
| eschnett/asdf-cxx | `0b5a316398ce6a4fb2cf678e4125898ad17fdff6` (8.0.0) |
| Integration branch | `upgrade/pirate-1.5` |
| Integration worktree | `/home/mtrudu/src/pirate-1.5-integration` |
| ksgpu worktree | `/home/mtrudu/src/ksgpu-1.5-integration` |
| Validation environment | `/home/mtrudu/software/miniforge3/envs/pirate-15-validation` |
| Original checkout | `/home/mtrudu/src/pirate-updated`, preserved on the backup branch |
| Pre-migration backup | `/home/mtrudu/pirate-backups/pre-1.5-20260908` |

Both packages were built and installed as editable 1.5.0 packages in the validation
environment. Python 3.11.15, NumPy 2.4.6, CuPy 14.1.1 and ASDF 5.3.0 were used.
Selected dependency versions, the test-group record and saved validation reports
are retained in [the evidence JSON](validation/pirate_1_5_evidence.json).
That file is an evidence snapshot, not a complete portable environment lockfile.

The initial server test exposed mixed system and Conda gRPC libraries. The
validation environment now uses its matching gRPC 1.51.1, libprotobuf 3.21.12 and
Abseil 20230125.0 stack. Obsolete `pirate_grpc_abi` activation/deactivation hooks
were disabled and preserved under the environment's
`etc/conda/disabled-pirate-grpc-abi-20260908` directory. After reactivation,
`LD_PRELOAD` was empty and all 44 inspected gRPC/Abseil/Protobuf libraries loaded
from this environment. `pip check` found no broken requirements.

## Implemented changes

- The custom CLI uses `run offline_grouper` and retains `run offline_dedisperser
  --save`, multipulse acquisition generation, and `test --ofg` under the new CLI.
- Custom configs and packaged fixtures use the 1.5 primary-tree schema. The two
  obsolete downsampling keys were removed from config entries; generated plan
  fields with those names remain valid.
- Geometry, GPU decoding, synthetic token generation and validation use
  `t8-p8-m8-mu8`, including the extra-DM byte. Dcores come explicitly from the
  producer. Native CPU plan batch bindings accept the full Dcores vector, while
  online grouper bindings obtain it from the handshake.
- The C++ decoder retains the negative-time scaling fix and established FMA
  timestamp rounding order. Multipulse injection remains available and tested.
- Saved S/N maps and trigger catalogs use version 3 and retain producer metadata.
  Old maps are preserved for the old environment or regeneration from raw data;
  changing a version label cannot convert their tokens.
- Protobuf stubs were regenerated, including the sifter event `tree_index` field.
  Server/grouper protocol 3 was exercised locally. The output tree index is
  distinct from the primary-tree family index.
- Clean/Gaussian/CPU/GPU/concentrated benchmarks use result schema 2; the kernel
  comparison uses schema 3; retained toy campaigns use schema 5. New defaults
  have `pirate15` result-directory names and explicit producer provenance.
- Active analytic-map, saved-map and campaign-analysis notebooks use the retained
  full-band method. Earlier comparisons were preserved in
  `peakfinder_tests/historical`; root exploratory notebooks are marked historical.

See [offline formats and notebook use](offline_maps_1_5.md) for the data contracts.
The [initial impact assessment](pirate_1_5_upgrade_assessment.md) is retained as
the historical pre-migration analysis, not the current implementation status.

## Validation evidence

The user ran the checks in the integration checkout after applying each patch.
The groups below can overlap; their counts are not added into a claimed unique
test total. No broad performance campaign was required for this migration.

| Check | Observed result |
| --- | --- |
| Native build, protobuf generation, editable installation | Completed for both 1.5.0 packages |
| CLI decoder `--amax`, offline milestone `--ofg` | Exit 0, seed 137 |
| Noise/pulse/multipulse CLI/ASDF tests `--sim` | Passed |
| Local server/grouper test `--serv` | Handshake plus seven chunks passed; maximum GPU/reference differences 0.0146–0.0188 in float16 test |
| Native plan batch decoding | 3 passed, 6 deselected |
| Producer metadata and clean batch benchmark | 20 passed |
| Gaussian benchmark and DM-reach analysis | 17 passed |
| CPU/GPU benchmark variants and grouping | 59 passed |
| Toy helpers and small executable campaigns | 19 passed |
| Three active notebooks and invalid-input rejection | 8 passed |

The offline milestone covers the migrated extraction, decoding, grouping,
startup/seam, strict-config, saved-map and catalog regressions. The notebook
checks execute analytic generation/export, saved-map decoding against C++, and
small campaign analysis; they reject old schemas, missing Dcores and incompatible
grid metadata. The local server test validates the upstream transport/event path,
not a new custom online grouping implementation.

### Raw acquisition to event catalog

Artifacts are in
`/home/mtrudu/pirate-validation/pirate-1.5-toy-20260908`.
The observation has one beam (100), 640 channels over 400–800 MHz, eight chunks of
2048 samples, and a metadata-derived cadence of 0.9984 ms. Two strong full-band
bursts were injected at DM 100 with TOAs 6 and 10 seconds.

| Burst | Injected S/N | Recovered S/N | TOA error | DM error | Nominal filter width |
| --- | ---: | ---: | ---: | ---: | ---: |
| 0 | 30 | 30.207567 | +0.621470 ms | +0.031078 | 5.9904 ms |
| 1 | 40 | 38.835812 | +0.211870 ms | +0.031078 | 11.9808 ms |

Raw input hashes, eight saved maps, two events and two member records were checked.
Events matched their source cells and the C++ decoder; all eight grouping windows
completed with authoritative startup provenance. Producer Dcores were
`[8, 8, 8, 4, 8, 8]`, and both event tokens used nonzero extra DM.
Injected widths were Gaussian sigma (2 and 4 ms), whereas recovered widths describe
the chosen filter. Their equality is not a recovery requirement.
This fixed case establishes recovery within one input sample and one native DM
bin; it does not establish general completeness or production throughput.

### Small benchmark campaigns

- `pirate-1.5-batch-smoke-20260908`: two beams, batch sizes 1 and 2, one DM reach
  and one timed iteration per configuration. Two timing rows, two summaries and
  twenty tree diagnostics passed schema, producer, signature, hash and formula
  validation; both configurations had zero candidates.
- `pirate-1.5-gaussian-smoke-20260908`: two beams, one trial, DM reach 8, corruption
  fractions 0%, 0.001% and 0.01%. Counts were respectively 0/20/196 pixels above
  threshold, 0/8/22 candidates, and 0/8/22 decoded/grouped events. Saved results,
  producer metadata, source hashes, campaign signature and timing/statistic
  formulas passed validation without rerunning the campaign.

Both campaigns recorded ten Dcores of 8 and the 1.5 token encoding. Their small
sample sizes validate compatibility and bookkeeping; no capacity claim is made.
The evidence JSON includes the reports and artifact hashes. Large ASDF inputs and
historical benchmark outputs remain outside Git in their experiment directories.

## Using this checkpoint

Activate the validation environment before using the integration build. In this
worktree arrangement, `python -P` avoids importing the preserved checkout merely
because it is the shell's current directory. To run a repository-only benchmark
module, also set `PYTHONPATH` to the integration checkout; `peakfinder_tests` is
not part of the installed runtime package.

Example commands, with experiment paths supplied by the caller:

```bash
python -P -m pirate_frb run offline_dedisperser ACQDIR DEDISPERSION.yml --save
python -P -m pirate_frb run offline_grouper ACQDIR GROUPER.yml --output events.asdf
PYTHONPATH=/home/mtrudu/src/pirate-1.5-integration python -P -m peakfinder_tests.test_peakfinder_recall --help
```

Use a fresh acquisition/result directory for new 1.5 experiments. Select the
validation environment as the notebook kernel. The ASDF inspection notebook takes
`ASDF_PATH` or `PIRATE_SNR_MAP`; the toy campaign analysis takes
`PEAKFINDER_RESULTS_DIR`. An explicitly selected old or inconsistent input fails.

## Remaining project phases

Phase 2 implements the custom online pipeline without requiring a classifier,
while retaining offline and online toy experiments around shared processing and
event contracts. Phase 3 measures end-to-end performance, documents reproducible
experiments, and prepares the code and report for sharing. This migration does
not claim that either later phase, a deployment, or publication has been completed.
