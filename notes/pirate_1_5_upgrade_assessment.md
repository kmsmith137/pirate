PIRATE 1.5.0 impact assessment — 2026-09-08

The update affects our custom offline peak-finding, decoding, and benchmark infrastructure substantially. The grouping algorithms and existing measurements remain useful, but our current code cannot run unchanged against 1.5.0. Treat this as a migration with regression checks, rather than just a pull and rebuild.

This assessment compares our working files with upstream source fetched into temporary audit repositories. No pull, merge, submodule update, installation, or build was performed in either working checkout. Verification was source inspection and a file-by-file three-way merge rehearsal; 1.5.0 runtime compatibility has not been tested.

**Exact revisions inspected**

| Repository | Local state | Upstream target |
| --- | --- | --- |
| PIRATE | `mtrudu/grouper-development`, `946d18434c075a0dfa0fc3191d11f12f51ec729a`, version 1.4.0 | `main`, `255edd26c1c77e3f154fc5accd8014f073582b66`, version 1.5.0 |
| ksgpu | Clean `chord` branch tracking `origin/chord`, `dfe9bd9` | `main`, `5c81be2dea54226587f80eb206a35ca92c7160c5`, version 1.5.0 |

PIRATE upstream contains 287 commits beyond our base and changes 224 files. Our work includes 15 modified tracked files and 133 untracked files before adding this assessment. Much of the grouper implementation, tests, benchmarks, and results is untracked. A backup branch alone would not preserve these working changes.

The fetched ksgpu `chord` branch ends at `f031e20`, version 1.2.0. A plain `git pull` there would follow `chord`, whereas PIRATE 1.5.0 explicitly requires `ksgpu >= 1.5.0`. The inspected ksgpu `main` is a descendant of the local ksgpu commit, with no local-only commits to reconcile.

**Impact on our work**

| Area | Finding and required action |
| --- | --- |
| CPU/GPU grouping algorithms | The representative selection and grouping logic remains reusable. It consumes decoded coordinates and geometry, so correct migration of those inputs is essential. Recheck partitioning, group membership, and timing after the producer changes. |
| Argmax token decoding | **Required algorithm change.** Our decoder treats bits 16–31 as one multiplet field. Upstream now uses `t | (p << 8) | (m << 16) | (mu << 24)`, with independent 8-bit `m` and `mu`. The extra DM coordinate `mu` participates in the DM/time calculation. |
| Peak-finder token validation | Our validity mask also interprets the upper 16 bits as one multiplet. It can discard legitimate new tokens with nonzero `mu` before decoding, changing which candidates survive local suppression. Update validation together with decoding. |
| Producer `Dcore` | `tree.Dcore` is removed. It is now provided by `GpuDedisperser.Dcores`, `ReferenceDedisperser.Dcores`, or online `FrbGrouper.dcores`. Our geometry, decoder, synthetic-map utilities, and fixtures read the removed attribute. |
| Plan APIs | `gpu_runnable=`, `make_incomplete_plan_from_yaml()`, `tree.pf`, and `tree.total_rank()` are removed/replaced. The plan batch decoder bindings used by our benchmarks are also removed; online grouper batch methods remain available. |
| Saved S/N maps | Our writer saves config and plan YAML, and our loader reconstructs a plan through the removed API. New plan YAML no longer carries `Dcore`. We must save the producer's per-tree values explicitly and identify the token encoding. |
| Existing results | Existing CSVs, plots, catalogs, and measurements remain records of the original producer and implementation. Preserve their metadata. Re-running their generators on 1.5.0 requires migration; numerical or performance equivalence is not established by the email. |
| Configurations | Update our custom `configs/dedispersion/toy_off.yml` and packaged fixtures under `pirate_frb/tests/data/`. Upstream supplies updated tracked standard configs. Remove the two obsolete keys only from `primary_trees` config entries; those names still belong in plan output. |
| CLI | Port our `offline_grouper` command into the new `run` group, retain offline dedispersion `--save`, and retain `test --ofg`. Update CLI tests, notebook code cells, documentation, and shell/subprocess invocations. |
| Online grouper/sifter | Upgrade server and grouper together for grouper protocol 3. Our custom toy-grouper path uses `create_events()`, which upstream updates. The only direct `FrbSifterEvents()` construction sites found in active Python source are upstream-owned and updated upstream. |
| `tree_index` | When adding a custom sifter export, send the exact producer output-tree index as int32. Our `primary_tree_index` identifies a family containing multiple early-trigger trees and is not interchangeable with the event's `tree_index`. Regenerate sifter stubs to expose the new field. |
| ASDF and pulse injection | The upstream ASDF writer adopts asdf-cxx 8.0.0 and ASDF standard 1.6.0 for valid float16 arrays. Our multi-pulse injection edits merge textually with those changes. Preserve `randomize_many()` and test C++/Python ASDF interoperability. Standard-compliant files do not automatically make old serialized plans compatible with new APIs. |
| Packaging | Retain our added modules, package fixtures, source-distribution config example, `asdf`, and `PyYAML >= 6.0`; adopt upstream's `ksgpu >= 1.5.0` in both runtime and build requirements. |
| `--write-delay` | Optional server I/O test behavior; it defaults to zero. It does not require changes to our grouping logic. |

The custom toy and packaged fixture values inspected follow the email's equal-or-halved width/weight rules and nondecreasing early-trigger counts. They still contain the removed keys and must be loaded with the new implementation to check all validation rules.

**Concrete code migration**

1. Introduce an explicit producer descriptor containing config, plan, per-tree `Dcores`, and token-encoding version. Pass it through saved-map loading, peak-finder geometry, GPU decoding, and benchmark generation. For the offline GPU writer, the authoritative values are `od.dd.Dcores` after initialization. Never substitute values inferred from the consumer's current kernel build.
2. Implement the new token layout and validate all four fields. In the upstream decoder, `pow2_K = tree.dm_downsampling >> tree.frequency_subbands.pf_rank`, and the coarse delay starts at `idm_coarse * pow2_K + mu`, with the primary-tree offset added afterward. Follow the complete upstream decoder, including subband bounds, time granularity, and time-coordinate conventions.
3. Replace plan API calls according to the table below. Preserve vectorized GPU decoding; use updated C++ scalar decoders as a correctness oracle, or deliberately restore suitable batch bindings that take producer `Dcores`. Do not replace benchmark batch calls with Python loops inside timed production paths.
4. Preserve our C++ fixes for negative signed shifts and the explicit FMA/timestamp rounding order when integrating the new decoder. The inspected upstream still contains the old signed shifts and implicit floating-point expression.
5. Give newly written saved-map files an explicit new schema/encoding identity. The present local map schema is `format_version=2`; bumping it to 3 is a reasonable implementation choice, not an upstream requirement. Keep historical files intact. Initially retain the old environment for historical map processing; if mixed-version reading is needed, add an explicit legacy reader with its own plan/encoding semantics. Deleting YAML fields or relabeling old files is insufficient.
6. Carry producer `Dcores` and encoding identity into catalog/benchmark provenance when retaining raw tokens. Preserve old metadata YAML verbatim in historical result directories.

| Current local usage | 1.5.0 migration |
| --- | --- |
| `DedispersionPlan(config, gpu_runnable=True)` | `DedispersionPlan(config)` for a complete GPU plan; obtain decoder `Dcores` from the actual producer. |
| `DedispersionPlan(config, gpu_runnable=False)` | For geometry-only work, `DedispersionPlan(config, mega_ringbuf=False, gpu_kernels=False)`. Choose flags based on what the caller needs. |
| `make_incomplete_plan_from_yaml(config_yaml, plan_yaml)` | Parse `config = DedispersionConfig.from_yaml_string(config_yaml)`, then call `DedispersionPlan.from_yaml_string(config, plan_yaml)` for new-format data. This rebuilds geometry and cross-checks the YAML; it is not an old-plan converter. |
| `tree.Dcore` | Explicit producer `Dcores[itree]`. |
| `tree.pf.max_width` | `tree.primary_tree.max_width`. |
| `tree.total_rank()` | `tree.tree_rank`. |
| `plan.decode_argmax(token, itree, idm, itime)` | `plan.decode_argmax(token, itree, Dcore, idm, itime)`. |
| `plan.decode_argmax_batch()` / `decode_argmax2_batch()` | Migrate our callers or deliberately supply updated bindings; upstream retains these methods on `FrbGrouper`, not `DedispersionPlan`. |

Local implementation sites include [GpuArgmaxDecoder.py](../pirate_frb/GpuArgmaxDecoder.py), [Peakfinders.py](../pirate_frb/Peakfinders.py), [FrbOfflineGrouper.py](../pirate_frb/FrbOfflineGrouper.py), [run_offline_dedisperser.py](../pirate_frb/run_offline_dedisperser.py), [TriggerCatalog.py](../pirate_frb/TriggerCatalog.py), and benchmark utilities [experiment_common.py](../peakfinder_tests/experiment_common.py), [fast_snrmap.py](../peakfinder_tests/fast_snrmap.py), [peakfinders.py](../peakfinder_tests/peakfinders.py), and [benchmark_peakfinder_batch_timing.py](../peakfinder_tests/benchmark_peakfinder_batch_timing.py).

**Merge rehearsal**

Thirteen of our fifteen modified tracked files also changed upstream. File-by-file three-way merging against the shared base found the following conflicts:

| File | Conflict blocks |
| --- | ---: |
| `Makefile` | 2 |
| `pirate_frb/__main__.py` | 5 |
| `pirate_frb/run_offline_dedisperser.py` | 1 |
| `pirate_frb/tests/__init__.py` | 1 |
| `pyproject.toml` | 1 |
| `src_lib/DedispersionPlan.cpp` | 1 |

No untracked local filename collides with a tracked upstream filename. This is a textual rehearsal, not a complete Git merge or compilation. Clean textual merges do not establish API compatibility. In particular, keep upstream's new CLI dispatch (`parser.set_defaults(func=...)` and `args.func(args)`), and attach our `offline_grouper` parser under `parse_run()`.

**Proposed update sequence — commands below have not been executed**

Use separate worktrees for both projects and a separate validation environment. This preserves the original source trees and compiled libraries, including the old ksgpu library that our existing PIRATE build may load. The commands pin the exact reviewed release commits; fetching later upstream work should trigger a fresh comparison.

1. Archive working changes and untracked files, then checkpoint active development source. Run from the existing PIRATE checkout. Choose a durable backup location with available space.

```bash
cd /home/mtrudu/src/pirate-updated
upgrade_backup=/home/mtrudu/pirate-backups/pre-1.5-20260908
mkdir -p "$upgrade_backup"
git status --short > "$upgrade_backup/status.txt"
git rev-parse HEAD > "$upgrade_backup/base-commit.txt"
git diff --binary HEAD > "$upgrade_backup/tracked-work.patch"
git ls-files --others --exclude-standard -z > "$upgrade_backup/untracked-files.list"
tar --null -T "$upgrade_backup/untracked-files.list" \
    -czf "$upgrade_backup/untracked-files.tar.gz"

git switch -c backup/grouper-before-1.5-20260908
git add -u
git add pirate_frb/ configs/dedispersion/toy_off.yml configs/offline_grouper/
git add ':(glob)peakfinder_tests/*.py' ':(glob)peakfinder_tests/*.ipynb'
git add analyze_offline_snrmap.ipynb inspect_asdf_shapes.ipynb
git add notes/pirate_1_5_upgrade_assessment.md
git diff --cached --stat
git status --short
```

Review the staged list to ensure active source, fixtures, and notebooks are covered. Benchmark outputs and `old_stuff/` are preserved in the archive and original checkout; they need not be added to the integration history. Git-ignored build artifacts are not in the archive and are retained in the original checkout. Then create the private checkpoint:

```bash
git commit -m "Checkpoint grouper development before PIRATE 1.5 migration"
```

2. Prepare a ksgpu worktree on the reviewed **main** revision. This avoids pulling the stale `chord` line into the build.

```bash
git -C /home/mtrudu/src/ksgpu fetch origin main
git -C /home/mtrudu/src/ksgpu worktree add -b upgrade/ksgpu-1.5 \
    /home/mtrudu/src/ksgpu-1.5-integration \
    5c81be2dea54226587f80eb206a35ca92c7160c5
```

Activate a separate environment with the existing CUDA/build dependencies. If the current tested build environment is an active Conda environment, it can be cloned with `conda create --name pirate-15-validation --clone "$CONDA_PREFIX"`, then activated with `conda activate pirate-15-validation`. Cloned editable package references must be replaced with the integration worktrees below. Keep the current interpreter/CUDA dependency stack for the first comparison rather than upgrading it simultaneously.

```bash
cd /home/mtrudu/src/ksgpu-1.5-integration
make -j 32 PYTHON=python
python -m pip install --no-build-isolation --no-deps -e .
python -c 'import ksgpu; print(ksgpu.__file__)'
```

3. Create the PIRATE integration worktree from our checkpoint and merge the reviewed upstream revision.

```bash
cd /home/mtrudu/src/pirate-updated
git fetch origin main
git worktree add -b upgrade/pirate-1.5 \
    /home/mtrudu/src/pirate-1.5-integration HEAD

cd /home/mtrudu/src/pirate-1.5-integration
git -c submodule.recurse=false merge --no-commit --no-ff \
    255edd26c1c77e3f154fc5accd8014f073582b66
```

The merge is expected to stop for conflicts. Resolve the six files above, preserve our additions, and perform the semantic migration before continuing to validation. Do not resolve by selecting whole-file "ours" or "theirs" versions. The original checkout still contains the original results and compiled 1.4.0 build.

4. Once the merged `.gitmodules` points to `eschnett/asdf-cxx`, synchronize the new worktree's submodule and initialize the commit selected by PIRATE:

```bash
git submodule sync --recursive
git submodule update --init --recursive
git submodule status --recursive
```

Do not use `git submodule update --remote`: the intended dependency is the pinned commit in the PIRATE release.

5. Refresh packaged fixture copies after the tracked standard configs have merged. Remove the obsolete keys from the custom `toy_off.yml` separately, preserving its other settings. Our packaging test requires the packaged fixtures to match the standard source configs byte for byte.

```bash
cp configs/dedispersion/toy.yml pirate_frb/tests/data/toy.yml
cp configs/dedispersion/chord_sb2_et.yml pirate_frb/tests/data/chord_sb2_et.yml
```

6. After the code migration, build in the validation environment. A new worktree starts without stale compiled objects. `make grpc` explicitly regenerates local protobuf outputs; a separately deployed sifter must also regenerate its own stubs to read `tree_index`.

```bash
make grpc PYTHON=python
make -j 32 PYTHON=python
python -m pip install --no-build-isolation --no-deps -e .
python -c 'import pirate_frb, ksgpu; print(pirate_frb.__file__); print(ksgpu.__file__)'
python -m pip check

python -m pirate_frb run offline_grouper --help
python -m pirate_frb run offline_dedisperser --help
python -m pirate_frb show dedisperser configs/dedispersion/toy_off.yml
python -m pirate_frb show dedisperser configs/dedispersion/chord_sb2_et.yml
python -m pirate_frb show dedisperser pirate_frb/tests/data/toy.yml
python -m pirate_frb show dedisperser pirate_frb/tests/data/chord_sb2_et.yml

python -m pirate_frb test --amax --sim --ofg -n 1
python -m pirate_frb test --serv -n 1
python -m pytest -q pirate_frb/tests/test_packaging.py
python -m pytest -q peakfinder_tests
```

These are post-migration validation commands, not commands that can pass on the current local code against 1.5.0. Ensure the `--ofg` runner and its CLI tests have been ported. The server tests matter because upstream's plan scalar tests do not exercise the online batch decoder bindings. Inspect skips and external-fixture availability; a test that returns early without input data does not demonstrate successful migration of saved acquisitions.

7. Run a small end-to-end comparison before new full benchmark sweeps: raw frame → 1.5.0 dedispersion → new saved-map schema → peak finder → GPU decoder → CPU/GPU grouping → catalog. Write to a new output directory. Include nonzero `mu`, early-trigger trees, startup and seam cases, overlapping injected pulses, and invalid tokens. Compare decoded fields with the updated C++ oracle, then compare CPU/GPU membership and timeouts. For scientific before/after comparisons, use the same immutable raw input when possible and record producer commits, geometry, `Dcores`, encoding, GPU, and configuration.

After validation, inspect `git diff --check` and the staged changes, commit the completed migration on `upgrade/pirate-1.5`, and use that branch for review. Change online launch commands and restart server/grouper together only after the coordinated upgrade is ready. Historical timing summaries should remain identified as the old implementation's measurements; new end-to-end performance claims require a 1.5.0 baseline.

**Upstream source evidence**

- The [new decoder](https://github.com/kmsmith137/pirate/blob/255edd26c1c77e3f154fc5accd8014f073582b66/src_lib/DedispersionPlan.cpp#L889) parses four independent bytes and incorporates `mu` into the delay calculation.
- The [plan contract](https://github.com/kmsmith137/pirate/blob/255edd26c1c77e3f154fc5accd8014f073582b66/include/pirate/DedispersionPlan.hpp#L103) explains YAML cross-checking and producer-owned `Dcore`; the [Python bindings](https://github.com/kmsmith137/pirate/blob/255edd26c1c77e3f154fc5accd8014f073582b66/src_pybind11/pirate_pybind11.cpp#L381) define the replacement APIs.
- The [tree definition](https://github.com/kmsmith137/pirate/blob/255edd26c1c77e3f154fc5accd8014f073582b66/include/pirate/DedispersionTree.hpp#L21) exposes the new geometry and primary-tree fields.
- [PIRATE dependencies](https://github.com/kmsmith137/pirate/blob/255edd26c1c77e3f154fc5accd8014f073582b66/pyproject.toml), [ksgpu main](https://github.com/kmsmith137/ksgpu/blob/5c81be2dea54226587f80eb206a35ca92c7160c5/pyproject.toml), and [ksgpu chord](https://github.com/kmsmith137/ksgpu/blob/f031e205a1b0926e6a0b1cf3d866a590fb771d86/pyproject.toml) establish the branch/version issue.
- The [new ASDF writer](https://github.com/kmsmith137/pirate/blob/255edd26c1c77e3f154fc5accd8014f073582b66/src_lib/AssembledFrame.cpp#L503) selects the ASDF standard needed for float16 arrays.

Temporary audit artifacts, including conflict previews and `merge_report.json`, are under `/tmp/pirate-15-audit.Y9cYEF/`. They are supplementary and may be removed by normal temporary-directory cleanup; the findings and update sequence above are recorded in this document.
