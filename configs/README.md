# configs/

Each subdirectory contains YAML config files used by the `pirate_frb` CLI
(`python -m pirate_frb <command> ...`). All configs are passed as positional arguments.

- **`dedispersion/`** -- GPU dedispersion pipeline configs (frequency zones, tree rank,
  downsampling levels, peak-finding parameters, batching, etc.). Used by two subcommands:

  - `pirate_frb show_dedisperser [-v] configs/dedispersion/chime.yml`
    -- parse and pretty-print the config and plan.
  - `pirate_frb time_dedisperser [-n NITER] configs/dedispersion/chime.yml`
    -- run GPU timing benchmarks.

  `toy.yml` is a subscale config for quick testing; the remaining files are
  production-scale configs for CHIME and CHORD (with subband/early-trigger variants).

  Production configs are also used by the build system (`makefile_helper.py`) to
  determine which CUDA kernels to autogenerate.

- **`frb_server/`** -- FRB search server configs (network addresses, memory allocation,
  file-writing threads, SSD/NFS paths, fake-X-engine parameters). Used by:

  ```
  pirate_frb run_server configs/frb_server/toy.yml        # start server
  pirate_frb run_server -s configs/frb_server/toy.yml     # send fake X-engine data
  ```

  `toy.yml` runs a single server on loopback for local testing;
  `cf05_production.yml` is a multi-server production config.

- **`hwtest/`** -- Hardware stress-test configs that drive parallel synthetic loads
  (network I/O, SSD writes, GPU dedispersion, PCIe and memory bandwidth). Used by:

  ```
  pirate_frb hwtest [-t SECONDS] configs/hwtest/cf00_all.yml    # receive side
  pirate_frb hwtest -s configs/hwtest/cf00_all.yml               # send side
  ```

  Each file enables a different subset of loads (e.g. `cf00_net.yml` for
  networking only, `cf00_all.yml` for everything, `cf00_blob.yml` for storage).

- **`xengine/`** -- X-engine metadata file defining the frequency-zone layout, beam
  configuration, and initial time sample. This file serves double duty:

  1. It documents the binary metadata header that each X-engine node sends at the
     start of a TCP stream to an FRB search node.
  2. It is used as configuration for the fake X-engine data sender
     (`pirate_frb run_server -s`).

  To inspect:

  ```
  pirate_frb show_xengine_metadata [-v] configs/xengine/xengine_metadata_v1.yml
  ```
