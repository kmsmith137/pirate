# Command-Line Reference

The `pirate_frb` command provides several subcommands for testing, timing, and running the dedispersion pipeline.

## Main Command

```{argparse}
---
module: pirate_frb.__main__
func: get_parser
prog: pirate_frb
---
```

## Available Subcommands

The command includes 11 subcommands:

- **test** - Run unit tests (by default, all tests are run)
- **time** - Run performance benchmarks (by default, all timings are run)
- **show_hardware** - Show hardware information
- **show_kernels** - Show registered CUDA kernels
- **make_subbands** - Create subband_counts with specified frequency range and width
- **show_dedisperser** - Parse a dedisperser config file and write YAML to stdout
- **show_random_config** - Generate random DedispersionConfig(s) and print as YAML
- **test_node** - Run test server
- **send** - Send data to test server
- **scratch** - Run scratch code (defined in src_lib/scratch.cu)
- **random_kernels** - A utility for maintaining makefile_helper.py

Each subcommand has its own set of options and arguments. Use `pirate_frb <subcommand> --help` for detailed information about a specific subcommand.

