# Getting Started

## Installation

### Prerequisites

Install cuda, cupy, curand, pybind11, yaml-cpp. On a CHIME/CHORD machine:

```bash
conda create -c conda-forge -n ENVNAME \
     cupy scipy matplotlib pybind11 yaml-cpp argcomplete
```

**Note**: We recommend the `miniforge` fork of conda, not the original conda.

### Install ksgpu

See instructions at https://github.com/kmsmith137/ksgpu (use `chord` branch).

**Warning**: If you're building `pirate`, then you need the `chord` branch of `ksgpu`, not the `main` branch.

### Install pirate

```bash
git clone https://github.com/kmsmith137/pirate
cd pirate
make -j 32

# Run tests
python -m pirate_frb test -n 1

# Optional: editable pip install
pip install pipmake
pip install --no-build-isolation -v -e .
```

### Future Rebuilds

For editable installs, you can rebuild with just `make`:

```bash
git pull
make -j 32   # no pip install needed, if existing install is editable
```

## Quick Start

### Running Tests

```bash
# Run all unit tests
pirate_frb test

# Run specific test category
pirate_frb test --dd    # GpuDedisperser tests
pirate_frb test --casm  # CASM beamformer tests
```

### Hardware Information

```bash
# Show hardware info
pirate_frb show_hardware
```

### Working with Dedisperser Configs

```bash
# View a dedisperser config
pirate_frb show_dedisperser configs/dedispersion/chime.yml

# View with verbose comments
pirate_frb show_dedisperser -v configs/dedispersion/chime.yml

# Test a config
pirate_frb show_dedisperser --test configs/dedispersion/chime.yml

# Time a config
pirate_frb show_dedisperser --time configs/dedispersion/chime.yml
```

### Generating Random Configs

```bash
# Generate a random valid config
pirate_frb show_random_config

# Generate multiple configs
pirate_frb show_random_config -n 5
```

## Next Steps

- See {doc}`cli-reference` for complete command-line documentation
- See {doc}`api/index` for Python API reference
- Contact: Kendrick Smith <kmsmith@perimeterinstitute.ca>

