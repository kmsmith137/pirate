# Offline maps with PIRATE 1.5

New saved S/N maps and trigger catalogs use format_version: 3. The array
layout remains one (DM, time) map per tree per beam/chunk, with a matching
uint32 argmax map.

Maps carry the exact config_yaml and plan_yaml, plus required fields:

    argmax_encoding: pirate-1.5:t8-p8-m8-mu8
    dcores: [2, 4, 4]  # illustrative; one actual producer value per tree

The four token bytes, from low to high, encode fine time, temporal profile,
frequency multiplet, and extra DM. Dcore determines the legal fine-time grid.
It comes from the actual OfflineDedisperser.dd.Dcores after initialization,
or from the grouper handshake for an online consumer. A reconstructed plan or
the consumer's installed kernels cannot substitute for this provenance.

The loader reconstructs geometry with DedispersionConfig.from_yaml_string
and DedispersionPlan.from_yaml_string. It checks the encoding, per-tree
Dcore bounds, shapes, dtypes, source coordinates, and consistency across
files. The runner passes Dcores to peakfinding and decoding and retains them
in catalog metadata.producer.

Use the nested CLI:

    python -P -m pirate_frb run offline_dedisperser ACQDIR CONFIG.yml --save
    python -P -m pirate_frb run offline_grouper ACQDIR GROUPER.yml --output events.asdf

The dedisperser writes map files beside the raw frames. Use a fresh experiment
directory for 1.5 output. Version-2 maps must be processed with the preserved
1.4 environment, or regenerated from raw frames with 1.5. Relabelling a version,
editing old plan YAML, or guessing Dcores does not convert the old tokens.
Historical catalogs remain readable with their matching older reader.

Producer-start provenance remains separate: if a v3 file lacks it, the loader
keeps startup unknown. --assume-steady-state is an explicit processing policy;
it does not supply or bypass missing decoder metadata.

The portable regression tests author valid 1.5 tokens on the toy configuration,
exercise the real ASDF writer/loader and GPU grouping pipeline, compare catalog
physics with C++, and reject missing, corrupt, legacy, or mixed provenance.
They do not replace a raw-frame-to-catalog toy replay.

This branch also provides native CPU batch decoding for offline benchmarks:

    integers = plan.decode_argmax_batch(tokens, itrees, idms, itimes, dcores=dcores)
    physical = plan.decode_argmax2_batch(itrees, *integers)

All inputs are contiguous one-dimensional NumPy arrays. Tokens use uint32;
coordinates and the required per-tree dcores use int64. Empty candidate batches
must be handled by the caller. The batch loop invokes the C++ scalar decoder,
preserving its validation and rounding. This is an extension on this development
branch; the upstream online grouper obtains its Dcores from the handshake.

The clean batch peakfinder benchmark now uses result schema version 2 and
records the explicit per-tree Dcores and token encoding. Its producer setup
constructs the GPU kernels used to define NEW synthetic inputs; it does not
infer metadata for old saved acquisitions. The ten CHORD map shapes remain
unchanged. Input generation, producer setup and validation remain outside the
timed peakfinding interval. Historical result metadata should stay unchanged.

The Gaussian-corruption timing benchmark and its single-campaign and DM-reach
analysis notebooks now use result schema version 2. New campaigns record Dcores,
the token encoding and each tree's token geometry in both metadata and the
campaign signature. The default result directories include pirate15.
The synthetic S/N generation and timed decoder/grouper boundaries are unchanged;
constant tokens have separate 8-bit multiplet and extra-DM fields (extra DM is
zero for this benchmark). Existing campaigns must not be relabelled or resumed
under the new schema.

The CPU, representative-GPU and concentrated-map benchmark variants also use
schema version 2 and default to new pirate15 result directories. Their producer
metadata is explicit; Gaussian campaign signatures include it. The kernel
comparison uses schema version 3 with the same producer metadata. Grouping
algorithms, synthetic population rules and timed regions remain unchanged.
The CPU notebook's GPU overlay is optional when no path was explicitly selected;
an explicitly selected incompatible or missing campaign still fails clearly.


Toy experiment helpers and notebooks
------------------------------------

The retained recall, separability and runtime campaigns use result schema 5 and
default to peakfinder_tests/results_final_pirate15. Their scientific parameters
record producer Dcores and token encoding. prepare_plan returns four values:
config, xengine metadata, consumer plan, Dcores. Real simulated acquisitions
decode with the actual OfflineDedisperser.dd.Dcores after initialization.

The active inspect_peakfinders_on_fast_snrmap.ipynb notebook generates an analytic
map and derives its chunk index from its actual cadence. It includes the new
extra-DM token state, a production footprint view, decoded candidates, DM-reach
sweep, local timing and optional NPZ export. Its approximate response is not a
raw-stream sensitivity model.

inspect_peakfinders_on_snrmap.ipynb uses the shared v3 ASDF validator and recorded
producer metadata. Set ASDF_PATH in its parameter cell or PIRATE_SNR_MAP in the
kernel environment. Single-map boundary exclusions and missing streaming context
mean its candidate table is not the final grouper event catalog.

analyze_peakfinder_tests.ipynb validates all three completed schema-5 campaigns
from one directory, including producer metadata and recorded grid coverage.
Set PEAKFINDER_RESULTS_DIR to that directory. Small grids are supported; large
performance campaigns belong to the later performance phase.

Earlier multi-method notebook derivations are preserved in
peakfinder_tests/historical. The two older root-level exploratory notebooks are
marked historical and retain their original cells and outputs. Their scientific
claims and APIs were not revalidated on 1.5; use the matching preserved checkout.
