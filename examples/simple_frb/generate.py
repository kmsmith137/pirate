"""Save the shared one-beam observation for offline dedispersion and grouping."""
import argparse
from pathlib import Path

import ksgpu
import yaml

from pirate_frb.LivePipeline import load_live_recipe
from pirate_frb.Observation import make_pulse
from pirate_frb.core import AssembledFrame, XEngineMetadata
from pirate_frb.utils import atomic_print


def generate(output, recipe_path):
    """Create a fresh directory containing frames and their resolved configs."""
    recipe, bundle = load_live_recipe(recipe_path)
    if len(bundle["beam_ids"]) != 1:
        raise ValueError("this introductory example requires exactly one beam")
    beam_id = bundle["beam_ids"][0]
    metadata = XEngineMetadata.from_yaml_string(bundle["metadata_yaml"])
    pulses = [make_pulse(burst, metadata) for burst in recipe["bursts"]]
    ntime, nchunks = bundle["samples_per_chunk"], bundle["nchunks"]
    for pulse in pulses:
        if pulse.it_start < 0 or pulse.it_end > ntime * nchunks:
            raise ValueError("a burst is clipped: extend duration or move its arrival time")

    output = Path(output).resolve()
    output.mkdir(parents=True, exist_ok=False)
    frames = output / "frames"
    frames.mkdir()
    (output / "metadata.yml").write_text(bundle["metadata_yaml"])
    (output / "dedispersion.yml").write_text(bundle["dedispersion_config_yaml"])
    (output / "grouper.yml").write_text(bundle["grouper_config_yaml"])
    recipe = dict(recipe, metadata="metadata.yml", dedispersion="dedispersion.yml")
    (output / "observation.yml").write_text(yaml.safe_dump(recipe, sort_keys=False))

    # Match the live observation's seed, pulse construction, and chunk order.
    ksgpu.seed_default_rng(recipe["observation"]["noise_seed"])
    for chunk in range(nchunks):
        frame = AssembledFrame.make_uninitialized(
            metadata, ntime=ntime, beam_id=beam_id, time_chunk_index=chunk)
        frame.randomize_many(normalize=True, gaussian=True, pulses=pulses,
                             dt_sp=chunk * ntime)
        path = frames / f"frame_b{beam_id}_t{chunk}.asdf"
        frame.write_asdf(str(path))
        atomic_print(f"Wrote chunk {chunk + 1}/{nchunks}: {path}")
    duration = nchunks * ntime * bundle["time_sample_ms"] * 1e-3
    atomic_print(f"Generated beam {beam_id}: {nchunks} chunks, {duration:.6f} seconds.")
    atomic_print(f"Resolved observation and search configs: {output}")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("output", type=Path, help="new output directory (must not exist)")
    parser.add_argument("--recipe", type=Path,
                        default=Path(__file__).with_name("observation.yml"),
                        help="shared offline/online observation YAML")
    args = parser.parse_args()
    generate(args.output, args.recipe)


if __name__ == "__main__":
    main()
