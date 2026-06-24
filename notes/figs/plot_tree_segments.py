#!/usr/bin/env python3
"""Generate notes/figs/tree_segments.pdf.

Shows every tree of configs/dedispersion/chord_sb2_et.yml as a line segment in
the (DM, triggering-latency) plane. Each tree searches a DM range [dm_min,dm_max]
and, for a burst at DM, triggers at a latency = 2^-delta * (full-band dispersion
delay), so the tree traces a straight segment from (dm_min, lat_min) to
(dm_max, lat_max). Segments are colored by the early-trigger earliness delta.

The picture shows the design at a glance: each downsampling level (ids) is one
DM octave; within an octave the non-early tree (delta=0) sits on the diagonal
latency = full delay, and each early trigger delta halves the latency at fixed DM.

All numbers are derived from the config (tree_rank, num_downsampling_levels,
early_triggers, time_sample_ms, dm_per_unit_delay); nothing is hardcoded.

Run:  python3 notes/figs/plot_tree_segments.py
"""
import os
from collections import defaultdict

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from pirate_frb import DedispersionConfig

_HERE = os.path.dirname(os.path.abspath(__file__))
CONFIG = os.path.join(_HERE, "..", "..", "configs", "dedispersion", "chord_sb2_et.yml")
OUTPUT = os.path.join(_HERE, "tree_segments.pdf")

# One color per early-trigger earliness delta (shared with plot_early_triggers.py).
DELTA_COLORS = ["#1f4e8c", "#2e8b3d", "#c44e1f", "#8a4fbf"]


def build_trees(config):
    """Return list of per-tree dicts (ids, delta, dm_lo/hi, lat_lo/hi), and the
    DM octave boundaries. Mirrors the DedispersionPlan tree construction."""
    r = config.tree_rank
    L = config.num_downsampling_levels
    dt = config.time_sample_ms / 1000.0          # seconds per (un-downsampled) sample
    dpud = config.dm_per_unit_delay()

    deltas = defaultdict(set)
    for s in range(L):
        deltas[s].add(0)                          # the non-early tree
    for et in config.early_triggers:
        deltas[et.ds_level].add(et.delta_rank)

    trees = []
    for s in range(L):
        d_lo = 0 if s == 0 else 2 ** (r + s - 1)  # full-band delay range (samples)
        d_hi = 2 ** (r + s)
        for delta in sorted(deltas[s], reverse=True):
            f = 2.0 ** (-delta)                   # sub-band / full-band delay ratio
            trees.append(dict(
                ids=s, delta=delta,
                dm_lo=d_lo * dpud, dm_hi=d_hi * dpud,
                lat_lo=f * d_lo * dt, lat_hi=f * d_hi * dt))
    octaves = [2 ** (r + s) * dpud for s in range(L)]
    return trees, octaves


def main(output=OUTPUT):
    plt.rcParams.update({
        "font.family": "serif",
        "mathtext.fontset": "cm",
        "font.size": 9,
        "axes.labelsize": 10.5,
        "xtick.labelsize": 9,
        "ytick.labelsize": 9,
        "axes.linewidth": 0.7,
    })

    config = DedispersionConfig.from_yaml(CONFIG)
    trees, octaves = build_trees(config)

    fig, ax = plt.subplots(figsize=(6.5, 4.0))

    # Dotted black guides at the DM-octave (downsampling-level) boundaries.
    for x in octaves:
        ax.axvline(x, color="black", ls=":", lw=0.9, alpha=0.55, zorder=0)

    seen = set()
    for t in trees:
        d = t["delta"]
        c = DELTA_COLORS[d]
        label = rf"$\delta={d}$" if d not in seen else None
        seen.add(d)
        ax.plot([t["dm_lo"], t["dm_hi"]], [t["lat_lo"], t["lat_hi"]],
                color=c, lw=2.4, solid_capstyle="round", zorder=3, label=label)
        ax.plot([t["dm_lo"], t["dm_hi"]], [t["lat_lo"], t["lat_hi"]],
                "o", ms=3.5, color=c, zorder=4)

    ax.set_xlim(0, octaves[-1] * 1.02)
    ax.set_ylim(0, max(t["lat_hi"] for t in trees) * 1.05)
    ax.set_xlabel(r"DM  (pc cm$^{-3}$)")
    ax.set_ylabel("triggering latency (s)")

    # Label each DM octave with its downsampling level along the top.
    lo = 0.0
    for s, hi in enumerate(octaves):
        ax.text(0.5 * (lo + hi), ax.get_ylim()[1] * 0.965, f"ids = {s}",
                ha="center", va="top", fontsize=10, fontweight="bold", color="black")
        lo = hi

    handles, labels = ax.get_legend_handles_labels()
    order = sorted(range(len(labels)), key=lambda i: labels[i])
    ax.legend([handles[i] for i in order], [labels[i] for i in order],
              title="earliness", loc="upper left", bbox_to_anchor=(1.01, 1.0),
              frameon=False, fontsize=9)

    fig.savefig(output, bbox_inches="tight")
    print(f"wrote {output}")


if __name__ == "__main__":
    main()
