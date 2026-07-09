#!/usr/bin/env python3
"""Generate notes/figs/tree_segments.pdf.

Shows every tree of configs/dedispersion/chord_sb2_et.yml as a line segment in
the (DM, triggering-latency) plane. Each tree searches a DM range [dm_min,dm_max]
and, for a burst at DM, triggers at a latency = 2^-e * (full-band dispersion
delay), so the tree traces a straight segment from (dm_min, lat_min) to
(dm_max, lat_max). Segments are colored by the early-trigger level e.

The picture shows the design at a glance: each primary tree (p) is one
DM octave; within an octave the non-early tree (e=0) sits on the diagonal
latency = full delay, and each early-trigger level halves the latency at fixed DM.

All numbers are derived from the config (toplevel_tree_rank, primary_trees,
time_sample_ms, dm_per_unit_delay); nothing is hardcoded.

Run:  python3 notes/figs/plot_tree_segments.py
"""
import os

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from pirate_frb import DedispersionConfig

_HERE = os.path.dirname(os.path.abspath(__file__))
CONFIG = os.path.join(_HERE, "..", "..", "configs", "dedispersion", "chord_sb2_et.yml")
OUTPUT = os.path.join(_HERE, "tree_segments.pdf")

# One color per early-trigger level e (shared with plot_early_triggers.py).
ET_COLORS = ["#1f4e8c", "#2e8b3d", "#c44e1f", "#8a4fbf"]


def build_trees(config):
    """Return list of per-tree dicts (p, e, dm_lo/hi, lat_lo/hi), and the
    DM octave boundaries. Mirrors the DedispersionPlan tree construction."""
    r = config.toplevel_tree_rank
    dt = config.time_sample_ms / 1000.0          # seconds per (un-downsampled) sample
    dpud = config.dm_per_unit_delay()

    trees = []
    for p, pt in enumerate(config.primary_trees):
        d_lo = 0 if p == 0 else 2 ** (r + p - 1)  # full-band delay range (samples)
        d_hi = 2 ** (r + p)
        for e in range(pt.num_early_triggers, -1, -1):   # earliest trigger first
            f = 2.0 ** (-e)                       # sub-band / full-band delay ratio
            trees.append(dict(
                p=p, e=e,
                dm_lo=d_lo * dpud, dm_hi=d_hi * dpud,
                lat_lo=f * d_lo * dt, lat_hi=f * d_hi * dt))
    octaves = [2 ** (r + p) * dpud for p in range(config.num_primary_trees)]
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

    # Dotted black guides at the DM-octave (primary-tree) boundaries.
    for x in octaves:
        ax.axvline(x, color="black", ls=":", lw=0.9, alpha=0.55, zorder=0)

    seen = set()
    for t in trees:
        e = t["e"]
        c = ET_COLORS[e]
        label = rf"$e={e}$" if e not in seen else None
        seen.add(e)
        ax.plot([t["dm_lo"], t["dm_hi"]], [t["lat_lo"], t["lat_hi"]],
                color=c, lw=2.4, solid_capstyle="round", zorder=3, label=label)
        ax.plot([t["dm_lo"], t["dm_hi"]], [t["lat_lo"], t["lat_hi"]],
                "o", ms=3.5, color=c, zorder=4)

    ax.set_xlim(0, octaves[-1] * 1.02)
    ax.set_ylim(0, max(t["lat_hi"] for t in trees) * 1.05)
    ax.set_xlabel(r"DM  (pc cm$^{-3}$)")
    ax.set_ylabel("triggering latency (s)")

    # Label each DM octave with its primary tree index along the top.
    lo = 0.0
    for p, hi in enumerate(octaves):
        ax.text(0.5 * (lo + hi), ax.get_ylim()[1] * 0.965, f"p = {p}",
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
